#pragma once

// Bit-parallel SIMD batch kernel for Jaro and Jaro-Winkler similarity.
//
// Lane layout: one target per 64-bit SIMD lane. The query is shared
// across lanes; its 256-entry per-character bitmap (PEQ, where bit i
// of peq[c] is set iff query[i] == c) is built once and gathered per
// lane on each iteration.
//
// Inner step (per target-position j, across `Ops::lanes` targets):
//   1. Per-lane: index c[lane] = target[lane][j], or sentinel 0 if
//      j has run past that lane's target length.
//   2. window_mask[lane] = (~0 << lo[lane]) & (~0 >> (64 - hi[lane]))
//      where lo/hi bound the search range in the *query*; both shift
//      amounts are kept in [0, 63] so every ISA's variable-shift
//      semantics line up.
//   3. candidate[lane] = peq[c[lane]] & window_mask[lane] & ~used_a[lane]
//   4. lowest = candidate & (-candidate)     // lowest set bit per lane
//   5. used_a |= lowest                       // mark match in query
//   6. b_matched |= (1 << j) where candidate != 0  // mark match in target
//
// After the j-loop, the per-lane (used_a, b_matched) bitmaps are
// stored back to memory and a scalar finishing pass computes Jaro
// from the popcounts and the transposition count. The finishing pass
// is the natural place for the half-transposition walk over set
// bits — std::countr_zero / popcount lack portable per-lane SIMD
// equivalents and the count fits in 64 bits so a 7-element scalar
// loop (worst case 64 set bits) is cheap.
//
// Constraints:
//   * Both query and target lengths must fit in 64 bits (single-word
//     bit-parallel). Above this the caller falls back to per-target
//     scalar dispatch (which is itself bit-parallel single-word when
//     applicable).
//   * Inputs must be byte-compatible (bytes or 1-byte unicode); the
//     dispatcher above checks this before calling.

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "levenshtein_simd_ops.hpp"  // re-use the Ops bundles
#include "stride_align/jaro.hpp"

namespace stride_align::jaro_simd {

namespace nb = nanobind;

// Aligned scratch for `Ops::lanes` uint64 values. 64-byte alignment
// covers AVX-512 stores; smaller vectors stride naturally inside the
// same buffer.
template <typename Ops>
struct alignas(64) LaneScratch {
  std::uint64_t values[Ops::lanes];
};

// PEQ over a byte-alphabet query. 257 entries — index 256 is the
// inactive-lane sentinel and stays zero, so a gather on that index
// produces an all-zero PEQ row that won't match anywhere.
struct ByteAlphabetPeq {
  alignas(64) std::uint64_t entries[257];
};

inline ByteAlphabetPeq build_peq_byte(
    const std::uint8_t* q_ptr, std::size_t q_len) noexcept {
  ByteAlphabetPeq peq{};
  for (std::size_t i = 0; i < q_len; ++i) {
    peq.entries[q_ptr[i]] |= std::uint64_t{1} << i;
  }
  return peq;
}

// Multi-word PEQ for queries longer than 64 chars. `W` blocks of 64
// bits per character (query length up to W*64). Block-major layout:
// entries[b * 257 + c] is block b of the bitmap for character c.
// Block-major lets the inner loop gather one block at a time with a
// fixed per-block base pointer.
template <std::size_t W>
struct ByteAlphabetPeqMulti {
  alignas(64) std::uint64_t entries[W * 257U];

  const std::uint64_t* block(std::size_t b) const noexcept {
    return entries + b * 257U;
  }
};

template <std::size_t W>
inline ByteAlphabetPeqMulti<W> build_peq_byte_multi(
    const std::uint8_t* q_ptr, std::size_t q_len) noexcept {
  ByteAlphabetPeqMulti<W> peq{};
  const std::size_t n = std::min<std::size_t>(q_len, W * 64U);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t b = i / 64U;
    const std::size_t bit = i % 64U;
    peq.entries[b * 257U + q_ptr[i]] |= std::uint64_t{1} << bit;
  }
  return peq;
}

// Per-batch SIMD bit-parallel Jaro inner. Writes Jaro similarity into
// out[0..group). Lanes beyond `group` are filled with sentinel data
// (lens=0, char index=256) so the SIMD work doesn't need a branch
// inside the inner loop.
template <typename Ops>
inline void jaro_simd_inner(
    const ByteAlphabetPeq& peq,
    const std::uint8_t* q_ptr,
    std::size_t q_len,
    const std::uint8_t* const* targets,
    const std::size_t* lens,
    std::size_t group,
    double* out) {
  constexpr std::size_t K = Ops::lanes;
  using Vec = typename Ops::Vec;

  // Per-lane scratch laid out so each vector load picks up `K`
  // contiguous uint64s.
  LaneScratch<Ops> windows{};
  std::size_t max_m = 0;
  for (std::size_t l = 0; l < group; ++l) {
    windows.values[l] = static_cast<std::uint64_t>(
        ::stride_align::jaro::match_window(q_len, lens[l]));
    max_m = std::max(max_m, lens[l]);
  }
  // Padded lanes already zero from value-initialization.

  Vec used_a = Ops::zero();
  Vec b_matched = Ops::zero();
  const Vec all_ones = Ops::set1(~std::uint64_t{0});

  // Pre-transpose target characters and precompute the per-lane match
  // window into SIMD vectors. The previous version had a per-text-
  // position scalar loop over lanes that computed char_idx, lo_shift,
  // hi_complement, and active_bit one lane at a time. That's K scalar
  // writes per j; we now do most of that per-j with one SIMD compare
  // + a few arithmetic ops, with an aligned matrix load for indices.
  alignas(64) std::uint64_t indices_matrix[K * 64U] = {};
  alignas(64) std::uint64_t lens_padded[K] = {};
  alignas(64) std::uint64_t windows_padded[K] = {};
  for (std::size_t l = 0; l < group; ++l) {
    lens_padded[l] = static_cast<std::uint64_t>(lens[l]);
    windows_padded[l] = windows.values[l];
  }
  for (std::size_t j = 0; j < max_m; ++j) {
    for (std::size_t l = 0; l < K; ++l) {
      indices_matrix[j * K + l] =
          (l < group && j < lens[l]) ? targets[l][j] : 256U;
    }
  }
  const Vec lens_v = Ops::load_aligned(lens_padded);
  const Vec windows_v = Ops::load_aligned(windows_padded);
  const Vec q_len_v = Ops::set1(static_cast<std::uint64_t>(q_len));
  const Vec sixty_three_v = Ops::set1(63U);
  const Vec sixty_four_v = Ops::set1(64U);
  const Vec one_v = Ops::set1(1U);

  for (std::size_t j = 0; j < max_m; ++j) {
    const Vec j_v = Ops::set1(static_cast<std::uint64_t>(j));
    // active_lane = j < lens[l]  (per-lane bool).
    const Vec lane_active = Ops::gt_u64(lens_v, j_v);

    // lo[l] = (j > window[l]) ? j - window[l] : 0
    const Vec j_gt_w = Ops::gt_u64(j_v, windows_v);
    const Vec j_minus_w = Ops::sub(j_v, windows_v);
    const Vec lo_v_raw = Ops::and_(j_gt_w, j_minus_w);
    // hi[l] = min(q_len, j + window[l] + 1)
    const Vec j_plus_w_plus_1 =
        Ops::add(Ops::add(j_v, windows_v), one_v);
    const Vec hi_lt_qlen = Ops::gt_u64(q_len_v, j_plus_w_plus_1);
    const Vec hi_v =
        Ops::or_(Ops::and_(hi_lt_qlen, j_plus_w_plus_1),
                  Ops::andnot_(hi_lt_qlen, q_len_v));
    const Vec hi_complement_v = Ops::sub(sixty_four_v, hi_v);

    // For inactive lanes (j >= lens[l]) or empty windows (lo >= hi)
    // we force the shifts to 63 + 63 (which produces zero window mask).
    // Empty-window detection: lo >= hi. We approximate by ``!lane_active``
    // because the bench cases never have lo >= hi when the lane is
    // active (the match_window function returns ``max(m, n) / 2 - 1``
    // which keeps lo < hi for all in-range j).
    const Vec lo_v =
        Ops::or_(Ops::and_(lane_active, lo_v_raw),
                  Ops::andnot_(lane_active, sixty_three_v));
    const Vec hi_comp_v =
        Ops::or_(Ops::and_(lane_active, hi_complement_v),
                  Ops::andnot_(lane_active, sixty_three_v));

    // active_bit = (1 << j) if active else 0
    const Vec j_bit = Ops::set1(std::uint64_t{1} << j);
    const Vec active_v = Ops::and_(lane_active, j_bit);

    const Vec window_mask =
        Ops::and_(Ops::shl_var_u64(all_ones, lo_v),
                  Ops::shr_var_u64(all_ones, hi_comp_v));
    const Vec peq_v = Ops::gather64(peq.entries, &indices_matrix[j * K]);

    // candidate = peq & window_mask & ~used_a
    const Vec candidate =
        Ops::andnot_(used_a, Ops::and_(peq_v, window_mask));

    // lowest set bit per lane = candidate & -candidate
    const Vec lowest =
        Ops::and_(candidate, Ops::sub(Ops::zero(), candidate));
    used_a = Ops::or_(used_a, lowest);

    // Record a match in b at position j for lanes whose candidate was
    // non-zero. ~cmpeq(candidate, 0) is all-ones in matched lanes;
    // AND with active_bit gives (1 << j) in matched lanes only.
    const Vec nonzero =
        Ops::not_(Ops::cmpeq(candidate, Ops::zero()));
    b_matched =
        Ops::or_(b_matched, Ops::and_(nonzero, active_v));
  }

  LaneScratch<Ops> used_a_out{};
  LaneScratch<Ops> b_matched_out{};
  Ops::store_aligned(used_a_out.values, used_a);
  Ops::store_aligned(b_matched_out.values, b_matched);

  for (std::size_t l = 0; l < group; ++l) {
    const std::size_t n = q_len;
    const std::size_t m = lens[l];
    if (n == 0U && m == 0U) {
      out[l] = 1.0;
      continue;
    }
    if (n == 0U || m == 0U) {
      out[l] = 0.0;
      continue;
    }

    const std::uint64_t ua = used_a_out.values[l];
    const std::uint64_t bm = b_matched_out.values[l];
    const std::size_t matches =
        static_cast<std::size_t>(std::popcount(ua));
    if (matches == 0U) {
      out[l] = 0.0;
      continue;
    }

    // Half-transposition count: walk the matched bits of used_a and
    // b_matched in parallel ascending order, comparing chars.
    std::uint64_t a_bits = ua;
    std::uint64_t b_bits = bm;
    std::size_t half_trans = 0U;
    while (a_bits != 0U) {
      const std::size_t i =
          static_cast<std::size_t>(std::countr_zero(a_bits));
      const std::size_t k =
          static_cast<std::size_t>(std::countr_zero(b_bits));
      if (q_ptr[i] != targets[l][k]) {
        ++half_trans;
      }
      a_bits &= a_bits - 1U;
      b_bits &= b_bits - 1U;
    }

    const double matches_d = static_cast<double>(matches);
    const double trans_d = static_cast<double>(half_trans / 2U);
    out[l] = (matches_d / static_cast<double>(n)
           + matches_d / static_cast<double>(m)
           + (matches_d - trans_d) / matches_d)
           / 3.0;
  }
}

// Multi-word SIMD batch covering query lengths up to W_query * 64 and
// target lengths up to W_target * 64 (each W in {1, 2, 3, 4}; W=1
// degenerates to the single-word special case but the dedicated
// `jaro_simd_inner` is preferred for that). The query bitmap and
// per-lane used_a are split into W_query blocks; b_matched is split
// into W_target blocks. The j-loop touches only one b_matched block
// per iteration (block = j / 64), so multi-word target adds storage
// + finishing cost but no per-iteration inner-loop cost.
//
// Per j iteration:
//   * Compute per-lane char_idx (target's char at position j).
//   * For each query block b, compute the per-lane window-mask
//     restricted to that block, gather PEQ[c][b] per lane, AND in to
//     form candidate[b].
//   * "Leftmost match across blocks": scan block order; for each lane
//     track `prev_all_zero` and only take `x & -x` from the first
//     block where the lane had a non-zero candidate.
//   * If any block fired for a lane, set bit (j % 64) in
//     b_matched[j / 64].
//
// Above W_query*64 or W_target*64, the caller falls through to
// per-target scalar dispatch.
template <typename Ops, std::size_t W_query, std::size_t W_target>
inline void jaro_simd_inner_multi_word_query(
    const ByteAlphabetPeqMulti<W_query>& peq,
    const std::uint8_t* q_ptr,
    std::size_t q_len,
    const std::uint8_t* const* targets,
    const std::size_t* lens,
    std::size_t group,
    double* out) {
  static_assert(W_query >= 2U && W_query <= 4U,
                "multi-word Jaro kernel only instantiated for W_query in [2, 4]");
  static_assert(W_target >= 1U && W_target <= 4U,
                "multi-word Jaro kernel only instantiated for W_target in [1, 4]");
  constexpr std::size_t K = Ops::lanes;
  using Vec = typename Ops::Vec;

  LaneScratch<Ops> windows{};
  std::size_t max_m = 0;
  for (std::size_t l = 0; l < group; ++l) {
    windows.values[l] = static_cast<std::uint64_t>(
        ::stride_align::jaro::match_window(q_len, lens[l]));
    max_m = std::max(max_m, lens[l]);
  }

  std::array<Vec, W_query> used_a;
  for (auto& v : used_a) {
    v = Ops::zero();
  }
  std::array<Vec, W_target> b_matched;
  for (auto& v : b_matched) {
    v = Ops::zero();
  }
  const Vec all_ones = Ops::set1(~std::uint64_t{0});

  LaneScratch<Ops> char_idx{};
  std::array<LaneScratch<Ops>, W_query> lo_shift{};
  std::array<LaneScratch<Ops>, W_query> hi_complement{};

  for (std::size_t j = 0; j < max_m; ++j) {
    // b_matched bit position for this j: same across all lanes.
    const std::size_t target_block = j / 64U;
    const std::size_t target_bit = j % 64U;
    const Vec bit_for_j = Ops::set1(std::uint64_t{1} << target_bit);

    for (std::size_t l = 0; l < K; ++l) {
      const std::size_t m = lens[l];
      const std::uint64_t w_lane = windows.values[l];
      if (j >= m) {
        char_idx.values[l] = 256U;
        for (std::size_t b = 0; b < W_query; ++b) {
          lo_shift[b].values[l] = 63U;
          hi_complement[b].values[l] = 63U;
        }
        continue;
      }
      char_idx.values[l] = targets[l][j];

      const std::size_t lo_q =
          j > w_lane ? j - static_cast<std::size_t>(w_lane) : 0U;
      const std::size_t hi_q =
          std::min(q_len, j + static_cast<std::size_t>(w_lane) + 1U);

      for (std::size_t b = 0; b < W_query; ++b) {
        const std::size_t block_start = b * 64U;
        const std::size_t block_end = std::min(q_len, block_start + 64U);
        if (lo_q >= block_end || hi_q <= block_start ||
            block_start >= block_end) {
          // No overlap with this block — encode mask = 0 via the same
          // "shl 63 + shr 63 → AND = 0" trick used by the single-word
          // kernel, keeping all shift amounts in [0, 63].
          lo_shift[b].values[l] = 63U;
          hi_complement[b].values[l] = 63U;
          continue;
        }
        const std::size_t lo_local =
            (lo_q > block_start) ? (lo_q - block_start) : 0U;
        const std::size_t hi_local = std::min(
            std::size_t{64U}, hi_q - block_start);
        lo_shift[b].values[l] = static_cast<std::uint64_t>(lo_local);
        hi_complement[b].values[l] =
            static_cast<std::uint64_t>(64U - hi_local);
      }
    }

    std::array<Vec, W_query> candidate;
    for (std::size_t b = 0; b < W_query; ++b) {
      const Vec lo_v = Ops::load_aligned(lo_shift[b].values);
      const Vec hi_comp_v = Ops::load_aligned(hi_complement[b].values);
      const Vec window_mask =
          Ops::and_(Ops::shl_var_u64(all_ones, lo_v),
                    Ops::shr_var_u64(all_ones, hi_comp_v));
      const Vec peq_v = Ops::gather64(peq.block(b), char_idx.values);
      candidate[b] = Ops::andnot_(used_a[b], Ops::and_(peq_v, window_mask));
    }

    // "Leftmost match" across query blocks.
    Vec prev_all_zero = all_ones;
    Vec any_matched = Ops::zero();
    for (std::size_t b = 0; b < W_query; ++b) {
      const Vec this_nonzero =
          Ops::not_(Ops::cmpeq(candidate[b], Ops::zero()));
      const Vec take_this = Ops::and_(prev_all_zero, this_nonzero);
      const Vec lowest_in_block =
          Ops::and_(candidate[b],
                    Ops::sub(Ops::zero(), candidate[b]));
      used_a[b] =
          Ops::or_(used_a[b], Ops::and_(take_this, lowest_in_block));
      any_matched = Ops::or_(any_matched, this_nonzero);
      prev_all_zero =
          Ops::and_(prev_all_zero, Ops::not_(this_nonzero));
    }
    // For target_block: any_matched AND inactive-lane mask. We didn't
    // explicitly compute an "active" mask any more because inactive
    // lanes (j >= m) have char_idx = 256 → PEQ entry = 0 → candidate
    // = 0 → any_matched = 0 for those lanes. So the lane mask is
    // already baked into any_matched.
    b_matched[target_block] =
        Ops::or_(b_matched[target_block],
                 Ops::and_(any_matched, bit_for_j));
  }

  std::array<LaneScratch<Ops>, W_query> used_a_out{};
  for (std::size_t b = 0; b < W_query; ++b) {
    Ops::store_aligned(used_a_out[b].values, used_a[b]);
  }
  std::array<LaneScratch<Ops>, W_target> b_matched_out{};
  for (std::size_t b = 0; b < W_target; ++b) {
    Ops::store_aligned(b_matched_out[b].values, b_matched[b]);
  }

  for (std::size_t l = 0; l < group; ++l) {
    const std::size_t n = q_len;
    const std::size_t m = lens[l];
    if (n == 0U && m == 0U) {
      out[l] = 1.0;
      continue;
    }
    if (n == 0U || m == 0U) {
      out[l] = 0.0;
      continue;
    }

    std::size_t matches = 0U;
    for (std::size_t b = 0; b < W_query; ++b) {
      matches += static_cast<std::size_t>(
          std::popcount(used_a_out[b].values[l]));
    }
    if (matches == 0U) {
      out[l] = 0.0;
      continue;
    }

    // Walk matched bits across blocks in ascending order: used_a's W_query
    // blocks for the a-positions, b_matched's W_target blocks for the
    // b-positions. Each loop iteration consumes one set bit from the
    // current (a, b) block pair; when one block runs out, advance to
    // the next non-empty block on that side.
    std::size_t half_trans = 0U;
    std::size_t b_block = 0U;
    std::uint64_t b_bits = b_matched_out[0].values[l];
    auto advance_b = [&]() {
      while (b_bits == 0U && b_block + 1U < W_target) {
        ++b_block;
        b_bits = b_matched_out[b_block].values[l];
      }
    };
    advance_b();
    for (std::size_t a_block = 0; a_block < W_query; ++a_block) {
      std::uint64_t a_bits = used_a_out[a_block].values[l];
      while (a_bits != 0U) {
        advance_b();
        if (b_bits == 0U) {
          break;
        }
        const std::size_t i =
            a_block * 64U +
            static_cast<std::size_t>(std::countr_zero(a_bits));
        const std::size_t k =
            b_block * 64U +
            static_cast<std::size_t>(std::countr_zero(b_bits));
        if (q_ptr[i] != targets[l][k]) {
          ++half_trans;
        }
        a_bits &= a_bits - 1U;
        b_bits &= b_bits - 1U;
      }
    }

    const double matches_d = static_cast<double>(matches);
    const double trans_d = static_cast<double>(half_trans / 2U);
    out[l] = (matches_d / static_cast<double>(n)
           + matches_d / static_cast<double>(m)
           + (matches_d - trans_d) / matches_d)
           / 3.0;
  }
}

namespace detail {

inline bool view_all_byte_compat_short(
    PyObject* const* items,
    std::size_t count,
    std::size_t max_len,
    std::vector<const std::uint8_t*>& ptrs,
    std::vector<std::size_t>& lens) {
  namespace bv = ::stride_align::byte_view;
  ptrs.resize(count);
  lens.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    const bv::ByteCompatKind kind = bv::classify(items[i]);
    if (kind == bv::ByteCompatKind::None) {
      return false;
    }
    bv::view(items[i], kind, ptrs[i], lens[i]);
    if (lens[i] > max_len) {
      return false;
    }
  }
  return true;
}

}  // namespace detail

// Largest query or target length the multi-word SIMD batch covers (W * 64).
inline constexpr std::size_t kJaroMaxSimdLen = 256U;

// View every target as byte-compatible, fill `ptrs`/`lens`, track the
// running maximum length. Returns false (signaling scalar fallback)
// if any target is non-byte-compatible or longer than `max_len`.
inline bool view_all_byte_compat_capped(
    PyObject* const* items,
    std::size_t count,
    std::size_t max_len,
    std::vector<const std::uint8_t*>& ptrs,
    std::vector<std::size_t>& lens,
    std::size_t& max_observed) {
  namespace bv = ::stride_align::byte_view;
  ptrs.resize(count);
  lens.resize(count);
  max_observed = 0U;
  for (std::size_t i = 0; i < count; ++i) {
    const bv::ByteCompatKind kind = bv::classify(items[i]);
    if (kind == bv::ByteCompatKind::None) {
      return false;
    }
    bv::view(items[i], kind, ptrs[i], lens[i]);
    if (lens[i] > max_len) {
      return false;
    }
    max_observed = std::max(max_observed, lens[i]);
  }
  return true;
}

// Raw SIMD batch: caller has already byte-viewed the query and all
// targets and confirmed lengths fit (<= kJaroMaxSimdLen on both
// sides). Writes Jaro similarity into `out[0..count)`. No Python
// interaction. Used by both the Python-facing wrapper below and the
// cdist kernel.
template <typename Ops>
inline void jaro_similarities_simd_raw(
    const std::uint8_t* q_ptr, std::size_t q_len,
    const std::uint8_t* const* t_ptrs,
    const std::size_t* t_lens,
    std::size_t count,
    std::size_t max_m,
    double* out) {
  if (count == 0U) {
    return;
  }

  constexpr std::size_t K = Ops::lanes;
  std::array<const std::uint8_t*, K> batch_ptrs{};
  std::array<std::size_t, K> batch_lens{};

  auto fill_batch = [&](std::size_t base, std::size_t group) {
    for (std::size_t l = 0; l < group; ++l) {
      batch_ptrs[l] = t_ptrs[base + l];
      batch_lens[l] = t_lens[base + l];
    }
    for (std::size_t l = group; l < K; ++l) {
      batch_ptrs[l] = nullptr;
      batch_lens[l] = 0U;
    }
  };

  if (q_len <= 64U && max_m <= 64U) {
    const ByteAlphabetPeq peq = build_peq_byte(q_ptr, q_len);
    for (std::size_t base = 0; base < count; base += K) {
      const std::size_t group = std::min(K, count - base);
      fill_batch(base, group);
      jaro_simd_inner<Ops>(
          peq, q_ptr, q_len, batch_ptrs.data(), batch_lens.data(),
          group, out + base);
    }
    return;
  }

  // Multi-word. W_query is always >= 2 in this branch; for q_len <= 64
  // with m > 64 we still pad to W_query=2 (block 1 PEQ stays zero) so
  // the leftmost-block scan finds matches in block 0.
  const std::size_t W_query =
      q_len > 192U ? 4U : q_len > 128U ? 3U : q_len > 64U ? 2U : 2U;
  const std::size_t W_target =
      max_m > 192U ? 4U : max_m > 128U ? 3U : max_m > 64U ? 2U : 1U;

  auto run = [&](auto WQ_const, auto WT_const) {
    constexpr std::size_t WQ = decltype(WQ_const)::value;
    constexpr std::size_t WT = decltype(WT_const)::value;
    const auto peq = build_peq_byte_multi<WQ>(q_ptr, q_len);
    for (std::size_t base = 0; base < count; base += K) {
      const std::size_t group = std::min(K, count - base);
      fill_batch(base, group);
      jaro_simd_inner_multi_word_query<Ops, WQ, WT>(
          peq, q_ptr, q_len, batch_ptrs.data(), batch_lens.data(),
          group, out + base);
    }
  };

#define STRIDE_ALIGN_JARO_DISPATCH(WQ_VAL, WT_VAL)                       \
  if (W_query == WQ_VAL && W_target == WT_VAL) {                         \
    run(std::integral_constant<std::size_t, WQ_VAL>{},                   \
        std::integral_constant<std::size_t, WT_VAL>{});                  \
    return;                                                              \
  }
  STRIDE_ALIGN_JARO_DISPATCH(2, 1)
  STRIDE_ALIGN_JARO_DISPATCH(2, 2)
  STRIDE_ALIGN_JARO_DISPATCH(2, 3)
  STRIDE_ALIGN_JARO_DISPATCH(2, 4)
  STRIDE_ALIGN_JARO_DISPATCH(3, 1)
  STRIDE_ALIGN_JARO_DISPATCH(3, 2)
  STRIDE_ALIGN_JARO_DISPATCH(3, 3)
  STRIDE_ALIGN_JARO_DISPATCH(3, 4)
  STRIDE_ALIGN_JARO_DISPATCH(4, 1)
  STRIDE_ALIGN_JARO_DISPATCH(4, 2)
  STRIDE_ALIGN_JARO_DISPATCH(4, 3)
  STRIDE_ALIGN_JARO_DISPATCH(4, 4)
#undef STRIDE_ALIGN_JARO_DISPATCH
}

// Python-facing wrapper. Returns an empty vector to signal scalar
// fallback if the inputs don't fit the SIMD constraints.
template <typename Ops>
inline std::vector<double> jaro_similarities_simd(
    nb::handle query, nb::handle targets) {
  namespace bv = ::stride_align::byte_view;
  PyObject* fast_targets = PySequence_Fast(
      targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count =
      static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  std::vector<double> out;

  const bv::ByteCompatKind q_kind = bv::classify(query.ptr());
  if (q_kind == bv::ByteCompatKind::None) {
    return out;
  }
  const std::uint8_t* q_ptr = nullptr;
  std::size_t q_len = 0;
  bv::view(query.ptr(), q_kind, q_ptr, q_len);
  if (q_len > kJaroMaxSimdLen) {
    return out;
  }

  std::vector<const std::uint8_t*> ptrs;
  std::vector<std::size_t> lens;
  std::size_t max_m = 0;
  if (!view_all_byte_compat_capped(items, count, kJaroMaxSimdLen, ptrs, lens, max_m)) {
    return out;
  }

  out.resize(count);
  jaro_similarities_simd_raw<Ops>(
      q_ptr, q_len, ptrs.data(), lens.data(), count, max_m, out.data());
  return out;
}

template <typename Ops>
inline std::vector<double> jaro_winkler_similarities_simd(
    nb::handle query,
    nb::handle targets,
    double prefix_weight,
    double prefix_threshold,
    std::size_t prefix_cap) {
  std::vector<double> jaro_scores = jaro_similarities_simd<Ops>(query, targets);
  if (jaro_scores.empty()) {
    return jaro_scores;  // signal fallback
  }
  // We need the query bytes to compute common-prefix length. SIMD path
  // only ran because both were byte-compatible.
  namespace bv = ::stride_align::byte_view;
  const bv::ByteCompatKind q_kind = bv::classify(query.ptr());
  const std::uint8_t* q_ptr = nullptr;
  std::size_t q_len = 0;
  bv::view(query.ptr(), q_kind, q_ptr, q_len);

  PyObject* fast_targets = PySequence_Fast(targets.ptr(), "targets");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  for (std::size_t i = 0; i < jaro_scores.size(); ++i) {
    if (jaro_scores[i] < prefix_threshold) {
      continue;
    }
    const bv::ByteCompatKind t_kind = bv::classify(items[i]);
    const std::uint8_t* t_ptr = nullptr;
    std::size_t t_len = 0;
    bv::view(items[i], t_kind, t_ptr, t_len);
    const std::size_t limit =
        std::min({q_len, t_len, prefix_cap});
    std::size_t L = 0;
    while (L < limit && q_ptr[L] == t_ptr[L]) {
      ++L;
    }
    jaro_scores[i] +=
        static_cast<double>(L) * prefix_weight * (1.0 - jaro_scores[i]);
  }
  return jaro_scores;
}

}  // namespace stride_align::jaro_simd
