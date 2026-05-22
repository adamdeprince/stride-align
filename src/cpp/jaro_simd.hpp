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

  LaneScratch<Ops> char_idx{};
  LaneScratch<Ops> lo_shift{};
  LaneScratch<Ops> hi_complement{};
  LaneScratch<Ops> active_bit{};

  for (std::size_t j = 0; j < max_m; ++j) {
    for (std::size_t l = 0; l < K; ++l) {
      const std::size_t m = lens[l];
      const std::uint64_t w = windows.values[l];
      if (j < m) {
        char_idx.values[l] = targets[l][j];
        const std::size_t lo = j > w ? j - static_cast<std::size_t>(w) : 0U;
        const std::size_t hi =
            std::min(q_len, j + static_cast<std::size_t>(w) + 1U);
        if (lo < hi) {
          lo_shift.values[l] = static_cast<std::uint64_t>(lo);
          hi_complement.values[l] = static_cast<std::uint64_t>(64U - hi);
          active_bit.values[l] = std::uint64_t{1} << j;
          continue;
        }
      }
      // Empty window or past-end: encode window_mask = 0 via shifts
      // that stay in [0, 63] but whose AND is empty. shl(~0, 63) =
      // 0x8000...; shr(~0, 63) = 0x1; their AND is 0.
      char_idx.values[l] = 256U;  // sentinel PEQ row (zeroed)
      lo_shift.values[l] = 63U;
      hi_complement.values[l] = 63U;
      active_bit.values[l] = 0U;
    }

    const Vec lo_v = Ops::load_aligned(lo_shift.values);
    const Vec hi_comp_v = Ops::load_aligned(hi_complement.values);
    const Vec active_v = Ops::load_aligned(active_bit.values);
    const Vec window_mask =
        Ops::and_(Ops::shl_var_u64(all_ones, lo_v),
                  Ops::shr_var_u64(all_ones, hi_comp_v));
    const Vec peq_v = Ops::gather64(peq.entries, char_idx.values);

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

// Multi-word SIMD batch for queries longer than 64 chars. Targets
// still ≤ 64 (b_matched stays a single 64-bit word per lane). W is
// the number of 64-bit blocks the query bitmap spans (1 to 4 covers
// query lengths 1..256).
//
// Per j iteration:
//   * Compute per-lane char_idx and active_bit (single 64-bit word
//     for b_matched, so j < 64 always).
//   * For each block b, compute the per-lane window-mask restricted
//     to that block (window in the query: bits [lo_q, hi_q) of the
//     query, intersected with the block's [b*64, (b+1)*64) range).
//   * Gather PEQ[c][b] per lane and AND in to form candidate[b].
//   * Across the W candidate vectors, pick the lowest block with any
//     non-zero lane and within that block take `x & -x` for the
//     leftmost match. Update used_a[b] only on lanes that picked
//     block b.
//   * If any of the W blocks fired for a lane, set bit j in
//     b_matched.
//
// Constraint: m ≤ 64 (b_matched is single-word). Above that the
// caller falls through to per-target scalar dispatch.
template <typename Ops, std::size_t W>
inline void jaro_simd_inner_multi_word_query(
    const ByteAlphabetPeqMulti<W>& peq,
    const std::uint8_t* q_ptr,
    std::size_t q_len,
    const std::uint8_t* const* targets,
    const std::size_t* lens,
    std::size_t group,
    double* out) {
  static_assert(W >= 2U && W <= 4U,
                "multi-word Jaro kernel only instantiated for W in [2, 4]");
  constexpr std::size_t K = Ops::lanes;
  using Vec = typename Ops::Vec;

  LaneScratch<Ops> windows{};
  std::size_t max_m = 0;
  for (std::size_t l = 0; l < group; ++l) {
    windows.values[l] = static_cast<std::uint64_t>(
        ::stride_align::jaro::match_window(q_len, lens[l]));
    max_m = std::max(max_m, lens[l]);
  }

  std::array<Vec, W> used_a;
  for (auto& v : used_a) {
    v = Ops::zero();
  }
  Vec b_matched = Ops::zero();
  const Vec all_ones = Ops::set1(~std::uint64_t{0});

  LaneScratch<Ops> char_idx{};
  LaneScratch<Ops> active_bit{};
  // Per-block scratch for the shift amounts that build the window
  // mask restricted to that block.
  std::array<LaneScratch<Ops>, W> lo_shift{};
  std::array<LaneScratch<Ops>, W> hi_complement{};

  for (std::size_t j = 0; j < max_m; ++j) {
    for (std::size_t l = 0; l < K; ++l) {
      const std::size_t m = lens[l];
      const std::uint64_t w_lane = windows.values[l];
      if (j >= m) {
        char_idx.values[l] = 256U;
        active_bit.values[l] = 0U;
        for (std::size_t b = 0; b < W; ++b) {
          lo_shift[b].values[l] = 63U;
          hi_complement[b].values[l] = 63U;
        }
        continue;
      }
      char_idx.values[l] = targets[l][j];
      active_bit.values[l] = std::uint64_t{1} << j;

      const std::size_t lo_q =
          j > w_lane ? j - static_cast<std::size_t>(w_lane) : 0U;
      const std::size_t hi_q =
          std::min(q_len, j + static_cast<std::size_t>(w_lane) + 1U);

      for (std::size_t b = 0; b < W; ++b) {
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

    const Vec active_v = Ops::load_aligned(active_bit.values);

    std::array<Vec, W> candidate;
    for (std::size_t b = 0; b < W; ++b) {
      const Vec lo_v = Ops::load_aligned(lo_shift[b].values);
      const Vec hi_comp_v = Ops::load_aligned(hi_complement[b].values);
      const Vec window_mask =
          Ops::and_(Ops::shl_var_u64(all_ones, lo_v),
                    Ops::shr_var_u64(all_ones, hi_comp_v));
      const Vec peq_v = Ops::gather64(peq.block(b), char_idx.values);
      candidate[b] = Ops::andnot_(used_a[b], Ops::and_(peq_v, window_mask));
    }

    // "Leftmost match" across blocks: pick the lowest block b that has
    // a non-zero candidate per lane, take its `x & -x`, leave higher
    // blocks unchanged for that lane.
    Vec prev_all_zero = all_ones;
    Vec any_matched = Ops::zero();
    for (std::size_t b = 0; b < W; ++b) {
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
    b_matched =
        Ops::or_(b_matched, Ops::and_(any_matched, active_v));
  }

  std::array<LaneScratch<Ops>, W> used_a_out{};
  for (std::size_t b = 0; b < W; ++b) {
    Ops::store_aligned(used_a_out[b].values, used_a[b]);
  }
  LaneScratch<Ops> b_matched_out{};
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

    std::size_t matches = 0U;
    for (std::size_t b = 0; b < W; ++b) {
      matches += static_cast<std::size_t>(
          std::popcount(used_a_out[b].values[l]));
    }
    if (matches == 0U) {
      out[l] = 0.0;
      continue;
    }

    // Walk matched bits across blocks in ascending order, parallel
    // with b_matched's bit walk.
    std::uint64_t b_bits = b_matched_out.values[l];
    std::size_t half_trans = 0U;
    for (std::size_t b = 0; b < W; ++b) {
      std::uint64_t a_bits = used_a_out[b].values[l];
      while (a_bits != 0U) {
        const std::size_t i =
            b * 64U + static_cast<std::size_t>(std::countr_zero(a_bits));
        const std::size_t k =
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

// Largest query length the multi-word SIMD batch covers (W * 64).
inline constexpr std::size_t kJaroMaxSimdQueryLen = 256U;

// SIMD batch entrypoint. Returns an empty vector when constraints are
// not met (query not byte-compatible, query length > 256, any target
// not byte-compatible, or any target length > 64); the caller then
// falls through to per-target scalar dispatch.
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
    return out;  // empty -> signal fallback
  }
  const std::uint8_t* q_ptr = nullptr;
  std::size_t q_len = 0;
  bv::view(query.ptr(), q_kind, q_ptr, q_len);
  if (q_len > kJaroMaxSimdQueryLen) {
    return out;
  }

  std::vector<const std::uint8_t*> ptrs;
  std::vector<std::size_t> lens;
  if (!detail::view_all_byte_compat_short(items, count, 64U, ptrs, lens)) {
    return out;
  }

  out.resize(count);
  if (count == 0U) {
    return out;
  }

  constexpr std::size_t K = Ops::lanes;
  std::array<const std::uint8_t*, K> batch_ptrs{};
  std::array<std::size_t, K> batch_lens{};

  auto fill_batch = [&](std::size_t base, std::size_t group) {
    for (std::size_t l = 0; l < group; ++l) {
      batch_ptrs[l] = ptrs[base + l];
      batch_lens[l] = lens[base + l];
    }
    for (std::size_t l = group; l < K; ++l) {
      batch_ptrs[l] = nullptr;
      batch_lens[l] = 0U;
    }
  };

  if (q_len > 64U) {
    // Multi-word query path: W in {2, 3, 4} for query lengths
    // 65..128, 129..192, 193..256. Targets still ≤ 64 (single-word
    // b_matched).
    const std::size_t W = (q_len + 63U) / 64U;
    auto run_w = [&](auto W_const) {
      constexpr std::size_t W_compile = decltype(W_const)::value;
      const auto peq = build_peq_byte_multi<W_compile>(q_ptr, q_len);
      for (std::size_t base = 0; base < count; base += K) {
        const std::size_t group = std::min(K, count - base);
        fill_batch(base, group);
        jaro_simd_inner_multi_word_query<Ops, W_compile>(
            peq, q_ptr, q_len, batch_ptrs.data(), batch_lens.data(),
            group, out.data() + base);
      }
    };
    if (W == 2U) {
      run_w(std::integral_constant<std::size_t, 2U>{});
    } else if (W == 3U) {
      run_w(std::integral_constant<std::size_t, 3U>{});
    } else {
      run_w(std::integral_constant<std::size_t, 4U>{});
    }
    return out;
  }

  // Single-word path
  const ByteAlphabetPeq peq = build_peq_byte(q_ptr, q_len);
  for (std::size_t base = 0; base < count; base += K) {
    const std::size_t group = std::min(K, count - base);
    fill_batch(base, group);
    jaro_simd_inner<Ops>(
        peq,
        q_ptr,
        q_len,
        batch_ptrs.data(),
        batch_lens.data(),
        group,
        out.data() + base);
  }

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
