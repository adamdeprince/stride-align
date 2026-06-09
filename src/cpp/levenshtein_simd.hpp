#pragma once

// Multi-target SIMD specialization for Levenshtein's *_scores entry points.
//
// Each SIMD lane holds one target's Myers state (vp, vn, score). Lane width
// is fixed at 64 bits, so the pattern length must fit in a single 64-bit
// word (m <= 64). Longer patterns fall through to the scalar dispatch in
// levenshtein_dispatch.hpp, which uses Hyyrö's multi-word algorithm.
//
// Per-backend `Ops` structs supply the SIMD primitives (Vec type, lane
// count, set/and/or/xor/add/shr63/shl1/gather/store). The kernel below is
// templated on Ops.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "levenshtein_dispatch.hpp"
#include "preprocess.hpp"
#include "stride_align/alignment.hpp"
#include "stride_align/levenshtein.hpp"

namespace stride_align::levenshtein_simd {

namespace nb = nanobind;

// Templated on HasCutoff so the no-cutoff path (the common case) doesn't pay
// for the extra threshold/done tracking. When HasCutoff is true, a lane
// freezes once its score exceeds `cutoff + (target_length - k - 1)` (the
// minimum possible final distance from this state), and on return its
// score is clamped to `cutoff + 1` (the sentinel rapidfuzz uses).
// `cutoffs` is a per-lane array of length `batch_count` (one cutoff per
// pair). Each lane bails when its own score exceeds its own cutoff plus
// its remaining-chars allowance, and bailed scores are clamped to
// `cutoffs[l] + 1` (per-lane sentinel). Pass nullptr together with
// `HasCutoff=false` to skip the bail path entirely.
template <typename Ops, bool HasCutoff = false>
inline void myers_batch_single_word(
    const std::uint64_t* peq,
    const std::uint8_t* const* targets,
    const std::size_t* target_lengths,
    std::size_t batch_count,
    std::size_t m,
    Score* out,
    const std::size_t* cutoffs = nullptr) {
  using Vec = typename Ops::Vec;
  constexpr std::size_t lanes = Ops::lanes;

  std::size_t max_len = 0;
  for (std::size_t l = 0; l < batch_count; ++l) {
    max_len = std::max(max_len, target_lengths[l]);
  }

  const std::uint64_t init_vp = (m == 64U) ? ~std::uint64_t{0}
                                            : ((std::uint64_t{1} << m) - 1U);
  Vec vp = Ops::set1(init_vp);
  Vec vn = Ops::zero();
  Vec score = Ops::set1(static_cast<std::uint64_t>(m));
  const Vec one_v = Ops::set1(1U);

  // Test mask for the pattern's last-row bit (Myers' score update fires on
  // bit m-1 of HP/HN, not bit 63 — those positions only coincide when m=64).
  const Vec top_bit_mask = Ops::set1(std::uint64_t{1} << (m - 1U));
  const Vec zero_v = Ops::zero();

  // Pre-transpose target characters into [max_len][lanes] uint64 matrix
  // once per batch. Inner loop just does an aligned gather of the
  // current k's lane indices, no per-text-position scalar branch.
  alignas(64) std::uint64_t indices_matrix[lanes * 64U] = {};
  for (std::size_t k = 0; k < max_len; ++k) {
    for (std::size_t l = 0; l < lanes; ++l) {
      indices_matrix[k * lanes + l] =
          (l < batch_count && k < target_lengths[l]) ? targets[l][k] : 0U;
    }
  }
  // Per-lane target lengths as a SIMD vector. The per-k active mask is
  // ``target_lens_v > k_v`` (single SIMD compare, replaces 8 scalar
  // branches per text position).
  alignas(64) std::uint64_t target_lengths_padded[lanes] = {};
  for (std::size_t l = 0; l < batch_count; ++l) {
    target_lengths_padded[l] = static_cast<std::uint64_t>(target_lengths[l]);
  }
  const Vec target_lens_v = Ops::load_aligned(target_lengths_padded);

  // Cutoff bookkeeping. `done` is sticky-per-lane (set once score crosses
  // the bail threshold). `threshold` carries `cutoffs[l] + remaining_chars`
  // for each lane; remaining_chars decrements by one per column. Lanes
  // past their target end stop receiving deltas via the active mask, so
  // their threshold value is irrelevant after that point. `batch_mask`
  // is all-ones in the first batch_count lanes (zero in the unused
  // tail) so the all-lanes-settled break test ignores unused slots.
  Vec done = Ops::zero();
  Vec threshold = Ops::zero();
  Vec batch_mask = Ops::zero();
  if constexpr (HasCutoff) {
    alignas(64) std::uint64_t init_threshold[lanes];
    alignas(64) std::uint64_t mask_data[lanes];
    for (std::size_t l = 0; l < lanes; ++l) {
      if (l < batch_count) {
        const std::size_t lane_cutoff = cutoffs[l];
        init_threshold[l] = (target_lengths[l] > 0)
            ? lane_cutoff + target_lengths[l] - 1U
            : lane_cutoff;
        mask_data[l] = ~std::uint64_t{0};
      } else {
        init_threshold[l] = 0;
        mask_data[l] = 0;
      }
    }
    threshold = Ops::load_aligned(init_threshold);
    batch_mask = Ops::load_aligned(mask_data);
  }

  for (std::size_t k = 0; k < max_len; ++k) {
    const Vec eq = Ops::gather64(peq, &indices_matrix[k * lanes]);
    // active_v: ~0 per lane where target_length > k, else 0.
    const Vec k_v = Ops::set1(static_cast<std::uint64_t>(k));
    const Vec active_v = Ops::gt_u64(target_lens_v, k_v);
    Vec eff_active = active_v;
    if constexpr (HasCutoff) {
      eff_active = Ops::andnot_(done, active_v);
    }

    const Vec x = Ops::or_(eq, vn);
    const Vec xv = Ops::and_(x, vp);
    const Vec sum = Ops::add(xv, vp);
    const Vec d0 = Ops::or_(Ops::xor_(sum, vp), x);
    const Vec hp = Ops::or_(vn, Ops::not_(Ops::or_(d0, vp)));
    const Vec hn = Ops::and_(d0, vp);

    // hp_set: 1 per lane if bit m-1 of hp is set, else 0. Same for hn.
    // and(hp, top_bit_mask) is either 0 or top_bit_mask; comparing != 0
    // (i.e. ~cmpeq_zero) -> all-ones / 0, & 1 -> 1 / 0.
    const Vec hp_bit = Ops::and_(hp, top_bit_mask);
    const Vec hn_bit = Ops::and_(hn, top_bit_mask);
    const Vec hp_eq_zero = Ops::cmpeq(hp_bit, zero_v);
    const Vec hn_eq_zero = Ops::cmpeq(hn_bit, zero_v);
    const Vec hp_set = Ops::andnot_(hp_eq_zero, one_v);
    const Vec hn_set = Ops::andnot_(hn_eq_zero, one_v);
    const Vec delta = Ops::and_(Ops::sub(hp_set, hn_set), eff_active);
    score = Ops::add(score, delta);

    const Vec hp_shift = Ops::or_(Ops::shl1(hp), one_v);
    const Vec hn_shift = Ops::shl1(hn);
    vp = Ops::or_(hn_shift, Ops::not_(Ops::or_(d0, hp_shift)));
    vn = Ops::and_(d0, hp_shift);

    if constexpr (HasCutoff) {
      // After this column completes, the minimum possible final score per
      // lane is score - (target_lengths[l] - k - 1). Bail when even the
      // best-case can no longer reach the cutoff. `threshold` holds
      // cutoff + (target_lengths[l] - k - 1). Gate the bail check with
      // `active_v` so lanes whose target is already exhausted don't get
      // false positives from threshold underflowing past zero.
      const Vec exceeded = Ops::and_(Ops::gt_u64(score, threshold), active_v);
      done = Ops::or_(done, exceeded);
      threshold = Ops::sub(threshold, one_v);

      // Stop the column loop entirely once every lane in the batch is
      // settled (target exhausted or bailed by cutoff). `settled =
      // ~active_v | done` is all-ones per lane that's settled; we look
      // for any unsettled lane that's still part of the batch.
      const Vec settled = Ops::or_(Ops::not_(active_v), done);
      const Vec live = Ops::andnot_(settled, batch_mask);
      if (Ops::is_zero(live)) {
        break;
      }
    }
  }

  alignas(64) std::uint64_t scores[lanes];
  if constexpr (HasCutoff) {
    // Clamp bail-time scores to cutoffs[l] + 1 so callers see a stable
    // per-lane sentinel regardless of how badly the lane diverged.
    alignas(64) std::uint64_t cutoff_plus_1_data[lanes];
    for (std::size_t l = 0; l < lanes; ++l) {
      cutoff_plus_1_data[l] = (l < batch_count) ? cutoffs[l] + 1U : 0;
    }
    const Vec cutoff_plus_1 = Ops::load_aligned(cutoff_plus_1_data);
    score = Ops::or_(
        Ops::and_(done, cutoff_plus_1),
        Ops::andnot_(done, score));
  }
  Ops::store_aligned(scores, score);
  for (std::size_t l = 0; l < batch_count; ++l) {
    out[l] = static_cast<Score>(scores[l]);
  }
}

// Validate that every materialized target is bytes/str of byte-compatible
// alphabet and that the pattern fits in 64 bits. Returns true if the SIMD
// batch path is usable for this call; otherwise the caller falls back to
// scalar dispatch.
inline bool eligible_for_simd(
    const TokenStorage& query_tokens,
    const std::vector<TokenStorage>& target_tokens) {
  if (!std::holds_alternative<std::vector<std::uint8_t>>(query_tokens)) {
    return false;
  }
  const auto& q = std::get<std::vector<std::uint8_t>>(query_tokens);
  if (q.size() > 64U || q.empty()) {
    return false;
  }
  for (const auto& storage : target_tokens) {
    if (!std::holds_alternative<std::vector<std::uint8_t>>(storage)) {
      return false;
    }
  }
  return true;
}

// Build a 256-entry PEQ table for a uint8 pattern of length <= 64.
inline std::array<std::uint64_t, 256> build_peq(std::span<const std::uint8_t> pattern) {
  std::array<std::uint64_t, 256> peq{};
  const std::uint64_t one = 1U;
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    peq[pattern[i]] |= one << i;
  }
  return peq;
}

// Build a multi-block PEQ table for a uint8 pattern of length > 64. The
// layout is `peq[b * 256 + c]` so block `b`'s 256-entry table starts at
// `peq.data() + b * 256` — exactly the base the SIMD `gather64(base,
// indices)` consumes. Sized at run-time because W depends on pattern
// length.
inline std::vector<std::uint64_t> build_peq_multi(
    std::span<const std::uint8_t> pattern,
    std::size_t W) {
  std::vector<std::uint64_t> peq(W * 256U, 0);
  const std::uint64_t one = 1U;
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    const std::size_t block = i / 64U;
    const std::size_t bit = i % 64U;
    peq[block * 256U + pattern[i]] |= one << bit;
  }
  return peq;
}

// Multi-word Myers in SIMD (one target per lane, pattern split into W
// 64-bit blocks). Each per-column step runs Hyyrö's horizontal carry
// chain across blocks per lane, in parallel across `lanes` targets. The
// wide add (xv + vp[b] + add_carry) is implemented in SIMD via two
// chained 64-bit adds with overflow detection (`gt_u64`).
// `cutoffs` is a per-lane array of length `batch_count` (one cutoff per
// pair). Same semantics as the single-word kernel: each lane bails on
// its own threshold and clamps to its own `cutoffs[l] + 1` sentinel.
template <typename Ops, std::size_t W, bool HasCutoff = false>
inline void myers_batch_multi_word(
    const std::uint64_t* peq,
    const std::uint8_t* const* targets,
    const std::size_t* target_lengths,
    std::size_t batch_count,
    std::size_t m,
    std::size_t last_bits,
    Score* out,
    const std::size_t* cutoffs = nullptr) {
  using Vec = typename Ops::Vec;
  constexpr std::size_t lanes = Ops::lanes;
  static_assert(W >= 2U, "myers_batch_multi_word is only for W >= 2; use single-word kernel for W == 1");

  std::size_t max_len = 0;
  for (std::size_t l = 0; l < batch_count; ++l) {
    max_len = std::max(max_len, target_lengths[l]);
  }

  // Per-block initial vp: last block uses `(1 << last_bits) - 1` if
  // last_bits < 64, all-ones otherwise. Earlier blocks are always
  // all-ones.
  Vec vp[W];
  Vec vn[W];
  for (std::size_t b = 0; b < W; ++b) {
    if (b == W - 1U && last_bits < 64U) {
      vp[b] = Ops::set1((std::uint64_t{1} << last_bits) - 1U);
    } else {
      vp[b] = Ops::set1(~std::uint64_t{0});
    }
    vn[b] = Ops::zero();
  }

  Vec score = Ops::set1(static_cast<std::uint64_t>(m));
  const Vec one_v = Ops::set1(1U);
  const Vec zero_v = Ops::zero();
  const Vec top_bit_last_mask = Ops::set1(std::uint64_t{1} << (last_bits - 1U));

  Vec done = Ops::zero();
  Vec threshold = Ops::zero();
  Vec batch_mask = Ops::zero();
  if constexpr (HasCutoff) {
    alignas(64) std::uint64_t init_threshold[lanes];
    alignas(64) std::uint64_t mask_data[lanes];
    for (std::size_t l = 0; l < lanes; ++l) {
      if (l < batch_count) {
        const std::size_t lane_cutoff = cutoffs[l];
        init_threshold[l] = (target_lengths[l] > 0)
            ? lane_cutoff + target_lengths[l] - 1U
            : lane_cutoff;
        mask_data[l] = ~std::uint64_t{0};
      } else {
        init_threshold[l] = 0;
        mask_data[l] = 0;
      }
    }
    threshold = Ops::load_aligned(init_threshold);
    batch_mask = Ops::load_aligned(mask_data);
  }

  alignas(64) std::uint64_t indices[lanes];
  alignas(64) std::uint64_t active[lanes];

  for (std::size_t k = 0; k < max_len; ++k) {
    for (std::size_t l = 0; l < lanes; ++l) {
      if (l < batch_count && k < target_lengths[l]) {
        indices[l] = targets[l][k];
        active[l] = ~std::uint64_t{0};
      } else {
        indices[l] = 0;
        active[l] = 0;
      }
    }
    const Vec active_v = Ops::load_aligned(active);
    Vec eff_active = active_v;
    if constexpr (HasCutoff) {
      eff_active = Ops::andnot_(done, active_v);
    }

    Vec add_carry = zero_v;
    Vec hp_carry = one_v;  // boundary: bit 0 of d0 gets a +1 carry
    Vec hn_carry = zero_v;
    Vec last_hp = zero_v;
    Vec last_hn = zero_v;

    for (std::size_t b = 0; b < W; ++b) {
      const Vec eq = Ops::gather64(peq + b * 256U, indices);

      const Vec x = Ops::or_(eq, vn[b]);
      const Vec xv = Ops::and_(x, vp[b]);

      // Wide add: xv + vp[b] + add_carry, split into two 64-bit adds
      // with overflow detection so the high half (carry to next block)
      // is correct for the full 128-bit sum.
      const Vec t1 = Ops::add(xv, vp[b]);
      const Vec carry1_mask = Ops::gt_u64(xv, t1);
      const Vec carry1 = Ops::and_(carry1_mask, one_v);
      const Vec t2 = Ops::add(t1, add_carry);
      const Vec carry2_mask = Ops::gt_u64(t1, t2);
      const Vec carry2 = Ops::and_(carry2_mask, one_v);
      add_carry = Ops::add(carry1, carry2);
      const Vec sum = t2;

      const Vec d0 = Ops::or_(Ops::xor_(sum, vp[b]), x);
      const Vec hp = Ops::or_(vn[b], Ops::not_(Ops::or_(d0, vp[b])));
      const Vec hn = Ops::and_(d0, vp[b]);

      const Vec hp_shift = Ops::or_(Ops::shl1(hp), hp_carry);
      const Vec hn_shift = Ops::or_(Ops::shl1(hn), hn_carry);
      hp_carry = Ops::shr63(hp);
      hn_carry = Ops::shr63(hn);

      vp[b] = Ops::or_(hn_shift, Ops::not_(Ops::or_(d0, hp_shift)));
      vn[b] = Ops::and_(d0, hp_shift);

      if (b == W - 1U) {
        last_hp = hp;
        last_hn = hn;
      }
    }

    const Vec hp_bit = Ops::and_(last_hp, top_bit_last_mask);
    const Vec hn_bit = Ops::and_(last_hn, top_bit_last_mask);
    const Vec hp_eq_zero = Ops::cmpeq(hp_bit, zero_v);
    const Vec hn_eq_zero = Ops::cmpeq(hn_bit, zero_v);
    const Vec hp_set = Ops::andnot_(hp_eq_zero, one_v);
    const Vec hn_set = Ops::andnot_(hn_eq_zero, one_v);
    const Vec delta = Ops::and_(Ops::sub(hp_set, hn_set), eff_active);
    score = Ops::add(score, delta);

    if constexpr (HasCutoff) {
      // See single-word kernel: gate the bail check with active_v.
      const Vec exceeded = Ops::and_(Ops::gt_u64(score, threshold), active_v);
      done = Ops::or_(done, exceeded);
      threshold = Ops::sub(threshold, one_v);

      const Vec settled = Ops::or_(Ops::not_(active_v), done);
      const Vec live = Ops::andnot_(settled, batch_mask);
      if (Ops::is_zero(live)) {
        break;
      }
    }
  }

  alignas(64) std::uint64_t scores[lanes];
  if constexpr (HasCutoff) {
    alignas(64) std::uint64_t cutoff_plus_1_data[lanes];
    for (std::size_t l = 0; l < lanes; ++l) {
      cutoff_plus_1_data[l] = (l < batch_count) ? cutoffs[l] + 1U : 0;
    }
    const Vec cutoff_plus_1 = Ops::load_aligned(cutoff_plus_1_data);
    score = Ops::or_(
        Ops::and_(done, cutoff_plus_1),
        Ops::andnot_(done, score));
  }
  Ops::store_aligned(scores, score);
  for (std::size_t l = 0; l < batch_count; ++l) {
    out[l] = static_cast<Score>(scores[l]);
  }
}

// Re-export the shared byte-view helpers under the SIMD namespace so
// existing call sites keep compiling. The actual implementations live in
// byte_view.hpp now that both the singular dispatch and the batch
// kernel use them.
using ByteCompatKind = ::stride_align::byte_view::ByteCompatKind;
using ::stride_align::byte_view::classify;
inline void byte_view(
    PyObject* obj,
    ByteCompatKind kind,
    const std::uint8_t*& ptr,
    std::size_t& len) {
  ::stride_align::byte_view::view(obj, kind, ptr, len);
}

// Decide whether the SIMD batch is large enough to justify releasing the
// GIL. The PyEval_SaveThread / PyEval_RestoreThread pair costs roughly
// 1 µs round-trip on modern CPython; the user constraint is that this
// overhead must stay under 2% of total kernel time, so the kernel must
// take at least ~50 µs before the release pays for itself.
//
// Per-column kernel work is ~15 SIMD ops at ~1 ns each (single-word).
// Total kernel time ≈ q_len * count * 15 / lanes nanoseconds. Solving
// `total >= 50,000 ns` for `q_len * count` gives the per-lane threshold
// 3333 * lanes. Multi-word (W ≥ 2) only inflates the per-column cost,
// so the same threshold stays conservative.
template <typename Ops>
constexpr inline bool should_release_gil(std::size_t count, std::size_t q_len) {
  return count * q_len > 3000U * Ops::lanes;
}

// RAII wrapper that releases the GIL only if the work estimate justifies
// it. Movable but not copyable. Uses the raw Python C API rather than
// nanobind's gil_scoped_release so the "do nothing" path is a literal
// no-op (no heap allocation, no exception machinery).
class ConditionalGilRelease {
 public:
  explicit ConditionalGilRelease(bool release) noexcept {
    if (release) {
      saved_ = PyEval_SaveThread();
    }
  }
  ~ConditionalGilRelease() {
    if (saved_ != nullptr) {
      PyEval_RestoreThread(saved_);
    }
  }
  ConditionalGilRelease(const ConditionalGilRelease&) = delete;
  ConditionalGilRelease& operator=(const ConditionalGilRelease&) = delete;

 private:
  PyThreadState* saved_ = nullptr;
};

// Try to view every target as bytes or 1-byte unicode (either is
// byte-identical to a `const uint8_t*` buffer). The check is per-target;
// each target picks its own kind. Returns true on success with `ptrs` /
// `lens` populated; returns false on the first target that is neither
// (caller falls back to scalar dispatch).
inline bool view_all_byte_compat(
    PyObject* const* items,
    std::size_t count,
    std::vector<const std::uint8_t*>& ptrs,
    std::vector<std::size_t>& lens) {
  ptrs.resize(count);
  lens.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    const ByteCompatKind kind = classify(items[i]);
    if (kind == ByteCompatKind::None) {
      return false;
    }
    byte_view(items[i], kind, ptrs[i], lens[i]);
  }
  return true;
}

// Largest pattern length the multi-word SIMD batch kernel covers.
inline constexpr std::size_t kLevMaxSimdPatternLen = 256U;

// Raw SIMD batch entry: caller has byte-viewed the query and all
// targets and confirmed q_len in (0, 256]. Writes distances into
// `out[0..count)`. No Python interaction (no fallback for wider
// inputs — caller validates).
//
// `cutoffs_per_pair` may be:
//   * nullptr: no cutoff applied, kernel runs to completion.
//   * a length-`count` array: each pair gets its own cutoff. Pairs
//     whose distance provably exceeds their cutoff get the per-pair
//     sentinel `cutoffs_per_pair[i] + 1` instead of the true distance.
template <typename Ops>
inline void levenshtein_scores_simd_raw_per_pair(
    const std::uint8_t* q_ptr, std::size_t q_len,
    const std::uint8_t* const* t_ptrs,
    const std::size_t* t_lens,
    std::size_t count,
    const std::size_t* cutoffs_per_pair,
    Score* out) {
  if (count == 0U) {
    return;
  }

  constexpr std::size_t lanes = Ops::lanes;
  std::array<const std::uint8_t*, lanes> batch_ptrs{};
  std::array<std::size_t, lanes> batch_lens{};
  std::array<std::size_t, lanes> batch_cutoffs{};
  const bool has_cutoff = cutoffs_per_pair != nullptr;

  const std::span<const std::uint8_t> pattern_span(q_ptr, q_len);
  if (q_len <= 64U) {
    const auto peq = build_peq(pattern_span);
    for (std::size_t b = 0; b < count; b += lanes) {
      const std::size_t batch_count = std::min(lanes, count - b);
      for (std::size_t l = 0; l < batch_count; ++l) {
        batch_ptrs[l] = t_ptrs[b + l];
        batch_lens[l] = t_lens[b + l];
      }
      for (std::size_t l = batch_count; l < lanes; ++l) {
        batch_ptrs[l] = nullptr;
        batch_lens[l] = 0;
      }
      if (has_cutoff) {
        for (std::size_t l = 0; l < batch_count; ++l) {
          batch_cutoffs[l] = cutoffs_per_pair[b + l];
        }
        myers_batch_single_word<Ops, true>(
            peq.data(), batch_ptrs.data(), batch_lens.data(),
            batch_count, q_len, out + b, batch_cutoffs.data());
      } else {
        myers_batch_single_word<Ops, false>(
            peq.data(), batch_ptrs.data(), batch_lens.data(),
            batch_count, q_len, out + b);
      }
    }
    return;
  }

  const std::size_t W = (q_len + 63U) / 64U;
  const std::size_t last_bits = q_len - (W - 1U) * 64U;
  const auto peq = build_peq_multi(pattern_span, W);

  auto run_batch = [&](auto W_const) {
    constexpr std::size_t W_compile = decltype(W_const)::value;
    for (std::size_t b = 0; b < count; b += lanes) {
      const std::size_t batch_count = std::min(lanes, count - b);
      for (std::size_t l = 0; l < batch_count; ++l) {
        batch_ptrs[l] = t_ptrs[b + l];
        batch_lens[l] = t_lens[b + l];
      }
      for (std::size_t l = batch_count; l < lanes; ++l) {
        batch_ptrs[l] = nullptr;
        batch_lens[l] = 0;
      }
      if (has_cutoff) {
        for (std::size_t l = 0; l < batch_count; ++l) {
          batch_cutoffs[l] = cutoffs_per_pair[b + l];
        }
        myers_batch_multi_word<Ops, W_compile, true>(
            peq.data(), batch_ptrs.data(), batch_lens.data(),
            batch_count, q_len, last_bits, out + b, batch_cutoffs.data());
      } else {
        myers_batch_multi_word<Ops, W_compile, false>(
            peq.data(), batch_ptrs.data(), batch_lens.data(),
            batch_count, q_len, last_bits, out + b);
      }
    }
  };

  if (W == 2U) {
    run_batch(std::integral_constant<std::size_t, 2U>{});
  } else if (W == 3U) {
    run_batch(std::integral_constant<std::size_t, 3U>{});
  } else /* W == 4 */ {
    run_batch(std::integral_constant<std::size_t, 4U>{});
  }
}

// Scalar-cutoff entry kept for backward compat with single-cutoff
// callers (per-pair Levenshtein, 1-to-N batch through the scorers
// module). Forwards to the per-pair entry with a uniform cutoff
// array built on the stack.
template <typename Ops>
inline void levenshtein_scores_simd_raw(
    const std::uint8_t* q_ptr, std::size_t q_len,
    const std::uint8_t* const* t_ptrs,
    const std::size_t* t_lens,
    std::size_t count,
    std::size_t cutoff,
    Score* out) {
  if (cutoff == ::stride_align::levenshtein::kNoCutoff) {
    levenshtein_scores_simd_raw_per_pair<Ops>(
        q_ptr, q_len, t_ptrs, t_lens, count, nullptr, out);
    return;
  }
  std::vector<std::size_t> uniform(count, cutoff);
  levenshtein_scores_simd_raw_per_pair<Ops>(
      q_ptr, q_len, t_ptrs, t_lens, count, uniform.data(), out);
}

template <typename Ops>
inline std::vector<Score> levenshtein_scores_simd(
    nb::handle query,
    nb::handle targets,
    std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
  PyObject* fast_targets =
      PySequence_Fast(targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count = static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  // Alias of the raw entry's constant for backward compat with anyone
  // referencing this name in this header.
  constexpr std::size_t kMaxSimdPatternLen = kLevMaxSimdPatternLen;

  // Fast path: query is bytes or a 1-byte unicode string, each target
  // is independently bytes or 1-byte unicode, and pattern length is in
  // (0, kMaxSimdPatternLen]. We read CPython buffers directly with no
  // per-target copy or allocation. The PyObjects stay alive through
  // `owner` for the duration of this function.
  const ByteCompatKind q_kind = classify(query.ptr());
  if (q_kind != ByteCompatKind::None) {
    const std::uint8_t* q_ptr = nullptr;
    std::size_t q_len = 0;
    byte_view(query.ptr(), q_kind, q_ptr, q_len);
    if (q_len > 0 && q_len <= kMaxSimdPatternLen) {
      std::vector<const std::uint8_t*> ptrs;
      std::vector<std::size_t> lens;
      if (view_all_byte_compat(items, count, ptrs, lens)) {
        std::vector<Score> out(count);
        const ConditionalGilRelease gil_release(should_release_gil<Ops>(count, q_len));
        levenshtein_scores_simd_raw<Ops>(
            q_ptr, q_len, ptrs.data(), lens.data(), count, cutoff, out.data());
        return out;
      }
    }
  }

  // Slow path: anything that doesn't fit the byte-compatible fast path
  // falls back to scalar dispatch — wider unicode, sequence-of-object
  // inputs, long patterns via Hyyrö's multi-word Myers.
  std::vector<Score> out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    out.push_back(::stride_align::levenshtein::dispatch_score(
        query, nb::handle(items[i]), cutoff));
  }
  return out;
}

template <typename Ops>
inline std::vector<double> levenshtein_normalized_scores_simd(
    nb::handle query,
    nb::handle targets,
    std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
  // Compute raw distances first, then normalize. We need target lengths,
  // so reuse the SIMD-eligibility scan path: re-materialize cheaply.
  PyObject* fast_targets =
      PySequence_Fast(targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count = static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  auto raw = levenshtein_scores_simd<Ops>(query, targets, cutoff);

  const std::size_t q_len = static_cast<std::size_t>(PyObject_Length(query.ptr()));

  std::vector<double> out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t t_len = static_cast<std::size_t>(PyObject_Length(items[i]));
    out.push_back(::stride_align::levenshtein::normalize(
        static_cast<std::size_t>(raw[i]), q_len, t_len));
  }
  return out;
}

}  // namespace stride_align::levenshtein_simd
