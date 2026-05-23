#pragma once

// SIMD multi-target batch for OSA (restricted Damerau-Levenshtein).
// Same architecture as `levenshtein_simd.hpp`: one target per SIMD lane,
// 64-bit lane state, pattern length capped at 64.
//
// The inner-loop recurrence adds two extras over Levenshtein:
//   * pm_old: peq[T[j-1]] from the previous column, kept per lane
//   * d0_prev: the previous column's d0 vector, kept per lane
//   * trans = (((~d0_prev) & pm) << 1) & pm_old
// OR'd into d0 to allow the OSA transposition diagonal at no extra
// column cost. The `~d0_prev` gate is exactly what rapidfuzz uses
// (Hyyrö 2002) and is required for correctness under OSA's
// "each character involved in at most one edit" restriction.

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

namespace stride_align::osa_simd {

namespace nb = nanobind;

using ::stride_align::byte_view::ByteCompatKind;
inline ByteCompatKind classify(PyObject* obj) {
  return ::stride_align::byte_view::classify(obj);
}
inline void byte_view(
    PyObject* obj,
    ByteCompatKind kind,
    const std::uint8_t*& ptr,
    std::size_t& len) {
  ::stride_align::byte_view::view(obj, kind, ptr, len);
}

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

inline std::array<std::uint64_t, 256> build_peq(std::span<const std::uint8_t> pattern) {
  std::array<std::uint64_t, 256> peq{};
  const std::uint64_t one = 1U;
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    peq[pattern[i]] |= one << i;
  }
  return peq;
}

// SIMD batch single-word OSA. Mirrors myers_batch_single_word in
// levenshtein_simd.hpp; the inner loop has an extra `trans` term
// folded into d0 and one extra carried register (pm_old, d0_prev).
// `cutoffs` is a per-lane array of length `batch_count` when
// `HasCutoff=true`, otherwise unused. OSA's per-column score delta is in
// {-1, 0, +1} just like Myers' Levenshtein (the transposition mask
// folds into d0 before delta extraction), so the same
// `score > cutoff + remaining_chars` bail criterion applies. Bailed
// lanes are clamped to their per-pair `cutoffs[l] + 1` sentinel.
template <typename Ops, bool HasCutoff = false>
inline void osa_batch_single_word(
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
  Vec pm_old = Ops::zero();
  Vec d0_prev = Ops::zero();
  const Vec one_v = Ops::set1(1U);
  const Vec top_bit_mask = Ops::set1(std::uint64_t{1} << (m - 1U));
  const Vec zero_v = Ops::zero();

  alignas(64) std::uint64_t indices[lanes];
  alignas(64) std::uint64_t active[lanes];

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
    for (std::size_t l = 0; l < lanes; ++l) {
      if (l < batch_count && k < target_lengths[l]) {
        indices[l] = targets[l][k];
        active[l] = ~std::uint64_t{0};
      } else {
        indices[l] = 0;
        active[l] = 0;
      }
    }
    const Vec pm = Ops::gather64(peq, indices);
    const Vec active_v = Ops::load_aligned(active);
    Vec eff_active = active_v;
    if constexpr (HasCutoff) {
      eff_active = Ops::andnot_(done, active_v);
    }

    // Transposition mask (Hyyrö 2002):
    //   trans = (((~d0_prev) & pm) << 1) & pm_old
    // Gate it with active_v so inactive lanes contribute no transposition.
    const Vec not_d0_prev_and_pm = Ops::andnot_(d0_prev, pm);
    const Vec trans =
        Ops::and_(Ops::and_(Ops::shl1(not_d0_prev_and_pm), pm_old), active_v);

    // Standard Myers d0 = (((pm & vp) + vp) ^ vp) | pm | vn.
    const Vec pm_and_vp = Ops::and_(pm, vp);
    const Vec sum = Ops::add(pm_and_vp, vp);
    Vec d0 = Ops::or_(Ops::or_(Ops::xor_(sum, vp), pm), vn);
    d0 = Ops::or_(d0, trans);

    const Vec hp = Ops::or_(vn, Ops::not_(Ops::or_(d0, vp)));
    const Vec hn = Ops::and_(d0, vp);

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
    vn = Ops::and_(hp_shift, d0);

    // Save pm and d0 for the next column's transposition mask.
    // Inactive lanes feed garbage into pm_old/d0_prev, but their
    // score is already frozen via the active_v mask on `delta`, so
    // downstream state corruption doesn't affect their output.
    pm_old = pm;
    d0_prev = d0;

    if constexpr (HasCutoff) {
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

// Raw SIMD batch: caller has byte-viewed query and all targets and
// confirmed q_len in (0, 64] (single-word OSA path). Writes OSA
// distances into `out[0..count)`. No Python interaction.
//
// `cutoffs_per_pair` may be:
//   * nullptr: no cutoff applied, kernel runs to completion.
//   * a length-`count` array: per-pair distance cutoffs. Pairs whose
//     distance provably exceeds their cutoff return the sentinel
//     `cutoffs_per_pair[i] + 1`.
template <typename Ops>
inline void osa_scores_simd_raw_per_pair(
    const std::uint8_t* q_ptr, std::size_t q_len,
    const std::uint8_t* const* t_ptrs,
    const std::size_t* t_lens,
    std::size_t count,
    const std::size_t* cutoffs_per_pair,
    Score* out) {
  if (count == 0U) {
    return;
  }
  const auto peq = build_peq(std::span<const std::uint8_t>(q_ptr, q_len));

  constexpr std::size_t lanes = Ops::lanes;
  std::array<const std::uint8_t*, lanes> batch_ptrs{};
  std::array<std::size_t, lanes> batch_lens{};
  std::array<std::size_t, lanes> batch_cutoffs{};
  const bool has_cutoff = cutoffs_per_pair != nullptr;

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
      osa_batch_single_word<Ops, true>(
          peq.data(), batch_ptrs.data(), batch_lens.data(),
          batch_count, q_len, out + b, batch_cutoffs.data());
    } else {
      osa_batch_single_word<Ops, false>(
          peq.data(), batch_ptrs.data(), batch_lens.data(),
          batch_count, q_len, out + b);
    }
  }
}

// Backward-compat entry that takes no cutoff.
template <typename Ops>
inline void osa_scores_simd_raw(
    const std::uint8_t* q_ptr, std::size_t q_len,
    const std::uint8_t* const* t_ptrs,
    const std::size_t* t_lens,
    std::size_t count,
    Score* out) {
  osa_scores_simd_raw_per_pair<Ops>(
      q_ptr, q_len, t_ptrs, t_lens, count, nullptr, out);
}

template <typename Ops>
inline std::vector<Score> osa_scores_simd(
    nb::handle query,
    nb::handle targets) {
  PyObject* fast_targets =
      PySequence_Fast(targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count = static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  const ByteCompatKind q_kind = classify(query.ptr());
  if (q_kind != ByteCompatKind::None) {
    const std::uint8_t* q_ptr = nullptr;
    std::size_t q_len = 0;
    byte_view(query.ptr(), q_kind, q_ptr, q_len);
    if (q_len > 0 && q_len <= 64U) {
      std::vector<const std::uint8_t*> ptrs;
      std::vector<std::size_t> lens;
      if (view_all_byte_compat(items, count, ptrs, lens)) {
        std::vector<Score> out(count);
        const ::stride_align::levenshtein_simd::ConditionalGilRelease gil_release(
            ::stride_align::levenshtein_simd::should_release_gil<Ops>(count, q_len));
        osa_scores_simd_raw<Ops>(
            q_ptr, q_len, ptrs.data(), lens.data(), count, out.data());
        return out;
      }
    }
  }

  std::vector<Score> out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    out.push_back(::stride_align::levenshtein::dispatch_osa_score(
        query, nb::handle(items[i])));
  }
  return out;
}

template <typename Ops>
inline std::vector<double> osa_normalized_scores_simd(
    nb::handle query,
    nb::handle targets) {
  PyObject* fast_targets =
      PySequence_Fast(targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count = static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  auto raw = osa_scores_simd<Ops>(query, targets);
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

}  // namespace stride_align::osa_simd
