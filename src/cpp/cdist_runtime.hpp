#pragma once

// Generic, non-SIMD framework shared by all cdist variants
// (`cdist`, `cdist_above_threshold`, `cdist_top_k`).
//
// Everything in this file is plain C++ / Python C API plumbing —
// snapshotting the input lists, byte-viewing each item, validating
// lengths, computing the cost-weighted tqdm total, building the
// per-row pair_cost. None of it is templated on a SIMD Ops bundle;
// the per-row SIMD work goes through `compute_row_int<Ops>` /
// `compute_row_double<Ops>` in `cdist_simd.hpp`, which the variants
// inject at the per-row callsite.
//
// Centralizing these primitives here eliminates the three-way
// duplication that previously lived in cdist_simd.hpp,
// cdist_threshold.hpp, and cdist_topk.hpp.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "topk.hpp"  // Scorer enum

namespace stride_align::cdist_runtime {

namespace nb = nanobind;
using Scorer = ::stride_align::topk::Scorer;

// Hard upper bound for the SIMD path — the Jaro multi-word kernel
// caps query and target lengths at 256 each. Above this the binding
// layer raises NotImplementedError.
inline constexpr std::size_t kCdistMaxLen = 256U;

inline bool scorer_returns_double(Scorer s) noexcept {
  switch (s) {
    case Scorer::Levenshtein:
    case Scorer::DamerauLevenshtein:
    case Scorer::Hamming:
    case Scorer::Indel:
    case Scorer::TrueDamerauLevenshtein:
      return false;
    case Scorer::LevenshteinNormalized:
    case Scorer::DamerauLevenshteinNormalized:
    case Scorer::HammingNormalized:
    case Scorer::Jaro:
    case Scorer::JaroWinkler:
    case Scorer::IndelNormalized:
    case Scorer::TrueDamerauLevenshteinNormalized:
      return true;
  }
  return true;
}

inline bool scorer_is_normalized(Scorer s) noexcept {
  return scorer_returns_double(s);
}

// 0 / 1.0 identity for the diagonal of a symmetric matrix when
// queries == targets and the metric admits a fixed self-similarity.
inline double diagonal_double(Scorer s) noexcept {
  switch (s) {
    case Scorer::LevenshteinNormalized:
    case Scorer::DamerauLevenshteinNormalized:
    case Scorer::HammingNormalized:
    case Scorer::Jaro:
    case Scorer::JaroWinkler:
    case Scorer::IndelNormalized:
    case Scorer::TrueDamerauLevenshteinNormalized:
      return 1.0;
    default:
      return 0.0;
  }
}

inline std::int64_t diagonal_int(Scorer /*s*/) noexcept {
  // All int-valued scorers in this library are distances, so
  // self-distance is 0.
  return 0;
}

// Cost model used for tqdm pacing. q_len * t_len gives a length-
// proportional unit; clamping to >= 1 so an empty pair still
// contributes one unit (the bar will still advance on degenerate
// inputs).
inline std::uint64_t pair_cost(std::size_t q_len, std::size_t t_len) noexcept {
  return static_cast<std::uint64_t>(
      std::max<std::size_t>(1U, q_len) * std::max<std::size_t>(1U, t_len));
}

// Tuple-snapshot a Python sequence into our state. Shallow: a new
// tuple holding new refs to the same string objects, so the byte
// pointers returned by byte_view stay valid for the snapshot's
// lifetime (strings are immutable). Captures PyObject*s for
// downstream tuple-construction.
inline bool snapshot(
    PyObject* sequence,
    nb::object& tuple_owner,
    std::vector<PyObject*>& objs,
    std::vector<const std::uint8_t*>& ptrs,
    std::vector<std::size_t>& lens) {
  namespace bv = ::stride_align::byte_view;
  PyObject* tup = PySequence_Tuple(sequence);
  if (tup == nullptr) {
    return false;
  }
  tuple_owner = nb::steal<nb::object>(tup);
  const auto count = static_cast<std::size_t>(PyTuple_GET_SIZE(tup));
  objs.resize(count);
  ptrs.resize(count);
  lens.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    PyObject* item = PyTuple_GET_ITEM(tup, i);
    objs[i] = item;
    const bv::ByteCompatKind kind = bv::classify(item);
    if (kind == bv::ByteCompatKind::None) {
      return false;
    }
    bv::view(item, kind, ptrs[i], lens[i]);
  }
  return true;
}

// Single-side variant for paths that don't need the PyObject* array
// (e.g. the matrix cdist that yields an ndarray, never a per-item
// tuple). Keeps the call sites tidy.
inline bool snapshot_no_objs(
    PyObject* sequence,
    nb::object& tuple_owner,
    std::vector<const std::uint8_t*>& ptrs,
    std::vector<std::size_t>& lens) {
  std::vector<PyObject*> discard;
  return snapshot(sequence, tuple_owner, discard, ptrs, lens);
}

// Raises NotImplementedError if any length exceeds the SIMD cap.
inline void validate_length_caps(
    const std::vector<std::size_t>& q_lens,
    const std::vector<std::size_t>& t_lens,
    bool symmetric) {
  for (std::size_t i = 0; i < q_lens.size(); ++i) {
    if (q_lens[i] > kCdistMaxLen) {
      PyErr_Format(
          PyExc_NotImplementedError,
          "cdist: query %zu has length %zu, above the SIMD cap of %zu",
          i, q_lens[i], kCdistMaxLen);
      throw nb::python_error();
    }
  }
  if (symmetric) {
    return;
  }
  for (std::size_t j = 0; j < t_lens.size(); ++j) {
    if (t_lens[j] > kCdistMaxLen) {
      PyErr_Format(
          PyExc_NotImplementedError,
          "cdist: target %zu has length %zu, above the SIMD cap of %zu",
          j, t_lens[j], kCdistMaxLen);
      throw nb::python_error();
    }
  }
}

// Hamming demands every string share one length. O(N + M): pick a
// reference length and verify every other.
inline void validate_hamming_lengths(
    Scorer scorer,
    const std::vector<std::size_t>& q_lens,
    const std::vector<std::size_t>& t_lens,
    bool symmetric) {
  if (scorer != Scorer::Hamming && scorer != Scorer::HammingNormalized) {
    return;
  }
  const std::size_t N = q_lens.size();
  const std::size_t M = t_lens.size();
  if (N == 0U && M == 0U) {
    return;
  }
  const std::size_t ref = (N > 0U) ? q_lens[0] : t_lens[0];
  for (std::size_t i = 1; i < N; ++i) {
    if (q_lens[i] != ref) {
      PyErr_Format(
          PyExc_ValueError,
          "Hamming requires equal-length inputs (query 0 has length %zu, "
          "query %zu has length %zu)",
          ref, i, q_lens[i]);
      throw nb::python_error();
    }
  }
  if (!symmetric) {
    for (std::size_t j = 0; j < M; ++j) {
      if (t_lens[j] != ref) {
        PyErr_Format(
            PyExc_ValueError,
            "Hamming requires equal-length inputs (query 0 has length %zu, "
            "target %zu has length %zu)",
            ref, j, t_lens[j]);
        throw nb::python_error();
      }
    }
  }
}

// Closed-form upper bound on the normalized similarity a scorer can
// possibly produce for two strings of the given lengths. Used by
// length-difference pruning in cdist_above_threshold and
// cdist_top_k: any pair where this bound is below the active
// threshold can't reach it and we skip the SIMD work entirely.
//
// All bounds are tight (achievable when q == t for any length) and
// safe (never exceed the value the actual kernel could return),
// which keeps pruning correct.
inline double max_normalized_similarity(
    Scorer scorer,
    std::size_t q_len, std::size_t t_len,
    double jw_prefix_weight,
    std::size_t jw_prefix_cap) noexcept {
  switch (scorer) {
    case Scorer::LevenshteinNormalized:
    case Scorer::DamerauLevenshteinNormalized:
    case Scorer::TrueDamerauLevenshteinNormalized: {
      // Distance d >= |q - t| (one edit per length difference).
      // Normalized sim = 1 - d / max(q, t)
      //               <= 1 - |q-t|/max  =  min/max.
      // Holds for OSA, true DL, and Levenshtein equally — none of
      // them can do better than one edit per length difference.
      if (q_len == 0U && t_len == 0U) {
        return 1.0;
      }
      const std::size_t mx = std::max(q_len, t_len);
      const std::size_t mn = std::min(q_len, t_len);
      return static_cast<double>(mn) / static_cast<double>(mx);
    }
    case Scorer::HammingNormalized:
      // Equal-length contract is validated upstream; pruning is
      // never useful here (every survived pair has equal length and
      // could reach 1.0).
      return q_len == t_len ? 1.0 : 0.0;
    case Scorer::Jaro: {
      // m_count <= min(q, t); max Jaro = (min/q + min/t + 1) / 3
      // = (2 + min(q,t)/max(q,t)) / 3 for any nonzero pair.
      if (q_len == 0U && t_len == 0U) {
        return 1.0;
      }
      if (q_len == 0U || t_len == 0U) {
        return 0.0;
      }
      const std::size_t mx = std::max(q_len, t_len);
      const std::size_t mn = std::min(q_len, t_len);
      return (2.0 + static_cast<double>(mn) / static_cast<double>(mx))
             / 3.0;
    }
    case Scorer::JaroWinkler: {
      // JW = jaro + L * pw * (1 - jaro). Both terms are monotone in
      // jaro, so the upper bound substitutes max_jaro and the maximum
      // possible L. We assume the prefix bonus is applied (i.e. that
      // jaro might cross the threshold for this pair); when jaro is
      // below the threshold the bound is loose but still valid.
      const double max_jaro = max_normalized_similarity(
          Scorer::Jaro, q_len, t_len, 0.0, 0U);
      const std::size_t L_max =
          std::min(jw_prefix_cap, std::min(q_len, t_len));
      return max_jaro
             + static_cast<double>(L_max) * jw_prefix_weight
                   * (1.0 - max_jaro);
    }
    case Scorer::IndelNormalized: {
      // Indel = q + t - 2*LCS, sim = 1 - indel/(q+t) = 2*LCS/(q+t).
      // Max LCS = min(q, t), so max sim = 2*min(q,t)/(q+t).
      if (q_len == 0U && t_len == 0U) {
        return 1.0;
      }
      const std::size_t total = q_len + t_len;
      const std::size_t mn = std::min(q_len, t_len);
      return 2.0 * static_cast<double>(mn) /
             static_cast<double>(total);
    }
    default:
      // Distance scorers and unknown values: no useful bound, so
      // never prune.
      return 1.0;
  }
}

// Convert a normalized-similarity threshold T in [0, 1] to the integer
// Levenshtein / OSA distance cutoff such that:
//   * Any pair with sim >= T has distance <= cutoff (the kernel must not
//     bail and miss a valid result).
//   * The kernel's bail sentinel `cutoff + 1` maps to a normalized
//     similarity strictly less than T (so the downstream filter rejects
//     bailed pairs correctly).
//
// The +1e-9 nudge guards against FP-floor returning X - 1 when
// (1 - T) * max_len is mathematically exactly the integer X but lands
// as X - epsilon in floating point (e.g. T = 0.6, max_len = 10).
//
// Returns max_len when T <= 0 (no effective bail) and 0 when T >= 1
// (only exact matches survive). Both extremes are safe per-pair cutoffs;
// callers don't need to special-case them.
inline std::size_t lev_distance_cutoff_for_normalized_threshold(
    double threshold, std::size_t q_len, std::size_t t_len) noexcept {
  const std::size_t max_len = std::max(q_len, t_len);
  if (max_len == 0U) {
    return 0U;
  }
  if (threshold <= 0.0) {
    return max_len;
  }
  if (threshold >= 1.0) {
    return 0U;
  }
  const double dmax =
      (1.0 - threshold) * static_cast<double>(max_len);
  return static_cast<std::size_t>(std::floor(dmax + 1e-9));
}

// Total tqdm work in length-product units. Symmetric mode counts
// only the strict upper triangle to match the actual compute.
inline std::uint64_t total_cost(
    const std::vector<std::size_t>& q_lens,
    const std::vector<std::size_t>& t_lens,
    bool symmetric) {
  const std::size_t N = q_lens.size();
  const std::size_t M = t_lens.size();
  std::uint64_t cost = 0;
  if (symmetric) {
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = i + 1U; j < M; ++j) {
        cost += pair_cost(q_lens[i], t_lens[j]);
      }
    }
  } else {
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j < M; ++j) {
        cost += pair_cost(q_lens[i], t_lens[j]);
      }
    }
  }
  return cost;
}

}  // namespace stride_align::cdist_runtime
