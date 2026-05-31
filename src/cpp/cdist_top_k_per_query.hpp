#pragma once

// Adaptive top-k targets for one query at a time.
//
// Variant of cdist_top_k that yields the k highest-similarity targets
// PER QUERY, rather than the k highest pairs globally across the
// whole queries x targets matrix. The Python surface is a generator
// (so streaming over many queries is natural); this C++ side runs
// the inner per-query loop with two optimisations folded in:
//
//   1. ``max_normalized_similarity`` length bound. Before scoring,
//      each (query, target) pair gets the closed-form upper bound on
//      its normalised similarity from cdist_runtime.hpp. When the
//      heap is full and the bound can't beat the current worst-in-
//      heap (the smallest similarity we'd still admit), the pair is
//      skipped before the SIMD kernel runs.
//
//   2. The heap is a fixed-size min-heap of (score, PyObject*).
//      Scoring is dispatched per-scorer to the same backend kernels
//      the singular `levenshtein_normalized_score`, `jaro_similarity`
//      etc. bindings use.
//
// Only normalised-similarity scorers are supported (the same set
// cdist_top_k accepts). Distance scorers and matrix mode would need
// a different bound function and a flipped comparison; deferred.

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>

#include "cdist_runtime.hpp"  // max_normalized_similarity, Scorer
#include "stride_align/jaro.hpp"
#include "hamming_dispatch.hpp"
#include "indel_dispatch.hpp"
#include "jaro_dispatch.hpp"
#include "levenshtein_dispatch.hpp"
#include "true_damerau_dispatch.hpp"
#include "preprocess.hpp"  // detail::reject_str_or_bytes_targets, throw_*

namespace stride_align::cdist_top_k_per_query {

namespace nb = nanobind;
using ::stride_align::cdist_runtime::max_normalized_similarity;
using ::stride_align::topk::Scorer;

// Score one (query, target) pair under the chosen scorer. Returns a
// normalised similarity in [0, 1]; out-of-domain pairs (e.g. Hamming
// on unequal lengths) propagate the kernel's own ValueError.
inline double score_one_pair(
    Scorer scorer,
    nb::handle query, nb::handle target,
    double jw_prefix_weight, double jw_prefix_threshold,
    std::size_t jw_prefix_cap) {
  switch (scorer) {
    case Scorer::LevenshteinNormalized:
      return ::stride_align::levenshtein::dispatch_normalized_score(
          query, target, ::stride_align::levenshtein::kNoCutoff);
    case Scorer::DamerauLevenshteinNormalized:
      return ::stride_align::levenshtein::dispatch_osa_normalized_score(
          query, target);
    case Scorer::TrueDamerauLevenshteinNormalized:
      return ::stride_align::true_damerau::dispatch_normalized_score(
          query, target);
    case Scorer::HammingNormalized:
      return ::stride_align::hamming::dispatch_normalized_score(
          query, target);
    case Scorer::IndelNormalized:
      return ::stride_align::indel::dispatch_normalized_score(
          query, target);
    case Scorer::Jaro:
      return ::stride_align::jaro::dispatch_similarity(query, target);
    case Scorer::JaroWinkler:
      return ::stride_align::jaro::dispatch_winkler_similarity(
          query, target,
          jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
    default:
      ::stride_align::detail::throw_type_error(
          "cdist_top_k_per_query: scorer must be a normalised-similarity "
          "scorer (LEVENSHTEIN_NORMALIZED, DAMERAU_LEVENSHTEIN_NORMALIZED, "
          "HAMMING_NORMALIZED, INDEL_NORMALIZED, "
          "TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED, JARO, or JARO_WINKLER)");
  }
}

// Adaptive top-k for one query against many targets, with length-
// difference pruning that tightens as the heap fills.
//
// ``targets`` may be any sequence (list, tuple, or anything that
// ``PySequence_Fast`` materialises). Materialisation cost is amortised
// over the per-query loop; the Python wrapper passes the same list
// in for every query so the cost is paid once across the whole sweep.
inline std::vector<std::pair<double, PyObject*>>
top_k_one_query(
    nb::handle query,
    nb::handle targets,
    int scorer_int, std::size_t k,
    double jw_prefix_weight, double jw_prefix_threshold,
    std::size_t jw_prefix_cap,
    bool pruning = false) {
  const auto scorer = static_cast<Scorer>(scorer_int);

  // Validate up front so we fail fast on the first query rather than
  // throwing from a different switch case in score_one_pair.
  switch (scorer) {
    case Scorer::LevenshteinNormalized:
    case Scorer::DamerauLevenshteinNormalized:
    case Scorer::TrueDamerauLevenshteinNormalized:
    case Scorer::HammingNormalized:
    case Scorer::IndelNormalized:
    case Scorer::Jaro:
    case Scorer::JaroWinkler:
      break;
    default:
      ::stride_align::detail::throw_type_error(
          "cdist_top_k_per_query: scorer must be a normalised-similarity "
          "scorer (LEVENSHTEIN_NORMALIZED, DAMERAU_LEVENSHTEIN_NORMALIZED, "
          "HAMMING_NORMALIZED, INDEL_NORMALIZED, "
          "TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED, JARO, or JARO_WINKLER)");
  }

  ::stride_align::detail::reject_str_or_bytes_targets(targets.ptr());
  PyObject* fast_targets = PySequence_Fast(
      targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
  const std::size_t count = static_cast<std::size_t>(
      PySequence_Fast_GET_SIZE(fast_targets));

  // k == 0: nothing to score, but stay consistent with the
  // batch top_k semantics (empty result).
  if (k == 0U || count == 0U) {
    return {};
  }

  const std::size_t q_len = static_cast<std::size_t>(
      PyObject_Length(query.ptr()));

  using Entry = std::pair<double, PyObject*>;
  // Min-heap: heap.front() is the smallest score we'd still admit.
  // std::push_heap / pop_heap default to a max-heap; using
  // ``std::greater`` on the first field flips that to a min-heap.
  auto greater_by_score = [](const Entry& a, const Entry& b) {
    return a.first > b.first;
  };
  std::vector<Entry> heap;
  heap.reserve(k);

  // The "current worst score we'd still keep". Once heap.size() == k
  // this equals heap.front().first; before that it's -inf so every
  // pair qualifies. We track it separately to skip the front() read
  // in the hot path; updates happen exactly at heap-full and on each
  // replacement.
  double worst_in_heap = -1.0;  // any real score is >= 0.

  for (std::size_t i = 0; i < count; ++i) {
    PyObject* target_obj = items[i];
    const std::size_t t_len = static_cast<std::size_t>(
        PyObject_Length(target_obj));

    // Closed-form length bound. ``max_sim`` is the highest possible
    // normalised similarity for ``(q_len, t_len)`` under this
    // scorer. Two ways to skip a pair before scoring:
    //   1. max_sim == 0 — pair is unscorable (e.g. Hamming on
    //      unequal lengths) and can't enter the heap.
    //   2. heap is full AND max_sim <= worst_in_heap — pair can't
    //      displace anything in the current top-k.
    // Bound 1 is a correctness gate (max_normalized_similarity
    // returns 0.0 for Hamming on unequal lengths; without this guard
    // the kernel would raise ValueError on the first unequal pair
    // and abort the whole iteration) and always runs. Bound 2 is
    // the actual perf-time pruning; off by default — set
    // ``pruning=True`` to turn it on.
    const double max_sim = max_normalized_similarity(
        scorer, q_len, t_len, jw_prefix_weight, jw_prefix_cap);
    if (max_sim <= 0.0) {
      continue;
    }
    if (pruning && heap.size() >= k && max_sim <= worst_in_heap) {
      continue;
    }

    const double score = score_one_pair(
        scorer, query, nb::handle(target_obj),
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);

    if (heap.size() < k) {
      heap.emplace_back(score, target_obj);
      std::push_heap(heap.begin(), heap.end(), greater_by_score);
      if (heap.size() == k) {
        worst_in_heap = heap.front().first;
      }
    } else if (score > worst_in_heap) {
      std::pop_heap(heap.begin(), heap.end(), greater_by_score);
      heap.back() = {score, target_obj};
      std::push_heap(heap.begin(), heap.end(), greater_by_score);
      worst_in_heap = heap.front().first;
    }
  }

  // Final sort: best (highest similarity) first.
  std::sort(heap.begin(), heap.end(),
            [](const Entry& a, const Entry& b) { return a.first > b.first; });
  return heap;
}

}  // namespace stride_align::cdist_top_k_per_query
