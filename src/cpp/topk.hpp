#pragma once

// k-best selection over a score vector that runs entirely in C++.
//
// Why this exists: every algo in the library has a `*_scores` batch entry
// point that returns one score per target. The natural follow-up question
// is "give me the best k", which in Python would normally go through
// numpy.argpartition + ndarray indexing, or heapq.nlargest. Both involve
// either a numpy round-trip plus per-row Python tuple construction, or a
// Python-level comparison per element. Folding the selection into the
// same C++ call that produces the scores removes both costs and keeps
// the Python boundary crossing single-shot for "top-k over N targets".
//
// Selection strategy: std::nth_element partitions the top-k indices in
// O(N) average. We don't sort within the top-k partition — the caller
// asked for "the k best", not "the k best in order". Skipping the
// O(k log k) sort buys ~5-15% on typical k vs the full
// nth_element+sort. If the user wants them ordered they can sort the
// returned list themselves (which is what extract() in
// rapidfuzz.process effectively forces you to do).
//
// The helper is templated on the score element type (Score / int64_t for
// distance, double for normalized) so the same selection logic serves
// every algorithm. Direction is a runtime flag because every callsite
// knows statically whether higher or lower is better — passing it as a
// bool keeps the helper non-templated on direction (no code duplication)
// without paying for it at runtime (the branch resolves before the
// inner loop in every caller).

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>

namespace stride_align::topk {

namespace nb = nanobind;

// Algorithm identifier for the unified extract() entry point. The enum
// only exists C++-side for switch readability — the Python-facing
// `Scorer` is a Python IntEnum defined in stride_align/__init__.py and
// passed across the boundary as a plain int. Defining the Python enum
// once in pure Python avoids nanobind's per-module type-registration
// crash that comes from binding the same nb::enum_ in every backend
// shared library. Keep these integer values in sync with the
// `Scorer` IntEnum in __init__.py — they are part of the user-visible
// contract.
enum class Scorer : int {
  Levenshtein = 0,
  LevenshteinNormalized = 1,
  DamerauLevenshtein = 2,
  DamerauLevenshteinNormalized = 3,
  Hamming = 4,
  HammingNormalized = 5,
  Jaro = 6,
  JaroWinkler = 7,
  Indel = 8,
  IndelNormalized = 9,
  TrueDamerauLevenshtein = 10,
  TrueDamerauLevenshteinNormalized = 11,
  // 12-15 dispatch Python-side in ``cdist()``; the C++ kernel never
  // sees these enum values. They live here so the integer-id contract
  // with the Python ``Scorer`` IntEnum stays one-to-one.
  SmithWaterman = 12,
  SmithWatermanNormalized = 13,
  NeedlemanWunsch = 14,
  NeedlemanWunschNormalized = 15,
};

// Build a Python list of at most `k` (item, score, index) tuples from
// the parallel arrays `items` and `scores`. `higher_is_better=true`
// returns the k largest scores (use for similarity / normalized
// scores); `false` returns the k smallest (use for raw distance
// scores).
//
// Order within the returned list is unspecified — we maintain a heap,
// not a sorted list. The contract is "this is the top-k by score",
// not "this is the top-k by score, in score order". Callers who want
// them sorted can `sorted(result, key=lambda r: r[1])` — that
// O(k log k) cost is opt-in instead of forced on every call.
//
// Implementation: streaming selection with a k-element heap. Most
// inputs are read once, compared to the heap root, and discarded; only
// the (few) ones that beat the current heap root trigger an
// O(log k) heap update. Working set is k pairs (a few hundred bytes
// for typical k = 1..50) and stays resident in L1, where a partition-
// based approach has to read or write an N-element auxiliary buffer.
// For k << N this is the right shape; for k approaching N the
// partition form would win in operation count, but that's not the
// "top-k" workload this entry point is for.
template <typename T>
inline nb::list make_top_k(
    PyObject* const* items,
    const std::vector<T>& scores,
    std::size_t k,
    bool higher_is_better) {
  const std::size_t count = scores.size();
  const std::size_t take = std::min(k, count);

  nb::list result;
  if (take == 0U) {
    return result;
  }

  using Pair = std::pair<T, std::size_t>;

  // Cheap path when caller asks for everything: skip selection.
  if (take == count) {
    for (std::size_t i = 0; i < count; ++i) {
      result.append(nb::make_tuple(
          nb::borrow(nb::handle(items[i])),
          scores[i],
          i));
    }
    return result;
  }

  // Heap orientation is inverted from "what we keep". For the
  // k *smallest* (lower-is-better), we want a max-heap so the root is
  // the *largest* of the k-so-far — a smaller incoming score evicts
  // it. Mirror for the k largest.
  std::vector<Pair> heap;
  heap.reserve(take);

  const auto run = [&](auto worse_than_root) {
    for (std::size_t i = 0; i < take; ++i) {
      heap.emplace_back(scores[i], i);
    }
    std::make_heap(heap.begin(), heap.end(), worse_than_root);
    for (std::size_t i = take; i < count; ++i) {
      const T s = scores[i];
      if (worse_than_root(Pair{s, i}, heap.front())) {
        std::pop_heap(heap.begin(), heap.end(), worse_than_root);
        heap.back() = {s, i};
        std::push_heap(heap.begin(), heap.end(), worse_than_root);
      }
    }
  };

  if (higher_is_better) {
    // min-heap on score: root is the worst (smallest) of the top-k.
    run([](const Pair& a, const Pair& b) { return a.first > b.first; });
  } else {
    // max-heap on score: root is the worst (largest) of the bottom-k.
    run([](const Pair& a, const Pair& b) { return a.first < b.first; });
  }

  for (const auto& pair : heap) {
    result.append(nb::make_tuple(
        nb::borrow(nb::handle(items[pair.second])),
        pair.first,
        pair.second));
  }
  return result;
}

}  // namespace stride_align::topk
