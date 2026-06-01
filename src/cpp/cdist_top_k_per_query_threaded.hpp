#pragma once

// Threaded variant of cdist_top_k_per_query.
//
// Mirrors cdist_topk's threading model: snapshot byte data under the
// GIL, release the GIL, spawn cpu_count workers each pulling the next
// query index, run that query's whole row through the same
// compute_row_double<Ops> SIMD kernel cdist_top_k uses, build a per-
// query top-k heap, store it indexed by the original query position.
// Re-acquire the GIL, materialise the result as a Python list of
// ``(query, [(score, target), ...])`` tuples in input order.
//
// Same input constraint as cdist_top_k: queries and targets must be
// byte-compatible (bytes or 1-byte unicode). Wide unicode falls back
// to the existing single-threaded per-pair path on the Python side —
// see the cdist_top_k_per_query wrapper in __init__.py.
//
// Length-difference pruning fires per pair via
// max_normalized_similarity: targets whose closed-form upper bound on
// similarity is <= 0 (e.g. Hamming with mismatched lengths) get
// dropped before the kernel runs. Per-query adaptive heap-min cutoff
// is not threaded here — it required sequential per-pair scoring and
// is the single-threaded path's job; the threaded path trades that
// for SIMD batch throughput.
//
// scorer / k / prefix_weight / prefix_threshold / prefix_cap behave
// identically to cdist_top_k.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>

#include "cdist_runtime.hpp"        // snapshot, max_normalized_similarity, Scorer
#include "cdist_simd.hpp"           // compute_row_double<Ops>
#include "cdist_threshold.hpp"      // scorer_is_normalized
#include "stride_align/jaro.hpp"
#include "preprocess.hpp"

namespace stride_align::cdist_top_k_per_query_threaded {

namespace nb = nanobind;
using ::stride_align::cdist_runtime::max_normalized_similarity;
using ::stride_align::cdist_runtime::snapshot;
using ::stride_align::topk::Scorer;

template <typename Ops>
inline nb::object cdist_top_k_per_query_threaded_impl(
    nb::handle queries_handle,
    nb::handle targets_handle,
    int scorer_int,
    std::size_t top_k,
    bool /*pruning*/,
    std::size_t cpu_count,
    double jw_prefix_weight,
    double jw_prefix_threshold,
    std::size_t jw_prefix_cap) {
  const Scorer scorer = static_cast<Scorer>(scorer_int);
  if (!::stride_align::cdist_threshold::scorer_is_normalized(scorer)) {
    PyErr_SetString(
        PyExc_ValueError,
        "cdist_top_k_per_query requires a normalized (similarity) scorer "
        "that returns a float in [0, 1].");
    throw nb::python_error();
  }

  // === Snapshot inputs under the GIL ============================
  nb::object q_tuple_owner;
  std::vector<PyObject*> q_objs;
  std::vector<const std::uint8_t*> q_ptrs;
  std::vector<std::size_t> q_lens;
  if (!snapshot(queries_handle.ptr(), q_tuple_owner, q_objs, q_ptrs, q_lens)) {
    PyErr_SetString(
        PyExc_NotImplementedError,
        "cdist_top_k_per_query threaded path requires byte-compatible "
        "inputs (bytes or 1-byte unicode). Wide unicode falls back to the "
        "single-threaded per-pair path on the Python side.");
    throw nb::python_error();
  }
  const std::size_t N = q_lens.size();

  nb::object t_tuple_owner;
  std::vector<PyObject*> t_objs;
  std::vector<const std::uint8_t*> t_ptrs;
  std::vector<std::size_t> t_lens;
  if (!snapshot(targets_handle.ptr(), t_tuple_owner, t_objs, t_ptrs, t_lens)) {
    PyErr_SetString(
        PyExc_NotImplementedError,
        "cdist_top_k_per_query threaded path requires byte-compatible "
        "inputs (bytes or 1-byte unicode). Wide unicode falls back to the "
        "single-threaded per-pair path on the Python side.");
    throw nb::python_error();
  }
  const std::size_t M = t_lens.size();

  // Length caps and Hamming equal-length check, same as cdist_top_k.
  constexpr std::size_t cap = ::stride_align::cdist_runtime::kCdistMaxLen;
  for (std::size_t i = 0; i < N; ++i) {
    if (q_lens[i] > cap) {
      PyErr_Format(
          PyExc_NotImplementedError,
          "cdist_top_k_per_query: query %zu length %zu > SIMD cap %zu",
          i, q_lens[i], cap);
      throw nb::python_error();
    }
  }
  for (std::size_t j = 0; j < M; ++j) {
    if (t_lens[j] > cap) {
      PyErr_Format(
          PyExc_NotImplementedError,
          "cdist_top_k_per_query: target %zu length %zu > SIMD cap %zu",
          j, t_lens[j], cap);
      throw nb::python_error();
    }
  }
  if (scorer == Scorer::HammingNormalized && (N > 0U && M > 0U)) {
    const std::size_t ref = q_lens[0];
    for (std::size_t i = 1; i < N; ++i) {
      if (q_lens[i] != ref) {
        PyErr_Format(
            PyExc_ValueError,
            "Hamming requires equal-length queries (query 0 length %zu, "
            "query %zu length %zu)",
            ref, i, q_lens[i]);
        throw nb::python_error();
      }
    }
    // Targets can be filtered by the length-bound prune.
  }

  // Per-query result buckets: heap-sorted descending by score.
  using Entry = std::pair<double, std::size_t>;  // (score, t_idx)
  std::vector<std::vector<Entry>> per_query(N);

  // Empty inputs: shortcut.
  if (N == 0U || M == 0U || top_k == 0U) {
    nb::list result;
    for (std::size_t i = 0; i < N; ++i) {
      result.append(nb::make_tuple(nb::handle(q_objs[i]), nb::list()));
    }
    return result;
  }

  const std::size_t num_threads =
      std::max<std::size_t>(1U, std::min(cpu_count, N));

  // === Parallel compute (GIL released) ==========================
  PyThreadState* saved_state = PyEval_SaveThread();

  std::atomic<std::size_t> next_q{0};
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  auto worker = [&]() {
    std::vector<double> row(M, 0.0);
    std::vector<const std::uint8_t*> cand_ptrs;
    std::vector<std::size_t> cand_lens;
    std::vector<std::size_t> cand_orig_j;
    cand_ptrs.reserve(M);
    cand_lens.reserve(M);
    cand_orig_j.reserve(M);

    auto greater_by_score = [](const Entry& a, const Entry& b) noexcept {
      return a.first > b.first;
    };

    while (true) {
      const std::size_t q_idx =
          next_q.fetch_add(1U, std::memory_order_relaxed);
      if (q_idx >= N) {
        return;
      }

      // Length-difference prune: drop unscorable pairs (max_sim == 0)
      // before they hit the SIMD batch. Hamming length-mismatch
      // targets land here.
      cand_ptrs.clear();
      cand_lens.clear();
      cand_orig_j.clear();
      std::size_t max_m = 0;
      for (std::size_t j = 0; j < M; ++j) {
        const double max_sim = max_normalized_similarity(
            scorer, q_lens[q_idx], t_lens[j],
            jw_prefix_weight, jw_prefix_cap);
        if (max_sim <= 0.0) {
          continue;
        }
        cand_ptrs.push_back(t_ptrs[j]);
        cand_lens.push_back(t_lens[j]);
        cand_orig_j.push_back(j);
        max_m = std::max(max_m, t_lens[j]);
      }

      if (cand_ptrs.empty()) {
        continue;
      }

      ::stride_align::cdist_simd::compute_row_double<Ops>(
          scorer, q_ptrs[q_idx], q_lens[q_idx],
          cand_ptrs.data(), cand_lens.data(),
          cand_ptrs.size(), max_m, row.data(),
          jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap,
          /*normalized_cutoff=*/0.0);

      // Pull top-k from the row into a per-query min-heap.
      std::vector<Entry>& heap = per_query[q_idx];
      heap.clear();
      heap.reserve(top_k);
      for (std::size_t k = 0; k < cand_ptrs.size(); ++k) {
        const double score = row[k];
        if (heap.size() < top_k) {
          heap.emplace_back(score, cand_orig_j[k]);
          std::push_heap(heap.begin(), heap.end(), greater_by_score);
        } else if (score > heap.front().first) {
          std::pop_heap(heap.begin(), heap.end(), greater_by_score);
          heap.back() = {score, cand_orig_j[k]};
          std::push_heap(heap.begin(), heap.end(), greater_by_score);
        }
      }
      // Sort descending so callers get best-first.
      std::sort(heap.begin(), heap.end(),
                [](const Entry& a, const Entry& b) noexcept {
                  return a.first > b.first;
                });
    }
  };

  for (std::size_t t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker);
  }
  for (auto& t : threads) {
    t.join();
  }

  PyEval_RestoreThread(saved_state);

  // === Build the Python result ===================================
  nb::list result;
  for (std::size_t q_idx = 0; q_idx < N; ++q_idx) {
    nb::list per;
    for (const auto& [score, t_idx] : per_query[q_idx]) {
      per.append(nb::make_tuple(score, nb::borrow(nb::handle(t_objs[t_idx]))));
    }
    result.append(nb::make_tuple(
        nb::borrow(nb::handle(q_objs[q_idx])), per));
  }
  return result;
}

}  // namespace stride_align::cdist_top_k_per_query_threaded
