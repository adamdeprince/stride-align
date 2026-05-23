#pragma once

// Top-k variant of cdist: returns the k highest-scoring
// (query, target) pairs as an unsorted list. Differs from the
// streaming `cdist_above_threshold` in two ways:
//
//   1. Workers don't push every match to a queue. Each worker keeps
//      its own bounded-size top-k heap (a std::vector + manual
//      std::push_heap / std::pop_heap so we can read the underlying
//      storage at the end without consuming it). After all workers
//      finish, the main thread merges the per-thread heaps into a
//      single top-k heap. Result: no heap mutex contention during the
//      hot loop, O(num_threads * k) memory peak.
//
//   2. The heap stores `(score, q_idx, t_idx)` — list indices, not
//      Python object pointers. No INCREF/DECREF traffic during the
//      compute. We only resolve the index back to the original
//      string object at the very end, when materializing the result
//      list of 3-tuples.
//
// Optional `reject_duplicates`: when true and the computed score is
// exactly 1.0, compare the underlying byte buffers; if they match,
// skip the pair (and its symmetric mirror). Useful for "find near-
// duplicates excluding actual duplicates" workflows.
//
// tqdm + cpu_count + GIL release behave the same as `cdist`.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

#include <nanobind/nanobind.h>

#include "cdist_simd.hpp"
#include "cdist_threshold.hpp"  // snapshot helper, scorer_is_normalized
#include "stride_align/jaro.hpp"
#include "topk.hpp"

namespace stride_align::cdist_topk {

namespace nb = nanobind;
using Scorer = ::stride_align::topk::Scorer;

struct HeapItem {
  double score;
  std::uint32_t q_idx;
  std::uint32_t t_idx;
};

// Min-heap-on-score ordering for std::push_heap / std::pop_heap:
// `a < b` in the priority_queue sense means "a has lower priority";
// the *highest-priority* element ends up at the top via push_heap.
// We want the *smallest score* on top (so a new item only displaces
// it if the new item's score is bigger). That means "smallest score
// = highest priority", which translates to `a < b iff a.score > b.score`.
struct HeapCompare {
  bool operator()(const HeapItem& a, const HeapItem& b) const noexcept {
    return a.score > b.score;
  }
};

// Push an item into a vector-backed min-heap of size <= top_k, only
// keeping it if it beats the current minimum. The vector remains
// heap-ordered after each call.
inline void push_top_k(
    std::vector<HeapItem>& vec, std::size_t top_k, HeapItem item) noexcept {
  if (vec.size() < top_k) {
    vec.push_back(item);
    std::push_heap(vec.begin(), vec.end(), HeapCompare{});
  } else if (top_k > 0U && item.score > vec.front().score) {
    std::pop_heap(vec.begin(), vec.end(), HeapCompare{});
    vec.back() = item;
    std::push_heap(vec.begin(), vec.end(), HeapCompare{});
  }
}

template <typename Ops>
inline nb::object cdist_top_k_impl(
    nb::handle queries_handle,
    nb::handle targets_handle,
    int scorer_int,
    std::size_t top_k,
    nb::object tqdm_factory,
    std::size_t cpu_count,
    bool reject_duplicates,
    double jw_prefix_weight,
    double jw_prefix_threshold,
    std::size_t jw_prefix_cap) {
  const Scorer scorer = static_cast<Scorer>(scorer_int);
  if (!::stride_align::cdist_threshold::scorer_is_normalized(scorer)) {
    PyErr_SetString(
        PyExc_ValueError,
        "cdist_top_k requires a normalized (similarity) scorer that "
        "returns a float in [0, 1].");
    throw nb::python_error();
  }

  const bool symmetric = (queries_handle.ptr() == targets_handle.ptr());

  // === Phase 1: snapshot inputs under the GIL =====================
  nb::object q_tuple_owner;
  std::vector<PyObject*> q_objs;
  std::vector<const std::uint8_t*> q_ptrs;
  std::vector<std::size_t> q_lens;
  if (!::stride_align::cdist_threshold::snapshot(
          queries_handle.ptr(), q_tuple_owner, q_objs, q_ptrs, q_lens)) {
    PyErr_SetString(
        PyExc_NotImplementedError,
        "cdist_top_k currently requires byte-compatible inputs "
        "(bytes / 1-byte unicode). Wider unicode is not yet supported.");
    throw nb::python_error();
  }
  const std::size_t N = q_lens.size();

  nb::object t_tuple_owner;
  std::vector<PyObject*> t_objs_storage;
  std::vector<const std::uint8_t*> t_ptrs_storage;
  std::vector<std::size_t> t_lens_storage;
  PyObject* const* t_objs;
  const std::uint8_t* const* t_ptrs;
  const std::size_t* t_lens;
  std::size_t M;
  if (symmetric) {
    t_objs = q_objs.data();
    t_ptrs = q_ptrs.data();
    t_lens = q_lens.data();
    M = N;
  } else {
    if (!::stride_align::cdist_threshold::snapshot(
            targets_handle.ptr(), t_tuple_owner, t_objs_storage,
            t_ptrs_storage, t_lens_storage)) {
      PyErr_SetString(
          PyExc_NotImplementedError,
          "cdist_top_k currently requires byte-compatible inputs "
          "(bytes / 1-byte unicode). Wider unicode is not yet supported.");
      throw nb::python_error();
    }
    t_objs = t_objs_storage.data();
    t_ptrs = t_ptrs_storage.data();
    t_lens = t_lens_storage.data();
    M = t_lens_storage.size();
  }

  // Length caps and Hamming equal-length check, same as cdist.
  constexpr std::size_t cap = ::stride_align::cdist_runtime::kCdistMaxLen;
  for (std::size_t i = 0; i < N; ++i) {
    if (q_lens[i] > cap) {
      PyErr_Format(
          PyExc_NotImplementedError,
          "cdist_top_k: query %zu length %zu > SIMD cap %zu",
          i, q_lens[i], cap);
      throw nb::python_error();
    }
  }
  if (!symmetric) {
    for (std::size_t j = 0; j < M; ++j) {
      if (t_lens[j] > cap) {
        PyErr_Format(
            PyExc_NotImplementedError,
            "cdist_top_k: target %zu length %zu > SIMD cap %zu",
            j, t_lens[j], cap);
        throw nb::python_error();
      }
    }
  }
  if (scorer == Scorer::HammingNormalized && (N > 0U || M > 0U)) {
    const std::size_t ref = (N > 0U) ? q_lens[0] : t_lens[0];
    for (std::size_t i = 1; i < N; ++i) {
      if (q_lens[i] != ref) {
        PyErr_Format(
            PyExc_ValueError,
            "Hamming requires equal-length inputs (query 0 length %zu, "
            "query %zu length %zu)",
            ref, i, q_lens[i]);
        throw nb::python_error();
      }
    }
    if (!symmetric) {
      for (std::size_t j = 0; j < M; ++j) {
        if (t_lens[j] != ref) {
          PyErr_Format(
              PyExc_ValueError,
              "Hamming requires equal-length inputs (query 0 length %zu, "
              "target %zu length %zu)",
              ref, j, t_lens[j]);
          throw nb::python_error();
        }
      }
    }
  }

  // tqdm setup.
  nb::object pbar;
  if (!tqdm_factory.is_none()) {
    std::uint64_t total_cost = 0;
    if (symmetric) {
      for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1U; j < M; ++j) {
          total_cost += ::stride_align::cdist_runtime::pair_cost(
              q_lens[i], t_lens[j]);
        }
      }
    } else {
      for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
          total_cost += ::stride_align::cdist_runtime::pair_cost(
              q_lens[i], t_lens[j]);
        }
      }
    }
    pbar = tqdm_factory(nb::arg("total") = total_cost);
  }
  const bool have_tqdm = pbar.is_valid() && !pbar.is_none();

  // Process queries in descending-length order so the global heap-min
  // bound rises faster. Rationale: the eventual top-k pairs tend to
  // be close-length matches; processing the longest queries first
  // surfaces those high-scoring pairs early, raising the shared
  // prune threshold before the short-query rows run. Short-query
  // rows then prune more aggressively against tall targets.
  //
  // q_perm[i_iter] gives the original query index for iteration step
  // i_iter. HeapItem still stores ORIGINAL indices, so emission and
  // the symmetric mirror are unaffected. In symmetric mode the
  // target permutation aliases the query permutation; in the
  // asymmetric case targets stay in input order (sorting targets
  // doesn't affect the bound, only query order does).
  std::vector<std::size_t> q_perm(N);
  std::iota(q_perm.begin(), q_perm.end(), 0U);
  std::stable_sort(q_perm.begin(), q_perm.end(),
                   [q_lens_ptr = q_lens.data()](std::size_t a, std::size_t b) {
                     return q_lens_ptr[a] > q_lens_ptr[b];
                   });
  std::vector<std::size_t> t_perm_storage;
  const std::size_t* t_perm;
  if (symmetric) {
    t_perm = q_perm.data();
  } else {
    t_perm_storage.resize(M);
    std::iota(t_perm_storage.begin(), t_perm_storage.end(), 0U);
    t_perm = t_perm_storage.data();
  }

  // === Phase 2: parallel compute (GIL released) ==================
  const std::size_t num_threads =
      std::max<std::size_t>(1U, std::min(cpu_count, std::max<std::size_t>(1U, N)));

  std::vector<std::vector<HeapItem>> per_thread_heaps(num_threads);
  for (auto& h : per_thread_heaps) {
    h.reserve(top_k);
  }

  std::atomic<std::size_t> next_row{0};
  std::atomic<std::size_t> thread_id_counter{0};
  std::mutex done_mtx;
  std::condition_variable done_cv;
  std::deque<std::uint64_t> done_queue;

  // Shared lower bound on the eventual merged-heap k-th score.
  // Every worker reads it as the prune threshold for each row, and
  // publishes its own heap's min back to it (CAS loop, since
  // std::atomic<double> doesn't expose fetch_max). The bound is
  // monotonically non-decreasing, so the pruning gets stricter as
  // work proceeds. Correctness: each thread's heap min is a lower
  // bound on the final k-th best (the merged top-k contains at
  // least k items each >= that thread's min), so the global atomic
  // is safe to prune against from any thread.
  std::atomic<double> global_min_bound{
      -std::numeric_limits<double>::infinity()};

  auto publish_heap_min =
      [&global_min_bound](double own_min) noexcept {
    double cur = global_min_bound.load(std::memory_order_relaxed);
    while (own_min > cur) {
      if (global_min_bound.compare_exchange_weak(
              cur, own_min, std::memory_order_relaxed)) {
        return;
      }
      // cur was updated by compare_exchange_weak to the latest value.
    }
  };

  auto process_row = [&](std::size_t i_iter,
                         std::vector<HeapItem>& heap,
                         std::vector<double>& row,
                         std::vector<const std::uint8_t*>& cand_ptrs,
                         std::vector<std::size_t>& cand_lens,
                         std::vector<std::size_t>& cand_orig_j,
                         std::uint64_t& row_cost_out) {
    const std::size_t q_idx = q_perm[i_iter];
    const std::size_t j_start = symmetric ? i_iter + 1U : 0U;
    const std::size_t row_count = M - j_start;
    row_cost_out = 0;
    if (row_count == 0U) {
      return;
    }

    // Snapshot the pruning threshold ONCE per row: max of our own
    // heap min (if full) and the global lower bound. -inf if our
    // heap isn't full yet, because then any pair could still make
    // it in.
    double prune_threshold =
        global_min_bound.load(std::memory_order_relaxed);
    if (heap.size() >= top_k && top_k > 0U) {
      prune_threshold = std::max(prune_threshold, heap.front().score);
    }

    cand_ptrs.clear();
    cand_lens.clear();
    cand_orig_j.clear();
    std::size_t max_m = 0;
    for (std::size_t k = 0; k < row_count; ++k) {
      const std::size_t j_iter = j_start + k;
      const std::size_t t_idx = t_perm[j_iter];
      row_cost_out += ::stride_align::cdist_runtime::pair_cost(
          q_lens[q_idx], t_lens[t_idx]);
      const double max_sim =
          ::stride_align::cdist_runtime::max_normalized_similarity(
              scorer, q_lens[q_idx], t_lens[t_idx],
              jw_prefix_weight, jw_prefix_cap);
      if (max_sim <= prune_threshold) {
        continue;
      }
      cand_ptrs.push_back(t_ptrs[t_idx]);
      cand_lens.push_back(t_lens[t_idx]);
      cand_orig_j.push_back(t_idx);
      max_m = std::max(max_m, t_lens[t_idx]);
    }

    if (cand_ptrs.empty()) {
      return;
    }

    ::stride_align::cdist_simd::compute_row_double<Ops>(
        scorer, q_ptrs[q_idx], q_lens[q_idx],
        cand_ptrs.data(), cand_lens.data(),
        cand_ptrs.size(), max_m, row.data(),
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);

    for (std::size_t k = 0; k < cand_ptrs.size(); ++k) {
      const double score = row[k];
      const std::size_t j = cand_orig_j[k];

      if (reject_duplicates && score == 1.0) {
        if (q_lens[q_idx] == t_lens[j] && q_lens[q_idx] > 0U &&
            std::memcmp(q_ptrs[q_idx], t_ptrs[j], q_lens[q_idx]) == 0) {
          continue;
        }
      }

      push_top_k(heap, top_k,
                 HeapItem{score, static_cast<std::uint32_t>(q_idx),
                          static_cast<std::uint32_t>(j)});
      if (symmetric) {
        push_top_k(heap, top_k,
                   HeapItem{score, static_cast<std::uint32_t>(j),
                            static_cast<std::uint32_t>(q_idx)});
      }
    }

    // Publish our new heap min to the global bound (CAS loop). Only
    // worth doing if our heap is now full; otherwise the bound stays
    // -inf as far as we're concerned.
    if (heap.size() >= top_k && top_k > 0U) {
      publish_heap_min(heap.front().score);
    }
  };

  PyThreadState* saved_state = PyEval_SaveThread();

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  auto worker = [&]() {
    const std::size_t tid =
        thread_id_counter.fetch_add(1U, std::memory_order_relaxed);
    auto& heap = per_thread_heaps[tid];
    std::vector<double> row(M, 0.0);
    std::vector<const std::uint8_t*> cand_ptrs;
    std::vector<std::size_t> cand_lens;
    std::vector<std::size_t> cand_orig_j;
    cand_ptrs.reserve(M);
    cand_lens.reserve(M);
    cand_orig_j.reserve(M);

    while (true) {
      const std::size_t i_iter =
          next_row.fetch_add(1U, std::memory_order_relaxed);
      if (i_iter >= N) {
        return;
      }
      std::uint64_t row_cost = 0;
      process_row(i_iter, heap, row, cand_ptrs, cand_lens, cand_orig_j,
                  row_cost);
      if (have_tqdm) {
        std::lock_guard<std::mutex> lk(done_mtx);
        done_queue.push_back(row_cost);
        done_cv.notify_one();
      }
    }
  };

  for (std::size_t t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker);
  }

  // Main thread is the tqdm dispatcher; it sleeps on the cv with the
  // GIL released, reacquires only for the brief bar.update call.
  if (have_tqdm) {
    std::size_t rows_seen = 0;
    while (rows_seen < N) {
      std::uint64_t cost;
      {
        std::unique_lock<std::mutex> lk(done_mtx);
        done_cv.wait(lk, [&]() { return !done_queue.empty(); });
        cost = done_queue.front();
        done_queue.pop_front();
      }
      if (cost > 0U) {
        PyEval_RestoreThread(saved_state);
        pbar.attr("update")(cost);
        saved_state = PyEval_SaveThread();
      }
      ++rows_seen;
    }
  }

  for (auto& t : threads) {
    t.join();
  }

  // Merge per-thread heaps. Still no GIL — pure C++ work.
  std::vector<HeapItem> merged;
  merged.reserve(std::min(top_k, num_threads * top_k));
  for (auto& heap : per_thread_heaps) {
    for (const auto& item : heap) {
      push_top_k(merged, top_k, item);
    }
  }

  PyEval_RestoreThread(saved_state);

  // === Phase 3: wrap up (GIL held) ==============================
  if (have_tqdm) {
    pbar.attr("close")();
  }

  // Build the unsorted result list of (score, query, target).
  // Heap order is not score-sorted; we emit the vector contents
  // in-place (which is heap order, not sorted order), honoring the
  // contract that callers do their own sort if they need one.
  nb::list result;
  for (const auto& item : merged) {
    result.append(nb::make_tuple(
        item.score,
        nb::borrow(nb::handle(q_objs[item.q_idx])),
        nb::borrow(nb::handle(t_objs[item.t_idx]))));
  }
  return result;
}

}  // namespace stride_align::cdist_topk
