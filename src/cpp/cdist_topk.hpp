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
#include <mutex>
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

  auto process_row = [&](std::size_t i,
                         std::vector<HeapItem>& heap,
                         std::vector<double>& row,
                         std::uint64_t& row_cost_out) {
    const std::size_t j_start = symmetric ? i + 1U : 0U;
    const std::size_t row_count = M - j_start;
    row_cost_out = 0;
    if (row_count == 0U) {
      return;
    }
    std::size_t max_m = 0;
    for (std::size_t k = 0; k < row_count; ++k) {
      max_m = std::max(max_m, t_lens[j_start + k]);
    }
    ::stride_align::cdist_simd::compute_row_double<Ops>(
        scorer, q_ptrs[i], q_lens[i],
        t_ptrs + j_start, t_lens + j_start,
        row_count, max_m, row.data(),
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);

    for (std::size_t k = 0; k < row_count; ++k) {
      const double score = row[k];
      const std::size_t j = j_start + k;
      row_cost_out += ::stride_align::cdist_runtime::pair_cost(
          q_lens[i], t_lens[j]);

      // reject_duplicates only triggers when the metric thinks the
      // strings are identical (score 1.0). At any other score we
      // skip the byte-compare entirely.
      if (reject_duplicates && score == 1.0) {
        if (q_lens[i] == t_lens[j] && q_lens[i] > 0U &&
            std::memcmp(q_ptrs[i], t_ptrs[j], q_lens[i]) == 0) {
          continue;  // forward pair is a content-duplicate
        }
      }

      push_top_k(heap, top_k,
                 HeapItem{score, static_cast<std::uint32_t>(i),
                          static_cast<std::uint32_t>(j)});
      if (symmetric) {
        // The mirror pair (j, i) — same byte contents in symmetric
        // mode (queries IS targets), so the duplicate check above
        // also covers the mirror. If we got here, the mirror is
        // safe to push.
        push_top_k(heap, top_k,
                   HeapItem{score, static_cast<std::uint32_t>(j),
                            static_cast<std::uint32_t>(i)});
      }
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
    while (true) {
      const std::size_t i = next_row.fetch_add(1U, std::memory_order_relaxed);
      if (i >= N) {
        return;
      }
      std::uint64_t row_cost = 0;
      process_row(i, heap, row, row_cost);
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
