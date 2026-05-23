#pragma once

// Streaming cdist for normalized-similarity scorers: yields
// `(score, query, target)` tuples for every pair whose normalized
// similarity is at least a caller-supplied threshold. The full
// O(N x M) matrix is never materialized — workers compute one
// query row at a time, scan it against the threshold, and feed the
// matching results into a bounded queue. The Python iterator drains
// the queue lazily, so steady-state memory is O(N + M + queue
// capacity).
//
// Threading: same shape as cdist — workers compute, the main thread
// (the one iterating the returned object) is the only one that
// touches the tqdm bar and constructs Python tuples for results. A
// single bounded MPMC queue carries two kinds of events: matched
// results and per-row "done" markers that the main thread converts
// to tqdm updates.
//
// Blocking: condition variables on both directions of the queue
// (workers block when full, main thread blocks when empty). No spin
// loops anywhere.
//
// Lifetime: the iterator owns the snapshot tuples and the worker
// threads. Its destructor flips a stop flag, drains the queue, and
// joins the workers — safe to abandon mid-iteration (e.g. an early
// `break` in a for-loop).

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "cdist_runtime.hpp"
#include "cdist_simd.hpp"
#include "stride_align/jaro.hpp"
#include "topk.hpp"

namespace stride_align::cdist_threshold {

namespace nb = nanobind;
using Scorer = ::stride_align::topk::Scorer;

// Re-export through the threshold namespace so existing callers don't
// have to rewrite — definition lives in cdist_runtime.
using ::stride_align::cdist_runtime::scorer_is_normalized;

// A single event in the worker→main queue. Either a matched pair or
// a row-completion marker that carries the cost for tqdm.
struct Event {
  enum class Kind : std::uint8_t { Result, RowDone };
  Kind kind;
  // For Result.
  double score = 0.0;
  std::size_t q_idx = 0;
  std::size_t t_idx = 0;
  // For RowDone.
  std::uint64_t cost = 0;
};

// Bounded queue with two condition variables. Workers push() blocks
// when full; consumer pop() blocks when empty. close() (called on
// iterator destruction) wakes every waiter so they can drain and
// exit.
class EventQueue {
 public:
  explicit EventQueue(std::size_t capacity) : capacity_{capacity} {}

  // Returns false if the queue was closed before the push could be
  // recorded (caller's worker should exit promptly in that case).
  bool push(const Event& e) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_not_full_.wait(lk, [&]() {
      return items_.size() < capacity_ || closed_;
    });
    if (closed_) {
      return false;
    }
    items_.push_back(e);
    cv_not_empty_.notify_one();
    return true;
  }

  // Pops the next event. Blocks until one is available. Returns false
  // if the queue is empty AND all workers have signalled completion;
  // the caller (Python __next__) translates that into StopIteration.
  bool pop(Event& out, const std::atomic<bool>& workers_done) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_not_empty_.wait(lk, [&]() {
      return !items_.empty() || workers_done.load(std::memory_order_acquire);
    });
    if (items_.empty()) {
      return false;
    }
    out = items_.front();
    items_.pop_front();
    cv_not_full_.notify_one();
    return true;
  }

  // Wake every waiter and refuse further pushes. Existing items are
  // preserved — the consumer can still drain them. (We rely on the
  // workers_done atomic to signal "drain and stop"; close() is for
  // the abandon path: iterator destroyed, no one is reading any
  // more.)
  void close() {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      closed_ = true;
    }
    cv_not_full_.notify_all();
    cv_not_empty_.notify_all();
  }

  void notify_workers_done() {
    // Just kick the consumer; workers_done atomic does the actual
    // signalling.
    cv_not_empty_.notify_all();
  }

 private:
  std::mutex mtx_;
  std::condition_variable cv_not_empty_;
  std::condition_variable cv_not_full_;
  std::deque<Event> items_;
  std::size_t capacity_;
  bool closed_ = false;
};

// Captures every Ops-dependent operation as a callback so the
// iterator class itself stays free of template parameters (only
// instantiated once per binding module, not per backend).
struct RowCompute {
  // Compute scores for query i against targets[j_start..j_start+count)
  // into `out[0..count)`. Pure compute, no Python touch (caller has
  // released the GIL).
  // Compute scores for the candidate targets (their byte pointers
  // and lengths) for a given query index. The candidate set may be a
  // dense slice of the targets (when no pruning kicks in) or a
  // scattered subset built from a length-difference prune.
  std::function<void(
      std::size_t i,
      const std::uint8_t* const* cand_ptrs,
      const std::size_t* cand_lens,
      std::size_t count,
      std::size_t max_m,
      double* out)>
      run;
};

// Returns the iterator's state to the Python class via shared_ptr so
// the lambda captures don't outlive the holder.
struct State {
  // Tuple snapshots (keep the strings alive for the iterator's
  // lifetime).
  nb::object q_tuple_owner;
  nb::object t_tuple_owner;
  // Borrowed PyObject* arrays — refs held by the tuple owners.
  std::vector<PyObject*> q_objs;
  std::vector<PyObject*> t_objs;
  std::vector<const std::uint8_t*> q_ptrs;
  std::vector<const std::uint8_t*> t_ptrs;
  std::vector<std::size_t> q_lens;
  std::vector<std::size_t> t_lens;
  std::size_t N = 0;
  std::size_t M = 0;
  bool symmetric = false;
  double threshold = 0.0;
  // For length-difference pruning.
  Scorer scorer{Scorer::Jaro};
  double jw_prefix_weight = 0.1;
  std::size_t jw_prefix_cap = 4;

  // Worker dispatch.
  std::atomic<std::size_t> next_row{0};
  std::atomic<std::size_t> workers_active{0};
  std::atomic<bool> workers_done{false};
  std::atomic<bool> stop_requested{false};
  std::vector<std::thread> workers;

  // Event queue (worker → main thread).
  std::unique_ptr<EventQueue> queue;

  // Optional tqdm bar (main thread accesses only).
  nb::object pbar;
  bool have_tqdm = false;
};

// Cost model: same length-product proxy used by cdist for pacing
// (lives in cdist_runtime.hpp).
using ::stride_align::cdist_runtime::pair_cost;

// Python-facing iterator. nb::class_-registered; __iter__ returns
// self, __next__ pops the next Result event (skipping any RowDone
// events it sees along the way, dispatching them to tqdm).
//
// Move-only on purpose: the destructor stops + joins workers, so any
// stray copy whose state pointer is still alive (e.g. a temporary
// inside an nb::cast call) would prematurely kill the workers. We
// guarantee a single live owner by deleting the copy operations; the
// move ctor transfers state_ and leaves the moved-from instance with
// an empty state_, which its destructor will see as nullptr and skip.
class ThresholdIterator {
 public:
  explicit ThresholdIterator(std::shared_ptr<State> state)
      : state_{std::move(state)} {}

  ThresholdIterator(const ThresholdIterator&) = delete;
  ThresholdIterator& operator=(const ThresholdIterator&) = delete;
  ThresholdIterator(ThresholdIterator&&) = default;
  ThresholdIterator& operator=(ThresholdIterator&&) = default;

  ~ThresholdIterator() {
    // Tell workers to stop, drain via close(), join.
    if (state_) {
      state_->stop_requested.store(true, std::memory_order_release);
      if (state_->queue) {
        state_->queue->close();
      }
      for (auto& t : state_->workers) {
        if (t.joinable()) {
          t.join();
        }
      }
      // Now safe to drop tqdm + tuples (GIL is held — destructor runs
      // from the main thread under Python's hold).
      if (state_->have_tqdm) {
        try {
          state_->pbar.attr("close")();
        } catch (...) {
          // best-effort close
        }
      }
    }
  }

  nb::object iter_self(nb::handle self) {
    return nb::borrow<nb::object>(self);
  }

  nb::object next() {
    while (true) {
      Event ev;
      bool got;
      {
        // The blocking pop releases the GIL so other Python threads
        // can run while we wait on an empty queue.
        nb::gil_scoped_release release_gil;
        got = state_->queue->pop(ev, state_->workers_done);
      }
      if (!got) {
        // Workers done + queue empty: end of stream.
        //
        // Avoid both `throw nb::stop_iteration()` AND
        // `throw nb::python_error()`: on macOS the per-module
        // visibility settings of the backend `.so` defeat the
        // cross-DSO RTTI lookup that nanobind's catch handlers
        // depend on, and the exception ends up routed through the
        // generic `std::exception` translator → bare RuntimeError.
        //
        // Setting the Python error and returning a NULL PyObject
        // bypasses C++ exception machinery entirely. The
        // vectorcall wrapper sees PyErr is set and propagates it
        // to the caller — Python recognizes the StopIteration and
        // ends the for-loop cleanly. Works identically on Linux
        // and macOS.
        PyErr_SetNone(PyExc_StopIteration);
        return nb::steal<nb::object>(nullptr);
      }
      if (ev.kind == Event::Kind::RowDone) {
        if (state_->have_tqdm && ev.cost > 0U) {
          state_->pbar.attr("update")(ev.cost);
        }
        continue;
      }
      // Result: build (score, query, target) tuple. The strings are
      // held alive by our tuple snapshots — borrow refs and let the
      // returned Python tuple INCREF them.
      return nb::make_tuple(
          ev.score,
          nb::borrow(nb::handle(state_->q_objs[ev.q_idx])),
          nb::borrow(nb::handle(state_->t_objs[ev.t_idx])));
    }
  }

 private:
  std::shared_ptr<State> state_;
};

// Spawn workers. Each worker grabs rows via the atomic counter, runs
// the per-backend row kernel via `compute.run`, scans the row for
// matches, and pushes results onto the queue.
inline void spawn_workers(
    std::shared_ptr<State> state,
    RowCompute compute,
    std::size_t cpu_count) {
  const std::size_t N = state->N;
  const std::size_t M = state->M;
  const bool symmetric = state->symmetric;
  const double threshold = state->threshold;

  const std::size_t num_threads =
      std::max<std::size_t>(1U, std::min(cpu_count, N));
  state->workers_active.store(num_threads, std::memory_order_release);

  for (std::size_t t = 0; t < num_threads; ++t) {
    state->workers.emplace_back([state, compute, symmetric, threshold, N, M]() {
      std::vector<double> row(M, 0.0);
      // Per-worker scratch for the candidate sub-list after length-
      // difference pruning. Reused across rows so we amortize the
      // initial allocation.
      std::vector<const std::uint8_t*> cand_ptrs;
      std::vector<std::size_t> cand_lens;
      std::vector<std::size_t> cand_orig_j;
      cand_ptrs.reserve(M);
      cand_lens.reserve(M);
      cand_orig_j.reserve(M);

      while (true) {
        if (state->stop_requested.load(std::memory_order_acquire)) {
          break;
        }
        const std::size_t i =
            state->next_row.fetch_add(1, std::memory_order_relaxed);
        if (i >= N) {
          break;
        }

        const std::size_t j_start = symmetric ? i + 1U : 0U;
        const std::size_t row_count = M - j_start;
        std::uint64_t cost = 0;
        if (row_count > 0U) {
          // Length-difference pruning: build the candidate sub-list
          // of targets whose max possible normalized similarity is
          // at least `threshold`. The cost contribution to tqdm
          // covers every target in the row (pruned or not) so the
          // bar tracks the work the user asked us to consider, not
          // the work we ended up actually doing.
          cand_ptrs.clear();
          cand_lens.clear();
          cand_orig_j.clear();
          std::size_t max_m = 0;
          for (std::size_t k = 0; k < row_count; ++k) {
            const std::size_t j = j_start + k;
            cost += pair_cost(state->q_lens[i], state->t_lens[j]);
            const double max_sim =
                ::stride_align::cdist_runtime::max_normalized_similarity(
                    state->scorer, state->q_lens[i], state->t_lens[j],
                    state->jw_prefix_weight, state->jw_prefix_cap);
            if (max_sim < threshold) {
              continue;
            }
            cand_ptrs.push_back(state->t_ptrs[j]);
            cand_lens.push_back(state->t_lens[j]);
            cand_orig_j.push_back(j);
            max_m = std::max(max_m, state->t_lens[j]);
          }

          if (!cand_ptrs.empty()) {
            compute.run(
                i, cand_ptrs.data(), cand_lens.data(),
                cand_ptrs.size(), max_m, row.data());

            // Scan + push matches. The candidate vector's k-th entry
            // corresponds to original target index cand_orig_j[k].
            for (std::size_t k = 0; k < cand_ptrs.size(); ++k) {
              if (row[k] >= threshold) {
                const std::size_t j = cand_orig_j[k];
                Event ev;
                ev.kind = Event::Kind::Result;
                ev.score = row[k];
                ev.q_idx = i;
                ev.t_idx = j;
                if (!state->queue->push(ev)) {
                  goto worker_exit;  // queue closed: iterator abandoned
                }
                if (symmetric) {
                  Event mirror;
                  mirror.kind = Event::Kind::Result;
                  mirror.score = row[k];
                  mirror.q_idx = j;
                  mirror.t_idx = i;
                  if (!state->queue->push(mirror)) {
                    goto worker_exit;
                  }
                }
              }
            }
          }
        }
        if (state->have_tqdm) {
          Event done;
          done.kind = Event::Kind::RowDone;
          done.cost = cost;
          if (!state->queue->push(done)) {
            goto worker_exit;
          }
        }
      }
    worker_exit:
      if (state->workers_active.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        // Last worker out. Signal end-of-stream to the consumer.
        state->workers_done.store(true, std::memory_order_release);
        state->queue->notify_workers_done();
      }
    });
  }
}

// Snapshot helper lives in cdist_runtime.hpp; re-exported here so
// the existing namespace-qualified call sites keep working.
using ::stride_align::cdist_runtime::snapshot;

template <typename Ops>
inline nb::object cdist_threshold_impl(
    nb::handle queries_handle,
    nb::handle targets_handle,
    int scorer_int,
    double threshold,
    nb::object tqdm_factory,
    std::size_t cpu_count,
    double jw_prefix_weight,
    double jw_prefix_threshold,
    std::size_t jw_prefix_cap) {
  const Scorer scorer = static_cast<Scorer>(scorer_int);
  if (!scorer_is_normalized(scorer)) {
    PyErr_SetString(
        PyExc_ValueError,
        "cdist_above_threshold requires a normalized (similarity) "
        "scorer that returns a float in [0, 1].");
    throw nb::python_error();
  }
  if (!(threshold >= 0.0 && threshold <= 1.0)) {
    PyErr_SetString(
        PyExc_ValueError,
        "threshold must be a float in [0, 1].");
    throw nb::python_error();
  }

  auto state = std::make_shared<State>();
  state->threshold = threshold;
  state->symmetric = (queries_handle.ptr() == targets_handle.ptr());

  if (!snapshot(queries_handle.ptr(), state->q_tuple_owner,
                state->q_objs, state->q_ptrs, state->q_lens)) {
    PyErr_SetString(
        PyExc_NotImplementedError,
        "cdist_above_threshold currently requires byte-compatible inputs "
        "(bytes / 1-byte unicode). Wider unicode is not yet supported.");
    throw nb::python_error();
  }
  state->N = state->q_lens.size();

  if (state->symmetric) {
    // Share the snapshot across both sides.
    state->t_tuple_owner = state->q_tuple_owner;
    state->t_objs = state->q_objs;
    state->t_ptrs = state->q_ptrs;
    state->t_lens = state->q_lens;
    state->M = state->N;
  } else {
    if (!snapshot(targets_handle.ptr(), state->t_tuple_owner,
                  state->t_objs, state->t_ptrs, state->t_lens)) {
      PyErr_SetString(
          PyExc_NotImplementedError,
          "cdist_above_threshold currently requires byte-compatible inputs "
          "(bytes / 1-byte unicode). Wider unicode is not yet supported.");
      throw nb::python_error();
    }
    state->M = state->t_lens.size();
  }

  // Length caps (same as cdist).
  constexpr std::size_t cap = ::stride_align::cdist_runtime::kCdistMaxLen;
  for (std::size_t i = 0; i < state->N; ++i) {
    if (state->q_lens[i] > cap) {
      PyErr_Format(
          PyExc_NotImplementedError,
          "cdist_above_threshold: query %zu length %zu > SIMD cap %zu",
          i, state->q_lens[i], cap);
      throw nb::python_error();
    }
  }
  if (!state->symmetric) {
    for (std::size_t j = 0; j < state->M; ++j) {
      if (state->t_lens[j] > cap) {
        PyErr_Format(
            PyExc_NotImplementedError,
            "cdist_above_threshold: target %zu length %zu > SIMD cap %zu",
            j, state->t_lens[j], cap);
        throw nb::python_error();
      }
    }
  }
  // Hamming-normalized needs equal lengths everywhere.
  if (scorer == Scorer::HammingNormalized &&
      (state->N > 0U || state->M > 0U)) {
    const std::size_t ref =
        (state->N > 0U) ? state->q_lens[0] : state->t_lens[0];
    for (std::size_t i = 1; i < state->N; ++i) {
      if (state->q_lens[i] != ref) {
        PyErr_Format(
            PyExc_ValueError,
            "Hamming requires equal-length inputs (query 0 length %zu, "
            "query %zu length %zu)",
            ref, i, state->q_lens[i]);
        throw nb::python_error();
      }
    }
    if (!state->symmetric) {
      for (std::size_t j = 0; j < state->M; ++j) {
        if (state->t_lens[j] != ref) {
          PyErr_Format(
              PyExc_ValueError,
              "Hamming requires equal-length inputs (query 0 length %zu, "
              "target %zu length %zu)",
              ref, j, state->t_lens[j]);
          throw nb::python_error();
        }
      }
    }
  }

  // tqdm setup (main thread only).
  if (!tqdm_factory.is_none()) {
    std::uint64_t total_cost = 0;
    if (state->symmetric) {
      for (std::size_t i = 0; i < state->N; ++i) {
        for (std::size_t j = i + 1U; j < state->M; ++j) {
          total_cost += pair_cost(state->q_lens[i], state->t_lens[j]);
        }
      }
    } else {
      for (std::size_t i = 0; i < state->N; ++i) {
        for (std::size_t j = 0; j < state->M; ++j) {
          total_cost += pair_cost(state->q_lens[i], state->t_lens[j]);
        }
      }
    }
    state->pbar = tqdm_factory(nb::arg("total") = total_cost);
    state->have_tqdm =
        state->pbar.is_valid() && !state->pbar.is_none();
  }

  // Bounded queue. 1024 slots is enough to keep workers fed without
  // letting matched-result backpressure explode memory; back-pressure
  // kicks in via cv_not_full_ when matches outpace the consumer.
  constexpr std::size_t kQueueCapacity = 1024U;
  state->queue = std::make_unique<EventQueue>(kQueueCapacity);

  // Stash the per-scorer prune inputs on the state so the worker
  // (which is type-erased — doesn't see Ops) can call
  // max_normalized_similarity directly.
  state->scorer = scorer;
  state->jw_prefix_weight = jw_prefix_weight;
  state->jw_prefix_cap = jw_prefix_cap;

  // Bind the per-backend row kernel into a type-erased callback so
  // the iterator class doesn't need to know about Ops. The callback
  // now takes the candidate target arrays directly (already gathered
  // by the worker's length-prune step); for the no-pruning case the
  // worker passes a contiguous slice of the original target arrays.
  State* s = state.get();
  RowCompute compute;
  compute.run = [s, scorer, jw_prefix_weight, jw_prefix_threshold,
                 jw_prefix_cap, threshold](
                    std::size_t i,
                    const std::uint8_t* const* cand_ptrs,
                    const std::size_t* cand_lens,
                    std::size_t count,
                    std::size_t max_m, double* out) {
    // Pass the active threshold so Lev/OSA kernels can push the
    // per-pair distance cutoff in and bail rows that can't reach it.
    ::stride_align::cdist_simd::compute_row_double<Ops>(
        scorer,
        s->q_ptrs[i], s->q_lens[i],
        cand_ptrs, cand_lens,
        count, max_m, out,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap,
        threshold);
  };

  // Spawn workers (with GIL held — std::thread launches don't need
  // it). Workers will release the GIL implicitly: they never call
  // Python.
  spawn_workers(state, std::move(compute), cpu_count);

  // Hand the state to the Python iterator object.
  return nb::cast(ThresholdIterator(state));
}

}  // namespace stride_align::cdist_threshold
