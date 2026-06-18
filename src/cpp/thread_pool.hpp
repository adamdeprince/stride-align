#pragma once

// Persistent worker pool for the cdist family.
//
// Why: every cdist / cdist_top_k / cdist_above_threshold call used to
// spawn a fresh std::vector<std::thread> and join it at the end. On a
// 16-core box that spawn+join costs ~0.12 ms per call, which dominates
// small all-pairs matrices (a 50x100 Indel cdist computes in ~0.07 ms,
// so the thread churn more than doubled the wall time and turned wins
// into losses against single-threaded upstream). A pool created once
// and reused across calls drops that per-call cost to a couple of
// condition-variable wakeups.
//
// Model: a classic "parallel-for with dynamic row stealing". Workers
// block on a condition variable; `parallel_for` publishes a job (a
// body + an item count), wakes the workers, and the CALLING thread
// participates as one more worker so no core sits idle. Items are
// claimed with an atomic fetch-add, which keeps the symmetric
// upper-triangle's decreasing row sizes load-balanced exactly like the
// old hand-rolled loop did.
//
// Contract:
//   * The caller MUST have released the GIL before calling
//     parallel_for (workers never touch Python objects).
//   * body(i) is invoked exactly once for each i in [0, n) and may run
//     concurrently for different i. The cdist bodies write disjoint
//     output rows, so they satisfy this.
//   * Jobs are serialized: only one parallel_for runs on the pool at a
//     time (a job mutex). Concurrent cdist calls from different Python
//     threads queue rather than corrupt shared job state; throughput is
//     unaffected because each job still uses every core in turn.
//   * Fork-safe for the common case (fork while the pool is idle, e.g.
//     multiprocessing): a PID check rebuilds the workers in the child.
//     Forking *during* an active parallel_for on another thread is not
//     supported (and is already undefined for most threaded libraries).

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#include <unistd.h>  // getpid / pid_t

namespace stride_align::threading {

class ThreadPool {
 public:
  static ThreadPool& instance() {
    static ThreadPool pool;
    return pool;
  }

  // Number of persistent worker threads (0 until first use).
  std::size_t worker_count() {
    std::lock_guard<std::mutex> job(job_mutex_);
    ensure_started();
    return workers_.size();
  }

  // Run body(i) for every i in [0, n), using up to `max_workers`
  // threads total (the calling thread plus pool workers). Blocks until
  // every item completes. The caller must not hold the GIL.
  void parallel_for(std::size_t n, std::size_t max_workers,
                    const std::function<void(std::size_t)>& body) {
    if (n == 0) {
      return;
    }
    // Single-threaded request: run inline, never touch the pool. This
    // is the zero-overhead path the cheap-scorer small matrices want.
    if (max_workers <= 1) {
      for (std::size_t i = 0; i < n; ++i) {
        body(i);
      }
      return;
    }

    std::lock_guard<std::mutex> job(job_mutex_);  // serialize jobs
    ensure_started();

    // Pool workers to wake = desired total minus the calling thread,
    // capped at the pool size.
    const std::size_t wake = std::min(max_workers - 1, workers_.size());

    {
      std::lock_guard<std::mutex> lk(m_);
      body_ = &body;
      n_ = n;
      next_.store(0, std::memory_order_relaxed);
      wake_ = wake;
      remaining_ = wake;
      ++job_id_;
    }
    cv_work_.notify_all();

    // The calling thread is one of the workers — steal items too.
    drain(body, n);

    if (wake > 0) {
      std::unique_lock<std::mutex> lk(m_);
      cv_done_.wait(lk, [&] { return remaining_ == 0; });
      body_ = nullptr;
    }
  }

 private:
  ThreadPool() = default;

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      shutdown_ = true;
    }
    cv_work_.notify_all();
    for (auto& t : workers_) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // Claim and run items until the queue is empty. Shared by the pool
  // workers and the calling thread.
  void drain(const std::function<void(std::size_t)>& body, std::size_t n) {
    for (;;) {
      const std::size_t i = next_.fetch_add(1, std::memory_order_relaxed);
      if (i >= n) {
        return;
      }
      body(i);
    }
  }

  void worker_main(std::size_t id) {
    std::uint64_t seen = 0;
    for (;;) {
      const std::function<void(std::size_t)>* body = nullptr;
      std::size_t n = 0;
      bool participate = false;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_work_.wait(lk, [&] { return shutdown_ || job_id_ != seen; });
        if (shutdown_) {
          return;
        }
        seen = job_id_;
        participate = id < wake_;
        body = body_;
        n = n_;
      }
      if (!participate) {
        continue;  // not needed for this job; wait for the next one
      }
      drain(*body, n);
      {
        std::lock_guard<std::mutex> lk(m_);
        if (--remaining_ == 0) {
          cv_done_.notify_all();
        }
      }
    }
  }

  // Lazily create the workers; rebuild them after a fork. Called only
  // under job_mutex_, so it is never racing another parallel_for.
  void ensure_started() {
    const pid_t cur = getpid();
    if (started_ && pid_ == cur) {
      return;
    }
    if (started_ && pid_ != cur) {
      // We are in a forked child: the parent's worker threads do not
      // exist here. The std::thread objects are joinable but refer to
      // nothing, and destroying a joinable thread calls std::terminate,
      // so we abandon the vector in place (leaking its small buffer
      // once per fork) rather than running those destructors.
      ::new (static_cast<void*>(&workers_)) std::vector<std::thread>();
      shutdown_ = false;
      job_id_ = 0;
      remaining_ = 0;
      wake_ = 0;
      next_.store(0, std::memory_order_relaxed);
    }
    std::size_t hw = std::thread::hardware_concurrency();
    if (hw == 0) {
      hw = 4;
    }
    workers_.reserve(hw);
    for (std::size_t i = 0; i < hw; ++i) {
      workers_.emplace_back([this, i] { worker_main(i); });
    }
    started_ = true;
    pid_ = cur;
  }

  // Job serialization: at most one parallel_for active on the pool.
  std::mutex job_mutex_;

  // Worker synchronization.
  std::mutex m_;
  std::condition_variable cv_work_;  // workers wait here for a job
  std::condition_variable cv_done_;  // caller waits here for completion
  std::vector<std::thread> workers_;

  bool shutdown_ = false;
  bool started_ = false;
  pid_t pid_ = 0;

  // Current job (published under m_; body_ stays valid until the
  // caller observes remaining_ == 0).
  const std::function<void(std::size_t)>* body_ = nullptr;
  std::size_t n_ = 0;
  std::atomic<std::size_t> next_{0};
  std::size_t wake_ = 0;       // pool workers participating in this job
  std::size_t remaining_ = 0;  // pool workers not yet finished
  std::uint64_t job_id_ = 0;   // bumped per job; workers compare against it
};

}  // namespace stride_align::threading
