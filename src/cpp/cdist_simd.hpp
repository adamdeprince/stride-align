#pragma once

// All-pairs SIMD batch ("cdist"). Computes an N x M distance- or
// similarity-matrix for every (query[i], target[j]) pair via the
// existing *_simd_raw entry points, with no per-row Python ABI cost.
//
// Single Python boundary crossing for the whole call (plus the
// optional tqdm progress callback): queries and targets are
// byte-viewed once up front, then the per-query loop calls the raw
// SIMD batch for each row.
//
// Symmetric optimization: when `queries is targets` and the metric is
// commutative (which all the supported scorers are), only the
// strictly upper triangle is computed; the lower triangle is mirrored
// and the diagonal is filled with the identity value (0 for distance,
// 1.0 for similarity / normalized).
//
// Progress: `tqdm_factory` is a Python callable returning a
// `tqdm`-like object. cdist calls it with `total=<estimated total
// work units>`, then calls `update(n)` after each query row with
// `n = work units for that row`. The cost model is
// `cost(q, t) = q_len * t_len` so the bar advances in time-proportional
// units even when the symmetric upper triangle has decreasing row
// sizes — the bar moves smoothly in wall-clock time.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "byte_view.hpp"
#include "cdist_runtime.hpp"
#include "hamming_simd.hpp"
#include "indel_simd.hpp"
#include "jaro_simd.hpp"
#include "levenshtein_simd.hpp"
#include "osa_simd.hpp"
#include "stride_align/alignment.hpp"
#include "stride_align/hamming.hpp"
#include "stride_align/indel.hpp"
#include "stride_align/jaro.hpp"
#include "stride_align/levenshtein.hpp"
#include "topk.hpp"  // Scorer enum

namespace stride_align::cdist_simd {

namespace nb = nanobind;

using Scorer = ::stride_align::topk::Scorer;

// 2-D NumPy matrix types used by the matrix variant. The factories
// themselves are out-of-line below; `make_int_matrix` /
// `make_double_matrix` allocate the storage on the heap and hand it
// to a capsule so the numpy view is zero-copy.
using ScoreMatrix = nb::ndarray<nb::numpy, std::int64_t, nb::ndim<2>>;
using NormalizedMatrix = nb::ndarray<nb::numpy, double, nb::ndim<2>>;

// Re-export the runtime constant via its old name for callers that
// referenced it through `cdist_simd::kCdistMaxLen`.
inline constexpr std::size_t kCdistMaxLen = ::stride_align::cdist_runtime::kCdistMaxLen;

// In-row normalize helpers (raw int distances -> doubles).
inline void normalize_lev_row(
    const std::int64_t* raw, double* out,
    std::size_t q_len, const std::size_t* t_lens, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = ::stride_align::levenshtein::normalize(
        static_cast<std::size_t>(raw[i]), q_len, t_lens[i]);
  }
}

inline void normalize_hamming_row(
    const std::int64_t* raw, double* out,
    std::size_t q_len, const std::size_t* /*t_lens*/, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = ::stride_align::hamming::normalize(
        static_cast<std::size_t>(raw[i]), q_len);
  }
}

inline void normalize_indel_row(
    const std::int64_t* raw, double* out,
    std::size_t q_len, const std::size_t* t_lens, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = ::stride_align::indel::normalize(
        static_cast<std::size_t>(raw[i]), q_len, t_lens[i]);
  }
}

// Compute one row's worth of raw scores into `row_out` (Score = int64
// for distance scorers; double for similarity scorers). Uses the
// per-scorer *_simd_raw entry points templated on Ops.
template <typename Ops>
inline void compute_row_int(
    Scorer scorer,
    const std::uint8_t* q_ptr, std::size_t q_len,
    const std::uint8_t* const* t_ptrs,
    const std::size_t* t_lens,
    std::size_t count,
    std::size_t max_m,
    std::int64_t* row_out) {
  switch (scorer) {
    case Scorer::Levenshtein:
      ::stride_align::levenshtein_simd::levenshtein_scores_simd_raw<Ops>(
          q_ptr, q_len, t_ptrs, t_lens, count,
          ::stride_align::levenshtein::kNoCutoff,
          reinterpret_cast<Score*>(row_out));
      return;
    case Scorer::DamerauLevenshtein:
      ::stride_align::osa_simd::osa_scores_simd_raw<Ops>(
          q_ptr, q_len, t_ptrs, t_lens, count,
          reinterpret_cast<Score*>(row_out));
      return;
    case Scorer::Hamming:
      ::stride_align::hamming_simd::hamming_scores_simd_raw(
          q_ptr, q_len, t_ptrs, t_lens, count,
          reinterpret_cast<Score*>(row_out));
      return;
    case Scorer::Indel:
      ::stride_align::indel_simd::indel_scores_simd_raw<Ops>(
          q_ptr, q_len, t_ptrs, t_lens, count,
          reinterpret_cast<Score*>(row_out));
      return;
    case Scorer::TrueDamerauLevenshtein: {
      // No SIMD path yet — scalar DP per pair. Hyyrö 2003 has a
      // bit-parallel form but it's significantly more complex than
      // OSA; deferred.
      Score* row = reinterpret_cast<Score*>(row_out);
      for (std::size_t i = 0; i < count; ++i) {
        row[i] = static_cast<Score>(
            ::stride_align::levenshtein::true_damerau_levenshtein_distance_u8(
                std::span<const std::uint8_t>(q_ptr, q_len),
                std::span<const std::uint8_t>(t_ptrs[i], t_lens[i])));
      }
      return;
    }
    default:
      PyErr_Format(PyExc_RuntimeError,
                   "cdist: unexpected int-valued scorer (%d)",
                   static_cast<int>(scorer));
      throw nb::python_error();
  }
  (void)max_m;
}

template <typename Ops>
inline void compute_row_double(
    Scorer scorer,
    const std::uint8_t* q_ptr, std::size_t q_len,
    const std::uint8_t* const* t_ptrs,
    const std::size_t* t_lens,
    std::size_t count,
    std::size_t max_m,
    double* row_out,
    double jw_prefix_weight,
    double jw_prefix_threshold,
    std::size_t jw_prefix_cap,
    double normalized_cutoff = 0.0) {
  // normalized_cutoff: if > 0, pairs with normalized similarity below this
  // can be early-exited by the kernel. Currently honored by
  // LevenshteinNormalized and DamerauLevenshteinNormalized; Hamming has
  // no useful early-exit (every byte must be processed) and Jaro has
  // multiple score terms whose early-exit is non-trivial.
  switch (scorer) {
    case Scorer::LevenshteinNormalized: {
      std::vector<Score> raw(count);
      if (normalized_cutoff > 0.0) {
        std::vector<std::size_t> cutoffs(count);
        for (std::size_t i = 0; i < count; ++i) {
          cutoffs[i] = ::stride_align::cdist_runtime::
              lev_distance_cutoff_for_normalized_threshold(
                  normalized_cutoff, q_len, t_lens[i]);
        }
        ::stride_align::levenshtein_simd::
            levenshtein_scores_simd_raw_per_pair<Ops>(
                q_ptr, q_len, t_ptrs, t_lens, count,
                cutoffs.data(), raw.data());
      } else {
        ::stride_align::levenshtein_simd::levenshtein_scores_simd_raw<Ops>(
            q_ptr, q_len, t_ptrs, t_lens, count,
            ::stride_align::levenshtein::kNoCutoff, raw.data());
      }
      normalize_lev_row(reinterpret_cast<const std::int64_t*>(raw.data()),
                        row_out, q_len, t_lens, count);
      return;
    }
    case Scorer::DamerauLevenshteinNormalized: {
      std::vector<Score> raw(count);
      if (normalized_cutoff > 0.0 && q_len > 0U && q_len <= 64U) {
        // OSA SIMD only supports single-word (q_len <= 64); the
        // _raw entry already gates on this. The cutoff variant only
        // helps when the SIMD kernel runs.
        std::vector<std::size_t> cutoffs(count);
        for (std::size_t i = 0; i < count; ++i) {
          cutoffs[i] = ::stride_align::cdist_runtime::
              lev_distance_cutoff_for_normalized_threshold(
                  normalized_cutoff, q_len, t_lens[i]);
        }
        ::stride_align::osa_simd::osa_scores_simd_raw_per_pair<Ops>(
            q_ptr, q_len, t_ptrs, t_lens, count,
            cutoffs.data(), raw.data());
      } else {
        ::stride_align::osa_simd::osa_scores_simd_raw<Ops>(
            q_ptr, q_len, t_ptrs, t_lens, count, raw.data());
      }
      normalize_lev_row(reinterpret_cast<const std::int64_t*>(raw.data()),
                        row_out, q_len, t_lens, count);
      return;
    }
    case Scorer::HammingNormalized: {
      std::vector<Score> raw(count);
      if (normalized_cutoff > 0.0) {
        // Hamming requires t_lens[i] == q_len upstream, so the
        // per-pair max is just q_len — same cutoff for every pair.
        // Build the per-pair array anyway so the kernel signature
        // matches the Lev/OSA shape.
        std::vector<std::size_t> cutoffs(count);
        const std::size_t cutoff =
            ::stride_align::cdist_runtime::
                lev_distance_cutoff_for_normalized_threshold(
                    normalized_cutoff, q_len, q_len);
        for (std::size_t i = 0; i < count; ++i) {
          cutoffs[i] = cutoff;
        }
        ::stride_align::hamming_simd::hamming_scores_simd_raw_per_pair(
            q_ptr, q_len, t_ptrs, t_lens, count,
            cutoffs.data(), raw.data());
      } else {
        ::stride_align::hamming_simd::hamming_scores_simd_raw(
            q_ptr, q_len, t_ptrs, t_lens, count, raw.data());
      }
      normalize_hamming_row(reinterpret_cast<const std::int64_t*>(raw.data()),
                            row_out, q_len, t_lens, count);
      return;
    }
    case Scorer::IndelNormalized: {
      std::vector<Score> raw(count);
      ::stride_align::indel_simd::indel_scores_simd_raw<Ops>(
          q_ptr, q_len, t_ptrs, t_lens, count, raw.data());
      normalize_indel_row(
          reinterpret_cast<const std::int64_t*>(raw.data()),
          row_out, q_len, t_lens, count);
      return;
    }
    case Scorer::TrueDamerauLevenshteinNormalized: {
      // Scalar per-pair, then normalize the same way as Lev (divide
      // by max(q_len, t_lens[i])).
      std::vector<Score> raw(count);
      for (std::size_t i = 0; i < count; ++i) {
        raw[i] = static_cast<Score>(
            ::stride_align::levenshtein::true_damerau_levenshtein_distance_u8(
                std::span<const std::uint8_t>(q_ptr, q_len),
                std::span<const std::uint8_t>(t_ptrs[i], t_lens[i])));
      }
      normalize_lev_row(
          reinterpret_cast<const std::int64_t*>(raw.data()),
          row_out, q_len, t_lens, count);
      return;
    }
    case Scorer::Jaro:
      ::stride_align::jaro_simd::jaro_similarities_simd_raw<Ops>(
          q_ptr, q_len, t_ptrs, t_lens, count, max_m, row_out);
      return;
    case Scorer::JaroWinkler: {
      ::stride_align::jaro_simd::jaro_similarities_simd_raw<Ops>(
          q_ptr, q_len, t_ptrs, t_lens, count, max_m, row_out);
      // Add Winkler prefix bonus per pair, in place.
      for (std::size_t i = 0; i < count; ++i) {
        if (row_out[i] < jw_prefix_threshold) {
          continue;
        }
        const std::size_t limit = std::min({q_len, t_lens[i], jw_prefix_cap});
        std::size_t L = 0;
        while (L < limit && q_ptr[L] == t_ptrs[i][L]) {
          ++L;
        }
        row_out[i] += static_cast<double>(L) * jw_prefix_weight *
                      (1.0 - row_out[i]);
      }
      return;
    }
    default:
      PyErr_Format(PyExc_RuntimeError,
                   "cdist: unexpected double-valued scorer (%d)",
                   static_cast<int>(scorer));
      throw nb::python_error();
  }
}

// Main cdist implementation. Templated on Ops (the SIMD bundle for
// the calling backend). Returns the result matrix as an nb::object
// holding either an int64 or float64 numpy ndarray depending on the
// scorer.
template <typename Ops>
inline nb::object cdist_impl(
    nb::handle queries_handle,
    nb::handle targets_handle,
    int scorer_int,
    nb::object tqdm_factory,
    std::size_t cpu_count,
    double jw_prefix_weight,
    double jw_prefix_threshold,
    std::size_t jw_prefix_cap) {
  namespace runtime = ::stride_align::cdist_runtime;
  const Scorer scorer = static_cast<Scorer>(scorer_int);
  const bool symmetric = (queries_handle.ptr() == targets_handle.ptr());

  // === Phase 1: snapshot inputs while holding the GIL ===============
  nb::object q_tuple_owner;
  std::vector<const std::uint8_t*> q_ptrs;
  std::vector<std::size_t> q_lens;
  if (!runtime::snapshot_no_objs(queries_handle.ptr(), q_tuple_owner, q_ptrs, q_lens)) {
    PyErr_SetString(
        PyExc_NotImplementedError,
        "cdist currently requires all queries to be byte-compatible "
        "(bytes / 1-byte unicode). Wider unicode is not yet supported.");
    throw nb::python_error();
  }
  const std::size_t N = q_lens.size();

  nb::object t_tuple_owner;
  std::vector<const std::uint8_t*> t_ptrs_storage;
  std::vector<std::size_t> t_lens_storage;
  const std::uint8_t* const* t_ptrs;
  const std::size_t* t_lens;
  std::size_t M;
  if (symmetric) {
    t_ptrs = q_ptrs.data();
    t_lens = q_lens.data();
    M = N;
  } else {
    if (!runtime::snapshot_no_objs(
            targets_handle.ptr(), t_tuple_owner, t_ptrs_storage, t_lens_storage)) {
      PyErr_SetString(
          PyExc_NotImplementedError,
          "cdist currently requires all targets to be byte-compatible "
          "(bytes / 1-byte unicode). Wider unicode is not yet supported.");
      throw nb::python_error();
    }
    t_ptrs = t_ptrs_storage.data();
    t_lens = t_lens_storage.data();
    M = t_lens_storage.size();
  }

  // Validate length caps and Hamming equal-length contract (both
  // raise on failure; both need the GIL for PyErr_Format).
  runtime::validate_length_caps(q_lens, symmetric ? q_lens : t_lens_storage, symmetric);
  runtime::validate_hamming_lengths(
      scorer, q_lens, symmetric ? q_lens : t_lens_storage, symmetric);

  const bool returns_double = runtime::scorer_returns_double(scorer);

  // Allocate the heap-backed matrix exactly once; defer the ndarray
  // wrapper until the return site.
  std::int64_t* int_data = nullptr;
  double* dbl_data = nullptr;
  nb::capsule data_owner;
  if (returns_double) {
    auto* heap = new std::vector<double>(N * M, 0.0);
    data_owner = nb::capsule(heap, [](void* p) noexcept {
      delete static_cast<std::vector<double>*>(p);
    });
    dbl_data = heap->data();
  } else {
    auto* heap = new std::vector<std::int64_t>(N * M, 0);
    data_owner = nb::capsule(heap, [](void* p) noexcept {
      delete static_cast<std::vector<std::int64_t>*>(p);
    });
    int_data = heap->data();
  }

  // tqdm setup (still under the GIL).
  nb::object pbar;
  if (!tqdm_factory.is_none()) {
    const std::uint64_t total =
        runtime::total_cost(q_lens, symmetric ? q_lens : t_lens_storage, symmetric);
    pbar = tqdm_factory(nb::arg("total") = total);
  }
  const bool have_tqdm = pbar.is_valid() && !pbar.is_none();

  // Pre-fill the diagonal for the symmetric path (still under GIL,
  // but cheap; could be done either side of the release).
  if (symmetric) {
    if (returns_double) {
      const double diag = runtime::diagonal_double(scorer);
      for (std::size_t i = 0; i < N; ++i) {
        dbl_data[i * M + i] = diag;
      }
    } else {
      const std::int64_t diag = runtime::diagonal_int(scorer);
      for (std::size_t i = 0; i < N; ++i) {
        int_data[i * M + i] = diag;
      }
    }
  }

  // === Phase 2: parallel compute (GIL released) ======================
  //
  // From here until phase 3, no thread touches a Python object except
  // briefly to update tqdm (each such update reacquires the GIL).
  // Work is distributed dynamically: every thread fetch-and-adds an
  // atomic row index, which keeps the symmetric-upper-triangle's
  // decreasing-row-size case load-balanced.

  const std::size_t num_threads =
      std::max<std::size_t>(1U, std::min(cpu_count, N));

  auto process_row = [&](std::size_t i) {
    const std::size_t j_start = symmetric ? i + 1U : 0U;
    const std::size_t row_count = M - j_start;
    if (row_count == 0U) {
      return;
    }
    std::size_t max_m = 0;
    for (std::size_t k = 0; k < row_count; ++k) {
      max_m = std::max(max_m, t_lens[j_start + k]);
    }
    const std::uint8_t* const* row_ptrs = t_ptrs + j_start;
    const std::size_t* row_lens = t_lens + j_start;

    if (returns_double) {
      double* row_out = dbl_data + i * M + j_start;
      compute_row_double<Ops>(
          scorer, q_ptrs[i], q_lens[i], row_ptrs, row_lens,
          row_count, max_m, row_out, jw_prefix_weight, jw_prefix_threshold,
          jw_prefix_cap);
      if (symmetric) {
        for (std::size_t k = 0; k < row_count; ++k) {
          const std::size_t j = j_start + k;
          dbl_data[j * M + i] = row_out[k];
        }
      }
    } else {
      std::int64_t* row_out = int_data + i * M + j_start;
      compute_row_int<Ops>(
          scorer, q_ptrs[i], q_lens[i], row_ptrs, row_lens,
          row_count, max_m, row_out);
      if (symmetric) {
        for (std::size_t k = 0; k < row_count; ++k) {
          const std::size_t j = j_start + k;
          int_data[j * M + i] = row_out[k];
        }
      }
    }
  };

  auto row_cost_of = [&](std::size_t i) -> std::uint64_t {
    const std::size_t j_start = symmetric ? i + 1U : 0U;
    const std::size_t row_count = M - j_start;
    std::uint64_t cost = 0;
    for (std::size_t k = 0; k < row_count; ++k) {
      cost += runtime::pair_cost(q_lens[i], t_lens[j_start + k]);
    }
    return cost;
  };

  // Workers don't touch Python — they just compute and push a
  // row-done notification onto a shared queue. The main thread is
  // the only one that calls into the tqdm bar; that keeps tqdm
  // implementations that aren't thread-safe (or that have thread
  // affinity) happy, and avoids a GIL-acquire from every worker.
  std::atomic<std::size_t> next_row{0};
  std::mutex done_mtx;
  std::condition_variable done_cv;
  std::deque<std::uint64_t> done_queue;  // per-row cost in pair units
  std::atomic<std::size_t> rows_finished{0};

  auto worker = [&]() {
    while (true) {
      const std::size_t i =
          next_row.fetch_add(1U, std::memory_order_relaxed);
      if (i >= N) {
        return;
      }
      process_row(i);
      const std::uint64_t cost = have_tqdm ? row_cost_of(i) : 0U;
      {
        std::lock_guard<std::mutex> lock(done_mtx);
        done_queue.push_back(cost);
      }
      rows_finished.fetch_add(1U, std::memory_order_release);
      done_cv.notify_one();
    }
  };

  if (num_threads == 1U) {
    // Single-threaded: still want the GIL released for the duration so
    // other Python threads can make progress, but no need for the
    // queue at all. Just run the row loop in-place.
    PyThreadState* saved_state = PyEval_SaveThread();
    for (std::size_t i = 0; i < N; ++i) {
      process_row(i);
      if (have_tqdm) {
        const std::uint64_t cost = row_cost_of(i);
        if (cost > 0U) {
          PyEval_RestoreThread(saved_state);
          pbar.attr("update")(cost);
          saved_state = PyEval_SaveThread();
        }
      }
    }
    PyEval_RestoreThread(saved_state);
  } else {
    // Multi-threaded: workers run with no GIL, main thread polls the
    // done queue and dispatches the tqdm updates. The main thread
    // sleeps on the cv with the GIL released so other Python threads
    // can run; it re-acquires the GIL only for the brief tqdm.update
    // call.
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    {
      PyThreadState* saved_state = PyEval_SaveThread();
      for (std::size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker);
      }
      // Main thread acts as the tqdm dispatcher.
      std::size_t dispatched = 0;
      while (dispatched < N) {
        std::uint64_t cost;
        {
          std::unique_lock<std::mutex> lk(done_mtx);
          done_cv.wait(lk, [&]() { return !done_queue.empty(); });
          cost = done_queue.front();
          done_queue.pop_front();
        }
        if (have_tqdm && cost > 0U) {
          PyEval_RestoreThread(saved_state);
          pbar.attr("update")(cost);
          saved_state = PyEval_SaveThread();
        }
        ++dispatched;
      }
      for (auto& t : threads) {
        t.join();
      }
      PyEval_RestoreThread(saved_state);
    }
  }

  // === Phase 3: wrap up (GIL held) ==================================

  if (pbar.is_valid() && !pbar.is_none()) {
    pbar.attr("close")();
  }

  const std::size_t shape[2] = {N, M};
  if (returns_double) {
    NormalizedMatrix mat(dbl_data, 2, shape, data_owner);
    return mat.cast();
  }
  ScoreMatrix mat(int_data, 2, shape, data_owner);
  return mat.cast();
}

}  // namespace stride_align::cdist_simd
