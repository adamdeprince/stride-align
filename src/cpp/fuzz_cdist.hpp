#pragma once

// Native all-pairs matrix ("cdist") for the per-pair fuzz scorers that
// have irreducible per-pair work — partial_ratio (sliding window),
// token_set_ratio (per-pair set intersection), WRatio (weighted blend),
// and the partial-token variants. The rapidfuzz-shim Python `cdist`
// otherwise falls to a per-pair Python loop, crossing the C/Python
// boundary on every one of the N*M cells; for WRatio that was ~36ms on
// an 80x120 matrix (0.33x upstream).
//
// Here the inputs are snapshotted to byte spans ONCE under the GIL
// (zero-copy into the Python str buffers, kept alive by the owners),
// then the GIL is released and the N*M grid is evaluated entirely in
// C++ via the same `*_bytes` engines the per-pair dispatchers use — so
// results are bit-identical to calling the scorer per pair. Work is
// thread-pooled by query row. Byte-incompatible (wide-unicode) input
// raises NotImplementedError so the Python layer can fall back to its
// per-pair loop (which handles codepoints).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "cdist_runtime.hpp"
#include "thread_pool.hpp"
#include "stride_align/partial_ratio.hpp"
#include "stride_align/token_ratios.hpp"
#include "stride_align/wratio.hpp"

namespace stride_align::fuzz_cdist {

namespace nb = nanobind;

// Scorer ids shared with the Python `_NATIVE_FUZZ_CDIST` table. Kept in
// sync by hand; the Python side is the single source of the mapping.
enum class FuzzScorer : int {
  PartialRatio = 0,
  TokenSetRatio = 1,
  WRatio = 2,
  PartialTokenSortRatio = 3,
  PartialTokenSetRatio = 4,
  TokenRatio = 5,
  PartialTokenRatio = 6,
};

// One pair, byte path. Returns the rapidfuzz-scale similarity in
// [0, 100] (engines return [0, 1]); matches the per-pair dispatchers.
inline double eval_bytes(FuzzScorer s,
                         std::span<const std::uint8_t> a,
                         std::span<const std::uint8_t> b) {
  namespace pr = ::stride_align::partial_ratio;
  namespace tr = ::stride_align::token_ratios;
  namespace wr = ::stride_align::wratio;
  switch (s) {
    case FuzzScorer::PartialRatio:
      return pr::partial_ratio_bytes(a, b) * 100.0;
    case FuzzScorer::TokenSetRatio:
      return tr::token_set_ratio_bytes(a, b) * 100.0;
    case FuzzScorer::WRatio:
      return wr::wratio_bytes(a, b, 0.0) * 100.0;
    case FuzzScorer::PartialTokenSortRatio:
      return wr::partial_token_sort_ratio_bytes(a, b) * 100.0;
    case FuzzScorer::PartialTokenSetRatio:
      return wr::partial_token_set_ratio_bytes(a, b) * 100.0;
    case FuzzScorer::TokenRatio: {
      const double x = tr::token_sort_ratio_bytes(a, b);
      const double y = tr::token_set_ratio_bytes(a, b);
      return (x > y ? x : y) * 100.0;
    }
    case FuzzScorer::PartialTokenRatio: {
      const double x = wr::partial_token_sort_ratio_bytes(a, b);
      const double y = wr::partial_token_set_ratio_bytes(a, b);
      return (x > y ? x : y) * 100.0;
    }
  }
  return 0.0;
}

using ResultMatrix = nb::ndarray<nb::numpy, double, nb::ndim<2>>;

inline nb::object fuzz_cdist(nb::handle queries,
                            nb::handle targets,
                            int scorer_id,
                            std::size_t cpu_count) {
  namespace runtime = ::stride_align::cdist_runtime;
  const FuzzScorer fs = static_cast<FuzzScorer>(scorer_id);

  // === Snapshot inputs as byte spans while holding the GIL ===========
  nb::object q_owner;
  std::vector<const std::uint8_t*> q_ptrs;
  std::vector<std::size_t> q_lens;
  if (!runtime::snapshot_no_objs(queries.ptr(), q_owner, q_ptrs, q_lens)) {
    PyErr_SetString(
        PyExc_NotImplementedError,
        "fuzz cdist fast path requires byte-compatible inputs "
        "(bytes / 1-byte unicode); fall back to the per-pair loop.");
    throw nb::python_error();
  }
  const std::size_t N = q_lens.size();

  const bool symmetric = (queries.ptr() == targets.ptr());
  nb::object t_owner;
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
            targets.ptr(), t_owner, t_ptrs_storage, t_lens_storage)) {
      PyErr_SetString(
          PyExc_NotImplementedError,
          "fuzz cdist fast path requires byte-compatible inputs "
          "(bytes / 1-byte unicode); fall back to the per-pair loop.");
      throw nb::python_error();
    }
    t_ptrs = t_ptrs_storage.data();
    t_lens = t_lens_storage.data();
    M = t_lens_storage.size();
  }

  auto* heap = new std::vector<double>(N * M, 0.0);
  nb::capsule data_owner(heap, [](void* p) noexcept {
    delete static_cast<std::vector<double>*>(p);
  });
  double* data = heap->data();

  // Thread count: resolve cpu_count==0 to all cores, cap at N. The fuzz
  // scorers are heavy per pair (WRatio is ~6 sub-ratios), so the work
  // gate trips at a far lower (Sum q_len)*(Sum t_len) than the cheap
  // SIMD-distance cdist's 2.5M.
  std::size_t nthreads = cpu_count;
  if (nthreads == 0) {
    nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 4;
  }
  nthreads = std::max<std::size_t>(1U, std::min(nthreads, N));
  {
    std::uint64_t sum_q = 0, sum_t = 0;
    for (std::size_t i = 0; i < N; ++i) sum_q += q_lens[i];
    for (std::size_t j = 0; j < M; ++j) sum_t += t_lens[j];
    constexpr std::uint64_t kFuzzThreadThreshold = 100000ULL;
    if (sum_q * sum_t < kFuzzThreadThreshold) nthreads = 1;
  }

  auto& pool = ::stride_align::threading::ThreadPool::instance();
  PyThreadState* saved_state = PyEval_SaveThread();
  pool.parallel_for(N, nthreads, [&](std::size_t i) {
    const std::span<const std::uint8_t> a(q_ptrs[i], q_lens[i]);
    double* row = data + i * M;
    for (std::size_t j = 0; j < M; ++j) {
      row[j] = eval_bytes(
          fs, a, std::span<const std::uint8_t>(t_ptrs[j], t_lens[j]));
    }
  });
  PyEval_RestoreThread(saved_state);

  const std::size_t shape[2] = {N, M};
  ResultMatrix mat(data, 2, shape, data_owner);
  return mat.cast();
}

}  // namespace stride_align::fuzz_cdist
