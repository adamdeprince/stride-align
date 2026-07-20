#pragma once

// Multi-target SIMD DTW: one target per SIMD lane.
//
// Static polymorphism (same shape as Myers / Farrar kernels):
//
//   * Cell Ops (SseF32Ops, Avx2I32Ops, …) are pure types with static
//     methods — no vtables, no function pointers.
//   * The DP sequence below is written once, templated on Ops; the
//     compiler monomorphises Ops::min / Ops::add / … into ISA code.
//   * Backend modules inject their Ops bundle via
//     ``dtw_distances_simd<Avx2DtwOps>(…)`` — import picks the .so,
//     the template picks the intrinsics.
//
// Scores match the Sakoe-Chiba scalar reference in
// include/stride_align/dtw.hpp (exact on int, a few ULPs on float).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include "dtw_simd_ops.hpp"
#include "stride_align/dtw.hpp"

namespace stride_align::dtw_simd {

// Generic batch DP. ``Ops`` supplies lanes / Vec / Cell and the
// arithmetic primitives; everything else is shared C++.
//
// ``batch_count`` is 1..Ops::lanes. Unused lanes are ignored on output.
// ``radii[l]`` is the absolute Sakoe-Chiba radius for lane ``l``.
template <typename Ops, typename Token, bool L2Squared>
inline void dtw_batch_kernel(
    const Token* query,
    std::size_t m,
    const Token* const* targets,
    const std::size_t* lengths,
    const std::size_t* radii,
    std::size_t batch_count,
    double* out) {
  using Cell = typename Ops::Cell;
  using Vec = typename Ops::Vec;
  constexpr std::size_t L = Ops::lanes;

  std::size_t max_n = 0;
  for (std::size_t l = 0; l < batch_count; ++l) {
    max_n = std::max(max_n, lengths[l]);
  }

  const Cell inf = Ops::inf_scalar();
  // Column-major-ish: prev[j * L + lane]
  std::vector<Cell> prev((max_n + 1U) * L, inf);
  std::vector<Cell> curr((max_n + 1U) * L, inf);

  for (std::size_t l = 0; l < batch_count; ++l) {
    prev[l] = Cell{0};  // prev[0] = 0 for active lanes
  }

  alignas(64) Cell sample_buf[L];
  alignas(64) Cell cell_buf[L];

  for (std::size_t i = 1; i <= m; ++i) {
    std::fill(curr.begin(), curr.end(), inf);

    const Cell q_cell = static_cast<Cell>(query[i - 1U]);
    const Vec qv = Ops::set1(q_cell);
    const Vec inf_v = Ops::infinity();

    for (std::size_t j = 1; j <= max_n; ++j) {
      for (std::size_t l = 0; l < L; ++l) {
        if (l < batch_count && j <= lengths[l]) {
          sample_buf[l] = static_cast<Cell>(targets[l][j - 1U]);
        } else {
          sample_buf[l] = Cell{0};
        }
      }
      const Vec tv = Ops::loadu(sample_buf);
      Vec d;
      if constexpr (L2Squared) {
        const Vec diff = Ops::sub(qv, tv);
        d = Ops::mul(diff, diff);
      } else {
        d = Ops::abs(Ops::sub(qv, tv));
      }

      const Vec diag = Ops::loadu(&prev[(j - 1U) * L]);
      const Vec up = Ops::loadu(&prev[j * L]);
      const Vec left = Ops::loadu(&curr[(j - 1U) * L]);
      const Vec best = Ops::min(Ops::min(diag, up), left);
      const Vec is_inf = Ops::cmp_eq(best, inf_v);
      Vec cell = Ops::add(d, best);
      cell = Ops::blend(is_inf, inf_v, cell);

      Ops::storeu(cell_buf, cell);
      for (std::size_t l = 0; l < L; ++l) {
        bool active = false;
        if (l < batch_count && j <= lengths[l]) {
          const std::size_t r = radii[l];
          const std::size_t j_lo = (i > r) ? (i - r) : 1U;
          const std::size_t j_hi = std::min(lengths[l], i + r);
          active = (j >= j_lo && j <= j_hi);
        }
        curr[j * L + l] = active ? cell_buf[l] : inf;
      }
    }
    prev.swap(curr);
  }

  for (std::size_t l = 0; l < batch_count; ++l) {
    const Cell v = prev[lengths[l] * L + l];
    if (v == inf) {
      out[l] = std::numeric_limits<double>::infinity();
    } else {
      out[l] = static_cast<double>(v);
    }
  }
}

// Chunk an arbitrary target list into Ops::lanes-wide waves.
// L2Squared is selected with if constexpr on a runtime enum via a
// dual instantiation (same idea as HasCutoff on the Myers batch).
template <typename Ops, typename Token>
inline void dtw_batch_all(
    const Token* query,
    std::size_t m,
    const std::vector<const Token*>& targets,
    const std::vector<std::size_t>& lengths,
    const std::vector<std::size_t>& radii,
    ::stride_align::dtw::DistanceKind dist,
    std::vector<double>& out) {
  const std::size_t n = targets.size();
  out.resize(n);
  constexpr std::size_t L = Ops::lanes;

  for (std::size_t base = 0; base < n; base += L) {
    const std::size_t batch = std::min(L, n - base);
    const Token* ptrs[L] = {};
    std::size_t lens[L] = {};
    std::size_t rads[L] = {};
    for (std::size_t l = 0; l < batch; ++l) {
      ptrs[l] = targets[base + l];
      lens[l] = lengths[base + l];
      rads[l] = radii[base + l];
    }
    if (dist == ::stride_align::dtw::DistanceKind::kL2Squared) {
      dtw_batch_kernel<Ops, Token, /*L2Squared=*/true>(
          query, m, ptrs, lens, rads, batch, out.data() + base);
    } else {
      dtw_batch_kernel<Ops, Token, /*L2Squared=*/false>(
          query, m, ptrs, lens, rads, batch, out.data() + base);
    }
  }
}

// Per-ISA bundle injected by a backend module. Nested cell Ops mirror
// how Avx512Ops::U32 nests a narrower sibling for the Myers path —
// one template parameter at the call site, dtype chooses the nested type.
//
//   Implementation::dtw_distances(...) {
//     return dtw_distances_simd<Avx2DtwOps>(...);
//   }
template <typename F32Ops, typename F64Ops, typename I32Ops>
struct DtwOpsBundle {
  using F32 = F32Ops;
  using F64 = F64Ops;
  using I32 = I32Ops;
};

#if defined(__SSE4_1__) || defined(__SSE4_2__)
using SseDtwOps = DtwOpsBundle<SseF32Ops, SseF64Ops, SseI32Ops>;
#endif
// Pure AVX (no AVX2): 256-bit float + SSE int32 cells. No backend
// module is pure-AVX today; the alias exists so a future module can
// inject the same way as AVX2.
#if defined(__AVX__) && defined(__SSE4_1__)
using AvxDtwOps = DtwOpsBundle<AvxF32Ops, AvxF64Ops, SseI32Ops>;
#endif
#if defined(__AVX2__)
using Avx2DtwOps = DtwOpsBundle<Avx2F32Ops, Avx2F64Ops, Avx2I32Ops>;
#endif
#if defined(__AVX512F__)
using Avx512DtwOps = DtwOpsBundle<Avx512F32Ops, Avx512F64Ops, Avx512I32Ops>;
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
using NeonDtwOps = DtwOpsBundle<NeonF32Ops, NeonF64Ops, NeonI32Ops>;
#endif
#if defined(__loongarch_sx)
using LsxDtwOps = DtwOpsBundle<LsxF32Ops, LsxF64Ops, LsxI32Ops>;
#endif
#if defined(__loongarch_asx)
using LasxDtwOps = DtwOpsBundle<LasxF32Ops, LasxF64Ops, LasxI32Ops>;
#endif

}  // namespace stride_align::dtw_simd
