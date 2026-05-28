#pragma once

// Dynamic Time Warping — scalar reference implementation.
//
// This is the correctness oracle and the fallback for backends that
// don't ship a SIMD specialisation. The SIMD batch kernel (phase
// C.1b) will live in src/cpp/dtw_simd.hpp and produce identical
// scores; the test suite cross-checks both against this scalar
// reference.
//
// Algorithm: full DTW with optional Sakoe-Chiba band.
// Distance functions: L1 (|x - y|) and L2-squared ((x - y)^2).
//
// Token / Cell separation mirrors the existing wide-Farrar pattern:
// `Token` is the input ndarray dtype; `Cell` is the DP accumulator
// type. For int16 inputs we widen the cell to int32 so the running
// sum can't overflow on reasonable-length sequences.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

namespace stride_align::dtw {

enum class DistanceKind {
  kL1,
  kL2Squared,
};

namespace detail {

// Per-cell distance. Widens to Cell before the subtract / abs to
// keep the int16 path from wrapping; the float paths compile to a
// straight subtract.
template <typename Token, typename Cell>
inline Cell cell_distance(Token x, Token y, DistanceKind dist) noexcept {
  if constexpr (std::is_floating_point_v<Cell>) {
    const Cell d = static_cast<Cell>(x) - static_cast<Cell>(y);
    return (dist == DistanceKind::kL2Squared) ? d * d : std::fabs(d);
  } else {
    const Cell xc = static_cast<Cell>(x);
    const Cell yc = static_cast<Cell>(y);
    const Cell d = xc - yc;
    if (dist == DistanceKind::kL2Squared) {
      return d * d;
    }
    // For signed integers, fabs would do a float cast. std::abs on
    // a wide-enough Cell is exact and lossless.
    return d < 0 ? -d : d;
  }
}

}  // namespace detail

// Resolve a window kwarg into an absolute Sakoe-Chiba radius in
// samples. A nullopt window means "no constraint" — returned as
// the maximum DP off-diagonal so the band check is a no-op.
//
// Caller responsibility:
//   * window_samples == nullopt        -> unconstrained
//   * window_samples == 0              -> diagonal only (still valid)
inline std::size_t resolve_band_radius(
    std::size_t query_size,
    std::size_t target_size,
    std::optional<std::size_t> window_samples) noexcept {
  if (!window_samples.has_value()) {
    return std::max(query_size, target_size);
  }
  // Even a 0-sample band is meaningful (diagonal only); preserve it.
  const std::size_t hard_max = std::max(query_size, target_size);
  return std::min(*window_samples, hard_max);
}

// Empty-input policy: the dispatcher raises ValueError before
// calling the kernel. The scalar reference still guards defensively
// against an empty span (returns +inf in DistanceKind units) so a
// misuse from a unit test surfaces as a sentinel instead of UB.
template <typename Token, typename Cell>
inline double dtw_score_scalar(
    std::span<const Token> query,
    std::span<const Token> target,
    DistanceKind dist,
    std::optional<std::size_t> window_samples) {
  const std::size_t m = query.size();
  const std::size_t n = target.size();
  if (m == 0 || n == 0) {
    return std::numeric_limits<double>::infinity();
  }

  const std::size_t radius = resolve_band_radius(m, n, window_samples);
  const Cell inf =
      std::numeric_limits<Cell>::has_infinity
          ? std::numeric_limits<Cell>::infinity()
          : std::numeric_limits<Cell>::max();

  // Two rolling rows. prev[j] = D(i-1, j); curr[j] = D(i, j).
  // Size n + 1: index 0 is the "left of the array" sentinel.
  std::vector<Cell> prev(n + 1, inf);
  std::vector<Cell> curr(n + 1, inf);
  prev[0] = Cell{0};

  for (std::size_t i = 1; i <= m; ++i) {
    curr[0] = inf;
    // Sakoe-Chiba band: only walk j in [max(1, i - radius),
    // min(n, i + radius)]. Outside the band stays +inf so the min
    // never picks it.
    const std::size_t j_lo = (i > radius) ? (i - radius) : 1;
    const std::size_t j_hi = std::min(n, i + radius);
    for (std::size_t j = j_lo; j <= j_hi; ++j) {
      const Cell d =
          detail::cell_distance<Token, Cell>(query[i - 1], target[j - 1], dist);
      const Cell candidate_diag = prev[j - 1];
      const Cell candidate_up = prev[j];
      const Cell candidate_left = curr[j - 1];
      const Cell prev_min =
          std::min(std::min(candidate_diag, candidate_up), candidate_left);
      // Saturate-on-overflow guard for ints: if prev_min is already
      // at the sentinel (inf for float, INT_MAX for int) we keep it
      // there rather than wrapping.
      if (prev_min == inf) {
        curr[j] = inf;
      } else {
        curr[j] = d + prev_min;
      }
    }
    std::swap(prev, curr);
    // Reset the row we're about to fill so the band's left/right
    // edges retain their +inf sentinel on the next pass.
    std::fill(curr.begin(), curr.end(), inf);
    curr[0] = inf;
  }

  // After the final swap, prev holds row m.
  const Cell final_cell = prev[n];
  if (final_cell == inf) {
    return std::numeric_limits<double>::infinity();
  }
  return static_cast<double>(final_cell);
}

}  // namespace stride_align::dtw
