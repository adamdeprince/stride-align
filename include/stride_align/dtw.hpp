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

// Keogh lower bound (LB_Keogh) for equal-length series under a
// Sakoe-Chiba radius ``r``. Builds the query envelope once and sums
// the distance of each target sample to the nearest point in
// [L_i, U_i]. Local costs are non-negative, so LB ≤ true DTW.
//
// Returns +inf if ``score_cutoff`` is set and LB already exceeds it
// (early reject). Unequal lengths return 0 (no bound applied here —
// callers may still use the band |m−n| > r impossibility check).
template <typename Token, typename Cell>
inline double lb_keogh(
    std::span<const Token> query,
    std::span<const Token> target,
    DistanceKind dist,
    std::size_t radius,
    std::optional<double> score_cutoff = std::nullopt) {
  const std::size_t n = query.size();
  if (n == 0 || n != target.size()) {
    return 0.0;
  }

  // Envelope of the query: U[i] = max Q[i-r .. i+r], L similarly.
  std::vector<Cell> upper(n);
  std::vector<Cell> lower(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t lo = (i > radius) ? (i - radius) : 0U;
    const std::size_t hi = std::min(n - 1U, i + radius);
    Cell u = static_cast<Cell>(query[lo]);
    Cell l = u;
    for (std::size_t k = lo + 1U; k <= hi; ++k) {
      const Cell v = static_cast<Cell>(query[k]);
      if (v > u) u = v;
      if (v < l) l = v;
    }
    upper[i] = u;
    lower[i] = l;
  }

  Cell sum = Cell{0};
  for (std::size_t i = 0; i < n; ++i) {
    const Cell c = static_cast<Cell>(target[i]);
    Cell d = Cell{0};
    if (c > upper[i]) {
      d = detail::cell_distance<Cell, Cell>(c, upper[i], dist);
    } else if (c < lower[i]) {
      d = detail::cell_distance<Cell, Cell>(c, lower[i], dist);
    }
    sum = static_cast<Cell>(sum + d);
    if (score_cutoff.has_value() &&
        static_cast<double>(sum) > *score_cutoff) {
      return std::numeric_limits<double>::infinity();
    }
  }
  return static_cast<double>(sum);
}

// Build query envelope once for a 1-vs-N batch (equal-length targets).
template <typename Token, typename Cell>
struct QueryEnvelope {
  std::vector<Cell> upper;
  std::vector<Cell> lower;
  std::size_t n = 0;
  std::size_t radius = 0;
};

template <typename Token, typename Cell>
inline QueryEnvelope<Token, Cell> build_query_envelope(
    std::span<const Token> query,
    std::size_t radius) {
  QueryEnvelope<Token, Cell> env;
  env.n = query.size();
  env.radius = radius;
  env.upper.resize(env.n);
  env.lower.resize(env.n);
  for (std::size_t i = 0; i < env.n; ++i) {
    const std::size_t lo = (i > radius) ? (i - radius) : 0U;
    const std::size_t hi = std::min(env.n - 1U, i + radius);
    Cell u = static_cast<Cell>(query[lo]);
    Cell l = u;
    for (std::size_t k = lo + 1U; k <= hi; ++k) {
      const Cell v = static_cast<Cell>(query[k]);
      if (v > u) u = v;
      if (v < l) l = v;
    }
    env.upper[i] = u;
    env.lower[i] = l;
  }
  return env;
}

// LB_Keogh against a pre-built query envelope (equal length only).
template <typename Token, typename Cell>
inline double lb_keogh_with_envelope(
    const QueryEnvelope<Token, Cell>& env,
    std::span<const Token> target,
    DistanceKind dist,
    std::optional<double> score_cutoff = std::nullopt) {
  if (env.n == 0 || env.n != target.size()) {
    return 0.0;
  }
  Cell sum = Cell{0};
  for (std::size_t i = 0; i < env.n; ++i) {
    const Cell c = static_cast<Cell>(target[i]);
    Cell d = Cell{0};
    if (c > env.upper[i]) {
      d = detail::cell_distance<Cell, Cell>(c, env.upper[i], dist);
    } else if (c < env.lower[i]) {
      d = detail::cell_distance<Cell, Cell>(c, env.lower[i], dist);
    }
    sum = static_cast<Cell>(sum + d);
    if (score_cutoff.has_value() &&
        static_cast<double>(sum) > *score_cutoff) {
      return std::numeric_limits<double>::infinity();
    }
  }
  return static_cast<double>(sum);
}

// Empty-input policy: the dispatcher raises ValueError before
// calling the kernel. The scalar reference still guards defensively
// against an empty span (returns +inf in DistanceKind units) so a
// misuse from a unit test surfaces as a sentinel instead of UB.
//
// ``score_cutoff``: when set, return +inf as soon as we can prove the
// final DTW exceeds the cutoff (LB_Keogh pre-filter for equal lengths,
// band impossibility, and per-row early abandon in the DP).
template <typename Token, typename Cell>
inline double dtw_score_scalar(
    std::span<const Token> query,
    std::span<const Token> target,
    DistanceKind dist,
    std::optional<std::size_t> window_samples,
    std::optional<double> score_cutoff = std::nullopt) {
  const std::size_t m = query.size();
  const std::size_t n = target.size();
  if (m == 0 || n == 0) {
    return std::numeric_limits<double>::infinity();
  }

  const std::size_t radius = resolve_band_radius(m, n, window_samples);
  // Sakoe-Chiba impossibility: warping path cannot connect ends.
  if (m > n + radius || n > m + radius) {
    return std::numeric_limits<double>::infinity();
  }

  // LB_Keogh (equal length): cheap reject under a finite cutoff.
  if (score_cutoff.has_value() && m == n) {
    const double lb =
        lb_keogh<Token, Cell>(query, target, dist, radius, score_cutoff);
    if (lb > *score_cutoff) {
      return std::numeric_limits<double>::infinity();
    }
  }

  const Cell inf =
      std::numeric_limits<Cell>::has_infinity
          ? std::numeric_limits<Cell>::infinity()
          : std::numeric_limits<Cell>::max();
  const bool has_cutoff = score_cutoff.has_value();
  const Cell cutoff_cell = has_cutoff
      ? static_cast<Cell>(*score_cutoff)
      : Cell{0};

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
    Cell row_min = inf;
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
      if (curr[j] < row_min) {
        row_min = curr[j];
      }
    }
    // Local costs ≥ 0 ⇒ if every live cell already exceeds cutoff,
    // the final score cannot recover.
    if (has_cutoff && row_min > cutoff_cell && row_min != inf) {
      return std::numeric_limits<double>::infinity();
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
  const double result = static_cast<double>(final_cell);
  if (has_cutoff && result > *score_cutoff) {
    return std::numeric_limits<double>::infinity();
  }
  return result;
}

}  // namespace stride_align::dtw
