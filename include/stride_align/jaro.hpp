#pragma once

// Jaro similarity (Jaro 1989) and Jaro-Winkler (Winkler 1990).
//
// Jaro counts how many characters of two strings can be paired up within
// a sliding window of floor(max(|a|, |b|) / 2) - 1 positions, plus a
// half-transposition penalty for matches that occur out of order. The
// resulting similarity is in [0, 1]; 1.0 = identical, 0.0 = no matches
// inside the window.
//
// Jaro-Winkler adds a length-of-common-prefix bonus (capped at 4 chars
// by convention) when the base Jaro score crosses a threshold (0.7 by
// convention, matching rapidfuzz). The bonus magnifies similarity for
// strings that agree at the start — useful for surname/place-name
// matching, less useful for arbitrary text.
//
// Neither metric has a DP recurrence; bit-parallel implementations live
// in src/cpp/jaro_simd.hpp and use a PEQ-style position bitmap of the
// query alphabet to find matches in O(1) per target character.
// This header carries the scalar reference + the threshold/weight
// defaults used by the bindings layer.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "stride_align/alignment.hpp"

namespace stride_align::jaro {

// Defaults match rapidfuzz's Jaro/JaroWinkler so that
// stride_align.jaro_winkler_similarity(a, b) == rapidfuzz at machine
// precision for byte inputs.
inline constexpr double kDefaultPrefixWeight = 0.1;
inline constexpr double kDefaultPrefixThreshold = 0.7;
inline constexpr std::size_t kDefaultPrefixCap = 4U;

// Half the maximum length, minus one, clamped to >= 0. Standard Jaro
// match window. For max <= 3 the formula underflows in unsigned
// arithmetic; the clamp gives "same-position matches only", which is
// what the floor would have produced anyway.
inline std::size_t match_window(std::size_t n, std::size_t m) noexcept {
  const std::size_t max_len = std::max(n, m);
  return max_len >= 2U ? max_len / 2U - 1U : 0U;
}

// Scalar Jaro similarity in [0, 1].
//
// Caller guarantees both spans use the same token width (templated, no
// implicit conversion). For empty inputs the convention is:
//   * both empty   → 1.0  (vacuously identical)
//   * one empty    → 0.0  (no possible match)
// This matches rapidfuzz.distance.Jaro.similarity.
template <typename Token>
inline double jaro_scalar(
    std::span<const Token> a,
    std::span<const Token> b) {
  const std::size_t n = a.size();
  const std::size_t m = b.size();
  if (n == 0U && m == 0U) {
    return 1.0;
  }
  if (n == 0U || m == 0U) {
    return 0.0;
  }

  const std::size_t window = match_window(n, m);

  std::vector<std::uint8_t> a_matched(n, 0U);
  std::vector<std::uint8_t> b_matched(m, 0U);
  std::size_t matches = 0U;

  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t lo = i > window ? i - window : 0U;
    const std::size_t hi = std::min(i + window + 1U, m);
    for (std::size_t j = lo; j < hi; ++j) {
      if (b_matched[j] != 0U) {
        continue;
      }
      if (a[i] != b[j]) {
        continue;
      }
      a_matched[i] = 1U;
      b_matched[j] = 1U;
      ++matches;
      break;
    }
  }

  if (matches == 0U) {
    return 0.0;
  }

  // Half-transposition count: walk matched chars of a and b in order;
  // each position where they differ contributes 1 to half_trans (which
  // is then divided by 2 for the formula).
  std::size_t half_trans = 0U;
  std::size_t k = 0U;
  for (std::size_t i = 0; i < n; ++i) {
    if (a_matched[i] == 0U) {
      continue;
    }
    while (b_matched[k] == 0U) {
      ++k;
    }
    if (a[i] != b[k]) {
      ++half_trans;
    }
    ++k;
  }

  // half_trans / 2 with INTEGER division — matches rapidfuzz exactly.
  // When the match permutation contains a cycle of odd length (say a 3-
  // cycle u→t→k→u), half_trans is odd (3 here), and the standard Jaro
  // formula's "transpositions = matches_out_of_order / 2" underestimates
  // the true transposition count from the cycle decomposition. The
  // literature and every common implementation accept that rounding-
  // down convention; matching it keeps our values bit-equivalent to
  // rapidfuzz's so users can swap one for the other.
  const double matches_d = static_cast<double>(matches);
  const double trans_d = static_cast<double>(half_trans / 2U);
  return (matches_d / static_cast<double>(n)
       + matches_d / static_cast<double>(m)
       + (matches_d - trans_d) / matches_d)
       / 3.0;
}

// Common-prefix length up to `cap` chars (inclusive of cap). Used by
// the Jaro-Winkler bonus; the standard literature uses cap = 4.
template <typename Token>
inline std::size_t common_prefix(
    std::span<const Token> a,
    std::span<const Token> b,
    std::size_t cap) noexcept {
  const std::size_t limit = std::min({a.size(), b.size(), cap});
  std::size_t i = 0;
  while (i < limit && a[i] == b[i]) {
    ++i;
  }
  return i;
}

// Jaro-Winkler similarity in [0, 1]. If the base Jaro score is at least
// `prefix_threshold`, add `L * prefix_weight * (1 - jaro)` where L is
// the common prefix length (capped at `prefix_cap`). Otherwise return
// the base Jaro score unchanged — this matches rapidfuzz's behavior
// (where below threshold no bonus is applied).
template <typename Token>
inline double jaro_winkler_scalar(
    std::span<const Token> a,
    std::span<const Token> b,
    double prefix_weight = kDefaultPrefixWeight,
    double prefix_threshold = kDefaultPrefixThreshold,
    std::size_t prefix_cap = kDefaultPrefixCap) {
  const double jaro = jaro_scalar<Token>(a, b);
  if (jaro < prefix_threshold) {
    return jaro;
  }
  const std::size_t L = common_prefix<Token>(a, b, prefix_cap);
  return jaro + static_cast<double>(L) * prefix_weight * (1.0 - jaro);
}

}  // namespace stride_align::jaro
