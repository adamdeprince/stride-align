#pragma once

// Indel distance — Levenshtein restricted to insertions and deletions
// (no substitutions). Equivalently:
//
//   indel(a, b) = |a| + |b| - 2 * LCS(a, b)
//
// Normalized similarity:
//
//   sim(a, b) = 1 - indel(a, b) / (|a| + |b|)
//             = 2 * LCS(a, b) / (|a| + |b|)
//
// The bit-parallel kernel uses the Allison-Dix (1986) LCS recurrence:
//   V starts as the all-ones m-bit vector.
//   For each text character c:
//     U = V & PEQ[c]
//     V = ((V + U) | (V - U)) & MASK
//   LCS = m - popcount(V).
//
// Hyyrö (2004) gives a cleaner derivation and the multi-word
// generalization; rapidfuzz uses the same recurrence.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "stride_align/alignment.hpp"

namespace stride_align::indel {

inline constexpr std::size_t kNoCutoff = std::numeric_limits<std::size_t>::max();

// Scalar DP reference. O(m*n) time, O(m) space (rolling rows).
// Correctness oracle for the bit-parallel paths.
template <typename Token>
inline std::size_t indel_dp(
    std::span<const Token> pattern,
    std::span<const Token> text) {
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0U) return n;
  if (n == 0U) return m;

  std::vector<std::size_t> prev(m + 1U);
  std::vector<std::size_t> curr(m + 1U);
  for (std::size_t i = 0; i <= m; ++i) {
    prev[i] = i;
  }
  for (std::size_t j = 1; j <= n; ++j) {
    curr[0] = j;
    for (std::size_t i = 1; i <= m; ++i) {
      if (pattern[i - 1U] == text[j - 1U]) {
        curr[i] = prev[i - 1U];
      } else {
        // No substitution: only insertion or deletion.
        curr[i] = std::min(prev[i] + 1U, curr[i - 1U] + 1U);
      }
    }
    std::swap(prev, curr);
  }
  return prev[m];
}

// Bit-parallel single-word indel for uint8 patterns of length <= 64.
//
// ``cutoff`` is the rapidfuzz-style score-cutoff: if the caller knows
// the result will only be used when distance <= cutoff, the kernel
// bails out of the per-character loop when its lower-bound estimate
// of the final distance exceeds the cutoff, returning ``cutoff + 1``
// (any value strictly greater than the cutoff carries the same
// "doesn't qualify" signal). The lower-bound formula: after
// processing ``j`` of ``n`` text chars with ``lcs_so_far = m -
// popcount(V)``, the final LCS can grow by at most ``n - j`` (each
// remaining text char can match at most one unused pattern slot), so
// the lower bound on final indel is
//   m + n - 2·(lcs_so_far + (n - j)) = m - n + 2j - 2·lcs_so_far.
// Equivalently the bail condition simplifies to
//   ``2*(j + popcount(V)) < m + n - cutoff``.
inline std::size_t indel_single_word_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) noexcept {
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0U) return n;
  if (n == 0U) return m;

  std::uint64_t peq[256] = {0};
  const std::uint64_t one = 1U;
  for (std::size_t i = 0; i < m; ++i) {
    peq[pattern[i]] |= one << i;
  }

  const std::uint64_t mask =
      (m == 64U) ? ~std::uint64_t{0} : ((one << m) - 1U);
  std::uint64_t V = mask;

  // Pre-compute the bail threshold once. The lower bound on final
  // indel after processing ``j`` of ``n`` text chars is
  //   2*(j + popcount(V)) - m - n
  // (derivation: final LCS ≤ (m - popcount(V)) + (n - j), so final
  // indel = m + n - 2·final_lcs ≥ 2j + 2·popcount(V) - m - n).
  // Bail when this lower bound exceeds the cutoff, i.e. when
  //   2*(j + popcount(V)) > m + n + cutoff.
  const bool has_cutoff = cutoff != kNoCutoff;
  const std::size_t bail_threshold = m + n + (has_cutoff ? cutoff : 0U);

  std::size_t j = 0;
  for (const std::uint8_t c : text) {
    const std::uint64_t U = V & peq[c];
    V = ((V + U) | (V - U)) & mask;
    ++j;
    if (has_cutoff) {
      const std::size_t lower_bound_score =
          2U * (j + static_cast<std::size_t>(std::popcount(V)));
      if (lower_bound_score > bail_threshold) {
        return cutoff + 1U;
      }
    }
  }
  const std::size_t lcs =
      m - static_cast<std::size_t>(std::popcount(V));
  return m + n - 2U * lcs;
}

// Token-generic bit-parallel for patterns up to 64 with arbitrary token
// alphabet. Uses a hashmap PEQ; slower than the uint8 path because of
// the lookup but still O(n*m/64) bit operations. Same cutoff
// semantics as ``indel_single_word_u8``.
template <typename Token>
inline std::size_t indel_distance(
    std::span<const Token> pattern,
    std::span<const Token> text,
    std::size_t cutoff = kNoCutoff) {
  static_assert(std::is_integral_v<Token> || std::is_unsigned_v<Token>);
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0U) return n;
  if (n == 0U) return m;
  if (m > 64U) {
    return indel_dp<Token>(pattern, text);
  }

  std::unordered_map<Token, std::uint64_t> peq;
  const std::uint64_t one = 1U;
  for (std::size_t i = 0; i < m; ++i) {
    peq[pattern[i]] |= one << i;
  }

  const std::uint64_t mask =
      (m == 64U) ? ~std::uint64_t{0} : ((one << m) - 1U);
  std::uint64_t V = mask;

  const bool has_cutoff = cutoff != kNoCutoff;
  const std::size_t bail_threshold = m + n + (has_cutoff ? cutoff : 0U);

  std::size_t j = 0;
  for (const auto c : text) {
    auto it = peq.find(c);
    const std::uint64_t pm = (it == peq.end()) ? 0U : it->second;
    const std::uint64_t U = V & pm;
    V = ((V + U) | (V - U)) & mask;
    ++j;
    if (has_cutoff) {
      const std::size_t lower_bound_score =
          2U * (j + static_cast<std::size_t>(std::popcount(V)));
      if (lower_bound_score > bail_threshold) {
        return cutoff + 1U;
      }
    }
  }
  const std::size_t lcs =
      m - static_cast<std::size_t>(std::popcount(V));
  return m + n - 2U * lcs;
}

// Convenience dispatch for u8 patterns: single-word bit-parallel for
// short patterns, scalar DP for longer. Cutoff is honoured by the
// bit-parallel path; the scalar DP path doesn't implement early-
// exit yet, so a cutoff there is informational only (the full
// distance is computed and returned).
inline std::size_t indel_distance_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) {
  if (pattern.size() > 0U && pattern.size() <= 64U) {
    return indel_single_word_u8(pattern, text, cutoff);
  }
  return indel_dp<std::uint8_t>(pattern, text);
}

// Normalized similarity. The denominator is |a| + |b| (NOT
// max(|a|, |b|) — Indel can be up to a_len + b_len, e.g.
// indel("aaa", "bbb") = 6).
inline double normalize(
    std::size_t distance, std::size_t a_len, std::size_t b_len) noexcept {
  const std::size_t total = a_len + b_len;
  if (total == 0U) return 1.0;
  return 1.0 -
         static_cast<double>(distance) / static_cast<double>(total);
}

}  // namespace stride_align::indel
