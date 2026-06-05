#pragma once

// Ratcliff-Obershelp similarity (Ratcliff & Metzener, 1988) — the
// algorithm Python's ``difflib.SequenceMatcher.ratio()`` ships.
//
// Algorithm (recursive longest-matching-substring):
//
//   1. Find the longest common substring of ``a`` and ``b``.
//   2. Recurse on the unmatched prefixes ``a[:i] / b[:j]`` and on the
//      unmatched suffixes ``a[i + k:] / b[j + k:]`` where the match
//      starts at ``a[i]`` / ``b[j]`` and has length ``k``.
//   3. Total matching characters ``M`` is the sum of all match
//      lengths found. Similarity is ``2 * M / (|a| + |b|)``.
//
// When ``a`` and ``b`` are both empty the similarity is 1.0 by
// convention (matches ``difflib.SequenceMatcher(None, "", "").ratio()``).
// When exactly one side is empty it is 0.0.
//
// The inner longest-common-substring step is the ``lcs_substring_info_range``
// DP from ``stride_align::lcs``; the recursion here adds the
// split-and-sum on top. Tiebreak on equal-length matches is "earliest
// in ``a``, then earliest in ``b``", matching the
// ``difflib.SequenceMatcher.find_longest_match`` contract.
//
// Source: textbook recursive Ratcliff-Obershelp. C++ here is original.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "stride_align/lcs.hpp"

namespace stride_align::ratcliff_obershelp {

using Codepoint = std::uint32_t;

namespace ro_detail {

// Sum of all match lengths recovered by the recursive
// longest-matching-substring scan on the half-open ranges
// ``a[a_lo, a_hi)`` and ``b[b_lo, b_hi)``. Uses an explicit
// work-stack instead of recursion to keep the call depth bounded
// regardless of pathological inputs.
inline std::size_t sum_match_lengths(const std::vector<Codepoint>& a,
                                       const std::vector<Codepoint>& b) {
  std::size_t total = 0;
  // Each work item is the rectangle we still need to scan.
  std::vector<std::array<std::size_t, 4>> stack;
  stack.reserve(8);
  stack.push_back({0, a.size(), 0, b.size()});
  while (!stack.empty()) {
    const auto rect = stack.back();
    stack.pop_back();
    const std::size_t a_lo = rect[0];
    const std::size_t a_hi = rect[1];
    const std::size_t b_lo = rect[2];
    const std::size_t b_hi = rect[3];
    if (a_lo >= a_hi || b_lo >= b_hi) continue;

    const auto info = ::stride_align::lcs::lcs_substring_info_range(
        a, a_lo, a_hi, b, b_lo, b_hi);
    if (info.length == 0) continue;

    total += info.length;
    const std::size_t a_match_start = info.end_a - info.length;
    const std::size_t b_match_start = info.end_b - info.length;
    // Push right half first, left half second; popping left next
    // (LIFO) explores in left-to-right order — matches difflib's
    // ``get_matching_blocks`` iteration order. Order does not affect
    // the total but matches the upstream behaviour for any future
    // matching-block exposure.
    if (info.end_a < a_hi && info.end_b < b_hi) {
      stack.push_back({info.end_a, a_hi, info.end_b, b_hi});
    }
    if (a_lo < a_match_start && b_lo < b_match_start) {
      stack.push_back({a_lo, a_match_start, b_lo, b_match_start});
    }
  }
  return total;
}

}  // namespace ro_detail

// Ratcliff-Obershelp similarity in ``[0, 1]``: ``2 * M / (|a| + |b|)``
// where ``M`` is the total length of all matching blocks produced by
// the recursive longest-common-substring split. Bit-exact with
// ``difflib.SequenceMatcher(None, a, b).ratio()`` on equal-character
// inputs (the difflib autojunk heuristic does not apply when
// ``autojunk`` is the default off, which is the case for our codepoint
// inputs since there is no junk character set).
inline double ratcliff_obershelp_similarity(
    const std::vector<Codepoint>& a,
    const std::vector<Codepoint>& b) {
  const std::size_t total_chars = a.size() + b.size();
  if (total_chars == 0) return 1.0;  // both empty -> identical
  const std::size_t M = ro_detail::sum_match_lengths(a, b);
  return 2.0 * static_cast<double>(M) / static_cast<double>(total_chars);
}

}  // namespace stride_align::ratcliff_obershelp
