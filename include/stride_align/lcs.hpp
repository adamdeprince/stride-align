#pragma once

// Longest Common Subsequence (LCS) and Longest Common Substring (LCSubstr).
//
// Two distinct algorithms with confusable names:
//
//   * **Subsequence** — characters need not be contiguous. ``ABCBDAB`` and
//     ``BDCAB`` have LCS ``BCAB`` (length 4). The recurrence is the same
//     one Indel distance uses (``indel = |a| + |b| - 2·LCS``).
//   * **Substring** — characters must be contiguous. The same two
//     strings have LCSubstr ``AB`` (length 2). Different DP:
//     ``dp[i][j] = dp[i-1][j-1] + 1`` on match, ``0`` on mismatch.
//
// Both run in ``O(m·n)`` time with two rolling rows for ``O(min(m,n))``
// space. The substring DP additionally tracks the running maximum and
// the position at which it occurred so the substring itself can be
// recovered by a single slice (no traceback table).
//
// Public API takes ``std::vector<Codepoint>`` — the dispatch wrapper
// widens Python ``str`` storage straight out of ``PyUnicode_DATA``
// into codepoints, same convention as the other stride-align entry
// points that work in codepoint space.
//
// Source: textbook dynamic programming (Hirschberg 1975 and Wagner-
// Fischer 1974). The C++ here is original.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace stride_align::lcs {

using Codepoint = std::uint32_t;

// Length of the longest common SUBSEQUENCE of ``a`` and ``b``.
//
// The two-row DP is standard: ``dp[i][j]`` is the LCS length of the
// first ``i`` characters of ``a`` and the first ``j`` characters of
// ``b``. The recurrence is ``dp[i-1][j-1] + 1`` on a character match
// and ``max(dp[i-1][j], dp[i][j-1])`` otherwise. We index ``a`` along
// the columns (shorter side) so the rows are as short as possible.
inline std::size_t lcs_length(const std::vector<Codepoint>& a,
                               const std::vector<Codepoint>& b) {
  if (a.empty() || b.empty()) return 0;
  // Put the shorter sequence on the column axis so the rolling rows
  // hold ``min(|a|, |b|) + 1`` cells.
  const auto& cols = a.size() <= b.size() ? a : b;
  const auto& rows = a.size() <= b.size() ? b : a;
  const std::size_t M = cols.size();
  std::vector<std::size_t> prev(M + 1, 0);
  std::vector<std::size_t> curr(M + 1, 0);
  for (std::size_t i = 1; i <= rows.size(); ++i) {
    curr[0] = 0;
    for (std::size_t j = 1; j <= M; ++j) {
      if (rows[i - 1] == cols[j - 1]) {
        curr[j] = prev[j - 1] + 1;
      } else {
        curr[j] = std::max(prev[j], curr[j - 1]);
      }
    }
    prev.swap(curr);
  }
  return prev[M];
}

// Result of the substring DP: ``length`` is the LCSubstr length;
// ``end_a`` is the one-past-the-last index in ``a`` of an occurrence
// of that substring, so ``a[end_a - length .. end_a)`` is the
// substring itself. When the inputs share no character, ``length``
// is 0 and ``end_a`` is 0.
struct LcsSubstringInfo {
  std::size_t length = 0;
  std::size_t end_a = 0;
};

// Length and end-position of the longest common SUBSTRING.
//
// The recurrence ``dp[i][j] = dp[i-1][j-1] + 1`` on match and ``0``
// on mismatch lets us collapse to two rolling rows along the ``b``
// axis. The running maximum is tracked as ``(length, end_a)`` so the
// substring can be recovered with a single slice on ``a``.
inline LcsSubstringInfo lcs_substring_info(
    const std::vector<Codepoint>& a,
    const std::vector<Codepoint>& b) {
  LcsSubstringInfo r;
  if (a.empty() || b.empty()) return r;
  const std::size_t N = b.size();
  std::vector<std::size_t> prev(N + 1, 0);
  std::vector<std::size_t> curr(N + 1, 0);
  for (std::size_t i = 1; i <= a.size(); ++i) {
    curr[0] = 0;
    for (std::size_t j = 1; j <= N; ++j) {
      if (a[i - 1] == b[j - 1]) {
        curr[j] = prev[j - 1] + 1;
        if (curr[j] > r.length) {
          r.length = curr[j];
          r.end_a = i;
        }
      } else {
        curr[j] = 0;
      }
    }
    prev.swap(curr);
  }
  return r;
}

// Length of the longest common SUBSTRING.
inline std::size_t lcs_substring_length(const std::vector<Codepoint>& a,
                                          const std::vector<Codepoint>& b) {
  return lcs_substring_info(a, b).length;
}

// The longest common SUBSTRING itself, as a codepoint vector taken
// from ``a``. When multiple substrings tie at the maximum length, the
// FIRST occurrence in ``a`` (smallest ``end_a``) is returned —
// matching ``std::search`` / ``str.find`` conventions.
inline std::vector<Codepoint> lcs_substring(const std::vector<Codepoint>& a,
                                              const std::vector<Codepoint>& b) {
  const auto info = lcs_substring_info(a, b);
  if (info.length == 0) return {};
  return std::vector<Codepoint>(a.begin() + info.end_a - info.length,
                                a.begin() + info.end_a);
}

}  // namespace stride_align::lcs
