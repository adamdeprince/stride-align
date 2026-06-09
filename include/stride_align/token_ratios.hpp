#pragma once

// ``sa.token_sort_ratio`` and ``sa.token_set_ratio`` kernels.
//
// Both functions used to live in ``src/stride_align/_fuzz.py`` as pure
// Python: split-on-whitespace, set algebra, sort, join, then call into
// the Indel kernel. Profiling on a representative ``m=54 n=66`` pair
// showed 71% of token_set_ratio's per-call time was Python overhead
// (tokenisation + set ops + sort + join), only 18% was the actual
// Indel kernel. Lifting the whole recipe into C++ closes that gap.
//
// Tokenisation is whitespace-only (``' '``, ``'\t'``, ``'\n'``,
// ``'\r'``) to match the Python ``str.split()`` convention. Tokens
// are returned as views over the input buffer — no copies — so the
// per-call allocation is one ``std::vector<std::span<...>>`` for the
// token list plus one output buffer for each candidate string built
// by the set-ratio path.
//
// Algorithm (matches the Python reference bit-exactly):
//
//   token_sort_ratio(a, b):
//     ja = sort_join(tokens(a))
//     jb = sort_join(tokens(b))
//     return indel_normalized(ja, jb)
//
//   token_set_ratio(a, b):
//     ta = dedupe_sorted(tokens(a))
//     tb = dedupe_sorted(tokens(b))
//     intersect, diff_a, diff_b = three-way merge over (ta, tb)
//     t0 = join(intersect)
//     t1 = join(intersect ++ diff_a)   # concat, NOT re-merged
//     t2 = join(intersect ++ diff_b)   #   "
//     return max(indel_normalized(t0, t1),
//                indel_normalized(t0, t2),
//                indel_normalized(t1, t2))
//
// The "concat NOT re-merged" detail matters: rapidfuzz preserves
// ``t1 = intersect_sorted then diff_a_sorted``, which is NOT the
// same as ``sorted(intersect ∪ diff_a)`` whenever the two have
// interleaving lex order. Tests pin this against rapidfuzz.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "stride_align/indel.hpp"
#include "stride_align/lcs.hpp"

namespace stride_align::token_ratios {

using Codepoint = ::stride_align::lcs::Codepoint;

// Whitespace classifier. Matches Python's ``str.split()`` default
// (split on any sequence of ASCII whitespace).
template <typename Token>
inline bool is_ws(Token c) noexcept {
  return c == Token{' '} || c == Token{'\t'} ||
         c == Token{'\n'} || c == Token{'\r'};
}

// Split ``s`` on runs of whitespace, returning views over the input
// buffer (no copies). Empty input yields an empty vector. Leading and
// trailing whitespace is skipped (matches ``str.split()``).
template <typename Token>
inline std::vector<std::span<const Token>> tokens_of(
    std::span<const Token> s) {
  std::vector<std::span<const Token>> out;
  std::size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && is_ws(s[i])) ++i;
    if (i >= s.size()) break;
    const std::size_t start = i;
    while (i < s.size() && !is_ws(s[i])) ++i;
    out.emplace_back(s.data() + start, i - start);
  }
  return out;
}

// Lexicographic comparison of two token spans. Used for sorting and
// the three-way set-merge.
template <typename Token>
struct LexLess {
  bool operator()(std::span<const Token> a,
                  std::span<const Token> b) const noexcept {
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
      if (a[i] != b[i]) return a[i] < b[i];
    }
    return a.size() < b.size();
  }
};

template <typename Token>
struct LexEqual {
  bool operator()(std::span<const Token> a,
                  std::span<const Token> b) const noexcept {
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
  }
};

// Join token spans with a single space separator, writing into
// ``out``. Caller may reuse ``out`` across calls — it's cleared before
// writing.
template <typename Token>
inline void join_with_space(
    const std::vector<std::span<const Token>>& parts,
    std::vector<Token>& out) {
  std::size_t total = 0;
  if (!parts.empty()) {
    for (const auto& p : parts) total += p.size();
    total += parts.size() - 1U;  // separators
  }
  out.clear();
  out.reserve(total);
  for (std::size_t k = 0; k < parts.size(); ++k) {
    if (k > 0) out.push_back(Token{' '});
    out.insert(out.end(), parts[k].begin(), parts[k].end());
  }
}

// Indel-normalised similarity for two prepared sequences:
//
//   1 - indel(a, b) / (|a| + |b|)
//
// Empty-both convention matches rapidfuzz: ``("", "") -> 1.0``.
template <typename Token>
inline double indel_normalized(
    std::span<const Token> a,
    std::span<const Token> b) {
  if (a.empty() && b.empty()) return 1.0;
  const std::size_t total = a.size() + b.size();
  if (total == 0U) return 1.0;
  const std::size_t d = ::stride_align::indel::indel_distance<Token>(a, b);
  return 1.0 - static_cast<double>(d) / static_cast<double>(total);
}

// token_sort_ratio engine. ``a`` and ``b`` are the raw inputs.
template <typename Token>
inline double token_sort_ratio_engine(
    std::span<const Token> a,
    std::span<const Token> b) {
  auto ta = tokens_of(a);
  auto tb = tokens_of(b);
  // Empty-both: vacuously identical. One empty, one not: 0.
  if (ta.empty() && tb.empty()) {
    return (a.empty() && b.empty()) ? 1.0 : 1.0;
  }
  std::sort(ta.begin(), ta.end(), LexLess<Token>{});
  std::sort(tb.begin(), tb.end(), LexLess<Token>{});
  std::vector<Token> ja, jb;
  join_with_space(ta, ja);
  join_with_space(tb, jb);
  return indel_normalized<Token>(
      std::span<const Token>(ja.data(), ja.size()),
      std::span<const Token>(jb.data(), jb.size()));
}

// token_set_ratio engine.
template <typename Token>
inline double token_set_ratio_engine(
    std::span<const Token> a,
    std::span<const Token> b) {
  auto ta = tokens_of(a);
  auto tb = tokens_of(b);
  // rapidfuzz: token_set_ratio returns 0 when either side has no
  // tokens. The intersection-empty path would otherwise return 1.0
  // ("vacuously identical empty intersection") on whitespace-only
  // inputs, which diverges from upstream.
  if (ta.empty() || tb.empty()) return 0.0;

  // Sort + dedup.
  const LexLess<Token> lt;
  std::sort(ta.begin(), ta.end(), lt);
  ta.erase(std::unique(ta.begin(), ta.end(), LexEqual<Token>{}), ta.end());
  std::sort(tb.begin(), tb.end(), lt);
  tb.erase(std::unique(tb.begin(), tb.end(), LexEqual<Token>{}), tb.end());

  // Three-way merge over sorted-unique token lists. Produces the
  // intersection (in either side's order — both are identical), and
  // the two set-differences.
  std::vector<std::span<const Token>> intersect, diff_a, diff_b;
  intersect.reserve(std::min(ta.size(), tb.size()));
  std::size_t i = 0, j = 0;
  while (i < ta.size() && j < tb.size()) {
    if (lt(ta[i], tb[j])) {
      diff_a.push_back(ta[i]); ++i;
    } else if (lt(tb[j], ta[i])) {
      diff_b.push_back(tb[j]); ++j;
    } else {
      intersect.push_back(ta[i]); ++i; ++j;
    }
  }
  while (i < ta.size()) diff_a.push_back(ta[i++]);
  while (j < tb.size()) diff_b.push_back(tb[j++]);

  // Build the three candidate strings. ``t1`` / ``t2`` are the
  // intersect-sorted list followed by the diff-sorted list — NOT
  // re-merged into a single sorted union. (rapidfuzz preserves this
  // because it changes the indel score whenever intersect and diff
  // overlap in lex order.)
  std::vector<std::span<const Token>> t1_parts(intersect);
  t1_parts.insert(t1_parts.end(), diff_a.begin(), diff_a.end());
  std::vector<std::span<const Token>> t2_parts(intersect);
  t2_parts.insert(t2_parts.end(), diff_b.begin(), diff_b.end());

  std::vector<Token> t0, t1, t2;
  join_with_space(intersect, t0);
  join_with_space(t1_parts, t1);
  join_with_space(t2_parts, t2);

  // Common short-circuit: if either side has no tokens unique to it,
  // ``t1`` (or ``t2``) collapses to ``t0`` and the corresponding
  // ratio is 1.0 — no need to compute it. Same for the third
  // candidate. This dominates the all-tokens-shared case (token_set
  // returns 1.0 cleanly) and the one-side-superset case (the
  // superset-vs-superset ratio is 1.0 because one is a relabelling
  // of the other).
  if (diff_a.empty() && diff_b.empty()) return 1.0;  // t0 == t1 == t2
  const double r0 = indel_normalized<Token>(
      std::span<const Token>(t0.data(), t0.size()),
      std::span<const Token>(t1.data(), t1.size()));
  if (r0 >= 1.0) return 1.0;
  const double r1 = indel_normalized<Token>(
      std::span<const Token>(t0.data(), t0.size()),
      std::span<const Token>(t2.data(), t2.size()));
  if (r1 >= 1.0) return 1.0;
  const double r2 = indel_normalized<Token>(
      std::span<const Token>(t1.data(), t1.size()),
      std::span<const Token>(t2.data(), t2.size()));
  return std::max({r0, r1, r2});
}

// Public byte-fast-path-or-codepoint dispatchers. The byte fast path
// drops to ``std::uint8_t`` token type when every codepoint in both
// inputs is < 256, which routes the per-window Indel call through the
// flat 256-row PEQ and the K-specialised multi-word kernel.
inline double token_sort_ratio(
    const std::vector<Codepoint>& a,
    const std::vector<Codepoint>& b) {
  bool fits_in_byte = true;
  for (const auto cp : a) { if (cp >= 256U) { fits_in_byte = false; break; } }
  if (fits_in_byte) {
    for (const auto cp : b) { if (cp >= 256U) { fits_in_byte = false; break; } }
  }
  if (fits_in_byte) {
    std::vector<std::uint8_t> ab(a.begin(), a.end());
    std::vector<std::uint8_t> bb(b.begin(), b.end());
    return token_sort_ratio_engine<std::uint8_t>(
        std::span<const std::uint8_t>(ab),
        std::span<const std::uint8_t>(bb));
  }
  return token_sort_ratio_engine<Codepoint>(
      std::span<const Codepoint>(a),
      std::span<const Codepoint>(b));
}

// Byte fast path entries: caller has already established
// byte-compatible inputs (ASCII Python str or bytes-like).
inline double token_sort_ratio_bytes(
    std::span<const std::uint8_t> a,
    std::span<const std::uint8_t> b) {
  return token_sort_ratio_engine<std::uint8_t>(a, b);
}
inline double token_set_ratio_bytes(
    std::span<const std::uint8_t> a,
    std::span<const std::uint8_t> b) {
  return token_set_ratio_engine<std::uint8_t>(a, b);
}

inline double token_set_ratio(
    const std::vector<Codepoint>& a,
    const std::vector<Codepoint>& b) {
  bool fits_in_byte = true;
  for (const auto cp : a) { if (cp >= 256U) { fits_in_byte = false; break; } }
  if (fits_in_byte) {
    for (const auto cp : b) { if (cp >= 256U) { fits_in_byte = false; break; } }
  }
  if (fits_in_byte) {
    std::vector<std::uint8_t> ab(a.begin(), a.end());
    std::vector<std::uint8_t> bb(b.begin(), b.end());
    return token_set_ratio_engine<std::uint8_t>(
        std::span<const std::uint8_t>(ab),
        std::span<const std::uint8_t>(bb));
  }
  return token_set_ratio_engine<Codepoint>(
      std::span<const Codepoint>(a),
      std::span<const Codepoint>(b));
}

}  // namespace stride_align::token_ratios
