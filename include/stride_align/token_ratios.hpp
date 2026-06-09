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

// Forward declarations of the thread-local scratch type and accessor
// used by the engines below. The full definitions appear later in
// the file — they need the engines' helper types.
template <typename Token>
struct TokenWorkspace;
template <typename Token>
inline TokenWorkspace<Token>& token_workspace();

// token_sort_ratio engine. Uses thread-local scratch buffers for
// the token lists and the joined output, so consecutive calls reuse
// allocations.
template <typename Token>
inline double token_sort_ratio_engine(
    std::span<const Token> a,
    std::span<const Token> b);

// token_set_ratio engine. Same thread-local scratch as the sort
// engine — the two share buffers when called in sequence (e.g. by
// the WRatio path below).
template <typename Token>
inline double token_set_ratio_engine(
    std::span<const Token> a,
    std::span<const Token> b);

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

// In-place tokenisation variant that writes into a caller-owned
// buffer. Used by the combined WRatio token path that shares a
// thread-local scratch across token_sort + token_set computations.
template <typename Token>
inline void tokens_of_inplace(
    std::span<const Token> s,
    std::vector<std::span<const Token>>& out) {
  out.clear();
  std::size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && is_ws(s[i])) ++i;
    if (i >= s.size()) break;
    const std::size_t start = i;
    while (i < s.size() && !is_ws(s[i])) ++i;
    out.emplace_back(s.data() + start, i - start);
  }
}

// Thread-local scratch for the combined token-sort + token-set path.
// Carries every per-call buffer (token lists, sorted-vs-unique
// views, candidate joins) so consecutive WRatio calls reuse the
// allocations.
template <typename Token>
struct TokenWorkspace {
  std::vector<std::span<const Token>> tokens_a, tokens_b;
  std::vector<std::span<const Token>> intersect, diff_a, diff_b;
  std::vector<std::span<const Token>> t1_parts, t2_parts;
  std::vector<Token> ja, jb;     // sort_ratio joined strings
  std::vector<Token> t0, t1, t2; // set_ratio candidate strings
};

template <typename Token>
inline TokenWorkspace<Token>& token_workspace() {
  thread_local TokenWorkspace<Token> w;
  return w;
}

// Combined entry: compute both token_sort_ratio AND token_set_ratio
// in one pass over the inputs. Used by ``wratio_engine`` in the
// length-similar branch (``len_ratio < 1.5``), where both ratios are
// needed. Tokenises ``a`` and ``b`` ONCE, sorts ONCE (with
// duplicates), reuses the sorted views for the dedupe + set-merge,
// and shares the thread-local join buffers across the 1 + 3 indel
// calls. Writes the two ratios into the out-params.
template <typename Token>
inline void compute_token_sort_and_set_ratio(
    std::span<const Token> a,
    std::span<const Token> b,
    double& sort_score,
    double& set_score) {
  auto& w = token_workspace<Token>();
  tokens_of_inplace(a, w.tokens_a);
  tokens_of_inplace(b, w.tokens_b);

  // Edge cases mirror the standalone engines.
  const bool a_empty = w.tokens_a.empty();
  const bool b_empty = w.tokens_b.empty();
  if (a_empty && b_empty) {
    sort_score = 1.0;
    set_score  = 0.0;  // rapidfuzz convention: empty/empty token_set is 0
    return;
  }
  if (a_empty || b_empty) {
    sort_score = 0.0;
    set_score  = 0.0;
    return;
  }

  const LexLess<Token> lt;
  std::sort(w.tokens_a.begin(), w.tokens_a.end(), lt);
  std::sort(w.tokens_b.begin(), w.tokens_b.end(), lt);

  // --- token_sort_ratio leg: join the (sorted, with duplicates) views.
  join_with_space(w.tokens_a, w.ja);
  join_with_space(w.tokens_b, w.jb);
  sort_score = indel_normalized<Token>(
      std::span<const Token>(w.ja.data(), w.ja.size()),
      std::span<const Token>(w.jb.data(), w.jb.size()));

  // --- token_set_ratio leg: dedupe the already-sorted token lists
  // in place. The set algebra then runs on the unique views without
  // a second sort pass.
  w.tokens_a.erase(std::unique(w.tokens_a.begin(), w.tokens_a.end(),
                                LexEqual<Token>{}),
                    w.tokens_a.end());
  w.tokens_b.erase(std::unique(w.tokens_b.begin(), w.tokens_b.end(),
                                LexEqual<Token>{}),
                    w.tokens_b.end());

  w.intersect.clear();
  w.diff_a.clear();
  w.diff_b.clear();
  std::size_t i = 0, j = 0;
  while (i < w.tokens_a.size() && j < w.tokens_b.size()) {
    if (lt(w.tokens_a[i], w.tokens_b[j])) {
      w.diff_a.push_back(w.tokens_a[i]); ++i;
    } else if (lt(w.tokens_b[j], w.tokens_a[i])) {
      w.diff_b.push_back(w.tokens_b[j]); ++j;
    } else {
      w.intersect.push_back(w.tokens_a[i]); ++i; ++j;
    }
  }
  while (i < w.tokens_a.size()) w.diff_a.push_back(w.tokens_a[i++]);
  while (j < w.tokens_b.size()) w.diff_b.push_back(w.tokens_b[j++]);

  if (w.diff_a.empty() && w.diff_b.empty()) {
    set_score = 1.0;
    return;
  }

  w.t1_parts.assign(w.intersect.begin(), w.intersect.end());
  w.t1_parts.insert(w.t1_parts.end(), w.diff_a.begin(), w.diff_a.end());
  w.t2_parts.assign(w.intersect.begin(), w.intersect.end());
  w.t2_parts.insert(w.t2_parts.end(), w.diff_b.begin(), w.diff_b.end());

  join_with_space(w.intersect, w.t0);
  join_with_space(w.t1_parts, w.t1);
  join_with_space(w.t2_parts, w.t2);

  const double r0 = indel_normalized<Token>(
      std::span<const Token>(w.t0.data(), w.t0.size()),
      std::span<const Token>(w.t1.data(), w.t1.size()));
  if (r0 >= 1.0) { set_score = 1.0; return; }
  const double r1 = indel_normalized<Token>(
      std::span<const Token>(w.t0.data(), w.t0.size()),
      std::span<const Token>(w.t2.data(), w.t2.size()));
  if (r1 >= 1.0) { set_score = 1.0; return; }
  const double r2 = indel_normalized<Token>(
      std::span<const Token>(w.t1.data(), w.t1.size()),
      std::span<const Token>(w.t2.data(), w.t2.size()));
  set_score = std::max({r0, r1, r2});
}

// Standalone engine definitions (forward-declared earlier in the file
// because they need ``TokenWorkspace``).

template <typename Token>
inline double token_sort_ratio_engine(
    std::span<const Token> a,
    std::span<const Token> b) {
  auto& w = token_workspace<Token>();
  tokens_of_inplace(a, w.tokens_a);
  tokens_of_inplace(b, w.tokens_b);
  if (w.tokens_a.empty() && w.tokens_b.empty()) {
    return 1.0;
  }
  std::sort(w.tokens_a.begin(), w.tokens_a.end(), LexLess<Token>{});
  std::sort(w.tokens_b.begin(), w.tokens_b.end(), LexLess<Token>{});
  join_with_space(w.tokens_a, w.ja);
  join_with_space(w.tokens_b, w.jb);
  return indel_normalized<Token>(
      std::span<const Token>(w.ja.data(), w.ja.size()),
      std::span<const Token>(w.jb.data(), w.jb.size()));
}

template <typename Token>
inline double token_set_ratio_engine(
    std::span<const Token> a,
    std::span<const Token> b) {
  auto& w = token_workspace<Token>();
  tokens_of_inplace(a, w.tokens_a);
  tokens_of_inplace(b, w.tokens_b);
  if (w.tokens_a.empty() || w.tokens_b.empty()) return 0.0;

  const LexLess<Token> lt;
  std::sort(w.tokens_a.begin(), w.tokens_a.end(), lt);
  w.tokens_a.erase(std::unique(w.tokens_a.begin(), w.tokens_a.end(),
                                LexEqual<Token>{}),
                    w.tokens_a.end());
  std::sort(w.tokens_b.begin(), w.tokens_b.end(), lt);
  w.tokens_b.erase(std::unique(w.tokens_b.begin(), w.tokens_b.end(),
                                LexEqual<Token>{}),
                    w.tokens_b.end());

  w.intersect.clear();
  w.diff_a.clear();
  w.diff_b.clear();
  std::size_t i = 0, j = 0;
  while (i < w.tokens_a.size() && j < w.tokens_b.size()) {
    if (lt(w.tokens_a[i], w.tokens_b[j])) {
      w.diff_a.push_back(w.tokens_a[i]); ++i;
    } else if (lt(w.tokens_b[j], w.tokens_a[i])) {
      w.diff_b.push_back(w.tokens_b[j]); ++j;
    } else {
      w.intersect.push_back(w.tokens_a[i]); ++i; ++j;
    }
  }
  while (i < w.tokens_a.size()) w.diff_a.push_back(w.tokens_a[i++]);
  while (j < w.tokens_b.size()) w.diff_b.push_back(w.tokens_b[j++]);

  if (w.diff_a.empty() && w.diff_b.empty()) return 1.0;

  w.t1_parts.assign(w.intersect.begin(), w.intersect.end());
  w.t1_parts.insert(w.t1_parts.end(), w.diff_a.begin(), w.diff_a.end());
  w.t2_parts.assign(w.intersect.begin(), w.intersect.end());
  w.t2_parts.insert(w.t2_parts.end(), w.diff_b.begin(), w.diff_b.end());

  join_with_space(w.intersect, w.t0);
  join_with_space(w.t1_parts, w.t1);
  join_with_space(w.t2_parts, w.t2);

  const double r0 = indel_normalized<Token>(
      std::span<const Token>(w.t0.data(), w.t0.size()),
      std::span<const Token>(w.t1.data(), w.t1.size()));
  if (r0 >= 1.0) return 1.0;
  const double r1 = indel_normalized<Token>(
      std::span<const Token>(w.t0.data(), w.t0.size()),
      std::span<const Token>(w.t2.data(), w.t2.size()));
  if (r1 >= 1.0) return 1.0;
  const double r2 = indel_normalized<Token>(
      std::span<const Token>(w.t1.data(), w.t1.size()),
      std::span<const Token>(w.t2.data(), w.t2.size()));
  return std::max({r0, r1, r2});
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
