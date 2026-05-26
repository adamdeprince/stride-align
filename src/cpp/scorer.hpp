#pragma once

// Substitution-score policies for the Smith-Waterman / Needleman-Wunsch
// kernels. The match/mismatch and matrix paths share the same DP
// recurrence; only the per-cell substitution cost lookup differs.
// Routing through a compile-time policy lets the compiler emit
// distinct code for each — a single integer compare for the
// match/mismatch case, a single matrix load for the matrix case —
// with no runtime branch in the hot loop.
//
// Existing match/mismatch callers see identical codegen to the
// pre-policy implementation: the `MatchMismatchScorer::substitute`
// call inlines, the `q == t ? match : mismatch` ternary becomes a
// branchless cmov on every backend we ship.

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace stride_align::scorer {

// Policy: substitution cost from a (query_token, target_token) pair
// via the classic match/mismatch model. The two scores are stored
// inline so the compiler can fold them into immediates when the
// scorer is instantiated at a known call site.
template <typename Cell>
struct MatchMismatchScorer {
  static_assert(std::is_integral_v<Cell>);

  Cell match;
  Cell mismatch;

  template <typename Token>
  inline Cell substitute(Token q, Token t) const noexcept {
    static_assert(std::is_integral_v<Token>);
    return q == t ? match : mismatch;
  }
};

// Policy: substitution cost from a contiguous row-major
// (alphabet × alphabet) matrix. The caller is responsible for
// guaranteeing every Token in both inputs is a valid index in
// [0, stride). Out-of-range tokens are undefined behaviour — the
// hot loop intentionally skips the bounds check.
//
// `stride` is also the alphabet size; the matrix layout is
// `matrix[q * stride + t]`. For BLOSUM62 with the standard 24-letter
// alphabet (20 amino acids + B Z X *), stride = 24 and the matrix
// has 576 cells.
template <typename Cell>
struct MatrixScorer {
  static_assert(std::is_integral_v<Cell>);

  const Cell* matrix;
  std::size_t stride;

  template <typename Token>
  inline Cell substitute(Token q, Token t) const noexcept {
    static_assert(std::is_integral_v<Token>);
    return matrix[static_cast<std::size_t>(q) * stride +
                  static_cast<std::size_t>(t)];
  }
};

// Trait: detect whether a Scorer is the match/mismatch variant. Used
// by fast-path code that wants to take advantage of the "all-match"
// or "match == mismatch" structural shortcuts the matrix path can't
// share (because the matrix can't be summarised by two scalars).
template <typename Scorer>
struct is_match_mismatch : std::false_type {};

template <typename Cell>
struct is_match_mismatch<MatchMismatchScorer<Cell>> : std::true_type {};

template <typename Scorer>
inline constexpr bool is_match_mismatch_v =
    is_match_mismatch<Scorer>::value;

}  // namespace stride_align::scorer
