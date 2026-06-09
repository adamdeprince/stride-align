#pragma once

// ``sa.WRatio`` / ``stride_align.rapidfuzz.fuzz.WRatio`` kernel.
//
// Reimplements rapidfuzz's documented WRatio recipe in C++ so the
// whole computation (ratio + len-ratio branch + token variants +
// partial variants) crosses the Python boundary exactly once instead
// of once per component. Closes the ~4× WRatio gap on x86/ARM that
// was driven entirely by Python wrapper overhead and intermediate
// ``float()`` / ``max()`` work in the previous Python recipe.
//
// Recipe (matches the Python reference bit-exactly):
//
//   base = indel_normalized(a, b)        # range [0, 1]
//   len_ratio = max(|a|, |b|) / min(|a|, |b|)
//   UNBASE_SCALE = 0.95
//
//   if len_ratio < 1.5:
//     # similar-length regime — token-based variants
//     token_sort = token_sort_ratio(a, b) * UNBASE_SCALE
//     token_set  = token_set_ratio(a, b)  * UNBASE_SCALE
//     score      = max(base, token_sort, token_set)
//   else:
//     # length-mismatched regime — partial variants dominate
//     partial_scale = 0.9 if len_ratio < 8 else 0.6
//     partial             = partial_ratio(a, b)             * partial_scale
//     partial_token_sort  = partial_token_sort_ratio(a, b)  * UNBASE_SCALE * partial_scale
//     partial_token_set   = partial_token_set_ratio(a, b)   * UNBASE_SCALE * partial_scale
//     score = max(base, partial, partial_token_sort, partial_token_set)
//
//   return score
//
// **Short-circuit logic.** Every non-base component is multiplied by
// a scale factor ``<= 0.95`` (in the similar-length branch) or
// ``<= 0.9 / 0.6`` (in the length-mismatched branch). The component
// value itself is in ``[0, 1]``, so the scaled value is bounded by
// the scale factor. Once ``base`` is already at least the maximum
// possible component score, no further computation can change the
// answer — return ``base`` and skip the (expensive) token / partial
// passes. This is the dominant fast path for high-quality matches.
//
// All component subroutines (token_sort, token_set, partial,
// partial_token_sort, partial_token_set) are inlined from
// ``token_ratios.hpp`` and ``partial_ratio.hpp`` so the whole
// recipe is a single function instantiation per (Token, K) pair.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "stride_align/indel.hpp"
#include "stride_align/lcs.hpp"
#include "stride_align/partial_ratio.hpp"
#include "stride_align/token_ratios.hpp"

namespace stride_align::wratio {

using Codepoint = ::stride_align::lcs::Codepoint;

// Combined ``partial_token_sort_ratio`` on already-tokenised inputs.
// Sorts the tokens, joins, hands off to ``partial_ratio``.
template <typename Token>
inline double partial_token_sort_ratio_engine(
    std::span<const Token> a,
    std::span<const Token> b) {
  auto ta = ::stride_align::token_ratios::tokens_of(a);
  auto tb = ::stride_align::token_ratios::tokens_of(b);
  std::sort(ta.begin(), ta.end(),
            ::stride_align::token_ratios::LexLess<Token>{});
  std::sort(tb.begin(), tb.end(),
            ::stride_align::token_ratios::LexLess<Token>{});
  std::vector<Token> ja, jb;
  ::stride_align::token_ratios::join_with_space(ta, ja);
  ::stride_align::token_ratios::join_with_space(tb, jb);
  if (ja.empty() && jb.empty()) return 1.0;
  if (ja.empty() || jb.empty()) return 0.0;

  // partial_ratio operates on codepoint vectors. For the byte fast
  // path we already have Token = uint8_t, so widen-by-copy is cheap.
  std::vector<Codepoint> ja_cp(ja.begin(), ja.end());
  std::vector<Codepoint> jb_cp(jb.begin(), jb.end());
  return ::stride_align::partial_ratio::partial_ratio(ja_cp, jb_cp);
}

// Combined ``partial_token_set_ratio``. Builds the three set
// candidates exactly as ``token_set_ratio_engine`` does, then runs
// ``partial_ratio`` over each pair and returns the max.
template <typename Token>
inline double partial_token_set_ratio_engine(
    std::span<const Token> a,
    std::span<const Token> b) {
  using namespace ::stride_align::token_ratios;
  auto ta = tokens_of(a);
  auto tb = tokens_of(b);
  if (ta.empty() || tb.empty()) return 0.0;

  const LexLess<Token> lt;
  std::sort(ta.begin(), ta.end(), lt);
  ta.erase(std::unique(ta.begin(), ta.end(), LexEqual<Token>{}), ta.end());
  std::sort(tb.begin(), tb.end(), lt);
  tb.erase(std::unique(tb.begin(), tb.end(), LexEqual<Token>{}), tb.end());

  std::vector<std::span<const Token>> intersect, diff_a, diff_b;
  std::size_t i = 0, j = 0;
  while (i < ta.size() && j < tb.size()) {
    if (lt(ta[i], tb[j])) { diff_a.push_back(ta[i]); ++i; }
    else if (lt(tb[j], ta[i])) { diff_b.push_back(tb[j]); ++j; }
    else { intersect.push_back(ta[i]); ++i; ++j; }
  }
  while (i < ta.size()) diff_a.push_back(ta[i++]);
  while (j < tb.size()) diff_b.push_back(tb[j++]);

  if (diff_a.empty() && diff_b.empty()) return 1.0;

  std::vector<std::span<const Token>> t1_parts(intersect);
  t1_parts.insert(t1_parts.end(), diff_a.begin(), diff_a.end());
  std::vector<std::span<const Token>> t2_parts(intersect);
  t2_parts.insert(t2_parts.end(), diff_b.begin(), diff_b.end());

  std::vector<Token> t0, t1, t2;
  join_with_space(intersect, t0);
  join_with_space(t1_parts, t1);
  join_with_space(t2_parts, t2);

  // Run partial_ratio on each pair (codepoint widening per pair).
  auto pr = [](const std::vector<Token>& x, const std::vector<Token>& y) -> double {
    if (x.empty() && y.empty()) return 1.0;
    if (x.empty() || y.empty()) return 0.0;
    std::vector<Codepoint> xc(x.begin(), x.end());
    std::vector<Codepoint> yc(y.begin(), y.end());
    return ::stride_align::partial_ratio::partial_ratio(xc, yc);
  };
  const double r0 = pr(t0, t1);
  if (r0 >= 1.0) return 1.0;
  const double r1 = pr(t0, t2);
  if (r1 >= 1.0) return 1.0;
  const double r2 = pr(t1, t2);
  return std::max({r0, r1, r2});
}

// WRatio engine. Returns the WRatio score in ``[0, 1]`` (caller
// multiplies by 100 for the rapidfuzz convention).
//
// ``score_cutoff`` is interpreted as a *normalised* cutoff in
// ``[0, 1]`` — values at or below this threshold are returned as 0.
// This mirrors rapidfuzz's documented behaviour on WRatio.
template <typename Token>
inline double wratio_engine(
    std::span<const Token> a,
    std::span<const Token> b,
    double score_cutoff /* normalised, [0, 1]; 0 == no cutoff */) {
  if (a.empty() || b.empty()) return 0.0;

  constexpr double UNBASE_SCALE = 0.95;

  const double base = ::stride_align::token_ratios::indel_normalized<Token>(a, b);
  const double len_a = static_cast<double>(a.size());
  const double len_b = static_cast<double>(b.size());
  const double len_ratio = (len_a > len_b ? len_a / len_b : len_b / len_a);

  double best = base;

  if (len_ratio < 1.5) {
    // Similar-length regime. Token variants scale by 0.95, so once
    // ``base >= 0.95`` no scaled token component can exceed it.
    if (best >= UNBASE_SCALE) {
      return best > score_cutoff ? best : 0.0;
    }
    const double ts = ::stride_align::token_ratios::token_sort_ratio_engine<Token>(a, b);
    const double ts_scaled = ts * UNBASE_SCALE;
    if (ts_scaled > best) best = ts_scaled;
    if (best >= UNBASE_SCALE) {
      return best > score_cutoff ? best : 0.0;
    }
    const double tx = ::stride_align::token_ratios::token_set_ratio_engine<Token>(a, b);
    const double tx_scaled = tx * UNBASE_SCALE;
    if (tx_scaled > best) best = tx_scaled;
    return best > score_cutoff ? best : 0.0;
  }

  // Length-mismatched regime. The dominant variant is ``partial`` (no
  // unbase penalty), so its scaling factor caps the achievable score.
  const double partial_scale = (len_ratio < 8.0) ? 0.9 : 0.6;
  // partial_token_* are scaled by UNBASE_SCALE * partial_scale, which
  // is strictly less than ``partial_scale``. So ``partial_scale`` is
  // the true ceiling for any non-base component here.
  if (best >= partial_scale) {
    return best > score_cutoff ? best : 0.0;
  }

  // Widen inputs to codepoints for the partial_ratio family (its
  // matching-block enumeration operates on codepoints internally).
  std::vector<Codepoint> a_cp(a.begin(), a.end());
  std::vector<Codepoint> b_cp(b.begin(), b.end());
  const double pr = ::stride_align::partial_ratio::partial_ratio(a_cp, b_cp);
  const double pr_scaled = pr * partial_scale;
  if (pr_scaled > best) best = pr_scaled;
  // After ``partial`` the ceiling drops to ``UNBASE_SCALE *
  // partial_scale`` because the remaining variants pay both penalties.
  const double remaining_ceiling = UNBASE_SCALE * partial_scale;
  if (best >= remaining_ceiling) {
    return best > score_cutoff ? best : 0.0;
  }

  const double pts = partial_token_sort_ratio_engine<Token>(a, b);
  const double pts_scaled = pts * remaining_ceiling;
  if (pts_scaled > best) best = pts_scaled;
  if (best >= remaining_ceiling) {
    return best > score_cutoff ? best : 0.0;
  }

  const double pset = partial_token_set_ratio_engine<Token>(a, b);
  const double pset_scaled = pset * remaining_ceiling;
  if (pset_scaled > best) best = pset_scaled;
  return best > score_cutoff ? best : 0.0;
}

// Public byte-fast-path-or-codepoint dispatcher.
inline double wratio(
    const std::vector<Codepoint>& a,
    const std::vector<Codepoint>& b,
    double score_cutoff = 0.0) {
  if (a.empty() || b.empty()) return 0.0;
  bool fits_in_byte = true;
  for (const auto cp : a) { if (cp >= 256U) { fits_in_byte = false; break; } }
  if (fits_in_byte) {
    for (const auto cp : b) { if (cp >= 256U) { fits_in_byte = false; break; } }
  }
  if (fits_in_byte) {
    std::vector<std::uint8_t> ab(a.begin(), a.end());
    std::vector<std::uint8_t> bb(b.begin(), b.end());
    return wratio_engine<std::uint8_t>(
        std::span<const std::uint8_t>(ab),
        std::span<const std::uint8_t>(bb),
        score_cutoff);
  }
  return wratio_engine<Codepoint>(
      std::span<const Codepoint>(a),
      std::span<const Codepoint>(b),
      score_cutoff);
}

}  // namespace stride_align::wratio
