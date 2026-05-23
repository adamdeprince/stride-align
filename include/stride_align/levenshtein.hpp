#pragma once

// Bit-parallel Levenshtein distance (Myers 1999 + Hyyrö 2003).
//
//   * myers_single_word_u8: pattern of length <= 64 over a uint8 alphabet.
//     One 64-bit add computes 64 cells of the DP row in parallel. This is
//     the hottest path for short bytes/ASCII queries.
//   * myers_multi_word_u8:  pattern of arbitrary length over uint8. The
//     pattern is split into ceil(m / 64) blocks; carries propagate between
//     blocks both through the inner-loop addition and through the
//     horizontal-shift step.
//   * myers_distance<Token>: hashmap-PEQ variant for any token type (used
//     for Python str inputs whose code-point alphabet exceeds 256). Same
//     bit-parallel inner loop, just a slower PEQ lookup.
//
// SIMD specialization (one target per SIMD lane, lane width 64) lives in
// each x86 backend header.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <unordered_map>
#include <vector>

#include "stride_align/alignment.hpp"

namespace stride_align::levenshtein {

// Sentinel meaning "no cutoff applied". The Myers entry points and the
// SIMD batch path accept a cutoff; when it equals kNoCutoff every
// column runs to completion and the exact distance is returned. When a
// finite cutoff is supplied and the algorithm proves the final score
// must exceed it, the call returns `cutoff + 1` (matching rapidfuzz's
// convention) without finishing the remaining columns.
inline constexpr std::size_t kNoCutoff = std::numeric_limits<std::size_t>::max();

namespace detail {

// Run the Myers/Hyyrö bit-parallel inner loop over a `text`, with PEQ lookup
// supplied by `peq_for(c)` returning a span of B `std::uint64_t` per text
// character. `B` is the number of 64-bit blocks the pattern is split into.
// If `cutoff` is not kNoCutoff and we can prove final >= cutoff + 1, the
// loop returns cutoff + 1 early.
template <typename Text, typename PeqFn>
inline std::size_t myers_inner(
    std::size_t m,
    std::size_t B,
    Text text,
    PeqFn peq_for,
    std::uint64_t* vp,
    std::uint64_t* vn,
    std::uint64_t top_bit_last,
    std::size_t initial_score,
    std::size_t cutoff = kNoCutoff) noexcept(noexcept(peq_for(typename Text::value_type{}))) {
  std::size_t score = initial_score;
  // Track remaining columns so we can prove min-final-score = score -
  // remaining and bail when it already exceeds cutoff.
  std::size_t k = 0;
  const std::size_t n = text.size();
  for (const auto c : text) {
    auto eq_blocks = peq_for(c);
    std::uint64_t add_carry = 0;
    std::uint64_t hp_carry_in = 1U;
    std::uint64_t hn_carry_in = 0;
    std::uint64_t last_hp = 0;
    std::uint64_t last_hn = 0;

    for (std::size_t b = 0; b < B; ++b) {
      const std::uint64_t eq = eq_blocks[b];
      const std::uint64_t x = eq | vn[b];
      const std::uint64_t xv = x & vp[b];

      const __uint128_t wide =
          static_cast<__uint128_t>(xv) +
          static_cast<__uint128_t>(vp[b]) +
          static_cast<__uint128_t>(add_carry);
      const std::uint64_t sum = static_cast<std::uint64_t>(wide);
      add_carry = static_cast<std::uint64_t>(wide >> 64);

      const std::uint64_t d0 = (sum ^ vp[b]) | x;
      const std::uint64_t hp = vn[b] | ~(d0 | vp[b]);
      const std::uint64_t hn = d0 & vp[b];

      const std::uint64_t hp_shift = (hp << 1) | hp_carry_in;
      const std::uint64_t hn_shift = (hn << 1) | hn_carry_in;
      hp_carry_in = hp >> 63;
      hn_carry_in = hn >> 63;

      vp[b] = hn_shift | ~(d0 | hp_shift);
      vn[b] = d0 & hp_shift;

      if (b == B - 1U) {
        last_hp = hp;
        last_hn = hn;
      }
    }

    if (last_hp & top_bit_last) {
      ++score;
    } else if (last_hn & top_bit_last) {
      --score;
    }
    (void)m;
    ++k;
    // Min possible final score is score - (n - k). Bail if even the
    // best-case can't reach cutoff.
    if (cutoff != kNoCutoff && score > cutoff + (n - k)) {
      return cutoff + 1U;
    }
  }
  return score;
}

inline void init_vp_vn(
    std::vector<std::uint64_t>& vp,
    std::vector<std::uint64_t>& vn,
    std::size_t m) {
  constexpr std::size_t kWord = 64U;
  const std::size_t B = (m + kWord - 1U) / kWord;
  vp.assign(B, ~std::uint64_t{0});
  vn.assign(B, 0);
  const std::size_t last_bits = m - (B - 1U) * kWord;
  if (last_bits < kWord) {
    vp.back() = (std::uint64_t{1} << last_bits) - 1U;
  }
}

}  // namespace detail

inline std::size_t myers_single_word_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) noexcept {
  const std::size_t m = pattern.size();
  if (m == 0) {
    return text.size();
  }
  if (text.empty()) {
    return m;
  }

  std::uint64_t peq[256] = {0};
  const std::uint64_t one = 1;
  for (std::size_t i = 0; i < m; ++i) {
    peq[pattern[i]] |= one << i;
  }

  const std::uint64_t top_bit = one << (m - 1);
  std::uint64_t vp = (m == 64U)
      ? ~std::uint64_t{0}
      : ((one << m) - 1);
  std::uint64_t vn = 0;
  std::size_t score = m;

  const std::size_t n = text.size();
  std::size_t k = 0;
  for (const std::uint8_t c : text) {
    const std::uint64_t eq = peq[c];
    const std::uint64_t x = eq | vn;
    const std::uint64_t d0 = (((x & vp) + vp) ^ vp) | x;
    const std::uint64_t hp = vn | ~(d0 | vp);
    const std::uint64_t hn = d0 & vp;
    if (hp & top_bit) {
      ++score;
    } else if (hn & top_bit) {
      --score;
    }
    const std::uint64_t hp_shift = (hp << 1) | one;
    const std::uint64_t hn_shift = (hn << 1);
    vp = hn_shift | ~(d0 | hp_shift);
    vn = d0 & hp_shift;
    ++k;
    if (cutoff != kNoCutoff && score > cutoff + (n - k)) {
      return cutoff + 1U;
    }
  }
  return score;
}

inline std::size_t myers_multi_word_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) {
  const std::size_t m = pattern.size();
  if (m == 0) {
    return text.size();
  }
  if (text.empty()) {
    return m;
  }
  if (m <= 64U) {
    return myers_single_word_u8(pattern, text, cutoff);
  }

  constexpr std::size_t kWord = 64U;
  const std::size_t B = (m + kWord - 1U) / kWord;

  std::vector<std::uint64_t> peq(static_cast<std::size_t>(256) * B, 0);
  const std::uint64_t one = 1;
  for (std::size_t i = 0; i < m; ++i) {
    peq[static_cast<std::size_t>(pattern[i]) * B + (i / kWord)] |=
        one << (i % kWord);
  }

  std::vector<std::uint64_t> vp;
  std::vector<std::uint64_t> vn;
  detail::init_vp_vn(vp, vn, m);

  const std::size_t last_bits = m - (B - 1U) * kWord;
  const std::uint64_t top_bit_last = std::uint64_t{1} << (last_bits - 1U);

  return detail::myers_inner(
      m,
      B,
      text,
      [&](std::uint8_t c) {
        return std::span<const std::uint64_t>(
            peq.data() + static_cast<std::size_t>(c) * B, B);
      },
      vp.data(),
      vn.data(),
      top_bit_last,
      m,
      cutoff);
}

template <typename Token>
std::size_t myers_distance(
    std::span<const Token> pattern,
    std::span<const Token> text,
    std::size_t cutoff = kNoCutoff) {
  static_assert(std::is_integral_v<Token> || std::is_unsigned_v<Token>);
  const std::size_t m = pattern.size();
  if (m == 0) {
    return text.size();
  }
  if (text.empty()) {
    return m;
  }

  constexpr std::size_t kWord = 64U;
  const std::size_t B = (m + kWord - 1U) / kWord;

  std::unordered_map<Token, std::vector<std::uint64_t>> peq;
  const std::uint64_t one = 1;
  for (std::size_t i = 0; i < m; ++i) {
    auto [it, inserted] = peq.try_emplace(pattern[i], std::vector<std::uint64_t>(B, 0));
    it->second[i / kWord] |= one << (i % kWord);
  }

  // Zero-row for tokens that never appear in the pattern. Sized once per
  // call so all per-call lambdas see the right B.
  const std::vector<std::uint64_t> zero_blocks(B, 0);

  std::vector<std::uint64_t> vp;
  std::vector<std::uint64_t> vn;
  detail::init_vp_vn(vp, vn, m);

  const std::size_t last_bits = m - (B - 1U) * kWord;
  const std::uint64_t top_bit_last = std::uint64_t{1} << (last_bits - 1U);

  return detail::myers_inner(
      m,
      B,
      text,
      [&](Token c) {
        auto it = peq.find(c);
        if (it == peq.end()) {
          return std::span<const std::uint64_t>(zero_blocks.data(), B);
        }
        return std::span<const std::uint64_t>(it->second.data(), B);
      },
      vp.data(),
      vn.data(),
      top_bit_last,
      m,
      cutoff);
}

// =================================================================
// Optimal String Alignment (OSA) distance — a.k.a. "restricted
// Damerau-Levenshtein". Same as Levenshtein but adjacent transpositions
// cost 1 instead of 2 substitutions. "Restricted" means each character
// can participate in at most one edit operation, so a transposition
// can't be combined with another edit on the same characters.
//
// Most Python users who ask for "Damerau-Levenshtein" actually want
// OSA — it's what rapidfuzz exposes as `OSA.distance` and is much
// faster to compute than true Damerau-Levenshtein (which needs an
// alphabet-sized auxiliary array per cell).
// =================================================================

namespace detail {

// Scalar DP reference implementation. O(m*n) time, O(m) space (rolling
// rows). Correctness oracle for the bit-parallel variants below; not on
// any hot path.
template <typename Token>
inline std::size_t osa_dp(
    std::span<const Token> pattern,
    std::span<const Token> text) {
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0) {
    return n;
  }
  if (n == 0) {
    return m;
  }

  // Three rolling rows: prev2[i] = d[i][j-2], prev1[i] = d[i][j-1],
  // curr[i] = d[i][j]. The transposition step reads d[i-2][j-2] which
  // is prev2[i-2].
  std::vector<std::size_t> prev2(m + 1U);
  std::vector<std::size_t> prev1(m + 1U);
  std::vector<std::size_t> curr(m + 1U);
  for (std::size_t i = 0; i <= m; ++i) {
    prev1[i] = i;
  }

  for (std::size_t j = 1; j <= n; ++j) {
    curr[0] = j;
    for (std::size_t i = 1; i <= m; ++i) {
      const std::size_t sub_cost =
          (pattern[i - 1U] == text[j - 1U]) ? 0U : 1U;
      std::size_t best = curr[i - 1U] + 1U;            // insertion
      best = std::min(best, prev1[i] + 1U);             // deletion
      best = std::min(best, prev1[i - 1U] + sub_cost);  // substitution
      if (i >= 2U && j >= 2U &&
          pattern[i - 1U] == text[j - 2U] &&
          pattern[i - 2U] == text[j - 1U]) {
        best = std::min(best, prev2[i - 2U] + 1U);  // transposition
      }
      curr[i] = best;
    }
    std::swap(prev2, prev1);
    std::swap(prev1, curr);
  }
  return prev1[m];
}

}  // namespace detail

// Bit-parallel OSA for uint8 patterns of length <= 64 (Hyyrö 2002).
// Augments Myers' Levenshtein recurrence with a transposition mask:
//   TR = (((~D0_prev) & PM) << 1) & PM_old
// Bit i of TR is set iff the (i-1, j-1) cell was *not* a "diagonal
// match" in the previous column AND P[i-1] == T[j] AND P[i] == T[j-1]
// — i.e. exactly the OSA transposition condition with the right
// predecessor state. The ~D0_prev gate is the non-obvious bit that
// keeps the algorithm correct under OSA's "each character touched at
// most once" restriction; without it, the kernel double-counts edits.
inline std::size_t osa_single_word_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text) noexcept {
  const std::size_t m = pattern.size();
  if (m == 0) {
    return text.size();
  }
  if (text.empty()) {
    return m;
  }

  std::uint64_t peq[256] = {0};
  const std::uint64_t one = 1U;
  for (std::size_t i = 0; i < m; ++i) {
    peq[pattern[i]] |= one << i;
  }

  const std::uint64_t top_bit = one << (m - 1U);
  std::uint64_t vp = (m == 64U)
      ? ~std::uint64_t{0}
      : ((one << m) - 1U);
  std::uint64_t vn = 0;
  std::uint64_t d0_prev = 0;
  std::uint64_t pm_old = 0;
  std::size_t score = m;

  for (const std::uint8_t c : text) {
    const std::uint64_t pm = peq[c];
    const std::uint64_t trans = (((~d0_prev) & pm) << 1) & pm_old;
    std::uint64_t d0 = (((pm & vp) + vp) ^ vp) | pm | vn;
    d0 |= trans;

    const std::uint64_t hp = vn | ~(d0 | vp);
    const std::uint64_t hn = d0 & vp;
    if (hp & top_bit) {
      ++score;
    } else if (hn & top_bit) {
      --score;
    }
    const std::uint64_t hp_shift = (hp << 1) | one;
    const std::uint64_t hn_shift = hn << 1;
    vp = hn_shift | ~(d0 | hp_shift);
    vn = hp_shift & d0;
    d0_prev = d0;
    pm_old = pm;
  }
  return score;
}

// OSA dispatch: bit-parallel single-word for short u8 patterns, scalar
// DP otherwise. A bit-parallel multi-word OSA (analogous to Hyyrö's
// multi-word Levenshtein) is doable but deferred — needs per-block
// carry propagation for the transposition mask as well as for the
// standard d0 chain.
inline std::size_t osa_distance_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text) {
  if (pattern.size() > 0 && pattern.size() <= 64U) {
    return osa_single_word_u8(pattern, text);
  }
  return detail::osa_dp<std::uint8_t>(pattern, text);
}

template <typename Token>
inline std::size_t osa_distance(
    std::span<const Token> pattern,
    std::span<const Token> text) {
  static_assert(std::is_integral_v<Token> || std::is_unsigned_v<Token>);
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0) {
    return n;
  }
  if (n == 0) {
    return m;
  }
  if (m > 64U) {
    return detail::osa_dp<Token>(pattern, text);
  }

  // Bit-parallel with hashmap PEQ for wide alphabets.
  std::unordered_map<Token, std::uint64_t> peq;
  const std::uint64_t one = 1U;
  for (std::size_t i = 0; i < m; ++i) {
    peq[pattern[i]] |= one << i;
  }

  const std::uint64_t top_bit = one << (m - 1U);
  std::uint64_t vp = (m == 64U) ? ~std::uint64_t{0} : ((one << m) - 1U);
  std::uint64_t vn = 0;
  std::uint64_t d0_prev = 0;
  std::uint64_t pm_old = 0;
  std::size_t score = m;

  for (const auto c : text) {
    auto it = peq.find(c);
    const std::uint64_t pm = (it == peq.end()) ? 0U : it->second;
    const std::uint64_t trans = (((~d0_prev) & pm) << 1) & pm_old;
    std::uint64_t d0 = (((pm & vp) + vp) ^ vp) | pm | vn;
    d0 |= trans;

    const std::uint64_t hp = vn | ~(d0 | vp);
    const std::uint64_t hn = d0 & vp;
    if (hp & top_bit) {
      ++score;
    } else if (hn & top_bit) {
      --score;
    }
    const std::uint64_t hp_shift = (hp << 1) | one;
    const std::uint64_t hn_shift = hn << 1;
    vp = hn_shift | ~(d0 | hp_shift);
    vn = hp_shift & d0;
    d0_prev = d0;
    pm_old = pm;
  }
  return score;
}

inline double normalize(std::size_t distance, std::size_t a_len, std::size_t b_len) noexcept {
  const std::size_t longer = (a_len > b_len) ? a_len : b_len;
  if (longer == 0U) {
    return 1.0;
  }
  const double ratio =
      static_cast<double>(distance) / static_cast<double>(longer);
  if (ratio >= 1.0) {
    return 0.0;
  }
  return 1.0 - ratio;
}

// =================================================================
// True Damerau-Levenshtein distance — the unrestricted form. Same
// edits as OSA (insertion, deletion, substitution, transposition of
// adjacent characters) but a character can participate in more than
// one edit. The standard example where OSA and true DL differ is
// "CA" -> "ABC":
//   OSA = 3 (no useful transposition; each edit fights for a single
//            character)
//   true DL = 2 (transpose C,A then insert B — though both edits
//                touch position 0)
//
// The algorithm uses an (n+2) by (m+2) DP table plus an alphabet-
// indexed "last row of each character" map. O(n*m) time, O(n*m + |Σ|)
// space. No bit-parallel form is shipped here — Hyyrö 2003 has one
// but it's significantly more complex than the OSA bit-parallel and
// rarely the bottleneck in practice. SIMD batching falls back to
// per-pair scalar dispatch.
//
// Rationale for keeping both: most callers asking for
// "Damerau-Levenshtein" actually want OSA (faster, equivalent in
// almost every realistic input), but a strict minority needs the
// unrestricted variant.
// =================================================================

namespace detail {

template <typename Token>
inline std::size_t true_damerau_dp(
    std::span<const Token> pattern,
    std::span<const Token> text) {
  const std::size_t n = pattern.size();
  const std::size_t m = text.size();
  if (n == 0U) return m;
  if (m == 0U) return n;

  const std::size_t inf = n + m;

  // (n+2) by (m+2) DP table. Rows 0 and column 0 are sentinels
  // populated with `inf` so the transposition lookup is never the
  // minimum when there's no valid prior occurrence.
  std::vector<std::vector<std::size_t>> h(
      n + 2U, std::vector<std::size_t>(m + 2U, 0));
  h[0][0] = inf;
  for (std::size_t i = 0; i <= n; ++i) {
    h[i + 1U][0] = inf;
    h[i + 1U][1] = i;
  }
  for (std::size_t j = 0; j <= m; ++j) {
    h[0][j + 1U] = inf;
    h[1][j + 1U] = j;
  }

  // `da` maps each character to the last row in which it appeared
  // (1-indexed; 0 means "not yet seen"). Using std::unordered_map so
  // the algorithm works for arbitrary integral tokens, not just bytes.
  std::unordered_map<Token, std::size_t> da;

  for (std::size_t i = 1; i <= n; ++i) {
    std::size_t db = 0;
    for (std::size_t j = 1; j <= m; ++j) {
      const Token tj = text[j - 1U];
      const std::size_t k = [&] {
        auto it = da.find(tj);
        return it == da.end() ? std::size_t{0} : it->second;
      }();
      const std::size_t l = db;
      std::size_t cost;
      if (pattern[i - 1U] == tj) {
        cost = 0;
        db = j;
      } else {
        cost = 1;
      }
      const std::size_t sub = h[i][j] + cost;
      const std::size_t ins = h[i + 1U][j] + 1U;
      const std::size_t del = h[i][j + 1U] + 1U;
      const std::size_t trans = h[k][l] +
          (i > k ? (i - k - 1U) : 0U) + 1U +
          (j > l ? (j - l - 1U) : 0U);
      // Note: when k == 0 or l == 0, h[k][l] = inf (or h[k][...] = j
      // which is large enough to dominate any other option).
      h[i + 1U][j + 1U] =
          std::min({sub, ins, del, trans});
    }
    da[pattern[i - 1U]] = i;
  }
  return h[n + 1U][m + 1U];
}

}  // namespace detail

template <typename Token>
inline std::size_t true_damerau_levenshtein_distance(
    std::span<const Token> pattern,
    std::span<const Token> text) {
  static_assert(std::is_integral_v<Token> || std::is_unsigned_v<Token>);
  return detail::true_damerau_dp<Token>(pattern, text);
}

inline std::size_t true_damerau_levenshtein_distance_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text) {
  return detail::true_damerau_dp<std::uint8_t>(pattern, text);
}

}  // namespace stride_align::levenshtein
