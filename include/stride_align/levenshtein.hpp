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

}  // namespace stride_align::levenshtein
