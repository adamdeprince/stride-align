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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
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

    // HP and HN top bits are mutually exclusive under Myers/Hyyrö.
    score += static_cast<std::size_t>((last_hp & top_bit_last) != 0U);
    score -= static_cast<std::size_t>((last_hn & top_bit_last) != 0U);
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
  const std::size_t n = text.size();
  if (m == 0) {
    return n;
  }
  if (n == 0) {
    return m;
  }
  // Triangle lower bound: distance >= |m - n|. Bail before the 2 KB
  // PEQ build when a finite cutoff already rules the pair out.
  if (cutoff != kNoCutoff) {
    const std::size_t len_diff = m > n ? m - n : n - m;
    if (len_diff > cutoff) {
      return cutoff + 1U;
    }
  }

  std::uint64_t peq[256] = {0};
  const std::uint64_t one = 1;
  for (std::size_t i = 0; i < m; ++i) {
    peq[pattern[i]] |= one << i;
  }

  const std::size_t shift = m - 1U;
  std::uint64_t vp = (m == 64U)
      ? ~std::uint64_t{0}
      : ((one << m) - 1);
  std::uint64_t vn = 0;
  std::size_t score = m;

  // Dual loops: the common no-cutoff path skips the early-exit branch
  // and uses a branchless score delta (HP/HN top bits are exclusive).
  if (cutoff == kNoCutoff) {
    for (const std::uint8_t c : text) {
      const std::uint64_t eq = peq[c];
      const std::uint64_t x = eq | vn;
      const std::uint64_t d0 = (((x & vp) + vp) ^ vp) | x;
      const std::uint64_t hp = vn | ~(d0 | vp);
      const std::uint64_t hn = d0 & vp;
      score += static_cast<std::size_t>((hp >> shift) & 1U);
      score -= static_cast<std::size_t>((hn >> shift) & 1U);
      const std::uint64_t hp_shift = (hp << 1) | one;
      const std::uint64_t hn_shift = (hn << 1);
      vp = hn_shift | ~(d0 | hp_shift);
      vn = d0 & hp_shift;
    }
    return score;
  }

  std::size_t k = 0;
  for (const std::uint8_t c : text) {
    const std::uint64_t eq = peq[c];
    const std::uint64_t x = eq | vn;
    const std::uint64_t d0 = (((x & vp) + vp) ^ vp) | x;
    const std::uint64_t hp = vn | ~(d0 | vp);
    const std::uint64_t hn = d0 & vp;
    score += static_cast<std::size_t>((hp >> shift) & 1U);
    score -= static_cast<std::size_t>((hn >> shift) & 1U);
    const std::uint64_t hp_shift = (hp << 1) | one;
    const std::uint64_t hn_shift = (hn << 1);
    vp = hn_shift | ~(d0 | hp_shift);
    vn = d0 & hp_shift;
    ++k;
    if (score > cutoff + (n - k)) {
      return cutoff + 1U;
    }
  }
  return score;
}

// Thread-local PEQ scratch for the multi-word Myers kernel. Same
// idea as ``indel.hpp::MultiWordU8Scratch`` — amortise the 256*K
// PEQ buffer allocation across calls. Dirty-symbol clearing avoids
// a full 256×K zero on every call when the alphabet is small.
struct MultiWordLevScratch {
  std::vector<std::uint64_t> peq;
  std::vector<std::uint64_t> vp;
  std::vector<std::uint64_t> vn;
  std::size_t k = 0;

  std::vector<std::uint8_t> peq_dirty;
  std::array<std::uint32_t, 256> peq_touch{};
  std::uint32_t peq_gen = 0;
  std::size_t peq_layout_k = 0;

  void resize_for(std::size_t K) {
    if (K > k) {
      peq.assign(256U * K, 0U);
      vp.resize(K);
      vn.resize(K);
      peq_dirty.clear();
      peq_layout_k = K;
      peq_touch.fill(0);
      peq_gen = 0;
      k = K;
    }
  }

  void peq_begin(std::size_t K) {
    resize_for(K);
    if (peq_layout_k != K) {
      std::fill_n(peq.data(), 256U * K, std::uint64_t{0});
      peq_dirty.clear();
      peq_layout_k = K;
    } else {
      for (const std::uint8_t s : peq_dirty) {
        std::fill_n(
            peq.data() + static_cast<std::size_t>(s) * K, K, std::uint64_t{0});
      }
      peq_dirty.clear();
    }
    if (++peq_gen == 0U) {
      peq_touch.fill(0);
      peq_gen = 1U;
    }
  }

  void peq_or_bit(std::uint8_t c, std::size_t bit_index, std::size_t K) {
    if (peq_touch[c] != peq_gen) {
      peq_touch[c] = peq_gen;
      peq_dirty.push_back(c);
    }
    peq[static_cast<std::size_t>(c) * K + (bit_index >> 6U)] |=
        std::uint64_t{1} << (bit_index & 63U);
  }

  void peq_build(std::span<const std::uint8_t> pattern, std::size_t K) {
    peq_begin(K);
    for (std::size_t i = 0; i < pattern.size(); ++i) {
      peq_or_bit(pattern[i], i, K);
    }
  }
};

inline MultiWordLevScratch& multi_word_lev_scratch() {
  thread_local MultiWordLevScratch s;
  return s;
}

// Forward declarations: the templated dispatcher below routes
// patterns in 65..256 chars to K=2/3/4 hand-specialised kernels.
inline std::size_t myers_multi_word_k2_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) noexcept;
inline std::size_t myers_multi_word_k3_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) noexcept;
inline std::size_t myers_multi_word_k4_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) noexcept;
inline std::size_t myers_multi_word_generic_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff);

// Public multi-word entry: dispatches by pattern length to the
// hand-specialised K=2/3/4 paths (m=65..256), or the generic
// heap-state variant for longer patterns.
inline std::size_t myers_multi_word_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) {
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0) return n;
  if (n == 0) return m;
  if (cutoff != kNoCutoff) {
    const std::size_t len_diff = m > n ? m - n : n - m;
    if (len_diff > cutoff) return cutoff + 1U;
  }
  if (m <= 64U) {
    return myers_single_word_u8(pattern, text, cutoff);
  }
  if (m <= 128U) return myers_multi_word_k2_u8(pattern, text, cutoff);
  if (m <= 192U) return myers_multi_word_k3_u8(pattern, text, cutoff);
  if (m <= 256U) return myers_multi_word_k4_u8(pattern, text, cutoff);
  return myers_multi_word_generic_u8(pattern, text, cutoff);
}

// Templated K = 2..4 Myers kernel: constexpr K means stack-resident
// ``vp[K]`` / ``vn[K]`` state arrays and a fully-unrolled per-block
// loop. The compiler pins vp/vn in registers across the text scan.
//
// Carry across blocks (the ``xv + vp[b] + add_carry`` 128-bit add in
// the original myers_inner) is decomposed into two
// ``__builtin_add_overflow`` calls so the carry chain rides the
// native carry-flag instruction on every supported arch (ADCX/ADOX
// on x86, ADCS on ARM).
//
// Above K = 4 the register file (16 GPRs on x86-64) overflows because
// every block needs vp + vn (2 regs) plus per-step temporaries — the
// dispatch above falls back to the heap-state generic kernel there.
template <std::size_t K>
inline std::size_t myers_multi_word_kN_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff) noexcept {
  static_assert(K >= 2 && K <= 4, "myers_multi_word_kN_u8 covers K=2..4");
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0U) return n;
  if (n == 0U) return m;
  const std::uint64_t one = 1U;

  MultiWordLevScratch& scr = multi_word_lev_scratch();
  scr.peq_build(pattern, K);

  // Stack-resident state. ``K`` is constexpr so the compiler treats
  // these as register-promotable fixed-size arrays.
  std::uint64_t vp[K];
  std::uint64_t vn[K];
  for (std::size_t k = 0; k < K; ++k) {
    vp[k] = ~std::uint64_t{0};
    vn[k] = 0;
  }
  const std::size_t last_bits = m - (K - 1U) * 64U;  // bits in last block
  if (last_bits < 64U) {
    vp[K - 1U] = (one << last_bits) - 1U;
  }
  const std::uint64_t top_bit_last = one << (last_bits - 1U);

  std::size_t score = m;
  std::size_t k_col = 0;
  const std::uint64_t* const peq_base = scr.peq.data();

  for (const std::uint8_t c : text) {
    const std::uint64_t* peq_row = peq_base + static_cast<std::size_t>(c) * K;

    std::uint64_t add_carry = 0;
    std::uint64_t hp_carry_in = 1U;
    std::uint64_t hn_carry_in = 0;
    std::uint64_t last_hp = 0;
    std::uint64_t last_hn = 0;

    // Per-block loop. Unrolled at compile time because K is constexpr.
    for (std::size_t b = 0; b < K; ++b) {
      const std::uint64_t eq = peq_row[b];
      const std::uint64_t x = eq | vn[b];
      const std::uint64_t xv = x & vp[b];

      // 128-bit-equivalent add: xv + vp[b] + add_carry. Carries
      // propagate via __builtin_add_overflow; the sum of the two
      // overflow flags is the next block's add_carry.
      std::uint64_t s1;
      const bool c1 = __builtin_add_overflow(xv, vp[b], &s1);
      std::uint64_t sum;
      const bool c2 = __builtin_add_overflow(s1, add_carry, &sum);
      add_carry = static_cast<std::uint64_t>(c1) + static_cast<std::uint64_t>(c2);

      const std::uint64_t d0 = (sum ^ vp[b]) | x;
      const std::uint64_t hp = vn[b] | ~(d0 | vp[b]);
      const std::uint64_t hn = d0 & vp[b];

      const std::uint64_t hp_shift = (hp << 1) | hp_carry_in;
      const std::uint64_t hn_shift = (hn << 1) | hn_carry_in;
      hp_carry_in = hp >> 63;
      hn_carry_in = hn >> 63;

      vp[b] = hn_shift | ~(d0 | hp_shift);
      vn[b] = d0 & hp_shift;

      if (b == K - 1U) {
        last_hp = hp;
        last_hn = hn;
      }
    }

    score += static_cast<std::size_t>((last_hp & top_bit_last) != 0U);
    score -= static_cast<std::size_t>((last_hn & top_bit_last) != 0U);
    ++k_col;
    if (cutoff != kNoCutoff && score > cutoff + (n - k_col)) {
      return cutoff + 1U;
    }
  }
  return score;
}

inline std::size_t myers_multi_word_k2_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff) noexcept {
  return myers_multi_word_kN_u8<2>(pattern, text, cutoff);
}
inline std::size_t myers_multi_word_k3_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff) noexcept {
  return myers_multi_word_kN_u8<3>(pattern, text, cutoff);
}
inline std::size_t myers_multi_word_k4_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff) noexcept {
  return myers_multi_word_kN_u8<4>(pattern, text, cutoff);
}

// Generic-K fallback for patterns of m > 256 chars. Heap-resident
// state via the thread-local scratch's vp/vn vectors. Same fused
// per-block recurrence as the K=2..4 paths; the inner loop runs the
// runtime-K loop body instead of a compile-time-unrolled one.
inline std::size_t myers_multi_word_generic_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff) {
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0U) return n;
  if (n == 0U) return m;
  constexpr std::size_t kWord = 64U;
  const std::size_t K = (m + kWord - 1U) / kWord;
  const std::uint64_t one = 1U;

  MultiWordLevScratch& scr = multi_word_lev_scratch();
  scr.peq_build(pattern, K);

  std::uint64_t* const vp = scr.vp.data();
  std::uint64_t* const vn = scr.vn.data();
  for (std::size_t k = 0; k < K; ++k) {
    vp[k] = ~std::uint64_t{0};
    vn[k] = 0;
  }
  const std::size_t last_bits = m - (K - 1U) * kWord;
  if (last_bits < kWord) {
    vp[K - 1U] = (one << last_bits) - 1U;
  }
  const std::uint64_t top_bit_last = one << (last_bits - 1U);

  std::size_t score = m;
  std::size_t k_col = 0;
  const std::uint64_t* const peq_base = scr.peq.data();

  for (const std::uint8_t c : text) {
    const std::uint64_t* peq_row = peq_base + static_cast<std::size_t>(c) * K;

    std::uint64_t add_carry = 0;
    std::uint64_t hp_carry_in = 1U;
    std::uint64_t hn_carry_in = 0;
    std::uint64_t last_hp = 0;
    std::uint64_t last_hn = 0;

    for (std::size_t b = 0; b < K; ++b) {
      const std::uint64_t eq = peq_row[b];
      const std::uint64_t x = eq | vn[b];
      const std::uint64_t xv = x & vp[b];

      std::uint64_t s1;
      const bool c1 = __builtin_add_overflow(xv, vp[b], &s1);
      std::uint64_t sum;
      const bool c2 = __builtin_add_overflow(s1, add_carry, &sum);
      add_carry = static_cast<std::uint64_t>(c1) + static_cast<std::uint64_t>(c2);

      const std::uint64_t d0 = (sum ^ vp[b]) | x;
      const std::uint64_t hp = vn[b] | ~(d0 | vp[b]);
      const std::uint64_t hn = d0 & vp[b];

      const std::uint64_t hp_shift = (hp << 1) | hp_carry_in;
      const std::uint64_t hn_shift = (hn << 1) | hn_carry_in;
      hp_carry_in = hp >> 63;
      hn_carry_in = hn >> 63;

      vp[b] = hn_shift | ~(d0 | hp_shift);
      vn[b] = d0 & hp_shift;

      if (b == K - 1U) {
        last_hp = hp;
        last_hn = hn;
      }
    }

    score += static_cast<std::size_t>((last_hp & top_bit_last) != 0U);
    score -= static_cast<std::size_t>((last_hn & top_bit_last) != 0U);
    ++k_col;
    if (cutoff != kNoCutoff && score > cutoff + (n - k_col)) {
      return cutoff + 1U;
    }
  }
  return score;
}

template <typename Token>
std::size_t myers_distance(
    std::span<const Token> pattern,
    std::span<const Token> text,
    std::size_t cutoff = kNoCutoff) {
  static_assert(std::is_integral_v<Token> || std::is_unsigned_v<Token>);
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0) {
    return n;
  }
  if (n == 0) {
    return m;
  }
  if (cutoff != kNoCutoff) {
    const std::size_t len_diff = m > n ? m - n : n - m;
    if (len_diff > cutoff) {
      return cutoff + 1U;
    }
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

  const std::size_t shift = m - 1U;
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
    // HP/HN top bits are mutually exclusive under Hyyrö OSA.
    score += static_cast<std::size_t>((hp >> shift) & 1U);
    score -= static_cast<std::size_t>((hn >> shift) & 1U);
    const std::uint64_t hp_shift = (hp << 1) | one;
    const std::uint64_t hn_shift = hn << 1;
    vp = hn_shift | ~(d0 | hp_shift);
    vn = hp_shift & d0;
    d0_prev = d0;
    pm_old = pm;
  }
  return score;
}

// Thread-local PEQ + state scratch for the multi-word OSA kernel.
// Carries the per-block ``vp``, ``vn``, plus the OSA cross-column
// state vectors ``d0_prev`` and ``pm_old``. Used by the generic-K
// fallback; the K=2/K=3 hand-spec paths keep everything on the stack.
struct MultiWordOsaScratch {
  std::vector<std::uint64_t> peq;
  std::vector<std::uint64_t> vp;
  std::vector<std::uint64_t> vn;
  std::vector<std::uint64_t> d0_prev;
  std::vector<std::uint64_t> pm_old;
  std::size_t k = 0;

  std::vector<std::uint8_t> peq_dirty;
  std::array<std::uint32_t, 256> peq_touch{};
  std::uint32_t peq_gen = 0;
  std::size_t peq_layout_k = 0;

  void resize_for(std::size_t K) {
    if (K > k) {
      peq.assign(256U * K, 0U);
      vp.resize(K);
      vn.resize(K);
      d0_prev.resize(K);
      pm_old.resize(K);
      peq_dirty.clear();
      peq_layout_k = K;
      peq_touch.fill(0);
      peq_gen = 0;
      k = K;
    }
  }

  void peq_begin(std::size_t K) {
    resize_for(K);
    if (peq_layout_k != K) {
      std::fill_n(peq.data(), 256U * K, std::uint64_t{0});
      peq_dirty.clear();
      peq_layout_k = K;
    } else {
      for (const std::uint8_t s : peq_dirty) {
        std::fill_n(
            peq.data() + static_cast<std::size_t>(s) * K, K, std::uint64_t{0});
      }
      peq_dirty.clear();
    }
    if (++peq_gen == 0U) {
      peq_touch.fill(0);
      peq_gen = 1U;
    }
  }

  void peq_or_bit(std::uint8_t c, std::size_t bit_index, std::size_t K) {
    if (peq_touch[c] != peq_gen) {
      peq_touch[c] = peq_gen;
      peq_dirty.push_back(c);
    }
    peq[static_cast<std::size_t>(c) * K + (bit_index >> 6U)] |=
        std::uint64_t{1} << (bit_index & 63U);
  }

  void peq_build(std::span<const std::uint8_t> pattern, std::size_t K) {
    peq_begin(K);
    for (std::size_t i = 0; i < pattern.size(); ++i) {
      peq_or_bit(pattern[i], i, K);
    }
  }
};

inline MultiWordOsaScratch& multi_word_osa_scratch() {
  thread_local MultiWordOsaScratch s;
  return s;
}

// Forward declarations for the K-hand-spec multi-word OSA paths.
inline std::size_t osa_multi_word_k2_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text) noexcept;
inline std::size_t osa_multi_word_k3_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text) noexcept;
inline std::size_t osa_multi_word_generic_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text);

// OSA dispatch: bit-parallel single-word for short u8 patterns,
// K-hand-spec multi-word for m = 65..192, generic multi-word for
// m > 192. The scalar DP fallback is no longer reached on the byte
// path — the new multi-word kernels (Hyyrö 2002 generalised across
// blocks with carry propagation on both the standard Myers add chain
// and the per-column transposition mask) handle every length.
inline std::size_t osa_distance_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text) {
  const std::size_t m = pattern.size();
  if (m == 0U) return text.size();
  if (text.empty()) return m;
  if (m <= 64U)   return osa_single_word_u8(pattern, text);
  if (m <= 128U)  return osa_multi_word_k2_u8(pattern, text);
  if (m <= 192U)  return osa_multi_word_k3_u8(pattern, text);
  return osa_multi_word_generic_u8(pattern, text);
}

// Templated K = 2..3 multi-word bit-parallel OSA. Constexpr ``K``
// keeps the per-block state arrays stack-resident and unrolls the
// inner block loop. State per block: ``vp[k]``, ``vn[k]``,
// ``d0_prev[k]``, ``pm_old[k]`` — 4 words/block, so K = 2 fits the
// x86-64 GPR file comfortably, K = 3 is tight but still register-
// resident on modern compilers.
//
// Carries propagated across blocks within one text column:
//   add_carry — 128-bit ``(pm & vp) + vp + add_carry`` add chain
//   hp_carry_in / hn_carry_in — left-shift of hp / hn
//   trans_carry_in — left-shift of the transposition pre-mask
//
// Block 0 of the hp shift initialises ``hp_carry_in = 1`` (matches
// the standard single-word kernel's ``| 1``). All other carries
// init to 0.
template <std::size_t K>
inline std::size_t osa_multi_word_kN_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text) noexcept {
  static_assert(K >= 2 && K <= 3, "osa_multi_word_kN_u8 covers K=2..3");
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0U) return n;
  if (n == 0U) return m;
  const std::uint64_t one = 1U;

  MultiWordOsaScratch& scr = multi_word_osa_scratch();
  scr.peq_build(pattern, K);

  std::uint64_t vp[K];
  std::uint64_t vn[K];
  std::uint64_t d0_prev[K];
  std::uint64_t pm_old[K];
  for (std::size_t k = 0; k < K; ++k) {
    vp[k] = ~std::uint64_t{0};
    vn[k] = 0;
    d0_prev[k] = 0;
    pm_old[k] = 0;
  }
  const std::size_t last_bits = m - (K - 1U) * 64U;
  if (last_bits < 64U) {
    vp[K - 1U] = (one << last_bits) - 1U;
  }
  const std::uint64_t top_bit_last = one << (last_bits - 1U);

  std::size_t score = m;
  const std::uint64_t* const peq_base = scr.peq.data();

  for (const std::uint8_t c : text) {
    const std::uint64_t* peq_row = peq_base + static_cast<std::size_t>(c) * K;

    std::uint64_t add_carry = 0;
    std::uint64_t hp_carry_in = 1U;
    std::uint64_t hn_carry_in = 0;
    std::uint64_t trans_carry_in = 0;
    std::uint64_t last_hp = 0;
    std::uint64_t last_hn = 0;

    for (std::size_t b = 0; b < K; ++b) {
      const std::uint64_t pm = peq_row[b];

      // Transposition mask: ``trans = (((~d0_prev) & pm) << 1) & pm_old``
      // with the ``<< 1`` shift carry crossing block boundaries.
      const std::uint64_t trans_premask = (~d0_prev[b]) & pm;
      const std::uint64_t trans_shift = (trans_premask << 1) | trans_carry_in;
      trans_carry_in = trans_premask >> 63;
      const std::uint64_t trans = trans_shift & pm_old[b];

      // Hyyrö single-step ``d0`` derivation, extended across blocks
      // with an add carry chain on ``(pm & vp) + vp``.
      const std::uint64_t pm_and_vp = pm & vp[b];
      std::uint64_t s1;
      const bool c1 = __builtin_add_overflow(pm_and_vp, vp[b], &s1);
      std::uint64_t sum;
      const bool c2 = __builtin_add_overflow(s1, add_carry, &sum);
      add_carry = static_cast<std::uint64_t>(c1) + static_cast<std::uint64_t>(c2);

      std::uint64_t d0 = (sum ^ vp[b]) | pm | vn[b];
      d0 |= trans;

      const std::uint64_t hp = vn[b] | ~(d0 | vp[b]);
      const std::uint64_t hn = d0 & vp[b];

      const std::uint64_t hp_shift = (hp << 1) | hp_carry_in;
      const std::uint64_t hn_shift = (hn << 1) | hn_carry_in;
      hp_carry_in = hp >> 63;
      hn_carry_in = hn >> 63;

      vp[b] = hn_shift | ~(d0 | hp_shift);
      vn[b] = d0 & hp_shift;

      d0_prev[b] = d0;
      pm_old[b] = pm;

      if (b == K - 1U) {
        last_hp = hp;
        last_hn = hn;
      }
    }

    score += static_cast<std::size_t>((last_hp & top_bit_last) != 0U);
    score -= static_cast<std::size_t>((last_hn & top_bit_last) != 0U);
  }
  return score;
}

inline std::size_t osa_multi_word_k2_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text) noexcept {
  return osa_multi_word_kN_u8<2>(pattern, text);
}
inline std::size_t osa_multi_word_k3_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text) noexcept {
  return osa_multi_word_kN_u8<3>(pattern, text);
}

// Generic-K fallback for m > 192. Same fused per-block recurrence as
// the K=2..3 paths but with a runtime K loop and heap-resident state.
inline std::size_t osa_multi_word_generic_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text) {
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0U) return n;
  if (n == 0U) return m;
  constexpr std::size_t kWord = 64U;
  const std::size_t K = (m + kWord - 1U) / kWord;
  const std::uint64_t one = 1U;

  MultiWordOsaScratch& scr = multi_word_osa_scratch();
  scr.peq_build(pattern, K);

  std::uint64_t* const vp = scr.vp.data();
  std::uint64_t* const vn = scr.vn.data();
  std::uint64_t* const d0_prev = scr.d0_prev.data();
  std::uint64_t* const pm_old = scr.pm_old.data();
  for (std::size_t k = 0; k < K; ++k) {
    vp[k] = ~std::uint64_t{0};
    vn[k] = 0;
    d0_prev[k] = 0;
    pm_old[k] = 0;
  }
  const std::size_t last_bits = m - (K - 1U) * kWord;
  if (last_bits < kWord) {
    vp[K - 1U] = (one << last_bits) - 1U;
  }
  const std::uint64_t top_bit_last = one << (last_bits - 1U);

  std::size_t score = m;
  const std::uint64_t* const peq_base = scr.peq.data();

  for (const std::uint8_t c : text) {
    const std::uint64_t* peq_row = peq_base + static_cast<std::size_t>(c) * K;

    std::uint64_t add_carry = 0;
    std::uint64_t hp_carry_in = 1U;
    std::uint64_t hn_carry_in = 0;
    std::uint64_t trans_carry_in = 0;
    std::uint64_t last_hp = 0;
    std::uint64_t last_hn = 0;

    for (std::size_t b = 0; b < K; ++b) {
      const std::uint64_t pm = peq_row[b];

      const std::uint64_t trans_premask = (~d0_prev[b]) & pm;
      const std::uint64_t trans_shift = (trans_premask << 1) | trans_carry_in;
      trans_carry_in = trans_premask >> 63;
      const std::uint64_t trans = trans_shift & pm_old[b];

      const std::uint64_t pm_and_vp = pm & vp[b];
      std::uint64_t s1;
      const bool c1 = __builtin_add_overflow(pm_and_vp, vp[b], &s1);
      std::uint64_t sum;
      const bool c2 = __builtin_add_overflow(s1, add_carry, &sum);
      add_carry = static_cast<std::uint64_t>(c1) + static_cast<std::uint64_t>(c2);

      std::uint64_t d0 = (sum ^ vp[b]) | pm | vn[b];
      d0 |= trans;

      const std::uint64_t hp = vn[b] | ~(d0 | vp[b]);
      const std::uint64_t hn = d0 & vp[b];

      const std::uint64_t hp_shift = (hp << 1) | hp_carry_in;
      const std::uint64_t hn_shift = (hn << 1) | hn_carry_in;
      hp_carry_in = hp >> 63;
      hn_carry_in = hn >> 63;

      vp[b] = hn_shift | ~(d0 | hp_shift);
      vn[b] = d0 & hp_shift;

      d0_prev[b] = d0;
      pm_old[b] = pm;

      if (b == K - 1U) {
        last_hp = hp;
        last_hn = hn;
      }
    }

    score += static_cast<std::size_t>((last_hp & top_bit_last) != 0U);
    score -= static_cast<std::size_t>((last_hn & top_bit_last) != 0U);
  }
  return score;
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

  const std::size_t shift = m - 1U;
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
    score += static_cast<std::size_t>((hp >> shift) & 1U);
    score -= static_cast<std::size_t>((hn >> shift) & 1U);
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

// Byte-specialised Lowrance-Wagner DP: flat ``(n+2) * (m+2)`` table
// (single allocation, contiguous memory) and ``std::array<size_t, 256>``
// for the "last-seen-row-per-character" tracker (no unordered_map
// hash / chain walk per cell). On ASCII inputs the cell cost drops
// from ~140 ns to ~30 ns each, which is the bulk of the
// rapidfuzz-shim ``DamerauLevenshtein.distance`` gap on
// medium-length pairs.
inline std::size_t true_damerau_levenshtein_distance_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text) {
  const std::size_t n = pattern.size();
  const std::size_t m = text.size();
  if (n == 0U) return m;
  if (m == 0U) return n;

  const std::size_t inf = n + m;
  const std::size_t stride = m + 2U;

  // Thread-local scratch — amortises the (n+2)*(m+2) allocation
  // across calls. Reset to ``inf`` only where the recurrence reads
  // the sentinel rows / columns.
  thread_local std::vector<std::size_t> h;
  h.assign((n + 2U) * stride, 0);

  h[0 * stride + 0] = inf;
  for (std::size_t i = 0; i <= n; ++i) {
    h[(i + 1U) * stride + 0] = inf;
    h[(i + 1U) * stride + 1] = i;
  }
  for (std::size_t j = 0; j <= m; ++j) {
    h[0 * stride + (j + 1U)] = inf;
    h[1 * stride + (j + 1U)] = j;
  }

  // ``da`` maps each byte to the last row it appeared in
  // (1-indexed). 0 means "not yet seen". 256 entries, fits in 2 KiB —
  // hot in L1 across the whole call.
  thread_local std::array<std::size_t, 256> da_buf;
  da_buf.fill(0);

  for (std::size_t i = 1; i <= n; ++i) {
    std::size_t db = 0;
    const std::uint8_t pi = pattern[i - 1U];
    const std::size_t* const row_prev = h.data() + i * stride;
    std::size_t* const row_curr = h.data() + (i + 1U) * stride;
    for (std::size_t j = 1; j <= m; ++j) {
      const std::uint8_t tj = text[j - 1U];
      const std::size_t k = da_buf[tj];
      const std::size_t l = db;

      std::size_t cost;
      if (pi == tj) {
        cost = 0;
        db = j;
      } else {
        cost = 1;
      }

      const std::size_t sub = row_prev[j] + cost;
      const std::size_t ins = row_curr[j] + 1U;
      const std::size_t del = row_prev[j + 1U] + 1U;
      const std::size_t trans = h[k * stride + l] +
          (i > k ? (i - k - 1U) : 0U) + 1U +
          (j > l ? (j - l - 1U) : 0U);

      std::size_t best = sub;
      if (ins < best) best = ins;
      if (del < best) best = del;
      if (trans < best) best = trans;
      row_curr[j + 1U] = best;
    }
    da_buf[pi] = i;
  }
  return h[(n + 1U) * stride + (m + 1U)];
}

}  // namespace stride_align::levenshtein
