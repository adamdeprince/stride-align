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

// Forward declarations: the dispatch helpers above and the
// templated ``indel_distance`` below need these visible before the
// concrete definitions further down in the file.
inline std::size_t indel_distance_k2_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) noexcept;
inline std::size_t indel_distance_k3_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) noexcept;
inline std::size_t indel_distance_k4_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) noexcept;
inline std::size_t indel_distance_k5_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) noexcept;
inline std::size_t indel_distance_k6_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) noexcept;
inline std::size_t indel_distance_k7_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) noexcept;
inline std::size_t indel_distance_k8_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) noexcept;

inline std::size_t indel_distance_multi_word_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff);

template <typename Token>
inline std::size_t indel_distance_multi_word(
    std::span<const Token> pattern,
    std::span<const Token> text,
    std::size_t cutoff = kNoCutoff);

// Token-generic bit-parallel for patterns of any length. Patterns of
// up to 64 elements use the cheap single-word path; longer patterns
// fall through to the multi-word Hyyrö generalisation. Same cutoff
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
    return indel_distance_multi_word<Token>(pattern, text, cutoff);
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
  const std::size_t m = pattern.size();
  if (m > 0U && m <= 64U) {
    return indel_single_word_u8(pattern, text, cutoff);
  }
  if (m > 64U && m <= 128U) {
    return indel_distance_k2_u8(pattern, text, cutoff);
  }
  if (m > 128U && m <= 192U) {
    return indel_distance_k3_u8(pattern, text, cutoff);
  }
  if (m > 192U && m <= 256U) {
    return indel_distance_k4_u8(pattern, text, cutoff);
  }
  if (m > 256U && m <= 320U) {
    return indel_distance_k5_u8(pattern, text, cutoff);
  }
  if (m > 320U && m <= 384U) {
    return indel_distance_k6_u8(pattern, text, cutoff);
  }
  if (m > 384U && m <= 448U) {
    return indel_distance_k7_u8(pattern, text, cutoff);
  }
  if (m > 448U && m <= 512U) {
    return indel_distance_k8_u8(pattern, text, cutoff);
  }
  return indel_distance_multi_word_u8(pattern, text, cutoff);
}

// Byte-alphabet multi-word Indel. Specialised for ``std::uint8_t``
// (and the codepoint-fits-in-byte fast path) — the 256-entry PEQ
// lives in a single contiguous allocation instead of the per-unique-
// char ``std::vector`` allocations that the generic ``unordered_map``
// path needs. For long ASCII / Latin-1 inputs this is the difference
// between a single 16 KB malloc and ~30 separate small allocations,
// which dominate the per-call cost for long patterns.
// Per-thread scratch buffers shared across the multi-word u8 kernel.
// The kernel only ever resizes the buffers upward, so reuse across
// calls amortises away the heap-allocation cost that was the
// dominant per-call expense for long patterns.
struct MultiWordU8Scratch {
  std::vector<std::uint64_t> peq;   // 256 * K entries
  std::vector<std::uint64_t> mask;  // K entries
  std::vector<std::uint64_t> V;     // K entries
  std::vector<std::uint64_t> sum;   // K entries
  std::vector<std::uint64_t> diff;  // K entries
  std::vector<std::uint64_t> U;     // K entries

  void resize_for(std::size_t K) {
    if (peq.size() < 256U * K) peq.resize(256U * K);
    if (mask.size() < K) mask.resize(K);
    if (V.size() < K)    V.resize(K);
    if (sum.size() < K)  sum.resize(K);
    if (diff.size() < K) diff.resize(K);
    if (U.size() < K)    U.resize(K);
  }
};

inline MultiWordU8Scratch& multi_word_u8_scratch() {
  thread_local MultiWordU8Scratch s;
  return s;
}

inline std::size_t indel_distance_multi_word_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff) {
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0U) return n;
  if (n == 0U) return m;
  const std::size_t K = (m + 63U) / 64U;
  const std::uint64_t one = 1U;

  MultiWordU8Scratch& scr = multi_word_u8_scratch();
  scr.resize_for(K);

  // Clear PEQ entries within the active 256 × K window.
  std::fill(scr.peq.begin(), scr.peq.begin() + 256U * K, std::uint64_t{0});

  for (std::size_t i = 0; i < m; ++i) {
    scr.peq[static_cast<std::size_t>(pattern[i]) * K + (i >> 6U)] |=
        one << (i & 63U);
  }

  std::uint64_t* const mask = scr.mask.data();
  for (std::size_t k = 0; k < K; ++k) mask[k] = ~std::uint64_t{0};
  if (m % 64U != 0U) {
    mask[K - 1] = (one << (m % 64U)) - 1U;
  }

  std::uint64_t* const V = scr.V.data();
  for (std::size_t k = 0; k < K; ++k) V[k] = mask[k];

  const bool has_cutoff = cutoff != kNoCutoff;
  const std::size_t bail_threshold = m + n + (has_cutoff ? cutoff : 0U);

  std::size_t j = 0;
  for (const std::uint8_t c : text) {
    const std::uint64_t* peq_row = scr.peq.data() + static_cast<std::size_t>(c) * K;

    // Fused per-block recurrence. The Hyyrö invariant ``U <= V``
    // bitwise means the per-block subtraction ``V - U`` never
    // underflows, so the borrow chain across blocks is identically
    // zero. Only the carry chain on the addition needs to
    // propagate — handled by ``__builtin_add_overflow``.
    std::uint64_t carry_in = 0;
    for (std::size_t k = 0; k < K; ++k) {
      const std::uint64_t Vk = V[k];
      const std::uint64_t Uk = Vk & peq_row[k];
      std::uint64_t s1;
      const bool c1 = __builtin_add_overflow(Vk, Uk, &s1);
      std::uint64_t s2;
      const bool c2 = __builtin_add_overflow(s1, carry_in, &s2);
      carry_in = static_cast<std::uint64_t>(c1) + static_cast<std::uint64_t>(c2);
      V[k] = (s2 | (Vk - Uk)) & mask[k];
    }

    ++j;
    if (has_cutoff) {
      std::size_t pc = 0;
      for (std::size_t k = 0; k < K; ++k) {
        pc += static_cast<std::size_t>(std::popcount(V[k]));
      }
      const std::size_t lower_bound_score = 2U * (j + pc);
      if (lower_bound_score > bail_threshold) {
        return cutoff + 1U;
      }
    }
  }

  std::size_t lcs_unmatched = 0;
  for (std::size_t k = 0; k < K; ++k) {
    lcs_unmatched += static_cast<std::size_t>(std::popcount(V[k]));
  }
  const std::size_t lcs = m - lcs_unmatched;
  return m + n - 2U * lcs;
}

// Hand-specialised K = 2 multi-word kernel for patterns 65..128
// characters. Architectural inspiration: rapidfuzz-cpp's
// ``lcs_unroll`` template specialisations (MIT, attribution in
// NOTICE). The two ideas inherited from their design are:
//
//   1. Per-block fused single-step Hyyrö recurrence —
//        U[k] = V[k] & PEQ[c][k]
//        sum  = V[k] + U[k] + carry_in   (carry propagates k → k+1)
//        V[k] = (sum | (V[k] - U[k])) & mask[k]
//      with NO separate borrow chain across blocks. The Hyyrö
//      invariant ``U[k] <= V[k]`` bitwise guarantees that the
//      per-block subtraction never underflows, so the cross-block
//      borrow term is mathematically zero. Eliminates one of the
//      four K-loops a naïve translation of Allison-Dix would emit.
//   2. State register-pinning. ``V0`` and ``V1`` are local
//      ``uint64_t`` (no array, no intermediate buffers), so the
//      compiler keeps them in registers across the entire text scan.
//
// Carry between block 0 and block 1 of the sum chain still needs
// propagation — handled with ``__builtin_add_overflow``, which the
// compiler lowers to ADCX/ADOX or equivalent.
//
// Correctness vs the generic multi-word kernel is verified on
// random fuzz (see tests/test_indel.py); the mathematical
// equivalence reduces to "the borrow chain is identically zero".
inline std::size_t indel_distance_k2_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff) noexcept {
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0U) return n;
  if (n == 0U) return m;
  // Caller guarantees 64 < m <= 128.
  constexpr std::size_t K = 2;
  const std::uint64_t one = 1U;

  MultiWordU8Scratch& scr = multi_word_u8_scratch();
  scr.resize_for(K);
  std::fill(scr.peq.begin(), scr.peq.begin() + 256U * K, std::uint64_t{0});

  for (std::size_t i = 0; i < m; ++i) {
    scr.peq[static_cast<std::size_t>(pattern[i]) * K + (i >> 6U)] |=
        one << (i & 63U);
  }

  const std::uint64_t mask0 = ~std::uint64_t{0};
  const std::uint64_t mask1 =
      (m % 64U == 0U) ? ~std::uint64_t{0} : ((one << (m % 64U)) - 1U);

  std::uint64_t V0 = mask0;
  std::uint64_t V1 = mask1;

  const std::uint64_t* const peq_base = scr.peq.data();

  const bool has_cutoff = cutoff != kNoCutoff;
  const std::size_t bail_threshold = m + n + (has_cutoff ? cutoff : 0U);

  std::size_t j = 0;
  for (const std::uint8_t c : text) {
    const std::uint64_t* peq_row = peq_base + static_cast<std::size_t>(c) * K;

    // Block 0: no carry in. The per-block fused expression.
    const std::uint64_t U0 = V0 & peq_row[0];
    std::uint64_t sum0;
    const bool carry01 = __builtin_add_overflow(V0, U0, &sum0);
    V0 = (sum0 | (V0 - U0)) & mask0;

    // Block 1: carry in from block 0; no carry out needed (K = 2).
    const std::uint64_t U1 = V1 & peq_row[1];
    std::uint64_t sum1;
    __builtin_add_overflow(V1, U1, &sum1);
    sum1 += static_cast<std::uint64_t>(carry01);
    V1 = (sum1 | (V1 - U1)) & mask1;

    ++j;
    if (has_cutoff) {
      const std::size_t pc =
          static_cast<std::size_t>(std::popcount(V0)) +
          static_cast<std::size_t>(std::popcount(V1));
      const std::size_t lower_bound_score = 2U * (j + pc);
      if (lower_bound_score > bail_threshold) {
        return cutoff + 1U;
      }
    }
  }

  const std::size_t lcs_unmatched =
      static_cast<std::size_t>(std::popcount(V0)) +
      static_cast<std::size_t>(std::popcount(V1));
  const std::size_t lcs = m - lcs_unmatched;
  return m + n - 2U * lcs;
}

// K = 3 hand-specialisation for ``129 <= m <= 192``. Same fused
// single-step recurrence as K=2, extended to three register-pinned
// blocks with a two-stage carry chain.
inline std::size_t indel_distance_k3_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff) noexcept {
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0U) return n;
  if (n == 0U) return m;
  // Caller guarantees 128 < m <= 192.
  constexpr std::size_t K = 3;
  const std::uint64_t one = 1U;

  MultiWordU8Scratch& scr = multi_word_u8_scratch();
  scr.resize_for(K);
  std::fill(scr.peq.begin(), scr.peq.begin() + 256U * K, std::uint64_t{0});

  for (std::size_t i = 0; i < m; ++i) {
    scr.peq[static_cast<std::size_t>(pattern[i]) * K + (i >> 6U)] |=
        one << (i & 63U);
  }

  const std::uint64_t mask0 = ~std::uint64_t{0};
  const std::uint64_t mask1 = ~std::uint64_t{0};
  const std::uint64_t mask2 =
      (m % 64U == 0U) ? ~std::uint64_t{0} : ((one << (m % 64U)) - 1U);

  std::uint64_t V0 = mask0;
  std::uint64_t V1 = mask1;
  std::uint64_t V2 = mask2;

  const std::uint64_t* const peq_base = scr.peq.data();

  const bool has_cutoff = cutoff != kNoCutoff;
  const std::size_t bail_threshold = m + n + (has_cutoff ? cutoff : 0U);

  std::size_t j = 0;
  for (const std::uint8_t c : text) {
    const std::uint64_t* peq_row = peq_base + static_cast<std::size_t>(c) * K;

    // Block 0: no carry in.
    const std::uint64_t U0 = V0 & peq_row[0];
    std::uint64_t sum0;
    const bool carry01 = __builtin_add_overflow(V0, U0, &sum0);
    V0 = (sum0 | (V0 - U0)) & mask0;

    // Block 1: carry in from block 0, carry out to block 2.
    const std::uint64_t U1 = V1 & peq_row[1];
    std::uint64_t sum1a;
    const bool carry_a = __builtin_add_overflow(V1, U1, &sum1a);
    std::uint64_t sum1;
    const bool carry_b =
        __builtin_add_overflow(sum1a, static_cast<std::uint64_t>(carry01), &sum1);
    const bool carry12 = carry_a || carry_b;
    V1 = (sum1 | (V1 - U1)) & mask1;

    // Block 2: carry in from block 1; no carry out (K = 3, last block).
    const std::uint64_t U2 = V2 & peq_row[2];
    std::uint64_t sum2;
    __builtin_add_overflow(V2, U2, &sum2);
    sum2 += static_cast<std::uint64_t>(carry12);
    V2 = (sum2 | (V2 - U2)) & mask2;

    ++j;
    if (has_cutoff) {
      const std::size_t pc =
          static_cast<std::size_t>(std::popcount(V0)) +
          static_cast<std::size_t>(std::popcount(V1)) +
          static_cast<std::size_t>(std::popcount(V2));
      const std::size_t lower_bound_score = 2U * (j + pc);
      if (lower_bound_score > bail_threshold) {
        return cutoff + 1U;
      }
    }
  }

  const std::size_t lcs_unmatched =
      static_cast<std::size_t>(std::popcount(V0)) +
      static_cast<std::size_t>(std::popcount(V1)) +
      static_cast<std::size_t>(std::popcount(V2));
  const std::size_t lcs = m - lcs_unmatched;
  return m + n - 2U * lcs;
}

// K = 4 hand-specialisation for ``193 <= m <= 256``. Four register-
// pinned blocks, three-stage carry chain. Beyond K = 4 the register
// pressure on x86-64 starts to spill V into stack, so the dispatch
// switches to the generic kernel.
inline std::size_t indel_distance_k4_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff) noexcept {
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0U) return n;
  if (n == 0U) return m;
  // Caller guarantees 192 < m <= 256.
  constexpr std::size_t K = 4;
  const std::uint64_t one = 1U;

  MultiWordU8Scratch& scr = multi_word_u8_scratch();
  scr.resize_for(K);
  std::fill(scr.peq.begin(), scr.peq.begin() + 256U * K, std::uint64_t{0});

  for (std::size_t i = 0; i < m; ++i) {
    scr.peq[static_cast<std::size_t>(pattern[i]) * K + (i >> 6U)] |=
        one << (i & 63U);
  }

  const std::uint64_t mask0 = ~std::uint64_t{0};
  const std::uint64_t mask1 = ~std::uint64_t{0};
  const std::uint64_t mask2 = ~std::uint64_t{0};
  const std::uint64_t mask3 =
      (m % 64U == 0U) ? ~std::uint64_t{0} : ((one << (m % 64U)) - 1U);

  std::uint64_t V0 = mask0;
  std::uint64_t V1 = mask1;
  std::uint64_t V2 = mask2;
  std::uint64_t V3 = mask3;

  const std::uint64_t* const peq_base = scr.peq.data();

  const bool has_cutoff = cutoff != kNoCutoff;
  const std::size_t bail_threshold = m + n + (has_cutoff ? cutoff : 0U);

  std::size_t j = 0;
  for (const std::uint8_t c : text) {
    const std::uint64_t* peq_row = peq_base + static_cast<std::size_t>(c) * K;

    // Block 0: no carry in.
    const std::uint64_t U0 = V0 & peq_row[0];
    std::uint64_t sum0;
    const bool carry01 = __builtin_add_overflow(V0, U0, &sum0);
    V0 = (sum0 | (V0 - U0)) & mask0;

    // Block 1: carry in from block 0, carry out to block 2.
    const std::uint64_t U1 = V1 & peq_row[1];
    std::uint64_t sum1a;
    const bool c1a = __builtin_add_overflow(V1, U1, &sum1a);
    std::uint64_t sum1;
    const bool c1b =
        __builtin_add_overflow(sum1a, static_cast<std::uint64_t>(carry01), &sum1);
    const bool carry12 = c1a || c1b;
    V1 = (sum1 | (V1 - U1)) & mask1;

    // Block 2: carry in from block 1, carry out to block 3.
    const std::uint64_t U2 = V2 & peq_row[2];
    std::uint64_t sum2a;
    const bool c2a = __builtin_add_overflow(V2, U2, &sum2a);
    std::uint64_t sum2;
    const bool c2b =
        __builtin_add_overflow(sum2a, static_cast<std::uint64_t>(carry12), &sum2);
    const bool carry23 = c2a || c2b;
    V2 = (sum2 | (V2 - U2)) & mask2;

    // Block 3: carry in from block 2; no carry out (K = 4, last block).
    const std::uint64_t U3 = V3 & peq_row[3];
    std::uint64_t sum3;
    __builtin_add_overflow(V3, U3, &sum3);
    sum3 += static_cast<std::uint64_t>(carry23);
    V3 = (sum3 | (V3 - U3)) & mask3;

    ++j;
    if (has_cutoff) {
      const std::size_t pc =
          static_cast<std::size_t>(std::popcount(V0)) +
          static_cast<std::size_t>(std::popcount(V1)) +
          static_cast<std::size_t>(std::popcount(V2)) +
          static_cast<std::size_t>(std::popcount(V3));
      const std::size_t lower_bound_score = 2U * (j + pc);
      if (lower_bound_score > bail_threshold) {
        return cutoff + 1U;
      }
    }
  }

  const std::size_t lcs_unmatched =
      static_cast<std::size_t>(std::popcount(V0)) +
      static_cast<std::size_t>(std::popcount(V1)) +
      static_cast<std::size_t>(std::popcount(V2)) +
      static_cast<std::size_t>(std::popcount(V3));
  const std::size_t lcs = m - lcs_unmatched;
  return m + n - 2U * lcs;
}

// Templated K = 5..8 implementation. ``K`` is a compile-time
// constant, so the per-block loop body unrolls and V[0..K-1] /
// mask[0..K-1] become register-resident stack arrays the compiler can
// keep in GPRs across the full text scan. Same fused single-step
// Hyyrö recurrence as the K = 2/3/4 specialisations — kept in one
// templated function instead of four hand-expanded variants because
// the duplication offers no clarity at that point.
//
// Above K = 8 the register file (16 GPRs on x86-64) overflows and the
// templated form's advantage disappears; for K >= 9 the dispatch
// falls back to ``indel_distance_multi_word_u8``, which uses the same
// fused-recurrence body but with a runtime K loop and a heap-backed
// V / mask buffer.
template <std::size_t K>
inline std::size_t indel_distance_kN_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff) noexcept {
  static_assert(K >= 2 && K <= 8, "indel_distance_kN_u8 covers K=2..8");
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0U) return n;
  if (n == 0U) return m;
  const std::uint64_t one = 1U;

  MultiWordU8Scratch& scr = multi_word_u8_scratch();
  scr.resize_for(K);
  std::fill(scr.peq.begin(), scr.peq.begin() + 256U * K, std::uint64_t{0});

  for (std::size_t i = 0; i < m; ++i) {
    scr.peq[static_cast<std::size_t>(pattern[i]) * K + (i >> 6U)] |=
        one << (i & 63U);
  }

  // Stack-resident per-block state. ``K`` is constexpr so these are
  // fixed-size arrays the compiler can promote to registers.
  std::uint64_t mask[K];
  for (std::size_t k = 0; k < K - 1U; ++k) mask[k] = ~std::uint64_t{0};
  mask[K - 1U] =
      (m % 64U == 0U) ? ~std::uint64_t{0} : ((one << (m % 64U)) - 1U);

  std::uint64_t V[K];
  for (std::size_t k = 0; k < K; ++k) V[k] = mask[k];

  const std::uint64_t* const peq_base = scr.peq.data();

  const bool has_cutoff = cutoff != kNoCutoff;
  const std::size_t bail_threshold = m + n + (has_cutoff ? cutoff : 0U);

  std::size_t j = 0;
  for (const std::uint8_t c : text) {
    const std::uint64_t* peq_row = peq_base + static_cast<std::size_t>(c) * K;

    // Block 0: no carry in. Carry-out feeds the next block.
    std::uint64_t carry;
    {
      const std::uint64_t U = V[0] & peq_row[0];
      std::uint64_t s;
      const bool c1 = __builtin_add_overflow(V[0], U, &s);
      carry = static_cast<std::uint64_t>(c1);
      V[0] = (s | (V[0] - U)) & mask[0];
    }

    // Middle blocks: carry in and carry out. The loop is unrolled by
    // the compiler because ``K`` is constexpr; the carry is a single
    // ``uint64_t`` (always 0 or 1 by the Hyyrö U <= V invariant) so
    // the chain lowers to ADCX/ADOX where available.
    for (std::size_t k = 1; k < K - 1U; ++k) {
      const std::uint64_t U = V[k] & peq_row[k];
      std::uint64_t s1, s2;
      const bool c1 = __builtin_add_overflow(V[k], U, &s1);
      const bool c2 = __builtin_add_overflow(s1, carry, &s2);
      carry = static_cast<std::uint64_t>(c1) + static_cast<std::uint64_t>(c2);
      V[k] = (s2 | (V[k] - U)) & mask[k];
    }

    // Last block: carry in, no carry out needed.
    {
      constexpr std::size_t k = K - 1U;
      const std::uint64_t U = V[k] & peq_row[k];
      std::uint64_t s;
      __builtin_add_overflow(V[k], U, &s);
      s += carry;
      V[k] = (s | (V[k] - U)) & mask[k];
    }

    ++j;
    if (has_cutoff) {
      std::size_t pc = 0;
      for (std::size_t k = 0; k < K; ++k) {
        pc += static_cast<std::size_t>(std::popcount(V[k]));
      }
      const std::size_t lower_bound_score = 2U * (j + pc);
      if (lower_bound_score > bail_threshold) {
        return cutoff + 1U;
      }
    }
  }

  std::size_t lcs_unmatched = 0;
  for (std::size_t k = 0; k < K; ++k) {
    lcs_unmatched += static_cast<std::size_t>(std::popcount(V[k]));
  }
  const std::size_t lcs = m - lcs_unmatched;
  return m + n - 2U * lcs;
}

inline std::size_t indel_distance_k5_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff) noexcept {
  return indel_distance_kN_u8<5>(pattern, text, cutoff);
}
inline std::size_t indel_distance_k6_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff) noexcept {
  return indel_distance_kN_u8<6>(pattern, text, cutoff);
}
inline std::size_t indel_distance_k7_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff) noexcept {
  return indel_distance_kN_u8<7>(pattern, text, cutoff);
}
inline std::size_t indel_distance_k8_u8(
    std::span<const std::uint8_t> pattern,
    std::span<const std::uint8_t> text,
    std::size_t cutoff) noexcept {
  return indel_distance_kN_u8<8>(pattern, text, cutoff);
}

// Multi-word bit-parallel Indel for patterns of arbitrary length.
//
// Hyyrö (2004) generalisation: split the pattern into ``K = ceil(m / 64)``
// blocks of 64 bits each. The V vector becomes ``V[K]``; the per-text-
// character recurrence
//   sum[k]  = V[k] + (V[k] & PEQ[c][k])    (with carry between blocks)
//   diff[k] = V[k] - (V[k] & PEQ[c][k])    (with borrow between blocks)
//   V'[k]   = (sum[k] | diff[k]) & mask[k]
// extends the single-word formula. The last block's ``mask`` masks
// off the bits beyond ``m % 64``.
//
// Memory: O(K) per call for the V buffer plus K-entry PEQ per
// observed alphabet character. PEQ is a hashmap for arbitrary
// ``Token``; the byte-alphabet specialisation stores a flat
// ``[256][K]`` array (rebuilt as a flat ``[256 * K]`` vector here so
// the same code path serves both).
template <typename Token>
inline std::size_t indel_distance_multi_word(
    std::span<const Token> pattern,
    std::span<const Token> text,
    std::size_t cutoff) {
  static_assert(std::is_integral_v<Token> || std::is_unsigned_v<Token>);
  const std::size_t m = pattern.size();
  const std::size_t n = text.size();
  if (m == 0U) return n;
  if (n == 0U) return m;
  const std::size_t K = (m + 63U) / 64U;
  const std::uint64_t one = 1U;

  // PEQ as ``Token -> std::vector<uint64_t>(K)``.
  std::unordered_map<Token, std::vector<std::uint64_t>> peq;
  for (std::size_t i = 0; i < m; ++i) {
    auto [it, inserted] = peq.try_emplace(pattern[i], K, std::uint64_t{0});
    it->second[i >> 6U] |= one << (i & 63U);
  }

  // Per-block masks: full ``~0`` for all but the last block, which
  // masks off the bits at indices >= ``m``.
  std::vector<std::uint64_t> mask(K, ~std::uint64_t{0});
  if (m % 64U != 0U) {
    mask.back() = (one << (m % 64U)) - 1U;
  }

  // V starts as the all-ones pattern (every position "unmatched").
  std::vector<std::uint64_t> V(K);
  for (std::size_t k = 0; k < K; ++k) V[k] = mask[k];

  std::vector<std::uint64_t> sum(K), diff(K), U(K);

  // Cutoff bookkeeping. The single-word lower bound generalises:
  // popcount(V) is summed across blocks.
  const bool has_cutoff = cutoff != kNoCutoff;
  const std::size_t bail_threshold = m + n + (has_cutoff ? cutoff : 0U);

  std::size_t j = 0;
  for (const auto c : text) {
    const auto it = peq.find(c);
    const std::uint64_t* peq_row = (it == peq.end()) ? nullptr : it->second.data();

    // U[k] = V[k] & PEQ[c][k]
    for (std::size_t k = 0; k < K; ++k) {
      U[k] = peq_row == nullptr ? std::uint64_t{0} : V[k] & peq_row[k];
    }

    // sum[k] = V[k] + U[k] with carry between blocks.
    std::uint64_t carry = 0;
    for (std::size_t k = 0; k < K; ++k) {
      const std::uint64_t a = V[k];
      const std::uint64_t b = U[k];
      const std::uint64_t s1 = a + b;
      const std::uint64_t s2 = s1 + carry;
      // Carry out: 1 if s1 < a (overflow on a+b) or (carry && s2 < s1).
      carry = static_cast<std::uint64_t>(s1 < a) +
              static_cast<std::uint64_t>(s2 < s1);
      sum[k] = s2;
    }

    // diff[k] = V[k] - U[k] with borrow between blocks.
    std::uint64_t borrow = 0;
    for (std::size_t k = 0; k < K; ++k) {
      const std::uint64_t a = V[k];
      const std::uint64_t b = U[k];
      const std::uint64_t d1 = a - b;
      const std::uint64_t d2 = d1 - borrow;
      // Borrow out: 1 if a < b OR (borrow && d1 == 0 && borrow > 0).
      borrow = static_cast<std::uint64_t>(a < b) +
               static_cast<std::uint64_t>(d1 < borrow);
      diff[k] = d2;
    }

    // V'[k] = (sum[k] | diff[k]) & mask[k]
    for (std::size_t k = 0; k < K; ++k) {
      V[k] = (sum[k] | diff[k]) & mask[k];
    }

    ++j;
    if (has_cutoff) {
      std::size_t pc = 0;
      for (std::size_t k = 0; k < K; ++k) {
        pc += static_cast<std::size_t>(std::popcount(V[k]));
      }
      const std::size_t lower_bound_score = 2U * (j + pc);
      if (lower_bound_score > bail_threshold) {
        return cutoff + 1U;
      }
    }
  }

  std::size_t lcs_unmatched = 0;
  for (std::size_t k = 0; k < K; ++k) {
    lcs_unmatched += static_cast<std::size_t>(std::popcount(V[k]));
  }
  const std::size_t lcs = m - lcs_unmatched;
  return m + n - 2U * lcs;
}

// Prepared bit-parallel pattern — PEQ built once, reused across many
// targets. The win is concentrated in batch / multi-window workloads
// (``partial_ratio`` evaluates the same pattern against multiple
// window slices) where the PEQ build cost would otherwise be paid
// per evaluation.
template <typename Token>
struct PreparedIndelPattern {
  static constexpr bool kIsByte = std::is_same_v<Token, std::uint8_t>;
  // Storage: 256-entry array for byte-alphabet (the indel_single_word_u8
  // shape), arbitrary hashmap otherwise. Both are queried the same way
  // via ``peq_of(c)``.
  std::conditional_t<kIsByte,
                     std::array<std::uint64_t, 256>,
                     std::unordered_map<Token, std::uint64_t>> peq{};
  std::uint64_t mask = 0;
  std::size_t m = 0;
  bool use_dp = false;  // true when m > 64; falls back to scalar DP
  std::span<const Token> pattern{};  // retained for the DP fallback path

  std::uint64_t peq_of(Token c) const noexcept {
    if constexpr (kIsByte) {
      return peq[c];
    } else {
      const auto it = peq.find(c);
      return it == peq.end() ? std::uint64_t{0} : it->second;
    }
  }
};

template <typename Token>
inline PreparedIndelPattern<Token> prepare_indel_pattern(
    std::span<const Token> pattern) {
  static_assert(std::is_integral_v<Token> || std::is_unsigned_v<Token>);
  PreparedIndelPattern<Token> prepared;
  prepared.m = pattern.size();
  prepared.pattern = pattern;
  if (prepared.m == 0U || prepared.m > 64U) {
    // For empty or multi-word patterns we don't pre-build the single-
    // word PEQ; the ``indel_distance_prepared`` path detects this and
    // routes to the multi-word kernel (which builds its own per-block
    // PEQ inline — see below for the prepared multi-word variant).
    prepared.use_dp = true;
    return prepared;
  }
  const std::uint64_t one = 1U;
  for (std::size_t i = 0; i < prepared.m; ++i) {
    if constexpr (PreparedIndelPattern<Token>::kIsByte) {
      prepared.peq[pattern[i]] |= one << i;
    } else {
      prepared.peq[pattern[i]] |= one << i;
    }
  }
  prepared.mask =
      (prepared.m == 64U) ? ~std::uint64_t{0} : ((one << prepared.m) - 1U);
  return prepared;
}

template <typename Token>
inline std::size_t indel_distance_prepared(
    const PreparedIndelPattern<Token>& prepared,
    std::span<const Token> text,
    std::size_t cutoff = kNoCutoff) {
  const std::size_t m = prepared.m;
  const std::size_t n = text.size();
  if (m == 0U) return n;
  if (n == 0U) return m;
  if (prepared.use_dp) {
    // Pattern too long for the single-word prepared PEQ: route to
    // the multi-word kernel. The multi-word PEQ build is amortised
    // over the text traversal already; pre-building it would require
    // a separate prepared-multi-word struct (TODO if profiling shows
    // it matters).
    return indel_distance_multi_word<Token>(prepared.pattern, text, cutoff);
  }
  std::uint64_t V = prepared.mask;
  const bool has_cutoff = cutoff != kNoCutoff;
  const std::size_t bail_threshold = m + n + (has_cutoff ? cutoff : 0U);
  std::size_t j = 0;
  for (const auto c : text) {
    const std::uint64_t U = V & prepared.peq_of(c);
    V = ((V + U) | (V - U)) & prepared.mask;
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
