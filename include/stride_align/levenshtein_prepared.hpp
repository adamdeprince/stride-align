#pragma once

// Persistent prepared state for Hyyrö's bit-parallel Levenshtein
// (Myers' algorithm). Pre-builds the pattern_match (PEQ) table once
// per query so each ``distance(target)`` call skips the per-call PEQ
// rebuild — the dominant non-kernel cost for the streaming-against-
// fixed-query workload.
//
// Today only the bytes / 1-byte unicode fast path has a prepared
// variant; wider tokens still go through the per-call hash-map PEQ
// build inside ``myers_distance<Token>``. Adding a wide prepared
// state is straightforward (cache the ``unordered_map<Token,
// vector<uint64_t>>`` instead of the flat 256*B vector) but isn't
// required to fix the original ``1v1`` bench bug.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "stride_align/levenshtein.hpp"

namespace stride_align::levenshtein {

struct PreparedLevenshteinScoreU8 {
  // Owning copy of the query bytes. We keep it so callers can
  // reflect/debug the prepared state and so the kernel can verify
  // text length math (mirrors the existing kernels' use of m).
  std::vector<std::uint8_t> pattern;
  // m = |pattern|. 0 means empty-query (returns |text|).
  std::size_t m = 0;
  // B = ceil(m / 64), the number of 64-bit blocks the PEQ spans.
  // Always 1 in the single_word case (m <= 64); m/64 rounded up
  // in the multi-word case.
  std::size_t B = 0;
  // single_word == true when m <= 64 and the single-word kernel
  // applies. PEQ is then 256 uint64s (one per byte value) with the
  // pattern positions bit-packed into the single word.
  // Otherwise the PEQ is 256*B uint64s laid out [c*B + block].
  bool single_word = false;
  // Bit-vector PEQ. Indexed by byte value (0..255) for the single-
  // word case, and by (byte * B + block_index) for the multi-word
  // case.
  std::vector<std::uint64_t> peq;
  // Multi-word only: the top bit of the last block (m mod 64
  // accounting). Mirrors the value computed inline in
  // myers_multi_word_u8.
  std::uint64_t top_bit_last = 0;
};

inline PreparedLevenshteinScoreU8 prepare_levenshtein_score_u8(
    std::span<const std::uint8_t> pattern) {
  PreparedLevenshteinScoreU8 prepared;
  prepared.pattern.assign(pattern.begin(), pattern.end());
  prepared.m = pattern.size();
  if (prepared.m == 0) {
    return prepared;
  }
  constexpr std::size_t kWord = 64U;
  const std::uint64_t one = 1;
  if (prepared.m <= kWord) {
    prepared.single_word = true;
    prepared.B = 1;
    prepared.peq.assign(256, 0);
    for (std::size_t i = 0; i < prepared.m; ++i) {
      prepared.peq[pattern[i]] |= one << i;
    }
  } else {
    prepared.single_word = false;
    prepared.B = (prepared.m + kWord - 1U) / kWord;
    prepared.peq.assign(static_cast<std::size_t>(256) * prepared.B, 0);
    for (std::size_t i = 0; i < prepared.m; ++i) {
      prepared.peq[static_cast<std::size_t>(pattern[i]) * prepared.B + (i / kWord)] |=
          one << (i % kWord);
    }
    const std::size_t last_bits = prepared.m - (prepared.B - 1U) * kWord;
    prepared.top_bit_last = std::uint64_t{1} << (last_bits - 1U);
  }
  return prepared;
}

// Single-word kernel running against a pre-built PEQ. Mirrors the
// inner loop of ``myers_single_word_u8`` exactly; only the PEQ build
// is hoisted out.
inline std::size_t myers_single_word_u8_with_peq(
    const std::uint64_t* peq,
    std::size_t m,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) noexcept {
  const std::uint64_t one = 1;
  const std::size_t n = text.size();
  if (m == 0U) return n;
  if (n == 0U) return m;
  if (cutoff != kNoCutoff) {
    const std::size_t len_diff = m > n ? m - n : n - m;
    if (len_diff > cutoff) return cutoff + 1U;
  }

  const std::size_t shift = m - 1U;
  std::uint64_t vp = (m == 64U) ? ~std::uint64_t{0} : ((one << m) - 1);
  std::uint64_t vn = 0;
  std::size_t score = m;

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

inline std::size_t levenshtein_score_prepared_u8(
    const PreparedLevenshteinScoreU8& prepared,
    std::span<const std::uint8_t> text,
    std::size_t cutoff = kNoCutoff) {
  if (prepared.m == 0) {
    return text.size();
  }
  if (text.empty()) {
    return prepared.m;
  }
  if (cutoff != kNoCutoff) {
    const std::size_t n = text.size();
    const std::size_t len_diff =
        prepared.m > n ? prepared.m - n : n - prepared.m;
    if (len_diff > cutoff) {
      return cutoff + 1U;
    }
  }
  if (prepared.single_word) {
    return myers_single_word_u8_with_peq(
        prepared.peq.data(), prepared.m, text, cutoff);
  }
  // Multi-word: reuse thread-local vp/vn scratch (amortised across
  // prepared 1-vs-N calls) and the existing detail::myers_inner with
  // our cached PEQ.
  const std::size_t B = prepared.B;
  MultiWordLevScratch& scr = multi_word_lev_scratch();
  scr.resize_for(B);
  std::fill_n(scr.vp.data(), B, ~std::uint64_t{0});
  std::fill_n(scr.vn.data(), B, std::uint64_t{0});
  constexpr std::size_t kWord = 64U;
  const std::size_t last_bits = prepared.m - (B - 1U) * kWord;
  if (last_bits < kWord) {
    scr.vp[B - 1U] = (std::uint64_t{1} << last_bits) - 1U;
  }
  return detail::myers_inner(
      prepared.m,
      B,
      text,
      [&prepared, B](std::uint8_t c) {
        return std::span<const std::uint64_t>(
            prepared.peq.data() + static_cast<std::size_t>(c) * B, B);
      },
      scr.vp.data(),
      scr.vn.data(),
      prepared.top_bit_last,
      prepared.m,
      cutoff);
}

}  // namespace stride_align::levenshtein
