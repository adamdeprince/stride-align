#pragma once

// Jaro similarity (Jaro 1989) and Jaro-Winkler (Winkler 1990).
//
// Jaro counts how many characters of two strings can be paired up within
// a sliding window of floor(max(|a|, |b|) / 2) - 1 positions, plus a
// half-transposition penalty for matches that occur out of order. The
// resulting similarity is in [0, 1]; 1.0 = identical, 0.0 = no matches
// inside the window.
//
// Jaro-Winkler adds a length-of-common-prefix bonus (capped at 4 chars
// by convention) when the base Jaro score crosses a threshold (0.7 by
// convention, matching rapidfuzz). The bonus magnifies similarity for
// strings that agree at the start — useful for surname/place-name
// matching, less useful for arbitrary text.
//
// Neither metric has a DP recurrence; bit-parallel implementations live
// in src/cpp/jaro_simd.hpp and use a PEQ-style position bitmap of the
// query alphabet to find matches in O(1) per target character.
// This header carries the scalar reference + the threshold/weight
// defaults used by the bindings layer.

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "stride_align/alignment.hpp"

namespace stride_align::jaro {

// Defaults match rapidfuzz's Jaro/JaroWinkler so that
// stride_align.jaro_winkler_similarity(a, b) == rapidfuzz at machine
// precision for byte inputs.
inline constexpr double kDefaultPrefixWeight = 0.1;
inline constexpr double kDefaultPrefixThreshold = 0.7;
inline constexpr std::size_t kDefaultPrefixCap = 4U;

// Half the maximum length, minus one, clamped to >= 0. Standard Jaro
// match window. For max <= 3 the formula underflows in unsigned
// arithmetic; the clamp gives "same-position matches only", which is
// what the floor would have produced anyway.
inline std::size_t match_window(std::size_t n, std::size_t m) noexcept {
  const std::size_t max_len = std::max(n, m);
  return max_len >= 2U ? max_len / 2U - 1U : 0U;
}

// Scalar Jaro similarity in [0, 1].
//
// Caller guarantees both spans use the same token width (templated, no
// implicit conversion). For empty inputs the convention is:
//   * both empty   → 1.0  (vacuously identical)
//   * one empty    → 0.0  (no possible match)
// This matches rapidfuzz.distance.Jaro.similarity.
template <typename Token>
inline double jaro_scalar(
    std::span<const Token> a,
    std::span<const Token> b) {
  const std::size_t n = a.size();
  const std::size_t m = b.size();
  if (n == 0U && m == 0U) {
    return 1.0;
  }
  if (n == 0U || m == 0U) {
    return 0.0;
  }

  const std::size_t window = match_window(n, m);

  std::vector<std::uint8_t> a_matched(n, 0U);
  std::vector<std::uint8_t> b_matched(m, 0U);
  std::size_t matches = 0U;

  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t lo = i > window ? i - window : 0U;
    const std::size_t hi = std::min(i + window + 1U, m);
    for (std::size_t j = lo; j < hi; ++j) {
      if (b_matched[j] != 0U) {
        continue;
      }
      if (a[i] != b[j]) {
        continue;
      }
      a_matched[i] = 1U;
      b_matched[j] = 1U;
      ++matches;
      break;
    }
  }

  if (matches == 0U) {
    return 0.0;
  }

  // Half-transposition count: walk matched chars of a and b in order;
  // each position where they differ contributes 1 to half_trans (which
  // is then divided by 2 for the formula).
  std::size_t half_trans = 0U;
  std::size_t k = 0U;
  for (std::size_t i = 0; i < n; ++i) {
    if (a_matched[i] == 0U) {
      continue;
    }
    while (b_matched[k] == 0U) {
      ++k;
    }
    if (a[i] != b[k]) {
      ++half_trans;
    }
    ++k;
  }

  // half_trans / 2 with INTEGER division — matches rapidfuzz exactly.
  // When the match permutation contains a cycle of odd length (say a 3-
  // cycle u→t→k→u), half_trans is odd (3 here), and the standard Jaro
  // formula's "transpositions = matches_out_of_order / 2" underestimates
  // the true transposition count from the cycle decomposition. The
  // literature and every common implementation accept that rounding-
  // down convention; matching it keeps our values bit-equivalent to
  // rapidfuzz's so users can swap one for the other.
  const double matches_d = static_cast<double>(matches);
  const double trans_d = static_cast<double>(half_trans / 2U);
  return (matches_d / static_cast<double>(n)
       + matches_d / static_cast<double>(m)
       + (matches_d - trans_d) / matches_d)
       / 3.0;
}

// ---------------------------------------------------------------------
// Bit-parallel Jaro (single 64-bit word per side).
//
// Replaces the O(n * window) scalar inner loop with an iteration over
// `b` where each step is an O(1) bitwise update: for j-th char of b,
// `peq[b[j]] & window_mask & ~used_a` is the set of unused matching
// positions in a; the lowest set bit picks the leftmost. After the
// scan, `used_a` and `b_matched` are popcount-able bitmaps, and
// transpositions fall out of a parallel walk over set bits.
//
// Constraint: both n and m must fit in 64 bits. The caller checks
// this before dispatching here; the scalar reference handles the
// long-string fallback.
// ---------------------------------------------------------------------

// Build PEQ over `a` for the byte alphabet (256 entries). Bit i of
// peq[c] is set iff a[i] == c. Returned by value because the caller
// usually owns it on the stack.
inline std::array<std::uint64_t, 256> build_peq_byte_64(
    std::span<const std::uint8_t> a) noexcept {
  std::array<std::uint64_t, 256> peq{};
  const std::size_t n = std::min<std::size_t>(a.size(), 64U);
  for (std::size_t i = 0; i < n; ++i) {
    peq[a[i]] |= std::uint64_t{1} << i;
  }
  return peq;
}

// Bits [lo, hi) of a 64-bit word. Caller guarantees 0 <= lo <= hi <= 64.
inline constexpr std::uint64_t bit_range(std::size_t lo, std::size_t hi) noexcept {
  // Build the [0, hi) half as a right-shift of all-ones; that
  // sidesteps the UB of `1ULL << 64`. The empty range hi==0 collapses
  // to zero because the high_bits term becomes ~0 >> 64 (= 0 under
  // unsigned arithmetic via the special case below).
  const std::uint64_t high_bits =
      (hi >= 64U) ? ~std::uint64_t{0} : (~std::uint64_t{0} >> (64U - hi));
  const std::uint64_t low_bits = (lo == 0U) ? 0U : ((std::uint64_t{1} << lo) - 1U);
  return high_bits & ~low_bits;
}

// One-shot bit-parallel Jaro for byte inputs with n <= 64 and m <= 64.
// Returns the (already-divided-by-3) similarity in [0, 1].
inline double jaro_bp_byte_64(
    std::span<const std::uint8_t> a,
    std::span<const std::uint8_t> b) noexcept {
  const std::size_t n = a.size();
  const std::size_t m = b.size();
  if (n == 0U && m == 0U) {
    return 1.0;
  }
  if (n == 0U || m == 0U) {
    return 0.0;
  }

  const std::size_t window = match_window(n, m);
  const auto peq = build_peq_byte_64(a);

  std::uint64_t used_a = 0;
  std::uint64_t b_matched = 0;
  for (std::size_t j = 0; j < m; ++j) {
    const std::size_t lo = j > window ? j - window : 0U;
    const std::size_t hi = std::min(j + window + 1U, n);
    if (lo >= hi) {
      continue;
    }
    const std::uint64_t window_mask = bit_range(lo, hi);
    const std::uint64_t candidate =
        peq[b[j]] & window_mask & ~used_a;
    if (candidate == 0U) {
      continue;
    }
    // Lowest set bit (= leftmost unused match position in a).
    const std::uint64_t lowest = candidate & (~candidate + 1U);
    used_a |= lowest;
    b_matched |= std::uint64_t{1} << j;
  }

  const std::size_t matches =
      static_cast<std::size_t>(std::popcount(used_a));
  if (matches == 0U) {
    return 0.0;
  }

  // Transposition count: walk set bits of used_a and b_matched in
  // ascending order and count mismatched pairs.
  std::uint64_t a_bits = used_a;
  std::uint64_t b_bits = b_matched;
  std::size_t half_trans = 0;
  while (a_bits != 0U) {
    const std::size_t i = static_cast<std::size_t>(std::countr_zero(a_bits));
    const std::size_t k = static_cast<std::size_t>(std::countr_zero(b_bits));
    if (a[i] != b[k]) {
      ++half_trans;
    }
    a_bits &= a_bits - 1U;
    b_bits &= b_bits - 1U;
  }

  const double matches_d = static_cast<double>(matches);
  const double trans_d = static_cast<double>(half_trans / 2U);
  return (matches_d / static_cast<double>(n)
       + matches_d / static_cast<double>(m)
       + (matches_d - trans_d) / matches_d)
       / 3.0;
}

// Arbitrary-length bit-parallel Jaro for byte inputs (no 64-char cap).
// A faithful bit-parallel transcription of jaro_scalar: it iterates `a`
// and, for each a[i], takes the leftmost not-yet-used position in b's
// window — the identical greedy, so the result is bit-for-bit equal to
// jaro_scalar at every length. The per-a-char match step is O(ceil(m/64))
// word ops instead of O(window) scalar compares, making the whole
// function O(n * ceil(m/64)) ~ O(n*m/64): 64x cheaper than the scalar
// O(n*window) path it replaces, and with no length cliff. PEQ is built
// over `b`; the W = ceil(m/64) word count is a runtime value.
inline double jaro_bp_byte_multiword(
    std::span<const std::uint8_t> a,
    std::span<const std::uint8_t> b) {
  const std::size_t n = a.size();
  const std::size_t m = b.size();
  if (n == 0U && m == 0U) {
    return 1.0;
  }
  if (n == 0U || m == 0U) {
    return 0.0;
  }

  const std::size_t window = match_window(n, m);
  const std::size_t wb = (m + 63U) / 64U;  // words spanning b's positions
  const std::size_t wa = (n + 63U) / 64U;  // words spanning a's positions

  // Thread-local scratch — amortise 256*wb PEQ + bitmap allocations
  // across long-string Jaro calls (same pattern as multi-word Indel).
  struct MultiWordJaroScratch {
    std::vector<std::uint64_t> peq;
    std::vector<std::uint64_t> used_b;
    std::vector<std::uint64_t> a_matched;
    std::vector<std::uint8_t> peq_dirty;
    std::array<std::uint32_t, 256> peq_touch{};
    std::uint32_t peq_gen = 0;
    std::size_t peq_layout_wb = 0;
  };
  thread_local MultiWordJaroScratch scr;

  if (scr.peq.size() < 256U * wb) {
    scr.peq.assign(256U * wb, 0U);
    scr.peq_dirty.clear();
    scr.peq_layout_wb = wb;
    scr.peq_touch.fill(0);
    scr.peq_gen = 0;
  } else if (scr.peq_layout_wb != wb) {
    std::fill_n(scr.peq.data(), 256U * wb, std::uint64_t{0});
    scr.peq_dirty.clear();
    scr.peq_layout_wb = wb;
  } else {
    for (const std::uint8_t s : scr.peq_dirty) {
      std::fill_n(
          scr.peq.data() + static_cast<std::size_t>(s) * wb, wb,
          std::uint64_t{0});
    }
    scr.peq_dirty.clear();
  }
  if (++scr.peq_gen == 0U) {
    scr.peq_touch.fill(0);
    scr.peq_gen = 1U;
  }

  // PEQ over b: bit j of peq row b[j] is set. 256 rows of `wb` words.
  for (std::size_t j = 0; j < m; ++j) {
    const std::uint8_t c = b[j];
    if (scr.peq_touch[c] != scr.peq_gen) {
      scr.peq_touch[c] = scr.peq_gen;
      scr.peq_dirty.push_back(c);
    }
    scr.peq[static_cast<std::size_t>(c) * wb + (j >> 6U)] |=
        std::uint64_t{1} << (j & 63U);
  }

  if (scr.used_b.size() < wb) scr.used_b.resize(wb);
  if (scr.a_matched.size() < wa) scr.a_matched.resize(wa);
  std::fill_n(scr.used_b.data(), wb, std::uint64_t{0});
  std::fill_n(scr.a_matched.data(), wa, std::uint64_t{0});
  std::size_t matches = 0U;

  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t lo = i > window ? i - window : 0U;
    const std::size_t hi = std::min(i + window + 1U, m);
    if (lo >= hi) {
      continue;
    }
    const std::uint64_t* row =
        scr.peq.data() + static_cast<std::size_t>(a[i]) * wb;
    // Leftmost unused b-position in [lo, hi) matching a[i]: scan words
    // low->high; the first word holding a candidate has the lowest
    // position, and its lowest set bit is the leftmost one.
    const std::size_t w_lo = lo >> 6U;
    const std::size_t w_hi = (hi - 1U) >> 6U;
    for (std::size_t w = w_lo; w <= w_hi; ++w) {
      const std::size_t base = w * 64U;
      const std::size_t blo = lo > base ? lo - base : 0U;
      const std::size_t bhi = std::min<std::size_t>(64U, hi - base);
      const std::uint64_t cand =
          row[w] & bit_range(blo, bhi) & ~scr.used_b[w];
      if (cand != 0U) {
        scr.used_b[w] |= cand & (~cand + 1U);
        scr.a_matched[i >> 6U] |= std::uint64_t{1} << (i & 63U);
        ++matches;
        break;
      }
    }
  }

  if (matches == 0U) {
    return 0.0;
  }

  // Transposition: lockstep ascending walk over the matched positions of
  // a (a_matched) and b (used_b), counting pairs whose chars differ.
  std::size_t half_trans = 0U;
  std::size_t wa_i = 0U;
  std::size_t wb_i = 0U;
  std::uint64_t a_bits = scr.a_matched[0];
  std::uint64_t b_bits = scr.used_b[0];
  for (std::size_t c = 0; c < matches; ++c) {
    while (a_bits == 0U) {
      a_bits = scr.a_matched[++wa_i];
    }
    while (b_bits == 0U) {
      b_bits = scr.used_b[++wb_i];
    }
    const std::size_t i =
        wa_i * 64U + static_cast<std::size_t>(std::countr_zero(a_bits));
    const std::size_t k =
        wb_i * 64U + static_cast<std::size_t>(std::countr_zero(b_bits));
    if (a[i] != b[k]) {
      ++half_trans;
    }
    a_bits &= a_bits - 1U;
    b_bits &= b_bits - 1U;
  }

  const double matches_d = static_cast<double>(matches);
  const double trans_d = static_cast<double>(half_trans / 2U);
  return (matches_d / static_cast<double>(n)
       + matches_d / static_cast<double>(m)
       + (matches_d - trans_d) / matches_d)
       / 3.0;
}

// Arbitrary-length, arbitrary-alphabet bit-parallel Jaro for wider tokens
// (UCS-2 / UCS-4 / object sequences). Same a-outer greedy as jaro_scalar
// — bit-identical at every length — but the PEQ is a hash map keyed by
// token value (the alphabet is sparse and can be huge, so a dense table
// is out). O(n * ceil(m/64)) with O(1) amortized map lookups; replaces
// the O(n*window) scalar reference on the wide-token path.
template <typename Token>
inline double jaro_bp_token_multiword(
    std::span<const Token> a,
    std::span<const Token> b) {
  const std::size_t n = a.size();
  const std::size_t m = b.size();
  if (n == 0U && m == 0U) {
    return 1.0;
  }
  if (n == 0U || m == 0U) {
    return 0.0;
  }

  const std::size_t window = match_window(n, m);
  const std::size_t wb = (m + 63U) / 64U;
  const std::size_t wa = (n + 63U) / 64U;

  // PEQ over b, keyed by token value: row[token] has bit j set for each
  // j where b[j] == token. `wb` words per distinct token.
  std::unordered_map<Token, std::vector<std::uint64_t>> peq;
  for (std::size_t j = 0; j < m; ++j) {
    auto& row = peq[b[j]];
    if (row.empty()) {
      row.assign(wb, 0U);
    }
    row[j >> 6U] |= std::uint64_t{1} << (j & 63U);
  }

  std::vector<std::uint64_t> used_b(wb, 0U);
  std::vector<std::uint64_t> a_matched(wa, 0U);
  std::size_t matches = 0U;

  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t lo = i > window ? i - window : 0U;
    const std::size_t hi = std::min(i + window + 1U, m);
    if (lo >= hi) {
      continue;
    }
    const auto it = peq.find(a[i]);
    if (it == peq.end()) {
      continue;
    }
    const std::uint64_t* row = it->second.data();
    const std::size_t w_lo = lo >> 6U;
    const std::size_t w_hi = (hi - 1U) >> 6U;
    for (std::size_t w = w_lo; w <= w_hi; ++w) {
      const std::size_t base = w * 64U;
      const std::size_t blo = lo > base ? lo - base : 0U;
      const std::size_t bhi = std::min<std::size_t>(64U, hi - base);
      const std::uint64_t cand = row[w] & bit_range(blo, bhi) & ~used_b[w];
      if (cand != 0U) {
        used_b[w] |= cand & (~cand + 1U);
        a_matched[i >> 6U] |= std::uint64_t{1} << (i & 63U);
        ++matches;
        break;
      }
    }
  }

  if (matches == 0U) {
    return 0.0;
  }

  std::size_t half_trans = 0U;
  std::size_t wa_i = 0U;
  std::size_t wb_i = 0U;
  std::uint64_t a_bits = a_matched[0];
  std::uint64_t b_bits = used_b[0];
  for (std::size_t c = 0; c < matches; ++c) {
    while (a_bits == 0U) {
      a_bits = a_matched[++wa_i];
    }
    while (b_bits == 0U) {
      b_bits = used_b[++wb_i];
    }
    const std::size_t i =
        wa_i * 64U + static_cast<std::size_t>(std::countr_zero(a_bits));
    const std::size_t k =
        wb_i * 64U + static_cast<std::size_t>(std::countr_zero(b_bits));
    if (a[i] != b[k]) {
      ++half_trans;
    }
    a_bits &= a_bits - 1U;
    b_bits &= b_bits - 1U;
  }

  const double matches_d = static_cast<double>(matches);
  const double trans_d = static_cast<double>(half_trans / 2U);
  return (matches_d / static_cast<double>(n)
       + matches_d / static_cast<double>(m)
       + (matches_d - trans_d) / matches_d)
       / 3.0;
}

// Common-prefix length up to `cap` chars (inclusive of cap). Used by
// the Jaro-Winkler bonus; the standard literature uses cap = 4.
template <typename Token>
inline std::size_t common_prefix(
    std::span<const Token> a,
    std::span<const Token> b,
    std::size_t cap) noexcept {
  const std::size_t limit = std::min({a.size(), b.size(), cap});
  std::size_t i = 0;
  while (i < limit && a[i] == b[i]) {
    ++i;
  }
  return i;
}

// Jaro-Winkler similarity in [0, 1]. If the base Jaro score is at least
// `prefix_threshold`, add `L * prefix_weight * (1 - jaro)` where L is
// the common prefix length (capped at `prefix_cap`). Otherwise return
// the base Jaro score unchanged — this matches rapidfuzz's behavior
// (where below threshold no bonus is applied).
template <typename Token>
inline double jaro_winkler_scalar(
    std::span<const Token> a,
    std::span<const Token> b,
    double prefix_weight = kDefaultPrefixWeight,
    double prefix_threshold = kDefaultPrefixThreshold,
    std::size_t prefix_cap = kDefaultPrefixCap) {
  const double jaro = jaro_scalar<Token>(a, b);
  if (jaro < prefix_threshold) {
    return jaro;
  }
  const std::size_t L = common_prefix<Token>(a, b, prefix_cap);
  return jaro + static_cast<double>(L) * prefix_weight * (1.0 - jaro);
}

}  // namespace stride_align::jaro
