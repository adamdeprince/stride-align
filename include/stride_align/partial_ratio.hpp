#pragma once

// ``sa.partial_ratio`` — best Indel-normalised similarity over a
// substring-anchored alignment of the shorter input inside the longer.
//
// Two-phase search (replaces an earlier fuzzywuzzy-style matching-
// blocks heuristic that systematically under-scored vs rapidfuzz when
// the optimal alignment was a boundary-anchored variable-length window):
//
//   1. **Interior phase** — best similarity over fixed-length-``n``
//      windows ``long[s : s+n]`` for each offset ``s ∈ [0, m-n]``,
//      where ``n = |shorter|`` and ``m = |longer|``.
//
//   2. **Boundary phase** — best similarity over **variable-length**
//      windows anchored at the long string's left edge
//      (``long[0 : i]`` for ``i ∈ [1, n-1]``) and at its right edge
//      (``long[m-i : m]`` for ``i ∈ [1, n-1]``). These capture the
//      case where the shorter only partially overlaps the longer at
//      its ends — the alignment shape that rapidfuzz reports via
//      ``partial_ratio_alignment`` with ``dest_end - dest_start < n``.
//
// The reported score is ``max{1 - indel(shorter, window) / (n + |window|)}``
// over every window the two phases enumerate.
//
// **Cutoff pushdown.** Every per-window Indel call carries a distance
// cutoff derived from the running best score: any window whose indel
// would not strictly exceed the current best is rejected before the
// per-character loop finishes. The cutoff is converted per window
// because the denominator ``n + |window|`` varies across the boundary
// phase.
//
// **Boundary char-set skip.** When extending a boundary window by one
// character, if that newly-added character does not appear in the
// shorter's character set, the LCS cannot grow but the denominator
// does — so normalised similarity strictly decreases. We skip those
// extensions without calling the Indel kernel. Sound and meaningfully
// fast on random / large-alphabet workloads.
//
// **Equal-length swap.** When ``|a| == |b|``, "partial" is ambiguous —
// either input can play the role of the pattern. The engine runs the
// one-direction search twice (with arguments swapped) and returns the
// better score. This matches rapidfuzz's behaviour on equal-length
// inputs.
//
// Architectural inspiration from rapidfuzz-cpp (MIT, Max Bachmann):
// the two-phase interior + boundary structure, the per-window
// monotonically-tightened cutoff, and the boundary char-set skip
// were absorbed from their implementation via a delegated source-
// reading agent so the code-authoring path here stays clean-room.
// The triangle-inequality branch-and-bound over offsets that
// rapidfuzz layers on the interior phase is not (yet) implemented
// here — a linear offset scan is functionally equivalent (same
// answer, slower in best case).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "stride_align/indel.hpp"
#include "stride_align/lcs.hpp"

namespace stride_align::partial_ratio {

using Codepoint = ::stride_align::lcs::Codepoint;

// Character-set membership over the shorter's alphabet, used for the
// boundary-extension skip. Specialised on Token: a 256-bit bitset
// for the byte fast path, ``std::unordered_set`` for the codepoint
// fallback.
template <typename Token>
class CharSet {
 public:
  void insert(Token c) {
    if constexpr (std::is_same_v<Token, std::uint8_t>) {
      const auto cc = static_cast<unsigned>(c);
      bits_[cc >> 6] |= std::uint64_t{1} << (cc & 63U);
    } else {
      set_.insert(c);
    }
  }
  bool contains(Token c) const {
    if constexpr (std::is_same_v<Token, std::uint8_t>) {
      const auto cc = static_cast<unsigned>(c);
      return ((bits_[cc >> 6] >> (cc & 63U)) & 1U) != 0U;
    } else {
      return set_.find(c) != set_.end();
    }
  }

 private:
  std::array<std::uint64_t, 4> bits_{};  // only used for byte path
  std::unordered_set<Token>    set_{};   // only used for codepoint path
};

// Convert a normalised-similarity best ``[0, 1]`` and a window-size
// total ``total = n + |window|`` to an integer indel cutoff such that
// any indel ``> cutoff`` cannot beat the current best.
//
//   sim > best   iff   indel < total * (1 - best)
//
// For integer indel that's ``indel <= ceil(total * (1 - best)) - 1``.
// The Indel kernel returns ``cutoff + 1`` for any indel ``> cutoff``,
// so passing this cutoff in tells the kernel to bail as soon as it
// can prove the answer won't improve ``best``.
inline std::size_t cutoff_for(double best, std::size_t total) {
  if (best <= 0.0) return ::stride_align::indel::kNoCutoff;
  const double max_d = static_cast<double>(total) * (1.0 - best);
  if (max_d <= 0.0) return 0U;
  const double ceil_d = std::ceil(max_d);
  const auto ceil_int = static_cast<std::size_t>(ceil_d);
  return ceil_int == 0U ? 0U : (ceil_int - 1U);
}

// One-direction search: pattern is ``short_s`` (n chars), text is
// ``long_s`` (m chars), with ``n <= m`` required. Returns the best
// normalised similarity in ``[0, 1]``. ``initial_best`` lets the
// caller seed the cutoff (used by the equal-length swap path so the
// second direction's cutoff starts where the first direction left
// off, not at zero).
template <typename Token>
inline double partial_ratio_one_direction(
    const std::vector<Token>& short_s,
    const std::vector<Token>& long_s,
    double initial_best) {
  const std::size_t n = short_s.size();
  const std::size_t m = long_s.size();
  if (n == 0U || m == 0U) return initial_best;
  if (n > m) return initial_best;  // caller invariant violated

  // Build the per-pattern PEQ once. Every window call reuses it.
  const auto prepared = ::stride_align::indel::prepare_indel_pattern<Token>(
      std::span<const Token>(short_s.data(), n));

  // Character set of the pattern — used by the boundary-extension
  // skip below.
  CharSet<Token> in_pattern;
  for (auto c : short_s) in_pattern.insert(c);

  double best = initial_best;

  // Per-window call: builds the right cutoff, runs the Indel kernel
  // against the prepared PEQ, and returns the **actual indel distance**
  // (or ``cutoff + 1`` if the kernel bailed). The score update on
  // ``best`` happens inline; callers care only about whether a perfect
  // match was found and (for the interior phase) about the cached
  // distance value used by the triangle-inequality pruning.
  auto run_window = [&](std::size_t start, std::size_t w) -> std::size_t {
    const std::size_t total = n + w;
    const std::size_t cutoff = cutoff_for(best, total);
    const std::size_t d = ::stride_align::indel::indel_distance_prepared<Token>(
        prepared,
        std::span<const Token>(long_s.data() + start, w),
        cutoff);
    if (cutoff != ::stride_align::indel::kNoCutoff && d > cutoff) {
      return d;  // bailed; ``d == cutoff + 1`` is a sound upper-bound
    }
    const double sim = 1.0 - static_cast<double>(d) / static_cast<double>(total);
    if (sim > best) best = sim;
    return d;
  };

  auto try_window = [&](std::size_t start, std::size_t w) -> bool {
    (void)run_window(start, w);
    return best >= 1.0;
  };

  // --- Interior phase: fixed-length-n windows over offsets ``[0, m-n]``.
  //
  // Divide-and-conquer evaluation guided by the triangle-inequality
  // lower bound: shifting the window by one offset can change the
  // indel distance by at most 1, so for any two evaluated offsets
  // ``a < b`` the minimum distance over the open interval ``(a, b)``
  // is bounded below by ``(D(a) + D(b) - (b - a)) / 2``. If even
  // that bound implies a score below the running ``best``, the entire
  // interval is pruned without further kernel calls. Both endpoints
  // and the midpoint are forced; the recursion handles the halves.
  if (n <= m) {
    const std::size_t max_offset = m - n;

    // Stack-resident scratch sized at compile-time-unknown ``max_offset+1``.
    // ``UINT64_MAX`` sentinel marks unevaluated offsets.
    constexpr std::size_t kUnevaluated = ~std::size_t{0};
    std::vector<std::size_t> D(max_offset + 1U, kUnevaluated);

    auto eval = [&](std::size_t s) -> std::size_t {
      if (D[s] != kUnevaluated) return D[s];
      D[s] = run_window(s, n);
      return D[s];
    };

    // Evaluate both endpoints so the recursion has both ends pinned.
    eval(0);
    if (best >= 1.0) return 1.0;
    if (max_offset > 0) {
      eval(max_offset);
      if (best >= 1.0) return 1.0;
    }

    // Lower bound on indel distance over the open interval ``(a, b)``.
    //
    // Shifting a fixed-length-n window by one offset drops one
    // character at the LEFT and adds one at the RIGHT. Each side can
    // independently change the LCS by 0 or 1, so the indel
    // ``= m + n - 2 * LCS`` can change by AT MOST 2 per offset step
    // (not 1, as one might naïvely expect for "shift by one position").
    // Hence the per-step bound is ``|D(s+1) - D(s)| <= 2``.
    //
    // Combining the two endpoint bounds:
    //   D(s) >= D(a) - 2 * (s - a)
    //   D(s) >= D(b) - 2 * (b - s)
    // Minimum over s in (a, b) of the max of those two lines is
    //   (D(a) + D(b)) / 2 - (b - a)
    // which is the tightest interval-wide lower bound. Clamped at 0
    // because indel distances are non-negative.
    auto interval_bound = [](std::size_t Da, std::size_t Db, std::size_t span) {
      const std::size_t sum = Da + Db;
      const std::size_t two_span = 2U * span;
      return sum <= two_span ? std::size_t{0} : ((sum - two_span) / 2U);
    };

    // Stack-based divide-and-conquer. The interior-phase total is
    // ``2 * n`` (windows are always length n there), so the cutoff
    // converts uniformly.
    std::vector<std::pair<std::size_t, std::size_t>> work;
    if (max_offset >= 2) work.push_back({0, max_offset});
    while (!work.empty()) {
      const auto [a, b] = work.back();
      work.pop_back();
      if (b <= a + 1U) continue;  // no interior offset

      const std::size_t bound = interval_bound(D[a], D[b], b - a);
      const std::size_t cutoff_d =
          cutoff_for(best, 2U * n);
      if (cutoff_d != ::stride_align::indel::kNoCutoff && bound > cutoff_d) {
        continue;  // entire interval can't beat current best
      }

      const std::size_t mid = a + (b - a) / 2U;
      eval(mid);
      if (best >= 1.0) return 1.0;

      // Recurse on the two halves.
      work.push_back({mid, b});
      work.push_back({a, mid});
    }
  }

  // --- Prefix boundary phase: long[0:i] for i in [1, n-1].
  //
  // Extending from length i-1 to length i adds long_s[i-1] at the
  // RIGHT of the window. If that character isn't in the pattern, the
  // LCS at length i equals the LCS at length i-1; the denominator
  // grows by 1; so normalised similarity strictly decreases. Skip
  // those extensions.
  for (std::size_t i = 1; i < n; ++i) {
    if (!in_pattern.contains(long_s[i - 1])) continue;
    if (try_window(0, i)) return 1.0;
  }

  // --- Suffix boundary phase: long[m-i:m] for i in [1, n-1].
  //
  // Extending from length i-1 to length i adds long_s[m-i] at the
  // LEFT of the window. Same skip applies.
  for (std::size_t i = 1; i < n; ++i) {
    if (!in_pattern.contains(long_s[m - i])) continue;
    if (try_window(m - i, i)) return 1.0;
  }

  return best;
}

// Engine: handles the empty / one-empty / equal-length cases on top
// of the one-direction search.
template <typename Token>
inline double partial_ratio_engine(
    const std::vector<Token>& a,
    const std::vector<Token>& b,
    const std::vector<Codepoint>& /*a_cps*/,
    const std::vector<Codepoint>& /*b_cps*/) {
  if (a.empty() && b.empty()) return 1.0;
  if (a.empty() || b.empty()) return 0.0;

  const bool a_short = a.size() <= b.size();
  const auto& short_s = a_short ? a : b;
  const auto& long_s  = a_short ? b : a;

  double best = partial_ratio_one_direction<Token>(short_s, long_s, 0.0);

  // Equal-length: "partial" is ambiguous — run the search with
  // arguments swapped too and take the better of the two. The second
  // call inherits the first call's best so its cutoff is tight from
  // the start.
  if (a.size() == b.size() && best < 1.0) {
    best = partial_ratio_one_direction<Token>(long_s, short_s, best);
  }
  return best;
}

// Byte fast path entry: caller has already established that both
// inputs are byte-compatible (ASCII Python str, bytes-like, or
// pre-narrowed). No widening or per-codepoint check needed.
inline double partial_ratio_bytes(
    std::span<const std::uint8_t> a,
    std::span<const std::uint8_t> b) {
  if (a.empty() && b.empty()) return 1.0;
  if (a.empty() || b.empty()) return 0.0;
  // Copy into vectors only because the engine signature expects
  // owning containers. (The internal slicing is span-based, so this
  // copy is the only allocation on the byte fast path.)
  std::vector<std::uint8_t> av(a.begin(), a.end());
  std::vector<std::uint8_t> bv(b.begin(), b.end());
  // The codepoint views are only used by the legacy engine signature
  // for the (now-removed) matching-block lookup; pass empty here.
  static const std::vector<Codepoint> empty_cps;
  return partial_ratio_engine<std::uint8_t>(av, bv, empty_cps, empty_cps);
}

// Public entry: routes through the byte fast path when both inputs
// fit in the [0, 256) range (the common case for ASCII / Latin-1
// text), and through the codepoint path otherwise. The byte path
// uses the flat 256-row PEQ and the K-specialised multi-word Indel
// kernels; the codepoint path uses the hashmap PEQ.
inline double partial_ratio(
    const std::vector<Codepoint>& a,
    const std::vector<Codepoint>& b) {
  if (a.empty() && b.empty()) return 1.0;
  if (a.empty() || b.empty()) return 0.0;

  bool fits_in_byte = true;
  for (const auto cp : a) { if (cp >= 256U) { fits_in_byte = false; break; } }
  if (fits_in_byte) {
    for (const auto cp : b) { if (cp >= 256U) { fits_in_byte = false; break; } }
  }

  if (fits_in_byte) {
    std::vector<std::uint8_t> ab(a.begin(), a.end());
    std::vector<std::uint8_t> bb(b.begin(), b.end());
    return partial_ratio_engine<std::uint8_t>(ab, bb, a, b);
  }
  return partial_ratio_engine<Codepoint>(a, b, a, b);
}

}  // namespace stride_align::partial_ratio
