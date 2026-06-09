#pragma once

// ``sa.partial_ratio`` — matching-block enumeration over the
// shorter/longer pair, with one Indel-normalised similarity per
// block-anchored window. Lifted out of the Python composition in
// ``src/stride_align/_fuzz.py`` so it pays one Python boundary
// crossing per call instead of one per matching block × candidate
// window.
//
// Algorithm (mirrors the Python reference exactly):
//
//   short, long = (a, b) if |a| <= |b| else (b, a)
//   blocks       = recursive_longest_common_substring_split(short, long)
//   for each block (s_pos, l_pos, k) in blocks:
//     window 1: block at LEFT edge of the short string
//                 [max(0, l_pos - s_pos), min(m, l_pos - s_pos + n))
//     window 2: block at RIGHT edge
//                 [max(0, l_pos + k - n), min(m, l_pos + k))
//     window 3: block region itself (only when k >= 4)
//                 [l_pos, l_pos + k)
//     score    = 1 - indel(short, window) / (|short| + |window|)
//     track the maximum
//
// The "k >= 4" threshold on the bare block-region candidate matches
// rapidfuzz's behaviour and keeps the shim from overshooting upstream
// on inputs where short matching anchors would otherwise drag the
// score up. See ``tests/test_fuzz.py::test_partial_ratio_*`` for the
// pinned cases.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stride_align/indel.hpp"
#include "stride_align/lcs.hpp"

namespace stride_align::partial_ratio {

using Codepoint = ::stride_align::lcs::Codepoint;

struct MatchingBlock {
  std::size_t s_pos = 0;  // start in short
  std::size_t l_pos = 0;  // start in long
  std::size_t k     = 0;  // length
};

// Recursive longest-common-substring split, iterative on a small
// stack. Returns blocks sorted by ``s_pos`` ascending — same order
// the Python reference produces. The LCS-substring DP operates on
// codepoints (the C++ ``lcs.hpp`` API), so the matching-block search
// always works in codepoint space even when the per-window Indel
// call uses a narrower token type.
inline std::vector<MatchingBlock> matching_blocks(
    const std::vector<Codepoint>& short_s,
    const std::vector<Codepoint>& long_s) {
  std::vector<MatchingBlock> blocks;
  if (short_s.empty() || long_s.empty()) return blocks;

  // (s_lo, s_hi, l_lo, l_hi) work tuples.
  struct Range { std::size_t s_lo, s_hi, l_lo, l_hi; };
  std::vector<Range> stack;
  stack.reserve(16);
  stack.push_back({0, short_s.size(), 0, long_s.size()});

  while (!stack.empty()) {
    const Range r = stack.back();
    stack.pop_back();
    if (r.s_lo >= r.s_hi || r.l_lo >= r.l_hi) continue;

    const auto info = ::stride_align::lcs::lcs_substring_info_range(
        short_s, r.s_lo, r.s_hi, long_s, r.l_lo, r.l_hi);
    if (info.length == 0) continue;

    const std::size_t s_pos = info.end_a - info.length;
    const std::size_t l_pos = info.end_b - info.length;

    // Push the RIGHT side of the split first so the LEFT side pops
    // next — that lets blocks accumulate roughly in ascending s_pos
    // order. (Final sort below makes it exact.)
    stack.push_back({s_pos + info.length, r.s_hi,
                      l_pos + info.length, r.l_hi});
    blocks.push_back({s_pos, l_pos, info.length});
    stack.push_back({r.s_lo, s_pos, r.l_lo, l_pos});
  }

  std::sort(blocks.begin(), blocks.end(),
            [](const MatchingBlock& a, const MatchingBlock& b) {
              return a.s_pos < b.s_pos;
            });
  return blocks;
}

// Underlying ``partial_ratio`` engine, templated on the token type
// used for the per-window Indel kernel. Matching-block enumeration
// always uses ``Codepoint`` (the LCS engine is codepoint-only), so
// for the byte fast path the caller is expected to have copied the
// short/long vectors into ``std::uint8_t`` form (cheap; the typical
// short is < 200 chars). ``short_cps`` / ``long_cps`` are the
// matching codepoint views used only for matching-block lookup.
template <typename Token>
inline double partial_ratio_engine(
    const std::vector<Token>& short_s,
    const std::vector<Token>& long_s,
    const std::vector<Codepoint>& short_cps,
    const std::vector<Codepoint>& long_cps) {
  const std::size_t n = short_s.size();
  const std::size_t m = long_s.size();

  const auto blocks = matching_blocks(short_cps, long_cps);
  if (blocks.empty()) return 0.0;

  // Prepared PEQ — built once, reused across every window candidate
  // below. Eliminates the per-window PEQ build that previously
  // dominated the call cost.
  const auto prepared = ::stride_align::indel::prepare_indel_pattern<Token>(
      std::span<const Token>(short_s.data(), short_s.size()));

  double best = 0.0;
  std::unordered_set<std::uint64_t> seen;

  auto try_window = [&](std::size_t start, std::size_t end) -> bool {
    if (end <= start || end > m) return false;
    const std::uint64_t key =
        (static_cast<std::uint64_t>(start) << 32) |
         static_cast<std::uint64_t>(end);
    if (!seen.insert(key).second) return false;
    const std::size_t window_len = end - start;
    const std::size_t total = n + window_len;
    // Push the current ``best`` as a kernel-level cutoff: any window
    // whose indel can't bring similarity above ``best`` is skipped
    // before the per-character loop finishes. Derivation:
    //   sim > best  iff  indel < total · (1 - best)
    // For integer indel that's ``indel <= ceil(max_allowed) - 1``;
    // the kernel returns cutoff + 1 for any indel > cutoff, so we
    // set ``cutoff = ceil(max_allowed) - 1``.
    std::size_t cutoff = ::stride_align::indel::kNoCutoff;
    if (best > 0.0) {
      const double max_allowed = static_cast<double>(total) * (1.0 - best);
      if (max_allowed <= 0.0) {
        // Even indel=0 (perfect match within window) would only tie,
        // not beat, the current best — skip without computing.
        return false;
      }
      // ceil(x) - 1, computed as ``static_cast<size_t>(max_allowed)``
      // when max_allowed is non-integer, ``size_t(max_allowed) - 1``
      // when it is. Pick the safe upper bound via the ceil idiom.
      const std::size_t ceiling = static_cast<std::size_t>(
          max_allowed == static_cast<double>(static_cast<std::size_t>(max_allowed))
              ? static_cast<std::size_t>(max_allowed)
              : static_cast<std::size_t>(max_allowed) + 1U);
      cutoff = ceiling == 0 ? 0 : ceiling - 1U;
    }
    const std::size_t indel =
        ::stride_align::indel::indel_distance_prepared<Token>(
            prepared,
            std::span<const Token>(long_s.data() + start, window_len),
            cutoff);
    if (cutoff != ::stride_align::indel::kNoCutoff && indel > cutoff) {
      // Kernel bailed — window can't beat best.
      return false;
    }
    const double sim = 1.0 - static_cast<double>(indel) / static_cast<double>(total);
    if (sim > best) best = sim;
    return best >= 1.0;
  };

  for (const auto& blk : blocks) {
    // Candidate 1: block at LEFT edge of short. Natural shift.
    // ``left_shift`` may be negative; clamp at 0. Width is ``n``
    // clamped at long's right boundary.
    const auto left_shift =
        static_cast<std::ptrdiff_t>(blk.l_pos) - static_cast<std::ptrdiff_t>(blk.s_pos);
    const std::size_t win1_start =
        left_shift > 0 ? static_cast<std::size_t>(left_shift) : 0U;
    const std::size_t win1_end = std::min(m, win1_start + n);
    if (try_window(win1_start, win1_end)) return 1.0;

    // Candidate 2: block at RIGHT edge of short. Window ends at
    // ``l_pos + k``. Width is ``n`` clamped at long's left boundary
    // (so shorter windows fall out naturally when the block is near
    // long's start).
    const std::size_t right_end = blk.l_pos + blk.k;
    const std::size_t win2_start = right_end > n ? right_end - n : 0U;
    const std::size_t win2_end = std::min(m, right_end);
    if (try_window(win2_start, win2_end)) return 1.0;

    // Candidate 3: block region itself — but only when the block is
    // at least 4 characters. See module-level comment.
    if (blk.k >= 4U) {
      if (try_window(blk.l_pos, blk.l_pos + blk.k)) return 1.0;
    }
  }

  return best;
}

// Public entry: routes through the byte fast path when both inputs
// fit in the [0, 256) range (the common case for ASCII / Latin-1
// text), and through the codepoint path otherwise. The byte path
// uses the flat 256-row multi-word PEQ; the codepoint path uses the
// generic hashmap PEQ.
inline double partial_ratio(
    const std::vector<Codepoint>& a,
    const std::vector<Codepoint>& b) {
  if (a.empty() && b.empty()) return 1.0;
  if (a.empty() || b.empty()) return 0.0;

  const bool a_is_short = a.size() <= b.size();
  const std::vector<Codepoint>& short_cps = a_is_short ? a : b;
  const std::vector<Codepoint>& long_cps  = a_is_short ? b : a;

  // Detect the byte-fast-path eligibility: every codepoint < 256.
  bool fits_in_byte = true;
  for (const auto cp : short_cps) {
    if (cp >= 256U) { fits_in_byte = false; break; }
  }
  if (fits_in_byte) {
    for (const auto cp : long_cps) {
      if (cp >= 256U) { fits_in_byte = false; break; }
    }
  }

  if (fits_in_byte) {
    // Cheap narrowing: codepoints we just verified all fit in a byte.
    std::vector<std::uint8_t> short_bytes(short_cps.begin(), short_cps.end());
    std::vector<std::uint8_t> long_bytes(long_cps.begin(),  long_cps.end());
    return partial_ratio_engine<std::uint8_t>(
        short_bytes, long_bytes, short_cps, long_cps);
  }
  return partial_ratio_engine<Codepoint>(
      short_cps, long_cps, short_cps, long_cps);
}

}  // namespace stride_align::partial_ratio
