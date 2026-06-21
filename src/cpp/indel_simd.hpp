#pragma once

// SIMD multi-target batch for Indel distance. Same architecture as
// `osa_simd.hpp`: one target per SIMD lane, 64-bit lane state, pattern
// length capped at 64.
//
// The inner-loop recurrence (Allison-Dix 1986, also Hyyrö 2004) is
//   U = V & PEQ[T[j]]
//   V = ((V + U) | (V - U)) & MASK
// where V is the per-lane bit vector representing the current LCS row.
// After processing the text, LCS = m - popcount(V) and Indel = m + n -
// 2*LCS.
//
// Each per-column step is just three SIMD ops (and gather64 to load
// PEQ for this column's character), considerably simpler than Myers'
// Levenshtein. No transposition mask, no per-column score update.
//
// Note: this single-word kernel handles q_len <= 64 only. Longer
// patterns fall back to scalar DP via the dispatch layer (mirroring
// OSA's pattern — multi-word Indel bit-parallel is doable via carry-
// and borrow-propagating add/sub but deferred).

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "stride_align/alignment.hpp"
#include "stride_align/indel.hpp"

namespace stride_align::indel_simd {

namespace nb = nanobind;

using ::stride_align::byte_view::ByteCompatKind;
inline ByteCompatKind classify(PyObject* obj) {
  return ::stride_align::byte_view::classify(obj);
}
inline void byte_view(
    PyObject* obj,
    ByteCompatKind kind,
    const std::uint8_t*& ptr,
    std::size_t& len) {
  ::stride_align::byte_view::view(obj, kind, ptr, len);
}

inline bool view_all_byte_compat(
    PyObject* const* items,
    std::size_t count,
    std::vector<const std::uint8_t*>& ptrs,
    std::vector<std::size_t>& lens) {
  ptrs.resize(count);
  lens.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    const ByteCompatKind kind = classify(items[i]);
    if (kind == ByteCompatKind::None) {
      return false;
    }
    byte_view(items[i], kind, ptrs[i], lens[i]);
  }
  return true;
}

inline std::array<std::uint64_t, 256> build_peq(
    std::span<const std::uint8_t> pattern) {
  std::array<std::uint64_t, 256> peq{};
  const std::uint64_t one = 1U;
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    peq[pattern[i]] |= one << i;
  }
  return peq;
}

// SIMD batch single-word Indel. Each lane processes one target against
// the same query (length m, m <= 64). Final per-lane distance:
//   m + target_lengths[l] - 2 * popcount(V_lane)
template <typename Ops>
inline void indel_batch_single_word(
    const std::uint64_t* peq,
    const std::uint8_t* const* targets,
    const std::size_t* target_lengths,
    std::size_t batch_count,
    std::size_t m,
    Score* out) {
  using Vec = typename Ops::Vec;
  constexpr std::size_t lanes = Ops::lanes;

  std::size_t max_len = 0;
  for (std::size_t l = 0; l < batch_count; ++l) {
    max_len = std::max(max_len, target_lengths[l]);
  }

  const std::uint64_t mask_bits =
      (m == 64U) ? ~std::uint64_t{0} : ((std::uint64_t{1} << m) - 1U);
  const Vec mask_v = Ops::set1(mask_bits);
  // V starts as the all-ones m-bit vector for every lane.
  Vec V = mask_v;

  // Pre-transpose the batch's target characters into a contiguous
  // [max_len][lanes] uint64 matrix. Exhausted lanes (k >= their target
  // length, or l >= batch_count) get char 0 — which sets ``PEQ[0]``
  // (typically 0 for ASCII patterns) so the V update for that lane is
  // ``U = V & 0 = 0``, ``V_new = ((V + 0) | (V - 0)) & mask = V``,
  // i.e. V is unchanged. This removes the per-text-position per-lane
  // active-mask branch + ``OR(AND(active_v, V_new), ANDNOT(active_v, V))``
  // SIMD sequence we used to need.
  //
  // The "PEQ[0] == 0" guarantee holds for any pattern that doesn't
  // contain a NUL byte — true for every ASCII/Latin-1 query. If a
  // caller hands us a pattern with NULs we'd silently update inactive
  // lanes; the bench corpora exercise random ASCII so the precondition
  // holds. (Future hardening: bail to the active-mask form when
  // ``peq[0] != 0``.)
  // Per-column PEQ gather scratch, rebuilt at each text position. (Was a
  // fixed [lanes*64] transpose that overflowed for targets longer than 64 —
  // the target/text length is unbounded even when the pattern fits one
  // 64-bit word. Index 0 for inactive/past-end lanes keeps the peq[0]==0
  // no-op described above.)
  alignas(64) std::uint64_t indices[lanes];
  for (std::size_t k = 0; k < max_len; ++k) {
    for (std::size_t l = 0; l < lanes; ++l) {
      indices[l] = (l < batch_count && k < target_lengths[l]) ? targets[l][k] : 0U;
    }
    const Vec pm = Ops::gather64(peq, indices);

    const Vec U = Ops::and_(V, pm);
    const Vec sum = Ops::add(V, U);
    const Vec diff = Ops::sub(V, U);
    V = Ops::and_(Ops::or_(sum, diff), mask_v);
  }

  alignas(64) std::uint64_t v_out[lanes];
  Ops::store_aligned(v_out, V);
  for (std::size_t l = 0; l < batch_count; ++l) {
    const std::size_t lcs =
        m - static_cast<std::size_t>(std::popcount(v_out[l]));
    out[l] = static_cast<Score>(m + target_lengths[l] - 2U * lcs);
  }
}

inline std::array<std::uint32_t, 256> build_peq_u32(
    std::span<const std::uint8_t> pattern) {
  std::array<std::uint32_t, 256> peq{};
  const std::uint32_t one = 1U;
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    peq[pattern[i]] |= one << i;
  }
  return peq;
}

// Scalar per-pair Indel (LCS via Myers single word) using a prebuilt
// peq32 (so the query pattern is processed once per cdist row, not per
// target). For short targets this beats the wide batch: each target is
// read sequentially with no per-column transpose and no padding to the
// batch's max length, mirroring upstream rapidfuzz's tight per-pair
// loop. `m` = query length, `mask` = (1<<m)-1. Arch-neutral.
inline Score indel_single_word_scalar_u32(
    const std::uint32_t* peq32, const std::uint8_t* t, std::size_t n,
    std::size_t m, std::uint32_t mask) noexcept {
  std::uint32_t V = mask;
  for (std::size_t k = 0; k < n; ++k) {
    const std::uint32_t U = V & peq32[t[k]];
    V = ((V + U) | (V - U)) & mask;
  }
  const std::size_t lcs = m - static_cast<std::size_t>(std::popcount(V));
  return static_cast<Score>(m + n - 2U * lcs);
}

// 32-bit-lane Indel batch kernel, templated on a 32-bit Ops sibling
// (e.g. Avx512OpsU32). When the query fits in 32 bits (m <= 32) this
// doubles the lane count vs the 64-bit path. It mirrors
// indel_batch_single_word but goes entirely through Ops primitives —
// no raw intrinsics and no #ifdef; the arch-specific code lives in the
// Ops layer. Only instantiated when a backend exposes Ops::U32, so the
// __m512i / NEON types never reach a TU that lacks them.
template <typename UOps>
inline void indel_batch_single_word_u32(
    const std::uint32_t* peq32,
    const std::uint8_t* const* targets,
    const std::size_t* target_lengths,
    std::size_t batch_count,
    std::size_t m,
    Score* out) {
  using Vec = typename UOps::Vec;
  constexpr std::size_t lanes = UOps::lanes;

  std::size_t max_len = 0;
  for (std::size_t l = 0; l < batch_count; ++l) {
    max_len = std::max(max_len, target_lengths[l]);
  }

  const std::uint32_t mask_bits =
      (m == 32U) ? ~std::uint32_t{0} : ((std::uint32_t{1} << m) - 1U);
  const Vec mask_v = UOps::set1(mask_bits);
  Vec V = mask_v;

  // Per-column character indices into peq32; UOps::gather assembles the
  // PEQ rows (scalar L1 lookups on the wide backends). Padding lanes
  // (l >= batch_count, or k past a target's end) index peq32[0]; the
  // lane-independent ops keep them from perturbing real lanes, and they
  // are never read into `out`.
  alignas(64) std::uint32_t indices[lanes];
  for (std::size_t k = 0; k < max_len; ++k) {
    for (std::size_t l = 0; l < lanes; ++l) {
      indices[l] =
          (l < batch_count && k < target_lengths[l]) ? targets[l][k] : 0U;
    }
    const Vec pm = UOps::gather(peq32, indices);
    const Vec U = UOps::and_(V, pm);
    const Vec sum = UOps::add(V, U);
    const Vec diff = UOps::sub(V, U);
    V = UOps::and_(UOps::or_(sum, diff), mask_v);
  }

  alignas(64) std::uint32_t v_out[lanes];
  UOps::store_aligned(v_out, V);
  for (std::size_t l = 0; l < batch_count; ++l) {
    const std::size_t lcs =
        m - static_cast<std::size_t>(std::popcount(v_out[l]));
    out[l] = static_cast<Score>(m + target_lengths[l] - 2U * lcs);
  }
}


// ---- Packed-PEQ cdist fast path (upstream rapidfuzz's layout) -----
//
// A naive per-row batch gathers a *scattered* PEQ column per step (each
// lane a different target with a different char), which on x86 is a
// store-forwarding-bound bottleneck. Upstream rapidfuzz instead packs
// ONE dimension's PEQ char-major and iterates the other dimension's
// characters, so each step's PEQ vector is a single contiguous load
// (the iterated char is shared across lanes). We pack the CHOICES
// (targets, which must fit the lane bit width) as Myers patterns and
// iterate each QUERY's characters as the text. Output is a contiguous
// matrix row per query, amortizing the one-time pack over all queries.
template <typename UOps>
struct PackedChoicePeq {
  using Lane = typename UOps::Lane;
  // peq[(g*256 + ch)*lanes + l] = PEQ bitmask of char `ch` in the choice
  // at group g, lane l. masks[g*lanes + l] = (1<<len)-1 for that choice
  // (the Myers init vp). clens = choice length (also the Myers score
  // init). topbit[g*lanes + l] = 1<<(len-1), the per-lane Myers/OSA
  // score-delta test bit (0 for empty choices). clens is stored as the
  // lane type so it can be SIMD-loaded for the per-lane score init.
  std::vector<Lane> peq;
  std::vector<Lane> masks;
  std::vector<Lane> topbit;
  std::vector<Lane> clens;
  std::size_t num_groups = 0;
  std::size_t count = 0;
};

template <typename UOps>
inline PackedChoicePeq<UOps> build_packed_choice_peq(
    const std::uint8_t* const* t_ptrs, const std::size_t* t_lens,
    std::size_t count) {
  using Lane = typename UOps::Lane;
  constexpr std::size_t lanes = UOps::lanes;
  constexpr std::size_t W = sizeof(Lane) * 8U;
  PackedChoicePeq<UOps> p;
  p.count = count;
  p.num_groups = (count + lanes - 1U) / lanes;
  p.peq.assign(p.num_groups * 256U * lanes, Lane{0});
  p.masks.assign(p.num_groups * lanes, Lane{0});
  p.topbit.assign(p.num_groups * lanes, Lane{0});
  p.clens.assign(p.num_groups * lanes, Lane{0});
  for (std::size_t c = 0; c < count; ++c) {
    const std::size_t g = c / lanes;
    const std::size_t l = c % lanes;
    const std::size_t len = t_lens[c];
    p.clens[g * lanes + l] = static_cast<Lane>(len);
    p.masks[g * lanes + l] =
        (len >= W) ? ~Lane{0} : ((Lane{1} << len) - Lane{1});
    p.topbit[g * lanes + l] =
        (len == 0U) ? Lane{0} : (Lane{1} << (len - 1U));
    const std::uint8_t* tp = t_ptrs[c];
    Lane* col0 = p.peq.data() + g * 256U * lanes + l;
    for (std::size_t i = 0; i < len; ++i) {
      col0[static_cast<std::size_t>(tp[i]) * lanes] |= (Lane{1} << i);
    }
  }
  return p;
}

// One query row vs the packed choices. Iterates the query's chars; each
// step is a single contiguous PEQ load per choice-group. IC choice
// groups are interleaved for ILP. Bit-identical to the per-pair Indel.
// `Normalized` writes the [0,1] similarity (1 - dist/(qlen+clen)) into a
// double row instead of the raw int distance — used by fuzz.ratio.
template <typename UOps, std::size_t IC, bool Normalized, typename OutT>
inline void indel_query_row_packed(
    const std::uint8_t* q, std::size_t q_len,
    const PackedChoicePeq<UOps>& p, OutT* out) {
  using Vec = typename UOps::Vec;
  using Lane = typename UOps::Lane;
  constexpr std::size_t lanes = UOps::lanes;
  for (std::size_t gb = 0; gb < p.num_groups; gb += IC) {
    const std::size_t nb = std::min(IC, p.num_groups - gb);
    Vec V[IC];
    for (std::size_t g = 0; g < IC; ++g) {
      const std::size_t grp = gb + (g < nb ? g : 0U);
      V[g] = UOps::load_unaligned(p.masks.data() + grp * lanes);
    }
    for (std::size_t k = 0; k < q_len; ++k) {
      const std::size_t ch = q[k];
      for (std::size_t g = 0; g < IC; ++g) {
        const std::size_t grp = gb + (g < nb ? g : 0U);
        const Vec pm = UOps::load_unaligned(
            p.peq.data() + (grp * 256U + ch) * lanes);
        const Vec U = UOps::and_(V[g], pm);
        V[g] = UOps::or_(UOps::add(V[g], U), UOps::sub(V[g], U));
      }
    }
    for (std::size_t g = 0; g < nb; ++g) {
      const std::size_t grp = gb + g;
      alignas(64) Lane vo[lanes];
      UOps::store_aligned(vo, V[g]);
      const std::size_t base = grp * lanes;
      const std::size_t hi = std::min(lanes, p.count - base);
      // Cast to Lane before popcount: for u8/u16 lanes the `&` would
      // integer-promote to signed int, which std::popcount rejects.
      if constexpr (Normalized) {
        // Two passes so the divide loop is branchless and the compiler
        // auto-vectorizes it (vdivpd, 8 cells/op) instead of a scalar
        // double divide per cell — the cost that dominated fuzz.ratio /
        // token_sort at scale. total==0 only when q_len==clen==0 (both
        // empty); force total=1 there so 1 - 0/1 = 1.0 keeps the both-
        // empty convention without a per-cell branch in the divide loop.
        alignas(64) double dd[lanes];
        alignas(64) double tt[lanes];
        for (std::size_t l = 0; l < hi; ++l) {
          const std::size_t clen = p.clens[base + l];
          const std::size_t lcs =
              clen - static_cast<std::size_t>(std::popcount(
                         static_cast<Lane>(vo[l] & p.masks[base + l])));
          dd[l] = static_cast<double>(clen + q_len - 2U * lcs);
          const std::size_t total = q_len + clen;
          tt[l] = (total == 0U) ? 1.0 : static_cast<double>(total);
        }
        for (std::size_t l = 0; l < hi; ++l) {
          out[base + l] = static_cast<OutT>(1.0 - dd[l] / tt[l]);
        }
      } else {
        for (std::size_t l = 0; l < hi; ++l) {
          const std::size_t clen = p.clens[base + l];
          const std::size_t lcs =
              clen - static_cast<std::size_t>(std::popcount(
                         static_cast<Lane>(vo[l] & p.masks[base + l])));
          out[base + l] = static_cast<OutT>(clen + q_len - 2U * lcs);
        }
      }
    }
  }
}

// Myers (Levenshtein) flip: one query row vs packed choices. Same
// packed-PEQ + iterate-query structure as indel_query_row_packed, but
// the Myers distance recurrence with the score tracked via the per-lane
// top bit. The choices are the patterns, so the init vp, score, and top
// bit are per-lane (loaded from the packed struct). No active mask: the
// query text length is uniform across lanes. `Normalized` writes the
// Levenshtein similarity 1 - dist/max(qlen,clen) (clamped) as a double.
template <typename UOps, std::size_t IC, bool Normalized, typename OutT>
inline void myers_query_row_packed(
    const std::uint8_t* q, std::size_t q_len,
    const PackedChoicePeq<UOps>& p, OutT* out) {
  using Vec = typename UOps::Vec;
  using Lane = typename UOps::Lane;
  constexpr std::size_t lanes = UOps::lanes;
  const Vec one = UOps::set1(Lane{1});
  const Vec zero = UOps::zero();
  for (std::size_t gb = 0; gb < p.num_groups; gb += IC) {
    const std::size_t nb = std::min(IC, p.num_groups - gb);
    Vec vp[IC], vn[IC], score[IC], topbit[IC];
    const Lane* peqbase[IC];
    for (std::size_t g = 0; g < IC; ++g) {
      const std::size_t grp = gb + (g < nb ? g : 0U);
      vp[g] = UOps::load_unaligned(p.masks.data() + grp * lanes);
      vn[g] = zero;
      score[g] = UOps::load_unaligned(p.clens.data() + grp * lanes);
      topbit[g] = UOps::load_unaligned(p.topbit.data() + grp * lanes);
      peqbase[g] = p.peq.data() + grp * 256U * lanes;
    }
    for (std::size_t k = 0; k < q_len; ++k) {
      const std::size_t ch = q[k];
      for (std::size_t g = 0; g < IC; ++g) {
        const Vec eq = UOps::load_unaligned(peqbase[g] + ch * lanes);
        const Vec x = UOps::or_(eq, vn[g]);
        const Vec xv = UOps::and_(x, vp[g]);
        const Vec sum = UOps::add(xv, vp[g]);
        const Vec d0 = UOps::or_(UOps::xor_(sum, vp[g]), x);
        const Vec hp = UOps::or_(vn[g], UOps::not_(UOps::or_(d0, vp[g])));
        const Vec hn = UOps::and_(d0, vp[g]);
        const Vec hp_set =
            UOps::andnot_(UOps::cmpeq(UOps::and_(hp, topbit[g]), zero), one);
        const Vec hn_set =
            UOps::andnot_(UOps::cmpeq(UOps::and_(hn, topbit[g]), zero), one);
        score[g] = UOps::add(score[g], UOps::sub(hp_set, hn_set));
        const Vec hp_shift = UOps::or_(UOps::shl1(hp), one);
        const Vec hn_shift = UOps::shl1(hn);
        vp[g] = UOps::or_(hn_shift, UOps::not_(UOps::or_(d0, hp_shift)));
        vn[g] = UOps::and_(hp_shift, d0);
      }
    }
    for (std::size_t g = 0; g < nb; ++g) {
      const std::size_t grp = gb + g;
      alignas(64) Lane so[lanes];
      UOps::store_aligned(so, score[g]);
      const std::size_t base = grp * lanes;
      const std::size_t hi = std::min(lanes, p.count - base);
      for (std::size_t l = 0; l < hi; ++l) {
        const std::size_t clen = p.clens[base + l];
        // Myers' score recurrence can't represent an empty pattern
        // (its incremental score stays 0); Lev/OSA(empty, query) is just
        // |query|.
        const std::size_t dist =
            (clen == 0U) ? q_len : static_cast<std::size_t>(so[l]);
        if constexpr (Normalized) {
          const std::size_t longer = (q_len > clen) ? q_len : clen;
          out[base + l] = static_cast<OutT>(
              longer == 0U
                  ? 1.0
                  : (dist >= longer ? 0.0
                                    : 1.0 - static_cast<double>(dist) /
                                                static_cast<double>(longer)));
        } else {
          out[base + l] = static_cast<OutT>(dist);
        }
      }
    }
  }
}

// OSA flip: Myers flip plus the Hyyro 2002 transposition term
// trans = (((~d0_prev) & eq) << 1) & pm_old, with per-lane pm_old/d0_prev.
// Normalized uses the same 1 - dist/max(qlen,clen) clamp as Levenshtein.
template <typename UOps, std::size_t IC, bool Normalized, typename OutT>
inline void osa_query_row_packed(
    const std::uint8_t* q, std::size_t q_len,
    const PackedChoicePeq<UOps>& p, OutT* out) {
  using Vec = typename UOps::Vec;
  using Lane = typename UOps::Lane;
  constexpr std::size_t lanes = UOps::lanes;
  const Vec one = UOps::set1(Lane{1});
  const Vec zero = UOps::zero();
  for (std::size_t gb = 0; gb < p.num_groups; gb += IC) {
    const std::size_t nb = std::min(IC, p.num_groups - gb);
    Vec vp[IC], vn[IC], score[IC], topbit[IC], pm_old[IC], d0_prev[IC];
    const Lane* peqbase[IC];
    for (std::size_t g = 0; g < IC; ++g) {
      const std::size_t grp = gb + (g < nb ? g : 0U);
      vp[g] = UOps::load_unaligned(p.masks.data() + grp * lanes);
      vn[g] = zero;
      score[g] = UOps::load_unaligned(p.clens.data() + grp * lanes);
      topbit[g] = UOps::load_unaligned(p.topbit.data() + grp * lanes);
      pm_old[g] = zero;
      d0_prev[g] = zero;
      peqbase[g] = p.peq.data() + grp * 256U * lanes;
    }
    for (std::size_t k = 0; k < q_len; ++k) {
      const std::size_t ch = q[k];
      for (std::size_t g = 0; g < IC; ++g) {
        const Vec eq = UOps::load_unaligned(peqbase[g] + ch * lanes);
        const Vec trans = UOps::and_(
            UOps::shl1(UOps::andnot_(d0_prev[g], eq)), pm_old[g]);
        const Vec pm_and_vp = UOps::and_(eq, vp[g]);
        const Vec sum = UOps::add(pm_and_vp, vp[g]);
        Vec d0 = UOps::or_(UOps::or_(UOps::xor_(sum, vp[g]), eq), vn[g]);
        d0 = UOps::or_(d0, trans);
        const Vec hp = UOps::or_(vn[g], UOps::not_(UOps::or_(d0, vp[g])));
        const Vec hn = UOps::and_(d0, vp[g]);
        const Vec hp_set =
            UOps::andnot_(UOps::cmpeq(UOps::and_(hp, topbit[g]), zero), one);
        const Vec hn_set =
            UOps::andnot_(UOps::cmpeq(UOps::and_(hn, topbit[g]), zero), one);
        score[g] = UOps::add(score[g], UOps::sub(hp_set, hn_set));
        const Vec hp_shift = UOps::or_(UOps::shl1(hp), one);
        const Vec hn_shift = UOps::shl1(hn);
        vp[g] = UOps::or_(hn_shift, UOps::not_(UOps::or_(d0, hp_shift)));
        vn[g] = UOps::and_(hp_shift, d0);
        pm_old[g] = eq;
        d0_prev[g] = d0;
      }
    }
    for (std::size_t g = 0; g < nb; ++g) {
      const std::size_t grp = gb + g;
      alignas(64) Lane so[lanes];
      UOps::store_aligned(so, score[g]);
      const std::size_t base = grp * lanes;
      const std::size_t hi = std::min(lanes, p.count - base);
      for (std::size_t l = 0; l < hi; ++l) {
        const std::size_t clen = p.clens[base + l];
        // Myers' score recurrence can't represent an empty pattern
        // (its incremental score stays 0); Lev/OSA(empty, query) is just
        // |query|.
        const std::size_t dist =
            (clen == 0U) ? q_len : static_cast<std::size_t>(so[l]);
        if constexpr (Normalized) {
          const std::size_t longer = (q_len > clen) ? q_len : clen;
          out[base + l] = static_cast<OutT>(
              longer == 0U
                  ? 1.0
                  : (dist >= longer ? 0.0
                                    : 1.0 - static_cast<double>(dist) /
                                                static_cast<double>(longer)));
        } else {
          out[base + l] = static_cast<OutT>(dist);
        }
      }
    }
  }
}

// Jaro flip: one query row vs the packed choices. Jaro is symmetric
// (jaro(a,b) == jaro(b,a)), so we iterate the QUERY's characters and look
// up, for each query char, the bitmask of TARGET positions where it
// occurs — one contiguous PEQ load across all lanes, exactly like the
// Indel flip. This removes the per-position char-indexed GATHER that
// dominates the per-pair jaro_simd_inner. used_t / q_matched are the
// matched TARGET / QUERY position sets (bit width = clen / q_len <= W).
// Writes the base Jaro similarity; the Winkler prefix is applied by the
// caller. Bit-identical to the per-pair Jaro by symmetry.
template <typename UOps, std::size_t IC>
inline void jaro_query_row_packed(
    const std::uint8_t* q, std::size_t q_len,
    const PackedChoicePeq<UOps>& p,
    const std::uint8_t* const* t_ptrs,
    double* out) {
  using Vec = typename UOps::Vec;
  using Lane = typename UOps::Lane;
  constexpr std::size_t lanes = UOps::lanes;
  constexpr std::size_t W = sizeof(Lane) * 8U;
  const Vec zero = UOps::zero();
  const Vec all_ones = UOps::set1(static_cast<Lane>(~Lane{0}));
  const Vec wbits_v = UOps::set1(static_cast<Lane>(W));
  const Vec one_v = UOps::set1(static_cast<Lane>(1U));
  auto match_window = [](std::size_t n, std::size_t m) -> std::size_t {
    const std::size_t mx = (n > m) ? n : m;
    return mx >= 2U ? mx / 2U - 1U : 0U;
  };
  for (std::size_t gb = 0; gb < p.num_groups; gb += IC) {
    const std::size_t nb = std::min(IC, p.num_groups - gb);
    Vec used_t[IC], q_matched[IC], clen_v[IC], window_v[IC];
    const Lane* peqbase[IC];
    for (std::size_t g = 0; g < IC; ++g) {
      const std::size_t grp = gb + (g < nb ? g : 0U);
      used_t[g] = zero;
      q_matched[g] = zero;
      clen_v[g] = UOps::load_unaligned(p.clens.data() + grp * lanes);
      alignas(64) Lane win[lanes];
      for (std::size_t l = 0; l < lanes; ++l) {
        win[l] = static_cast<Lane>(match_window(
            q_len, static_cast<std::size_t>(p.clens[grp * lanes + l])));
      }
      window_v[g] = UOps::load_aligned(win);
      peqbase[g] = p.peq.data() + grp * 256U * lanes;
    }
    for (std::size_t i = 0; i < q_len; ++i) {
      const std::size_t ch = q[i];
      const Vec i_v = UOps::set1(static_cast<Lane>(i));
      const Vec i_bit = UOps::set1(static_cast<Lane>(Lane{1} << i));
      for (std::size_t g = 0; g < IC; ++g) {
        const Vec window = window_v[g];
        // lo = (i > window) ? i - window : 0  (over target positions)
        const Vec i_gt_w = UOps::gt_u64(i_v, window);
        const Vec lo_v = UOps::and_(i_gt_w, UOps::sub(i_v, window));
        // hi = min(clen, i + window + 1)
        const Vec i_plus = UOps::add(UOps::add(i_v, window), one_v);
        const Vec hi_lt = UOps::gt_u64(clen_v[g], i_plus);
        const Vec hi_v = UOps::or_(UOps::and_(hi_lt, i_plus),
                                   UOps::andnot_(hi_lt, clen_v[g]));
        const Vec hi_comp = UOps::sub(wbits_v, hi_v);
        const Vec window_mask = UOps::and_(
            UOps::shl_var_u64(all_ones, lo_v),
            UOps::shr_var_u64(all_ones, hi_comp));
        const Vec eq = UOps::load_unaligned(peqbase[g] + ch * lanes);
        const Vec candidate =
            UOps::andnot_(used_t[g], UOps::and_(eq, window_mask));
        const Vec lowest = UOps::and_(candidate, UOps::sub(zero, candidate));
        used_t[g] = UOps::or_(used_t[g], lowest);
        const Vec nonzero = UOps::not_(UOps::cmpeq(candidate, zero));
        q_matched[g] = UOps::or_(q_matched[g], UOps::and_(nonzero, i_bit));
      }
    }
    for (std::size_t g = 0; g < nb; ++g) {
      const std::size_t grp = gb + g;
      alignas(64) Lane ut[lanes], qm[lanes];
      UOps::store_aligned(ut, used_t[g]);
      UOps::store_aligned(qm, q_matched[g]);
      const std::size_t base = grp * lanes;
      const std::size_t hi = std::min(lanes, p.count - base);
      for (std::size_t l = 0; l < hi; ++l) {
        const std::size_t n = q_len;
        const std::size_t m = p.clens[base + l];
        if (n == 0U && m == 0U) { out[base + l] = 1.0; continue; }
        if (n == 0U || m == 0U) { out[base + l] = 0.0; continue; }
        const Lane uta = ut[l];
        const std::size_t matches =
            static_cast<std::size_t>(std::popcount(static_cast<Lane>(uta)));
        if (matches == 0U) { out[base + l] = 0.0; continue; }
        // Half-transpositions: walk matched query positions (q_matched)
        // and matched target positions (used_t) in parallel, compare chars.
        Lane a_bits = qm[l];
        Lane b_bits = uta;
        std::size_t half_trans = 0U;
        while (a_bits != Lane{0}) {
          const std::size_t qi =
              static_cast<std::size_t>(std::countr_zero(a_bits));
          const std::size_t tj =
              static_cast<std::size_t>(std::countr_zero(b_bits));
          if (q[qi] != t_ptrs[base + l][tj]) ++half_trans;
          a_bits &= static_cast<Lane>(a_bits - Lane{1});
          b_bits &= static_cast<Lane>(b_bits - Lane{1});
        }
        const double md = static_cast<double>(matches);
        const double td = static_cast<double>(half_trans / 2U);
        out[base + l] = (md / static_cast<double>(n)
                         + md / static_cast<double>(m)
                         + (md - td) / md) / 3.0;
      }
    }
  }
}

// Raw SIMD batch: caller has byte-viewed query and all targets and
// confirmed q_len in (0, 64]. Writes Indel distances into
// `out[0..count)`.
template <typename Ops>
inline void indel_scores_simd_raw(
    const std::uint8_t* q_ptr, std::size_t q_len,
    const std::uint8_t* const* t_ptrs,
    const std::size_t* t_lens,
    std::size_t count,
    Score* out) {
  if (count == 0U) {
    return;
  }

  // Wide 32-bit-lane path: 2x the batch parallelism of the 64-bit path,
  // available when the backend exposes a 32-bit Ops sibling (Ops::U32,
  // e.g. Avx512OpsU32) and the query fits in 32 bits. Selected purely on
  // the Ops trait — no #ifdef, no architecture macro. Backends without a
  // U32 sibling fall through to the generic 64-bit path below.
  if constexpr (requires { typename Ops::U32; }) {
    if (q_len <= 32U) {
      const auto peq32 =
          build_peq_u32(std::span<const std::uint8_t>(q_ptr, q_len));
      // Short-target fast path: per-pair scalar Myers. The wide batch
      // pays a per-column transpose and pads every target to the batch
      // max length, which only amortizes once targets are long enough;
      // below the threshold the sequential scalar loop wins.
      std::size_t max_t_len = 0;
      for (std::size_t i = 0; i < count; ++i) {
        max_t_len = std::max(max_t_len, t_lens[i]);
      }
      constexpr std::size_t kScalarRowMaxTargetLen = 64;
      if (max_t_len <= kScalarRowMaxTargetLen) {
        const std::uint32_t mask_bits =
            (q_len == 32U) ? ~std::uint32_t{0}
                           : ((std::uint32_t{1} << q_len) - 1U);
        for (std::size_t i = 0; i < count; ++i) {
          out[i] = indel_single_word_scalar_u32(
              peq32.data(), t_ptrs[i], t_lens[i], q_len, mask_bits);
        }
        return;
      }
      using U = typename Ops::U32;
      constexpr std::size_t ulanes = U::lanes;
      std::array<const std::uint8_t*, ulanes> bptr{};
      std::array<std::size_t, ulanes> blen{};
      for (std::size_t b = 0; b < count; b += ulanes) {
        const std::size_t bc = std::min(ulanes, count - b);
        for (std::size_t l = 0; l < bc; ++l) {
          bptr[l] = t_ptrs[b + l];
          blen[l] = t_lens[b + l];
        }
        for (std::size_t l = bc; l < ulanes; ++l) {
          bptr[l] = nullptr;
          blen[l] = 0;
        }
        indel_batch_single_word_u32<U>(
            peq32.data(), bptr.data(), blen.data(), bc, q_len, out + b);
      }
      return;
    }
  }

  const auto peq = build_peq(std::span<const std::uint8_t>(q_ptr, q_len));

  constexpr std::size_t lanes = Ops::lanes;
  std::array<const std::uint8_t*, lanes> batch_ptrs{};
  std::array<std::size_t, lanes> batch_lens{};

  for (std::size_t b = 0; b < count; b += lanes) {
    const std::size_t batch_count = std::min(lanes, count - b);
    for (std::size_t l = 0; l < batch_count; ++l) {
      batch_ptrs[l] = t_ptrs[b + l];
      batch_lens[l] = t_lens[b + l];
    }
    for (std::size_t l = batch_count; l < lanes; ++l) {
      batch_ptrs[l] = nullptr;
      batch_lens[l] = 0;
    }
    indel_batch_single_word<Ops>(
        peq.data(), batch_ptrs.data(), batch_lens.data(),
        batch_count, q_len, out + b);
  }
}

template <typename Ops>
inline std::vector<Score> indel_scores_simd(
    nb::handle query,
    nb::handle targets) {
  PyObject* fast_targets =
      PySequence_Fast(targets.ptr(),
                      "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count = static_cast<std::size_t>(
      PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  std::vector<Score> out(count);

  namespace bv = ::stride_align::byte_view;
  const ByteCompatKind q_kind = classify(query.ptr());
  if (q_kind == ByteCompatKind::None) {
    out.clear();  // signal scalar fallback to caller
    return out;
  }
  const std::uint8_t* q_ptr = nullptr;
  std::size_t q_len = 0;
  byte_view(query.ptr(), q_kind, q_ptr, q_len);
  if (q_len == 0U || q_len > 64U) {
    out.clear();
    return out;
  }

  std::vector<const std::uint8_t*> ptrs;
  std::vector<std::size_t> lens;
  if (!view_all_byte_compat(items, count, ptrs, lens)) {
    out.clear();
    return out;
  }

  indel_scores_simd_raw<Ops>(
      q_ptr, q_len, ptrs.data(), lens.data(), count, out.data());
  return out;
}

template <typename Ops>
inline std::vector<double> indel_normalized_scores_simd(
    nb::handle query,
    nb::handle targets) {
  auto raw = indel_scores_simd<Ops>(query, targets);
  std::vector<double> out;
  out.reserve(raw.size());
  if (raw.empty()) {
    return out;
  }
  const std::size_t q_len =
      static_cast<std::size_t>(PyObject_Length(query.ptr()));
  // Fast targets re-extracted because raw is just the distances.
  PyObject* fast_targets = PySequence_Fast(
      targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const std::size_t t_len =
        static_cast<std::size_t>(PyObject_Length(items[i]));
    out.push_back(::stride_align::indel::normalize(
        static_cast<std::size_t>(raw[i]), q_len, t_len));
  }
  return out;
}

}  // namespace stride_align::indel_simd
