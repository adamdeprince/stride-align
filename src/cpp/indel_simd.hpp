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
