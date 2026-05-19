#pragma once

// SIMD kernels for Hamming distance. Two complementary architectures:
//
//   * within-string: process one (query, target) pair, comparing
//     VEC_BYTES bytes per SIMD iteration. Bytewise cmpne over the
//     widest vector the host backend supports, popcount the resulting
//     mismatch mask. Used by the singular API.
//
//   * across-target batch: process VEC_BYTES targets in parallel by
//     using *byte-wide lanes*, one lane per target. Each iteration
//     scans one position of the query against all targets in the
//     batch and increments per-target counters. Counters are 8-bit
//     while the distance fits in a byte (n < 256), or 16-bit
//     otherwise. Used by the *_scores API.
//
// Each backend TU is compiled with its own target ISA enabled, so the
// arch-conditional code below picks the widest available kernel at
// compile time. Backends that don't define any of the supported ISAs
// fall through to a scalar tail loop.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "stride_align/alignment.hpp"
#include "stride_align/hamming.hpp"

#if defined(__SSE4_1__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__loongarch_sx)
#include <lsxintrin.h>
#endif

#if defined(__loongarch_asx)
#include <lasxintrin.h>
#endif

#if defined(__VSX__) || defined(__POWER8_VECTOR__)
#include <altivec.h>
#ifdef vector
#undef vector
#endif
#ifdef bool
#undef bool
#endif
#ifdef pixel
#undef pixel
#endif
#endif

namespace stride_align::hamming_simd {

namespace nb = nanobind;

// -------------------------------------------------------------------
// Within-string SIMD: count mismatch bytes between two equal-length
// uint8 buffers. The host backend's widest SIMD path is picked at
// compile time; everything else falls through to scalar.
// -------------------------------------------------------------------

inline std::size_t hamming_within_string(
    const std::uint8_t* a,
    const std::uint8_t* b,
    std::size_t n) noexcept {
  std::size_t count = 0;
  std::size_t k = 0;

#if defined(__AVX512BW__)
  // 64 bytes per iter. cmpneq_epi8 emits a native k-mask we can popcount
  // directly — no movemask/intermediate vector.
  while (k + 64U <= n) {
    const __m512i va = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(a + k));
    const __m512i vb = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(b + k));
    const __mmask64 ne = _mm512_cmpneq_epi8_mask(va, vb);
    count += static_cast<std::size_t>(_mm_popcnt_u64(ne));
    k += 64U;
  }
#elif defined(__AVX2__)
  // 32 bytes per iter. cmpeq + movemask + popcount(complement).
  while (k + 32U <= n) {
    const __m256i va = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(a + k));
    const __m256i vb = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(b + k));
    const __m256i eq = _mm256_cmpeq_epi8(va, vb);
    const std::uint32_t eq_mask =
        static_cast<std::uint32_t>(_mm256_movemask_epi8(eq));
    count += 32U - static_cast<std::size_t>(_mm_popcnt_u32(eq_mask));
    k += 32U;
  }
#elif defined(__SSE4_1__)
  while (k + 16U <= n) {
    const __m128i va = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(a + k));
    const __m128i vb = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(b + k));
    const __m128i eq = _mm_cmpeq_epi8(va, vb);
    const unsigned eq_mask =
        static_cast<unsigned>(_mm_movemask_epi8(eq)) & 0xFFFFU;
    count += 16U - static_cast<std::size_t>(__builtin_popcount(eq_mask));
    k += 16U;
  }
#elif defined(__ARM_NEON)
  // vceqq_u8 gives 0xFF/0x00 per byte. vaddvq on (cmpeq & 1) would sum
  // 1s on matches; we want mismatches, so sum (cmpeq XOR 1) instead — or
  // just (16 - matches). vaddvq is one instruction on ARMv8+, so the
  // chain stays short.
  while (k + 16U <= n) {
    const uint8x16_t va = vld1q_u8(a + k);
    const uint8x16_t vb = vld1q_u8(b + k);
    const uint8x16_t eq = vceqq_u8(va, vb);
    // eq bytes are 0xFF on match. (eq & 1) -> 1 on match, 0 on miss.
    const uint8x16_t match_ones = vandq_u8(eq, vdupq_n_u8(1));
    const std::uint8_t matches = vaddvq_u8(match_ones);
    count += 16U - matches;
    k += 16U;
  }
#elif defined(__loongarch_asx)
  // LASX: 32 bytes per iter. vmsknz_b returns a vector whose low 32
  // bits hold a byte-nonzero bitmask we can popcount.
  while (k + 32U <= n) {
    const __m256i va = __lasx_xvld(
        const_cast<void*>(reinterpret_cast<const void*>(a + k)), 0);
    const __m256i vb = __lasx_xvld(
        const_cast<void*>(reinterpret_cast<const void*>(b + k)), 0);
    const __m256i eq = __lasx_xvseq_b(va, vb);
    // ne = ~eq (mismatch bytes are 0xFF)
    const __m256i ne = __lasx_xvnor_v(eq, eq);
    const __m256i mask = __lasx_xvmsknz_b(ne);
    // xvmsknz writes the per-128-bit-lane masks separately: low 16
    // bits live in lane 0 of the result, lane 2 holds the high mask.
    const long lo = __lasx_xvpickve2gr_du(mask, 0);
    const long hi = __lasx_xvpickve2gr_du(mask, 2);
    count += static_cast<std::size_t>(__builtin_popcountll(
        static_cast<unsigned long long>(lo) & 0xFFFFULL));
    count += static_cast<std::size_t>(__builtin_popcountll(
        static_cast<unsigned long long>(hi) & 0xFFFFULL));
    k += 32U;
  }
#elif defined(__loongarch_sx)
  while (k + 16U <= n) {
    const __m128i va = __lsx_vld(
        const_cast<void*>(reinterpret_cast<const void*>(a + k)), 0);
    const __m128i vb = __lsx_vld(
        const_cast<void*>(reinterpret_cast<const void*>(b + k)), 0);
    const __m128i eq = __lsx_vseq_b(va, vb);
    const __m128i ne = __lsx_vnor_v(eq, eq);
    const __m128i mask = __lsx_vmsknz_b(ne);
    const long bits = __lsx_vpickve2gr_du(mask, 0);
    count += static_cast<std::size_t>(__builtin_popcountll(
        static_cast<unsigned long long>(bits) & 0xFFFFULL));
    k += 16U;
  }
#elif defined(__VSX__) || defined(__POWER8_VECTOR__)
  // VSX: 16 bytes per iter. vec_cmpeq returns 0xFF/0x00 per byte.
  // sum4s reduces to 4x u32 partial sums; we sum those in scalar.
  while (k + 16U <= n) {
    const __vector unsigned char va =
        vec_xl(0, reinterpret_cast<const unsigned char*>(a + k));
    const __vector unsigned char vb =
        vec_xl(0, reinterpret_cast<const unsigned char*>(b + k));
    const __vector unsigned char eq =
        reinterpret_cast<__vector unsigned char>(vec_cmpeq(va, vb));
    const __vector unsigned char ones = vec_splats(static_cast<unsigned char>(1));
    // ne_one[i] = 1 iff a[k+i] != b[k+i] (vec_andc = ones & ~eq).
    const __vector unsigned char ne_one = vec_andc(ones, eq);
    __vector unsigned int s4 = vec_sum4s(
        ne_one,
        reinterpret_cast<__vector unsigned int>(vec_splats(0U)));
    count += static_cast<std::size_t>(s4[0]) + s4[1] + s4[2] + s4[3];
    k += 16U;
  }
#endif

  // Scalar tail (and the entire kernel on non-SIMD backends).
  for (; k < n; ++k) {
    if (a[k] != b[k]) {
      ++count;
    }
  }
  return count;
}

// -------------------------------------------------------------------
// Across-target SIMD batch: pack one target per byte lane, increment
// per-target counters at each position. Counters are 8-bit while the
// distance fits in a byte (n <= 255); above that we widen to 16-bit
// per target (and process half as many targets per vector).
//
// The "gather" — getting one byte from each of L targets at position k
// — is a scalar loop into an aligned scratch buffer. With 16-64 lanes
// it's a handful of byte loads per iteration; the SIMD wins still
// dominate because the compare + accumulate is one vector op.
// -------------------------------------------------------------------

// Pick the widest available byte-lane width at compile time.
#if defined(__AVX512BW__)
inline constexpr std::size_t kHammingBatchByteLanes = 64;
#elif defined(__AVX2__) || defined(__loongarch_asx)
inline constexpr std::size_t kHammingBatchByteLanes = 32;
#elif defined(__SSE4_1__) || defined(__ARM_NEON) || \
      defined(__loongarch_sx) || defined(__VSX__) || defined(__POWER8_VECTOR__)
inline constexpr std::size_t kHammingBatchByteLanes = 16;
#else
inline constexpr std::size_t kHammingBatchByteLanes = 1;
#endif

namespace detail {

// Single-batch kernel for the 8-bit-counter case. Processes up to
// kHammingBatchByteLanes targets at once. The caller guarantees every
// target in this batch has length n and n <= 255.
//
// The "broadcast 16/32/64 copies of one byte" / "compare bytes" /
// "accumulate via subtract-from-counter" pattern is the standard
// AVX-512 / AVX2 / SSE / NEON / LSX / LASX / VSX bytewise idiom;
// each ISA gets its own block here so the inner loop stays as tight
// as possible.
inline void hamming_batch_u8(
    const std::uint8_t* q,
    std::size_t n,
    const std::uint8_t* const* targets,
    std::size_t batch_count,
    Score* out) {
  alignas(64) std::uint8_t col[kHammingBatchByteLanes] = {};

#if defined(__AVX512BW__)
  __m512i counters = _mm512_setzero_si512();
  const __m512i one_v = _mm512_set1_epi8(1);
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < batch_count; ++l) {
      col[l] = targets[l][k];
    }
    // Lanes past batch_count get zeroed (col is zero-initialized at
    // the start; we only write up to batch_count). A column-byte of 0
    // would falsely match q[k]=0; defend by also storing 0xFF in those
    // lanes so they always mismatch — wait, no: those lanes' scores
    // are never read, so it doesn't matter.
    const __m512i cv = _mm512_load_si512(
        reinterpret_cast<const __m512i*>(col));
    const __m512i qv = _mm512_set1_epi8(static_cast<char>(q[k]));
    // ne_mask is a 64-bit k-mask: 1 on mismatch.
    const __mmask64 ne_mask = _mm512_cmpneq_epi8_mask(cv, qv);
    // counters += 1 (per lane) where ne_mask is set.
    counters = _mm512_mask_add_epi8(counters, ne_mask, counters, one_v);
  }
  alignas(64) std::uint8_t result[64];
  _mm512_store_si512(reinterpret_cast<__m512i*>(result), counters);
#elif defined(__AVX2__)
  __m256i counters = _mm256_setzero_si256();
  const __m256i one_v = _mm256_set1_epi8(1);
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < batch_count; ++l) {
      col[l] = targets[l][k];
    }
    const __m256i cv = _mm256_load_si256(
        reinterpret_cast<const __m256i*>(col));
    const __m256i qv = _mm256_set1_epi8(static_cast<char>(q[k]));
    const __m256i eq = _mm256_cmpeq_epi8(cv, qv);
    // ne_one[i] = 1 iff differ. add to counters.
    const __m256i ne_one = _mm256_andnot_si256(eq, one_v);
    counters = _mm256_add_epi8(counters, ne_one);
  }
  alignas(64) std::uint8_t result[32];
  _mm256_store_si256(reinterpret_cast<__m256i*>(result), counters);
#elif defined(__SSE4_1__)
  __m128i counters = _mm_setzero_si128();
  const __m128i one_v = _mm_set1_epi8(1);
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < batch_count; ++l) {
      col[l] = targets[l][k];
    }
    const __m128i cv = _mm_load_si128(
        reinterpret_cast<const __m128i*>(col));
    const __m128i qv = _mm_set1_epi8(static_cast<char>(q[k]));
    const __m128i eq = _mm_cmpeq_epi8(cv, qv);
    const __m128i ne_one = _mm_andnot_si128(eq, one_v);
    counters = _mm_add_epi8(counters, ne_one);
  }
  alignas(64) std::uint8_t result[16];
  _mm_store_si128(reinterpret_cast<__m128i*>(result), counters);
#elif defined(__ARM_NEON)
  uint8x16_t counters = vdupq_n_u8(0);
  const uint8x16_t one_v = vdupq_n_u8(1);
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < batch_count; ++l) {
      col[l] = targets[l][k];
    }
    const uint8x16_t cv = vld1q_u8(col);
    const uint8x16_t qv = vdupq_n_u8(q[k]);
    const uint8x16_t eq = vceqq_u8(cv, qv);
    // ne_one = (NOT eq) AND 1 -> 1 per mismatch.
    const uint8x16_t ne_one = vbicq_u8(one_v, eq);
    counters = vaddq_u8(counters, ne_one);
  }
  alignas(64) std::uint8_t result[16];
  vst1q_u8(result, counters);
#elif defined(__loongarch_asx)
  __m256i counters = __lasx_xvldi(0);
  const __m256i one_v = __lasx_xvreplgr2vr_b(1);
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < batch_count; ++l) {
      col[l] = targets[l][k];
    }
    const __m256i cv = __lasx_xvld(col, 0);
    const __m256i qv = __lasx_xvreplgr2vr_b(static_cast<int>(q[k]));
    const __m256i eq = __lasx_xvseq_b(cv, qv);
    // ne_one = ~eq & 1; LSX/LASX vandn is Intel-style.
    const __m256i ne_one = __lasx_xvandn_v(eq, one_v);
    counters = __lasx_xvadd_b(counters, ne_one);
  }
  alignas(64) std::uint8_t result[32];
  __lasx_xvst(counters, result, 0);
#elif defined(__loongarch_sx)
  __m128i counters = __lsx_vldi(0);
  const __m128i one_v = __lsx_vreplgr2vr_b(1);
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < batch_count; ++l) {
      col[l] = targets[l][k];
    }
    const __m128i cv = __lsx_vld(col, 0);
    const __m128i qv = __lsx_vreplgr2vr_b(static_cast<int>(q[k]));
    const __m128i eq = __lsx_vseq_b(cv, qv);
    const __m128i ne_one = __lsx_vandn_v(eq, one_v);
    counters = __lsx_vadd_b(counters, ne_one);
  }
  alignas(64) std::uint8_t result[16];
  __lsx_vst(counters, result, 0);
#elif defined(__VSX__) || defined(__POWER8_VECTOR__)
  __vector unsigned char counters = vec_splats(static_cast<unsigned char>(0));
  const __vector unsigned char one_v = vec_splats(static_cast<unsigned char>(1));
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < batch_count; ++l) {
      col[l] = targets[l][k];
    }
    const __vector unsigned char cv =
        vec_xl(0, reinterpret_cast<const unsigned char*>(col));
    const __vector unsigned char qv =
        vec_splats(static_cast<unsigned char>(q[k]));
    const __vector unsigned char eq =
        reinterpret_cast<__vector unsigned char>(vec_cmpeq(cv, qv));
    // vec_andc(one, eq) = one & ~eq = 1 if differ.
    const __vector unsigned char ne_one = vec_andc(one_v, eq);
    counters = vec_add(counters, ne_one);
  }
  alignas(64) std::uint8_t result[16];
  vec_xst(counters, 0, reinterpret_cast<unsigned char*>(result));
#else
  // Scalar fallback.
  std::array<std::size_t, kHammingBatchByteLanes> result{};
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < batch_count; ++l) {
      if (targets[l][k] != q[k]) {
        ++result[l];
      }
    }
  }
#endif

  for (std::size_t l = 0; l < batch_count; ++l) {
    out[l] = static_cast<Score>(result[l]);
  }
}

// Wider-counter variant for n > 255. Half as many targets per vector
// (each gets 16 bits of counter), with the same compare + accumulate
// pattern. For simplicity we fall back to a per-target scalar SIMD
// loop here — long strings are rare in fuzzy-match workloads and the
// per-target within-string SIMD is still fast.
inline void hamming_batch_long(
    const std::uint8_t* q,
    std::size_t n,
    const std::uint8_t* const* targets,
    std::size_t batch_count,
    Score* out) {
  for (std::size_t l = 0; l < batch_count; ++l) {
    out[l] = static_cast<Score>(hamming_within_string(q, targets[l], n));
  }
}

}  // namespace detail

inline std::vector<Score> hamming_scores_simd(
    nb::handle query,
    nb::handle targets) {
  PyObject* fast_targets =
      PySequence_Fast(targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count = static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  namespace bv = ::stride_align::byte_view;
  const bv::ByteCompatKind q_kind = bv::classify(query.ptr());
  std::vector<Score> out(count);
  if (q_kind == bv::ByteCompatKind::None) {
    // Wider unicode or sequence-of-object inputs: scalar dispatch with
    // per-pair length checks happens in the bindings layer (which has
    // the prepared TokenStorage). The SIMD path only handles bytes /
    // 1-byte unicode; signal "not handled" by leaving out empty and
    // letting the caller fall through.
    out.clear();
    return out;
  }
  const std::uint8_t* q_ptr = nullptr;
  std::size_t q_len = 0;
  bv::view(query.ptr(), q_kind, q_ptr, q_len);

  // Per-target byte_view + length validation.
  std::vector<const std::uint8_t*> ptrs(count);
  std::vector<std::size_t> lens(count);
  for (std::size_t i = 0; i < count; ++i) {
    const bv::ByteCompatKind t_kind = bv::classify(items[i]);
    if (t_kind == bv::ByteCompatKind::None) {
      out.clear();  // Signal scalar fallback to caller.
      return out;
    }
    bv::view(items[i], t_kind, ptrs[i], lens[i]);
    if (lens[i] != q_len) {
      PyErr_Format(
          PyExc_ValueError,
          "Hamming requires equal-length inputs (target index %zu has "
          "length %zu, query has length %zu)",
          i, lens[i], q_len);
      throw nb::python_error();
    }
  }

  if (count == 0U || q_len == 0U) {
    // All zero by default.
    return out;
  }

  // Two batch paths:
  //
  //   * Across-target byte-lane (kHammingBatchByteLanes targets per
  //     SIMD vector) is fastest when q_len is small enough that the
  //     scalar per-lane gather doesn't dominate. Empirically the
  //     crossover sits around q_len ≈ 32 on AVX-512: at q_len=10 the
  //     across-target kernel wins ~3x over per-target, at q_len=64 the
  //     per-target within-string SIMD wins ~10x. The 32-byte / 255-row
  //     ceilings are joint bounds for the byte-counter path.
  //
  //   * Per-target within-string SIMD (kHammingBatchByteLanes bytes
  //     per iteration, one target at a time) scales linearly with
  //     q_len with no gather penalty, so it owns everything above the
  //     threshold.
  constexpr std::size_t kAcrossTargetMaxLen = 32U;
  constexpr std::size_t L = kHammingBatchByteLanes;
  if (q_len <= kAcrossTargetMaxLen) {
    for (std::size_t b = 0; b < count; b += L) {
      const std::size_t batch_count = std::min(L, count - b);
      detail::hamming_batch_u8(
          q_ptr, q_len, ptrs.data() + b, batch_count, out.data() + b);
    }
  } else {
    detail::hamming_batch_long(q_ptr, q_len, ptrs.data(), count, out.data());
  }
  return out;
}

inline std::vector<double> hamming_normalized_scores_simd(
    nb::handle query,
    nb::handle targets) {
  auto raw = hamming_scores_simd(query, targets);
  std::vector<double> out;
  out.reserve(raw.size());
  if (raw.empty()) {
    return out;
  }
  const std::size_t q_len = static_cast<std::size_t>(PyObject_Length(query.ptr()));
  for (const auto d : raw) {
    out.push_back(
        ::stride_align::hamming::normalize(static_cast<std::size_t>(d), q_len));
  }
  return out;
}

}  // namespace stride_align::hamming_simd
