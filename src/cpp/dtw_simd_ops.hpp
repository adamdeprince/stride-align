#pragma once

// DTW SIMD primitive bundles — one Ops struct per (ISA × cell type).
//
// Lane assignment is one target per lane (same shape as the Myers batch
// kernels). Cell types:
//   * F32 / F64 — float DP for float32 / float64 inputs
//   * I32       — int32 DP for int16 inputs (widened)
//
// Each Ops exposes: lanes, Cell, Vec, set1, loadu, storeu, add, sub, mul,
// min, abs, infinity, cmp_eq, blend (mask ? a : b), inf_scalar.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__SSE4_1__) || defined(__SSE4_2__) || defined(__AVX__) || \
    defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#if defined(__loongarch_sx)
#include <lsxintrin.h>
#endif

#if defined(__loongarch_asx)
#include <lasxintrin.h>
#endif

namespace stride_align::dtw_simd {

// =====================================================================
// SSE4.1 / SSE4.2  (128-bit)
// =====================================================================
#if defined(__SSE4_1__) || defined(__SSE4_2__)

struct SseF32Ops {
  static constexpr std::size_t lanes = 4;
  using Cell = float;
  using Vec = __m128;

  static Cell inf_scalar() {
    return std::numeric_limits<float>::infinity();
  }
  static Vec set1(Cell x) { return _mm_set1_ps(x); }
  static Vec zero() { return _mm_setzero_ps(); }
  static Vec infinity() { return _mm_set1_ps(inf_scalar()); }
  static Vec loadu(const Cell* p) { return _mm_loadu_ps(p); }
  static void storeu(Cell* p, Vec v) { _mm_storeu_ps(p, v); }
  static Vec add(Vec a, Vec b) { return _mm_add_ps(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm_sub_ps(a, b); }
  static Vec mul(Vec a, Vec b) { return _mm_mul_ps(a, b); }
  static Vec min(Vec a, Vec b) { return _mm_min_ps(a, b); }
  static Vec abs(Vec a) {
    return _mm_andnot_ps(_mm_set1_ps(-0.0f), a);
  }
  static Vec cmp_eq(Vec a, Vec b) { return _mm_cmpeq_ps(a, b); }
  // mask lanes all-ones → take a, else b
  static Vec blend(Vec mask, Vec a, Vec b) { return _mm_blendv_ps(b, a, mask); }
};

struct SseF64Ops {
  static constexpr std::size_t lanes = 2;
  using Cell = double;
  using Vec = __m128d;

  static Cell inf_scalar() {
    return std::numeric_limits<double>::infinity();
  }
  static Vec set1(Cell x) { return _mm_set1_pd(x); }
  static Vec zero() { return _mm_setzero_pd(); }
  static Vec infinity() { return _mm_set1_pd(inf_scalar()); }
  static Vec loadu(const Cell* p) { return _mm_loadu_pd(p); }
  static void storeu(Cell* p, Vec v) { _mm_storeu_pd(p, v); }
  static Vec add(Vec a, Vec b) { return _mm_add_pd(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm_sub_pd(a, b); }
  static Vec mul(Vec a, Vec b) { return _mm_mul_pd(a, b); }
  static Vec min(Vec a, Vec b) { return _mm_min_pd(a, b); }
  static Vec abs(Vec a) {
    return _mm_andnot_pd(_mm_set1_pd(-0.0), a);
  }
  static Vec cmp_eq(Vec a, Vec b) { return _mm_cmpeq_pd(a, b); }
  static Vec blend(Vec mask, Vec a, Vec b) { return _mm_blendv_pd(b, a, mask); }
};

struct SseI32Ops {
  static constexpr std::size_t lanes = 4;
  using Cell = std::int32_t;
  using Vec = __m128i;

  static Cell inf_scalar() { return std::numeric_limits<std::int32_t>::max(); }
  static Vec set1(Cell x) { return _mm_set1_epi32(x); }
  static Vec zero() { return _mm_setzero_si128(); }
  static Vec infinity() { return _mm_set1_epi32(inf_scalar()); }
  static Vec loadu(const Cell* p) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  }
  static void storeu(Cell* p, Vec v) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
  }
  static Vec add(Vec a, Vec b) { return _mm_add_epi32(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm_sub_epi32(a, b); }
  static Vec mul(Vec a, Vec b) { return _mm_mullo_epi32(a, b); }
  static Vec min(Vec a, Vec b) { return _mm_min_epi32(a, b); }
  static Vec abs(Vec a) { return _mm_abs_epi32(a); }
  static Vec cmp_eq(Vec a, Vec b) { return _mm_cmpeq_epi32(a, b); }
  static Vec blend(Vec mask, Vec a, Vec b) {
    return _mm_blendv_epi8(b, a, mask);
  }
};

#endif  // SSE4.1/4.2

// =====================================================================
// AVX (256-bit float only; no full integer AVX)
// =====================================================================
#if defined(__AVX__)

struct AvxF32Ops {
  static constexpr std::size_t lanes = 8;
  using Cell = float;
  using Vec = __m256;

  static Cell inf_scalar() {
    return std::numeric_limits<float>::infinity();
  }
  static Vec set1(Cell x) { return _mm256_set1_ps(x); }
  static Vec zero() { return _mm256_setzero_ps(); }
  static Vec infinity() { return _mm256_set1_ps(inf_scalar()); }
  static Vec loadu(const Cell* p) { return _mm256_loadu_ps(p); }
  static void storeu(Cell* p, Vec v) { _mm256_storeu_ps(p, v); }
  static Vec add(Vec a, Vec b) { return _mm256_add_ps(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm256_sub_ps(a, b); }
  static Vec mul(Vec a, Vec b) { return _mm256_mul_ps(a, b); }
  static Vec min(Vec a, Vec b) { return _mm256_min_ps(a, b); }
  static Vec abs(Vec a) {
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a);
  }
  static Vec cmp_eq(Vec a, Vec b) {
    return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
  }
  static Vec blend(Vec mask, Vec a, Vec b) {
    return _mm256_blendv_ps(b, a, mask);
  }
};

struct AvxF64Ops {
  static constexpr std::size_t lanes = 4;
  using Cell = double;
  using Vec = __m256d;

  static Cell inf_scalar() {
    return std::numeric_limits<double>::infinity();
  }
  static Vec set1(Cell x) { return _mm256_set1_pd(x); }
  static Vec zero() { return _mm256_setzero_pd(); }
  static Vec infinity() { return _mm256_set1_pd(inf_scalar()); }
  static Vec loadu(const Cell* p) { return _mm256_loadu_pd(p); }
  static void storeu(Cell* p, Vec v) { _mm256_storeu_pd(p, v); }
  static Vec add(Vec a, Vec b) { return _mm256_add_pd(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm256_sub_pd(a, b); }
  static Vec mul(Vec a, Vec b) { return _mm256_mul_pd(a, b); }
  static Vec min(Vec a, Vec b) { return _mm256_min_pd(a, b); }
  static Vec abs(Vec a) {
    return _mm256_andnot_pd(_mm256_set1_pd(-0.0), a);
  }
  static Vec cmp_eq(Vec a, Vec b) {
    return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
  }
  static Vec blend(Vec mask, Vec a, Vec b) {
    return _mm256_blendv_pd(b, a, mask);
  }
};

#endif  // __AVX__

// =====================================================================
// AVX2 (256-bit + integer)
// =====================================================================
#if defined(__AVX2__)

struct Avx2F32Ops {
  static constexpr std::size_t lanes = 8;
  using Cell = float;
  using Vec = __m256;

  static Cell inf_scalar() {
    return std::numeric_limits<float>::infinity();
  }
  static Vec set1(Cell x) { return _mm256_set1_ps(x); }
  static Vec zero() { return _mm256_setzero_ps(); }
  static Vec infinity() { return _mm256_set1_ps(inf_scalar()); }
  static Vec loadu(const Cell* p) { return _mm256_loadu_ps(p); }
  static void storeu(Cell* p, Vec v) { _mm256_storeu_ps(p, v); }
  static Vec add(Vec a, Vec b) { return _mm256_add_ps(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm256_sub_ps(a, b); }
  static Vec mul(Vec a, Vec b) { return _mm256_mul_ps(a, b); }
  static Vec min(Vec a, Vec b) { return _mm256_min_ps(a, b); }
  static Vec abs(Vec a) {
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a);
  }
  static Vec cmp_eq(Vec a, Vec b) {
    return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
  }
  static Vec blend(Vec mask, Vec a, Vec b) {
    return _mm256_blendv_ps(b, a, mask);
  }
};

struct Avx2F64Ops {
  static constexpr std::size_t lanes = 4;
  using Cell = double;
  using Vec = __m256d;

  static Cell inf_scalar() {
    return std::numeric_limits<double>::infinity();
  }
  static Vec set1(Cell x) { return _mm256_set1_pd(x); }
  static Vec zero() { return _mm256_setzero_pd(); }
  static Vec infinity() { return _mm256_set1_pd(inf_scalar()); }
  static Vec loadu(const Cell* p) { return _mm256_loadu_pd(p); }
  static void storeu(Cell* p, Vec v) { _mm256_storeu_pd(p, v); }
  static Vec add(Vec a, Vec b) { return _mm256_add_pd(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm256_sub_pd(a, b); }
  static Vec mul(Vec a, Vec b) { return _mm256_mul_pd(a, b); }
  static Vec min(Vec a, Vec b) { return _mm256_min_pd(a, b); }
  static Vec abs(Vec a) {
    return _mm256_andnot_pd(_mm256_set1_pd(-0.0), a);
  }
  static Vec cmp_eq(Vec a, Vec b) {
    return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
  }
  static Vec blend(Vec mask, Vec a, Vec b) {
    return _mm256_blendv_pd(b, a, mask);
  }
};

struct Avx2I32Ops {
  static constexpr std::size_t lanes = 8;
  using Cell = std::int32_t;
  using Vec = __m256i;

  static Cell inf_scalar() { return std::numeric_limits<std::int32_t>::max(); }
  static Vec set1(Cell x) { return _mm256_set1_epi32(x); }
  static Vec zero() { return _mm256_setzero_si256(); }
  static Vec infinity() { return _mm256_set1_epi32(inf_scalar()); }
  static Vec loadu(const Cell* p) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  }
  static void storeu(Cell* p, Vec v) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
  }
  static Vec add(Vec a, Vec b) { return _mm256_add_epi32(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm256_sub_epi32(a, b); }
  static Vec mul(Vec a, Vec b) { return _mm256_mullo_epi32(a, b); }
  static Vec min(Vec a, Vec b) { return _mm256_min_epi32(a, b); }
  static Vec abs(Vec a) { return _mm256_abs_epi32(a); }
  static Vec cmp_eq(Vec a, Vec b) { return _mm256_cmpeq_epi32(a, b); }
  static Vec blend(Vec mask, Vec a, Vec b) {
    return _mm256_blendv_epi8(b, a, mask);
  }
};

#endif  // __AVX2__

// =====================================================================
// AVX-512
// =====================================================================
#if defined(__AVX512F__)

struct Avx512F32Ops {
  static constexpr std::size_t lanes = 16;
  using Cell = float;
  using Vec = __m512;

  static Cell inf_scalar() {
    return std::numeric_limits<float>::infinity();
  }
  static Vec set1(Cell x) { return _mm512_set1_ps(x); }
  static Vec zero() { return _mm512_setzero_ps(); }
  static Vec infinity() { return _mm512_set1_ps(inf_scalar()); }
  static Vec loadu(const Cell* p) { return _mm512_loadu_ps(p); }
  static void storeu(Cell* p, Vec v) { _mm512_storeu_ps(p, v); }
  static Vec add(Vec a, Vec b) { return _mm512_add_ps(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm512_sub_ps(a, b); }
  static Vec mul(Vec a, Vec b) { return _mm512_mul_ps(a, b); }
  static Vec min(Vec a, Vec b) { return _mm512_min_ps(a, b); }
  static Vec abs(Vec a) { return _mm512_abs_ps(a); }
  static Vec cmp_eq(Vec a, Vec b) {
    return _mm512_castsi512_ps(
        _mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ), -1));
  }
  static Vec blend(Vec mask, Vec a, Vec b) {
    // mask is all-ones / zero per lane from cmp_eq; DQ float bitwise is fine
    // now that the avx512bwvl backend requires avx512dq.
    return _mm512_or_ps(_mm512_and_ps(mask, a), _mm512_andnot_ps(mask, b));
  }
};

struct Avx512F64Ops {
  static constexpr std::size_t lanes = 8;
  using Cell = double;
  using Vec = __m512d;

  static Cell inf_scalar() {
    return std::numeric_limits<double>::infinity();
  }
  static Vec set1(Cell x) { return _mm512_set1_pd(x); }
  static Vec zero() { return _mm512_setzero_pd(); }
  static Vec infinity() { return _mm512_set1_pd(inf_scalar()); }
  static Vec loadu(const Cell* p) { return _mm512_loadu_pd(p); }
  static void storeu(Cell* p, Vec v) { _mm512_storeu_pd(p, v); }
  static Vec add(Vec a, Vec b) { return _mm512_add_pd(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm512_sub_pd(a, b); }
  static Vec mul(Vec a, Vec b) { return _mm512_mul_pd(a, b); }
  static Vec min(Vec a, Vec b) { return _mm512_min_pd(a, b); }
  static Vec abs(Vec a) { return _mm512_abs_pd(a); }
  static Vec cmp_eq(Vec a, Vec b) {
    return _mm512_castsi512_pd(
        _mm512_maskz_set1_epi64(_mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ), -1LL));
  }
  static Vec blend(Vec mask, Vec a, Vec b) {
    return _mm512_or_pd(_mm512_and_pd(mask, a), _mm512_andnot_pd(mask, b));
  }
};

struct Avx512I32Ops {
  static constexpr std::size_t lanes = 16;
  using Cell = std::int32_t;
  using Vec = __m512i;

  static Cell inf_scalar() { return std::numeric_limits<std::int32_t>::max(); }
  static Vec set1(Cell x) { return _mm512_set1_epi32(x); }
  static Vec zero() { return _mm512_setzero_si512(); }
  static Vec infinity() { return _mm512_set1_epi32(inf_scalar()); }
  static Vec loadu(const Cell* p) {
    return _mm512_loadu_si512(reinterpret_cast<const void*>(p));
  }
  static void storeu(Cell* p, Vec v) {
    _mm512_storeu_si512(reinterpret_cast<void*>(p), v);
  }
  static Vec add(Vec a, Vec b) { return _mm512_add_epi32(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm512_sub_epi32(a, b); }
  static Vec mul(Vec a, Vec b) { return _mm512_mullo_epi32(a, b); }
  static Vec min(Vec a, Vec b) { return _mm512_min_epi32(a, b); }
  static Vec abs(Vec a) { return _mm512_abs_epi32(a); }
  static Vec cmp_eq(Vec a, Vec b) {
    return _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(a, b), -1);
  }
  static Vec blend(Vec mask, Vec a, Vec b) {
    return _mm512_or_si512(_mm512_and_si512(mask, a), _mm512_andnot_si512(mask, b));
  }
};

#endif  // __AVX512F__

// =====================================================================
// ARM NEON
// =====================================================================
#if defined(__ARM_NEON) || defined(__ARM_NEON__)

struct NeonF32Ops {
  static constexpr std::size_t lanes = 4;
  using Cell = float;
  using Vec = float32x4_t;

  static Cell inf_scalar() {
    return std::numeric_limits<float>::infinity();
  }
  static Vec set1(Cell x) { return vdupq_n_f32(x); }
  static Vec zero() { return vdupq_n_f32(0.0f); }
  static Vec infinity() { return vdupq_n_f32(inf_scalar()); }
  static Vec loadu(const Cell* p) { return vld1q_f32(p); }
  static void storeu(Cell* p, Vec v) { vst1q_f32(p, v); }
  static Vec add(Vec a, Vec b) { return vaddq_f32(a, b); }
  static Vec sub(Vec a, Vec b) { return vsubq_f32(a, b); }
  static Vec mul(Vec a, Vec b) { return vmulq_f32(a, b); }
  static Vec min(Vec a, Vec b) { return vminq_f32(a, b); }
  static Vec abs(Vec a) { return vabsq_f32(a); }
  static Vec cmp_eq(Vec a, Vec b) {
    return vreinterpretq_f32_u32(vceqq_f32(a, b));
  }
  static Vec blend(Vec mask, Vec a, Vec b) {
    return vbslq_f32(vreinterpretq_u32_f32(mask), a, b);
  }
};

struct NeonF64Ops {
  static constexpr std::size_t lanes = 2;
  using Cell = double;
  using Vec = float64x2_t;

  static Cell inf_scalar() {
    return std::numeric_limits<double>::infinity();
  }
  static Vec set1(Cell x) { return vdupq_n_f64(x); }
  static Vec zero() { return vdupq_n_f64(0.0); }
  static Vec infinity() { return vdupq_n_f64(inf_scalar()); }
  static Vec loadu(const Cell* p) { return vld1q_f64(p); }
  static void storeu(Cell* p, Vec v) { vst1q_f64(p, v); }
  static Vec add(Vec a, Vec b) { return vaddq_f64(a, b); }
  static Vec sub(Vec a, Vec b) { return vsubq_f64(a, b); }
  static Vec mul(Vec a, Vec b) { return vmulq_f64(a, b); }
  static Vec min(Vec a, Vec b) { return vminq_f64(a, b); }
  static Vec abs(Vec a) { return vabsq_f64(a); }
  static Vec cmp_eq(Vec a, Vec b) {
    return vreinterpretq_f64_u64(vceqq_f64(a, b));
  }
  static Vec blend(Vec mask, Vec a, Vec b) {
    return vbslq_f64(vreinterpretq_u64_f64(mask), a, b);
  }
};

struct NeonI32Ops {
  static constexpr std::size_t lanes = 4;
  using Cell = std::int32_t;
  using Vec = int32x4_t;

  static Cell inf_scalar() { return std::numeric_limits<std::int32_t>::max(); }
  static Vec set1(Cell x) { return vdupq_n_s32(x); }
  static Vec zero() { return vdupq_n_s32(0); }
  static Vec infinity() { return vdupq_n_s32(inf_scalar()); }
  static Vec loadu(const Cell* p) { return vld1q_s32(p); }
  static void storeu(Cell* p, Vec v) { vst1q_s32(p, v); }
  static Vec add(Vec a, Vec b) { return vaddq_s32(a, b); }
  static Vec sub(Vec a, Vec b) { return vsubq_s32(a, b); }
  static Vec mul(Vec a, Vec b) { return vmulq_s32(a, b); }
  static Vec min(Vec a, Vec b) { return vminq_s32(a, b); }
  static Vec abs(Vec a) { return vabsq_s32(a); }
  static Vec cmp_eq(Vec a, Vec b) {
    return vreinterpretq_s32_u32(vceqq_s32(a, b));
  }
  static Vec blend(Vec mask, Vec a, Vec b) {
    return vbslq_s32(vreinterpretq_u32_s32(mask), a, b);
  }
};

#endif  // NEON

// =====================================================================
// LoongArch LSX (128-bit)
// =====================================================================
#if defined(__loongarch_sx)

struct LsxF32Ops {
  static constexpr std::size_t lanes = 4;
  using Cell = float;
  using Vec = __m128;

  static Cell inf_scalar() {
    return std::numeric_limits<float>::infinity();
  }
  static Vec set1(Cell x) {
    alignas(16) Cell tmp[4] = {x, x, x, x};
    return reinterpret_cast<Vec>(
        __lsx_vld(const_cast<void*>(static_cast<const void*>(tmp)), 0));
  }
  static Vec zero() { return set1(0.0f); }
  static Vec infinity() { return set1(inf_scalar()); }
  static Vec loadu(const Cell* p) {
    return reinterpret_cast<Vec>(
        __lsx_vld(const_cast<void*>(static_cast<const void*>(p)), 0));
  }
  static void storeu(Cell* p, Vec v) {
    __lsx_vst(reinterpret_cast<__m128i>(v), static_cast<void*>(p), 0);
  }
  static Vec add(Vec a, Vec b) {
    return reinterpret_cast<Vec>(
        __lsx_vfadd_s(reinterpret_cast<__m128>(a), reinterpret_cast<__m128>(b)));
  }
  static Vec sub(Vec a, Vec b) {
    return reinterpret_cast<Vec>(
        __lsx_vfsub_s(reinterpret_cast<__m128>(a), reinterpret_cast<__m128>(b)));
  }
  static Vec mul(Vec a, Vec b) {
    return reinterpret_cast<Vec>(
        __lsx_vfmul_s(reinterpret_cast<__m128>(a), reinterpret_cast<__m128>(b)));
  }
  static Vec min(Vec a, Vec b) {
    return reinterpret_cast<Vec>(
        __lsx_vfmin_s(reinterpret_cast<__m128>(a), reinterpret_cast<__m128>(b)));
  }
  static Vec abs(Vec a) {
    return reinterpret_cast<Vec>(
        __lsx_vbitclri_w(reinterpret_cast<__m128i>(a), 31));
  }
  static Vec cmp_eq(Vec a, Vec b) {
    return reinterpret_cast<Vec>(
        __lsx_vfcmp_ceq_s(reinterpret_cast<__m128>(a), reinterpret_cast<__m128>(b)));
  }
  static Vec blend(Vec mask, Vec a, Vec b) {
    // bitsel: (mask & a) | (~mask & b) via vbitsel_v(a, b, mask) =
    // (mask & a) | (not mask & b) — check ISA: vbitsel_v(vd, vj, vk) where
    // result = (vk & vj) | (~vk & vd). So vd=b, vj=a, vk=mask.
    return reinterpret_cast<Vec>(__lsx_vbitsel_v(
        reinterpret_cast<__m128i>(b),
        reinterpret_cast<__m128i>(a),
        reinterpret_cast<__m128i>(mask)));
  }
};

// Fix set1 to use set1_f

struct LsxF64Ops {
  static constexpr std::size_t lanes = 2;
  using Cell = double;
  using Vec = __m128d;

  static Cell inf_scalar() {
    return std::numeric_limits<double>::infinity();
  }
  static Vec set1(Cell x) {
    alignas(16) Cell tmp[2] = {x, x};
    return reinterpret_cast<Vec>(
        __lsx_vld(const_cast<void*>(static_cast<const void*>(tmp)), 0));
  }
  static Vec zero() { return set1(0.0); }
  static Vec infinity() { return set1(inf_scalar()); }
  static Vec loadu(const Cell* p) {
    return reinterpret_cast<Vec>(
        __lsx_vld(const_cast<void*>(static_cast<const void*>(p)), 0));
  }
  static void storeu(Cell* p, Vec v) {
    __lsx_vst(reinterpret_cast<__m128i>(v), static_cast<void*>(p), 0);
  }
  static Vec add(Vec a, Vec b) {
    return reinterpret_cast<Vec>(
        __lsx_vfadd_d(reinterpret_cast<__m128d>(a), reinterpret_cast<__m128d>(b)));
  }
  static Vec sub(Vec a, Vec b) {
    return reinterpret_cast<Vec>(
        __lsx_vfsub_d(reinterpret_cast<__m128d>(a), reinterpret_cast<__m128d>(b)));
  }
  static Vec mul(Vec a, Vec b) {
    return reinterpret_cast<Vec>(
        __lsx_vfmul_d(reinterpret_cast<__m128d>(a), reinterpret_cast<__m128d>(b)));
  }
  static Vec min(Vec a, Vec b) {
    return reinterpret_cast<Vec>(
        __lsx_vfmin_d(reinterpret_cast<__m128d>(a), reinterpret_cast<__m128d>(b)));
  }
  static Vec abs(Vec a) {
    return reinterpret_cast<Vec>(
        __lsx_vbitclri_d(reinterpret_cast<__m128i>(a), 63));
  }
  static Vec cmp_eq(Vec a, Vec b) {
    return reinterpret_cast<Vec>(
        __lsx_vfcmp_ceq_d(reinterpret_cast<__m128d>(a), reinterpret_cast<__m128d>(b)));
  }
  static Vec blend(Vec mask, Vec a, Vec b) {
    return reinterpret_cast<Vec>(__lsx_vbitsel_v(
        reinterpret_cast<__m128i>(b),
        reinterpret_cast<__m128i>(a),
        reinterpret_cast<__m128i>(mask)));
  }
};

struct LsxI32Ops {
  static constexpr std::size_t lanes = 4;
  using Cell = std::int32_t;
  using Vec = __m128i;

  static Cell inf_scalar() { return std::numeric_limits<std::int32_t>::max(); }
  static Vec set1(Cell x) { return __lsx_vreplgr2vr_w(x); }
  static Vec zero() { return __lsx_vreplgr2vr_w(0); }
  static Vec infinity() { return __lsx_vreplgr2vr_w(inf_scalar()); }
  static Vec loadu(const Cell* p) {
    return __lsx_vld(const_cast<void*>(static_cast<const void*>(p)), 0);
  }
  static void storeu(Cell* p, Vec v) {
    __lsx_vst(v, static_cast<void*>(p), 0);
  }
  static Vec add(Vec a, Vec b) { return __lsx_vadd_w(a, b); }
  static Vec sub(Vec a, Vec b) { return __lsx_vsub_w(a, b); }
  static Vec mul(Vec a, Vec b) { return __lsx_vmul_w(a, b); }
  static Vec min(Vec a, Vec b) { return __lsx_vmin_w(a, b); }
  static Vec abs(Vec a) { return __lsx_vabsd_w(a, zero()); }
  static Vec cmp_eq(Vec a, Vec b) { return __lsx_vseq_w(a, b); }
  static Vec blend(Vec mask, Vec a, Vec b) {
    return __lsx_vbitsel_v(b, a, mask);
  }
};

#endif  // __loongarch_sx

// =====================================================================
// LoongArch LASX (256-bit)
// =====================================================================
#if defined(__loongarch_asx)

struct LasxF32Ops {
  static constexpr std::size_t lanes = 8;
  using Cell = float;
  using Vec = __m256;

  static Cell inf_scalar() {
    return std::numeric_limits<float>::infinity();
  }
  static Vec set1(Cell x) {
    alignas(32) Cell tmp[8];
    for (int i = 0; i < 8; ++i) tmp[i] = x;
    return reinterpret_cast<Vec>(
        __lasx_xvld(const_cast<void*>(static_cast<const void*>(tmp)), 0));
  }
  static Vec zero() { return set1(0.0f); }
  static Vec infinity() { return set1(inf_scalar()); }
  static Vec loadu(const Cell* p) {
    return reinterpret_cast<Vec>(
        __lasx_xvld(const_cast<void*>(static_cast<const void*>(p)), 0));
  }
  static void storeu(Cell* p, Vec v) {
    __lasx_xvst(reinterpret_cast<__m256i>(v), static_cast<void*>(p), 0);
  }
  static Vec add(Vec a, Vec b) {
    return reinterpret_cast<Vec>(__lasx_xvfadd_s(
        reinterpret_cast<__m256>(a), reinterpret_cast<__m256>(b)));
  }
  static Vec sub(Vec a, Vec b) {
    return reinterpret_cast<Vec>(__lasx_xvfsub_s(
        reinterpret_cast<__m256>(a), reinterpret_cast<__m256>(b)));
  }
  static Vec mul(Vec a, Vec b) {
    return reinterpret_cast<Vec>(__lasx_xvfmul_s(
        reinterpret_cast<__m256>(a), reinterpret_cast<__m256>(b)));
  }
  static Vec min(Vec a, Vec b) {
    return reinterpret_cast<Vec>(__lasx_xvfmin_s(
        reinterpret_cast<__m256>(a), reinterpret_cast<__m256>(b)));
  }
  static Vec abs(Vec a) {
    return reinterpret_cast<Vec>(
        __lasx_xvbitclri_w(reinterpret_cast<__m256i>(a), 31));
  }
  static Vec cmp_eq(Vec a, Vec b) {
    return reinterpret_cast<Vec>(__lasx_xvfcmp_ceq_s(
        reinterpret_cast<__m256>(a), reinterpret_cast<__m256>(b)));
  }
  static Vec blend(Vec mask, Vec a, Vec b) {
    return reinterpret_cast<Vec>(__lasx_xvbitsel_v(
        reinterpret_cast<__m256i>(b),
        reinterpret_cast<__m256i>(a),
        reinterpret_cast<__m256i>(mask)));
  }
};

struct LasxF64Ops {
  static constexpr std::size_t lanes = 4;
  using Cell = double;
  using Vec = __m256d;

  static Cell inf_scalar() {
    return std::numeric_limits<double>::infinity();
  }
  static Vec set1(Cell x) {
    alignas(32) Cell tmp[4] = {x, x, x, x};
    return reinterpret_cast<Vec>(
        __lasx_xvld(const_cast<void*>(static_cast<const void*>(tmp)), 0));
  }
  static Vec zero() { return set1(0.0); }
  static Vec infinity() { return set1(inf_scalar()); }
  static Vec loadu(const Cell* p) {
    return reinterpret_cast<Vec>(
        __lasx_xvld(const_cast<void*>(static_cast<const void*>(p)), 0));
  }
  static void storeu(Cell* p, Vec v) {
    __lasx_xvst(reinterpret_cast<__m256i>(v), static_cast<void*>(p), 0);
  }
  static Vec add(Vec a, Vec b) {
    return reinterpret_cast<Vec>(__lasx_xvfadd_d(
        reinterpret_cast<__m256d>(a), reinterpret_cast<__m256d>(b)));
  }
  static Vec sub(Vec a, Vec b) {
    return reinterpret_cast<Vec>(__lasx_xvfsub_d(
        reinterpret_cast<__m256d>(a), reinterpret_cast<__m256d>(b)));
  }
  static Vec mul(Vec a, Vec b) {
    return reinterpret_cast<Vec>(__lasx_xvfmul_d(
        reinterpret_cast<__m256d>(a), reinterpret_cast<__m256d>(b)));
  }
  static Vec min(Vec a, Vec b) {
    return reinterpret_cast<Vec>(__lasx_xvfmin_d(
        reinterpret_cast<__m256d>(a), reinterpret_cast<__m256d>(b)));
  }
  static Vec abs(Vec a) {
    return reinterpret_cast<Vec>(
        __lasx_xvbitclri_d(reinterpret_cast<__m256i>(a), 63));
  }
  static Vec cmp_eq(Vec a, Vec b) {
    return reinterpret_cast<Vec>(__lasx_xvfcmp_ceq_d(
        reinterpret_cast<__m256d>(a), reinterpret_cast<__m256d>(b)));
  }
  static Vec blend(Vec mask, Vec a, Vec b) {
    return reinterpret_cast<Vec>(__lasx_xvbitsel_v(
        reinterpret_cast<__m256i>(b),
        reinterpret_cast<__m256i>(a),
        reinterpret_cast<__m256i>(mask)));
  }
};

struct LasxI32Ops {
  static constexpr std::size_t lanes = 8;
  using Cell = std::int32_t;
  using Vec = __m256i;

  static Cell inf_scalar() { return std::numeric_limits<std::int32_t>::max(); }
  static Vec set1(Cell x) { return __lasx_xvreplgr2vr_w(x); }
  static Vec zero() { return __lasx_xvreplgr2vr_w(0); }
  static Vec infinity() { return __lasx_xvreplgr2vr_w(inf_scalar()); }
  static Vec loadu(const Cell* p) {
    return __lasx_xvld(const_cast<void*>(static_cast<const void*>(p)), 0);
  }
  static void storeu(Cell* p, Vec v) {
    __lasx_xvst(v, static_cast<void*>(p), 0);
  }
  static Vec add(Vec a, Vec b) { return __lasx_xvadd_w(a, b); }
  static Vec sub(Vec a, Vec b) { return __lasx_xvsub_w(a, b); }
  static Vec mul(Vec a, Vec b) { return __lasx_xvmul_w(a, b); }
  static Vec min(Vec a, Vec b) { return __lasx_xvmin_w(a, b); }
  static Vec abs(Vec a) { return __lasx_xvabsd_w(a, zero()); }
  static Vec cmp_eq(Vec a, Vec b) { return __lasx_xvseq_w(a, b); }
  static Vec blend(Vec mask, Vec a, Vec b) {
    return __lasx_xvbitsel_v(b, a, mask);
  }
};

#endif  // __loongarch_asx

}  // namespace stride_align::dtw_simd
