#pragma once

// SIMD primitive bundles consumed by levenshtein_simd::myers_batch_single_word.
// Each bundle is gated by the matching __target__ pragma the backend was
// compiled with; backends that don't enable the corresponding ISA simply
// won't reference its bundle.

#include <cstdint>

#if defined(__SSE4_1__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace stride_align::levenshtein_simd {

#if defined(__SSE4_1__)

struct SseOps {
  static constexpr std::size_t lanes = 2;
  using Vec = __m128i;

  static Vec set1(std::uint64_t v) {
    return _mm_set1_epi64x(static_cast<long long>(v));
  }
  static Vec zero() { return _mm_setzero_si128(); }
  static Vec and_(Vec a, Vec b) { return _mm_and_si128(a, b); }
  static Vec or_(Vec a, Vec b) { return _mm_or_si128(a, b); }
  static Vec xor_(Vec a, Vec b) { return _mm_xor_si128(a, b); }
  static Vec not_(Vec a) {
    return _mm_xor_si128(a, _mm_set1_epi64x(static_cast<long long>(-1LL)));
  }
  static Vec add(Vec a, Vec b) { return _mm_add_epi64(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm_sub_epi64(a, b); }
  // shift left by 1 = add to self
  static Vec shl1(Vec a) { return _mm_add_epi64(a, a); }
  // Per-lane: all-ones if a == b, else zero (64-bit lane).
  static Vec cmpeq(Vec a, Vec b) { return _mm_cmpeq_epi64(a, b); }
  // (~a) & b
  static Vec andnot_(Vec a, Vec b) { return _mm_andnot_si128(a, b); }
  static Vec gather64(const std::uint64_t* base, const std::uint64_t* indices) {
    return _mm_set_epi64x(
        static_cast<long long>(base[indices[1]]),
        static_cast<long long>(base[indices[0]]));
  }
  static Vec load_aligned(const std::uint64_t* data) {
    return _mm_load_si128(reinterpret_cast<const __m128i*>(data));
  }
  static void store_aligned(std::uint64_t* dst, Vec v) {
    _mm_store_si128(reinterpret_cast<__m128i*>(dst), v);
  }
};

#endif  // __SSE4_1__

#if defined(__AVX2__)

struct Avx2Ops {
  static constexpr std::size_t lanes = 4;
  using Vec = __m256i;

  static Vec set1(std::uint64_t v) {
    return _mm256_set1_epi64x(static_cast<long long>(v));
  }
  static Vec zero() { return _mm256_setzero_si256(); }
  static Vec and_(Vec a, Vec b) { return _mm256_and_si256(a, b); }
  static Vec or_(Vec a, Vec b) { return _mm256_or_si256(a, b); }
  static Vec xor_(Vec a, Vec b) { return _mm256_xor_si256(a, b); }
  static Vec not_(Vec a) {
    return _mm256_xor_si256(a, _mm256_set1_epi64x(static_cast<long long>(-1LL)));
  }
  static Vec add(Vec a, Vec b) { return _mm256_add_epi64(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm256_sub_epi64(a, b); }
  static Vec shl1(Vec a) { return _mm256_add_epi64(a, a); }
  static Vec cmpeq(Vec a, Vec b) { return _mm256_cmpeq_epi64(a, b); }
  static Vec andnot_(Vec a, Vec b) { return _mm256_andnot_si256(a, b); }
  static Vec gather64(const std::uint64_t* base, const std::uint64_t* indices) {
    const __m256i idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices));
    return _mm256_i64gather_epi64(
        reinterpret_cast<const long long*>(base), idx, 8);
  }
  static Vec load_aligned(const std::uint64_t* data) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(data));
  }
  static void store_aligned(std::uint64_t* dst, Vec v) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(dst), v);
  }
};

#endif  // __AVX2__

#if defined(__AVX512F__)

struct Avx512Ops {
  static constexpr std::size_t lanes = 8;
  using Vec = __m512i;

  static Vec set1(std::uint64_t v) {
    return _mm512_set1_epi64(static_cast<long long>(v));
  }
  static Vec zero() { return _mm512_setzero_si512(); }
  static Vec and_(Vec a, Vec b) { return _mm512_and_si512(a, b); }
  static Vec or_(Vec a, Vec b) { return _mm512_or_si512(a, b); }
  static Vec xor_(Vec a, Vec b) { return _mm512_xor_si512(a, b); }
  static Vec not_(Vec a) {
    return _mm512_xor_si512(a, _mm512_set1_epi64(static_cast<long long>(-1LL)));
  }
  static Vec add(Vec a, Vec b) { return _mm512_add_epi64(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm512_sub_epi64(a, b); }
  static Vec shl1(Vec a) { return _mm512_add_epi64(a, a); }
  static Vec cmpeq(Vec a, Vec b) {
    // AVX-512 produces a mask; expand back to a vector of all-ones / zeros.
    return _mm512_maskz_set1_epi64(_mm512_cmpeq_epi64_mask(a, b), -1LL);
  }
  static Vec andnot_(Vec a, Vec b) { return _mm512_andnot_si512(a, b); }
  static Vec gather64(const std::uint64_t* base, const std::uint64_t* indices) {
    const __m512i idx = _mm512_load_si512(reinterpret_cast<const void*>(indices));
    return _mm512_i64gather_epi64(idx, reinterpret_cast<const void*>(base), 8);
  }
  static Vec load_aligned(const std::uint64_t* data) {
    return _mm512_load_si512(reinterpret_cast<const void*>(data));
  }
  static void store_aligned(std::uint64_t* dst, Vec v) {
    _mm512_store_si512(reinterpret_cast<void*>(dst), v);
  }
};

#endif  // __AVX512F__

}  // namespace stride_align::levenshtein_simd
