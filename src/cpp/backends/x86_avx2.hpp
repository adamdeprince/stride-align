#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include <nanobind/nanobind.h>

#include "backends/farrar_fixed_kernel.hpp"
#include "backends/x86_fixed_kernel.hpp"

namespace stride_align::backend_avx2 {

namespace nb = nanobind;

#if defined(__GNUC__) || defined(__clang__)
#define STRIDE_ALIGN_X86_BASELINE __attribute__((target("default")))
#else
#define STRIDE_ALIGN_X86_BASELINE
#endif

template <int ShiftBytes>
inline __m256i shift_left_zero_256(__m256i vector) {
  const __m128i low = _mm256_castsi256_si128(vector);
  const __m128i high = _mm256_extracti128_si256(vector, 1);
  const __m128i low_shifted = _mm_slli_si128(low, ShiftBytes);
  const __m128i high_shifted =
      _mm_or_si128(_mm_slli_si128(high, ShiftBytes), _mm_srli_si128(low, 16 - ShiftBytes));
  return _mm256_inserti128_si256(_mm256_castsi128_si256(low_shifted), high_shifted, 1);
}

template <typename Token, typename Cell>
struct SimdOps;

template <>
struct SimdOps<std::uint8_t, std::int8_t> {
  using vector_type = __m256i;
  static constexpr std::size_t alignment = 32;
  static constexpr std::size_t lane_count = 32;
  static constexpr bool has_vector_max = true;

  static vector_type load_tokens(const std::uint8_t* values) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }

  static vector_type load_cells(const std::int8_t* values) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }

  static void store_cells(std::int8_t* values, vector_type vector) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(values), vector);
  }

  static vector_type set1(std::int8_t value) {
    return _mm256_set1_epi8(value);
  }

  static vector_type zero() {
    return _mm256_setzero_si256();
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return _mm256_add_epi8(lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return _mm256_max_epi8(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return shift_left_zero_256<1>(vector);
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return _mm256_movemask_epi8(_mm256_cmpgt_epi8(lhs, rhs)) != 0;
  }

  static vector_type substitution(
      const std::uint8_t* query,
      const std::uint8_t* target,
      std::int8_t match_score,
      std::int8_t mismatch_score) {
    const vector_type mask = _mm256_cmpeq_epi8(load_tokens(query), load_tokens(target));
    return _mm256_blendv_epi8(set1(mismatch_score), set1(match_score), mask);
  }
};

template <>
struct SimdOps<std::uint16_t, std::int16_t> {
  using vector_type = __m256i;
  static constexpr std::size_t alignment = 32;
  static constexpr std::size_t lane_count = 16;
  static constexpr bool has_vector_max = true;

  static vector_type load_tokens(const std::uint16_t* values) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }

  static vector_type load_cells(const std::int16_t* values) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }

  static void store_cells(std::int16_t* values, vector_type vector) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(values), vector);
  }

  static vector_type set1(std::int16_t value) {
    return _mm256_set1_epi16(value);
  }

  static vector_type zero() {
    return _mm256_setzero_si256();
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return _mm256_add_epi16(lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return _mm256_max_epi16(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return shift_left_zero_256<2>(vector);
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return _mm256_movemask_epi8(_mm256_cmpgt_epi16(lhs, rhs)) != 0;
  }

  static vector_type substitution(
      const std::uint16_t* query,
      const std::uint16_t* target,
      std::int16_t match_score,
      std::int16_t mismatch_score) {
    const vector_type mask = _mm256_cmpeq_epi16(load_tokens(query), load_tokens(target));
    return _mm256_blendv_epi8(set1(mismatch_score), set1(match_score), mask);
  }
};

template <>
struct SimdOps<std::uint32_t, std::int32_t> {
  using vector_type = __m256i;
  static constexpr std::size_t alignment = 32;
  static constexpr std::size_t lane_count = 8;
  static constexpr bool has_vector_max = true;

  static vector_type load_tokens(const std::uint32_t* values) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }

  static vector_type load_cells(const std::int32_t* values) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }

  static void store_cells(std::int32_t* values, vector_type vector) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(values), vector);
  }

  static vector_type set1(std::int32_t value) {
    return _mm256_set1_epi32(value);
  }

  static vector_type zero() {
    return _mm256_setzero_si256();
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return _mm256_add_epi32(lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return _mm256_max_epi32(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return shift_left_zero_256<4>(vector);
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return _mm256_movemask_epi8(_mm256_cmpgt_epi32(lhs, rhs)) != 0;
  }

  static vector_type substitution(
      const std::uint32_t* query,
      const std::uint32_t* target,
      std::int32_t match_score,
      std::int32_t mismatch_score) {
    const vector_type mask = _mm256_cmpeq_epi32(load_tokens(query), load_tokens(target));
    return _mm256_blendv_epi8(set1(mismatch_score), set1(match_score), mask);
  }
};

template <>
struct SimdOps<std::uint64_t, std::int64_t> {
  using vector_type = __m256i;
  static constexpr std::size_t alignment = 32;
  static constexpr std::size_t lane_count = 4;
  static constexpr bool has_vector_max = false;

  static vector_type load_tokens(const std::uint64_t* values) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }

  static vector_type load_cells(const std::int64_t* values) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }

  static void store_cells(std::int64_t* values, vector_type vector) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(values), vector);
  }

  static vector_type set1(std::int64_t value) {
    return _mm256_set1_epi64x(value);
  }

  static vector_type zero() {
    return _mm256_setzero_si256();
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return _mm256_add_epi64(lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    const vector_type mask = _mm256_cmpgt_epi64(lhs, rhs);
    return _mm256_blendv_epi8(rhs, lhs, mask);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return shift_left_zero_256<8>(vector);
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return _mm256_movemask_epi8(_mm256_cmpgt_epi64(lhs, rhs)) != 0;
  }

  static vector_type substitution(
      const std::uint64_t* query,
      const std::uint64_t* target,
      std::int64_t match_score,
      std::int64_t mismatch_score) {
    const vector_type mask = _mm256_cmpeq_epi64(load_tokens(query), load_tokens(target));
    return _mm256_blendv_epi8(set1(mismatch_score), set1(match_score), mask);
  }
};

struct TargetImplementation {
  static Score smith_waterman_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return x86_fixed_kernel::detail::dispatch_score<SimdOps, true>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static AlignmentResult smith_waterman_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return x86_fixed_kernel::detail::dispatch_traceback<SimdOps, true>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static Score smith_waterman_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_farrar_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::dispatch_score<SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static Score needleman_wunsch_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return x86_fixed_kernel::detail::dispatch_score<SimdOps, false>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static AlignmentResult needleman_wunsch_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return x86_fixed_kernel::detail::dispatch_traceback<SimdOps, false>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }
};

struct Implementation {
  static STRIDE_ALIGN_X86_BASELINE bool supported_on_this_machine() noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2") != 0;
#else
    return true;
#endif
  }

  static STRIDE_ALIGN_X86_BASELINE void ensure_supported() {
    if (supported_on_this_machine()) {
      return;
    }

    PyErr_SetString(PyExc_RuntimeError, "x86 AVX2 backend is not available on this machine");
    throw nb::python_error();
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE AlignmentResult smith_waterman_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_path(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_farrar_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE Score needleman_wunsch_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE AlignmentResult needleman_wunsch_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_path(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }
};

#undef STRIDE_ALIGN_X86_BASELINE

}  // namespace stride_align::backend_avx2
