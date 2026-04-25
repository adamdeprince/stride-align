#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include <nanobind/nanobind.h>

#include "backends/x86_fixed_kernel.hpp"

namespace stride_align::backend_avx512bwvl {

namespace nb = nanobind;

#if defined(__GNUC__) || defined(__clang__)
#define STRIDE_ALIGN_X86_BASELINE __attribute__((target("default")))
#else
#define STRIDE_ALIGN_X86_BASELINE
#endif

template <typename Token, typename Cell>
struct SimdOps;

template <>
struct SimdOps<std::uint8_t, std::int8_t> {
  using vector_type = __m512i;
  static constexpr std::size_t alignment = 64;
  static constexpr std::size_t lane_count = 64;
  static constexpr bool has_vector_max = true;

  static vector_type load_tokens(const std::uint8_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_cells(const std::int8_t* values) {
    return _mm512_loadu_si512(values);
  }

  static void store_cells(std::int8_t* values, vector_type vector) {
    _mm512_storeu_si512(values, vector);
  }

  static vector_type set1(std::int8_t value) {
    return _mm512_set1_epi8(value);
  }

  static vector_type zero() {
    return _mm512_setzero_si512();
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return _mm512_add_epi8(lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return _mm512_max_epi8(lhs, rhs);
  }

  static vector_type substitution(
      const std::uint8_t* query,
      const std::uint8_t* target,
      std::int8_t match_score,
      std::int8_t mismatch_score) {
    const __mmask64 mask = _mm512_cmpeq_epi8_mask(load_tokens(query), load_tokens(target));
    return _mm512_mask_blend_epi8(mask, set1(mismatch_score), set1(match_score));
  }
};

template <>
struct SimdOps<std::uint16_t, std::int16_t> {
  using vector_type = __m512i;
  static constexpr std::size_t alignment = 64;
  static constexpr std::size_t lane_count = 32;
  static constexpr bool has_vector_max = true;

  static vector_type load_tokens(const std::uint16_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_cells(const std::int16_t* values) {
    return _mm512_loadu_si512(values);
  }

  static void store_cells(std::int16_t* values, vector_type vector) {
    _mm512_storeu_si512(values, vector);
  }

  static vector_type set1(std::int16_t value) {
    return _mm512_set1_epi16(value);
  }

  static vector_type zero() {
    return _mm512_setzero_si512();
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return _mm512_add_epi16(lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return _mm512_max_epi16(lhs, rhs);
  }

  static vector_type substitution(
      const std::uint16_t* query,
      const std::uint16_t* target,
      std::int16_t match_score,
      std::int16_t mismatch_score) {
    const __mmask32 mask = _mm512_cmpeq_epi16_mask(load_tokens(query), load_tokens(target));
    return _mm512_mask_blend_epi16(mask, set1(mismatch_score), set1(match_score));
  }
};

template <>
struct SimdOps<std::uint32_t, std::int32_t> {
  using vector_type = __m512i;
  static constexpr std::size_t alignment = 64;
  static constexpr std::size_t lane_count = 16;
  static constexpr bool has_vector_max = true;

  static vector_type load_tokens(const std::uint32_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_cells(const std::int32_t* values) {
    return _mm512_loadu_si512(values);
  }

  static void store_cells(std::int32_t* values, vector_type vector) {
    _mm512_storeu_si512(values, vector);
  }

  static vector_type set1(std::int32_t value) {
    return _mm512_set1_epi32(value);
  }

  static vector_type zero() {
    return _mm512_setzero_si512();
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return _mm512_add_epi32(lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return _mm512_max_epi32(lhs, rhs);
  }

  static vector_type substitution(
      const std::uint32_t* query,
      const std::uint32_t* target,
      std::int32_t match_score,
      std::int32_t mismatch_score) {
    const __mmask16 mask = _mm512_cmpeq_epi32_mask(load_tokens(query), load_tokens(target));
    return _mm512_mask_blend_epi32(mask, set1(mismatch_score), set1(match_score));
  }
};

template <>
struct SimdOps<std::uint64_t, std::int64_t> {
  using vector_type = __m512i;
  static constexpr std::size_t alignment = 64;
  static constexpr std::size_t lane_count = 8;
  static constexpr bool has_vector_max = true;

  static vector_type load_tokens(const std::uint64_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_cells(const std::int64_t* values) {
    return _mm512_loadu_si512(values);
  }

  static void store_cells(std::int64_t* values, vector_type vector) {
    _mm512_storeu_si512(values, vector);
  }

  static vector_type set1(std::int64_t value) {
    return _mm512_set1_epi64(value);
  }

  static vector_type zero() {
    return _mm512_setzero_si512();
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return _mm512_add_epi64(lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return _mm512_max_epi64(lhs, rhs);
  }

  static vector_type substitution(
      const std::uint64_t* query,
      const std::uint64_t* target,
      std::int64_t match_score,
      std::int64_t mismatch_score) {
    const __mmask8 mask = _mm512_cmpeq_epi64_mask(load_tokens(query), load_tokens(target));
    return _mm512_mask_blend_epi64(mask, set1(mismatch_score), set1(match_score));
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
    return __builtin_cpu_supports("avx512f") != 0 && __builtin_cpu_supports("avx512bw") != 0 &&
           __builtin_cpu_supports("avx512vl") != 0;
#else
    return true;
#endif
  }

  static STRIDE_ALIGN_X86_BASELINE void ensure_supported() {
    if (supported_on_this_machine()) {
      return;
    }

    PyErr_SetString(
        PyExc_RuntimeError,
        "x86 AVX-512BWVL backend is not available on this machine");
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

}  // namespace stride_align::backend_avx512bwvl
