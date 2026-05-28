#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include <nanobind/nanobind.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC push_options
#pragma GCC target("avx10.1-512")
#endif

#include "backends/affine_fixed_kernel.hpp"
#include "backends/farrar_fixed_kernel.hpp"
#include "backends/profile_traceback.hpp"
#include "backends/x86_fixed_kernel.hpp"
#include "cdist_simd.hpp"
#include "cdist_threshold.hpp"
#include "cdist_topk.hpp"
#include "jaro_simd.hpp"
#include "levenshtein_simd.hpp"
#include "levenshtein_simd_ops.hpp"
#include "indel_simd.hpp"
#include "osa_simd.hpp"

namespace stride_align::backend_avx10_512 {

namespace nb = nanobind;

#if defined(__GNUC__) || defined(__clang__)
#define STRIDE_ALIGN_X86_BASELINE __attribute__((target("default")))
#else
#define STRIDE_ALIGN_X86_BASELINE
#endif

template <int ShiftBytes>
inline __m512i shift_left_zero_512(__m512i vector) {
  static_assert(ShiftBytes == 1 || ShiftBytes == 2 || ShiftBytes == 4 || ShiftBytes == 8);
  const __m512i previous_lane_indices = _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 0);
  if constexpr (ShiftBytes == 8) {
    return _mm512_maskz_permutexvar_epi64(0xFE, previous_lane_indices, vector);
  } else {
    const __m512i shifted = _mm512_slli_epi64(vector, ShiftBytes * 8);
    const __m512i carry = _mm512_srli_epi64(vector, 64 - ShiftBytes * 8);
    const __m512i carry_from_previous_lane =
        _mm512_maskz_permutexvar_epi64(0xFE, previous_lane_indices, carry);
    return _mm512_or_si512(shifted, carry_from_previous_lane);
  }
}

template <typename Token, typename Cell>
struct SimdOps;

template <>
struct SimdOps<std::uint8_t, std::int8_t> {
  using vector_type = __m512i;
  using mask_type = __mmask64;
  static constexpr std::size_t alignment = 64;
  static constexpr std::size_t lane_count = 64;
  static constexpr bool has_vector_max = true;

  static vector_type load_tokens(const std::uint8_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_cells(const std::int8_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_aligned_cells(const std::int8_t* values) {
    return _mm512_load_si512(values);
  }

  static void store_cells(std::int8_t* values, vector_type vector) {
    _mm512_storeu_si512(values, vector);
  }

  static void store_aligned_cells(std::int8_t* values, vector_type vector) {
    _mm512_store_si512(values, vector);
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

  static vector_type shift_left_zero(vector_type vector) {
    return shift_left_zero_512<1>(vector);
  }

  static vector_type shift_left_insert(vector_type vector, std::int8_t inserted) {
    return _mm512_mask_blend_epi8(0x1, shift_left_zero_512<1>(vector), set1(inserted));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return _mm512_cmpgt_epi8_mask(lhs, rhs) != 0;
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return _mm512_cmpgt_epi8_mask(lhs, rhs);
  }

  static mask_type empty_mask() {
    return 0;
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return lhs | rhs;
  }

  static bool any_mask(mask_type mask) {
    return mask != 0;
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
  using mask_type = __mmask32;
  static constexpr std::size_t alignment = 64;
  static constexpr std::size_t lane_count = 32;
  static constexpr bool has_vector_max = true;

  static vector_type load_tokens(const std::uint16_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_cells(const std::int16_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_aligned_cells(const std::int16_t* values) {
    return _mm512_load_si512(values);
  }

  static void store_cells(std::int16_t* values, vector_type vector) {
    _mm512_storeu_si512(values, vector);
  }

  static void store_aligned_cells(std::int16_t* values, vector_type vector) {
    _mm512_store_si512(values, vector);
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

  static vector_type shift_left_zero(vector_type vector) {
    return shift_left_zero_512<2>(vector);
  }

  static vector_type shift_left_insert(vector_type vector, std::int16_t inserted) {
    return _mm512_mask_blend_epi16(0x1, shift_left_zero_512<2>(vector), set1(inserted));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return _mm512_cmpgt_epi16_mask(lhs, rhs) != 0;
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return _mm512_cmpgt_epi16_mask(lhs, rhs);
  }

  static mask_type empty_mask() {
    return 0;
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return lhs | rhs;
  }

  static bool any_mask(mask_type mask) {
    return mask != 0;
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
  using mask_type = __mmask16;
  static constexpr std::size_t alignment = 64;
  static constexpr std::size_t lane_count = 16;
  static constexpr bool has_vector_max = true;

  static vector_type load_tokens(const std::uint32_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_cells(const std::int32_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_aligned_cells(const std::int32_t* values) {
    return _mm512_load_si512(values);
  }

  static void store_cells(std::int32_t* values, vector_type vector) {
    _mm512_storeu_si512(values, vector);
  }

  static void store_aligned_cells(std::int32_t* values, vector_type vector) {
    _mm512_store_si512(values, vector);
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

  static vector_type shift_left_zero(vector_type vector) {
    return shift_left_zero_512<4>(vector);
  }

  static vector_type shift_left_insert(vector_type vector, std::int32_t inserted) {
    return _mm512_mask_blend_epi32(0x1, shift_left_zero_512<4>(vector), set1(inserted));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return _mm512_cmpgt_epi32_mask(lhs, rhs) != 0;
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return _mm512_cmpgt_epi32_mask(lhs, rhs);
  }

  static mask_type empty_mask() {
    return 0;
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return lhs | rhs;
  }

  static bool any_mask(mask_type mask) {
    return mask != 0;
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
  using mask_type = __mmask8;
  static constexpr std::size_t alignment = 64;
  static constexpr std::size_t lane_count = 8;
  static constexpr bool has_vector_max = true;

  static vector_type load_tokens(const std::uint64_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_cells(const std::int64_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_aligned_cells(const std::int64_t* values) {
    return _mm512_load_si512(values);
  }

  static void store_cells(std::int64_t* values, vector_type vector) {
    _mm512_storeu_si512(values, vector);
  }

  static void store_aligned_cells(std::int64_t* values, vector_type vector) {
    _mm512_store_si512(values, vector);
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

  static vector_type shift_left_zero(vector_type vector) {
    return shift_left_zero_512<8>(vector);
  }

  static vector_type shift_left_insert(vector_type vector, std::int64_t inserted) {
    return _mm512_mask_blend_epi64(0x1, shift_left_zero_512<8>(vector), set1(inserted));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return _mm512_cmpgt_epi64_mask(lhs, rhs) != 0;
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return _mm512_cmpgt_epi64_mask(lhs, rhs);
  }

  static mask_type empty_mask() {
    return 0;
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return lhs | rhs;
  }

  static bool any_mask(mask_type mask) {
    return mask != 0;
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
  using PreparedSmithWatermanFarrarScore =
      farrar_fixed_kernel::detail::PreparedScore<SimdOps>;
  using PreparedAffineScore =
      farrar_fixed_kernel::detail::PreparedAffineScore<SimdOps>;

  static Score smith_waterman_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    // Wide-uint64 ndarray dispatch: matching 64-bit SIMD dtypes
    // bypass tokenisation; Cell width is int64 in the kernel.
    {
      namespace nv = ::stride_align::numpy_view;
      auto q_view = nv::try_acquire(query.ptr());
      auto t_view = nv::try_acquire(target.ptr());
      if (::stride_align::farrar_detail::ndarray_pair_is_wide_uint64(q_view, t_view)) {
        const auto wide = ::stride_align::farrar_detail::prepare_farrar_alignment_wide_uint64(
            q_view, t_view, match_score, mismatch_score, gap_score, gap_score, width);
        return farrar_fixed_kernel::detail::wide_dispatch_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
    // Wide-uint32 ndarray dispatch: matching 32-bit SIMD dtypes
    // bypass tokenisation; Cell width is int32 in the kernel.
    {
      namespace nv = ::stride_align::numpy_view;
      auto q_view = nv::try_acquire(query.ptr());
      auto t_view = nv::try_acquire(target.ptr());
      if (::stride_align::farrar_detail::ndarray_pair_is_wide_uint32(q_view, t_view)) {
        const auto wide = ::stride_align::farrar_detail::prepare_farrar_alignment_wide_uint32(
            q_view, t_view, match_score, mismatch_score, gap_score, gap_score, width);
        return farrar_fixed_kernel::detail::wide_dispatch_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
    // Wide-uint16 ndarray dispatch (Phase B parallel path). When both
    // sides are 2-byte SIMD-friendly numpy dtypes (int16/uint16/float16)
    // we skip tokenisation entirely and feed the buffers straight into
    // the 16-bit Farrar kernel.
    {
      namespace nv = ::stride_align::numpy_view;
      auto q_view = nv::try_acquire(query.ptr());
      auto t_view = nv::try_acquire(target.ptr());
      if (::stride_align::farrar_detail::ndarray_pair_is_wide_uint16(q_view, t_view)) {
        const auto wide = ::stride_align::farrar_detail::prepare_farrar_alignment_wide_uint16(
            q_view, t_view, match_score, mismatch_score, gap_score, gap_score, width);
        return farrar_fixed_kernel::detail::wide_dispatch_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
    // Unicode auto-promote: UCS-2 / UCS-4 strings with >256 distinct
    // codepoints route to the 16-bit Farrar kernel via uint16 tokens.
    // If the combined alphabet also exceeds 65 535 distinct codepoints
    // we fall through to the 32-bit Farrar kernel feeding raw UCS-4
    // codepoints as uint32 tokens (no tokenisation map).
    // UCS-1 strings (Latin-1) can have at most 256 distinct codepoints
    // by construction, so we skip the pre-scan entirely for them.
    if (PyUnicode_Check(query.ptr()) && PyUnicode_Check(target.ptr()) &&
        (PyUnicode_KIND(query.ptr()) != PyUnicode_1BYTE_KIND ||
         PyUnicode_KIND(target.ptr()) != PyUnicode_1BYTE_KIND)) {
      if (!::stride_align::farrar_detail::unicode_distinct_count_within(
              query.ptr(), target.ptr(), 256U)) {
        if (!::stride_align::farrar_detail::unicode_alphabet_within_uint16(
                query.ptr(), target.ptr())) {
          const auto wide =
              ::stride_align::farrar_detail::prepare_farrar_alignment_wide_unicode_uint32(
                  query.ptr(), target.ptr(),
                  match_score, mismatch_score, gap_score, gap_score, width);
          return farrar_fixed_kernel::detail::wide_dispatch_score<SimdOps>(
              wide, match_score, mismatch_score, gap_score);
        }
        const auto wide =
            ::stride_align::farrar_detail::prepare_farrar_alignment_wide_unicode_uint16(
                query.ptr(), target.ptr(),
                match_score, mismatch_score, gap_score, gap_score, width);
        return farrar_fixed_kernel::detail::wide_dispatch_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
    if (gap_score > 0) {
      const auto prepared =
          prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
      return x86_fixed_kernel::detail::dispatch_score<SimdOps, true>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    }
    const auto prepared =
        prepare_linear_score_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::dispatch_score<SimdOps>(
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
    return profile_traceback::linear_path<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static AlignmentPath smith_waterman_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return profile_traceback::linear_path_info<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
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

  static PreparedSmithWatermanFarrarScore prepare_smith_waterman_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_farrar_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::prepare_score<SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static Score smith_waterman_farrar_score_prepared(
      PreparedSmithWatermanFarrarScore& prepared) {
    return farrar_fixed_kernel::detail::dispatch_prepared_score<SimdOps>(prepared);
  }

  static Score smith_waterman_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    if (auto wide = farrar_fixed_kernel::detail::try_wide_affine_score<SimdOps, true>(
            query, target, match_score, mismatch_score,
            gap_open_score, gap_extend_score, width);
        wide) {
      return *wide;
    }
    if (gap_open_score <= 0 && gap_extend_score <= 0) {
      const auto prepared = prepare_farrar_alignment(
          query,
          target,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score,
          width);
      return affine_fixed_kernel::detail::dispatch_compact_byte_score<SimdOps>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    }
    return profile_traceback::affine_score<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static AlignmentResult smith_waterman_affine_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    return profile_traceback::affine_path<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static AlignmentPath smith_waterman_affine_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    return profile_traceback::affine_path_info<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static Score smith_waterman_affine_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    const auto prepared = prepare_farrar_alignment(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    return affine_fixed_kernel::detail::dispatch_compact_byte_score<SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static PreparedAffineScore prepare_smith_waterman_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    const auto prepared = prepare_farrar_alignment(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    return farrar_fixed_kernel::detail::prepare_affine_score<SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static Score smith_waterman_affine_score_prepared(PreparedAffineScore& prepared) {
    return farrar_fixed_kernel::detail::dispatch_prepared_affine_score<SimdOps>(prepared);
  }

  static PreparedAffineScore prepare_smith_waterman_affine_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    return prepare_smith_waterman_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static Score smith_waterman_affine_farrar_score_prepared(PreparedAffineScore& prepared) {
    return smith_waterman_affine_score_prepared(prepared);
  }

  static PreparedAffineScore prepare_needleman_wunsch_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    const auto prepared = prepare_farrar_alignment(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    return farrar_fixed_kernel::detail::prepare_affine_score<SimdOps, true>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static Score needleman_wunsch_affine_score_prepared(PreparedAffineScore& prepared) {
    return farrar_fixed_kernel::detail::dispatch_prepared_global_affine_score<SimdOps>(prepared);
  }

  static Score needleman_wunsch_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    // Wide-uint64 ndarray dispatch: matching 64-bit SIMD dtypes
    // bypass tokenisation; Cell width is int64 in the kernel.
    {
      namespace nv = ::stride_align::numpy_view;
      auto q_view = nv::try_acquire(query.ptr());
      auto t_view = nv::try_acquire(target.ptr());
      if (::stride_align::farrar_detail::ndarray_pair_is_wide_uint64(q_view, t_view)) {
        const auto wide = ::stride_align::farrar_detail::prepare_farrar_alignment_wide_uint64(
            q_view, t_view, match_score, mismatch_score, gap_score, gap_score, width);
        return farrar_fixed_kernel::detail::wide_dispatch_global_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
    // Wide-uint32 ndarray dispatch: matching 32-bit SIMD dtypes
    // bypass tokenisation; Cell width is int32 in the kernel.
    {
      namespace nv = ::stride_align::numpy_view;
      auto q_view = nv::try_acquire(query.ptr());
      auto t_view = nv::try_acquire(target.ptr());
      if (::stride_align::farrar_detail::ndarray_pair_is_wide_uint32(q_view, t_view)) {
        const auto wide = ::stride_align::farrar_detail::prepare_farrar_alignment_wide_uint32(
            q_view, t_view, match_score, mismatch_score, gap_score, gap_score, width);
        return farrar_fixed_kernel::detail::wide_dispatch_global_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
    {
      namespace nv = ::stride_align::numpy_view;
      auto q_view = nv::try_acquire(query.ptr());
      auto t_view = nv::try_acquire(target.ptr());
      if (::stride_align::farrar_detail::ndarray_pair_is_wide_uint16(q_view, t_view)) {
        const auto wide = ::stride_align::farrar_detail::prepare_farrar_alignment_wide_uint16(
            q_view, t_view, match_score, mismatch_score, gap_score, gap_score, width);
        return farrar_fixed_kernel::detail::wide_dispatch_global_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
    // Unicode auto-promote: UCS-2 / UCS-4 strings with >256 distinct
    // codepoints route to the 16-bit Farrar kernel via uint16 tokens.
    // If the combined alphabet also exceeds 65 535 distinct codepoints
    // we fall through to the 32-bit Farrar kernel feeding raw UCS-4
    // codepoints as uint32 tokens (no tokenisation map).
    // UCS-1 strings (Latin-1) can have at most 256 distinct codepoints
    // by construction, so we skip the pre-scan entirely for them.
    if (PyUnicode_Check(query.ptr()) && PyUnicode_Check(target.ptr()) &&
        (PyUnicode_KIND(query.ptr()) != PyUnicode_1BYTE_KIND ||
         PyUnicode_KIND(target.ptr()) != PyUnicode_1BYTE_KIND)) {
      if (!::stride_align::farrar_detail::unicode_distinct_count_within(
              query.ptr(), target.ptr(), 256U)) {
        if (!::stride_align::farrar_detail::unicode_alphabet_within_uint16(
                query.ptr(), target.ptr())) {
          const auto wide =
              ::stride_align::farrar_detail::prepare_farrar_alignment_wide_unicode_uint32(
                  query.ptr(), target.ptr(),
                  match_score, mismatch_score, gap_score, gap_score, width);
          return farrar_fixed_kernel::detail::wide_dispatch_global_score<SimdOps>(
              wide, match_score, mismatch_score, gap_score);
        }
        const auto wide =
            ::stride_align::farrar_detail::prepare_farrar_alignment_wide_unicode_uint16(
                query.ptr(), target.ptr(),
                match_score, mismatch_score, gap_score, gap_score, width);
        return farrar_fixed_kernel::detail::wide_dispatch_global_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
    if (gap_score > 0) {
      const auto prepared =
          prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
      return x86_fixed_kernel::detail::dispatch_score<SimdOps, false>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    }
    const auto prepared =
        prepare_linear_score_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::dispatch_global_score<SimdOps>(
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
    return profile_traceback::linear_path<false>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static AlignmentPath needleman_wunsch_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return profile_traceback::linear_path_info<false>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static Score needleman_wunsch_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    if (auto wide = farrar_fixed_kernel::detail::try_wide_affine_score<SimdOps, false>(
            query, target, match_score, mismatch_score,
            gap_open_score, gap_extend_score, width);
        wide) {
      return *wide;
    }
    const auto prepared = prepare_farrar_alignment(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    return farrar_fixed_kernel::detail::dispatch_global_affine_score<SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static AlignmentResult needleman_wunsch_affine_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    return profile_traceback::affine_path<false>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static AlignmentPath needleman_wunsch_affine_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    return profile_traceback::affine_path_info<false>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static std::vector<Score> levenshtein_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    return ::stride_align::levenshtein_simd::levenshtein_scores_simd<
        ::stride_align::levenshtein_simd::Avx512Ops>(query, targets, cutoff);
  }

  static std::vector<double> levenshtein_normalized_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    return ::stride_align::levenshtein_simd::levenshtein_normalized_scores_simd<
        ::stride_align::levenshtein_simd::Avx512Ops>(query, targets, cutoff);
  }

  static std::vector<Score> damerau_levenshtein_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::osa_simd::osa_scores_simd<
        ::stride_align::levenshtein_simd::Avx512Ops>(query, targets);
  }

  static std::vector<double> damerau_levenshtein_normalized_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::osa_simd::osa_normalized_scores_simd<
        ::stride_align::levenshtein_simd::Avx512Ops>(query, targets);
  }

  static std::vector<Score> indel_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::indel_simd::indel_scores_simd<
        ::stride_align::levenshtein_simd::Avx512Ops>(query, targets);
  }

  static std::vector<double> indel_normalized_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::indel_simd::indel_normalized_scores_simd<
        ::stride_align::levenshtein_simd::Avx512Ops>(query, targets);
  }

  static std::vector<double> jaro_similarities(
      nb::handle query, nb::handle targets) {
    return ::stride_align::jaro_simd::jaro_similarities_simd<
        ::stride_align::levenshtein_simd::Avx512Ops>(query, targets);
  }

  static std::vector<double> jaro_winkler_similarities(
      nb::handle query,
      nb::handle targets,
      double prefix_weight,
      double prefix_threshold,
      std::size_t prefix_cap) {
    return ::stride_align::jaro_simd::jaro_winkler_similarities_simd<
        ::stride_align::levenshtein_simd::Avx512Ops>(
        query, targets, prefix_weight, prefix_threshold, prefix_cap);
  }

  static nb::object cdist(
      nb::handle queries, nb::handle targets, int scorer,
      nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    return ::stride_align::cdist_simd::cdist_impl<
        ::stride_align::levenshtein_simd::Avx512Ops>(
        queries, targets, scorer, tqdm_factory, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static nb::object cdist_above_threshold(
      nb::handle queries, nb::handle targets, int scorer,
      double threshold, nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    return ::stride_align::cdist_threshold::cdist_threshold_impl<
        ::stride_align::levenshtein_simd::Avx512Ops>(
        queries, targets, scorer, threshold, tqdm_factory, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static nb::object cdist_top_k(
      nb::handle queries, nb::handle targets, int scorer,
      std::size_t k, nb::object tqdm_factory, std::size_t cpu_count,
      bool reject_duplicates,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    return ::stride_align::cdist_topk::cdist_top_k_impl<
        ::stride_align::levenshtein_simd::Avx512Ops>(
        queries, targets, scorer, k, tqdm_factory, cpu_count,
        reject_duplicates,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  // ----- Matrix-mode entry points -------------------------------------
  // Forwards to the shared dispatch helpers in farrar_fixed_kernel.hpp
  // with this backend's SimdOps. See backend_avx512bwvl for full docs.
  static Score smith_waterman_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    return farrar_fixed_kernel::detail::matrix_score_dispatch_helper<SimdOps, true>(
        query_indices, target_indices, matrix_buffer, stride, gap_score);
  }

  static Score needleman_wunsch_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    return farrar_fixed_kernel::detail::matrix_score_dispatch_helper<SimdOps, false>(
        query_indices, target_indices, matrix_buffer, stride, gap_score);
  }

  static std::vector<Score> smith_waterman_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    return farrar_fixed_kernel::detail::matrix_scores_dispatch_helper<SimdOps, true>(
        query_indices, targets, matrix_buffer, stride, gap_score);
  }

  static std::vector<Score> needleman_wunsch_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    return farrar_fixed_kernel::detail::matrix_scores_dispatch_helper<SimdOps, false>(
        query_indices, targets, matrix_buffer, stride, gap_score);
  }

  static Score smith_waterman_affine_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    return farrar_fixed_kernel::detail::matrix_affine_score_dispatch_helper<SimdOps, true>(
        query_indices, target_indices, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static Score needleman_wunsch_affine_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    return farrar_fixed_kernel::detail::matrix_affine_score_dispatch_helper<SimdOps, false>(
        query_indices, target_indices, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static std::vector<Score> smith_waterman_affine_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    return farrar_fixed_kernel::detail::matrix_affine_scores_dispatch_helper<SimdOps, true>(
        query_indices, targets, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static std::vector<Score> needleman_wunsch_affine_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    return farrar_fixed_kernel::detail::matrix_affine_scores_dispatch_helper<SimdOps, false>(
        query_indices, targets, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }
};

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC pop_options
#endif

struct Implementation {
  using PreparedSmithWatermanFarrarScore =
      TargetImplementation::PreparedSmithWatermanFarrarScore;
  using PreparedAffineScore = TargetImplementation::PreparedAffineScore;

  static STRIDE_ALIGN_X86_BASELINE bool supported_on_this_machine() noexcept {
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 14)
    return __builtin_cpu_supports("avx512f") != 0 && __builtin_cpu_supports("avx512bw") != 0 &&
        __builtin_cpu_supports("avx512vl") != 0 && __builtin_cpu_supports("avx10.1-512") != 0;
#else
    return false;
#endif
  }

  static STRIDE_ALIGN_X86_BASELINE void ensure_supported() {
    if (supported_on_this_machine()) {
      return;
    }

    PyErr_SetString(PyExc_RuntimeError, "x86 AVX10.1 512-bit backend is not available on this machine");
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

  static STRIDE_ALIGN_X86_BASELINE AlignmentPath smith_waterman_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_path_info(
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

  static STRIDE_ALIGN_X86_BASELINE PreparedSmithWatermanFarrarScore
  prepare_smith_waterman_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_smith_waterman_farrar_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_farrar_score_prepared(
      PreparedSmithWatermanFarrarScore& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_farrar_score_prepared(prepared);
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE AlignmentResult smith_waterman_affine_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_path(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE AlignmentPath smith_waterman_affine_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_path_info(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_affine_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_farrar_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE PreparedAffineScore prepare_smith_waterman_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_smith_waterman_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_affine_score_prepared(PreparedAffineScore& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_score_prepared(prepared);
  }

  static STRIDE_ALIGN_X86_BASELINE PreparedAffineScore prepare_smith_waterman_affine_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_smith_waterman_affine_farrar_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_affine_farrar_score_prepared(
      PreparedAffineScore& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_farrar_score_prepared(prepared);
  }

  static STRIDE_ALIGN_X86_BASELINE PreparedAffineScore prepare_needleman_wunsch_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_needleman_wunsch_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE Score needleman_wunsch_affine_score_prepared(PreparedAffineScore& prepared) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_score_prepared(prepared);
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

  static STRIDE_ALIGN_X86_BASELINE AlignmentPath needleman_wunsch_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_path_info(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE Score needleman_wunsch_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE AlignmentResult needleman_wunsch_affine_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_path(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE AlignmentPath needleman_wunsch_affine_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_path_info(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> levenshtein_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    ensure_supported();
    return TargetImplementation::levenshtein_scores(query, targets, cutoff);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<double> levenshtein_normalized_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    ensure_supported();
    return TargetImplementation::levenshtein_normalized_scores(query, targets, cutoff);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> damerau_levenshtein_scores(
      nb::handle query, nb::handle targets) {
    ensure_supported();
    return TargetImplementation::damerau_levenshtein_scores(query, targets);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<double>
  damerau_levenshtein_normalized_scores(nb::handle query, nb::handle targets) {
    ensure_supported();
    return TargetImplementation::damerau_levenshtein_normalized_scores(query, targets);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<double>
  jaro_similarities(nb::handle query, nb::handle targets) {
    ensure_supported();
    return TargetImplementation::jaro_similarities(query, targets);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<double>
  jaro_winkler_similarities(
      nb::handle query,
      nb::handle targets,
      double prefix_weight,
      double prefix_threshold,
      std::size_t prefix_cap) {
    ensure_supported();
    return TargetImplementation::jaro_winkler_similarities(
        query, targets, prefix_weight, prefix_threshold, prefix_cap);
  }

  static STRIDE_ALIGN_X86_BASELINE nb::object cdist(
      nb::handle queries, nb::handle targets, int scorer,
      nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    ensure_supported();
    return TargetImplementation::cdist(
        queries, targets, scorer, tqdm_factory, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static STRIDE_ALIGN_X86_BASELINE nb::object cdist_above_threshold(
      nb::handle queries, nb::handle targets, int scorer,
      double threshold, nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    ensure_supported();
    return TargetImplementation::cdist_above_threshold(
        queries, targets, scorer, threshold, tqdm_factory, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static STRIDE_ALIGN_X86_BASELINE nb::object cdist_top_k(
      nb::handle queries, nb::handle targets, int scorer,
      std::size_t k, nb::object tqdm_factory, std::size_t cpu_count,
      bool reject_duplicates,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    ensure_supported();
    return TargetImplementation::cdist_top_k(
        queries, targets, scorer, k, tqdm_factory, cpu_count,
        reject_duplicates,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  // ----- Matrix-mode entry points (public wrapper) --------------------
  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    ensure_supported();
    return TargetImplementation::smith_waterman_score_matrix(
        query_indices, target_indices, matrix_buffer, stride, gap_score);
  }

  static STRIDE_ALIGN_X86_BASELINE Score needleman_wunsch_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_score_matrix(
        query_indices, target_indices, matrix_buffer, stride, gap_score);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> smith_waterman_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    ensure_supported();
    return TargetImplementation::smith_waterman_scores_matrix(
        query_indices, targets, matrix_buffer, stride, gap_score);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> needleman_wunsch_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_scores_matrix(
        query_indices, targets, matrix_buffer, stride, gap_score);
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_affine_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_score_matrix(
        query_indices, target_indices, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static STRIDE_ALIGN_X86_BASELINE Score needleman_wunsch_affine_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_score_matrix(
        query_indices, target_indices, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> smith_waterman_affine_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_scores_matrix(
        query_indices, targets, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> needleman_wunsch_affine_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_scores_matrix(
        query_indices, targets, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static constexpr BackendKind backend_kind = BackendKind::x86_avx10_512;
};

#undef STRIDE_ALIGN_X86_BASELINE

}  // namespace stride_align::backend_avx10_512
