#pragma once

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

#include <nanobind/nanobind.h>

#include "backends/affine_fixed_kernel.hpp"
#include "backends/farrar_fixed_kernel.hpp"
#include "backends/profile_traceback.hpp"
#include "backends/arm_neon_kernel.hpp"

namespace stride_align::arm_neon128_backend {

namespace nb = nanobind;

inline bool any_mask(uint8x16_t mask) {
  const uint64x2_t words = vreinterpretq_u64_u8(mask);
  return (vgetq_lane_u64(words, 0) | vgetq_lane_u64(words, 1)) != 0;
}

inline bool any_mask(uint16x8_t mask) {
  const uint64x2_t words = vreinterpretq_u64_u16(mask);
  return (vgetq_lane_u64(words, 0) | vgetq_lane_u64(words, 1)) != 0;
}

inline bool any_mask(uint32x4_t mask) {
  const uint64x2_t words = vreinterpretq_u64_u32(mask);
  return (vgetq_lane_u64(words, 0) | vgetq_lane_u64(words, 1)) != 0;
}

inline bool any_mask(uint64x2_t mask) {
  return (vgetq_lane_u64(mask, 0) | vgetq_lane_u64(mask, 1)) != 0;
}

template <typename Token, typename Cell>
struct SimdOps;

template <>
struct SimdOps<std::uint8_t, std::int8_t> {
  using vector_type = int8x16_t;
  using mask_type = uint8x16_t;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 16;
  static constexpr bool has_vector_max = true;

  static uint8x16_t load_tokens(const std::uint8_t* values) {
    return vld1q_u8(values);
  }

  static vector_type load_cells(const std::int8_t* values) {
    return vld1q_s8(values);
  }

  static void store_cells(std::int8_t* values, vector_type vector) {
    vst1q_s8(values, vector);
  }

  static vector_type set1(std::int8_t value) {
    return vdupq_n_s8(value);
  }

  static vector_type zero() {
    return vdupq_n_s8(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return vaddq_s8(lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return vmaxq_s8(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return vreinterpretq_s8_u8(vextq_u8(vdupq_n_u8(0), vreinterpretq_u8_s8(vector), 15));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return any_mask(vcgtq_s8(lhs, rhs));
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return vcgtq_s8(lhs, rhs);
  }

  static mask_type empty_mask() {
    return vdupq_n_u8(0);
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return vorrq_u8(lhs, rhs);
  }

  static bool any_mask(mask_type mask) {
    return arm_neon128_backend::any_mask(mask);
  }

  static vector_type substitution(
      const std::uint8_t* query,
      const std::uint8_t* target,
      std::int8_t match_score,
      std::int8_t mismatch_score) {
    const uint8x16_t mask = vceqq_u8(load_tokens(query), load_tokens(target));
    return vbslq_s8(mask, set1(match_score), set1(mismatch_score));
  }
};

template <>
struct SimdOps<std::uint16_t, std::int16_t> {
  using vector_type = int16x8_t;
  using mask_type = uint16x8_t;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 8;
  static constexpr bool has_vector_max = true;

  static uint16x8_t load_tokens(const std::uint16_t* values) {
    return vld1q_u16(values);
  }

  static vector_type load_cells(const std::int16_t* values) {
    return vld1q_s16(values);
  }

  static void store_cells(std::int16_t* values, vector_type vector) {
    vst1q_s16(values, vector);
  }

  static vector_type set1(std::int16_t value) {
    return vdupq_n_s16(value);
  }

  static vector_type zero() {
    return vdupq_n_s16(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return vaddq_s16(lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return vmaxq_s16(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return vreinterpretq_s16_u8(vextq_u8(vdupq_n_u8(0), vreinterpretq_u8_s16(vector), 14));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return any_mask(vcgtq_s16(lhs, rhs));
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return vcgtq_s16(lhs, rhs);
  }

  static mask_type empty_mask() {
    return vdupq_n_u16(0);
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return vorrq_u16(lhs, rhs);
  }

  static bool any_mask(mask_type mask) {
    return arm_neon128_backend::any_mask(mask);
  }

  static vector_type substitution(
      const std::uint16_t* query,
      const std::uint16_t* target,
      std::int16_t match_score,
      std::int16_t mismatch_score) {
    const uint16x8_t mask = vceqq_u16(load_tokens(query), load_tokens(target));
    return vbslq_s16(mask, set1(match_score), set1(mismatch_score));
  }
};

template <>
struct SimdOps<std::uint32_t, std::int32_t> {
  using vector_type = int32x4_t;
  using mask_type = uint32x4_t;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 4;
  static constexpr bool has_vector_max = true;

  static uint32x4_t load_tokens(const std::uint32_t* values) {
    return vld1q_u32(values);
  }

  static vector_type load_cells(const std::int32_t* values) {
    return vld1q_s32(values);
  }

  static void store_cells(std::int32_t* values, vector_type vector) {
    vst1q_s32(values, vector);
  }

  static vector_type set1(std::int32_t value) {
    return vdupq_n_s32(value);
  }

  static vector_type zero() {
    return vdupq_n_s32(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return vaddq_s32(lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return vmaxq_s32(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return vreinterpretq_s32_u8(vextq_u8(vdupq_n_u8(0), vreinterpretq_u8_s32(vector), 12));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return any_mask(vcgtq_s32(lhs, rhs));
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return vcgtq_s32(lhs, rhs);
  }

  static mask_type empty_mask() {
    return vdupq_n_u32(0);
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return vorrq_u32(lhs, rhs);
  }

  static bool any_mask(mask_type mask) {
    return arm_neon128_backend::any_mask(mask);
  }

  static vector_type substitution(
      const std::uint32_t* query,
      const std::uint32_t* target,
      std::int32_t match_score,
      std::int32_t mismatch_score) {
    const uint32x4_t mask = vceqq_u32(load_tokens(query), load_tokens(target));
    return vbslq_s32(mask, set1(match_score), set1(mismatch_score));
  }
};

template <>
struct SimdOps<std::uint64_t, std::int64_t> {
  using vector_type = int64x2_t;
  using mask_type = uint64x2_t;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 2;
  static constexpr bool has_vector_max = true;

  static uint64x2_t load_tokens(const std::uint64_t* values) {
    return vld1q_u64(values);
  }

  static vector_type load_cells(const std::int64_t* values) {
    return vld1q_s64(values);
  }

  static void store_cells(std::int64_t* values, vector_type vector) {
    vst1q_s64(values, vector);
  }

  static vector_type set1(std::int64_t value) {
    return vdupq_n_s64(value);
  }

  static vector_type zero() {
    return vdupq_n_s64(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return vaddq_s64(lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    const uint64x2_t mask = vcgtq_s64(lhs, rhs);
    return vbslq_s64(mask, lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return vreinterpretq_s64_u8(vextq_u8(vdupq_n_u8(0), vreinterpretq_u8_s64(vector), 8));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return any_mask(vcgtq_s64(lhs, rhs));
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return vcgtq_s64(lhs, rhs);
  }

  static mask_type empty_mask() {
    return vdupq_n_u64(0);
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return vorrq_u64(lhs, rhs);
  }

  static bool any_mask(mask_type mask) {
    return arm_neon128_backend::any_mask(mask);
  }

  static vector_type substitution(
      const std::uint64_t* query,
      const std::uint64_t* target,
      std::int64_t match_score,
      std::int64_t mismatch_score) {
    const uint64x2_t mask = vceqq_u64(load_tokens(query), load_tokens(target));
    return vbslq_s64(mask, set1(match_score), set1(mismatch_score));
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
    if (gap_score > 0) {
      const auto prepared =
          prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
      return arm_neon_kernel::detail::dispatch_score<SimdOps, true>(
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
    return farrar_fixed_kernel::detail::prepare_affine_score<SimdOps>(
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
    if (gap_score > 0) {
      const auto prepared =
          prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
      return arm_neon_kernel::detail::dispatch_score<SimdOps, false>(
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
};

}  // namespace stride_align::arm_neon128_backend
