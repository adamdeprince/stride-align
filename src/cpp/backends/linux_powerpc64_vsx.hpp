#pragma once

#include <altivec.h>
#undef bool
#undef pixel
#undef vector

#include <sys/auxv.h>
#include <asm/cputable.h>

#include <cstddef>
#include <cstdint>

#include <nanobind/nanobind.h>

#include "backends/affine_fixed_kernel.hpp"
#include "backends/farrar_fixed_kernel.hpp"
#include "backends/powerpc_vsx_kernel.hpp"
#include "backends/profile_traceback.hpp"

namespace stride_align::backend_linux_powerpc64_vsx {

namespace nb = nanobind;

using ByteMask = __vector unsigned char;

template <typename Token, typename Cell>
struct SimdOps;

template <>
struct SimdOps<std::uint8_t, std::int8_t> {
  using token_vector_type = __vector unsigned char;
  using vector_type = __vector signed char;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 16;
  static constexpr bool has_vector_max = true;

  static token_vector_type load_tokens(const std::uint8_t* values) {
    return vec_vsx_ld(0, reinterpret_cast<const token_vector_type*>(values));
  }

  static vector_type load_cells(const std::int8_t* values) {
    return vec_vsx_ld(0, reinterpret_cast<const vector_type*>(values));
  }

  static void store_cells(std::int8_t* values, vector_type vector) {
    vec_vsx_st(vector, 0, values);
  }

  static vector_type set1(std::int8_t value) {
    return vec_splats(value);
  }

  static vector_type zero() {
    return vec_splats(static_cast<std::int8_t>(0));
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return vec_add(lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return vec_max(lhs, rhs);
  }

  static vector_type substitution(
      const std::uint8_t* query,
      const std::uint8_t* target,
      std::int8_t match_score,
      std::int8_t mismatch_score) {
    const ByteMask mask = (ByteMask)vec_cmpeq(load_tokens(query), load_tokens(target));
    return vec_sel(set1(mismatch_score), set1(match_score), mask);
  }
};

template <>
struct SimdOps<std::uint16_t, std::int16_t> {
  using token_vector_type = __vector unsigned short;
  using vector_type = __vector signed short;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 8;
  static constexpr bool has_vector_max = true;

  static token_vector_type load_tokens(const std::uint16_t* values) {
    return vec_vsx_ld(0, reinterpret_cast<const token_vector_type*>(values));
  }

  static vector_type load_cells(const std::int16_t* values) {
    return vec_vsx_ld(0, reinterpret_cast<const vector_type*>(values));
  }

  static void store_cells(std::int16_t* values, vector_type vector) {
    vec_vsx_st(vector, 0, values);
  }

  static vector_type set1(std::int16_t value) {
    return vec_splats(value);
  }

  static vector_type zero() {
    return vec_splats(static_cast<std::int16_t>(0));
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return vec_add(lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return vec_max(lhs, rhs);
  }

  static vector_type substitution(
      const std::uint16_t* query,
      const std::uint16_t* target,
      std::int16_t match_score,
      std::int16_t mismatch_score) {
    const ByteMask mask = (ByteMask)vec_cmpeq(load_tokens(query), load_tokens(target));
    return vec_sel(set1(mismatch_score), set1(match_score), mask);
  }
};

template <>
struct SimdOps<std::uint32_t, std::int32_t> {
  using token_vector_type = __vector unsigned int;
  using vector_type = __vector signed int;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 4;
  static constexpr bool has_vector_max = true;

  static token_vector_type load_tokens(const std::uint32_t* values) {
    return vec_vsx_ld(0, reinterpret_cast<const token_vector_type*>(values));
  }

  static vector_type load_cells(const std::int32_t* values) {
    return vec_vsx_ld(0, reinterpret_cast<const vector_type*>(values));
  }

  static void store_cells(std::int32_t* values, vector_type vector) {
    vec_vsx_st(vector, 0, values);
  }

  static vector_type set1(std::int32_t value) {
    return vec_splats(value);
  }

  static vector_type zero() {
    return vec_splats(static_cast<std::int32_t>(0));
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return vec_add(lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return vec_max(lhs, rhs);
  }

  static vector_type substitution(
      const std::uint32_t* query,
      const std::uint32_t* target,
      std::int32_t match_score,
      std::int32_t mismatch_score) {
    const ByteMask mask = (ByteMask)vec_cmpeq(load_tokens(query), load_tokens(target));
    return vec_sel(set1(mismatch_score), set1(match_score), mask);
  }
};

template <>
struct SimdOps<std::uint64_t, std::int64_t> {
  using token_vector_type = __vector unsigned long long;
  using vector_type = __vector signed long long;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 2;
  static constexpr bool has_vector_max = false;

  static token_vector_type load_tokens(const std::uint64_t* values) {
    return vec_vsx_ld(0, reinterpret_cast<const token_vector_type*>(values));
  }

  static vector_type load_cells(const std::int64_t* values) {
    return vec_vsx_ld(0, reinterpret_cast<const vector_type*>(values));
  }

  static void store_cells(std::int64_t* values, vector_type vector) {
    vec_vsx_st(vector, 0, values);
  }

  static vector_type set1(std::int64_t value) {
    return vec_splats(value);
  }

  static vector_type zero() {
    return set1(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return vec_add(lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    alignas(alignment) std::int64_t left[lane_count] = {};
    alignas(alignment) std::int64_t right[lane_count] = {};
    alignas(alignment) std::int64_t output[lane_count] = {};
    store_cells(left, lhs);
    store_cells(right, rhs);
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
      output[lane] = left[lane] > right[lane] ? left[lane] : right[lane];
    }
    return load_cells(output);
  }

  static vector_type substitution(
      const std::uint64_t* query,
      const std::uint64_t* target,
      std::int64_t match_score,
      std::int64_t mismatch_score) {
    const ByteMask mask = (ByteMask)vec_cmpeq(load_tokens(query), load_tokens(target));
    return vec_sel(set1(mismatch_score), set1(match_score), mask);
  }
};

struct TargetImplementation {
  using PreparedSmithWatermanFarrarScore =
      farrar_fixed_kernel::detail::PreparedScore<SimdOps>;

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
      return powerpc_vsx_kernel::detail::dispatch_score<SimdOps, true>(
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
    ensure_supported();
    return TargetImplementation::smith_waterman_path_info(
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
      return powerpc_vsx_kernel::detail::dispatch_score<SimdOps, false>(
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
    ensure_supported();
    return TargetImplementation::needleman_wunsch_path_info(
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
    return affine::needleman_wunsch_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
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

  static AlignmentPath needleman_wunsch_affine_path_info(
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
};

struct Implementation {
  using PreparedSmithWatermanFarrarScore =
      TargetImplementation::PreparedSmithWatermanFarrarScore;

  static bool supported_on_this_machine() noexcept {
    return (getauxval(AT_HWCAP) & PPC_FEATURE_HAS_VSX) != 0;
  }

  static void ensure_supported() {
    if (supported_on_this_machine()) {
      return;
    }

    PyErr_SetString(PyExc_RuntimeError, "Linux PowerPC64 VSX backend is not available on this machine");
    throw nb::python_error();
  }

  static Score smith_waterman_score(
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

  static AlignmentResult smith_waterman_path(
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

  static Score smith_waterman_farrar_score(
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

  static PreparedSmithWatermanFarrarScore prepare_smith_waterman_farrar_score(
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

  static Score smith_waterman_farrar_score_prepared(
      PreparedSmithWatermanFarrarScore& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_farrar_score_prepared(prepared);
  }

  static Score smith_waterman_affine_score(
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

  static AlignmentResult smith_waterman_affine_path(
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

  static Score smith_waterman_affine_farrar_score(
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

  static Score needleman_wunsch_score(
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

  static AlignmentResult needleman_wunsch_path(
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

  static Score needleman_wunsch_affine_score(
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

  static AlignmentResult needleman_wunsch_affine_path(
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
};

}  // namespace stride_align::backend_linux_powerpc64_vsx
