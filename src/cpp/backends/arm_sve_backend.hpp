#pragma once

#include <arm_sve.h>

#include <cstddef>
#include <cstdint>

#include <nanobind/nanobind.h>

#include "backends/arm_sve_kernel.hpp"

namespace stride_align::arm_sve_backend {

namespace nb = nanobind;

template <typename Token, typename Cell>
struct SimdOps;

template <>
struct SimdOps<std::uint8_t, std::int8_t> {
  using vector_type = svint8_t;
  static constexpr bool has_vector_max = true;

  static std::size_t lane_count() {
    return svcntb();
  }

  static svbool_t predicate(std::size_t count) {
    return svwhilelt_b8(static_cast<std::uint64_t>(0), static_cast<std::uint64_t>(count));
  }

  static svuint8_t load_tokens(const std::uint8_t* values, std::size_t count) {
    return svld1_u8(predicate(count), values);
  }

  static vector_type load_cells(const std::int8_t* values, std::size_t count) {
    return svld1_s8(predicate(count), values);
  }

  static void store_cells(std::int8_t* values, vector_type vector, std::size_t count) {
    svst1_s8(predicate(count), values, vector);
  }

  static vector_type set1(std::int8_t value, std::size_t) {
    return svdup_n_s8(value);
  }

  static vector_type zero(std::size_t) {
    return svdup_n_s8(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs, std::size_t count) {
    return svadd_s8_x(predicate(count), lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs, std::size_t count) {
    return svmax_s8_x(predicate(count), lhs, rhs);
  }

  static vector_type substitution(
      const std::uint8_t* query,
      const std::uint8_t* target,
      std::int8_t match_score,
      std::int8_t mismatch_score,
      std::size_t count) {
    const svbool_t pg = predicate(count);
    const svbool_t mask = svcmpeq_u8(pg, load_tokens(query, count), load_tokens(target, count));
    return svsel_s8(mask, set1(match_score, count), set1(mismatch_score, count));
  }
};

template <>
struct SimdOps<std::uint16_t, std::int16_t> {
  using vector_type = svint16_t;
  static constexpr bool has_vector_max = true;

  static std::size_t lane_count() {
    return svcnth();
  }

  static svbool_t predicate(std::size_t count) {
    return svwhilelt_b16(static_cast<std::uint64_t>(0), static_cast<std::uint64_t>(count));
  }

  static svuint16_t load_tokens(const std::uint16_t* values, std::size_t count) {
    return svld1_u16(predicate(count), values);
  }

  static vector_type load_cells(const std::int16_t* values, std::size_t count) {
    return svld1_s16(predicate(count), values);
  }

  static void store_cells(std::int16_t* values, vector_type vector, std::size_t count) {
    svst1_s16(predicate(count), values, vector);
  }

  static vector_type set1(std::int16_t value, std::size_t) {
    return svdup_n_s16(value);
  }

  static vector_type zero(std::size_t) {
    return svdup_n_s16(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs, std::size_t count) {
    return svadd_s16_x(predicate(count), lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs, std::size_t count) {
    return svmax_s16_x(predicate(count), lhs, rhs);
  }

  static vector_type substitution(
      const std::uint16_t* query,
      const std::uint16_t* target,
      std::int16_t match_score,
      std::int16_t mismatch_score,
      std::size_t count) {
    const svbool_t pg = predicate(count);
    const svbool_t mask = svcmpeq_u16(pg, load_tokens(query, count), load_tokens(target, count));
    return svsel_s16(mask, set1(match_score, count), set1(mismatch_score, count));
  }
};

template <>
struct SimdOps<std::uint32_t, std::int32_t> {
  using vector_type = svint32_t;
  static constexpr bool has_vector_max = true;

  static std::size_t lane_count() {
    return svcntw();
  }

  static svbool_t predicate(std::size_t count) {
    return svwhilelt_b32(static_cast<std::uint64_t>(0), static_cast<std::uint64_t>(count));
  }

  static svuint32_t load_tokens(const std::uint32_t* values, std::size_t count) {
    return svld1_u32(predicate(count), values);
  }

  static vector_type load_cells(const std::int32_t* values, std::size_t count) {
    return svld1_s32(predicate(count), values);
  }

  static void store_cells(std::int32_t* values, vector_type vector, std::size_t count) {
    svst1_s32(predicate(count), values, vector);
  }

  static vector_type set1(std::int32_t value, std::size_t) {
    return svdup_n_s32(value);
  }

  static vector_type zero(std::size_t) {
    return svdup_n_s32(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs, std::size_t count) {
    return svadd_s32_x(predicate(count), lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs, std::size_t count) {
    return svmax_s32_x(predicate(count), lhs, rhs);
  }

  static vector_type substitution(
      const std::uint32_t* query,
      const std::uint32_t* target,
      std::int32_t match_score,
      std::int32_t mismatch_score,
      std::size_t count) {
    const svbool_t pg = predicate(count);
    const svbool_t mask = svcmpeq_u32(pg, load_tokens(query, count), load_tokens(target, count));
    return svsel_s32(mask, set1(match_score, count), set1(mismatch_score, count));
  }
};

template <>
struct SimdOps<std::uint64_t, std::int64_t> {
  using vector_type = svint64_t;
  static constexpr bool has_vector_max = true;

  static std::size_t lane_count() {
    return svcntd();
  }

  static svbool_t predicate(std::size_t count) {
    return svwhilelt_b64(static_cast<std::uint64_t>(0), static_cast<std::uint64_t>(count));
  }

  static svuint64_t load_tokens(const std::uint64_t* values, std::size_t count) {
    return svld1_u64(predicate(count), values);
  }

  static vector_type load_cells(const std::int64_t* values, std::size_t count) {
    return svld1_s64(predicate(count), values);
  }

  static void store_cells(std::int64_t* values, vector_type vector, std::size_t count) {
    svst1_s64(predicate(count), values, vector);
  }

  static vector_type set1(std::int64_t value, std::size_t) {
    return svdup_n_s64(value);
  }

  static vector_type zero(std::size_t) {
    return svdup_n_s64(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs, std::size_t count) {
    return svadd_s64_x(predicate(count), lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs, std::size_t count) {
    return svmax_s64_x(predicate(count), lhs, rhs);
  }

  static vector_type substitution(
      const std::uint64_t* query,
      const std::uint64_t* target,
      std::int64_t match_score,
      std::int64_t mismatch_score,
      std::size_t count) {
    const svbool_t pg = predicate(count);
    const svbool_t mask = svcmpeq_u64(pg, load_tokens(query, count), load_tokens(target, count));
    return svsel_s64(mask, set1(match_score, count), set1(mismatch_score, count));
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
    return arm_sve_kernel::detail::dispatch_score<SimdOps, true>(
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
    return arm_sve_kernel::detail::dispatch_traceback<SimdOps, true>(
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
    return arm_sve_kernel::detail::dispatch_score<SimdOps, false>(
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
    return arm_sve_kernel::detail::dispatch_traceback<SimdOps, false>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }
};

}  // namespace stride_align::arm_sve_backend
