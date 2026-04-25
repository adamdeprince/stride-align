#pragma once

#include <riscv_vector.h>

#include <sys/auxv.h>
#include <asm/hwcap.h>

#include <cstddef>
#include <cstdint>

#include <nanobind/nanobind.h>

#include "backends/riscv_rvv_kernel.hpp"

namespace stride_align::backend_linux_riscv64_rvv {

namespace nb = nanobind;

template <typename Token, typename Cell>
struct SimdOps;

template <>
struct SimdOps<std::uint8_t, std::int8_t> {
  using vector_type = vint8m1_t;
  static constexpr bool has_vector_max = true;

  static std::size_t lane_count() {
    return __riscv_vsetvlmax_e8m1();
  }

  static vuint8m1_t load_tokens(const std::uint8_t* values, std::size_t count) {
    return __riscv_vle8_v_u8m1(values, count);
  }

  static vector_type load_cells(const std::int8_t* values, std::size_t count) {
    return __riscv_vle8_v_i8m1(values, count);
  }

  static void store_cells(std::int8_t* values, vector_type vector, std::size_t count) {
    __riscv_vse8_v_i8m1(values, vector, count);
  }

  static vector_type set1(std::int8_t value, std::size_t count) {
    return __riscv_vmv_v_x_i8m1(value, count);
  }

  static vector_type zero(std::size_t count) {
    return __riscv_vmv_v_x_i8m1(0, count);
  }

  static vector_type add(vector_type lhs, vector_type rhs, std::size_t count) {
    return __riscv_vadd(lhs, rhs, count);
  }

  static vector_type max(vector_type lhs, vector_type rhs, std::size_t count) {
    return __riscv_vmax(lhs, rhs, count);
  }

  static vector_type substitution(
      const std::uint8_t* query,
      const std::uint8_t* target,
      std::int8_t match_score,
      std::int8_t mismatch_score,
      std::size_t count) {
    const auto mask = __riscv_vmseq(load_tokens(query, count), load_tokens(target, count), count);
    return __riscv_vmerge(set1(mismatch_score, count), set1(match_score, count), mask, count);
  }
};

template <>
struct SimdOps<std::uint16_t, std::int16_t> {
  using vector_type = vint16m1_t;
  static constexpr bool has_vector_max = true;

  static std::size_t lane_count() {
    return __riscv_vsetvlmax_e16m1();
  }

  static vuint16m1_t load_tokens(const std::uint16_t* values, std::size_t count) {
    return __riscv_vle16_v_u16m1(values, count);
  }

  static vector_type load_cells(const std::int16_t* values, std::size_t count) {
    return __riscv_vle16_v_i16m1(values, count);
  }

  static void store_cells(std::int16_t* values, vector_type vector, std::size_t count) {
    __riscv_vse16_v_i16m1(values, vector, count);
  }

  static vector_type set1(std::int16_t value, std::size_t count) {
    return __riscv_vmv_v_x_i16m1(value, count);
  }

  static vector_type zero(std::size_t count) {
    return __riscv_vmv_v_x_i16m1(0, count);
  }

  static vector_type add(vector_type lhs, vector_type rhs, std::size_t count) {
    return __riscv_vadd(lhs, rhs, count);
  }

  static vector_type max(vector_type lhs, vector_type rhs, std::size_t count) {
    return __riscv_vmax(lhs, rhs, count);
  }

  static vector_type substitution(
      const std::uint16_t* query,
      const std::uint16_t* target,
      std::int16_t match_score,
      std::int16_t mismatch_score,
      std::size_t count) {
    const auto mask = __riscv_vmseq(load_tokens(query, count), load_tokens(target, count), count);
    return __riscv_vmerge(set1(mismatch_score, count), set1(match_score, count), mask, count);
  }
};

template <>
struct SimdOps<std::uint32_t, std::int32_t> {
  using vector_type = vint32m1_t;
  static constexpr bool has_vector_max = true;

  static std::size_t lane_count() {
    return __riscv_vsetvlmax_e32m1();
  }

  static vuint32m1_t load_tokens(const std::uint32_t* values, std::size_t count) {
    return __riscv_vle32_v_u32m1(values, count);
  }

  static vector_type load_cells(const std::int32_t* values, std::size_t count) {
    return __riscv_vle32_v_i32m1(values, count);
  }

  static void store_cells(std::int32_t* values, vector_type vector, std::size_t count) {
    __riscv_vse32_v_i32m1(values, vector, count);
  }

  static vector_type set1(std::int32_t value, std::size_t count) {
    return __riscv_vmv_v_x_i32m1(value, count);
  }

  static vector_type zero(std::size_t count) {
    return __riscv_vmv_v_x_i32m1(0, count);
  }

  static vector_type add(vector_type lhs, vector_type rhs, std::size_t count) {
    return __riscv_vadd(lhs, rhs, count);
  }

  static vector_type max(vector_type lhs, vector_type rhs, std::size_t count) {
    return __riscv_vmax(lhs, rhs, count);
  }

  static vector_type substitution(
      const std::uint32_t* query,
      const std::uint32_t* target,
      std::int32_t match_score,
      std::int32_t mismatch_score,
      std::size_t count) {
    const auto mask = __riscv_vmseq(load_tokens(query, count), load_tokens(target, count), count);
    return __riscv_vmerge(set1(mismatch_score, count), set1(match_score, count), mask, count);
  }
};

template <>
struct SimdOps<std::uint64_t, std::int64_t> {
  using vector_type = vint64m1_t;
  static constexpr bool has_vector_max = true;

  static std::size_t lane_count() {
    return __riscv_vsetvlmax_e64m1();
  }

  static vuint64m1_t load_tokens(const std::uint64_t* values, std::size_t count) {
    return __riscv_vle64_v_u64m1(values, count);
  }

  static vector_type load_cells(const std::int64_t* values, std::size_t count) {
    return __riscv_vle64_v_i64m1(values, count);
  }

  static void store_cells(std::int64_t* values, vector_type vector, std::size_t count) {
    __riscv_vse64_v_i64m1(values, vector, count);
  }

  static vector_type set1(std::int64_t value, std::size_t count) {
    return __riscv_vmv_v_x_i64m1(value, count);
  }

  static vector_type zero(std::size_t count) {
    return __riscv_vmv_v_x_i64m1(0, count);
  }

  static vector_type add(vector_type lhs, vector_type rhs, std::size_t count) {
    return __riscv_vadd(lhs, rhs, count);
  }

  static vector_type max(vector_type lhs, vector_type rhs, std::size_t count) {
    return __riscv_vmax(lhs, rhs, count);
  }

  static vector_type substitution(
      const std::uint64_t* query,
      const std::uint64_t* target,
      std::int64_t match_score,
      std::int64_t mismatch_score,
      std::size_t count) {
    const auto mask = __riscv_vmseq(load_tokens(query, count), load_tokens(target, count), count);
    return __riscv_vmerge(set1(mismatch_score, count), set1(match_score, count), mask, count);
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
    return riscv_rvv_kernel::detail::dispatch_score<SimdOps, true>(
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
    return riscv_rvv_kernel::detail::dispatch_traceback<SimdOps, true>(
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
    return riscv_rvv_kernel::detail::dispatch_score<SimdOps, false>(
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
    return riscv_rvv_kernel::detail::dispatch_traceback<SimdOps, false>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }
};

struct Implementation {
  static bool supported_on_this_machine() noexcept {
#ifdef COMPAT_HWCAP_ISA_V
    return (getauxval(AT_HWCAP) & COMPAT_HWCAP_ISA_V) != 0;
#else
    return false;
#endif
  }

  static void ensure_supported() {
    if (supported_on_this_machine()) {
      return;
    }

    PyErr_SetString(PyExc_RuntimeError, "Linux RISC-V RVV backend is not available on this machine");
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
};

}  // namespace stride_align::backend_linux_riscv64_rvv
