#pragma once

#include <sys/auxv.h>
#include <asm/hwcap.h>

#include <nanobind/nanobind.h>

#include "backends/arm_sve_backend.hpp"

namespace stride_align::backend_linux_aarch64_sve {

namespace nb = nanobind;

using TargetImplementation = stride_align::arm_sve_backend::TargetImplementation;

struct Implementation {
  static bool supported_on_this_machine() noexcept {
    return (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
  }

  static void ensure_supported() {
    if (supported_on_this_machine()) {
      return;
    }

    PyErr_SetString(PyExc_RuntimeError, "Linux AArch64 SVE backend is not available on this machine");
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

}  // namespace stride_align::backend_linux_aarch64_sve
