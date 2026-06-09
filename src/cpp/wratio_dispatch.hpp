#pragma once

// nanobind-facing dispatch for ``sa._wratio_kernel``. Widens the
// inputs to codepoint vectors, then runs the C++ WRatio engine.

#include <nanobind/nanobind.h>

#include "lcs_dispatch.hpp"  // widen_to_codepoints
#include "stride_align/wratio.hpp"

namespace stride_align::wratio {

namespace nb = nanobind;

inline double dispatch_wratio(nb::handle a, nb::handle b, double score_cutoff) {
  return wratio(
      ::stride_align::lcs::widen_to_codepoints(a),
      ::stride_align::lcs::widen_to_codepoints(b),
      score_cutoff);
}

}  // namespace stride_align::wratio
