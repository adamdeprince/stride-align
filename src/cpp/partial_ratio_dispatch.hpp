#pragma once

// nanobind-facing dispatch for ``sa._partial_ratio_kernel``. Reuses
// the codepoint widening helper from ``lcs_dispatch.hpp`` (the partial
// ratio algorithm operates on the same codepoint vector representation
// LCS does), then runs the C++ matching-block algorithm with the
// Indel kernel.

#include <cstddef>

#include <nanobind/nanobind.h>

#include "lcs_dispatch.hpp"  // widen_to_codepoints
#include "stride_align/partial_ratio.hpp"

namespace stride_align::partial_ratio {

namespace nb = nanobind;

inline double dispatch_partial_ratio(nb::handle a, nb::handle b) {
  return partial_ratio(
      ::stride_align::lcs::widen_to_codepoints(a),
      ::stride_align::lcs::widen_to_codepoints(b));
}

}  // namespace stride_align::partial_ratio
