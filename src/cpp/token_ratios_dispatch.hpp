#pragma once

// nanobind-facing dispatch for ``sa._token_sort_ratio_kernel`` and
// ``sa._token_set_ratio_kernel``. Widens the inputs to codepoint
// vectors via the shared LCS helper, then runs the C++ token-ratio
// engines (which themselves drop to the byte fast path when both
// inputs fit in [0, 256)).

#include <nanobind/nanobind.h>

#include "lcs_dispatch.hpp"  // widen_to_codepoints
#include "stride_align/token_ratios.hpp"

namespace stride_align::token_ratios {

namespace nb = nanobind;

inline double dispatch_token_sort_ratio(nb::handle a, nb::handle b) {
  return token_sort_ratio(
      ::stride_align::lcs::widen_to_codepoints(a),
      ::stride_align::lcs::widen_to_codepoints(b));
}

inline double dispatch_token_set_ratio(nb::handle a, nb::handle b) {
  return token_set_ratio(
      ::stride_align::lcs::widen_to_codepoints(a),
      ::stride_align::lcs::widen_to_codepoints(b));
}

}  // namespace stride_align::token_ratios
