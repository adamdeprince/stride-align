#pragma once

// nanobind-facing dispatch for ``sa.ratcliff_obershelp_similarity``.
// Reuses the codepoint widening helper from ``lcs_dispatch.hpp`` since
// Ratcliff-Obershelp drives the same range-based LCSubstr DP.

#include <cstddef>
#include <vector>

#include <nanobind/nanobind.h>

#include "lcs_dispatch.hpp"
#include "stride_align/ratcliff_obershelp.hpp"

namespace stride_align::ratcliff_obershelp {

namespace nb = nanobind;

inline double dispatch_ratcliff_obershelp_similarity(nb::handle a,
                                                       nb::handle b) {
  return ratcliff_obershelp_similarity(
      ::stride_align::lcs::widen_to_codepoints(a),
      ::stride_align::lcs::widen_to_codepoints(b));
}

// Batch form: one query against ``len(targets)`` targets. The query is
// widened once and reused for every target — same pattern as the
// existing ``*_similarities`` entry points.
inline std::vector<double> dispatch_ratcliff_obershelp_similarities(
    nb::handle query, nb::handle targets) {
  std::vector<double> out;
  if (!PySequence_Check(targets.ptr())) {
    ::stride_align::detail::throw_type_error(
        "targets must be an iterable of strings");
    return out;
  }
  PyObject* fast = PySequence_Fast(
      targets.ptr(), "targets must be an iterable of strings");
  if (fast == nullptr) throw nb::python_error();
  nb::object owner = nb::steal<nb::object>(fast);
  const auto n = static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast));
  PyObject** items = PySequence_Fast_ITEMS(fast);
  const auto query_cps = ::stride_align::lcs::widen_to_codepoints(query);
  out.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto target_cps =
        ::stride_align::lcs::widen_to_codepoints(nb::handle(items[i]));
    out[i] = ratcliff_obershelp_similarity(query_cps, target_cps);
  }
  return out;
}

}  // namespace stride_align::ratcliff_obershelp
