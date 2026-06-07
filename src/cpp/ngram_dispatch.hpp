#pragma once

// nanobind-facing dispatch for the n-gram similarity family
// (``sa.jaccard``, ``sa.dice``, ``sa.cosine``, ``sa.overlap`` and
// their batch ``*_similarities`` forms). All four metrics share the
// codepoint widening helper from ``lcs_dispatch.hpp`` and the
// multiset-stats one-pass comparison from ``stride_align::ngram``.
//
// Batch dispatch builds the query multiset once and reuses it across
// every target — same convention as the existing
// ``ratcliff_obershelp_similarities`` and ``jaro_similarities``
// entry points.

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>

#include "lcs_dispatch.hpp"
#include "stride_align/ngram.hpp"

namespace stride_align::ngram {

namespace nb = nanobind;

inline void check_n(std::size_t n) {
  if (n == 0) {
    ::stride_align::detail::throw_value_error(
        "n must be >= 1 for n-gram similarity");
  }
}

inline double dispatch_jaccard(nb::handle a, nb::handle b, std::size_t n) {
  check_n(n);
  return jaccard(::stride_align::lcs::widen_to_codepoints(a),
                 ::stride_align::lcs::widen_to_codepoints(b), n);
}

inline double dispatch_dice(nb::handle a, nb::handle b, std::size_t n) {
  check_n(n);
  return dice(::stride_align::lcs::widen_to_codepoints(a),
              ::stride_align::lcs::widen_to_codepoints(b), n);
}

inline double dispatch_cosine(nb::handle a, nb::handle b, std::size_t n) {
  check_n(n);
  return cosine(::stride_align::lcs::widen_to_codepoints(a),
                ::stride_align::lcs::widen_to_codepoints(b), n);
}

inline double dispatch_overlap(nb::handle a, nb::handle b, std::size_t n) {
  check_n(n);
  return overlap(::stride_align::lcs::widen_to_codepoints(a),
                 ::stride_align::lcs::widen_to_codepoints(b), n);
}

// Batch dispatch: build the query multiset once, walk the targets,
// recompute stats per pair, hand back ``len(targets)`` similarities
// as ``ndarray[float64]``. The four batch entry points differ only in
// which ``*_from_stats`` formula they call.
template <typename FromStats>
inline std::vector<double> dispatch_batch(nb::handle query, nb::handle targets,
                                            std::size_t n,
                                            FromStats from_stats) {
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
  const auto count = static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast));
  PyObject** items = PySequence_Fast_ITEMS(fast);
  const auto query_cps = ::stride_align::lcs::widen_to_codepoints(query);
  const auto query_ms = build_multiset(query_cps, n);
  out.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto target_cps =
        ::stride_align::lcs::widen_to_codepoints(nb::handle(items[i]));
    const auto target_ms = build_multiset(target_cps, n);
    const auto stats = compare(query_ms, target_ms);
    out[i] = from_stats(stats);
  }
  return out;
}

inline std::vector<double> dispatch_jaccard_similarities(
    nb::handle query, nb::handle targets, std::size_t n) {
  check_n(n);
  return dispatch_batch(query, targets, n, jaccard_from_stats);
}

inline std::vector<double> dispatch_dice_similarities(
    nb::handle query, nb::handle targets, std::size_t n) {
  check_n(n);
  return dispatch_batch(query, targets, n, dice_from_stats);
}

inline std::vector<double> dispatch_cosine_similarities(
    nb::handle query, nb::handle targets, std::size_t n) {
  check_n(n);
  return dispatch_batch(query, targets, n, cosine_from_stats);
}

inline std::vector<double> dispatch_overlap_similarities(
    nb::handle query, nb::handle targets, std::size_t n) {
  check_n(n);
  return dispatch_batch(query, targets, n, overlap_from_stats);
}

}  // namespace stride_align::ngram
