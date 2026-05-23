#pragma once

// Per-pair and batch dispatch for true Damerau-Levenshtein (the
// unrestricted form — characters may participate in multiple edits).
// Scalar DP only; no bit-parallel form yet. The binding layer calls
// these instead of routing through `m.attr(...)` for each pair to
// avoid the Python boundary cost.

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <variant>
#include <vector>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "preprocess.hpp"
#include "stride_align/levenshtein.hpp"

namespace stride_align::true_damerau {

namespace nb = nanobind;

inline std::size_t dispatch_distance_raw(
    nb::handle query, nb::handle target) {
  namespace bv = ::stride_align::byte_view;
  const bv::ByteCompatKind q_kind = bv::classify(query.ptr());
  const bv::ByteCompatKind t_kind = bv::classify(target.ptr());
  if (q_kind != bv::ByteCompatKind::None &&
      t_kind != bv::ByteCompatKind::None) {
    const std::uint8_t* q_ptr = nullptr;
    std::size_t q_len = 0;
    const std::uint8_t* t_ptr = nullptr;
    std::size_t t_len = 0;
    bv::view(query.ptr(), q_kind, q_ptr, q_len);
    bv::view(target.ptr(), t_kind, t_ptr, t_len);
    return ::stride_align::levenshtein::
        true_damerau_levenshtein_distance_u8(
            std::span<const std::uint8_t>(q_ptr, q_len),
            std::span<const std::uint8_t>(t_ptr, t_len));
  }

  const auto prepared = ::stride_align::prepare_alignment(
      query, target, /*match=*/1, /*mismatch=*/-1,
      /*gap_open=*/-1, /*gap_extend=*/-1, /*width=*/0U);
  return std::visit(
      [](const auto& q, const auto& t) -> std::size_t {
        using QT = typename std::decay_t<decltype(q)>::value_type;
        using TT = typename std::decay_t<decltype(t)>::value_type;
        if constexpr (std::is_same_v<QT, TT>) {
          return ::stride_align::levenshtein::
              true_damerau_levenshtein_distance<QT>(
                  std::span<const QT>(q.data(), q.size()),
                  std::span<const QT>(t.data(), t.size()));
        } else {
          PyErr_SetString(
              PyExc_RuntimeError,
              "true_damerau_levenshtein: mismatched token widths");
          throw nb::python_error();
        }
      },
      prepared.query_tokens,
      prepared.target_tokens);
}

inline Score dispatch_score(nb::handle query, nb::handle target) {
  return static_cast<Score>(dispatch_distance_raw(query, target));
}

inline double dispatch_normalized_score(
    nb::handle query, nb::handle target) {
  const std::size_t d = dispatch_distance_raw(query, target);
  const std::size_t q_len =
      static_cast<std::size_t>(PyObject_Length(query.ptr()));
  const std::size_t t_len =
      static_cast<std::size_t>(PyObject_Length(target.ptr()));
  return ::stride_align::levenshtein::normalize(d, q_len, t_len);
}

inline std::vector<Score> dispatch_scores(
    nb::handle query, nb::handle targets) {
  PyObject* fast_targets = PySequence_Fast(
      targets.ptr(),
      "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count = static_cast<std::size_t>(
      PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
  std::vector<Score> out(count);
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = dispatch_score(query, nb::handle(items[i]));
  }
  return out;
}

inline std::vector<double> dispatch_normalized_scores(
    nb::handle query, nb::handle targets) {
  PyObject* fast_targets = PySequence_Fast(
      targets.ptr(),
      "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count = static_cast<std::size_t>(
      PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
  std::vector<double> out(count);
  const std::size_t q_len =
      static_cast<std::size_t>(PyObject_Length(query.ptr()));
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t d = dispatch_distance_raw(
        query, nb::handle(items[i]));
    const std::size_t t_len = static_cast<std::size_t>(
        PyObject_Length(items[i]));
    out[i] = ::stride_align::levenshtein::normalize(d, q_len, t_len);
  }
  return out;
}

}  // namespace stride_align::true_damerau
