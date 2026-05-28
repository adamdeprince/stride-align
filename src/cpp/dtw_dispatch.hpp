#pragma once

// Python-facing dispatch for DTW.
//
// Classifies the (query, target) inputs, picks the right Token /
// Cell pair, validates window + distance kwargs, runs the scalar
// reference kernel. Phase C.1b will plug a SIMD batch kernel into
// the same dispatch surface without changing the Python API.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <nanobind/nanobind.h>

#include "numpy_view.hpp"
#include "preprocess.hpp"  // detail::throw_type_error / throw_value_error
#include "stride_align/alignment.hpp"
#include "stride_align/dtw.hpp"

namespace stride_align::dtw {

namespace nb = nanobind;

namespace dispatch_detail {

// Parse the Python-side `distance` kwarg. None -> per-dtype default
// (caller picks); a string -> kL1 or kL2Squared.
inline DistanceKind parse_distance(
    nb::object distance_obj,
    bool prefer_l1_default) {
  if (distance_obj.is_none()) {
    return prefer_l1_default ? DistanceKind::kL1 : DistanceKind::kL2Squared;
  }
  std::string s;
  try {
    s = nb::cast<std::string>(distance_obj);
  } catch (const nb::cast_error&) {
    ::stride_align::detail::throw_type_error(
        "distance= must be \"l1\", \"l2_squared\", or None");
  }
  if (s == "l1") {
    return DistanceKind::kL1;
  }
  if (s == "l2_squared") {
    return DistanceKind::kL2Squared;
  }
  ::stride_align::detail::throw_value_error(
      "distance= must be \"l1\" or \"l2_squared\"");
  // unreachable
  return DistanceKind::kL1;
}

// Parse the Python-side `window` kwarg. None -> unconstrained;
// int -> absolute Sakoe-Chiba radius in samples; float in (0, 1] ->
// fraction of max(|q|, |t|).
inline std::optional<std::size_t> parse_window(
    nb::object window_obj,
    std::size_t q_size,
    std::size_t t_size) {
  if (window_obj.is_none()) {
    return std::nullopt;
  }
  // Try int first; if that fails, try float.
  if (PyLong_Check(window_obj.ptr())) {
    const long long val = nb::cast<long long>(window_obj);
    if (val < 0) {
      ::stride_align::detail::throw_value_error(
          "window= must be non-negative");
    }
    return static_cast<std::size_t>(val);
  }
  if (PyFloat_Check(window_obj.ptr())) {
    const double val = nb::cast<double>(window_obj);
    if (val <= 0.0 || val > 1.0) {
      ::stride_align::detail::throw_value_error(
          "window= as float must be in (0, 1]");
    }
    const std::size_t span = std::max(q_size, t_size);
    // Round up so window=1.0 includes everything.
    const auto radius = static_cast<std::size_t>(
        std::ceil(val * static_cast<double>(span)));
    return radius;
  }
  ::stride_align::detail::throw_type_error(
      "window= must be None, int (samples), or float in (0, 1] (fraction)");
  return std::nullopt;
}

// Classify a single ndarray input for DTW: returns true on
// supported dtype, false otherwise. Caller raises if false.
inline bool is_supported_dtw_dtype(
    ::stride_align::numpy_view::NdarrayDtype dtype) noexcept {
  using D = ::stride_align::numpy_view::NdarrayDtype;
  return dtype == D::Float32 || dtype == D::Float64 || dtype == D::Int16;
}

inline bool prefer_l1_default(
    ::stride_align::numpy_view::NdarrayDtype dtype) noexcept {
  return dtype == ::stride_align::numpy_view::NdarrayDtype::Int16;
}

}  // namespace dispatch_detail

// Run scalar-reference DTW on a single (query, target) ndarray
// pair. The dispatcher widens int16 inputs to int32 cells; float
// inputs run with matching cell types.
inline double dispatch_dtw(
    nb::handle query,
    nb::handle target,
    nb::object window_obj,
    nb::object distance_obj) {
  namespace nv = ::stride_align::numpy_view;
  auto q_view = nv::try_acquire(query.ptr());
  auto t_view = nv::try_acquire(target.ptr());
  if (q_view.unsupported_buffer || t_view.unsupported_buffer) {
    ::stride_align::detail::throw_type_error(
        "dtw inputs must be ndarrays with float32, float64, or int16 dtype");
  }
  if (!q_view.acquired || !t_view.acquired) {
    ::stride_align::detail::throw_type_error(
        "dtw requires ndarray inputs (sequences / bytes / str not supported)");
  }
  if (q_view.dtype != t_view.dtype) {
    ::stride_align::detail::throw_type_error(
        "dtw query and target must share dtype");
  }
  if (!dispatch_detail::is_supported_dtw_dtype(q_view.dtype)) {
    ::stride_align::detail::throw_type_error(
        "dtw requires float32, float64, or int16 ndarray dtype");
  }
  const std::size_t q_size = q_view.element_count();
  const std::size_t t_size = t_view.element_count();
  if (q_size == 0 || t_size == 0) {
    ::stride_align::detail::throw_value_error(
        "dtw inputs must be non-empty");
  }

  const auto window = dispatch_detail::parse_window(window_obj, q_size, t_size);
  const auto dist = dispatch_detail::parse_distance(
      distance_obj, dispatch_detail::prefer_l1_default(q_view.dtype));

  using D = nv::NdarrayDtype;
  switch (q_view.dtype) {
    case D::Float32: {
      const auto* q_ptr = static_cast<const float*>(q_view.data());
      const auto* t_ptr = static_cast<const float*>(t_view.data());
      return dtw_score_scalar<float, float>(
          std::span<const float>(q_ptr, q_size),
          std::span<const float>(t_ptr, t_size),
          dist, window);
    }
    case D::Float64: {
      const auto* q_ptr = static_cast<const double*>(q_view.data());
      const auto* t_ptr = static_cast<const double*>(t_view.data());
      return dtw_score_scalar<double, double>(
          std::span<const double>(q_ptr, q_size),
          std::span<const double>(t_ptr, t_size),
          dist, window);
    }
    case D::Int16: {
      const auto* q_ptr = static_cast<const std::int16_t*>(q_view.data());
      const auto* t_ptr = static_cast<const std::int16_t*>(t_view.data());
      return dtw_score_scalar<std::int16_t, std::int32_t>(
          std::span<const std::int16_t>(q_ptr, q_size),
          std::span<const std::int16_t>(t_ptr, t_size),
          dist, window);
    }
    default:
      // is_supported_dtw_dtype guard above prevents us getting here.
      ::stride_align::detail::throw_type_error(
          "dtw: internal — unsupported dtype slipped through dispatch");
  }
  return 0.0;  // unreachable
}

inline std::vector<double> dispatch_dtw_many(
    nb::handle query,
    nb::handle targets,
    nb::object window_obj,
    nb::object distance_obj) {
  namespace nv = ::stride_align::numpy_view;
  PyObject* fast = PySequence_Fast(
      targets.ptr(), "targets must be a sequence of target ndarrays");
  if (fast == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast);
  const auto target_count =
      static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast));
  PyObject* const* items = PySequence_Fast_ITEMS(fast);

  auto q_view = nv::try_acquire(query.ptr());
  if (q_view.unsupported_buffer || !q_view.acquired) {
    ::stride_align::detail::throw_type_error(
        "dtw_distances query must be an ndarray with float32, float64, or int16 dtype");
  }
  if (!dispatch_detail::is_supported_dtw_dtype(q_view.dtype)) {
    ::stride_align::detail::throw_type_error(
        "dtw_distances requires float32, float64, or int16 ndarray dtype");
  }
  const std::size_t q_size = q_view.element_count();
  if (q_size == 0) {
    ::stride_align::detail::throw_value_error(
        "dtw_distances query must be non-empty");
  }

  const bool prefer_l1 = dispatch_detail::prefer_l1_default(q_view.dtype);
  const auto dist = dispatch_detail::parse_distance(distance_obj, prefer_l1);

  std::vector<double> results;
  results.reserve(target_count);
  for (std::size_t i = 0; i < target_count; ++i) {
    auto t_view = nv::try_acquire(items[i]);
    if (t_view.unsupported_buffer || !t_view.acquired) {
      ::stride_align::detail::throw_type_error(
          "dtw_distances targets must all be ndarrays with the same dtype as query");
    }
    if (t_view.dtype != q_view.dtype) {
      ::stride_align::detail::throw_type_error(
          "dtw_distances: target dtype must match query dtype");
    }
    const std::size_t t_size = t_view.element_count();
    if (t_size == 0) {
      ::stride_align::detail::throw_value_error(
          "dtw_distances: every target must be non-empty");
    }
    const auto window = dispatch_detail::parse_window(window_obj, q_size, t_size);

    using D = nv::NdarrayDtype;
    switch (q_view.dtype) {
      case D::Float32: {
        const auto* q_ptr = static_cast<const float*>(q_view.data());
        const auto* t_ptr = static_cast<const float*>(t_view.data());
        results.push_back(dtw_score_scalar<float, float>(
            std::span<const float>(q_ptr, q_size),
            std::span<const float>(t_ptr, t_size),
            dist, window));
        break;
      }
      case D::Float64: {
        const auto* q_ptr = static_cast<const double*>(q_view.data());
        const auto* t_ptr = static_cast<const double*>(t_view.data());
        results.push_back(dtw_score_scalar<double, double>(
            std::span<const double>(q_ptr, q_size),
            std::span<const double>(t_ptr, t_size),
            dist, window));
        break;
      }
      case D::Int16: {
        const auto* q_ptr = static_cast<const std::int16_t*>(q_view.data());
        const auto* t_ptr = static_cast<const std::int16_t*>(t_view.data());
        results.push_back(dtw_score_scalar<std::int16_t, std::int32_t>(
            std::span<const std::int16_t>(q_ptr, q_size),
            std::span<const std::int16_t>(t_ptr, t_size),
            dist, window));
        break;
      }
      default:
        ::stride_align::detail::throw_type_error(
            "dtw_distances: internal — unsupported dtype slipped through dispatch");
    }
  }
  return results;
}

}  // namespace stride_align::dtw
