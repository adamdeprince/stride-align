#pragma once

// Python-facing dispatch for DTW.
//
// Classifies the (query, target) inputs, picks the right Token /
// Cell pair, validates window + distance kwargs, runs the scalar
// reference kernel. Phase C.1b will plug a SIMD batch kernel into
// the same dispatch surface without changing the Python API.

#include <cmath>
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

// SIMD batch kernel + Ops. Backend modules include this header under
// their ISA pragma (or compile flags); CPU dispatch is by which
// extension is imported, not by runtime multi-versioning here.
#include "dtw_simd.hpp"

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

// Parse optional score_cutoff: None → nullopt; float/int → finite ≥ 0.
inline std::optional<double> parse_score_cutoff(nb::object cutoff_obj) {
  if (cutoff_obj.is_none()) {
    return std::nullopt;
  }
  double val = 0.0;
  try {
    val = nb::cast<double>(cutoff_obj);
  } catch (const nb::cast_error&) {
    ::stride_align::detail::throw_type_error(
        "score_cutoff= must be a number or None");
  }
  if (!(val >= 0.0) || !std::isfinite(val)) {
    ::stride_align::detail::throw_value_error(
        "score_cutoff= must be a finite non-negative number");
  }
  return val;
}

}  // namespace dispatch_detail

// Run scalar-reference DTW on a single (query, target) ndarray
// pair. The dispatcher widens int16 inputs to int32 cells; float
// inputs run with matching cell types.
inline double dispatch_dtw(
    nb::handle query,
    nb::handle target,
    nb::object window_obj,
    nb::object distance_obj,
    nb::object score_cutoff_obj = nb::none()) {
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
  const auto cutoff = dispatch_detail::parse_score_cutoff(score_cutoff_obj);

  using D = nv::NdarrayDtype;
  switch (q_view.dtype) {
    case D::Float32: {
      const auto* q_ptr = static_cast<const float*>(q_view.data());
      const auto* t_ptr = static_cast<const float*>(t_view.data());
      return dtw_score_scalar<float, float>(
          std::span<const float>(q_ptr, q_size),
          std::span<const float>(t_ptr, t_size),
          dist, window, cutoff);
    }
    case D::Float64: {
      const auto* q_ptr = static_cast<const double*>(q_view.data());
      const auto* t_ptr = static_cast<const double*>(t_view.data());
      return dtw_score_scalar<double, double>(
          std::span<const double>(q_ptr, q_size),
          std::span<const double>(t_ptr, t_size),
          dist, window, cutoff);
    }
    case D::Int16: {
      const auto* q_ptr = static_cast<const std::int16_t*>(q_view.data());
      const auto* t_ptr = static_cast<const std::int16_t*>(t_view.data());
      return dtw_score_scalar<std::int16_t, std::int32_t>(
          std::span<const std::int16_t>(q_ptr, q_size),
          std::span<const std::int16_t>(t_ptr, t_size),
          dist, window, cutoff);
    }
    default:
      // is_supported_dtw_dtype guard above prevents us getting here.
      ::stride_align::detail::throw_type_error(
          "dtw: internal — unsupported dtype slipped through dispatch");
  }
  return 0.0;  // unreachable
}

namespace dispatch_detail {

// Shared validation: acquire query + every target, compute per-target
// radii, return dtype. Views must stay alive for the caller's lifetime
// (callers store the view objects in a vector).
struct PreparedBatch {
  ::stride_align::numpy_view::View q_view;
  std::vector<::stride_align::numpy_view::View> t_views;
  std::vector<std::size_t> lengths;
  std::vector<std::size_t> radii;
  DistanceKind dist{};
  std::size_t q_size = 0;
  std::optional<double> score_cutoff;
};

inline PreparedBatch prepare_batch(
    nb::handle query,
    nb::handle targets,
    nb::object window_obj,
    nb::object distance_obj,
    nb::object score_cutoff_obj = nb::none()) {
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

  PreparedBatch prep;
  prep.q_view = nv::try_acquire(query.ptr());
  if (prep.q_view.unsupported_buffer || !prep.q_view.acquired) {
    ::stride_align::detail::throw_type_error(
        "dtw_distances query must be an ndarray with float32, float64, or int16 dtype");
  }
  if (!is_supported_dtw_dtype(prep.q_view.dtype)) {
    ::stride_align::detail::throw_type_error(
        "dtw_distances requires float32, float64, or int16 ndarray dtype");
  }
  prep.q_size = prep.q_view.element_count();
  if (prep.q_size == 0) {
    ::stride_align::detail::throw_value_error(
        "dtw_distances query must be non-empty");
  }

  prep.dist = parse_distance(distance_obj, prefer_l1_default(prep.q_view.dtype));
  prep.score_cutoff = parse_score_cutoff(score_cutoff_obj);
  prep.t_views.reserve(target_count);
  prep.lengths.reserve(target_count);
  prep.radii.reserve(target_count);

  for (std::size_t i = 0; i < target_count; ++i) {
    auto t_view = nv::try_acquire(items[i]);
    if (t_view.unsupported_buffer || !t_view.acquired) {
      ::stride_align::detail::throw_type_error(
          "dtw_distances targets must all be ndarrays with the same dtype as query");
    }
    if (t_view.dtype != prep.q_view.dtype) {
      ::stride_align::detail::throw_type_error(
          "dtw_distances: target dtype must match query dtype");
    }
    const std::size_t t_size = t_view.element_count();
    if (t_size == 0) {
      ::stride_align::detail::throw_value_error(
          "dtw_distances: every target must be non-empty");
    }
    const auto window = parse_window(window_obj, prep.q_size, t_size);
    prep.lengths.push_back(t_size);
    prep.radii.push_back(
        resolve_band_radius(prep.q_size, t_size, window));
    prep.t_views.push_back(std::move(t_view));
  }
  (void)owner;
  return prep;
}

// Run scalar DTW with band + optional LB_Keogh / early abandon.
template <typename Token, typename Cell>
inline double score_one_prepared(
    const Token* q_ptr,
    std::size_t q_size,
    const Token* t_ptr,
    std::size_t t_size,
    std::size_t radius,
    DistanceKind dist,
    std::optional<double> score_cutoff) {
  // Pass radius as an explicit absolute window (not nullopt) so the
  // kernel's band matches the batch-prep resolve.
  return dtw_score_scalar<Token, Cell>(
      std::span<const Token>(q_ptr, q_size),
      std::span<const Token>(t_ptr, t_size),
      dist,
      std::optional<std::size_t>(radius),
      score_cutoff);
}

}  // namespace dispatch_detail

inline std::vector<double> dispatch_dtw_many(
    nb::handle query,
    nb::handle targets,
    nb::object window_obj,
    nb::object distance_obj,
    nb::object score_cutoff_obj = nb::none()) {
  auto prep = dispatch_detail::prepare_batch(
      query, targets, window_obj, distance_obj, score_cutoff_obj);
  const std::size_t target_count = prep.t_views.size();
  std::vector<double> results(target_count);

  using D = ::stride_align::numpy_view::NdarrayDtype;
  switch (prep.q_view.dtype) {
    case D::Float32: {
      const auto* q_ptr = static_cast<const float*>(prep.q_view.data());
      for (std::size_t i = 0; i < target_count; ++i) {
        const auto* t_ptr =
            static_cast<const float*>(prep.t_views[i].data());
        results[i] = dispatch_detail::score_one_prepared<float, float>(
            q_ptr, prep.q_size, t_ptr, prep.lengths[i], prep.radii[i],
            prep.dist, prep.score_cutoff);
      }
      break;
    }
    case D::Float64: {
      const auto* q_ptr = static_cast<const double*>(prep.q_view.data());
      for (std::size_t i = 0; i < target_count; ++i) {
        const auto* t_ptr =
            static_cast<const double*>(prep.t_views[i].data());
        results[i] = dispatch_detail::score_one_prepared<double, double>(
            q_ptr, prep.q_size, t_ptr, prep.lengths[i], prep.radii[i],
            prep.dist, prep.score_cutoff);
      }
      break;
    }
    case D::Int16: {
      const auto* q_ptr =
          static_cast<const std::int16_t*>(prep.q_view.data());
      for (std::size_t i = 0; i < target_count; ++i) {
        const auto* t_ptr =
            static_cast<const std::int16_t*>(prep.t_views[i].data());
        results[i] =
            dispatch_detail::score_one_prepared<std::int16_t, std::int32_t>(
                q_ptr, prep.q_size, t_ptr, prep.lengths[i], prep.radii[i],
                prep.dist, prep.score_cutoff);
      }
      break;
    }
    default:
      ::stride_align::detail::throw_type_error(
          "dtw_distances: internal — unsupported dtype slipped through dispatch");
  }
  return results;
}

// SIMD batch with LB_Keogh prefilter when score_cutoff is set.
// Survivors are packed into lane-width waves; rejected targets get +inf.
template <typename Ops, typename Token>
inline void dtw_batch_all_with_cutoff(
    const Token* query,
    std::size_t m,
    const std::vector<const Token*>& targets,
    const std::vector<std::size_t>& lengths,
    const std::vector<std::size_t>& radii,
    DistanceKind dist,
    std::optional<double> score_cutoff,
    std::vector<double>& out) {
  using Cell = typename Ops::Cell;
  const std::size_t n = targets.size();
  out.assign(n, std::numeric_limits<double>::infinity());

  // Indices that still need full DTW after cheap filters.
  std::vector<std::size_t> live;
  live.reserve(n);

  // Pre-build envelope once if any equal-length target might use Keogh.
  std::optional<QueryEnvelope<Token, Cell>> env;
  if (score_cutoff.has_value()) {
    // Use max radius among equal-length targets for envelope; per-target
    // radius is applied inside lb_keogh_with_envelope via env.radius —
    // rebuild per distinct radius only if needed. For simplicity rebuild
    // envelope with each target's radius when lengths match.
  }

  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t tlen = lengths[i];
    const std::size_t r = radii[i];
    // Band impossibility.
    if (m > tlen + r || tlen > m + r) {
      continue;  // already +inf
    }
    if (score_cutoff.has_value() && m == tlen) {
      const double lb = lb_keogh<Token, Cell>(
          std::span<const Token>(query, m),
          std::span<const Token>(targets[i], tlen),
          dist, r, score_cutoff);
      if (lb > *score_cutoff) {
        continue;
      }
    }
    live.push_back(i);
  }

  if (live.empty()) {
    return;
  }

  // Pack survivors for SIMD waves.
  std::vector<const Token*> packed_ptrs(live.size());
  std::vector<std::size_t> packed_lens(live.size());
  std::vector<std::size_t> packed_rads(live.size());
  for (std::size_t k = 0; k < live.size(); ++k) {
    const std::size_t i = live[k];
    packed_ptrs[k] = targets[i];
    packed_lens[k] = lengths[i];
    packed_rads[k] = radii[i];
  }

  std::vector<double> packed_out;
  ::stride_align::dtw_simd::dtw_batch_all<Ops, Token>(
      query, m, packed_ptrs, packed_lens, packed_rads, dist, packed_out);

  for (std::size_t k = 0; k < live.size(); ++k) {
    double v = packed_out[k];
    if (score_cutoff.has_value() && v > *score_cutoff) {
      v = std::numeric_limits<double>::infinity();
    }
    out[live[k]] = v;
  }
}

// SIMD batch entry used by backend Implementation::dtw_distances.
//
// ``DtwOps`` is a DtwOpsBundle (e.g. Avx2DtwOps): one type parameter
// injects the whole ISA. Dtype selects nested Ops::F32 / F64 / I32;
// the shared prepare + chunk + DP sequence stays generic C++.
//
// Same injection style as ``levenshtein_scores_simd<Ops>``.
template <typename DtwOps>
inline std::vector<double> dtw_distances_simd(
    nb::handle query,
    nb::handle targets,
    nb::object window_obj,
    nb::object distance_obj,
    nb::object score_cutoff_obj = nb::none()) {
  auto prep = dispatch_detail::prepare_batch(
      query, targets, window_obj, distance_obj, score_cutoff_obj);
  std::vector<double> results;
  using D = ::stride_align::numpy_view::NdarrayDtype;
  switch (prep.q_view.dtype) {
    case D::Float32: {
      const auto* q_ptr = static_cast<const float*>(prep.q_view.data());
      std::vector<const float*> tptrs(prep.t_views.size());
      for (std::size_t i = 0; i < prep.t_views.size(); ++i) {
        tptrs[i] = static_cast<const float*>(prep.t_views[i].data());
      }
      dtw_batch_all_with_cutoff<typename DtwOps::F32, float>(
          q_ptr, prep.q_size, tptrs, prep.lengths, prep.radii, prep.dist,
          prep.score_cutoff, results);
      break;
    }
    case D::Float64: {
      const auto* q_ptr = static_cast<const double*>(prep.q_view.data());
      std::vector<const double*> tptrs(prep.t_views.size());
      for (std::size_t i = 0; i < prep.t_views.size(); ++i) {
        tptrs[i] = static_cast<const double*>(prep.t_views[i].data());
      }
      dtw_batch_all_with_cutoff<typename DtwOps::F64, double>(
          q_ptr, prep.q_size, tptrs, prep.lengths, prep.radii, prep.dist,
          prep.score_cutoff, results);
      break;
    }
    case D::Int16: {
      const auto* q_ptr =
          static_cast<const std::int16_t*>(prep.q_view.data());
      std::vector<const std::int16_t*> tptrs(prep.t_views.size());
      for (std::size_t i = 0; i < prep.t_views.size(); ++i) {
        tptrs[i] = static_cast<const std::int16_t*>(prep.t_views[i].data());
      }
      dtw_batch_all_with_cutoff<typename DtwOps::I32, std::int16_t>(
          q_ptr, prep.q_size, tptrs, prep.lengths, prep.radii, prep.dist,
          prep.score_cutoff, results);
      break;
    }
    default:
      ::stride_align::detail::throw_type_error(
          "dtw_distances: internal — unsupported dtype slipped through dispatch");
  }
  return results;
}

}  // namespace stride_align::dtw

