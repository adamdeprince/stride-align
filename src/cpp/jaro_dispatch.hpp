#pragma once

// Singular and batch dispatch for Jaro and Jaro-Winkler. Same pattern
// as hamming_dispatch.hpp: take the zero-copy bytes/1-byte-unicode
// fast path when both inputs are byte-compatible, otherwise fall
// through to a scalar reference over prepared tokens (which handles
// wider unicode and object sequences). The bit-parallel SIMD batch
// kernels live in jaro_simd.hpp; this header is the entry point that
// decides byte-fast-path vs scalar-fallback for one (query, target)
// pair and for the *_scores batch.

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <variant>
#include <vector>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "preprocess.hpp"
#include "stride_align/jaro.hpp"

namespace stride_align::jaro {

namespace nb = nanobind;

namespace detail {

// One scalar (query, target) dispatch that handles both byte and
// wider-token paths. Returns the (already-normalized) Jaro similarity.
inline double dispatch_jaro_one(nb::handle query, nb::handle target) {
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
    return jaro_scalar<std::uint8_t>(
        std::span<const std::uint8_t>(q_ptr, q_len),
        std::span<const std::uint8_t>(t_ptr, t_len));
  }

  // Wider tokens: borrow the alignment preprocess infrastructure to
  // get a uniform-width token storage on both sides, then dispatch
  // the same scalar template.
  const auto prepared = ::stride_align::prepare_alignment(
      query, target,
      /*match_score=*/1,
      /*mismatch_score=*/-1,
      /*gap_open_score=*/-1,
      /*gap_extend_score=*/-1,
      /*width=*/0U);
  return std::visit(
      [](const auto& q, const auto& t) -> double {
        using QT = typename std::decay_t<decltype(q)>::value_type;
        using TT = typename std::decay_t<decltype(t)>::value_type;
        if constexpr (std::is_same_v<QT, TT>) {
          return jaro_scalar<QT>(
              std::span<const QT>(q.data(), q.size()),
              std::span<const QT>(t.data(), t.size()));
        } else {
          PyErr_SetString(
              PyExc_RuntimeError,
              "jaro: mismatched token widths from preprocess");
          throw nb::python_error();
        }
      },
      prepared.query_tokens,
      prepared.target_tokens);
}

// Compute the common prefix length over arbitrary nb::handles. Bytes /
// 1-byte unicode take the zero-copy path; everything else falls
// through to a per-element rich-compare over the underlying sequence
// (slow but correct, used only when needed). The cap mirrors the
// Jaro-Winkler convention (usually 4).
inline std::size_t common_prefix_dispatch(
    nb::handle query, nb::handle target, std::size_t cap) {
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
    return ::stride_align::jaro::common_prefix<std::uint8_t>(
        std::span<const std::uint8_t>(q_ptr, q_len),
        std::span<const std::uint8_t>(t_ptr, t_len), cap);
  }

  // Same preprocess for token-typed wider inputs.
  const auto prepared = ::stride_align::prepare_alignment(
      query, target, 1, -1, -1, -1, 0U);
  return std::visit(
      [cap](const auto& q, const auto& t) -> std::size_t {
        using QT = typename std::decay_t<decltype(q)>::value_type;
        using TT = typename std::decay_t<decltype(t)>::value_type;
        if constexpr (std::is_same_v<QT, TT>) {
          return ::stride_align::jaro::common_prefix<QT>(
              std::span<const QT>(q.data(), q.size()),
              std::span<const QT>(t.data(), t.size()), cap);
        } else {
          PyErr_SetString(
              PyExc_RuntimeError,
              "jaro: mismatched token widths from preprocess");
          throw nb::python_error();
        }
      },
      prepared.query_tokens,
      prepared.target_tokens);
}

}  // namespace detail

inline double dispatch_similarity(nb::handle query, nb::handle target) {
  return detail::dispatch_jaro_one(query, target);
}

inline double dispatch_winkler_similarity(
    nb::handle query,
    nb::handle target,
    double prefix_weight,
    double prefix_threshold,
    std::size_t prefix_cap) {
  const double jaro = detail::dispatch_jaro_one(query, target);
  if (jaro < prefix_threshold) {
    return jaro;
  }
  const std::size_t L = detail::common_prefix_dispatch(query, target, prefix_cap);
  return jaro + static_cast<double>(L) * prefix_weight * (1.0 - jaro);
}

inline std::vector<double> dispatch_similarities(
    nb::handle query, nb::handle targets) {
  PyObject* fast_targets = PySequence_Fast(
      targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count =
      static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  std::vector<double> out(count);
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = detail::dispatch_jaro_one(query, nb::handle(items[i]));
  }
  return out;
}

inline std::vector<double> dispatch_winkler_similarities(
    nb::handle query,
    nb::handle targets,
    double prefix_weight,
    double prefix_threshold,
    std::size_t prefix_cap) {
  PyObject* fast_targets = PySequence_Fast(
      targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count =
      static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  std::vector<double> out(count);
  for (std::size_t i = 0; i < count; ++i) {
    const nb::handle target(items[i]);
    const double jaro = detail::dispatch_jaro_one(query, target);
    if (jaro < prefix_threshold) {
      out[i] = jaro;
      continue;
    }
    const std::size_t L =
        detail::common_prefix_dispatch(query, target, prefix_cap);
    out[i] = jaro + static_cast<double>(L) * prefix_weight * (1.0 - jaro);
  }
  return out;
}

}  // namespace stride_align::jaro
