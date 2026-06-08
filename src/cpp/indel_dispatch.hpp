#pragma once

// Singular and batch dispatch for Indel distance. Same shape as
// hamming_dispatch.hpp / jaro_dispatch.hpp: take the zero-copy bytes /
// 1-byte-unicode fast path when both inputs are byte-compatible,
// otherwise fall through to a scalar reference over prepared tokens
// (handles wider unicode and object sequences).

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <variant>
#include <vector>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "preprocess.hpp"
#include "stride_align/indel.hpp"

namespace stride_align::indel {

namespace nb = nanobind;

namespace detail {

inline std::size_t dispatch_indel_one(
    nb::handle query, nb::handle target,
    std::size_t cutoff = ::stride_align::indel::kNoCutoff) {
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
    return ::stride_align::indel::indel_distance_u8(
        std::span<const std::uint8_t>(q_ptr, q_len),
        std::span<const std::uint8_t>(t_ptr, t_len),
        cutoff);
  }

  const auto prepared = ::stride_align::prepare_alignment(
      query, target,
      /*match_score=*/1,
      /*mismatch_score=*/-1,
      /*gap_open_score=*/-1,
      /*gap_extend_score=*/-1,
      /*width=*/0U);
  return std::visit(
      [cutoff](const auto& q, const auto& t) -> std::size_t {
        using QT = typename std::decay_t<decltype(q)>::value_type;
        using TT = typename std::decay_t<decltype(t)>::value_type;
        if constexpr (std::is_same_v<QT, TT>) {
          return ::stride_align::indel::indel_distance<QT>(
              std::span<const QT>(q.data(), q.size()),
              std::span<const QT>(t.data(), t.size()),
              cutoff);
        } else {
          PyErr_SetString(
              PyExc_RuntimeError,
              "indel: mismatched token widths from preprocess");
          throw nb::python_error();
        }
      },
      prepared.query_tokens,
      prepared.target_tokens);
}

}  // namespace detail

inline Score dispatch_score(nb::handle query, nb::handle target) {
  return static_cast<Score>(
      detail::dispatch_indel_one(query, target));
}

inline double dispatch_normalized_score(
    nb::handle query, nb::handle target,
    double score_cutoff = 0.0) {
  // rapidfuzz semantics: ``score_cutoff`` is the minimum normalized
  // similarity the caller cares about; values below that get returned
  // as 0.0. Translate to a max-distance cutoff via the closed-form
  //   distance = (|q| + |t|) * (1 - sim) => sim < cutoff iff
  //   distance > (|q| + |t|) * (1 - cutoff).
  const std::size_t q_len =
      static_cast<std::size_t>(PyObject_Length(query.ptr()));
  const std::size_t t_len =
      static_cast<std::size_t>(PyObject_Length(target.ptr()));
  std::size_t distance_cutoff = ::stride_align::indel::kNoCutoff;
  if (score_cutoff > 0.0 && score_cutoff <= 1.0) {
    const double allowed = static_cast<double>(q_len + t_len) *
                           (1.0 - score_cutoff);
    distance_cutoff = static_cast<std::size_t>(allowed);
  }
  const std::size_t d = detail::dispatch_indel_one(
      query, target, distance_cutoff);
  const double sim = normalize(d, q_len, t_len);
  return (score_cutoff > 0.0 && sim < score_cutoff) ? 0.0 : sim;
}

inline std::vector<Score> dispatch_scores(
    nb::handle query,
    nb::handle targets) {
  PyObject* fast_targets =
      PySequence_Fast(targets.ptr(),
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
    nb::handle query,
    nb::handle targets) {
  auto raw = dispatch_scores(query, targets);
  const std::size_t q_len =
      static_cast<std::size_t>(PyObject_Length(query.ptr()));
  PyObject* fast_targets = PySequence_Fast(
      targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
  std::vector<double> out;
  out.reserve(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const std::size_t t_len =
        static_cast<std::size_t>(PyObject_Length(items[i]));
    out.push_back(normalize(
        static_cast<std::size_t>(raw[i]), q_len, t_len));
  }
  return out;
}

}  // namespace stride_align::indel
