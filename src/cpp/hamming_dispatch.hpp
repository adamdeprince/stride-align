#pragma once

// Singular Hamming dispatch with the zero-copy bytes / 1-byte-unicode
// fast path. Wider unicode and sequence-of-object inputs fall through
// to the scalar reference in stride_align/hamming.hpp.

#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "hamming_simd.hpp"
#include "preprocess.hpp"
#include "stride_align/alignment.hpp"
#include "stride_align/hamming.hpp"

namespace stride_align::hamming {

namespace nb = nanobind;

inline void require_equal_length(std::size_t q_len, std::size_t t_len) {
  if (q_len != t_len) {
    PyErr_Format(
        PyExc_ValueError,
        "Hamming requires equal-length inputs (query has length %zu, "
        "target has length %zu)",
        q_len, t_len);
    throw nb::python_error();
  }
}

inline Score dispatch_score(nb::handle query, nb::handle target) {
  namespace bv = ::stride_align::byte_view;
  // Zero-copy path for bytes / 1-byte unicode.
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
    require_equal_length(q_len, t_len);
    return static_cast<Score>(
        ::stride_align::hamming_simd::hamming_within_string(
            q_ptr, t_ptr, q_len));
  }

  // Fall back to scalar with prepared tokens (handles wider unicode).
  // Reuse the existing preprocess infrastructure.
  const auto prepared = prepare_alignment(
      query, target,
      /*match_score=*/1,
      /*mismatch_score=*/-1,
      /*gap_open_score=*/-1,
      /*gap_extend_score=*/-1,
      /*width=*/0U);
  return std::visit(
      [](const auto& q, const auto& t) -> Score {
        using QT = typename std::decay_t<decltype(q)>::value_type;
        using TT = typename std::decay_t<decltype(t)>::value_type;
        if constexpr (std::is_same_v<QT, TT>) {
          require_equal_length(q.size(), t.size());
          return static_cast<Score>(hamming_scalar<QT>(
              std::span<const QT>(q.data(), q.size()),
              std::span<const QT>(t.data(), t.size())));
        } else {
          PyErr_SetString(
              PyExc_RuntimeError,
              "hamming: mismatched token widths from preprocess");
          throw nb::python_error();
        }
      },
      prepared.query_tokens,
      prepared.target_tokens);
}

inline double dispatch_normalized_score(nb::handle query, nb::handle target) {
  // For normalization we need the length. Use whichever path
  // dispatch_score takes; the length is the same either way.
  namespace bv = ::stride_align::byte_view;
  const bv::ByteCompatKind q_kind = bv::classify(query.ptr());
  std::size_t n = 0;
  Score d = 0;
  if (q_kind != bv::ByteCompatKind::None) {
    const bv::ByteCompatKind t_kind = bv::classify(target.ptr());
    if (t_kind != bv::ByteCompatKind::None) {
      const std::uint8_t* q_ptr = nullptr;
      const std::uint8_t* t_ptr = nullptr;
      std::size_t t_len = 0;
      bv::view(query.ptr(), q_kind, q_ptr, n);
      bv::view(target.ptr(), t_kind, t_ptr, t_len);
      require_equal_length(n, t_len);
      d = static_cast<Score>(
          ::stride_align::hamming_simd::hamming_within_string(
              q_ptr, t_ptr, n));
      return normalize(static_cast<std::size_t>(d), n);
    }
  }
  // Fall back to scalar.
  d = dispatch_score(query, target);
  n = static_cast<std::size_t>(PyObject_Length(query.ptr()));
  return normalize(static_cast<std::size_t>(d), n);
}

inline std::vector<Score> dispatch_scores(
    nb::handle query,
    nb::handle targets) {
  // Try the SIMD batch first; it handles its own bytes-vs-wider fork
  // and signals "not handled" by returning an empty vector.
  auto out = ::stride_align::hamming_simd::hamming_scores_simd(query, targets);
  if (!out.empty()) {
    return out;
  }

  // Wider-unicode fallback: per-pair scalar dispatch.
  PyObject* fast_targets =
      PySequence_Fast(targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count = static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
  out.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = dispatch_score(query, nb::handle(items[i]));
  }
  return out;
}

inline std::vector<double> dispatch_normalized_scores(
    nb::handle query,
    nb::handle targets) {
  auto raw = dispatch_scores(query, targets);
  const std::size_t q_len = static_cast<std::size_t>(PyObject_Length(query.ptr()));
  std::vector<double> out;
  out.reserve(raw.size());
  for (const auto d : raw) {
    out.push_back(normalize(static_cast<std::size_t>(d), q_len));
  }
  return out;
}

}  // namespace stride_align::hamming
