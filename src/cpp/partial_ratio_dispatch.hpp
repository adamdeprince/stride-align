#pragma once

// nanobind-facing dispatch for ``sa._partial_ratio_kernel``. Tries a
// zero-copy byte view first (Python ``str`` with 1-byte storage kind
// or ``bytes``-like) before falling back to shared token preparation
// for wide Unicode or arbitrary hashable Python sequences. The byte
// view skips both token preparation and the temporary ``std::vector``
// allocation and the per-codepoint ``< 256`` check, which together
// saved ~600 ns/call on profiling of the hot path.

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <variant>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "preprocess.hpp"
#include "stride_align/partial_ratio.hpp"

namespace stride_align::partial_ratio {

namespace nb = nanobind;

inline double dispatch_partial_ratio(nb::handle a, nb::handle b) {
  namespace bv = ::stride_align::byte_view;
  const bv::ByteCompatKind ka = bv::classify(a.ptr());
  const bv::ByteCompatKind kb = bv::classify(b.ptr());
  if (ka != bv::ByteCompatKind::None && kb != bv::ByteCompatKind::None) {
    const std::uint8_t* ap = nullptr; std::size_t alen = 0;
    const std::uint8_t* bp = nullptr; std::size_t blen = 0;
    bv::view(a.ptr(), ka, ap, alen);
    bv::view(b.ptr(), kb, bp, blen);
    return partial_ratio_bytes(
        std::span<const std::uint8_t>(ap, alen),
        std::span<const std::uint8_t>(bp, blen));
  }
  // The shared alignment preprocessor handles wide Unicode and, more
  // importantly for the TheFuzz facade, arbitrary Python sequences of
  // immutable hashable elements. Both inputs are encoded together, so
  // equal objects receive equal compact integer tokens.
  const auto prepared = ::stride_align::prepare_alignment(
      a, b,
      /*match_score=*/1,
      /*mismatch_score=*/-1,
      /*gap_open_score=*/-1,
      /*gap_extend_score=*/-1,
      /*width=*/0U);
  return std::visit(
      [](const auto& av, const auto& bv) -> double {
        using A = typename std::decay_t<decltype(av)>::value_type;
        using B = typename std::decay_t<decltype(bv)>::value_type;
        if constexpr (std::is_same_v<A, B>) {
          return partial_ratio_engine<A>(
              std::span<const A>(av), std::span<const A>(bv));
        } else {
          PyErr_SetString(
              PyExc_RuntimeError,
              "partial_ratio: mismatched token widths from preprocess");
          throw nb::python_error();
        }
      },
      prepared.query_tokens,
      prepared.target_tokens);
}

}  // namespace stride_align::partial_ratio
