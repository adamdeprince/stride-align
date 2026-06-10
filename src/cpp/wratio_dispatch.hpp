#pragma once

// nanobind-facing dispatch for ``sa._wratio_kernel``. Tries a
// zero-copy byte view first (ASCII Python str or bytes-like) before
// falling back to the codepoint widening path.

#include <cstddef>
#include <cstdint>
#include <span>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "lcs_dispatch.hpp"  // widen_to_codepoints
#include "stride_align/wratio.hpp"

namespace stride_align::wratio {

namespace nb = nanobind;

inline double dispatch_wratio(nb::handle a, nb::handle b, double score_cutoff) {
  namespace bv = ::stride_align::byte_view;
  const bv::ByteCompatKind ka = bv::classify(a.ptr());
  const bv::ByteCompatKind kb = bv::classify(b.ptr());
  if (ka != bv::ByteCompatKind::None && kb != bv::ByteCompatKind::None) {
    const std::uint8_t* ap = nullptr; std::size_t alen = 0;
    const std::uint8_t* bp = nullptr; std::size_t blen = 0;
    bv::view(a.ptr(), ka, ap, alen);
    bv::view(b.ptr(), kb, bp, blen);
    return wratio_bytes(
        std::span<const std::uint8_t>(ap, alen),
        std::span<const std::uint8_t>(bp, blen),
        score_cutoff);
  }
  return wratio(
      ::stride_align::lcs::widen_to_codepoints(a),
      ::stride_align::lcs::widen_to_codepoints(b),
      score_cutoff);
}

}  // namespace stride_align::wratio
