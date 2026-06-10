#pragma once

// nanobind-facing dispatch for ``sa._token_sort_ratio_kernel`` and
// ``sa._token_set_ratio_kernel``. Tries a zero-copy byte view first
// (ASCII Python str or bytes-like) before falling back to the
// codepoint widening path.

#include <cstddef>
#include <cstdint>
#include <span>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "lcs_dispatch.hpp"  // widen_to_codepoints
#include "stride_align/token_ratios.hpp"

namespace stride_align::token_ratios {

namespace nb = nanobind;

inline double dispatch_token_sort_ratio(nb::handle a, nb::handle b) {
  namespace bv = ::stride_align::byte_view;
  const bv::ByteCompatKind ka = bv::classify(a.ptr());
  const bv::ByteCompatKind kb = bv::classify(b.ptr());
  if (ka != bv::ByteCompatKind::None && kb != bv::ByteCompatKind::None) {
    const std::uint8_t* ap = nullptr; std::size_t alen = 0;
    const std::uint8_t* bp = nullptr; std::size_t blen = 0;
    bv::view(a.ptr(), ka, ap, alen);
    bv::view(b.ptr(), kb, bp, blen);
    return token_sort_ratio_bytes(
        std::span<const std::uint8_t>(ap, alen),
        std::span<const std::uint8_t>(bp, blen));
  }
  return token_sort_ratio(
      ::stride_align::lcs::widen_to_codepoints(a),
      ::stride_align::lcs::widen_to_codepoints(b));
}

inline double dispatch_token_set_ratio(nb::handle a, nb::handle b) {
  namespace bv = ::stride_align::byte_view;
  const bv::ByteCompatKind ka = bv::classify(a.ptr());
  const bv::ByteCompatKind kb = bv::classify(b.ptr());
  if (ka != bv::ByteCompatKind::None && kb != bv::ByteCompatKind::None) {
    const std::uint8_t* ap = nullptr; std::size_t alen = 0;
    const std::uint8_t* bp = nullptr; std::size_t blen = 0;
    bv::view(a.ptr(), ka, ap, alen);
    bv::view(b.ptr(), kb, bp, blen);
    return token_set_ratio_bytes(
        std::span<const std::uint8_t>(ap, alen),
        std::span<const std::uint8_t>(bp, blen));
  }
  return token_set_ratio(
      ::stride_align::lcs::widen_to_codepoints(a),
      ::stride_align::lcs::widen_to_codepoints(b));
}

}  // namespace stride_align::token_ratios
