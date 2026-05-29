#pragma once

// nanobind-facing dispatch for the Soundex encoder. Accepts bytes,
// 1-byte unicode (UCS-1), wider unicode (UCS-2 / UCS-4), and
// passes a ``std::string_view`` over the ASCII slice to the
// algorithm. Non-ASCII codepoints in wider strings are skipped —
// the algorithm itself enforces that, so wider-unicode inputs flow
// through after a per-codepoint extraction step.

#include <cstdint>
#include <string>
#include <string_view>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "preprocess.hpp"
#include "stride_align/soundex.hpp"

namespace stride_align::phonetic {

namespace nb = nanobind;

inline std::string dispatch_soundex(nb::handle input) {
  namespace bv = ::stride_align::byte_view;

  // bytes / 1-byte unicode hot path — pass the raw buffer straight
  // through. The algorithm itself filters non-ASCII bytes.
  const bv::ByteCompatKind kind = bv::classify(input.ptr());
  if (kind != bv::ByteCompatKind::None) {
    const std::uint8_t* ptr = nullptr;
    std::size_t len = 0;
    bv::view(input.ptr(), kind, ptr, len);
    return soundex(std::string_view(reinterpret_cast<const char*>(ptr), len));
  }

  // Wider unicode: extract just the codepoints in the ASCII
  // letter range; everything else is dropped before the algorithm
  // even sees it. UCS-2 / UCS-4 inputs reach here.
  if (PyUnicode_Check(input.ptr())) {
    const auto length =
        static_cast<std::size_t>(PyUnicode_GET_LENGTH(input.ptr()));
    const int py_kind = PyUnicode_KIND(input.ptr());
    void* data = PyUnicode_DATA(input.ptr());
    std::string ascii_only;
    ascii_only.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
      const auto cp = static_cast<std::uint32_t>(
          PyUnicode_READ(py_kind, data, static_cast<Py_ssize_t>(i)));
      if (cp < 128U) ascii_only.push_back(static_cast<char>(cp));
    }
    return soundex(std::string_view(ascii_only));
  }

  ::stride_align::detail::throw_type_error(
      "soundex requires a str or bytes-like input");
  return {};  // unreachable
}

}  // namespace stride_align::phonetic
