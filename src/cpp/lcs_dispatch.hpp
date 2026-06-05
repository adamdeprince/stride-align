#pragma once

// nanobind-facing dispatch for ``sa.lcs_length``, ``sa.lcs_substring_length``,
// and ``sa.lcs_substring``. Inputs are Python ``str`` / ``bytes``; the
// engine works in codepoint space, so dispatch widens ``PyUnicode_DATA``
// straight into ``std::vector<Codepoint>`` without UTF-8 round-tripping.

#include <cstdint>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "preprocess.hpp"
#include "stride_align/lcs.hpp"

namespace stride_align::lcs {

namespace nb = nanobind;

// Widen a Python ``str``/``bytes`` into a codepoint vector. ``bytes``
// is taken as Latin-1 (each byte its own codepoint), matching the
// other codepoint-engine dispatchers in stride-align.
inline std::vector<Codepoint> widen_to_codepoints(nb::handle input) {
  std::vector<Codepoint> out;
  if (PyUnicode_Check(input.ptr())) {
    const auto length =
        static_cast<std::size_t>(PyUnicode_GET_LENGTH(input.ptr()));
    const int py_kind = PyUnicode_KIND(input.ptr());
    void* data = PyUnicode_DATA(input.ptr());
    out.resize(length);
    if (py_kind == PyUnicode_1BYTE_KIND) {
      const auto* p = static_cast<const std::uint8_t*>(data);
      for (std::size_t i = 0; i < length; ++i) out[i] = p[i];
    } else if (py_kind == PyUnicode_2BYTE_KIND) {
      const auto* p = static_cast<const std::uint16_t*>(data);
      for (std::size_t i = 0; i < length; ++i) out[i] = p[i];
    } else {
      const auto* p = static_cast<const std::uint32_t*>(data);
      for (std::size_t i = 0; i < length; ++i) out[i] = p[i];
    }
    return out;
  }
  namespace bv = ::stride_align::byte_view;
  const bv::ByteCompatKind kind = bv::classify(input.ptr());
  if (kind != bv::ByteCompatKind::Bytes) {
    ::stride_align::detail::throw_type_error(
        "lcs functions require a str or bytes-like input");
    return out;
  }
  const std::uint8_t* ptr = nullptr;
  std::size_t len = 0;
  bv::view(input.ptr(), kind, ptr, len);
  out.assign(ptr, ptr + len);
  return out;
}

inline std::size_t dispatch_lcs_length(nb::handle a, nb::handle b) {
  return lcs_length(widen_to_codepoints(a), widen_to_codepoints(b));
}

inline std::size_t dispatch_lcs_substring_length(nb::handle a, nb::handle b) {
  return lcs_substring_length(widen_to_codepoints(a),
                              widen_to_codepoints(b));
}

// Returns ``str`` when both inputs are str (or one is str and the other
// bytes — codepoints are codepoints), and ``bytes`` when both are
// bytes. Mixed str/bytes returns ``str`` (codepoint form).
inline nb::object dispatch_lcs_substring(nb::handle a, nb::handle b) {
  const auto a_cps = widen_to_codepoints(a);
  const auto b_cps = widen_to_codepoints(b);
  const auto result = lcs_substring(a_cps, b_cps);

  const bool both_bytes = !PyUnicode_Check(a.ptr()) && !PyUnicode_Check(b.ptr());
  if (both_bytes) {
    std::string out;
    out.reserve(result.size());
    for (const auto cp : result) {
      // Latin-1 round-trip: bytes paths only see codepoints < 256.
      out.push_back(static_cast<char>(cp & 0xFF));
    }
    return nb::steal(PyBytes_FromStringAndSize(
        out.data(), static_cast<Py_ssize_t>(out.size())));
  }

  // str path: encode the codepoint vector to UTF-8, hand to CPython.
  std::string utf8;
  utf8.reserve(result.size());
  for (const auto cp : result) {
    if (cp < 0x80) {
      utf8.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      utf8.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      utf8.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      utf8.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      utf8.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      utf8.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      utf8.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      utf8.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      utf8.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      utf8.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }
  return nb::steal(PyUnicode_DecodeUTF8(
      utf8.data(), static_cast<Py_ssize_t>(utf8.size()), nullptr));
}

}  // namespace stride_align::lcs
