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
#include "stride_align/metaphone.hpp"
#include "stride_align/nysiis.hpp"
#include "stride_align/match_rating.hpp"
#include "stride_align/caverphone.hpp"

namespace stride_align::phonetic {

namespace nb = nanobind;

// Peel ``input`` (bytes / 1-byte unicode / wider unicode) to an
// ASCII-only ``std::string`` for the phonetic encoders. The
// algorithms themselves drop non-letter bytes / codepoints, so we
// just need to surface a ``string_view`` of ASCII-or-below bytes.
inline std::string peel_to_ascii_string(nb::handle input) {
  namespace bv = ::stride_align::byte_view;
  const bv::ByteCompatKind kind = bv::classify(input.ptr());
  if (kind != bv::ByteCompatKind::None) {
    const std::uint8_t* ptr = nullptr;
    std::size_t len = 0;
    bv::view(input.ptr(), kind, ptr, len);
    return std::string(reinterpret_cast<const char*>(ptr), len);
  }
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
    return ascii_only;
  }
  ::stride_align::detail::throw_type_error(
      "phonetic encoder requires a str or bytes-like input");
  return {};
}

// Shared helper: peel ``input`` and run ``encoder`` on it.
template <typename Encoder>
inline std::string dispatch_ascii_encoder(nb::handle input, Encoder&& encoder) {
  const std::string s = peel_to_ascii_string(input);
  return encoder(std::string_view(s));
}

inline std::string dispatch_soundex(nb::handle input) {
  return dispatch_ascii_encoder(input,
      [](std::string_view sv) { return soundex(sv); });
}

inline std::string dispatch_metaphone(
    nb::handle input, MetaphoneVariant variant) {
  return dispatch_ascii_encoder(input,
      [variant](std::string_view sv) { return metaphone(sv, variant); });
}

inline std::string dispatch_nysiis(nb::handle input) {
  return dispatch_ascii_encoder(input,
      [](std::string_view sv) { return nysiis(sv); });
}

inline std::string dispatch_match_rating_codex(nb::handle input) {
  return dispatch_ascii_encoder(input,
      [](std::string_view sv) { return match_rating_codex(sv); });
}

// Two-input comparator: peel both inputs to ASCII, then run the
// comparator (which encodes once internally).
inline bool dispatch_match_rating_compare(nb::handle a, nb::handle b) {
  const std::string sa = peel_to_ascii_string(a);
  const std::string sb = peel_to_ascii_string(b);
  return match_rating_compare(std::string_view(sa), std::string_view(sb));
}

inline std::string dispatch_caverphone(nb::handle input) {
  return dispatch_ascii_encoder(input,
      [](std::string_view sv) { return caverphone(sv); });
}

}  // namespace stride_align::phonetic
