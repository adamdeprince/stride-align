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
#include "stride_align/cologne_phonetic.hpp"
#include "stride_align/double_metaphone.hpp"

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

// Cologne wants the German umlauts and ß preserved so they preprocess
// to their Latin-letter equivalents — the ASCII-strip peel that the
// other encoders share would drop them silently. The engine now runs
// in codepoint space: Python ``str`` storage (``PyUnicode_KIND`` is
// 1/2/4 bytes per codepoint) widens straight into a
// ``vector<Codepoint>`` here, and the codepoint-keyed umlaut fold
// (Ä = U+00C4, ä = U+00E4, ß = U+00DF, ...) picks it up directly.
// ``bytes`` input is interpreted as Latin-1 codepoints (each byte is
// its own codepoint), which means ``b"M\xfcller"`` folds correctly
// but ``"Müller".encode("utf-8")`` (two bytes 0xC3 0xBC) does not —
// pass ``str`` for UTF-8 input.
inline std::string dispatch_cologne_phonetic(nb::handle input) {
  std::vector<Codepoint> codepoints;
  if (PyUnicode_Check(input.ptr())) {
    const auto length =
        static_cast<std::size_t>(PyUnicode_GET_LENGTH(input.ptr()));
    const int py_kind = PyUnicode_KIND(input.ptr());
    void* data = PyUnicode_DATA(input.ptr());
    codepoints.resize(length);
    if (py_kind == PyUnicode_1BYTE_KIND) {
      const auto* p = static_cast<const std::uint8_t*>(data);
      for (std::size_t i = 0; i < length; ++i) codepoints[i] = p[i];
    } else if (py_kind == PyUnicode_2BYTE_KIND) {
      const auto* p = static_cast<const std::uint16_t*>(data);
      for (std::size_t i = 0; i < length; ++i) codepoints[i] = p[i];
    } else {
      const auto* p = static_cast<const std::uint32_t*>(data);
      for (std::size_t i = 0; i < length; ++i) codepoints[i] = p[i];
    }
  } else {
    namespace bv = ::stride_align::byte_view;
    const bv::ByteCompatKind kind = bv::classify(input.ptr());
    if (kind != bv::ByteCompatKind::Bytes) {
      ::stride_align::detail::throw_type_error(
          "phonetic encoder requires a str or bytes-like input");
      return {};
    }
    const std::uint8_t* ptr = nullptr;
    std::size_t len = 0;
    bv::view(input.ptr(), kind, ptr, len);
    codepoints.assign(ptr, ptr + len);
  }
  return cologne_phonetic(codepoints);
}

inline std::pair<std::string, std::string>
dispatch_double_metaphone(nb::handle input,
                          std::size_t max_length,
                          DoubleMetaphoneVariant variant) {
  const std::string s = peel_to_ascii_string(input);
  DoubleMetaphoneResult r =
      double_metaphone(std::string_view(s), max_length, variant);
  return {std::move(r.primary), std::move(r.alternate)};
}

// Two-input "encode both, compare" helpers. Implemented at the
// dispatch layer rather than as Python wrappers so callers pay one
// binding-call frame instead of two backend calls + a Python compare.
// Empty codes (unencodable input) never match anything.

inline bool dispatch_soundex_equal(nb::handle a, nb::handle b) {
  const std::string sa = peel_to_ascii_string(a);
  const std::string sb = peel_to_ascii_string(b);
  const std::string ca = soundex(std::string_view(sa));
  if (ca.empty()) return false;
  return ca == soundex(std::string_view(sb));
}

inline bool dispatch_metaphone_equal(
    nb::handle a, nb::handle b, MetaphoneVariant variant) {
  const std::string sa = peel_to_ascii_string(a);
  const std::string sb = peel_to_ascii_string(b);
  const std::string ca = metaphone(std::string_view(sa), variant);
  if (ca.empty()) return false;
  return ca == metaphone(std::string_view(sb), variant);
}

inline bool dispatch_nysiis_equal(nb::handle a, nb::handle b) {
  const std::string sa = peel_to_ascii_string(a);
  const std::string sb = peel_to_ascii_string(b);
  const std::string ca = nysiis(std::string_view(sa));
  if (ca.empty()) return false;
  return ca == nysiis(std::string_view(sb));
}

}  // namespace stride_align::phonetic
