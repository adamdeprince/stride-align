#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "stride_align/utf8.hpp"

namespace stride_align::encoded {

// A host adapter supplies the character width reported by its native
// encoding implementation. No conversion table is needed: stride-align only
// needs stable equality tokens, so a multibyte character is represented by
// its original bytes. PostgreSQL text cannot contain NUL, which makes the
// big-endian packing below collision-free for its one-to-four-byte server
// encodings (shorter encodings cannot alias a longer sequence padded by NUL).
struct EncodingProfile {
  // One means that every byte is one character. Two and four are supported
  // for fixed-width hosts even though PostgreSQL 18 has no such server
  // encoding. Zero selects the variable-width reader.
  std::size_t fixed_width = 0;
  std::size_t max_width = 4;
};

struct TokenizedText {
  std::vector<std::uint32_t> tokens;
  // Byte offsets has tokens.size() + 1 entries and always ends at input.size().
  std::vector<std::size_t> byte_offsets;
};

namespace detail {

inline std::uint32_t native_token(
    const char* input,
    std::size_t width) {
  if (width == 0U || width > sizeof(std::uint32_t)) {
    throw std::invalid_argument(
        "native character width must be between one and four bytes");
  }
  std::uint32_t token = 0;
  for (std::size_t index = 0; index < width; ++index) {
    const auto byte = static_cast<std::uint8_t>(input[index]);
    if (byte == 0U) {
      throw std::invalid_argument("native text contains an embedded NUL byte");
    }
    token = (token << 8U) | byte;
  }
  return token;
}

template <typename CharacterWidth>
inline std::size_t character_width(
    std::string_view remaining,
    const EncodingProfile& profile,
    CharacterWidth&& variable_width) {
  const std::size_t width = profile.fixed_width != 0U
      ? profile.fixed_width
      : static_cast<std::size_t>(variable_width(remaining));
  if (width == 0U || width > profile.max_width || width > remaining.size() ||
      width > sizeof(std::uint32_t)) {
    throw std::invalid_argument("invalid character boundary in native encoding");
  }
  return width;
}

template <typename Token>
inline std::vector<Token> fixed_tokens(
    std::string_view input,
    std::size_t width) {
  if (width == 0U || input.size() % width != 0U) {
    throw std::invalid_argument("truncated fixed-width native string");
  }
  std::vector<Token> output;
  output.reserve(input.size() / width);
  for (std::size_t offset = 0; offset < input.size(); offset += width) {
    output.push_back(static_cast<Token>(native_token(input.data() + offset, width)));
  }
  return output;
}

}  // namespace detail

template <typename CharacterWidth>
inline TokenizedText tokenize(
    std::string_view input,
    const EncodingProfile& profile,
    CharacterWidth&& variable_width) {
  TokenizedText output;
  output.tokens.reserve(input.size());
  output.byte_offsets.reserve(input.size() + 1U);
  std::size_t offset = 0;
  while (offset < input.size()) {
    output.byte_offsets.push_back(offset);
    const std::string_view remaining = input.substr(offset);
    const std::size_t width = detail::character_width(
        remaining, profile, variable_width);
    output.tokens.push_back(detail::native_token(input.data() + offset, width));
    offset += width;
  }
  output.byte_offsets.push_back(input.size());
  return output;
}

template <typename CharacterWidth>
inline std::size_t character_count(
    std::string_view input,
    const EncodingProfile& profile,
    CharacterWidth&& variable_width) {
  if (profile.fixed_width != 0U) {
    if (input.size() % profile.fixed_width != 0U) {
      throw std::invalid_argument("truncated fixed-width native string");
    }
    return input.size() / profile.fixed_width;
  }
  std::size_t count = 0;
  std::size_t offset = 0;
  while (offset < input.size()) {
    const std::size_t width = detail::character_width(
        input.substr(offset), profile, variable_width);
    offset += width;
    ++count;
  }
  return count;
}

template <typename CharacterWidth>
inline utf8::PreparedPair prepare_pair(
    std::string_view query,
    std::string_view target,
    const EncodingProfile& profile,
    CharacterWidth&& variable_width,
    std::size_t pack_threshold = 64U) {
  const auto bytes = [](std::string_view input) {
    return std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(input.data()), input.size());
  };

  // Every supported single-byte encoding, including SQL_ASCII and LATIN1,
  // is already the ideal 8-bit representation. High-bit bytes are characters,
  // not a reason to transcode or promote.
  if (profile.fixed_width == 1U) {
    return {utf8::TokenWidth::u8, true, false, bytes(query), bytes(target)};
  }

  if (profile.fixed_width == 2U) {
    return {
        utf8::TokenWidth::u16,
        false,
        false,
        detail::fixed_tokens<std::uint16_t>(query, 2U),
        detail::fixed_tokens<std::uint16_t>(target, 2U)};
  }
  if (profile.fixed_width == 4U) {
    return {
        utf8::TokenWidth::u32,
        false,
        false,
        detail::fixed_tokens<std::uint32_t>(query, 4U),
        detail::fixed_tokens<std::uint32_t>(target, 4U)};
  }
  if (profile.fixed_width != 0U) {
    throw std::invalid_argument("unsupported fixed native character width");
  }

  // Variable-width multibyte databases still get the zero-copy ASCII path.
  if (utf8::is_ascii(query) && utf8::is_ascii(target)) {
    return {utf8::TokenWidth::u8, true, false, bytes(query), bytes(target)};
  }

  // Variable-width encodings use opaque native character keys, so equality
  // remains exact without translating the source text to UTF-8.
  auto query_tokens = tokenize(query, profile, variable_width).tokens;
  auto target_tokens = tokenize(target, profile, variable_width).tokens;

  // Match the UTF-8 appliance: short pairs avoid building a hash table and
  // use the smallest fixed-width lane that can hold their native character
  // keys. Long pairs amortize the Swiss-table pass and benefit from dense
  // cardinality packing.
  if (query_tokens.size() + target_tokens.size() < pack_threshold) {
    std::uint32_t maximum = 0;
    for (const std::uint32_t token : query_tokens) {
      maximum = std::max(maximum, token);
    }
    for (const std::uint32_t token : target_tokens) {
      maximum = std::max(maximum, token);
    }
    if (maximum <= std::numeric_limits<std::uint16_t>::max()) {
      return {
          utf8::TokenWidth::u16,
          false,
          false,
          utf8::detail::narrow<std::uint16_t>(query_tokens),
          utf8::detail::narrow<std::uint16_t>(target_tokens)};
    }
    return {
        utf8::TokenWidth::u32,
        false,
        false,
        std::move(query_tokens),
        std::move(target_tokens)};
  }
  auto packed = utf8::detail::pack_pair(query_tokens, target_tokens);
  if (packed.distinct <=
      static_cast<std::size_t>(std::numeric_limits<std::uint8_t>::max())) {
    return {
        utf8::TokenWidth::u8,
        false,
        true,
        utf8::detail::narrow<std::uint8_t>(packed.query),
        utf8::detail::narrow<std::uint8_t>(packed.target)};
  }
  if (packed.distinct <=
      static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max())) {
    return {
        utf8::TokenWidth::u16,
        false,
        true,
        utf8::detail::narrow<std::uint16_t>(packed.query),
        utf8::detail::narrow<std::uint16_t>(packed.target)};
  }
  return {
      utf8::TokenWidth::u32,
      false,
      true,
      std::move(packed.query),
      std::move(packed.target)};
}

}  // namespace stride_align::encoded
