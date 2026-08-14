#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#if defined(__AVX512BW__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#if defined(__loongarch_asx)
#include <lsxintrin.h>
#include <lasxintrin.h>
#elif defined(__loongarch_sx)
#include <lsxintrin.h>
#endif

namespace stride_align::utf8 {

enum class TokenWidth : std::uint8_t {
  u8 = 1,
  u16 = 2,
  u32 = 4,
};

enum class PreparationMode : std::uint8_t {
  pair,
  streaming,
};

enum class NonAsciiPolicy : std::uint8_t {
  // Decode directly to Unicode code points. Pair preparation uses UCS-2 when
  // both strings fit the BMP, otherwise UCS-4.
  fixed_width,
  // For sufficiently long pairs, remap the combined alphabet to dense tokens
  // using the narrowest supported width. Equality is preserved exactly.
  pack_long,
};

struct PreparationOptions {
  PreparationMode mode = PreparationMode::pair;
  NonAsciiPolicy non_ascii = NonAsciiPolicy::pack_long;
  std::size_t pack_threshold = 64;
};

using TokenBuffer = std::variant<
    std::span<const std::uint8_t>,
    std::vector<std::uint8_t>,
    std::vector<std::uint16_t>,
    std::vector<std::uint32_t>>;

struct PreparedPair {
  TokenWidth width = TokenWidth::u8;
  bool borrowed_ascii = true;
  bool packed = false;
  TokenBuffer query;
  TokenBuffer target;

  // A borrowed 8-bit fast path stores non-owning spans. This is ASCII for the
  // UTF-8 adapter; native single-byte host adapters may also borrow high-bit
  // characters. The source views must outlive every consumer of this pair.

  std::size_t query_size() const noexcept {
    return std::visit([](const auto& value) { return value.size(); }, query);
  }

  std::size_t target_size() const noexcept {
    return std::visit([](const auto& value) { return value.size(); }, target);
  }
};

class InvalidUtf8 : public std::invalid_argument {
 public:
  InvalidUtf8(std::size_t offset, const char* reason)
      : std::invalid_argument(
            "invalid UTF-8 at byte " + std::to_string(offset) + ": " + reason),
        offset_(offset) {}

  std::size_t offset() const noexcept { return offset_; }

 private:
  std::size_t offset_;
};

inline bool is_ascii(std::span<const std::uint8_t> input) noexcept {
  const auto* data = input.data();
  std::size_t index = 0;

#if defined(__AVX512BW__)
  for (; index + 64U <= input.size(); index += 64U) {
    const __m512i value = _mm512_loadu_si512(
        reinterpret_cast<const void*>(data + index));
    if (_mm512_movepi8_mask(value) != 0U) {
      return false;
    }
  }
#elif defined(__AVX2__)
  for (; index + 32U <= input.size(); index += 32U) {
    const __m256i value = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(data + index));
    if (_mm256_movemask_epi8(value) != 0) {
      return false;
    }
  }
#elif defined(__SSE2__)
  for (; index + 16U <= input.size(); index += 16U) {
    const __m128i value = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(data + index));
    if (_mm_movemask_epi8(value) != 0) {
      return false;
    }
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  for (; index + 16U <= input.size(); index += 16U) {
    const uint8x16_t value = vld1q_u8(data + index);
    const uint8x16_t high_bits = vandq_u8(value, vdupq_n_u8(0x80U));
#if defined(__aarch64__)
    if (vmaxvq_u8(high_bits) != 0U) {
      return false;
    }
#else
    const uint64x2_t words = vreinterpretq_u64_u8(high_bits);
    if ((vgetq_lane_u64(words, 0) | vgetq_lane_u64(words, 1)) != 0U) {
      return false;
    }
#endif
  }
#elif defined(__loongarch_asx)
  for (; index + 32U <= input.size(); index += 32U) {
    const __m256i value = __lasx_xvld(
        const_cast<void*>(static_cast<const void*>(data + index)), 0);
    const __m256i high_bits = __lasx_xvmskltz_b(value);
    if ((__lasx_xvpickve2gr_du(high_bits, 0) |
         __lasx_xvpickve2gr_du(high_bits, 1) |
         __lasx_xvpickve2gr_du(high_bits, 2) |
         __lasx_xvpickve2gr_du(high_bits, 3)) != 0UL) {
      return false;
    }
  }
#elif defined(__loongarch_sx)
  for (; index + 16U <= input.size(); index += 16U) {
    const __m128i value = __lsx_vld(
        const_cast<void*>(static_cast<const void*>(data + index)), 0);
    const __m128i high_bits = __lsx_vmskltz_b(value);
    if ((__lsx_vpickve2gr_du(high_bits, 0) |
         __lsx_vpickve2gr_du(high_bits, 1)) != 0UL) {
      return false;
    }
  }
#endif

  constexpr std::size_t kWordBytes = sizeof(std::size_t);
  constexpr std::size_t kHighBits = std::numeric_limits<std::size_t>::max() / 0xffU * 0x80U;
  for (; index + kWordBytes <= input.size(); index += kWordBytes) {
    std::size_t word = 0;
    std::memcpy(&word, data + index, kWordBytes);
    if ((word & kHighBits) != 0U) {
      return false;
    }
  }
  for (; index < input.size(); ++index) {
    if ((data[index] & 0x80U) != 0U) {
      return false;
    }
  }
  return true;
}

inline bool is_ascii(std::string_view input) noexcept {
  return is_ascii(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(input.data()), input.size()));
}

namespace detail {

inline bool continuation(std::uint8_t value) noexcept {
  return (value & 0xc0U) == 0x80U;
}

inline std::vector<std::uint32_t> decode(std::string_view input) {
  const auto* data = reinterpret_cast<const std::uint8_t*>(input.data());
  std::vector<std::uint32_t> output;
  output.reserve(input.size());

  std::size_t index = 0;
  while (index < input.size()) {
    const std::uint8_t first = data[index];
    if (first < 0x80U) {
      output.push_back(first);
      ++index;
      continue;
    }

    std::uint32_t codepoint = 0;
    std::size_t length = 0;
    if (first >= 0xc2U && first <= 0xdfU) {
      codepoint = first & 0x1fU;
      length = 2U;
    } else if (first >= 0xe0U && first <= 0xefU) {
      codepoint = first & 0x0fU;
      length = 3U;
    } else if (first >= 0xf0U && first <= 0xf4U) {
      codepoint = first & 0x07U;
      length = 4U;
    } else {
      throw InvalidUtf8(index, "invalid leading byte");
    }

    if (index + length > input.size()) {
      throw InvalidUtf8(index, "truncated sequence");
    }
    for (std::size_t part = 1; part < length; ++part) {
      if (!continuation(data[index + part])) {
        throw InvalidUtf8(index + part, "invalid continuation byte");
      }
      codepoint = (codepoint << 6U) | (data[index + part] & 0x3fU);
    }

    if ((length == 3U && codepoint < 0x800U) ||
        (length == 4U && codepoint < 0x10000U)) {
      throw InvalidUtf8(index, "overlong sequence");
    }
    if (codepoint >= 0xd800U && codepoint <= 0xdfffU) {
      throw InvalidUtf8(index, "surrogate code point");
    }
    if (codepoint > 0x10ffffU) {
      throw InvalidUtf8(index, "code point above U+10FFFF");
    }

    output.push_back(codepoint);
    index += length;
  }
  return output;
}

inline std::size_t table_capacity(std::size_t expected) {
  constexpr std::size_t kHighestPowerOfTwo =
      std::size_t{1} << (std::numeric_limits<std::size_t>::digits - 1U);
  if (expected > (kHighestPowerOfTwo - 1U) / 2U) {
    throw std::length_error("Swiss table capacity overflow");
  }
  return std::bit_ceil(std::max<std::size_t>(expected * 2U + 1U, 16U));
}

inline std::uint32_t control_mask(
    const std::uint8_t* control,
    std::uint8_t needle) noexcept {
#if defined(__SSE2__)
  const __m128i values = _mm_loadu_si128(
      reinterpret_cast<const __m128i*>(control));
  const __m128i matches = _mm_cmpeq_epi8(
      values, _mm_set1_epi8(static_cast<char>(needle)));
  return static_cast<std::uint32_t>(_mm_movemask_epi8(matches));
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  const uint8x16_t matches = vceqq_u8(vld1q_u8(control), vdupq_n_u8(needle));
#if defined(__aarch64__)
  static constexpr std::array<std::uint8_t, 16> kBits{
      1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U,
      1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U};
  const uint8x16_t weighted = vandq_u8(matches, vld1q_u8(kBits.data()));
  const std::uint32_t low = vaddv_u8(vget_low_u8(weighted));
  const std::uint32_t high = vaddv_u8(vget_high_u8(weighted));
  return low | (high << 8U);
#else
  std::array<std::uint8_t, 16> lanes{};
  vst1q_u8(lanes.data(), matches);
  std::uint32_t mask = 0;
  for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
    mask |= static_cast<std::uint32_t>(lanes[lane] != 0U) << lane;
  }
  return mask;
#endif
#elif defined(__loongarch_sx)
  const __m128i values = __lsx_vld(
      const_cast<void*>(static_cast<const void*>(control)), 0);
  const __m128i matches = __lsx_vseq_b(
      values, __lsx_vreplgr2vr_b(static_cast<int>(needle)));
  const __m128i mask = __lsx_vmskltz_b(matches);
  return static_cast<std::uint32_t>(__lsx_vpickve2gr_du(mask, 0));
#else
  std::uint32_t mask = 0;
  for (std::size_t lane = 0; lane < 16U; ++lane) {
    mask |= static_cast<std::uint32_t>(control[lane] == needle) << lane;
  }
  return mask;
#endif
}

template <typename Token>
class SwissMap {
 public:
  explicit SwissMap(std::size_t expected) {
    const std::size_t capacity = table_capacity(expected);
    control_.assign(capacity, kEmpty);
    keys_.resize(capacity);
    values_.resize(capacity);
    mask_ = capacity - 1U;
  }

  Token intern(std::uint32_t key) {
    const std::uint64_t hash = mix(key);
    std::size_t group =
        (static_cast<std::size_t>(hash) & mask_) & ~(kGroupWidth - 1U);
    const std::uint8_t fingerprint = static_cast<std::uint8_t>((hash >> 57U) & 0x7fU);
    for (;;) {
      std::uint32_t matches = control_mask(control_.data() + group, fingerprint);
      while (matches != 0U) {
        const auto lane = static_cast<std::size_t>(std::countr_zero(matches));
        const std::size_t slot = group + lane;
        if (keys_[slot] == key) return values_[slot];
        matches &= matches - 1U;
      }

      const std::uint32_t empty = control_mask(control_.data() + group, kEmpty);
      if (empty != 0U) {
        if (size_ >= static_cast<std::size_t>(std::numeric_limits<Token>::max())) {
          throw std::length_error("combined alphabet does not fit requested token width");
        }
        const auto lane = static_cast<std::size_t>(std::countr_zero(empty));
        const std::size_t slot = group + lane;
        const Token value = static_cast<Token>(++size_);
        control_[slot] = fingerprint;
        keys_[slot] = key;
        values_[slot] = value;
        return value;
      }
      group = (group + kGroupWidth) & mask_;
    }
  }

  std::size_t size() const noexcept { return size_; }

 private:
  static constexpr std::uint8_t kEmpty = 0x80U;
  static constexpr std::size_t kGroupWidth = 16U;

  static std::uint64_t mix(std::uint32_t value) noexcept {
    std::uint64_t result = value;
    result ^= result >> 16U;
    result *= 0x7feb352dULL;
    result ^= result >> 15U;
    result *= 0x846ca68bULL;
    result ^= result >> 16U;
    return result;
  }

  std::vector<std::uint8_t> control_;
  std::vector<std::uint32_t> keys_;
  std::vector<Token> values_;
  std::size_t mask_ = 0;
  std::size_t size_ = 0;
};

struct PackedPair32 {
  std::vector<std::uint32_t> query;
  std::vector<std::uint32_t> target;
  std::size_t distinct = 0;
};

inline PackedPair32 pack_pair(
    std::span<const std::uint32_t> query,
    std::span<const std::uint32_t> target) {
  if (query.size() > std::numeric_limits<std::size_t>::max() - target.size()) {
    throw std::length_error("combined string length overflow");
  }
  SwissMap<std::uint32_t> map(query.size() + target.size());
  PackedPair32 packed;
  packed.query.reserve(query.size());
  packed.target.reserve(target.size());
  for (const std::uint32_t codepoint : query) {
    packed.query.push_back(map.intern(codepoint));
  }
  for (const std::uint32_t codepoint : target) {
    packed.target.push_back(map.intern(codepoint));
  }
  packed.distinct = map.size();
  return packed;
}

template <typename Token>
inline std::vector<Token> narrow(std::span<const std::uint32_t> input) {
  std::vector<Token> output;
  output.reserve(input.size());
  for (const std::uint32_t codepoint : input) {
    output.push_back(static_cast<Token>(codepoint));
  }
  return output;
}

}  // namespace detail

inline PreparedPair prepare_pair(
    std::string_view query,
    std::string_view target,
    PreparationOptions options = {}) {
  const bool query_ascii = is_ascii(query);
  const bool target_ascii = is_ascii(target);

  if (options.mode == PreparationMode::streaming) {
    const auto promote = [](std::string_view input, bool ascii) {
      if (!ascii) return detail::decode(input);
      std::vector<std::uint32_t> output;
      output.reserve(input.size());
      for (const unsigned char value : input) output.push_back(value);
      return output;
    };
    return PreparedPair{
        TokenWidth::u32,
        false,
        false,
        promote(query, query_ascii),
        promote(target, target_ascii)};
  }

  if (query_ascii && target_ascii) {
    auto view = [](std::string_view input) {
      return std::span<const std::uint8_t>(
          reinterpret_cast<const std::uint8_t*>(input.data()), input.size());
    };
    return PreparedPair{
        TokenWidth::u8, true, false, view(query), view(target)};
  }

  auto query_codepoints = detail::decode(query);
  auto target_codepoints = detail::decode(target);

  if (query_codepoints.size() >
      std::numeric_limits<std::size_t>::max() - target_codepoints.size()) {
    throw std::length_error("combined string length overflow");
  }
  const std::size_t total_length =
      query_codepoints.size() + target_codepoints.size();
  if (options.non_ascii == NonAsciiPolicy::pack_long &&
      total_length >= options.pack_threshold) {
    auto packed = detail::pack_pair(query_codepoints, target_codepoints);
    if (packed.distinct <=
        static_cast<std::size_t>(std::numeric_limits<std::uint8_t>::max())) {
      return PreparedPair{
          TokenWidth::u8, false, true,
          detail::narrow<std::uint8_t>(packed.query),
          detail::narrow<std::uint8_t>(packed.target)};
    }
    if (packed.distinct <=
        static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max())) {
      return PreparedPair{
          TokenWidth::u16, false, true,
          detail::narrow<std::uint16_t>(packed.query),
          detail::narrow<std::uint16_t>(packed.target)};
    }
    return PreparedPair{
        TokenWidth::u32, false, true,
        std::move(packed.query), std::move(packed.target)};
  }

  const auto max_codepoint = [](const std::vector<std::uint32_t>& input) {
    std::uint32_t result = 0;
    for (const std::uint32_t value : input) result = std::max(result, value);
    return result;
  };
  if (std::max(max_codepoint(query_codepoints), max_codepoint(target_codepoints)) <= 0xffffU) {
    return PreparedPair{
        TokenWidth::u16,
        false,
        false,
        detail::narrow<std::uint16_t>(query_codepoints),
        detail::narrow<std::uint16_t>(target_codepoints)};
  }
  return PreparedPair{
      TokenWidth::u32,
      false,
      false,
      std::move(query_codepoints),
      std::move(target_codepoints)};
}

inline std::vector<std::uint32_t> prepare_streaming(std::string_view input) {
  // Streaming callers deliberately pay a uniform UCS-4 representation cost:
  // no later string can force already-prepared state to change token width.
  if (is_ascii(input)) {
    std::vector<std::uint32_t> output;
    output.reserve(input.size());
    for (const unsigned char value : input) output.push_back(value);
    return output;
  }
  return detail::decode(input);
}

}  // namespace stride_align::utf8
