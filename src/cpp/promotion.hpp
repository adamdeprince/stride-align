#pragma once

#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace stride_align::promote {

template <typename To, typename From>
std::vector<To> widen(std::span<const From> input) {
  std::vector<To> output(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    output[index] = static_cast<To>(input[index]);
  }
  return output;
}

template <>
inline std::vector<std::uint16_t> widen<std::uint16_t, std::uint8_t>(
    std::span<const std::uint8_t> input) {
  std::vector<std::uint16_t> output(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    output[index] = input[index];
  }
  return output;
}

template <>
inline std::vector<std::uint32_t> widen<std::uint32_t, std::uint16_t>(
    std::span<const std::uint16_t> input) {
  std::vector<std::uint32_t> output(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    output[index] = input[index];
  }
  return output;
}

template <>
inline std::vector<std::uint32_t> widen<std::uint32_t, std::uint8_t>(
    std::span<const std::uint8_t> input) {
  std::vector<std::uint32_t> output(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    output[index] = input[index];
  }
  return output;
}

template <>
inline std::vector<std::uint64_t> widen<std::uint64_t, std::uint8_t>(
    std::span<const std::uint8_t> input) {
  std::vector<std::uint64_t> output(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    output[index] = input[index];
  }
  return output;
}

template <>
inline std::vector<std::uint64_t> widen<std::uint64_t, std::uint16_t>(
    std::span<const std::uint16_t> input) {
  std::vector<std::uint64_t> output(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    output[index] = input[index];
  }
  return output;
}

template <>
inline std::vector<std::uint64_t> widen<std::uint64_t, std::uint32_t>(
    std::span<const std::uint32_t> input) {
  std::vector<std::uint64_t> output(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    output[index] = input[index];
  }
  return output;
}

template <typename To, typename From>
std::vector<To> cast_vector(std::span<const From> input) {
  if constexpr (std::is_same_v<To, From>) {
    return std::vector<To>(input.begin(), input.end());
  } else if constexpr (sizeof(To) > sizeof(From)) {
    return widen<To, From>(input);
  } else {
    std::vector<To> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
      output[index] = static_cast<To>(input[index]);
    }
    return output;
  }
}

}  // namespace stride_align::promote
