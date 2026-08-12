#pragma once

// Hamming distance — count of positions where two equal-length sequences
// differ. Unlike Levenshtein, this requires identical lengths; host adapters
// decide how to report a length mismatch.
//
// No DP recurrence, no carry chains: every position is independent. That
// makes it the most SIMD-friendly distance metric in this library —
// see src/cpp/hamming_simd.hpp for the within-string and across-target
// kernels.

#include <cstddef>
#include <cstdint>
#include <span>

namespace stride_align::hamming {

// Contiguous byte fast path. The deliberately simple reduction is easier for
// compilers to vectorize than a hand-written per-ISA implementation on every
// backend we support.
inline std::size_t hamming_u8(
    std::span<const std::uint8_t> a,
    std::span<const std::uint8_t> b) noexcept {
  std::size_t distance = a.size();
  for (std::size_t index = 0; index < a.size(); ++index) {
    distance -= static_cast<std::size_t>(a[index] == b[index]);
  }
  return distance;
}

// Scalar reference. Used as correctness oracle and as the fallback when
// inputs aren't byte-compatible (wider unicode etc.). Caller guarantees
// `a.size() == b.size()`.
template <typename Token>
inline std::size_t hamming_scalar(
    std::span<const Token> a,
    std::span<const Token> b) noexcept {
  const std::size_t n = a.size();
  std::size_t count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      ++count;
    }
  }
  return count;
}

inline double normalize(std::size_t distance, std::size_t n) noexcept {
  if (n == 0U) {
    return 1.0;
  }
  return 1.0 - (static_cast<double>(distance) / static_cast<double>(n));
}

}  // namespace stride_align::hamming
