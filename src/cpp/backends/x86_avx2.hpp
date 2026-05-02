#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC push_options
#pragma GCC target("avx2")
#endif

#include "backends/affine_fixed_kernel.hpp"
#include "backends/farrar_fixed_kernel.hpp"
#include "backends/profile_traceback.hpp"
#include "backends/x86_fixed_kernel.hpp"

namespace stride_align::backend_avx2 {

namespace nb = nanobind;

#if defined(__GNUC__) || defined(__clang__)
#define STRIDE_ALIGN_X86_BASELINE __attribute__((target("default")))
#else
#define STRIDE_ALIGN_X86_BASELINE
#endif

template <int ShiftBytes>
inline __m256i shift_left_zero_256(__m256i vector) {
  const __m256i shifted = _mm256_slli_si256(vector, ShiftBytes);
  const __m256i carry_source = _mm256_permute2x128_si256(vector, vector, 0x08);
  const __m256i carry = _mm256_srli_si256(carry_source, 16 - ShiftBytes);
  return _mm256_or_si256(shifted, carry);
}

template <int LaneBytes>
inline __m256i first_lane_mask_256() {
  static_assert(LaneBytes == 1 || LaneBytes == 2 || LaneBytes == 4 || LaneBytes == 8);
  if constexpr (LaneBytes == 8) {
    return _mm256_set_epi64x(0, 0, 0, -1LL);
  } else {
    return _mm256_set_epi64x(0, 0, 0, (1LL << (LaneBytes * 8)) - 1LL);
  }
}

template <int ByteCount>
inline __m256i first_bytes_mask_256() {
  static_assert(ByteCount == 1 || ByteCount == 2 || ByteCount == 4 ||
      ByteCount == 8 || ByteCount == 16);
  if constexpr (ByteCount == 16) {
    return _mm256_set_epi64x(0, 0, -1LL, -1LL);
  } else if constexpr (ByteCount > 8) {
    return _mm256_set_epi64x(
        0,
        0,
        (1LL << ((ByteCount - 8) * 8)) - 1LL,
        -1LL);
  } else if constexpr (ByteCount == 8) {
    return _mm256_set_epi64x(0, 0, 0, -1LL);
  } else {
    return _mm256_set_epi64x(0, 0, 0, (1LL << (ByteCount * 8)) - 1LL);
  }
}

template <int ByteCount>
inline __m256i leading_bytes_mask_256() {
  static_assert(ByteCount >= 0 && ByteCount <= 32);
  constexpr auto mask_word = [](int byte_count) constexpr -> long long {
    if (byte_count <= 0) {
      return 0;
    }
    if (byte_count >= 8) {
      return -1LL;
    }
    return static_cast<long long>((1ULL << (byte_count * 8)) - 1ULL);
  };
  return _mm256_set_epi64x(
      mask_word(ByteCount - 24),
      mask_word(ByteCount - 16),
      mask_word(ByteCount - 8),
      mask_word(ByteCount));
}

template <int LaneBytes>
inline __m256i shift_left_insert_256(__m256i vector, __m256i inserted) {
  return _mm256_blendv_epi8(
      shift_left_zero_256<LaneBytes>(vector),
      inserted,
      first_lane_mask_256<LaneBytes>());
}

template <int ByteShift>
inline __m256i shift_left_insert_bytes_256(__m256i vector, __m256i inserted) {
  return _mm256_blendv_epi8(
      shift_left_zero_256<ByteShift>(vector),
      inserted,
      first_bytes_mask_256<ByteShift>());
}

template <typename Cell>
inline __m256i set1_cell_256(Cell value) {
  if constexpr (sizeof(Cell) == 1) {
    return _mm256_set1_epi8(static_cast<char>(value));
  } else if constexpr (sizeof(Cell) == 2) {
    return _mm256_set1_epi16(value);
  } else if constexpr (sizeof(Cell) == 4) {
    return _mm256_set1_epi32(value);
  } else {
    return _mm256_set1_epi64x(value);
  }
}

template <typename Cell>
inline __m256i add_cell_256(__m256i lhs, __m256i rhs) {
  if constexpr (sizeof(Cell) == 1) {
    return _mm256_add_epi8(lhs, rhs);
  } else if constexpr (sizeof(Cell) == 2) {
    return _mm256_add_epi16(lhs, rhs);
  } else if constexpr (sizeof(Cell) == 4) {
    return _mm256_add_epi32(lhs, rhs);
  } else {
    return _mm256_add_epi64(lhs, rhs);
  }
}

template <typename Cell>
inline __m256i eq_cell_256(__m256i lhs, __m256i rhs) {
  if constexpr (sizeof(Cell) == 1) {
    return _mm256_cmpeq_epi8(lhs, rhs);
  } else if constexpr (sizeof(Cell) == 2) {
    return _mm256_cmpeq_epi16(lhs, rhs);
  } else if constexpr (sizeof(Cell) == 4) {
    return _mm256_cmpeq_epi32(lhs, rhs);
  } else {
    return _mm256_cmpeq_epi64(lhs, rhs);
  }
}

template <typename Cell>
inline __m256i max_cell_256(__m256i lhs, __m256i rhs) {
  if constexpr (sizeof(Cell) == 1) {
    return _mm256_max_epi8(lhs, rhs);
  } else if constexpr (sizeof(Cell) == 2) {
    return _mm256_max_epi16(lhs, rhs);
  } else if constexpr (sizeof(Cell) == 4) {
    return _mm256_max_epi32(lhs, rhs);
  } else {
    const __m256i mask = _mm256_cmpgt_epi64(lhs, rhs);
    return _mm256_blendv_epi8(rhs, lhs, mask);
  }
}

template <typename Cell>
inline __m256i add_sentinel_cell_256(__m256i lhs, __m256i rhs, Cell sentinel) {
  const __m256i sum = add_cell_256<Cell>(lhs, rhs);
  const __m256i sentinel_vector = set1_cell_256(sentinel);
  const __m256i mask = eq_cell_256<Cell>(lhs, sentinel_vector);
  return _mm256_blendv_epi8(sum, sentinel_vector, mask);
}

template <int LaneBytes, int LaneCount, typename Cell>
inline __m256i global_lazy_f_prefix_carry_256(
    __m256i final_f,
    std::size_t segment_count,
    Cell gap_extend_score,
    Cell low_score) {
  const Score lane_span_gap =
      static_cast<Score>(segment_count) * static_cast<Score>(gap_extend_score);
  const __m256i low_vector = set1_cell_256(low_score);
  __m256i prefix = shift_left_insert_bytes_256<LaneBytes>(final_f, low_vector);

  if constexpr (LaneCount > 1) {
    __m256i candidate = shift_left_insert_bytes_256<LaneBytes>(prefix, low_vector);
    candidate = add_sentinel_cell_256<Cell>(
        candidate,
        set1_cell_256(static_cast<Cell>(lane_span_gap)),
        low_score);
    prefix = max_cell_256<Cell>(prefix, candidate);
  }
  if constexpr (LaneCount > 2) {
    __m256i candidate = shift_left_insert_bytes_256<LaneBytes * 2>(prefix, low_vector);
    candidate = add_sentinel_cell_256<Cell>(
        candidate,
        set1_cell_256(static_cast<Cell>(lane_span_gap * 2)),
        low_score);
    prefix = max_cell_256<Cell>(prefix, candidate);
  }
  if constexpr (LaneCount > 4) {
    __m256i candidate = shift_left_insert_bytes_256<LaneBytes * 4>(prefix, low_vector);
    candidate = add_sentinel_cell_256<Cell>(
        candidate,
        set1_cell_256(static_cast<Cell>(lane_span_gap * 4)),
        low_score);
    prefix = max_cell_256<Cell>(prefix, candidate);
  }
  if constexpr (LaneCount > 8) {
    __m256i candidate = shift_left_insert_bytes_256<LaneBytes * 8>(prefix, low_vector);
    candidate = add_sentinel_cell_256<Cell>(
        candidate,
        set1_cell_256(static_cast<Cell>(lane_span_gap * 8)),
        low_score);
    prefix = max_cell_256<Cell>(prefix, candidate);
  }
  if constexpr (LaneCount > 16) {
    __m256i candidate = shift_left_insert_bytes_256<LaneBytes * 16>(prefix, low_vector);
    candidate = add_sentinel_cell_256<Cell>(
        candidate,
        set1_cell_256(static_cast<Cell>(lane_span_gap * 16)),
        low_score);
    prefix = max_cell_256<Cell>(prefix, candidate);
  }

  return prefix;
}

inline __m256i global_lazy_f_prefix_carry_i16_no_padding_256(
    __m256i final_f,
    std::size_t segment_count,
    std::int16_t gap_extend_score,
    std::int16_t low_score) {
  const Score lane_span_gap =
      static_cast<Score>(segment_count) * static_cast<Score>(gap_extend_score);
  const __m256i low_vector = _mm256_set1_epi16(low_score);
  __m256i prefix = shift_left_insert_256<2>(final_f, low_vector);

  {
    __m256i candidate = shift_left_zero_256<2>(prefix);
    candidate = _mm256_add_epi16(
        candidate,
        _mm256_set1_epi16(static_cast<std::int16_t>(lane_span_gap)));
    candidate = _mm256_blendv_epi8(candidate, low_vector, leading_bytes_mask_256<4>());
    prefix = _mm256_max_epi16(prefix, candidate);
  }
  {
    __m256i candidate = shift_left_zero_256<4>(prefix);
    candidate = _mm256_add_epi16(
        candidate,
        _mm256_set1_epi16(static_cast<std::int16_t>(lane_span_gap * 2)));
    candidate = _mm256_blendv_epi8(candidate, low_vector, leading_bytes_mask_256<6>());
    prefix = _mm256_max_epi16(prefix, candidate);
  }
  {
    __m256i candidate = shift_left_zero_256<8>(prefix);
    candidate = _mm256_add_epi16(
        candidate,
        _mm256_set1_epi16(static_cast<std::int16_t>(lane_span_gap * 4)));
    candidate = _mm256_blendv_epi8(candidate, low_vector, leading_bytes_mask_256<10>());
    prefix = _mm256_max_epi16(prefix, candidate);
  }
  {
    __m256i candidate = shift_left_zero_256<16>(prefix);
    candidate = _mm256_add_epi16(
        candidate,
        _mm256_set1_epi16(static_cast<std::int16_t>(lane_span_gap * 8)));
    candidate = _mm256_blendv_epi8(candidate, low_vector, leading_bytes_mask_256<18>());
    prefix = _mm256_max_epi16(prefix, candidate);
  }

  return prefix;
}

inline std::uint32_t compress_even_bits_32(std::uint32_t mask) {
  mask &= 0x55555555U;
  mask = (mask | (mask >> 1U)) & 0x33333333U;
  mask = (mask | (mask >> 2U)) & 0x0F0F0F0FU;
  mask = (mask | (mask >> 4U)) & 0x00FF00FFU;
  mask = (mask | (mask >> 8U)) & 0x0000FFFFU;
  return mask;
}

inline std::uint32_t lane_mask_epi16_256(__m256i mask) {
  return compress_even_bits_32(
      static_cast<std::uint32_t>(_mm256_movemask_epi8(mask)) >> 1U);
}

template <typename Token, typename Cell>
struct SimdOps;

template <>
struct SimdOps<std::uint8_t, std::int8_t> {
  using vector_type = __m256i;
  static constexpr std::size_t alignment = 32;
  static constexpr std::size_t lane_count = 32;
  static constexpr bool has_vector_max = true;

  static vector_type load_tokens(const std::uint8_t* values) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }

  static vector_type load_cells(const std::int8_t* values) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }

  static vector_type load_aligned_cells(const std::int8_t* values) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(values));
  }

  static void store_cells(std::int8_t* values, vector_type vector) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(values), vector);
  }

  static void store_aligned_cells(std::int8_t* values, vector_type vector) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(values), vector);
  }

  static vector_type set1(std::int8_t value) {
    return _mm256_set1_epi8(value);
  }

  static vector_type zero() {
    return _mm256_setzero_si256();
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return _mm256_add_epi8(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int8_t sentinel) {
    const vector_type sum = _mm256_add_epi8(lhs, rhs);
    const vector_type mask = _mm256_cmpeq_epi8(lhs, set1(sentinel));
    return _mm256_blendv_epi8(sum, set1(sentinel), mask);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return _mm256_max_epi8(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return shift_left_zero_256<1>(vector);
  }

  static vector_type shift_left_insert(vector_type vector, std::int8_t inserted) {
    return shift_left_insert_256<1>(vector, set1(inserted));
  }

  static vector_type global_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int8_t gap_extend_score,
      std::int8_t low_score) {
    return global_lazy_f_prefix_carry_256<1, lane_count>(
        final_f,
        segment_count,
        gap_extend_score,
        low_score);
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return _mm256_movemask_epi8(_mm256_cmpgt_epi8(lhs, rhs)) != 0;
  }

  static vector_type greater_mask(vector_type lhs, vector_type rhs) {
    return _mm256_cmpgt_epi8(lhs, rhs);
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return static_cast<std::uint32_t>(_mm256_movemask_epi8(_mm256_cmpgt_epi8(lhs, rhs)));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return static_cast<std::uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(lhs, rhs)));
  }

  static vector_type bit_or(vector_type lhs, vector_type rhs) {
    return _mm256_or_si256(lhs, rhs);
  }

  static bool any_nonzero(vector_type value) {
    return _mm256_testz_si256(value, value) == 0;
  }

  static vector_type substitution(
      const std::uint8_t* query,
      const std::uint8_t* target,
      std::int8_t match_score,
      std::int8_t mismatch_score) {
    const vector_type mask = _mm256_cmpeq_epi8(load_tokens(query), load_tokens(target));
    return _mm256_blendv_epi8(set1(mismatch_score), set1(match_score), mask);
  }
};

template <>
struct SimdOps<std::uint16_t, std::int16_t> {
  using vector_type = __m256i;
  static constexpr std::size_t alignment = 32;
  static constexpr std::size_t lane_count = 16;
  static constexpr bool has_vector_max = true;
  static constexpr bool dense_global_lazy_f_scan = true;
  static constexpr bool plain_global_main_f_after_first_segment = true;
  static constexpr bool global_main_f_segment64_unroll = true;

  static vector_type load_tokens(const std::uint16_t* values) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }

  static vector_type load_cells(const std::int16_t* values) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }

  static vector_type load_aligned_cells(const std::int16_t* values) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(values));
  }

  static void store_cells(std::int16_t* values, vector_type vector) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(values), vector);
  }

  static void store_aligned_cells(std::int16_t* values, vector_type vector) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(values), vector);
  }

  static vector_type set1(std::int16_t value) {
    return _mm256_set1_epi16(value);
  }

  static vector_type zero() {
    return _mm256_setzero_si256();
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return _mm256_add_epi16(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int16_t sentinel) {
    const vector_type sum = _mm256_add_epi16(lhs, rhs);
    const vector_type mask = _mm256_cmpeq_epi16(lhs, set1(sentinel));
    return _mm256_blendv_epi8(sum, set1(sentinel), mask);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return _mm256_max_epi16(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return shift_left_zero_256<2>(vector);
  }

  static vector_type shift_left_insert(vector_type vector, std::int16_t inserted) {
    return shift_left_insert_256<2>(vector, set1(inserted));
  }

  static vector_type global_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int16_t gap_extend_score,
      std::int16_t low_score) {
    return global_lazy_f_prefix_carry_256<2, lane_count>(
        final_f,
        segment_count,
        gap_extend_score,
        low_score);
  }

  static vector_type global_lazy_f_prefix_carry_no_padding(
      vector_type final_f,
      std::size_t segment_count,
      std::int16_t gap_extend_score,
      std::int16_t low_score) {
    return global_lazy_f_prefix_carry_i16_no_padding_256(
        final_f,
        segment_count,
        gap_extend_score,
        low_score);
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return _mm256_movemask_epi8(_mm256_cmpgt_epi16(lhs, rhs)) != 0;
  }

  static vector_type greater_mask(vector_type lhs, vector_type rhs) {
    return _mm256_cmpgt_epi16(lhs, rhs);
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return lane_mask_epi16_256(_mm256_cmpgt_epi16(lhs, rhs));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return lane_mask_epi16_256(_mm256_cmpeq_epi16(lhs, rhs));
  }

  static vector_type bit_or(vector_type lhs, vector_type rhs) {
    return _mm256_or_si256(lhs, rhs);
  }

  static bool any_nonzero(vector_type value) {
    return _mm256_testz_si256(value, value) == 0;
  }

  static vector_type substitution(
      const std::uint16_t* query,
      const std::uint16_t* target,
      std::int16_t match_score,
      std::int16_t mismatch_score) {
    const vector_type mask = _mm256_cmpeq_epi16(load_tokens(query), load_tokens(target));
    return _mm256_blendv_epi8(set1(mismatch_score), set1(match_score), mask);
  }
};

template <>
struct SimdOps<std::uint32_t, std::int32_t> {
  using vector_type = __m256i;
  static constexpr std::size_t alignment = 32;
  static constexpr std::size_t lane_count = 8;
  static constexpr bool has_vector_max = true;
  static constexpr bool dense_global_lazy_f_scan = true;
  static constexpr bool plain_global_main_f_after_first_segment = true;
  static constexpr bool global_main_f_segment128_unroll = true;

  static vector_type load_tokens(const std::uint32_t* values) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }

  static vector_type load_cells(const std::int32_t* values) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }

  static vector_type load_aligned_cells(const std::int32_t* values) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(values));
  }

  static void store_cells(std::int32_t* values, vector_type vector) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(values), vector);
  }

  static void store_aligned_cells(std::int32_t* values, vector_type vector) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(values), vector);
  }

  static vector_type set1(std::int32_t value) {
    return _mm256_set1_epi32(value);
  }

  static vector_type zero() {
    return _mm256_setzero_si256();
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return _mm256_add_epi32(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int32_t sentinel) {
    const vector_type sum = _mm256_add_epi32(lhs, rhs);
    const vector_type mask = _mm256_cmpeq_epi32(lhs, set1(sentinel));
    return _mm256_blendv_epi8(sum, set1(sentinel), mask);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return _mm256_max_epi32(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return shift_left_zero_256<4>(vector);
  }

  static vector_type shift_left_insert(vector_type vector, std::int32_t inserted) {
    return shift_left_insert_256<4>(vector, set1(inserted));
  }

  static vector_type global_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int32_t gap_extend_score,
      std::int32_t low_score) {
    return global_lazy_f_prefix_carry_256<4, lane_count>(
        final_f,
        segment_count,
        gap_extend_score,
        low_score);
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return _mm256_movemask_epi8(_mm256_cmpgt_epi32(lhs, rhs)) != 0;
  }

  static vector_type greater_mask(vector_type lhs, vector_type rhs) {
    return _mm256_cmpgt_epi32(lhs, rhs);
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return static_cast<std::uint32_t>(
        _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(lhs, rhs))));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return static_cast<std::uint32_t>(
        _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(lhs, rhs))));
  }

  static vector_type bit_or(vector_type lhs, vector_type rhs) {
    return _mm256_or_si256(lhs, rhs);
  }

  static bool any_nonzero(vector_type value) {
    return _mm256_testz_si256(value, value) == 0;
  }

  static vector_type substitution(
      const std::uint32_t* query,
      const std::uint32_t* target,
      std::int32_t match_score,
      std::int32_t mismatch_score) {
    const vector_type mask = _mm256_cmpeq_epi32(load_tokens(query), load_tokens(target));
    return _mm256_blendv_epi8(set1(mismatch_score), set1(match_score), mask);
  }
};

template <>
struct SimdOps<std::uint64_t, std::int64_t> {
  using vector_type = __m256i;
  static constexpr std::size_t alignment = 32;
  static constexpr std::size_t lane_count = 4;
  static constexpr bool has_vector_max = false;

  static vector_type load_tokens(const std::uint64_t* values) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }

  static vector_type load_cells(const std::int64_t* values) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
  }

  static vector_type load_aligned_cells(const std::int64_t* values) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(values));
  }

  static void store_cells(std::int64_t* values, vector_type vector) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(values), vector);
  }

  static void store_aligned_cells(std::int64_t* values, vector_type vector) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(values), vector);
  }

  static vector_type set1(std::int64_t value) {
    return _mm256_set1_epi64x(value);
  }

  static vector_type zero() {
    return _mm256_setzero_si256();
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return _mm256_add_epi64(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int64_t sentinel) {
    const vector_type sum = _mm256_add_epi64(lhs, rhs);
    const vector_type mask = _mm256_cmpeq_epi64(lhs, set1(sentinel));
    return _mm256_blendv_epi8(sum, set1(sentinel), mask);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    const vector_type mask = _mm256_cmpgt_epi64(lhs, rhs);
    return _mm256_blendv_epi8(rhs, lhs, mask);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return shift_left_zero_256<8>(vector);
  }

  static vector_type shift_left_insert(vector_type vector, std::int64_t inserted) {
    return shift_left_insert_256<8>(vector, set1(inserted));
  }

  static vector_type global_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int64_t gap_extend_score,
      std::int64_t low_score) {
    return global_lazy_f_prefix_carry_256<8, lane_count>(
        final_f,
        segment_count,
        gap_extend_score,
        low_score);
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return _mm256_movemask_epi8(_mm256_cmpgt_epi64(lhs, rhs)) != 0;
  }

  static vector_type greater_mask(vector_type lhs, vector_type rhs) {
    return _mm256_cmpgt_epi64(lhs, rhs);
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return static_cast<std::uint32_t>(
        _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(lhs, rhs))));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return static_cast<std::uint32_t>(
        _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(lhs, rhs))));
  }

  static vector_type bit_or(vector_type lhs, vector_type rhs) {
    return _mm256_or_si256(lhs, rhs);
  }

  static bool any_nonzero(vector_type value) {
    return _mm256_testz_si256(value, value) == 0;
  }

  static vector_type substitution(
      const std::uint64_t* query,
      const std::uint64_t* target,
      std::int64_t match_score,
      std::int64_t mismatch_score) {
    const vector_type mask = _mm256_cmpeq_epi64(load_tokens(query), load_tokens(target));
    return _mm256_blendv_epi8(set1(mismatch_score), set1(match_score), mask);
  }
};

struct TargetImplementation {
  using PreparedSmithWatermanFarrarScore =
      farrar_fixed_kernel::detail::PreparedScore<SimdOps>;
  using PreparedAffineScore =
      farrar_fixed_kernel::detail::PreparedAffineScore<SimdOps>;

  static Score smith_waterman_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    if (gap_score > 0) {
      const auto prepared =
          prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
      return x86_fixed_kernel::detail::dispatch_score<SimdOps, true>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    }
    const auto prepared =
        prepare_linear_score_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::dispatch_score<SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static std::vector<Score> smith_waterman_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_farrar_batch_alignment(query, targets, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::dispatch_score_many<SimdOps, true>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static AlignmentResult smith_waterman_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    if (gap_score <= 0) {
      const auto output_prepared =
          prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
      const auto prepared =
          prepare_farrar_alignment(query, target, match_score, mismatch_score, gap_score, width);
      const auto path = farrar_fixed_kernel::detail::dispatch_linear_sw_masked_traceback<SimdOps>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
      return profile_traceback::detail::materialize_alignment_result(output_prepared, path);
    }
    return profile_traceback::linear_path<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static AlignmentPath smith_waterman_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    if (gap_score <= 0) {
      const auto prepared =
          prepare_farrar_alignment(query, target, match_score, mismatch_score, gap_score, width);
      return farrar_fixed_kernel::detail::dispatch_linear_sw_masked_path_info<SimdOps>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    }
    return profile_traceback::linear_path_info<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static std::string smith_waterman_linear_cigar(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    if (gap_score <= 0) {
      const auto prepared =
          prepare_farrar_alignment(query, target, match_score, mismatch_score, gap_score, width);
      return farrar_fixed_kernel::detail::dispatch_linear_sw_masked_cigar<SimdOps>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    }
    return profile_traceback::linear_cigar<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static Score smith_waterman_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_farrar_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::dispatch_score<SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static std::vector<Score> smith_waterman_farrar_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return smith_waterman_scores(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static PreparedSmithWatermanFarrarScore prepare_smith_waterman_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_farrar_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::prepare_score<SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static Score smith_waterman_farrar_score_prepared(
      PreparedSmithWatermanFarrarScore& prepared) {
    return farrar_fixed_kernel::detail::dispatch_prepared_score<SimdOps>(prepared);
  }

  static Score smith_waterman_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    if (gap_open_score <= 0 && gap_extend_score <= 0) {
      const auto prepared = prepare_farrar_alignment(
          query,
          target,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score,
          width);
      return affine_fixed_kernel::detail::dispatch_compact_byte_score<SimdOps>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    }
    return profile_traceback::affine_score<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static std::vector<Score> smith_waterman_affine_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    const auto prepared = prepare_farrar_batch_alignment(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    return farrar_fixed_kernel::detail::dispatch_affine_score_many<SimdOps, true>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static AlignmentResult smith_waterman_affine_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    const auto output_prepared = prepare_alignment(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    const auto prepared = prepare_farrar_alignment(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    const auto path = farrar_fixed_kernel::detail::dispatch_affine_striped_path_info<SimdOps, true>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
    return profile_traceback::detail::materialize_alignment_result(output_prepared, path);
  }

  static AlignmentPath smith_waterman_affine_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    const auto prepared = prepare_farrar_alignment(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    return farrar_fixed_kernel::detail::dispatch_affine_striped_path_info<SimdOps, true>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static std::string smith_waterman_affine_cigar(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    const Score expected_score = smith_waterman_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    return profile_traceback::affine_cigar_with_score<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width,
        expected_score);
  }

  static Score smith_waterman_affine_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    const auto prepared = prepare_farrar_alignment(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    return affine_fixed_kernel::detail::dispatch_compact_byte_score<SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static std::vector<Score> smith_waterman_affine_farrar_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    return smith_waterman_affine_scores(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static PreparedAffineScore prepare_smith_waterman_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    const auto prepared = prepare_farrar_alignment(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    return farrar_fixed_kernel::detail::prepare_affine_score<SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static Score smith_waterman_affine_score_prepared(PreparedAffineScore& prepared) {
    return farrar_fixed_kernel::detail::dispatch_prepared_affine_score<SimdOps>(prepared);
  }

  static PreparedAffineScore prepare_smith_waterman_affine_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    return prepare_smith_waterman_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static Score smith_waterman_affine_farrar_score_prepared(PreparedAffineScore& prepared) {
    return smith_waterman_affine_score_prepared(prepared);
  }

  static PreparedAffineScore prepare_needleman_wunsch_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    const auto prepared = prepare_farrar_alignment(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    return farrar_fixed_kernel::detail::prepare_affine_score<SimdOps, true>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static Score needleman_wunsch_affine_score_prepared(PreparedAffineScore& prepared) {
    return farrar_fixed_kernel::detail::dispatch_prepared_global_affine_score<SimdOps>(prepared);
  }

  static Score needleman_wunsch_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    if (gap_score > 0) {
      const auto prepared =
          prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
      return x86_fixed_kernel::detail::dispatch_score<SimdOps, false>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    }
    const auto prepared =
        prepare_linear_score_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::dispatch_global_score<SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static std::vector<Score> needleman_wunsch_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_farrar_batch_alignment(query, targets, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::dispatch_score_many<SimdOps, false>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static AlignmentResult needleman_wunsch_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return profile_traceback::linear_path<false>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static AlignmentPath needleman_wunsch_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return profile_traceback::linear_path_info<false>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static std::string needleman_wunsch_linear_cigar(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return profile_traceback::linear_cigar<false>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static Score needleman_wunsch_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    const auto prepared = prepare_farrar_alignment(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    return farrar_fixed_kernel::detail::dispatch_global_affine_score<SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static std::vector<Score> needleman_wunsch_affine_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    const auto prepared = prepare_farrar_batch_alignment(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    return farrar_fixed_kernel::detail::dispatch_affine_score_many<SimdOps, false>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static AlignmentResult needleman_wunsch_affine_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    const auto output_prepared = prepare_alignment(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    const auto prepared = prepare_farrar_alignment(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    const auto path = farrar_fixed_kernel::detail::dispatch_affine_striped_path_info<SimdOps, false>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
    return profile_traceback::detail::materialize_alignment_result(output_prepared, path);
  }

  static AlignmentPath needleman_wunsch_affine_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    const auto prepared = prepare_farrar_alignment(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    return farrar_fixed_kernel::detail::dispatch_affine_striped_path_info<SimdOps, false>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static std::string needleman_wunsch_affine_cigar(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    const Score expected_score = needleman_wunsch_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    return profile_traceback::affine_cigar_with_score<false>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width,
        expected_score);
  }
};

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC pop_options
#endif

struct Implementation {
  using PreparedSmithWatermanFarrarScore =
      TargetImplementation::PreparedSmithWatermanFarrarScore;
  using PreparedAffineScore = TargetImplementation::PreparedAffineScore;

  static STRIDE_ALIGN_X86_BASELINE bool supported_on_this_machine() noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2") != 0;
#else
    return true;
#endif
  }

  static STRIDE_ALIGN_X86_BASELINE void ensure_supported() {
    if (supported_on_this_machine()) {
      return;
    }

    PyErr_SetString(PyExc_RuntimeError, "x86 AVX2 backend is not available on this machine");
    throw nb::python_error();
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> smith_waterman_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_scores(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE AlignmentResult smith_waterman_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_path(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE AlignmentPath smith_waterman_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_path_info(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::string smith_waterman_linear_cigar(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_linear_cigar(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_farrar_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> smith_waterman_farrar_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_farrar_scores(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE PreparedSmithWatermanFarrarScore
  prepare_smith_waterman_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_smith_waterman_farrar_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_farrar_score_prepared(
      PreparedSmithWatermanFarrarScore& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_farrar_score_prepared(prepared);
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> smith_waterman_affine_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_scores(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE AlignmentResult smith_waterman_affine_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_path(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE AlignmentPath smith_waterman_affine_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_path_info(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::string smith_waterman_affine_cigar(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_cigar(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_affine_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_farrar_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> smith_waterman_affine_farrar_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_farrar_scores(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE PreparedAffineScore prepare_smith_waterman_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_smith_waterman_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_affine_score_prepared(PreparedAffineScore& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_score_prepared(prepared);
  }

  static STRIDE_ALIGN_X86_BASELINE PreparedAffineScore prepare_smith_waterman_affine_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_smith_waterman_affine_farrar_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_affine_farrar_score_prepared(
      PreparedAffineScore& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_farrar_score_prepared(prepared);
  }

  static STRIDE_ALIGN_X86_BASELINE PreparedAffineScore prepare_needleman_wunsch_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_needleman_wunsch_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE Score needleman_wunsch_affine_score_prepared(PreparedAffineScore& prepared) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_score_prepared(prepared);
  }

  static STRIDE_ALIGN_X86_BASELINE Score needleman_wunsch_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> needleman_wunsch_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_scores(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE AlignmentResult needleman_wunsch_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_path(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE AlignmentPath needleman_wunsch_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_path_info(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::string needleman_wunsch_linear_cigar(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_linear_cigar(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE Score needleman_wunsch_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> needleman_wunsch_affine_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_scores(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE AlignmentResult needleman_wunsch_affine_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_path(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE AlignmentPath needleman_wunsch_affine_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_path_info(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::string needleman_wunsch_affine_cigar(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_cigar(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }
};

#undef STRIDE_ALIGN_X86_BASELINE

}  // namespace stride_align::backend_avx2
