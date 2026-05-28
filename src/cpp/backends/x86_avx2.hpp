#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
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
#include "cdist_simd.hpp"
#include "cdist_threshold.hpp"
#include "cdist_topk.hpp"
#include "jaro_simd.hpp"
#include "levenshtein_simd.hpp"
#include "levenshtein_simd_ops.hpp"
#include "indel_simd.hpp"
#include "osa_simd.hpp"

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

inline __m256i local_linear_sw_lazy_f_prefix_carry_i16_256(
    __m256i final_f,
    std::size_t segment_count,
    std::int16_t gap_score) {
  return global_lazy_f_prefix_carry_i16_no_padding_256(
      final_f,
      segment_count,
      gap_score,
      std::int16_t{0});
}

inline __m256i local_linear_sw_lazy_f_prefix_carry_i32_256(
    __m256i final_f,
    std::size_t segment_count,
    std::int32_t gap_score) {
  return global_lazy_f_prefix_carry_256<4, 8>(
      final_f,
      segment_count,
      gap_score,
      std::int32_t{0});
}

inline __m256i local_linear_sw_lazy_f_prefix_carry_i16_64_256(
    __m256i final_f,
    __m256i lane_gap_1,
    __m256i lane_gap_2,
    __m256i lane_gap_4,
    __m256i lane_gap_8,
    __m256i zero) {
  __m256i prefix = _mm256_max_epi16(shift_left_zero_256<2>(final_f), zero);
  __m256i candidate = _mm256_add_epi16(shift_left_zero_256<2>(prefix), lane_gap_1);
  prefix = _mm256_max_epi16(prefix, _mm256_max_epi16(candidate, zero));
  candidate = _mm256_add_epi16(shift_left_zero_256<4>(prefix), lane_gap_2);
  prefix = _mm256_max_epi16(prefix, _mm256_max_epi16(candidate, zero));
  candidate = _mm256_add_epi16(shift_left_zero_256<8>(prefix), lane_gap_4);
  prefix = _mm256_max_epi16(prefix, _mm256_max_epi16(candidate, zero));
  candidate = _mm256_add_epi16(shift_left_zero_256<16>(prefix), lane_gap_8);
  return _mm256_max_epi16(prefix, _mm256_max_epi16(candidate, zero));
}

inline __m256i local_linear_sw_lazy_f_prefix_carry_i32_128_256(
    __m256i final_f,
    __m256i lane_gap_1,
    __m256i lane_gap_2,
    __m256i lane_gap_4,
    __m256i zero) {
  __m256i prefix = _mm256_max_epi32(shift_left_zero_256<4>(final_f), zero);
  __m256i candidate = _mm256_add_epi32(shift_left_zero_256<4>(prefix), lane_gap_1);
  prefix = _mm256_max_epi32(prefix, _mm256_max_epi32(candidate, zero));
  candidate = _mm256_add_epi32(shift_left_zero_256<8>(prefix), lane_gap_2);
  prefix = _mm256_max_epi32(prefix, _mm256_max_epi32(candidate, zero));
  candidate = _mm256_add_epi32(shift_left_zero_256<16>(prefix), lane_gap_4);
  return _mm256_max_epi32(prefix, _mm256_max_epi32(candidate, zero));
}

inline Score reduce_max_i16_256(__m256i value) {
  alignas(32) std::int16_t scores[16] = {};
  _mm256_store_si256(reinterpret_cast<__m256i*>(scores), value);
  std::int16_t best = 0;
  for (const auto score : scores) {
    best = std::max(best, score);
  }
  return static_cast<Score>(best);
}

inline Score reduce_max_i32_256(__m256i value) {
  alignas(32) std::int32_t scores[8] = {};
  _mm256_store_si256(reinterpret_cast<__m256i*>(scores), value);
  std::int32_t best = 0;
  for (const auto score : scores) {
    best = std::max(best, score);
  }
  return static_cast<Score>(best);
}

#define STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment_index) \
  do { \
    const __m256i v_profile = _mm256_load_si256(profile_row + (segment_index)); \
    __m256i v_e = _mm256_load_si256(e_store + (segment_index)); \
    v_h = _mm256_add_epi16(v_h, v_profile); \
    v_h = _mm256_max_epi16(v_h, v_e); \
    v_h = _mm256_max_epi16(v_h, v_f); \
    v_h = _mm256_max_epi16(v_h, zero); \
    _mm256_store_si256(h_store + (segment_index), v_h); \
    best = _mm256_max_epi16(best, v_h); \
    const __m256i v_h_gap = _mm256_add_epi16(v_h, gap); \
    v_e = _mm256_max_epi16(_mm256_add_epi16(v_e, gap), v_h_gap); \
    _mm256_store_si256(e_store + (segment_index), v_e); \
    v_f = _mm256_max_epi16(_mm256_add_epi16(v_f, gap), v_h_gap); \
    v_h = _mm256_load_si256(h_load + (segment_index)); \
  } while (false)

#define STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP_CORRECTED(segment_index) \
  do { \
    const __m256i v_profile = _mm256_load_si256(profile_row + (segment_index)); \
    __m256i v_e = _mm256_load_si256(e_store + (segment_index)); \
    v_h = _mm256_add_epi16(v_h, v_profile); \
    v_h = _mm256_max_epi16(v_h, v_e); \
    v_h = _mm256_max_epi16(v_h, v_f); \
    v_h = _mm256_max_epi16(v_h, zero); \
    _mm256_store_si256(h_store + (segment_index), v_h); \
    best = _mm256_max_epi16(best, v_h); \
    const __m256i v_h_gap = _mm256_add_epi16(v_h, gap); \
    v_e = _mm256_max_epi16(_mm256_add_epi16(v_e, gap), v_h_gap); \
    _mm256_store_si256(e_store + (segment_index), v_e); \
    v_f = _mm256_max_epi16(_mm256_add_epi16(v_f, gap), v_h_gap); \
    v_h = _mm256_load_si256(h_load + (segment_index)); \
    v_h = _mm256_max_epi16(v_h, pending_h); \
    best = _mm256_max_epi16(best, v_h); \
    pending_h = _mm256_add_epi16(pending_h, gap); \
  } while (false)

#define STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN(segment_index) \
  do { \
    __m256i v_h_scan = _mm256_load_si256(h_store + (segment_index)); \
    v_h_scan = _mm256_max_epi16(v_h_scan, v_f); \
    _mm256_store_si256(h_store + (segment_index), v_h_scan); \
    best = _mm256_max_epi16(best, v_h_scan); \
    v_f = _mm256_add_epi16(v_f, gap); \
  } while (false)

#define STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN_BOUNDED(segment_index) \
  do { \
    __m256i v_h_scan = _mm256_load_si256(h_store + (segment_index)); \
    v_h_scan = _mm256_max_epi16(v_h_scan, v_f); \
    _mm256_store_si256(h_store + (segment_index), v_h_scan); \
    best = _mm256_max_epi16(best, v_h_scan); \
    v_f = _mm256_add_epi16(v_f, gap); \
    const __m256i v_continue = _mm256_cmpgt_epi16(v_f, _mm256_add_epi16(v_h_scan, gap)); \
    if (_mm256_movemask_epi8(v_continue) == 0) { \
      goto stride_align_avx2_local_sw_i16_bounded_scan_done; \
    } \
  } while (false)

#define STRIDE_ALIGN_AVX2_LOCAL_SW_I16_FLUSH_PENDING(segment_index) \
  do { \
    __m256i v_h_flush = _mm256_load_si256(h_store + (segment_index)); \
    v_h_flush = _mm256_max_epi16(v_h_flush, pending_h); \
    best = _mm256_max_epi16(best, v_h_flush); \
    pending_h = _mm256_add_epi16(pending_h, gap); \
  } while (false)

inline Score local_sw_score_exact_fill_i16_64(
    farrar_fixed_kernel::detail::PreparedScoreState<std::int16_t>& state) {
  if (state.fast_score.has_value()) {
    return *state.fast_score;
  }
  if (state.gap_score > 0 || state.query_size != 1024U || state.segment_count != 64U ||
      state.target_profile_offsets.empty()) {
    return 0;
  }

  std::fill(state.h_store.begin(), state.h_store.end(), std::int16_t{0});
  std::fill(state.h_load.begin(), state.h_load.end(), std::int16_t{0});
  std::fill(state.e_store.begin(), state.e_store.end(), std::int16_t{0});

  const __m256i zero = _mm256_setzero_si256();
  const __m256i gap = _mm256_set1_epi16(state.gap_score);
  const auto lane_gap = static_cast<std::int16_t>(state.gap_score * 64);
  const __m256i lane_gap_1 = _mm256_set1_epi16(lane_gap);
  const __m256i lane_gap_2 = _mm256_set1_epi16(static_cast<std::int16_t>(lane_gap * 2));
  const __m256i lane_gap_4 = _mm256_set1_epi16(static_cast<std::int16_t>(lane_gap * 4));
  const __m256i lane_gap_8 = _mm256_set1_epi16(static_cast<std::int16_t>(lane_gap * 8));
  const __m256i segment_gap_63 =
      _mm256_set1_epi16(static_cast<std::int16_t>(state.gap_score * 63));
  __m256i best = zero;
  __m256i pending_f = zero;
  bool has_pending_f = false;
  __m256i* h_store = reinterpret_cast<__m256i*>(state.h_store.data());
  __m256i* h_load = reinterpret_cast<__m256i*>(state.h_load.data());
  __m256i* e_store = reinterpret_cast<__m256i*>(state.e_store.data());
  const auto* profile_cells = state.profile.data();
  constexpr std::size_t state_cells = 1024U;
  constexpr std::size_t deferred_correction_min_profile_rows = 48U;
  const bool use_bounded_correction =
      state.kernel_strategy == farrar_fixed_kernel::detail::ScoreKernelStrategy::automatic ||
      state.kernel_strategy == farrar_fixed_kernel::detail::ScoreKernelStrategy::bounded ||
      state.kernel_strategy == farrar_fixed_kernel::detail::ScoreKernelStrategy::compact;
  const bool use_compact_loop =
      state.kernel_strategy == farrar_fixed_kernel::detail::ScoreKernelStrategy::automatic ||
      state.kernel_strategy == farrar_fixed_kernel::detail::ScoreKernelStrategy::compact;
  const bool use_deferred_correction =
      state.kernel_strategy == farrar_fixed_kernel::detail::ScoreKernelStrategy::deferred ||
      state.kernel_strategy ==
          farrar_fixed_kernel::detail::ScoreKernelStrategy::deferred_unroll4 ||
      (state.kernel_strategy == farrar_fixed_kernel::detail::ScoreKernelStrategy::automatic &&
       state.profile.size() / state_cells >= deferred_correction_min_profile_rows);

  if (!use_deferred_correction || use_bounded_correction) {
    for (const auto profile_offset : state.target_profile_offsets) {
      std::swap(h_store, h_load);

      __m256i v_h = shift_left_zero_256<2>(_mm256_load_si256(h_load + 63));
      __m256i v_f = zero;
      const __m256i* profile_row =
          reinterpret_cast<const __m256i*>(profile_cells + profile_offset);

      if (use_compact_loop) {
        for (std::size_t segment = 0; segment < 64U; ++segment) {
          STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment);
        }
      } else {
        for (std::size_t segment = 0; segment < 64U; segment += 8U) {
          STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment + 1U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment + 2U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment + 3U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment + 4U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment + 5U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment + 6U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment + 7U);
        }
      }

      v_f = local_linear_sw_lazy_f_prefix_carry_i16_64_256(
          v_f,
          lane_gap_1,
          lane_gap_2,
          lane_gap_4,
          lane_gap_8,
          zero);
      if (_mm256_movemask_epi8(_mm256_cmpgt_epi16(v_f, zero)) != 0) {
        if (use_bounded_correction) {
          if (use_compact_loop) {
            for (std::size_t segment = 0; segment < 64U; ++segment) {
              STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN_BOUNDED(segment);
            }
          } else {
            for (std::size_t segment = 0; segment < 64U; segment += 8U) {
              STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN_BOUNDED(segment);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN_BOUNDED(segment + 1U);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN_BOUNDED(segment + 2U);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN_BOUNDED(segment + 3U);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN_BOUNDED(segment + 4U);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN_BOUNDED(segment + 5U);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN_BOUNDED(segment + 6U);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN_BOUNDED(segment + 7U);
            }
          }
stride_align_avx2_local_sw_i16_bounded_scan_done:
          ;
        } else {
          for (std::size_t segment = 0; segment < 64U; segment += 8U) {
            STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN(segment);
            STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN(segment + 1U);
            STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN(segment + 2U);
            STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN(segment + 3U);
            STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN(segment + 4U);
            STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN(segment + 5U);
            STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN(segment + 6U);
            STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN(segment + 7U);
          }
        }
      }
    }

    return reduce_max_i16_256(best);
  }

  for (const auto profile_offset : state.target_profile_offsets) {
    std::swap(h_store, h_load);

    __m256i v_f = zero;
    const __m256i* profile_row =
        reinterpret_cast<const __m256i*>(profile_cells + profile_offset);

    if (has_pending_f) {
      __m256i pending_h = pending_f;
      __m256i last_h = _mm256_load_si256(h_load + 63);
      last_h = _mm256_max_epi16(last_h, _mm256_add_epi16(pending_f, segment_gap_63));
      __m256i v_h = shift_left_zero_256<2>(last_h);

      for (std::size_t segment = 0; segment < 64U; segment += 8U) {
        STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP_CORRECTED(segment);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP_CORRECTED(segment + 1U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP_CORRECTED(segment + 2U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP_CORRECTED(segment + 3U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP_CORRECTED(segment + 4U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP_CORRECTED(segment + 5U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP_CORRECTED(segment + 6U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP_CORRECTED(segment + 7U);
      }
    } else {
      __m256i v_h = shift_left_zero_256<2>(_mm256_load_si256(h_load + 63));

      for (std::size_t segment = 0; segment < 64U; segment += 8U) {
        STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment + 1U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment + 2U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment + 3U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment + 4U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment + 5U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment + 6U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP(segment + 7U);
      }
    }

    v_f = local_linear_sw_lazy_f_prefix_carry_i16_64_256(
        v_f,
        lane_gap_1,
        lane_gap_2,
        lane_gap_4,
        lane_gap_8,
        zero);
    has_pending_f = _mm256_movemask_epi8(_mm256_cmpgt_epi16(v_f, zero)) != 0;
    if (has_pending_f) {
      pending_f = v_f;
    } else {
      pending_f = zero;
    }
  }

  if (has_pending_f) {
    __m256i pending_h = pending_f;
    for (std::size_t segment = 0; segment < 64U; segment += 8U) {
      STRIDE_ALIGN_AVX2_LOCAL_SW_I16_FLUSH_PENDING(segment);
      STRIDE_ALIGN_AVX2_LOCAL_SW_I16_FLUSH_PENDING(segment + 1U);
      STRIDE_ALIGN_AVX2_LOCAL_SW_I16_FLUSH_PENDING(segment + 2U);
      STRIDE_ALIGN_AVX2_LOCAL_SW_I16_FLUSH_PENDING(segment + 3U);
      STRIDE_ALIGN_AVX2_LOCAL_SW_I16_FLUSH_PENDING(segment + 4U);
      STRIDE_ALIGN_AVX2_LOCAL_SW_I16_FLUSH_PENDING(segment + 5U);
      STRIDE_ALIGN_AVX2_LOCAL_SW_I16_FLUSH_PENDING(segment + 6U);
      STRIDE_ALIGN_AVX2_LOCAL_SW_I16_FLUSH_PENDING(segment + 7U);
    }
  }

  return reduce_max_i16_256(best);
}

#undef STRIDE_ALIGN_AVX2_LOCAL_SW_I16_FLUSH_PENDING
#undef STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN_BOUNDED
#undef STRIDE_ALIGN_AVX2_LOCAL_SW_I16_SCAN
#undef STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP_CORRECTED
#undef STRIDE_ALIGN_AVX2_LOCAL_SW_I16_STEP

#define STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment_index) \
  do { \
    const __m256i v_profile = _mm256_load_si256(profile_row + (segment_index)); \
    __m256i v_e = _mm256_load_si256(e_store + (segment_index)); \
    v_h = _mm256_add_epi32(v_h, v_profile); \
    v_h = _mm256_max_epi32(v_h, v_e); \
    v_h = _mm256_max_epi32(v_h, v_f); \
    v_h = _mm256_max_epi32(v_h, zero); \
    _mm256_store_si256(h_store + (segment_index), v_h); \
    best = _mm256_max_epi32(best, v_h); \
    const __m256i v_h_gap = _mm256_add_epi32(v_h, gap); \
    v_e = _mm256_max_epi32(_mm256_add_epi32(v_e, gap), v_h_gap); \
    _mm256_store_si256(e_store + (segment_index), v_e); \
    v_f = _mm256_max_epi32(_mm256_add_epi32(v_f, gap), v_h_gap); \
    v_h = _mm256_load_si256(h_load + (segment_index)); \
  } while (false)

#define STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN(segment_index) \
  do { \
    __m256i v_h_scan = _mm256_load_si256(h_store + (segment_index)); \
    v_h_scan = _mm256_max_epi32(v_h_scan, v_f); \
    _mm256_store_si256(h_store + (segment_index), v_h_scan); \
    best = _mm256_max_epi32(best, v_h_scan); \
    v_f = _mm256_add_epi32(v_f, gap); \
  } while (false)

#define STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN_BOUNDED(segment_index) \
  do { \
    __m256i v_h_scan = _mm256_load_si256(h_store + (segment_index)); \
    v_h_scan = _mm256_max_epi32(v_h_scan, v_f); \
    _mm256_store_si256(h_store + (segment_index), v_h_scan); \
    best = _mm256_max_epi32(best, v_h_scan); \
    v_f = _mm256_add_epi32(v_f, gap); \
    const __m256i v_continue = _mm256_cmpgt_epi32(v_f, _mm256_add_epi32(v_h_scan, gap)); \
    if (_mm256_movemask_epi8(v_continue) == 0) { \
      goto stride_align_avx2_local_sw_i32_bounded_scan_done; \
    } \
  } while (false)

#define STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP_CORRECTED(segment_index) \
  do { \
    const __m256i v_profile = _mm256_load_si256(profile_row + (segment_index)); \
    __m256i v_e = _mm256_load_si256(e_store + (segment_index)); \
    v_h = _mm256_add_epi32(v_h, v_profile); \
    v_h = _mm256_max_epi32(v_h, v_e); \
    v_h = _mm256_max_epi32(v_h, v_f); \
    v_h = _mm256_max_epi32(v_h, zero); \
    _mm256_store_si256(h_store + (segment_index), v_h); \
    best = _mm256_max_epi32(best, v_h); \
    const __m256i v_h_gap = _mm256_add_epi32(v_h, gap); \
    v_e = _mm256_max_epi32(_mm256_add_epi32(v_e, gap), v_h_gap); \
    _mm256_store_si256(e_store + (segment_index), v_e); \
    v_f = _mm256_max_epi32(_mm256_add_epi32(v_f, gap), v_h_gap); \
    v_h = _mm256_load_si256(h_load + (segment_index)); \
    v_h = _mm256_max_epi32(v_h, pending_h); \
    best = _mm256_max_epi32(best, v_h); \
    pending_h = _mm256_add_epi32(pending_h, gap); \
  } while (false)

#define STRIDE_ALIGN_AVX2_LOCAL_SW_I32_FLUSH_PENDING(segment_index) \
  do { \
    __m256i v_h_flush = _mm256_load_si256(h_store + (segment_index)); \
    v_h_flush = _mm256_max_epi32(v_h_flush, pending_h); \
    best = _mm256_max_epi32(best, v_h_flush); \
    pending_h = _mm256_add_epi32(pending_h, gap); \
  } while (false)

inline Score local_sw_score_exact_fill_i32_128(
    farrar_fixed_kernel::detail::PreparedScoreState<std::int32_t>& state) {
  if (state.fast_score.has_value()) {
    return *state.fast_score;
  }
  if (state.gap_score > 0 || state.query_size != 1024U || state.segment_count != 128U ||
      state.target_profile_offsets.empty()) {
    return 0;
  }

  std::fill(state.h_store.begin(), state.h_store.end(), std::int32_t{0});
  std::fill(state.h_load.begin(), state.h_load.end(), std::int32_t{0});
  std::fill(state.e_store.begin(), state.e_store.end(), std::int32_t{0});

  const __m256i zero = _mm256_setzero_si256();
  const __m256i gap = _mm256_set1_epi32(state.gap_score);
  const auto lane_gap = static_cast<std::int32_t>(state.gap_score * 128);
  const __m256i lane_gap_1 = _mm256_set1_epi32(lane_gap);
  const __m256i lane_gap_2 = _mm256_set1_epi32(lane_gap * 2);
  const __m256i lane_gap_4 = _mm256_set1_epi32(lane_gap * 4);
  const __m256i segment_gap_127 = _mm256_set1_epi32(state.gap_score * 127);
  __m256i best = zero;
  __m256i pending_f = zero;
  bool has_pending_f = false;
  __m256i* h_store = reinterpret_cast<__m256i*>(state.h_store.data());
  __m256i* h_load = reinterpret_cast<__m256i*>(state.h_load.data());
  __m256i* e_store = reinterpret_cast<__m256i*>(state.e_store.data());
  const auto* profile_cells = state.profile.data();
  const bool use_bounded_correction =
      state.kernel_strategy == farrar_fixed_kernel::detail::ScoreKernelStrategy::automatic ||
      state.kernel_strategy == farrar_fixed_kernel::detail::ScoreKernelStrategy::bounded ||
      state.kernel_strategy == farrar_fixed_kernel::detail::ScoreKernelStrategy::bounded_unroll4 ||
      state.kernel_strategy == farrar_fixed_kernel::detail::ScoreKernelStrategy::compact;
  const bool use_bounded_unroll4 =
      state.kernel_strategy == farrar_fixed_kernel::detail::ScoreKernelStrategy::bounded_unroll4;
  const bool use_compact_loop =
      state.kernel_strategy == farrar_fixed_kernel::detail::ScoreKernelStrategy::automatic ||
      state.kernel_strategy == farrar_fixed_kernel::detail::ScoreKernelStrategy::compact;
  const bool use_deferred_correction =
      state.kernel_strategy == farrar_fixed_kernel::detail::ScoreKernelStrategy::deferred ||
      state.kernel_strategy ==
          farrar_fixed_kernel::detail::ScoreKernelStrategy::deferred_unroll4;
  const bool use_deferred_unroll4 =
      state.kernel_strategy ==
      farrar_fixed_kernel::detail::ScoreKernelStrategy::deferred_unroll4;

  if (!use_deferred_correction || use_bounded_correction) {
    for (const auto profile_offset : state.target_profile_offsets) {
      std::swap(h_store, h_load);

      __m256i v_h = shift_left_zero_256<4>(_mm256_load_si256(h_load + 127));
      __m256i v_f = zero;
      const __m256i* profile_row =
          reinterpret_cast<const __m256i*>(profile_cells + profile_offset);

      if (use_compact_loop) {
        for (std::size_t segment = 0; segment < 128U; ++segment) {
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment);
        }
      } else if (use_bounded_unroll4) {
        for (std::size_t segment = 0; segment < 128U; segment += 4U) {
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 1U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 2U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 3U);
        }
      } else {
        for (std::size_t segment = 0; segment < 128U; segment += 8U) {
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 1U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 2U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 3U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 4U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 5U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 6U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 7U);
        }
      }

      v_f = local_linear_sw_lazy_f_prefix_carry_i32_128_256(
          v_f,
          lane_gap_1,
          lane_gap_2,
          lane_gap_4,
          zero);
      if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(v_f, zero)) != 0) {
        if (use_bounded_correction) {
          if (use_compact_loop) {
            for (std::size_t segment = 0; segment < 128U; ++segment) {
              STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN_BOUNDED(segment);
            }
          } else if (use_bounded_unroll4) {
            for (std::size_t segment = 0; segment < 128U; segment += 4U) {
              STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN_BOUNDED(segment);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN_BOUNDED(segment + 1U);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN_BOUNDED(segment + 2U);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN_BOUNDED(segment + 3U);
            }
          } else {
            for (std::size_t segment = 0; segment < 128U; segment += 8U) {
              STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN_BOUNDED(segment);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN_BOUNDED(segment + 1U);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN_BOUNDED(segment + 2U);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN_BOUNDED(segment + 3U);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN_BOUNDED(segment + 4U);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN_BOUNDED(segment + 5U);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN_BOUNDED(segment + 6U);
              STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN_BOUNDED(segment + 7U);
            }
          }
stride_align_avx2_local_sw_i32_bounded_scan_done:
          ;
        } else {
          for (std::size_t segment = 0; segment < 128U; segment += 8U) {
            STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN(segment);
            STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN(segment + 1U);
            STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN(segment + 2U);
            STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN(segment + 3U);
            STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN(segment + 4U);
            STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN(segment + 5U);
            STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN(segment + 6U);
            STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN(segment + 7U);
          }
        }
      }
    }

    return reduce_max_i32_256(best);
  }

  for (const auto profile_offset : state.target_profile_offsets) {
    std::swap(h_store, h_load);

    __m256i v_f = zero;
    const __m256i* profile_row =
        reinterpret_cast<const __m256i*>(profile_cells + profile_offset);

    if (has_pending_f) {
      __m256i pending_h = pending_f;
      __m256i last_h = _mm256_load_si256(h_load + 127);
      last_h = _mm256_max_epi32(last_h, _mm256_add_epi32(pending_f, segment_gap_127));
      __m256i v_h = shift_left_zero_256<4>(last_h);

      if (use_deferred_unroll4) {
        for (std::size_t segment = 0; segment < 128U; segment += 4U) {
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP_CORRECTED(segment);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP_CORRECTED(segment + 1U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP_CORRECTED(segment + 2U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP_CORRECTED(segment + 3U);
        }
      } else {
        for (std::size_t segment = 0; segment < 128U; segment += 8U) {
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP_CORRECTED(segment);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP_CORRECTED(segment + 1U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP_CORRECTED(segment + 2U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP_CORRECTED(segment + 3U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP_CORRECTED(segment + 4U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP_CORRECTED(segment + 5U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP_CORRECTED(segment + 6U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP_CORRECTED(segment + 7U);
        }
      }
    } else {
      __m256i v_h = shift_left_zero_256<4>(_mm256_load_si256(h_load + 127));

      if (use_deferred_unroll4) {
        for (std::size_t segment = 0; segment < 128U; segment += 4U) {
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 1U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 2U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 3U);
        }
      } else {
        for (std::size_t segment = 0; segment < 128U; segment += 8U) {
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 1U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 2U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 3U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 4U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 5U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 6U);
          STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP(segment + 7U);
        }
      }
    }

    v_f = local_linear_sw_lazy_f_prefix_carry_i32_128_256(
        v_f,
        lane_gap_1,
        lane_gap_2,
        lane_gap_4,
        zero);
    has_pending_f = _mm256_movemask_epi8(_mm256_cmpgt_epi32(v_f, zero)) != 0;
    if (has_pending_f) {
      pending_f = v_f;
    } else {
      pending_f = zero;
    }
  }

  if (has_pending_f) {
    __m256i pending_h = pending_f;
    if (use_deferred_unroll4) {
      for (std::size_t segment = 0; segment < 128U; segment += 4U) {
        STRIDE_ALIGN_AVX2_LOCAL_SW_I32_FLUSH_PENDING(segment);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I32_FLUSH_PENDING(segment + 1U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I32_FLUSH_PENDING(segment + 2U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I32_FLUSH_PENDING(segment + 3U);
      }
    } else {
      for (std::size_t segment = 0; segment < 128U; segment += 8U) {
        STRIDE_ALIGN_AVX2_LOCAL_SW_I32_FLUSH_PENDING(segment);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I32_FLUSH_PENDING(segment + 1U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I32_FLUSH_PENDING(segment + 2U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I32_FLUSH_PENDING(segment + 3U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I32_FLUSH_PENDING(segment + 4U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I32_FLUSH_PENDING(segment + 5U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I32_FLUSH_PENDING(segment + 6U);
        STRIDE_ALIGN_AVX2_LOCAL_SW_I32_FLUSH_PENDING(segment + 7U);
      }
    }
  }

  return reduce_max_i32_256(best);
}

inline Score local_affine_score_exact_fill_i32_128(
    farrar_fixed_kernel::detail::PreparedAffineScoreState<std::int32_t>& state,
    std::span<const std::size_t> target_profile_offsets) {
  if (state.query_size != 1024U || state.segment_count != 128U ||
      target_profile_offsets.empty() || state.gap_open_score > state.gap_extend_score ||
      state.gap_extend_score > 0) {
    return 0;
  }

  std::fill(state.h_store.begin(), state.h_store.end(), std::int32_t{0});
  std::fill(state.h_load.begin(), state.h_load.end(), std::int32_t{0});
  std::fill(state.e_store.begin(), state.e_store.end(), std::int32_t{0});

  const __m256i zero = _mm256_setzero_si256();
  const __m256i gap_open = _mm256_set1_epi32(state.gap_open_score);
  const __m256i gap_extend = _mm256_set1_epi32(state.gap_extend_score);
  const auto lane_gap = static_cast<std::int32_t>(state.gap_extend_score * 128);
  const __m256i lane_gap_1 = _mm256_set1_epi32(lane_gap);
  const __m256i lane_gap_2 = _mm256_set1_epi32(lane_gap * 2);
  const __m256i lane_gap_4 = _mm256_set1_epi32(lane_gap * 4);
  __m256i best = zero;
  __m256i* h_store = reinterpret_cast<__m256i*>(state.h_store.data());
  __m256i* h_load = reinterpret_cast<__m256i*>(state.h_load.data());
  __m256i* e_store = reinterpret_cast<__m256i*>(state.e_store.data());
  const auto* profile_cells = state.profile.data();

  for (const auto profile_offset : target_profile_offsets) {
    std::swap(h_store, h_load);

    __m256i v_h = shift_left_zero_256<4>(_mm256_load_si256(h_load + 127));
    __m256i v_f = zero;
    const __m256i* profile_row =
        reinterpret_cast<const __m256i*>(profile_cells + profile_offset);

    for (std::size_t segment = 0; segment < 128U; ++segment) {
      const __m256i v_profile = _mm256_load_si256(profile_row + segment);
      __m256i v_e = _mm256_load_si256(e_store + segment);
      v_h = _mm256_add_epi32(v_h, v_profile);
      v_h = _mm256_max_epi32(v_h, v_e);
      v_h = _mm256_max_epi32(v_h, v_f);
      v_h = _mm256_max_epi32(v_h, zero);
      _mm256_store_si256(h_store + segment, v_h);
      best = _mm256_max_epi32(best, v_h);

      const __m256i v_h_open = _mm256_add_epi32(v_h, gap_open);
      v_e = _mm256_max_epi32(_mm256_add_epi32(v_e, gap_extend), v_h_open);
      _mm256_store_si256(e_store + segment, v_e);
      v_f = _mm256_max_epi32(_mm256_add_epi32(v_f, gap_extend), v_h_open);
      v_h = _mm256_load_si256(h_load + segment);
    }

    v_f = local_linear_sw_lazy_f_prefix_carry_i32_128_256(
        v_f,
        lane_gap_1,
        lane_gap_2,
        lane_gap_4,
        zero);
    if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(v_f, zero)) != 0) {
      for (std::size_t segment = 0; segment < 128U; ++segment) {
        const __m256i v_h_previous = _mm256_load_si256(h_store + segment);
        const __m256i v_h = _mm256_max_epi32(v_h_previous, v_f);
        _mm256_store_si256(h_store + segment, v_h);
        best = _mm256_max_epi32(best, v_h);
        v_f = _mm256_add_epi32(v_f, gap_extend);
      }
    }
  }

  return reduce_max_i32_256(best);
}

#undef STRIDE_ALIGN_AVX2_LOCAL_SW_I32_FLUSH_PENDING
#undef STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP_CORRECTED
#undef STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN_BOUNDED
#undef STRIDE_ALIGN_AVX2_LOCAL_SW_I32_SCAN
#undef STRIDE_ALIGN_AVX2_LOCAL_SW_I32_STEP

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
  static constexpr bool local_sw_score_exact_segment32 = true;
  static constexpr bool bounded_local_sw_lazy_f_scan = true;

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

  static vector_type local_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int8_t gap_score) {
    const auto span_gap = static_cast<std::int8_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_score));
    const auto zero_vector = zero();
    auto prefix = max(final_f, zero_vector);
    auto shifted = add(shift_left_insert_bytes_256<1>(prefix, zero_vector), set1(span_gap));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_bytes_256<2>(prefix, zero_vector),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 2)));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_bytes_256<4>(prefix, zero_vector),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 4)));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_bytes_256<8>(prefix, zero_vector),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 8)));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_bytes_256<16>(prefix, zero_vector),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 16)));
    prefix = max(prefix, shifted);
    return shift_left_insert_bytes_256<1>(max(prefix, zero_vector), zero_vector);
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
  static constexpr bool local_sw_score_exact_segment64 = true;
  static constexpr bool target_ordered_profile_high_cardinality = true;
  static constexpr std::size_t target_ordered_profile_min_rows = 32U;

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

  static vector_type local_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int16_t gap_score) {
    return local_linear_sw_lazy_f_prefix_carry_i16_256(
        final_f,
        segment_count,
        gap_score);
  }

  static Score local_sw_score_exact_segment64_raw(
      farrar_fixed_kernel::detail::PreparedScoreState<std::int16_t>& state) {
    return local_sw_score_exact_fill_i16_64(state);
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
  static constexpr bool local_sw_score_exact_segment128 = true;
  static constexpr bool target_ordered_profile_high_cardinality = true;
  static constexpr std::size_t target_ordered_profile_min_rows = 48U;

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

  static vector_type local_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int32_t gap_score) {
    return local_linear_sw_lazy_f_prefix_carry_i32_256(
        final_f,
        segment_count,
        gap_score);
  }

  static Score local_sw_score_exact_segment128_raw(
      farrar_fixed_kernel::detail::PreparedScoreState<std::int32_t>& state) {
    return local_sw_score_exact_fill_i32_128(state);
  }

  static Score local_affine_score_exact_segment128_raw(
      farrar_fixed_kernel::detail::PreparedAffineScoreState<std::int32_t>& state,
      std::span<const std::size_t> target_profile_offsets) {
    return local_affine_score_exact_fill_i32_128(state, target_profile_offsets);
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
  using PreparedScoreBatch =
      farrar_fixed_kernel::detail::PreparedScoreBatch<SimdOps>;
  using PreparedAffineScoreBatch =
      farrar_fixed_kernel::detail::PreparedAffineScoreBatch<SimdOps>;

  static Score smith_waterman_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    // Wide-uint64 ndarray dispatch: matching 64-bit SIMD dtypes
    // bypass tokenisation; Cell width is int64 in the kernel.
    {
      namespace nv = ::stride_align::numpy_view;
      auto q_view = nv::try_acquire(query.ptr());
      auto t_view = nv::try_acquire(target.ptr());
      if (::stride_align::farrar_detail::ndarray_pair_is_wide_uint64(q_view, t_view)) {
        const auto wide = ::stride_align::farrar_detail::prepare_farrar_alignment_wide_uint64(
            q_view, t_view, match_score, mismatch_score, gap_score, gap_score, width);
        return farrar_fixed_kernel::detail::wide_dispatch_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
    // Wide-uint32 ndarray dispatch: matching 32-bit SIMD dtypes
    // bypass tokenisation; Cell width is int32 in the kernel.
    {
      namespace nv = ::stride_align::numpy_view;
      auto q_view = nv::try_acquire(query.ptr());
      auto t_view = nv::try_acquire(target.ptr());
      if (::stride_align::farrar_detail::ndarray_pair_is_wide_uint32(q_view, t_view)) {
        const auto wide = ::stride_align::farrar_detail::prepare_farrar_alignment_wide_uint32(
            q_view, t_view, match_score, mismatch_score, gap_score, gap_score, width);
        return farrar_fixed_kernel::detail::wide_dispatch_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
    // Wide-uint16 ndarray dispatch (Phase B parallel path). When both
    // sides are 2-byte SIMD-friendly numpy dtypes (int16/uint16/float16)
    // we skip tokenisation entirely and feed the buffers straight into
    // the 16-bit Farrar kernel.
    {
      namespace nv = ::stride_align::numpy_view;
      auto q_view = nv::try_acquire(query.ptr());
      auto t_view = nv::try_acquire(target.ptr());
      if (::stride_align::farrar_detail::ndarray_pair_is_wide_uint16(q_view, t_view)) {
        const auto wide = ::stride_align::farrar_detail::prepare_farrar_alignment_wide_uint16(
            q_view, t_view, match_score, mismatch_score, gap_score, gap_score, width);
        return farrar_fixed_kernel::detail::wide_dispatch_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
    // Unicode auto-promote: UCS-2 / UCS-4 strings with >256 distinct
    // codepoints route to the 16-bit Farrar kernel via uint16 tokens.
    // If the combined alphabet also exceeds 65 535 distinct codepoints
    // we fall through to the 32-bit Farrar kernel feeding raw UCS-4
    // codepoints as uint32 tokens (no tokenisation map).
    // UCS-1 strings (Latin-1) can have at most 256 distinct codepoints
    // by construction, so we skip the pre-scan entirely for them.
    if (PyUnicode_Check(query.ptr()) && PyUnicode_Check(target.ptr()) &&
        (PyUnicode_KIND(query.ptr()) != PyUnicode_1BYTE_KIND ||
         PyUnicode_KIND(target.ptr()) != PyUnicode_1BYTE_KIND)) {
      if (!::stride_align::farrar_detail::unicode_distinct_count_within(
              query.ptr(), target.ptr(), 256U)) {
        if (!::stride_align::farrar_detail::unicode_alphabet_within_uint16(
                query.ptr(), target.ptr())) {
          const auto wide =
              ::stride_align::farrar_detail::prepare_farrar_alignment_wide_unicode_uint32(
                  query.ptr(), target.ptr(),
                  match_score, mismatch_score, gap_score, gap_score, width);
          return farrar_fixed_kernel::detail::wide_dispatch_score<SimdOps>(
              wide, match_score, mismatch_score, gap_score);
        }
        const auto wide =
            ::stride_align::farrar_detail::prepare_farrar_alignment_wide_unicode_uint16(
                query.ptr(), target.ptr(),
                match_score, mismatch_score, gap_score, gap_score, width);
        return farrar_fixed_kernel::detail::wide_dispatch_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
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
    if (auto wide = farrar_fixed_kernel::detail::try_wide_score_batch<SimdOps, true>(
            query, targets, match_score, mismatch_score, gap_score, width);
        wide) {
      return std::move(*wide);
    }
    const auto prepared =
        prepare_farrar_batch_alignment(query, targets, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::dispatch_score_many<SimdOps, true>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static PreparedScoreBatch prepare_smith_waterman_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_farrar_batch_alignment(query, targets, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::prepare_score_batch<SimdOps, true>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static std::vector<Score> smith_waterman_scores_prepared(PreparedScoreBatch& prepared) {
    return farrar_fixed_kernel::detail::dispatch_prepared_score_many<SimdOps, true>(prepared);
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
      const auto trace =
          farrar_fixed_kernel::detail::dispatch_linear_sw_score_first_masked_cigar_trace<
              SimdOps>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
      farrar_fixed_kernel::detail::LinearTracebackResult path;
      path.score = trace.score;
      path.query_start = trace.query_start;
      path.query_end = trace.query_end;
      path.target_start = trace.target_start;
      path.target_end = trace.target_end;
      path.operations = expand_cigar(trace.cigar);
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
      return farrar_fixed_kernel::detail::dispatch_linear_sw_score_first_masked_cigar_path_info<
          SimdOps>(
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
      return farrar_fixed_kernel::detail::dispatch_linear_sw_score_first_masked_cigar<SimdOps>(
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

  static PreparedScoreBatch prepare_smith_waterman_farrar_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return prepare_smith_waterman_scores(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static std::vector<Score> smith_waterman_farrar_scores_prepared(
      PreparedScoreBatch& prepared) {
    return smith_waterman_scores_prepared(prepared);
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
    if (auto wide = farrar_fixed_kernel::detail::try_wide_affine_score<SimdOps, true>(
            query, target, match_score, mismatch_score,
            gap_open_score, gap_extend_score, width);
        wide) {
      return *wide;
    }
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
    if (auto wide = farrar_fixed_kernel::detail::try_wide_affine_score_batch<SimdOps, true>(
            query, targets, match_score, mismatch_score,
            gap_open_score, gap_extend_score, width);
        wide) {
      return std::move(*wide);
    }
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

  static PreparedAffineScoreBatch prepare_smith_waterman_affine_scores(
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
    return farrar_fixed_kernel::detail::prepare_affine_score_batch<SimdOps, true>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static std::vector<Score> smith_waterman_affine_scores_prepared(
      PreparedAffineScoreBatch& prepared) {
    return farrar_fixed_kernel::detail::dispatch_prepared_affine_score_many<SimdOps, true>(
        prepared);
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
    const Score expected_score = smith_waterman_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    return profile_traceback::affine_path_info_with_score<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width,
        expected_score);
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

  static PreparedAffineScoreBatch prepare_smith_waterman_affine_farrar_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    return prepare_smith_waterman_affine_scores(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static std::vector<Score> smith_waterman_affine_farrar_scores_prepared(
      PreparedAffineScoreBatch& prepared) {
    return smith_waterman_affine_scores_prepared(prepared);
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
    // Wide-uint64 ndarray dispatch: matching 64-bit SIMD dtypes
    // bypass tokenisation; Cell width is int64 in the kernel.
    {
      namespace nv = ::stride_align::numpy_view;
      auto q_view = nv::try_acquire(query.ptr());
      auto t_view = nv::try_acquire(target.ptr());
      if (::stride_align::farrar_detail::ndarray_pair_is_wide_uint64(q_view, t_view)) {
        const auto wide = ::stride_align::farrar_detail::prepare_farrar_alignment_wide_uint64(
            q_view, t_view, match_score, mismatch_score, gap_score, gap_score, width);
        return farrar_fixed_kernel::detail::wide_dispatch_global_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
    // Wide-uint32 ndarray dispatch: matching 32-bit SIMD dtypes
    // bypass tokenisation; Cell width is int32 in the kernel.
    {
      namespace nv = ::stride_align::numpy_view;
      auto q_view = nv::try_acquire(query.ptr());
      auto t_view = nv::try_acquire(target.ptr());
      if (::stride_align::farrar_detail::ndarray_pair_is_wide_uint32(q_view, t_view)) {
        const auto wide = ::stride_align::farrar_detail::prepare_farrar_alignment_wide_uint32(
            q_view, t_view, match_score, mismatch_score, gap_score, gap_score, width);
        return farrar_fixed_kernel::detail::wide_dispatch_global_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
    {
      namespace nv = ::stride_align::numpy_view;
      auto q_view = nv::try_acquire(query.ptr());
      auto t_view = nv::try_acquire(target.ptr());
      if (::stride_align::farrar_detail::ndarray_pair_is_wide_uint16(q_view, t_view)) {
        const auto wide = ::stride_align::farrar_detail::prepare_farrar_alignment_wide_uint16(
            q_view, t_view, match_score, mismatch_score, gap_score, gap_score, width);
        return farrar_fixed_kernel::detail::wide_dispatch_global_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
    // Unicode auto-promote: UCS-2 / UCS-4 strings with >256 distinct
    // codepoints route to the 16-bit Farrar kernel via uint16 tokens.
    // If the combined alphabet also exceeds 65 535 distinct codepoints
    // we fall through to the 32-bit Farrar kernel feeding raw UCS-4
    // codepoints as uint32 tokens (no tokenisation map).
    // UCS-1 strings (Latin-1) can have at most 256 distinct codepoints
    // by construction, so we skip the pre-scan entirely for them.
    if (PyUnicode_Check(query.ptr()) && PyUnicode_Check(target.ptr()) &&
        (PyUnicode_KIND(query.ptr()) != PyUnicode_1BYTE_KIND ||
         PyUnicode_KIND(target.ptr()) != PyUnicode_1BYTE_KIND)) {
      if (!::stride_align::farrar_detail::unicode_distinct_count_within(
              query.ptr(), target.ptr(), 256U)) {
        if (!::stride_align::farrar_detail::unicode_alphabet_within_uint16(
                query.ptr(), target.ptr())) {
          const auto wide =
              ::stride_align::farrar_detail::prepare_farrar_alignment_wide_unicode_uint32(
                  query.ptr(), target.ptr(),
                  match_score, mismatch_score, gap_score, gap_score, width);
          return farrar_fixed_kernel::detail::wide_dispatch_global_score<SimdOps>(
              wide, match_score, mismatch_score, gap_score);
        }
        const auto wide =
            ::stride_align::farrar_detail::prepare_farrar_alignment_wide_unicode_uint16(
                query.ptr(), target.ptr(),
                match_score, mismatch_score, gap_score, gap_score, width);
        return farrar_fixed_kernel::detail::wide_dispatch_global_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
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
    if (auto wide = farrar_fixed_kernel::detail::try_wide_score_batch<SimdOps, false>(
            query, targets, match_score, mismatch_score, gap_score, width);
        wide) {
      return std::move(*wide);
    }
    const auto prepared =
        prepare_farrar_batch_alignment(query, targets, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::dispatch_score_many<SimdOps, false>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static PreparedScoreBatch prepare_needleman_wunsch_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_farrar_batch_alignment(query, targets, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::prepare_score_batch<SimdOps, false>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static std::vector<Score> needleman_wunsch_scores_prepared(PreparedScoreBatch& prepared) {
    return farrar_fixed_kernel::detail::dispatch_prepared_score_many<SimdOps, false>(
        prepared);
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
    if (auto wide = farrar_fixed_kernel::detail::try_wide_affine_score<SimdOps, false>(
            query, target, match_score, mismatch_score,
            gap_open_score, gap_extend_score, width);
        wide) {
      return *wide;
    }
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
    if (auto wide = farrar_fixed_kernel::detail::try_wide_affine_score_batch<SimdOps, false>(
            query, targets, match_score, mismatch_score,
            gap_open_score, gap_extend_score, width);
        wide) {
      return std::move(*wide);
    }
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

  static PreparedAffineScoreBatch prepare_needleman_wunsch_affine_scores(
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
    return farrar_fixed_kernel::detail::prepare_affine_score_batch<SimdOps, false>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static std::vector<Score> needleman_wunsch_affine_scores_prepared(
      PreparedAffineScoreBatch& prepared) {
    return farrar_fixed_kernel::detail::dispatch_prepared_affine_score_many<SimdOps, false>(
        prepared);
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
    const Score expected_score = needleman_wunsch_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
    return profile_traceback::affine_path_info_with_score<false>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width,
        expected_score);
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

  static std::vector<Score> levenshtein_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    return ::stride_align::levenshtein_simd::levenshtein_scores_simd<
        ::stride_align::levenshtein_simd::Avx2Ops>(query, targets, cutoff);
  }

  static std::vector<double> levenshtein_normalized_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    return ::stride_align::levenshtein_simd::levenshtein_normalized_scores_simd<
        ::stride_align::levenshtein_simd::Avx2Ops>(query, targets, cutoff);
  }

  static std::vector<Score> damerau_levenshtein_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::osa_simd::osa_scores_simd<
        ::stride_align::levenshtein_simd::Avx2Ops>(query, targets);
  }

  static std::vector<double> damerau_levenshtein_normalized_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::osa_simd::osa_normalized_scores_simd<
        ::stride_align::levenshtein_simd::Avx2Ops>(query, targets);
  }

  static std::vector<Score> indel_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::indel_simd::indel_scores_simd<
        ::stride_align::levenshtein_simd::Avx2Ops>(query, targets);
  }

  static std::vector<double> indel_normalized_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::indel_simd::indel_normalized_scores_simd<
        ::stride_align::levenshtein_simd::Avx2Ops>(query, targets);
  }

  static std::vector<double> jaro_similarities(
      nb::handle query, nb::handle targets) {
    return ::stride_align::jaro_simd::jaro_similarities_simd<
        ::stride_align::levenshtein_simd::Avx2Ops>(query, targets);
  }

  static std::vector<double> jaro_winkler_similarities(
      nb::handle query,
      nb::handle targets,
      double prefix_weight,
      double prefix_threshold,
      std::size_t prefix_cap) {
    return ::stride_align::jaro_simd::jaro_winkler_similarities_simd<
        ::stride_align::levenshtein_simd::Avx2Ops>(
        query, targets, prefix_weight, prefix_threshold, prefix_cap);
  }

  static nb::object cdist(
      nb::handle queries, nb::handle targets, int scorer,
      nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    return ::stride_align::cdist_simd::cdist_impl<
        ::stride_align::levenshtein_simd::Avx2Ops>(
        queries, targets, scorer, tqdm_factory, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static nb::object cdist_above_threshold(
      nb::handle queries, nb::handle targets, int scorer,
      double threshold, nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    return ::stride_align::cdist_threshold::cdist_threshold_impl<
        ::stride_align::levenshtein_simd::Avx2Ops>(
        queries, targets, scorer, threshold, tqdm_factory, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static nb::object cdist_top_k(
      nb::handle queries, nb::handle targets, int scorer,
      std::size_t k, nb::object tqdm_factory, std::size_t cpu_count,
      bool reject_duplicates,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    return ::stride_align::cdist_topk::cdist_top_k_impl<
        ::stride_align::levenshtein_simd::Avx2Ops>(
        queries, targets, scorer, k, tqdm_factory, cpu_count,
        reject_duplicates,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  // ----- Matrix-mode entry points -------------------------------------
  // Forwards to the shared dispatch helpers in farrar_fixed_kernel.hpp
  // with this backend's SimdOps. See backend_avx512bwvl for full docs.
  static Score smith_waterman_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    return farrar_fixed_kernel::detail::matrix_score_dispatch_helper<SimdOps, true>(
        query_indices, target_indices, matrix_buffer, stride, gap_score);
  }

  static Score needleman_wunsch_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    return farrar_fixed_kernel::detail::matrix_score_dispatch_helper<SimdOps, false>(
        query_indices, target_indices, matrix_buffer, stride, gap_score);
  }

  static std::vector<Score> smith_waterman_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    return farrar_fixed_kernel::detail::matrix_scores_dispatch_helper<SimdOps, true>(
        query_indices, targets, matrix_buffer, stride, gap_score);
  }

  static std::vector<Score> needleman_wunsch_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    return farrar_fixed_kernel::detail::matrix_scores_dispatch_helper<SimdOps, false>(
        query_indices, targets, matrix_buffer, stride, gap_score);
  }

  static Score smith_waterman_affine_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    return farrar_fixed_kernel::detail::matrix_affine_score_dispatch_helper<SimdOps, true>(
        query_indices, target_indices, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static Score needleman_wunsch_affine_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    return farrar_fixed_kernel::detail::matrix_affine_score_dispatch_helper<SimdOps, false>(
        query_indices, target_indices, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static std::vector<Score> smith_waterman_affine_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    return farrar_fixed_kernel::detail::matrix_affine_scores_dispatch_helper<SimdOps, true>(
        query_indices, targets, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static std::vector<Score> needleman_wunsch_affine_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    return farrar_fixed_kernel::detail::matrix_affine_scores_dispatch_helper<SimdOps, false>(
        query_indices, targets, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }
};

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC pop_options
#endif

struct Implementation {
  using PreparedSmithWatermanFarrarScore =
      TargetImplementation::PreparedSmithWatermanFarrarScore;
  using PreparedAffineScore = TargetImplementation::PreparedAffineScore;
  using PreparedScoreBatch = TargetImplementation::PreparedScoreBatch;
  using PreparedAffineScoreBatch = TargetImplementation::PreparedAffineScoreBatch;

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

  static STRIDE_ALIGN_X86_BASELINE PreparedScoreBatch prepare_smith_waterman_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_smith_waterman_scores(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> smith_waterman_scores_prepared(
      PreparedScoreBatch& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_scores_prepared(prepared);
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

  static STRIDE_ALIGN_X86_BASELINE PreparedScoreBatch prepare_smith_waterman_farrar_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_smith_waterman_farrar_scores(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> smith_waterman_farrar_scores_prepared(
      PreparedScoreBatch& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_farrar_scores_prepared(prepared);
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

  static STRIDE_ALIGN_X86_BASELINE PreparedAffineScoreBatch
  prepare_smith_waterman_affine_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_smith_waterman_affine_scores(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> smith_waterman_affine_scores_prepared(
      PreparedAffineScoreBatch& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_scores_prepared(prepared);
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

  static STRIDE_ALIGN_X86_BASELINE PreparedAffineScoreBatch
  prepare_smith_waterman_affine_farrar_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_smith_waterman_affine_farrar_scores(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score>
  smith_waterman_affine_farrar_scores_prepared(PreparedAffineScoreBatch& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_farrar_scores_prepared(prepared);
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

  static STRIDE_ALIGN_X86_BASELINE PreparedScoreBatch prepare_needleman_wunsch_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_needleman_wunsch_scores(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> needleman_wunsch_scores_prepared(
      PreparedScoreBatch& prepared) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_scores_prepared(prepared);
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

  static STRIDE_ALIGN_X86_BASELINE PreparedAffineScoreBatch
  prepare_needleman_wunsch_affine_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_needleman_wunsch_affine_scores(
        query,
        targets,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> needleman_wunsch_affine_scores_prepared(
      PreparedAffineScoreBatch& prepared) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_scores_prepared(prepared);
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

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> levenshtein_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    ensure_supported();
    return TargetImplementation::levenshtein_scores(query, targets, cutoff);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<double> levenshtein_normalized_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    ensure_supported();
    return TargetImplementation::levenshtein_normalized_scores(query, targets, cutoff);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> damerau_levenshtein_scores(
      nb::handle query, nb::handle targets) {
    ensure_supported();
    return TargetImplementation::damerau_levenshtein_scores(query, targets);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<double>
  damerau_levenshtein_normalized_scores(nb::handle query, nb::handle targets) {
    ensure_supported();
    return TargetImplementation::damerau_levenshtein_normalized_scores(query, targets);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<double>
  jaro_similarities(nb::handle query, nb::handle targets) {
    ensure_supported();
    return TargetImplementation::jaro_similarities(query, targets);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<double>
  jaro_winkler_similarities(
      nb::handle query,
      nb::handle targets,
      double prefix_weight,
      double prefix_threshold,
      std::size_t prefix_cap) {
    ensure_supported();
    return TargetImplementation::jaro_winkler_similarities(
        query, targets, prefix_weight, prefix_threshold, prefix_cap);
  }

  static STRIDE_ALIGN_X86_BASELINE nb::object cdist(
      nb::handle queries, nb::handle targets, int scorer,
      nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    ensure_supported();
    return TargetImplementation::cdist(
        queries, targets, scorer, tqdm_factory, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static STRIDE_ALIGN_X86_BASELINE nb::object cdist_above_threshold(
      nb::handle queries, nb::handle targets, int scorer,
      double threshold, nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    ensure_supported();
    return TargetImplementation::cdist_above_threshold(
        queries, targets, scorer, threshold, tqdm_factory, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static STRIDE_ALIGN_X86_BASELINE nb::object cdist_top_k(
      nb::handle queries, nb::handle targets, int scorer,
      std::size_t k, nb::object tqdm_factory, std::size_t cpu_count,
      bool reject_duplicates,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    ensure_supported();
    return TargetImplementation::cdist_top_k(
        queries, targets, scorer, k, tqdm_factory, cpu_count,
        reject_duplicates,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  // ----- Matrix-mode entry points (public wrapper) --------------------
  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    ensure_supported();
    return TargetImplementation::smith_waterman_score_matrix(
        query_indices, target_indices, matrix_buffer, stride, gap_score);
  }

  static STRIDE_ALIGN_X86_BASELINE Score needleman_wunsch_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_score_matrix(
        query_indices, target_indices, matrix_buffer, stride, gap_score);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> smith_waterman_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    ensure_supported();
    return TargetImplementation::smith_waterman_scores_matrix(
        query_indices, targets, matrix_buffer, stride, gap_score);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> needleman_wunsch_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_scores_matrix(
        query_indices, targets, matrix_buffer, stride, gap_score);
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_affine_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_score_matrix(
        query_indices, target_indices, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static STRIDE_ALIGN_X86_BASELINE Score needleman_wunsch_affine_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_score_matrix(
        query_indices, target_indices, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> smith_waterman_affine_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_scores_matrix(
        query_indices, targets, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> needleman_wunsch_affine_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_scores_matrix(
        query_indices, targets, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static constexpr BackendKind backend_kind = BackendKind::x86_avx2;
};

#undef STRIDE_ALIGN_X86_BASELINE

}  // namespace stride_align::backend_avx2
