#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512vl,avx512dq")
#endif

#include "backends/affine_fixed_kernel.hpp"
#include "backends/farrar_fixed_kernel.hpp"
#include "backends/profile_traceback.hpp"
#include "backends/x86_fixed_kernel.hpp"
#include "byte_view.hpp"
#include "cdist_simd.hpp"
#include "farrar_preprocess.hpp"
#include "cdist_threshold.hpp"
#include "cdist_topk.hpp"
#include "dtw_dispatch.hpp"
#include "jaro_simd.hpp"
#include "levenshtein_simd.hpp"
#include "levenshtein_simd_ops.hpp"
#include "indel_simd.hpp"
#include "osa_simd.hpp"

namespace stride_align::backend_avx512bwvl {

namespace nb = nanobind;

#if defined(__GNUC__) || defined(__clang__)
#define STRIDE_ALIGN_X86_BASELINE __attribute__((target("default")))
#else
#define STRIDE_ALIGN_X86_BASELINE
#endif

template <int ShiftBytes>
inline __m512i shift_left_zero_512(__m512i vector) {
  static_assert(ShiftBytes == 1 || ShiftBytes == 2 || ShiftBytes == 4 || ShiftBytes == 8);
  const __m512i previous_lane_indices = _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 0);
  if constexpr (ShiftBytes == 8) {
    return _mm512_maskz_permutexvar_epi64(0xFE, previous_lane_indices, vector);
  } else {
    const __m512i shifted = _mm512_slli_epi64(vector, ShiftBytes * 8);
    const __m512i carry = _mm512_srli_epi64(vector, 64 - ShiftBytes * 8);
    const __m512i carry_from_previous_lane =
        _mm512_maskz_permutexvar_epi64(0xFE, previous_lane_indices, carry);
    return _mm512_or_si512(shifted, carry_from_previous_lane);
  }
}

template <int ByteShift>
inline __m512i shift_left_zero_bytes_512(__m512i vector) {
  static_assert(ByteShift == 1 || ByteShift == 2 || ByteShift == 4 ||
      ByteShift == 8 || ByteShift == 16 || ByteShift == 32);
  if constexpr (ByteShift <= 8) {
    return shift_left_zero_512<ByteShift>(vector);
  } else if constexpr (ByteShift == 16) {
    const __m512i indices = _mm512_set_epi64(5, 4, 3, 2, 1, 0, 0, 0);
    return _mm512_maskz_permutexvar_epi64(0xFC, indices, vector);
  } else {
    const __m512i indices = _mm512_set_epi64(3, 2, 1, 0, 0, 0, 0, 0);
    return _mm512_maskz_permutexvar_epi64(0xF0, indices, vector);
  }
}

template <int ByteCount>
inline constexpr __mmask64 first_bytes_mask_512() {
  static_assert(ByteCount >= 1 && ByteCount <= 32);
  return static_cast<__mmask64>((std::uint64_t{1} << ByteCount) - 1U);
}

template <int ByteShift>
inline __m512i shift_left_insert_bytes_512(__m512i vector, __m512i inserted) {
  return _mm512_mask_blend_epi8(
      first_bytes_mask_512<ByteShift>(),
      shift_left_zero_bytes_512<ByteShift>(vector),
      inserted);
}

template <typename Cell>
inline __m512i set1_cell_512(Cell value) {
  if constexpr (sizeof(Cell) == 1) {
    return _mm512_set1_epi8(static_cast<char>(value));
  } else if constexpr (sizeof(Cell) == 2) {
    return _mm512_set1_epi16(value);
  } else if constexpr (sizeof(Cell) == 4) {
    return _mm512_set1_epi32(value);
  } else {
    return _mm512_set1_epi64(value);
  }
}

template <typename Cell>
inline __m512i add_cell_512(__m512i lhs, __m512i rhs) {
  if constexpr (sizeof(Cell) == 1) {
    return _mm512_add_epi8(lhs, rhs);
  } else if constexpr (sizeof(Cell) == 2) {
    return _mm512_add_epi16(lhs, rhs);
  } else if constexpr (sizeof(Cell) == 4) {
    return _mm512_add_epi32(lhs, rhs);
  } else {
    return _mm512_add_epi64(lhs, rhs);
  }
}

template <typename Cell>
inline __m512i max_cell_512(__m512i lhs, __m512i rhs) {
  if constexpr (sizeof(Cell) == 1) {
    return _mm512_max_epi8(lhs, rhs);
  } else if constexpr (sizeof(Cell) == 2) {
    return _mm512_max_epi16(lhs, rhs);
  } else if constexpr (sizeof(Cell) == 4) {
    return _mm512_max_epi32(lhs, rhs);
  } else {
    return _mm512_max_epi64(lhs, rhs);
  }
}

template <typename Cell>
inline __m512i add_sentinel_cell_512(__m512i lhs, __m512i rhs, Cell sentinel) {
  const __m512i sum = add_cell_512<Cell>(lhs, rhs);
  const __m512i sentinel_vector = set1_cell_512(sentinel);
  if constexpr (sizeof(Cell) == 1) {
    const __mmask64 mask = _mm512_cmpeq_epi8_mask(lhs, sentinel_vector);
    return _mm512_mask_blend_epi8(mask, sum, sentinel_vector);
  } else if constexpr (sizeof(Cell) == 2) {
    const __mmask32 mask = _mm512_cmpeq_epi16_mask(lhs, sentinel_vector);
    return _mm512_mask_blend_epi16(mask, sum, sentinel_vector);
  } else if constexpr (sizeof(Cell) == 4) {
    const __mmask16 mask = _mm512_cmpeq_epi32_mask(lhs, sentinel_vector);
    return _mm512_mask_blend_epi32(mask, sum, sentinel_vector);
  } else {
    const __mmask8 mask = _mm512_cmpeq_epi64_mask(lhs, sentinel_vector);
    return _mm512_mask_blend_epi64(mask, sum, sentinel_vector);
  }
}

template <int LaneBytes, int LaneCount, typename Cell>
inline __m512i global_lazy_f_prefix_carry_512(
    __m512i final_f,
    std::size_t segment_count,
    Cell gap_extend_score,
    Cell low_score) {
  const Score lane_span_gap =
      static_cast<Score>(segment_count) * static_cast<Score>(gap_extend_score);
  const __m512i low_vector = set1_cell_512(low_score);
  __m512i prefix = shift_left_insert_bytes_512<LaneBytes>(final_f, low_vector);

  if constexpr (LaneCount > 1) {
    __m512i candidate = shift_left_insert_bytes_512<LaneBytes>(prefix, low_vector);
    candidate = add_sentinel_cell_512<Cell>(
        candidate,
        set1_cell_512(static_cast<Cell>(lane_span_gap)),
        low_score);
    prefix = max_cell_512<Cell>(prefix, candidate);
  }
  if constexpr (LaneCount > 2) {
    __m512i candidate = shift_left_insert_bytes_512<LaneBytes * 2>(prefix, low_vector);
    candidate = add_sentinel_cell_512<Cell>(
        candidate,
        set1_cell_512(static_cast<Cell>(lane_span_gap * 2)),
        low_score);
    prefix = max_cell_512<Cell>(prefix, candidate);
  }
  if constexpr (LaneCount > 4) {
    __m512i candidate = shift_left_insert_bytes_512<LaneBytes * 4>(prefix, low_vector);
    candidate = add_sentinel_cell_512<Cell>(
        candidate,
        set1_cell_512(static_cast<Cell>(lane_span_gap * 4)),
        low_score);
    prefix = max_cell_512<Cell>(prefix, candidate);
  }
  if constexpr (LaneCount > 8) {
    __m512i candidate = shift_left_insert_bytes_512<LaneBytes * 8>(prefix, low_vector);
    candidate = add_sentinel_cell_512<Cell>(
        candidate,
        set1_cell_512(static_cast<Cell>(lane_span_gap * 8)),
        low_score);
    prefix = max_cell_512<Cell>(prefix, candidate);
  }
  if constexpr (LaneCount > 16) {
    __m512i candidate = shift_left_insert_bytes_512<LaneBytes * 16>(prefix, low_vector);
    candidate = add_sentinel_cell_512<Cell>(
        candidate,
        set1_cell_512(static_cast<Cell>(lane_span_gap * 16)),
        low_score);
    prefix = max_cell_512<Cell>(prefix, candidate);
  }
  if constexpr (LaneCount > 32) {
    __m512i candidate = shift_left_insert_bytes_512<LaneBytes * 32>(prefix, low_vector);
    candidate = add_sentinel_cell_512<Cell>(
        candidate,
        set1_cell_512(static_cast<Cell>(lane_span_gap * 32)),
        low_score);
    prefix = max_cell_512<Cell>(prefix, candidate);
  }

  return prefix;
}

// Local-SW affine score exact-fill kernel for 1024-character queries striped
// across AVX-512 width32 lanes (16 lanes -> 64 segments). Mirrors the AVX2
// 128-segment kernel in x86_avx2.hpp: branchless H/E/profile DP body, log-step
// prefix carry for lazy-F, and a no-E correction loop. The correction loop
// updates H but intentionally does NOT update E, matching parasail's annotated
// affine SW correction shape (see docs/x86_algorithmic_deltas.txt section 51).
inline Score local_affine_score_exact_fill_i32_64(
    farrar_fixed_kernel::detail::PreparedAffineScoreState<std::int32_t>& state,
    std::span<const std::size_t> target_profile_offsets) {
  if (state.query_size != 64U * 16U || state.segment_count != 64U ||
      target_profile_offsets.empty() || state.gap_open_score > state.gap_extend_score ||
      state.gap_extend_score > 0) {
    return 0;
  }

  std::fill(state.h_store.begin(), state.h_store.end(), std::int32_t{0});
  std::fill(state.h_load.begin(), state.h_load.end(), std::int32_t{0});
  std::fill(state.e_store.begin(), state.e_store.end(), std::int32_t{0});

  const __m512i zero = _mm512_setzero_si512();
  const __m512i gap_open = _mm512_set1_epi32(state.gap_open_score);
  const __m512i gap_extend = _mm512_set1_epi32(state.gap_extend_score);
  const auto span_gap = static_cast<std::int32_t>(
      static_cast<Score>(64) * static_cast<Score>(state.gap_extend_score));
  const __m512i span_gap_1 = _mm512_set1_epi32(span_gap);
  const __m512i span_gap_2 = _mm512_set1_epi32(
      static_cast<std::int32_t>(static_cast<Score>(span_gap) * 2));
  const __m512i span_gap_4 = _mm512_set1_epi32(
      static_cast<std::int32_t>(static_cast<Score>(span_gap) * 4));
  const __m512i span_gap_8 = _mm512_set1_epi32(
      static_cast<std::int32_t>(static_cast<Score>(span_gap) * 8));
  __m512i best = zero;
  __m512i* h_store = reinterpret_cast<__m512i*>(state.h_store.data());
  __m512i* h_load = reinterpret_cast<__m512i*>(state.h_load.data());
  __m512i* e_store = reinterpret_cast<__m512i*>(state.e_store.data());
  const auto* profile_cells = state.profile.data();

  for (const auto profile_offset : target_profile_offsets) {
    std::swap(h_store, h_load);

    __m512i v_h = shift_left_zero_bytes_512<4>(_mm512_load_si512(h_load + 63));
    __m512i v_f = zero;
    const __m512i* profile_row =
        reinterpret_cast<const __m512i*>(profile_cells + profile_offset);

    for (std::size_t segment = 0; segment < 64U; ++segment) {
      const __m512i v_profile = _mm512_load_si512(profile_row + segment);
      __m512i v_e = _mm512_load_si512(e_store + segment);
      v_h = _mm512_add_epi32(v_h, v_profile);
      v_h = _mm512_max_epi32(v_h, v_e);
      v_h = _mm512_max_epi32(v_h, v_f);
      v_h = _mm512_max_epi32(v_h, zero);
      _mm512_store_si512(h_store + segment, v_h);
      best = _mm512_max_epi32(best, v_h);

      const __m512i v_h_open = _mm512_add_epi32(v_h, gap_open);
      v_e = _mm512_max_epi32(_mm512_add_epi32(v_e, gap_extend), v_h_open);
      _mm512_store_si512(e_store + segment, v_e);
      v_f = _mm512_max_epi32(_mm512_add_epi32(v_f, gap_extend), v_h_open);
      v_h = _mm512_load_si512(h_load + segment);
    }

    // Log-step prefix carry for 16 lanes: combine each lane with the best of
    // its earlier neighbors plus the cumulative cross-lane gap penalty.
    __m512i prefix = _mm512_max_epi32(v_f, zero);
    __m512i shifted = _mm512_add_epi32(
        shift_left_insert_bytes_512<4>(prefix, zero), span_gap_1);
    prefix = _mm512_max_epi32(prefix, shifted);
    shifted = _mm512_add_epi32(
        shift_left_insert_bytes_512<8>(prefix, zero), span_gap_2);
    prefix = _mm512_max_epi32(prefix, shifted);
    shifted = _mm512_add_epi32(
        shift_left_insert_bytes_512<16>(prefix, zero), span_gap_4);
    prefix = _mm512_max_epi32(prefix, shifted);
    shifted = _mm512_add_epi32(
        shift_left_insert_bytes_512<32>(prefix, zero), span_gap_8);
    prefix = _mm512_max_epi32(prefix, shifted);
    v_f = shift_left_insert_bytes_512<4>(_mm512_max_epi32(prefix, zero), zero);

    if (_mm512_cmpgt_epi32_mask(v_f, zero) != 0) {
      for (std::size_t segment = 0; segment < 64U; ++segment) {
        const __m512i v_h_previous = _mm512_load_si512(h_store + segment);
        const __m512i v_h_corrected = _mm512_max_epi32(v_h_previous, v_f);
        _mm512_store_si512(h_store + segment, v_h_corrected);
        best = _mm512_max_epi32(best, v_h_corrected);
        v_f = _mm512_add_epi32(v_f, gap_extend);
      }
    }
  }

  return static_cast<Score>(_mm512_reduce_max_epi32(best));
}

// Log-depth prefix carry for local linear SW lazy-F on 32-segment exact-fill
// (32 segments * 32 i16 lanes = 1024). Mirrors Ops::local_lazy_f_prefix_carry
// for SimdOps<uint16_t, int16_t>, specialized so the exact-fill kernel can
// keep constants in registers.
inline __m512i local_linear_sw_lazy_f_prefix_carry_i16_32_512(
    __m512i final_f,
    __m512i span_gap_1,
    __m512i span_gap_2,
    __m512i span_gap_4,
    __m512i span_gap_8,
    __m512i span_gap_16,
    __m512i zero) {
  __m512i prefix = _mm512_max_epi16(final_f, zero);
  __m512i shifted =
      _mm512_add_epi16(shift_left_insert_bytes_512<2>(prefix, zero), span_gap_1);
  prefix = _mm512_max_epi16(prefix, shifted);
  shifted =
      _mm512_add_epi16(shift_left_insert_bytes_512<4>(prefix, zero), span_gap_2);
  prefix = _mm512_max_epi16(prefix, shifted);
  shifted =
      _mm512_add_epi16(shift_left_insert_bytes_512<8>(prefix, zero), span_gap_4);
  prefix = _mm512_max_epi16(prefix, shifted);
  shifted =
      _mm512_add_epi16(shift_left_insert_bytes_512<16>(prefix, zero), span_gap_8);
  prefix = _mm512_max_epi16(prefix, shifted);
  shifted =
      _mm512_add_epi16(shift_left_insert_bytes_512<32>(prefix, zero), span_gap_16);
  prefix = _mm512_max_epi16(prefix, shifted);
  return shift_left_insert_bytes_512<2>(_mm512_max_epi16(prefix, zero), zero);
}

// Specialized exact-fill local SW for AVX-512 i16 / length-1024 (32 segments).
//
// Hot path is **deferred** lazy-F: fold the F wavefront into the next column's
// diagonal chain instead of a separate H rewrite. Optimized for AVX-512:
//   - 8-way unrolled main body (macro-inlined; no lambda / no 24-ZMM tiles —
//     those spilled and regressed)
//   - 2-wide software pipeline on loads: prefetch + load next segment's
//     profile/E/prev-H under the current segment's arithmetic
//   - 4 independent best accumulators (best is off the H critical path)
//   - dual-column soft pipeline: prefetch next column profile + H/E
//   - branchless pending inject (max(v_f, 0))
//
// Materialized uses a sound active-count scan (ceil(max_f/|gap|)) with mid-loop
// F<=0 exit — kept for explicit strategy selection / comparison.
inline Score local_sw_score_exact_fill_i16_32(
    farrar_fixed_kernel::detail::PreparedScoreState<std::int16_t>& state) {
  if (state.fast_score.has_value()) {
    return *state.fast_score;
  }
  if (state.gap_score > 0 || state.query_size != 32U * 32U || state.segment_count != 32U ||
      state.target_profile_offsets.empty()) {
    return 0;
  }

  std::fill(state.h_store.begin(), state.h_store.end(), std::int16_t{0});
  std::fill(state.h_load.begin(), state.h_load.end(), std::int16_t{0});
  std::fill(state.e_store.begin(), state.e_store.end(), std::int16_t{0});

  const __m512i zero = _mm512_setzero_si512();
  const __m512i gap = _mm512_set1_epi16(state.gap_score);
  const auto span = static_cast<std::int16_t>(
      static_cast<Score>(state.gap_score) * static_cast<Score>(32));
  const __m512i span_gap_1 = _mm512_set1_epi16(span);
  const __m512i span_gap_2 =
      _mm512_set1_epi16(static_cast<std::int16_t>(static_cast<Score>(span) * 2));
  const __m512i span_gap_4 =
      _mm512_set1_epi16(static_cast<std::int16_t>(static_cast<Score>(span) * 4));
  const __m512i span_gap_8 =
      _mm512_set1_epi16(static_cast<std::int16_t>(static_cast<Score>(span) * 8));
  const __m512i span_gap_16 =
      _mm512_set1_epi16(static_cast<std::int16_t>(static_cast<Score>(span) * 16));
  const __m512i segment_gap_31 = _mm512_set1_epi16(static_cast<std::int16_t>(
      static_cast<Score>(state.gap_score) * static_cast<Score>(31)));
  // Split best off the serial H chain: 4 independent max trees.
  __m512i best0 = zero;
  __m512i best1 = zero;
  __m512i best2 = zero;
  __m512i best3 = zero;
  __m512i pending_f = zero;
  __m512i* h_store = reinterpret_cast<__m512i*>(state.h_store.data());
  __m512i* h_load = reinterpret_cast<__m512i*>(state.h_load.data());
  __m512i* e_store = reinterpret_cast<__m512i*>(state.e_store.data());
  const auto* profile_cells = state.profile.data();
  // Prefer deferred on this specialized path (including automatic).
  const bool use_deferred_correction =
      state.kernel_strategy != farrar_fixed_kernel::detail::ScoreKernelStrategy::materialized &&
      state.kernel_strategy != farrar_fixed_kernel::detail::ScoreKernelStrategy::compact &&
      state.kernel_strategy != farrar_fixed_kernel::detail::ScoreKernelStrategy::bounded &&
      state.kernel_strategy != farrar_fixed_kernel::detail::ScoreKernelStrategy::bounded_unroll4;

#define STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP(segment_index) \
  do { \
    const __m512i v_profile = _mm512_load_si512(profile_row + (segment_index)); \
    __m512i v_e = _mm512_load_si512(e_store + (segment_index)); \
    v_h = _mm512_add_epi16(v_h, v_profile); \
    v_h = _mm512_max_epi16(v_h, v_e); \
    v_h = _mm512_max_epi16(v_h, v_f); \
    v_h = _mm512_max_epi16(v_h, zero); \
    _mm512_store_si512(h_store + (segment_index), v_h); \
    best0 = _mm512_max_epi16(best0, v_h); \
    const __m512i v_h_gap = _mm512_add_epi16(v_h, gap); \
    v_e = _mm512_max_epi16(_mm512_add_epi16(v_e, gap), v_h_gap); \
    _mm512_store_si512(e_store + (segment_index), v_e); \
    v_f = _mm512_max_epi16(_mm512_add_epi16(v_f, gap), v_h_gap); \
    v_h = _mm512_load_si512(h_load + (segment_index)); \
  } while (false)

  // Deferred step with software-pipelined loads: profile/E/prev-H for
  // this segment are already in registers; next segment's loads are
  // issued under this segment's arithmetic (hides L1 latency).
#define STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED( \
    segment_index, v_profile, v_e_in, v_h_prev, best_acc, \
    next_profile, next_e, next_h_prev) \
  do { \
    __m512i v_e = (v_e_in); \
    v_h = _mm512_add_epi16(v_h, (v_profile)); \
    v_h = _mm512_max_epi16(v_h, v_e); \
    v_h = _mm512_max_epi16(v_h, v_f); \
    v_h = _mm512_max_epi16(v_h, zero); \
    _mm512_store_si512(h_store + (segment_index), v_h); \
    (best_acc) = _mm512_max_epi16((best_acc), v_h); \
    const __m512i v_h_gap = _mm512_add_epi16(v_h, gap); \
    v_e = _mm512_max_epi16(_mm512_add_epi16(v_e, gap), v_h_gap); \
    _mm512_store_si512(e_store + (segment_index), v_e); \
    v_f = _mm512_max_epi16(_mm512_add_epi16(v_f, gap), v_h_gap); \
    /* Issue next segment loads under the store latency. */ \
    (next_profile) = _mm512_load_si512(profile_row + (segment_index) + 1U); \
    (next_e) = _mm512_load_si512(e_store + (segment_index) + 1U); \
    (next_h_prev) = _mm512_load_si512(h_load + (segment_index) + 1U); \
    v_h = _mm512_max_epi16((v_h_prev), pending_h); \
    pending_h = _mm512_add_epi16(pending_h, gap); \
  } while (false)

#define STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED_TAIL( \
    segment_index, v_profile, v_e_in, v_h_prev, best_acc) \
  do { \
    __m512i v_e = (v_e_in); \
    v_h = _mm512_add_epi16(v_h, (v_profile)); \
    v_h = _mm512_max_epi16(v_h, v_e); \
    v_h = _mm512_max_epi16(v_h, v_f); \
    v_h = _mm512_max_epi16(v_h, zero); \
    _mm512_store_si512(h_store + (segment_index), v_h); \
    (best_acc) = _mm512_max_epi16((best_acc), v_h); \
    const __m512i v_h_gap = _mm512_add_epi16(v_h, gap); \
    v_e = _mm512_max_epi16(_mm512_add_epi16(v_e, gap), v_h_gap); \
    _mm512_store_si512(e_store + (segment_index), v_e); \
    v_f = _mm512_max_epi16(_mm512_add_epi16(v_f, gap), v_h_gap); \
    v_h = _mm512_max_epi16((v_h_prev), pending_h); \
    pending_h = _mm512_add_epi16(pending_h, gap); \
  } while (false)

#define STRIDE_ALIGN_AVX512_LOCAL_SW_I16_SCAN(segment_index) \
  do { \
    __m512i v_h_scan = _mm512_max_epi16(_mm512_load_si512(h_store + (segment_index)), v_f); \
    _mm512_store_si512(h_store + (segment_index), v_h_scan); \
    best0 = _mm512_max_epi16(best0, v_h_scan); \
    v_f = _mm512_add_epi16(v_f, gap); \
  } while (false)

  auto reduce_max_i16_512 = [](__m512i value) -> Score {
    value = _mm512_max_epi16(value, _mm512_shuffle_i32x4(value, value, 0x4E));
    value = _mm512_max_epi16(value, _mm512_shuffle_i32x4(value, value, 0xB1));
    __m256i v256 = _mm512_castsi512_si256(value);
    v256 = _mm256_max_epi16(v256, _mm256_permute4x64_epi64(v256, 0x4E));
    __m128i v128 = _mm256_castsi256_si128(v256);
    v128 = _mm_max_epi16(v128, _mm_srli_si128(v128, 8));
    v128 = _mm_max_epi16(v128, _mm_srli_si128(v128, 4));
    v128 = _mm_max_epi16(v128, _mm_srli_si128(v128, 2));
    return static_cast<Score>(static_cast<std::int16_t>(_mm_extract_epi16(v128, 0)));
  };

  if (!use_deferred_correction) {
    for (const auto profile_offset : state.target_profile_offsets) {
      std::swap(h_store, h_load);

      __m512i v_h = shift_left_zero_512<2>(_mm512_load_si512(h_load + 31));
      __m512i v_f = zero;
      const __m512i* profile_row =
          reinterpret_cast<const __m512i*>(profile_cells + profile_offset);

      for (std::size_t segment = 0; segment < 32U; segment += 8U) {
        STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP(segment);
        STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP(segment + 1U);
        STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP(segment + 2U);
        STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP(segment + 3U);
        STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP(segment + 4U);
        STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP(segment + 5U);
        STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP(segment + 6U);
        STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP(segment + 7U);
      }

      v_f = local_linear_sw_lazy_f_prefix_carry_i16_32_512(
          v_f,
          span_gap_1,
          span_gap_2,
          span_gap_4,
          span_gap_8,
          span_gap_16,
          zero);
      if (_mm512_cmpgt_epi16_mask(v_f, zero) != 0) {
        std::size_t active = 32U;
        if (state.gap_score < 0) {
          __m512i t = v_f;
          t = _mm512_max_epi16(t, _mm512_shuffle_i32x4(t, t, 0x4E));
          t = _mm512_max_epi16(t, _mm512_shuffle_i32x4(t, t, 0xB1));
          __m256i t256 = _mm512_castsi512_si256(t);
          t256 = _mm256_max_epi16(t256, _mm256_permute4x64_epi64(t256, 0x4E));
          __m128i t128 = _mm256_castsi256_si128(t256);
          t128 = _mm_max_epi16(t128, _mm_srli_si128(t128, 8));
          t128 = _mm_max_epi16(t128, _mm_srli_si128(t128, 4));
          t128 = _mm_max_epi16(t128, _mm_srli_si128(t128, 2));
          const auto mf = static_cast<std::int16_t>(_mm_extract_epi16(t128, 0));
          if (mf <= 0) {
            active = 0;
          } else {
            const std::int64_t gap_mag = -static_cast<std::int64_t>(state.gap_score);
            const std::size_t needed = static_cast<std::size_t>(
                (static_cast<std::int64_t>(mf) + gap_mag - 1) / gap_mag);
            if (needed < active) {
              active = needed;
            }
          }
        }
        for (std::size_t segment = 0; segment < active; ++segment) {
          STRIDE_ALIGN_AVX512_LOCAL_SW_I16_SCAN(segment);
          if (_mm512_cmpgt_epi16_mask(v_f, zero) == 0) {
            break;
          }
        }
      }
    }
  } else {
    const auto& profile_offsets = state.target_profile_offsets;
    const std::size_t n_cols = profile_offsets.size();
    for (std::size_t col = 0; col < n_cols; ++col) {
      const auto profile_offset = profile_offsets[col];
      const __m512i* profile_row =
          reinterpret_cast<const __m512i*>(profile_cells + profile_offset);

      // Dual-column soft pipeline: pull next column while this one runs.
      if (col + 1U < n_cols) {
        const auto* next_prof = profile_cells + profile_offsets[col + 1U];
        _mm_prefetch(reinterpret_cast<const char*>(next_prof), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(next_prof) + 1024, _MM_HINT_T0);
        // Current h_store becomes next h_load after swap.
        _mm_prefetch(reinterpret_cast<const char*>(h_store), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(h_store) + 1024, _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(e_store), _MM_HINT_T0);
      }

      std::swap(h_store, h_load);

      __m512i v_f = zero;
      __m512i pending_h = pending_f;
      __m512i last_h = _mm512_load_si512(h_load + 31);
      last_h = _mm512_max_epi16(
          last_h, _mm512_add_epi16(pending_f, segment_gap_31));
      __m512i v_h = shift_left_zero_512<2>(last_h);

      // Prime the software pipeline with segment 0 loads.
      __m512i p_cur = _mm512_load_si512(profile_row + 0);
      __m512i e_cur = _mm512_load_si512(e_store + 0);
      __m512i hp_cur = _mm512_load_si512(h_load + 0);
      __m512i p_next, e_next, hp_next;

      // Segments 0..30: each step loads segment+1 under its arithmetic.
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          0, p_cur, e_cur, hp_cur, best0, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          1, p_cur, e_cur, hp_cur, best1, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          2, p_cur, e_cur, hp_cur, best2, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          3, p_cur, e_cur, hp_cur, best3, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          4, p_cur, e_cur, hp_cur, best0, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          5, p_cur, e_cur, hp_cur, best1, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          6, p_cur, e_cur, hp_cur, best2, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          7, p_cur, e_cur, hp_cur, best3, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          8, p_cur, e_cur, hp_cur, best0, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          9, p_cur, e_cur, hp_cur, best1, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          10, p_cur, e_cur, hp_cur, best2, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          11, p_cur, e_cur, hp_cur, best3, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          12, p_cur, e_cur, hp_cur, best0, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          13, p_cur, e_cur, hp_cur, best1, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          14, p_cur, e_cur, hp_cur, best2, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          15, p_cur, e_cur, hp_cur, best3, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          16, p_cur, e_cur, hp_cur, best0, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          17, p_cur, e_cur, hp_cur, best1, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          18, p_cur, e_cur, hp_cur, best2, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          19, p_cur, e_cur, hp_cur, best3, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          20, p_cur, e_cur, hp_cur, best0, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          21, p_cur, e_cur, hp_cur, best1, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          22, p_cur, e_cur, hp_cur, best2, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          23, p_cur, e_cur, hp_cur, best3, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          24, p_cur, e_cur, hp_cur, best0, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          25, p_cur, e_cur, hp_cur, best1, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          26, p_cur, e_cur, hp_cur, best2, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          27, p_cur, e_cur, hp_cur, best3, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          28, p_cur, e_cur, hp_cur, best0, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          29, p_cur, e_cur, hp_cur, best1, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED(
          30, p_cur, e_cur, hp_cur, best2, p_next, e_next, hp_next);
      p_cur = p_next; e_cur = e_next; hp_cur = hp_next;
      // Segment 31: no next-segment load.
      STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED_TAIL(
          31, p_cur, e_cur, hp_cur, best3);

      v_f = local_linear_sw_lazy_f_prefix_carry_i16_32_512(
          v_f,
          span_gap_1,
          span_gap_2,
          span_gap_4,
          span_gap_8,
          span_gap_16,
          zero);
      pending_f = _mm512_max_epi16(v_f, zero);
      best0 = _mm512_max_epi16(best0, pending_f);
    }
  }

#undef STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP
#undef STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED
#undef STRIDE_ALIGN_AVX512_LOCAL_SW_I16_STEP_DEFERRED_TAIL
#undef STRIDE_ALIGN_AVX512_LOCAL_SW_I16_SCAN

  __m512i best = _mm512_max_epi16(_mm512_max_epi16(best0, best1),
                                  _mm512_max_epi16(best2, best3));
  return reduce_max_i16_512(best);
}

// Shared reduce for dual-target kernels.
inline Score reduce_max_i16_512_local(__m512i value) {
  value = _mm512_max_epi16(value, _mm512_shuffle_i32x4(value, value, 0x4E));
  value = _mm512_max_epi16(value, _mm512_shuffle_i32x4(value, value, 0xB1));
  __m256i v256 = _mm512_castsi512_si256(value);
  v256 = _mm256_max_epi16(v256, _mm256_permute4x64_epi64(v256, 0x4E));
  __m128i v128 = _mm256_castsi256_si128(v256);
  v128 = _mm_max_epi16(v128, _mm_srli_si128(v128, 8));
  v128 = _mm_max_epi16(v128, _mm_srli_si128(v128, 4));
  v128 = _mm_max_epi16(v128, _mm_srli_si128(v128, 2));
  return static_cast<Score>(static_cast<std::int16_t>(_mm_extract_epi16(v128, 0)));
}

inline void zero_sw_i16_32_buffers(
    __m512i* h_store0,
    __m512i* h_load0,
    __m512i* e_store0,
    __m512i* h_store1,
    __m512i* h_load1,
    __m512i* e_store1) {
  // 32 segments × 64 B × 6 arrays = 12 KiB; memset is fine and keeps the
  // dual prolog off the hot column loop.
  std::memset(h_store0, 0, 32U * 64U);
  std::memset(h_load0, 0, 32U * 64U);
  std::memset(e_store0, 0, 32U * 64U);
  std::memset(h_store1, 0, 32U * 64U);
  std::memset(h_load1, 0, 32U * 64U);
  std::memset(e_store1, 0, 32U * 64U);
}

// Equal-length dual-target hot path: no unequal-finish branches, one-sided
// soft-pipeline on target 0 (hides L1 latency without dual-pipeline ZMM
// pressure), interleaved target-1 simple steps for ILP.
inline std::pair<Score, Score> local_sw_score_exact_fill_i16_32_dual_equal(
    const std::int16_t* profile_cells,
    std::span<const std::size_t> offsets0,
    std::span<const std::size_t> offsets1,
    std::int16_t gap_score,
    __m512i* h_store0,
    __m512i* h_load0,
    __m512i* e_store0,
    __m512i* h_store1,
    __m512i* h_load1,
    __m512i* e_store1) {
  const std::size_t n_cols = offsets0.size();
  // Caller guarantees offsets0.size() == offsets1.size().

  const __m512i zero = _mm512_setzero_si512();
  const __m512i gap = _mm512_set1_epi16(gap_score);
  const auto span = static_cast<std::int16_t>(
      static_cast<Score>(gap_score) * static_cast<Score>(32));
  const __m512i span_gap_1 = _mm512_set1_epi16(span);
  const __m512i span_gap_2 =
      _mm512_set1_epi16(static_cast<std::int16_t>(static_cast<Score>(span) * 2));
  const __m512i span_gap_4 =
      _mm512_set1_epi16(static_cast<std::int16_t>(static_cast<Score>(span) * 4));
  const __m512i span_gap_8 =
      _mm512_set1_epi16(static_cast<std::int16_t>(static_cast<Score>(span) * 8));
  const __m512i span_gap_16 =
      _mm512_set1_epi16(static_cast<std::int16_t>(static_cast<Score>(span) * 16));
  const __m512i segment_gap_31 = _mm512_set1_epi16(static_cast<std::int16_t>(
      static_cast<Score>(gap_score) * static_cast<Score>(31)));

  zero_sw_i16_32_buffers(h_store0, h_load0, e_store0, h_store1, h_load1, e_store1);

  __m512i best0 = zero;
  __m512i best1 = zero;
  __m512i pending_f0 = zero;
  __m512i pending_f1 = zero;

  // Target 0: soft-pipelined deferred step (loads for segment+1 issued under
  // this segment's arithmetic — same shape as single-target hot path).
#define STRIDE_ALIGN_AVX512_DUAL_EQ_T0_STEP( \
    segment_index, profile_row, h_store, h_load, e_store, \
    v_h, v_f, pending_h, best_acc, \
    v_profile, v_e_in, v_h_prev, next_profile, next_e, next_h_prev) \
  do { \
    __m512i v_e = (v_e_in); \
    (v_h) = _mm512_add_epi16((v_h), (v_profile)); \
    (v_h) = _mm512_max_epi16((v_h), v_e); \
    (v_h) = _mm512_max_epi16((v_h), (v_f)); \
    (v_h) = _mm512_max_epi16((v_h), zero); \
    _mm512_store_si512((h_store) + (segment_index), (v_h)); \
    (best_acc) = _mm512_max_epi16((best_acc), (v_h)); \
    const __m512i v_h_gap = _mm512_add_epi16((v_h), gap); \
    v_e = _mm512_max_epi16(_mm512_add_epi16(v_e, gap), v_h_gap); \
    _mm512_store_si512((e_store) + (segment_index), v_e); \
    (v_f) = _mm512_max_epi16(_mm512_add_epi16((v_f), gap), v_h_gap); \
    (next_profile) = _mm512_load_si512((profile_row) + (segment_index) + 1U); \
    (next_e) = _mm512_load_si512((e_store) + (segment_index) + 1U); \
    (next_h_prev) = _mm512_load_si512((h_load) + (segment_index) + 1U); \
    (v_h) = _mm512_max_epi16((v_h_prev), (pending_h)); \
    (pending_h) = _mm512_add_epi16((pending_h), gap); \
  } while (false)

#define STRIDE_ALIGN_AVX512_DUAL_EQ_T0_TAIL( \
    segment_index, h_store, e_store, \
    v_h, v_f, pending_h, best_acc, v_profile, v_e_in, v_h_prev) \
  do { \
    __m512i v_e = (v_e_in); \
    (v_h) = _mm512_add_epi16((v_h), (v_profile)); \
    (v_h) = _mm512_max_epi16((v_h), v_e); \
    (v_h) = _mm512_max_epi16((v_h), (v_f)); \
    (v_h) = _mm512_max_epi16((v_h), zero); \
    _mm512_store_si512((h_store) + (segment_index), (v_h)); \
    (best_acc) = _mm512_max_epi16((best_acc), (v_h)); \
    const __m512i v_h_gap = _mm512_add_epi16((v_h), gap); \
    v_e = _mm512_max_epi16(_mm512_add_epi16(v_e, gap), v_h_gap); \
    _mm512_store_si512((e_store) + (segment_index), v_e); \
    (v_f) = _mm512_max_epi16(_mm512_add_epi16((v_f), gap), v_h_gap); \
    (v_h) = _mm512_max_epi16((v_h_prev), (pending_h)); \
    (pending_h) = _mm512_add_epi16((pending_h), gap); \
  } while (false)

  // Target 1: simple deferred step (no next-segment pipeline — keeps ZMM
  // budget under control while still feeding a second independent H chain).
#define STRIDE_ALIGN_AVX512_DUAL_EQ_T1_STEP( \
    segment_index, profile_row, h_store, h_load, e_store, \
    v_h, v_f, pending_h, best_acc) \
  do { \
    const __m512i v_profile = _mm512_load_si512((profile_row) + (segment_index)); \
    __m512i v_e = _mm512_load_si512((e_store) + (segment_index)); \
    (v_h) = _mm512_add_epi16((v_h), v_profile); \
    (v_h) = _mm512_max_epi16((v_h), v_e); \
    (v_h) = _mm512_max_epi16((v_h), (v_f)); \
    (v_h) = _mm512_max_epi16((v_h), zero); \
    _mm512_store_si512((h_store) + (segment_index), (v_h)); \
    (best_acc) = _mm512_max_epi16((best_acc), (v_h)); \
    const __m512i v_h_gap = _mm512_add_epi16((v_h), gap); \
    v_e = _mm512_max_epi16(_mm512_add_epi16(v_e, gap), v_h_gap); \
    _mm512_store_si512((e_store) + (segment_index), v_e); \
    (v_f) = _mm512_max_epi16(_mm512_add_epi16((v_f), gap), v_h_gap); \
    const __m512i v_h_prev = _mm512_load_si512((h_load) + (segment_index)); \
    (v_h) = _mm512_max_epi16(v_h_prev, (pending_h)); \
    (pending_h) = _mm512_add_epi16((pending_h), gap); \
  } while (false)

  for (std::size_t col = 0; col < n_cols; ++col) {
    if (col + 1U < n_cols) {
      const auto* next0 = profile_cells + offsets0[col + 1U];
      const auto* next1 = profile_cells + offsets1[col + 1U];
      _mm_prefetch(reinterpret_cast<const char*>(next0), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(next0) + 1024, _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(next1), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(next1) + 1024, _MM_HINT_T0);
      // After swap, current h_store becomes next column's h_load.
      _mm_prefetch(reinterpret_cast<const char*>(h_store0), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(h_store1), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(e_store0), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(e_store1), _MM_HINT_T0);
    }

    std::swap(h_store0, h_load0);
    std::swap(h_store1, h_load1);

    const __m512i* profile_row0 =
        reinterpret_cast<const __m512i*>(profile_cells + offsets0[col]);
    const __m512i* profile_row1 =
        reinterpret_cast<const __m512i*>(profile_cells + offsets1[col]);

    __m512i v_f0 = zero;
    __m512i v_f1 = zero;
    __m512i pending_h0 = pending_f0;
    __m512i pending_h1 = pending_f1;

    __m512i last_h0 = _mm512_load_si512(h_load0 + 31);
    last_h0 = _mm512_max_epi16(
        last_h0, _mm512_add_epi16(pending_f0, segment_gap_31));
    __m512i v_h0 = shift_left_zero_512<2>(last_h0);

    __m512i last_h1 = _mm512_load_si512(h_load1 + 31);
    last_h1 = _mm512_max_epi16(
        last_h1, _mm512_add_epi16(pending_f1, segment_gap_31));
    __m512i v_h1 = shift_left_zero_512<2>(last_h1);

    // Prime target-0 software pipeline.
    __m512i p0_cur = _mm512_load_si512(profile_row0 + 0);
    __m512i e0_cur = _mm512_load_si512(e_store0 + 0);
    __m512i hp0_cur = _mm512_load_si512(h_load0 + 0);
    __m512i p0_next, e0_next, hp0_next;

    // Fully unrolled 0..30 (t0 soft-pipe + t1 simple interleaved), then
    // segment 31 tails. Matches the single-target style that won on avx10.
#define STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(seg) \
  do { \
    STRIDE_ALIGN_AVX512_DUAL_EQ_T0_STEP( \
        (seg), profile_row0, h_store0, h_load0, e_store0, \
        v_h0, v_f0, pending_h0, best0, \
        p0_cur, e0_cur, hp0_cur, p0_next, e0_next, hp0_next); \
    p0_cur = p0_next; \
    e0_cur = e0_next; \
    hp0_cur = hp0_next; \
    STRIDE_ALIGN_AVX512_DUAL_EQ_T1_STEP( \
        (seg), profile_row1, h_store1, h_load1, e_store1, \
        v_h1, v_f1, pending_h1, best1); \
  } while (false)

    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(0);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(1);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(2);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(3);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(4);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(5);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(6);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(7);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(8);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(9);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(10);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(11);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(12);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(13);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(14);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(15);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(16);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(17);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(18);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(19);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(20);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(21);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(22);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(23);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(24);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(25);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(26);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(27);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(28);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(29);
    STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR(30);
#undef STRIDE_ALIGN_AVX512_DUAL_EQ_PAIR

    STRIDE_ALIGN_AVX512_DUAL_EQ_T0_TAIL(
        31, h_store0, e_store0, v_h0, v_f0, pending_h0, best0, p0_cur, e0_cur, hp0_cur);
    STRIDE_ALIGN_AVX512_DUAL_EQ_T1_STEP(
        31, profile_row1, h_store1, h_load1, e_store1, v_h1, v_f1, pending_h1, best1);

    v_f0 = local_linear_sw_lazy_f_prefix_carry_i16_32_512(
        v_f0, span_gap_1, span_gap_2, span_gap_4, span_gap_8, span_gap_16, zero);
    v_f1 = local_linear_sw_lazy_f_prefix_carry_i16_32_512(
        v_f1, span_gap_1, span_gap_2, span_gap_4, span_gap_8, span_gap_16, zero);
    pending_f0 = _mm512_max_epi16(v_f0, zero);
    pending_f1 = _mm512_max_epi16(v_f1, zero);
    best0 = _mm512_max_epi16(best0, pending_f0);
    best1 = _mm512_max_epi16(best1, pending_f1);
  }

#undef STRIDE_ALIGN_AVX512_DUAL_EQ_T0_STEP
#undef STRIDE_ALIGN_AVX512_DUAL_EQ_T0_TAIL
#undef STRIDE_ALIGN_AVX512_DUAL_EQ_T1_STEP

  return {reduce_max_i16_512_local(best0), reduce_max_i16_512_local(best1)};
}

// Dual-target deferred exact-fill: score two targets in lockstep with
// independent H/E/pending chains. Equal lengths take the specialized path
// above; unequal lengths dual-process min(n0,n1) then finish the longer
// target alone.
inline std::pair<Score, Score> local_sw_score_exact_fill_i16_32_dual(
    const std::int16_t* profile_cells,
    std::span<const std::size_t> offsets0,
    std::span<const std::size_t> offsets1,
    std::int16_t gap_score,
    __m512i* h_store0,
    __m512i* h_load0,
    __m512i* e_store0,
    __m512i* h_store1,
    __m512i* h_load1,
    __m512i* e_store1) {
  const std::size_t n0 = offsets0.size();
  const std::size_t n1 = offsets1.size();
  if (n0 == n1) {
    return local_sw_score_exact_fill_i16_32_dual_equal(
        profile_cells,
        offsets0,
        offsets1,
        gap_score,
        h_store0,
        h_load0,
        e_store0,
        h_store1,
        h_load1,
        e_store1);
  }

  const __m512i zero = _mm512_setzero_si512();
  const __m512i gap = _mm512_set1_epi16(gap_score);
  const auto span = static_cast<std::int16_t>(
      static_cast<Score>(gap_score) * static_cast<Score>(32));
  const __m512i span_gap_1 = _mm512_set1_epi16(span);
  const __m512i span_gap_2 =
      _mm512_set1_epi16(static_cast<std::int16_t>(static_cast<Score>(span) * 2));
  const __m512i span_gap_4 =
      _mm512_set1_epi16(static_cast<std::int16_t>(static_cast<Score>(span) * 4));
  const __m512i span_gap_8 =
      _mm512_set1_epi16(static_cast<std::int16_t>(static_cast<Score>(span) * 8));
  const __m512i span_gap_16 =
      _mm512_set1_epi16(static_cast<std::int16_t>(static_cast<Score>(span) * 16));
  const __m512i segment_gap_31 = _mm512_set1_epi16(static_cast<std::int16_t>(
      static_cast<Score>(gap_score) * static_cast<Score>(31)));

  zero_sw_i16_32_buffers(h_store0, h_load0, e_store0, h_store1, h_load1, e_store1);

  __m512i best0 = zero;
  __m512i best1 = zero;
  __m512i pending_f0 = zero;
  __m512i pending_f1 = zero;

#define STRIDE_ALIGN_AVX512_DUAL_SW_I16_STEP( \
    segment_index, profile_row, h_store, h_load, e_store, \
    v_h, v_f, pending_h, best_acc) \
  do { \
    const __m512i v_profile = _mm512_load_si512((profile_row) + (segment_index)); \
    __m512i v_e = _mm512_load_si512((e_store) + (segment_index)); \
    (v_h) = _mm512_add_epi16((v_h), v_profile); \
    (v_h) = _mm512_max_epi16((v_h), v_e); \
    (v_h) = _mm512_max_epi16((v_h), (v_f)); \
    (v_h) = _mm512_max_epi16((v_h), zero); \
    _mm512_store_si512((h_store) + (segment_index), (v_h)); \
    (best_acc) = _mm512_max_epi16((best_acc), (v_h)); \
    const __m512i v_h_gap = _mm512_add_epi16((v_h), gap); \
    v_e = _mm512_max_epi16(_mm512_add_epi16(v_e, gap), v_h_gap); \
    _mm512_store_si512((e_store) + (segment_index), v_e); \
    (v_f) = _mm512_max_epi16(_mm512_add_epi16((v_f), gap), v_h_gap); \
    const __m512i v_h_prev = _mm512_load_si512((h_load) + (segment_index)); \
    (v_h) = _mm512_max_epi16(v_h_prev, (pending_h)); \
    (pending_h) = _mm512_add_epi16((pending_h), gap); \
  } while (false)

  auto run_deferred_column =
      [&](__m512i*& h_store,
          __m512i*& h_load,
          __m512i* e_store,
          const std::size_t profile_offset,
          __m512i& pending_f,
          __m512i& best) {
        std::swap(h_store, h_load);
        const __m512i* profile_row =
            reinterpret_cast<const __m512i*>(profile_cells + profile_offset);
        __m512i v_f = zero;
        __m512i pending_h = pending_f;
        __m512i last_h = _mm512_load_si512(h_load + 31);
        last_h = _mm512_max_epi16(
            last_h, _mm512_add_epi16(pending_f, segment_gap_31));
        __m512i v_h = shift_left_zero_512<2>(last_h);

        for (std::size_t segment = 0; segment < 32U; segment += 4U) {
          STRIDE_ALIGN_AVX512_DUAL_SW_I16_STEP(
              segment, profile_row, h_store, h_load, e_store, v_h, v_f, pending_h, best);
          STRIDE_ALIGN_AVX512_DUAL_SW_I16_STEP(
              segment + 1U, profile_row, h_store, h_load, e_store, v_h, v_f, pending_h, best);
          STRIDE_ALIGN_AVX512_DUAL_SW_I16_STEP(
              segment + 2U, profile_row, h_store, h_load, e_store, v_h, v_f, pending_h, best);
          STRIDE_ALIGN_AVX512_DUAL_SW_I16_STEP(
              segment + 3U, profile_row, h_store, h_load, e_store, v_h, v_f, pending_h, best);
        }

        v_f = local_linear_sw_lazy_f_prefix_carry_i16_32_512(
            v_f, span_gap_1, span_gap_2, span_gap_4, span_gap_8, span_gap_16, zero);
        pending_f = _mm512_max_epi16(v_f, zero);
        best = _mm512_max_epi16(best, pending_f);
      };

  const std::size_t n_pair = n0 < n1 ? n0 : n1;

  for (std::size_t col = 0; col < n_pair; ++col) {
    if (col + 1U < n_pair) {
      _mm_prefetch(
          reinterpret_cast<const char*>(profile_cells + offsets0[col + 1U]),
          _MM_HINT_T0);
      _mm_prefetch(
          reinterpret_cast<const char*>(profile_cells + offsets1[col + 1U]),
          _MM_HINT_T0);
    }

    std::swap(h_store0, h_load0);
    std::swap(h_store1, h_load1);

    const __m512i* profile_row0 =
        reinterpret_cast<const __m512i*>(profile_cells + offsets0[col]);
    const __m512i* profile_row1 =
        reinterpret_cast<const __m512i*>(profile_cells + offsets1[col]);

    __m512i v_f0 = zero;
    __m512i v_f1 = zero;
    __m512i pending_h0 = pending_f0;
    __m512i pending_h1 = pending_f1;

    __m512i last_h0 = _mm512_load_si512(h_load0 + 31);
    last_h0 = _mm512_max_epi16(
        last_h0, _mm512_add_epi16(pending_f0, segment_gap_31));
    __m512i v_h0 = shift_left_zero_512<2>(last_h0);

    __m512i last_h1 = _mm512_load_si512(h_load1 + 31);
    last_h1 = _mm512_max_epi16(
        last_h1, _mm512_add_epi16(pending_f1, segment_gap_31));
    __m512i v_h1 = shift_left_zero_512<2>(last_h1);

    for (std::size_t segment = 0; segment < 32U; segment += 4U) {
      STRIDE_ALIGN_AVX512_DUAL_SW_I16_STEP(
          segment, profile_row0, h_store0, h_load0, e_store0, v_h0, v_f0, pending_h0, best0);
      STRIDE_ALIGN_AVX512_DUAL_SW_I16_STEP(
          segment, profile_row1, h_store1, h_load1, e_store1, v_h1, v_f1, pending_h1, best1);
      STRIDE_ALIGN_AVX512_DUAL_SW_I16_STEP(
          segment + 1U, profile_row0, h_store0, h_load0, e_store0, v_h0, v_f0, pending_h0, best0);
      STRIDE_ALIGN_AVX512_DUAL_SW_I16_STEP(
          segment + 1U, profile_row1, h_store1, h_load1, e_store1, v_h1, v_f1, pending_h1, best1);
      STRIDE_ALIGN_AVX512_DUAL_SW_I16_STEP(
          segment + 2U, profile_row0, h_store0, h_load0, e_store0, v_h0, v_f0, pending_h0, best0);
      STRIDE_ALIGN_AVX512_DUAL_SW_I16_STEP(
          segment + 2U, profile_row1, h_store1, h_load1, e_store1, v_h1, v_f1, pending_h1, best1);
      STRIDE_ALIGN_AVX512_DUAL_SW_I16_STEP(
          segment + 3U, profile_row0, h_store0, h_load0, e_store0, v_h0, v_f0, pending_h0, best0);
      STRIDE_ALIGN_AVX512_DUAL_SW_I16_STEP(
          segment + 3U, profile_row1, h_store1, h_load1, e_store1, v_h1, v_f1, pending_h1, best1);
    }

    v_f0 = local_linear_sw_lazy_f_prefix_carry_i16_32_512(
        v_f0, span_gap_1, span_gap_2, span_gap_4, span_gap_8, span_gap_16, zero);
    v_f1 = local_linear_sw_lazy_f_prefix_carry_i16_32_512(
        v_f1, span_gap_1, span_gap_2, span_gap_4, span_gap_8, span_gap_16, zero);
    pending_f0 = _mm512_max_epi16(v_f0, zero);
    pending_f1 = _mm512_max_epi16(v_f1, zero);
    best0 = _mm512_max_epi16(best0, pending_f0);
    best1 = _mm512_max_epi16(best1, pending_f1);
  }

  for (std::size_t col = n_pair; col < n0; ++col) {
    run_deferred_column(
        h_store0, h_load0, e_store0, offsets0[col], pending_f0, best0);
  }
  for (std::size_t col = n_pair; col < n1; ++col) {
    run_deferred_column(
        h_store1, h_load1, e_store1, offsets1[col], pending_f1, best1);
  }

#undef STRIDE_ALIGN_AVX512_DUAL_SW_I16_STEP

  return {reduce_max_i16_512_local(best0), reduce_max_i16_512_local(best1)};
}

// Batch 1:many path for dual-target deferred exact-fill. Returns true if the
// entire batch was handled (caller must not fall through to sequential).
inline bool try_score_batch_exact_fill_i16_32_dual(
    farrar_fixed_kernel::detail::PreparedScoreBatchState<std::int16_t>& batch,
    std::vector<Score>& scores) {
  auto& state = batch.state;
  if (state.gap_score > 0 || state.query_size != 32U * 32U ||
      state.segment_count != 32U || state.profile.empty() ||
      batch.target_sizes.empty()) {
    return false;
  }
  // Dual path is deferred-only (matches hot automatic path).
  if (state.kernel_strategy ==
          farrar_fixed_kernel::detail::ScoreKernelStrategy::materialized ||
      state.kernel_strategy ==
          farrar_fixed_kernel::detail::ScoreKernelStrategy::compact ||
      state.kernel_strategy ==
          farrar_fixed_kernel::detail::ScoreKernelStrategy::bounded ||
      state.kernel_strategy ==
          farrar_fixed_kernel::detail::ScoreKernelStrategy::bounded_unroll4) {
    return false;
  }

  const std::size_t n_targets = batch.target_sizes.size();
  scores.assign(n_targets, 0);

  // Ensure primary buffers are sized (prepare should have done this).
  constexpr std::size_t state_cells = 32U * 32U;
  if (state.h_store.size() < state_cells) {
    state.h_store.resize(state_cells);
  }
  if (state.h_load.size() < state_cells) {
    state.h_load.resize(state_cells);
  }
  if (state.e_store.size() < state_cells) {
    state.e_store.resize(state_cells);
  }

  // Secondary target H/E storage — thread_local so 1:many batch calls do not
  // reallocate 6 KiB of striped state every call (hot microbench path).
  thread_local farrar_fixed_kernel::detail::AlignedVector<std::int16_t> h_store1;
  thread_local farrar_fixed_kernel::detail::AlignedVector<std::int16_t> h_load1;
  thread_local farrar_fixed_kernel::detail::AlignedVector<std::int16_t> e_store1;
  if (h_store1.size() < state_cells) {
    h_store1.resize(state_cells);
    h_load1.resize(state_cells);
    e_store1.resize(state_cells);
  }

  const auto* profile_cells = state.profile.data();
  std::vector<std::size_t> eligible;
  eligible.reserve(n_targets);

  for (std::size_t index = 0; index < n_targets; ++index) {
    if (index < batch.fast_scores.size() && batch.fast_scores[index].has_value()) {
      scores[index] = *batch.fast_scores[index];
      continue;
    }
    if (index >= batch.target_profile_offsets.size() ||
        batch.target_profile_offsets[index].empty()) {
      scores[index] = 0;
      continue;
    }
    eligible.push_back(index);
  }

  auto buffers0 = [&]() {
    return std::tuple{
        reinterpret_cast<__m512i*>(state.h_store.data()),
        reinterpret_cast<__m512i*>(state.h_load.data()),
        reinterpret_cast<__m512i*>(state.e_store.data())};
  };
  auto buffers1 = [&]() {
    return std::tuple{
        reinterpret_cast<__m512i*>(h_store1.data()),
        reinterpret_cast<__m512i*>(h_load1.data()),
        reinterpret_cast<__m512i*>(e_store1.data())};
  };

  // Prefer equal-length pairs so the specialized dual_equal hot path runs
  // more often on mixed-length batches. Sort eligible by target length,
  // pair adjacent, then score leftovers singly.
  std::stable_sort(
      eligible.begin(),
      eligible.end(),
      [&](std::size_t a, std::size_t b) {
        return batch.target_sizes[a] < batch.target_sizes[b];
      });

  std::size_t ei = 0;
  while (ei + 1U < eligible.size()) {
    const std::size_t i0 = eligible[ei];
    // Prefer a same-length partner when available.
    std::size_t i1 = eligible[ei + 1U];
    if (batch.target_sizes[i0] != batch.target_sizes[i1]) {
      // Search forward for equal length; if none, take next (unequal path).
      for (std::size_t j = ei + 2U; j < eligible.size(); ++j) {
        if (batch.target_sizes[eligible[j]] == batch.target_sizes[i0]) {
          i1 = eligible[j];
          // Swap into position ei+1 so we can advance by 2 cleanly.
          std::swap(eligible[ei + 1U], eligible[j]);
          i1 = eligible[ei + 1U];
          break;
        }
      }
    }
    const auto& off0 = batch.target_profile_offsets[i0];
    const auto& off1 = batch.target_profile_offsets[i1];
    auto [hs0, hl0, es0] = buffers0();
    auto [hs1, hl1, es1] = buffers1();
    const auto pair = local_sw_score_exact_fill_i16_32_dual(
        profile_cells,
        std::span<const std::size_t>(off0.data(), off0.size()),
        std::span<const std::size_t>(off1.data(), off1.size()),
        state.gap_score,
        hs0,
        hl0,
        es0,
        hs1,
        hl1,
        es1);
    scores[i0] = pair.first;
    scores[i1] = pair.second;
    ei += 2U;
  }

  if (ei < eligible.size()) {
    const std::size_t i0 = eligible[ei];
    state.target_size = batch.target_sizes[i0];
    state.fast_score = std::nullopt;
    state.target_profile_offsets = batch.target_profile_offsets[i0];
    scores[i0] = local_sw_score_exact_fill_i16_32(state);
  }

  return true;
}

template <typename Token, typename Cell>
struct SimdOps;

template <>
struct SimdOps<std::uint8_t, std::int8_t> {
  using vector_type = __m512i;
  using mask_type = __mmask64;
  static constexpr std::size_t alignment = 64;
  static constexpr std::size_t lane_count = 64;
  static constexpr bool has_vector_max = true;
  static constexpr bool local_sw_score_exact_segment16 = true;
  static constexpr bool bounded_local_sw_lazy_f_scan = true;

  static vector_type load_tokens(const std::uint8_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_cells(const std::int8_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_aligned_cells(const std::int8_t* values) {
    return _mm512_load_si512(values);
  }

  static void store_cells(std::int8_t* values, vector_type vector) {
    _mm512_storeu_si512(values, vector);
  }

  static void store_aligned_cells(std::int8_t* values, vector_type vector) {
    _mm512_store_si512(values, vector);
  }

  static vector_type set1(std::int8_t value) {
    return _mm512_set1_epi8(value);
  }

  static vector_type zero() {
    return _mm512_setzero_si512();
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return _mm512_add_epi8(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int8_t sentinel) {
    const vector_type sum = _mm512_add_epi8(lhs, rhs);
    const __mmask64 mask = _mm512_cmpeq_epi8_mask(lhs, set1(sentinel));
    return _mm512_mask_blend_epi8(mask, sum, set1(sentinel));
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return _mm512_max_epi8(lhs, rhs);
  }

  static std::int8_t reduce_max(vector_type vector) {
    vector = _mm512_max_epi8(vector, _mm512_shuffle_i32x4(vector, vector, 0x4E));
    vector = _mm512_max_epi8(vector, _mm512_shuffle_i32x4(vector, vector, 0xB1));
    const __m128i reduced = _mm512_castsi512_si128(vector);
    alignas(16) std::int8_t lanes[16];
    _mm_store_si128(reinterpret_cast<__m128i*>(lanes), reduced);
    std::int8_t best = lanes[0];
    for (std::size_t lane = 1; lane < 16; ++lane) {
      best = lanes[lane] > best ? lanes[lane] : best;
    }
    return best;
  }

  static vector_type shift_left_zero(vector_type vector) {
    return shift_left_zero_512<1>(vector);
  }

  static vector_type shift_left_insert(vector_type vector, std::int8_t inserted) {
    return _mm512_mask_blend_epi8(0x1, shift_left_zero_512<1>(vector), set1(inserted));
  }

  static vector_type global_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int8_t gap_extend_score,
      std::int8_t low_score) {
    return global_lazy_f_prefix_carry_512<1, lane_count>(
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
    auto shifted = add(shift_left_insert_bytes_512<1>(prefix, zero_vector), set1(span_gap));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_bytes_512<2>(prefix, zero_vector),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 2)));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_bytes_512<4>(prefix, zero_vector),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 4)));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_bytes_512<8>(prefix, zero_vector),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 8)));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_bytes_512<16>(prefix, zero_vector),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 16)));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_bytes_512<32>(prefix, zero_vector),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 32)));
    prefix = max(prefix, shifted);
    return shift_left_insert_bytes_512<1>(max(prefix, zero_vector), zero_vector);
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return _mm512_cmpgt_epi8_mask(lhs, rhs) != 0;
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return _mm512_cmpgt_epi8_mask(lhs, rhs);
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return static_cast<std::uint64_t>(_mm512_cmpgt_epi8_mask(lhs, rhs));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return static_cast<std::uint64_t>(_mm512_cmpeq_epi8_mask(lhs, rhs));
  }

  static mask_type empty_mask() {
    return 0;
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return lhs | rhs;
  }

  static bool any_mask(mask_type mask) {
    return mask != 0;
  }

  static vector_type substitution(
      const std::uint8_t* query,
      const std::uint8_t* target,
      std::int8_t match_score,
      std::int8_t mismatch_score) {
    const __mmask64 mask = _mm512_cmpeq_epi8_mask(load_tokens(query), load_tokens(target));
    return _mm512_mask_blend_epi8(mask, set1(mismatch_score), set1(match_score));
  }
};

template <>
struct SimdOps<std::uint16_t, std::int16_t> {
  using vector_type = __m512i;
  using mask_type = __mmask32;
  static constexpr std::size_t alignment = 64;
  static constexpr std::size_t lane_count = 32;
  static constexpr bool has_vector_max = true;
  static constexpr bool dense_global_lazy_f_scan = true;
  static constexpr bool plain_global_main_f_after_first_segment = true;
  static constexpr bool global_main_f_segment32_unroll = true;
  static constexpr bool local_sw_score_exact_segment32 = true;
  static constexpr bool local_sw_score_exact_dual_segment32 = true;
  static constexpr bool bounded_local_sw_lazy_f_scan = true;

  static Score local_sw_score_exact_segment32_raw(
      farrar_fixed_kernel::detail::PreparedScoreState<std::int16_t>& state) {
    return local_sw_score_exact_fill_i16_32(state);
  }

  // Dual-target 1:many batch hook for deferred exact-fill i16×1024.
  static bool try_score_batch_exact_fill_dual(
      farrar_fixed_kernel::detail::PreparedScoreBatchState<std::int16_t>& batch,
      std::vector<Score>& scores) {
    return try_score_batch_exact_fill_i16_32_dual(batch, scores);
  }

  static vector_type load_tokens(const std::uint16_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_cells(const std::int16_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_aligned_cells(const std::int16_t* values) {
    return _mm512_load_si512(values);
  }

  static void store_cells(std::int16_t* values, vector_type vector) {
    _mm512_storeu_si512(values, vector);
  }

  static void store_aligned_cells(std::int16_t* values, vector_type vector) {
    _mm512_store_si512(values, vector);
  }

  static void store_masked_cells(std::int16_t* values, mask_type mask, vector_type vector) {
    _mm512_mask_storeu_epi16(values, mask, vector);
  }

  static vector_type set1(std::int16_t value) {
    return _mm512_set1_epi16(value);
  }

  static vector_type zero() {
    return _mm512_setzero_si512();
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return _mm512_add_epi16(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int16_t sentinel) {
    const vector_type sum = _mm512_add_epi16(lhs, rhs);
    const __mmask32 mask = _mm512_cmpeq_epi16_mask(lhs, set1(sentinel));
    return _mm512_mask_blend_epi16(mask, sum, set1(sentinel));
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return _mm512_max_epi16(lhs, rhs);
  }

  static std::int16_t reduce_max(vector_type vector) {
    vector = _mm512_max_epi16(vector, _mm512_shuffle_i32x4(vector, vector, 0x4E));
    vector = _mm512_max_epi16(vector, _mm512_shuffle_i32x4(vector, vector, 0xB1));
    const __m128i reduced = _mm512_castsi512_si128(vector);
    alignas(16) std::int16_t lanes[8];
    _mm_store_si128(reinterpret_cast<__m128i*>(lanes), reduced);
    std::int16_t best = lanes[0];
    for (std::size_t lane = 1; lane < 8; ++lane) {
      best = lanes[lane] > best ? lanes[lane] : best;
    }
    return best;
  }

  static vector_type shift_left_zero(vector_type vector) {
    return shift_left_zero_512<2>(vector);
  }

  static vector_type shift_left_insert(vector_type vector, std::int16_t inserted) {
    return _mm512_mask_blend_epi16(0x1, shift_left_zero_512<2>(vector), set1(inserted));
  }

  static vector_type global_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int16_t gap_extend_score,
      std::int16_t low_score) {
    return global_lazy_f_prefix_carry_512<2, lane_count>(
        final_f,
        segment_count,
        gap_extend_score,
        low_score);
  }

  static vector_type local_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int16_t gap_score) {
    const auto span_gap = static_cast<std::int16_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_score));
    const auto zero_vector = zero();
    auto prefix = max(final_f, zero_vector);
    auto shifted = add(shift_left_insert_bytes_512<2>(prefix, zero_vector), set1(span_gap));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_bytes_512<4>(prefix, zero_vector),
        set1(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 2)));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_bytes_512<8>(prefix, zero_vector),
        set1(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 4)));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_bytes_512<16>(prefix, zero_vector),
        set1(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 8)));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_bytes_512<32>(prefix, zero_vector),
        set1(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 16)));
    prefix = max(prefix, shifted);
    return shift_left_insert_bytes_512<2>(max(prefix, zero_vector), zero_vector);
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return _mm512_cmpgt_epi16_mask(lhs, rhs) != 0;
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return _mm512_cmpgt_epi16_mask(lhs, rhs);
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return static_cast<std::uint64_t>(_mm512_cmpgt_epi16_mask(lhs, rhs));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return static_cast<std::uint64_t>(_mm512_cmpeq_epi16_mask(lhs, rhs));
  }

  static mask_type empty_mask() {
    return 0;
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return lhs | rhs;
  }

  static bool any_mask(mask_type mask) {
    return mask != 0;
  }

  static vector_type substitution(
      const std::uint16_t* query,
      const std::uint16_t* target,
      std::int16_t match_score,
      std::int16_t mismatch_score) {
    const __mmask32 mask = _mm512_cmpeq_epi16_mask(load_tokens(query), load_tokens(target));
    return _mm512_mask_blend_epi16(mask, set1(mismatch_score), set1(match_score));
  }
};

template <>
struct SimdOps<std::uint32_t, std::int32_t> {
  using vector_type = __m512i;
  using mask_type = __mmask16;
  static constexpr std::size_t alignment = 64;
  static constexpr std::size_t lane_count = 16;
  static constexpr bool has_vector_max = true;
  static constexpr bool dense_global_lazy_f_scan = true;
  static constexpr bool plain_global_main_f_after_first_segment = true;
  static constexpr bool global_main_f_segment64_unroll = true;
  static constexpr bool local_sw_score_exact_segment64 = true;
  static constexpr bool bounded_local_sw_lazy_f_scan = true;

  static vector_type load_tokens(const std::uint32_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_cells(const std::int32_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_aligned_cells(const std::int32_t* values) {
    return _mm512_load_si512(values);
  }

  static void store_cells(std::int32_t* values, vector_type vector) {
    _mm512_storeu_si512(values, vector);
  }

  static void store_aligned_cells(std::int32_t* values, vector_type vector) {
    _mm512_store_si512(values, vector);
  }

  static vector_type set1(std::int32_t value) {
    return _mm512_set1_epi32(value);
  }

  static vector_type zero() {
    return _mm512_setzero_si512();
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return _mm512_add_epi32(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int32_t sentinel) {
    const vector_type sum = _mm512_add_epi32(lhs, rhs);
    const __mmask16 mask = _mm512_cmpeq_epi32_mask(lhs, set1(sentinel));
    return _mm512_mask_blend_epi32(mask, sum, set1(sentinel));
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return _mm512_max_epi32(lhs, rhs);
  }

  static std::int32_t reduce_max(vector_type vector) {
    return static_cast<std::int32_t>(_mm512_reduce_max_epi32(vector));
  }

  static vector_type shift_left_zero(vector_type vector) {
    return shift_left_zero_512<4>(vector);
  }

  static vector_type shift_left_insert(vector_type vector, std::int32_t inserted) {
    return _mm512_mask_blend_epi32(0x1, shift_left_zero_512<4>(vector), set1(inserted));
  }

  static vector_type global_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int32_t gap_extend_score,
      std::int32_t low_score) {
    return global_lazy_f_prefix_carry_512<4, lane_count>(
        final_f,
        segment_count,
        gap_extend_score,
        low_score);
  }

  static vector_type local_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int32_t gap_score) {
    const auto span_gap = static_cast<std::int32_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_score));
    const auto zero_vector = zero();
    auto prefix = max(final_f, zero_vector);
    auto shifted = add(shift_left_insert_bytes_512<4>(prefix, zero_vector), set1(span_gap));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_bytes_512<8>(prefix, zero_vector),
        set1(static_cast<std::int32_t>(static_cast<Score>(span_gap) * 2)));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_bytes_512<16>(prefix, zero_vector),
        set1(static_cast<std::int32_t>(static_cast<Score>(span_gap) * 4)));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_bytes_512<32>(prefix, zero_vector),
        set1(static_cast<std::int32_t>(static_cast<Score>(span_gap) * 8)));
    prefix = max(prefix, shifted);
    return shift_left_insert_bytes_512<4>(max(prefix, zero_vector), zero_vector);
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return _mm512_cmpgt_epi32_mask(lhs, rhs) != 0;
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return _mm512_cmpgt_epi32_mask(lhs, rhs);
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return static_cast<std::uint64_t>(_mm512_cmpgt_epi32_mask(lhs, rhs));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return static_cast<std::uint64_t>(_mm512_cmpeq_epi32_mask(lhs, rhs));
  }

  static mask_type empty_mask() {
    return 0;
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return lhs | rhs;
  }

  static bool any_mask(mask_type mask) {
    return mask != 0;
  }

  static vector_type substitution(
      const std::uint32_t* query,
      const std::uint32_t* target,
      std::int32_t match_score,
      std::int32_t mismatch_score) {
    const __mmask16 mask = _mm512_cmpeq_epi32_mask(load_tokens(query), load_tokens(target));
    return _mm512_mask_blend_epi32(mask, set1(mismatch_score), set1(match_score));
  }

  static Score local_affine_score_exact_segment64_raw(
      farrar_fixed_kernel::detail::PreparedAffineScoreState<std::int32_t>& state,
      std::span<const std::size_t> target_profile_offsets) {
    return local_affine_score_exact_fill_i32_64(state, target_profile_offsets);
  }
};

template <>
struct SimdOps<std::uint64_t, std::int64_t> {
  using vector_type = __m512i;
  using mask_type = __mmask8;
  static constexpr std::size_t alignment = 64;
  static constexpr std::size_t lane_count = 8;
  static constexpr bool has_vector_max = true;

  static vector_type load_tokens(const std::uint64_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_cells(const std::int64_t* values) {
    return _mm512_loadu_si512(values);
  }

  static vector_type load_aligned_cells(const std::int64_t* values) {
    return _mm512_load_si512(values);
  }

  static void store_cells(std::int64_t* values, vector_type vector) {
    _mm512_storeu_si512(values, vector);
  }

  static void store_aligned_cells(std::int64_t* values, vector_type vector) {
    _mm512_store_si512(values, vector);
  }

  static vector_type set1(std::int64_t value) {
    return _mm512_set1_epi64(value);
  }

  static vector_type zero() {
    return _mm512_setzero_si512();
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return _mm512_add_epi64(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int64_t sentinel) {
    const vector_type sum = _mm512_add_epi64(lhs, rhs);
    const __mmask8 mask = _mm512_cmpeq_epi64_mask(lhs, set1(sentinel));
    return _mm512_mask_blend_epi64(mask, sum, set1(sentinel));
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return _mm512_max_epi64(lhs, rhs);
  }

  static std::int64_t reduce_max(vector_type vector) {
    return static_cast<std::int64_t>(_mm512_reduce_max_epi64(vector));
  }

  static vector_type shift_left_zero(vector_type vector) {
    return shift_left_zero_512<8>(vector);
  }

  static vector_type shift_left_insert(vector_type vector, std::int64_t inserted) {
    return _mm512_mask_blend_epi64(0x1, shift_left_zero_512<8>(vector), set1(inserted));
  }

  static vector_type global_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int64_t gap_extend_score,
      std::int64_t low_score) {
    return global_lazy_f_prefix_carry_512<8, lane_count>(
        final_f,
        segment_count,
        gap_extend_score,
        low_score);
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return _mm512_cmpgt_epi64_mask(lhs, rhs) != 0;
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return _mm512_cmpgt_epi64_mask(lhs, rhs);
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return static_cast<std::uint64_t>(_mm512_cmpgt_epi64_mask(lhs, rhs));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return static_cast<std::uint64_t>(_mm512_cmpeq_epi64_mask(lhs, rhs));
  }

  static mask_type empty_mask() {
    return 0;
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return lhs | rhs;
  }

  static bool any_mask(mask_type mask) {
    return mask != 0;
  }

  static vector_type substitution(
      const std::uint64_t* query,
      const std::uint64_t* target,
      std::int64_t match_score,
      std::int64_t mismatch_score) {
    const __mmask8 mask = _mm512_cmpeq_epi64_mask(load_tokens(query), load_tokens(target));
    return _mm512_mask_blend_epi64(mask, set1(mismatch_score), set1(match_score));
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
        ::stride_align::levenshtein_simd::Avx512Ops>(query, targets, cutoff);
  }

  static std::vector<double> levenshtein_normalized_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    return ::stride_align::levenshtein_simd::levenshtein_normalized_scores_simd<
        ::stride_align::levenshtein_simd::Avx512Ops>(query, targets, cutoff);
  }

  static std::vector<Score> damerau_levenshtein_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::osa_simd::osa_scores_simd<
        ::stride_align::levenshtein_simd::Avx512Ops>(query, targets);
  }

  static std::vector<double> damerau_levenshtein_normalized_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::osa_simd::osa_normalized_scores_simd<
        ::stride_align::levenshtein_simd::Avx512Ops>(query, targets);
  }

  static std::vector<Score> indel_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::indel_simd::indel_scores_simd<
        ::stride_align::levenshtein_simd::Avx512Ops>(query, targets);
  }

  static std::vector<double> indel_normalized_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::indel_simd::indel_normalized_scores_simd<
        ::stride_align::levenshtein_simd::Avx512Ops>(query, targets);
  }

  static std::vector<double> jaro_similarities(
      nb::handle query, nb::handle targets) {
    return ::stride_align::jaro_simd::jaro_similarities_simd<
        ::stride_align::levenshtein_simd::Avx512Ops>(query, targets);
  }

  static std::vector<double> jaro_winkler_similarities(
      nb::handle query,
      nb::handle targets,
      double prefix_weight,
      double prefix_threshold,
      std::size_t prefix_cap) {
    return ::stride_align::jaro_simd::jaro_winkler_similarities_simd<
        ::stride_align::levenshtein_simd::Avx512Ops>(
        query, targets, prefix_weight, prefix_threshold, prefix_cap);
  }

  static nb::object cdist(
      nb::handle queries, nb::handle targets, int scorer,
      nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    return ::stride_align::cdist_simd::cdist_impl<
        ::stride_align::levenshtein_simd::Avx512Ops>(
        queries, targets, scorer, tqdm_factory, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static nb::object cdist_above_threshold(
      nb::handle queries, nb::handle targets, int scorer,
      double threshold, nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    return ::stride_align::cdist_threshold::cdist_threshold_impl<
        ::stride_align::levenshtein_simd::Avx512Ops>(
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
        ::stride_align::levenshtein_simd::Avx512Ops>(
        queries, targets, scorer, k, tqdm_factory, cpu_count,
        reject_duplicates,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static nb::object cdist_top_k_per_query_threaded(
      nb::handle queries, nb::handle targets, int scorer,
      std::size_t k, bool pruning, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    return ::stride_align::cdist_top_k_per_query_threaded::
        cdist_top_k_per_query_threaded_impl<
            ::stride_align::levenshtein_simd::Avx512Ops>(
        queries, targets, scorer, k, pruning, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  // Matrix-mode entry points. Inputs are already alphabet indices
  // (bytes / 1-byte unicode) and a contiguous row-major (stride × stride)
  // int8 substitution matrix; the SubstitutionMatrix Python wrapper does
  // the str → indices encoding. Routed through the same striped Farrar
  // inner loop as match/mismatch — only the query profile differs.
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

  // Batch matrix-mode entry points: one query, N targets. Builds the
  // query profile once (covering every token in [0, stride)) and reuses
  // it across all targets. Per-target work then collapses to setting
  // up target_profile_offsets and running the inner DP loop.
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

  // Affine matrix-mode entry points. Gap penalties are separated into
  // open and extend; the kernel uses the same striped query profile as
  // linear gap mode but runs the affine DP recurrence (separate E/F
  // tracks with open vs. extend).
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
    return __builtin_cpu_supports("avx512f") != 0 &&
           __builtin_cpu_supports("avx512bw") != 0 &&
           __builtin_cpu_supports("avx512vl") != 0 &&
           __builtin_cpu_supports("avx512dq") != 0;
#else
    return true;
#endif
  }

  static STRIDE_ALIGN_X86_BASELINE void ensure_supported() {
    if (supported_on_this_machine()) {
      return;
    }

    PyErr_SetString(
        PyExc_RuntimeError,
        "x86 AVX-512BWVL backend is not available on this machine");
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

  static nb::object cdist_top_k_per_query_threaded(
      nb::handle queries, nb::handle targets, int scorer,
      std::size_t k, bool pruning, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    return TargetImplementation::cdist_top_k_per_query_threaded(
        queries, targets, scorer, k, pruning, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_score_matrix(
      nb::handle query_indices,
      nb::handle target_indices,
      nb::handle matrix_buffer,
      std::size_t stride,
      Score gap_score) {
    ensure_supported();
    return TargetImplementation::smith_waterman_score_matrix(
        query_indices, target_indices, matrix_buffer, stride, gap_score);
  }

  static STRIDE_ALIGN_X86_BASELINE Score needleman_wunsch_score_matrix(
      nb::handle query_indices,
      nb::handle target_indices,
      nb::handle matrix_buffer,
      std::size_t stride,
      Score gap_score) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_score_matrix(
        query_indices, target_indices, matrix_buffer, stride, gap_score);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> smith_waterman_scores_matrix(
      nb::handle query_indices,
      nb::handle targets,
      nb::handle matrix_buffer,
      std::size_t stride,
      Score gap_score) {
    ensure_supported();
    return TargetImplementation::smith_waterman_scores_matrix(
        query_indices, targets, matrix_buffer, stride, gap_score);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> needleman_wunsch_scores_matrix(
      nb::handle query_indices,
      nb::handle targets,
      nb::handle matrix_buffer,
      std::size_t stride,
      Score gap_score) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_scores_matrix(
        query_indices, targets, matrix_buffer, stride, gap_score);
  }

  static STRIDE_ALIGN_X86_BASELINE Score smith_waterman_affine_score_matrix(
      nb::handle query_indices,
      nb::handle target_indices,
      nb::handle matrix_buffer,
      std::size_t stride,
      Score gap_open_score,
      Score gap_extend_score) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_score_matrix(
        query_indices, target_indices, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static STRIDE_ALIGN_X86_BASELINE Score needleman_wunsch_affine_score_matrix(
      nb::handle query_indices,
      nb::handle target_indices,
      nb::handle matrix_buffer,
      std::size_t stride,
      Score gap_open_score,
      Score gap_extend_score) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_score_matrix(
        query_indices, target_indices, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> smith_waterman_affine_scores_matrix(
      nb::handle query_indices,
      nb::handle targets,
      nb::handle matrix_buffer,
      std::size_t stride,
      Score gap_open_score,
      Score gap_extend_score) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_scores_matrix(
        query_indices, targets, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static STRIDE_ALIGN_X86_BASELINE std::vector<Score> needleman_wunsch_affine_scores_matrix(
      nb::handle query_indices,
      nb::handle targets,
      nb::handle matrix_buffer,
      std::size_t stride,
      Score gap_open_score,
      Score gap_extend_score) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_scores_matrix(
        query_indices, targets, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }


  static std::vector<double> dtw_distances(
      nb::handle query, nb::handle targets,
      nb::object window, nb::object distance,
      nb::object score_cutoff = nb::none()) {

    return ::stride_align::dtw::dtw_distances_simd<
        ::stride_align::dtw_simd::Avx512DtwOps>(query, targets, window, distance, score_cutoff);
  }

  static constexpr BackendKind backend_kind = BackendKind::x86_avx512bwvl;
};

#undef STRIDE_ALIGN_X86_BASELINE

}  // namespace stride_align::backend_avx512bwvl
