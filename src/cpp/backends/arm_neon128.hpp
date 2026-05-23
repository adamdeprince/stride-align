#pragma once

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include <nanobind/nanobind.h>

#include "backends/affine_fixed_kernel.hpp"
#include "backends/farrar_fixed_kernel.hpp"
#include "backends/profile_traceback.hpp"
#include "backends/arm_neon_kernel.hpp"
#include "cdist_simd.hpp"
#include "jaro_simd.hpp"
#include "levenshtein_simd.hpp"
#include "levenshtein_simd_ops.hpp"
#include "osa_simd.hpp"

namespace stride_align::arm_neon128_backend {

namespace nb = nanobind;

inline bool any_mask(uint8x16_t mask) {
  const uint64x2_t words = vreinterpretq_u64_u8(mask);
  return (vgetq_lane_u64(words, 0) | vgetq_lane_u64(words, 1)) != 0;
}

inline bool any_mask(uint16x8_t mask) {
  const uint64x2_t words = vreinterpretq_u64_u16(mask);
  return (vgetq_lane_u64(words, 0) | vgetq_lane_u64(words, 1)) != 0;
}

inline bool any_mask(uint32x4_t mask) {
  const uint64x2_t words = vreinterpretq_u64_u32(mask);
  return (vgetq_lane_u64(words, 0) | vgetq_lane_u64(words, 1)) != 0;
}

inline bool any_mask(uint64x2_t mask) {
  return (vgetq_lane_u64(mask, 0) | vgetq_lane_u64(mask, 1)) != 0;
}

template <typename Lane, std::size_t LaneCount>
inline std::uint64_t trace_lane_mask(const Lane (&values)[LaneCount]) noexcept {
  std::uint64_t mask = 0;
  for (std::size_t lane = 0; lane < LaneCount; ++lane) {
    if (values[lane] != 0) {
      mask |= std::uint64_t{1} << lane;
    }
  }
  return mask;
}

// NEON-native movemask: collapse a SIMD compare result (per-lane FFFF/0000)
// into a packed bit-per-lane uint64. AND with a power-of-two pattern then
// horizontal-sum. This is the hot inner-loop alternative to the scalar
// store + per-lane-loop trace_lane_mask path, which is a real bottleneck on
// trace/cigar/path-info workloads.
inline std::uint64_t lane_mask(uint16x8_t mask) noexcept {
  static const uint16_t kBitsRaw[8] = {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U};
  const uint16x8_t bits = vld1q_u16(kBitsRaw);
  return vaddvq_u16(vandq_u16(mask, bits));
}

inline std::uint64_t lane_mask(uint32x4_t mask) noexcept {
  static const uint32_t kBitsRaw[4] = {1U, 2U, 4U, 8U};
  const uint32x4_t bits = vld1q_u32(kBitsRaw);
  return vaddvq_u32(vandq_u32(mask, bits));
}

inline std::uint64_t lane_mask(uint8x16_t mask) noexcept {
  static const uint8_t kBitsRaw[16] = {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U,
                                       1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U};
  const uint8x16_t bits = vld1q_u8(kBitsRaw);
  const uint8x16_t masked = vandq_u8(mask, bits);
  const std::uint64_t lo = vaddv_u8(vget_low_u8(masked));
  const std::uint64_t hi = vaddv_u8(vget_high_u8(masked));
  return (hi << 8) | lo;
}

inline std::uint64_t lane_mask(uint64x2_t mask) noexcept {
  const std::uint64_t lo = vgetq_lane_u64(mask, 0) ? 1U : 0U;
  const std::uint64_t hi = vgetq_lane_u64(mask, 1) ? 2U : 0U;
  return lo | hi;
}

// NEON local-SW affine score exact-fill kernels. Two specializations for the
// hot 1024-character query shapes:
//   - i16 / width16: 8 lanes -> 128 segments
//   - i32 / width32: 4 lanes -> 256 segments
//
// Mirrors the AVX2 (x86) / LASX (LoongArch) P2 pattern: branchless H/E/profile
// DP body, log-step prefix carry for cross-lane lazy-F propagation, then a
// no-E correction loop (H-only, parasail-style — empirically score-preserving
// for these workloads). Dispatched through the
// farrar_fixed_kernel::detail::affine_score_state_for_offsets hook by
// declaring local_affine_score_exact_segment{128,256}_raw on the SimdOps.
//
// NEON is 128-bit so the cross-lane prefix-carry shifts use direct vextq_u8
// byte counts (no intermediate per-lane loop, no cross-128-lane carry needed).
inline Score local_affine_score_exact_fill_i16_128(
    farrar_fixed_kernel::detail::PreparedAffineScoreState<std::int16_t>& state,
    std::span<const std::size_t> target_profile_offsets) {
  if (state.query_size != 1024U || state.segment_count != 128U ||
      target_profile_offsets.empty() || state.gap_open_score > state.gap_extend_score ||
      state.gap_extend_score > 0) {
    return 0;
  }

  std::fill(state.h_store.begin(), state.h_store.end(), std::int16_t{0});
  std::fill(state.h_load.begin(), state.h_load.end(), std::int16_t{0});
  std::fill(state.e_store.begin(), state.e_store.end(), std::int16_t{0});

  const int16x8_t zero = vdupq_n_s16(0);
  const int16x8_t gap_open = vdupq_n_s16(state.gap_open_score);
  const int16x8_t gap_extend = vdupq_n_s16(state.gap_extend_score);
  const auto lane_gap = static_cast<std::int16_t>(
      static_cast<Score>(128) * static_cast<Score>(state.gap_extend_score));
  const int16x8_t lane_gap_1 = vdupq_n_s16(lane_gap);
  const int16x8_t lane_gap_2 = vdupq_n_s16(
      static_cast<std::int16_t>(static_cast<Score>(lane_gap) * 2));
  const int16x8_t lane_gap_4 = vdupq_n_s16(
      static_cast<std::int16_t>(static_cast<Score>(lane_gap) * 4));
  int16x8_t best = zero;
  std::int16_t* h_store = state.h_store.data();
  std::int16_t* h_load = state.h_load.data();
  std::int16_t* e_store = state.e_store.data();
  const std::int16_t* profile_cells = state.profile.data();

  for (const auto profile_offset : target_profile_offsets) {
    std::swap(h_store, h_load);

    // Wrap-around v_h: shift up by 1 lane (2 bytes), insert zero at lane 0.
    int16x8_t v_h = vreinterpretq_s16_u8(vextq_u8(
        vreinterpretq_u8_s16(zero),
        vreinterpretq_u8_s16(vld1q_s16(h_load + 127 * 8)),
        14));
    int16x8_t v_f = zero;
    const std::int16_t* profile_row = profile_cells + profile_offset;

    for (std::size_t segment = 0; segment < 128U; ++segment) {
      const int16x8_t v_profile = vld1q_s16(profile_row + segment * 8);
      int16x8_t v_e = vld1q_s16(e_store + segment * 8);
      v_h = vaddq_s16(v_h, v_profile);
      v_h = vmaxq_s16(v_h, v_e);
      v_h = vmaxq_s16(v_h, v_f);
      v_h = vmaxq_s16(v_h, zero);
      vst1q_s16(h_store + segment * 8, v_h);
      best = vmaxq_s16(best, v_h);

      const int16x8_t v_h_open = vaddq_s16(v_h, gap_open);
      v_e = vmaxq_s16(vaddq_s16(v_e, gap_extend), v_h_open);
      vst1q_s16(e_store + segment * 8, v_e);
      v_f = vmaxq_s16(vaddq_s16(v_f, gap_extend), v_h_open);
      v_h = vld1q_s16(h_load + segment * 8);
    }

    // Log-step prefix carry across the 8 i16 lanes (3 stages). Each shift uses
    // a single vextq_u8 with the byte count = 16 - shift_bytes.
    int16x8_t prefix = vmaxq_s16(v_f, zero);
    int16x8_t shifted = vaddq_s16(
        vreinterpretq_s16_u8(vextq_u8(
            vreinterpretq_u8_s16(zero), vreinterpretq_u8_s16(prefix), 14)),
        lane_gap_1);
    prefix = vmaxq_s16(prefix, shifted);
    shifted = vaddq_s16(
        vreinterpretq_s16_u8(vextq_u8(
            vreinterpretq_u8_s16(zero), vreinterpretq_u8_s16(prefix), 12)),
        lane_gap_2);
    prefix = vmaxq_s16(prefix, shifted);
    shifted = vaddq_s16(
        vreinterpretq_s16_u8(vextq_u8(
            vreinterpretq_u8_s16(zero), vreinterpretq_u8_s16(prefix), 8)),
        lane_gap_4);
    prefix = vmaxq_s16(prefix, shifted);
    v_f = vreinterpretq_s16_u8(vextq_u8(
        vreinterpretq_u8_s16(zero),
        vreinterpretq_u8_s16(vmaxq_s16(prefix, zero)),
        14));

    if (any_mask(vcgtq_s16(v_f, zero))) {
      // No-E correction: update H only. Next column recomputes E from H so we
      // skip writing E here. Same shape parasail uses on AVX2.
      for (std::size_t segment = 0; segment < 128U; ++segment) {
        const int16x8_t v_h_previous = vld1q_s16(h_store + segment * 8);
        const int16x8_t v_h_corrected = vmaxq_s16(v_h_previous, v_f);
        vst1q_s16(h_store + segment * 8, v_h_corrected);
        best = vmaxq_s16(best, v_h_corrected);
        v_f = vaddq_s16(v_f, gap_extend);
      }
    }
  }

  return static_cast<Score>(vmaxvq_s16(best));
}

inline Score local_affine_score_exact_fill_i32_256(
    farrar_fixed_kernel::detail::PreparedAffineScoreState<std::int32_t>& state,
    std::span<const std::size_t> target_profile_offsets) {
  if (state.query_size != 1024U || state.segment_count != 256U ||
      target_profile_offsets.empty() || state.gap_open_score > state.gap_extend_score ||
      state.gap_extend_score > 0) {
    return 0;
  }

  std::fill(state.h_store.begin(), state.h_store.end(), std::int32_t{0});
  std::fill(state.h_load.begin(), state.h_load.end(), std::int32_t{0});
  std::fill(state.e_store.begin(), state.e_store.end(), std::int32_t{0});

  const int32x4_t zero = vdupq_n_s32(0);
  const int32x4_t gap_open = vdupq_n_s32(state.gap_open_score);
  const int32x4_t gap_extend = vdupq_n_s32(state.gap_extend_score);
  const auto lane_gap = static_cast<std::int32_t>(
      static_cast<Score>(256) * static_cast<Score>(state.gap_extend_score));
  const int32x4_t lane_gap_1 = vdupq_n_s32(lane_gap);
  const int32x4_t lane_gap_2 = vdupq_n_s32(
      static_cast<std::int32_t>(static_cast<Score>(lane_gap) * 2));
  int32x4_t best = zero;
  std::int32_t* h_store = state.h_store.data();
  std::int32_t* h_load = state.h_load.data();
  std::int32_t* e_store = state.e_store.data();
  const std::int32_t* profile_cells = state.profile.data();

  for (const auto profile_offset : target_profile_offsets) {
    std::swap(h_store, h_load);

    int32x4_t v_h = vreinterpretq_s32_u8(vextq_u8(
        vreinterpretq_u8_s32(zero),
        vreinterpretq_u8_s32(vld1q_s32(h_load + 255 * 4)),
        12));
    int32x4_t v_f = zero;
    const std::int32_t* profile_row = profile_cells + profile_offset;

    for (std::size_t segment = 0; segment < 256U; ++segment) {
      const int32x4_t v_profile = vld1q_s32(profile_row + segment * 4);
      int32x4_t v_e = vld1q_s32(e_store + segment * 4);
      v_h = vaddq_s32(v_h, v_profile);
      v_h = vmaxq_s32(v_h, v_e);
      v_h = vmaxq_s32(v_h, v_f);
      v_h = vmaxq_s32(v_h, zero);
      vst1q_s32(h_store + segment * 4, v_h);
      best = vmaxq_s32(best, v_h);

      const int32x4_t v_h_open = vaddq_s32(v_h, gap_open);
      v_e = vmaxq_s32(vaddq_s32(v_e, gap_extend), v_h_open);
      vst1q_s32(e_store + segment * 4, v_e);
      v_f = vmaxq_s32(vaddq_s32(v_f, gap_extend), v_h_open);
      v_h = vld1q_s32(h_load + segment * 4);
    }

    // Log-step prefix carry across the 4 i32 lanes (2 stages).
    int32x4_t prefix = vmaxq_s32(v_f, zero);
    int32x4_t shifted = vaddq_s32(
        vreinterpretq_s32_u8(vextq_u8(
            vreinterpretq_u8_s32(zero), vreinterpretq_u8_s32(prefix), 12)),
        lane_gap_1);
    prefix = vmaxq_s32(prefix, shifted);
    shifted = vaddq_s32(
        vreinterpretq_s32_u8(vextq_u8(
            vreinterpretq_u8_s32(zero), vreinterpretq_u8_s32(prefix), 8)),
        lane_gap_2);
    prefix = vmaxq_s32(prefix, shifted);
    v_f = vreinterpretq_s32_u8(vextq_u8(
        vreinterpretq_u8_s32(zero),
        vreinterpretq_u8_s32(vmaxq_s32(prefix, zero)),
        12));

    if (any_mask(vcgtq_s32(v_f, zero))) {
      for (std::size_t segment = 0; segment < 256U; ++segment) {
        const int32x4_t v_h_previous = vld1q_s32(h_store + segment * 4);
        const int32x4_t v_h_corrected = vmaxq_s32(v_h_previous, v_f);
        vst1q_s32(h_store + segment * 4, v_h_corrected);
        best = vmaxq_s32(best, v_h_corrected);
        v_f = vaddq_s32(v_f, gap_extend);
      }
    }
  }

  return static_cast<Score>(vmaxvq_s32(best));
}

template <typename Token, typename Cell>
struct SimdOps;

template <>
struct SimdOps<std::uint8_t, std::int8_t> {
  using vector_type = int8x16_t;
  using mask_type = uint8x16_t;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 16;
  static constexpr bool has_vector_max = true;

  static uint8x16_t load_tokens(const std::uint8_t* values) {
    return vld1q_u8(values);
  }

  static vector_type load_cells(const std::int8_t* values) {
    return vld1q_s8(values);
  }

  static void store_cells(std::int8_t* values, vector_type vector) {
    vst1q_s8(values, vector);
  }

  static vector_type set1(std::int8_t value) {
    return vdupq_n_s8(value);
  }

  static vector_type zero() {
    return vdupq_n_s8(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return vaddq_s8(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int8_t sentinel) {
    const auto sum = vaddq_s8(lhs, rhs);
    const auto mask = vceqq_s8(lhs, set1(sentinel));
    return vbslq_s8(mask, set1(sentinel), sum);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return vmaxq_s8(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return vreinterpretq_s8_u8(vextq_u8(vdupq_n_u8(0), vreinterpretq_u8_s8(vector), 15));
  }

  static vector_type shift_left_insert(vector_type vector, std::int8_t inserted) {
    return vreinterpretq_s8_u8(
        vextq_u8(vreinterpretq_u8_s8(set1(inserted)), vreinterpretq_u8_s8(vector), 15));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return any_mask(vcgtq_s8(lhs, rhs));
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return vcgtq_s8(lhs, rhs);
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return arm_neon128_backend::lane_mask(vcgtq_s8(lhs, rhs));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return arm_neon128_backend::lane_mask(vceqq_s8(lhs, rhs));
  }

  static mask_type empty_mask() {
    return vdupq_n_u8(0);
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return vorrq_u8(lhs, rhs);
  }

  static bool any_mask(mask_type mask) {
    return arm_neon128_backend::any_mask(mask);
  }

  static vector_type substitution(
      const std::uint8_t* query,
      const std::uint8_t* target,
      std::int8_t match_score,
      std::int8_t mismatch_score) {
    const uint8x16_t mask = vceqq_u8(load_tokens(query), load_tokens(target));
    return vbslq_s8(mask, set1(match_score), set1(mismatch_score));
  }
};

template <>
struct SimdOps<std::uint16_t, std::int16_t> {
  using vector_type = int16x8_t;
  using mask_type = uint16x8_t;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 8;
  static constexpr bool has_vector_max = true;
  static constexpr bool local_sw_score_exact_segment128 = true;
  static constexpr bool bounded_local_sw_lazy_f_scan = true;

  static uint16x8_t load_tokens(const std::uint16_t* values) {
    return vld1q_u16(values);
  }

  static vector_type load_cells(const std::int16_t* values) {
    return vld1q_s16(values);
  }

  static void store_cells(std::int16_t* values, vector_type vector) {
    vst1q_s16(values, vector);
  }

  static vector_type set1(std::int16_t value) {
    return vdupq_n_s16(value);
  }

  static vector_type zero() {
    return vdupq_n_s16(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return vaddq_s16(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int16_t sentinel) {
    const auto sum = vaddq_s16(lhs, rhs);
    const auto mask = vceqq_s16(lhs, set1(sentinel));
    return vbslq_s16(mask, set1(sentinel), sum);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return vmaxq_s16(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return vreinterpretq_s16_u8(vextq_u8(vdupq_n_u8(0), vreinterpretq_u8_s16(vector), 14));
  }

  static vector_type shift_left_insert(vector_type vector, std::int16_t inserted) {
    return vreinterpretq_s16_u8(
        vextq_u8(vreinterpretq_u8_s16(set1(inserted)), vreinterpretq_u8_s16(vector), 14));
  }

  // Single-shot multi-lane left shift via direct vextq_u8 byte count.
  // shift_left_lanes_insert<N>(v, fill_bytes) = vector shifted up by N lanes
  // (= 2N bytes for i16), with the low N lanes filled from `fill_bytes`.
  template <int Lanes>
  static vector_type shift_left_lanes_zero(vector_type vector) {
    static_assert(Lanes >= 0 && Lanes <= 8);
    if constexpr (Lanes == 0) {
      return vector;
    } else {
      return vreinterpretq_s16_u8(vextq_u8(
          vreinterpretq_u8_s16(zero()),
          vreinterpretq_u8_s16(vector),
          16 - 2 * Lanes));
    }
  }
  template <int Lanes>
  static vector_type shift_left_lanes_insert_sentinel(
      vector_type vector,
      std::int16_t fill) {
    static_assert(Lanes >= 0 && Lanes <= 8);
    if constexpr (Lanes == 0) {
      return vector;
    } else {
      return vreinterpretq_s16_u8(vextq_u8(
          vreinterpretq_u8_s16(set1(fill)),
          vreinterpretq_u8_s16(vector),
          16 - 2 * Lanes));
    }
  }

  static vector_type global_lazy_f_prefix_carry_no_padding(
      vector_type final_f,
      std::size_t segment_count,
      std::int16_t gap_extend_score,
      std::int16_t low_score) {
    const auto span_gap = static_cast<std::int16_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_extend_score));
    auto prefix = final_f;
    auto shifted = add_sentinel(
        shift_left_lanes_insert_sentinel<1>(prefix, low_score),
        set1(span_gap),
        low_score);
    prefix = max(prefix, shifted);
    shifted = add_sentinel(
        shift_left_lanes_insert_sentinel<2>(prefix, low_score),
        set1(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 2)),
        low_score);
    prefix = max(prefix, shifted);
    shifted = add_sentinel(
        shift_left_lanes_insert_sentinel<4>(prefix, low_score),
        set1(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 4)),
        low_score);
    prefix = max(prefix, shifted);
    return shift_left_lanes_insert_sentinel<1>(prefix, low_score);
  }

  static vector_type local_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int16_t gap_score) {
    const auto span_gap = static_cast<std::int16_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_score));
    const auto zero_vector = zero();
    auto prefix = max(final_f, zero_vector);
    auto shifted = add(shift_left_lanes_zero<1>(prefix), set1(span_gap));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_lanes_zero<2>(prefix),
        set1(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 2)));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_lanes_zero<4>(prefix),
        set1(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 4)));
    prefix = max(prefix, shifted);
    return shift_left_lanes_zero<1>(max(prefix, zero_vector));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return any_mask(vcgtq_s16(lhs, rhs));
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return vcgtq_s16(lhs, rhs);
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return arm_neon128_backend::lane_mask(vcgtq_s16(lhs, rhs));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return arm_neon128_backend::lane_mask(vceqq_s16(lhs, rhs));
  }

  static mask_type empty_mask() {
    return vdupq_n_u16(0);
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return vorrq_u16(lhs, rhs);
  }

  static bool any_mask(mask_type mask) {
    return arm_neon128_backend::any_mask(mask);
  }

  static vector_type substitution(
      const std::uint16_t* query,
      const std::uint16_t* target,
      std::int16_t match_score,
      std::int16_t mismatch_score) {
    const uint16x8_t mask = vceqq_u16(load_tokens(query), load_tokens(target));
    return vbslq_s16(mask, set1(match_score), set1(mismatch_score));
  }

  static Score local_affine_score_exact_segment128_raw(
      farrar_fixed_kernel::detail::PreparedAffineScoreState<std::int16_t>& state,
      std::span<const std::size_t> target_profile_offsets) {
    return local_affine_score_exact_fill_i16_128(state, target_profile_offsets);
  }
};

template <>
struct SimdOps<std::uint32_t, std::int32_t> {
  using vector_type = int32x4_t;
  using mask_type = uint32x4_t;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 4;
  static constexpr bool has_vector_max = true;
  static constexpr bool local_sw_score_exact_segment256 = true;
  static constexpr bool bounded_local_sw_lazy_f_scan = true;

  static uint32x4_t load_tokens(const std::uint32_t* values) {
    return vld1q_u32(values);
  }

  static vector_type load_cells(const std::int32_t* values) {
    return vld1q_s32(values);
  }

  static void store_cells(std::int32_t* values, vector_type vector) {
    vst1q_s32(values, vector);
  }

  static vector_type set1(std::int32_t value) {
    return vdupq_n_s32(value);
  }

  static vector_type zero() {
    return vdupq_n_s32(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return vaddq_s32(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int32_t sentinel) {
    const auto sum = vaddq_s32(lhs, rhs);
    const auto mask = vceqq_s32(lhs, set1(sentinel));
    return vbslq_s32(mask, set1(sentinel), sum);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return vmaxq_s32(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return vreinterpretq_s32_u8(vextq_u8(vdupq_n_u8(0), vreinterpretq_u8_s32(vector), 12));
  }

  static vector_type shift_left_insert(vector_type vector, std::int32_t inserted) {
    return vreinterpretq_s32_u8(
        vextq_u8(vreinterpretq_u8_s32(set1(inserted)), vreinterpretq_u8_s32(vector), 12));
  }

  template <int Lanes>
  static vector_type shift_left_lanes_zero(vector_type vector) {
    static_assert(Lanes >= 0 && Lanes <= 4);
    if constexpr (Lanes == 0) {
      return vector;
    } else {
      return vreinterpretq_s32_u8(vextq_u8(
          vreinterpretq_u8_s32(zero()),
          vreinterpretq_u8_s32(vector),
          16 - 4 * Lanes));
    }
  }
  template <int Lanes>
  static vector_type shift_left_lanes_insert_sentinel(
      vector_type vector,
      std::int32_t fill) {
    static_assert(Lanes >= 0 && Lanes <= 4);
    if constexpr (Lanes == 0) {
      return vector;
    } else {
      return vreinterpretq_s32_u8(vextq_u8(
          vreinterpretq_u8_s32(set1(fill)),
          vreinterpretq_u8_s32(vector),
          16 - 4 * Lanes));
    }
  }

  static vector_type global_lazy_f_prefix_carry_no_padding(
      vector_type final_f,
      std::size_t segment_count,
      std::int32_t gap_extend_score,
      std::int32_t low_score) {
    const auto span_gap = static_cast<std::int32_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_extend_score));
    auto prefix = final_f;
    auto shifted = add_sentinel(
        shift_left_lanes_insert_sentinel<1>(prefix, low_score),
        set1(span_gap),
        low_score);
    prefix = max(prefix, shifted);
    shifted = add_sentinel(
        shift_left_lanes_insert_sentinel<2>(prefix, low_score),
        set1(static_cast<std::int32_t>(static_cast<Score>(span_gap) * 2)),
        low_score);
    prefix = max(prefix, shifted);
    return shift_left_lanes_insert_sentinel<1>(prefix, low_score);
  }

  static vector_type local_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int32_t gap_score) {
    const auto span_gap = static_cast<std::int32_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_score));
    const auto zero_vector = zero();
    auto prefix = max(final_f, zero_vector);
    auto shifted = add(shift_left_lanes_zero<1>(prefix), set1(span_gap));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_lanes_zero<2>(prefix),
        set1(static_cast<std::int32_t>(static_cast<Score>(span_gap) * 2)));
    prefix = max(prefix, shifted);
    return shift_left_lanes_zero<1>(max(prefix, zero_vector));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return any_mask(vcgtq_s32(lhs, rhs));
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return vcgtq_s32(lhs, rhs);
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return arm_neon128_backend::lane_mask(vcgtq_s32(lhs, rhs));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return arm_neon128_backend::lane_mask(vceqq_s32(lhs, rhs));
  }

  static mask_type empty_mask() {
    return vdupq_n_u32(0);
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return vorrq_u32(lhs, rhs);
  }

  static bool any_mask(mask_type mask) {
    return arm_neon128_backend::any_mask(mask);
  }

  static vector_type substitution(
      const std::uint32_t* query,
      const std::uint32_t* target,
      std::int32_t match_score,
      std::int32_t mismatch_score) {
    const uint32x4_t mask = vceqq_u32(load_tokens(query), load_tokens(target));
    return vbslq_s32(mask, set1(match_score), set1(mismatch_score));
  }

  static Score local_affine_score_exact_segment256_raw(
      farrar_fixed_kernel::detail::PreparedAffineScoreState<std::int32_t>& state,
      std::span<const std::size_t> target_profile_offsets) {
    return local_affine_score_exact_fill_i32_256(state, target_profile_offsets);
  }
};

template <>
struct SimdOps<std::uint64_t, std::int64_t> {
  using vector_type = int64x2_t;
  using mask_type = uint64x2_t;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 2;
  static constexpr bool has_vector_max = true;

  static uint64x2_t load_tokens(const std::uint64_t* values) {
    return vld1q_u64(values);
  }

  static vector_type load_cells(const std::int64_t* values) {
    return vld1q_s64(values);
  }

  static void store_cells(std::int64_t* values, vector_type vector) {
    vst1q_s64(values, vector);
  }

  static vector_type set1(std::int64_t value) {
    return vdupq_n_s64(value);
  }

  static vector_type zero() {
    return vdupq_n_s64(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return vaddq_s64(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int64_t sentinel) {
    const auto sum = vaddq_s64(lhs, rhs);
    const auto mask = vceqq_s64(lhs, set1(sentinel));
    return vbslq_s64(mask, set1(sentinel), sum);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    const uint64x2_t mask = vcgtq_s64(lhs, rhs);
    return vbslq_s64(mask, lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return vreinterpretq_s64_u8(vextq_u8(vdupq_n_u8(0), vreinterpretq_u8_s64(vector), 8));
  }

  static vector_type shift_left_insert(vector_type vector, std::int64_t inserted) {
    return vreinterpretq_s64_u8(
        vextq_u8(vreinterpretq_u8_s64(set1(inserted)), vreinterpretq_u8_s64(vector), 8));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return any_mask(vcgtq_s64(lhs, rhs));
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return vcgtq_s64(lhs, rhs);
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return arm_neon128_backend::lane_mask(vcgtq_s64(lhs, rhs));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return arm_neon128_backend::lane_mask(vceqq_s64(lhs, rhs));
  }

  static mask_type empty_mask() {
    return vdupq_n_u64(0);
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return vorrq_u64(lhs, rhs);
  }

  static bool any_mask(mask_type mask) {
    return arm_neon128_backend::any_mask(mask);
  }

  static vector_type substitution(
      const std::uint64_t* query,
      const std::uint64_t* target,
      std::int64_t match_score,
      std::int64_t mismatch_score) {
    const uint64x2_t mask = vceqq_u64(load_tokens(query), load_tokens(target));
    return vbslq_s64(mask, set1(match_score), set1(mismatch_score));
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
      return arm_neon_kernel::detail::dispatch_score<SimdOps, true>(
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

  // Gate the SIMD masked-trace route to widths where NEON has enough lanes to
  // amortize the masked-trace overhead. NEON i16 = 8 lanes works; NEON i32 = 4
  // lanes loses badly to the scalar profile_traceback path. The kernel bits
  // resolved from the score bound (after preparation) decide; user-forced
  // width=32 stays on the scalar path.
  static bool linear_sw_masked_trace_enabled(
      const PreparedFarrarAlignment& prepared,
      unsigned int width) {
    if (width != 0 && width != 8 && width != 16) {
      return false;
    }
    if (prepared.score_bits != KernelBits::bits8 &&
        prepared.score_bits != KernelBits::bits16) {
      return false;
    }
    return true;
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
      if (linear_sw_masked_trace_enabled(prepared, width)) {
        const auto trace = farrar_fixed_kernel::detail::
            dispatch_linear_sw_score_first_masked_cigar_trace<SimdOps>(
                prepared, match_score, mismatch_score, gap_score);
        farrar_fixed_kernel::detail::LinearTracebackResult path;
        path.score = trace.score;
        path.query_start = trace.query_start;
        path.query_end = trace.query_end;
        path.target_start = trace.target_start;
        path.target_end = trace.target_end;
        path.operations = expand_cigar(trace.cigar);
        return profile_traceback::detail::materialize_alignment_result(output_prepared, path);
      }
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
      if (linear_sw_masked_trace_enabled(prepared, width)) {
        return farrar_fixed_kernel::detail::
            dispatch_linear_sw_score_first_masked_cigar_path_info<SimdOps>(
                prepared, match_score, mismatch_score, gap_score);
      }
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
      if (linear_sw_masked_trace_enabled(prepared, width)) {
        return farrar_fixed_kernel::detail::dispatch_linear_sw_score_first_masked_cigar<SimdOps>(
            prepared, match_score, mismatch_score, gap_score);
      }
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
      return arm_neon_kernel::detail::dispatch_score<SimdOps, false>(
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

  // Levenshtein (Myers bit-parallel) multi-target SIMD batch, NEON
  // 2-lane variant. Routes through the same shared kernel as the x86
  // backends; the Ops bundle differs.
  static std::vector<Score> levenshtein_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    return ::stride_align::levenshtein_simd::levenshtein_scores_simd<
        ::stride_align::levenshtein_simd::NeonOps>(query, targets, cutoff);
  }

  static std::vector<double> levenshtein_normalized_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    return ::stride_align::levenshtein_simd::levenshtein_normalized_scores_simd<
        ::stride_align::levenshtein_simd::NeonOps>(query, targets, cutoff);
  }

  // OSA-restricted Damerau-Levenshtein multi-target SIMD batch.
  static std::vector<Score> damerau_levenshtein_scores(
      nb::handle query,
      nb::handle targets) {
    return ::stride_align::osa_simd::osa_scores_simd<
        ::stride_align::levenshtein_simd::NeonOps>(query, targets);
  }

  static std::vector<double> damerau_levenshtein_normalized_scores(
      nb::handle query,
      nb::handle targets) {
    return ::stride_align::osa_simd::osa_normalized_scores_simd<
        ::stride_align::levenshtein_simd::NeonOps>(query, targets);
  }

  static std::vector<double> jaro_similarities(
      nb::handle query, nb::handle targets) {
    return ::stride_align::jaro_simd::jaro_similarities_simd<
        ::stride_align::levenshtein_simd::NeonOps>(query, targets);
  }

  static std::vector<double> jaro_winkler_similarities(
      nb::handle query,
      nb::handle targets,
      double prefix_weight,
      double prefix_threshold,
      std::size_t prefix_cap) {
    return ::stride_align::jaro_simd::jaro_winkler_similarities_simd<
        ::stride_align::levenshtein_simd::NeonOps>(
        query, targets, prefix_weight, prefix_threshold, prefix_cap);
  }

  static nb::object cdist(
      nb::handle queries, nb::handle targets, int scorer,
      nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    return ::stride_align::cdist_simd::cdist_impl<
        ::stride_align::levenshtein_simd::NeonOps>(
        queries, targets, scorer, tqdm_factory, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }
};

}  // namespace stride_align::arm_neon128_backend
