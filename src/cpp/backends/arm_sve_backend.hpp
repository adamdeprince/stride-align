#pragma once

#include <arm_sve.h>

#include <cstddef>
#include <cstdint>

#include <nanobind/nanobind.h>

#include "backends/affine_scalable_kernel.hpp"
#include "backends/arm_sve_kernel.hpp"
#include "backends/farrar_scalable_kernel.hpp"
#include "backends/profile_traceback.hpp"

namespace stride_align::arm_sve_backend {

namespace nb = nanobind;

#if defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS == 128
// Specialized exact-fill SW affine kernel for query_size=1024,
// segment_count=128, i16 cells. Mirrors NEON's
// arm_neon128_backend::local_affine_score_exact_fill_i16_128 — the english
// and chinese benchmark passes default to length=1024, which makes this the
// hot path for affine sw-farrar-score at width 16. The hardcoded loop
// bound lets the compiler tighten the inner block; the lazy-F propagation
// is replaced by a parallel prefix-carry (3 svext+max stages) followed by a
// single 'no-E' correction pass.
inline Score local_affine_score_exact_fill_i16_128_sve(
    farrar_scalable_kernel::detail::PreparedAffineScoreState<std::int16_t>& state,
    std::span<const std::size_t> target_profile_offsets) {
  if (state.query_size != 1024U || state.segment_count != 128U ||
      target_profile_offsets.empty() || state.gap_open_score > state.gap_extend_score ||
      state.gap_extend_score > 0) {
    return 0;
  }

  std::fill(state.h_store.begin(), state.h_store.end(), std::int16_t{0});
  std::fill(state.h_load.begin(), state.h_load.end(), std::int16_t{0});
  std::fill(state.e_store.begin(), state.e_store.end(), std::int16_t{0});

  const svbool_t pg = svptrue_b16();
  const svint16_t zero = svdup_n_s16(0);
  const svint16_t gap_open = svdup_n_s16(state.gap_open_score);
  const svint16_t gap_extend = svdup_n_s16(state.gap_extend_score);
  const auto lane_gap = static_cast<std::int16_t>(
      static_cast<Score>(128) * static_cast<Score>(state.gap_extend_score));
  const svint16_t lane_gap_1 = svdup_n_s16(lane_gap);
  const svint16_t lane_gap_2 = svdup_n_s16(
      static_cast<std::int16_t>(static_cast<Score>(lane_gap) * 2));
  const svint16_t lane_gap_4 = svdup_n_s16(
      static_cast<std::int16_t>(static_cast<Score>(lane_gap) * 4));
  svint16_t best = zero;
  std::int16_t* h_store = state.h_store.data();
  std::int16_t* h_load = state.h_load.data();
  std::int16_t* e_store = state.e_store.data();
  const std::int16_t* profile_cells = state.profile.data();

  for (const auto profile_offset : target_profile_offsets) {
    std::swap(h_store, h_load);

    // Wrap-around v_h: shift up by 1 lane, insert zero at lane 0.
    svint16_t v_h = svext_s16(zero, svld1_s16(pg, h_load + 127 * 8), 7);
    svint16_t v_f = zero;
    const std::int16_t* profile_row = profile_cells + profile_offset;

    for (std::size_t segment = 0; segment < 128U; ++segment) {
      const svint16_t v_profile = svld1_s16(pg, profile_row + segment * 8);
      svint16_t v_e = svld1_s16(pg, e_store + segment * 8);
      v_h = svadd_s16_x(pg, v_h, v_profile);
      v_h = svmax_s16_x(pg, v_h, v_e);
      v_h = svmax_s16_x(pg, v_h, v_f);
      v_h = svmax_s16_x(pg, v_h, zero);
      svst1_s16(pg, h_store + segment * 8, v_h);
      best = svmax_s16_x(pg, best, v_h);

      const svint16_t v_h_open = svadd_s16_x(pg, v_h, gap_open);
      v_e = svmax_s16_x(pg, svadd_s16_x(pg, v_e, gap_extend), v_h_open);
      svst1_s16(pg, e_store + segment * 8, v_e);
      v_f = svmax_s16_x(pg, svadd_s16_x(pg, v_f, gap_extend), v_h_open);
      v_h = svld1_s16(pg, h_load + segment * 8);
    }

    // Log-step prefix carry across the 8 i16 lanes (3 stages). svext_s16
    // with imm3=8-N gives "shift left by N lanes, zero-filling the low".
    svint16_t prefix = svmax_s16_x(pg, v_f, zero);
    svint16_t shifted =
        svadd_s16_x(pg, svext_s16(zero, prefix, 7), lane_gap_1);
    prefix = svmax_s16_x(pg, prefix, shifted);
    shifted = svadd_s16_x(pg, svext_s16(zero, prefix, 6), lane_gap_2);
    prefix = svmax_s16_x(pg, prefix, shifted);
    shifted = svadd_s16_x(pg, svext_s16(zero, prefix, 4), lane_gap_4);
    prefix = svmax_s16_x(pg, prefix, shifted);
    v_f = svext_s16(zero, svmax_s16_x(pg, prefix, zero), 7);

    if (svptest_any(pg, svcmpgt_s16(pg, v_f, zero))) {
      // No-E correction: update H only. Next column recomputes E from H so
      // we skip writing E here, matching the parasail AVX2 trick that NEON
      // ports for this exact-fill shape.
      for (std::size_t segment = 0; segment < 128U; ++segment) {
        const svint16_t v_h_previous = svld1_s16(pg, h_store + segment * 8);
        const svint16_t v_h_corrected = svmax_s16_x(pg, v_h_previous, v_f);
        svst1_s16(pg, h_store + segment * 8, v_h_corrected);
        best = svmax_s16_x(pg, best, v_h_corrected);
        v_f = svadd_s16_x(pg, v_f, gap_extend);
      }
    }
  }

  return static_cast<Score>(svmaxv_s16(pg, best));
}

// i32 / segment_count=256 variant. Used for width=32 with query_size=1024.
inline Score local_affine_score_exact_fill_i32_256_sve(
    farrar_scalable_kernel::detail::PreparedAffineScoreState<std::int32_t>& state,
    std::span<const std::size_t> target_profile_offsets) {
  if (state.query_size != 1024U || state.segment_count != 256U ||
      target_profile_offsets.empty() || state.gap_open_score > state.gap_extend_score ||
      state.gap_extend_score > 0) {
    return 0;
  }

  std::fill(state.h_store.begin(), state.h_store.end(), std::int32_t{0});
  std::fill(state.h_load.begin(), state.h_load.end(), std::int32_t{0});
  std::fill(state.e_store.begin(), state.e_store.end(), std::int32_t{0});

  const svbool_t pg = svptrue_b32();
  const svint32_t zero = svdup_n_s32(0);
  const svint32_t gap_open = svdup_n_s32(state.gap_open_score);
  const svint32_t gap_extend = svdup_n_s32(state.gap_extend_score);
  const auto lane_gap = static_cast<std::int32_t>(
      static_cast<Score>(256) * static_cast<Score>(state.gap_extend_score));
  const svint32_t lane_gap_1 = svdup_n_s32(lane_gap);
  const svint32_t lane_gap_2 = svdup_n_s32(
      static_cast<std::int32_t>(static_cast<Score>(lane_gap) * 2));
  svint32_t best = zero;
  std::int32_t* h_store = state.h_store.data();
  std::int32_t* h_load = state.h_load.data();
  std::int32_t* e_store = state.e_store.data();
  const std::int32_t* profile_cells = state.profile.data();

  for (const auto profile_offset : target_profile_offsets) {
    std::swap(h_store, h_load);

    svint32_t v_h = svext_s32(zero, svld1_s32(pg, h_load + 255 * 4), 3);
    svint32_t v_f = zero;
    const std::int32_t* profile_row = profile_cells + profile_offset;

    for (std::size_t segment = 0; segment < 256U; ++segment) {
      const svint32_t v_profile = svld1_s32(pg, profile_row + segment * 4);
      svint32_t v_e = svld1_s32(pg, e_store + segment * 4);
      v_h = svadd_s32_x(pg, v_h, v_profile);
      v_h = svmax_s32_x(pg, v_h, v_e);
      v_h = svmax_s32_x(pg, v_h, v_f);
      v_h = svmax_s32_x(pg, v_h, zero);
      svst1_s32(pg, h_store + segment * 4, v_h);
      best = svmax_s32_x(pg, best, v_h);

      const svint32_t v_h_open = svadd_s32_x(pg, v_h, gap_open);
      v_e = svmax_s32_x(pg, svadd_s32_x(pg, v_e, gap_extend), v_h_open);
      svst1_s32(pg, e_store + segment * 4, v_e);
      v_f = svmax_s32_x(pg, svadd_s32_x(pg, v_f, gap_extend), v_h_open);
      v_h = svld1_s32(pg, h_load + segment * 4);
    }

    // 2-stage log-step prefix carry across the 4 i32 lanes.
    svint32_t prefix = svmax_s32_x(pg, v_f, zero);
    svint32_t shifted =
        svadd_s32_x(pg, svext_s32(zero, prefix, 3), lane_gap_1);
    prefix = svmax_s32_x(pg, prefix, shifted);
    shifted = svadd_s32_x(pg, svext_s32(zero, prefix, 2), lane_gap_2);
    prefix = svmax_s32_x(pg, prefix, shifted);
    v_f = svext_s32(zero, svmax_s32_x(pg, prefix, zero), 3);

    if (svptest_any(pg, svcmpgt_s32(pg, v_f, zero))) {
      for (std::size_t segment = 0; segment < 256U; ++segment) {
        const svint32_t v_h_previous = svld1_s32(pg, h_store + segment * 4);
        const svint32_t v_h_corrected = svmax_s32_x(pg, v_h_previous, v_f);
        svst1_s32(pg, h_store + segment * 4, v_h_corrected);
        best = svmax_s32_x(pg, best, v_h_corrected);
        v_f = svadd_s32_x(pg, v_f, gap_extend);
      }
    }
  }

  return static_cast<Score>(svmaxv_s32(pg, best));
}
#endif

template <typename Token, typename Cell>
struct SimdOps;

template <>
struct SimdOps<std::uint8_t, std::int8_t> {
  using vector_type = svint8_t;
  static constexpr bool has_vector_max = true;

#if defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS > 0
  // When the SVE vector length is pinned at compile time (via
  // -msve-vector-bits=N) lane_count becomes a constexpr. The scalable kernel
  // can then unroll inner loops and treat lane offsets as compile-time
  // values, matching the NEON fixed kernel's codegen.
  static constexpr std::size_t lane_count() {
    return static_cast<std::size_t>(__ARM_FEATURE_SVE_BITS) / 8U;
  }
#else
  static std::size_t lane_count() {
    return svcntb();
  }
#endif

  // Callers always operate on full register width (kernels pad their scratch
  // buffers to lane_count()), so use svptrue_b8() instead of recomputing
  // svwhilelt_b8(0, count) on every call. The full predicate lets the
  // compiler fold the mask and schedule the inner loop more aggressively.
  static svbool_t predicate(std::size_t /*count*/) {
    return svptrue_b8();
  }

  static svuint8_t load_tokens(const std::uint8_t* values, std::size_t) {
    return svld1_u8(svptrue_b8(), values);
  }

  static vector_type load_cells(const std::int8_t* values, std::size_t) {
    return svld1_s8(svptrue_b8(), values);
  }

  static void store_cells(std::int8_t* values, vector_type vector, std::size_t) {
    svst1_s8(svptrue_b8(), values, vector);
  }

  static vector_type set1(std::int8_t value, std::size_t) {
    return svdup_n_s8(value);
  }

  static vector_type zero(std::size_t) {
    return svdup_n_s8(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs, std::size_t) {
    return svadd_s8_x(svptrue_b8(), lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs, std::size_t) {
    return svmax_s8_x(svptrue_b8(), lhs, rhs);
  }

  static std::int8_t reduce_max(vector_type vector, std::size_t) {
    return svmaxv_s8(svptrue_b8(), vector);
  }

  static bool any_gt(vector_type lhs, vector_type rhs, std::size_t) {
    const svbool_t pg = svptrue_b8();
    return svptest_any(pg, svcmpgt_s8(pg, lhs, rhs));
  }

  static vector_type substitution(
      const std::uint8_t* query,
      const std::uint8_t* target,
      std::int8_t match_score,
      std::int8_t mismatch_score,
      std::size_t) {
    const svbool_t pg = svptrue_b8();
    const svbool_t mask = svcmpeq_u8(pg, svld1_u8(pg, query), svld1_u8(pg, target));
    return svsel_s8(mask, svdup_n_s8(match_score), svdup_n_s8(mismatch_score));
  }

  static vector_type shift_left_zero(vector_type vector, std::size_t) {
    return svinsr_n_s8(vector, 0);
  }

  static vector_type shift_left_insert(vector_type vector, std::int8_t inserted, std::size_t) {
    return svinsr_n_s8(vector, inserted);
  }

  static vector_type first_lane_vector(std::int8_t first, std::int8_t rest, std::size_t) {
    return svinsr_n_s8(svdup_n_s8(rest), first);
  }

  static vector_type add_sentinel(
      vector_type lhs,
      vector_type rhs,
      std::int8_t sentinel,
      std::size_t) {
    const svbool_t pg = svptrue_b8();
    const svbool_t keep = svcmpeq_n_s8(pg, lhs, sentinel);
    return svsel_s8(keep, lhs, svadd_s8_x(pg, lhs, rhs));
  }
};

template <>
struct SimdOps<std::uint16_t, std::int16_t> {
  using vector_type = svint16_t;
  static constexpr bool has_vector_max = true;

#if defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS > 0
  static constexpr std::size_t lane_count() {
    return static_cast<std::size_t>(__ARM_FEATURE_SVE_BITS) / 16U;
  }
#else
  static std::size_t lane_count() {
    return svcnth();
  }
#endif

  static svbool_t predicate(std::size_t /*count*/) {
    return svptrue_b16();
  }

  static svuint16_t load_tokens(const std::uint16_t* values, std::size_t) {
    return svld1_u16(svptrue_b16(), values);
  }

  static vector_type load_cells(const std::int16_t* values, std::size_t) {
    return svld1_s16(svptrue_b16(), values);
  }

  static void store_cells(std::int16_t* values, vector_type vector, std::size_t) {
    svst1_s16(svptrue_b16(), values, vector);
  }

  static vector_type set1(std::int16_t value, std::size_t) {
    return svdup_n_s16(value);
  }

  static vector_type zero(std::size_t) {
    return svdup_n_s16(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs, std::size_t) {
    return svadd_s16_x(svptrue_b16(), lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs, std::size_t) {
    return svmax_s16_x(svptrue_b16(), lhs, rhs);
  }

  static std::int16_t reduce_max(vector_type vector, std::size_t) {
    return svmaxv_s16(svptrue_b16(), vector);
  }

  static bool any_gt(vector_type lhs, vector_type rhs, std::size_t) {
    const svbool_t pg = svptrue_b16();
    return svptest_any(pg, svcmpgt_s16(pg, lhs, rhs));
  }

  static vector_type substitution(
      const std::uint16_t* query,
      const std::uint16_t* target,
      std::int16_t match_score,
      std::int16_t mismatch_score,
      std::size_t) {
    const svbool_t pg = svptrue_b16();
    const svbool_t mask = svcmpeq_u16(pg, svld1_u16(pg, query), svld1_u16(pg, target));
    return svsel_s16(mask, svdup_n_s16(match_score), svdup_n_s16(mismatch_score));
  }

  static vector_type shift_left_zero(vector_type vector, std::size_t) {
    return svinsr_n_s16(vector, 0);
  }

  static vector_type shift_left_insert(vector_type vector, std::int16_t inserted, std::size_t) {
    return svinsr_n_s16(vector, inserted);
  }

  static vector_type first_lane_vector(std::int16_t first, std::int16_t rest, std::size_t) {
    return svinsr_n_s16(svdup_n_s16(rest), first);
  }

  static vector_type add_sentinel(
      vector_type lhs,
      vector_type rhs,
      std::int16_t sentinel,
      std::size_t) {
    const svbool_t pg = svptrue_b16();
    const svbool_t keep = svcmpeq_n_s16(pg, lhs, sentinel);
    return svsel_s16(keep, lhs, svadd_s16_x(pg, lhs, rhs));
  }

#if defined(__ARM_FEATURE_SVE2)
  // SVE2-only saturating signed add. For rhs <= 0 (gap costs) and INT_MIN as
  // the affine sentinel, svqadd does the right thing in a single instruction:
  // INT_MIN + (<=0) saturates to INT_MIN (sentinel preserved); real + (<=0)
  // stays in range so no spurious saturation. Replaces the 3-op cmpeq+add+sel
  // dance of add_sentinel for the gap-add call sites. Vector-vector svqadd
  // is the headline SVE2 feature SVE1 lacks (SVE1 only has the predicated
  // scalar-broadcast form).
  static vector_type saturating_add(vector_type lhs, vector_type rhs) {
    return svqadd_s16(lhs, rhs);
  }
#endif

#if defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS == 128
  // Multi-lane left shift used by the lazy-F prefix-carry. Only defined
  // when SVE_BITS is pinned because svext_*'s third argument must be an
  // immediate, which requires the lane count to be a compile-time constant.
  template <std::size_t Lanes>
  static vector_type shift_left_lanes_zero(vector_type vector) {
    static_assert(Lanes >= 0 && Lanes <= 8);
    if constexpr (Lanes == 0) {
      return vector;
    } else {
      return svext_s16(svdup_n_s16(0), vector, 8 - Lanes);
    }
  }

  template <std::size_t Lanes>
  static vector_type shift_left_lanes_insert_sentinel(
      vector_type vector,
      std::int16_t fill) {
    static_assert(Lanes >= 0 && Lanes <= 8);
    if constexpr (Lanes == 0) {
      return vector;
    } else {
      return svext_s16(svdup_n_s16(fill), vector, 8 - Lanes);
    }
  }

  static vector_type local_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int16_t gap_score) {
    const auto span_gap = static_cast<std::int16_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_score));
    const auto zero_vector = svdup_n_s16(0);
    auto prefix = svmax_s16_x(svptrue_b16(), final_f, zero_vector);
    auto shifted = svadd_s16_x(
        svptrue_b16(),
        shift_left_lanes_zero<1>(prefix),
        svdup_n_s16(span_gap));
    prefix = svmax_s16_x(svptrue_b16(), prefix, shifted);
    shifted = svadd_s16_x(
        svptrue_b16(),
        shift_left_lanes_zero<2>(prefix),
        svdup_n_s16(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 2)));
    prefix = svmax_s16_x(svptrue_b16(), prefix, shifted);
    shifted = svadd_s16_x(
        svptrue_b16(),
        shift_left_lanes_zero<4>(prefix),
        svdup_n_s16(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 4)));
    prefix = svmax_s16_x(svptrue_b16(), prefix, shifted);
    return shift_left_lanes_zero<1>(svmax_s16_x(svptrue_b16(), prefix, zero_vector));
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
        svdup_n_s16(span_gap),
        low_score,
        0);
    prefix = svmax_s16_x(svptrue_b16(), prefix, shifted);
    shifted = add_sentinel(
        shift_left_lanes_insert_sentinel<2>(prefix, low_score),
        svdup_n_s16(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 2)),
        low_score,
        0);
    prefix = svmax_s16_x(svptrue_b16(), prefix, shifted);
    shifted = add_sentinel(
        shift_left_lanes_insert_sentinel<4>(prefix, low_score),
        svdup_n_s16(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 4)),
        low_score,
        0);
    prefix = svmax_s16_x(svptrue_b16(), prefix, shifted);
    return shift_left_lanes_insert_sentinel<1>(prefix, low_score);
  }

  static Score local_affine_score_exact_segment128_raw(
      farrar_scalable_kernel::detail::PreparedAffineScoreState<std::int16_t>& state,
      std::span<const std::size_t> target_profile_offsets) {
    return local_affine_score_exact_fill_i16_128_sve(state, target_profile_offsets);
  }
#endif
};

template <>
struct SimdOps<std::uint32_t, std::int32_t> {
  using vector_type = svint32_t;
  static constexpr bool has_vector_max = true;

#if defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS > 0
  static constexpr std::size_t lane_count() {
    return static_cast<std::size_t>(__ARM_FEATURE_SVE_BITS) / 32U;
  }
#else
  static std::size_t lane_count() {
    return svcntw();
  }
#endif

  static svbool_t predicate(std::size_t /*count*/) {
    return svptrue_b32();
  }

  static svuint32_t load_tokens(const std::uint32_t* values, std::size_t) {
    return svld1_u32(svptrue_b32(), values);
  }

  static vector_type load_cells(const std::int32_t* values, std::size_t) {
    return svld1_s32(svptrue_b32(), values);
  }

  static void store_cells(std::int32_t* values, vector_type vector, std::size_t) {
    svst1_s32(svptrue_b32(), values, vector);
  }

  static vector_type set1(std::int32_t value, std::size_t) {
    return svdup_n_s32(value);
  }

  static vector_type zero(std::size_t) {
    return svdup_n_s32(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs, std::size_t) {
    return svadd_s32_x(svptrue_b32(), lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs, std::size_t) {
    return svmax_s32_x(svptrue_b32(), lhs, rhs);
  }

  static std::int32_t reduce_max(vector_type vector, std::size_t) {
    return svmaxv_s32(svptrue_b32(), vector);
  }

  static bool any_gt(vector_type lhs, vector_type rhs, std::size_t) {
    const svbool_t pg = svptrue_b32();
    return svptest_any(pg, svcmpgt_s32(pg, lhs, rhs));
  }

  static vector_type substitution(
      const std::uint32_t* query,
      const std::uint32_t* target,
      std::int32_t match_score,
      std::int32_t mismatch_score,
      std::size_t) {
    const svbool_t pg = svptrue_b32();
    const svbool_t mask = svcmpeq_u32(pg, svld1_u32(pg, query), svld1_u32(pg, target));
    return svsel_s32(mask, svdup_n_s32(match_score), svdup_n_s32(mismatch_score));
  }

  static vector_type shift_left_zero(vector_type vector, std::size_t) {
    return svinsr_n_s32(vector, 0);
  }

  static vector_type shift_left_insert(vector_type vector, std::int32_t inserted, std::size_t) {
    return svinsr_n_s32(vector, inserted);
  }

  static vector_type first_lane_vector(std::int32_t first, std::int32_t rest, std::size_t) {
    return svinsr_n_s32(svdup_n_s32(rest), first);
  }

  static vector_type add_sentinel(
      vector_type lhs,
      vector_type rhs,
      std::int32_t sentinel,
      std::size_t) {
    const svbool_t pg = svptrue_b32();
    const svbool_t keep = svcmpeq_n_s32(pg, lhs, sentinel);
    return svsel_s32(keep, lhs, svadd_s32_x(pg, lhs, rhs));
  }

#if defined(__ARM_FEATURE_SVE2)
  static vector_type saturating_add(vector_type lhs, vector_type rhs) {
    return svqadd_s32(lhs, rhs);
  }
#endif

#if defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS == 128
  template <std::size_t Lanes>
  static vector_type shift_left_lanes_zero(vector_type vector) {
    static_assert(Lanes >= 0 && Lanes <= 4);
    if constexpr (Lanes == 0) {
      return vector;
    } else {
      return svext_s32(svdup_n_s32(0), vector, 4 - Lanes);
    }
  }

  template <std::size_t Lanes>
  static vector_type shift_left_lanes_insert_sentinel(
      vector_type vector,
      std::int32_t fill) {
    static_assert(Lanes >= 0 && Lanes <= 4);
    if constexpr (Lanes == 0) {
      return vector;
    } else {
      return svext_s32(svdup_n_s32(fill), vector, 4 - Lanes);
    }
  }

  static vector_type local_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int32_t gap_score) {
    const auto span_gap = static_cast<std::int32_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_score));
    const auto zero_vector = svdup_n_s32(0);
    auto prefix = svmax_s32_x(svptrue_b32(), final_f, zero_vector);
    auto shifted = svadd_s32_x(
        svptrue_b32(),
        shift_left_lanes_zero<1>(prefix),
        svdup_n_s32(span_gap));
    prefix = svmax_s32_x(svptrue_b32(), prefix, shifted);
    shifted = svadd_s32_x(
        svptrue_b32(),
        shift_left_lanes_zero<2>(prefix),
        svdup_n_s32(static_cast<std::int32_t>(static_cast<Score>(span_gap) * 2)));
    prefix = svmax_s32_x(svptrue_b32(), prefix, shifted);
    return shift_left_lanes_zero<1>(svmax_s32_x(svptrue_b32(), prefix, zero_vector));
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
        svdup_n_s32(span_gap),
        low_score,
        0);
    prefix = svmax_s32_x(svptrue_b32(), prefix, shifted);
    shifted = add_sentinel(
        shift_left_lanes_insert_sentinel<2>(prefix, low_score),
        svdup_n_s32(static_cast<std::int32_t>(static_cast<Score>(span_gap) * 2)),
        low_score,
        0);
    prefix = svmax_s32_x(svptrue_b32(), prefix, shifted);
    return shift_left_lanes_insert_sentinel<1>(prefix, low_score);
  }

  static Score local_affine_score_exact_segment256_raw(
      farrar_scalable_kernel::detail::PreparedAffineScoreState<std::int32_t>& state,
      std::span<const std::size_t> target_profile_offsets) {
    return local_affine_score_exact_fill_i32_256_sve(state, target_profile_offsets);
  }
#endif
};

template <>
struct SimdOps<std::uint64_t, std::int64_t> {
  using vector_type = svint64_t;
  static constexpr bool has_vector_max = true;

#if defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS > 0
  static constexpr std::size_t lane_count() {
    return static_cast<std::size_t>(__ARM_FEATURE_SVE_BITS) / 64U;
  }
#else
  static std::size_t lane_count() {
    return svcntd();
  }
#endif

  static svbool_t predicate(std::size_t /*count*/) {
    return svptrue_b64();
  }

  static svuint64_t load_tokens(const std::uint64_t* values, std::size_t) {
    return svld1_u64(svptrue_b64(), values);
  }

  static vector_type load_cells(const std::int64_t* values, std::size_t) {
    return svld1_s64(svptrue_b64(), values);
  }

  static void store_cells(std::int64_t* values, vector_type vector, std::size_t) {
    svst1_s64(svptrue_b64(), values, vector);
  }

  static vector_type set1(std::int64_t value, std::size_t) {
    return svdup_n_s64(value);
  }

  static vector_type zero(std::size_t) {
    return svdup_n_s64(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs, std::size_t) {
    return svadd_s64_x(svptrue_b64(), lhs, rhs);
  }

  static vector_type max(vector_type lhs, vector_type rhs, std::size_t) {
    return svmax_s64_x(svptrue_b64(), lhs, rhs);
  }

  static std::int64_t reduce_max(vector_type vector, std::size_t) {
    return svmaxv_s64(svptrue_b64(), vector);
  }

  static bool any_gt(vector_type lhs, vector_type rhs, std::size_t) {
    const svbool_t pg = svptrue_b64();
    return svptest_any(pg, svcmpgt_s64(pg, lhs, rhs));
  }

  static vector_type substitution(
      const std::uint64_t* query,
      const std::uint64_t* target,
      std::int64_t match_score,
      std::int64_t mismatch_score,
      std::size_t) {
    const svbool_t pg = svptrue_b64();
    const svbool_t mask = svcmpeq_u64(pg, svld1_u64(pg, query), svld1_u64(pg, target));
    return svsel_s64(mask, svdup_n_s64(match_score), svdup_n_s64(mismatch_score));
  }

  static vector_type shift_left_zero(vector_type vector, std::size_t) {
    return svinsr_n_s64(vector, 0);
  }

  static vector_type shift_left_insert(vector_type vector, std::int64_t inserted, std::size_t) {
    return svinsr_n_s64(vector, inserted);
  }

  static vector_type first_lane_vector(std::int64_t first, std::int64_t rest, std::size_t) {
    return svinsr_n_s64(svdup_n_s64(rest), first);
  }

  static vector_type add_sentinel(
      vector_type lhs,
      vector_type rhs,
      std::int64_t sentinel,
      std::size_t) {
    const svbool_t pg = svptrue_b64();
    const svbool_t keep = svcmpeq_n_s64(pg, lhs, sentinel);
    return svsel_s64(keep, lhs, svadd_s64_x(pg, lhs, rhs));
  }
};

struct TargetImplementation {
  using PreparedSmithWatermanFarrarScore =
      farrar_scalable_kernel::detail::PreparedScore<SimdOps>;
  using PreparedAffineScore =
      farrar_scalable_kernel::detail::PreparedAffineScore<SimdOps>;

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
      return arm_sve_kernel::detail::dispatch_score<SimdOps, true>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    }
    const auto prepared =
        prepare_linear_score_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return farrar_scalable_kernel::detail::dispatch_score<SimdOps>(
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
    return profile_traceback::linear_path_info<true>(
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
    return farrar_scalable_kernel::detail::dispatch_score<SimdOps>(
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
    return farrar_scalable_kernel::detail::prepare_score<SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static Score smith_waterman_farrar_score_prepared(
      PreparedSmithWatermanFarrarScore& prepared) {
    return farrar_scalable_kernel::detail::dispatch_prepared_score<SimdOps>(prepared);
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
      return affine_scalable_kernel::detail::dispatch_compact_byte_score<SimdOps>(
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
    return profile_traceback::affine_path<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static AlignmentPath smith_waterman_affine_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    // Compute the score with the fast Farrar SIMD path first, then feed it
    // into the traceback so it can prune the search. Matches the NEON pattern
    // in arm_neon128_backend and avoids a redundant full DP traceback.
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
    return affine_scalable_kernel::detail::dispatch_compact_byte_score<SimdOps>(
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
    return farrar_scalable_kernel::detail::prepare_affine_score<SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static Score smith_waterman_affine_score_prepared(PreparedAffineScore& prepared) {
    return farrar_scalable_kernel::detail::dispatch_prepared_affine_score<SimdOps>(prepared);
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
    return farrar_scalable_kernel::detail::prepare_affine_score<SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static Score needleman_wunsch_affine_score_prepared(PreparedAffineScore& prepared) {
    return farrar_scalable_kernel::detail::dispatch_prepared_global_affine_score<SimdOps>(prepared);
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
      return arm_sve_kernel::detail::dispatch_score<SimdOps, false>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    }
    const auto prepared =
        prepare_linear_score_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return farrar_scalable_kernel::detail::dispatch_global_score<SimdOps>(
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
    return farrar_scalable_kernel::detail::dispatch_global_affine_score<SimdOps>(
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
    return profile_traceback::affine_path<false>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
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

}  // namespace stride_align::arm_sve_backend
