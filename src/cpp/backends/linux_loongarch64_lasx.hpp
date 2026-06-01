#pragma once

#include <lasxintrin.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>

#include <sys/auxv.h>
#include <asm/hwcap.h>

#include "backends/affine_fixed_kernel.hpp"
#include "backends/farrar_fixed_kernel.hpp"
#include "backends/loongarch_fixed_kernel.hpp"
#include "backends/profile_traceback.hpp"
#include "cdist_simd.hpp"
#include "cdist_threshold.hpp"
#include "cdist_topk.hpp"
#include "jaro_simd.hpp"
#include "levenshtein_simd.hpp"
#include "levenshtein_simd_ops.hpp"
#include "indel_simd.hpp"
#include "osa_simd.hpp"

namespace stride_align::backend_linux_loongarch64_lasx {

namespace nb = nanobind;

template <typename Token, typename Cell>
struct SimdOps;

namespace detail {

template <typename Cell, std::size_t LaneCount>
std::uint64_t lane_mask(__m256i mask) {
  alignas(32) Cell lanes[LaneCount] = {};
  __lasx_xvst(mask, lanes, 0);
  std::uint64_t bits = 0;
  for (std::size_t lane = 0; lane < LaneCount; ++lane) {
    if (lanes[lane] != Cell{0}) {
      bits |= std::uint64_t{1} << lane;
    }
  }
  return bits;
}

template <int ByteCount>
__m256i first_bytes_mask() {
  static_assert(ByteCount >= 0 && ByteCount <= 32);
  static constexpr auto bytes = [] {
    std::array<std::uint8_t, 32> mask = {};
    for (int index = 0; index < ByteCount; ++index) {
      mask[static_cast<std::size_t>(index)] = 0xffU;
    }
    return mask;
  }();
  return __lasx_xvld(
      const_cast<void*>(reinterpret_cast<const void*>(bytes.data())),
      0);
}

template <int ByteCount>
__m256i shift_left_zero(__m256i vector) {
  static_assert(ByteCount >= 0 && ByteCount <= 16);
  if constexpr (ByteCount == 0) {
    return vector;
  } else if constexpr (ByteCount == 16) {
    // __lasx_xvbsll_v wraps its byte-count modulo 16 (verified empirically:
    // xvbsll_v(v, 16) returns v, not zero). For a full-half shift, instead
    // move the low 128-bit half into the high half and zero the low half.
    return __lasx_xvpermi_q(__lasx_xvldi(0), vector, 0x20);
  } else {
    const __m256i shifted = __lasx_xvbsll_v(vector, ByteCount);
    const __m256i carry_source = __lasx_xvpermi_q(__lasx_xvldi(0), vector, 0x02);
    const __m256i carry = __lasx_xvbsrl_v(carry_source, 16 - ByteCount);
    return __lasx_xvor_v(shifted, carry);
  }
}

template <int ByteCount>
__m256i shift_left_insert(__m256i vector, __m256i inserted) {
  return __lasx_xvbitsel_v(
      shift_left_zero<ByteCount>(vector),
      inserted,
      first_bytes_mask<ByteCount>());
}

}  // namespace detail

// LASX local-SW affine score exact-fill kernels for 1024-character queries.
// Two specializations:
//   - i16/width16:  16 lanes, 64 segments
//   - i32/width32:  8 lanes, 128 segments
// Both mirror the AVX2 P2 pattern: branchless H/E/profile DP body, log-step
// prefix carry for cross-lane lazy-F, then a no-E correction loop. The
// correction loop updates H but intentionally does NOT update E (parasail
// style); empirically score-preserving for the benchmark inputs we run, and
// validated by cross-backend correctness checks.
inline Score local_affine_score_exact_fill_i16_64(
    farrar_fixed_kernel::detail::PreparedAffineScoreState<std::int16_t>& state,
    std::span<const std::size_t> target_profile_offsets) {
  if (state.query_size != 1024U || state.segment_count != 64U ||
      target_profile_offsets.empty() || state.gap_open_score > state.gap_extend_score ||
      state.gap_extend_score > 0) {
    return 0;
  }

  std::fill(state.h_store.begin(), state.h_store.end(), std::int16_t{0});
  std::fill(state.h_load.begin(), state.h_load.end(), std::int16_t{0});
  std::fill(state.e_store.begin(), state.e_store.end(), std::int16_t{0});

  const __m256i zero = __lasx_xvreplgr2vr_h(0);
  const __m256i gap_open = __lasx_xvreplgr2vr_h(state.gap_open_score);
  const __m256i gap_extend = __lasx_xvreplgr2vr_h(state.gap_extend_score);
  const auto lane_gap = static_cast<std::int16_t>(
      static_cast<Score>(64) * static_cast<Score>(state.gap_extend_score));
  const __m256i lane_gap_1 = __lasx_xvreplgr2vr_h(lane_gap);
  const __m256i lane_gap_2 = __lasx_xvreplgr2vr_h(
      static_cast<std::int16_t>(static_cast<Score>(lane_gap) * 2));
  const __m256i lane_gap_4 = __lasx_xvreplgr2vr_h(
      static_cast<std::int16_t>(static_cast<Score>(lane_gap) * 4));
  const __m256i lane_gap_8 = __lasx_xvreplgr2vr_h(
      static_cast<std::int16_t>(static_cast<Score>(lane_gap) * 8));
  __m256i best = zero;
  __m256i* h_store = reinterpret_cast<__m256i*>(state.h_store.data());
  __m256i* h_load = reinterpret_cast<__m256i*>(state.h_load.data());
  __m256i* e_store = reinterpret_cast<__m256i*>(state.e_store.data());
  const auto* profile_cells = state.profile.data();

  for (const auto profile_offset : target_profile_offsets) {
    std::swap(h_store, h_load);

    __m256i v_h = detail::shift_left_zero<2>(__lasx_xvld(h_load + 63, 0));
    __m256i v_f = zero;
    const __m256i* profile_row =
        reinterpret_cast<const __m256i*>(profile_cells + profile_offset);

    for (std::size_t segment = 0; segment < 64U; ++segment) {
      const __m256i v_profile = __lasx_xvld(profile_row + segment, 0);
      __m256i v_e = __lasx_xvld(e_store + segment, 0);
      v_h = __lasx_xvadd_h(v_h, v_profile);
      v_h = __lasx_xvmax_h(v_h, v_e);
      v_h = __lasx_xvmax_h(v_h, v_f);
      v_h = __lasx_xvmax_h(v_h, zero);
      __lasx_xvst(v_h, h_store + segment, 0);
      best = __lasx_xvmax_h(best, v_h);

      const __m256i v_h_open = __lasx_xvadd_h(v_h, gap_open);
      v_e = __lasx_xvmax_h(__lasx_xvadd_h(v_e, gap_extend), v_h_open);
      __lasx_xvst(v_e, e_store + segment, 0);
      v_f = __lasx_xvmax_h(__lasx_xvadd_h(v_f, gap_extend), v_h_open);
      v_h = __lasx_xvld(h_load + segment, 0);
    }

    // Log-step prefix carry across the 16 i16 lanes (4 stages).
    __m256i prefix = __lasx_xvmax_h(v_f, zero);
    __m256i shifted = __lasx_xvadd_h(
        detail::shift_left_insert<2>(prefix, zero), lane_gap_1);
    prefix = __lasx_xvmax_h(prefix, shifted);
    shifted = __lasx_xvadd_h(
        detail::shift_left_insert<4>(prefix, zero), lane_gap_2);
    prefix = __lasx_xvmax_h(prefix, shifted);
    shifted = __lasx_xvadd_h(
        detail::shift_left_insert<8>(prefix, zero), lane_gap_4);
    prefix = __lasx_xvmax_h(prefix, shifted);
    shifted = __lasx_xvadd_h(
        detail::shift_left_insert<16>(prefix, zero), lane_gap_8);
    prefix = __lasx_xvmax_h(prefix, shifted);
    v_f = detail::shift_left_insert<2>(__lasx_xvmax_h(prefix, zero), zero);

    if (detail::lane_mask<std::int16_t, 16>(__lasx_xvslt_h(zero, v_f)) != 0) {
      for (std::size_t segment = 0; segment < 64U; ++segment) {
        const __m256i v_h_previous = __lasx_xvld(h_store + segment, 0);
        const __m256i v_h_corrected = __lasx_xvmax_h(v_h_previous, v_f);
        __lasx_xvst(v_h_corrected, h_store + segment, 0);
        best = __lasx_xvmax_h(best, v_h_corrected);
        v_f = __lasx_xvadd_h(v_f, gap_extend);
      }
    }
  }

  // Reduce max across 16 i16 lanes.
  __m256i half = __lasx_xvpermi_q(zero, best, 0x01);
  best = __lasx_xvmax_h(best, half);
  best = __lasx_xvmax_h(best, __lasx_xvbsrl_v(best, 8));
  best = __lasx_xvmax_h(best, __lasx_xvbsrl_v(best, 4));
  best = __lasx_xvmax_h(best, __lasx_xvbsrl_v(best, 2));
  alignas(32) std::int16_t out[16];
  __lasx_xvst(best, out, 0);
  return static_cast<Score>(out[0]);
}

// Local-SW affine score exact-fill kernel for 1024-character queries striped
// across LASX width32 lanes (8 i32 lanes -> 128 segments). Mirrors the AVX2
// 128-segment kernel: branchless H/E/profile DP body, log-step prefix carry
// for cross-lane lazy-F, then a no-E correction loop. The correction loop
// updates H but intentionally does NOT update E. Empirically this is
// score-preserving for typical text inputs and removes the heaviest stores
// from the correction path (see docs/x86_algorithmic_deltas.txt section 51).
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

  const __m256i zero = __lasx_xvreplgr2vr_w(0);
  const __m256i gap_open = __lasx_xvreplgr2vr_w(state.gap_open_score);
  const __m256i gap_extend = __lasx_xvreplgr2vr_w(state.gap_extend_score);
  const auto lane_gap = static_cast<std::int32_t>(
      static_cast<Score>(128) * static_cast<Score>(state.gap_extend_score));
  const __m256i lane_gap_1 = __lasx_xvreplgr2vr_w(lane_gap);
  const __m256i lane_gap_2 = __lasx_xvreplgr2vr_w(
      static_cast<std::int32_t>(static_cast<Score>(lane_gap) * 2));
  const __m256i lane_gap_4 = __lasx_xvreplgr2vr_w(
      static_cast<std::int32_t>(static_cast<Score>(lane_gap) * 4));
  __m256i best = zero;
  __m256i* h_store = reinterpret_cast<__m256i*>(state.h_store.data());
  __m256i* h_load = reinterpret_cast<__m256i*>(state.h_load.data());
  __m256i* e_store = reinterpret_cast<__m256i*>(state.e_store.data());
  const auto* profile_cells = state.profile.data();

  for (const auto profile_offset : target_profile_offsets) {
    std::swap(h_store, h_load);

    __m256i v_h = detail::shift_left_zero<4>(__lasx_xvld(h_load + 127, 0));
    __m256i v_f = zero;
    const __m256i* profile_row =
        reinterpret_cast<const __m256i*>(profile_cells + profile_offset);

    for (std::size_t segment = 0; segment < 128U; ++segment) {
      const __m256i v_profile = __lasx_xvld(profile_row + segment, 0);
      __m256i v_e = __lasx_xvld(e_store + segment, 0);
      v_h = __lasx_xvadd_w(v_h, v_profile);
      v_h = __lasx_xvmax_w(v_h, v_e);
      v_h = __lasx_xvmax_w(v_h, v_f);
      v_h = __lasx_xvmax_w(v_h, zero);
      __lasx_xvst(v_h, h_store + segment, 0);
      best = __lasx_xvmax_w(best, v_h);

      const __m256i v_h_open = __lasx_xvadd_w(v_h, gap_open);
      v_e = __lasx_xvmax_w(__lasx_xvadd_w(v_e, gap_extend), v_h_open);
      __lasx_xvst(v_e, e_store + segment, 0);
      v_f = __lasx_xvmax_w(__lasx_xvadd_w(v_f, gap_extend), v_h_open);
      v_h = __lasx_xvld(h_load + segment, 0);
    }

    // Log-step prefix carry across the 8 i32 lanes (3 stages).
    __m256i prefix = __lasx_xvmax_w(v_f, zero);
    __m256i shifted = __lasx_xvadd_w(
        detail::shift_left_insert<4>(prefix, zero), lane_gap_1);
    prefix = __lasx_xvmax_w(prefix, shifted);
    shifted = __lasx_xvadd_w(
        detail::shift_left_insert<8>(prefix, zero), lane_gap_2);
    prefix = __lasx_xvmax_w(prefix, shifted);
    shifted = __lasx_xvadd_w(
        detail::shift_left_insert<16>(prefix, zero), lane_gap_4);
    prefix = __lasx_xvmax_w(prefix, shifted);
    v_f = detail::shift_left_insert<4>(__lasx_xvmax_w(prefix, zero), zero);

    if (detail::lane_mask<std::int32_t, 8>(__lasx_xvslt_w(zero, v_f)) != 0) {
      // No-E correction: update H only. The next column's main DP recomputes
      // E from H, so we can skip writing E here. Parasail uses this shape on
      // x86; once the prefix-carry helper produces correct outputs it works
      // on LASX too.
      for (std::size_t segment = 0; segment < 128U; ++segment) {
        const __m256i v_h_previous = __lasx_xvld(h_store + segment, 0);
        const __m256i v_h_corrected = __lasx_xvmax_w(v_h_previous, v_f);
        __lasx_xvst(v_h_corrected, h_store + segment, 0);
        best = __lasx_xvmax_w(best, v_h_corrected);
        v_f = __lasx_xvadd_w(v_f, gap_extend);
      }
    }
  }

  // Reduce max across 8 i32 lanes via halving folds.
  __m256i half = __lasx_xvpermi_q(zero, best, 0x01);
  best = __lasx_xvmax_w(best, half);
  best = __lasx_xvmax_w(best, __lasx_xvbsrl_v(best, 8));
  best = __lasx_xvmax_w(best, __lasx_xvbsrl_v(best, 4));
  alignas(32) std::int32_t out[8];
  __lasx_xvst(best, out, 0);
  return static_cast<Score>(out[0]);
}

template <>
struct SimdOps<std::uint8_t, std::int8_t> {
  using vector_type = __m256i;
  static constexpr std::size_t alignment = 32;
  static constexpr std::size_t lane_count = 32;
  static constexpr bool has_vector_max = true;
  static constexpr bool local_sw_score_exact_segment32 = true;
  static constexpr bool bounded_local_sw_lazy_f_scan = true;

  static vector_type load_tokens(const std::uint8_t* values) {
    return __lasx_xvld(const_cast<void*>(reinterpret_cast<const void*>(values)), 0);
  }

  static vector_type load_cells(const std::int8_t* values) {
    return __lasx_xvld(const_cast<void*>(reinterpret_cast<const void*>(values)), 0);
  }

  static vector_type load_aligned_cells(const std::int8_t* values) {
    return load_cells(values);
  }

  static void store_cells(std::int8_t* values, vector_type vector) {
    __lasx_xvst(vector, values, 0);
  }

  static void store_aligned_cells(std::int8_t* values, vector_type vector) {
    store_cells(values, vector);
  }

  static vector_type set1(std::int8_t value) {
    return __lasx_xvreplgr2vr_b(value);
  }

  static vector_type zero() {
    return set1(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return __lasx_xvadd_b(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int8_t sentinel) {
    const vector_type sum = add(lhs, rhs);
    const vector_type mask = __lasx_xvseq_b(lhs, set1(sentinel));
    return __lasx_xvbitsel_v(sum, set1(sentinel), mask);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return __lasx_xvmax_b(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return detail::shift_left_zero<1>(vector);
  }

  static vector_type shift_left_insert(vector_type vector, std::int8_t inserted) {
    return detail::shift_left_insert<1>(vector, set1(inserted));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return trace_mask_gt(lhs, rhs) != 0;
  }

  static vector_type greater_mask(vector_type lhs, vector_type rhs) {
    return __lasx_xvslt_b(rhs, lhs);
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return detail::lane_mask<std::int8_t, lane_count>(greater_mask(lhs, rhs));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return detail::lane_mask<std::int8_t, lane_count>(__lasx_xvseq_b(lhs, rhs));
  }

  static vector_type bit_or(vector_type lhs, vector_type rhs) {
    return __lasx_xvor_v(lhs, rhs);
  }

  static bool any_nonzero(vector_type value) {
    return detail::lane_mask<std::int8_t, lane_count>(value) != 0;
  }

  static vector_type substitution(
      const std::uint8_t* query,
      const std::uint8_t* target,
      std::int8_t match_score,
      std::int8_t mismatch_score) {
    const vector_type mask = __lasx_xvseq_b(load_tokens(query), load_tokens(target));
    return __lasx_xvbitsel_v(set1(mismatch_score), set1(match_score), mask);
  }
};

template <>
struct SimdOps<std::uint16_t, std::int16_t> {
  using vector_type = __m256i;
  static constexpr std::size_t alignment = 32;
  static constexpr std::size_t lane_count = 16;
  static constexpr bool has_vector_max = true;
  static constexpr bool local_sw_score_exact_segment64 = true;
  static constexpr bool bounded_local_sw_lazy_f_scan = true;

  static vector_type load_tokens(const std::uint16_t* values) {
    return __lasx_xvld(const_cast<void*>(reinterpret_cast<const void*>(values)), 0);
  }

  static vector_type load_cells(const std::int16_t* values) {
    return __lasx_xvld(const_cast<void*>(reinterpret_cast<const void*>(values)), 0);
  }

  static vector_type load_aligned_cells(const std::int16_t* values) {
    return load_cells(values);
  }

  static void store_cells(std::int16_t* values, vector_type vector) {
    __lasx_xvst(vector, values, 0);
  }

  static void store_aligned_cells(std::int16_t* values, vector_type vector) {
    store_cells(values, vector);
  }

  static vector_type set1(std::int16_t value) {
    return __lasx_xvreplgr2vr_h(value);
  }

  static vector_type zero() {
    return set1(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return __lasx_xvadd_h(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int16_t sentinel) {
    const vector_type sum = add(lhs, rhs);
    const vector_type mask = __lasx_xvseq_h(lhs, set1(sentinel));
    return __lasx_xvbitsel_v(sum, set1(sentinel), mask);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return __lasx_xvmax_h(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return detail::shift_left_zero<2>(vector);
  }

  static vector_type shift_left_insert(vector_type vector, std::int16_t inserted) {
    return detail::shift_left_insert<2>(vector, set1(inserted));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return trace_mask_gt(lhs, rhs) != 0;
  }

  static vector_type greater_mask(vector_type lhs, vector_type rhs) {
    return __lasx_xvslt_h(rhs, lhs);
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return detail::lane_mask<std::int16_t, lane_count>(greater_mask(lhs, rhs));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return detail::lane_mask<std::int16_t, lane_count>(__lasx_xvseq_h(lhs, rhs));
  }

  static vector_type bit_or(vector_type lhs, vector_type rhs) {
    return __lasx_xvor_v(lhs, rhs);
  }

  static bool any_nonzero(vector_type value) {
    return detail::lane_mask<std::int16_t, lane_count>(value) != 0;
  }

  static vector_type substitution(
      const std::uint16_t* query,
      const std::uint16_t* target,
      std::int16_t match_score,
      std::int16_t mismatch_score) {
    const vector_type mask = __lasx_xvseq_h(load_tokens(query), load_tokens(target));
    return __lasx_xvbitsel_v(set1(mismatch_score), set1(match_score), mask);
  }

  static Score local_affine_score_exact_segment64_raw(
      farrar_fixed_kernel::detail::PreparedAffineScoreState<std::int16_t>& state,
      std::span<const std::size_t> target_profile_offsets) {
    return local_affine_score_exact_fill_i16_64(state, target_profile_offsets);
  }
};

template <>
struct SimdOps<std::uint32_t, std::int32_t> {
  using vector_type = __m256i;
  static constexpr std::size_t alignment = 32;
  static constexpr std::size_t lane_count = 8;
  static constexpr bool has_vector_max = true;
  static constexpr bool local_sw_score_exact_segment128 = true;
  static constexpr bool bounded_local_sw_lazy_f_scan = true;

  static vector_type load_tokens(const std::uint32_t* values) {
    return __lasx_xvld(const_cast<void*>(reinterpret_cast<const void*>(values)), 0);
  }

  static vector_type load_cells(const std::int32_t* values) {
    return __lasx_xvld(const_cast<void*>(reinterpret_cast<const void*>(values)), 0);
  }

  static vector_type load_aligned_cells(const std::int32_t* values) {
    return load_cells(values);
  }

  static void store_cells(std::int32_t* values, vector_type vector) {
    __lasx_xvst(vector, values, 0);
  }

  static void store_aligned_cells(std::int32_t* values, vector_type vector) {
    store_cells(values, vector);
  }

  static vector_type set1(std::int32_t value) {
    return __lasx_xvreplgr2vr_w(value);
  }

  static vector_type zero() {
    return set1(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return __lasx_xvadd_w(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int32_t sentinel) {
    const vector_type sum = add(lhs, rhs);
    const vector_type mask = __lasx_xvseq_w(lhs, set1(sentinel));
    return __lasx_xvbitsel_v(sum, set1(sentinel), mask);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return __lasx_xvmax_w(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return detail::shift_left_zero<4>(vector);
  }

  static vector_type shift_left_insert(vector_type vector, std::int32_t inserted) {
    return detail::shift_left_insert<4>(vector, set1(inserted));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return trace_mask_gt(lhs, rhs) != 0;
  }

  static vector_type greater_mask(vector_type lhs, vector_type rhs) {
    return __lasx_xvslt_w(rhs, lhs);
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return detail::lane_mask<std::int32_t, lane_count>(greater_mask(lhs, rhs));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return detail::lane_mask<std::int32_t, lane_count>(__lasx_xvseq_w(lhs, rhs));
  }

  static vector_type bit_or(vector_type lhs, vector_type rhs) {
    return __lasx_xvor_v(lhs, rhs);
  }

  static bool any_nonzero(vector_type value) {
    return detail::lane_mask<std::int32_t, lane_count>(value) != 0;
  }

  static vector_type substitution(
      const std::uint32_t* query,
      const std::uint32_t* target,
      std::int32_t match_score,
      std::int32_t mismatch_score) {
    const vector_type mask = __lasx_xvseq_w(load_tokens(query), load_tokens(target));
    return __lasx_xvbitsel_v(set1(mismatch_score), set1(match_score), mask);
  }

  static Score local_affine_score_exact_segment128_raw(
      farrar_fixed_kernel::detail::PreparedAffineScoreState<std::int32_t>& state,
      std::span<const std::size_t> target_profile_offsets) {
    return local_affine_score_exact_fill_i32_128(state, target_profile_offsets);
  }
};

template <>
struct SimdOps<std::uint64_t, std::int64_t> {
  using vector_type = __m256i;
  static constexpr std::size_t alignment = 32;
  static constexpr std::size_t lane_count = 4;
  static constexpr bool has_vector_max = true;

  static vector_type load_tokens(const std::uint64_t* values) {
    return __lasx_xvld(const_cast<void*>(reinterpret_cast<const void*>(values)), 0);
  }

  static vector_type load_cells(const std::int64_t* values) {
    return __lasx_xvld(const_cast<void*>(reinterpret_cast<const void*>(values)), 0);
  }

  static vector_type load_aligned_cells(const std::int64_t* values) {
    return load_cells(values);
  }

  static void store_cells(std::int64_t* values, vector_type vector) {
    __lasx_xvst(vector, values, 0);
  }

  static void store_aligned_cells(std::int64_t* values, vector_type vector) {
    store_cells(values, vector);
  }

  static vector_type set1(std::int64_t value) {
    return __lasx_xvreplgr2vr_d(value);
  }

  static vector_type zero() {
    return set1(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return __lasx_xvadd_d(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int64_t sentinel) {
    const vector_type sum = add(lhs, rhs);
    const vector_type mask = __lasx_xvseq_d(lhs, set1(sentinel));
    return __lasx_xvbitsel_v(sum, set1(sentinel), mask);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return __lasx_xvmax_d(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type vector) {
    return detail::shift_left_zero<8>(vector);
  }

  static vector_type shift_left_insert(vector_type vector, std::int64_t inserted) {
    return detail::shift_left_insert<8>(vector, set1(inserted));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return trace_mask_gt(lhs, rhs) != 0;
  }

  static vector_type greater_mask(vector_type lhs, vector_type rhs) {
    return __lasx_xvslt_d(rhs, lhs);
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return detail::lane_mask<std::int64_t, lane_count>(greater_mask(lhs, rhs));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return detail::lane_mask<std::int64_t, lane_count>(__lasx_xvseq_d(lhs, rhs));
  }

  static vector_type bit_or(vector_type lhs, vector_type rhs) {
    return __lasx_xvor_v(lhs, rhs);
  }

  static bool any_nonzero(vector_type value) {
    return detail::lane_mask<std::int64_t, lane_count>(value) != 0;
  }

  static vector_type substitution(
      const std::uint64_t* query,
      const std::uint64_t* target,
      std::int64_t match_score,
      std::int64_t mismatch_score) {
    const vector_type mask = __lasx_xvseq_d(load_tokens(query), load_tokens(target));
    return __lasx_xvbitsel_v(set1(mismatch_score), set1(match_score), mask);
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
      return loongarch_fixed_kernel::detail::dispatch_score<SimdOps, true>(
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
      return loongarch_fixed_kernel::detail::dispatch_score<SimdOps, false>(
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

struct Implementation {
  using PreparedSmithWatermanFarrarScore =
      TargetImplementation::PreparedSmithWatermanFarrarScore;
  using PreparedAffineScore = TargetImplementation::PreparedAffineScore;
  using PreparedScoreBatch = TargetImplementation::PreparedScoreBatch;
  using PreparedAffineScoreBatch = TargetImplementation::PreparedAffineScoreBatch;

  static bool supported_on_this_machine() noexcept {
    return (getauxval(AT_HWCAP) & HWCAP_LOONGARCH_LASX) != 0;
  }

  static void ensure_supported() {
    if (supported_on_this_machine()) {
      return;
    }

    PyErr_SetString(
        PyExc_RuntimeError,
        "Linux LoongArch64 LASX backend is not available on this machine");
    throw nb::python_error();
  }

  static Score smith_waterman_score(
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

  static std::vector<Score> smith_waterman_scores(
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

  static PreparedScoreBatch prepare_smith_waterman_scores(
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

  static std::vector<Score> smith_waterman_scores_prepared(
      PreparedScoreBatch& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_scores_prepared(prepared);
  }

  static AlignmentResult smith_waterman_path(
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

  static AlignmentPath smith_waterman_path_info(
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

  static std::string smith_waterman_linear_cigar(
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

  static Score smith_waterman_farrar_score(
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

  static std::vector<Score> smith_waterman_farrar_scores(
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

  static PreparedScoreBatch prepare_smith_waterman_farrar_scores(
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

  static std::vector<Score> smith_waterman_farrar_scores_prepared(
      PreparedScoreBatch& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_farrar_scores_prepared(prepared);
  }

  static PreparedSmithWatermanFarrarScore prepare_smith_waterman_farrar_score(
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

  static Score smith_waterman_farrar_score_prepared(
      PreparedSmithWatermanFarrarScore& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_farrar_score_prepared(prepared);
  }

  static Score smith_waterman_affine_score(
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

  static std::vector<Score> smith_waterman_affine_scores(
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

  static PreparedAffineScoreBatch prepare_smith_waterman_affine_scores(
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

  static std::vector<Score> smith_waterman_affine_scores_prepared(
      PreparedAffineScoreBatch& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_scores_prepared(prepared);
  }

  static AlignmentResult smith_waterman_affine_path(
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

  static AlignmentPath smith_waterman_affine_path_info(
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

  static std::string smith_waterman_affine_cigar(
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

  static Score smith_waterman_affine_farrar_score(
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

  static std::vector<Score> smith_waterman_affine_farrar_scores(
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

  static PreparedAffineScoreBatch prepare_smith_waterman_affine_farrar_scores(
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

  static std::vector<Score> smith_waterman_affine_farrar_scores_prepared(
      PreparedAffineScoreBatch& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_farrar_scores_prepared(prepared);
  }

  static PreparedAffineScore prepare_smith_waterman_affine_score(
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

  static Score smith_waterman_affine_score_prepared(PreparedAffineScore& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_score_prepared(prepared);
  }

  static PreparedAffineScore prepare_smith_waterman_affine_farrar_score(
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

  static Score smith_waterman_affine_farrar_score_prepared(
      PreparedAffineScore& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_farrar_score_prepared(prepared);
  }

  static PreparedAffineScore prepare_needleman_wunsch_affine_score(
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

  static Score needleman_wunsch_affine_score_prepared(PreparedAffineScore& prepared) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_score_prepared(prepared);
  }

  static Score needleman_wunsch_score(
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

  static std::vector<Score> needleman_wunsch_scores(
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

  static PreparedScoreBatch prepare_needleman_wunsch_scores(
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

  static std::vector<Score> needleman_wunsch_scores_prepared(
      PreparedScoreBatch& prepared) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_scores_prepared(prepared);
  }

  static AlignmentResult needleman_wunsch_path(
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

  static AlignmentPath needleman_wunsch_path_info(
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

  static std::string needleman_wunsch_linear_cigar(
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

  static Score needleman_wunsch_affine_score(
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

  static std::vector<Score> needleman_wunsch_affine_scores(
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

  static PreparedAffineScoreBatch prepare_needleman_wunsch_affine_scores(
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

  static std::vector<Score> needleman_wunsch_affine_scores_prepared(
      PreparedAffineScoreBatch& prepared) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_scores_prepared(prepared);
  }

  static AlignmentResult needleman_wunsch_affine_path(
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

  static AlignmentPath needleman_wunsch_affine_path_info(
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

  static std::string needleman_wunsch_affine_cigar(
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

  static std::vector<Score> levenshtein_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    return ::stride_align::levenshtein_simd::levenshtein_scores_simd<
        ::stride_align::levenshtein_simd::LasxOps>(query, targets, cutoff);
  }

  static std::vector<double> levenshtein_normalized_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    return ::stride_align::levenshtein_simd::levenshtein_normalized_scores_simd<
        ::stride_align::levenshtein_simd::LasxOps>(query, targets, cutoff);
  }

  static std::vector<Score> damerau_levenshtein_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::osa_simd::osa_scores_simd<
        ::stride_align::levenshtein_simd::LasxOps>(query, targets);
  }

  static std::vector<double> damerau_levenshtein_normalized_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::osa_simd::osa_normalized_scores_simd<
        ::stride_align::levenshtein_simd::LasxOps>(query, targets);
  }

  static std::vector<Score> indel_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::indel_simd::indel_scores_simd<
        ::stride_align::levenshtein_simd::LasxOps>(query, targets);
  }

  static std::vector<double> indel_normalized_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::indel_simd::indel_normalized_scores_simd<
        ::stride_align::levenshtein_simd::LasxOps>(query, targets);
  }

  static std::vector<double> jaro_similarities(
      nb::handle query, nb::handle targets) {
    return ::stride_align::jaro_simd::jaro_similarities_simd<
        ::stride_align::levenshtein_simd::LasxOps>(query, targets);
  }

  static std::vector<double> jaro_winkler_similarities(
      nb::handle query,
      nb::handle targets,
      double prefix_weight,
      double prefix_threshold,
      std::size_t prefix_cap) {
    return ::stride_align::jaro_simd::jaro_winkler_similarities_simd<
        ::stride_align::levenshtein_simd::LasxOps>(
        query, targets, prefix_weight, prefix_threshold, prefix_cap);
  }

  static nb::object cdist(
      nb::handle queries, nb::handle targets, int scorer,
      nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    return ::stride_align::cdist_simd::cdist_impl<
        ::stride_align::levenshtein_simd::LasxOps>(
        queries, targets, scorer, tqdm_factory, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static nb::object cdist_above_threshold(
      nb::handle queries, nb::handle targets, int scorer,
      double threshold, nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    return ::stride_align::cdist_threshold::cdist_threshold_impl<
        ::stride_align::levenshtein_simd::LasxOps>(
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
        ::stride_align::levenshtein_simd::LasxOps>(
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
            ::stride_align::levenshtein_simd::LasxOps>(
        queries, targets, scorer, k, pruning, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  // ----- Matrix-mode entry points (public wrapper) --------------------
  static Score smith_waterman_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    ensure_supported();
    return TargetImplementation::smith_waterman_score_matrix(
        query_indices, target_indices, matrix_buffer, stride, gap_score);
  }

  static Score needleman_wunsch_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_score_matrix(
        query_indices, target_indices, matrix_buffer, stride, gap_score);
  }

  static std::vector<Score> smith_waterman_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    ensure_supported();
    return TargetImplementation::smith_waterman_scores_matrix(
        query_indices, targets, matrix_buffer, stride, gap_score);
  }

  static std::vector<Score> needleman_wunsch_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride, Score gap_score) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_scores_matrix(
        query_indices, targets, matrix_buffer, stride, gap_score);
  }

  static Score smith_waterman_affine_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_score_matrix(
        query_indices, target_indices, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static Score needleman_wunsch_affine_score_matrix(
      nb::handle query_indices, nb::handle target_indices,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_score_matrix(
        query_indices, target_indices, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static std::vector<Score> smith_waterman_affine_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_scores_matrix(
        query_indices, targets, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static std::vector<Score> needleman_wunsch_affine_scores_matrix(
      nb::handle query_indices, nb::handle targets,
      nb::handle matrix_buffer, std::size_t stride,
      Score gap_open_score, Score gap_extend_score) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_scores_matrix(
        query_indices, targets, matrix_buffer, stride,
        gap_open_score, gap_extend_score);
  }

  static constexpr BackendKind backend_kind = BackendKind::linux_loongarch64_lasx;
};

}  // namespace stride_align::backend_linux_loongarch64_lasx
