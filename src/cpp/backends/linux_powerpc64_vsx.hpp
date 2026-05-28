#pragma once

#include <altivec.h>
#undef bool
#undef pixel
#undef vector

#include <sys/auxv.h>
#include <asm/cputable.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include <nanobind/nanobind.h>

#include <stride_align/alignment.hpp>

#include "backends/affine_fixed_kernel.hpp"
#include "backends/farrar_fixed_kernel.hpp"
#include "backends/powerpc_vsx_kernel.hpp"
#include "backends/profile_traceback.hpp"
#include "cdist_simd.hpp"
#include "cdist_threshold.hpp"
#include "cdist_topk.hpp"
#include "jaro_simd.hpp"
#include "levenshtein_simd.hpp"
#include "levenshtein_simd_ops.hpp"
#include "indel_simd.hpp"
#include "osa_simd.hpp"

namespace stride_align::backend_linux_powerpc64_vsx {

namespace nb = nanobind;

// Power8 VSX on ppc64le notes:
//
//   * Lane numbering on Linux ppc64le is little-endian: lane 0 is at the
//     lowest memory address. The GCC altivec builtins largely hide the BE
//     hardware semantics, but a few cross-lane operations (vec_sld in
//     particular) still expose the BE Programming Interface Manual model.
//     We rely on vec_perm for cross-lane shifts because GCC normalizes its
//     byte indices into LE-natural order on ppc64le, which makes the code
//     match the SSE/NEON/LSX equivalents one-to-one.
//
//   * vec_max for __vector signed long long is a Power9 (ISA 3.0) instruction.
//     On Power8 we fall back to vec_cmpgt + vec_sel (only 2 lanes wide, so
//     the cost is small).
//
//   * vec_adds (saturating) exists for 8 and 16 bit only. We do not rely on
//     saturating add in the kernels; add_sentinel emulates the only place
//     where the existing Ops contract uses saturation-like behavior.
//
//   * vec_cmpgt for signed long long requires ISA 2.07 (Power8), so it works
//     here. trace_mask_* uses a vector store + scalar fan-out. Power8 has
//     vbpermq (one-instruction bit gather) that could collapse a per-lane
//     compare result into a packed bitmap with control bytes [127, 119, 111,
//     ..., 15, 7] for LE-natural lane order, but we measured the trace-mask
//     step at ~4% of the masked-traceback runtime on 1024x1024 English, so
//     it's not a bottleneck worth carrying complexity for right now.

namespace detail {

template <int ByteShift>
inline __vector unsigned char shift_left_perm() {
  static_assert(ByteShift >= 0 && ByteShift <= 16);
  // result_LE[i] for i <  ByteShift -> b_byte[i]      (perm index 16 + i)
  // result_LE[i] for i >= ByteShift -> a_byte[i-N]    (perm index i - ByteShift)
  //
  // Using 16+i in the low-half is important when `b` carries a multi-byte
  // value (e.g. set1(int16_t) packs two-byte pairs). Using just 16 would
  // splat byte 0 of b across every low result byte, dropping the upper
  // bytes of the inserted scalar.
  __vector unsigned char perm;
  for (int i = 0; i < 16; ++i) {
    if (i < ByteShift) {
      reinterpret_cast<unsigned char*>(&perm)[i] =
          static_cast<unsigned char>(16 + i);
    } else {
      reinterpret_cast<unsigned char*>(&perm)[i] =
          static_cast<unsigned char>(i - ByteShift);
    }
  }
  return perm;
}

template <int ByteShift, typename Vec>
inline Vec shift_left_zero_bytes(Vec v) {
  if constexpr (ByteShift == 0) {
    return v;
  } else {
    const __vector unsigned char z = vec_splats(static_cast<unsigned char>(0));
    const __vector unsigned char perm = shift_left_perm<ByteShift>();
    return reinterpret_cast<Vec>(
        vec_perm(reinterpret_cast<__vector unsigned char>(v), z, perm));
  }
}

template <int ByteShift, typename Vec>
inline Vec shift_left_insert_bytes(Vec v, Vec inserted) {
  if constexpr (ByteShift == 0) {
    return v;
  } else {
    const __vector unsigned char perm = shift_left_perm<ByteShift>();
    return reinterpret_cast<Vec>(vec_perm(
        reinterpret_cast<__vector unsigned char>(v),
        reinterpret_cast<__vector unsigned char>(inserted),
        perm));
  }
}

template <typename Cell, std::size_t LaneCount, typename Vec>
inline std::uint64_t collapse_to_lane_bits(Vec mask) {
  alignas(16) Cell lanes[LaneCount];
  vec_xst(mask, 0, reinterpret_cast<Vec*>(lanes));
  std::uint64_t bits = 0;
  for (std::size_t lane = 0; lane < LaneCount; ++lane) {
    if (lanes[lane] != Cell{0}) {
      bits |= std::uint64_t{1} << lane;
    }
  }
  return bits;
}

inline bool vsx_any_nonzero_128(__vector unsigned long long v) {
  // Two 64-bit lanes; OR them and test.
  return (vec_extract(v, 0) | vec_extract(v, 1)) != 0ULL;
}

}  // namespace detail

template <typename Token, typename Cell>
struct SimdOps;

// ---- 8-bit lanes (16 lanes / vector) ----
template <>
struct SimdOps<std::uint8_t, std::int8_t> {
  using token_vector_type = __vector unsigned char;
  using vector_type = __vector signed char;
  using mask_type = __vector signed char;  // 0x00 / 0xFF per byte lane
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 16;
  static constexpr bool has_vector_max = true;
  static constexpr bool bounded_local_sw_lazy_f_scan = true;

  static token_vector_type load_tokens(const std::uint8_t* values) {
    return vec_xl(0, values);
  }

  static vector_type load_cells(const std::int8_t* values) {
    return vec_xl(0, values);
  }

  static vector_type load_aligned_cells(const std::int8_t* values) {
    return vec_ld(0, reinterpret_cast<const vector_type*>(values));
  }

  static void store_cells(std::int8_t* values, vector_type v) {
    vec_xst(v, 0, values);
  }

  static void store_aligned_cells(std::int8_t* values, vector_type v) {
    vec_st(v, 0, reinterpret_cast<vector_type*>(values));
  }

  static vector_type set1(std::int8_t value) {
    return vec_splats(static_cast<signed char>(value));
  }

  static vector_type zero() {
    return set1(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return vec_add(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int8_t sentinel) {
    const vector_type sum = vec_add(lhs, rhs);
    const __vector __bool char hit = vec_cmpeq(lhs, set1(sentinel));
    return vec_sel(sum, set1(sentinel), hit);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return vec_max(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type v) {
    return detail::shift_left_zero_bytes<1>(v);
  }

  static vector_type shift_left_insert(vector_type v, std::int8_t inserted) {
    return detail::shift_left_insert_bytes<1>(v, set1(inserted));
  }

  template <int Lanes>
  static vector_type shift_left_lanes_zero(vector_type v) {
    static_assert(Lanes >= 0 && Lanes <= 16);
    return detail::shift_left_zero_bytes<Lanes>(v);
  }

  template <int Lanes>
  static vector_type shift_left_lanes_insert_sentinel(vector_type v, std::int8_t fill) {
    static_assert(Lanes >= 0 && Lanes <= 16);
    return detail::shift_left_insert_bytes<Lanes>(v, set1(fill));
  }

  static vector_type local_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int8_t gap_score) {
    const auto span_gap = static_cast<std::int8_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_score));
    const auto z = zero();
    auto prefix = max(final_f, z);
    auto shifted = add(detail::shift_left_zero_bytes<1>(prefix), set1(span_gap));
    prefix = max(prefix, shifted);
    shifted = add(
        detail::shift_left_zero_bytes<2>(prefix),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 2)));
    prefix = max(prefix, shifted);
    shifted = add(
        detail::shift_left_zero_bytes<4>(prefix),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 4)));
    prefix = max(prefix, shifted);
    shifted = add(
        detail::shift_left_zero_bytes<8>(prefix),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 8)));
    prefix = max(prefix, shifted);
    return detail::shift_left_zero_bytes<1>(max(prefix, z));
  }

  static vector_type global_lazy_f_prefix_carry_no_padding(
      vector_type final_f,
      std::size_t segment_count,
      std::int8_t gap_extend_score,
      std::int8_t low_score) {
    const auto span_gap = static_cast<std::int8_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_extend_score));
    auto prefix = final_f;
    auto shifted = add_sentinel(
        detail::shift_left_insert_bytes<1>(prefix, set1(low_score)),
        set1(span_gap),
        low_score);
    prefix = max(prefix, shifted);
    shifted = add_sentinel(
        detail::shift_left_insert_bytes<2>(prefix, set1(low_score)),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 2)),
        low_score);
    prefix = max(prefix, shifted);
    shifted = add_sentinel(
        detail::shift_left_insert_bytes<4>(prefix, set1(low_score)),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 4)),
        low_score);
    prefix = max(prefix, shifted);
    shifted = add_sentinel(
        detail::shift_left_insert_bytes<8>(prefix, set1(low_score)),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 8)),
        low_score);
    prefix = max(prefix, shifted);
    return detail::shift_left_insert_bytes<1>(prefix, set1(low_score));
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return reinterpret_cast<mask_type>(vec_cmpgt(lhs, rhs));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return vec_any_gt(lhs, rhs) != 0;
  }

  static mask_type empty_mask() {
    return zero();
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return vec_or(lhs, rhs);
  }

  static bool any_mask(mask_type mask) {
    return detail::vsx_any_nonzero_128(
        reinterpret_cast<__vector unsigned long long>(mask));
  }

  static vector_type bit_or(vector_type lhs, vector_type rhs) {
    return vec_or(lhs, rhs);
  }

  static bool any_nonzero(vector_type value) {
    return detail::vsx_any_nonzero_128(
        reinterpret_cast<__vector unsigned long long>(value));
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return detail::collapse_to_lane_bits<std::int8_t, lane_count>(
        reinterpret_cast<vector_type>(vec_cmpgt(lhs, rhs)));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return detail::collapse_to_lane_bits<std::int8_t, lane_count>(
        reinterpret_cast<vector_type>(vec_cmpeq(lhs, rhs)));
  }

  static std::int8_t reduce_max(vector_type v) {
    // 16 -> 8: shift right 8 bytes (using LE-natural perm) then max.
    auto half = reinterpret_cast<vector_type>(
        vec_sld(reinterpret_cast<__vector unsigned char>(zero()),
                reinterpret_cast<__vector unsigned char>(v),
                8));
    auto reduced = vec_max(v, half);
    // 8 -> 4: shift right 4 bytes
    half = reinterpret_cast<vector_type>(
        vec_sld(reinterpret_cast<__vector unsigned char>(zero()),
                reinterpret_cast<__vector unsigned char>(reduced),
                12));
    reduced = vec_max(reduced, half);
    // 4 -> 2: shift right 2 bytes
    half = reinterpret_cast<vector_type>(
        vec_sld(reinterpret_cast<__vector unsigned char>(zero()),
                reinterpret_cast<__vector unsigned char>(reduced),
                14));
    reduced = vec_max(reduced, half);
    // 2 -> 1: shift right 1 byte
    half = reinterpret_cast<vector_type>(
        vec_sld(reinterpret_cast<__vector unsigned char>(zero()),
                reinterpret_cast<__vector unsigned char>(reduced),
                15));
    reduced = vec_max(reduced, half);
    alignas(16) std::int8_t lanes[16];
    store_aligned_cells(lanes, reduced);
    return lanes[0];
  }

  static void store_masked_cells(std::int8_t* values, vector_type v, mask_type mask) {
    const vector_type previous = load_cells(values);
    const vector_type updated = vec_sel(
        previous,
        v,
        reinterpret_cast<__vector __bool char>(mask));
    store_cells(values, updated);
  }

  static vector_type substitution(
      const std::uint8_t* query,
      const std::uint8_t* target,
      std::int8_t match_score,
      std::int8_t mismatch_score) {
    const __vector __bool char hit =
        vec_cmpeq(load_tokens(query), load_tokens(target));
    return vec_sel(set1(mismatch_score), set1(match_score), hit);
  }
};

// ---- 16-bit lanes (8 lanes / vector) ----
template <>
struct SimdOps<std::uint16_t, std::int16_t> {
  using token_vector_type = __vector unsigned short;
  using vector_type = __vector signed short;
  using mask_type = __vector signed short;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 8;
  static constexpr bool has_vector_max = true;
  static constexpr bool bounded_local_sw_lazy_f_scan = true;

  static token_vector_type load_tokens(const std::uint16_t* values) {
    return vec_xl(0, values);
  }

  static vector_type load_cells(const std::int16_t* values) {
    return vec_xl(0, values);
  }

  static vector_type load_aligned_cells(const std::int16_t* values) {
    return vec_ld(0, reinterpret_cast<const vector_type*>(values));
  }

  static void store_cells(std::int16_t* values, vector_type v) {
    vec_xst(v, 0, values);
  }

  static void store_aligned_cells(std::int16_t* values, vector_type v) {
    vec_st(v, 0, reinterpret_cast<vector_type*>(values));
  }

  static vector_type set1(std::int16_t value) {
    return vec_splats(static_cast<signed short>(value));
  }

  static vector_type zero() {
    return set1(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return vec_add(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int16_t sentinel) {
    const vector_type sum = vec_add(lhs, rhs);
    const __vector __bool short hit = vec_cmpeq(lhs, set1(sentinel));
    return vec_sel(sum, set1(sentinel), hit);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return vec_max(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type v) {
    return detail::shift_left_zero_bytes<2>(v);
  }

  static vector_type shift_left_insert(vector_type v, std::int16_t inserted) {
    return detail::shift_left_insert_bytes<2>(v, set1(inserted));
  }

  template <int Lanes>
  static vector_type shift_left_lanes_zero(vector_type v) {
    static_assert(Lanes >= 0 && Lanes <= 8);
    return detail::shift_left_zero_bytes<2 * Lanes>(v);
  }

  template <int Lanes>
  static vector_type shift_left_lanes_insert_sentinel(vector_type v, std::int16_t fill) {
    static_assert(Lanes >= 0 && Lanes <= 8);
    return detail::shift_left_insert_bytes<2 * Lanes>(v, set1(fill));
  }

  static vector_type local_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int16_t gap_score) {
    const auto span_gap = static_cast<std::int16_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_score));
    const auto z = zero();
    auto prefix = max(final_f, z);
    auto shifted = add(detail::shift_left_zero_bytes<2>(prefix), set1(span_gap));
    prefix = max(prefix, shifted);
    shifted = add(
        detail::shift_left_zero_bytes<4>(prefix),
        set1(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 2)));
    prefix = max(prefix, shifted);
    shifted = add(
        detail::shift_left_zero_bytes<8>(prefix),
        set1(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 4)));
    prefix = max(prefix, shifted);
    return detail::shift_left_zero_bytes<2>(max(prefix, z));
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
        detail::shift_left_insert_bytes<2>(prefix, set1(low_score)),
        set1(span_gap),
        low_score);
    prefix = max(prefix, shifted);
    shifted = add_sentinel(
        detail::shift_left_insert_bytes<4>(prefix, set1(low_score)),
        set1(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 2)),
        low_score);
    prefix = max(prefix, shifted);
    shifted = add_sentinel(
        detail::shift_left_insert_bytes<8>(prefix, set1(low_score)),
        set1(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 4)),
        low_score);
    prefix = max(prefix, shifted);
    return detail::shift_left_insert_bytes<2>(prefix, set1(low_score));
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return reinterpret_cast<mask_type>(vec_cmpgt(lhs, rhs));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return vec_any_gt(lhs, rhs) != 0;
  }

  static mask_type empty_mask() {
    return zero();
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return vec_or(lhs, rhs);
  }

  static bool any_mask(mask_type mask) {
    return detail::vsx_any_nonzero_128(
        reinterpret_cast<__vector unsigned long long>(mask));
  }

  static vector_type bit_or(vector_type lhs, vector_type rhs) {
    return vec_or(lhs, rhs);
  }

  static bool any_nonzero(vector_type value) {
    return detail::vsx_any_nonzero_128(
        reinterpret_cast<__vector unsigned long long>(value));
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return detail::collapse_to_lane_bits<std::int16_t, lane_count>(
        reinterpret_cast<vector_type>(vec_cmpgt(lhs, rhs)));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return detail::collapse_to_lane_bits<std::int16_t, lane_count>(
        reinterpret_cast<vector_type>(vec_cmpeq(lhs, rhs)));
  }

  static std::int16_t reduce_max(vector_type v) {
    auto half = reinterpret_cast<vector_type>(
        vec_sld(reinterpret_cast<__vector unsigned char>(zero()),
                reinterpret_cast<__vector unsigned char>(v),
                8));
    auto reduced = vec_max(v, half);
    half = reinterpret_cast<vector_type>(
        vec_sld(reinterpret_cast<__vector unsigned char>(zero()),
                reinterpret_cast<__vector unsigned char>(reduced),
                12));
    reduced = vec_max(reduced, half);
    half = reinterpret_cast<vector_type>(
        vec_sld(reinterpret_cast<__vector unsigned char>(zero()),
                reinterpret_cast<__vector unsigned char>(reduced),
                14));
    reduced = vec_max(reduced, half);
    alignas(16) std::int16_t lanes[8];
    store_aligned_cells(lanes, reduced);
    return lanes[0];
  }

  static void store_masked_cells(std::int16_t* values, vector_type v, mask_type mask) {
    const vector_type previous = load_cells(values);
    const vector_type updated = vec_sel(
        previous,
        v,
        reinterpret_cast<__vector __bool short>(mask));
    store_cells(values, updated);
  }

  static vector_type substitution(
      const std::uint16_t* query,
      const std::uint16_t* target,
      std::int16_t match_score,
      std::int16_t mismatch_score) {
    const __vector __bool short hit =
        vec_cmpeq(load_tokens(query), load_tokens(target));
    return vec_sel(set1(mismatch_score), set1(match_score), hit);
  }
};

// ---- 32-bit lanes (4 lanes / vector) ----
template <>
struct SimdOps<std::uint32_t, std::int32_t> {
  using token_vector_type = __vector unsigned int;
  using vector_type = __vector signed int;
  using mask_type = __vector signed int;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 4;
  static constexpr bool has_vector_max = true;
  static constexpr bool bounded_local_sw_lazy_f_scan = true;

  static token_vector_type load_tokens(const std::uint32_t* values) {
    return vec_xl(0, values);
  }

  static vector_type load_cells(const std::int32_t* values) {
    return vec_xl(0, values);
  }

  static vector_type load_aligned_cells(const std::int32_t* values) {
    return vec_ld(0, reinterpret_cast<const vector_type*>(values));
  }

  static void store_cells(std::int32_t* values, vector_type v) {
    vec_xst(v, 0, values);
  }

  static void store_aligned_cells(std::int32_t* values, vector_type v) {
    vec_st(v, 0, reinterpret_cast<vector_type*>(values));
  }

  static vector_type set1(std::int32_t value) {
    return vec_splats(static_cast<signed int>(value));
  }

  static vector_type zero() {
    return set1(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return vec_add(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int32_t sentinel) {
    const vector_type sum = vec_add(lhs, rhs);
    const __vector __bool int hit = vec_cmpeq(lhs, set1(sentinel));
    return vec_sel(sum, set1(sentinel), hit);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    return vec_max(lhs, rhs);
  }

  static vector_type shift_left_zero(vector_type v) {
    return detail::shift_left_zero_bytes<4>(v);
  }

  static vector_type shift_left_insert(vector_type v, std::int32_t inserted) {
    return detail::shift_left_insert_bytes<4>(v, set1(inserted));
  }

  template <int Lanes>
  static vector_type shift_left_lanes_zero(vector_type v) {
    static_assert(Lanes >= 0 && Lanes <= 4);
    return detail::shift_left_zero_bytes<4 * Lanes>(v);
  }

  template <int Lanes>
  static vector_type shift_left_lanes_insert_sentinel(vector_type v, std::int32_t fill) {
    static_assert(Lanes >= 0 && Lanes <= 4);
    return detail::shift_left_insert_bytes<4 * Lanes>(v, set1(fill));
  }

  static vector_type local_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int32_t gap_score) {
    const auto span_gap = static_cast<std::int32_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_score));
    const auto z = zero();
    auto prefix = max(final_f, z);
    auto shifted = add(detail::shift_left_zero_bytes<4>(prefix), set1(span_gap));
    prefix = max(prefix, shifted);
    shifted = add(
        detail::shift_left_zero_bytes<8>(prefix),
        set1(static_cast<std::int32_t>(static_cast<Score>(span_gap) * 2)));
    prefix = max(prefix, shifted);
    return detail::shift_left_zero_bytes<4>(max(prefix, z));
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
        detail::shift_left_insert_bytes<4>(prefix, set1(low_score)),
        set1(span_gap),
        low_score);
    prefix = max(prefix, shifted);
    shifted = add_sentinel(
        detail::shift_left_insert_bytes<8>(prefix, set1(low_score)),
        set1(static_cast<std::int32_t>(static_cast<Score>(span_gap) * 2)),
        low_score);
    prefix = max(prefix, shifted);
    return detail::shift_left_insert_bytes<4>(prefix, set1(low_score));
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return reinterpret_cast<mask_type>(vec_cmpgt(lhs, rhs));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return vec_any_gt(lhs, rhs) != 0;
  }

  static mask_type empty_mask() {
    return zero();
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return vec_or(lhs, rhs);
  }

  static bool any_mask(mask_type mask) {
    return detail::vsx_any_nonzero_128(
        reinterpret_cast<__vector unsigned long long>(mask));
  }

  static vector_type bit_or(vector_type lhs, vector_type rhs) {
    return vec_or(lhs, rhs);
  }

  static bool any_nonzero(vector_type value) {
    return detail::vsx_any_nonzero_128(
        reinterpret_cast<__vector unsigned long long>(value));
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return detail::collapse_to_lane_bits<std::int32_t, lane_count>(
        reinterpret_cast<vector_type>(vec_cmpgt(lhs, rhs)));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return detail::collapse_to_lane_bits<std::int32_t, lane_count>(
        reinterpret_cast<vector_type>(vec_cmpeq(lhs, rhs)));
  }

  static std::int32_t reduce_max(vector_type v) {
    auto half = reinterpret_cast<vector_type>(
        vec_sld(reinterpret_cast<__vector unsigned char>(zero()),
                reinterpret_cast<__vector unsigned char>(v),
                8));
    auto reduced = vec_max(v, half);
    half = reinterpret_cast<vector_type>(
        vec_sld(reinterpret_cast<__vector unsigned char>(zero()),
                reinterpret_cast<__vector unsigned char>(reduced),
                12));
    reduced = vec_max(reduced, half);
    alignas(16) std::int32_t lanes[4];
    store_aligned_cells(lanes, reduced);
    return lanes[0];
  }

  static void store_masked_cells(std::int32_t* values, vector_type v, mask_type mask) {
    const vector_type previous = load_cells(values);
    const vector_type updated = vec_sel(
        previous,
        v,
        reinterpret_cast<__vector __bool int>(mask));
    store_cells(values, updated);
  }

  static vector_type substitution(
      const std::uint32_t* query,
      const std::uint32_t* target,
      std::int32_t match_score,
      std::int32_t mismatch_score) {
    const __vector __bool int hit =
        vec_cmpeq(load_tokens(query), load_tokens(target));
    return vec_sel(set1(mismatch_score), set1(match_score), hit);
  }
};

// ---- 64-bit lanes (2 lanes / vector) ----
// Power8 has signed-long-long vec_cmpgt/vec_cmpeq (ISA 2.07) but no vec_max
// for long long (added in ISA 3.0 / Power9), so max is open-coded.
template <>
struct SimdOps<std::uint64_t, std::int64_t> {
  using token_vector_type = __vector unsigned long long;
  using vector_type = __vector signed long long;
  using mask_type = __vector signed long long;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 2;
  static constexpr bool has_vector_max = true;  // we implement max manually below
  static constexpr bool bounded_local_sw_lazy_f_scan = true;

  static token_vector_type load_tokens(const std::uint64_t* values) {
    return vec_xl(0, reinterpret_cast<const unsigned long long*>(values));
  }

  static vector_type load_cells(const std::int64_t* values) {
    return vec_xl(0, reinterpret_cast<const signed long long*>(values));
  }

  static vector_type load_aligned_cells(const std::int64_t* values) {
    return vec_ld(0, reinterpret_cast<const vector_type*>(values));
  }

  static void store_cells(std::int64_t* values, vector_type v) {
    vec_xst(v, 0, reinterpret_cast<signed long long*>(values));
  }

  static void store_aligned_cells(std::int64_t* values, vector_type v) {
    vec_st(v, 0, reinterpret_cast<vector_type*>(values));
  }

  static vector_type set1(std::int64_t value) {
    return vec_splats(static_cast<signed long long>(value));
  }

  static vector_type zero() {
    return set1(0);
  }

  static vector_type add(vector_type lhs, vector_type rhs) {
    return vec_add(lhs, rhs);
  }

  static vector_type add_sentinel(vector_type lhs, vector_type rhs, std::int64_t sentinel) {
    const vector_type sum = vec_add(lhs, rhs);
    const __vector __bool long long hit = vec_cmpeq(lhs, set1(sentinel));
    return vec_sel(sum, set1(sentinel), hit);
  }

  static vector_type max(vector_type lhs, vector_type rhs) {
    // No native vec_max for signed long long on Power8 (added in Power9).
    // Synthesize via cmpgt + select; both operands of vec_sel must match
    // the lane type, and vec_cmpgt for signed long long is available.
    const __vector __bool long long lhs_is_greater = vec_cmpgt(lhs, rhs);
    return vec_sel(rhs, lhs, lhs_is_greater);
  }

  static vector_type shift_left_zero(vector_type v) {
    return detail::shift_left_zero_bytes<8>(v);
  }

  static vector_type shift_left_insert(vector_type v, std::int64_t inserted) {
    return detail::shift_left_insert_bytes<8>(v, set1(inserted));
  }

  template <int Lanes>
  static vector_type shift_left_lanes_zero(vector_type v) {
    static_assert(Lanes >= 0 && Lanes <= 2);
    return detail::shift_left_zero_bytes<8 * Lanes>(v);
  }

  template <int Lanes>
  static vector_type shift_left_lanes_insert_sentinel(vector_type v, std::int64_t fill) {
    static_assert(Lanes >= 0 && Lanes <= 2);
    return detail::shift_left_insert_bytes<8 * Lanes>(v, set1(fill));
  }

  static vector_type local_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int64_t gap_score) {
    const auto span_gap = static_cast<std::int64_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_score));
    const auto z = zero();
    auto prefix = max(final_f, z);
    auto shifted = add(detail::shift_left_zero_bytes<8>(prefix), set1(span_gap));
    prefix = max(prefix, shifted);
    return detail::shift_left_zero_bytes<8>(max(prefix, z));
  }

  static vector_type global_lazy_f_prefix_carry_no_padding(
      vector_type final_f,
      std::size_t segment_count,
      std::int64_t gap_extend_score,
      std::int64_t low_score) {
    const auto span_gap = static_cast<std::int64_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_extend_score));
    auto prefix = final_f;
    auto shifted = add_sentinel(
        detail::shift_left_insert_bytes<8>(prefix, set1(low_score)),
        set1(span_gap),
        low_score);
    prefix = max(prefix, shifted);
    return detail::shift_left_insert_bytes<8>(prefix, set1(low_score));
  }

  static mask_type greater_mask(vector_type lhs, vector_type rhs) {
    return reinterpret_cast<mask_type>(vec_cmpgt(lhs, rhs));
  }

  static bool any_gt(vector_type lhs, vector_type rhs) {
    return vec_any_gt(lhs, rhs) != 0;
  }

  static mask_type empty_mask() {
    return zero();
  }

  static mask_type mask_or(mask_type lhs, mask_type rhs) {
    return vec_or(lhs, rhs);
  }

  static bool any_mask(mask_type mask) {
    return detail::vsx_any_nonzero_128(
        reinterpret_cast<__vector unsigned long long>(mask));
  }

  static vector_type bit_or(vector_type lhs, vector_type rhs) {
    return vec_or(lhs, rhs);
  }

  static bool any_nonzero(vector_type value) {
    return detail::vsx_any_nonzero_128(
        reinterpret_cast<__vector unsigned long long>(value));
  }

  static std::uint64_t trace_mask_gt(vector_type lhs, vector_type rhs) {
    return detail::collapse_to_lane_bits<std::int64_t, lane_count>(
        reinterpret_cast<vector_type>(vec_cmpgt(lhs, rhs)));
  }

  static std::uint64_t trace_mask_eq(vector_type lhs, vector_type rhs) {
    return detail::collapse_to_lane_bits<std::int64_t, lane_count>(
        reinterpret_cast<vector_type>(vec_cmpeq(lhs, rhs)));
  }

  static std::int64_t reduce_max(vector_type v) {
    const auto half = reinterpret_cast<vector_type>(
        vec_sld(reinterpret_cast<__vector unsigned char>(zero()),
                reinterpret_cast<__vector unsigned char>(v),
                8));
    const auto reduced = max(v, half);
    alignas(16) std::int64_t lanes[2];
    store_aligned_cells(lanes, reduced);
    return lanes[0];
  }

  static void store_masked_cells(std::int64_t* values, vector_type v, mask_type mask) {
    const vector_type previous = load_cells(values);
    const vector_type updated = vec_sel(
        previous,
        v,
        reinterpret_cast<__vector __bool long long>(mask));
    store_cells(values, updated);
  }

  static vector_type substitution(
      const std::uint64_t* query,
      const std::uint64_t* target,
      std::int64_t match_score,
      std::int64_t mismatch_score) {
    const __vector __bool long long hit =
        vec_cmpeq(load_tokens(query), load_tokens(target));
    return vec_sel(set1(mismatch_score), set1(match_score), hit);
  }
};

namespace detail_impl {

template <bool LocalAlignment>
inline Score linear_score(
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
      if constexpr (LocalAlignment) {
        return farrar_fixed_kernel::detail::wide_dispatch_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      } else {
        return farrar_fixed_kernel::detail::wide_dispatch_global_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
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
      if constexpr (LocalAlignment) {
        return farrar_fixed_kernel::detail::wide_dispatch_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      } else {
        return farrar_fixed_kernel::detail::wide_dispatch_global_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
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
      if constexpr (LocalAlignment) {
        return farrar_fixed_kernel::detail::wide_dispatch_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      } else {
        return farrar_fixed_kernel::detail::wide_dispatch_global_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
  }
  // Unicode auto-promote: UCS-2/UCS-4 strings with >256 distinct
  // codepoints route to the 16-bit Farrar kernel via uint16 tokens.
  // Above 65 535 distinct, we fall through to the 32-bit Farrar
  // kernel with raw UCS-4 codepoints as uint32 tokens. UCS-1 strings
  // can have at most 256 distinct codepoints by construction.
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
        if constexpr (LocalAlignment) {
          return farrar_fixed_kernel::detail::wide_dispatch_score<SimdOps>(
              wide, match_score, mismatch_score, gap_score);
        } else {
          return farrar_fixed_kernel::detail::wide_dispatch_global_score<SimdOps>(
              wide, match_score, mismatch_score, gap_score);
        }
      }
      const auto wide =
          ::stride_align::farrar_detail::prepare_farrar_alignment_wide_unicode_uint16(
              query.ptr(), target.ptr(),
              match_score, mismatch_score, gap_score, gap_score, width);
      if constexpr (LocalAlignment) {
        return farrar_fixed_kernel::detail::wide_dispatch_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      } else {
        return farrar_fixed_kernel::detail::wide_dispatch_global_score<SimdOps>(
            wide, match_score, mismatch_score, gap_score);
      }
    }
  }
  if (gap_score > 0) {
    const auto prepared =
        prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return powerpc_vsx_kernel::detail::dispatch_score<SimdOps, LocalAlignment>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }
  const auto prepared =
      prepare_linear_score_alignment(query, target, match_score, mismatch_score, gap_score, width);
  if constexpr (LocalAlignment) {
    return farrar_fixed_kernel::detail::dispatch_score<SimdOps>(
        prepared, match_score, mismatch_score, gap_score);
  } else {
    return farrar_fixed_kernel::detail::dispatch_global_score<SimdOps>(
        prepared, match_score, mismatch_score, gap_score);
  }
}

}  // namespace detail_impl

struct TargetImplementation {
  using PreparedSmithWatermanFarrarScore =
      farrar_fixed_kernel::detail::PreparedScore<SimdOps>;
  using PreparedAffineScore =
      farrar_fixed_kernel::detail::PreparedAffineScore<SimdOps>;

  // ---- Linear Smith-Waterman ----
  static Score smith_waterman_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return detail_impl::linear_score<true>(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static std::vector<Score> smith_waterman_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared = prepare_farrar_batch_alignment(
        query, targets, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::dispatch_score_many<SimdOps, true>(
        prepared, match_score, mismatch_score, gap_score);
  }

  // NOTE: linear SW path / path-info / cigar stay on profile_traceback on
  // Power8. The AVX2/Loongson "masked" striped traceback wins on ISAs with
  // a fast lane-bitmask extract (pmovmskb on x86, __lsx_vpickod on LSX) and
  // a fast random-access scan of the striped trace table. Measured on real
  // POWER8 silicon (KVM-virtualized, 1 core, 4.157 GHz, AT15 GCC 11.4):
  // dispatch_linear_sw_score_first_masked_cigar_* lands at 0.46-0.48x of
  // the scalar byte-table profile_traceback path on 1024x1024 English. The
  // bulk of the regression is the striped trace walk's poor cache locality
  // versus the row-major byte table, not the mask extraction itself. We
  // therefore route linear SW traceback through profile_traceback, which
  // still benefits from the much faster SIMD score kernels for endpoint
  // selection internally.
  static AlignmentResult smith_waterman_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return profile_traceback::linear_path<true>(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static AlignmentPath smith_waterman_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return profile_traceback::linear_path_info<true>(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static std::string smith_waterman_linear_cigar(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return profile_traceback::linear_cigar<true>(
        query, target, match_score, mismatch_score, gap_score, width);
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
        prepared, match_score, mismatch_score, gap_score);
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
        prepared, match_score, mismatch_score, gap_score);
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
        query, targets, match_score, mismatch_score, gap_score, width);
  }

  // ---- Affine Smith-Waterman ----
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

  // NOTE: affine path / path-info also stay on profile_traceback on Power8.
  // The striped affine traceback emits trace bytes one lane at a time inside
  // a vectorized DP loop, which on real POWER8 (KVM-virtualized, 1 core)
  // measured 0.57-0.60x of the scalar byte-table path on 1024x1024 English
  // affine traceback. The affine *CIGAR* entry point below still uses the
  // SIMD score kernel + scalar reverse-build and beats generic by ~4.6x
  // because it skips the trace table entirely.
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
    return profile_traceback::affine_path_info<true>(
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

  // ---- Linear Needleman-Wunsch ----
  static Score needleman_wunsch_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return detail_impl::linear_score<false>(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static std::vector<Score> needleman_wunsch_scores(
      nb::handle query,
      nb::handle targets,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared = prepare_farrar_batch_alignment(
        query, targets, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::dispatch_score_many<SimdOps, false>(
        prepared, match_score, mismatch_score, gap_score);
  }

  static AlignmentResult needleman_wunsch_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return profile_traceback::linear_path<false>(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static AlignmentPath needleman_wunsch_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return profile_traceback::linear_path_info<false>(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static std::string needleman_wunsch_linear_cigar(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return profile_traceback::linear_cigar<false>(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  // ---- Affine Needleman-Wunsch ----
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
    return profile_traceback::affine_path_info<false>(
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

  static bool supported_on_this_machine() noexcept {
    return (getauxval(AT_HWCAP) & PPC_FEATURE_HAS_VSX) != 0;
  }

  static void ensure_supported() {
    if (supported_on_this_machine()) {
      return;
    }
    PyErr_SetString(
        PyExc_RuntimeError,
        "Linux PowerPC64 VSX backend is not available on this machine");
    throw nb::python_error();
  }

#define STRIDE_ALIGN_VSX_FWD(METHOD, ...)                                  \
  ensure_supported();                                                      \
  return TargetImplementation::METHOD(__VA_ARGS__)

  static Score smith_waterman_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(smith_waterman_score,
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static std::vector<Score> smith_waterman_scores(
      nb::handle query, nb::handle targets,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(smith_waterman_scores,
        query, targets, match_score, mismatch_score, gap_score, width);
  }

  static AlignmentResult smith_waterman_path(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(smith_waterman_path,
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static AlignmentPath smith_waterman_path_info(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(smith_waterman_path_info,
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static std::string smith_waterman_linear_cigar(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(smith_waterman_linear_cigar,
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static Score smith_waterman_farrar_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(smith_waterman_farrar_score,
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static PreparedSmithWatermanFarrarScore prepare_smith_waterman_farrar_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(prepare_smith_waterman_farrar_score,
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static Score smith_waterman_farrar_score_prepared(
      PreparedSmithWatermanFarrarScore& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_farrar_score_prepared(prepared);
  }

  static std::vector<Score> smith_waterman_farrar_scores(
      nb::handle query, nb::handle targets,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(smith_waterman_farrar_scores,
        query, targets, match_score, mismatch_score, gap_score, width);
  }

  static Score smith_waterman_affine_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(smith_waterman_affine_score,
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static std::vector<Score> smith_waterman_affine_scores(
      nb::handle query, nb::handle targets,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(smith_waterman_affine_scores,
        query, targets, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static AlignmentResult smith_waterman_affine_path(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(smith_waterman_affine_path,
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static AlignmentPath smith_waterman_affine_path_info(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(smith_waterman_affine_path_info,
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static std::string smith_waterman_affine_cigar(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(smith_waterman_affine_cigar,
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static Score smith_waterman_affine_farrar_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(smith_waterman_affine_farrar_score,
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static PreparedAffineScore prepare_smith_waterman_affine_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(prepare_smith_waterman_affine_score,
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static Score smith_waterman_affine_score_prepared(PreparedAffineScore& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_score_prepared(prepared);
  }

  static PreparedAffineScore prepare_smith_waterman_affine_farrar_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(prepare_smith_waterman_affine_farrar_score,
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static Score smith_waterman_affine_farrar_score_prepared(PreparedAffineScore& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_farrar_score_prepared(prepared);
  }

  static Score needleman_wunsch_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(needleman_wunsch_score,
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static std::vector<Score> needleman_wunsch_scores(
      nb::handle query, nb::handle targets,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(needleman_wunsch_scores,
        query, targets, match_score, mismatch_score, gap_score, width);
  }

  static AlignmentResult needleman_wunsch_path(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(needleman_wunsch_path,
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static AlignmentPath needleman_wunsch_path_info(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(needleman_wunsch_path_info,
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static std::string needleman_wunsch_linear_cigar(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(needleman_wunsch_linear_cigar,
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static Score needleman_wunsch_affine_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(needleman_wunsch_affine_score,
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static std::vector<Score> needleman_wunsch_affine_scores(
      nb::handle query, nb::handle targets,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(needleman_wunsch_affine_scores,
        query, targets, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static AlignmentResult needleman_wunsch_affine_path(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(needleman_wunsch_affine_path,
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static AlignmentPath needleman_wunsch_affine_path_info(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(needleman_wunsch_affine_path_info,
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static std::string needleman_wunsch_affine_cigar(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(needleman_wunsch_affine_cigar,
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static PreparedAffineScore prepare_needleman_wunsch_affine_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    STRIDE_ALIGN_VSX_FWD(prepare_needleman_wunsch_affine_score,
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static Score needleman_wunsch_affine_score_prepared(PreparedAffineScore& prepared) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_score_prepared(prepared);
  }

  // Levenshtein + Damerau-Levenshtein SIMD batch via VsxOps (128-bit,
  // 2 lanes). Power8 ISA 2.07 has native unsigned cmpgt_u64 so the
  // bit-parallel kernel ports cleanly.
  static std::vector<Score> levenshtein_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    ensure_supported();
    return ::stride_align::levenshtein_simd::levenshtein_scores_simd<
        ::stride_align::levenshtein_simd::VsxOps>(query, targets, cutoff);
  }

  static std::vector<double> levenshtein_normalized_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    ensure_supported();
    return ::stride_align::levenshtein_simd::levenshtein_normalized_scores_simd<
        ::stride_align::levenshtein_simd::VsxOps>(query, targets, cutoff);
  }

  static std::vector<Score> damerau_levenshtein_scores(
      nb::handle query, nb::handle targets) {
    ensure_supported();
    return ::stride_align::osa_simd::osa_scores_simd<
        ::stride_align::levenshtein_simd::VsxOps>(query, targets);
  }

  static std::vector<double> damerau_levenshtein_normalized_scores(
      nb::handle query, nb::handle targets) {
    ensure_supported();
    return ::stride_align::osa_simd::osa_normalized_scores_simd<
        ::stride_align::levenshtein_simd::VsxOps>(query, targets);
  }

  static std::vector<Score> indel_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::indel_simd::indel_scores_simd<
        ::stride_align::levenshtein_simd::VsxOps>(query, targets);
  }

  static std::vector<double> indel_normalized_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::indel_simd::indel_normalized_scores_simd<
        ::stride_align::levenshtein_simd::VsxOps>(query, targets);
  }

  static std::vector<double> jaro_similarities(
      nb::handle query, nb::handle targets) {
    ensure_supported();
    return ::stride_align::jaro_simd::jaro_similarities_simd<
        ::stride_align::levenshtein_simd::VsxOps>(query, targets);
  }

  static std::vector<double> jaro_winkler_similarities(
      nb::handle query,
      nb::handle targets,
      double prefix_weight,
      double prefix_threshold,
      std::size_t prefix_cap) {
    ensure_supported();
    return ::stride_align::jaro_simd::jaro_winkler_similarities_simd<
        ::stride_align::levenshtein_simd::VsxOps>(
        query, targets, prefix_weight, prefix_threshold, prefix_cap);
  }

  static nb::object cdist(
      nb::handle queries, nb::handle targets, int scorer,
      nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    ensure_supported();
    return ::stride_align::cdist_simd::cdist_impl<
        ::stride_align::levenshtein_simd::VsxOps>(
        queries, targets, scorer, tqdm_factory, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static nb::object cdist_above_threshold(
      nb::handle queries, nb::handle targets, int scorer,
      double threshold, nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    ensure_supported();
    return ::stride_align::cdist_threshold::cdist_threshold_impl<
        ::stride_align::levenshtein_simd::VsxOps>(
        queries, targets, scorer, threshold, tqdm_factory, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static nb::object cdist_top_k(
      nb::handle queries, nb::handle targets, int scorer,
      std::size_t k, nb::object tqdm_factory, std::size_t cpu_count,
      bool reject_duplicates,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    ensure_supported();
    return ::stride_align::cdist_topk::cdist_top_k_impl<
        ::stride_align::levenshtein_simd::VsxOps>(
        queries, targets, scorer, k, tqdm_factory, cpu_count,
        reject_duplicates,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static constexpr BackendKind backend_kind = BackendKind::linux_powerpc64_vsx;

#undef STRIDE_ALIGN_VSX_FWD
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

};

}  // namespace stride_align::backend_linux_powerpc64_vsx
