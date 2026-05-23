#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <nanobind/nanobind.h>

#include "backends/affine_scalable_kernel.hpp"
#include "backends/farrar_scalable_kernel.hpp"
#include "backends/profile_traceback.hpp"

// SPARC T4/T5 VIS3 SIMD backend. 64-bit vector register width (4xi16 / 2xi32
// / 8xi8 lanes). T5 doesn't have VIS4's partitioned max/min, so we lean on
// GCC's pattern recognition for vector compare + mask + select, which lowers
// to fcmpgt + cmask + bshuffle + fxor (~8 instructions per max). Everything
// else is direct VIS1/VIS3 builtins.
namespace stride_align::sparc_vis3_backend {

namespace nb = nanobind;

// GCC vector types. These map to a single 64-bit FP register (which VIS reuses
// for integer SIMD).
typedef short v4hi __attribute__((vector_size(8)));
typedef int   v2si __attribute__((vector_size(8)));
typedef char  v8qi __attribute__((vector_size(8)));

namespace detail {

// Compile-time-built permute index for shift_left_zero<N> on N-lane vector.
template <int Lanes>
constexpr auto shift_left_zero_indices() {
  if constexpr (Lanes == 4) {
    return v4hi{3, 4, 5, 6};   // [zero[3], v[0], v[1], v[2]] -> shift left by 1 lane
  } else if constexpr (Lanes == 2) {
    return v2si{1, 2};
  }
}

template <int Lanes, typename Cell>
struct ShiftInsertIndices;

template <> struct ShiftInsertIndices<4, std::int16_t> {
  static v4hi value() { return v4hi{3, 4, 5, 6}; }
};
template <> struct ShiftInsertIndices<2, std::int32_t> {
  static v2si value() { return v2si{1, 2}; }
};

}  // namespace detail

template <typename Token, typename Cell>
struct SimdOps;

// ---------- 4 x int16 ----------
template <>
struct SimdOps<std::uint16_t, std::int16_t> {
  using vector_type = v4hi;
  static constexpr bool has_vector_max = true;

  static constexpr std::size_t lane_count() { return 4; }

  static vector_type load_cells(const std::int16_t* values, std::size_t) {
    vector_type v;
    std::memcpy(&v, values, sizeof(v));
    return v;
  }

  static void store_cells(std::int16_t* values, vector_type v, std::size_t) {
    std::memcpy(values, &v, sizeof(v));
  }

  static vector_type set1(std::int16_t value, std::size_t) {
    return vector_type{value, value, value, value};
  }

  static vector_type zero(std::size_t) {
    return vector_type{0, 0, 0, 0};
  }

  static vector_type add(vector_type a, vector_type b, std::size_t) {
    return __builtin_vis_fpadd16(a, b);
  }

  // VIS3 has no fpmax16 (that's VIS4). Compiler folds this to
  // fcmpgt16 + cmask16 + fxor + bshuffle + fand + fxor (~8 instructions).
  static vector_type max(vector_type a, vector_type b, std::size_t) {
    vector_type m = a > b;
    return (m & a) | (~m & b);
  }

  // Scalar reduction over 4 lanes. Two compares + cmovs.
  static std::int16_t reduce_max(vector_type v, std::size_t) {
    std::int16_t r[4];
    std::memcpy(r, &v, sizeof(v));
    std::int16_t m01 = r[0] > r[1] ? r[0] : r[1];
    std::int16_t m23 = r[2] > r[3] ? r[2] : r[3];
    return m01 > m23 ? m01 : m23;
  }

  // fcmpgt16 returns 4-bit int (1 bit per lane); non-zero means any lane gt.
  static bool any_gt(vector_type a, vector_type b, std::size_t) {
    return __builtin_vis_fcmpgt16(a, b) != 0;
  }

  static vector_type substitution(
      const std::uint16_t* query,
      const std::uint16_t* target,
      std::int16_t match_score,
      std::int16_t mismatch_score,
      std::size_t) {
    vector_type q, t;
    std::memcpy(&q, query, sizeof(q));
    std::memcpy(&t, target, sizeof(t));
    vector_type eq = (q == t);
    vector_type m = set1(match_score, 0);
    vector_type x = set1(mismatch_score, 0);
    return (eq & m) | (~eq & x);
  }

  // Shift left by 1 lane (= 2 bytes), inserting 0 at lane 0.
  // GCC emits faligndata via __builtin_shuffle on v4hi.
  static vector_type shift_left_zero(vector_type v, std::size_t) {
    vector_type z = zero(0);
    return __builtin_shuffle(z, v, vector_type{3, 4, 5, 6});
  }

  static vector_type shift_left_insert(vector_type v, std::int16_t inserted, std::size_t) {
    vector_type ins = set1(inserted, 0);
    return __builtin_shuffle(ins, v, vector_type{3, 4, 5, 6});
  }

  static vector_type first_lane_vector(std::int16_t first, std::int16_t rest, std::size_t) {
    return vector_type{first, rest, rest, rest};
  }

  // Sentinel-aware add. INT16_MIN as sentinel; (a == sentinel ? sentinel : a+b).
  static vector_type add_sentinel(
      vector_type lhs,
      vector_type rhs,
      std::int16_t sentinel,
      std::size_t) {
    vector_type sent = set1(sentinel, 0);
    vector_type sum = __builtin_vis_fpadd16(lhs, rhs);
    vector_type keep = (lhs == sent);
    return (keep & sent) | (~keep & sum);
  }

  // VIS3 signed saturating add. Used by add_sentinel_negative_rhs in the
  // affine kernels: when rhs <= 0 (gap_open/gap_extend) and sentinel is
  // INT16_MIN, fpadds16 gives the same result as add_sentinel in one
  // instruction (INT_MIN + (<=0) saturates to INT_MIN; normal + small_neg
  // stays in range). Mirrors the SVE2 svqadd_s16 optimization that closes
  // the NW-affine perf gap.
  static vector_type saturating_add(vector_type lhs, vector_type rhs) {
    return __builtin_vis_fpadds16(lhs, rhs);
  }

  // Parallel prefix-carry lazy-F. 4 lanes = 2 log-steps (shift by 1, then 2).
  // GCC lowers __builtin_shuffle with constant indices to faligndata via
  // GSR.ALIGN, so the shift stays in the SIMD pipeline.
  template <std::size_t Lanes>
  static vector_type shift_left_lanes_zero(vector_type v) {
    static_assert(Lanes >= 0 && Lanes <= 4);
    vector_type z = vector_type{0, 0, 0, 0};
    if constexpr (Lanes == 0) {
      return v;
    } else if constexpr (Lanes == 1) {
      return __builtin_shuffle(z, v, vector_type{3, 4, 5, 6});
    } else if constexpr (Lanes == 2) {
      return __builtin_shuffle(z, v, vector_type{2, 3, 4, 5});
    } else {
      return z;
    }
  }

  template <std::size_t Lanes>
  static vector_type shift_left_lanes_insert_sentinel(vector_type v, std::int16_t fill) {
    static_assert(Lanes >= 0 && Lanes <= 4);
    vector_type ins = vector_type{fill, fill, fill, fill};
    if constexpr (Lanes == 0) {
      return v;
    } else if constexpr (Lanes == 1) {
      return __builtin_shuffle(ins, v, vector_type{3, 4, 5, 6});
    } else if constexpr (Lanes == 2) {
      return __builtin_shuffle(ins, v, vector_type{2, 3, 4, 5});
    } else {
      return ins;
    }
  }

  static vector_type local_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int16_t gap_score) {
    const auto span_gap = static_cast<std::int16_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_score));
    vector_type zero_vector = zero(0);
    vector_type prefix = max(final_f, zero_vector, 0);
    vector_type shifted = add(shift_left_lanes_zero<1>(prefix), set1(span_gap, 0), 0);
    prefix = max(prefix, shifted, 0);
    shifted = add(
        shift_left_lanes_zero<2>(prefix),
        set1(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 2), 0),
        0);
    prefix = max(prefix, shifted, 0);
    return shift_left_lanes_zero<1>(max(prefix, zero_vector, 0));
  }

  static vector_type global_lazy_f_prefix_carry_no_padding(
      vector_type final_f,
      std::size_t segment_count,
      std::int16_t gap_extend_score,
      std::int16_t low_score) {
    const auto span_gap = static_cast<std::int16_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_extend_score));
    vector_type prefix = final_f;
    vector_type shifted = add_sentinel(
        shift_left_lanes_insert_sentinel<1>(prefix, low_score),
        set1(span_gap, 0),
        low_score,
        0);
    prefix = max(prefix, shifted, 0);
    shifted = add_sentinel(
        shift_left_lanes_insert_sentinel<2>(prefix, low_score),
        set1(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 2), 0),
        low_score,
        0);
    prefix = max(prefix, shifted, 0);
    return shift_left_lanes_insert_sentinel<1>(prefix, low_score);
  }
};

// ---------- 2 x int32 ----------
template <>
struct SimdOps<std::uint32_t, std::int32_t> {
  using vector_type = v2si;
  static constexpr bool has_vector_max = true;

  static constexpr std::size_t lane_count() { return 2; }

  static vector_type load_cells(const std::int32_t* values, std::size_t) {
    vector_type v;
    std::memcpy(&v, values, sizeof(v));
    return v;
  }

  static void store_cells(std::int32_t* values, vector_type v, std::size_t) {
    std::memcpy(values, &v, sizeof(v));
  }

  static vector_type set1(std::int32_t value, std::size_t) {
    return vector_type{value, value};
  }

  static vector_type zero(std::size_t) {
    return vector_type{0, 0};
  }

  static vector_type add(vector_type a, vector_type b, std::size_t) {
    return __builtin_vis_fpadd32(a, b);
  }

  static vector_type max(vector_type a, vector_type b, std::size_t) {
    vector_type m = a > b;
    return (m & a) | (~m & b);
  }

  static std::int32_t reduce_max(vector_type v, std::size_t) {
    std::int32_t r[2];
    std::memcpy(r, &v, sizeof(v));
    return r[0] > r[1] ? r[0] : r[1];
  }

  static bool any_gt(vector_type a, vector_type b, std::size_t) {
    return __builtin_vis_fcmpgt32(a, b) != 0;
  }

  static vector_type substitution(
      const std::uint32_t* query,
      const std::uint32_t* target,
      std::int32_t match_score,
      std::int32_t mismatch_score,
      std::size_t) {
    vector_type q, t;
    std::memcpy(&q, query, sizeof(q));
    std::memcpy(&t, target, sizeof(t));
    vector_type eq = (q == t);
    vector_type m = set1(match_score, 0);
    vector_type x = set1(mismatch_score, 0);
    return (eq & m) | (~eq & x);
  }

  static vector_type shift_left_zero(vector_type v, std::size_t) {
    vector_type z = zero(0);
    return __builtin_shuffle(z, v, vector_type{1, 2});
  }

  static vector_type shift_left_insert(vector_type v, std::int32_t inserted, std::size_t) {
    vector_type ins = set1(inserted, 0);
    return __builtin_shuffle(ins, v, vector_type{1, 2});
  }

  static vector_type first_lane_vector(std::int32_t first, std::int32_t rest, std::size_t) {
    return vector_type{first, rest};
  }

  static vector_type add_sentinel(
      vector_type lhs,
      vector_type rhs,
      std::int32_t sentinel,
      std::size_t) {
    vector_type sent = set1(sentinel, 0);
    vector_type sum = __builtin_vis_fpadd32(lhs, rhs);
    vector_type keep = (lhs == sent);
    return (keep & sent) | (~keep & sum);
  }

  // VIS3 fpadds32 (signed saturating add for 2xi32). Same semantics as
  // SVE2 svqadd_s32 — collapses sentinel-preserving add to a single
  // instruction when rhs <= 0.
  static vector_type saturating_add(vector_type lhs, vector_type rhs) {
    return __builtin_vis_fpadds32(lhs, rhs);
  }

  // 2 lanes = 1 log-step prefix carry (shift by 1).
  template <std::size_t Lanes>
  static vector_type shift_left_lanes_zero(vector_type v) {
    static_assert(Lanes >= 0 && Lanes <= 2);
    vector_type z = vector_type{0, 0};
    if constexpr (Lanes == 0) {
      return v;
    } else if constexpr (Lanes == 1) {
      return __builtin_shuffle(z, v, vector_type{1, 2});
    } else {
      return z;
    }
  }

  template <std::size_t Lanes>
  static vector_type shift_left_lanes_insert_sentinel(vector_type v, std::int32_t fill) {
    static_assert(Lanes >= 0 && Lanes <= 2);
    vector_type ins = vector_type{fill, fill};
    if constexpr (Lanes == 0) {
      return v;
    } else if constexpr (Lanes == 1) {
      return __builtin_shuffle(ins, v, vector_type{1, 2});
    } else {
      return ins;
    }
  }

  static vector_type local_lazy_f_prefix_carry(
      vector_type final_f,
      std::size_t segment_count,
      std::int32_t gap_score) {
    const auto span_gap = static_cast<std::int32_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_score));
    vector_type zero_vector = zero(0);
    vector_type prefix = max(final_f, zero_vector, 0);
    vector_type shifted = add(shift_left_lanes_zero<1>(prefix), set1(span_gap, 0), 0);
    prefix = max(prefix, shifted, 0);
    return shift_left_lanes_zero<1>(max(prefix, zero_vector, 0));
  }

  static vector_type global_lazy_f_prefix_carry_no_padding(
      vector_type final_f,
      std::size_t segment_count,
      std::int32_t gap_extend_score,
      std::int32_t low_score) {
    const auto span_gap = static_cast<std::int32_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_extend_score));
    vector_type prefix = final_f;
    vector_type shifted = add_sentinel(
        shift_left_lanes_insert_sentinel<1>(prefix, low_score),
        set1(span_gap, 0),
        low_score,
        0);
    prefix = max(prefix, shifted, 0);
    return shift_left_lanes_insert_sentinel<1>(prefix, low_score);
  }
};

// ---------- 8 x int8 ----------
// SPARC VIS has no fpadd8 / fpcmpgt8, so this lane shape is a scalar
// fallback. GCC's vector_size ops compile to 8 scalar adds/cmps; still
// correct for the kernel's bits8 width path.
template <>
struct SimdOps<std::uint8_t, std::int8_t> {
  using vector_type = v8qi;
  static constexpr bool has_vector_max = true;

  static constexpr std::size_t lane_count() { return 8; }

  static vector_type load_cells(const std::int8_t* values, std::size_t) {
    vector_type v;
    std::memcpy(&v, values, sizeof(v));
    return v;
  }

  static void store_cells(std::int8_t* values, vector_type v, std::size_t) {
    std::memcpy(values, &v, sizeof(v));
  }

  static vector_type set1(std::int8_t value, std::size_t) {
    return vector_type{value, value, value, value, value, value, value, value};
  }

  static vector_type zero(std::size_t) {
    return vector_type{0, 0, 0, 0, 0, 0, 0, 0};
  }

  static vector_type add(vector_type a, vector_type b, std::size_t) {
    return a + b;
  }

  static vector_type max(vector_type a, vector_type b, std::size_t) {
    vector_type m = a > b;
    return (m & a) | (~m & b);
  }

  static std::int8_t reduce_max(vector_type v, std::size_t) {
    std::int8_t r[8];
    std::memcpy(r, &v, sizeof(v));
    std::int8_t best = r[0];
    for (int i = 1; i < 8; ++i) if (r[i] > best) best = r[i];
    return best;
  }

  static bool any_gt(vector_type a, vector_type b, std::size_t) {
    vector_type m = a > b;
    std::uint8_t r[8];
    std::memcpy(r, &m, sizeof(m));
    for (int i = 0; i < 8; ++i) if (r[i]) return true;
    return false;
  }

  static vector_type substitution(
      const std::uint8_t* query,
      const std::uint8_t* target,
      std::int8_t match_score,
      std::int8_t mismatch_score,
      std::size_t) {
    vector_type q, t;
    std::memcpy(&q, query, sizeof(q));
    std::memcpy(&t, target, sizeof(t));
    vector_type eq = (q == t);
    vector_type m = set1(match_score, 0);
    vector_type x = set1(mismatch_score, 0);
    return (eq & m) | (~eq & x);
  }

  static vector_type shift_left_zero(vector_type v, std::size_t) {
    vector_type z = zero(0);
    return __builtin_shuffle(z, v, vector_type{7, 8, 9, 10, 11, 12, 13, 14});
  }

  static vector_type shift_left_insert(vector_type v, std::int8_t inserted, std::size_t) {
    vector_type ins = set1(inserted, 0);
    return __builtin_shuffle(ins, v, vector_type{7, 8, 9, 10, 11, 12, 13, 14});
  }

  static vector_type first_lane_vector(std::int8_t first, std::int8_t rest, std::size_t) {
    return vector_type{first, rest, rest, rest, rest, rest, rest, rest};
  }

  static vector_type add_sentinel(
      vector_type lhs,
      vector_type rhs,
      std::int8_t sentinel,
      std::size_t) {
    vector_type sent = set1(sentinel, 0);
    vector_type sum = lhs + rhs;
    vector_type keep = (lhs == sent);
    return (keep & sent) | (~keep & sum);
  }
};

// ---------- 1 x int64 ----------
// SPARC has no 64-bit partitioned ops; with one lane per 64-bit register
// the kernel's bits64 path is plain scalar.
template <>
struct SimdOps<std::uint64_t, std::int64_t> {
  using vector_type = std::int64_t;
  static constexpr bool has_vector_max = true;

  static constexpr std::size_t lane_count() { return 1; }

  static vector_type load_cells(const std::int64_t* values, std::size_t) {
    return values[0];
  }

  static void store_cells(std::int64_t* values, vector_type v, std::size_t) {
    values[0] = v;
  }

  static vector_type set1(std::int64_t value, std::size_t) { return value; }

  static vector_type zero(std::size_t) { return 0; }

  static vector_type add(vector_type a, vector_type b, std::size_t) { return a + b; }

  static vector_type max(vector_type a, vector_type b, std::size_t) {
    return a > b ? a : b;
  }

  static std::int64_t reduce_max(vector_type v, std::size_t) { return v; }

  static bool any_gt(vector_type a, vector_type b, std::size_t) { return a > b; }

  static vector_type substitution(
      const std::uint64_t* query,
      const std::uint64_t* target,
      std::int64_t match_score,
      std::int64_t mismatch_score,
      std::size_t) {
    return (*query == *target) ? match_score : mismatch_score;
  }

  static vector_type shift_left_zero(vector_type, std::size_t) { return 0; }

  static vector_type shift_left_insert(vector_type, std::int64_t inserted, std::size_t) {
    return inserted;
  }

  static vector_type first_lane_vector(std::int64_t first, std::int64_t, std::size_t) {
    return first;
  }

  static vector_type add_sentinel(
      vector_type lhs,
      vector_type rhs,
      std::int64_t sentinel,
      std::size_t) {
    return lhs == sentinel ? sentinel : lhs + rhs;
  }
};

// Entry-point Implementation. Dispatches to farrar_scalable_kernel for the
// SIMD-friendly paths; falls back to profile_traceback for affine path/cigar.
// Mirrors arm_sve_backend::TargetImplementation in structure.
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
    const auto prepared =
        prepare_linear_score_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return farrar_scalable_kernel::detail::dispatch_score<SimdOps>(
        prepared, match_score, mismatch_score, gap_score);
  }

  static AlignmentResult smith_waterman_path(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    return profile_traceback::linear_path<true>(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static AlignmentPath smith_waterman_path_info(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    return profile_traceback::linear_path_info<true>(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static Score smith_waterman_farrar_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    const auto prepared =
        prepare_farrar_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return farrar_scalable_kernel::detail::dispatch_score<SimdOps>(
        prepared, match_score, mismatch_score, gap_score);
  }

  static PreparedSmithWatermanFarrarScore prepare_smith_waterman_farrar_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    const auto prepared =
        prepare_farrar_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return farrar_scalable_kernel::detail::prepare_score<SimdOps>(
        prepared, match_score, mismatch_score, gap_score);
  }

  static Score smith_waterman_farrar_score_prepared(
      PreparedSmithWatermanFarrarScore& prepared) {
    return farrar_scalable_kernel::detail::dispatch_prepared_score<SimdOps>(prepared);
  }

  // --- Affine (SW) ---
  static Score smith_waterman_affine_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    if (gap_open_score <= 0 && gap_extend_score <= 0) {
      const auto prepared = prepare_farrar_alignment(
          query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
      return affine_scalable_kernel::detail::dispatch_compact_byte_score<SimdOps>(
          prepared, match_score, mismatch_score, gap_open_score, gap_extend_score);
    }
    return profile_traceback::affine_score<true>(
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static AlignmentResult smith_waterman_affine_path(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    return profile_traceback::affine_path<true>(
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static AlignmentPath smith_waterman_affine_path_info(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    const Score expected = smith_waterman_affine_score(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
    return profile_traceback::affine_path_info_with_score<true>(
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width, expected);
  }

  static std::string smith_waterman_affine_cigar(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    const Score expected = smith_waterman_affine_score(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
    return profile_traceback::affine_cigar_with_score<true>(
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width, expected);
  }

  static Score smith_waterman_affine_farrar_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    const auto prepared = prepare_farrar_alignment(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
    return affine_scalable_kernel::detail::dispatch_compact_byte_score<SimdOps>(
        prepared, match_score, mismatch_score, gap_open_score, gap_extend_score);
  }

  static PreparedAffineScore prepare_smith_waterman_affine_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    const auto prepared = prepare_farrar_alignment(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
    return farrar_scalable_kernel::detail::prepare_affine_score<SimdOps>(
        prepared, match_score, mismatch_score, gap_open_score, gap_extend_score);
  }

  static Score smith_waterman_affine_score_prepared(PreparedAffineScore& prepared) {
    return farrar_scalable_kernel::detail::dispatch_prepared_affine_score<SimdOps>(prepared);
  }

  static PreparedAffineScore prepare_smith_waterman_affine_farrar_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    return prepare_smith_waterman_affine_score(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
  }

  static Score smith_waterman_affine_farrar_score_prepared(PreparedAffineScore& prepared) {
    return smith_waterman_affine_score_prepared(prepared);
  }

  // --- NW (global) ---
  static PreparedAffineScore prepare_needleman_wunsch_affine_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    const auto prepared = prepare_farrar_alignment(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
    return farrar_scalable_kernel::detail::prepare_affine_score<SimdOps>(
        prepared, match_score, mismatch_score, gap_open_score, gap_extend_score);
  }

  static Score needleman_wunsch_affine_score_prepared(PreparedAffineScore& prepared) {
    return farrar_scalable_kernel::detail::dispatch_prepared_global_affine_score<SimdOps>(prepared);
  }

  static Score needleman_wunsch_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    const auto prepared = prepare_linear_score_alignment(
        query, target, match_score, mismatch_score, gap_score, width);
    return farrar_scalable_kernel::detail::dispatch_global_score<SimdOps>(
        prepared, match_score, mismatch_score, gap_score);
  }

  static AlignmentResult needleman_wunsch_path(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    return profile_traceback::linear_path<false>(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static AlignmentPath needleman_wunsch_path_info(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    return profile_traceback::linear_path_info<false>(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static Score needleman_wunsch_affine_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    const auto prepared = prepare_farrar_alignment(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
    return farrar_scalable_kernel::detail::dispatch_global_affine_score<SimdOps>(
        prepared, match_score, mismatch_score, gap_open_score, gap_extend_score);
  }

  static AlignmentResult needleman_wunsch_affine_path(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    return profile_traceback::affine_path<false>(
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width);
  }

  static AlignmentPath needleman_wunsch_affine_path_info(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    const Score expected = needleman_wunsch_affine_score(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
    return profile_traceback::affine_path_info_with_score<false>(
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width, expected);
  }

  static std::string needleman_wunsch_affine_cigar(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    const Score expected = needleman_wunsch_affine_score(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
    return profile_traceback::affine_cigar_with_score<false>(
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score, width, expected);
  }
};

}  // namespace stride_align::sparc_vis3_backend
