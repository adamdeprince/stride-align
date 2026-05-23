#pragma once

#include <smmintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>

#include "backends/affine_fixed_kernel.hpp"
#include "backends/farrar_fixed_kernel.hpp"
#include "backends/generic.hpp"
#include "backends/profile_traceback.hpp"
#include "cdist_simd.hpp"
#include "jaro_simd.hpp"
#include "levenshtein_simd.hpp"
#include "levenshtein_simd_ops.hpp"
#include "osa_simd.hpp"

namespace stride_align::backend_sse41 {

namespace nb = nanobind;
namespace generic_detail = stride_align::backend_generic::detail;

namespace detail {

template <int LaneBytes>
inline __m128i first_lane_mask_128() {
  static_assert(LaneBytes == 1 || LaneBytes == 2 || LaneBytes == 4 || LaneBytes == 8);
  if constexpr (LaneBytes == 8) {
    return _mm_set_epi64x(0, -1LL);
  } else {
    return _mm_set_epi64x(0, (1LL << (LaneBytes * 8)) - 1LL);
  }
}

template <int LaneBytes>
inline __m128i shift_left_insert_128(__m128i vector, __m128i inserted) {
  return _mm_blendv_epi8(
      _mm_slli_si128(vector, LaneBytes),
      inserted,
      first_lane_mask_128<LaneBytes>());
}

template <typename Cell>
struct DiagonalState {
  std::size_t row_start = 1;
  std::vector<Cell> scores;
};

template <typename Token, typename Cell>
struct SimdOps;

template <>
struct SimdOps<std::uint8_t, std::int8_t> {
  using vector_type = __m128i;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 16;
  static constexpr bool local_sw_score_exact_segment64 = true;
  static constexpr bool bounded_local_sw_lazy_f_scan = true;

  static __m128i load_tokens(const std::uint8_t* values) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
  }

  static __m128i load_cells(const std::int8_t* values) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
  }

  static __m128i load_aligned_cells(const std::int8_t* values) {
    return _mm_load_si128(reinterpret_cast<const __m128i*>(values));
  }

  static void store_cells(std::int8_t* values, __m128i vector) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(values), vector);
  }

  static void store_aligned_cells(std::int8_t* values, __m128i vector) {
    _mm_store_si128(reinterpret_cast<__m128i*>(values), vector);
  }

  static __m128i set1(std::int8_t value) {
    return _mm_set1_epi8(value);
  }

  static __m128i zero() {
    return _mm_setzero_si128();
  }

  static __m128i add(__m128i lhs, __m128i rhs) {
    return _mm_add_epi8(lhs, rhs);
  }

  static __m128i add_sentinel(__m128i lhs, __m128i rhs, std::int8_t sentinel) {
    const __m128i sum = _mm_add_epi8(lhs, rhs);
    const __m128i mask = _mm_cmpeq_epi8(lhs, set1(sentinel));
    return _mm_blendv_epi8(sum, set1(sentinel), mask);
  }

  static __m128i max(__m128i lhs, __m128i rhs) {
    return _mm_max_epi8(lhs, rhs);
  }

  static __m128i shift_left_zero(__m128i vector) {
    return _mm_slli_si128(vector, 1);
  }

  static __m128i shift_left_insert(__m128i vector, std::int8_t inserted) {
    return shift_left_insert_128<1>(vector, set1(inserted));
  }

  static __m128i local_lazy_f_prefix_carry(
      __m128i final_f,
      std::size_t segment_count,
      std::int8_t gap_score) {
    const auto span_gap = static_cast<std::int8_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_score));
    const auto zero_vector = zero();
    auto prefix = max(final_f, zero_vector);
    auto shifted = add(shift_left_insert_128<1>(prefix, zero_vector), set1(span_gap));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_128<2>(prefix, zero_vector),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 2)));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_128<4>(prefix, zero_vector),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 4)));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_128<8>(prefix, zero_vector),
        set1(static_cast<std::int8_t>(static_cast<Score>(span_gap) * 8)));
    prefix = max(prefix, shifted);
    return shift_left_insert_128<1>(max(prefix, zero_vector), zero_vector);
  }

  static bool any_gt(__m128i lhs, __m128i rhs) {
    return _mm_movemask_epi8(_mm_cmpgt_epi8(lhs, rhs)) != 0;
  }

  static __m128i greater_mask(__m128i lhs, __m128i rhs) {
    return _mm_cmpgt_epi8(lhs, rhs);
  }

  static __m128i bit_or(__m128i lhs, __m128i rhs) {
    return _mm_or_si128(lhs, rhs);
  }

  static bool any_nonzero(__m128i value) {
    return _mm_testz_si128(value, value) == 0;
  }

  static __m128i substitution(
      const std::uint8_t* query,
      const std::uint8_t* target,
      std::int8_t match_score,
      std::int8_t mismatch_score) {
    const __m128i mask = _mm_cmpeq_epi8(load_tokens(query), load_tokens(target));
    return _mm_blendv_epi8(set1(mismatch_score), set1(match_score), mask);
  }
};

template <>
struct SimdOps<std::uint16_t, std::int16_t> {
  using vector_type = __m128i;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 8;
  static constexpr bool local_sw_score_exact_segment128 = true;
  static constexpr bool bounded_local_sw_lazy_f_scan = true;

  static __m128i load_tokens(const std::uint16_t* values) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
  }

  static __m128i load_cells(const std::int16_t* values) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
  }

  static __m128i load_aligned_cells(const std::int16_t* values) {
    return _mm_load_si128(reinterpret_cast<const __m128i*>(values));
  }

  static void store_cells(std::int16_t* values, __m128i vector) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(values), vector);
  }

  static void store_aligned_cells(std::int16_t* values, __m128i vector) {
    _mm_store_si128(reinterpret_cast<__m128i*>(values), vector);
  }

  static __m128i set1(std::int16_t value) {
    return _mm_set1_epi16(value);
  }

  static __m128i zero() {
    return _mm_setzero_si128();
  }

  static __m128i add(__m128i lhs, __m128i rhs) {
    return _mm_add_epi16(lhs, rhs);
  }

  static __m128i add_sentinel(__m128i lhs, __m128i rhs, std::int16_t sentinel) {
    const __m128i sum = _mm_add_epi16(lhs, rhs);
    const __m128i mask = _mm_cmpeq_epi16(lhs, set1(sentinel));
    return _mm_blendv_epi8(sum, set1(sentinel), mask);
  }

  static __m128i max(__m128i lhs, __m128i rhs) {
    return _mm_max_epi16(lhs, rhs);
  }

  static __m128i shift_left_zero(__m128i vector) {
    return _mm_slli_si128(vector, 2);
  }

  static __m128i shift_left_insert(__m128i vector, std::int16_t inserted) {
    return shift_left_insert_128<2>(vector, set1(inserted));
  }

  static __m128i local_lazy_f_prefix_carry(
      __m128i final_f,
      std::size_t segment_count,
      std::int16_t gap_score) {
    const auto span_gap = static_cast<std::int16_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_score));
    const auto zero_vector = zero();
    auto prefix = max(final_f, zero_vector);
    auto shifted = add(shift_left_insert_128<2>(prefix, zero_vector), set1(span_gap));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_128<4>(prefix, zero_vector),
        set1(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 2)));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_128<8>(prefix, zero_vector),
        set1(static_cast<std::int16_t>(static_cast<Score>(span_gap) * 4)));
    prefix = max(prefix, shifted);
    return shift_left_insert_128<2>(max(prefix, zero_vector), zero_vector);
  }

  static bool any_gt(__m128i lhs, __m128i rhs) {
    return _mm_movemask_epi8(_mm_cmpgt_epi16(lhs, rhs)) != 0;
  }

  static __m128i greater_mask(__m128i lhs, __m128i rhs) {
    return _mm_cmpgt_epi16(lhs, rhs);
  }

  static __m128i bit_or(__m128i lhs, __m128i rhs) {
    return _mm_or_si128(lhs, rhs);
  }

  static bool any_nonzero(__m128i value) {
    return _mm_testz_si128(value, value) == 0;
  }

  static __m128i substitution(
      const std::uint16_t* query,
      const std::uint16_t* target,
      std::int16_t match_score,
      std::int16_t mismatch_score) {
    const __m128i mask = _mm_cmpeq_epi16(load_tokens(query), load_tokens(target));
    return _mm_blendv_epi8(set1(mismatch_score), set1(match_score), mask);
  }
};

template <>
struct SimdOps<std::uint32_t, std::int32_t> {
  using vector_type = __m128i;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 4;
  static constexpr bool local_sw_score_exact_segment256 = true;
  static constexpr bool bounded_local_sw_lazy_f_scan = true;

  static __m128i load_tokens(const std::uint32_t* values) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
  }

  static __m128i load_cells(const std::int32_t* values) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
  }

  static __m128i load_aligned_cells(const std::int32_t* values) {
    return _mm_load_si128(reinterpret_cast<const __m128i*>(values));
  }

  static void store_cells(std::int32_t* values, __m128i vector) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(values), vector);
  }

  static void store_aligned_cells(std::int32_t* values, __m128i vector) {
    _mm_store_si128(reinterpret_cast<__m128i*>(values), vector);
  }

  static __m128i set1(std::int32_t value) {
    return _mm_set1_epi32(value);
  }

  static __m128i zero() {
    return _mm_setzero_si128();
  }

  static __m128i add(__m128i lhs, __m128i rhs) {
    return _mm_add_epi32(lhs, rhs);
  }

  static __m128i add_sentinel(__m128i lhs, __m128i rhs, std::int32_t sentinel) {
    const __m128i sum = _mm_add_epi32(lhs, rhs);
    const __m128i mask = _mm_cmpeq_epi32(lhs, set1(sentinel));
    return _mm_blendv_epi8(sum, set1(sentinel), mask);
  }

  static __m128i max(__m128i lhs, __m128i rhs) {
    return _mm_max_epi32(lhs, rhs);
  }

  static __m128i shift_left_zero(__m128i vector) {
    return _mm_slli_si128(vector, 4);
  }

  static __m128i shift_left_insert(__m128i vector, std::int32_t inserted) {
    return shift_left_insert_128<4>(vector, set1(inserted));
  }

  static __m128i local_lazy_f_prefix_carry(
      __m128i final_f,
      std::size_t segment_count,
      std::int32_t gap_score) {
    const auto span_gap = static_cast<std::int32_t>(
        static_cast<Score>(segment_count) * static_cast<Score>(gap_score));
    const auto zero_vector = zero();
    auto prefix = max(final_f, zero_vector);
    auto shifted = add(shift_left_insert_128<4>(prefix, zero_vector), set1(span_gap));
    prefix = max(prefix, shifted);
    shifted = add(
        shift_left_insert_128<8>(prefix, zero_vector),
        set1(static_cast<std::int32_t>(static_cast<Score>(span_gap) * 2)));
    prefix = max(prefix, shifted);
    return shift_left_insert_128<4>(max(prefix, zero_vector), zero_vector);
  }

  static bool any_gt(__m128i lhs, __m128i rhs) {
    return _mm_movemask_epi8(_mm_cmpgt_epi32(lhs, rhs)) != 0;
  }

  static __m128i greater_mask(__m128i lhs, __m128i rhs) {
    return _mm_cmpgt_epi32(lhs, rhs);
  }

  static __m128i bit_or(__m128i lhs, __m128i rhs) {
    return _mm_or_si128(lhs, rhs);
  }

  static bool any_nonzero(__m128i value) {
    return _mm_testz_si128(value, value) == 0;
  }

  static __m128i substitution(
      const std::uint32_t* query,
      const std::uint32_t* target,
      std::int32_t match_score,
      std::int32_t mismatch_score) {
    const __m128i mask = _mm_cmpeq_epi32(load_tokens(query), load_tokens(target));
    return _mm_blendv_epi8(set1(mismatch_score), set1(match_score), mask);
  }
};

template <>
struct SimdOps<std::uint64_t, std::int64_t> {
  using vector_type = __m128i;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t lane_count = 2;

  static __m128i load_tokens(const std::uint64_t* values) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
  }

  static __m128i load_cells(const std::int64_t* values) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
  }

  static __m128i load_aligned_cells(const std::int64_t* values) {
    return _mm_load_si128(reinterpret_cast<const __m128i*>(values));
  }

  static void store_cells(std::int64_t* values, __m128i vector) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(values), vector);
  }

  static void store_aligned_cells(std::int64_t* values, __m128i vector) {
    _mm_store_si128(reinterpret_cast<__m128i*>(values), vector);
  }

  static __m128i set1(std::int64_t value) {
    return _mm_set1_epi64x(value);
  }

  static __m128i zero() {
    return _mm_setzero_si128();
  }

  static __m128i add(__m128i lhs, __m128i rhs) {
    return _mm_add_epi64(lhs, rhs);
  }

  static __m128i add_sentinel(__m128i lhs, __m128i rhs, std::int64_t sentinel) {
    const __m128i sum = _mm_add_epi64(lhs, rhs);
    const __m128i mask = _mm_cmpeq_epi64(lhs, set1(sentinel));
    return _mm_blendv_epi8(sum, set1(sentinel), mask);
  }

  static __m128i max(__m128i lhs, __m128i rhs) {
    alignas(alignment) std::int64_t left[lane_count] = {};
    alignas(alignment) std::int64_t right[lane_count] = {};
    alignas(alignment) std::int64_t output[lane_count] = {};
    store_cells(left, lhs);
    store_cells(right, rhs);
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
      output[lane] = left[lane] > right[lane] ? left[lane] : right[lane];
    }
    return load_cells(output);
  }

  static __m128i shift_left_zero(__m128i vector) {
    return _mm_slli_si128(vector, 8);
  }

  static __m128i shift_left_insert(__m128i vector, std::int64_t inserted) {
    return shift_left_insert_128<8>(vector, set1(inserted));
  }

  static __m128i substitution(
      const std::uint64_t* query,
      const std::uint64_t* target,
      std::int64_t match_score,
      std::int64_t mismatch_score) {
    const __m128i mask = _mm_cmpeq_epi64(load_tokens(query), load_tokens(target));
    return _mm_blendv_epi8(set1(mismatch_score), set1(match_score), mask);
  }
};

template <bool LocalAlignment, typename Cell>
Cell boundary_score(std::size_t row, std::size_t column, Cell gap_score) {
  if constexpr (LocalAlignment) {
    return 0;
  }

  return static_cast<Cell>(
      (static_cast<Score>(row) + static_cast<Score>(column)) * static_cast<Score>(gap_score));
}

template <bool LocalAlignment, typename Cell>
Cell lookup_score(
    const DiagonalState<Cell>& diagonal,
    std::size_t row,
    std::size_t column,
    Cell gap_score) {
  if (row == 0 || column == 0) {
    return boundary_score<LocalAlignment>(row, column, gap_score);
  }

  const std::size_t diagonal_end = diagonal.row_start + diagonal.scores.size();
  if (diagonal.scores.empty() || row < diagonal.row_start || row >= diagonal_end) {
    return 0;
  }

  return diagonal.scores[row - diagonal.row_start];
}

template <typename Token, typename Cell, bool LocalAlignment, bool TrackDirections>
using KernelResult =
    std::conditional_t<TrackDirections, generic_detail::TracebackResult, Score>;

template <typename Token, typename Cell, bool LocalAlignment, bool TrackDirections>
KernelResult<Token, Cell, LocalAlignment, TrackDirections> run_kernel(
    std::span<const Token> query,
    std::span<const Token> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  using TraceDirection = generic_detail::TraceDirection;
  using TracebackResult = generic_detail::TracebackResult;
  using Ops = SimdOps<Token, Cell>;

  const std::size_t row_count = query.size() + 1;
  const std::size_t column_count = target.size() + 1;
  const auto direction_index = [column_count](std::size_t row, std::size_t column) {
    return row * column_count + column;
  };

  std::vector<TraceDirection> directions;
  if constexpr (TrackDirections) {
    directions.assign(row_count * column_count, TraceDirection::stop);
    if constexpr (!LocalAlignment) {
      for (std::size_t row = 1; row < row_count; ++row) {
        directions[direction_index(row, 0)] = TraceDirection::up;
      }
      for (std::size_t column = 1; column < column_count; ++column) {
        directions[direction_index(0, column)] = TraceDirection::left;
      }
    }
  }

  DiagonalState<Cell> previous_previous;
  DiagonalState<Cell> previous;

  Cell best_score = 0;
  std::size_t best_row = 0;
  std::size_t best_column = 0;
  Cell final_score = boundary_score<false>(query.size(), target.size(), gap_score);

  for (std::size_t diagonal = 2; diagonal <= query.size() + target.size(); ++diagonal) {
    const std::size_t current_row_start = diagonal > target.size() ? diagonal - target.size() : 1;
    const std::size_t current_row_end = std::min(query.size(), diagonal - 1);
    if (current_row_start > current_row_end) {
      continue;
    }

    DiagonalState<Cell> current;
    current.row_start = current_row_start;
    current.scores.assign(current_row_end - current_row_start + 1, 0);

    constexpr std::size_t lane_count = Ops::lane_count;
    const auto gap_vector = Ops::set1(gap_score);

    for (std::size_t offset = 0; offset < current.scores.size(); offset += lane_count) {
      const std::size_t count = std::min(lane_count, current.scores.size() - offset);

      alignas(16) Token query_tokens[lane_count] = {};
      alignas(16) Token target_tokens[lane_count] = {};
      alignas(16) Cell diagonal_scores[lane_count] = {};
      alignas(16) Cell up_scores[lane_count] = {};
      alignas(16) Cell left_scores[lane_count] = {};
      alignas(16) Cell cell_scores[lane_count] = {};

      for (std::size_t lane = 0; lane < count; ++lane) {
        const std::size_t row = current.row_start + offset + lane;
        const std::size_t column = diagonal - row;

        query_tokens[lane] = query[row - 1];
        target_tokens[lane] = target[column - 1];
        diagonal_scores[lane] = lookup_score<LocalAlignment>(
            previous_previous,
            row - 1,
            column - 1,
            gap_score);
        up_scores[lane] = lookup_score<LocalAlignment>(
            previous,
            row - 1,
            column,
            gap_score);
        left_scores[lane] = lookup_score<LocalAlignment>(
            previous,
            row,
            column - 1,
            gap_score);
      }

      const __m128i substitution_vector =
          Ops::substitution(query_tokens, target_tokens, match_score, mismatch_score);
      const __m128i diagonal_vector =
          Ops::add(Ops::load_cells(diagonal_scores), substitution_vector);
      const __m128i up_vector = Ops::add(Ops::load_cells(up_scores), gap_vector);
      const __m128i left_vector = Ops::add(Ops::load_cells(left_scores), gap_vector);

      Ops::store_cells(diagonal_scores, diagonal_vector);
      Ops::store_cells(up_scores, up_vector);
      Ops::store_cells(left_scores, left_vector);

      if constexpr (sizeof(Cell) < sizeof(std::int64_t)) {
        __m128i cell_vector = Ops::max(diagonal_vector, Ops::max(up_vector, left_vector));
        if constexpr (LocalAlignment) {
          cell_vector = Ops::max(cell_vector, Ops::zero());
        }
        Ops::store_cells(cell_scores, cell_vector);
      }

      for (std::size_t lane = 0; lane < count; ++lane) {
        const std::size_t row = current.row_start + offset + lane;
        const std::size_t column = diagonal - row;

        Cell selected_score = diagonal_scores[lane];
        TraceDirection direction = TraceDirection::diagonal;

        if (up_scores[lane] > selected_score) {
          selected_score = up_scores[lane];
          direction = TraceDirection::up;
        }

        if (left_scores[lane] > selected_score) {
          selected_score = left_scores[lane];
          direction = TraceDirection::left;
        }

        if constexpr (LocalAlignment) {
          if (selected_score <= 0) {
            selected_score = 0;
            direction = TraceDirection::stop;
          }
        }

        Cell cell = selected_score;
        if constexpr (sizeof(Cell) < sizeof(std::int64_t)) {
          cell = cell_scores[lane];
        }

        current.scores[offset + lane] = cell;

        if constexpr (TrackDirections) {
          directions[direction_index(row, column)] = direction;
        }

        if constexpr (LocalAlignment) {
          if (cell > best_score) {
            best_score = cell;
            best_row = row;
            best_column = column;
          }
        } else if (row == query.size() && column == target.size()) {
          final_score = cell;
        }
      }
    }

    previous_previous = std::move(previous);
    previous = std::move(current);
  }

  if constexpr (!TrackDirections) {
    if constexpr (LocalAlignment) {
      return static_cast<Score>(best_score);
    }
    return static_cast<Score>(final_score);
  } else {
    if constexpr (!LocalAlignment) {
      best_row = query.size();
      best_column = target.size();
      best_score = final_score;
    }

    TracebackResult result;
    result.score = static_cast<Score>(best_score);
    result.query_end = best_row;
    result.target_end = best_column;

    std::size_t row = best_row;
    std::size_t column = best_column;

    while (row > 0 || column > 0) {
      const TraceDirection direction = directions[direction_index(row, column)];
      if (direction == TraceDirection::stop) {
        break;
      }

      if (direction == TraceDirection::diagonal) {
        result.operations.push_back(query[row - 1] == target[column - 1] ? '=' : 'X');
        --row;
        --column;
        continue;
      }

      if (direction == TraceDirection::up) {
        result.operations.push_back('D');
        --row;
        continue;
      }

      result.operations.push_back('I');
      --column;
    }

    std::reverse(result.operations.begin(), result.operations.end());
    result.query_start = row;
    result.target_start = column;
    return result;
  }
}

template <bool LocalAlignment>
Score dispatch_score(
    const PreparedAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  if (const auto fast = generic_detail::dispatch_fast_score<LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
      fast.has_value()) {
    return *fast;
  }

  switch (prepared.kernel_bits) {
    case KernelBits::bits8: {
      const auto& query = std::get<std::vector<std::uint8_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint8_t>>(prepared.target_tokens);
      return run_kernel<std::uint8_t, std::int8_t, LocalAlignment, false>(
          std::span<const std::uint8_t>(query.data(), query.size()),
          std::span<const std::uint8_t>(target.data(), target.size()),
          static_cast<std::int8_t>(match_score),
          static_cast<std::int8_t>(mismatch_score),
          static_cast<std::int8_t>(gap_score));
    }
    case KernelBits::bits16: {
      const auto& query = std::get<std::vector<std::uint16_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint16_t>>(prepared.target_tokens);
      return run_kernel<std::uint16_t, std::int16_t, LocalAlignment, false>(
          std::span<const std::uint16_t>(query.data(), query.size()),
          std::span<const std::uint16_t>(target.data(), target.size()),
          static_cast<std::int16_t>(match_score),
          static_cast<std::int16_t>(mismatch_score),
          static_cast<std::int16_t>(gap_score));
    }
    case KernelBits::bits32: {
      const auto& query = std::get<std::vector<std::uint32_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint32_t>>(prepared.target_tokens);
      return run_kernel<std::uint32_t, std::int32_t, LocalAlignment, false>(
          std::span<const std::uint32_t>(query.data(), query.size()),
          std::span<const std::uint32_t>(target.data(), target.size()),
          static_cast<std::int32_t>(match_score),
          static_cast<std::int32_t>(mismatch_score),
          static_cast<std::int32_t>(gap_score));
    }
    case KernelBits::bits64: {
      const auto& query = std::get<std::vector<std::uint64_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint64_t>>(prepared.target_tokens);
      return run_kernel<std::uint64_t, std::int64_t, LocalAlignment, false>(
          std::span<const std::uint64_t>(query.data(), query.size()),
          std::span<const std::uint64_t>(target.data(), target.size()),
          static_cast<std::int64_t>(match_score),
          static_cast<std::int64_t>(mismatch_score),
          static_cast<std::int64_t>(gap_score));
    }
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported kernel width");
  throw nb::python_error();
}

template <bool LocalAlignment>
AlignmentResult dispatch_traceback(
    const PreparedAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  if (const auto fast = generic_detail::dispatch_fast_traceback<LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
      fast.has_value()) {
    return *fast;
  }

  switch (prepared.kernel_bits) {
    case KernelBits::bits8: {
      const auto& query = std::get<std::vector<std::uint8_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint8_t>>(prepared.target_tokens);
      return generic_detail::build_alignment_result<std::uint8_t>(
          prepared,
          run_kernel<std::uint8_t, std::int8_t, LocalAlignment, true>(
              std::span<const std::uint8_t>(query.data(), query.size()),
              std::span<const std::uint8_t>(target.data(), target.size()),
              static_cast<std::int8_t>(match_score),
              static_cast<std::int8_t>(mismatch_score),
              static_cast<std::int8_t>(gap_score)),
          query,
          target);
    }
    case KernelBits::bits16: {
      const auto& query = std::get<std::vector<std::uint16_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint16_t>>(prepared.target_tokens);
      return generic_detail::build_alignment_result<std::uint16_t>(
          prepared,
          run_kernel<std::uint16_t, std::int16_t, LocalAlignment, true>(
              std::span<const std::uint16_t>(query.data(), query.size()),
              std::span<const std::uint16_t>(target.data(), target.size()),
              static_cast<std::int16_t>(match_score),
              static_cast<std::int16_t>(mismatch_score),
              static_cast<std::int16_t>(gap_score)),
          query,
          target);
    }
    case KernelBits::bits32: {
      const auto& query = std::get<std::vector<std::uint32_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint32_t>>(prepared.target_tokens);
      return generic_detail::build_alignment_result<std::uint32_t>(
          prepared,
          run_kernel<std::uint32_t, std::int32_t, LocalAlignment, true>(
              std::span<const std::uint32_t>(query.data(), query.size()),
              std::span<const std::uint32_t>(target.data(), target.size()),
              static_cast<std::int32_t>(match_score),
              static_cast<std::int32_t>(mismatch_score),
              static_cast<std::int32_t>(gap_score)),
          query,
          target);
    }
    case KernelBits::bits64: {
      const auto& query = std::get<std::vector<std::uint64_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint64_t>>(prepared.target_tokens);
      return generic_detail::build_alignment_result<std::uint64_t>(
          prepared,
          run_kernel<std::uint64_t, std::int64_t, LocalAlignment, true>(
              std::span<const std::uint64_t>(query.data(), query.size()),
              std::span<const std::uint64_t>(target.data(), target.size()),
              static_cast<std::int64_t>(match_score),
              static_cast<std::int64_t>(mismatch_score),
              static_cast<std::int64_t>(gap_score)),
          query,
          target);
    }
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported kernel width");
  throw nb::python_error();
}

}  // namespace detail

struct Implementation {
  using PreparedSmithWatermanFarrarScore =
      farrar_fixed_kernel::detail::PreparedScore<detail::SimdOps>;
  using PreparedAffineScore =
      farrar_fixed_kernel::detail::PreparedAffineScore<detail::SimdOps>;

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
      return detail::dispatch_score<true>(prepared, match_score, mismatch_score, gap_score);
    }
    const auto prepared =
        prepare_linear_score_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::dispatch_score<detail::SimdOps>(
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
    return farrar_fixed_kernel::detail::dispatch_score_many<detail::SimdOps, true>(
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

  static std::string smith_waterman_linear_cigar(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
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
    return farrar_fixed_kernel::detail::dispatch_score<detail::SimdOps>(
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
    return farrar_fixed_kernel::detail::prepare_score<detail::SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static Score smith_waterman_farrar_score_prepared(
      PreparedSmithWatermanFarrarScore& prepared) {
    return farrar_fixed_kernel::detail::dispatch_prepared_score<detail::SimdOps>(prepared);
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
      return affine_fixed_kernel::detail::dispatch_compact_byte_score<detail::SimdOps>(
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
    return farrar_fixed_kernel::detail::dispatch_affine_score_many<detail::SimdOps, true>(
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
    const auto path = farrar_fixed_kernel::detail::dispatch_affine_striped_path_info<detail::SimdOps, true>(
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
    return farrar_fixed_kernel::detail::dispatch_affine_striped_path_info<detail::SimdOps, true>(
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
    return affine_fixed_kernel::detail::dispatch_compact_byte_score<detail::SimdOps>(
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
    return farrar_fixed_kernel::detail::prepare_affine_score<detail::SimdOps>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static Score smith_waterman_affine_score_prepared(PreparedAffineScore& prepared) {
    return farrar_fixed_kernel::detail::dispatch_prepared_affine_score<detail::SimdOps>(prepared);
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
    return farrar_fixed_kernel::detail::prepare_affine_score<detail::SimdOps, true>(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static Score needleman_wunsch_affine_score_prepared(PreparedAffineScore& prepared) {
    return farrar_fixed_kernel::detail::dispatch_prepared_global_affine_score<detail::SimdOps>(prepared);
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
      return detail::dispatch_score<false>(prepared, match_score, mismatch_score, gap_score);
    }
    const auto prepared =
        prepare_linear_score_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return farrar_fixed_kernel::detail::dispatch_global_score<detail::SimdOps>(
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
    return farrar_fixed_kernel::detail::dispatch_score_many<detail::SimdOps, false>(
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
    return farrar_fixed_kernel::detail::dispatch_global_affine_score<detail::SimdOps>(
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
    return farrar_fixed_kernel::detail::dispatch_affine_score_many<detail::SimdOps, false>(
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
    const auto path = farrar_fixed_kernel::detail::dispatch_affine_striped_path_info<detail::SimdOps, false>(
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
    return farrar_fixed_kernel::detail::dispatch_affine_striped_path_info<detail::SimdOps, false>(
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

  static std::vector<Score> levenshtein_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    return ::stride_align::levenshtein_simd::levenshtein_scores_simd<
        ::stride_align::levenshtein_simd::SseOps>(query, targets, cutoff);
  }

  static std::vector<double> levenshtein_normalized_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    return ::stride_align::levenshtein_simd::levenshtein_normalized_scores_simd<
        ::stride_align::levenshtein_simd::SseOps>(query, targets, cutoff);
  }

  static std::vector<Score> damerau_levenshtein_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::osa_simd::osa_scores_simd<
        ::stride_align::levenshtein_simd::SseOps>(query, targets);
  }

  static std::vector<double> damerau_levenshtein_normalized_scores(
      nb::handle query, nb::handle targets) {
    return ::stride_align::osa_simd::osa_normalized_scores_simd<
        ::stride_align::levenshtein_simd::SseOps>(query, targets);
  }

  // Jaro / Jaro-Winkler multi-target SIMD batch. Single-word kernel,
  // one target per 64-bit lane. Returns an empty vector when inputs
  // don't fit (bytes-only, lens <= 64); the binding layer falls
  // through to scalar dispatch in that case.
  static std::vector<double> jaro_similarities(
      nb::handle query, nb::handle targets) {
    return ::stride_align::jaro_simd::jaro_similarities_simd<
        ::stride_align::levenshtein_simd::SseOps>(query, targets);
  }

  static std::vector<double> jaro_winkler_similarities(
      nb::handle query,
      nb::handle targets,
      double prefix_weight,
      double prefix_threshold,
      std::size_t prefix_cap) {
    return ::stride_align::jaro_simd::jaro_winkler_similarities_simd<
        ::stride_align::levenshtein_simd::SseOps>(
        query, targets, prefix_weight, prefix_threshold, prefix_cap);
  }

  static nb::object cdist(
      nb::handle queries, nb::handle targets, int scorer,
      nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    return ::stride_align::cdist_simd::cdist_impl<
        ::stride_align::levenshtein_simd::SseOps>(
        queries, targets, scorer, tqdm_factory, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static constexpr BackendKind backend_kind = BackendKind::x86_sse41;
};

}  // namespace stride_align::backend_sse41
