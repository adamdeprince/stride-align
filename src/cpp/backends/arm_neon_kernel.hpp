#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>

#include "backends/generic.hpp"

namespace stride_align::arm_neon_kernel {

namespace nb = nanobind;
namespace generic_detail = stride_align::backend_generic::detail;

namespace detail {

template <typename Cell>
struct DiagonalState {
  std::size_t row_start = 1;
  std::vector<Cell> scores;
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

template <template <typename, typename> class OpsTemplate, typename Token, typename Cell, bool TrackDirections>
using KernelResult = std::conditional_t<TrackDirections, generic_detail::TracebackResult, Score>;

template <
    template <typename, typename> class OpsTemplate,
    typename Token,
    typename Cell,
    bool LocalAlignment,
    bool TrackDirections>
KernelResult<OpsTemplate, Token, Cell, TrackDirections> run_kernel(
    std::span<const Token> query,
    std::span<const Token> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  using TraceDirection = generic_detail::TraceDirection;
  using TracebackResult = generic_detail::TracebackResult;
  using Ops = OpsTemplate<Token, Cell>;

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

  struct Scratch {
    alignas(Ops::alignment) Token query_tokens[Ops::lane_count] = {};
    alignas(Ops::alignment) Token target_tokens[Ops::lane_count] = {};
    alignas(Ops::alignment) Cell diagonal_scores[Ops::lane_count] = {};
    alignas(Ops::alignment) Cell up_scores[Ops::lane_count] = {};
    alignas(Ops::alignment) Cell left_scores[Ops::lane_count] = {};
    alignas(Ops::alignment) Cell cell_scores[Ops::lane_count] = {};
  };

  DiagonalState<Cell> previous_previous;
  DiagonalState<Cell> previous;
  Scratch scratch;

  Cell best_score = 0;
  std::size_t best_row = 0;
  std::size_t best_column = 0;
  Cell final_score = boundary_score<false>(query.size(), target.size(), gap_score);

  constexpr std::size_t lane_count = Ops::lane_count;
  const auto gap_vector = Ops::set1(gap_score);

  for (std::size_t diagonal = 2; diagonal <= query.size() + target.size(); ++diagonal) {
    const std::size_t current_row_start = diagonal > target.size() ? diagonal - target.size() : 1;
    const std::size_t current_row_end = std::min(query.size(), diagonal - 1);
    if (current_row_start > current_row_end) {
      continue;
    }

    DiagonalState<Cell> current;
    current.row_start = current_row_start;
    current.scores.assign(current_row_end - current_row_start + 1, 0);

    for (std::size_t offset = 0; offset < current.scores.size(); offset += lane_count) {
      const std::size_t count = std::min(lane_count, current.scores.size() - offset);

      for (std::size_t lane = 0; lane < count; ++lane) {
        const std::size_t row = current.row_start + offset + lane;
        const std::size_t column = diagonal - row;

        scratch.query_tokens[lane] = query[row - 1];
        scratch.target_tokens[lane] = target[column - 1];
        scratch.diagonal_scores[lane] = lookup_score<LocalAlignment>(
            previous_previous,
            row - 1,
            column - 1,
            gap_score);
        scratch.up_scores[lane] = lookup_score<LocalAlignment>(previous, row - 1, column, gap_score);
        scratch.left_scores[lane] = lookup_score<LocalAlignment>(previous, row, column - 1, gap_score);
      }

      const auto substitution_vector =
          Ops::substitution(scratch.query_tokens, scratch.target_tokens, match_score, mismatch_score);
      const auto diagonal_vector = Ops::add(Ops::load_cells(scratch.diagonal_scores), substitution_vector);
      const auto up_vector = Ops::add(Ops::load_cells(scratch.up_scores), gap_vector);
      const auto left_vector = Ops::add(Ops::load_cells(scratch.left_scores), gap_vector);

      Ops::store_cells(scratch.diagonal_scores, diagonal_vector);
      Ops::store_cells(scratch.up_scores, up_vector);
      Ops::store_cells(scratch.left_scores, left_vector);

      if constexpr (Ops::has_vector_max) {
        auto cell_vector = Ops::max(diagonal_vector, Ops::max(up_vector, left_vector));
        if constexpr (LocalAlignment) {
          cell_vector = Ops::max(cell_vector, Ops::zero());
        }
        Ops::store_cells(scratch.cell_scores, cell_vector);
      }

      for (std::size_t lane = 0; lane < count; ++lane) {
        const std::size_t row = current.row_start + offset + lane;
        const std::size_t column = diagonal - row;

        Cell selected_score = scratch.diagonal_scores[lane];
        TraceDirection direction = TraceDirection::diagonal;

        if (scratch.up_scores[lane] > selected_score) {
          selected_score = scratch.up_scores[lane];
          direction = TraceDirection::up;
        }

        if (scratch.left_scores[lane] > selected_score) {
          selected_score = scratch.left_scores[lane];
          direction = TraceDirection::left;
        }

        if constexpr (LocalAlignment) {
          if (selected_score <= 0) {
            selected_score = 0;
            direction = TraceDirection::stop;
          }
        }

        Cell cell = selected_score;
        if constexpr (Ops::has_vector_max) {
          cell = scratch.cell_scores[lane];
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

template <template <typename, typename> class OpsTemplate, bool LocalAlignment>
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
      return run_kernel<OpsTemplate, std::uint8_t, std::int8_t, LocalAlignment, false>(
          std::span<const std::uint8_t>(query.data(), query.size()),
          std::span<const std::uint8_t>(target.data(), target.size()),
          static_cast<std::int8_t>(match_score),
          static_cast<std::int8_t>(mismatch_score),
          static_cast<std::int8_t>(gap_score));
    }
    case KernelBits::bits16: {
      const auto& query = std::get<std::vector<std::uint16_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint16_t>>(prepared.target_tokens);
      return run_kernel<OpsTemplate, std::uint16_t, std::int16_t, LocalAlignment, false>(
          std::span<const std::uint16_t>(query.data(), query.size()),
          std::span<const std::uint16_t>(target.data(), target.size()),
          static_cast<std::int16_t>(match_score),
          static_cast<std::int16_t>(mismatch_score),
          static_cast<std::int16_t>(gap_score));
    }
    case KernelBits::bits32: {
      const auto& query = std::get<std::vector<std::uint32_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint32_t>>(prepared.target_tokens);
      return run_kernel<OpsTemplate, std::uint32_t, std::int32_t, LocalAlignment, false>(
          std::span<const std::uint32_t>(query.data(), query.size()),
          std::span<const std::uint32_t>(target.data(), target.size()),
          static_cast<std::int32_t>(match_score),
          static_cast<std::int32_t>(mismatch_score),
          static_cast<std::int32_t>(gap_score));
    }
    case KernelBits::bits64: {
      const auto& query = std::get<std::vector<std::uint64_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint64_t>>(prepared.target_tokens);
      return run_kernel<OpsTemplate, std::uint64_t, std::int64_t, LocalAlignment, false>(
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

template <template <typename, typename> class OpsTemplate, bool LocalAlignment>
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
          run_kernel<OpsTemplate, std::uint8_t, std::int8_t, LocalAlignment, true>(
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
          run_kernel<OpsTemplate, std::uint16_t, std::int16_t, LocalAlignment, true>(
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
          run_kernel<OpsTemplate, std::uint32_t, std::int32_t, LocalAlignment, true>(
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
          run_kernel<OpsTemplate, std::uint64_t, std::int64_t, LocalAlignment, true>(
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

}  // namespace stride_align::arm_neon_kernel
