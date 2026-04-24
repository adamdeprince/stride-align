#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>

#include "backends/generic.hpp"

namespace stride_align::scalable_backend {

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

  const std::size_t lane_count = Ops::lane_count();
  if (lane_count == 0) {
    PyErr_SetString(PyExc_RuntimeError, "scalable backend reported zero active lanes");
    throw nb::python_error();
  }

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

  std::vector<Token> query_tokens(lane_count);
  std::vector<Token> target_tokens(lane_count);
  std::vector<Cell> diagonal_scores(lane_count);
  std::vector<Cell> up_scores(lane_count);
  std::vector<Cell> left_scores(lane_count);
  std::vector<Cell> cell_scores(lane_count);

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

    for (std::size_t offset = 0; offset < current.scores.size(); offset += lane_count) {
      const std::size_t count = std::min(lane_count, current.scores.size() - offset);

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
        up_scores[lane] = lookup_score<LocalAlignment>(previous, row - 1, column, gap_score);
        left_scores[lane] = lookup_score<LocalAlignment>(previous, row, column - 1, gap_score);
      }

      const auto gap_vector = Ops::set1(gap_score, count);
      const auto substitution_vector =
          Ops::substitution(query_tokens.data(), target_tokens.data(), match_score, mismatch_score, count);
      const auto diagonal_vector = Ops::add(Ops::load_cells(diagonal_scores.data(), count), substitution_vector, count);
      const auto up_vector = Ops::add(Ops::load_cells(up_scores.data(), count), gap_vector, count);
      const auto left_vector = Ops::add(Ops::load_cells(left_scores.data(), count), gap_vector, count);

      Ops::store_cells(diagonal_scores.data(), diagonal_vector, count);
      Ops::store_cells(up_scores.data(), up_vector, count);
      Ops::store_cells(left_scores.data(), left_vector, count);

      if constexpr (Ops::has_vector_max) {
        auto cell_vector = Ops::max(diagonal_vector, Ops::max(up_vector, left_vector, count), count);
        if constexpr (LocalAlignment) {
          cell_vector = Ops::max(cell_vector, Ops::zero(count), count);
        }
        Ops::store_cells(cell_scores.data(), cell_vector, count);
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
        if constexpr (Ops::has_vector_max) {
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

    return TracebackResult{
        .score = static_cast<Score>(best_score),
        .end_row = best_row,
        .end_column = best_column,
        .directions = std::move(directions),
        .column_count = column_count,
    };
  }
}

template <template <typename, typename> class OpsTemplate, bool LocalAlignment>
Score dispatch_score(
    const PreparedAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
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

}  // namespace stride_align::scalable_backend
