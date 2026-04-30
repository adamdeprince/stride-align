#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>

#include "backends/farrar_scalable_kernel.hpp"
#include "backends/generic.hpp"
#include "farrar_preprocess.hpp"

namespace stride_align::affine_scalable_kernel {

namespace nb = nanobind;
namespace generic_detail = stride_align::backend_generic::detail;

namespace detail {

enum class TraceState : std::uint8_t {
  h,
  e,
  f,
};

template <typename Cell>
struct ScoreToken;

template <>
struct ScoreToken<std::int8_t> {
  using type = std::uint8_t;
};

template <>
struct ScoreToken<std::int16_t> {
  using type = std::uint16_t;
};

template <>
struct ScoreToken<std::int32_t> {
  using type = std::uint32_t;
};

template <>
struct ScoreToken<std::int64_t> {
  using type = std::uint64_t;
};

template <typename Cell>
struct DiagonalState {
  std::size_t row_start = 1;
  std::vector<Cell> h_scores;
  std::vector<Cell> e_scores;
  std::vector<Cell> f_scores;
};

template <typename Cell>
inline constexpr Cell negative_infinity_v = std::numeric_limits<Cell>::min();

template <typename Cell>
Cell safe_add(Cell lhs, Cell rhs) noexcept {
  if (lhs == negative_infinity_v<Cell>) {
    return negative_infinity_v<Cell>;
  }
  return static_cast<Cell>(static_cast<Score>(lhs) + static_cast<Score>(rhs));
}

template <typename Cell>
Cell gap_cost(std::size_t length, Cell gap_open_score, Cell gap_extend_score) noexcept {
  if (length == 0) {
    return 0;
  }
  return static_cast<Cell>(
      static_cast<Score>(gap_open_score) +
      static_cast<Score>(length - 1U) * static_cast<Score>(gap_extend_score));
}

template <bool LocalAlignment, typename Cell>
Cell boundary_h(std::size_t row, std::size_t column, Cell gap_open_score, Cell gap_extend_score) {
  if constexpr (LocalAlignment) {
    return 0;
  }

  if (row == 0 && column == 0) {
    return 0;
  }
  if (row == 0) {
    return gap_cost(column, gap_open_score, gap_extend_score);
  }
  if (column == 0) {
    return gap_cost(row, gap_open_score, gap_extend_score);
  }
  return negative_infinity_v<Cell>;
}

template <bool LocalAlignment, typename Cell>
Cell boundary_e(std::size_t row, std::size_t column, Cell gap_open_score, Cell gap_extend_score) {
  if constexpr (LocalAlignment) {
    return negative_infinity_v<Cell>;
  }

  if (column == 0 && row > 0) {
    return gap_cost(row, gap_open_score, gap_extend_score);
  }
  return negative_infinity_v<Cell>;
}

template <bool LocalAlignment, typename Cell>
Cell boundary_f(std::size_t row, std::size_t column, Cell gap_open_score, Cell gap_extend_score) {
  if constexpr (LocalAlignment) {
    return negative_infinity_v<Cell>;
  }

  if (row == 0 && column > 0) {
    return gap_cost(column, gap_open_score, gap_extend_score);
  }
  return negative_infinity_v<Cell>;
}

template <typename Cell>
bool contains_row(const DiagonalState<Cell>& diagonal, std::size_t row) noexcept {
  const std::size_t diagonal_end = diagonal.row_start + diagonal.h_scores.size();
  return !diagonal.h_scores.empty() && row >= diagonal.row_start && row < diagonal_end;
}

template <bool LocalAlignment, typename Cell>
Cell lookup_h(
    const DiagonalState<Cell>& diagonal,
    std::size_t row,
    std::size_t column,
    Cell gap_open_score,
    Cell gap_extend_score) {
  if (row == 0 || column == 0) {
    return boundary_h<LocalAlignment>(row, column, gap_open_score, gap_extend_score);
  }
  if (!contains_row(diagonal, row)) {
    return LocalAlignment ? Cell{0} : negative_infinity_v<Cell>;
  }
  return diagonal.h_scores[row - diagonal.row_start];
}

template <bool LocalAlignment, typename Cell>
Cell lookup_e(
    const DiagonalState<Cell>& diagonal,
    std::size_t row,
    std::size_t column,
    Cell gap_open_score,
    Cell gap_extend_score) {
  if (row == 0 || column == 0) {
    return boundary_e<LocalAlignment>(row, column, gap_open_score, gap_extend_score);
  }
  if (!contains_row(diagonal, row)) {
    return negative_infinity_v<Cell>;
  }
  return diagonal.e_scores[row - diagonal.row_start];
}

template <bool LocalAlignment, typename Cell>
Cell lookup_f(
    const DiagonalState<Cell>& diagonal,
    std::size_t row,
    std::size_t column,
    Cell gap_open_score,
    Cell gap_extend_score) {
  if (row == 0 || column == 0) {
    return boundary_f<LocalAlignment>(row, column, gap_open_score, gap_extend_score);
  }
  if (!contains_row(diagonal, row)) {
    return negative_infinity_v<Cell>;
  }
  return diagonal.f_scores[row - diagonal.row_start];
}

template <typename Token, typename Cell>
Cell substitution_score(
    Token query_token,
    Token target_token,
    Cell match_score,
    Cell mismatch_score) noexcept {
  return query_token == target_token ? match_score : mismatch_score;
}

template <typename Cell>
bool local_best_is_better(
    Cell candidate_score,
    std::size_t candidate_row,
    std::size_t candidate_column,
    Cell best_score,
    std::size_t best_row,
    std::size_t best_column) noexcept {
  if (candidate_score > best_score) {
    return true;
  }
  return candidate_score > 0 && candidate_score == best_score &&
      (best_row == 0 || candidate_row < best_row ||
       (candidate_row == best_row && candidate_column < best_column));
}

template <bool TrackDirections>
struct DirectionStorage {
  std::vector<generic_detail::TraceDirection> h_sources;
  std::vector<std::uint8_t> e_extends;
  std::vector<std::uint8_t> f_extends;
};

template <bool TrackDirections, bool LocalAlignment>
void initialize_boundaries(
    DirectionStorage<TrackDirections>& directions,
    std::size_t row_count,
    std::size_t column_count) {
  if constexpr (TrackDirections) {
    directions.h_sources.assign(
        row_count * column_count,
        generic_detail::TraceDirection::stop);
    directions.e_extends.assign(row_count * column_count, 0);
    directions.f_extends.assign(row_count * column_count, 0);
    if constexpr (!LocalAlignment) {
      for (std::size_t row = 1; row < row_count; ++row) {
        directions.h_sources[row * column_count] = generic_detail::TraceDirection::up;
        directions.e_extends[row * column_count] = row > 1 ? 1 : 0;
      }
      for (std::size_t column = 1; column < column_count; ++column) {
        directions.h_sources[column] = generic_detail::TraceDirection::left;
        directions.f_extends[column] = column > 1 ? 1 : 0;
      }
    }
  }
}

template <typename Token, typename Cell, bool TrackDirections>
generic_detail::TracebackResult traceback_result(
    std::span<const Token> query,
    std::span<const Token> target,
    const DirectionStorage<TrackDirections>& directions,
    Cell score,
    std::size_t best_row,
    std::size_t best_column) {
  static_assert(TrackDirections);

  const std::size_t column_count = target.size() + 1U;
  const auto index = [column_count](std::size_t row, std::size_t column) {
    return row * column_count + column;
  };

  generic_detail::TracebackResult result;
  result.score = static_cast<Score>(score);
  result.query_end = best_row;
  result.target_end = best_column;

  std::size_t row = best_row;
  std::size_t column = best_column;
  TraceState state = TraceState::h;

  while (row > 0 || column > 0) {
    const std::size_t cell_index = index(row, column);
    if (state == TraceState::h) {
      const auto direction = directions.h_sources[cell_index];
      if (direction == generic_detail::TraceDirection::stop) {
        break;
      }
      if (direction == generic_detail::TraceDirection::diagonal) {
        result.operations.push_back(query[row - 1U] == target[column - 1U] ? 'M' : 'X');
        --row;
        --column;
        continue;
      }
      state = direction == generic_detail::TraceDirection::up ? TraceState::e : TraceState::f;
      continue;
    }

    if (state == TraceState::e) {
      result.operations.push_back('D');
      const bool continues_gap = directions.e_extends[cell_index] != 0;
      --row;
      state = continues_gap ? TraceState::e : TraceState::h;
      continue;
    }

    result.operations.push_back('I');
    const bool continues_gap = directions.f_extends[cell_index] != 0;
    --column;
    state = continues_gap ? TraceState::f : TraceState::h;
  }

  std::reverse(result.operations.begin(), result.operations.end());
  result.query_start = row;
  result.target_start = column;
  return result;
}

template <
    template <typename, typename> class OpsTemplate,
    typename Token,
    typename Cell,
    bool LocalAlignment,
    bool TrackDirections,
    bool CompactByteTokens>
std::conditional_t<TrackDirections, generic_detail::TracebackResult, Score> run_kernel(
    std::span<const Token> query,
    std::span<const Token> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_open_score,
    Cell gap_extend_score) {
  using VectorToken = std::conditional_t<
      CompactByteTokens,
      typename ScoreToken<Cell>::type,
      Token>;
  using Ops = OpsTemplate<VectorToken, Cell>;

  const std::size_t lane_count = Ops::lane_count();
  if (lane_count == 0) {
    PyErr_SetString(PyExc_RuntimeError, "scalable affine backend reported zero active lanes");
    throw nb::python_error();
  }

  const std::size_t row_count = query.size() + 1U;
  const std::size_t column_count = target.size() + 1U;
  const auto direction_index = [column_count](std::size_t row, std::size_t column) {
    return row * column_count + column;
  };

  DirectionStorage<TrackDirections> directions;
  initialize_boundaries<TrackDirections, LocalAlignment>(directions, row_count, column_count);

  DiagonalState<Cell> previous_previous;
  DiagonalState<Cell> previous;

  std::vector<VectorToken> vector_query_tokens(lane_count);
  std::vector<VectorToken> vector_target_tokens(lane_count);
  std::vector<Cell> diagonal_scores(lane_count);
  std::vector<Cell> e_open_scores(lane_count);
  std::vector<Cell> e_extend_scores(lane_count);
  std::vector<Cell> f_open_scores(lane_count);
  std::vector<Cell> f_extend_scores(lane_count);
  std::vector<Cell> e_scores(lane_count);
  std::vector<Cell> f_scores(lane_count);
  std::vector<Cell> h_scores(lane_count);
  std::vector<std::uint8_t> e_extends(lane_count);
  std::vector<std::uint8_t> f_extends(lane_count);

  Cell best_score = 0;
  std::size_t best_row = 0;
  std::size_t best_column = 0;
  Cell final_score = boundary_h<false>(query.size(), target.size(), gap_open_score, gap_extend_score);

  for (std::size_t diagonal = 2; diagonal <= query.size() + target.size(); ++diagonal) {
    const std::size_t current_row_start = diagonal > target.size() ? diagonal - target.size() : 1;
    const std::size_t current_row_end = std::min(query.size(), diagonal - 1U);
    if (current_row_start > current_row_end) {
      continue;
    }

    DiagonalState<Cell> current;
    current.row_start = current_row_start;
    const std::size_t current_size = current_row_end - current_row_start + 1U;
    current.h_scores.assign(current_size, 0);
    current.e_scores.assign(current_size, negative_infinity_v<Cell>);
    current.f_scores.assign(current_size, negative_infinity_v<Cell>);

    for (std::size_t offset = 0; offset < current_size; offset += lane_count) {
      const std::size_t count = std::min(lane_count, current_size - offset);

      for (std::size_t lane = 0; lane < count; ++lane) {
        const std::size_t row = current.row_start + offset + lane;
        const std::size_t column = diagonal - row;

        const auto query_token = query[row - 1U];
        const auto target_token = target[column - 1U];

        if constexpr (CompactByteTokens) {
          diagonal_scores[lane] = safe_add(
              lookup_h<LocalAlignment>(
                  previous_previous,
                  row - 1U,
                  column - 1U,
                  gap_open_score,
                  gap_extend_score),
              substitution_score(query_token, target_token, match_score, mismatch_score));
        } else {
          vector_query_tokens[lane] = static_cast<VectorToken>(query_token);
          vector_target_tokens[lane] = static_cast<VectorToken>(target_token);
          diagonal_scores[lane] = lookup_h<LocalAlignment>(
              previous_previous,
              row - 1U,
              column - 1U,
              gap_open_score,
              gap_extend_score);
        }

        const Cell up_h = lookup_h<LocalAlignment>(
            previous,
            row - 1U,
            column,
            gap_open_score,
            gap_extend_score);
        const Cell up_e = lookup_e<LocalAlignment>(
            previous,
            row - 1U,
            column,
            gap_open_score,
            gap_extend_score);
        const Cell left_h = lookup_h<LocalAlignment>(
            previous,
            row,
            column - 1U,
            gap_open_score,
            gap_extend_score);
        const Cell left_f = lookup_f<LocalAlignment>(
            previous,
            row,
            column - 1U,
            gap_open_score,
            gap_extend_score);

        e_open_scores[lane] = safe_add(up_h, gap_open_score);
        e_extend_scores[lane] = safe_add(up_e, gap_extend_score);
        f_open_scores[lane] = safe_add(left_h, gap_open_score);
        f_extend_scores[lane] = safe_add(left_f, gap_extend_score);
        e_extends[lane] =
            up_e != negative_infinity_v<Cell> && e_extend_scores[lane] >= e_open_scores[lane];
        f_extends[lane] =
            left_f != negative_infinity_v<Cell> && f_extend_scores[lane] >= f_open_scores[lane];
      }

      auto diagonal_vector = Ops::load_cells(diagonal_scores.data(), count);
      if constexpr (!CompactByteTokens) {
        const auto substitution_vector = Ops::substitution(
            vector_query_tokens.data(),
            vector_target_tokens.data(),
            match_score,
            mismatch_score,
            count);
        diagonal_vector = Ops::add(diagonal_vector, substitution_vector, count);
        Ops::store_cells(diagonal_scores.data(), diagonal_vector, count);
      }

      const auto e_vector = Ops::max(
          Ops::load_cells(e_open_scores.data(), count),
          Ops::load_cells(e_extend_scores.data(), count),
          count);
      const auto f_vector = Ops::max(
          Ops::load_cells(f_open_scores.data(), count),
          Ops::load_cells(f_extend_scores.data(), count),
          count);
      auto h_vector = Ops::max(diagonal_vector, Ops::max(e_vector, f_vector, count), count);
      if constexpr (LocalAlignment) {
        h_vector = Ops::max(h_vector, Ops::zero(count), count);
      }

      Ops::store_cells(e_scores.data(), e_vector, count);
      Ops::store_cells(f_scores.data(), f_vector, count);
      Ops::store_cells(h_scores.data(), h_vector, count);

      for (std::size_t lane = 0; lane < count; ++lane) {
        const std::size_t row = current.row_start + offset + lane;
        const std::size_t column = diagonal - row;

        generic_detail::TraceDirection direction = generic_detail::TraceDirection::diagonal;
        Cell selected_score = diagonal_scores[lane];
        if (e_scores[lane] > selected_score) {
          selected_score = e_scores[lane];
          direction = generic_detail::TraceDirection::up;
        }
        if (f_scores[lane] > selected_score) {
          selected_score = f_scores[lane];
          direction = generic_detail::TraceDirection::left;
        }
        if constexpr (LocalAlignment) {
          if (selected_score <= 0) {
            selected_score = 0;
            direction = generic_detail::TraceDirection::stop;
          }
        }

        current.h_scores[offset + lane] = h_scores[lane];
        current.e_scores[offset + lane] = e_scores[lane];
        current.f_scores[offset + lane] = f_scores[lane];

        if constexpr (TrackDirections) {
          const auto cell_index = direction_index(row, column);
          directions.h_sources[cell_index] = direction;
          directions.e_extends[cell_index] = e_extends[lane] != 0 ? 1 : 0;
          directions.f_extends[cell_index] = f_extends[lane] != 0 ? 1 : 0;
        }

        if constexpr (LocalAlignment) {
          if (local_best_is_better(
                  h_scores[lane],
                  row,
                  column,
                  best_score,
                  best_row,
                  best_column)) {
            best_score = h_scores[lane];
            best_row = row;
            best_column = column;
          }
        } else if (row == query.size() && column == target.size()) {
          final_score = h_scores[lane];
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
    return traceback_result<Token, Cell, TrackDirections>(
        query,
        target,
        directions,
        best_score,
        best_row,
        best_column);
  }
}

template <
    template <typename, typename> class OpsTemplate,
    typename Token,
    typename Cell,
    bool LocalAlignment,
    bool TrackDirections>
std::conditional_t<TrackDirections, generic_detail::TracebackResult, Score> run_kernel(
    std::span<const Token> query,
    std::span<const Token> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_open_score,
    Cell gap_extend_score) {
  return run_kernel<OpsTemplate, Token, Cell, LocalAlignment, TrackDirections, false>(
      query,
      target,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score);
}

template <template <typename, typename> class OpsTemplate, typename Cell>
Score run_compact_byte_score(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  return run_kernel<
      OpsTemplate,
      std::uint8_t,
      Cell,
      true,
      false,
      true>(
      query,
      target,
      static_cast<Cell>(match_score),
      static_cast<Cell>(mismatch_score),
      static_cast<Cell>(gap_open_score),
      static_cast<Cell>(gap_extend_score));
}

template <template <typename, typename> class OpsTemplate, bool LocalAlignment>
Score dispatch_score(
    const PreparedAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  switch (prepared.kernel_bits) {
    case KernelBits::bits8: {
      const auto& query = std::get<std::vector<std::uint8_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint8_t>>(prepared.target_tokens);
      return run_kernel<OpsTemplate, std::uint8_t, std::int8_t, LocalAlignment, false>(
          std::span<const std::uint8_t>(query.data(), query.size()),
          std::span<const std::uint8_t>(target.data(), target.size()),
          static_cast<std::int8_t>(match_score),
          static_cast<std::int8_t>(mismatch_score),
          static_cast<std::int8_t>(gap_open_score),
          static_cast<std::int8_t>(gap_extend_score));
    }
    case KernelBits::bits16: {
      const auto& query = std::get<std::vector<std::uint16_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint16_t>>(prepared.target_tokens);
      return run_kernel<OpsTemplate, std::uint16_t, std::int16_t, LocalAlignment, false>(
          std::span<const std::uint16_t>(query.data(), query.size()),
          std::span<const std::uint16_t>(target.data(), target.size()),
          static_cast<std::int16_t>(match_score),
          static_cast<std::int16_t>(mismatch_score),
          static_cast<std::int16_t>(gap_open_score),
          static_cast<std::int16_t>(gap_extend_score));
    }
    case KernelBits::bits32: {
      const auto& query = std::get<std::vector<std::uint32_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint32_t>>(prepared.target_tokens);
      return run_kernel<OpsTemplate, std::uint32_t, std::int32_t, LocalAlignment, false>(
          std::span<const std::uint32_t>(query.data(), query.size()),
          std::span<const std::uint32_t>(target.data(), target.size()),
          static_cast<std::int32_t>(match_score),
          static_cast<std::int32_t>(mismatch_score),
          static_cast<std::int32_t>(gap_open_score),
          static_cast<std::int32_t>(gap_extend_score));
    }
    case KernelBits::bits64: {
      const auto& query = std::get<std::vector<std::uint64_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint64_t>>(prepared.target_tokens);
      return run_kernel<OpsTemplate, std::uint64_t, std::int64_t, LocalAlignment, false>(
          std::span<const std::uint64_t>(query.data(), query.size()),
          std::span<const std::uint64_t>(target.data(), target.size()),
          static_cast<std::int64_t>(match_score),
          static_cast<std::int64_t>(mismatch_score),
          static_cast<std::int64_t>(gap_open_score),
          static_cast<std::int64_t>(gap_extend_score));
    }
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported scalable affine kernel width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate, bool LocalAlignment>
AlignmentResult dispatch_traceback(
    const PreparedAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
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
              static_cast<std::int8_t>(gap_open_score),
              static_cast<std::int8_t>(gap_extend_score)),
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
              static_cast<std::int16_t>(gap_open_score),
              static_cast<std::int16_t>(gap_extend_score)),
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
              static_cast<std::int32_t>(gap_open_score),
              static_cast<std::int32_t>(gap_extend_score)),
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
              static_cast<std::int64_t>(gap_open_score),
              static_cast<std::int64_t>(gap_extend_score)),
          query,
          target);
    }
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported scalable affine kernel width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate>
Score dispatch_compact_byte_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return farrar_scalable_kernel::detail::affine_score<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits16:
      return farrar_scalable_kernel::detail::affine_score<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits32:
      return farrar_scalable_kernel::detail::affine_score<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits64:
      return farrar_scalable_kernel::detail::affine_score<OpsTemplate, std::int64_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported scalable affine Farrar score width");
  throw nb::python_error();
}

}  // namespace detail

}  // namespace stride_align::affine_scalable_kernel
