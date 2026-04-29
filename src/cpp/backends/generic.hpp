#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <nanobind/nanobind.h>

#include "cpu.hpp"
#include "farrar_preprocess.hpp"
#include "preprocess.hpp"
#include "stride_align/alignment.hpp"

namespace stride_align::backend_generic {

namespace nb = nanobind;

namespace detail {

enum class TraceDirection : std::uint8_t {
  stop,
  diagonal,
  up,
  left,
};

struct TracebackResult {
  Score score = 0;
  std::size_t query_start = 0;
  std::size_t query_end = 0;
  std::size_t target_start = 0;
  std::size_t target_end = 0;
  std::string operations;
};

template <typename Token, typename Cell>
inline Cell substitution_score(
    Token query_base,
    Token target_base,
    Cell match_score,
    Cell mismatch_score) noexcept {
  static_assert(std::is_integral_v<Token>);
  static_assert(std::is_integral_v<Cell>);
  static_assert(sizeof(Token) == sizeof(Cell));
  return query_base == target_base ? match_score : mismatch_score;
}

template <typename Token, typename Cell, bool LocalAlignment>
Score score_only(
    std::span<const Token> query,
    std::span<const Token> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  std::vector<Cell> previous(target.size() + 1, 0);
  std::vector<Cell> current(target.size() + 1, 0);

  if constexpr (!LocalAlignment) {
    for (std::size_t column = 0; column <= target.size(); ++column) {
      previous[column] = static_cast<Cell>(static_cast<Score>(column) * gap_score);
    }
  }

  Cell best_score = 0;
  if constexpr (!LocalAlignment) {
    best_score = previous.back();
  }

  for (std::size_t row = 1; row <= query.size(); ++row) {
    current[0] = LocalAlignment ? 0 : static_cast<Cell>(static_cast<Score>(row) * gap_score);

    for (std::size_t column = 1; column <= target.size(); ++column) {
      const Cell diagonal = static_cast<Cell>(
          previous[column - 1] +
          substitution_score<Token, Cell>(
              query[row - 1],
              target[column - 1],
              match_score,
              mismatch_score));
      const Cell up = static_cast<Cell>(previous[column] + gap_score);
      const Cell left = static_cast<Cell>(current[column - 1] + gap_score);

      Cell cell = std::max({diagonal, up, left});
      if constexpr (LocalAlignment) {
        cell = std::max<Cell>(0, cell);
        if (cell > best_score) {
          best_score = cell;
        }
      }

      current[column] = cell;
    }

    std::swap(previous, current);
  }

  if constexpr (LocalAlignment) {
    return static_cast<Score>(best_score);
  }

  return static_cast<Score>(previous.back());
}

template <typename Cell>
Score farrar_score_only(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  std::vector<Cell> previous(target.size() + 1, 0);
  std::vector<Cell> current(target.size() + 1, 0);
  Cell best_score = 0;

  for (std::size_t row = 1; row <= query.size(); ++row) {
    current[0] = 0;
    for (std::size_t column = 1; column <= target.size(); ++column) {
      const Cell diagonal = static_cast<Cell>(
          previous[column - 1] +
          (query[row - 1] == target[column - 1] ? match_score : mismatch_score));
      const Cell up = static_cast<Cell>(previous[column] + gap_score);
      const Cell left = static_cast<Cell>(current[column - 1] + gap_score);
      const Cell cell = std::max<Cell>(0, std::max({diagonal, up, left}));
      current[column] = cell;
      best_score = std::max(best_score, cell);
    }
    std::swap(previous, current);
  }

  return static_cast<Score>(best_score);
}

template <typename Token, typename Cell, bool LocalAlignment>
TracebackResult traceback(
    std::span<const Token> query,
    std::span<const Token> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  const std::size_t row_count = query.size() + 1;
  const std::size_t column_count = target.size() + 1;
  std::vector<Cell> scores(row_count * column_count, 0);
  std::vector<TraceDirection> directions(row_count * column_count, TraceDirection::stop);

  const auto index = [column_count](std::size_t row, std::size_t column) {
    return row * column_count + column;
  };

  if constexpr (!LocalAlignment) {
    for (std::size_t row = 1; row < row_count; ++row) {
      scores[index(row, 0)] = static_cast<Cell>(static_cast<Score>(row) * gap_score);
      directions[index(row, 0)] = TraceDirection::up;
    }

    for (std::size_t column = 1; column < column_count; ++column) {
      scores[index(0, column)] = static_cast<Cell>(static_cast<Score>(column) * gap_score);
      directions[index(0, column)] = TraceDirection::left;
    }
  }

  Cell best_score = 0;
  std::size_t best_row = 0;
  std::size_t best_column = 0;

  for (std::size_t row = 1; row < row_count; ++row) {
    for (std::size_t column = 1; column < column_count; ++column) {
      const Cell diagonal = static_cast<Cell>(
          scores[index(row - 1, column - 1)] +
          substitution_score<Token, Cell>(
              query[row - 1],
              target[column - 1],
              match_score,
              mismatch_score));
      const Cell up = static_cast<Cell>(scores[index(row - 1, column)] + gap_score);
      const Cell left = static_cast<Cell>(scores[index(row, column - 1)] + gap_score);

      Cell cell = diagonal;
      TraceDirection direction = TraceDirection::diagonal;

      if (up > cell) {
        cell = up;
        direction = TraceDirection::up;
      }
      if (left > cell) {
        cell = left;
        direction = TraceDirection::left;
      }

      if constexpr (LocalAlignment) {
        if (cell <= 0) {
          cell = 0;
          direction = TraceDirection::stop;
        }
      }

      scores[index(row, column)] = cell;
      directions[index(row, column)] = direction;

      if constexpr (LocalAlignment) {
        if (cell > best_score) {
          best_score = cell;
          best_row = row;
          best_column = column;
        }
      }
    }
  }

  if constexpr (!LocalAlignment) {
    best_row = query.size();
    best_column = target.size();
    best_score = scores[index(best_row, best_column)];
  }

  TracebackResult result;
  result.score = static_cast<Score>(best_score);
  result.query_end = best_row;
  result.target_end = best_column;

  std::size_t row = best_row;
  std::size_t column = best_column;

  while (row > 0 || column > 0) {
    const TraceDirection direction = directions[index(row, column)];
    if (direction == TraceDirection::stop) {
      break;
    }

    if (direction == TraceDirection::diagonal) {
      result.operations.push_back(query[row - 1] == target[column - 1] ? 'M' : 'X');
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

template <typename Token>
AlignmentResult build_alignment_result(
    const PreparedAlignment& prepared,
    const TracebackResult& trace,
    const std::vector<Token>& query_tokens,
    const std::vector<Token>& target_tokens) {
  AlignmentResult result;
  result.score = trace.score;
  result.query_start = trace.query_start;
  result.query_end = trace.query_end;
  result.target_start = trace.target_start;
  result.target_end = trace.target_end;
  result.operations = trace.operations;
  result.aligned_query = materialize_query_output<Token>(
      prepared,
      std::span<const Token>(query_tokens.data(), query_tokens.size()),
      trace.query_start,
      trace.operations);
  result.aligned_target = materialize_target_output<Token>(
      prepared,
      std::span<const Token>(target_tokens.data(), target_tokens.size()),
      trace.target_start,
      trace.operations);
  return result;
}

template <bool LocalAlignment>
Score dispatch_score(
    const PreparedAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.kernel_bits) {
    case KernelBits::bits8: {
      const auto& query = std::get<std::vector<std::uint8_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint8_t>>(prepared.target_tokens);
      return score_only<std::uint8_t, std::int8_t, LocalAlignment>(
          std::span<const std::uint8_t>(query.data(), query.size()),
          std::span<const std::uint8_t>(target.data(), target.size()),
          static_cast<std::int8_t>(match_score),
          static_cast<std::int8_t>(mismatch_score),
          static_cast<std::int8_t>(gap_score));
    }
    case KernelBits::bits16: {
      const auto& query = std::get<std::vector<std::uint16_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint16_t>>(prepared.target_tokens);
      return score_only<std::uint16_t, std::int16_t, LocalAlignment>(
          std::span<const std::uint16_t>(query.data(), query.size()),
          std::span<const std::uint16_t>(target.data(), target.size()),
          static_cast<std::int16_t>(match_score),
          static_cast<std::int16_t>(mismatch_score),
          static_cast<std::int16_t>(gap_score));
    }
    case KernelBits::bits32: {
      const auto& query = std::get<std::vector<std::uint32_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint32_t>>(prepared.target_tokens);
      return score_only<std::uint32_t, std::int32_t, LocalAlignment>(
          std::span<const std::uint32_t>(query.data(), query.size()),
          std::span<const std::uint32_t>(target.data(), target.size()),
          static_cast<std::int32_t>(match_score),
          static_cast<std::int32_t>(mismatch_score),
          static_cast<std::int32_t>(gap_score));
    }
    case KernelBits::bits64: {
      const auto& query = std::get<std::vector<std::uint64_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint64_t>>(prepared.target_tokens);
      return score_only<std::uint64_t, std::int64_t, LocalAlignment>(
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
  switch (prepared.kernel_bits) {
    case KernelBits::bits8: {
      const auto& query = std::get<std::vector<std::uint8_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint8_t>>(prepared.target_tokens);
      return build_alignment_result<std::uint8_t>(
          prepared,
          traceback<std::uint8_t, std::int8_t, LocalAlignment>(
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
      return build_alignment_result<std::uint16_t>(
          prepared,
          traceback<std::uint16_t, std::int16_t, LocalAlignment>(
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
      return build_alignment_result<std::uint32_t>(
          prepared,
          traceback<std::uint32_t, std::int32_t, LocalAlignment>(
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
      return build_alignment_result<std::uint64_t>(
          prepared,
          traceback<std::uint64_t, std::int64_t, LocalAlignment>(
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

inline Score dispatch_farrar_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());

  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return farrar_score_only<std::int8_t>(
          query,
          target,
          static_cast<std::int8_t>(match_score),
          static_cast<std::int8_t>(mismatch_score),
          static_cast<std::int8_t>(gap_score));
    case KernelBits::bits16:
      return farrar_score_only<std::int16_t>(
          query,
          target,
          static_cast<std::int16_t>(match_score),
          static_cast<std::int16_t>(mismatch_score),
          static_cast<std::int16_t>(gap_score));
    case KernelBits::bits32:
      return farrar_score_only<std::int32_t>(
          query,
          target,
          static_cast<std::int32_t>(match_score),
          static_cast<std::int32_t>(mismatch_score),
          static_cast<std::int32_t>(gap_score));
    case KernelBits::bits64:
      return farrar_score_only<std::int64_t>(
          query,
          target,
          static_cast<std::int64_t>(match_score),
          static_cast<std::int64_t>(mismatch_score),
          static_cast<std::int64_t>(gap_score));
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported Farrar score width");
  throw nb::python_error();
}

}  // namespace detail

template <BackendKind Kind>
struct Implementation {
  static Score smith_waterman_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return detail::dispatch_score<true>(prepared, match_score, mismatch_score, gap_score);
  }

  static AlignmentResult smith_waterman_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return detail::dispatch_traceback<true>(prepared, match_score, mismatch_score, gap_score);
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
    return detail::dispatch_farrar_score(prepared, match_score, mismatch_score, gap_score);
  }

  static Score needleman_wunsch_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return detail::dispatch_score<false>(prepared, match_score, mismatch_score, gap_score);
  }

  static AlignmentResult needleman_wunsch_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return detail::dispatch_traceback<false>(prepared, match_score, mismatch_score, gap_score);
  }

  static constexpr BackendKind backend_kind = Kind;
};

}  // namespace stride_align::backend_generic
