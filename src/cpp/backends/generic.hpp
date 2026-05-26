#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "cpu.hpp"
#include "farrar_preprocess.hpp"
#include "preprocess.hpp"
#include "scorer.hpp"
#include "backends/score_fast_paths.hpp"
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

// Scorer-policy core for `score_only`. Hot loop dispatches the
// per-cell substitution through `scorer.substitute(q, t)`, which
// inlines to either a single cmov (`MatchMismatchScorer`) or a
// single matrix load (`MatrixScorer`). The existing 4-arg signature
// below forwards to this with a `MatchMismatchScorer` value
// constructed at the call site — sizeof(scorer) is two cells, the
// compiler folds it into immediates, and the codegen ends up
// identical to the pre-policy version.
template <typename Token, typename Cell, bool LocalAlignment, typename Scorer>
Score score_only(
    std::span<const Token> query,
    std::span<const Token> target,
    Scorer scorer,
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
          scorer.substitute(query[row - 1], target[column - 1]));
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

// Back-compat: the existing match/mismatch entry point becomes a
// thin wrapper that constructs the MatchMismatchScorer at the call
// site. Existing call sites compile unchanged.
template <typename Token, typename Cell, bool LocalAlignment>
inline Score score_only(
    std::span<const Token> query,
    std::span<const Token> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  return score_only<Token, Cell, LocalAlignment>(
      query, target,
      ::stride_align::scorer::MatchMismatchScorer<Cell>{match_score,
                                                       mismatch_score},
      gap_score);
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

template <typename Token, typename Cell, bool LocalAlignment, typename Scorer>
TracebackResult traceback(
    std::span<const Token> query,
    std::span<const Token> target,
    Scorer scorer,
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
          scorer.substitute(query[row - 1], target[column - 1]));
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

// Back-compat traceback wrapper for match/mismatch callers.
template <typename Token, typename Cell, bool LocalAlignment>
inline TracebackResult traceback(
    std::span<const Token> query,
    std::span<const Token> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  return traceback<Token, Cell, LocalAlignment>(
      query, target,
      ::stride_align::scorer::MatchMismatchScorer<Cell>{match_score,
                                                       mismatch_score},
      gap_score);
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

template <typename Token, typename Cell>
TracebackResult token_independent_global_traceback(
    std::span<const Token> query,
    std::span<const Token> target,
    Cell substitution_score,
    Cell gap_score) {
  TracebackResult result;
  result.query_end = query.size();
  result.target_end = target.size();
  result.score = score_fast_paths::token_independent_global_score(
      query.size(),
      target.size(),
      substitution_score,
      gap_score);

  const std::size_t diagonal_count =
      static_cast<Score>(substitution_score) >= 2 * static_cast<Score>(gap_score)
          ? std::min(query.size(), target.size())
          : 0;
  result.operations.reserve(query.size() + target.size() - diagonal_count);
  for (std::size_t index = 0; index < diagonal_count; ++index) {
    result.operations.push_back(query[index] == target[index] ? '=' : 'X');
  }
  result.operations.append(query.size() - diagonal_count, 'D');
  result.operations.append(target.size() - diagonal_count, 'I');
  return result;
}

template <typename Token, typename Cell>
TracebackResult token_independent_local_traceback(
    std::span<const Token> query,
    std::span<const Token> target,
    Cell substitution_score) {
  TracebackResult result;
  const std::size_t diagonal_count = std::min(query.size(), target.size());
  result.score = static_cast<Score>(diagonal_count) * static_cast<Score>(substitution_score);
  result.query_end = diagonal_count;
  result.target_end = diagonal_count;
  result.operations.reserve(diagonal_count);
  for (std::size_t index = 0; index < diagonal_count; ++index) {
    result.operations.push_back(query[index] == target[index] ? '=' : 'X');
  }
  return result;
}

template <typename Token, typename Cell, bool LocalAlignment>
std::optional<TracebackResult> fast_traceback(
    std::span<const Token> query,
    std::span<const Token> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  if (query.empty() || target.empty()) {
    TracebackResult result;
    if constexpr (!LocalAlignment) {
      result.score =
          static_cast<Score>(query.size() + target.size()) * static_cast<Score>(gap_score);
      result.query_end = query.size();
      result.target_end = target.size();
      result.operations.reserve(query.size() + target.size());
      result.operations.append(query.size(), 'D');
      result.operations.append(target.size(), 'I');
    }
    return result;
  }

  if constexpr (LocalAlignment) {
    if (match_score <= 0 && mismatch_score <= 0 && gap_score <= 0) {
      return TracebackResult{};
    }
  }

  if (match_score == mismatch_score) {
    if constexpr (LocalAlignment) {
      if (match_score > 0 && gap_score <= 0) {
        return token_independent_local_traceback(query, target, match_score);
      }
    } else {
      return token_independent_global_traceback(query, target, match_score, gap_score);
    }
  }

  if ((!LocalAlignment || match_score > 0) &&
      score_fast_paths::identity_fast_path_is_safe(match_score, mismatch_score, gap_score) &&
      score_fast_paths::tokens_equal(query, target)) {
    TracebackResult result;
    result.score = static_cast<Score>(query.size()) * static_cast<Score>(match_score);
    result.query_end = query.size();
    result.target_end = target.size();
    result.operations.assign(query.size(), '=');
    return result;
  }

  if constexpr (LocalAlignment) {
    if (score_fast_paths::local_zero_fast_path_is_safe(
            match_score,
            mismatch_score,
            gap_score) &&
        score_fast_paths::small_symbols_are_disjoint(query, target)) {
      return TracebackResult{};
    }
  }

  return std::nullopt;
}

template <bool LocalAlignment>
std::optional<Score> dispatch_fast_score(
    const PreparedAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.kernel_bits) {
    case KernelBits::bits8: {
      const auto& query = std::get<std::vector<std::uint8_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint8_t>>(prepared.target_tokens);
      return score_fast_paths::fast_score_only<std::uint8_t, std::int8_t, LocalAlignment>(
          std::span<const std::uint8_t>(query.data(), query.size()),
          std::span<const std::uint8_t>(target.data(), target.size()),
          static_cast<std::int8_t>(match_score),
          static_cast<std::int8_t>(mismatch_score),
          static_cast<std::int8_t>(gap_score));
    }
    case KernelBits::bits16: {
      const auto& query = std::get<std::vector<std::uint16_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint16_t>>(prepared.target_tokens);
      return score_fast_paths::fast_score_only<std::uint16_t, std::int16_t, LocalAlignment>(
          std::span<const std::uint16_t>(query.data(), query.size()),
          std::span<const std::uint16_t>(target.data(), target.size()),
          static_cast<std::int16_t>(match_score),
          static_cast<std::int16_t>(mismatch_score),
          static_cast<std::int16_t>(gap_score));
    }
    case KernelBits::bits32: {
      const auto& query = std::get<std::vector<std::uint32_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint32_t>>(prepared.target_tokens);
      return score_fast_paths::fast_score_only<std::uint32_t, std::int32_t, LocalAlignment>(
          std::span<const std::uint32_t>(query.data(), query.size()),
          std::span<const std::uint32_t>(target.data(), target.size()),
          static_cast<std::int32_t>(match_score),
          static_cast<std::int32_t>(mismatch_score),
          static_cast<std::int32_t>(gap_score));
    }
    case KernelBits::bits64: {
      const auto& query = std::get<std::vector<std::uint64_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint64_t>>(prepared.target_tokens);
      return score_fast_paths::fast_score_only<std::uint64_t, std::int64_t, LocalAlignment>(
          std::span<const std::uint64_t>(query.data(), query.size()),
          std::span<const std::uint64_t>(target.data(), target.size()),
          static_cast<std::int64_t>(match_score),
          static_cast<std::int64_t>(mismatch_score),
          static_cast<std::int64_t>(gap_score));
    }
  }

  return std::nullopt;
}

template <bool LocalAlignment>
std::optional<AlignmentResult> dispatch_fast_traceback(
    const PreparedAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.kernel_bits) {
    case KernelBits::bits8: {
      const auto& query = std::get<std::vector<std::uint8_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint8_t>>(prepared.target_tokens);
      const auto trace = fast_traceback<std::uint8_t, std::int8_t, LocalAlignment>(
          std::span<const std::uint8_t>(query.data(), query.size()),
          std::span<const std::uint8_t>(target.data(), target.size()),
          static_cast<std::int8_t>(match_score),
          static_cast<std::int8_t>(mismatch_score),
          static_cast<std::int8_t>(gap_score));
      if (trace.has_value()) {
        return build_alignment_result<std::uint8_t>(prepared, *trace, query, target);
      }
      return std::nullopt;
    }
    case KernelBits::bits16: {
      const auto& query = std::get<std::vector<std::uint16_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint16_t>>(prepared.target_tokens);
      const auto trace = fast_traceback<std::uint16_t, std::int16_t, LocalAlignment>(
          std::span<const std::uint16_t>(query.data(), query.size()),
          std::span<const std::uint16_t>(target.data(), target.size()),
          static_cast<std::int16_t>(match_score),
          static_cast<std::int16_t>(mismatch_score),
          static_cast<std::int16_t>(gap_score));
      if (trace.has_value()) {
        return build_alignment_result<std::uint16_t>(prepared, *trace, query, target);
      }
      return std::nullopt;
    }
    case KernelBits::bits32: {
      const auto& query = std::get<std::vector<std::uint32_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint32_t>>(prepared.target_tokens);
      const auto trace = fast_traceback<std::uint32_t, std::int32_t, LocalAlignment>(
          std::span<const std::uint32_t>(query.data(), query.size()),
          std::span<const std::uint32_t>(target.data(), target.size()),
          static_cast<std::int32_t>(match_score),
          static_cast<std::int32_t>(mismatch_score),
          static_cast<std::int32_t>(gap_score));
      if (trace.has_value()) {
        return build_alignment_result<std::uint32_t>(prepared, *trace, query, target);
      }
      return std::nullopt;
    }
    case KernelBits::bits64: {
      const auto& query = std::get<std::vector<std::uint64_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint64_t>>(prepared.target_tokens);
      const auto trace = fast_traceback<std::uint64_t, std::int64_t, LocalAlignment>(
          std::span<const std::uint64_t>(query.data(), query.size()),
          std::span<const std::uint64_t>(target.data(), target.size()),
          static_cast<std::int64_t>(match_score),
          static_cast<std::int64_t>(mismatch_score),
          static_cast<std::int64_t>(gap_score));
      if (trace.has_value()) {
        return build_alignment_result<std::uint64_t>(prepared, *trace, query, target);
      }
      return std::nullopt;
    }
  }

  return std::nullopt;
}

template <bool LocalAlignment>
Score dispatch_score(
    const PreparedAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  if (const auto fast = dispatch_fast_score<LocalAlignment>(
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
  if (const auto fast = dispatch_fast_traceback<LocalAlignment>(
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

  if (const auto fast = score_fast_paths::fast_score_only<std::uint8_t, std::int64_t, true>(
          query,
          target,
          static_cast<std::int64_t>(match_score),
          static_cast<std::int64_t>(mismatch_score),
          static_cast<std::int64_t>(gap_score));
      fast.has_value()) {
    return *fast;
  }

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

struct PreparedFarrarScore {
  PreparedFarrarAlignment prepared;
  Score match_score = 2;
  Score mismatch_score = -1;
  Score gap_score = -1;
};

// Matrix-mode score dispatch. Caller passes byte sequences whose
// values are already alphabet indices in [0, stride) plus a row-
// major (stride × stride) int8 matrix. No fast-path shortcuts apply
// — those rely on the match/mismatch summary that the matrix path
// doesn't share — so we go straight to the Scorer-policy core.
//
// Cell width is selected from the score bound: NW alignments span
// O(query_len + target_len) cells so the worst-case score grows
// linearly. Hardcoding int8 here is wrong for any non-trivial
// alignment.
inline std::int32_t matrix_max_abs(const std::int8_t* matrix, std::size_t stride) {
  std::int32_t max_abs = 0;
  const std::size_t cells = stride * stride;
  for (std::size_t i = 0; i < cells; ++i) {
    const std::int32_t value = matrix[i];
    const std::int32_t magnitude = value < 0 ? -value : value;
    if (magnitude > max_abs) {
      max_abs = magnitude;
    }
  }
  return max_abs;
}

template <bool LocalAlignment>
inline Score dispatch_matrix_score_u8(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    const std::int8_t* matrix,
    std::size_t stride,
    Score gap_score) {
  const std::int32_t gap_magnitude = gap_score < 0 ? -gap_score : gap_score;
  const std::int32_t max_abs =
      std::max(matrix_max_abs(matrix, stride), gap_magnitude);
  const std::uint64_t bound =
      static_cast<std::uint64_t>(query.size() + target.size()) *
      static_cast<std::uint64_t>(max_abs);

  const auto run = [&](auto cell_tag) -> Score {
    using Cell = typename decltype(cell_tag)::type;
    ::stride_align::scorer::MatrixScorer<Cell> scorer{matrix, stride};
    return score_only<std::uint8_t, Cell, LocalAlignment>(
        query, target, scorer, static_cast<Cell>(gap_score));
  };

  if (bound <= static_cast<std::uint64_t>(std::numeric_limits<std::int8_t>::max())) {
    return run(std::type_identity<std::int8_t>{});
  }
  if (bound <= static_cast<std::uint64_t>(std::numeric_limits<std::int16_t>::max())) {
    return run(std::type_identity<std::int16_t>{});
  }
  if (bound <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
    return run(std::type_identity<std::int32_t>{});
  }
  return run(std::type_identity<std::int64_t>{});
}

template <bool LocalAlignment>
inline AlignmentResult dispatch_matrix_traceback_u8(
    const PreparedAlignment& prepared,
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    const std::int8_t* matrix,
    std::size_t stride,
    Score gap_score) {
  // Traceback materialises the score grid in memory; pick the widest
  // cell type unconditionally to avoid yet another bound calculation
  // and keep parity with the score-only path's correctness in the
  // worst case.
  ::stride_align::scorer::MatrixScorer<std::int64_t> scorer{matrix, stride};
  auto trace =
      traceback<std::uint8_t, std::int64_t, LocalAlignment>(
          query, target, scorer, static_cast<std::int64_t>(gap_score));
  const auto& q_vec = std::get<std::vector<std::uint8_t>>(prepared.query_tokens);
  const auto& t_vec = std::get<std::vector<std::uint8_t>>(prepared.target_tokens);
  return build_alignment_result<std::uint8_t>(prepared, trace, q_vec, t_vec);
}

}  // namespace detail

template <BackendKind Kind>
struct Implementation {
  using PreparedSmithWatermanFarrarScore = detail::PreparedFarrarScore;

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

  static PreparedSmithWatermanFarrarScore prepare_smith_waterman_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return PreparedSmithWatermanFarrarScore{
        prepare_farrar_alignment(query, target, match_score, mismatch_score, gap_score, width),
        match_score,
        mismatch_score,
        gap_score};
  }

  static Score smith_waterman_farrar_score_prepared(
      const PreparedSmithWatermanFarrarScore& prepared) {
    return detail::dispatch_farrar_score(
        prepared.prepared,
        prepared.match_score,
        prepared.mismatch_score,
        prepared.gap_score);
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

  // --- Matrix-mode alignment (Phase 1: score-only, 8-bit cells) ----
  //
  // The caller is responsible for pre-mapping query / target tokens
  // to alphabet indices in [0, stride). The matrix buffer is
  // row-major (stride × stride) `int8_t`. These methods bypass the
  // PreparedAlignment / fast-path infrastructure entirely — fast
  // paths assume a match/mismatch summary that doesn't generalize
  // to arbitrary matrices.

  static Score smith_waterman_score_matrix(
      nb::handle query_indices,
      nb::handle target_indices,
      nb::handle matrix_buffer,
      std::size_t stride,
      Score gap_score) {
    return matrix_score_dispatch<true>(
        query_indices, target_indices, matrix_buffer, stride, gap_score);
  }

  static Score needleman_wunsch_score_matrix(
      nb::handle query_indices,
      nb::handle target_indices,
      nb::handle matrix_buffer,
      std::size_t stride,
      Score gap_score) {
    return matrix_score_dispatch<false>(
        query_indices, target_indices, matrix_buffer, stride, gap_score);
  }

  // Scalar batch matrix mode: a fallback for backends without a
  // dedicated SIMD batch kernel. Each pair runs the same scalar
  // dispatch the singular path uses — no profile reuse, since the
  // scalar kernel doesn't carry one.
  static std::vector<Score> smith_waterman_scores_matrix(
      nb::handle query_indices,
      nb::handle targets,
      nb::handle matrix_buffer,
      std::size_t stride,
      Score gap_score) {
    return matrix_scores_dispatch<true>(
        query_indices, targets, matrix_buffer, stride, gap_score);
  }

  static std::vector<Score> needleman_wunsch_scores_matrix(
      nb::handle query_indices,
      nb::handle targets,
      nb::handle matrix_buffer,
      std::size_t stride,
      Score gap_score) {
    return matrix_scores_dispatch<false>(
        query_indices, targets, matrix_buffer, stride, gap_score);
  }

  static constexpr BackendKind backend_kind = Kind;

 private:
  template <bool LocalAlignment>
  static Score matrix_score_dispatch(
      nb::handle query_indices,
      nb::handle target_indices,
      nb::handle matrix_buffer,
      std::size_t stride,
      Score gap_score) {
    namespace bv = ::stride_align::byte_view;
    const auto q_kind = bv::classify(query_indices.ptr());
    const auto t_kind = bv::classify(target_indices.ptr());
    const auto m_kind = bv::classify(matrix_buffer.ptr());
    if (q_kind == bv::ByteCompatKind::None ||
        t_kind == bv::ByteCompatKind::None ||
        m_kind == bv::ByteCompatKind::None) {
      PyErr_SetString(
          PyExc_TypeError,
          "matrix-mode alignment expects byte-compatible inputs "
          "(bytes or 1-byte unicode); use SubstitutionMatrix.encode "
          "to map sequences to alphabet indices.");
      throw nb::python_error();
    }
    const std::uint8_t* q_ptr = nullptr;
    std::size_t q_len = 0;
    const std::uint8_t* t_ptr = nullptr;
    std::size_t t_len = 0;
    const std::uint8_t* m_ptr = nullptr;
    std::size_t m_len = 0;
    bv::view(query_indices.ptr(), q_kind, q_ptr, q_len);
    bv::view(target_indices.ptr(), t_kind, t_ptr, t_len);
    bv::view(matrix_buffer.ptr(), m_kind, m_ptr, m_len);
    if (m_len != stride * stride) {
      PyErr_Format(
          PyExc_ValueError,
          "matrix buffer size %zu does not match stride * stride = %zu",
          m_len, stride * stride);
      throw nb::python_error();
    }
    return detail::dispatch_matrix_score_u8<LocalAlignment>(
        std::span<const std::uint8_t>(q_ptr, q_len),
        std::span<const std::uint8_t>(t_ptr, t_len),
        reinterpret_cast<const std::int8_t*>(m_ptr),
        stride,
        gap_score);
  }

  template <bool LocalAlignment>
  static std::vector<Score> matrix_scores_dispatch(
      nb::handle query_indices,
      nb::handle targets,
      nb::handle matrix_buffer,
      std::size_t stride,
      Score gap_score) {
    PyObject* fast_targets = PySequence_Fast(
        targets.ptr(), "targets must be a sequence of target sequences");
    if (fast_targets == nullptr) {
      throw nb::python_error();
    }
    nb::object owner = nb::steal<nb::object>(fast_targets);
    const auto target_count = static_cast<std::size_t>(
        PySequence_Fast_GET_SIZE(fast_targets));
    PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

    std::vector<Score> scores;
    scores.reserve(target_count);
    for (std::size_t i = 0; i < target_count; ++i) {
      nb::handle item(items[i]);
      scores.push_back(matrix_score_dispatch<LocalAlignment>(
          query_indices, item, matrix_buffer, stride, gap_score));
    }
    return scores;
  }
};

}  // namespace stride_align::backend_generic
