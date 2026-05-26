#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <nanobind/nanobind.h>

#include "farrar_preprocess.hpp"
#include "preprocess.hpp"
#include "scorer.hpp"
#include "stride_align/alignment.hpp"

namespace stride_align::affine {

namespace nb = nanobind;

namespace detail {

enum class TraceDirection : std::uint8_t {
  stop,
  diagonal,
  up,
  left,
};

enum class TraceState : std::uint8_t {
  h,
  e,
  f,
};

struct TracebackResult {
  Score score = 0;
  std::size_t query_start = 0;
  std::size_t query_end = 0;
  std::size_t target_start = 0;
  std::size_t target_end = 0;
  std::string operations;
};

inline constexpr Score negative_infinity = std::numeric_limits<Score>::min() / 4;

inline Score add_score(Score lhs, Score rhs) noexcept {
  if (lhs <= negative_infinity / 2) {
    return negative_infinity;
  }
  return lhs + rhs;
}

inline Score gap_cost(std::size_t length, Score gap_open_score, Score gap_extend_score) noexcept {
  if (length == 0) {
    return 0;
  }
  return gap_open_score + static_cast<Score>(length - 1U) * gap_extend_score;
}

template <typename Token>
inline Score substitution_score(
    Token query_base,
    Token target_base,
    Score match_score,
    Score mismatch_score) noexcept {
  static_assert(std::is_integral_v<Token>);
  return query_base == target_base ? match_score : mismatch_score;
}

// Scorer-policy core. The match/mismatch and matrix paths share this DP
// recurrence; only `scorer.substitute(q, t)` differs. The compiler folds
// MatchMismatchScorer's two-cell payload into immediates and the per-cell
// dispatch becomes the same cmov / matrix-load it was pre-policy.
template <typename Token, bool LocalAlignment, typename Scorer>
Score score_only(
    std::span<const Token> query,
    std::span<const Token> target,
    Scorer scorer,
    Score gap_open_score,
    Score gap_extend_score) {
  const std::size_t columns = target.size() + 1U;
  std::vector<Score> previous_h(columns, 0);
  std::vector<Score> current_h(columns, 0);
  std::vector<Score> previous_e(columns, negative_infinity);
  std::vector<Score> current_e(columns, negative_infinity);

  if constexpr (!LocalAlignment) {
    previous_h[0] = 0;
    for (std::size_t column = 1; column < columns; ++column) {
      previous_h[column] = gap_cost(column, gap_open_score, gap_extend_score);
    }
  }

  Score best_score = 0;
  if constexpr (!LocalAlignment) {
    best_score = previous_h.back();
  }

  for (std::size_t row = 1; row <= query.size(); ++row) {
    if constexpr (LocalAlignment) {
      current_h[0] = 0;
      current_e[0] = negative_infinity;
    } else {
      current_h[0] = gap_cost(row, gap_open_score, gap_extend_score);
      current_e[0] = current_h[0];
    }

    Score f_score = negative_infinity;

    for (std::size_t column = 1; column <= target.size(); ++column) {
      const Score diagonal = add_score(
          previous_h[column - 1U],
          static_cast<Score>(scorer.substitute(query[row - 1U], target[column - 1U])));
      const Score e_score = std::max(
          add_score(previous_h[column], gap_open_score),
          add_score(previous_e[column], gap_extend_score));
      f_score = std::max(
          add_score(current_h[column - 1U], gap_open_score),
          add_score(f_score, gap_extend_score));

      Score cell = std::max({diagonal, e_score, f_score});
      if constexpr (LocalAlignment) {
        cell = std::max<Score>(0, cell);
        best_score = std::max(best_score, cell);
      }

      current_h[column] = cell;
      current_e[column] = e_score;
    }

    std::swap(previous_h, current_h);
    std::swap(previous_e, current_e);
  }

  if constexpr (LocalAlignment) {
    return best_score;
  }
  return previous_h.back();
}

// Back-compat wrapper: existing match/mismatch callers compile unchanged.
template <typename Token, bool LocalAlignment>
inline Score score_only(
    std::span<const Token> query,
    std::span<const Token> target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  return score_only<Token, LocalAlignment>(
      query,
      target,
      ::stride_align::scorer::MatchMismatchScorer<Score>{match_score, mismatch_score},
      gap_open_score,
      gap_extend_score);
}

// Matrix-mode scalar affine score dispatch. Inputs are byte indices in
// [0, stride) plus a row-major (stride × stride) int8 matrix. Uses Score
// (int64) cells unconditionally — the scalar fallback prioritises
// correctness over width-selection complexity.
template <bool LocalAlignment>
inline Score score_only_matrix(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    const std::int8_t* matrix,
    std::size_t stride,
    Score gap_open_score,
    Score gap_extend_score) {
  ::stride_align::scorer::MatrixScorer<Score> scorer{matrix, stride};
  return score_only<std::uint8_t, LocalAlignment>(
      query, target, scorer, gap_open_score, gap_extend_score);
}

template <typename Token, bool LocalAlignment>
TracebackResult traceback(
    std::span<const Token> query,
    std::span<const Token> target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  const std::size_t rows = query.size() + 1U;
  const std::size_t columns = target.size() + 1U;
  const auto index = [columns](std::size_t row, std::size_t column) {
    return row * columns + column;
  };

  std::vector<Score> h(rows * columns, LocalAlignment ? 0 : negative_infinity);
  std::vector<Score> e(rows * columns, negative_infinity);
  std::vector<Score> f(rows * columns, negative_infinity);

  h[index(0, 0)] = 0;
  if constexpr (!LocalAlignment) {
    for (std::size_t row = 1; row < rows; ++row) {
      const Score score = gap_cost(row, gap_open_score, gap_extend_score);
      h[index(row, 0)] = score;
      e[index(row, 0)] = score;
    }
    for (std::size_t column = 1; column < columns; ++column) {
      const Score score = gap_cost(column, gap_open_score, gap_extend_score);
      h[index(0, column)] = score;
      f[index(0, column)] = score;
    }
  }

  Score best_score = 0;
  std::size_t best_row = 0;
  std::size_t best_column = 0;

  for (std::size_t row = 1; row < rows; ++row) {
    for (std::size_t column = 1; column < columns; ++column) {
      const auto cell_index = index(row, column);
      e[cell_index] = std::max(
          add_score(h[index(row - 1U, column)], gap_open_score),
          add_score(e[index(row - 1U, column)], gap_extend_score));
      f[cell_index] = std::max(
          add_score(h[index(row, column - 1U)], gap_open_score),
          add_score(f[index(row, column - 1U)], gap_extend_score));

      const Score diagonal = add_score(
          h[index(row - 1U, column - 1U)],
          substitution_score<Token>(
              query[row - 1U],
              target[column - 1U],
              match_score,
              mismatch_score));
      Score cell = std::max({diagonal, e[cell_index], f[cell_index]});
      if constexpr (LocalAlignment) {
        cell = std::max<Score>(0, cell);
      }
      h[cell_index] = cell;

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
    best_score = h[index(best_row, best_column)];
  }

  TracebackResult result;
  result.score = best_score;
  result.query_end = best_row;
  result.target_end = best_column;

  std::size_t row = best_row;
  std::size_t column = best_column;
  TraceState state = TraceState::h;

  while (row > 0 || column > 0) {
    const auto cell_index = index(row, column);
    if constexpr (LocalAlignment) {
      if (state == TraceState::h && h[cell_index] <= 0) {
        break;
      }
    }

    if (state == TraceState::h) {
      if (row > 0 && column > 0) {
        const Score diagonal = add_score(
            h[index(row - 1U, column - 1U)],
            substitution_score<Token>(
                query[row - 1U],
                target[column - 1U],
                match_score,
                mismatch_score));
        if (h[cell_index] == diagonal) {
          result.operations.push_back(query[row - 1U] == target[column - 1U] ? '=' : 'X');
          --row;
          --column;
          continue;
        }
      }

      if (row > 0 && h[cell_index] == e[cell_index]) {
        state = TraceState::e;
        continue;
      }
      if (column > 0 && h[cell_index] == f[cell_index]) {
        state = TraceState::f;
        continue;
      }

      break;
    }

    if (state == TraceState::e) {
      result.operations.push_back('D');
      const bool continues_gap =
          row > 1U && e[cell_index] == add_score(e[index(row - 1U, column)], gap_extend_score);
      --row;
      state = continues_gap ? TraceState::e : TraceState::h;
      continue;
    }

    result.operations.push_back('I');
    const bool continues_gap =
        column > 1U && f[cell_index] == add_score(f[index(row, column - 1U)], gap_extend_score);
    --column;
    state = continues_gap ? TraceState::f : TraceState::h;
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
    Score gap_open_score,
    Score gap_extend_score) {
  switch (prepared.kernel_bits) {
    case KernelBits::bits8: {
      const auto& query = std::get<std::vector<std::uint8_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint8_t>>(prepared.target_tokens);
      return score_only<std::uint8_t, LocalAlignment>(
          std::span<const std::uint8_t>(query.data(), query.size()),
          std::span<const std::uint8_t>(target.data(), target.size()),
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    }
    case KernelBits::bits16: {
      const auto& query = std::get<std::vector<std::uint16_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint16_t>>(prepared.target_tokens);
      return score_only<std::uint16_t, LocalAlignment>(
          std::span<const std::uint16_t>(query.data(), query.size()),
          std::span<const std::uint16_t>(target.data(), target.size()),
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    }
    case KernelBits::bits32: {
      const auto& query = std::get<std::vector<std::uint32_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint32_t>>(prepared.target_tokens);
      return score_only<std::uint32_t, LocalAlignment>(
          std::span<const std::uint32_t>(query.data(), query.size()),
          std::span<const std::uint32_t>(target.data(), target.size()),
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    }
    case KernelBits::bits64: {
      const auto& query = std::get<std::vector<std::uint64_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint64_t>>(prepared.target_tokens);
      return score_only<std::uint64_t, LocalAlignment>(
          std::span<const std::uint64_t>(query.data(), query.size()),
          std::span<const std::uint64_t>(target.data(), target.size()),
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
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
    Score gap_open_score,
    Score gap_extend_score) {
  switch (prepared.kernel_bits) {
    case KernelBits::bits8: {
      const auto& query = std::get<std::vector<std::uint8_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint8_t>>(prepared.target_tokens);
      return build_alignment_result<std::uint8_t>(
          prepared,
          traceback<std::uint8_t, LocalAlignment>(
              std::span<const std::uint8_t>(query.data(), query.size()),
              std::span<const std::uint8_t>(target.data(), target.size()),
              match_score,
              mismatch_score,
              gap_open_score,
              gap_extend_score),
          query,
          target);
    }
    case KernelBits::bits16: {
      const auto& query = std::get<std::vector<std::uint16_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint16_t>>(prepared.target_tokens);
      return build_alignment_result<std::uint16_t>(
          prepared,
          traceback<std::uint16_t, LocalAlignment>(
              std::span<const std::uint16_t>(query.data(), query.size()),
              std::span<const std::uint16_t>(target.data(), target.size()),
              match_score,
              mismatch_score,
              gap_open_score,
              gap_extend_score),
          query,
          target);
    }
    case KernelBits::bits32: {
      const auto& query = std::get<std::vector<std::uint32_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint32_t>>(prepared.target_tokens);
      return build_alignment_result<std::uint32_t>(
          prepared,
          traceback<std::uint32_t, LocalAlignment>(
              std::span<const std::uint32_t>(query.data(), query.size()),
              std::span<const std::uint32_t>(target.data(), target.size()),
              match_score,
              mismatch_score,
              gap_open_score,
              gap_extend_score),
          query,
          target);
    }
    case KernelBits::bits64: {
      const auto& query = std::get<std::vector<std::uint64_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint64_t>>(prepared.target_tokens);
      return build_alignment_result<std::uint64_t>(
          prepared,
          traceback<std::uint64_t, LocalAlignment>(
              std::span<const std::uint64_t>(query.data(), query.size()),
              std::span<const std::uint64_t>(target.data(), target.size()),
              match_score,
              mismatch_score,
              gap_open_score,
              gap_extend_score),
          query,
          target);
    }
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported kernel width");
  throw nb::python_error();
}

}  // namespace detail

inline Score smith_waterman_score(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  const auto prepared = prepare_alignment(
      query,
      target,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score,
      width);
  return detail::dispatch_score<true>(
      prepared,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score);
}

inline AlignmentResult smith_waterman_path(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  const auto prepared = prepare_alignment(
      query,
      target,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score,
      width);
  return detail::dispatch_traceback<true>(
      prepared,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score);
}

inline Score smith_waterman_farrar_score(
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
  return detail::score_only<std::uint8_t, true>(
      std::span<const std::uint8_t>(prepared.query_tokens.data(), prepared.query_tokens.size()),
      std::span<const std::uint8_t>(prepared.target_tokens.data(), prepared.target_tokens.size()),
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score);
}

inline Score needleman_wunsch_score(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  const auto prepared = prepare_alignment(
      query,
      target,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score,
      width);
  return detail::dispatch_score<false>(
      prepared,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score);
}

inline AlignmentResult needleman_wunsch_path(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  const auto prepared = prepare_alignment(
      query,
      target,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score,
      width);
  return detail::dispatch_traceback<false>(
      prepared,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score);
}

}  // namespace stride_align::affine
