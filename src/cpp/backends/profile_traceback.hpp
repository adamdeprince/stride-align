#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include <nanobind/nanobind.h>

#include "affine.hpp"
#include "farrar_preprocess.hpp"
#include "preprocess.hpp"
#include "stride_align/alignment.hpp"

namespace stride_align::profile_traceback {

namespace nb = nanobind;

namespace detail {

enum class Direction : std::uint8_t {
  stop,
  diagonal,
  up,
  left,
};

inline constexpr std::uint8_t direction_mask = 0x03U;
inline constexpr std::uint8_t up_continue_bit = 0x04U;
inline constexpr std::uint8_t left_continue_bit = 0x08U;

inline std::uint8_t pack_trace(
    Direction direction,
    bool up_continues,
    bool left_continues) noexcept {
  return static_cast<std::uint8_t>(
      static_cast<std::uint8_t>(direction) |
      (up_continues ? up_continue_bit : 0U) |
      (left_continues ? left_continue_bit : 0U));
}

inline Direction trace_direction(std::uint8_t trace) noexcept {
  return static_cast<Direction>(trace & direction_mask);
}

inline bool trace_up_continues(std::uint8_t trace) noexcept {
  return (trace & up_continue_bit) != 0;
}

inline bool trace_left_continues(std::uint8_t trace) noexcept {
  return (trace & left_continue_bit) != 0;
}

enum class AffineState : std::uint8_t {
  h,
  up,
  left,
};

template <typename TraceCell>
class TraceTable {
 public:
  // Interior cells are overwritten by DP; callers initialize only readable boundaries.
  explicit TraceTable(std::size_t cell_count)
      : data_(std::make_unique_for_overwrite<TraceCell[]>(cell_count)) {}

  TraceCell& operator[](std::size_t index) noexcept {
    return data_[index];
  }

  const TraceCell& operator[](std::size_t index) const noexcept {
    return data_[index];
  }

 private:
  std::unique_ptr<TraceCell[]> data_;
};

template <typename Cell>
inline constexpr Cell negative_infinity_v = std::numeric_limits<Cell>::lowest();

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

template <typename Token, typename Cell>
Cell substitution_score(
    Token query_base,
    Token target_base,
    Cell match_score,
    Cell mismatch_score) noexcept {
  static_assert(std::is_integral_v<Token>);
  static_assert(std::is_integral_v<Cell>);
  return query_base == target_base ? match_score : mismatch_score;
}

inline std::size_t matrix_index(
    std::size_t row,
    std::size_t column,
    std::size_t columns) noexcept {
  return row * columns + column;
}

struct TokenRange {
  const std::uint8_t* data = nullptr;
  std::size_t begin = 0;
  std::size_t end = 0;
  bool reversed = false;

  std::size_t size() const noexcept {
    return end - begin;
  }

  bool empty() const noexcept {
    return size() == 0;
  }

  std::uint8_t operator[](std::size_t index) const noexcept {
    return reversed ? data[end - 1U - index] : data[begin + index];
  }

  TokenRange subrange(std::size_t offset, std::size_t count) const noexcept {
    if (!reversed) {
      return {data, begin + offset, begin + offset + count, false};
    }
    return {data, end - offset - count, end - offset, true};
  }

  TokenRange reversed_view() const noexcept {
    return {data, begin, end, !reversed};
  }
};

inline TokenRange make_range(std::span<const std::uint8_t> values) noexcept {
  return {values.data(), 0U, values.size(), false};
}

template <typename Cell>
void append_repeated_operation(std::string& operations, char operation, std::size_t count) {
  operations.append(count, operation);
}

template <typename Cell>
void linear_global_last_row(
    const TokenRange& query,
    const TokenRange& target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score,
    std::vector<Cell>& scores) {
  const std::size_t columns = target.size() + 1U;
  scores.resize(columns);
  std::vector<Cell> current(columns, Cell{0});

  for (std::size_t column = 0; column < columns; ++column) {
    scores[column] =
        static_cast<Cell>(static_cast<Score>(column) * static_cast<Score>(gap_score));
  }

  for (std::size_t row = 1; row <= query.size(); ++row) {
    current[0] =
        static_cast<Cell>(static_cast<Score>(row) * static_cast<Score>(gap_score));
    for (std::size_t column = 1; column < columns; ++column) {
      const Cell diagonal = static_cast<Cell>(
          scores[column - 1U] +
          substitution_score<std::uint8_t, Cell>(
              query[row - 1U],
              target[column - 1U],
              match_score,
              mismatch_score));
      const Cell up = static_cast<Cell>(scores[column] + gap_score);
      const Cell left = static_cast<Cell>(current[column - 1U] + gap_score);
      current[column] = std::max({diagonal, up, left});
    }
    std::swap(scores, current);
  }
}

template <typename Cell>
std::string linear_small_global_operations(
    const TokenRange& query,
    const TokenRange& target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  const std::size_t rows = query.size() + 1U;
  const std::size_t columns = target.size() + 1U;
  std::vector<Cell> previous(columns, Cell{0});
  std::vector<Cell> current(columns, Cell{0});
  std::vector<Direction> directions(rows * columns, Direction::stop);

  for (std::size_t column = 1; column < columns; ++column) {
    previous[column] =
        static_cast<Cell>(static_cast<Score>(column) * static_cast<Score>(gap_score));
    directions[matrix_index(0, column, columns)] = Direction::left;
  }

  for (std::size_t row = 1; row < rows; ++row) {
    current[0] =
        static_cast<Cell>(static_cast<Score>(row) * static_cast<Score>(gap_score));
    directions[matrix_index(row, 0, columns)] = Direction::up;

    for (std::size_t column = 1; column < columns; ++column) {
      const std::size_t cell_index = matrix_index(row, column, columns);
      const Cell diagonal = static_cast<Cell>(
          previous[column - 1U] +
          substitution_score<std::uint8_t, Cell>(
              query[row - 1U],
              target[column - 1U],
              match_score,
              mismatch_score));
      const Cell up = static_cast<Cell>(previous[column] + gap_score);
      const Cell left = static_cast<Cell>(current[column - 1U] + gap_score);

      Cell cell = diagonal;
      Direction direction = Direction::diagonal;
      if (up > cell) {
        cell = up;
        direction = Direction::up;
      }
      if (left > cell) {
        cell = left;
        direction = Direction::left;
      }

      current[column] = cell;
      directions[cell_index] = direction;
    }
    std::swap(previous, current);
  }

  std::string reversed_operations;
  reversed_operations.reserve(query.size() + target.size());
  std::size_t row = query.size();
  std::size_t column = target.size();
  while (row > 0 || column > 0) {
    const Direction direction = directions[matrix_index(row, column, columns)];
    if (direction == Direction::diagonal) {
      reversed_operations.push_back(query[row - 1U] == target[column - 1U] ? 'M' : 'X');
      --row;
      --column;
      continue;
    }
    if (direction == Direction::up) {
      reversed_operations.push_back('D');
      --row;
      continue;
    }
    reversed_operations.push_back('I');
    --column;
  }
  std::reverse(reversed_operations.begin(), reversed_operations.end());
  return reversed_operations;
}

template <typename Cell>
void linear_hirschberg_operations(
    const TokenRange& query,
    const TokenRange& target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score,
    std::string& operations) {
  constexpr std::size_t small_cell_threshold = 4096U;
  if (query.empty()) {
    append_repeated_operation<Cell>(operations, 'I', target.size());
    return;
  }
  if (target.empty()) {
    append_repeated_operation<Cell>(operations, 'D', query.size());
    return;
  }
  if (query.size() * target.size() <= small_cell_threshold) {
    operations += linear_small_global_operations<Cell>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score);
    return;
  }

  const std::size_t midpoint = query.size() / 2U;
  const TokenRange query_left = query.subrange(0U, midpoint);
  const TokenRange query_right = query.subrange(midpoint, query.size() - midpoint);

  std::vector<Cell> forward;
  std::vector<Cell> reverse;
  linear_global_last_row<Cell>(
      query_left,
      target,
      match_score,
      mismatch_score,
      gap_score,
      forward);
  linear_global_last_row<Cell>(
      query_right.reversed_view(),
      target.reversed_view(),
      match_score,
      mismatch_score,
      gap_score,
      reverse);

  std::size_t split = 0;
  Cell best = static_cast<Cell>(forward[0] + reverse[target.size()]);
  for (std::size_t column = 1; column <= target.size(); ++column) {
    const Cell score = static_cast<Cell>(forward[column] + reverse[target.size() - column]);
    if (score > best) {
      best = score;
      split = column;
    }
  }

  linear_hirschberg_operations<Cell>(
      query_left,
      target.subrange(0U, split),
      match_score,
      mismatch_score,
      gap_score,
      operations);
  linear_hirschberg_operations<Cell>(
      query_right,
      target.subrange(split, target.size() - split),
      match_score,
      mismatch_score,
      gap_score,
      operations);
}

template <typename Cell>
std::string linear_global_trace_free_cigar(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  std::string operations;
  operations.reserve(query.size() + target.size());
  const TokenRange query_range = make_range(query);
  const TokenRange target_range = make_range(target);

  linear_hirschberg_operations<Cell>(
      query_range,
      target_range,
      match_score,
      mismatch_score,
      gap_score,
      operations);

  return build_cigar(operations);
}

template <typename Cell>
struct LocalEndpoint {
  Cell score = 0;
  std::size_t row = 0;
  std::size_t column = 0;
};

template <typename Cell>
LocalEndpoint<Cell> linear_local_endpoint(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  const std::size_t columns = target.size() + 1U;
  std::vector<Cell> previous(columns, Cell{0});
  std::vector<Cell> current(columns, Cell{0});
  LocalEndpoint<Cell> best;

  for (std::size_t row = 1; row <= query.size(); ++row) {
    current[0] = 0;
    for (std::size_t column = 1; column < columns; ++column) {
      const Cell diagonal = static_cast<Cell>(
          previous[column - 1U] +
          substitution_score<std::uint8_t, Cell>(
              query[row - 1U],
              target[column - 1U],
              match_score,
              mismatch_score));
      const Cell up = static_cast<Cell>(previous[column] + gap_score);
      const Cell left = static_cast<Cell>(current[column - 1U] + gap_score);
      const Cell cell = std::max({diagonal, up, left, Cell{0}});
      current[column] = cell;
      if (cell > best.score) {
        best = {cell, row, column};
      }
    }
    std::swap(previous, current);
  }

  return best;
}

template <typename Cell>
std::optional<std::pair<std::size_t, std::size_t>> linear_local_anchored_start(
    const TokenRange& reversed_query_prefix,
    const TokenRange& reversed_target_prefix,
    Cell target_score,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  const std::size_t columns = reversed_target_prefix.size() + 1U;
  std::vector<Cell> previous(columns, Cell{0});
  std::vector<Cell> current(columns, Cell{0});

  for (std::size_t column = 1; column < columns; ++column) {
    previous[column] = static_cast<Cell>(
        previous[column - 1U] + gap_score);
  }

  for (std::size_t row = 1; row <= reversed_query_prefix.size(); ++row) {
    current[0] = static_cast<Cell>(
        previous[0] + gap_score);
    for (std::size_t column = 1; column < columns; ++column) {
      const Cell diagonal = static_cast<Cell>(
          previous[column - 1U] +
          substitution_score<std::uint8_t, Cell>(
              reversed_query_prefix[row - 1U],
              reversed_target_prefix[column - 1U],
              match_score,
              mismatch_score));
      const Cell up = static_cast<Cell>(previous[column] + gap_score);
      const Cell left = static_cast<Cell>(current[column - 1U] + gap_score);
      const Cell cell = std::max({diagonal, up, left});
      current[column] = cell;
      if (cell == target_score) {
        return std::pair<std::size_t, std::size_t>{row, column};
      }
    }
    std::swap(previous, current);
  }

  return std::nullopt;
}

template <typename Cell>
std::optional<std::string> linear_local_trace_free_cigar(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  if (query.empty() || target.empty()) {
    return std::string();
  }

  const LocalEndpoint<Cell> endpoint =
      linear_local_endpoint<Cell>(query, target, match_score, mismatch_score, gap_score);
  if (endpoint.score <= 0) {
    return std::string();
  }

  const TokenRange query_range = make_range(query);
  const TokenRange target_range = make_range(target);
  const TokenRange reversed_query_prefix =
      query_range.subrange(0U, endpoint.row).reversed_view();
  const TokenRange reversed_target_prefix =
      target_range.subrange(0U, endpoint.column).reversed_view();
  const auto consumed = linear_local_anchored_start<Cell>(
      reversed_query_prefix,
      reversed_target_prefix,
      endpoint.score,
      match_score,
      mismatch_score,
      gap_score);
  if (!consumed.has_value()) {
    return std::nullopt;
  }

  const std::size_t query_start = endpoint.row - consumed->first;
  const std::size_t target_start = endpoint.column - consumed->second;
  const TokenRange query_window =
      query_range.subrange(query_start, endpoint.row - query_start);
  const TokenRange target_window =
      target_range.subrange(target_start, endpoint.column - target_start);

  std::vector<Cell> check_scores;
  linear_global_last_row<Cell>(
      query_window,
      target_window,
      match_score,
      mismatch_score,
      gap_score,
      check_scores);
  if (check_scores.empty() || check_scores.back() != endpoint.score) {
    return std::nullopt;
  }

  std::string operations;
  operations.reserve(query_window.size() + target_window.size());
  linear_hirschberg_operations<Cell>(
      query_window,
      target_window,
      match_score,
      mismatch_score,
      gap_score,
      operations);
  return build_cigar(operations);
}

template <typename Cell, bool LocalAlignment>
AlignmentPath linear_path_info(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  const std::size_t rows = query.size() + 1U;
  const std::size_t columns = target.size() + 1U;
  std::vector<Cell> previous(columns, Cell{0});
  std::vector<Cell> current(columns, Cell{0});
  TraceTable<Direction> directions(rows * columns);

  directions[matrix_index(0, 0, columns)] = Direction::stop;
  if constexpr (!LocalAlignment) {
    for (std::size_t column = 1; column < columns; ++column) {
      previous[column] =
          static_cast<Cell>(static_cast<Score>(column) * static_cast<Score>(gap_score));
      directions[matrix_index(0, column, columns)] = Direction::left;
    }
  } else {
    for (std::size_t column = 1; column < columns; ++column) {
      directions[matrix_index(0, column, columns)] = Direction::stop;
    }
  }

  [[maybe_unused]] Cell best_score = 0;
  std::size_t best_row = 0;
  std::size_t best_column = 0;
  if constexpr (!LocalAlignment) {
    best_row = query.size();
    best_column = target.size();
  }

  for (std::size_t row = 1; row < rows; ++row) {
    if constexpr (LocalAlignment) {
      current[0] = 0;
      directions[matrix_index(row, 0, columns)] = Direction::stop;
    } else {
      current[0] =
          static_cast<Cell>(static_cast<Score>(row) * static_cast<Score>(gap_score));
      directions[matrix_index(row, 0, columns)] = Direction::up;
    }

    for (std::size_t column = 1; column < columns; ++column) {
      const std::size_t cell_index = matrix_index(row, column, columns);
      const Cell diagonal = static_cast<Cell>(
          previous[column - 1U] +
          substitution_score<std::uint8_t, Cell>(
              query[row - 1U],
              target[column - 1U],
              match_score,
              mismatch_score));
      const Cell up = static_cast<Cell>(previous[column] + gap_score);
      const Cell left = static_cast<Cell>(current[column - 1U] + gap_score);

      Cell cell = diagonal;
      Direction direction = Direction::diagonal;
      if (up > cell) {
        cell = up;
        direction = Direction::up;
      }
      if (left > cell) {
        cell = left;
        direction = Direction::left;
      }
      if constexpr (LocalAlignment) {
        if (cell <= 0) {
          cell = 0;
          direction = Direction::stop;
        }
      }

      current[column] = cell;
      directions[cell_index] = direction;
      if constexpr (LocalAlignment) {
        if (cell > best_score) {
          best_score = cell;
          best_row = row;
          best_column = column;
        }
      }
    }
    std::swap(previous, current);
  }

  if constexpr (!LocalAlignment) {
    best_score = previous.back();
  }

  std::string operations;
  operations.reserve(query.size() + target.size());
  std::size_t row = best_row;
  std::size_t column = best_column;
  while (row > 0 || column > 0) {
    const Direction direction = directions[matrix_index(row, column, columns)];
    if (direction == Direction::stop) {
      break;
    }

    if (direction == Direction::diagonal) {
      operations.push_back(query[row - 1U] == target[column - 1U] ? 'M' : 'X');
      --row;
      --column;
      continue;
    }

    if (direction == Direction::up) {
      operations.push_back('D');
      --row;
      continue;
    }

    operations.push_back('I');
    --column;
  }

  std::reverse(operations.begin(), operations.end());
  return make_alignment_path(
      static_cast<Score>(best_score),
      row,
      best_row,
      column,
      best_column,
      operations);
}

template <typename Cell, bool LocalAlignment>
std::string linear_cigar(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  const std::size_t cell_count = query.size() * target.size();
  if constexpr (!LocalAlignment) {
    constexpr std::size_t linear_space_cigar_cell_threshold = 16U * 1024U * 1024U;
    if (gap_score <= 0 && cell_count >= linear_space_cigar_cell_threshold) {
      return linear_global_trace_free_cigar<Cell>(
          query,
          target,
          match_score,
          mismatch_score,
          gap_score);
    }
  }
  if constexpr (LocalAlignment) {
    (void) cell_count;
  }

  const std::size_t rows = query.size() + 1U;
  const std::size_t columns = target.size() + 1U;
  std::vector<Cell> previous(columns, Cell{0});
  std::vector<Cell> current(columns, Cell{0});
  TraceTable<Direction> directions(rows * columns);

  directions[matrix_index(0, 0, columns)] = Direction::stop;
  if constexpr (!LocalAlignment) {
    for (std::size_t column = 1; column < columns; ++column) {
      previous[column] =
          static_cast<Cell>(static_cast<Score>(column) * static_cast<Score>(gap_score));
      directions[matrix_index(0, column, columns)] = Direction::left;
    }
  } else {
    for (std::size_t column = 1; column < columns; ++column) {
      directions[matrix_index(0, column, columns)] = Direction::stop;
    }
  }

  [[maybe_unused]] Cell best_score = 0;
  std::size_t best_row = 0;
  std::size_t best_column = 0;
  if constexpr (!LocalAlignment) {
    best_row = query.size();
    best_column = target.size();
  }

  for (std::size_t row = 1; row < rows; ++row) {
    if constexpr (LocalAlignment) {
      current[0] = 0;
      directions[matrix_index(row, 0, columns)] = Direction::stop;
    } else {
      current[0] =
          static_cast<Cell>(static_cast<Score>(row) * static_cast<Score>(gap_score));
      directions[matrix_index(row, 0, columns)] = Direction::up;
    }

    for (std::size_t column = 1; column < columns; ++column) {
      const std::size_t cell_index = matrix_index(row, column, columns);
      const Cell diagonal = static_cast<Cell>(
          previous[column - 1U] +
          substitution_score<std::uint8_t, Cell>(
              query[row - 1U],
              target[column - 1U],
              match_score,
              mismatch_score));
      const Cell up = static_cast<Cell>(previous[column] + gap_score);
      const Cell left = static_cast<Cell>(current[column - 1U] + gap_score);

      Cell cell = diagonal;
      Direction direction = Direction::diagonal;
      if (up > cell) {
        cell = up;
        direction = Direction::up;
      }
      if (left > cell) {
        cell = left;
        direction = Direction::left;
      }
      if constexpr (LocalAlignment) {
        if (cell <= 0) {
          cell = 0;
          direction = Direction::stop;
        }
      }

      current[column] = cell;
      directions[cell_index] = direction;
      if constexpr (LocalAlignment) {
        if (cell > best_score) {
          best_score = cell;
          best_row = row;
          best_column = column;
        }
      }
    }
    std::swap(previous, current);
  }

  ReverseCigarBuilder cigar;
  std::size_t row = best_row;
  std::size_t column = best_column;
  while (row > 0 || column > 0) {
    const Direction direction = directions[matrix_index(row, column, columns)];
    if (direction == Direction::stop) {
      break;
    }

    if (direction == Direction::diagonal) {
      cigar.push(query[row - 1U] == target[column - 1U] ? 'M' : 'X');
      --row;
      --column;
      continue;
    }

    if (direction == Direction::up) {
      cigar.push('D');
      --row;
      continue;
    }

    cigar.push('I');
    --column;
  }

  return cigar.str();
}

template <typename Cell, bool LocalAlignment>
Score affine_score(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_open_score,
    Cell gap_extend_score) {
  if constexpr (LocalAlignment) {
    if (query.empty() || target.empty()) {
      return 0;
    }
  }

  const std::size_t columns = target.size() + 1U;
  std::vector<Cell> previous_h(columns, LocalAlignment ? Cell{0} : negative_infinity_v<Cell>);
  std::vector<Cell> current_h(columns, LocalAlignment ? Cell{0} : negative_infinity_v<Cell>);
  std::vector<Cell> previous_up(columns, negative_infinity_v<Cell>);
  std::vector<Cell> current_up(columns, negative_infinity_v<Cell>);

  previous_h[0] = 0;
  if constexpr (!LocalAlignment) {
    for (std::size_t column = 1; column < columns; ++column) {
      previous_h[column] = gap_cost<Cell>(column, gap_open_score, gap_extend_score);
    }
  }

  [[maybe_unused]] Cell best_score = 0;
  if constexpr (!LocalAlignment) {
    best_score = previous_h.back();
  }

  for (std::size_t row = 1; row <= query.size(); ++row) {
    if constexpr (LocalAlignment) {
      current_h[0] = 0;
      current_up[0] = negative_infinity_v<Cell>;
    } else {
      current_h[0] = gap_cost<Cell>(row, gap_open_score, gap_extend_score);
      current_up[0] = current_h[0];
    }

    Cell left_score = negative_infinity_v<Cell>;
    for (std::size_t column = 1; column <= target.size(); ++column) {
      const Cell diagonal = safe_add<Cell>(
          previous_h[column - 1U],
          substitution_score<std::uint8_t, Cell>(
              query[row - 1U],
              target[column - 1U],
              match_score,
              mismatch_score));
      const Cell up_score = std::max(
          safe_add<Cell>(previous_h[column], gap_open_score),
          safe_add<Cell>(previous_up[column], gap_extend_score));
      left_score = std::max(
          safe_add<Cell>(current_h[column - 1U], gap_open_score),
          safe_add<Cell>(left_score, gap_extend_score));

      Cell cell = std::max({diagonal, up_score, left_score});
      if constexpr (LocalAlignment) {
        cell = std::max<Cell>(0, cell);
        best_score = std::max(best_score, cell);
      }

      current_h[column] = cell;
      current_up[column] = up_score;
    }

    std::swap(previous_h, current_h);
    std::swap(previous_up, current_up);
  }

  if constexpr (LocalAlignment) {
    return static_cast<Score>(best_score);
  }
  return static_cast<Score>(previous_h.back());
}

template <typename Cell, bool LocalAlignment>
AlignmentPath affine_path_info(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_open_score,
    Cell gap_extend_score) {
  const std::size_t rows = query.size() + 1U;
  const std::size_t columns = target.size() + 1U;
  const std::size_t cell_count = rows * columns;

  std::vector<Cell> previous_h(columns, LocalAlignment ? Cell{0} : negative_infinity_v<Cell>);
  std::vector<Cell> current_h(columns, LocalAlignment ? Cell{0} : negative_infinity_v<Cell>);
  std::vector<Cell> previous_up(columns, negative_infinity_v<Cell>);
  std::vector<Cell> current_up(columns, negative_infinity_v<Cell>);
  TraceTable<std::uint8_t> trace(cell_count);

  previous_h[0] = 0;
  trace[matrix_index(0, 0, columns)] = pack_trace(Direction::stop, false, false);
  if constexpr (!LocalAlignment) {
    for (std::size_t column = 1; column < columns; ++column) {
      const std::size_t cell_index = matrix_index(0, column, columns);
      previous_h[column] = gap_cost<Cell>(column, gap_open_score, gap_extend_score);
      trace[cell_index] = pack_trace(Direction::left, false, column > 1U);
    }
  } else {
    for (std::size_t column = 1; column < columns; ++column) {
      trace[matrix_index(0, column, columns)] = pack_trace(Direction::stop, false, false);
    }
  }

  [[maybe_unused]] Cell best_score = 0;
  std::size_t best_row = 0;
  std::size_t best_column = 0;

  for (std::size_t row = 1; row < rows; ++row) {
    if constexpr (LocalAlignment) {
      current_h[0] = 0;
      current_up[0] = negative_infinity_v<Cell>;
      trace[matrix_index(row, 0, columns)] = pack_trace(Direction::stop, false, false);
    } else {
      const std::size_t row_index = matrix_index(row, 0, columns);
      current_h[0] = gap_cost<Cell>(row, gap_open_score, gap_extend_score);
      current_up[0] = current_h[0];
      trace[row_index] = pack_trace(Direction::up, row > 1U, false);
    }

    Cell left_score = negative_infinity_v<Cell>;
    for (std::size_t column = 1; column < columns; ++column) {
      const std::size_t cell_index = matrix_index(row, column, columns);
      const Cell up_open = safe_add<Cell>(previous_h[column], gap_open_score);
      const Cell up_extend = safe_add<Cell>(previous_up[column], gap_extend_score);
      const Cell up_score = std::max(up_open, up_extend);
      const bool up_continues_gap = up_extend >= up_open;

      const Cell left_open = safe_add<Cell>(current_h[column - 1U], gap_open_score);
      const Cell left_extend = safe_add<Cell>(left_score, gap_extend_score);
      left_score = std::max(left_open, left_extend);
      const bool left_continues_gap = left_extend >= left_open;

      const Cell diagonal = safe_add<Cell>(
          previous_h[column - 1U],
          substitution_score<std::uint8_t, Cell>(
              query[row - 1U],
              target[column - 1U],
              match_score,
              mismatch_score));

      Cell cell = diagonal;
      Direction source = Direction::diagonal;
      if (up_score > cell) {
        cell = up_score;
        source = Direction::up;
      }
      if (left_score > cell) {
        cell = left_score;
        source = Direction::left;
      }
      if constexpr (LocalAlignment) {
        if (cell <= 0) {
          cell = 0;
          source = Direction::stop;
        }
      }

      current_h[column] = cell;
      current_up[column] = up_score;
      trace[cell_index] = pack_trace(source, up_continues_gap, left_continues_gap);

      if constexpr (LocalAlignment) {
        if (cell > best_score) {
          best_score = cell;
          best_row = row;
          best_column = column;
        }
      }
    }
    std::swap(previous_h, current_h);
    std::swap(previous_up, current_up);
  }

  if constexpr (!LocalAlignment) {
    best_row = query.size();
    best_column = target.size();
    best_score = previous_h.back();
  }

  std::string operations;
  operations.reserve(query.size() + target.size());
  std::size_t row = best_row;
  std::size_t column = best_column;
  AffineState state = AffineState::h;

  while (row > 0 || column > 0) {
    const std::size_t cell_index = matrix_index(row, column, columns);
    const std::uint8_t trace_cell = trace[cell_index];
    if constexpr (LocalAlignment) {
      if (state == AffineState::h && trace_direction(trace_cell) == Direction::stop) {
        break;
      }
    }

    if (state == AffineState::h) {
      if (row == 0 && column > 0) {
        state = AffineState::left;
        continue;
      }
      if (column == 0 && row > 0) {
        state = AffineState::up;
        continue;
      }

      const Direction source = trace_direction(trace_cell);
      if (source == Direction::diagonal) {
        operations.push_back(query[row - 1U] == target[column - 1U] ? 'M' : 'X');
        --row;
        --column;
        continue;
      }
      if (source == Direction::up) {
        state = AffineState::up;
        continue;
      }
      if (source == Direction::left) {
        state = AffineState::left;
        continue;
      }
      break;
    }

    if (state == AffineState::up) {
      operations.push_back('D');
      const bool continues_gap = row > 1U && trace_up_continues(trace_cell);
      --row;
      state = continues_gap ? AffineState::up : AffineState::h;
      continue;
    }

    operations.push_back('I');
    const bool continues_gap = column > 1U && trace_left_continues(trace_cell);
    --column;
    state = continues_gap ? AffineState::left : AffineState::h;
  }

  std::reverse(operations.begin(), operations.end());
  return make_alignment_path(
      static_cast<Score>(best_score),
      row,
      best_row,
      column,
      best_column,
      operations);
}

template <typename Cell, bool LocalAlignment>
std::string affine_cigar(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_open_score,
    Cell gap_extend_score) {
  const std::size_t rows = query.size() + 1U;
  const std::size_t columns = target.size() + 1U;
  const std::size_t cell_count = rows * columns;

  std::vector<Cell> previous_h(columns, LocalAlignment ? Cell{0} : negative_infinity_v<Cell>);
  std::vector<Cell> current_h(columns, LocalAlignment ? Cell{0} : negative_infinity_v<Cell>);
  std::vector<Cell> previous_up(columns, negative_infinity_v<Cell>);
  std::vector<Cell> current_up(columns, negative_infinity_v<Cell>);
  TraceTable<std::uint8_t> trace(cell_count);

  previous_h[0] = 0;
  trace[matrix_index(0, 0, columns)] = pack_trace(Direction::stop, false, false);
  if constexpr (!LocalAlignment) {
    for (std::size_t column = 1; column < columns; ++column) {
      const std::size_t cell_index = matrix_index(0, column, columns);
      previous_h[column] = gap_cost<Cell>(column, gap_open_score, gap_extend_score);
      trace[cell_index] = pack_trace(Direction::left, false, column > 1U);
    }
  } else {
    for (std::size_t column = 1; column < columns; ++column) {
      trace[matrix_index(0, column, columns)] = pack_trace(Direction::stop, false, false);
    }
  }

  [[maybe_unused]] Cell best_score = 0;
  std::size_t best_row = 0;
  std::size_t best_column = 0;

  for (std::size_t row = 1; row < rows; ++row) {
    if constexpr (LocalAlignment) {
      current_h[0] = 0;
      current_up[0] = negative_infinity_v<Cell>;
      trace[matrix_index(row, 0, columns)] = pack_trace(Direction::stop, false, false);
    } else {
      const std::size_t row_index = matrix_index(row, 0, columns);
      current_h[0] = gap_cost<Cell>(row, gap_open_score, gap_extend_score);
      current_up[0] = current_h[0];
      trace[row_index] = pack_trace(Direction::up, row > 1U, false);
    }

    Cell left_score = negative_infinity_v<Cell>;
    for (std::size_t column = 1; column < columns; ++column) {
      const std::size_t cell_index = matrix_index(row, column, columns);
      const Cell up_open = safe_add<Cell>(previous_h[column], gap_open_score);
      const Cell up_extend = safe_add<Cell>(previous_up[column], gap_extend_score);
      const Cell up_score = std::max(up_open, up_extend);
      const bool up_continues_gap = up_extend >= up_open;

      const Cell left_open = safe_add<Cell>(current_h[column - 1U], gap_open_score);
      const Cell left_extend = safe_add<Cell>(left_score, gap_extend_score);
      left_score = std::max(left_open, left_extend);
      const bool left_continues_gap = left_extend >= left_open;

      const Cell diagonal = safe_add<Cell>(
          previous_h[column - 1U],
          substitution_score<std::uint8_t, Cell>(
              query[row - 1U],
              target[column - 1U],
              match_score,
              mismatch_score));

      Cell cell = diagonal;
      Direction source = Direction::diagonal;
      if (up_score > cell) {
        cell = up_score;
        source = Direction::up;
      }
      if (left_score > cell) {
        cell = left_score;
        source = Direction::left;
      }
      if constexpr (LocalAlignment) {
        if (cell <= 0) {
          cell = 0;
          source = Direction::stop;
        }
      }

      current_h[column] = cell;
      current_up[column] = up_score;
      trace[cell_index] = pack_trace(source, up_continues_gap, left_continues_gap);

      if constexpr (LocalAlignment) {
        if (cell > best_score) {
          best_score = cell;
          best_row = row;
          best_column = column;
        }
      }
    }
    std::swap(previous_h, current_h);
    std::swap(previous_up, current_up);
  }

  if constexpr (!LocalAlignment) {
    best_row = query.size();
    best_column = target.size();
  }

  ReverseCigarBuilder cigar;
  std::size_t row = best_row;
  std::size_t column = best_column;
  AffineState state = AffineState::h;

  while (row > 0 || column > 0) {
    const std::size_t cell_index = matrix_index(row, column, columns);
    const std::uint8_t trace_cell = trace[cell_index];
    if constexpr (LocalAlignment) {
      if (state == AffineState::h && trace_direction(trace_cell) == Direction::stop) {
        break;
      }
    }

    if (state == AffineState::h) {
      if (row == 0 && column > 0) {
        state = AffineState::left;
        continue;
      }
      if (column == 0 && row > 0) {
        state = AffineState::up;
        continue;
      }

      const Direction source = trace_direction(trace_cell);
      if (source == Direction::diagonal) {
        cigar.push(query[row - 1U] == target[column - 1U] ? 'M' : 'X');
        --row;
        --column;
        continue;
      }
      if (source == Direction::up) {
        state = AffineState::up;
        continue;
      }
      if (source == Direction::left) {
        state = AffineState::left;
        continue;
      }
      break;
    }

    if (state == AffineState::up) {
      cigar.push('D');
      const bool continues_gap = row > 1U && trace_up_continues(trace_cell);
      --row;
      state = continues_gap ? AffineState::up : AffineState::h;
      continue;
    }

    cigar.push('I');
    const bool continues_gap = column > 1U && trace_left_continues(trace_cell);
    --column;
    state = continues_gap ? AffineState::left : AffineState::h;
  }

  return cigar.str();
}

template <typename Token>
AlignmentResult materialize_alignment_result(
    const PreparedAlignment& prepared,
    const AlignmentPath& path,
    const std::vector<Token>& query_tokens,
    const std::vector<Token>& target_tokens) {
  AlignmentResult result;
  result.score = path.score;
  result.query_start = path.query_start;
  result.query_end = path.query_end;
  result.target_start = path.target_start;
  result.target_end = path.target_end;
  result.operations = path.operations;
  result.aligned_query = materialize_query_output<Token>(
      prepared,
      std::span<const Token>(query_tokens.data(), query_tokens.size()),
      path.query_start,
      path.operations);
  result.aligned_target = materialize_target_output<Token>(
      prepared,
      std::span<const Token>(target_tokens.data(), target_tokens.size()),
      path.target_start,
      path.operations);
  return result;
}

inline AlignmentResult materialize_alignment_result(
    const PreparedAlignment& prepared,
    const AlignmentPath& path) {
  switch (prepared.kernel_bits) {
    case KernelBits::bits8:
      return materialize_alignment_result<std::uint8_t>(
          prepared,
          path,
          std::get<std::vector<std::uint8_t>>(prepared.query_tokens),
          std::get<std::vector<std::uint8_t>>(prepared.target_tokens));
    case KernelBits::bits16:
      return materialize_alignment_result<std::uint16_t>(
          prepared,
          path,
          std::get<std::vector<std::uint16_t>>(prepared.query_tokens),
          std::get<std::vector<std::uint16_t>>(prepared.target_tokens));
    case KernelBits::bits32:
      return materialize_alignment_result<std::uint32_t>(
          prepared,
          path,
          std::get<std::vector<std::uint32_t>>(prepared.query_tokens),
          std::get<std::vector<std::uint32_t>>(prepared.target_tokens));
    case KernelBits::bits64:
      return materialize_alignment_result<std::uint64_t>(
          prepared,
          path,
          std::get<std::vector<std::uint64_t>>(prepared.query_tokens),
          std::get<std::vector<std::uint64_t>>(prepared.target_tokens));
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported kernel width");
  throw nb::python_error();
}

template <typename Fn>
auto dispatch_profile_width(const PreparedFarrarAlignment& prepared, Fn&& fn) {
  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());

  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return fn.template operator()<std::int8_t>(query, target);
    case KernelBits::bits16:
      return fn.template operator()<std::int16_t>(query, target);
    case KernelBits::bits32:
      return fn.template operator()<std::int32_t>(query, target);
    case KernelBits::bits64:
      return fn.template operator()<std::int64_t>(query, target);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported profile traceback width");
  throw nb::python_error();
}

}  // namespace detail

template <bool LocalAlignment>
AlignmentPath linear_path_info(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_score,
    unsigned int width) {
  const auto prepared =
      prepare_farrar_alignment(query, target, match_score, mismatch_score, gap_score, width);
  return detail::dispatch_profile_width(
      prepared,
      [&]<typename Cell>(
          std::span<const std::uint8_t> query_tokens,
          std::span<const std::uint8_t> target_tokens) {
        return detail::linear_path_info<Cell, LocalAlignment>(
            query_tokens,
            target_tokens,
            static_cast<Cell>(match_score),
            static_cast<Cell>(mismatch_score),
            static_cast<Cell>(gap_score));
      });
}

template <bool LocalAlignment>
std::string linear_cigar(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_score,
    unsigned int width) {
  const auto prepared =
      prepare_farrar_alignment(query, target, match_score, mismatch_score, gap_score, width);
  return detail::dispatch_profile_width(
      prepared,
      [&]<typename Cell>(
          std::span<const std::uint8_t> query_tokens,
          std::span<const std::uint8_t> target_tokens) {
        return detail::linear_cigar<Cell, LocalAlignment>(
            query_tokens,
            target_tokens,
            static_cast<Cell>(match_score),
            static_cast<Cell>(mismatch_score),
            static_cast<Cell>(gap_score));
      });
}

template <bool LocalAlignment>
std::optional<std::string> linear_trace_free_cigar(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_score,
    unsigned int width) {
  const auto prepared =
      prepare_farrar_alignment(query, target, match_score, mismatch_score, gap_score, width);
  return detail::dispatch_profile_width(
      prepared,
      [&]<typename Cell>(
          std::span<const std::uint8_t> query_tokens,
          std::span<const std::uint8_t> target_tokens) -> std::optional<std::string> {
        if constexpr (LocalAlignment) {
          return detail::linear_local_trace_free_cigar<Cell>(
              query_tokens,
              target_tokens,
              static_cast<Cell>(match_score),
              static_cast<Cell>(mismatch_score),
              static_cast<Cell>(gap_score));
        } else {
          return detail::linear_global_trace_free_cigar<Cell>(
              query_tokens,
              target_tokens,
              static_cast<Cell>(match_score),
              static_cast<Cell>(mismatch_score),
              static_cast<Cell>(gap_score));
        }
      });
}

template <bool LocalAlignment>
AlignmentResult linear_path(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_score,
    unsigned int width) {
  const auto output_prepared =
      prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
  const auto path =
      linear_path_info<LocalAlignment>(query, target, match_score, mismatch_score, gap_score, width);
  return detail::materialize_alignment_result(output_prepared, path);
}

template <bool LocalAlignment>
Score affine_score(
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
  return detail::dispatch_profile_width(
      prepared,
      [&]<typename Cell>(
          std::span<const std::uint8_t> query_tokens,
          std::span<const std::uint8_t> target_tokens) {
        return detail::affine_score<Cell, LocalAlignment>(
            query_tokens,
            target_tokens,
            static_cast<Cell>(match_score),
            static_cast<Cell>(mismatch_score),
            static_cast<Cell>(gap_open_score),
            static_cast<Cell>(gap_extend_score));
      });
}

template <bool LocalAlignment>
AlignmentPath affine_path_info(
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
  return detail::dispatch_profile_width(
      prepared,
      [&]<typename Cell>(
          std::span<const std::uint8_t> query_tokens,
          std::span<const std::uint8_t> target_tokens) {
        return detail::affine_path_info<Cell, LocalAlignment>(
            query_tokens,
            target_tokens,
            static_cast<Cell>(match_score),
            static_cast<Cell>(mismatch_score),
            static_cast<Cell>(gap_open_score),
            static_cast<Cell>(gap_extend_score));
      });
}

template <bool LocalAlignment>
std::string affine_cigar(
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
  return detail::dispatch_profile_width(
      prepared,
      [&]<typename Cell>(
          std::span<const std::uint8_t> query_tokens,
          std::span<const std::uint8_t> target_tokens) {
        return detail::affine_cigar<Cell, LocalAlignment>(
            query_tokens,
            target_tokens,
            static_cast<Cell>(match_score),
            static_cast<Cell>(mismatch_score),
            static_cast<Cell>(gap_open_score),
            static_cast<Cell>(gap_extend_score));
      });
}

template <bool LocalAlignment>
AlignmentResult affine_path(
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
  const auto path = affine_path_info<LocalAlignment>(
      query,
      target,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score,
      width);
  return detail::materialize_alignment_result(output_prepared, path);
}

}  // namespace stride_align::profile_traceback
