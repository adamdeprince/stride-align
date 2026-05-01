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

struct PreparedAffineCigar {
  PreparedFarrarAlignment prepared;
  Score match_score = 2;
  Score mismatch_score = -1;
  Score gap_open_score = -1;
  Score gap_extend_score = -1;
  std::optional<Score> expected_score = std::nullopt;
};

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

inline std::size_t absolute_difference(std::size_t lhs, std::size_t rhs) noexcept {
  return lhs >= rhs ? lhs - rhs : rhs - lhs;
}

inline std::size_t default_affine_cigar_band_radius(
    std::size_t query_size,
    std::size_t target_size) noexcept {
  const std::size_t maximum_size = std::max(query_size, target_size);
  const std::size_t slack = std::clamp<std::size_t>(maximum_size / 16U, 32U, 128U);
  return absolute_difference(query_size, target_size) + slack;
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

template <typename Cell, bool LocalAlignment>
std::optional<std::string> affine_banded_cigar(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_open_score,
    Cell gap_extend_score,
    Score expected_score) {
  if constexpr (LocalAlignment) {
    if (expected_score <= 0) {
      return std::string();
    }
  }

  const std::size_t query_size = query.size();
  const std::size_t target_size = target.size();
  const std::size_t radius = default_affine_cigar_band_radius(query_size, target_size);
  if (radius >= std::max(query_size, target_size)) {
    return std::nullopt;
  }

  const std::size_t rows = query_size + 1U;
  const std::size_t columns = target_size + 1U;
  const std::size_t cell_count = rows * columns;
  constexpr Cell negative_infinity = negative_infinity_v<Cell>;

  std::vector<Cell> previous_h(columns, negative_infinity);
  std::vector<Cell> current_h(columns, negative_infinity);
  std::vector<Cell> previous_up(columns, negative_infinity);
  std::vector<Cell> current_up(columns, negative_infinity);
  TraceTable<std::uint8_t> trace(cell_count);

  const std::size_t first_row_max = std::min(target_size, radius);
  previous_h[0] = 0;
  trace[matrix_index(0, 0, columns)] = pack_trace(Direction::stop, false, false);
  if constexpr (!LocalAlignment) {
    for (std::size_t column = 1U; column <= first_row_max; ++column) {
      previous_h[column] = gap_cost<Cell>(column, gap_open_score, gap_extend_score);
      trace[matrix_index(0, column, columns)] =
          pack_trace(Direction::left, false, column > 1U);
    }
  } else {
    for (std::size_t column = 1U; column <= first_row_max; ++column) {
      previous_h[column] = 0;
      trace[matrix_index(0, column, columns)] = pack_trace(Direction::stop, false, false);
    }
  }

  Cell best_score = 0;
  std::size_t best_row = 0;
  std::size_t best_column = 0;

  for (std::size_t row = 1U; row <= query_size; ++row) {
    const std::size_t row_min = row > radius ? row - radius : 0U;
    const std::size_t row_max = std::min(target_size, row + radius);
    if (row_min > row_max) {
      return std::nullopt;
    }

    const std::size_t clear_begin = row_min > 0U ? row_min - 1U : 0U;
    const std::size_t clear_end = std::min(columns, row_max + 2U);
    std::fill(current_h.begin() + static_cast<std::ptrdiff_t>(clear_begin),
              current_h.begin() + static_cast<std::ptrdiff_t>(clear_end),
              negative_infinity);
    std::fill(current_up.begin() + static_cast<std::ptrdiff_t>(clear_begin),
              current_up.begin() + static_cast<std::ptrdiff_t>(clear_end),
              negative_infinity);

    if (row_min == 0U) {
      if constexpr (LocalAlignment) {
        current_h[0] = 0;
        current_up[0] = negative_infinity;
        trace[matrix_index(row, 0, columns)] = pack_trace(Direction::stop, false, false);
      } else {
        current_h[0] = gap_cost<Cell>(row, gap_open_score, gap_extend_score);
        current_up[0] = current_h[0];
        trace[matrix_index(row, 0, columns)] =
            pack_trace(Direction::up, row > 1U, false);
      }
    }

    Cell left_score = negative_infinity;
    const std::size_t start_column = std::max<std::size_t>(1U, row_min);
    for (std::size_t column = start_column; column <= row_max; ++column) {
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
      trace[matrix_index(row, column, columns)] =
          pack_trace(source, up_continues_gap, left_continues_gap);

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
    best_score = previous_h[target_size];
    best_row = query_size;
    best_column = target_size;
  }
  if (best_score == negative_infinity || static_cast<Score>(best_score) != expected_score) {
    return std::nullopt;
  }

  ReverseCigarBuilder cigar;
  std::size_t row = best_row;
  std::size_t column = best_column;
  AffineState state = AffineState::h;

  while (row > 0 || column > 0) {
    if (absolute_difference(row, column) > radius) {
      return std::nullopt;
    }

    const std::uint8_t trace_cell = trace[matrix_index(row, column, columns)];
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
      return std::nullopt;
    }

    if (state == AffineState::up) {
      if (row == 0U) {
        return std::nullopt;
      }
      cigar.push('D');
      const bool continues_gap = row > 1U && trace_up_continues(trace_cell);
      --row;
      state = continues_gap ? AffineState::up : AffineState::h;
      continue;
    }

    if (column == 0U) {
      return std::nullopt;
    }
    cigar.push('I');
    const bool continues_gap = column > 1U && trace_left_continues(trace_cell);
    --column;
    state = continues_gap ? AffineState::left : AffineState::h;
  }

  return cigar.str();
}

template <typename Cell, bool LocalAlignment>
std::string affine_cigar_fast(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_open_score,
    Cell gap_extend_score,
    std::optional<Score> expected_score = std::nullopt) {
  const Score verified_score = expected_score.has_value()
      ? *expected_score
      : affine_score<Cell, LocalAlignment>(
            query,
            target,
            match_score,
            mismatch_score,
            gap_open_score,
            gap_extend_score);
  if (auto cigar = affine_banded_cigar<Cell, LocalAlignment>(
          query,
          target,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score,
          verified_score);
      cigar.has_value()) {
    return *cigar;
  }
  return affine_cigar<Cell, LocalAlignment>(
      query,
      target,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score);
}

template <typename Cell>
struct AffineCigarCheckpoints {
  std::size_t block_size = 0;
  std::size_t checkpoint_count = 0;
  std::size_t columns = 0;
  std::vector<Cell> h;
  std::vector<Cell> up;

  Cell* h_at(std::size_t checkpoint_index) noexcept {
    return h.data() + checkpoint_index * columns;
  }

  const Cell* h_at(std::size_t checkpoint_index) const noexcept {
    return h.data() + checkpoint_index * columns;
  }

  Cell* up_at(std::size_t checkpoint_index) noexcept {
    return up.data() + checkpoint_index * columns;
  }

  const Cell* up_at(std::size_t checkpoint_index) const noexcept {
    return up.data() + checkpoint_index * columns;
  }
};

inline std::size_t affine_cigar_checkpoint_block_size(std::size_t columns) noexcept {
  constexpr std::size_t target_trace_bytes = 64U * 1024U;
  const std::size_t raw_block_size = std::max<std::size_t>(1U, target_trace_bytes / columns);
  return std::clamp<std::size_t>(raw_block_size, 16U, 128U);
}

template <typename Cell>
AffineCigarCheckpoints<Cell> make_affine_cigar_checkpoints(
    std::size_t query_size,
    std::size_t target_size) {
  AffineCigarCheckpoints<Cell> checkpoints;
  checkpoints.columns = target_size + 1U;
  checkpoints.block_size = affine_cigar_checkpoint_block_size(checkpoints.columns);
  checkpoints.checkpoint_count = query_size / checkpoints.block_size + 1U;
  checkpoints.h.resize(checkpoints.checkpoint_count * checkpoints.columns);
  checkpoints.up.resize(checkpoints.checkpoint_count * checkpoints.columns);
  return checkpoints;
}

template <typename Cell>
void save_affine_cigar_checkpoint(
    AffineCigarCheckpoints<Cell>& checkpoints,
    std::size_t checkpoint_index,
    const std::vector<Cell>& h,
    const std::vector<Cell>& up) {
  std::copy_n(h.data(), checkpoints.columns, checkpoints.h_at(checkpoint_index));
  std::copy_n(up.data(), checkpoints.columns, checkpoints.up_at(checkpoint_index));
}

template <typename Cell, bool LocalAlignment>
LocalEndpoint<Cell> affine_cigar_checkpoint_forward(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_open_score,
    Cell gap_extend_score,
    AffineCigarCheckpoints<Cell>& checkpoints) {
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
  save_affine_cigar_checkpoint(checkpoints, 0U, previous_h, previous_up);

  LocalEndpoint<Cell> endpoint;
  if constexpr (!LocalAlignment) {
    endpoint = {previous_h.back(), query.size(), target.size()};
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
    for (std::size_t column = 1; column < columns; ++column) {
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
        if (cell > endpoint.score) {
          endpoint = {cell, row, column};
        }
      }

      current_h[column] = cell;
      current_up[column] = up_score;
    }

    std::swap(previous_h, current_h);
    std::swap(previous_up, current_up);
    if (row % checkpoints.block_size == 0) {
      save_affine_cigar_checkpoint(
          checkpoints,
          row / checkpoints.block_size,
          previous_h,
          previous_up);
    }
  }

  if constexpr (!LocalAlignment) {
    endpoint = {previous_h.back(), query.size(), target.size()};
  }
  return endpoint;
}

template <typename Cell, bool LocalAlignment>
void affine_cigar_recompute_trace_block(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_open_score,
    Cell gap_extend_score,
    const AffineCigarCheckpoints<Cell>& checkpoints,
    std::size_t block_start,
    std::size_t block_end,
    std::vector<std::uint8_t>& block_trace) {
  const std::size_t columns = target.size() + 1U;
  const std::size_t block_rows = block_end - block_start;
  block_trace.resize(block_rows * columns);

  std::vector<Cell> previous_h(columns, LocalAlignment ? Cell{0} : negative_infinity_v<Cell>);
  std::vector<Cell> current_h(columns, LocalAlignment ? Cell{0} : negative_infinity_v<Cell>);
  std::vector<Cell> previous_up(columns, negative_infinity_v<Cell>);
  std::vector<Cell> current_up(columns, negative_infinity_v<Cell>);

  const std::size_t checkpoint_index = block_start / checkpoints.block_size;
  std::copy_n(checkpoints.h_at(checkpoint_index), columns, previous_h.data());
  std::copy_n(checkpoints.up_at(checkpoint_index), columns, previous_up.data());

  for (std::size_t row = block_start + 1U; row <= block_end; ++row) {
    const std::size_t trace_row_offset = (row - block_start - 1U) * columns;
    if constexpr (LocalAlignment) {
      current_h[0] = 0;
      current_up[0] = negative_infinity_v<Cell>;
      block_trace[trace_row_offset] = pack_trace(Direction::stop, false, false);
    } else {
      current_h[0] = gap_cost<Cell>(row, gap_open_score, gap_extend_score);
      current_up[0] = current_h[0];
      block_trace[trace_row_offset] = pack_trace(Direction::up, row > 1U, false);
    }

    Cell left_score = negative_infinity_v<Cell>;
    for (std::size_t column = 1; column < columns; ++column) {
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
      block_trace[trace_row_offset + column] =
          pack_trace(source, up_continues_gap, left_continues_gap);
    }

    std::swap(previous_h, current_h);
    std::swap(previous_up, current_up);
  }
}

template <typename Cell, bool LocalAlignment>
std::string affine_checkpointed_cigar(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_open_score,
    Cell gap_extend_score) {
  if constexpr (LocalAlignment) {
    if (query.empty() || target.empty()) {
      return std::string();
    }
  }

  auto checkpoints = make_affine_cigar_checkpoints<Cell>(query.size(), target.size());
  const LocalEndpoint<Cell> endpoint = affine_cigar_checkpoint_forward<Cell, LocalAlignment>(
      query,
      target,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score,
      checkpoints);
  if constexpr (LocalAlignment) {
    if (endpoint.score <= 0) {
      return std::string();
    }
  }

  ReverseCigarBuilder cigar;
  std::vector<std::uint8_t> block_trace;
  std::size_t row = endpoint.row;
  std::size_t column = endpoint.column;
  AffineState state = AffineState::h;

  while (row > 0 || column > 0) {
    if constexpr (LocalAlignment) {
      if (row == 0 || column == 0) {
        break;
      }
    } else {
      if (row == 0) {
        while (column > 0) {
          cigar.push('I');
          --column;
        }
        break;
      }
      if (column == 0) {
        while (row > 0) {
          cigar.push('D');
          --row;
        }
        break;
      }
    }

    const std::size_t block_start =
        ((row - 1U) / checkpoints.block_size) * checkpoints.block_size;
    const std::size_t block_end = std::min(block_start + checkpoints.block_size, query.size());
    affine_cigar_recompute_trace_block<Cell, LocalAlignment>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        checkpoints,
        block_start,
        block_end,
        block_trace);

    while (row > block_start) {
      const std::size_t trace_index = (row - block_start - 1U) * checkpoints.columns + column;
      const std::uint8_t trace_cell = block_trace[trace_index];
      if constexpr (LocalAlignment) {
        if (state == AffineState::h && trace_direction(trace_cell) == Direction::stop) {
          row = 0;
          column = 0;
          break;
        }
      }

      if (state == AffineState::h) {
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
        row = 0;
        column = 0;
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
  }

  return cigar.str();
}

template <typename Token, typename Path>
AlignmentResult materialize_alignment_result(
    const PreparedAlignment& prepared,
    const Path& path,
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

template <typename Token>
AlignmentResult materialize_alignment_result(
    const PreparedAlignment& prepared,
    const AlignmentPath& path,
    const std::vector<Token>& query_tokens,
    const std::vector<Token>& target_tokens) {
  return materialize_alignment_result<Token, AlignmentPath>(
      prepared,
      path,
      query_tokens,
      target_tokens);
}

template <typename Path>
inline AlignmentResult materialize_alignment_result(
    const PreparedAlignment& prepared,
    const Path& path) {
  switch (prepared.kernel_bits) {
    case KernelBits::bits8:
      return materialize_alignment_result<std::uint8_t, Path>(
          prepared,
          path,
          std::get<std::vector<std::uint8_t>>(prepared.query_tokens),
          std::get<std::vector<std::uint8_t>>(prepared.target_tokens));
    case KernelBits::bits16:
      return materialize_alignment_result<std::uint16_t, Path>(
          prepared,
          path,
          std::get<std::vector<std::uint16_t>>(prepared.query_tokens),
          std::get<std::vector<std::uint16_t>>(prepared.target_tokens));
    case KernelBits::bits32:
      return materialize_alignment_result<std::uint32_t, Path>(
          prepared,
          path,
          std::get<std::vector<std::uint32_t>>(prepared.query_tokens),
          std::get<std::vector<std::uint32_t>>(prepared.target_tokens));
    case KernelBits::bits64:
      return materialize_alignment_result<std::uint64_t, Path>(
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
        return detail::affine_cigar_fast<Cell, LocalAlignment>(
            query_tokens,
            target_tokens,
            static_cast<Cell>(match_score),
            static_cast<Cell>(mismatch_score),
            static_cast<Cell>(gap_open_score),
            static_cast<Cell>(gap_extend_score));
      });
}

template <bool LocalAlignment>
std::string affine_cigar_with_score(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width,
    Score expected_score) {
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
        return detail::affine_cigar_fast<Cell, LocalAlignment>(
            query_tokens,
            target_tokens,
            static_cast<Cell>(match_score),
            static_cast<Cell>(mismatch_score),
            static_cast<Cell>(gap_open_score),
            static_cast<Cell>(gap_extend_score),
            expected_score);
      });
}

inline PreparedAffineCigar prepare_affine_cigar(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width,
    std::optional<Score> expected_score = std::nullopt) {
  return PreparedAffineCigar{
      prepare_farrar_alignment(
          query,
          target,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score,
          width),
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score,
      expected_score};
}

template <bool LocalAlignment>
std::string affine_cigar_prepared(PreparedAffineCigar& prepared) {
  return detail::dispatch_profile_width(
      prepared.prepared,
      [&]<typename Cell>(
          std::span<const std::uint8_t> query_tokens,
          std::span<const std::uint8_t> target_tokens) {
        return detail::affine_cigar_fast<Cell, LocalAlignment>(
            query_tokens,
            target_tokens,
            static_cast<Cell>(prepared.match_score),
            static_cast<Cell>(prepared.mismatch_score),
            static_cast<Cell>(prepared.gap_open_score),
            static_cast<Cell>(prepared.gap_extend_score),
            prepared.expected_score);
      });
}

template <bool LocalAlignment>
std::string affine_checkpointed_cigar(
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
        return detail::affine_checkpointed_cigar<Cell, LocalAlignment>(
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
