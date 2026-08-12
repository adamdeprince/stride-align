#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "stride_align/types.hpp"

namespace stride_align::pairwise_alignment {

template <typename Token, bool Local>
inline Score linear_score_scalar(
    std::span<const Token> query,
    std::span<const Token> target,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  std::vector<Score> previous(target.size() + 1U, 0);
  std::vector<Score> current(target.size() + 1U, 0);
  if constexpr (!Local) {
    for (std::size_t column = 1; column <= target.size(); ++column) {
      previous[column] = static_cast<Score>(column) * gap_score;
    }
  }

  Score best = 0;
  for (std::size_t row = 1; row <= query.size(); ++row) {
    current[0] = Local ? 0 : static_cast<Score>(row) * gap_score;
    for (std::size_t column = 1; column <= target.size(); ++column) {
      const Score diagonal = previous[column - 1U] +
          (query[row - 1U] == target[column - 1U] ? match_score : mismatch_score);
      const Score up = previous[column] + gap_score;
      const Score left = current[column - 1U] + gap_score;
      Score value = std::max({diagonal, up, left});
      if constexpr (Local) {
        value = std::max<Score>(0, value);
        best = std::max(best, value);
      }
      current[column] = value;
    }
    std::swap(previous, current);
  }
  if constexpr (Local) return best;
  return previous.back();
}

namespace detail {

inline constexpr std::size_t kWavefrontThreshold = 1024U;

inline bool use_wavefront(std::size_t query_size, std::size_t target_size) noexcept {
  if (std::min(query_size, target_size) < 16U) return false;
  return query_size > kWavefrontThreshold / target_size ||
      query_size * target_size >= kWavefrontThreshold;
}

inline std::uint64_t magnitude(Score value) noexcept {
  if (value >= 0) return static_cast<std::uint64_t>(value);
  return static_cast<std::uint64_t>(-(value + 1)) + 1U;
}

template <typename Cell>
inline bool conservative_score_bound_fits(
    std::size_t query_size,
    std::size_t target_size,
    std::initializer_list<Score> scores) noexcept {
  std::uint64_t maximum_magnitude = 0;
  for (const Score score : scores) {
    maximum_magnitude = std::max(maximum_magnitude, magnitude(score));
  }
  if (maximum_magnitude == 0U) return true;

  constexpr auto kCellMaximum = static_cast<std::uint64_t>(
      std::numeric_limits<Cell>::max());
  if (query_size > kCellMaximum || target_size > kCellMaximum) return false;
  const std::uint64_t steps =
      static_cast<std::uint64_t>(query_size) + target_size;
  return steps <= kCellMaximum / maximum_magnitude;
}

inline std::uint64_t saturating_multiply(
    std::uint64_t lhs,
    std::uint64_t rhs) noexcept {
  constexpr auto kMaximum = std::numeric_limits<std::uint64_t>::max();
  if (lhs != 0U && rhs > kMaximum / lhs) return kMaximum;
  return lhs * rhs;
}

inline std::uint64_t saturating_add(
    std::uint64_t lhs,
    std::uint64_t rhs) noexcept {
  constexpr auto kMaximum = std::numeric_limits<std::uint64_t>::max();
  if (rhs > kMaximum - lhs) return kMaximum;
  return lhs + rhs;
}

template <typename Cell>
inline bool linear_score_bound_fits(
    std::size_t query_size,
    std::size_t target_size,
    Score match_score,
    Score mismatch_score,
    Score gap_score) noexcept {
  const auto common = static_cast<std::uint64_t>(
      std::min(query_size, target_size));
  const auto difference = static_cast<std::uint64_t>(
      std::max(query_size, target_size) - std::min(query_size, target_size));
  const auto total = saturating_add(
      static_cast<std::uint64_t>(query_size),
      static_cast<std::uint64_t>(target_size));
  const std::uint64_t substitution = std::max(
      magnitude(match_score), magnitude(mismatch_score));
  const std::uint64_t gap = magnitude(gap_score);

  // A path either substitutes a shared position or consumes its symbols as
  // gaps. The maximum absolute bound therefore occurs at an endpoint: all
  // gaps, or every shared position substituted plus the length difference.
  const std::uint64_t all_gaps = saturating_multiply(total, gap);
  const std::uint64_t all_substitutions = saturating_add(
      saturating_multiply(common, substitution),
      saturating_multiply(difference, gap));
  const std::uint64_t bound = std::max(all_gaps, all_substitutions);
  return bound <= static_cast<std::uint64_t>(
      std::numeric_limits<Cell>::max());
}

template <typename Cell>
inline constexpr Cell negative_infinity_v = std::numeric_limits<Cell>::lowest();

template <typename Cell>
inline Cell add(Cell value, Cell delta) noexcept {
  return value == negative_infinity_v<Cell>
      ? negative_infinity_v<Cell>
      : static_cast<Cell>(value + delta);
}

template <typename Token, typename Cell, bool Local>
Score linear_score_wavefront(
    std::span<const Token> query,
    std::span<const Token> target,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  if (query.size() > target.size()) std::swap(query, target);
  if (query.size() > std::numeric_limits<std::size_t>::max() - target.size()) {
    throw std::length_error("alignment dimensions overflow");
  }

  const std::size_t query_size = query.size();
  const std::size_t target_size = target.size();
  const std::size_t final_diagonal = query_size + target_size;
  std::vector<Token> reversed_target(target.rbegin(), target.rend());

  const auto make_diagonal = [&] {
    return std::vector<Cell>(query_size + 1U, negative_infinity_v<Cell>);
  };
  std::array<std::vector<Cell>, 3> h{
      make_diagonal(), make_diagonal(), make_diagonal()};
  h[0][0] = 0;
  Cell best = 0;

  const Cell match = static_cast<Cell>(match_score);
  const Cell mismatch = static_cast<Cell>(mismatch_score);
  const Cell gap = static_cast<Cell>(gap_score);

  for (std::size_t diagonal = 1U; diagonal <= final_diagonal; ++diagonal) {
    auto& current = h[diagonal % 3U];
    const auto& previous = h[(diagonal - 1U) % 3U];
    const auto& previous2 = h[(diagonal + 1U) % 3U];
    const std::size_t first = diagonal > target_size
        ? diagonal - target_size
        : 0U;
    const std::size_t last = std::min(query_size, diagonal);

    if (first == 0U) {
      current[0] = Local
          ? Cell{0}
          : static_cast<Cell>(static_cast<Score>(diagonal) * gap_score);
    }
    if (last == diagonal) {
      current[last] = Local
          ? Cell{0}
          : static_cast<Cell>(static_cast<Score>(diagonal) * gap_score);
    }

    const std::size_t interior_first = std::max<std::size_t>(1U, first);
    const std::size_t interior_last = std::min(last, diagonal - 1U);
    if (interior_first > interior_last) continue;

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
    for (std::size_t row = interior_first; row <= interior_last; ++row) {
      const std::size_t column = diagonal - row;
      const Cell substitution = query[row - 1U] ==
              reversed_target[target_size - column]
          ? match
          : mismatch;
      const Cell diagonal_score = add(previous2[row - 1U], substitution);
      const Cell up = add(previous[row - 1U], gap);
      const Cell left = add(previous[row], gap);
      Cell value = std::max(diagonal_score, up);
      value = std::max(value, left);
      if constexpr (Local) {
        value = std::max<Cell>(0, value);
        best = std::max(best, value);
      }
      current[row] = value;
    }
  }

  if constexpr (Local) return static_cast<Score>(best);
  return static_cast<Score>(h[final_diagonal % 3U][query_size]);
}

}  // namespace detail

template <typename Token, bool Local>
inline Score linear_score(
    std::span<const Token> query,
    std::span<const Token> target,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  if (!detail::use_wavefront(query.size(), target.size())) {
    return linear_score_scalar<Token, Local>(
        query, target, match_score, mismatch_score, gap_score);
  }
  if (detail::linear_score_bound_fits<std::int8_t>(
          query.size(), target.size(), match_score, mismatch_score, gap_score)) {
    return detail::linear_score_wavefront<Token, std::int8_t, Local>(
        query, target, match_score, mismatch_score, gap_score);
  }
  if (detail::linear_score_bound_fits<std::int16_t>(
          query.size(), target.size(), match_score, mismatch_score, gap_score)) {
    return detail::linear_score_wavefront<Token, std::int16_t, Local>(
        query, target, match_score, mismatch_score, gap_score);
  }
  if (detail::linear_score_bound_fits<std::int32_t>(
          query.size(), target.size(), match_score, mismatch_score, gap_score)) {
    return detail::linear_score_wavefront<Token, std::int32_t, Local>(
        query, target, match_score, mismatch_score, gap_score);
  }
  return detail::linear_score_wavefront<Token, Score, Local>(
      query, target, match_score, mismatch_score, gap_score);
}

inline constexpr Score kNegativeInfinity = std::numeric_limits<Score>::min() / 4;

inline Score add_or_negative_infinity(Score value, Score delta) noexcept {
  return value == kNegativeInfinity ? kNegativeInfinity : value + delta;
}

inline Score gap_cost(
    std::size_t length,
    Score gap_open_score,
    Score gap_extend_score) noexcept {
  if (length == 0U) return 0;
  return gap_open_score + static_cast<Score>(length - 1U) * gap_extend_score;
}

template <typename Token, bool Local>
inline Score affine_score_scalar(
    std::span<const Token> query,
    std::span<const Token> target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  const std::size_t columns = target.size() + 1U;
  std::vector<Score> previous_h(columns, 0);
  std::vector<Score> current_h(columns, 0);
  std::vector<Score> previous_vertical(columns, kNegativeInfinity);
  std::vector<Score> current_vertical(columns, kNegativeInfinity);

  if constexpr (!Local) {
    for (std::size_t column = 1; column < columns; ++column) {
      previous_h[column] = gap_cost(column, gap_open_score, gap_extend_score);
    }
  }

  Score best = 0;
  for (std::size_t row = 1; row <= query.size(); ++row) {
    current_h[0] = Local ? 0 : gap_cost(row, gap_open_score, gap_extend_score);
    current_vertical[0] = Local ? kNegativeInfinity : current_h[0];
    Score horizontal = kNegativeInfinity;

    for (std::size_t column = 1; column <= target.size(); ++column) {
      current_vertical[column] = std::max(
          previous_h[column] + gap_open_score,
          add_or_negative_infinity(previous_vertical[column], gap_extend_score));
      horizontal = std::max(
          current_h[column - 1U] + gap_open_score,
          add_or_negative_infinity(horizontal, gap_extend_score));
      const Score diagonal = previous_h[column - 1U] +
          (query[row - 1U] == target[column - 1U] ? match_score : mismatch_score);
      Score value = std::max({diagonal, current_vertical[column], horizontal});
      if constexpr (Local) {
        value = std::max<Score>(0, value);
        best = std::max(best, value);
      }
      current_h[column] = value;
    }
    std::swap(previous_h, current_h);
    std::swap(previous_vertical, current_vertical);
  }
  if constexpr (Local) return best;
  return previous_h.back();
}

namespace detail {

template <typename Token, typename Cell, bool Local>
Score affine_score_wavefront(
    std::span<const Token> query,
    std::span<const Token> target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  if (query.size() > target.size()) std::swap(query, target);
  if (query.size() > std::numeric_limits<std::size_t>::max() - target.size()) {
    throw std::length_error("alignment dimensions overflow");
  }

  const std::size_t query_size = query.size();
  const std::size_t target_size = target.size();
  const std::size_t final_diagonal = query_size + target_size;
  std::vector<Token> reversed_target(target.rbegin(), target.rend());

  const auto make_diagonal = [&] {
    return std::vector<Cell>(query_size + 1U, negative_infinity_v<Cell>);
  };
  std::array<std::vector<Cell>, 3> h{
      make_diagonal(), make_diagonal(), make_diagonal()};
  std::array<std::vector<Cell>, 3> vertical{
      make_diagonal(), make_diagonal(), make_diagonal()};
  std::array<std::vector<Cell>, 3> horizontal{
      make_diagonal(), make_diagonal(), make_diagonal()};
  h[0][0] = 0;
  Cell best = 0;

  const Cell match = static_cast<Cell>(match_score);
  const Cell mismatch = static_cast<Cell>(mismatch_score);
  const Cell gap_open = static_cast<Cell>(gap_open_score);
  const Cell gap_extend = static_cast<Cell>(gap_extend_score);

  for (std::size_t diagonal = 1U; diagonal <= final_diagonal; ++diagonal) {
    auto& current_h = h[diagonal % 3U];
    auto& current_vertical = vertical[diagonal % 3U];
    auto& current_horizontal = horizontal[diagonal % 3U];
    const auto& previous_h = h[(diagonal - 1U) % 3U];
    const auto& previous_vertical = vertical[(diagonal - 1U) % 3U];
    const auto& previous_horizontal = horizontal[(diagonal - 1U) % 3U];
    const auto& previous2_h = h[(diagonal + 1U) % 3U];
    const std::size_t first = diagonal > target_size
        ? diagonal - target_size
        : 0U;
    const std::size_t last = std::min(query_size, diagonal);

    if (first == 0U) {
      current_h[0] = Local
          ? Cell{0}
          : static_cast<Cell>(gap_cost(
                diagonal, gap_open_score, gap_extend_score));
      current_vertical[0] = negative_infinity_v<Cell>;
      current_horizontal[0] = Local
          ? negative_infinity_v<Cell>
          : current_h[0];
    }
    if (last == diagonal) {
      current_h[last] = Local
          ? Cell{0}
          : static_cast<Cell>(gap_cost(
                diagonal, gap_open_score, gap_extend_score));
      current_vertical[last] = Local
          ? negative_infinity_v<Cell>
          : current_h[last];
      current_horizontal[last] = negative_infinity_v<Cell>;
    }

    const std::size_t interior_first = std::max<std::size_t>(1U, first);
    const std::size_t interior_last = std::min(last, diagonal - 1U);
    if (interior_first > interior_last) continue;

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
    for (std::size_t row = interior_first; row <= interior_last; ++row) {
      const std::size_t column = diagonal - row;
      const Cell substitution = query[row - 1U] ==
              reversed_target[target_size - column]
          ? match
          : mismatch;
      const Cell vertical_score = std::max(
          add(previous_h[row - 1U], gap_open),
          add(previous_vertical[row - 1U], gap_extend));
      const Cell horizontal_score = std::max(
          add(previous_h[row], gap_open),
          add(previous_horizontal[row], gap_extend));
      const Cell diagonal_score = add(previous2_h[row - 1U], substitution);
      Cell value = std::max(diagonal_score, vertical_score);
      value = std::max(value, horizontal_score);
      if constexpr (Local) {
        value = std::max<Cell>(0, value);
        best = std::max(best, value);
      }
      current_h[row] = value;
      current_vertical[row] = vertical_score;
      current_horizontal[row] = horizontal_score;
    }
  }

  if constexpr (Local) return static_cast<Score>(best);
  return static_cast<Score>(h[final_diagonal % 3U][query_size]);
}

}  // namespace detail

template <typename Token, bool Local>
inline Score affine_score(
    std::span<const Token> query,
    std::span<const Token> target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  if (!detail::use_wavefront(query.size(), target.size())) {
    return affine_score_scalar<Token, Local>(
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score);
  }
  const auto scores = {
      match_score, mismatch_score, gap_open_score, gap_extend_score};
  if (detail::conservative_score_bound_fits<std::int8_t>(
          query.size(), target.size(), scores)) {
    return detail::affine_score_wavefront<Token, std::int8_t, Local>(
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score);
  }
  if (detail::conservative_score_bound_fits<std::int16_t>(
          query.size(), target.size(), scores)) {
    return detail::affine_score_wavefront<Token, std::int16_t, Local>(
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score);
  }
  if (detail::conservative_score_bound_fits<std::int32_t>(
          query.size(), target.size(), scores)) {
    return detail::affine_score_wavefront<Token, std::int32_t, Local>(
        query, target, match_score, mismatch_score,
        gap_open_score, gap_extend_score);
  }
  return detail::affine_score_wavefront<Token, Score, Local>(
      query, target, match_score, mismatch_score,
      gap_open_score, gap_extend_score);
}

}  // namespace stride_align::pairwise_alignment
