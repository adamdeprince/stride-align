#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "stride_align/hamming.hpp"
#include "stride_align/indel.hpp"
#include "stride_align/jaro.hpp"
#include "stride_align/levenshtein.hpp"
#include "stride_align/alignment.hpp"
#include "stride_align/pairwise_alignment.hpp"
#include "stride_align/utf8.hpp"

namespace stride_align::core {

namespace detail {

inline std::span<const std::uint8_t> u8_span(const utf8::TokenBuffer& buffer) {
  if (const auto* borrowed = std::get_if<std::span<const std::uint8_t>>(&buffer)) {
    return *borrowed;
  }
  const auto& owned = std::get<std::vector<std::uint8_t>>(buffer);
  return std::span<const std::uint8_t>(owned.data(), owned.size());
}

template <typename Function>
decltype(auto) visit_pair(const utf8::PreparedPair& pair, Function&& function) {
  switch (pair.width) {
    case utf8::TokenWidth::u8:
      return function(u8_span(pair.query), u8_span(pair.target));
    case utf8::TokenWidth::u16: {
      const auto& query = std::get<std::vector<std::uint16_t>>(pair.query);
      const auto& target = std::get<std::vector<std::uint16_t>>(pair.target);
      return function(
          std::span<const std::uint16_t>(query.data(), query.size()),
          std::span<const std::uint16_t>(target.data(), target.size()));
    }
    case utf8::TokenWidth::u32: {
      const auto& query = std::get<std::vector<std::uint32_t>>(pair.query);
      const auto& target = std::get<std::vector<std::uint32_t>>(pair.target);
      return function(
          std::span<const std::uint32_t>(query.data(), query.size()),
          std::span<const std::uint32_t>(target.data(), target.size()));
    }
  }
  throw std::logic_error("unsupported token width");
}

template <typename Token>
inline std::size_t levenshtein_typed(
    std::span<const Token> query,
    std::span<const Token> target,
    std::size_t cutoff) {
  std::span<const Token> pattern = query;
  std::span<const Token> text = target;
  if (pattern.size() > text.size()) std::swap(pattern, text);
  if constexpr (std::is_same_v<Token, std::uint8_t>) {
    return pattern.size() <= 64U
        ? levenshtein::myers_single_word_u8(pattern, text, cutoff)
        : levenshtein::myers_multi_word_u8(pattern, text, cutoff);
  } else {
    return levenshtein::myers_distance<Token>(pattern, text, cutoff);
  }
}

template <typename Token>
inline std::size_t indel_typed(
    std::span<const Token> query,
    std::span<const Token> target,
    std::size_t cutoff) {
  std::span<const Token> pattern = query;
  std::span<const Token> text = target;
  if (pattern.size() > text.size()) std::swap(pattern, text);
  if constexpr (std::is_same_v<Token, std::uint8_t>) {
    return indel::indel_distance_u8(pattern, text, cutoff);
  } else {
    return indel::indel_distance<Token>(pattern, text, cutoff);
  }
}

template <typename Token>
inline double jaro_typed(
    std::span<const Token> query,
    std::span<const Token> target) {
  if constexpr (std::is_same_v<Token, std::uint8_t>) {
    return query.size() <= 64U && target.size() <= 64U
        ? jaro::jaro_bp_byte_64(query, target)
        : jaro::jaro_bp_byte_multiword(query, target);
  } else {
    return jaro::jaro_bp_token_multiword<Token>(query, target);
  }
}

}  // namespace detail

inline std::size_t levenshtein_distance(
    const utf8::PreparedPair& pair,
    std::size_t cutoff = levenshtein::kNoCutoff) {
  return detail::visit_pair(pair, [cutoff](auto query, auto target) {
    return detail::levenshtein_typed(query, target, cutoff);
  });
}

inline double levenshtein_similarity(const utf8::PreparedPair& pair) {
  return levenshtein::normalize(
      levenshtein_distance(pair), pair.query_size(), pair.target_size());
}

inline std::size_t osa_distance(const utf8::PreparedPair& pair) {
  return detail::visit_pair(pair, [](auto query, auto target) {
    using Token = typename decltype(query)::value_type;
    if constexpr (std::is_same_v<Token, std::uint8_t>) {
      return levenshtein::osa_distance_u8(query, target);
    } else {
      return levenshtein::osa_distance<Token>(query, target);
    }
  });
}

inline double osa_similarity(const utf8::PreparedPair& pair) {
  return levenshtein::normalize(
      osa_distance(pair), pair.query_size(), pair.target_size());
}

inline std::size_t true_damerau_levenshtein_distance(
    const utf8::PreparedPair& pair) {
  return detail::visit_pair(pair, [](auto query, auto target) {
    using Token = typename decltype(query)::value_type;
    if constexpr (std::is_same_v<Token, std::uint8_t>) {
      return levenshtein::true_damerau_levenshtein_distance_u8(query, target);
    } else {
      return levenshtein::true_damerau_levenshtein_distance<Token>(query, target);
    }
  });
}

inline double true_damerau_levenshtein_similarity(const utf8::PreparedPair& pair) {
  return levenshtein::normalize(
      true_damerau_levenshtein_distance(pair), pair.query_size(), pair.target_size());
}

inline std::size_t indel_distance(
    const utf8::PreparedPair& pair,
    std::size_t cutoff = indel::kNoCutoff) {
  return detail::visit_pair(pair, [cutoff](auto query, auto target) {
    return detail::indel_typed(query, target, cutoff);
  });
}

inline double indel_similarity(const utf8::PreparedPair& pair) {
  return indel::normalize(
      indel_distance(pair), pair.query_size(), pair.target_size());
}

inline std::size_t hamming_distance(const utf8::PreparedPair& pair) {
  if (pair.query_size() != pair.target_size()) {
    throw std::invalid_argument("Hamming distance requires equal-length strings");
  }
  return detail::visit_pair(pair, [](auto query, auto target) {
    using Token = typename decltype(query)::value_type;
    if constexpr (std::is_same_v<Token, std::uint8_t>) {
      return hamming::hamming_u8(query, target);
    } else {
      return hamming::hamming_scalar<Token>(query, target);
    }
  });
}

inline double hamming_similarity(const utf8::PreparedPair& pair) {
  return hamming::normalize(hamming_distance(pair), pair.query_size());
}

inline double jaro_similarity(const utf8::PreparedPair& pair) {
  return detail::visit_pair(pair, [](auto query, auto target) {
    return detail::jaro_typed(query, target);
  });
}

inline double jaro_winkler_similarity(
    const utf8::PreparedPair& pair,
    double prefix_weight = jaro::kDefaultPrefixWeight,
    double prefix_threshold = jaro::kDefaultPrefixThreshold,
    std::size_t prefix_cap = jaro::kDefaultPrefixCap) {
  return detail::visit_pair(pair, [&](auto query, auto target) {
    using Token = typename decltype(query)::value_type;
    const double base = detail::jaro_typed(query, target);
    if (base < prefix_threshold) return base;
    const std::size_t prefix = jaro::common_prefix<Token>(query, target, prefix_cap);
    return base + static_cast<double>(prefix) * prefix_weight * (1.0 - base);
  });
}

inline Score smith_waterman_score(
    const utf8::PreparedPair& pair,
    Score match_score = 2,
    Score mismatch_score = -1,
    Score gap_score = -1) {
  return detail::visit_pair(pair, [&](auto query, auto target) {
    return pairwise_alignment::linear_score<
        typename decltype(query)::value_type, true>(
            query, target, match_score, mismatch_score, gap_score);
  });
}

inline Score needleman_wunsch_score(
    const utf8::PreparedPair& pair,
    Score match_score = 2,
    Score mismatch_score = -1,
    Score gap_score = -1) {
  return detail::visit_pair(pair, [&](auto query, auto target) {
    return pairwise_alignment::linear_score<
        typename decltype(query)::value_type, false>(
            query, target, match_score, mismatch_score, gap_score);
  });
}

inline Score smith_waterman_affine_score(
    const utf8::PreparedPair& pair,
    Score match_score = 2,
    Score mismatch_score = -1,
    Score gap_open_score = -2,
    Score gap_extend_score = -1) {
  return detail::visit_pair(pair, [&](auto query, auto target) {
    return pairwise_alignment::affine_score<
        typename decltype(query)::value_type, true>(
            query, target, match_score, mismatch_score,
            gap_open_score, gap_extend_score);
  });
}

inline Score needleman_wunsch_affine_score(
    const utf8::PreparedPair& pair,
    Score match_score = 2,
    Score mismatch_score = -1,
    Score gap_open_score = -2,
    Score gap_extend_score = -1) {
  return detail::visit_pair(pair, [&](auto query, auto target) {
    return pairwise_alignment::affine_score<
        typename decltype(query)::value_type, false>(
            query, target, match_score, mismatch_score,
            gap_open_score, gap_extend_score);
  });
}

namespace detail {

enum class TraceDirection : std::uint8_t { stop, diagonal, up, left };
enum class TraceState : std::uint8_t { h, vertical, horizontal };

template <typename Token, bool Local>
AlignmentPath linear_path_typed(
    std::span<const Token> query,
    std::span<const Token> target,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  const std::size_t rows = query.size() + 1U;
  const std::size_t columns = target.size() + 1U;
  if (rows != 0U && columns > std::numeric_limits<std::size_t>::max() / rows) {
    throw std::length_error("alignment traceback dimensions overflow");
  }
  const auto index = [columns](std::size_t row, std::size_t column) {
    return row * columns + column;
  };
  std::vector<Score> scores(rows * columns, 0);
  std::vector<TraceDirection> directions(rows * columns, TraceDirection::stop);
  if constexpr (!Local) {
    for (std::size_t row = 1; row < rows; ++row) {
      scores[index(row, 0)] = static_cast<Score>(row) * gap_score;
      directions[index(row, 0)] = TraceDirection::up;
    }
    for (std::size_t column = 1; column < columns; ++column) {
      scores[index(0, column)] = static_cast<Score>(column) * gap_score;
      directions[index(0, column)] = TraceDirection::left;
    }
  }

  Score best_score = 0;
  std::size_t best_row = 0;
  std::size_t best_column = 0;
  for (std::size_t row = 1; row < rows; ++row) {
    for (std::size_t column = 1; column < columns; ++column) {
      const Score diagonal = scores[index(row - 1U, column - 1U)] +
          (query[row - 1U] == target[column - 1U] ? match_score : mismatch_score);
      const Score up = scores[index(row - 1U, column)] + gap_score;
      const Score left = scores[index(row, column - 1U)] + gap_score;
      Score value = diagonal;
      TraceDirection direction = TraceDirection::diagonal;
      if (up > value) {
        value = up;
        direction = TraceDirection::up;
      }
      if (left > value) {
        value = left;
        direction = TraceDirection::left;
      }
      if constexpr (Local) {
        if (value <= 0) {
          value = 0;
          direction = TraceDirection::stop;
        }
      }
      scores[index(row, column)] = value;
      directions[index(row, column)] = direction;
      if constexpr (Local) {
        if (value > best_score) {
          best_score = value;
          best_row = row;
          best_column = column;
        }
      }
    }
  }
  if constexpr (!Local) {
    best_row = query.size();
    best_column = target.size();
    best_score = scores[index(best_row, best_column)];
  }

  const std::size_t query_end = best_row;
  const std::size_t target_end = best_column;
  std::string operations;
  operations.reserve(best_row + best_column);
  while (best_row > 0U || best_column > 0U) {
    const TraceDirection direction = directions[index(best_row, best_column)];
    if (direction == TraceDirection::stop) break;
    if (direction == TraceDirection::diagonal) {
      operations.push_back(
          query[best_row - 1U] == target[best_column - 1U] ? '=' : 'X');
      --best_row;
      --best_column;
    } else if (direction == TraceDirection::up) {
      operations.push_back('D');
      --best_row;
    } else {
      operations.push_back('I');
      --best_column;
    }
  }
  std::reverse(operations.begin(), operations.end());
  return make_alignment_path(
      best_score, best_row, query_end, best_column, target_end, operations);
}

inline constexpr Score kTraceNegativeInfinity =
    std::numeric_limits<Score>::lowest() / 4;

inline Score trace_add(Score value, Score delta) noexcept {
  return value <= kTraceNegativeInfinity / 2
      ? kTraceNegativeInfinity
      : value + delta;
}

template <typename Token, bool Local>
AlignmentPath affine_path_typed(
    std::span<const Token> query,
    std::span<const Token> target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  const std::size_t rows = query.size() + 1U;
  const std::size_t columns = target.size() + 1U;
  if (rows != 0U && columns > std::numeric_limits<std::size_t>::max() / rows) {
    throw std::length_error("alignment traceback dimensions overflow");
  }
  const auto index = [columns](std::size_t row, std::size_t column) {
    return row * columns + column;
  };
  const auto gap_cost = [&](std::size_t length) {
    return length == 0U
        ? Score{0}
        : gap_open_score + static_cast<Score>(length - 1U) * gap_extend_score;
  };
  std::vector<Score> h(
      rows * columns, Local ? Score{0} : kTraceNegativeInfinity);
  std::vector<Score> vertical(rows * columns, kTraceNegativeInfinity);
  std::vector<Score> horizontal(rows * columns, kTraceNegativeInfinity);
  h[index(0, 0)] = 0;
  if constexpr (!Local) {
    for (std::size_t row = 1; row < rows; ++row) {
      h[index(row, 0)] = gap_cost(row);
      vertical[index(row, 0)] = h[index(row, 0)];
    }
    for (std::size_t column = 1; column < columns; ++column) {
      h[index(0, column)] = gap_cost(column);
      horizontal[index(0, column)] = h[index(0, column)];
    }
  }
  Score best_score = 0;
  std::size_t best_row = 0;
  std::size_t best_column = 0;
  for (std::size_t row = 1; row < rows; ++row) {
    for (std::size_t column = 1; column < columns; ++column) {
      const std::size_t cell = index(row, column);
      vertical[cell] = std::max(
          trace_add(h[index(row - 1U, column)], gap_open_score),
          trace_add(vertical[index(row - 1U, column)], gap_extend_score));
      horizontal[cell] = std::max(
          trace_add(h[index(row, column - 1U)], gap_open_score),
          trace_add(horizontal[index(row, column - 1U)], gap_extend_score));
      const Score diagonal = trace_add(
          h[index(row - 1U, column - 1U)],
          query[row - 1U] == target[column - 1U]
              ? match_score
              : mismatch_score);
      Score value = std::max({diagonal, vertical[cell], horizontal[cell]});
      if constexpr (Local) value = std::max<Score>(0, value);
      h[cell] = value;
      if constexpr (Local) {
        if (value > best_score) {
          best_score = value;
          best_row = row;
          best_column = column;
        }
      }
    }
  }
  if constexpr (!Local) {
    best_row = query.size();
    best_column = target.size();
    best_score = h[index(best_row, best_column)];
  }
  const std::size_t query_end = best_row;
  const std::size_t target_end = best_column;
  TraceState state = TraceState::h;
  std::string operations;
  operations.reserve(best_row + best_column);
  while (best_row > 0U || best_column > 0U) {
    const std::size_t cell = index(best_row, best_column);
    if constexpr (Local) {
      if (state == TraceState::h && h[cell] <= 0) break;
    }
    if (state == TraceState::h) {
      if (best_row > 0U && best_column > 0U) {
        const Score diagonal = trace_add(
            h[index(best_row - 1U, best_column - 1U)],
            query[best_row - 1U] == target[best_column - 1U]
                ? match_score
                : mismatch_score);
        if (h[cell] == diagonal) {
          operations.push_back(
              query[best_row - 1U] == target[best_column - 1U] ? '=' : 'X');
          --best_row;
          --best_column;
          continue;
        }
      }
      if (best_row > 0U && h[cell] == vertical[cell]) {
        state = TraceState::vertical;
        continue;
      }
      if (best_column > 0U && h[cell] == horizontal[cell]) {
        state = TraceState::horizontal;
        continue;
      }
      break;
    }
    if (state == TraceState::vertical) {
      operations.push_back('D');
      const bool continues = best_row > 1U &&
          vertical[cell] == trace_add(
              vertical[index(best_row - 1U, best_column)], gap_extend_score);
      --best_row;
      state = continues ? TraceState::vertical : TraceState::h;
    } else {
      operations.push_back('I');
      const bool continues = best_column > 1U &&
          horizontal[cell] == trace_add(
              horizontal[index(best_row, best_column - 1U)], gap_extend_score);
      --best_column;
      state = continues ? TraceState::horizontal : TraceState::h;
    }
  }
  std::reverse(operations.begin(), operations.end());
  return make_alignment_path(
      best_score, best_row, query_end, best_column, target_end, operations);
}

}  // namespace detail

inline AlignmentPath smith_waterman_path(
    const utf8::PreparedPair& pair,
    Score match_score = 2,
    Score mismatch_score = -1,
    Score gap_score = -1) {
  return detail::visit_pair(pair, [&](auto query, auto target) {
    return detail::linear_path_typed<
        typename decltype(query)::value_type, true>(
            query, target, match_score, mismatch_score, gap_score);
  });
}

inline AlignmentPath needleman_wunsch_path(
    const utf8::PreparedPair& pair,
    Score match_score = 2,
    Score mismatch_score = -1,
    Score gap_score = -1) {
  return detail::visit_pair(pair, [&](auto query, auto target) {
    return detail::linear_path_typed<
        typename decltype(query)::value_type, false>(
            query, target, match_score, mismatch_score, gap_score);
  });
}

inline AlignmentPath smith_waterman_affine_path(
    const utf8::PreparedPair& pair,
    Score match_score = 2,
    Score mismatch_score = -1,
    Score gap_open_score = -2,
    Score gap_extend_score = -1) {
  return detail::visit_pair(pair, [&](auto query, auto target) {
    return detail::affine_path_typed<
        typename decltype(query)::value_type, true>(
            query, target, match_score, mismatch_score,
            gap_open_score, gap_extend_score);
  });
}

inline AlignmentPath needleman_wunsch_affine_path(
    const utf8::PreparedPair& pair,
    Score match_score = 2,
    Score mismatch_score = -1,
    Score gap_open_score = -2,
    Score gap_extend_score = -1) {
  return detail::visit_pair(pair, [&](auto query, auto target) {
    return detail::affine_path_typed<
        typename decltype(query)::value_type, false>(
            query, target, match_score, mismatch_score,
            gap_open_score, gap_extend_score);
  });
}

// Host-neutral substitution-matrix kernels.  The binding supplies an encoded
// token sequence and a score lookup; no Python/nanobind types cross this API.
template <bool Local, typename Lookup>
inline Score substitution_matrix_affine_score(
    std::span<const std::uint16_t> query,
    std::span<const std::uint16_t> target,
    Lookup&& lookup,
    Score gap_open_score,
    Score gap_extend_score) {
  const std::size_t columns = target.size() + 1U;
  constexpr Score negative_infinity =
      std::numeric_limits<Score>::lowest() / 4;
  const auto add = [](Score value, Score delta) {
    return value <= negative_infinity / 2 ? negative_infinity : value + delta;
  };
  const auto gap_cost = [&](std::size_t length) {
    return length == 0U
        ? Score{0}
        : gap_open_score + static_cast<Score>(length - 1U) * gap_extend_score;
  };
  std::vector<Score> previous(columns, Local ? Score{0} : negative_infinity);
  std::vector<Score> current(columns, Local ? Score{0} : negative_infinity);
  std::vector<Score> current_horizontal(columns, negative_infinity);
  std::vector<Score> previous_vertical(columns, negative_infinity);
  std::vector<Score> current_vertical(columns, negative_infinity);
  if constexpr (!Local) {
    previous[0] = 0;
    for (std::size_t column = 1; column < columns; ++column) {
      previous[column] = gap_cost(column);
    }
  }
  Score best = 0;
  for (std::size_t row = 1; row <= query.size(); ++row) {
    current[0] = Local ? 0 : gap_cost(row);
    current_vertical[0] = Local ? negative_infinity : current[0];
    current_horizontal[0] = negative_infinity;
    for (std::size_t column = 1; column < columns; ++column) {
      current_vertical[column] = std::max(
          add(previous[column], gap_open_score),
          add(previous_vertical[column], gap_extend_score));
      current_horizontal[column] = std::max(
          add(current[column - 1U], gap_open_score),
          add(current_horizontal[column - 1U], gap_extend_score));
      const Score diagonal = add(
          previous[column - 1U],
          static_cast<Score>(lookup(
              query[row - 1U], target[column - 1U])));
      Score value = std::max(
          {diagonal, current_vertical[column], current_horizontal[column]});
      if constexpr (Local) {
        value = std::max<Score>(0, value);
        best = std::max(best, value);
      }
      current[column] = value;
    }
    std::swap(previous, current);
    std::swap(previous_vertical, current_vertical);
  }
  if constexpr (Local) return best;
  return previous.back();
}

template <bool Local, typename Lookup>
inline AlignmentPath substitution_matrix_affine_path(
    std::span<const std::uint16_t> query,
    std::span<const std::uint16_t> target,
    Lookup&& lookup,
    Score gap_open_score,
    Score gap_extend_score) {
  const std::size_t rows = query.size() + 1U;
  const std::size_t columns = target.size() + 1U;
  if (rows != 0U && columns > std::numeric_limits<std::size_t>::max() / rows) {
    throw std::length_error("alignment traceback dimensions overflow");
  }
  constexpr Score negative_infinity =
      std::numeric_limits<Score>::lowest() / 4;
  const auto index = [columns](std::size_t row, std::size_t column) {
    return row * columns + column;
  };
  const auto add = [](Score value, Score delta) {
    return value <= negative_infinity / 2 ? negative_infinity : value + delta;
  };
  const auto gap_cost = [&](std::size_t length) {
    return length == 0U
        ? Score{0}
        : gap_open_score + static_cast<Score>(length - 1U) * gap_extend_score;
  };
  std::vector<Score> h(rows * columns, Local ? Score{0} : negative_infinity);
  std::vector<Score> vertical(rows * columns, negative_infinity);
  std::vector<Score> horizontal(rows * columns, negative_infinity);
  h[0] = 0;
  if constexpr (!Local) {
    for (std::size_t row = 1; row < rows; ++row) {
      h[index(row, 0)] = gap_cost(row);
      vertical[index(row, 0)] = h[index(row, 0)];
    }
    for (std::size_t column = 1; column < columns; ++column) {
      h[index(0, column)] = gap_cost(column);
      horizontal[index(0, column)] = h[index(0, column)];
    }
  }
  Score best_score = 0;
  std::size_t best_row = 0;
  std::size_t best_column = 0;
  for (std::size_t row = 1; row < rows; ++row) {
    for (std::size_t column = 1; column < columns; ++column) {
      const std::size_t cell = index(row, column);
      vertical[cell] = std::max(
          add(h[index(row - 1U, column)], gap_open_score),
          add(vertical[index(row - 1U, column)], gap_extend_score));
      horizontal[cell] = std::max(
          add(h[index(row, column - 1U)], gap_open_score),
          add(horizontal[index(row, column - 1U)], gap_extend_score));
      const Score diagonal = add(
          h[index(row - 1U, column - 1U)],
          static_cast<Score>(lookup(
              query[row - 1U], target[column - 1U])));
      Score value = std::max({diagonal, vertical[cell], horizontal[cell]});
      if constexpr (Local) value = std::max<Score>(0, value);
      h[cell] = value;
      if constexpr (Local) {
        if (value > best_score) {
          best_score = value;
          best_row = row;
          best_column = column;
        }
      }
    }
  }
  if constexpr (!Local) {
    best_row = query.size();
    best_column = target.size();
    best_score = h[index(best_row, best_column)];
  }
  const std::size_t query_end = best_row;
  const std::size_t target_end = best_column;
  detail::TraceState state = detail::TraceState::h;
  std::string operations;
  operations.reserve(best_row + best_column);
  while (best_row > 0U || best_column > 0U) {
    const std::size_t cell = index(best_row, best_column);
    if constexpr (Local) {
      if (state == detail::TraceState::h && h[cell] <= 0) break;
    }
    if (state == detail::TraceState::h) {
      if (best_row > 0U && best_column > 0U) {
        const Score diagonal = add(
            h[index(best_row - 1U, best_column - 1U)],
            static_cast<Score>(lookup(
                query[best_row - 1U], target[best_column - 1U])));
        if (h[cell] == diagonal) {
          operations.push_back(
              query[best_row - 1U] == target[best_column - 1U] ? '=' : 'X');
          --best_row;
          --best_column;
          continue;
        }
      }
      if (best_row > 0U && h[cell] == vertical[cell]) {
        state = detail::TraceState::vertical;
        continue;
      }
      if (best_column > 0U && h[cell] == horizontal[cell]) {
        state = detail::TraceState::horizontal;
        continue;
      }
      break;
    }
    if (state == detail::TraceState::vertical) {
      operations.push_back('D');
      const bool continues = best_row > 1U &&
          vertical[cell] == add(
              vertical[index(best_row - 1U, best_column)], gap_extend_score);
      --best_row;
      state = continues ? detail::TraceState::vertical : detail::TraceState::h;
    } else {
      operations.push_back('I');
      const bool continues = best_column > 1U &&
          horizontal[cell] == add(
              horizontal[index(best_row, best_column - 1U)], gap_extend_score);
      --best_column;
      state = continues ? detail::TraceState::horizontal : detail::TraceState::h;
    }
  }
  std::reverse(operations.begin(), operations.end());
  return make_alignment_path(
      best_score, best_row, query_end, best_column, target_end, operations);
}

}  // namespace stride_align::core
