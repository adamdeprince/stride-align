#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "stride_align/hamming.hpp"
#include "stride_align/indel.hpp"
#include "stride_align/jaro.hpp"
#include "stride_align/levenshtein.hpp"
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

}  // namespace stride_align::core
