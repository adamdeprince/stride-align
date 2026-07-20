#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "stride_align/alignment.hpp"

namespace stride_align::score_fast_paths {

template <typename Cell>
bool identity_fast_path_is_safe(Cell match_score, Cell mismatch_score, Cell gap_score) noexcept {
  return match_score >= 0 && mismatch_score <= match_score && gap_score <= 0;
}

template <typename Cell>
bool local_zero_fast_path_is_safe(Cell match_score, Cell mismatch_score, Cell gap_score) noexcept {
  return match_score > 0 && mismatch_score <= 0 && gap_score <= 0;
}

template <typename Token>
bool tokens_equal(std::span<const Token> query, std::span<const Token> target) {
  if (query.size() != target.size()) {
    return false;
  }
  if (query.empty()) {
    return true;
  }
  // Byte (and other 1-byte) tokens: one memcmp is cheaper/clearer than
  // a per-element std::equal loop for the identity fast path.
  if constexpr (sizeof(Token) == 1) {
    return std::memcmp(query.data(), target.data(), query.size()) == 0;
  } else {
    return std::equal(query.begin(), query.end(), target.begin());
  }
}

template <typename Token>
bool small_symbols_are_disjoint(std::span<const Token> query, std::span<const Token> target) {
  std::span<const Token> indexed = query;
  std::span<const Token> probed = target;
  if (target.size() < query.size()) {
    indexed = target;
    probed = query;
  }

  std::array<bool, 256> seen = {};
  for (const Token token : indexed) {
    const auto index = static_cast<std::uint64_t>(token);
    if (index > 255) {
      return false;
    }
    seen[static_cast<std::uint8_t>(index)] = true;
  }

  for (const Token token : probed) {
    const auto index = static_cast<std::uint64_t>(token);
    if (index > 255) {
      return false;
    }
    if (seen[static_cast<std::uint8_t>(index)]) {
      return false;
    }
  }

  return true;
}

template <typename Cell>
Score token_independent_global_score(
    std::size_t query_size,
    std::size_t target_size,
    Cell substitution_score,
    Cell gap_score) {
  const std::size_t diagonal_count =
      static_cast<Score>(substitution_score) >= 2 * static_cast<Score>(gap_score)
          ? std::min(query_size, target_size)
          : 0;
  return static_cast<Score>(diagonal_count) * static_cast<Score>(substitution_score) +
      static_cast<Score>(query_size + target_size - 2 * diagonal_count) *
          static_cast<Score>(gap_score);
}

template <typename Cell>
Score token_independent_local_score(
    std::size_t query_size,
    std::size_t target_size,
    Cell substitution_score,
    Cell gap_score) {
  if (query_size == 0 || target_size == 0) {
    return 0;
  }
  if (substitution_score <= 0 && gap_score <= 0) {
    return 0;
  }
  if (gap_score <= 0) {
    return substitution_score > 0
        ? static_cast<Score>(std::min(query_size, target_size)) *
              static_cast<Score>(substitution_score)
        : 0;
  }
  if (substitution_score <= 0) {
    return static_cast<Score>(query_size + target_size - 1) * static_cast<Score>(gap_score);
  }
  if (static_cast<Score>(substitution_score) >= 2 * static_cast<Score>(gap_score)) {
    const std::size_t diagonal_count = std::min(query_size, target_size);
    return static_cast<Score>(diagonal_count) * static_cast<Score>(substitution_score) +
        static_cast<Score>(query_size + target_size - 2 * diagonal_count) *
            static_cast<Score>(gap_score);
  }

  const Score gap_only =
      static_cast<Score>(query_size + target_size - 1) * static_cast<Score>(gap_score);
  const Score one_diagonal =
      static_cast<Score>(substitution_score) +
      static_cast<Score>(query_size + target_size - 2) * static_cast<Score>(gap_score);
  return std::max(gap_only, one_diagonal);
}

// Bit-parallel LCS length (Allison-Dix / Hyyrö). Byte alphabets use a
// flat 256 × word_count PEQ; wider tokens keep the hashmap path.
template <typename Token>
std::size_t lcs_length_bit_parallel(std::span<const Token> query, std::span<const Token> target) {
  if (query.empty() || target.empty()) {
    return 0;
  }

  const std::size_t word_count = (query.size() + 63U) / 64U;

  if constexpr (std::is_same_v<Token, std::uint8_t>) {
    // Flat PEQ + dirty-symbol clear via thread-local scratch.
    struct ByteLcsScratch {
      std::vector<std::uint64_t> peq;
      std::vector<std::uint64_t> row;
      std::vector<std::uint8_t> dirty;
      std::array<std::uint32_t, 256> touch{};
      std::uint32_t gen = 0;
      std::size_t layout_w = 0;
    };
    thread_local ByteLcsScratch scr;

    if (scr.peq.size() < 256U * word_count) {
      scr.peq.assign(256U * word_count, 0U);
      scr.dirty.clear();
      scr.layout_w = word_count;
      scr.touch.fill(0);
      scr.gen = 0;
    } else if (scr.layout_w != word_count) {
      std::fill_n(scr.peq.data(), 256U * word_count, std::uint64_t{0});
      scr.dirty.clear();
      scr.layout_w = word_count;
    } else {
      for (const std::uint8_t s : scr.dirty) {
        std::fill_n(
            scr.peq.data() + static_cast<std::size_t>(s) * word_count,
            word_count, std::uint64_t{0});
      }
      scr.dirty.clear();
    }
    if (++scr.gen == 0U) {
      scr.touch.fill(0);
      scr.gen = 1U;
    }

    for (std::size_t index = 0; index < query.size(); ++index) {
      const std::uint8_t c = query[index];
      if (scr.touch[c] != scr.gen) {
        scr.touch[c] = scr.gen;
        scr.dirty.push_back(c);
      }
      scr.peq[static_cast<std::size_t>(c) * word_count + (index / 64U)] |=
          std::uint64_t{1} << (index % 64U);
    }

    if (scr.row.size() < word_count) scr.row.resize(word_count);
    std::fill_n(scr.row.data(), word_count, std::uint64_t{0});

    for (const std::uint8_t token : target) {
      const std::uint64_t* mask_row =
          scr.peq.data() + static_cast<std::size_t>(token) * word_count;
      std::uint64_t shift_carry = 1;
      std::uint64_t borrow = 0;

      for (std::size_t word = 0; word < word_count; ++word) {
        const std::uint64_t previous = scr.row[word];
        const std::uint64_t mask = mask_row[word];
        const std::uint64_t x = mask | previous;
        const std::uint64_t y = (previous << 1U) | shift_carry;
        shift_carry = previous >> 63U;

        const std::uint64_t first_difference = x - y;
        const std::uint64_t first_borrow = x < y ? 1U : 0U;
        const std::uint64_t difference = first_difference - borrow;
        borrow = (first_borrow != 0 || first_difference < borrow) ? 1U : 0U;
        scr.row[word] = x & ~difference;
      }
    }

    std::size_t length = 0;
    for (std::size_t word = 0; word < word_count; ++word) {
      length += static_cast<std::size_t>(std::popcount(scr.row[word]));
    }
    return length;
  } else {
    std::unordered_map<Token, std::vector<std::uint64_t>> symbol_masks;
    symbol_masks.reserve(std::min<std::size_t>(query.size(), 256U));
    for (std::size_t index = 0; index < query.size(); ++index) {
      auto& mask = symbol_masks[query[index]];
      if (mask.empty()) {
        mask.assign(word_count, 0);
      }
      mask[index / 64U] |= std::uint64_t{1} << (index % 64U);
    }

    std::vector<std::uint64_t> row(word_count, 0);
    for (const Token token : target) {
      const auto mask_iterator = symbol_masks.find(token);
      std::uint64_t shift_carry = 1;
      std::uint64_t borrow = 0;

      for (std::size_t word = 0; word < word_count; ++word) {
        const std::uint64_t previous = row[word];
        const std::uint64_t mask =
            mask_iterator == symbol_masks.end() ? 0 : mask_iterator->second[word];
        const std::uint64_t x = mask | previous;
        const std::uint64_t y = (previous << 1U) | shift_carry;
        shift_carry = previous >> 63U;

        const std::uint64_t first_difference = x - y;
        const std::uint64_t first_borrow = x < y ? 1U : 0U;
        const std::uint64_t difference = first_difference - borrow;
        borrow = (first_borrow != 0 || first_difference < borrow) ? 1U : 0U;
        row[word] = x & ~difference;
      }
    }

    std::size_t length = 0;
    for (const auto word : row) {
      length += static_cast<std::size_t>(std::popcount(word));
    }
    return length;
  }
}

template <typename Token, typename Cell>
std::optional<Score> zero_gap_lcs_score(
    std::span<const Token> query,
    std::span<const Token> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  if (gap_score != 0 || match_score <= 0 || mismatch_score > 0 ||
      mismatch_score >= match_score) {
    return std::nullopt;
  }

  const auto length = lcs_length_bit_parallel(query, target);
  return static_cast<Score>(length) * static_cast<Score>(match_score);
}

template <typename Token, typename Cell, bool LocalAlignment>
std::optional<Score> fast_score_only(
    std::span<const Token> query,
    std::span<const Token> target,
    Cell match_score,
    Cell mismatch_score,
    Cell gap_score) {
  if (query.empty() || target.empty()) {
    if constexpr (LocalAlignment) {
      return 0;
    } else {
      return static_cast<Score>(query.size() + target.size()) * static_cast<Score>(gap_score);
    }
  }

  if constexpr (LocalAlignment) {
    if (match_score <= 0 && mismatch_score <= 0 && gap_score <= 0) {
      return 0;
    }
  }

  if (match_score == mismatch_score) {
    if constexpr (LocalAlignment) {
      return token_independent_local_score(
          query.size(),
          target.size(),
          match_score,
          gap_score);
    } else {
      return token_independent_global_score(
          query.size(),
          target.size(),
          match_score,
          gap_score);
    }
  }

  if ((!LocalAlignment || match_score > 0) &&
      identity_fast_path_is_safe(match_score, mismatch_score, gap_score) &&
      tokens_equal(query, target)) {
    return static_cast<Score>(query.size()) * static_cast<Score>(match_score);
  }

  if (const auto lcs_score =
          zero_gap_lcs_score(query, target, match_score, mismatch_score, gap_score);
      lcs_score.has_value()) {
    return *lcs_score;
  }

  if constexpr (LocalAlignment) {
    if (local_zero_fast_path_is_safe(match_score, mismatch_score, gap_score) &&
        small_symbols_are_disjoint(query, target)) {
      return 0;
    }
  }

  return std::nullopt;
}

}  // namespace stride_align::score_fast_paths
