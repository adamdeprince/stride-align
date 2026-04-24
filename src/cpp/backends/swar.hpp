#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <nanobind/nanobind.h>

#include "backends/generic.hpp"
#include "backends/wide_kernel.hpp"

namespace stride_align::backend_swar {

namespace nb = nanobind;
namespace generic_detail = stride_align::backend_generic::detail;

namespace detail {

using PackedWord = std::uint64_t;

template <typename Lane>
inline constexpr PackedWord lane_mask_v =
    sizeof(Lane) == sizeof(PackedWord)
        ? std::numeric_limits<PackedWord>::max()
        : ((PackedWord{1} << (sizeof(Lane) * 8U)) - PackedWord{1});

template <typename Lane>
Lane unpack_lane(PackedWord packed, std::size_t lane_index) {
  using Raw = std::make_unsigned_t<Lane>;
  constexpr std::size_t bit_width = sizeof(Lane) * 8U;
  const PackedWord raw_word = (packed >> (lane_index * bit_width)) & lane_mask_v<Lane>;
  return std::bit_cast<Lane>(static_cast<Raw>(raw_word));
}

template <typename Lane>
PackedWord pack_lane(Lane value, std::size_t lane_index) {
  using Raw = std::make_unsigned_t<Lane>;
  constexpr std::size_t bit_width = sizeof(Lane) * 8U;
  const auto raw = static_cast<PackedWord>(std::bit_cast<Raw>(value));
  return raw << (lane_index * bit_width);
}

template <typename Lane>
PackedWord pack_values(const Lane* values, std::size_t count) {
  PackedWord packed = 0;
  for (std::size_t lane = 0; lane < count; ++lane) {
    packed |= pack_lane<Lane>(values[lane], lane);
  }
  return packed;
}

template <typename Lane>
void unpack_values(PackedWord packed, Lane* values, std::size_t count) {
  for (std::size_t lane = 0; lane < count; ++lane) {
    values[lane] = unpack_lane<Lane>(packed, lane);
  }
}

template <typename Token, typename Cell>
struct SimdOps {
  using vector_type = PackedWord;
  static_assert(std::is_unsigned_v<Token>);
  static_assert(std::is_signed_v<Cell>);
  static_assert(sizeof(Token) == sizeof(Cell));
  static_assert(sizeof(Cell) == 1 || sizeof(Cell) == 2 || sizeof(Cell) == 4);

  static constexpr std::size_t alignment = alignof(PackedWord);
  static constexpr std::size_t lane_count = sizeof(PackedWord) / sizeof(Cell);
  static constexpr bool has_vector_max = true;

  static PackedWord load_tokens(const Token* values) {
    return pack_values<Token>(values, lane_count);
  }

  static PackedWord load_cells(const Cell* values) {
    return pack_values<Cell>(values, lane_count);
  }

  static void store_cells(Cell* values, PackedWord packed) {
    unpack_values<Cell>(packed, values, lane_count);
  }

  static PackedWord set1(Cell value) {
    PackedWord packed = 0;
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
      packed |= pack_lane<Cell>(value, lane);
    }
    return packed;
  }

  static PackedWord zero() {
    return 0;
  }

  static PackedWord add(PackedWord lhs, PackedWord rhs) {
    PackedWord packed = 0;
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
      const Cell sum = static_cast<Cell>(
          unpack_lane<Cell>(lhs, lane) + unpack_lane<Cell>(rhs, lane));
      packed |= pack_lane<Cell>(sum, lane);
    }
    return packed;
  }

  static PackedWord max(PackedWord lhs, PackedWord rhs) {
    PackedWord packed = 0;
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
      const Cell left = unpack_lane<Cell>(lhs, lane);
      const Cell right = unpack_lane<Cell>(rhs, lane);
      packed |= pack_lane<Cell>(left > right ? left : right, lane);
    }
    return packed;
  }

  static PackedWord substitution(
      const Token* query,
      const Token* target,
      Cell match_score,
      Cell mismatch_score) {
    PackedWord packed = 0;
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
      packed |= pack_lane<Cell>(query[lane] == target[lane] ? match_score : mismatch_score, lane);
    }
    return packed;
  }
};

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
      return wide_backend::detail::run_kernel<SimdOps, std::uint8_t, std::int8_t, LocalAlignment, false>(
          std::span<const std::uint8_t>(query.data(), query.size()),
          std::span<const std::uint8_t>(target.data(), target.size()),
          static_cast<std::int8_t>(match_score),
          static_cast<std::int8_t>(mismatch_score),
          static_cast<std::int8_t>(gap_score));
    }
    case KernelBits::bits16: {
      const auto& query = std::get<std::vector<std::uint16_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint16_t>>(prepared.target_tokens);
      return wide_backend::detail::run_kernel<SimdOps, std::uint16_t, std::int16_t, LocalAlignment, false>(
          std::span<const std::uint16_t>(query.data(), query.size()),
          std::span<const std::uint16_t>(target.data(), target.size()),
          static_cast<std::int16_t>(match_score),
          static_cast<std::int16_t>(mismatch_score),
          static_cast<std::int16_t>(gap_score));
    }
    case KernelBits::bits32: {
      const auto& query = std::get<std::vector<std::uint32_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint32_t>>(prepared.target_tokens);
      return wide_backend::detail::run_kernel<SimdOps, std::uint32_t, std::int32_t, LocalAlignment, false>(
          std::span<const std::uint32_t>(query.data(), query.size()),
          std::span<const std::uint32_t>(target.data(), target.size()),
          static_cast<std::int32_t>(match_score),
          static_cast<std::int32_t>(mismatch_score),
          static_cast<std::int32_t>(gap_score));
    }
    case KernelBits::bits64:
      return generic_detail::dispatch_score<LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
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
      return generic_detail::build_alignment_result<std::uint8_t>(
          prepared,
          wide_backend::detail::run_kernel<SimdOps, std::uint8_t, std::int8_t, LocalAlignment, true>(
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
      return generic_detail::build_alignment_result<std::uint16_t>(
          prepared,
          wide_backend::detail::run_kernel<SimdOps, std::uint16_t, std::int16_t, LocalAlignment, true>(
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
      return generic_detail::build_alignment_result<std::uint32_t>(
          prepared,
          wide_backend::detail::run_kernel<SimdOps, std::uint32_t, std::int32_t, LocalAlignment, true>(
              std::span<const std::uint32_t>(query.data(), query.size()),
              std::span<const std::uint32_t>(target.data(), target.size()),
              static_cast<std::int32_t>(match_score),
              static_cast<std::int32_t>(mismatch_score),
              static_cast<std::int32_t>(gap_score)),
          query,
          target);
    }
    case KernelBits::bits64:
      return generic_detail::dispatch_traceback<LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported kernel width");
  throw nb::python_error();
}

}  // namespace detail

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
};

}  // namespace stride_align::backend_swar
