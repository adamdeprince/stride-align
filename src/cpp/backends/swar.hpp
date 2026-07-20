#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <variant>

#include <nanobind/nanobind.h>

#include "affine.hpp"
#include "backends/affine_fixed_kernel.hpp"
#include "backends/farrar_fixed_kernel.hpp"
#include "backends/generic.hpp"
#include "backends/profile_traceback.hpp"
#include "dtw_dispatch.hpp"

namespace stride_align::backend_swar {

namespace nb = nanobind;
namespace generic_detail = stride_align::backend_generic::detail;

namespace detail {

using PackedWord = std::uint64_t;

template <typename Lane>
inline constexpr std::size_t lane_bits_v = sizeof(Lane) * 8U;

template <typename Lane>
inline constexpr PackedWord lane_mask_v =
    lane_bits_v<Lane> == 64U
        ? std::numeric_limits<PackedWord>::max()
        : ((PackedWord{1} << lane_bits_v<Lane>) - PackedWord{1});

// `std::bit_cast` is only in libstdc++ 11+ (the C++20 *language* feature
// `__builtin_bit_cast` ships earlier — gcc 10 has the builtin but not
// the stdlib wrapper). Both forms produce identical code; routing
// through this helper lets us build on gcc 10 toolchains while still
// using the standard name when it's available.
template <typename To, typename From>
constexpr To stride_align_bit_cast(const From& from) noexcept {
#if defined(__cpp_lib_bit_cast) && __cpp_lib_bit_cast >= 201806L
  return std::bit_cast<To>(from);
#else
  return __builtin_bit_cast(To, from);
#endif
}

template <typename Lane>
Lane unpack_lane(PackedWord packed, std::size_t lane) {
  using RawLane = std::make_unsigned_t<Lane>;
  const auto raw = static_cast<RawLane>(
      (packed >> (lane * lane_bits_v<Lane>)) & lane_mask_v<Lane>);
  return stride_align_bit_cast<Lane>(raw);
}

template <typename Lane>
PackedWord pack_lane(Lane value, std::size_t lane) {
  using RawLane = std::make_unsigned_t<Lane>;
  const auto raw =
      static_cast<PackedWord>(stride_align_bit_cast<RawLane>(value));
  return (raw & lane_mask_v<Lane>) << (lane * lane_bits_v<Lane>);
}

template <typename Lane, std::size_t LaneCount>
PackedWord pack_values(const Lane* values) {
  if constexpr (std::endian::native == std::endian::little) {
    PackedWord packed = 0;
    std::memcpy(&packed, values, sizeof(PackedWord));
    return packed;
  }

  PackedWord packed = 0;
  for (std::size_t lane = 0; lane < LaneCount; ++lane) {
    packed |= pack_lane(values[lane], lane);
  }
  return packed;
}

template <typename Lane, std::size_t LaneCount>
void unpack_values(PackedWord packed, Lane* values) {
  if constexpr (std::endian::native == std::endian::little) {
    std::memcpy(values, &packed, sizeof(PackedWord));
    return;
  }

  for (std::size_t lane = 0; lane < LaneCount; ++lane) {
    values[lane] = unpack_lane<Lane>(packed, lane);
  }
}

template <typename Token, typename Cell>
struct SimdOps {
  using vector_type = PackedWord;
  static_assert(std::is_unsigned_v<Token>);
  static_assert(std::is_signed_v<Cell>);
  static_assert(sizeof(Token) == sizeof(Cell));
  static_assert(
      sizeof(Cell) == 1 || sizeof(Cell) == 2 || sizeof(Cell) == 4,
      "SWAR intentionally supports only 8-, 16-, and 32-bit lanes");

  static constexpr std::size_t alignment = alignof(PackedWord);
  static constexpr std::size_t lane_count = sizeof(PackedWord) / sizeof(Cell);

  static consteval PackedWord set1_raw(PackedWord lane_value) {
    PackedWord packed = 0;
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
      packed |= (lane_value & lane_mask_v<Cell>) << (lane * lane_bits_v<Cell>);
    }
    return packed;
  }

  static constexpr PackedWord sign_bits = set1_raw(PackedWord{1} << (lane_bits_v<Cell> - 1U));
  static constexpr PackedWord value_bits = ~sign_bits;

  static PackedWord load_tokens(const Token* values) {
    return pack_values<Token, lane_count>(values);
  }

  static PackedWord load_cells(const Cell* values) {
    return pack_values<Cell, lane_count>(values);
  }

  static PackedWord load_aligned_cells(const Cell* values) {
    return load_cells(values);
  }

  static void store_cells(Cell* values, PackedWord packed) {
    unpack_values<Cell, lane_count>(packed, values);
  }

  static void store_aligned_cells(Cell* values, PackedWord packed) {
    store_cells(values, packed);
  }

  static PackedWord set1(Cell value) {
    PackedWord packed = 0;
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
      packed |= pack_lane(value, lane);
    }
    return packed;
  }

  static PackedWord zero() {
    return 0;
  }

  static PackedWord add(PackedWord lhs, PackedWord rhs) {
    return ((lhs & value_bits) + (rhs & value_bits)) ^ ((lhs ^ rhs) & sign_bits);
  }

  static PackedWord max(PackedWord lhs, PackedWord rhs) {
    PackedWord packed = 0;
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
      const Cell left = unpack_lane<Cell>(lhs, lane);
      const Cell right = unpack_lane<Cell>(rhs, lane);
      packed |= pack_lane(left > right ? left : right, lane);
    }
    return packed;
  }

  static PackedWord shift_left_zero(PackedWord packed) {
    return packed << lane_bits_v<Cell>;
  }

  static bool any_gt(PackedWord lhs, PackedWord rhs) {
    return any_nonzero(greater_mask(lhs, rhs));
  }

  static PackedWord greater_mask(PackedWord lhs, PackedWord rhs) {
    PackedWord packed = 0;
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
      const Cell left = unpack_lane<Cell>(lhs, lane);
      const Cell right = unpack_lane<Cell>(rhs, lane);
      packed |= left > right ? pack_lane(static_cast<Cell>(-1), lane) : PackedWord{0};
    }
    return packed;
  }

  static PackedWord bit_or(PackedWord lhs, PackedWord rhs) {
    return lhs | rhs;
  }

  static bool any_nonzero(PackedWord packed) {
    return packed != 0;
  }

  static Cell reduce_max(PackedWord packed) {
    Cell best = 0;
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
      best = std::max(best, unpack_lane<Cell>(packed, lane));
    }
    return best;
  }

  static PackedWord substitution(
      const Token* query,
      const Token* target,
      Cell match_score,
      Cell mismatch_score) {
    PackedWord packed = 0;
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
      packed |= pack_lane(query[lane] == target[lane] ? match_score : mismatch_score, lane);
    }
    return packed;
  }
};

using GenericImplementation = backend_generic::Implementation<BackendKind::swar>;
using GenericPreparedFarrarScore = GenericImplementation::PreparedSmithWatermanFarrarScore;

struct PreparedSmithWatermanFarrarScore {
  std::variant<
      farrar_fixed_kernel::detail::PreparedScoreState<std::int8_t>,
      farrar_fixed_kernel::detail::PreparedScoreState<std::int16_t>,
      farrar_fixed_kernel::detail::PreparedScoreState<std::int32_t>,
      GenericPreparedFarrarScore>
      state;
};

struct GenericPreparedAffineScore {
  PreparedFarrarAlignment prepared;
  Score match_score = 2;
  Score mismatch_score = -1;
  Score gap_open_score = -1;
  Score gap_extend_score = -1;
};

struct PreparedAffineScore {
  std::variant<
      farrar_fixed_kernel::detail::PreparedAffineScoreState<std::int8_t>,
      farrar_fixed_kernel::detail::PreparedAffineScoreState<std::int16_t>,
      farrar_fixed_kernel::detail::PreparedAffineScoreState<std::int32_t>,
      GenericPreparedAffineScore>
      state;
};

inline Score dispatch_farrar_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return farrar_fixed_kernel::detail::score<SimdOps, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits16:
      return farrar_fixed_kernel::detail::score<SimdOps, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits32:
      return farrar_fixed_kernel::detail::score<SimdOps, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits64:
      return generic_detail::dispatch_farrar_score(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported Farrar score width");
  throw nb::python_error();
}

inline PreparedSmithWatermanFarrarScore prepare_farrar_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  PreparedSmithWatermanFarrarScore output;
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      output.state =
          farrar_fixed_kernel::detail::prepare_score_state<SimdOps, std::int8_t>(
              prepared,
              match_score,
              mismatch_score,
              gap_score);
      return output;
    case KernelBits::bits16:
      output.state =
          farrar_fixed_kernel::detail::prepare_score_state<SimdOps, std::int16_t>(
              prepared,
              match_score,
              mismatch_score,
              gap_score);
      return output;
    case KernelBits::bits32:
      output.state =
          farrar_fixed_kernel::detail::prepare_score_state<SimdOps, std::int32_t>(
              prepared,
              match_score,
              mismatch_score,
              gap_score);
      return output;
    case KernelBits::bits64:
      output.state = GenericPreparedFarrarScore{prepared, match_score, mismatch_score, gap_score};
      return output;
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported Farrar score width");
  throw nb::python_error();
}

inline Score dispatch_prepared_farrar_score(PreparedSmithWatermanFarrarScore& prepared) {
  return std::visit(
      [](auto& state) -> Score {
        using State = std::decay_t<decltype(state)>;
        if constexpr (std::is_same_v<State, GenericPreparedFarrarScore>) {
          return generic_detail::dispatch_farrar_score(
              state.prepared,
              state.match_score,
              state.mismatch_score,
              state.gap_score);
        } else {
          using Cell = typename State::cell_type;
          return farrar_fixed_kernel::detail::score_state<SimdOps, Cell>(state);
        }
      },
      prepared.state);
}

inline PreparedAffineScore prepare_affine_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  PreparedAffineScore output;
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      output.state =
          farrar_fixed_kernel::detail::prepare_affine_score_state<SimdOps, std::int8_t>(
              prepared,
              match_score,
              mismatch_score,
              gap_open_score,
              gap_extend_score);
      return output;
    case KernelBits::bits16:
      output.state =
          farrar_fixed_kernel::detail::prepare_affine_score_state<SimdOps, std::int16_t>(
              prepared,
              match_score,
              mismatch_score,
              gap_open_score,
              gap_extend_score);
      return output;
    case KernelBits::bits32:
      output.state =
          farrar_fixed_kernel::detail::prepare_affine_score_state<SimdOps, std::int32_t>(
              prepared,
              match_score,
              mismatch_score,
              gap_open_score,
              gap_extend_score);
      return output;
    case KernelBits::bits64:
      output.state = GenericPreparedAffineScore{
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score};
      return output;
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported SWAR affine score width");
  throw nb::python_error();
}

inline Score dispatch_prepared_affine_score(PreparedAffineScore& prepared) {
  return std::visit(
      [](auto& state) -> Score {
        using State = std::decay_t<decltype(state)>;
        if constexpr (std::is_same_v<State, GenericPreparedAffineScore>) {
          const auto query = std::span<const std::uint8_t>(
              state.prepared.query_tokens.data(),
              state.prepared.query_tokens.size());
          const auto target = std::span<const std::uint8_t>(
              state.prepared.target_tokens.data(),
              state.prepared.target_tokens.size());
          return affine::detail::score_only<std::uint8_t, true>(
              query,
              target,
              state.match_score,
              state.mismatch_score,
              state.gap_open_score,
              state.gap_extend_score);
        } else {
          using Cell = typename State::cell_type;
          return farrar_fixed_kernel::detail::affine_score_state<SimdOps, Cell>(state);
        }
      },
      prepared.state);
}

inline Score dispatch_prepared_global_affine_score(PreparedAffineScore& prepared) {
  return std::visit(
      [](auto& state) -> Score {
        using State = std::decay_t<decltype(state)>;
        if constexpr (std::is_same_v<State, GenericPreparedAffineScore>) {
          const auto query = std::span<const std::uint8_t>(
              state.prepared.query_tokens.data(),
              state.prepared.query_tokens.size());
          const auto target = std::span<const std::uint8_t>(
              state.prepared.target_tokens.data(),
              state.prepared.target_tokens.size());
          return affine::detail::score_only<std::uint8_t, false>(
              query,
              target,
              state.match_score,
              state.mismatch_score,
              state.gap_open_score,
              state.gap_extend_score);
        } else {
          using Cell = typename State::cell_type;
          return farrar_fixed_kernel::detail::global_affine_score_state<SimdOps, Cell>(state);
        }
      },
      prepared.state);
}

template <bool LocalAlignment>
inline Score dispatch_affine_score(
    const PreparedAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  switch (prepared.kernel_bits) {
    case KernelBits::bits8: {
      const auto& query = std::get<std::vector<std::uint8_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint8_t>>(prepared.target_tokens);
      return affine_fixed_kernel::detail::run_kernel<
          SimdOps,
          std::uint8_t,
          std::int8_t,
          LocalAlignment,
          false>(
          std::span<const std::uint8_t>(query.data(), query.size()),
          std::span<const std::uint8_t>(target.data(), target.size()),
          static_cast<std::int8_t>(match_score),
          static_cast<std::int8_t>(mismatch_score),
          static_cast<std::int8_t>(gap_open_score),
          static_cast<std::int8_t>(gap_extend_score));
    }
    case KernelBits::bits16: {
      const auto& query = std::get<std::vector<std::uint16_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint16_t>>(prepared.target_tokens);
      return affine_fixed_kernel::detail::run_kernel<
          SimdOps,
          std::uint16_t,
          std::int16_t,
          LocalAlignment,
          false>(
          std::span<const std::uint16_t>(query.data(), query.size()),
          std::span<const std::uint16_t>(target.data(), target.size()),
          static_cast<std::int16_t>(match_score),
          static_cast<std::int16_t>(mismatch_score),
          static_cast<std::int16_t>(gap_open_score),
          static_cast<std::int16_t>(gap_extend_score));
    }
    case KernelBits::bits32: {
      const auto& query = std::get<std::vector<std::uint32_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint32_t>>(prepared.target_tokens);
      return affine_fixed_kernel::detail::run_kernel<
          SimdOps,
          std::uint32_t,
          std::int32_t,
          LocalAlignment,
          false>(
          std::span<const std::uint32_t>(query.data(), query.size()),
          std::span<const std::uint32_t>(target.data(), target.size()),
          static_cast<std::int32_t>(match_score),
          static_cast<std::int32_t>(mismatch_score),
          static_cast<std::int32_t>(gap_open_score),
          static_cast<std::int32_t>(gap_extend_score));
    }
    case KernelBits::bits64:
      return affine::detail::dispatch_score<LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported SWAR affine kernel width");
  throw nb::python_error();
}

template <bool LocalAlignment>
inline AlignmentResult dispatch_affine_traceback(
    const PreparedAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  switch (prepared.kernel_bits) {
    case KernelBits::bits8: {
      const auto& query = std::get<std::vector<std::uint8_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint8_t>>(prepared.target_tokens);
      return generic_detail::build_alignment_result<std::uint8_t>(
          prepared,
          affine_fixed_kernel::detail::run_kernel<
              SimdOps,
              std::uint8_t,
              std::int8_t,
              LocalAlignment,
              true>(
              std::span<const std::uint8_t>(query.data(), query.size()),
              std::span<const std::uint8_t>(target.data(), target.size()),
              static_cast<std::int8_t>(match_score),
              static_cast<std::int8_t>(mismatch_score),
              static_cast<std::int8_t>(gap_open_score),
              static_cast<std::int8_t>(gap_extend_score)),
          query,
          target);
    }
    case KernelBits::bits16: {
      const auto& query = std::get<std::vector<std::uint16_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint16_t>>(prepared.target_tokens);
      return generic_detail::build_alignment_result<std::uint16_t>(
          prepared,
          affine_fixed_kernel::detail::run_kernel<
              SimdOps,
              std::uint16_t,
              std::int16_t,
              LocalAlignment,
              true>(
              std::span<const std::uint16_t>(query.data(), query.size()),
              std::span<const std::uint16_t>(target.data(), target.size()),
              static_cast<std::int16_t>(match_score),
              static_cast<std::int16_t>(mismatch_score),
              static_cast<std::int16_t>(gap_open_score),
              static_cast<std::int16_t>(gap_extend_score)),
          query,
          target);
    }
    case KernelBits::bits32: {
      const auto& query = std::get<std::vector<std::uint32_t>>(prepared.query_tokens);
      const auto& target = std::get<std::vector<std::uint32_t>>(prepared.target_tokens);
      return generic_detail::build_alignment_result<std::uint32_t>(
          prepared,
          affine_fixed_kernel::detail::run_kernel<
              SimdOps,
              std::uint32_t,
              std::int32_t,
              LocalAlignment,
              true>(
              std::span<const std::uint32_t>(query.data(), query.size()),
              std::span<const std::uint32_t>(target.data(), target.size()),
              static_cast<std::int32_t>(match_score),
              static_cast<std::int32_t>(mismatch_score),
              static_cast<std::int32_t>(gap_open_score),
              static_cast<std::int32_t>(gap_extend_score)),
          query,
          target);
    }
    case KernelBits::bits64:
      return affine::detail::dispatch_traceback<LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported SWAR affine kernel width");
  throw nb::python_error();
}

inline Score dispatch_affine_farrar_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());

  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return farrar_fixed_kernel::detail::affine_score<SimdOps, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits16:
      return farrar_fixed_kernel::detail::affine_score<SimdOps, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits32:
      return farrar_fixed_kernel::detail::affine_score<SimdOps, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits64:
      return affine::detail::score_only<std::uint8_t, true>(
          query,
          target,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported SWAR affine Farrar score width");
  throw nb::python_error();
}

inline Score dispatch_global_affine_farrar_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());

  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return farrar_fixed_kernel::detail::global_affine_score<SimdOps, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits16:
      return farrar_fixed_kernel::detail::global_affine_score<SimdOps, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits32:
      return farrar_fixed_kernel::detail::global_affine_score<SimdOps, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits64:
      return affine::detail::score_only<std::uint8_t, false>(
          query,
          target,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported SWAR global affine Farrar score width");
  throw nb::python_error();
}

}  // namespace detail

struct Implementation {
  using PreparedSmithWatermanFarrarScore = detail::PreparedSmithWatermanFarrarScore;
  using PreparedAffineScore = detail::PreparedAffineScore;

  static Score smith_waterman_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    const auto prepared =
        prepare_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return generic_detail::dispatch_score<true>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static AlignmentResult smith_waterman_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return profile_traceback::linear_path<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static AlignmentPath smith_waterman_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return profile_traceback::linear_path_info<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
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
    const auto prepared =
        prepare_farrar_alignment(query, target, match_score, mismatch_score, gap_score, width);
    return detail::prepare_farrar_score(prepared, match_score, mismatch_score, gap_score);
  }

  static Score smith_waterman_farrar_score_prepared(
      PreparedSmithWatermanFarrarScore& prepared) {
    return detail::dispatch_prepared_farrar_score(prepared);
  }

  static Score smith_waterman_affine_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    if (gap_open_score <= 0 && gap_extend_score <= 0) {
      const auto prepared = prepare_farrar_alignment(
          query,
          target,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score,
          width);
      return detail::dispatch_affine_farrar_score(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    }
    return profile_traceback::affine_score<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static AlignmentResult smith_waterman_affine_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    return profile_traceback::affine_path<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static AlignmentPath smith_waterman_affine_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    return profile_traceback::affine_path_info<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static Score smith_waterman_affine_farrar_score(
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
    return detail::dispatch_affine_farrar_score(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static PreparedAffineScore prepare_smith_waterman_affine_score(
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
    return detail::prepare_affine_score(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static Score smith_waterman_affine_score_prepared(PreparedAffineScore& prepared) {
    return detail::dispatch_prepared_affine_score(prepared);
  }

  static PreparedAffineScore prepare_smith_waterman_affine_farrar_score(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    return prepare_smith_waterman_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static Score smith_waterman_affine_farrar_score_prepared(PreparedAffineScore& prepared) {
    return smith_waterman_affine_score_prepared(prepared);
  }

  static PreparedAffineScore prepare_needleman_wunsch_affine_score(
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
    return detail::prepare_affine_score(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static Score needleman_wunsch_affine_score_prepared(PreparedAffineScore& prepared) {
    return detail::dispatch_prepared_global_affine_score(prepared);
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
    return generic_detail::dispatch_score<false>(
        prepared,
        match_score,
        mismatch_score,
        gap_score);
  }

  static AlignmentResult needleman_wunsch_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return profile_traceback::linear_path<false>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static AlignmentPath needleman_wunsch_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_score,
      unsigned int width) {
    return profile_traceback::linear_path_info<false>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  }

  static Score needleman_wunsch_affine_score(
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
    return detail::dispatch_global_affine_farrar_score(
        prepared,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
  }

  static AlignmentResult needleman_wunsch_affine_path(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    return profile_traceback::affine_path<false>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  static AlignmentPath needleman_wunsch_affine_path_info(
      nb::handle query,
      nb::handle target,
      Score match_score,
      Score mismatch_score,
      Score gap_open_score,
      Score gap_extend_score,
      unsigned int width) {
    return profile_traceback::affine_path_info<false>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }


  static std::vector<double> dtw_distances(
      nb::handle query, nb::handle targets,
      nb::object window, nb::object distance,
      nb::object score_cutoff = nb::none()) {

    return ::stride_align::dtw::dispatch_dtw_many(query, targets, window, distance, score_cutoff);
  }

  static constexpr BackendKind backend_kind = BackendKind::swar;
};

}  // namespace stride_align::backend_swar
