#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <nanobind/nanobind.h>

#include "backends/score_fast_paths.hpp"
#include "farrar_preprocess.hpp"
#include "stride_align/alignment.hpp"

namespace stride_align::farrar_fixed_kernel {

namespace nb = nanobind;

namespace detail {

template <typename Cell>
struct ScoreToken;

template <>
struct ScoreToken<std::int8_t> {
  using type = std::uint8_t;
};

template <>
struct ScoreToken<std::int16_t> {
  using type = std::uint16_t;
};

template <>
struct ScoreToken<std::int32_t> {
  using type = std::uint32_t;
};

template <>
struct ScoreToken<std::int64_t> {
  using type = std::uint64_t;
};

template <template <typename, typename> class OpsTemplate, typename Cell>
using ScoreOps = OpsTemplate<typename ScoreToken<Cell>::type, Cell>;

template <typename Ops, typename = void>
struct UseDenseGlobalLazyFScan : std::false_type {};

template <typename Ops>
struct UseDenseGlobalLazyFScan<
    Ops,
    std::void_t<decltype(Ops::dense_global_lazy_f_scan)>>
    : std::bool_constant<Ops::dense_global_lazy_f_scan> {};

template <typename Ops, typename = void>
struct UseMaskedDenseGlobalLazyFScan : std::false_type {};

template <typename Ops>
struct UseMaskedDenseGlobalLazyFScan<
    Ops,
    std::void_t<decltype(Ops::masked_dense_global_lazy_f_scan)>>
    : std::bool_constant<Ops::masked_dense_global_lazy_f_scan> {};

template <typename Ops, typename = void>
struct UsePlainGlobalMainFAfterFirstSegment : std::false_type {};

template <typename Ops>
struct UsePlainGlobalMainFAfterFirstSegment<
    Ops,
    std::void_t<decltype(Ops::plain_global_main_f_after_first_segment)>>
    : std::bool_constant<Ops::plain_global_main_f_after_first_segment> {};

template <typename Ops, typename = void>
struct UseGlobalMainFSegment64Unroll : std::false_type {};

template <typename Ops>
struct UseGlobalMainFSegment64Unroll<
    Ops,
    std::void_t<decltype(Ops::global_main_f_segment64_unroll)>>
    : std::bool_constant<Ops::global_main_f_segment64_unroll> {};

template <typename Ops, typename = void>
struct UseGlobalMainFSegment32Unroll : std::false_type {};

template <typename Ops>
struct UseGlobalMainFSegment32Unroll<
    Ops,
    std::void_t<decltype(Ops::global_main_f_segment32_unroll)>>
    : std::bool_constant<Ops::global_main_f_segment32_unroll> {};

template <typename Ops, typename = void>
struct UseGlobalMainFSegment128Unroll : std::false_type {};

template <typename Ops>
struct UseGlobalMainFSegment128Unroll<
    Ops,
    std::void_t<decltype(Ops::global_main_f_segment128_unroll)>>
    : std::bool_constant<Ops::global_main_f_segment128_unroll> {};

template <typename T, std::size_t Alignment>
struct AlignedAllocator {
  using value_type = T;

  AlignedAllocator() noexcept = default;

  template <typename U>
  constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  [[nodiscard]] T* allocate(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }
    return static_cast<T*>(
        ::operator new(count * sizeof(T), std::align_val_t{Alignment}));
  }

  void deallocate(T* pointer, std::size_t) noexcept {
    ::operator delete(pointer, std::align_val_t{Alignment});
  }

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(
    const AlignedAllocator<T, Alignment>&,
    const AlignedAllocator<U, Alignment>&) noexcept {
  return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(
    const AlignedAllocator<T, Alignment>&,
    const AlignedAllocator<U, Alignment>&) noexcept {
  return false;
}

template <typename Cell>
using AlignedVector = std::vector<Cell, AlignedAllocator<Cell, 64>>;

inline constexpr std::uint16_t missing_profile_index =
    std::numeric_limits<std::uint16_t>::max();

enum class ScoreProfileLayout {
  automatic,
  token_major,
  target_ordered,
  blocked_target_ordered,
  compact_observed,
};

enum class ScoreKernelStrategy {
  automatic,
  materialized,
  deferred,
  deferred_unroll4,
  bounded,
  bounded_unroll4,
  compact,
};

template <typename Cell>
struct BuiltScoreProfile {
  AlignedVector<Cell> profile;
  std::vector<std::size_t> target_profile_offsets;
};

template <typename Cell>
struct PreparedScoreState {
  using cell_type = Cell;

  Cell gap_score = 0;
  std::optional<Score> fast_score;
  std::size_t query_size = 0;
  std::size_t target_size = 0;
  std::size_t segment_count = 0;
  ScoreKernelStrategy kernel_strategy = ScoreKernelStrategy::automatic;
  std::array<std::uint16_t, 256> profile_indices = {};
  std::vector<std::size_t> target_profile_offsets;
  AlignedVector<Cell> profile;
  AlignedVector<Cell> h_store;
  AlignedVector<Cell> h_load;
  AlignedVector<Cell> e_store;
};

template <typename Cell>
struct PreparedAffineScoreState {
  using cell_type = Cell;

  Cell gap_open_score = 0;
  Cell gap_extend_score = 0;
  std::size_t query_size = 0;
  std::size_t target_size = 0;
  std::size_t segment_count = 0;
  std::array<std::uint16_t, 256> profile_indices = {};
  std::vector<std::size_t> target_profile_offsets;
  AlignedVector<Cell> profile;
  AlignedVector<Cell> h_store;
  AlignedVector<Cell> h_load;
  AlignedVector<Cell> e_store;
  AlignedVector<Cell> global_h_initial;
  AlignedVector<Cell> global_e_initial;
};

template <typename Cell>
struct PreparedScoreBatchState {
  using cell_type = Cell;

  PreparedScoreState<Cell> state;
  std::vector<std::vector<std::size_t>> target_profile_offsets;
  std::vector<std::size_t> target_sizes;
  std::vector<std::optional<Score>> fast_scores;
};

template <typename Cell>
struct PreparedAffineScoreBatchState {
  using cell_type = Cell;

  PreparedAffineScoreState<Cell> state;
  std::vector<std::vector<std::size_t>> target_profile_offsets;
  std::vector<std::size_t> target_sizes;
};

template <template <typename, typename> class OpsTemplate>
struct PreparedScore {
  std::variant<
      PreparedScoreState<std::int8_t>,
      PreparedScoreState<std::int16_t>,
      PreparedScoreState<std::int32_t>,
      PreparedScoreState<std::int64_t>>
      state;
};

template <template <typename, typename> class OpsTemplate>
struct PreparedAffineScore {
  std::variant<
      PreparedAffineScoreState<std::int8_t>,
      PreparedAffineScoreState<std::int16_t>,
      PreparedAffineScoreState<std::int32_t>,
      PreparedAffineScoreState<std::int64_t>>
      state;
};

template <template <typename, typename> class OpsTemplate>
struct PreparedScoreBatch {
  std::variant<
      PreparedScoreBatchState<std::int8_t>,
      PreparedScoreBatchState<std::int16_t>,
      PreparedScoreBatchState<std::int32_t>,
      PreparedScoreBatchState<std::int64_t>>
      state;
};

template <template <typename, typename> class OpsTemplate>
struct PreparedAffineScoreBatch {
  std::variant<
      PreparedAffineScoreBatchState<std::int8_t>,
      PreparedAffineScoreBatchState<std::int16_t>,
      PreparedAffineScoreBatchState<std::int32_t>,
      PreparedAffineScoreBatchState<std::int64_t>>
      state;
};

enum class TraceDirection : std::uint8_t {
  stop = 0,
  diagonal = 1,
  up = 2,
  left = 3,
};

enum class TraceState : std::uint8_t {
  h,
  up,
  left,
};

inline constexpr std::uint8_t trace_direction_mask = 0x03U;
inline constexpr std::uint8_t trace_up_continue_bit = 0x04U;
inline constexpr std::uint8_t trace_left_continue_bit = 0x08U;

inline std::uint8_t pack_trace(
    TraceDirection direction,
    bool up_continues,
    bool left_continues) noexcept {
  return static_cast<std::uint8_t>(
      static_cast<std::uint8_t>(direction) |
      (up_continues ? trace_up_continue_bit : 0U) |
      (left_continues ? trace_left_continue_bit : 0U));
}

inline TraceDirection trace_direction(std::uint8_t value) noexcept {
  return static_cast<TraceDirection>(value & trace_direction_mask);
}

inline bool trace_up_continues(std::uint8_t value) noexcept {
  return (value & trace_up_continue_bit) != 0;
}

inline bool trace_left_continues(std::uint8_t value) noexcept {
  return (value & trace_left_continue_bit) != 0;
}

inline std::size_t trace_striped_index(
    std::size_t row,
    std::size_t column,
    std::size_t segment_count,
    std::size_t lane_count,
    std::size_t state_cell_count) noexcept {
  const std::size_t query_index = row - 1U;
  const std::size_t lane = query_index / segment_count;
  const std::size_t segment = query_index % segment_count;
  return column * state_cell_count + segment * lane_count + lane;
}

inline bool local_trace_best_is_better(
    Score candidate_score,
    std::size_t candidate_row,
    std::size_t candidate_column,
    Score best_score,
    std::size_t best_row,
    std::size_t best_column) noexcept {
  if (candidate_score > best_score) {
    return true;
  }
  return candidate_score > 0 && candidate_score == best_score &&
      (best_row == 0 || candidate_row < best_row ||
       (candidate_row == best_row && candidate_column < best_column));
}

template <typename Ops, typename Cell>
typename Ops::vector_type shift_left_zero(typename Ops::vector_type vector) {
  if constexpr (requires { Ops::shift_left_zero(vector); }) {
    return Ops::shift_left_zero(vector);
  }

  alignas(Ops::alignment) Cell input[Ops::lane_count] = {};
  alignas(Ops::alignment) Cell output[Ops::lane_count] = {};
  Ops::store_cells(input, vector);
  for (std::size_t lane = 1; lane < Ops::lane_count; ++lane) {
    output[lane] = input[lane - 1];
  }
  return Ops::load_cells(output);
}

template <typename Ops, typename Cell>
typename Ops::vector_type shift_left_insert(
    typename Ops::vector_type vector,
    Cell inserted) {
  if constexpr (requires { Ops::shift_left_insert(vector, inserted); }) {
    return Ops::shift_left_insert(vector, inserted);
  }

  alignas(Ops::alignment) Cell input[Ops::lane_count] = {};
  alignas(Ops::alignment) Cell output[Ops::lane_count] = {};
  Ops::store_cells(input, vector);
  output[0] = inserted;
  for (std::size_t lane = 1; lane < Ops::lane_count; ++lane) {
    output[lane] = input[lane - 1];
  }
  return Ops::load_cells(output);
}

template <typename Ops, typename Cell>
typename Ops::vector_type first_lane_vector(Cell first, Cell rest) {
  if constexpr (requires { Ops::shift_left_insert(Ops::set1(rest), first); }) {
    return Ops::shift_left_insert(Ops::set1(rest), first);
  }

  alignas(Ops::alignment) Cell values[Ops::lane_count] = {};
  std::fill(values, values + Ops::lane_count, rest);
  values[0] = first;
  return Ops::load_cells(values);
}

template <std::size_t LaneCount>
void shift_bool_lanes_left_zero(std::array<std::uint8_t, LaneCount>& values) {
  for (std::size_t lane = LaneCount - 1U; lane > 0; --lane) {
    values[lane] = values[lane - 1U];
  }
  values[0] = 0;
}

template <typename Ops, typename Cell>
typename Ops::vector_type load_state_cells(const Cell* values) {
  if constexpr (requires { Ops::load_aligned_cells(values); }) {
    return Ops::load_aligned_cells(values);
  }

  return Ops::load_cells(values);
}

template <typename Ops, typename Cell>
void store_state_cells(Cell* values, typename Ops::vector_type vector) {
  if constexpr (requires { Ops::store_aligned_cells(values, vector); }) {
    Ops::store_aligned_cells(values, vector);
  } else {
    Ops::store_cells(values, vector);
  }
}

template <typename Ops, typename Cell>
Cell reduce_max(typename Ops::vector_type vector) {
  if constexpr (requires { Ops::reduce_max(vector); }) {
    return Ops::reduce_max(vector);
  }

  alignas(Ops::alignment) Cell scores[Ops::lane_count] = {};
  Ops::store_cells(scores, vector);
  Cell best_score = 0;
  for (std::size_t lane = 0; lane < Ops::lane_count; ++lane) {
    best_score = std::max(best_score, scores[lane]);
  }
  return best_score;
}

template <typename Ops, typename Cell>
bool any_greater(typename Ops::vector_type lhs, typename Ops::vector_type rhs) {
  if constexpr (requires { Ops::any_gt(lhs, rhs); }) {
    return Ops::any_gt(lhs, rhs);
  }

  alignas(Ops::alignment) Cell left[Ops::lane_count] = {};
  alignas(Ops::alignment) Cell right[Ops::lane_count] = {};
  Ops::store_cells(left, lhs);
  Ops::store_cells(right, rhs);
  for (std::size_t lane = 0; lane < Ops::lane_count; ++lane) {
    if (left[lane] > right[lane]) {
      return true;
    }
  }
  return false;
}

template <typename Ops, typename Cell>
typename Ops::vector_type add_sentinel(
    typename Ops::vector_type lhs,
    typename Ops::vector_type rhs,
    Cell sentinel) {
  if constexpr (requires { Ops::add_sentinel(lhs, rhs, sentinel); }) {
    return Ops::add_sentinel(lhs, rhs, sentinel);
  }

  alignas(Ops::alignment) Cell left[Ops::lane_count] = {};
  alignas(Ops::alignment) Cell right[Ops::lane_count] = {};
  alignas(Ops::alignment) Cell output[Ops::lane_count] = {};
  Ops::store_cells(left, lhs);
  Ops::store_cells(right, rhs);
  for (std::size_t lane = 0; lane < Ops::lane_count; ++lane) {
    output[lane] = left[lane] == sentinel
        ? sentinel
        : static_cast<Cell>(
              static_cast<Score>(left[lane]) + static_cast<Score>(right[lane]));
  }
  return Ops::load_cells(output);
}

template <typename Ops, typename Cell, bool PreserveSentinel>
typename Ops::vector_type add_valid_or_sentinel(
    typename Ops::vector_type lhs,
    typename Ops::vector_type rhs,
    Cell sentinel) {
  if constexpr (PreserveSentinel) {
    return add_sentinel<Ops, Cell>(lhs, rhs, sentinel);
  } else {
    return Ops::add(lhs, rhs);
  }
}

inline std::vector<std::uint8_t> collect_profile_tokens(
    std::span<const std::uint8_t> target,
    std::array<std::uint16_t, 256>& profile_indices) {
  profile_indices.fill(missing_profile_index);
  std::vector<std::uint8_t> profile_tokens;
  profile_tokens.reserve(std::min<std::size_t>(target.size(), 256U));

  for (const auto token : target) {
    if (profile_indices[token] != missing_profile_index) {
      continue;
    }
    profile_indices[token] = static_cast<std::uint16_t>(profile_tokens.size());
    profile_tokens.push_back(token);
  }

  return profile_tokens;
}

inline std::vector<std::uint8_t> collect_profile_tokens(
    const std::vector<std::vector<std::uint8_t>>& targets,
    std::array<std::uint16_t, 256>& profile_indices) {
  profile_indices.fill(missing_profile_index);
  std::vector<std::uint8_t> profile_tokens;
  profile_tokens.reserve(256U);

  for (const auto& target : targets) {
    for (const auto token : target) {
      if (profile_indices[token] != missing_profile_index) {
        continue;
      }
      profile_indices[token] = static_cast<std::uint16_t>(profile_tokens.size());
      profile_tokens.push_back(token);
    }
  }

  return profile_tokens;
}

template <template <typename, typename> class OpsTemplate, typename Cell>
void fill_score_profile_row(
    Cell* lanes,
    std::span<const std::uint8_t> query,
    std::uint8_t token,
    Cell match_score,
    Cell mismatch_score,
    std::size_t segment,
    std::size_t segment_count) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;
  for (std::size_t lane = 0; lane < lane_count; ++lane) {
    const std::size_t query_index = lane * segment_count + segment;
    lanes[lane] = query_index < query.size() && query[query_index] == token ? match_score
                                                                             : mismatch_score;
  }
}

template <template <typename, typename> class OpsTemplate, typename Cell>
AlignedVector<Cell> build_profile(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> profile_tokens,
    Cell match_score,
    Cell mismatch_score,
    std::size_t segment_count) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;
  AlignedVector<Cell> profile(profile_tokens.size() * segment_count * lane_count, mismatch_score);

  for (std::size_t profile_index = 0; profile_index < profile_tokens.size(); ++profile_index) {
    const auto token = profile_tokens[profile_index];
    for (std::size_t segment = 0; segment < segment_count; ++segment) {
      Cell* lanes = profile.data() + ((profile_index * segment_count + segment) * lane_count);
      fill_score_profile_row<OpsTemplate, Cell>(
          lanes,
          query,
          token,
          match_score,
          mismatch_score,
          segment,
          segment_count);
    }
  }

  return profile;
}

template <template <typename, typename> class OpsTemplate, typename Cell>
AlignedVector<Cell> build_target_ordered_profile(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Cell match_score,
    Cell mismatch_score,
    std::size_t segment_count) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;
  AlignedVector<Cell> profile(target.size() * segment_count * lane_count, mismatch_score);

  for (std::size_t target_index = 0; target_index < target.size(); ++target_index) {
    const auto token = target[target_index];
    for (std::size_t segment = 0; segment < segment_count; ++segment) {
      Cell* lanes = profile.data() + ((target_index * segment_count + segment) * lane_count);
      fill_score_profile_row<OpsTemplate, Cell>(
          lanes,
          query,
          token,
          match_score,
          mismatch_score,
          segment,
          segment_count);
    }
  }

  return profile;
}

template <template <typename, typename> class OpsTemplate, typename Cell>
BuiltScoreProfile<Cell> build_blocked_target_ordered_profile(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Cell match_score,
    Cell mismatch_score,
    std::size_t segment_count,
    std::size_t block_size = 64U) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;
  const auto state_cells = segment_count * lane_count;
  BuiltScoreProfile<Cell> built;
  built.target_profile_offsets.reserve(target.size());
  if (target.empty() || state_cells == 0) {
    return built;
  }
  if (block_size == 0) {
    block_size = 64U;
  }

  std::array<std::uint16_t, 256> block_indices = {};
  std::vector<std::uint8_t> block_tokens;
  block_tokens.reserve(std::min<std::size_t>(block_size, 256U));

  for (std::size_t block_start = 0; block_start < target.size(); block_start += block_size) {
    const auto block_end = std::min(target.size(), block_start + block_size);
    block_indices.fill(missing_profile_index);
    block_tokens.clear();

    for (std::size_t target_index = block_start; target_index < block_end; ++target_index) {
      const auto token = target[target_index];
      if (block_indices[token] != missing_profile_index) {
        continue;
      }
      block_indices[token] = static_cast<std::uint16_t>(block_tokens.size());
      block_tokens.push_back(token);
    }

    const auto base_row = built.profile.size() / state_cells;
    built.profile.resize(built.profile.size() + block_tokens.size() * state_cells);
    for (std::size_t profile_index = 0; profile_index < block_tokens.size(); ++profile_index) {
      const auto token = block_tokens[profile_index];
      for (std::size_t segment = 0; segment < segment_count; ++segment) {
        Cell* lanes = built.profile.data() +
            ((base_row + profile_index) * segment_count + segment) * lane_count;
        fill_score_profile_row<OpsTemplate, Cell>(
            lanes,
            query,
            token,
            match_score,
            mismatch_score,
            segment,
            segment_count);
      }
    }

    for (std::size_t target_index = block_start; target_index < block_end; ++target_index) {
      built.target_profile_offsets.push_back(
          (base_row + static_cast<std::size_t>(block_indices[target[target_index]])) *
          state_cells);
    }
  }

  return built;
}

inline bool should_use_target_ordered_profile(
    std::size_t target_size,
    std::size_t profile_token_count) noexcept {
  return target_size >= 128U && profile_token_count * 4U >= target_size * 3U;
}

template <typename Ops>
ScoreProfileLayout resolve_score_profile_layout(
    ScoreProfileLayout requested_layout,
    std::size_t target_size,
    std::size_t profile_token_count) noexcept {
  if (requested_layout != ScoreProfileLayout::automatic) {
    return requested_layout;
  }

  if constexpr (requires { Ops::blocked_target_ordered_profile_min_rows; }) {
    if (profile_token_count >= Ops::blocked_target_ordered_profile_min_rows) {
      return ScoreProfileLayout::blocked_target_ordered;
    }
  }

  bool use_target_ordered_profile = false;
  if constexpr (requires { Ops::target_ordered_profile_high_cardinality; }) {
    use_target_ordered_profile = Ops::target_ordered_profile_high_cardinality &&
        should_use_target_ordered_profile(target_size, profile_token_count);
  }
  if constexpr (requires { Ops::target_ordered_profile_min_rows; }) {
    use_target_ordered_profile = use_target_ordered_profile ||
        profile_token_count >= Ops::target_ordered_profile_min_rows;
  }
  return use_target_ordered_profile ? ScoreProfileLayout::target_ordered
                                    : ScoreProfileLayout::token_major;
}

template <template <typename, typename> class OpsTemplate, typename Cell, bool UseFastPath = true>
PreparedScoreState<Cell> prepare_score_state(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score,
    ScoreProfileLayout profile_layout = ScoreProfileLayout::automatic,
    std::size_t profile_block_size = 64U,
    ScoreKernelStrategy kernel_strategy = ScoreKernelStrategy::automatic) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;

  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());

  const auto match = static_cast<Cell>(match_score);
  const auto mismatch = static_cast<Cell>(mismatch_score);
  PreparedScoreState<Cell> state;
  state.gap_score = static_cast<Cell>(gap_score);
  state.query_size = query.size();
  state.target_size = target.size();
  state.kernel_strategy = kernel_strategy;
  if constexpr (UseFastPath) {
    state.fast_score = score_fast_paths::fast_score_only<
        std::uint8_t,
        std::int64_t,
        true>(
        query,
        target,
        static_cast<std::int64_t>(match_score),
        static_cast<std::int64_t>(mismatch_score),
        static_cast<std::int64_t>(gap_score));
    if (state.fast_score.has_value()) {
      state.profile_indices.fill(missing_profile_index);
      return state;
    }
  }

  if (query.empty() || target.empty()) {
    state.profile_indices.fill(missing_profile_index);
    return state;
  }

  state.segment_count = (query.size() + lane_count - 1U) / lane_count;
  const auto profile_tokens = collect_profile_tokens(target, state.profile_indices);
  state.target_profile_offsets.reserve(target.size());
  const auto state_cells = state.segment_count * lane_count;
  std::size_t selected_profile_block_size = profile_block_size;
  if constexpr (requires { Ops::blocked_target_ordered_profile_block_size; }) {
    if (profile_layout == ScoreProfileLayout::automatic) {
      selected_profile_block_size = Ops::blocked_target_ordered_profile_block_size;
    }
  }
  switch (resolve_score_profile_layout<Ops>(
      profile_layout,
      target.size(),
      profile_tokens.size())) {
    case ScoreProfileLayout::target_ordered:
      state.profile = build_target_ordered_profile<OpsTemplate, Cell>(
          query,
          target,
          match,
          mismatch,
          state.segment_count);
      for (std::size_t target_index = 0; target_index < target.size(); ++target_index) {
        state.target_profile_offsets.push_back(target_index * state_cells);
      }
      break;
    case ScoreProfileLayout::blocked_target_ordered: {
      auto built = build_blocked_target_ordered_profile<OpsTemplate, Cell>(
          query,
          target,
          match,
          mismatch,
          state.segment_count,
          selected_profile_block_size);
      state.profile = std::move(built.profile);
      state.target_profile_offsets = std::move(built.target_profile_offsets);
      break;
    }
    case ScoreProfileLayout::automatic:
    case ScoreProfileLayout::token_major:
    case ScoreProfileLayout::compact_observed:
      state.profile = build_profile<OpsTemplate, Cell>(
          query,
          profile_tokens,
          match,
          mismatch,
          state.segment_count);
      for (const auto token : target) {
        state.target_profile_offsets.push_back(
            static_cast<std::size_t>(state.profile_indices[token]) * state_cells);
      }
      break;
  }
  state.h_store.resize(state_cells);
  state.h_load.resize(state_cells);
  state.e_store.resize(state_cells);
  return state;
}

template <typename Cell>
void initialize_global_column_zero(
    AlignedVector<Cell>& h_store,
    std::size_t query_size,
    std::size_t segment_count,
    std::size_t lane_count,
    Cell gap_score) {
  const Cell low_score = std::numeric_limits<Cell>::lowest();
  for (std::size_t segment = 0; segment < segment_count; ++segment) {
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
      const std::size_t query_index = lane * segment_count + segment;
      Cell score = low_score;
      if (query_index < query_size) {
        score = static_cast<Cell>(
            static_cast<Score>(query_index + 1U) * static_cast<Score>(gap_score));
      }
      h_store[segment * lane_count + lane] = score;
    }
  }
}

template <typename Cell>
Cell striped_row_value(
    const Cell* values,
    std::size_t row,
    std::size_t segment_count,
    std::size_t lane_count) {
  const std::size_t query_index = row - 1U;
  const std::size_t lane = query_index / segment_count;
  const std::size_t segment = query_index % segment_count;
  return values[segment * lane_count + lane];
}

template <typename Cell>
Cell affine_gap_cost(
    std::size_t length,
    Cell gap_open_score,
    Cell gap_extend_score) noexcept {
  if (length == 0) {
    return 0;
  }
  return static_cast<Cell>(
      static_cast<Score>(gap_open_score) +
      static_cast<Score>(length - 1U) * static_cast<Score>(gap_extend_score));
}

template <typename Cell>
void initialize_global_affine_column_zero(
    AlignedVector<Cell>& h_store,
    AlignedVector<Cell>& e_store,
    std::size_t query_size,
    std::size_t segment_count,
    std::size_t lane_count,
    Cell gap_open_score,
    Cell gap_extend_score) {
  const Cell low_score = std::numeric_limits<Cell>::lowest();
  for (std::size_t segment = 0; segment < segment_count; ++segment) {
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
      const std::size_t query_index = lane * segment_count + segment;
      Cell h_score = low_score;
      Cell e_score = low_score;
      if (query_index < query_size) {
        h_score = affine_gap_cost<Cell>(query_index + 1U, gap_open_score, gap_extend_score);
        e_score = static_cast<Cell>(
            static_cast<Score>(h_score) + static_cast<Score>(gap_open_score));
      }
      h_store[segment * lane_count + lane] = h_score;
      e_store[segment * lane_count + lane] = e_score;
    }
  }
}

template <typename Cell>
void prepare_global_affine_initial_column(
    PreparedAffineScoreState<Cell>& state,
    std::size_t lane_count) {
  const auto state_cells = state.segment_count * lane_count;
  state.global_h_initial.resize(state_cells);
  state.global_e_initial.resize(state_cells);
  initialize_global_affine_column_zero(
      state.global_h_initial,
      state.global_e_initial,
      state.query_size,
      state.segment_count,
      lane_count,
      state.gap_open_score,
      state.gap_extend_score);
}

template <typename Cell>
void reset_global_affine_initial_column(
    PreparedAffineScoreState<Cell>& state,
    std::size_t lane_count) {
  const auto state_cells = state.segment_count * lane_count;
  if (state.h_store.size() != state_cells) {
    state.h_store.resize(state_cells);
  }
  if (state.e_store.size() != state_cells) {
    state.e_store.resize(state_cells);
  }
  if (state.global_h_initial.size() == state_cells &&
      state.global_e_initial.size() == state_cells) {
    std::copy(
        state.global_h_initial.begin(),
        state.global_h_initial.end(),
        state.h_store.begin());
    std::copy(
        state.global_e_initial.begin(),
        state.global_e_initial.end(),
        state.e_store.begin());
    return;
  }

  initialize_global_affine_column_zero(
      state.h_store,
      state.e_store,
      state.query_size,
      state.segment_count,
      lane_count,
      state.gap_open_score,
      state.gap_extend_score);
}

template <template <typename, typename> class OpsTemplate, typename Cell>
PreparedScoreState<Cell> prepare_global_score_state(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;

  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());

  const auto match = static_cast<Cell>(match_score);
  const auto mismatch = static_cast<Cell>(mismatch_score);
  PreparedScoreState<Cell> state;
  state.gap_score = static_cast<Cell>(gap_score);
  state.query_size = query.size();
  state.target_size = target.size();
  state.fast_score = score_fast_paths::fast_score_only<
      std::uint8_t,
      std::int64_t,
      false>(
      query,
      target,
      static_cast<std::int64_t>(match_score),
      static_cast<std::int64_t>(mismatch_score),
      static_cast<std::int64_t>(gap_score));
  if (state.fast_score.has_value()) {
    state.profile_indices.fill(missing_profile_index);
    return state;
  }

  if (query.empty() || target.empty()) {
    state.profile_indices.fill(missing_profile_index);
    return state;
  }

  state.segment_count = (query.size() + lane_count - 1U) / lane_count;
  const auto profile_tokens = collect_profile_tokens(target, state.profile_indices);
  state.profile =
      build_profile<OpsTemplate, Cell>(query, profile_tokens, match, mismatch, state.segment_count);
  state.target_profile_offsets.reserve(target.size());
  for (const auto token : target) {
    state.target_profile_offsets.push_back(
        static_cast<std::size_t>(state.profile_indices[token]) * state.segment_count * lane_count);
  }
  const auto state_cells = state.segment_count * lane_count;
  state.h_store.resize(state_cells);
  state.h_load.resize(state_cells);
  return state;
}

template <
    template <typename, typename> class OpsTemplate,
    typename Cell,
    bool PrepareGlobalInitial = false>
PreparedAffineScoreState<Cell> prepare_affine_score_state(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;

  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());

  PreparedAffineScoreState<Cell> state;
  state.gap_open_score = static_cast<Cell>(gap_open_score);
  state.gap_extend_score = static_cast<Cell>(gap_extend_score);
  state.query_size = query.size();
  state.target_size = target.size();
  if (query.empty() || target.empty()) {
    state.profile_indices.fill(missing_profile_index);
    return state;
  }

  state.segment_count = (query.size() + lane_count - 1U) / lane_count;
  const auto profile_tokens = collect_profile_tokens(target, state.profile_indices);
  state.profile = build_profile<OpsTemplate, Cell>(
      query,
      profile_tokens,
      static_cast<Cell>(match_score),
      static_cast<Cell>(mismatch_score),
      state.segment_count);
  state.target_profile_offsets.reserve(target.size());
  for (const auto token : target) {
    state.target_profile_offsets.push_back(
        static_cast<std::size_t>(state.profile_indices[token]) * state.segment_count * lane_count);
  }

  const auto state_cells = state.segment_count * lane_count;
  state.h_store.resize(state_cells);
  state.h_load.resize(state_cells);
  state.e_store.resize(state_cells);
  if constexpr (PrepareGlobalInitial) {
    prepare_global_affine_initial_column(state, lane_count);
  }
  return state;
}

template <template <typename, typename> class OpsTemplate, typename Cell, bool LocalAlignment>
PreparedScoreBatchState<Cell> prepare_score_batch_state(
    const PreparedFarrarBatchAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score,
    ScoreProfileLayout profile_layout = ScoreProfileLayout::automatic,
    std::size_t profile_block_size = 64U,
    ScoreKernelStrategy kernel_strategy = ScoreKernelStrategy::automatic) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;

  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());

  PreparedScoreBatchState<Cell> batch;
  auto& state = batch.state;
  state.gap_score = static_cast<Cell>(gap_score);
  state.query_size = query.size();
  state.kernel_strategy = kernel_strategy;
  state.segment_count = query.empty() ? 0 : (query.size() + lane_count - 1U) / lane_count;

  batch.target_sizes.reserve(prepared.target_tokens.size());
  batch.target_profile_offsets.reserve(prepared.target_tokens.size());
  batch.fast_scores.reserve(prepared.target_tokens.size());
  for (const auto& target : prepared.target_tokens) {
    const auto target_span = std::span<const std::uint8_t>(target.data(), target.size());
    batch.target_sizes.push_back(target.size());
    batch.fast_scores.push_back(score_fast_paths::fast_score_only<
        std::uint8_t,
        std::int64_t,
        LocalAlignment>(
        query,
        target_span,
        static_cast<std::int64_t>(match_score),
        static_cast<std::int64_t>(mismatch_score),
        static_cast<std::int64_t>(gap_score)));
  }

  if (query.empty() || prepared.target_tokens.empty()) {
    state.profile_indices.fill(missing_profile_index);
    return batch;
  }

  const auto profile_tokens = collect_profile_tokens(prepared.target_tokens, state.profile_indices);
  const auto state_cells = state.segment_count * lane_count;
  std::size_t total_target_size = 0;
  for (const auto& target : prepared.target_tokens) {
    total_target_size += target.size();
  }
  const auto requested_profile_layout =
      profile_layout == ScoreProfileLayout::automatic ? ScoreProfileLayout::compact_observed
                                                      : profile_layout;
  const auto resolved_profile_layout = resolve_score_profile_layout<Ops>(
      requested_profile_layout,
      total_target_size,
      profile_tokens.size());
  std::size_t selected_profile_block_size = profile_block_size;
  if constexpr (requires { Ops::blocked_target_ordered_profile_block_size; }) {
    if (profile_layout == ScoreProfileLayout::automatic) {
      selected_profile_block_size = Ops::blocked_target_ordered_profile_block_size;
    }
  }

  switch (resolved_profile_layout) {
    case ScoreProfileLayout::target_ordered:
      for (const auto& target : prepared.target_tokens) {
        const auto target_span = std::span<const std::uint8_t>(target.data(), target.size());
        const auto base_row = state.profile.size() / state_cells;
        auto profile = build_target_ordered_profile<OpsTemplate, Cell>(
            query,
            target_span,
            static_cast<Cell>(match_score),
            static_cast<Cell>(mismatch_score),
            state.segment_count);
        state.profile.insert(state.profile.end(), profile.begin(), profile.end());
        auto& offsets = batch.target_profile_offsets.emplace_back();
        offsets.reserve(target.size());
        for (std::size_t target_index = 0; target_index < target.size(); ++target_index) {
          offsets.push_back((base_row + target_index) * state_cells);
        }
      }
      break;
    case ScoreProfileLayout::blocked_target_ordered:
      for (const auto& target : prepared.target_tokens) {
        const auto target_span = std::span<const std::uint8_t>(target.data(), target.size());
        const auto base_row = state.profile.size() / state_cells;
        auto built = build_blocked_target_ordered_profile<OpsTemplate, Cell>(
            query,
            target_span,
            static_cast<Cell>(match_score),
            static_cast<Cell>(mismatch_score),
            state.segment_count,
            selected_profile_block_size);
        state.profile.insert(state.profile.end(), built.profile.begin(), built.profile.end());
        auto& offsets = batch.target_profile_offsets.emplace_back();
        offsets.reserve(built.target_profile_offsets.size());
        for (const auto offset : built.target_profile_offsets) {
          offsets.push_back(offset + base_row * state_cells);
        }
      }
      break;
    case ScoreProfileLayout::automatic:
    case ScoreProfileLayout::token_major:
    case ScoreProfileLayout::compact_observed:
      state.profile = build_profile<OpsTemplate, Cell>(
          query,
          profile_tokens,
          static_cast<Cell>(match_score),
          static_cast<Cell>(mismatch_score),
          state.segment_count);

      for (const auto& target : prepared.target_tokens) {
        auto& offsets = batch.target_profile_offsets.emplace_back();
        offsets.reserve(target.size());
        for (const auto token : target) {
          offsets.push_back(
              static_cast<std::size_t>(state.profile_indices[token]) * state_cells);
        }
      }
      break;
  }

  state.h_store.resize(state_cells);
  state.h_load.resize(state_cells);
  if constexpr (LocalAlignment) {
    state.e_store.resize(state_cells);
  }
  return batch;
}

template <
    template <typename, typename> class OpsTemplate,
    typename Cell,
    bool PrepareGlobalInitial = false>
PreparedAffineScoreBatchState<Cell> prepare_affine_score_batch_state(
    const PreparedFarrarBatchAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;

  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());

  PreparedAffineScoreBatchState<Cell> batch;
  auto& state = batch.state;
  state.gap_open_score = static_cast<Cell>(gap_open_score);
  state.gap_extend_score = static_cast<Cell>(gap_extend_score);
  state.query_size = query.size();
  state.segment_count = query.empty() ? 0 : (query.size() + lane_count - 1U) / lane_count;

  batch.target_sizes.reserve(prepared.target_tokens.size());
  batch.target_profile_offsets.reserve(prepared.target_tokens.size());
  for (const auto& target : prepared.target_tokens) {
    batch.target_sizes.push_back(target.size());
  }

  if (query.empty() || prepared.target_tokens.empty()) {
    state.profile_indices.fill(missing_profile_index);
    return batch;
  }

  const auto profile_tokens = collect_profile_tokens(prepared.target_tokens, state.profile_indices);
  state.profile = build_profile<OpsTemplate, Cell>(
      query,
      profile_tokens,
      static_cast<Cell>(match_score),
      static_cast<Cell>(mismatch_score),
      state.segment_count);

  for (const auto& target : prepared.target_tokens) {
    auto& offsets = batch.target_profile_offsets.emplace_back();
    offsets.reserve(target.size());
    for (const auto token : target) {
      offsets.push_back(
          static_cast<std::size_t>(state.profile_indices[token]) *
          state.segment_count *
          lane_count);
    }
  }

  const auto state_cells = state.segment_count * lane_count;
  state.h_store.resize(state_cells);
  state.h_load.resize(state_cells);
  state.e_store.resize(state_cells);
  if constexpr (PrepareGlobalInitial) {
    prepare_global_affine_initial_column(state, lane_count);
  }
  return batch;
}

template <typename Ops, typename Cell>
bool scan_lazy_f(
    Cell* h_store_data,
    Cell* e_store_data,
    std::size_t segment_count,
    typename Ops::vector_type& v_f,
    typename Ops::vector_type gap_open_vector,
    typename Ops::vector_type gap_extend_vector,
    typename Ops::vector_type& best_vector) {
  constexpr std::size_t lane_count = Ops::lane_count;

  bool propagated = false;
  for (std::size_t segment = 0; segment < segment_count; ++segment) {
    Cell* h_store_segment = h_store_data + segment * lane_count;
    Cell* e_segment = e_store_data + segment * lane_count;

    const auto v_h_previous = load_state_cells<Ops, Cell>(h_store_segment);
    const bool segment_propagated = any_greater<Ops, Cell>(v_f, v_h_previous);
    auto v_h = Ops::max(v_h_previous, v_f);
    propagated = propagated || segment_propagated;

    if (segment_propagated) {
      store_state_cells<Ops, Cell>(h_store_segment, v_h);
      best_vector = Ops::max(best_vector, v_h);

      const auto v_h_open = Ops::add(v_h, gap_open_vector);
      auto v_e = load_state_cells<Ops, Cell>(e_segment);
      v_e = Ops::max(v_e, v_h_open);
      store_state_cells<Ops, Cell>(e_segment, v_e);
      v_f = Ops::max(Ops::add(v_f, gap_extend_vector), v_h_open);
    } else {
      const auto v_h_open = Ops::add(v_h, gap_open_vector);
      v_f = Ops::max(Ops::add(v_f, gap_extend_vector), v_h_open);
    }
  }

  return propagated;
}

template <typename Ops, typename Cell, bool PreserveSentinel = true>
bool scan_global_lazy_f(
    Cell* h_store_data,
    Cell* e_store_data,
    std::size_t segment_count,
    typename Ops::vector_type& v_f,
    typename Ops::vector_type gap_open_vector,
    typename Ops::vector_type gap_extend_vector,
    Cell low_score) {
  constexpr std::size_t lane_count = Ops::lane_count;

  bool propagated = false;
  for (std::size_t segment = 0; segment < segment_count; ++segment) {
    Cell* h_store_segment = h_store_data + segment * lane_count;
    Cell* e_segment = e_store_data + segment * lane_count;

    const auto v_h_previous = load_state_cells<Ops, Cell>(h_store_segment);
    const bool segment_propagated = any_greater<Ops, Cell>(v_f, v_h_previous);
    auto v_h = Ops::max(v_h_previous, v_f);
    propagated = propagated || segment_propagated;

    const auto v_h_open =
        add_valid_or_sentinel<Ops, Cell, PreserveSentinel>(v_h, gap_open_vector, low_score);
    if (segment_propagated) {
      store_state_cells<Ops, Cell>(h_store_segment, v_h);
      auto v_e = load_state_cells<Ops, Cell>(e_segment);
      v_e = Ops::max(v_e, v_h_open);
      store_state_cells<Ops, Cell>(e_segment, v_e);
    }

    v_f = Ops::max(add_sentinel<Ops, Cell>(v_f, gap_extend_vector, low_score), v_h_open);
  }

  return propagated;
}

template <typename Ops, typename Cell>
void scan_global_lazy_f_no_padding_dense(
    Cell* h_store_data,
    Cell* e_store_data,
    std::size_t segment_count,
    typename Ops::vector_type& v_f,
    typename Ops::vector_type gap_open_vector,
    typename Ops::vector_type gap_extend_vector,
    Cell low_score) {
  constexpr std::size_t lane_count = Ops::lane_count;

  if (segment_count == 0) {
    return;
  }

  {
    Cell* h_store_segment = h_store_data;
    Cell* e_segment = e_store_data;
    const auto v_h_previous = load_state_cells<Ops, Cell>(h_store_segment);
    const auto v_h = Ops::max(v_h_previous, v_f);
    store_state_cells<Ops, Cell>(h_store_segment, v_h);

    const auto v_h_open = Ops::add(v_h, gap_open_vector);
    auto v_e = load_state_cells<Ops, Cell>(e_segment);
    v_e = Ops::max(v_e, v_h_open);
    store_state_cells<Ops, Cell>(e_segment, v_e);
    v_f = Ops::max(add_sentinel<Ops, Cell>(v_f, gap_extend_vector, low_score), v_h_open);
  }

  for (std::size_t segment = 1; segment < segment_count; ++segment) {
    Cell* h_store_segment = h_store_data + segment * lane_count;
    Cell* e_segment = e_store_data + segment * lane_count;

    const auto v_h_previous = load_state_cells<Ops, Cell>(h_store_segment);
    const auto v_h = Ops::max(v_h_previous, v_f);
    store_state_cells<Ops, Cell>(h_store_segment, v_h);

    const auto v_h_open = Ops::add(v_h, gap_open_vector);
    auto v_e = load_state_cells<Ops, Cell>(e_segment);
    v_e = Ops::max(v_e, v_h_open);
    store_state_cells<Ops, Cell>(e_segment, v_e);
    v_f = Ops::max(Ops::add(v_f, gap_extend_vector), v_h_open);
  }
}

template <typename Ops, typename Cell>
void scan_global_lazy_f_no_padding_dense_masked(
    Cell* h_store_data,
    Cell* e_store_data,
    std::size_t segment_count,
    typename Ops::vector_type& v_f,
    typename Ops::vector_type gap_open_vector,
    typename Ops::vector_type gap_extend_vector,
    Cell low_score) {
  constexpr std::size_t lane_count = Ops::lane_count;

  if (segment_count == 0) {
    return;
  }

  {
    Cell* h_store_segment = h_store_data;
    Cell* e_segment = e_store_data;
    const auto v_h_previous = load_state_cells<Ops, Cell>(h_store_segment);
    const auto h_store_mask = Ops::greater_mask(v_f, v_h_previous);
    const auto v_h = Ops::max(v_h_previous, v_f);
    Ops::store_masked_cells(h_store_segment, h_store_mask, v_h);

    const auto v_h_open = Ops::add(v_h, gap_open_vector);
    auto v_e = load_state_cells<Ops, Cell>(e_segment);
    const auto e_store_mask = Ops::greater_mask(v_h_open, v_e);
    v_e = Ops::max(v_e, v_h_open);
    Ops::store_masked_cells(e_segment, e_store_mask, v_e);
    v_f = Ops::max(add_sentinel<Ops, Cell>(v_f, gap_extend_vector, low_score), v_h_open);
  }

  for (std::size_t segment = 1; segment < segment_count; ++segment) {
    Cell* h_store_segment = h_store_data + segment * lane_count;
    Cell* e_segment = e_store_data + segment * lane_count;

    const auto v_h_previous = load_state_cells<Ops, Cell>(h_store_segment);
    const auto h_store_mask = Ops::greater_mask(v_f, v_h_previous);
    const auto v_h = Ops::max(v_h_previous, v_f);
    Ops::store_masked_cells(h_store_segment, h_store_mask, v_h);

    const auto v_h_open = Ops::add(v_h, gap_open_vector);
    auto v_e = load_state_cells<Ops, Cell>(e_segment);
    const auto e_store_mask = Ops::greater_mask(v_h_open, v_e);
    v_e = Ops::max(v_e, v_h_open);
    Ops::store_masked_cells(e_segment, e_store_mask, v_e);
    v_f = Ops::max(Ops::add(v_f, gap_extend_vector), v_h_open);
  }
}

template <typename Ops, typename Cell>
typename Ops::vector_type global_lazy_f_prefix_carry(
    typename Ops::vector_type final_f,
    std::size_t segment_count,
    Cell gap_extend_score,
    Cell low_score) {
  if constexpr (requires {
                  Ops::global_lazy_f_prefix_carry(
                      final_f,
                      segment_count,
                      gap_extend_score,
                      low_score);
                }) {
    return Ops::global_lazy_f_prefix_carry(
        final_f,
        segment_count,
        gap_extend_score,
        low_score);
  }

  alignas(Ops::alignment) Cell output[Ops::lane_count] = {};
  alignas(Ops::alignment) Cell final_scores[Ops::lane_count] = {};
  Ops::store_cells(final_scores, final_f);

  const Score lane_span_gap =
      static_cast<Score>(segment_count) * static_cast<Score>(gap_extend_score);
  Cell previous_lane_output = low_score;
  output[0] = low_score;
  for (std::size_t lane = 0; lane < Ops::lane_count - 1U; ++lane) {
    Cell lane_output = final_scores[lane];
    if (previous_lane_output != low_score) {
      const Cell continued = static_cast<Cell>(
          static_cast<Score>(previous_lane_output) + lane_span_gap);
      lane_output = std::max(lane_output, continued);
    }
    previous_lane_output = lane_output;
    output[lane + 1U] = previous_lane_output;
  }

  return Ops::load_cells(output);
}

template <typename Ops, typename Cell>
typename Ops::vector_type global_lazy_f_prefix_carry_no_padding(
    typename Ops::vector_type final_f,
    std::size_t segment_count,
    Cell gap_extend_score,
    Cell low_score) {
  if constexpr (requires {
                  Ops::global_lazy_f_prefix_carry_no_padding(
                      final_f,
                      segment_count,
                      gap_extend_score,
                      low_score);
                }) {
    return Ops::global_lazy_f_prefix_carry_no_padding(
        final_f,
        segment_count,
        gap_extend_score,
        low_score);
  }

  return global_lazy_f_prefix_carry<Ops, Cell>(
      final_f,
      segment_count,
      gap_extend_score,
      low_score);
}

template <typename Ops, typename Cell>
typename Ops::vector_type local_lazy_f_prefix_carry(
    typename Ops::vector_type final_f,
    std::size_t segment_count,
    Cell gap_extend_score) {
  if constexpr (requires {
                  Ops::local_lazy_f_prefix_carry(
                      final_f,
                      segment_count,
                      gap_extend_score);
                }) {
    return Ops::local_lazy_f_prefix_carry(
        final_f,
        segment_count,
        gap_extend_score);
  }

  alignas(Ops::alignment) Cell output[Ops::lane_count] = {};
  alignas(Ops::alignment) Cell final_scores[Ops::lane_count] = {};
  Ops::store_cells(final_scores, final_f);

  const Score lane_span_gap =
      static_cast<Score>(segment_count) * static_cast<Score>(gap_extend_score);
  Cell previous_lane_output = 0;
  output[0] = 0;
  for (std::size_t lane = 0; lane < Ops::lane_count - 1U; ++lane) {
    Cell lane_output = final_scores[lane];
    if (previous_lane_output > 0) {
      const Cell continued = static_cast<Cell>(
          static_cast<Score>(previous_lane_output) + lane_span_gap);
      lane_output = std::max(lane_output, continued);
    }
    previous_lane_output = std::max<Cell>(lane_output, 0);
    output[lane + 1U] = previous_lane_output;
  }

  return Ops::load_cells(output);
}

template <typename Ops, typename Cell>
void scan_local_linear_lazy_f_once(
    Cell* h_store_data,
    std::size_t segment_count,
    typename Ops::vector_type& v_f,
    typename Ops::vector_type gap_vector,
    typename Ops::vector_type& best_vector) {
  constexpr std::size_t lane_count = Ops::lane_count;

  for (std::size_t segment = 0; segment < segment_count; ++segment) {
    Cell* h_store_segment = h_store_data + segment * lane_count;
    auto v_h_segment = load_state_cells<Ops, Cell>(h_store_segment);
    v_h_segment = Ops::max(v_h_segment, v_f);
    store_state_cells<Ops, Cell>(h_store_segment, v_h_segment);
    best_vector = Ops::max(best_vector, v_h_segment);
    v_f = Ops::add(v_f, gap_vector);
  }
}

template <typename Ops, typename Cell>
void scan_local_linear_lazy_f_once_bounded(
    Cell* h_store_data,
    std::size_t segment_count,
    typename Ops::vector_type& v_f,
    typename Ops::vector_type gap_vector,
    typename Ops::vector_type& best_vector) {
  constexpr std::size_t lane_count = Ops::lane_count;

  for (std::size_t segment = 0; segment < segment_count; ++segment) {
    Cell* h_store_segment = h_store_data + segment * lane_count;
    auto v_h_segment = load_state_cells<Ops, Cell>(h_store_segment);
    v_h_segment = Ops::max(v_h_segment, v_f);
    store_state_cells<Ops, Cell>(h_store_segment, v_h_segment);
    best_vector = Ops::max(best_vector, v_h_segment);
    v_f = Ops::add(v_f, gap_vector);
    if (!any_greater<Ops, Cell>(v_f, Ops::add(v_h_segment, gap_vector))) {
      return;
    }
  }
}

template <typename Ops, typename Cell, std::size_t LaneCount, bool LocalAlignment>
typename Ops::vector_type trace_lazy_f_prefix_carry(
    typename Ops::vector_type final_f,
    std::array<std::uint8_t, LaneCount>& continue_lanes,
    std::size_t segment_count,
    Cell gap_extend_score,
    Cell low_score) {
  static_assert(LaneCount == Ops::lane_count);

  alignas(Ops::alignment) Cell output[LaneCount] = {};
  alignas(Ops::alignment) Cell final_scores[LaneCount] = {};
  Ops::store_cells(final_scores, final_f);

  std::array<std::uint8_t, LaneCount> output_continue = {};
  const Score lane_span_gap =
      static_cast<Score>(segment_count) * static_cast<Score>(gap_extend_score);

  Cell previous_lane_output = LocalAlignment ? Cell{0} : low_score;
  bool previous_lane_active = false;
  output[0] = LocalAlignment ? Cell{0} : low_score;

  for (std::size_t lane = 0; lane < LaneCount - 1U; ++lane) {
    Cell lane_output = final_scores[lane];
    std::uint8_t lane_continues = continue_lanes[lane];

    if (previous_lane_active) {
      const Cell continued = static_cast<Cell>(
          static_cast<Score>(previous_lane_output) + lane_span_gap);
      if (continued >= lane_output) {
        lane_output = continued;
        lane_continues = 1U;
      }
    }

    if constexpr (LocalAlignment) {
      if (lane_output > 0) {
        previous_lane_output = lane_output;
        previous_lane_active = true;
        output[lane + 1U] = lane_output;
        output_continue[lane + 1U] = lane_continues;
      } else {
        previous_lane_output = 0;
        previous_lane_active = false;
        output[lane + 1U] = 0;
        output_continue[lane + 1U] = 0;
      }
    } else {
      previous_lane_output = lane_output;
      previous_lane_active = lane_output != low_score;
      output[lane + 1U] = lane_output;
      output_continue[lane + 1U] = lane_continues;
    }
  }

  continue_lanes = output_continue;
  return Ops::load_cells(output);
}

template <template <typename, typename> class OpsTemplate, typename Cell>
Score affine_score_state_for_offsets(
    PreparedAffineScoreState<Cell>& state,
    std::span<const std::size_t> target_profile_offsets) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;

  if (state.segment_count == 0 || target_profile_offsets.empty()) {
    return 0;
  }
  if constexpr (requires { Ops::local_affine_score_exact_segment128_raw(state, target_profile_offsets); }) {
    if (state.query_size == 1024U && state.segment_count == 128U &&
        state.gap_open_score <= state.gap_extend_score && state.gap_extend_score <= 0) {
      return Ops::local_affine_score_exact_segment128_raw(state, target_profile_offsets);
    }
  }
  std::fill(state.h_store.begin(), state.h_store.end(), Cell{0});
  std::fill(state.h_load.begin(), state.h_load.end(), Cell{0});
  std::fill(state.e_store.begin(), state.e_store.end(), Cell{0});

  const auto zero_vector = Ops::zero();
  const auto gap_open_vector = Ops::set1(state.gap_open_score);
  const auto gap_extend_vector = Ops::set1(state.gap_extend_score);
  const bool can_stop_lazy_f = state.gap_open_score <= 0 && state.gap_extend_score <= 0;
  const bool can_prefix_lazy_f =
      state.gap_open_score <= state.gap_extend_score && state.gap_extend_score <= 0;
  auto best_vector = zero_vector;
  Cell* h_store_data = state.h_store.data();
  Cell* h_load_data = state.h_load.data();
  Cell* e_store_data = state.e_store.data();
  const Cell* profile_data = state.profile.data();

  for (const auto profile_offset : target_profile_offsets) {
    std::swap(h_store_data, h_load_data);

    auto v_h = shift_left_zero<Ops, Cell>(
        load_state_cells<Ops, Cell>(h_load_data + ((state.segment_count - 1U) * lane_count)));
    auto v_f = zero_vector;
    const Cell* profile_row = profile_data + profile_offset;

    for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
      Cell* h_store_segment = h_store_data + segment * lane_count;
      Cell* h_load_segment = h_load_data + segment * lane_count;
      Cell* e_segment = e_store_data + segment * lane_count;
      const Cell* profile_segment = profile_row + segment * lane_count;

      const auto v_profile = load_state_cells<Ops, Cell>(profile_segment);
      auto v_e = load_state_cells<Ops, Cell>(e_segment);
      v_h = Ops::add(v_h, v_profile);
      v_h = Ops::max(v_h, v_e);
      v_h = Ops::max(v_h, v_f);
      v_h = Ops::max(v_h, zero_vector);
      store_state_cells<Ops, Cell>(h_store_segment, v_h);
      best_vector = Ops::max(best_vector, v_h);

      const auto v_h_open = Ops::add(v_h, gap_open_vector);
      v_e = Ops::max(Ops::add(v_e, gap_extend_vector), v_h_open);
      store_state_cells<Ops, Cell>(e_segment, v_e);
      v_f = Ops::max(Ops::add(v_f, gap_extend_vector), v_h_open);
      v_h = load_state_cells<Ops, Cell>(h_load_segment);
    }

    if (can_prefix_lazy_f) {
      v_f = local_lazy_f_prefix_carry<Ops, Cell>(
          v_f,
          state.segment_count,
          state.gap_extend_score);
      if (any_greater<Ops, Cell>(v_f, zero_vector)) {
        scan_lazy_f<Ops, Cell>(
            h_store_data,
            e_store_data,
            state.segment_count,
            v_f,
            gap_open_vector,
            gap_extend_vector,
            best_vector);
      }
    } else {
      for (std::size_t iteration = 0; iteration < lane_count; ++iteration) {
        v_f = shift_left_zero<Ops, Cell>(v_f);
        const bool propagated = scan_lazy_f<Ops, Cell>(
          h_store_data,
          e_store_data,
          state.segment_count,
          v_f,
          gap_open_vector,
          gap_extend_vector,
          best_vector);

        (void) propagated;
        if (can_stop_lazy_f && !any_greater<Ops, Cell>(v_f, zero_vector)) {
          break;
        }
      }
    }
  }

  return static_cast<Score>(reduce_max<Ops, Cell>(best_vector));
}

template <template <typename, typename> class OpsTemplate, typename Cell>
Score affine_score_state(PreparedAffineScoreState<Cell>& state) {
  return affine_score_state_for_offsets<OpsTemplate, Cell>(
      state,
      std::span<const std::size_t>(
          state.target_profile_offsets.data(),
          state.target_profile_offsets.size()));
}

template <template <typename, typename> class OpsTemplate, typename Cell, bool PreserveSentinel>
Score global_affine_score_state_impl(PreparedAffineScoreState<Cell>& state) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;

  if (state.query_size == 0 && state.target_size == 0) {
    return 0;
  }
  if (state.query_size == 0) {
    return affine_gap_cost<Cell>(
        state.target_size,
        state.gap_open_score,
        state.gap_extend_score);
  }
  if (state.target_size == 0) {
    return affine_gap_cost<Cell>(
        state.query_size,
        state.gap_open_score,
        state.gap_extend_score);
  }

  reset_global_affine_initial_column(state, lane_count);
  std::fill(state.h_load.begin(), state.h_load.end(), std::numeric_limits<Cell>::lowest());

  const auto gap_open_vector = Ops::set1(state.gap_open_score);
  const auto gap_extend_vector = Ops::set1(state.gap_extend_score);
  const Cell low_score = std::numeric_limits<Cell>::lowest();
  Cell* h_store_data = state.h_store.data();
  Cell* h_load_data = state.h_load.data();
  Cell* e_store_data = state.e_store.data();
  const Cell* profile_data = state.profile.data();

  for (std::size_t target_index = 0; target_index < state.target_profile_offsets.size();
       ++target_index) {
    std::swap(h_store_data, h_load_data);

    const Cell top_left = affine_gap_cost<Cell>(
        target_index,
        state.gap_open_score,
        state.gap_extend_score);
    const Cell top_score = affine_gap_cost<Cell>(
        target_index + 1U,
        state.gap_open_score,
        state.gap_extend_score);
    const Cell first_f = static_cast<Cell>(
        static_cast<Score>(top_score) + static_cast<Score>(state.gap_open_score));
    auto v_h = shift_left_insert<Ops, Cell>(
        load_state_cells<Ops, Cell>(h_load_data + ((state.segment_count - 1U) * lane_count)),
        top_left);
    auto v_f = first_lane_vector<Ops, Cell>(first_f, low_score);
    const Cell* profile_row = profile_data + state.target_profile_offsets[target_index];

    for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
      Cell* h_store_segment = h_store_data + segment * lane_count;
      Cell* h_load_segment = h_load_data + segment * lane_count;
      Cell* e_segment = e_store_data + segment * lane_count;
      const Cell* profile_segment = profile_row + segment * lane_count;

      const auto v_profile = load_state_cells<Ops, Cell>(profile_segment);
      auto v_e = load_state_cells<Ops, Cell>(e_segment);
      v_h = add_valid_or_sentinel<Ops, Cell, PreserveSentinel>(
          v_h,
          v_profile,
          low_score);
      v_h = Ops::max(v_h, v_e);
      v_h = Ops::max(v_h, v_f);
      store_state_cells<Ops, Cell>(h_store_segment, v_h);

      const auto v_h_open = add_valid_or_sentinel<Ops, Cell, PreserveSentinel>(
          v_h,
          gap_open_vector,
          low_score);
      v_e = Ops::max(
          add_valid_or_sentinel<Ops, Cell, PreserveSentinel>(
              v_e,
              gap_extend_vector,
              low_score),
          v_h_open);
      store_state_cells<Ops, Cell>(e_segment, v_e);
      v_f = Ops::max(add_sentinel<Ops, Cell>(v_f, gap_extend_vector, low_score), v_h_open);
      v_h = load_state_cells<Ops, Cell>(h_load_segment);
    }

    const bool can_prefix_lazy_f =
        state.gap_open_score <= state.gap_extend_score && state.gap_extend_score <= 0;
    if (can_prefix_lazy_f) {
      if constexpr (PreserveSentinel) {
        v_f = global_lazy_f_prefix_carry<Ops, Cell>(
            v_f,
            state.segment_count,
            state.gap_extend_score,
            low_score);
      } else {
        v_f = global_lazy_f_prefix_carry_no_padding<Ops, Cell>(
            v_f,
            state.segment_count,
            state.gap_extend_score,
            low_score);
      }
      scan_global_lazy_f<Ops, Cell, PreserveSentinel>(
          h_store_data,
          e_store_data,
          state.segment_count,
          v_f,
          gap_open_vector,
          gap_extend_vector,
          low_score);
    } else {
      for (std::size_t iteration = 0; iteration < lane_count; ++iteration) {
        v_f = shift_left_insert<Ops, Cell>(v_f, low_score);
        const bool propagated = scan_global_lazy_f<Ops, Cell, PreserveSentinel>(
            h_store_data,
            e_store_data,
            state.segment_count,
            v_f,
            gap_open_vector,
            gap_extend_vector,
            low_score);

        (void) propagated;
      }
    }
  }

  return static_cast<Score>(
      striped_row_value(h_store_data, state.query_size, state.segment_count, lane_count));
}

template <template <typename, typename> class OpsTemplate, typename Cell>
Score global_affine_score_state_equal_length_no_padding(
    PreparedAffineScoreState<Cell>& state) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;

  reset_global_affine_initial_column(state, lane_count);
  std::fill(state.h_load.begin(), state.h_load.end(), std::numeric_limits<Cell>::lowest());

  const auto gap_open_vector = Ops::set1(state.gap_open_score);
  const auto gap_extend_vector = Ops::set1(state.gap_extend_score);
  const Cell low_score = std::numeric_limits<Cell>::lowest();
  const auto low_vector = Ops::set1(low_score);
  Cell* h_store_data = state.h_store.data();
  Cell* h_load_data = state.h_load.data();
  Cell* e_store_data = state.e_store.data();
  const Cell* profile_data = state.profile.data();
  const bool can_prefix_lazy_f =
      state.gap_open_score <= state.gap_extend_score && state.gap_extend_score <= 0;

  Cell top_left = 0;
  Cell next_top_left = state.gap_open_score;
  Cell first_f = static_cast<Cell>(
      static_cast<Score>(state.gap_open_score) +
      static_cast<Score>(state.gap_open_score));

  for (const auto profile_offset : state.target_profile_offsets) {
    std::swap(h_store_data, h_load_data);

    auto v_h = shift_left_insert<Ops, Cell>(
        load_state_cells<Ops, Cell>(h_load_data + ((state.segment_count - 1U) * lane_count)),
        top_left);
    auto v_f = shift_left_insert<Ops, Cell>(low_vector, first_f);
    const Cell* profile_row = profile_data + profile_offset;

    if constexpr (UsePlainGlobalMainFAfterFirstSegment<Ops>::value) {
      auto process_first_segment = [&]() {
        Cell* h_store_segment = h_store_data;
        Cell* h_load_segment = h_load_data;
        Cell* e_segment = e_store_data;
        const Cell* profile_segment = profile_row;

        const auto v_profile = load_state_cells<Ops, Cell>(profile_segment);
        auto v_e = load_state_cells<Ops, Cell>(e_segment);
        v_h = Ops::add(v_h, v_profile);
        v_h = Ops::max(v_h, v_e);
        v_h = Ops::max(v_h, v_f);
        store_state_cells<Ops, Cell>(h_store_segment, v_h);

        const auto v_h_open = Ops::add(v_h, gap_open_vector);
        v_e = Ops::max(Ops::add(v_e, gap_extend_vector), v_h_open);
        store_state_cells<Ops, Cell>(e_segment, v_e);
        v_f = Ops::max(add_sentinel<Ops, Cell>(v_f, gap_extend_vector, low_score), v_h_open);
        v_h = load_state_cells<Ops, Cell>(h_load_segment);
      };

      auto process_plain_segment = [&](std::size_t segment) {
        Cell* h_store_segment = h_store_data + segment * lane_count;
        Cell* h_load_segment = h_load_data + segment * lane_count;
        Cell* e_segment = e_store_data + segment * lane_count;
        const Cell* profile_segment = profile_row + segment * lane_count;

        const auto v_profile = load_state_cells<Ops, Cell>(profile_segment);
        auto v_e = load_state_cells<Ops, Cell>(e_segment);
        v_h = Ops::add(v_h, v_profile);
        v_h = Ops::max(v_h, v_e);
        v_h = Ops::max(v_h, v_f);
        store_state_cells<Ops, Cell>(h_store_segment, v_h);

        const auto v_h_open = Ops::add(v_h, gap_open_vector);
        v_e = Ops::max(Ops::add(v_e, gap_extend_vector), v_h_open);
        store_state_cells<Ops, Cell>(e_segment, v_e);
        v_f = Ops::max(Ops::add(v_f, gap_extend_vector), v_h_open);
        v_h = load_state_cells<Ops, Cell>(h_load_segment);
      };

      auto process_plain_tail = [&]() {
        for (std::size_t segment = 1; segment < state.segment_count; ++segment) {
          process_plain_segment(segment);
        }
      };

      auto process_unrolled_plain_segments = [&](std::size_t exact_segment_count) {
        process_plain_segment(1U);
        process_plain_segment(2U);
        process_plain_segment(3U);
        for (std::size_t segment = 4U; segment < exact_segment_count; segment += 4U) {
          process_plain_segment(segment);
          process_plain_segment(segment + 1U);
          process_plain_segment(segment + 2U);
          process_plain_segment(segment + 3U);
        }
      };

      process_first_segment();
      if constexpr (UseGlobalMainFSegment128Unroll<Ops>::value) {
        if (state.segment_count == 128U) {
          process_unrolled_plain_segments(128U);
        } else if constexpr (UseGlobalMainFSegment64Unroll<Ops>::value) {
          if (state.segment_count == 64U) {
            process_unrolled_plain_segments(64U);
          } else if constexpr (UseGlobalMainFSegment32Unroll<Ops>::value) {
            if (state.segment_count == 32U) {
              process_unrolled_plain_segments(32U);
            } else {
              process_plain_tail();
            }
          } else {
            process_plain_tail();
          }
        } else if constexpr (UseGlobalMainFSegment32Unroll<Ops>::value) {
          if (state.segment_count == 32U) {
            process_unrolled_plain_segments(32U);
          } else {
            process_plain_tail();
          }
        } else {
          process_plain_tail();
        }
      } else if constexpr (UseGlobalMainFSegment64Unroll<Ops>::value) {
        if (state.segment_count == 64U) {
          process_unrolled_plain_segments(64U);
        } else if constexpr (UseGlobalMainFSegment32Unroll<Ops>::value) {
          if (state.segment_count == 32U) {
            process_unrolled_plain_segments(32U);
          } else {
            process_plain_tail();
          }
        } else {
          process_plain_tail();
        }
      } else if constexpr (UseGlobalMainFSegment32Unroll<Ops>::value) {
        if (state.segment_count == 32U) {
          process_unrolled_plain_segments(32U);
        } else {
          process_plain_tail();
        }
      } else {
        process_plain_tail();
      }
    } else {
      for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
        Cell* h_store_segment = h_store_data + segment * lane_count;
        Cell* h_load_segment = h_load_data + segment * lane_count;
        Cell* e_segment = e_store_data + segment * lane_count;
        const Cell* profile_segment = profile_row + segment * lane_count;

        const auto v_profile = load_state_cells<Ops, Cell>(profile_segment);
        auto v_e = load_state_cells<Ops, Cell>(e_segment);
        v_h = Ops::add(v_h, v_profile);
        v_h = Ops::max(v_h, v_e);
        v_h = Ops::max(v_h, v_f);
        store_state_cells<Ops, Cell>(h_store_segment, v_h);

        const auto v_h_open = Ops::add(v_h, gap_open_vector);
        v_e = Ops::max(Ops::add(v_e, gap_extend_vector), v_h_open);
        store_state_cells<Ops, Cell>(e_segment, v_e);
        v_f = Ops::max(add_sentinel<Ops, Cell>(v_f, gap_extend_vector, low_score), v_h_open);
        v_h = load_state_cells<Ops, Cell>(h_load_segment);
      }
    }

    if (can_prefix_lazy_f) {
      v_f = global_lazy_f_prefix_carry_no_padding<Ops, Cell>(
          v_f,
          state.segment_count,
          state.gap_extend_score,
          low_score);
      if constexpr (UseDenseGlobalLazyFScan<Ops>::value) {
        if constexpr (UseMaskedDenseGlobalLazyFScan<Ops>::value) {
          scan_global_lazy_f_no_padding_dense_masked<Ops, Cell>(
              h_store_data,
              e_store_data,
              state.segment_count,
              v_f,
              gap_open_vector,
              gap_extend_vector,
              low_score);
        } else {
          scan_global_lazy_f_no_padding_dense<Ops, Cell>(
              h_store_data,
              e_store_data,
              state.segment_count,
              v_f,
              gap_open_vector,
              gap_extend_vector,
              low_score);
        }
      } else {
        scan_global_lazy_f<Ops, Cell, false>(
            h_store_data,
            e_store_data,
            state.segment_count,
            v_f,
            gap_open_vector,
            gap_extend_vector,
            low_score);
      }
    } else {
      for (std::size_t iteration = 0; iteration < lane_count; ++iteration) {
        v_f = shift_left_insert<Ops, Cell>(v_f, low_score);
        const bool propagated = scan_global_lazy_f<Ops, Cell, false>(
            h_store_data,
            e_store_data,
            state.segment_count,
            v_f,
            gap_open_vector,
            gap_extend_vector,
            low_score);

        (void) propagated;
      }
    }

    top_left = next_top_left;
    next_top_left = static_cast<Cell>(
        static_cast<Score>(next_top_left) +
        static_cast<Score>(state.gap_extend_score));
    first_f = static_cast<Cell>(
        static_cast<Score>(first_f) +
        static_cast<Score>(state.gap_extend_score));
  }

  return static_cast<Score>(h_store_data[state.segment_count * lane_count - 1U]);
}

template <template <typename, typename> class OpsTemplate, typename Cell>
Score global_affine_score_state(PreparedAffineScoreState<Cell>& state) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  if (state.query_size != 0 &&
      state.query_size == state.segment_count * Ops::lane_count) {
    if (state.query_size == state.target_size) {
      return global_affine_score_state_equal_length_no_padding<OpsTemplate, Cell>(state);
    }
    return global_affine_score_state_impl<OpsTemplate, Cell, false>(state);
  }
  return global_affine_score_state_impl<OpsTemplate, Cell, true>(state);
}

template <template <typename, typename> class OpsTemplate, typename Cell>
Score affine_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  auto state = prepare_affine_score_state<OpsTemplate, Cell>(
      prepared,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score);
  return affine_score_state<OpsTemplate, Cell>(state);
}

template <template <typename, typename> class OpsTemplate, typename Cell>
Score global_affine_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  auto state = prepare_affine_score_state<OpsTemplate, Cell>(
      prepared,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score);
  return global_affine_score_state<OpsTemplate, Cell>(state);
}

inline AlignmentPath boundary_affine_path(
    std::size_t query_size,
    std::size_t target_size,
    Score gap_open_score,
    Score gap_extend_score,
    bool local_alignment) {
  if (local_alignment) {
    return make_alignment_path(0, 0, 0, 0, 0, "");
  }

  std::string operations;
  operations.reserve(query_size + target_size);
  operations.append(query_size, 'D');
  operations.append(target_size, 'I');
  const Score score = query_size != 0
      ? static_cast<Score>(affine_gap_cost<Score>(query_size, gap_open_score, gap_extend_score))
      : static_cast<Score>(affine_gap_cost<Score>(target_size, gap_open_score, gap_extend_score));
  return make_alignment_path(score, 0, query_size, 0, target_size, operations);
}

inline std::string boundary_affine_cigar(
    std::size_t query_size,
    std::size_t target_size,
    bool local_alignment) {
  if (local_alignment) {
    return "";
  }

  std::string cigar;
  if (query_size != 0) {
    cigar += std::to_string(query_size);
    cigar.push_back('D');
  }
  if (target_size != 0) {
    cigar += std::to_string(target_size);
    cigar.push_back('I');
  }
  return cigar;
}

template <typename Cell>
AlignmentPath build_affine_path_from_striped_trace(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    const std::vector<std::uint8_t>& trace,
    Score score,
    std::size_t best_row,
    std::size_t best_column,
    bool local_alignment,
    std::size_t segment_count,
    std::size_t lane_count) {
  const std::size_t state_cell_count = segment_count * lane_count;
  std::string operations;
  operations.reserve(query.size() + target.size());

  std::size_t row = best_row;
  std::size_t column = best_column;
  TraceState state = TraceState::h;

  while (row > 0 || column > 0) {
    if (row == 0) {
      if (local_alignment) {
        break;
      }
      operations.push_back('I');
      --column;
      state = column > 0 ? TraceState::left : TraceState::h;
      continue;
    }
    if (column == 0) {
      if (local_alignment) {
        break;
      }
      operations.push_back('D');
      --row;
      state = row > 0 ? TraceState::up : TraceState::h;
      continue;
    }

    const std::uint8_t trace_cell =
        trace[trace_striped_index(row, column, segment_count, lane_count, state_cell_count)];

    if (state == TraceState::h) {
      const TraceDirection direction = trace_direction(trace_cell);
      if (local_alignment && direction == TraceDirection::stop) {
        break;
      }
      if (direction == TraceDirection::diagonal) {
        operations.push_back(query[row - 1U] == target[column - 1U] ? 'M' : 'X');
        --row;
        --column;
        continue;
      }
      if (direction == TraceDirection::up) {
        state = TraceState::up;
        continue;
      }
      if (direction == TraceDirection::left) {
        state = TraceState::left;
        continue;
      }
      break;
    }

    if (state == TraceState::up) {
      operations.push_back('D');
      const bool continues = row > 1U && trace_up_continues(trace_cell);
      --row;
      state = continues ? TraceState::up : TraceState::h;
      continue;
    }

    operations.push_back('I');
    const bool continues = column > 1U && trace_left_continues(trace_cell);
    --column;
    state = continues ? TraceState::left : TraceState::h;
  }

  std::reverse(operations.begin(), operations.end());
  return make_alignment_path(score, row, best_row, column, best_column, operations);
}

template <typename Cell>
std::string build_affine_cigar_from_striped_trace(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    const std::vector<std::uint8_t>& trace,
    std::size_t best_row,
    std::size_t best_column,
    bool local_alignment,
    std::size_t segment_count,
    std::size_t lane_count) {
  const std::size_t state_cell_count = segment_count * lane_count;
  ReverseCigarBuilder cigar;

  std::size_t row = best_row;
  std::size_t column = best_column;
  TraceState state = TraceState::h;

  while (row > 0 || column > 0) {
    if (row == 0) {
      if (local_alignment) {
        break;
      }
      cigar.push('I');
      --column;
      state = column > 0 ? TraceState::left : TraceState::h;
      continue;
    }
    if (column == 0) {
      if (local_alignment) {
        break;
      }
      cigar.push('D');
      --row;
      state = row > 0 ? TraceState::up : TraceState::h;
      continue;
    }

    const std::uint8_t trace_cell =
        trace[trace_striped_index(row, column, segment_count, lane_count, state_cell_count)];

    if (state == TraceState::h) {
      const TraceDirection direction = trace_direction(trace_cell);
      if (local_alignment && direction == TraceDirection::stop) {
        break;
      }
      if (direction == TraceDirection::diagonal) {
        cigar.push(query[row - 1U] == target[column - 1U] ? 'M' : 'X');
        --row;
        --column;
        continue;
      }
      if (direction == TraceDirection::up) {
        state = TraceState::up;
        continue;
      }
      if (direction == TraceDirection::left) {
        state = TraceState::left;
        continue;
      }
      break;
    }

    if (state == TraceState::up) {
      cigar.push('D');
      const bool continues = row > 1U && trace_up_continues(trace_cell);
      --row;
      state = continues ? TraceState::up : TraceState::h;
      continue;
    }

    cigar.push('I');
    const bool continues = column > 1U && trace_left_continues(trace_cell);
    --column;
    state = continues ? TraceState::left : TraceState::h;
  }

  return cigar.str();
}

template <
    template <typename, typename> class OpsTemplate,
    typename Cell,
    bool LocalAlignment,
    bool CigarOnly = false>
auto affine_striped_traceback_state(
    PreparedAffineScoreState<Cell>& state,
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;

  if (query.empty() || target.empty()) {
    if constexpr (CigarOnly) {
      return boundary_affine_cigar(query.size(), target.size(), LocalAlignment);
    } else {
      return boundary_affine_path(
          query.size(),
          target.size(),
          static_cast<Score>(state.gap_open_score),
          static_cast<Score>(state.gap_extend_score),
          LocalAlignment);
    }
  }

  const std::size_t state_cell_count = state.segment_count * lane_count;
  std::vector<std::uint8_t> trace(
      (target.size() + 1U) * state_cell_count,
      pack_trace(TraceDirection::stop, false, false));

  std::vector<std::uint8_t> e_continue_store(state_cell_count, 0);

  if constexpr (LocalAlignment) {
    std::fill(state.h_store.begin(), state.h_store.end(), Cell{0});
    std::fill(state.h_load.begin(), state.h_load.end(), Cell{0});
    std::fill(state.e_store.begin(), state.e_store.end(), Cell{0});
  } else {
    reset_global_affine_initial_column(state, lane_count);
    std::fill(state.h_load.begin(), state.h_load.end(), std::numeric_limits<Cell>::lowest());
  }

  const auto zero_vector = Ops::zero();
  const auto gap_open_vector = Ops::set1(state.gap_open_score);
  const auto gap_extend_vector = Ops::set1(state.gap_extend_score);
  const Cell low_score = std::numeric_limits<Cell>::lowest();
  const bool can_stop_lazy_f = state.gap_open_score <= 0 && state.gap_extend_score <= 0;
  const bool can_prefix_lazy_f =
      state.gap_open_score <= state.gap_extend_score && state.gap_extend_score <= 0;
  Cell* h_store_data = state.h_store.data();
  Cell* h_load_data = state.h_load.data();
  Cell* e_store_data = state.e_store.data();
  const Cell* profile_data = state.profile.data();

  Score best_score = 0;
  std::size_t best_row = 0;
  std::size_t best_column = 0;

  alignas(Ops::alignment) Cell diagonal_scores[lane_count] = {};
  alignas(Ops::alignment) Cell e_scores[lane_count] = {};
  alignas(Ops::alignment) Cell f_scores[lane_count] = {};
  alignas(Ops::alignment) Cell h_scores[lane_count] = {};
  alignas(Ops::alignment) Cell e_extend_scores[lane_count] = {};
  alignas(Ops::alignment) Cell e_open_scores[lane_count] = {};
  alignas(Ops::alignment) Cell f_extend_scores[lane_count] = {};
  alignas(Ops::alignment) Cell f_open_scores[lane_count] = {};
  std::array<std::uint8_t, lane_count> f_continue_lanes = {};

  for (std::size_t target_index = 0; target_index < state.target_profile_offsets.size();
       ++target_index) {
    const std::size_t column = target_index + 1U;
    std::swap(h_store_data, h_load_data);
    std::fill(f_continue_lanes.begin(), f_continue_lanes.end(), 0);

    typename Ops::vector_type v_h;
    typename Ops::vector_type v_f;
    if constexpr (LocalAlignment) {
      v_h = shift_left_zero<Ops, Cell>(
          load_state_cells<Ops, Cell>(h_load_data + ((state.segment_count - 1U) * lane_count)));
      v_f = zero_vector;
    } else {
      const Cell top_left = affine_gap_cost<Cell>(
          target_index,
          state.gap_open_score,
          state.gap_extend_score);
      const Cell top_score = affine_gap_cost<Cell>(
          target_index + 1U,
          state.gap_open_score,
          state.gap_extend_score);
      const Cell first_f = static_cast<Cell>(
          static_cast<Score>(top_score) + static_cast<Score>(state.gap_open_score));
      v_h = shift_left_insert<Ops, Cell>(
          load_state_cells<Ops, Cell>(h_load_data + ((state.segment_count - 1U) * lane_count)),
          top_left);
      v_f = first_lane_vector<Ops, Cell>(first_f, low_score);
    }

    const Cell* profile_row = profile_data + state.target_profile_offsets[target_index];

    for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
      Cell* h_store_segment = h_store_data + segment * lane_count;
      Cell* h_load_segment = h_load_data + segment * lane_count;
      Cell* e_segment = e_store_data + segment * lane_count;
      const Cell* profile_segment = profile_row + segment * lane_count;

      const auto v_profile = load_state_cells<Ops, Cell>(profile_segment);
      auto v_e = load_state_cells<Ops, Cell>(e_segment);
      auto v_diagonal = LocalAlignment
          ? Ops::add(v_h, v_profile)
          : add_sentinel<Ops, Cell>(v_h, v_profile, low_score);
      auto v_cell = Ops::max(v_diagonal, v_e);
      v_cell = Ops::max(v_cell, v_f);
      if constexpr (LocalAlignment) {
        v_cell = Ops::max(v_cell, zero_vector);
      }
      store_state_cells<Ops, Cell>(h_store_segment, v_cell);

      Ops::store_cells(diagonal_scores, v_diagonal);
      Ops::store_cells(e_scores, v_e);
      Ops::store_cells(f_scores, v_f);
      Ops::store_cells(h_scores, v_cell);

      for (std::size_t lane = 0; lane < lane_count; ++lane) {
        const std::size_t query_index = lane * state.segment_count + segment;
        if (query_index >= query.size()) {
          continue;
        }

        const std::size_t row = query_index + 1U;
        const std::size_t state_index = segment * lane_count + lane;
        Cell selected_score = diagonal_scores[lane];
        TraceDirection direction = TraceDirection::diagonal;
        if (f_scores[lane] > selected_score) {
          selected_score = f_scores[lane];
          direction = TraceDirection::up;
        }
        if (e_scores[lane] > selected_score) {
          selected_score = e_scores[lane];
          direction = TraceDirection::left;
        }
        if constexpr (LocalAlignment) {
          if (selected_score <= 0) {
            direction = TraceDirection::stop;
          }
        }

        trace[trace_striped_index(row, column, state.segment_count, lane_count, state_cell_count)] =
            pack_trace(
                direction,
                f_continue_lanes[lane] != 0,
                e_continue_store[state_index] != 0);

        if constexpr (LocalAlignment) {
          const Score cell_score = static_cast<Score>(h_scores[lane]);
          if (local_trace_best_is_better(
                  cell_score,
                  row,
                  column,
                  best_score,
                  best_row,
                  best_column)) {
            best_score = cell_score;
            best_row = row;
            best_column = column;
          }
        }
      }

      const auto v_h_open = LocalAlignment
          ? Ops::add(v_cell, gap_open_vector)
          : add_sentinel<Ops, Cell>(v_cell, gap_open_vector, low_score);
      const auto v_e_extend = LocalAlignment
          ? Ops::add(v_e, gap_extend_vector)
          : add_sentinel<Ops, Cell>(v_e, gap_extend_vector, low_score);
      const auto v_f_extend = LocalAlignment
          ? Ops::add(v_f, gap_extend_vector)
          : add_sentinel<Ops, Cell>(v_f, gap_extend_vector, low_score);
      v_e = Ops::max(v_e_extend, v_h_open);
      v_f = Ops::max(v_f_extend, v_h_open);
      store_state_cells<Ops, Cell>(e_segment, v_e);

      Ops::store_cells(e_extend_scores, v_e_extend);
      Ops::store_cells(e_open_scores, v_h_open);
      Ops::store_cells(f_extend_scores, v_f_extend);
      Ops::store_cells(f_open_scores, v_h_open);
      for (std::size_t lane = 0; lane < lane_count; ++lane) {
        const std::size_t query_index = lane * state.segment_count + segment;
        if (query_index >= query.size()) {
          continue;
        }
        const std::size_t state_index = segment * lane_count + lane;
        e_continue_store[state_index] =
            e_extend_scores[lane] >= e_open_scores[lane] ? 1U : 0U;
        f_continue_lanes[lane] =
            f_extend_scores[lane] >= f_open_scores[lane] ? 1U : 0U;
      }

      v_h = load_state_cells<Ops, Cell>(h_load_segment);
    }

    const auto scan_trace_lazy_f = [&]() {
      bool propagated = false;
      for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
        Cell* h_store_segment = h_store_data + segment * lane_count;
        Cell* e_segment = e_store_data + segment * lane_count;

        const auto v_h_previous = load_state_cells<Ops, Cell>(h_store_segment);
        auto v_h_updated = Ops::max(v_h_previous, v_f);
        store_state_cells<Ops, Cell>(h_store_segment, v_h_updated);

        Ops::store_cells(h_scores, v_h_updated);
        alignas(Ops::alignment) Cell previous_h_scores[lane_count] = {};
        Ops::store_cells(previous_h_scores, v_h_previous);
        Ops::store_cells(f_scores, v_f);

        for (std::size_t lane = 0; lane < lane_count; ++lane) {
          const std::size_t query_index = lane * state.segment_count + segment;
          if (query_index >= query.size()) {
            continue;
          }
          if (f_scores[lane] > previous_h_scores[lane]) {
            propagated = true;
            const std::size_t row = query_index + 1U;
            trace[trace_striped_index(row, column, state.segment_count, lane_count, state_cell_count)] =
                pack_trace(TraceDirection::up, f_continue_lanes[lane] != 0, false);

            if constexpr (LocalAlignment) {
              const Score cell_score = static_cast<Score>(h_scores[lane]);
              if (local_trace_best_is_better(
                      cell_score,
                      row,
                      column,
                      best_score,
                      best_row,
                      best_column)) {
                best_score = cell_score;
                best_row = row;
                best_column = column;
              }
            }
          }
        }

        auto v_e = load_state_cells<Ops, Cell>(e_segment);
        const auto v_h_open = LocalAlignment
            ? Ops::add(v_h_updated, gap_open_vector)
            : add_sentinel<Ops, Cell>(v_h_updated, gap_open_vector, low_score);
        const auto v_e_updated = Ops::max(v_e, v_h_open);
        store_state_cells<Ops, Cell>(e_segment, v_e_updated);

        Ops::store_cells(e_scores, v_e);
        Ops::store_cells(e_open_scores, v_h_open);
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
          const std::size_t query_index = lane * state.segment_count + segment;
          if (query_index >= query.size()) {
            continue;
          }
          if (e_open_scores[lane] > e_scores[lane]) {
            e_continue_store[segment * lane_count + lane] = 0;
          }
        }

        const auto v_f_extend = LocalAlignment
            ? Ops::add(v_f, gap_extend_vector)
            : add_sentinel<Ops, Cell>(v_f, gap_extend_vector, low_score);
        v_f = Ops::max(v_f_extend, v_h_open);
        Ops::store_cells(f_extend_scores, v_f_extend);
        Ops::store_cells(f_open_scores, v_h_open);
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
          const std::size_t query_index = lane * state.segment_count + segment;
          if (query_index >= query.size()) {
            continue;
          }
          f_continue_lanes[lane] =
              f_extend_scores[lane] >= f_open_scores[lane] ? 1U : 0U;
        }
      }

      return propagated;
    };

    if (can_prefix_lazy_f) {
      if constexpr (LocalAlignment) {
        v_f = trace_lazy_f_prefix_carry<Ops, Cell, lane_count, true>(
            v_f,
            f_continue_lanes,
            state.segment_count,
            state.gap_extend_score,
            low_score);
        if (any_greater<Ops, Cell>(v_f, zero_vector)) {
          scan_trace_lazy_f();
        }
      } else {
        v_f = trace_lazy_f_prefix_carry<Ops, Cell, lane_count, false>(
            v_f,
            f_continue_lanes,
            state.segment_count,
            state.gap_extend_score,
            low_score);
        scan_trace_lazy_f();
      }
    } else {
      const auto lazy_insert = [&]<bool IsLocal>() {
        if constexpr (IsLocal) {
          v_f = shift_left_zero<Ops, Cell>(v_f);
        } else {
          v_f = shift_left_insert<Ops, Cell>(v_f, low_score);
        }
        shift_bool_lanes_left_zero(f_continue_lanes);
      };

      for (std::size_t iteration = 0; iteration < lane_count; ++iteration) {
        if constexpr (LocalAlignment) {
          lazy_insert.template operator()<true>();
        } else {
          lazy_insert.template operator()<false>();
        }

        const bool propagated = scan_trace_lazy_f();
        if constexpr (LocalAlignment) {
          if (can_stop_lazy_f && !any_greater<Ops, Cell>(v_f, zero_vector)) {
            break;
          }
        } else {
          (void) propagated;
        }
      }
    }
  }

  if constexpr (!LocalAlignment) {
    best_row = query.size();
    best_column = target.size();
    best_score = static_cast<Score>(
        striped_row_value(h_store_data, state.query_size, state.segment_count, lane_count));
  }

  if constexpr (CigarOnly) {
    return build_affine_cigar_from_striped_trace<Cell>(
        query,
        target,
        trace,
        best_row,
        best_column,
        LocalAlignment,
        state.segment_count,
        lane_count);
  } else {
    return build_affine_path_from_striped_trace<Cell>(
        query,
        target,
        trace,
        best_score,
        best_row,
        best_column,
        LocalAlignment,
        state.segment_count,
        lane_count);
  }
}

template <template <typename, typename> class OpsTemplate, typename Cell, bool LocalAlignment>
AlignmentPath affine_striped_path_info(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  auto state = prepare_affine_score_state<OpsTemplate, Cell>(
      prepared,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score);
  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());
  return affine_striped_traceback_state<OpsTemplate, Cell, LocalAlignment, false>(
      state,
      query,
      target);
}

template <template <typename, typename> class OpsTemplate, typename Cell, bool LocalAlignment>
std::string affine_striped_cigar(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  auto state = prepare_affine_score_state<OpsTemplate, Cell>(
      prepared,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score);
  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());
  return affine_striped_traceback_state<OpsTemplate, Cell, LocalAlignment, true>(
      state,
      query,
      target);
}

template <template <typename, typename> class OpsTemplate, bool LocalAlignment>
AlignmentPath dispatch_affine_striped_path_info(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return affine_striped_path_info<OpsTemplate, std::int8_t, LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits16:
      return affine_striped_path_info<OpsTemplate, std::int16_t, LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits32:
      return affine_striped_path_info<OpsTemplate, std::int32_t, LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits64:
      return affine_striped_path_info<OpsTemplate, std::int64_t, LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported affine striped traceback width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate, bool LocalAlignment>
std::string dispatch_affine_striped_cigar(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return affine_striped_cigar<OpsTemplate, std::int8_t, LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits16:
      return affine_striped_cigar<OpsTemplate, std::int16_t, LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits32:
      return affine_striped_cigar<OpsTemplate, std::int32_t, LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits64:
      return affine_striped_cigar<OpsTemplate, std::int64_t, LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported affine striped traceback width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate>
Score dispatch_affine_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return affine_score<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits16:
      return affine_score<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits32:
      return affine_score<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits64:
      return affine_score<OpsTemplate, std::int64_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported affine Farrar score width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate>
Score dispatch_global_affine_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return global_affine_score<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits16:
      return global_affine_score<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits32:
      return global_affine_score<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
    case KernelBits::bits64:
      return global_affine_score<OpsTemplate, std::int64_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported global affine Farrar score width");
  throw nb::python_error();
}

template <typename Ops, typename Cell>
void local_sw_score_main_segment(
    Cell* h_store_data,
    Cell* h_load_data,
    Cell* e_store_data,
    const Cell* profile_row,
    std::size_t segment,
    typename Ops::vector_type& v_h,
    typename Ops::vector_type& v_f,
    typename Ops::vector_type gap_vector,
    typename Ops::vector_type zero_vector,
    typename Ops::vector_type& best_vector) {
  constexpr std::size_t lane_count = Ops::lane_count;

  Cell* h_store_segment = h_store_data + segment * lane_count;
  Cell* h_load_segment = h_load_data + segment * lane_count;
  Cell* e_segment = e_store_data + segment * lane_count;
  const Cell* profile_segment = profile_row + segment * lane_count;

  const auto v_profile = load_state_cells<Ops, Cell>(profile_segment);
  auto v_e = load_state_cells<Ops, Cell>(e_segment);
  v_h = Ops::add(v_h, v_profile);
  v_h = Ops::max(v_h, v_e);
  v_h = Ops::max(v_h, v_f);
  v_h = Ops::max(v_h, zero_vector);
  store_state_cells<Ops, Cell>(h_store_segment, v_h);
  best_vector = Ops::max(best_vector, v_h);

  const auto v_h_gap = Ops::add(v_h, gap_vector);
  v_e = Ops::max(Ops::add(v_e, gap_vector), v_h_gap);
  store_state_cells<Ops, Cell>(e_segment, v_e);
  v_f = Ops::max(Ops::add(v_f, gap_vector), v_h_gap);
  v_h = load_state_cells<Ops, Cell>(h_load_segment);
}

template <typename Ops, typename Cell, std::size_t SegmentCount>
Score score_state_exact_fill_local_sw(PreparedScoreState<Cell>& state) {
  constexpr std::size_t lane_count = Ops::lane_count;

  if (state.fast_score.has_value()) {
    return *state.fast_score;
  }
  if (state.segment_count != SegmentCount || state.target_profile_offsets.empty()) {
    return 0;
  }

  std::fill(state.h_store.begin(), state.h_store.end(), Cell{0});
  std::fill(state.h_load.begin(), state.h_load.end(), Cell{0});
  std::fill(state.e_store.begin(), state.e_store.end(), Cell{0});

  const auto zero_vector = Ops::zero();
  const auto gap_vector = Ops::set1(state.gap_score);
  auto best_vector = zero_vector;
  Cell* h_store_data = state.h_store.data();
  Cell* h_load_data = state.h_load.data();
  Cell* e_store_data = state.e_store.data();
  const Cell* profile_data = state.profile.data();
  const bool use_compact_loop = state.kernel_strategy == ScoreKernelStrategy::compact;

  for (const auto profile_offset : state.target_profile_offsets) {
    std::swap(h_store_data, h_load_data);

    auto v_h = shift_left_zero<Ops, Cell>(
        load_state_cells<Ops, Cell>(h_load_data + ((SegmentCount - 1U) * lane_count)));
    auto v_f = zero_vector;
    const Cell* profile_row = profile_data + profile_offset;

    if (use_compact_loop) {
      for (std::size_t segment = 0; segment < SegmentCount; ++segment) {
        local_sw_score_main_segment<Ops, Cell>(
            h_store_data,
            h_load_data,
            e_store_data,
            profile_row,
            segment,
            v_h,
            v_f,
            gap_vector,
            zero_vector,
            best_vector);
      }
    } else {
      for (std::size_t segment = 0; segment < SegmentCount; segment += 4U) {
        local_sw_score_main_segment<Ops, Cell>(
            h_store_data,
            h_load_data,
            e_store_data,
            profile_row,
            segment,
            v_h,
            v_f,
            gap_vector,
            zero_vector,
            best_vector);
        local_sw_score_main_segment<Ops, Cell>(
            h_store_data,
            h_load_data,
            e_store_data,
            profile_row,
            segment + 1U,
            v_h,
            v_f,
            gap_vector,
            zero_vector,
            best_vector);
        local_sw_score_main_segment<Ops, Cell>(
            h_store_data,
            h_load_data,
            e_store_data,
            profile_row,
            segment + 2U,
            v_h,
            v_f,
            gap_vector,
            zero_vector,
            best_vector);
        local_sw_score_main_segment<Ops, Cell>(
            h_store_data,
            h_load_data,
            e_store_data,
            profile_row,
            segment + 3U,
            v_h,
            v_f,
            gap_vector,
            zero_vector,
            best_vector);
      }
    }

    v_f = local_lazy_f_prefix_carry<Ops, Cell>(
        v_f,
        SegmentCount,
        state.gap_score);
    if (any_greater<Ops, Cell>(v_f, zero_vector)) {
      const bool use_bounded_correction =
          state.kernel_strategy == ScoreKernelStrategy::automatic ||
          state.kernel_strategy == ScoreKernelStrategy::bounded ||
          state.kernel_strategy == ScoreKernelStrategy::compact;
      if constexpr (requires { Ops::bounded_local_sw_lazy_f_scan; }) {
        if constexpr (Ops::bounded_local_sw_lazy_f_scan) {
          if (use_bounded_correction) {
            scan_local_linear_lazy_f_once_bounded<Ops, Cell>(
                h_store_data,
                SegmentCount,
                v_f,
                gap_vector,
                best_vector);
          } else {
            scan_local_linear_lazy_f_once<Ops, Cell>(
                h_store_data,
                SegmentCount,
                v_f,
                gap_vector,
                best_vector);
          }
        } else {
          scan_local_linear_lazy_f_once<Ops, Cell>(
              h_store_data,
              SegmentCount,
              v_f,
              gap_vector,
              best_vector);
        }
      } else {
        scan_local_linear_lazy_f_once<Ops, Cell>(
            h_store_data,
            SegmentCount,
            v_f,
            gap_vector,
            best_vector);
      }
    }
  }

  return static_cast<Score>(reduce_max<Ops, Cell>(best_vector));
}

template <template <typename, typename> class OpsTemplate, typename Cell>
Score score_state(PreparedScoreState<Cell>& state) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;

  if (state.fast_score.has_value()) {
    return *state.fast_score;
  }
  if (state.segment_count == 0 || state.target_profile_offsets.empty()) {
    return 0;
  }
  if (state.gap_score <= 0 && state.query_size == state.segment_count * lane_count) {
    if constexpr (requires { Ops::local_sw_score_exact_segment32; }) {
      if constexpr (Ops::local_sw_score_exact_segment32) {
        if (state.segment_count == 32U) {
          if constexpr (requires { Ops::local_sw_score_exact_segment32_raw(state); }) {
            return Ops::local_sw_score_exact_segment32_raw(state);
          }
          return score_state_exact_fill_local_sw<Ops, Cell, 32U>(state);
        }
      }
    }
    if constexpr (requires { Ops::local_sw_score_exact_segment64; }) {
      if constexpr (Ops::local_sw_score_exact_segment64) {
        if (state.segment_count == 64U) {
          if constexpr (requires { Ops::local_sw_score_exact_segment64_raw(state); }) {
            return Ops::local_sw_score_exact_segment64_raw(state);
          }
          return score_state_exact_fill_local_sw<Ops, Cell, 64U>(state);
        }
      }
    }
    if constexpr (requires { Ops::local_sw_score_exact_segment128; }) {
      if constexpr (Ops::local_sw_score_exact_segment128) {
        if (state.segment_count == 128U) {
          if constexpr (requires { Ops::local_sw_score_exact_segment128_raw(state); }) {
            return Ops::local_sw_score_exact_segment128_raw(state);
          }
          return score_state_exact_fill_local_sw<Ops, Cell, 128U>(state);
        }
      }
    }
  }

  std::fill(state.h_store.begin(), state.h_store.end(), Cell{0});
  std::fill(state.h_load.begin(), state.h_load.end(), Cell{0});
  std::fill(state.e_store.begin(), state.e_store.end(), Cell{0});

  const auto zero_vector = Ops::zero();
  const auto gap_vector = Ops::set1(state.gap_score);
  const bool track_lazy_best = state.gap_score > 0;
  auto best_vector = zero_vector;
  Cell* h_store_data = state.h_store.data();
  Cell* h_load_data = state.h_load.data();
  Cell* e_store_data = state.e_store.data();
  const Cell* profile_data = state.profile.data();

  for (const auto profile_offset : state.target_profile_offsets) {
    std::swap(h_store_data, h_load_data);

    auto v_h = shift_left_zero<Ops, Cell>(
        load_state_cells<Ops, Cell>(h_load_data + ((state.segment_count - 1U) * lane_count)));
    auto v_f = zero_vector;
    const Cell* profile_row = profile_data + profile_offset;

    for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
      Cell* h_store_segment = h_store_data + segment * lane_count;
      Cell* h_load_segment = h_load_data + segment * lane_count;
      Cell* e_segment = e_store_data + segment * lane_count;
      const Cell* profile_segment = profile_row + segment * lane_count;

      const auto v_profile = load_state_cells<Ops, Cell>(profile_segment);
      auto v_e = load_state_cells<Ops, Cell>(e_segment);
      v_h = Ops::add(v_h, v_profile);
      v_h = Ops::max(v_h, v_e);
      v_h = Ops::max(v_h, v_f);
      v_h = Ops::max(v_h, zero_vector);
      store_state_cells<Ops, Cell>(h_store_segment, v_h);
      best_vector = Ops::max(best_vector, v_h);

      const auto v_h_gap = Ops::add(v_h, gap_vector);
      v_e = Ops::max(Ops::add(v_e, gap_vector), v_h_gap);
      store_state_cells<Ops, Cell>(e_segment, v_e);
      v_f = Ops::max(Ops::add(v_f, gap_vector), v_h_gap);
      v_h = load_state_cells<Ops, Cell>(h_load_segment);
    }

    // Parasail/SWPS3 can stop the lazy-F scan inside the segment loop by
    // comparing against H-gap, but that deliberately excludes adjacent
    // insertion/deletion paths. Keep full exact propagation and optimize only
    // the convergence bookkeeping here.
    if constexpr (requires(typename Ops::vector_type value) {
                    Ops::greater_mask(value, value);
                    Ops::bit_or(value, value);
                    Ops::any_nonzero(value);
                  }) {
      for (std::size_t iteration = 0; iteration < lane_count; ++iteration) {
        v_f = shift_left_zero<Ops, Cell>(v_f);
        auto propagated_vector = zero_vector;

        for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
          Cell* h_store_segment = h_store_data + segment * lane_count;
          auto v_h_segment = load_state_cells<Ops, Cell>(h_store_segment);
          propagated_vector =
              Ops::bit_or(propagated_vector, Ops::greater_mask(v_f, v_h_segment));
          v_h_segment = Ops::max(v_h_segment, v_f);
          store_state_cells<Ops, Cell>(h_store_segment, v_h_segment);
          if (track_lazy_best) {
            best_vector = Ops::max(best_vector, v_h_segment);
          }

          v_f = Ops::add(v_f, gap_vector);
        }

        if (state.gap_score <= 0 && !Ops::any_nonzero(propagated_vector)) {
          break;
        }
      }
    } else if constexpr (requires(typename Ops::vector_type value, typename Ops::mask_type mask) {
                           Ops::empty_mask();
                           Ops::greater_mask(value, value);
                           Ops::mask_or(mask, mask);
                           Ops::any_mask(mask);
                         }) {
      for (std::size_t iteration = 0; iteration < lane_count; ++iteration) {
        v_f = shift_left_zero<Ops, Cell>(v_f);
        auto propagated_mask = Ops::empty_mask();

        for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
          Cell* h_store_segment = h_store_data + segment * lane_count;
          auto v_h_segment = load_state_cells<Ops, Cell>(h_store_segment);
          propagated_mask =
              Ops::mask_or(propagated_mask, Ops::greater_mask(v_f, v_h_segment));
          v_h_segment = Ops::max(v_h_segment, v_f);
          store_state_cells<Ops, Cell>(h_store_segment, v_h_segment);
          if (track_lazy_best) {
            best_vector = Ops::max(best_vector, v_h_segment);
          }

          v_f = Ops::add(v_f, gap_vector);
        }

        if (state.gap_score <= 0 && !Ops::any_mask(propagated_mask)) {
          break;
        }
      }
    } else {
      for (std::size_t iteration = 0; iteration < lane_count; ++iteration) {
        v_f = shift_left_zero<Ops, Cell>(v_f);
        bool propagated = false;

        for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
          Cell* h_store_segment = h_store_data + segment * lane_count;
          auto v_h_segment = load_state_cells<Ops, Cell>(h_store_segment);
          propagated = propagated || any_greater<Ops, Cell>(v_f, v_h_segment);
          v_h_segment = Ops::max(v_h_segment, v_f);
          store_state_cells<Ops, Cell>(h_store_segment, v_h_segment);
          if (track_lazy_best) {
            best_vector = Ops::max(best_vector, v_h_segment);
          }

          v_f = Ops::add(v_f, gap_vector);
        }

        if (state.gap_score <= 0 && !propagated) {
          break;
        }
      }
    }
  }

  return static_cast<Score>(reduce_max<Ops, Cell>(best_vector));
}

template <template <typename, typename> class OpsTemplate, typename Cell>
Score global_score_state(PreparedScoreState<Cell>& state) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;

  if (state.fast_score.has_value()) {
    return *state.fast_score;
  }
  if (state.query_size == 0 || state.target_size == 0) {
    return static_cast<Score>(state.query_size + state.target_size) *
        static_cast<Score>(state.gap_score);
  }

  initialize_global_column_zero(
      state.h_store,
      state.query_size,
      state.segment_count,
      lane_count,
      state.gap_score);
  std::fill(state.h_load.begin(), state.h_load.end(), std::numeric_limits<Cell>::lowest());

  const auto gap_vector = Ops::set1(state.gap_score);
  const Cell low_score = std::numeric_limits<Cell>::lowest();
  Cell* h_store_data = state.h_store.data();
  Cell* h_load_data = state.h_load.data();
  const Cell* profile_data = state.profile.data();

  for (std::size_t target_index = 0; target_index < state.target_profile_offsets.size();
       ++target_index) {
    std::swap(h_store_data, h_load_data);

    const Cell top_left = static_cast<Cell>(
        static_cast<Score>(target_index) * static_cast<Score>(state.gap_score));
    const Cell first_up = static_cast<Cell>(
        static_cast<Score>(target_index + 2U) * static_cast<Score>(state.gap_score));
    auto v_h = shift_left_insert<Ops, Cell>(
        load_state_cells<Ops, Cell>(h_load_data + ((state.segment_count - 1U) * lane_count)),
        top_left);
    auto v_up = first_lane_vector<Ops, Cell>(first_up, low_score);
    const Cell* profile_row =
        profile_data + state.target_profile_offsets[target_index];

    for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
      Cell* h_store_segment = h_store_data + segment * lane_count;
      Cell* h_load_segment = h_load_data + segment * lane_count;
      const Cell* profile_segment = profile_row + segment * lane_count;

      const auto v_profile = load_state_cells<Ops, Cell>(profile_segment);
      const auto v_left = Ops::add(load_state_cells<Ops, Cell>(h_load_segment), gap_vector);
      auto v_cell = Ops::add(v_h, v_profile);
      v_cell = Ops::max(v_cell, v_left);
      v_cell = Ops::max(v_cell, v_up);
      store_state_cells<Ops, Cell>(h_store_segment, v_cell);

      v_up = Ops::add(v_cell, gap_vector);
      v_h = load_state_cells<Ops, Cell>(h_load_segment);
    }

    for (std::size_t iteration = 0; iteration < lane_count; ++iteration) {
      v_up = shift_left_insert<Ops, Cell>(v_up, low_score);
      bool propagated = false;

      for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
        Cell* h_store_segment = h_store_data + segment * lane_count;
        auto v_h_segment = load_state_cells<Ops, Cell>(h_store_segment);
        propagated = propagated || any_greater<Ops, Cell>(v_up, v_h_segment);
        v_h_segment = Ops::max(v_h_segment, v_up);
        store_state_cells<Ops, Cell>(h_store_segment, v_h_segment);
        v_up = Ops::add(v_h_segment, gap_vector);
      }

      if (state.gap_score <= 0 && !propagated) {
        break;
      }
    }
  }

  return static_cast<Score>(
      striped_row_value(h_store_data, state.query_size, state.segment_count, lane_count));
}

template <template <typename, typename> class OpsTemplate, typename Cell>
Score score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  auto state = prepare_score_state<OpsTemplate, Cell>(
      prepared,
      match_score,
      mismatch_score,
      gap_score);
  return score_state<OpsTemplate, Cell>(state);
}

template <template <typename, typename> class OpsTemplate, typename Cell>
Score global_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  auto state = prepare_global_score_state<OpsTemplate, Cell>(
      prepared,
      match_score,
      mismatch_score,
      gap_score);
  return global_score_state<OpsTemplate, Cell>(state);
}

struct LinearLocalEndpoint {
  Score score = 0;
  std::size_t row = 0;
  std::size_t column = 0;
};

struct LinearTracebackResult {
  Score score = 0;
  std::size_t query_start = 0;
  std::size_t query_end = 0;
  std::size_t target_start = 0;
  std::size_t target_end = 0;
  std::string operations;
};

struct LinearCigarTrace {
  Score score = 0;
  std::size_t query_start = 0;
  std::size_t query_end = 0;
  std::size_t target_start = 0;
  std::size_t target_end = 0;
  std::string cigar;
};

enum class LinearMaskedTraceOutput {
  path_info,
  traceback,
  cigar,
  cigar_trace,
};

template <typename Cell>
struct LinearCigarCheckpoints {
  std::size_t block_size = 64U;
  std::size_t checkpoint_count = 0;
  std::size_t state_cell_count = 0;
  AlignedVector<Cell> h;
  AlignedVector<Cell> e;

  const Cell* h_at(std::size_t index) const noexcept {
    return h.data() + index * state_cell_count;
  }

  const Cell* e_at(std::size_t index) const noexcept {
    return e.data() + index * state_cell_count;
  }

  Cell* h_at(std::size_t index) noexcept {
    return h.data() + index * state_cell_count;
  }

  Cell* e_at(std::size_t index) noexcept {
    return e.data() + index * state_cell_count;
  }
};

inline std::size_t linear_sw_cigar_checkpoint_block_size(
    std::size_t query_size,
    std::size_t target_size) noexcept {
  (void) query_size;
  (void) target_size;
  return 64U;
}

template <typename Cell>
LinearCigarCheckpoints<Cell> make_linear_cigar_checkpoints(
    std::size_t query_size,
    std::size_t target_size,
    std::size_t state_cell_count) {
  LinearCigarCheckpoints<Cell> checkpoints;
  checkpoints.block_size = linear_sw_cigar_checkpoint_block_size(query_size, target_size);
  checkpoints.checkpoint_count = (target_size + checkpoints.block_size - 1U) /
          checkpoints.block_size +
      1U;
  checkpoints.state_cell_count = state_cell_count;
  checkpoints.h.resize(checkpoints.checkpoint_count * state_cell_count);
  checkpoints.e.resize(checkpoints.checkpoint_count * state_cell_count);
  return checkpoints;
}

template <typename Cell>
void save_linear_cigar_checkpoint(
    LinearCigarCheckpoints<Cell>& checkpoints,
    std::size_t checkpoint_index,
    const Cell* h_values,
    const Cell* e_values) {
  std::copy_n(
      h_values,
      checkpoints.state_cell_count,
      checkpoints.h_at(checkpoint_index));
  std::copy_n(
      e_values,
      checkpoints.state_cell_count,
      checkpoints.e_at(checkpoint_index));
}

template <typename Ops, typename Cell>
void update_linear_local_endpoint_from_lanes(
    const Cell* h_scores,
    std::size_t segment,
    std::size_t column,
    std::size_t query_size,
    std::size_t segment_count,
    LinearLocalEndpoint& best) {
  constexpr std::size_t lane_count = Ops::lane_count;
  for (std::size_t lane = 0; lane < lane_count; ++lane) {
    const std::size_t query_index = lane * segment_count + segment;
    if (query_index >= query_size) {
      continue;
    }

    const Score cell_score = static_cast<Score>(h_scores[lane]);
    const std::size_t row = query_index + 1U;
    if (local_trace_best_is_better(
            cell_score,
            row,
            column,
            best.score,
            best.row,
            best.column)) {
      best = {cell_score, row, column};
    }
  }
}

struct LinearMaskedTrace {
  std::size_t segment_count = 0;
  std::vector<std::uint64_t> low_bits;
  std::vector<std::uint64_t> high_bits;

  LinearMaskedTrace(std::size_t target_size, std::size_t segments)
      : segment_count(segments),
        low_bits(target_size * segments, 0),
        high_bits(target_size * segments, 0) {}

  std::size_t index(std::size_t column, std::size_t segment) const noexcept {
    return (column - 1U) * segment_count + segment;
  }

  void set(
      std::size_t column,
      std::size_t segment,
      std::uint64_t low_mask,
      std::uint64_t high_mask) noexcept {
    const std::size_t mask_index = index(column, segment);
    low_bits[mask_index] = low_mask;
    high_bits[mask_index] = high_mask;
  }

  void force_up(std::size_t column, std::size_t segment, std::uint64_t mask) noexcept {
    const std::size_t mask_index = index(column, segment);
    low_bits[mask_index] &= ~mask;
    high_bits[mask_index] |= mask;
  }

  void force_up_when_not_diagonal(
      std::size_t column,
      std::size_t segment,
      std::uint64_t mask) noexcept {
    const std::size_t mask_index = index(column, segment);
    const std::uint64_t diagonal_mask = low_bits[mask_index] & ~high_bits[mask_index];
    mask &= ~diagonal_mask;
    low_bits[mask_index] &= ~mask;
    high_bits[mask_index] |= mask;
  }

  TraceDirection direction(
      std::size_t row,
      std::size_t column,
      std::size_t lane_count) const noexcept {
    const std::size_t query_index = row - 1U;
    const std::size_t lane = query_index / segment_count;
    const std::size_t segment = query_index % segment_count;
    if (lane >= lane_count) {
      return TraceDirection::stop;
    }

    const std::uint64_t mask = std::uint64_t{1} << lane;
    const std::size_t mask_index = index(column, segment);
    const bool low_bit = (low_bits[mask_index] & mask) != 0;
    const bool high_bit = (high_bits[mask_index] & mask) != 0;
    if (!high_bit) {
      return low_bit ? TraceDirection::diagonal : TraceDirection::stop;
    }
    return low_bit ? TraceDirection::left : TraceDirection::up;
  }
};

inline std::vector<std::uint64_t> make_striped_valid_lane_masks(
    std::size_t query_size,
    std::size_t segment_count,
    std::size_t lane_count) {
  std::vector<std::uint64_t> masks(segment_count, 0);
  for (std::size_t segment = 0; segment < segment_count; ++segment) {
    std::uint64_t mask = 0;
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
      const std::size_t query_index = lane * segment_count + segment;
      if (query_index < query_size) {
        mask |= std::uint64_t{1} << lane;
      }
    }
    masks[segment] = mask;
  }
  return masks;
}

template <typename Ops>
std::uint64_t trace_mask_gt(typename Ops::vector_type lhs, typename Ops::vector_type rhs) {
  if constexpr (requires { Ops::trace_mask_gt(lhs, rhs); }) {
    return Ops::trace_mask_gt(lhs, rhs);
  } else {
    static_assert(sizeof(Ops) == 0, "SimdOps must provide trace_mask_gt for masked traceback");
  }
}

template <typename Ops>
std::uint64_t trace_mask_eq(typename Ops::vector_type lhs, typename Ops::vector_type rhs) {
  if constexpr (requires { Ops::trace_mask_eq(lhs, rhs); }) {
    return Ops::trace_mask_eq(lhs, rhs);
  } else {
    static_assert(sizeof(Ops) == 0, "SimdOps must provide trace_mask_eq for masked traceback");
  }
}

inline std::uint64_t earlier_row_lane_mask(
    std::size_t segment,
    std::size_t best_row,
    std::size_t segment_count,
    std::size_t lane_count) noexcept {
  if (best_row <= segment + 1U) {
    return 0;
  }

  const std::size_t row_delta = best_row - segment - 1U;
  std::size_t lane_count_before_best =
      (row_delta + segment_count - 1U) / segment_count;
  lane_count_before_best = std::min(lane_count_before_best, lane_count);
  if (lane_count_before_best >= 64U) {
    return ~std::uint64_t{0};
  }
  return (std::uint64_t{1} << lane_count_before_best) - 1U;
}

template <typename Ops, typename Cell>
void update_linear_local_endpoint_from_vector(
    typename Ops::vector_type h_scores_vector,
    std::uint64_t candidate_mask,
    std::size_t segment,
    std::size_t column,
    std::size_t query_size,
    std::size_t segment_count,
    LinearLocalEndpoint& best) {
  if (candidate_mask == 0) {
    return;
  }

  const auto best_vector = Ops::set1(static_cast<Cell>(best.score));
  std::uint64_t greater_mask =
      trace_mask_gt<Ops>(h_scores_vector, best_vector) & candidate_mask;

  if (greater_mask != 0) {
    alignas(Ops::alignment) Cell h_scores[Ops::lane_count] = {};
    Ops::store_cells(h_scores, h_scores_vector);
    while (greater_mask != 0) {
      const auto lane = static_cast<std::size_t>(std::countr_zero(greater_mask));
      greater_mask &= greater_mask - 1U;

      const std::size_t query_index = lane * segment_count + segment;
      if (query_index >= query_size) {
        continue;
      }

      const Score cell_score = static_cast<Score>(h_scores[lane]);
      const std::size_t row = query_index + 1U;
      if (local_trace_best_is_better(
              cell_score,
              row,
              column,
              best.score,
              best.row,
              best.column)) {
        best = {cell_score, row, column};
      }
    }
    return;
  }

  if (best.score <= 0) {
    return;
  }

  const std::uint64_t tie_mask =
      trace_mask_eq<Ops>(h_scores_vector, best_vector) &
      candidate_mask &
      earlier_row_lane_mask(segment, best.row, segment_count, Ops::lane_count);
  if (tie_mask == 0) {
    return;
  }

  const auto lane = static_cast<std::size_t>(std::countr_zero(tie_mask));
  const std::size_t row = lane * segment_count + segment + 1U;
  best = {best.score, row, column};
}

template <typename Ops, typename Cell>
void update_linear_local_endpoint_for_score_from_mask(
    std::uint64_t candidate_mask,
    std::size_t segment,
    std::size_t column,
    std::size_t query_size,
    std::size_t segment_count,
    Score score,
    LinearLocalEndpoint& best) {
  while (candidate_mask != 0) {
    const auto lane = static_cast<std::size_t>(std::countr_zero(candidate_mask));
    candidate_mask &= candidate_mask - 1U;

    const std::size_t query_index = lane * segment_count + segment;
    if (query_index >= query_size) {
      continue;
    }

    const std::size_t row = query_index + 1U;
    if (local_trace_best_is_better(
            score,
            row,
            column,
            best.score,
            best.row,
            best.column)) {
      best = {score, row, column};
    }
  }
}

inline std::string build_linear_operations_from_masked_trace(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    const LinearMaskedTrace& trace,
    std::size_t best_row,
    std::size_t best_column,
    std::size_t lane_count,
    std::size_t& query_start,
    std::size_t& target_start) {
  std::string operations;
  operations.reserve(best_row + best_column);

  std::size_t row = best_row;
  std::size_t column = best_column;
  while (row > 0 && column > 0) {
    const auto direction = trace.direction(row, column, lane_count);
    if (direction == TraceDirection::stop) {
      break;
    }
    if (direction == TraceDirection::diagonal) {
      operations.push_back(query[row - 1U] == target[column - 1U] ? 'M' : 'X');
      --row;
      --column;
      continue;
    }
    if (direction == TraceDirection::up) {
      operations.push_back('D');
      --row;
      continue;
    }
    if (direction == TraceDirection::left) {
      operations.push_back('I');
      --column;
      continue;
    }
    break;
  }

  std::reverse(operations.begin(), operations.end());
  query_start = row;
  target_start = column;
  return operations;
}

inline AlignmentPath build_linear_path_from_masked_trace(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    const LinearMaskedTrace& trace,
    Score score,
    std::size_t best_row,
    std::size_t best_column,
    std::size_t lane_count) {
  std::size_t query_start = best_row;
  std::size_t target_start = best_column;
  const auto operations = build_linear_operations_from_masked_trace(
      query,
      target,
      trace,
      best_row,
      best_column,
      lane_count,
      query_start,
      target_start);
  return make_alignment_path(
      score,
      query_start,
      best_row,
      target_start,
      best_column,
      operations);
}

inline LinearTracebackResult build_linear_traceback_from_masked_trace(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    const LinearMaskedTrace& trace,
    Score score,
    std::size_t best_row,
    std::size_t best_column,
    std::size_t lane_count) {
  LinearTracebackResult result;
  result.score = score;
  result.query_end = best_row;
  result.target_end = best_column;
  result.query_start = best_row;
  result.target_start = best_column;
  result.operations = build_linear_operations_from_masked_trace(
      query,
      target,
      trace,
      best_row,
      best_column,
      lane_count,
      result.query_start,
      result.target_start);
  return result;
}

inline LinearCigarTrace build_linear_cigar_trace_from_masked_trace(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    const LinearMaskedTrace& trace,
    Score score,
    std::size_t best_row,
    std::size_t best_column,
    std::size_t lane_count);

inline std::string build_linear_cigar_from_masked_trace(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    const LinearMaskedTrace& trace,
    std::size_t best_row,
    std::size_t best_column,
    std::size_t lane_count) {
  return build_linear_cigar_trace_from_masked_trace(
      query,
      target,
      trace,
      0,
      best_row,
      best_column,
      lane_count)
      .cigar;
}

inline LinearCigarTrace build_linear_cigar_trace_from_masked_trace(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    const LinearMaskedTrace& trace,
    Score score,
    std::size_t best_row,
    std::size_t best_column,
    std::size_t lane_count) {
  LinearCigarTrace result;
  result.score = score;
  result.query_end = best_row;
  result.target_end = best_column;

  ReverseCigarBuilder cigar;

  std::size_t row = best_row;
  std::size_t column = best_column;
  while (row > 0 && column > 0) {
    const auto direction = trace.direction(row, column, lane_count);
    if (direction == TraceDirection::stop) {
      break;
    }
    if (direction == TraceDirection::diagonal) {
      cigar.push(query[row - 1U] == target[column - 1U] ? 'M' : 'X');
      --row;
      --column;
      continue;
    }
    if (direction == TraceDirection::up) {
      cigar.push('D');
      --row;
      continue;
    }
    if (direction == TraceDirection::left) {
      cigar.push('I');
      --column;
      continue;
    }
    break;
  }

  result.query_start = row;
  result.target_start = column;
  result.cigar = cigar.str();
  return result;
}

template <
    template <typename, typename> class OpsTemplate,
    typename Cell,
    LinearMaskedTraceOutput Output,
    bool ScoreKnown = false>
auto linear_sw_masked_trace_state(
    PreparedScoreState<Cell>& state,
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    Score known_score = 0) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;

  if (query.empty() || target.empty() || state.segment_count == 0) {
    if constexpr (Output == LinearMaskedTraceOutput::cigar) {
      return std::string();
    } else if constexpr (Output == LinearMaskedTraceOutput::cigar_trace) {
      return LinearCigarTrace{};
    } else if constexpr (Output == LinearMaskedTraceOutput::traceback) {
      return LinearTracebackResult{};
    } else {
      return make_alignment_path(0, 0, 0, 0, 0, "");
    }
  }

  LinearMaskedTrace trace(target.size(), state.segment_count);
  const auto valid_masks = make_striped_valid_lane_masks(
      state.query_size,
      state.segment_count,
      lane_count);

  std::fill(state.h_store.begin(), state.h_store.end(), Cell{0});
  std::fill(state.h_load.begin(), state.h_load.end(), Cell{0});
  std::fill(state.e_store.begin(), state.e_store.end(), Cell{0});

  const auto zero_vector = Ops::zero();
  const auto gap_vector = Ops::set1(state.gap_score);
  [[maybe_unused]] const auto known_score_vector = Ops::set1(static_cast<Cell>(known_score));
  Cell* h_store_data = state.h_store.data();
  Cell* h_load_data = state.h_load.data();
  Cell* e_store_data = state.e_store.data();
  const Cell* profile_data = state.profile.data();
  LinearLocalEndpoint best;

  for (std::size_t target_index = 0; target_index < state.target_profile_offsets.size();
       ++target_index) {
    const std::size_t column = target_index + 1U;
    std::swap(h_store_data, h_load_data);

    auto v_h = shift_left_zero<Ops, Cell>(
        load_state_cells<Ops, Cell>(h_load_data + ((state.segment_count - 1U) * lane_count)));
    auto v_f = zero_vector;
    const Cell* profile_row = profile_data + state.target_profile_offsets[target_index];

    for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
      Cell* h_store_segment = h_store_data + segment * lane_count;
      Cell* h_load_segment = h_load_data + segment * lane_count;
      Cell* e_segment = e_store_data + segment * lane_count;
      const Cell* profile_segment = profile_row + segment * lane_count;
      const std::uint64_t valid_mask = valid_masks[segment];

      const auto v_profile = load_state_cells<Ops, Cell>(profile_segment);
      auto v_e = load_state_cells<Ops, Cell>(e_segment);
      const auto v_diagonal = Ops::add(v_h, v_profile);
      const std::uint64_t up_better_mask =
          trace_mask_gt<Ops>(v_f, v_diagonal) & valid_mask;
      auto v_selected = Ops::max(v_diagonal, v_f);
      const std::uint64_t left_better_mask =
          trace_mask_gt<Ops>(v_e, v_selected) & valid_mask;
      v_selected = Ops::max(v_selected, v_e);
      auto v_cell = Ops::max(v_selected, zero_vector);
      store_state_cells<Ops, Cell>(h_store_segment, v_cell);

      const std::uint64_t positive_mask =
          trace_mask_gt<Ops>(v_cell, zero_vector) & valid_mask;
      const std::uint64_t left_mask = positive_mask & left_better_mask;
      const std::uint64_t up_mask = positive_mask & up_better_mask & ~left_better_mask;
      const std::uint64_t low_mask = positive_mask & ~up_mask;
      const std::uint64_t high_mask = up_mask | left_mask;
      trace.set(column, segment, low_mask, high_mask);

      if constexpr (ScoreKnown) {
        const std::uint64_t endpoint_mask =
            trace_mask_eq<Ops>(v_cell, known_score_vector) & positive_mask;
        update_linear_local_endpoint_for_score_from_mask<Ops, Cell>(
            endpoint_mask,
            segment,
            column,
            state.query_size,
            state.segment_count,
            known_score,
            best);
      } else {
        update_linear_local_endpoint_from_vector<Ops, Cell>(
            v_cell,
            positive_mask,
            segment,
            column,
            state.query_size,
            state.segment_count,
            best);
      }

      const auto v_h_gap = Ops::add(v_cell, gap_vector);
      v_e = Ops::max(Ops::add(v_e, gap_vector), v_h_gap);
      store_state_cells<Ops, Cell>(e_segment, v_e);
      v_f = Ops::max(Ops::add(v_f, gap_vector), v_h_gap);
      v_h = load_state_cells<Ops, Cell>(h_load_segment);
    }

    for (std::size_t iteration = 0; iteration < lane_count; ++iteration) {
      v_f = shift_left_zero<Ops, Cell>(v_f);
      bool propagated = false;

      for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
        Cell* h_store_segment = h_store_data + segment * lane_count;
        const std::uint64_t valid_mask = valid_masks[segment];
        const auto v_h_previous = load_state_cells<Ops, Cell>(h_store_segment);
        const std::uint64_t propagated_mask =
            trace_mask_gt<Ops>(v_f, v_h_previous) & valid_mask;
        const std::uint64_t tied_up_mask =
            trace_mask_eq<Ops>(v_f, v_h_previous) &
            trace_mask_gt<Ops>(v_f, zero_vector) &
            valid_mask;
        propagated = propagated || propagated_mask != 0;
        auto v_h_updated = Ops::max(v_h_previous, v_f);
        store_state_cells<Ops, Cell>(h_store_segment, v_h_updated);

        if (propagated_mask != 0) {
          trace.force_up(column, segment, propagated_mask);
          if constexpr (ScoreKnown) {
            const std::uint64_t endpoint_mask =
                trace_mask_eq<Ops>(v_h_updated, known_score_vector) & propagated_mask;
            update_linear_local_endpoint_for_score_from_mask<Ops, Cell>(
                endpoint_mask,
                segment,
                column,
                state.query_size,
                state.segment_count,
                known_score,
                best);
          } else {
            update_linear_local_endpoint_from_vector<Ops, Cell>(
                v_h_updated,
                propagated_mask,
                segment,
                column,
                state.query_size,
                state.segment_count,
                best);
          }
        }
        if (tied_up_mask != 0) {
          trace.force_up_when_not_diagonal(column, segment, tied_up_mask);
        }

        v_f = Ops::add(v_h_updated, gap_vector);
      }

      if (state.gap_score <= 0 && !propagated) {
        break;
      }
    }
  }

  if (best.score <= 0) {
    if constexpr (Output == LinearMaskedTraceOutput::cigar) {
      return std::string();
    } else if constexpr (Output == LinearMaskedTraceOutput::cigar_trace) {
      return LinearCigarTrace{};
    } else if constexpr (Output == LinearMaskedTraceOutput::traceback) {
      return LinearTracebackResult{};
    } else {
      return make_alignment_path(0, 0, 0, 0, 0, "");
    }
  }

  if constexpr (Output == LinearMaskedTraceOutput::cigar) {
    return build_linear_cigar_from_masked_trace(
        query,
        target,
        trace,
        best.row,
        best.column,
        lane_count);
  } else if constexpr (Output == LinearMaskedTraceOutput::cigar_trace) {
    return build_linear_cigar_trace_from_masked_trace(
        query,
        target,
        trace,
        best.score,
        best.row,
        best.column,
        lane_count);
  } else if constexpr (Output == LinearMaskedTraceOutput::traceback) {
    return build_linear_traceback_from_masked_trace(
        query,
        target,
        trace,
        best.score,
        best.row,
        best.column,
        lane_count);
  } else {
    return build_linear_path_from_masked_trace(
        query,
        target,
        trace,
        best.score,
        best.row,
        best.column,
        lane_count);
  }
}

template <template <typename, typename> class OpsTemplate, typename Cell>
AlignmentPath linear_sw_masked_path_info(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  auto state = prepare_score_state<OpsTemplate, Cell, false>(
      prepared,
      match_score,
      mismatch_score,
      gap_score);
  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());
  return linear_sw_masked_trace_state<
      OpsTemplate,
      Cell,
      LinearMaskedTraceOutput::path_info>(state, query, target);
}

template <template <typename, typename> class OpsTemplate, typename Cell>
LinearTracebackResult linear_sw_masked_traceback(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  auto state = prepare_score_state<OpsTemplate, Cell, false>(
      prepared,
      match_score,
      mismatch_score,
      gap_score);
  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());
  return linear_sw_masked_trace_state<
      OpsTemplate,
      Cell,
      LinearMaskedTraceOutput::traceback>(state, query, target);
}

template <template <typename, typename> class OpsTemplate, typename Cell>
std::string linear_sw_masked_cigar(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  auto state = prepare_score_state<OpsTemplate, Cell, false>(
      prepared,
      match_score,
      mismatch_score,
      gap_score);
  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());
  return linear_sw_masked_trace_state<
      OpsTemplate,
      Cell,
      LinearMaskedTraceOutput::cigar>(state, query, target);
}

template <template <typename, typename> class OpsTemplate, typename Cell>
LinearCigarTrace linear_sw_masked_cigar_trace(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  auto state = prepare_score_state<OpsTemplate, Cell, false>(
      prepared,
      match_score,
      mismatch_score,
      gap_score);
  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());
  return linear_sw_masked_trace_state<
      OpsTemplate,
      Cell,
      LinearMaskedTraceOutput::cigar_trace>(state, query, target);
}

template <template <typename, typename> class OpsTemplate, typename Cell>
AlignmentPath linear_sw_masked_cigar_path_info(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  const auto trace = linear_sw_masked_cigar_trace<OpsTemplate, Cell>(
      prepared,
      match_score,
      mismatch_score,
      gap_score);
  return make_alignment_path_from_cigar(
      trace.score,
      trace.query_start,
      trace.query_end,
      trace.target_start,
      trace.target_end,
      trace.cigar);
}

template <template <typename, typename> class OpsTemplate, typename Cell>
LinearCigarTrace linear_sw_score_first_masked_cigar_trace(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  auto state = prepare_score_state<OpsTemplate, Cell, false>(
      prepared,
      match_score,
      mismatch_score,
      gap_score);
  const Score score = score_state<OpsTemplate, Cell>(state);
  if (score <= 0) {
    return LinearCigarTrace{};
  }

  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());
  return linear_sw_masked_trace_state<
      OpsTemplate,
      Cell,
      LinearMaskedTraceOutput::cigar_trace,
      true>(state, query, target, score);
}

template <template <typename, typename> class OpsTemplate, typename Cell>
std::string linear_sw_score_first_masked_cigar(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  return linear_sw_score_first_masked_cigar_trace<OpsTemplate, Cell>(
      prepared,
      match_score,
      mismatch_score,
      gap_score)
      .cigar;
}

template <template <typename, typename> class OpsTemplate, typename Cell>
AlignmentPath linear_sw_score_first_masked_cigar_path_info(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  const auto trace = linear_sw_score_first_masked_cigar_trace<OpsTemplate, Cell>(
      prepared,
      match_score,
      mismatch_score,
      gap_score);
  return make_alignment_path_from_cigar(
      trace.score,
      trace.query_start,
      trace.query_end,
      trace.target_start,
      trace.target_end,
      trace.cigar);
}

template <template <typename, typename> class OpsTemplate>
AlignmentPath dispatch_linear_sw_masked_path_info(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return linear_sw_masked_path_info<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits16:
      return linear_sw_masked_path_info<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits32:
      return linear_sw_masked_path_info<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits64:
      return linear_sw_masked_path_info<OpsTemplate, std::int64_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported masked linear SW traceback width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate>
LinearTracebackResult dispatch_linear_sw_masked_traceback(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return linear_sw_masked_traceback<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits16:
      return linear_sw_masked_traceback<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits32:
      return linear_sw_masked_traceback<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits64:
      return linear_sw_masked_traceback<OpsTemplate, std::int64_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported masked linear SW traceback width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate>
std::string dispatch_linear_sw_masked_cigar(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return linear_sw_masked_cigar<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits16:
      return linear_sw_masked_cigar<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits32:
      return linear_sw_masked_cigar<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits64:
      return linear_sw_masked_cigar<OpsTemplate, std::int64_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported masked linear SW CIGAR width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate>
LinearCigarTrace dispatch_linear_sw_masked_cigar_trace(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return linear_sw_masked_cigar_trace<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits16:
      return linear_sw_masked_cigar_trace<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits32:
      return linear_sw_masked_cigar_trace<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits64:
      return linear_sw_masked_cigar_trace<OpsTemplate, std::int64_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported masked linear SW CIGAR width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate>
AlignmentPath dispatch_linear_sw_masked_cigar_path_info(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return linear_sw_masked_cigar_path_info<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits16:
      return linear_sw_masked_cigar_path_info<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits32:
      return linear_sw_masked_cigar_path_info<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits64:
      return linear_sw_masked_cigar_path_info<OpsTemplate, std::int64_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported masked linear SW path-info width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate>
LinearCigarTrace dispatch_linear_sw_score_first_masked_cigar_trace(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return linear_sw_score_first_masked_cigar_trace<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits16:
      return linear_sw_score_first_masked_cigar_trace<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits32:
      return linear_sw_score_first_masked_cigar_trace<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits64:
      return linear_sw_score_first_masked_cigar_trace<OpsTemplate, std::int64_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported score-first masked linear SW CIGAR width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate>
std::string dispatch_linear_sw_score_first_masked_cigar(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return linear_sw_score_first_masked_cigar<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits16:
      return linear_sw_score_first_masked_cigar<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits32:
      return linear_sw_score_first_masked_cigar<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits64:
      return linear_sw_score_first_masked_cigar<OpsTemplate, std::int64_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported score-first masked linear SW CIGAR width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate>
AlignmentPath dispatch_linear_sw_score_first_masked_cigar_path_info(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return linear_sw_score_first_masked_cigar_path_info<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits16:
      return linear_sw_score_first_masked_cigar_path_info<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits32:
      return linear_sw_score_first_masked_cigar_path_info<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits64:
      return linear_sw_score_first_masked_cigar_path_info<OpsTemplate, std::int64_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported score-first masked linear SW path-info width");
  throw nb::python_error();
}

template <typename Cell>
std::string build_linear_cigar_from_striped_trace(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target,
    const std::vector<std::uint8_t>& trace,
    std::size_t best_row,
    std::size_t best_column,
    std::size_t segment_count,
    std::size_t lane_count) {
  const std::size_t state_cell_count = segment_count * lane_count;
  ReverseCigarBuilder cigar;

  std::size_t row = best_row;
  std::size_t column = best_column;
  while (row > 0 && column > 0) {
    const auto direction = static_cast<TraceDirection>(
        trace[trace_striped_index(row, column, segment_count, lane_count, state_cell_count)]);
    if (direction == TraceDirection::stop) {
      break;
    }
    if (direction == TraceDirection::diagonal) {
      cigar.push(query[row - 1U] == target[column - 1U] ? 'M' : 'X');
      --row;
      --column;
      continue;
    }
    if (direction == TraceDirection::up) {
      cigar.push('D');
      --row;
      continue;
    }
    if (direction == TraceDirection::left) {
      cigar.push('I');
      --column;
      continue;
    }
    break;
  }

  return cigar.str();
}

template <template <typename, typename> class OpsTemplate, typename Cell>
std::string linear_sw_striped_cigar_state(
    PreparedScoreState<Cell>& state,
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> target) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;

  if (query.empty() || target.empty() || state.segment_count == 0) {
    return "";
  }

  const std::size_t state_cell_count = state.segment_count * lane_count;
  std::vector<std::uint8_t> trace(
      (target.size() + 1U) * state_cell_count,
      static_cast<std::uint8_t>(TraceDirection::stop));

  std::fill(state.h_store.begin(), state.h_store.end(), Cell{0});
  std::fill(state.h_load.begin(), state.h_load.end(), Cell{0});
  std::fill(state.e_store.begin(), state.e_store.end(), Cell{0});

  const auto zero_vector = Ops::zero();
  const auto gap_vector = Ops::set1(state.gap_score);
  Cell* h_store_data = state.h_store.data();
  Cell* h_load_data = state.h_load.data();
  Cell* e_store_data = state.e_store.data();
  const Cell* profile_data = state.profile.data();
  LinearLocalEndpoint best;

  alignas(Ops::alignment) Cell diagonal_scores[lane_count] = {};
  alignas(Ops::alignment) Cell e_scores[lane_count] = {};
  alignas(Ops::alignment) Cell f_scores[lane_count] = {};
  alignas(Ops::alignment) Cell h_scores[lane_count] = {};
  alignas(Ops::alignment) Cell previous_h_scores[lane_count] = {};

  for (std::size_t target_index = 0; target_index < state.target_profile_offsets.size();
       ++target_index) {
    const std::size_t column = target_index + 1U;
    std::swap(h_store_data, h_load_data);

    auto v_h = shift_left_zero<Ops, Cell>(
        load_state_cells<Ops, Cell>(h_load_data + ((state.segment_count - 1U) * lane_count)));
    auto v_f = zero_vector;
    const Cell* profile_row = profile_data + state.target_profile_offsets[target_index];

    for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
      Cell* h_store_segment = h_store_data + segment * lane_count;
      Cell* h_load_segment = h_load_data + segment * lane_count;
      Cell* e_segment = e_store_data + segment * lane_count;
      const Cell* profile_segment = profile_row + segment * lane_count;

      const auto v_profile = load_state_cells<Ops, Cell>(profile_segment);
      auto v_e = load_state_cells<Ops, Cell>(e_segment);
      const auto v_diagonal = Ops::add(v_h, v_profile);
      auto v_cell = Ops::max(v_diagonal, v_e);
      v_cell = Ops::max(v_cell, v_f);
      v_cell = Ops::max(v_cell, zero_vector);
      store_state_cells<Ops, Cell>(h_store_segment, v_cell);

      Ops::store_cells(diagonal_scores, v_diagonal);
      Ops::store_cells(e_scores, v_e);
      Ops::store_cells(f_scores, v_f);
      Ops::store_cells(h_scores, v_cell);

      for (std::size_t lane = 0; lane < lane_count; ++lane) {
        const std::size_t query_index = lane * state.segment_count + segment;
        if (query_index >= query.size()) {
          continue;
        }

        Cell selected_score = diagonal_scores[lane];
        TraceDirection direction = TraceDirection::diagonal;
        if (f_scores[lane] > selected_score) {
          selected_score = f_scores[lane];
          direction = TraceDirection::up;
        }
        if (e_scores[lane] > selected_score) {
          selected_score = e_scores[lane];
          direction = TraceDirection::left;
        }
        if (selected_score <= 0) {
          direction = TraceDirection::stop;
        }

        const std::size_t row = query_index + 1U;
        trace[trace_striped_index(
            row,
            column,
            state.segment_count,
            lane_count,
            state_cell_count)] = static_cast<std::uint8_t>(direction);
      }

      update_linear_local_endpoint_from_lanes<Ops, Cell>(
          h_scores,
          segment,
          column,
          state.query_size,
          state.segment_count,
          best);

      const auto v_h_gap = Ops::add(v_cell, gap_vector);
      v_e = Ops::max(Ops::add(v_e, gap_vector), v_h_gap);
      store_state_cells<Ops, Cell>(e_segment, v_e);
      v_f = Ops::max(Ops::add(v_f, gap_vector), v_h_gap);
      v_h = load_state_cells<Ops, Cell>(h_load_segment);
    }

    for (std::size_t iteration = 0; iteration < lane_count; ++iteration) {
      v_f = shift_left_zero<Ops, Cell>(v_f);
      bool propagated = false;

      for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
        Cell* h_store_segment = h_store_data + segment * lane_count;
        auto v_h_previous = load_state_cells<Ops, Cell>(h_store_segment);
        propagated = propagated || any_greater<Ops, Cell>(v_f, v_h_previous);
        auto v_h_updated = Ops::max(v_h_previous, v_f);
        store_state_cells<Ops, Cell>(h_store_segment, v_h_updated);

        Ops::store_cells(previous_h_scores, v_h_previous);
        Ops::store_cells(f_scores, v_f);
        Ops::store_cells(h_scores, v_h_updated);
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
          const std::size_t query_index = lane * state.segment_count + segment;
          if (query_index >= query.size()) {
            continue;
          }
          if (f_scores[lane] > previous_h_scores[lane]) {
            const std::size_t row = query_index + 1U;
            trace[trace_striped_index(
                row,
                column,
                state.segment_count,
                lane_count,
                state_cell_count)] = static_cast<std::uint8_t>(TraceDirection::up);
          }
        }

        update_linear_local_endpoint_from_lanes<Ops, Cell>(
            h_scores,
            segment,
            column,
            state.query_size,
            state.segment_count,
            best);

        v_f = Ops::add(v_f, gap_vector);
      }

      if (state.gap_score <= 0 && !propagated) {
        break;
      }
    }
  }

  if (best.score <= 0) {
    return "";
  }

  return build_linear_cigar_from_striped_trace<Cell>(
      query,
      target,
      trace,
      best.row,
      best.column,
      state.segment_count,
      lane_count);
}

template <template <typename, typename> class OpsTemplate, typename Cell>
std::string linear_sw_striped_cigar(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  auto state = prepare_score_state<OpsTemplate, Cell, false>(
      prepared,
      match_score,
      mismatch_score,
      gap_score);
  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());
  return linear_sw_striped_cigar_state<OpsTemplate, Cell>(state, query, target);
}

template <template <typename, typename> class OpsTemplate>
std::string dispatch_linear_sw_striped_cigar(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return linear_sw_striped_cigar<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits16:
      return linear_sw_striped_cigar<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits32:
      return linear_sw_striped_cigar<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits64:
      return linear_sw_striped_cigar<OpsTemplate, std::int64_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported linear SW striped CIGAR width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate, typename Cell>
LinearLocalEndpoint linear_sw_checkpoint_forward(
    PreparedScoreState<Cell>& state,
    LinearCigarCheckpoints<Cell>& checkpoints) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;

  LinearLocalEndpoint best;
  const std::size_t state_cell_count = state.segment_count * lane_count;
  std::fill(state.h_store.begin(), state.h_store.end(), Cell{0});
  std::fill(state.h_load.begin(), state.h_load.end(), Cell{0});
  std::fill(state.e_store.begin(), state.e_store.end(), Cell{0});

  const auto zero_vector = Ops::zero();
  const auto gap_vector = Ops::set1(state.gap_score);
  Cell* h_store_data = state.h_store.data();
  Cell* h_load_data = state.h_load.data();
  Cell* e_store_data = state.e_store.data();
  const Cell* profile_data = state.profile.data();
  alignas(Ops::alignment) Cell h_scores[lane_count] = {};

  save_linear_cigar_checkpoint(checkpoints, 0U, h_store_data, e_store_data);

  for (std::size_t target_index = 0; target_index < state.target_profile_offsets.size();
       ++target_index) {
    const std::size_t column = target_index + 1U;
    std::swap(h_store_data, h_load_data);

    auto v_h = shift_left_zero<Ops, Cell>(
        load_state_cells<Ops, Cell>(h_load_data + ((state.segment_count - 1U) * lane_count)));
    auto v_f = zero_vector;
    const Cell* profile_row = profile_data + state.target_profile_offsets[target_index];

    for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
      Cell* h_store_segment = h_store_data + segment * lane_count;
      Cell* h_load_segment = h_load_data + segment * lane_count;
      Cell* e_segment = e_store_data + segment * lane_count;
      const Cell* profile_segment = profile_row + segment * lane_count;

      const auto v_profile = load_state_cells<Ops, Cell>(profile_segment);
      auto v_e = load_state_cells<Ops, Cell>(e_segment);
      v_h = Ops::add(v_h, v_profile);
      v_h = Ops::max(v_h, v_e);
      v_h = Ops::max(v_h, v_f);
      v_h = Ops::max(v_h, zero_vector);
      store_state_cells<Ops, Cell>(h_store_segment, v_h);

      Ops::store_cells(h_scores, v_h);
      update_linear_local_endpoint_from_lanes<Ops, Cell>(
          h_scores,
          segment,
          column,
          state.query_size,
          state.segment_count,
          best);

      const auto v_h_gap = Ops::add(v_h, gap_vector);
      v_e = Ops::max(Ops::add(v_e, gap_vector), v_h_gap);
      store_state_cells<Ops, Cell>(e_segment, v_e);
      v_f = Ops::max(Ops::add(v_f, gap_vector), v_h_gap);
      v_h = load_state_cells<Ops, Cell>(h_load_segment);
    }

    for (std::size_t iteration = 0; iteration < lane_count; ++iteration) {
      v_f = shift_left_zero<Ops, Cell>(v_f);
      bool propagated = false;

      for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
        Cell* h_store_segment = h_store_data + segment * lane_count;
        auto v_h_segment = load_state_cells<Ops, Cell>(h_store_segment);
        propagated = propagated || any_greater<Ops, Cell>(v_f, v_h_segment);
        v_h_segment = Ops::max(v_h_segment, v_f);
        store_state_cells<Ops, Cell>(h_store_segment, v_h_segment);

        Ops::store_cells(h_scores, v_h_segment);
        update_linear_local_endpoint_from_lanes<Ops, Cell>(
            h_scores,
            segment,
            column,
            state.query_size,
            state.segment_count,
            best);

        v_f = Ops::add(v_f, gap_vector);
      }

      if (state.gap_score <= 0 && !propagated) {
        break;
      }
    }

    if (column % checkpoints.block_size == 0) {
      save_linear_cigar_checkpoint(
          checkpoints,
          column / checkpoints.block_size,
          h_store_data,
          e_store_data);
    }
  }

  if (state.target_size % checkpoints.block_size != 0) {
    save_linear_cigar_checkpoint(
        checkpoints,
        checkpoints.checkpoint_count - 1U,
        h_store_data,
        e_store_data);
  }

  (void) state_cell_count;
  return best;
}

template <template <typename, typename> class OpsTemplate, typename Cell>
void linear_sw_recompute_trace_block(
    PreparedScoreState<Cell>& state,
    const LinearCigarCheckpoints<Cell>& checkpoints,
    std::size_t block_start,
    std::size_t block_end,
    std::vector<std::uint8_t>& trace) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;
  const std::size_t state_cell_count = state.segment_count * lane_count;
  const std::size_t block_width = block_end - block_start;

  trace.resize((block_width + 1U) * state_cell_count);
  const std::size_t checkpoint_index = block_start / checkpoints.block_size;
  std::copy_n(checkpoints.h_at(checkpoint_index), state_cell_count, state.h_store.data());
  std::copy_n(checkpoints.e_at(checkpoint_index), state_cell_count, state.e_store.data());

  const auto zero_vector = Ops::zero();
  const auto gap_vector = Ops::set1(state.gap_score);
  Cell* h_store_data = state.h_store.data();
  Cell* h_load_data = state.h_load.data();
  Cell* e_store_data = state.e_store.data();
  const Cell* profile_data = state.profile.data();

  alignas(Ops::alignment) Cell diagonal_scores[lane_count] = {};
  alignas(Ops::alignment) Cell e_scores[lane_count] = {};
  alignas(Ops::alignment) Cell f_scores[lane_count] = {};
  alignas(Ops::alignment) Cell h_scores[lane_count] = {};
  alignas(Ops::alignment) Cell previous_h_scores[lane_count] = {};

  for (std::size_t target_index = block_start; target_index < block_end; ++target_index) {
    const std::size_t local_column = target_index - block_start + 1U;
    std::swap(h_store_data, h_load_data);

    auto v_h = shift_left_zero<Ops, Cell>(
        load_state_cells<Ops, Cell>(h_load_data + ((state.segment_count - 1U) * lane_count)));
    auto v_f = zero_vector;
    const Cell* profile_row = profile_data + state.target_profile_offsets[target_index];

    for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
      Cell* h_store_segment = h_store_data + segment * lane_count;
      Cell* h_load_segment = h_load_data + segment * lane_count;
      Cell* e_segment = e_store_data + segment * lane_count;
      const Cell* profile_segment = profile_row + segment * lane_count;

      const auto v_profile = load_state_cells<Ops, Cell>(profile_segment);
      auto v_e = load_state_cells<Ops, Cell>(e_segment);
      const auto v_diagonal = Ops::add(v_h, v_profile);
      auto v_cell = Ops::max(v_diagonal, v_e);
      v_cell = Ops::max(v_cell, v_f);
      v_cell = Ops::max(v_cell, zero_vector);
      store_state_cells<Ops, Cell>(h_store_segment, v_cell);

      Ops::store_cells(diagonal_scores, v_diagonal);
      Ops::store_cells(e_scores, v_e);
      Ops::store_cells(f_scores, v_f);
      Ops::store_cells(h_scores, v_cell);

      for (std::size_t lane = 0; lane < lane_count; ++lane) {
        const std::size_t query_index = lane * state.segment_count + segment;
        if (query_index >= state.query_size) {
          continue;
        }

        Cell selected_score = diagonal_scores[lane];
        TraceDirection direction = TraceDirection::diagonal;
        if (f_scores[lane] > selected_score) {
          selected_score = f_scores[lane];
          direction = TraceDirection::up;
        }
        if (e_scores[lane] > selected_score) {
          selected_score = e_scores[lane];
          direction = TraceDirection::left;
        }
        if (selected_score <= 0) {
          direction = TraceDirection::stop;
        }

        const std::size_t row = query_index + 1U;
        trace[trace_striped_index(
            row,
            local_column,
            state.segment_count,
            lane_count,
            state_cell_count)] = static_cast<std::uint8_t>(direction);
      }

      const auto v_h_gap = Ops::add(v_cell, gap_vector);
      v_e = Ops::max(Ops::add(v_e, gap_vector), v_h_gap);
      store_state_cells<Ops, Cell>(e_segment, v_e);
      v_f = Ops::max(Ops::add(v_f, gap_vector), v_h_gap);
      v_h = load_state_cells<Ops, Cell>(h_load_segment);
    }

    for (std::size_t iteration = 0; iteration < lane_count; ++iteration) {
      v_f = shift_left_zero<Ops, Cell>(v_f);
      bool propagated = false;

      for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
        Cell* h_store_segment = h_store_data + segment * lane_count;
        auto v_h_previous = load_state_cells<Ops, Cell>(h_store_segment);
        propagated = propagated || any_greater<Ops, Cell>(v_f, v_h_previous);
        auto v_h_updated = Ops::max(v_h_previous, v_f);
        store_state_cells<Ops, Cell>(h_store_segment, v_h_updated);

        Ops::store_cells(previous_h_scores, v_h_previous);
        Ops::store_cells(f_scores, v_f);
        Ops::store_cells(h_scores, v_h_updated);
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
          const std::size_t query_index = lane * state.segment_count + segment;
          if (query_index >= state.query_size) {
            continue;
          }
          if (f_scores[lane] > previous_h_scores[lane]) {
            const std::size_t row = query_index + 1U;
            trace[trace_striped_index(
                row,
                local_column,
                state.segment_count,
                lane_count,
                state_cell_count)] = static_cast<std::uint8_t>(TraceDirection::up);
          }
        }

        v_f = Ops::add(v_f, gap_vector);
      }

      if (state.gap_score <= 0 && !propagated) {
        break;
      }
    }
  }
}

template <template <typename, typename> class OpsTemplate, typename Cell>
std::string linear_sw_checkpointed_cigar(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  if (prepared.query_tokens.empty() || prepared.target_tokens.empty()) {
    return "";
  }

  auto state = prepare_score_state<OpsTemplate, Cell, false>(
      prepared,
      match_score,
      mismatch_score,
      gap_score);
  if (state.segment_count == 0 || state.target_profile_offsets.empty()) {
    return "";
  }

  auto checkpoints = make_linear_cigar_checkpoints<Cell>(
      state.query_size,
      state.target_size,
      state.segment_count * ScoreOps<OpsTemplate, Cell>::lane_count);
  const LinearLocalEndpoint endpoint =
      linear_sw_checkpoint_forward<OpsTemplate, Cell>(state, checkpoints);
  if (endpoint.score <= 0) {
    return "";
  }

  ReverseCigarBuilder cigar;
  std::vector<std::uint8_t> trace;
  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;
  const std::size_t state_cell_count = state.segment_count * lane_count;

  std::size_t row = endpoint.row;
  std::size_t column = endpoint.column;
  bool done = false;
  while (row > 0 && column > 0 && !done) {
    const std::size_t block_start =
        ((column - 1U) / checkpoints.block_size) * checkpoints.block_size;
    const std::size_t block_end = std::min(block_start + checkpoints.block_size, target.size());
    linear_sw_recompute_trace_block<OpsTemplate, Cell>(
        state,
        checkpoints,
        block_start,
        block_end,
        trace);

    while (row > 0 && column > block_start) {
      const std::size_t local_column = column - block_start;
      const auto direction = static_cast<TraceDirection>(
          trace[trace_striped_index(
              row,
              local_column,
              state.segment_count,
              lane_count,
              state_cell_count)]);
      if (direction == TraceDirection::stop) {
        done = true;
        break;
      }
      if (direction == TraceDirection::diagonal) {
        cigar.push(query[row - 1U] == target[column - 1U] ? 'M' : 'X');
        --row;
        --column;
        continue;
      }
      if (direction == TraceDirection::up) {
        cigar.push('D');
        --row;
        continue;
      }
      if (direction == TraceDirection::left) {
        cigar.push('I');
        --column;
        continue;
      }
      done = true;
      break;
    }

    if (block_start == 0 && column == 0) {
      break;
    }
  }

  return cigar.str();
}

template <template <typename, typename> class OpsTemplate>
std::string dispatch_linear_sw_checkpointed_cigar(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return linear_sw_checkpointed_cigar<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits16:
      return linear_sw_checkpointed_cigar<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits32:
      return linear_sw_checkpointed_cigar<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits64:
      return linear_sw_checkpointed_cigar<OpsTemplate, std::int64_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported checkpointed SW CIGAR width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate>
PreparedScore<OpsTemplate> prepare_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score,
    ScoreProfileLayout profile_layout = ScoreProfileLayout::automatic,
    std::size_t profile_block_size = 64U,
    ScoreKernelStrategy kernel_strategy = ScoreKernelStrategy::automatic) {
  PreparedScore<OpsTemplate> output;
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      output.state = prepare_score_state<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score,
          profile_layout,
          profile_block_size,
          kernel_strategy);
      return output;
    case KernelBits::bits16:
      output.state = prepare_score_state<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score,
          profile_layout,
          profile_block_size,
          kernel_strategy);
      return output;
    case KernelBits::bits32:
      output.state = prepare_score_state<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score,
          profile_layout,
          profile_block_size,
          kernel_strategy);
      return output;
    case KernelBits::bits64:
      output.state = prepare_score_state<OpsTemplate, std::int64_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score,
          profile_layout,
          profile_block_size,
          kernel_strategy);
      return output;
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported Farrar score width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate, bool PrepareGlobalInitial = false>
PreparedAffineScore<OpsTemplate> prepare_affine_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  PreparedAffineScore<OpsTemplate> output;
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      output.state = prepare_affine_score_state<OpsTemplate, std::int8_t, PrepareGlobalInitial>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return output;
    case KernelBits::bits16:
      output.state = prepare_affine_score_state<OpsTemplate, std::int16_t, PrepareGlobalInitial>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return output;
    case KernelBits::bits32:
      output.state = prepare_affine_score_state<OpsTemplate, std::int32_t, PrepareGlobalInitial>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return output;
    case KernelBits::bits64:
      output.state = prepare_affine_score_state<OpsTemplate, std::int64_t, PrepareGlobalInitial>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return output;
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported affine Farrar score width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate>
Score dispatch_prepared_score(PreparedScore<OpsTemplate>& prepared) {
  return std::visit(
      [](auto& state) -> Score {
        using State = std::decay_t<decltype(state)>;
        using Cell = typename State::cell_type;
        return score_state<OpsTemplate, Cell>(state);
      },
      prepared.state);
}

template <template <typename, typename> class OpsTemplate>
Score dispatch_prepared_affine_score(PreparedAffineScore<OpsTemplate>& prepared) {
  return std::visit(
      [](auto& state) -> Score {
        using State = std::decay_t<decltype(state)>;
        using Cell = typename State::cell_type;
        return affine_score_state<OpsTemplate, Cell>(state);
      },
      prepared.state);
}

template <template <typename, typename> class OpsTemplate>
Score dispatch_prepared_global_affine_score(PreparedAffineScore<OpsTemplate>& prepared) {
  return std::visit(
      [](auto& state) -> Score {
        using State = std::decay_t<decltype(state)>;
        using Cell = typename State::cell_type;
        return global_affine_score_state<OpsTemplate, Cell>(state);
      },
      prepared.state);
}

template <template <typename, typename> class OpsTemplate, bool LocalAlignment>
PreparedScoreBatch<OpsTemplate> prepare_score_batch(
    const PreparedFarrarBatchAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score,
    ScoreProfileLayout profile_layout = ScoreProfileLayout::automatic,
    std::size_t profile_block_size = 64U,
    ScoreKernelStrategy kernel_strategy = ScoreKernelStrategy::automatic) {
  PreparedScoreBatch<OpsTemplate> output;
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      output.state = prepare_score_batch_state<OpsTemplate, std::int8_t, LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_score,
          profile_layout,
          profile_block_size,
          kernel_strategy);
      return output;
    case KernelBits::bits16:
      output.state = prepare_score_batch_state<OpsTemplate, std::int16_t, LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_score,
          profile_layout,
          profile_block_size,
          kernel_strategy);
      return output;
    case KernelBits::bits32:
      output.state = prepare_score_batch_state<OpsTemplate, std::int32_t, LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_score,
          profile_layout,
          profile_block_size,
          kernel_strategy);
      return output;
    case KernelBits::bits64:
      output.state = prepare_score_batch_state<OpsTemplate, std::int64_t, LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_score,
          profile_layout,
          profile_block_size,
          kernel_strategy);
      return output;
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported prepared batch Farrar score width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate, bool LocalAlignment>
PreparedAffineScoreBatch<OpsTemplate> prepare_affine_score_batch(
    const PreparedFarrarBatchAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  PreparedAffineScoreBatch<OpsTemplate> output;
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      output.state = prepare_affine_score_batch_state<
          OpsTemplate,
          std::int8_t,
          !LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return output;
    case KernelBits::bits16:
      output.state = prepare_affine_score_batch_state<
          OpsTemplate,
          std::int16_t,
          !LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return output;
    case KernelBits::bits32:
      output.state = prepare_affine_score_batch_state<
          OpsTemplate,
          std::int32_t,
          !LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return output;
    case KernelBits::bits64:
      output.state = prepare_affine_score_batch_state<
          OpsTemplate,
          std::int64_t,
          !LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return output;
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported prepared batch affine Farrar score width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate, typename Cell, bool LocalAlignment>
std::vector<Score> score_batch_state(PreparedScoreBatchState<Cell>& batch) {
  std::vector<Score> scores;
  scores.reserve(batch.target_sizes.size());
  for (std::size_t index = 0; index < batch.target_sizes.size(); ++index) {
    batch.state.target_size = batch.target_sizes[index];
    batch.state.fast_score = batch.fast_scores[index];
    if (index < batch.target_profile_offsets.size()) {
      batch.state.target_profile_offsets = batch.target_profile_offsets[index];
    } else {
      batch.state.target_profile_offsets.clear();
    }
    if constexpr (LocalAlignment) {
      scores.push_back(score_state<OpsTemplate, Cell>(batch.state));
    } else {
      scores.push_back(global_score_state<OpsTemplate, Cell>(batch.state));
    }
  }
  return scores;
}

template <template <typename, typename> class OpsTemplate, bool LocalAlignment>
std::vector<Score> dispatch_prepared_score_many(PreparedScoreBatch<OpsTemplate>& prepared) {
  return std::visit(
      [](auto& state) -> std::vector<Score> {
        using State = std::decay_t<decltype(state)>;
        using Cell = typename State::cell_type;
        return score_batch_state<OpsTemplate, Cell, LocalAlignment>(state);
      },
      prepared.state);
}

template <template <typename, typename> class OpsTemplate, typename Cell, bool LocalAlignment>
std::vector<Score> affine_score_batch_state(PreparedAffineScoreBatchState<Cell>& batch) {
  std::vector<Score> scores;
  scores.reserve(batch.target_sizes.size());
  for (std::size_t index = 0; index < batch.target_sizes.size(); ++index) {
    batch.state.target_size = batch.target_sizes[index];
    if constexpr (LocalAlignment) {
      const auto offsets =
          index < batch.target_profile_offsets.size()
              ? std::span<const std::size_t>(
                    batch.target_profile_offsets[index].data(),
                    batch.target_profile_offsets[index].size())
              : std::span<const std::size_t>();
      scores.push_back(affine_score_state_for_offsets<OpsTemplate, Cell>(
          batch.state,
          offsets));
    } else {
      if (index < batch.target_profile_offsets.size()) {
        batch.state.target_profile_offsets = batch.target_profile_offsets[index];
      } else {
        batch.state.target_profile_offsets.clear();
      }
      scores.push_back(global_affine_score_state<OpsTemplate, Cell>(batch.state));
    }
  }
  return scores;
}

template <template <typename, typename> class OpsTemplate, bool LocalAlignment>
std::vector<Score> dispatch_prepared_affine_score_many(
    PreparedAffineScoreBatch<OpsTemplate>& prepared) {
  return std::visit(
      [](auto& state) -> std::vector<Score> {
        using State = std::decay_t<decltype(state)>;
        using Cell = typename State::cell_type;
        return affine_score_batch_state<OpsTemplate, Cell, LocalAlignment>(state);
      },
      prepared.state);
}

template <template <typename, typename> class OpsTemplate, bool LocalAlignment>
std::vector<Score> dispatch_score_many(
    const PreparedFarrarBatchAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8: {
      auto state = prepare_score_batch_state<OpsTemplate, std::int8_t, LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
      return score_batch_state<OpsTemplate, std::int8_t, LocalAlignment>(state);
    }
    case KernelBits::bits16: {
      auto state = prepare_score_batch_state<OpsTemplate, std::int16_t, LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
      return score_batch_state<OpsTemplate, std::int16_t, LocalAlignment>(state);
    }
    case KernelBits::bits32: {
      auto state = prepare_score_batch_state<OpsTemplate, std::int32_t, LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
      return score_batch_state<OpsTemplate, std::int32_t, LocalAlignment>(state);
    }
    case KernelBits::bits64: {
      auto state = prepare_score_batch_state<OpsTemplate, std::int64_t, LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
      return score_batch_state<OpsTemplate, std::int64_t, LocalAlignment>(state);
    }
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported batch Farrar score width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate, bool LocalAlignment>
std::vector<Score> dispatch_affine_score_many(
    const PreparedFarrarBatchAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8: {
      auto state = prepare_affine_score_batch_state<OpsTemplate, std::int8_t, !LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return affine_score_batch_state<OpsTemplate, std::int8_t, LocalAlignment>(state);
    }
    case KernelBits::bits16: {
      auto state = prepare_affine_score_batch_state<OpsTemplate, std::int16_t, !LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return affine_score_batch_state<OpsTemplate, std::int16_t, LocalAlignment>(state);
    }
    case KernelBits::bits32: {
      auto state = prepare_affine_score_batch_state<OpsTemplate, std::int32_t, !LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return affine_score_batch_state<OpsTemplate, std::int32_t, LocalAlignment>(state);
    }
    case KernelBits::bits64: {
      auto state = prepare_affine_score_batch_state<OpsTemplate, std::int64_t, !LocalAlignment>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return affine_score_batch_state<OpsTemplate, std::int64_t, LocalAlignment>(state);
    }
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported batch affine Farrar score width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate>
Score dispatch_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return score<OpsTemplate, std::int8_t>(prepared, match_score, mismatch_score, gap_score);
    case KernelBits::bits16:
      return score<OpsTemplate, std::int16_t>(prepared, match_score, mismatch_score, gap_score);
    case KernelBits::bits32:
      return score<OpsTemplate, std::int32_t>(prepared, match_score, mismatch_score, gap_score);
    case KernelBits::bits64:
      return score<OpsTemplate, std::int64_t>(prepared, match_score, mismatch_score, gap_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported Farrar score width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate>
Score dispatch_global_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      return global_score<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits16:
      return global_score<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits32:
      return global_score<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
    case KernelBits::bits64:
      return global_score<OpsTemplate, std::int64_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported global Farrar score width");
  throw nb::python_error();
}

}  // namespace detail

}  // namespace stride_align::farrar_fixed_kernel
