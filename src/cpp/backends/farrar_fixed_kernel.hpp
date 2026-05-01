#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
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

template <typename Cell>
struct PreparedScoreState {
  using cell_type = Cell;

  Cell gap_score = 0;
  std::optional<Score> fast_score;
  std::size_t query_size = 0;
  std::size_t target_size = 0;
  std::size_t segment_count = 0;
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
      for (std::size_t lane = 0; lane < lane_count; ++lane) {
        const std::size_t query_index = lane * segment_count + segment;
        lanes[lane] = query_index < query.size() && query[query_index] == token ? match_score
                                                                                 : mismatch_score;
      }
    }
  }

  return profile;
}

template <template <typename, typename> class OpsTemplate, typename Cell>
PreparedScoreState<Cell> prepare_score_state(
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

template <template <typename, typename> class OpsTemplate, typename Cell>
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
  return state;
}

template <template <typename, typename> class OpsTemplate, typename Cell, bool LocalAlignment>
PreparedScoreBatchState<Cell> prepare_score_batch_state(
    const PreparedFarrarBatchAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;

  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());

  PreparedScoreBatchState<Cell> batch;
  auto& state = batch.state;
  state.gap_score = static_cast<Cell>(gap_score);
  state.query_size = query.size();
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
  if constexpr (LocalAlignment) {
    state.e_store.resize(state_cells);
  }
  return batch;
}

template <template <typename, typename> class OpsTemplate, typename Cell>
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

template <typename Ops, typename Cell>
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

    const auto v_h_open = add_sentinel<Ops, Cell>(v_h, gap_open_vector, low_score);
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
typename Ops::vector_type global_lazy_f_prefix_carry(
    typename Ops::vector_type final_f,
    std::size_t segment_count,
    Cell gap_extend_score,
    Cell low_score) {
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
typename Ops::vector_type local_lazy_f_prefix_carry(
    typename Ops::vector_type final_f,
    std::size_t segment_count,
    Cell gap_extend_score) {
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
Score affine_score_state(PreparedAffineScoreState<Cell>& state) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  constexpr std::size_t lane_count = Ops::lane_count;

  if (state.segment_count == 0 || state.target_profile_offsets.empty()) {
    return 0;
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
Score global_affine_score_state(PreparedAffineScoreState<Cell>& state) {
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

  initialize_global_affine_column_zero(
      state.h_store,
      state.e_store,
      state.query_size,
      state.segment_count,
      lane_count,
      state.gap_open_score,
      state.gap_extend_score);
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
      v_h = add_sentinel<Ops, Cell>(v_h, v_profile, low_score);
      v_h = Ops::max(v_h, v_e);
      v_h = Ops::max(v_h, v_f);
      store_state_cells<Ops, Cell>(h_store_segment, v_h);

      const auto v_h_open = add_sentinel<Ops, Cell>(v_h, gap_open_vector, low_score);
      v_e = Ops::max(add_sentinel<Ops, Cell>(v_e, gap_extend_vector, low_score), v_h_open);
      store_state_cells<Ops, Cell>(e_segment, v_e);
      v_f = Ops::max(add_sentinel<Ops, Cell>(v_f, gap_extend_vector, low_score), v_h_open);
      v_h = load_state_cells<Ops, Cell>(h_load_segment);
    }

    const bool can_prefix_lazy_f =
        state.gap_open_score <= state.gap_extend_score && state.gap_extend_score <= 0;
    if (can_prefix_lazy_f) {
      v_f = global_lazy_f_prefix_carry<Ops, Cell>(
          v_f,
          state.segment_count,
          state.gap_extend_score,
          low_score);
      scan_global_lazy_f<Ops, Cell>(
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
        const bool propagated = scan_global_lazy_f<Ops, Cell>(
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
    initialize_global_affine_column_zero(
        state.h_store,
        state.e_store,
        state.query_size,
        state.segment_count,
        lane_count,
        state.gap_open_score,
        state.gap_extend_score);
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

template <template <typename, typename> class OpsTemplate>
PreparedScore<OpsTemplate> prepare_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  PreparedScore<OpsTemplate> output;
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      output.state = prepare_score_state<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
      return output;
    case KernelBits::bits16:
      output.state = prepare_score_state<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
      return output;
    case KernelBits::bits32:
      output.state = prepare_score_state<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
      return output;
    case KernelBits::bits64:
      output.state = prepare_score_state<OpsTemplate, std::int64_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_score);
      return output;
  }

  PyErr_SetString(PyExc_RuntimeError, "unsupported Farrar score width");
  throw nb::python_error();
}

template <template <typename, typename> class OpsTemplate>
PreparedAffineScore<OpsTemplate> prepare_affine_score(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) {
  PreparedAffineScore<OpsTemplate> output;
  switch (prepared.score_bits) {
    case KernelBits::bits8:
      output.state = prepare_affine_score_state<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return output;
    case KernelBits::bits16:
      output.state = prepare_affine_score_state<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return output;
    case KernelBits::bits32:
      output.state = prepare_affine_score_state<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return output;
    case KernelBits::bits64:
      output.state = prepare_affine_score_state<OpsTemplate, std::int64_t>(
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

template <template <typename, typename> class OpsTemplate, typename Cell, bool LocalAlignment>
std::vector<Score> affine_score_batch_state(PreparedAffineScoreBatchState<Cell>& batch) {
  std::vector<Score> scores;
  scores.reserve(batch.target_sizes.size());
  for (std::size_t index = 0; index < batch.target_sizes.size(); ++index) {
    batch.state.target_size = batch.target_sizes[index];
    if (index < batch.target_profile_offsets.size()) {
      batch.state.target_profile_offsets = batch.target_profile_offsets[index];
    } else {
      batch.state.target_profile_offsets.clear();
    }
    if constexpr (LocalAlignment) {
      scores.push_back(affine_score_state<OpsTemplate, Cell>(batch.state));
    } else {
      scores.push_back(global_affine_score_state<OpsTemplate, Cell>(batch.state));
    }
  }
  return scores;
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
      auto state = prepare_affine_score_batch_state<OpsTemplate, std::int8_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return affine_score_batch_state<OpsTemplate, std::int8_t, LocalAlignment>(state);
    }
    case KernelBits::bits16: {
      auto state = prepare_affine_score_batch_state<OpsTemplate, std::int16_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return affine_score_batch_state<OpsTemplate, std::int16_t, LocalAlignment>(state);
    }
    case KernelBits::bits32: {
      auto state = prepare_affine_score_batch_state<OpsTemplate, std::int32_t>(
          prepared,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score);
      return affine_score_batch_state<OpsTemplate, std::int32_t, LocalAlignment>(state);
    }
    case KernelBits::bits64: {
      auto state = prepare_affine_score_batch_state<OpsTemplate, std::int64_t>(
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
