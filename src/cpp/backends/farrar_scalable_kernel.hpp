#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <variant>
#include <vector>

#include <nanobind/nanobind.h>

#include "backends/score_fast_paths.hpp"
#include "farrar_preprocess.hpp"

namespace stride_align::farrar_scalable_kernel {

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

inline constexpr std::uint16_t missing_profile_index =
    std::numeric_limits<std::uint16_t>::max();

template <typename Cell>
struct PreparedScoreState {
  using cell_type = Cell;

  Cell gap_score = 0;
  std::optional<Score> fast_score;
  std::size_t query_size = 0;
  std::size_t target_size = 0;
  std::size_t lane_count = 0;
  std::size_t segment_count = 0;
  std::array<std::uint16_t, 256> profile_indices = {};
  std::vector<std::size_t> target_profile_offsets;
  std::vector<Cell> profile;
  std::vector<Cell> h_store;
  std::vector<Cell> h_load;
  std::vector<Cell> e_store;
};

template <typename Cell>
struct PreparedAffineScoreState {
  using cell_type = Cell;

  Cell gap_open_score = 0;
  Cell gap_extend_score = 0;
  std::size_t lane_count = 0;
  std::size_t segment_count = 0;
  std::array<std::uint16_t, 256> profile_indices = {};
  std::vector<std::size_t> target_profile_offsets;
  std::vector<Cell> profile;
  std::vector<Cell> h_store;
  std::vector<Cell> h_load;
  std::vector<Cell> e_store;
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

template <typename Ops, typename Cell>
typename Ops::vector_type shift_left_zero(
    typename Ops::vector_type vector,
    std::size_t lane_count) {
  if constexpr (requires { Ops::shift_left_zero(vector, lane_count); }) {
    return Ops::shift_left_zero(vector, lane_count);
  }

  std::vector<Cell> input(lane_count, 0);
  std::vector<Cell> output(lane_count, 0);
  Ops::store_cells(input.data(), vector, lane_count);
  for (std::size_t lane = 1; lane < lane_count; ++lane) {
    output[lane] = input[lane - 1];
  }
  return Ops::load_cells(output.data(), lane_count);
}

template <typename Ops, typename Cell>
typename Ops::vector_type shift_left_insert(
    typename Ops::vector_type vector,
    Cell inserted,
    std::size_t lane_count) {
  std::vector<Cell> input(lane_count, 0);
  std::vector<Cell> output(lane_count, 0);
  Ops::store_cells(input.data(), vector, lane_count);
  output[0] = inserted;
  for (std::size_t lane = 1; lane < lane_count; ++lane) {
    output[lane] = input[lane - 1];
  }
  return Ops::load_cells(output.data(), lane_count);
}

template <typename Ops, typename Cell>
typename Ops::vector_type first_lane_vector(Cell first, Cell rest, std::size_t lane_count) {
  std::vector<Cell> values(lane_count, rest);
  values[0] = first;
  return Ops::load_cells(values.data(), lane_count);
}

template <typename Ops, typename Cell>
Cell reduce_max(typename Ops::vector_type vector, std::size_t lane_count) {
  if constexpr (requires { Ops::reduce_max(vector, lane_count); }) {
    return Ops::reduce_max(vector, lane_count);
  }

  std::vector<Cell> scores(lane_count, 0);
  Ops::store_cells(scores.data(), vector, lane_count);
  Cell best_score = 0;
  for (std::size_t lane = 0; lane < lane_count; ++lane) {
    best_score = std::max(best_score, scores[lane]);
  }
  return best_score;
}

template <typename Ops, typename Cell>
bool any_greater(
    typename Ops::vector_type lhs,
    typename Ops::vector_type rhs,
    std::size_t lane_count) {
  if constexpr (requires { Ops::any_gt(lhs, rhs, lane_count); }) {
    return Ops::any_gt(lhs, rhs, lane_count);
  }

  std::vector<Cell> left(lane_count, 0);
  std::vector<Cell> right(lane_count, 0);
  Ops::store_cells(left.data(), lhs, lane_count);
  Ops::store_cells(right.data(), rhs, lane_count);
  for (std::size_t lane = 0; lane < lane_count; ++lane) {
    if (left[lane] > right[lane]) {
      return true;
    }
  }
  return false;
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

template <typename Cell>
std::vector<Cell> build_profile(
    std::span<const std::uint8_t> query,
    std::span<const std::uint8_t> profile_tokens,
    Cell match_score,
    Cell mismatch_score,
    std::size_t segment_count,
    std::size_t lane_count) {
  std::vector<Cell> profile(profile_tokens.size() * segment_count * lane_count, mismatch_score);

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
  const std::size_t lane_count = Ops::lane_count();
  if (lane_count == 0) {
    PyErr_SetString(PyExc_RuntimeError, "scalable SIMD backend reported zero active lanes");
    throw nb::python_error();
  }

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
  state.lane_count = lane_count;
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
  state.profile = build_profile<Cell>(
      query,
      profile_tokens,
      match,
      mismatch,
      state.segment_count,
      lane_count);
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
    std::vector<Cell>& h_store,
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

template <template <typename, typename> class OpsTemplate, typename Cell>
PreparedScoreState<Cell> prepare_global_score_state(
    const PreparedFarrarAlignment& prepared,
    Score match_score,
    Score mismatch_score,
    Score gap_score) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  const std::size_t lane_count = Ops::lane_count();
  if (lane_count == 0) {
    PyErr_SetString(PyExc_RuntimeError, "scalable SIMD backend reported zero active lanes");
    throw nb::python_error();
  }

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
  state.lane_count = lane_count;
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
  state.profile = build_profile<Cell>(
      query,
      profile_tokens,
      match,
      mismatch,
      state.segment_count,
      lane_count);
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
  const std::size_t lane_count = Ops::lane_count();
  if (lane_count == 0) {
    PyErr_SetString(PyExc_RuntimeError, "scalable SIMD backend reported zero active lanes");
    throw nb::python_error();
  }

  const auto query = std::span<const std::uint8_t>(
      prepared.query_tokens.data(),
      prepared.query_tokens.size());
  const auto target = std::span<const std::uint8_t>(
      prepared.target_tokens.data(),
      prepared.target_tokens.size());

  PreparedAffineScoreState<Cell> state;
  state.gap_open_score = static_cast<Cell>(gap_open_score);
  state.gap_extend_score = static_cast<Cell>(gap_extend_score);
  state.lane_count = lane_count;
  if (query.empty() || target.empty()) {
    state.profile_indices.fill(missing_profile_index);
    return state;
  }

  state.segment_count = (query.size() + lane_count - 1U) / lane_count;
  const auto profile_tokens = collect_profile_tokens(target, state.profile_indices);
  state.profile = build_profile<Cell>(
      query,
      profile_tokens,
      static_cast<Cell>(match_score),
      static_cast<Cell>(mismatch_score),
      state.segment_count,
      lane_count);
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

template <typename Ops, typename Cell>
void update_lazy_f(
    Cell* h_store_data,
    Cell* e_store_data,
    std::size_t segment_count,
    std::size_t lane_count,
    typename Ops::vector_type& v_f,
    typename Ops::vector_type gap_open_vector,
    typename Ops::vector_type gap_extend_vector,
    typename Ops::vector_type& best_vector) {
  for (std::size_t segment = 0; segment < segment_count; ++segment) {
    Cell* h_store_segment = h_store_data + segment * lane_count;
    Cell* e_segment = e_store_data + segment * lane_count;

    const auto v_h_previous = Ops::load_cells(h_store_segment, lane_count);
    auto v_h = Ops::max(v_h_previous, v_f, lane_count);
    Ops::store_cells(h_store_segment, v_h, lane_count);
    best_vector = Ops::max(best_vector, v_h, lane_count);

    const auto v_h_open = Ops::add(v_h, gap_open_vector, lane_count);
    auto v_e = Ops::load_cells(e_segment, lane_count);
    v_e = Ops::max(v_e, v_h_open, lane_count);
    Ops::store_cells(e_segment, v_e, lane_count);
    v_f = Ops::max(
        Ops::add(v_f, gap_extend_vector, lane_count),
        v_h_open,
        lane_count);
  }
}

template <template <typename, typename> class OpsTemplate, typename Cell>
Score affine_score_state(PreparedAffineScoreState<Cell>& state) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  const std::size_t lane_count = state.lane_count;

  if (state.segment_count == 0 || state.target_profile_offsets.empty()) {
    return 0;
  }

  std::fill(state.h_store.begin(), state.h_store.end(), Cell{0});
  std::fill(state.h_load.begin(), state.h_load.end(), Cell{0});
  std::fill(state.e_store.begin(), state.e_store.end(), Cell{0});

  const auto zero_vector = Ops::zero(lane_count);
  const auto gap_open_vector = Ops::set1(state.gap_open_score, lane_count);
  const auto gap_extend_vector = Ops::set1(state.gap_extend_score, lane_count);
  const bool can_stop_lazy_f = state.gap_open_score <= 0 && state.gap_extend_score <= 0;
  auto best_vector = zero_vector;
  Cell* h_store_data = state.h_store.data();
  Cell* h_load_data = state.h_load.data();
  Cell* e_store_data = state.e_store.data();
  const Cell* profile_data = state.profile.data();

  for (const auto profile_offset : state.target_profile_offsets) {
    std::swap(h_store_data, h_load_data);

    auto v_h = shift_left_zero<Ops, Cell>(
        Ops::load_cells(h_load_data + ((state.segment_count - 1U) * lane_count), lane_count),
        lane_count);
    auto v_f = zero_vector;
    const Cell* profile_row = profile_data + profile_offset;

    for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
      Cell* h_store_segment = h_store_data + segment * lane_count;
      Cell* h_load_segment = h_load_data + segment * lane_count;
      Cell* e_segment = e_store_data + segment * lane_count;
      const Cell* profile_segment = profile_row + segment * lane_count;

      const auto v_profile = Ops::load_cells(profile_segment, lane_count);
      auto v_e = Ops::load_cells(e_segment, lane_count);
      v_h = Ops::add(v_h, v_profile, lane_count);
      v_h = Ops::max(v_h, v_e, lane_count);
      v_h = Ops::max(v_h, v_f, lane_count);
      v_h = Ops::max(v_h, zero_vector, lane_count);
      Ops::store_cells(h_store_segment, v_h, lane_count);
      best_vector = Ops::max(best_vector, v_h, lane_count);

      const auto v_h_open = Ops::add(v_h, gap_open_vector, lane_count);
      v_e = Ops::max(
          Ops::add(v_e, gap_extend_vector, lane_count),
          v_h_open,
          lane_count);
      Ops::store_cells(e_segment, v_e, lane_count);
      v_f = Ops::max(
          Ops::add(v_f, gap_extend_vector, lane_count),
          v_h_open,
          lane_count);
      v_h = Ops::load_cells(h_load_segment, lane_count);
    }

    for (std::size_t iteration = 0; iteration < lane_count; ++iteration) {
      v_f = shift_left_zero<Ops, Cell>(v_f, lane_count);
      update_lazy_f<Ops, Cell>(
          h_store_data,
          e_store_data,
          state.segment_count,
          lane_count,
          v_f,
          gap_open_vector,
          gap_extend_vector,
          best_vector);
      if (can_stop_lazy_f && !any_greater<Ops, Cell>(v_f, zero_vector, lane_count)) {
        break;
      }
    }
  }

  return static_cast<Score>(reduce_max<Ops, Cell>(best_vector, lane_count));
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

template <template <typename, typename> class OpsTemplate, typename Cell>
Score score_state(PreparedScoreState<Cell>& state) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  const std::size_t lane_count = state.lane_count;

  if (state.fast_score.has_value()) {
    return *state.fast_score;
  }
  if (state.segment_count == 0 || state.target_profile_offsets.empty()) {
    return 0;
  }

  std::fill(state.h_store.begin(), state.h_store.end(), Cell{0});
  std::fill(state.h_load.begin(), state.h_load.end(), Cell{0});
  std::fill(state.e_store.begin(), state.e_store.end(), Cell{0});

  const auto zero_vector = Ops::zero(lane_count);
  const auto gap_vector = Ops::set1(state.gap_score, lane_count);
  const bool track_lazy_best = state.gap_score > 0;
  auto best_vector = zero_vector;
  Cell* h_store_data = state.h_store.data();
  Cell* h_load_data = state.h_load.data();
  Cell* e_store_data = state.e_store.data();
  const Cell* profile_data = state.profile.data();

  for (const auto profile_offset : state.target_profile_offsets) {
    std::swap(h_store_data, h_load_data);

    auto v_h = shift_left_zero<Ops, Cell>(
        Ops::load_cells(h_load_data + ((state.segment_count - 1U) * lane_count), lane_count),
        lane_count);
    auto v_f = zero_vector;
    const Cell* profile_row = profile_data + profile_offset;

    for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
      Cell* h_store_segment = h_store_data + segment * lane_count;
      Cell* h_load_segment = h_load_data + segment * lane_count;
      Cell* e_segment = e_store_data + segment * lane_count;
      const Cell* profile_segment = profile_row + segment * lane_count;

      const auto v_profile = Ops::load_cells(profile_segment, lane_count);
      auto v_e = Ops::load_cells(e_segment, lane_count);
      v_h = Ops::add(v_h, v_profile, lane_count);
      v_h = Ops::max(v_h, v_e, lane_count);
      v_h = Ops::max(v_h, v_f, lane_count);
      v_h = Ops::max(v_h, zero_vector, lane_count);
      Ops::store_cells(h_store_segment, v_h, lane_count);
      best_vector = Ops::max(best_vector, v_h, lane_count);

      const auto v_h_gap = Ops::add(v_h, gap_vector, lane_count);
      v_e = Ops::max(Ops::add(v_e, gap_vector, lane_count), v_h_gap, lane_count);
      Ops::store_cells(e_segment, v_e, lane_count);
      v_f = Ops::max(Ops::add(v_f, gap_vector, lane_count), v_h_gap, lane_count);
      v_h = Ops::load_cells(h_load_segment, lane_count);
    }

    for (std::size_t iteration = 0; iteration < lane_count; ++iteration) {
      v_f = shift_left_zero<Ops, Cell>(v_f, lane_count);
      bool propagated = false;
      for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
        Cell* h_store_segment = h_store_data + segment * lane_count;
        auto v_h_segment = Ops::load_cells(h_store_segment, lane_count);
        propagated =
            propagated || any_greater<Ops, Cell>(v_f, v_h_segment, lane_count);
        v_h_segment = Ops::max(v_h_segment, v_f, lane_count);
        Ops::store_cells(h_store_segment, v_h_segment, lane_count);
        if (track_lazy_best) {
          best_vector = Ops::max(best_vector, v_h_segment, lane_count);
        }
        v_f = Ops::add(v_f, gap_vector, lane_count);
      }

      if (state.gap_score <= 0 && !propagated) {
        break;
      }
    }
  }

  return static_cast<Score>(reduce_max<Ops, Cell>(best_vector, lane_count));
}

template <template <typename, typename> class OpsTemplate, typename Cell>
Score global_score_state(PreparedScoreState<Cell>& state) {
  using Ops = ScoreOps<OpsTemplate, Cell>;
  const std::size_t lane_count = state.lane_count;

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

  const auto gap_vector = Ops::set1(state.gap_score, lane_count);
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
        Ops::load_cells(h_load_data + ((state.segment_count - 1U) * lane_count), lane_count),
        top_left,
        lane_count);
    auto v_up = first_lane_vector<Ops, Cell>(first_up, low_score, lane_count);
    const Cell* profile_row =
        profile_data + state.target_profile_offsets[target_index];

    for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
      Cell* h_store_segment = h_store_data + segment * lane_count;
      Cell* h_load_segment = h_load_data + segment * lane_count;
      const Cell* profile_segment = profile_row + segment * lane_count;

      const auto v_profile = Ops::load_cells(profile_segment, lane_count);
      const auto v_left = Ops::add(
          Ops::load_cells(h_load_segment, lane_count),
          gap_vector,
          lane_count);
      auto v_cell = Ops::add(v_h, v_profile, lane_count);
      v_cell = Ops::max(v_cell, v_left, lane_count);
      v_cell = Ops::max(v_cell, v_up, lane_count);
      Ops::store_cells(h_store_segment, v_cell, lane_count);

      v_up = Ops::add(v_cell, gap_vector, lane_count);
      v_h = Ops::load_cells(h_load_segment, lane_count);
    }

    for (std::size_t iteration = 0; iteration < lane_count; ++iteration) {
      v_up = shift_left_insert<Ops, Cell>(v_up, low_score, lane_count);
      bool propagated = false;

      for (std::size_t segment = 0; segment < state.segment_count; ++segment) {
        Cell* h_store_segment = h_store_data + segment * lane_count;
        auto v_h_segment = Ops::load_cells(h_store_segment, lane_count);
        propagated =
            propagated || any_greater<Ops, Cell>(v_up, v_h_segment, lane_count);
        v_h_segment = Ops::max(v_h_segment, v_up, lane_count);
        Ops::store_cells(h_store_segment, v_h_segment, lane_count);
        v_up = Ops::add(v_h_segment, gap_vector, lane_count);
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

}  // namespace stride_align::farrar_scalable_kernel
