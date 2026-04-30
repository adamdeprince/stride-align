#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <nanobind/nanobind.h>

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

template <typename Cell>
std::vector<Cell> build_profile(
    std::span<const std::uint8_t> query,
    Cell match_score,
    Cell mismatch_score,
    std::size_t segment_count,
    std::size_t lane_count) {
  std::vector<Cell> profile(256U * segment_count * lane_count, mismatch_score);

  for (std::size_t token = 0; token < 256U; ++token) {
    for (std::size_t segment = 0; segment < segment_count; ++segment) {
      Cell* lanes = profile.data() + ((token * segment_count + segment) * lane_count);
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
Score score(
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
  if (query.empty() || target.empty()) {
    return 0;
  }

  const auto match = static_cast<Cell>(match_score);
  const auto mismatch = static_cast<Cell>(mismatch_score);
  const auto gap = static_cast<Cell>(gap_score);
  const std::size_t segment_count = (query.size() + lane_count - 1U) / lane_count;
  const auto profile = build_profile<Cell>(query, match, mismatch, segment_count, lane_count);

  std::vector<Cell> h_store(segment_count * lane_count, 0);
  std::vector<Cell> h_load(segment_count * lane_count, 0);
  std::vector<Cell> e_store(segment_count * lane_count, 0);

  const auto zero_vector = Ops::zero(lane_count);
  const auto gap_vector = Ops::set1(gap, lane_count);
  auto best_vector = zero_vector;

  for (const std::uint8_t target_token : target) {
    std::swap(h_store, h_load);

    auto v_h = shift_left_zero<Ops, Cell>(
        Ops::load_cells(h_load.data() + ((segment_count - 1U) * lane_count), lane_count),
        lane_count);
    auto v_f = zero_vector;

    for (std::size_t segment = 0; segment < segment_count; ++segment) {
      Cell* h_store_segment = h_store.data() + segment * lane_count;
      Cell* h_load_segment = h_load.data() + segment * lane_count;
      Cell* e_segment = e_store.data() + segment * lane_count;
      const Cell* profile_segment =
          profile.data() +
          ((static_cast<std::size_t>(target_token) * segment_count + segment) * lane_count);

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
      for (std::size_t segment = 0; segment < segment_count; ++segment) {
        Cell* h_store_segment = h_store.data() + segment * lane_count;
        auto v_h_segment = Ops::load_cells(h_store_segment, lane_count);
        v_h_segment = Ops::max(v_h_segment, v_f, lane_count);
        Ops::store_cells(h_store_segment, v_h_segment, lane_count);
        best_vector = Ops::max(best_vector, v_h_segment, lane_count);
        v_f = Ops::add(v_f, gap_vector, lane_count);
        if (gap <= 0 && !any_greater<Ops, Cell>(v_f, zero_vector, lane_count)) {
          goto lazy_f_done;
        }
      }
    }

lazy_f_done:
    ;
  }

  return static_cast<Score>(reduce_max<Ops, Cell>(best_vector, lane_count));
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

}  // namespace detail

}  // namespace stride_align::farrar_scalable_kernel
