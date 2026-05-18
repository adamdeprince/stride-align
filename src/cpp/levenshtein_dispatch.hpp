#pragma once

// Shared scalar dispatch for Levenshtein over Python str / bytes / sequence
// inputs. Each backend's Implementation forwards to these.
//
// SIMD-specialized variants (one target per SIMD lane) live in the
// individual x86 backend headers; they wrap these scalar paths in a
// multi-target loop and override the *_scores methods.

#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include <nanobind/nanobind.h>

#include "preprocess.hpp"
#include "stride_align/alignment.hpp"
#include "stride_align/levenshtein.hpp"

namespace stride_align::levenshtein {

namespace nb = nanobind;

namespace dispatch_detail {

// Levenshtein doesn't use match/mismatch/gap scores. We pass dummies to
// prepare_alignment so it can pick a kernel_bits compatible with the
// alphabet. The scores are irrelevant for correctness.
inline PreparedAlignment prepare(nb::handle query, nb::handle target) {
  return prepare_alignment(query, target, /*match_score=*/1,
                           /*mismatch_score=*/-1,
                           /*gap_open_score=*/-1,
                           /*gap_extend_score=*/-1,
                           /*width=*/0U);
}

template <typename Token>
inline Score score_one(
    const std::vector<Token>& query,
    const std::vector<Token>& target) {
  if constexpr (sizeof(Token) == 1U) {
    return static_cast<Score>(myers_multi_word_u8(
        std::span<const std::uint8_t>(query.data(), query.size()),
        std::span<const std::uint8_t>(target.data(), target.size())));
  } else {
    return static_cast<Score>(myers_distance<Token>(
        std::span<const Token>(query.data(), query.size()),
        std::span<const Token>(target.data(), target.size())));
  }
}

inline Score score_visit(
    const TokenStorage& query_tokens,
    const TokenStorage& target_tokens) {
  return std::visit(
      [](const auto& q, const auto& t) -> Score {
        using QT = typename std::decay_t<decltype(q)>::value_type;
        using TT = typename std::decay_t<decltype(t)>::value_type;
        if constexpr (std::is_same_v<QT, TT>) {
          return score_one(q, t);
        } else {
          // prepare_alignment casts both to the same kernel_bits, so this
          // branch should be unreachable.
          PyErr_SetString(
              PyExc_RuntimeError,
              "levenshtein: mismatched token widths from preprocess");
          throw nb::python_error();
        }
      },
      query_tokens,
      target_tokens);
}

}  // namespace dispatch_detail

inline Score dispatch_score(nb::handle query, nb::handle target) {
  const auto prepared = dispatch_detail::prepare(query, target);
  return dispatch_detail::score_visit(prepared.query_tokens, prepared.target_tokens);
}

inline double dispatch_normalized_score(nb::handle query, nb::handle target) {
  const auto prepared = dispatch_detail::prepare(query, target);
  const auto q_len = std::visit([](const auto& v) { return v.size(); }, prepared.query_tokens);
  const auto t_len = std::visit([](const auto& v) { return v.size(); }, prepared.target_tokens);
  const Score distance = dispatch_detail::score_visit(prepared.query_tokens, prepared.target_tokens);
  return normalize(static_cast<std::size_t>(distance), q_len, t_len);
}

inline std::vector<Score> dispatch_scores(nb::handle query, nb::handle targets) {
  PyObject* fast_targets =
      PySequence_Fast(targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count = static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  std::vector<Score> out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    out.push_back(dispatch_score(query, nb::handle(items[i])));
  }
  return out;
}

inline std::vector<double> dispatch_normalized_scores(nb::handle query, nb::handle targets) {
  PyObject* fast_targets =
      PySequence_Fast(targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count = static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  std::vector<double> out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    out.push_back(dispatch_normalized_score(query, nb::handle(items[i])));
  }
  return out;
}

}  // namespace stride_align::levenshtein
