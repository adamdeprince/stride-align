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

#include "byte_view.hpp"
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
    const std::vector<Token>& target,
    std::size_t cutoff = kNoCutoff) {
  if constexpr (sizeof(Token) == 1U) {
    return static_cast<Score>(myers_multi_word_u8(
        std::span<const std::uint8_t>(query.data(), query.size()),
        std::span<const std::uint8_t>(target.data(), target.size()),
        cutoff));
  } else {
    return static_cast<Score>(myers_distance<Token>(
        std::span<const Token>(query.data(), query.size()),
        std::span<const Token>(target.data(), target.size()),
        cutoff));
  }
}

inline Score score_visit(
    const TokenStorage& query_tokens,
    const TokenStorage& target_tokens,
    std::size_t cutoff = kNoCutoff) {
  return std::visit(
      [cutoff](const auto& q, const auto& t) -> Score {
        using QT = typename std::decay_t<decltype(q)>::value_type;
        using TT = typename std::decay_t<decltype(t)>::value_type;
        if constexpr (std::is_same_v<QT, TT>) {
          return score_one(q, t, cutoff);
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

// Damerau-Levenshtein (OSA-restricted) scalar dispatch. Mirrors the
// Levenshtein dispatch above: hashmap-PEQ for wide unicode, byte
// short-circuit, length normalization.
template <typename Token>
inline Score osa_score_one(
    const std::vector<Token>& query,
    const std::vector<Token>& target) {
  if constexpr (sizeof(Token) == 1U) {
    return static_cast<Score>(osa_distance_u8(
        std::span<const std::uint8_t>(query.data(), query.size()),
        std::span<const std::uint8_t>(target.data(), target.size())));
  } else {
    return static_cast<Score>(osa_distance<Token>(
        std::span<const Token>(query.data(), query.size()),
        std::span<const Token>(target.data(), target.size())));
  }
}

inline Score osa_score_visit(
    const TokenStorage& query_tokens,
    const TokenStorage& target_tokens) {
  return std::visit(
      [](const auto& q, const auto& t) -> Score {
        using QT = typename std::decay_t<decltype(q)>::value_type;
        using TT = typename std::decay_t<decltype(t)>::value_type;
        if constexpr (std::is_same_v<QT, TT>) {
          return osa_score_one(q, t);
        } else {
          PyErr_SetString(
              PyExc_RuntimeError,
              "damerau_levenshtein: mismatched token widths from preprocess");
          throw nb::python_error();
        }
      },
      query_tokens,
      target_tokens);
}

}  // namespace dispatch_detail

// Try to score (query, target) without the prepare_alignment vector
// copy. Returns true on success and writes the distance + lengths.
// Eligibility: both inputs are bytes or 1-byte unicode (their byte
// representations are then identical, so we can run scalar Myers
// directly on the CPython buffer). Wider unicode falls back to the
// preprocessed path.
inline bool try_bytes_fast_path(
    nb::handle query,
    nb::handle target,
    std::size_t cutoff,
    Score& out_distance,
    std::size_t& out_q_len,
    std::size_t& out_t_len) {
  namespace bv = ::stride_align::byte_view;
  const bv::ByteCompatKind q_kind = bv::classify(query.ptr());
  if (q_kind == bv::ByteCompatKind::None) {
    return false;
  }
  const bv::ByteCompatKind t_kind = bv::classify(target.ptr());
  if (t_kind == bv::ByteCompatKind::None) {
    return false;
  }
  const std::uint8_t* q_ptr = nullptr;
  std::size_t q_len = 0;
  const std::uint8_t* t_ptr = nullptr;
  std::size_t t_len = 0;
  bv::view(query.ptr(), q_kind, q_ptr, q_len);
  bv::view(target.ptr(), t_kind, t_ptr, t_len);
  out_distance = static_cast<Score>(myers_multi_word_u8(
      std::span<const std::uint8_t>(q_ptr, q_len),
      std::span<const std::uint8_t>(t_ptr, t_len),
      cutoff));
  out_q_len = q_len;
  out_t_len = t_len;
  return true;
}

inline Score dispatch_score(
    nb::handle query,
    nb::handle target,
    std::size_t cutoff = kNoCutoff) {
  Score distance = 0;
  std::size_t q_len = 0;
  std::size_t t_len = 0;
  if (try_bytes_fast_path(query, target, cutoff, distance, q_len, t_len)) {
    return distance;
  }
  const auto prepared = dispatch_detail::prepare(query, target);
  return dispatch_detail::score_visit(prepared.query_tokens, prepared.target_tokens, cutoff);
}

inline double dispatch_normalized_score(
    nb::handle query,
    nb::handle target,
    std::size_t cutoff = kNoCutoff) {
  Score distance = 0;
  std::size_t q_len = 0;
  std::size_t t_len = 0;
  if (try_bytes_fast_path(query, target, cutoff, distance, q_len, t_len)) {
    return normalize(static_cast<std::size_t>(distance), q_len, t_len);
  }
  const auto prepared = dispatch_detail::prepare(query, target);
  const auto qn = std::visit([](const auto& v) { return v.size(); }, prepared.query_tokens);
  const auto tn = std::visit([](const auto& v) { return v.size(); }, prepared.target_tokens);
  const Score d = dispatch_detail::score_visit(
      prepared.query_tokens, prepared.target_tokens, cutoff);
  return normalize(static_cast<std::size_t>(d), qn, tn);
}

inline std::vector<Score> dispatch_scores(
    nb::handle query,
    nb::handle targets,
    std::size_t cutoff = kNoCutoff) {
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
    out.push_back(dispatch_score(query, nb::handle(items[i]), cutoff));
  }
  return out;
}

inline std::vector<double> dispatch_normalized_scores(
    nb::handle query,
    nb::handle targets,
    std::size_t cutoff = kNoCutoff) {
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
    out.push_back(dispatch_normalized_score(query, nb::handle(items[i]), cutoff));
  }
  return out;
}

// -----------------------------------------------------------------------
// Damerau-Levenshtein (OSA) dispatch. Same shape as the Levenshtein
// entry points above, just substituting osa_distance_u8 / osa_distance
// for the Myers variants. No cutoff support yet — adding it requires
// the same per-column bail logic on top of OSA's pm_old tracking.
// -----------------------------------------------------------------------

// Zero-copy bytes / 1-byte-unicode fast path for singular OSA score.
inline bool try_bytes_fast_path_osa(
    nb::handle query,
    nb::handle target,
    Score& out_distance,
    std::size_t& out_q_len,
    std::size_t& out_t_len) {
  namespace bv = ::stride_align::byte_view;
  const bv::ByteCompatKind q_kind = bv::classify(query.ptr());
  if (q_kind == bv::ByteCompatKind::None) {
    return false;
  }
  const bv::ByteCompatKind t_kind = bv::classify(target.ptr());
  if (t_kind == bv::ByteCompatKind::None) {
    return false;
  }
  const std::uint8_t* q_ptr = nullptr;
  std::size_t q_len = 0;
  const std::uint8_t* t_ptr = nullptr;
  std::size_t t_len = 0;
  bv::view(query.ptr(), q_kind, q_ptr, q_len);
  bv::view(target.ptr(), t_kind, t_ptr, t_len);
  out_distance = static_cast<Score>(osa_distance_u8(
      std::span<const std::uint8_t>(q_ptr, q_len),
      std::span<const std::uint8_t>(t_ptr, t_len)));
  out_q_len = q_len;
  out_t_len = t_len;
  return true;
}

inline Score dispatch_osa_score(nb::handle query, nb::handle target) {
  Score distance = 0;
  std::size_t q_len = 0;
  std::size_t t_len = 0;
  if (try_bytes_fast_path_osa(query, target, distance, q_len, t_len)) {
    return distance;
  }
  const auto prepared = dispatch_detail::prepare(query, target);
  return dispatch_detail::osa_score_visit(prepared.query_tokens, prepared.target_tokens);
}

inline double dispatch_osa_normalized_score(nb::handle query, nb::handle target) {
  Score distance = 0;
  std::size_t q_len = 0;
  std::size_t t_len = 0;
  if (try_bytes_fast_path_osa(query, target, distance, q_len, t_len)) {
    return normalize(static_cast<std::size_t>(distance), q_len, t_len);
  }
  const auto prepared = dispatch_detail::prepare(query, target);
  const auto qn = std::visit([](const auto& v) { return v.size(); }, prepared.query_tokens);
  const auto tn = std::visit([](const auto& v) { return v.size(); }, prepared.target_tokens);
  const Score d = dispatch_detail::osa_score_visit(
      prepared.query_tokens, prepared.target_tokens);
  return normalize(static_cast<std::size_t>(d), qn, tn);
}

inline std::vector<Score> dispatch_osa_scores(nb::handle query, nb::handle targets) {
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
    out.push_back(dispatch_osa_score(query, nb::handle(items[i])));
  }
  return out;
}

inline std::vector<double> dispatch_osa_normalized_scores(
    nb::handle query,
    nb::handle targets) {
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
    out.push_back(dispatch_osa_normalized_score(query, nb::handle(items[i])));
  }
  return out;
}

}  // namespace stride_align::levenshtein
