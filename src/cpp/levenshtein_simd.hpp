#pragma once

// Multi-target SIMD specialization for Levenshtein's *_scores entry points.
//
// Each SIMD lane holds one target's Myers state (vp, vn, score). Lane width
// is fixed at 64 bits, so the pattern length must fit in a single 64-bit
// word (m <= 64). Longer patterns fall through to the scalar dispatch in
// levenshtein_dispatch.hpp, which uses Hyyrö's multi-word algorithm.
//
// Per-backend `Ops` structs supply the SIMD primitives (Vec type, lane
// count, set/and/or/xor/add/shr63/shl1/gather/store). The kernel below is
// templated on Ops.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>

#include "preprocess.hpp"
#include "stride_align/alignment.hpp"
#include "stride_align/levenshtein.hpp"

namespace stride_align::levenshtein_simd {

namespace nb = nanobind;

template <typename Ops>
inline void myers_batch_single_word(
    const std::uint64_t* peq,
    const std::uint8_t* const* targets,
    const std::size_t* target_lengths,
    std::size_t batch_count,
    std::size_t m,
    Score* out) {
  using Vec = typename Ops::Vec;
  constexpr std::size_t lanes = Ops::lanes;

  std::size_t max_len = 0;
  for (std::size_t l = 0; l < batch_count; ++l) {
    max_len = std::max(max_len, target_lengths[l]);
  }

  const std::uint64_t init_vp = (m == 64U) ? ~std::uint64_t{0}
                                            : ((std::uint64_t{1} << m) - 1U);
  Vec vp = Ops::set1(init_vp);
  Vec vn = Ops::zero();
  Vec score = Ops::set1(static_cast<std::uint64_t>(m));
  const Vec one_v = Ops::set1(1U);

  // Test mask for the pattern's last-row bit (Myers' score update fires on
  // bit m-1 of HP/HN, not bit 63 — those positions only coincide when m=64).
  const Vec top_bit_mask = Ops::set1(std::uint64_t{1} << (m - 1U));
  const Vec zero_v = Ops::zero();

  alignas(64) std::uint64_t indices[lanes];
  alignas(64) std::uint64_t active[lanes];

  for (std::size_t k = 0; k < max_len; ++k) {
    for (std::size_t l = 0; l < lanes; ++l) {
      if (l < batch_count && k < target_lengths[l]) {
        indices[l] = targets[l][k];
        active[l] = ~std::uint64_t{0};
      } else {
        indices[l] = 0;
        active[l] = 0;
      }
    }

    const Vec eq = Ops::gather64(peq, indices);
    const Vec active_v = Ops::load_aligned(active);

    const Vec x = Ops::or_(eq, vn);
    const Vec xv = Ops::and_(x, vp);
    const Vec sum = Ops::add(xv, vp);
    const Vec d0 = Ops::or_(Ops::xor_(sum, vp), x);
    const Vec hp = Ops::or_(vn, Ops::not_(Ops::or_(d0, vp)));
    const Vec hn = Ops::and_(d0, vp);

    // hp_set: 1 per lane if bit m-1 of hp is set, else 0. Same for hn.
    // and(hp, top_bit_mask) is either 0 or top_bit_mask; comparing != 0
    // (i.e. ~cmpeq_zero) -> all-ones / 0, & 1 -> 1 / 0.
    const Vec hp_bit = Ops::and_(hp, top_bit_mask);
    const Vec hn_bit = Ops::and_(hn, top_bit_mask);
    const Vec hp_eq_zero = Ops::cmpeq(hp_bit, zero_v);
    const Vec hn_eq_zero = Ops::cmpeq(hn_bit, zero_v);
    const Vec hp_set = Ops::andnot_(hp_eq_zero, one_v);
    const Vec hn_set = Ops::andnot_(hn_eq_zero, one_v);
    const Vec delta = Ops::and_(Ops::sub(hp_set, hn_set), active_v);
    score = Ops::add(score, delta);

    const Vec hp_shift = Ops::or_(Ops::shl1(hp), one_v);
    const Vec hn_shift = Ops::shl1(hn);
    vp = Ops::or_(hn_shift, Ops::not_(Ops::or_(d0, hp_shift)));
    vn = Ops::and_(d0, hp_shift);
  }

  alignas(64) std::uint64_t scores[lanes];
  Ops::store_aligned(scores, score);
  for (std::size_t l = 0; l < batch_count; ++l) {
    out[l] = static_cast<Score>(scores[l]);
  }
}

// Validate that every materialized target is bytes/str of byte-compatible
// alphabet and that the pattern fits in 64 bits. Returns true if the SIMD
// batch path is usable for this call; otherwise the caller falls back to
// scalar dispatch.
inline bool eligible_for_simd(
    const TokenStorage& query_tokens,
    const std::vector<TokenStorage>& target_tokens) {
  if (!std::holds_alternative<std::vector<std::uint8_t>>(query_tokens)) {
    return false;
  }
  const auto& q = std::get<std::vector<std::uint8_t>>(query_tokens);
  if (q.size() > 64U || q.empty()) {
    return false;
  }
  for (const auto& storage : target_tokens) {
    if (!std::holds_alternative<std::vector<std::uint8_t>>(storage)) {
      return false;
    }
  }
  return true;
}

// Build a 256-entry PEQ table for a uint8 pattern of length <= 64.
inline std::array<std::uint64_t, 256> build_peq(std::span<const std::uint8_t> pattern) {
  std::array<std::uint64_t, 256> peq{};
  const std::uint64_t one = 1U;
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    peq[pattern[i]] |= one << i;
  }
  return peq;
}

template <typename Ops>
inline std::vector<Score> levenshtein_scores_simd(
    nb::handle query,
    nb::handle targets) {
  // Materialize each target so we can probe lengths + bytes uniformly.
  PyObject* fast_targets =
      PySequence_Fast(targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count = static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  // Tokenize query once. If the query is wider than uint8 we can't use the
  // SIMD batch and fall back per-row to the scalar dispatch.
  std::uint64_t q_symbol_limit = 0;
  TokenStorage q_storage;
  if (PyBytes_Check(query.ptr()) != 0) {
    q_storage = detail::copy_bytes_tokens(query.ptr(), q_symbol_limit);
  } else if (PyUnicode_Check(query.ptr()) != 0) {
    q_storage = detail::copy_unicode_tokens(query.ptr(), q_symbol_limit);
  } else {
    // Not bytes or str (e.g. sequence of objects). Fall back per-row.
    std::vector<Score> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      out.push_back(::stride_align::levenshtein::dispatch_score(query, nb::handle(items[i])));
    }
    return out;
  }

  // Pre-tokenize all targets matching the query kind. If any target widens
  // beyond uint8 (or doesn't match the query kind), fall back per-row.
  const bool query_is_bytes = PyBytes_Check(query.ptr()) != 0;
  std::vector<TokenStorage> target_storage;
  target_storage.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    PyObject* item = items[i];
    if (query_is_bytes != (PyBytes_Check(item) != 0)) {
      std::vector<Score> out;
      out.reserve(count);
      for (std::size_t j = 0; j < count; ++j) {
        out.push_back(::stride_align::levenshtein::dispatch_score(query, nb::handle(items[j])));
      }
      return out;
    }
    std::uint64_t t_symbol_limit = 0;
    if (query_is_bytes) {
      target_storage.push_back(detail::copy_bytes_tokens(item, t_symbol_limit));
    } else {
      target_storage.push_back(detail::copy_unicode_tokens(item, t_symbol_limit));
    }
  }

  if (!eligible_for_simd(q_storage, target_storage)) {
    std::vector<Score> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      out.push_back(::stride_align::levenshtein::dispatch_score(query, nb::handle(items[i])));
    }
    return out;
  }

  const auto& q_vec = std::get<std::vector<std::uint8_t>>(q_storage);
  const auto peq = build_peq(std::span<const std::uint8_t>(q_vec.data(), q_vec.size()));

  // Stack pointer+length arrays per batch.
  constexpr std::size_t lanes = Ops::lanes;
  std::vector<Score> out(count);
  std::array<const std::uint8_t*, lanes> ptrs{};
  std::array<std::size_t, lanes> lens{};

  for (std::size_t b = 0; b < count; b += lanes) {
    const std::size_t batch_count = std::min(lanes, count - b);
    for (std::size_t l = 0; l < batch_count; ++l) {
      const auto& tv = std::get<std::vector<std::uint8_t>>(target_storage[b + l]);
      ptrs[l] = tv.data();
      lens[l] = tv.size();
    }
    for (std::size_t l = batch_count; l < lanes; ++l) {
      ptrs[l] = nullptr;
      lens[l] = 0;
    }
    myers_batch_single_word<Ops>(
        peq.data(),
        ptrs.data(),
        lens.data(),
        batch_count,
        q_vec.size(),
        out.data() + b);
  }

  return out;
}

template <typename Ops>
inline std::vector<double> levenshtein_normalized_scores_simd(
    nb::handle query,
    nb::handle targets) {
  // Compute raw distances first, then normalize. We need target lengths,
  // so reuse the SIMD-eligibility scan path: re-materialize cheaply.
  PyObject* fast_targets =
      PySequence_Fast(targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count = static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  auto raw = levenshtein_scores_simd<Ops>(query, targets);

  const std::size_t q_len = static_cast<std::size_t>(PyObject_Length(query.ptr()));

  std::vector<double> out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t t_len = static_cast<std::size_t>(PyObject_Length(items[i]));
    out.push_back(::stride_align::levenshtein::normalize(
        static_cast<std::size_t>(raw[i]), q_len, t_len));
  }
  return out;
}

}  // namespace stride_align::levenshtein_simd
