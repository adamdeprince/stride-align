#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstdio>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "affine.hpp"
#include "backends/generic.hpp"
#include "backends/profile_traceback.hpp"
#include "cdist_threshold.hpp"
#include "cdist_topk.hpp"
#include "hamming_dispatch.hpp"
#include "indel_dispatch.hpp"
#include "jaro_dispatch.hpp"
#include "levenshtein_dispatch.hpp"
#include "true_damerau_dispatch.hpp"
#include "stride_align/alignment.hpp"
#include "stride_align/hamming.hpp"
#include "stride_align/jaro.hpp"
#include "stride_align/levenshtein.hpp"
#include "stride_align/levenshtein_prepared.hpp"
#include "stride_align/dtw.hpp"
#include "dtw_dispatch.hpp"
#include "stride_align/soundex.hpp"
#include "soundex_dispatch.hpp"
#include "byte_view.hpp"
#include "topk.hpp"

namespace stride_align::bindings {

namespace nb = nanobind;

struct GapScores {
  Score open = -1;
  Score extend = -1;

  bool is_linear() const noexcept {
    return open == extend;
  }
};

inline std::uint64_t digest_cigar_string(const std::string& cigar) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const unsigned char character : cigar) {
    hash ^= static_cast<std::uint64_t>(character);
    hash *= 1099511628211ULL;
  }
  hash ^= static_cast<std::uint64_t>(cigar.size());
  hash *= 1099511628211ULL;
  return hash;
}

inline GapScores resolve_gap_scores(
    Score gap_score,
    nb::object gap_open_score,
    nb::object gap_extend_score) {
  GapScores scores;
  scores.open = gap_open_score.is_none() ? gap_score : nb::cast<Score>(gap_open_score);
  scores.extend = gap_extend_score.is_none() ? scores.open : nb::cast<Score>(gap_extend_score);
  return scores;
}

[[noreturn]] inline void throw_affine_prepared_error() {
  PyErr_SetString(
      PyExc_ValueError,
      "prepared Farrar scoring currently requires linear gaps; "
      "gap_open_score and gap_extend_score must match");
  throw nb::python_error();
}

using ScoreNDArray = nb::ndarray<nb::numpy, std::int64_t, nb::ndim<1>>;
using NormalizedNDArray = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

inline NormalizedNDArray as_normalized_ndarray(std::vector<double> values) {
  auto* heap_vec = new std::vector<double>(std::move(values));
  nb::capsule owner(heap_vec, [](void* p) noexcept {
    delete static_cast<std::vector<double>*>(p);
  });
  const std::size_t shape[1] = {heap_vec->size()};
  return NormalizedNDArray(heap_vec->data(), 1, shape, owner);
}

inline ScoreNDArray as_score_ndarray(std::vector<Score> scores) {
  // Transfer ownership of the std::vector to a capsule so the numpy array
  // can reference its buffer zero-copy. Static_assert keeps the contract
  // explicit if Score ever changes width.
  static_assert(sizeof(Score) == sizeof(std::int64_t),
                "Score must be 64-bit to alias np.int64 without copying");
  auto* heap_vec = new std::vector<Score>(std::move(scores));
  nb::capsule owner(heap_vec, [](void* p) noexcept {
    delete static_cast<std::vector<Score>*>(p);
  });
  const std::size_t shape[1] = {heap_vec->size()};
  return ScoreNDArray(
      reinterpret_cast<std::int64_t*>(heap_vec->data()),
      1,
      shape,
      owner);
}

template <typename Function>
ScoreNDArray call_score_many(nb::handle targets, Function&& function) {
  PyObject* fast_targets =
      PySequence_Fast(targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }

  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto size = static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  std::vector<Score> scores;
  scores.reserve(size);
  for (std::size_t index = 0; index < size; ++index) {
    scores.push_back(function(nb::handle(items[index])));
  }
  return as_score_ndarray(std::move(scores));
}

inline void bind_alignment_result_type(nb::module_& m) {
  nb::class_<AlignmentResult>(m, "AlignmentResult")
      .def(
          "__init__",
          [](AlignmentResult* self, Score score, std::size_t query_start,
             std::size_t query_end, std::size_t target_start,
             std::size_t target_end, nb::object aligned_query,
             nb::object aligned_target, const std::string& operations) {
            new (self) AlignmentResult();
            self->score = score;
            self->query_start = query_start;
            self->query_end = query_end;
            self->target_start = target_start;
            self->target_end = target_end;
            self->aligned_query = std::move(aligned_query);
            self->aligned_target = std::move(aligned_target);
            self->operations = operations;
          },
          nb::arg("score") = 0,
          nb::arg("query_start") = 0,
          nb::arg("query_end") = 0,
          nb::arg("target_start") = 0,
          nb::arg("target_end") = 0,
          nb::arg("aligned_query") = nb::none(),
          nb::arg("aligned_target") = nb::none(),
          nb::arg("operations") = std::string())
      .def_ro("score", &AlignmentResult::score)
      .def_ro("query_start", &AlignmentResult::query_start)
      .def_ro("query_end", &AlignmentResult::query_end)
      .def_ro("target_start", &AlignmentResult::target_start)
      .def_ro("target_end", &AlignmentResult::target_end)
      .def_ro("aligned_query", &AlignmentResult::aligned_query)
      .def_ro("aligned_target", &AlignmentResult::aligned_target)
      .def_ro("operations", &AlignmentResult::operations)
      .def("__repr__", [](const AlignmentResult& result) {
        std::ostringstream stream;
        stream << "AlignmentResult(score=" << result.score << ", query=[" << result.query_start
               << ", " << result.query_end << "), target=[" << result.target_start << ", "
               << result.target_end << "), operations='" << result.operations << "')";
        return stream.str();
      });
}

inline void bind_alignment_path_type(nb::module_& m) {
  nb::class_<AlignmentPath>(m, "AlignmentPath")
      .def_ro("score", &AlignmentPath::score)
      .def_ro("query_start", &AlignmentPath::query_start)
      .def_ro("query_end", &AlignmentPath::query_end)
      .def_ro("target_start", &AlignmentPath::target_start)
      .def_ro("target_end", &AlignmentPath::target_end)
      .def_ro("operations", &AlignmentPath::operations)
      .def_ro("cigar", &AlignmentPath::cigar)
      .def_ro("matches", &AlignmentPath::matches)
      .def_ro("mismatches", &AlignmentPath::mismatches)
      .def_ro("insertions", &AlignmentPath::insertions)
      .def_ro("deletions", &AlignmentPath::deletions)
      .def_ro("aligned_length", &AlignmentPath::aligned_length)
      .def("__repr__", [](const AlignmentPath& path) {
        std::ostringstream stream;
        stream << "AlignmentPath(score=" << path.score << ", query=[" << path.query_start
               << ", " << path.query_end << "), target=[" << path.target_start << ", "
               << path.target_end << "), cigar='" << path.cigar << "')";
        return stream.str();
      });
}

template <typename Implementation>
void ensure_backend_supported_for_fallback() {
  if constexpr (requires { Implementation::ensure_supported(); }) {
    Implementation::ensure_supported();
  }
}

template <typename Implementation>
Score call_smith_waterman_affine_score(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  if constexpr (requires {
                  Implementation::smith_waterman_affine_score(
                      query,
                      target,
                      match_score,
                      mismatch_score,
                      gap_open_score,
                      gap_extend_score,
                      width);
                }) {
    return Implementation::smith_waterman_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  } else {
    ensure_backend_supported_for_fallback<Implementation>();
    return affine::smith_waterman_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }
}

template <typename Implementation>
AlignmentResult call_smith_waterman_affine_path(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  if constexpr (requires {
                  Implementation::smith_waterman_affine_path(
                      query,
                      target,
                      match_score,
                      mismatch_score,
                      gap_open_score,
                      gap_extend_score,
                      width);
                }) {
    return Implementation::smith_waterman_affine_path(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  } else {
    ensure_backend_supported_for_fallback<Implementation>();
    return affine::smith_waterman_path(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }
}

template <typename Implementation>
Score call_smith_waterman_affine_farrar_score(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  if constexpr (requires {
                  Implementation::smith_waterman_affine_farrar_score(
                      query,
                      target,
                      match_score,
                      mismatch_score,
                      gap_open_score,
                      gap_extend_score,
                      width);
                }) {
    return Implementation::smith_waterman_affine_farrar_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  } else {
    ensure_backend_supported_for_fallback<Implementation>();
    return affine::smith_waterman_farrar_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }
}

template <typename Implementation>
Score call_needleman_wunsch_affine_score(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  if constexpr (requires {
                  Implementation::needleman_wunsch_affine_score(
                      query,
                      target,
                      match_score,
                      mismatch_score,
                      gap_open_score,
                      gap_extend_score,
                      width);
                }) {
    return Implementation::needleman_wunsch_affine_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  } else {
    ensure_backend_supported_for_fallback<Implementation>();
    return affine::needleman_wunsch_score(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }
}

template <typename Implementation>
AlignmentResult call_needleman_wunsch_affine_path(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  if constexpr (requires {
                  Implementation::needleman_wunsch_affine_path(
                      query,
                      target,
                      match_score,
                      mismatch_score,
                      gap_open_score,
                      gap_extend_score,
                      width);
                }) {
    return Implementation::needleman_wunsch_affine_path(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  } else {
    ensure_backend_supported_for_fallback<Implementation>();
    return affine::needleman_wunsch_path(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }
}

template <typename Implementation>
AlignmentPath call_smith_waterman_path_info(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_score,
    unsigned int width) {
  if constexpr (requires {
                  Implementation::smith_waterman_path_info(
                      query,
                      target,
                      match_score,
                      mismatch_score,
                      gap_score,
                      width);
                }) {
    return Implementation::smith_waterman_path_info(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  } else {
    return make_alignment_path(Implementation::smith_waterman_path(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width));
  }
}

template <typename Implementation>
AlignmentPath call_needleman_wunsch_path_info(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_score,
    unsigned int width) {
  if constexpr (requires {
                  Implementation::needleman_wunsch_path_info(
                      query,
                      target,
                      match_score,
                      mismatch_score,
                      gap_score,
                      width);
                }) {
    return Implementation::needleman_wunsch_path_info(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width);
  } else {
    return make_alignment_path(Implementation::needleman_wunsch_path(
        query,
        target,
        match_score,
        mismatch_score,
        gap_score,
        width));
  }
}

template <typename Implementation>
AlignmentPath call_smith_waterman_affine_path_info(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  if constexpr (requires {
                  Implementation::smith_waterman_affine_path_info(
                      query,
                      target,
                      match_score,
                      mismatch_score,
                      gap_open_score,
                      gap_extend_score,
                      width);
                }) {
    return Implementation::smith_waterman_affine_path_info(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  } else {
    return make_alignment_path(call_smith_waterman_affine_path<Implementation>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width));
  }
}

template <typename Implementation>
AlignmentPath call_needleman_wunsch_affine_path_info(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  if constexpr (requires {
                  Implementation::needleman_wunsch_affine_path_info(
                      query,
                      target,
                      match_score,
                      mismatch_score,
                      gap_open_score,
                      gap_extend_score,
                      width);
                }) {
    return Implementation::needleman_wunsch_affine_path_info(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  } else {
    return make_alignment_path(call_needleman_wunsch_affine_path<Implementation>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width));
  }
}

template <typename Implementation>
std::string call_smith_waterman_cigar(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  if (gap_open_score != gap_extend_score) {
    if constexpr (requires {
                    Implementation::smith_waterman_affine_cigar(
                        query,
                        target,
                        match_score,
                        mismatch_score,
                        gap_open_score,
                        gap_extend_score,
                        width);
                  }) {
      return Implementation::smith_waterman_affine_cigar(
          query,
          target,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score,
          width);
    }
    ensure_backend_supported_for_fallback<Implementation>();
    return profile_traceback::affine_cigar<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }
  if constexpr (requires {
                  Implementation::smith_waterman_linear_cigar(
                      query,
                      target,
                      match_score,
                      mismatch_score,
                      gap_open_score,
                      width);
                }) {
    if (gap_open_score <= 0) {
      return Implementation::smith_waterman_linear_cigar(
          query,
          target,
          match_score,
          mismatch_score,
          gap_open_score,
          width);
    }
  }
  if (gap_open_score <= 0) {
    ensure_backend_supported_for_fallback<Implementation>();
    return profile_traceback::linear_cigar<true>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        width);
  }
  return call_smith_waterman_path_info<Implementation>(
      query,
      target,
      match_score,
      mismatch_score,
      gap_open_score,
      width)
      .cigar;
}

template <typename Implementation>
std::string call_smith_waterman_trade_cigar(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  if (gap_open_score != gap_extend_score || gap_open_score > 0) {
    return call_smith_waterman_cigar<Implementation>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }

  ensure_backend_supported_for_fallback<Implementation>();
  if (auto cigar = profile_traceback::linear_trace_free_cigar<true>(
          query,
          target,
          match_score,
          mismatch_score,
          gap_open_score,
          width);
      cigar.has_value()) {
    return *cigar;
  }

  return call_smith_waterman_cigar<Implementation>(
      query,
      target,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score,
      width);
}

template <typename Implementation>
std::string call_needleman_wunsch_cigar(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  if (gap_open_score != gap_extend_score) {
    if constexpr (requires {
                    Implementation::needleman_wunsch_affine_cigar(
                        query,
                        target,
                        match_score,
                        mismatch_score,
                        gap_open_score,
                        gap_extend_score,
                        width);
                  }) {
      return Implementation::needleman_wunsch_affine_cigar(
          query,
          target,
          match_score,
          mismatch_score,
          gap_open_score,
          gap_extend_score,
          width);
    }
    ensure_backend_supported_for_fallback<Implementation>();
    return profile_traceback::affine_cigar<false>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score,
        width);
  }
  if constexpr (requires {
                  Implementation::needleman_wunsch_linear_cigar(
                      query,
                      target,
                      match_score,
                      mismatch_score,
                      gap_open_score,
                      width);
                }) {
    if (gap_open_score <= 0) {
      return Implementation::needleman_wunsch_linear_cigar(
          query,
          target,
          match_score,
          mismatch_score,
          gap_open_score,
          width);
    }
  }
  if (gap_open_score <= 0) {
    ensure_backend_supported_for_fallback<Implementation>();
    return profile_traceback::linear_cigar<false>(
        query,
        target,
        match_score,
        mismatch_score,
        gap_open_score,
        width);
  }
  return call_needleman_wunsch_path_info<Implementation>(
      query,
      target,
      match_score,
      mismatch_score,
      gap_open_score,
      width)
      .cigar;
}

template <typename Implementation>
std::uint64_t call_smith_waterman_cigar_digest(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  return digest_cigar_string(call_smith_waterman_cigar<Implementation>(
      query,
      target,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score,
      width));
}

template <typename Implementation>
std::uint64_t call_needleman_wunsch_cigar_digest(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  return digest_cigar_string(call_needleman_wunsch_cigar<Implementation>(
      query,
      target,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score,
      width));
}

template <typename Implementation>
std::string call_smith_waterman_affine_checkpointed_cigar(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  ensure_backend_supported_for_fallback<Implementation>();
  return profile_traceback::affine_checkpointed_cigar<true>(
      query,
      target,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score,
      width);
}

template <typename Implementation>
std::string call_needleman_wunsch_affine_checkpointed_cigar(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  ensure_backend_supported_for_fallback<Implementation>();
  return profile_traceback::affine_checkpointed_cigar<false>(
      query,
      target,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score,
      width);
}

template <typename Implementation>
void bind_backend_module(nb::module_& m, const char* doc) {
  m.doc() = doc;

  bind_alignment_result_type(m);
  bind_alignment_path_type(m);
  nb::class_<profile_traceback::PreparedAffineCigar>(m, "_PreparedAffineCigar");

  m.def(
      "smith_waterman_score",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        if (!gaps.is_linear()) {
          return call_smith_waterman_affine_score<Implementation>(
              query,
              target,
              match_score,
              mismatch_score,
              gaps.open,
              gaps.extend,
              forced_width);
        }
        return Implementation::smith_waterman_score(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  // Matrix-mode Smith-Waterman score. Caller-supplied substitution
  // matrix replaces the match/mismatch pair. Phase 1: linear gap,
  // 8-bit cells, scalar generic kernel. Each backend can opt into a
  // SIMD-matrix kernel later by defining `smith_waterman_score_matrix`
  // on its Implementation. The Python wrapper validates that
  // `matrix` and `match_score`/`mismatch_score` are mutually
  // exclusive.
  m.def(
      "smith_waterman_score_matrix",
      [](nb::handle query_indices,
         nb::handle target_indices,
         nb::handle matrix_buffer,
         std::size_t stride,
         Score gap_score) {
        if constexpr (requires {
                        Implementation::smith_waterman_score_matrix(
                            query_indices, target_indices,
                            matrix_buffer, stride, gap_score);
                      }) {
          return Implementation::smith_waterman_score_matrix(
              query_indices, target_indices, matrix_buffer, stride, gap_score);
        } else {
          return ::stride_align::backend_generic::Implementation<
              ::stride_align::BackendKind::generic>::smith_waterman_score_matrix(
              query_indices, target_indices, matrix_buffer, stride, gap_score);
        }
      },
      nb::arg("query_indices"),
      nb::arg("target_indices"),
      nb::arg("matrix_buffer"),
      nb::arg("stride"),
      nb::arg("gap_score") = -1);

  m.def(
      "smith_waterman_scores_matrix",
      [](nb::handle query_indices,
         nb::handle targets,
         nb::handle matrix_buffer,
         std::size_t stride,
         Score gap_score) {
        if constexpr (requires {
                        Implementation::smith_waterman_scores_matrix(
                            query_indices, targets,
                            matrix_buffer, stride, gap_score);
                      }) {
          return Implementation::smith_waterman_scores_matrix(
              query_indices, targets, matrix_buffer, stride, gap_score);
        } else {
          return ::stride_align::backend_generic::Implementation<
              ::stride_align::BackendKind::generic>::smith_waterman_scores_matrix(
              query_indices, targets, matrix_buffer, stride, gap_score);
        }
      },
      nb::arg("query_indices"),
      nb::arg("targets"),
      nb::arg("matrix_buffer"),
      nb::arg("stride"),
      nb::arg("gap_score") = -1);

  m.def(
      "smith_waterman_affine_score_matrix",
      [](nb::handle query_indices,
         nb::handle target_indices,
         nb::handle matrix_buffer,
         std::size_t stride,
         Score gap_open_score,
         Score gap_extend_score) {
        if constexpr (requires {
                        Implementation::smith_waterman_affine_score_matrix(
                            query_indices, target_indices,
                            matrix_buffer, stride,
                            gap_open_score, gap_extend_score);
                      }) {
          return Implementation::smith_waterman_affine_score_matrix(
              query_indices, target_indices,
              matrix_buffer, stride, gap_open_score, gap_extend_score);
        } else {
          return ::stride_align::backend_generic::Implementation<
              ::stride_align::BackendKind::generic>::smith_waterman_affine_score_matrix(
              query_indices, target_indices,
              matrix_buffer, stride, gap_open_score, gap_extend_score);
        }
      },
      nb::arg("query_indices"),
      nb::arg("target_indices"),
      nb::arg("matrix_buffer"),
      nb::arg("stride"),
      nb::arg("gap_open_score"),
      nb::arg("gap_extend_score"));

  m.def(
      "smith_waterman_path_info_matrix",
      [](nb::handle query_indices,
         nb::handle target_indices,
         nb::handle matrix_buffer,
         std::size_t stride,
         Score gap_score) {
        if constexpr (requires {
                        Implementation::smith_waterman_path_info_matrix(
                            query_indices, target_indices,
                            matrix_buffer, stride, gap_score);
                      }) {
          return Implementation::smith_waterman_path_info_matrix(
              query_indices, target_indices, matrix_buffer, stride, gap_score);
        } else {
          return ::stride_align::backend_generic::Implementation<
              ::stride_align::BackendKind::generic>::smith_waterman_path_info_matrix(
              query_indices, target_indices, matrix_buffer, stride, gap_score);
        }
      },
      nb::arg("query_indices"),
      nb::arg("target_indices"),
      nb::arg("matrix_buffer"),
      nb::arg("stride"),
      nb::arg("gap_score") = -1);

  m.def(
      "smith_waterman_affine_path_info_matrix",
      [](nb::handle query_indices,
         nb::handle target_indices,
         nb::handle matrix_buffer,
         std::size_t stride,
         Score gap_open_score,
         Score gap_extend_score) {
        if constexpr (requires {
                        Implementation::smith_waterman_affine_path_info_matrix(
                            query_indices, target_indices,
                            matrix_buffer, stride,
                            gap_open_score, gap_extend_score);
                      }) {
          return Implementation::smith_waterman_affine_path_info_matrix(
              query_indices, target_indices,
              matrix_buffer, stride, gap_open_score, gap_extend_score);
        } else {
          return ::stride_align::backend_generic::Implementation<
              ::stride_align::BackendKind::generic>::smith_waterman_affine_path_info_matrix(
              query_indices, target_indices,
              matrix_buffer, stride, gap_open_score, gap_extend_score);
        }
      },
      nb::arg("query_indices"),
      nb::arg("target_indices"),
      nb::arg("matrix_buffer"),
      nb::arg("stride"),
      nb::arg("gap_open_score"),
      nb::arg("gap_extend_score"));

  m.def(
      "smith_waterman_affine_scores_matrix",
      [](nb::handle query_indices,
         nb::handle targets,
         nb::handle matrix_buffer,
         std::size_t stride,
         Score gap_open_score,
         Score gap_extend_score) {
        if constexpr (requires {
                        Implementation::smith_waterman_affine_scores_matrix(
                            query_indices, targets,
                            matrix_buffer, stride,
                            gap_open_score, gap_extend_score);
                      }) {
          return Implementation::smith_waterman_affine_scores_matrix(
              query_indices, targets,
              matrix_buffer, stride, gap_open_score, gap_extend_score);
        } else {
          return ::stride_align::backend_generic::Implementation<
              ::stride_align::BackendKind::generic>::smith_waterman_affine_scores_matrix(
              query_indices, targets,
              matrix_buffer, stride, gap_open_score, gap_extend_score);
        }
      },
      nb::arg("query_indices"),
      nb::arg("targets"),
      nb::arg("matrix_buffer"),
      nb::arg("stride"),
      nb::arg("gap_open_score"),
      nb::arg("gap_extend_score"));

  m.def(
      "smith_waterman_scores",
      [](nb::handle query,
         nb::handle targets,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        if (!gaps.is_linear()) {
          if constexpr (requires {
                          Implementation::smith_waterman_affine_scores(
                              query,
                              targets,
                              match_score,
                              mismatch_score,
                              gaps.open,
                              gaps.extend,
                              forced_width);
                        }) {
            if (gaps.open <= 0 && gaps.extend <= 0) {
              return as_score_ndarray(Implementation::smith_waterman_affine_scores(
                  query,
                  targets,
                  match_score,
                  mismatch_score,
                  gaps.open,
                  gaps.extend,
                  forced_width));
            }
          }
        } else {
          if constexpr (requires {
                          Implementation::smith_waterman_scores(
                              query,
                              targets,
                              match_score,
                              mismatch_score,
                              gaps.open,
                              forced_width);
                        }) {
            if (gaps.open <= 0) {
              return as_score_ndarray(Implementation::smith_waterman_scores(
                  query,
                  targets,
                  match_score,
                  mismatch_score,
                  gaps.open,
                  forced_width));
            }
          }
        }
        return call_score_many(targets, [&](nb::handle target) {
          if (!gaps.is_linear()) {
            return call_smith_waterman_affine_score<Implementation>(
                query,
                target,
                match_score,
                mismatch_score,
                gaps.open,
                gaps.extend,
                forced_width);
          }
          return Implementation::smith_waterman_score(
              query,
              target,
              match_score,
              mismatch_score,
              gaps.open,
              forced_width);
        });
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "smith_waterman_path",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        if (!gaps.is_linear()) {
          return call_smith_waterman_affine_path<Implementation>(
              query,
              target,
              match_score,
              mismatch_score,
              gaps.open,
              gaps.extend,
              forced_width);
        }
        return Implementation::smith_waterman_path(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "smith_waterman_path_info",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        if (!gaps.is_linear()) {
          return call_smith_waterman_affine_path_info<Implementation>(
              query,
              target,
              match_score,
              mismatch_score,
              gaps.open,
              gaps.extend,
              forced_width);
        }
        return call_smith_waterman_path_info<Implementation>(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "smith_waterman_cigar",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        return call_smith_waterman_cigar<Implementation>(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            gaps.extend,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "_smith_waterman_cigar_digest",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        return call_smith_waterman_cigar_digest<Implementation>(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            gaps.extend,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "_smith_waterman_affine_checkpointed_cigar",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        return call_smith_waterman_affine_checkpointed_cigar<Implementation>(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            gaps.extend,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "_prepare_smith_waterman_affine_cigar",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        const Score expected_score = call_smith_waterman_affine_score<Implementation>(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            gaps.extend,
            forced_width);
        return profile_traceback::prepare_affine_cigar(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            gaps.extend,
            forced_width,
            expected_score);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "_smith_waterman_affine_cigar_prepared",
      [](profile_traceback::PreparedAffineCigar& prepared) {
        return profile_traceback::affine_cigar_prepared<true>(prepared);
      },
      nb::arg("prepared"));

  m.def(
      "smith_waterman_trace_cigar",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        return call_smith_waterman_cigar<Implementation>(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            gaps.extend,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "smith_waterman_trade_cigar",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        return call_smith_waterman_trade_cigar<Implementation>(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            gaps.extend,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "smith_waterman_farrar_score",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        if (!gaps.is_linear()) {
          return call_smith_waterman_affine_farrar_score<Implementation>(
              query,
              target,
              match_score,
              mismatch_score,
              gaps.open,
              gaps.extend,
              forced_width);
        }
        return Implementation::smith_waterman_farrar_score(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "smith_waterman_farrar_scores",
      [](nb::handle query,
         nb::handle targets,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        if (!gaps.is_linear()) {
          if constexpr (requires {
                          Implementation::smith_waterman_affine_farrar_scores(
                              query,
                              targets,
                              match_score,
                              mismatch_score,
                              gaps.open,
                              gaps.extend,
                              forced_width);
                        }) {
            if (gaps.open <= 0 && gaps.extend <= 0) {
              return as_score_ndarray(Implementation::smith_waterman_affine_farrar_scores(
                  query,
                  targets,
                  match_score,
                  mismatch_score,
                  gaps.open,
                  gaps.extend,
                  forced_width));
            }
          }
        } else {
          if constexpr (requires {
                          Implementation::smith_waterman_farrar_scores(
                              query,
                              targets,
                              match_score,
                              mismatch_score,
                              gaps.open,
                              forced_width);
                        }) {
            if (gaps.open <= 0) {
              return as_score_ndarray(Implementation::smith_waterman_farrar_scores(
                  query,
                  targets,
                  match_score,
                  mismatch_score,
                  gaps.open,
                  forced_width));
            }
          }
        }
        return call_score_many(targets, [&](nb::handle target) {
          if (!gaps.is_linear()) {
            return call_smith_waterman_affine_farrar_score<Implementation>(
                query,
                target,
                match_score,
                mismatch_score,
                gaps.open,
                gaps.extend,
                forced_width);
          }
          return Implementation::smith_waterman_farrar_score(
              query,
              target,
              match_score,
              mismatch_score,
              gaps.open,
              forced_width);
        });
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  if constexpr (requires { typename Implementation::PreparedSmithWatermanFarrarScore; }) {
    using Prepared = typename Implementation::PreparedSmithWatermanFarrarScore;
    nb::class_<Prepared>(m, "_PreparedSmithWatermanFarrarScore");

    m.def(
        "_prepare_smith_waterman_farrar_score",
        [](nb::handle query,
           nb::handle target,
           Score match_score,
           Score mismatch_score,
           Score gap_score,
           nb::object gap_open_score,
           nb::object gap_extend_score,
           nb::object width) {
          const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
          const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
          if (!gaps.is_linear()) {
            ensure_backend_supported_for_fallback<Implementation>();
            throw_affine_prepared_error();
          }
          return Implementation::prepare_smith_waterman_farrar_score(
              query,
              target,
              match_score,
              mismatch_score,
              gaps.open,
              forced_width);
        },
        nb::arg("query"),
        nb::arg("target"),
        nb::kw_only(),
        nb::arg("match_score") = 2,
        nb::arg("mismatch_score") = -1,
        nb::arg("gap_score") = -1,
        nb::arg("gap_open_score") = nb::none(),
        nb::arg("gap_extend_score") = nb::none(),
        nb::arg("width") = nb::none());

    m.def(
        "_smith_waterman_farrar_score_prepared",
        [](Prepared& prepared) {
          return Implementation::smith_waterman_farrar_score_prepared(prepared);
        },
        nb::arg("prepared"));
  }

  if constexpr (requires { typename Implementation::PreparedAffineScore; }) {
    using PreparedAffine = typename Implementation::PreparedAffineScore;
    nb::class_<PreparedAffine>(m, "_PreparedAffineScore");

    m.def(
        "_prepare_smith_waterman_affine_score",
        [](nb::handle query,
           nb::handle target,
           Score match_score,
           Score mismatch_score,
           Score gap_score,
           nb::object gap_open_score,
           nb::object gap_extend_score,
           nb::object width) {
          const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
          const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
          return Implementation::prepare_smith_waterman_affine_score(
              query,
              target,
              match_score,
              mismatch_score,
              gaps.open,
              gaps.extend,
              forced_width);
        },
        nb::arg("query"),
        nb::arg("target"),
        nb::kw_only(),
        nb::arg("match_score") = 2,
        nb::arg("mismatch_score") = -1,
        nb::arg("gap_score") = -1,
        nb::arg("gap_open_score") = nb::none(),
        nb::arg("gap_extend_score") = nb::none(),
        nb::arg("width") = nb::none());

    m.def(
        "_smith_waterman_affine_score_prepared",
        [](PreparedAffine& prepared) {
          return Implementation::smith_waterman_affine_score_prepared(prepared);
        },
        nb::arg("prepared"));

    m.def(
        "_prepare_smith_waterman_affine_farrar_score",
        [](nb::handle query,
           nb::handle target,
           Score match_score,
           Score mismatch_score,
           Score gap_score,
           nb::object gap_open_score,
           nb::object gap_extend_score,
           nb::object width) {
          const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
          const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
          return Implementation::prepare_smith_waterman_affine_farrar_score(
              query,
              target,
              match_score,
              mismatch_score,
              gaps.open,
              gaps.extend,
              forced_width);
        },
        nb::arg("query"),
        nb::arg("target"),
        nb::kw_only(),
        nb::arg("match_score") = 2,
        nb::arg("mismatch_score") = -1,
        nb::arg("gap_score") = -1,
        nb::arg("gap_open_score") = nb::none(),
        nb::arg("gap_extend_score") = nb::none(),
        nb::arg("width") = nb::none());

    m.def(
        "_smith_waterman_affine_farrar_score_prepared",
        [](PreparedAffine& prepared) {
          return Implementation::smith_waterman_affine_farrar_score_prepared(prepared);
        },
        nb::arg("prepared"));

    m.def(
        "_prepare_needleman_wunsch_affine_score",
        [](nb::handle query,
           nb::handle target,
           Score match_score,
           Score mismatch_score,
           Score gap_score,
           nb::object gap_open_score,
           nb::object gap_extend_score,
           nb::object width) {
          const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
          const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
          return Implementation::prepare_needleman_wunsch_affine_score(
              query,
              target,
              match_score,
              mismatch_score,
              gaps.open,
              gaps.extend,
              forced_width);
        },
        nb::arg("query"),
        nb::arg("target"),
        nb::kw_only(),
        nb::arg("match_score") = 2,
        nb::arg("mismatch_score") = -1,
        nb::arg("gap_score") = -1,
        nb::arg("gap_open_score") = nb::none(),
        nb::arg("gap_extend_score") = nb::none(),
        nb::arg("width") = nb::none());

    m.def(
        "_needleman_wunsch_affine_score_prepared",
        [](PreparedAffine& prepared) {
          return Implementation::needleman_wunsch_affine_score_prepared(prepared);
        },
        nb::arg("prepared"));
  }

  if constexpr (requires { typename Implementation::PreparedScoreBatch; }) {
    using PreparedScoreBatch = typename Implementation::PreparedScoreBatch;
    nb::class_<PreparedScoreBatch>(m, "_PreparedScoreBatch");

    m.def(
        "_prepare_smith_waterman_scores",
        [](nb::handle query,
           nb::handle targets,
           Score match_score,
           Score mismatch_score,
           Score gap_score,
           nb::object gap_open_score,
           nb::object gap_extend_score,
           nb::object width) {
          const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
          const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
          if (!gaps.is_linear()) {
            ensure_backend_supported_for_fallback<Implementation>();
            throw_affine_prepared_error();
          }
          return Implementation::prepare_smith_waterman_scores(
              query,
              targets,
              match_score,
              mismatch_score,
              gaps.open,
              forced_width);
        },
        nb::arg("query"),
        nb::arg("targets"),
        nb::kw_only(),
        nb::arg("match_score") = 2,
        nb::arg("mismatch_score") = -1,
        nb::arg("gap_score") = -1,
        nb::arg("gap_open_score") = nb::none(),
        nb::arg("gap_extend_score") = nb::none(),
        nb::arg("width") = nb::none());

    m.def(
        "_smith_waterman_scores_prepared",
        [](PreparedScoreBatch& prepared) {
          return as_score_ndarray(Implementation::smith_waterman_scores_prepared(prepared));
        },
        nb::arg("prepared"));

    m.def(
        "_prepare_smith_waterman_farrar_scores",
        [](nb::handle query,
           nb::handle targets,
           Score match_score,
           Score mismatch_score,
           Score gap_score,
           nb::object gap_open_score,
           nb::object gap_extend_score,
           nb::object width) {
          const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
          const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
          if (!gaps.is_linear()) {
            ensure_backend_supported_for_fallback<Implementation>();
            throw_affine_prepared_error();
          }
          return Implementation::prepare_smith_waterman_farrar_scores(
              query,
              targets,
              match_score,
              mismatch_score,
              gaps.open,
              forced_width);
        },
        nb::arg("query"),
        nb::arg("targets"),
        nb::kw_only(),
        nb::arg("match_score") = 2,
        nb::arg("mismatch_score") = -1,
        nb::arg("gap_score") = -1,
        nb::arg("gap_open_score") = nb::none(),
        nb::arg("gap_extend_score") = nb::none(),
        nb::arg("width") = nb::none());

    m.def(
        "_smith_waterman_farrar_scores_prepared",
        [](PreparedScoreBatch& prepared) {
          return as_score_ndarray(Implementation::smith_waterman_farrar_scores_prepared(prepared));
        },
        nb::arg("prepared"));

    m.def(
        "_prepare_needleman_wunsch_scores",
        [](nb::handle query,
           nb::handle targets,
           Score match_score,
           Score mismatch_score,
           Score gap_score,
           nb::object gap_open_score,
           nb::object gap_extend_score,
           nb::object width) {
          const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
          const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
          if (!gaps.is_linear()) {
            ensure_backend_supported_for_fallback<Implementation>();
            throw_affine_prepared_error();
          }
          return Implementation::prepare_needleman_wunsch_scores(
              query,
              targets,
              match_score,
              mismatch_score,
              gaps.open,
              forced_width);
        },
        nb::arg("query"),
        nb::arg("targets"),
        nb::kw_only(),
        nb::arg("match_score") = 2,
        nb::arg("mismatch_score") = -1,
        nb::arg("gap_score") = -1,
        nb::arg("gap_open_score") = nb::none(),
        nb::arg("gap_extend_score") = nb::none(),
        nb::arg("width") = nb::none());

    m.def(
        "_needleman_wunsch_scores_prepared",
        [](PreparedScoreBatch& prepared) {
          return as_score_ndarray(Implementation::needleman_wunsch_scores_prepared(prepared));
        },
        nb::arg("prepared"));
  }

  if constexpr (requires { typename Implementation::PreparedAffineScoreBatch; }) {
    using PreparedAffineScoreBatch = typename Implementation::PreparedAffineScoreBatch;
    nb::class_<PreparedAffineScoreBatch>(m, "_PreparedAffineScoreBatch");

    m.def(
        "_prepare_smith_waterman_affine_scores",
        [](nb::handle query,
           nb::handle targets,
           Score match_score,
           Score mismatch_score,
           Score gap_score,
           nb::object gap_open_score,
           nb::object gap_extend_score,
           nb::object width) {
          const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
          const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
          return Implementation::prepare_smith_waterman_affine_scores(
              query,
              targets,
              match_score,
              mismatch_score,
              gaps.open,
              gaps.extend,
              forced_width);
        },
        nb::arg("query"),
        nb::arg("targets"),
        nb::kw_only(),
        nb::arg("match_score") = 2,
        nb::arg("mismatch_score") = -1,
        nb::arg("gap_score") = -1,
        nb::arg("gap_open_score") = nb::none(),
        nb::arg("gap_extend_score") = nb::none(),
        nb::arg("width") = nb::none());

    m.def(
        "_smith_waterman_affine_scores_prepared",
        [](PreparedAffineScoreBatch& prepared) {
          return as_score_ndarray(
              Implementation::smith_waterman_affine_scores_prepared(prepared));
        },
        nb::arg("prepared"));

    m.def(
        "_prepare_smith_waterman_affine_farrar_scores",
        [](nb::handle query,
           nb::handle targets,
           Score match_score,
           Score mismatch_score,
           Score gap_score,
           nb::object gap_open_score,
           nb::object gap_extend_score,
           nb::object width) {
          const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
          const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
          return Implementation::prepare_smith_waterman_affine_farrar_scores(
              query,
              targets,
              match_score,
              mismatch_score,
              gaps.open,
              gaps.extend,
              forced_width);
        },
        nb::arg("query"),
        nb::arg("targets"),
        nb::kw_only(),
        nb::arg("match_score") = 2,
        nb::arg("mismatch_score") = -1,
        nb::arg("gap_score") = -1,
        nb::arg("gap_open_score") = nb::none(),
        nb::arg("gap_extend_score") = nb::none(),
        nb::arg("width") = nb::none());

    m.def(
        "_smith_waterman_affine_farrar_scores_prepared",
        [](PreparedAffineScoreBatch& prepared) {
          return as_score_ndarray(
              Implementation::smith_waterman_affine_farrar_scores_prepared(prepared));
        },
        nb::arg("prepared"));

    m.def(
        "_prepare_needleman_wunsch_affine_scores",
        [](nb::handle query,
           nb::handle targets,
           Score match_score,
           Score mismatch_score,
           Score gap_score,
           nb::object gap_open_score,
           nb::object gap_extend_score,
           nb::object width) {
          const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
          const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
          return Implementation::prepare_needleman_wunsch_affine_scores(
              query,
              targets,
              match_score,
              mismatch_score,
              gaps.open,
              gaps.extend,
              forced_width);
        },
        nb::arg("query"),
        nb::arg("targets"),
        nb::kw_only(),
        nb::arg("match_score") = 2,
        nb::arg("mismatch_score") = -1,
        nb::arg("gap_score") = -1,
        nb::arg("gap_open_score") = nb::none(),
        nb::arg("gap_extend_score") = nb::none(),
        nb::arg("width") = nb::none());

    m.def(
        "_needleman_wunsch_affine_scores_prepared",
        [](PreparedAffineScoreBatch& prepared) {
          return as_score_ndarray(
              Implementation::needleman_wunsch_affine_scores_prepared(prepared));
        },
        nb::arg("prepared"));
  }

  m.def(
      "needleman_wunsch_score",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        if (!gaps.is_linear()) {
          return call_needleman_wunsch_affine_score<Implementation>(
              query,
              target,
              match_score,
              mismatch_score,
              gaps.open,
              gaps.extend,
              forced_width);
        }
        return Implementation::needleman_wunsch_score(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  // Matrix-mode Needleman-Wunsch — same shape as the SW variant.
  m.def(
      "needleman_wunsch_score_matrix",
      [](nb::handle query_indices,
         nb::handle target_indices,
         nb::handle matrix_buffer,
         std::size_t stride,
         Score gap_score) {
        if constexpr (requires {
                        Implementation::needleman_wunsch_score_matrix(
                            query_indices, target_indices,
                            matrix_buffer, stride, gap_score);
                      }) {
          return Implementation::needleman_wunsch_score_matrix(
              query_indices, target_indices, matrix_buffer, stride, gap_score);
        } else {
          return ::stride_align::backend_generic::Implementation<
              ::stride_align::BackendKind::generic>::needleman_wunsch_score_matrix(
              query_indices, target_indices, matrix_buffer, stride, gap_score);
        }
      },
      nb::arg("query_indices"),
      nb::arg("target_indices"),
      nb::arg("matrix_buffer"),
      nb::arg("stride"),
      nb::arg("gap_score") = -1);

  m.def(
      "needleman_wunsch_scores_matrix",
      [](nb::handle query_indices,
         nb::handle targets,
         nb::handle matrix_buffer,
         std::size_t stride,
         Score gap_score) {
        if constexpr (requires {
                        Implementation::needleman_wunsch_scores_matrix(
                            query_indices, targets,
                            matrix_buffer, stride, gap_score);
                      }) {
          return Implementation::needleman_wunsch_scores_matrix(
              query_indices, targets, matrix_buffer, stride, gap_score);
        } else {
          return ::stride_align::backend_generic::Implementation<
              ::stride_align::BackendKind::generic>::needleman_wunsch_scores_matrix(
              query_indices, targets, matrix_buffer, stride, gap_score);
        }
      },
      nb::arg("query_indices"),
      nb::arg("targets"),
      nb::arg("matrix_buffer"),
      nb::arg("stride"),
      nb::arg("gap_score") = -1);

  m.def(
      "needleman_wunsch_affine_score_matrix",
      [](nb::handle query_indices,
         nb::handle target_indices,
         nb::handle matrix_buffer,
         std::size_t stride,
         Score gap_open_score,
         Score gap_extend_score) {
        if constexpr (requires {
                        Implementation::needleman_wunsch_affine_score_matrix(
                            query_indices, target_indices,
                            matrix_buffer, stride,
                            gap_open_score, gap_extend_score);
                      }) {
          return Implementation::needleman_wunsch_affine_score_matrix(
              query_indices, target_indices,
              matrix_buffer, stride, gap_open_score, gap_extend_score);
        } else {
          return ::stride_align::backend_generic::Implementation<
              ::stride_align::BackendKind::generic>::needleman_wunsch_affine_score_matrix(
              query_indices, target_indices,
              matrix_buffer, stride, gap_open_score, gap_extend_score);
        }
      },
      nb::arg("query_indices"),
      nb::arg("target_indices"),
      nb::arg("matrix_buffer"),
      nb::arg("stride"),
      nb::arg("gap_open_score"),
      nb::arg("gap_extend_score"));

  m.def(
      "needleman_wunsch_path_info_matrix",
      [](nb::handle query_indices,
         nb::handle target_indices,
         nb::handle matrix_buffer,
         std::size_t stride,
         Score gap_score) {
        if constexpr (requires {
                        Implementation::needleman_wunsch_path_info_matrix(
                            query_indices, target_indices,
                            matrix_buffer, stride, gap_score);
                      }) {
          return Implementation::needleman_wunsch_path_info_matrix(
              query_indices, target_indices, matrix_buffer, stride, gap_score);
        } else {
          return ::stride_align::backend_generic::Implementation<
              ::stride_align::BackendKind::generic>::needleman_wunsch_path_info_matrix(
              query_indices, target_indices, matrix_buffer, stride, gap_score);
        }
      },
      nb::arg("query_indices"),
      nb::arg("target_indices"),
      nb::arg("matrix_buffer"),
      nb::arg("stride"),
      nb::arg("gap_score") = -1);

  m.def(
      "needleman_wunsch_affine_path_info_matrix",
      [](nb::handle query_indices,
         nb::handle target_indices,
         nb::handle matrix_buffer,
         std::size_t stride,
         Score gap_open_score,
         Score gap_extend_score) {
        if constexpr (requires {
                        Implementation::needleman_wunsch_affine_path_info_matrix(
                            query_indices, target_indices,
                            matrix_buffer, stride,
                            gap_open_score, gap_extend_score);
                      }) {
          return Implementation::needleman_wunsch_affine_path_info_matrix(
              query_indices, target_indices,
              matrix_buffer, stride, gap_open_score, gap_extend_score);
        } else {
          return ::stride_align::backend_generic::Implementation<
              ::stride_align::BackendKind::generic>::needleman_wunsch_affine_path_info_matrix(
              query_indices, target_indices,
              matrix_buffer, stride, gap_open_score, gap_extend_score);
        }
      },
      nb::arg("query_indices"),
      nb::arg("target_indices"),
      nb::arg("matrix_buffer"),
      nb::arg("stride"),
      nb::arg("gap_open_score"),
      nb::arg("gap_extend_score"));

  m.def(
      "needleman_wunsch_affine_scores_matrix",
      [](nb::handle query_indices,
         nb::handle targets,
         nb::handle matrix_buffer,
         std::size_t stride,
         Score gap_open_score,
         Score gap_extend_score) {
        if constexpr (requires {
                        Implementation::needleman_wunsch_affine_scores_matrix(
                            query_indices, targets,
                            matrix_buffer, stride,
                            gap_open_score, gap_extend_score);
                      }) {
          return Implementation::needleman_wunsch_affine_scores_matrix(
              query_indices, targets,
              matrix_buffer, stride, gap_open_score, gap_extend_score);
        } else {
          return ::stride_align::backend_generic::Implementation<
              ::stride_align::BackendKind::generic>::needleman_wunsch_affine_scores_matrix(
              query_indices, targets,
              matrix_buffer, stride, gap_open_score, gap_extend_score);
        }
      },
      nb::arg("query_indices"),
      nb::arg("targets"),
      nb::arg("matrix_buffer"),
      nb::arg("stride"),
      nb::arg("gap_open_score"),
      nb::arg("gap_extend_score"));

  m.def(
      "needleman_wunsch_scores",
      [](nb::handle query,
         nb::handle targets,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        if (!gaps.is_linear()) {
          if constexpr (requires {
                          Implementation::needleman_wunsch_affine_scores(
                              query,
                              targets,
                              match_score,
                              mismatch_score,
                              gaps.open,
                              gaps.extend,
                              forced_width);
                        }) {
            if (gaps.open <= 0 && gaps.extend <= 0) {
              return as_score_ndarray(Implementation::needleman_wunsch_affine_scores(
                  query,
                  targets,
                  match_score,
                  mismatch_score,
                  gaps.open,
                  gaps.extend,
                  forced_width));
            }
          }
        } else {
          if constexpr (requires {
                          Implementation::needleman_wunsch_scores(
                              query,
                              targets,
                              match_score,
                              mismatch_score,
                              gaps.open,
                              forced_width);
                        }) {
            if (gaps.open <= 0) {
              return as_score_ndarray(Implementation::needleman_wunsch_scores(
                  query,
                  targets,
                  match_score,
                  mismatch_score,
                  gaps.open,
                  forced_width));
            }
          }
        }
        return call_score_many(targets, [&](nb::handle target) {
          if (!gaps.is_linear()) {
            return call_needleman_wunsch_affine_score<Implementation>(
                query,
                target,
                match_score,
                mismatch_score,
                gaps.open,
                gaps.extend,
                forced_width);
          }
          return Implementation::needleman_wunsch_score(
              query,
              target,
              match_score,
              mismatch_score,
              gaps.open,
              forced_width);
        });
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "needleman_wunsch_path",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        if (!gaps.is_linear()) {
          return call_needleman_wunsch_affine_path<Implementation>(
              query,
              target,
              match_score,
              mismatch_score,
              gaps.open,
              gaps.extend,
              forced_width);
        }
        return Implementation::needleman_wunsch_path(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "needleman_wunsch_path_info",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        if (!gaps.is_linear()) {
          return call_needleman_wunsch_affine_path_info<Implementation>(
              query,
              target,
              match_score,
              mismatch_score,
              gaps.open,
              gaps.extend,
              forced_width);
        }
        return call_needleman_wunsch_path_info<Implementation>(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "needleman_wunsch_cigar",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        return call_needleman_wunsch_cigar<Implementation>(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            gaps.extend,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "_needleman_wunsch_cigar_digest",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        return call_needleman_wunsch_cigar_digest<Implementation>(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            gaps.extend,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "_needleman_wunsch_affine_checkpointed_cigar",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        return call_needleman_wunsch_affine_checkpointed_cigar<Implementation>(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            gaps.extend,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "_prepare_needleman_wunsch_affine_cigar",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        const Score expected_score = call_needleman_wunsch_affine_score<Implementation>(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            gaps.extend,
            forced_width);
        return profile_traceback::prepare_affine_cigar(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            gaps.extend,
            forced_width,
            expected_score);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "_needleman_wunsch_affine_cigar_prepared",
      [](profile_traceback::PreparedAffineCigar& prepared) {
        return profile_traceback::affine_cigar_prepared<false>(prepared);
      },
      nb::arg("prepared"));

  m.def(
      "needleman_wunsch_trace_cigar",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        return call_needleman_wunsch_cigar<Implementation>(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            gaps.extend,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "needleman_wunsch_trade_cigar",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps = resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        return call_needleman_wunsch_cigar<Implementation>(
            query,
            target,
            match_score,
            mismatch_score,
            gaps.open,
            gaps.extend,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  // --- Levenshtein -------------------------------------------------------
  //
  // No match/mismatch/gap scores: Levenshtein is fixed at unit cost. The
  // batch *_scores entry points dispatch on the backend's compile-time
  // BackendKind to a SIMD multi-target Myers kernel (each SIMD lane runs
  // one target's state, pattern <= 64 chars). Backends that don't define
  // a matching ISA group fall through to the shared scalar dispatch in
  // levenshtein_dispatch.hpp, which uses Hyyrö's multi-word Myers and so
  // has no length cap.

  // score_cutoff: when set, the kernel returns as soon as it can prove a
  // pair's final distance must exceed the cutoff; the returned score is
  // clamped to cutoff + 1 (rapidfuzz convention). Defaults to None,
  // which disables the optimization and runs to completion (exact
  // distance).
  auto to_cutoff = [](nb::object cutoff_obj) -> std::size_t {
    if (cutoff_obj.is_none()) {
      return ::stride_align::levenshtein::kNoCutoff;
    }
    return static_cast<std::size_t>(nb::cast<long long>(cutoff_obj));
  };

  m.def(
      "levenshtein_score",
      [to_cutoff](nb::handle query, nb::handle target, nb::object score_cutoff) {
        return ::stride_align::levenshtein::dispatch_score(
            query, target, to_cutoff(score_cutoff));
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::arg("score_cutoff") = nb::none());

  m.def(
      "levenshtein_normalized_score",
      [to_cutoff](nb::handle query, nb::handle target, nb::object score_cutoff) {
        return ::stride_align::levenshtein::dispatch_normalized_score(
            query, target, to_cutoff(score_cutoff));
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::arg("score_cutoff") = nb::none());

  // Persistent prepared Levenshtein scorer. Pre-builds the Myers
  // pattern_match (PEQ) table for a fixed query and reuses it across
  // every distance(target) call. Bytes / 1-byte-unicode queries take
  // the prepared path; wider queries fall back to the per-call
  // dispatch (the Python ``LevenshteinScorer`` checks ``.has_prepared``
  // and switches accordingly).
  using PreparedLev = ::stride_align::levenshtein::PreparedLevenshteinScoreU8;
  nb::class_<PreparedLev>(m, "_PreparedLevenshteinScore")
      .def_prop_ro("query_size", [](const PreparedLev& self) {
        return self.m;
      })
      .def_prop_ro("single_word", [](const PreparedLev& self) {
        return self.single_word;
      });

  m.def(
      "_prepare_levenshtein_score",
      [](nb::handle query) -> nb::object {
        namespace bv = ::stride_align::byte_view;
        const bv::ByteCompatKind kind = bv::classify(query.ptr());
        if (kind == bv::ByteCompatKind::None) {
          // Wider unicode / sequence inputs aren't supported by the
          // bytes-only prepared kernel yet. Return None and let the
          // Python LevenshteinScorer fall back to the per-call path.
          return nb::none();
        }
        const std::uint8_t* ptr = nullptr;
        std::size_t len = 0;
        bv::view(query.ptr(), kind, ptr, len);
        PreparedLev prepared =
            ::stride_align::levenshtein::prepare_levenshtein_score_u8(
                std::span<const std::uint8_t>(ptr, len));
        return nb::cast(std::move(prepared));
      },
      nb::arg("query"));

  m.def(
      "_levenshtein_score_prepared",
      [to_cutoff](
          const PreparedLev& prepared,
          nb::handle target,
          nb::object score_cutoff) -> nb::object {
        namespace bv = ::stride_align::byte_view;
        const bv::ByteCompatKind kind = bv::classify(target.ptr());
        if (kind == bv::ByteCompatKind::None) {
          // Wider target — caller should fall back. Returning None
          // tells the Python wrapper to retry through the unprepared
          // dispatch.
          return nb::none();
        }
        const std::uint8_t* ptr = nullptr;
        std::size_t len = 0;
        bv::view(target.ptr(), kind, ptr, len);
        const std::size_t cutoff = to_cutoff(score_cutoff);
        const std::size_t distance =
            ::stride_align::levenshtein::levenshtein_score_prepared_u8(
                prepared, std::span<const std::uint8_t>(ptr, len), cutoff);
        return nb::cast(static_cast<long long>(distance));
      },
      nb::arg("prepared"),
      nb::arg("target"),
      nb::arg("score_cutoff") = nb::none());

  // Each backend that wants the multi-target SIMD kernel exposes
  // Implementation::levenshtein_scores / _normalized_scores as a static
  // method inside its pragma-target scope (so the templated Myers kernel
  // is instantiated with the matching ISA macros visible). Backends that
  // don't define the method fall through to the scalar dispatch in
  // levenshtein_dispatch.hpp, which uses Hyyrö's multi-word Myers and has
  // no length cap. The `requires`-check below is the only dispatch point;
  // no #ifdef survives here.
  m.def(
      "levenshtein_scores",
      [to_cutoff](nb::handle query, nb::handle targets, nb::object score_cutoff) {
        const std::size_t cutoff = to_cutoff(score_cutoff);
        if constexpr (requires {
                        Implementation::levenshtein_scores(query, targets, cutoff);
                      }) {
          return as_score_ndarray(
              Implementation::levenshtein_scores(query, targets, cutoff));
        } else {
          return as_score_ndarray(
              ::stride_align::levenshtein::dispatch_scores(query, targets, cutoff));
        }
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::arg("score_cutoff") = nb::none());

  m.def(
      "levenshtein_normalized_scores",
      [to_cutoff](nb::handle query, nb::handle targets, nb::object score_cutoff) {
        const std::size_t cutoff = to_cutoff(score_cutoff);
        if constexpr (requires {
                        Implementation::levenshtein_normalized_scores(query, targets, cutoff);
                      }) {
          return as_normalized_ndarray(
              Implementation::levenshtein_normalized_scores(query, targets, cutoff));
        } else {
          return as_normalized_ndarray(
              ::stride_align::levenshtein::dispatch_normalized_scores(
                  query, targets, cutoff));
        }
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::arg("score_cutoff") = nb::none());

  // --- Damerau-Levenshtein (OSA-restricted) ----------------------------
  //
  // Same dispatch shape as Levenshtein: requires-check picks the backend
  // SIMD batch kernel when available, otherwise the shared scalar OSA
  // path (single-word bit-parallel for m <= 64, rolling-row DP above).
  // No cutoff yet — adding it requires per-column bail with pm_old in
  // SIMD lanes, which is doable but not implemented.

  m.def(
      "damerau_levenshtein_score",
      [](nb::handle query, nb::handle target) {
        return ::stride_align::levenshtein::dispatch_osa_score(query, target);
      },
      nb::arg("query"),
      nb::arg("target"));

  m.def(
      "damerau_levenshtein_normalized_score",
      [](nb::handle query, nb::handle target) {
        return ::stride_align::levenshtein::dispatch_osa_normalized_score(query, target);
      },
      nb::arg("query"),
      nb::arg("target"));

  m.def(
      "damerau_levenshtein_scores",
      [](nb::handle query, nb::handle targets) {
        if constexpr (requires {
                        Implementation::damerau_levenshtein_scores(query, targets);
                      }) {
          return as_score_ndarray(
              Implementation::damerau_levenshtein_scores(query, targets));
        } else {
          return as_score_ndarray(
              ::stride_align::levenshtein::dispatch_osa_scores(query, targets));
        }
      },
      nb::arg("query"),
      nb::arg("targets"));

  m.def(
      "damerau_levenshtein_normalized_scores",
      [](nb::handle query, nb::handle targets) {
        if constexpr (requires {
                        Implementation::damerau_levenshtein_normalized_scores(query, targets);
                      }) {
          return as_normalized_ndarray(
              Implementation::damerau_levenshtein_normalized_scores(query, targets));
        } else {
          return as_normalized_ndarray(
              ::stride_align::levenshtein::dispatch_osa_normalized_scores(
                  query, targets));
        }
      },
      nb::arg("query"),
      nb::arg("targets"));

  // --- Hamming ----------------------------------------------------------
  //
  // Requires `len(query) == len(target)`; mismatched lengths raise
  // ValueError. The SIMD path uses two custom kernels: within-string
  // bytewise cmpne + popcount for the singular API, and across-target
  // byte-lane counters for the batch API (one target per byte lane,
  // up to 16/32/64 targets in parallel depending on ISA). Wider unicode
  // and sequence-of-object inputs fall through to scalar.

  // ---- DTW (scalar reference, phase C.1a) -------------------------------
  // SIMD batch comes in phase C.1b. The C++ surface here is final; the
  // SIMD layer plugs into dispatch_dtw / dispatch_dtw_many without
  // touching the bindings.

  m.def(
      "dtw",
      [](nb::handle query, nb::handle target,
         nb::object window, nb::object distance) {
        return ::stride_align::dtw::dispatch_dtw(query, target, window, distance);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("window") = nb::none(),
      nb::arg("distance") = nb::none());

  m.def(
      "dtw_distances",
      [](nb::handle query, nb::handle targets,
         nb::object window, nb::object distance) {
        const auto distances = ::stride_align::dtw::dispatch_dtw_many(
            query, targets, window, distance);
        // Return ndarray[float64]. Mirrors as_score_ndarray for the
        // existing batch APIs but float64-typed since DTW results
        // aren't int64.
        const std::size_t n = distances.size();
        auto* buffer = new double[n];
        for (std::size_t i = 0; i < n; ++i) buffer[i] = distances[i];
        nb::capsule owner(buffer, [](void* p) noexcept {
          delete[] static_cast<double*>(p);
        });
        return nb::ndarray<nb::numpy, double, nb::shape<-1>>(
            buffer, {n}, owner);
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::kw_only(),
      nb::arg("window") = nb::none(),
      nb::arg("distance") = nb::none());

  m.def(
      "hamming_score",
      [](nb::handle query, nb::handle target) {
        return ::stride_align::hamming::dispatch_score(query, target);
      },
      nb::arg("query"),
      nb::arg("target"));

  // ---- Phonetic encoders (phase D.2) ------------------------------------

  m.def(
      "soundex",
      [](nb::handle s) {
        return ::stride_align::phonetic::dispatch_soundex(s);
      },
      nb::arg("s"));

  // MetaphoneVariant is a Python IntEnum in stride_align/__init__.py;
  // crossing the binding boundary as a plain int avoids the
  // per-backend nb::enum_ duplicate-registration trap (same pattern
  // as the Scorer enum).
  m.def(
      "metaphone",
      [](nb::handle s, int variant) {
        const auto v = static_cast<::stride_align::phonetic::MetaphoneVariant>(variant);
        return ::stride_align::phonetic::dispatch_metaphone(s, v);
      },
      nb::arg("s"),
      nb::kw_only(),
      nb::arg("variant") = 0);  // 0 = kPhilips

  m.def(
      "nysiis",
      [](nb::handle s) {
        return ::stride_align::phonetic::dispatch_nysiis(s);
      },
      nb::arg("s"));

  m.def(
      "match_rating_codex",
      [](nb::handle s) {
        return ::stride_align::phonetic::dispatch_match_rating_codex(s);
      },
      nb::arg("s"));

  m.def(
      "match_rating_compare",
      [](nb::handle a, nb::handle b) {
        return ::stride_align::phonetic::dispatch_match_rating_compare(a, b);
      },
      nb::arg("a"),
      nb::arg("b"));

  m.def(
      "caverphone",
      [](nb::handle s) {
        return ::stride_align::phonetic::dispatch_caverphone(s);
      },
      nb::arg("s"));

  m.def(
      "hamming_normalized_score",
      [](nb::handle query, nb::handle target) {
        return ::stride_align::hamming::dispatch_normalized_score(query, target);
      },
      nb::arg("query"),
      nb::arg("target"));

  m.def(
      "hamming_scores",
      [](nb::handle query, nb::handle targets) {
        return as_score_ndarray(
            ::stride_align::hamming::dispatch_scores(query, targets));
      },
      nb::arg("query"),
      nb::arg("targets"));

  m.def(
      "hamming_normalized_scores",
      [](nb::handle query, nb::handle targets) {
        return as_normalized_ndarray(
            ::stride_align::hamming::dispatch_normalized_scores(query, targets));
      },
      nb::arg("query"),
      nb::arg("targets"));

  // --- Indel distance -------------------------------------------------
  //
  // Levenshtein restricted to insertions/deletions — no substitutions.
  // Equivalent to |a| + |b| - 2 * LCS(a, b). Normalized form divides by
  // (|a| + |b|), not max(|a|, |b|), so the score is bounded in [0, 1].
  //
  // SIMD path: Allison-Dix (1986) bit-parallel LCS for single-word
  // patterns (m <= 64), one target per 64-bit lane. Multi-word patterns
  // and wider-unicode inputs fall through to scalar DP.

  m.def(
      "indel_score",
      [](nb::handle query, nb::handle target) {
        return ::stride_align::indel::dispatch_score(query, target);
      },
      nb::arg("query"),
      nb::arg("target"));

  m.def(
      "indel_normalized_score",
      [](nb::handle query, nb::handle target) {
        return ::stride_align::indel::dispatch_normalized_score(query, target);
      },
      nb::arg("query"),
      nb::arg("target"));

  m.def(
      "indel_scores",
      [](nb::handle query, nb::handle targets) {
        std::vector<Score> result;
        if constexpr (requires {
                        Implementation::indel_scores(query, targets);
                      }) {
          result = Implementation::indel_scores(query, targets);
        } else {
          result = ::stride_align::indel::dispatch_scores(query, targets);
        }
        return as_score_ndarray(std::move(result));
      },
      nb::arg("query"),
      nb::arg("targets"));

  m.def(
      "indel_normalized_scores",
      [](nb::handle query, nb::handle targets) {
        std::vector<double> result;
        if constexpr (requires {
                        Implementation::indel_normalized_scores(query, targets);
                      }) {
          result = Implementation::indel_normalized_scores(query, targets);
        } else {
          result = ::stride_align::indel::dispatch_normalized_scores(
              query, targets);
        }
        return as_normalized_ndarray(std::move(result));
      },
      nb::arg("query"),
      nb::arg("targets"));

  // --- True Damerau-Levenshtein ---------------------------------------
  //
  // Unrestricted DL: each character can participate in multiple edits
  // (unlike OSA, which we ship under the `damerau_levenshtein_*` name).
  // Scalar DP only — no bit-parallel form yet (Hyyrö 2003 exists but
  // is significantly more complex than OSA's bit-parallel and rarely
  // the bottleneck).

  m.def(
      "true_damerau_levenshtein_score",
      [](nb::handle query, nb::handle target) {
        return ::stride_align::true_damerau::dispatch_score(query, target);
      },
      nb::arg("query"),
      nb::arg("target"));

  m.def(
      "true_damerau_levenshtein_normalized_score",
      [](nb::handle query, nb::handle target) {
        return ::stride_align::true_damerau::dispatch_normalized_score(
            query, target);
      },
      nb::arg("query"),
      nb::arg("target"));

  m.def(
      "true_damerau_levenshtein_scores",
      [](nb::handle query, nb::handle targets) {
        return as_score_ndarray(
            ::stride_align::true_damerau::dispatch_scores(query, targets));
      },
      nb::arg("query"),
      nb::arg("targets"));

  m.def(
      "true_damerau_levenshtein_normalized_scores",
      [](nb::handle query, nb::handle targets) {
        return as_normalized_ndarray(
            ::stride_align::true_damerau::dispatch_normalized_scores(
                query, targets));
      },
      nb::arg("query"),
      nb::arg("targets"));

  // --- Jaro / Jaro-Winkler ---------------------------------------------
  //
  // Both are similarity in [0, 1] (higher = closer); there's no
  // separate "distance" form, since Jaro doesn't admit a meaningful
  // bound to subtract from. Winkler adds a prefix bonus capped at
  // `prefix_cap` characters when the base Jaro score crosses
  // `prefix_threshold` (default 0.7 to match rapidfuzz).

  m.def(
      "jaro_similarity",
      [](nb::handle query, nb::handle target) {
        return ::stride_align::jaro::dispatch_similarity(query, target);
      },
      nb::arg("query"),
      nb::arg("target"));

  m.def(
      "jaro_winkler_similarity",
      [](nb::handle query,
         nb::handle target,
         double prefix_weight,
         double prefix_threshold,
         std::size_t prefix_cap) {
        return ::stride_align::jaro::dispatch_winkler_similarity(
            query, target, prefix_weight, prefix_threshold, prefix_cap);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("prefix_weight") = ::stride_align::jaro::kDefaultPrefixWeight,
      nb::arg("prefix_threshold") = ::stride_align::jaro::kDefaultPrefixThreshold,
      nb::arg("prefix_cap") = ::stride_align::jaro::kDefaultPrefixCap);

  m.def(
      "jaro_similarities",
      [](nb::handle query, nb::handle targets) {
        if constexpr (requires {
                        Implementation::jaro_similarities(query, targets);
                      }) {
          auto result = Implementation::jaro_similarities(query, targets);
          if (!result.empty()) {
            return as_normalized_ndarray(std::move(result));
          }
        }
        return as_normalized_ndarray(
            ::stride_align::jaro::dispatch_similarities(query, targets));
      },
      nb::arg("query"),
      nb::arg("targets"));

  m.def(
      "jaro_winkler_similarities",
      [](nb::handle query,
         nb::handle targets,
         double prefix_weight,
         double prefix_threshold,
         std::size_t prefix_cap) {
        if constexpr (requires {
                        Implementation::jaro_winkler_similarities(
                            query, targets,
                            prefix_weight, prefix_threshold, prefix_cap);
                      }) {
          auto result = Implementation::jaro_winkler_similarities(
              query, targets, prefix_weight, prefix_threshold, prefix_cap);
          if (!result.empty()) {
            return as_normalized_ndarray(std::move(result));
          }
        }
        return as_normalized_ndarray(
            ::stride_align::jaro::dispatch_winkler_similarities(
                query, targets, prefix_weight, prefix_threshold, prefix_cap));
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::kw_only(),
      nb::arg("prefix_weight") = ::stride_align::jaro::kDefaultPrefixWeight,
      nb::arg("prefix_threshold") = ::stride_align::jaro::kDefaultPrefixThreshold,
      nb::arg("prefix_cap") = ::stride_align::jaro::kDefaultPrefixCap);

  // -------------------------------------------------------------------
  // k-best selection (`*_top_k` and the unified `extract`)
  // -------------------------------------------------------------------
  //
  // Each `*_top_k` function runs the same SIMD batch kernel as the
  // matching `*_scores` entry point, then folds an O(N) average
  // partial-sort over the score vector into the same C++ call. The
  // result is `list[(target, score, index)]` — one Python boundary
  // crossing for "score N targets and return the best k", versus
  // *_scores → numpy.argpartition → indexed gather. See
  // src/cpp/topk.hpp for the selection helper and the rationale.
  //
  // Direction is encoded in the function name: `*_top_k` returns the
  // k *lowest* distances (smaller = closer); `*_normalized_top_k`
  // returns the k *highest* similarities (larger = closer). That
  // matches the underlying score's natural direction and avoids a
  // `descending=` flag the caller has to remember.
  //
  // `extract(query, targets, scorer, k)` is a thin convenience that
  // routes by Scorer enum to the same kernels. Smith-Waterman is not
  // in the enum because its score depends on match/mismatch/gap
  // parameters — call `smith_waterman_top_k` directly for that.

  // The user-facing `Scorer` is a Python IntEnum defined in
  // stride_align/__init__.py. nanobind's nb::enum_ would force a fresh
  // C++ type registration in every backend shared library (which
  // crashes nanobind), so the binding boundary takes a plain int and
  // re-interprets it as the C++ Scorer enum inside extract().

  m.def(
      "levenshtein_top_k",
      [](nb::handle query, nb::handle targets, std::size_t k) {
        PyObject* fast_targets = PySequence_Fast(
            targets.ptr(), "targets must be a sequence of target sequences");
        if (fast_targets == nullptr) {
          throw nb::python_error();
        }
        nb::object owner = nb::steal<nb::object>(fast_targets);
        PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
        std::vector<Score> scores;
        if constexpr (requires {
                        Implementation::levenshtein_scores(query, targets, ::stride_align::levenshtein::kNoCutoff);
                      }) {
          scores = Implementation::levenshtein_scores(query, targets, ::stride_align::levenshtein::kNoCutoff);
        } else {
          scores = ::stride_align::levenshtein::dispatch_scores(query, targets, ::stride_align::levenshtein::kNoCutoff);
        }
        return ::stride_align::topk::make_top_k(
            items, scores, k, /*higher_is_better=*/false);
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::arg("k") = std::size_t{5});

  m.def(
      "levenshtein_normalized_top_k",
      [](nb::handle query, nb::handle targets, std::size_t k) {
        PyObject* fast_targets = PySequence_Fast(
            targets.ptr(), "targets must be a sequence of target sequences");
        if (fast_targets == nullptr) {
          throw nb::python_error();
        }
        nb::object owner = nb::steal<nb::object>(fast_targets);
        PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
        std::vector<double> scores;
        if constexpr (requires {
                        Implementation::levenshtein_normalized_scores(query, targets, ::stride_align::levenshtein::kNoCutoff);
                      }) {
          scores = Implementation::levenshtein_normalized_scores(query, targets, ::stride_align::levenshtein::kNoCutoff);
        } else {
          scores = ::stride_align::levenshtein::dispatch_normalized_scores(
              query, targets, ::stride_align::levenshtein::kNoCutoff);
        }
        return ::stride_align::topk::make_top_k(
            items, scores, k, /*higher_is_better=*/true);
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::arg("k") = std::size_t{5});

  m.def(
      "damerau_levenshtein_top_k",
      [](nb::handle query, nb::handle targets, std::size_t k) {
        PyObject* fast_targets = PySequence_Fast(
            targets.ptr(), "targets must be a sequence of target sequences");
        if (fast_targets == nullptr) {
          throw nb::python_error();
        }
        nb::object owner = nb::steal<nb::object>(fast_targets);
        PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
        std::vector<Score> scores;
        if constexpr (requires {
                        Implementation::damerau_levenshtein_scores(query, targets);
                      }) {
          scores = Implementation::damerau_levenshtein_scores(query, targets);
        } else {
          scores = ::stride_align::levenshtein::dispatch_osa_scores(query, targets);
        }
        return ::stride_align::topk::make_top_k(
            items, scores, k, /*higher_is_better=*/false);
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::arg("k") = std::size_t{5});

  m.def(
      "damerau_levenshtein_normalized_top_k",
      [](nb::handle query, nb::handle targets, std::size_t k) {
        PyObject* fast_targets = PySequence_Fast(
            targets.ptr(), "targets must be a sequence of target sequences");
        if (fast_targets == nullptr) {
          throw nb::python_error();
        }
        nb::object owner = nb::steal<nb::object>(fast_targets);
        PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
        std::vector<double> scores;
        if constexpr (requires {
                        Implementation::damerau_levenshtein_normalized_scores(query, targets);
                      }) {
          scores = Implementation::damerau_levenshtein_normalized_scores(query, targets);
        } else {
          scores = ::stride_align::levenshtein::dispatch_osa_normalized_scores(
              query, targets);
        }
        return ::stride_align::topk::make_top_k(
            items, scores, k, /*higher_is_better=*/true);
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::arg("k") = std::size_t{5});

  m.def(
      "hamming_top_k",
      [](nb::handle query, nb::handle targets, std::size_t k) {
        PyObject* fast_targets = PySequence_Fast(
            targets.ptr(), "targets must be a sequence of target sequences");
        if (fast_targets == nullptr) {
          throw nb::python_error();
        }
        nb::object owner = nb::steal<nb::object>(fast_targets);
        PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
        auto scores = ::stride_align::hamming::dispatch_scores(query, targets);
        return ::stride_align::topk::make_top_k(
            items, scores, k, /*higher_is_better=*/false);
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::arg("k") = std::size_t{5});

  m.def(
      "hamming_normalized_top_k",
      [](nb::handle query, nb::handle targets, std::size_t k) {
        PyObject* fast_targets = PySequence_Fast(
            targets.ptr(), "targets must be a sequence of target sequences");
        if (fast_targets == nullptr) {
          throw nb::python_error();
        }
        nb::object owner = nb::steal<nb::object>(fast_targets);
        PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
        auto scores = ::stride_align::hamming::dispatch_normalized_scores(query, targets);
        return ::stride_align::topk::make_top_k(
            items, scores, k, /*higher_is_better=*/true);
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::arg("k") = std::size_t{5});

  m.def(
      "indel_top_k",
      [](nb::handle query, nb::handle targets, std::size_t k) {
        PyObject* fast_targets = PySequence_Fast(
            targets.ptr(), "targets must be a sequence of target sequences");
        if (fast_targets == nullptr) {
          throw nb::python_error();
        }
        nb::object owner = nb::steal<nb::object>(fast_targets);
        PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
        std::vector<Score> scores;
        if constexpr (requires {
                        Implementation::indel_scores(query, targets);
                      }) {
          scores = Implementation::indel_scores(query, targets);
        } else {
          scores = ::stride_align::indel::dispatch_scores(query, targets);
        }
        return ::stride_align::topk::make_top_k(
            items, scores, k, /*higher_is_better=*/false);
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::arg("k") = std::size_t{5});

  m.def(
      "indel_normalized_top_k",
      [](nb::handle query, nb::handle targets, std::size_t k) {
        PyObject* fast_targets = PySequence_Fast(
            targets.ptr(), "targets must be a sequence of target sequences");
        if (fast_targets == nullptr) {
          throw nb::python_error();
        }
        nb::object owner = nb::steal<nb::object>(fast_targets);
        PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
        std::vector<double> scores;
        if constexpr (requires {
                        Implementation::indel_normalized_scores(query, targets);
                      }) {
          scores = Implementation::indel_normalized_scores(query, targets);
        } else {
          scores = ::stride_align::indel::dispatch_normalized_scores(
              query, targets);
        }
        return ::stride_align::topk::make_top_k(
            items, scores, k, /*higher_is_better=*/true);
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::arg("k") = std::size_t{5});

  m.def(
      "true_damerau_levenshtein_top_k",
      [](nb::handle query, nb::handle targets, std::size_t k) {
        PyObject* fast_targets = PySequence_Fast(
            targets.ptr(),
            "targets must be a sequence of target sequences");
        if (fast_targets == nullptr) {
          throw nb::python_error();
        }
        nb::object owner = nb::steal<nb::object>(fast_targets);
        PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
        auto scores =
            ::stride_align::true_damerau::dispatch_scores(query, targets);
        return ::stride_align::topk::make_top_k(
            items, scores, k, /*higher_is_better=*/false);
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::arg("k") = std::size_t{5});

  m.def(
      "true_damerau_levenshtein_normalized_top_k",
      [](nb::handle query, nb::handle targets, std::size_t k) {
        PyObject* fast_targets = PySequence_Fast(
            targets.ptr(),
            "targets must be a sequence of target sequences");
        if (fast_targets == nullptr) {
          throw nb::python_error();
        }
        nb::object owner = nb::steal<nb::object>(fast_targets);
        PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
        auto scores =
            ::stride_align::true_damerau::dispatch_normalized_scores(
                query, targets);
        return ::stride_align::topk::make_top_k(
            items, scores, k, /*higher_is_better=*/true);
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::arg("k") = std::size_t{5});

  m.def(
      "jaro_top_k",
      [](nb::handle query, nb::handle targets, std::size_t k) {
        PyObject* fast_targets = PySequence_Fast(
            targets.ptr(), "targets must be a sequence of target sequences");
        if (fast_targets == nullptr) {
          throw nb::python_error();
        }
        nb::object owner = nb::steal<nb::object>(fast_targets);
        PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
        std::vector<double> scores;
        if constexpr (requires {
                        Implementation::jaro_similarities(query, targets);
                      }) {
          scores = Implementation::jaro_similarities(query, targets);
        }
        if (scores.empty()) {
          scores = ::stride_align::jaro::dispatch_similarities(query, targets);
        }
        return ::stride_align::topk::make_top_k(
            items, scores, k, /*higher_is_better=*/true);
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::arg("k") = std::size_t{5});

  m.def(
      "jaro_winkler_top_k",
      [](nb::handle query,
         nb::handle targets,
         std::size_t k,
         double prefix_weight,
         double prefix_threshold,
         std::size_t prefix_cap) {
        PyObject* fast_targets = PySequence_Fast(
            targets.ptr(), "targets must be a sequence of target sequences");
        if (fast_targets == nullptr) {
          throw nb::python_error();
        }
        nb::object owner = nb::steal<nb::object>(fast_targets);
        PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
        std::vector<double> scores;
        if constexpr (requires {
                        Implementation::jaro_winkler_similarities(
                            query, targets,
                            prefix_weight, prefix_threshold, prefix_cap);
                      }) {
          scores = Implementation::jaro_winkler_similarities(
              query, targets, prefix_weight, prefix_threshold, prefix_cap);
        }
        if (scores.empty()) {
          scores = ::stride_align::jaro::dispatch_winkler_similarities(
              query, targets, prefix_weight, prefix_threshold, prefix_cap);
        }
        return ::stride_align::topk::make_top_k(
            items, scores, k, /*higher_is_better=*/true);
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::kw_only(),
      nb::arg("k") = std::size_t{5},
      nb::arg("prefix_weight") = ::stride_align::jaro::kDefaultPrefixWeight,
      nb::arg("prefix_threshold") = ::stride_align::jaro::kDefaultPrefixThreshold,
      nb::arg("prefix_cap") = ::stride_align::jaro::kDefaultPrefixCap);

  m.def(
      "smith_waterman_top_k",
      [](nb::handle query,
         nb::handle targets,
         std::size_t k,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object gap_open_score,
         nb::object gap_extend_score,
         nb::object width) {
        PyObject* fast_targets = PySequence_Fast(
            targets.ptr(), "targets must be a sequence of target sequences");
        if (fast_targets == nullptr) {
          throw nb::python_error();
        }
        nb::object owner = nb::steal<nb::object>(fast_targets);
        PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
        const unsigned int forced_width =
            width.is_none() ? 0U : nb::cast<unsigned int>(width);
        const auto gaps =
            resolve_gap_scores(gap_score, gap_open_score, gap_extend_score);
        std::vector<Score> scores;
        if (!gaps.is_linear()) {
          if constexpr (requires {
                          Implementation::smith_waterman_affine_scores(
                              query,
                              targets,
                              match_score,
                              mismatch_score,
                              gaps.open,
                              gaps.extend,
                              forced_width);
                        }) {
            if (gaps.open <= 0 && gaps.extend <= 0) {
              scores = Implementation::smith_waterman_affine_scores(
                  query,
                  targets,
                  match_score,
                  mismatch_score,
                  gaps.open,
                  gaps.extend,
                  forced_width);
            }
          }
        } else {
          if constexpr (requires {
                          Implementation::smith_waterman_scores(
                              query,
                              targets,
                              match_score,
                              mismatch_score,
                              gaps.open,
                              forced_width);
                        }) {
            if (gaps.open <= 0) {
              scores = Implementation::smith_waterman_scores(
                  query,
                  targets,
                  match_score,
                  mismatch_score,
                  gaps.open,
                  forced_width);
            }
          }
        }
        if (scores.empty()) {
          // Fall back to scalar per-target like smith_waterman_scores does.
          const auto count =
              static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
          scores.reserve(count);
          for (std::size_t i = 0; i < count; ++i) {
            nb::handle target(items[i]);
            if (!gaps.is_linear()) {
              scores.push_back(call_smith_waterman_affine_score<Implementation>(
                  query,
                  target,
                  match_score,
                  mismatch_score,
                  gaps.open,
                  gaps.extend,
                  forced_width));
            } else {
              scores.push_back(Implementation::smith_waterman_score(
                  query,
                  target,
                  match_score,
                  mismatch_score,
                  gaps.open,
                  forced_width));
            }
          }
        }
        // Smith-Waterman is a similarity score — higher is better.
        return ::stride_align::topk::make_top_k(
            items, scores, k, /*higher_is_better=*/true);
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::kw_only(),
      nb::arg("k") = std::size_t{5},
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("gap_open_score") = nb::none(),
      nb::arg("gap_extend_score") = nb::none(),
      nb::arg("width") = nb::none());

  m.def(
      "extract",
      [](nb::handle query,
         nb::handle targets,
         int scorer_int,
         std::size_t k) {
        PyObject* fast_targets = PySequence_Fast(
            targets.ptr(), "targets must be a sequence of target sequences");
        if (fast_targets == nullptr) {
          throw nb::python_error();
        }
        nb::object owner = nb::steal<nb::object>(fast_targets);
        PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);
        using ::stride_align::topk::Scorer;
        const auto scorer = static_cast<Scorer>(scorer_int);
        switch (scorer) {
          case Scorer::Levenshtein: {
            std::vector<Score> scores;
            if constexpr (requires {
                            Implementation::levenshtein_scores(query, targets, ::stride_align::levenshtein::kNoCutoff);
                          }) {
              scores = Implementation::levenshtein_scores(query, targets, ::stride_align::levenshtein::kNoCutoff);
            } else {
              scores = ::stride_align::levenshtein::dispatch_scores(
                  query, targets, ::stride_align::levenshtein::kNoCutoff);
            }
            return ::stride_align::topk::make_top_k(items, scores, k, false);
          }
          case Scorer::LevenshteinNormalized: {
            std::vector<double> scores;
            if constexpr (requires {
                            Implementation::levenshtein_normalized_scores(
                                query, targets, ::stride_align::levenshtein::kNoCutoff);
                          }) {
              scores = Implementation::levenshtein_normalized_scores(
                  query, targets, ::stride_align::levenshtein::kNoCutoff);
            } else {
              scores = ::stride_align::levenshtein::dispatch_normalized_scores(
                  query, targets, ::stride_align::levenshtein::kNoCutoff);
            }
            return ::stride_align::topk::make_top_k(items, scores, k, true);
          }
          case Scorer::DamerauLevenshtein: {
            std::vector<Score> scores;
            if constexpr (requires {
                            Implementation::damerau_levenshtein_scores(query, targets);
                          }) {
              scores = Implementation::damerau_levenshtein_scores(query, targets);
            } else {
              scores = ::stride_align::levenshtein::dispatch_osa_scores(query, targets);
            }
            return ::stride_align::topk::make_top_k(items, scores, k, false);
          }
          case Scorer::DamerauLevenshteinNormalized: {
            std::vector<double> scores;
            if constexpr (requires {
                            Implementation::damerau_levenshtein_normalized_scores(
                                query, targets);
                          }) {
              scores = Implementation::damerau_levenshtein_normalized_scores(
                  query, targets);
            } else {
              scores =
                  ::stride_align::levenshtein::dispatch_osa_normalized_scores(
                      query, targets);
            }
            return ::stride_align::topk::make_top_k(items, scores, k, true);
          }
          case Scorer::Hamming: {
            auto scores =
                ::stride_align::hamming::dispatch_scores(query, targets);
            return ::stride_align::topk::make_top_k(items, scores, k, false);
          }
          case Scorer::HammingNormalized: {
            auto scores = ::stride_align::hamming::dispatch_normalized_scores(
                query, targets);
            return ::stride_align::topk::make_top_k(items, scores, k, true);
          }
          case Scorer::Indel: {
            std::vector<Score> scores;
            if constexpr (requires {
                            Implementation::indel_scores(query, targets);
                          }) {
              scores = Implementation::indel_scores(query, targets);
            } else {
              scores =
                  ::stride_align::indel::dispatch_scores(query, targets);
            }
            return ::stride_align::topk::make_top_k(items, scores, k, false);
          }
          case Scorer::IndelNormalized: {
            std::vector<double> scores;
            if constexpr (requires {
                            Implementation::indel_normalized_scores(
                                query, targets);
                          }) {
              scores =
                  Implementation::indel_normalized_scores(query, targets);
            } else {
              scores = ::stride_align::indel::dispatch_normalized_scores(
                  query, targets);
            }
            return ::stride_align::topk::make_top_k(items, scores, k, true);
          }
          case Scorer::TrueDamerauLevenshtein: {
            auto scores =
                ::stride_align::true_damerau::dispatch_scores(
                    query, targets);
            return ::stride_align::topk::make_top_k(
                items, scores, k, false);
          }
          case Scorer::TrueDamerauLevenshteinNormalized: {
            auto scores =
                ::stride_align::true_damerau::dispatch_normalized_scores(
                    query, targets);
            return ::stride_align::topk::make_top_k(items, scores, k, true);
          }
          case Scorer::Jaro: {
            std::vector<double> scores;
            if constexpr (requires {
                            Implementation::jaro_similarities(query, targets);
                          }) {
              scores = Implementation::jaro_similarities(query, targets);
            }
            if (scores.empty()) {
              scores = ::stride_align::jaro::dispatch_similarities(
                  query, targets);
            }
            return ::stride_align::topk::make_top_k(items, scores, k, true);
          }
          case Scorer::JaroWinkler: {
            // extract() uses default Winkler parameters; callers who
            // need a non-default prefix_weight / threshold / cap should
            // call jaro_winkler_top_k directly.
            std::vector<double> scores;
            if constexpr (requires {
                            Implementation::jaro_winkler_similarities(
                                query, targets,
                                ::stride_align::jaro::kDefaultPrefixWeight,
                                ::stride_align::jaro::kDefaultPrefixThreshold,
                                ::stride_align::jaro::kDefaultPrefixCap);
                          }) {
              scores = Implementation::jaro_winkler_similarities(
                  query, targets,
                  ::stride_align::jaro::kDefaultPrefixWeight,
                  ::stride_align::jaro::kDefaultPrefixThreshold,
                  ::stride_align::jaro::kDefaultPrefixCap);
            }
            if (scores.empty()) {
              scores = ::stride_align::jaro::dispatch_winkler_similarities(
                  query,
                  targets,
                  ::stride_align::jaro::kDefaultPrefixWeight,
                  ::stride_align::jaro::kDefaultPrefixThreshold,
                  ::stride_align::jaro::kDefaultPrefixCap);
            }
            return ::stride_align::topk::make_top_k(items, scores, k, true);
          }
        }
        PyErr_SetString(PyExc_ValueError, "unknown Scorer value");
        throw nb::python_error();
      },
      nb::arg("query"),
      nb::arg("targets"),
      nb::kw_only(),
      nb::arg("scorer"),
      nb::arg("k") = std::size_t{5});

  // -------------------------------------------------------------------
  // cdist: all-pairs distance / similarity matrix
  // -------------------------------------------------------------------
  //
  // The scorer can be passed three ways:
  //   * `Scorer.LEVENSHTEIN` etc — the IntEnum directly (Python int).
  //   * `_BACKEND.levenshtein_scores` etc — a C++-bound function from
  //     this module, looked up in the per-module scorer table built
  //     below.
  //   * Any other callable registered via `_register_scorer(fn, id)`,
  //     used by stride_align/__init__.py to register the top-level
  //     Python wrappers (which are what users normally pass).
  //
  // All three forms resolve to a single Scorer enum value in C++ once,
  // up front. The per-row dispatch is a plain C++ switch in
  // cdist_impl — no per-row Python calls.

  using ::stride_align::topk::Scorer;
  // Per-module map: PyObject* (function) -> Scorer enum value. Static
  // local: one map per `bind_backend_module<Implementation>`
  // instantiation, so each backend (.so) has its own table populated
  // with its own bound functions plus whatever the Python wrapper
  // layer registers later.
  static std::unordered_map<PyObject*, int> scorer_map;

  auto register_attr = [&m](const char* name, Scorer s) {
    if (m.attr(name).ptr() != nullptr) {
      scorer_map[m.attr(name).ptr()] = static_cast<int>(s);
    }
  };
  register_attr("levenshtein_scores", Scorer::Levenshtein);
  register_attr("levenshtein_normalized_scores", Scorer::LevenshteinNormalized);
  register_attr("damerau_levenshtein_scores", Scorer::DamerauLevenshtein);
  register_attr(
      "damerau_levenshtein_normalized_scores",
      Scorer::DamerauLevenshteinNormalized);
  register_attr("hamming_scores", Scorer::Hamming);
  register_attr("hamming_normalized_scores", Scorer::HammingNormalized);
  register_attr("indel_scores", Scorer::Indel);
  register_attr("indel_normalized_scores", Scorer::IndelNormalized);
  register_attr("true_damerau_levenshtein_scores",
                Scorer::TrueDamerauLevenshtein);
  register_attr("true_damerau_levenshtein_normalized_scores",
                Scorer::TrueDamerauLevenshteinNormalized);
  register_attr("jaro_similarities", Scorer::Jaro);
  register_attr("jaro_winkler_similarities", Scorer::JaroWinkler);

  m.def(
      "_register_scorer",
      [](nb::object fn, int scorer_id) {
        scorer_map[fn.ptr()] = scorer_id;
      },
      nb::arg("fn"),
      nb::arg("scorer_id"));

  m.def(
      "cdist",
      [](nb::object queries,
         nb::object targets,
         nb::object scorer_obj,
         nb::object tqdm_factory,
         std::size_t cpu_count,
         double prefix_weight,
         double prefix_threshold,
         std::size_t prefix_cap) {
        int scorer_id;
        // IntEnum subclasses int, so PyLong_Check is the fast path
        // for the canonical `Scorer.LEVENSHTEIN` usage.
        if (PyLong_Check(scorer_obj.ptr())) {
          scorer_id = nb::cast<int>(scorer_obj);
        } else {
          auto it = scorer_map.find(scorer_obj.ptr());
          if (it == scorer_map.end()) {
            PyErr_SetString(
                PyExc_ValueError,
                "cdist scorer must be a Scorer enum value or a known "
                "stride_align scoring function (e.g. "
                "stride_align.levenshtein_scores).");
            throw nb::python_error();
          }
          scorer_id = it->second;
        }

        if constexpr (requires {
                        Implementation::cdist(queries, targets, scorer_id,
                                              tqdm_factory, cpu_count,
                                              prefix_weight,
                                              prefix_threshold, prefix_cap);
                      }) {
          return Implementation::cdist(
              queries, targets, scorer_id, tqdm_factory, cpu_count,
              prefix_weight, prefix_threshold, prefix_cap);
        } else {
          PyErr_SetString(
              PyExc_NotImplementedError,
              "cdist is not implemented for this backend; "
              "use a SIMD-capable backend (any non-generic).");
          throw nb::python_error();
        }
      },
      nb::arg("queries"),
      nb::arg("targets"),
      nb::kw_only(),
      nb::arg("scorer"),
      nb::arg("tqdm") = nb::none(),
      nb::arg("cpu_count") = std::size_t{1},
      nb::arg("prefix_weight") = ::stride_align::jaro::kDefaultPrefixWeight,
      nb::arg("prefix_threshold") = ::stride_align::jaro::kDefaultPrefixThreshold,
      nb::arg("prefix_cap") = ::stride_align::jaro::kDefaultPrefixCap);

  // ----------------------------------------------------------------
  // cdist_above_threshold: streaming, filtered cdist.
  //
  // Returns an iterator that yields (score, query, target) tuples
  // for every pair whose normalized similarity is at least the
  // caller's threshold. Memory stays O(N + M) regardless of the
  // match count — workers feed a bounded queue, the main thread
  // drains it lazily via __next__.
  // ----------------------------------------------------------------

  nb::class_<::stride_align::cdist_threshold::ThresholdIterator>(
      m, "_ThresholdIterator")
      .def(
          "__iter__",
          [](nb::handle self) {
            return nb::borrow<nb::object>(self);
          })
      .def(
          "__next__",
          [](::stride_align::cdist_threshold::ThresholdIterator& self) {
            return self.next();
          });

  m.def(
      "cdist_above_threshold",
      [](nb::object queries,
         nb::object targets,
         nb::object scorer_obj,
         double threshold,
         nb::object tqdm_factory,
         std::size_t cpu_count,
         double prefix_weight,
         double prefix_threshold,
         std::size_t prefix_cap) {
        int scorer_id;
        if (PyLong_Check(scorer_obj.ptr())) {
          scorer_id = nb::cast<int>(scorer_obj);
        } else {
          auto it = scorer_map.find(scorer_obj.ptr());
          if (it == scorer_map.end()) {
            PyErr_SetString(
                PyExc_ValueError,
                "cdist_above_threshold scorer must be a normalized "
                "Scorer enum value or a known stride_align scoring "
                "function (e.g. stride_align.jaro_similarities).");
            throw nb::python_error();
          }
          scorer_id = it->second;
        }

        if constexpr (requires {
                        Implementation::cdist_above_threshold(
                            queries, targets, scorer_id, threshold,
                            tqdm_factory, cpu_count, prefix_weight,
                            prefix_threshold, prefix_cap);
                      }) {
          return Implementation::cdist_above_threshold(
              queries, targets, scorer_id, threshold, tqdm_factory,
              cpu_count, prefix_weight, prefix_threshold, prefix_cap);
        } else {
          PyErr_SetString(
              PyExc_NotImplementedError,
              "cdist_above_threshold is not implemented for this backend; "
              "use a SIMD-capable backend (any non-generic).");
          throw nb::python_error();
        }
      },
      nb::arg("queries"),
      nb::arg("targets"),
      nb::kw_only(),
      nb::arg("scorer"),
      nb::arg("threshold"),
      nb::arg("tqdm") = nb::none(),
      nb::arg("cpu_count") = std::size_t{1},
      nb::arg("prefix_weight") = ::stride_align::jaro::kDefaultPrefixWeight,
      nb::arg("prefix_threshold") = ::stride_align::jaro::kDefaultPrefixThreshold,
      nb::arg("prefix_cap") = ::stride_align::jaro::kDefaultPrefixCap);

  // ----------------------------------------------------------------
  // cdist_top_k: heap-based variant of the streaming filtered cdist.
  // Returns an unsorted list of the k highest-scoring pairs.
  // ----------------------------------------------------------------

  m.def(
      "cdist_top_k",
      [](nb::object queries,
         nb::object targets,
         nb::object scorer_obj,
         std::size_t k,
         nb::object tqdm_factory,
         std::size_t cpu_count,
         bool reject_duplicates,
         double prefix_weight,
         double prefix_threshold,
         std::size_t prefix_cap) {
        int scorer_id;
        if (PyLong_Check(scorer_obj.ptr())) {
          scorer_id = nb::cast<int>(scorer_obj);
        } else {
          auto it = scorer_map.find(scorer_obj.ptr());
          if (it == scorer_map.end()) {
            PyErr_SetString(
                PyExc_ValueError,
                "cdist_top_k scorer must be a normalized Scorer enum "
                "value or a known stride_align scoring function.");
            throw nb::python_error();
          }
          scorer_id = it->second;
        }

        if constexpr (requires {
                        Implementation::cdist_top_k(
                            queries, targets, scorer_id, k, tqdm_factory,
                            cpu_count, reject_duplicates, prefix_weight,
                            prefix_threshold, prefix_cap);
                      }) {
          return Implementation::cdist_top_k(
              queries, targets, scorer_id, k, tqdm_factory, cpu_count,
              reject_duplicates, prefix_weight, prefix_threshold,
              prefix_cap);
        } else {
          PyErr_SetString(
              PyExc_NotImplementedError,
              "cdist_top_k is not implemented for this backend; "
              "use a SIMD-capable backend (any non-generic).");
          throw nb::python_error();
        }
      },
      nb::arg("queries"),
      nb::arg("targets"),
      nb::kw_only(),
      nb::arg("scorer"),
      nb::arg("k"),
      nb::arg("tqdm") = nb::none(),
      nb::arg("cpu_count") = std::size_t{1},
      nb::arg("reject_duplicates") = false,
      nb::arg("prefix_weight") = ::stride_align::jaro::kDefaultPrefixWeight,
      nb::arg("prefix_threshold") = ::stride_align::jaro::kDefaultPrefixThreshold,
      nb::arg("prefix_cap") = ::stride_align::jaro::kDefaultPrefixCap);
}

}  // namespace stride_align::bindings
