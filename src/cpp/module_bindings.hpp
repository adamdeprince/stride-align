#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <sstream>
#include <vector>

#include "affine.hpp"
#include "backends/profile_traceback.hpp"
#include "levenshtein_dispatch.hpp"
#include "stride_align/alignment.hpp"
#include "stride_align/levenshtein.hpp"

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

  m.def(
      "levenshtein_score",
      [](nb::handle query, nb::handle target) {
        return ::stride_align::levenshtein::dispatch_score(query, target);
      },
      nb::arg("query"),
      nb::arg("target"));

  m.def(
      "levenshtein_normalized_score",
      [](nb::handle query, nb::handle target) {
        return ::stride_align::levenshtein::dispatch_normalized_score(query, target);
      },
      nb::arg("query"),
      nb::arg("target"));

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
      [](nb::handle query, nb::handle targets) {
        if constexpr (requires {
                        Implementation::levenshtein_scores(query, targets);
                      }) {
          return as_score_ndarray(Implementation::levenshtein_scores(query, targets));
        } else {
          return as_score_ndarray(
              ::stride_align::levenshtein::dispatch_scores(query, targets));
        }
      },
      nb::arg("query"),
      nb::arg("targets"));

  m.def(
      "levenshtein_normalized_scores",
      [](nb::handle query, nb::handle targets) {
        if constexpr (requires {
                        Implementation::levenshtein_normalized_scores(query, targets);
                      }) {
          return as_normalized_ndarray(
              Implementation::levenshtein_normalized_scores(query, targets));
        } else {
          return as_normalized_ndarray(
              ::stride_align::levenshtein::dispatch_normalized_scores(query, targets));
        }
      },
      nb::arg("query"),
      nb::arg("targets"));
}

}  // namespace stride_align::bindings
