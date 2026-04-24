#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <sstream>

#include "stride_align/alignment.hpp"

namespace stride_align::bindings {

namespace nb = nanobind;

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

template <typename Implementation>
void bind_backend_module(nb::module_& m, const char* doc) {
  m.doc() = doc;

  bind_alignment_result_type(m);

  m.def(
      "smith_waterman_score",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        return Implementation::smith_waterman_score(
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("width") = nb::none());

  m.def(
      "smith_waterman_path",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        return Implementation::smith_waterman_path(
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("width") = nb::none());

  m.def(
      "needleman_wunsch_score",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        return Implementation::needleman_wunsch_score(
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("width") = nb::none());

  m.def(
      "needleman_wunsch_path",
      [](nb::handle query,
         nb::handle target,
         Score match_score,
         Score mismatch_score,
         Score gap_score,
         nb::object width) {
        const unsigned int forced_width = width.is_none() ? 0U : nb::cast<unsigned int>(width);
        return Implementation::needleman_wunsch_path(
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            forced_width);
      },
      nb::arg("query"),
      nb::arg("target"),
      nb::kw_only(),
      nb::arg("match_score") = 2,
      nb::arg("mismatch_score") = -1,
      nb::arg("gap_score") = -1,
      nb::arg("width") = nb::none());
}

}  // namespace stride_align::bindings
