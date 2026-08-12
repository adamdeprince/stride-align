#pragma once

#include <cstddef>
#include <string>

#include <nanobind/nanobind.h>

#include "stride_align/alignment.hpp"

namespace stride_align {

// Python-facing traceback result. The host-neutral core returns AlignmentPath;
// only this adapter owns Python objects used to materialize aligned sequences.
struct AlignmentResult {
  Score score = 0;
  std::size_t query_start = 0;
  std::size_t query_end = 0;
  std::size_t target_start = 0;
  std::size_t target_end = 0;
  nanobind::object aligned_query;
  nanobind::object aligned_target;
  std::string operations;
};

inline AlignmentPath make_alignment_path(const AlignmentResult& result) {
  return make_alignment_path(
      result.score,
      result.query_start,
      result.query_end,
      result.target_start,
      result.target_end,
      result.operations);
}

}  // namespace stride_align
