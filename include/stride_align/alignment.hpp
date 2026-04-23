#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <nanobind/nanobind.h>

namespace stride_align {

namespace nb = nanobind;

using Score = std::int64_t;

struct AlignmentResult {
  Score score = 0;
  std::size_t query_start = 0;
  std::size_t query_end = 0;
  std::size_t target_start = 0;
  std::size_t target_end = 0;
  nb::object aligned_query;
  nb::object aligned_target;
  std::string operations;
};

}  // namespace stride_align
