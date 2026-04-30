#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

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

struct AlignmentPath {
  Score score = 0;
  std::size_t query_start = 0;
  std::size_t query_end = 0;
  std::size_t target_start = 0;
  std::size_t target_end = 0;
  std::string operations;
  std::string cigar;
  std::size_t matches = 0;
  std::size_t mismatches = 0;
  std::size_t insertions = 0;
  std::size_t deletions = 0;
  std::size_t aligned_length = 0;
};

inline std::string build_cigar(std::string_view operations) {
  std::string cigar;
  if (operations.empty()) {
    return cigar;
  }

  char current = operations.front();
  std::size_t count = 0;
  for (const char operation : operations) {
    if (operation == current) {
      ++count;
      continue;
    }
    cigar += std::to_string(count);
    cigar.push_back(current);
    current = operation;
    count = 1;
  }
  cigar += std::to_string(count);
  cigar.push_back(current);
  return cigar;
}

inline AlignmentPath make_alignment_path(
    Score score,
    std::size_t query_start,
    std::size_t query_end,
    std::size_t target_start,
    std::size_t target_end,
    std::string_view operations) {
  AlignmentPath path;
  path.score = score;
  path.query_start = query_start;
  path.query_end = query_end;
  path.target_start = target_start;
  path.target_end = target_end;
  path.operations = std::string(operations);
  path.cigar = build_cigar(operations);
  path.aligned_length = operations.size();

  for (const char operation : operations) {
    switch (operation) {
      case 'M':
        ++path.matches;
        break;
      case 'X':
        ++path.mismatches;
        break;
      case 'I':
        ++path.insertions;
        break;
      case 'D':
        ++path.deletions;
        break;
      default:
        break;
    }
  }

  return path;
}

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
