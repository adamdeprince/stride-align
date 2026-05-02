#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

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

inline std::string expand_cigar(std::string_view cigar) {
  std::string operations;
  std::size_t count = 0;
  for (const char value : cigar) {
    if (value >= '0' && value <= '9') {
      count = count * 10U + static_cast<std::size_t>(value - '0');
      continue;
    }
    operations.append(count, value);
    count = 0;
  }
  return operations;
}

class ReverseCigarBuilder {
 public:
  void push(char operation) {
    if (!runs_.empty() && runs_.back().operation == operation) {
      ++runs_.back().count;
      return;
    }
    runs_.push_back({operation, 1U});
  }

  std::string str() const {
    std::string cigar;
    for (auto it = runs_.rbegin(); it != runs_.rend(); ++it) {
      cigar += std::to_string(it->count);
      cigar.push_back(it->operation);
    }
    return cigar;
  }

 private:
  struct Run {
    char operation;
    std::size_t count;
  };

  std::vector<Run> runs_;
};

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

inline AlignmentPath make_alignment_path_from_cigar(
    Score score,
    std::size_t query_start,
    std::size_t query_end,
    std::size_t target_start,
    std::size_t target_end,
    std::string_view cigar) {
  AlignmentPath path;
  path.score = score;
  path.query_start = query_start;
  path.query_end = query_end;
  path.target_start = target_start;
  path.target_end = target_end;
  path.cigar = std::string(cigar);

  std::size_t count = 0;
  for (const char value : cigar) {
    if (value >= '0' && value <= '9') {
      count = count * 10U + static_cast<std::size_t>(value - '0');
      continue;
    }
    path.operations.append(count, value);
    path.aligned_length += count;
    switch (value) {
      case 'M':
        path.matches += count;
        break;
      case 'X':
        path.mismatches += count;
        break;
      case 'I':
        path.insertions += count;
        break;
      case 'D':
        path.deletions += count;
        break;
      default:
        break;
    }
    count = 0;
  }
  return path;
}

inline AlignmentPath make_global_alignment_path_from_cigar(
    Score score,
    std::size_t query_size,
    std::size_t target_size,
    std::string_view cigar) {
  return make_alignment_path_from_cigar(score, 0U, query_size, 0U, target_size, cigar);
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
