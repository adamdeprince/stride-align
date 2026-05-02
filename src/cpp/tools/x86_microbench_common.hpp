#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "farrar_preprocess.hpp"
#include "stride_align/alignment.hpp"

namespace stride_align::microbench {

struct Options {
  std::string backend = "avx2";
  std::string variant = "nw-affine-score";
  std::string pass_name = "english";
  std::string shape = "1:1";
  std::size_t query_length = 1024;
  std::size_t target_length = 1024;
  std::size_t many_count = 8;
  std::size_t iterations = 1000;
  std::size_t warmups = 10;
  unsigned int width = 16;
  int seed = 1;
  stride_align::Score match_score = 2;
  stride_align::Score mismatch_score = -1;
  stride_align::Score gap_open_score = -2;
  stride_align::Score gap_extend_score = -1;
};

struct Workload {
  std::vector<std::uint8_t> query;
  std::vector<std::vector<std::uint8_t>> targets;
};

struct PreparedWorkload {
  stride_align::PreparedFarrarAlignment single;
  stride_align::PreparedFarrarBatchAlignment batch;
};

struct RunResult {
  double prepare_seconds = 0.0;
  double run_seconds = 0.0;
  std::uint64_t checksum = 0;
  stride_align::Score last_score = 0;
  std::size_t calls = 0;
  std::size_t scored_targets = 0;
};

bool supports_avx2() noexcept;
bool supports_avx512bwvl() noexcept;
bool supports_parasail() noexcept;

RunResult run_avx2_backend(const PreparedWorkload& prepared, const Options& options);
RunResult run_avx512bwvl_backend(const PreparedWorkload& prepared, const Options& options);
RunResult run_parasail_backend(const PreparedWorkload& prepared, const Options& options);

}  // namespace stride_align::microbench
