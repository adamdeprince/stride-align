#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "farrar_preprocess.hpp"
#include "stride_align/alignment.hpp"

namespace stride_align::microbench {

struct Options {
#if defined(__aarch64__) || defined(_M_ARM64)
  std::string backend = "neon";
#else
  std::string backend = "avx2";
#endif
  std::string variant = "nw-affine-score";
  std::string pass_name = "english";
  std::string shape = "1:1";
  std::string profile_layout = "auto";
  std::string sw_farrar_i32_strategy = "auto";
  std::size_t query_length = 1024;
  std::size_t target_length = 1024;
  std::size_t many_count = 8;
  std::size_t iterations = 1000;
  std::size_t warmups = 10;
  std::size_t samples = 1;
  std::size_t profile_block_size = 64;
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

// Dual-target deferred exact-fill correctness fuzz (AVX-512 i16×1024 path).
// Returns 0 on success; prints failures to stderr and returns non-zero on mismatch.
int verify_dual_sw_exact_fill_avx512bwvl();

RunResult run_avx2_backend(const PreparedWorkload& prepared, const Options& options);
RunResult run_avx512bwvl_backend(const PreparedWorkload& prepared, const Options& options);
RunResult run_parasail_backend(const PreparedWorkload& prepared, const Options& options);

#if defined(__aarch64__) || defined(_M_ARM64)
bool supports_neon() noexcept;
RunResult run_neon_backend(const PreparedWorkload& prepared, const Options& options);
#endif

}  // namespace stride_align::microbench
