#pragma once

#include <chrono>
#include <cstdint>
#include <stdexcept>

#include "backends/farrar_fixed_kernel.hpp"
#include "x86_microbench_common.hpp"

namespace stride_align::microbench {

using Clock = std::chrono::steady_clock;

inline std::uint64_t checksum_add(std::uint64_t checksum, stride_align::Score score) noexcept {
  const auto mixed = static_cast<std::uint64_t>(score) + 0x9E3779B97F4A7C15ULL;
  return (checksum ^ mixed) * 1099511628211ULL;
}

template <template <typename, typename> class OpsTemplate>
RunResult run_prepared_single(
    const stride_align::PreparedFarrarAlignment& prepared_alignment,
    const Options& options) {
  RunResult result;
  const auto prepare_start = Clock::now();
  auto prepared = stride_align::farrar_fixed_kernel::detail::prepare_affine_score<
      OpsTemplate,
      true>(
      prepared_alignment,
      options.match_score,
      options.mismatch_score,
      options.gap_open_score,
      options.gap_extend_score);
  const auto prepare_end = Clock::now();

  for (std::size_t index = 0; index < options.warmups; ++index) {
    result.last_score =
        stride_align::farrar_fixed_kernel::detail::dispatch_prepared_global_affine_score<
            OpsTemplate>(prepared);
  }

  const auto run_start = Clock::now();
  for (std::size_t index = 0; index < options.iterations; ++index) {
    result.last_score =
        stride_align::farrar_fixed_kernel::detail::dispatch_prepared_global_affine_score<
            OpsTemplate>(prepared);
    result.checksum = checksum_add(result.checksum, result.last_score);
  }
  const auto run_end = Clock::now();

  result.prepare_seconds = std::chrono::duration<double>(prepare_end - prepare_start).count();
  result.run_seconds = std::chrono::duration<double>(run_end - run_start).count();
  result.calls = options.iterations;
  result.scored_targets = options.iterations;
  return result;
}

template <template <typename, typename> class OpsTemplate, typename Cell>
RunResult run_prepared_batch_cell(
    const stride_align::PreparedFarrarBatchAlignment& prepared_alignment,
    const Options& options) {
  RunResult result;
  const auto prepare_start = Clock::now();
  auto prepared = stride_align::farrar_fixed_kernel::detail::prepare_affine_score_batch_state<
      OpsTemplate,
      Cell,
      true>(
      prepared_alignment,
      options.match_score,
      options.mismatch_score,
      options.gap_open_score,
      options.gap_extend_score);
  const auto prepare_end = Clock::now();

  for (std::size_t index = 0; index < options.warmups; ++index) {
    const auto scores =
        stride_align::farrar_fixed_kernel::detail::affine_score_batch_state<
            OpsTemplate,
            Cell,
            false>(prepared);
    result.last_score = scores.empty() ? 0 : scores.back();
  }

  const auto run_start = Clock::now();
  for (std::size_t index = 0; index < options.iterations; ++index) {
    const auto scores =
        stride_align::farrar_fixed_kernel::detail::affine_score_batch_state<
            OpsTemplate,
            Cell,
            false>(prepared);
    for (const auto score : scores) {
      result.last_score = score;
      result.checksum = checksum_add(result.checksum, score);
    }
  }
  const auto run_end = Clock::now();

  result.prepare_seconds = std::chrono::duration<double>(prepare_end - prepare_start).count();
  result.run_seconds = std::chrono::duration<double>(run_end - run_start).count();
  result.calls = options.iterations;
  result.scored_targets = options.iterations * prepared_alignment.target_tokens.size();
  return result;
}

template <template <typename, typename> class OpsTemplate>
RunResult run_prepared_batch(
    const stride_align::PreparedFarrarBatchAlignment& prepared_alignment,
    const Options& options) {
  switch (prepared_alignment.score_bits) {
    case stride_align::KernelBits::bits8:
      return run_prepared_batch_cell<OpsTemplate, std::int8_t>(prepared_alignment, options);
    case stride_align::KernelBits::bits16:
      return run_prepared_batch_cell<OpsTemplate, std::int16_t>(prepared_alignment, options);
    case stride_align::KernelBits::bits32:
      return run_prepared_batch_cell<OpsTemplate, std::int32_t>(prepared_alignment, options);
    case stride_align::KernelBits::bits64:
      return run_prepared_batch_cell<OpsTemplate, std::int64_t>(prepared_alignment, options);
  }
  throw std::runtime_error("unsupported score width");
}

}  // namespace stride_align::microbench
