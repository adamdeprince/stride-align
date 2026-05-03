#pragma once

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "backends/farrar_fixed_kernel.hpp"
#include "x86_microbench_common.hpp"

namespace stride_align::microbench {

using Clock = std::chrono::steady_clock;

inline std::uint64_t checksum_add(std::uint64_t checksum, stride_align::Score score) noexcept {
  const auto mixed = static_cast<std::uint64_t>(score) + 0x9E3779B97F4A7C15ULL;
  return (checksum ^ mixed) * 1099511628211ULL;
}

inline stride_align::farrar_fixed_kernel::detail::ScoreProfileLayout profile_layout_option(
    const Options& options) {
  using Layout = stride_align::farrar_fixed_kernel::detail::ScoreProfileLayout;
  if (options.profile_layout == "auto") {
    return Layout::automatic;
  }
  if (options.profile_layout == "token-major") {
    return Layout::token_major;
  }
  if (options.profile_layout == "target-ordered") {
    return Layout::target_ordered;
  }
  if (options.profile_layout == "blocked-target-ordered") {
    return Layout::blocked_target_ordered;
  }
  if (options.profile_layout == "compact-observed") {
    return Layout::compact_observed;
  }
  throw std::runtime_error("unsupported profile layout");
}

inline stride_align::farrar_fixed_kernel::detail::ScoreKernelStrategy sw_farrar_i32_strategy_option(
    const Options& options) {
  using Strategy = stride_align::farrar_fixed_kernel::detail::ScoreKernelStrategy;
  if (options.sw_farrar_i32_strategy == "auto") {
    return Strategy::automatic;
  }
  if (options.sw_farrar_i32_strategy == "materialized") {
    return Strategy::materialized;
  }
  if (options.sw_farrar_i32_strategy == "deferred") {
    return Strategy::deferred;
  }
  if (options.sw_farrar_i32_strategy == "deferred-u4") {
    return Strategy::deferred_unroll4;
  }
  if (options.sw_farrar_i32_strategy == "bounded") {
    return Strategy::bounded;
  }
  throw std::runtime_error("unsupported SW Farrar width32 strategy");
}

template <template <typename, typename> class OpsTemplate>
RunResult run_nw_affine_score_single(
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
RunResult run_nw_affine_score_batch_cell(
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
RunResult run_nw_affine_score_batch(
    const stride_align::PreparedFarrarBatchAlignment& prepared_alignment,
    const Options& options) {
  switch (prepared_alignment.score_bits) {
    case stride_align::KernelBits::bits8:
      return run_nw_affine_score_batch_cell<OpsTemplate, std::int8_t>(
          prepared_alignment,
          options);
    case stride_align::KernelBits::bits16:
      return run_nw_affine_score_batch_cell<OpsTemplate, std::int16_t>(
          prepared_alignment,
          options);
    case stride_align::KernelBits::bits32:
      return run_nw_affine_score_batch_cell<OpsTemplate, std::int32_t>(
          prepared_alignment,
          options);
    case stride_align::KernelBits::bits64:
      return run_nw_affine_score_batch_cell<OpsTemplate, std::int64_t>(
          prepared_alignment,
          options);
  }
  throw std::runtime_error("unsupported score width");
}

template <template <typename, typename> class OpsTemplate>
RunResult run_sw_farrar_score_single(
    const stride_align::PreparedFarrarAlignment& prepared_alignment,
    const Options& options) {
  RunResult result;
  const auto prepare_start = Clock::now();
  auto prepared = stride_align::farrar_fixed_kernel::detail::prepare_score<OpsTemplate>(
      prepared_alignment,
      options.match_score,
      options.mismatch_score,
      options.gap_extend_score,
      profile_layout_option(options),
      options.profile_block_size,
      sw_farrar_i32_strategy_option(options));
  const auto prepare_end = Clock::now();

  for (std::size_t index = 0; index < options.warmups; ++index) {
    result.last_score =
        stride_align::farrar_fixed_kernel::detail::dispatch_prepared_score<OpsTemplate>(
            prepared);
  }

  const auto run_start = Clock::now();
  for (std::size_t index = 0; index < options.iterations; ++index) {
    result.last_score =
        stride_align::farrar_fixed_kernel::detail::dispatch_prepared_score<OpsTemplate>(
            prepared);
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
RunResult run_sw_farrar_score_batch_cell(
    const stride_align::PreparedFarrarBatchAlignment& prepared_alignment,
    const Options& options) {
  RunResult result;
  const auto prepare_start = Clock::now();
  auto prepared = stride_align::farrar_fixed_kernel::detail::prepare_score_batch_state<
      OpsTemplate,
      Cell,
      true>(
      prepared_alignment,
      options.match_score,
      options.mismatch_score,
      options.gap_extend_score,
      profile_layout_option(options),
      options.profile_block_size,
      sw_farrar_i32_strategy_option(options));
  const auto prepare_end = Clock::now();

  for (std::size_t index = 0; index < options.warmups; ++index) {
    const auto scores =
        stride_align::farrar_fixed_kernel::detail::score_batch_state<
            OpsTemplate,
            Cell,
            true>(prepared);
    result.last_score = scores.empty() ? 0 : scores.back();
  }

  const auto run_start = Clock::now();
  for (std::size_t index = 0; index < options.iterations; ++index) {
    const auto scores =
        stride_align::farrar_fixed_kernel::detail::score_batch_state<
            OpsTemplate,
            Cell,
            true>(prepared);
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
RunResult run_sw_farrar_score_batch(
    const stride_align::PreparedFarrarBatchAlignment& prepared_alignment,
    const Options& options) {
  switch (prepared_alignment.score_bits) {
    case stride_align::KernelBits::bits8:
      return run_sw_farrar_score_batch_cell<OpsTemplate, std::int8_t>(
          prepared_alignment,
          options);
    case stride_align::KernelBits::bits16:
      return run_sw_farrar_score_batch_cell<OpsTemplate, std::int16_t>(
          prepared_alignment,
          options);
    case stride_align::KernelBits::bits32:
      return run_sw_farrar_score_batch_cell<OpsTemplate, std::int32_t>(
          prepared_alignment,
          options);
    case stride_align::KernelBits::bits64:
      return run_sw_farrar_score_batch_cell<OpsTemplate, std::int64_t>(
          prepared_alignment,
          options);
  }
  throw std::runtime_error("unsupported score width");
}

template <template <typename, typename> class OpsTemplate>
RunResult run_sw_cigar_single(
    const stride_align::PreparedFarrarAlignment& prepared_alignment,
    const Options& options) {
  RunResult result;
  const auto run_once = [&]() {
    return stride_align::farrar_fixed_kernel::detail::dispatch_linear_sw_masked_cigar<
        OpsTemplate>(
        prepared_alignment,
        options.match_score,
        options.mismatch_score,
        options.gap_extend_score);
  };

  for (std::size_t index = 0; index < options.warmups; ++index) {
    const auto cigar = run_once();
    result.last_score = static_cast<stride_align::Score>(cigar.size());
  }

  const auto run_start = Clock::now();
  for (std::size_t index = 0; index < options.iterations; ++index) {
    const auto cigar = run_once();
    result.last_score = static_cast<stride_align::Score>(cigar.size());
    result.checksum = checksum_add(result.checksum, result.last_score);
  }
  const auto run_end = Clock::now();

  result.run_seconds = std::chrono::duration<double>(run_end - run_start).count();
  result.calls = options.iterations;
  result.scored_targets = options.iterations;
  return result;
}

template <template <typename, typename> class OpsTemplate>
RunResult run_sw_checkpointed_cigar_single(
    const stride_align::PreparedFarrarAlignment& prepared_alignment,
    const Options& options) {
  RunResult result;
  const auto run_once = [&]() {
    return stride_align::farrar_fixed_kernel::detail::dispatch_linear_sw_checkpointed_cigar<
        OpsTemplate>(
        prepared_alignment,
        options.match_score,
        options.mismatch_score,
        options.gap_extend_score);
  };

  for (std::size_t index = 0; index < options.warmups; ++index) {
    const auto cigar = run_once();
    result.last_score = static_cast<stride_align::Score>(cigar.size());
  }

  const auto run_start = Clock::now();
  for (std::size_t index = 0; index < options.iterations; ++index) {
    const auto cigar = run_once();
    result.last_score = static_cast<stride_align::Score>(cigar.size());
    result.checksum = checksum_add(result.checksum, result.last_score);
  }
  const auto run_end = Clock::now();

  result.run_seconds = std::chrono::duration<double>(run_end - run_start).count();
  result.calls = options.iterations;
  result.scored_targets = options.iterations;
  return result;
}

template <template <typename, typename> class OpsTemplate>
RunResult run_sw_path_info_single(
    const stride_align::PreparedFarrarAlignment& prepared_alignment,
    const Options& options) {
  RunResult result;
  const auto run_once = [&]() {
    return stride_align::farrar_fixed_kernel::detail::dispatch_linear_sw_masked_path_info<
        OpsTemplate>(
        prepared_alignment,
        options.match_score,
        options.mismatch_score,
        options.gap_extend_score);
  };

  for (std::size_t index = 0; index < options.warmups; ++index) {
    const auto path = run_once();
    result.last_score = path.score;
  }

  const auto run_start = Clock::now();
  for (std::size_t index = 0; index < options.iterations; ++index) {
    const auto path = run_once();
    result.last_score = path.score;
    result.checksum = checksum_add(
        checksum_add(result.checksum, path.score),
        static_cast<stride_align::Score>(path.operations.size()));
  }
  const auto run_end = Clock::now();

  result.run_seconds = std::chrono::duration<double>(run_end - run_start).count();
  result.calls = options.iterations;
  result.scored_targets = options.iterations;
  return result;
}

template <template <typename, typename> class OpsTemplate>
RunResult run_sw_cigar_path_info_single(
    const stride_align::PreparedFarrarAlignment& prepared_alignment,
    const Options& options) {
  RunResult result;
  const auto run_once = [&]() {
    return stride_align::farrar_fixed_kernel::detail::dispatch_linear_sw_masked_cigar_path_info<
        OpsTemplate>(
        prepared_alignment,
        options.match_score,
        options.mismatch_score,
        options.gap_extend_score);
  };

  for (std::size_t index = 0; index < options.warmups; ++index) {
    const auto path = run_once();
    result.last_score = path.score;
  }

  const auto run_start = Clock::now();
  for (std::size_t index = 0; index < options.iterations; ++index) {
    const auto path = run_once();
    result.last_score = path.score;
    result.checksum = checksum_add(
        checksum_add(result.checksum, path.score),
        static_cast<stride_align::Score>(path.operations.size()));
  }
  const auto run_end = Clock::now();

  result.run_seconds = std::chrono::duration<double>(run_end - run_start).count();
  result.calls = options.iterations;
  result.scored_targets = options.iterations;
  return result;
}

template <template <typename, typename> class OpsTemplate>
RunResult run_sw_scorefirst_cigar_single(
    const stride_align::PreparedFarrarAlignment& prepared_alignment,
    const Options& options) {
  RunResult result;
  const auto run_once = [&]() {
    return stride_align::farrar_fixed_kernel::detail::
        dispatch_linear_sw_score_first_masked_cigar<OpsTemplate>(
            prepared_alignment,
            options.match_score,
            options.mismatch_score,
            options.gap_extend_score);
  };

  for (std::size_t index = 0; index < options.warmups; ++index) {
    const auto cigar = run_once();
    result.last_score = static_cast<stride_align::Score>(cigar.size());
  }

  const auto run_start = Clock::now();
  for (std::size_t index = 0; index < options.iterations; ++index) {
    const auto cigar = run_once();
    result.last_score = static_cast<stride_align::Score>(cigar.size());
    result.checksum = checksum_add(result.checksum, result.last_score);
  }
  const auto run_end = Clock::now();

  result.run_seconds = std::chrono::duration<double>(run_end - run_start).count();
  result.calls = options.iterations;
  result.scored_targets = options.iterations;
  return result;
}

template <template <typename, typename> class OpsTemplate>
RunResult run_sw_scorefirst_path_info_single(
    const stride_align::PreparedFarrarAlignment& prepared_alignment,
    const Options& options) {
  RunResult result;
  const auto run_once = [&]() {
    return stride_align::farrar_fixed_kernel::detail::
        dispatch_linear_sw_score_first_masked_cigar_path_info<OpsTemplate>(
            prepared_alignment,
            options.match_score,
            options.mismatch_score,
            options.gap_extend_score);
  };

  for (std::size_t index = 0; index < options.warmups; ++index) {
    const auto path = run_once();
    result.last_score = path.score;
  }

  const auto run_start = Clock::now();
  for (std::size_t index = 0; index < options.iterations; ++index) {
    const auto path = run_once();
    result.last_score = path.score;
    result.checksum = checksum_add(
        checksum_add(result.checksum, path.score),
        static_cast<stride_align::Score>(path.operations.size()));
  }
  const auto run_end = Clock::now();

  result.run_seconds = std::chrono::duration<double>(run_end - run_start).count();
  result.calls = options.iterations;
  result.scored_targets = options.iterations;
  return result;
}

template <template <typename, typename> class OpsTemplate>
RunResult run_backend_variant(const PreparedWorkload& prepared, const Options& options) {
  if (options.variant == "nw-affine-score") {
    if (options.shape == "1:many") {
      return run_nw_affine_score_batch<OpsTemplate>(prepared.batch, options);
    }
    return run_nw_affine_score_single<OpsTemplate>(prepared.single, options);
  }
  if (options.variant == "sw-farrar-score") {
    if (options.shape == "1:many") {
      return run_sw_farrar_score_batch<OpsTemplate>(prepared.batch, options);
    }
    return run_sw_farrar_score_single<OpsTemplate>(prepared.single, options);
  }
  if (options.variant == "sw-cigar") {
    return run_sw_cigar_single<OpsTemplate>(prepared.single, options);
  }
  if (options.variant == "sw-checkpointed-cigar") {
    return run_sw_checkpointed_cigar_single<OpsTemplate>(prepared.single, options);
  }
  if (options.variant == "sw-path-info") {
    return run_sw_path_info_single<OpsTemplate>(prepared.single, options);
  }
  if (options.variant == "sw-cigar-path-info") {
    return run_sw_cigar_path_info_single<OpsTemplate>(prepared.single, options);
  }
  if (options.variant == "sw-scorefirst-cigar") {
    return run_sw_scorefirst_cigar_single<OpsTemplate>(prepared.single, options);
  }
  if (options.variant == "sw-scorefirst-path-info") {
    return run_sw_scorefirst_path_info_single<OpsTemplate>(prepared.single, options);
  }
  throw std::runtime_error("unsupported variant");
}

}  // namespace stride_align::microbench
