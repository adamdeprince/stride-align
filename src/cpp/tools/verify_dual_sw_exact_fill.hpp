#pragma once

// Shared dual-target deferred SW exact-fill correctness harness.
// Compares score_batch_state (dual when eligible) vs sequential score_state.
// Used by x86 (AVX-512) and arm64 (NEON) microbench --verify-dual.

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "backends/farrar_fixed_kernel.hpp"
#include "farrar_preprocess.hpp"
#include "stride_align/alignment.hpp"

namespace stride_align::microbench::verify_dual_detail {

inline std::vector<std::uint8_t> make_english_tokens(
    std::size_t length,
    std::uint32_t seed) {
  static constexpr char kCorpus[] =
      "The quick brown fox watches the city wake under a low grey sky. "
      "People cross the station concourse with coffee, folded papers, and quiet plans. "
      "A street musician repeats a careful phrase while buses hiss at the curb. "
      "In the office, someone rewrites a paragraph until the tone is direct and useful. "
      "Human text is uneven: spaces cluster, punctuation interrupts, and words return later. ";
  static constexpr std::size_t kCorpusLen = sizeof(kCorpus) - 1U;
  std::vector<std::uint8_t> out;
  out.reserve(length);
  std::uint32_t rng = seed * 1664525U + 1013904223U;
  for (std::size_t i = 0; i < length; ++i) {
    rng = rng * 1664525U + 1013904223U;
    const std::size_t idx = (seed * 29U + i + (rng >> 16)) % kCorpusLen;
    out.push_back(static_cast<std::uint8_t>(kCorpus[idx]));
  }
  for (std::size_t i = 37U; i < out.size(); i += 41U) {
    rng = rng * 1664525U + 1013904223U;
    out[i] = static_cast<std::uint8_t>('a' + (rng % 26U));
  }
  return out;
}

inline PreparedFarrarBatchAlignment make_batch(
    std::size_t query_len,
    const std::vector<std::size_t>& target_lens,
    std::uint32_t seed_base) {
  PreparedFarrarBatchAlignment prepared;
  prepared.score_bits = KernelBits::bits16;
  prepared.query_tokens = make_english_tokens(query_len, seed_base);
  prepared.target_tokens.reserve(target_lens.size());
  for (std::size_t i = 0; i < target_lens.size(); ++i) {
    prepared.target_tokens.push_back(make_english_tokens(
        target_lens[i],
        seed_base + 17U + static_cast<std::uint32_t>(i) * 101U));
  }
  prepared.symbol_count = 64;
  prepared.score_bound = static_cast<std::uint64_t>(query_len) * 2U;
  return prepared;
}

struct CaseResult {
  std::string name;
  bool ok = true;
  std::string detail;
};

template <template <typename, typename> class OpsTemplate>
CaseResult run_case(
    const std::string& name,
    std::size_t query_len,
    const std::vector<std::size_t>& target_lens,
    farrar_fixed_kernel::detail::ScoreKernelStrategy strategy,
    std::uint32_t seed,
    Score match = 2,
    Score mismatch = -1,
    Score gap = -1) {
  using namespace farrar_fixed_kernel::detail;
  using Cell = std::int16_t;
  CaseResult result;
  result.name = name;

  auto prepared = make_batch(query_len, target_lens, seed);
  for (std::size_t i = 0; i < target_lens.size(); ++i) {
    if (target_lens[i] == 0) {
      prepared.target_tokens[i].clear();
    }
  }

  auto batch = prepare_score_batch_state<OpsTemplate, Cell, true>(
      prepared,
      match,
      mismatch,
      gap,
      ScoreProfileLayout::automatic,
      64U,
      strategy,
      true);

  const auto batch_scores = score_batch_state<OpsTemplate, Cell, true>(batch);

  batch = prepare_score_batch_state<OpsTemplate, Cell, true>(
      prepared,
      match,
      mismatch,
      gap,
      ScoreProfileLayout::automatic,
      64U,
      strategy,
      true);

  std::vector<Score> seq_scores;
  seq_scores.reserve(batch.target_sizes.size());
  for (std::size_t index = 0; index < batch.target_sizes.size(); ++index) {
    batch.state.target_size = batch.target_sizes[index];
    batch.state.fast_score = batch.fast_scores[index];
    if (index < batch.target_profile_offsets.size()) {
      batch.state.target_profile_offsets = batch.target_profile_offsets[index];
    } else {
      batch.state.target_profile_offsets.clear();
    }
    seq_scores.push_back(score_state<OpsTemplate, Cell>(batch.state));
  }

  if (batch_scores.size() != seq_scores.size()) {
    result.ok = false;
    result.detail = "size mismatch batch=" + std::to_string(batch_scores.size()) +
        " seq=" + std::to_string(seq_scores.size());
    return result;
  }
  for (std::size_t i = 0; i < batch_scores.size(); ++i) {
    if (batch_scores[i] != seq_scores[i]) {
      result.ok = false;
      result.detail = "target[" + std::to_string(i) + "] batch=" +
          std::to_string(batch_scores[i]) + " seq=" + std::to_string(seq_scores[i]) +
          " len=" + std::to_string(target_lens[i]);
      return result;
    }
  }
  result.detail = "n=" + std::to_string(batch_scores.size()) + " ok";
  return result;
}

template <template <typename, typename> class OpsTemplate>
int run_verify_dual_cases(const char* label) {
  using farrar_fixed_kernel::detail::ScoreKernelStrategy;
  std::vector<CaseResult> cases;

  cases.push_back(run_case<OpsTemplate>(
      "equal-1024x2-auto", 1024, {1024, 1024}, ScoreKernelStrategy::automatic, 1));
  cases.push_back(run_case<OpsTemplate>(
      "equal-1024x8-auto",
      1024,
      {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
      ScoreKernelStrategy::automatic,
      2));
  cases.push_back(run_case<OpsTemplate>(
      "equal-1024x2-deferred", 1024, {1024, 1024}, ScoreKernelStrategy::deferred, 3));
  cases.push_back(run_case<OpsTemplate>(
      "equal-1024x3-odd-leftover",
      1024,
      {1024, 1024, 1024},
      ScoreKernelStrategy::automatic,
      4));
  cases.push_back(run_case<OpsTemplate>(
      "equal-1024x1-single-leftover", 1024, {1024}, ScoreKernelStrategy::automatic, 5));
  cases.push_back(run_case<OpsTemplate>(
      "equal-1024x7-odd",
      1024,
      {1024, 1024, 1024, 1024, 1024, 1024, 1024},
      ScoreKernelStrategy::deferred,
      6));

  cases.push_back(run_case<OpsTemplate>(
      "unequal-800-1200", 1024, {800, 1200}, ScoreKernelStrategy::automatic, 7));
  cases.push_back(run_case<OpsTemplate>(
      "unequal-mix-4", 1024, {500, 1024, 800, 1200}, ScoreKernelStrategy::deferred, 8));
  cases.push_back(run_case<OpsTemplate>(
      "unequal-pair-equal-mix",
      1024,
      {900, 900, 700, 1100, 1024},
      ScoreKernelStrategy::automatic,
      9));

  cases.push_back(run_case<OpsTemplate>(
      "empty-among-batch", 1024, {1024, 0, 1024, 512}, ScoreKernelStrategy::automatic, 10));
  cases.push_back(run_case<OpsTemplate>(
      "all-empty", 1024, {0, 0}, ScoreKernelStrategy::automatic, 11));

  cases.push_back(run_case<OpsTemplate>(
      "strategy-materialized-1024x4",
      1024,
      {1024, 1024, 1024, 1024},
      ScoreKernelStrategy::materialized,
      12));
  cases.push_back(run_case<OpsTemplate>(
      "strategy-bounded-1024x4",
      1024,
      {1024, 1024, 1024, 1024},
      ScoreKernelStrategy::bounded,
      13));
  cases.push_back(run_case<OpsTemplate>(
      "strategy-compact-1024x2", 1024, {1024, 1024}, ScoreKernelStrategy::compact, 14));
  cases.push_back(run_case<OpsTemplate>(
      "strategy-deferred-u4-1024x2",
      1024,
      {1024, 1024},
      ScoreKernelStrategy::deferred_unroll4,
      15));

  cases.push_back(run_case<OpsTemplate>(
      "query-512-not-exact-fill", 512, {512, 512, 400}, ScoreKernelStrategy::automatic, 16));
  cases.push_back(run_case<OpsTemplate>(
      "query-100-short", 100, {100, 80, 120, 100}, ScoreKernelStrategy::automatic, 17));

  for (std::uint32_t seed = 100; seed < 120; ++seed) {
    cases.push_back(run_case<OpsTemplate>(
        "seed-sweep-" + std::to_string(seed),
        1024,
        {1024, 1024, 1024, 1024},
        ScoreKernelStrategy::automatic,
        seed));
  }

  cases.push_back(run_case<OpsTemplate>(
      "scores-m8-mm9-g1", 1024, {1024, 1024}, ScoreKernelStrategy::automatic, 200, 8, -9, -1));
  cases.push_back(run_case<OpsTemplate>(
      "scores-m2-mm3-g2",
      1024,
      {1024, 900, 1024},
      ScoreKernelStrategy::deferred,
      201,
      2,
      -3,
      -2));

  int failed = 0;
  int passed = 0;
  for (const auto& c : cases) {
    if (c.ok) {
      ++passed;
      std::cout << "PASS  " << c.name << "  (" << c.detail << ")\n";
    } else {
      ++failed;
      std::cerr << "FAIL  " << c.name << "  " << c.detail << "\n";
    }
  }
  std::cout << "verify-dual (" << label << "): " << passed << " passed, " << failed
            << " failed, " << (passed + failed) << " total\n";
  return failed == 0 ? 0 : 1;
}

}  // namespace stride_align::microbench::verify_dual_detail
