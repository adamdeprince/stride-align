#include "backends/x86_avx512bwvl.hpp"
#include "x86_microbench_kernels.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace stride_align::microbench {

#if defined(__GNUC__) || defined(__clang__)
#define STRIDE_ALIGN_AVX512BWVL_TARGET \
  __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
#else
#define STRIDE_ALIGN_AVX512BWVL_TARGET
#endif

bool supports_avx512bwvl() noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") != 0 &&
      __builtin_cpu_supports("avx512bw") != 0 &&
      __builtin_cpu_supports("avx512vl") != 0 &&
      __builtin_cpu_supports("avx512dq") != 0;
#else
  return true;
#endif
}

STRIDE_ALIGN_AVX512BWVL_TARGET RunResult run_avx512bwvl_backend(
    const PreparedWorkload& prepared,
    const Options& options) {
  return run_backend_variant<stride_align::backend_avx512bwvl::SimdOps>(prepared, options);
}

namespace {

// OpsTemplate for score_batch_state / score_state (class template <Token, Cell>).
#define STRIDE_ALIGN_VERIFY_OPS stride_align::backend_avx512bwvl::SimdOps
using Cell = std::int16_t;
using namespace stride_align::farrar_fixed_kernel::detail;

std::vector<std::uint8_t> make_english_tokens(std::size_t length, std::uint32_t seed) {
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
  // Mutate a few positions so targets differ.
  for (std::size_t i = 37U; i < out.size(); i += 41U) {
    rng = rng * 1664525U + 1013904223U;
    out[i] = static_cast<std::uint8_t>('a' + (rng % 26U));
  }
  return out;
}

PreparedFarrarBatchAlignment make_batch(
    std::size_t query_len,
    const std::vector<std::size_t>& target_lens,
    std::uint32_t seed_base) {
  PreparedFarrarBatchAlignment prepared;
  prepared.score_bits = stride_align::KernelBits::bits16;
  prepared.query_tokens = make_english_tokens(query_len, seed_base);
  prepared.target_tokens.reserve(target_lens.size());
  for (std::size_t i = 0; i < target_lens.size(); ++i) {
    prepared.target_tokens.push_back(
        make_english_tokens(target_lens[i], seed_base + 17U + static_cast<std::uint32_t>(i) * 101U));
  }
  prepared.symbol_count = 64;  // approximate; not used by prepare_score_batch_state path tightly
  prepared.score_bound = static_cast<std::uint64_t>(query_len) * 2U;
  return prepared;
}

// Sequential reference: score each target via score_state (single-target exact-fill).
// Does NOT go through score_batch_state, so dual is not used.
std::vector<stride_align::Score> sequential_scores(
    PreparedScoreBatchState<Cell>& batch) {
  std::vector<stride_align::Score> scores;
  scores.reserve(batch.target_sizes.size());
  for (std::size_t index = 0; index < batch.target_sizes.size(); ++index) {
    batch.state.target_size = batch.target_sizes[index];
    batch.state.fast_score = batch.fast_scores[index];
    if (index < batch.target_profile_offsets.size()) {
      batch.state.target_profile_offsets = batch.target_profile_offsets[index];
    } else {
      batch.state.target_profile_offsets.clear();
    }
    scores.push_back(score_state<STRIDE_ALIGN_VERIFY_OPS, Cell>(batch.state));
  }
  return scores;
}

struct CaseResult {
  std::string name;
  bool ok = true;
  std::string detail;
};

CaseResult run_case(
    const std::string& name,
    std::size_t query_len,
    const std::vector<std::size_t>& target_lens,
    ScoreKernelStrategy strategy,
    std::uint32_t seed,
    stride_align::Score match = 2,
    stride_align::Score mismatch = -1,
    stride_align::Score gap = -1) {
  CaseResult result;
  result.name = name;

  auto prepared = make_batch(query_len, target_lens, seed);
  // Inject an empty target when length is 0.
  for (std::size_t i = 0; i < target_lens.size(); ++i) {
    if (target_lens[i] == 0) {
      prepared.target_tokens[i].clear();
    }
  }

  auto batch = prepare_score_batch_state<STRIDE_ALIGN_VERIFY_OPS, Cell, true>(
      prepared,
      match,
      mismatch,
      gap,
      ScoreProfileLayout::automatic,
      64U,
      strategy,
      true);

  // Dual / batch path (may use dual for eligible deferred i16×1024).
  const auto batch_scores = score_batch_state<STRIDE_ALIGN_VERIFY_OPS, Cell, true>(batch);

  // Re-prepare for clean sequential (batch path may have mutated buffers).
  batch = prepare_score_batch_state<STRIDE_ALIGN_VERIFY_OPS, Cell, true>(
      prepared,
      match,
      mismatch,
      gap,
      ScoreProfileLayout::automatic,
      64U,
      strategy,
      true);
  const auto seq_scores = sequential_scores(batch);

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

}  // namespace

STRIDE_ALIGN_AVX512BWVL_TARGET int verify_dual_sw_exact_fill_avx512bwvl() {
  if (!supports_avx512bwvl()) {
    std::cerr << "verify-dual: AVX-512 F/BW/VL/DQ not available\n";
    return 2;
  }

  std::vector<CaseResult> cases;

  // --- Equal-length dual hot path (dual_equal) ---
  cases.push_back(run_case(
      "equal-1024x2-auto",
      1024,
      {1024, 1024},
      ScoreKernelStrategy::automatic,
      1));
  cases.push_back(run_case(
      "equal-1024x8-auto",
      1024,
      {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
      ScoreKernelStrategy::automatic,
      2));
  cases.push_back(run_case(
      "equal-1024x2-deferred",
      1024,
      {1024, 1024},
      ScoreKernelStrategy::deferred,
      3));
  cases.push_back(run_case(
      "equal-1024x3-odd-leftover",
      1024,
      {1024, 1024, 1024},
      ScoreKernelStrategy::automatic,
      4));
  cases.push_back(run_case(
      "equal-1024x1-single-leftover",
      1024,
      {1024},
      ScoreKernelStrategy::automatic,
      5));
  cases.push_back(run_case(
      "equal-1024x7-odd",
      1024,
      {1024, 1024, 1024, 1024, 1024, 1024, 1024},
      ScoreKernelStrategy::deferred,
      6));

  // --- Unequal lengths (dual min + finish longer) ---
  cases.push_back(run_case(
      "unequal-800-1200",
      1024,
      {800, 1200},
      ScoreKernelStrategy::automatic,
      7));
  cases.push_back(run_case(
      "unequal-mix-4",
      1024,
      {500, 1024, 800, 1200},
      ScoreKernelStrategy::deferred,
      8));
  cases.push_back(run_case(
      "unequal-pair-equal-mix",
      1024,
      {900, 900, 700, 1100, 1024},
      ScoreKernelStrategy::automatic,
      9));

  // --- Empty / degenerate targets ---
  cases.push_back(run_case(
      "empty-among-batch",
      1024,
      {1024, 0, 1024, 512},
      ScoreKernelStrategy::automatic,
      10));
  cases.push_back(run_case(
      "all-empty",
      1024,
      {0, 0},
      ScoreKernelStrategy::automatic,
      11));

  // --- Strategy fallthrough (dual must NOT claim non-deferred strategies) ---
  // Scores still must match sequential single-target for the same strategy.
  cases.push_back(run_case(
      "strategy-materialized-1024x4",
      1024,
      {1024, 1024, 1024, 1024},
      ScoreKernelStrategy::materialized,
      12));
  cases.push_back(run_case(
      "strategy-bounded-1024x4",
      1024,
      {1024, 1024, 1024, 1024},
      ScoreKernelStrategy::bounded,
      13));
  cases.push_back(run_case(
      "strategy-compact-1024x2",
      1024,
      {1024, 1024},
      ScoreKernelStrategy::compact,
      14));
  cases.push_back(run_case(
      "strategy-deferred-u4-1024x2",
      1024,
      {1024, 1024},
      ScoreKernelStrategy::deferred_unroll4,
      15));

  // --- Non-exact-fill query length (dual should fall through; still correct) ---
  cases.push_back(run_case(
      "query-512-not-exact-fill",
      512,
      {512, 512, 400},
      ScoreKernelStrategy::automatic,
      16));
  cases.push_back(run_case(
      "query-100-short",
      100,
      {100, 80, 120, 100},
      ScoreKernelStrategy::automatic,
      17));

  // --- Seed sweep on equal dual hot path ---
  for (std::uint32_t seed = 100; seed < 120; ++seed) {
    cases.push_back(run_case(
        "seed-sweep-" + std::to_string(seed),
        1024,
        {1024, 1024, 1024, 1024},
        ScoreKernelStrategy::automatic,
        seed));
  }

  // --- Different gap/match on dual path ---
  cases.push_back(run_case(
      "scores-m8-mm9-g1",
      1024,
      {1024, 1024},
      ScoreKernelStrategy::automatic,
      200,
      8,
      -9,
      -1));
  cases.push_back(run_case(
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
  std::cout << "verify-dual: " << passed << " passed, " << failed << " failed, "
            << (passed + failed) << " total\n";
  return failed == 0 ? 0 : 1;
}

#undef STRIDE_ALIGN_VERIFY_OPS
#undef STRIDE_ALIGN_AVX512BWVL_TARGET

}  // namespace stride_align::microbench
