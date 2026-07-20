#include <algorithm>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "x86_microbench_common.hpp"

namespace {

using namespace stride_align::microbench;

[[noreturn]] void usage_error(std::string_view message) {
  throw std::invalid_argument(std::string(message));
}

std::size_t parse_size(std::string_view value, std::string_view option) {
  std::size_t parsed = 0;
  std::size_t index = 0;
  try {
    parsed = std::stoull(std::string(value), &index);
  } catch (const std::exception&) {
    usage_error(std::string(option) + " expects an unsigned integer");
  }
  if (index != value.size()) {
    usage_error(std::string(option) + " expects an unsigned integer");
  }
  return parsed;
}

int parse_int(std::string_view value, std::string_view option) {
  int parsed = 0;
  std::size_t index = 0;
  try {
    parsed = std::stoi(std::string(value), &index);
  } catch (const std::exception&) {
    usage_error(std::string(option) + " expects an integer");
  }
  if (index != value.size()) {
    usage_error(std::string(option) + " expects an integer");
  }
  return parsed;
}

void print_help(std::ostream& output) {
  output
      << "Usage: stride_align_x86_microbench [options]\n\n"
      << "Profiles native C++ SIMD score and traceback kernels.\n\n"
      << "Options:\n"
      << "  --backend avx2|avx512bwvl|parasail"
#if defined(__aarch64__) || defined(_M_ARM64)
      << "|neon"
#endif
      << "\n"
      << "                                      SIMD/backend implementation to run (default: "
#if defined(__aarch64__) || defined(_M_ARM64)
      << "neon"
#else
      << "avx2"
#endif
      << ")\n"
      << "  --variant nw-affine-score|sw-farrar-score|sw-affine-farrar-score|sw-affine-cigar|sw-affine-path-info|sw-cigar|sw-path-info|sw-cigar-path-info|sw-scorefirst-cigar|sw-scorefirst-path-info|sw-checkpointed-cigar\n"
      << "                                      Kernel variant to run (default: nw-affine-score)\n"
      << "  --pass english|chinese           Text-like token distribution (default: english)\n"
      << "  --shape 1:1|1:many               Prepared single or prepared batch path (default: 1:1)\n"
      << "  --profile-layout auto|token-major|target-ordered|blocked-target-ordered|compact-observed\n"
      << "                                      Farrar score profile layout A/B switch (default: auto)\n"
      << "  --profile-block-size N           Block size for blocked-target-ordered layout (default: 64)\n"
      << "  --sw-farrar-i32-strategy auto|materialized|deferred|deferred-u4|bounded|bounded-u4|compact\n"
      << "                                      AVX2 exact-fill SW Farrar correction strategy A/B switch (default: auto)\n"
      << "  --length N                       Set query and target length (default: 1024)\n"
      << "  --query-length N                 Query length override\n"
      << "  --target-length N                Target length override\n"
      << "  --many-count N                   Targets per 1:many batch call (default: 8)\n"
      << "  --iterations N                   Timed iterations (default: 1000)\n"
      << "  --warmups N                      Warmup iterations (default: 10)\n"
      << "  --samples N                      Repeat timed run N times and report median/best (default: 1)\n"
      << "  --width 0|8|16|32|64             Forced score width, 0 means auto (default: 16)\n"
      << "  --match N                        Match score (default: 2)\n"
      << "  --mismatch N                     Mismatch score (default: -1)\n"
      << "  --gap-open N                     Affine gap-open score (default: -2)\n"
      << "  --gap-extend N                   Affine gap-extend score (default: -1)\n"
      << "  --seed N                         Deterministic corpus offset/mutation seed (default: 1)\n"
      << "  --verify-dual                    Run dual-target SW exact-fill correctness fuzz\n"
      << "                                      (avx512bwvl on x86; neon on arm64 microbench)\n"
      << "  --help                           Show this help text\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  bool query_length_set = false;
  bool target_length_set = false;

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      print_help(std::cout);
      std::exit(0);
    }
    if (argument == "--verify-dual") {
      // Handled in main before normal bench path; accept here so parse succeeds.
      continue;
    }

    const auto require_value = [&](std::string_view option) -> std::string_view {
      if (index + 1 >= argc) {
        usage_error(std::string(option) + " requires a value");
      }
      return std::string_view(argv[++index]);
    };

    if (argument == "--backend") {
      options.backend = require_value(argument);
    } else if (argument == "--variant") {
      options.variant = require_value(argument);
    } else if (argument == "--pass") {
      options.pass_name = require_value(argument);
    } else if (argument == "--shape") {
      options.shape = require_value(argument);
    } else if (argument == "--profile-layout") {
      options.profile_layout = require_value(argument);
    } else if (argument == "--profile-block-size") {
      options.profile_block_size = parse_size(require_value(argument), argument);
    } else if (argument == "--sw-farrar-i32-strategy") {
      options.sw_farrar_i32_strategy = require_value(argument);
    } else if (argument == "--length") {
      const auto length = parse_size(require_value(argument), argument);
      options.query_length = length;
      options.target_length = length;
      query_length_set = true;
      target_length_set = true;
    } else if (argument == "--query-length") {
      options.query_length = parse_size(require_value(argument), argument);
      query_length_set = true;
    } else if (argument == "--target-length") {
      options.target_length = parse_size(require_value(argument), argument);
      target_length_set = true;
    } else if (argument == "--many-count") {
      options.many_count = parse_size(require_value(argument), argument);
    } else if (argument == "--iterations") {
      options.iterations = parse_size(require_value(argument), argument);
    } else if (argument == "--warmups") {
      options.warmups = parse_size(require_value(argument), argument);
    } else if (argument == "--samples") {
      options.samples = parse_size(require_value(argument), argument);
    } else if (argument == "--width") {
      options.width = static_cast<unsigned int>(parse_size(require_value(argument), argument));
    } else if (argument == "--match") {
      options.match_score = parse_int(require_value(argument), argument);
    } else if (argument == "--mismatch") {
      options.mismatch_score = parse_int(require_value(argument), argument);
    } else if (argument == "--gap-open") {
      options.gap_open_score = parse_int(require_value(argument), argument);
    } else if (argument == "--gap-extend") {
      options.gap_extend_score = parse_int(require_value(argument), argument);
    } else if (argument == "--seed") {
      options.seed = parse_int(require_value(argument), argument);
    } else {
      usage_error("unknown option: " + std::string(argument));
    }
  }

  if (options.pass_name == "english-short" && !query_length_set && !target_length_set) {
    options.query_length = 31;
    options.target_length = 31;
  }
  if (options.backend != "avx2" && options.backend != "avx512bwvl" &&
      options.backend != "parasail"
#if defined(__aarch64__) || defined(_M_ARM64)
      && options.backend != "neon"
#endif
  ) {
    usage_error("--backend must be avx2, avx512bwvl, parasail, or neon when built on arm64");
  }
  if (options.variant != "nw-affine-score" && options.variant != "sw-farrar-score" &&
      options.variant != "sw-affine-farrar-score" &&
      options.variant != "sw-affine-cigar" &&
      options.variant != "sw-affine-path-info" &&
      options.variant != "sw-cigar" && options.variant != "sw-path-info" &&
      options.variant != "sw-cigar-path-info" &&
      options.variant != "sw-scorefirst-cigar" &&
      options.variant != "sw-scorefirst-path-info" &&
      options.variant != "sw-checkpointed-cigar") {
    usage_error("--variant must be one of the documented score or traceback variants");
  }
  if (options.pass_name != "english" && options.pass_name != "english-short" &&
      options.pass_name != "chinese") {
    usage_error("--pass must be english, english-short, or chinese");
  }
  if (options.shape != "1:1" && options.shape != "1:many") {
    usage_error("--shape must be 1:1 or 1:many");
  }
  if (options.profile_layout != "auto" && options.profile_layout != "token-major" &&
      options.profile_layout != "target-ordered" &&
      options.profile_layout != "blocked-target-ordered" &&
      options.profile_layout != "compact-observed") {
    usage_error(
        "--profile-layout must be auto, token-major, target-ordered, blocked-target-ordered, or compact-observed");
  }
  if (options.sw_farrar_i32_strategy != "auto" &&
      options.sw_farrar_i32_strategy != "materialized" &&
      options.sw_farrar_i32_strategy != "deferred" &&
      options.sw_farrar_i32_strategy != "deferred-u4" &&
      options.sw_farrar_i32_strategy != "bounded" &&
      options.sw_farrar_i32_strategy != "bounded-u4" &&
      options.sw_farrar_i32_strategy != "compact") {
    usage_error(
        "--sw-farrar-i32-strategy must be auto, materialized, deferred, deferred-u4, bounded, bounded-u4, or compact");
  }
  if (options.many_count == 0) {
    usage_error("--many-count must be at least 1");
  }
  if (options.shape == "1:many" &&
      (options.variant == "sw-cigar" || options.variant == "sw-path-info" ||
       options.variant == "sw-affine-cigar" ||
       options.variant == "sw-affine-path-info" ||
       options.variant == "sw-cigar-path-info" ||
       options.variant == "sw-scorefirst-cigar" ||
       options.variant == "sw-scorefirst-path-info" ||
       options.variant == "sw-checkpointed-cigar")) {
    usage_error("traceback variants are 1:1-only native microbench rows");
  }
  if (options.iterations == 0) {
    usage_error("--iterations must be at least 1");
  }
  if (options.samples == 0) {
    usage_error("--samples must be at least 1");
  }
  if (options.profile_block_size == 0) {
    usage_error("--profile-block-size must be at least 1");
  }
  if (options.width != 0 && options.width != 8 && options.width != 16 &&
      options.width != 32 && options.width != 64) {
    usage_error("--width must be 0, 8, 16, 32, or 64");
  }
  return options;
}

std::vector<std::uint8_t> make_english_corpus() {
  const std::string corpus =
      "The quick brown fox watches the city wake under a low grey sky. "
      "People cross the station concourse with coffee, folded papers, and quiet plans. "
      "A street musician repeats a careful phrase while buses hiss at the curb. "
      "In the office, someone rewrites a paragraph until the tone is direct and useful. "
      "Human text is uneven: spaces cluster, punctuation interrupts, and words return later. ";
  return {corpus.begin(), corpus.end()};
}

std::vector<std::uint8_t> make_chinese_like_corpus() {
  std::vector<std::uint8_t> corpus;
  corpus.reserve(320);
  for (std::size_t index = 0; index < 240; ++index) {
    const auto common = static_cast<std::uint8_t>((index % 19U) + 1U);
    const auto varied = static_cast<std::uint8_t>(((index * 37U + index / 3U * 11U) % 61U) + 1U);
    corpus.push_back((index % 4U == 0U) ? common : varied);
    if (index % 23U == 0U) {
      corpus.push_back(static_cast<std::uint8_t>(62U));
    }
    if (index % 37U == 0U) {
      corpus.push_back(static_cast<std::uint8_t>(63U));
    }
  }
  return corpus;
}

std::vector<std::uint8_t> repeat_window(
    std::span<const std::uint8_t> corpus,
    std::size_t length,
    std::size_t offset) {
  std::vector<std::uint8_t> output;
  output.reserve(length);
  if (corpus.empty()) {
    return output;
  }
  for (std::size_t index = 0; index < length; ++index) {
    output.push_back(corpus[(offset + index) % corpus.size()]);
  }
  return output;
}

std::uint32_t next_random(std::uint32_t& state) noexcept {
  state = state * 1664525U + 1013904223U;
  return state;
}

void mutate_tokens(
    std::vector<std::uint8_t>& tokens,
    std::span<const std::uint8_t> replacements,
    int seed) {
  if (tokens.empty() || replacements.empty()) {
    return;
  }

  std::uint32_t rng = static_cast<std::uint32_t>(seed) * 747796405U + 2891336453U;
  const std::size_t stride = 37U + static_cast<std::size_t>(seed >= 0 ? seed : -seed) % 17U;
  for (std::size_t index = stride - 1U; index < tokens.size(); index += stride) {
    tokens[index] = replacements[next_random(rng) % replacements.size()];
  }
}

Workload build_workload(const Options& options) {
  std::vector<std::uint8_t> corpus;
  std::vector<std::uint8_t> replacements;
  if (options.pass_name == "english" || options.pass_name == "english-short") {
    corpus = make_english_corpus();
    const std::string replacement_text = " etaoinshrdlucmfwypvbgkqjxz,.;:-";
    replacements.assign(replacement_text.begin(), replacement_text.end());
  } else {
    corpus = make_chinese_like_corpus();
    for (std::uint8_t token = 1; token <= 40; ++token) {
      replacements.push_back(token);
    }
  }

  Workload workload;
  const auto offset =
      (static_cast<std::size_t>(options.seed >= 0 ? options.seed : -options.seed) * 29U) %
      corpus.size();
  workload.query = repeat_window(corpus, options.query_length, offset);

  const std::size_t target_count = options.shape == "1:1" ? 1U : options.many_count;
  workload.targets.reserve(target_count);
  for (std::size_t target_index = 0; target_index < target_count; ++target_index) {
    const auto target_offset = (offset + target_index * 53U) % corpus.size();
    auto target = repeat_window(corpus, options.target_length, target_offset);
    mutate_tokens(
        target,
        replacements,
        options.seed + static_cast<int>(options.pass_name.size()) +
            static_cast<int>(target_index * 101U));
    workload.targets.push_back(std::move(target));
  }
  return workload;
}

std::uint64_t count_symbols(const Workload& workload) {
  bool seen[256] = {};
  std::uint64_t count = 0;
  const auto mark = [&](std::uint8_t token) {
    if (!seen[token]) {
      seen[token] = true;
      ++count;
    }
  };
  for (const auto token : workload.query) {
    mark(token);
  }
  for (const auto& target : workload.targets) {
    for (const auto token : target) {
      mark(token);
    }
  }
  return count;
}

stride_align::KernelBits forced_bits(stride_align::KernelBits automatic, unsigned int width) {
  if (width == 0) {
    return automatic;
  }
  const auto forced = static_cast<stride_align::KernelBits>(width);
  if (width < static_cast<unsigned int>(automatic)) {
    usage_error("--width cannot be narrower than the automatic score width");
  }
  return forced;
}

std::uint64_t max_target_size(const Workload& workload) {
  std::size_t maximum = 0;
  for (const auto& target : workload.targets) {
    maximum = std::max(maximum, target.size());
  }
  return maximum;
}

PreparedWorkload prepare_workload(const Workload& workload, const Options& options) {
  const auto score_bound = stride_align::detail::compute_score_bound(
      workload.query.size(),
      max_target_size(workload),
      options.match_score,
      options.mismatch_score,
      options.gap_open_score,
      options.gap_extend_score);
  const auto score_bits =
      forced_bits(stride_align::farrar_detail::select_score_bits(score_bound), options.width);
  const auto symbol_count = count_symbols(workload);

  PreparedWorkload prepared;
  prepared.batch.score_bits = score_bits;
  prepared.batch.symbol_count = symbol_count;
  prepared.batch.score_bound = score_bound;
  prepared.batch.query_tokens = workload.query;
  prepared.batch.target_tokens = workload.targets;

  prepared.single.score_bits = score_bits;
  prepared.single.symbol_count = symbol_count;
  prepared.single.score_bound = score_bound;
  prepared.single.query_tokens = workload.query;
  prepared.single.target_tokens = workload.targets.front();
  return prepared;
}

std::string bits_name(stride_align::KernelBits bits) {
  return std::to_string(static_cast<unsigned int>(bits));
}

RunResult run_backend(const PreparedWorkload& prepared, const Options& options) {
#if defined(__aarch64__) || defined(_M_ARM64)
  if (options.backend == "neon") {
    if (!supports_neon()) {
      usage_error("NEON is not available on this machine");
    }
    return run_neon_backend(prepared, options);
  }
#endif

  if (options.backend == "avx2") {
    if (!supports_avx2()) {
      usage_error("AVX2 is not available on this machine");
    }
    return run_avx2_backend(prepared, options);
  }

  if (options.backend == "parasail") {
    if (!supports_parasail()) {
      usage_error("native parasail microbench support was not compiled in");
    }
    return run_parasail_backend(prepared, options);
  }

  if (!supports_avx512bwvl()) {
    usage_error("AVX512BWVL requires AVX-512F, AVX-512BW, and AVX-512VL");
  }
  return run_avx512bwvl_backend(prepared, options);
}

double ns_per_target(
    const RunResult& result,
    std::size_t targets_per_call) {
  const auto scored_targets = result.scored_targets == 0 ? targets_per_call : result.scored_targets;
  return result.run_seconds * 1.0e9 / static_cast<double>(scored_targets);
}

double cells_per_second(
    const Workload& workload,
    const RunResult& result) {
  const auto cells_per_target =
      static_cast<double>(workload.query.size()) *
      static_cast<double>(workload.targets.front().size());
  const auto total_cells = cells_per_target * static_cast<double>(result.scored_targets);
  return total_cells / result.run_seconds;
}

std::vector<RunResult> run_backend_samples(
    const PreparedWorkload& prepared,
    const Options& options) {
  std::vector<RunResult> results;
  results.reserve(options.samples);
  for (std::size_t sample = 0; sample < options.samples; ++sample) {
    results.push_back(run_backend(prepared, options));
  }
  return results;
}

const RunResult& median_result(
    const std::vector<RunResult>& results,
    std::size_t targets_per_call) {
  if (results.empty()) {
    usage_error("no benchmark samples collected");
  }
  std::vector<std::size_t> order(results.size());
  for (std::size_t index = 0; index < order.size(); ++index) {
    order[index] = index;
  }
  std::sort(
      order.begin(),
      order.end(),
      [&](std::size_t lhs, std::size_t rhs) {
        return ns_per_target(results[lhs], targets_per_call) <
            ns_per_target(results[rhs], targets_per_call);
      });
  return results[order[order.size() / 2U]];
}

double best_ns_per_target(
    const std::vector<RunResult>& results,
    std::size_t targets_per_call) {
  double best = std::numeric_limits<double>::infinity();
  for (const auto& result : results) {
    best = std::min(best, ns_per_target(result, targets_per_call));
  }
  return best;
}

void print_result(
    const Options& options,
    const Workload& workload,
    const PreparedWorkload& prepared,
    const RunResult& result,
    const std::vector<RunResult>& samples) {
  const std::size_t targets_per_call = options.shape == "1:many" ? workload.targets.size() : 1U;
  const auto ns_per_call = result.run_seconds * 1.0e9 / static_cast<double>(result.calls);
  const auto target_ns = ns_per_target(result, targets_per_call);
  const auto target_cells_per_second = cells_per_second(workload, result);
  const auto best_ns = best_ns_per_target(samples, targets_per_call);

  std::cout << std::setprecision(9)
            << "backend=" << options.backend
            << " variant=" << options.variant
            << " mode=prepared"
            << " shape=" << options.shape
            << " profile_layout=" << options.profile_layout
            << " profile_block_size=" << options.profile_block_size
            << " sw_farrar_i32_strategy=" << options.sw_farrar_i32_strategy
            << " pass=" << options.pass_name
            << " width=" << bits_name(prepared.batch.score_bits)
            << " query_length=" << workload.query.size()
            << " target_length=" << workload.targets.front().size()
            << " targets_per_call=" << targets_per_call
            << " iterations=" << options.iterations
            << " warmups=" << options.warmups
            << " samples=" << options.samples
            << " score=" << result.last_score
            << " checksum=" << result.checksum
            << " prepare_s=" << result.prepare_seconds
            << " run_s=" << result.run_seconds
            << " ns_per_call=" << ns_per_call
            << " ns_per_target=" << target_ns
            << " cells_per_s=" << target_cells_per_second;
  if (options.samples > 1U) {
    std::cout << " median_ns_per_target=" << target_ns
              << " best_ns_per_target=" << best_ns
              << " best_cells_per_s="
              << (target_cells_per_second * target_ns / best_ns);
  }
  std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bool verify_dual = false;
    for (int index = 1; index < argc; ++index) {
      if (std::string_view(argv[index]) == "--verify-dual") {
        verify_dual = true;
        break;
      }
    }
    if (verify_dual) {
      // x86: AVX-512 dual harness. arm64 microbench build: NEON dual harness
      // (see arm_neon_microbench_backend.cpp).
      return verify_dual_sw_exact_fill_avx512bwvl();
    }

    const auto options = parse_options(argc, argv);
    const auto workload = build_workload(options);
    const auto prepared = prepare_workload(workload, options);
    const auto results = run_backend_samples(prepared, options);
    const std::size_t targets_per_call = options.shape == "1:many" ? workload.targets.size() : 1U;
    const auto& result = median_result(results, targets_per_call);
    print_result(options, workload, prepared, result, results);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "stride_align_x86_microbench: " << error.what() << '\n';
    return 2;
  }
}
