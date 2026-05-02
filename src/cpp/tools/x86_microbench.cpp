#include <algorithm>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
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
      << "Profiles native C++ prepared affine Needleman-Wunsch score kernels.\n\n"
      << "Options:\n"
      << "  --backend avx2|avx512bwvl        SIMD backend to run (default: avx2)\n"
      << "  --variant nw-affine-score|sw-farrar-score|sw-cigar|sw-path-info|sw-cigar-path-info|sw-scorefirst-cigar|sw-scorefirst-path-info|sw-checkpointed-cigar\n"
      << "                                      Kernel variant to run (default: nw-affine-score)\n"
      << "  --pass english|chinese           Text-like token distribution (default: english)\n"
      << "  --shape 1:1|1:many               Prepared single or prepared batch path (default: 1:1)\n"
      << "  --length N                       Set query and target length (default: 1024)\n"
      << "  --query-length N                 Query length override\n"
      << "  --target-length N                Target length override\n"
      << "  --many-count N                   Targets per 1:many batch call (default: 8)\n"
      << "  --iterations N                   Timed iterations (default: 1000)\n"
      << "  --warmups N                      Warmup iterations (default: 10)\n"
      << "  --width 0|8|16|32|64             Forced score width, 0 means auto (default: 16)\n"
      << "  --match N                        Match score (default: 2)\n"
      << "  --mismatch N                     Mismatch score (default: -1)\n"
      << "  --gap-open N                     Affine gap-open score (default: -2)\n"
      << "  --gap-extend N                   Affine gap-extend score (default: -1)\n"
      << "  --seed N                         Deterministic corpus offset/mutation seed (default: 1)\n"
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
  if (options.backend != "avx2" && options.backend != "avx512bwvl") {
    usage_error("--backend must be avx2 or avx512bwvl");
  }
  if (options.variant != "nw-affine-score" && options.variant != "sw-farrar-score" &&
      options.variant != "sw-cigar" && options.variant != "sw-path-info" &&
      options.variant != "sw-cigar-path-info" &&
      options.variant != "sw-scorefirst-cigar" &&
      options.variant != "sw-scorefirst-path-info" &&
      options.variant != "sw-checkpointed-cigar") {
    usage_error("--variant must be nw-affine-score, sw-farrar-score, sw-cigar, or sw-path-info");
  }
  if (options.pass_name != "english" && options.pass_name != "english-short" &&
      options.pass_name != "chinese") {
    usage_error("--pass must be english, english-short, or chinese");
  }
  if (options.shape != "1:1" && options.shape != "1:many") {
    usage_error("--shape must be 1:1 or 1:many");
  }
  if (options.many_count == 0) {
    usage_error("--many-count must be at least 1");
  }
  if (options.shape == "1:many" &&
      (options.variant == "sw-cigar" || options.variant == "sw-path-info" ||
       options.variant == "sw-cigar-path-info" ||
       options.variant == "sw-scorefirst-cigar" ||
       options.variant == "sw-scorefirst-path-info" ||
       options.variant == "sw-checkpointed-cigar")) {
    usage_error("--variant sw-cigar and sw-path-info are 1:1-only native microbench rows");
  }
  if (options.iterations == 0) {
    usage_error("--iterations must be at least 1");
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
  if (options.backend == "avx2") {
    if (!supports_avx2()) {
      usage_error("AVX2 is not available on this machine");
    }
    return run_avx2_backend(prepared, options);
  }

  if (!supports_avx512bwvl()) {
    usage_error("AVX512BWVL requires AVX-512F, AVX-512BW, and AVX-512VL");
  }
  return run_avx512bwvl_backend(prepared, options);
}

void print_result(
    const Options& options,
    const Workload& workload,
    const PreparedWorkload& prepared,
    const RunResult& result) {
  const std::size_t targets_per_call = options.shape == "1:many" ? workload.targets.size() : 1U;
  const auto cells_per_target =
      static_cast<double>(workload.query.size()) * static_cast<double>(workload.targets.front().size());
  const auto total_cells = cells_per_target * static_cast<double>(result.scored_targets);
  const auto cells_per_second = total_cells / result.run_seconds;
  const auto ns_per_call = result.run_seconds * 1.0e9 / static_cast<double>(result.calls);
  const auto ns_per_target =
      result.run_seconds * 1.0e9 / static_cast<double>(result.scored_targets);

  std::cout << std::setprecision(9)
            << "backend=" << options.backend
            << " variant=" << options.variant
            << " mode=prepared"
            << " shape=" << options.shape
            << " pass=" << options.pass_name
            << " width=" << bits_name(prepared.batch.score_bits)
            << " query_length=" << workload.query.size()
            << " target_length=" << workload.targets.front().size()
            << " targets_per_call=" << targets_per_call
            << " iterations=" << options.iterations
            << " warmups=" << options.warmups
            << " score=" << result.last_score
            << " checksum=" << result.checksum
            << " prepare_s=" << result.prepare_seconds
            << " run_s=" << result.run_seconds
            << " ns_per_call=" << ns_per_call
            << " ns_per_target=" << ns_per_target
            << " cells_per_s=" << cells_per_second
            << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    const auto workload = build_workload(options);
    const auto prepared = prepare_workload(workload, options);
    const auto result = run_backend(prepared, options);
    print_result(options, workload, prepared, result);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "stride_align_x86_microbench: " << error.what() << '\n';
    return 2;
  }
}
