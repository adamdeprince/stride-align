#include "x86_microbench_common.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(STRIDE_ALIGN_HAVE_PARASAIL_MICROBENCH)
#include <parasail.h>
#endif

namespace stride_align::microbench {

namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t checksum_add(std::uint64_t checksum, stride_align::Score score) noexcept {
  const auto mixed = static_cast<std::uint64_t>(score) + 0x9E3779B97F4A7C15ULL;
  return (checksum ^ mixed) * 1099511628211ULL;
}

#if defined(STRIDE_ALIGN_HAVE_PARASAIL_MICROBENCH)

struct MatrixDeleter {
  void operator()(parasail_matrix_t* matrix) const noexcept {
    parasail_matrix_free(matrix);
  }
};

struct ProfileDeleter {
  void operator()(parasail_profile_t* profile) const noexcept {
    parasail_profile_free(profile);
  }
};

struct ResultDeleter {
  void operator()(parasail_result_t* result) const noexcept {
    parasail_result_free(result);
  }
};

using MatrixPtr = std::unique_ptr<parasail_matrix_t, MatrixDeleter>;
using ProfilePtr = std::unique_ptr<parasail_profile_t, ProfileDeleter>;
using ResultPtr = std::unique_ptr<parasail_result_t, ResultDeleter>;

constexpr std::array<unsigned char, 201> safe_alphabet = [] {
  std::array<unsigned char, 201> alphabet = {};
  std::size_t index = 0;
  for (unsigned int codepoint = 1; codepoint < 256; ++codepoint) {
    if (codepoint == static_cast<unsigned int>('*') ||
        codepoint == static_cast<unsigned int>('\\')) {
      continue;
    }
    if ((static_cast<unsigned int>('A') <= codepoint &&
            codepoint <= static_cast<unsigned int>('Z')) ||
        (static_cast<unsigned int>('a') <= codepoint &&
            codepoint <= static_cast<unsigned int>('z'))) {
      continue;
    }
    alphabet[index++] = static_cast<unsigned char>(codepoint);
  }
  return alphabet;
}();

int checked_length(std::size_t length) {
  if (length > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("parasail C API length exceeds int range");
  }
  return static_cast<int>(length);
}

struct ParasailInputs {
  std::vector<char> query;
  std::vector<std::vector<char>> targets;
  std::string alphabet;
};

ParasailInputs translate_inputs(const PreparedWorkload& prepared, const Options& options) {
  const auto& query_tokens = prepared.batch.query_tokens;
  const auto& target_tokens =
      options.shape == "1:many" ? prepared.batch.target_tokens :
                                  std::vector<std::vector<std::uint8_t>>{prepared.single.target_tokens};

  bool seen[256] = {};
  std::array<char, 256> translation = {};
  std::size_t symbol_count = 0;

  const auto mark = [&](std::uint8_t token) {
    if (seen[token]) {
      return;
    }
    if (symbol_count >= safe_alphabet.size()) {
      throw std::runtime_error("parasail safe alphabet exhausted for native microbench");
    }
    seen[token] = true;
    translation[token] = static_cast<char>(safe_alphabet[symbol_count++]);
  };

  for (const auto token : query_tokens) {
    mark(token);
  }
  for (const auto& target : target_tokens) {
    for (const auto token : target) {
      mark(token);
    }
  }

  ParasailInputs inputs;
  inputs.alphabet.reserve(symbol_count);
  for (std::size_t index = 0; index < symbol_count; ++index) {
    inputs.alphabet.push_back(static_cast<char>(safe_alphabet[index]));
  }

  inputs.query.reserve(query_tokens.size());
  for (const auto token : query_tokens) {
    inputs.query.push_back(translation[token]);
  }

  inputs.targets.reserve(target_tokens.size());
  for (const auto& target : target_tokens) {
    auto& translated = inputs.targets.emplace_back();
    translated.reserve(target.size());
    for (const auto token : target) {
      translated.push_back(translation[token]);
    }
  }
  return inputs;
}

ProfilePtr create_profile(
    const ParasailInputs& inputs,
    const parasail_matrix_t* matrix,
    stride_align::KernelBits bits) {
  const char* query = inputs.query.data();
  const int query_length = checked_length(inputs.query.size());
  switch (bits) {
    case stride_align::KernelBits::bits8:
      return ProfilePtr(parasail_profile_create_avx_256_8(query, query_length, matrix));
    case stride_align::KernelBits::bits16:
      return ProfilePtr(parasail_profile_create_avx_256_16(query, query_length, matrix));
    case stride_align::KernelBits::bits32:
      return ProfilePtr(parasail_profile_create_avx_256_32(query, query_length, matrix));
    case stride_align::KernelBits::bits64:
      return ProfilePtr(parasail_profile_create_avx_256_64(query, query_length, matrix));
  }
  throw std::runtime_error("unsupported parasail score width");
}

ResultPtr run_sw_profile(
    const parasail_profile_t* profile,
    const std::vector<char>& target,
    stride_align::KernelBits bits,
    int gap_penalty) {
  const char* target_data = target.data();
  const int target_length = checked_length(target.size());
  switch (bits) {
    case stride_align::KernelBits::bits8:
      return ResultPtr(
          parasail_sw_striped_profile_avx2_256_8(profile, target_data, target_length, gap_penalty, gap_penalty));
    case stride_align::KernelBits::bits16:
      return ResultPtr(
          parasail_sw_striped_profile_avx2_256_16(profile, target_data, target_length, gap_penalty, gap_penalty));
    case stride_align::KernelBits::bits32:
      return ResultPtr(
          parasail_sw_striped_profile_avx2_256_32(profile, target_data, target_length, gap_penalty, gap_penalty));
    case stride_align::KernelBits::bits64:
      return ResultPtr(
          parasail_sw_striped_profile_avx2_256_64(profile, target_data, target_length, gap_penalty, gap_penalty));
  }
  throw std::runtime_error("unsupported parasail score width");
}

ResultPtr run_nw_profile(
    const parasail_profile_t* profile,
    const std::vector<char>& target,
    stride_align::KernelBits bits,
    int gap_open_penalty,
    int gap_extend_penalty) {
  const char* target_data = target.data();
  const int target_length = checked_length(target.size());
  switch (bits) {
    case stride_align::KernelBits::bits8:
      return ResultPtr(parasail_nw_striped_profile_avx2_256_8(
          profile,
          target_data,
          target_length,
          gap_open_penalty,
          gap_extend_penalty));
    case stride_align::KernelBits::bits16:
      return ResultPtr(parasail_nw_striped_profile_avx2_256_16(
          profile,
          target_data,
          target_length,
          gap_open_penalty,
          gap_extend_penalty));
    case stride_align::KernelBits::bits32:
      return ResultPtr(parasail_nw_striped_profile_avx2_256_32(
          profile,
          target_data,
          target_length,
          gap_open_penalty,
          gap_extend_penalty));
    case stride_align::KernelBits::bits64:
      return ResultPtr(parasail_nw_striped_profile_avx2_256_64(
          profile,
          target_data,
          target_length,
          gap_open_penalty,
          gap_extend_penalty));
  }
  throw std::runtime_error("unsupported parasail score width");
}

stride_align::Score result_score(ResultPtr result) {
  if (!result) {
    throw std::runtime_error("parasail returned a null result");
  }
  return static_cast<stride_align::Score>(parasail_result_get_score(result.get()));
}

#endif

}  // namespace

bool supports_parasail() noexcept {
#if defined(STRIDE_ALIGN_HAVE_PARASAIL_MICROBENCH)
  return true;
#else
  return false;
#endif
}

RunResult run_parasail_backend(const PreparedWorkload& prepared, const Options& options) {
#if defined(STRIDE_ALIGN_HAVE_PARASAIL_MICROBENCH)
  if (options.variant != "sw-farrar-score" && options.variant != "nw-affine-score") {
    throw std::runtime_error("native parasail microbench supports sw-farrar-score and nw-affine-score");
  }
  if (options.gap_open_score >= 0 || options.gap_extend_score >= 0) {
    throw std::runtime_error("parasail native microbench expects non-positive gap scores");
  }

  RunResult result;
  const auto prepare_start = Clock::now();
  auto inputs = translate_inputs(prepared, options);
  MatrixPtr matrix(parasail_matrix_create(
      inputs.alphabet.c_str(),
      static_cast<int>(options.match_score),
      static_cast<int>(options.mismatch_score)));
  if (!matrix) {
    throw std::runtime_error("parasail_matrix_create failed");
  }
  auto profile = create_profile(inputs, matrix.get(), prepared.batch.score_bits);
  if (!profile) {
    throw std::runtime_error("parasail_profile_create failed");
  }
  const auto prepare_end = Clock::now();

  const auto run_once = [&](const std::vector<char>& target) -> stride_align::Score {
    if (options.variant == "sw-farrar-score") {
      return result_score(run_sw_profile(
          profile.get(),
          target,
          prepared.batch.score_bits,
          static_cast<int>(-options.gap_extend_score)));
    }
    return result_score(run_nw_profile(
        profile.get(),
        target,
        prepared.batch.score_bits,
        static_cast<int>(-options.gap_open_score),
        static_cast<int>(-options.gap_extend_score)));
  };

  for (std::size_t index = 0; index < options.warmups; ++index) {
    for (const auto& target : inputs.targets) {
      result.last_score = run_once(target);
    }
  }

  const auto run_start = Clock::now();
  for (std::size_t index = 0; index < options.iterations; ++index) {
    for (const auto& target : inputs.targets) {
      result.last_score = run_once(target);
      result.checksum = checksum_add(result.checksum, result.last_score);
    }
  }
  const auto run_end = Clock::now();

  result.prepare_seconds = std::chrono::duration<double>(prepare_end - prepare_start).count();
  result.run_seconds = std::chrono::duration<double>(run_end - run_start).count();
  result.calls = options.iterations;
  result.scored_targets = options.iterations * inputs.targets.size();
  return result;
#else
  (void)prepared;
  (void)options;
  throw std::runtime_error("native parasail microbench support was not compiled in");
#endif
}

}  // namespace stride_align::microbench
