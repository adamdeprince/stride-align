#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mg_procedure.h"

#include "embedded_data.hpp"
#include "stride_align/batch.hpp"
#include "stride_align/beider_morse.hpp"
#include "stride_align/caverphone.hpp"
#include "stride_align/cologne_phonetic.hpp"
#include "stride_align/core.hpp"
#include "stride_align/daitch_mokotoff.hpp"
#include "stride_align/double_metaphone.hpp"
#include "stride_align/dtw.hpp"
#include "stride_align/lcs.hpp"
#include "stride_align/match_rating.hpp"
#include "stride_align/metaphone.hpp"
#include "stride_align/ngram.hpp"
#include "stride_align/nysiis.hpp"
#include "stride_align/partial_ratio.hpp"
#include "stride_align/ratcliff_obershelp.hpp"
#include "stride_align/soundex.hpp"
#include "stride_align/token_ratios.hpp"
#include "stride_align/wratio.hpp"
#include "target_profile.hpp"

#ifndef STRIDE_ALIGN_MEMGRAPH_SIMD_LEVEL
#define STRIDE_ALIGN_MEMGRAPH_SIMD_LEVEL "unspecified"
#endif

#ifndef STRIDE_ALIGN_MEMGRAPH_VERSION
#define STRIDE_ALIGN_MEMGRAPH_VERSION "unknown"
#endif

namespace {

namespace batch = ::stride_align::batch;
namespace core = ::stride_align::core;
namespace utf8 = ::stride_align::utf8;

using Scorer = batch::Scorer;
using Score = ::stride_align::Score;

constexpr std::int64_t kInheritedGapExtend =
    std::numeric_limits<std::int64_t>::min();
constexpr std::int64_t kDefaultCdistChunkSize = 256;
constexpr std::int64_t kMaximumCdistChunkSize = 65536;

void check_mgp(mgp_error error, std::string_view operation) {
  if (error == mgp_error::MGP_ERROR_NO_ERROR) return;
  throw std::runtime_error(
      std::string(operation) + " failed with MGP error " +
      std::to_string(static_cast<int>(error)));
}

mgp_value* argument(mgp_list* arguments, std::size_t index) {
  mgp_value* value = nullptr;
  check_mgp(mgp_list_at(arguments, index, &value), "mgp_list_at");
  return value;
}

std::string_view string_argument(mgp_list* arguments, std::size_t index) {
  const char* value = nullptr;
  check_mgp(
      mgp_value_get_string(argument(arguments, index), &value),
      "mgp_value_get_string");
  return value;
}

std::int64_t integer_argument(mgp_list* arguments, std::size_t index) {
  std::int64_t value = 0;
  check_mgp(
      mgp_value_get_int(argument(arguments, index), &value),
      "mgp_value_get_int");
  return value;
}

double number_argument(mgp_list* arguments, std::size_t index) {
  mgp_value* value = argument(arguments, index);
  mgp_value_type type{};
  check_mgp(mgp_value_get_type(value, &type), "mgp_value_get_type");
  if (type == mgp_value_type::MGP_VALUE_TYPE_INT) {
    std::int64_t integer = 0;
    check_mgp(mgp_value_get_int(value, &integer), "mgp_value_get_int");
    return static_cast<double>(integer);
  }
  double output = 0.0;
  check_mgp(mgp_value_get_double(value, &output), "mgp_value_get_double");
  return output;
}

bool boolean_argument(mgp_list* arguments, std::size_t index) {
  int value = 0;
  check_mgp(
      mgp_value_get_bool(argument(arguments, index), &value),
      "mgp_value_get_bool");
  return value != 0;
}

std::vector<std::uint32_t> codepoints_argument(
    mgp_list* arguments,
    std::size_t index) {
  return utf8::prepare_streaming(string_argument(arguments, index));
}

std::string encode_utf8(std::span<const std::uint32_t> input) {
  std::string output;
  output.reserve(input.size());
  for (const std::uint32_t value : input) {
    if (value <= 0x7fU) {
      output.push_back(static_cast<char>(value));
    } else if (value <= 0x7ffU) {
      output.push_back(static_cast<char>(0xc0U | (value >> 6U)));
      output.push_back(static_cast<char>(0x80U | (value & 0x3fU)));
    } else if (value <= 0xffffU) {
      output.push_back(static_cast<char>(0xe0U | (value >> 12U)));
      output.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | (value & 0x3fU)));
    } else {
      output.push_back(static_cast<char>(0xf0U | (value >> 18U)));
      output.push_back(static_cast<char>(0x80U | ((value >> 12U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | (value & 0x3fU)));
    }
  }
  return output;
}

class OwnedValue {
 public:
  OwnedValue() = default;
  explicit OwnedValue(mgp_value* value) : value_(value) {}
  OwnedValue(const OwnedValue&) = delete;
  OwnedValue& operator=(const OwnedValue&) = delete;
  OwnedValue(OwnedValue&& other) noexcept
      : value_(std::exchange(other.value_, nullptr)) {}
  OwnedValue& operator=(OwnedValue&& other) noexcept {
    if (this == &other) return *this;
    if (value_ != nullptr) mgp_value_destroy(value_);
    value_ = std::exchange(other.value_, nullptr);
    return *this;
  }
  ~OwnedValue() {
    if (value_ != nullptr) mgp_value_destroy(value_);
  }

  mgp_value* get() const noexcept { return value_; }

 private:
  mgp_value* value_ = nullptr;
};

OwnedValue make_value(std::int64_t value, mgp_memory* memory);
OwnedValue make_value(double value, mgp_memory* memory);
OwnedValue make_value(bool value, mgp_memory* memory);
OwnedValue make_value(std::nullptr_t, mgp_memory* memory);
OwnedValue make_value(std::string_view value, mgp_memory* memory);

class MapBuilder {
 public:
  explicit MapBuilder(mgp_memory* memory) : memory_(memory) {
    check_mgp(mgp_map_make_empty(memory, &map_), "mgp_map_make_empty");
  }
  MapBuilder(const MapBuilder&) = delete;
  MapBuilder& operator=(const MapBuilder&) = delete;
  ~MapBuilder() {
    if (map_ != nullptr) mgp_map_destroy(map_);
  }

  template <typename Value>
  void insert(const char* name, Value&& value) {
    OwnedValue item = make_value(std::forward<Value>(value), memory_);
    check_mgp(mgp_map_insert(map_, name, item.get()), "mgp_map_insert");
  }

  OwnedValue finish() {
    mgp_value* value = nullptr;
    check_mgp(mgp_value_make_map(map_, &value), "mgp_value_make_map");
    map_ = nullptr;
    return OwnedValue(value);
  }

 private:
  mgp_memory* memory_;
  mgp_map* map_ = nullptr;
};

OwnedValue make_value(std::int64_t value, mgp_memory* memory) {
  mgp_value* output = nullptr;
  check_mgp(mgp_value_make_int(value, memory, &output), "mgp_value_make_int");
  return OwnedValue(output);
}

OwnedValue make_value(double value, mgp_memory* memory) {
  mgp_value* output = nullptr;
  check_mgp(
      mgp_value_make_double(value, memory, &output),
      "mgp_value_make_double");
  return OwnedValue(output);
}

OwnedValue make_value(bool value, mgp_memory* memory) {
  mgp_value* output = nullptr;
  check_mgp(
      mgp_value_make_bool(value ? 1 : 0, memory, &output),
      "mgp_value_make_bool");
  return OwnedValue(output);
}

OwnedValue make_value(std::nullptr_t, mgp_memory* memory) {
  mgp_value* output = nullptr;
  check_mgp(mgp_value_make_null(memory, &output), "mgp_value_make_null");
  return OwnedValue(output);
}

OwnedValue make_value(std::string_view value, mgp_memory* memory) {
  const std::string terminated(value);
  mgp_value* output = nullptr;
  check_mgp(
      mgp_value_make_string(terminated.c_str(), memory, &output),
      "mgp_value_make_string");
  return OwnedValue(output);
}

template <typename Value>
void set_function_value(
    mgp_func_result* result,
    mgp_memory* memory,
    Value&& output) {
  OwnedValue value = make_value(std::forward<Value>(output), memory);
  check_mgp(
      mgp_func_result_set_value(result, value.get(), memory),
      "mgp_func_result_set_value");
}

void set_function_value(
    mgp_func_result* result,
    mgp_memory* memory,
    OwnedValue output) {
  check_mgp(
      mgp_func_result_set_value(result, output.get(), memory),
      "mgp_func_result_set_value");
}

void set_function_error(
    mgp_func_result* result,
    mgp_memory* memory,
    const char* message) noexcept {
  static_cast<void>(mgp_func_result_set_error_msg(result, message, memory));
}

template <typename Function>
void run_function(
    mgp_func_result* result,
    mgp_memory* memory,
    Function&& function) noexcept {
  try {
    function();
  } catch (const std::exception& error) {
    set_function_error(result, memory, error.what());
  } catch (...) {
    set_function_error(result, memory, "unknown stride-align error");
  }
}

utf8::PreparedPair pair_argument(mgp_list* arguments) {
  return utf8::prepare_pair(
      string_argument(arguments, 0), string_argument(arguments, 1));
}

template <Scorer Metric>
void integer_score(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const auto pair = pair_argument(arguments);
    const double score = batch::score_prepared(Metric, pair);
    set_function_value(result, memory, static_cast<std::int64_t>(score));
  });
}

template <Scorer Metric>
void real_score(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const auto pair = pair_argument(arguments);
    set_function_value(result, memory, batch::score_prepared(Metric, pair));
  });
}

template <bool Indel, bool Normalized>
void cutoff_score(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const auto pair = pair_argument(arguments);
    if constexpr (Normalized) {
      const double cutoff = number_argument(arguments, 2);
      if (cutoff == -1.0) {
        set_function_value(
            result, memory,
            Indel ? core::indel_similarity(pair)
                  : core::levenshtein_similarity(pair));
        return;
      }
      if (!std::isfinite(cutoff) || cutoff < 0.0 || cutoff > 1.0) {
        throw std::invalid_argument(
            "score_cutoff must be -1 or between 0 and 1");
      }
      const std::size_t maximum = Indel
          ? pair.query_size() + pair.target_size()
          : std::max(pair.query_size(), pair.target_size());
      const std::size_t distance_cutoff =
          batch::normalized_distance_cutoff(cutoff, maximum);
      const std::size_t distance = Indel
          ? core::indel_distance(pair, distance_cutoff)
          : core::levenshtein_distance(pair, distance_cutoff);
      double similarity = Indel
          ? ::stride_align::indel::normalize(
                distance, pair.query_size(), pair.target_size())
          : ::stride_align::levenshtein::normalize(
                distance, pair.query_size(), pair.target_size());
      if (similarity < cutoff) similarity = 0.0;
      set_function_value(result, memory, similarity);
    } else {
      const std::int64_t input = integer_argument(arguments, 2);
      if (input == -1) {
        const std::size_t distance = Indel
            ? core::indel_distance(pair)
            : core::levenshtein_distance(pair);
        set_function_value(
            result, memory, static_cast<std::int64_t>(distance));
        return;
      }
      if (input < 0) {
        throw std::invalid_argument(
            "score_cutoff must be -1 or non-negative");
      }
      const std::size_t cutoff = static_cast<std::size_t>(input);
      const std::size_t distance = Indel
          ? core::indel_distance(pair, cutoff)
          : core::levenshtein_distance(pair, cutoff);
      set_function_value(
          result, memory, static_cast<std::int64_t>(distance));
    }
  });
}

void jaro_winkler_score(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const double weight = number_argument(arguments, 2);
    const double threshold = number_argument(arguments, 3);
    const std::int64_t cap = integer_argument(arguments, 4);
    if (!std::isfinite(weight) || weight < 0.0 ||
        !std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0 ||
        cap < 0) {
      throw std::invalid_argument("invalid Jaro-Winkler parameters");
    }
    const auto pair = pair_argument(arguments);
    set_function_value(
        result, memory,
        core::jaro_winkler_similarity(
            pair, weight, threshold, static_cast<std::size_t>(cap)));
  });
}

batch::ScoreOptions alignment_options(mgp_list* arguments, std::size_t offset) {
  batch::ScoreOptions options;
  options.match_score = integer_argument(arguments, offset);
  options.mismatch_score = integer_argument(arguments, offset + 1U);
  options.gap_open_score = integer_argument(arguments, offset + 2U);
  const std::int64_t gap_extend = integer_argument(arguments, offset + 3U);
  options.gap_extend_score = gap_extend == kInheritedGapExtend
      ? options.gap_open_score
      : gap_extend;
  return options;
}

template <bool Local, bool Normalized>
void alignment_score(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const auto pair = pair_argument(arguments);
    const auto options = alignment_options(arguments, 2);
    const Scorer scorer = Local
        ? (Normalized ? Scorer::smith_waterman_normalized
                      : Scorer::smith_waterman)
        : (Normalized ? Scorer::needleman_wunsch_normalized
                      : Scorer::needleman_wunsch);
    const double output = batch::score_prepared(scorer, pair, options);
    if constexpr (Normalized) {
      set_function_value(result, memory, output);
    } else {
      set_function_value(result, memory, static_cast<std::int64_t>(output));
    }
  });
}

batch::ScoreOptions dynamic_options(mgp_list* arguments, std::size_t offset) {
  auto options = alignment_options(arguments, offset);
  options.prefix_weight = number_argument(arguments, offset + 4U);
  options.prefix_threshold = number_argument(arguments, offset + 5U);
  const std::int64_t prefix_cap = integer_argument(arguments, offset + 6U);
  if (!std::isfinite(options.prefix_weight) || options.prefix_weight < 0.0 ||
      !std::isfinite(options.prefix_threshold) ||
      options.prefix_threshold < 0.0 || options.prefix_threshold > 1.0 ||
      prefix_cap < 0) {
    throw std::invalid_argument("invalid scorer parameters");
  }
  options.prefix_cap = static_cast<std::size_t>(prefix_cap);
  return options;
}

void dynamic_score(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const auto pair = pair_argument(arguments);
    const Scorer scorer = batch::parse_scorer(string_argument(arguments, 2));
    const auto options = dynamic_options(arguments, 3);
    set_function_value(
        result, memory, batch::score_prepared(scorer, pair, options));
  });
}

enum class LcsOperation {
  length,
  substring_length,
  substring,
};

template <LcsOperation Operation>
void lcs_function(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const auto left = codepoints_argument(arguments, 0);
    const auto right = codepoints_argument(arguments, 1);
    if constexpr (Operation == LcsOperation::length) {
      set_function_value(
          result, memory,
          static_cast<std::int64_t>(::stride_align::lcs::lcs_length(left, right)));
    } else if constexpr (Operation == LcsOperation::substring_length) {
      set_function_value(
          result, memory,
          static_cast<std::int64_t>(
              ::stride_align::lcs::lcs_substring_length(left, right)));
    } else {
      const auto substring = ::stride_align::lcs::lcs_substring(left, right);
      set_function_value(result, memory, std::string_view(encode_utf8(substring)));
    }
  });
}

enum class NgramMetric {
  jaccard,
  dice,
  cosine,
  overlap,
};

template <NgramMetric Metric>
void ngram_function(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const std::int64_t input_n = integer_argument(arguments, 2);
    if (input_n < 0) throw std::invalid_argument("n must be non-negative");
    const std::size_t n = static_cast<std::size_t>(input_n);
    const auto left = codepoints_argument(arguments, 0);
    const auto right = codepoints_argument(arguments, 1);
    double output = 0.0;
    if constexpr (Metric == NgramMetric::jaccard) {
      output = ::stride_align::ngram::jaccard(left, right, n);
    } else if constexpr (Metric == NgramMetric::dice) {
      output = ::stride_align::ngram::dice(left, right, n);
    } else if constexpr (Metric == NgramMetric::cosine) {
      output = ::stride_align::ngram::cosine(left, right, n);
    } else {
      output = ::stride_align::ngram::overlap(left, right, n);
    }
    set_function_value(result, memory, output);
  });
}

enum class StringSimilarity {
  ratcliff,
  partial,
  token_sort,
  token_set,
  partial_token_sort,
  partial_token_set,
  weighted,
};

double string_similarity(
    StringSimilarity metric,
    const std::vector<std::uint32_t>& left,
    const std::vector<std::uint32_t>& right) {
  const auto left_span = std::span<const std::uint32_t>(left);
  const auto right_span = std::span<const std::uint32_t>(right);
  switch (metric) {
    case StringSimilarity::ratcliff:
      return ::stride_align::ratcliff_obershelp::
          ratcliff_obershelp_similarity(left, right);
    case StringSimilarity::partial:
      return ::stride_align::partial_ratio::partial_ratio(left, right);
    case StringSimilarity::token_sort:
      return ::stride_align::token_ratios::token_sort_ratio(left, right);
    case StringSimilarity::token_set:
      return ::stride_align::token_ratios::token_set_ratio(left, right);
    case StringSimilarity::partial_token_sort:
      return ::stride_align::wratio::
          partial_token_sort_ratio_engine<std::uint32_t>(left_span, right_span);
    case StringSimilarity::partial_token_set:
      return ::stride_align::wratio::
          partial_token_set_ratio_engine<std::uint32_t>(left_span, right_span);
    case StringSimilarity::weighted:
      return ::stride_align::wratio::native_wratio(left, right);
  }
  throw std::logic_error("unknown string similarity");
}

template <StringSimilarity Metric>
void string_similarity_function(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    set_function_value(
        result, memory,
        string_similarity(
            Metric, codepoints_argument(arguments, 0),
            codepoints_argument(arguments, 1)));
  });
}

std::vector<std::vector<std::uint32_t>> word_tokens(
    std::span<const std::uint32_t> input) {
  std::vector<std::vector<std::uint32_t>> output;
  std::vector<std::uint32_t> token;
  for (const std::uint32_t value : input) {
    const bool whitespace = value == 0x20U ||
        (value >= 0x09U && value <= 0x0dU) || value == 0x85U ||
        value == 0xa0U || value == 0x1680U ||
        (value >= 0x2000U && value <= 0x200aU) || value == 0x2028U ||
        value == 0x2029U || value == 0x202fU || value == 0x205fU ||
        value == 0x3000U;
    if (whitespace) {
      if (!token.empty()) output.push_back(std::move(token));
      token.clear();
    } else {
      token.push_back(value);
    }
  }
  if (!token.empty()) output.push_back(std::move(token));
  return output;
}

double inner_similarity(
    std::string_view name,
    const std::vector<std::uint32_t>& left,
    const std::vector<std::uint32_t>& right) {
  const std::string left_text = encode_utf8(left);
  const std::string right_text = encode_utf8(right);
  const auto pair = utf8::prepare_pair(left_text, right_text);
  if (name == "jaro") return core::jaro_similarity(pair);
  if (name == "jaro_winkler") return core::jaro_winkler_similarity(pair);
  if (name == "levenshtein_ratio" || name == "levenshtein_normalized") {
    return core::levenshtein_similarity(pair);
  }
  if (name == "indel_ratio" || name == "indel_normalized") {
    return core::indel_similarity(pair);
  }
  throw std::invalid_argument(
      "unknown Monge-Elkan inner similarity: " + std::string(name));
}

double monge_direction(
    const std::vector<std::vector<std::uint32_t>>& left,
    const std::vector<std::vector<std::uint32_t>>& right,
    std::string_view inner) {
  double total = 0.0;
  for (const auto& token : left) {
    double best = 0.0;
    for (const auto& candidate : right) {
      best = std::max(best, inner_similarity(inner, token, candidate));
      if (best >= 1.0) break;
    }
    total += best;
  }
  return total / static_cast<double>(left.size());
}

void monge_elkan(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const auto left_points = codepoints_argument(arguments, 0);
    const auto right_points = codepoints_argument(arguments, 1);
    const auto left = word_tokens(left_points);
    const auto right = word_tokens(right_points);
    if (left.empty() && right.empty()) {
      set_function_value(result, memory, 1.0);
      return;
    }
    if (left.empty() || right.empty()) {
      set_function_value(result, memory, 0.0);
      return;
    }
    const std::string_view inner = string_argument(arguments, 2);
    const bool symmetric = boolean_argument(arguments, 3);
    const double forward = monge_direction(left, right, inner);
    set_function_value(
        result, memory,
        symmetric
            ? (forward + monge_direction(right, left, inner)) / 2.0
            : forward);
  });
}

template <typename Function>
void unary_string_function(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const std::string output = Function{}(string_argument(arguments, 0));
    set_function_value(result, memory, std::string_view(output));
  });
}

struct Soundex {
  std::string operator()(std::string_view input) const {
    return ::stride_align::phonetic::soundex(input);
  }
};

struct Nysiis {
  std::string operator()(std::string_view input) const {
    return ::stride_align::phonetic::nysiis(input);
  }
};

struct MatchRatingCodex {
  std::string operator()(std::string_view input) const {
    return ::stride_align::phonetic::match_rating_codex(input);
  }
};

struct Caverphone {
  std::string operator()(std::string_view input) const {
    return ::stride_align::phonetic::caverphone(input);
  }
};

template <typename Function>
void encoded_equal(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const std::string left = Function{}(string_argument(arguments, 0));
    const std::string right = Function{}(string_argument(arguments, 1));
    set_function_value(result, memory, !left.empty() && left == right);
  });
}

void metaphone(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const std::int64_t variant_input = integer_argument(arguments, 1);
    if (variant_input < 0 || variant_input > 1) {
      throw std::invalid_argument("variant must be 0 or 1");
    }
    const auto variant = static_cast<::stride_align::phonetic::MetaphoneVariant>(
        variant_input);
    const std::string output = ::stride_align::phonetic::metaphone(
        string_argument(arguments, 0), variant);
    set_function_value(result, memory, std::string_view(output));
  });
}

void metaphone_equal(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const std::int64_t variant_input = integer_argument(arguments, 2);
    if (variant_input < 0 || variant_input > 1) {
      throw std::invalid_argument("variant must be 0 or 1");
    }
    const auto variant = static_cast<::stride_align::phonetic::MetaphoneVariant>(
        variant_input);
    const std::string left = ::stride_align::phonetic::metaphone(
        string_argument(arguments, 0), variant);
    const std::string right = ::stride_align::phonetic::metaphone(
        string_argument(arguments, 1), variant);
    set_function_value(result, memory, !left.empty() && left == right);
  });
}

void match_rating_compare(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    set_function_value(
        result, memory,
        ::stride_align::phonetic::match_rating_compare(
            string_argument(arguments, 0), string_argument(arguments, 1)));
  });
}

void cologne_phonetic(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const std::string output = ::stride_align::phonetic::cologne_phonetic(
        codepoints_argument(arguments, 0));
    set_function_value(result, memory, std::string_view(output));
  });
}

void daitch_mokotoff(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const std::string output = ::stride_align::phonetic::daitch_mokotoff(
        codepoints_argument(arguments, 0), boolean_argument(arguments, 1),
        boolean_argument(arguments, 2));
    set_function_value(result, memory, std::string_view(output));
  });
}

void double_metaphone(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const std::int64_t maximum = integer_argument(arguments, 1);
    const std::int64_t variant_input = integer_argument(arguments, 2);
    if (maximum < 0) {
      throw std::invalid_argument("max_length must be non-negative");
    }
    if (variant_input < 0 || variant_input > 1) {
      throw std::invalid_argument("variant must be 0 or 1");
    }
    const auto encoded = ::stride_align::phonetic::double_metaphone(
        string_argument(arguments, 0), static_cast<std::size_t>(maximum),
        static_cast<::stride_align::phonetic::DoubleMetaphoneVariant>(
            variant_input));
    MapBuilder output(memory);
    output.insert("primary", std::string_view(encoded.primary));
    output.insert("alternate", std::string_view(encoded.alternate));
    set_function_value(result, memory, output.finish());
  });
}

void beider_morse(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const std::int64_t rule_input = integer_argument(arguments, 1);
    const std::int64_t maximum = integer_argument(arguments, 3);
    if (rule_input < 0 || rule_input > 1) {
      throw std::invalid_argument("rule_type must be 0 or 1");
    }
    if (maximum < 0) {
      throw std::invalid_argument("max_phonemes must be non-negative");
    }
    static const bool registered = [] {
      ::stride_align::phonetic::bmpm_register_resources(
          ::stride_align_memgraph_embedded::bmpm_resources());
      return true;
    }();
    static_cast<void>(registered);
    const std::string output = ::stride_align::phonetic::beider_morse(
        codepoints_argument(arguments, 0),
        static_cast<::stride_align::phonetic::BmpmRuleType>(rule_input),
        boolean_argument(arguments, 2), static_cast<std::size_t>(maximum));
    set_function_value(result, memory, std::string_view(output));
  });
}

std::vector<double> number_list_argument(
    mgp_list* arguments,
    std::size_t index) {
  mgp_list* input = nullptr;
  check_mgp(
      mgp_value_get_list(argument(arguments, index), &input),
      "mgp_value_get_list");
  std::size_t count = 0;
  check_mgp(mgp_list_size(input, &count), "mgp_list_size");
  std::vector<double> output;
  output.reserve(count);
  for (std::size_t item = 0; item < count; ++item) {
    mgp_value* value = nullptr;
    check_mgp(mgp_list_at(input, item, &value), "mgp_list_at");
    mgp_value_type type{};
    check_mgp(mgp_value_get_type(value, &type), "mgp_value_get_type");
    if (type == mgp_value_type::MGP_VALUE_TYPE_INT) {
      std::int64_t integer = 0;
      check_mgp(mgp_value_get_int(value, &integer), "mgp_value_get_int");
      output.push_back(static_cast<double>(integer));
    } else {
      double number = 0.0;
      check_mgp(mgp_value_get_double(value, &number), "mgp_value_get_double");
      output.push_back(number);
    }
  }
  return output;
}

std::optional<std::size_t> dtw_window(
    double input,
    std::size_t query_size,
    std::size_t target_size) {
  if (input == -1.0) return std::nullopt;
  if (input < 0.0 || !std::isfinite(input)) {
    throw std::invalid_argument("window must be -1 or non-negative and finite");
  }
  if (input > 0.0 && input < 1.0) {
    return static_cast<std::size_t>(std::ceil(
        input * static_cast<double>(std::max(query_size, target_size))));
  }
  if (std::trunc(input) != input) {
    throw std::invalid_argument(
        "window must be an integer radius or a fraction in (0, 1)");
  }
  return static_cast<std::size_t>(input);
}

void dtw(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const auto query = number_list_argument(arguments, 0);
    const auto target = number_list_argument(arguments, 1);
    if (query.empty() || target.empty()) {
      throw std::invalid_argument("DTW inputs must not be empty");
    }
    const double window_input = number_argument(arguments, 2);
    const std::string_view distance = string_argument(arguments, 3);
    const double cutoff_input = number_argument(arguments, 4);
    const auto kind = distance == "l1"
        ? ::stride_align::dtw::DistanceKind::kL1
        : ::stride_align::dtw::DistanceKind::kL2Squared;
    if (distance != "l1" && distance != "l2_squared") {
      throw std::invalid_argument("distance must be 'l1' or 'l2_squared'");
    }
    std::optional<double> cutoff;
    if (cutoff_input != -1.0) {
      if (!std::isfinite(cutoff_input) || cutoff_input < 0.0) {
        throw std::invalid_argument(
            "score_cutoff must be -1 or non-negative and finite");
      }
      cutoff = cutoff_input;
    }
    set_function_value(
        result, memory,
        ::stride_align::dtw::dtw_score_scalar<double, double>(
            query, target, kind,
            dtw_window(window_input, query.size(), target.size()), cutoff));
  });
}

std::pair<std::string, std::string> aligned_strings(
    const ::stride_align::AlignmentPath& path,
    std::span<const std::uint32_t> query,
    std::span<const std::uint32_t> target) {
  std::vector<std::uint32_t> aligned_query;
  std::vector<std::uint32_t> aligned_target;
  aligned_query.reserve(path.operations.size());
  aligned_target.reserve(path.operations.size());
  std::size_t query_index = path.query_start;
  std::size_t target_index = path.target_start;
  for (const char operation : path.operations) {
    if (operation == '=' || operation == 'X') {
      aligned_query.push_back(query[query_index++]);
      aligned_target.push_back(target[target_index++]);
    } else if (operation == 'D') {
      aligned_query.push_back(query[query_index++]);
      aligned_target.push_back('-');
    } else if (operation == 'I') {
      aligned_query.push_back('-');
      aligned_target.push_back(target[target_index++]);
    } else {
      throw std::logic_error("unknown alignment path operation");
    }
  }
  return {encode_utf8(aligned_query), encode_utf8(aligned_target)};
}

template <bool Local, bool CigarOnly>
void alignment_path(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const std::string query_text(string_argument(arguments, 0));
    const std::string target_text(string_argument(arguments, 1));
    const auto options = alignment_options(arguments, 2);
    const auto pair = utf8::prepare_pair(query_text, target_text);
    ::stride_align::AlignmentPath path;
    if (options.gap_open_score == options.gap_extend_score) {
      path = Local
          ? core::smith_waterman_path(
                pair, options.match_score, options.mismatch_score,
                options.gap_open_score)
          : core::needleman_wunsch_path(
                pair, options.match_score, options.mismatch_score,
                options.gap_open_score);
    } else {
      path = Local
          ? core::smith_waterman_affine_path(
                pair, options.match_score, options.mismatch_score,
                options.gap_open_score, options.gap_extend_score)
          : core::needleman_wunsch_affine_path(
                pair, options.match_score, options.mismatch_score,
                options.gap_open_score, options.gap_extend_score);
    }
    if constexpr (CigarOnly) {
      set_function_value(result, memory, std::string_view(path.cigar));
      return;
    }
    const auto query = utf8::prepare_streaming(query_text);
    const auto target = utf8::prepare_streaming(target_text);
    const auto [aligned_query, aligned_target] =
        aligned_strings(path, query, target);
    MapBuilder output(memory);
    output.insert("score", static_cast<std::int64_t>(path.score));
    output.insert("query_start", static_cast<std::int64_t>(path.query_start));
    output.insert("query_end", static_cast<std::int64_t>(path.query_end));
    output.insert("target_start", static_cast<std::int64_t>(path.target_start));
    output.insert("target_end", static_cast<std::int64_t>(path.target_end));
    output.insert("operations", std::string_view(path.operations));
    output.insert("cigar", std::string_view(path.cigar));
    output.insert("matches", static_cast<std::int64_t>(path.matches));
    output.insert("mismatches", static_cast<std::int64_t>(path.mismatches));
    output.insert("insertions", static_cast<std::int64_t>(path.insertions));
    output.insert("deletions", static_cast<std::int64_t>(path.deletions));
    output.insert(
        "aligned_length", static_cast<std::int64_t>(path.aligned_length));
    output.insert("aligned_query", std::string_view(aligned_query));
    output.insert("aligned_target", std::string_view(aligned_target));
    set_function_value(result, memory, output.finish());
  });
}

struct Matrix {
  std::string sql_name;
  std::string name;
  std::string alphabet;
  std::uint16_t wildcard = 0;
  Score gap_score = -1;
  Score gap_open = 0;
  Score gap_extend = 0;
  bool has_affine = false;
  std::vector<std::int8_t> values;
  std::unordered_map<std::uint32_t, std::uint16_t> indices;

  std::size_t stride() const noexcept { return indices.size(); }

  std::int8_t lookup(std::uint16_t left, std::uint16_t right) const {
    return values[static_cast<std::size_t>(left) * stride() + right];
  }
};

std::string canonical_matrix_name(std::string_view input) {
  std::string output;
  output.reserve(input.size());
  for (const char value : input) {
    if (value >= 'A' && value <= 'Z') {
      output.push_back(static_cast<char>(value - 'A' + 'a'));
    } else if (value == '-' || value == '.') {
      output.push_back('_');
    } else {
      output.push_back(value);
    }
  }
  if (output == "nuc_4_4") output = "nuc44";
  if (output == "ascii") output = "ascii_text";
  if (output == "qwerty") output = "keyboard:qwerty";
  return output;
}

Matrix finish_matrix(Matrix matrix, std::string_view wildcard) {
  const auto alphabet = utf8::prepare_streaming(matrix.alphabet);
  if (alphabet.empty() || alphabet.size() > 256U) {
    throw std::invalid_argument(
        "substitution-matrix alphabet must contain 1 through 256 symbols");
  }
  for (std::size_t index = 0; index < alphabet.size(); ++index) {
    if (!matrix.indices.emplace(
            alphabet[index], static_cast<std::uint16_t>(index)).second) {
      throw std::invalid_argument(
          "substitution-matrix alphabet contains duplicate symbols");
    }
  }
  const auto wildcard_points = utf8::prepare_streaming(wildcard);
  if (wildcard_points.size() != 1U ||
      !matrix.indices.contains(wildcard_points.front())) {
    throw std::invalid_argument(
        "substitution-matrix wildcard must be one symbol in the alphabet");
  }
  matrix.wildcard = matrix.indices.at(wildcard_points.front());
  if (matrix.values.size() != alphabet.size() * alphabet.size()) {
    throw std::invalid_argument(
        "substitution-matrix score grid does not match the alphabet");
  }
  return matrix;
}

std::vector<Matrix> build_matrices() {
  std::vector<Matrix> output;
  for (auto& record : ::stride_align_memgraph_embedded::matrices()) {
    Matrix matrix;
    matrix.sql_name = record.sql_name;
    matrix.name = record.name;
    if (matrix.sql_name == "ascii_text" ||
        matrix.sql_name.starts_with("keyboard:")) {
      matrix.alphabet.reserve(128U);
      for (std::uint32_t value = 0; value < 128U; ++value) {
        matrix.alphabet.push_back(static_cast<char>(value));
      }
    } else {
      matrix.alphabet = record.alphabet;
    }
    matrix.gap_score = record.gap_score;
    matrix.gap_open = record.gap_open;
    matrix.gap_extend = record.gap_extend;
    matrix.has_affine = record.has_affine;
    matrix.values = std::move(record.values);
    output.push_back(finish_matrix(std::move(matrix), record.wildcard));
  }
  return output;
}

const std::vector<Matrix>& matrices() {
  static const std::vector<Matrix> value = build_matrices();
  return value;
}

const Matrix& resolve_matrix(std::string_view input) {
  const std::string canonical = canonical_matrix_name(input);
  for (const Matrix& matrix : matrices()) {
    if (canonical_matrix_name(matrix.sql_name) == canonical ||
        canonical_matrix_name(matrix.name) == canonical) {
      return matrix;
    }
  }
  throw std::invalid_argument(
      "unknown stride-align matrix: " + std::string(input));
}

std::vector<std::uint16_t> matrix_encode(
    const Matrix& matrix,
    std::string_view input) {
  const auto points = utf8::prepare_streaming(input);
  std::vector<std::uint16_t> output;
  output.reserve(points.size());
  for (const std::uint32_t value : points) {
    const auto found = matrix.indices.find(value);
    output.push_back(
        found == matrix.indices.end() ? matrix.wildcard : found->second);
  }
  return output;
}

Score matrix_score(
    const Matrix& matrix,
    std::string_view query,
    std::string_view target,
    bool local,
    Score gap_open,
    Score gap_extend) {
  const auto left = matrix_encode(matrix, query);
  const auto right = matrix_encode(matrix, target);
  const auto lookup = [&](std::uint16_t a, std::uint16_t b) {
    return matrix.lookup(a, b);
  };
  return local
      ? core::substitution_matrix_affine_score<true>(
            left, right, lookup, gap_open, gap_extend)
      : core::substitution_matrix_affine_score<false>(
            left, right, lookup, gap_open, gap_extend);
}

::stride_align::AlignmentPath matrix_path(
    const Matrix& matrix,
    std::string_view query,
    std::string_view target,
    bool local,
    Score gap_open,
    Score gap_extend) {
  const auto left = matrix_encode(matrix, query);
  const auto right = matrix_encode(matrix, target);
  const auto lookup = [&](std::uint16_t a, std::uint16_t b) {
    return matrix.lookup(a, b);
  };
  return local
      ? core::substitution_matrix_affine_path<true>(
            left, right, lookup, gap_open, gap_extend)
      : core::substitution_matrix_affine_path<false>(
            left, right, lookup, gap_open, gap_extend);
}

void set_path_value(
    mgp_func_result* result,
    mgp_memory* memory,
    const ::stride_align::AlignmentPath& path,
    std::string_view query_text,
    std::string_view target_text) {
  const auto query = utf8::prepare_streaming(query_text);
  const auto target = utf8::prepare_streaming(target_text);
  const auto [aligned_query, aligned_target] =
      aligned_strings(path, query, target);
  MapBuilder output(memory);
  output.insert("score", static_cast<std::int64_t>(path.score));
  output.insert("query_start", static_cast<std::int64_t>(path.query_start));
  output.insert("query_end", static_cast<std::int64_t>(path.query_end));
  output.insert("target_start", static_cast<std::int64_t>(path.target_start));
  output.insert("target_end", static_cast<std::int64_t>(path.target_end));
  output.insert("operations", std::string_view(path.operations));
  output.insert("cigar", std::string_view(path.cigar));
  output.insert("matches", static_cast<std::int64_t>(path.matches));
  output.insert("mismatches", static_cast<std::int64_t>(path.mismatches));
  output.insert("insertions", static_cast<std::int64_t>(path.insertions));
  output.insert("deletions", static_cast<std::int64_t>(path.deletions));
  output.insert("aligned_length", static_cast<std::int64_t>(path.aligned_length));
  output.insert("aligned_query", std::string_view(aligned_query));
  output.insert("aligned_target", std::string_view(aligned_target));
  set_function_value(result, memory, output.finish());
}

std::pair<Score, Score> matrix_gaps(
    const Matrix& matrix,
    std::int64_t gap_open,
    std::int64_t gap_extend,
    bool use_matrix_defaults) {
  if (gap_open == kInheritedGapExtend) {
    if (!use_matrix_defaults) return {-1, -1};
    return matrix.has_affine
        ? std::pair<Score, Score>{matrix.gap_open, matrix.gap_extend}
        : std::pair<Score, Score>{matrix.gap_score, matrix.gap_score};
  }
  return {
      gap_open,
      gap_extend == kInheritedGapExtend ? gap_open : gap_extend,
  };
}

void matrix_info(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const Matrix& matrix = resolve_matrix(string_argument(arguments, 0));
    MapBuilder output(memory);
    output.insert("name", std::string_view(matrix.name));
    output.insert("alphabet", std::string_view(matrix.alphabet));
    output.insert("stride", static_cast<std::int64_t>(matrix.stride()));
    output.insert(
        "wildcard_index", static_cast<std::int64_t>(matrix.wildcard));
    output.insert("gap_score", static_cast<std::int64_t>(matrix.gap_score));
    if (matrix.has_affine) {
      output.insert("gap_open", static_cast<std::int64_t>(matrix.gap_open));
      output.insert("gap_extend", static_cast<std::int64_t>(matrix.gap_extend));
    } else {
      output.insert("gap_open", nullptr);
      output.insert("gap_extend", nullptr);
    }
    set_function_value(result, memory, output.finish());
  });
}

void matrix_score_step_limit(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const Matrix& matrix = resolve_matrix(string_argument(arguments, 0));
    const auto [gap_open, gap_extend] = matrix_gaps(
        matrix, integer_argument(arguments, 1),
        integer_argument(arguments, 2), true);
    std::int64_t maximum = 0;
    for (const std::int8_t value : matrix.values) {
      maximum = std::max<std::int64_t>(
          maximum, std::abs(static_cast<int>(value)));
    }
    maximum = std::max<std::int64_t>(maximum, std::abs(gap_open));
    maximum = std::max<std::int64_t>(maximum, std::abs(gap_extend));
    set_function_value(result, memory, maximum);
  });
}

void substitution_matrix_score(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const Matrix& matrix = resolve_matrix(string_argument(arguments, 0));
    set_function_value(
        result, memory,
        static_cast<std::int64_t>(matrix_score(
            matrix, string_argument(arguments, 1),
            string_argument(arguments, 2), true,
            matrix.gap_score, matrix.gap_score)));
  });
}

template <bool Local, bool Path, bool CigarOnly>
void matrix_alignment(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    const std::string_view query = string_argument(arguments, 0);
    const std::string_view target = string_argument(arguments, 1);
    const Matrix& matrix = resolve_matrix(string_argument(arguments, 2));
    const auto [gap_open, gap_extend] = matrix_gaps(
        matrix, integer_argument(arguments, 3),
        integer_argument(arguments, 4), false);
    if constexpr (!Path) {
      set_function_value(
          result, memory,
          static_cast<std::int64_t>(matrix_score(
              matrix, query, target, Local, gap_open, gap_extend)));
    } else {
      const auto path = matrix_path(
          matrix, query, target, Local, gap_open, gap_extend);
      if constexpr (CigarOnly) {
        set_function_value(result, memory, std::string_view(path.cigar));
      } else {
        set_path_value(result, memory, path, query, target);
      }
    }
  });
}

void simd_level(
    mgp_list*,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    set_function_value(
        result, memory, std::string_view(STRIDE_ALIGN_MEMGRAPH_SIMD_LEVEL));
  });
}

void version(
    mgp_list*,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    set_function_value(
        result, memory, std::string_view(STRIDE_ALIGN_MEMGRAPH_VERSION));
  });
}

void backend_is_available(
    mgp_list* arguments,
    mgp_func_context*,
    mgp_func_result* result,
    mgp_memory* memory) noexcept {
  run_function(result, memory, [&] {
    set_function_value(
        result, memory,
        string_argument(arguments, 0) == STRIDE_ALIGN_MEMGRAPH_SIMD_LEVEL);
  });
}

struct CdistState {
  std::vector<std::vector<std::uint32_t>> queries;
  std::vector<std::vector<std::uint32_t>> targets;
  Scorer scorer = Scorer::levenshtein;
  batch::ScoreOptions options;
  std::size_t query_index = 0;
  std::size_t target_index = 0;
  std::size_t chunk_size = static_cast<std::size_t>(kDefaultCdistChunkSize);
  std::string error;
};

thread_local std::optional<CdistState> cdist_state;

std::vector<std::vector<std::uint32_t>> streaming_strings(
    mgp_list* arguments,
    std::size_t index) {
  mgp_list* input = nullptr;
  check_mgp(
      mgp_value_get_list(argument(arguments, index), &input),
      "mgp_value_get_list");
  std::size_t count = 0;
  check_mgp(mgp_list_size(input, &count), "mgp_list_size");
  std::vector<std::vector<std::uint32_t>> output;
  output.reserve(count);
  for (std::size_t item = 0; item < count; ++item) {
    mgp_value* value = nullptr;
    check_mgp(mgp_list_at(input, item, &value), "mgp_list_at");
    const char* text = nullptr;
    check_mgp(mgp_value_get_string(value, &text), "mgp_value_get_string");
    output.push_back(utf8::prepare_streaming(text));
  }
  return output;
}

utf8::PreparedPair streaming_pair(
    const std::vector<std::uint32_t>& query,
    const std::vector<std::uint32_t>& target) {
  utf8::PreparedPair pair;
  pair.width = utf8::TokenWidth::u32;
  pair.borrowed_ascii = false;
  pair.packed = false;
  pair.query = std::span<const std::uint32_t>(query.data(), query.size());
  pair.target = std::span<const std::uint32_t>(target.data(), target.size());
  return pair;
}

void cdist_initialize(mgp_list* arguments, mgp_graph*, mgp_memory*) noexcept {
  cdist_state.emplace();
  try {
    cdist_state->queries = streaming_strings(arguments, 0);
    cdist_state->targets = streaming_strings(arguments, 1);
    cdist_state->scorer = batch::parse_scorer(string_argument(arguments, 2));
    cdist_state->options = dynamic_options(arguments, 3);
    const std::int64_t chunk_size = integer_argument(arguments, 10);
    if (chunk_size <= 0 || chunk_size > kMaximumCdistChunkSize) {
      throw std::invalid_argument("chunk_size must be between 1 and 65536");
    }
    cdist_state->chunk_size = static_cast<std::size_t>(chunk_size);
  } catch (const std::exception& error) {
    cdist_state->error = error.what();
  } catch (...) {
    cdist_state->error = "unknown stride-align cdist initialization error";
  }
}

template <typename Value>
void insert_record_value(
    mgp_result_record* record,
    const char* name,
    Value&& output,
    mgp_memory* memory) {
  OwnedValue value = make_value(std::forward<Value>(output), memory);
  check_mgp(
      mgp_result_record_insert(record, name, value.get()),
      "mgp_result_record_insert");
}

void cdist(
    mgp_list*,
    mgp_graph* graph,
    mgp_result* result,
    mgp_memory* memory) noexcept {
  try {
    if (!cdist_state.has_value()) {
      throw std::runtime_error("cdist stream was not initialized");
    }
    CdistState& state = *cdist_state;
    if (!state.error.empty()) {
      static_cast<void>(mgp_result_set_error_msg(result, state.error.c_str()));
      return;
    }
    if (state.queries.empty() || state.targets.empty() ||
        state.query_index >= state.queries.size()) {
      return;
    }

    check_mgp(
        mgp_result_reserve(result, state.chunk_size),
        "mgp_result_reserve");
    std::size_t produced = 0;
    while (produced < state.chunk_size &&
           state.query_index < state.queries.size()) {
      if (mgp_must_abort(graph) != 0) {
        throw std::runtime_error("cdist was aborted by Memgraph");
      }
      const auto pair = streaming_pair(
          state.queries[state.query_index],
          state.targets[state.target_index]);
      const double score = batch::score_prepared(
          state.scorer, pair, state.options);

      mgp_result_record* record = nullptr;
      check_mgp(
          mgp_result_new_record(result, &record),
          "mgp_result_new_record");
      insert_record_value(
          record, "query_index",
          static_cast<std::int64_t>(state.query_index), memory);
      insert_record_value(
          record, "target_index",
          static_cast<std::int64_t>(state.target_index), memory);
      insert_record_value(record, "score", score, memory);

      ++produced;
      ++state.target_index;
      if (state.target_index == state.targets.size()) {
        state.target_index = 0;
        ++state.query_index;
      }
    }
  } catch (const std::exception& error) {
    static_cast<void>(mgp_result_set_error_msg(result, error.what()));
  } catch (...) {
    static_cast<void>(
        mgp_result_set_error_msg(result, "unknown stride-align cdist error"));
  }
}

void cdist_cleanup() noexcept {
  cdist_state.reset();
}

struct Types {
  mgp_type* boolean = nullptr;
  mgp_type* integer = nullptr;
  mgp_type* number = nullptr;
  mgp_type* string = nullptr;
  mgp_type* numbers = nullptr;
  mgp_type* strings = nullptr;

  Types() {
    check_mgp(mgp_type_bool(&boolean), "mgp_type_bool");
    check_mgp(mgp_type_int(&integer), "mgp_type_int");
    check_mgp(mgp_type_number(&number), "mgp_type_number");
    check_mgp(mgp_type_string(&string), "mgp_type_string");
    check_mgp(mgp_type_list(number, &numbers), "mgp_type_list");
    check_mgp(mgp_type_list(string, &strings), "mgp_type_list");
  }
};

mgp_func* add_function(
    mgp_module* module,
    const char* name,
    mgp_func_cb callback) {
  mgp_func* function = nullptr;
  check_mgp(
      mgp_module_add_function(module, name, callback, &function),
      "mgp_module_add_function");
  return function;
}

void add_required(mgp_func* function, const char* name, mgp_type* type) {
  check_mgp(mgp_func_add_arg(function, name, type), "mgp_func_add_arg");
}

template <typename Value>
void add_optional(
    mgp_func* function,
    const char* name,
    mgp_type* type,
    Value&& default_value,
    mgp_memory* memory) {
  OwnedValue value = make_value(std::forward<Value>(default_value), memory);
  check_mgp(
      mgp_func_add_opt_arg(function, name, type, value.get()),
      "mgp_func_add_opt_arg");
}

void add_pair_function(
    mgp_module* module,
    const Types& types,
    const char* name,
    mgp_func_cb callback) {
  mgp_func* function = add_function(module, name, callback);
  add_required(function, "query", types.string);
  add_required(function, "target", types.string);
}

void add_cutoff_function(
    mgp_module* module,
    mgp_memory* memory,
    const Types& types,
    const char* name,
    mgp_func_cb callback,
    bool normalized) {
  mgp_func* function = add_function(module, name, callback);
  add_required(function, "query", types.string);
  add_required(function, "target", types.string);
  if (normalized) {
    add_optional(function, "score_cutoff", types.number, -1.0, memory);
  } else {
    add_optional(function, "score_cutoff", types.integer, std::int64_t{-1}, memory);
  }
}

void add_jaro_winkler_function(
    mgp_module* module,
    mgp_memory* memory,
    const Types& types,
    const char* name) {
  mgp_func* function = add_function(module, name, jaro_winkler_score);
  add_required(function, "query", types.string);
  add_required(function, "target", types.string);
  add_optional(function, "prefix_weight", types.number, 0.1, memory);
  add_optional(function, "prefix_threshold", types.number, 0.7, memory);
  add_optional(function, "prefix_cap", types.integer, std::int64_t{4}, memory);
}

void add_alignment_options(
    mgp_func* function,
    mgp_memory* memory,
    const Types& types) {
  add_optional(function, "match_score", types.integer, std::int64_t{2}, memory);
  add_optional(function, "mismatch_score", types.integer, std::int64_t{-1}, memory);
  add_optional(function, "gap_open_score", types.integer, std::int64_t{-1}, memory);
  add_optional(
      function, "gap_extend_score", types.integer,
      kInheritedGapExtend, memory);
}

void add_alignment_function(
    mgp_module* module,
    mgp_memory* memory,
    const Types& types,
    const char* name,
    mgp_func_cb callback) {
  mgp_func* function = add_function(module, name, callback);
  add_required(function, "query", types.string);
  add_required(function, "target", types.string);
  add_alignment_options(function, memory, types);
}

void add_dynamic_options(
    mgp_func* function,
    mgp_memory* memory,
    const Types& types) {
  add_alignment_options(function, memory, types);
  add_optional(function, "prefix_weight", types.number, 0.1, memory);
  add_optional(function, "prefix_threshold", types.number, 0.7, memory);
  add_optional(function, "prefix_cap", types.integer, std::int64_t{4}, memory);
}

void add_ngram_function(
    mgp_module* module,
    mgp_memory* memory,
    const Types& types,
    const char* name,
    mgp_func_cb callback) {
  mgp_func* function = add_function(module, name, callback);
  add_required(function, "query", types.string);
  add_required(function, "target", types.string);
  add_optional(function, "n", types.integer, std::int64_t{2}, memory);
}

void add_unary_string_function(
    mgp_module* module,
    const Types& types,
    const char* name,
    mgp_func_cb callback) {
  mgp_func* function = add_function(module, name, callback);
  add_required(function, "text", types.string);
}

template <bool Local>
void add_path_functions(
    mgp_module* module,
    mgp_memory* memory,
    const Types& types) {
  if constexpr (Local) {
    for (const char* name : {
             "smith_waterman_path", "smith_waterman_path_info"}) {
      add_alignment_function(
          module, memory, types, name, alignment_path<true, false>);
    }
    for (const char* name : {
             "smith_waterman_cigar", "smith_waterman_trace_cigar",
             "smith_waterman_trade_cigar"}) {
      add_alignment_function(
          module, memory, types, name, alignment_path<true, true>);
    }
  } else {
    for (const char* name : {
             "needleman_wunsch_path", "needleman_wunsch_path_info"}) {
      add_alignment_function(
          module, memory, types, name, alignment_path<false, false>);
    }
    for (const char* name : {
             "needleman_wunsch_cigar", "needleman_wunsch_trace_cigar",
             "needleman_wunsch_trade_cigar"}) {
      add_alignment_function(
          module, memory, types, name, alignment_path<false, true>);
    }
  }
}

void register_scalar_functions(
    mgp_module* module,
    mgp_memory* memory,
    const Types& types) {
  add_pair_function(
      module, types, "levenshtein", integer_score<Scorer::levenshtein>);
  add_pair_function(
      module, types, "levenshtein_similarity",
      real_score<Scorer::levenshtein_normalized>);
  add_pair_function(
      module, types, "osa", integer_score<Scorer::damerau_levenshtein>);
  add_pair_function(
      module, types, "osa_similarity",
      real_score<Scorer::damerau_levenshtein_normalized>);
  add_pair_function(
      module, types, "true_damerau_levenshtein",
      integer_score<Scorer::true_damerau_levenshtein>);
  add_pair_function(
      module, types, "true_damerau_levenshtein_similarity",
      real_score<Scorer::true_damerau_levenshtein_normalized>);
  add_pair_function(
      module, types, "indel", integer_score<Scorer::indel>);
  add_pair_function(
      module, types, "indel_similarity",
      real_score<Scorer::indel_normalized>);
  add_pair_function(
      module, types, "hamming", integer_score<Scorer::hamming>);
  add_pair_function(
      module, types, "hamming_similarity",
      real_score<Scorer::hamming_normalized>);
  add_pair_function(module, types, "jaro", real_score<Scorer::jaro>);
  add_jaro_winkler_function(module, memory, types, "jaro_winkler");

  add_cutoff_function(
      module, memory, types, "levenshtein_score",
      cutoff_score<false, false>, false);
  add_cutoff_function(
      module, memory, types, "levenshtein_normalized_score",
      cutoff_score<false, true>, true);
  add_pair_function(
      module, types, "damerau_levenshtein_score",
      integer_score<Scorer::damerau_levenshtein>);
  add_pair_function(
      module, types, "damerau_levenshtein_normalized_score",
      real_score<Scorer::damerau_levenshtein_normalized>);
  add_pair_function(
      module, types, "true_damerau_levenshtein_score",
      integer_score<Scorer::true_damerau_levenshtein>);
  add_pair_function(
      module, types, "true_damerau_levenshtein_normalized_score",
      real_score<Scorer::true_damerau_levenshtein_normalized>);
  add_cutoff_function(
      module, memory, types, "indel_score",
      cutoff_score<true, false>, false);
  add_cutoff_function(
      module, memory, types, "indel_normalized_score",
      cutoff_score<true, true>, true);
  add_pair_function(
      module, types, "hamming_score", integer_score<Scorer::hamming>);
  add_pair_function(
      module, types, "hamming_normalized_score",
      real_score<Scorer::hamming_normalized>);
  add_pair_function(module, types, "jaro_similarity", real_score<Scorer::jaro>);
  add_jaro_winkler_function(module, memory, types, "jaro_winkler_similarity");

  for (const char* name : {
           "smith_waterman", "smith_waterman_affine",
           "smith_waterman_score", "smith_waterman_farrar_score"}) {
    add_alignment_function(
        module, memory, types, name, alignment_score<true, false>);
  }
  for (const char* name : {
           "smith_waterman_normalized_score",
           "smith_waterman_farrar_normalized_score"}) {
    add_alignment_function(
        module, memory, types, name, alignment_score<true, true>);
  }
  for (const char* name : {
           "needleman_wunsch", "needleman_wunsch_affine",
           "needleman_wunsch_score"}) {
    add_alignment_function(
        module, memory, types, name, alignment_score<false, false>);
  }
  add_alignment_function(
      module, memory, types, "needleman_wunsch_normalized_score",
      alignment_score<false, true>);

  mgp_func* score_function = add_function(module, "score", dynamic_score);
  add_required(score_function, "query", types.string);
  add_required(score_function, "target", types.string);
  add_required(score_function, "scorer", types.string);
  add_dynamic_options(score_function, memory, types);

  static_cast<void>(add_function(module, "simd_level", simd_level));
  static_cast<void>(add_function(module, "detect_best_backend", simd_level));
  static_cast<void>(add_function(module, "version", version));
  mgp_func* available = add_function(
      module, "backend_is_available", backend_is_available);
  add_required(available, "backend", types.string);
}

void register_algorithm_functions(
    mgp_module* module,
    mgp_memory* memory,
    const Types& types) {
  add_pair_function(
      module, types, "lcs_length", lcs_function<LcsOperation::length>);
  add_pair_function(
      module, types, "lcs_substring_length",
      lcs_function<LcsOperation::substring_length>);
  add_pair_function(
      module, types, "lcs_substring", lcs_function<LcsOperation::substring>);

  add_ngram_function(
      module, memory, types, "jaccard", ngram_function<NgramMetric::jaccard>);
  add_ngram_function(
      module, memory, types, "dice", ngram_function<NgramMetric::dice>);
  add_ngram_function(
      module, memory, types, "cosine", ngram_function<NgramMetric::cosine>);
  add_ngram_function(
      module, memory, types, "overlap", ngram_function<NgramMetric::overlap>);

  add_pair_function(
      module, types, "ratcliff_obershelp_similarity",
      string_similarity_function<StringSimilarity::ratcliff>);
  add_pair_function(
      module, types, "partial_ratio",
      string_similarity_function<StringSimilarity::partial>);
  add_pair_function(
      module, types, "token_sort_ratio",
      string_similarity_function<StringSimilarity::token_sort>);
  add_pair_function(
      module, types, "token_set_ratio",
      string_similarity_function<StringSimilarity::token_set>);
  add_pair_function(
      module, types, "partial_token_sort_ratio",
      string_similarity_function<StringSimilarity::partial_token_sort>);
  add_pair_function(
      module, types, "partial_token_set_ratio",
      string_similarity_function<StringSimilarity::partial_token_set>);
  add_pair_function(
      module, types, "wratio",
      string_similarity_function<StringSimilarity::weighted>);
  add_pair_function(
      module, types, "WRatio",
      string_similarity_function<StringSimilarity::weighted>);

  mgp_func* monge = add_function(module, "monge_elkan", monge_elkan);
  add_required(monge, "query", types.string);
  add_required(monge, "target", types.string);
  add_optional(
      monge, "inner", types.string, std::string_view("jaro"), memory);
  add_optional(monge, "symmetric", types.boolean, false, memory);

  add_unary_string_function(
      module, types, "soundex", unary_string_function<Soundex>);
  add_pair_function(
      module, types, "soundex_equal", encoded_equal<Soundex>);
  add_unary_string_function(
      module, types, "nysiis", unary_string_function<Nysiis>);
  add_pair_function(
      module, types, "nysiis_equal", encoded_equal<Nysiis>);
  add_unary_string_function(
      module, types, "match_rating_codex",
      unary_string_function<MatchRatingCodex>);
  add_pair_function(
      module, types, "match_rating_compare", match_rating_compare);
  add_unary_string_function(
      module, types, "caverphone", unary_string_function<Caverphone>);
  add_unary_string_function(
      module, types, "cologne_phonetic", cologne_phonetic);

  mgp_func* metaphone_function = add_function(module, "metaphone", metaphone);
  add_required(metaphone_function, "text", types.string);
  add_optional(
      metaphone_function, "variant", types.integer, std::int64_t{0}, memory);
  mgp_func* metaphone_equal_function =
      add_function(module, "metaphone_equal", metaphone_equal);
  add_required(metaphone_equal_function, "query", types.string);
  add_required(metaphone_equal_function, "target", types.string);
  add_optional(
      metaphone_equal_function, "variant", types.integer,
      std::int64_t{0}, memory);

  mgp_func* daitch = add_function(module, "daitch_mokotoff", daitch_mokotoff);
  add_required(daitch, "text", types.string);
  add_optional(daitch, "branching", types.boolean, true, memory);
  add_optional(daitch, "folding", types.boolean, true, memory);

  mgp_func* double_metaphone_function =
      add_function(module, "double_metaphone", double_metaphone);
  add_required(double_metaphone_function, "text", types.string);
  add_optional(
      double_metaphone_function, "max_length", types.integer,
      std::int64_t{64}, memory);
  add_optional(
      double_metaphone_function, "variant", types.integer,
      std::int64_t{0}, memory);

  mgp_func* beider = add_function(module, "beider_morse", beider_morse);
  add_required(beider, "text", types.string);
  add_optional(beider, "rule_type", types.integer, std::int64_t{0}, memory);
  add_optional(beider, "concat", types.boolean, true, memory);
  add_optional(
      beider, "max_phonemes", types.integer, std::int64_t{20}, memory);

  mgp_func* dtw_function = add_function(module, "dtw", dtw);
  add_required(dtw_function, "query", types.numbers);
  add_required(dtw_function, "target", types.numbers);
  add_optional(dtw_function, "window", types.number, -1.0, memory);
  add_optional(
      dtw_function, "distance", types.string,
      std::string_view("l2_squared"), memory);
  add_optional(dtw_function, "score_cutoff", types.number, -1.0, memory);

  add_path_functions<true>(module, memory, types);
  add_path_functions<false>(module, memory, types);
}

void add_matrix_alignment_options(
    mgp_func* function,
    mgp_memory* memory,
    const Types& types) {
  add_optional(
      function, "gap_open_score", types.integer,
      kInheritedGapExtend, memory);
  add_optional(
      function, "gap_extend_score", types.integer,
      kInheritedGapExtend, memory);
}

void add_matrix_alignment_function(
    mgp_module* module,
    mgp_memory* memory,
    const Types& types,
    const char* name,
    mgp_func_cb callback) {
  mgp_func* function = add_function(module, name, callback);
  add_required(function, "query", types.string);
  add_required(function, "target", types.string);
  add_required(function, "matrix", types.string);
  add_matrix_alignment_options(function, memory, types);
}

template <bool Local>
void register_matrix_alignment_functions(
    mgp_module* module,
    mgp_memory* memory,
    const Types& types) {
  if constexpr (Local) {
    add_matrix_alignment_function(
        module, memory, types, "smith_waterman_matrix_score",
        matrix_alignment<true, false, false>);
    for (const char* name : {
             "smith_waterman_matrix_path",
             "smith_waterman_matrix_path_info"}) {
      add_matrix_alignment_function(
          module, memory, types, name,
          matrix_alignment<true, true, false>);
    }
    add_matrix_alignment_function(
        module, memory, types, "smith_waterman_matrix_cigar",
        matrix_alignment<true, true, true>);
  } else {
    add_matrix_alignment_function(
        module, memory, types, "needleman_wunsch_matrix_score",
        matrix_alignment<false, false, false>);
    for (const char* name : {
             "needleman_wunsch_matrix_path",
             "needleman_wunsch_matrix_path_info"}) {
      add_matrix_alignment_function(
          module, memory, types, name,
          matrix_alignment<false, true, false>);
    }
    add_matrix_alignment_function(
        module, memory, types, "needleman_wunsch_matrix_cigar",
        matrix_alignment<false, true, true>);
  }
}

void register_matrix_functions(
    mgp_module* module,
    mgp_memory* memory,
    const Types& types) {
  mgp_func* info = add_function(module, "matrix_info", matrix_info);
  add_required(info, "matrix", types.string);

  mgp_func* step =
      add_function(module, "matrix_score_step_limit", matrix_score_step_limit);
  add_required(step, "matrix", types.string);
  add_optional(
      step, "gap_open_score", types.integer,
      kInheritedGapExtend, memory);
  add_optional(
      step, "gap_extend_score", types.integer,
      kInheritedGapExtend, memory);

  mgp_func* substitution = add_function(
      module, "substitution_matrix_score", substitution_matrix_score);
  add_required(substitution, "matrix", types.string);
  add_required(substitution, "query", types.string);
  add_required(substitution, "target", types.string);

  register_matrix_alignment_functions<true>(module, memory, types);
  register_matrix_alignment_functions<false>(module, memory, types);
}

void register_cdist(
    mgp_module* module,
    mgp_memory* memory,
    const Types& types) {
  mgp_proc* procedure = nullptr;
  check_mgp(
      mgp_module_add_batch_read_procedure(
          module, "cdist", cdist, cdist_initialize, cdist_cleanup,
          &procedure),
      "mgp_module_add_batch_read_procedure");
  check_mgp(
      mgp_proc_add_arg(procedure, "queries", types.strings),
      "mgp_proc_add_arg");
  check_mgp(
      mgp_proc_add_arg(procedure, "targets", types.strings),
      "mgp_proc_add_arg");
  check_mgp(
      mgp_proc_add_arg(procedure, "scorer", types.string),
      "mgp_proc_add_arg");

  auto add_proc_optional = [&](const char* name, mgp_type* type, auto value) {
    OwnedValue default_value = make_value(value, memory);
    check_mgp(
        mgp_proc_add_opt_arg(procedure, name, type, default_value.get()),
        "mgp_proc_add_opt_arg");
  };
  add_proc_optional("match_score", types.integer, std::int64_t{2});
  add_proc_optional("mismatch_score", types.integer, std::int64_t{-1});
  add_proc_optional("gap_open_score", types.integer, std::int64_t{-1});
  add_proc_optional("gap_extend_score", types.integer, kInheritedGapExtend);
  add_proc_optional("prefix_weight", types.number, 0.1);
  add_proc_optional("prefix_threshold", types.number, 0.7);
  add_proc_optional("prefix_cap", types.integer, std::int64_t{4});
  add_proc_optional(
      "chunk_size", types.integer, kDefaultCdistChunkSize);

  check_mgp(
      mgp_proc_add_result(procedure, "query_index", types.integer),
      "mgp_proc_add_result");
  check_mgp(
      mgp_proc_add_result(procedure, "target_index", types.integer),
      "mgp_proc_add_result");
  check_mgp(
      mgp_proc_add_result(procedure, "score", types.number),
      "mgp_proc_add_result");
}

}  // namespace

#if defined(__GNUC__) || defined(__clang__)
#define STRIDE_ALIGN_MGP_EXPORT __attribute__((visibility("default")))
#else
#define STRIDE_ALIGN_MGP_EXPORT
#endif

extern "C" STRIDE_ALIGN_MGP_EXPORT int mgp_init_module(
    mgp_module* module,
    mgp_memory* memory) {
  try {
    const Types types;
    register_scalar_functions(module, memory, types);
    register_algorithm_functions(module, memory, types);
    register_matrix_functions(module, memory, types);
    register_cdist(module, memory, types);
    return 0;
  } catch (const std::exception& error) {
    static_cast<void>(
        mgp_log(mgp_log_level::MGP_LOG_LEVEL_ERROR, error.what()));
    return 1;
  } catch (...) {
    static_cast<void>(mgp_log(
        mgp_log_level::MGP_LOG_LEVEL_ERROR,
        "unknown error while loading stride-align"));
    return 1;
  }
}

extern "C" STRIDE_ALIGN_MGP_EXPORT int mgp_shutdown_module() {
  cdist_cleanup();
  return 0;
}
