//go:build !stridealign_prebuilt

#include "bridge.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "embedded_data.hpp"
#include "stride_align/batch.hpp"
#include "stride_align/beider_morse.hpp"
#include "stride_align/caverphone.hpp"
#include "stride_align/cologne_phonetic.hpp"
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

namespace {

namespace batch = ::stride_align::batch;
namespace core = ::stride_align::core;
namespace utf8 = ::stride_align::utf8;

using Score = ::stride_align::Score;
using Scorer = batch::Scorer;

constexpr std::string_view kVersion = "0.6.0";

std::string_view string_view(stride_string input) {
  if (input.data == nullptr) {
    if (input.size != 0U) {
      throw std::invalid_argument("string data is null with a non-zero size");
    }
    return {};
  }
  return {input.data, input.size};
}

void clear(stride_bytes* value) noexcept {
  if (value != nullptr) *value = {};
}

stride_bytes copy_bytes(std::string_view input) {
  stride_bytes output{};
  if (input.empty()) return output;
  output.data = static_cast<char*>(std::malloc(input.size()));
  if (output.data == nullptr) throw std::bad_alloc();
  std::memcpy(output.data, input.data(), input.size());
  output.size = input.size();
  return output;
}

template <typename Value>
Value* copy_values(std::span<const Value> input) {
  if (input.empty()) return nullptr;
  if (input.size() > std::numeric_limits<std::size_t>::max() / sizeof(Value)) {
    throw std::length_error("native result allocation overflow");
  }
  auto* output = static_cast<Value*>(std::malloc(input.size_bytes()));
  if (output == nullptr) throw std::bad_alloc();
  std::memcpy(output, input.data(), input.size_bytes());
  return output;
}

void set_error(stride_bytes* error, std::string_view message) noexcept {
  if (error == nullptr) return;
  try {
    *error = copy_bytes(message);
  } catch (...) {
    *error = {};
  }
}

template <typename Function>
int guarded(stride_bytes* error, Function&& function) noexcept {
  clear(error);
  try {
    function();
    return 0;
  } catch (const std::exception& exception) {
    set_error(error, exception.what());
  } catch (...) {
    set_error(error, "unknown stride-align C++ exception");
  }
  return 1;
}

Scorer scorer_value(int input) {
  if (input < static_cast<int>(Scorer::levenshtein) ||
      input > static_cast<int>(Scorer::needleman_wunsch_normalized)) {
    throw std::invalid_argument("unknown stride-align scorer value");
  }
  return static_cast<Scorer>(input);
}

batch::ScoreOptions score_options(stride_score_options input) {
  if (!std::isfinite(input.prefix_weight) || input.prefix_weight < 0.0 ||
      !std::isfinite(input.prefix_threshold) ||
      input.prefix_threshold < 0.0 || input.prefix_threshold > 1.0) {
    throw std::invalid_argument("invalid scorer prefix parameters");
  }
  return {
      input.match_score,
      input.mismatch_score,
      input.gap_open_score,
      input.gap_extend_score,
      input.prefix_weight,
      input.prefix_threshold,
      input.prefix_cap,
  };
}

std::vector<std::string_view> string_views(stride_strings input) {
  if (input.count == 0U) {
    if (input.data_size != 0U) {
      throw std::invalid_argument("empty string collection has non-empty data");
    }
    return {};
  }
  if (input.offsets == nullptr) {
    throw std::invalid_argument("string collection offsets are null");
  }
  if (input.data == nullptr && input.data_size != 0U) {
    throw std::invalid_argument("string collection data is null");
  }
  if (input.offsets[0] != 0U || input.offsets[input.count] != input.data_size) {
    throw std::invalid_argument("string collection offsets do not cover its data");
  }
  const char* base = input.data == nullptr ? "" : input.data;
  std::vector<std::string_view> output;
  output.reserve(input.count);
  for (std::size_t index = 0; index < input.count; ++index) {
    const std::size_t begin = input.offsets[index];
    const std::size_t end = input.offsets[index + 1U];
    if (begin > end || end > input.data_size) {
      throw std::invalid_argument("string collection offsets are not monotonic");
    }
    output.emplace_back(base + begin, end - begin);
  }
  return output;
}

std::vector<std::optional<batch::Text>> texts(stride_strings input) {
  const auto views = string_views(input);
  std::vector<std::optional<batch::Text>> output;
  output.reserve(views.size());
  for (const std::string_view value : views) {
    output.emplace_back(batch::Text(std::string(value)));
  }
  return output;
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

double string_similarity(
    int operation,
    const std::vector<std::uint32_t>& left,
    const std::vector<std::uint32_t>& right) {
  const auto left_span = std::span<const std::uint32_t>(left);
  const auto right_span = std::span<const std::uint32_t>(right);
  switch (operation) {
    case STRIDE_PAIR_RATCLIFF_OBERSHELP:
      return ::stride_align::ratcliff_obershelp::
          ratcliff_obershelp_similarity(left, right);
    case STRIDE_PAIR_PARTIAL_RATIO:
      return ::stride_align::partial_ratio::partial_ratio(left, right);
    case STRIDE_PAIR_TOKEN_SORT_RATIO:
      return ::stride_align::token_ratios::token_sort_ratio(left, right);
    case STRIDE_PAIR_TOKEN_SET_RATIO:
      return ::stride_align::token_ratios::token_set_ratio(left, right);
    case STRIDE_PAIR_PARTIAL_TOKEN_SORT_RATIO:
      return ::stride_align::wratio::
          partial_token_sort_ratio_engine<std::uint32_t>(left_span, right_span);
    case STRIDE_PAIR_PARTIAL_TOKEN_SET_RATIO:
      return ::stride_align::wratio::
          partial_token_set_ratio_engine<std::uint32_t>(left_span, right_span);
    case STRIDE_PAIR_WRATIO:
      return ::stride_align::wratio::native_wratio(left, right);
    default:
      throw std::invalid_argument("unknown string-similarity operation");
  }
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
  const auto pair = utf8::prepare_pair(encode_utf8(left), encode_utf8(right));
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

double monge_elkan(
    const std::vector<std::uint32_t>& left_points,
    const std::vector<std::uint32_t>& right_points,
    std::string_view inner,
    bool symmetric) {
  const auto left = word_tokens(left_points);
  const auto right = word_tokens(right_points);
  if (left.empty() && right.empty()) return 1.0;
  if (left.empty() || right.empty()) return 0.0;
  const double forward = monge_direction(left, right, inner);
  return symmetric
      ? (forward + monge_direction(right, left, inner)) / 2.0
      : forward;
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

void copy_path(
    const ::stride_align::AlignmentPath& source,
    std::string_view query_text,
    std::string_view target_text,
    stride_alignment_path* output) {
  if (output == nullptr) throw std::invalid_argument("path output is null");
  *output = {};
  const auto query = utf8::prepare_streaming(query_text);
  const auto target = utf8::prepare_streaming(target_text);
  const auto [aligned_query, aligned_target] =
      aligned_strings(source, query, target);
  output->score = source.score;
  output->query_start = source.query_start;
  output->query_end = source.query_end;
  output->target_start = source.target_start;
  output->target_end = source.target_end;
  output->operations = copy_bytes(source.operations);
  try {
    output->cigar = copy_bytes(source.cigar);
    output->aligned_query = copy_bytes(aligned_query);
    output->aligned_target = copy_bytes(aligned_target);
  } catch (...) {
    stride_go_free_path(output);
    throw;
  }
  output->matches = source.matches;
  output->mismatches = source.mismatches;
  output->insertions = source.insertions;
  output->deletions = source.deletions;
  output->aligned_length = source.aligned_length;
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

Matrix finish_matrix(Matrix matrix) {
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
  if (matrix.wildcard >= alphabet.size()) {
    throw std::invalid_argument(
        "substitution-matrix wildcard index is outside the alphabet");
  }
  if (alphabet.size() >
      std::numeric_limits<std::size_t>::max() / alphabet.size() ||
      matrix.values.size() != alphabet.size() * alphabet.size()) {
    throw std::invalid_argument(
        "substitution-matrix score grid does not match the alphabet");
  }
  return matrix;
}

Matrix matrix_from_record(
    ::stride_align_go_embedded::MatrixRecord record) {
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
  const auto wildcard = utf8::prepare_streaming(record.wildcard);
  const auto alphabet = utf8::prepare_streaming(matrix.alphabet);
  const auto found = std::find(alphabet.begin(), alphabet.end(), wildcard.at(0));
  if (found == alphabet.end()) {
    throw std::logic_error("embedded matrix wildcard is absent");
  }
  matrix.wildcard = static_cast<std::uint16_t>(found - alphabet.begin());
  return finish_matrix(std::move(matrix));
}

std::vector<Matrix> build_matrices() {
  std::vector<Matrix> output;
  for (auto& record : ::stride_align_go_embedded::matrices()) {
    output.push_back(matrix_from_record(std::move(record)));
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

Matrix matrix_from_view(stride_matrix_view input) {
  Matrix matrix;
  matrix.name = std::string(string_view(input.name));
  matrix.sql_name = matrix.name;
  matrix.alphabet = std::string(string_view(input.alphabet));
  matrix.wildcard = static_cast<std::uint16_t>(input.wildcard_index);
  matrix.gap_score = input.gap_score;
  matrix.gap_open = input.gap_open_score;
  matrix.gap_extend = input.gap_extend_score;
  matrix.has_affine = input.has_affine != 0;
  if (input.values == nullptr && input.values_size != 0U) {
    throw std::invalid_argument("substitution-matrix values are null");
  }
  if (input.values_size != 0U) {
    matrix.values.assign(input.values, input.values + input.values_size);
  }
  return finish_matrix(std::move(matrix));
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

stride_string_list copy_string_list(
    const std::vector<std::string_view>& input) {
  stride_string_list output{};
  if (input.size() == std::numeric_limits<std::size_t>::max()) {
    throw std::length_error("string-list offset allocation overflow");
  }
  std::vector<std::size_t> offsets;
  offsets.reserve(input.size() + 1U);
  offsets.push_back(0U);
  std::string flattened;
  for (const std::string_view value : input) {
    if (value.size() > std::numeric_limits<std::size_t>::max() - flattened.size()) {
      throw std::length_error("string-list data allocation overflow");
    }
    flattened.append(value);
    offsets.push_back(flattened.size());
  }
  output.data_size = flattened.size();
  output.count = input.size();
  if (!flattened.empty()) {
    output.data = copy_bytes(flattened).data;
  }
  try {
    output.offsets = copy_values<std::size_t>(offsets);
  } catch (...) {
    std::free(output.data);
    throw;
  }
  return output;
}

double pair_number_value(
    int operation,
    std::string_view query,
    std::string_view target,
    std::size_t integer_option,
    std::string_view string_option,
    bool bool_option) {
  const auto left = utf8::prepare_streaming(query);
  const auto right = utf8::prepare_streaming(target);
  switch (operation) {
    case STRIDE_PAIR_LCS_LENGTH:
      return static_cast<double>(::stride_align::lcs::lcs_length(left, right));
    case STRIDE_PAIR_LCS_SUBSTRING_LENGTH:
      return static_cast<double>(
          ::stride_align::lcs::lcs_substring_length(left, right));
    case STRIDE_PAIR_JACCARD:
      return ::stride_align::ngram::jaccard(left, right, integer_option);
    case STRIDE_PAIR_DICE:
      return ::stride_align::ngram::dice(left, right, integer_option);
    case STRIDE_PAIR_COSINE:
      return ::stride_align::ngram::cosine(left, right, integer_option);
    case STRIDE_PAIR_OVERLAP:
      return ::stride_align::ngram::overlap(left, right, integer_option);
    case STRIDE_PAIR_MONGE_ELKAN:
      return monge_elkan(left, right, string_option, bool_option);
    default:
      return string_similarity(operation, left, right);
  }
}

bool pair_bool_value(
    int operation,
    std::string_view left,
    std::string_view right,
    int variant) {
  switch (operation) {
    case STRIDE_PAIR_SOUNDEX_EQUAL: {
      const std::string a = ::stride_align::phonetic::soundex(left);
      const std::string b = ::stride_align::phonetic::soundex(right);
      return !a.empty() && a == b;
    }
    case STRIDE_PAIR_METAPHONE_EQUAL: {
      if (variant < 0 || variant > 1) {
        throw std::invalid_argument("metaphone variant must be 0 or 1");
      }
      const auto selected =
          static_cast<::stride_align::phonetic::MetaphoneVariant>(variant);
      const std::string a = ::stride_align::phonetic::metaphone(left, selected);
      const std::string b = ::stride_align::phonetic::metaphone(right, selected);
      return !a.empty() && a == b;
    }
    case STRIDE_PAIR_NYSIIS_EQUAL: {
      const std::string a = ::stride_align::phonetic::nysiis(left);
      const std::string b = ::stride_align::phonetic::nysiis(right);
      return !a.empty() && a == b;
    }
    case STRIDE_PAIR_MATCH_RATING_COMPARE:
      return ::stride_align::phonetic::match_rating_compare(left, right);
    default:
      throw std::invalid_argument("unknown pair-bool operation");
  }
}

std::string unary_string_value(
    int operation,
    std::string_view value,
    std::int64_t integer_option_a,
    std::int64_t integer_option_b,
    bool bool_option_a,
    bool bool_option_b) {
  switch (operation) {
    case STRIDE_UNARY_SOUNDEX:
      return ::stride_align::phonetic::soundex(value);
    case STRIDE_UNARY_METAPHONE:
      if (integer_option_a < 0 || integer_option_a > 1) {
        throw std::invalid_argument("metaphone variant must be 0 or 1");
      }
      return ::stride_align::phonetic::metaphone(
          value, static_cast<::stride_align::phonetic::MetaphoneVariant>(
                     integer_option_a));
    case STRIDE_UNARY_NYSIIS:
      return ::stride_align::phonetic::nysiis(value);
    case STRIDE_UNARY_MATCH_RATING_CODEX:
      return ::stride_align::phonetic::match_rating_codex(value);
    case STRIDE_UNARY_CAVERPHONE:
      return ::stride_align::phonetic::caverphone(value);
    case STRIDE_UNARY_COLOGNE_PHONETIC:
      return ::stride_align::phonetic::cologne_phonetic(
          utf8::prepare_streaming(value));
    case STRIDE_UNARY_DAITCH_MOKOTOFF:
      return ::stride_align::phonetic::daitch_mokotoff(
          utf8::prepare_streaming(value), bool_option_a, bool_option_b);
    case STRIDE_UNARY_BEIDER_MORSE: {
      if (integer_option_a < 0 || integer_option_a > 1) {
        throw std::invalid_argument("Beider-Morse rule type must be 0 or 1");
      }
      if (integer_option_b < 0) {
        throw std::invalid_argument("max_phonemes must be non-negative");
      }
      static const bool registered = [] {
        ::stride_align::phonetic::bmpm_register_resources(
            ::stride_align_go_embedded::bmpm_resources());
        return true;
      }();
      static_cast<void>(registered);
      return ::stride_align::phonetic::beider_morse(
          utf8::prepare_streaming(value),
          static_cast<::stride_align::phonetic::BmpmRuleType>(integer_option_a),
          bool_option_a, static_cast<std::size_t>(integer_option_b));
    }
    default:
      throw std::invalid_argument("unknown unary-string operation");
  }
}

double dtw_value(
    std::span<const double> query,
    std::span<const double> target,
    double window,
    int distance_kind,
    double score_cutoff) {
  if (query.empty() || target.empty()) {
    throw std::invalid_argument("DTW inputs must not be empty");
  }
  if (distance_kind < 0 || distance_kind > 1) {
    throw std::invalid_argument("unknown DTW distance kind");
  }
  std::optional<double> cutoff;
  if (score_cutoff != -1.0) {
    if (!std::isfinite(score_cutoff) || score_cutoff < 0.0) {
      throw std::invalid_argument(
          "score_cutoff must be -1 or non-negative and finite");
    }
    cutoff = score_cutoff;
  }
  return ::stride_align::dtw::dtw_score_scalar<double, double>(
      query, target,
      distance_kind == 0
          ? ::stride_align::dtw::DistanceKind::kL1
          : ::stride_align::dtw::DistanceKind::kL2Squared,
      dtw_window(window, query.size(), target.size()), cutoff);
}

}  // namespace

extern "C" {

const char* stride_go_version(void) {
  return kVersion.data();
}

const char* stride_go_simd_level(void) {
  return ::stride_align_go::simd_level().data();
}

void stride_go_free(void* data) {
  std::free(data);
}

void stride_go_free_path(stride_alignment_path* path) {
  if (path == nullptr) return;
  std::free(path->operations.data);
  std::free(path->cigar.data);
  std::free(path->aligned_query.data);
  std::free(path->aligned_target.data);
  *path = {};
}

void stride_go_free_matrix(stride_matrix_data* matrix) {
  if (matrix == nullptr) return;
  std::free(matrix->name.data);
  std::free(matrix->alphabet.data);
  std::free(matrix->values);
  *matrix = {};
}

int stride_go_score(
    int scorer,
    stride_string query,
    stride_string target,
    stride_score_options options,
    double* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) throw std::invalid_argument("score output is null");
    *output = batch::score(
        scorer_value(scorer), string_view(query), string_view(target),
        score_options(options));
  });
}

int stride_go_scores(
    int scorer,
    stride_string query,
    stride_strings targets_input,
    stride_score_options options,
    stride_doubles* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) throw std::invalid_argument("scores output is null");
    *output = {};
    const batch::Text query_text(std::string(string_view(query)));
    const auto target_texts = texts(targets_input);
    const auto values = batch::scores(
        query_text, target_texts, scorer_value(scorer), score_options(options));
    std::vector<double> flattened;
    flattened.reserve(values.size());
    for (const auto& value : values) flattened.push_back(value.value());
    output->data = copy_values<double>(flattened);
    output->size = flattened.size();
  });
}

int stride_go_top_k(
    int scorer,
    stride_string query,
    stride_strings targets_input,
    size_t k,
    stride_score_options options,
    int skip_invalid_hamming,
    stride_ranked_matches* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) throw std::invalid_argument("top-k output is null");
    *output = {};
    const batch::Text query_text(std::string(string_view(query)));
    const auto target_texts = texts(targets_input);
    const auto values = batch::top_k(
        query_text, target_texts, scorer_value(scorer), k,
        score_options(options), skip_invalid_hamming != 0);
    std::vector<stride_ranked_match> flattened;
    flattened.reserve(values.size());
    for (const auto& value : values) {
      flattened.push_back({value.score, value.index});
    }
    output->data = copy_values<stride_ranked_match>(flattened);
    output->size = flattened.size();
  });
}

int stride_go_cdist(
    int scorer,
    stride_strings queries_input,
    stride_strings targets_input,
    stride_score_options options,
    stride_doubles* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) throw std::invalid_argument("cdist output is null");
    *output = {};
    const auto queries = texts(queries_input);
    const auto targets = texts(targets_input);
    const auto values = batch::cdist(
        queries, targets, scorer_value(scorer), score_options(options));
    std::vector<double> flattened;
    if (!queries.empty() &&
        targets.size() > std::numeric_limits<std::size_t>::max() / queries.size()) {
      throw std::length_error("cdist result size overflow");
    }
    flattened.reserve(queries.size() * targets.size());
    for (const auto& row : values) {
      for (const auto& value : row) flattened.push_back(value.value());
    }
    output->data = copy_values<double>(flattened);
    output->size = flattened.size();
  });
}

int stride_go_cdist_above_threshold(
    int scorer,
    stride_strings queries_input,
    stride_strings targets_input,
    double threshold,
    stride_score_options options,
    stride_matrix_matches* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) {
      throw std::invalid_argument("threshold cdist output is null");
    }
    *output = {};
    const auto queries = texts(queries_input);
    const auto targets = texts(targets_input);
    const auto values = batch::cdist_above_threshold(
        queries, targets, scorer_value(scorer), threshold,
        score_options(options));
    std::vector<stride_matrix_match> flattened;
    flattened.reserve(values.size());
    for (const auto& value : values) {
      flattened.push_back(
          {value.score, value.query_index, value.target_index});
    }
    output->data = copy_values<stride_matrix_match>(flattened);
    output->size = flattened.size();
  });
}

int stride_go_cdist_top_k(
    int scorer,
    stride_strings queries_input,
    stride_strings targets_input,
    size_t k,
    int reject_duplicates,
    stride_score_options options,
    stride_matrix_matches* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) throw std::invalid_argument("cdist top-k output is null");
    *output = {};
    const auto queries = texts(queries_input);
    const auto targets = texts(targets_input);
    const auto values = batch::cdist_top_k(
        queries, targets, scorer_value(scorer), k,
        reject_duplicates != 0, score_options(options));
    std::vector<stride_matrix_match> flattened;
    flattened.reserve(values.size());
    for (const auto& value : values) {
      flattened.push_back(
          {value.score, value.query_index, value.target_index});
    }
    output->data = copy_values<stride_matrix_match>(flattened);
    output->size = flattened.size();
  });
}

int stride_go_cdist_top_k_per_query(
    int scorer,
    stride_strings queries_input,
    stride_strings targets_input,
    size_t k,
    stride_score_options options,
    stride_grouped_matches* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) {
      throw std::invalid_argument("per-query top-k output is null");
    }
    *output = {};
    const auto queries = texts(queries_input);
    const auto targets = texts(targets_input);
    const auto groups = batch::cdist_top_k_per_query(
        queries, targets, scorer_value(scorer), k, score_options(options));
    std::vector<stride_ranked_match> flattened;
    std::vector<std::size_t> offsets;
    offsets.reserve(groups.size() + 1U);
    offsets.push_back(0U);
    for (const auto& group : groups) {
      for (const auto& value : group) {
        flattened.push_back({value.score, value.index});
      }
      offsets.push_back(flattened.size());
    }
    output->data = copy_values<stride_ranked_match>(flattened);
    try {
      output->offsets = copy_values<std::size_t>(offsets);
    } catch (...) {
      std::free(output->data);
      throw;
    }
    output->data_size = flattened.size();
    output->query_count = groups.size();
  });
}

int stride_go_pair_number(
    int operation,
    stride_string query,
    stride_string target,
    size_t integer_option,
    stride_string string_option,
    int bool_option,
    double* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) {
      throw std::invalid_argument("pair-number output is null");
    }
    *output = pair_number_value(
        operation, string_view(query), string_view(target), integer_option,
        string_view(string_option), bool_option != 0);
  });
}

int stride_go_pair_numbers(
    int operation,
    stride_string query,
    stride_strings targets_input,
    size_t integer_option,
    stride_string string_option,
    int bool_option,
    stride_doubles* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) {
      throw std::invalid_argument("pair-numbers output is null");
    }
    *output = {};
    const std::string_view query_value = string_view(query);
    const std::string_view option_value = string_view(string_option);
    const auto targets = string_views(targets_input);
    std::vector<double> values;
    values.reserve(targets.size());
    for (const std::string_view target : targets) {
      values.push_back(pair_number_value(
          operation, query_value, target, integer_option, option_value,
          bool_option != 0));
    }
    output->data = copy_values<double>(values);
    output->size = values.size();
  });
}

int stride_go_lcs_substring(
    stride_string query,
    stride_string target,
    stride_bytes* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) throw std::invalid_argument("LCS output is null");
    *output = {};
    const auto left = utf8::prepare_streaming(string_view(query));
    const auto right = utf8::prepare_streaming(string_view(target));
    *output = copy_bytes(encode_utf8(
        ::stride_align::lcs::lcs_substring(left, right)));
  });
}

int stride_go_lcs_substrings(
    stride_string query,
    stride_strings targets_input,
    stride_string_list* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) {
      throw std::invalid_argument("LCS string-list output is null");
    }
    *output = {};
    const auto left = utf8::prepare_streaming(string_view(query));
    const auto targets = string_views(targets_input);
    std::vector<std::string> values;
    values.reserve(targets.size());
    for (const std::string_view target : targets) {
      const auto right = utf8::prepare_streaming(target);
      values.push_back(encode_utf8(
          ::stride_align::lcs::lcs_substring(left, right)));
    }
    std::vector<std::string_view> views;
    views.reserve(values.size());
    for (const std::string& value : values) views.push_back(value);
    *output = copy_string_list(views);
  });
}

int stride_go_pair_bool(
    int operation,
    stride_string query,
    stride_string target,
    int variant,
    int* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) throw std::invalid_argument("pair-bool output is null");
    *output = pair_bool_value(
        operation, string_view(query), string_view(target), variant) ? 1 : 0;
  });
}

int stride_go_pair_bools(
    int operation,
    stride_string query,
    stride_strings targets_input,
    int variant,
    stride_uints* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) {
      throw std::invalid_argument("pair-bools output is null");
    }
    *output = {};
    const std::string_view query_value = string_view(query);
    const auto targets = string_views(targets_input);
    std::vector<std::uint64_t> values;
    values.reserve(targets.size());
    for (const std::string_view target : targets) {
      values.push_back(pair_bool_value(
          operation, query_value, target, variant) ? 1U : 0U);
    }
    output->data = copy_values<std::uint64_t>(values);
    output->size = values.size();
  });
}

int stride_go_unary_string(
    int operation,
    stride_string input,
    int64_t integer_option_a,
    int64_t integer_option_b,
    int bool_option_a,
    int bool_option_b,
    stride_bytes* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) throw std::invalid_argument("string output is null");
    *output = {};
    *output = copy_bytes(unary_string_value(
        operation, string_view(input), integer_option_a, integer_option_b,
        bool_option_a != 0, bool_option_b != 0));
  });
}

int stride_go_unary_strings(
    int operation,
    stride_strings inputs,
    int64_t integer_option_a,
    int64_t integer_option_b,
    int bool_option_a,
    int bool_option_b,
    stride_string_list* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) {
      throw std::invalid_argument("string-list output is null");
    }
    *output = {};
    const auto input_views = string_views(inputs);
    std::vector<std::string> values;
    values.reserve(input_views.size());
    for (const std::string_view input : input_views) {
      values.push_back(unary_string_value(
          operation, input, integer_option_a, integer_option_b,
          bool_option_a != 0, bool_option_b != 0));
    }
    std::vector<std::string_view> views;
    views.reserve(values.size());
    for (const std::string& value : values) views.push_back(value);
    *output = copy_string_list(views);
  });
}

int stride_go_double_metaphone(
    stride_string input,
    size_t maximum_length,
    int variant,
    stride_bytes* primary,
    stride_bytes* alternate,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (primary == nullptr || alternate == nullptr) {
      throw std::invalid_argument("double-metaphone output is null");
    }
    *primary = {};
    *alternate = {};
    if (variant < 0 || variant > 1) {
      throw std::invalid_argument("double-metaphone variant must be 0 or 1");
    }
    const auto result = ::stride_align::phonetic::double_metaphone(
        string_view(input), maximum_length,
        static_cast<::stride_align::phonetic::DoubleMetaphoneVariant>(variant));
    *primary = copy_bytes(result.primary);
    try {
      *alternate = copy_bytes(result.alternate);
    } catch (...) {
      std::free(primary->data);
      *primary = {};
      throw;
    }
  });
}

int stride_go_double_metaphones(
    stride_strings inputs,
    size_t maximum_length,
    int variant,
    stride_string_list* primaries,
    stride_string_list* alternates,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (primaries == nullptr || alternates == nullptr) {
      throw std::invalid_argument("double-metaphone list output is null");
    }
    *primaries = {};
    *alternates = {};
    if (variant < 0 || variant > 1) {
      throw std::invalid_argument("double-metaphone variant must be 0 or 1");
    }
    const auto input_views = string_views(inputs);
    std::vector<std::string> primary_values;
    std::vector<std::string> alternate_values;
    primary_values.reserve(input_views.size());
    alternate_values.reserve(input_views.size());
    const auto selected =
        static_cast<::stride_align::phonetic::DoubleMetaphoneVariant>(variant);
    for (const std::string_view input : input_views) {
      auto result = ::stride_align::phonetic::double_metaphone(
          input, maximum_length, selected);
      primary_values.push_back(std::move(result.primary));
      alternate_values.push_back(std::move(result.alternate));
    }
    std::vector<std::string_view> primary_views;
    std::vector<std::string_view> alternate_views;
    primary_views.reserve(primary_values.size());
    alternate_views.reserve(alternate_values.size());
    for (const std::string& value : primary_values) primary_views.push_back(value);
    for (const std::string& value : alternate_values) {
      alternate_views.push_back(value);
    }
    *primaries = copy_string_list(primary_views);
    try {
      *alternates = copy_string_list(alternate_views);
    } catch (...) {
      std::free(primaries->data);
      std::free(primaries->offsets);
      *primaries = {};
      throw;
    }
  });
}

int stride_go_dtw(
    const double* query,
    size_t query_size,
    const double* target,
    size_t target_size,
    double window,
    int distance_kind,
    double score_cutoff,
    double* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) throw std::invalid_argument("DTW output is null");
    if ((query == nullptr && query_size != 0U) ||
        (target == nullptr && target_size != 0U)) {
      throw std::invalid_argument("DTW input data is null");
    }
    *output = dtw_value(
        std::span<const double>(query, query_size),
        std::span<const double>(target, target_size),
        window, distance_kind, score_cutoff);
  });
}

int stride_go_dtw_distances(
    const double* query,
    size_t query_size,
    stride_double_sequences targets,
    double window,
    int distance_kind,
    double score_cutoff,
    stride_doubles* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) throw std::invalid_argument("DTW batch output is null");
    *output = {};
    if (query == nullptr && query_size != 0U) {
      throw std::invalid_argument("DTW query data is null");
    }
    if (targets.offsets == nullptr && targets.count != 0U) {
      throw std::invalid_argument("DTW target offsets are null");
    }
    if (targets.data == nullptr && targets.data_size != 0U) {
      throw std::invalid_argument("DTW target data is null");
    }
    if (targets.count != 0U &&
        (targets.offsets[0] != 0U ||
         targets.offsets[targets.count] != targets.data_size)) {
      throw std::invalid_argument("DTW target offsets do not cover their data");
    }
    const auto query_span = std::span<const double>(query, query_size);
    const double* base = targets.data == nullptr ? query : targets.data;
    std::vector<double> values;
    values.reserve(targets.count);
    for (std::size_t index = 0; index < targets.count; ++index) {
      const std::size_t begin = targets.offsets[index];
      const std::size_t end = targets.offsets[index + 1U];
      if (begin > end || end > targets.data_size) {
        throw std::invalid_argument("DTW target offsets are not monotonic");
      }
      values.push_back(dtw_value(
          query_span, std::span<const double>(base + begin, end - begin),
          window, distance_kind, score_cutoff));
    }
    output->data = copy_values<double>(values);
    output->size = values.size();
  });
}

int stride_go_alignment_path(
    int local,
    stride_string query_input,
    stride_string target_input,
    stride_score_options options_input,
    stride_alignment_path* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) throw std::invalid_argument("path output is null");
    *output = {};
    const std::string_view query = string_view(query_input);
    const std::string_view target = string_view(target_input);
    const auto options = score_options(options_input);
    const auto pair = utf8::prepare_pair(query, target);
    ::stride_align::AlignmentPath path;
    if (options.gap_open_score == options.gap_extend_score) {
      path = local != 0
          ? core::smith_waterman_path(
                pair, options.match_score, options.mismatch_score,
                options.gap_open_score)
          : core::needleman_wunsch_path(
                pair, options.match_score, options.mismatch_score,
                options.gap_open_score);
    } else {
      path = local != 0
          ? core::smith_waterman_affine_path(
                pair, options.match_score, options.mismatch_score,
                options.gap_open_score, options.gap_extend_score)
          : core::needleman_wunsch_affine_path(
                pair, options.match_score, options.mismatch_score,
                options.gap_open_score, options.gap_extend_score);
    }
    copy_path(path, query, target, output);
  });
}

int stride_go_matrix_available(
    stride_string_list* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) {
      throw std::invalid_argument("matrix catalog output is null");
    }
    *output = {};
    std::vector<std::string_view> names;
    names.reserve(matrices().size());
    for (const Matrix& matrix : matrices()) names.push_back(matrix.sql_name);
    *output = copy_string_list(names);
  });
}

int stride_go_matrix_get(
    stride_string name,
    stride_matrix_data* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) throw std::invalid_argument("matrix output is null");
    *output = {};
    const Matrix& matrix = resolve_matrix(string_view(name));
    output->name = copy_bytes(matrix.name);
    try {
      output->alphabet = copy_bytes(matrix.alphabet);
      output->values = copy_values<std::int8_t>(matrix.values);
    } catch (...) {
      stride_go_free_matrix(output);
      throw;
    }
    output->wildcard_index = matrix.wildcard;
    output->gap_score = matrix.gap_score;
    output->gap_open_score = matrix.gap_open;
    output->gap_extend_score = matrix.gap_extend;
    output->has_affine = matrix.has_affine ? 1 : 0;
    output->values_size = matrix.values.size();
  });
}

int stride_go_matrix_score_step_limit(
    stride_matrix_view matrix_input,
    int64_t gap_open_score,
    int64_t gap_extend_score,
    int64_t* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) {
      throw std::invalid_argument("matrix step-limit output is null");
    }
    const Matrix matrix = matrix_from_view(matrix_input);
    std::int64_t maximum = 0;
    for (const std::int8_t value : matrix.values) {
      maximum = std::max<std::int64_t>(
          maximum, std::abs(static_cast<int>(value)));
    }
    maximum = std::max(maximum, std::abs(gap_open_score));
    maximum = std::max(maximum, std::abs(gap_extend_score));
    *output = maximum;
  });
}

int stride_go_matrix_encode(
    stride_matrix_view matrix_input,
    stride_string input,
    stride_uints* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) {
      throw std::invalid_argument("matrix encoding output is null");
    }
    *output = {};
    const Matrix matrix = matrix_from_view(matrix_input);
    const auto encoded = matrix_encode(matrix, string_view(input));
    std::vector<std::uint64_t> widened(encoded.begin(), encoded.end());
    output->data = copy_values<std::uint64_t>(widened);
    output->size = widened.size();
  });
}

int stride_go_matrix_score(
    int local,
    stride_matrix_view matrix_input,
    stride_string query,
    stride_string target,
    int64_t gap_open_score,
    int64_t gap_extend_score,
    int64_t* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) throw std::invalid_argument("matrix score output is null");
    const Matrix matrix = matrix_from_view(matrix_input);
    *output = matrix_score(
        matrix, string_view(query), string_view(target), local != 0,
        gap_open_score, gap_extend_score);
  });
}

int stride_go_matrix_cdist(
    int local,
    stride_matrix_view matrix_input,
    stride_strings queries_input,
    stride_strings targets_input,
    int64_t gap_open_score,
    int64_t gap_extend_score,
    stride_doubles* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) {
      throw std::invalid_argument("matrix cdist output is null");
    }
    *output = {};
    const Matrix matrix = matrix_from_view(matrix_input);
    const auto query_views = string_views(queries_input);
    const auto target_views = string_views(targets_input);
    if (!query_views.empty() &&
        target_views.size() >
            std::numeric_limits<std::size_t>::max() / query_views.size()) {
      throw std::length_error("matrix cdist result size overflow");
    }
    std::vector<std::vector<std::uint16_t>> queries;
    std::vector<std::vector<std::uint16_t>> targets;
    queries.reserve(query_views.size());
    targets.reserve(target_views.size());
    for (const std::string_view value : query_views) {
      queries.push_back(matrix_encode(matrix, value));
    }
    for (const std::string_view value : target_views) {
      targets.push_back(matrix_encode(matrix, value));
    }
    const auto lookup = [&](std::uint16_t left, std::uint16_t right) {
      return matrix.lookup(left, right);
    };
    std::vector<double> values;
    values.reserve(queries.size() * targets.size());
    for (const auto& query : queries) {
      for (const auto& target : targets) {
        const Score score = local != 0
            ? core::substitution_matrix_affine_score<true>(
                  query, target, lookup, gap_open_score, gap_extend_score)
            : core::substitution_matrix_affine_score<false>(
                  query, target, lookup, gap_open_score, gap_extend_score);
        values.push_back(static_cast<double>(score));
      }
    }
    output->data = copy_values<double>(values);
    output->size = values.size();
  });
}

int stride_go_matrix_path(
    int local,
    stride_matrix_view matrix_input,
    stride_string query_input,
    stride_string target_input,
    int64_t gap_open_score,
    int64_t gap_extend_score,
    stride_alignment_path* output,
    stride_bytes* error) {
  return guarded(error, [&] {
    if (output == nullptr) throw std::invalid_argument("matrix path output is null");
    *output = {};
    const Matrix matrix = matrix_from_view(matrix_input);
    const std::string_view query = string_view(query_input);
    const std::string_view target = string_view(target_input);
    const auto path = matrix_path(
        matrix, query, target, local != 0,
        gap_open_score, gap_extend_score);
    copy_path(path, query, target, output);
  });
}

}  // extern "C"
