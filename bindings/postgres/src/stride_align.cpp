extern "C" {
#include "postgres.h"

#include "catalog/pg_type_d.h"
#include "fmgr.h"
#include "mb/pg_wchar.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/fmgrprotos.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
}

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "target_profile.hpp"

#include "stride_align/alignment.hpp"
#include "stride_align/batch.hpp"
#include "stride_align/caverphone.hpp"
#include "stride_align/cologne_phonetic.hpp"
#include "stride_align/core.hpp"
#include "stride_align/daitch_mokotoff.hpp"
#include "stride_align/double_metaphone.hpp"
#include "stride_align/dtw.hpp"
#include "stride_align/encoded.hpp"
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

#ifndef STRIDE_ALIGN_POSTGRES_SIMD_LEVEL
#define STRIDE_ALIGN_POSTGRES_SIMD_LEVEL "unspecified"
#endif

extern "C" {
PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(stride_pg_score);
PG_FUNCTION_INFO_V1(stride_pg_scores);
PG_FUNCTION_INFO_V1(stride_pg_top_k);
PG_FUNCTION_INFO_V1(stride_pg_cdist);
PG_FUNCTION_INFO_V1(stride_pg_cdist_above_threshold);
PG_FUNCTION_INFO_V1(stride_pg_cdist_top_k);
PG_FUNCTION_INFO_V1(stride_pg_cdist_top_k_per_query);
PG_FUNCTION_INFO_V1(stride_pg_lcs_length);
PG_FUNCTION_INFO_V1(stride_pg_lcs_substring_length);
PG_FUNCTION_INFO_V1(stride_pg_lcs_substring);
PG_FUNCTION_INFO_V1(stride_pg_ngram);
PG_FUNCTION_INFO_V1(stride_pg_ngram_scores);
PG_FUNCTION_INFO_V1(stride_pg_similarity);
PG_FUNCTION_INFO_V1(stride_pg_similarity_scores);
PG_FUNCTION_INFO_V1(stride_pg_monge_elkan);
PG_FUNCTION_INFO_V1(stride_pg_phonetic);
PG_FUNCTION_INFO_V1(stride_pg_phonetic_equal);
PG_FUNCTION_INFO_V1(stride_pg_double_metaphone);
PG_FUNCTION_INFO_V1(stride_pg_dtw);
PG_FUNCTION_INFO_V1(stride_pg_dtw_distances);
PG_FUNCTION_INFO_V1(stride_pg_alignment_path);
PG_FUNCTION_INFO_V1(stride_pg_backend);
PG_FUNCTION_INFO_V1(stride_pg_encoding);
PG_FUNCTION_INFO_V1(stride_pg_prepare_info);
}

namespace {

namespace batch = ::stride_align::batch;

struct InputText {
  text* datum = nullptr;

  std::string_view view() const noexcept {
    return std::string_view(
        VARDATA_ANY(datum),
        static_cast<std::size_t>(VARSIZE_ANY_EXHDR(datum)));
  }
};

InputText input_text(FunctionCallInfo fcinfo, int argument) {
  if (PG_ARGISNULL(argument)) {
    throw std::invalid_argument("required text argument is NULL");
  }
  return {PG_GETARG_TEXT_PP(argument)};
}

std::string text_string(FunctionCallInfo fcinfo, int argument) {
  const auto input = input_text(fcinfo, argument);
  return std::string(input.view());
}

struct NativeEncoding {
  int id = PG_SQL_ASCII;
  ::stride_align::encoded::EncodingProfile profile;

  static NativeEncoding database() {
    NativeEncoding result;
    result.id = GetDatabaseEncoding();
    result.profile.max_width = static_cast<std::size_t>(
        pg_encoding_max_length(result.id));
    result.profile.fixed_width = result.profile.max_width == 1U ? 1U : 0U;
    return result;
  }

  std::size_t width(std::string_view remaining) const {
    return static_cast<std::size_t>(pg_encoding_mblen_or_incomplete(
        id, remaining.data(), remaining.size()));
  }

  std::size_t length(std::string_view input) const {
    return ::stride_align::encoded::character_count(
        input, profile,
        [this](std::string_view remaining) { return width(remaining); });
  }

  ::stride_align::utf8::PreparedPair prepare(
      std::string_view query,
      std::string_view target) const {
    return ::stride_align::encoded::prepare_pair(
        query, target, profile,
        [this](std::string_view remaining) { return width(remaining); });
  }

  ::stride_align::encoded::TokenizedText tokenize(
      std::string_view input) const {
    return ::stride_align::encoded::tokenize(
        input, profile,
        [this](std::string_view remaining) { return width(remaining); });
  }

  std::vector<std::uint32_t> semantic_tokens(std::string_view input) const {
    // Unicode-sensitive algorithms retain their existing behavior in UTF-8
    // databases. Other server encodings deliberately keep non-ASCII symbols
    // opaque while preserving ASCII letters and whitespace literally.
    if (id == PG_UTF8) return ::stride_align::utf8::prepare_streaming(input);
    return tokenize(input).tokens;
  }
};

struct NativePrepare {
  NativeEncoding encoding;

  ::stride_align::utf8::PreparedPair operator()(
      std::string_view query,
      std::string_view target) const {
    return encoding.prepare(query, target);
  }
};

batch::ScoreOptions score_options(FunctionCallInfo fcinfo, int first) {
  batch::ScoreOptions options;
  options.match_score = PG_GETARG_INT64(first);
  options.mismatch_score = PG_GETARG_INT64(first + 1);
  options.gap_open_score = PG_GETARG_INT64(first + 2);
  options.gap_extend_score = PG_GETARG_INT64(first + 3);
  options.prefix_weight = PG_GETARG_FLOAT8(first + 4);
  options.prefix_threshold = PG_GETARG_FLOAT8(first + 5);
  const std::int64_t prefix_cap = PG_GETARG_INT64(first + 6);
  if (prefix_cap < 0) {
    throw std::invalid_argument("prefix_cap must be non-negative");
  }
  options.prefix_cap = static_cast<std::size_t>(prefix_cap);
  if (!std::isfinite(options.prefix_weight) || options.prefix_weight < 0.0 ||
      !std::isfinite(options.prefix_threshold) ||
      options.prefix_threshold < 0.0 || options.prefix_threshold > 1.0) {
    throw std::invalid_argument(
        "prefix_weight must be non-negative and prefix_threshold must be between 0 and 1");
  }
  return options;
}

std::optional<double> optional_double(FunctionCallInfo fcinfo, int argument) {
  return argument < PG_NARGS() && !PG_ARGISNULL(argument)
      ? std::optional<double>(PG_GETARG_FLOAT8(argument))
      : std::nullopt;
}

std::optional<std::size_t> optional_size(
    FunctionCallInfo fcinfo,
    int argument,
    const char* name) {
  if (argument >= PG_NARGS() || PG_ARGISNULL(argument)) return std::nullopt;
  const std::int64_t input = PG_GETARG_INT64(argument);
  if (input < 0) {
    throw std::invalid_argument(std::string(name) + " must be non-negative");
  }
  return static_cast<std::size_t>(input);
}

std::size_t required_size(
    FunctionCallInfo fcinfo,
    int argument,
    const char* name) {
  const auto value = optional_size(fcinfo, argument, name);
  if (!value.has_value()) {
    throw std::invalid_argument(std::string(name) + " cannot be NULL");
  }
  return *value;
}

std::vector<std::optional<batch::Text>> text_array(
    ArrayType* input,
    const NativeEncoding& encoding) {
  if (ARR_NDIM(input) > 1) {
    throw std::invalid_argument("text input must be a one-dimensional array");
  }
  Datum* values = nullptr;
  bool* nulls = nullptr;
  int count = 0;
  deconstruct_array(
      input, TEXTOID, -1, false, TYPALIGN_INT,
      &values, &nulls, &count);
  std::vector<std::optional<batch::Text>> output;
  output.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    if (nulls[index]) {
      output.emplace_back(std::nullopt);
      continue;
    }
    text* value = DatumGetTextPP(values[index]);
    const std::string_view bytes(
        VARDATA_ANY(value),
        static_cast<std::size_t>(VARSIZE_ANY_EXHDR(value)));
    output.emplace_back(batch::Text(std::string(bytes), encoding.length(bytes)));
  }
  return output;
}

batch::Text one_text(const InputText& input, const NativeEncoding& encoding) {
  return batch::Text(std::string(input.view()), encoding.length(input.view()));
}

std::string json_escape(std::string_view input) {
  std::string output;
  output.reserve(input.size() + 2U);
  output.push_back('"');
  constexpr char hexadecimal[] = "0123456789abcdef";
  for (const unsigned char byte : input) {
    switch (byte) {
      case '"': output += "\\\""; break;
      case '\\': output += "\\\\"; break;
      case '\b': output += "\\b"; break;
      case '\f': output += "\\f"; break;
      case '\n': output += "\\n"; break;
      case '\r': output += "\\r"; break;
      case '\t': output += "\\t"; break;
      default:
        if (byte < 0x20U) {
          output += "\\u00";
          output.push_back(hexadecimal[byte >> 4U]);
          output.push_back(hexadecimal[byte & 0x0fU]);
        } else {
          output.push_back(static_cast<char>(byte));
        }
    }
  }
  output.push_back('"');
  return output;
}

std::string json_number(double value, bool integral) {
  if (!std::isfinite(value)) return "null";
  if (integral) return std::to_string(static_cast<std::int64_t>(value));
  std::ostringstream output;
  output << std::setprecision(17) << value;
  return output.str();
}

Datum jsonb_datum(const std::string& json) {
  return DirectFunctionCall1(jsonb_in, CStringGetDatum(json.c_str()));
}

Datum text_datum(std::string_view value) {
  return PointerGetDatum(cstring_to_text_with_len(
      value.data(), static_cast<int>(value.size())));
}

Datum float_array_datum(
    std::span<const std::optional<double>> values,
    std::span<const int> dimensions = {}) {
  Datum* elements = static_cast<Datum*>(
      palloc(sizeof(Datum) * std::max<std::size_t>(values.size(), 1U)));
  bool* nulls = static_cast<bool*>(
      palloc(sizeof(bool) * std::max<std::size_t>(values.size(), 1U)));
  for (std::size_t index = 0; index < values.size(); ++index) {
    nulls[index] = !values[index].has_value();
    elements[index] = Float8GetDatum(values[index].value_or(0.0));
  }
  int16 element_width = 0;
  bool element_by_value = false;
  char element_alignment = 0;
  get_typlenbyvalalign(
      FLOAT8OID, &element_width, &element_by_value, &element_alignment);
  if (dimensions.empty()) {
    std::array<int, 1> one_dimension{static_cast<int>(values.size())};
    std::array<int, 1> one_lower_bound{1};
    return PointerGetDatum(construct_md_array(
        elements, nulls, 1, one_dimension.data(), one_lower_bound.data(),
        FLOAT8OID, element_width, element_by_value, element_alignment));
  }
  std::vector<int> lower_bounds(dimensions.size(), 1);
  return PointerGetDatum(construct_md_array(
      elements, nulls,
      static_cast<int>(dimensions.size()),
      const_cast<int*>(dimensions.data()), lower_bounds.data(),
      FLOAT8OID, element_width, element_by_value, element_alignment));
}

std::string ranked_json(
    std::span<const batch::RankedMatch> matches,
    std::span<const std::optional<batch::Text>> targets,
    bool integral,
    bool best) {
  if (best && matches.empty()) return "null";
  std::string output = best ? "" : "[";
  for (std::size_t index = 0; index < matches.size(); ++index) {
    if (!best && index != 0U) output.push_back(',');
    const auto& match = matches[index];
    output += "{\"target\":" + json_escape(targets[match.index]->bytes);
    output += ",\"score\":" + json_number(match.score, integral);
    output += ",\"index\":" + std::to_string(match.index) + "}";
    if (best) break;
  }
  if (!best) output.push_back(']');
  return output;
}

std::string matrix_matches_json(
    std::span<const batch::MatrixMatch> matches,
    std::span<const std::optional<batch::Text>> queries,
    std::span<const std::optional<batch::Text>> targets) {
  std::string output = "[";
  for (std::size_t index = 0; index < matches.size(); ++index) {
    if (index != 0U) output.push_back(',');
    const auto& match = matches[index];
    output += "{\"score\":" + json_number(match.score, false);
    output += ",\"query\":" + json_escape(queries[match.query_index]->bytes);
    output += ",\"target\":" + json_escape(targets[match.target_index]->bytes);
    output += ",\"query_index\":" + std::to_string(match.query_index);
    output += ",\"target_index\":" + std::to_string(match.target_index) + "}";
  }
  output.push_back(']');
  return output;
}

template <typename Function>
Datum catch_errors(Function&& function) {
  std::array<char, 1024> message{};
  try {
    return function();
  } catch (const std::exception& error) {
    std::snprintf(message.data(), message.size(), "%s", error.what());
  } catch (...) {
    std::snprintf(message.data(), message.size(), "%s", "unknown C++ exception");
  }
  ereport(ERROR,
          (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
           errmsg("stride-align: %s", message.data())));
  return static_cast<Datum>(0);
}

std::vector<double> numeric_array(ArrayType* input, const char* name) {
  if (ARR_NDIM(input) != 1) {
    throw std::invalid_argument(std::string(name) + " must be one-dimensional");
  }
  const Oid type = ARR_ELEMTYPE(input);
  int16 width = 0;
  bool by_value = false;
  char alignment = 0;
  get_typlenbyvalalign(type, &width, &by_value, &alignment);
  Datum* values = nullptr;
  bool* nulls = nullptr;
  int count = 0;
  deconstruct_array(
      input, type, width, by_value, alignment,
      &values, &nulls, &count);
  if (count == 0) {
    throw std::invalid_argument(std::string(name) + " must be non-empty");
  }
  std::vector<double> output;
  output.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    if (nulls[index]) {
      throw std::invalid_argument(std::string(name) + " cannot contain NULL");
    }
    double value = 0.0;
    switch (type) {
      case FLOAT8OID: value = DatumGetFloat8(values[index]); break;
      case FLOAT4OID: value = static_cast<double>(DatumGetFloat4(values[index])); break;
      case INT2OID: value = static_cast<double>(DatumGetInt16(values[index])); break;
      case INT4OID: value = static_cast<double>(DatumGetInt32(values[index])); break;
      case INT8OID: value = static_cast<double>(DatumGetInt64(values[index])); break;
      default:
        throw std::invalid_argument(
            std::string(name) + " must use a built-in numeric array type");
    }
    if (!std::isfinite(value)) {
      throw std::invalid_argument(std::string(name) + " must contain finite values");
    }
    output.push_back(value);
  }
  return output;
}

std::optional<std::size_t> dtw_window(
    std::optional<double> input,
    std::size_t query_size,
    std::size_t target_size) {
  if (!input.has_value()) return std::nullopt;
  if (*input < 0.0 || !std::isfinite(*input)) {
    throw std::invalid_argument("window must be non-negative and finite");
  }
  if (*input > 0.0 && *input < 1.0) {
    return static_cast<std::size_t>(std::ceil(
        *input * static_cast<double>(std::max(query_size, target_size))));
  }
  if (std::trunc(*input) != *input) {
    throw std::invalid_argument(
        "window must be an integer radius or a fraction in (0, 1)");
  }
  return static_cast<std::size_t>(*input);
}

double dtw_score(
    const std::vector<double>& query,
    const std::vector<double>& target,
    std::optional<double> window,
    std::string_view distance,
    std::optional<double> cutoff) {
  if (distance != "l1" && distance != "l2_squared") {
    throw std::invalid_argument("distance must be 'l1' or 'l2_squared'");
  }
  if (cutoff.has_value() && (*cutoff < 0.0 || !std::isfinite(*cutoff))) {
    throw std::invalid_argument("score_cutoff must be non-negative and finite");
  }
  return ::stride_align::dtw::dtw_score_scalar<double, double>(
      query, target,
      distance == "l1"
          ? ::stride_align::dtw::DistanceKind::kL1
          : ::stride_align::dtw::DistanceKind::kL2Squared,
      dtw_window(window, query.size(), target.size()), cutoff);
}

}  // namespace

extern "C" PGDLLEXPORT Datum
stride_pg_score(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const InputText query = input_text(fcinfo, 0);
    const InputText target = input_text(fcinfo, 1);
    const batch::Scorer scorer = batch::parse_scorer(text_string(fcinfo, 2));
    const batch::ScoreOptions options = score_options(fcinfo, 3);
    const NativeEncoding encoding = NativeEncoding::database();
    const double result = batch::score_with(
        scorer, query.view(), target.view(), NativePrepare{encoding}, options,
        optional_double(fcinfo, 10), optional_size(fcinfo, 11, "distance_cutoff"));
    return Float8GetDatum(result);
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_scores(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const batch::Text query = one_text(input_text(fcinfo, 0), encoding);
    const auto targets = text_array(PG_GETARG_ARRAYTYPE_P(1), encoding);
    const batch::Scorer scorer = batch::parse_scorer(text_string(fcinfo, 2));
    const batch::ScoreOptions options = score_options(fcinfo, 3);
    std::vector<std::optional<double>> scores;
    scores.reserve(targets.size());
    const auto similarity_cutoff = optional_double(fcinfo, 10);
    const auto distance_cutoff = optional_size(fcinfo, 11, "distance_cutoff");
    for (const auto& target : targets) {
      scores.push_back(target.has_value()
          ? std::optional<double>(batch::score_with(
                scorer, query.bytes, target->bytes, NativePrepare{encoding},
                options, similarity_cutoff, distance_cutoff))
          : std::nullopt);
    }
    return float_array_datum(scores);
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_top_k(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const batch::Text query = one_text(input_text(fcinfo, 0), encoding);
    const auto targets = text_array(PG_GETARG_ARRAYTYPE_P(1), encoding);
    const batch::Scorer scorer = batch::parse_scorer(text_string(fcinfo, 2));
    const std::size_t k = required_size(fcinfo, 3, "k");
    const bool best = PG_GETARG_BOOL(4);
    const batch::ScoreOptions options = score_options(fcinfo, 5);
    const auto matches = batch::top_k_with(
        query, targets, scorer, best ? 1U : k,
        NativePrepare{encoding}, options);
    return jsonb_datum(ranked_json(
        matches, targets, batch::is_integral(scorer), best));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_cdist(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const auto queries = text_array(PG_GETARG_ARRAYTYPE_P(0), encoding);
    const auto targets = text_array(PG_GETARG_ARRAYTYPE_P(1), encoding);
    const batch::Scorer scorer = batch::parse_scorer(text_string(fcinfo, 2));
    const batch::ScoreOptions options = score_options(fcinfo, 3);
    const auto matrix = batch::cdist_with(
        queries, targets, scorer, NativePrepare{encoding}, options);
    std::vector<std::optional<double>> flat;
    flat.reserve(queries.size() * targets.size());
    for (const auto& row : matrix) flat.insert(flat.end(), row.begin(), row.end());
    const std::array<int, 2> dimensions{
        static_cast<int>(queries.size()), static_cast<int>(targets.size())};
    return float_array_datum(flat, dimensions);
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_cdist_above_threshold(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const auto queries = text_array(PG_GETARG_ARRAYTYPE_P(0), encoding);
    const auto targets = text_array(PG_GETARG_ARRAYTYPE_P(1), encoding);
    const batch::Scorer scorer = batch::parse_scorer(text_string(fcinfo, 2));
    const double threshold = PG_GETARG_FLOAT8(3);
    const batch::ScoreOptions options = score_options(fcinfo, 4);
    const auto matches = batch::cdist_above_threshold_with(
        queries, targets, scorer, threshold,
        NativePrepare{encoding}, options);
    return jsonb_datum(matrix_matches_json(matches, queries, targets));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_cdist_top_k(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const auto queries = text_array(PG_GETARG_ARRAYTYPE_P(0), encoding);
    const auto targets = text_array(PG_GETARG_ARRAYTYPE_P(1), encoding);
    const batch::Scorer scorer = batch::parse_scorer(text_string(fcinfo, 2));
    const std::size_t k = required_size(fcinfo, 3, "k");
    const bool reject_duplicates = PG_GETARG_BOOL(4);
    const batch::ScoreOptions options = score_options(fcinfo, 5);
    const auto matches = batch::cdist_top_k_with(
        queries, targets, scorer, k, reject_duplicates,
        NativePrepare{encoding}, options);
    return jsonb_datum(matrix_matches_json(matches, queries, targets));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_cdist_top_k_per_query(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const auto queries = text_array(PG_GETARG_ARRAYTYPE_P(0), encoding);
    const auto targets = text_array(PG_GETARG_ARRAYTYPE_P(1), encoding);
    const batch::Scorer scorer = batch::parse_scorer(text_string(fcinfo, 2));
    const std::size_t k = required_size(fcinfo, 3, "k");
    const batch::ScoreOptions options = score_options(fcinfo, 4);
    const auto rows = batch::cdist_top_k_per_query_with(
        queries, targets, scorer, k, NativePrepare{encoding}, options);
    std::string json = "[";
    for (std::size_t query_index = 0; query_index < rows.size(); ++query_index) {
      if (query_index != 0U) json.push_back(',');
      json += "{\"query_index\":" + std::to_string(query_index);
      json += ",\"query\":";
      json += queries[query_index].has_value()
          ? json_escape(queries[query_index]->bytes) : "null";
      json += ",\"matches\":" + ranked_json(
          rows[query_index], targets, batch::is_integral(scorer), false) + "}";
    }
    json.push_back(']');
    return jsonb_datum(json);
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_lcs_length(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const auto left = encoding.semantic_tokens(input_text(fcinfo, 0).view());
    const auto right = encoding.semantic_tokens(input_text(fcinfo, 1).view());
    return Int64GetDatum(static_cast<std::int64_t>(
        ::stride_align::lcs::lcs_length(left, right)));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_lcs_substring_length(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const auto left = encoding.semantic_tokens(input_text(fcinfo, 0).view());
    const auto right = encoding.semantic_tokens(input_text(fcinfo, 1).view());
    return Int64GetDatum(static_cast<std::int64_t>(
        ::stride_align::lcs::lcs_substring_length(left, right)));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_lcs_substring(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const std::string_view left_bytes = input_text(fcinfo, 0).view();
    const std::string_view right_bytes = input_text(fcinfo, 1).view();
    const auto left = encoding.tokenize(left_bytes);
    const auto right = encoding.tokenize(right_bytes);
    const auto info = ::stride_align::lcs::lcs_substring_info(
        left.tokens, right.tokens);
    if (info.length == 0U) return text_datum("");
    const std::size_t first = info.end_a - info.length;
    return text_datum(left_bytes.substr(
        left.byte_offsets[first],
        left.byte_offsets[info.end_a] - left.byte_offsets[first]));
  });
}

enum class NgramMetric { jaccard, dice, cosine, overlap };

NgramMetric ngram_metric(std::string_view name) {
  if (name == "jaccard") return NgramMetric::jaccard;
  if (name == "dice") return NgramMetric::dice;
  if (name == "cosine") return NgramMetric::cosine;
  if (name == "overlap") return NgramMetric::overlap;
  throw std::invalid_argument("unknown n-gram metric: " + std::string(name));
}

double ngram_score(
    NgramMetric metric,
    const std::vector<std::uint32_t>& left,
    const std::vector<std::uint32_t>& right,
    std::size_t n) {
  if (n == 0U) throw std::invalid_argument("n must be positive");
  switch (metric) {
    case NgramMetric::jaccard:
      return ::stride_align::ngram::jaccard(left, right, n);
    case NgramMetric::dice:
      return ::stride_align::ngram::dice(left, right, n);
    case NgramMetric::cosine:
      return ::stride_align::ngram::cosine(left, right, n);
    case NgramMetric::overlap:
      return ::stride_align::ngram::overlap(left, right, n);
  }
  return 0.0;
}

extern "C" PGDLLEXPORT Datum
stride_pg_ngram(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const auto left = encoding.semantic_tokens(input_text(fcinfo, 0).view());
    const auto right = encoding.semantic_tokens(input_text(fcinfo, 1).view());
    return Float8GetDatum(ngram_score(
        ngram_metric(text_string(fcinfo, 2)), left, right,
        required_size(fcinfo, 3, "n")));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_ngram_scores(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const auto left = encoding.semantic_tokens(input_text(fcinfo, 0).view());
    const auto targets = text_array(PG_GETARG_ARRAYTYPE_P(1), encoding);
    const auto metric = ngram_metric(text_string(fcinfo, 2));
    const std::size_t n = required_size(fcinfo, 3, "n");
    std::vector<std::optional<double>> output;
    output.reserve(targets.size());
    for (const auto& target : targets) {
      output.push_back(target.has_value()
          ? std::optional<double>(ngram_score(
                metric, left, encoding.semantic_tokens(target->bytes), n))
          : std::nullopt);
    }
    return float_array_datum(output);
  });
}

enum class SimilarityMetric {
  ratcliff,
  partial,
  token_sort,
  token_set,
  partial_token_sort,
  partial_token_set,
  weighted,
};

SimilarityMetric similarity_metric(std::string_view name) {
  if (name == "ratcliff_obershelp") return SimilarityMetric::ratcliff;
  if (name == "partial_ratio") return SimilarityMetric::partial;
  if (name == "token_sort_ratio") return SimilarityMetric::token_sort;
  if (name == "token_set_ratio") return SimilarityMetric::token_set;
  if (name == "partial_token_sort_ratio") {
    return SimilarityMetric::partial_token_sort;
  }
  if (name == "partial_token_set_ratio") {
    return SimilarityMetric::partial_token_set;
  }
  if (name == "wratio") return SimilarityMetric::weighted;
  throw std::invalid_argument("unknown similarity metric: " + std::string(name));
}

double text_similarity(
    SimilarityMetric metric,
    const std::vector<std::uint32_t>& left,
    const std::vector<std::uint32_t>& right) {
  const auto left_span = std::span<const std::uint32_t>(left);
  const auto right_span = std::span<const std::uint32_t>(right);
  switch (metric) {
    case SimilarityMetric::ratcliff:
      return ::stride_align::ratcliff_obershelp::ratcliff_obershelp_similarity(
          left, right);
    case SimilarityMetric::partial:
      return ::stride_align::partial_ratio::partial_ratio(left, right);
    case SimilarityMetric::token_sort:
      return ::stride_align::token_ratios::token_sort_ratio(left, right);
    case SimilarityMetric::token_set:
      return ::stride_align::token_ratios::token_set_ratio(left, right);
    case SimilarityMetric::partial_token_sort:
      return ::stride_align::wratio::partial_token_sort_ratio_engine<std::uint32_t>(
          left_span, right_span);
    case SimilarityMetric::partial_token_set:
      return ::stride_align::wratio::partial_token_set_ratio_engine<std::uint32_t>(
          left_span, right_span);
    case SimilarityMetric::weighted:
      return ::stride_align::wratio::native_wratio(left, right);
  }
  return 0.0;
}

extern "C" PGDLLEXPORT Datum
stride_pg_similarity(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    return Float8GetDatum(text_similarity(
        similarity_metric(text_string(fcinfo, 2)),
        encoding.semantic_tokens(input_text(fcinfo, 0).view()),
        encoding.semantic_tokens(input_text(fcinfo, 1).view())));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_similarity_scores(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const auto left = encoding.semantic_tokens(input_text(fcinfo, 0).view());
    const auto targets = text_array(PG_GETARG_ARRAYTYPE_P(1), encoding);
    const auto metric = similarity_metric(text_string(fcinfo, 2));
    std::vector<std::optional<double>> output;
    output.reserve(targets.size());
    for (const auto& target : targets) {
      output.push_back(target.has_value()
          ? std::optional<double>(text_similarity(
                metric, left, encoding.semantic_tokens(target->bytes)))
          : std::nullopt);
    }
    return float_array_datum(output);
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

::stride_align::utf8::PreparedPair u32_pair(
    const std::vector<std::uint32_t>& left,
    const std::vector<std::uint32_t>& right) {
  return {
      ::stride_align::utf8::TokenWidth::u32,
      false,
      false,
      left,
      right};
}

double monge_inner(
    batch::Scorer scorer,
    const std::vector<std::uint32_t>& left,
    const std::vector<std::uint32_t>& right) {
  if (scorer != batch::Scorer::jaro &&
      scorer != batch::Scorer::jaro_winkler &&
      scorer != batch::Scorer::levenshtein_normalized &&
      scorer != batch::Scorer::indel_normalized) {
    throw std::invalid_argument(
        "Monge-Elkan inner scorer must be jaro, jaro_winkler, "
        "levenshtein_normalized, or indel_normalized");
  }
  return batch::score_prepared(scorer, u32_pair(left, right));
}

double monge_direction(
    const std::vector<std::vector<std::uint32_t>>& left,
    const std::vector<std::vector<std::uint32_t>>& right,
    batch::Scorer scorer) {
  double total = 0.0;
  for (const auto& token : left) {
    double best = 0.0;
    for (const auto& candidate : right) {
      best = std::max(best, monge_inner(scorer, token, candidate));
    }
    total += best;
  }
  return total / static_cast<double>(left.size());
}

extern "C" PGDLLEXPORT Datum
stride_pg_monge_elkan(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const auto left = word_tokens(
        encoding.semantic_tokens(input_text(fcinfo, 0).view()));
    const auto right = word_tokens(
        encoding.semantic_tokens(input_text(fcinfo, 1).view()));
    if (left.empty() && right.empty()) return Float8GetDatum(1.0);
    if (left.empty() || right.empty()) return Float8GetDatum(0.0);
    const batch::Scorer scorer = batch::parse_scorer(text_string(fcinfo, 2));
    const double forward = monge_direction(left, right, scorer);
    return Float8GetDatum(PG_GETARG_BOOL(3)
        ? (forward + monge_direction(right, left, scorer)) / 2.0
        : forward);
  });
}

std::string phonetic_value(
    std::string_view algorithm,
    std::string_view input,
    const NativeEncoding& encoding,
    std::int64_t first,
    bool first_flag,
    bool second_flag) {
  if (algorithm == "soundex") return ::stride_align::phonetic::soundex(input);
  if (algorithm == "metaphone") {
    const auto variant = static_cast<::stride_align::phonetic::MetaphoneVariant>(first);
    if (variant != ::stride_align::phonetic::MetaphoneVariant::kPhilips &&
        variant != ::stride_align::phonetic::MetaphoneVariant::kJellyfish) {
      throw std::invalid_argument("variant must be 0 or 1");
    }
    return ::stride_align::phonetic::metaphone(input, variant);
  }
  if (algorithm == "nysiis") return ::stride_align::phonetic::nysiis(input);
  if (algorithm == "match_rating_codex") {
    return ::stride_align::phonetic::match_rating_codex(input);
  }
  if (algorithm == "caverphone") {
    return ::stride_align::phonetic::caverphone(input);
  }
  if (algorithm == "cologne_phonetic") {
    return ::stride_align::phonetic::cologne_phonetic(
        encoding.semantic_tokens(input));
  }
  if (algorithm == "daitch_mokotoff") {
    return ::stride_align::phonetic::daitch_mokotoff(
        encoding.semantic_tokens(input), first_flag, second_flag);
  }
  throw std::invalid_argument("unknown phonetic algorithm: " + std::string(algorithm));
}

extern "C" PGDLLEXPORT Datum
stride_pg_phonetic(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const std::string algorithm = text_string(fcinfo, 0);
    const std::string_view input = input_text(fcinfo, 1).view();
    const std::int64_t first = PG_GETARG_INT64(2);
    return text_datum(phonetic_value(
        algorithm, input, encoding, first,
        PG_GETARG_BOOL(3), PG_GETARG_BOOL(4)));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_phonetic_equal(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const std::string algorithm = text_string(fcinfo, 0);
    const std::string_view left = input_text(fcinfo, 1).view();
    const std::string_view right = input_text(fcinfo, 2).view();
    const std::int64_t first = PG_GETARG_INT64(3);
    if (algorithm == "match_rating_compare") {
      return BoolGetDatum(::stride_align::phonetic::match_rating_compare(left, right));
    }
    const std::string left_value = phonetic_value(
        algorithm, left, encoding, first, true, true);
    const std::string right_value = phonetic_value(
        algorithm, right, encoding, first, true, true);
    return BoolGetDatum(!left_value.empty() && left_value == right_value);
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_double_metaphone(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const std::size_t maximum = required_size(fcinfo, 1, "max_length");
    const std::int64_t raw_variant = PG_GETARG_INT64(2);
    const auto variant = static_cast<
        ::stride_align::phonetic::DoubleMetaphoneVariant>(raw_variant);
    const auto value = ::stride_align::phonetic::double_metaphone(
        input_text(fcinfo, 0).view(), maximum, variant);
    return jsonb_datum(
        "{\"primary\":" + json_escape(value.primary) +
        ",\"alternate\":" + json_escape(value.alternate) + "}");
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_dtw(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const auto query = numeric_array(PG_GETARG_ARRAYTYPE_P(0), "query");
    const auto target = numeric_array(PG_GETARG_ARRAYTYPE_P(1), "target");
    return Float8GetDatum(dtw_score(
        query, target, optional_double(fcinfo, 2), text_string(fcinfo, 3),
        optional_double(fcinfo, 4)));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_dtw_distances(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const auto query = numeric_array(PG_GETARG_ARRAYTYPE_P(0), "query");
    ArrayType* targets_array = PG_GETARG_ARRAYTYPE_P(1);
    if (ARR_NDIM(targets_array) != 2) {
      throw std::invalid_argument("targets must be a rectangular two-dimensional array");
    }
    const int* dimensions = ARR_DIMS(targets_array);
    const std::size_t rows = static_cast<std::size_t>(dimensions[0]);
    const std::size_t columns = static_cast<std::size_t>(dimensions[1]);
    const Oid type = ARR_ELEMTYPE(targets_array);
    int16 width = 0;
    bool by_value = false;
    char alignment = 0;
    get_typlenbyvalalign(type, &width, &by_value, &alignment);
    Datum* values = nullptr;
    bool* nulls = nullptr;
    int count = 0;
    deconstruct_array(
        targets_array, type, width, by_value, alignment,
        &values, &nulls, &count);
    std::vector<std::optional<double>> output;
    output.reserve(rows);
    for (std::size_t row = 0; row < rows; ++row) {
      std::vector<double> target;
      target.reserve(columns);
      for (std::size_t column = 0; column < columns; ++column) {
        const std::size_t index = row * columns + column;
        if (nulls[index]) {
          throw std::invalid_argument("targets cannot contain NULL");
        }
        double number = 0.0;
        switch (type) {
          case FLOAT8OID: number = DatumGetFloat8(values[index]); break;
          case FLOAT4OID: number = DatumGetFloat4(values[index]); break;
          case INT2OID: number = DatumGetInt16(values[index]); break;
          case INT4OID: number = DatumGetInt32(values[index]); break;
          case INT8OID: number = static_cast<double>(DatumGetInt64(values[index])); break;
          default:
            throw std::invalid_argument("targets must use a built-in numeric array type");
        }
        target.push_back(number);
      }
      output.emplace_back(dtw_score(
          query, target, optional_double(fcinfo, 2), text_string(fcinfo, 3),
          optional_double(fcinfo, 4)));
    }
    return float_array_datum(output);
  });
}

std::string aligned_text(
    const ::stride_align::AlignmentPath& path,
    std::string_view query,
    std::string_view target,
    const ::stride_align::encoded::TokenizedText& query_tokens,
    const ::stride_align::encoded::TokenizedText& target_tokens,
    bool query_side) {
  std::string output;
  std::size_t query_index = path.query_start;
  std::size_t target_index = path.target_start;
  for (const char operation : path.operations) {
    if (operation == '=' || operation == 'X') {
      const std::string_view source = query_side ? query : target;
      const auto& tokens = query_side ? query_tokens : target_tokens;
      const std::size_t index = query_side ? query_index : target_index;
      output.append(
          source.data() + tokens.byte_offsets[index],
          tokens.byte_offsets[index + 1U] - tokens.byte_offsets[index]);
      ++query_index;
      ++target_index;
    } else if (operation == 'D') {
      if (query_side) {
        output.append(
            query.data() + query_tokens.byte_offsets[query_index],
            query_tokens.byte_offsets[query_index + 1U] -
                query_tokens.byte_offsets[query_index]);
      } else {
        output.push_back('-');
      }
      ++query_index;
    } else {
      if (query_side) {
        output.push_back('-');
      } else {
        output.append(
            target.data() + target_tokens.byte_offsets[target_index],
            target_tokens.byte_offsets[target_index + 1U] -
                target_tokens.byte_offsets[target_index]);
      }
      ++target_index;
    }
  }
  return output;
}

extern "C" PGDLLEXPORT Datum
stride_pg_alignment_path(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const std::string_view query = input_text(fcinfo, 0).view();
    const std::string_view target = input_text(fcinfo, 1).view();
    const bool local = PG_GETARG_BOOL(2);
    const ::stride_align::Score match = PG_GETARG_INT64(3);
    const ::stride_align::Score mismatch = PG_GETARG_INT64(4);
    const ::stride_align::Score gap_open = PG_GETARG_INT64(5);
    const ::stride_align::Score gap_extend = PG_GETARG_INT64(6);
    const auto pair = encoding.prepare(query, target);
    ::stride_align::AlignmentPath path;
    if (gap_open == gap_extend) {
      path = local
          ? ::stride_align::core::smith_waterman_path(
                pair, match, mismatch, gap_open)
          : ::stride_align::core::needleman_wunsch_path(
                pair, match, mismatch, gap_open);
    } else {
      path = local
          ? ::stride_align::core::smith_waterman_affine_path(
                pair, match, mismatch, gap_open, gap_extend)
          : ::stride_align::core::needleman_wunsch_affine_path(
                pair, match, mismatch, gap_open, gap_extend);
    }
    const auto query_tokens = encoding.tokenize(query);
    const auto target_tokens = encoding.tokenize(target);
    std::string json = "{\"score\":" + std::to_string(path.score);
    json += ",\"query_start\":" + std::to_string(path.query_start);
    json += ",\"query_end\":" + std::to_string(path.query_end);
    json += ",\"target_start\":" + std::to_string(path.target_start);
    json += ",\"target_end\":" + std::to_string(path.target_end);
    json += ",\"operations\":" + json_escape(path.operations);
    json += ",\"cigar\":" + json_escape(path.cigar);
    json += ",\"matches\":" + std::to_string(path.matches);
    json += ",\"mismatches\":" + std::to_string(path.mismatches);
    json += ",\"insertions\":" + std::to_string(path.insertions);
    json += ",\"deletions\":" + std::to_string(path.deletions);
    json += ",\"aligned_length\":" + std::to_string(path.aligned_length);
    json += ",\"aligned_query\":" + json_escape(aligned_text(
        path, query, target, query_tokens, target_tokens, true));
    json += ",\"aligned_target\":" + json_escape(aligned_text(
        path, query, target, query_tokens, target_tokens, false)) + "}";
    return jsonb_datum(json);
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_backend(PG_FUNCTION_ARGS) {
  (void)fcinfo;
  return text_datum(STRIDE_ALIGN_POSTGRES_SIMD_LEVEL);
}

extern "C" PGDLLEXPORT Datum
stride_pg_encoding(PG_FUNCTION_ARGS) {
  (void)fcinfo;
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    std::string json = "{\"name\":" +
        json_escape(GetDatabaseEncodingName());
    json += ",\"id\":" + std::to_string(encoding.id);
    json += ",\"max_width\":" + std::to_string(encoding.profile.max_width);
    json += ",\"fixed_width\":" + std::to_string(encoding.profile.fixed_width);
    json += ",\"transcodes\":false}";
    return jsonb_datum(json);
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_prepare_info(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const std::string_view query = input_text(fcinfo, 0).view();
    const std::string_view target = input_text(fcinfo, 1).view();
    const auto prepared = encoding.prepare(query, target);
    const char* width = "u32";
    switch (prepared.width) {
      case ::stride_align::utf8::TokenWidth::u8: width = "u8"; break;
      case ::stride_align::utf8::TokenWidth::u16: width = "u16"; break;
      case ::stride_align::utf8::TokenWidth::u32: width = "u32"; break;
    }
    std::string json = "{\"width\":" + json_escape(width);
    json += ",\"borrowed\":" +
        std::string(prepared.borrowed_ascii ? "true" : "false");
    json += ",\"packed\":" + std::string(prepared.packed ? "true" : "false");
    json += ",\"ascii\":" + std::string(
        ::stride_align::utf8::is_ascii(query) &&
                ::stride_align::utf8::is_ascii(target)
            ? "true" : "false") + "}";
    return jsonb_datum(json);
  });
}
