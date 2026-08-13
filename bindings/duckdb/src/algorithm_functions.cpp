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
#include <utility>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"

#include "stride_align/alignment.hpp"
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
#include "stride_align_duckdb/adapter.hpp"
#include "stride_align_duckdb/registration.hpp"

namespace duckdb::stride_align_extension {
namespace {

namespace adapter = ::duckdb::stride_align_adapter;

std::vector<std::uint32_t> codepoints(const Value& value) {
  return ::stride_align::utf8::prepare_streaming(adapter::string_value(value));
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

template <typename Function>
void ExecuteCodepointInteger(
    DataChunk& arguments,
    ExpressionState&,
    Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    return Value::BIGINT(static_cast<std::int64_t>(Function{}(
        codepoints(adapter::argument(arguments, 0, row)),
        codepoints(adapter::argument(arguments, 1, row)))));
  });
}

struct LcsLength {
  std::size_t operator()(
      const std::vector<std::uint32_t>& left,
      const std::vector<std::uint32_t>& right) const {
    return ::stride_align::lcs::lcs_length(left, right);
  }
};

struct LcsSubstringLength {
  std::size_t operator()(
      const std::vector<std::uint32_t>& left,
      const std::vector<std::uint32_t>& right) const {
    return ::stride_align::lcs::lcs_substring_length(left, right);
  }
};

void LcsSubstring(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const auto left = codepoints(adapter::argument(arguments, 0, row));
    const auto right = codepoints(adapter::argument(arguments, 1, row));
    return Value(encode_utf8(
        ::stride_align::lcs::lcs_substring(left, right)));
  });
}

enum class NgramMetric { jaccard, dice, cosine, overlap };

template <NgramMetric Metric, bool Many>
void Ngram(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const auto left = codepoints(adapter::argument(arguments, 0, row));
    const std::size_t n = arguments.ColumnCount() == (Many ? 3U : 3U)
        ? adapter::nonnegative_size(
              adapter::argument(arguments, 2, row), "n")
        : 2U;
    if (n == 0U) throw std::invalid_argument("n must be positive");
    const auto compare = [&](const std::vector<std::uint32_t>& right) {
      if constexpr (Metric == NgramMetric::jaccard) {
        return ::stride_align::ngram::jaccard(left, right, n);
      } else if constexpr (Metric == NgramMetric::dice) {
        return ::stride_align::ngram::dice(left, right, n);
      } else if constexpr (Metric == NgramMetric::cosine) {
        return ::stride_align::ngram::cosine(left, right, n);
      } else {
        return ::stride_align::ngram::overlap(left, right, n);
      }
    };
    if constexpr (!Many) {
      return Value::DOUBLE(compare(
          codepoints(adapter::argument(arguments, 1, row))));
    } else {
      const Value list = adapter::argument(arguments, 1, row);
      const auto& values = ListValue::GetChildren(list);
      vector<Value> output;
      output.reserve(values.size());
      for (const Value& value : values) {
        output.push_back(value.IsNull()
            ? Value(LogicalType::DOUBLE)
            : Value::DOUBLE(compare(codepoints(value))));
      }
      return Value::LIST(LogicalType::DOUBLE, std::move(output));
    }
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

double similarity(
    SimilarityMetric metric,
    const std::vector<std::uint32_t>& left,
    const std::vector<std::uint32_t>& right) {
  const auto left_span = std::span<const std::uint32_t>(left.data(), left.size());
  const auto right_span = std::span<const std::uint32_t>(right.data(), right.size());
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

template <SimilarityMetric Metric, bool Many>
void Similarity(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const auto left = codepoints(adapter::argument(arguments, 0, row));
    if constexpr (!Many) {
      return Value::DOUBLE(similarity(
          Metric, left, codepoints(adapter::argument(arguments, 1, row))));
    } else {
      // Keep the owning Value alive while iterating its child references.
      const Value list = adapter::argument(arguments, 1, row);
      const auto& values = ListValue::GetChildren(list);
      vector<Value> output;
      output.reserve(values.size());
      for (const Value& value : values) {
        output.push_back(value.IsNull()
            ? Value(LogicalType::DOUBLE)
            : Value::DOUBLE(similarity(Metric, left, codepoints(value))));
      }
      return Value::LIST(LogicalType::DOUBLE, std::move(output));
    }
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
  const auto pair = ::stride_align::utf8::prepare_pair(left_text, right_text);
  if (name == "jaro") return ::stride_align::core::jaro_similarity(pair);
  if (name == "jaro_winkler") {
    return ::stride_align::core::jaro_winkler_similarity(pair);
  }
  if (name == "levenshtein_ratio" || name == "levenshtein_normalized") {
    return ::stride_align::core::levenshtein_similarity(pair);
  }
  if (name == "indel_ratio" || name == "indel_normalized") {
    return ::stride_align::core::indel_similarity(pair);
  }
  throw std::invalid_argument("unknown Monge-Elkan inner similarity: " +
                              std::string(name));
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
    }
    total += best;
  }
  return total / static_cast<double>(left.size());
}

void MongeElkan(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const auto left_points = codepoints(adapter::argument(arguments, 0, row));
    const auto right_points = codepoints(adapter::argument(arguments, 1, row));
    const auto left = word_tokens(left_points);
    const auto right = word_tokens(right_points);
    if (left.empty() && right.empty()) return Value::DOUBLE(1.0);
    if (left.empty() || right.empty()) return Value::DOUBLE(0.0);
    const std::string inner = arguments.ColumnCount() >= 3
        ? adapter::string_value(adapter::argument(arguments, 2, row))
        : "jaro";
    const bool symmetric = arguments.ColumnCount() >= 4 &&
        adapter::boolean_value(adapter::argument(arguments, 3, row));
    const double forward = monge_direction(left, right, inner);
    return Value::DOUBLE(symmetric
        ? (forward + monge_direction(right, left, inner)) / 2.0
        : forward);
  });
}

template <typename Function>
void UnaryString(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    return Value(Function{}(adapter::string_value(
        adapter::argument(arguments, 0, row))));
  });
}

struct Soundex {
  std::string operator()(std::string_view value) const {
    return ::stride_align::phonetic::soundex(value);
  }
};
struct Nysiis {
  std::string operator()(std::string_view value) const {
    return ::stride_align::phonetic::nysiis(value);
  }
};
struct MatchRatingCodex {
  std::string operator()(std::string_view value) const {
    return ::stride_align::phonetic::match_rating_codex(value);
  }
};
struct Caverphone {
  std::string operator()(std::string_view value) const {
    return ::stride_align::phonetic::caverphone(value);
  }
};

template <typename Function>
void EncodedEqual(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const std::string left = Function{}(adapter::string_value(
        adapter::argument(arguments, 0, row)));
    const std::string right = Function{}(adapter::string_value(
        adapter::argument(arguments, 1, row)));
    return Value::BOOLEAN(!left.empty() && left == right);
  });
}

void Metaphone(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const auto variant = arguments.ColumnCount() >= 2
        ? static_cast<::stride_align::phonetic::MetaphoneVariant>(
              adapter::nonnegative_size(
                  adapter::argument(arguments, 1, row), "variant"))
        : ::stride_align::phonetic::MetaphoneVariant::kPhilips;
    if (variant != ::stride_align::phonetic::MetaphoneVariant::kPhilips &&
        variant != ::stride_align::phonetic::MetaphoneVariant::kJellyfish) {
      throw std::invalid_argument("variant must be 0 or 1");
    }
    return Value(::stride_align::phonetic::metaphone(
        adapter::string_value(adapter::argument(arguments, 0, row)), variant));
  });
}

void MetaphoneEqual(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const auto variant = arguments.ColumnCount() >= 3
        ? static_cast<::stride_align::phonetic::MetaphoneVariant>(
              adapter::nonnegative_size(
                  adapter::argument(arguments, 2, row), "variant"))
        : ::stride_align::phonetic::MetaphoneVariant::kPhilips;
    const std::string left = ::stride_align::phonetic::metaphone(
        adapter::string_value(adapter::argument(arguments, 0, row)), variant);
    const std::string right = ::stride_align::phonetic::metaphone(
        adapter::string_value(adapter::argument(arguments, 1, row)), variant);
    return Value::BOOLEAN(!left.empty() && left == right);
  });
}

void MatchRatingCompare(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    return Value::BOOLEAN(::stride_align::phonetic::match_rating_compare(
        adapter::string_value(adapter::argument(arguments, 0, row)),
        adapter::string_value(adapter::argument(arguments, 1, row))));
  });
}

void ColognePhonetic(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    return Value(::stride_align::phonetic::cologne_phonetic(
        codepoints(adapter::argument(arguments, 0, row))));
  });
}

void DaitchMokotoff(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const bool branching = arguments.ColumnCount() < 2 ||
        adapter::boolean_value(adapter::argument(arguments, 1, row));
    const bool folding = arguments.ColumnCount() < 3 ||
        adapter::boolean_value(adapter::argument(arguments, 2, row));
    return Value(::stride_align::phonetic::daitch_mokotoff(
        codepoints(adapter::argument(arguments, 0, row)), branching, folding));
  });
}

void DoubleMetaphone(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const std::size_t maximum = arguments.ColumnCount() >= 2
        ? adapter::nonnegative_size(
              adapter::argument(arguments, 1, row), "max_length")
        : 64U;
    const auto variant = arguments.ColumnCount() >= 3
        ? static_cast<::stride_align::phonetic::DoubleMetaphoneVariant>(
              adapter::nonnegative_size(
                  adapter::argument(arguments, 2, row), "variant"))
        : ::stride_align::phonetic::DoubleMetaphoneVariant::kCommons;
    const auto encoded = ::stride_align::phonetic::double_metaphone(
        adapter::string_value(adapter::argument(arguments, 0, row)),
        maximum, variant);
    return Value::STRUCT({
        {"primary", Value(encoded.primary)},
        {"alternate", Value(encoded.alternate)},
    });
  });
}

std::optional<std::size_t> dtw_window(
    double input,
    std::size_t query_size,
    std::size_t target_size) {
  if (input < 0.0 || !std::isfinite(input)) {
    throw std::invalid_argument("window must be non-negative and finite");
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

double dtw_score(
    const std::vector<double>& query,
    const std::vector<double>& target,
    std::optional<double> window,
    std::string_view distance,
    std::optional<double> cutoff) {
  const auto kind = distance == "l1"
      ? ::stride_align::dtw::DistanceKind::kL1
      : ::stride_align::dtw::DistanceKind::kL2Squared;
  if (distance != "l1" && distance != "l2_squared") {
    throw std::invalid_argument("distance must be 'l1' or 'l2_squared'");
  }
  const auto resolved = window.has_value()
      ? dtw_window(*window, query.size(), target.size())
      : std::nullopt;
  return ::stride_align::dtw::dtw_score_scalar<double, double>(
      query, target, kind, resolved, cutoff);
}

template <bool IntegerInput>
void Dtw(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const auto query = adapter::double_list(
        adapter::argument(arguments, 0, row), "query");
    const auto target = adapter::double_list(
        adapter::argument(arguments, 1, row), "target");
    std::optional<double> window;
    std::string distance = IntegerInput ? "l1" : "l2_squared";
    std::optional<double> cutoff;
    if (arguments.ColumnCount() >= 3) {
      window = adapter::double_value(adapter::argument(arguments, 2, row));
    }
    if (arguments.ColumnCount() >= 4) {
      distance = adapter::string_value(adapter::argument(arguments, 3, row));
    }
    if (arguments.ColumnCount() >= 5) {
      cutoff = adapter::double_value(adapter::argument(arguments, 4, row));
      if (*cutoff < 0.0) {
        throw std::invalid_argument("score_cutoff must be non-negative");
      }
    }
    return Value::DOUBLE(dtw_score(
        query, target, window, distance, cutoff));
  });
}

template <bool IntegerInput>
void DtwDistances(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const auto query = adapter::double_list(
        adapter::argument(arguments, 0, row), "query");
    const auto targets = adapter::nested_double_list(
        adapter::argument(arguments, 1, row), "targets");
    std::optional<double> window;
    std::string distance = IntegerInput ? "l1" : "l2_squared";
    std::optional<double> cutoff;
    if (arguments.ColumnCount() >= 3) {
      window = adapter::double_value(adapter::argument(arguments, 2, row));
    }
    if (arguments.ColumnCount() >= 4) {
      distance = adapter::string_value(adapter::argument(arguments, 3, row));
    }
    if (arguments.ColumnCount() >= 5) {
      cutoff = adapter::double_value(adapter::argument(arguments, 4, row));
    }
    vector<Value> output;
    output.reserve(targets.size());
    for (const auto& target : targets) {
      output.push_back(Value::DOUBLE(
          dtw_score(query, target, window, distance, cutoff)));
    }
    return Value::LIST(LogicalType::DOUBLE, std::move(output));
  });
}

LogicalType alignment_path_type() {
  return LogicalType::STRUCT({
      {"score", LogicalType::BIGINT},
      {"query_start", LogicalType::UBIGINT},
      {"query_end", LogicalType::UBIGINT},
      {"target_start", LogicalType::UBIGINT},
      {"target_end", LogicalType::UBIGINT},
      {"operations", LogicalType::VARCHAR},
      {"cigar", LogicalType::VARCHAR},
      {"matches", LogicalType::UBIGINT},
      {"mismatches", LogicalType::UBIGINT},
      {"insertions", LogicalType::UBIGINT},
      {"deletions", LogicalType::UBIGINT},
      {"aligned_length", LogicalType::UBIGINT},
      {"aligned_query", LogicalType::VARCHAR},
      {"aligned_target", LogicalType::VARCHAR},
  });
}

Value alignment_path_value(
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
      aligned_query.push_back('-');
      aligned_target.push_back('-');
      aligned_query.back() = query[query_index++];
    } else {
      aligned_query.push_back('-');
      aligned_target.push_back(target[target_index++]);
    }
  }
  vector<Value> fields;
  fields.push_back(Value::BIGINT(path.score));
  fields.push_back(Value::UBIGINT(path.query_start));
  fields.push_back(Value::UBIGINT(path.query_end));
  fields.push_back(Value::UBIGINT(path.target_start));
  fields.push_back(Value::UBIGINT(path.target_end));
  fields.emplace_back(path.operations);
  fields.emplace_back(path.cigar);
  fields.push_back(Value::UBIGINT(path.matches));
  fields.push_back(Value::UBIGINT(path.mismatches));
  fields.push_back(Value::UBIGINT(path.insertions));
  fields.push_back(Value::UBIGINT(path.deletions));
  fields.push_back(Value::UBIGINT(path.aligned_length));
  fields.emplace_back(encode_utf8(aligned_query));
  fields.emplace_back(encode_utf8(aligned_target));
  return Value::STRUCT(alignment_path_type(), std::move(fields));
}

template <bool Local, bool CigarOnly>
void AlignmentPath(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const std::string query_text =
        adapter::string_value(adapter::argument(arguments, 0, row));
    const std::string target_text =
        adapter::string_value(adapter::argument(arguments, 1, row));
    ::stride_align::Score match = 2;
    ::stride_align::Score mismatch = -1;
    ::stride_align::Score gap_open = -1;
    ::stride_align::Score gap_extend = -1;
    if (arguments.ColumnCount() >= 5) {
      match = adapter::score_value(adapter::argument(arguments, 2, row));
      mismatch = adapter::score_value(adapter::argument(arguments, 3, row));
      gap_open = adapter::score_value(adapter::argument(arguments, 4, row));
      gap_extend = arguments.ColumnCount() == 6
          ? adapter::score_value(adapter::argument(arguments, 5, row))
          : gap_open;
    }
    const auto pair = ::stride_align::utf8::prepare_pair(query_text, target_text);
    ::stride_align::AlignmentPath path;
    if (gap_open == gap_extend) {
      if constexpr (Local) {
        path = ::stride_align::core::smith_waterman_path(
            pair, match, mismatch, gap_open);
      } else {
        path = ::stride_align::core::needleman_wunsch_path(
            pair, match, mismatch, gap_open);
      }
    } else {
      if constexpr (Local) {
        path = ::stride_align::core::smith_waterman_affine_path(
            pair, match, mismatch, gap_open, gap_extend);
      } else {
        path = ::stride_align::core::needleman_wunsch_affine_path(
            pair, match, mismatch, gap_open, gap_extend);
      }
    }
    if constexpr (CigarOnly) {
      return Value(path.cigar);
    } else {
      const auto query = ::stride_align::utf8::prepare_streaming(query_text);
      const auto target = ::stride_align::utf8::prepare_streaming(target_text);
      return alignment_path_value(path, query, target);
    }
  });
}

void register_function(
    ExtensionLoader& loader,
    const char* name,
    std::vector<LogicalType> arguments,
    LogicalType result,
    scalar_function_t function) {
  loader.RegisterFunction(ScalarFunction(
      name, std::move(arguments), std::move(result), std::move(function)));
}

void register_with_optional_integer(
    ExtensionLoader& loader,
    const char* name,
    scalar_function_t function,
    LogicalType result = LogicalType::DOUBLE) {
  ScalarFunctionSet set(name);
  set.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::VARCHAR}, result, function));
  set.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT},
      result, function));
  loader.RegisterFunction(std::move(set));
}

template <SimilarityMetric Metric>
void register_similarity_family(
    ExtensionLoader& loader,
    const char* singular,
    const char* plural = nullptr) {
  register_function(
      loader, singular,
      {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::DOUBLE,
      Similarity<Metric, false>);
  if (plural != nullptr) {
    register_function(
        loader, plural,
        {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::VARCHAR)},
        LogicalType::LIST(LogicalType::DOUBLE),
        Similarity<Metric, true>);
  }
}

template <bool Local>
void register_path_family(ExtensionLoader& loader) {
  const char* stem = Local ? "smith_waterman" : "needleman_wunsch";
  const std::vector<std::string> suffixes = {
      "path", "path_info"};
  for (const auto& suffix : suffixes) {
    ScalarFunctionSet set("stride_" + std::string(stem) + "_" + suffix);
    set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR},
        alignment_path_type(), AlignmentPath<Local, false>));
    set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
        alignment_path_type(), AlignmentPath<Local, false>));
    set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::BIGINT, LogicalType::BIGINT,
         LogicalType::BIGINT, LogicalType::BIGINT},
        alignment_path_type(), AlignmentPath<Local, false>));
    loader.RegisterFunction(std::move(set));
  }
  for (const auto& suffix : {"cigar", "trace_cigar", "trade_cigar"}) {
    ScalarFunctionSet set("stride_" + std::string(stem) + "_" + suffix);
    set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::VARCHAR, AlignmentPath<Local, true>));
    set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::VARCHAR, AlignmentPath<Local, true>));
    set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::BIGINT, LogicalType::BIGINT,
         LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::VARCHAR, AlignmentPath<Local, true>));
    loader.RegisterFunction(std::move(set));
  }
}

}  // namespace

void RegisterAlgorithmFunctions(ExtensionLoader& loader) {
  register_function(
      loader, "stride_lcs_length",
      {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BIGINT,
      ExecuteCodepointInteger<LcsLength>);
  register_function(
      loader, "stride_lcs_substring_length",
      {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BIGINT,
      ExecuteCodepointInteger<LcsSubstringLength>);
  register_function(
      loader, "stride_lcs_substring",
      {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR,
      LcsSubstring);

  const LogicalType strings = LogicalType::LIST(LogicalType::VARCHAR);
#define STRIDE_NGRAM(metric)                                                     \
  register_with_optional_integer(                                               \
      loader, "stride_" #metric, Ngram<NgramMetric::metric, false>);            \
  register_function(                                                            \
      loader, "stride_" #metric "_similarities",                               \
      {LogicalType::VARCHAR, strings}, LogicalType::LIST(LogicalType::DOUBLE),   \
      Ngram<NgramMetric::metric, true>);                                         \
  register_function(                                                            \
      loader, "stride_" #metric "_similarities",                               \
      {LogicalType::VARCHAR, strings, LogicalType::BIGINT},                      \
      LogicalType::LIST(LogicalType::DOUBLE), Ngram<NgramMetric::metric, true>)
  STRIDE_NGRAM(jaccard);
  STRIDE_NGRAM(dice);
  STRIDE_NGRAM(cosine);
  STRIDE_NGRAM(overlap);
#undef STRIDE_NGRAM

  register_similarity_family<SimilarityMetric::ratcliff>(
      loader, "stride_ratcliff_obershelp_similarity",
      "stride_ratcliff_obershelp_similarities");
  register_similarity_family<SimilarityMetric::partial>(
      loader, "stride_partial_ratio", "stride_partial_ratios");
  register_similarity_family<SimilarityMetric::token_sort>(
      loader, "stride_token_sort_ratio", "stride_token_sort_ratios");
  register_similarity_family<SimilarityMetric::token_set>(
      loader, "stride_token_set_ratio", "stride_token_set_ratios");
  register_similarity_family<SimilarityMetric::partial_token_sort>(
      loader, "stride_partial_token_sort_ratio",
      "stride_partial_token_sort_ratios");
  register_similarity_family<SimilarityMetric::partial_token_set>(
      loader, "stride_partial_token_set_ratio",
      "stride_partial_token_set_ratios");
  register_similarity_family<SimilarityMetric::weighted>(
      loader, "stride_wratio", "stride_wratios");

  ScalarFunctionSet monge("stride_monge_elkan");
  monge.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::VARCHAR},
      LogicalType::DOUBLE, MongeElkan));
  monge.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
      LogicalType::DOUBLE, MongeElkan));
  monge.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::VARCHAR,
       LogicalType::VARCHAR, LogicalType::BOOLEAN},
      LogicalType::DOUBLE, MongeElkan));
  loader.RegisterFunction(std::move(monge));

  register_function(loader, "stride_soundex", {LogicalType::VARCHAR},
                    LogicalType::VARCHAR, UnaryString<Soundex>);
  register_function(loader, "stride_soundex_equal",
                    {LogicalType::VARCHAR, LogicalType::VARCHAR},
                    LogicalType::BOOLEAN, EncodedEqual<Soundex>);
  register_function(loader, "stride_nysiis", {LogicalType::VARCHAR},
                    LogicalType::VARCHAR, UnaryString<Nysiis>);
  register_function(loader, "stride_nysiis_equal",
                    {LogicalType::VARCHAR, LogicalType::VARCHAR},
                    LogicalType::BOOLEAN, EncodedEqual<Nysiis>);
  register_function(loader, "stride_match_rating_codex", {LogicalType::VARCHAR},
                    LogicalType::VARCHAR, UnaryString<MatchRatingCodex>);
  register_function(loader, "stride_match_rating_compare",
                    {LogicalType::VARCHAR, LogicalType::VARCHAR},
                    LogicalType::BOOLEAN, MatchRatingCompare);
  register_function(loader, "stride_caverphone", {LogicalType::VARCHAR},
                    LogicalType::VARCHAR, UnaryString<Caverphone>);
  register_function(loader, "stride_cologne_phonetic", {LogicalType::VARCHAR},
                    LogicalType::VARCHAR, ColognePhonetic);

  ScalarFunctionSet metaphone("stride_metaphone");
  metaphone.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR}, LogicalType::VARCHAR, Metaphone));
  metaphone.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::BIGINT},
      LogicalType::VARCHAR, Metaphone));
  loader.RegisterFunction(std::move(metaphone));
  ScalarFunctionSet metaphone_equal("stride_metaphone_equal");
  metaphone_equal.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::VARCHAR},
      LogicalType::BOOLEAN, MetaphoneEqual));
  metaphone_equal.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT},
      LogicalType::BOOLEAN, MetaphoneEqual));
  loader.RegisterFunction(std::move(metaphone_equal));

  ScalarFunctionSet daitch("stride_daitch_mokotoff");
  daitch.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR}, LogicalType::VARCHAR, DaitchMokotoff));
  daitch.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::BOOLEAN},
      LogicalType::VARCHAR, DaitchMokotoff));
  daitch.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::BOOLEAN},
      LogicalType::VARCHAR, DaitchMokotoff));
  loader.RegisterFunction(std::move(daitch));

  const LogicalType double_metaphone_type = LogicalType::STRUCT({
      {"primary", LogicalType::VARCHAR}, {"alternate", LogicalType::VARCHAR}});
  ScalarFunctionSet double_metaphone("stride_double_metaphone");
  double_metaphone.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR}, double_metaphone_type, DoubleMetaphone));
  double_metaphone.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::BIGINT},
      double_metaphone_type, DoubleMetaphone));
  double_metaphone.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT},
      double_metaphone_type, DoubleMetaphone));
  loader.RegisterFunction(std::move(double_metaphone));

  const LogicalType numbers = LogicalType::LIST(LogicalType::DOUBLE);
  const LogicalType integers = LogicalType::LIST(LogicalType::SMALLINT);
  ScalarFunctionSet dtw("stride_dtw");
  dtw.AddFunction(ScalarFunction(
      {numbers, numbers}, LogicalType::DOUBLE, Dtw<false>));
  dtw.AddFunction(ScalarFunction(
      {numbers, numbers, LogicalType::DOUBLE},
      LogicalType::DOUBLE, Dtw<false>));
  dtw.AddFunction(ScalarFunction(
      {numbers, numbers, LogicalType::DOUBLE, LogicalType::VARCHAR},
      LogicalType::DOUBLE, Dtw<false>));
  dtw.AddFunction(ScalarFunction(
      {numbers, numbers, LogicalType::DOUBLE, LogicalType::VARCHAR,
       LogicalType::DOUBLE},
      LogicalType::DOUBLE, Dtw<false>));
  dtw.AddFunction(ScalarFunction(
      {integers, integers}, LogicalType::DOUBLE, Dtw<true>));
  dtw.AddFunction(ScalarFunction(
      {integers, integers, LogicalType::DOUBLE},
      LogicalType::DOUBLE, Dtw<true>));
  dtw.AddFunction(ScalarFunction(
      {integers, integers, LogicalType::DOUBLE, LogicalType::VARCHAR},
      LogicalType::DOUBLE, Dtw<true>));
  dtw.AddFunction(ScalarFunction(
      {integers, integers, LogicalType::DOUBLE, LogicalType::VARCHAR,
       LogicalType::DOUBLE}, LogicalType::DOUBLE, Dtw<true>));
  loader.RegisterFunction(std::move(dtw));
  ScalarFunctionSet dtw_distances("stride_dtw_distances");
  dtw_distances.AddFunction(ScalarFunction(
      {numbers, LogicalType::LIST(numbers)},
      LogicalType::LIST(LogicalType::DOUBLE), DtwDistances<false>));
  dtw_distances.AddFunction(ScalarFunction(
      {numbers, LogicalType::LIST(numbers), LogicalType::DOUBLE},
      LogicalType::LIST(LogicalType::DOUBLE), DtwDistances<false>));
  dtw_distances.AddFunction(ScalarFunction(
      {numbers, LogicalType::LIST(numbers), LogicalType::DOUBLE,
       LogicalType::VARCHAR},
      LogicalType::LIST(LogicalType::DOUBLE), DtwDistances<false>));
  dtw_distances.AddFunction(ScalarFunction(
      {numbers, LogicalType::LIST(numbers), LogicalType::DOUBLE,
       LogicalType::VARCHAR, LogicalType::DOUBLE},
      LogicalType::LIST(LogicalType::DOUBLE), DtwDistances<false>));
  dtw_distances.AddFunction(ScalarFunction(
      {integers, LogicalType::LIST(integers)},
      LogicalType::LIST(LogicalType::DOUBLE), DtwDistances<true>));
  dtw_distances.AddFunction(ScalarFunction(
      {integers, LogicalType::LIST(integers), LogicalType::DOUBLE},
      LogicalType::LIST(LogicalType::DOUBLE), DtwDistances<true>));
  dtw_distances.AddFunction(ScalarFunction(
      {integers, LogicalType::LIST(integers), LogicalType::DOUBLE,
       LogicalType::VARCHAR},
      LogicalType::LIST(LogicalType::DOUBLE), DtwDistances<true>));
  dtw_distances.AddFunction(ScalarFunction(
      {integers, LogicalType::LIST(integers), LogicalType::DOUBLE,
       LogicalType::VARCHAR, LogicalType::DOUBLE},
      LogicalType::LIST(LogicalType::DOUBLE), DtwDistances<true>));
  loader.RegisterFunction(std::move(dtw_distances));

  register_path_family<true>(loader);
  register_path_family<false>(loader);
}

}  // namespace duckdb::stride_align_extension
