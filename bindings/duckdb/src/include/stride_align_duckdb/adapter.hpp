#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"

#include "stride_align/batch.hpp"

namespace duckdb::stride_align_adapter {

inline Value argument(DataChunk& arguments, idx_t column, idx_t row) {
  return arguments.data[column].GetValue(row);
}

inline std::string string_value(const Value& value) {
  return StringValue::Get(value);
}

inline std::int64_t integer_value(const Value& value) {
  return value.GetValue<std::int64_t>();
}

inline std::uint64_t unsigned_value(const Value& value) {
  return value.GetValue<std::uint64_t>();
}

inline double double_value(const Value& value) {
  return value.GetValue<double>();
}

inline bool boolean_value(const Value& value) {
  return value.GetValue<bool>();
}

inline std::size_t nonnegative_size(const Value& value, const char* name) {
  const std::int64_t input = integer_value(value);
  if (input < 0) {
    throw std::invalid_argument(std::string(name) + " must be non-negative");
  }
  return static_cast<std::size_t>(input);
}

inline ::stride_align::Score score_value(const Value& value) {
  return static_cast<::stride_align::Score>(integer_value(value));
}

template <typename Function>
void execute_rows(
    DataChunk& arguments,
    Vector& result,
    Function&& function,
    bool automatic_nulls = true) {
  arguments.Flatten();
  for (idx_t row = 0; row < arguments.size(); ++row) {
    if (automatic_nulls) {
      bool has_null = false;
      for (idx_t column = 0; column < arguments.ColumnCount(); ++column) {
        if (argument(arguments, column, row).IsNull()) {
          has_null = true;
          break;
        }
      }
      if (has_null) {
        result.SetValue(row, Value());
        continue;
      }
    }
    try {
      result.SetValue(row, function(row));
    } catch (const InvalidInputException&) {
      throw;
    } catch (const std::exception& error) {
      throw InvalidInputException(error.what());
    }
  }
}

inline std::vector<std::optional<::stride_align::batch::Text>> text_list(
    const Value& value) {
  const auto& children = ListValue::GetChildren(value);
  std::vector<std::optional<::stride_align::batch::Text>> output;
  output.reserve(children.size());
  for (const Value& child : children) {
    if (child.IsNull()) {
      output.emplace_back(std::nullopt);
    } else {
      output.emplace_back(::stride_align::batch::Text(string_value(child)));
    }
  }
  return output;
}

inline std::vector<double> double_list(const Value& value, const char* name) {
  const auto& children = ListValue::GetChildren(value);
  if (children.empty()) {
    throw std::invalid_argument(std::string(name) + " must be non-empty");
  }
  std::vector<double> output;
  output.reserve(children.size());
  for (const Value& child : children) {
    if (child.IsNull()) {
      throw std::invalid_argument(
          std::string(name) + " cannot contain NULL values");
    }
    const double number = double_value(child);
    if (!std::isfinite(number)) {
      throw std::invalid_argument(
          std::string(name) + " must contain finite values");
    }
    output.push_back(number);
  }
  return output;
}

inline std::vector<std::vector<double>> nested_double_list(
    const Value& value,
    const char* name) {
  const auto& children = ListValue::GetChildren(value);
  std::vector<std::vector<double>> output;
  output.reserve(children.size());
  for (const Value& child : children) {
    if (child.IsNull()) {
      throw std::invalid_argument(
          std::string(name) + " cannot contain NULL sequences");
    }
    output.push_back(double_list(child, name));
  }
  return output;
}

inline LogicalType ranked_match_type(LogicalType score_type) {
  return LogicalType::STRUCT({
      {"target", LogicalType::VARCHAR},
      {"score", std::move(score_type)},
      {"index", LogicalType::UBIGINT},
  });
}

inline LogicalType matrix_match_type() {
  return LogicalType::STRUCT({
      {"score", LogicalType::DOUBLE},
      {"query", LogicalType::VARCHAR},
      {"target", LogicalType::VARCHAR},
      {"query_index", LogicalType::UBIGINT},
      {"target_index", LogicalType::UBIGINT},
  });
}

inline LogicalType per_query_match_type() {
  const auto child = LogicalType::STRUCT({
      {"score", LogicalType::DOUBLE},
      {"target", LogicalType::VARCHAR},
      {"target_index", LogicalType::UBIGINT},
  });
  return LogicalType::STRUCT({
      {"query", LogicalType::VARCHAR},
      {"query_index", LogicalType::UBIGINT},
      {"matches", LogicalType::LIST(child)},
  });
}

inline Value numeric_value(double value, bool integral) {
  return integral
      ? Value::BIGINT(static_cast<std::int64_t>(value))
      : Value::DOUBLE(value);
}

inline Value score_list(
    const std::vector<std::optional<double>>& scores,
    bool integral) {
  vector<Value> values;
  values.reserve(scores.size());
  const LogicalType child_type = integral
      ? LogicalType::BIGINT
      : LogicalType::DOUBLE;
  for (const auto& score : scores) {
    values.push_back(score.has_value()
        ? numeric_value(*score, integral)
        : Value(child_type));
  }
  return Value::LIST(child_type, std::move(values));
}

inline Value ranked_value(
    const ::stride_align::batch::RankedMatch& match,
    std::span<const std::optional<::stride_align::batch::Text>> targets,
    bool integral) {
  const LogicalType type = ranked_match_type(
      integral ? LogicalType::BIGINT : LogicalType::DOUBLE);
  vector<Value> fields;
  fields.emplace_back(targets[match.index]->bytes);
  fields.push_back(numeric_value(match.score, integral));
  fields.push_back(Value::UBIGINT(match.index));
  return Value::STRUCT(type, std::move(fields));
}

inline Value ranked_list(
    const std::vector<::stride_align::batch::RankedMatch>& matches,
    std::span<const std::optional<::stride_align::batch::Text>> targets,
    bool integral) {
  const LogicalType child_type = ranked_match_type(
      integral ? LogicalType::BIGINT : LogicalType::DOUBLE);
  vector<Value> output;
  output.reserve(matches.size());
  for (const auto& match : matches) {
    output.push_back(ranked_value(match, targets, integral));
  }
  return Value::LIST(child_type, std::move(output));
}

inline Value distance_matrix_value(
    const ::stride_align::batch::DistanceMatrix& matrix) {
  const LogicalType row_type = LogicalType::LIST(LogicalType::DOUBLE);
  vector<Value> rows;
  rows.reserve(matrix.size());
  for (const auto& row : matrix) {
    vector<Value> values;
    values.reserve(row.size());
    for (const auto& score : row) {
      values.push_back(score.has_value()
          ? Value::DOUBLE(*score)
          : Value(LogicalType::DOUBLE));
    }
    rows.push_back(Value::LIST(LogicalType::DOUBLE, std::move(values)));
  }
  return Value::LIST(row_type, std::move(rows));
}

inline Value matrix_matches_value(
    const std::vector<::stride_align::batch::MatrixMatch>& matches,
    std::span<const std::optional<::stride_align::batch::Text>> queries,
    std::span<const std::optional<::stride_align::batch::Text>> targets) {
  const LogicalType child_type = matrix_match_type();
  vector<Value> output;
  output.reserve(matches.size());
  for (const auto& match : matches) {
    vector<Value> fields;
    fields.push_back(Value::DOUBLE(match.score));
    fields.emplace_back(queries[match.query_index]->bytes);
    fields.emplace_back(targets[match.target_index]->bytes);
    fields.push_back(Value::UBIGINT(match.query_index));
    fields.push_back(Value::UBIGINT(match.target_index));
    output.push_back(Value::STRUCT(child_type, std::move(fields)));
  }
  return Value::LIST(child_type, std::move(output));
}

inline Value per_query_matches_value(
    const std::vector<std::vector<::stride_align::batch::RankedMatch>>& matches,
    std::span<const std::optional<::stride_align::batch::Text>> queries,
    std::span<const std::optional<::stride_align::batch::Text>> targets) {
  const LogicalType match_type = LogicalType::STRUCT({
      {"score", LogicalType::DOUBLE},
      {"target", LogicalType::VARCHAR},
      {"target_index", LogicalType::UBIGINT},
  });
  const LogicalType child_type = per_query_match_type();
  vector<Value> output;
  output.reserve(matches.size());
  for (std::size_t query_index = 0; query_index < matches.size(); ++query_index) {
    vector<Value> child_matches;
    child_matches.reserve(matches[query_index].size());
    for (const auto& match : matches[query_index]) {
      vector<Value> fields;
      fields.push_back(Value::DOUBLE(match.score));
      fields.emplace_back(targets[match.index]->bytes);
      fields.push_back(Value::UBIGINT(match.index));
      child_matches.push_back(Value::STRUCT(match_type, std::move(fields)));
    }
    vector<Value> fields;
    if (queries[query_index].has_value()) {
      fields.emplace_back(queries[query_index]->bytes);
    } else {
      fields.emplace_back(LogicalType::VARCHAR);
    }
    fields.push_back(Value::UBIGINT(query_index));
    fields.push_back(Value::LIST(match_type, std::move(child_matches)));
    output.push_back(Value::STRUCT(child_type, std::move(fields)));
  }
  return Value::LIST(child_type, std::move(output));
}

}  // namespace duckdb::stride_align_adapter
