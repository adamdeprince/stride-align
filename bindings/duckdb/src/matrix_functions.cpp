#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"

#include "embedded_data.hpp"
#include "stride_align/beider_morse.hpp"
#include "stride_align/core.hpp"
#include "stride_align_duckdb/adapter.hpp"
#include "stride_align_duckdb/registration.hpp"

namespace duckdb::stride_align_extension {
namespace {

namespace adapter = ::duckdb::stride_align_adapter;

struct Matrix {
  std::string sql_name;
  std::string name;
  std::string alphabet;
  std::uint8_t wildcard = 0;
  ::stride_align::Score gap_score = -1;
  ::stride_align::Score gap_open = 0;
  ::stride_align::Score gap_extend = 0;
  bool has_affine = false;
  std::vector<std::int8_t> values;
  std::unordered_map<std::uint32_t, std::uint16_t> indices;

  std::size_t stride() const noexcept { return indices.size(); }

  std::int8_t lookup(std::uint16_t left, std::uint16_t right) const {
    return values[static_cast<std::size_t>(left) * stride() + right];
  }
};

::stride_align::Score matrix_score(
    const Matrix& matrix,
    std::string_view query,
    std::string_view target,
    bool local,
    ::stride_align::Score gap_open,
    ::stride_align::Score gap_extend);

std::vector<std::uint32_t> decode(std::string_view input);
std::string encode_utf8(std::span<const std::uint32_t> input);

LogicalType matrix_type() {
  return LogicalType::STRUCT({
      {"name", LogicalType::VARCHAR},
      {"alphabet", LogicalType::VARCHAR},
      {"scores", LogicalType::BLOB},
      {"wildcard", LogicalType::VARCHAR},
      {"gap_score", LogicalType::BIGINT},
      {"gap_open", LogicalType::BIGINT},
      {"gap_extend", LogicalType::BIGINT},
  });
}

std::vector<LogicalType> matrix_input_types() {
  return {LogicalType(LogicalType::VARCHAR), matrix_type()};
}

Matrix finish_matrix(Matrix matrix, std::string_view wildcard) {
  const auto alphabet = decode(matrix.alphabet);
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
  const auto wildcard_points = decode(wildcard);
  if (wildcard_points.size() != 1U ||
      !matrix.indices.contains(wildcard_points.front())) {
    throw std::invalid_argument(
        "substitution-matrix wildcard must be one symbol in the alphabet");
  }
  matrix.wildcard = static_cast<std::uint8_t>(
      matrix.indices.at(wildcard_points.front()));
  if (matrix.values.size() != alphabet.size() * alphabet.size()) {
    throw std::invalid_argument(
        "substitution-matrix score grid does not match the alphabet");
  }
  return matrix;
}

std::string matrix_wildcard(const Matrix& matrix) {
  const auto alphabet = decode(matrix.alphabet);
  return encode_utf8(std::span<const std::uint32_t>(
      alphabet.data() + matrix.wildcard, 1U));
}

Value matrix_value(const Matrix& matrix) {
  std::string scores;
  scores.resize(matrix.values.size());
  for (std::size_t index = 0; index < matrix.values.size(); ++index) {
    scores[index] = static_cast<char>(matrix.values[index]);
  }
  vector<Value> fields;
  fields.emplace_back(matrix.name);
  fields.emplace_back(matrix.alphabet);
  fields.push_back(Value::BLOB_RAW(scores));
  fields.emplace_back(matrix_wildcard(matrix));
  fields.push_back(Value::BIGINT(matrix.gap_score));
  fields.push_back(matrix.has_affine
      ? Value::BIGINT(matrix.gap_open) : Value(LogicalType::BIGINT));
  fields.push_back(matrix.has_affine
      ? Value::BIGINT(matrix.gap_extend) : Value(LogicalType::BIGINT));
  return Value::STRUCT(matrix_type(), std::move(fields));
}

Matrix matrix_from_value(const Value& value) {
  const auto& fields = StructValue::GetChildren(value);
  if (fields.size() != 7U) {
    throw std::invalid_argument("invalid stride-align matrix value");
  }
  Matrix matrix;
  matrix.sql_name = adapter::string_value(fields[0]);
  matrix.name = matrix.sql_name;
  matrix.alphabet = adapter::string_value(fields[1]);
  const std::string scores = StringValue::Get(fields[2]);
  matrix.values.reserve(scores.size());
  for (const unsigned char byte : scores) {
    matrix.values.push_back(static_cast<std::int8_t>(byte));
  }
  matrix.gap_score = adapter::score_value(fields[4]);
  if (fields[5].IsNull() != fields[6].IsNull()) {
    throw std::invalid_argument(
        "matrix gap_open and gap_extend must both be set or both be NULL");
  }
  if (!fields[5].IsNull()) {
    matrix.has_affine = true;
    matrix.gap_open = adapter::score_value(fields[5]);
    matrix.gap_extend = adapter::score_value(fields[6]);
  }
  return finish_matrix(matrix, adapter::string_value(fields[3]));
}

std::vector<std::uint32_t> decode(std::string_view input) {
  return ::stride_align::utf8::prepare_streaming(input);
}

std::string encode_utf8(std::span<const std::uint32_t> input) {
  std::string output;
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

std::vector<Matrix> build_matrices() {
  std::vector<Matrix> output;
  for (auto& record : ::duckdb::stride_align_embedded::matrices()) {
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
  throw std::invalid_argument("unknown stride-align matrix: " +
                              std::string(input));
}

struct MatrixArgument {
  const Matrix* borrowed = nullptr;
  std::optional<Matrix> owned;

  const Matrix& get() const noexcept {
    return owned.has_value() ? *owned : *borrowed;
  }
};

MatrixArgument matrix_argument(const Value& input) {
  if (input.type().id() == LogicalTypeId::STRUCT) {
    MatrixArgument result;
    result.owned = matrix_from_value(input);
    return result;
  }
  const std::string name = adapter::string_value(input);
  MatrixArgument result;
  result.borrowed = &resolve_matrix(std::string_view(name));
  return result;
}

Matrix identity_matrix(
    std::string alphabet,
    ::stride_align::Score match,
    ::stride_align::Score mismatch,
    std::string wildcard,
    std::string name,
    ::stride_align::Score gap_score,
    std::optional<::stride_align::Score> gap_open,
    std::optional<::stride_align::Score> gap_extend) {
  auto symbols = decode(alphabet);
  const auto wildcard_points = decode(wildcard);
  if (wildcard_points.size() != 1U) {
    throw std::invalid_argument("wildcard must be one Unicode symbol");
  }
  if (std::find(symbols.begin(), symbols.end(), wildcard_points.front()) ==
      symbols.end()) {
    symbols.push_back(wildcard_points.front());
    alphabet = encode_utf8(symbols);
  }
  if (match < -128 || match > 127 || mismatch < -128 || mismatch > 127) {
    throw std::invalid_argument("matrix scores must fit signed int8");
  }
  Matrix matrix;
  matrix.sql_name = name;
  matrix.name = std::move(name);
  matrix.alphabet = std::move(alphabet);
  matrix.gap_score = gap_score;
  matrix.has_affine = gap_open.has_value() || gap_extend.has_value();
  if (matrix.has_affine) {
    if (!gap_open.has_value() || !gap_extend.has_value()) {
      throw std::invalid_argument(
          "gap_open and gap_extend must both be set or both be NULL");
    }
    matrix.gap_open = *gap_open;
    matrix.gap_extend = *gap_extend;
  }
  matrix.values.assign(symbols.size() * symbols.size(),
                       static_cast<std::int8_t>(mismatch));
  for (std::size_t index = 0; index < symbols.size(); ++index) {
    matrix.values[index * symbols.size() + index] =
        static_cast<std::int8_t>(match);
  }
  const std::size_t wildcard_index = static_cast<std::size_t>(std::find(
      symbols.begin(), symbols.end(), wildcard_points.front()) - symbols.begin());
  for (std::size_t index = 0; index < symbols.size(); ++index) {
    matrix.values[wildcard_index * symbols.size() + index] =
        static_cast<std::int8_t>(mismatch);
    matrix.values[index * symbols.size() + wildcard_index] =
        static_cast<std::int8_t>(mismatch);
  }
  return finish_matrix(std::move(matrix), wildcard);
}

std::optional<::stride_align::Score> optional_score(const Value& value) {
  return value.IsNull()
      ? std::nullopt
      : std::optional<::stride_align::Score>(adapter::score_value(value));
}

std::string default_ascii_alphabet() {
  std::string alphabet;
  alphabet.reserve(128U);
  for (std::uint32_t value = 0; value < 128U; ++value) {
    alphabet.push_back(static_cast<char>(value));
  }
  return alphabet;
}

std::string optional_string(
    DataChunk& arguments,
    idx_t row,
    idx_t column,
    std::string fallback) {
  return arguments.ColumnCount() > column
      ? adapter::string_value(adapter::argument(arguments, column, row))
      : std::move(fallback);
}

::stride_align::Score optional_integer(
    DataChunk& arguments,
    idx_t row,
    idx_t column,
    ::stride_align::Score fallback) {
  return arguments.ColumnCount() > column
      ? adapter::score_value(adapter::argument(arguments, column, row))
      : fallback;
}

std::optional<::stride_align::Score> optional_nullable_integer(
    DataChunk& arguments,
    idx_t row,
    idx_t column) {
  return arguments.ColumnCount() > column
      ? optional_score(adapter::argument(arguments, column, row))
      : std::nullopt;
}

void IdentityMatrix(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    return matrix_value(identity_matrix(
        adapter::string_value(adapter::argument(arguments, 0, row)),
        optional_integer(arguments, row, 1, 1),
        optional_integer(arguments, row, 2, 0),
        optional_string(arguments, row, 3, "*"),
        optional_string(arguments, row, 4, "IDENTITY"),
        optional_integer(arguments, row, 5, -1),
        optional_nullable_integer(arguments, row, 6),
        optional_nullable_integer(arguments, row, 7)));
  }, false);
}

void AsciiMatrix(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    return matrix_value(identity_matrix(
        default_ascii_alphabet(),
        optional_integer(arguments, row, 0, 1),
        optional_integer(arguments, row, 1, -1),
        std::string(1U, static_cast<char>(127)),
        optional_string(arguments, row, 2, "ASCII"),
        optional_integer(arguments, row, 3, -1),
        optional_nullable_integer(arguments, row, 4),
        optional_nullable_integer(arguments, row, 5)));
  }, false);
}

void MatrixFromNcbiText(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const std::string text =
        adapter::string_value(adapter::argument(arguments, 0, row));
    std::istringstream stream(text);
    std::string line;
    std::vector<std::string> header;
    std::vector<std::int8_t> values;
    std::vector<std::string> row_labels;
    while (std::getline(stream, line)) {
      const std::size_t first = line.find_first_not_of(" \t\r");
      if (first == std::string::npos || line[first] == '#') continue;
      std::istringstream tokens(line);
      std::vector<std::string> fields;
      std::string field;
      while (tokens >> field) fields.push_back(std::move(field));
      if (header.empty()) {
        header = std::move(fields);
        continue;
      }
      if (fields.size() != header.size() + 1U) {
        throw std::invalid_argument(
            "NCBI matrix row has the wrong number of values");
      }
      row_labels.push_back(fields.front());
      for (std::size_t index = 1; index < fields.size(); ++index) {
        std::size_t consumed = 0;
        const long parsed = std::stol(fields[index], &consumed);
        if (consumed != fields[index].size() || parsed < -128 || parsed > 127) {
          throw std::invalid_argument(
              "NCBI matrix score must be a signed int8 integer");
        }
        values.push_back(static_cast<std::int8_t>(parsed));
      }
    }
    if (header.empty() || row_labels != header) {
      throw std::invalid_argument(
          "NCBI matrix row labels do not match the column header");
    }
    Matrix matrix;
    matrix.name = optional_string(arguments, row, 1, "<unnamed>");
    matrix.sql_name = matrix.name;
    for (const std::string& symbol : header) matrix.alphabet += symbol;
    matrix.values = std::move(values);
    matrix.gap_score = optional_integer(arguments, row, 2, -4);
    const auto gap_open = optional_nullable_integer(arguments, row, 4);
    const auto gap_extend = optional_nullable_integer(arguments, row, 5);
    matrix.has_affine = gap_open.has_value() || gap_extend.has_value();
    if (matrix.has_affine) {
      if (!gap_open.has_value() || !gap_extend.has_value()) {
        throw std::invalid_argument(
            "gap_open and gap_extend must both be set or both be NULL");
      }
      matrix.gap_open = *gap_open;
      matrix.gap_extend = *gap_extend;
    }
    return matrix_value(finish_matrix(
        std::move(matrix),
        optional_string(arguments, row, 3, "X")));
  }, false);
}

void SubstitutionMatrix(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    Matrix matrix;
    matrix.name = adapter::string_value(adapter::argument(arguments, 0, row));
    matrix.sql_name = matrix.name;
    matrix.alphabet = adapter::string_value(adapter::argument(arguments, 1, row));
    const Value score_grid = adapter::argument(arguments, 2, row);
    const auto& score_rows = ListValue::GetChildren(score_grid);
    const auto symbols = decode(matrix.alphabet);
    if (score_rows.size() != symbols.size()) {
      throw std::invalid_argument(
          "substitution-matrix row count does not match the alphabet");
    }
    matrix.values.reserve(symbols.size() * symbols.size());
    for (const Value& score_row : score_rows) {
      if (score_row.IsNull()) {
        throw std::invalid_argument(
            "substitution-matrix score grid cannot contain NULL rows");
      }
      const auto& values = ListValue::GetChildren(score_row);
      if (values.size() != symbols.size()) {
        throw std::invalid_argument(
            "substitution-matrix column count does not match the alphabet");
      }
      for (const Value& value : values) {
        if (value.IsNull()) {
          throw std::invalid_argument(
              "substitution-matrix score grid cannot contain NULL values");
        }
        const auto score = adapter::score_value(value);
        if (score < -128 || score > 127) {
          throw std::invalid_argument(
              "substitution-matrix scores must fit signed int8");
        }
        matrix.values.push_back(static_cast<std::int8_t>(score));
      }
    }
    matrix.gap_score = optional_integer(arguments, row, 3, -4);
    const std::string wildcard = optional_string(arguments, row, 4, "X");
    const auto gap_open = optional_nullable_integer(arguments, row, 5);
    const auto gap_extend = optional_nullable_integer(arguments, row, 6);
    matrix.has_affine = gap_open.has_value() || gap_extend.has_value();
    if (matrix.has_affine) {
      if (!gap_open.has_value() || !gap_extend.has_value()) {
        throw std::invalid_argument(
            "gap_open and gap_extend must both be set or both be NULL");
      }
      matrix.gap_open = *gap_open;
      matrix.gap_extend = *gap_extend;
    }
    return matrix_value(finish_matrix(std::move(matrix), wildcard));
  }, false);
}

Matrix matrix_from_npy(
    std::string_view bytes,
    std::string alphabet,
    std::string name,
    std::string wildcard,
    bool transpose,
    ::stride_align::Score gap_score,
    std::optional<::stride_align::Score> gap_open,
    std::optional<::stride_align::Score> gap_extend) {
  constexpr unsigned char kMagic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
  if (bytes.size() < 10U ||
      std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) != 0) {
    throw std::invalid_argument("not a NumPy .npy file");
  }
  const auto byte = [&](std::size_t index) {
    return static_cast<std::uint32_t>(
        static_cast<unsigned char>(bytes[index]));
  };
  const std::uint32_t major = byte(6U);
  std::size_t header_offset = 0U;
  std::size_t header_size = 0U;
  if (major == 1U) {
    header_offset = 10U;
    header_size = byte(8U) | (byte(9U) << 8U);
  } else if (major == 2U || major == 3U) {
    if (bytes.size() < 12U) throw std::invalid_argument("truncated .npy header");
    header_offset = 12U;
    header_size = byte(8U) | (byte(9U) << 8U) |
        (byte(10U) << 16U) | (byte(11U) << 24U);
  } else {
    throw std::invalid_argument("unsupported NumPy .npy version");
  }
  if (header_offset + header_size > bytes.size()) {
    throw std::invalid_argument("truncated .npy header");
  }
  const std::string_view header = bytes.substr(header_offset, header_size);
  if (header.find("'descr': '|i1'") == std::string_view::npos &&
      header.find("\"descr\": \"|i1\"") == std::string_view::npos &&
      header.find("'descr': '<i1'") == std::string_view::npos &&
      header.find("\"descr\": \"<i1\"") == std::string_view::npos) {
    throw std::invalid_argument("keyboard .npy matrix must use signed int8 values");
  }
  if (header.find("fortran_order')") != std::string_view::npos ||
      header.find("'fortran_order': True") != std::string_view::npos ||
      header.find("\"fortran_order\": true") != std::string_view::npos) {
    throw std::invalid_argument("Fortran-order .npy matrices are not supported");
  }
  const std::size_t shape_key = header.find("shape");
  const std::size_t opening = shape_key == std::string_view::npos
      ? std::string_view::npos : header.find('(', shape_key);
  const std::size_t comma = opening == std::string_view::npos
      ? std::string_view::npos : header.find(',', opening + 1U);
  const std::size_t closing = comma == std::string_view::npos
      ? std::string_view::npos : header.find(')', comma + 1U);
  if (opening == std::string_view::npos || comma == std::string_view::npos ||
      closing == std::string_view::npos) {
    throw std::invalid_argument("could not read shape from keyboard .npy file");
  }
  const auto parse_dimension = [](std::string_view value) {
    const std::size_t first = value.find_first_not_of(" \t");
    const std::size_t last = value.find_last_not_of(" \t,");
    if (first == std::string_view::npos || last == std::string_view::npos) {
      throw std::invalid_argument("invalid keyboard .npy shape");
    }
    std::size_t result = 0U;
    for (const char character : value.substr(first, last - first + 1U)) {
      if (character < '0' || character > '9') {
        throw std::invalid_argument("invalid keyboard .npy shape");
      }
      result = result * 10U + static_cast<std::size_t>(character - '0');
    }
    return result;
  };
  const std::size_t rows = parse_dimension(
      header.substr(opening + 1U, comma - opening - 1U));
  const std::size_t columns = parse_dimension(
      header.substr(comma + 1U, closing - comma - 1U));
  const std::size_t stride = decode(alphabet).size();
  if (rows != stride || columns != stride ||
      rows > std::numeric_limits<std::size_t>::max() / columns) {
    throw std::invalid_argument(
        "keyboard matrix shape does not match the alphabet size");
  }
  const std::size_t payload_offset = header_offset + header_size;
  const std::size_t payload_size = rows * columns;
  if (payload_offset + payload_size > bytes.size()) {
    throw std::invalid_argument("truncated keyboard .npy payload");
  }
  Matrix matrix;
  matrix.name = std::move(name);
  matrix.sql_name = matrix.name;
  matrix.alphabet = std::move(alphabet);
  matrix.gap_score = gap_score;
  matrix.has_affine = gap_open.has_value() || gap_extend.has_value();
  if (matrix.has_affine) {
    if (!gap_open.has_value() || !gap_extend.has_value()) {
      throw std::invalid_argument(
          "gap_open and gap_extend must both be set or both be NULL");
    }
    matrix.gap_open = *gap_open;
    matrix.gap_extend = *gap_extend;
  }
  matrix.values.resize(payload_size);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      const std::size_t source = row * columns + column;
      const std::size_t destination = transpose
          ? column * rows + row : source;
      matrix.values[destination] = static_cast<std::int8_t>(
          static_cast<unsigned char>(bytes[payload_offset + source]));
    }
  }
  return finish_matrix(std::move(matrix), wildcard);
}

void KeyboardFromNpy(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const std::string bytes = StringValue::Get(
        adapter::argument(arguments, 0, row));
    const std::string alphabet = optional_string(
        arguments, row, 2, default_ascii_alphabet());
    const auto symbols = decode(alphabet);
    if (symbols.empty()) {
      throw std::invalid_argument("keyboard alphabet cannot be empty");
    }
    const std::string default_wildcard = encode_utf8(
        std::span<const std::uint32_t>(symbols.data() + symbols.size() - 1U, 1U));
    const std::string wildcard = optional_string(
        arguments, row, 3, default_wildcard);
    return matrix_value(matrix_from_npy(
        bytes, alphabet,
        optional_string(arguments, row, 1, "KEYBOARD"), wildcard,
        arguments.ColumnCount() > 4
            ? adapter::boolean_value(adapter::argument(arguments, 4, row))
            : false,
        optional_integer(arguments, row, 5, -1),
        optional_nullable_integer(arguments, row, 6),
        optional_nullable_integer(arguments, row, 7)));
  }, false);
}

void KeyboardFromConfusionCounts(
    DataChunk& arguments,
    ExpressionState&,
    Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const std::string alphabet = optional_string(
        arguments, row, 1, default_ascii_alphabet());
    const auto symbols = decode(alphabet);
    if (symbols.empty()) {
      throw std::invalid_argument("keyboard alphabet cannot be empty");
    }
    const Value count_grid = adapter::argument(arguments, 0, row);
    const auto counts = adapter::nested_double_list(count_grid, "counts");
    if (counts.size() != symbols.size()) {
      throw std::invalid_argument("counts matrix shape does not match the alphabet");
    }
    for (const auto& values : counts) {
      if (values.size() != symbols.size()) {
        throw std::invalid_argument("counts matrix shape does not match the alphabet");
      }
      for (const double value : values) {
        if (value < 0.0) {
          throw std::invalid_argument("counts must be finite non-negative numbers");
        }
      }
    }
    const double scale = arguments.ColumnCount() > 3
        ? adapter::double_value(adapter::argument(arguments, 3, row)) : 2.0;
    const auto match_margin = optional_integer(arguments, row, 4, 4);
    if (!std::isfinite(scale)) {
      throw std::invalid_argument("scale must be finite");
    }
    std::optional<double> requested_floor;
    if (arguments.ColumnCount() > 5 &&
        !adapter::argument(arguments, 5, row).IsNull()) {
      requested_floor = adapter::double_value(
          adapter::argument(arguments, 5, row));
      if (!std::isfinite(*requested_floor)) {
        throw std::invalid_argument("floor must be finite or NULL");
      }
    }
    std::vector<double> row_totals(symbols.size(), 0.0);
    std::vector<double> column_totals(symbols.size(), 0.0);
    double total = 0.0;
    for (std::size_t left = 0; left < symbols.size(); ++left) {
      for (std::size_t right = 0; right < symbols.size(); ++right) {
        row_totals[left] += counts[left][right];
        column_totals[right] += counts[left][right];
        total += counts[left][right];
      }
    }
    std::vector<double> raw(symbols.size() * symbols.size(), 0.0);
    std::optional<double> smallest_finite;
    if (total != 0.0) {
      for (std::size_t left = 0; left < symbols.size(); ++left) {
        for (std::size_t right = 0; right < symbols.size(); ++right) {
          const double denominator = row_totals[left] * column_totals[right];
          const double value = denominator == 0.0 || counts[left][right] == 0.0
              ? -std::numeric_limits<double>::infinity()
              : scale * std::log2(counts[left][right] * total / denominator);
          raw[left * symbols.size() + right] = value;
          if (std::isfinite(value) &&
              (!smallest_finite.has_value() || value < *smallest_finite)) {
            smallest_finite = value;
          }
        }
      }
    }
    const double floor = requested_floor.value_or(
        smallest_finite.has_value() ? std::floor(*smallest_finite) : -1.0);
    Matrix matrix;
    matrix.name = optional_string(arguments, row, 2, "KEYBOARD");
    matrix.sql_name = matrix.name;
    matrix.alphabet = alphabet;
    matrix.gap_score = optional_integer(arguments, row, 7, -1);
    const auto gap_open = optional_nullable_integer(arguments, row, 8);
    const auto gap_extend = optional_nullable_integer(arguments, row, 9);
    matrix.has_affine = gap_open.has_value() || gap_extend.has_value();
    if (matrix.has_affine) {
      if (!gap_open.has_value() || !gap_extend.has_value()) {
        throw std::invalid_argument(
            "gap_open and gap_extend must both be set or both be NULL");
      }
      matrix.gap_open = *gap_open;
      matrix.gap_extend = *gap_extend;
    }
    matrix.values.resize(raw.size());
    if (total == 0.0) {
      std::fill(matrix.values.begin(), matrix.values.end(), -1);
      for (std::size_t index = 0; index < symbols.size(); ++index) {
        matrix.values[index * symbols.size() + index] = 1;
      }
    } else {
      for (std::size_t index = 0; index < raw.size(); ++index) {
        const double value = std::isfinite(raw[index]) ? raw[index] : floor;
        matrix.values[index] = static_cast<std::int8_t>(std::clamp(
            std::nearbyint(value), -128.0, 127.0));
      }
      for (std::size_t index = 0; index < symbols.size(); ++index) {
        matrix.values[index * symbols.size() + index] = -128;
      }
      const auto best = *std::max_element(
          matrix.values.begin(), matrix.values.end());
      const auto diagonal = std::min<::stride_align::Score>(
          127, std::max<::stride_align::Score>(best, 0) + match_margin);
      if (diagonal < -128) {
        throw std::invalid_argument("match_margin produces an invalid score");
      }
      for (std::size_t index = 0; index < symbols.size(); ++index) {
        matrix.values[index * symbols.size() + index] =
            static_cast<std::int8_t>(diagonal);
      }
    }
    const std::string default_wildcard = encode_utf8(
        std::span<const std::uint32_t>(symbols.data() + symbols.size() - 1U, 1U));
    return matrix_value(finish_matrix(
        std::move(matrix),
        optional_string(arguments, row, 6, default_wildcard)));
  }, false);
}

void MatrixTranspose(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const MatrixArgument resolved = matrix_argument(
        adapter::argument(arguments, 0, row));
    Matrix matrix = resolved.get();
    const std::size_t stride = matrix.stride();
    for (std::size_t left = 0; left < stride; ++left) {
      for (std::size_t right = left + 1U; right < stride; ++right) {
        std::swap(matrix.values[left * stride + right],
                  matrix.values[right * stride + left]);
      }
    }
    matrix.name = arguments.ColumnCount() > 1
        ? adapter::string_value(adapter::argument(arguments, 1, row))
        : matrix.name + ".T";
    matrix.sql_name = matrix.name;
    return matrix_value(matrix);
  });
}

void SubstitutionMatrixScore(
    DataChunk& arguments,
    ExpressionState&,
    Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const MatrixArgument resolved = matrix_argument(
        adapter::argument(arguments, 0, row));
    const Matrix& matrix = resolved.get();
    return Value::BIGINT(matrix_score(
        matrix,
        adapter::string_value(adapter::argument(arguments, 1, row)),
        adapter::string_value(adapter::argument(arguments, 2, row)),
        true, matrix.gap_score, matrix.gap_score));
  });
}

std::vector<std::uint16_t> matrix_encode(
    const Matrix& matrix,
    std::string_view input) {
  const auto points = decode(input);
  std::vector<std::uint16_t> output;
  output.reserve(points.size());
  for (const std::uint32_t value : points) {
    const auto found = matrix.indices.find(value);
    output.push_back(found == matrix.indices.end()
        ? matrix.wildcard
        : found->second);
  }
  return output;
}

::stride_align::Score matrix_score(
    const Matrix& matrix,
    std::string_view query,
    std::string_view target,
    bool local,
    ::stride_align::Score gap_open,
    ::stride_align::Score gap_extend) {
  const auto left = matrix_encode(matrix, query);
  const auto right = matrix_encode(matrix, target);
  const auto lookup = [&](std::uint16_t a, std::uint16_t b) {
    return matrix.lookup(a, b);
  };
  return local
      ? ::stride_align::core::substitution_matrix_affine_score<true>(
            left, right, lookup, gap_open, gap_extend)
      : ::stride_align::core::substitution_matrix_affine_score<false>(
            left, right, lookup, gap_open, gap_extend);
}

::stride_align::AlignmentPath matrix_path(
    const Matrix& matrix,
    std::string_view query,
    std::string_view target,
    bool local,
    ::stride_align::Score gap_open,
    ::stride_align::Score gap_extend) {
  const auto left = matrix_encode(matrix, query);
  const auto right = matrix_encode(matrix, target);
  const auto lookup = [&](std::uint16_t a, std::uint16_t b) {
    return matrix.lookup(a, b);
  };
  return local
      ? ::stride_align::core::substitution_matrix_affine_path<true>(
            left, right, lookup, gap_open, gap_extend)
      : ::stride_align::core::substitution_matrix_affine_path<false>(
            left, right, lookup, gap_open, gap_extend);
}

LogicalType path_type() {
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

Value path_value(
    const ::stride_align::AlignmentPath& path,
    std::string_view query_text,
    std::string_view target_text) {
  const auto query = decode(query_text);
  const auto target = decode(target_text);
  std::vector<std::uint32_t> aligned_query;
  std::vector<std::uint32_t> aligned_target;
  std::size_t query_index = path.query_start;
  std::size_t target_index = path.target_start;
  for (const char operation : path.operations) {
    if (operation == '=' || operation == 'X') {
      aligned_query.push_back(query[query_index++]);
      aligned_target.push_back(target[target_index++]);
    } else if (operation == 'D') {
      aligned_query.push_back(query[query_index++]);
      aligned_target.push_back('-');
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
  return Value::STRUCT(path_type(), std::move(fields));
}

std::pair<::stride_align::Score, ::stride_align::Score> gaps(
    const Matrix& matrix,
    DataChunk& arguments,
    idx_t row,
    idx_t first,
    bool use_matrix_defaults = true) {
  if (arguments.ColumnCount() >= first + 2) {
    return {
        adapter::score_value(adapter::argument(arguments, first, row)),
        adapter::score_value(adapter::argument(arguments, first + 1, row)),
    };
  }
  if (arguments.ColumnCount() >= first + 1) {
    const auto gap = adapter::score_value(adapter::argument(arguments, first, row));
    return {gap, gap};
  }
  if (use_matrix_defaults) {
    if (matrix.has_affine) return {matrix.gap_open, matrix.gap_extend};
    return {matrix.gap_score, matrix.gap_score};
  }
  // Match the Python alignment APIs: supplying a substitution matrix does not
  // silently replace their default linear gap score.
  return {-1, -1};
}

void MatrixAvailable(DataChunk&, ExpressionState&, Vector& result) {
  vector<Value> output;
  for (const Matrix& matrix : matrices()) output.emplace_back(matrix.sql_name);
  result.SetVectorType(VectorType::CONSTANT_VECTOR);
  result.SetValue(0, Value::LIST(LogicalType::VARCHAR, std::move(output)));
}

void KeyboardAvailable(DataChunk&, ExpressionState&, Vector& result) {
  vector<Value> output;
  for (const Matrix& matrix : matrices()) {
    constexpr std::string_view prefix = "keyboard:";
    if (matrix.sql_name.starts_with(prefix)) {
      output.emplace_back(matrix.sql_name.substr(prefix.size()));
    }
  }
  result.SetVectorType(VectorType::CONSTANT_VECTOR);
  result.SetValue(0, Value::LIST(LogicalType::VARCHAR, std::move(output)));
}

void MatrixInfo(DataChunk& arguments, ExpressionState&, Vector& result) {
  const LogicalType type = LogicalType::STRUCT({
      {"name", LogicalType::VARCHAR},
      {"alphabet", LogicalType::VARCHAR},
      {"stride", LogicalType::UBIGINT},
      {"wildcard_index", LogicalType::UBIGINT},
      {"gap_score", LogicalType::BIGINT},
      {"gap_open", LogicalType::BIGINT},
      {"gap_extend", LogicalType::BIGINT},
  });
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const MatrixArgument resolved = matrix_argument(
        adapter::argument(arguments, 0, row));
    const Matrix& matrix = resolved.get();
    vector<Value> fields;
    fields.emplace_back(matrix.name);
    fields.emplace_back(matrix.alphabet);
    fields.push_back(Value::UBIGINT(matrix.stride()));
    fields.push_back(Value::UBIGINT(matrix.wildcard));
    fields.push_back(Value::BIGINT(matrix.gap_score));
    fields.push_back(matrix.has_affine
        ? Value::BIGINT(matrix.gap_open) : Value(LogicalType::BIGINT));
    fields.push_back(matrix.has_affine
        ? Value::BIGINT(matrix.gap_extend) : Value(LogicalType::BIGINT));
    return Value::STRUCT(type, std::move(fields));
  });
}

void MatrixEncode(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const MatrixArgument resolved = matrix_argument(
        adapter::argument(arguments, 0, row));
    const Matrix& matrix = resolved.get();
    const auto encoded = matrix_encode(
        matrix, adapter::string_value(adapter::argument(arguments, 1, row)));
    vector<Value> values;
    for (const std::uint16_t value : encoded) {
      values.push_back(Value::USMALLINT(value));
    }
    return Value::LIST(LogicalType::USMALLINT, std::move(values));
  });
}

void MatrixScoreStepLimit(
    DataChunk& arguments,
    ExpressionState&,
    Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const MatrixArgument resolved = matrix_argument(
        adapter::argument(arguments, 0, row));
    const Matrix& matrix = resolved.get();
    std::int64_t maximum = 0;
    for (const std::int8_t value : matrix.values) {
      maximum = std::max<std::int64_t>(maximum, std::abs(static_cast<int>(value)));
    }
    const auto [gap_open, gap_extend] = gaps(matrix, arguments, row, 1);
    maximum = std::max<std::int64_t>(maximum, std::abs(gap_open));
    maximum = std::max<std::int64_t>(maximum, std::abs(gap_extend));
    return Value::BIGINT(maximum);
  });
}

template <bool Local, bool Path, bool CigarOnly>
void MatrixAlignment(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const std::string query =
        adapter::string_value(adapter::argument(arguments, 0, row));
    const std::string target =
        adapter::string_value(adapter::argument(arguments, 1, row));
    const MatrixArgument resolved = matrix_argument(
        adapter::argument(arguments, 2, row));
    const Matrix& matrix = resolved.get();
    const auto [gap_open, gap_extend] = gaps(matrix, arguments, row, 3, false);
    if constexpr (!Path) {
      return Value::BIGINT(matrix_score(
          matrix, query, target, Local, gap_open, gap_extend));
    } else {
      const auto path = matrix_path(
          matrix, query, target, Local, gap_open, gap_extend);
      if constexpr (CigarOnly) {
        return Value(path.cigar);
      } else {
        return path_value(path, query, target);
      }
    }
  });
}

template <bool Local>
void MatrixScores(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const std::string query =
        adapter::string_value(adapter::argument(arguments, 0, row));
    const auto targets = adapter::text_list(adapter::argument(arguments, 1, row));
    const MatrixArgument resolved = matrix_argument(
        adapter::argument(arguments, 2, row));
    const Matrix& matrix = resolved.get();
    const auto [gap_open, gap_extend] = gaps(matrix, arguments, row, 3, false);
    vector<Value> output;
    output.reserve(targets.size());
    for (const auto& target : targets) {
      output.push_back(target.has_value()
          ? Value::BIGINT(matrix_score(
                matrix, query, target->bytes, Local, gap_open, gap_extend))
          : Value(LogicalType::BIGINT));
    }
    return Value::LIST(LogicalType::BIGINT, std::move(output));
  });
}

template <bool Local>
void MatrixCDist(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const auto queries = adapter::text_list(adapter::argument(arguments, 0, row));
    const auto targets = adapter::text_list(adapter::argument(arguments, 1, row));
    const MatrixArgument resolved = matrix_argument(
        adapter::argument(arguments, 2, row));
    const Matrix& matrix = resolved.get();
    const auto [gap_open, gap_extend] = gaps(matrix, arguments, row, 3, false);
    vector<Value> rows;
    for (const auto& query : queries) {
      vector<Value> values;
      for (const auto& target : targets) {
        values.push_back(query.has_value() && target.has_value()
            ? Value::BIGINT(matrix_score(
                  matrix, query->bytes, target->bytes, Local,
                  gap_open, gap_extend))
            : Value(LogicalType::BIGINT));
      }
      rows.push_back(Value::LIST(LogicalType::BIGINT, std::move(values)));
    }
    return Value::LIST(
        LogicalType::LIST(LogicalType::BIGINT), std::move(rows));
  });
}

LogicalType matrix_match_type() {
  return LogicalType::STRUCT({
      {"score", LogicalType::BIGINT},
      {"query", LogicalType::VARCHAR},
      {"target", LogicalType::VARCHAR},
      {"query_index", LogicalType::UBIGINT},
      {"target_index", LogicalType::UBIGINT},
  });
}

struct MatrixMatch {
  ::stride_align::Score score = 0;
  std::size_t query_index = 0;
  std::size_t target_index = 0;
};

bool better_matrix_match(
    const MatrixMatch& left,
    const MatrixMatch& right) noexcept {
  if (left.score != right.score) return left.score > right.score;
  if (left.query_index != right.query_index) {
    return left.query_index < right.query_index;
  }
  return left.target_index < right.target_index;
}

void insert_matrix_match(
    std::vector<MatrixMatch>& matches,
    MatrixMatch candidate,
    std::size_t k) {
  if (k == 0U) return;
  if (matches.size() < k) {
    matches.push_back(candidate);
    std::push_heap(
        matches.begin(), matches.end(), better_matrix_match);
    return;
  }
  if (!better_matrix_match(candidate, matches.front())) return;
  std::pop_heap(matches.begin(), matches.end(), better_matrix_match);
  matches.back() = candidate;
  std::push_heap(matches.begin(), matches.end(), better_matrix_match);
}

Value matrix_matches_value(
    std::span<const MatrixMatch> matches,
    std::span<const std::optional<::stride_align::batch::Text>> queries,
    std::span<const std::optional<::stride_align::batch::Text>> targets) {
  vector<Value> output;
  output.reserve(matches.size());
  const LogicalType type = matrix_match_type();
  for (const MatrixMatch& match : matches) {
    vector<Value> fields;
    fields.push_back(Value::BIGINT(match.score));
    fields.emplace_back(queries[match.query_index]->bytes);
    fields.emplace_back(targets[match.target_index]->bytes);
    fields.push_back(Value::UBIGINT(match.query_index));
    fields.push_back(Value::UBIGINT(match.target_index));
    output.push_back(Value::STRUCT(type, std::move(fields)));
  }
  return Value::LIST(type, std::move(output));
}

template <bool TopK>
void MatrixRank(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const auto queries = adapter::text_list(adapter::argument(arguments, 0, row));
    const auto targets = adapter::text_list(adapter::argument(arguments, 1, row));
    const MatrixArgument resolved = matrix_argument(
        adapter::argument(arguments, 2, row));
    const Matrix& matrix = resolved.get();
    const bool local = adapter::boolean_value(adapter::argument(arguments, 3, row));
    const std::int64_t parameter =
        adapter::integer_value(adapter::argument(arguments, 4, row));
    const auto [gap_open, gap_extend] = gaps(matrix, arguments, row, 5, false);
    std::vector<MatrixMatch> matches;
    const std::size_t k = TopK
        ? adapter::nonnegative_size(
              adapter::argument(arguments, 4, row), "k")
        : 0U;
    if constexpr (TopK) {
      if (k == 0U) {
        return matrix_matches_value(matches, queries, targets);
      }
      const std::size_t maximum_pairs = targets.empty() ||
              queries.size() <=
                  std::numeric_limits<std::size_t>::max() / targets.size()
          ? queries.size() * targets.size()
          : std::numeric_limits<std::size_t>::max();
      matches.reserve(std::min(k, maximum_pairs));
    }
    for (std::size_t query_index = 0;
         query_index < queries.size(); ++query_index) {
      if (!queries[query_index].has_value()) continue;
      for (std::size_t target_index = 0;
           target_index < targets.size(); ++target_index) {
        if (!targets[target_index].has_value()) continue;
        const ::stride_align::Score score = matrix_score(
            matrix, queries[query_index]->bytes,
            targets[target_index]->bytes, local, gap_open, gap_extend);
        if constexpr (TopK) {
          insert_matrix_match(
              matches, {score, query_index, target_index}, k);
        } else if (score >= parameter) {
          matches.push_back({score, query_index, target_index});
        }
      }
    }
    if constexpr (TopK) {
      std::sort(matches.begin(), matches.end(), better_matrix_match);
    }
    return matrix_matches_value(matches, queries, targets);
  });
}

void BeiderMorse(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const auto rule = arguments.ColumnCount() >= 2
        ? static_cast<::stride_align::phonetic::BmpmRuleType>(
              adapter::nonnegative_size(
                  adapter::argument(arguments, 1, row), "rule_type"))
        : ::stride_align::phonetic::BmpmRuleType::kApprox;
    if (rule != ::stride_align::phonetic::BmpmRuleType::kApprox &&
        rule != ::stride_align::phonetic::BmpmRuleType::kExact) {
      throw std::invalid_argument("rule_type must be 0 or 1");
    }
    const bool concat = arguments.ColumnCount() < 3 ||
        adapter::boolean_value(adapter::argument(arguments, 2, row));
    const std::size_t maximum = arguments.ColumnCount() < 4
        ? 20U
        : adapter::nonnegative_size(
              adapter::argument(arguments, 3, row), "max_phonemes");
    return Value(::stride_align::phonetic::beider_morse(
        decode(adapter::string_value(adapter::argument(arguments, 0, row))),
        rule, concat, maximum));
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

ScalarFunction special_function(
    std::vector<LogicalType> arguments,
    LogicalType result,
    scalar_function_t function) {
  ScalarFunction value(
      std::move(arguments), std::move(result), std::move(function));
  value.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
  return value;
}

template <bool Local>
void register_matrix_alignment(ExtensionLoader& loader) {
  const std::string stem = Local ? "smith_waterman" : "needleman_wunsch";
  for (const auto& suffix : {"matrix_score", "matrix_scores"}) {
    ScalarFunctionSet set("stride_" + stem + "_" + suffix);
    const bool many = std::string_view(suffix).ends_with("scores");
    const LogicalType second = many
        ? LogicalType::LIST(LogicalType::VARCHAR)
        : LogicalType::VARCHAR;
    const LogicalType return_type = many
        ? LogicalType::LIST(LogicalType::BIGINT)
        : LogicalType::BIGINT;
    const scalar_function_t function = many
        ? scalar_function_t(MatrixScores<Local>)
        : scalar_function_t(MatrixAlignment<Local, false, false>);
    for (const LogicalType& matrix_input : matrix_input_types()) {
      set.AddFunction(ScalarFunction(
          {LogicalType::VARCHAR, second, matrix_input},
          return_type, function));
      set.AddFunction(ScalarFunction(
          {LogicalType::VARCHAR, second, matrix_input,
           LogicalType::BIGINT}, return_type, function));
      set.AddFunction(ScalarFunction(
          {LogicalType::VARCHAR, second, matrix_input,
           LogicalType::BIGINT, LogicalType::BIGINT}, return_type, function));
    }
    loader.RegisterFunction(std::move(set));
  }
  for (const auto& suffix : {"matrix_path", "matrix_path_info"}) {
    ScalarFunctionSet set("stride_" + stem + "_" + suffix);
    for (const LogicalType& matrix_input : matrix_input_types()) {
      set.AddFunction(ScalarFunction(
          {LogicalType::VARCHAR, LogicalType::VARCHAR, matrix_input},
          path_type(), MatrixAlignment<Local, true, false>));
      set.AddFunction(ScalarFunction(
          {LogicalType::VARCHAR, LogicalType::VARCHAR, matrix_input,
           LogicalType::BIGINT},
          path_type(), MatrixAlignment<Local, true, false>));
      set.AddFunction(ScalarFunction(
          {LogicalType::VARCHAR, LogicalType::VARCHAR, matrix_input,
           LogicalType::BIGINT, LogicalType::BIGINT},
          path_type(), MatrixAlignment<Local, true, false>));
    }
    loader.RegisterFunction(std::move(set));
  }
  ScalarFunctionSet cigar("stride_" + stem + "_matrix_cigar");
  for (const LogicalType& matrix_input : matrix_input_types()) {
    cigar.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, matrix_input},
        LogicalType::VARCHAR, MatrixAlignment<Local, true, true>));
    cigar.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, matrix_input,
         LogicalType::BIGINT},
        LogicalType::VARCHAR, MatrixAlignment<Local, true, true>));
    cigar.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, matrix_input,
         LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::VARCHAR, MatrixAlignment<Local, true, true>));
  }
  loader.RegisterFunction(std::move(cigar));

  ScalarFunctionSet cdist("stride_cdist_matrix_" +
                          std::string(Local ? "local" : "global"));
  for (const LogicalType& matrix_input : matrix_input_types()) {
    cdist.AddFunction(ScalarFunction(
        {LogicalType::LIST(LogicalType::VARCHAR),
         LogicalType::LIST(LogicalType::VARCHAR), matrix_input},
        LogicalType::LIST(LogicalType::LIST(LogicalType::BIGINT)),
        MatrixCDist<Local>));
    cdist.AddFunction(ScalarFunction(
        {LogicalType::LIST(LogicalType::VARCHAR),
         LogicalType::LIST(LogicalType::VARCHAR), matrix_input,
         LogicalType::BIGINT},
        LogicalType::LIST(LogicalType::LIST(LogicalType::BIGINT)),
        MatrixCDist<Local>));
    cdist.AddFunction(ScalarFunction(
        {LogicalType::LIST(LogicalType::VARCHAR),
         LogicalType::LIST(LogicalType::VARCHAR), matrix_input,
         LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::LIST(LogicalType::LIST(LogicalType::BIGINT)),
        MatrixCDist<Local>));
  }
  loader.RegisterFunction(std::move(cdist));
}

}  // namespace

void RegisterMatrixFunctions(ExtensionLoader& loader) {
  // Resources are copied into the extension at build time.  Registration is
  // once per process; the BMPM engine itself builds its immutable tables on
  // first use.
  ::stride_align::phonetic::bmpm_register_resources(
      ::duckdb::stride_align_embedded::bmpm_resources());

  register_function(loader, "stride_matrix_available", {},
                    LogicalType::LIST(LogicalType::VARCHAR), MatrixAvailable);
  register_function(loader, "stride_keyboard_available", {},
                    LogicalType::LIST(LogicalType::VARCHAR), KeyboardAvailable);
  const LogicalType info_type = LogicalType::STRUCT({
          {"name", LogicalType::VARCHAR}, {"alphabet", LogicalType::VARCHAR},
          {"stride", LogicalType::UBIGINT},
          {"wildcard_index", LogicalType::UBIGINT},
          {"gap_score", LogicalType::BIGINT},
          {"gap_open", LogicalType::BIGINT},
          {"gap_extend", LogicalType::BIGINT}});
  ScalarFunctionSet info("stride_matrix_info");
  for (const LogicalType& input : matrix_input_types()) {
    info.AddFunction(ScalarFunction({input}, info_type, MatrixInfo));
  }
  loader.RegisterFunction(std::move(info));
  ScalarFunctionSet encode("stride_matrix_encode");
  for (const LogicalType& input : matrix_input_types()) {
    encode.AddFunction(ScalarFunction(
        {input, LogicalType::VARCHAR},
        LogicalType::LIST(LogicalType::USMALLINT), MatrixEncode));
  }
  loader.RegisterFunction(std::move(encode));
  ScalarFunctionSet step("stride_matrix_score_step_limit");
  for (const LogicalType& input : matrix_input_types()) {
    step.AddFunction(ScalarFunction(
        {input}, LogicalType::BIGINT, MatrixScoreStepLimit));
    step.AddFunction(ScalarFunction(
        {input, LogicalType::BIGINT},
        LogicalType::BIGINT, MatrixScoreStepLimit));
    step.AddFunction(ScalarFunction(
        {input, LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::BIGINT, MatrixScoreStepLimit));
  }
  loader.RegisterFunction(std::move(step));

  ScalarFunctionSet identity("stride_identity_matrix");
  const std::vector<LogicalType> identity_arguments = {
      LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT,
      LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT,
      LogicalType::BIGINT, LogicalType::BIGINT};
  for (std::size_t count = 1U; count <= identity_arguments.size(); ++count) {
    identity.AddFunction(special_function(
        std::vector<LogicalType>(
            identity_arguments.begin(), identity_arguments.begin() + count),
        matrix_type(), IdentityMatrix));
  }
  loader.RegisterFunction(std::move(identity));
  ScalarFunctionSet ascii("stride_ascii_matrix");
  const std::vector<LogicalType> ascii_arguments = {
      LogicalType::BIGINT, LogicalType::BIGINT,
      LogicalType::VARCHAR, LogicalType::BIGINT,
      LogicalType::BIGINT, LogicalType::BIGINT};
  for (std::size_t count = 0U; count <= ascii_arguments.size(); ++count) {
    ascii.AddFunction(special_function(
        std::vector<LogicalType>(
            ascii_arguments.begin(), ascii_arguments.begin() + count),
        matrix_type(), AsciiMatrix));
  }
  loader.RegisterFunction(std::move(ascii));
  ScalarFunctionSet from_ncbi("stride_matrix_from_ncbi_text");
  const std::vector<LogicalType> ncbi_arguments = {
      LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT,
      LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT};
  for (std::size_t count = 1U; count <= ncbi_arguments.size(); ++count) {
    from_ncbi.AddFunction(special_function(
        std::vector<LogicalType>(
            ncbi_arguments.begin(), ncbi_arguments.begin() + count),
        matrix_type(), MatrixFromNcbiText));
  }
  loader.RegisterFunction(std::move(from_ncbi));

  const LogicalType score_grid = LogicalType::LIST(
      LogicalType::LIST(LogicalType::BIGINT));
  ScalarFunctionSet substitution("stride_substitution_matrix");
  const std::vector<LogicalType> substitution_arguments = {
      LogicalType::VARCHAR, LogicalType::VARCHAR, score_grid,
      LogicalType::BIGINT, LogicalType::VARCHAR,
      LogicalType::BIGINT, LogicalType::BIGINT};
  for (std::size_t count = 3U;
       count <= substitution_arguments.size(); ++count) {
    substitution.AddFunction(special_function(
        std::vector<LogicalType>(substitution_arguments.begin(),
                                 substitution_arguments.begin() + count),
        matrix_type(), SubstitutionMatrix));
  }
  loader.RegisterFunction(std::move(substitution));

  const LogicalType count_grid = LogicalType::LIST(
      LogicalType::LIST(LogicalType::DOUBLE));
  ScalarFunctionSet keyboard_counts("stride_keyboard_from_confusion_counts");
  const std::vector<LogicalType> keyboard_count_arguments = {
      count_grid, LogicalType::VARCHAR, LogicalType::VARCHAR,
      LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::DOUBLE,
      LogicalType::VARCHAR, LogicalType::BIGINT,
      LogicalType::BIGINT, LogicalType::BIGINT};
  for (std::size_t count = 1U;
       count <= keyboard_count_arguments.size(); ++count) {
    keyboard_counts.AddFunction(special_function(
        std::vector<LogicalType>(keyboard_count_arguments.begin(),
                                 keyboard_count_arguments.begin() + count),
        matrix_type(), KeyboardFromConfusionCounts));
  }
  loader.RegisterFunction(std::move(keyboard_counts));

  ScalarFunctionSet keyboard_npy("stride_keyboard_from_npy");
  const std::vector<LogicalType> keyboard_npy_arguments = {
      LogicalType::BLOB, LogicalType::VARCHAR, LogicalType::VARCHAR,
      LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::BIGINT,
      LogicalType::BIGINT, LogicalType::BIGINT};
  for (std::size_t count = 1U;
       count <= keyboard_npy_arguments.size(); ++count) {
    keyboard_npy.AddFunction(special_function(
        std::vector<LogicalType>(keyboard_npy_arguments.begin(),
                                 keyboard_npy_arguments.begin() + count),
        matrix_type(), KeyboardFromNpy));
  }
  loader.RegisterFunction(std::move(keyboard_npy));

  ScalarFunctionSet transpose("stride_matrix_transpose");
  ScalarFunctionSet substitution_score("stride_substitution_matrix_score");
  for (const LogicalType& input : matrix_input_types()) {
    transpose.AddFunction(ScalarFunction(
        {input}, matrix_type(), MatrixTranspose));
    transpose.AddFunction(ScalarFunction(
        {input, LogicalType::VARCHAR}, matrix_type(), MatrixTranspose));
    substitution_score.AddFunction(ScalarFunction(
        {input, LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BIGINT, SubstitutionMatrixScore));
  }
  loader.RegisterFunction(std::move(transpose));
  loader.RegisterFunction(std::move(substitution_score));

  register_matrix_alignment<true>(loader);
  register_matrix_alignment<false>(loader);

  const LogicalType strings = LogicalType::LIST(LogicalType::VARCHAR);
  for (const auto& definition : {
           std::pair<const char*, scalar_function_t>(
               "stride_cdist_matrix_above_threshold", MatrixRank<false>),
           std::pair<const char*, scalar_function_t>(
               "stride_cdist_matrix_top_k", MatrixRank<true>)}) {
    ScalarFunctionSet functions(definition.first);
    for (const LogicalType& matrix_input : matrix_input_types()) {
      functions.AddFunction(ScalarFunction(
          {strings, strings, matrix_input, LogicalType::BOOLEAN,
           LogicalType::BIGINT},
          LogicalType::LIST(matrix_match_type()), definition.second));
      functions.AddFunction(ScalarFunction(
          {strings, strings, matrix_input, LogicalType::BOOLEAN,
           LogicalType::BIGINT, LogicalType::BIGINT},
          LogicalType::LIST(matrix_match_type()), definition.second));
      functions.AddFunction(ScalarFunction(
          {strings, strings, matrix_input, LogicalType::BOOLEAN,
           LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
          LogicalType::LIST(matrix_match_type()), definition.second));
    }
    loader.RegisterFunction(std::move(functions));
  }

  ScalarFunctionSet bmpm("stride_beider_morse");
  bmpm.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR}, LogicalType::VARCHAR, BeiderMorse));
  bmpm.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::BIGINT},
      LogicalType::VARCHAR, BeiderMorse));
  bmpm.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BOOLEAN},
      LogicalType::VARCHAR, BeiderMorse));
  bmpm.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::BIGINT,
       LogicalType::BOOLEAN, LogicalType::BIGINT},
      LogicalType::VARCHAR, BeiderMorse));
  loader.RegisterFunction(std::move(bmpm));
}

}  // namespace duckdb::stride_align_extension
