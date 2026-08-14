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
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "embedded_data.hpp"
#include "stride_align/alignment.hpp"
#include "stride_align/beider_morse.hpp"
#include "stride_align/core.hpp"
#include "stride_align/encoded.hpp"

extern "C" {
PG_FUNCTION_INFO_V1(stride_pg_beider_morse);
PG_FUNCTION_INFO_V1(stride_pg_matrix_available);
PG_FUNCTION_INFO_V1(stride_pg_keyboard_available);
PG_FUNCTION_INFO_V1(stride_pg_matrix_info);
PG_FUNCTION_INFO_V1(stride_pg_matrix_encode);
PG_FUNCTION_INFO_V1(stride_pg_matrix_score_step_limit);
PG_FUNCTION_INFO_V1(stride_pg_identity_matrix);
PG_FUNCTION_INFO_V1(stride_pg_ascii_matrix);
PG_FUNCTION_INFO_V1(stride_pg_matrix_from_ncbi_text);
PG_FUNCTION_INFO_V1(stride_pg_substitution_matrix);
PG_FUNCTION_INFO_V1(stride_pg_keyboard_from_confusion_counts);
PG_FUNCTION_INFO_V1(stride_pg_keyboard_from_npy);
PG_FUNCTION_INFO_V1(stride_pg_matrix_transpose);
PG_FUNCTION_INFO_V1(stride_pg_substitution_matrix_score);
PG_FUNCTION_INFO_V1(stride_pg_matrix_alignment);
PG_FUNCTION_INFO_V1(stride_pg_matrix_scores);
PG_FUNCTION_INFO_V1(stride_pg_matrix_cdist);
PG_FUNCTION_INFO_V1(stride_pg_matrix_rank);
}

namespace {

using Score = ::stride_align::Score;

struct NativeEncoding {
  int id = PG_SQL_ASCII;
  ::stride_align::encoded::EncodingProfile profile;

  static NativeEncoding database() {
    NativeEncoding output;
    output.id = GetDatabaseEncoding();
    output.profile.max_width =
        static_cast<std::size_t>(pg_encoding_max_length(output.id));
    output.profile.fixed_width = output.profile.max_width == 1U ? 1U : 0U;
    return output;
  }

  std::size_t width(std::string_view remaining) const {
    return static_cast<std::size_t>(pg_encoding_mblen_or_incomplete(
        id, remaining.data(), remaining.size()));
  }

  ::stride_align::encoded::TokenizedText tokenize(
      std::string_view input) const {
    return ::stride_align::encoded::tokenize(
        input, profile,
        [this](std::string_view remaining) { return width(remaining); });
  }

  std::vector<std::uint32_t> semantic_tokens(
      std::string_view input) const {
    if (id == PG_UTF8) return ::stride_align::utf8::prepare_streaming(input);
    return tokenize(input).tokens;
  }
};

struct InputText {
  text* value = nullptr;

  std::string_view view() const noexcept {
    return {
        VARDATA_ANY(value),
        static_cast<std::size_t>(VARSIZE_ANY_EXHDR(value))};
  }
};

InputText input_text(FunctionCallInfo fcinfo, int argument) {
  if (argument >= PG_NARGS() || PG_ARGISNULL(argument)) {
    throw std::invalid_argument("required text argument is NULL");
  }
  return {PG_GETARG_TEXT_PP(argument)};
}

std::string text_string(FunctionCallInfo fcinfo, int argument) {
  const auto input = input_text(fcinfo, argument);
  return std::string(input.view());
}

std::optional<std::string> optional_text(
    FunctionCallInfo fcinfo,
    int argument) {
  if (argument >= PG_NARGS() || PG_ARGISNULL(argument)) return std::nullopt;
  return text_string(fcinfo, argument);
}

std::optional<Score> optional_score(
    FunctionCallInfo fcinfo,
    int argument) {
  if (argument >= PG_NARGS() || PG_ARGISNULL(argument)) return std::nullopt;
  return static_cast<Score>(PG_GETARG_INT64(argument));
}

std::size_t nonnegative_size(
    FunctionCallInfo fcinfo,
    int argument,
    const char* name) {
  if (argument >= PG_NARGS() || PG_ARGISNULL(argument)) {
    throw std::invalid_argument(std::string(name) + " cannot be NULL");
  }
  const std::int64_t value = PG_GETARG_INT64(argument);
  if (value < 0) {
    throw std::invalid_argument(std::string(name) + " must be non-negative");
  }
  return static_cast<std::size_t>(value);
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

Datum jsonb_datum(const std::string& json) {
  return DirectFunctionCall1(jsonb_in, CStringGetDatum(json.c_str()));
}

Datum text_datum(std::string_view value) {
  return PointerGetDatum(cstring_to_text_with_len(
      value.data(), static_cast<int>(value.size())));
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

struct Matrix {
  std::string sql_name;
  std::string name;
  std::string alphabet_bytes;
  std::vector<std::uint32_t> alphabet;
  std::uint16_t wildcard = 0;
  Score gap_score = -1;
  Score gap_open = 0;
  Score gap_extend = 0;
  bool has_affine = false;
  std::vector<std::int8_t> values;
  std::unordered_map<std::uint32_t, std::uint16_t> indices;

  std::size_t stride() const noexcept { return alphabet.size(); }

  std::int8_t lookup(std::uint16_t left, std::uint16_t right) const {
    return values[static_cast<std::size_t>(left) * stride() + right];
  }
};

void validate_gaps(
    Matrix& matrix,
    std::optional<Score> gap_open,
    std::optional<Score> gap_extend) {
  matrix.has_affine = gap_open.has_value() || gap_extend.has_value();
  if (matrix.has_affine) {
    if (!gap_open.has_value() || !gap_extend.has_value()) {
      throw std::invalid_argument(
          "gap_open and gap_extend must both be set or both be NULL");
    }
    matrix.gap_open = *gap_open;
    matrix.gap_extend = *gap_extend;
  }
}

Matrix finish_matrix(Matrix matrix, std::uint32_t wildcard_token) {
  if (matrix.alphabet.empty() || matrix.alphabet.size() > 256U) {
    throw std::invalid_argument(
        "substitution-matrix alphabet must contain 1 through 256 symbols");
  }
  for (std::size_t index = 0; index < matrix.alphabet.size(); ++index) {
    if (!matrix.indices.emplace(
            matrix.alphabet[index], static_cast<std::uint16_t>(index)).second) {
      throw std::invalid_argument(
          "substitution-matrix alphabet contains duplicate symbols");
    }
  }
  const auto wildcard = matrix.indices.find(wildcard_token);
  if (wildcard == matrix.indices.end()) {
    throw std::invalid_argument(
        "substitution-matrix wildcard must be one symbol in the alphabet");
  }
  matrix.wildcard = wildcard->second;
  if (matrix.values.size() != matrix.stride() * matrix.stride()) {
    throw std::invalid_argument(
        "substitution-matrix score grid does not match the alphabet");
  }
  return matrix;
}

Matrix finish_deserialized(Matrix matrix) {
  if (matrix.alphabet.empty() || matrix.alphabet.size() > 256U ||
      matrix.wildcard >= matrix.alphabet.size() ||
      matrix.values.size() != matrix.alphabet.size() * matrix.alphabet.size()) {
    throw std::invalid_argument("invalid stride-align matrix value");
  }
  for (std::size_t index = 0; index < matrix.alphabet.size(); ++index) {
    if (!matrix.indices.emplace(
            matrix.alphabet[index], static_cast<std::uint16_t>(index)).second) {
      throw std::invalid_argument(
          "substitution-matrix alphabet contains duplicate symbols");
    }
  }
  return matrix;
}

std::string canonical_matrix_name(std::string_view input) {
  std::string output;
  output.reserve(input.size());
  for (const char character : input) {
    if (character >= 'A' && character <= 'Z') {
      output.push_back(static_cast<char>(character - 'A' + 'a'));
    } else if (character == '-' || character == '.') {
      output.push_back('_');
    } else {
      output.push_back(character);
    }
  }
  if (output == "nuc_4_4") output = "nuc44";
  if (output == "ascii") output = "ascii_text";
  if (output == "qwerty") output = "keyboard:qwerty";
  return output;
}

std::vector<Matrix> build_matrices() {
  std::vector<Matrix> output;
  for (auto& record : ::stride_align_postgres_embedded::matrices()) {
    Matrix matrix;
    matrix.sql_name = record.sql_name;
    matrix.name = record.name;
    matrix.gap_score = record.gap_score;
    matrix.gap_open = record.gap_open;
    matrix.gap_extend = record.gap_extend;
    matrix.has_affine = record.has_affine;
    matrix.values = std::move(record.values);
    std::uint32_t wildcard_token = 0;
    if (matrix.sql_name == "ascii_text" ||
        matrix.sql_name.starts_with("keyboard:")) {
      matrix.alphabet.reserve(128U);
      matrix.alphabet_bytes.reserve(128U);
      for (std::uint32_t value = 0; value < 128U; ++value) {
        matrix.alphabet.push_back(value);
        matrix.alphabet_bytes.push_back(static_cast<char>(value));
      }
      wildcard_token = 127U;
    } else {
      matrix.alphabet_bytes = record.alphabet;
      for (const unsigned char byte : matrix.alphabet_bytes) {
        matrix.alphabet.push_back(byte);
      }
      wildcard_token = static_cast<unsigned char>(record.wildcard[0]);
    }
    output.push_back(finish_matrix(std::move(matrix), wildcard_token));
  }
  return output;
}

const std::vector<Matrix>& matrices() {
  static const std::vector<Matrix> value = build_matrices();
  return value;
}

const Matrix& resolve_matrix(std::string_view name) {
  const std::string canonical = canonical_matrix_name(name);
  for (const Matrix& matrix : matrices()) {
    if (canonical_matrix_name(matrix.sql_name) == canonical ||
        canonical_matrix_name(matrix.name) == canonical) {
      return matrix;
    }
  }
  throw std::invalid_argument(
      "unknown stride-align matrix: " + std::string(name));
}

template <typename Integer>
void append_integer(std::string& output, Integer value) {
  using Unsigned = std::make_unsigned_t<Integer>;
  Unsigned encoded = static_cast<Unsigned>(value);
  for (std::size_t index = 0; index < sizeof(Integer); ++index) {
    output.push_back(static_cast<char>(encoded & 0xffU));
    encoded = static_cast<Unsigned>(encoded >> 8U);
  }
}

template <typename Integer>
Integer read_integer(std::string_view input, std::size_t& offset) {
  if (offset > input.size() || input.size() - offset < sizeof(Integer)) {
    throw std::invalid_argument("truncated stride-align matrix value");
  }
  using Unsigned = std::make_unsigned_t<Integer>;
  Unsigned value = 0;
  for (std::size_t index = 0; index < sizeof(Integer); ++index) {
    value |= static_cast<Unsigned>(
        static_cast<unsigned char>(input[offset + index])) << (index * 8U);
  }
  offset += sizeof(Integer);
  return static_cast<Integer>(value);
}

std::string serialize_matrix(const Matrix& matrix) {
  constexpr std::string_view magic = "STRDMAT1";
  std::string output(magic);
  append_integer<std::uint32_t>(output, matrix.name.size());
  append_integer<std::uint32_t>(output, matrix.alphabet_bytes.size());
  append_integer<std::uint32_t>(output, matrix.alphabet.size());
  append_integer<std::uint32_t>(output, matrix.values.size());
  append_integer<std::uint16_t>(output, matrix.wildcard);
  append_integer<std::uint8_t>(output, matrix.has_affine ? 1U : 0U);
  append_integer<std::int64_t>(output, matrix.gap_score);
  append_integer<std::int64_t>(output, matrix.gap_open);
  append_integer<std::int64_t>(output, matrix.gap_extend);
  output += matrix.name;
  output += matrix.alphabet_bytes;
  for (const std::uint32_t token : matrix.alphabet) {
    append_integer<std::uint32_t>(output, token);
  }
  for (const std::int8_t value : matrix.values) {
    output.push_back(static_cast<char>(value));
  }
  return output;
}

Matrix deserialize_matrix(std::string_view input) {
  constexpr std::string_view magic = "STRDMAT1";
  if (!input.starts_with(magic)) {
    throw std::invalid_argument("invalid stride-align matrix value");
  }
  std::size_t offset = magic.size();
  const std::size_t name_size = read_integer<std::uint32_t>(input, offset);
  const std::size_t alphabet_size = read_integer<std::uint32_t>(input, offset);
  const std::size_t token_count = read_integer<std::uint32_t>(input, offset);
  const std::size_t value_count = read_integer<std::uint32_t>(input, offset);
  Matrix matrix;
  matrix.wildcard = read_integer<std::uint16_t>(input, offset);
  matrix.has_affine = read_integer<std::uint8_t>(input, offset) != 0U;
  matrix.gap_score = read_integer<std::int64_t>(input, offset);
  matrix.gap_open = read_integer<std::int64_t>(input, offset);
  matrix.gap_extend = read_integer<std::int64_t>(input, offset);
  const std::size_t tail_size = name_size + alphabet_size +
      token_count * sizeof(std::uint32_t) + value_count;
  if (offset > input.size() || tail_size != input.size() - offset) {
    throw std::invalid_argument("invalid stride-align matrix value size");
  }
  matrix.name.assign(input.substr(offset, name_size));
  matrix.sql_name = matrix.name;
  offset += name_size;
  matrix.alphabet_bytes.assign(input.substr(offset, alphabet_size));
  offset += alphabet_size;
  matrix.alphabet.reserve(token_count);
  for (std::size_t index = 0; index < token_count; ++index) {
    matrix.alphabet.push_back(read_integer<std::uint32_t>(input, offset));
  }
  matrix.values.reserve(value_count);
  for (std::size_t index = 0; index < value_count; ++index) {
    matrix.values.push_back(static_cast<std::int8_t>(input[offset++]));
  }
  return finish_deserialized(std::move(matrix));
}

Datum bytea_datum(std::string_view value) {
  bytea* output = static_cast<bytea*>(palloc(VARHDRSZ + value.size()));
  SET_VARSIZE(output, VARHDRSZ + value.size());
  std::memcpy(VARDATA(output), value.data(), value.size());
  return PointerGetDatum(output);
}

std::string_view input_bytea(FunctionCallInfo fcinfo, int argument) {
  bytea* input = PG_GETARG_BYTEA_PP(argument);
  return {
      VARDATA_ANY(input),
      static_cast<std::size_t>(VARSIZE_ANY_EXHDR(input))};
}

struct MatrixArgument {
  const Matrix* borrowed = nullptr;
  std::optional<Matrix> owned;

  const Matrix& get() const noexcept {
    return owned.has_value() ? *owned : *borrowed;
  }
};

MatrixArgument matrix_argument(FunctionCallInfo fcinfo, int argument) {
  if (PG_ARGISNULL(argument)) {
    throw std::invalid_argument("matrix cannot be NULL");
  }
  const Oid type = get_fn_expr_argtype(fcinfo->flinfo, argument);
  MatrixArgument output;
  if (type == TEXTOID) {
    output.borrowed = &resolve_matrix(input_text(fcinfo, argument).view());
  } else if (type == BYTEAOID) {
    output.owned = deserialize_matrix(input_bytea(fcinfo, argument));
  } else {
    throw std::invalid_argument("matrix must be a built-in name or stride-align bytea value");
  }
  return output;
}

std::vector<std::optional<std::string>> text_array(ArrayType* input) {
  if (ARR_NDIM(input) > 1) {
    throw std::invalid_argument("text input must be a one-dimensional array");
  }
  Datum* values = nullptr;
  bool* nulls = nullptr;
  int count = 0;
  deconstruct_array(
      input, TEXTOID, -1, false, TYPALIGN_INT,
      &values, &nulls, &count);
  std::vector<std::optional<std::string>> output;
  output.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    if (nulls[index]) {
      output.emplace_back(std::nullopt);
    } else {
      text* value = DatumGetTextPP(values[index]);
      output.emplace_back(std::string(
          VARDATA_ANY(value),
          static_cast<std::size_t>(VARSIZE_ANY_EXHDR(value))));
    }
  }
  return output;
}

Datum text_array_datum(std::span<const std::string> values) {
  Datum* elements = static_cast<Datum*>(
      palloc(sizeof(Datum) * std::max<std::size_t>(values.size(), 1U)));
  for (std::size_t index = 0; index < values.size(); ++index) {
    elements[index] = text_datum(values[index]);
  }
  return PointerGetDatum(construct_array(
      elements, static_cast<int>(values.size()), TEXTOID,
      -1, false, TYPALIGN_INT));
}

Datum int16_array_datum(std::span<const std::uint16_t> values) {
  Datum* elements = static_cast<Datum*>(
      palloc(sizeof(Datum) * std::max<std::size_t>(values.size(), 1U)));
  for (std::size_t index = 0; index < values.size(); ++index) {
    elements[index] = Int16GetDatum(static_cast<std::int16_t>(values[index]));
  }
  return PointerGetDatum(construct_array(
      elements, static_cast<int>(values.size()), INT2OID,
      sizeof(int16), true, TYPALIGN_SHORT));
}

Datum score_array_datum(
    std::span<const std::optional<Score>> values,
    std::span<const int> dimensions = {}) {
  Datum* elements = static_cast<Datum*>(
      palloc(sizeof(Datum) * std::max<std::size_t>(values.size(), 1U)));
  bool* nulls = static_cast<bool*>(
      palloc(sizeof(bool) * std::max<std::size_t>(values.size(), 1U)));
  for (std::size_t index = 0; index < values.size(); ++index) {
    nulls[index] = !values[index].has_value();
    elements[index] = Int64GetDatum(values[index].value_or(0));
  }
  int16 element_width = 0;
  bool element_by_value = false;
  char element_alignment = 0;
  get_typlenbyvalalign(
      INT8OID, &element_width, &element_by_value, &element_alignment);
  if (dimensions.empty()) {
    std::array<int, 1> one_dimension{static_cast<int>(values.size())};
    std::array<int, 1> one_lower_bound{1};
    return PointerGetDatum(construct_md_array(
        elements, nulls, 1,
        one_dimension.data(), one_lower_bound.data(),
        INT8OID, element_width, element_by_value, element_alignment));
  }
  std::vector<int> lower_bounds(dimensions.size(), 1);
  return PointerGetDatum(construct_md_array(
      elements, nulls, static_cast<int>(dimensions.size()),
      const_cast<int*>(dimensions.data()), lower_bounds.data(),
      INT8OID, element_width, element_by_value, element_alignment));
}

double numeric_datum(Oid type, Datum value, const char* name) {
  double output = 0.0;
  switch (type) {
    case FLOAT8OID: output = DatumGetFloat8(value); break;
    case FLOAT4OID: output = DatumGetFloat4(value); break;
    case INT2OID: output = DatumGetInt16(value); break;
    case INT4OID: output = DatumGetInt32(value); break;
    case INT8OID: output = static_cast<double>(DatumGetInt64(value)); break;
    default:
      throw std::invalid_argument(
          std::string(name) + " must use a built-in numeric array type");
  }
  if (!std::isfinite(output)) {
    throw std::invalid_argument(std::string(name) + " must contain finite values");
  }
  return output;
}

std::vector<std::vector<double>> numeric_grid(
    ArrayType* input,
    const char* name) {
  if (ARR_NDIM(input) != 2) {
    throw std::invalid_argument(std::string(name) + " must be two-dimensional");
  }
  const std::size_t rows = static_cast<std::size_t>(ARR_DIMS(input)[0]);
  const std::size_t columns = static_cast<std::size_t>(ARR_DIMS(input)[1]);
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
  std::vector<std::vector<double>> output(rows, std::vector<double>(columns));
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      const std::size_t index = row * columns + column;
      if (nulls[index]) {
        throw std::invalid_argument(std::string(name) + " cannot contain NULL");
      }
      output[row][column] = numeric_datum(type, values[index], name);
    }
  }
  return output;
}

std::pair<Score, Score> alignment_gaps(
    const Matrix& matrix,
    FunctionCallInfo fcinfo,
    int first,
    bool matrix_defaults) {
  const auto first_gap = optional_score(fcinfo, first);
  const auto second_gap = optional_score(fcinfo, first + 1);
  if (second_gap.has_value() && !first_gap.has_value()) {
    throw std::invalid_argument("gap_open cannot be NULL when gap_extend is set");
  }
  if (first_gap.has_value()) {
    return {*first_gap, second_gap.value_or(*first_gap)};
  }
  if (matrix_defaults) {
    return matrix.has_affine
        ? std::pair<Score, Score>{matrix.gap_open, matrix.gap_extend}
        : std::pair<Score, Score>{matrix.gap_score, matrix.gap_score};
  }
  return {-1, -1};
}

std::vector<std::uint16_t> matrix_encode(
    const Matrix& matrix,
    std::string_view input,
    const NativeEncoding& encoding) {
  const auto tokens = encoding.tokenize(input).tokens;
  std::vector<std::uint16_t> output;
  output.reserve(tokens.size());
  for (const std::uint32_t token : tokens) {
    const auto found = matrix.indices.find(token);
    output.push_back(found == matrix.indices.end()
        ? matrix.wildcard : found->second);
  }
  return output;
}

Score matrix_score(
    const Matrix& matrix,
    std::string_view query,
    std::string_view target,
    const NativeEncoding& encoding,
    bool local,
    Score gap_open,
    Score gap_extend) {
  const auto left = matrix_encode(matrix, query, encoding);
  const auto right = matrix_encode(matrix, target, encoding);
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
    const NativeEncoding& encoding,
    bool local,
    Score gap_open,
    Score gap_extend) {
  const auto left = matrix_encode(matrix, query, encoding);
  const auto right = matrix_encode(matrix, target, encoding);
  const auto lookup = [&](std::uint16_t a, std::uint16_t b) {
    return matrix.lookup(a, b);
  };
  return local
      ? ::stride_align::core::substitution_matrix_affine_path<true>(
            left, right, lookup, gap_open, gap_extend)
      : ::stride_align::core::substitution_matrix_affine_path<false>(
            left, right, lookup, gap_open, gap_extend);
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

std::string path_json(
    const ::stride_align::AlignmentPath& path,
    std::string_view query,
    std::string_view target,
    const NativeEncoding& encoding) {
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
  return json;
}

std::string display_alphabet(const Matrix& matrix) {
  if (matrix.alphabet_bytes.find('\0') == std::string::npos) {
    return matrix.alphabet_bytes;
  }
  std::string output;
  constexpr char hexadecimal[] = "0123456789abcdef";
  for (const unsigned char byte : matrix.alphabet_bytes) {
    if (byte < 0x20U || byte == 0x7fU) {
      output += "\\x";
      output.push_back(hexadecimal[byte >> 4U]);
      output.push_back(hexadecimal[byte & 0x0fU]);
    } else {
      output.push_back(static_cast<char>(byte));
    }
  }
  return output;
}

std::string matrix_info_json(const Matrix& matrix) {
  std::string output = "{\"name\":" + json_escape(matrix.name);
  output += ",\"alphabet\":" + json_escape(display_alphabet(matrix));
  output += ",\"stride\":" + std::to_string(matrix.stride());
  output += ",\"wildcard_index\":" + std::to_string(matrix.wildcard);
  output += ",\"gap_score\":" + std::to_string(matrix.gap_score);
  output += ",\"gap_open\":" +
      (matrix.has_affine ? std::to_string(matrix.gap_open) : "null");
  output += ",\"gap_extend\":" +
      (matrix.has_affine ? std::to_string(matrix.gap_extend) : "null") + "}";
  return output;
}

Matrix identity_matrix(
    std::string alphabet_bytes,
    const NativeEncoding& encoding,
    Score match,
    Score mismatch,
    std::string wildcard_bytes,
    std::string name,
    Score gap_score,
    std::optional<Score> gap_open,
    std::optional<Score> gap_extend) {
  auto alphabet = encoding.tokenize(alphabet_bytes).tokens;
  const auto wildcard = encoding.tokenize(wildcard_bytes).tokens;
  if (wildcard.size() != 1U) {
    throw std::invalid_argument("wildcard must be one character");
  }
  if (std::find(alphabet.begin(), alphabet.end(), wildcard.front()) ==
      alphabet.end()) {
    alphabet.push_back(wildcard.front());
    alphabet_bytes += wildcard_bytes;
  }
  if (match < -128 || match > 127 || mismatch < -128 || mismatch > 127) {
    throw std::invalid_argument("matrix scores must fit signed int8");
  }
  Matrix matrix;
  matrix.sql_name = name;
  matrix.name = std::move(name);
  matrix.alphabet_bytes = std::move(alphabet_bytes);
  matrix.alphabet = std::move(alphabet);
  matrix.gap_score = gap_score;
  validate_gaps(matrix, gap_open, gap_extend);
  matrix.values.assign(
      matrix.alphabet.size() * matrix.alphabet.size(),
      static_cast<std::int8_t>(mismatch));
  for (std::size_t index = 0; index < matrix.alphabet.size(); ++index) {
    matrix.values[index * matrix.alphabet.size() + index] =
        static_cast<std::int8_t>(match);
  }
  const std::size_t wildcard_index = static_cast<std::size_t>(std::find(
      matrix.alphabet.begin(), matrix.alphabet.end(), wildcard.front()) -
      matrix.alphabet.begin());
  for (std::size_t index = 0; index < matrix.alphabet.size(); ++index) {
    matrix.values[wildcard_index * matrix.alphabet.size() + index] =
        static_cast<std::int8_t>(mismatch);
    matrix.values[index * matrix.alphabet.size() + wildcard_index] =
        static_cast<std::int8_t>(mismatch);
  }
  return finish_matrix(std::move(matrix), wildcard.front());
}

Matrix matrix_from_npy(
    std::string_view bytes,
    std::string alphabet_bytes,
    std::vector<std::uint32_t> alphabet,
    std::string name,
    std::uint32_t wildcard,
    bool transpose,
    Score gap_score,
    std::optional<Score> gap_open,
    std::optional<Score> gap_extend) {
  constexpr unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
  if (bytes.size() < 10U ||
      std::memcmp(bytes.data(), magic, sizeof(magic)) != 0) {
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
  if (header.find("'fortran_order': True") != std::string_view::npos ||
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
    std::size_t output = 0;
    for (const char character : value.substr(first, last - first + 1U)) {
      if (character < '0' || character > '9') {
        throw std::invalid_argument("invalid keyboard .npy shape");
      }
      output = output * 10U + static_cast<std::size_t>(character - '0');
    }
    return output;
  };
  const std::size_t rows = parse_dimension(
      header.substr(opening + 1U, comma - opening - 1U));
  const std::size_t columns = parse_dimension(
      header.substr(comma + 1U, closing - comma - 1U));
  if (rows != alphabet.size() || columns != alphabet.size() ||
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
  matrix.alphabet_bytes = std::move(alphabet_bytes);
  matrix.alphabet = std::move(alphabet);
  matrix.gap_score = gap_score;
  validate_gaps(matrix, gap_open, gap_extend);
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

struct MatrixMatch {
  Score score = 0;
  std::size_t query_index = 0;
  std::size_t target_index = 0;
};

bool better_match(const MatrixMatch& left, const MatrixMatch& right) noexcept {
  if (left.score != right.score) return left.score > right.score;
  if (left.query_index != right.query_index) {
    return left.query_index < right.query_index;
  }
  return left.target_index < right.target_index;
}

void insert_match(
    std::vector<MatrixMatch>& matches,
    MatrixMatch candidate,
    std::size_t k) {
  if (k == 0U) return;
  if (matches.size() < k) {
    matches.push_back(candidate);
    std::push_heap(matches.begin(), matches.end(), better_match);
    return;
  }
  if (!better_match(candidate, matches.front())) return;
  std::pop_heap(matches.begin(), matches.end(), better_match);
  matches.back() = candidate;
  std::push_heap(matches.begin(), matches.end(), better_match);
}

std::string matches_json(
    std::span<const MatrixMatch> matches,
    std::span<const std::optional<std::string>> queries,
    std::span<const std::optional<std::string>> targets) {
  std::string output = "[";
  for (std::size_t index = 0; index < matches.size(); ++index) {
    if (index != 0U) output.push_back(',');
    const auto& match = matches[index];
    output += "{\"score\":" + std::to_string(match.score);
    output += ",\"query\":" + json_escape(*queries[match.query_index]);
    output += ",\"target\":" + json_escape(*targets[match.target_index]);
    output += ",\"query_index\":" + std::to_string(match.query_index);
    output += ",\"target_index\":" + std::to_string(match.target_index) + "}";
  }
  output.push_back(']');
  return output;
}

void register_bmpm_resources() {
  static std::once_flag once;
  std::call_once(once, [] {
    ::stride_align::phonetic::bmpm_register_resources(
        ::stride_align_postgres_embedded::bmpm_resources());
  });
}

}  // namespace

extern "C" PGDLLEXPORT Datum
stride_pg_beider_morse(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    register_bmpm_resources();
    const std::int64_t raw_rule = PG_GETARG_INT64(1);
    const auto rule = static_cast<::stride_align::phonetic::BmpmRuleType>(raw_rule);
    if (rule != ::stride_align::phonetic::BmpmRuleType::kApprox &&
        rule != ::stride_align::phonetic::BmpmRuleType::kExact) {
      throw std::invalid_argument("rule_type must be 0 or 1");
    }
    const std::size_t maximum = nonnegative_size(fcinfo, 3, "max_phonemes");
    const NativeEncoding encoding = NativeEncoding::database();
    return text_datum(::stride_align::phonetic::beider_morse(
        encoding.semantic_tokens(input_text(fcinfo, 0).view()),
        rule, PG_GETARG_BOOL(2), maximum));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_matrix_available(PG_FUNCTION_ARGS) {
  (void)fcinfo;
  return catch_errors([&]() -> Datum {
    std::vector<std::string> names;
    names.reserve(matrices().size());
    for (const Matrix& matrix : matrices()) names.push_back(matrix.sql_name);
    return text_array_datum(names);
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_keyboard_available(PG_FUNCTION_ARGS) {
  (void)fcinfo;
  return catch_errors([&]() -> Datum {
    std::vector<std::string> names;
    for (const Matrix& matrix : matrices()) {
      constexpr std::string_view prefix = "keyboard:";
      if (matrix.sql_name.starts_with(prefix)) {
        names.push_back(matrix.sql_name.substr(prefix.size()));
      }
    }
    return text_array_datum(names);
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_matrix_info(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const auto resolved = matrix_argument(fcinfo, 0);
    return jsonb_datum(matrix_info_json(resolved.get()));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_matrix_encode(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const auto resolved = matrix_argument(fcinfo, 0);
    const auto encoded = matrix_encode(
        resolved.get(), input_text(fcinfo, 1).view(), NativeEncoding::database());
    return int16_array_datum(encoded);
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_matrix_score_step_limit(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const auto resolved = matrix_argument(fcinfo, 0);
    const Matrix& matrix = resolved.get();
    std::int64_t maximum = 0;
    for (const std::int8_t value : matrix.values) {
      maximum = std::max<std::int64_t>(maximum, std::abs(static_cast<int>(value)));
    }
    const auto gaps = alignment_gaps(matrix, fcinfo, 1, true);
    maximum = std::max<std::int64_t>(maximum, std::abs(gaps.first));
    maximum = std::max<std::int64_t>(maximum, std::abs(gaps.second));
    return Int64GetDatum(maximum);
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_identity_matrix(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    const Matrix matrix = identity_matrix(
        text_string(fcinfo, 0), encoding,
        PG_GETARG_INT64(1), PG_GETARG_INT64(2), text_string(fcinfo, 3),
        text_string(fcinfo, 4), PG_GETARG_INT64(5),
        optional_score(fcinfo, 6), optional_score(fcinfo, 7));
    return bytea_datum(serialize_matrix(matrix));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_ascii_matrix(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const Score match = PG_GETARG_INT64(0);
    const Score mismatch = PG_GETARG_INT64(1);
    if (match < -128 || match > 127 || mismatch < -128 || mismatch > 127) {
      throw std::invalid_argument("matrix scores must fit signed int8");
    }
    Matrix matrix;
    matrix.name = text_string(fcinfo, 2);
    matrix.sql_name = matrix.name;
    matrix.gap_score = PG_GETARG_INT64(3);
    validate_gaps(matrix, optional_score(fcinfo, 4), optional_score(fcinfo, 5));
    for (std::uint32_t value = 0; value < 128U; ++value) {
      matrix.alphabet.push_back(value);
      matrix.alphabet_bytes.push_back(static_cast<char>(value));
    }
    matrix.values.assign(128U * 128U, static_cast<std::int8_t>(mismatch));
    for (std::size_t index = 0; index < 128U; ++index) {
      matrix.values[index * 128U + index] = static_cast<std::int8_t>(match);
      matrix.values[127U * 128U + index] = static_cast<std::int8_t>(mismatch);
      matrix.values[index * 128U + 127U] = static_cast<std::int8_t>(mismatch);
    }
    matrix = finish_matrix(std::move(matrix), 127U);
    return bytea_datum(serialize_matrix(matrix));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_matrix_from_ncbi_text(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const std::string source = text_string(fcinfo, 0);
    std::istringstream stream(source);
    std::string line;
    std::vector<std::string> header;
    std::vector<std::string> row_labels;
    std::vector<std::int8_t> values;
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
        throw std::invalid_argument("NCBI matrix row has the wrong number of values");
      }
      row_labels.push_back(fields.front());
      for (std::size_t index = 1; index < fields.size(); ++index) {
        std::size_t consumed = 0;
        const long parsed = std::stol(fields[index], &consumed);
        if (consumed != fields[index].size() || parsed < -128 || parsed > 127) {
          throw std::invalid_argument("NCBI matrix score must fit signed int8");
        }
        values.push_back(static_cast<std::int8_t>(parsed));
      }
    }
    if (header.empty() || row_labels != header) {
      throw std::invalid_argument("NCBI matrix row labels do not match the column header");
    }
    Matrix matrix;
    matrix.name = text_string(fcinfo, 1);
    matrix.sql_name = matrix.name;
    matrix.gap_score = PG_GETARG_INT64(2);
    validate_gaps(matrix, optional_score(fcinfo, 4), optional_score(fcinfo, 5));
    const NativeEncoding encoding = NativeEncoding::database();
    for (const std::string& symbol : header) matrix.alphabet_bytes += symbol;
    matrix.alphabet = encoding.tokenize(matrix.alphabet_bytes).tokens;
    matrix.values = std::move(values);
    const auto wildcard = encoding.tokenize(text_string(fcinfo, 3)).tokens;
    if (wildcard.size() != 1U) throw std::invalid_argument("wildcard must be one character");
    matrix = finish_matrix(std::move(matrix), wildcard.front());
    return bytea_datum(serialize_matrix(matrix));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_substitution_matrix(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    Matrix matrix;
    matrix.name = text_string(fcinfo, 0);
    matrix.sql_name = matrix.name;
    matrix.alphabet_bytes = text_string(fcinfo, 1);
    const NativeEncoding encoding = NativeEncoding::database();
    matrix.alphabet = encoding.tokenize(matrix.alphabet_bytes).tokens;
    const auto grid = numeric_grid(PG_GETARG_ARRAYTYPE_P(2), "scores");
    if (grid.size() != matrix.alphabet.size()) {
      throw std::invalid_argument("score-grid row count does not match the alphabet");
    }
    for (const auto& row : grid) {
      if (row.size() != matrix.alphabet.size()) {
        throw std::invalid_argument("score-grid column count does not match the alphabet");
      }
      for (const double value : row) {
        if (std::trunc(value) != value || value < -128.0 || value > 127.0) {
          throw std::invalid_argument("matrix scores must be signed int8 integers");
        }
        matrix.values.push_back(static_cast<std::int8_t>(value));
      }
    }
    matrix.gap_score = PG_GETARG_INT64(3);
    validate_gaps(matrix, optional_score(fcinfo, 5), optional_score(fcinfo, 6));
    const auto wildcard = encoding.tokenize(text_string(fcinfo, 4)).tokens;
    if (wildcard.size() != 1U) throw std::invalid_argument("wildcard must be one character");
    matrix = finish_matrix(std::move(matrix), wildcard.front());
    return bytea_datum(serialize_matrix(matrix));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_keyboard_from_confusion_counts(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    std::string alphabet_bytes;
    std::vector<std::uint32_t> alphabet;
    if (PG_ARGISNULL(1)) {
      alphabet.reserve(128U);
      alphabet_bytes.reserve(128U);
      for (std::uint32_t value = 0; value < 128U; ++value) {
        alphabet.push_back(value);
        alphabet_bytes.push_back(static_cast<char>(value));
      }
    } else {
      alphabet_bytes = text_string(fcinfo, 1);
      alphabet = encoding.tokenize(alphabet_bytes).tokens;
    }
    if (alphabet.empty()) throw std::invalid_argument("keyboard alphabet cannot be empty");
    const auto counts = numeric_grid(PG_GETARG_ARRAYTYPE_P(0), "counts");
    if (counts.size() != alphabet.size()) {
      throw std::invalid_argument("counts matrix shape does not match the alphabet");
    }
    for (const auto& row : counts) {
      if (row.size() != alphabet.size()) {
        throw std::invalid_argument("counts matrix shape does not match the alphabet");
      }
      for (const double value : row) {
        if (value < 0.0) throw std::invalid_argument("counts must be non-negative");
      }
    }
    const double scale = PG_GETARG_FLOAT8(3);
    const Score match_margin = PG_GETARG_INT64(4);
    const std::optional<double> requested_floor = PG_ARGISNULL(5)
        ? std::nullopt : std::optional<double>(PG_GETARG_FLOAT8(5));
    if (!std::isfinite(scale) ||
        (requested_floor.has_value() && !std::isfinite(*requested_floor))) {
      throw std::invalid_argument("scale and floor must be finite");
    }
    std::vector<double> row_totals(alphabet.size(), 0.0);
    std::vector<double> column_totals(alphabet.size(), 0.0);
    double total = 0.0;
    for (std::size_t left = 0; left < alphabet.size(); ++left) {
      for (std::size_t right = 0; right < alphabet.size(); ++right) {
        row_totals[left] += counts[left][right];
        column_totals[right] += counts[left][right];
        total += counts[left][right];
      }
    }
    std::vector<double> raw(alphabet.size() * alphabet.size(), 0.0);
    std::optional<double> smallest;
    if (total != 0.0) {
      for (std::size_t left = 0; left < alphabet.size(); ++left) {
        for (std::size_t right = 0; right < alphabet.size(); ++right) {
          const double denominator = row_totals[left] * column_totals[right];
          const double value = denominator == 0.0 || counts[left][right] == 0.0
              ? -std::numeric_limits<double>::infinity()
              : scale * std::log2(counts[left][right] * total / denominator);
          raw[left * alphabet.size() + right] = value;
          if (std::isfinite(value) && (!smallest.has_value() || value < *smallest)) {
            smallest = value;
          }
        }
      }
    }
    const double floor = requested_floor.value_or(
        smallest.has_value() ? std::floor(*smallest) : -1.0);
    Matrix matrix;
    matrix.name = text_string(fcinfo, 2);
    matrix.sql_name = matrix.name;
    matrix.alphabet_bytes = std::move(alphabet_bytes);
    matrix.alphabet = std::move(alphabet);
    matrix.gap_score = PG_GETARG_INT64(7);
    validate_gaps(matrix, optional_score(fcinfo, 8), optional_score(fcinfo, 9));
    matrix.values.resize(raw.size());
    if (total == 0.0) {
      std::fill(matrix.values.begin(), matrix.values.end(), -1);
      for (std::size_t index = 0; index < matrix.alphabet.size(); ++index) {
        matrix.values[index * matrix.alphabet.size() + index] = 1;
      }
    } else {
      for (std::size_t index = 0; index < raw.size(); ++index) {
        const double value = std::isfinite(raw[index]) ? raw[index] : floor;
        matrix.values[index] = static_cast<std::int8_t>(std::clamp(
            std::nearbyint(value), -128.0, 127.0));
      }
      for (std::size_t index = 0; index < matrix.alphabet.size(); ++index) {
        matrix.values[index * matrix.alphabet.size() + index] = -128;
      }
      const auto best = *std::max_element(matrix.values.begin(), matrix.values.end());
      const Score diagonal = std::min<Score>(
          127, std::max<Score>(best, 0) + match_margin);
      if (diagonal < -128) throw std::invalid_argument("match_margin produces an invalid score");
      for (std::size_t index = 0; index < matrix.alphabet.size(); ++index) {
        matrix.values[index * matrix.alphabet.size() + index] =
            static_cast<std::int8_t>(diagonal);
      }
    }
    std::uint32_t wildcard = matrix.alphabet.back();
    if (!PG_ARGISNULL(6)) {
      const auto tokens = encoding.tokenize(text_string(fcinfo, 6)).tokens;
      if (tokens.size() != 1U) throw std::invalid_argument("wildcard must be one character");
      wildcard = tokens.front();
    }
    matrix = finish_matrix(std::move(matrix), wildcard);
    return bytea_datum(serialize_matrix(matrix));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_keyboard_from_npy(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const NativeEncoding encoding = NativeEncoding::database();
    std::string alphabet_bytes;
    std::vector<std::uint32_t> alphabet;
    if (PG_ARGISNULL(2)) {
      alphabet.reserve(128U);
      alphabet_bytes.reserve(128U);
      for (std::uint32_t value = 0; value < 128U; ++value) {
        alphabet.push_back(value);
        alphabet_bytes.push_back(static_cast<char>(value));
      }
    } else {
      alphabet_bytes = text_string(fcinfo, 2);
      alphabet = encoding.tokenize(alphabet_bytes).tokens;
    }
    if (alphabet.empty()) throw std::invalid_argument("keyboard alphabet cannot be empty");
    std::uint32_t wildcard = alphabet.back();
    if (!PG_ARGISNULL(3)) {
      const auto tokens = encoding.tokenize(text_string(fcinfo, 3)).tokens;
      if (tokens.size() != 1U) throw std::invalid_argument("wildcard must be one character");
      wildcard = tokens.front();
    }
    const Matrix matrix = matrix_from_npy(
        input_bytea(fcinfo, 0), std::move(alphabet_bytes), std::move(alphabet),
        text_string(fcinfo, 1), wildcard, PG_GETARG_BOOL(4),
        PG_GETARG_INT64(5), optional_score(fcinfo, 6), optional_score(fcinfo, 7));
    return bytea_datum(serialize_matrix(matrix));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_matrix_transpose(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const auto resolved = matrix_argument(fcinfo, 0);
    Matrix matrix = resolved.get();
    for (std::size_t left = 0; left < matrix.stride(); ++left) {
      for (std::size_t right = left + 1U; right < matrix.stride(); ++right) {
        std::swap(matrix.values[left * matrix.stride() + right],
                  matrix.values[right * matrix.stride() + left]);
      }
    }
    matrix.name = optional_text(fcinfo, 1).value_or(matrix.name + ".T");
    matrix.sql_name = matrix.name;
    return bytea_datum(serialize_matrix(matrix));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_substitution_matrix_score(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const auto resolved = matrix_argument(fcinfo, 0);
    const Matrix& matrix = resolved.get();
    return Int64GetDatum(matrix_score(
        matrix, input_text(fcinfo, 1).view(), input_text(fcinfo, 2).view(),
        NativeEncoding::database(), true, matrix.gap_score, matrix.gap_score));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_matrix_alignment(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const std::string_view query = input_text(fcinfo, 0).view();
    const std::string_view target = input_text(fcinfo, 1).view();
    const auto resolved = matrix_argument(fcinfo, 2);
    const Matrix& matrix = resolved.get();
    const bool local = PG_GETARG_BOOL(3);
    const bool return_path = PG_GETARG_BOOL(4);
    const bool cigar_only = PG_GETARG_BOOL(5);
    const auto gaps = alignment_gaps(matrix, fcinfo, 6, false);
    const NativeEncoding encoding = NativeEncoding::database();
    if (!return_path) {
      return Int64GetDatum(matrix_score(
          matrix, query, target, encoding, local, gaps.first, gaps.second));
    }
    const auto path = matrix_path(
        matrix, query, target, encoding, local, gaps.first, gaps.second);
    return cigar_only ? text_datum(path.cigar)
                      : jsonb_datum(path_json(path, query, target, encoding));
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_matrix_scores(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const std::string_view query = input_text(fcinfo, 0).view();
    const auto targets = text_array(PG_GETARG_ARRAYTYPE_P(1));
    const auto resolved = matrix_argument(fcinfo, 2);
    const Matrix& matrix = resolved.get();
    const bool local = PG_GETARG_BOOL(3);
    const auto gaps = alignment_gaps(matrix, fcinfo, 4, false);
    const NativeEncoding encoding = NativeEncoding::database();
    std::vector<std::optional<Score>> output;
    output.reserve(targets.size());
    for (const auto& target : targets) {
      output.push_back(target.has_value()
          ? std::optional<Score>(matrix_score(
                matrix, query, *target, encoding, local, gaps.first, gaps.second))
          : std::nullopt);
    }
    return score_array_datum(output);
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_matrix_cdist(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const auto queries = text_array(PG_GETARG_ARRAYTYPE_P(0));
    const auto targets = text_array(PG_GETARG_ARRAYTYPE_P(1));
    const auto resolved = matrix_argument(fcinfo, 2);
    const Matrix& matrix = resolved.get();
    const bool local = PG_GETARG_BOOL(3);
    const auto gaps = alignment_gaps(matrix, fcinfo, 4, false);
    const NativeEncoding encoding = NativeEncoding::database();
    std::vector<std::optional<Score>> output;
    output.reserve(queries.size() * targets.size());
    for (const auto& query : queries) {
      for (const auto& target : targets) {
        output.push_back(query.has_value() && target.has_value()
            ? std::optional<Score>(matrix_score(
                  matrix, *query, *target, encoding,
                  local, gaps.first, gaps.second))
            : std::nullopt);
      }
    }
    const std::array<int, 2> dimensions{
        static_cast<int>(queries.size()), static_cast<int>(targets.size())};
    return score_array_datum(output, dimensions);
  });
}

extern "C" PGDLLEXPORT Datum
stride_pg_matrix_rank(PG_FUNCTION_ARGS) {
  return catch_errors([&]() -> Datum {
    const auto queries = text_array(PG_GETARG_ARRAYTYPE_P(0));
    const auto targets = text_array(PG_GETARG_ARRAYTYPE_P(1));
    const auto resolved = matrix_argument(fcinfo, 2);
    const Matrix& matrix = resolved.get();
    const bool local = PG_GETARG_BOOL(3);
    const std::int64_t parameter = PG_GETARG_INT64(4);
    const bool top_k = PG_GETARG_BOOL(5);
    const auto gaps = alignment_gaps(matrix, fcinfo, 6, false);
    const NativeEncoding encoding = NativeEncoding::database();
    std::vector<MatrixMatch> matches;
    const std::size_t k = top_k
        ? nonnegative_size(fcinfo, 4, "k") : 0U;
    if (top_k) {
      const std::size_t maximum = targets.empty() ||
              queries.size() <= std::numeric_limits<std::size_t>::max() / targets.size()
          ? queries.size() * targets.size()
          : std::numeric_limits<std::size_t>::max();
      matches.reserve(std::min(k, maximum));
    }
    for (std::size_t query_index = 0; query_index < queries.size(); ++query_index) {
      if (!queries[query_index].has_value()) continue;
      for (std::size_t target_index = 0; target_index < targets.size(); ++target_index) {
        if (!targets[target_index].has_value()) continue;
        const Score score = matrix_score(
            matrix, *queries[query_index], *targets[target_index], encoding,
            local, gaps.first, gaps.second);
        if (top_k) {
          insert_match(matches, {score, query_index, target_index}, k);
        } else if (score >= parameter) {
          matches.push_back({score, query_index, target_index});
        }
      }
    }
    if (top_k) std::sort(matches.begin(), matches.end(), better_match);
    return jsonb_datum(matches_json(matches, queries, targets));
  });
}
