#define DUCKDB_EXTENSION_MAIN

#include "stride_align_extension.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <string_view>

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include "stride_align_duckdb/target_profile.hpp"
#include "stride_align_duckdb/registration.hpp"
#include "stride_align/batch.hpp"
#include "stride_align/core.hpp"

#ifndef STRIDE_ALIGN_DUCKDB_SIMD_LEVEL
#define STRIDE_ALIGN_DUCKDB_SIMD_LEVEL "unspecified"
#endif

namespace duckdb {
namespace {

using PreparedPair = ::stride_align::utf8::PreparedPair;

inline std::string_view StringView(const string_t& value) noexcept {
  return std::string_view(value.GetDataUnsafe(), value.GetSize());
}

using IntegerScorer = std::int64_t (*)(const PreparedPair&);
using DoubleScorer = double (*)(const PreparedPair&);

template <IntegerScorer Scorer>
void ExecuteInteger(
    DataChunk& arguments,
    ExpressionState&,
    Vector& result) {
  BinaryExecutor::Execute<string_t, string_t, std::int64_t>(
      arguments.data[0], arguments.data[1], result, arguments.size(),
      [](string_t query, string_t target) {
        try {
          // PreparedPair borrows clean ASCII bytes. Keep DuckDB's string_t
          // values alive until scoring has consumed those spans.
          const auto pair = ::stride_align::utf8::prepare_pair(
              StringView(query), StringView(target));
          return Scorer(pair);
        } catch (const std::exception& error) {
          throw InvalidInputException(error.what());
        }
      });
}

template <DoubleScorer Scorer>
void ExecuteDouble(
    DataChunk& arguments,
    ExpressionState&,
    Vector& result) {
  BinaryExecutor::Execute<string_t, string_t, double>(
      arguments.data[0], arguments.data[1], result, arguments.size(),
      [](string_t query, string_t target) {
        try {
          const auto pair = ::stride_align::utf8::prepare_pair(
              StringView(query), StringView(target));
          return Scorer(pair);
        } catch (const std::exception& error) {
          throw InvalidInputException(error.what());
        }
      });
}

template <bool Local, bool Normalized>
void ExecuteAlignment(DataChunk& arguments, ExpressionState&, Vector& result) {
  arguments.Flatten();
  for (idx_t row = 0; row < arguments.size(); ++row) {
    bool has_null = false;
    for (idx_t column = 0; column < arguments.ColumnCount(); ++column) {
      if (arguments.data[column].GetValue(row).IsNull()) {
        has_null = true;
        break;
      }
    }
    if (has_null) {
      result.SetValue(row, Value());
      continue;
    }
    try {
      const std::string query =
          arguments.data[0].GetValue(row).GetValue<std::string>();
      const std::string target =
          arguments.data[1].GetValue(row).GetValue<std::string>();
      const ::stride_align::Score match =
          arguments.data[2].GetValue(row).GetValue<std::int64_t>();
      const ::stride_align::Score mismatch =
          arguments.data[3].GetValue(row).GetValue<std::int64_t>();
      const ::stride_align::Score gap_open =
          arguments.data[4].GetValue(row).GetValue<std::int64_t>();
      const ::stride_align::Score gap_extend = arguments.ColumnCount() == 5
          ? gap_open
          : arguments.data[5].GetValue(row).GetValue<std::int64_t>();
      const auto pair = ::stride_align::utf8::prepare_pair(query, target);
      const ::stride_align::Score raw = Local
          ? (gap_open == gap_extend
                ? ::stride_align::core::smith_waterman_score(
                      pair, match, mismatch, gap_open)
                : ::stride_align::core::smith_waterman_affine_score(
                      pair, match, mismatch, gap_open, gap_extend))
          : (gap_open == gap_extend
                ? ::stride_align::core::needleman_wunsch_score(
                      pair, match, mismatch, gap_open)
                : ::stride_align::core::needleman_wunsch_affine_score(
                      pair, match, mismatch, gap_open, gap_extend));
      if constexpr (Normalized) {
        result.SetValue(row, Value::DOUBLE(
            ::stride_align::batch::normalized_alignment(
                raw, pair.query_size(), pair.target_size(), Local, match)));
      } else {
        result.SetValue(row, Value::BIGINT(raw));
      }
    } catch (const std::exception& error) {
      throw InvalidInputException(error.what());
    }
  }
}

void ExecuteJaroWinkler(
    DataChunk& arguments,
    ExpressionState&,
    Vector& result) {
  arguments.Flatten();
  for (idx_t row = 0; row < arguments.size(); ++row) {
    bool has_null = false;
    for (idx_t column = 0; column < arguments.ColumnCount(); ++column) {
      if (arguments.data[column].GetValue(row).IsNull()) has_null = true;
    }
    if (has_null) {
      result.SetValue(row, Value());
      continue;
    }
    try {
      const std::string query =
          arguments.data[0].GetValue(row).GetValue<std::string>();
      const std::string target =
          arguments.data[1].GetValue(row).GetValue<std::string>();
      const double weight = arguments.data[2].GetValue(row).GetValue<double>();
      const double threshold = arguments.data[3].GetValue(row).GetValue<double>();
      const std::int64_t cap =
          arguments.data[4].GetValue(row).GetValue<std::int64_t>();
      if (!std::isfinite(weight) || weight < 0.0 ||
          !std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0 ||
          cap < 0) {
        throw std::invalid_argument("invalid Jaro-Winkler parameters");
      }
      const auto pair = ::stride_align::utf8::prepare_pair(query, target);
      result.SetValue(row, Value::DOUBLE(
          ::stride_align::core::jaro_winkler_similarity(
              pair, weight, threshold, static_cast<std::size_t>(cap))));
    } catch (const std::exception& error) {
      throw InvalidInputException(error.what());
    }
  }
}

template <bool Normalized, bool Indel>
void ExecuteCutoffDistance(
    DataChunk& arguments,
    ExpressionState&,
    Vector& result) {
  arguments.Flatten();
  for (idx_t row = 0; row < arguments.size(); ++row) {
    bool has_null = false;
    for (idx_t column = 0; column < arguments.ColumnCount(); ++column) {
      if (arguments.data[column].GetValue(row).IsNull()) has_null = true;
    }
    if (has_null) {
      result.SetValue(row, Value());
      continue;
    }
    try {
      const std::string query =
          arguments.data[0].GetValue(row).GetValue<std::string>();
      const std::string target =
          arguments.data[1].GetValue(row).GetValue<std::string>();
      const auto pair = ::stride_align::utf8::prepare_pair(query, target);
      if constexpr (Normalized) {
        const double cutoff =
            arguments.data[2].GetValue(row).GetValue<double>();
        if (!std::isfinite(cutoff) || cutoff < 0.0 || cutoff > 1.0) {
          throw std::invalid_argument("score_cutoff must be between 0 and 1");
        }
        const std::size_t maximum = Indel
            ? pair.query_size() + pair.target_size()
            : std::max(pair.query_size(), pair.target_size());
        const std::size_t distance_cutoff =
            ::stride_align::batch::normalized_distance_cutoff(
                cutoff, maximum);
        const std::size_t distance = Indel
            ? ::stride_align::core::indel_distance(pair, distance_cutoff)
            : ::stride_align::core::levenshtein_distance(pair, distance_cutoff);
        double similarity = Indel
            ? ::stride_align::indel::normalize(
                  distance, pair.query_size(), pair.target_size())
            : ::stride_align::levenshtein::normalize(
                  distance, pair.query_size(), pair.target_size());
        if (similarity < cutoff) similarity = 0.0;
        result.SetValue(row, Value::DOUBLE(similarity));
      } else {
        const std::int64_t input =
            arguments.data[2].GetValue(row).GetValue<std::int64_t>();
        if (input < 0) {
          throw std::invalid_argument("score_cutoff must be non-negative");
        }
        const std::size_t cutoff = static_cast<std::size_t>(input);
        const std::size_t distance = Indel
            ? ::stride_align::core::indel_distance(pair, cutoff)
            : ::stride_align::core::levenshtein_distance(pair, cutoff);
        result.SetValue(row, Value::BIGINT(static_cast<std::int64_t>(distance)));
      }
    } catch (const std::exception& error) {
      throw InvalidInputException(error.what());
    }
  }
}

std::int64_t Levenshtein(const PreparedPair& pair) {
  return static_cast<std::int64_t>(::stride_align::core::levenshtein_distance(pair));
}

double LevenshteinSimilarity(const PreparedPair& pair) {
  return ::stride_align::core::levenshtein_similarity(pair);
}

std::int64_t Osa(const PreparedPair& pair) {
  return static_cast<std::int64_t>(::stride_align::core::osa_distance(pair));
}

double OsaSimilarity(const PreparedPair& pair) {
  return ::stride_align::core::osa_similarity(pair);
}

std::int64_t TrueDamerauLevenshtein(const PreparedPair& pair) {
  return static_cast<std::int64_t>(
      ::stride_align::core::true_damerau_levenshtein_distance(pair));
}

double TrueDamerauLevenshteinSimilarity(const PreparedPair& pair) {
  return ::stride_align::core::true_damerau_levenshtein_similarity(pair);
}

std::int64_t Indel(const PreparedPair& pair) {
  return static_cast<std::int64_t>(::stride_align::core::indel_distance(pair));
}

double IndelSimilarity(const PreparedPair& pair) {
  return ::stride_align::core::indel_similarity(pair);
}

std::int64_t Hamming(const PreparedPair& pair) {
  return static_cast<std::int64_t>(::stride_align::core::hamming_distance(pair));
}

double HammingSimilarity(const PreparedPair& pair) {
  return ::stride_align::core::hamming_similarity(pair);
}

double Jaro(const PreparedPair& pair) {
  return ::stride_align::core::jaro_similarity(pair);
}

double JaroWinkler(const PreparedPair& pair) {
  return ::stride_align::core::jaro_winkler_similarity(pair);
}

std::int64_t SmithWaterman(const PreparedPair& pair) {
  return ::stride_align::core::smith_waterman_score(pair);
}

std::int64_t NeedlemanWunsch(const PreparedPair& pair) {
  return ::stride_align::core::needleman_wunsch_score(pair);
}

std::int64_t SmithWatermanAffine(const PreparedPair& pair) {
  return ::stride_align::core::smith_waterman_affine_score(pair);
}

std::int64_t NeedlemanWunschAffine(const PreparedPair& pair) {
  return ::stride_align::core::needleman_wunsch_affine_score(pair);
}

double SmithWatermanSimilarity(const PreparedPair& pair) {
  return ::stride_align::batch::normalized_alignment(
      ::stride_align::core::smith_waterman_score(pair),
      pair.query_size(), pair.target_size(), true, 2);
}

double NeedlemanWunschSimilarity(const PreparedPair& pair) {
  return ::stride_align::batch::normalized_alignment(
      ::stride_align::core::needleman_wunsch_score(pair),
      pair.query_size(), pair.target_size(), false, 2);
}

void SimdLevel(DataChunk&, ExpressionState&, Vector& result) {
  result.SetVectorType(VectorType::CONSTANT_VECTOR);
  ConstantVector::GetData<string_t>(result)[0] =
      StringVector::AddString(result, STRIDE_ALIGN_DUCKDB_SIMD_LEVEL);
}

void AvailableBackends(DataChunk&, ExpressionState&, Vector& result) {
  result.SetVectorType(VectorType::CONSTANT_VECTOR);
  result.SetValue(0, Value::LIST(
      LogicalType::VARCHAR, {Value(STRIDE_ALIGN_DUCKDB_SIMD_LEVEL)}));
}

void BackendIsAvailable(
    DataChunk& arguments,
    ExpressionState&,
    Vector& result) {
  UnaryExecutor::Execute<string_t, bool>(
      arguments.data[0], result, arguments.size(), [](string_t value) {
        return StringView(value) == STRIDE_ALIGN_DUCKDB_SIMD_LEVEL;
      });
}

template <IntegerScorer Scorer>
void RegisterInteger(ExtensionLoader& loader, const char* name) {
  loader.RegisterFunction(ScalarFunction(
      name,
      {LogicalType::VARCHAR, LogicalType::VARCHAR},
      LogicalType::BIGINT,
      ExecuteInteger<Scorer>));
}

template <DoubleScorer Scorer>
void RegisterDouble(ExtensionLoader& loader, const char* name) {
  loader.RegisterFunction(ScalarFunction(
      name,
      {LogicalType::VARCHAR, LogicalType::VARCHAR},
      LogicalType::DOUBLE,
      ExecuteDouble<Scorer>));
}

template <
    bool Normalized,
    bool Indel,
    IntegerScorer IntegerFunction,
    DoubleScorer DoubleFunction>
void RegisterCutoffDistance(
    ExtensionLoader& loader,
    const char* name) {
  ScalarFunctionSet functions(name);
  if constexpr (Normalized) {
    functions.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::DOUBLE,
        ExecuteDouble<DoubleFunction>));
    functions.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::DOUBLE},
        LogicalType::DOUBLE, ExecuteCutoffDistance<true, Indel>));
  } else {
    functions.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BIGINT,
        ExecuteInteger<IntegerFunction>));
    functions.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT},
        LogicalType::BIGINT, ExecuteCutoffDistance<false, Indel>));
  }
  loader.RegisterFunction(std::move(functions));
}

template <bool Local, bool Normalized>
void RegisterAlignmentOverload(ExtensionLoader& loader, const char* name) {
  ScalarFunctionSet functions(name);
  if constexpr (Normalized) {
    functions.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::DOUBLE,
        ExecuteDouble<Local ? SmithWatermanSimilarity : NeedlemanWunschSimilarity>));
  } else {
    functions.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BIGINT,
        ExecuteInteger<Local ? SmithWaterman : NeedlemanWunsch>));
  }
  functions.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::VARCHAR,
       LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
      Normalized ? LogicalType::DOUBLE : LogicalType::BIGINT,
      ExecuteAlignment<Local, Normalized>));
  functions.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::VARCHAR,
       LogicalType::BIGINT, LogicalType::BIGINT,
       LogicalType::BIGINT, LogicalType::BIGINT},
      Normalized ? LogicalType::DOUBLE : LogicalType::BIGINT,
      ExecuteAlignment<Local, Normalized>));
  loader.RegisterFunction(std::move(functions));
}

void RegisterJaroWinklerOverload(ExtensionLoader& loader, const char* name) {
  ScalarFunctionSet functions(name);
  functions.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::DOUBLE,
      ExecuteDouble<JaroWinkler>));
  functions.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::VARCHAR,
       LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BIGINT},
      LogicalType::DOUBLE, ExecuteJaroWinkler));
  loader.RegisterFunction(std::move(functions));
}

}  // namespace

static void LoadInternal(ExtensionLoader& loader) {
  RegisterInteger<Levenshtein>(loader, "stride_levenshtein");
  RegisterDouble<LevenshteinSimilarity>(loader, "stride_levenshtein_similarity");
  RegisterInteger<Osa>(loader, "stride_osa");
  RegisterDouble<OsaSimilarity>(loader, "stride_osa_similarity");
  RegisterInteger<TrueDamerauLevenshtein>(
      loader, "stride_true_damerau_levenshtein");
  RegisterDouble<TrueDamerauLevenshteinSimilarity>(
      loader, "stride_true_damerau_levenshtein_similarity");
  RegisterInteger<Indel>(loader, "stride_indel");
  RegisterDouble<IndelSimilarity>(loader, "stride_indel_similarity");
  RegisterInteger<Hamming>(loader, "stride_hamming");
  RegisterDouble<HammingSimilarity>(loader, "stride_hamming_similarity");
  RegisterDouble<Jaro>(loader, "stride_jaro");
  RegisterJaroWinklerOverload(loader, "stride_jaro_winkler");
  RegisterInteger<SmithWaterman>(loader, "stride_smith_waterman");
  RegisterInteger<NeedlemanWunsch>(loader, "stride_needleman_wunsch");
  RegisterInteger<SmithWatermanAffine>(loader, "stride_smith_waterman_affine");
  RegisterInteger<NeedlemanWunschAffine>(loader, "stride_needleman_wunsch_affine");
  RegisterCutoffDistance<false, false, Levenshtein, LevenshteinSimilarity>(
      loader, "stride_levenshtein_score");
  RegisterCutoffDistance<true, false, Levenshtein, LevenshteinSimilarity>(
      loader, "stride_levenshtein_normalized_score");
  RegisterInteger<Osa>(loader, "stride_damerau_levenshtein_score");
  RegisterDouble<OsaSimilarity>(
      loader, "stride_damerau_levenshtein_normalized_score");
  RegisterInteger<TrueDamerauLevenshtein>(
      loader, "stride_true_damerau_levenshtein_score");
  RegisterDouble<TrueDamerauLevenshteinSimilarity>(
      loader, "stride_true_damerau_levenshtein_normalized_score");
  RegisterCutoffDistance<false, true, Indel, IndelSimilarity>(
      loader, "stride_indel_score");
  RegisterCutoffDistance<true, true, Indel, IndelSimilarity>(
      loader, "stride_indel_normalized_score");
  RegisterInteger<Hamming>(loader, "stride_hamming_score");
  RegisterDouble<HammingSimilarity>(loader, "stride_hamming_normalized_score");
  RegisterDouble<Jaro>(loader, "stride_jaro_similarity");
  RegisterJaroWinklerOverload(loader, "stride_jaro_winkler_similarity");
  RegisterAlignmentOverload<true, false>(
      loader, "stride_smith_waterman_score");
  RegisterAlignmentOverload<true, true>(
      loader, "stride_smith_waterman_normalized_score");
  RegisterAlignmentOverload<true, false>(
      loader, "stride_smith_waterman_farrar_score");
  RegisterAlignmentOverload<true, true>(
      loader, "stride_smith_waterman_farrar_normalized_score");
  RegisterAlignmentOverload<false, false>(
      loader, "stride_needleman_wunsch_score");
  RegisterAlignmentOverload<false, true>(
      loader, "stride_needleman_wunsch_normalized_score");
  loader.RegisterFunction(ScalarFunction(
      "stride_align_simd_level", {}, LogicalType::VARCHAR, SimdLevel));
  loader.RegisterFunction(ScalarFunction(
      "stride_detect_best_backend", {}, LogicalType::VARCHAR, SimdLevel));
  loader.RegisterFunction(ScalarFunction(
      "stride_available_backends", {},
      LogicalType::LIST(LogicalType::VARCHAR), AvailableBackends));
  loader.RegisterFunction(ScalarFunction(
      "stride_backend_is_available", {LogicalType::VARCHAR},
      LogicalType::BOOLEAN, BackendIsAvailable));
  stride_align_extension::RegisterBatchFunctions(loader);
  stride_align_extension::RegisterAlgorithmFunctions(loader);
  stride_align_extension::RegisterMatrixFunctions(loader);
}

void StrideAlignExtension::Load(ExtensionLoader& loader) {
  LoadInternal(loader);
}

std::string StrideAlignExtension::Name() {
  return "stride_align";
}

std::string StrideAlignExtension::Version() const {
#ifdef EXT_VERSION_STRIDE_ALIGN
  return EXT_VERSION_STRIDE_ALIGN;
#else
  return "";
#endif
}

}  // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(stride_align, loader) {
  duckdb::LoadInternal(loader);
}

}
