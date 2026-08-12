#define DUCKDB_EXTENSION_MAIN

#include "stride_align_extension.hpp"

#include <cstdint>
#include <string_view>

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include "stride_align_duckdb/target_profile.hpp"
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

void SimdLevel(DataChunk&, ExpressionState&, Vector& result) {
  result.SetVectorType(VectorType::CONSTANT_VECTOR);
  ConstantVector::GetData<string_t>(result)[0] =
      StringVector::AddString(result, STRIDE_ALIGN_DUCKDB_SIMD_LEVEL);
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
  RegisterDouble<JaroWinkler>(loader, "stride_jaro_winkler");
  RegisterInteger<SmithWaterman>(loader, "stride_smith_waterman");
  RegisterInteger<NeedlemanWunsch>(loader, "stride_needleman_wunsch");
  RegisterInteger<SmithWatermanAffine>(loader, "stride_smith_waterman_affine");
  RegisterInteger<NeedlemanWunschAffine>(loader, "stride_needleman_wunsch_affine");
  loader.RegisterFunction(ScalarFunction(
      "stride_align_simd_level", {}, LogicalType::VARCHAR, SimdLevel));
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
