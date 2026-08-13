#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"

#include "stride_align/batch.hpp"
#include "stride_align_duckdb/adapter.hpp"
#include "stride_align_duckdb/registration.hpp"

namespace duckdb::stride_align_extension {
namespace {

namespace batch = ::stride_align::batch;
namespace adapter = ::duckdb::stride_align_adapter;

const LogicalType kStrings = LogicalType::LIST(LogicalType::VARCHAR);

batch::ScoreOptions alignment_options(
    DataChunk& arguments,
    idx_t row,
    idx_t first,
    bool affine = true) {
  batch::ScoreOptions options;
  options.match_score = adapter::score_value(adapter::argument(arguments, first, row));
  options.mismatch_score =
      adapter::score_value(adapter::argument(arguments, first + 1, row));
  options.gap_open_score =
      adapter::score_value(adapter::argument(arguments, first + 2, row));
  options.gap_extend_score = affine
      ? adapter::score_value(adapter::argument(arguments, first + 3, row))
      : options.gap_open_score;
  return options;
}

batch::ScoreOptions jaro_winkler_options(
    DataChunk& arguments,
    idx_t row,
    idx_t first) {
  batch::ScoreOptions options;
  options.prefix_weight =
      adapter::double_value(adapter::argument(arguments, first, row));
  options.prefix_threshold =
      adapter::double_value(adapter::argument(arguments, first + 1, row));
  options.prefix_cap =
      adapter::nonnegative_size(
          adapter::argument(arguments, first + 2, row), "prefix_cap");
  if (!std::isfinite(options.prefix_weight) || options.prefix_weight < 0.0 ||
      !std::isfinite(options.prefix_threshold) ||
      options.prefix_threshold < 0.0 || options.prefix_threshold > 1.0) {
    throw std::invalid_argument(
        "prefix_weight must be non-negative and prefix_threshold must be between 0 and 1");
  }
  return options;
}

batch::Scorer scorer_value(const Value& value) {
  return batch::parse_scorer(value.ToString());
}

batch::ScoreOptions complete_options(
    DataChunk& arguments,
    idx_t row,
    idx_t first) {
  batch::ScoreOptions options = alignment_options(arguments, row, first);
  const batch::ScoreOptions prefix = jaro_winkler_options(
      arguments, row, first + 4U);
  options.prefix_weight = prefix.prefix_weight;
  options.prefix_threshold = prefix.prefix_threshold;
  options.prefix_cap = prefix.prefix_cap;
  return options;
}

template <batch::Scorer Scorer, bool Cutoff>
void FixedScores(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const batch::Text query(
        adapter::string_value(adapter::argument(arguments, 0, row)));
    const auto targets = adapter::text_list(adapter::argument(arguments, 1, row));
    batch::ScoreOptions options;
    if constexpr (Scorer == batch::Scorer::jaro_winkler) {
      if (arguments.ColumnCount() == 5) {
        options = jaro_winkler_options(arguments, row, 2);
      }
    } else if constexpr (
        Scorer == batch::Scorer::smith_waterman ||
        Scorer == batch::Scorer::smith_waterman_normalized ||
        Scorer == batch::Scorer::needleman_wunsch ||
        Scorer == batch::Scorer::needleman_wunsch_normalized) {
      if (arguments.ColumnCount() == 5 || arguments.ColumnCount() == 6) {
        options = alignment_options(
            arguments, row, 2, arguments.ColumnCount() == 6);
      }
    }
    if constexpr (!Cutoff) {
      return adapter::score_list(
          batch::scores(query, targets, Scorer, options),
          batch::is_integral(Scorer));
    } else {
      std::vector<std::optional<double>> output;
      output.reserve(targets.size());
      if constexpr (
          Scorer == batch::Scorer::levenshtein ||
          Scorer == batch::Scorer::indel) {
        const std::size_t cutoff = adapter::nonnegative_size(
            adapter::argument(arguments, 2, row), "score_cutoff");
        for (const auto& target : targets) {
          output.push_back(target.has_value()
              ? std::optional<double>(batch::score(
                    Scorer, query.bytes, target->bytes, options,
                    std::nullopt, cutoff))
              : std::nullopt);
        }
      } else {
        const double cutoff =
            adapter::double_value(adapter::argument(arguments, 2, row));
        if (!std::isfinite(cutoff) || cutoff < 0.0 || cutoff > 1.0) {
          throw std::invalid_argument("score_cutoff must be between 0 and 1");
        }
        for (const auto& target : targets) {
          if (!target.has_value()) {
            output.emplace_back(std::nullopt);
            continue;
          }
          double value = batch::score(
              Scorer, query.bytes, target->bytes, options, cutoff);
          if (value < cutoff) value = 0.0;
          output.emplace_back(value);
        }
      }
      return adapter::score_list(output, batch::is_integral(Scorer));
    }
  });
}

void DynamicScores(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const batch::Text query(
        adapter::string_value(adapter::argument(arguments, 0, row)));
    const auto targets = adapter::text_list(adapter::argument(arguments, 1, row));
    const batch::Scorer scorer = scorer_value(adapter::argument(arguments, 2, row));
    return adapter::score_list(
        batch::scores(query, targets, scorer),
        /*integral=*/false);
  });
}

template <batch::Scorer Scorer, bool Best>
void FixedRank(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const batch::Text query(
        adapter::string_value(adapter::argument(arguments, 0, row)));
    const auto targets = adapter::text_list(adapter::argument(arguments, 1, row));
    const std::size_t k = Best
        ? 1U
        : (arguments.ColumnCount() >= 3
              ? adapter::nonnegative_size(
                    adapter::argument(arguments, 2, row), "k")
              : 5U);
    batch::ScoreOptions options;
    if constexpr (Scorer == batch::Scorer::jaro_winkler) {
      const idx_t first = arguments.ColumnCount() == 5 ? 2 : 3;
      if (arguments.ColumnCount() == first + 3) {
        options = jaro_winkler_options(arguments, row, first);
      }
    } else if constexpr (Scorer == batch::Scorer::smith_waterman) {
      const idx_t first = Best ? 2 : 3;
      if (arguments.ColumnCount() == first + 3 ||
          arguments.ColumnCount() == first + 4) {
        options = alignment_options(
            arguments, row, first,
            arguments.ColumnCount() == first + 4);
      }
    }
    const auto matches = batch::top_k(query, targets, Scorer, k, options);
    if constexpr (Best) {
      if (matches.empty()) {
        return Value(adapter::ranked_match_type(
            batch::is_integral(Scorer)
                ? LogicalType::BIGINT
                : LogicalType::DOUBLE));
      }
      return adapter::ranked_value(
          matches.front(), targets, batch::is_integral(Scorer));
    } else {
      return adapter::ranked_list(
          matches, targets, batch::is_integral(Scorer));
    }
  });
}

template <bool Best>
void DynamicRank(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const batch::Text query(
        adapter::string_value(adapter::argument(arguments, 0, row)));
    const auto targets = adapter::text_list(adapter::argument(arguments, 1, row));
    const batch::Scorer scorer = scorer_value(adapter::argument(arguments, 2, row));
    const std::size_t k = Best
        ? 1U
        : (arguments.ColumnCount() >= 4
              ? adapter::nonnegative_size(
                    adapter::argument(arguments, 3, row), "k")
              : 5U);
    const auto matches = batch::top_k(query, targets, scorer, k);
    if constexpr (Best) {
      if (matches.empty()) {
        return Value(adapter::ranked_match_type(LogicalType::DOUBLE));
      }
      return adapter::ranked_value(matches.front(), targets, false);
    } else {
      return adapter::ranked_list(matches, targets, false);
    }
  });
}

void CDist(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const auto queries = adapter::text_list(adapter::argument(arguments, 0, row));
    const auto targets = adapter::text_list(adapter::argument(arguments, 1, row));
    const batch::Scorer scorer = scorer_value(adapter::argument(arguments, 2, row));
    batch::ScoreOptions options;
    const bool winkler = scorer == batch::Scorer::jaro_winkler;
    if (arguments.ColumnCount() == 6U) {
      options = winkler
          ? jaro_winkler_options(arguments, row, 3U)
          : alignment_options(arguments, row, 3U, false);
    } else if (arguments.ColumnCount() == 7U) {
      options = alignment_options(arguments, row, 3U);
    } else if (arguments.ColumnCount() == 9U) {
      options = alignment_options(arguments, row, 3U, false);
      const batch::ScoreOptions prefix =
          jaro_winkler_options(arguments, row, 6U);
      options.prefix_weight = prefix.prefix_weight;
      options.prefix_threshold = prefix.prefix_threshold;
      options.prefix_cap = prefix.prefix_cap;
    } else if (arguments.ColumnCount() == 10U) {
      options = complete_options(arguments, row, 3U);
    }
    return adapter::distance_matrix_value(
        batch::cdist(queries, targets, scorer, options));
  });
}

void CDistAboveThreshold(
    DataChunk& arguments,
    ExpressionState&,
    Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const auto queries = adapter::text_list(adapter::argument(arguments, 0, row));
    const auto targets = adapter::text_list(adapter::argument(arguments, 1, row));
    const batch::Scorer scorer = scorer_value(adapter::argument(arguments, 2, row));
    const double threshold =
        adapter::double_value(adapter::argument(arguments, 3, row));
    batch::ScoreOptions options;
    const bool winkler = scorer == batch::Scorer::jaro_winkler;
    if (arguments.ColumnCount() == 7U) {
      options = winkler
          ? jaro_winkler_options(arguments, row, 4U)
          : alignment_options(arguments, row, 4U, false);
    } else if (arguments.ColumnCount() == 8U) {
      options = alignment_options(arguments, row, 4U);
    } else if (arguments.ColumnCount() == 10U) {
      options = alignment_options(arguments, row, 4U, false);
      const batch::ScoreOptions prefix =
          jaro_winkler_options(arguments, row, 7U);
      options.prefix_weight = prefix.prefix_weight;
      options.prefix_threshold = prefix.prefix_threshold;
      options.prefix_cap = prefix.prefix_cap;
    } else if (arguments.ColumnCount() == 11U) {
      options = complete_options(arguments, row, 4U);
    }
    return adapter::matrix_matches_value(
        batch::cdist_above_threshold(
            queries, targets, scorer, threshold, options),
        queries, targets);
  });
}

void CDistTopK(DataChunk& arguments, ExpressionState&, Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const auto queries = adapter::text_list(adapter::argument(arguments, 0, row));
    const auto targets = adapter::text_list(adapter::argument(arguments, 1, row));
    const batch::Scorer scorer = scorer_value(adapter::argument(arguments, 2, row));
    const std::size_t k = adapter::nonnegative_size(
        adapter::argument(arguments, 3, row), "k");
    const bool reject_duplicates = arguments.ColumnCount() >= 5U &&
        adapter::boolean_value(adapter::argument(arguments, 4, row));
    batch::ScoreOptions options;
    const bool winkler = scorer == batch::Scorer::jaro_winkler;
    if (arguments.ColumnCount() == 8U) {
      options = winkler
          ? jaro_winkler_options(arguments, row, 5U)
          : alignment_options(arguments, row, 5U, false);
    } else if (arguments.ColumnCount() == 9U) {
      options = alignment_options(arguments, row, 5U);
    } else if (arguments.ColumnCount() == 11U) {
      options = alignment_options(arguments, row, 5U, false);
      const batch::ScoreOptions prefix =
          jaro_winkler_options(arguments, row, 8U);
      options.prefix_weight = prefix.prefix_weight;
      options.prefix_threshold = prefix.prefix_threshold;
      options.prefix_cap = prefix.prefix_cap;
    } else if (arguments.ColumnCount() == 12U) {
      options = complete_options(arguments, row, 5U);
    }
    return adapter::matrix_matches_value(
        batch::cdist_top_k(
            queries, targets, scorer, k, reject_duplicates, options),
        queries, targets);
  });
}

void CDistTopKPerQuery(
    DataChunk& arguments,
    ExpressionState&,
    Vector& result) {
  adapter::execute_rows(arguments, result, [&](idx_t row) {
    const auto queries = adapter::text_list(adapter::argument(arguments, 0, row));
    const auto targets = adapter::text_list(adapter::argument(arguments, 1, row));
    const batch::Scorer scorer = scorer_value(adapter::argument(arguments, 2, row));
    const std::size_t k = arguments.ColumnCount() >= 4
        ? adapter::nonnegative_size(adapter::argument(arguments, 3, row), "k")
        : 5U;
    batch::ScoreOptions options;
    if (arguments.ColumnCount() == 8U) {
      options = jaro_winkler_options(arguments, row, 5U);
    }
    // DuckDB has the complete lists before execution. Adaptive length and
    // heap cutoffs are therefore always safe; a fifth `pruning` argument is
    // accepted for Python/R call-shape parity but cannot make the optimized
    // path less correct.
    return adapter::per_query_matches_value(
        batch::cdist_top_k_per_query(queries, targets, scorer, k, options),
        queries, targets);
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

template <batch::Scorer Scorer>
void register_scores(ExtensionLoader& loader, const char* name) {
  const LogicalType score_type = batch::is_integral(Scorer)
      ? LogicalType::BIGINT
      : LogicalType::DOUBLE;
  ScalarFunctionSet functions(name);
  functions.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, kStrings},
      LogicalType::LIST(score_type),
      FixedScores<Scorer, false>));
  if constexpr (
      Scorer == batch::Scorer::levenshtein ||
      Scorer == batch::Scorer::indel) {
    functions.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, kStrings, LogicalType::BIGINT},
        LogicalType::LIST(score_type), FixedScores<Scorer, true>));
  } else if constexpr (
      Scorer == batch::Scorer::levenshtein_normalized ||
      Scorer == batch::Scorer::indel_normalized) {
    functions.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, kStrings, LogicalType::DOUBLE},
        LogicalType::LIST(score_type), FixedScores<Scorer, true>));
  }
  if constexpr (Scorer == batch::Scorer::jaro_winkler) {
    functions.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, kStrings, LogicalType::DOUBLE,
         LogicalType::DOUBLE, LogicalType::BIGINT},
        LogicalType::LIST(score_type),
        FixedScores<Scorer, false>));
  } else if constexpr (
      Scorer == batch::Scorer::smith_waterman ||
      Scorer == batch::Scorer::smith_waterman_normalized ||
      Scorer == batch::Scorer::needleman_wunsch ||
      Scorer == batch::Scorer::needleman_wunsch_normalized) {
    functions.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, kStrings, LogicalType::BIGINT,
         LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::LIST(score_type), FixedScores<Scorer, false>));
    functions.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, kStrings, LogicalType::BIGINT,
         LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::LIST(score_type),
        FixedScores<Scorer, false>));
  }
  loader.RegisterFunction(std::move(functions));
}

template <batch::Scorer Scorer>
void register_rank(
    ExtensionLoader& loader,
    const char* top_k_name,
    const char* best_name) {
  const LogicalType score_type = batch::is_integral(Scorer)
      ? LogicalType::BIGINT
      : LogicalType::DOUBLE;
  const LogicalType match_type = adapter::ranked_match_type(score_type);
  ScalarFunctionSet top_k(top_k_name);
  top_k.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, kStrings},
      LogicalType::LIST(match_type),
      FixedRank<Scorer, false>));
  top_k.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, kStrings, LogicalType::BIGINT},
      LogicalType::LIST(match_type),
      FixedRank<Scorer, false>));
  if constexpr (Scorer == batch::Scorer::jaro_winkler) {
    top_k.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, kStrings, LogicalType::BIGINT,
         LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BIGINT},
        LogicalType::LIST(match_type),
        FixedRank<Scorer, false>));
  } else if constexpr (Scorer == batch::Scorer::smith_waterman) {
    top_k.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, kStrings, LogicalType::BIGINT,
         LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::LIST(match_type), FixedRank<Scorer, false>));
    top_k.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, kStrings, LogicalType::BIGINT,
         LogicalType::BIGINT, LogicalType::BIGINT,
         LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::LIST(match_type),
        FixedRank<Scorer, false>));
  }
  loader.RegisterFunction(std::move(top_k));

  ScalarFunctionSet best(best_name);
  best.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, kStrings}, match_type,
      FixedRank<Scorer, true>));
  if constexpr (Scorer == batch::Scorer::jaro_winkler) {
    best.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, kStrings, LogicalType::DOUBLE,
         LogicalType::DOUBLE, LogicalType::BIGINT},
        match_type, FixedRank<Scorer, true>));
  } else if constexpr (Scorer == batch::Scorer::smith_waterman) {
    best.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, kStrings, LogicalType::BIGINT,
         LogicalType::BIGINT, LogicalType::BIGINT},
        match_type, FixedRank<Scorer, true>));
    best.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, kStrings, LogicalType::BIGINT,
         LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
        match_type, FixedRank<Scorer, true>));
  }
  loader.RegisterFunction(std::move(best));
}

void register_cdist(ExtensionLoader& loader) {
  ScalarFunctionSet cdist("stride_cdist");
  cdist.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR},
      LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), CDist));
  cdist.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR,
       LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BIGINT},
      LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), CDist));
  cdist.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR,
       LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
      LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), CDist));
  cdist.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR,
       LogicalType::BIGINT, LogicalType::BIGINT,
       LogicalType::BIGINT, LogicalType::BIGINT},
      LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), CDist));
  cdist.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR,
       LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT,
       LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BIGINT},
      LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), CDist));
  cdist.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR,
       LogicalType::BIGINT, LogicalType::BIGINT,
       LogicalType::BIGINT, LogicalType::BIGINT,
       LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BIGINT},
      LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), CDist));
  loader.RegisterFunction(std::move(cdist));

  ScalarFunctionSet above("stride_cdist_above_threshold");
  above.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR, LogicalType::DOUBLE},
      LogicalType::LIST(adapter::matrix_match_type()),
      CDistAboveThreshold));
  above.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR, LogicalType::DOUBLE,
       LogicalType::BIGINT, LogicalType::BIGINT,
       LogicalType::BIGINT, LogicalType::BIGINT,
       LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BIGINT},
      LogicalType::LIST(adapter::matrix_match_type()),
      CDistAboveThreshold));
  above.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR, LogicalType::DOUBLE,
       LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BIGINT},
      LogicalType::LIST(adapter::matrix_match_type()),
      CDistAboveThreshold));
  above.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR, LogicalType::DOUBLE,
       LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
      LogicalType::LIST(adapter::matrix_match_type()),
      CDistAboveThreshold));
  above.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR, LogicalType::DOUBLE,
       LogicalType::BIGINT, LogicalType::BIGINT,
       LogicalType::BIGINT, LogicalType::BIGINT},
      LogicalType::LIST(adapter::matrix_match_type()),
      CDistAboveThreshold));
  above.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR, LogicalType::DOUBLE,
       LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT,
       LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BIGINT},
      LogicalType::LIST(adapter::matrix_match_type()),
      CDistAboveThreshold));
  loader.RegisterFunction(std::move(above));

  ScalarFunctionSet global_top_k("stride_cdist_top_k");
  global_top_k.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR, LogicalType::BIGINT},
      LogicalType::LIST(adapter::matrix_match_type()), CDistTopK));
  global_top_k.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR,
       LogicalType::BIGINT, LogicalType::BOOLEAN,
       LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BIGINT},
      LogicalType::LIST(adapter::matrix_match_type()), CDistTopK));
  global_top_k.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR,
       LogicalType::BIGINT, LogicalType::BOOLEAN,
       LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
      LogicalType::LIST(adapter::matrix_match_type()), CDistTopK));
  global_top_k.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR,
       LogicalType::BIGINT, LogicalType::BOOLEAN,
       LogicalType::BIGINT, LogicalType::BIGINT,
       LogicalType::BIGINT, LogicalType::BIGINT},
      LogicalType::LIST(adapter::matrix_match_type()), CDistTopK));
  global_top_k.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR,
       LogicalType::BIGINT, LogicalType::BOOLEAN,
       LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT,
       LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BIGINT},
      LogicalType::LIST(adapter::matrix_match_type()), CDistTopK));
  global_top_k.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR,
       LogicalType::BIGINT, LogicalType::BOOLEAN,
       LogicalType::BIGINT, LogicalType::BIGINT,
       LogicalType::BIGINT, LogicalType::BIGINT,
       LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BIGINT},
      LogicalType::LIST(adapter::matrix_match_type()), CDistTopK));
  global_top_k.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR,
       LogicalType::BIGINT, LogicalType::BOOLEAN},
      LogicalType::LIST(adapter::matrix_match_type()), CDistTopK));
  loader.RegisterFunction(std::move(global_top_k));

  ScalarFunctionSet per_query("stride_cdist_top_k_per_query");
  per_query.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR},
      LogicalType::LIST(adapter::per_query_match_type()),
      CDistTopKPerQuery));
  per_query.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR,
       LogicalType::BIGINT, LogicalType::BOOLEAN,
       LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BIGINT},
      LogicalType::LIST(adapter::per_query_match_type()),
      CDistTopKPerQuery));
  per_query.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR, LogicalType::BIGINT},
      LogicalType::LIST(adapter::per_query_match_type()),
      CDistTopKPerQuery));
  per_query.AddFunction(ScalarFunction(
      {kStrings, kStrings, LogicalType::VARCHAR,
       LogicalType::BIGINT, LogicalType::BOOLEAN},
      LogicalType::LIST(adapter::per_query_match_type()),
      CDistTopKPerQuery));
  loader.RegisterFunction(std::move(per_query));
}

}  // namespace

void RegisterBatchFunctions(ExtensionLoader& loader) {
  register_scores<batch::Scorer::levenshtein>(
      loader, "stride_levenshtein_scores");
  register_scores<batch::Scorer::levenshtein_normalized>(
      loader, "stride_levenshtein_normalized_scores");
  register_scores<batch::Scorer::damerau_levenshtein>(
      loader, "stride_damerau_levenshtein_scores");
  register_scores<batch::Scorer::damerau_levenshtein_normalized>(
      loader, "stride_damerau_levenshtein_normalized_scores");
  register_scores<batch::Scorer::hamming>(loader, "stride_hamming_scores");
  register_scores<batch::Scorer::hamming_normalized>(
      loader, "stride_hamming_normalized_scores");
  register_scores<batch::Scorer::jaro>(loader, "stride_jaro_similarities");
  register_scores<batch::Scorer::jaro_winkler>(
      loader, "stride_jaro_winkler_similarities");
  register_scores<batch::Scorer::indel>(loader, "stride_indel_scores");
  register_scores<batch::Scorer::indel_normalized>(
      loader, "stride_indel_normalized_scores");
  register_scores<batch::Scorer::true_damerau_levenshtein>(
      loader, "stride_true_damerau_levenshtein_scores");
  register_scores<batch::Scorer::true_damerau_levenshtein_normalized>(
      loader, "stride_true_damerau_levenshtein_normalized_scores");
  register_scores<batch::Scorer::smith_waterman>(
      loader, "stride_smith_waterman_scores");
  register_scores<batch::Scorer::smith_waterman_normalized>(
      loader, "stride_smith_waterman_normalized_scores");
  register_scores<batch::Scorer::smith_waterman>(
      loader, "stride_smith_waterman_farrar_scores");
  register_scores<batch::Scorer::smith_waterman_normalized>(
      loader, "stride_smith_waterman_farrar_normalized_scores");
  register_scores<batch::Scorer::needleman_wunsch>(
      loader, "stride_needleman_wunsch_scores");
  register_scores<batch::Scorer::needleman_wunsch_normalized>(
      loader, "stride_needleman_wunsch_normalized_scores");

  register_function(
      loader, "stride_scores",
      {LogicalType::VARCHAR, kStrings, LogicalType::VARCHAR},
      LogicalType::LIST(LogicalType::DOUBLE), DynamicScores);

#define STRIDE_REGISTER_RANK(metric, stem)                                      \
  register_rank<batch::Scorer::metric>(                                         \
      loader, "stride_" stem "_top_k", "stride_" stem "_best")
  STRIDE_REGISTER_RANK(levenshtein, "levenshtein");
  STRIDE_REGISTER_RANK(levenshtein_normalized, "levenshtein_normalized");
  STRIDE_REGISTER_RANK(damerau_levenshtein, "damerau_levenshtein");
  STRIDE_REGISTER_RANK(
      damerau_levenshtein_normalized,
      "damerau_levenshtein_normalized");
  STRIDE_REGISTER_RANK(hamming, "hamming");
  STRIDE_REGISTER_RANK(hamming_normalized, "hamming_normalized");
  STRIDE_REGISTER_RANK(indel, "indel");
  STRIDE_REGISTER_RANK(indel_normalized, "indel_normalized");
  STRIDE_REGISTER_RANK(
      true_damerau_levenshtein,
      "true_damerau_levenshtein");
  STRIDE_REGISTER_RANK(
      true_damerau_levenshtein_normalized,
      "true_damerau_levenshtein_normalized");
  STRIDE_REGISTER_RANK(jaro, "jaro");
  STRIDE_REGISTER_RANK(jaro_winkler, "jaro_winkler");
  STRIDE_REGISTER_RANK(smith_waterman, "smith_waterman");
#undef STRIDE_REGISTER_RANK

  ScalarFunctionSet extract("stride_extract");
  extract.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, kStrings, LogicalType::VARCHAR},
      LogicalType::LIST(adapter::ranked_match_type(LogicalType::DOUBLE)),
      DynamicRank<false>));
  extract.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, kStrings, LogicalType::VARCHAR,
       LogicalType::BIGINT},
      LogicalType::LIST(adapter::ranked_match_type(LogicalType::DOUBLE)),
      DynamicRank<false>));
  loader.RegisterFunction(std::move(extract));
  register_function(
      loader, "stride_extract_best",
      {LogicalType::VARCHAR, kStrings, LogicalType::VARCHAR},
      adapter::ranked_match_type(LogicalType::DOUBLE),
      DynamicRank<true>);

  register_cdist(loader);
}

}  // namespace duckdb::stride_align_extension
