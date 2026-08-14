#pragma once

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

#include "stride_align/core.hpp"

namespace stride_align::batch {

// This is the host-neutral scorer catalog shared by language adapters.  The
// numeric values intentionally match the Python Scorer enum.
enum class Scorer : std::uint8_t {
  levenshtein = 0,
  levenshtein_normalized = 1,
  damerau_levenshtein = 2,
  damerau_levenshtein_normalized = 3,
  hamming = 4,
  hamming_normalized = 5,
  jaro = 6,
  jaro_winkler = 7,
  indel = 8,
  indel_normalized = 9,
  true_damerau_levenshtein = 10,
  true_damerau_levenshtein_normalized = 11,
  smith_waterman = 12,
  smith_waterman_normalized = 13,
  needleman_wunsch = 14,
  needleman_wunsch_normalized = 15,
};

struct ScoreOptions {
  Score match_score = 2;
  Score mismatch_score = -1;
  Score gap_open_score = -1;
  Score gap_extend_score = -1;
  double prefix_weight = 0.1;
  double prefix_threshold = 0.7;
  std::size_t prefix_cap = 4;
};

inline std::string canonical_scorer_name(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (const char character : value) {
    if (character == '-') {
      result.push_back('_');
    } else if (character >= 'A' && character <= 'Z') {
      result.push_back(static_cast<char>(character - 'A' + 'a'));
    } else {
      result.push_back(character);
    }
  }
  if (result.starts_with("stride_")) result.erase(0, 7);
  return result;
}

inline Scorer parse_scorer(std::string_view value) {
  const std::string name = canonical_scorer_name(value);
  if (name == "levenshtein" || name == "levenshtein_score" || name == "0") {
    return Scorer::levenshtein;
  }
  if (name == "levenshtein_normalized" ||
      name == "levenshtein_normalized_score" || name == "1") {
    return Scorer::levenshtein_normalized;
  }
  if (name == "damerau_levenshtein" || name == "damerau_levenshtein_score" ||
      name == "osa" || name == "2") {
    return Scorer::damerau_levenshtein;
  }
  if (name == "damerau_levenshtein_normalized" ||
      name == "damerau_levenshtein_normalized_score" ||
      name == "osa_normalized" || name == "3") {
    return Scorer::damerau_levenshtein_normalized;
  }
  if (name == "hamming" || name == "hamming_score" || name == "4") {
    return Scorer::hamming;
  }
  if (name == "hamming_normalized" || name == "hamming_normalized_score" ||
      name == "5") {
    return Scorer::hamming_normalized;
  }
  if (name == "jaro" || name == "jaro_similarity" || name == "6") {
    return Scorer::jaro;
  }
  if (name == "jaro_winkler" || name == "jaro_winkler_similarity" ||
      name == "7") {
    return Scorer::jaro_winkler;
  }
  if (name == "indel" || name == "indel_score" || name == "8") {
    return Scorer::indel;
  }
  if (name == "indel_normalized" || name == "indel_normalized_score" ||
      name == "9") {
    return Scorer::indel_normalized;
  }
  if (name == "true_damerau_levenshtein" ||
      name == "true_damerau_levenshtein_score" || name == "10") {
    return Scorer::true_damerau_levenshtein;
  }
  if (name == "true_damerau_levenshtein_normalized" ||
      name == "true_damerau_levenshtein_normalized_score" || name == "11") {
    return Scorer::true_damerau_levenshtein_normalized;
  }
  if (name == "smith_waterman" || name == "smith_waterman_score" ||
      name == "sw" || name == "local" || name == "12") {
    return Scorer::smith_waterman;
  }
  if (name == "smith_waterman_normalized" ||
      name == "smith_waterman_normalized_score" || name == "13") {
    return Scorer::smith_waterman_normalized;
  }
  if (name == "needleman_wunsch" || name == "needleman_wunsch_score" ||
      name == "nw" || name == "global" || name == "14") {
    return Scorer::needleman_wunsch;
  }
  if (name == "needleman_wunsch_normalized" ||
      name == "needleman_wunsch_normalized_score" || name == "15") {
    return Scorer::needleman_wunsch_normalized;
  }
  throw std::invalid_argument("unknown stride-align scorer: " + std::string(value));
}

inline bool is_normalized_or_similarity(Scorer scorer) noexcept {
  switch (scorer) {
    case Scorer::levenshtein_normalized:
    case Scorer::damerau_levenshtein_normalized:
    case Scorer::hamming_normalized:
    case Scorer::jaro:
    case Scorer::jaro_winkler:
    case Scorer::indel_normalized:
    case Scorer::true_damerau_levenshtein_normalized:
    case Scorer::smith_waterman_normalized:
    case Scorer::needleman_wunsch_normalized:
      return true;
    default:
      return false;
  }
}

inline bool higher_is_better(Scorer scorer) noexcept {
  switch (scorer) {
    case Scorer::levenshtein:
    case Scorer::damerau_levenshtein:
    case Scorer::hamming:
    case Scorer::indel:
    case Scorer::true_damerau_levenshtein:
      return false;
    default:
      return true;
  }
}

inline bool is_integral(Scorer scorer) noexcept {
  switch (scorer) {
    case Scorer::levenshtein:
    case Scorer::damerau_levenshtein:
    case Scorer::hamming:
    case Scorer::indel:
    case Scorer::true_damerau_levenshtein:
    case Scorer::smith_waterman:
    case Scorer::needleman_wunsch:
      return true;
    default:
      return false;
  }
}

inline bool is_hamming(Scorer scorer) noexcept {
  return scorer == Scorer::hamming || scorer == Scorer::hamming_normalized;
}

inline double normalized_alignment(
    Score raw,
    std::size_t query_size,
    std::size_t target_size,
    bool local,
    Score match_score) {
  if (match_score <= 0) {
    throw std::invalid_argument(
        "match_score must be positive for normalized alignment scores");
  }
  if (query_size == 0U && target_size == 0U) return 1.0;
  const std::size_t length = local
      ? std::min(query_size, target_size)
      : std::max(query_size, target_size);
  if (length == 0U) return 0.0;
  const double denominator =
      static_cast<double>(length) * static_cast<double>(match_score);
  return std::clamp(static_cast<double>(raw) / denominator, 0.0, 1.0);
}

inline std::size_t unicode_length(std::string_view text) {
  if (utf8::is_ascii(text)) return text.size();
  return utf8::prepare_streaming(text).size();
}

// Convert a normalized-similarity floor into an inclusive integer distance
// cutoff. The epsilon protects exact integer boundaries such as
// (1 - 0.8) * 10 from binary floating-point rounding to 1.999999....
inline std::size_t normalized_distance_cutoff(
    double similarity,
    std::size_t maximum_distance) noexcept {
  if (similarity <= 0.0) return maximum_distance;
  if (similarity >= 1.0) return 0U;
  const double allowed =
      (1.0 - similarity) * static_cast<double>(maximum_distance);
  return static_cast<std::size_t>(std::floor(allowed + 1e-9));
}

struct Text {
  std::string bytes;
  std::size_t length = 0;

  Text() = default;
  explicit Text(std::string value)
      : bytes(std::move(value)), length(unicode_length(bytes)) {}
  Text(std::string value, std::size_t character_length)
      : bytes(std::move(value)), length(character_length) {}
};

inline double score_prepared(
    Scorer scorer,
    const utf8::PreparedPair& pair,
    const ScoreOptions& options = {},
    std::optional<double> similarity_cutoff = std::nullopt,
    std::optional<std::size_t> distance_cutoff = std::nullopt) {
  switch (scorer) {
    case Scorer::levenshtein:
      return static_cast<double>(core::levenshtein_distance(
          pair, distance_cutoff.value_or(levenshtein::kNoCutoff)));
    case Scorer::levenshtein_normalized: {
      std::size_t cutoff = levenshtein::kNoCutoff;
      if (similarity_cutoff.has_value()) {
        cutoff = normalized_distance_cutoff(
            *similarity_cutoff,
            std::max(pair.query_size(), pair.target_size()));
      }
      return levenshtein::normalize(
          core::levenshtein_distance(pair, cutoff),
          pair.query_size(), pair.target_size());
    }
    case Scorer::damerau_levenshtein:
      return static_cast<double>(core::osa_distance(pair));
    case Scorer::damerau_levenshtein_normalized:
      return core::osa_similarity(pair);
    case Scorer::hamming:
      return static_cast<double>(core::hamming_distance(pair));
    case Scorer::hamming_normalized:
      return core::hamming_similarity(pair);
    case Scorer::jaro:
      return core::jaro_similarity(pair);
    case Scorer::jaro_winkler:
      return core::jaro_winkler_similarity(
          pair, options.prefix_weight, options.prefix_threshold,
          options.prefix_cap);
    case Scorer::indel:
      return static_cast<double>(core::indel_distance(
          pair, distance_cutoff.value_or(indel::kNoCutoff)));
    case Scorer::indel_normalized: {
      std::size_t cutoff = indel::kNoCutoff;
      if (similarity_cutoff.has_value()) {
        cutoff = normalized_distance_cutoff(
            *similarity_cutoff, pair.query_size() + pair.target_size());
      }
      return indel::normalize(
          core::indel_distance(pair, cutoff),
          pair.query_size(), pair.target_size());
    }
    case Scorer::true_damerau_levenshtein:
      return static_cast<double>(core::true_damerau_levenshtein_distance(pair));
    case Scorer::true_damerau_levenshtein_normalized:
      return core::true_damerau_levenshtein_similarity(pair);
    case Scorer::smith_waterman: {
      const Score raw = options.gap_open_score == options.gap_extend_score
          ? core::smith_waterman_score(
                pair, options.match_score, options.mismatch_score,
                options.gap_open_score)
          : core::smith_waterman_affine_score(
                pair, options.match_score, options.mismatch_score,
                options.gap_open_score, options.gap_extend_score);
      return static_cast<double>(raw);
    }
    case Scorer::smith_waterman_normalized: {
      const Score raw = options.gap_open_score == options.gap_extend_score
          ? core::smith_waterman_score(
                pair, options.match_score, options.mismatch_score,
                options.gap_open_score)
          : core::smith_waterman_affine_score(
                pair, options.match_score, options.mismatch_score,
                options.gap_open_score, options.gap_extend_score);
      return normalized_alignment(
          raw, pair.query_size(), pair.target_size(), true,
          options.match_score);
    }
    case Scorer::needleman_wunsch: {
      const Score raw = options.gap_open_score == options.gap_extend_score
          ? core::needleman_wunsch_score(
                pair, options.match_score, options.mismatch_score,
                options.gap_open_score)
          : core::needleman_wunsch_affine_score(
                pair, options.match_score, options.mismatch_score,
                options.gap_open_score, options.gap_extend_score);
      return static_cast<double>(raw);
    }
    case Scorer::needleman_wunsch_normalized: {
      const Score raw = options.gap_open_score == options.gap_extend_score
          ? core::needleman_wunsch_score(
                pair, options.match_score, options.mismatch_score,
                options.gap_open_score)
          : core::needleman_wunsch_affine_score(
                pair, options.match_score, options.mismatch_score,
                options.gap_open_score, options.gap_extend_score);
      return normalized_alignment(
          raw, pair.query_size(), pair.target_size(), false,
          options.match_score);
    }
  }
  throw std::logic_error("unsupported stride-align scorer");
}

template <typename PreparePair>
inline double score_with(
    Scorer scorer,
    std::string_view query,
    std::string_view target,
    PreparePair&& prepare_pair,
    const ScoreOptions& options = {},
    std::optional<double> similarity_cutoff = std::nullopt,
    std::optional<std::size_t> distance_cutoff = std::nullopt) {
  const auto pair = prepare_pair(query, target);
  return score_prepared(
      scorer, pair, options, similarity_cutoff, distance_cutoff);
}

inline double score(
    Scorer scorer,
    std::string_view query,
    std::string_view target,
    const ScoreOptions& options = {},
    std::optional<double> similarity_cutoff = std::nullopt,
    std::optional<std::size_t> distance_cutoff = std::nullopt) {
  return score_with(
      scorer, query, target,
      [](std::string_view left, std::string_view right) {
        return utf8::prepare_pair(left, right);
      },
      options, similarity_cutoff, distance_cutoff);
}

struct Utf8PreparePair {
  utf8::PreparedPair operator()(
      std::string_view query,
      std::string_view target) const {
    return utf8::prepare_pair(query, target);
  }
};

// Cheap upper bound used before invoking a normalized scorer. Length-only
// bounds let threshold and top-k operations avoid preparing or analysing a
// pair that cannot possibly enter the result set. These are valid upper
// bounds; some are deliberately loose when scorer parameters are unusual.
inline double maximum_similarity(
    Scorer scorer,
    std::size_t query_size,
    std::size_t target_size,
    const ScoreOptions& options = {}) noexcept {
  const std::size_t longer = std::max(query_size, target_size);
  const std::size_t shorter = std::min(query_size, target_size);
  if (longer == 0U) return 1.0;
  switch (scorer) {
    case Scorer::levenshtein_normalized:
    case Scorer::damerau_levenshtein_normalized:
    case Scorer::true_damerau_levenshtein_normalized:
      return static_cast<double>(shorter) / static_cast<double>(longer);
    case Scorer::indel_normalized: {
      const std::size_t total = query_size + target_size;
      return total == 0U
          ? 1.0
          : (2.0 * static_cast<double>(shorter)) /
                static_cast<double>(total);
    }
    case Scorer::hamming_normalized:
      return query_size == target_size ? 1.0 : 0.0;
    case Scorer::jaro:
      if (shorter == 0U) return 0.0;
      return (2.0 + static_cast<double>(shorter) /
                        static_cast<double>(longer)) /
          3.0;
    case Scorer::jaro_winkler: {
      const double jaro_bound = maximum_similarity(
          Scorer::jaro, query_size, target_size, options);
      if (jaro_bound < options.prefix_threshold) return jaro_bound;
      const double boost = static_cast<double>(
          std::min(options.prefix_cap, shorter)) * options.prefix_weight;
      if (boost <= 1.0) {
        return jaro_bound + boost * (1.0 - jaro_bound);
      }
      // Unusually large weights can make the Winkler transform non-monotone;
      // keep the bound conservative instead of risking an invalid prune.
      return std::numeric_limits<double>::infinity();
    }
    default:
      return 1.0;
  }
}

inline std::size_t minimum_distance(
    Scorer scorer,
    std::size_t query_size,
    std::size_t target_size) noexcept {
  const std::size_t difference = query_size > target_size
      ? query_size - target_size
      : target_size - query_size;
  switch (scorer) {
    case Scorer::levenshtein:
    case Scorer::damerau_levenshtein:
    case Scorer::indel:
    case Scorer::true_damerau_levenshtein:
      return difference;
    case Scorer::hamming:
      return query_size == target_size
          ? 0U : std::numeric_limits<std::size_t>::max();
    default:
      return 0U;
  }
}

struct RankedMatch {
  double score = 0.0;
  std::size_t index = 0;
};

inline bool better_than(
    const RankedMatch& left,
    const RankedMatch& right,
    bool higher) noexcept {
  if (left.score != right.score) {
    return higher ? left.score > right.score : left.score < right.score;
  }
  return left.index < right.index;
}

inline void insert_ranked(
    std::vector<RankedMatch>& output,
    RankedMatch candidate,
    std::size_t k,
    bool higher) {
  if (k == 0U) return;
  const auto better = [higher](
                          const RankedMatch& left,
                          const RankedMatch& right) {
    return better_than(left, right, higher);
  };
  if (output.size() < k) {
    output.push_back(candidate);
    std::push_heap(output.begin(), output.end(), better);
    return;
  }
  // `better` makes the heap root the worst retained match.
  if (!better(candidate, output.front())) return;
  std::pop_heap(output.begin(), output.end(), better);
  output.back() = candidate;
  std::push_heap(output.begin(), output.end(), better);
}

template <typename PreparePair>
inline std::vector<RankedMatch> top_k_with(
    const Text& query,
    std::span<const std::optional<Text>> targets,
    Scorer scorer,
    std::size_t k,
    PreparePair&& prepare_pair,
    const ScoreOptions& options = {},
    bool skip_invalid_hamming = false) {
  std::vector<RankedMatch> output;
  if (k == 0U) return output;
  output.reserve(std::min(k, targets.size()));
  const bool higher = higher_is_better(scorer);
  for (std::size_t index = 0; index < targets.size(); ++index) {
    if (!targets[index].has_value()) continue;
    const Text& target = *targets[index];
    if (is_hamming(scorer) && query.length != target.length) {
      if (skip_invalid_hamming) continue;
      throw std::invalid_argument("Hamming distance requires equal-length strings");
    }
    std::optional<double> cutoff;
    std::optional<std::size_t> distance_cutoff;
    if (higher && output.size() == k && k != 0U) {
      cutoff = output.front().score;
      if (is_normalized_or_similarity(scorer) &&
          maximum_similarity(
              scorer, query.length, target.length, options) < *cutoff) {
        continue;
      }
    } else if (!higher && output.size() == k && k != 0U) {
      const double worst = output.front().score;
      if (static_cast<double>(minimum_distance(
              scorer, query.length, target.length)) > worst) {
        continue;
      }
      if ((scorer == Scorer::levenshtein || scorer == Scorer::indel) &&
          worst >= 0.0 &&
          worst < static_cast<double>(
              std::numeric_limits<std::size_t>::max())) {
        distance_cutoff = static_cast<std::size_t>(worst);
      }
    }
    insert_ranked(
        output,
        {score_with(
             scorer, query.bytes, target.bytes, prepare_pair, options,
             cutoff, distance_cutoff),
         index},
        k, higher);
  }
  std::sort(
      output.begin(), output.end(),
      [higher](const RankedMatch& left, const RankedMatch& right) {
        return better_than(left, right, higher);
      });
  return output;
}

inline std::vector<RankedMatch> top_k(
    const Text& query,
    std::span<const std::optional<Text>> targets,
    Scorer scorer,
    std::size_t k,
    const ScoreOptions& options = {},
    bool skip_invalid_hamming = false) {
  return top_k_with(
      query, targets, scorer, k, Utf8PreparePair{}, options,
      skip_invalid_hamming);
}

template <typename PreparePair>
inline std::vector<std::optional<double>> scores_with(
    const Text& query,
    std::span<const std::optional<Text>> targets,
    Scorer scorer,
    PreparePair&& prepare_pair,
    const ScoreOptions& options = {}) {
  std::vector<std::optional<double>> output;
  output.reserve(targets.size());
  for (const auto& target : targets) {
    if (!target.has_value()) {
      output.emplace_back(std::nullopt);
    } else {
      output.emplace_back(score_with(
          scorer, query.bytes, target->bytes, prepare_pair, options));
    }
  }
  return output;
}

inline std::vector<std::optional<double>> scores(
    const Text& query,
    std::span<const std::optional<Text>> targets,
    Scorer scorer,
    const ScoreOptions& options = {}) {
  return scores_with(query, targets, scorer, Utf8PreparePair{}, options);
}

inline bool same_texts(
    std::span<const std::optional<Text>> left,
    std::span<const std::optional<Text>> right) noexcept {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (left[index].has_value() != right[index].has_value()) return false;
    if (left[index].has_value() &&
        left[index]->bytes != right[index]->bytes) return false;
  }
  return true;
}

using DistanceMatrix = std::vector<std::vector<std::optional<double>>>;

template <typename PreparePair>
inline DistanceMatrix cdist_with(
    std::span<const std::optional<Text>> queries,
    std::span<const std::optional<Text>> targets,
    Scorer scorer,
    PreparePair&& prepare_pair,
    const ScoreOptions& options = {}) {
  DistanceMatrix output(
      queries.size(),
      std::vector<std::optional<double>>(targets.size(), std::nullopt));
  const bool symmetric = same_texts(queries, targets);
  for (std::size_t query_index = 0; query_index < queries.size(); ++query_index) {
    if (!queries[query_index].has_value()) continue;
    const std::size_t target_begin = symmetric ? query_index : 0U;
    for (std::size_t target_index = target_begin;
         target_index < targets.size(); ++target_index) {
      if (!targets[target_index].has_value()) continue;
      const double value = score_with(
          scorer, queries[query_index]->bytes,
          targets[target_index]->bytes, prepare_pair, options);
      output[query_index][target_index] = value;
      if (symmetric && query_index != target_index) {
        output[target_index][query_index] = value;
      }
    }
  }
  return output;
}

inline DistanceMatrix cdist(
    std::span<const std::optional<Text>> queries,
    std::span<const std::optional<Text>> targets,
    Scorer scorer,
    const ScoreOptions& options = {}) {
  return cdist_with(queries, targets, scorer, Utf8PreparePair{}, options);
}

struct MatrixMatch {
  double score = 0.0;
  std::size_t query_index = 0;
  std::size_t target_index = 0;
};

inline bool better_than(
    const MatrixMatch& left,
    const MatrixMatch& right) noexcept {
  if (left.score != right.score) return left.score > right.score;
  if (left.query_index != right.query_index) {
    return left.query_index < right.query_index;
  }
  return left.target_index < right.target_index;
}

inline void insert_ranked(
    std::vector<MatrixMatch>& output,
    MatrixMatch candidate,
    std::size_t k) {
  if (k == 0U) return;
  const auto better = [](const MatrixMatch& left, const MatrixMatch& right) {
    return better_than(left, right);
  };
  if (output.size() < k) {
    output.push_back(candidate);
    std::push_heap(output.begin(), output.end(), better);
    return;
  }
  if (!better_than(candidate, output.front())) return;
  std::pop_heap(output.begin(), output.end(), better);
  output.back() = candidate;
  std::push_heap(output.begin(), output.end(), better);
}

template <typename PreparePair>
inline std::vector<MatrixMatch> cdist_above_threshold_with(
    std::span<const std::optional<Text>> queries,
    std::span<const std::optional<Text>> targets,
    Scorer scorer,
    double threshold,
    PreparePair&& prepare_pair,
    const ScoreOptions& options = {}) {
  if (!is_normalized_or_similarity(scorer)) {
    throw std::invalid_argument(
        "cdist_above_threshold requires a normalized or similarity scorer");
  }
  if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
    throw std::invalid_argument("threshold must be between 0 and 1");
  }
  std::vector<MatrixMatch> output;
  for (std::size_t query_index = 0; query_index < queries.size(); ++query_index) {
    if (!queries[query_index].has_value()) continue;
    for (std::size_t target_index = 0; target_index < targets.size(); ++target_index) {
      if (!targets[target_index].has_value()) continue;
      const Text& query = *queries[query_index];
      const Text& target = *targets[target_index];
      if (is_hamming(scorer) && query.length != target.length) {
        throw std::invalid_argument("Hamming distance requires equal-length strings");
      }
      if (maximum_similarity(
              scorer, query.length, target.length, options) < threshold) {
        continue;
      }
      const double value = score_with(
          scorer, query.bytes, target.bytes, prepare_pair, options, threshold);
      if (value >= threshold) {
        output.push_back({value, query_index, target_index});
      }
    }
  }
  return output;
}

inline std::vector<MatrixMatch> cdist_above_threshold(
    std::span<const std::optional<Text>> queries,
    std::span<const std::optional<Text>> targets,
    Scorer scorer,
    double threshold,
    const ScoreOptions& options = {}) {
  return cdist_above_threshold_with(
      queries, targets, scorer, threshold, Utf8PreparePair{}, options);
}

template <typename PreparePair>
inline std::vector<MatrixMatch> cdist_top_k_with(
    std::span<const std::optional<Text>> queries,
    std::span<const std::optional<Text>> targets,
    Scorer scorer,
    std::size_t k,
    bool reject_duplicates,
    PreparePair&& prepare_pair,
    const ScoreOptions& options = {}) {
  if (!is_normalized_or_similarity(scorer)) {
    throw std::invalid_argument(
        "cdist_top_k requires a normalized or similarity scorer");
  }
  std::vector<MatrixMatch> output;
  if (k == 0U) return output;
  const std::size_t maximum_pairs = targets.empty() ||
          queries.size() <= std::numeric_limits<std::size_t>::max() / targets.size()
      ? queries.size() * targets.size()
      : std::numeric_limits<std::size_t>::max();
  output.reserve(std::min(k, maximum_pairs));
  for (std::size_t query_index = 0; query_index < queries.size(); ++query_index) {
    if (!queries[query_index].has_value()) continue;
    for (std::size_t target_index = 0; target_index < targets.size(); ++target_index) {
      if (!targets[target_index].has_value()) continue;
      const Text& query = *queries[query_index];
      const Text& target = *targets[target_index];
      if (reject_duplicates && query.bytes == target.bytes) continue;
      if (is_hamming(scorer) && query.length != target.length) {
        throw std::invalid_argument("Hamming distance requires equal-length strings");
      }
      std::optional<double> cutoff;
      if (output.size() == k && k != 0U) {
        cutoff = output.front().score;
        if (maximum_similarity(
                scorer, query.length, target.length, options) < *cutoff) {
          continue;
        }
      }
      insert_ranked(
          output,
          {score_with(
               scorer, query.bytes, target.bytes, prepare_pair, options,
               cutoff),
           query_index, target_index},
          k);
    }
  }
  std::sort(
      output.begin(), output.end(),
      [](const MatrixMatch& left, const MatrixMatch& right) {
        return better_than(left, right);
      });
  return output;
}

inline std::vector<MatrixMatch> cdist_top_k(
    std::span<const std::optional<Text>> queries,
    std::span<const std::optional<Text>> targets,
    Scorer scorer,
    std::size_t k,
    bool reject_duplicates,
    const ScoreOptions& options = {}) {
  return cdist_top_k_with(
      queries, targets, scorer, k, reject_duplicates,
      Utf8PreparePair{}, options);
}

template <typename PreparePair>
inline std::vector<std::vector<RankedMatch>> cdist_top_k_per_query_with(
    std::span<const std::optional<Text>> queries,
    std::span<const std::optional<Text>> targets,
    Scorer scorer,
    std::size_t k,
    PreparePair&& prepare_pair,
    const ScoreOptions& options = {}) {
  if (!is_normalized_or_similarity(scorer)) {
    throw std::invalid_argument(
        "cdist_top_k_per_query requires a normalized or similarity scorer");
  }
  std::vector<std::vector<RankedMatch>> output;
  output.reserve(queries.size());
  for (const auto& query : queries) {
    if (!query.has_value()) {
      output.emplace_back();
      continue;
    }
    output.push_back(top_k_with(
        *query, targets, scorer, k, prepare_pair, options,
        /*skip_invalid_hamming=*/true));
  }
  return output;
}

inline std::vector<std::vector<RankedMatch>> cdist_top_k_per_query(
    std::span<const std::optional<Text>> queries,
    std::span<const std::optional<Text>> targets,
    Scorer scorer,
    std::size_t k,
    const ScoreOptions& options = {}) {
  return cdist_top_k_per_query_with(
      queries, targets, scorer, k, Utf8PreparePair{}, options);
}

}  // namespace stride_align::batch
