#pragma once

#ifndef STRIDE_ALIGN_R_DLL
#error "STRIDE_ALIGN_R_DLL must name this R shared library"
#endif

#define R_NO_REMAP
#include <R.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>
#include <Rinternals.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "target_profile.hpp"
#include "stride_align/core.hpp"

#if defined(STRIDE_ALIGN_R_PRIMARY)
#include "cpu_detect.hpp"
#include "stride_align/alignment.hpp"
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
#endif

namespace {

class Protection final {
 public:
  Protection() = default;
  Protection(const Protection&) = delete;
  Protection& operator=(const Protection&) = delete;

  ~Protection() {
    if (count_ != 0) UNPROTECT(count_);
  }

  SEXP add(SEXP value) {
    PROTECT(value);
    ++count_;
    return value;
  }

 private:
  int count_ = 0;
};

class Utf8Input final {
 public:
  static Utf8Input from_charsxp(SEXP value) {
    if (TYPEOF(value) != CHARSXP) {
      throw std::invalid_argument("character vector contains a non-string element");
    }

    const auto length = static_cast<std::size_t>(Rf_length(value));
    const std::string_view bytes(CHAR(value), length);

    // Clean 7-bit strings can be consumed directly from the immutable CHARSXP.
    const bool ascii = ::stride_align::utf8::is_ascii(bytes);
    if (ascii || Rf_getCharCE(value) == CE_UTF8) {
      return Utf8Input(bytes.data(), bytes.size(), ascii);
    }

    if (Rf_getCharCE(value) == CE_BYTES) {
      throw std::invalid_argument(
          "non-ASCII strings marked as bytes do not have a Unicode encoding");
    }

    // R vectors can also contain native- or Latin-1-marked elements. Translate
    // those to UTF-8 and copy immediately: R owns the translation buffer.
    return Utf8Input(std::string(Rf_translateCharUTF8(value)));
  }

  Utf8Input(Utf8Input&&) noexcept = default;
  Utf8Input& operator=(Utf8Input&&) noexcept = default;

  std::string_view view() const noexcept {
    if (owns_bytes_) return owned_;
    return std::string_view(borrowed_, size_);
  }

  bool is_ascii() const noexcept { return ascii_; }

 private:
  Utf8Input(const char* bytes, std::size_t size, bool ascii) noexcept
      : borrowed_(bytes), size_(size), owns_bytes_(false), ascii_(ascii) {}

  explicit Utf8Input(std::string bytes)
      : owned_(std::move(bytes)), owns_bytes_(true), ascii_(false) {}

  const char* borrowed_ = nullptr;
  std::size_t size_ = 0;
  std::string owned_;
  bool owns_bytes_ = false;
  bool ascii_ = false;
};

#if defined(STRIDE_ALIGN_R_PRIMARY)
std::vector<std::uint32_t> codepoints(const Utf8Input& input) {
  return ::stride_align::utf8::prepare_streaming(input.view());
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

std::string utf8_string(SEXP charsxp) {
  return std::string(Utf8Input::from_charsxp(charsxp).view());
}

SEXP utf8_charsxp(const std::string& value) {
  return Rf_mkCharLenCE(
      value.data(), static_cast<int>(value.size()), CE_UTF8);
}

template <typename Function>
SEXP apply_string_unary(SEXP input, Function&& function) {
  if (TYPEOF(input) != STRSXP) {
    throw std::invalid_argument("input must be a character vector");
  }
  Protection protection;
  SEXP result = protection.add(Rf_allocVector(STRSXP, XLENGTH(input)));
  for (R_xlen_t index = 0; index < XLENGTH(input); ++index) {
    SEXP chars = STRING_ELT(input, index);
    if (chars == NA_STRING) {
      SET_STRING_ELT(result, index, NA_STRING);
      continue;
    }
    SET_STRING_ELT(result, index, utf8_charsxp(function(chars)));
  }
  return result;
}
#endif

::stride_align::utf8::PreparedPair prepare_pair(
    const Utf8Input& query,
    const Utf8Input& target) {
  if (!query.is_ascii() || !target.is_ascii()) {
    return ::stride_align::utf8::prepare_pair(query.view(), target.view());
  }

  const auto as_bytes = [](std::string_view value) {
    return std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
  };
  return {
      ::stride_align::utf8::TokenWidth::u8,
      true,
      false,
      as_bytes(query.view()),
      as_bytes(target.view()),
  };
}

struct VectorShape {
  R_xlen_t query_size;
  R_xlen_t target_size;
  R_xlen_t output_size;
};

VectorShape vector_shape(SEXP query, SEXP target) {
  if (TYPEOF(query) != STRSXP || TYPEOF(target) != STRSXP) {
    throw std::invalid_argument("query and target must be character vectors");
  }

  const R_xlen_t query_size = XLENGTH(query);
  const R_xlen_t target_size = XLENGTH(target);
  if (query_size == target_size) {
    return {query_size, target_size, query_size};
  }
  if (query_size == 1) {
    return {query_size, target_size, target_size};
  }
  if (target_size == 1) {
    return {query_size, target_size, query_size};
  }
  throw std::invalid_argument(
      "query and target must have equal lengths, or one must have length one");
}

#if defined(STRIDE_ALIGN_R_PRIMARY)
template <typename Function>
SEXP apply_codepoint_pairwise(SEXP query, SEXP target, Function&& function) {
  const VectorShape shape = vector_shape(query, target);
  Protection protection;
  SEXP result = protection.add(Rf_allocVector(REALSXP, shape.output_size));
  double* output = REAL(result);
  for (R_xlen_t index = 0; index < shape.output_size; ++index) {
    const R_xlen_t query_index = shape.query_size == 1 ? 0 : index;
    const R_xlen_t target_index = shape.target_size == 1 ? 0 : index;
    SEXP query_chars = STRING_ELT(query, query_index);
    SEXP target_chars = STRING_ELT(target, target_index);
    if (query_chars == NA_STRING || target_chars == NA_STRING) {
      output[index] = NA_REAL;
      continue;
    }
    const auto query_input = Utf8Input::from_charsxp(query_chars);
    const auto target_input = Utf8Input::from_charsxp(target_chars);
    output[index] = static_cast<double>(function(
        codepoints(query_input), codepoints(target_input)));
  }
  return result;
}

template <typename Function>
SEXP apply_logical_pairwise(SEXP query, SEXP target, Function&& function) {
  const VectorShape shape = vector_shape(query, target);
  Protection protection;
  SEXP result = protection.add(Rf_allocVector(LGLSXP, shape.output_size));
  for (R_xlen_t index = 0; index < shape.output_size; ++index) {
    const R_xlen_t query_index = shape.query_size == 1 ? 0 : index;
    const R_xlen_t target_index = shape.target_size == 1 ? 0 : index;
    SEXP query_chars = STRING_ELT(query, query_index);
    SEXP target_chars = STRING_ELT(target, target_index);
    if (query_chars == NA_STRING || target_chars == NA_STRING) {
      LOGICAL(result)[index] = NA_LOGICAL;
      continue;
    }
    LOGICAL(result)[index] = function(query_chars, target_chars) ? TRUE : FALSE;
  }
  return result;
}
#endif

double numeric_scalar(SEXP value, const char* name) {
  if (XLENGTH(value) != 1 ||
      (TYPEOF(value) != REALSXP && TYPEOF(value) != INTSXP)) {
    throw std::invalid_argument(std::string(name) + " must be one numeric value");
  }

  const double result = TYPEOF(value) == REALSXP
      ? REAL(value)[0]
      : static_cast<double>(INTEGER(value)[0]);
  if (!std::isfinite(result) ||
      (TYPEOF(value) == INTSXP && INTEGER(value)[0] == NA_INTEGER)) {
    throw std::invalid_argument(std::string(name) + " must be finite and non-missing");
  }
  return result;
}

::stride_align::Score score_scalar(SEXP value, const char* name) {
  const double result = numeric_scalar(value, name);
  constexpr double kLargestExactInteger = 9007199254740991.0;
  if (std::trunc(result) != result || std::abs(result) > kLargestExactInteger) {
    throw std::invalid_argument(
        std::string(name) + " must be an exactly representable integer");
  }
  return static_cast<::stride_align::Score>(result);
}

std::size_t size_scalar(SEXP value, const char* name) {
  const double result = numeric_scalar(value, name);
  constexpr double kLargestExactInteger = 9007199254740991.0;
  if (std::trunc(result) != result || result < 0.0 ||
      result > kLargestExactInteger) {
    throw std::invalid_argument(
        std::string(name) + " must be a non-negative integer");
  }
  return static_cast<std::size_t>(result);
}

template <typename Scorer>
SEXP apply_pairwise(SEXP query, SEXP target, Scorer&& scorer) {
  const VectorShape shape = vector_shape(query, target);
  Protection protection;
  SEXP result = protection.add(Rf_allocVector(REALSXP, shape.output_size));
  double* output = REAL(result);

  for (R_xlen_t index = 0; index < shape.output_size; ++index) {
    const R_xlen_t query_index = shape.query_size == 1 ? 0 : index;
    const R_xlen_t target_index = shape.target_size == 1 ? 0 : index;
    SEXP query_chars = STRING_ELT(query, query_index);
    SEXP target_chars = STRING_ELT(target, target_index);
    if (query_chars == NA_STRING || target_chars == NA_STRING) {
      output[index] = NA_REAL;
      continue;
    }

    try {
      const Utf8Input query_utf8 = Utf8Input::from_charsxp(query_chars);
      const Utf8Input target_utf8 = Utf8Input::from_charsxp(target_chars);

      // PreparedPair borrows only all-ASCII bytes. Every non-ASCII u8/u16/u32
      // representation is owned by the pair and dies at the end of this call.
      const auto pair = prepare_pair(query_utf8, target_utf8);
      output[index] = static_cast<double>(scorer(pair));
    } catch (const std::exception& error) {
      throw std::runtime_error(
          "element " + std::to_string(static_cast<long long>(index + 1)) +
          ": " + error.what());
    }
  }

  return result;
}

template <typename Function>
SEXP call_safely(Function&& function) {
  std::array<char, 1024> message{};
  try {
    return function();
  } catch (const std::exception& error) {
    std::snprintf(message.data(), message.size(), "%s", error.what());
  } catch (...) {
    std::snprintf(message.data(), message.size(), "%s", "unknown C++ exception");
  }

  // All C++ operation-local buffers have unwound before R performs its jump.
  Rf_error("stride-align: %s", message.data());
  return R_NilValue;
}

SEXP stride_r_levenshtein(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_pairwise(query, target, [](const auto& pair) {
      return ::stride_align::core::levenshtein_distance(pair);
    });
  });
}

SEXP stride_r_levenshtein_similarity(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_pairwise(query, target, [](const auto& pair) {
      return ::stride_align::core::levenshtein_similarity(pair);
    });
  });
}

SEXP stride_r_osa(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_pairwise(query, target, [](const auto& pair) {
      return ::stride_align::core::osa_distance(pair);
    });
  });
}

SEXP stride_r_osa_similarity(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_pairwise(query, target, [](const auto& pair) {
      return ::stride_align::core::osa_similarity(pair);
    });
  });
}

SEXP stride_r_true_damerau_levenshtein(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_pairwise(query, target, [](const auto& pair) {
      return ::stride_align::core::true_damerau_levenshtein_distance(pair);
    });
  });
}

SEXP stride_r_true_damerau_levenshtein_similarity(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_pairwise(query, target, [](const auto& pair) {
      return ::stride_align::core::true_damerau_levenshtein_similarity(pair);
    });
  });
}

SEXP stride_r_indel(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_pairwise(query, target, [](const auto& pair) {
      return ::stride_align::core::indel_distance(pair);
    });
  });
}

SEXP stride_r_indel_similarity(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_pairwise(query, target, [](const auto& pair) {
      return ::stride_align::core::indel_similarity(pair);
    });
  });
}

SEXP stride_r_hamming(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_pairwise(query, target, [](const auto& pair) {
      return ::stride_align::core::hamming_distance(pair);
    });
  });
}

SEXP stride_r_hamming_similarity(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_pairwise(query, target, [](const auto& pair) {
      return ::stride_align::core::hamming_similarity(pair);
    });
  });
}

SEXP stride_r_jaro(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_pairwise(query, target, [](const auto& pair) {
      return ::stride_align::core::jaro_similarity(pair);
    });
  });
}

SEXP stride_r_jaro_winkler(
    SEXP query,
    SEXP target,
    SEXP prefix_weight,
    SEXP prefix_threshold,
    SEXP prefix_cap) {
  return call_safely([&] {
    const double weight = numeric_scalar(prefix_weight, "prefix_weight");
    const double threshold = numeric_scalar(prefix_threshold, "prefix_threshold");
    const std::size_t cap = size_scalar(prefix_cap, "prefix_cap");
    return apply_pairwise(query, target, [&](const auto& pair) {
      return ::stride_align::core::jaro_winkler_similarity(
          pair, weight, threshold, cap);
    });
  });
}

SEXP stride_r_smith_waterman(
    SEXP query,
    SEXP target,
    SEXP match_score,
    SEXP mismatch_score,
    SEXP gap_score) {
  return call_safely([&] {
    const auto match = score_scalar(match_score, "match_score");
    const auto mismatch = score_scalar(mismatch_score, "mismatch_score");
    const auto gap = score_scalar(gap_score, "gap_score");
    return apply_pairwise(query, target, [&](const auto& pair) {
      return ::stride_align::core::smith_waterman_score(
          pair, match, mismatch, gap);
    });
  });
}

SEXP stride_r_needleman_wunsch(
    SEXP query,
    SEXP target,
    SEXP match_score,
    SEXP mismatch_score,
    SEXP gap_score) {
  return call_safely([&] {
    const auto match = score_scalar(match_score, "match_score");
    const auto mismatch = score_scalar(mismatch_score, "mismatch_score");
    const auto gap = score_scalar(gap_score, "gap_score");
    return apply_pairwise(query, target, [&](const auto& pair) {
      return ::stride_align::core::needleman_wunsch_score(
          pair, match, mismatch, gap);
    });
  });
}

SEXP stride_r_smith_waterman_affine(
    SEXP query,
    SEXP target,
    SEXP match_score,
    SEXP mismatch_score,
    SEXP gap_open_score,
    SEXP gap_extend_score) {
  return call_safely([&] {
    const auto match = score_scalar(match_score, "match_score");
    const auto mismatch = score_scalar(mismatch_score, "mismatch_score");
    const auto gap_open = score_scalar(gap_open_score, "gap_open_score");
    const auto gap_extend = score_scalar(gap_extend_score, "gap_extend_score");
    return apply_pairwise(query, target, [&](const auto& pair) {
      return ::stride_align::core::smith_waterman_affine_score(
          pair, match, mismatch, gap_open, gap_extend);
    });
  });
}

SEXP stride_r_needleman_wunsch_affine(
    SEXP query,
    SEXP target,
    SEXP match_score,
    SEXP mismatch_score,
    SEXP gap_open_score,
    SEXP gap_extend_score) {
  return call_safely([&] {
    const auto match = score_scalar(match_score, "match_score");
    const auto mismatch = score_scalar(mismatch_score, "mismatch_score");
    const auto gap_open = score_scalar(gap_open_score, "gap_open_score");
    const auto gap_extend = score_scalar(gap_extend_score, "gap_extend_score");
    return apply_pairwise(query, target, [&](const auto& pair) {
      return ::stride_align::core::needleman_wunsch_affine_score(
          pair, match, mismatch, gap_open, gap_extend);
    });
  });
}

#if defined(STRIDE_ALIGN_R_PRIMARY)
SEXP stride_r_lcs_length(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_codepoint_pairwise(query, target, [](const auto& left, const auto& right) {
      return ::stride_align::lcs::lcs_length(left, right);
    });
  });
}

SEXP stride_r_lcs_substring_length(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_codepoint_pairwise(query, target, [](const auto& left, const auto& right) {
      return ::stride_align::lcs::lcs_substring_length(left, right);
    });
  });
}

SEXP stride_r_lcs_substring(SEXP query, SEXP target) {
  return call_safely([&] {
    const VectorShape shape = vector_shape(query, target);
    Protection protection;
    SEXP result = protection.add(Rf_allocVector(STRSXP, shape.output_size));
    for (R_xlen_t index = 0; index < shape.output_size; ++index) {
      const R_xlen_t query_index = shape.query_size == 1 ? 0 : index;
      const R_xlen_t target_index = shape.target_size == 1 ? 0 : index;
      SEXP query_chars = STRING_ELT(query, query_index);
      SEXP target_chars = STRING_ELT(target, target_index);
      if (query_chars == NA_STRING || target_chars == NA_STRING) {
        SET_STRING_ELT(result, index, NA_STRING);
        continue;
      }
      const auto left = codepoints(Utf8Input::from_charsxp(query_chars));
      const auto right = codepoints(Utf8Input::from_charsxp(target_chars));
      const auto substring = ::stride_align::lcs::lcs_substring(left, right);
      SET_STRING_ELT(result, index, utf8_charsxp(encode_utf8(substring)));
    }
    return result;
  });
}

enum class NgramMetric { jaccard, dice, cosine, overlap };

SEXP stride_r_ngram(SEXP query, SEXP target, SEXP n, NgramMetric metric) {
  return call_safely([&] {
    const std::size_t size = size_scalar(n, "n");
    if (size == 0U) throw std::invalid_argument("n must be positive");
    return apply_codepoint_pairwise(query, target, [&](const auto& left, const auto& right) {
      switch (metric) {
        case NgramMetric::jaccard: return ::stride_align::ngram::jaccard(left, right, size);
        case NgramMetric::dice: return ::stride_align::ngram::dice(left, right, size);
        case NgramMetric::cosine: return ::stride_align::ngram::cosine(left, right, size);
        case NgramMetric::overlap: return ::stride_align::ngram::overlap(left, right, size);
      }
      return 0.0;
    });
  });
}

SEXP stride_r_jaccard(SEXP query, SEXP target, SEXP n) {
  return stride_r_ngram(query, target, n, NgramMetric::jaccard);
}
SEXP stride_r_dice(SEXP query, SEXP target, SEXP n) {
  return stride_r_ngram(query, target, n, NgramMetric::dice);
}
SEXP stride_r_cosine(SEXP query, SEXP target, SEXP n) {
  return stride_r_ngram(query, target, n, NgramMetric::cosine);
}
SEXP stride_r_overlap(SEXP query, SEXP target, SEXP n) {
  return stride_r_ngram(query, target, n, NgramMetric::overlap);
}

SEXP stride_r_ratcliff_obershelp(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_codepoint_pairwise(query, target, [](const auto& left, const auto& right) {
      return ::stride_align::ratcliff_obershelp::ratcliff_obershelp_similarity(left, right);
    });
  });
}

enum class TokenMetric {
  partial,
  token_sort,
  token_set,
  partial_token_sort,
  partial_token_set,
  weighted,
};

SEXP stride_r_token_metric(SEXP query, SEXP target, TokenMetric metric) {
  return call_safely([&] {
    return apply_codepoint_pairwise(query, target, [&](const auto& left, const auto& right) {
      const auto left_span = std::span<const std::uint32_t>(left.data(), left.size());
      const auto right_span = std::span<const std::uint32_t>(right.data(), right.size());
      switch (metric) {
        case TokenMetric::partial:
          return ::stride_align::partial_ratio::partial_ratio(left, right);
        case TokenMetric::token_sort:
          return ::stride_align::token_ratios::token_sort_ratio(left, right);
        case TokenMetric::token_set:
          return ::stride_align::token_ratios::token_set_ratio(left, right);
        case TokenMetric::partial_token_sort:
          return ::stride_align::wratio::partial_token_sort_ratio_engine<std::uint32_t>(
              left_span, right_span);
        case TokenMetric::partial_token_set:
          return ::stride_align::wratio::partial_token_set_ratio_engine<std::uint32_t>(
              left_span, right_span);
        case TokenMetric::weighted:
          return ::stride_align::wratio::native_wratio(left, right);
      }
      return 0.0;
    });
  });
}

SEXP stride_r_partial_ratio(SEXP query, SEXP target) {
  return stride_r_token_metric(query, target, TokenMetric::partial);
}
SEXP stride_r_token_sort_ratio(SEXP query, SEXP target) {
  return stride_r_token_metric(query, target, TokenMetric::token_sort);
}
SEXP stride_r_token_set_ratio(SEXP query, SEXP target) {
  return stride_r_token_metric(query, target, TokenMetric::token_set);
}
SEXP stride_r_partial_token_sort_ratio(SEXP query, SEXP target) {
  return stride_r_token_metric(query, target, TokenMetric::partial_token_sort);
}
SEXP stride_r_partial_token_set_ratio(SEXP query, SEXP target) {
  return stride_r_token_metric(query, target, TokenMetric::partial_token_set);
}
SEXP stride_r_wratio(SEXP query, SEXP target) {
  return stride_r_token_metric(query, target, TokenMetric::weighted);
}

SEXP stride_r_soundex(SEXP input) {
  return call_safely([&] {
    return apply_string_unary(input, [](SEXP chars) {
      return ::stride_align::phonetic::soundex(utf8_string(chars));
    });
  });
}

SEXP stride_r_soundex_equal(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_logical_pairwise(query, target, [](SEXP left, SEXP right) {
      return ::stride_align::phonetic::soundex(utf8_string(left)) ==
          ::stride_align::phonetic::soundex(utf8_string(right));
    });
  });
}

SEXP stride_r_metaphone(SEXP input, SEXP variant) {
  return call_safely([&] {
    const auto selected = static_cast<::stride_align::phonetic::MetaphoneVariant>(
        size_scalar(variant, "variant"));
    if (selected != ::stride_align::phonetic::MetaphoneVariant::kPhilips &&
        selected != ::stride_align::phonetic::MetaphoneVariant::kJellyfish) {
      throw std::invalid_argument("variant must be 0 or 1");
    }
    return apply_string_unary(input, [&](SEXP chars) {
      return ::stride_align::phonetic::metaphone(utf8_string(chars), selected);
    });
  });
}

SEXP stride_r_metaphone_equal(SEXP query, SEXP target, SEXP variant) {
  return call_safely([&] {
    const auto selected = static_cast<::stride_align::phonetic::MetaphoneVariant>(
        size_scalar(variant, "variant"));
    return apply_logical_pairwise(query, target, [&](SEXP left, SEXP right) {
      return ::stride_align::phonetic::metaphone(utf8_string(left), selected) ==
          ::stride_align::phonetic::metaphone(utf8_string(right), selected);
    });
  });
}

SEXP stride_r_nysiis(SEXP input) {
  return call_safely([&] {
    return apply_string_unary(input, [](SEXP chars) {
      return ::stride_align::phonetic::nysiis(utf8_string(chars));
    });
  });
}

SEXP stride_r_nysiis_equal(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_logical_pairwise(query, target, [](SEXP left, SEXP right) {
      return ::stride_align::phonetic::nysiis(utf8_string(left)) ==
          ::stride_align::phonetic::nysiis(utf8_string(right));
    });
  });
}

SEXP stride_r_match_rating_codex(SEXP input) {
  return call_safely([&] {
    return apply_string_unary(input, [](SEXP chars) {
      return ::stride_align::phonetic::match_rating_codex(utf8_string(chars));
    });
  });
}

SEXP stride_r_match_rating_compare(SEXP query, SEXP target) {
  return call_safely([&] {
    return apply_logical_pairwise(query, target, [](SEXP left, SEXP right) {
      return ::stride_align::phonetic::match_rating_compare(
          utf8_string(left), utf8_string(right));
    });
  });
}

SEXP stride_r_caverphone(SEXP input) {
  return call_safely([&] {
    return apply_string_unary(input, [](SEXP chars) {
      return ::stride_align::phonetic::caverphone(utf8_string(chars));
    });
  });
}

SEXP stride_r_cologne_phonetic(SEXP input) {
  return call_safely([&] {
    return apply_string_unary(input, [](SEXP chars) {
      return ::stride_align::phonetic::cologne_phonetic(
          codepoints(Utf8Input::from_charsxp(chars)));
    });
  });
}

SEXP stride_r_daitch_mokotoff(SEXP input, SEXP branching, SEXP folding) {
  return call_safely([&] {
    if (XLENGTH(branching) != 1 || TYPEOF(branching) != LGLSXP ||
        LOGICAL(branching)[0] == NA_LOGICAL || XLENGTH(folding) != 1 ||
        TYPEOF(folding) != LGLSXP || LOGICAL(folding)[0] == NA_LOGICAL) {
      throw std::invalid_argument("branching and folding must be non-missing logical scalars");
    }
    const bool branch = LOGICAL(branching)[0] != 0;
    const bool fold = LOGICAL(folding)[0] != 0;
    return apply_string_unary(input, [&](SEXP chars) {
      return ::stride_align::phonetic::daitch_mokotoff(
          codepoints(Utf8Input::from_charsxp(chars)), branch, fold);
    });
  });
}

SEXP stride_r_double_metaphone(SEXP input, SEXP max_length, SEXP variant) {
  return call_safely([&] {
    if (TYPEOF(input) != STRSXP) {
      throw std::invalid_argument("input must be a character vector");
    }
    const std::size_t maximum = size_scalar(max_length, "max_length");
    const auto selected = static_cast<::stride_align::phonetic::DoubleMetaphoneVariant>(
        size_scalar(variant, "variant"));
    Protection protection;
    SEXP primary = protection.add(Rf_allocVector(STRSXP, XLENGTH(input)));
    SEXP alternate = protection.add(Rf_allocVector(STRSXP, XLENGTH(input)));
    SEXP result = protection.add(Rf_allocVector(VECSXP, 2));
    for (R_xlen_t index = 0; index < XLENGTH(input); ++index) {
      SEXP chars = STRING_ELT(input, index);
      if (chars == NA_STRING) {
        SET_STRING_ELT(primary, index, NA_STRING);
        SET_STRING_ELT(alternate, index, NA_STRING);
        continue;
      }
      const auto value = ::stride_align::phonetic::double_metaphone(
          utf8_string(chars), maximum, selected);
      SET_STRING_ELT(primary, index, utf8_charsxp(value.primary));
      SET_STRING_ELT(alternate, index, utf8_charsxp(value.alternate));
    }
    SET_VECTOR_ELT(result, 0, primary);
    SET_VECTOR_ELT(result, 1, alternate);
    return result;
  });
}

SEXP stride_r_bmpm_register(SEXP resources) {
  return call_safely([&] {
    if (TYPEOF(resources) != STRSXP) {
      throw std::invalid_argument("BMPM resources must be a named character vector");
    }
    SEXP names = Rf_getAttrib(resources, R_NamesSymbol);
    if (TYPEOF(names) != STRSXP || XLENGTH(names) != XLENGTH(resources)) {
      throw std::invalid_argument("BMPM resources must have one name per value");
    }
    std::unordered_map<std::string, std::string> table;
    table.reserve(static_cast<std::size_t>(XLENGTH(resources)));
    for (R_xlen_t index = 0; index < XLENGTH(resources); ++index) {
      SEXP name = STRING_ELT(names, index);
      SEXP value = STRING_ELT(resources, index);
      if (name == NA_STRING || value == NA_STRING) {
        throw std::invalid_argument("BMPM resource names and values cannot be missing");
      }
      table.emplace(CHAR(name), utf8_string(value));
    }
    ::stride_align::phonetic::bmpm_register_resources(table);
    return R_NilValue;
  });
}

SEXP stride_r_beider_morse(
    SEXP input,
    SEXP rule_type,
    SEXP concat,
    SEXP max_phonemes) {
  return call_safely([&] {
    const auto rule = static_cast<::stride_align::phonetic::BmpmRuleType>(
        size_scalar(rule_type, "rule_type"));
    if (rule != ::stride_align::phonetic::BmpmRuleType::kApprox &&
        rule != ::stride_align::phonetic::BmpmRuleType::kExact) {
      throw std::invalid_argument("rule_type must be 0 or 1");
    }
    if (TYPEOF(concat) != LGLSXP || XLENGTH(concat) != 1 ||
        LOGICAL(concat)[0] == NA_LOGICAL) {
      throw std::invalid_argument("concat must be one non-missing logical value");
    }
    const bool join = LOGICAL(concat)[0] != 0;
    const std::size_t maximum = size_scalar(max_phonemes, "max_phonemes");
    return apply_string_unary(input, [&](SEXP chars) {
      return ::stride_align::phonetic::beider_morse(
          codepoints(Utf8Input::from_charsxp(chars)), rule, join, maximum);
    });
  });
}

std::vector<double> numeric_sequence(SEXP value, const char* name) {
  if (TYPEOF(value) != REALSXP && TYPEOF(value) != INTSXP) {
    throw std::invalid_argument(std::string(name) + " must be a numeric vector");
  }
  if (XLENGTH(value) == 0) {
    throw std::invalid_argument(std::string(name) + " must be non-empty");
  }
  std::vector<double> output(static_cast<std::size_t>(XLENGTH(value)));
  for (R_xlen_t index = 0; index < XLENGTH(value); ++index) {
    const double number = TYPEOF(value) == REALSXP
        ? REAL(value)[index]
        : static_cast<double>(INTEGER(value)[index]);
    if (!std::isfinite(number) ||
        (TYPEOF(value) == INTSXP && INTEGER(value)[index] == NA_INTEGER)) {
      throw std::invalid_argument(std::string(name) + " must contain finite values");
    }
    output[static_cast<std::size_t>(index)] = number;
  }
  return output;
}

std::optional<std::size_t> dtw_window(
    SEXP value, std::size_t query_size, std::size_t target_size) {
  if (value == R_NilValue) return std::nullopt;
  const double number = numeric_scalar(value, "window");
  if (number < 0.0) throw std::invalid_argument("window must be non-negative");
  if (number > 0.0 && number < 1.0) {
    return static_cast<std::size_t>(std::ceil(
        number * static_cast<double>(std::max(query_size, target_size))));
  }
  if (std::trunc(number) != number) {
    throw std::invalid_argument("window must be an integer radius or a fraction in (0, 1)");
  }
  return static_cast<std::size_t>(number);
}

SEXP stride_r_dtw(
    SEXP query,
    SEXP target,
    SEXP window,
    SEXP distance,
    SEXP score_cutoff) {
  return call_safely([&] {
    const auto left = numeric_sequence(query, "query");
    const auto right = numeric_sequence(target, "target");
    if (TYPEOF(distance) != STRSXP || XLENGTH(distance) != 1 ||
        STRING_ELT(distance, 0) == NA_STRING) {
      throw std::invalid_argument("distance must be 'l1' or 'l2_squared'");
    }
    const std::string metric = CHAR(STRING_ELT(distance, 0));
    const auto kind = metric == "l1"
        ? ::stride_align::dtw::DistanceKind::kL1
        : ::stride_align::dtw::DistanceKind::kL2Squared;
    if (metric != "l1" && metric != "l2_squared") {
      throw std::invalid_argument("distance must be 'l1' or 'l2_squared'");
    }
    std::optional<double> cutoff;
    if (score_cutoff != R_NilValue) {
      cutoff = numeric_scalar(score_cutoff, "score_cutoff");
      if (*cutoff < 0.0) throw std::invalid_argument("score_cutoff must be non-negative");
    }
    const double result = ::stride_align::dtw::dtw_score_scalar<double, double>(
        left, right, kind, dtw_window(window, left.size(), right.size()), cutoff);
    return Rf_ScalarReal(result);
  });
}

SEXP alignment_path_to_r(
    const ::stride_align::AlignmentPath& path,
    std::span<const std::uint32_t> query,
    std::span<const std::uint32_t> target) {
  Protection protection;
  constexpr int kFieldCount = 13;
  SEXP result = protection.add(Rf_allocVector(VECSXP, kFieldCount));
  SEXP names = protection.add(Rf_allocVector(STRSXP, kFieldCount));
  const std::array<const char*, kFieldCount> field_names{
      "score", "query_start", "query_end", "target_start", "target_end",
      "operations", "cigar", "matches", "mismatches", "insertions",
      "deletions", "aligned_length", "aligned"};
  for (int index = 0; index < kFieldCount; ++index) {
    SET_STRING_ELT(names, index, Rf_mkChar(field_names[index]));
  }
  SET_VECTOR_ELT(result, 0, Rf_ScalarReal(static_cast<double>(path.score)));
  SET_VECTOR_ELT(result, 1, Rf_ScalarReal(static_cast<double>(path.query_start)));
  SET_VECTOR_ELT(result, 2, Rf_ScalarReal(static_cast<double>(path.query_end)));
  SET_VECTOR_ELT(result, 3, Rf_ScalarReal(static_cast<double>(path.target_start)));
  SET_VECTOR_ELT(result, 4, Rf_ScalarReal(static_cast<double>(path.target_end)));
  SET_VECTOR_ELT(result, 5, Rf_mkString(path.operations.c_str()));
  SET_VECTOR_ELT(result, 6, Rf_mkString(path.cigar.c_str()));
  SET_VECTOR_ELT(result, 7, Rf_ScalarReal(static_cast<double>(path.matches)));
  SET_VECTOR_ELT(result, 8, Rf_ScalarReal(static_cast<double>(path.mismatches)));
  SET_VECTOR_ELT(result, 9, Rf_ScalarReal(static_cast<double>(path.insertions)));
  SET_VECTOR_ELT(result, 10, Rf_ScalarReal(static_cast<double>(path.deletions)));
  SET_VECTOR_ELT(result, 11, Rf_ScalarReal(static_cast<double>(path.aligned_length)));

  std::vector<std::uint32_t> aligned_query;
  std::vector<std::uint32_t> aligned_target;
  aligned_query.reserve(path.operations.size());
  aligned_target.reserve(path.operations.size());
  std::size_t query_index = path.query_start;
  std::size_t target_index = path.target_start;
  constexpr std::uint32_t kGap = static_cast<std::uint32_t>('-');
  for (const char operation : path.operations) {
    if (operation == '=' || operation == 'X') {
      aligned_query.push_back(query[query_index++]);
      aligned_target.push_back(target[target_index++]);
    } else if (operation == 'D') {
      aligned_query.push_back(query[query_index++]);
      aligned_target.push_back(kGap);
    } else {
      aligned_query.push_back(kGap);
      aligned_target.push_back(target[target_index++]);
    }
  }
  SEXP aligned = protection.add(Rf_allocVector(STRSXP, 2));
  SET_STRING_ELT(aligned, 0, utf8_charsxp(encode_utf8(aligned_query)));
  SET_STRING_ELT(aligned, 1, utf8_charsxp(encode_utf8(aligned_target)));
  SEXP aligned_names = protection.add(Rf_allocVector(STRSXP, 2));
  SET_STRING_ELT(aligned_names, 0, Rf_mkChar("query"));
  SET_STRING_ELT(aligned_names, 1, Rf_mkChar("target"));
  Rf_setAttrib(aligned, R_NamesSymbol, aligned_names);
  SET_VECTOR_ELT(result, 12, aligned);
  Rf_setAttrib(result, R_NamesSymbol, names);
  SEXP classes = protection.add(Rf_allocVector(STRSXP, 2));
  SET_STRING_ELT(classes, 0, Rf_mkChar("stride_alignment_result"));
  SET_STRING_ELT(classes, 1, Rf_mkChar("list"));
  Rf_setAttrib(result, R_ClassSymbol, classes);
  return result;
}

SEXP stride_r_alignment_path(
    SEXP query,
    SEXP target,
    SEXP local,
    SEXP match_score,
    SEXP mismatch_score,
    SEXP gap_open_score,
    SEXP gap_extend_score) {
  return call_safely([&] {
    if (TYPEOF(query) != STRSXP || XLENGTH(query) != 1 ||
        TYPEOF(target) != STRSXP || XLENGTH(target) != 1 ||
        STRING_ELT(query, 0) == NA_STRING || STRING_ELT(target, 0) == NA_STRING) {
      throw std::invalid_argument(
          "query and target must each be one non-missing character string");
    }
    if (TYPEOF(local) != LGLSXP || XLENGTH(local) != 1 ||
        LOGICAL(local)[0] == NA_LOGICAL) {
      throw std::invalid_argument("local must be one non-missing logical value");
    }
    const auto match = score_scalar(match_score, "match_score");
    const auto mismatch = score_scalar(mismatch_score, "mismatch_score");
    const auto gap_open = score_scalar(gap_open_score, "gap_open_score");
    const auto gap_extend = score_scalar(gap_extend_score, "gap_extend_score");
    const auto query_input = Utf8Input::from_charsxp(STRING_ELT(query, 0));
    const auto target_input = Utf8Input::from_charsxp(STRING_ELT(target, 0));
    const auto pair = prepare_pair(query_input, target_input);
    const bool is_local = LOGICAL(local)[0] != 0;
    ::stride_align::AlignmentPath path;
    if (gap_open == gap_extend) {
      path = is_local
          ? ::stride_align::core::smith_waterman_path(
                pair, match, mismatch, gap_open)
          : ::stride_align::core::needleman_wunsch_path(
                pair, match, mismatch, gap_open);
    } else {
      path = is_local
          ? ::stride_align::core::smith_waterman_affine_path(
                pair, match, mismatch, gap_open, gap_extend)
          : ::stride_align::core::needleman_wunsch_affine_path(
                pair, match, mismatch, gap_open, gap_extend);
    }
    const auto query_points = codepoints(query_input);
    const auto target_points = codepoints(target_input);
    return alignment_path_to_r(path, query_points, target_points);
  });
}

SEXP stride_r_backend_candidates() {
  return call_safely([] { return stride_r_backend_candidates_impl(); });
}
#endif

const R_CallMethodDef kCallMethods[] = {
    {"stride_r_levenshtein", (DL_FUNC)&stride_r_levenshtein, 2},
    {"stride_r_levenshtein_similarity", (DL_FUNC)&stride_r_levenshtein_similarity, 2},
    {"stride_r_osa", (DL_FUNC)&stride_r_osa, 2},
    {"stride_r_osa_similarity", (DL_FUNC)&stride_r_osa_similarity, 2},
    {"stride_r_true_damerau_levenshtein", (DL_FUNC)&stride_r_true_damerau_levenshtein, 2},
    {"stride_r_true_damerau_levenshtein_similarity", (DL_FUNC)&stride_r_true_damerau_levenshtein_similarity, 2},
    {"stride_r_indel", (DL_FUNC)&stride_r_indel, 2},
    {"stride_r_indel_similarity", (DL_FUNC)&stride_r_indel_similarity, 2},
    {"stride_r_hamming", (DL_FUNC)&stride_r_hamming, 2},
    {"stride_r_hamming_similarity", (DL_FUNC)&stride_r_hamming_similarity, 2},
    {"stride_r_jaro", (DL_FUNC)&stride_r_jaro, 2},
    {"stride_r_jaro_winkler", (DL_FUNC)&stride_r_jaro_winkler, 5},
    {"stride_r_smith_waterman", (DL_FUNC)&stride_r_smith_waterman, 5},
    {"stride_r_needleman_wunsch", (DL_FUNC)&stride_r_needleman_wunsch, 5},
    {"stride_r_smith_waterman_affine", (DL_FUNC)&stride_r_smith_waterman_affine, 6},
    {"stride_r_needleman_wunsch_affine", (DL_FUNC)&stride_r_needleman_wunsch_affine, 6},
#if defined(STRIDE_ALIGN_R_PRIMARY)
    {"stride_r_backend_candidates", (DL_FUNC)&stride_r_backend_candidates, 0},
    {"stride_r_lcs_length", (DL_FUNC)&stride_r_lcs_length, 2},
    {"stride_r_lcs_substring_length", (DL_FUNC)&stride_r_lcs_substring_length, 2},
    {"stride_r_lcs_substring", (DL_FUNC)&stride_r_lcs_substring, 2},
    {"stride_r_jaccard", (DL_FUNC)&stride_r_jaccard, 3},
    {"stride_r_dice", (DL_FUNC)&stride_r_dice, 3},
    {"stride_r_cosine", (DL_FUNC)&stride_r_cosine, 3},
    {"stride_r_overlap", (DL_FUNC)&stride_r_overlap, 3},
    {"stride_r_ratcliff_obershelp", (DL_FUNC)&stride_r_ratcliff_obershelp, 2},
    {"stride_r_partial_ratio", (DL_FUNC)&stride_r_partial_ratio, 2},
    {"stride_r_token_sort_ratio", (DL_FUNC)&stride_r_token_sort_ratio, 2},
    {"stride_r_token_set_ratio", (DL_FUNC)&stride_r_token_set_ratio, 2},
    {"stride_r_partial_token_sort_ratio", (DL_FUNC)&stride_r_partial_token_sort_ratio, 2},
    {"stride_r_partial_token_set_ratio", (DL_FUNC)&stride_r_partial_token_set_ratio, 2},
    {"stride_r_wratio", (DL_FUNC)&stride_r_wratio, 2},
    {"stride_r_soundex", (DL_FUNC)&stride_r_soundex, 1},
    {"stride_r_soundex_equal", (DL_FUNC)&stride_r_soundex_equal, 2},
    {"stride_r_metaphone", (DL_FUNC)&stride_r_metaphone, 2},
    {"stride_r_metaphone_equal", (DL_FUNC)&stride_r_metaphone_equal, 3},
    {"stride_r_nysiis", (DL_FUNC)&stride_r_nysiis, 1},
    {"stride_r_nysiis_equal", (DL_FUNC)&stride_r_nysiis_equal, 2},
    {"stride_r_match_rating_codex", (DL_FUNC)&stride_r_match_rating_codex, 1},
    {"stride_r_match_rating_compare", (DL_FUNC)&stride_r_match_rating_compare, 2},
    {"stride_r_caverphone", (DL_FUNC)&stride_r_caverphone, 1},
    {"stride_r_cologne_phonetic", (DL_FUNC)&stride_r_cologne_phonetic, 1},
    {"stride_r_daitch_mokotoff", (DL_FUNC)&stride_r_daitch_mokotoff, 3},
    {"stride_r_double_metaphone", (DL_FUNC)&stride_r_double_metaphone, 3},
    {"stride_r_bmpm_register", (DL_FUNC)&stride_r_bmpm_register, 1},
    {"stride_r_beider_morse", (DL_FUNC)&stride_r_beider_morse, 4},
    {"stride_r_dtw", (DL_FUNC)&stride_r_dtw, 5},
    {"stride_r_alignment_path", (DL_FUNC)&stride_r_alignment_path, 7},
#endif
    {nullptr, nullptr, 0},
};

}  // namespace

#define STRIDE_ALIGN_R_CONCAT_INNER(left, right) left##right
#define STRIDE_ALIGN_R_CONCAT(left, right) STRIDE_ALIGN_R_CONCAT_INNER(left, right)

extern "C" void attribute_visible
STRIDE_ALIGN_R_CONCAT(R_init_, STRIDE_ALIGN_R_DLL)(DllInfo* dll) {
  R_registerRoutines(dll, nullptr, kCallMethods, nullptr, nullptr);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}
