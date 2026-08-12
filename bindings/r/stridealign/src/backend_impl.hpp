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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "target_profile.hpp"
#include "stride_align/core.hpp"

#if defined(STRIDE_ALIGN_R_PRIMARY)
#include "cpu_detect.hpp"
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
