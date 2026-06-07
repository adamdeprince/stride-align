#pragma once

#include <Python.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include <nanobind/nanobind.h>

#include "numpy_view.hpp"
#include "promotion.hpp"
#include "stride_align/alignment.hpp"

namespace stride_align {

namespace nb = nanobind;

enum class OutputKind : std::uint8_t {
  bytes,
  unicode,
  sequence,
};

enum class KernelBits : std::uint8_t {
  bits8 = 8,
  bits16 = 16,
  bits32 = 32,
  bits64 = 64,
};

using TokenStorage = std::variant<
    std::vector<std::uint8_t>,
    std::vector<std::uint16_t>,
    std::vector<std::uint32_t>,
    std::vector<std::uint64_t>>;

struct PreparedAlignment {
  OutputKind output_kind = OutputKind::sequence;
  KernelBits kernel_bits = KernelBits::bits64;
  std::uint64_t symbol_limit = 0;
  std::uint64_t score_bound = 0;
  TokenStorage query_tokens;
  TokenStorage target_tokens;
  std::vector<nb::object> query_items;
  std::vector<nb::object> target_items;
};

namespace detail {

[[noreturn]] inline void throw_type_error(const std::string& message) {
  PyErr_SetString(PyExc_TypeError, message.c_str());
  throw nb::python_error();
}

[[noreturn]] inline void throw_overflow_error(const std::string& message) {
  PyErr_SetString(PyExc_OverflowError, message.c_str());
  throw nb::python_error();
}

[[noreturn]] inline void throw_value_error(const std::string& message) {
  PyErr_SetString(PyExc_ValueError, message.c_str());
  throw nb::python_error();
}

// Reject ``str`` / ``bytes`` as a batch ``targets`` argument. Without
// this, PySequence_Fast happily treats a single string as a sequence
// of 1-char strings (or single integers, for bytes), which is never
// what the caller meant. Centralised so every batch binding can call
// it before falling through to PySequence_Fast — saves the Python-
// level ``isinstance(targets, (str, bytes))`` guard that used to live
// in every wrapper.
inline void reject_str_or_bytes_targets(PyObject* targets) {
  if (PyUnicode_Check(targets) != 0 || PyBytes_Check(targets) != 0) {
    throw_type_error(
        "targets must be an iterable of target sequences, "
        "not a single str/bytes");
  }
}

inline nb::object checked_steal(PyObject* object) {
  if (object == nullptr) {
    throw nb::python_error();
  }
  return nb::steal<nb::object>(object);
}

inline std::uint64_t magnitude(Score value) noexcept {
  if (value >= 0) {
    return static_cast<std::uint64_t>(value);
  }

  return static_cast<std::uint64_t>(-(value + 1)) + 1U;
}

inline std::uint64_t saturating_add(std::uint64_t lhs, std::uint64_t rhs) noexcept {
  const auto limit = std::numeric_limits<std::uint64_t>::max();
  if (limit - lhs < rhs) {
    return limit;
  }
  return lhs + rhs;
}

inline std::uint64_t saturating_multiply(std::uint64_t lhs, std::uint64_t rhs) noexcept {
  if (lhs == 0 || rhs == 0) {
    return 0;
  }

  const auto limit = std::numeric_limits<std::uint64_t>::max();
  if (lhs > limit / rhs) {
    return limit;
  }
  return lhs * rhs;
}

inline std::uint64_t compute_score_bound(
    std::size_t query_size,
    std::size_t target_size,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score) noexcept {
  const auto step_limit = std::max({
      magnitude(match_score),
      magnitude(mismatch_score),
      magnitude(gap_open_score),
      magnitude(gap_extend_score),
  });
  const auto path_length = saturating_add(
      static_cast<std::uint64_t>(query_size),
      static_cast<std::uint64_t>(target_size));
  return saturating_multiply(path_length, step_limit);
}

inline std::uint64_t compute_score_bound(
    std::size_t query_size,
    std::size_t target_size,
    Score match_score,
    Score mismatch_score,
    Score gap_score) noexcept {
  return compute_score_bound(
      query_size,
      target_size,
      match_score,
      mismatch_score,
      gap_score,
      gap_score);
}

// Matrix-mode analogue: the substitution matrix replaces the per-cell
// match / mismatch scores, so the step_limit becomes
// ``max(|matrix entry|, |gap_open|, |gap_extend|)``. The caller is
// expected to have computed the matrix max-abs once (see
// ``SubstitutionMatrix.max_abs`` Python-side, or
// ``matrix_max_abs_with_*`` C++-side in ``farrar_fixed_kernel.hpp``)
// rather than scanning the matrix every call. Multiplies the
// step_limit by ``query + target`` length to bound the worst-case
// path-summed absolute score, same shape as the match/mismatch path
// above; ``select_kernel_bits`` then picks the narrowest cell that
// holds the bound.
inline std::uint64_t compute_score_bound_matrix(
    std::size_t query_size,
    std::size_t target_size,
    std::uint32_t matrix_max_abs,
    Score gap_open_score,
    Score gap_extend_score) noexcept {
  const auto step_limit = std::max({
      static_cast<std::uint64_t>(matrix_max_abs),
      magnitude(gap_open_score),
      magnitude(gap_extend_score),
  });
  const auto path_length = saturating_add(
      static_cast<std::uint64_t>(query_size),
      static_cast<std::uint64_t>(target_size));
  return saturating_multiply(path_length, step_limit);
}

inline std::uint64_t compute_score_bound_matrix(
    std::size_t query_size,
    std::size_t target_size,
    std::uint32_t matrix_max_abs,
    Score gap_score) noexcept {
  return compute_score_bound_matrix(
      query_size,
      target_size,
      matrix_max_abs,
      gap_score,
      gap_score);
}

inline KernelBits select_kernel_bits(std::uint64_t symbol_limit, std::uint64_t score_bound) {
  if (score_bound <= static_cast<std::uint64_t>(std::numeric_limits<std::int8_t>::max()) &&
      symbol_limit <= static_cast<std::uint64_t>(std::numeric_limits<std::uint8_t>::max())) {
    return KernelBits::bits8;
  }

  if (score_bound <= static_cast<std::uint64_t>(std::numeric_limits<std::int16_t>::max()) &&
      symbol_limit <= static_cast<std::uint64_t>(std::numeric_limits<std::uint16_t>::max())) {
    return KernelBits::bits16;
  }

  if (score_bound <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) &&
      symbol_limit <= static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
    return KernelBits::bits32;
  }

  if (score_bound > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw_overflow_error("alignment score bound exceeds the 64-bit kernel capacity");
  }

  return KernelBits::bits64;
}

inline KernelBits parse_forced_kernel_bits(unsigned int width) {
  switch (width) {
    case 8:
      return KernelBits::bits8;
    case 16:
      return KernelBits::bits16;
    case 32:
      return KernelBits::bits32;
    case 64:
      return KernelBits::bits64;
    default:
      throw_value_error("width must be None, 0, 8, 16, 32, or 64");
  }
}

inline KernelBits apply_forced_kernel_bits(KernelBits automatic, unsigned int width) {
  if (width == 0) {
    return automatic;
  }

  const auto forced = parse_forced_kernel_bits(width);
  if (static_cast<unsigned int>(forced) < static_cast<unsigned int>(automatic)) {
    throw_value_error("width cannot be narrower than the automatically selected kernel width");
  }

  return forced;
}

template <typename Integer>
std::uint64_t max_symbol_value(std::span<const Integer> values) noexcept {
  std::uint64_t max_value = 0;
  for (const auto value : values) {
    max_value = std::max(max_value, static_cast<std::uint64_t>(value));
  }
  return max_value;
}

inline TokenStorage copy_bytes_tokens(PyObject* bytes_object, std::uint64_t& symbol_limit) {
  char* raw_data = nullptr;
  Py_ssize_t size = 0;
  if (PyBytes_AsStringAndSize(bytes_object, &raw_data, &size) != 0) {
    throw nb::python_error();
  }

  std::vector<std::uint8_t> tokens(static_cast<std::size_t>(size));
  if (size > 0) {
    std::memcpy(tokens.data(), raw_data, static_cast<std::size_t>(size));
  }
  symbol_limit = max_symbol_value<std::uint8_t>(tokens);
  return tokens;
}

inline TokenStorage copy_unicode_tokens(PyObject* unicode_object, std::uint64_t& symbol_limit) {
  // PyUnicode_READY skipped: no-op on 3.12+ and references the private
  // _PyUnicode_Ready symbol that nanobind's macOS sym list filters out on
  // 3.10/3.11. Strings reaching us from the Python C API are already ready.

  const auto size = static_cast<std::size_t>(PyUnicode_GET_LENGTH(unicode_object));
  switch (PyUnicode_KIND(unicode_object)) {
    case PyUnicode_1BYTE_KIND: {
      const auto* data = PyUnicode_1BYTE_DATA(unicode_object);
      std::vector<std::uint8_t> tokens(size);
      if (size > 0) {
        std::memcpy(tokens.data(), data, size * sizeof(std::uint8_t));
      }
      symbol_limit = max_symbol_value<std::uint8_t>(tokens);
      return tokens;
    }
    case PyUnicode_2BYTE_KIND: {
      const auto* data = PyUnicode_2BYTE_DATA(unicode_object);
      std::vector<std::uint16_t> tokens(size);
      if (size > 0) {
        std::memcpy(tokens.data(), data, size * sizeof(std::uint16_t));
      }
      symbol_limit = max_symbol_value<std::uint16_t>(tokens);
      return tokens;
    }
    case PyUnicode_4BYTE_KIND: {
      const auto* data = PyUnicode_4BYTE_DATA(unicode_object);
      std::vector<std::uint32_t> tokens(size);
      if (size > 0) {
        std::memcpy(tokens.data(), data, size * sizeof(std::uint32_t));
      }
      symbol_limit = max_symbol_value<std::uint32_t>(tokens);
      return tokens;
    }
    default:
      throw_type_error("unsupported Python Unicode storage kind");
  }
}

inline std::vector<nb::object> materialize_sequence_items(
    nb::handle input,
    std::string_view argument_name) {
  const std::string error_message =
      std::string(argument_name) +
      " must be bytes, str, or a sequence of immutable hashable objects";
  PyObject* fast_sequence = PySequence_Fast(input.ptr(), error_message.c_str());
  if (fast_sequence == nullptr) {
    throw nb::python_error();
  }

  nb::object owner = nb::steal<nb::object>(fast_sequence);
  const auto size = static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_sequence));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_sequence);

  std::vector<nb::object> output;
  output.reserve(size);
  for (std::size_t index = 0; index < size; ++index) {
    output.emplace_back(nb::borrow<nb::object>(items[index]));
  }

  return output;
}

inline std::pair<std::vector<std::uint64_t>, std::vector<std::uint64_t>> serialize_object_tokens(
    const std::vector<nb::object>& query_items,
    const std::vector<nb::object>& target_items,
    std::uint64_t& symbol_limit) {
  nb::dict token_map;
  std::uint64_t next_token = 0;

  const auto encode = [&](const std::vector<nb::object>& items) {
    std::vector<std::uint64_t> encoded;
    encoded.reserve(items.size());

    for (const auto& item : items) {
      PyObject* existing = PyDict_GetItemWithError(token_map.ptr(), item.ptr());
      if (existing != nullptr) {
        const auto token = PyLong_AsUnsignedLongLong(existing);
        if (token == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
          throw nb::python_error();
        }
        encoded.push_back(static_cast<std::uint64_t>(token));
        continue;
      }

      if (PyErr_Occurred()) {
        throw nb::python_error();
      }

      if (next_token == std::numeric_limits<std::uint64_t>::max()) {
        throw_overflow_error("too many unique objects to serialize into 64-bit tokens");
      }

      ++next_token;
      nb::int_ token_value(next_token);
      if (PyDict_SetItem(token_map.ptr(), item.ptr(), token_value.ptr()) != 0) {
        throw nb::python_error();
      }
      encoded.push_back(next_token);
    }

    return encoded;
  };

  auto query_tokens = encode(query_items);
  auto target_tokens = encode(target_items);
  symbol_limit = next_token;
  return {std::move(query_tokens), std::move(target_tokens)};
}

template <typename To>
TokenStorage cast_storage(const TokenStorage& storage) {
  return std::visit(
      [](const auto& tokens) -> TokenStorage {
        using From = typename std::decay_t<decltype(tokens)>::value_type;
        return TokenStorage(
            promote::cast_vector<To, From>(std::span<const From>(tokens.data(), tokens.size())));
      },
      storage);
}

inline TokenStorage cast_storage(const TokenStorage& storage, KernelBits kernel_bits) {
  switch (kernel_bits) {
    case KernelBits::bits8:
      return cast_storage<std::uint8_t>(storage);
    case KernelBits::bits16:
      return cast_storage<std::uint16_t>(storage);
    case KernelBits::bits32:
      return cast_storage<std::uint32_t>(storage);
    case KernelBits::bits64:
      return cast_storage<std::uint64_t>(storage);
  }

  throw_type_error("unsupported kernel bit width");
}

inline nb::object make_bytes_output(
    std::span<const std::uint64_t> symbols,
    std::size_t start_index,
    std::string_view operations,
    char gap_operation) {
  std::string output;
  output.reserve(operations.size());

  std::size_t index = start_index;
  for (const char operation : operations) {
    if (operation == gap_operation) {
      output.push_back('-');
      continue;
    }

    output.push_back(static_cast<char>(static_cast<unsigned char>(symbols[index])));
    ++index;
  }

  return nb::bytes(output.data(), output.size());
}

inline void append_utf8(std::string& output, Py_UCS4 codepoint) {
  if (codepoint <= 0x7fU) {
    output.push_back(static_cast<char>(codepoint));
    return;
  }

  if (codepoint <= 0x7ffU) {
    output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    return;
  }

  if (codepoint <= 0xffffU) {
    output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
    output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    return;
  }

  output.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
  output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
  output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
  output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
}

inline nb::object make_unicode_output(
    std::span<const std::uint64_t> symbols,
    std::size_t start_index,
    std::string_view operations,
    char gap_operation) {
  std::string output;
  output.reserve(operations.size());

  std::size_t index = start_index;
  for (const char operation : operations) {
    if (operation == gap_operation) {
      output.push_back('-');
      continue;
    }

    append_utf8(output, static_cast<Py_UCS4>(symbols[index]));
    ++index;
  }

  return detail::checked_steal(
      PyUnicode_FromStringAndSize(output.data(), static_cast<Py_ssize_t>(output.size())));
}

inline nb::object make_sequence_output(
    const std::vector<nb::object>& items,
    std::size_t start_index,
    std::string_view operations,
    char gap_operation) {
  nb::list output;
  std::size_t index = start_index;

  for (const char operation : operations) {
    if (operation == gap_operation) {
      output.append(nb::none());
      continue;
    }

    output.append(items[index]);
    ++index;
  }

  return detail::checked_steal(PyList_AsTuple(output.ptr()));
}

}  // namespace detail

inline PreparedAlignment prepare_alignment(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  const bool query_is_bytes = PyBytes_Check(query.ptr()) != 0;
  const bool target_is_bytes = PyBytes_Check(target.ptr()) != 0;
  const bool query_is_unicode = PyUnicode_Check(query.ptr()) != 0;
  const bool target_is_unicode = PyUnicode_Check(target.ptr()) != 0;

  if ((query_is_bytes && target_is_unicode) || (query_is_unicode && target_is_bytes)) {
    detail::throw_type_error("bytes and str inputs cannot be aligned directly against each other");
  }

  // ndarray inputs are accepted on the Farrar score-only path
  // (smith_waterman_score / needleman_wunsch_score), not on the
  // traceback / path / cigar paths that route through this function.
  // Raise here rather than letting ndarrays silently take the slow
  // PyObject sequence path. Unsupported-dtype ndarrays (object,
  // record, non-contiguous) also raise.
  for (auto* obj : {query.ptr(), target.ptr()}) {
    if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
      continue;
    }
    auto v = ::stride_align::numpy_view::try_acquire(obj);
    if (v.acquired || v.unsupported_buffer) {
      detail::throw_type_error(
          "ndarray inputs are only supported on the score-only path "
          "(smith_waterman_score / needleman_wunsch_score); "
          "traceback / path / cigar variants do not yet support "
          "ndarray inputs");
    }
  }

  if (query_is_bytes && target_is_bytes) {
    std::uint64_t query_symbol_limit = 0;
    std::uint64_t target_symbol_limit = 0;
    TokenStorage query_tokens = detail::copy_bytes_tokens(query.ptr(), query_symbol_limit);
    TokenStorage target_tokens = detail::copy_bytes_tokens(target.ptr(), target_symbol_limit);

    PreparedAlignment prepared;
    prepared.output_kind = OutputKind::bytes;
    prepared.symbol_limit = std::max(query_symbol_limit, target_symbol_limit);
    prepared.score_bound = detail::compute_score_bound(
        std::get<std::vector<std::uint8_t>>(query_tokens).size(),
        std::get<std::vector<std::uint8_t>>(target_tokens).size(),
        match_score,
        mismatch_score,
        gap_open_score,
        gap_extend_score);
    prepared.kernel_bits = detail::apply_forced_kernel_bits(
        detail::select_kernel_bits(prepared.symbol_limit, prepared.score_bound),
        width);
    prepared.query_tokens = detail::cast_storage(query_tokens, prepared.kernel_bits);
    prepared.target_tokens = detail::cast_storage(target_tokens, prepared.kernel_bits);
    return prepared;
  }

  if (query_is_unicode && target_is_unicode) {
    std::uint64_t query_symbol_limit = 0;
    std::uint64_t target_symbol_limit = 0;
    TokenStorage query_tokens = detail::copy_unicode_tokens(query.ptr(), query_symbol_limit);
    TokenStorage target_tokens = detail::copy_unicode_tokens(target.ptr(), target_symbol_limit);

    const auto query_size = std::visit(
        [](const auto& tokens) { return tokens.size(); },
        query_tokens);
    const auto target_size = std::visit(
        [](const auto& tokens) { return tokens.size(); },
        target_tokens);

    PreparedAlignment prepared;
    prepared.output_kind = OutputKind::unicode;
    prepared.symbol_limit = std::max(query_symbol_limit, target_symbol_limit);
    prepared.score_bound =
        detail::compute_score_bound(
            query_size,
            target_size,
            match_score,
            mismatch_score,
            gap_open_score,
            gap_extend_score);
    prepared.kernel_bits = detail::apply_forced_kernel_bits(
        detail::select_kernel_bits(prepared.symbol_limit, prepared.score_bound),
        width);
    prepared.query_tokens = detail::cast_storage(query_tokens, prepared.kernel_bits);
    prepared.target_tokens = detail::cast_storage(target_tokens, prepared.kernel_bits);
    return prepared;
  }

  PreparedAlignment prepared;
  prepared.output_kind = OutputKind::sequence;
  prepared.query_items = detail::materialize_sequence_items(query, "query");
  prepared.target_items = detail::materialize_sequence_items(target, "target");

  std::uint64_t symbol_limit = 0;
  auto [query_tokens64, target_tokens64] = detail::serialize_object_tokens(
      prepared.query_items,
      prepared.target_items,
      symbol_limit);
  prepared.symbol_limit = symbol_limit;
  prepared.score_bound = detail::compute_score_bound(
      prepared.query_items.size(),
      prepared.target_items.size(),
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score);
  prepared.kernel_bits = detail::apply_forced_kernel_bits(
      detail::select_kernel_bits(prepared.symbol_limit, prepared.score_bound),
      width);
  prepared.query_tokens = detail::cast_storage(TokenStorage(std::move(query_tokens64)), prepared.kernel_bits);
  prepared.target_tokens = detail::cast_storage(TokenStorage(std::move(target_tokens64)), prepared.kernel_bits);
  return prepared;
}

inline PreparedAlignment prepare_alignment(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_score,
    unsigned int width) {
  return prepare_alignment(
      query,
      target,
      match_score,
      mismatch_score,
      gap_score,
      gap_score,
      width);
}

template <typename Token>
nb::object materialize_query_output(
    const PreparedAlignment& prepared,
    std::span<const Token> query_tokens,
    std::size_t query_start,
    std::string_view operations) {
  if (prepared.output_kind == OutputKind::sequence) {
    return detail::make_sequence_output(prepared.query_items, query_start, operations, 'I');
  }

  std::vector<std::uint64_t> widened =
      promote::cast_vector<std::uint64_t, Token>(query_tokens);
  if (prepared.output_kind == OutputKind::bytes) {
    return detail::make_bytes_output(widened, query_start, operations, 'I');
  }

  return detail::make_unicode_output(widened, query_start, operations, 'I');
}

template <typename Token>
nb::object materialize_target_output(
    const PreparedAlignment& prepared,
    std::span<const Token> target_tokens,
    std::size_t target_start,
    std::string_view operations) {
  if (prepared.output_kind == OutputKind::sequence) {
    return detail::make_sequence_output(prepared.target_items, target_start, operations, 'D');
  }

  std::vector<std::uint64_t> widened =
      promote::cast_vector<std::uint64_t, Token>(target_tokens);
  if (prepared.output_kind == OutputKind::bytes) {
    return detail::make_bytes_output(widened, target_start, operations, 'D');
  }

  return detail::make_unicode_output(widened, target_start, operations, 'D');
}

}  // namespace stride_align
