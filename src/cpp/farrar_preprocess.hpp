#pragma once

#include <Python.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>

#include "preprocess.hpp"

namespace stride_align {

namespace nb = nanobind;

struct PreparedFarrarAlignment {
  KernelBits score_bits = KernelBits::bits64;
  std::uint64_t symbol_count = 0;
  std::uint64_t score_bound = 0;
  std::vector<std::uint8_t> query_tokens;
  std::vector<std::uint8_t> target_tokens;
};

struct PreparedFarrarBatchAlignment {
  KernelBits score_bits = KernelBits::bits64;
  std::uint64_t symbol_count = 0;
  std::uint64_t score_bound = 0;
  std::vector<std::uint8_t> query_tokens;
  std::vector<std::vector<std::uint8_t>> target_tokens;
};

namespace farrar_detail {

inline KernelBits select_score_bits(std::uint64_t score_bound) {
  if (score_bound <= static_cast<std::uint64_t>(std::numeric_limits<std::int8_t>::max())) {
    return KernelBits::bits8;
  }
  if (score_bound <= static_cast<std::uint64_t>(std::numeric_limits<std::int16_t>::max())) {
    return KernelBits::bits16;
  }
  if (score_bound <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
    return KernelBits::bits32;
  }
  if (score_bound > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    detail::throw_overflow_error("alignment score bound exceeds the 64-bit kernel capacity");
  }
  return KernelBits::bits64;
}

inline std::uint8_t next_compact_token(std::size_t size) {
  if (size >= 256) {
    detail::throw_value_error("Farrar token channel supports at most 256 distinct symbols");
  }
  return static_cast<std::uint8_t>(size);
}

inline std::vector<std::uint8_t> copy_bytes_tokens_8(PyObject* bytes_object) {
  char* raw_data = nullptr;
  Py_ssize_t size = 0;
  if (PyBytes_AsStringAndSize(bytes_object, &raw_data, &size) != 0) {
    throw nb::python_error();
  }

  std::vector<std::uint8_t> tokens(static_cast<std::size_t>(size));
  if (size > 0) {
    std::memcpy(tokens.data(), raw_data, static_cast<std::size_t>(size));
  }
  return tokens;
}

inline std::uint64_t byte_symbol_count(
    const std::vector<std::uint8_t>& query,
    const std::vector<std::uint8_t>& target) {
  bool seen[256] = {};
  std::uint64_t count = 0;
  const auto mark = [&](std::uint8_t token) {
    if (!seen[token]) {
      seen[token] = true;
      ++count;
    }
  };
  for (const auto token : query) {
    mark(token);
  }
  for (const auto token : target) {
    mark(token);
  }
  return count;
}

inline std::uint64_t byte_symbol_count(
    const std::vector<std::uint8_t>& query,
    const std::vector<std::vector<std::uint8_t>>& targets) {
  bool seen[256] = {};
  std::uint64_t count = 0;
  const auto mark = [&](std::uint8_t token) {
    if (!seen[token]) {
      seen[token] = true;
      ++count;
    }
  };
  for (const auto token : query) {
    mark(token);
  }
  for (const auto& target : targets) {
    for (const auto token : target) {
      mark(token);
    }
  }
  return count;
}

inline std::vector<std::uint8_t> compact_unicode_tokens(
    PyObject* unicode_object,
    std::unordered_map<Py_UCS4, std::uint8_t>& token_map) {
  if (PyUnicode_READY(unicode_object) != 0) {
    throw nb::python_error();
  }

  const auto size = static_cast<std::size_t>(PyUnicode_GET_LENGTH(unicode_object));
  const int kind = PyUnicode_KIND(unicode_object);
  void* data = PyUnicode_DATA(unicode_object);

  std::vector<std::uint8_t> tokens;
  tokens.reserve(size);
  for (std::size_t index = 0; index < size; ++index) {
    const auto codepoint = static_cast<Py_UCS4>(
        PyUnicode_READ(kind, data, static_cast<Py_ssize_t>(index)));
    auto iterator = token_map.find(codepoint);
    if (iterator == token_map.end()) {
      iterator = token_map.emplace(codepoint, next_compact_token(token_map.size())).first;
    }
    tokens.push_back(iterator->second);
  }
  return tokens;
}

inline std::size_t direct_unicode_lookup_size(PyObject* unicode_object) {
  if (PyUnicode_READY(unicode_object) != 0) {
    throw nb::python_error();
  }

  switch (PyUnicode_KIND(unicode_object)) {
    case PyUnicode_1BYTE_KIND:
      return 256U;
    case PyUnicode_2BYTE_KIND:
      return 65536U;
    default:
      return 0U;
  }
}

struct DirectUnicodeTokenMap {
  static constexpr std::uint16_t missing = std::numeric_limits<std::uint16_t>::max();

  explicit DirectUnicodeTokenMap(std::size_t lookup_size) : lookup(lookup_size, missing) {}

  std::vector<std::uint8_t> encode(PyObject* unicode_object) {
    if (PyUnicode_READY(unicode_object) != 0) {
      throw nb::python_error();
    }

    const auto size = static_cast<std::size_t>(PyUnicode_GET_LENGTH(unicode_object));
    const int kind = PyUnicode_KIND(unicode_object);
    void* data = PyUnicode_DATA(unicode_object);

    std::vector<std::uint8_t> tokens;
    tokens.reserve(size);
    for (std::size_t index = 0; index < size; ++index) {
      const auto codepoint = static_cast<std::size_t>(
          PyUnicode_READ(kind, data, static_cast<Py_ssize_t>(index)));
      std::uint16_t& entry = lookup[codepoint];
      if (entry == missing) {
        entry = next_compact_token(symbol_count);
        ++symbol_count;
      }
      tokens.push_back(static_cast<std::uint8_t>(entry));
    }
    return tokens;
  }

  std::vector<std::uint16_t> lookup;
  std::size_t symbol_count = 0;
};

inline std::vector<nb::object> sequence_items(nb::handle input, std::string_view argument_name) {
  const std::string message =
      std::string(argument_name) +
      " must be bytes, str, or a sequence of immutable hashable objects";
  PyObject* fast_sequence = PySequence_Fast(input.ptr(), message.c_str());
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

inline std::vector<nb::object> sequence_items(PyObject* input, std::string_view argument_name) {
  return sequence_items(nb::handle(input), argument_name);
}

inline std::pair<std::vector<std::uint8_t>, std::vector<std::uint8_t>> compact_object_tokens(
    const std::vector<nb::object>& query_items,
    const std::vector<nb::object>& target_items,
    std::uint64_t& symbol_count) {
  nb::dict token_map;

  const auto encode = [&](const std::vector<nb::object>& items) {
    std::vector<std::uint8_t> encoded;
    encoded.reserve(items.size());

    for (const auto& item : items) {
      PyObject* existing = PyDict_GetItemWithError(token_map.ptr(), item.ptr());
      if (existing != nullptr) {
        const auto token = PyLong_AsUnsignedLong(existing);
        if (token == static_cast<unsigned long>(-1) && PyErr_Occurred()) {
          throw nb::python_error();
        }
        encoded.push_back(static_cast<std::uint8_t>(token));
        continue;
      }

      if (PyErr_Occurred()) {
        throw nb::python_error();
      }

      const auto next_token = next_compact_token(static_cast<std::size_t>(symbol_count));
      nb::int_ token_value(next_token);
      if (PyDict_SetItem(token_map.ptr(), item.ptr(), token_value.ptr()) != 0) {
        throw nb::python_error();
      }
      ++symbol_count;
      encoded.push_back(next_token);
    }

    return encoded;
  };

  auto query_tokens = encode(query_items);
  auto target_tokens = encode(target_items);
  return {std::move(query_tokens), std::move(target_tokens)};
}

inline std::vector<std::uint8_t> compact_object_tokens(
    const std::vector<nb::object>& items,
    nb::dict& token_map,
    std::uint64_t& symbol_count) {
  std::vector<std::uint8_t> encoded;
  encoded.reserve(items.size());

  for (const auto& item : items) {
    PyObject* existing = PyDict_GetItemWithError(token_map.ptr(), item.ptr());
    if (existing != nullptr) {
      const auto token = PyLong_AsUnsignedLong(existing);
      if (token == static_cast<unsigned long>(-1) && PyErr_Occurred()) {
        throw nb::python_error();
      }
      encoded.push_back(static_cast<std::uint8_t>(token));
      continue;
    }

    if (PyErr_Occurred()) {
      throw nb::python_error();
    }

    const auto next_token = next_compact_token(static_cast<std::size_t>(symbol_count));
    nb::int_ token_value(next_token);
    if (PyDict_SetItem(token_map.ptr(), item.ptr(), token_value.ptr()) != 0) {
      throw nb::python_error();
    }
    ++symbol_count;
    encoded.push_back(next_token);
  }

  return encoded;
}

}  // namespace farrar_detail

inline PreparedFarrarAlignment prepare_farrar_alignment(
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

  PreparedFarrarAlignment prepared;

  if (query_is_bytes && target_is_bytes) {
    prepared.query_tokens = farrar_detail::copy_bytes_tokens_8(query.ptr());
    prepared.target_tokens = farrar_detail::copy_bytes_tokens_8(target.ptr());
    prepared.symbol_count =
        farrar_detail::byte_symbol_count(prepared.query_tokens, prepared.target_tokens);
  } else if (query_is_unicode && target_is_unicode) {
    const std::size_t query_lookup_size = farrar_detail::direct_unicode_lookup_size(query.ptr());
    const std::size_t target_lookup_size = farrar_detail::direct_unicode_lookup_size(target.ptr());
    const std::size_t lookup_size = query_lookup_size != 0U && target_lookup_size != 0U
        ? std::max(query_lookup_size, target_lookup_size)
        : 0U;
    if (lookup_size != 0U) {
      farrar_detail::DirectUnicodeTokenMap token_map(lookup_size);
      prepared.query_tokens = token_map.encode(query.ptr());
      prepared.target_tokens = token_map.encode(target.ptr());
      prepared.symbol_count = token_map.symbol_count;
    } else {
      std::unordered_map<Py_UCS4, std::uint8_t> token_map;
      prepared.query_tokens = farrar_detail::compact_unicode_tokens(query.ptr(), token_map);
      prepared.target_tokens = farrar_detail::compact_unicode_tokens(target.ptr(), token_map);
      prepared.symbol_count = token_map.size();
    }
  } else {
    auto query_items = farrar_detail::sequence_items(query, "query");
    auto target_items = farrar_detail::sequence_items(target, "target");
    auto [query_tokens, target_tokens] =
        farrar_detail::compact_object_tokens(query_items, target_items, prepared.symbol_count);
    prepared.query_tokens = std::move(query_tokens);
    prepared.target_tokens = std::move(target_tokens);
  }

  prepared.score_bound = detail::compute_score_bound(
      prepared.query_tokens.size(),
      prepared.target_tokens.size(),
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score);
  prepared.score_bits = detail::apply_forced_kernel_bits(
      farrar_detail::select_score_bits(prepared.score_bound),
      width);
  return prepared;
}

inline PreparedFarrarAlignment prepare_farrar_alignment(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_score,
    unsigned int width) {
  return prepare_farrar_alignment(
      query,
      target,
      match_score,
      mismatch_score,
      gap_score,
      gap_score,
      width);
}

inline PreparedFarrarBatchAlignment prepare_farrar_batch_alignment(
    nb::handle query,
    nb::handle targets,
    Score match_score,
    Score mismatch_score,
    Score gap_open_score,
    Score gap_extend_score,
    unsigned int width) {
  PyObject* fast_targets =
      PySequence_Fast(targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }

  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto target_count = static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  const bool query_is_bytes = PyBytes_Check(query.ptr()) != 0;
  const bool query_is_unicode = PyUnicode_Check(query.ptr()) != 0;

  bool all_targets_bytes = target_count > 0;
  bool all_targets_unicode = target_count > 0;
  for (std::size_t index = 0; index < target_count; ++index) {
    all_targets_bytes = all_targets_bytes && PyBytes_Check(items[index]) != 0;
    all_targets_unicode = all_targets_unicode && PyUnicode_Check(items[index]) != 0;
  }

  if ((query_is_bytes && all_targets_unicode) || (query_is_unicode && all_targets_bytes)) {
    detail::throw_type_error("bytes and str inputs cannot be aligned directly against each other");
  }

  PreparedFarrarBatchAlignment prepared;
  prepared.target_tokens.reserve(target_count);

  if (query_is_bytes && all_targets_bytes) {
    prepared.query_tokens = farrar_detail::copy_bytes_tokens_8(query.ptr());
    for (std::size_t index = 0; index < target_count; ++index) {
      prepared.target_tokens.push_back(farrar_detail::copy_bytes_tokens_8(items[index]));
    }
    prepared.symbol_count =
        farrar_detail::byte_symbol_count(prepared.query_tokens, prepared.target_tokens);
  } else if (query_is_unicode && all_targets_unicode) {
    std::size_t lookup_size = farrar_detail::direct_unicode_lookup_size(query.ptr());
    for (std::size_t index = 0; index < target_count && lookup_size != 0U; ++index) {
      const std::size_t target_lookup_size =
          farrar_detail::direct_unicode_lookup_size(items[index]);
      if (target_lookup_size == 0U) {
        lookup_size = 0U;
        break;
      }
      lookup_size = std::max(lookup_size, target_lookup_size);
    }

    if (lookup_size != 0U) {
      farrar_detail::DirectUnicodeTokenMap token_map(lookup_size);
      prepared.query_tokens = token_map.encode(query.ptr());
      for (std::size_t index = 0; index < target_count; ++index) {
        prepared.target_tokens.push_back(token_map.encode(items[index]));
      }
      prepared.symbol_count = token_map.symbol_count;
    } else {
      std::unordered_map<Py_UCS4, std::uint8_t> token_map;
      prepared.query_tokens = farrar_detail::compact_unicode_tokens(query.ptr(), token_map);
      for (std::size_t index = 0; index < target_count; ++index) {
        prepared.target_tokens.push_back(
            farrar_detail::compact_unicode_tokens(items[index], token_map));
      }
      prepared.symbol_count = token_map.size();
    }
  } else {
    nb::dict token_map;
    auto query_items = farrar_detail::sequence_items(query, "query");
    prepared.query_tokens =
        farrar_detail::compact_object_tokens(query_items, token_map, prepared.symbol_count);
    for (std::size_t index = 0; index < target_count; ++index) {
      auto target_items = farrar_detail::sequence_items(items[index], "target");
      prepared.target_tokens.push_back(
          farrar_detail::compact_object_tokens(target_items, token_map, prepared.symbol_count));
    }
  }

  std::size_t max_target_size = 0;
  for (const auto& target : prepared.target_tokens) {
    max_target_size = std::max(max_target_size, target.size());
  }
  prepared.score_bound = detail::compute_score_bound(
      prepared.query_tokens.size(),
      max_target_size,
      match_score,
      mismatch_score,
      gap_open_score,
      gap_extend_score);
  prepared.score_bits = detail::apply_forced_kernel_bits(
      farrar_detail::select_score_bits(prepared.score_bound),
      width);
  return prepared;
}

inline PreparedFarrarBatchAlignment prepare_farrar_batch_alignment(
    nb::handle query,
    nb::handle targets,
    Score match_score,
    Score mismatch_score,
    Score gap_score,
    unsigned int width) {
  return prepare_farrar_batch_alignment(
      query,
      targets,
      match_score,
      mismatch_score,
      gap_score,
      gap_score,
      width);
}

inline unsigned int unicode_storage_bits(nb::handle value) {
  if (PyUnicode_Check(value.ptr()) == 0) {
    return 0;
  }
  if (PyUnicode_READY(value.ptr()) != 0) {
    throw nb::python_error();
  }

  switch (PyUnicode_KIND(value.ptr())) {
    case PyUnicode_1BYTE_KIND:
      return 8;
    case PyUnicode_2BYTE_KIND:
      return 16;
    case PyUnicode_4BYTE_KIND:
      return 32;
    default:
      detail::throw_type_error("unsupported Python Unicode storage kind");
  }
}

inline void validate_linear_score_width_for_unicode(
    nb::handle query,
    nb::handle target,
    unsigned int width) {
  if (width == 0) {
    return;
  }

  const unsigned int required_width =
      std::max(unicode_storage_bits(query), unicode_storage_bits(target));
  if (required_width != 0 && width < required_width) {
    detail::throw_value_error("width cannot be narrower than the automatically selected kernel width");
  }
}

inline PreparedFarrarAlignment prepare_linear_score_alignment(
    nb::handle query,
    nb::handle target,
    Score match_score,
    Score mismatch_score,
    Score gap_score,
    unsigned int width) {
  validate_linear_score_width_for_unicode(query, target, width);
  return prepare_farrar_alignment(
      query,
      target,
      match_score,
      mismatch_score,
      gap_score,
      width);
}

}  // namespace stride_align
