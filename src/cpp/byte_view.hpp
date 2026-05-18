#pragma once

// Zero-copy byte view of a Python bytes / 1-byte-unicode object. Used by
// both the SIMD batch kernel and the singular levenshtein_score path so
// that bytes/ASCII inputs skip the prepare_alignment vector copy.

#include <cstddef>
#include <cstdint>

#include <nanobind/nanobind.h>

namespace stride_align::byte_view {

namespace nb = nanobind;

enum class ByteCompatKind : std::uint8_t { None, Bytes, Unicode1Byte };

inline ByteCompatKind classify(PyObject* obj) {
  if (PyBytes_Check(obj) != 0) {
    return ByteCompatKind::Bytes;
  }
  if (PyUnicode_Check(obj) != 0 &&
      PyUnicode_KIND(obj) == PyUnicode_1BYTE_KIND) {
    return ByteCompatKind::Unicode1Byte;
  }
  return ByteCompatKind::None;
}

inline void view(
    PyObject* obj,
    ByteCompatKind kind,
    const std::uint8_t*& ptr,
    std::size_t& len) {
  if (kind == ByteCompatKind::Bytes) {
    char* raw = nullptr;
    Py_ssize_t size = 0;
    if (PyBytes_AsStringAndSize(obj, &raw, &size) != 0) {
      throw nb::python_error();
    }
    ptr = reinterpret_cast<const std::uint8_t*>(raw);
    len = static_cast<std::size_t>(size);
  } else {
    // Unicode1Byte. PyUnicode_1BYTE_DATA returns a pointer the object
    // owns; callers must keep the object alive across the kernel's
    // lifetime.
    ptr = PyUnicode_1BYTE_DATA(obj);
    len = static_cast<std::size_t>(PyUnicode_GET_LENGTH(obj));
  }
}

}  // namespace stride_align::byte_view
