#pragma once

// Zero-copy view of a numpy ndarray. We detect numpy (and any other
// fixed-width SIMD-friendly buffer-protocol object) via the Python
// buffer protocol with PyBUF_FORMAT, then expose the raw pointer +
// length + element size + signed/float-ness so the caller can dispatch
// to the matching kernel cell width.
//
// We deliberately reject object-dtype arrays (format='O') — those are
// boxed PyObjects, which is what the sequence-iteration path is for.
// We also reject non-contiguous arrays so the caller can treat the
// buffer as a flat element stream.
//
// Bytes / 1-byte unicode are NOT classified as ndarray here — those
// are detected with PyBytes_Check / PyUnicode_Check before us and
// routed through their dedicated paths.

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <nanobind/nanobind.h>

namespace stride_align::numpy_view {

namespace nb = nanobind;

enum class NdarrayDtype : std::uint8_t {
  None,
  Int8,
  UInt8,
  Int16,
  UInt16,
  Int32,
  UInt32,
  Int64,
  UInt64,
  Float16,
  Float32,
  Float64,
};

inline std::size_t element_size(NdarrayDtype dtype) noexcept {
  switch (dtype) {
    case NdarrayDtype::Int8:
    case NdarrayDtype::UInt8:
      return 1;
    case NdarrayDtype::Int16:
    case NdarrayDtype::UInt16:
    case NdarrayDtype::Float16:
      return 2;
    case NdarrayDtype::Int32:
    case NdarrayDtype::UInt32:
    case NdarrayDtype::Float32:
      return 4;
    case NdarrayDtype::Int64:
    case NdarrayDtype::UInt64:
    case NdarrayDtype::Float64:
      return 8;
    case NdarrayDtype::None:
      return 0;
  }
  return 0;
}

inline std::string_view dtype_name(NdarrayDtype dtype) noexcept {
  switch (dtype) {
    case NdarrayDtype::Int8:    return "int8";
    case NdarrayDtype::UInt8:   return "uint8";
    case NdarrayDtype::Int16:   return "int16";
    case NdarrayDtype::UInt16:  return "uint16";
    case NdarrayDtype::Int32:   return "int32";
    case NdarrayDtype::UInt32:  return "uint32";
    case NdarrayDtype::Int64:   return "int64";
    case NdarrayDtype::UInt64:  return "uint64";
    case NdarrayDtype::Float16: return "float16";
    case NdarrayDtype::Float32: return "float32";
    case NdarrayDtype::Float64: return "float64";
    case NdarrayDtype::None:    return "<none>";
  }
  return "<unknown>";
}

// Parse a PEP 3118 buffer format string. Skip leading byte-order /
// alignment markers ('@', '=', '<', '>', '|'). On native little-endian
// hosts (every platform we currently support), the kernel's bit-
// pattern equality semantics are insensitive to the prefix because
// nothing reinterprets bytes across word boundaries.
inline NdarrayDtype parse_format(const char* format) noexcept {
  if (format == nullptr) {
    return NdarrayDtype::None;
  }
  while (*format == '@' || *format == '=' || *format == '<' ||
         *format == '>' || *format == '|') {
    ++format;
  }
  // Single-char dtype codes only — multi-character struct fields, named
  // fields, or strides are not supported.
  if (format[0] == '\0' || format[1] != '\0') {
    return NdarrayDtype::None;
  }
  switch (format[0]) {
    case 'b': return NdarrayDtype::Int8;
    case 'B': return NdarrayDtype::UInt8;
    case 'h': return NdarrayDtype::Int16;
    case 'H': return NdarrayDtype::UInt16;
    // 'i'/'I' are platform-dependent in width — on LP64 Linux they are
    // 32-bit; on LLP64 Windows they are also 32-bit. We treat them as
    // 32-bit unconditionally. Same with 'l'/'L' on LP64 (64-bit). Use
    // 'q'/'Q' as canonical 64-bit codes.
    case 'i': return NdarrayDtype::Int32;
    case 'I': return NdarrayDtype::UInt32;
    case 'l': return sizeof(long) == 8 ? NdarrayDtype::Int64 : NdarrayDtype::Int32;
    case 'L': return sizeof(unsigned long) == 8 ? NdarrayDtype::UInt64 : NdarrayDtype::UInt32;
    case 'q': return NdarrayDtype::Int64;
    case 'Q': return NdarrayDtype::UInt64;
    case 'e': return NdarrayDtype::Float16;
    case 'f': return NdarrayDtype::Float32;
    case 'd': return NdarrayDtype::Float64;
    default:  return NdarrayDtype::None;
  }
}

struct View {
  Py_buffer buffer{};
  bool acquired = false;
  // True if the object exposes the buffer protocol but the dtype is
  // not one of the supported SIMD-friendly types (e.g. object arrays,
  // record dtypes, or strided/non-contiguous arrays). The caller can
  // use this to distinguish "treat as non-ndarray" from "explicitly
  // reject an ndarray-like object with unsupported dtype".
  bool unsupported_buffer = false;
  NdarrayDtype dtype = NdarrayDtype::None;

  View() = default;
  View(const View&) = delete;
  View& operator=(const View&) = delete;
  View(View&& other) noexcept
      : buffer(other.buffer),
        acquired(other.acquired),
        unsupported_buffer(other.unsupported_buffer),
        dtype(other.dtype) {
    other.acquired = false;
    other.unsupported_buffer = false;
    other.dtype = NdarrayDtype::None;
  }
  View& operator=(View&& other) noexcept {
    if (this != &other) {
      release();
      buffer = other.buffer;
      acquired = other.acquired;
      unsupported_buffer = other.unsupported_buffer;
      dtype = other.dtype;
      other.acquired = false;
      other.unsupported_buffer = false;
      other.dtype = NdarrayDtype::None;
    }
    return *this;
  }
  ~View() { release(); }

  void release() noexcept {
    if (acquired) {
      PyBuffer_Release(&buffer);
      acquired = false;
      dtype = NdarrayDtype::None;
    }
  }

  std::size_t element_count() const noexcept {
    if (!acquired || dtype == NdarrayDtype::None) {
      return 0;
    }
    const auto elem = element_size(dtype);
    return elem == 0 ? 0 : static_cast<std::size_t>(buffer.len) / elem;
  }

  const void* data() const noexcept { return acquired ? buffer.buf : nullptr; }
};

// Attempt to acquire a numpy-style view. On success the returned View
// has `acquired=true` and `dtype` set. On the unsupported-buffer case
// (object-dtype array, record dtype, non-contiguous, etc.) the View
// has `acquired=false` and `unsupported_buffer=true` — the caller
// should raise rather than silently fall through to the sequence path.
// Bytes / unicode are NOT treated as ndarrays here; the caller must
// check those first.
inline View try_acquire(PyObject* obj) noexcept {
  View view;
  if (obj == nullptr) {
    return view;
  }
  // Skip if we can route this through a more specific path.
  if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
    return view;
  }
  if (PyObject_CheckBuffer(obj) == 0) {
    return view;
  }
  // PyBUF_C_CONTIGUOUS | PyBUF_FORMAT — we need both the format string
  // (so we know the dtype) and contiguous storage (so we can read the
  // buffer as a flat element stream). Non-contiguous arrays fall here
  // too and get reported as unsupported.
  if (PyObject_GetBuffer(obj, &view.buffer,
                         PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) {
    PyErr_Clear();
    // Object exposes the buffer protocol but the request failed —
    // typically a non-contiguous or otherwise unsupported buffer.
    view.unsupported_buffer = true;
    return view;
  }
  view.acquired = true;
  view.dtype = parse_format(view.buffer.format);
  if (view.dtype == NdarrayDtype::None) {
    // Recognised as a buffer but not a SIMD-friendly dtype (object
    // dtype 'O', record dtype, struct, etc.). Release and signal
    // "unsupported" so the caller can raise.
    view.release();
    view.unsupported_buffer = true;
  }
  return view;
}

}  // namespace stride_align::numpy_view
