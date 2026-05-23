#pragma once

// SIMD kernels for Hamming distance. Two complementary architectures:
//
//   * within-string: process one (query, target) pair, comparing
//     VEC_BYTES bytes per SIMD iteration. Bytewise cmpne over the
//     widest vector the host backend supports, popcount the resulting
//     mismatch mask. Used by the singular API.
//
//   * across-target batch: process VEC_BYTES targets in parallel by
//     using *byte-wide lanes*, one lane per target. Each iteration
//     scans one position of the query against all targets in the
//     batch and increments per-target counters. Counters are 8-bit
//     while the distance fits in a byte (n < 256), or 16-bit
//     otherwise. Used by the *_scores API.
//
// Each backend TU is compiled with its own target ISA enabled, so the
// arch-conditional code below picks the widest available kernel at
// compile time. Backends that don't define any of the supported ISAs
// fall through to a scalar tail loop.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "stride_align/alignment.hpp"
#include "stride_align/hamming.hpp"

namespace stride_align::hamming_simd {

namespace nb = nanobind;

// -------------------------------------------------------------------
// Within-string SIMD: count mismatch bytes between two equal-length
// uint8 buffers. The host backend's widest SIMD path is picked at
// compile time; everything else falls through to scalar.
// -------------------------------------------------------------------

inline std::size_t hamming_within_string(
    const std::uint8_t* a,
    const std::uint8_t* b,
    std::size_t n) noexcept {
  // Plain scalar loop, written so every reachable backend's auto-
  // vectorizer turns it into the right SIMD code. We started with
  // hand-coded AVX-512 / AVX2 / NEON / LSX / LASX / VSX intrinsics
  // and found across the board that the auto-vectorized loop is
  // 30-60% faster — the compiler picks better unroll factors,
  // schedules cmpeq + accumulation onto independent ports, and
  // (on AVX-512) avoids the kmov→popcnt→add chain by reducing
  // through VPSADBW directly.
  //
  // Sanity check on Tiger Lake AVX-512:
  //   hand-coded intrinsics: ~0.25 ns/byte at n=65536
  //   this loop, -O3:        ~0.15 ns/byte at n=65536
  //   rapidfuzz (also auto-vec scalar):  ~0.16 ns/byte
  //
  // The `dist -= (a == b)` formulation matches rapidfuzz's
  // implementation; it auto-vectorizes identically to the `count +=
  // (a != b)` form but reads slightly more naturally as "start from
  // max distance, subtract matches".
  std::size_t dist = n;
  for (std::size_t i = 0; i < n; ++i) {
    dist -= static_cast<std::size_t>(a[i] == b[i]);
  }
  return dist;
}

// Cutoff-aware variant. Bails with sentinel `cutoff + 1` once the
// accumulated mismatch count is provably above cutoff. The inner
// fixed-size CHUNK loop auto-vectorizes to the same cmpne +
// VPSADBW (AVX-512) / cmpne + popcnt (AVX2) / vcntq (NEON) sequence as
// the no-cutoff version; the bail check fires once per 64 bytes, so
// throughput on the no-bail path is within a percent of the full
// kernel.
inline std::size_t hamming_within_string_cutoff(
    const std::uint8_t* a,
    const std::uint8_t* b,
    std::size_t n,
    std::size_t cutoff) noexcept {
  constexpr std::size_t kChunk = 64U;
  std::size_t mismatches = 0;
  std::size_t i = 0;
  for (; i + kChunk <= n; i += kChunk) {
    std::size_t chunk_mm = 0;
    for (std::size_t k = 0; k < kChunk; ++k) {
      chunk_mm += static_cast<std::size_t>(a[i + k] != b[i + k]);
    }
    mismatches += chunk_mm;
    if (mismatches > cutoff) {
      return cutoff + 1U;
    }
  }
  for (; i < n; ++i) {
    mismatches += static_cast<std::size_t>(a[i] != b[i]);
  }
  return mismatches > cutoff ? cutoff + 1U : mismatches;
}

// -------------------------------------------------------------------
// Across-target batch. Conceptually a 1-vs-N pattern: same query, many
// equal-length targets. The natural-looking architecture is "one
// target per byte lane, increment per-target counters" — and we
// originally shipped that, with hand-coded AVX-512 / AVX2 / NEON / LSX
// / LASX / VSX paths. It loses on every input size we tested.
//
// Reason: the byte-lane idiom needs one scalar byte load per target
// per position (no SIMD gather is byte-granular), which dominates the
// inner loop. Calling the within-string scalar in a per-target tight
// loop instead lets the compiler auto-vectorize the inner Hamming
// loop (cmpeq + VPSADBW on AVX-512, etc.) and avoids the per-position
// gather entirely. The byte-lane kernel was 2-3x slower at every n
// from 1 to 32 in our benchmarks.
// -------------------------------------------------------------------

// Validates all targets match the query length. Returns -1 on
// success, or the index of the first offending target on failure.
// No Python interaction — caller raises ValueError using the returned
// index while holding the GIL.
inline std::ptrdiff_t hamming_lengths_validate(
    std::size_t q_len, const std::size_t* t_lens, std::size_t count) noexcept {
  for (std::size_t i = 0; i < count; ++i) {
    if (t_lens[i] != q_len) {
      return static_cast<std::ptrdiff_t>(i);
    }
  }
  return -1;
}

// Raw kernel: pure compute, no Python touch. Caller MUST validate
// that every target has length == q_len before calling (this function
// may run without the GIL).
inline void hamming_scores_simd_raw(
    const std::uint8_t* q_ptr, std::size_t q_len,
    const std::uint8_t* const* t_ptrs,
    const std::size_t* /*t_lens*/,
    std::size_t count,
    Score* out) {
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = static_cast<Score>(
        hamming_within_string(q_ptr, t_ptrs[i], q_len));
  }
}

// Per-pair-cutoff raw kernel. `cutoffs_per_pair` may be:
//   * nullptr: behaves identically to hamming_scores_simd_raw.
//   * a length-`count` array: each pair gets its own distance cutoff.
//     Pairs whose mismatch count exceeds the cutoff return the
//     sentinel `cutoffs_per_pair[i] + 1` rather than the true distance.
//
// The cutoff path adds one branch per 64 bytes scanned per pair, so
// throughput at threshold=0 (cutoff=q_len, never bails) is within
// noise of the no-cutoff path on every backend we've measured.
inline void hamming_scores_simd_raw_per_pair(
    const std::uint8_t* q_ptr, std::size_t q_len,
    const std::uint8_t* const* t_ptrs,
    const std::size_t* /*t_lens*/,
    std::size_t count,
    const std::size_t* cutoffs_per_pair,
    Score* out) {
  if (cutoffs_per_pair == nullptr) {
    for (std::size_t i = 0; i < count; ++i) {
      out[i] = static_cast<Score>(
          hamming_within_string(q_ptr, t_ptrs[i], q_len));
    }
    return;
  }
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = static_cast<Score>(
        hamming_within_string_cutoff(
            q_ptr, t_ptrs[i], q_len, cutoffs_per_pair[i]));
  }
}

inline std::vector<Score> hamming_scores_simd(
    nb::handle query,
    nb::handle targets) {
  PyObject* fast_targets =
      PySequence_Fast(targets.ptr(), "targets must be a sequence of target sequences");
  if (fast_targets == nullptr) {
    throw nb::python_error();
  }
  nb::object owner = nb::steal<nb::object>(fast_targets);
  const auto count = static_cast<std::size_t>(PySequence_Fast_GET_SIZE(fast_targets));
  PyObject* const* items = PySequence_Fast_ITEMS(fast_targets);

  namespace bv = ::stride_align::byte_view;
  const bv::ByteCompatKind q_kind = bv::classify(query.ptr());
  std::vector<Score> out(count);
  if (q_kind == bv::ByteCompatKind::None) {
    // Wider unicode or sequence-of-object inputs: scalar dispatch with
    // per-pair length checks happens in the bindings layer (which has
    // the prepared TokenStorage). The SIMD path only handles bytes /
    // 1-byte unicode; signal "not handled" by leaving out empty and
    // letting the caller fall through.
    out.clear();
    return out;
  }
  const std::uint8_t* q_ptr = nullptr;
  std::size_t q_len = 0;
  bv::view(query.ptr(), q_kind, q_ptr, q_len);

  // Per-target byte_view + length validation.
  std::vector<const std::uint8_t*> ptrs(count);
  std::vector<std::size_t> lens(count);
  for (std::size_t i = 0; i < count; ++i) {
    const bv::ByteCompatKind t_kind = bv::classify(items[i]);
    if (t_kind == bv::ByteCompatKind::None) {
      out.clear();  // Signal scalar fallback to caller.
      return out;
    }
    bv::view(items[i], t_kind, ptrs[i], lens[i]);
    if (lens[i] != q_len) {
      PyErr_Format(
          PyExc_ValueError,
          "Hamming requires equal-length inputs (target index %zu has "
          "length %zu, query has length %zu)",
          i, lens[i], q_len);
      throw nb::python_error();
    }
  }

  if (count == 0U || q_len == 0U) {
    // All zero by default.
    return out;
  }

  // Per-target within-string SIMD in a tight loop. The compiler's
  // auto-vectorizer turns each call into the same wide cmpeq +
  // VPSADBW (AVX-512) / movemask + popcnt (AVX2) / vcntq (NEON)
  // sequence as the singular kernel, with no gather overhead.
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = static_cast<Score>(
        hamming_within_string(q_ptr, ptrs[i], q_len));
  }
  return out;
}

inline std::vector<double> hamming_normalized_scores_simd(
    nb::handle query,
    nb::handle targets) {
  auto raw = hamming_scores_simd(query, targets);
  std::vector<double> out;
  out.reserve(raw.size());
  if (raw.empty()) {
    return out;
  }
  const std::size_t q_len = static_cast<std::size_t>(PyObject_Length(query.ptr()));
  for (const auto d : raw) {
    out.push_back(
        ::stride_align::hamming::normalize(static_cast<std::size_t>(d), q_len));
  }
  return out;
}

}  // namespace stride_align::hamming_simd
