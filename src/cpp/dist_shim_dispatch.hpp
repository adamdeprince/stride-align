#pragma once

// nanobind-facing dispatchers for the rapidfuzz-shim ``distance``
// classes (``stride_align.rapidfuzz.distance.Indel`` /
// ``Levenshtein``). Each method handles the rapidfuzz contract
// (optional ``processor``, integer/float ``score_cutoff``, score
// translation) entirely in C++ so the Python boundary is crossed
// once per call.
//
// Currently covers Indel and Levenshtein — the two distance classes
// the cross-arch bench measures and the ones that show measurable
// per-call wrapper overhead on the short-pair workload.

#include <cstddef>
#include <cstdint>
#include <span>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "lcs_dispatch.hpp"
#include "levenshtein_dispatch.hpp"  // dispatch_score / dispatch_normalized_score
#include "stride_align/indel.hpp"
#include "stride_align/levenshtein.hpp"
#include "fuzz_shim_dispatch.hpp"  // prepare(), run_kernel()

namespace stride_align::dist_shim {

namespace nb = nanobind;
using ::stride_align::fuzz_shim::prepare;
using ::stride_align::fuzz_shim::run_kernel;
using ::stride_align::fuzz_shim::Prepped;

// Length lookup that mirrors the Python wrappers' notion of "string
// length": Unicode codepoint count for ``str``, byte count for
// ``bytes``, generic ``len()`` for everything else.
inline std::size_t length_of(nb::handle h) {
  if (PyUnicode_Check(h.ptr())) {
    return static_cast<std::size_t>(PyUnicode_GET_LENGTH(h.ptr()));
  }
  if (PyBytes_Check(h.ptr())) {
    return static_cast<std::size_t>(PyBytes_GET_SIZE(h.ptr()));
  }
  const Py_ssize_t n = PyObject_Length(h.ptr());
  if (n < 0) throw nb::python_error();
  return static_cast<std::size_t>(n);
}

// ---- Indel -----------------------------------------------------------------

inline nb::object dispatch_Indel_distance(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  std::size_t cutoff = ::stride_align::indel::kNoCutoff;
  if (!score_cutoff.is_none()) {
    cutoff = static_cast<std::size_t>(nb::cast<std::int64_t>(score_cutoff));
  }
  const std::size_t d = static_cast<std::size_t>(run_kernel(p.ah, p.bh,
      [cutoff](std::span<const std::uint8_t> a,
                std::span<const std::uint8_t> b) -> double {
        return static_cast<double>(
            ::stride_align::indel::indel_distance_u8(a, b, cutoff));
      },
      [cutoff](const std::vector<lcs::Codepoint>& a,
                const std::vector<lcs::Codepoint>& b) -> double {
        return static_cast<double>(::stride_align::indel::indel_distance<lcs::Codepoint>(
            std::span<const lcs::Codepoint>(a),
            std::span<const lcs::Codepoint>(b), cutoff));
      }));
  return nb::cast(d);
}

inline nb::object dispatch_Indel_similarity(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  const std::size_t max_d = length_of(p.ah) + length_of(p.bh);
  std::size_t distance_cutoff = ::stride_align::indel::kNoCutoff;
  if (!score_cutoff.is_none()) {
    const std::size_t sim_cutoff = static_cast<std::size_t>(
        nb::cast<std::int64_t>(score_cutoff));
    distance_cutoff = (max_d > sim_cutoff) ? (max_d - sim_cutoff) : 0U;
  }
  const std::size_t d = static_cast<std::size_t>(run_kernel(p.ah, p.bh,
      [distance_cutoff](std::span<const std::uint8_t> a,
                         std::span<const std::uint8_t> b) -> double {
        return static_cast<double>(
            ::stride_align::indel::indel_distance_u8(a, b, distance_cutoff));
      },
      [distance_cutoff](const std::vector<lcs::Codepoint>& a,
                         const std::vector<lcs::Codepoint>& b) -> double {
        return static_cast<double>(::stride_align::indel::indel_distance<lcs::Codepoint>(
            std::span<const lcs::Codepoint>(a),
            std::span<const lcs::Codepoint>(b), distance_cutoff));
      }));
  return nb::cast(max_d - d);
}

inline double dispatch_Indel_normalized_similarity(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  // Push the float cutoff down to the kernel via an integer indel
  // cutoff. similarity = 1 - indel / total; sim >= sc iff
  // indel <= total * (1 - sc).
  const std::size_t total = length_of(p.ah) + length_of(p.bh);
  if (total == 0U) return 1.0;
  std::size_t distance_cutoff = ::stride_align::indel::kNoCutoff;
  if (!score_cutoff.is_none()) {
    const double sc = nb::cast<double>(score_cutoff);
    const double allow = static_cast<double>(total) * (1.0 - sc);
    if (allow < 0.0) return 0.0;
    distance_cutoff = static_cast<std::size_t>(allow);  // floor
  }
  const std::size_t d = static_cast<std::size_t>(run_kernel(p.ah, p.bh,
      [distance_cutoff](std::span<const std::uint8_t> a,
                         std::span<const std::uint8_t> b) -> double {
        return static_cast<double>(
            ::stride_align::indel::indel_distance_u8(a, b, distance_cutoff));
      },
      [distance_cutoff](const std::vector<lcs::Codepoint>& a,
                         const std::vector<lcs::Codepoint>& b) -> double {
        return static_cast<double>(::stride_align::indel::indel_distance<lcs::Codepoint>(
            std::span<const lcs::Codepoint>(a),
            std::span<const lcs::Codepoint>(b), distance_cutoff));
      }));
  const double sim = 1.0 - static_cast<double>(d) / static_cast<double>(total);
  if (!score_cutoff.is_none() && sim < nb::cast<double>(score_cutoff)) {
    return 0.0;
  }
  return sim;
}

inline double dispatch_Indel_normalized_distance(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  // normalized_distance = 1 - normalized_similarity. Translate cutoff.
  nb::object sim_cutoff;
  nb::handle sim_cutoff_h = nb::none();
  if (!score_cutoff.is_none()) {
    const double sc = nb::cast<double>(score_cutoff);
    sim_cutoff = nb::cast(1.0 - sc);
    sim_cutoff_h = sim_cutoff;
  }
  return 1.0 - dispatch_Indel_normalized_similarity(s1, s2, processor, sim_cutoff_h);
}

// ---- Levenshtein -----------------------------------------------------------
//
// Delegates to ``levenshtein::dispatch_score`` /
// ``levenshtein::dispatch_normalized_score`` (the same fast paths the
// public ``sa.levenshtein_score`` / ``sa.levenshtein_normalized_score``
// bindings use) rather than re-implementing the byte/codepoint split
// in lambdas.

inline nb::object dispatch_Levenshtein_distance(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff;
  if (!score_cutoff.is_none()) {
    cutoff = static_cast<std::size_t>(nb::cast<std::int64_t>(score_cutoff));
  }
  const std::size_t d = static_cast<std::size_t>(
      ::stride_align::levenshtein::dispatch_score(p.ah, p.bh, cutoff));
  return nb::cast(d);
}

inline nb::object dispatch_Levenshtein_similarity(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  const std::size_t la = length_of(p.ah);
  const std::size_t lb = length_of(p.bh);
  const std::size_t max_d = la > lb ? la : lb;
  std::size_t distance_cutoff = ::stride_align::levenshtein::kNoCutoff;
  if (!score_cutoff.is_none()) {
    const std::size_t sim_cutoff = static_cast<std::size_t>(
        nb::cast<std::int64_t>(score_cutoff));
    distance_cutoff = (max_d > sim_cutoff) ? (max_d - sim_cutoff) : 0U;
  }
  const std::size_t d = static_cast<std::size_t>(
      ::stride_align::levenshtein::dispatch_score(p.ah, p.bh, distance_cutoff));
  return nb::cast(max_d - d);
}

inline double dispatch_Levenshtein_normalized_similarity(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff;
  if (!score_cutoff.is_none()) {
    const std::size_t la = length_of(p.ah);
    const std::size_t lb = length_of(p.bh);
    const std::size_t max_d = la > lb ? la : lb;
    if (max_d == 0U) return 1.0;
    const double sc = nb::cast<double>(score_cutoff);
    const double allow = static_cast<double>(max_d) * (1.0 - sc);
    if (allow < 0.0) return 0.0;
    cutoff = static_cast<std::size_t>(allow);
  }
  const double sim =
      ::stride_align::levenshtein::dispatch_normalized_score(p.ah, p.bh, cutoff);
  if (!score_cutoff.is_none() && sim < nb::cast<double>(score_cutoff)) {
    return 0.0;
  }
  return sim;
}

inline double dispatch_Levenshtein_normalized_distance(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  nb::object sim_cutoff;
  nb::handle sim_cutoff_h = nb::none();
  if (!score_cutoff.is_none()) {
    const double sc = nb::cast<double>(score_cutoff);
    sim_cutoff = nb::cast(1.0 - sc);
    sim_cutoff_h = sim_cutoff;
  }
  return 1.0 - dispatch_Levenshtein_normalized_similarity(s1, s2, processor, sim_cutoff_h);
}

}  // namespace stride_align::dist_shim
