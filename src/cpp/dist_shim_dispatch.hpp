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
#include "jaro_dispatch.hpp"          // jaro::dispatch_similarity / winkler
#include "lcs_dispatch.hpp"           // lcs::dispatch_lcs_length
#include "levenshtein_dispatch.hpp"   // dispatch_score / dispatch_normalized_score
#include "stride_align/hamming.hpp"
#include "stride_align/indel.hpp"
#include "stride_align/jaro.hpp"
#include "stride_align/levenshtein.hpp"
#include "stride_align/lcs.hpp"
#include "fuzz_shim_dispatch.hpp"     // prepare(), run_kernel()

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

// ---- Hamming --------------------------------------------------------------
//
// rapidfuzz convention: with ``pad=True`` (default) the longer side
// counts its overflow as mismatches; ``pad=False`` raises on unequal
// length. The C++ side does the padded form unconditionally — the
// pad=False guard stays in the Python shim because it's a precondition
// check, not a hot-path computation.

inline nb::object dispatch_Hamming_distance(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  (void)score_cutoff;  // Hamming has no kernel-level cutoff yet
  const Prepped p = prepare(s1, s2, processor);
  const std::size_t la = length_of(p.ah);
  const std::size_t lb = length_of(p.bh);
  const std::size_t common = la < lb ? la : lb;
  std::size_t mismatches = 0;
  if (common > 0) {
    const double m = run_kernel(p.ah, p.bh,
        [common](std::span<const std::uint8_t> a,
                  std::span<const std::uint8_t> b) -> double {
          return static_cast<double>(
              ::stride_align::hamming::hamming_scalar<std::uint8_t>(
                  a.first(common), b.first(common)));
        },
        [common](const std::vector<lcs::Codepoint>& a,
                  const std::vector<lcs::Codepoint>& b) -> double {
          return static_cast<double>(
              ::stride_align::hamming::hamming_scalar<lcs::Codepoint>(
                  std::span<const lcs::Codepoint>(a.data(), common),
                  std::span<const lcs::Codepoint>(b.data(), common)));
        });
    mismatches = static_cast<std::size_t>(m);
  }
  return nb::cast(mismatches + (la > lb ? la - lb : lb - la));
}

inline nb::object dispatch_Hamming_similarity(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  const std::size_t la = length_of(p.ah);
  const std::size_t lb = length_of(p.bh);
  const std::size_t max_d = la > lb ? la : lb;
  nb::object d_obj = dispatch_Hamming_distance(s1, s2, processor, score_cutoff);
  const std::size_t d = nb::cast<std::size_t>(d_obj);
  return nb::cast(max_d - d);
}

inline double dispatch_Hamming_normalized_distance(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  (void)score_cutoff;
  const Prepped p = prepare(s1, s2, processor);
  const std::size_t la = length_of(p.ah);
  const std::size_t lb = length_of(p.bh);
  const std::size_t max_d = la > lb ? la : lb;
  if (max_d == 0U) return 0.0;
  const std::size_t d = nb::cast<std::size_t>(
      dispatch_Hamming_distance(s1, s2, processor, nb::none()));
  return static_cast<double>(d) / static_cast<double>(max_d);
}

inline double dispatch_Hamming_normalized_similarity(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  return 1.0 - dispatch_Hamming_normalized_distance(s1, s2, processor, score_cutoff);
}

// ---- Jaro -----------------------------------------------------------------

inline double dispatch_Jaro_similarity(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  const double sim = ::stride_align::jaro::dispatch_similarity(p.ah, p.bh);
  if (!score_cutoff.is_none() && sim < nb::cast<double>(score_cutoff)) return 0.0;
  return sim;
}

inline double dispatch_Jaro_distance(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  (void)score_cutoff;
  const Prepped p = prepare(s1, s2, processor);
  return 1.0 - ::stride_align::jaro::dispatch_similarity(p.ah, p.bh);
}

inline double dispatch_Jaro_normalized_similarity(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  return dispatch_Jaro_similarity(s1, s2, processor, score_cutoff);
}

inline double dispatch_Jaro_normalized_distance(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  return dispatch_Jaro_distance(s1, s2, processor, score_cutoff);
}

// ---- JaroWinkler ----------------------------------------------------------

inline double dispatch_JaroWinkler_similarity(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff,
    double prefix_weight) {
  const Prepped p = prepare(s1, s2, processor);
  const double sim = ::stride_align::jaro::dispatch_winkler_similarity(
      p.ah, p.bh, prefix_weight,
      ::stride_align::jaro::kDefaultPrefixThreshold,
      ::stride_align::jaro::kDefaultPrefixCap);
  if (!score_cutoff.is_none() && sim < nb::cast<double>(score_cutoff)) return 0.0;
  return sim;
}

inline double dispatch_JaroWinkler_distance(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff,
    double prefix_weight) {
  (void)score_cutoff;
  const Prepped p = prepare(s1, s2, processor);
  return 1.0 - ::stride_align::jaro::dispatch_winkler_similarity(
      p.ah, p.bh, prefix_weight,
      ::stride_align::jaro::kDefaultPrefixThreshold,
      ::stride_align::jaro::kDefaultPrefixCap);
}

inline double dispatch_JaroWinkler_normalized_similarity(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff,
    double prefix_weight) {
  return dispatch_JaroWinkler_similarity(s1, s2, processor, score_cutoff, prefix_weight);
}

inline double dispatch_JaroWinkler_normalized_distance(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff,
    double prefix_weight) {
  return dispatch_JaroWinkler_distance(s1, s2, processor, score_cutoff, prefix_weight);
}

// ---- OSA (restricted Damerau-Levenshtein) ---------------------------------
//
// In stride-align the kernel is named ``damerau_levenshtein_*`` for
// historical reasons (it's the OSA / restricted form); the unrestricted
// true Damerau-Levenshtein lives in ``true_damerau_levenshtein_*``.

inline nb::object dispatch_OSA_distance(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  (void)score_cutoff;
  const Prepped p = prepare(s1, s2, processor);
  const std::size_t d = static_cast<std::size_t>(run_kernel(p.ah, p.bh,
      [](std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) -> double {
        return static_cast<double>(
            ::stride_align::levenshtein::osa_distance_u8(a, b));
      },
      [](const std::vector<lcs::Codepoint>& a,
         const std::vector<lcs::Codepoint>& b) -> double {
        return static_cast<double>(::stride_align::levenshtein::osa_distance<lcs::Codepoint>(
            std::span<const lcs::Codepoint>(a),
            std::span<const lcs::Codepoint>(b)));
      }));
  return nb::cast(d);
}

inline nb::object dispatch_OSA_similarity(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  const std::size_t la = length_of(p.ah);
  const std::size_t lb = length_of(p.bh);
  const std::size_t max_d = la > lb ? la : lb;
  const std::size_t d = nb::cast<std::size_t>(
      dispatch_OSA_distance(s1, s2, processor, score_cutoff));
  return nb::cast(max_d - d);
}

inline double dispatch_OSA_normalized_distance(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  (void)score_cutoff;
  const Prepped p = prepare(s1, s2, processor);
  const std::size_t la = length_of(p.ah);
  const std::size_t lb = length_of(p.bh);
  const std::size_t max_d = la > lb ? la : lb;
  if (max_d == 0U) return 0.0;
  const std::size_t d = nb::cast<std::size_t>(
      dispatch_OSA_distance(s1, s2, processor, nb::none()));
  return static_cast<double>(d) / static_cast<double>(max_d);
}

inline double dispatch_OSA_normalized_similarity(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const double sim = 1.0 - dispatch_OSA_normalized_distance(s1, s2, processor, nb::none());
  if (!score_cutoff.is_none() && sim < nb::cast<double>(score_cutoff)) return 0.0;
  return sim;
}

// ---- DamerauLevenshtein (true, unrestricted) ------------------------------

inline nb::object dispatch_DamerauLevenshtein_distance(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  (void)score_cutoff;
  const Prepped p = prepare(s1, s2, processor);
  const std::size_t d = static_cast<std::size_t>(run_kernel(p.ah, p.bh,
      [](std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) -> double {
        return static_cast<double>(
            ::stride_align::levenshtein::true_damerau_levenshtein_distance_u8(a, b));
      },
      [](const std::vector<lcs::Codepoint>& a,
         const std::vector<lcs::Codepoint>& b) -> double {
        return static_cast<double>(::stride_align::levenshtein::true_damerau_levenshtein_distance<lcs::Codepoint>(
            std::span<const lcs::Codepoint>(a),
            std::span<const lcs::Codepoint>(b)));
      }));
  return nb::cast(d);
}

inline nb::object dispatch_DamerauLevenshtein_similarity(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  const std::size_t la = length_of(p.ah);
  const std::size_t lb = length_of(p.bh);
  const std::size_t max_d = la > lb ? la : lb;
  const std::size_t d = nb::cast<std::size_t>(
      dispatch_DamerauLevenshtein_distance(s1, s2, processor, score_cutoff));
  return nb::cast(max_d - d);
}

inline double dispatch_DamerauLevenshtein_normalized_distance(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  (void)score_cutoff;
  const Prepped p = prepare(s1, s2, processor);
  const std::size_t la = length_of(p.ah);
  const std::size_t lb = length_of(p.bh);
  const std::size_t max_d = la > lb ? la : lb;
  if (max_d == 0U) return 0.0;
  const std::size_t d = nb::cast<std::size_t>(
      dispatch_DamerauLevenshtein_distance(s1, s2, processor, nb::none()));
  return static_cast<double>(d) / static_cast<double>(max_d);
}

inline double dispatch_DamerauLevenshtein_normalized_similarity(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const double sim = 1.0 - dispatch_DamerauLevenshtein_normalized_distance(s1, s2, processor, nb::none());
  if (!score_cutoff.is_none() && sim < nb::cast<double>(score_cutoff)) return 0.0;
  return sim;
}

// ---- LCSseq ---------------------------------------------------------------
//
// Derived from the bit-parallel Indel kernel via the algebraic
// identity ``indel(a, b) = |a| + |b| - 2 * LCS(a, b)`` — i.e.
// ``LCS = (|a| + |b| - indel) / 2`` and
// ``LCSseq.distance = max(|a|, |b|) - LCS``. Routing through the
// already-fast Indel multi-word kernel sidesteps the previous LCS
// scalar-DP fallback, which was 33-61× slower than rapidfuzz on
// multi-word inputs.

inline std::size_t lcs_via_indel(nb::handle a, nb::handle b,
                                  std::size_t la, std::size_t lb) {
  const std::size_t indel = run_kernel(a, b,
      [](std::span<const std::uint8_t> ax, std::span<const std::uint8_t> bx) -> double {
        return static_cast<double>(
            ::stride_align::indel::indel_distance_u8(ax, bx));
      },
      [](const std::vector<lcs::Codepoint>& ax,
         const std::vector<lcs::Codepoint>& bx) -> double {
        return static_cast<double>(::stride_align::indel::indel_distance<lcs::Codepoint>(
            std::span<const lcs::Codepoint>(ax),
            std::span<const lcs::Codepoint>(bx)));
      });
  return (la + lb - indel) / 2U;
}

inline nb::object dispatch_LCSseq_similarity(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  (void)score_cutoff;
  const Prepped p = prepare(s1, s2, processor);
  const std::size_t lcs = lcs_via_indel(p.ah, p.bh,
      length_of(p.ah), length_of(p.bh));
  return nb::cast(lcs);
}

inline nb::object dispatch_LCSseq_distance(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  (void)score_cutoff;
  const Prepped p = prepare(s1, s2, processor);
  const std::size_t la = length_of(p.ah);
  const std::size_t lb = length_of(p.bh);
  const std::size_t max_d = la > lb ? la : lb;
  const std::size_t lcs = lcs_via_indel(p.ah, p.bh, la, lb);
  return nb::cast(max_d - lcs);
}

inline double dispatch_LCSseq_normalized_distance(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  (void)score_cutoff;
  const Prepped p = prepare(s1, s2, processor);
  const std::size_t la = length_of(p.ah);
  const std::size_t lb = length_of(p.bh);
  const std::size_t max_d = la > lb ? la : lb;
  if (max_d == 0U) return 0.0;
  const std::size_t lcs = lcs_via_indel(p.ah, p.bh, la, lb);
  return static_cast<double>(max_d - lcs) / static_cast<double>(max_d);
}

inline double dispatch_LCSseq_normalized_similarity(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const double sim = 1.0 - dispatch_LCSseq_normalized_distance(s1, s2, processor, nb::none());
  if (!score_cutoff.is_none() && sim < nb::cast<double>(score_cutoff)) return 0.0;
  return sim;
}

}  // namespace stride_align::dist_shim
