#pragma once

// nanobind-facing dispatchers for the rapidfuzz-shim fuzz family
// (``stride_align.rapidfuzz.fuzz.ratio`` and friends). Each binding
// handles the rapidfuzz contract — optional ``processor`` callable
// applied to both inputs, then the kernel, then ``× 100``, then the
// ``score_cutoff`` clamp — entirely in C++ so the Python boundary is
// crossed exactly once per call instead of through a multi-step
// Python wrapper.
//
// All bindings share a byte fast path: when ``processor`` is
// ``None`` and both inputs satisfy ``byte_view::classify`` (ASCII
// ``str`` or ``bytes``-like), the raw bytes flow straight into the
// per-engine ``*_bytes`` entry — no allocation, no codepoint round-
// trip. When ``processor`` is set we let it transform the inputs
// (it returns Python ``str``) and then re-classify; the resulting
// ``str`` is normally still ASCII so the fast path still applies.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include <nanobind/nanobind.h>

#include "byte_view.hpp"
#include "lcs_dispatch.hpp"  // widen_to_codepoints
#include "stride_align/indel.hpp"
#include "stride_align/partial_ratio.hpp"
#include "stride_align/token_ratios.hpp"
#include "stride_align/wratio.hpp"

namespace stride_align::fuzz_shim {

namespace nb = nanobind;

// Score-cutoff clamp: rapidfuzz convention returns ``0.0`` when the
// score is strictly below the cutoff, unchanged otherwise.
inline double clamp_score(double score, nb::handle score_cutoff) {
  if (score_cutoff.is_none()) return score;
  const double cutoff = nb::cast<double>(score_cutoff);
  return score < cutoff ? 0.0 : score;
}

// Run an arbitrary kernel against either byte spans (fast path) or
// codepoint vectors (fallback), depending on the input type. The
// caller passes two lambdas: one for the byte path, one for the
// codepoint path. The byte path is taken when ``a`` and ``b`` are
// both ASCII Python ``str`` or ``bytes``-like.
template <typename ByteFn, typename CodepointFn>
inline double run_kernel(
    nb::handle a,
    nb::handle b,
    ByteFn byte_fn,
    CodepointFn codepoint_fn) {
  namespace bv = ::stride_align::byte_view;
  const bv::ByteCompatKind ka = bv::classify(a.ptr());
  const bv::ByteCompatKind kb = bv::classify(b.ptr());
  if (ka != bv::ByteCompatKind::None && kb != bv::ByteCompatKind::None) {
    const std::uint8_t* ap = nullptr; std::size_t alen = 0;
    const std::uint8_t* bp = nullptr; std::size_t blen = 0;
    bv::view(a.ptr(), ka, ap, alen);
    bv::view(b.ptr(), kb, bp, blen);
    return byte_fn(std::span<const std::uint8_t>(ap, alen),
                    std::span<const std::uint8_t>(bp, blen));
  }
  return codepoint_fn(
      ::stride_align::lcs::widen_to_codepoints(a),
      ::stride_align::lcs::widen_to_codepoints(b));
}

// Common preprocessing: apply optional processor, return the prepared
// (a, b) handles. Empty-input handling is left to each engine — they
// each have their own convention (``ratio("", "")`` is 1.0,
// ``token_set_ratio("", "")`` is 0.0, etc.) so a uniform short-
// circuit here would diverge from rapidfuzz on the edge cases.
struct Prepped {
  nb::object holder_a;  // owns transformed value if processor was applied
  nb::object holder_b;
  nb::handle ah;        // either s1 or holder_a
  nb::handle bh;
};

inline Prepped prepare(
    nb::handle s1,
    nb::handle s2,
    nb::handle processor) {
  Prepped p;
  if (processor.is_none()) {
    p.ah = s1;
    p.bh = s2;
  } else {
    p.holder_a = nb::steal(PyObject_CallFunctionObjArgs(processor.ptr(), s1.ptr(), nullptr));
    if (!p.holder_a.is_valid()) throw nb::python_error();
    p.holder_b = nb::steal(PyObject_CallFunctionObjArgs(processor.ptr(), s2.ptr(), nullptr));
    if (!p.holder_b.is_valid()) throw nb::python_error();
    p.ah = p.holder_a;
    p.bh = p.holder_b;
  }
  return p;
}

// ---- ratio -----------------------------------------------------------------

inline double dispatch_ratio(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  // Empty-case handling is in each engine.

  // Indel-normalised similarity, scaled to 0..100. score_cutoff is
  // pushed into the kernel via an integer indel cutoff.
  std::size_t indel_cutoff = ::stride_align::indel::kNoCutoff;
  if (!score_cutoff.is_none()) {
    const double sc = nb::cast<double>(score_cutoff);
    // similarity = 1 - indel / total; sim >= sc/100 iff indel <= total * (1 - sc/100).
    // Need total to compute — defer to inside the kernel call by leaving cutoff alone.
    (void)sc;
  }
  const double sim = run_kernel(p.ah, p.bh,
      [&](std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
        const std::size_t total = a.size() + b.size();
        if (total == 0U) return 1.0;
        const std::size_t d =
            ::stride_align::indel::indel_distance_u8(a, b, indel_cutoff);
        return 1.0 - static_cast<double>(d) / static_cast<double>(total);
      },
      [&](const std::vector<lcs::Codepoint>& a,
          const std::vector<lcs::Codepoint>& b) {
        const std::size_t total = a.size() + b.size();
        if (total == 0U) return 1.0;
        const std::size_t d = ::stride_align::indel::indel_distance<lcs::Codepoint>(
            std::span<const lcs::Codepoint>(a),
            std::span<const lcs::Codepoint>(b), indel_cutoff);
        return 1.0 - static_cast<double>(d) / static_cast<double>(total);
      });
  return clamp_score(sim * 100.0, score_cutoff);
}

// ---- partial_ratio ---------------------------------------------------------

inline double dispatch_partial_ratio(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  // Empty-case handling is in each engine.
  const double sim = run_kernel(p.ah, p.bh,
      [](std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
        return ::stride_align::partial_ratio::partial_ratio_bytes(a, b);
      },
      [](const std::vector<lcs::Codepoint>& a,
         const std::vector<lcs::Codepoint>& b) {
        return ::stride_align::partial_ratio::partial_ratio(a, b);
      });
  return clamp_score(sim * 100.0, score_cutoff);
}

// ---- token_sort_ratio ------------------------------------------------------

inline double dispatch_token_sort_ratio(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  // token_*_ratio engines themselves handle empty cases per rapidfuzz convention.
  const double sim = run_kernel(p.ah, p.bh,
      [](std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
        return ::stride_align::token_ratios::token_sort_ratio_bytes(a, b);
      },
      [](const std::vector<lcs::Codepoint>& a,
         const std::vector<lcs::Codepoint>& b) {
        return ::stride_align::token_ratios::token_sort_ratio(a, b);
      });
  return clamp_score(sim * 100.0, score_cutoff);
}

// ---- token_set_ratio -------------------------------------------------------

inline double dispatch_token_set_ratio(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  const double sim = run_kernel(p.ah, p.bh,
      [](std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
        return ::stride_align::token_ratios::token_set_ratio_bytes(a, b);
      },
      [](const std::vector<lcs::Codepoint>& a,
         const std::vector<lcs::Codepoint>& b) {
        return ::stride_align::token_ratios::token_set_ratio(a, b);
      });
  return clamp_score(sim * 100.0, score_cutoff);
}

// ---- partial_token_sort_ratio ----------------------------------------------

inline double dispatch_partial_token_sort_ratio(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  // Empty-case handling is in each engine.
  const double sim = run_kernel(p.ah, p.bh,
      [](std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
        return ::stride_align::wratio::partial_token_sort_ratio_bytes(a, b);
      },
      [](const std::vector<lcs::Codepoint>& a,
         const std::vector<lcs::Codepoint>& b) {
        return ::stride_align::wratio::partial_token_sort_ratio_engine<lcs::Codepoint>(
            std::span<const lcs::Codepoint>(a),
            std::span<const lcs::Codepoint>(b));
      });
  return clamp_score(sim * 100.0, score_cutoff);
}

// ---- partial_token_set_ratio -----------------------------------------------

inline double dispatch_partial_token_set_ratio(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  // Empty-case handling is in each engine.
  const double sim = run_kernel(p.ah, p.bh,
      [](std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
        return ::stride_align::wratio::partial_token_set_ratio_bytes(a, b);
      },
      [](const std::vector<lcs::Codepoint>& a,
         const std::vector<lcs::Codepoint>& b) {
        return ::stride_align::wratio::partial_token_set_ratio_engine<lcs::Codepoint>(
            std::span<const lcs::Codepoint>(a),
            std::span<const lcs::Codepoint>(b));
      });
  return clamp_score(sim * 100.0, score_cutoff);
}

// ---- token_ratio = max(token_sort, token_set) ------------------------------

inline double dispatch_token_ratio(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  double sim;
  sim = run_kernel(p.ah, p.bh,
      [](std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
        const double ts = ::stride_align::token_ratios::token_sort_ratio_bytes(a, b);
        const double tx = ::stride_align::token_ratios::token_set_ratio_bytes(a, b);
        return ts > tx ? ts : tx;
      },
      [](const std::vector<lcs::Codepoint>& a,
         const std::vector<lcs::Codepoint>& b) {
        const double ts = ::stride_align::token_ratios::token_sort_ratio(a, b);
        const double tx = ::stride_align::token_ratios::token_set_ratio(a, b);
        return ts > tx ? ts : tx;
      });
  return clamp_score(sim * 100.0, score_cutoff);
}

// ---- partial_token_ratio = max(partial_token_sort, partial_token_set) ------

inline double dispatch_partial_token_ratio(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  // Empty-case handling is in each engine.
  const double sim = run_kernel(p.ah, p.bh,
      [](std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
        const double pts = ::stride_align::wratio::partial_token_sort_ratio_bytes(a, b);
        const double pxs = ::stride_align::wratio::partial_token_set_ratio_bytes(a, b);
        return pts > pxs ? pts : pxs;
      },
      [](const std::vector<lcs::Codepoint>& a,
         const std::vector<lcs::Codepoint>& b) {
        const double pts = ::stride_align::wratio::partial_token_sort_ratio_engine<lcs::Codepoint>(
            std::span<const lcs::Codepoint>(a),
            std::span<const lcs::Codepoint>(b));
        const double pxs = ::stride_align::wratio::partial_token_set_ratio_engine<lcs::Codepoint>(
            std::span<const lcs::Codepoint>(a),
            std::span<const lcs::Codepoint>(b));
        return pts > pxs ? pts : pxs;
      });
  return clamp_score(sim * 100.0, score_cutoff);
}

// ---- WRatio ----------------------------------------------------------------

inline double dispatch_wratio(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  // Empty-case handling is in each engine.
  // The WRatio kernel takes its score_cutoff in normalised form to
  // drive its internal short-circuit. Pass it through if set.
  double cutoff_norm = 0.0;
  if (!score_cutoff.is_none()) {
    cutoff_norm = nb::cast<double>(score_cutoff) / 100.0;
  }
  const double sim = run_kernel(p.ah, p.bh,
      [cutoff_norm](std::span<const std::uint8_t> a,
                     std::span<const std::uint8_t> b) {
        return ::stride_align::wratio::wratio_bytes(a, b, cutoff_norm);
      },
      [cutoff_norm](const std::vector<lcs::Codepoint>& a,
                     const std::vector<lcs::Codepoint>& b) {
        return ::stride_align::wratio::wratio(a, b, cutoff_norm);
      });
  return clamp_score(sim * 100.0, score_cutoff);
}

// ---- QRatio ---------------------------------------------------------------
//
// Since rapidfuzz 3.0 QRatio is algebraically identical to ratio
// EXCEPT on empty inputs: rapidfuzz's QRatio returns 0.0 for the
// empty-empty case (and any-empty), while ratio returns 100.0 for
// empty-empty. We handle the divergence here.
inline double dispatch_qratio(
    nb::handle s1, nb::handle s2,
    nb::handle processor, nb::handle score_cutoff) {
  const Prepped p = prepare(s1, s2, processor);
  auto is_empty = [](nb::handle h) -> bool {
    if (PyUnicode_Check(h.ptr())) return PyUnicode_GET_LENGTH(h.ptr()) == 0;
    if (PyBytes_Check(h.ptr()))  return PyBytes_GET_SIZE(h.ptr()) == 0;
    const Py_ssize_t n = PyObject_Length(h.ptr());
    if (n < 0) throw nb::python_error();
    return n == 0;
  };
  if (is_empty(p.ah) || is_empty(p.bh)) {
    return clamp_score(0.0, score_cutoff);
  }
  return dispatch_ratio(s1, s2, processor, score_cutoff);
}

}  // namespace stride_align::fuzz_shim
