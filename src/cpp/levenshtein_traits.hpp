#pragma once

// Compile-time mapping BackendKind -> SIMD ops bundle for Levenshtein.
//
// This is the single file where ISA feature macros (__SSE4_1__, __AVX2__,
// __AVX512F__) appear: each #ifdef gates exactly one explicit template
// specialization. The bindings consume the trait via the primary template
// (which holds `Ops = void`) and any specializations that survive the
// preprocessor pass; no #ifdef survives at the call site.

#include <type_traits>

#include "cpu.hpp"
#include "levenshtein_simd_ops.hpp"

namespace stride_align::levenshtein_simd {

template <BackendKind Kind, typename = void>
struct OpsFor {
  using type = void;  // no SIMD specialization -> caller falls back to scalar
};

#if defined(__SSE4_1__)
template <>
struct OpsFor<BackendKind::x86_sse41> {
  using type = SseOps;
};
#endif

#if defined(__AVX2__)
template <>
struct OpsFor<BackendKind::x86_avx2> {
  using type = Avx2Ops;
};
template <>
struct OpsFor<BackendKind::x86_avx10_256> {
  using type = Avx2Ops;
};
#endif

#if defined(__AVX512F__)
template <>
struct OpsFor<BackendKind::x86_avx512bwvl> {
  using type = Avx512Ops;
};
template <>
struct OpsFor<BackendKind::x86_avx10_512> {
  using type = Avx512Ops;
};
#endif

template <BackendKind Kind>
using OpsFor_t = typename OpsFor<Kind>::type;

}  // namespace stride_align::levenshtein_simd
