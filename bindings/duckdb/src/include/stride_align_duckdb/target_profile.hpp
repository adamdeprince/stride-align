#pragma once

#include <string_view>

// Package profiles are selected at configure time and specialized by the
// compiler's built-in ISA macros. Keep these checks close to the extension so
// a mislabeled artifact fails during compilation instead of at load time.

#if (defined(STRIDE_ALIGN_DUCKDB_TARGET_GENERIC) + \
     defined(STRIDE_ALIGN_DUCKDB_TARGET_NATIVE) + \
     defined(STRIDE_ALIGN_DUCKDB_TARGET_AVX2) + \
     defined(STRIDE_ALIGN_DUCKDB_TARGET_AVX512BWVL) + \
     defined(STRIDE_ALIGN_DUCKDB_TARGET_NEON) + \
     defined(STRIDE_ALIGN_DUCKDB_TARGET_LSX) + \
     defined(STRIDE_ALIGN_DUCKDB_TARGET_LASX)) != 1
#error "Exactly one DuckDB package target must be selected"
#endif

#if defined(STRIDE_ALIGN_DUCKDB_TARGET_GENERIC) && \
    (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#error "The generic x86 DuckDB profile unexpectedly enables AVX"
#endif

#if defined(STRIDE_ALIGN_DUCKDB_TARGET_GENERIC) && \
    defined(__loongarch__) && \
    (defined(__loongarch_sx) || defined(__loongarch_asx))
#error "The generic LoongArch DuckDB profile unexpectedly enables LSX/LASX"
#endif

#if defined(STRIDE_ALIGN_DUCKDB_TARGET_AVX2) && !defined(__AVX2__)
#error "The AVX2 DuckDB profile did not select the AVX2 templates"
#endif

#if defined(STRIDE_ALIGN_DUCKDB_TARGET_AVX2) && defined(__AVX512F__)
#error "The AVX2 DuckDB profile unexpectedly enables AVX-512"
#endif

#if defined(STRIDE_ALIGN_DUCKDB_TARGET_AVX512BWVL) && \
    (!defined(__AVX512F__) || !defined(__AVX512BW__) || \
     !defined(__AVX512VL__) || !defined(__AVX512DQ__))
#error "The AVX-512 DuckDB profile did not select the F/BW/VL/DQ templates"
#endif

#if defined(STRIDE_ALIGN_DUCKDB_TARGET_NEON) && \
    !defined(__ARM_NEON) && !defined(__ARM_NEON__)
#error "The NEON DuckDB profile did not select the NEON templates"
#endif

#if defined(STRIDE_ALIGN_DUCKDB_TARGET_LSX) && \
    (!defined(__loongarch_sx) || defined(__loongarch_asx))
#error "The LSX DuckDB profile must select LSX without LASX"
#endif

#if defined(STRIDE_ALIGN_DUCKDB_TARGET_LASX) && \
    (!defined(__loongarch_sx) || !defined(__loongarch_asx))
#error "The LASX DuckDB profile did not select the LASX templates"
#endif

#if defined(STRIDE_ALIGN_DUCKDB_TARGET_LA464)
static_assert(
    std::string_view(__loongarch_arch) == "la464" &&
        std::string_view(__loongarch_tune) == "la464",
    "The LA464 DuckDB profile has the wrong -march/-mtune target");
#endif

#if defined(STRIDE_ALIGN_DUCKDB_TARGET_LA664)
static_assert(
    std::string_view(__loongarch_arch) == "la664" &&
        std::string_view(__loongarch_tune) == "la664",
    "The LA664 DuckDB profile has the wrong -march/-mtune target");
#endif
