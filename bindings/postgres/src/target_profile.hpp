#pragma once

#include <string_view>

#if (defined(STRIDE_ALIGN_POSTGRES_TARGET_GENERIC) + \
     defined(STRIDE_ALIGN_POSTGRES_TARGET_NATIVE) + \
     defined(STRIDE_ALIGN_POSTGRES_TARGET_AVX2) + \
     defined(STRIDE_ALIGN_POSTGRES_TARGET_AVX512BWVL) + \
     defined(STRIDE_ALIGN_POSTGRES_TARGET_NEON) + \
     defined(STRIDE_ALIGN_POSTGRES_TARGET_POWER8_VSX) + \
     defined(STRIDE_ALIGN_POSTGRES_TARGET_LSX) + \
     defined(STRIDE_ALIGN_POSTGRES_TARGET_LASX)) != 1
#error "Exactly one PostgreSQL package target must be selected"
#endif

#if defined(STRIDE_ALIGN_POSTGRES_TARGET_GENERIC) && \
    (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#error "The generic x86 PostgreSQL profile unexpectedly enables AVX"
#endif

#if defined(STRIDE_ALIGN_POSTGRES_TARGET_AVX2) && \
    (!defined(__AVX2__) || defined(__AVX512F__))
#error "The AVX2 PostgreSQL profile must select AVX2 without AVX-512"
#endif

#if defined(STRIDE_ALIGN_POSTGRES_TARGET_AVX512BWVL) && \
    (!defined(__AVX512F__) || !defined(__AVX512BW__) || \
     !defined(__AVX512VL__) || !defined(__AVX512DQ__))
#error "The AVX-512 PostgreSQL profile requires F/BW/VL/DQ"
#endif

#if defined(STRIDE_ALIGN_POSTGRES_TARGET_NEON) && \
    !defined(__ARM_NEON) && !defined(__ARM_NEON__)
#error "The NEON PostgreSQL profile did not select NEON"
#endif

#if defined(STRIDE_ALIGN_POSTGRES_TARGET_POWER8_VSX) && \
    (!defined(__VSX__) || !defined(__POWER8_VECTOR__))
#error "The POWER8 PostgreSQL profile did not select POWER8 VSX"
#endif

#if defined(STRIDE_ALIGN_POSTGRES_TARGET_LSX) && \
    (!defined(__loongarch_sx) || defined(__loongarch_asx))
#error "The LSX PostgreSQL profile must select LSX without LASX"
#endif

#if defined(STRIDE_ALIGN_POSTGRES_TARGET_LASX) && \
    (!defined(__loongarch_sx) || !defined(__loongarch_asx))
#error "The LASX PostgreSQL profile did not select LASX"
#endif

#if defined(STRIDE_ALIGN_POSTGRES_TARGET_LA464)
static_assert(
    std::string_view(__loongarch_arch) == "la464" &&
        std::string_view(__loongarch_tune) == "la464",
    "The LA464 PostgreSQL profile has the wrong -march/-mtune target");
#endif

#if defined(STRIDE_ALIGN_POSTGRES_TARGET_LA664)
static_assert(
    std::string_view(__loongarch_arch) == "la664" &&
        std::string_view(__loongarch_tune) == "la664",
    "The LA664 PostgreSQL profile has the wrong -march/-mtune target");
#endif
