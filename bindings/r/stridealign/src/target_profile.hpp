#pragma once

// Each R backend is a separate DLL compiled with one explicit ISA target.
// Fail compilation if the profile name, compiler flags, and selected template
// specialization do not agree.

#if (defined(STRIDE_ALIGN_R_TARGET_GENERIC) + \
     defined(STRIDE_ALIGN_R_TARGET_SSE41) + \
     defined(STRIDE_ALIGN_R_TARGET_AVX2) + \
     defined(STRIDE_ALIGN_R_TARGET_AVX512BWVL) + \
     defined(STRIDE_ALIGN_R_TARGET_NEON) + \
     defined(STRIDE_ALIGN_R_TARGET_SVE) + \
     defined(STRIDE_ALIGN_R_TARGET_SVE2) + \
     defined(STRIDE_ALIGN_R_TARGET_LSX) + \
     defined(STRIDE_ALIGN_R_TARGET_LASX) + \
     defined(STRIDE_ALIGN_R_TARGET_VSX) + \
     defined(STRIDE_ALIGN_R_TARGET_RVV)) != 1
#error "Exactly one R backend target must be selected"
#endif

#if defined(STRIDE_ALIGN_R_TARGET_GENERIC) && \
    (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#error "The generic x86 R backend unexpectedly enables AVX"
#endif

#if defined(STRIDE_ALIGN_R_TARGET_GENERIC) && defined(__loongarch__) && \
    (defined(__loongarch_sx) || defined(__loongarch_asx))
#error "The generic LoongArch R backend unexpectedly enables LSX or LASX"
#endif

#if defined(STRIDE_ALIGN_R_TARGET_SSE41) && \
    (!defined(__SSE4_1__) || defined(__AVX__))
#error "The SSE4.1 R backend must select SSE4.1 without AVX"
#endif

#if defined(STRIDE_ALIGN_R_TARGET_AVX2) && \
    (!defined(__AVX2__) || !defined(__BMI__) || !defined(__BMI2__) || \
     !defined(__F16C__) || !defined(__FMA__) || !defined(__LZCNT__) || \
     !defined(__MOVBE__) || defined(__AVX512F__))
#error "The AVX2 R backend must select x86-64-v3 without AVX-512"
#endif

#if defined(STRIDE_ALIGN_R_TARGET_AVX512BWVL) && \
    (!defined(__AVX512F__) || !defined(__AVX512BW__) || \
     !defined(__AVX512CD__) || !defined(__AVX512VL__) || \
     !defined(__AVX512DQ__))
#error "The AVX-512 R backend requires the x86-64-v4 feature set"
#endif

#if defined(STRIDE_ALIGN_R_TARGET_NEON) && \
    !defined(__ARM_NEON) && !defined(__ARM_NEON__)
#error "The NEON R backend did not select NEON"
#endif

#if defined(STRIDE_ALIGN_R_TARGET_SVE) && \
    (!defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SVE2))
#error "The SVE R backend must select SVE without SVE2"
#endif

#if defined(STRIDE_ALIGN_R_TARGET_SVE2) && !defined(__ARM_FEATURE_SVE2)
#error "The SVE2 R backend did not select SVE2"
#endif

#if defined(STRIDE_ALIGN_R_TARGET_LSX) && \
    (!defined(__loongarch_sx) || defined(__loongarch_asx))
#error "The LSX R backend must select LSX without LASX"
#endif

#if defined(STRIDE_ALIGN_R_TARGET_LASX) && \
    (!defined(__loongarch_sx) || !defined(__loongarch_asx))
#error "The LASX R backend did not select LASX"
#endif

#if defined(STRIDE_ALIGN_R_TARGET_VSX) && !defined(__VSX__)
#error "The VSX R backend did not select VSX"
#endif

#if defined(STRIDE_ALIGN_R_TARGET_RVV) && !defined(__riscv_vector)
#error "The RVV R backend did not select the vector extension"
#endif
