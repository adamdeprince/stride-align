#include "cpu_detect.hpp"

#include <string_view>
#include <vector>

#if defined(__linux__) && \
    (defined(__aarch64__) || defined(__loongarch__) || \
     defined(__powerpc64__) || defined(__ppc64__) || defined(__riscv))
#include <sys/auxv.h>
#endif

#if defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/prctl.h>
#endif

#if defined(__linux__) && defined(__loongarch__)
#include <asm/hwcap.h>
#endif

#if defined(__linux__) && \
    (defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__))
#include <asm/cputable.h>
#endif

#if defined(__linux__) && defined(__riscv)
#include <asm/hwcap.h>
#endif

namespace {

bool supports_sse41() noexcept {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  return __builtin_cpu_supports("sse4.1") != 0;
#else
  return false;
#endif
}

bool supports_x86_64_v3() noexcept {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") != 0 &&
      __builtin_cpu_supports("bmi") != 0 &&
      __builtin_cpu_supports("bmi2") != 0 &&
      __builtin_cpu_supports("f16c") != 0 &&
      __builtin_cpu_supports("fma") != 0 &&
      __builtin_cpu_supports("lzcnt") != 0 &&
      __builtin_cpu_supports("movbe") != 0;
#else
  return false;
#endif
}

bool supports_avx512bwvl() noexcept {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  return supports_x86_64_v3() &&
      __builtin_cpu_supports("avx512f") != 0 &&
      __builtin_cpu_supports("avx512bw") != 0 &&
      __builtin_cpu_supports("avx512cd") != 0 &&
      __builtin_cpu_supports("avx512vl") != 0 &&
      __builtin_cpu_supports("avx512dq") != 0;
#else
  return false;
#endif
}

bool supports_neon() noexcept {
#if defined(__APPLE__) && defined(__aarch64__)
  return true;
#elif defined(__linux__) && defined(__aarch64__)
  return (getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0;
#else
  return false;
#endif
}

bool supports_sve() noexcept {
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP_SVE) && \
    defined(PR_SVE_GET_VL) && defined(PR_SVE_VL_LEN_MASK)
  if ((getauxval(AT_HWCAP) & HWCAP_SVE) == 0) return false;
  const int vector_length = prctl(PR_SVE_GET_VL);
  return vector_length >= 0 &&
      (vector_length & PR_SVE_VL_LEN_MASK) == 16;
#else
  return false;
#endif
}

bool supports_sve2() noexcept {
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP2_SVE2)
  return supports_sve() && (getauxval(AT_HWCAP2) & HWCAP2_SVE2) != 0;
#else
  return false;
#endif
}

bool supports_lsx() noexcept {
#if defined(__linux__) && defined(__loongarch__)
  return (getauxval(AT_HWCAP) & HWCAP_LOONGARCH_LSX) != 0;
#else
  return false;
#endif
}

bool supports_lasx() noexcept {
#if defined(__linux__) && defined(__loongarch__)
  return (getauxval(AT_HWCAP) & HWCAP_LOONGARCH_LASX) != 0;
#else
  return false;
#endif
}

bool supports_vsx() noexcept {
#if defined(__linux__) && \
    (defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__))
  return (getauxval(AT_HWCAP) & PPC_FEATURE_HAS_VSX) != 0;
#else
  return false;
#endif
}

bool supports_rvv() noexcept {
#if defined(__linux__) && defined(__riscv) && defined(COMPAT_HWCAP_ISA_V)
  return (getauxval(AT_HWCAP) & COMPAT_HWCAP_ISA_V) != 0;
#else
  return false;
#endif
}

}  // namespace

SEXP stride_r_backend_candidates_impl() {
  std::vector<std::string_view> names;

#if defined(STRIDE_ALIGN_R_HAVE_AVX512BWVL)
  if (supports_avx512bwvl()) names.emplace_back("avx512bwvl");
#endif
#if defined(STRIDE_ALIGN_R_HAVE_AVX2)
  if (supports_x86_64_v3()) names.emplace_back("avx2");
#endif
#if defined(STRIDE_ALIGN_R_HAVE_SSE41)
  if (supports_sse41()) names.emplace_back("sse41");
#endif
#if defined(STRIDE_ALIGN_R_HAVE_SVE2)
  if (supports_sve2()) names.emplace_back("sve2");
#endif
#if defined(STRIDE_ALIGN_R_HAVE_SVE)
  if (supports_sve()) names.emplace_back("sve");
#endif
#if defined(STRIDE_ALIGN_R_HAVE_NEON)
  if (supports_neon()) names.emplace_back("neon");
#endif
#if defined(STRIDE_ALIGN_R_HAVE_LASX)
  if (supports_lasx()) names.emplace_back("lasx");
#endif
#if defined(STRIDE_ALIGN_R_HAVE_LSX)
  if (supports_lsx()) names.emplace_back("lsx");
#endif
#if defined(STRIDE_ALIGN_R_HAVE_VSX)
  if (supports_vsx()) names.emplace_back("vsx");
#endif
#if defined(STRIDE_ALIGN_R_HAVE_RVV)
  if (supports_rvv()) names.emplace_back("rvv");
#endif

  names.emplace_back("generic");
  SEXP result = PROTECT(Rf_allocVector(STRSXP, static_cast<R_xlen_t>(names.size())));
  for (R_xlen_t index = 0; index < static_cast<R_xlen_t>(names.size()); ++index) {
    const auto name = names[static_cast<std::size_t>(index)];
    SET_STRING_ELT(
        result,
        index,
        Rf_mkCharLenCE(name.data(), static_cast<int>(name.size()), CE_UTF8));
  }
  UNPROTECT(1);
  return result;
}
