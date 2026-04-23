#include "cpu.hpp"

#include <array>

namespace stride_align {

namespace {

struct BackendMetadata {
  BackendKind kind;
  const char* name;
};

constexpr std::array kBackends = {
    BackendMetadata{BackendKind::generic, "generic"},
    BackendMetadata{BackendKind::x86_sse41, "x86_sse41"},
    BackendMetadata{BackendKind::x86_avx2, "x86_avx2"},
    BackendMetadata{BackendKind::x86_avx512bwvl, "x86_avx512bwvl"},
    BackendMetadata{BackendKind::x86_avx10_256, "x86_avx10_256"},
    BackendMetadata{BackendKind::x86_avx10_512, "x86_avx10_512"},
    BackendMetadata{BackendKind::linux_aarch64_asimd, "linux_aarch64_asimd"},
    BackendMetadata{BackendKind::linux_aarch64_neon, "linux_aarch64_neon"},
    BackendMetadata{BackendKind::linux_aarch64_sve, "linux_aarch64_sve"},
    BackendMetadata{BackendKind::linux_aarch64_sve2, "linux_aarch64_sve2"},
    BackendMetadata{BackendKind::macos_arm64_neon, "macos_arm64_neon"},
    BackendMetadata{BackendKind::linux_loongarch64_lsx, "linux_loongarch64_lsx"},
    BackendMetadata{BackendKind::linux_loongarch64_lasx, "linux_loongarch64_lasx"},
    BackendMetadata{BackendKind::linux_powerpc64_vsx, "linux_powerpc64_vsx"},
    BackendMetadata{BackendKind::linux_riscv64_rvv, "linux_riscv64_rvv"},
};

bool x86_supports_sse41() noexcept {
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)) && \
    (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  return __builtin_cpu_supports("sse4.1") != 0;
#else
  return false;
#endif
}

bool x86_supports_avx2() noexcept {
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)) && \
    (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") != 0;
#else
  return false;
#endif
}

bool x86_supports_avx512bwvl() noexcept {
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)) && \
    (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") != 0 && __builtin_cpu_supports("avx512bw") != 0 &&
      __builtin_cpu_supports("avx512vl") != 0;
#else
  return false;
#endif
}

bool x86_supports_avx10_256() noexcept {
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)) && \
    defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 14)
  __builtin_cpu_init();
  return x86_supports_avx512bwvl() != 0 && __builtin_cpu_supports("avx10.1-256") != 0;
#else
  return false;
#endif
}

bool x86_supports_avx10_512() noexcept {
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)) && \
    defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 14)
  __builtin_cpu_init();
  return x86_supports_avx512bwvl() != 0 && __builtin_cpu_supports("avx10.1-512") != 0;
#else
  return false;
#endif
}

}  // namespace

std::string_view backend_name(BackendKind kind) noexcept {
  for (const auto& backend : kBackends) {
    if (backend.kind == kind) {
      return backend.name;
    }
  }

  return "unknown";
}

bool backend_is_compiled(BackendKind kind) noexcept {
  switch (kind) {
    case BackendKind::generic:
      return true;
    case BackendKind::x86_sse41:
#ifdef STRIDE_ALIGN_HAVE_X86_SSE41
      return true;
#else
      return false;
#endif
    case BackendKind::x86_avx2:
#ifdef STRIDE_ALIGN_HAVE_X86_AVX2
      return true;
#else
      return false;
#endif
    case BackendKind::x86_avx512bwvl:
#ifdef STRIDE_ALIGN_HAVE_X86_AVX512BWVL
      return true;
#else
      return false;
#endif
    case BackendKind::x86_avx10_256:
#ifdef STRIDE_ALIGN_HAVE_X86_AVX10_256
      return true;
#else
      return false;
#endif
    case BackendKind::x86_avx10_512:
#ifdef STRIDE_ALIGN_HAVE_X86_AVX10_512
      return true;
#else
      return false;
#endif
    case BackendKind::linux_aarch64_asimd:
#ifdef STRIDE_ALIGN_HAVE_LINUX_AARCH64_ASIMD
      return true;
#else
      return false;
#endif
    case BackendKind::linux_aarch64_neon:
#ifdef STRIDE_ALIGN_HAVE_LINUX_AARCH64_NEON
      return true;
#else
      return false;
#endif
    case BackendKind::linux_aarch64_sve:
#ifdef STRIDE_ALIGN_HAVE_LINUX_AARCH64_SVE
      return true;
#else
      return false;
#endif
    case BackendKind::linux_aarch64_sve2:
#ifdef STRIDE_ALIGN_HAVE_LINUX_AARCH64_SVE2
      return true;
#else
      return false;
#endif
    case BackendKind::macos_arm64_neon:
#ifdef STRIDE_ALIGN_HAVE_MACOS_ARM64_NEON
      return true;
#else
      return false;
#endif
    case BackendKind::linux_loongarch64_lsx:
#ifdef STRIDE_ALIGN_HAVE_LINUX_LOONGARCH64_LSX
      return true;
#else
      return false;
#endif
    case BackendKind::linux_loongarch64_lasx:
#ifdef STRIDE_ALIGN_HAVE_LINUX_LOONGARCH64_LASX
      return true;
#else
      return false;
#endif
    case BackendKind::linux_powerpc64_vsx:
#ifdef STRIDE_ALIGN_HAVE_LINUX_POWERPC64_VSX
      return true;
#else
      return false;
#endif
    case BackendKind::linux_riscv64_rvv:
#ifdef STRIDE_ALIGN_HAVE_LINUX_RISCV64_RVV
      return true;
#else
      return false;
#endif
  }

  return false;
}

bool backend_is_available(BackendKind kind) noexcept {
  if (!backend_is_compiled(kind)) {
    return false;
  }

  switch (kind) {
    case BackendKind::generic:
      return true;
    case BackendKind::x86_sse41:
      return x86_supports_sse41();
    case BackendKind::x86_avx2:
      return x86_supports_avx2();
    case BackendKind::x86_avx512bwvl:
      return x86_supports_avx512bwvl();
    case BackendKind::x86_avx10_256:
      return x86_supports_avx10_256();
    case BackendKind::x86_avx10_512:
      return x86_supports_avx10_512();
    case BackendKind::linux_aarch64_asimd:
    case BackendKind::linux_aarch64_neon:
    case BackendKind::linux_aarch64_sve:
    case BackendKind::linux_aarch64_sve2:
    case BackendKind::macos_arm64_neon:
    case BackendKind::linux_loongarch64_lsx:
    case BackendKind::linux_loongarch64_lasx:
    case BackendKind::linux_powerpc64_vsx:
    case BackendKind::linux_riscv64_rvv:
      return true;
  }

  return false;
}

BackendKind detect_best_backend() noexcept {
  constexpr std::array priority = {
      BackendKind::x86_avx10_512,
      BackendKind::x86_avx10_256,
      BackendKind::x86_avx512bwvl,
      BackendKind::x86_avx2,
      BackendKind::x86_sse41,
      BackendKind::linux_aarch64_sve2,
      BackendKind::linux_aarch64_sve,
      BackendKind::linux_aarch64_neon,
      BackendKind::linux_aarch64_asimd,
      BackendKind::macos_arm64_neon,
      BackendKind::linux_loongarch64_lasx,
      BackendKind::linux_loongarch64_lsx,
      BackendKind::linux_powerpc64_vsx,
      BackendKind::linux_riscv64_rvv,
      BackendKind::generic,
  };

  for (const auto kind : priority) {
    if (backend_is_available(kind)) {
      return kind;
    }
  }

  return BackendKind::generic;
}

std::vector<BackendRecord> available_backends() {
  std::vector<BackendRecord> records;
  records.reserve(kBackends.size());

  for (const auto& backend : kBackends) {
    records.push_back(BackendRecord{
        .kind = backend.kind,
        .name = backend.name,
        .compiled = backend_is_compiled(backend.kind),
        .available = backend_is_available(backend.kind),
    });
  }

  return records;
}

}  // namespace stride_align
