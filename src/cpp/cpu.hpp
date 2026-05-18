#pragma once

#include <string_view>
#include <vector>

namespace stride_align {

enum class BackendKind {
  generic,
  swar,
  x86_sse41,
  x86_avx2,
  x86_avx512bwvl,
  x86_avx10_256,
  x86_avx10_512,
  linux_aarch64_neon,
  linux_aarch64_sve,
  linux_aarch64_sve2,
  macos_arm64_neon,
  linux_loongarch64_lsx,
  linux_loongarch64_lasx,
  linux_powerpc64_vsx,
  linux_riscv64_rvv,
};

struct BackendRecord {
  BackendKind kind;
  const char* name;
  bool compiled;
  bool available;
};

std::string_view backend_name(BackendKind kind) noexcept;
bool backend_is_compiled(BackendKind kind) noexcept;
bool backend_is_available(BackendKind kind) noexcept;
BackendKind detect_best_backend() noexcept;
std::vector<BackendRecord> available_backends();

}  // namespace stride_align
