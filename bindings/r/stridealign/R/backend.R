BackendKind <- list(
  GENERIC = "generic",
  SWAR = "swar",
  X86_SSE41 = "sse41",
  X86_AVX2 = "avx2",
  X86_AVX512BWVL = "avx512bwvl",
  X86_AVX10_256 = "avx10_256",
  X86_AVX10_512 = "avx10_512",
  LINUX_AARCH64_NEON = "neon",
  LINUX_AARCH64_SVE = "sve",
  LINUX_AARCH64_SVE2 = "sve2",
  MACOS_ARM64_NEON = "neon",
  LINUX_LOONGARCH64_LSX = "lsx",
  LINUX_LOONGARCH64_LASX = "lasx",
  LINUX_POWERPC64_VSX = "vsx",
  LINUX_RISCV64_RVV = "rvv"
)

BackendRecord <- function(kind, compiled = TRUE, available = TRUE) {
  name <- as.character(kind)
  structure(list(
    kind = name,
    name = name,
    compiled = isTRUE(compiled),
    available = isTRUE(available)
  ), class = "stride_backend_record")
}

available_backends <- function() {
  lapply(stride_available_backends(), BackendRecord)
}

backend_is_available <- function(kind) {
  as.character(kind) %in% stride_available_backends()
}

detect_best_backend <- function() stride_backend()
