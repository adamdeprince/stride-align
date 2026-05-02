#include "backends/x86_avx512bwvl.hpp"
#include "x86_microbench_kernels.hpp"

namespace stride_align::microbench {

#if defined(__GNUC__) || defined(__clang__)
#define STRIDE_ALIGN_AVX512BWVL_TARGET __attribute__((target("avx512f,avx512bw,avx512vl")))
#else
#define STRIDE_ALIGN_AVX512BWVL_TARGET
#endif

bool supports_avx512bwvl() noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") != 0 &&
      __builtin_cpu_supports("avx512bw") != 0 &&
      __builtin_cpu_supports("avx512vl") != 0;
#else
  return true;
#endif
}

STRIDE_ALIGN_AVX512BWVL_TARGET RunResult run_avx512bwvl_backend(
    const PreparedWorkload& prepared,
    const Options& options) {
  return run_backend_variant<stride_align::backend_avx512bwvl::SimdOps>(prepared, options);
}

#undef STRIDE_ALIGN_AVX512BWVL_TARGET

}  // namespace stride_align::microbench
