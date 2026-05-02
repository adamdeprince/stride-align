#include "backends/x86_avx2.hpp"
#include "x86_microbench_kernels.hpp"

namespace stride_align::microbench {

#if defined(__GNUC__) || defined(__clang__)
#define STRIDE_ALIGN_AVX2_TARGET __attribute__((target("avx2")))
#else
#define STRIDE_ALIGN_AVX2_TARGET
#endif

bool supports_avx2() noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") != 0;
#else
  return true;
#endif
}

STRIDE_ALIGN_AVX2_TARGET RunResult run_avx2_backend(
    const PreparedWorkload& prepared,
    const Options& options) {
  if (options.shape == "1:many") {
    return run_prepared_batch<stride_align::backend_avx2::SimdOps>(prepared.batch, options);
  }
  return run_prepared_single<stride_align::backend_avx2::SimdOps>(prepared.single, options);
}

#undef STRIDE_ALIGN_AVX2_TARGET

}  // namespace stride_align::microbench
