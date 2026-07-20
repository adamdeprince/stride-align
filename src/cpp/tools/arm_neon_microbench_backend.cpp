#include "backends/arm_neon128.hpp"
#include "verify_dual_sw_exact_fill.hpp"
#include "x86_microbench_kernels.hpp"

#include <stdexcept>

namespace stride_align::microbench {

bool supports_avx2() noexcept {
  return false;
}

bool supports_avx512bwvl() noexcept {
  return false;
}

bool supports_parasail() noexcept {
  return false;
}

RunResult run_avx2_backend(const PreparedWorkload&, const Options&) {
  throw std::runtime_error("AVX2 microbench backend is not available in the arm64 build");
}

RunResult run_avx512bwvl_backend(const PreparedWorkload&, const Options&) {
  throw std::runtime_error("AVX512BWVL microbench backend is not available in the arm64 build");
}

int verify_dual_sw_exact_fill_avx512bwvl() {
  // On arm64 microbench builds, --verify-dual runs the NEON harness instead.
  return verify_dual_detail::run_verify_dual_cases<
      stride_align::arm_neon128_backend::SimdOps>("neon");
}

RunResult run_parasail_backend(const PreparedWorkload&, const Options&) {
  throw std::runtime_error("parasail microbench backend is not linked in the arm64 build");
}

bool supports_neon() noexcept {
  return true;
}

RunResult run_neon_backend(const PreparedWorkload& prepared, const Options& options) {
  return run_backend_variant<stride_align::arm_neon128_backend::SimdOps>(prepared, options);
}

}  // namespace stride_align::microbench
