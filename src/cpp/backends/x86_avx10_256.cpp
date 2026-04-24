#include "backends/x86_avx10_256.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation = stride_align::backend_avx10_256::Implementation;

}  // namespace

NB_MODULE(_avx10_256, m) {
  stride_align::bindings::bind_backend_module<Implementation>(
      m,
      "x86 AVX10.1 256-bit alignment backend.");
}
