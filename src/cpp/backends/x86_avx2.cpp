#include "backends/x86_avx2.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation = stride_align::backend_avx2::Implementation;

}  // namespace

NB_MODULE(_avx2, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "x86 AVX2 alignment backend.");
}
