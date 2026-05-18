#include "backends/x86_avx2.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation = stride_align::backend_avx2::Implementation;

#if defined(__GNUC__) || defined(__clang__)
#define STRIDE_ALIGN_X86_MODULE_BASELINE __attribute__((target("default")))
#else
#define STRIDE_ALIGN_X86_MODULE_BASELINE
#endif

STRIDE_ALIGN_X86_MODULE_BASELINE void bind_module(nanobind::module_& m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "x86 AVX2 alignment backend.");
}

}  // namespace

NB_MODULE(_avx2, m) {
  bind_module(m);
}

#undef STRIDE_ALIGN_X86_MODULE_BASELINE
