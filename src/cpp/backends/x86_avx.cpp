#include "backends/generic.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation =
    stride_align::backend_generic::Implementation<stride_align::BackendKind::x86_avx2>;

}  // namespace

NB_MODULE(_avx2, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "x86 AVX2 alignment backend.");
}
