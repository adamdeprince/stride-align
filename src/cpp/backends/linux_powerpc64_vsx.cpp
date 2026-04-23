#include "backends/generic.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation =
    stride_align::backend_generic::Implementation<stride_align::BackendKind::linux_powerpc64_vsx>;

}  // namespace

NB_MODULE(_vsx, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "Linux PowerPC64 VSX alignment backend.");
}
