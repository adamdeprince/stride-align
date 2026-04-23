#include "backends/generic.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation =
    stride_align::backend_generic::Implementation<stride_align::BackendKind::linux_loongarch64_lsx>;

}  // namespace

NB_MODULE(_lsx, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "Linux LoongArch64 LSX alignment backend.");
}
