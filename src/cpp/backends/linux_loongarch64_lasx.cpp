#include "backends/generic.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation =
    stride_align::backend_generic::Implementation<stride_align::BackendKind::linux_loongarch64_lasx>;

}  // namespace

NB_MODULE(_lasx, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "Linux LoongArch64 LASX alignment backend.");
}
