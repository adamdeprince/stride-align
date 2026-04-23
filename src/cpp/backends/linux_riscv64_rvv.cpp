#include "backends/generic.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation =
    stride_align::backend_generic::Implementation<stride_align::BackendKind::linux_riscv64_rvv>;

}  // namespace

NB_MODULE(_rvv, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "Linux RISC-V RVV alignment backend.");
}
