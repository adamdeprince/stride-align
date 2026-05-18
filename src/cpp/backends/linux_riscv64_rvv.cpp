#include "backends/linux_riscv64_rvv.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation = stride_align::backend_linux_riscv64_rvv::Implementation;

}  // namespace

NB_MODULE(_rvv, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "Linux RISC-V RVV alignment backend.");
}
