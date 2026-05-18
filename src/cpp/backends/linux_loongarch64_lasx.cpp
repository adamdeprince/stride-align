#include "backends/linux_loongarch64_lasx.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation = stride_align::backend_linux_loongarch64_lasx::Implementation;

}  // namespace

NB_MODULE(_lasx, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "Linux LoongArch64 LASX alignment backend.");
}
