#include "backends/linux_loongarch64_lsx.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation = stride_align::backend_linux_loongarch64_lsx::Implementation;

}  // namespace

NB_MODULE(_lsx, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "Linux LoongArch64 LSX alignment backend.");
}
