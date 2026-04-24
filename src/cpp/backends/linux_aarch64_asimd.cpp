#include "backends/linux_aarch64_asimd.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation = stride_align::backend_linux_aarch64_asimd::Implementation;

}  // namespace

NB_MODULE(_asimd, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "Linux AArch64 ASIMD alignment backend.");
}
