#include "backends/linux_aarch64_neon.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation = stride_align::backend_linux_aarch64_neon::Implementation;

}  // namespace

NB_MODULE(_neon, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "Linux AArch64 NEON alignment backend.");
}
