#include "backends/linux_powerpc64_vsx.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation = stride_align::backend_linux_powerpc64_vsx::Implementation;

}  // namespace

NB_MODULE(_vsx, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "Linux PowerPC64 VSX alignment backend.");
}
