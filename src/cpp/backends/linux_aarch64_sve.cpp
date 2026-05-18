#include "backends/linux_aarch64_sve.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation = stride_align::backend_linux_aarch64_sve::Implementation;

}  // namespace

NB_MODULE(_sve, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "Linux AArch64 SVE alignment backend.");
}
