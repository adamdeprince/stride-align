#include "backends/generic.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation =
    stride_align::backend_generic::Implementation<stride_align::BackendKind::linux_aarch64_sve2>;

}  // namespace

NB_MODULE(_sve2, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "Linux AArch64 SVE2 alignment backend.");
}
