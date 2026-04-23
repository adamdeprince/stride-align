#include "backends/generic.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation =
    stride_align::backend_generic::Implementation<stride_align::BackendKind::linux_aarch64_asimd>;

}  // namespace

NB_MODULE(_asimd, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "Linux AArch64 ASIMD alignment backend.");
}
