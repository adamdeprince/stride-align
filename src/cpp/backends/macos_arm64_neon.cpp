#include "backends/generic.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation =
    stride_align::backend_generic::Implementation<stride_align::BackendKind::macos_arm64_neon>;

}  // namespace

NB_MODULE(_macos_arm64_neon, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "macOS arm64 NEON alignment backend.");
}
