#include "backends/generic.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation =
    stride_align::backend_generic::Implementation<stride_align::BackendKind::x86_sse41>;

}  // namespace

NB_MODULE(_sse41, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "x86 SSE4.1 alignment backend.");
}
