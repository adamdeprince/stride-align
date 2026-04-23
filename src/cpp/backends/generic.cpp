#include "backends/generic.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation = stride_align::backend_generic::Implementation<stride_align::BackendKind::generic>;

}  // namespace

NB_MODULE(_generic, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "Generic alignment backend.");
}
