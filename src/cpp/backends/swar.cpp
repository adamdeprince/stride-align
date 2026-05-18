#include "backends/swar.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation = stride_align::backend_swar::Implementation;

}  // namespace

NB_MODULE(_swar, m) {
  stride_align::bindings::bind_backend_module<Implementation>(
      m,
      "64-bit SWAR Farrar alignment backend.");
}
