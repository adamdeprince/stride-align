#include "backends/solaris_sparc_vis3.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation = stride_align::backend_solaris_sparc_vis3::Implementation;

}  // namespace

NB_MODULE(_vis3, m) {
  stride_align::bindings::bind_backend_module<Implementation>(m, "Solaris SPARC VIS3 alignment backend.");
}
