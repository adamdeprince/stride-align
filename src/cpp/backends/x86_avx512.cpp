#include "backends/x86_avx512bwvl.hpp"
#include "module_bindings.hpp"

namespace {

using Implementation = stride_align::backend_avx512bwvl::Implementation;

}  // namespace

NB_MODULE(_avx512bwvl, m) {
  stride_align::bindings::bind_backend_module<Implementation>(
      m,
      "x86 AVX-512F/BW/VL alignment backend.");
}
