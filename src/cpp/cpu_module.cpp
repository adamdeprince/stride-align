#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>

#include "cpu.hpp"

namespace nb = nanobind;
using namespace stride_align;

NB_MODULE(_cpu, m) {
  nb::enum_<BackendKind>(m, "BackendKind")
      .value("GENERIC", BackendKind::generic)
      .value("SWAR", BackendKind::swar)
      .value("X86_SSE41", BackendKind::x86_sse41)
      .value("X86_AVX2", BackendKind::x86_avx2)
      .value("X86_AVX512BWVL", BackendKind::x86_avx512bwvl)
      .value("X86_AVX10_256", BackendKind::x86_avx10_256)
      .value("X86_AVX10_512", BackendKind::x86_avx10_512)
      .value("LINUX_AARCH64_NEON", BackendKind::linux_aarch64_neon)
      .value("LINUX_AARCH64_SVE", BackendKind::linux_aarch64_sve)
      .value("LINUX_AARCH64_SVE2", BackendKind::linux_aarch64_sve2)
      .value("MACOS_ARM64_NEON", BackendKind::macos_arm64_neon)
      .value("LINUX_LOONGARCH64_LSX", BackendKind::linux_loongarch64_lsx)
      .value("LINUX_LOONGARCH64_LASX", BackendKind::linux_loongarch64_lasx)
      .value("LINUX_POWERPC64_VSX", BackendKind::linux_powerpc64_vsx)
      .value("LINUX_RISCV64_RVV", BackendKind::linux_riscv64_rvv)
      .value("SOLARIS_SPARC_VIS3", BackendKind::solaris_sparc_vis3);

  nb::class_<BackendRecord>(m, "BackendRecord")
      .def_ro("kind", &BackendRecord::kind)
      .def_prop_ro("name", [](const BackendRecord& record) { return std::string(record.name); })
      .def_ro("compiled", &BackendRecord::compiled)
      .def_ro("available", &BackendRecord::available);

  m.def("detect_best_backend", &detect_best_backend);
  m.def("backend_is_available", &backend_is_available, nb::arg("kind"));
  m.def("available_backends", &available_backends);
}
