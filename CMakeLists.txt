cmake_minimum_required(VERSION 3.26)
project(stride_align VERSION 0.6.0 LANGUAGES CXX)

option(STRIDE_ALIGN_BUILD_PYTHON "Build the CPython/nanobind adapters" ON)
option(STRIDE_ALIGN_BUILD_CPP_TESTS "Build the host-neutral C++ core tests" OFF)
set(STRIDE_ALIGN_INSTALL_CPP_DEFAULT ON)
if(SKBUILD)
  set(STRIDE_ALIGN_INSTALL_CPP_DEFAULT OFF)
endif()
option(
  STRIDE_ALIGN_INSTALL_CPP
  "Install the header-only C++ core and CMake package metadata"
  ${STRIDE_ALIGN_INSTALL_CPP_DEFAULT}
)

add_library(stride_align_core INTERFACE)
add_library(stride_align::core ALIAS stride_align_core)
set_target_properties(stride_align_core PROPERTIES EXPORT_NAME core)
target_compile_features(stride_align_core INTERFACE cxx_std_20)
target_include_directories(
  stride_align_core
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(STRIDE_ALIGN_BUILD_CPP_TESTS)
  enable_testing()
  add_executable(stride_align_core_test tests/cpp/test_core.cpp)
  target_link_libraries(stride_align_core_test PRIVATE stride_align::core)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(stride_align_core_test PRIVATE -O3 -Wall -Wextra -Wpedantic)
  endif()
  add_test(NAME stride_align_core_test COMMAND stride_align_core_test)
endif()

if(STRIDE_ALIGN_INSTALL_CPP)
  include(CMakePackageConfigHelpers)
  include(GNUInstallDirs)
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(
    TARGETS stride_align_core
    EXPORT stride_alignTargets
  )
  install(
    EXPORT stride_alignTargets
    FILE stride_alignTargets.cmake
    NAMESPACE stride_align::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stride_align
  )
  configure_package_config_file(
    cmake/stride_alignConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/stride_alignConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stride_align
  )
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/stride_alignConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )
  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/stride_alignConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/stride_alignConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stride_align
  )
endif()

if(NOT STRIDE_ALIGN_BUILD_PYTHON)
  return()
endif()

if(NOT SKBUILD)
  message(
    WARNING
    "This project is intended to be built through scikit-build-core.\n"
    "Typical commands:\n"
    "  pip install .\n"
    "  pip install --no-build-isolation -ve .\n"
  )
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" STRIDE_ALIGN_SYSTEM_PROCESSOR)

foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_CXX_FLAGS_${config} "-O3")
endforeach()

include(CheckCXXCompilerFlag)
include(CheckIPOSupported)

option(STRIDE_ALIGN_ENABLE_LTO "Enable interprocedural/link-time optimization" OFF)
option(STRIDE_ALIGN_BUILD_MICROBENCH "Build native C++ profiling microbenchmarks" OFF)
set(STRIDE_ALIGN_STATIC_CXX_RUNTIME_DEFAULT OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "^loongarch64$")
  set(STRIDE_ALIGN_STATIC_CXX_RUNTIME_DEFAULT ON)
endif()
option(
  STRIDE_ALIGN_STATIC_CXX_RUNTIME
  "Statically link the C++ runtime into Python extension modules"
  ${STRIDE_ALIGN_STATIC_CXX_RUNTIME_DEFAULT}
)
option(
  STRIDE_ALIGN_PERF_SYMBOLS
  "Build with debug symbols, frame pointers, and unstripped nanobind modules for perf"
  OFF
)
set(STRIDE_ALIGN_PGO_MODE "OFF" CACHE STRING "PGO mode: OFF, GENERATE, or USE")
set_property(CACHE STRIDE_ALIGN_PGO_MODE PROPERTY STRINGS OFF GENERATE USE)
set(STRIDE_ALIGN_PGO_DIR "${CMAKE_BINARY_DIR}/pgo" CACHE PATH "Directory for PGO profile data")
set(
  STRIDE_ALIGN_PGO_PROFILE_FILE
  "${STRIDE_ALIGN_PGO_DIR}/stride_align.profdata"
  CACHE FILEPATH
  "Clang PGO profile data file used when STRIDE_ALIGN_PGO_MODE=USE"
)

string(TOUPPER "${STRIDE_ALIGN_PGO_MODE}" STRIDE_ALIGN_PGO_MODE_NORMALIZED)
if(NOT STRIDE_ALIGN_PGO_MODE_NORMALIZED MATCHES "^(OFF|GENERATE|USE)$")
  message(FATAL_ERROR "STRIDE_ALIGN_PGO_MODE must be OFF, GENERATE, or USE")
endif()

# stride-align builds per-CPython-version wheels rather than a single
# abi3 wheel. The Stable ABI (Py_LIMITED_API) hides the fast Unicode
# accessors we rely on — PyUnicode_KIND, PyUnicode_*_DATA,
# PyUnicode_GET_LENGTH, PyList_GET_SIZE, PyTuple_GET_SIZE — because
# they live in cpython/unicodeobject.h, which is excluded when
# Py_LIMITED_API is set. Their function-form equivalents work but
# either copy the string (PyUnicode_AsUCS4Copy) or are one C call per
# codepoint (PyUnicode_ReadChar), which defeats the zero-copy SIMD
# path that's the whole point of routing UCS-2/UCS-4 inputs to wider
# lanes for CJK. So we don't request Development.SABIModule, nanobind's
# STABLE_ABI silently falls back to per-version tagging, and we ship
# one wheel per minor Python version. ``requires-python = ">=3.12"`` in
# pyproject.toml still bounds the matrix to 3.12 / 3.13 / 3.14.
set(STRIDE_ALIGN_PYTHON_COMPONENTS Interpreter Development.Module)
if(STRIDE_ALIGN_BUILD_MICROBENCH)
  list(APPEND STRIDE_ALIGN_PYTHON_COMPONENTS Development.Embed)
endif()

find_package(Python REQUIRED COMPONENTS ${STRIDE_ALIGN_PYTHON_COMPONENTS})
find_package(nanobind CONFIG REQUIRED)

set(STRIDE_ALIGN_IPO_SUPPORTED FALSE)
if(STRIDE_ALIGN_ENABLE_LTO)
  check_ipo_supported(
    RESULT STRIDE_ALIGN_IPO_SUPPORTED
    OUTPUT STRIDE_ALIGN_IPO_ERROR
    LANGUAGES CXX
  )
  if(NOT STRIDE_ALIGN_IPO_SUPPORTED)
    message(WARNING "IPO/LTO requested but unsupported: ${STRIDE_ALIGN_IPO_ERROR}")
  endif()
endif()

function(apply_stride_align_optimization_flags target_name)
  if(STRIDE_ALIGN_ENABLE_LTO AND STRIDE_ALIGN_IPO_SUPPORTED)
    set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()

  if(STRIDE_ALIGN_PGO_MODE_NORMALIZED STREQUAL "OFF")
    return()
  endif()

  file(MAKE_DIRECTORY "${STRIDE_ALIGN_PGO_DIR}")

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(STRIDE_ALIGN_PGO_MODE_NORMALIZED STREQUAL "GENERATE")
      target_compile_options(
        ${target_name}
        PRIVATE
          "-fprofile-generate=${STRIDE_ALIGN_PGO_DIR}"
          -fprofile-update=atomic
      )
      target_link_options(${target_name} PRIVATE "-fprofile-generate=${STRIDE_ALIGN_PGO_DIR}")
    elseif(STRIDE_ALIGN_PGO_MODE_NORMALIZED STREQUAL "USE")
      target_compile_options(
        ${target_name}
        PRIVATE
          "-fprofile-use=${STRIDE_ALIGN_PGO_DIR}"
          -fprofile-correction
          -Wno-missing-profile
      )
      target_link_options(${target_name} PRIVATE "-fprofile-use=${STRIDE_ALIGN_PGO_DIR}")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(STRIDE_ALIGN_PGO_MODE_NORMALIZED STREQUAL "GENERATE")
      target_compile_options(
        ${target_name}
        PRIVATE
          "-fprofile-instr-generate=${STRIDE_ALIGN_PGO_DIR}/stride_align-%m.profraw"
      )
      target_link_options(
        ${target_name}
        PRIVATE
          "-fprofile-instr-generate=${STRIDE_ALIGN_PGO_DIR}/stride_align-%m.profraw"
      )
    elseif(STRIDE_ALIGN_PGO_MODE_NORMALIZED STREQUAL "USE")
      target_compile_options(
        ${target_name}
        PRIVATE
          "-fprofile-instr-use=${STRIDE_ALIGN_PGO_PROFILE_FILE}"
      )
      target_link_options(
        ${target_name}
        PRIVATE
          "-fprofile-instr-use=${STRIDE_ALIGN_PGO_PROFILE_FILE}"
      )
    endif()
  else()
    message(FATAL_ERROR "PGO is only configured for GNU and Clang-like C++ compilers")
  endif()
endfunction()

function(configure_stride_align_target target_name)
  # C++20 is sufficient — we use std::popcount, <bit>, consteval,
  # if-constexpr-requires; nothing from the C++23 stdlib (no expected,
  # no format/print, no flat_*, no generator, no mdspan). Keeping the
  # required standard at 20 lets older toolchains (gcc 10, e.g. on
  # POWER8 Ubuntu 20.04) build the project.
  target_compile_features(${target_name} PRIVATE cxx_std_20)
  target_include_directories(${target_name} PRIVATE include src/cpp)

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${target_name} PRIVATE -O3 -Wall -Wextra -Wpedantic)
  endif()

  if(MSVC)
    target_compile_options(${target_name} PRIVATE /O2 /W4)
  endif()

  if(STRIDE_ALIGN_STATIC_CXX_RUNTIME)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_link_options(${target_name} PRIVATE -static-libstdc++ -static-libgcc)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      target_link_options(${target_name} PRIVATE -static-libstdc++)
    else()
      message(
        WARNING
        "STRIDE_ALIGN_STATIC_CXX_RUNTIME is enabled but no static C++ runtime flags "
        "are configured for ${CMAKE_CXX_COMPILER_ID}"
      )
    endif()
  endif()

  if(STRIDE_ALIGN_PERF_SYMBOLS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      target_compile_options(
        ${target_name}
        PRIVATE
          -g
          -fno-omit-frame-pointer
          -fno-optimize-sibling-calls
      )
      target_link_options(${target_name} PRIVATE -g)
    elseif(MSVC)
      target_compile_options(${target_name} PRIVATE /Zi /Oy-)
      target_link_options(${target_name} PRIVATE /DEBUG)
    endif()
  endif()

  apply_stride_align_optimization_flags(${target_name})
endfunction()

set(STRIDE_ALIGN_NANOBIND_MODULE_OPTIONS STABLE_ABI)
if(STRIDE_ALIGN_PERF_SYMBOLS)
  list(APPEND STRIDE_ALIGN_NANOBIND_MODULE_OPTIONS NOMINSIZE NOSTRIP)
endif()

function(add_stride_align_backend module_name backend_source)
  nanobind_add_module(${module_name} ${STRIDE_ALIGN_NANOBIND_MODULE_OPTIONS} ${backend_source})
  configure_stride_align_target(${module_name})
  install(TARGETS ${module_name} LIBRARY DESTINATION stride_align)
endfunction()

set(STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS STRIDE_ALIGN_HAVE_GENERIC=1)

nanobind_add_module(
  _cpu
  ${STRIDE_ALIGN_NANOBIND_MODULE_OPTIONS}
  src/cpp/cpu_module.cpp
  src/cpp/cpu.cpp
)
configure_stride_align_target(_cpu)

add_stride_align_backend(_generic src/cpp/backends/generic.cpp)

# Beider-Morse Phonetic Matching ships its 280 KB GENERIC rule data via
# Python's importlib.resources; the C++ engine is linked into only the
# ``_generic`` module so the rule trie + LangRegistry singletons exist
# in one ``.so`` per wheel. Python re-exports ``beider_morse`` from the
# generic backend regardless of CPU dispatch. The compile definition
# gates the binding in ``src/cpp/module_bindings.hpp``.
target_sources(_generic PRIVATE src/cpp/beider_morse_impl.cpp)
target_compile_definitions(_generic PRIVATE STRIDE_ALIGN_BMPM_BACKEND=1)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  add_stride_align_backend(_swar src/cpp/backends/swar.cpp)
  list(APPEND STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS STRIDE_ALIGN_HAVE_SWAR=1)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)$")
  add_stride_align_backend(_sse41 src/cpp/backends/x86_sse.cpp)
  add_stride_align_backend(_avx2 src/cpp/backends/x86_avx2.cpp)
  add_stride_align_backend(_avx512bwvl src/cpp/backends/x86_avx512.cpp)

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(_sse41 PRIVATE -msse4.1)
  endif()

  list(
    APPEND STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS
    STRIDE_ALIGN_HAVE_X86_SSE41=1
    STRIDE_ALIGN_HAVE_X86_AVX2=1
    STRIDE_ALIGN_HAVE_X86_AVX512BWVL=1
  )

  # AVX10.1 backends are kept in-tree but NOT built by default. GCC's
  # avx10.1-256/512 target codegen (through GCC 16-trunk) compiles the SW/NW
  # alignment kernels ~2x slower than the classic avx512f/bw/vl backend on
  # AVX-512 hardware -- measured on a Granite Rapids Xeon 6975P-C, sw-score is
  # 0.55x avx2 under avx10.1-512 but 1.28x avx2 under avx512bwvl -- and AVX10.1
  # exposes no instruction these kernels use beyond AVX-512. detect_best_backend()
  # dispatches avx512bwvl instead; its #ifdef'd priority entries reappear
  # automatically when this option is turned back on for a toolchain whose
  # AVX10 codegen has caught up.
  option(STRIDE_ALIGN_ENABLE_X86_AVX10
    "Build the x86 AVX10.1 backends (off: classic AVX-512 is faster on current toolchains)" OFF)

  if(STRIDE_ALIGN_ENABLE_X86_AVX10)
    check_cxx_compiler_flag("-mavx10.1-256" STRIDE_ALIGN_COMPILER_SUPPORTS_X86_AVX10_256)
    check_cxx_compiler_flag("-mavx10.1-512" STRIDE_ALIGN_COMPILER_SUPPORTS_X86_AVX10_512)

    if(STRIDE_ALIGN_COMPILER_SUPPORTS_X86_AVX10_256)
      add_stride_align_backend(_avx10_256 src/cpp/backends/x86_avx10_256.cpp)
      list(APPEND STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS STRIDE_ALIGN_HAVE_X86_AVX10_256=1)
    endif()

    if(STRIDE_ALIGN_COMPILER_SUPPORTS_X86_AVX10_512)
      add_stride_align_backend(_avx10_512 src/cpp/backends/x86_avx10_512.cpp)
      list(APPEND STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS STRIDE_ALIGN_HAVE_X86_AVX10_512=1)
    endif()
  endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  add_stride_align_backend(_neon src/cpp/backends/linux_aarch64_neon.cpp)
  add_stride_align_backend(_sve src/cpp/backends/linux_aarch64_sve.cpp)
  add_stride_align_backend(_sve2 src/cpp/backends/linux_aarch64_sve2.cpp)

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(_neon PRIVATE -march=armv8-a+simd)
    # Pin SVE register length to 128 bits. This makes svcnt*() compile-time
    # constants so the kernels' inner loops are unrolled and scratch arrays
    # become stack-resident. The matching supported_on_this_machine() check
    # in linux_aarch64_sve{,2}.hpp refuses to load on hardware with a
    # different SVE register width.
    target_compile_options(_sve PRIVATE -march=armv8.2-a+sve -msve-vector-bits=128)
    target_compile_options(_sve2 PRIVATE -march=armv9-a+sve2 -msve-vector-bits=128)
  endif()

  list(
    APPEND STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS
    STRIDE_ALIGN_HAVE_LINUX_AARCH64_NEON=1
    STRIDE_ALIGN_HAVE_LINUX_AARCH64_SVE=1
    STRIDE_ALIGN_HAVE_LINUX_AARCH64_SVE2=1
  )
endif()

if(APPLE AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  add_stride_align_backend(_macos_arm64_neon src/cpp/backends/macos_arm64_neon.cpp)
  list(APPEND STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS STRIDE_ALIGN_HAVE_MACOS_ARM64_NEON=1)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "^loongarch64$")
  add_stride_align_backend(_lsx src/cpp/backends/linux_loongarch64_lsx.cpp)
  add_stride_align_backend(_lasx src/cpp/backends/linux_loongarch64_lasx.cpp)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    # Some vendor toolchains default every translation unit to LSX. Keep the
    # dispatcher and fallback backends on the baseline ISA so importing the
    # package remains safe before runtime feature selection has occurred.
    foreach(target_name _cpu _generic _swar)
      target_compile_options(
        ${target_name}
        PRIVATE -march=loongarch64 -mtune=generic -mno-lsx -mno-lasx
      )
    endforeach()
    target_compile_options(_lsx PRIVATE -march=la464 -mtune=la464 -mlsx -mno-lasx)
    target_compile_options(_lasx PRIVATE -march=la464 -mtune=la464 -mlsx -mlasx)
  endif()
  list(
    APPEND STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS
    STRIDE_ALIGN_HAVE_LINUX_LOONGARCH64_LSX=1
    STRIDE_ALIGN_HAVE_LINUX_LOONGARCH64_LASX=1
  )
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "^(ppc64|ppc64le|powerpc64|powerpc64le)$")
  add_stride_align_backend(_vsx src/cpp/backends/linux_powerpc64_vsx.cpp)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(_vsx PRIVATE -mcpu=power8 -mtune=power8 -mvsx)
  endif()
  list(APPEND STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS STRIDE_ALIGN_HAVE_LINUX_POWERPC64_VSX=1)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "^riscv64$")
  add_stride_align_backend(_rvv src/cpp/backends/linux_riscv64_rvv.cpp)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(_rvv PRIVATE -march=rv64gcv)
  endif()
  list(APPEND STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS STRIDE_ALIGN_HAVE_LINUX_RISCV64_RVV=1)
endif()

target_compile_definitions(_cpu PRIVATE ${STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS})
install(TARGETS _cpu LIBRARY DESTINATION stride_align)

if(STRIDE_ALIGN_BUILD_MICROBENCH)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)$")
    add_executable(
      stride_align_x86_microbench
      src/cpp/tools/x86_microbench.cpp
      src/cpp/tools/x86_microbench_avx2.cpp
      src/cpp/tools/x86_microbench_avx512bwvl.cpp
      src/cpp/tools/x86_microbench_parasail.cpp
    )
    configure_stride_align_target(stride_align_x86_microbench)
    nanobind_build_library(nanobind-static)
    target_link_libraries(stride_align_x86_microbench PRIVATE nanobind-static Python::Python)

    execute_process(
      COMMAND
        "${Python_EXECUTABLE}"
        -c
        "import pathlib, parasail; p=pathlib.Path(parasail.__file__).resolve().parent; print(p/'include'); print(p/'libparasail.so')"
      RESULT_VARIABLE STRIDE_ALIGN_PARASAIL_QUERY_RESULT
      OUTPUT_VARIABLE STRIDE_ALIGN_PARASAIL_QUERY_OUTPUT
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(STRIDE_ALIGN_PARASAIL_QUERY_RESULT EQUAL 0)
      string(REPLACE "\n" ";" STRIDE_ALIGN_PARASAIL_QUERY_LINES "${STRIDE_ALIGN_PARASAIL_QUERY_OUTPUT}")
      list(LENGTH STRIDE_ALIGN_PARASAIL_QUERY_LINES STRIDE_ALIGN_PARASAIL_QUERY_LINE_COUNT)
      if(STRIDE_ALIGN_PARASAIL_QUERY_LINE_COUNT GREATER_EQUAL 2)
        list(GET STRIDE_ALIGN_PARASAIL_QUERY_LINES 0 STRIDE_ALIGN_PARASAIL_INCLUDE_DIR)
        list(GET STRIDE_ALIGN_PARASAIL_QUERY_LINES 1 STRIDE_ALIGN_PARASAIL_LIBRARY)
        get_filename_component(
          STRIDE_ALIGN_PARASAIL_LIBRARY_DIR
          "${STRIDE_ALIGN_PARASAIL_LIBRARY}"
          DIRECTORY
        )
        if(
          EXISTS "${STRIDE_ALIGN_PARASAIL_INCLUDE_DIR}/parasail.h" AND
          EXISTS "${STRIDE_ALIGN_PARASAIL_LIBRARY}"
        )
          target_compile_definitions(
            stride_align_x86_microbench
            PRIVATE STRIDE_ALIGN_HAVE_PARASAIL_MICROBENCH=1
          )
          target_include_directories(
            stride_align_x86_microbench
            PRIVATE "${STRIDE_ALIGN_PARASAIL_INCLUDE_DIR}"
          )
          target_link_libraries(
            stride_align_x86_microbench
            PRIVATE "${STRIDE_ALIGN_PARASAIL_LIBRARY}"
          )
          set_property(
            TARGET stride_align_x86_microbench
            APPEND
            PROPERTY BUILD_RPATH "${STRIDE_ALIGN_PARASAIL_LIBRARY_DIR};${CMAKE_CURRENT_BINARY_DIR}"
          )
          execute_process(
            COMMAND
              "${CMAKE_COMMAND}" -E create_symlink
              "${STRIDE_ALIGN_PARASAIL_LIBRARY}"
              "${CMAKE_CURRENT_BINARY_DIR}/libparasail.so.8"
          )
          set(STRIDE_ALIGN_PARASAIL_MICROBENCH_ENABLED TRUE)
        endif()
      endif()
    endif()
    if(NOT STRIDE_ALIGN_PARASAIL_MICROBENCH_ENABLED)
      file(
        GLOB
        STRIDE_ALIGN_PARASAIL_VENV_DIRS
        "${CMAKE_SOURCE_DIR}/.venv/lib/python*/site-packages/parasail"
      )
      if(STRIDE_ALIGN_PARASAIL_VENV_DIRS)
        list(GET STRIDE_ALIGN_PARASAIL_VENV_DIRS 0 STRIDE_ALIGN_PARASAIL_VENV_DIR)
        set(STRIDE_ALIGN_PARASAIL_INCLUDE_DIR "${STRIDE_ALIGN_PARASAIL_VENV_DIR}/include")
        set(STRIDE_ALIGN_PARASAIL_LIBRARY "${STRIDE_ALIGN_PARASAIL_VENV_DIR}/libparasail.so")
        if(
          EXISTS "${STRIDE_ALIGN_PARASAIL_INCLUDE_DIR}/parasail.h" AND
          EXISTS "${STRIDE_ALIGN_PARASAIL_LIBRARY}"
        )
          target_compile_definitions(
            stride_align_x86_microbench
            PRIVATE STRIDE_ALIGN_HAVE_PARASAIL_MICROBENCH=1
          )
          target_include_directories(
            stride_align_x86_microbench
            PRIVATE "${STRIDE_ALIGN_PARASAIL_INCLUDE_DIR}"
          )
          target_link_libraries(
            stride_align_x86_microbench
            PRIVATE "${STRIDE_ALIGN_PARASAIL_LIBRARY}"
          )
          set_property(
            TARGET stride_align_x86_microbench
            APPEND
            PROPERTY BUILD_RPATH "${STRIDE_ALIGN_PARASAIL_VENV_DIR};${CMAKE_CURRENT_BINARY_DIR}"
          )
          execute_process(
            COMMAND
              "${CMAKE_COMMAND}" -E create_symlink
              "${STRIDE_ALIGN_PARASAIL_LIBRARY}"
              "${CMAKE_CURRENT_BINARY_DIR}/libparasail.so.8"
          )
          set(STRIDE_ALIGN_PARASAIL_MICROBENCH_ENABLED TRUE)
        endif()
      endif()
    endif()
  elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    add_executable(
      stride_align_arm_neon_microbench
      src/cpp/tools/x86_microbench.cpp
      src/cpp/tools/arm_neon_microbench_backend.cpp
    )
    configure_stride_align_target(stride_align_arm_neon_microbench)
    nanobind_build_library(nanobind-static)
    target_link_libraries(stride_align_arm_neon_microbench PRIVATE nanobind-static Python::Python)
  else()
    message(WARNING "STRIDE_ALIGN_BUILD_MICROBENCH is currently implemented for x86 and macOS arm64 targets only")
  endif()
endif()
