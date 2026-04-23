cmake_minimum_required(VERSION 3.26)
project(stride_align LANGUAGES CXX)

if(NOT SKBUILD)
  message(
    WARNING
    "This project is intended to be built through scikit-build-core.\n"
    "Typical commands:\n"
    "  pip install .\n"
    "  pip install --no-build-isolation -ve .\n"
  )
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_CXX_FLAGS_${config} "-O3")
endforeach()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
include(CheckCXXCompilerFlag)

function(configure_stride_align_target target_name)
  target_compile_features(${target_name} PRIVATE cxx_std_23)
  target_include_directories(${target_name} PRIVATE include src/cpp)

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${target_name} PRIVATE -O3 -Wall -Wextra -Wpedantic)
  endif()

  if(MSVC)
    target_compile_options(${target_name} PRIVATE /O2 /W4)
  endif()
endfunction()

function(add_stride_align_backend module_name backend_source)
  nanobind_add_module(${module_name} STABLE_ABI ${backend_source})
  configure_stride_align_target(${module_name})
  install(TARGETS ${module_name} LIBRARY DESTINATION stride_align)
endfunction()

set(STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS STRIDE_ALIGN_HAVE_GENERIC=1)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" STRIDE_ALIGN_SYSTEM_PROCESSOR)

nanobind_add_module(_cpu STABLE_ABI src/cpp/cpu_module.cpp src/cpp/cpu.cpp)
configure_stride_align_target(_cpu)

add_stride_align_backend(_generic src/cpp/backends/generic.cpp)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)$")
  add_stride_align_backend(_sse41 src/cpp/backends/x86_sse.cpp)
  add_stride_align_backend(_avx2 src/cpp/backends/x86_avx.cpp)
  add_stride_align_backend(_avx512bwvl src/cpp/backends/x86_avx512.cpp)

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(_sse41 PRIVATE -msse4.1)
    target_compile_options(_avx2 PRIVATE -mavx2)
    target_compile_options(_avx512bwvl PRIVATE -mavx512f -mavx512bw -mavx512vl)
  endif()

  list(
    APPEND STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS
    STRIDE_ALIGN_HAVE_X86_SSE41=1
    STRIDE_ALIGN_HAVE_X86_AVX2=1
    STRIDE_ALIGN_HAVE_X86_AVX512BWVL=1
  )

  check_cxx_compiler_flag("-mavx10.1-256" STRIDE_ALIGN_COMPILER_SUPPORTS_X86_AVX10_256)
  check_cxx_compiler_flag("-mavx10.1-512" STRIDE_ALIGN_COMPILER_SUPPORTS_X86_AVX10_512)

  if(STRIDE_ALIGN_COMPILER_SUPPORTS_X86_AVX10_256)
    add_stride_align_backend(_avx10_256 src/cpp/backends/x86_avx10_256.cpp)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      target_compile_options(_avx10_256 PRIVATE -mavx10.1-256)
    endif()
    list(APPEND STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS STRIDE_ALIGN_HAVE_X86_AVX10_256=1)
  endif()

  if(STRIDE_ALIGN_COMPILER_SUPPORTS_X86_AVX10_512)
    add_stride_align_backend(_avx10_512 src/cpp/backends/x86_avx10_512.cpp)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      target_compile_options(_avx10_512 PRIVATE -mavx10.1-512)
    endif()
    list(APPEND STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS STRIDE_ALIGN_HAVE_X86_AVX10_512=1)
  endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  add_stride_align_backend(_asimd src/cpp/backends/linux_aarch64_asimd.cpp)
  add_stride_align_backend(_neon src/cpp/backends/linux_aarch64_neon.cpp)
  add_stride_align_backend(_sve src/cpp/backends/linux_aarch64_sve.cpp)
  add_stride_align_backend(_sve2 src/cpp/backends/linux_aarch64_sve2.cpp)

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(_asimd PRIVATE -march=armv8-a+simd)
    target_compile_options(_neon PRIVATE -march=armv8-a+simd)
    target_compile_options(_sve PRIVATE -march=armv8.2-a+sve)
    target_compile_options(_sve2 PRIVATE -march=armv9-a+sve2)
  endif()

  list(
    APPEND STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS
    STRIDE_ALIGN_HAVE_LINUX_AARCH64_ASIMD=1
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
  list(
    APPEND STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS
    STRIDE_ALIGN_HAVE_LINUX_LOONGARCH64_LSX=1
    STRIDE_ALIGN_HAVE_LINUX_LOONGARCH64_LASX=1
  )
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "^(ppc64|ppc64le|powerpc64)$")
  add_stride_align_backend(_vsx src/cpp/backends/linux_powerpc64_vsx.cpp)
  list(APPEND STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS STRIDE_ALIGN_HAVE_LINUX_POWERPC64_VSX=1)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "^riscv64$")
  add_stride_align_backend(_rvv src/cpp/backends/linux_riscv64_rvv.cpp)
  list(APPEND STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS STRIDE_ALIGN_HAVE_LINUX_RISCV64_RVV=1)
endif()

target_compile_definitions(_cpu PRIVATE ${STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS})
install(TARGETS _cpu LIBRARY DESTINATION stride_align)
