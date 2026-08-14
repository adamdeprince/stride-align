cmake_minimum_required(VERSION 3.26)

file(READ "${CMAKE_CURRENT_LIST_DIR}/../../CMakeLists.txt" stride_align_root_cmake)
string(
  REGEX MATCH
  "project\\(stride_align VERSION ([0-9]+\\.[0-9]+\\.[0-9]+)"
  stride_align_version_match
  "${stride_align_root_cmake}"
)
if(NOT CMAKE_MATCH_1)
  message(FATAL_ERROR "Could not read the stride-align version from the root CMakeLists.txt")
endif()
set(STRIDE_ALIGN_MEMGRAPH_VERSION "${CMAKE_MATCH_1}")

project(
  stride_align_memgraph
  VERSION "${STRIDE_ALIGN_MEMGRAPH_VERSION}"
  LANGUAGES CXX
)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(
  STRIDE_ALIGN_MEMGRAPH_EMBEDDED_DATA
  "${CMAKE_CURRENT_BINARY_DIR}/generated/embedded_data.hpp"
)
file(
  GLOB STRIDE_ALIGN_MEMGRAPH_BMPM_RESOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/../../src/stride_align/bmpm_data/*.txt"
)
file(
  GLOB STRIDE_ALIGN_MEMGRAPH_MATRIX_RESOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/../../src/stride_align/matrix_data/*"
  "${CMAKE_CURRENT_LIST_DIR}/../../src/stride_align/matrices/keyboard_data/*.npy"
)
add_custom_command(
  OUTPUT "${STRIDE_ALIGN_MEMGRAPH_EMBEDDED_DATA}"
  COMMAND
    "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_LIST_DIR}/../duckdb/generate_embedded_data.py"
    --source-root "${CMAKE_CURRENT_LIST_DIR}/../.."
    --namespace stride_align_memgraph_embedded
    --output "${STRIDE_ALIGN_MEMGRAPH_EMBEDDED_DATA}"
  DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/../duckdb/generate_embedded_data.py"
    "${CMAKE_CURRENT_LIST_DIR}/../../src/stride_align/matrices/__init__.py"
    ${STRIDE_ALIGN_MEMGRAPH_BMPM_RESOURCES}
    ${STRIDE_ALIGN_MEMGRAPH_MATRIX_RESOURCES}
  VERBATIM
)

set(
  STRIDE_ALIGN_MEMGRAPH_API_DIR
  ""
  CACHE PATH
  "Directory containing Memgraph's public mg_procedure.h header"
)
if(STRIDE_ALIGN_MEMGRAPH_API_DIR)
  find_path(
    STRIDE_ALIGN_MEMGRAPH_INCLUDE_DIR
    mg_procedure.h
    PATHS "${STRIDE_ALIGN_MEMGRAPH_API_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
  )
else()
  find_path(
    STRIDE_ALIGN_MEMGRAPH_INCLUDE_DIR
    mg_procedure.h
    PATH_SUFFIXES memgraph
    REQUIRED
  )
endif()

set(
  STRIDE_ALIGN_MEMGRAPH_SIMD
  "native"
  CACHE STRING
  "SIMD package profile: generic, native, avx2, avx512bwvl, neon, power8_vsx, la464_lsx, la464_lasx, or la664_lasx"
)
set_property(
  CACHE STRIDE_ALIGN_MEMGRAPH_SIMD
  PROPERTY STRINGS
    generic
    native
    avx2
    avx512bwvl
    neon
    power8_vsx
    la464_lsx
    la464_lasx
    la664_lasx
)

add_library(
  stride_align MODULE
  src/stride_align.cpp
  ../../src/cpp/beider_morse_impl.cpp
  "${STRIDE_ALIGN_MEMGRAPH_EMBEDDED_DATA}"
)
set_target_properties(
  stride_align
  PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    PREFIX ""
    OUTPUT_NAME "stride_align"
    VISIBILITY_INLINES_HIDDEN YES
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(
    stride_align
    PRIVATE
      "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/src/exports.map"
  )
  set_property(
    TARGET stride_align
    APPEND
    PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/exports.map"
  )
endif()
target_include_directories(
  stride_align
  PRIVATE
    "${STRIDE_ALIGN_MEMGRAPH_INCLUDE_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/../../include"
    "${CMAKE_CURRENT_LIST_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
target_compile_definitions(
  stride_align
  PRIVATE
    STRIDE_ALIGN_MEMGRAPH_SIMD_LEVEL="${STRIDE_ALIGN_MEMGRAPH_SIMD}"
    STRIDE_ALIGN_MEMGRAPH_VERSION="${PROJECT_VERSION}"
)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" stride_align_processor)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(stride_align PRIVATE -O3 -Wall -Wextra -Wpedantic)

  if(STRIDE_ALIGN_MEMGRAPH_SIMD STREQUAL "native")
    if(APPLE AND stride_align_processor MATCHES "^(aarch64|arm64)$")
      target_compile_options(stride_align PRIVATE -mcpu=native)
    else()
      target_compile_options(stride_align PRIVATE -march=native)
    endif()
    target_compile_definitions(
      stride_align PRIVATE STRIDE_ALIGN_MEMGRAPH_TARGET_NATIVE=1
    )
  elseif(STRIDE_ALIGN_MEMGRAPH_SIMD STREQUAL "generic")
    if(stride_align_processor MATCHES "^(x86_64|amd64|i[3-6]86)$")
      target_compile_options(
        stride_align
        PRIVATE
          -march=x86-64
          -mtune=generic
          -msse2
          -mno-avx
          -mno-avx2
          -mno-avx512f
      )
    elseif(stride_align_processor MATCHES "^loongarch64$")
      target_compile_options(
        stride_align
        PRIVATE -march=loongarch64 -mtune=generic -mno-lsx -mno-lasx
      )
    endif()
    target_compile_definitions(
      stride_align PRIVATE STRIDE_ALIGN_MEMGRAPH_TARGET_GENERIC=1
    )
  elseif(STRIDE_ALIGN_MEMGRAPH_SIMD STREQUAL "avx2")
    if(NOT stride_align_processor MATCHES "^(x86_64|amd64|i[3-6]86)$")
      message(FATAL_ERROR "The avx2 Memgraph profile requires x86")
    endif()
    target_compile_options(stride_align PRIVATE -march=x86-64-v3 -mtune=generic)
    target_compile_definitions(
      stride_align PRIVATE STRIDE_ALIGN_MEMGRAPH_TARGET_AVX2=1
    )
  elseif(STRIDE_ALIGN_MEMGRAPH_SIMD STREQUAL "avx512bwvl")
    if(NOT stride_align_processor MATCHES "^(x86_64|amd64|i[3-6]86)$")
      message(FATAL_ERROR "The avx512bwvl Memgraph profile requires x86")
    endif()
    target_compile_options(stride_align PRIVATE -march=x86-64-v4 -mtune=generic)
    target_compile_definitions(
      stride_align PRIVATE STRIDE_ALIGN_MEMGRAPH_TARGET_AVX512BWVL=1
    )
  elseif(STRIDE_ALIGN_MEMGRAPH_SIMD STREQUAL "neon")
    if(NOT stride_align_processor MATCHES "^(aarch64|arm64)$")
      message(FATAL_ERROR "The neon Memgraph profile requires AArch64")
    endif()
    if(APPLE)
      target_compile_options(stride_align PRIVATE -mcpu=apple-m1)
    else()
      target_compile_options(stride_align PRIVATE -march=armv8-a+simd -mtune=generic)
    endif()
    target_compile_definitions(
      stride_align PRIVATE STRIDE_ALIGN_MEMGRAPH_TARGET_NEON=1
    )
  elseif(STRIDE_ALIGN_MEMGRAPH_SIMD STREQUAL "power8_vsx")
    if(NOT stride_align_processor MATCHES "^(ppc64|ppc64le|powerpc64|powerpc64le)$")
      message(FATAL_ERROR "The power8_vsx Memgraph profile requires PowerPC64")
    endif()
    target_compile_options(stride_align PRIVATE -mcpu=power8 -mtune=power8 -mvsx)
    target_compile_definitions(
      stride_align PRIVATE STRIDE_ALIGN_MEMGRAPH_TARGET_POWER8_VSX=1
    )
  elseif(STRIDE_ALIGN_MEMGRAPH_SIMD STREQUAL "la464_lsx")
    if(NOT stride_align_processor MATCHES "^loongarch64$")
      message(FATAL_ERROR "The la464_lsx Memgraph profile requires LoongArch64")
    endif()
    target_compile_options(
      stride_align PRIVATE -march=la464 -mtune=la464 -mlsx -mno-lasx
    )
    target_compile_definitions(
      stride_align
      PRIVATE
        STRIDE_ALIGN_MEMGRAPH_TARGET_LSX=1
        STRIDE_ALIGN_MEMGRAPH_TARGET_LA464=1
    )
  elseif(STRIDE_ALIGN_MEMGRAPH_SIMD STREQUAL "la464_lasx")
    if(NOT stride_align_processor MATCHES "^loongarch64$")
      message(FATAL_ERROR "The la464_lasx Memgraph profile requires LoongArch64")
    endif()
    target_compile_options(
      stride_align PRIVATE -march=la464 -mtune=la464 -mlsx -mlasx
    )
    target_compile_definitions(
      stride_align
      PRIVATE
        STRIDE_ALIGN_MEMGRAPH_TARGET_LASX=1
        STRIDE_ALIGN_MEMGRAPH_TARGET_LA464=1
    )
  elseif(STRIDE_ALIGN_MEMGRAPH_SIMD STREQUAL "la664_lasx")
    if(NOT stride_align_processor MATCHES "^loongarch64$")
      message(FATAL_ERROR "The la664_lasx Memgraph profile requires LoongArch64")
    endif()
    target_compile_options(
      stride_align PRIVATE -march=la664 -mtune=la664 -mlsx -mlasx
    )
    target_compile_definitions(
      stride_align
      PRIVATE
        STRIDE_ALIGN_MEMGRAPH_TARGET_LASX=1
        STRIDE_ALIGN_MEMGRAPH_TARGET_LA664=1
    )
  else()
    message(FATAL_ERROR "Unknown STRIDE_ALIGN_MEMGRAPH_SIMD=${STRIDE_ALIGN_MEMGRAPH_SIMD}")
  endif()
endif()

option(
  STRIDE_ALIGN_MEMGRAPH_STATIC_CXX_RUNTIME
  "Statically link the C++ runtime into the Linux query module"
  ON
)
if(
  CMAKE_SYSTEM_NAME STREQUAL "Linux"
  AND STRIDE_ALIGN_MEMGRAPH_STATIC_CXX_RUNTIME
  AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
)
  target_link_options(
    stride_align
    PRIVATE
      -static-libstdc++
      -static-libgcc
      -Wl,--exclude-libs,ALL
  )
endif()

option(
  STRIDE_ALIGN_MEMGRAPH_STRIP
  "Strip the packaged Linux query module at link time"
  ON
)
if(
  CMAKE_SYSTEM_NAME STREQUAL "Linux"
  AND STRIDE_ALIGN_MEMGRAPH_STRIP
  AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"
)
  target_link_options(stride_align PRIVATE -s)
endif()

if(APPLE)
  # The MGP symbols are supplied by Memgraph when the module is loaded.
  target_link_options(stride_align PRIVATE -undefined dynamic_lookup)
endif()

include(GNUInstallDirs)
install(
  TARGETS stride_align
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/memgraph/query_modules"
)
