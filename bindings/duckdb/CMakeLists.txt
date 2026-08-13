cmake_minimum_required(VERSION 3.26)

set(TARGET_NAME stride_align)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME} LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(STRIDE_ALIGN_EMBEDDED_DATA ${CMAKE_CURRENT_BINARY_DIR}/generated/embedded_data.hpp)
file(GLOB STRIDE_ALIGN_BMPM_RESOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_LIST_DIR}/../../src/stride_align/bmpm_data/*.txt)
file(GLOB STRIDE_ALIGN_MATRIX_RESOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_LIST_DIR}/../../src/stride_align/matrix_data/*
  ${CMAKE_CURRENT_LIST_DIR}/../../src/stride_align/matrices/keyboard_data/*.npy)
add_custom_command(
  OUTPUT ${STRIDE_ALIGN_EMBEDDED_DATA}
  COMMAND
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate_embedded_data.py
    --source-root ${CMAKE_CURRENT_LIST_DIR}/../..
    --output ${STRIDE_ALIGN_EMBEDDED_DATA}
  DEPENDS
    ${CMAKE_CURRENT_LIST_DIR}/generate_embedded_data.py
    ${CMAKE_CURRENT_LIST_DIR}/../../src/stride_align/matrices/__init__.py
    ${CMAKE_CURRENT_LIST_DIR}/../../src/stride_align/matrices/keyboard.py
    ${STRIDE_ALIGN_BMPM_RESOURCES}
    ${STRIDE_ALIGN_MATRIX_RESOURCES}
  VERBATIM
)

set(
  STRIDE_ALIGN_DUCKDB_SIMD
  "native"
  CACHE STRING
  "SIMD package profile: generic, native, avx2, avx512bwvl, neon, la464_lsx, la464_lasx, or la664_lasx"
)
set_property(
  CACHE STRIDE_ALIGN_DUCKDB_SIMD
  PROPERTY STRINGS
    generic
    native
    avx2
    avx512bwvl
    neon
    la464_lsx
    la464_lasx
    la664_lasx
)

set(
  EXTENSION_SOURCES
  src/stride_align_extension.cpp
  src/batch_functions.cpp
  src/algorithm_functions.cpp
  src/matrix_functions.cpp
  ../../src/cpp/beider_morse_impl.cpp
  ${STRIDE_ALIGN_EMBEDDED_DATA}
)
build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

function(configure_stride_align_duckdb_target target_name)
  target_compile_features(${target_name} PRIVATE cxx_std_20)
  target_include_directories(
    ${target_name}
    PRIVATE
      ${CMAKE_CURRENT_LIST_DIR}/src/include
      ${CMAKE_CURRENT_LIST_DIR}/../../include
      ${CMAKE_CURRENT_BINARY_DIR}/generated
  )
  target_compile_definitions(
    ${target_name}
    PRIVATE STRIDE_ALIGN_DUCKDB_SIMD_LEVEL="${STRIDE_ALIGN_DUCKDB_SIMD}"
  )

  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" stride_align_processor)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${target_name} PRIVATE -O3)

    if(STRIDE_ALIGN_DUCKDB_SIMD STREQUAL "native")
      target_compile_options(${target_name} PRIVATE -march=native)
      target_compile_definitions(
        ${target_name} PRIVATE STRIDE_ALIGN_DUCKDB_TARGET_NATIVE=1
      )
    elseif(STRIDE_ALIGN_DUCKDB_SIMD STREQUAL "generic")
      if(stride_align_processor MATCHES "^(x86_64|amd64|i[3-6]86)$")
        target_compile_options(
          ${target_name}
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
          ${target_name}
          PRIVATE
            -march=loongarch64
            -mtune=generic
            -mno-lsx
            -mno-lasx
        )
      endif()
      target_compile_definitions(
        ${target_name} PRIVATE STRIDE_ALIGN_DUCKDB_TARGET_GENERIC=1
      )
    elseif(STRIDE_ALIGN_DUCKDB_SIMD STREQUAL "avx2")
      if(NOT stride_align_processor MATCHES "^(x86_64|amd64|i[3-6]86)$")
        message(FATAL_ERROR "The avx2 DuckDB package profile requires x86")
      endif()
      target_compile_options(
        ${target_name} PRIVATE -march=x86-64-v3 -mtune=generic
      )
      target_compile_definitions(
        ${target_name} PRIVATE STRIDE_ALIGN_DUCKDB_TARGET_AVX2=1
      )
    elseif(STRIDE_ALIGN_DUCKDB_SIMD STREQUAL "avx512bwvl")
      if(NOT stride_align_processor MATCHES "^(x86_64|amd64|i[3-6]86)$")
        message(FATAL_ERROR "The avx512bwvl DuckDB package profile requires x86")
      endif()
      target_compile_options(
        ${target_name}
        PRIVATE -march=x86-64-v4 -mtune=generic
      )
      target_compile_definitions(
        ${target_name} PRIVATE STRIDE_ALIGN_DUCKDB_TARGET_AVX512BWVL=1
      )
    elseif(STRIDE_ALIGN_DUCKDB_SIMD STREQUAL "neon")
      if(NOT stride_align_processor MATCHES "^(aarch64|arm64)$")
        message(FATAL_ERROR "The neon DuckDB package profile requires AArch64")
      endif()
      if(APPLE)
        # Every macOS arm64 machine is Apple Silicon. M1 is the portable CPU
        # baseline and gives Clang an explicit target for auto-vectorization.
        target_compile_options(${target_name} PRIVATE -mcpu=apple-m1)
      else()
        target_compile_options(
          ${target_name} PRIVATE -march=armv8-a+simd -mtune=generic
        )
      endif()
      target_compile_definitions(
        ${target_name} PRIVATE STRIDE_ALIGN_DUCKDB_TARGET_NEON=1
      )
    elseif(STRIDE_ALIGN_DUCKDB_SIMD STREQUAL "la464_lsx")
      if(NOT stride_align_processor MATCHES "^loongarch64$")
        message(FATAL_ERROR "The la464_lsx DuckDB package profile requires LoongArch64")
      endif()
      target_compile_options(
        ${target_name}
        PRIVATE -march=la464 -mtune=la464 -mlsx -mno-lasx
      )
      target_compile_definitions(
        ${target_name}
        PRIVATE
          STRIDE_ALIGN_DUCKDB_TARGET_LSX=1
          STRIDE_ALIGN_DUCKDB_TARGET_LA464=1
      )
    elseif(STRIDE_ALIGN_DUCKDB_SIMD STREQUAL "la464_lasx")
      if(NOT stride_align_processor MATCHES "^loongarch64$")
        message(FATAL_ERROR "The la464_lasx DuckDB package profile requires LoongArch64")
      endif()
      target_compile_options(
        ${target_name}
        PRIVATE -march=la464 -mtune=la464 -mlsx -mlasx
      )
      target_compile_definitions(
        ${target_name}
        PRIVATE
          STRIDE_ALIGN_DUCKDB_TARGET_LASX=1
          STRIDE_ALIGN_DUCKDB_TARGET_LA464=1
      )
    elseif(STRIDE_ALIGN_DUCKDB_SIMD STREQUAL "la664_lasx")
      if(NOT stride_align_processor MATCHES "^loongarch64$")
        message(FATAL_ERROR "The la664_lasx DuckDB package profile requires LoongArch64")
      endif()
      target_compile_options(
        ${target_name}
        PRIVATE -march=la664 -mtune=la664 -mlsx -mlasx
      )
      target_compile_definitions(
        ${target_name}
        PRIVATE
          STRIDE_ALIGN_DUCKDB_TARGET_LASX=1
          STRIDE_ALIGN_DUCKDB_TARGET_LA664=1
      )
    else()
      message(
        FATAL_ERROR
        "Unknown STRIDE_ALIGN_DUCKDB_SIMD=${STRIDE_ALIGN_DUCKDB_SIMD}"
      )
    endif()
  endif()
endfunction()

configure_stride_align_duckdb_target(${EXTENSION_NAME})
configure_stride_align_duckdb_target(${LOADABLE_EXTENSION_NAME})

# Loadable packages must not inherit the build host's libstdc++ version.
# This is especially important on LoongArch, where the C++20 compiler is
# newer than the distribution runtime, but it also keeps Linux x86 packages
# usable with older DuckDB installations.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_options(
    ${LOADABLE_EXTENSION_NAME}
    PRIVATE -static-libstdc++ -static-libgcc
  )
endif()

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
)
