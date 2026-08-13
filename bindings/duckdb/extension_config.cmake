# DuckDB v1.5.5 checks EXTENSION_STATIC_BUILD before declaring its default.
# On the first configure of a fresh build directory that means its GNU
# function/data-section flags are skipped even though the option later becomes
# ON. Restore them before DuckDB creates libduckdb_static so --gc-sections can
# discard unused core code from the loadable extension.
if(
  EXTENSION_STATIC_BUILD
  AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
  AND NOT CMAKE_CXX_FLAGS_RELEASE MATCHES "ffunction-sections"
)
  string(
    APPEND CMAKE_CXX_FLAGS_RELEASE
    " -ffunction-sections -fdata-sections"
  )
endif()

# DuckDB's bundled jemalloc does not initialize correctly on the supported
# LoongArch toolchains: even a SELECT version() fails its first small
# allocation. This config is included before DuckDB decides whether to add the
# allocator, so keep LoongArch packages and validation clients on libc malloc.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" stride_align_extension_processor)
if(
  CMAKE_SYSTEM_NAME STREQUAL "Linux"
  AND stride_align_extension_processor MATCHES "^loongarch64$"
)
  set(
    ENABLE_JEMALLOC
    OFF
    CACHE BOOL
    "Use jemalloc as the memory allocator for DuckDB"
    FORCE
  )
endif()

duckdb_extension_load(
  stride_align
  DONT_LINK
  LOAD_TESTS
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
)
