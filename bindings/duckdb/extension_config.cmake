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

duckdb_extension_load(
  stride_align
  DONT_LINK
  LOAD_TESTS
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
)
