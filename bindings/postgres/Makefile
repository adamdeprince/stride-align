EXTENSION = stride_align
MODULE_big = stride_align
DATA = stride_align--0.6.0.sql
OBJS = src/stride_align.o src/matrix_functions.o src/beider_morse_impl.o
GENERATED_DATA = generated/embedded_data.hpp
.DEFAULT_GOAL := all

PG_CONFIG ?= pg_config
STRIDE_ALIGN_PYTHON ?= python3
PGXS := $(shell $(PG_CONFIG) --pgxs)

EMBEDDED_RESOURCES := \
  ../../src/stride_align/matrices/__init__.py \
  $(wildcard ../../src/stride_align/bmpm_data/*.txt) \
  $(wildcard ../../src/stride_align/matrix_data/*) \
  $(wildcard ../../src/stride_align/matrices/keyboard_data/*.npy)

POSTGRES_MAJOR := $(shell $(PG_CONFIG) --version | sed -E 's/^PostgreSQL ([0-9]+).*/\1/')
ifeq ($(filter 16 17 18,$(POSTGRES_MAJOR)),)
$(error stride-align supports PostgreSQL 16, 17, and 18; PG_CONFIG reports $(POSTGRES_MAJOR))
endif

STRIDE_ALIGN_POSTGRES_SIMD ?= native
HOST_CPU := $(shell uname -m)
HOST_OS := $(shell uname -s)

PG_CPPFLAGS += -I$(CURDIR)/../../include -I$(CURDIR)/src -I$(CURDIR)/generated
PG_CXXFLAGS += -std=c++20 -O3 -Wall -Wextra -Wpedantic

ifeq ($(STRIDE_ALIGN_POSTGRES_SIMD),generic)
  PG_CPPFLAGS += -DSTRIDE_ALIGN_POSTGRES_TARGET_GENERIC=1 -DSTRIDE_ALIGN_POSTGRES_SIMD_LEVEL=\"generic\"
  ifneq ($(filter x86_64 amd64,$(HOST_CPU)),)
    PG_CXXFLAGS += -march=x86-64 -mtune=generic -msse2 -mno-avx -mno-avx2 -mno-avx512f
  else ifneq ($(filter loongarch64,$(HOST_CPU)),)
    PG_CXXFLAGS += -march=loongarch64 -mtune=generic -mno-lsx -mno-lasx
  endif
else ifeq ($(STRIDE_ALIGN_POSTGRES_SIMD),native)
  PG_CPPFLAGS += -DSTRIDE_ALIGN_POSTGRES_TARGET_NATIVE=1 -DSTRIDE_ALIGN_POSTGRES_SIMD_LEVEL=\"native\"
  PG_CXXFLAGS += -march=native
else ifeq ($(STRIDE_ALIGN_POSTGRES_SIMD),avx2)
  ifeq ($(filter x86_64 amd64,$(HOST_CPU)),)
    $(error The avx2 PostgreSQL profile requires x86_64; host is $(HOST_CPU))
  endif
  PG_CPPFLAGS += -DSTRIDE_ALIGN_POSTGRES_TARGET_AVX2=1 -DSTRIDE_ALIGN_POSTGRES_SIMD_LEVEL=\"avx2\"
  PG_CXXFLAGS += -march=x86-64-v3 -mtune=generic
else ifeq ($(STRIDE_ALIGN_POSTGRES_SIMD),avx512bwvl)
  ifeq ($(filter x86_64 amd64,$(HOST_CPU)),)
    $(error The avx512bwvl PostgreSQL profile requires x86_64; host is $(HOST_CPU))
  endif
  PG_CPPFLAGS += -DSTRIDE_ALIGN_POSTGRES_TARGET_AVX512BWVL=1 -DSTRIDE_ALIGN_POSTGRES_SIMD_LEVEL=\"avx512bwvl\"
  PG_CXXFLAGS += -march=x86-64-v4 -mtune=generic
else ifeq ($(STRIDE_ALIGN_POSTGRES_SIMD),neon)
  ifeq ($(filter aarch64 arm64,$(HOST_CPU)),)
    $(error The neon PostgreSQL profile requires AArch64; host is $(HOST_CPU))
  endif
  PG_CPPFLAGS += -DSTRIDE_ALIGN_POSTGRES_TARGET_NEON=1 -DSTRIDE_ALIGN_POSTGRES_SIMD_LEVEL=\"neon\"
  ifeq ($(HOST_OS),Darwin)
    PG_CXXFLAGS += -mcpu=apple-m1
  else
    PG_CXXFLAGS += -march=armv8-a+simd -mtune=generic
  endif
else ifeq ($(STRIDE_ALIGN_POSTGRES_SIMD),power8_vsx)
  ifeq ($(filter ppc64 ppc64le powerpc64 powerpc64le,$(HOST_CPU)),)
    $(error The power8_vsx PostgreSQL profile requires PowerPC64; host is $(HOST_CPU))
  endif
  PG_CPPFLAGS += -DSTRIDE_ALIGN_POSTGRES_TARGET_POWER8_VSX=1 -DSTRIDE_ALIGN_POSTGRES_SIMD_LEVEL=\"power8_vsx\"
  PG_CXXFLAGS += -mcpu=power8 -mtune=power8 -mvsx
else ifeq ($(STRIDE_ALIGN_POSTGRES_SIMD),la464_lsx)
  ifneq ($(HOST_CPU),loongarch64)
    $(error The la464_lsx PostgreSQL profile requires LoongArch64; host is $(HOST_CPU))
  endif
  PG_CPPFLAGS += -DSTRIDE_ALIGN_POSTGRES_TARGET_LSX=1 -DSTRIDE_ALIGN_POSTGRES_TARGET_LA464=1 -DSTRIDE_ALIGN_POSTGRES_SIMD_LEVEL=\"la464_lsx\"
  PG_CXXFLAGS += -march=la464 -mtune=la464 -mlsx -mno-lasx
else ifeq ($(STRIDE_ALIGN_POSTGRES_SIMD),la464_lasx)
  ifneq ($(HOST_CPU),loongarch64)
    $(error The la464_lasx PostgreSQL profile requires LoongArch64; host is $(HOST_CPU))
  endif
  PG_CPPFLAGS += -DSTRIDE_ALIGN_POSTGRES_TARGET_LASX=1 -DSTRIDE_ALIGN_POSTGRES_TARGET_LA464=1 -DSTRIDE_ALIGN_POSTGRES_SIMD_LEVEL=\"la464_lasx\"
  PG_CXXFLAGS += -march=la464 -mtune=la464 -mlsx -mlasx
else ifeq ($(STRIDE_ALIGN_POSTGRES_SIMD),la664_lasx)
  ifneq ($(HOST_CPU),loongarch64)
    $(error The la664_lasx PostgreSQL profile requires LoongArch64; host is $(HOST_CPU))
  endif
  PG_CPPFLAGS += -DSTRIDE_ALIGN_POSTGRES_TARGET_LASX=1 -DSTRIDE_ALIGN_POSTGRES_TARGET_LA664=1 -DSTRIDE_ALIGN_POSTGRES_SIMD_LEVEL=\"la664_lasx\"
  PG_CXXFLAGS += -march=la664 -mtune=la664 -mlsx -mlasx
else
$(error unknown STRIDE_ALIGN_POSTGRES_SIMD=$(STRIDE_ALIGN_POSTGRES_SIMD))
endif

ifeq ($(HOST_OS),Darwin)
  SHLIB_LINK += -lc++
else
  SHLIB_LINK += -lstdc++
endif

# PGXS has C rules but does not consistently provide a C++ suffix rule across
# PostgreSQL distributions. Keep the compiler and all server flags explicit.
src/%.o: src/%.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(PICFLAGS) -c $< -o $@

$(GENERATED_DATA): ../duckdb/generate_embedded_data.py $(EMBEDDED_RESOURCES)
	@mkdir -p generated
	$(STRIDE_ALIGN_PYTHON) ../duckdb/generate_embedded_data.py \
		--source-root ../.. \
		--namespace stride_align_postgres_embedded \
		--output $@

stride_align--0.6.0.sql: generate_sql.py
	$(STRIDE_ALIGN_PYTHON) generate_sql.py

src/matrix_functions.o: $(GENERATED_DATA)

EXTRA_CLEAN += $(GENERATED_DATA)

include $(PGXS)
