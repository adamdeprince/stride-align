# Host-neutral C++ core

The C++20 data plane under `include/stride_align/` has no Python or nanobind
dependency. The CPython-specific result ownership and binding code lives under
`src/cpp/`; other hosts should depend only on `stride_align::core`.

## Build and consume

```sh
cmake -S . -B build/core -G Ninja \
  -DSTRIDE_ALIGN_BUILD_PYTHON=OFF \
  -DSTRIDE_ALIGN_INSTALL_CPP=ON
cmake --build build/core
cmake --install build/core --prefix /your/prefix
```

```cmake
find_package(stride_align CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE stride_align::core)
```

```cpp
#include <stride_align/core.hpp>

auto pair = stride_align::utf8::prepare_pair("kitten", "sitting");
auto distance = stride_align::core::levenshtein_distance(pair);
```

The facade currently exposes Levenshtein, OSA, unrestricted
Damerau-Levenshtein, Indel, Hamming, Jaro, Jaro-Winkler, and linear or affine
Smith-Waterman/Needleman-Wunsch scores.

## UTF-8 preparation contract

`utf8::prepare_pair` validates UTF-8 and preserves Unicode codepoint equality:

- A clean ASCII pair is returned as borrowed `uint8_t` spans without decoding
  or allocation. Keep both source strings alive while using the pair.
- A short non-ASCII pair owns UCS-2 tokens when every codepoint is in the BMP,
  otherwise it owns UCS-4 tokens.
- A long non-ASCII pair uses one shared Swiss table with SIMD-probed 16-byte
  fingerprint groups to assign dense token IDs, then stores those IDs in the
  narrowest of 8, 16, or 32 bits.
- `utf8::prepare_streaming` and `PreparationMode::streaming` always own UCS-4
  tokens, including for ASCII. This keeps state stable when later strings are
  not available during preparation.

Packing is valid for equality-based algorithms. A future algorithm that uses
the numeric value or ordering of codepoints must request
`NonAsciiPolicy::fixed_width`.

`utf8::is_ascii` uses the SIMD level selected when the consumer is compiled:
AVX-512BW, AVX2, SSE2, NEON, LSX, or LASX, with word and scalar tails. Swiss
table control groups are SIMD-probed on x86, AArch64, and LoongArch.

Long linear and affine SW/NW inputs use an anti-diagonal wavefront whose
independent cells are SIMD-vectorized by Clang/GCC. A conservative score bound
selects signed 8-, 16-, 32-, or 64-bit cells without saturation; short inputs
retain the lower-overhead rolling-row kernel.

## Host boundary

Host adapters own conversion and lifetime concerns:

```text
host string/value -> UTF-8 preparation -> typed core algorithm -> host result
```

The Python adapter can continue using CPython's fixed-width Unicode storage.
The DuckDB adapter instead accepts `VARCHAR` UTF-8 and must score borrowed
ASCII pairs before DuckDB's `string_t` values leave scope. See
[`bindings/duckdb/`](../bindings/duckdb/README.md).

The R adapter treats every `CHARSXP` as immutable and never changes a
`STRSXP`. Clean ASCII bytes are borrowed only for the synchronous native call.
Every non-ASCII dense or fixed-width token sequence is owned by an
operation-local buffer. Native- and Latin-1-marked R strings are translated to
UTF-8 and copied before preparation; non-ASCII strings marked as raw bytes are
rejected. See [`bindings/r/`](../bindings/r/README.md).
