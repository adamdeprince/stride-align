# POWER8 / gcc-10 build workarounds — TO BE REVERTED

POWER8 (Ubuntu 20.04, repo at `/home/kxpoem/dev/stride-align`) only has
gcc 9.4 and gcc 10.5 in its system package set, and libstdc++ paired
with those is 10.x. gcc-10's libstdc++ ships C++20 language support but
misses several C++20 *library* additions that landed in libstdc++ 11.

We carry a few workarounds so the project can build on this host until
the in-progress gcc 16 build on the user's build machine lands and
gets installed on POWER8. **Every workaround below should be reverted
once POWER8 has gcc 16.** This file exists so they're easy to find
and remove.

## 1. CMake C++ standard lowered from 23 → 20

**Files:**
* `CMakeLists.txt:14` — `set(CMAKE_CXX_STANDARD 20)`
* `CMakeLists.txt:142` — `target_compile_features(${target_name} PRIVATE cxx_std_20)`

**Why:** CMake's `cxx_std_23` compiler-feature lookup requires gcc 11+;
gcc 10.5 is rejected even though the *language* features we use are
all C++20. The codebase doesn't actually use any C++23 stdlib feature
— this was always nominal.

**Revert:** Restore both lines to `23` and update the comment block
above the `target_compile_features` call.

**Commit:** `eb510c1` ("build: require cxx_std_20 instead of
cxx_std_23")

## 2. `std::bit_cast` → `__builtin_bit_cast` fallback

**File:** `src/cpp/backends/swar.hpp`, lines ~39-50

**Why:** libstdc++ 10 has the language builtin `__builtin_bit_cast`
but not the `std::bit_cast` template wrapper. Added a
`stride_align_bit_cast<To>(from)` helper gated on
`__cpp_lib_bit_cast >= 201806L`.

**Revert:** Delete the `stride_align_bit_cast` helper template and
restore both call sites to `std::bit_cast<Lane>(raw)` /
`std::bit_cast<RawLane>(value)`.

**Commit:** `6853b8c` ("swar: route bit_cast through a helper that
falls back to __builtin_bit_cast")

## 3. `std::make_unique_for_overwrite` → `new TraceCell[n]` fallback

**File:** `src/cpp/backends/profile_traceback.hpp`, around line 91 in
the `TraceTable` constructor

**Why:** libstdc++ 10 doesn't have `std::make_unique_for_overwrite`
(C++20 P1020). The fallback uses `new TraceCell[n]` under
`__cpp_lib_smart_ptr_for_overwrite` guard. Semantic difference: the
fallback value-initializes the cells (a zero-fill we don't strictly
need since the DP rewrites them), but that's not on a hot path for
the gcc-10 build.

**Revert:** Replace the `#if/#else/#endif` block with the original
unconditional `: data_(std::make_unique_for_overwrite<TraceCell[]>(
std::max<std::size_t>(cell_count, 1U))) {}` and delete the comment
block.

**Commit:** `080c01a` ("profile_traceback + docs: gcc-10 fallback for
make_unique_for_overwrite")

## How to revert all three after gcc 16 ships

When the new toolchain is on POWER8:

```sh
# Identify the commits to revert
git log --oneline -- \
    CMakeLists.txt \
    src/cpp/backends/swar.hpp \
    src/cpp/backends/profile_traceback.hpp | grep -i "gcc-10\|cxx_std_20\|bit_cast\|make_unique_for_overwrite"

# Revert each
git revert <eb510c1> <6853b8c> <make_unique_for_overwrite-commit-if-any>

# Verify on POWER8 with gcc 16:
ssh power8 'cd /home/kxpoem/dev/stride-align && rm -rf build && \
    .venv/bin/pip install -e . --no-build-isolation \
    --config-settings=cmake.define.STRIDE_ALIGN_STATIC_CXX_RUNTIME=ON && \
    .venv/bin/python -m pytest tests/'
```

(`STRIDE_ALIGN_STATIC_CXX_RUNTIME=ON` may also become unnecessary
once libc on the host catches up — re-check whether the runtime
needs `__libc_single_threaded` once you rebuild with gcc 16's
libstdc++.)
