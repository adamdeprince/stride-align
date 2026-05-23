# Adding a new algorithm to stride-align

A grep-able checklist for the touch points a new algorithm has to hit
before it's safe to ship. The library bundles three loosely-coupled
things, each with its own integration surface:

* **Scorers** — pairwise string distance / similarity (Levenshtein,
  Damerau-Levenshtein, Hamming, Jaro, Jaro-Winkler). These plug into
  the `Scorer` enum, the cdist variants, top-k, the function-reference
  dispatch table, and the Python wrapper module.
* **Alignment algorithms** — SW / NW / Farrar SW. These are wired
  through the per-backend `Implementation` bundle and the alignment-
  specific bindings; they do **not** go through the `Scorer` enum.
* **SIMD backends** — a new ISA / architecture. The algorithms don't
  change; you're publishing a new `Implementation<BackendKind::X>`
  module and teaching the runtime to pick it.

Most "I'm adding a new algorithm" tasks mean **scorer**. The bulk of
this doc is the scorer checklist; the other two cases get shorter
sections at the end.

If you skip a step here, the library will probably still compile —
backends fall back gracefully, the Python module imports something
that "works" — and the regression won't show up until someone uses
the new code through `cdist_top_k` or `extract()` and gets the wrong
answer or a runtime crash. Walk the whole list.

---

## Adding a new scorer

### 1. Public C++ header — `include/stride_align/{name}.hpp`

* Scalar reference impl, templated on `Token`, used as both the
  correctness oracle for SIMD and the fallback path when inputs aren't
  byte-compatible.
* `normalize(distance, denom)` helper (for distance scorers) or a clear
  contract for the [0, 1] mapping (for similarity scorers).
* Any constants the SIMD path needs to share with the binding layer
  (e.g. `kDefaultPrefixWeight` for Jaro-Winkler).

### 2. SIMD kernel — `src/cpp/{name}_simd.hpp`

* `template <typename Ops> {name}_scores_simd_raw(...)` — batch of one
  query vs many targets, no normalization.
* `template <typename Ops> {name}_scores_simd(...)` — convenience
  wrapper that handles the size sweep and any post-processing.
* If the algorithm needs a fundamentally new SIMD primitive (e.g. a
  per-byte minimum reduction the existing `Ops` bundle doesn't expose),
  add it to `levenshtein_simd_ops.hpp` or the analogous ops header
  **and** add the corresponding implementation to every backend kernel
  header under `src/cpp/backends/` (`x86_*.hpp`, `arm_neon_kernel.hpp`,
  `arm_sve_kernel.hpp`, `powerpc_vsx_kernel.hpp`,
  `loongarch_fixed_kernel.hpp`, `riscv_rvv_kernel.hpp`). Missing one
  Ops member on a single backend is a silent compile error in that
  backend only — verify with a build that exercises every arch.

### 3. Per-pair dispatch — `src/cpp/{name}_dispatch.hpp`

* `dispatch_score(query, target)` and `dispatch_scores(query, targets)`
  — the non-SIMD entry points used when inputs aren't byte-compatible
  or when no SIMD backend is available.

### 4. Backend `Implementation` bundles — `src/cpp/backends/*.hpp`

Every backend bundle (`x86_sse41.hpp`, `x86_avx2.hpp`,
`x86_avx512bwvl.hpp`, `x86_avx10_256.hpp`, `x86_avx10_512.hpp`,
`linux_aarch64_neon.hpp`, `linux_aarch64_sve.hpp`,
`linux_aarch64_sve2.hpp`, `macos_arm64_neon.hpp`, `arm_neon128.hpp`,
`linux_loongarch64_lsx.hpp`, `linux_loongarch64_lasx.hpp`,
`linux_powerpc64_vsx.hpp`, `linux_riscv64_rvv.hpp`) exposes a static
method per scorer:

```cpp
static std::vector<double> {name}_similarities(...);  // or *_scores
```

If you don't add the method to every bundle, the binding layer's
`if constexpr (requires { ... })` falls back to the scalar dispatch on
that backend silently — the user installs the wheel and notices that
the new scorer is 10× slower on their machine than it should be.

### 5. `Scorer` enum — keep C++ and Python in sync

* `src/cpp/topk.hpp` — append to the `Scorer` enum class with the
  **next** integer (don't reuse a previously-deleted value, the
  integer is part of the user-visible API contract per the comment at
  the top of the enum).
* `src/stride_align/__init__.py` — append the matching uppercase entry
  to the `Scorer(enum.IntEnum)` class with the same integer.

### 6. `cdist_runtime.hpp` helpers

Each of these has a `switch (scorer)` that the compiler will warn on
when a case is missing (we build with `-Wall -Wextra -Wpedantic`).
Don't dismiss the warning:

* `scorer_returns_double` — `true` for similarity / normalized, `false`
  for raw distance.
* `scorer_is_normalized` — currently delegates to the above; keep them
  consistent if you split them later.
* `diagonal_double` / `diagonal_int` — the self-similarity value used
  to fill the diagonal in symmetric cdist mode (1.0 for similarities,
  0 for distances).
* `max_normalized_similarity` — **closed-form upper bound** for length-
  difference pruning in `cdist_above_threshold` and `cdist_top_k`. Must
  be tight (achievable when `q == t`) and safe (never exceeds the
  value the actual kernel could return). Distance-only scorers can
  return 1.0 here and skip pruning. See the comments in
  `cdist_runtime.hpp` for examples (Lev/JW derivations).

### 7. cdist row dispatch — `src/cpp/cdist_simd.hpp`

* `compute_row_int` switch — add a case for distance variants.
* `compute_row_double` switch — add a case for similarity / normalized
  variants.

The default branch raises `RuntimeError` with the scorer integer —
useful but only fires if you forgot the case.

### 8. Structural validators

If the scorer imposes a structural constraint on inputs (Hamming
requires equal lengths, anything else might too), add a validator next
to `validate_hamming_lengths` in `cdist_runtime.hpp` and call it from
each cdist variant's entry point. Otherwise the SIMD path will return
garbage or read past a buffer.

### 9. Python bindings — `src/cpp/module_bindings.hpp`

Per-scorer `m.def(...)` blocks for:

* `{name}_score(query, target)` — single pair
* `{name}_normalized_score(query, target)` — single pair, [0, 1]
* `{name}_scores(query, targets)` — 1×N batch
* `{name}_normalized_scores(query, targets)` — 1×N batch, [0, 1]
* `{name}_top_k(query, targets, k)` — heap over the batch
* `{name}_normalized_top_k(query, targets, k)` — heap, normalized

Plus the function-reference → `Scorer` enum registration so callers
can pass `sa.{name}_scores` instead of `Scorer.NAME`:

```cpp
register_attr("{name}_scores", Scorer::Name);
register_attr("{name}_normalized_scores", Scorer::NameNormalized);
```

This is what makes `cdist(qs, ts, scorer=sa.jaro_similarities)` work
identically to `scorer=sa.Scorer.JARO`.

### 10. Python wrappers — `src/stride_align/__init__.py`

* Public functions: `{name}_score`, `{name}_normalized_score`,
  `{name}_scores`, `{name}_normalized_scores`, `{name}_top_k`,
  `{name}_normalized_top_k`, `{name}_best`, `{name}_normalized_best`.
* `Scorer.NAME` and `Scorer.NAME_NORMALIZED` enum entries (matching
  the C++ side).
* `_register_top_level_scorers()` — add both the raw and normalized
  variants to the `pairs` list. This is what registers them with the
  C++ scorer table so the function-reference dispatch works at the
  cdist boundary.
* `__all__` — add every new top-level name.

### 11. Pure-Python fallback — `src/stride_align/_pybackend.py`

Optional but recommended: Levenshtein has one, the others don't, which
means stride-align is unusable for those scorers on architectures
without a C++ backend. If you want the scorer to work on every
platform (including ones we haven't built a wheel for yet), add a
scalar reference implementation here. It's also the obvious place to
read when a future contributor needs an unambiguous spec for what the
scorer is supposed to compute.

### 12. Tests

The library has a strong convention that every scorer is tested
through every public path. Mirror what exists for an existing scorer:

* `tests/test_{name}.py` — scalar correctness, edge cases (empty
  strings, length 1, unicode), comparison vs the scalar reference.
* `tests/test_cdist.py` — add the scorer to the parametrized
  full-matrix test against the scalar reference.
* `tests/test_cdist_above_threshold.py` — at least one parametrized
  case if the scorer is normalized.
* `tests/test_cdist_top_k.py` — at least one parametrized case.
* `tests/test_cdist_length_pruning.py` — add to `PRUNED_SCORERS` if
  the new scorer is normalized **and** has a useful length-difference
  bound (i.e. step 6's `max_normalized_similarity` returns something
  tighter than 1.0).
* `tests/test_top_k.py` — `{name}_top_k` and `{name}_best`.
* `tests/test_api.py` — import-and-call smoke test for every public
  name added in step 10.

### 13. Cross-arch validation

The CI matrix doesn't cover every architecture we ship to (POWER8,
RISC-V, LoongArch in particular get hand-tested). Before shipping a
new scorer:

* Run `pytest tests/` on at least one non-x86 target.
* If you added new `Ops` primitives in step 2, verify a build on every
  architecture in `src/cpp/backends/` actually compiles and tests
  pass. A silent `if constexpr (requires { ... })` fallback hides a
  missing SIMD method as "scalar fallback" rather than a build error.

### 14. Documentation

* `README.md` — the algorithm table near the top, the "API" section,
  and (if it's a notable algorithm) a short example.
* `CITATION.bib` — the canonical reference paper for the algorithm.
* `BENCHMARK.md` — only if you add a benchmark column comparing
  against another library (rapidfuzz, Levenshtein, etc.).

---

## Adding a new alignment algorithm (SW / NW variant)

Alignment algorithms (Smith-Waterman, Needleman-Wunsch, Farrar SW)
don't go through the `Scorer` enum. They're wired through the
per-backend `Implementation` bundle directly.

1. Add the static methods to every `Implementation<BackendKind::*>` in
   `src/cpp/backends/*.hpp` and to the generic backend in
   `src/cpp/backends/generic.hpp`. The same warning from scorers
   applies: missing on one backend = silent scalar fallback.
2. SIMD kernel headers under `src/cpp/backends/` need the new kernel
   per ISA family (`x86_fixed_kernel.hpp`, `arm_neon_kernel.hpp`,
   `arm_sve_kernel.hpp`, etc.).
3. Python bindings — `m.def("smith_waterman_X_score", ...)` etc. in
   `src/cpp/module_bindings.hpp`.
4. Python wrappers and `__all__` — `src/stride_align/__init__.py`.
5. Pure-Python reference in `_pybackend.py` — strongly recommended,
   this is the spec for what the C++ should compute.
6. Tests — same shape as the scorer checklist (scalar correctness,
   per-arch parity, the top-k/best wrappers).

There's **no length-pruning bound** for alignment algorithms because
they don't go through the cdist variants. Skip step 6 from the scorer
checklist.

---

## Adding a new SIMD backend (architecture)

You're publishing a new `Implementation<BackendKind::X>` module; the
algorithms themselves don't change.

1. **`src/cpp/backends/{platform_arch_isa}.hpp`** — the `Implementation`
   struct exposing every scorer + alignment method, using the
   appropriate kernel header.
2. **`src/cpp/backends/{platform_arch_isa}.cpp`** — one-line
   `NB_MODULE(_name, m)` that calls
   `bindings::bind_backend_module<Implementation>(m, "...")`.
3. **CMakeLists.txt** — `add_stride_align_backend(_name src/cpp/backends/X.cpp)`
   gated by the appropriate `CMAKE_SYSTEM_PROCESSOR` check and ISA
   feature flag. Also add to `STRIDE_ALIGN_CPU_COMPILE_DEFINITIONS`.
4. **`src/cpp/cpu.cpp` / `cpu.hpp`** — runtime detection of the new
   ISA flag (CPUID on x86, HWCAP on Linux ARM, /proc/cpuinfo elsewhere).
5. **`src/stride_align/_cpu.py`** — the `BackendKind` enum and its
   `_backend_module_name()` mapping in `__init__.py`.
6. **`src/stride_align/__init__.py`** — `_select_backend()` priority
   order. Put the new backend ahead of any backend it strictly
   supersedes (e.g. AVX-512 over AVX2 on systems with both).
7. **Wheels** — `pyproject.toml` cibuildwheel config if you want
   prebuilt wheels for the new arch.
8. **Tests** — `pytest tests/` on real hardware; force-select the new
   backend via `BackendKind.X` if you need to bypass auto-detection.
9. **Benchmarks** — produce a CSV with `tools/benchmark_libs.py` and
   commit it under `benchmarks/{arch}-{date}.csv`.
10. **Docs** — `README.md` supported-architecture list.

---

## Things that catch you out

* **Scorer enum integer values are user-visible.** The Python IntEnum
  and the C++ enum class must agree on integer values, and those
  integers are committed to as an API contract. Append, don't reuse
  deleted values.
* **The function-reference table is what makes `scorer=sa.jaro_similarities`
  work.** Forgetting `register_attr` in step 9 means the only way to
  invoke the new scorer through `cdist` is the enum, and that's not
  caught by any existing test — add a `test_function_reference_scorer`
  parametrized over the new name.
* **`if constexpr (requires { ... })` is a silent fallback.** It's
  there for backends that legitimately don't implement a method, but
  it also masks "I forgot to add this method to the AVX2 backend." A
  CI matrix that runs on every arch would catch this; until we have
  that, audit by hand.
* **Length-pruning is correctness-load-bearing.** A bound that's too
  loose (returning 1.0 always) just disables pruning — fine. A bound
  that's too tight (returning something *below* what the kernel can
  actually return for that pair) drops valid results from cdist
  outputs. The tests in `test_cdist_length_pruning.py` check this by
  comparing against the un-pruned full cdist; add the new scorer
  there.
* **GIL release isn't optional for batches.** Every cdist variant
  releases the GIL during compute and reacquires only for tqdm
  updates. If a new SIMD scorer doesn't release the GIL it'll block
  every other Python thread; the `test_gil_released_during_compute`
  pattern in the cdist tests covers this.
* **Symmetric optimization.** `cdist(qs, qs, ...)` does only the strict
  upper triangle. If the scorer is asymmetric (`score(a, b) != score(b, a)`),
  this would be wrong — but all current scorers are symmetric. If you
  add an asymmetric one, the symmetric path needs to be disabled for
  it, and that's a runtime check in the `cdist` entry point, not just
  a comment.
