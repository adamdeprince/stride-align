# Changelog

All notable changes to `stride-align` are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-05-24

### Added

* **Indel distance** (`Scorer.INDEL` / `Scorer.INDEL_NORMALIZED`).
  Levenshtein restricted to insertions and deletions; equivalent to
  `|a| + |b| - 2 * LCS(a, b)`. Bit-parallel single-word kernel uses
  the Allison-Dix (1986) recurrence; multi-word patterns fall back
  to scalar DP. Public API: `indel_score`, `indel_normalized_score`,
  `indel_scores`, `indel_normalized_scores`, `indel_top_k`,
  `indel_best`, and the corresponding normalized variants. Wired
  through every backend, `cdist`, `cdist_above_threshold`,
  `cdist_top_k`, and the function-reference dispatch in `extract`.

* **True (unrestricted) Damerau-Levenshtein**
  (`Scorer.TRUE_DAMERAU_LEVENSHTEIN` /
  `Scorer.TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED`). The unrestricted
  form where a single character may participate in multiple edits.
  Diverges from OSA on overlapping transpositions
  (e.g. `"ca"`→`"abc"`: OSA=3, true-DL=2). Scalar DP only; no
  bit-parallel kernel yet (Hyyrö 2003 exists but is significantly
  more complex than OSA's bit-parallel and rarely the bottleneck).
  Existing `Scorer.DAMERAU_LEVENSHTEIN` continues to refer to OSA —
  the API name is unchanged.

* **Length-difference pruning** for `cdist_above_threshold` and
  `cdist_top_k`. Each pair is gated by a closed-form upper bound on
  the achievable normalized similarity before any SIMD work runs;
  bounds are scorer-specific (`min/max` for Lev/OSA/true-DL,
  `(2 + min/max)/3` for Jaro, `2*min/(q+t)` for Indel, `1.0` if
  equal-length for Hamming).

* **`cdist_top_k` row-sort by query length, descending.** Longest
  queries processed first so close-length high-scoring pairs
  surface early and the shared `global_min_bound` atomic reaches a
  useful value before the short-query rows run.

* **Per-pair cutoff push-down into the SIMD kernels.** Myers
  (Levenshtein single-word + multi-word), OSA single-word, and the
  Hamming inner loop all bail when the running distance plus
  remaining-chars allowance proves the pair can't reach its cutoff;
  bailed lanes return the per-pair `cutoff + 1` sentinel.

* **`docs/adding-a-new-algorithm.md`**: grep-able checklist for the
  touch points (`Scorer` enum, runtime helpers, cdist switches,
  bindings, per-backend Implementation methods, tests) a new
  scorer / alignment algorithm / SIMD backend has to hit.

* **Python 3.9 support.** The three `match` blocks in the Python
  layer became dict lookups; `from __future__ import annotations`
  was already in place project-wide. `pyproject.toml`
  `requires-python = ">=3.9"`, classifiers extended.

### Changed

* **Lowered the build-time C++ requirement from C++23 to C++20.**
  The project doesn't actually use any C++23 library feature — the
  `cxx_std_23` setting was aspirational. Lowering it lets gcc 10
  toolchains build the project (POWER8 Ubuntu 20.04 ships gcc 9.4
  and 10.5). Two stdlib gaps in gcc-10 libstdc++ are bridged with
  feature-test-gated fallbacks (`std::bit_cast` →
  `__builtin_bit_cast`, `std::make_unique_for_overwrite` → plain
  `new T[n]`). See `docs/power8-gcc10-workarounds.md` for the full
  list and the revert recipe once gcc 16 lands.

### Fixed

* **`cdist_above_threshold` iterator on macOS and LoongArch64.** The
  end-of-stream signal previously used `throw nb::stop_iteration()`,
  which relies on cross-DSO RTTI matching for nanobind's
  `builtin_exception`. macOS's two-level namespace and at least one
  LoongArch toolchain configuration defeat that lookup, and the
  exception ended up routed through nanobind's generic
  `std::exception` translator → bare `RuntimeError` instead of
  Python's `StopIteration`. Replaced with the C-API path
  (`PyErr_SetNone(PyExc_StopIteration)` plus a null `nb::object`
  return), which bypasses C++ exception machinery entirely. Fixes
  91 macOS test failures.

## [0.2.0] - 2026-05-19

Backfilled from git history; this entry was not in the tree at
the v0.2.0 tag.

### Added

* **Levenshtein scoring** (`Scorer.LEVENSHTEIN` /
  `Scorer.LEVENSHTEIN_NORMALIZED`). Myers (1999) bit-parallel
  scalar reference plus a SIMD batch kernel (one target per 64-bit
  lane) on every backend. Single-word path for patterns ≤ 64
  characters; multi-word kernel (W=2/3/4) for 65–256.
* `score_cutoff` parameter on the Levenshtein and per-target Lev
  scores APIs — bails per-target once the lower-bound score
  exceeds the cutoff; results that exceed the cap come back as
  `cutoff + 1` (rapidfuzz convention).
* **Damerau-Levenshtein (OSA-restricted, Hyyrö 2002)** —
  `Scorer.DAMERAU_LEVENSHTEIN` / `Scorer.DAMERAU_LEVENSHTEIN_NORMALIZED`.
  Scalar DP + bit-parallel scalar + SIMD batch on x86, NEON
  (Linux + macOS), SVE / SVE2, LSX / LASX, and PowerPC VSX. The
  "OSA-restricted" form is what rapidfuzz exposes as `OSA.distance`
  and is what most callers asking for "Damerau-Levenshtein"
  actually want.
* Cross-architecture benchmark sweeps for Levenshtein and OSA
  recorded under `benchmarks/` and summarized in
  [BENCHMARK.md](BENCHMARK.md). Highlight: AVX-512 LEV /
  DAML between 3.0x – 4.2x rapidfuzz on short targets; Mac M4
  NEON 5.5x – 8.5x python-Levenshtein.

### Changed

* LSX / LASX `vandn` semantics corrected to Intel-style `(~a) & b`
  (the LoongArch naming previously implied the opposite operand
  order). Affects the LSX / LASX backends only.
* `tools/benchmark_libs.py` extended with Levenshtein columns
  (`rapidfuzz`, `editdistance`, `Levenshtein`).
* New `tools/correctness_check.py` script.
* README localizations and the language carousel were dropped (English
  only for this release); the HTML build was regenerated to match.
  README now documents the LoongArch64 wheel sideload from the GitHub
  release (PyPI does not accept the `linux_loongarch64` platform tag).

## [0.1.0] - 2026-05-18

Initial public release.

### Added

* **Smith-Waterman and Needleman-Wunsch** sequence alignment with
  a nanobind C++23 backend and runtime SIMD dispatch.
* Backends:
  * x86: SSE4.1, AVX2, AVX-512 BWVL, AVX10-256, AVX10-512
  * ARM: Linux NEON / ASIMD, SVE, SVE2; macOS arm64 NEON
  * LoongArch (Loongson): LSX, LASX
  * PowerPC64: VSX
  * RISC-V: RVV (stub)
  * Portable: SWAR + pure-Python fallback
* Public API:
  * `smith_waterman_score` / `needleman_wunsch_score` /
    `smith_waterman_farrar_score`
  * Plural `*_scores` returning zero-copy `numpy.ndarray[int64]`
  * `*_normalized_score` / `*_normalized_scores` returning
    `float64` (length-normalized similarity in [0, 1])
  * Path / path-info / CIGAR variants for both SW and NW
  * Affine and linear gap models, score widths 8 / 16 / 32 / 64
  * String, bytes, and arbitrary-token sequence inputs
* CIGAR output uses the extended SAM convention (`=` sequence
  match, `X` mismatch, `I` / `D` indel). `build_cigar` /
  `ReverseCigarBuilder` emit digits via `std::to_chars` into a
  stack buffer with pre-reserved capacity (1.2x–2.2x speedup per
  row over the naive formulation).
* Benchmarks vs parasail (geomean across 80-row sweeps, 2026-05-18):
  Intel AVX-512 BWVL **1.752x**, AVX2 1.377x; Graviton4 NEON
  1.138x; Mac M4 NEON 1.065x; Loongson LASX **4.909x** (vs
  generic), **7.517x** (vs patched parasail, 1:1); Power8 VSX
  **3.772x**.
* Docs / tooling:
  * README in 16 languages with RTL-ready CSS (en, zh-CN, zh-TW,
    ja, de, ko, fr, es, pt-BR, ru, vi, id, hi, ar, tr, pl) — note:
    the translations were dropped again in v0.2.0 and the
    Simplified Chinese reintroduced in v0.3.0.
  * Themed `html/` rendition of every README + BENCHMARK.
  * `BENCHMARK.md` cross-architecture writeup.
  * Two runnable demos (Bible-verse nearest match + spell checker).
