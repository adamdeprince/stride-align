# Changelog

All notable changes to `stride-align` are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

## [0.2.0]

(Existing behavior at this tag was not previously tracked in this
file; future releases will list specific deltas.)
