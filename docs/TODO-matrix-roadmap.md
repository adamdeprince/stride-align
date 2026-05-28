# TODO: matrix-mode roadmap (Phase 5+)

**Status:** parked, ongoing multi-phase effort.
**Goal:** finish the substitution-matrix story. Today the matrix
kernel handles **linear-gap SW/NW score-only** on a subset of x86
backends; everything else (affine, batch, traceback, more SIMD
backends, more bundled matrices) needs lifting.

This file consolidates the open items previously sketched in
`memory/project_matrix_roadmap.md` and the conversation history.

## What works today

* `matrix=` kwarg on `smith_waterman_score` and
  `needleman_wunsch_score` (singular) — linear gap, score-only.
* `matrix=` on `smith_waterman_scores` and `needleman_wunsch_scores`
  (batch, score-only, linear gap) — `_dispatch_matrix_many`.
* `matrix=` on the `_path` / `_path_info` / `_cigar` variants (single
  pair, linear gap) — `_dispatch_matrix_path_info`.
* Bundled matrices: BLOSUM45/50/62/80/90, PAM30/70/250.
* As of Phase B 2.9: wide ndarray inputs explicitly rejected with a
  clear TypeError; uint8/int8 ndarrays still accepted (each value an
  index into the matrix alphabet).

## Open items

### A. Propagate matrix kernels to the remaining SIMD backends

Currently the matrix-mode SIMD entry points are wired on the same
backends as Phase B 2.7 / 2.8 (sse41, avx2, avx512bwvl, lsx, lasx,
neon, avx10_256, avx10_512). Power/VSX is skipped pending the RAM
replacement. SVE/SVE2/RVV use the scalable kernel and would need
their own `matrix_score_state` mirror — same scope as the wide-token
deferral.

### B. Affine matrix scoring

`smith_waterman_affine_score(query, target, matrix=...)` and the NW
equivalent need to dispatch through a new
`_dispatch_matrix_affine` (already a thin wrapper) onto an
`affine_score_matrix` C++ entry. The Farrar `prepare_matrix_affine_
score_state` template exists in `farrar_fixed_kernel.hpp` but isn't
wired through all backends.

### C. Batch matrix (1 query × N targets)

`_dispatch_matrix_many` exists for linear; need
`_dispatch_matrix_affine_many` and the corresponding C++ batch
matrix entry. Pattern matches the wide_score_batch amortisation: one
shared query profile across all targets, per-target rewrite of
target_profile_offsets.

### D. Traceback / path / cigar with matrix=

`_dispatch_matrix_path_info` handles the linear-gap case; the affine
version is still TODO. The kernels exist in `farrar_fixed_kernel.hpp`
(`matrix_affine_path_info` etc.) but the dispatcher hasn't been
extended.

### E. More bundled matrices

Easy adds: BLOSUM30, BLOSUM35, BLOSUM40, BLOSUM55, BLOSUM65,
BLOSUM75, BLOSUM85, BLOSUM100. PAM10, PAM40, PAM80, PAM120, PAM160,
PAM200. Each is a ~5-minute task: copy the canonical NCBI matrix
values into `src/stride_align/matrices.py`, export, add a metadata
test similar to `test_blosum62_metadata`.

### F. Matrix correctness fuzzer

Compare matrix-mode scores against a reference implementation
(numpy DP with the matrix in float64) for randomized
matrix/query/target triples. Catches off-by-one and alphabet-index
bugs that the canonical-matrix tests miss.

## Recommended ordering

1. **B (affine matrix score)** — biggest user-visible feature gap;
   most callers using BLOSUM expect affine.
2. **C (batch matrix score)** — `_dispatch_matrix_many` is the
   batch entry users land on for 1-vs-many bioinformatics queries.
3. **D (matrix traceback affine)** — depends on B landing first.
4. **E (more matrices)** — independent, can interleave.
5. **A (SVE/SVE2/RVV propagation)** — lowest priority; the fixed-
   kernel backends already cover x86 + ARM NEON + Loongson + power.
6. **F (fuzzer)** — write once, run on every change.

## Scope notes

* B + C land together since the affine kernels exist; the work is
  wiring the per-backend dispatch and tests.
* D follows B because affine matrix traceback depends on B's affine
  matrix score.
* E (more bundled matrices) is just data + a metadata test per
  matrix; can interleave with anything.
* F (correctness fuzzer) is the small standalone item that pays
  back forever.
* A (SVE/SVE2/RVV propagation) is intentionally lowest priority —
  the fixed-kernel backends already cover x86 + ARM NEON + Loongson +
  Power.

## Related

- [memory/project_matrix_roadmap.md](../) — earlier sketch (memory).
- [src/cpp/backends/farrar_fixed_kernel.hpp](../src/cpp/backends/farrar_fixed_kernel.hpp)
  — `prepare_matrix_*` and `matrix_*` kernel templates.
- [src/stride_align/__init__.py](../src/stride_align/__init__.py)
  — `_dispatch_matrix*` dispatchers (look for the rejection helper
  `_reject_wide_ndarray_for_matrix` from Phase B 2.9).
