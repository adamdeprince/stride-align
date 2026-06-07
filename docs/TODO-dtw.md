# TODO: Dynamic Time Warping (DTW) and adjacent numeric-sequence kernels

**Status:** **ON HOLD** at the user's direction (2026-06-08).
**C.1a scalar reference is shipped** (`sa.dtw`, `sa.dtw_distances`);
SIMD batch kernels (C.1b/c) and downstream phases (C.2 LB_Keogh, C.3
subsequence DTW, C.4 LCSS+ERP, C.5 warping path, C.6 Fréchet, C.7
multidim) are not in progress. The Phase D character-similarity
gap-filling work that previously gated this phase is now complete
(see below); the only thing keeping Phase C paused is the explicit
hold. Resume when the user says so — nothing in the codebase blocks
restart.

**Goal (unchanged):** add SIMD-accelerated numeric-sequence distance
kernels — DTW first, then subsequence DTW, LCSS, ERP, Fréchet — to
stride-align, sharing the existing Python dispatch / backend
selection / numpy ndarray validation infrastructure.

## Already shipped (Phase C.1a)

`sa.dtw` and `sa.dtw_distances` ship the scalar reference kernel
under the API decisions resolved at the bottom of this doc:

* **Naming:** `sa.dtw(query, target)` for the singular,
  `sa.dtw_distances(query, targets)` for the batch.
  Matches the existing `levenshtein_score` / `levenshtein_scores`
  pluralisation.
* **`window=` units:** `None` (unconstrained), `int` (absolute
  Sakoe-Chiba radius), or `float` in `(0, 1]` (fraction of
  `max(len(query), len(target))`). Single overloaded kwarg.
* **`distance=` default:** per-dtype — `"l1"` for `int16`,
  `"l2_squared"` for `float32` / `float64`. Pass an explicit string
  to override.
* **Empty inputs:** raise `ValueError` (not `0` or `+inf`).
* **dtype constraint:** `float32`, `float64`, or `int16`; both
  sides must share dtype; `bytes` / `str` / sequence rejected.

C.1a covers correctness, the Python dispatcher, dtype rejection,
empty-input handling, and the test suite (`tests/test_dtw.py`,
219 lines). The scalar reference is what every SIMD kernel will be
cross-checked against in C.1b/c.

## Outstanding scope (Phase C.1b and beyond)

## Scope (v1 = phase C.1)

* **Algorithm**: full DTW, score-only (no warping path yet).
  C.1a (scalar) shipped; C.1b/c/d listed under "Phase split" below.
* **Constraints**: Sakoe-Chiba band as a `window` kwarg
  (`r`-fraction of `max(|q|, |t|)`); unconstrained DTW when `window`
  is `None`.
* **Distance functions**:
  - `"l1"` — `|x - y|` (default; safest for int16).
  - `"l2_squared"` — `(x - y)^2` (returns squared DTW; user can
    `sqrt()` the result if they need a metric).
* **Token / Cell types**:

  | Token (input dtype) | Cell (DP accumulator) | Why |
  | --- | --- | --- |
  | `np.float32` | `float` | scientific / ML default |
  | `np.float64` | `double` | high precision, sensor calibration |
  | `np.int16` | `std::int32_t` | audio (PCM), ADC sensors |

  Cell widens from int16 → int32 so the DP accumulator can't
  overflow on reasonable-length sequences (worst case
  `65535 * len < 2^31` so `len < ~33 000`; beyond that we'd add a
  saturating-int32 variant or reject).

* **Multidimensional**: 1D only in v1. d-dim is phase C.7 — leave
  the API door open via `query: np.ndarray` (2-D allowed at the
  signature level, but raises NotImplementedError until C.7).

* **Excluded from v1**: int8, int32, int64, float16; warping path;
  multi-channel; FastDTW; PrunedDTW; LB_Keogh (phase C.2).

## Python API (v1)

```python
import stride_align as sa

# Singular: 1 query vs 1 target.
sa.dtw(query, target, *, window=None, distance="l1") -> float
sa.dtw_subsequence(query, target, *, ...) -> tuple[float, int]  # phase C.3
sa.lcss(query, target, *, epsilon, delta=None) -> float          # phase C.4
sa.erp(query, target, *, g, distance="l1") -> float              # phase C.4
sa.frechet(query, target) -> float                                # phase C.6

# Batch: 1 query vs N targets.
sa.dtw_distances(query, targets, *, window=None, distance="l1") -> np.ndarray
```

`query` and each target are `np.ndarray` of dtype `float32`,
`float64`, or `int16` (sequences / bytes / str are rejected — DTW
isn't defined on tokens). All elements of a batch must share dtype
(mirrors the existing wide-Farrar rule). `window` accepts `None`,
an integer (absolute Sakoe-Chiba radius), or a float in `(0, 1]`
(fraction of `max(|q|, |t|)`).

### Errors

* Wrong dtype → `TypeError("dtw requires float32, float64, or int16 ndarray; got <dtype>")`.
* Mixed dtypes in batch → `TypeError` (same as existing wide batch).
* 2-D query/target in v1 → `NotImplementedError("multidimensional DTW arrives in phase C.7; flatten or wait")`.
* `window` ≤ 0 → `ValueError`.

## C++ surface

```
include/stride_align/dtw.hpp
    namespace stride_align::dtw {
      enum class DistanceKind { kL1, kL2Squared };

      // Scalar reference: correctness oracle + fallback when SIMD
      // backend isn't available.
      template <typename Cell>
      Score dtw_score_scalar(std::span<const Cell> query,
                             std::span<const Cell> target,
                             DistanceKind dist,
                             std::optional<std::size_t> window);

      // Same shape, parameterised on the Token type used to read
      // the input buffer (so int16 inputs widen to int32 cells
      // inside the kernel).
      template <typename Token, typename Cell>
      Score dtw_score_scalar_widening(std::span<const Token> query,
                                      std::span<const Token> target,
                                      DistanceKind dist,
                                      std::optional<std::size_t> window);
    }

src/cpp/dtw_simd_ops.hpp
    namespace stride_align::dtw {
      // Mirrors the existing farrar_fixed_kernel SimdOps shape so
      // future Token/Cell pairs slot in without restructuring.
      // v1 instantiations:
      //   SimdOps<float32, float32>
      //   SimdOps<float64, float64>
      //   SimdOps<int16,   int32>     // widening for overflow safety
      //
      // Each specialisation exposes:
      //   vector_type, lane_count
      //   Ops::load / store / loadu_token (token-width load that
      //     promotes to Cell on the int16 path)
      //   Ops::min(a, b)            // float min / int32 min
      //   Ops::add(a, b)            // float add / int32 add
      //   Ops::abs_diff(a, b)       // |a - b| for L1
      //   Ops::diff_squared(a, b)   // (a-b)^2 for L2_squared
      //   Ops::infinity()           // FLT_MAX / DBL_MAX / INT32_MAX
      //   Ops::reduce_min(vec)
      template <typename Token, typename Cell>
      struct SimdOps;  // primary template, undefined; specialisations follow
    }

src/cpp/dtw_simd.hpp
    template <template <typename, typename> class OpsTemplate,
              typename Token, typename Cell>
    std::vector<Score> dtw_batch_simd(
        std::span<const Token> query,
        std::span<const std::span<const Token>> targets,
        DistanceKind dist,
        std::optional<std::size_t> window);

src/cpp/dtw_dispatch.hpp
    inline Score dispatch_dtw(nb::handle query, nb::handle target,
                              DistanceKind dist,
                              std::optional<std::size_t> window);
    inline std::vector<Score> dispatch_dtw_many(nb::handle query,
                                                nb::handle targets,
                                                DistanceKind dist,
                                                std::optional<std::size_t> window);
```

`Score` is `double` end-to-end (the C++ DP runs in `Cell`, but the
Python-facing result widens to `double` so the int16 kernel can
return distances larger than `INT32_MAX` without a separate
return-type path).

## SIMD batch shape

DTW's natural batch lane assignment is **one target per SIMD lane**,
the same shape stride-align already uses for the Levenshtein SIMD
batch (`levenshtein_simd.hpp`):

* The DP column for query position `i` reads `query[i]` (scalar,
  broadcast to all lanes) and the per-lane current target sample.
* Each lane runs its own DP row in lockstep with the others.
* Lanes finish in lockstep (no early-exit per lane in v1; the
  cutoff / `LB_Keogh` early-exit is phase C.2).

Lane counts (Token width):

| Backend | float32 | float64 | int16 (Cell=int32) |
| --- | ---: | ---: | ---: |
| SSE4.1 | 4 | 2 | 4 |
| AVX2 | 8 | 4 | 8 |
| AVX-512BWVL / AVX10/512 | 16 | 8 | 16 |
| AVX10/256 | 8 | 4 | 8 |
| ARM NEON 128 | 4 | 2 | 4 |
| Loongson LSX | 4 | 2 | 4 |
| Loongson LASX | 8 | 4 | 8 |

(The int16-cells-stay-int16 throughput-maximal variant — 16-lane
int16 / 32-lane in AVX-512 — is a phase C.N optimisation, not v1.
v1 widens to int32 for overflow safety.)

## Audio specifics (why int16 v1 is non-negotiable)

* PCM audio is int16 native (44.1 kHz × 16-bit mono = ~88 KB/s).
* Reading audio frames through numpy gives int16 ndarrays for
  free; converting to float32 for DTW costs a cache-pass + a
  type-cast pipeline at the boundary.
* Per-cell distance on int16 PCM: max `|q - t|` is 65535. For an
  L1 DTW with query length ~1000 (≈23 ms at 44.1 kHz), max
  accumulated distance is ~6.5×10^7, comfortably in int32.
* L2-squared is risky for raw PCM (65535² × 1000 = 4.3×10^12,
  overflows int32 around length ~470). v1 supports L2-squared on
  int16 only when `window` + length combinations stay safe — the
  dispatcher computes the worst-case accumulator and rejects with
  a clear "use L1 or float32" message.

## Python dispatch

The dispatcher mirrors the existing wide-Farrar routing in
`src/stride_align/__init__.py`:

* If query/target are ndarray with dtype in `{float32, float64,
  int16}` → DTW SIMD batch on the best available x86 / NEON / LSX /
  LASX backend.
* Other dtypes → `TypeError`.
* Sequence / bytes / str → `TypeError` (not a numeric domain).
* SVE / SVE2 / RVV / Power VSX backends are not wired in v1; on
  those platforms the Python dispatcher prefers the NEON / VSX
  fallback if available, else the scalar reference. (Same
  `_FIXED_KERNEL_WIDE_PRIORITY` trick.)

## Phase split (work breakdown)

| Phase | Deliverable | Status | Files touched |
| --- | --- | --- | --- |
| **C.1a** | Scalar reference, Python API, dispatcher, dtype rejection, full test suite (correctness + Sakoe-Chiba edge cases) | ✅ shipped | `include/stride_align/dtw.hpp`, `src/cpp/dtw_dispatch.hpp`, `src/stride_align/__init__.py`, `tests/test_dtw.py`, `src/cpp/module_bindings.hpp` |
| **C.1b** | x86 SIMD batch kernels (float32 / float64 / int16) on sse41 / avx2 / avx512bwvl / avx10_256 / avx10_512 | on hold | `src/cpp/dtw_simd_ops.hpp` (x86 specialisations), `src/cpp/dtw_simd.hpp`, per-backend wiring |
| **C.1c** | ARM NEON + Loongson LSX/LASX SIMD batch kernels | on hold | additional `dtw_simd_ops.hpp` specialisations, per-backend wiring |
| **C.1d** | Benchmarks vs `dtaidistance`, `tslearn.metrics.dtw`, `dtw-python`, audio-domain int16 sanity vs float32 | on hold | `tools/refresh_dtw_benchmarks.py`, `benchmarks/intel-dtw-<date>.csv` |
| **C.2** | LB_Keogh lower-bound prune + early-exit cutoff for the batch path | on hold | new `lb_keogh.hpp`, dispatcher extension |
| **C.3** | Subsequence DTW (`sa.dtw_subsequence(query, target)` returning `(distance, start_index)`) | on hold | new `dtw_subsequence.hpp` |
| **C.4** | LCSS + ERP (`sa.lcss(query, target, epsilon=…)`, `sa.erp(query, target, g=…)`) | on hold | new `lcss.hpp`, `erp.hpp` |
| **C.5** | DTW warping-path traceback | on hold | extend `dtw.hpp` with a path-returning variant |
| **C.6** | Fréchet distance (`sa.frechet`) | on hold | new `frechet.hpp` |
| **C.7** | Multidimensional / multivariate inputs (the `query: np.ndarray` 2-D doors) | on hold | dispatcher + DP loop updates |

C.2-C.7 each sit on top of C.1 without needing kernel rewrites.

## API decisions — resolved during C.1a

These were open questions before C.1a shipped; the shipped
implementation pins each one. Listed here so a reader of this TODO
sees the API contract C.1b/c will need to honour.

1. **Function naming.** Resolved as `sa.dtw()` singular,
   `sa.dtw_distances()` batch. Matches the `levenshtein_score` /
   `levenshtein_scores` pluralisation convention.
2. **`window=` units.** Single overloaded kwarg: `None` (unconstrained),
   `int` (absolute Sakoe-Chiba radius), or `float` in `(0, 1]`
   (fraction of `max(len(query), len(target))`).
3. **`distance=` default.** Per-dtype: `"l1"` for `int16`,
   `"l2_squared"` for `float32` / `float64`. Caller can pass either
   string explicitly.
4. **Empty-sequence semantics.** `sa.dtw([], …)` raises `ValueError`.
   Neither `0` nor `+inf` is silently returned; callers must guard
   their inputs.

## When the hold lifts

The Phase D character-similarity gap-fillers (D.1 n-gram sets,
D.2 phonetic encoders, D.3 token-ratio family, D.4 LCS / LC
substring, D.5 Ratcliff-Obershelp, D.6 Monge-Elkan, D.7 extra
phonetics including Beider-Morse + Daitch-Mokotoff) all shipped on
main between 2026-05 and 2026-06. The SW/NW Unicode no-exception
test sweep also shipped (2026-06-08, commit `39316b6`). So the
gating dependency previously cited at the bottom of
`docs/TODO-character-similarity-phase-D.md` is satisfied; only the
user-direction hold remains.

When the hold lifts, the natural restart order is C.1b (x86 SIMD)
→ C.1c (ARM + Loongson SIMD) → C.1d (benchmarks) → C.2 (LB_Keogh
for early-exit on the batch path) → the rest.

## Related

- [docs/TODO-wide-traceback.md](TODO-wide-traceback.md) — adjacent
  effort on the string-alignment side.
- [docs/TODO-matrix-roadmap.md](TODO-matrix-roadmap.md) — same
  shape for the substitution-matrix work.
- [memory/feedback_short_strings_priority.md](../memory) — DTW is
  a different priority lane from the short-string fuzzy matching
  hot path; v1 should not regress short-string Levenshtein
  throughput.
