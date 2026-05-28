# TODO: Dynamic Time Warping (DTW) and adjacent numeric-sequence kernels

**Status:** approved scope, ready to implement in phases.
**Goal:** add SIMD-accelerated numeric-sequence distance kernels — DTW
first, then subsequence DTW, LCSS, ERP, Fréchet — to stride-align,
sharing the existing Python dispatch / backend selection / numpy
ndarray validation infrastructure.

## Scope (v1 = phase C.1)

* **Algorithm**: full DTW, score-only (no warping path yet).
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

| Phase | Deliverable | Files touched |
| --- | --- | --- |
| **C.1a** | Scalar reference, Python API, dispatcher, dtype rejection, full test suite (correctness + Sakoe-Chiba edge cases) | `include/stride_align/dtw.hpp`, `src/cpp/dtw_dispatch.hpp`, `src/stride_align/__init__.py`, `tests/test_dtw.py`, `src/cpp/module_bindings.hpp` |
| **C.1b** | x86 SIMD batch kernels (float32 / float64 / int16) on sse41 / avx2 / avx512bwvl / avx10_256 / avx10_512 | `src/cpp/dtw_simd_ops.hpp` (x86 specialisations), `src/cpp/dtw_simd.hpp`, per-backend wiring |
| **C.1c** | ARM NEON + Loongson LSX/LASX SIMD batch kernels | additional `dtw_simd_ops.hpp` specialisations, per-backend wiring |
| **C.1d** | Benchmarks vs `dtaidistance`, `tslearn.metrics.dtw`, `dtw-python`, audio-domain int16 sanity vs float32 | `tools/refresh_dtw_benchmarks.py`, `benchmarks/intel-dtw-<date>.csv` |

Subsequent phases (C.2 = LB_Keogh, C.3 = subsequence DTW, C.4 =
LCSS+ERP, C.5 = warping path, C.6 = Fréchet, C.7 = multidim) each
sit on top of C.1 without needing kernel rewrites.

## Open API decisions to settle before C.1a starts

1. **Function naming**: `sa.dtw()` vs `sa.dtw_distance()`. The
   existing string surface uses `levenshtein_score` /
   `levenshtein_distance` interchangeably. I'd lean `sa.dtw()` for
   the singular and `sa.dtw_distances()` for the batch to match the
   `levenshtein_score` / `levenshtein_scores` convention, even
   though "score" is a misnomer here (lower is better, no "score"
   semantic). Open to bikeshedding.

2. **`window` units**: integer = absolute radius in samples, float
   = fraction of `max(|q|, |t|)`. Or rename to two kwargs
   (`window_samples` / `window_ratio`) for clarity? Existing
   convention in the field is the single `window` kwarg with
   overload.

3. **`distance` defaults**: `"l1"` is safe for int16; `"l2_squared"`
   is conventional for float DTW. Pick a per-dtype default
   (`"l1"` for int16, `"l2_squared"` for float) or one universal
   default (`"l1"` everywhere, requiring float users to opt into
   `"l2_squared"`)? I'd lean per-dtype default — matches what
   users on each domain expect — but happy to standardise on
   `"l1"` if you'd rather.

4. **Empty-sequence semantics**: `dtw([], [])` = 0?
   `dtw([], non_empty)` = `+inf`? Or raise? Convention is
   inconsistent across libraries; pick one and document.

## Related

- [docs/TODO-wide-traceback.md](TODO-wide-traceback.md) — adjacent
  effort on the string-alignment side.
- [docs/TODO-matrix-roadmap.md](TODO-matrix-roadmap.md) — same
  shape for the substitution-matrix work.
- [memory/feedback_short_strings_priority.md](../memory) — DTW is
  a different priority lane from the short-string fuzzy matching
  hot path; v1 should not regress short-string Levenshtein
  throughput.
