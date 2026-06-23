# Edit distance

Algorithms that count the cheapest sequence of single-character
operations needed to transform one sequence into another:
**Levenshtein**, **Damerau-Levenshtein** (two flavours),
**Indel**, **Hamming**.

All edit-distance scorers in this module accept `bytes`, `str`
(UCS-1 / UCS-2 / UCS-4 — Chinese, Japanese, Korean, Arabic, emoji
all route into the SIMD path without a UTF-8 round-trip), and any
NumPy integer `ndarray`. Inputs at different widths are not mixed
within one call.

## Output-processing variants

Every algorithm exposes the same family of entry points; the suffix
controls how results are returned. `<algo>` below stands for any of
`levenshtein`, `damerau_levenshtein`, `true_damerau_levenshtein`,
`indel`, `hamming`.

| Suffix | Signature | Return type | Use when |
| --- | --- | --- | --- |
| `<algo>_scores(query, targets)` | one-to-many distance | `ndarray[int64]` shape `(len(targets),)` | you have one query and want raw distances to every candidate |
| `<algo>_normalized_scores(query, targets)` | one-to-many similarity | `ndarray[float64]` in `[0, 1]` | you want length-normalised similarity (1.0 == identical, 0.0 == fully different) |
| `<algo>_best(query, targets)` | one-to-one | `(target, int_distance)` or `None` | you only need the single closest target — saves the heap step |
| `<algo>_normalized_best(query, targets)` | one-to-one normalised | `(target, float_similarity)` or `None` | best target by normalised similarity rather than raw distance |

Cross-product (queries × targets) and top-k workloads live on the
`cdist` family — see [cdist.md](cdist.md).

All variants release the GIL while the SIMD kernel runs, dispatch
the best backend for the current CPU, and accept `score_cutoff` /
`prefix_weight` / `n=` kwargs where the algorithm supports them.

## Algorithms

### `levenshtein_*` — classic Levenshtein

Cost-1 insertions, deletions, and substitutions. Bit-parallel Myers
backend (single-word for `len <= 64`, K-specialised multi-word
kernels for K=2..K=4, generic K beyond). Supports `score_cutoff`
for early bail.

```python
import stride_align as sa

sa.levenshtein_scores("kitten", ["sitting", "kitchen", "mittens"])
# array([3, 2, 2], dtype=int64)

sa.levenshtein_normalized_best("北京大学", ["清华大学", "北京交通大学", "复旦大学"])
# ('北京交通大学', 0.5)
```

### `damerau_levenshtein_*` — OSA (optimal-string-alignment) variant

Adds cost-1 transposition of adjacent characters, with the
restriction that a substring can be edited at most once. This is
what most libraries (and `rapidfuzz.distance.OSA`) actually
implement; what they call "Damerau-Levenshtein" is usually OSA.
Cheap kernel, suitable for sub-100-char fuzzy matching.

### `true_damerau_levenshtein_*` — unrestricted Damerau-Levenshtein

The unrestricted version that allows multiple edits on the same
substring — Damerau's original 1964 definition. Byte-specialised
flat Lowrance-Wagner DP with an array-indexed last-occurrence
table. Use when correctness on overlapping transpositions matters
(linguistic analysis, OCR error models).

### `indel_*` — insert-and-delete only

Substitutions disallowed; matches the algorithm behind
`rapidfuzz.distance.Indel` and the underlying engine for
`fuzz.ratio`. Identity: `indel_normalized_score == fuzz.ratio /
100`. Hyyrö fused single-step recurrence with kernel-level
`score_cutoff` early-exit.

### `hamming_*` — equal-length substitutions

Cost-1 substitutions only, requires `len(query) == len(target)` —
raises `ValueError` on the first mismatched target. Optional `pad`
kwarg in the rapidfuzz shim form.

## Cutoff push-down

Edit-distance kernels accept `score_cutoff=k`. If a target's
provable lower bound on the distance exceeds `k`, the per-target
loop bails out and the lane reports `k + 1` (the cutoff sentinel)
instead of finishing. Cheaper than computing the full DP and
filtering afterwards.

```python
sa.levenshtein_scores("kitten", targets, score_cutoff=2)
# Any target whose true distance is >= 3 returns 3 (the sentinel).
```

## SIMD dispatch

The runtime picks the widest supported backend at first call
(`stride_align._cpu.detect_best_backend()`) — x86 SSE4.1 / AVX2 /
AVX-512BW+VL / AVX10-256 / AVX10-512, ARM NEON / SVE / SVE2,
LoongArch LSX / LASX, PowerPC VSX, with a scalar fallback. The
choice is logged once at backend import. No per-call dispatch
overhead.
