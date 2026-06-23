# Similarity scorers

Algorithms that score how *alike* two sequences are on a
similarity scale (higher == more alike), as opposed to the cost
metrics in [edit-distance.md](edit-distance.md). Covers
**Jaro**, **Jaro-Winkler**, **n-gram** (Jaccard / Sørensen-Dice /
cosine / overlap), **Ratcliff-Obershelp**, and the
**Monge-Elkan** composite scorer.

All scorers accept `bytes`, `str` (UCS-1 / UCS-2 / UCS-4), and
NumPy `ndarray`. UCS-2 inputs route to a 16-bit-token SIMD path —
Chinese, Japanese and Korean strings hit the same kernel as ASCII
rather than being downcoded to bytes.

## Output-processing variants

| Suffix | Signature | Return type | Use when |
| --- | --- | --- | --- |
| `<algo>_similarities(query, targets)` | one-to-many | `ndarray[float64]` in `[0, 1]` | raw similarity scores per target |
| `<algo>_best(query, targets)` | one-to-one | `(target, float_similarity)` or `None` | only need the closest match |

Multi-query (queries × targets), thresholded, or top-k workloads
live on the `cdist` family — see [cdist.md](cdist.md).

## Algorithms

### `jaro_similarities` — Jaro similarity

Matching-character ratio with transposition penalty. SIMD path
computes per-text-position window state in registers (`lo_v`,
`hi_complement_v`, `active_v`); no scalar prologue.

```python
import stride_align as sa
sa.jaro_similarities("dixon", ["dicksonx", "dickson"])
# array([0.79166667, 0.76666667])
```

### `jaro_winkler_similarities` — Jaro-Winkler

Jaro with a prefix bonus for shared leading characters. Tunable
via `prefix_weight` (default `0.1`), `prefix_threshold`
(`0.7`), and `prefix_cap` (`4` — maximum prefix length to
score, per Winkler's spec).

```python
sa.jaro_winkler_similarities(
    "martha", ["marhta", "marty"],
    prefix_weight=0.1, prefix_cap=4,
)
```

### n-gram similarities: `jaccard_similarities`, `dice_similarities`, `cosine_similarities`, `overlap_similarities`

Token-set similarity over character n-grams. Tunable via
`n=` (default 2 — character bigrams). UCS-2 inputs use a 32-bit
n-gram representation that fits 4× more tokens per SIMD lane than
the 64-bit fallback.

| Function | Formula |
| --- | --- |
| `jaccard_similarities` | `|A ∩ B| / |A ∪ B|` |
| `dice_similarities` | `2 |A ∩ B| / (|A| + |B|)` |
| `cosine_similarities` | `|A ∩ B| / sqrt(|A| · |B|)` |
| `overlap_similarities` | `|A ∩ B| / min(|A|, |B|)` |

```python
sa.dice_similarities("Stride-Align", ["StrideAlign", "Stride", "Alignment"], n=3)
```

### `ratcliff_obershelp_similarities` — Ratcliff-Obershelp

Recursive longest-common-substring match ratio. Backs
`difflib.SequenceMatcher.ratio()` semantics.

### `MongeElkan` — composite token-level scorer

Symmetric mean of best per-token match over two token lists. The
inner scorer is any one-to-many similarity function — pass a
`stride_align` `_similarities` function as `inner=` for full SIMD
speed, or a Python callable as a fallback.

```python
from stride_align._monge_elkan import monge_elkan_similarity
score = monge_elkan_similarity(
    "Department of Computer Science",
    "Computer Science Department",
    inner=sa.jaro_winkler_similarities,
)
```

## Algebraic identity with Indel

`indel_normalized_score(a, b) == fuzz.ratio(a, b) / 100`. This
lets one bit-parallel Indel kernel back four public surfaces
(`indel_normalized_*`, `fuzz.ratio`, `fuzz.QRatio`, and the
rapidfuzz `Indel` distance class) with no kernel duplication.

## SIMD dispatch

Same runtime backend selection as the edit-distance family — the
widest available kernel for the current CPU is chosen once at
import. See [edit-distance.md#simd-dispatch](edit-distance.md#simd-dispatch).
