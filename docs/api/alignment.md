# Sequence alignment

**Smith-Waterman** (local) and **Needleman-Wunsch** (global)
dynamic-programming alignment with substitution matrices, linear
or affine gaps, traceback to alignment path / CIGAR, and a
Farrar-vectorised score-only fast path.

Suitable for bioinformatics, fuzzy matching with custom
character-similarity, and any workload that needs the actual
alignment rather than just the score.

## Output-processing variants

The alignment surface has its own variant family because the
outputs are richer than a single distance. `<algo>` below is
`smith_waterman` or `needleman_wunsch`.

| Suffix | Signature | Return type | Use when |
| --- | --- | --- | --- |
| `<algo>_score(query, target, …)` | one-to-one score | `int` | you only need the alignment score |
| `<algo>_normalized_score(query, target, …)` | one-to-one normalised | `float` in `[0, 1]` | length-normalised score |
| `<algo>_scores(query, targets, …)` | one-to-many | `ndarray[int64]` | score against every target |
| `<algo>_normalized_scores(query, targets, …)` | one-to-many normalised | `ndarray[float64]` | normalised scores against every target |
| `<algo>_path(query, target, …)` | traceback path | `AlignmentPath` | you need the path coordinates |
| `<algo>_path_info(query, target, …)` | path + score | `AlignmentResult` | path plus pre-computed score |
| `<algo>_cigar(query, target, …)` | CIGAR string | `str` | SAM/BAM-style CIGAR output |
| `<algo>_trace_cigar(query, target, …)` | extended CIGAR | `str` | `=`/`X` distinguished from `M` |
| `<algo>_trade_cigar(query, target, …)` | extended CIGAR + score | `(str, int)` | CIGAR plus the score that produced it |

Smith-Waterman additionally exposes a top-k helper:

```python
sa.smith_waterman_top_k(query, targets, k=5, …)
# -> list[tuple[target, score, target_index]]
```

There's also a **Farrar** score-only fast path:

```python
sa.smith_waterman_farrar_score(query, target, …)
sa.smith_waterman_farrar_normalized_score(query, target, …)
sa.smith_waterman_farrar_scores(query, targets, …)
sa.smith_waterman_farrar_normalized_scores(query, targets, …)
```

The Farrar layout interleaves DP cells across SIMD lanes; it's
the fastest path when you only need the score and have a small
short-string query against many targets. For traceback or CIGAR
output, use the non-Farrar entry points.

## Scoring parameters

Every alignment entry point accepts the same scoring kwargs:

| Keyword | Default | Meaning |
| --- | --- | --- |
| `match_score` | `2` | bonus per matching position |
| `mismatch_score` | `-1` | penalty per mismatched position |
| `matrix` | `None` | `SubstitutionMatrix` overriding match/mismatch — see [matrices.md](matrices.md) |
| `gap_score` | `-1` | linear gap penalty (per gap character) |
| `gap_open_score` | `None` | when set together with `gap_extend_score`, switches to affine: cost is `gap_open + k * gap_extend` for a gap of length `k` |
| `gap_extend_score` | `None` | per-character cost inside an open gap |
| `width` | `None` | force a specific SIMD lane width (`0` = scalar, `8` / `16` / `32` / `64` bits). `None` = auto |

If `matrix=` is set, `match_score` / `mismatch_score` are ignored;
the matrix is the source of truth for per-pair substitution
scores. Built-in BLOSUM and PAM tables and a NCBI-text loader are
in [matrices.md](matrices.md).

## Examples

```python
import stride_align as sa
from stride_align.matrices import BLOSUM62

# Local alignment with affine gaps and BLOSUM62
sa.smith_waterman_score(
    "MQNS", "RMQDL",
    matrix=BLOSUM62,
    gap_open_score=-10,
    gap_extend_score=-1,
)

# Top-5 hits across a target list with custom match/mismatch
sa.smith_waterman_top_k(
    "kitten", ["sitting", "sitten", "smitten", "bitten", "kitty", "kit"],
    k=5, match_score=3, mismatch_score=-2, gap_score=-1,
)

# CIGAR with score for use in downstream SAM tooling
cigar, score = sa.smith_waterman_trade_cigar("ACGTAC", "ACGTAG")
```

## AlignmentResult / AlignmentPath

`AlignmentResult` is returned by `_path_info` and carries `score`,
`query_begin`, `query_end`, `target_begin`, `target_end`, and a
nested `AlignmentPath` with the cell-by-cell traceback. Both are
nanobind-bound C++ types but are pure value objects from Python's
perspective.

## SIMD dispatch

Alignment runs on the same per-CPU backend the rest of the
library uses. The internal selector picks a fixed-width kernel
based on input size and the `width=` hint (defaulting to the
narrowest width that won't overflow). See
[edit-distance.md#simd-dispatch](edit-distance.md#simd-dispatch).
