# All-pairs cdist

`cdist` and its pruned variants compute every (query × target)
score in one SIMD batch with the GIL released and per-row work
parallelised across threads. Use these when you have many queries
**and** many targets.

The four entry points differ in what they return — full matrix,
above-threshold pairs, top-k matches, or top-k *per query*. All
share the same `scorer=` argument and pruning machinery.

## Output-processing variants

| Function | Returns | Use when |
| --- | --- | --- |
| `cdist(queries, targets, *, scorer, …)` | `ndarray` shape `(Q, T)` | you need the full score matrix |
| `cdist_above_threshold(queries, targets, *, scorer, threshold, …)` | iterator of `(score, query, target)` | you only care about pairs above a similarity floor |
| `cdist_top_k(queries, targets, *, scorer, k, …)` | list of at most `k` `(score, query, target)` records | global k-nearest-neighbours across the whole grid |
| `cdist_top_k_per_query(queries, targets, *, scorer, k, …)` | iterator of `(query, list[(score, target)])` | every query gets its own top-k neighbour list |

The two top-k forms have different cardinality. `cdist_top_k(..., k=5)`
returns no more than five records from the entire `Q × T` grid.
`cdist_top_k_per_query(..., k=5)` returns one group per query and therefore
up to `5 × Q` records overall.

`cdist_top_k_per_query` accepts `cpu_count=` (default `0` =
auto-detect, capped sensibly; `1` = single-threaded). Per-query
rows run on a worker pool sharing one byte-snapshot of the target
list under the released GIL.

## The `Scorer` enum

`scorer=` accepts a `stride_align.Scorer` enum value (cheapest;
dispatches through the C++ batch kernel) or any Python callable
that takes two sequences and returns a score (slower fallback).

| Value | Algorithm |
| --- | --- |
| `Scorer.LEVENSHTEIN` / `LEVENSHTEIN_NORMALIZED` | edit distance |
| `Scorer.DAMERAU_LEVENSHTEIN` / `..._NORMALIZED` | OSA |
| `Scorer.HAMMING` / `HAMMING_NORMALIZED` | equal-length only |
| `Scorer.JARO` / `JARO_WINKLER` | similarity |
| `Scorer.INDEL` / `INDEL_NORMALIZED` | insert + delete only |
| `Scorer.TRUE_DAMERAU_LEVENSHTEIN` / `..._NORMALIZED` | unrestricted DL |
| `Scorer.SMITH_WATERMAN` / `..._NORMALIZED` | local alignment |
| `Scorer.NEEDLEMAN_WUNSCH` / `..._NORMALIZED` | global alignment |

`SMITH_WATERMAN` / `NEEDLEMAN_WUNSCH` row-loop through the SIMD
single-query kernels (they accept the same `match_score` /
`mismatch_score` / `matrix` / `gap_*` kwargs). Threading via
`cpu_count` still parallelises rows because the per-row kernels
release the GIL.

## Pruning

All four entry points share three correctness-preserving
optimisations:

1. **Length-difference pruning.** Each `(q, t)` pair is gated by
   a closed-form upper bound on achievable normalised similarity
   before any SIMD work runs. Bounds: `min/max` for
   Levenshtein / OSA / true-DL, `(2 + min/max)/3` for Jaro,
   `2 * min / (q + t)` for Indel, `1.0` if equal-length for
   Hamming.
2. **Row-sort by query length, descending.** `cdist_top_k`
   processes the longest queries first so close-length
   high-scoring pairs surface early and the shared
   `global_min_bound` atomic reaches a useful value before the
   short-query rows run.
3. **Per-pair cutoff push-down into the SIMD kernel.** Levenshtein
   / OSA / Hamming inner loops bail per lane once the running
   score exceeds the per-pair cutoff plus the remaining-chars
   allowance; bailed lanes return the `cutoff + 1` sentinel.

The combination yields >100x speedup at `threshold=0.99` on
random-ASCII benchmarks against an unpruned `cdist` — see
[BENCHMARK.md](../../BENCHMARK.md) for the cross-arch numbers.

## Examples

```python
import stride_align as sa

# Full matrix
M = sa.cdist(queries, targets, scorer=sa.Scorer.LEVENSHTEIN_NORMALIZED)
# M.shape == (len(queries), len(targets)), dtype float64

# Threshold filter — get every pair within 90% Jaro-Winkler similarity
for qi, ti, score in sa.cdist_above_threshold(
    queries, targets,
    scorer=sa.Scorer.JARO_WINKLER, threshold=0.9,
):
    ...

# Top-100 most-similar pairs across the whole grid
matches = list(sa.cdist_top_k(
    queries, targets,
    scorer=sa.Scorer.INDEL_NORMALIZED, k=100,
))

# Per-query 5-nearest-neighbours, 8-way parallel
for query, matches in sa.cdist_top_k_per_query(
    queries, targets,
    scorer=sa.Scorer.JARO_WINKLER, k=5, cpu_count=8,
):
    # matches contains up to 5 (score, target) tuples for query.
    ...
```

## Choosing the right entry point

| You want… | Use |
| --- | --- |
| every pair's score | `cdist` |
| only pairs above a similarity floor | `cdist_above_threshold` |
| the global k most-similar pairs | `cdist_top_k` |
| each query's own k nearest neighbours | `cdist_top_k_per_query` |

Single-query workloads (one query × many targets) should use the
per-algorithm `_scores` / `_best` / `_normalized_*` / `_top_k`
entry points instead — they skip the cross-product setup. See
[edit-distance.md](edit-distance.md) and
[similarity.md](similarity.md).
