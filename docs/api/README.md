# stride-align API reference

Native `stride_align` API surface, grouped by what you're doing.
Most pages share the same shape: a small "output-processing
variants" table that explains the suffix family
(`_scores` / `_normalized_scores` / `_best` / `_top_k`) once, then
an algorithm list with one paragraph each.

## Native API (primary)

| Page | Covers |
| --- | --- |
| [edit-distance.md](edit-distance.md) | Levenshtein, Damerau-Levenshtein (OSA + unrestricted), Indel, Hamming |
| [similarity.md](similarity.md) | Jaro, Jaro-Winkler, n-gram (Jaccard / Dice / cosine / overlap), Ratcliff-Obershelp, Monge-Elkan |
| [alignment.md](alignment.md) | Smith-Waterman, Needleman-Wunsch, Farrar score-only, traceback, CIGAR |
| [cdist.md](cdist.md) | All-pairs `cdist`, `cdist_above_threshold`, `cdist_top_k`, `cdist_top_k_per_query`, `Scorer` enum, pruning |
| [matrices.md](matrices.md) | `SubstitutionMatrix`, built-in BLOSUM and PAM, NCBI text loader |
| [dtw.md](dtw.md) | Dynamic Time Warping with Sakoe-Chiba band and alternative local metrics |
| [phonetic.md](phonetic.md) | Soundex, Metaphone, Double Metaphone, NYSIIS, MRA, Caverphone 2, Cologne Phonetic, Daitch-Mokotoff, Beider-Morse |

## Compatibility shims (migration aids)

| Page | Covers |
| --- | --- |
| [rapidfuzz-shim.md](rapidfuzz-shim.md) | `stride_align.rapidfuzz.{fuzz, distance, process, utils}` — drop-in replacement for `rapidfuzz`. |
| [thefuzz-shim.md](thefuzz-shim.md) | `stride_align.thefuzz.{fuzz, process, utils}` — work-alike for TheFuzz 0.22.1's integer scorers and extraction API. |
| [parasail-shim.md](parasail-shim.md) | `stride_align.parasail` — drop-in replacement for `parasail-python` (Smith-Waterman / Needleman-Wunsch / semi-global, BLOSUM/PAM, `Result` / `Cigar` / `Traceback`). |
| [jellyfish-shim.md](jellyfish-shim.md) | `stride_align.jellyfish` — work-alike for Jellyfish's edit, similarity, and phonetic functions. |

These shims are migration aids; new code should call the native
API directly.

## Two dimensions

The API has two orthogonal dimensions: **which algorithm** and
**what shape you want the output in**. Rather than enumerating
every (algorithm × shape) combination, each page documents the
shape suffixes once at the top and then lists the algorithms.
Common output suffixes:

| Suffix | Returns |
| --- | --- |
| `_score(query, target)` | single int / float |
| `_normalized_score(query, target)` | single float in `[0, 1]` |
| `_scores(query, targets)` | `ndarray` of int / float, one per target |
| `_normalized_scores(query, targets)` | `ndarray[float64]` in `[0, 1]`, one per target |
| `_best(query, targets)` | `(target, score)` or `None` |
| `_normalized_best(query, targets)` | `(target, normalised_score)` or `None` |
| `_top_k(query, targets, k=…)` | list of top-k `(target, score)` |
| `cdist*` family | full grid / threshold / top-k over many queries × many targets |

Alignment and DTW have a richer variant family because they
return paths, CIGARs, and per-target traceback — see
[alignment.md](alignment.md).

## Cross-cutting

- **Inputs**: every kernel accepts `bytes`, `str` (UCS-1 / UCS-2
  / UCS-4 — zero-copy, no `.encode("utf-8")` round-trip), and
  NumPy integer `ndarray`. UCS-2 inputs go straight into a
  16-bit-token SIMD path so Chinese / Japanese / Korean strings
  use the same kernel as ASCII rather than being downcoded to
  bytes.
- **GIL**: every SIMD kernel releases the GIL while running. The
  cdist family parallelises rows across `cpu_count` worker
  threads.
- **Backend dispatch**: the widest SIMD kernel for the current
  CPU is chosen once at import. See
  [edit-distance.md#simd-dispatch](edit-distance.md#simd-dispatch)
  for the priority order.
- **Cutoff push-down**: edit-distance entry points accept
  `score_cutoff=k` and bail per lane when the lower bound on
  distance exceeds `k` — saves work over full DP + filter.

## See also

- [../../README.md](../../README.md) — project overview, install,
  quick start
- [../../BENCHMARK.md](../../BENCHMARK.md) — performance numbers
  vs parasail, rapidfuzz, python-Levenshtein, editdistance
- [../../CHANGELOG.md](../../CHANGELOG.md) — version history
- [../adding-a-new-algorithm.md](../adding-a-new-algorithm.md) —
  internals if you want to add a new SIMD-backed kernel
