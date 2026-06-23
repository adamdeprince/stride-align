# parasail compatibility shim

`stride_align.parasail` is a **drop-in compatibility layer** for
existing parasail Python codebases. It exposes the same module
surface as upstream `parasail-python` — `sw`, `nw`, `sg` and their
`_trace` / `_stats` variants, `matrix_create`, the BLOSUM and PAM
tables, the `Result` / `Cigar` / `Traceback` / `Matrix` classes,
and the capability-detection helpers — dispatching every call into
the native `stride_align` sequence-alignment kernels.

This is a **migration aid**, not stride-align's primary alignment
API. New code should call the native `smith_waterman_*`,
`needleman_wunsch_*`, and `SubstitutionMatrix` entry points
directly — see [alignment.md](alignment.md) and
[matrices.md](matrices.md). They give you more output shapes
(`_path`, `_path_info`, `_trade_cigar`), batch `_scores` over a
target list, the `_top_k` helper, the Farrar score-only fast path,
and substitution-matrix-aware `cdist`.

## One-line migration

```python
# Before:
# import parasail

# After:
import stride_align.parasail as parasail
```

Most call sites stay unchanged.

## What's covered

| Upstream surface | Status |
| --- | --- |
| `parasail.sw(s1, s2, open, extend, matrix)` | covered |
| `parasail.sw_trace(...)` | covered (returns `Result` with `.cigar` and `.traceback`) |
| `parasail.sw_stats(...)` | covered (stats fields: `.length`, `.matches`, `.mismatches`, `.similar`) |
| `parasail.nw(...)` / `nw_trace` / `nw_stats` | covered |
| `parasail.sg(...)` / `sg_trace` / `sg_stats` | covered (both ends free for both sequences) |
| `parasail.matrix_create(alphabet, match, mismatch, case_sensitive=None)` | covered (returns a `Matrix` whose `.size`, `.matrix`, `.mapper`, `.name`, `.min`, `.max`, `.copy`, `.set_value` mirror upstream) |
| `parasail.blosum45` / `50` / `62` / `80` / `90`, `pam30` / `70` / `250` | covered |
| `Result.score`, `.end_query`, `.end_ref` (inclusive last index, parasail convention) | covered |
| `Cigar.decode` (bytes), `.beg_query`, `.beg_ref`, `.len` | covered |
| `Traceback.query`, `.ref`, `.comp` (the `|.|` matched-region annotation) | covered |
| Kernel-suffix aliases (`sw_striped_avx2_16`, `nw_scan_64`, …) | covered via module-level `__getattr__` — they alias to the matching core entry. stride-align picks the kernel internally; explicit suffix has no effect |
| `can_use_sse2` / `can_use_sse41` / `can_use_avx2` / `can_use_altivec` / `can_use_neon` | covered (report what the loaded stride-align backend supports) |

The shim follows the **BLAST gap-cost convention** parasail uses:
gap penalties are *positive* numbers and a length-`N` gap costs
`open + (N − 1) * extend`. The native `stride_align` API uses
*negative* `gap_open_score` / `gap_extend_score` matching the
literature convention. The shim handles the sign / formula
translation at the boundary.

## Known divergences

| Difference | Behaviour |
| --- | --- |
| Local-alignment CIGAR span | Native: CIGAR covers the local-alignment region only. Parasail: CIGAR is prepended with leading edits so it spans from position 0 of each sequence. `Cigar.beg_query` / `Cigar.beg_ref` carry the actual local-alignment start, so callers reading both fields together get the same information |
| `sg_qb_de` style semi-global mode selectors | not yet supported — raises `AttributeError` from `__getattr__`. Plain `sg` (both ends free for both sequences) is supported |
| `parasail.dnafull`, `parasail.nuc44` | not yet provided. Use `matrix_create("ACGTN", match, mismatch)` for a hand-rolled DNA matrix, or pass a `SubstitutionMatrix` to the native API |

## Limitations vs the native API

| Native feature | Shim coverage |
| --- | --- |
| `smith_waterman_normalized_score` / `_normalized_scores` | no — use native |
| `smith_waterman_scores(query, targets)` batch | no — call `sw(...)` in a loop, or use native batch |
| `smith_waterman_top_k` | no — use native |
| `cdist(..., scorer=Scorer.SMITH_WATERMAN, matrix=…)` | no — use native |
| Farrar score-only fast path | no — use native `smith_waterman_farrar_*` |
| Built-in matrix as a regular `SubstitutionMatrix` | `parasail.blosum62` is a `parasail`-shim `Matrix`; for native use, import from `stride_align.matrices` instead |

## Migration recommendation

If you're starting from a parasail codebase:

1. Run the one-line import swap and your existing tests.
2. Replace any per-target loop over `parasail.sw(query, t,
   open, extend, m)` with `sa.smith_waterman_scores(query,
   targets, matrix=m, gap_open_score=-open,
   gap_extend_score=-extend)` — saves the per-call Python frame
   and runs the targets in one batch.
3. Replace any "score-only" call site with the Farrar fast path
   (`sa.smith_waterman_farrar_score` / `_scores`) — same SIMD
   layout parasail uses internally, but without the
   alignment-result object overhead.
4. For top-k / threshold workloads, use
   `sa.cdist_top_k(queries, targets, scorer=Scorer.SMITH_WATERMAN,
   matrix=…, …)` or `sa.cdist_above_threshold(...)`. Parasail has
   no equivalent.

## See also

- [alignment.md](alignment.md) — the native Smith-Waterman /
  Needleman-Wunsch surface with the full variant family
- [matrices.md](matrices.md) — `SubstitutionMatrix`, BLOSUM, PAM,
  and the NCBI-text loader
- [rapidfuzz-shim.md](rapidfuzz-shim.md) — the other compatibility
  shim, for fuzzy-match codebases
