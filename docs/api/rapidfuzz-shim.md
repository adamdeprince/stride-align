# rapidfuzz compatibility shim

`stride_align.rapidfuzz` is a **drop-in compatibility layer** for
existing rapidfuzz codebases. It exposes the same module / class /
function surface as upstream `rapidfuzz`, dispatching every call
into the native `stride_align` kernels.

This is a **migration aid**, not stride-align's primary API. New
code should call the native entry points (the rest of `docs/api/`)
directly — they're more explicit about return shape, give you the
full SIMD vocabulary (`cdist_above_threshold`, `cdist_top_k_per_query`,
substitution-matrix-aware alignment, phonetic encoders), and don't
carry the rapidfuzz scale convention.

If you already have a working rapidfuzz codebase, the one-line
import swap below is the fastest way to pick up stride-align's
SIMD kernels and pruning machinery without touching any call sites.

## One-line migration

```python
# Before:
# import rapidfuzz
# from rapidfuzz import fuzz, distance, process

# After:
import stride_align.rapidfuzz as rapidfuzz
from stride_align.rapidfuzz import fuzz, distance, process
```

The rest of your code stays unchanged.

## What's covered

| Upstream surface | Status |
| --- | --- |
| `rapidfuzz.fuzz.ratio` | covered (Indel kernel, `score_cutoff` push-down) |
| `rapidfuzz.fuzz.partial_ratio` | covered (interior + boundary phase scan to match rapidfuzz semantics) |
| `rapidfuzz.fuzz.token_sort_ratio` | covered |
| `rapidfuzz.fuzz.token_set_ratio` | covered |
| `rapidfuzz.fuzz.partial_token_sort_ratio` | covered |
| `rapidfuzz.fuzz.partial_token_set_ratio` | covered |
| `rapidfuzz.fuzz.WRatio` | covered |
| `rapidfuzz.fuzz.QRatio` | covered |
| `rapidfuzz.distance.Levenshtein` | `distance`, `similarity`, `normalized_distance`, `normalized_similarity`, `editops`, `opcodes` |
| `rapidfuzz.distance.Indel` | same four methods |
| `rapidfuzz.distance.Hamming` | same four methods (with `pad=` kwarg) |
| `rapidfuzz.distance.Jaro` | same four methods |
| `rapidfuzz.distance.JaroWinkler` | same four methods |
| `rapidfuzz.distance.DamerauLevenshtein` | unrestricted DL variant |
| `rapidfuzz.distance.OSA` | optimal-string-alignment variant |
| `rapidfuzz.distance.LCSseq` | routed through Indel via `LCS = (m + n − indel) / 2` |
| `rapidfuzz.process.extract` | fast-path dispatches built-in scorers through `sa.cdist_top_k_per_query` |
| `rapidfuzz.process.extractOne` | fast-path on built-in scorers |
| `rapidfuzz.process.extract_iter` | covered |
| `rapidfuzz.process.cdist` | fast-path on built-in scorers (multi-threaded SIMD kernel, GIL released) |
| `rapidfuzz.utils.default_process` | covered |
| `score_cutoff=` plumbing | through to the kernel for Indel-based scorers |

The shim returns scores on the rapidfuzz **0–100 scale** for
`fuzz.*` and the **0–1 scale** for `distance.*.normalized_*` (matching
upstream). The native API returns `[0, 1]` everywhere — multiply by
100 if you're moving back and forth.

## Performance

Full-surface bench (108 workloads × 3 architectures) ships in
[BENCHMARK.md](../../BENCHMARK.md#rapidfuzz-shim-full-surface-cross-arch-2026-06-10).
Geomean (upstream / shim, > 1.0 = shim faster):

| Host | Backend | Geomean | Wins/Ties/Losses |
| --- | --- | ---: | ---: |
| Intel AWS | `x86_avx10_512` | 0.94x | 60 / 4 / 44 |
| Mac M-series | `macos_arm64_neon` | 0.98x | 76 / 1 / 31 |
| Loongson | `linux_loongarch64_lasx` | 49.17x | 108 / 0 / 0 |

Loongson is a clean sweep because upstream rapidfuzz ships no
LoongArch wheel. Intel and Mac geomeans are near-parity with the
losses concentrated in Jaro/JaroWinkler single-pair medium-length
workloads and the cdist throughput ceiling — the wide majority of
methods land at parity or better than upstream.

## Limitations vs the native API

| Native feature | Shim coverage |
| --- | --- |
| `cdist_above_threshold` | no — use `sa.cdist_above_threshold` |
| `cdist_top_k_per_query` with `cpu_count` tuning | partial — `process.extract` uses sensible defaults |
| Substitution matrices in `cdist` | no — use native `sa.cdist(..., matrix=…)` |
| Smith-Waterman / Needleman-Wunsch scorers | no — use native `sa.smith_waterman_*` |
| Phonetic encoders | no — use native `sa.soundex` etc. |
| DTW | no — use native `sa.dtw_distances` |
| Beider-Morse | no — use native `sa.beider_morse` |
| Unicode wide-token Farrar | yes (UCS-2 / UCS-4 inputs route to the wide-token kernel automatically) |

When in doubt, prefer the native API. The shim is here so you can
swap one import and get most of the speedup; it's not a strict
superset of stride-align's capabilities.

## Migration recommendation

If you're starting from a rapidfuzz codebase:

1. Run the one-line import swap and your existing tests.
2. Replace any `rapidfuzz.process.cdist` call where you need
   threshold or per-query top-k with `sa.cdist_above_threshold` /
   `sa.cdist_top_k_per_query` — those are faster and have a more
   direct return shape.
3. Replace any `rapidfuzz.fuzz.ratio` call inside a hot loop with
   `sa.indel_normalized_score` (or the bulk `_normalized_scores`
   form) — saves a Python frame per call.
4. Phonetic / DTW / Smith-Waterman / matrix-aware alignment have
   no rapidfuzz equivalent; use the native API from the start.
