# Benchmark Summary

Generated on 2026-05-13 with Python 3.13 in the project virtualenv.

Raw results are in [`benchmark.csv`](benchmark.csv). The CSV contains 320 data
rows: English and Chinese text workloads, linear and affine scoring, widths 16
and 32, `1:1` and `1:many` shapes, and generic/x86/parasail backends.

Command:

```bash
taskset -c 2 .venv/bin/python -m stride_align.benchmark \
  --backends generic x86_avx2 x86_avx512bwvl parasail \
  --variants sw-farrar-score sw-score nw-score sw-path-info nw-path-info sw-cigar nw-cigar \
  --passes english chinese \
  --shapes all \
  --scoring-cases linear affine \
  --widths 16 32 \
  --iterations 15 \
  --warmups 3 \
  --timing-split \
  --format csv > benchmark.csv
```

The comparison ratios below use median runtime. A ratio above `1.0x` means
stride-align is faster than parasail for the same row:

```text
ratio = parasail_median_seconds / stride_align_median_seconds
```

## Overall Position vs Parasail

| Backend | Comparable rows | Wins vs parasail | Geomean vs parasail | Median ratio |
| --- | ---: | ---: | ---: | ---: |
| `x86_avx2` | 80 | 52 | 1.237x | 1.168x |
| `x86_avx512bwvl` | 80 | 66 | 1.451x | 1.390x |
| `generic` | 80 | 8 | 0.224x | 0.163x |

Score-only rows:

| Backend | Rows | Wins vs parasail | Geomean vs parasail | Median ratio |
| --- | ---: | ---: | ---: | ---: |
| `x86_avx2` | 48 | 28 | 1.139x | 1.100x |
| `x86_avx512bwvl` | 48 | 39 | 1.348x | 1.368x |

Path/CIGAR rows:

| Backend | Rows | Wins vs parasail | Geomean vs parasail | Median ratio |
| --- | ---: | ---: | ---: | ---: |
| `x86_avx2` | 32 | 24 | 1.402x | 1.519x |
| `x86_avx512bwvl` | 32 | 27 | 1.620x | 1.530x |

## By Variant

AVX2:

| Variant | Rows | Wins | Geomean vs parasail | Median ratio |
| --- | ---: | ---: | ---: | ---: |
| `nw-cigar` | 8 | 8 | 2.127x | 2.268x |
| `nw-path-info` | 8 | 8 | 2.200x | 2.545x |
| `nw-score` | 16 | 7 | 0.955x | 0.927x |
| `sw-cigar` | 8 | 3 | 0.760x | 0.806x |
| `sw-farrar-score` | 16 | 9 | 1.118x | 1.068x |
| `sw-path-info` | 8 | 5 | 1.085x | 1.421x |
| `sw-score` | 16 | 12 | 1.383x | 1.592x |

AVX512BWVL:

| Variant | Rows | Wins | Geomean vs parasail | Median ratio |
| --- | ---: | ---: | ---: | ---: |
| `nw-cigar` | 8 | 8 | 1.962x | 1.739x |
| `nw-path-info` | 8 | 8 | 2.160x | 2.214x |
| `nw-score` | 16 | 15 | 1.430x | 1.554x |
| `sw-cigar` | 8 | 4 | 1.090x | 1.052x |
| `sw-farrar-score` | 16 | 8 | 0.979x | 0.974x |
| `sw-path-info` | 8 | 7 | 1.493x | 1.299x |
| `sw-score` | 16 | 16 | 1.751x | 1.686x |

## Main Takeaways

AVX512BWVL is the strongest backend in this sweep. It wins 66 of 80 comparable
rows and is especially strong on `sw-score`, `nw-score`, and NW path/CIGAR
variants.

AVX2 is ahead overall, but the losses are concentrated. `nw-path-info` and
`nw-cigar` are strong; `sw-score` is strong overall; the biggest remaining AVX2
problems are linear SW path/CIGAR and some NW score rows.

`sw-farrar-score` is close but still uneven. AVX2 is slightly ahead on geomean
for this sweep, while AVX512BWVL is slightly behind. The worst AVX512 row is
Chinese affine `1:many` width32 `sw-farrar-score`, where AVX512BWVL is about
0.64x parasail.

The generic backend is useful as a correctness/performance baseline, not as a
parasail competitor. It loses most score-only rows badly, although a few
linear NW path/CIGAR rows are competitive.

## Worst Current Rows

AVX2 worst rows vs parasail:

| Ratio | Pass | Case | Shape | Variant | Width |
| ---: | --- | --- | --- | --- | ---: |
| 0.293x | chinese | linear | 1:1 | `sw-path-info` | 16 |
| 0.336x | english | linear | 1:1 | `sw-score` | 16 |
| 0.355x | chinese | linear | 1:1 | `sw-cigar` | 16 |
| 0.437x | english | linear | 1:1 | `sw-cigar` | 16 |
| 0.463x | chinese | linear | 1:1 | `sw-cigar` | 32 |

AVX512BWVL worst rows vs parasail:

| Ratio | Pass | Case | Shape | Variant | Width |
| ---: | --- | --- | --- | --- | ---: |
| 0.644x | chinese | affine | 1:many | `sw-farrar-score` | 32 |
| 0.674x | english | linear | 1:1 | `sw-cigar` | 32 |
| 0.677x | chinese | affine | 1:1 | `sw-farrar-score` | 32 |
| 0.728x | chinese | linear | 1:1 | `nw-score` | 16 |
| 0.744x | chinese | linear | 1:1 | `sw-cigar` | 16 |

## Notes

These numbers are intended for engineering direction, not publication-grade
claims. The run is pinned with `taskset -c 2`, but it still uses short
15-iteration medians and includes Python benchmark orchestration. For final
claims, rerun longer pinned sweeps and native microbench rows for the specific
kernels under discussion.
