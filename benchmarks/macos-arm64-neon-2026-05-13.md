# macOS Arm64 NEON Benchmarks - 2026-05-13

These results are separate from the x86 and Loongson benchmark families.

Host: `wopr`, macOS 15.3.1 arm64, Python 3.13 from Homebrew, Apple clang 17.
The benchmark venv contains `parasail==1.3.4`; the installed parasail Python
package reports bundled parasail library version `2.6.2`.

Raw artifacts:

| Artifact | Contents |
| --- | --- |
| [`macos-arm64-neon-score-native-2026-05-13.csv`](macos-arm64-neon-score-native-2026-05-13.csv) | Native score-only rows for `generic`, `swar`, and `macos_arm64_neon`. |
| [`macos-arm64-neon-score-parasail-2026-05-13.csv`](macos-arm64-neon-score-parasail-2026-05-13.csv) | Score-only rows for `generic`, `swar`, `macos_arm64_neon`, and `parasail`, including `1:1` and `1:many`. |
| [`macos-arm64-neon-path-parasail-2026-05-13.csv`](macos-arm64-neon-path-parasail-2026-05-13.csv) | Path/CIGAR timing-split rows for `generic`, `swar`, `macos_arm64_neon`, and `parasail`, `1:1` only. |

Install note: upstream `pip install parasail` needed Homebrew autotools and
libtool available on `PATH`, including `/opt/homebrew/bin` for `glibtoolize`.

## Native Position

Against the generic backend, `macos_arm64_neon` is a real speedup for score-only
work:

| Backend | Score Rows | Wins vs Generic | Geomean vs Generic | Median vs Generic |
| --- | ---: | ---: | ---: | ---: |
| `macos_arm64_neon` | 48 | 44 | 2.40x | 2.65x |
| `swar` | 48 | 5 | 0.64x | 0.65x |

For path/CIGAR rows, NEON is only modestly faster than generic overall:

| Backend | Path Rows | Wins vs Generic | Geomean vs Generic | Median vs Generic |
| --- | ---: | ---: | ---: | ---: |
| `macos_arm64_neon` | 32 | 23 | 1.10x | 1.04x |
| `swar` | 32 | 24 | 1.11x | 1.04x |

## Position vs Parasail

The ratio below is `parasail_median_seconds / stride_align_median_seconds`; a
value above `1.0x` means stride-align is faster than parasail for that row.

| Backend | Comparable Rows | Wins vs Parasail | Geomean vs Parasail | Median Ratio |
| --- | ---: | ---: | ---: | ---: |
| `macos_arm64_neon` | 80 | 10 | 0.72x | 0.75x |
| `swar` | 80 | 10 | 0.33x | 0.29x |
| `generic` | 80 | 10 | 0.41x | 0.35x |

Score-only:

| Backend | Rows | Wins vs Parasail | Geomean vs Parasail | Median Ratio |
| --- | ---: | ---: | ---: | ---: |
| `macos_arm64_neon` | 48 | 0 | 0.71x | 0.75x |

Path/CIGAR:

| Backend | Rows | Wins vs Parasail | Geomean vs Parasail | Median Ratio |
| --- | ---: | ---: | ---: | ---: |
| `macos_arm64_neon` | 32 | 10 | 0.75x | 0.79x |

By variant for `macos_arm64_neon`:

| Variant | Rows | Wins | Geomean vs Parasail | Median Ratio |
| --- | ---: | ---: | ---: | ---: |
| `nw-cigar` | 8 | 4 | 1.12x | 1.15x |
| `nw-path-info` | 8 | 4 | 1.00x | 1.42x |
| `nw-score` | 16 | 0 | 0.61x | 0.71x |
| `sw-cigar` | 8 | 1 | 0.56x | 0.62x |
| `sw-farrar-score` | 16 | 0 | 0.67x | 0.66x |
| `sw-path-info` | 8 | 1 | 0.50x | 0.54x |
| `sw-score` | 16 | 0 | 0.85x | 0.90x |

Worst `macos_arm64_neon` rows vs parasail:

| Ratio | Pass | Case | Shape | Variant | Width |
| ---: | --- | --- | --- | --- | ---: |
| 0.21x | chinese | affine | 1:1 | `sw-path-info` | 16 |
| 0.27x | english | affine | 1:1 | `sw-path-info` | 16 |
| 0.29x | chinese | affine | 1:1 | `sw-cigar` | 16 |
| 0.36x | english | affine | 1:1 | `sw-cigar` | 16 |
| 0.38x | english | affine | 1:1 | `nw-path-info` | 16 |

## Recommendations

1. Target affine SW traceback first. The largest losses are `sw-path-info` and
   `sw-cigar` affine width16, which points to trace representation and decode
   traffic rather than score-only DP throughput.
2. Port the x86 CIGAR-first affine traceback structure to NEON only after
   measuring the hot loops; the current NEON path rows are near generic, not
   near parasail.
3. Add a native mac arm64 microbench path before doing instruction scheduling.
   Python orchestration is acceptable for direction, but the remaining parasail
   gap needs source-level profiling.
4. Improve `nw-score` affine/global score separately from SW. NEON loses every
   `nw-score` comparison to parasail and width32 affine `1:many` is a visible
   regression.
5. Keep SWAR out of the mac performance path. It is roughly generic for some
   path rows but far behind NEON and parasail for score-only work.
6. Add Apple performance-counter profiling for `sw-farrar-score` width16 and
   width32. The score-only gap is broad enough that profile layout and lazy-F
   correction should be measured, not guessed.
