# macOS Arm64 NEON Benchmarks - 2026-05-14

These results are separate from the x86 and Loongson benchmark families.

Host: `wopr`, macOS 15.3.1 arm64, Python 3.13 from Homebrew, Apple clang 17.
The benchmark venv contains `parasail==1.3.4`; the installed parasail Python
package reports bundled parasail library version `2.6.2`.

Raw artifacts:

| Artifact | Contents |
| --- | --- |
| [`macos-arm64-neon-focused-2026-05-14.csv`](macos-arm64-neon-focused-2026-05-14.csv) | Focused `generic`, `macos_arm64_neon`, and `parasail` rows for `sw-path-info`, `sw-cigar`, `nw-score`, and `sw-farrar-score`, including `1:1` and `1:many`. |
| [`macos-arm64-neon-microbench-2026-05-14.txt`](macos-arm64-neon-microbench-2026-05-14.txt) | Native NEON microbench rows for affine SW CIGAR/path-info, affine NW score, and SW Farrar score. |
| [`macos-arm64-neon-nw-affine-primitives-2026-05-14.csv`](macos-arm64-neon-nw-affine-primitives-2026-05-14.csv) | Narrow affine `nw-score` run after adding NEON vector primitives. |
| [`macos-arm64-neon-nw-affine-fastpaths-2026-05-14.csv`](macos-arm64-neon-nw-affine-fastpaths-2026-05-14.csv) | Negative-control run: dense/plain global-affine fast-path flags enabled for NEON. This lost and should stay disabled. |

## Position vs Parasail

The ratio below is `parasail_median_seconds / stride_align_median_seconds`; a
value above `1.0x` means stride-align is faster than parasail for that row.

| Group | Rows | Wins vs Parasail | Geomean vs Parasail | Median Ratio |
| --- | ---: | ---: | ---: | ---: |
| Focused sweep | 48 | 3 | 0.720x | 0.714x |
| Score-only | 32 | 0 | 0.683x | 0.684x |
| Path/CIGAR | 16 | 3 | 0.801x | 0.821x |
| Affine `nw-score` | 8 | 0 | 0.647x | 0.667x |
| Affine SW path/CIGAR | 8 | 1 | 0.838x | 0.904x |
| `sw-farrar-score` | 16 | 0 | 0.673x | 0.663x |

## Changes Measured

Affine SW path/CIGAR is no longer the worst area. The NEON public affine SW
path-info and CIGAR calls now use the CIGAR-first/profile traceback path with
score verification instead of materializing full striped path state for the
Python API. In the current focused sweep, English affine `1:1` `sw-path-info`
width16 is at `1.002x` parasail and `sw-cigar` width16 is at `0.987x`.

Affine global `nw-score` improved from the 2026-05-13 parasail sweep by about
`1.27x` geomean after adding NEON vector primitives for sentinel-preserving
adds, lane insert shifts, and exact no-padding prefix carry. It still loses to
parasail, especially width32 `1:many`, which is about `0.59x` parasail.

The attempted dense/plain global-affine fast-path flags were a clear loss. The
narrow `macos-arm64-neon-nw-affine-fastpaths-2026-05-14.csv` run dropped affine
`nw-score` to about `0.30x` parasail geomean. Keep those flags disabled for
NEON unless the loop structure is redesigned.

## Worst Rows

| Ratio | Pass | Case | Shape | Variant | Width |
| ---: | --- | --- | --- | --- | ---: |
| 0.527x | chinese | linear | 1:1 | `sw-path-info` | 16 |
| 0.553x | chinese | linear | 1:many | `sw-farrar-score` | 16 |
| 0.554x | english | linear | 1:many | `sw-farrar-score` | 16 |
| 0.577x | english | linear | 1:1 | `sw-farrar-score` | 16 |
| 0.589x | chinese | affine | 1:many | `nw-score` | 32 |
| 0.595x | english | affine | 1:many | `nw-score` | 32 |

## Recommendations

1. Target linear SW trace/CIGAR width16 next. It is now the worst macOS NEON row, and the loss is trace/output path specific rather than affine score DP.
2. Work on `sw-farrar-score` width16 before width32. Width16 loses in both `1:1` and `1:many`, and native microbench shows the kernel is otherwise stable enough to profile.
3. For affine `nw-score`, do not re-enable dense/plain global-affine flags. The next attempt should be a NEON-specific loop redesign, not the x86-oriented shared flag path.
4. Add a native parasail comparison path for the arm64 microbench if we need instruction-level parity work. The current native harness measures NEON only.
5. Keep the public affine SW CIGAR/path-info route separate from the striped path materialization benchmark. The public path is competitive; the native striped traceback rows are intentionally slower and diagnostic.
6. Profile with Apple Instruments or `sample` before micro-optimizing NEON. Current losses are broad enough that guessing between profile loads, trace traffic, and lazy-F propagation is not useful.
7. Continue treating SWAR as a reference backend on macOS; it is not a performance target.
