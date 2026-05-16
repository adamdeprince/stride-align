# Benchmark Summary

This file contains separate benchmark families. The original `benchmark.csv`
summary below is x86-only. Loongson LSX/LASX, macOS arm64 NEON, and Linux
PowerPC64 VSX (Power8) results are listed separately and should not be
compared directly with the x86 rows.

Generated on 2026-05-13 with Python 3.13 in the project virtualenv.

Raw results are in [`benchmark.csv`](benchmark.csv). The CSV contains 320 data
rows: English and Chinese text workloads, linear and affine scoring, widths 16
and 32, `1:1` and `1:many` shapes, and generic/x86/parasail backends.
The focused Intel exact-fill follow-up is in
[`benchmarks/x86-sw-farrar-exactfill-study-2026-05-14.csv`](benchmarks/x86-sw-farrar-exactfill-study-2026-05-14.csv).

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

The 2026-05-14 focused linear `sw-farrar-score` exact-fill run shows the Intel
hooks are working: SSE4.1 is `6.04x` generic, AVX2 is `10.67x` generic, and
AVX512BWVL is `14.41x` generic by focused geomean. Compared with the prior
`benchmark.csv` rows, AVX2 moved about `1.04x` because width16/32 already had
raw exact-fill kernels, while AVX512BWVL moved about `1.47x`.

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

## Loongson LSX/LASX - 2026-05-13

Raw Loongson artifacts are separate from `benchmark.csv`:

| Artifact | Contents |
| --- | --- |
| [`benchmarks/loongson-score-native-2026-05-13.csv`](benchmarks/loongson-score-native-2026-05-13.csv) | Native `generic`, `swar`, LSX, and LASX score rows for `1:1` and `1:many`. |
| [`benchmarks/loongson-score-1to1-parasail-2026-05-13.csv`](benchmarks/loongson-score-1to1-parasail-2026-05-13.csv) | Direct `sw-score`/`nw-score` `1:1` comparison against patched generic LoongArch parasail. |
| [`benchmarks/loongson-path-native-2026-05-13.csv`](benchmarks/loongson-path-native-2026-05-13.csv) | Native path/CIGAR timing split rows, no parasail. |
| [`benchmarks/loongson-sw-farrar-exactfill-baseline-2026-05-14.csv`](benchmarks/loongson-sw-farrar-exactfill-baseline-2026-05-14.csv) | Focused linear `sw-farrar-score` baseline before exact-fill hooks. |
| [`benchmarks/loongson-sw-farrar-exactfill-study-2026-05-14.csv`](benchmarks/loongson-sw-farrar-exactfill-study-2026-05-14.csv) | Focused linear `sw-farrar-score` run after exact-fill hooks. |
| [`benchmarks/loongson-2026-05-13.md`](benchmarks/loongson-2026-05-13.md) | Loongson-specific notes and recommendations. |

Build context: `loongson`, Python 3.13.13, GCC 15.2.0, CMake 4.3.2. The
LoongArch Python extension modules were built with static C++ runtime linkage;
`ldd` shows no dynamic `libstdc++`/`libgcc` dependency.

Parasail status: upstream `pip install parasail` failed on LoongArch. A patched
source build works for direct score calls after treating LoongArch as a non-x86
`cpuid` stub target, but it is generic parasail, not LSX/LASX optimized. Its
profile API returned NULL profiles and trace/CIGAR was not usable, so parasail
is included only for direct `sw-score`/`nw-score` `1:1` rows.

Loongson score-only takeaways:

| Backend | Native Score Rows | Median vs Generic | Best vs Generic |
| --- | ---: | ---: | ---: |
| `linux_loongarch64_lsx` | 48 | 3.93x | 8.86x |
| `linux_loongarch64_lasx` | 48 | 6.79x | 16.05x |
| `swar` | 48 | 0.59x | 1.02x |

Exact-fill linear `sw-farrar-score` hooks were added after the mac NEON study.
On the focused 2026-05-14 Loongson run, LSX improved `1.73x` geomean and LASX
improved `1.63x` geomean over the pre-change focused baseline. After the
change, focused linear `sw-farrar-score` is `10.24x` generic on LSX and
`17.50x` generic on LASX.

Against patched generic LoongArch parasail direct score rows:

| Backend | Comparable Rows | Wins | Geomean vs Parasail | Median vs Parasail |
| --- | ---: | ---: | ---: | ---: |
| `linux_loongarch64_lsx` | 16 | 16 | 5.31x | 5.85x |
| `linux_loongarch64_lasx` | 16 | 16 | 7.52x | 6.85x |

Loongson path/CIGAR takeaways:

LSX/LASX are not uniformly better for path-producing work yet. Affine CIGAR
rows are strong, with LASX reaching about `4.32x` generic, but linear SW
path/CIGAR is the weak point. The worst LSX linear SW path row is `0.34x`
generic and the worst LASX row is `0.48x` generic, which points at trace-table
and materialization traffic rather than raw score DP throughput.

Recommended Loongson next steps:

1. Keep exact-fill LSX/LASX score hooks enabled; they are a large score-only win.
2. Target a Loongson-specific linear SW trace/CIGAR redesign before doing instruction scheduling.
3. Start with LASX width16/width32 trace traffic reduction, then port the measured structure to LSX.
4. Add a native Loongson microbench/perf entrypoint before micro-optimizing LSX/LASX loops.
5. Treat parasail as a generic LoongArch comparison unless a maintained LSX/LASX parasail build becomes available.

## macOS Arm64 NEON - 2026-05-14

Raw macOS arm64 artifacts are separate from `benchmark.csv`:

| Artifact | Contents |
| --- | --- |
| [`benchmarks/macos-arm64-neon-score-native-2026-05-13.csv`](benchmarks/macos-arm64-neon-score-native-2026-05-13.csv) | Native `generic`, `swar`, and NEON score rows for `1:1` and `1:many`. |
| [`benchmarks/macos-arm64-neon-score-parasail-2026-05-13.csv`](benchmarks/macos-arm64-neon-score-parasail-2026-05-13.csv) | Score rows including locally installed parasail. |
| [`benchmarks/macos-arm64-neon-path-parasail-2026-05-13.csv`](benchmarks/macos-arm64-neon-path-parasail-2026-05-13.csv) | Path/CIGAR timing-split rows including locally installed parasail. |
| [`benchmarks/macos-arm64-neon-2026-05-13.md`](benchmarks/macos-arm64-neon-2026-05-13.md) | macOS arm64-specific notes and recommendations. |
| [`benchmarks/macos-arm64-neon-focused-2026-05-14.csv`](benchmarks/macos-arm64-neon-focused-2026-05-14.csv) | Focused post-optimization comparison against parasail. |
| [`benchmarks/macos-arm64-neon-microbench-2026-05-14.txt`](benchmarks/macos-arm64-neon-microbench-2026-05-14.txt) | Native NEON microbench rows. |
| [`benchmarks/macos-arm64-neon-sw-farrar-parasail-study-2026-05-14.csv`](benchmarks/macos-arm64-neon-sw-farrar-parasail-study-2026-05-14.csv) | Focused run after adding exact-fill linear SW Farrar NEON score paths. |
| [`benchmarks/macos-arm64-neon-linear-trace-onepass-parasail-study-2026-05-14.csv`](benchmarks/macos-arm64-neon-linear-trace-onepass-parasail-study-2026-05-14.csv) | Negative-control run for one-pass striped linear SW trace routing. This lost and was reverted. |
| [`benchmarks/macos-arm64-neon-2026-05-14.md`](benchmarks/macos-arm64-neon-2026-05-14.md) | Updated macOS arm64 notes and recommendations. |

Build context: `wopr`, macOS 15.3.1 arm64, Python 3.13 from Homebrew, Apple
clang 17. Parasail was installed locally as `parasail==1.3.4`; the bundled
parasail library reports version `2.6.2`. Installing parasail from source
required Homebrew autotools/libtool and `/opt/homebrew/bin` on `PATH` for
`glibtoolize`.

macOS arm64 native score-only takeaways:

| Backend | Score Rows | Wins vs Generic | Geomean vs Generic | Median vs Generic |
| --- | ---: | ---: | ---: | ---: |
| `macos_arm64_neon` | 48 | 44 | 2.40x | 2.65x |
| `swar` | 48 | 5 | 0.64x | 0.65x |

Current focused comparison against locally installed parasail:

| Group | Rows | Wins vs Parasail | Geomean vs Parasail | Median Ratio |
| --- | ---: | ---: | ---: | ---: |
| Focused sweep | 48 | 3 | 0.720x | 0.714x |
| Score-only | 32 | 0 | 0.683x | 0.684x |
| Path/CIGAR | 16 | 3 | 0.801x | 0.821x |
| Affine `nw-score` | 8 | 0 | 0.647x | 0.667x |
| Affine SW path/CIGAR | 8 | 1 | 0.838x | 0.904x |
| `sw-farrar-score` | 16 | 0 | 0.673x | 0.663x |

Main macOS arm64 takeaways:

NEON is a real improvement over generic score-only code, but it is still not
parasail-competitive overall. The focused 2026-05-14 sweep is `0.720x`
parasail geomean across 48 comparable rows.

Affine SW path/CIGAR improved materially and is no longer the worst macOS NEON
area. Affine global `nw-score` improved about `1.27x` geomean over the
2026-05-13 parasail sweep after adding NEON vector primitives, but still loses
all parasail comparisons. The attempted dense/plain global-affine fast-path
flags were measured separately and lost; keep them disabled for NEON.

The Parasail-inspired exact-fill linear SW Farrar score path improved the
linear `sw-farrar-score` rows by `1.45x` geomean over the previous mac NEON
focused sweep. Width32 is now effectively at parasail parity for English and
Chinese `1:1` rows. A separate one-pass striped trace experiment lost badly
(`0.35x` parasail geomean and `0.45x` of the prior NEON trace baseline), so it
was reverted.

Recommended macOS arm64 next steps:

1. Keep the exact-fill linear SW Farrar score path enabled; it is the retained win from the Parasail comparison pass.
2. Work on `sw-farrar-score` width16 before width32.
3. Do not use the shared masked trace helpers for NEON linear SW path/CIGAR without redesigning the trace representation or decode path.
4. Redesign affine `nw-score` with a NEON-specific loop if we revisit it; do not reuse the x86-oriented dense/plain flags.
5. Add a native parasail comparison mode to the arm64 microbench before instruction-level parity work.
6. Keep SWAR out of the mac performance path except as a correctness/reference backend.

## Linux PowerPC64 VSX (Power8) - 2026-05-17

Raw Power8 artifact is separate from `benchmark.csv`:

| Artifact | Contents |
| --- | --- |
| [`benchmarks/power8-vsx-2026-05-17.csv`](benchmarks/power8-vsx-2026-05-17.csv) | Native `generic`, `swar`, and `linux_powerpc64_vsx` rows for `english`/`chinese`, `linear`/`affine`, widths 16 and 32, `1:1` and `1:many` shapes, with `--timing-split`. |
| [`benchmarks/power8-vsx-2026-05-17.md`](benchmarks/power8-vsx-2026-05-17.md) | Power8-specific notes, semantic-delta writeup, and recommendations. |

Build context: real POWER8 silicon (PVR `004b 0201`, 4.157 GHz), KVM-virtualized
as a single-core `pSeries` guest. Ubuntu 20.04 ppc64le, IBM Advance Toolchain
15.0 (GCC 11.4.1), Python 3.13.13 from miniforge, CMake 4.3.2 + Ninja from pip.
Parasail was not built (no upstream ppc64le wheel, source build not attempted).
All ratios are against `generic` on the same machine.

Power8 VSX score-only takeaways:

| Backend | Score Rows | Geomean vs Generic | Median vs Generic | Best vs Generic |
| --- | ---: | ---: | ---: | ---: |
| `linux_powerpc64_vsx` | 48 | 6.61x | 6.75x | 16.82x |
| `swar`                | 48 | 0.60x | 0.53x |  1.03x |

Power8 VSX path/CIGAR takeaways:

| Backend | Path Rows | Geomean vs Generic | Median vs Generic |
| --- | ---: | ---: | ---: |
| `linux_powerpc64_vsx` | 32 | 1.50x | 1.14x |
| `swar`                | 32 | 1.10x | 1.04x |

Power8 VSX overall vs generic:

| Backend | Rows | Wins vs Generic | Geomean | Median | Worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linux_powerpc64_vsx` | 80 | 79 | 3.65x | 4.13x | 0.96x |
| `swar`                | 80 | 27 | 0.77x | 0.98x | 0.41x |

Power8 VSX is a uniform win or tie. The 0.96x worst case is a short 1:1
linear path-info row at width 16 where SIMD setup dominates. The biggest
wins (~16x) are `sw-farrar-score` and `sw-score` for both English and
Chinese 1024x1024 at width 16. Affine `sw-cigar` and `nw-cigar` reach 4.2x
because they route through the SIMD score kernel +
`profile_traceback::affine_cigar_with_score` and skip the trace table.

Routing decisions that differ from x86_avx2 / Loongson: linear SW path /
path-info / linear CIGAR and affine path / path-info stay on
`profile_traceback` on Power8. The shared masked striped traceback wins on
ISAs with one-instruction lane bitmask extract and good striped-trace cache
behavior. On Power8 they measured 0.46-0.60x of the scalar byte-table path,
so we kept the helper hooks (`trace_mask_*`, `mask_or`, `store_masked_cells`,
the `vbpermq`-based collapse described inline) but route the public path /
path-info / linear CIGAR APIs through `profile_traceback`. The affine CIGAR
entry stays on the SIMD score kernel + scalar reverse-build, which beats
generic by 2-4x without a trace table.

SWAR is a regression on Power8 across most rows (geomean `0.77x` generic).
AT15 GCC 11.4 auto-vectorizes the generic score loop well enough that
SWAR's 64-bit packed lanes give no benefit. SWAR is still useful as a
correctness/reference backend.

Recommended Power8 next steps:

1. Try a `vbpermq`-based `trace_mask_*` on real hardware; combined with a row-major linear SW trace table this might bring the masked path above 1.0x of generic.
2. Add a Power8 `local_affine_score_exact_segment*_raw` mirroring the NEON helpers if the 1024-character query shape becomes a target; current `sw-farrar-score` is already 7-8x ahead of generic.
3. Re-bench on a multi-core / non-virtualized Power8 host to characterise SMT throughput and shared L2/L3 effects.
4. Build parasail from source for ppc64le and add a parasail column to the next sweep.
5. Investigate why SWAR loses to generic on Power8 via an asm dump of the generic score loop.
