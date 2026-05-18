# Benchmark Summary

Stride-align is benchmarked across five hardware families: Intel x86 (AVX2,
AVX512BWVL), ARM Linux aarch64 on AWS Graviton4 (NEON/ASIMD, SVE, SVE2), ARM
macOS arm64 on Apple M-series (NEON), LoongArch64 on Loongson (LSX, LASX), and
PowerPC64 VSX on Power8. Each family ran on a different host with a different
parasail build (or no parasail at all), so numbers should be read within a
family, not across families. Raw CSVs live in `benchmark.csv` (x86) and
`benchmarks/*.csv` (everything else).

All ratios are median-runtime ratios. A ratio above `1.0x` means stride-align
is faster than the named baseline for that row:

```text
ratio = baseline_median_seconds / stride_align_median_seconds
```

## At a glance

| Family | Best stride-align backend | Baseline | Rows | Geomean | Median | Worst | Best |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Intel x86 | `x86_avx512bwvl` | parasail | 80 | **1.752x** | 1.678x | 0.909x | 3.794x |
| Intel x86 | `x86_avx2` | parasail | 80 | 1.377x | 1.302x | 0.474x | 3.513x |
| ARM Graviton4 (Linux) | `linux_aarch64_neon`/`asimd` | parasail | 80 | **1.149x** | 1.137x | 0.261x | 2.636x |
| ARM Graviton4 (Linux) | `linux_aarch64_sve2`† | parasail | 80 | 0.450x | 0.625x | 0.011x | 2.638x |
| ARM Mac M4 (macOS) | `macos_arm64_neon` | parasail | 48 | 0.720x | 0.711x | 0.527x | 1.016x |
| Loongson LoongArch64 | `linux_loongarch64_lasx` | patched parasail (1:1 score) | 16 | **7.517x** | 6.502x | 4.315x | 22.365x |
| Loongson LoongArch64 | `linux_loongarch64_lasx` | generic (native, score-only) | 48 | 6.220x | 6.781x | 2.472x | 16.052x |
| Power8 VSX (Linux) | `linux_powerpc64_vsx` | generic (no parasail) | 80 | **3.652x** | 4.126x | 0.960x | 16.821x |

† The `graviton4-arm-simd-parasail-2026-05-16.csv` artifact pre-dates the
`5174571` SVE/SVE2 fix in the git log, which advertises `0.45x → 1.04x/1.08x`
parasail. The 0.450x row above is the pre-fix snapshot in the CSV; the
post-fix numbers have not been re-captured into a tracked artifact.

## Intel x86 - 2026-05-18

Raw artifacts: [`benchmark.csv`](benchmark.csv) and the focused linear
`sw-farrar-score` exact-fill follow-up
[`benchmarks/x86-sw-farrar-exactfill-study-2026-05-14.csv`](benchmarks/x86-sw-farrar-exactfill-study-2026-05-14.csv).

Build context: 11th Gen Intel Core i7-1195G7, Python 3.13 in the project
virtualenv, host pinned with `taskset -c 2`, regenerated 2026-05-18 after the
CIGAR builder rework (`to_chars`-based digit emission + capacity reservation
in `build_cigar` and `ReverseCigarBuilder`). Parasail is the bundled
`parasail==1.3.4` wheel. The CSV contains 320 data rows: English and Chinese
workloads, linear and affine scoring, widths 16 and 32, `1:1` and `1:many`
shapes, and `generic`/`x86_avx2`/`x86_avx512bwvl`/`parasail` backends.

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

### Overall vs parasail

| Backend | Rows | Wins | Geomean | Median | Worst | Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `x86_avx512bwvl` | 80 | 77 | 1.752x | 1.678x | 0.909x | 3.794x |
| `x86_avx2` | 80 | 66 | 1.377x | 1.302x | 0.474x | 3.513x |
| `generic` | 80 |  8 | 0.222x | 0.176x | 0.058x | 1.615x |

Score-only rows (16 sw-farrar-score + 16 sw-score + 16 nw-score):

| Backend | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `x86_avx512bwvl` | 48 | 48 | 1.767x | 1.762x |
| `x86_avx2` | 48 | 40 | 1.297x | 1.173x |

Path/CIGAR rows (8 each of sw-path-info, nw-path-info, sw-cigar, nw-cigar):

| Backend | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `x86_avx512bwvl` | 32 | 29 | 1.730x | 1.524x |
| `x86_avx2` | 32 | 26 | 1.508x | 1.524x |

### By variant

AVX2:

| Variant | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `nw-path-info`    |  8 |  8 | 2.172x | 2.170x |
| `nw-cigar`        |  8 |  8 | 2.153x | 2.231x |
| `sw-score`        | 16 | 16 | 1.763x | 1.708x |
| `nw-score`        | 16 | 13 | 1.122x | 1.143x |
| `sw-farrar-score` | 16 | 11 | 1.103x | 1.045x |
| `sw-path-info`    |  8 |  5 | 1.060x | 1.061x |
| `sw-cigar`        |  8 |  5 | 1.043x | 1.051x |

AVX512BWVL:

| Variant | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `sw-score`        | 16 | 16 | 2.295x | 2.274x |
| `nw-path-info`    |  8 |  8 | 2.210x | 2.337x |
| `nw-cigar`        |  8 |  8 | 2.193x | 2.327x |
| `nw-score`        | 16 | 16 | 1.673x | 1.694x |
| `sw-farrar-score` | 16 | 16 | 1.436x | 1.314x |
| `sw-path-info`    |  8 |  7 | 1.373x | 1.412x |
| `sw-cigar`        |  8 |  6 | 1.346x | 1.414x |

### Worst rows vs parasail

AVX2:

| Ratio | Pass | Case | Shape | Variant | Width |
| ---: | --- | --- | --- | --- | ---: |
| 0.474x | chinese | linear | 1:1 | `sw-cigar`     | 16 |
| 0.493x | chinese | linear | 1:1 | `sw-path-info` | 16 |
| 0.763x | english | linear | 1:1 | `sw-cigar`     | 16 |
| 0.799x | english | linear | 1:1 | `sw-path-info` | 16 |
| 0.888x | chinese | linear | 1:1 | `nw-score`     | 32 |

AVX512BWVL:

| Ratio | Pass | Case | Shape | Variant | Width |
| ---: | --- | --- | --- | --- | ---: |
| 0.909x | chinese | linear | 1:1 | `sw-cigar`     | 16 |
| 0.943x | chinese | linear | 1:1 | `sw-path-info` | 16 |
| 0.951x | chinese | linear | 1:1 | `sw-cigar`     | 32 |
| 1.034x | chinese | linear | 1:1 | `sw-path-info` | 32 |
| 1.079x | chinese | affine | 1:1 | `sw-cigar`     | 16 |

### Takeaways

AVX512BWVL is the strongest Intel backend, winning **77 of 80** comparable
rows (every score-only row) at 1.752x parasail geomean. AVX2 wins 66 of 80 at
1.377x geomean. Both backends now beat parasail on every variant by geomean —
`sw-cigar` is the historically weakest variant but is positive (1.043x AVX2,
1.346x AVX512BWVL) instead of losing.

The 2026-05-18 sweep folded in the CIGAR builder rework (`to_chars` digit
emission and capacity reservation in `build_cigar` / `ReverseCigarBuilder`),
which dropped the path/CIGAR floor. AVX2 went from 1.237x → 1.377x overall
parasail geomean (`1.11x`), AVX512BWVL went from 1.451x → 1.752x (`1.21x`).
`sw-cigar` specifically moved from 0.760x → 1.043x on AVX2 and 1.090x →
1.346x on AVX512BWVL.

The 2026-05-14 focused linear `sw-farrar-score` exact-fill run still applies:
SSE4.1 reaches `6.04x` generic, AVX2 `10.67x`, and AVX512BWVL `14.41x` by
focused geomean.

AVX2's only remaining sub-parity rows are short linear SW 1:1 cigar/path-info
at width 16 (`sw-cigar` and `sw-path-info`); AVX512BWVL's worst row is
0.909x. Both backends now have only a handful of rows below 1.0x.

`generic` is for correctness/baseline reference, not as a parasail competitor.
It loses every score-only row badly; a handful of linear NW path/CIGAR rows
are competitive but not consistently.

## ARM Graviton4 (Linux aarch64) - 2026-05-16

Raw artifact:
[`benchmarks/graviton4-arm-simd-parasail-2026-05-16.csv`](benchmarks/graviton4-arm-simd-parasail-2026-05-16.csv)
(480 rows across English/Chinese, linear/affine, widths 16/32, `1:1` and
`1:many`, with `--timing-split`).

Build context: AWS Graviton4 (Neoverse V2). Backends measured:
`linux_aarch64_neon`, `linux_aarch64_asimd`, `linux_aarch64_sve`,
`linux_aarch64_sve2`, plus `generic` and `parasail`.

NEON and ASIMD report essentially identical numbers because commit `617d282`
merged the Linux ASIMD backend into the NEON backend; both names now resolve
to the same kernel set.

### Overall vs parasail

| Backend | Rows | Wins | Geomean | Median | Worst | Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `linux_aarch64_neon`  | 80 | 51 | 1.149x | 1.137x | 0.261x | 2.634x |
| `linux_aarch64_asimd` | 80 | 51 | 1.149x | 1.138x | 0.261x | 2.636x |
| `linux_aarch64_sve2`† | 80 | 24 | 0.450x | 0.625x | 0.011x | 2.638x |
| `linux_aarch64_sve`†  | 80 | 24 | 0.450x | 0.616x | 0.011x | 2.649x |

Score-only:

| Backend | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `linux_aarch64_neon`  | 48 | 28 | 1.113x | 1.101x |
| `linux_aarch64_sve2`† | 48 |  8 | 0.280x | 0.426x |

Score-only vs `generic` (raw SIMD lift, no parasail involved):

| Backend | Rows | Geomean | Median | Best |
| --- | ---: | ---: | ---: | ---: |
| `linux_aarch64_neon`  | 48 | 3.056x | 3.053x | 9.106x |
| `linux_aarch64_sve2`† | 48 | 0.768x | 0.849x | 5.235x |

### By variant (NEON / ASIMD vs parasail)

| Variant | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `nw-path-info`    |  8 | 8 | 2.050x | 1.937x |
| `sw-score`        | 16 | 16 | 1.494x | 1.496x |
| `nw-cigar`        |  8 | 6 | 1.323x | 1.477x |
| `sw-path-info`    |  8 | 7 | 1.165x | 1.170x |
| `sw-farrar-score` | 16 | 4 | 0.967x | 0.939x |
| `nw-score`        | 16 | 8 | 0.954x | 0.945x |
| `sw-cigar`        |  8 | 2 | 0.665x | 0.741x |

### Worst rows vs parasail (NEON)

| Ratio | Pass | Case | Shape | Variant | Width |
| ---: | --- | --- | --- | --- | ---: |
| 0.261x | chinese | affine | 1:1 | `sw-cigar` | 16 |
| 0.424x | english | affine | 1:1 | `sw-cigar` | 16 |
| 0.532x | english | affine | 1:1 | `nw-cigar` | 16 |
| 0.541x | chinese | linear | 1:1 | `sw-cigar` | 16 |
| 0.585x | chinese | affine | 1:1 | `nw-cigar` | 16 |

### Takeaways

NEON / ASIMD is the headline Graviton4 backend. It wins `sw-score`/`nw-path-info`
families decisively (~1.5x and ~2.0x geomean) and clears `1.0x` on most
score-only rows. The weak point is affine `sw-cigar` at width 16, where
parasail's striped trace-table representation runs roughly 2-4x ahead.

† The SVE and SVE2 numbers in this artifact are pre-fix. Commit `5174571`
(`SVE/SVE2 backend: 0.45x -> 1.04x/1.08x parasail on Graviton4`) gates
`shift_left_insert`/`first_lane_vector`/`add_sentinel` behind
`if constexpr (requires {...})`, provides SVE intrinsic overrides, pins SVE to
128-bit registers with `-msve-vector-bits=128`, adds a parallel prefix-carry
lazy-F, and routes affine SW/NW CIGAR/path-info through `profile_traceback`
plus ported NEON exact-fill kernels for the 1024-character query shape. After
that commit the live SVE/SVE2 backends are reported as `1.04x`/`1.08x`
parasail geomean respectively, but no fresh CSV has been captured. The
table-row numbers reflect what is in this tree's CSV, which is the broken
pre-fix snapshot — re-run on Graviton4 to refresh.

### Recommended Graviton4 next steps

1. Re-run the focused parasail sweep with the patched SVE/SVE2 path enabled and store as `graviton4-arm-simd-parasail-YYYY-MM-DD.csv` so the headline numbers in this file no longer carry the asterisk.
2. Target affine `sw-cigar` width 16 next — both NEON and the SVE/SVE2 fixes still leave that row at 0.26-0.59x parasail.
3. Drop one of `linux_aarch64_neon` / `linux_aarch64_asimd` from the published backend list now that they share kernels; keep one alias for compatibility.

## ARM macOS arm64 (Apple M-series) - 2026-05-14

Raw macOS arm64 artifacts:

| Artifact | Contents |
| --- | --- |
| [`benchmarks/macos-arm64-neon-score-native-2026-05-13.csv`](benchmarks/macos-arm64-neon-score-native-2026-05-13.csv) | Native `generic`, `swar`, and NEON score rows for `1:1` and `1:many`. |
| [`benchmarks/macos-arm64-neon-score-parasail-2026-05-13.csv`](benchmarks/macos-arm64-neon-score-parasail-2026-05-13.csv) | Score rows including locally installed parasail. |
| [`benchmarks/macos-arm64-neon-path-parasail-2026-05-13.csv`](benchmarks/macos-arm64-neon-path-parasail-2026-05-13.csv) | Path/CIGAR timing-split rows including parasail. |
| [`benchmarks/macos-arm64-neon-focused-2026-05-14.csv`](benchmarks/macos-arm64-neon-focused-2026-05-14.csv) | Focused post-optimization comparison against parasail. |
| [`benchmarks/macos-arm64-neon-microbench-2026-05-14.txt`](benchmarks/macos-arm64-neon-microbench-2026-05-14.txt) | Native NEON microbench. |
| [`benchmarks/macos-arm64-neon-sw-farrar-parasail-study-2026-05-14.csv`](benchmarks/macos-arm64-neon-sw-farrar-parasail-study-2026-05-14.csv) | Focused run after adding exact-fill linear SW Farrar score paths. |
| [`benchmarks/macos-arm64-neon-linear-trace-onepass-parasail-study-2026-05-14.csv`](benchmarks/macos-arm64-neon-linear-trace-onepass-parasail-study-2026-05-14.csv) | Negative-control one-pass striped linear SW trace experiment (reverted). |
| [`benchmarks/macos-arm64-neon-2026-05-13.md`](benchmarks/macos-arm64-neon-2026-05-13.md), [`benchmarks/macos-arm64-neon-2026-05-14.md`](benchmarks/macos-arm64-neon-2026-05-14.md) | macOS-specific notes and recommendations. |

Build context: macOS 15.3.1 on Apple M-series (host `wopr`), Python 3.13 from
Homebrew, Apple clang 17. Parasail is locally installed `parasail==1.3.4`
backed by parasail library `2.6.2`. Installing parasail from source on
homebrew required autotools/libtool with `/opt/homebrew/bin` on `PATH` for
`glibtoolize`.

This is a **different chip and toolchain** from Graviton4 — do not transfer
ratios between the two ARM sections.

### Score-only vs `generic` (native sweep)

| Backend | Rows | Wins | Geomean | Median | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `macos_arm64_neon` | 48 | 44 | 2.385x | 2.506x | 5.074x |
| `swar`             | 48 |  5 | 0.638x | 0.636x | 1.005x |

### Focused comparison vs parasail (2026-05-14)

| Group | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| Overall | 48 | 3 | 0.720x | 0.711x |
| Score-only | 32 | 0 | 0.683x | 0.680x |
| Path/CIGAR | 16 | 3 | 0.801x | 0.820x |

### Takeaways

NEON on Mac is a real lift over generic score code (`~2.4x` geomean), but is
not yet parasail-competitive — the focused 2026-05-14 sweep is `0.720x`
parasail geomean across 48 rows. Affine SW path/CIGAR improved materially
after exact-fill linear SW Farrar score paths landed (linear `sw-farrar-score`
moved `~1.45x` geomean over the prior mac focused sweep, with width 32 at
parasail parity for English/Chinese 1:1).

A one-pass striped linear SW trace experiment lost badly (`0.35x` parasail
geomean and `0.45x` of the prior NEON trace baseline) and was reverted.

### Recommended Mac M-series next steps

1. Keep the exact-fill linear SW Farrar score path enabled.
2. Work on `sw-farrar-score` width 16 before width 32.
3. Do not reuse the shared masked-trace helpers for NEON linear SW path/CIGAR without redesigning the trace representation.
4. Redesign affine `nw-score` with a NEON-specific loop if revisited; do not reuse the x86-oriented dense/plain flags (they were measured and lost).
5. Add a native parasail comparison mode to the arm64 microbench before instruction-level parity work.
6. Keep SWAR off the mac performance path (geomean `0.64x` generic) — correctness/reference only.

## Loongson LoongArch64 (LSX/LASX) - 2026-05-13

Raw Loongson artifacts:

| Artifact | Contents |
| --- | --- |
| [`benchmarks/loongson-score-native-2026-05-13.csv`](benchmarks/loongson-score-native-2026-05-13.csv) | Native `generic`, `swar`, LSX, and LASX score rows for `1:1` and `1:many`. |
| [`benchmarks/loongson-score-1to1-parasail-2026-05-13.csv`](benchmarks/loongson-score-1to1-parasail-2026-05-13.csv) | `sw-score`/`nw-score` `1:1` comparison against patched generic LoongArch parasail. |
| [`benchmarks/loongson-path-native-2026-05-13.csv`](benchmarks/loongson-path-native-2026-05-13.csv) | Native path/CIGAR timing-split rows, no parasail. |
| [`benchmarks/loongson-sw-farrar-exactfill-baseline-2026-05-14.csv`](benchmarks/loongson-sw-farrar-exactfill-baseline-2026-05-14.csv) | Focused linear `sw-farrar-score` baseline before exact-fill hooks. |
| [`benchmarks/loongson-sw-farrar-exactfill-study-2026-05-14.csv`](benchmarks/loongson-sw-farrar-exactfill-study-2026-05-14.csv) | Focused linear `sw-farrar-score` run after exact-fill hooks. |
| [`benchmarks/loongson-2026-05-13.md`](benchmarks/loongson-2026-05-13.md) | Loongson-specific notes and recommendations. |

Build context: Loongson 3A6000-class host, Python 3.13.13, GCC 15.2.0, CMake
4.3.2. The LoongArch Python extension modules were built with static C++
runtime linkage; `ldd` shows no dynamic `libstdc++`/`libgcc` dependency.

Parasail status: upstream `pip install parasail` failed on LoongArch. A
patched source build works for direct score calls after treating LoongArch as
a non-x86 `cpuid` stub target, but it is generic parasail, not LSX/LASX
optimized. Its profile API returned NULL profiles and trace/CIGAR was not
usable, so parasail is included only for direct `sw-score`/`nw-score` `1:1`
score rows.

### Score-only vs `generic` (native)

| Backend | Rows | Geomean | Median | Best |
| --- | ---: | ---: | ---: | ---: |
| `linux_loongarch64_lasx` | 48 | 6.220x | 6.781x | 16.052x |
| `linux_loongarch64_lsx`  | 48 | 4.234x | 3.926x |  8.858x |
| `swar`                   | 48 | 0.627x | 0.587x |  1.020x |

### Score-only vs patched LoongArch parasail (1:1)

| Backend | Rows | Wins | Geomean | Median | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linux_loongarch64_lasx` | 16 | 16 | 7.517x | 6.502x | 22.365x |
| `linux_loongarch64_lsx`  | 16 | 16 | 5.307x | 5.395x | 13.205x |
| `swar`                   | 16 |  8 | 1.061x | 1.161x |  2.201x |

### Path / CIGAR vs `generic` (native, no parasail)

| Backend | Rows | Geomean | Median |
| --- | ---: | ---: | ---: |
| `linux_loongarch64_lasx` | 32 | 1.244x | 1.055x |
| `linux_loongarch64_lsx`  | 32 | 1.177x | 1.206x |
| `swar`                   | 32 | 1.144x | 1.032x |

### Takeaways

LASX is the clear Loongson win: 6.2x geomean over generic on score-only, 7.5x
over patched parasail on 1:1 direct score calls, and up to 22.4x on the best
row. LSX trails LASX by roughly the expected 2x register-width factor.

Exact-fill linear `sw-farrar-score` hooks added after the mac NEON study
improved LSX `1.73x` geomean and LASX `1.63x` geomean over the pre-change
focused baseline. After the change, focused linear `sw-farrar-score` is
`10.24x` generic on LSX and `17.50x` generic on LASX.

Path/CIGAR is not uniformly better. Affine CIGAR is strong (LASX `~4.32x`
generic), but linear SW path/CIGAR is the weak point: the worst LSX linear SW
path row is `0.34x` generic and the worst LASX row is `0.48x` generic,
pointing at trace-table and materialization traffic rather than raw score DP
throughput.

SWAR is essentially a regression on score-only (`~0.63x` generic) but is
roughly at parity with parasail on the patched 1:1 score comparison (`1.06x`
geomean) — useful as a correctness reference, not as a performance path.

### Recommended Loongson next steps

1. Keep exact-fill LSX/LASX score hooks enabled — large score-only win.
2. Target a Loongson-specific linear SW trace/CIGAR redesign before doing instruction scheduling.
3. Start with LASX width 16/32 trace-traffic reduction, then port to LSX.
4. Add a native Loongson microbench/perf entrypoint before micro-optimizing LSX/LASX loops.
5. Treat parasail as a generic LoongArch comparison until a maintained LSX/LASX parasail build becomes available.

## PowerPC64 VSX (Power8 Linux) - 2026-05-17

Raw Power8 artifacts:

| Artifact | Contents |
| --- | --- |
| [`benchmarks/power8-vsx-2026-05-17.csv`](benchmarks/power8-vsx-2026-05-17.csv) | Native `generic`, `swar`, and `linux_powerpc64_vsx` rows for `english`/`chinese`, `linear`/`affine`, widths 16/32, `1:1` and `1:many`, with `--timing-split`. |
| [`benchmarks/power8-vsx-2026-05-17.md`](benchmarks/power8-vsx-2026-05-17.md) | Power8-specific notes, semantic-delta writeup, and recommendations. |

Build context: real POWER8 silicon (PVR `004b 0201`, 4.157 GHz), KVM-virtualized
as a single-core `pSeries` guest. Ubuntu 20.04 ppc64le, IBM Advance Toolchain
15.0 (GCC 11.4.1), Python 3.13.13 from miniforge, CMake 4.3.2 + Ninja from
pip. **Parasail was not built** (no upstream ppc64le wheel; source build not
attempted), so all ratios in this section are against `generic` on the same
machine.

### Overall vs `generic`

| Backend | Rows | Wins | Geomean | Median | Worst | Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `linux_powerpc64_vsx` | 80 | 73 | 3.652x | 4.126x | 0.960x | 16.821x |
| `swar`                | 80 | 28 | 0.767x | 0.975x | 0.410x |  1.598x |

### Score-only vs `generic`

| Backend | Rows | Wins | Geomean | Median | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linux_powerpc64_vsx` | 48 | 48 | 6.608x | 6.753x | 16.821x |
| `swar`                | 48 |  7 | 0.602x | 0.534x |  1.030x |

### Path / CIGAR vs `generic`

| Backend | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `linux_powerpc64_vsx` | 32 | 25 | 1.501x | 1.139x |
| `swar`                | 32 | 21 | 1.104x | 1.042x |

### By variant (VSX vs generic)

| Variant | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `sw-farrar-score` | 16 | 16 | 7.753x | 7.072x |
| `sw-score`        | 16 | 16 | 7.585x | 7.100x |
| `nw-score`        | 16 | 16 | 4.906x | 5.118x |
| `nw-cigar`        |  8 |  6 | 1.890x | 2.028x |
| `sw-cigar`        |  8 |  5 | 1.875x | 2.004x |
| `sw-path-info`    |  8 |  8 | 1.228x | 1.206x |
| `nw-path-info`    |  8 |  6 | 1.167x | 1.139x |

### Worst rows vs generic

| Ratio | Pass | Case | Shape | Variant | Width |
| ---: | --- | --- | --- | --- | ---: |
| 0.960x | english | linear | 1:1 | `nw-path-info` | 16 |
| 0.964x | chinese | linear | 1:1 | `sw-cigar` | 16 |
| 0.965x | english | linear | 1:1 | `sw-cigar` | 16 |
| 0.971x | chinese | linear | 1:1 | `nw-path-info` | 16 |
| 0.984x | english | linear | 1:1 | `nw-cigar` | 32 |

### Takeaways

Power8 VSX is a uniform win or tie. The `0.96x` worst case is a short 1:1
linear path-info row at width 16 where SIMD setup dominates. The biggest wins
(~16x) are `sw-farrar-score` and `sw-score` for both English and Chinese
1024x1024 at width 16. Affine `sw-cigar` and `nw-cigar` reach ~4.2x because
they route through the SIMD score kernel plus
`profile_traceback::affine_cigar_with_score` and skip the trace table.

Routing decisions that differ from x86_avx2 / Loongson: linear SW path /
path-info / linear CIGAR and affine path / path-info stay on
`profile_traceback` on Power8. The shared masked striped traceback wins on
ISAs with one-instruction lane bitmask extract and good striped-trace cache
behavior. On Power8 they measured 0.46-0.60x of the scalar byte-table path,
so the helper hooks (`trace_mask_*`, `mask_or`, `store_masked_cells`, the
`vbpermq`-based collapse described inline) are present but the public path /
path-info / linear CIGAR APIs route through `profile_traceback`. The affine
CIGAR entry stays on the SIMD score kernel + scalar reverse-build, beating
generic by 2-4x without a trace table.

SWAR regresses on Power8 across most rows (geomean `0.77x` generic). AT15 GCC
11.4 auto-vectorizes the generic score loop well enough that SWAR's 64-bit
packed lanes give no benefit. SWAR remains useful as a correctness/reference
backend only.

### Recommended Power8 next steps

1. Try a `vbpermq`-based `trace_mask_*` on real hardware; combined with a row-major linear SW trace table this might bring the masked path above `1.0x` of generic.
2. Add a Power8 `local_affine_score_exact_segment*_raw` mirroring the NEON helpers if the 1024-character query shape becomes a target; current `sw-farrar-score` is already 7-8x ahead of generic.
3. Re-bench on a multi-core / non-virtualized Power8 host to characterise SMT throughput and shared L2/L3 effects.
4. Build parasail from source for ppc64le and add a parasail column to the next sweep — every other family in this file has at least one parasail point of reference.
5. Investigate why SWAR loses to generic on Power8 via an asm dump of the generic score loop.

## Notes on comparing across families

These numbers are intended for engineering direction, not publication-grade
claims. Different families used different baselines (parasail where
available, otherwise `generic`), different sweep sizes, different host pinning
strategies, and different parasail builds (bundled wheel, locally compiled,
patched-for-LoongArch, or absent). For any cross-family claim, rerun the
relevant sweeps with matched conditions and use native microbench rows for
the specific kernels under discussion.
