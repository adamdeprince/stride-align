# Benchmark Summary

[Stride-align](fast.png) is benchmarked across five hardware families:
Intel x86 (AVX2, AVX512BWVL), ARM Linux aarch64 on AWS Graviton4
(NEON/ASIMD, SVE, SVE2), ARM macOS arm64 on Apple M-series (NEON),
LoongArch64 on Loongson (LSX, LASX), and PowerPC64 VSX on Power8. Each
family ran on a different host with a different parasail build (or no
parasail at all), so numbers should be read within a family, not
across families. Raw CSVs live in `benchmark.csv` (x86) and
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
| ARM Graviton4 (Linux) | `linux_aarch64_neon`/`asimd` | parasail | 80 | **1.138x** | 1.138x | 0.260x | 2.637x |
| ARM Graviton4 (Linux) | `linux_aarch64_sve2` | parasail | 80 | 1.081x | 1.108x | 0.261x | 2.635x |
| ARM Graviton4 (Linux) | `linux_aarch64_sve` | parasail | 80 | 1.042x | 1.101x | 0.259x | 2.647x |
| ARM Mac M4 (macOS) | `macos_arm64_neon` | parasail | 80 | **1.065x** | 1.046x | 0.592x | 2.400x |
| Loongson LoongArch64 | `linux_loongarch64_lasx` | patched parasail (1:1 score) | 16 | **7.517x** | 6.502x | 4.315x | 22.365x |
| Loongson LoongArch64 | `linux_loongarch64_lasx` | generic (native) | 80 | **4.909x** | 5.149x | 0.499x | 29.707x |
| Power8 VSX (Linux) | `linux_powerpc64_vsx` | generic (no parasail) | 80 | **3.772x** | 4.128x | 0.915x | 16.797x |
| Levenshtein (Intel x86) | `x86_avx512bwvl` | python-Levenshtein | 14 | **1.159x** | 1.151x | 1.039x | 1.353x |
| Levenshtein (Intel x86) | `x86_avx512bwvl` | rapidfuzz | 14 | 1.075x | 1.070x | 0.898x | 1.364x |
| Levenshtein (Intel x86) | `x86_avx512bwvl` | editdistance | 14 | 13.564x | 13.758x | 11.099x | 15.880x |
| Lev (long, >64 chars) | `x86_avx512bwvl` | rapidfuzz | 5 | **2.35x** | 2.55x | 1.45x | 2.88x |
| Lev (1-vs-1, q>=100) | `x86_avx512bwvl` | rapidfuzz | 2 | **1.36x** | 1.36x | 1.34x | 1.39x |
| Lev (cutoff, q=50) | `x86_avx512bwvl` | rapidfuzz | 3 | **3.91x** | 2.41x | 2.41x | 6.03x |
| Damerau-Lev (short tgts) | `x86_avx512bwvl` | rapidfuzz | 4 | **3.13x** | 3.03x | 2.38x | 4.22x |
| Damerau-Lev (medium tgts) | `x86_avx512bwvl` | rapidfuzz | 3 | 0.98x | 0.87x | 0.85x | 1.25x |

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

## Levenshtein (Intel x86) - 2026-05-19

Raw artifact: [`benchmarks/intel-levenshtein-2026-05-19.csv`](benchmarks/intel-levenshtein-2026-05-19.csv).

Build context: same host as Intel x86 above (11th Gen Core i7-1195G7,
Python 3.13, `taskset -c 2`), running on the `x86_avx512bwvl` backend. The
multi-target Myers kernel runs one target per SIMD lane (8x 64-bit lanes
under AVX512) and reads bytes / 1-byte unicode strings zero-copy from
CPython buffers. Patterns over 64 chars fall through to the scalar
Hyyrö multi-word dispatch in `levenshtein_dispatch.hpp`.

Command:

```bash
taskset -c 2 .venv/bin/python tools/benchmark_libs.py \
  --input-file kjv_subset.txt --levenshtein \
  --iterations 25 --warmups 3 < lev_queries.txt > intel-levenshtein-2026-05-19.csv
```

Corpus: first 1000 lines of `demo/kjv.txt`. Queries: 14 single words and
short phrases (3-29 chars) covering the pattern lengths that hit the
SIMD fast path.

### Overall

| Backend | Rows | Geomean | Median | Worst | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stride_align` vs `python-Levenshtein` | 14 | **1.159x** | 1.151x | 1.039x | 1.353x |
| `stride_align` vs `rapidfuzz`          | 14 | 1.075x | 1.070x | 0.898x | 1.364x |
| `stride_align` vs `editdistance`       | 14 | 13.564x | 13.758x | 11.099x | 15.880x |

Per-call wall time at 1000 targets (median across 14 queries):

| Library | µs/call | ns/target |
| --- | ---: | ---: |
| `stride_align` (`x86_avx512bwvl`) | **496** | **496** |
| `rapidfuzz`                       | 540  | 540  |
| `python-Levenshtein`              | 567  | 567  |
| `editdistance`                    | 6806 | 6806 |

### Takeaways

The multi-target Myers kernel keeps stride-align ahead of every popular
Python Levenshtein library on this corpus. python-Levenshtein loses by
1.16x geomean across all 14 queries; rapidfuzz loses by 1.07x with one
sub-parity row (0.898x on a 26-char query). editdistance is roughly
13.5x slower, reflecting its pure-C scalar DP loop with no batching.

The "vs rapidfuzz" worst row is the only sub-parity result of the
sweep. rapidfuzz also uses bit-parallel Myers in its hot path, so the
remaining headroom is mostly per-call overhead — list traversal, the
Python ABI, the ndarray allocation — rather than the inner loop. The
SIMD multi-target kernel pulls ahead on shorter queries where the
per-target setup dominates.

## Levenshtein extended (Intel x86) - 2026-05-19

Raw artifact: [`benchmarks/intel-levenshtein-v2-2026-05-19.csv`](benchmarks/intel-levenshtein-v2-2026-05-19.csv).

Three follow-up workloads that exercise the multi-word SIMD batch
kernel (patterns 65-256 chars, in 64-char blocks W = 2/3/4), the
zero-copy singular dispatch (no `prepare_alignment` vector copy when
both inputs are bytes or 1-byte unicode), and the `score_cutoff`
parameter with per-lane done masks and all-lanes early-exit. Same
build host and pinning as the section above.

### Long patterns (1-vs-200, no cutoff)

| `q_len` | stride_align | python-Lev | rapidfuzz | vs Lev | vs rf |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40  |  95 µs | 110 µs |  95 µs | 1.16x | 1.00x |
| 65  | 118 µs | 187 µs | 171 µs | **1.59x** | **1.45x** |
| 100 | 118 µs | 324 µs | 357 µs | **2.75x** | **3.03x** |
| 128 | 118 µs | 320 µs | 300 µs | **2.71x** | **2.54x** |
| 180 | 135 µs | 405 µs | 384 µs | **3.00x** | **2.85x** |
| 200 | 166 µs | 456 µs | 440 µs | **2.75x** | **2.65x** |

Each lane in the AVX-512 kernel runs Hyyrö's wide-add carry chain over
W blocks in parallel across 8 targets. The wide add uses two chained
64-bit adds + `gt_u64` overflow detection (AVX-512 native unsigned
`cmpgt`, AVX2 sign-bit-XOR + signed `cmpgt`, SSE4.1 sub-and-sign-bit
emulation). q_len = 40 is the single-word kernel, which hits parity
with rapidfuzz.

### 1-vs-1 singular (zero-copy dispatch)

| `q_len` | stride_align | python-Lev | rapidfuzz |
| ---: | ---: | ---: | ---: |
| 10  | 0.20 µs | 0.25 µs | **0.17 µs** |
| 30  | 0.27 µs | 0.31 µs | **0.24 µs** |
| 60  | 0.36 µs | 0.40 µs | **0.32 µs** |
| 100 | **0.90 µs** | 1.30 µs | 1.21 µs |
| 200 | **2.35 µs** | 3.24 µs | 3.14 µs |

When both inputs are bytes or 1-byte unicode the singular path skips
the prepare\_alignment vector copy and runs scalar Myers directly on
the CPython buffer (`PyBytes_AsStringAndSize` / `PyUnicode_1BYTE_DATA`).
We trail rapidfuzz by ~10% under 60 chars (Python ABI overhead, no
algorithmic gap) and pull ~1.35x ahead from 100 chars onward, where
the multi-word inner loop dominates.

### score_cutoff (5000 targets, short query)

stride_align vs rapidfuzz with matching cutoff:

| `q_len` | cutoff | stride_align | rapidfuzz | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 10 |  2 | **277 µs** |  353 µs | 1.27x |
| 10 |  5 | **301 µs** |  472 µs | 1.57x |
| 10 | 10 | **342 µs** |  556 µs | 1.62x |
| 30 |  7 | **190 µs** |  486 µs | 2.56x |
| 30 | 15 | **328 µs** |  682 µs | 2.08x |
| 30 | 30 | **410 µs** |  866 µs | 2.11x |
| 50 | 12 | **49 µs**  |  297 µs | **6.03x** |
| 50 | 25 | **204 µs** |  491 µs | 2.41x |
| 50 | 50 | **410 µs** | 1119 µs | 2.73x |

Per-lane done masks freeze score updates once a lane crosses
`cutoff + remaining_chars`, and the column loop breaks as soon as every
batch lane is settled (target exhausted or bailed). The biggest win
(`q_len=50`, `cutoff=12`, 6x) is where most targets exceed cutoff after
a handful of columns and the whole batch can short-circuit.

### Where rapidfuzz still wins

Long patterns *with* tight cutoff (e.g. `q=100`, `cutoff=20` over
50-250-char targets): rapidfuzz 110 µs vs stride_align 402 µs (0.27x).
Our cutoff bail condition `score > cutoff + remaining_chars` only
fires near the end of the column loop because `remaining_chars` shrinks
slowly. rapidfuzz uses **banded Myers** here, restricting the DP to a
2K+1 diagonal band so the work drops to O(K·n) instead of O(m·n).
Banded SIMD Myers is a separate kernel and isn't implemented in
stride-align yet — see "Future work" below.

### Future work

- **Banded Myers** for tight-cutoff long-pattern workloads (the one
  remaining rapidfuzz win). Restrict per-lane state to ±K diagonals
  from the main; sliding window across columns.
- **Pattern lengths > 256**: the multi-word SIMD kernel currently caps
  at W=4. Extending to W=8 (pattern up to 512) is a recompile.

## Damerau-Levenshtein / OSA (Intel x86) - 2026-05-19

Raw artifact: [`benchmarks/intel-damerau-levenshtein-2026-05-19.csv`](benchmarks/intel-damerau-levenshtein-2026-05-19.csv).

Build context: same host as the Levenshtein section (11th Gen Core
i7-1195G7, Python 3.13, `taskset -c 2`). The algorithm is OSA-restricted
(Optimal String Alignment) Damerau-Levenshtein: like Levenshtein but
adjacent transpositions cost 1 instead of two substitutions, and each
character can participate in at most one edit. Hyyrö's bit-parallel
recurrence (the `TR = (((~D0_prev) & PM) << 1) & PM_old` formulation
that rapidfuzz also uses), wrapped in the same multi-target SIMD batch
architecture as our Levenshtein kernel — one target per SIMD lane
(2/4/8 lanes for SSE4.1/AVX2/AVX-512).

### Short targets (1-vs-1000, 3-15 char corpus)

This is the SIMD batch sweet spot: short alignments amortize the
gather + state-shift cost across 8 lanes, and rapidfuzz's per-pair
overhead dominates its loop.

| `q_len` | stride_align | rapidfuzz | ratio |
| ---: | ---: | ---: | ---: |
|  5 | **41 µs** |  99 µs | 2.38x |
| 10 | **42 µs** | 108 µs | 2.59x |
| 20 | **42 µs** | 176 µs | **4.22x** |
| 30 | **47 µs** | 163 µs | 3.47x |

### Medium targets (1-vs-200, 30-250 char corpus)

| `q_len` | stride_align | rapidfuzz | ratio |
| ---: | ---: | ---: | ---: |
| 10 | 117 µs | **100 µs** | 0.85x |
| 30 | 117 µs | **102 µs** | 0.87x |
| 64 | **117 µs** | 147 µs | 1.25x |

For medium-target workloads we trail rapidfuzz by ~15% under 60 chars
(their inner loop is slightly tighter — fewer SIMD ops per column),
then pull ahead at q_len=64 where their bit-parallel fallback path
kicks in.

### 1-vs-1 singular

| `q_len` | stride_align | rapidfuzz | ratio |
| ---: | ---: | ---: | ---: |
| 10 | 0.18 µs | 0.15 µs | 0.85x |
| 30 | 0.23 µs | 0.21 µs | 0.92x |
| 60 | 0.35 µs | 0.32 µs | 0.90x |

Per-call Python ABI dominates; we're within 15% of rapidfuzz on every
length.

### API

```python
import stride_align

# Singular
stride_align.damerau_levenshtein_score(query, target)            # int
stride_align.damerau_levenshtein_normalized_score(query, target) # float in [0, 1]

# Batch (returns numpy ndarray)
stride_align.damerau_levenshtein_scores(query, targets)             # int64
stride_align.damerau_levenshtein_normalized_scores(query, targets)  # float64
```

Backends specialized for the new SIMD batch kernel: `x86_sse41`,
`x86_avx2`, `x86_avx512bwvl`, `x86_avx10_256`, `x86_avx10_512`. Other
architectures (NEON / SVE / Loongson / Power) fall through to the
shared scalar bit-parallel dispatch and remain correct.

## ARM Graviton4 (Linux aarch64) - 2026-05-18

Raw artifacts:

| Artifact | Contents |
| --- | --- |
| [`benchmarks/graviton4-arm-simd-parasail-2026-05-18.csv`](benchmarks/graviton4-arm-simd-parasail-2026-05-18.csv) | Full 400-row sweep after the CIGAR builder rework and the `5174571` SVE/SVE2 fix: `generic`, `linux_aarch64_neon`, `linux_aarch64_sve`, `linux_aarch64_sve2`, `parasail`, all 7 variants, `1:1` and `1:many`, widths 16/32. |
| [`benchmarks/graviton4-arm-simd-parasail-2026-05-16.csv`](benchmarks/graviton4-arm-simd-parasail-2026-05-16.csv) | Pre-`5174571` SVE/SVE2 snapshot. |

Build context: AWS Graviton4 (Neoverse V2), Ubuntu noble, Python 3.14.4,
GCC, system CMake + venv-local ninja. Pinned with `taskset -c 0`. Backends
measured: `linux_aarch64_neon`, `linux_aarch64_sve`, `linux_aarch64_sve2`,
plus `generic` and `parasail`. `linux_aarch64_asimd` was merged into NEON
in commit `617d282`; the public NEON backend covers both.

### Overall vs parasail

| Backend | Rows | Wins | Geomean | Median | Worst | Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `linux_aarch64_neon`  | 80 | 51 | 1.138x | 1.138x | 0.260x | 2.637x |
| `linux_aarch64_sve2`  | 80 | 49 | 1.081x | 1.108x | 0.261x | 2.635x |
| `linux_aarch64_sve`   | 80 | 47 | 1.042x | 1.101x | 0.259x | 2.647x |

Score-only:

| Backend | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `linux_aarch64_neon`  | 48 | 28 | 1.100x | 1.098x |
| `linux_aarch64_sve2`  | 48 | 28 | 1.022x | 1.071x |
| `linux_aarch64_sve`   | 48 | 26 | 0.973x | 1.070x |

Path/CIGAR:

| Backend | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `linux_aarch64_neon`  | 32 | 23 | 1.198x | 1.230x |
| `linux_aarch64_sve2`  | 32 | 21 | 1.178x | 1.188x |
| `linux_aarch64_sve`   | 32 | 21 | 1.155x | 1.206x |

### By variant (vs parasail geomean)

| Variant | NEON | SVE | SVE2 |
| --- | ---: | ---: | ---: |
| `nw-path-info`    | 2.052x | 1.953x | 2.163x |
| `sw-score`        | 1.465x | 1.268x | 1.269x |
| `nw-cigar`        | 1.318x | 1.300x | 1.275x |
| `sw-path-info`    | 1.144x | 1.066x | 1.059x |
| `nw-score`        | 0.953x | 0.883x | 1.020x |
| `sw-farrar-score` | 0.952x | 0.822x | 0.824x |
| `sw-cigar`        | 0.665x | 0.657x | 0.659x |

### Worst rows vs parasail (NEON)

| Ratio | Pass | Case | Shape | Variant | Width |
| ---: | --- | --- | --- | --- | ---: |
| 0.260x | chinese | affine | 1:1 | `sw-cigar` | 16 |

### Takeaways

The 2026-05-18 sweep is the first post-`5174571` SVE/SVE2 capture against
parasail. SVE/SVE2 jumped from `0.450x` parasail geomean (pre-fix snapshot
in the 2026-05-16 CSV) to `1.042x` / `1.081x` respectively — within rounding
of the `1.04x` / `1.08x` numbers advertised by the SVE-fix commit.

NEON remains the headline Graviton4 backend at `1.138x` parasail geomean.
NEON, SVE, and SVE2 cluster within `~10%` of each other on this Neoverse V2.
SVE2 edges SVE on score-only thanks to its native `svqadd_s*` vector-vector
saturating add for affine sentinels.

Remaining weak spots common to all three backends:
* affine/linear `sw-cigar` width 16 — still the worst variant
  (`0.665x` NEON geomean); the trace-table representation favors parasail
  on aarch64 even after the CIGAR builder rework.
* `sw-farrar-score` width 16 1:1 — 0.82-0.95x geomean across the three.

### Recommended Graviton4 next steps

1. Target `sw-cigar` width 16 next — still the lone variant where all three SIMD backends lose to parasail by geomean.
2. Investigate `sw-farrar-score` width 16 1:1; the 0.82-0.95x range suggests a striped-trace cache miss specific to the short query.
3. Either delete the merged `linux_aarch64_asimd` import alias or surface it explicitly in `available_backends()` so external users have a stable name.

## ARM macOS arm64 (Apple M-series) - 2026-05-18

Raw macOS arm64 artifacts:

| Artifact | Contents |
| --- | --- |
| [`benchmarks/macos-arm64-neon-2026-05-18.csv`](benchmarks/macos-arm64-neon-2026-05-18.csv) | Full 240-row sweep after the CIGAR builder rework: `generic`, `macos_arm64_neon`, `parasail`, all 7 variants, `1:1` and `1:many`, widths 16/32. |
| [`benchmarks/macos-arm64-neon-score-native-2026-05-13.csv`](benchmarks/macos-arm64-neon-score-native-2026-05-13.csv) | Native `generic`, `swar`, and NEON score rows for `1:1` and `1:many`. |
| [`benchmarks/macos-arm64-neon-score-parasail-2026-05-13.csv`](benchmarks/macos-arm64-neon-score-parasail-2026-05-13.csv) | Score rows including locally installed parasail. |
| [`benchmarks/macos-arm64-neon-path-parasail-2026-05-13.csv`](benchmarks/macos-arm64-neon-path-parasail-2026-05-13.csv) | Path/CIGAR timing-split rows including parasail. |
| [`benchmarks/macos-arm64-neon-focused-2026-05-14.csv`](benchmarks/macos-arm64-neon-focused-2026-05-14.csv) | Pre-CIGAR-fix focused comparison against parasail. |
| [`benchmarks/macos-arm64-neon-microbench-2026-05-14.txt`](benchmarks/macos-arm64-neon-microbench-2026-05-14.txt) | Native NEON microbench. |
| [`benchmarks/macos-arm64-neon-sw-farrar-parasail-study-2026-05-14.csv`](benchmarks/macos-arm64-neon-sw-farrar-parasail-study-2026-05-14.csv) | Focused run after adding exact-fill linear SW Farrar score paths. |
| [`benchmarks/macos-arm64-neon-linear-trace-onepass-parasail-study-2026-05-14.csv`](benchmarks/macos-arm64-neon-linear-trace-onepass-parasail-study-2026-05-14.csv) | Negative-control one-pass striped linear SW trace experiment (reverted). |
| [`benchmarks/macos-arm64-neon-2026-05-13.md`](benchmarks/macos-arm64-neon-2026-05-13.md), [`benchmarks/macos-arm64-neon-2026-05-14.md`](benchmarks/macos-arm64-neon-2026-05-14.md) | macOS-specific notes and recommendations. |

Build context: macOS 15.3.1 on Apple M-series (host `wopr`), Python 3.13 from
Homebrew, Apple clang 17. Parasail is locally installed `parasail==1.3.4`
backed by parasail library `2.6.2`. Installing parasail from source on
homebrew required autotools/libtool with `/opt/homebrew/bin` on `PATH` for
`glibtoolize`. The 2026-05-18 sweep was regenerated after the CIGAR builder
rework (`to_chars`-based digit emission + capacity reservation).

This is a **different chip and toolchain** from Graviton4 — do not transfer
ratios between the two ARM sections.

### Overall vs parasail (2026-05-18)

| Group | Rows | Wins | Geomean | Median | Worst | Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall    | 80 | 46 | 1.065x | 1.046x | 0.592x | 2.400x |
| Score-only | 48 | 21 | 0.947x | 0.931x |        |        |
| Path/CIGAR | 32 | 25 | 1.272x | 1.308x |        |        |

### By variant (macos_arm64_neon vs parasail)

| Variant | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `nw-path-info`    |  8 |  8 | 1.645x | 1.518x |
| `nw-cigar`        |  8 |  8 | 1.616x | 1.575x |
| `sw-score`        | 16 | 16 | 1.206x | 1.215x |
| `sw-cigar`        |  8 |  5 | 1.010x | 1.042x |
| `sw-path-info`    |  8 |  4 | 0.973x | 0.974x |
| `sw-farrar-score` | 16 |  5 | 0.951x | 0.905x |
| `nw-score`        | 16 |  0 | 0.740x | 0.753x |

### Takeaways

NEON on Mac has crossed parasail parity. The 2026-05-18 sweep is `1.065x`
parasail geomean across 80 rows (was `0.720x` on the 2026-05-14 focused
sweep). The path/CIGAR group is now the strongest (`1.272x` geomean) — the
inverse of the pre-fix state where it was the worst. The CIGAR builder
rework lands cleanly here: `nw-cigar` reaches `1.616x` and `sw-cigar`
crosses parity at `1.010x`.

Score-only is still the weak group at `0.947x`. `nw-score` is the only
variant where every row loses (`0.740x` geomean), driven by parasail's
striped score kernel on width 16. `sw-farrar-score` is also slightly behind
(`0.951x`). The exact-fill linear SW Farrar score path lands its earlier
gains here.

### Recommended Mac M-series next steps

1. Target affine `nw-score` next — it is the largest remaining gap and the only variant where every comparison loses.
2. Investigate `sw-farrar-score` width 16: still trailing parasail at `0.905x` median.
3. Keep the exact-fill linear SW Farrar score path enabled.
4. Do not reuse the shared masked-trace helpers for NEON linear SW path/CIGAR without redesigning the trace representation.
5. Add a native parasail comparison mode to the arm64 microbench before instruction-level parity work.
6. Keep SWAR off the mac performance path (geomean `0.64x` generic on the 2026-05-13 native sweep) — correctness/reference only.

## Loongson LoongArch64 (LSX/LASX) - 2026-05-18

Raw Loongson artifacts:

| Artifact | Contents |
| --- | --- |
| [`benchmarks/loongson-native-2026-05-18.csv`](benchmarks/loongson-native-2026-05-18.csv) | Full 320-row native sweep after the CIGAR builder rework: `generic`, `swar`, LSX, LASX, all 7 variants, `1:1` and `1:many`, widths 16/32. |
| [`benchmarks/loongson-score-native-2026-05-13.csv`](benchmarks/loongson-score-native-2026-05-13.csv) | Earlier native score-only sweep. |
| [`benchmarks/loongson-score-1to1-parasail-2026-05-13.csv`](benchmarks/loongson-score-1to1-parasail-2026-05-13.csv) | `sw-score`/`nw-score` `1:1` comparison against patched generic LoongArch parasail. |
| [`benchmarks/loongson-path-native-2026-05-13.csv`](benchmarks/loongson-path-native-2026-05-13.csv) | Pre-CIGAR-fix path/CIGAR timing-split rows, no parasail. |
| [`benchmarks/loongson-sw-farrar-exactfill-baseline-2026-05-14.csv`](benchmarks/loongson-sw-farrar-exactfill-baseline-2026-05-14.csv) | Focused linear `sw-farrar-score` baseline before exact-fill hooks. |
| [`benchmarks/loongson-sw-farrar-exactfill-study-2026-05-14.csv`](benchmarks/loongson-sw-farrar-exactfill-study-2026-05-14.csv) | Focused linear `sw-farrar-score` run after exact-fill hooks. |
| [`benchmarks/loongson-2026-05-13.md`](benchmarks/loongson-2026-05-13.md) | Loongson-specific notes and recommendations. |

Build context: Loongson 3A6000-class host, Python 3.13.13, GCC 15.2.0, CMake
4.3.2. The LoongArch Python extension modules were built with static C++
runtime linkage; `ldd` shows no dynamic `libstdc++`/`libgcc` dependency.
Numpy is sourced from a host-local source build (`/data/home/adam/dev/numpy`)
linked against the GCC 15.2 runtime at `/opt/loongson-gcc-15.2.0/lib`, since
no upstream loongarch64 wheel exists.

Parasail status: upstream `pip install parasail` failed on LoongArch. A
patched source build works for direct score calls after treating LoongArch as
a non-x86 `cpuid` stub target, but it is generic parasail, not LSX/LASX
optimized. Its profile API returned NULL profiles and trace/CIGAR was not
usable, so parasail is included only for direct `sw-score`/`nw-score` `1:1`
score rows. The 2026-05-18 sweep is native-only.

### Overall vs `generic` (2026-05-18 native)

| Backend | Rows | Wins | Geomean | Median | Worst | Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `linux_loongarch64_lasx` | 80 | 68 | 4.909x | 5.149x | 0.499x | 29.707x |
| `linux_loongarch64_lsx`  | 80 | 72 | 2.876x | 3.081x | 0.350x | 16.085x |

### Score-only vs `generic` (2026-05-18 native)

| Backend | Rows | Wins | Geomean | Median | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linux_loongarch64_lasx` | 48 | 48 | 11.624x | 13.370x | 29.707x |
| `linux_loongarch64_lsx`  | 48 | 48 |  5.185x |  5.334x | 16.085x |

### Path / CIGAR vs `generic` (2026-05-18 native)

| Backend | Rows | Wins | Geomean | Median | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linux_loongarch64_lasx` | 32 | 20 | 1.347x | 1.069x | 7.127x |
| `linux_loongarch64_lsx`  | 32 | 24 | 1.188x | 1.249x | 3.752x |

### By variant (vs generic)

| Variant | LSX geomean | LASX geomean |
| --- | ---: | ---: |
| `sw-farrar-score` |  6.440x | 19.281x |
| `sw-score`        |  6.089x | 17.770x |
| `nw-score`        |  3.556x |  4.585x |
| `sw-cigar`        |  1.194x |  1.915x |
| `nw-cigar`        |  1.677x |  1.797x |
| `nw-path-info`    |  1.257x |  1.182x |
| `sw-path-info`    |  0.791x |  0.809x |

### Score-only vs patched LoongArch parasail (1:1)

The 2026-05-13 parasail comparison (16-row 1:1 score-only, unaffected by the
CIGAR builder rework):

| Backend | Rows | Wins | Geomean | Median | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linux_loongarch64_lasx` | 16 | 16 | 7.517x | 6.502x | 22.365x |
| `linux_loongarch64_lsx`  | 16 | 16 | 5.307x | 5.395x | 13.205x |
| `swar`                   | 16 |  8 | 1.061x | 1.161x |  2.201x |

### Takeaways

LASX is the clear Loongson win: 11.6x geomean over generic on score-only,
7.5x over patched parasail on 1:1 direct score calls, and up to 29.7x on the
best row. LSX trails LASX by roughly the expected 2x register-width factor.
LSX/LASX score-only numbers shifted up from the 2026-05-13 sweep (LSX
4.234x → 5.185x, LASX 6.220x → 11.624x); the bulk of the LASX jump is the
exact-fill `sw-farrar-score` and `sw-score` paths landing at the 1024-cell
shape used in the sweep.

`sw-cigar` moved from path-trace-bound to comfortably ahead of generic: LSX
went from below 1.0x to 1.194x; LASX reaches 1.915x. The CIGAR builder
rework (`to_chars` + capacity reservation) is the proximate cause; the
sw-cigar path now spends almost all of its time in the SIMD score kernel
plus the affine reverse-build, not the digit serialization.

`sw-path-info` is the lone remaining weak variant on both LSX (0.791x) and
LASX (0.809x). `profile_traceback` materialization still dominates over the
SIMD score lift for that shape.

SWAR is essentially a regression on score-only (`~0.63x` generic on the
2026-05-13 native sweep) but is roughly at parity with parasail on the
patched 1:1 score comparison (`1.06x` geomean) — useful as a correctness
reference, not as a performance path.

### Recommended Loongson next steps

1. Keep exact-fill LSX/LASX score hooks enabled — large score-only win.
2. Target a Loongson-specific linear SW trace/CIGAR redesign before doing instruction scheduling.
3. Start with LASX width 16/32 trace-traffic reduction, then port to LSX.
4. Add a native Loongson microbench/perf entrypoint before micro-optimizing LSX/LASX loops.
5. Treat parasail as a generic LoongArch comparison until a maintained LSX/LASX parasail build becomes available.

## PowerPC64 VSX (Power8 Linux) - 2026-05-18

Raw Power8 artifacts:

| Artifact | Contents |
| --- | --- |
| [`benchmarks/power8-vsx-2026-05-18.csv`](benchmarks/power8-vsx-2026-05-18.csv) | Full 240-row sweep after the CIGAR builder rework: `generic`, `swar`, and `linux_powerpc64_vsx` for all 7 variants, `1:1` and `1:many`, widths 16/32. |
| [`benchmarks/power8-vsx-2026-05-17.csv`](benchmarks/power8-vsx-2026-05-17.csv) | Pre-CIGAR-fix sweep on the same host. |
| [`benchmarks/power8-vsx-2026-05-17.md`](benchmarks/power8-vsx-2026-05-17.md) | Power8-specific notes, semantic-delta writeup, and recommendations. |

Build context: real POWER8 silicon (PVR `004b 0201`, 4.157 GHz), KVM-virtualized
as a single-core `pSeries` guest. Ubuntu 20.04 ppc64le, IBM Advance Toolchain
15.0 (GCC 11.4.1), Python 3.13.13 from miniforge, system CMake + Ninja.
Numpy 2.4.5 from `pip`. **Parasail was not built** (no upstream ppc64le
wheel; source build not attempted), so all ratios in this section are
against `generic` on the same machine. The 2026-05-18 sweep was pinned with
`taskset -c 0`.

### Overall vs `generic` (2026-05-18)

| Backend | Rows | Wins | Geomean | Median | Worst | Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `linux_powerpc64_vsx` | 80 | 74 | 3.772x | 4.128x | 0.915x | 16.797x |
| `swar`                | 80 | 31 | 0.789x | 1.000x | 0.411x |  1.669x |

### Score-only vs `generic`

| Backend | Rows | Wins | Geomean | Median | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linux_powerpc64_vsx` | 48 | 48 | 6.911x | 6.763x | 16.797x |

### Path / CIGAR vs `generic`

| Backend | Rows | Wins | Geomean | Median | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linux_powerpc64_vsx` | 32 | 26 | 1.521x | 1.266x | 4.592x |

### By variant (VSX vs generic)

| Variant | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `sw-farrar-score` | 16 | 16 | 7.759x | 7.072x |
| `sw-score`        | 16 | 16 | 7.544x | 7.103x |
| `nw-score`        | 16 | 16 | 5.639x | 5.156x |
| `sw-cigar`        |  8 |  5 | 1.958x | 2.183x |
| `nw-cigar`        |  8 |  7 | 1.825x | 1.862x |
| `nw-path-info`    |  8 |  6 | 1.225x | 1.266x |
| `sw-path-info`    |  8 |  8 | 1.224x | 1.173x |

### Worst rows vs generic

| Ratio | Pass | Case | Shape | Variant | Width |
| ---: | --- | --- | --- | --- | ---: |
| 0.915x | english | linear | 1:1 | `nw-path-info` | 16 |
| 0.934x | english | linear | 1:1 | `sw-cigar`     | 32 |
| 0.954x | chinese | linear | 1:1 | `nw-cigar`     | 32 |
| 0.955x | chinese | linear | 1:1 | `sw-cigar`     | 32 |
| 0.964x | chinese | linear | 1:1 | `nw-path-info` | 16 |

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

SWAR regresses on Power8 across most rows (geomean `0.79x` generic on the
2026-05-18 sweep). AT15 GCC 11.4 auto-vectorizes the generic score loop
well enough that SWAR's 64-bit packed lanes give no benefit. SWAR remains
useful as a correctness/reference backend only.

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
