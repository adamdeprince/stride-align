# Benchmarks

<div class="benchmark-hero" id="current-results">
  <article><span>AVX-512 / PARASAIL</span><strong>1.682×</strong><small>overall geomean · 80 rows</small></article>
  <article><span>COMPARABLE ROWS</span><strong>79/80</strong><small>won on Intel Granite Rapids</small></article>
  <article><span>PATH + CIGAR</span><strong>1.883×</strong><small>geomean · 32/32 wins</small></article>
  <article><span>CHINESE TEXT</span><strong>1.548×</strong><small>geomean · 39/40 wins</small></article>
</div>

**Reviewed 2026-07-18.** The lead results use the current pinned Intel
AVX-512 sweep. Every headline below names its baseline, host, date, and raw
artifact. Older architecture runs remain on this page as dated reference
measurements; they are not blended into a cross-machine average.

[Download the canonical Intel CSV](benchmark.csv) ·
[Download the immutable 2026-07-18 snapshot](benchmarks/intel-avx512-parasail-2026-07-18.csv) ·
[Inspect the benchmark CLI](https://github.com/adamdeprince/stride-align/blob/main/src/stride_align/benchmark.py)

## Read this first

A ratio above `1.0x` means stride-align is faster. Ratios always divide the
named baseline's median runtime by stride-align's median runtime:

```text
ratio = baseline_median_seconds / stride_align_median_seconds
```

Only rows from the same host, corpus, scoring mode, and output contract are
combined. Parasail comparisons pair score with score and trace/CIGAR with
path or CIGAR output. Native Loongson and Power8 tables use stride-align's
portable `generic` backend when no comparable optimized third-party backend
exists. Those native speedups are useful within that host, but they are not
Parasail or rapidfuzz claims.

## Current results

### Intel AVX-512 vs Parasail

The 2026-07-18 sweep covers English and Chinese inputs, linear and affine
gaps, 16- and 32-bit scores, `1:1` and `1:many` shapes, score-only kernels,
traceback, and CIGAR generation.

| Slice | Rows | Wins | Geomean | Median | Worst | Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 80 | **79** | **1.682x** | 1.656x | 0.874x | 3.629x |
| English | 40 | **40** | **1.827x** | 1.766x | 1.015x | 3.365x |
| Chinese | 40 | **39** | **1.548x** | 1.459x | 0.874x | 3.629x |
| Path / CIGAR | 32 | **32** | **1.883x** | 1.737x | 1.043x | 3.629x |

The one loss is 16-bit, linear-gap, `1:1` Needleman-Wunsch score on the
Chinese pass at 0.874x. It is reported rather than hidden. Python's UCS-2
and UCS-4 objects are first-class SIMD inputs; the Chinese pass does not
benchmark UTF-8 byte representations or charge an external transcode to the
competitor.

### Current companion benchmarks

| Workload | Host / backend | Baseline | Rows | Geomean | Worst | Best | Artifact |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Levenshtein, 1 query × 1,000 KJV targets | Intel AVX-512 | rapidfuzz | 14 | **3.749x** | 3.690x | 3.800x | [CSV](benchmarks/intel-levenshtein-2026-07-17.csv) |
| OSA/Damerau, short targets | Intel AVX-512 | rapidfuzz OSA | 4 | **7.703x** | 6.732x | 9.000x | [CSV](benchmarks/intel-damerau-levenshtein-2026-07-17.csv) |
| Score-only alignment | Loongson LASX | stride-align `generic` | 48 | **11.905x** | 2.928x | 33.108x | [CSV](benchmarks/loongson-native-2026-07-17.csv) |

The Loongson row is intentionally an architecture aside: it measures the
LASX backend against this project's portable implementation, not against
rapidfuzz or Parasail.

## Freshness ledger

| Dataset | Latest artifact | Status on this page |
| --- | --- | --- |
| Intel alignment vs Parasail | [2026-07-18](benchmarks/intel-avx512-parasail-2026-07-18.csv) | **Current lead evidence** |
| Intel Levenshtein and OSA vs rapidfuzz | [Levenshtein](benchmarks/intel-levenshtein-2026-07-17.csv) · [OSA](benchmarks/intel-damerau-levenshtein-2026-07-17.csv) | **Current** |
| Loongson LSX/LASX vs generic | [2026-07-17](benchmarks/loongson-native-2026-07-17.csv) | **Current architecture reference** |
| rapidfuzz shim full surface | [Mac](benchmarks/shim-full-mac-2026-06-17.json) · [Intel AVX10](benchmarks/shim-full-avx10-2026-06-17.json) · [Loongson 2026-06-10](benchmarks/shim-full-loongson-2026-06-10.json) | Reference only; AVX10 is de-selected |
| Graviton4 alignment vs Parasail | [2026-05-18](benchmarks/graviton4-arm-simd-parasail-2026-05-18.csv) | Dated architecture reference |
| Apple M-series alignment vs Parasail | [2026-05-18](benchmarks/macos-arm64-neon-2026-05-18.csv) | Dated architecture reference |
| Power8 VSX vs generic | [2026-05-18](benchmarks/power8-vsx-2026-05-18.csv) | Dated architecture reference |
| Loongson score-only vs patched Parasail | [2026-05-13](benchmarks/loongson-score-1to1-parasail-2026-05-13.csv) | Historical baseline only |

The detailed sections below retain the full methodology and weak rows. A
newer artifact must replace both its detailed section and its ledger entry;
old runs are never silently presented as current.

## Intel x86 - 2026-07-18

Raw artifacts: [`benchmark.csv`](benchmark.csv) is the current Intel sweep;
[`benchmarks/intel-avx512-parasail-2026-07-18.csv`](benchmarks/intel-avx512-parasail-2026-07-18.csv)
is the immutable dated snapshot of the same data.

Build context: Intel Xeon 6975P-C (Granite Rapids, 4 vCPU on AWS host
`avx10`), Python 3.14 in the project virtualenv, host pinned with
`taskset -c 2`, compiled with GCC 16.1, regenerated 2026-07-18 after the
0.6.0 striped SW kernel rework (affine prefix-lazy-F speedup + sound
linear-SW lazy-F correction; see
[`docs/known-issue-bounded-lazy-f-scan.md`](docs/known-issue-bounded-lazy-f-scan.md)).
Parasail is the bundled `parasail==1.3.4` wheel. The CSV contains 320 data
rows: English and Chinese workloads, linear and affine scoring, widths 16
and 32, `1:1` and `1:many` shapes, and
`generic`/`x86_avx2`/`x86_avx512bwvl`/`parasail` backends.
`x86_avx512bwvl` is the auto-selected backend here; the AVX10.1 backends are
de-selected because current GCC AVX10 codegen compiles the alignment kernels
~2x slower than the classic `avx512f/bw/vl` path (see `CMakeLists.txt`).

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
| `x86_avx512bwvl` | 80 | 79 | 1.682x | 1.656x | 0.874x | 3.629x |
| `x86_avx2` | 80 | 55 | 1.319x | 1.248x | 0.517x | 3.452x |
| `generic` | 80 |  8 | 0.259x | 0.200x | 0.082x | 1.732x |

Score-only rows (16 sw-farrar-score + 16 sw-score + 16 nw-score):

| Backend | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `x86_avx512bwvl` | 48 | 47 | 1.559x | 1.558x |
| `x86_avx2` | 48 | 29 | 1.159x | 1.189x |

Path/CIGAR rows (8 each of sw-path-info, nw-path-info, sw-cigar, nw-cigar):

| Backend | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `x86_avx512bwvl` | 32 | 32 | 1.883x | 1.737x |
| `x86_avx2` | 32 | 26 | 1.601x | 1.602x |

### By variant

AVX512BWVL:

| Variant | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `nw-path-info`    |  8 |  8 | 2.356x | 2.495x |
| `nw-cigar`        |  8 |  8 | 2.341x | 2.472x |
| `sw-score`        | 16 | 16 | 1.989x | 2.022x |
| `sw-cigar`        |  8 |  8 | 1.511x | 1.550x |
| `sw-path-info`    |  8 |  8 | 1.510x | 1.552x |
| `nw-score`        | 16 | 15 | 1.409x | 1.454x |
| `sw-farrar-score` | 16 | 16 | 1.353x | 1.118x |

AVX2:

| Variant | Rows | Wins | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `nw-path-info`    |  8 |  8 | 2.266x | 2.317x |
| `nw-cigar`        |  8 |  8 | 2.258x | 2.286x |
| `sw-score`        | 16 | 16 | 1.579x | 1.631x |
| `sw-path-info`    |  8 |  5 | 1.134x | 1.211x |
| `sw-cigar`        |  8 |  5 | 1.131x | 1.208x |
| `sw-farrar-score` | 16 |  6 | 1.041x | 0.885x |
| `nw-score`        | 16 |  7 | 0.946x | 0.985x |

### Worst rows vs parasail

AVX512BWVL:

| Ratio | Pass | Case | Shape | Variant | Width |
| ---: | --- | --- | --- | --- | ---: |
| 0.874x | chinese | linear | 1:1 | `nw-score`         | 16 |
| 1.015x | english | linear | 1:many | `sw-farrar-score` | 32 |
| 1.032x | chinese | linear | 1:many | `sw-farrar-score` | 32 |
| 1.035x | chinese | linear | 1:1 | `sw-farrar-score`   | 32 |
| 1.036x | english | linear | 1:1 | `sw-farrar-score`   | 32 |

AVX2:

| Ratio | Pass | Case | Shape | Variant | Width |
| ---: | --- | --- | --- | --- | ---: |
| 0.517x | chinese | linear | 1:1 | `sw-path-info` | 16 |
| 0.519x | chinese | linear | 1:1 | `sw-cigar`     | 16 |
| 0.672x | chinese | linear | 1:1 | `nw-score`     | 32 |
| 0.709x | chinese | linear | 1:1 | `nw-score`     | 16 |
| 0.760x | english | linear | 1:1 | `nw-score`     | 32 |

### Takeaways

AVX512BWVL is still the strongest Intel backend, winning **79 of 80**
comparable rows (47 of 48 score-only) at **1.682x** parasail geomean.
Path/CIGAR is a clean sweep (32/32, 1.883x geomean). The only sub-parity
row is short Chinese linear `nw-score` at 0.874x. The linear
`sw-farrar-score` cells now all sit above 1.0x despite the residual cost of
the 0.6.0 sound lazy-F correction on AVX-512 (see
[`TODO.md`](TODO.md) item 2 and the README kernel-update note). Affine and
path/CIGAR variants are unaffected or improved. AVX2 wins 55 of 80 at
1.319x geomean; its remaining sub-parity rows are short linear Chinese 1:1
cigar/path-info and `nw-score`.

This sweep ran on a Granite Rapids Xeon 6975P-C (4 vCPU, AWS host `avx10`)
compiled with GCC 16.1. The `x86_avx10_512` / `x86_avx10_256` backends are
**de-selected**: current GCC AVX10.1 target codegen compiles these striped
SW/NW kernels ~2x slower than the classic `avx512f/bw/vl` backend on this
hardware, so the dispatcher prefers `avx512bwvl` — which is also the fastest
measured backend here. See the AVX10 note in `CMakeLists.txt`.

`generic` is for correctness/baseline reference, not as a parasail competitor;
it loses every score-only row, though a handful of linear NW path/CIGAR rows
are competitive.

## Levenshtein (Intel x86) - 2026-07-17

Raw artifact: [`benchmarks/intel-levenshtein-2026-07-17.csv`](benchmarks/intel-levenshtein-2026-07-17.csv).

Build context: same host as Intel x86 above (Granite Rapids Xeon 6975P-C /
`avx10`, Python 3.14, `taskset -c 2`, GCC 16.1), running on the
`x86_avx512bwvl` backend. The multi-target Myers kernel runs one target per
SIMD lane (8x 64-bit lanes under AVX512) and reads bytes / 1-byte unicode
strings zero-copy from CPython buffers. Patterns over 64 chars fall through
to the scalar Hyyrö multi-word dispatch in `levenshtein_dispatch.hpp`.

Command:

```bash
taskset -c 2 .venv/bin/python tools/refresh_x86_benchmarks.py --bench kjv --date 2026-07-17
```

Corpus: first 1000 lines of `demo/kjv.txt`. Queries: 14 single words and
short phrases (3-29 chars) covering the pattern lengths that hit the
SIMD fast path.

### Overall

| Backend | Rows | Geomean | Median | Worst | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stride_align` vs `rapidfuzz`    | 14 | **3.749x** | 3.754x | 3.690x | 3.800x |
| `stride_align` vs `editdistance` | 14 | 42.379x | 41.501x | 39.149x | 47.014x |

Per-call wall time at 1000 targets (median across 14 queries):

| Library | µs/call | ns/target |
| --- | ---: | ---: |
| `stride_align` (`x86_avx512bwvl`) | **141** | **141** |
| `rapidfuzz`                       | 526  | 526  |
| `editdistance`                    | 5852 | 5852 |

### Takeaways

On a server Granite Rapids part the 8-lane AVX512 multi-target Myers kernel
pulls well ahead of rapidfuzz on this batch corpus — **3.75x** geomean (141
µs/call vs 526), with all 14 queries between 3.69x and 3.80x. The margin is
wider than the older laptop sweep because the 8-lane batch scales on a
server AVX-512 datapath: stride-align's per-call time roughly quarters while
rapidfuzz's scalar-per-target loop is unchanged. editdistance is ~42x
slower, reflecting its pure-C scalar DP loop with no batching.

(python-Levenshtein is omitted from this run — `benchmark_libs.py
--levenshtein` compares against rapidfuzz and editdistance.)

## Levenshtein extended (Intel x86) - 2026-07-17

Raw artifact: [`benchmarks/intel-levenshtein-v2-2026-07-17.csv`](benchmarks/intel-levenshtein-v2-2026-07-17.csv).

Three follow-up workloads that exercise the multi-word SIMD batch
kernel (patterns 65-256 chars, in 64-char blocks W = 2/3/4), the
zero-copy singular dispatch (no `prepare_alignment` vector copy when
both inputs are bytes or 1-byte unicode), and the `score_cutoff`
parameter with per-lane done masks and all-lanes early-exit. Same
build host and pinning as the section above (`avx10`, 2026-07-17).

### Long patterns (1-vs-200, no cutoff)

| `q_len` | stride_align | rapidfuzz | vs rf |
| ---: | ---: | ---: | ---: |
| 40  |  20 µs |  45 µs | **2.28x** |
| 65  |  40 µs | 150 µs | **3.78x** |
| 100 |  60 µs | 216 µs | **3.59x** |
| 128 |  76 µs | 265 µs | **3.46x** |
| 180 | 128 µs | 437 µs | **3.42x** |
| 200 | 155 µs | 556 µs | **3.59x** |

Each lane in the AVX-512 kernel runs Hyyrö's wide-add carry chain over
W blocks in parallel across 8 targets. The wide add uses two chained
64-bit adds + `gt_u64` overflow detection (AVX-512 native unsigned
`cmpgt`, AVX2 sign-bit-XOR + signed `cmpgt`, SSE4.1 sub-and-sign-bit
emulation). q_len = 40 is the single-word kernel; on this server part it
already runs 2.28x rapidfuzz, and the multi-word blocks (q ≥ 65) hold
3.4–3.8x as the 8-lane batch amortizes the per-target setup.

### 1-vs-1 singular (zero-copy dispatch)

| `q_len` | stride_align | rapidfuzz |
| ---: | ---: | ---: |
| 10  | **0.12 µs** | 0.14 µs |
| 30  | **0.18 µs** | 0.21 µs |
| 60  | **0.29 µs** | 0.32 µs |
| 100 | **0.56 µs** | 1.10 µs |
| 200 | **1.59 µs** | 2.75 µs |

When both inputs are bytes or 1-byte unicode the singular path skips
the prepare\_alignment vector copy and runs scalar Myers directly on
the CPython buffer (`PyBytes_AsStringAndSize` / `PyUnicode_1BYTE_DATA`).
On this part stride-align edges rapidfuzz at every length — a hair ahead
under 60 chars (0.12 vs 0.14 µs at q=10, Python ABI bound) and ~1.7–2.0x
from 100 chars onward, where the multi-word inner loop dominates.

### score_cutoff (5000 targets, short query)

stride_align vs rapidfuzz with matching cutoff:

| `q_len` | cutoff | stride_align | rapidfuzz | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 10 |  2 | **103 µs** |  412 µs | **4.00x** |
| 10 |  5 | **139 µs** |  597 µs | **4.31x** |
| 10 | 10 | **161 µs** |  597 µs | **3.70x** |
| 30 |  7 | **215 µs** |  947 µs | **4.40x** |
| 30 | 15 | **318 µs** |  948 µs | **2.98x** |
| 30 | 30 | **406 µs** |  944 µs | 2.33x |
| 50 | 12 | **329 µs** | 1300 µs | **3.96x** |
| 50 | 25 | **490 µs** | 1301 µs | **2.65x** |
| 50 | 50 | **650 µs** | 1302 µs | 2.00x |

Per-lane done masks freeze score updates once a lane crosses
`cutoff + remaining_chars`, and the column loop breaks as soon as every
batch lane is settled (target exhausted or bailed). The tightest cutoffs
win most (3–4x) — most targets exceed cutoff after a handful of columns so
the whole batch short-circuits — while looser cutoffs settle toward the
no-cutoff batch ratio.

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

## Damerau-Levenshtein / OSA (Intel x86) - 2026-07-17

Raw artifact: [`benchmarks/intel-damerau-levenshtein-2026-07-17.csv`](benchmarks/intel-damerau-levenshtein-2026-07-17.csv).

Build context: same host as the Levenshtein section (Granite Rapids Xeon
6975P-C / `avx10`, Python 3.14, `taskset -c 2`, GCC 16.1, 2026-07-17). The
algorithm is OSA-restricted (Optimal String Alignment) Damerau-Levenshtein:
like Levenshtein but adjacent transpositions cost 1 instead of two
substitutions, and each character can participate in at most one edit.
Hyyrö's bit-parallel recurrence (the `TR = (((~D0_prev) & PM) << 1) &
PM_old` formulation that rapidfuzz also uses), wrapped in the same
multi-target SIMD batch architecture as our Levenshtein kernel — one
target per SIMD lane (2/4/8 lanes for SSE4.1/AVX2/AVX-512). Query and
targets are equal-length random byte strings (single-word regime, q ≤ 64).

### Short targets (1-vs-1000)

This is the SIMD batch sweet spot: short alignments amortize the
gather + state-shift cost across 8 lanes, and rapidfuzz's per-pair
overhead dominates its loop.

| `q_len` | stride_align | rapidfuzz | ratio |
| ---: | ---: | ---: | ---: |
|  5 | **8.9 µs** |  79.8 µs | **9.00x** |
| 10 | **11.9 µs** |  96.6 µs | **8.10x** |
| 20 | **18.3 µs** | 131.0 µs | **7.16x** |
| 30 | **25.0 µs** | 168.3 µs | **6.73x** |

### Medium batch (1-vs-200)

| `q_len` | stride_align | rapidfuzz | ratio |
| ---: | ---: | ---: | ---: |
| 10 | **3.0 µs** | 19.4 µs | **6.44x** |
| 30 | **5.6 µs** | 33.8 µs | **6.02x** |
| 64 | **30.4 µs** | 85.4 µs | **2.81x** |

The win narrows toward q=64 as each lane's single-word OSA recurrence
lengthens, but the 200-target batch still amortizes setup across 8 lanes.
(Long, multi-word targets — q > 64 — are not exercised by this sweep.)

### 1-vs-1 singular

| `q_len` | stride_align | rapidfuzz | ratio |
| ---: | ---: | ---: | ---: |
| 10 | **0.105 µs** | 0.115 µs | **1.09x** |
| 30 | **0.173 µs** | 0.196 µs | **1.13x** |
| 60 | **0.287 µs** | 0.303 µs | **1.06x** |

Per-call Python ABI dominates; stride-align now edges rapidfuzz at every
length (the scalar OSA path runs directly on the CPython buffer).

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
`x86_avx2`, `x86_avx512bwvl`, `x86_avx10_256`, `x86_avx10_512`, and
(added 2026-05-19) `macos_arm64_neon` and `linux_aarch64_neon` via the
shared `NeonOps` bundle. Other architectures (SVE / Loongson / Power)
still fall through to the shared scalar bit-parallel dispatch and
remain correct.

<div class="benchmark-note">
  <strong>Dated architecture reference archive</strong>
  <p>The following May and June runs are the newest saved receipts for their hosts, but they predate the July kernel work. Their conclusions and follow-up lists describe the measured revision—not the current roadmap. They remain here for regression history and architecture-specific context.</p>
</div>

## Levenshtein + Damerau-Levenshtein (Power8 VSX) - 2026-05-19

Raw artifact: [`benchmarks/power8-lev-osa-2026-05-19.txt`](benchmarks/power8-lev-osa-2026-05-19.txt).

Build context: Power8 KVM-virtualized core (4.157 GHz), Ubuntu 20.04,
Python 3.13, AT15.0 GCC 11.4 at `/opt/at15.0/bin/g++` (the system GCC
9.4 is too old for `cxx_std_23`), CMake 4.3.2. `VsxOps` is the new
128-bit / 2-lane bundle (same shape as SSE / NEON / LSX) using
`__vector unsigned long long` and Altivec intrinsics. Power8's ISA
2.07 has native unsigned `vec_cmpgt` for 64-bit lanes so the kernel
ports without emulation.

No rapidfuzz / python-Levenshtein wheels exist for ppc64le on PyPI;
comparison is against our generic backend (which runs tight bit-parallel
Myers/OSA scalars).

### Levenshtein 1-vs-1000 short (3-15 char corpus)

| `q_len` | generic | VSX | ratio |
| ---: | ---: | ---: | ---: |
|  5 | 103 µs | **44 µs** | 2.37x |
| 10 | 109 µs | **44 µs** | 2.49x |
| 20 | 111 µs | **44 µs** | 2.54x |
| 30 | 125 µs | **44 µs** | **2.87x** |

### Levenshtein 1-vs-200 medium (30-250 char corpus)

| `q_len` | generic | VSX | ratio |
| ---: | ---: | ---: | ---: |
|  10 | 153 µs |  98 µs | 1.56x |
|  64 | 200 µs |  98 µs | 2.04x |
| 100 | 468 µs | 172 µs | 2.72x |
| 200 | 675 µs | 223 µs | **3.03x** |

Multi-word kernel takes over at q=100; the ratio grows because the
W-block SIMD scales with q while generic scalar's overhead scales
linearly with q too.

### Damerau-Levenshtein 1-vs-1000 short

| `q_len` | generic | VSX | ratio |
| ---: | ---: | ---: | ---: |
|  5 | 109 µs | **48 µs** | 2.25x |
| 10 | 108 µs | **48 µs** | 2.22x |
| 20 | 112 µs | **48 µs** | 2.30x |
| 30 | 124 µs | **48 µs** | **2.57x** |

### Damerau-Levenshtein 1-vs-200 medium

| `q_len` | generic | VSX | ratio |
| ---: | ---: | ---: | ---: |
| 10 | 159 µs | 109 µs | 1.46x |
| 32 | 182 µs | 109 µs | 1.67x |
| 64 | 204 µs | 109 µs | **1.87x** |

Unlike LSX (which trails generic on the 2-lane Damerau medium
workload), Power8 VSX wins consistently here. Power8's faster
`vec_extract` for the per-lane gather setup and the native unsigned
`vec_cmpgt` keep the 2-lane SIMD competitive even on short-pattern
inner loops.

## Levenshtein + Damerau-Levenshtein (Graviton4 NEON/SVE/SVE2) - 2026-05-19

Raw artifact: [`benchmarks/graviton4-lev-osa-2026-05-19.csv`](benchmarks/graviton4-lev-osa-2026-05-19.csv).

Build context: AWS Graviton4 (Neoverse V2, 1 vCPU c8g.medium), Ubuntu
24.04, Python 3.14, GCC 13.x. The Graviton4 host has only 1.8 GiB RAM
and 1 vCPU, so the build required `CMAKE_BUILD_PARALLEL_LEVEL=1` and a
4 GiB swapfile to keep cc1plus from OOM-killing on the template-heavy
TUs.

All three ARM backends (`linux_aarch64_neon`, `linux_aarch64_sve`,
`linux_aarch64_sve2`) share the same SIMD path: the SVE backends are
built with `-msve-vector-bits=128`, so they hold the same 2 lanes of
64-bit as NEON. Both wire through `NeonOps` rather than a separate
`SveOps` bundle — the bit-parallel Lev/OSA kernel uses no
SVE-specific feature.

### Levenshtein 1-vs-1000 short (3-15 char corpus)

| `q_len` | stride_align | python-Levenshtein | ratio |
| ---: | ---: | ---: | ---: |
|  5 |  53 µs | 140 µs | 2.67x |
| 10 |  53 µs | 148 µs | 2.80x |
| 20 |  53 µs | 176 µs | 3.33x |
| 30 |  53 µs | 213 µs | **4.05x** |

### Levenshtein 1-vs-200 medium (30-250 char corpus)

| `q_len` | stride_align | python-Levenshtein | ratio |
| ---: | ---: | ---: | ---: |
|  10 | 150 µs | **119 µs** | 0.80x |
|  32 | 150 µs | **121 µs** | 0.81x |
|  64 | 150 µs | **127 µs** | 0.85x |
| 100 | **163 µs** | 256 µs | 1.57x |
| 200 | **224 µs** | 434 µs | 1.94x |

Single-word multi-target (q ≤ 64): we trail python-Levenshtein on
medium-length targets because the per-target SIMD setup outpaces the
2-lane parallelism gain. Multi-word kicks in at q=100; we then pull
ahead 1.57-1.94x.

### Damerau-Levenshtein 1-vs-1000 short

| `q_len` | stride_align | rapidfuzz OSA | ratio |
| ---: | ---: | ---: | ---: |
|  5 |  57 µs | 129 µs | 2.27x |
| 10 |  57 µs | 142 µs | 2.50x |
| 20 |  57 µs | 179 µs | 3.15x |
| 30 |  57 µs | 221 µs | **3.89x** |

### NEON vs SVE vs SVE2

All three ARM backends produce identical results and identical
performance (53 µs for q=10 short, etc.). The auto-detect picks SVE2
on Graviton4 since it ranks first in the priority list, but routing
through `NeonOps` means swapping backends is observationally a no-op.

## Levenshtein + Damerau-Levenshtein (Loongson LASX/LSX) - 2026-05-19

Raw artifact: [`benchmarks/loongson-lev-osa-2026-05-19.txt`](benchmarks/loongson-lev-osa-2026-05-19.txt).

Build context: Loongson 3A6000 (LoongArch64), Kylin V10 SP1, Python
3.13, GCC 15.2.0 (`/opt/loongson-gcc-15.2.0`), CMake 4.3.2. No
rapidfuzz / python-Levenshtein wheels exist for LoongArch on PyPI, so
the comparison is against our generic scalar backend (which already
runs the bit-parallel Myers / OSA kernels in tight C++).

`LsxOps` is 128-bit / 2 lanes (similar to SSE & NEON);
`LasxOps` is 256-bit / 4 lanes (similar to AVX2).

### Caveat on `vandn`

Initial port had LSX/LASX returning negative scores on simple inputs.
Root cause: `__lsx_vandn_v(a, b)` returns `~a & b` (Intel-style),
contrary to what the LoongArch ISA reference's mnemonic name "VANDN"
suggested. The fix was a single-line operand swap; correctness on the
generic-reference test set is now 3200/3200 across q_lens 10/32/64/100.

### Levenshtein 1-vs-1000 short (3-15 char corpus)

| `q_len` | generic | LSX | LASX |
| ---: | ---: | ---: | ---: |
|  5 | 103 µs | 67 µs (1.54x) | **49 µs (2.10x)** |
| 10 | 108 µs | 67 µs (1.61x) | **49 µs (2.18x)** |
| 30 | 126 µs | 67 µs (1.88x) | **49 µs (2.56x)** |

### Levenshtein 1-vs-200 medium (30-250 char corpus)

| `q_len` | generic | LSX | LASX |
| ---: | ---: | ---: | ---: |
|  10 | 175 µs | 162 µs (1.08x) | **114 µs (1.54x)** |
|  64 | 185 µs | 162 µs (1.14x) | **114 µs (1.63x)** |
| 100 | 492 µs | 203 µs (2.43x) | **147 µs (3.34x)** |
| 200 | 777 µs | 351 µs (2.22x) | **260 µs (2.99x)** |

The multi-word kernel (q_len > 64, W=2/3) pulls ahead more
dramatically than the single-word range because the SIMD batch
amortizes the wide-add carry chain across 4 lanes (LASX) on the
LoongArch's 3 GHz cores.

### Damerau-Levenshtein 1-vs-1000 short

| `q_len` | generic | LSX | LASX |
| ---: | ---: | ---: | ---: |
|  5 |  97 µs | 91 µs (1.06x) | **62 µs (1.57x)** |
| 10 | 102 µs | 91 µs (1.12x) | **61 µs (1.66x)** |
| 30 | 121 µs | 91 µs (1.32x) | **61 µs (1.97x)** |

### Damerau-Levenshtein 1-vs-200 medium

| `q_len` | generic | LSX | LASX |
| ---: | ---: | ---: | ---: |
| 10 | 176 µs | 249 µs (0.71x) | **155 µs (1.14x)** |
| 32 | 182 µs | 249 µs (0.73x) | **155 µs (1.17x)** |
| 64 | 189 µs | 249 µs (0.76x) | **155 µs (1.21x)** |

LSX trails the generic backend on the OSA medium workload: 2 lanes of
SIMD overhead (extra gather / mask / state-shift cost) outpaces the
parallelism gain when the per-target scalar bit-parallel Myers loop is
already tight. LASX keeps 4-lane parallelism worthwhile. Backend
auto-detect picks LASX where available, so this only matters on
machines that lack LASX.

## Levenshtein + Damerau-Levenshtein (Mac M4 NEON) - 2026-05-19

Raw artifact: [`benchmarks/macos-arm64-neon-lev-osa-2026-05-19.csv`](benchmarks/macos-arm64-neon-lev-osa-2026-05-19.csv).

Build context: Apple M4 (T6041), macOS 15.x, Python 3.13 in the
project virtualenv. Uses the new `macos_arm64_neon` SIMD batch kernel
(2 lanes × 64-bit, NEON intrinsics in `levenshtein_simd_ops.hpp`). The
Mac is 2-lane (NEON 128-bit), so per-call SIMD speedup is smaller than
on AVX-512 (8 lanes); the win comes from skipping Python ABI per-pair
overhead on the batch path.

### Levenshtein, 1-vs-1000 short targets (3-15 char corpus)

| `q_len` | stride_align | python-Levenshtein | ratio |
| ---: | ---: | ---: | ---: |
|  5 | **17 µs** |  92 µs | 5.49x |
| 10 | **17 µs** |  97 µs | 5.80x |
| 20 | **17 µs** | 118 µs | 7.04x |
| 30 | **17 µs** | 143 µs | **8.54x** |

### Levenshtein, 1-vs-200 medium targets (30-250 char corpus)

| `q_len` | stride_align | python-Levenshtein | ratio |
| ---: | ---: | ---: | ---: |
|  10 |  80 µs |  89 µs | 1.12x |
|  32 |  80 µs |  94 µs | 1.18x |
|  64 |  80 µs |  95 µs | 1.19x |
| 100 |  94 µs | 152 µs | 1.61x |
| 200 | **112 µs** | 260 µs | **2.33x** |

The 100/200-char rows exercise the multi-word kernel (W=2/3); the
ratio grows because python-Levenshtein's overhead scales with pattern
length while our W-block SIMD scales with `q_len / (64 * lanes)`.

### Damerau-Levenshtein, 1-vs-1000 short

| `q_len` | stride_align | rapidfuzz OSA | ratio |
| ---: | ---: | ---: | ---: |
|  5 | **20 µs** |  87 µs | 4.35x |
| 10 | **20 µs** |  95 µs | 4.81x |
| 20 | **20 µs** | 120 µs | 6.06x |
| 30 | **20 µs** | 148 µs | **7.45x** |

### 1-vs-1 singular

Parity territory — Python ABI dominates, no algorithmic gap.

| `q_len` | Lev sa / Lev rf | OSA sa / OSA rf |
| ---: | ---: | ---: |
| 10 | 0.13 / 0.13 µs | 0.13 / 0.13 µs |
| 30 | 0.21 / 0.21 µs | **0.17** / 0.21 µs |
| 60 | **0.29** / 0.33 µs | 0.29 / 0.29 µs |

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

### 2026-05-18 Graviton4 follow-up notes

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

### 2026-05-18 Mac follow-up notes

1. Target affine `nw-score` next — it is the largest remaining gap and the only variant where every comparison loses.
2. Investigate `sw-farrar-score` width 16: still trailing parasail at `0.905x` median.
3. Keep the exact-fill linear SW Farrar score path enabled.
4. Do not reuse the shared masked-trace helpers for NEON linear SW path/CIGAR without redesigning the trace representation.
5. Add a native parasail comparison mode to the arm64 microbench before instruction-level parity work.
6. Keep SWAR off the mac performance path (geomean `0.64x` generic on the 2026-05-13 native sweep) — correctness/reference only.

## Loongson LoongArch64 (LSX/LASX) - 2026-07-17

Raw Loongson artifacts:

| Artifact | Contents |
| --- | --- |
| [`benchmarks/loongson-native-2026-07-17.csv`](benchmarks/loongson-native-2026-07-17.csv) | Full 320-row native sweep after 0.6.0 SW kernel rework + LASX/LSX micro-opts: `generic`, `swar`, LSX, LASX, all 7 variants, `1:1` and `1:many`, widths 16/32. |
| [`benchmarks/loongson-native-2026-05-18.csv`](benchmarks/loongson-native-2026-05-18.csv) | Prior full 320-row native sweep (pre-0.6.0). |
| [`benchmarks/loongson-score-1to1-parasail-2026-05-13.csv`](benchmarks/loongson-score-1to1-parasail-2026-05-13.csv) | `sw-score`/`nw-score` `1:1` comparison against patched generic LoongArch parasail. |
| [`benchmarks/loongson-2026-05-13.md`](benchmarks/loongson-2026-05-13.md) | Loongson-specific notes and recommendations. |

Build context: Loongson 3A6000-class host (Kylin V10 SP1, old-world
`/lib64/ld.so.1`), Python 3.12.13, **GCC 15.2.0** at
`/opt/loongson-gcc-15.2.0`, CMake 4.3.2. Regenerated 2026-07-17. A matching
**new-world** wheel was also built with GCC 16.1.0
(`/opt/loongson-gcc-16.1.0/wrappers`) but cannot run on this host (requires
`GLIBC_2.38` / the new-world loader). Unit tests: **1641 passed, 396 skipped,
0 failed** (old-world); the five bounded-lazy-F counter-examples all return
the true scores under LASX/LSX.

Parasail status unchanged: no LoongArch wheel; the 2026-05-13 patched 1:1
score comparison remains the parasail baseline below. This 2026-07-17 sweep
is native-only.

### Overall vs `generic` (2026-07-17 native)

| Backend | Rows | Wins | Geomean | Median | Worst | Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `linux_loongarch64_lasx` | 80 | 68 | **4.959x** | 5.192x | 0.495x | 33.108x |
| `linux_loongarch64_lsx`  | 80 | 71 | **3.630x** | 4.299x | 0.401x | 16.179x |

### Score-only vs `generic` (2026-07-17 native)

| Backend | Rows | Wins | Geomean | Median | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linux_loongarch64_lasx` | 48 | 48 | **11.905x** | 12.073x | 33.108x |
| `linux_loongarch64_lsx`  | 48 | 48 |  **7.269x** |  7.383x | 16.179x |

### Path / CIGAR vs `generic` (2026-07-17 native)

| Backend | Rows | Wins | Geomean | Median | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linux_loongarch64_lasx` | 32 | 20 | 1.333x | 1.079x | 7.294x |
| `linux_loongarch64_lsx`  | 32 | 23 | 1.281x | 1.178x | 6.087x |

### By variant (vs generic)

| Variant | LSX geomean | LASX geomean |
| --- | ---: | ---: |
| `sw-farrar-score` | 10.596x | 19.805x |
| `sw-score`        |  9.921x | 18.049x |
| `nw-score`        |  3.654x |  4.721x |
| `sw-cigar`        |  1.620x |  1.933x |
| `nw-cigar`        |  1.650x |  1.765x |
| `nw-path-info`    |  1.219x |  1.147x |
| `sw-path-info`    |  0.827x |  0.808x |

### Score-only vs patched LoongArch parasail (1:1)

The 2026-05-13 parasail comparison (16-row 1:1 score-only; still the best
parasail baseline available on this host):

| Backend | Rows | Wins | Geomean | Median | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linux_loongarch64_lasx` | 16 | 16 | 7.517x | 6.502x | 22.365x |
| `linux_loongarch64_lsx`  | 16 | 16 | 5.307x | 5.395x | 13.205x |
| `swar`                   | 16 |  8 | 1.061x | 1.161x |  2.201x |

### Takeaways

LASX remains the clear Loongson win: **11.9x** geomean over generic on
score-only (was 11.6x on 2026-05-18), peak **33.1x**, and 7.5x over patched
parasail on 1:1 score. LSX score-only jumped hard: **5.19x → 7.27x**
geomean, driven by adding the missing vector `reduce_max` (was a scalar
spill every score call) and the same sound deferred / active-count lazy-F
path as the other backends.

LASX-specific micro-opts landing this round (on top of the shared 0.6.0
kernel rework):

1. **`Ops::reduce_max`** for i8/i16/i32 — log-tree fold with `xvmax` +
   `xvbsrl` / `xvpermi_q` (was shared scalar stack spill).
2. **`Ops::local_lazy_f_prefix_carry`** for i8/i16/i32 — LASX previously
   fell through to the scalar store/loop prefix carry every column.
3. **Sound active-count** on the affine exact-fill H-only F scan
   (`ceil(max_f / |gap|)`), plus 4-way unrolled main DP bodies for the
   i16/64 and i32/128 exact-fill kernels.

`sw-path-info` is still the lone weak variant (LSX 0.83x / LASX 0.81x) —
`profile_traceback` materialization dominates the SIMD score lift. SWAR
remains a score-only regression vs generic (~0.68x) and is for correctness
only.

### Recommended Loongson next steps

1. Keep exact-fill LSX/LASX score hooks + the new `reduce_max` /
   `local_lazy_f_prefix_carry` members — large score-only win.
2. Target a Loongson-specific linear SW trace/CIGAR redesign before
   further instruction scheduling.
3. Add a native Loongson microbench entrypoint for tight A/B of the
   exact-fill kernels.
4. Treat parasail as a generic LoongArch comparison until a maintained
   LSX/LASX parasail build becomes available.

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

### 2026-05-18 Power8 follow-up notes

1. Try a `vbpermq`-based `trace_mask_*` on real hardware; combined with a row-major linear SW trace table this might bring the masked path above `1.0x` of generic.
2. Add a Power8 `local_affine_score_exact_segment*_raw` mirroring the NEON helpers if the 1024-character query shape becomes a target; current `sw-farrar-score` is already 7-8x ahead of generic.
3. Re-bench on a multi-core / non-virtualized Power8 host to characterise SMT throughput and shared L2/L3 effects.
4. Build parasail from source for ppc64le and add a parasail column to the next sweep — every other family in this file has at least one parasail point of reference.
5. Investigate why SWAR loses to generic on Power8 via an asm dump of the generic score loop.

## Jaro + Jaro-Winkler (cross-arch) - 2026-05-23

First cross-arch sweep of the new Jaro / Jaro-Winkler SIMD batch
kernels. One target per 64-bit SIMD lane; the query's per-byte PEQ is
gathered per-lane on each iteration, and the per-lane window mask is
built via the new `shl_var_u64` / `shr_var_u64` Ops primitives. After
the SIMD inner loop, a scalar finishing pass per lane computes
match/transposition counts from the bitmaps.

Same workload everywhere: random lowercase strings, one query of the
listed length, 1000 targets of the same length. Median of 3 runs of
50 iterations each. Baseline is `rapidfuzz.distance.Jaro.similarity`
called in a Python list comprehension — the natural "fuzzy match one
query against many targets" pattern.

Output is bit-equivalent to rapidfuzz across all listed backends:
verified on 500 random batches × ~25 targets each (~12,500 pairs per
backend); 0 mismatches at machine precision.

### Singular SIMD batch (one query, 1000 targets)

| Host / backend | Query len | stride-align | rapidfuzz | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Tiger Lake `x86_avx512bwvl` | 12 | 40.7 us | 181.5 us | **4.46x** |
| Tiger Lake `x86_avx512bwvl` | 32 | 105.1 us | 289.2 us | **2.75x** |
| Graviton4 `linux_aarch64_neon` | 12 | 43 us | 269 us | **6.26x** |
| Graviton4 `linux_aarch64_neon` | 32 | 100 us | 353 us | **3.53x** |
| Apple M-series `macos_arm64_neon` | 12 | 16 us | 151 us | **9.36x** |
| Apple M-series `macos_arm64_neon` | 32 | 48 us | 183 us | **3.86x** |
| Loongson `linux_loongarch64_lasx` | 12 | 87 us | 13,952 us | 161x |
| Loongson `linux_loongarch64_lasx` | 32 | 187 us | 49,299 us | 263x |
| Power8 `linux_powerpc64_vsx` | 12 | 194 us | 600 us | **3.09x** |
| Power8 `linux_powerpc64_vsx` | 32 | 467 us | 719 us | **1.54x** |

The Loongson ratios are dramatic because rapidfuzz has no LSX/LASX
SIMD path on LoongArch64 — it falls through to a scalar C kernel,
while our LSX/LASX bit-parallel batch fans out 2/4 targets per vector
iteration.

### One bug surfaced during deployment

VSX (`*reinterpret_cast<Vec*>(ptr)` for `load_aligned`/`store_aligned`)
ran into a strict-aliasing miscompile under GCC -O3: the scalar
writes to per-iteration `LaneScratch` could be reordered past the
same-block Vec read, silently dropping lane-1 match updates on every
2-target group. The Levenshtein SIMD kernel uses the same primitives
but a different scratch pattern, so it didn't trip. Fix: switch VSX
to `vec_xl` / `vec_xst`, the proper VSX load/store intrinsics.
Documented in commit `8ae4905`.

### Multi-word query batch (q_len in (64, 256], m_len ≤ 64)

Same workload shape, query length stretched into the multi-word path
(W = 2 for q in (64, 128], W = 3 for (128, 192], W = 4 for (192, 256]).
Targets stay short; b_matched fits in a single word.

| Host / backend | q_len | m_len | N | stride-align | rapidfuzz | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiger Lake `x86_avx512bwvl` | 50 | 20 | 500 | 45 us | 227 us | **5.01x** |
| Tiger Lake `x86_avx512bwvl` | 100 | 20 | 500 | 44 us | 255 us | **5.81x** |
| Tiger Lake `x86_avx512bwvl` | 150 | 20 | 500 | 52 us | 299 us | **5.79x** |
| Tiger Lake `x86_avx512bwvl` | 200 | 20 | 500 | 60 us | 328 us | **5.45x** |
| Graviton4 `linux_aarch64_neon` | 100 | 20 | 500 | 53 us | 238 us | **4.52x** |
| Apple M-series `macos_arm64_neon` | 100 | 20 | 500 | 25 us | 124 us | **4.96x** |
| Loongson `linux_loongarch64_lasx` | 100 | 20 | 500 | 110 us | 39,461 us | 358x |
| Power8 `linux_powerpc64_vsx` | 100 | 20 | 500 | 1,532 us | 581 us | 0.38x |

Power8 is the one regression. The 2-block (W=2) inner loop doubles
the gather count per j vs the single-word path, and Power8's VSX
gather is emulated as scalar `vec_extract`/`vec_insert` (no native
ppc gather instruction at this lane count). The per-iteration
overhead exceeds rapidfuzz's tight scalar loop at this size.
Workaround: if q_len ≤ 64 the single-word path stays 3x ahead;
above 64 on Power8 specifically, prefer the per-target scalar
dispatch (which the singular-API path already uses for q > 64).
Future work: native pre-shuffle of the gather indices, or a Power-
specific tuned gather using `vec_perm`.

### Multi-word target batch (q_len ≤ 256, m_len in (64, 256])

The second multi-word axis: target length crossing the 64-bit
register boundary, in addition to (or independent of) the query
length. b_matched becomes `std::array<Vec, W_target>`; the inner
loop only updates block `j / 64` so the per-iteration work is the
same as single-word target.

| Host / backend | q | m | N | stride-align | rapidfuzz | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiger Lake `x86_avx512bwvl` | 100 | 100 | 500 | 196 us | 789 us | **4.02x** |
| Tiger Lake `x86_avx512bwvl` | 200 | 150 | 500 | 440 us | 1305 us | **2.97x** |
| Tiger Lake `x86_avx512bwvl` | 60 | 100 | 500 | 171 us | 631 us | **3.70x** |
| Tiger Lake `x86_avx512bwvl` | 80 | 200 | 500 | 343 us | 1230 us | **3.58x** |
| Graviton4 `linux_aarch64_neon` | 100 | 100 | 500 | 246 us | 643 us | **2.62x** |
| Apple M-series `macos_arm64_neon` | 100 | 100 | 500 | 100 us | 296 us | **2.95x** |
| Loongson `linux_loongarch64_lasx` | 100 | 100 | 500 | 509 us | 142,489 us | 279x |

The dispatch picks `(W_query, W_target)` from the actual lengths in
the batch, so short-target inputs still get `W_target = 1` (no wasted
work). 16 instantiations max per backend.

### Constraints (v0.3.0)

* SIMD path covers query lengths up to 256 AND target lengths up to
  256 (W = 1..4 blocks per side). Above 256 on either side it falls
  through to per-target scalar dispatch (bit-parallel single-word
  for ≤ 64 inputs and the scalar reference above).
* Byte-compatible inputs (bytes / 1-byte unicode). Wider unicode
  falls through to scalar via the prepared-token path.

### Levenshtein audit (no changes needed)

Lev's SIMD batch already handles query lengths up to 256 via the
same W = 1..4 multi-word pattern. Target length on Lev's side is
just an iteration count over the inner DP loop — no per-target
register-width constraint — so multi-word target is a non-issue for
Lev. Above q_len = 256, Lev's scalar dispatch picks up via Hyyrö's
multi-word Myers (no upper bound on q_len). Future work to extend
the SIMD batch beyond W = 4 is small but the use case (queries >
256 chars in batches of 1000+) is rare.

## cdist pruning + cutoff push-down (Intel x86) - 2026-05-24

Three optimizations stacked on top of `cdist_above_threshold` and
`cdist_top_k`:

1. **Length-difference pruning.** Each pair `(q, t)` is gated by a
   closed-form upper bound on the achievable normalized similarity
   before any SIMD work runs. Bounds: `min/max` for Lev / OSA /
   true-DL, `(2 + min/max)/3` for Jaro, `2*min/(q+t)` for Indel,
   `1.0` if equal-length for Hamming.
2. **Row-sort by query length, descending.** `cdist_top_k`
   processes the longest queries first so close-length high-scoring
   pairs surface early and the shared `global_min_bound` atomic
   reaches a useful value before the short-query rows run.
3. **Per-pair cutoff push-down into the SIMD kernel.** The Myers /
   OSA / Hamming inner loops bail per lane when the score exceeds
   the per-pair cutoff plus the remaining-chars allowance; bailed
   lanes return the `cutoff + 1` sentinel. Lev/OSA use
   `floor((1-T)*max(|q|, |t|) + 1e-9)`; Hamming uses
   `floor((1-T)*|q|)`. Indel's bit-parallel Allison-Dix doesn't
   track a running distance per column, and Jaro/JW have multi-term
   scores without a clean per-column bail, so those two scorer
   families benefit from length pruning only.

All three are correctness-preserving — tests in
`tests/test_cdist_length_pruning.py` pin the result set against the
un-pruned full `cdist` matrix at multiple thresholds and the
floating-point integer-boundary edges.

### Setup

Tiger Lake `x86_avx512bwvl`, N=400 queries × M=400 targets =
160,000 pairs, random lowercase ASCII. Lengths 4–40 for Lev / OSA /
Indel / Jaro / JW; lengths 100 (equal-length) for Hamming.
`cpu_count=4`. Reproduce via `tools/bench_cdist_pruning.py --scorer
<name>`.

### `cdist_above_threshold` throughput (pairs/sec)

| Scorer | T=0 | T=0.3 | T=0.5 | T=0.7 | T=0.85 | T=0.95 | T=0.99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEVENSHTEIN_NORMALIZED        | 0.49M | 12.7M |  31.3M |  53.7M | 116M | 226M | **290M** |
| DAMERAU_LEVENSHTEIN_NORMALIZED| 0.50M | 13.0M |  35.5M |  70.2M | 147M | 190M | **297M** |
| HAMMING_NORMALIZED (n=100)    | 0.46M | 22.5M | 122M   | 108M   | 110M | 114M | **136M** |
| INDEL_NORMALIZED              | 0.47M | 11.3M |  24.7M |  39.0M |  66M | 143M | **286M** |
| JARO                          | 0.53M |  0.5M |   2.2M |  12.6M |  24M |  47M | **129M** |
| JARO_WINKLER                  | 0.51M |  0.5M |   1.8M |  11.9M |  14M |  41M | **141M** |

### Speedup ratio vs `T=0` (same scorer, same workload)

| Scorer | T=0.3 | T=0.5 | T=0.7 | T=0.85 | T=0.95 | T=0.99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LEVENSHTEIN_NORMALIZED        | **26x** |  **64x** | **109x** | **236x** | **458x** | **587x** |
| DAMERAU_LEVENSHTEIN_NORMALIZED| **26x** |  **72x** | **141x** | **298x** | **384x** | **598x** |
| HAMMING_NORMALIZED            | **49x** | **264x** | **233x** | **238x** | **246x** | **293x** |
| INDEL_NORMALIZED              | **24x** |  **53x** |  **83x** | **142x** | **306x** | **611x** |
| JARO                          |  0.9x   |  4.1x    |  **24x** |  **46x** |  **90x** | **245x** |
| JARO_WINKLER                  |  1.0x   |  3.5x    |  **23x** |  **27x** |  **80x** | **275x** |

Jaro / Jaro-Winkler show no benefit until the threshold rises above
the natural-distribution floor of the `(2 + min/max)/3` length
bound. For length 4–40 random strings that happens around T ≈ 0.7;
above that the bound rules out most pairs and the speedup compounds.

### `cdist_top_k` throughput (pairs/sec)

| Scorer | k=1 | k=10 | k=100 | k=1000 | k=10000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LEVENSHTEIN_NORMALIZED        |  8.0M | 25.3M | 21.1M | 17.0M |  8.5M |
| DAMERAU_LEVENSHTEIN_NORMALIZED| 20.0M | 23.4M | 19.4M | 18.6M | 12.2M |
| HAMMING_NORMALIZED (n=100)    | 78.0M | 68.9M | 71.1M | 61.6M | 24.0M |
| INDEL_NORMALIZED              | 29.1M | 24.5M | 21.5M | 17.4M |  7.5M |
| JARO                          | 15.3M | 14.2M | 13.9M | 12.6M |  8.9M |
| JARO_WINKLER                  | 15.1M | 12.2M | 13.6M | 13.4M | 10.0M |

The row-sort matters most at small `k`: with the longest queries
processed first, the global heap-min bound rises early and the
per-pair cutoff push-down has a tight value to compare against for
the bulk of the remaining rows. At very large `k` the heap rarely
fills with strong matches so the bound stays close to the
`(1.0 - safe margin)` floor and the kernel-level cutoff doesn't bite.

### Cross-arch throughput at `T=0.99` (pairs/sec)

Same script, same workload, four different SIMD backends.

| Scorer | Tiger Lake `x86_avx512bwvl` | Graviton4 `linux_aarch64_neon` | Mac M-series `macos_arm64_neon` | Loongson `linux_loongarch64_lasx` |
| --- | ---: | ---: | ---: | ---: |
| LEVENSHTEIN_NORMALIZED        | 290M | 318M |   996M | 370M |
| DAMERAU_LEVENSHTEIN_NORMALIZED| 297M | 318M | 1,014M | 426M |
| HAMMING_NORMALIZED (n=100)    | 136M |  70M |   295M | 139M |
| INDEL_NORMALIZED              | 286M | 272M |   995M | 402M |
| JARO                          | 129M | 160M |   784M | 190M |
| JARO_WINKLER                  | 141M | 105M |   543M | 136M |

### Cross-arch speedup vs `T=0` (same scorer, same host)

| Scorer | Tiger Lake | Graviton4 | Mac M-series | Loongson |
| --- | ---: | ---: | ---: | ---: |
| LEVENSHTEIN_NORMALIZED        | 587x | 699x |   537x | **1,199x** |
| DAMERAU_LEVENSHTEIN_NORMALIZED| 598x | 714x |   478x | **1,353x** |
| HAMMING_NORMALIZED            | 293x | 145x |   147x |   487x  |
| INDEL_NORMALIZED              | 611x | 567x |   540x | **1,408x** |
| JARO                          | 245x | 339x |   429x |   642x  |
| JARO_WINKLER                  | 275x | 225x |   293x |   472x  |

The speedup ratios are algorithmic — the bound math is independent
of ISA, so the cross-host spread reflects only how much the
un-pruned baseline costs vs the post-pruning hot path on each
machine. Mac's M-series tops the absolute throughput because the
inner SIMD loops are bit-parallel ops that the Apple core hands
back at high IPC; Loongson posts the largest *ratio* because its
un-pruned baseline (full Myers / OSA / Indel scan per pair at
length 4–40) is slowest in absolute terms.

Power8 numbers are deferred (host RAM too tight for a full `-O3`
rebuild — see `docs/power8-gcc10-workarounds.md`).

### Reading the numbers

The relative speedups carry over across hosts; the absolute
throughput numbers don't. The pre-pruning baseline (`T=0`) is
`cdist_above_threshold` running every pair through full SIMD —
equivalent to a full `cdist` plus the iterator overhead — so it's
the right "no optimization" reference for the speedup ratios.


## rapidfuzz shim full surface (dated reference) - 2026-06-17

Comprehensive comparison of every public entry point in the
`stride_align.rapidfuzz` shim (10 `fuzz` methods, 8 distance classes
× 4 methods, `process.cdist` + `process.extract` with several
scorers) against upstream rapidfuzz on the same deterministic
corpus. Generated by `tools/rapidfuzz_shim_full_bench.py`.

Raw receipts: [Mac M4 Max](benchmarks/shim-full-mac-2026-06-17.json),
[Intel AVX10](benchmarks/shim-full-avx10-2026-06-17.json), and
[Loongson LASX](benchmarks/shim-full-loongson-2026-06-10.json).

`ratio = upstream_ms / shim_ms` (BENCHMARK.md convention): values
> 1.0 mean shim is faster than upstream. Min-of-5 wall time per
workload. `rapidfuzz 3.14.5` was used on all three hosts. Mac M4 Max and
Intel were measured 2026-06-17; Loongson was measured 2026-06-10.

**This table is retained for surface-level diagnostics, not as a current
Intel headline.** Its Intel column used `_avx10_512`, which has since been
de-selected because it loses to the classic AVX-512 backend on this host.
No saved full-surface `_avx512bwvl` JSON exists, so this page deliberately
makes no replacement Intel full-surface claim. The current Intel alignment
evidence is the 2026-07-18 Parasail sweep at the top of the page.

### At a glance

| Host | Backend | Rows | Geomean | Median | Worst | Best | Wins/Ties/Losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mac M4 Max | `_macos_arm64_neon` | 108 | **1.34x** | 1.34x | 0.55x | 2.93x | 95/5/8 |
| avx10 (Intel AWS) | `_avx10_512` | 108 | **1.02x** | 1.13x | 0.15x | 1.77x | 68/13/27 |
| Loongson (2026-06-10) | `_lasx` | 108 | **49.17x** | 53.07x | 8.51x | 314.31x | 108/0/0 |

Win/tie/loss buckets use the raw shim/upstream runtime ratio: ≤ 1.05 = win,
1.05–1.15 = tie, > 1.15 = loss. The Loongson comparison is not comparable
to the Mac and Intel rows: upstream rapidfuzz ships no optimized LoongArch
wheel, while stride-align uses LASX.

### Per-workload comparison

`ratio = upstream / shim` per host (bold = shim loses, > 1.15x slower).

| Workload | avx10 (`_avx10_512`) | Mac (`_macos_arm64_neon`) | Loongson (`_lasx`) |
| --- | ---: | ---: | ---: |
| distance.DamerauLevenshtein.distance_K2 | 0.927x | 1.194x | 51.66x |
| distance.DamerauLevenshtein.distance_K3 | 0.932x | 1.185x | 51.08x |
| distance.DamerauLevenshtein.distance_K4 | 0.929x | 1.167x | 50.10x |
| distance.DamerauLevenshtein.distance_medium | 0.927x | 1.219x | 52.32x |
| distance.DamerauLevenshtein.distance_short | **0.826x** | 1.210x | 46.55x |
| distance.DamerauLevenshtein.normalized_distance_medium | 0.936x | 1.240x | 52.19x |
| distance.DamerauLevenshtein.normalized_distance_short | 0.875x | 1.343x | 45.04x |
| distance.DamerauLevenshtein.normalized_similarity_medium | 0.931x | 1.227x | 52.04x |
| distance.DamerauLevenshtein.normalized_similarity_short | **0.861x** | 1.302x | 45.11x |
| distance.DamerauLevenshtein.similarity_medium | 0.923x | 1.217x | 52.26x |
| distance.DamerauLevenshtein.similarity_short | **0.821x** | 1.205x | 45.78x |
| distance.Hamming.distance_equal_medium | **0.713x** | 0.913x | 30.46x |
| distance.Hamming.distance_equal_short | **0.856x** | **0.771x** | 14.71x |
| distance.Hamming.normalized_distance_equal_medium | 1.318x | 1.762x | 30.27x |
| distance.Hamming.normalized_distance_equal_short | 1.619x | 2.001x | 17.60x |
| distance.Hamming.normalized_similarity_equal_medium | 1.320x | 1.853x | 35.12x |
| distance.Hamming.normalized_similarity_equal_short | 1.729x | 2.014x | 23.12x |
| distance.Hamming.similarity_equal_medium | **0.737x** | **0.833x** | 32.69x |
| distance.Hamming.similarity_equal_short | **0.833x** | **0.775x** | 17.49x |
| distance.Indel.distance_K2 | 1.112x | 1.047x | 68.14x |
| distance.Indel.distance_K3 | 1.057x | 0.884x | 58.93x |
| distance.Indel.distance_K4 | 1.206x | 0.989x | 48.30x |
| distance.Indel.distance_medium | 1.236x | 1.100x | 65.94x |
| distance.Indel.distance_short | 1.136x | 1.232x | 31.08x |
| distance.Indel.normalized_distance_medium | 1.532x | 1.410x | 60.78x |
| distance.Indel.normalized_distance_short | 1.540x | 1.579x | 28.88x |
| distance.Indel.normalized_similarity_medium | 1.617x | 1.313x | 64.71x |
| distance.Indel.normalized_similarity_short | 1.774x | 1.922x | 33.05x |
| distance.Indel.similarity_medium | 1.254x | 1.095x | 66.12x |
| distance.Indel.similarity_short | 1.092x | 1.213x | 32.00x |
| distance.Jaro.distance_medium | 1.466x | 1.285x | 25.52x |
| distance.Jaro.distance_short | 1.264x | 1.597x | 54.64x |
| distance.Jaro.normalized_distance_medium | 1.456x | 1.469x | 25.57x |
| distance.Jaro.normalized_distance_short | 1.282x | 1.634x | 54.76x |
| distance.Jaro.normalized_similarity_medium | 1.456x | 1.419x | 25.46x |
| distance.Jaro.normalized_similarity_short | 1.266x | 1.629x | 52.66x |
| distance.Jaro.similarity_medium | 1.493x | 1.378x | 25.43x |
| distance.Jaro.similarity_short | 1.309x | 1.646x | 53.48x |
| distance.JaroWinkler.distance_medium | 1.485x | 1.373x | 25.85x |
| distance.JaroWinkler.distance_short | 1.270x | 1.638x | 59.88x |
| distance.JaroWinkler.normalized_distance_medium | 1.458x | 1.152x | 25.89x |
| distance.JaroWinkler.normalized_distance_short | 1.259x | 1.598x | 59.77x |
| distance.JaroWinkler.normalized_similarity_medium | 1.491x | 1.333x | 25.80x |
| distance.JaroWinkler.normalized_similarity_short | 1.295x | 1.522x | 59.07x |
| distance.JaroWinkler.similarity_medium | 1.440x | 1.225x | 25.76x |
| distance.JaroWinkler.similarity_short | 1.282x | 1.547x | 58.24x |
| distance.LCSseq.distance_K2 | 1.147x | 1.072x | 68.77x |
| distance.LCSseq.distance_K3 | 1.084x | 0.979x | 58.87x |
| distance.LCSseq.distance_K4 | 1.184x | 0.992x | 48.42x |
| distance.LCSseq.distance_medium | 1.224x | 1.157x | 65.69x |
| distance.LCSseq.distance_short | 1.130x | 1.236x | 30.49x |
| distance.LCSseq.normalized_distance_medium | 1.520x | 1.488x | 62.30x |
| distance.LCSseq.normalized_distance_short | 1.600x | 1.900x | 29.66x |
| distance.LCSseq.normalized_similarity_medium | 1.456x | 1.434x | 62.08x |
| distance.LCSseq.normalized_similarity_short | 1.448x | 1.741x | 31.21x |
| distance.LCSseq.similarity_medium | 1.225x | 1.103x | 64.81x |
| distance.LCSseq.similarity_short | 1.134x | 1.243x | 29.05x |
| distance.Levenshtein.distance_K2 | 1.526x | 1.712x | 132.0x |
| distance.Levenshtein.distance_K3 | 1.505x | 1.989x | 100.8x |
| distance.Levenshtein.distance_K4 | 1.515x | 2.097x | 96.32x |
| distance.Levenshtein.distance_medium | 1.075x | 1.303x | 141.6x |
| distance.Levenshtein.distance_short | 1.185x | 1.381x | 63.50x |
| distance.Levenshtein.normalized_distance_medium | 1.196x | 1.322x | 136.6x |
| distance.Levenshtein.normalized_distance_short | 1.488x | 1.910x | 59.70x |
| distance.Levenshtein.normalized_similarity_medium | 1.266x | 1.592x | 140.2x |
| distance.Levenshtein.normalized_similarity_short | 1.632x | 1.879x | 64.82x |
| distance.Levenshtein.similarity_medium | 1.068x | 1.198x | 142.7x |
| distance.Levenshtein.similarity_short | 1.197x | 1.355x | 65.77x |
| distance.OSA.distance_K2 | 1.269x | 1.815x | 145.9x |
| distance.OSA.distance_K3 | 1.221x | 1.901x | 107.6x |
| distance.OSA.distance_K4 | **0.746x** | 1.522x | 62.09x |
| distance.OSA.distance_medium | 0.931x | 1.265x | 157.6x |
| distance.OSA.distance_short | 0.994x | 1.316x | 70.25x |
| distance.OSA.normalized_distance_medium | 1.022x | 1.437x | 142.8x |
| distance.OSA.normalized_distance_short | 1.165x | 1.628x | 55.76x |
| distance.OSA.normalized_similarity_medium | 1.002x | 1.358x | 140.6x |
| distance.OSA.normalized_similarity_short | 1.108x | 1.535x | 55.68x |
| distance.OSA.similarity_medium | 0.905x | 1.190x | 152.3x |
| distance.OSA.similarity_short | 0.892x | 1.098x | 63.17x |
| fuzz.QRatio_medium | 1.546x | 1.401x | 68.64x |
| fuzz.QRatio_short | 1.678x | 1.828x | 38.72x |
| fuzz.WRatio_medium | **0.575x** | **0.865x** | 35.27x |
| fuzz.WRatio_short | **0.590x** | 1.100x | 10.94x |
| fuzz.partial_ratio_medium | **0.718x** | 1.016x | 76.85x |
| fuzz.partial_ratio_short | 0.922x | 1.397x | 36.79x |
| fuzz.partial_token_ratio_medium | **0.378x** | **0.549x** | 35.68x |
| fuzz.partial_token_ratio_short | **0.494x** | **0.803x** | 13.57x |
| fuzz.partial_token_set_ratio_medium | **0.728x** | 1.050x | 69.83x |
| fuzz.partial_token_set_ratio_short | **0.836x** | 1.324x | 24.40x |
| fuzz.partial_token_sort_ratio_medium | **0.724x** | 1.042x | 71.86x |
| fuzz.partial_token_sort_ratio_short | **0.869x** | 1.376x | 27.49x |
| fuzz.ratio_medium | 1.537x | 1.484x | 68.24x |
| fuzz.ratio_short | 1.719x | 1.993x | 39.44x |
| fuzz.token_ratio_medium | **0.737x** | 1.728x | 9.749x |
| fuzz.token_ratio_short | **0.756x** | 2.222x | 9.572x |
| fuzz.token_set_ratio_medium | 0.914x | 2.350x | 9.283x |
| fuzz.token_set_ratio_short | 0.974x | 2.930x | 8.514x |
| fuzz.token_sort_ratio_medium | 1.034x | 2.139x | 9.340x |
| fuzz.token_sort_ratio_short | 1.128x | 2.533x | 8.891x |
| process.cdist_50x100_Indel_distance | **0.154x** | 0.904x | 169.8x |
| process.cdist_50x100_Jaro_similarity | **0.449x** | **0.784x** | 278.4x |
| process.cdist_50x100_Levenshtein_distance | **0.317x** | 1.842x | 314.3x |
| process.cdist_50x100_OSA_distance | **0.323x** | 1.787x | 298.9x |
| process.cdist_50x100_fuzz_ratio | **0.169x** | 1.120x | 192.0x |
| process.extract_10k_Indel_distance | **0.649x** | 0.927x | 74.58x |
| process.extract_10k_Levenshtein_distance | 1.177x | 1.504x | 185.7x |
| process.extract_10k_fuzz_WRatio | **0.377x** | **0.586x** | 15.02x |
| process.extract_10k_fuzz_ratio | **0.642x** | 0.935x | 92.70x |

## Notes on comparing across families

These numbers are intended for engineering direction, not publication-grade
claims. Different families used different baselines (parasail where
available, otherwise `generic`), different sweep sizes, different host pinning
strategies, and different parasail builds (bundled wheel, locally compiled,
patched-for-LoongArch, or absent). For any cross-family claim, rerun the
relevant sweeps with matched conditions and use native microbench rows for
the specific kernels under discussion.
