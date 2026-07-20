# TODO — striped SW kernel performance (status)

Two performance-only regressions from the 0.6.0 SW kernel rework (affine
prefix-lazy-F speedup + the bounded→deferred lazy-F **correctness** fix, see
`docs/known-issue-bounded-lazy-f-scan.md`). **Scores remain correct.** Item 1
and item 2 are addressed in-tree below; item 3 (upstream parasail) is still
open and must be done in a parasail fork without reading third-party sources
from this workspace.

Baseline for A/B: pre-rework commit **`5573936`** ("incumbent"); current
`main` is the "fixed" side. Measurement recipe at the end.

---

## 1. AVX2 affine local SW: i32 / length-1024 exact-fill dip — **FIXED**

**Was.** `--variant sw-affine-farrar-score --width 32` ~8% slower than the
incumbent at **exactly length 1024 on AVX2** because the general affine path
got ~40% faster while the segment128 exact-fill kernel did not.

**Fix.** Dropped the AVX2 i32 `local_affine_score_exact_segment128_raw` hook
so `affine_score_state_for_offsets` falls through to the improved general
path at that corner. The specialized kernel
`local_affine_score_exact_fill_i32_128` is retained in `x86_avx2.hpp` for
retuning but is no longer dispatched. NEON / LASX / AVX-512 exact-fill hooks
are unchanged (they still win or are neutral).

Also: `scan_lazy_f_h_only` (affine prefix-lazy-F H-only scan) now uses the
same sound `ceil(max_f / |gap|)` active-count bound as the linear scan.

**Verify.** A/B on `naamah`: fixed w32/len-1024 should track the general-path
~5.9 Gc/s band (not the old exact-fill ~5.3), with **no change** at other
widths/lengths, scores still matching `materialized`/scalar reference.

---

## 2. AVX-512 linear local SW: cost of the correctness fix — **IMPROVED**

**Was.** Bounded→deferred correctness cost ~3% on AVX2, ~2% on NEON, ~0% on
LASX, but **~22% on AVX-512** (`--variant sw-farrar-score`, len 1024, w16):
deferred ≈ materialized ≈ 12.4 Gc/s vs unsound bounded 16.0 Gc/s.

**Fixes applied (measured on `avx10`, 2026-07-17).**

1. **Score-only final fold of deferred F** → single `max(best, pending_f)`
   when `gap < 0` (F wavefront monotone decreasing).

2. **AVX-512 specialized exact-fill** `local_sw_score_exact_fill_i16_32`
   (`x86_avx512bwvl.hpp`):
   - 8-way unrolled main body
   - deferred pending injects **only into the diagonal** (no best-max on
     that critical path)
   - F score contribution folded once per column via `max(best, v_f)`
   - branchless `pending=0` when inactive
   - SIMD horizontal `reduce_max` (no 32-lane scalar spill)

3. **Result (len 1024, w16, english, 1:many):**
   | Strategy | Gc/s | notes |
   | --- | ---: | --- |
   | unsound bounded (incumbent) | ~16.0 | too-low scores on counter-examples |
   | post-correctness fix (auto) | ~12.4 | was deferred ≈ materialized |
   | **2026-07-17 auto/deferred** | **~13.83** | score 1947, all counter-examples OK |
   | residual cost vs unsound | **~14%** | down from ~22% |

4. **2026-07-19 on `avx10` (branchless pending + profile prefetch):**
   | Strategy | Gc/s | score |
   | --- | ---: | ---: |
   | **auto / deferred** | **~13.90** | 1947 |
   | materialized | ~11.05 | 1947 |
   | bounded (sound scan) | ~11.06 | 1947 |
   | residual vs historical unsound ~16.0 | **~13%** | — |

   Changes: remove scalar `has_pending_f` branch (always inject
   `max(v_f,0)`); prefetch next column profile. Score stable at 1947.

5. **2026-07-19 later on `avx10` (soft-pipeline loads + split best):**
   | Strategy | Gc/s | score |
   | --- | ---: | ---: |
   | **auto / deferred** | **~13.99** (best) | 1947 |
   | residual vs historical unsound ~16.0 | **~12.5%** | — |

   Changes in `local_sw_score_exact_fill_i16_32` deferred path:
   - 2-wide software pipeline: load next segment’s profile/E/prev-H
     under current segment arithmetic
   - 4 independent `best` accumulators (off H critical path)
   - dual-column prefetch of next profile + H/E arrays
   - rejected: 8-wide bulk tile of 24 ZMMs (register spill, ~0.5% slower)

6. **2026-07-19 dual-target deferred exact-fill (1:many):**
   | Path | Gc/s | score / checksum |
   | --- | ---: | --- |
   | sequential 1:many (dual OFF) | **~13.98** | 1947 / identical |
   | dual-target v1 (interleaved simple) | ~14.95 | 1947 / identical |
   | **dual-target v2 equal + 1-sided soft-pipe + full unroll** | **~15.09** | 1947 / identical |
   | 1:1 single-target (unchanged) | ~14.03 | 1985 |
   | residual 1:many vs historical unsound ~16.0 | **~5.7%** | down from ~12.5% |

   `local_sw_score_exact_fill_i16_32_dual` / `_dual_equal` + batch hook
   `try_score_batch_exact_fill_i16_32_dual` in `x86_avx512bwvl.hpp`:
   - lockstep 2 targets with independent H/E/pending/best
   - **equal-length fast path** (no unequal-finish branches; microbench
     1:many is always equal length)
   - **one-sided soft-pipeline** on target 0 (profile/E/prev-H for
     segment+1 under arithmetic); target 1 keeps simple deferred steps
     to stay under ZMM spill threshold
   - fully unrolled segment pairs 0..30 + tails (same style as 1:1 hot path)
   - dual-column prefetch of next profile (+1 KiB) and H/E for both targets
   - batch: prefer equal-length pairing; `thread_local` secondary H/E
     (no 6 KiB alloc per batch call)
   - unequal lengths: dual for `min(n0,n1)`, finish longer alone
   - odd count leftover → optimized single-target kernel
   - wired via `Ops::try_score_batch_exact_fill_dual` in
     `score_batch_state` (`farrar_fixed_kernel.hpp`)
   - **Correctness:** checksum bit-identical dual ON vs sequential OFF
     (`17173579280882299504`, 200×8 english/w16/len-1024)
   - **Fuzz (2026-07-20):** `stride_align_x86_microbench --verify-dual`
     — **39/39 PASS** (equal/odd/unequal/empty/strategy fallthrough/
     non-exact-fill query lengths/seed sweep/alt scores). Also
     `tests/test_dual_sw_batch.py` (batch vs sequential singles).

**Tried and rejected:** suffix-min early-exit (reverse `g[s]=min_t(H[t]+(t-s)|gap|)`
then stop when `F ≤ g[s]`) — sound, but reverse pass cost exceeded early-exit
savings on this shape (~10 Gc/s). Bulk 24-ZMM tile loads (spill). Soft-pipeline
on *both* dual targets (register pressure) not taken; one-sided is enough.

**Still open:** 1:1 residual (~12.5% to historical unsound ~16 Gc/s) is
unchanged; dual only helps 1:many.

### Profile of dual equal path (avx10, 2026-07-20)

Host: Xeon 6975P-C (Granite Rapids, 4 vCPU). Workload: dual 1:many,
english, w16, len 1024, many=8, auto/deferred.

| Signal | Value | Reading |
| --- | ---: | --- |
| IPC | **~2.15** | healthy vector throughput |
| `cycle_activity.stalls_total` | **~2.1%** of cycles | not stall-dominated |
| `exe_activity.bound_on_loads` | **~1.4%** | **not load-stall bound** |
| `exe_activity.bound_on_stores` | **~0%** | store buffer fine |
| `stalls_l1d_miss` / mem L1d stalls | **≪1%** | L1 misses rarely stall |
| L1 load miss rate | **~10%** of retired loads | almost all refill from **L2** (L2 miss ~0) |
| `cycle_activity.cycles_mem_any` | **~93%** | mem ops in flight constantly, but overlapped |
| `uops_retired.stalls` | **~50%** of cycles | **dependence / retire-limited** |
| Port 0 / 5_11 | heavy | vector ALU |
| Port 1 | ~idle | port imbalance (arch-specific) |
| Samples in dual kernel | **~99%** | `local_sw_score_exact_fill_i16_32_dual` (equal path inlined) |

**Conclusion:** 1:many dual is **not** waiting on H store/load *miss latency*.
H/E/profile traffic is L2-resident and well overlapped. The limiter is the
**serial H recurrence** (per-target dependence chain) + vector ALU work —
exactly what dual helped by adding a second independent chain.

**Implication for further dual micro-opts:**
- Soft-pipeline both targets / more prefetch: **low expected value** (mem stalls already ~0).
- 4-target lockstep: only plausible dual-side lever (more independent chains),
  but high ZMM spill risk and diminishing returns toward the unsound ~16 Gc/s
  ceiling (which also deleted H-rewrite work, not just latency).
- Closing residual on **1:1** needs a different approach (not dual).

### Cross-backend transfer (NEON local / LASX loongson) — 2026-07-20

What AVX-512 taught: residual was dependence-chain limited; dual-target 1:many
helped; soft-pipeline/mem tricks were secondary.

| Backend | i16@1024 shape | Linear exact-fill | Deferred vs mat. | 1:1 Gc/s | 1:many Gc/s | Dual? |
| --- | --- | --- | ---: | ---: | ---: | --- |
| AVX-512 | 32 segs × 32 lanes | specialized + dual | was −22%, now dual ~15.1 | ~14.0 | **~15.1** | yes |
| **NEON (M4)** | **128 segs × 8 lanes** | generic only (affine has raw) | deferred **~+22%** vs mat. | **~3.91** | **~3.85** (flat) | no |
| **LASX (loongson)** | 64 segs × 16 lanes | generic + prefix-carry (affine has raw) | hist. residual ~0% | ~4.4* | ~5.2* | no |

\*LASX via Python on installed 0.6.x tree (distinct mutated targets); not native microbench.

**What transfers:**

1. **Dual-target 1:many (highest value)** — portable idea. NEON especially:
   longest H chain (128 segs) and 1:many ≈ 1:1 today, so headroom looks like
   pre-dual AVX-512. LASX next (64 segs). Same batch hook pattern
   (`try_score_batch_exact_fill_dual`); body is ISA-specific.
2. **Branchless pending in generic deferred** — small, shared
   (`has_pending_f` still in `score_state_exact_fill_local_sw`). Cheap on both.
3. **Specialized linear exact-fill raw** (unroll / soft-pipe) — **lower priority**.
   NEON/LASX never had the 22% deferred cliff; deferred already wins. Port only
   if dual leaves residual and profile says so.
4. **Soft-pipeline / dual-column prefetch** — **skip first** unless a NEON/LASX
   profile shows load stalls (AVX-512 did not).

**What not to expect:** AVX-512’s specialized single-target climb (13.8→14.0)
was fighting a unique residual. NEON/LASX deferred is already the right default.

### Dual-target port results (2026-07-20)

Shared helper: `score_exact_fill_dual_local_sw` +
`try_score_batch_exact_fill_dual_generic` in `farrar_fixed_kernel.hpp`.
Hooks: NEON i16×128 segs, LASX i16×64 segs, AVX-512 keeps its specialized dual.

| Host | Path | 1:many (×8) Gc/s | vs sequential dual-OFF | Correctness |
| --- | --- | ---: | --- | --- |
| **NEON M4** | dual ON | **~8.3** (best ~8.39) | **~2.0×** (~4.17 OFF) | checksum identical |
| **NEON M4** | dual OFF | ~4.17 | — | score 1947 |
| **NEON M4** | 1:1 | ~4.2 | — | score 1985 |
| **LASX loongson** | dual ON | ~3.24 (Python) | **~1.00×** (neutral) | batch==seq |
| **LASX loongson** | dual OFF | ~3.23 (Python) | — | batch==seq |

**NEON:** dual is a large win (long 128-seg H chain + M4 OoO).  
**LASX:** dual is correct but throughput-neutral in Python 1:many (leave enabled;
  no regression). Further LASX dual micro-opts low priority without a native
  microbench/profile.

**Correctness hygiene (M4 NEON, 2026-07-20):**
- `pytest tests/test_dual_sw_batch.py` → **10/10 passed**
- `stride_align_arm_neon_microbench --verify-dual` → **39/39 passed**
  (shared harness `tools/verify_dual_sw_exact_fill.hpp`; arm64 microbench
  routes `--verify-dual` to NEON Ops)

**NEON dual profile + 4-target (M4, 2026-07-20):**
- `sample(1)`: ~100% of samples in `score_exact_fill_dual_local_sw` segment
  body (`local_sw_score_main_segment_corrected`); memset negligible.
- Dual ON vs OFF many=8: **~8.13 / ~4.21 Gc/s ≈ 1.93×** (chain-limited).
- Target-length scaling (Q=1024, T=256…2048): cells/s stays ~8.2–8.3
  (not a mem-bandwidth collapse).
- **4-target lockstep** (`EnableQuad=true` on NEON): many=8 best
  **~9.49 Gc/s** (~14–17% over dual-only ~8.3). verify-dual still 39/39.
  Kept enabled for NEON only (LASX stays dual-only).

---

## 3. parasail striped SW — same bug upstream — **OPEN (out of tree)**

The identical unsound lazy-F early-exit exists in upstream parasail's striped
SW. A validated correctness-only patch (delete the early-exit) lives in
`adamdeprince/parasail` branch `fix-striped-lazy-f-early-exit` and fixes the
five counter-examples, but always running the full lazy-F loop is a real
slowdown on the common case — the blocker for an upstream PR.

**Fix direction (do this in the parasail fork, not by reading parasail
sources from this workspace):** port stride-align's **deferred** correction
(fold F into the next column). Reference implementation after the item-2
work:

- `score_state_exact_fill_local_sw` in
  `src/cpp/backends/farrar_fixed_kernel.hpp`
- AVX2: `local_sw_score_exact_fill_i16_64` / `_i32_128` in `x86_avx2.hpp`
- AVX-512: `local_sw_score_exact_fill_i16_32` in `x86_avx512bwvl.hpp`

Alternative: sound early-exit bounded by `ceil(max_lane(v_f)/|gap|)`
(`scan_local_linear_lazy_f_once`). On AVX-512 the deferred fold needs the
specialized kernel (item 2); a fully-general parasail fix inherits that.

Bundle for the fork work (outside this repo): `parasail_bug_note.md`,
`parasail_bug_repro.py`, `parasail_bug_examples.fasta`.

---

## How to measure (native microbench)

```bash
nanobind_dir="$(.venv/bin/python -m nanobind --cmake_dir)"
cmake -S . -B build/perf -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSTRIDE_ALIGN_BUILD_MICROBENCH=ON -DPython_EXECUTABLE=.venv/bin/python \
  -Dnanobind_DIR="$nanobind_dir"
cmake --build build/perf --target stride_align_x86_microbench   # or _arm_neon_microbench

# A/B one config (fixed tree vs a 5573936 worktree/build):
build/perf/stride_align_x86_microbench --backend avx2 \
  --variant sw-affine-farrar-score --shape 1:many --width 32 --length 1024 \
  --pass english --warmups 20 --samples 5 --iterations 300
# fields: score=... cells_per_s=...
```

Hosts: `naamah` (AVX2), `avx10` (AVX-512, `--backend avx512bwvl`), local
Apple M4 (`neon`), `loongson` (LASX; Python-level batch timing). Strategy A/B
on the linear path: `--sw-farrar-i32-strategy
bounded|deferred|materialized|auto`.

## Operational details — A/B scripts on naamah (AVX2)

**Host roles.** `naamah` = AVX2. `avx10` = AVX-512. local Apple M4 = `neon`.
`loongson` = LASX. Incumbent = `5573936`; "fixed" = current tree.

### Build both microbenches
```bash
CM=/tmp/lazyf_hunt/venv/bin/cmake; PY=/tmp/lazyf_hunt/venv/bin/python
ND="$("$PY" -m nanobind --cmake_dir)"
build(){ "$CM" -S "$1" -B "$1/build/perf" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSTRIDE_ALIGN_BUILD_MICROBENCH=ON -DPython_EXECUTABLE="$PY" -Dnanobind_DIR="$ND"
  "$CM" --build "$1/build/perf" --target stride_align_x86_microbench -j; }

git -C <repo> -c core.hooksPath=/dev/null worktree add -f /tmp/sa-inc 5573936
build /tmp/sa-inc
rsync -a --exclude=build --exclude=.git /tmp/sa-inc/ /tmp/sa-fixed/
rsync -a /tmp/fixsrc/cpp/ /tmp/sa-fixed/src/cpp/ ; rsync -a /tmp/fixsrc/include/ /tmp/sa-fixed/include/
build /tmp/sa-fixed
```

### Item 1 isolating sweep (affine dip was ONLY at len=1024)
```bash
for len in 768 896 960 1024 1088 1152 1280 1536 2048; do
  b="--backend avx2 --variant sw-affine-farrar-score --shape 1:many --width 32 --length $len --pass english --warmups 20 --samples 5 --iterations 300"
  echo "len=$len inc=$(taskset -c 3 $INC $b|grep -oE 'cells_per_s=[0-9.e+]+'|cut -d= -f2) fix=$(taskset -c 3 $FIX $b|grep -oE 'cells_per_s=[0-9.e+]+'|cut -d= -f2)"
done
```

### loongson LASX
Fixed side = pip install 0.6.0+ wheel; compare
`smith_waterman_farrar_scores(q, [t]*8)` at len 1024. Toolchain:
`tools/_build_loongson.sh`.
