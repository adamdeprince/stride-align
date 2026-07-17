# TODO — two known performance regressions in the striped SW kernels

Both landed with the 0.6.0 SW kernel rework (affine prefix-lazy-F speedup + the
bounded→deferred lazy-F **correctness** fix, see
`docs/known-issue-bounded-lazy-f-scan.md`). **Both are performance-only — every
score is correct.** Each is self-contained and scoped below for a future fix.
The measurement recipe is at the end.

Baseline for A/B: the pre-rework commit is **`5573936`** ("incumbent"); current
`main` is the "fixed" side. All numbers below are the native microbench
(`stride_align_x86_microbench` / `_arm_neon_microbench`), 1×8 batch (`--shape
1:many`), length 1024, `--match 8 --mismatch -9`, unless noted.

---

## 1. AVX2 affine local SW: ~8% regression at i32 / length-1024 only

**Symptom.** `--variant sw-affine-farrar-score --width 32` is ~8% *slower* than
the incumbent at **exactly length 1024 on AVX2**. Every other affine config is
+26…+46%. NEON shows 1.00 at that config (no dip) and AVX-512 shows +1%, so it
is **AVX2 + i32 + len-1024 specific** — not a general regression.

Measured on `naamah` (AVX2, sw-affine-farrar-score, width 32), stable across runs:
```
len  768: incumbent 4.18  fixed 5.80  (+39%)
len  960: incumbent 4.26  fixed 5.91  (+39%)
len 1024: incumbent 5.87  fixed 5.32  (-9%)   <-- the corner
len 1088: incumbent 4.31  fixed 5.97  (+38%)
len 2048: incumbent 4.17  fixed 6.07  (+46%)
```
Note the incumbent *spikes* to 5.87 only at len-1024 (its exact-fill fast path);
the fixed side dips to ~5.3 there. It is the incumbent spike vs the fixed dip.

**Root cause.** At width 32 (i32) / length 1024 the query occupies exactly
`segment_count * lane_count` (128 × 8 for AVX2 i32), so the **affine exact-fill**
path engages instead of the general path:
- dispatch: `src/cpp/backends/farrar_fixed_kernel.hpp:1858`
  `affine_score_state_for_offsets`, gate at `:1870` (`exact_fill_query`),
  the **segment128** branch at `:1886–1893` calling
  `Ops::local_affine_score_exact_segment128_raw`;
- the AVX2 kernel it calls: `src/cpp/backends/x86_avx2.hpp:1330`
  `local_affine_score_exact_segment128_raw`.

The 0.6.0 rework sped up the **general** affine path (the fallback after the gate,
`farrar_fixed_kernel.hpp:~1908` onward) by ~40%, but left the segment128
exact-fill kernel unchanged. So at len-1024/i32 the code still drops into an
exact-fill kernel that is now **slower than the improved general path** (fixed
exact-fill ≈ 5.3 vs fixed general ≈ 5.9). The incumbent's general path was slow
(~4.2), so for it the exact-fill (5.87) was a win; after the general path got
fast, the exact-fill is a net loss — but only for i32.

**Do not** disable segment128 wholesale: the *same* segment128 exact-fill is a
**win for i16** (i16/len-2048 → segment128, +42%). This is specific to i32.

**Fix direction.** In the `affine_score_state_for_offsets` gate
(`farrar_fixed_kernel.hpp:1872–1893`), make the **segment128** branch fall
through to the general path when `sizeof(Cell) == 4` (i32) on AVX2 — i.e. an
AVX2-i32-specific bypass — since the general path is faster there. Alternatively,
optimize `x86_avx2.hpp:1330 local_affine_score_exact_segment128_raw` to beat the
general path. Either way it **must be AVX2/i32-specific**: NEON, AVX-512, and i16
exact-fill are fine or winning; do not regress them.

**Verify.** A/B on `naamah`: fixed w32/len-1024 should recover ~5.3 → ~5.9 Gc/s,
with **no change** at other widths/lengths (still +38–46%), and scores must still
equal the `materialized`/scalar reference.

---

## 2. AVX-512 linear local SW: ~22% cost of the correctness fix

**Symptom.** The bounded→deferred correctness fix costs ~3% on AVX2, ~2% on NEON,
~0% on LASX — but **~22% on AVX-512** (`--variant sw-farrar-score`, len 1024,
w16), measured on `avx10`:
```
incumbent  bounded      (UNSOUND, fast)   16.0 Gc/s   <- what we can no longer use
fixed      deferred     (correct)         12.4 Gc/s
           materialized (correct, naive)  12.5 Gc/s
```
On AVX2, `deferred` (13.4) is much faster than `materialized` (10.8) and buys
back nearly all of `bounded`'s speed. On **AVX-512** `deferred ≈ materialized ≈
12.5` — the deferred trick gives nothing there, so the correct answer costs the
full ~22%. **This is the one place correctness is genuinely expensive.**

**Root cause.** The only *fast* correction on AVX-512 was the unsound bounded
early-exit (`bounded_local_sw_lazy_f_scan = true` for the AVX-512 Ops at
`src/cpp/backends/x86_avx512bwvl.hpp:310, 469, 628`), which stopped the lazy-F
scan as soon as `v_f <= h + gap`, assuming `h` monotonic across striped segments
— false, hence the too-low scores that motivated the fix. The two sound
replacements both land ~12.4 on AVX-512:
- deferred next-column fold: `score_state_exact_fill_local_sw`
  (`farrar_fixed_kernel.hpp:3485`, gate `use_deferred_correction` at `:3524`,
  branch at `:3675`) — doesn't beat materialized on AVX-512;
- sound bounded scan `scan_local_linear_lazy_f_once`
  (`farrar_fixed_kernel.hpp:1734`), which stops after
  `ceil(max_lane(v_f) / |gap|)` segments (the sound active-count at `:1751–1763`)
  — provably correct but conservative (it uses the *global* max `v_f`).

**Fix direction.** Find a **sound** early-exit for the AVX-512 exact-fill that
recovers toward 16 Gc/s. Ideas:
- Tighten the active-count at `:1751–1763` from a global `max_lane(v_f)` bound to
  **per-lane** active counts (each lane's F can only propagate
  `ceil(v_f[lane]/|gap|)` segments), so lanes that can't propagate stop early
  without dragging the whole vector to the global bound.
- Exploit local-SW `h ≥ 0`: the carried F beats `h` for at most a bounded run;
  a per-segment sound test that is cheaper than the deferred fold on AVX-512.
- Or an AVX-512-specific deferred/fold variant that actually beats
  `materialized` there (the current one doesn't).

**Must stay exact.** Regression-guard with
`docs/bounded-lazy-f-counterexamples.txt` +
`tests/test_bounded_lazy_f_regression.py`, then re-bench on `avx10`: fixed
len-1024 should move from ~12.4 toward ~16 Gc/s while still scoring 1947 on the
default corpus and correct on every counter-example.

---

## 3. parasail striped SW — same bug upstream, and the correctness fix regresses *its* performance

The identical unsound lazy-F early-exit exists in upstream parasail's striped SW
(every `src/sw_striped_*.c`; AVX2-16 is `sw_striped_avx2_256_16.c:331`,
`if (! _mm256_movemask_epi8(_mm256_cmpgt_epi16(vF, vH))) goto end;`). It was
located, patched and **validated** in a fork (`adamdeprince/parasail`, branch
`fix-striped-lazy-f-early-exit`): the patched `sw_striped_avx2_256_16` now
returns the correct optimum on the five counter-examples (was
3310/4897/2601/2986/4463 → now 3320/4907/2628/3010/4480, matching `sw_scan`).
Bundle in `~`: `parasail_bug_note.md`, `parasail_bug_repro.py`,
`parasail_bug_examples.fasta`.

**Why it isn't PR-ready yet.** The validated patch simply *removes* the
early-exit, so the lazy-F loop always runs the full `k < segWidth` pass instead
of bailing after 1–2 iterations. On the common case (F does not travel far)
that's a real slowdown, and upstream will not take a perf regression to fix a
rare mis-score. **This is the blocker to fix.**

**Fix direction — sound AND fast (do this instead of just deleting the exit).**
Port stride-align's **deferred** correction: fold the F contribution into the
next column rather than guessing when to stop. It is exact and costs only ~3% on
AVX2 vs the unsound exit (stride-align reference:
`score_state_exact_fill_local_sw`, `src/cpp/backends/farrar_fixed_kernel.hpp:3485`;
deferred branch at `:3675`). Alternative: a sound early-exit bounded by
`ceil(max_lane(v_f)/|gap|)` segments (`scan_local_linear_lazy_f_once`,
`:1751–1763`). **Caveat:** on AVX-512 the deferred fold does *not* recover the
speed — that's item 2's open problem — so a fully-general parasail fix inherits
item 2; but for the AVX2 striped variants deferred is enough to avoid a
regression. Measure the regression by benching `parasail_sw_striped_avx2_256_16`
on random length-1024 inputs (the common case): unpatched vs patched vs deferred.

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

Hosts used for the numbers above: `naamah` (AVX2), `avx10` (AVX-512, backend
`avx512bwvl`), local Apple M4 (`neon`), `loongson` (LASX; no native microbench —
Python-level `smith_waterman_farrar_scores` batch timing). For AVX-512 use
`--backend avx512bwvl`. Strategy A/B on the linear path: `--sw-farrar-i32-strategy
bounded|deferred|materialized|auto` (incumbent binary has a real `bounded`; the
fixed binary deprecates it to the unbounded scan).

## Operational details — the exact scripts I ran on naamah (AVX2)

**Host roles.** `naamah` = AVX2 (AMD, no AVX-512). `avx10` = AVX-512 (`--backend
avx512bwvl`). local Apple M4 = `neon`. `loongson` = LASX (loongarch64, **no
native microbench**). Incumbent baseline = commit `5573936`; "fixed" = `main`.

### Build both microbenches (incumbent 5573936 vs fixed working tree)
```bash
# a uv venv with cmake+nanobind (created earlier); any cmake+nanobind works
CM=/tmp/lazyf_hunt/venv/bin/cmake; PY=/tmp/lazyf_hunt/venv/bin/python
ND="$("$PY" -m nanobind --cmake_dir)"
build(){ "$CM" -S "$1" -B "$1/build/perf" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSTRIDE_ALIGN_BUILD_MICROBENCH=ON -DPython_EXECUTABLE="$PY" -Dnanobind_DIR="$ND"
  "$CM" --build "$1/build/perf" --target stride_align_x86_microbench -j; }

# incumbent worktree (hooksPath=/dev/null skips the LFS post-checkout hook)
git -C <repo> -c core.hooksPath=/dev/null worktree add -f /tmp/sa-inc 5573936
build /tmp/sa-inc
# fixed = a copy of the incumbent tree with the fixed src OVERLAID
rsync -a --exclude=build --exclude=.git /tmp/sa-inc/ /tmp/sa-fixed/
# (rsync'd from the mac first: <repo>/src/cpp -> /tmp/fixsrc/cpp, <repo>/include -> /tmp/fixsrc/include)
rsync -a /tmp/fixsrc/cpp/ /tmp/sa-fixed/src/cpp/ ; rsync -a /tmp/fixsrc/include/ /tmp/sa-fixed/include/
build /tmp/sa-fixed
# INC=/tmp/sa-inc/build/perf/stride_align_x86_microbench
# FIX=/tmp/sa-fixed/build/perf/stride_align_x86_microbench
```

### The A/B sweep + strategy-spread (`bench_ab.sh`, run verbatim on naamah)
```bash
#!/usr/bin/env bash
set -u
INC=/tmp/lazyf_hunt/incumbent/build/perf/stride_align_x86_microbench
FIX=/tmp/sa-fixed/build/perf/stride_align_x86_microbench
CPU="${CPU:-3}"
COMMON="--backend avx2 --shape 1:many --warmups 20 --samples 5 --iterations 300"
cells(){ taskset -c "$CPU" "$1" $2 2>/dev/null | grep -oE 'cells_per_s=[0-9.e+]+' | head -1 | cut -d= -f2; }
score(){ taskset -c "$CPU" "$1" $2 2>/dev/null | grep -oE 'score=[0-9-]+' | head -1 | cut -d= -f2; }
ratio(){ awk -v a="$1" -v b="$2" 'BEGIN{ if(a+0>0) printf "%.3f", b/a; else printf "n/a" }'; }
gc(){ awk -v x="$1" 'BEGIN{ printf "%.2f", x/1e9 }'; }
# A: fixed-vs-incumbent throughput (auto strategy)
for variant in sw-farrar-score sw-affine-farrar-score nw-affine-score; do
  for w in 16 32; do for len in 512 1024 2048; do for pass in english chinese; do
    base="$COMMON --variant $variant --width $w --length $len --pass $pass"
    ci=$(cells "$INC" "$base"); cf=$(cells "$FIX" "$base"); : "${ci:=0}" "${cf:=0}"
    printf "%-22s %2s %5s %-8s %9s %9s %6s\n" "$variant" "$w" "$len" "$pass" "$(gc $ci)" "$(gc $cf)" "$(ratio $ci $cf)"
  done; done; done; echo "----"
done
# B: linear-SW correctness-cost strategy spread (incumbent has the real bounded)
base="$COMMON --variant sw-farrar-score --width 16 --length 1024 --pass english"
for strat in bounded deferred materialized auto; do
  printf "  incumbent %-13s %9s Gc/s (score=%s)\n" "$strat" \
    "$(gc "$(cells "$INC" "$base --sw-farrar-i32-strategy $strat")")" \
    "$(score "$INC" "$base --sw-farrar-i32-strategy $strat")"
done
printf "  fixed     %-13s %9s Gc/s (score=%s)\n" auto \
  "$(gc "$(cells "$FIX" "$base")")" "$(score "$FIX" "$base")"
```

### Item 1's isolating sweep (the affine dip is ONLY at len=1024)
```bash
for len in 768 896 960 1024 1088 1152 1280 1536 2048; do
  b="--backend avx2 --variant sw-affine-farrar-score --shape 1:many --width 32 --length $len --pass english --warmups 20 --samples 5 --iterations 300"
  echo "len=$len inc=$(taskset -c 3 $INC $b|grep -oE 'cells_per_s=[0-9.e+]+'|cut -d= -f2) fix=$(taskset -c 3 $FIX $b|grep -oE 'cells_per_s=[0-9.e+]+'|cut -d= -f2)"
done
```

### parasail: build the fork + validate the striped fix (naamah, AVX2)
```bash
rsync -az --exclude=.git --exclude=build ~/dev/parasail/ naamah:/tmp/parasail-fork/
cmake -S /tmp/parasail-fork -B /tmp/parasail-fork/build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5   # old cmake_minimum_required needs this
cmake --build /tmp/parasail-fork/build -j                     # NOTE: tests/test_verify_traces.c fails
                                                              # to compile on strict gcc -- unrelated;
                                                              # libparasail.so still builds fine
LIB=$(find /tmp/parasail-fork/build -name 'libparasail.so*' | head -1)
INC=$(dirname "$(find /tmp/parasail-fork -maxdepth 2 -name parasail.h | head -1)")
gcc -O2 -mavx2 -I"$INC" /tmp/parasail-fork/validate_striped_fix.c "$LIB" -o /tmp/validate
LD_LIBRARY_PATH=$(dirname "$LIB") /tmp/validate   # -> "VALIDATED: patched striped correct on all 5"
```
`validate_striped_fix.c` embeds the five pairs and calls the patched
`parasail_sw_striped_avx2_256_16` vs `parasail_sw_scan_avx2_256_16`; regenerate
it from `~/parasail_bug_examples.fasta`. Reuse the same harness (swap the striped
call for a deferred variant) to bench the item-3 perf regression on random inputs.

### loongson LASX (no native microbench → Python batch timing)
Fixed side = `pip install` the released 0.6.0 old-world wheel; incumbent = build
`5573936` with old-world GCC 15.2 (`CC=/opt/loongson-gcc-15.2.0/bin/gcc`,
`CXX=.../g++`, `PATH` includes `/opt/loongson-cmake-4.3.2/bin`, backend deps
offline from `/tmp/sa-bw`, venv `--system-site-packages` for numpy). Then time
`smith_waterman_farrar_scores(q, [t]*8)` at len 1024 on each; the fixed/incumbent
ratio is the LASX cost (~0%). Toolchain setup is in `tools/_build_loongson.sh`.
