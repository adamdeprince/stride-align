# SME Experiment — Pickup Notes

This file records where the experimental SME backend work was paused. The goal
of the experiment was to add a Smith-Waterman specialization for "one short
query against hundreds-to-thousands of short targets" on Apple Silicon with
SME (M4+), defaulting to NEON for everything else.

Nothing in the live codebase changed during this experiment — it stopped at
the toolchain-debugging stage. The C++ source tree is exactly as it was
before; no SME files were created, no build targets added.

---

## Status

**Paused.** Apple Clang 17 has a codegen bug that makes the SVE/SME intrinsic
path unusable on M4. An inline-asm workaround was proved to work, but using
it means writing the SME kernel in 150-300 lines of hand-rolled assembly,
which the user judged too costly for the experimental scope. Revisit if
Apple Clang ships a fix or if the workload data motivates the asm investment.

---

## Goal recap

User direction: build an experimental SME specialization that defaults to
NEON except for `1 short query x many short targets`. The "many" group must
be split into two buckets:

- **Fits in SME**: most targets, processed through the SME fast path.
- **Doesn't fit**: long-tail outliers, processed through the existing NEON
  per-target striped Farrar path.

Four candidate heuristics for length-variance handling (user-provided):

1. **Length-bucket the batch** — sort/group targets by length; one ZA tile
   per bucket; a small fraction of long-tail targets get their own (smaller,
   less-parallel) bucket. Standard approach for batched protein alignment
   libraries.
2. **Two-tier dispatch** — bulk of targets through the SME-resident fast
   path; >2 sigma length outliers through the existing striped per-target
   path. Small classification overhead, SME pass stays full-width.
3. **Truncate-and-retry** — run all targets to length L in the tile, mark
   any whose endpoint score was not reached as unfinished, re-process the
   unfinished ones in a second pass. Works if the long-tail fraction is
   small.
4. **Mask-on-completion** — keep targets in the tile but mask their
   contribution after they reach their endpoint. Wastes some ZA capacity
   but avoids re-batching overhead. Best when length variance is modest
   (within 2-3x).

User's explicit simplification: the initial split is **"fits in SME vs.
doesn't fit"** (heuristic #1 collapsed to two buckets). Heuristics #2-4 are
ways to refine the "fits" bucket later.

---

## Hardware target and tiers

The dispatcher must distinguish SME-only from SME2-capable hosts:

| Tier        | FEAT_SME | FEAT_SME2 | Chip example (per user)       |
| ----------- | -------- | --------- | ----------------------------- |
| NEON only   | 0        | 0         | any Apple silicon pre-M4      |
| SME path    | 1        | 0         | M3 (per user; needs verification) |
| SME2 path   | 1        | 1         | M4 Max (verified, see below)  |

The base SME path uses streaming-SVE registers (SVL-wide vectors). SME2
adds multi-vector loads/predicates (`LD1H {z0.h, z1.h}` etc.) that the
column inner loop can use; gate those behind the SME2 check at runtime.

Detection on macOS is via sysctl, e.g.:

```c
int v = 0; size_t n = sizeof v;
sysctlbyname("hw.optional.arm.FEAT_SME2", &v, &n, nullptr, 0);
```

Verified probe output on the M4 Max test host `wopr`:

```
hw.optional.arm.FEAT_SME:    1
hw.optional.arm.FEAT_SME2:   1
SME_I8I32:   1
SME_I16I32:  1
SME_F32F32:  1
FEAT_SME_I16I64: 1
SVL on M4 Max: 512 bits (svcntb in streaming = 64; cntd in streaming = 8)
```

Whether M3 actually reports `FEAT_SME=1` is unconfirmed; this experiment
trusts the user's statement that M3 is SME-only. The dispatcher logic is
chip-name-agnostic — it keys off the feature bits — so the M3 question is
not a blocker.

---

## Apple Clang 17 codegen bug (root cause of the SIGILL)

**Symptom:** any function marked `__arm_locally_streaming` that has SVE
register footprint SIGILLs on entry on M4 Max, even though `FEAT_SME=1`
and `__arm_has_sme()` returns 1.

**Cause:** Apple Clang's prologue emits a non-streaming SVE counting
instruction (`CNTD`) before `SMSTART sm`. It uses `CNTD` to compute the
SVE callee-save stack area size and a DWARF `vg` (vector granule)
unwind offset. On hosts that have both non-streaming SVE and streaming
SVE, `CNTD` works in either mode. Apple M4 has **streaming SVE only** —
plain `FEAT_SVE` is not present — so `CNTD` outside `smstart` is illegal
and the process dies before reaching the function body.

Sample faulty prologue (Apple Clang 17, `-O2 -march=armv9-a+sme`):

```
__Z9test_cntbv:
    rdsvl   x9, #1      ; OK: RDSVL works outside streaming on FEAT_SME
    lsr     x9, x9, #3
    str     x9, [sp, #-80]!
    cntd    x9          ; <-- SIGILL on M4 (non-streaming SVE)
    str     x9, [sp, #8]
    ... stack saves ...
    smstart sm          ; too late
    rdvl    x0, #1      ; only this is "in streaming"
    smstop  sm
```

The correct fix in the compiler would be to use `RDSVL`-derived math for
both the stack frame and the CFI offset, not `CNTD`. Whether and when
this lands in Apple's Clang is unknown.

**Reproducer commit kit** — drop the four `cpp` files below in `/tmp`,
compile with `clang++ -O2 -march=armv9-a+sme -std=c++23`, run.

- `sme_min.cpp`: `__arm_locally_streaming` function with **no** SVE
  intrinsics. Should print and exit 0.
- `sme_step.cpp`: progressively introduces `svcntb`, `svptrue_b16`,
  `svdup_n_s16`, `svld1_s16`. First call should SIGILL (exit 132).
- `sme_rdsvl.cpp`: inline asm `rdsvl x0, #1` outside streaming. Returns
  64 on M4 (= 512-bit SVL / 8), exits 0.
- `sme_cntd.cpp`: inline asm `cntd x0` outside streaming. **SIGILLs**
  on M4.
- `sme_asm.cpp`: inline asm block doing `smstart sm; rdsvl; cntd; smstop sm`.
  Returns `(64, 8)`, exits 0 — this is the working pattern.

The contents of all of these are short enough to recreate; just remember
`__arm_locally_streaming` (keyword, not `__attribute__`).

---

## Workaround that works

Wrap the entire kernel body in one inline-asm block. The compiler never
sees SVE intrinsics, so it never inserts the buggy prologue. Pattern:

```cpp
asm volatile(
    "smstart sm\n\t"
    // ... all SVE/SME instructions here ...
    "smstop  sm\n\t"
    : /* outputs */
    : /* inputs */
    : "memory", "x0", "x1", /* etc. */
);
```

This was verified on M4 with `rdsvl` + `cntd` inside the block: both run
correctly, the block enters/exits streaming cleanly, exit 0.

Implication: the actual SME kernel must be **hand-written asm** (either
inline asm in C++ or a separate `.s` file). C-level SVE/SME intrinsics
are off the table for the foreseeable future of Apple Clang 17.

A clean separation: dispatcher in C++ (sysctl detection, bucket split,
NEON fallback), and one or two `.s` files containing the streaming-SVE
batched-SW kernel(s), linked via `extern "C"` declarations.

---

## Pickup plan

Two pieces of work, independent:

### 1. Dispatcher (no SME knowledge required)

Build entirely in C++/NEON. Deliverable: a new entry point
`smith_waterman_farrar_scores_dispatched(query, targets, ...)` on the
mac NEON backend that:

a. Detects `FEAT_SME` / `FEAT_SME2` via `sysctlbyname` once at module
   init, caches the result.
b. Decides whether the workload qualifies as "1 short query x many short
   targets":
   - one query, multiple targets
   - target count above some threshold (start with N >= 16)
   - representative target length below some threshold (start with
     median target length <= 256 chars)
   Outside this regime, hand off to the existing NEON 1:many path.
c. Inside the regime, classify each target as fits-in-SME-budget
   (length <= `SME_TARGET_LIMIT`, start with 256) or outlier.
d. Route the "fits" bucket through the SME kernel hook if available,
   else through NEON one-by-one.
e. Route the "doesn't fit" bucket through the existing NEON striped
   path.
f. Stitch the two result lists back together in original input order.

With the SME kernel stubbed out as "loop over targets calling
`smith_waterman_farrar_score`", this is already a measurable artifact —
it lets us see whether the bucket split itself has overhead worth
caring about, and gives a clean drop-in point for the real kernel.

Naming convention proposed: the dispatcher lives on the macOS NEON
backend as a `_sme_dispatched` private hook so it's clearly separate
from the public NEON API; the public `smith_waterman_scores` does not
need to change.

### 2. SME kernel (only after #1 is in place)

Hand-roll a streaming-SVE batched Smith-Waterman in inline asm or a
`.s` file. Sketch of the design:

- **Inter-task SIMD**: lanes = different targets, all targets process
  the same query position in lockstep. At SVL=512, that's 32 lanes
  of i16 (32 targets per stream).
- **Storage**: H, E, F columns sized `2 * max_padded_target_length *
  N_targets * sizeof(int16_t)`. For N=32, padded length=256, that's
  32 KB — fits in L1.
- **Padding**: targets shorter than the bucket's `L` get padded; the
  padding character must not match any query character (use a token
  outside the alphabet) so their contributions decay to zero / the
  SW floor.
- **Outer loop**: query position i; inner loop: target position j.
- **Score extraction**: at end of each target's actual length, snapshot
  the running max for that lane; combine after the streaming pass.

For the SME2 path, the inner loop can use multi-vector loads
(`LD1H {z0.h, z1.h}, p0/z, [...]`) to fetch two target columns per
issue cycle. Gate this behind the FEAT_SME2 runtime flag.

If the SME2 path doesn't materially beat the SME base path on this
workload, drop the SME2 specialization. Keep the dispatch simple.

Skeleton signature for the C++ side:

```cpp
extern "C" void stride_align_sme_batch_sw_score_i16(
    const std::int16_t* query,        // length q_len
    std::int32_t q_len,
    const std::int16_t* targets_pad,  // [N x L] padded targets
    std::int32_t n_targets,
    std::int32_t pad_len,             // L (same for the whole bucket)
    const std::int32_t* target_lens,  // actual length of each target
    std::int16_t match_score,
    std::int16_t mismatch_score,
    std::int16_t gap_score,
    std::int16_t* out_scores          // length N
);
```

Implement in `src/cpp/backends/sme/macos_arm64_sme_batch.s` (or `.cpp`
if you go with inline asm). Build it with `-march=armv9-a+sme+sme2` so
SME2 instructions assemble; the runtime SME2 check still decides
whether the kernel actually issues SME2 ops.

---

## What was actually done during this session

- Confirmed M4 Max has FEAT_SME=1 and FEAT_SME2=1 with full integer
  outer-product support.
- Wrote and ran five probe binaries on wopr that pinned down exactly
  which instruction SIGILLs and why.
- Read Apple Clang 17's `arm_sme.h` and the assembly it produces for
  trivial streaming functions.
- Verified the inline-asm workaround works end-to-end.

**No source file in the stride-align repo was modified during this
experiment.** All probe files lived under `/tmp/` on wopr.

---

## Open questions for whoever picks this up

1. Is the M3 (or any pre-M4 Apple silicon) actually `FEAT_SME=1`? Not
   verified during this experiment; user's statement was assumed.
2. Does Apple Clang 18+ fix the `CNTD`-before-`SMSTART` bug? Worth a
   one-line probe before committing to the inline-asm path.
3. Is there any signal that Apple's Accelerate framework already exposes
   a streaming-SVE batched DP somewhere? Linking against it might be a
   shortcut around hand-rolling the kernel — but it would have to be
   inspected to see if it actually does SW DP.
4. What's the right `SME_TARGET_LIMIT`? Pick something defensible
   (e.g., padded SME L1 budget) and let the benchmark choose between
   the inner heuristics (#2-4 from above) on top of the basic
   fits/doesn't-fit split.
