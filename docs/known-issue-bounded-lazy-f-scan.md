# Known issue: the "bounded" Smith-Waterman shortcut could report scores that are too low

**Status:** Fixed on 2026-07-16. This document is a post-mortem, written so that
a developer who has *never* worked on sequence alignment can follow what went
wrong, see real inputs that broke it, and check the conclusion with tools that
share no code with this project.

**Component:** the striped Smith-Waterman kernel
(`src/cpp/backends/farrar_fixed_kernel.hpp` and the AVX2 specializations in
`src/cpp/backends/x86_avx2.hpp`).

**Impact while it was live:** for certain inputs the local Smith-Waterman score
came back **too low**. No crash, no error — just a silently wrong number. It was
rare enough that a normal random test sweep never hit it (see §6).

---

## 1. What this code actually computes

Smith-Waterman finds the **single best-matching stretch** between two strings and
gives it a score. Think of sliding two strings past each other and rewarding the
best-aligned window:

- each matching character adds to the score (here **+8**),
- each mismatched character subtracts (here **−9**),
- and if you have to *skip* characters in one string to keep the alignment going
  — a "gap" — each skipped character costs a little (here **−1**).

The reported score is the best total you can reach over every possible window and
every possible set of skips. This is the workhorse behind fuzzy string matching,
spell-checking, plagiarism/diff tooling, and DNA/protein comparison. The only
thing that matters for this bug is that **the number is supposed to be exact.**

## 2. The shortcut that was wrong

Computing that score quickly on modern CPUs uses a well-known trick (Farrar's
"striped" method) that processes 16 or 32 positions at once with SIMD
instructions. One part of the calculation — handling gaps that run *vertically*
down the string — can't be finished in that first parallel pass, so the
algorithm makes a **correction sweep** afterward. This is the "lazy-F" step.

Picture the correction sweep as walking down a column of cells, carrying a
running "best gap credit" value. That carried value **shrinks by the gap penalty
(−1) at every step**, because a longer gap costs more. At each cell the sweep
asks: *does my carried credit make this cell better than it already is?*

The incumbent code had a performance shortcut called the **bounded** sweep. It
stopped walking early — the moment the carried credit was no longer big enough to
improve the current cell:

```
stop as soon as   carried_credit <= h[cell] + gap
```

The bet behind stopping: *"if my credit can't help this cell, it can't help any
cell further down either."*

## 3. Why that bet is false

That bet is only safe if the cells keep getting **bigger** (or stay level) as you
walk down. They don't. The cell values wiggle up and down. A cell further down
can be *much smaller* than the one you just stopped at — and your carried credit,
even after shrinking a few more steps, can still be large enough to lift it.

A tiny illustration. Say the cells down the column are:

```
cell 0: h = 10
cell 1: h =  2
cell 2: h =  8
```

and the sweep enters carrying a credit of 9 that loses 1 per step.

- At **cell 0** the shortcut checks its credit (9) against a big cell (10),
  decides "I can't improve a 10," and **stops.**
- But **cell 1** is only 2. Had the sweep continued, its credit (now ~8) would
  have lifted that cell from 2 up to ~8 — a real, correct improvement that the
  final score depends on.

The shortcut never looked at cell 1, so the score came out too low. That is the
entire bug: **cell heights are not monotonic, but the early-exit assumed they
were.**

## 4. Real inputs that break it

These are not hand-waved — they were found by a search that deliberately built
the shape §3 needs, then confirmed against independent references (§5). Each
input is a pair of length-1024 strings over a tiny alphabet, scored with
match +8 / mismatch −9 / gap −1.

| example | bounded (buggy) | correct answer | error |
|:---:|:---:|:---:|:---:|
| 1 | 3310 | **3320** | −10 |
| 2 | 4897 | **4907** | −10 |
| 3 | 2601 | **2628** | −27 |
| 4 | 2986 | **3010** | −24 |
| 5 | 4463 | **4480** | −17 |

**What these inputs have in common** is exactly the §3 trigger: a long run of one
repeated symbol that appears in *both* strings but at **shifted positions**, so
the best alignment has to cross a gap to line the two runs up — which is what
builds a large "carried credit" for the sweep to mishandle:

| example | run in string A | run in string B | position shift |
|:---:|:---|:---|:---:|
| 1 | 61×`5` at index 548 | 60×`5` at index 300 | 248 |
| 2 | 306×`4` at index 529 | 306×`4` at index 584 | 55 |
| 3 | 168×`1` at index 73 | 169×`1` at index 643 | 570 |
| 4 | 220×`5` at index 147 | 136×`6` at index 338 | 191 |
| 5 | 114×`4` at index 315 | 116×`4` at index 426 | 111 |

The **complete sequences** for all five examples are provided verbatim, in two
forms so you can feed them to any tool:

- [`bounded-lazy-f-counterexamples.txt`](bounded-lazy-f-counterexamples.txt) —
  integer tokens (the canonical form the search emitted); and
- [`bounded-lazy-f-counterexamples.fasta`](bounded-lazy-f-counterexamples.fasta) —
  the same inputs as `A`–`G` letter strings, loadable by any alignment tool
  (Biopython, EMBOSS, parasail, an online aligner, …). Each record's header line
  also carries that pair's `bounded=` and `correct=` scores.

The letter mapping is just `token 1→A … 7→G`; only character *equality* affects
the score, so the two forms are the same alignment problem.

## 5. How we know the shortcut is wrong — not our reference

A fair objection: *maybe the "correct answer" column is itself computed by our
code, so maybe we are the ones who are wrong.* So the same five inputs were run
through **two implementations that share no code with stride-align**:

1. **A from-scratch textbook Smith-Waterman** — the literal dynamic-programming
   recurrence you'd find in a textbook, ~30 lines of plain Python with no
   cleverness and no SIMD. It cannot have a "striped sweep" bug because it has no
   striping.
2. **Biopython's `PairwiseAligner`** — a widely used, independently written
   bioinformatics library that does *not* use Farrar's striped method at all.

Both agree with each other, both agree with the "correct answer" column, and both
reject the bounded value — on all five inputs:

```
 #   len  bounded  materialized  textbook-DP  biopython  verdict
 1  1024     3310          3320         3320       3320  bounded WRONG (refs=materialized)
 2  1024     4897          4907         4907       4907  bounded WRONG (refs=materialized)
 3  1024     2601          2628         2628       2628  bounded WRONG (refs=materialized)
 4  1024     2986          3010         3010       3010  bounded WRONG (refs=materialized)
 5  1024     4463          4480         4480       4480  bounded WRONG (refs=materialized)
```

That is four independent computations of the same score — stride-align's own
full (non-shortcut) sweep (`materialized`), a scalar oracle inside the search
tool (`scalar`), a textbook Python DP, and Biopython — all landing on the higher
value, with only the bounded shortcut coming up short. The shortcut is the
outlier.

**Reproduce the correct answer** (turnkey, no C++ build needed):

```bash
pip install biopython                    # enables the third-party cross-check
python tools/verify_bounded_lazy_f.py    # re-scores all five; prints the table above
```

**Or score them with stride-align itself** and watch the fixed kernel return the
true value (the pre-fix kernel returned the `bounded` column):

```python
import stride_align as sa
# query, target = example 2 from bounded-lazy-f-counterexamples.fasta (A..G strings)
sa.smith_waterman_farrar_score(query, target,
                               match_score=8, mismatch_score=-9, gap_score=-1)
# -> 4907   (the buggy bounded scan reported 4897)
```

The `bounded=` values recorded in the data files are the actual outputs of the
pre-fix AVX2 kernel; regenerating them requires building the kernel at the
pre-fix commit and forcing its bounded strategy.

## 6. Why ordinary testing never caught it

Before the crafted search, the bounded sweep was hammered with **~165,000 random
inputs** — including deliberately gap-friendly scoring — and produced **zero**
disagreements. The crafted search found **five in its first nine inputs.**

The difference is structure. Random strings almost never contain a long repeated
run, so they almost never build a "carried credit" large enough to expose the
early-exit. The bug lives in a corner of the input space that random sampling
effectively never visits. It only fell out once we generated inputs *designed* to
have the §3 shape (long runs placed at offset positions). The lesson worth
keeping: for a shortcut justified by an assumption about the data, fuzzing that
doesn't target the assumption's failure mode can pass indefinitely while the bug
is real.

## 7. The fix

- The bounded early-exit is **removed** from every path. All callers now use the
  full sweep, which always walks far enough to be correct.
- `scan_local_linear_lazy_f_once_bounded` remains only as a `[[deprecated]]`
  alias of the correct unbounded sweep, so any stray reference still computes the
  right answer.
- Speed is recovered a *different, sound* way — a **deferred** correction that
  folds the gap contribution into the next column instead of guessing when to
  stop. On AVX2 the measured cost is ~3% at the length where the exact-fill path
  engages (deferred 13.4 vs the buggy bounded 13.8 Gcells/s, and neutral-to-
  faster at other lengths), so correctness is essentially free there — the naive
  fully-correct scan (`materialized`, 10.8 Gcells/s) would instead have cost
  ~22%, which is exactly what `deferred` avoids. On AVX-512 there is no equally-
  sound trick that recaptures the last ~20%, so that path now trades a bit of
  speed for a correct answer — the right trade for a library whose scores are
  supposed to be exact.

Keeping both the deferred and full sweeps is intentional: two implementations,
both correct, chosen by profile size.

## 8. Guarding against regressions

The five inputs are wired into the suite as
`tests/test_bounded_lazy_f_regression.py`. It scores each pair through
`smith_waterman_farrar_score` (and `smith_waterman_score`) and asserts the true
value from the table above — so the bounded shortcut cannot quietly return. The
test reads the counter-examples from `bounded-lazy-f-counterexamples.txt` and
**fails loudly** if that fixture is ever moved, rather than silently skipping the
regression.

## References

- Kernel and correction sweep: `src/cpp/backends/farrar_fixed_kernel.hpp`
  (`scan_local_linear_lazy_f_once`), AVX2 exact-fill in
  `src/cpp/backends/x86_avx2.hpp`.
- Counter-examples (verbatim inputs): `docs/bounded-lazy-f-counterexamples.txt`.
- Independent cross-check tool: `tools/verify_bounded_lazy_f.py`.
