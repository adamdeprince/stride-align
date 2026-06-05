# stride-align

**Languages:** [English](README.md) · [简体中文](README.zh-CN.md)

`stride-align` is a SIMD-accelerated Python library for fuzzy string
matching, sequence alignment, phonetic encoding, and time-series
distance. It ships Smith-Waterman and Needleman-Wunsch alignment,
Levenshtein and Damerau-Levenshtein edit distance, Jaro and
Jaro-Winkler similarity, Hamming and Indel distance, Dynamic Time
Warping (`int16` / `float32` / `float64`), and the standard phonetic
encoders — Soundex, Metaphone (Philips and jellyfish variants),
Double Metaphone (Apache Commons and Python-package variants),
NYSIIS, Match Rating Approach, and Caverphone 2. Scoring kernels are
hand-vectorised behind a runtime CPU dispatcher: x86 SSE4.1 / AVX2 /
AVX-512BW+VL / AVX10-256 / AVX10-512, ARM NEON and SVE/SVE2,
LoongArch LSX and LASX, PowerPC VSX, with a scalar fallback. Python
bindings use nanobind with vectorcall on every entry point, target
Python 3.12+, and accept `bytes`, `str` (UCS-1/UCS-2/UCS-4
zero-copy), and NumPy `ndarray`.

Built for high-throughput fuzzy-match workloads — record linkage,
deduplication, search-as-you-type, NLP token similarity,
bioinformatics local alignment — with first-class CJK: UCS-2 inputs
route to a 16-bit-token Farrar kernel rather than being downconverted
to bytes, so Chinese, Japanese and Korean strings hit the same SIMD
path as ASCII. The all-pairs surface (`cdist`,
`cdist_above_threshold`, `cdist_top_k`, `cdist_top_k_per_query`)
combines per-target SIMD scoring with closed-form length-difference
pruning that skips work when a target's max possible similarity
provably can't clear the running heap minimum or threshold.
Substitution matrices (BLOSUM, PAM) and affine gaps are supported on
the alignment path. Correctness is validated on real x86, Apple
Silicon, Graviton ARM, Loongson LoongArch, and POWER8 hardware —
benchmarks at
[stride-align.com/BENCHMARK.html](https://stride-align.com/BENCHMARK.html).

Instead of giving you a lecture, we're going to learn by doing.
Let's dive right into how it works.

## Installation

```bash
pip install stride-align
```

Prebuilt wheels cover Linux x86_64 (glibc and musl), macOS arm64,
Linux aarch64, and Linux ppc64le on CPython 3.12 / 3.13 / 3.14.
Other (platform, Python) pairs fall back to the PyPI source
distribution and compile locally; you'll need a C++20 compiler and
CMake ≥ 3.26.

**Loongson / LoongArch64 users:** wheels live on GitHub Releases
rather than PyPI, and you pick between the old-world and new-world
binary worlds — see
[LoongArch installation](#loongarch-installation) further down.

First, just a disclaimer: I'm not using religious texts here to push
an agenda - for this demo I need multiple largish public domain
documents that have the same meaning but are phrased differently. The
Bible just happens to fit that demo requirement freakishly well.

Imagine we have two sentences - let's use the first sentence in
Genesis for this:

In the American Standard Version we have: "In the beginning God
created the heavens and the earth."

In the King James Version we have: "In the beginning God created the
heaven and the earth."

We can see with our eyes there's a difference - heavens vs heaven.
But how do we quantify this difference? We'd use this little bit of
code:

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

When we run this it prints:

```python
0.9907407407407407
```

Normalized scores are between `0` and `1`. A score of `1` means the
inputs are an exact match under the default scoring model. Scores near
`0` mean the inputs have little in common, though Smith-Waterman may
still find small local matches inside otherwise unrelated strings.

Now let's change the text and see what happens to the score.

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

and Python prints

```
0.12222222222222222
```

Starting to get the idea? The more similar the strings, the higher the score.

Let's build a bigger example, something that gives us a feel for the
library's performance. You'll probably notice that we switch between
Smith-Waterman and Needleman-Wunsch and may be wondering which to use
when. Use Needleman-Wunsch when you want to compare the whole input
against the whole input. Use Smith-Waterman when you want to find the
best matching region inside larger inputs.

Okay, let's move on to the demo code. You need `requests` for this
part of the demo:

```bash
pip install requests
```

```python
import os, time, requests
import stride_align as sa

if not os.path.exists("kjv.txt"):
    response = requests.get("https://openbible.com/textfiles/kjv.txt")
    response.raise_for_status()
    response.encoding = "utf-8-sig"
    open("kjv.txt", "w", encoding="utf-8").write(response.text)

lines = [line.strip().lower() for line in open("kjv.txt")][2:]

while True:
    if not (query := input("Enter a snippet to match.  Press enter to end.\n")):
        break
    t = time.perf_counter()
    scores = sa.needleman_wunsch_normalized_scores(query.lower(), lines)
    best = int(scores.argmax())
    print()
    print("Score:", float(scores[best]))
    print(lines[best])
    print("Search time: %0.2fms" % ((time.perf_counter() - t) * 1000))
    print()
    print()
```

Now how can we use this? Suppose we have a random Bible verse and
want to know what chapter and verse it comes from. `grep` you say?
Oh, heavens, no, we made a mistake. The verse we have is from a
different translation, say the Catholic Public Domain, and what we
have on our computer is the King James Bible. `grep`'s exact string
matching won't work here. How do we find the chapter and verse? We
search for the "closest" or "most similar" string using `stride-align`,
of course.

In our demo the first part concerns itself with downloading and
caching. The good folks at [Open Bible](https://openbible.com) put
this text where it's HTTP-reachable, but we want to be respectful of
their IT budget so we cache what we download. It's just good
citizenship.

In the next part we load all of the lines into a list. We remove
newlines and make everything lower case because we don't want to get
all fiddly about whether we're holding the shift key.

Lastly that `while True:` loop collects a line of text, presumably the
Bible verse from the Catholic version of the Bible we want to look up
the chapter and verse for, and matches it against all of the lines in
the King James Bible using the batch form of Needleman-Wunsch. It
returns an array of scores. We use `argmax()` to find the best-scoring
line and then print the line associated with that index. Let's try it.

I'm going to use Jeremiah 4:28 from the Catholic Bible - it's actually
quite different from the same verse in the King James Bible. Let's see
what happens ...

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

... and we found it! And pretty quickly too.


Now let's do another demo: spell checking.

This is a toy spell checker, not a production one. It ignores punctuation,
capitalization, word frequency, proper nouns, and context. The point is to
show the same one-query-against-many-candidates pattern on a familiar task.

```python
import os, sys
import stride_align as sa

paths = ['/usr/share/dict/words',
         '/usr/dict/words',
         '/var/lib/dict/words',
         '/etc/dictionaries-common/words']

for path in paths:
    if os.path.exists(path):
        break
else:
    print("Sorry, I can't find your dictionary", file=sys.stderr)
    exit(1)


words = [line.strip().lower() for line in open(path)]


for line in sys.stdin:
    new_line = []
    for word in line.split():
        scores = sa.needleman_wunsch_normalized_scores(word.lower(), words)
        word = words[int(scores.argmax())]
        new_line.append(word)
    print(' '.join(new_line), flush=True)
```

The first thing this script does is try to find our operating system's
list of correctly spelled words. Its location can vary from
distribution to distribution. Once we've found it, we load it, strip
off newlines and start the act of spell checking.

The spell checking looks a lot like the matching we did before. For
each candidate word, we match it against all of the words in our list
of correctly spelled words, use `argmax()` to find the highest-scoring
candidate, and replace the word with that candidate. We could speed
things up with some optimizations, like not searching for a match for
correctly spelled words, but this is a demo and that optimization is
left as an exercise for the reader.

Let's see how it works!

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## Details

The current scaffold provides:

- Needleman-Wunsch score-only alignment
- Needleman-Wunsch alignment with traceback
- Smith-Waterman score-only alignment
- Smith-Waterman alignment with traceback
- A backend layout that matches the specialization pattern used in `massive-speedup`
- CPU/backend detection and Python-side backend dispatch

The native boundary accepts:

- `bytes` against `bytes`
- `str` against `str`
- sequences of immutable hashable Python objects
- mixed sequence/object inputs where a `str` or `bytes` side is treated as a sequence

Direct `bytes` versus `str` pairs raise `TypeError`.

The current implementations are generic dynamic-programming kernels with preprocessing
that serializes Python inputs into 8, 16, 32, or 64-bit token streams. SIMD-specialized
backends can replace the backend translation units later without changing the Python API.

Score-only functions return numeric scores. The normalized variants return
scores between `0` and `1`. Path functions return alignment result objects
containing the score, aligned values, operations, and CIGAR-style summaries
where available.

## API

```python
import stride_align

score = stride_align.needleman_wunsch_score("ACGT", "ACCT")
scores = stride_align.Scores("ACGT", variant="needleman_wunsch").compare(["ACCT", "AGGT"])
result = stride_align.smith_waterman_path("ACCGT", "CCG")
wide_result = stride_align.smith_waterman_path("ACCGT", "CCG", width=64)
object_result = stride_align.needleman_wunsch_path(
    [frozenset({1}), frozenset({2})],
    [frozenset({1}), frozenset({3})],
)

print(score)
print(scores)
print(result.score, result.aligned_query, result.aligned_target, result.operations)
print(wide_result.score)
print(object_result.aligned_query, object_result.aligned_target)
```

Use `Scores(...).compare([...])` or the `*_scores()` functions for one-query
against many-target score workloads. That path prepares the query/profile once
and is the preferred performance API for repeated English/Chinese text
comparisons.

Traceback outputs preserve the paired fast-path type:

- `str` inputs return aligned `str`
- `bytes` inputs return aligned `bytes`
- sequence/object inputs return aligned `tuple` values with `None` gaps

Pass `width=8`, `16`, `32`, or `64` to force the internal token/scoring width
instead of using automatic selection.

Some functions expose CIGAR strings, short for "Concise Idiosyncratic
Gapped Alignment Report". CIGAR is the compact alignment-operation notation
used by SAM/BAM tooling. If you want the full formal version, see the
[SAM specification](https://samtools.github.io/hts-specs/SAMv1.pdf).

### Substitution matrices (BLOSUM, PAM)

For protein alignment, `stride_align.matrices` ships the canonical
BLOSUM and PAM substitution matrices. Pass any of them via the
`matrix=` kwarg on `smith_waterman_score`, `needleman_wunsch_score`,
or their `_scores` batch counterparts:

```python
import stride_align
from stride_align.matrices import blosum62, pam250

# Local alignment, NCBI standard BLOSUM62 with affine gaps (open=-11,
# extend=-1). matrix= is mutually exclusive with match_score / mismatch_score.
stride_align.smith_waterman_score(
    "HEAGAWGHEE", "PAWHEAE",
    matrix=blosum62,
    gap_open_score=-11, gap_extend_score=-1,
)

# Batch (1 query × N targets) with profile reuse — the recommended
# path for "score one query against a library".
stride_align.smith_waterman_scores(
    "HEAGAWGHEE",
    ["PAWHEAE", "HEAGAWGHEE", "MEEPS"],
    matrix=pam250, gap_open_score=-14, gap_extend_score=-2,
)

# Custom matrices: parse any NCBI-format text file
custom = stride_align.matrices.SubstitutionMatrix.from_ncbi_text(
    open("/path/to/BLOSUM62").read(),
    name="BLOSUM62",
    gap_open=-11, gap_extend=-1,
)
```

Each built-in `SubstitutionMatrix` exposes its alphabet, matrix data
(`int8` ndarray), and recommended gap defaults via `.gap_score`
(linear), `.gap_open`, and `.gap_extend`. Both linear gaps (`gap_score=`)
and affine gaps (`gap_open_score=` + `gap_extend_score=`) are
supported on the AVX-512 backend; other SIMD backends currently fall
back to the scalar generic kernel for matrix-mode.

The shipped matrix values come from the NCBI BLAST distribution
[`ftp.ncbi.nih.gov/blast/matrices/`](https://ftp.ncbi.nih.gov/blast/matrices/),
which carries the canonical reference scores. The original
publications are:

- **BLOSUM45 / 50 / 62 / 80 / 90** — Henikoff S., Henikoff J.G. (1992).
  *Amino acid substitution matrices from protein blocks*. PNAS
  89(22):10915–10919.
  [doi:10.1073/pnas.89.22.10915](https://doi.org/10.1073/pnas.89.22.10915)
  &nbsp;·&nbsp;
  [PDF (open access)](https://www.pnas.org/doi/pdf/10.1073/pnas.89.22.10915)
- **PAM30 / 70 / 250** — Dayhoff M.O., Schwartz R.M., Orcutt B.C.
  (1978). *A model of evolutionary change in proteins*. In *Atlas of
  Protein Sequence and Structure*, vol. 5, supplement 3, pages 345–352.
  National Biomedical Research Foundation, Washington, D.C. (Book
  chapter; not available online as an open PDF. A widely cited
  follow-on derivation appears in Schwartz R.M., Dayhoff M.O. (1978),
  *Matrices for detecting distant relationships*, same volume,
  pages 353–358.)

### Edit-distance scorers

Beyond Smith-Waterman and Needleman-Wunsch, `stride-align` exposes
six unit-cost edit-distance and similarity metrics — each with its
own SIMD-batched code path:

```python
import stride_align

# Levenshtein (Myers 1999 bit-parallel) — inserts, deletes, substitutes
stride_align.levenshtein_score("kitten", "sitting")               # -> 3
stride_align.levenshtein_normalized_score("kitten", "sitting")    # -> 0.571...
stride_align.levenshtein_scores("kitten", ["kit", "sitting"])     # -> ndarray[int64]

# Optional `score_cutoff` (rapidfuzz convention): bail early per-target,
# results that exceed the cutoff come back as `cutoff + 1`.
stride_align.levenshtein_scores(query, targets, score_cutoff=3)

# Damerau-Levenshtein (OSA-restricted, Hyyrö 2002) — adds adjacent
# transposition at unit cost. This is what rapidfuzz exposes as
# OSA.distance and is what most callers asking for
# "Damerau-Levenshtein" actually want.
stride_align.damerau_levenshtein_score("ab", "ba")                # -> 1

# True Damerau-Levenshtein — the unrestricted form, where one
# character may participate in more than one edit. Slower (no
# bit-parallel kernel yet) but matches rapidfuzz.distance.DamerauLevenshtein
# exactly. Diverges from OSA on overlapping transpositions, e.g.
# "ca" -> "abc": OSA=3, true-DL=2.
stride_align.true_damerau_levenshtein_score("ca", "abc")          # -> 2

# Indel — Levenshtein restricted to insertions and deletions, no
# substitutions. Equivalent to |a| + |b| - 2 * LCS(a, b). Bit-
# parallel Allison-Dix (1986) inner loop.
stride_align.indel_score("kitten", "sitting")                     # -> 5

# Hamming — count of positions where two equal-length strings differ.
# Cutoff variant bails the byte loop once mismatches exceed the cap.
stride_align.hamming_score("100", "110")                          # -> 1

# Jaro / Jaro-Winkler — similarities in [0, 1]; Winkler adds a
# capped prefix bonus.
stride_align.jaro_similarity("martha", "marhta")                  # -> 0.944...
stride_align.jaro_winkler_similarity("martha", "marhta")          # -> 0.961...
```

The batch variants (`*_scores`, `*_similarities`) pack one target
per SIMD lane on every supported backend:

- x86: SSE4.1 / AVX2 / AVX-512 / AVX10-256 / AVX10-512
- ARM: NEON (Linux + macOS), SVE / SVE2
- LoongArch: LSX / LASX
- PowerPC: VSX

For Lev / OSA, patterns up to 64 chars run a single-word Myers;
65–256 chars use the multi-word kernel (W=2/3/4). Indel and OSA
fall back to scalar bit-parallel for patterns >64 (multi-word
generalization deferred); true-DL is scalar DP only.

### Longest Common Subsequence + Substring

Two related but distinct dynamic programs, both shipped:

```python
import stride_align as sa

# Longest Common Subsequence — characters need not be contiguous.
# "ABCBDAB" and "BDCAB" share "BCAB" (length 4).
sa.lcs_length("ABCBDAB", "BDCAB")                    # -> 4

# Closed-form relation to Indel distance: indel = |a| + |b| - 2·LCS.
sa.indel_score("kitten", "sitting") == \
    len("kitten") + len("sitting") - 2 * sa.lcs_length("kitten", "sitting")
# -> True

# Longest Common Substring — characters MUST be contiguous.
sa.lcs_substring_length("ABCBDAB", "BDCAB")          # -> 2
sa.lcs_substring("ABCBDAB", "BDCAB")                 # -> "AB"

# Result type matches inputs: bytes in, bytes out.
sa.lcs_substring(b"hello world", b"world hello")     # -> b"hello"

# Codepoint engine — non-ASCII is first-class.
sa.lcs_substring("Müller", "Mueller")                # -> "ller"
```

Both DPs are scalar `O(m·n)` time with two rolling rows for
`O(min(m,n))` (subsequence) or `O(|b|)` (substring) space. When
multiple substrings tie at the maximum length, the first occurrence
in `a` is returned (matches `str.find` convention).

### Ratcliff-Obershelp similarity

The algorithm Python's `difflib.SequenceMatcher().ratio()` ships,
which `rapidfuzz` does not — recursive longest-matching-substring
split, summed match lengths divided by total length:

```python
import stride_align as sa

sa.ratcliff_obershelp_similarity("kitten", "sitting")
# -> 0.6153846153846154

# Bit-exact with difflib at autojunk=False (we have no junk
# character heuristic):
import difflib
sa.ratcliff_obershelp_similarity("ABCBDAB", "BDCAB") == \
    difflib.SequenceMatcher(None, "ABCBDAB", "BDCAB", autojunk=False).ratio()
# -> True

# Batch form: one query against many targets, returned as
# ndarray[float64].
sa.ratcliff_obershelp_similarities("kitten",
                                    ["sitting", "kitten", "kit"])
# -> array([0.61538462, 1.        , 0.66666667])
```

Not commutative — the inner longest-common-substring tiebreak
(`earliest in a, then earliest in b`) means the recursion splits
leftover ranges differently for `(a, b)` vs `(b, a)`, and the total
match length can differ. Faithful to difflib, which has the same
property; `sa.ratcliff_obershelp_similarity("ABCBDAB", "BDCAB")`
gives `0.333…` while the reverse gives `0.667…`. Pin both
directions if your tests need an order-independent metric.

### Phonetic encoders

For name matching, deduplication, and search-as-you-type, `stride-align`
ships the full standard phonetic-encoder family. Each encoder maps a
string to a short code such that names that *sound* similar share a
code, regardless of spelling:

```python
import stride_align as sa

# American Soundex (Russell & Odell, 1918). 4-character code.
sa.soundex("Robert")                                     # -> "R163"
sa.soundex("Rupert")                                     # -> "R163"
sa.soundex_equal("Robert", "Rupert")                     # -> True

# Metaphone (Lawrence Philips, 1990) — two-letter and longer
# spec-correct variants. The published 1990 spec and the popular
# jellyfish library disagree on a handful of edge cases; the variant
# kwarg picks the rule family.
sa.metaphone("Schmidt")                                  # -> "SKMTT"  (PHILIPS, spec)
sa.metaphone("Schmidt", variant=sa.MetaphoneVariant.JELLYFISH)  # -> "SXMTT"
sa.metaphone_equal("Schmidt", "Smith")                   # -> False

# Double Metaphone (Lawrence Philips, 2000) — primary and alternate
# codes; the alternate captures plausible non-English pronunciations.
# COMMONS is the faithful Apache Commons Codec port; PYTHON is bug-
# compat with the metaphone PyPI package.
sa.double_metaphone("Schwartz")                          # -> ("XRTS", "XFRTS")
sa.double_metaphone("Hugh")                              # -> ("H", "")
sa.double_metaphone("Hugh",
    variant=sa.DoubleMetaphoneVariant.PYTHON)            # -> ("HH", "")

# NYSIIS (Taft, 1970). More discriminative than Soundex for English
# names — "Watkins" / "Wilkins" / "Wilkinson" don't collide.
sa.nysiis("Watkins"), sa.nysiis("Wilkins")               # -> ("WATCAN", "WALCAN")

# Match Rating Approach (Moore, Western Airlines, 1977). A codex plus
# a pairwise comparator with length-difference + rating-threshold rules.
sa.match_rating_codex("Christopher")                     # -> "CHRPHR"
sa.match_rating_compare("Robert", "Rupert")              # -> True

# Caverphone 2.0 (Hood, 2004). Fixed-length 10-character code,
# right-padded with '1'. Designed for late-19th-century New Zealand
# electoral rolls but widely applied to general English-language
# name matching.
sa.caverphone("Stevenson")                               # -> "STFNSN1111"
```

All seven encoders are dispatched through the same byte-extraction
helper, accept `str` and `bytes` inputs interchangeably, and skip
non-letter / non-ASCII codepoints before encoding — pre-normalise
with `unicodedata.normalize("NFKD", s)` if you want accent folding.
Cross-checked against the canonical Apache Commons Codec reference
data and the `jellyfish`, `metaphone`, and `doublemetaphone` PyPI
packages.

### Dynamic Time Warping

For aligning numeric sequences whose timing or speed varies — audio
signals, gesture / sensor traces, financial time series —
`stride-align` exposes Dynamic Time Warping with optional Sakoe-Chiba
band:

```python
import numpy as np
import stride_align as sa

q = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
t = np.array([1.0, 2.0, 2.5, 4.0, 5.0])

# Default distance follows the dtype:
#   float32 / float64 -> L2-squared, (x - y)^2
#   int16             -> L1, |x - y|  (audio convention)
sa.dtw(q, t)                                              # -> 0.25

# Sakoe-Chiba band: int radius or fraction of max(|q|, |t|).
sa.dtw(q, t, window=2)
sa.dtw(q, t, window=0.2)

# Explicit distance.
sa.dtw(q.astype(np.int16), t.astype(np.int16), distance="l1")

# Batch.
sa.dtw_distances(q, [t, t * 2, t + 0.5], window=2)
```

Inputs must be NumPy `ndarray` with matching dtype (`float32`,
`float64`, or `int16` — the natural audio dtype). Other dtypes and
non-ndarray inputs raise `TypeError`.

### `cdist`, `cdist_above_threshold`, `cdist_top_k`, `cdist_top_k_per_query`

For all-pairs scoring across two lists of strings, `stride-align`
ships three matrix-style entry points:

```python
qs = ["kitten", "sitting", "kit"]
ts = ["kitten", "kit", "sitting", "biting"]

# Full N×M similarity matrix — ndarray[float64] (similarity scorers)
# or ndarray[int64] (distance scorers).
sa.cdist(qs, ts, scorer=sa.Scorer.JARO)

# Streaming filter — yields only pairs whose similarity exceeds the
# threshold. Workers feed a bounded queue; the caller drains it.
# Length pruning + per-pair cutoff push-down into the kernel skip
# most of the work at high thresholds.
for score, q, t in sa.cdist_above_threshold(
    qs, ts, scorer=sa.Scorer.LEVENSHTEIN_NORMALIZED, threshold=0.7,
):
    ...

# Top-k by score — returns at most k highest-scoring (or lowest, for
# distance scorers) (score, query, target) tuples. Heaps are
# per-thread; a shared atomic global-min bound lets the per-pair
# cutoff push-down lift the prune threshold as work progresses.
sa.cdist_top_k(qs, ts, scorer=sa.Scorer.JARO, k=10)

# Top-k targets PER QUERY, yielded as a generator. Differs from
# cdist_top_k (which returns the k highest pairs globally) by keeping
# a separate top-k heap per query. With pruning=True, the worst-in-
# heap score adapts as scoring progresses and targets whose closed-
# form length-difference upper bound on similarity can't beat it
# are skipped before the kernel runs — a big win on workloads with
# wide length variation.
for query, top in sa.cdist_top_k_per_query(
    qs, ts, scorer=sa.Scorer.LEVENSHTEIN_NORMALIZED, k=5, pruning=True,
):
    # top is [(score, target), ...] sorted descending
    ...
```

At high thresholds the pruning is dramatic — see the cross-arch
table in [BENCHMARK.md](https://stride-align.com/BENCHMARK.html) (the `cdist pruning` rows).
Loongson LASX in particular flips the expected ranking against
Tiger Lake AVX-512 at T=0.99; the comparison report lives at
[docs/loongson-vs-tiger-lake-cdist-2026-05-24.md](docs/loongson-vs-tiger-lake-cdist-2026-05-24.md).

See [BENCHMARK.md](https://stride-align.com/BENCHMARK.html) for full cross-architecture numbers.

## Optimizations and Benchmarks

Careful attention has been, and continues to be, paid to `stride-align`'s
performance story. The library includes SIMD optimization for a variety of
common targets, including x86, Arm, and LoongArch.

**LoongArch / Loongson.** The Loongson optimization story is especially
telling: for the checked benchmark case -- English text, 16-bit score width,
score-only Smith-Waterman -- the LASX backend is 16x faster than the generic
backend and **22.4x** faster than Parasail.

If you are a researcher using Loongson servers and benefiting from this
speedup, citations, bug reports, benchmark cases, and tiny inexpensive Chinese
souvenirs are appreciated. Tea, calligraphy bookmarks, paper-cut ornaments,
Chinese knot charms, panda keychains, and small dragon desk objects are all
welcome. Please do not send anything expensive or anything that requires
customs paperwork.

See [complete benchmarks](https://stride-align.com/BENCHMARK.html).

## Native Microbench

For perf profiling without Python frames or benchmark orchestration, configure a
native x86 microbench build:

```bash
nanobind_dir="$(.venv/bin/python -m nanobind --cmake_dir)"
cmake -S . -B build/perf \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSTRIDE_ALIGN_BUILD_MICROBENCH=ON \
  -DSTRIDE_ALIGN_PERF_SYMBOLS=ON \
  -DPython_EXECUTABLE=.venv/bin/python \
  -Dnanobind_DIR="$nanobind_dir"
cmake --build build/perf --target stride_align_x86_microbench
build/perf/stride_align_x86_microbench --backend avx2 --shape 1:many --pass english --width 16
python tools/x86_microbench_regression.py \
  --binary build/perf/stride_align_x86_microbench \
  --cpu 2 \
  --backends avx2,avx512bwvl \
  --shapes 1:1,1:many \
  --passes english,chinese \
  --widths 16,32 \
  --write-json /tmp/stride-align-x86-microbench.json
.venv/bin/python tools/pinned_benchmark_sweep.py \
  --output-dir /tmp/stride-align-pinned \
  --cpu 2 \
  --iterations 15 \
  --warmups 3
```

`STRIDE_ALIGN_PERF_SYMBOLS=ON` keeps nanobind modules unstripped and adds debug
symbols plus frame pointers while preserving `-O3`.

The checked-in native microbench baseline lives at
`benchmarks/x86_microbench_baseline.json`. Treat it as a local guardrail with a
loose threshold, not as a cross-machine SLA.


## LoongArch installation

LoongArch wheels are **not on PyPI** (PyPI doesn't index the
`linux_loongarch64` platform tag), so they ship through a different
channel:

| Channel | URL prefix |
| --- | --- |
| GitHub Releases (primary) | `https://github.com/adamdeprince/stride-align/releases/download/v0.4.0/` |
| `stride-align.com` mirror | `https://stride-align.com/wheels/v0.4.0/` |

Same wheels on both, pick whichever loads faster from your network.
The mirror is convenient when GitHub egress is slow from inside
China; GitHub Releases is the canonical home.

Install NumPy from your distro first (loongarch64 NumPy wheels are
sparse on PyPI, and the distro one is usually ABI-compatible with
the rest of the system):

```bash
sudo apt install python3-numpy
PY=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
```

### Old-world vs new-world: what to pick

LoongArch hardware runs in one of two mutually incompatible binary
**worlds**. They differ in two things:

1. **Which dynamic loader the executable references** (this is the
   filename baked into the ELF header at link time).
2. **Which glibc ABI version the binary depends on**.

A wheel built for one world will not load on the other — the loader
filename doesn't exist on the wrong side, and the symbols would
mismatch even if it did. We ship one wheel per world:

| World | Loader | glibc | Typical hosts | Wheel build tag |
| --- | --- | --- | --- | --- |
| **Old-world** | `/lib64/ld.so.1` | 2.28-era | Stock Kylin, original Loongson distros | `1.oldworld` |
| **New-world** | `/lib64/ld-linux-loongarch-lp64d.so.1` | ≥ 2.36 | Recent LoongArch distros, anything where the new loader has been installed | `1.newworld` |

Both wheels are statically linked against libstdc++ / libgcc so the
only thing separating them is the loader / glibc ABI.

**Pick the right one with this one-liner**:

```bash
test -e /lib64/ld-linux-loongarch-lp64d.so.1 && echo new-world || echo old-world
```

If you see `new-world`, the loader is in place — use the new-world
wheel. If you see `old-world`, either install the old-world wheel,
or run the one-time sudo symlink below to enter new-world land
(safe — it's a new filename, not a replacement, so existing
old-world binaries keep working).

### Old-world wheel

```bash
pip install \
  https://github.com/adamdeprince/stride-align/releases/download/v0.4.0/stride_align-0.4.0-1.oldworld-${PY}-${PY}-linux_loongarch64.whl
```

Mirror:

```bash
pip install \
  https://stride-align.com/wheels/v0.4.0/stride_align-0.4.0-1.oldworld-${PY}-${PY}-linux_loongarch64.whl
```

### New-world wheel

The new-world wheel needs the new loader available at the path the
ELF references. One sudo step, once per box, leaves old-world
binaries unaffected:

```bash
sudo ln -sf /opt/loongson-gcc-16.1.0/sysroot/lib64/ld-linux-loongarch-lp64d.so.1 \
            /lib64/ld-linux-loongarch-lp64d.so.1
```

(Distro packagers usually drop an equivalent symlink as part of the
new-world transition, in which case you can skip this.)

Then:

```bash
pip install \
  https://github.com/adamdeprince/stride-align/releases/download/v0.4.0/stride_align-0.4.0-1.newworld-${PY}-${PY}-linux_loongarch64.whl
```

Mirror:

```bash
pip install \
  https://stride-align.com/wheels/v0.4.0/stride_align-0.4.0-1.newworld-${PY}-${PY}-linux_loongarch64.whl
```

### Other notes

Prebuilt LoongArch64 wheels are available for Python 3.12, 3.13,
and 3.14 — in both worlds — on both mirrors. The build details
(toolchains, RPATH wrapper, static C++ runtime) live in
[docs/loongson-build.md](docs/loongson-build.md). If you're on a
different Python or want to build from source, `pip install
stride-align` falls back to the PyPI source distribution and
compiles the LSX/LASX kernels locally.


## Citations

If you use my software in your research, please cite me.

```bibtex
@software{deprince_stride_align,
  author       = {DePrince, Adam},
  title        = {stride-align: Fast Smith-Waterman and Needleman-Wunsch alignment for Python},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/adamdeprince/stride-align},
  note         = {Python/C++ library for sequence and string alignment}
}
```
