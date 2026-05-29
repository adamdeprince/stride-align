# TODO: Character similarity gap-filling (Phase D)

**Status:** queued, runs before continuing the numeric / DTW work in phase C.
**Goal:** round out the character-based similarity surface to match what
mainstream fuzzy-matching libraries (rapidfuzz, jellyfish, fuzzywuzzy,
textdistance) expose, so users picking up stride-align don't have to
reach for a second library for table-stakes operations like Jaccard or
Soundex.

The numeric DTW path (phase C.1b — x86 SIMD batch and beyond — is on
hold until phase D ships.

## Where stride-align stands today

Already shipped (and committed-baseline as of `d26910b`):

* Smith-Waterman (local, linear + affine + matrix + path / CIGAR).
* Needleman-Wunsch (global, linear + affine + matrix + path / CIGAR).
* Levenshtein (+ `LevenshteinScorer` with prepared PEQ).
* Damerau-Levenshtein OSA (restricted transpositions).
* True Damerau-Levenshtein (unrestricted transpositions).
* Hamming.
* Indel.
* Jaro and Jaro-Winkler.
* DTW scalar reference (phase C.1a; on hold for SIMD until phase D
  finishes).

This covers the edit-distance and alignment families exhaustively for
the byte / 1-byte-unicode hot path and wider tokens via Phase B's
wide kernels. The seven gaps below sit in **adjacent families** that
stride-align hasn't touched.

## Phase D.1 — n-gram set similarity (Jaccard, Sørensen-Dice, Cosine, Tversky, Overlap)

Workhorse family for dedup, MinHash-style retrieval, and rapidfuzz's
`fuzz.token_*_ratio` substrate. Biggest practical gap.

**Algorithms**

| API | Formula | Notes |
| --- | --- | --- |
| `sa.jaccard(s1, s2, *, n=2)` | `|A ∩ B| / |A ∪ B|` | Symmetric; harsh on short strings |
| `sa.sorensen_dice(s1, s2, *, n=2)` | `2|A ∩ B| / (|A| + |B|)` | Softer; common in NLP |
| `sa.tversky(s1, s2, *, n=2, alpha, beta)` | `|A ∩ B| / (|A ∩ B| + α|A−B| + β|B−A|)` | Asymmetric |
| `sa.overlap_coefficient(s1, s2, *, n=2)` | `|A ∩ B| / min(|A|, |B|)` | Containment matching |
| `sa.cosine_ngram(s1, s2, *, n=2)` | `(v_A · v_B) / (‖v_A‖·‖v_B‖)` on n-gram counts | Multiset variant |

Plus the batch variants and normalised forms following the existing
`*_score` / `*_scores` / `*_normalized_score` convention.

**C++ surface**

```
include/stride_align/ngram_set.hpp
  template <typename Token>
  struct NgramSet {
    // sorted, deduplicated 64-bit ngram hashes.
    std::vector<std::uint64_t> hashes;
    std::size_t total_count;  // for cosine / multiset variants
  };

  inline NgramSet<...> build_ngrams(span<const Token> s, std::size_t n);

  inline double jaccard(const NgramSet& a, const NgramSet& b);
  inline double sorensen_dice(const NgramSet& a, const NgramSet& b);
  inline double tversky(const NgramSet& a, const NgramSet& b, double alpha, double beta);
  inline double overlap_coefficient(const NgramSet& a, const NgramSet& b);
  inline double cosine_multiset(const NgramSetMulti& a, const NgramSetMulti& b);
```

The expensive operations are (1) n-gram hashing and (2) sorted-set
intersection. The dispatcher picks the narrowest representation
that holds the n-gram key **exactly**. Whenever the n-gram exceeds
64 bits the kernel stores each n-gram as a packed byte blob of
width `stride = n * sizeof(Token)` — one code path regardless of
Token — in a sorted contiguous buffer, intersected by a two-pointer
scalar `memcmp` merge. Slow per intersection but still correct, no
answer denied, no silent collisions, no padding waste.

**Representation dispatch (exact, no collisions):**

| Input | n-gram bits | n=1 | n=2 | n=3 | n=4 | n=5+ |
| --- | ---: | --- | --- | --- | --- | --- |
| **bytes / `uint8` / `int8`** (8 b) | 8n | bitset 32 B | bitset 8 KB | `uint32` SIMD | `uint32` SIMD | `uint64` SIMD through n=8, then packed-byte scalar |
| **UCS-2 / `uint16` / `int16`** (16 b) | 16n | bitset 8 KB | **`uint32` SIMD (32 b)** | **`uint64` SIMD (48 b)** | **`uint64` SIMD (64 b)** | packed-byte scalar (stride=2n) |
| **UCS-4 / `uint32` / `int32`** (32 b) | 32n | `uint32` SIMD | `uint64` SIMD (64 b) | packed-byte scalar (stride=4n) | packed-byte scalar | packed-byte scalar |
| **`int64` / `uint64` / `float*`** (64 b) | 64n | `uint64` SIMD | packed-byte scalar (stride=8n) | packed-byte scalar | packed-byte scalar | packed-byte scalar |
| **`list` / `tuple` of objects** | — | PyDict-keyed multiset / set | … | … | … | … |

* Bitset rows are one or two AVX2 ANDs plus POPCNT regardless of
  input length.
* `uint32` rows use 8-lane AVX2 / 16-lane AVX-512 SIMD intersection
  (broadcast candidate, `vpcmpeqd`, `vpmovmskd`, `popcnt`) — the
  fast path for Chinese n=2.
* `uint64` rows use 4-lane AVX2 / 8-lane AVX-512 SIMD intersection
  — half the throughput of `uint32`, still SIMD, exact for common
  UCS-2 n=3 / n=4 and UCS-4 n=2.
* **Packed-byte scalar** rows store each n-gram as `stride =
  n * sizeof(Token)` raw bytes in a flat sorted buffer. A UCS-2
  5-gram is 10 bytes, a bytes 10-gram is 10 bytes, a UCS-4
  trigram is 12 bytes — natural width, no padding waste.
  Intersection is a two-pointer scalar `memcmp` merge. One code
  path for every (Token, n) above the SIMD boundary; the runtime
  stride parameter is the only thing that changes.
* List-of-objects falls back further still: a PyDict-keyed
  multiset / set computed against the Python objects directly.
  Slowest, necessary for arbitrary hashable token sequences.

**Prepared / streaming variant:**

```python
scorer = sa.JaccardScorer(query, n=2)
scorer.similarity(target)
scorer.similarities(targets)
```

Mirrors `LevenshteinScorer`. Builds the query representation
(bitset or sorted hash vector) once; each subsequent call is the
intersection step only. This is where the win compounds for the
1-query × many-targets workload.

## Phase D.2 — phonetic encoders (Soundex, Metaphone, Double Metaphone)

Encoders, not distances — they emit a string (or tuple of strings).
You compare those for equality, or pass through Levenshtein. Pure
character-state-machine implementations; no SIMD speedup worth
chasing for the encoders themselves (state machines don't
vectorise). The value is having a uniform Python API and a tested
C++ implementation alongside the rest of stride-align.

| API | Output | Notes |
| --- | --- | --- |
| `sa.soundex(s)` | 4-char ASCII | American Soundex (1918). Length-clipped + 0-padded. |
| `sa.metaphone(s)` | variable ASCII | Lawrence Philips' 1990 encoder. |
| `sa.double_metaphone(s)` | `tuple[str, str]` | Primary + alternate; non-English friendly. |

Plus the obvious `sa.soundex_equal(s1, s2)` convenience wrappers.

**C++ surface**

```
include/stride_align/phonetic/soundex.hpp
include/stride_align/phonetic/metaphone.hpp
include/stride_align/phonetic/double_metaphone.hpp
src/cpp/phonetic_dispatch.hpp
```

**Notes:** Encoders are 100-300 LOC each; tests are the real cost
because the rules have many edge cases.

## Phase D.3 — token ratio family (rapidfuzz parity)

Builds on phase D.1's tokenisation + the existing Levenshtein.
Closes the direct rapidfuzz API gap: users porting from
`rapidfuzz.fuzz.token_set_ratio(...)` get a one-for-one replacement.

| API | Definition |
| --- | --- |
| `sa.token_sort_ratio(s1, s2)` | Split on whitespace, sort tokens, Levenshtein-ratio the joined strings. |
| `sa.token_set_ratio(s1, s2)` | Set intersection / difference of tokens, three Lev-ratio comparisons, take the max. |
| `sa.partial_token_sort_ratio(s1, s2)` | Token-sort + partial-ratio (best-matching substring). |
| `sa.partial_token_set_ratio(s1, s2)` | Token-set + partial-ratio. |
| `sa.wratio(s1, s2)` | rapidfuzz's weighted blend of the above. |

**Notes:** Pure Python composition on top of phase D.1 and existing
Levenshtein; no new C++ kernels.

## Phase D.4 — Longest Common Subsequence + Substring

LCS is implicit today (`indel = m + n − 2·LCS`) but not exposed; LC
Substring is a different DP and genuinely missing.

| API | Definition | DP |
| --- | --- | --- |
| `sa.lcs_length(s1, s2)` | Length of the longest common **subsequence** (not necessarily contiguous) | Same recurrence as Indel; one-liner shim |
| `sa.lcs_substring_length(s1, s2)` | Length of the longest common **substring** (contiguous) | Different: `dp[i][j] = dp[i-1][j-1] + 1` if match else 0 |
| `sa.lcs_substring(s1, s2)` | The substring itself | One traceback step after the DP |

**Notes:** The substring DP is SIMD-friendly (same anti-diagonal
pattern as SW), but a scalar implementation is fine for v1.

## Phase D.5 — Ratcliff-Obershelp

Python's `difflib.SequenceMatcher` algorithm. Recursive longest
matching substring; users coming from `difflib` look for this and
don't find it in `rapidfuzz` either, so a real differentiator.

| API | Notes |
| --- | --- |
| `sa.ratcliff_obershelp(s1, s2)` | The classic ratio in `[0, 1]` |
| `sa.ratcliff_obershelp_matching_blocks(s1, s2)` | The matching-block list (for diff-style use cases) |

**C++ surface**

```
include/stride_align/ratcliff_obershelp.hpp
  // Recursive longest matching substring + recurse on non-matching parts.
  // Worst case O(n*m); average ~ O((n+m) log (n+m)) on real text.
```

**Notes:** The recursive split is straightforward; the longest-
matching-substring inner step shares code with phase D.4.

## Phase D.6 — Monge-Elkan

Multi-token hybrid metric — for each token in s1, find the best-
matching token in s2 by an inner similarity (default Jaro or
Levenshtein-ratio); average across s1's tokens. Standard in record
linkage.

| API | Notes |
| --- | --- |
| `sa.monge_elkan(s1, s2, *, inner="jaro")` | `inner` selects the per-token metric: `"jaro"`, `"jaro_winkler"`, `"levenshtein_ratio"`. |

**Notes:** Pure Python composition on top of the existing per-token
metrics; no new C++ kernels.

## Phase D.7 — additional phonetic encoders

Lower priority; ship only when a real user asks. Bundle so the work
can be done in one focused session.

| API | Notes |
| --- | --- |
| `sa.nysiis(s)` | US Census, longer than Soundex. |
| `sa.caverphone(s)` | New Zealand origin, international names. |
| `sa.match_rating_codex(s)` | + `sa.match_rating_comparison(s1, s2) -> bool` |
| `sa.beider_morse(s, lang=None)` | Multi-language; substantial implementation (large rule tables). |

**Notes:** Soundex / NYSIIS / Caverphone are each small encoders;
Match Rating Approach is small + a comparison rule; Beider-Morse is
much larger than the rest of D combined (huge rule tables + language
detection).

## Ordering / dependency graph

```
D.1 (n-gram sets) ──┬─► D.3 (token ratios — needs D.1)
                    └─► D.6 (Monge-Elkan — needs D.1 for cosine-on-tokens variant)

D.2 (Soundex / Metaphone / Double Metaphone) — standalone

D.4 (LCS / LC Substring) ──► D.5 (Ratcliff-Obershelp — needs D.4)

D.7 (extra phonetics) — standalone, low priority
```

Recommended sequence:

1. **D.2** (phonetic basics) — small, mostly self-contained, immediate
   user-visible gap closure.
2. **D.1** (n-gram set similarity) — biggest practical impact; unblocks
   D.3 and D.6.
3. **D.3** (rapidfuzz token-ratio parity) — fast follow once D.1 ships.
4. **D.4** (LCS / LC Substring) — small, useful.
5. **D.5** (Ratcliff-Obershelp) — differentiator vs rapidfuzz; depends
   on D.4.
6. **D.6** (Monge-Elkan) — record-linkage niche; depends on D.1.
7. **D.7** (extra phonetics) — only if asked.

## Effort sizing

Each of D.1-D.6 is well-scoped: a single algorithm family per
phase, clean dispatch boundaries against the existing Python
surface, and tests that follow the established `test_*.py`
patterns. D.7 is open-ended depending on which encoders ship —
Soundex / NYSIIS / Caverphone are tight; Beider-Morse alone is
larger than the rest of D combined.

## After phase D

Resume numerics:

* Phase C.1b — DTW x86 SIMD batch kernel (4-5 days).
* Phase C.1c — DTW ARM / Loongson SIMD (2-3 days).
* Phase C.1d — DTW benchmarks (2 days).
* Phase C.2+ — LB_Keogh, subsequence DTW, LCSS, ERP, Fréchet, warping
  path, multidim. See [docs/TODO-dtw.md](TODO-dtw.md).

## Related

- [docs/TODO-dtw.md](TODO-dtw.md) — paused numeric work resuming after
  this phase D.
- [docs/TODO-wide-traceback.md](TODO-wide-traceback.md) — separate
  string-side TODO (wide-token traceback / path / CIGAR).
- [docs/TODO-matrix-roadmap.md](TODO-matrix-roadmap.md) — substitution-
  matrix continuation.
