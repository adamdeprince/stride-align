# Phase D external-source audit

Inventory of every external source that contributes (or is explicitly
excluded from contributing) code, data, structure, or test oracles to
the Phase D character-similarity gap-fillers in `stride-align`:

* **D.1** — n-gram set similarity (Jaccard, Dice, Cosine, Overlap)
* **D.3** — token-ratio family (`token_sort_ratio`, `token_set_ratio`,
  `partial_ratio`, `partial_token_sort_ratio`,
  `partial_token_set_ratio`, `WRatio`)
* **D.4** — Longest Common Subsequence + Longest Common Substring
* **D.5** — Ratcliff-Obershelp similarity
* **D.6** — Monge-Elkan multi-token hybrid

The project ships under Apache-2.0; every source actually used must be
compatible. The companion audit for the phonetic encoders lives in
[`docs/bmpm-external-sources.md`](bmpm-external-sources.md).

## Posture, in one paragraph

Phase D is implemented from algorithm descriptions only. No third-
party source code is read, copied, or paraphrased into the production
path. The production path imports nothing beyond stride-align's own
C++ kernels, Python stdlib (built-in operations on `str` / `list` /
`set` / `dict`), nanobind for the dispatcher boundary, and numpy for
return arrays. Runtime test oracles are limited to libraries with
explicitly-compatible licences (PSF, MIT, BSD) and are invoked as
black boxes — no oracle's source code is consulted.

## 1. stride-align's own C++ kernels (Apache-2.0)

The base similarity primitives that the Phase D Python compositions
call into:

| Kernel | Used by | Header |
| --- | --- | --- |
| Longest Common Subsequence DP | D.1 cross-check, D.3 `partial_ratio` indirectly, D.5 indirectly | `include/stride_align/lcs.hpp` |
| Longest Common Substring DP | D.4, D.5, D.3 `partial_ratio` | `include/stride_align/lcs.hpp` |
| Bit-parallel Indel kernel | D.3 entire family (`indel_normalized_score` is rapidfuzz's `fuzz.ratio / 100`) | `include/stride_align/indel*.hpp` |
| Bit-parallel Levenshtein | D.6 `inner="levenshtein_ratio"` | `include/stride_align/levenshtein*.hpp` |
| Jaro | D.6 `inner="jaro"` (default) | `include/stride_align/jaro.hpp` |
| Jaro-Winkler | D.6 `inner="jaro_winkler"` | `include/stride_align/jaro_winkler.hpp` |

All under Apache-2.0; no third-party licence interaction.

## 2. Algorithm references (descriptions only, no code copied)

The published algorithms below are implemented from formula. No
specific implementation's source code was read or paraphrased.

| Phase | Algorithm | Original publication |
| --- | --- | --- |
| D.1 | Jaccard coefficient | Jaccard (1901), *Étude comparative de la distribution florale*. |
| D.1 | Sørensen-Dice | Sørensen (1948); Dice (1945). |
| D.1 | Overlap (Szymkiewicz-Simpson) | Szymkiewicz (1934); Simpson (1943). |
| D.1 | Cosine on multiset frequency vectors | Standard linear-algebra formulation. |
| D.3 | `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, `partial_token_*_ratio`, `WRatio` | Described in the rapidfuzz documentation (https://rapidfuzz.github.io/RapidFuzz/) and in the original FuzzyWuzzy blog post. The **descriptions** are referenced; the **implementations** are not — see §4 below. |
| D.4 | LCS subsequence DP | Hunt & Szymanski (1977); also Wagner & Fischer (1974). |
| D.4 | LC Substring DP | Standard `dp[i][j] = dp[i-1][j-1] + 1` recurrence; appears in any algorithms textbook. |
| D.5 | Ratcliff-Obershelp | Ratcliff & Metzener (1988), *Pattern Matching: The Gestalt Approach*, Dr. Dobb's Journal. |
| D.6 | Monge-Elkan | Monge & Elkan (1996), *The Field Matching Problem: Algorithms and Applications*, KDD '96. |

These are mathematical / textbook references, not licensed code.

## 3. Permitted runtime test oracles

Used **only** from `tests/`. None is imported by any production code
shipped in the wheel.

### 3a. rapidfuzz

- **URL:** https://github.com/rapidfuzz/rapidfuzz
- **License:** MIT (Apache-2.0 compatible)
- **What is used:** Parity oracle for `tests/test_fuzz.py`. The test
  imports `rapidfuzz.fuzz` and calls `token_sort_ratio`,
  `token_set_ratio`, `partial_ratio`, `partial_token_sort_ratio`,
  `partial_token_set_ratio` to verify that stride-align matches.
  rapidfuzz is **invoked as a black box** — its source code, fixtures,
  and internal algorithms were not read.
- **What is *not* used:** the rapidfuzz C++ kernel source, its
  algorithm internals, its test fixtures.
- **Attribution / NOTICE:** Not required (MIT, used only as a test
  oracle, not redistributed).
- **Files affected:** `tests/test_fuzz.py`.

### 3b. Python `difflib.SequenceMatcher`

- **URL:** Python standard library
- **License:** PSF (Apache-2.0 compatible)
- **What is used:** `tests/test_ratcliff_obershelp.py` calls
  `difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()` as the
  bit-exact reference for `sa.ratcliff_obershelp_similarity`. Invoked
  as a black box; the CPython `Lib/difflib.py` source was **not**
  read. An earlier design considered importing
  `SequenceMatcher.get_matching_blocks()` into the production path of
  `sa.partial_ratio` — this was rejected so that production code uses
  stride-align primitives only.
- **What is *not* used:** the CPython `Lib/difflib.py` source code.
- **Attribution / NOTICE:** Not required (PSF, stdlib).
- **Files affected:** `tests/test_ratcliff_obershelp.py`.

### 3c. jellyfish

- **URL:** https://github.com/jamesturk/jellyfish
- **License:** BSD-2-Clause (Apache-2.0 compatible)
- **What is used:** Pre-existing Soundex / Metaphone test oracle, no
  Phase D use. Listed here for completeness.
- **Files affected:** None for Phase D.

### 3d. CPython stdlib `math`, `random`, `statistics`

- **License:** PSF (Apache-2.0 compatible)
- **What is used:** Standard-library numeric helpers in tests (e.g.
  `math.sqrt` to hand-verify the cosine fixture in
  `tests/test_ngram.py`). Behaviour, not source.
- **Files affected:** All `tests/test_*.py` that compute pinned
  expected values.

## 4. Explicitly excluded sources

The following are excluded from contributing **any** source code,
data, comments, generator scripts, test fixtures, or expected-output
vectors to any Phase D file.

### 4a. FuzzyWuzzy

- **URL:** https://github.com/seatgeek/fuzzywuzzy
- **License:** **GPL-2.0**
- **Why excluded:** Apache-2.0 cannot incorporate GPL-licensed code.
  rapidfuzz is the MIT clean-room replacement for FuzzyWuzzy's API
  surface; stride-align uses rapidfuzz only as a runtime test oracle
  (see §3a), never reads FuzzyWuzzy source, fixtures, or test vectors.
- **Apache-2.0 compatible:** No.
- **Files affected:** None.

### 4b. Any other GPL/AGPL/LGPL fuzzy-matching, n-gram, LCS, Ratcliff-Obershelp, or Monge-Elkan port

- **Scope:** Every implementation distributed under GPL-1, GPL-2,
  GPL-3, AGPL, LGPL, or any later version or variant — whether
  existing today, dormant, or written after this document — is
  prohibited as a source on the same basis as the BMPM audit §3d.
- **Examples (not exhaustive):** abydos
  (`abydos.distance.{Jaccard,SorensenDice,Overlap,LCSseq,RatcliffObershelp,MongeElkan,…}`,
  GPL-3.0); any other GPL port discovered later.
- **What is used:** Nothing.
- **Files affected:** None.

### 4c. rapidfuzz internal source code

- **License:** MIT (compatible) — but **not read** by policy.
- **Why excluded from reading:** Keeps the parity claim honest:
  "stride-align matches rapidfuzz's *documented API behaviour*, derived
  from the formula", not "stride-align is a port of rapidfuzz".
- **Files affected:** None.

### 4d. CPython `Lib/difflib.py` internal source code

- **License:** PSF (compatible) — but **not read** by policy.
- **Why excluded from reading:** Same reasoning as §4c — the
  Ratcliff-Obershelp implementation is from the original Dr. Dobb's
  article, not a port of difflib's `SequenceMatcher` class.
- **Files affected:** None.

### 4e. textdistance

- **URL:** https://github.com/life4/textdistance
- **License:** MIT
- **Why excluded:** Not installed in this environment; no source was
  read. Documented here so the audit covers an obvious candidate
  oracle that **could** have been used and explicitly was not.
- **Files affected:** None.

## 5. stride-align repository structure

- **Location:** This repository.
- **License:** Apache-2.0 (same project).
- **What is reused as pattern:** Code-structure conventions — a
  single header for each kernel under `include/stride_align/`, a
  dispatcher in `src/cpp/*_dispatch.hpp`, registration in
  `src/cpp/module_bindings.hpp`, Python re-exports in
  `src/stride_align/__init__.py`, tests in `tests/test_*.py`.
- **Files affected:** All Phase D feature files.

## 6. Per-phase file inventory

What each phase added or modified, for traceability against this
audit.

### D.1 — n-gram set similarity

- New: `include/stride_align/ngram.hpp`
- New: `src/cpp/ngram_dispatch.hpp`
- New: `tests/test_ngram.py`
- Modified: `src/cpp/module_bindings.hpp`,
  `src/stride_align/__init__.py`, `README.md`, `CHANGELOG.md`

### D.3 — token-ratio family

- New: `src/stride_align/_fuzz.py` (pure-Python composition)
- New: `tests/test_fuzz.py`
- Modified: `src/stride_align/__init__.py`, `README.md`,
  `CHANGELOG.md`, `docs/TODO-character-similarity-phase-D.md`

### D.4 — LCS + LC Substring

- New: `include/stride_align/lcs.hpp`
- New: `src/cpp/lcs_dispatch.hpp`
- New: `tests/test_lcs.py`
- Modified: `src/cpp/module_bindings.hpp`,
  `src/stride_align/__init__.py`, `README.md`, `CHANGELOG.md`

### D.5 — Ratcliff-Obershelp

- New: `include/stride_align/ratcliff_obershelp.hpp`
- New: `tests/test_ratcliff_obershelp.py`
- Modified: `src/cpp/module_bindings.hpp`,
  `src/cpp/lcs_dispatch.hpp` (shared `widen_to_codepoints`),
  `src/stride_align/__init__.py`, `README.md`, `CHANGELOG.md`

### D.6 — Monge-Elkan

- New: `src/stride_align/_monge_elkan.py` (pure-Python composition)
- New: `tests/test_monge_elkan.py`
- Modified: `src/stride_align/__init__.py`, `README.md`,
  `CHANGELOG.md`, `docs/TODO-character-similarity-phase-D.md`

## Decisions implied by this audit

1. **No GPL contamination.** No GPL/AGPL/LGPL source touched any
   Phase D production or test file. FuzzyWuzzy is the largest named
   exclusion; the rapidfuzz MIT replacement serves as the test oracle.
2. **No third-party code in the production path.** Phase D's Python
   compositions (D.3 and D.6) import only `stride_align` itself
   (Apache-2.0). The bit-exact rapidfuzz parity claims in D.3 are
   honoured by `sa.indel_normalized_score` being algebraically equal
   to `rapidfuzz.fuzz.ratio / 100` — verified by formula, not by
   reading rapidfuzz internals.
3. **Test oracles are black boxes.** rapidfuzz, difflib, jellyfish are
   invoked through their public APIs only. None of their source code
   was read for the Phase D work.
4. **Algorithm references are descriptions, not code.** The classic
   papers and textbook formulations cited in §2 are the implementation
   basis; no specific implementation served as a starting point.
5. **NOTICE unchanged for Phase D.** None of the runtime dependencies
   added or used by Phase D is redistributed in the wheel, and none
   has an attribution clause that activates at the test-only level
   used here.
