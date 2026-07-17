# Phonetic encoder external-source audit (Phase D.2 + D.7)

Inventory of every external source that contributes (or is explicitly
excluded from contributing) code, data, structure, or test oracles
to the phonetic-encoder family shipped in `stride-align`:

| API | Algorithm |
| --- | --- |
| `sa.soundex` / `sa.soundex_equal` | American Soundex (Russell & Odell, 1918) |
| `sa.metaphone` / `sa.metaphone_equal` | Metaphone (Philips, 1990) |
| `sa.double_metaphone` | Double Metaphone (Philips, 2000) |
| `sa.nysiis` / `sa.nysiis_equal` | NYSIIS (Taft, 1970) |
| `sa.caverphone` | Caverphone 2.0 (Hood, 2004) |
| `sa.match_rating_codex` / `sa.match_rating_compare` | Match Rating Approach (Moore, 1977) |
| `sa.cologne_phonetic` | Kölner Phonetik (Postel, 1969) |

Beider-Morse Phonetic Matching (`sa.beider_morse`) and Daitch-Mokotoff
Soundex (`sa.daitch_mokotoff`) are large enough — and use vendored rule
data — that they have their own audit in
[`docs/bmpm-external-sources.md`](bmpm-external-sources.md).

The non-phonetic Phase D items (D.1 n-gram set similarity, D.3
token-ratio family, D.4 LCS / LC Substring, D.5 Ratcliff-Obershelp,
D.6 Monge-Elkan) are audited in
[`docs/phase-D-external-sources.md`](phase-D-external-sources.md).

The project ships under Apache-2.0; every source actually used must
be compatible.

## Posture, in one paragraph

Every encoder in the native `stride_align` API is an **original C++ implementation** written
from the published algorithm description and the Apache Commons Codec
class as a structural reference. No third-party source code is
copied or paraphrased into that native production path. The C++ namespace
boundaries (e.g. `stride_align::phonetic::soundex_detail`) and the
control-flow structure (cursor advance, rule cascade) follow the
corresponding Apache Commons Codec class so that cross-checking
against the Java reference stays tractable. Comments are paraphrased,
not copied. The separate `stride_align.jellyfish` migration facade
contains MIT-licensed compatibility variants adapted from Jellyfish
1.2.1; §3a and `NOTICE` record that provenance explicitly.

## 1. Apache Commons Codec

- **URL:** https://commons.apache.org/proper/commons-codec/
- **Git:** https://github.com/apache/commons-codec —
  `src/main/java/org/apache/commons/codec/language/{Soundex,Metaphone,DoubleMetaphone,Nysiis,Caverphone2,MatchRatingApproachEncoder,ColognePhonetic}.java`
- **License:** Apache-2.0
- **What is used:**
  - **Algorithmic structure.** Each stride-align encoder mirrors the
    method boundaries and rule-cascade ordering of the corresponding
    Apache Commons Codec class. Re-implementation in C++, not a
    line-by-line copy.
  - **API behaviour.** `soundex` length-4 padding, Metaphone's
    published-spec rule branch (CH-after-S → K, terminal GH silent),
    Double Metaphone's `(primary, alternate)` tuple shape, NYSIIS's
    leading-letter rewrites and 6-character truncation, Caverphone 2's
    fixed-length 10-character padding, Match Rating Approach's
    consonant-skeleton codex and codex-difference comparator, Cologne
    Phonetic's context-sensitive C/X/D/T/P branching and adjacency-
    collapse step.
- **Attribution / NOTICE:** Required. The `NOTICE` file under
  "Phonetic encoder ports" lists each header / `*Reference:*` pair.
- **Apache-2.0 compatible:** Yes (same licence).
- **Files affected:**
  - `include/stride_align/soundex.hpp`
  - `include/stride_align/metaphone.hpp`
  - `include/stride_align/double_metaphone.hpp`
  - `include/stride_align/nysiis.hpp`
  - `include/stride_align/caverphone.hpp`
  - `include/stride_align/match_rating.hpp`
  - `include/stride_align/cologne_phonetic.hpp`
  - `NOTICE` (attribution bullets)

## 2. Original algorithm publications

Each encoder originates in a published paper; the C++ implementation
follows the description in those papers. No code is copied — these
references are algorithmic descriptions, not licensed implementations.

| Encoder | Publication |
| --- | --- |
| Soundex | Russell, R. C.; Odell, M. K., U.S. Patent 1,261,167 (1918); codified by the U.S. Census Bureau. |
| Metaphone | Philips, L. "Hanging on the Metaphone." *Computer Language Magazine* 7(12), December 1990, pp. 39-44. |
| Double Metaphone | Philips, L. "The Double Metaphone Search Algorithm." *C/C++ Users Journal*, June 2000. |
| NYSIIS | Taft, R. L. "Name Search Techniques." New York State Identification and Intelligence System, 1970. |
| Caverphone 2.0 | Hood, D. "Caverphone Revisited." Technical Report, University of Otago, 2004. |
| Match Rating Approach | Moore, G. (Western Airlines), 1977. |
| Kölner Phonetik | Postel, H.-J. "Die Kölner Phonetik." IBM Nachrichten 19, 1969, pp. 925-931. |

## 3. Third-party compatibility references and test oracles

The libraries below are listed in the `phonetic-compat` pyproject
extra. Native phonetic tests remain hand-pinned and do not import an
oracle. The Jellyfish facade has an additional optional differential
battery because exact third-party API compatibility is its purpose.

### 3a. jellyfish

- **URL:** https://codeberg.org/jpt/jellyfish
- **License:** MIT
- **What is used:** The rule ordering and boundary behaviour of
  Jellyfish 1.2.1's Soundex, Metaphone, NYSIIS, Match Rating,
  non-overlapping-chunk Jaccard, and long-tolerance Jaro-Winkler
  implementations are adapted in `src/stride_align/jellyfish.py`.
  Native distance/Jaro kernels remain original stride-align code.
  `tests/test_jellyfish_shim.py` optionally imports the installed
  package as a differential oracle.
- **Attribution / NOTICE:** The full upstream MIT notice is retained
  in `NOTICE`, and the derived module carries an attribution header.
- **Apache-2.0 compatible:** Yes.
- **Files affected:** `src/stride_align/jellyfish.py`,
  `tests/test_jellyfish_shim.py`, `docs/api/jellyfish-shim.md`,
  `NOTICE`.

### 3b. `metaphone` PyPI package (oubiwann/metaphone)

- **URL:** https://github.com/oubiwann/metaphone
- **License:** BSD
- **What is used:** Listed in `phonetic-compat` as an oracle and
  mentioned in `double_metaphone.hpp` comments
  to document the known `DoubleMetaphoneVariant.PYTHON` branch
  (a known bug in the `metaphone` package's GH-after-vowel handler
  that produces `"Hugh"` → `"HH"` where Apache Commons Codec and the
  faithful Lawrence Philips spec produce `"H"`). Not imported by
  stride-align's tests.
- **Attribution / NOTICE:** Mentioned as an optional test oracle in
  `NOTICE`; no source is distributed.
- **Apache-2.0 compatible:** Yes.
- **Files affected:** None at test time; mentioned in comments only.

### 3c. pyphonetics

- **URL:** https://github.com/Lilykos/pyphonetics
- **License:** MIT (per trove classifier; the package's `License`
  metadata field reads `UNKNOWN` but the OSI classifier is
  authoritative)
- **What is used:** Listed in `phonetic-compat` extras for downstream
  cross-checks. Not imported by stride-align's tests.
- **Attribution / NOTICE:** Mentioned as an optional test oracle in
  `NOTICE`; no source is distributed.
- **Apache-2.0 compatible:** Yes.
- **Files affected:** None at test time.

### 3d. Apache Commons Codec canonical test vectors

- **URL:** Same git repo as source §1 —
  `src/test/java/org/apache/commons/codec/language/*Test.java`
- **License:** Apache-2.0
- **What is used:** Small named fixtures (Robert / Rupert /
  Schmidt / Christopher / Watkins / Stevenson / Wikipedia /
  Breschnew / Müller, etc.) translated into Python literals in
  `tests/test_phonetic.py`. Each fixture is paired with a comment
  naming the algorithm and the variant settings it pins. Per the
  project's "tests are documentation" rule, no bulk import of
  upstream CSVs.
- **Attribution / NOTICE:** Covered by the encoder-port bullets in
  §1; no additional `NOTICE` line needed.
- **Apache-2.0 compatible:** Yes.
- **Files affected:** `tests/test_phonetic.py`.

## 4. Explicitly excluded sources

The following are excluded from contributing **any** source code,
data, comments, generator scripts, test fixtures, or expected-output
vectors to any phonetic-encoder file.

### 4a. abydos

- **URL:** https://github.com/chrislit/abydos
- **License:** **GPL-3.0-or-later**
- **Why excluded:** Apache-2.0 cannot incorporate GPL-licensed code.
  abydos has the largest published Python collection of phonetic
  encoders, and many of them duplicate the algorithms stride-align
  ships (Soundex variants, Metaphone, NYSIIS, Caverphone, MRA,
  Cologne). None of its code, comments, or test vectors have been
  read or copied. The abydos copy present in any local
  development `.venv` is for unrelated browsing only and is **never**
  imported from `tests/`, `tools/`, or any other distributable path.
- **Apache-2.0 compatible:** No.
- **Files affected:** None.

### 4b. Any other GPL/AGPL/LGPL phonetic-encoder port

- **Scope:** Every other phonetic-encoder implementation distributed
  under GPL-1, GPL-2, GPL-3, AGPL, LGPL, or any later version or
  variant — whether existing today, dormant, or written after this
  document — is prohibited as a source on the same basis as the BMPM
  audit §3d. The prohibition covers source code, rule tables,
  expected-output vectors, generator scripts, and any other
  distributable artifact derived from a GPL port — not only verbatim
  copies but also paraphrases, transliterations, and table
  conversions.
- **What is used:** Nothing.
- **Apache-2.0 compatible:** No.
- **Files affected:** None.

### 4c. Lawrence Philips' BASIC listing for Metaphone (1990)

- **URL:** Computer Language Magazine 7(12) printed listing.
- **License:** Unclear; the magazine code listing was never released
  under an OSI-approved licence. Michael Kuhn's 1991 C port
  documented discrepancies between the BASIC code and the verbal
  description.
- **What is used:** Nothing directly. stride-align's Metaphone follows
  the **verbal description** branch of the spec (the branch Apache
  Commons Codec also follows), not the printed BASIC. The header
  comment notes Kuhn's caveat as historical context.
- **Apache-2.0 compatible:** N/A — not used directly.
- **Files affected:** None.

## 5. stride-align repository structure

- **Location:** This repository.
- **License:** Apache-2.0 (same project).
- **What is reused as pattern:** Code-structure conventions — a
  single header for each encoder under `include/stride_align/`,
  shared dispatch in `src/cpp/soundex_dispatch.hpp`, registration in
  `src/cpp/module_bindings.hpp`, Python re-exports in
  `src/stride_align/__init__.py`, tests in `tests/test_phonetic.py`.
- **Files affected:** All phonetic-encoder feature files.

## 6. Per-encoder file inventory

| Encoder | Header | Dispatch | Tests | NOTICE bullet |
| --- | --- | --- | --- | --- |
| Soundex | `include/stride_align/soundex.hpp` | `src/cpp/soundex_dispatch.hpp` | `tests/test_phonetic.py` (`test_soundex_*`) | yes |
| Metaphone | `include/stride_align/metaphone.hpp` | `src/cpp/soundex_dispatch.hpp` | `test_metaphone_*` | yes |
| Double Metaphone | `include/stride_align/double_metaphone.hpp` | `src/cpp/soundex_dispatch.hpp` | `test_double_metaphone_*` | yes |
| NYSIIS | `include/stride_align/nysiis.hpp` | `src/cpp/soundex_dispatch.hpp` | `test_nysiis_*` | yes |
| Caverphone | `include/stride_align/caverphone.hpp` | `src/cpp/soundex_dispatch.hpp` | `test_caverphone_*` | yes |
| Match Rating | `include/stride_align/match_rating.hpp` | `src/cpp/soundex_dispatch.hpp` | `test_match_rating_*` | yes |
| Cologne Phonetic | `include/stride_align/cologne_phonetic.hpp` | `src/cpp/soundex_dispatch.hpp` | `test_cologne_phonetic_*` | yes |

The shared dispatch lives in `src/cpp/soundex_dispatch.hpp` regardless
of which encoder it serves — the name is historical (Soundex was the
first one added). Each encoder is registered in
`src/cpp/module_bindings.hpp` and re-exported in
`src/stride_align/__init__.py`.

## Decisions implied by this audit

1. **Single primary source.** Apache Commons Codec is the only
   external source whose algorithmic structure enters stride-align's
   phonetic-encoder family. Each port has a `NOTICE` bullet naming
   the corresponding Java class.
2. **No GPL contamination.** abydos and every other GPL/LGPL/AGPL
   phonetic-encoder port are excluded from code, comments, test
   fixtures, and oracle calls. See §4.
3. **Native tests do not import oracles.** Every test in
   `tests/test_phonetic.py` uses hand-pinned expected values. The
   work-alike facade is tested separately in
   `tests/test_jellyfish_shim.py`; its optional differential battery
   imports Jellyfish when the `phonetic-compat` extra is installed.
4. **Test fixtures stay small and named.** Canonical Apache Commons
   Codec test vectors are translated into Python literals one at a
   time, each paired with a comment naming the algorithm and variant
   it pins.
5. **Variants are explicit.** Where multiple widely-used
   implementations disagree on rule edge cases (Metaphone CH-after-S,
   Metaphone terminal-GH, Double Metaphone GH-after-vowel near the
   start), stride-align exposes an enum (`MetaphoneVariant`,
   `DoubleMetaphoneVariant`) and documents the divergence in the
   header comment. No silent picking.
