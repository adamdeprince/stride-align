# BMPM external-source audit

Inventory of every external source that would (or explicitly would not)
contribute code, data, structure, or comments to a Beider-Morse Phonetic
Matching (BMPM) implementation in `stride-align`. The project ships
under Apache-2.0; all sources actually used must be compatible.

## 1. Apache Commons Codec — `org.apache.commons.codec.language.bm.*`

- **URL:** https://commons.apache.org/proper/commons-codec/
- **Git:** https://github.com/apache/commons-codec —
  `src/main/java/org/apache/commons/codec/language/bm/` and
  `src/main/resources/org/apache/commons/codec/language/bm/`
- **License:** Apache-2.0
- **What is used:**
  - **Algorithm / code structure.** PhoneticEngine, Rule, Lang,
    Languages, NameType, RuleType, RulesApplication, PhonemeBuilder.
    The C++ port mirrors method boundaries and control flow so that
    cross-checking against the Java reference remains tractable; it is
    a re-implementation, not a line-by-line copy.
  - **API behavior.** Same `encode()` semantics. `NameType` =
    {`GENERIC`, `ASHKENAZI`, `SEPHARDIC`}. `RuleType` = {`APPROX`,
    `EXACT`}. `concat` flag. `maxPhonemes` cap (default 20).
  - **Rule tables.** Every `gen_*.txt`, `ash_*.txt`, `sep_*.txt`, plus
    `gen_lang.txt` / `gen_languages.txt` and the Ashkenazi / Sephardic
    equivalents. Vendored unmodified into
    `docs/upstream/commons-codec-bmpm/` with their original ASF headers
    intact, then transformed into C++ `constexpr` tables by a generator
    script. Both the vendored text and the generated header are
    derivative works of the upstream resource files.
  - **Comments.** Paraphrased only; no verbatim copy of upstream
    JavaDoc.
- **Attribution / NOTICE:** Required. A bullet is added to
  `NOTICE` parallel to the existing entries
  (`include/stride_align/double_metaphone.hpp` etc.) naming the
  upstream Java classes and the vendored resource directory.
  Apache-2.0 §4 obliges preservation.
- **Apache-2.0 compatible:** Yes (same licence).
- **Files affected:**
  - New: `include/stride_align/beider_morse.hpp`
  - New: `docs/upstream/commons-codec-bmpm/*.txt` (vendored verbatim)
  - New: `tools/generate_bmpm_tables.py`
  - New: `src/cpp/beider_morse_tables.hpp` (generator output, checked in)
  - Modified: `NOTICE`
  - Modified: `src/cpp/soundex_dispatch.hpp`
  - Modified: `src/cpp/module_bindings.hpp`
  - Modified: `src/stride_align/__init__.py`
  - Modified: `README.md` (phonetic-encoders section + BMPM example)
  - Modified: `CMakeLists.txt` (if the generated header needs listing)

## 2. Original Beider-Morse PHP/JavaScript reference

- **URL:** http://stevemorse.org/phonetics/bmpm.htm
  (Alexander Beider and Stephen P. Morse)
- **License:** Unclear. No OSI-approved licence file is published on
  that page; redistribution has been informally permitted by the
  authors but never under a formal licence stride-align can audit.
- **What is used:** Nothing directly. Apache Commons Codec already
  absorbed the algorithm under Apache-2.0; stride-align's port descends
  from the Commons Codec layer and never touches stevemorse.org files.
- **Attribution / NOTICE:** Not a licence obligation here. A scholarly
  acknowledgement of Beider & Morse in the algorithm header comment is
  good practice and is included for that reason only.
- **Apache-2.0 compatible:** N/A — not used directly, so its licence
  posture does not reach our distribution.
- **Files affected:** None. Mentioned only in the algorithm header
  comment as historical credit.

## 3. Sources excluded — every GPL-licensed BMPM port

Every BMPM port distributed under the GNU General Public License (any
version) is prohibited as a source of code, data, structure, tests,
comments, or test oracles. GPL is one-way incompatible with Apache-2.0
for incorporation: pulling any non-trivial chunk from a GPL port into
this codebase, a generator, or a build artifact (including test
fixtures or rule constants shipped with the wheel) would taint the
wheel under GPL obligations. The prohibition covers source code, rule
tables, expected-output vectors, generator scripts, and any other
distributable artifact derived from a GPL port — not only verbatim
copies but also paraphrases, transliterations, and table conversions.

The named ports below are known instances; this list is not
exhaustive. **Any other GPL-licensed BMPM implementation, port,
fork, or derivative — including ones not yet written — is excluded
on the same basis.**

### 3a. abydos — `abydos.phonetic._beider_morse{,_data}`

- **URL:** https://github.com/chrislit/abydos
- **License:** GPL-3.0-or-later (file header: `Abydos is free
  software: you can redistribute it and/or modify it under the terms
  of the GNU General Public License...`).
- **What is used:** Nothing. The abydos copy installed in the
  project's `.venv` is for unrelated reading only and is never
  imported from `tests/`, `tools/`, or any other distributable path.
- **Attribution / NOTICE:** N/A (not used).
- **Apache-2.0 compatible:** No.
- **Files affected:** None.

### 3b. chrislit/bmpm — PHP reference port

- **URL:** https://github.com/chrislit/bmpm
- **License:** GPL-3.0. A PHP-language tracking copy of the
  stevemorse.org reference, maintained as the upstream that abydos
  ported from.
- **What is used:** Nothing.
- **Attribution / NOTICE:** N/A (not used).
- **Apache-2.0 compatible:** No.
- **Files affected:** None.

### 3c. Haran/BMDMSoundex — combined BMPM + Daitch-Mokotoff port

- **URL:** https://github.com/Haran/BMDMSoundex
  (project page: https://haran.github.io/BMDMSoundex/)
- **License:** GPL-3.0. Combines BMPM and Daitch-Mokotoff Soundex
  in one distribution.
- **What is used:** Nothing — for either BMPM or Daitch-Mokotoff.
  The Daitch-Mokotoff rule data in stride-align comes solely from
  the Apache Commons Codec `dmrules.txt` already vendored in
  `docs/upstream/apache-commons-codec-dmrules.txt`.
- **Attribution / NOTICE:** N/A (not used).
- **Apache-2.0 compatible:** No.
- **Files affected:** None.

### 3d. Any other GPL BMPM port

- **Scope:** Every other BMPM implementation distributed under GPL-1,
  GPL-2, GPL-3, AGPL, LGPL, or any later version or variant — whether
  existing today, dormant, or written after this document — is
  prohibited as a source on the same basis as §3a–§3c.
- **What is used:** Nothing.
- **Apache-2.0 compatible:** No.
- **Files affected:** None.

## 4. jellyfish

- **URL:** https://github.com/jamesturk/jellyfish
- **License:** BSD-2-Clause
- **What is used:** Not used for BMPM. jellyfish has no BMPM
  implementation. It remains a Soundex / Metaphone test oracle in
  `tests/test_phonetic.py`; no change for this work.
- **Attribution / NOTICE:** None for BMPM.
- **Apache-2.0 compatible:** Yes.
- **Files affected:** None.

## 5. Apache Commons Codec test vectors

- **URL:** Same git repo as source #1 —
  `src/test/java/org/apache/commons/codec/language/bm/BeiderMorseEncoderTest.java`
  and adjacent fixture files.
- **License:** Apache-2.0
- **What is used:** A small set of fixed input → expected-output pairs
  used as test fixtures (e.g. the canonical Schmidt / Goldman / Renault
  examples), translated into Python literals in
  `tests/test_beider_morse.py`. Per the project's
  "tests are documentation" rule, each fixture is paired with a
  comment naming the `NameType` / `RuleType` / `concat` settings it
  pins.
- **Attribution / NOTICE:** Covered by the BMPM bullet under source
  #1; no additional NOTICE line needed.
- **Apache-2.0 compatible:** Yes.
- **Files affected:**
  - New: `tests/test_beider_morse.py`

## 6. stride-align's existing phonetic-encoder layout

- **URL / location:** This repository.
- **License:** Apache-2.0 (same project).
- **What is used:** Code-structure pattern only — single header in
  `include/stride_align/`, dispatcher wrapper in
  `src/cpp/soundex_dispatch.hpp`, registration in
  `src/cpp/module_bindings.hpp`, Python re-export in
  `src/stride_align/__init__.py`. Same pattern as
  `cologne_phonetic`, `double_metaphone`, etc.
- **Attribution / NOTICE:** None (internal).
- **Apache-2.0 compatible:** Yes.
- **Files affected:** Listed under source #1.

## 7. CPython and nanobind

- **URL:** https://www.python.org/, https://github.com/wjakob/nanobind
- **License:** PSF (Python) / BSD-3-Clause (nanobind)
- **What is used:** API behaviour only — `nb::handle`, `nb::arg`,
  `nb::make_tuple`, the same idioms used throughout
  `module_bindings.hpp`. No copying.
- **Attribution / NOTICE:** None beyond the existing build / link
  dependency.
- **Apache-2.0 compatible:** Yes.
- **Files affected:** None new beyond the registration line.

## Decisions implied by this audit

1. **Single primary source.** Apache Commons Codec is the only
   external source whose code or data enters stride-align. Licence
   bookkeeping reduces to one `NOTICE` bullet.
2. **No GPL contamination.** Every GPL-licensed BMPM port is
   excluded from code, data, generators, and test oracles — abydos
   (Python), chrislit/bmpm (PHP), Haran/BMDMSoundex (combined
   BMPM + DM), and any other existing or future GPL/LGPL/AGPL port.
   The abydos copy in `.venv` is never imported from `tests/` or
   `tools/`. See §3.
3. **Vendor raw resource files unmodified.** Each upstream `.txt`
   rule file is dropped into `docs/upstream/commons-codec-bmpm/` with
   its original ASF header preserved. The generator's output is then
   clearly a derivative of those preserved files rather than a
   provenance-stripped copy.
4. **Generator script, not hand-conversion.** `tools/generate_bmpm_tables.py`
   parses the vendored `.txt` files into `constexpr` tables in a
   checked-in generated header. The generated header carries an
   "auto-generated — do not edit by hand" banner plus the ASF
   licence preamble.
5. **Test fixtures stay small and named.** No bulk import of upstream
   CSVs; each fixture is a named constant tied to a documented Commons
   Codec test case.
