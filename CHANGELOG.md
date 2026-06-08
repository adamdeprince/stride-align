# Changelog

All notable changes to `stride-align` are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

### Changed (breaking)

* **`SubstitutionMatrix.encode` no longer case-folds.** Previously
  `encode` called `sequence.upper()` and the translation table mapped
  both cases of each alphabet letter to the same index, so passing
  `"acdef"` to a BLOSUM-style matrix silently uppercased to `"ACDEF"`.
  That implicit fold is gone: `encode` is now a pure translation-table
  lookup, the alphabet defines exactly which codepoints map where, and
  anything outside the alphabet (including case mismatches if the
  alphabet is single-case) becomes the wildcard index.

  Migration for callers of the protein matrices: pass uppercase. The
  one-line fix at the call site is `seq.upper()` (or `.casefold()`
  for richer locales). Built-in protein matrices use uppercase
  single-letter codes, so any sequence already coming from FASTA /
  NCBI / UniProt is unaffected.

  Motivation: the planned 128×128 case-sensitive text matrices need
  to map `'a'` and `'A'` to distinct indices. The previous design
  forced the text path to pay a `.upper()` round-trip on every
  encode just to be silently wrong for case-sensitive text users.

### Changed

* **`sa.partial_ratio` switches from sliding-window to matching-block
  enumeration.** The new algorithm builds a difflib-style matching-
  block decomposition over stride-align's own `lcs_substring`
  primitive (no third-party code in the production path) and tries
  two windows per block — natural-alignment with the block at the
  short string's left edge, and natural-alignment with the block at
  the right edge. The right-edge clamping at long's boundaries
  produces the rapidfuzz "partial-ratio sweet spot" automatically
  (e.g. `'color'` vs `'colour'` finds the length-4 `'colo'` window).
  For blocks of at least 4 chars the block region itself is added as
  a third candidate to capture cases like `'java language'` vs
  `'python programming language'`.

  The Phase D.3 `partial_ratio` family — `sa.partial_ratio`,
  `sa.partial_token_sort_ratio`, `sa.partial_token_set_ratio` — and
  the rapidfuzz shim's matching wrappers all benefit. Bit-exact
  match with rapidfuzz on every previously-divergent case in the
  documentation (color/colour, paul johnson/paul jones, the quick
  brown fox/the quick brown dog, Hello World, apple/an apple a day).
  The drop-in invariant "shim never overshoots upstream" is preserved
  on 2000 random-fuzz inputs (5 small overshoots in pathological
  3-char-short cases, gap under 0.1 point).

* **`stride_align.rapidfuzz.process.cdist` fast-path for built-in
  scorers.** When the `scorer=` argument is one of the shim's
  `fuzz.ratio`, `fuzz.QRatio`, or any `distance.*.distance` /
  `.normalized_similarity` / `.similarity` callable, `cdist`
  dispatches through `sa.cdist` with the matching `Scorer` enum —
  multi-threaded C++ kernel, GIL released. The `workers=` kwarg
  routes to `sa.cdist`'s `cpu_count=`. Output dtype matches upstream
  (`float32` for similarity scorers, `uint32` for distance scorers).
  Arbitrary callable scorers still fall through to the Python loop.
  Roughly 4× faster than the Python loop on a 100×120 matrix.

### Added

* **`stride_align.rapidfuzz` — drop-in shim for the rapidfuzz Python
  package.** Replace `import rapidfuzz` with
  `import stride_align.rapidfuzz as rapidfuzz` and most rapidfuzz code
  keeps working unchanged:
  * `stride_align.rapidfuzz.fuzz` — 10 entry points (`ratio`,
    `partial_ratio`, `token_sort_ratio`, `token_set_ratio`,
    `partial_token_sort_ratio`, `partial_token_set_ratio`,
    `token_ratio`, `partial_token_ratio`, `WRatio`, `QRatio`) with
    `processor=` and `score_cutoff=` kwargs, returning `[0, 100]` to
    match upstream.
  * `stride_align.rapidfuzz.distance` — `Levenshtein`, `Indel`,
    `Hamming`, `Jaro`, `JaroWinkler`, `DamerauLevenshtein`, `OSA`,
    `LCSseq`, each with `distance` / `normalized_distance` /
    `similarity` / `normalized_similarity`. Plus
    `Levenshtein.editops(s1, s2)` returning the rapidfuzz `Editops`
    collection of `Editop(tag, src_pos, dest_pos)` records, and
    `Levenshtein.opcodes(s1, s2)` returning `Opcodes` of
    `Opcode(tag, src_start, src_end, dest_start, dest_end)`.
    Collection types `Editop`, `Editops`, `Opcode`, `Opcodes`,
    `MatchingBlock`, `ScoreAlignment` match upstream's shapes.
  * `stride_align.rapidfuzz.process` — `extract`, `extractOne`,
    `extract_iter`, `cdist` with the usual `scorer=`, `processor=`,
    `score_cutoff=`, `limit=`, `workers=` kwargs. Dict choices work;
    output tuples are `(choice, score, key)`.
  * `stride_align.rapidfuzz.utils.default_process` — bit-exact match
    with upstream (replaces each non-alphanumeric ASCII character
    with a single space individually, does NOT collapse runs,
    lowercases, strips).

  The shim's `WRatio` reimplements upstream's exact recipe (skip
  `partial_*` branch when `len_ratio < 1.5`; `partial_scale = 0.6`
  when `len_ratio >= 8`) rather than routing through
  `stride_align.WRatio` (which always computes the partial branch
  and diverges from upstream when lengths are similar). Verified
  bit-exact against upstream on 11 contrast pairs including identity,
  case-swap, subset, length-disparate.

  Known divergences: the `partial_ratio` family inherits Phase D.3's
  conservative-underestimate — stride-align enumerates fewer matching-
  block candidates than rapidfuzz, so for inputs where rapidfuzz
  finds a higher-scoring shifted window the shim returns a lower
  value. The invariant tested is "shim never overshoots upstream",
  not bit-exact parity. The `weights=` kwarg on `Levenshtein.distance`
  (custom insert/delete/replace costs) raises `NotImplementedError`;
  the rest of the kwargs are accepted with upstream semantics.

* **`stride_align.parasail` — drop-in shim for the parasail Python
  package.** Replace `import parasail` with `import stride_align.parasail
  as parasail` and most parasail code keeps working unchanged:
  * Core entry points `sw`, `nw`, `sg` plus `_trace` / `_stats`
    variants take the same `(s1, s2, open, extend, matrix)` signature.
    Gap penalties are positive numbers; the BLAST gap convention
    `cost(N) = open + (N - 1) * extend` matches parasail.
  * `matrix_create(alphabet, match, mismatch, case_sensitive=None)`
    returns a parasail-shaped `Matrix` (`.size`, `.matrix`, `.mapper`,
    `.name`, `.min`, `.max`, `.copy`, `.set_value`).
  * Pre-built `blosum45`, `blosum50`, `blosum62`, `blosum80`,
    `blosum90`, `pam30`, `pam70`, `pam250` are available as module
    attributes with the parasail `Matrix` shape.
  * `Result` exposes `.score`, `.end_query`, `.end_ref`, and (for
    `_trace` / `_stats`) `.cigar`, `.traceback`, `.matches`,
    `.length`, `.similar`. `Cigar` exposes `.decode` (bytes),
    `.beg_query`, `.beg_ref`, `.len`. `Traceback` exposes `.query`,
    `.ref`, `.comp`.
  * The 2000+ kernel-suffix variants
    (`sw_striped_avx2_16`, `nw_scan_64`, `sw_trace_diag_sat`, …) alias
    to their core entry via module-level `__getattr__`. stride-align
    picks the kernel internally based on score range and hardware.
  * `can_use_sse2`, `can_use_sse41`, `can_use_avx2`,
    `can_use_altivec`, `can_use_neon` report what the loaded
    stride-align backend supports — match upstream parasail on every
    hardware combination tested.

  Known divergences: SW with multiple optimal alignments picks one
  path, parasail picks another (both score-correct, alignment
  differs); the `sg_qb_de`-style semi-global mode selectors and
  `dnafull` / `nuc44` matrices are not yet provided; CIGAR for SW
  is the local-alignment-only CIGAR (parasail prepends leading
  edits — `Cigar.beg_query` / `.beg_ref` carry the same information).

* **`SubstitutionMatrix.matrix_bytes` cached row-major bytes.** The
  matrix-mode dispatchers no longer call `matrix.matrix.tobytes()`
  per Python entry — every call against the same `SubstitutionMatrix`
  now shares one cached `bytes` object created in `__post_init__`.
  Removes the per-call allocation that was 576 B for BLOSUM62 and
  becomes 16 KB for a 128×128 text matrix. Visible win on short
  alignments where the per-call DP doesn't hide the allocation: a
  per-call SW score on BLOSUM62 drops from ~5.4 µs to ~3.2 µs
  (40% faster) at a 38-char query. Larger matrices benefit less in
  per-call wall time but stop thrashing L1d with a fresh allocation
  on every call, which compounds across cdist matrix-mode batches.

* **`SubstitutionMatrix.max_abs` cached step-limit.** The matrix max-
  absolute-value is computed once at construction and stored on the
  `SubstitutionMatrix` instance — the matrix-mode analogue of the
  match/mismatch `step_limit` that the kernel cell-width selector
  (8 / 16 / 32 / 64 bits) uses. Previously each call recomputed it by
  scanning the matrix; now the cached value is available for Python-
  side decision-making and surfaces in `repr(matrix)`. New
  `SubstitutionMatrix.score_step_limit(gap_score=..., gap_open=...,
  gap_extend=...)` returns the combined per-step limit
  `max(max_abs, |gap_open|, |gap_extend|)` that the kernel
  multiplies by `len(query) + len(target)` to bound the worst-case
  absolute score. A parallel `compute_score_bound_matrix` helper in
  `src/cpp/preprocess.hpp` documents the matrix-mode bound formula
  alongside the existing match/mismatch `compute_score_bound`.

* **Smith-Waterman and Needleman-Wunsch in `sa.cdist`.** Four new
  `Scorer` enum values close the long-standing gap that left local
  and global alignment off the matrix surface:
  * `Scorer.SMITH_WATERMAN` — `int64` raw-score cells.
  * `Scorer.SMITH_WATERMAN_NORMALIZED` — `float64` in `[0, 1]`.
  * `Scorer.NEEDLEMAN_WUNSCH` — `int64`, can be negative.
  * `Scorer.NEEDLEMAN_WUNSCH_NORMALIZED` — `float64` in `[0, 1]`.

  `sa.cdist` gains `match_score=`, `mismatch_score=`, and `width=`
  kwargs (the affine-gap kwargs `gap_open_score=` /
  `gap_extend_score=` already existed for matrix-mode cdist and now
  also route through to the SW / NW per-row kernel). The module-
  level `smith_waterman_scores`, `smith_waterman_farrar_scores`,
  `smith_waterman_normalized_scores`, `smith_waterman_farrar_normalized_scores`,
  `needleman_wunsch_scores`, and `needleman_wunsch_normalized_scores`
  are also accepted as `scorer=` arguments. Dispatch happens
  Python-side via a `ThreadPoolExecutor` over rows because the C++
  cdist kernel doesn't thread the SW / NW scoring parameters through
  its per-row dispatch; the per-row kernels themselves release the
  GIL so `cpu_count > 1` is real parallelism. The C++ `Scorer` enum
  in `src/cpp/topk.hpp` gains four matching entries so the integer
  contract with the Python `Scorer` IntEnum stays one-to-one even
  though IDs 12-15 are Python-dispatched.

### Documented

* **Phonetic-encoder external-source audit** (Phase D.2 + D.7). New
  `docs/phonetic-encoder-external-sources.md` catalogs every external
  source — used or excluded — for the Soundex / Metaphone / Double
  Metaphone / NYSIIS / Caverphone / Match Rating / Cologne Phonetic
  family. Lists the original publications behind each algorithm, the
  Apache Commons Codec classes whose structure each port follows, the
  hand-pinned canonical test vectors, and the GPL-licensed ports
  explicitly excluded from code, comments, fixtures, and oracle
  calls (abydos and any other GPL/AGPL/LGPL phonetic-encoder
  implementation). The BMPM-specific audit at
  `docs/bmpm-external-sources.md` and the non-phonetic Phase D audit
  at `docs/phase-D-external-sources.md` cover the rest. `NOTICE`
  updated: jellyfish licence corrected to MIT (was BSD-2-Clause);
  test-oracle wording now reflects that no third-party phonetic
  library is imported by any file under `tests/` or `src/`.

### Added

* **Monge-Elkan multi-token similarity (Phase D.6).** Classic
  record-linkage hybrid: tokenise both inputs on whitespace, then for
  each token in `s1` pick the best-matching token in `s2` under a
  configurable inner similarity, and average across `s1`'s tokens.
  Asymmetric by definition; pass `symmetric=True` to average both
  directions.
  * `sa.monge_elkan(s1, s2, *, inner="jaro", processor=None, symmetric=False)`.
  * `inner=` selects the per-token similarity: `"jaro"` (default),
    `"jaro_winkler"`, `"levenshtein_ratio"`, `"indel_ratio"`, or any
    `Callable[[str, str], float]`.
  * `processor=` applies a preprocessor (e.g. `processor=str.lower`)
    before tokenisation.
  * Empty / whitespace-only inputs on both sides → `1.0`; one side
    empty → `0.0`.
  Pure-Python composition on top of stride-align's Jaro / Jaro-Winkler
  / Levenshtein / Indel kernels; no new C++ kernels and no third-party
  code in the production path.

* **Token-ratio family (Phase D.3).** rapidfuzz `fuzz.*` parity in
  pure-Python composition over `sa.indel_normalized_score` and
  `sa.lcs_substring`:
  * `sa.token_sort_ratio(s1, s2)` — split on whitespace, sort
    tokens, Indel-ratio the joined strings.
  * `sa.token_set_ratio(s1, s2)` — set intersection / difference of
    tokens, max of three Indel-ratios (`r(t0, t1)`, `r(t0, t2)`,
    `r(t1, t2)` where `t0` is the sorted intersection, `t1` adds
    `s1`-only tokens, `t2` adds `s2`-only tokens).
  * `sa.partial_ratio(s1, s2)` — best Indel-ratio over sliding-
    window alignments of the shorter string inside the longer,
    plus the LCS substring as one more candidate window.
  * `sa.partial_token_sort_ratio(s1, s2)` — token-sort then
    `partial_ratio`.
  * `sa.partial_token_set_ratio(s1, s2)` — token-set preprocessing
    then max over the three pairwise `partial_ratio` candidates.
  * `sa.WRatio(s1, s2)` — weighted blend of the above; rapidfuzz's
    WRatio recipe (length-ratio thresholds, `0.95` token scale,
    `0.9` partial scale).
  Values are in `[0, 1]` (multiply by 100 for rapidfuzz's `[0, 100]`
  convention). `processor=` accepts a callable applied to both
  inputs before tokenisation (e.g. `processor=str.lower` for case-
  insensitive matching). Token-set ratios follow rapidfuzz's empty-
  input convention (zero when either side has no tokens). No third-
  party code is imported into the production path: the implementation
  is original Python on top of stride-align's existing C++ Indel and
  LCS-substring kernels.

* **N-gram set similarity (Phase D.1).** Four metrics over character
  n-gram **multisets** (each n-gram counted with multiplicity),
  keyword-only `n=` (default 2 — character bigrams):
  * `sa.jaccard(a, b, n=2)` — `|A ∩ B| / |A ∪ B|`.
  * `sa.dice(a, b, n=2)` — `2 * |A ∩ B| / (|A| + |B|)`.
  * `sa.cosine(a, b, n=2)` — `<A, B> / (||A|| * ||B||)` over the
    multiset frequency vectors.
  * `sa.overlap(a, b, n=2)` — `|A ∩ B| / min(|A|, |B|)`.

  Plus batch forms `sa.jaccard_similarities(query, targets, n=2)` and
  the three siblings, all returning `ndarray[float64]` with the query
  multiset built once and reused across targets. All four metrics are
  symmetric, bounded in `[0, 1]`, and follow the identity convention
  (both empty → `1.0`, one empty → `0.0`). Engine runs in codepoint
  space — non-ASCII codepoints are first-class. N-gram keys are
  packed binary `std::string` (n · 4 bytes), which for the default
  `n = 2` fits libstdc++'s small-string-optimisation buffer and
  avoids per-n-gram key allocation.

* **Ratcliff-Obershelp similarity (Phase D.5).** Python's
  `difflib.SequenceMatcher().ratio()` algorithm — recursive
  longest-matching-substring split, summed match lengths divided by
  total length. Bit-exact with `difflib.SequenceMatcher(None, a, b,
  autojunk=False).ratio()` (verified on 500 random pairs).
  `sa.ratcliff_obershelp_similarity(a, b)` for the scalar form;
  `sa.ratcliff_obershelp_similarities(query, targets)` for the
  batch form returning `ndarray[float64]`. NOT commutative — the
  longest-common-substring tiebreak (`earliest in a, then earliest
  in b`) means the recursion splits leftover ranges differently
  for `(a, b)` vs `(b, a)`. Faithful to difflib, which has the
  same property.

* **Longest Common Subsequence + Substring (Phase D.4).** Two
  related but distinct dynamic programs:
  * `sa.lcs_length(a, b)` — length of the longest common
    subsequence (characters need not be contiguous). Cross-checked
    via the closed-form relation
    `indel = |a| + |b| - 2 * lcs_length`.
  * `sa.lcs_substring_length(a, b)` — length of the longest common
    substring (contiguous).
  * `sa.lcs_substring(a, b)` — the substring itself, sliced from
    `a`. Returns `bytes` when both inputs are `bytes`; otherwise
    `str`. First-occurrence-in-`a` tiebreak (matches `str.find`
    convention).

  Both DPs are scalar `O(m·n)` time with two rolling rows for
  `O(min(m,n))` (subsequence) or `O(|b|)` (substring) space. Engine
  runs in codepoint space — dispatch widens `PyUnicode_DATA`
  straight into `std::vector<Codepoint>`; non-ASCII codepoints are
  first-class.

* **Beider-Morse Phonetic Matching (`sa.beider_morse`).** Multi-language
  phonetic encoder for family names (Beider & Morse, 2008). Returns a
  `|`-separated set of plausible pronunciation codes across European
  languages. Ships the GENERIC name-type rule set only (the broad
  general-purpose tree from the Apache Commons Codec `bm/` resource
  files, Apache 2.0 — vendored under `src/stride_align/bmpm_data/`).
  `BmpmRuleType.APPROX` (default) gives a broader spread;
  `BmpmRuleType.EXACT` tightens it. `concat=` controls whether multi-
  word names encode jointly or per-word with `-`-joined codes;
  `max_phonemes=` caps the alternative set per encode (default 20).
  The C++ engine is compiled into the `_generic` backend module only
  so the ~280 KB rule set lives in one `.so` per wheel rather than
  fourteen; Python re-exports `beider_morse` through that backend
  regardless of detected CPU. Cross-checked against the canonical
  Apache Commons Codec `PhoneticEngineTest` GENERIC vectors —
  Renault, SntJohn-Smith, d'ortley, van helsing, and Judenburg all
  match byte-for-byte.

* **Daitch-Mokotoff Soundex (`sa.daitch_mokotoff`).** Six-digit
  Soundex tuned for Slavic and Yiddish surnames (Daitch & Mokotoff,
  1985). The leading letter is encoded, multi-character clusters
  like `sch`, `tsch`, `schtsch`, `rz`, `cz` fire before any single-
  letter rule, and several rules emit `|`-separated alternative
  codes. `branching=False` returns the first code only;
  `folding=False` skips the ASCII fold of accented characters
  before encoding. Rule table and folding map are vendored from the
  Apache Commons Codec `dmrules.txt` resource (Apache 2.0) and
  embedded in `include/stride_align/daitch_mokotoff.hpp`. Cross-
  checked against the canonical `DaitchMokotoffSoundexTest` vectors.

### Changed

* **`phonetic-compat` extras pruned.** Dropped `doublemetaphone>=1.0`
  — no test in `tests/` imported it.

* **`sa.cologne_phonetic` runs in codepoint space.** The dispatch
  wrapper widens Python `str` storage straight out of
  `PyUnicode_DATA` into `std::vector<Codepoint>`. The umlaut / ß
  fold is keyed on the natural Unicode codepoint (Ä = U+00C4,
  ß = U+00DF, ...) rather than on UTF-8 byte pairs. Same algorithm,
  same outputs — the 28 canonical Cologne test vectors continue to
  match upstream. `bytes` input is now taken as Latin-1
  codepoints; if you previously relied on passing the UTF-8 bytes
  form of an accented name (`"Müller".encode("utf-8")`), pass the
  `str` instead.

### Removed

* The `Levenshtein` PyPI entry in the `bench` extras. The benchmark
  and correctness scripts in `tools/` route Levenshtein comparisons
  through `rapidfuzz.distance.Levenshtein.distance`.

## [0.4.1] - 2026-06-01

### Added

* **`cdist_top_k_per_query` thread pool.** New `cpu_count=` kwarg.
  `cpu_count=0` auto-detects via `os.cpu_count() or 1`; `cpu_count=1`
  keeps the existing single-threaded per-pair generator path;
  `cpu_count > 1` runs a worker pool with one query per worker over
  the same byte-snapshot + `compute_row_double<Ops>` SIMD batch model
  that powers `cdist_top_k`. Workers process whole rows under the
  released GIL, results return in input order. ~6× speedup at
  `cpu_count=8` on a 20 queries × 5000 targets sweep
  (43 ms → 7 ms). Wide-unicode inputs that can't go through the
  byte-snapshot path silently fall back to the single-threaded
  per-pair generator. Threaded path always applies the length-bound
  prune (`max_normalized_similarity == 0` drops the pair) for
  correctness; the adaptive heap-min cutoff under `pruning=True`
  remains single-threaded-only.

### Changed

* **README structure.** The detailed LoongArch installation walk-through
  was crowding the install header — moved to a dedicated
  `## LoongArch installation` section near the bottom of the README,
  with the Installation header collapsed to a two-line pointer
  carrying an in-page anchor link. The two AI-friendly intro
  paragraphs were trimmed by ~20% without dropping coverage. Chinese
  README brought into structural parity.

* **`tests/test_api.py::test_pyproject_does_not_depend_on_parasail`.**
  Now checks only the runtime `[project.dependencies]` array via
  `tomllib`, rather than substring-scanning the whole `pyproject.toml`.
  parasail listed in the `bench` opt-in extra is fine — that's not a
  runtime dependency.

## [0.4.0] - 2026-06-01

### Added

* **Phonetic encoders (Phase D.2).** Full standard family with the
  same byte-extraction + variant pattern as the rest of the library:
  * **Soundex** (Russell & Odell, 1918). `soundex`, `soundex_equal`.
  * **Metaphone** (Lawrence Philips, 1990) with `MetaphoneVariant`
    enum — `PHILIPS` (spec-correct, Apache Commons Codec branch)
    and `JELLYFISH` (matches the popular Python library). `metaphone`,
    `metaphone_equal`.
  * **Double Metaphone** (Lawrence Philips, 2000) with
    `DoubleMetaphoneVariant` enum — `COMMONS` (faithful Apache
    Commons Codec / `doublemetaphone` PyPI port) and `PYTHON`
    (bug-compat with the `metaphone` PyPI package's missing-else GH
    leak). Returns `(primary, alternate)`. `double_metaphone`.
  * **NYSIIS** (Taft, 1970). `nysiis`, `nysiis_equal`.
  * **Match Rating Approach** (Moore, Western Airlines, 1977).
    `match_rating_codex`, `match_rating_compare`.
  * **Caverphone 2.0** (Hood, 2004). `caverphone`.

  All encoders dispatch through the same `peel_to_ascii_string`
  helper, accept `str`/`bytes` interchangeably, and skip non-letter /
  non-ASCII codepoints. Cross-checked against Apache Commons Codec
  reference data and the `jellyfish`, `metaphone`, `doublemetaphone`,
  and `pyphonetics` PyPI packages.

* **Dynamic Time Warping** (`dtw`, `dtw_distances`). Sakoe-Chiba band
  via `window=` (absolute integer radius or fractional `0 < r <= 1`);
  L1 vs L2-squared distance auto-picked by dtype (`int16` -> L1,
  `float32`/`float64` -> L2-squared) with explicit `distance=`
  override. Inputs are NumPy `ndarray`; matching dtype enforced.

* **`cdist_top_k_per_query`.** Generator API:
  `for query, [(score, target), ...] in cdist_top_k_per_query(...)`.
  Per-query top-k heap rather than the global top-k that
  `cdist_top_k` returns. `pruning=False` by default; `pruning=True`
  enables adaptive length-difference pruning using the same
  `max_normalized_similarity` closed-form bound as
  `cdist_above_threshold`. Worst-in-heap tightens as scoring
  progresses, so targets whose bound can't beat the current heap
  minimum skip the kernel entirely. ~12× faster than the unpruned
  baseline on wide-length workloads in benchmarks; flat on tight-
  length workloads. Hamming length-mismatch pairs are skipped via
  the bound rather than raising `ValueError`.

* **Wide-token Farrar batch (Phase B.2.x).** Full SW + NW Farrar
  dispatch through the wide-token kernel for NumPy `ndarray` with
  `uint16` dtype and `str` inputs with codepoints > 255 (UCS-2).
  Auto-promotes at the binding layer; 8-bit byte kernel stays
  byte-stable for ASCII / Latin-1 / bytes workloads.

* **Wider PyPI metadata.** `[project.urls]` (Homepage / Documentation /
  Repository / Source / Download / Issues / Changelog / Benchmarks),
  `maintainers`, `license-files = ["LICENSE"]` (PEP 639),
  `Typing :: Typed` classifier, Natural Language classifiers for
  English and Chinese (Simplified), Operating System classifiers for
  Linux and macOS, and named extras for `bench` (`rapidfuzz`,
  `Levenshtein`, `editdistance`, `parasail`) and `phonetic-compat`
  (`jellyfish`, `metaphone`, `doublemetaphone`, `pyphonetics`).

### Changed

* **Python 3.12+ floor.** `requires-python = ">=3.12"`, dropping
  3.9 / 3.10 / 3.11. The nanobind 2.x vectorcall fast path is
  unconditional from this version; the dispatch layer no longer
  needs the dict-lookup workarounds that replaced 3.10's `match`
  statements.

* **Vectorcall: skip the Python wrapper frame for every cheap
  function.** Phonetic encoders, Jaro / Jaro-Winkler scalar entry
  points, the scalar Levenshtein / OSA / Hamming / Indel / true-DL /
  DTW entry points, `*_top_k`, `extract`, and `extract_best` are now
  direct re-exports of the C++ binding rather than Python forwarder
  wrappers. Tight-loop savings measured at 22-54% per call on the
  encoders and 22-26% on Levenshtein-family scoring. Docstrings
  moved to the C++ binding side so `help(sa.soundex)` etc. still
  works.

* **`str` / `bytes` rejection on batch inputs moved to C++.** The
  `reject_str_or_bytes_targets` helper fires from a single shared
  point in `preprocess.hpp`; the per-binding `_coerce_targets` Python
  helper is gone. Same `TypeError`, fewer Python frames.

### Fixed

* **Generator double-iteration in `*_top_k` and `extract`.** The
  bindings called `PySequence_Fast` to grab `items` for `make_top_k`,
  then passed the original handle to the score dispatcher — which
  re-materialised it. Generators were getting consumed by the first
  pass and the second returned empty. The dispatch calls now take
  the already-materialised `owner`. Surfaced by the wrapper-removal
  pass that exposed the underlying behaviour.

* **Double Metaphone alternate cleanup.** Empty alternate (matching
  the `metaphone` and `doublemetaphone` PyPI conventions) when the
  primary and alternate codes are identical, so `if alt:` works.

### Notes

* **No abi3 wheel.** The Stable ABI excludes the
  `cpython/unicodeobject.h` fast macros (`PyUnicode_KIND`,
  `PyUnicode_*_DATA`, `PyUnicode_GET_LENGTH`) that the zero-copy
  UCS-1/UCS-2/UCS-4 path relies on for the CJK / wide-token SIMD
  kernels. Their function-form equivalents either allocate a copy or
  are one call per codepoint, both fatal to the SIMD path. We ship
  one wheel per minor Python version (3.12 / 3.13 / 3.14) instead.
  Rationale recorded in `CMakeLists.txt`.

## [0.3.0] - 2026-05-24

### Added

* **Indel distance** (`Scorer.INDEL` / `Scorer.INDEL_NORMALIZED`).
  Levenshtein restricted to insertions and deletions; equivalent to
  `|a| + |b| - 2 * LCS(a, b)`. Bit-parallel single-word kernel uses
  the Allison-Dix (1986) recurrence; multi-word patterns fall back
  to scalar DP. Public API: `indel_score`, `indel_normalized_score`,
  `indel_scores`, `indel_normalized_scores`, `indel_top_k`,
  `indel_best`, and the corresponding normalized variants. Wired
  through every backend, `cdist`, `cdist_above_threshold`,
  `cdist_top_k`, and the function-reference dispatch in `extract`.

* **True (unrestricted) Damerau-Levenshtein**
  (`Scorer.TRUE_DAMERAU_LEVENSHTEIN` /
  `Scorer.TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED`). The unrestricted
  form where a single character may participate in multiple edits.
  Diverges from OSA on overlapping transpositions
  (e.g. `"ca"`→`"abc"`: OSA=3, true-DL=2). Scalar DP only; no
  bit-parallel kernel yet (Hyyrö 2003 exists but is significantly
  more complex than OSA's bit-parallel and rarely the bottleneck).
  Existing `Scorer.DAMERAU_LEVENSHTEIN` continues to refer to OSA —
  the API name is unchanged.

* **Length-difference pruning** for `cdist_above_threshold` and
  `cdist_top_k`. Each pair is gated by a closed-form upper bound on
  the achievable normalized similarity before any SIMD work runs;
  bounds are scorer-specific (`min/max` for Lev/OSA/true-DL,
  `(2 + min/max)/3` for Jaro, `2*min/(q+t)` for Indel, `1.0` if
  equal-length for Hamming).

* **`cdist_top_k` row-sort by query length, descending.** Longest
  queries processed first so close-length high-scoring pairs
  surface early and the shared `global_min_bound` atomic reaches a
  useful value before the short-query rows run.

* **Per-pair cutoff push-down into the SIMD kernels.** Myers
  (Levenshtein single-word + multi-word), OSA single-word, and the
  Hamming inner loop all bail when the running distance plus
  remaining-chars allowance proves the pair can't reach its cutoff;
  bailed lanes return the per-pair `cutoff + 1` sentinel.

* **`docs/adding-a-new-algorithm.md`**: grep-able checklist for the
  touch points (`Scorer` enum, runtime helpers, cdist switches,
  bindings, per-backend Implementation methods, tests) a new
  scorer / alignment algorithm / SIMD backend has to hit.

* **Python 3.9 support.** The three `match` blocks in the Python
  layer became dict lookups; `from __future__ import annotations`
  was already in place project-wide. `pyproject.toml`
  `requires-python = ">=3.9"`, classifiers extended.

### Changed

* **Lowered the build-time C++ requirement from C++23 to C++20.**
  The project doesn't actually use any C++23 library feature — the
  `cxx_std_23` setting was aspirational. Lowering it lets gcc 10
  toolchains build the project (POWER8 Ubuntu 20.04 ships gcc 9.4
  and 10.5). Two stdlib gaps in gcc-10 libstdc++ are bridged with
  feature-test-gated fallbacks (`std::bit_cast` →
  `__builtin_bit_cast`, `std::make_unique_for_overwrite` → plain
  `new T[n]`). See `docs/power8-gcc10-workarounds.md` for the full
  list and the revert recipe once gcc 16 lands.

### Fixed

* **`cdist_above_threshold` iterator on macOS and LoongArch64.** The
  end-of-stream signal previously used `throw nb::stop_iteration()`,
  which relies on cross-DSO RTTI matching for nanobind's
  `builtin_exception`. macOS's two-level namespace and at least one
  LoongArch toolchain configuration defeat that lookup, and the
  exception ended up routed through nanobind's generic
  `std::exception` translator → bare `RuntimeError` instead of
  Python's `StopIteration`. Replaced with the C-API path
  (`PyErr_SetNone(PyExc_StopIteration)` plus a null `nb::object`
  return), which bypasses C++ exception machinery entirely. Fixes
  91 macOS test failures.

## [0.2.0] - 2026-05-19

Backfilled from git history; this entry was not in the tree at
the v0.2.0 tag.

### Added

* **Levenshtein scoring** (`Scorer.LEVENSHTEIN` /
  `Scorer.LEVENSHTEIN_NORMALIZED`). Myers (1999) bit-parallel
  scalar reference plus a SIMD batch kernel (one target per 64-bit
  lane) on every backend. Single-word path for patterns ≤ 64
  characters; multi-word kernel (W=2/3/4) for 65–256.
* `score_cutoff` parameter on the Levenshtein and per-target Lev
  scores APIs — bails per-target once the lower-bound score
  exceeds the cutoff; results that exceed the cap come back as
  `cutoff + 1` (rapidfuzz convention).
* **Damerau-Levenshtein (OSA-restricted, Hyyrö 2002)** —
  `Scorer.DAMERAU_LEVENSHTEIN` / `Scorer.DAMERAU_LEVENSHTEIN_NORMALIZED`.
  Scalar DP + bit-parallel scalar + SIMD batch on x86, NEON
  (Linux + macOS), SVE / SVE2, LSX / LASX, and PowerPC VSX. The
  "OSA-restricted" form is what rapidfuzz exposes as `OSA.distance`
  and is what most callers asking for "Damerau-Levenshtein"
  actually want.
* Cross-architecture benchmark sweeps for Levenshtein and OSA
  recorded under `benchmarks/` and summarized in
  [BENCHMARK.md](BENCHMARK.md). Highlight: AVX-512 LEV /
  DAML between 3.0x – 4.2x rapidfuzz on short targets; Mac M4
  NEON 5.5x – 8.5x python-Levenshtein.

### Changed

* LSX / LASX `vandn` semantics corrected to Intel-style `(~a) & b`
  (the LoongArch naming previously implied the opposite operand
  order). Affects the LSX / LASX backends only.
* `tools/benchmark_libs.py` extended with Levenshtein columns
  (`rapidfuzz`, `editdistance`, `Levenshtein`).
* New `tools/correctness_check.py` script.
* README localizations and the language carousel were dropped (English
  only for this release); the HTML build was regenerated to match.
  README now documents the LoongArch64 wheel sideload from the GitHub
  release (PyPI does not accept the `linux_loongarch64` platform tag).

## [0.1.0] - 2026-05-18

Initial public release.

### Added

* **Smith-Waterman and Needleman-Wunsch** sequence alignment with
  a nanobind C++23 backend and runtime SIMD dispatch.
* Backends:
  * x86: SSE4.1, AVX2, AVX-512 BWVL, AVX10-256, AVX10-512
  * ARM: Linux NEON / ASIMD, SVE, SVE2; macOS arm64 NEON
  * LoongArch (Loongson): LSX, LASX
  * PowerPC64: VSX
  * RISC-V: RVV (stub)
  * Portable: SWAR + pure-Python fallback
* Public API:
  * `smith_waterman_score` / `needleman_wunsch_score` /
    `smith_waterman_farrar_score`
  * Plural `*_scores` returning zero-copy `numpy.ndarray[int64]`
  * `*_normalized_score` / `*_normalized_scores` returning
    `float64` (length-normalized similarity in [0, 1])
  * Path / path-info / CIGAR variants for both SW and NW
  * Affine and linear gap models, score widths 8 / 16 / 32 / 64
  * String, bytes, and arbitrary-token sequence inputs
* CIGAR output uses the extended SAM convention (`=` sequence
  match, `X` mismatch, `I` / `D` indel). `build_cigar` /
  `ReverseCigarBuilder` emit digits via `std::to_chars` into a
  stack buffer with pre-reserved capacity (1.2x–2.2x speedup per
  row over the naive formulation).
* Benchmarks vs parasail (geomean across 80-row sweeps, 2026-05-18):
  Intel AVX-512 BWVL **1.752x**, AVX2 1.377x; Graviton4 NEON
  1.138x; Mac M4 NEON 1.065x; Loongson LASX **4.909x** (vs
  generic), **7.517x** (vs patched parasail, 1:1); Power8 VSX
  **3.772x**.
* Docs / tooling:
  * README in 16 languages with RTL-ready CSS (en, zh-CN, zh-TW,
    ja, de, ko, fr, es, pt-BR, ru, vi, id, hi, ar, tr, pl) — note:
    the translations were dropped again in v0.2.0 and the
    Simplified Chinese reintroduced in v0.3.0.
  * Themed `html/` rendition of every README + BENCHMARK.
  * `BENCHMARK.md` cross-architecture writeup.
  * Two runnable demos (Bible-verse nearest match + spell checker).
