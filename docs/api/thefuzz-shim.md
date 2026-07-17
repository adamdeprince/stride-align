# TheFuzz compatibility shim

`stride_align.thefuzz` is a work-alike facade for TheFuzz 0.22.1.
Change the import and retain TheFuzz's integer scores, preprocessing,
selection helpers, and result shapes:

```python
# Before:
# from thefuzz import fuzz, process

# After:
from stride_align.thefuzz import fuzz, process

fuzz.ratio("this is a test", "this is a test!")  # 97
process.extractOne("cowboys", ["New York Jets", "Dallas Cowboys"])
# ('Dallas Cowboys', 90)
```

This is a migration facade. New code that does not need TheFuzz's exact
integer conventions can use `stride_align.rapidfuzz` for floating-point
scores or the native `[0, 1]` APIs directly.

## Covered surface

| Module | Functions |
| --- | --- |
| `fuzz` | `ratio`, `partial_ratio`, `token_sort_ratio`, `partial_token_sort_ratio`, `token_set_ratio`, `partial_token_set_ratio` |
| `fuzz` aliases/composites | `QRatio`, `UQRatio`, `WRatio`, `UWRatio` |
| `process` | `extractWithoutOrder`, `extract`, `extractBests`, `extractOne`, `dedupe` |
| `utils` | `ascii_only`, `full_process`, public `translation_table` |

The package-level `__version__` is `"0.22.1"`, identifying the upstream
compatibility target rather than the installed stride-align release.

## Compatibility details

TheFuzz differs in observable ways from its RapidFuzz engine. The facade
retains them:

- Every built-in scorer returns an `int` from 0 through 100 using Python's
  `int(round(score))` behavior, including bankers-rounding boundaries.
- Token scorers, `QRatio`, and `WRatio` run `utils.full_process` by default.
  `force_ascii=True` preserves TheFuzz's historical rule: delete code points
  128 through 255, not every non-ASCII character. `UQRatio` and `UWRatio`
  disable that deletion.
- `full_process` replaces each non-alphanumeric character separately,
  lowercases with Unicode simple-lower rules, and strips the ends. It does
  not collapse internal spaces.
- `ratio` and `partial_ratio` accept arbitrary Python sequences of hashable
  elements. Generic sequence elements first receive TheFuzz/RapidFuzz's
  compatibility normalization; the result then goes through stride-align's
  shared compact-token encoder and native Indel/partial-ratio kernels.
- Extraction ranks the unrounded score, rounds only the returned built-in
  score, preserves input order on exact ties, and skips `None` choices.
  Iterable choices return `(choice, score)`; mappings return
  `(choice, score, key)`.
- Custom scorers retain their original numeric result rather than being
  integer-rounded.

The in-tree test suite runs independently with pinned expectations. When
TheFuzz 0.22.1 is installed, it also runs randomized differential batteries
over strings, Unicode, byte buffers, arbitrary hashable-element sequences,
ranking, cutoffs, mappings, and all public functions.

## Known limits

The `partial_ratio` family uses stride-align's native block-anchored partial
matcher. It never overshoots the upstream score, but it can underestimate by
a few points on rare pairs where RapidFuzz finds a shifted optimal window.
That also affects partial-token scorers and can flow into `WRatio`.

`full_process` follows the Unicode tables in the running Python release.
Newly assigned characters can therefore be classified differently from the
fixed Unicode table compiled into a particular RapidFuzz release until both
projects update to the same Unicode version.

## TheFuzz versus FuzzyWuzzy

This facade targets the MIT-licensed **TheFuzz 0.22.1** package. It does not
claim a separately tested `fuzzywuzzy` facade: the legacy FuzzyWuzzy source is
GPL-2.0 and is explicitly excluded from this project. Code using their common
core scorer names will often be portable, but TheFuzz 0.22.1 is the behavioral
contract tested here.

## Native equivalents

- [rapidfuzz-shim.md](rapidfuzz-shim.md) — floating-point 0–100 scores,
  distance classes, `cdist`, and RapidFuzz result shapes
- [edit-distance.md](edit-distance.md) — native Indel and Levenshtein
  single/batch/top-k operations
- [similarity.md](similarity.md) — native token ratios and other similarity
  families on the `[0, 1]` scale

Implementation and license provenance is recorded in
[thefuzz-shim-external-sources.md](../thefuzz-shim-external-sources.md).
