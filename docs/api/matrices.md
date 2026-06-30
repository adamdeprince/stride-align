# Substitution matrices

`stride_align.matrices.SubstitutionMatrix` is the per-pair scoring
table consumed by Smith-Waterman / Needleman-Wunsch and the
matrix-aware `cdist` paths. A catalog of built-in protein (BLOSUM,
PAM), nucleotide (NUC.4.4, simple DNA), and case-sensitive ASCII
matrices — plus an NCBI-text parser and rule-built factories —
covers the common bioinformatics and text-similarity workloads;
fully custom matrices over any alphabet are first-class too.

## The `SubstitutionMatrix` dataclass

```python
from stride_align.matrices import SubstitutionMatrix
import numpy as np

m = SubstitutionMatrix(
    name="MyMatrix",
    alphabet="ACGT",                       # str — codepoints → row/col indices
    matrix=np.array([                      # int8 ndarray, shape (n, n)
        [ 2, -1, -1, -1],
        [-1,  2, -1, -1],
        [-1, -1,  2, -1],
        [-1, -1, -1,  2],
    ], dtype=np.int8),
    gap_score=-2,                          # default linear-gap penalty
    wildcard="*",                          # optional sentinel for "not in alphabet"
)
```

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | `str` | human-readable identifier |
| `alphabet` | `str` | each codepoint maps to row/col index by position |
| `matrix` | `np.ndarray[int8]` shape `(n, n)` | symmetric or asymmetric per-pair score |
| `gap_score` | `int` | default linear-gap penalty |
| `wildcard` | `str` (single char) or `None` | codepoint that absorbs "outside the alphabet" |

Optional `gap_open` / `gap_extend` fields override the alignment-
call defaults; explicit `gap_open_score=` / `gap_extend_score=`
keyword args on the alignment call still take precedence.

## Methods

| Method | Purpose |
| --- | --- |
| `encode(sequence: str) -> bytes` | translate a string to alphabet indices for direct kernel consumption (zero-copy when the alphabet is single-byte) |
| `score(query: str, target: str) -> int` | look up the substitution score for two single characters |
| `stride() -> int` | the alignment-kernel-internal stride (== `len(alphabet)`) |
| `score_step_limit(*, gap_score=None, gap_open=None, gap_extend=None) -> int` | maximum DP step size given the scoring scheme; useful for choosing `width=` on the alignment call |
| `from_ncbi_text(text, *, name=None, gap_score=-4, wildcard='X')` (classmethod) | parse an NCBI-format text dump (e.g. a saved BLOSUM62 table) into a `SubstitutionMatrix` |

`encode` is a pure translation-table lookup since 0.5.0 — case-folding
is no longer implicit. Pass `seq.upper()` if your alphabet is
uppercase and the input might not be (see CHANGELOG `[0.5.0]
SubstitutionMatrix.encode no longer case-folds`).

## Built-in catalog

Every matrix below is a `SubstitutionMatrix` instance you can pass
directly as `matrix=` to any Smith-Waterman / Needleman-Wunsch /
`cdist` entry point. The empirical biological matrices (BLOSUM, PAM,
NUC.4.4) are the public-domain tables from the NCBI BLAST distribution
— the numbers are bundled verbatim under `stride_align/matrix_data/`
and parsed once at import. Each carries the affine gap penalties NCBI
conventionally pairs with it as `gap_open` / `gap_extend` defaults;
override per call with `gap_open_score=` / `gap_extend_score=` (or
`gap_score=` for a linear gap).

```python
import stride_align as sa
from stride_align.matrices import blosum62

sa.smith_waterman_score(
    "MQDRVKRPMNAFIVWSRDQRRKMALEN",
    "KQDLRKLSVRNCAFRLRVKMWERPPNAFIVWSRDQRRKMALEN",
    matrix=blosum62,
    gap_open_score=-10,
    gap_extend_score=-1,
)
```

### Protein — BLOSUM

```python
from stride_align.matrices import (
    blosum30, blosum35, blosum40, blosum45, blosum50, blosum55,
    blosum60, blosum62, blosum65, blosum70, blosum75, blosum80,
    blosum85, blosum90, blosum100,
)
```

The full BLOSUM series, 30 through 100. *Lower* numbers target more
divergent sequences, *higher* numbers more similar ones; `blosum62` is
the de-facto default for protein search. All use the NCBI 24-letter
alphabet `ARNDCQEGHILKMFPSTWYVBZX*` with `X` as the wildcard and `*` the
BLAST stop character.

### Protein — PAM

```python
from stride_align.matrices import pam10, pam30, pam250, pam500  # …and every step in between
```

The complete PAM (Point Accepted Mutation) series: `pam10`, `pam20`, …,
`pam500` — every multiple of 10 from 10 to 500 (50 matrices). The
relationship is inverted relative to BLOSUM: *low* PAM numbers model
closely related sequences, *high* numbers model distant ones (`pam250`
is the classic distant-homology choice; the high end `pam400`–`pam500`
targets very deep, divergent comparisons). Same 24-letter alphabet as
BLOSUM.

### Nucleotide

```python
from stride_align.matrices import nuc44, dna_match
```

| Matrix | Alphabet | Use |
| --- | --- | --- |
| `nuc44` | `ATGCSWRYKMBVHDN` (IUPAC) | NCBI NUC.4.4 / "DNAfull" — scores IUPAC ambiguity codes (e.g. `R` = A/G), with `N` the any-base wildcard. Match +5, transversion -4. |
| `dna_match` | `ACGTN` | A plain +5 / -4 match-mismatch matrix over the four bases (`N` wildcard) — the classic BLASTN default; use when your sequences contain only A/C/G/T. |

### Text / generated

```python
from stride_align.matrices import ascii_text, identity_matrix, ascii_matrix
```

For text-similarity work where capitalisation and punctuation are signal
rather than noise:

| Symbol | Kind | Description |
| --- | --- | --- |
| `ascii_text` | instance | A ready-made case-sensitive 128×128 ASCII matrix (`'a'` ≠ `'A'`): equal codepoints score +1, differing -1. `chr(127)` is the wildcard slot that absorbs any non-ASCII input. |
| `ascii_matrix(*, match=1, mismatch=-1, ...)` | factory | Build a custom-scored variant of the above. |
| `identity_matrix(alphabet, *, match=1, mismatch=0, wildcard="*", ...)` | factory | An identity matrix over any single-codepoint alphabet — equal symbols score `match`, everything else `mismatch`. The wildcard is appended if absent and never matches (not even itself), so unknown input cannot score a spurious hit. |

```python
import stride_align as sa
from stride_align.matrices import identity_matrix, ascii_text

# An identity matrix over a custom alphabet (RNA, +2 / -1, gap -2):
rna = identity_matrix("ACGU", match=2, mismatch=-1, gap_score=-2)
sa.smith_waterman_score("ACGUACGU", "ACGUUCGU", matrix=rna)

# Plain text, case matters; non-ASCII folds to the wildcard:
sa.needleman_wunsch_score("Café", "cafe", matrix=ascii_text)
```

Both factories accept `name=`, `gap_score=`, `gap_open=`, and
`gap_extend=` so the resulting matrix carries the defaults you want.

## Keyboard typo matrices

`stride_align.matrices.keyboard` provides example substitution matrices
built from **real human typing errors** — the Aalto "136 Million
Keystrokes" dataset, used with the authors' permission. They score how
likely it is that one character was typed when another was meant, so an
aligner treats a plausible slip as a near-match rather than a flat mismatch.
The `qwerty` matrix ships bundled (`keyboard.available()` lists what's
present); build matrices for other layouts or alphabets from your own
confusion data with `from_confusion_counts`.

```python
import stride_align as sa
from stride_align.matrices import keyboard

sa.smith_waterman_score("teh", "the", matrix=keyboard.qwerty)
```

### Orientation: the query is the misspelled side

Like every stride_align matrix these are `m[a][b]` with **a = query, b =
target** — and here **a is the character actually typed (the mistake) and b
is the character that was intended (the correction)**. So pass the
misspelled / user-entered string as the query and the canonical string as
the target.

These matrices are **asymmetric** (unlike symmetric BLOSUM/PAM), so
orientation matters. If your query is instead the *correct* string, flip the
matrix with NumPy's `.T` (transpose), which swaps the axes to
`m[intended][misspelled]`:

```python
import numpy as np
from stride_align.matrices import SubstitutionMatrix, keyboard

fwd = keyboard.qwerty
rev = SubstitutionMatrix(
    name=fwd.name + ".T", alphabet=fwd.alphabet,
    matrix=np.ascontiguousarray(fwd.matrix.T),   # .T swaps query/target
    gap_score=fwd.gap_score, wildcard=fwd.wildcard,
)
```

### Building and loading

| Function | Purpose |
| --- | --- |
| `from_confusion_counts(counts, *, alphabet=ASCII_ALPHABET, scale=2.0, match_margin=4, floor=None, ...)` | build a log-odds matrix from a `{(typed, intended): count}` mapping (or an `(N, N)` count array) |
| `from_npy(path, *, transpose=False, ...)` | load an int8 `(N, N)` `.npy` matrix (e.g. from `tools/build_keyboard_matrices.py`); `transpose=True` applies the `.T` flip |
| `available()` | names of the bundled matrices in this install |

Scoring is a log-odds ratio (`scale · log2(observed / expected)`, the same
construction BLOSUM/PAM use), with the identity diagonal set above the
strongest substitution so an exact match always wins. The bundled matrices
are derived log-odds artifacts, not the raw keystroke data; see the
repository `NOTICE` and `docs/keyboard-matrix-external-sources.md` for the
attribution and the scope of the permission.

## Fully custom matrices

The catalog covers the common cases, but `SubstitutionMatrix` is
alphabet-agnostic — for full control, construct one directly from any
single-codepoint alphabet and any `(n, n)` int8 matrix (see the
dataclass example at the top of this page), or load one from NCBI text.

## Loading from NCBI text

The `from_ncbi_text` classmethod handles the on-disk format NCBI
distributes (whitespace-separated header line of letters, then one
row per letter). Useful if you want to bundle a custom matrix
without checking the values into Python source:

```python
import pathlib
from stride_align.matrices import SubstitutionMatrix

m = SubstitutionMatrix.from_ncbi_text(
    pathlib.Path("/data/custom-matrix.txt").read_text(),
    name="custom",
    gap_score=-4,
    wildcard="X",
)
```
