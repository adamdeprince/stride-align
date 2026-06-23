# Substitution matrices

`stride_align.matrices.SubstitutionMatrix` is the per-pair scoring
table consumed by Smith-Waterman / Needleman-Wunsch and the
matrix-aware `cdist` paths. Built-in BLOSUM and PAM matrices plus
a NCBI-text parser cover the common bioinformatics workloads;
custom matrices for text-similarity (case-sensitive, mixed
alphabet) are first-class too.

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

## Built-in matrices

```python
from stride_align.matrices import (
    blosum45, blosum50, blosum62, blosum80, blosum90,
    pam30, pam70, pam250,
)
```

All eight use the NCBI 24-letter alphabet `ARNDCQEGHILKMFPSTWYVBZX*`
with `*` as the BLAST stop-character wildcard. Each is a
`SubstitutionMatrix` instance ready to pass as `matrix=`:

```python
sa.smith_waterman_score(
    "MQDRVKRPMNAFIVWSRDQRRKMALEN",
    "KQDLRKLSVRNCAFRLRVKMWERPPNAFIVWSRDQRRKMALEN",
    matrix=blosum62,
    gap_open_score=-10,
    gap_extend_score=-1,
)
```

## Custom text matrices

The built-ins are biological, but `SubstitutionMatrix` is
alphabet-agnostic — pass any single-codepoint alphabet and any
`(n, n)` int8 matrix. A roadmap item is a 128x128 case-sensitive
ASCII matrix that maps `'a'` and `'A'` to distinct indices for
text-similarity workloads where capitalisation is signal rather
than noise.

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
