# Keyboard typo matrices — data provenance & permission

The optional keyboard substitution matrices exposed through
`stride_align.matrices.keyboard` are derived from the Aalto University
"136 Million Keystrokes" dataset.

## Dataset

> Vivek Dhakal, Anna Maria Feit, Per Ola Kristensson, Antti Oulasvirta.
> "Observations on Typing from 136 Million Keystrokes."
> Proceedings of the 2018 CHI Conference on Human Factors in Computing
> Systems (CHI '18), 2018. <https://doi.org/10.1145/3173574.3174220>
> Dataset: <https://userinterfaces.aalto.fi/136Mkeystrokes/>

The dataset's public terms permit research / non-commercial use with
attribution.

## Permission

The stride-align author obtained **explicit written permission** from one of
the dataset authors (Antti Oulasvirta, 2025-05-31) to build and publish a
small number of derived scoring matrices — roughly 3–4 NumPy matrices, e.g. a
generic matrix plus per-layout matrices (QWERTY, QWERTZ, …) — as part of
stride-align, **under stride-align's Apache-2.0 licence**, as examples of how
real typo data can improve text alignment.

Scope of the grant:

- Only **derived scoring matrices** are published. The **raw keystroke data
  is not redistributed**.
- The derived matrices ship under **Apache-2.0** as part of stride-align.
- Attribution (the citation above) is retained.

Courtesy follow-up: send the dataset author a link once the matrices are
published.

## How the matrices are built

`tools/build_keyboard_matrices.py` reads the dataset zip, extracts backspace
self-corrections ("typed X, Backspace, typed Y") per keyboard layout, and
turns the per-layout confusion counts into **log-odds** substitution
matrices (`score = scale · log2(observed / expected)`, the same construction
BLOSUM/PAM use). The identity diagonal is set above the strongest
substitution so an exact match always wins. The matrix is oriented
`m[a][b]` = `m[query][target]` (row = the typed/mistake char, column = the
intended/correction char).

The raw dataset and the script's full multi-layout output are not committed
to this repo; only the small curated set of published matrices is.
