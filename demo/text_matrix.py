"""Smith-Waterman / Needleman-Wunsch with a custom substitution
matrix for human-readable English text.

Plain Levenshtein gives every edit equal weight: ``"Hello"`` vs
``"HELLO"`` is the same distance as ``"Hello"`` vs ``"12345"``. For
fuzzy text matching that's usually wrong — we'd like to say a case
swap is *nearly identical*, a vowel-for-vowel swap is *similar*, a
consonant-for-consonant swap is *less similar*, and a letter-for-
digit swap is *quite different*.

This demo builds a 75-character alphabet covering lowercase,
uppercase, digits, the most common English punctuation, and a
wildcard. It scores each (a, b) cell by character class:

  * exact match                                              +6
  * case swap            (Hello vs HELLO)                    +4
  * vowel-vs-vowel mismatch (e.g. 'a' vs 'e')                +1
  * consonant-vs-consonant mismatch                           0
  * letter-vs-letter cross-case mismatch                     -2
  * letter-vs-digit                                          -3
  * letter-vs-punct, digit-vs-punct                          -4
  * wildcard / out-of-alphabet                               -4

Then it shows:

  1. Pure-Python encode round-trip with the new case-sensitive
     contract.
  2. Pairwise NW scoring vs Levenshtein for a set of contrast pairs
     so you can read off where the matrix changes the ranking.
  3. SW local search: "find this snippet in this paragraph" with
     affine gaps. Reports the aligned substring and its score.
"""

from __future__ import annotations

import numpy as np

import stride_align as sa
from stride_align.matrices import SubstitutionMatrix


# ----- Build the English text alphabet ---------------------------------

LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS    = "0123456789"
PUNCT     = " .,!?;:'\"-()"
WILDCARD  = "~"  # not present anywhere else in the alphabet

ALPHABET = LOWERCASE + UPPERCASE + DIGITS + PUNCT + WILDCARD
assert len(set(ALPHABET)) == len(ALPHABET), "alphabet has duplicates"

VOWELS_LOWER = set("aeiou")
VOWELS_UPPER = set("AEIOU")


def _class(c: str) -> str:
    if c in LOWERCASE:  return "lower"
    if c in UPPERCASE:  return "upper"
    if c in DIGITS:     return "digit"
    if c in PUNCT:      return "punct"
    return "wild"


def _score_pair(a: str, b: str) -> int:
    """Score one ordered pair (a, b) per the rules above."""
    if a == WILDCARD or b == WILDCARD:
        return -4
    if a == b:
        return 6
    ac, bc = _class(a), _class(b)
    # Letter pairs.
    if ac in ("lower", "upper") and bc in ("lower", "upper"):
        if a.lower() == b.lower():       # case swap
            return 4
        a_is_vowel = a in VOWELS_LOWER or a in VOWELS_UPPER
        b_is_vowel = b in VOWELS_LOWER or b in VOWELS_UPPER
        if ac == bc:
            return 1 if (a_is_vowel and b_is_vowel) else 0
        return -2                         # cross-case different letters
    if {ac, bc} == {"lower", "digit"} or {ac, bc} == {"upper", "digit"}:
        return -3
    if "punct" in (ac, bc):
        return -4
    if ac == "digit" and bc == "digit":
        return 0                          # digit-vs-digit: neutral
    return -4


def build_text_matrix() -> SubstitutionMatrix:
    n = len(ALPHABET)
    grid = np.zeros((n, n), dtype=np.int8)
    for i, a in enumerate(ALPHABET):
        for j, b in enumerate(ALPHABET):
            grid[i, j] = _score_pair(a, b)
    return SubstitutionMatrix(
        name="english-text",
        alphabet=ALPHABET,
        matrix=grid,
        gap_score=-3,
        gap_open=-5,
        gap_extend=-1,
        wildcard=WILDCARD,
    )


# ----- Demo --------------------------------------------------------------

def banner(text: str) -> None:
    print()
    print(text)
    print("=" * len(text))


def main() -> None:
    matrix = build_text_matrix()
    banner("Matrix shape and step-limit")
    print(f"  alphabet size           : {matrix.stride}")
    print(f"  max |matrix entry|      : {matrix.max_abs}")
    print(f"  gap_open / gap_extend   : {matrix.gap_open} / {matrix.gap_extend}")
    print(f"  step_limit (defaults)   : {matrix.score_step_limit()}")
    print()
    print("  diagonal (exact match)      : '" + "', '".join(
        f"{int(matrix.matrix[matrix.alphabet.index(c), matrix.alphabet.index(c)])}"
        for c in "aAeE5!"
    ) + "'")
    print("  'a' vs 'A' (case swap)      :", int(
        matrix.matrix[matrix.alphabet.index("a"), matrix.alphabet.index("A")]))
    print("  'a' vs 'e' (vowel-vowel)    :", int(
        matrix.matrix[matrix.alphabet.index("a"), matrix.alphabet.index("e")]))
    print("  'b' vs 'd' (cons-cons)      :", int(
        matrix.matrix[matrix.alphabet.index("b"), matrix.alphabet.index("d")]))
    print("  'a' vs 'b' (vowel-cons)     :", int(
        matrix.matrix[matrix.alphabet.index("a"), matrix.alphabet.index("b")]))
    print("  'a' vs '5' (letter-digit)   :", int(
        matrix.matrix[matrix.alphabet.index("a"), matrix.alphabet.index("5")]))
    print("  'a' vs '!' (letter-punct)   :", int(
        matrix.matrix[matrix.alphabet.index("a"), matrix.alphabet.index("!")]))

    banner("encode round-trip (case-sensitive contract)")
    print(f"  encode('Hello')         = {list(matrix.encode('Hello'))}")
    print(f"  encode('HELLO')         = {list(matrix.encode('HELLO'))}")
    print("  ('h' and 'H' map to different indices — no implicit fold)")

    banner("NW global score vs plain Levenshtein")
    pairs = [
        ("Hello, world!",  "Hello, world!"),    # identical
        ("Hello, world!",  "HELLO, WORLD!"),    # case swap throughout
        ("color",          "colour"),           # one insertion
        ("color",          "kolor"),            # cross-case-different start
        ("color",          "12345"),            # totally unrelated
        ("the quick brown", "the quikc brwon"), # adjacent-transposition style
        ("Hello",          "Hxllo"),            # vowel-for-consonant
        ("Hello",          "Hallo"),            # vowel-for-vowel
    ]
    name_w = max(len(repr(a)) + len(repr(b)) for a, b in pairs) + 4
    print(f"  {'pair':<{name_w}} {'matrix NW':>10}  {'Lev':>5}  {'Lev ratio':>10}")
    for a, b in pairs:
        nw = sa.needleman_wunsch_score(
            a, b, matrix=matrix,
            gap_open_score=matrix.gap_open, gap_extend_score=matrix.gap_extend,
        )
        lev = sa.levenshtein_score(a, b)
        ratio = sa.levenshtein_normalized_score(a, b)
        pair = f"{a!r} vs {b!r}"
        print(f"  {pair:<{name_w}} {nw:>10}  {lev:>5}  {ratio:>10.3f}")

    banner("SW local search — find a snippet inside a paragraph")
    paragraph = (
        "The quick brown fox jumps over the lazy dog. "
        "A stitch in time saves nine. "
        "All happy families are alike; each unhappy family is unhappy "
        "in its own way."
    )
    needles = [
        "Quick Brown FOX",            # case-shuffled exact
        "lzay dog",                   # transposed
        "stitch in time",             # clean substring
        "happy familys",              # near-miss spelling
        "moose on the loose",         # not present
    ]
    print(f"  paragraph: {paragraph!r}")
    print()
    for needle in needles:
        result = sa.smith_waterman_path(
            needle, paragraph, matrix=matrix,
            gap_open_score=matrix.gap_open, gap_extend_score=matrix.gap_extend,
        )
        # Strip alignment gaps for a readable echo of what matched.
        matched = result.aligned_target.replace("-", "")
        print(f"  needle: {needle!r}")
        print(f"    score             : {result.score}")
        print(f"    matched substring : {matched!r}")
        print(f"    target positions  : [{result.target_start}, {result.target_end})")
        print(f"    cigar             : {result.operations}")
        print()


if __name__ == "__main__":
    main()
