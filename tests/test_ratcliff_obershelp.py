"""Ratcliff-Obershelp similarity — Phase D.5.

Python's ``difflib.SequenceMatcher().ratio()`` algorithm.

The oracle is the stdlib ``difflib.SequenceMatcher`` itself, called
with ``autojunk=False`` so the comparison stays apples-to-apples
(our engine has no junk-character heuristic, so the autojunk-on form
diverges only when difflib's heuristic decides to treat some
character as junk — typically only on long inputs with very
frequent characters).
"""

from __future__ import annotations

import difflib

import pytest

import stride_align as sa


# ---- Cross-check against difflib --------------------------------

PAIRS = [
    # Identity / empty.
    ("", ""),
    ("a", ""),
    ("", "b"),
    ("abc", "abc"),
    ("abc", "xyz"),
    # Single edit (substitution).
    ("kitten", "sitting"),
    # Reordering.
    ("hello world", "world hello"),
    # Textbook LCS example.
    ("ABCBDAB", "BDCAB"),
    # Unicode codepoints.
    ("Müller", "Mueller"),
    ("difflib", "diff_lib"),
    # Single-character mismatch.
    ("abcdef", "abcgef"),
    # Long-ish identical-prefix mismatch.
    ("the quick brown fox", "the quick brown dog"),
    # Repeated character patterns.
    ("aaaabbbb", "bbbbaaaa"),
    ("aaaaaaaa", "aaaaaaaab"),
    # Pure-prefix / pure-suffix.
    ("prefix", "prefixmore"),
    ("morerunning", "running"),
]


@pytest.mark.parametrize("a, b", PAIRS)
def test_matches_difflib_autojunk_false(a: str, b: str) -> None:
    ours = sa.ratcliff_obershelp_similarity(a, b)
    theirs = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    assert ours == pytest.approx(theirs, abs=1e-12)


def test_identical_inputs_score_one() -> None:
    # The defining property: identical inputs score exactly 1.0.
    assert sa.ratcliff_obershelp_similarity("hello", "hello") == 1.0
    # Empty inputs are identical by convention (matches difflib).
    assert sa.ratcliff_obershelp_similarity("", "") == 1.0


def test_no_overlap_scores_zero() -> None:
    # Disjoint character alphabets give 0.0.
    assert sa.ratcliff_obershelp_similarity("abc", "xyz") == 0.0
    assert sa.ratcliff_obershelp_similarity("xyz", "abc") == 0.0


def test_one_side_empty_scores_zero() -> None:
    assert sa.ratcliff_obershelp_similarity("abc", "") == 0.0
    assert sa.ratcliff_obershelp_similarity("", "abc") == 0.0


def test_not_strictly_symmetric() -> None:
    # Ratcliff-Obershelp is NOT commutative — the tiebreak on the
    # inner longest-common-substring step ("earliest in a, then
    # earliest in b") means the recursion splits the leftover ranges
    # differently depending on which side is a. Total match lengths
    # can therefore differ between ``(a, b)`` and ``(b, a)``. This is
    # not a bug — it is faithful to ``difflib.SequenceMatcher`` which
    # has the same property. Pin one canonical asymmetric pair so the
    # behaviour is documented.
    forward  = sa.ratcliff_obershelp_similarity("ABCBDAB", "BDCAB")
    backward = sa.ratcliff_obershelp_similarity("BDCAB",   "ABCBDAB")
    assert forward != backward
    assert forward  == pytest.approx(
        difflib.SequenceMatcher(None, "ABCBDAB", "BDCAB", autojunk=False).ratio(),
        abs=1e-12,
    )
    assert backward == pytest.approx(
        difflib.SequenceMatcher(None, "BDCAB", "ABCBDAB", autojunk=False).ratio(),
        abs=1e-12,
    )


def test_bytes_input_accepted() -> None:
    # bytes path widens each byte to a Latin-1 codepoint, so on ASCII
    # the result is identical to the str form.
    assert sa.ratcliff_obershelp_similarity(b"hello", b"hello") == 1.0
    assert sa.ratcliff_obershelp_similarity(b"abc", b"xyz") == 0.0
    assert sa.ratcliff_obershelp_similarity(b"kitten", b"sitting") == \
           sa.ratcliff_obershelp_similarity("kitten", "sitting")


def test_result_is_in_unit_interval() -> None:
    # The defining contract: similarity in [0, 1].
    import random
    random.seed(17)
    for _ in range(50):
        m = random.randrange(0, 30)
        n = random.randrange(0, 30)
        a = "".join(random.choice("abcde") for _ in range(m))
        b = "".join(random.choice("abcde") for _ in range(n))
        r = sa.ratcliff_obershelp_similarity(a, b)
        assert 0.0 <= r <= 1.0, (a, b, r)


# ---- Batch ----------------------------------------------------

def test_similarities_batch_matches_per_pair() -> None:
    query = "kitten"
    targets = ["sitting", "kitten", "kit", "biting"]
    batch = sa.ratcliff_obershelp_similarities(query, targets)
    expected = [
        sa.ratcliff_obershelp_similarity(query, t) for t in targets
    ]
    for got, want in zip(batch.tolist(), expected):
        assert got == pytest.approx(want, abs=1e-12)


def test_similarities_returns_float64_ndarray() -> None:
    import numpy as np
    batch = sa.ratcliff_obershelp_similarities(
        "kitten", ["sitting", "kit"],
    )
    assert isinstance(batch, np.ndarray)
    assert batch.dtype == np.float64
    assert batch.shape == (2,)


def test_similarities_rejects_str_or_bytes_targets() -> None:
    # Same defensive shape as the other ``*_similarities`` entrypoints.
    with pytest.raises(TypeError, match="not a single str/bytes"):
        sa.ratcliff_obershelp_similarities("query", "single")
    with pytest.raises(TypeError, match="not a single str/bytes"):
        sa.ratcliff_obershelp_similarities("query", b"single")


def test_similarities_accepts_generator() -> None:
    # Generator gets materialised inside the wrapper.
    def gen():
        yield "sitting"
        yield "kit"
    batch = sa.ratcliff_obershelp_similarities("kitten", gen())
    assert batch.shape == (2,)
