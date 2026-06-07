"""Token-ratio family — Phase D.3.

Six entry points that close the rapidfuzz ``fuzz.*`` API gap:

* ``sa.token_sort_ratio(s1, s2)``
* ``sa.token_set_ratio(s1, s2)``
* ``sa.partial_ratio(s1, s2)``
* ``sa.partial_token_sort_ratio(s1, s2)``
* ``sa.partial_token_set_ratio(s1, s2)``
* ``sa.WRatio(s1, s2)``

All return values in ``[0, 1]`` (stride-align convention; rapidfuzz
returns ``[0, 100]``).

The implementations are pure-Python compositions over
``sa.indel_normalized_score`` and ``sa.lcs_substring`` — no third-
party code is imported in the production path. rapidfuzz is used here
as a runtime oracle for the cases where bit-exact parity holds; the
``partial_ratio`` family has well-defined hand-computed values pinned
in this file (sliding-window of length ``min(|a|, |b|)`` plus the LCS
substring as an extra candidate), which can diverge from rapidfuzz by
a small margin when both inputs are the same length.
"""

from __future__ import annotations

import math

import pytest

import stride_align as sa


# ---- Hand-computed pinned values --------------------------------

# ``2·LCS / (|a|+|b|)`` is the Indel-normalised similarity formula.
def _indel(a: str, b: str) -> float:
    lcs = sa.lcs_length(a, b)
    if not a and not b:
        return 1.0
    return 2.0 * lcs / (len(a) + len(b))


# ---- token_sort_ratio --------------------------------------------

@pytest.mark.parametrize(
    "s1, s2, expected",
    [
        # Sorted joins are identical -> 1.0.
        ("fuzzy wuzzy", "wuzzy fuzzy",                 1.0),
        ("hello world", "world hello",                 1.0),
        ("the cat sat",  "sat cat the",                1.0),
        # Same set of tokens, plus extras on one side.
        ("foo bar",     "foo bar baz",                 _indel("bar foo", "bar baz foo")),
        # Identical inputs.
        ("kitten",      "kitten",                      1.0),
        # No overlap.
        ("abc def",     "xyz uvw",                     _indel("abc def", "uvw xyz")),
        # Empty/whitespace conventions match rapidfuzz: both empty
        # (or whitespace-only) joins compare as 1.0 via Indel.
        ("",            "",                            1.0),
        ("   ",         "   ",                         1.0),
    ],
)
def test_token_sort_ratio(s1, s2, expected) -> None:
    assert sa.token_sort_ratio(s1, s2) == pytest.approx(expected, abs=1e-12)


def test_token_sort_ratio_one_empty() -> None:
    # rapidfuzz: 0.0 when one side has no tokens and the other does.
    assert sa.token_sort_ratio("abc", "") == 0.0
    assert sa.token_sort_ratio("", "abc") == 0.0


# ---- token_set_ratio ---------------------------------------------

def test_token_set_ratio_full_overlap_returns_one() -> None:
    # All s1 tokens are a subset of s2: intersection covers s1, and
    # one of the three pairwise indel ratios is identity.
    assert sa.token_set_ratio("foo bar", "foo bar baz") == 1.0
    assert sa.token_set_ratio("apple", "an apple a day") == 1.0


def test_token_set_ratio_pinned_value() -> None:
    # 'fuzzy wuzzy was a bear' vs 'wuzzy fuzzy was a bear':
    # set(t1) == set(t2) -> intersection covers everything, set diffs
    # are empty -> three indel ratios all equal 1.0.
    assert sa.token_set_ratio(
        "fuzzy wuzzy was a bear",
        "wuzzy fuzzy was a bear",
    ) == 1.0


def test_token_set_ratio_disjoint_alphabets() -> None:
    # Sets are disjoint -> t0 = "", t1 = "abc def", t2 = "uvw xyz".
    # Three indel ratios are r("", t1) = 0, r("", t2) = 0,
    # r(t1, t2) = some value, max is r(t1, t2).
    expected = _indel("abc def", "uvw xyz")
    assert sa.token_set_ratio("abc def", "uvw xyz") == pytest.approx(expected, abs=1e-12)


def test_token_set_ratio_both_empty_zero() -> None:
    # rapidfuzz convention: empty token sets on both sides -> 0.0
    # (not 1.0 like token_sort_ratio).
    assert sa.token_set_ratio("", "") == 0.0
    assert sa.token_set_ratio("   ", "   ") == 0.0


def test_token_set_ratio_one_empty_zero() -> None:
    assert sa.token_set_ratio("abc", "") == 0.0
    assert sa.token_set_ratio("", "abc def") == 0.0


# ---- partial_ratio -----------------------------------------------

def test_partial_ratio_substring_inside_longer() -> None:
    # 'apple' is a substring of 'an apple a day' -> sliding-window
    # finds the perfect alignment.
    assert sa.partial_ratio("apple", "an apple a day") == 1.0
    assert sa.partial_ratio("foo bar", "foo bar baz") == 1.0


def test_partial_ratio_short_buried_in_long_via_lcs_window() -> None:
    # The shorter string has no length-13 perfect alignment, but the
    # LCS substring (' language', length 9) provides the best
    # candidate window.
    short, long = "java language", "python programming language"
    # LCS substring = ' language' (9 chars), entirely contained in
    # short. ratio('java language', ' language') = 2*9 / (13+9) = 18/22.
    assert sa.partial_ratio(short, long) == pytest.approx(18 / 22, abs=1e-12)


def test_partial_ratio_hello_world_world_hello() -> None:
    # Equal-length inputs ('hello world', 'world hello'):
    # sliding window = full-length alignment = indel ratio = 5/11
    # LCS substring = 'hello' or 'world' (length 5) -> ratio of full
    # short against length-5 window: 2*5/(11+5) = 10/16 = 0.625
    assert sa.partial_ratio("hello world", "world hello") == pytest.approx(0.625, abs=1e-12)


def test_partial_ratio_identity() -> None:
    assert sa.partial_ratio("kitten", "kitten") == 1.0


def test_partial_ratio_empty() -> None:
    assert sa.partial_ratio("", "") == 1.0
    assert sa.partial_ratio("abc", "") == 0.0
    assert sa.partial_ratio("", "abc") == 0.0


def test_partial_ratio_symmetric() -> None:
    # partial_ratio is symmetric in a, b because the algorithm picks
    # the shorter / longer roles independently of argument order.
    pairs = [
        ("apple", "an apple a day"),
        ("java language", "python programming language"),
        ("hello world", "world hello"),
    ]
    for a, b in pairs:
        assert sa.partial_ratio(a, b) == pytest.approx(
            sa.partial_ratio(b, a), abs=1e-12
        )


# ---- partial_token_sort_ratio / partial_token_set_ratio ---------

def test_partial_token_sort_ratio_reorderings_score_one() -> None:
    # Token-sort makes the joined strings equal, then partial_ratio
    # of equal strings is 1.0.
    assert sa.partial_token_sort_ratio("foo bar", "bar foo") == 1.0
    assert sa.partial_token_sort_ratio("hello world", "world hello") == 1.0


def test_partial_token_set_ratio_subset_returns_one() -> None:
    # When tokens of one input are a subset of the other, the
    # intersection-only candidate matches identity in partial_ratio.
    assert sa.partial_token_set_ratio("foo bar", "foo bar baz") == 1.0
    assert sa.partial_token_set_ratio("apple", "an apple a day") == 1.0


def test_partial_token_set_ratio_empty_zero() -> None:
    assert sa.partial_token_set_ratio("", "") == 0.0
    assert sa.partial_token_set_ratio("abc", "") == 0.0


# ---- WRatio ------------------------------------------------------

def test_wratio_identity() -> None:
    assert sa.WRatio("kitten", "kitten") == 1.0
    assert sa.WRatio("the quick brown fox", "the quick brown fox") == 1.0


def test_wratio_empty() -> None:
    assert sa.WRatio("", "") == 1.0
    assert sa.WRatio("abc", "") == 0.0
    assert sa.WRatio("", "abc") == 0.0


def test_wratio_unit_interval() -> None:
    import random
    random.seed(7)
    alphabet = "abc def ghi"
    for _ in range(40):
        m, n = random.randrange(0, 20), random.randrange(0, 20)
        a = "".join(random.choice(alphabet) for _ in range(m)).strip()
        b = "".join(random.choice(alphabet) for _ in range(n)).strip()
        r = sa.WRatio(a, b)
        assert 0.0 <= r <= 1.0, (a, b, r)


# ---- rapidfuzz cross-check (parity oracle) ----------------------

# These cases match bit-exactly. The token_sort and token_set
# families always match because they reduce to indel-normalized
# similarity over deterministic tokenisation. partial_ratio matches
# whenever the optimal window length is either ``min(|a|, |b|)``
# or the LCS substring length; the small set of equal-length inputs
# where rapidfuzz finds a tighter shifted window is excluded.

rapidfuzz = pytest.importorskip("rapidfuzz")
from rapidfuzz import fuzz as _rf_fuzz  # noqa: E402


PARITY_CASES = [
    ("kitten", "sitting"),
    ("hello world", "world hello"),
    ("foo bar", "foo bar baz"),
    ("apple", "an apple a day"),
    ("python programming language", "java language"),
    ("fuzzy wuzzy", "wuzzy fuzzy bear"),
    ("a quick brown fox", "fox"),
    ("a quick brown fox", "quick"),
    ("the cat in the hat", "cat hat"),
]


@pytest.mark.parametrize("s1, s2", PARITY_CASES)
def test_token_sort_ratio_matches_rapidfuzz(s1, s2) -> None:
    ours = sa.token_sort_ratio(s1, s2)
    theirs = _rf_fuzz.token_sort_ratio(s1, s2) / 100.0
    assert ours == pytest.approx(theirs, abs=1e-9)


@pytest.mark.parametrize("s1, s2", PARITY_CASES)
def test_token_set_ratio_matches_rapidfuzz(s1, s2) -> None:
    ours = sa.token_set_ratio(s1, s2)
    theirs = _rf_fuzz.token_set_ratio(s1, s2) / 100.0
    assert ours == pytest.approx(theirs, abs=1e-9)


@pytest.mark.parametrize("s1, s2", PARITY_CASES)
def test_partial_ratio_matches_rapidfuzz(s1, s2) -> None:
    ours = sa.partial_ratio(s1, s2)
    theirs = _rf_fuzz.partial_ratio(s1, s2) / 100.0
    assert ours == pytest.approx(theirs, abs=1e-9)


@pytest.mark.parametrize("s1, s2", PARITY_CASES)
def test_partial_token_sort_ratio_matches_rapidfuzz(s1, s2) -> None:
    ours = sa.partial_token_sort_ratio(s1, s2)
    theirs = _rf_fuzz.partial_token_sort_ratio(s1, s2) / 100.0
    assert ours == pytest.approx(theirs, abs=1e-9)


@pytest.mark.parametrize("s1, s2", PARITY_CASES)
def test_partial_token_set_ratio_matches_rapidfuzz(s1, s2) -> None:
    ours = sa.partial_token_set_ratio(s1, s2)
    theirs = _rf_fuzz.partial_token_set_ratio(s1, s2) / 100.0
    assert ours == pytest.approx(theirs, abs=1e-9)


# Equal-length inputs where rapidfuzz finds a window shifted by one
# character — sliding window plus LCS substring is conservative and
# underestimates the rapidfuzz value by a small margin. Tracked but
# not enforced as parity.
PARTIAL_RATIO_DIVERGE_CASES = [
    ("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear"),
    ("the quick brown fox",    "the quick brown dog"),
]


@pytest.mark.parametrize("s1, s2", PARTIAL_RATIO_DIVERGE_CASES)
def test_partial_ratio_conservative_vs_rapidfuzz(s1, s2) -> None:
    ours = sa.partial_ratio(s1, s2)
    theirs = _rf_fuzz.partial_ratio(s1, s2) / 100.0
    # Never overshoots rapidfuzz; gap stays small.
    assert ours <= theirs + 1e-9
    assert theirs - ours < 0.05


# ---- Bytes input -------------------------------------------------

def test_token_sort_ratio_accepts_bytes() -> None:
    # bytes widened as Latin-1, same convention as the C++ engines.
    assert sa.token_sort_ratio(b"hello world", b"world hello") == 1.0


def test_partial_ratio_accepts_bytes() -> None:
    assert sa.partial_ratio(b"apple", b"an apple a day") == 1.0


# ---- Processor callback -----------------------------------------

def test_processor_preprocesses_inputs() -> None:
    # Case-insensitive token sort: pass str.lower.
    assert sa.token_sort_ratio("FOO BAR", "bar foo", processor=str.lower) == 1.0
    assert sa.token_set_ratio("Foo Bar", "BAR FOO", processor=str.lower) == 1.0


# ---- Invalid input ----------------------------------------------

def test_non_str_non_bytes_raises_type_error() -> None:
    with pytest.raises(TypeError, match="must be str or bytes"):
        sa.token_sort_ratio(42, "hello")
    with pytest.raises(TypeError, match="must be str or bytes"):
        sa.partial_ratio("hello", None)
