"""Longest Common Subsequence / Substring (Phase D.4).

Two related but distinct algorithms:

* **Subsequence** — ``lcs_length``. Characters need not be contiguous.
  Same recurrence as Indel; the closed-form relation
  ``indel = |a| + |b| - 2 * lcs_length`` lets us double-check against
  the existing ``indel_score``.

* **Substring** — ``lcs_substring_length`` / ``lcs_substring``.
  Characters must be contiguous. Different DP; the substring is
  recovered with a single slice on the running-max position in ``a``.
"""

from __future__ import annotations

import pytest

import stride_align as sa


# ---- LCS subsequence ---------------------------------------------

@pytest.mark.parametrize(
    "a, b, expected",
    [
        # Textbook example — Cormen et al, Introduction to Algorithms.
        ("ABCBDAB", "BDCAB", 4),                # BCAB
        # Wikipedia / Hirschberg 1975.
        ("AGGTAB", "GXTXAYB", 4),               # GTAB
        # Identical strings — LCS is the whole string.
        ("hello", "hello", 5),
        # One side empty.
        ("", "abc", 0),
        ("abc", "", 0),
        ("", "", 0),
        # Disjoint alphabets.
        ("abc", "xyz", 0),
        # Repeated characters that don't align.
        ("aabb", "bbaa", 2),                    # aa or bb
        # Subset (one is a subsequence of the other).
        ("abc", "axbxc", 3),                    # abc
    ],
)
def test_lcs_length(a: str, b: str, expected: int) -> None:
    assert sa.lcs_length(a, b) == expected


def test_lcs_length_symmetric() -> None:
    # LCS is symmetric in its two arguments.
    assert sa.lcs_length("ABCBDAB", "BDCAB") == sa.lcs_length("BDCAB", "ABCBDAB")


def test_lcs_length_matches_indel_relation() -> None:
    # ``indel = |a| + |b| - 2 * lcs_length`` is the defining
    # relationship between Indel distance and LCS.
    pairs = [
        ("kitten", "sitting"),
        ("hello", "world"),
        ("ABCBDAB", "BDCAB"),
        ("", "anything"),
    ]
    for a, b in pairs:
        lcs = sa.lcs_length(a, b)
        indel = sa.indel_score(a, b)
        assert indel == len(a) + len(b) - 2 * lcs, (a, b, lcs, indel)


# ---- LCS substring ----------------------------------------------

@pytest.mark.parametrize(
    "a, b, expected_len, expected_str",
    [
        # The classic ABCBDAB / BDCAB example — substring is much
        # smaller than the subsequence.
        ("ABCBDAB", "BDCAB", 2, "AB"),
        # ``Müller`` and ``Mueller`` share the suffix ``ller``.
        ("Müller", "Mueller", 4, "ller"),
        # Identical strings: substring is the whole string.
        ("hello", "hello", 5, "hello"),
        # No shared characters.
        ("abc", "xyz", 0, ""),
        # Empty inputs.
        ("", "abc", 0, ""),
        ("abc", "", 0, ""),
        # Tiebreaking — when multiple substrings tie at the maximum
        # length, the FIRST occurrence in ``a`` is returned (matches
        # ``str.find`` convention).
        ("hello world", "world hello", 5, "hello"),
        # Common prefix.
        ("prefixA", "prefixB", 6, "prefix"),
        # Common suffix.
        ("Asuffix", "Bsuffix", 6, "suffix"),
        # Middle substring.
        ("xxABCyy", "zzABCzz", 3, "ABC"),
    ],
)
def test_lcs_substring(
    a: str, b: str, expected_len: int, expected_str: str
) -> None:
    assert sa.lcs_substring_length(a, b) == expected_len
    assert sa.lcs_substring(a, b) == expected_str


def test_lcs_substring_bytes_input_returns_bytes() -> None:
    # When both inputs are bytes the result is bytes.
    assert sa.lcs_substring(b"hello world", b"world hello") == b"hello"


def test_lcs_substring_str_and_bytes_are_consistent_on_ascii() -> None:
    # For ASCII the str and bytes forms of an input give the same
    # numeric length (the type of the returned substring differs).
    assert sa.lcs_substring_length("ABCDE", "BCDEF") == \
           sa.lcs_substring_length(b"ABCDE", b"BCDEF")


def test_lcs_substring_unicode_codepoints() -> None:
    # Non-ASCII codepoints are first-class — the engine works in
    # codepoint space, not bytes.
    assert sa.lcs_substring("αβγδε", "βγδ") == "βγδ"
    assert sa.lcs_substring_length("αβγδε", "βγδ") == 3


# ---- Edge cases --------------------------------------------------

def test_lcs_substring_disjoint_returns_empty() -> None:
    assert sa.lcs_substring("abc", "xyz") == ""
    assert sa.lcs_substring_length("abc", "xyz") == 0


def test_lcs_substring_single_char_match() -> None:
    # The longest common substring of "abc" and "b" is just "b".
    assert sa.lcs_substring("abc", "b") == "b"
    assert sa.lcs_substring_length("abc", "b") == 1


def test_lcs_lengths_are_non_negative_ints() -> None:
    # API contract: lengths come back as Python int (non-negative).
    for a, b in [("foo", "bar"), ("", ""), ("x", "x")]:
        n = sa.lcs_length(a, b)
        m = sa.lcs_substring_length(a, b)
        assert isinstance(n, int) and n >= 0
        assert isinstance(m, int) and m >= 0


def test_lcs_substring_length_le_lcs_length() -> None:
    # By definition, a substring is also a subsequence — the substring
    # length is bounded above by the subsequence length.
    pairs = [
        ("ABCBDAB", "BDCAB"),
        ("hello world", "world hello"),
        ("kitten", "sitting"),
        ("Müller", "Mueller"),
    ]
    for a, b in pairs:
        assert sa.lcs_substring_length(a, b) <= sa.lcs_length(a, b)
