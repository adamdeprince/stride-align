"""Phonetic encoders — Phase D.2.

Holds Soundex today; Metaphone and Double Metaphone tests land
here too as those encoders ship.

Soundex test vectors are the standard set: Knuth TAOCP vol. 3
examples, the US Census reference cases, and the classic
``Ashcraft`` / ``Tymczak`` cases that pin the H/W transparency
rule. Cross-checked against jellyfish's implementation.
"""

from __future__ import annotations

import pytest

import stride_align as sa


# ---- Soundex: canonical reference cases -------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        # Knuth TAOCP vol 3, Sec 6 Sortable Names — these are the
        # canonical Soundex test vectors quoted by basically every
        # implementation.
        ("Robert", "R163"),
        ("Rupert", "R163"),
        ("Rubin", "R150"),
        ("Ashcraft", "A261"),
        ("Ashcroft", "A261"),
        ("Tymczak", "T522"),
        ("Pfister", "P236"),
        ("Honeyman", "H555"),
        # Short / single-letter pads with zeros.
        ("R", "R000"),
        ("Lee", "L000"),
        # Lowercase round-trips identically.
        ("robert", "R163"),
        ("RUPERT", "R163"),
        # Equal-code prefix at start: "Smith" S-M-I-T-H, M=5, T=3.
        ("Smith", "S530"),
        # Punctuation / digits inside the name are ignored.
        ("O'Brien", "O165"),
        ("Mc-Donald", "M235"),
    ],
)
def test_soundex_canonical(name: str, expected: str) -> None:
    assert sa.soundex(name) == expected


# ---- Soundex: edge cases ----------------------------------------

def test_soundex_empty_input() -> None:
    assert sa.soundex("") == ""


def test_soundex_no_letters() -> None:
    assert sa.soundex("12345") == ""
    assert sa.soundex(",,,") == ""
    assert sa.soundex(" \n\t") == ""


def test_soundex_bytes_input() -> None:
    assert sa.soundex(b"Robert") == "R163"
    assert sa.soundex(b"") == ""
    assert sa.soundex(b"123") == ""


def test_soundex_non_ascii_skipped() -> None:
    # The accented letters are not ASCII, so they're skipped; the
    # ASCII letters that remain dictate the code. Callers wanting
    # accent-folding pre-normalise before calling.
    assert sa.soundex("café") == sa.soundex("cf")          # 'a' and 'é' skipped
    assert sa.soundex("Müller") == sa.soundex("Mller")
    # Pure CJK: no ASCII letters, empty output.
    assert sa.soundex("你好世界") == ""


def test_soundex_h_w_transparency() -> None:
    # The classic H/W transparency cases.
    # "Ashcraft": A-S-H-C-R-A-F-T. S=2, H transparent, C=2 → collapse.
    assert sa.soundex("Ashcraft") == "A261"
    # "Tymczak": T-Y-M-C-Z-A-K. C=2, Z=2 → collapse, K=2 still emits
    # because vowel A breaks adjacency.
    assert sa.soundex("Tymczak") == "T522"
    # Manual H-between-same-coded: "Bobcat". B=1, B=1 collapse,
    # C=2, T=3.
    assert sa.soundex("Bobcat") == "B123"


def test_soundex_truncates_at_four() -> None:
    # Long names truncate; padding only fires when output < 4.
    assert sa.soundex("Blackwell") == "B424"
    assert sa.soundex("Strzelinski") == "S362"


def test_soundex_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="str or bytes"):
        sa.soundex(12345)
    with pytest.raises(TypeError):
        sa.soundex(["R", "o", "b", "e", "r", "t"])
    with pytest.raises(TypeError):
        sa.soundex(None)


# ---- Soundex: equal-comparison convenience ----------------------

def test_soundex_equal_matches() -> None:
    assert sa.soundex_equal("Robert", "Rupert")     # both R163
    assert sa.soundex_equal("Smith", "Smyth")       # both S530
    assert sa.soundex_equal("ashcraft", "Ashcroft") # both A261, case-insensitive


def test_soundex_equal_mismatches() -> None:
    assert not sa.soundex_equal("Robert", "Smith")
    assert not sa.soundex_equal("Robert", "")
    assert not sa.soundex_equal("", "")             # both empty → False (no valid code)


def test_soundex_equal_bytes_and_str() -> None:
    # Caller can mix bytes and str on either side.
    assert sa.soundex_equal(b"Robert", "Rupert")
