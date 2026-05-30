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


# ---- Metaphone: canonical cases (spec-correct per Philips 1990 /
#      Apache Commons Codec; differs from jellyfish on CH-after-S
#      and GH-at-end-of-word — see metaphone.hpp comment block) ---------------

@pytest.mark.parametrize(
    "name,expected",
    [
        # Classic test set.
        ("Thompson", "0MPSN"),
        ("Catherine", "K0RN"),
        ("Kathryn", "K0RN"),     # same Metaphone as Catherine
        ("Schmidt", "SKMTT"),    # SCH → SK (spec), not SX (jellyfish)
        ("Robert", "RBRT"),
        ("Smith", "SM0"),
        ("Caesar", "KSR"),
        # The H/W-handling siblings.
        ("Ashcraft", "AXKRFT"),
        ("Tymczak", "TMKSK"),
        ("Pfister", "PFSTR"),
        ("Honeyman", "HNMN"),
        ("Lloyd", "LT"),
        ("McDonald", "MKTNLT"),
        ("Hwang", "HWNK"),
        ("Williamson", "WLMSN"),
        ("Vandyke", "FNTK"),
        ("Phillips", "FLPS"),
        ("Xavier", "SFR"),
        ("Zhang", "SHNK"),
        ("Yamaguchi", "YMKX"),
        ("Bobcat", "BBKT"),
        ("Blackwell", "BLKWL"),
        # Initial-pair simplifications (AE-, GN-, KN-, PN-, WR-, WH-, X-).
        ("Aether", "E0R"),
        ("Gnaeus", "NS"),
        ("Knight", "NT"),         # GH dropped (not end of word)
        ("Wright", "RT"),         # GH dropped (not end of word)
        # B at end after M drops.
        ("Lamb", "LM"),
        # T before CH is silent.
        ("Match", "MX"),
        # CH emits X (except after S, where SCH → SK above).
        ("Chess", "XS"),
        ("Quiche", "KX"),
        # GH at end of word is silent (spec); not "KH" (jellyfish).
        ("Hugh", "H"),
        ("Through", "0R"),
        ("Tough", "T"),
        ("Cough", "K"),
        ("Plough", "PL"),
        # GH before a vowel falls through: G emits K, H emits H.
        ("Ghost", "KHST"),
        # GH not at end drops both.
        ("Caught", "KT"),
        # H after vowel and not before vowel drops.
        ("Hour", "HR"),
    ],
)
def test_metaphone_canonical(name: str, expected: str) -> None:
    assert sa.metaphone(name) == expected


def test_metaphone_empty_input() -> None:
    assert sa.metaphone("") == ""


def test_metaphone_no_letters() -> None:
    assert sa.metaphone("12345") == ""
    assert sa.metaphone("   ") == ""


def test_metaphone_bytes_input() -> None:
    assert sa.metaphone(b"Robert") == "RBRT"
    assert sa.metaphone(b"") == ""


def test_metaphone_case_insensitive() -> None:
    assert sa.metaphone("ROBERT") == sa.metaphone("robert") == "RBRT"


def test_metaphone_non_ascii_skipped() -> None:
    # Non-ASCII codepoints drop out before encoding.
    assert sa.metaphone("café") == sa.metaphone("cf")
    assert sa.metaphone("你好") == ""


def test_metaphone_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="str or bytes"):
        sa.metaphone(12345)
    with pytest.raises(TypeError):
        sa.metaphone(None)


def test_metaphone_equal() -> None:
    # Catherine / Kathryn collide.
    assert sa.metaphone_equal("Catherine", "Kathryn")
    # Distinct names don't.
    assert not sa.metaphone_equal("Robert", "Smith")
    # Both empty → False (no valid code).
    assert not sa.metaphone_equal("", "")


def test_metaphone_equal_case_insensitive() -> None:
    assert sa.metaphone_equal("ROBERT", "robert")


# ---- Metaphone variants ---------------------------------------------------

def test_metaphone_variant_default_is_philips() -> None:
    # Default and explicit PHILIPS are equivalent.
    for name in ("Schmidt", "Hugh", "Through", "Wright", "Caught", "Ghost"):
        assert sa.metaphone(name) == sa.metaphone(name, variant=sa.MetaphoneVariant.PHILIPS)


@pytest.mark.parametrize(
    "name,philips,jellyfish",
    [
        # SCH-after-S rule: Philips emits K, jellyfish emits X.
        ("Schmidt", "SKMTT", "SXMTT"),
        # GH at end of word: Philips drops both, jellyfish emits K + leaves H.
        ("Hugh", "H", "HKH"),
        ("Through", "0R", "0RKH"),
        ("Tough", "T", "TKH"),
        ("Cough", "K", "KKH"),
        ("Plough", "PL", "PLKH"),
        # Cases where both variants agree (sanity).
        ("Robert", "RBRT", "RBRT"),
        ("Catherine", "K0RN", "K0RN"),
        ("Knight", "NT", "NT"),
        ("Wright", "RT", "RT"),
        ("Caught", "KT", "KT"),
        ("Ghost", "KHST", "KHST"),
    ],
)
def test_metaphone_variant_outputs(name: str, philips: str, jellyfish: str) -> None:
    assert sa.metaphone(name, variant=sa.MetaphoneVariant.PHILIPS) == philips
    assert sa.metaphone(name, variant=sa.MetaphoneVariant.JELLYFISH) == jellyfish


def test_metaphone_variant_equal_forwards_kwarg() -> None:
    # Schmidt collides with Smith only under the PHILIPS variant
    # (SKMTT vs SM0 are different; SXMTT vs SM0 are different too).
    # Pick a case where the variants actually disagree on a collision.
    # Hugh and "Hue" — under PHILIPS both encode to "H"; under jellyfish
    # Hugh is "HKH" and Hue is "H", so they differ.
    assert sa.metaphone_equal("Hugh", "Hue", variant=sa.MetaphoneVariant.PHILIPS)
    assert not sa.metaphone_equal("Hugh", "Hue", variant=sa.MetaphoneVariant.JELLYFISH)


# ---- NYSIIS (Taft 1970, modern non-truncating form) ---------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        # Classic English surnames — pinned against jellyfish.
        ("Watkins", "WATCAN"),
        ("Wilkins", "WALCAN"),
        ("Wilkinson", "WALCANSAN"),
        ("Robert", "RABAD"),
        ("Smith", "SNAT"),
        ("Schmidt", "SNAD"),
        ("Catherine", "CATARAN"),
        ("Kathryn", "CATRYN"),
        ("Knight", "NAGT"),       # KN -> NN prefix; the doubled N dedupes
        ("Wright", "WRAGT"),
        ("Caesar", "CASAR"),
        ("Macarthur", "MCARTAR"),  # MAC -> MCC prefix
        ("Phillips", "FALAP"),     # PH -> FF prefix
        ("Pfister", "FASTAR"),     # PF -> FF prefix
        ("Schwartz", "SWART"),     # SCH -> SSS prefix
        # Names that probe specific rules.
        ("Hawthorne", "HATARN"),
        ("Yarborough", "YARBARAG"),
        ("Thompson", "TANPSAN"),
        ("Anderson", "ANDARSAN"),
        ("Johnson", "JANSAN"),     # H between vowel and consonant drops
        ("Williams", "WALAN"),
        ("Hugh", "HAG"),
        ("Through", "TRAG"),
        ("Caught", "CAGT"),
        ("Plough", "PLAG"),
        ("Lloyd", "LAYD"),
        ("Honeyman", "HANAYNAN"),
        ("McDonald", "MCDANALD"),
        ("Sarah", "SAR"),
        ("Michael", "MACAL"),
        ("Hannah", "HAN"),
        ("Joshua", "JAS"),
        ("William", "WALAN"),
    ],
)
def test_nysiis_canonical(name: str, expected: str) -> None:
    assert sa.nysiis(name) == expected


def test_nysiis_empty_input() -> None:
    assert sa.nysiis("") == ""


def test_nysiis_no_letters() -> None:
    assert sa.nysiis("12345") == ""


def test_nysiis_bytes_input() -> None:
    assert sa.nysiis(b"Robert") == "RABAD"
    assert sa.nysiis(b"") == ""


def test_nysiis_case_insensitive() -> None:
    assert sa.nysiis("ROBERT") == sa.nysiis("robert") == "RABAD"


def test_nysiis_non_ascii_skipped() -> None:
    assert sa.nysiis("café") == sa.nysiis("caf")
    assert sa.nysiis("你好") == ""


def test_nysiis_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="str or bytes"):
        sa.nysiis(12345)


def test_nysiis_equal() -> None:
    # Sound-alikes NYSIIS collides intentionally.
    assert sa.nysiis_equal("Catherine", "Catharine")   # both CATARAN
    assert sa.nysiis_equal("Watkins", "Watkin")        # both WATCAN (trailing S drops)
    # Distinct names don't collide.
    assert not sa.nysiis_equal("Robert", "Smith")
    assert not sa.nysiis_equal("Robert", "Roberto")    # RABAD vs RABART


# ---- Match Rating Approach (Moore 1977) -------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("Robert", "RBRT"),
        ("Rupert", "RPRT"),
        ("Smith", "SMTH"),
        ("Smyth", "SMYTH"),
        ("Schmidt", "SCHMDT"),
        ("Catherine", "CTHRN"),
        ("Kathryn", "KTHRYN"),
        ("Christopher", "CHRPHR"),    # 8 chars deduped -> first 3 + last 3
        ("Williams", "WLMS"),
        ("Williamson", "WLMSN"),
        ("Jonathan", "JNTHN"),
        ("John", "JHN"),
        ("Hannah", "HNH"),
        ("Sarah", "SRH"),
        ("Michael", "MCHL"),
        ("Watkins", "WTKNS"),
        ("Wilkinson", "WLKNSN"),
        ("Lloyd", "LYD"),         # consecutive LL collapses
        ("Honeyman", "HNYMN"),
        ("McDonald", "MCDNLD"),
        ("Caesar", "CSR"),
        ("Aether", "ATHR"),
        ("Knight", "KNGHT"),
        ("Wright", "WRGHT"),
    ],
)
def test_match_rating_codex_canonical(name: str, expected: str) -> None:
    assert sa.match_rating_codex(name) == expected


def test_match_rating_codex_empty_input() -> None:
    assert sa.match_rating_codex("") == ""


def test_match_rating_codex_no_letters() -> None:
    assert sa.match_rating_codex("12345") == ""


def test_match_rating_codex_bytes_input() -> None:
    assert sa.match_rating_codex(b"Robert") == "RBRT"
    assert sa.match_rating_codex(b"") == ""


def test_match_rating_codex_case_insensitive() -> None:
    assert sa.match_rating_codex("ROBERT") == sa.match_rating_codex("robert") == "RBRT"


def test_match_rating_codex_non_ascii_skipped() -> None:
    assert sa.match_rating_codex("café") == sa.match_rating_codex("caf")
    assert sa.match_rating_codex("你好") == ""


def test_match_rating_codex_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="str or bytes"):
        sa.match_rating_codex(12345)


def test_match_rating_codex_keeps_first_vowel() -> None:
    # The first letter is preserved even when it's a vowel.
    assert sa.match_rating_codex("Aether") == "ATHR"
    assert sa.match_rating_codex("Oscar") == "OSCR"


def test_match_rating_codex_truncation_first_three_last_three() -> None:
    # Names whose deduped consonant skeleton exceeds 6 chars get
    # collapsed to "first three + last three".
    assert sa.match_rating_codex("Christopher") == "CHRPHR"
    # Same-shape input pinned long enough to exercise truncation.
    assert len(sa.match_rating_codex("Williamsonburg")) == 6


# Comparator: MRA-true match cases (both pairs sound similar).
@pytest.mark.parametrize(
    "a,b",
    [
        ("Robert", "Rupert"),
        ("Smith", "Smyth"),
        ("Catherine", "Kathryn"),
        ("Williams", "Williamson"),
        ("Knight", "Night"),
        ("John", "Jon"),
        ("Sarah", "Sara"),
        ("Christopher", "Chris"),
        ("Lloyd", "Floyd"),
        ("Hannah", "Hana"),
    ],
)
def test_match_rating_compare_match(a: str, b: str) -> None:
    assert sa.match_rating_compare(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        ("Robert", "Smith"),
        ("Catherine", "Robert"),
        ("Sarah", "Robert"),
    ],
)
def test_match_rating_compare_distinct(a: str, b: str) -> None:
    assert not sa.match_rating_compare(a, b)


def test_match_rating_compare_empty() -> None:
    # Empty codex on either side yields no match.
    assert not sa.match_rating_compare("", "Robert")
    assert not sa.match_rating_compare("Robert", "")
    assert not sa.match_rating_compare("", "")


def test_match_rating_compare_symmetric() -> None:
    # Compare is symmetric.
    assert sa.match_rating_compare("Robert", "Rupert") == sa.match_rating_compare("Rupert", "Robert")
    assert sa.match_rating_compare("Smith", "Schmidt") == sa.match_rating_compare("Schmidt", "Smith")


def test_match_rating_compare_bytes_and_str() -> None:
    # bytes and str interchangeable on either side.
    assert sa.match_rating_compare(b"Robert", "Rupert")
    assert sa.match_rating_compare("Catherine", b"Kathryn")


# ---- Caverphone 2 (Hood 2004) --------------------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        # All from the Apache Commons Codec Caverphone2Test reference set.
        ("Stevenson", "STFNSN1111"),
        ("Peter", "PTA1111111"),
        ("Peady", "PTA1111111"),
        ("Thompson", "TMPSN11111"),
        ("Smith", "SMT1111111"),
        ("Schmidt", "SKMT111111"),
        # All of these encode to "AT11111111" per the published spec.
        ("add", "AT11111111"),
        ("aid", "AT11111111"),
        ("at", "AT11111111"),
        ("art", "AT11111111"),
        ("eat", "AT11111111"),
        ("earth", "AT11111111"),
        ("head", "AT11111111"),
        ("hit", "AT11111111"),
        ("hot", "AT11111111"),
        ("hold", "AT11111111"),
        ("hard", "AT11111111"),
        ("heart", "AT11111111"),
        ("it", "AT11111111"),
        ("out", "AT11111111"),
        ("old", "AT11111111"),
        # Final-mb rule + double-mb sanity.
        ("mb", "M111111111"),
        ("mbmb", "MPM1111111"),
        # The RTA family.
        ("rather", "RTA1111111"),
        ("ready", "RTA1111111"),
        ("writer", "RTA1111111"),
        # The SSA family.
        ("social", "SSA1111111"),
        # The APA family — exercises the "L between vowels" rule
        # subtly: "able" drops L (l-3 at end, not 3-l-3); "appear"
        # has no L.
        ("able", "APA1111111"),
        ("appear", "APA1111111"),
        # KLN family member where L is preserved between vowel markers.
        ("Cailean", "KLN1111111"),
    ],
)
def test_caverphone_canonical(name: str, expected: str) -> None:
    assert sa.caverphone(name) == expected


def test_caverphone_empty_input() -> None:
    # Empty input still produces a 10-character pad code.
    assert sa.caverphone("") == "1111111111"


def test_caverphone_no_letters() -> None:
    # Non-letter input behaves like empty.
    assert sa.caverphone("12345") == "1111111111"


def test_caverphone_bytes_input() -> None:
    assert sa.caverphone(b"Stevenson") == "STFNSN1111"
    assert sa.caverphone(b"") == "1111111111"


def test_caverphone_case_insensitive() -> None:
    assert sa.caverphone("STEVENSON") == sa.caverphone("stevenson") == "STFNSN1111"


def test_caverphone_non_ascii_skipped() -> None:
    assert sa.caverphone("café") == sa.caverphone("caf")
    assert sa.caverphone("你好") == "1111111111"


def test_caverphone_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="str or bytes"):
        sa.caverphone(12345)


def test_caverphone_fixed_length_10() -> None:
    # Every output is exactly 10 characters; long inputs truncate.
    for name in ("a", "Stevenson", "Christopher", "Antidisestablishmentarianism"):
        assert len(sa.caverphone(name)) == 10
    # Smith / Smyth diverge here (Y is treated as non-vowel by
    # this NYSIIS variant, so SMYTH -> SNYT vs SMITH -> SNAT).
    assert not sa.nysiis_equal("Smith", "Smyth")
    # Both empty -> False (no valid code).
    assert not sa.nysiis_equal("", "")


# ---------------------------------------------------------------------------
# Double Metaphone
# ---------------------------------------------------------------------------

# Reference data: cross-checked against the ``doublemetaphone`` PyPI
# package (a direct port of Apache Commons Codec's DoubleMetaphone)
# and against published Apache Commons Codec test vectors. Where the
# ``metaphone`` PyPI package disagrees, we exercise both variants
# explicitly under DoubleMetaphoneVariant.PYTHON.


@pytest.mark.parametrize(
    "name,primary,alternate",
    [
        # Single-encoding names — alternate is empty.
        ("Smith",    "SM0",  "XMT"),
        ("Robert",   "RPRT", ""),
        ("Knight",   "NT",   ""),
        ("Wright",   "RT",   ""),
        ("Caesar",   "SSR",  ""),
        ("Lloyd",    "LT",   ""),
        ("Lamb",     "LMP",  ""),
        ("Match",    "MX",   ""),
        ("Chess",    "XS",   ""),
        ("Honeyman", "HNMN", ""),
        # Names with a second pronunciation: alternate diverges.
        ("Schmidt",  "XMT",  "SMT"),
        ("Schwartz", "XRTS", "XFRTS"),
        ("Pawlowski","PLSK", "PLFSK"),
        # GH-after-vowel cases where Apache Commons / DoubleMetaphone
        # PyPI agree that GH is silent. The ``metaphone`` PyPI port
        # disagrees and gets covered by the variant test below.
        ("Hugh",     "H",    ""),
        ("High",     "H",    ""),
        # GH that ends a -OUGH/-AUGH word emits 'F'.
        ("Tough",    "TF",   ""),
        ("Cough",    "KF",   ""),
        # Silent-start: silent-K, silent-W, silent-G.
        ("Gnaeus",   "NS",   ""),
        # Slavic / Germanic ending TZ -> "S"/"TS" was the bug we
        # caught — primary should be a single S, not doubled.
        ("Tymczak",  "TMSK", "TMXK"),
    ],
)
def test_double_metaphone_canonical(
    name: str, primary: str, alternate: str
) -> None:
    assert sa.double_metaphone(name) == (primary, alternate)


def test_double_metaphone_empty_input() -> None:
    assert sa.double_metaphone("") == ("", "")


def test_double_metaphone_no_letters() -> None:
    assert sa.double_metaphone("12345") == ("", "")


def test_double_metaphone_bytes_input() -> None:
    assert sa.double_metaphone(b"Smith") == ("SM0", "XMT")


def test_double_metaphone_case_insensitive() -> None:
    assert sa.double_metaphone("SMITH") == sa.double_metaphone("smith")


def test_double_metaphone_non_ascii_skipped() -> None:
    # Non-ASCII codepoints are stripped before encoding.
    assert sa.double_metaphone("Müller") == sa.double_metaphone("Mller")


def test_double_metaphone_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="str or bytes"):
        sa.double_metaphone(12345)


def test_double_metaphone_max_length_truncates() -> None:
    # The classical Philips form caps both codes at 4.
    primary, alternate = sa.double_metaphone("Christopher", max_length=4)
    assert len(primary) <= 4
    assert len(alternate) <= 4
    # The default cap (64) leaves the full code intact.
    primary_full, _ = sa.double_metaphone("Christopher")
    assert len(primary_full) > 4
    assert primary_full.startswith(primary)


def test_double_metaphone_alternate_empty_when_same_as_primary() -> None:
    # Names with no second pronunciation have an empty alternate
    # rather than a duplicate of primary — callers can use truthy
    # tests like ``if alt:``.
    _, alt = sa.double_metaphone("Robert")
    assert alt == ""


def test_double_metaphone_variant_python_bug_compat() -> None:
    # The ``metaphone`` PyPI package has a missing-else bug in its
    # GH branch that doubles the prior letter for "Hugh"/"High".
    # DoubleMetaphoneVariant.PYTHON reproduces that exactly for
    # callers cross-checking against that library.
    assert sa.double_metaphone(
        "Hugh", variant=sa.DoubleMetaphoneVariant.PYTHON
    ) == ("HH", "")
    assert sa.double_metaphone(
        "High", variant=sa.DoubleMetaphoneVariant.PYTHON
    ) == ("HH", "")
    # Names without the GH-after-vowel-at-start pattern are
    # unaffected by the variant flag.
    assert sa.double_metaphone(
        "Smith", variant=sa.DoubleMetaphoneVariant.PYTHON
    ) == sa.double_metaphone("Smith")


def test_double_metaphone_variant_default_is_commons() -> None:
    # Default behaviour matches DoubleMetaphoneVariant.COMMONS.
    assert sa.double_metaphone("Hugh") == sa.double_metaphone(
        "Hugh", variant=sa.DoubleMetaphoneVariant.COMMONS
    )
    assert sa.DoubleMetaphoneVariant.COMMONS == 0


def test_double_metaphone_variant_intenum() -> None:
    # The enum exposes COMMONS=0 and PYTHON=1 as int-coercible
    # values — matches the MetaphoneVariant pattern.
    assert int(sa.DoubleMetaphoneVariant.COMMONS) == 0
    assert int(sa.DoubleMetaphoneVariant.PYTHON) == 1


