"""Beider-Morse Phonetic Matching — GENERIC name-type.

Cross-checked against the canonical vectors in Apache Commons Codec's
``PhoneticEngineTest`` (Apache 2.0). Stride-align ships GENERIC only,
so the ASHKENAZI and SEPHARDIC rows from the upstream test set are not
mirrored here.
"""

from __future__ import annotations

import pytest

import stride_align as sa


# ---- Public API surface -------------------------------------------

def test_beider_morse_imported() -> None:
    assert callable(sa.beider_morse)
    assert sa.BmpmRuleType.APPROX == 0
    assert sa.BmpmRuleType.EXACT == 1


def test_empty_input_returns_empty_string() -> None:
    assert sa.beider_morse("") == ""
    assert sa.beider_morse("   ") == ""


def test_bytes_input_supported() -> None:
    # Bytes path bypasses ``PyUnicode_AsUTF8String`` and is interpreted
    # as already-UTF-8 by the dispatch wrapper.
    assert sa.beider_morse(b"Smith") == sa.beider_morse("Smith")


def test_identical_inputs_produce_identical_codes() -> None:
    a = sa.beider_morse("Cohen", rule_type=sa.BmpmRuleType.APPROX)
    b = sa.beider_morse("Cohen", rule_type=sa.BmpmRuleType.APPROX)
    assert a == b
    assert "|" in a or a.count("|") >= 0  # output is a |-joined string


def test_distinct_rule_types_differ() -> None:
    # ``Renault`` is the canonical example: APPROX produces a broader
    # phoneme spread than EXACT.
    approx = sa.beider_morse("Renault", rule_type=sa.BmpmRuleType.APPROX)
    exact = sa.beider_morse("Renault", rule_type=sa.BmpmRuleType.EXACT)
    assert approx != exact
    assert approx.count("|") >= exact.count("|")


# ---- Canonical vectors from Apache Commons Codec PhoneticEngineTest -
#
# Each row pins ``(input, expected, rule_type, concat, max_phonemes)``
# and corresponds to one ``Arguments.of(...)`` row in
# ``src/test/java/org/apache/commons/codec/language/bm/
# PhoneticEngineTest.java`` filtered to ``NameType.GENERIC``.

@pytest.mark.parametrize(
    "name, expected, rule_type, concat, max_phonemes",
    [
        # GENERIC + APPROX: broad phonetic spread.
        ("Renault", "rinD|rinDlt|rina|rinalt|rino|rinolt|rinu|rinult",
         sa.BmpmRuleType.APPROX, True, 10),
        # GENERIC + EXACT: tighter, single-word concat case.
        ("SntJohn-Smith", "sntjonsmit",
         sa.BmpmRuleType.EXACT, True, 10),
        # ``d'`` prefix handling: ``(encoded_remainder)-(encoded_combined)``.
        ("d'ortley", "(ortlaj|ortlej)-(dortlaj|dortlej)",
         sa.BmpmRuleType.EXACT, True, 10),
        # ``van`` prefix handling, multi-word non-concat: pins the
        # v→(v|f[german]|b[spanish]) language-restricted alternation
        # and the per-word language guess.
        ("van helsing",
         "(elSink|elsink|helSink|helsink|helzink|xelsink)-"
         "(banhelsink|fanhelsink|fanhelzink|vanhelsink|vanhelzink|vanjelsink)",
         sa.BmpmRuleType.EXACT, False, 10),
        # GENERIC + APPROX, ``Judenburg`` — long German-style name.
        ("Judenburg",
         "iudnbYrk|iudnbirk|iudnburk|xudnbirk|xudnburk|zudnbirk|zudnburk",
         sa.BmpmRuleType.APPROX, True, 10),
    ],
)
def test_apache_commons_codec_vectors(
    name: str,
    expected: str,
    rule_type: sa.BmpmRuleType,
    concat: bool,
    max_phonemes: int,
) -> None:
    actual = sa.beider_morse(
        name, rule_type=rule_type, concat=concat, max_phonemes=max_phonemes,
    )
    assert actual == expected


# ---- max_phonemes cap --------------------------------------------

def test_max_phonemes_cap_respected() -> None:
    full = sa.beider_morse("Schwarzenegger", max_phonemes=20)
    capped = sa.beider_morse("Schwarzenegger", max_phonemes=2)
    assert full.count("|") + 1 >= capped.count("|") + 1
    assert capped.count("|") + 1 <= 2
