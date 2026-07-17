"""Compatibility tests for ``stride_align.thefuzz``.

Pinned tests run with no third-party package installed. When TheFuzz 0.22.1
is available, the second half of the module runs differential batteries
against its public API.
"""

from __future__ import annotations

import inspect
import random

import pytest

import stride_align.thefuzz as shim
from stride_align.thefuzz import fuzz, process, utils

try:
    import thefuzz as upstream
    from thefuzz import fuzz as up_fuzz
    from thefuzz import process as up_process
    from thefuzz import utils as up_utils
except ImportError:  # pragma: no cover - exercised in dependency-minimal CI
    upstream = up_fuzz = up_process = up_utils = None


FUZZ_NAMES = [
    "ratio",
    "partial_ratio",
    "token_sort_ratio",
    "partial_token_sort_ratio",
    "token_set_ratio",
    "partial_token_set_ratio",
    "QRatio",
    "UQRatio",
    "WRatio",
    "UWRatio",
]


def test_package_surface_and_target_version() -> None:
    assert shim.__version__ == "0.22.1"
    assert shim.__all__ == ["fuzz", "process", "utils"]
    assert fuzz.__all__ == FUZZ_NAMES
    assert process.default_scorer is fuzz.WRatio
    assert process.default_processor is utils.full_process


def test_function_signatures_match_legacy_surface() -> None:
    assert str(inspect.signature(fuzz.ratio)) == "(s1, s2)"
    assert str(inspect.signature(fuzz.token_sort_ratio)) == (
        "(s1, s2, force_ascii=True, full_process=True)"
    )
    assert str(inspect.signature(fuzz.UWRatio)) == "(s1, s2, full_process=True)"
    assert list(inspect.signature(process.extractBests).parameters) == [
        "query",
        "choices",
        "processor",
        "scorer",
        "score_cutoff",
        "limit",
    ]


@pytest.mark.parametrize(
    "name,left,right,expected",
    [
        ("ratio", "this is a test", "this is a test!", 97),
        ("partial_ratio", "this is a test", "this is a test!", 100),
        ("token_sort_ratio", "fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear", 100),
        ("token_set_ratio", "fuzzy was a bear", "fuzzy fuzzy was a bear", 100),
        ("partial_token_sort_ratio", "fuzzy was a bear", "wuzzy fuzzy was a bear", 100),
        ("QRatio", "New York!", "new york", 100),
        ("WRatio", "foo bar baz", "foo bar", 90),
    ],
)
def test_pinned_fuzz_examples(name, left, right, expected) -> None:
    result = getattr(fuzz, name)(left, right)
    assert result == expected
    assert type(result) is int


@pytest.mark.parametrize("name", FUZZ_NAMES)
def test_none_returns_zero(name) -> None:
    assert getattr(fuzz, name)(None, "text") == 0


def test_empty_input_conventions() -> None:
    assert fuzz.ratio("", "") == 100
    assert fuzz.partial_ratio("", "") == 100
    assert fuzz.token_sort_ratio("", "") == 100
    assert fuzz.token_set_ratio("", "") == 0
    assert fuzz.QRatio("", "") == 0
    assert fuzz.WRatio("", "") == 0


def test_force_ascii_and_unicode_aliases() -> None:
    assert fuzz.QRatio("Café", "Cafe") == 86
    assert fuzz.UQRatio("Café", "Cafe") == 75
    assert fuzz.UWRatio("東京", "東京") == 100
    assert fuzz.UQRatio is not fuzz.QRatio


def test_half_integer_rounding_matches_thefuzz() -> None:
    # These land exactly on a bankers-rounding boundary. The first is one
    # ulp above 66.5 upstream; the second is exactly 28.5.
    assert fuzz.WRatio("ΣΟ_ΣİİΟΟΣΣΣ_İ__ΣİΣİİΣΟ_İΣ", "_ΣΟΟΣΟ_ΣİΣΟ_") == 67
    assert fuzz.WRatio("aaayy", "bc cxa bc ") == 28


def test_arbitrary_hashable_sequences_use_native_token_path() -> None:
    assert fuzz.ratio(["a", 1], ["a", 2]) == 50
    assert fuzz.partial_ratio(["a", 1], ["z", "a", 1, "q"]) == 100
    assert fuzz.token_sort_ratio(["b", " ", "a"], ["a", " ", "b"], full_process=False) == 100
    assert fuzz.WRatio(bytearray(b"b a"), bytearray(b"a b"), full_process=False) == 95


def test_unhashable_sequence_elements_raise() -> None:
    with pytest.raises(TypeError):
        fuzz.ratio([["unhashable"]], [["unhashable"]])


@pytest.mark.parametrize(
    "value,force_ascii,expected",
    [
        ("Hello, World!", False, "hello  world"),
        ("Café—東京", False, "café 東京"),
        ("Café—東京", True, "caf 東京"),
        ("İstanbul", False, "istanbul"),
        ("ΟΣ", False, "οσ"),
        ("foo_bar", False, "foo bar"),
        (b"caf\xe9", False, "café"),
        (123, True, "123"),
    ],
)
def test_full_process(value, force_ascii, expected) -> None:
    assert utils.full_process(value, force_ascii) == expected


def test_utils_legacy_edge_cases() -> None:
    assert utils.ascii_only("Café東京") == "Caf東京"
    assert len(utils.translation_table) == 128
    with pytest.raises(TypeError, match="sentence must be a String"):
        utils.full_process(123)


def test_process_extract_sequence_and_mapping_shapes() -> None:
    choices = ["Atlanta Falcons", "New York Jets", "New York Giants", "Dallas Cowboys"]
    assert process.extract("new york jets", choices, limit=2) == [
        ("New York Jets", 100),
        ("New York Giants", 79),
    ]
    assert process.extractOne("cowboys", choices) == ("Dallas Cowboys", 90)

    mapping = {"falcons": "Atlanta Falcons", "jets": "New York Jets"}
    assert process.extractOne("new york jets", mapping) == (
        "New York Jets",
        100,
        "jets",
    )


def test_process_extract_without_order_and_none_choices() -> None:
    assert list(process.extractWithoutOrder("a", ["A", None, "b"])) == [
        ("A", 100),
        ("b", 0),
    ]
    assert process.extractOne("a", [], score_cutoff=50) is None


def test_process_custom_scorer_is_not_integer_rounded() -> None:
    def custom(left, right):
        return 10.5 if left == right else -2

    assert process.extract("a", ["a", "b"], processor=None, scorer=custom) == [("a", 10.5)]


def test_process_cutoff_and_limit_validation() -> None:
    assert process.extractBests("a", ["a", "b"], limit=0) == []
    assert process.extractBests("a", ["a", "b"], limit=1.2) == [("a", 100)]
    with pytest.raises(RuntimeError, match="vector"):
        process.extractBests("a", ["a"], limit=-1)
    with pytest.raises(TypeError, match="range of 0.0 - 100.0"):
        process.extractOne("a", ["a"], score_cutoff=101)


def test_dedupe_representatives_and_noop_identity() -> None:
    values = [
        "Frodo Baggin",
        "Frodo Baggins",
        "F. Baggins",
        "Samwise G.",
        "Gandalf",
        "Bilbo Baggins",
    ]
    assert set(process.dedupe(values)) == {
        "Frodo Baggins",
        "Samwise G.",
        "Gandalf",
    }
    unique = ["alpha", "beta"]
    assert process.dedupe(unique, threshold=100) is unique


# ---------------------------------------------------------------------------
# Optional TheFuzz 0.22.1 differential oracle
# ---------------------------------------------------------------------------


requires_upstream = pytest.mark.skipif(
    upstream is None,
    reason="install thefuzz==0.22.1 to run the differential oracle",
)


@requires_upstream
@pytest.mark.parametrize(
    "left,right",
    [
        ("this is a test", "this is a test!"),
        ("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear"),
        ("Café", "Cafe"),
        ("東京 大学", "大学 東京"),
        ("", ""),
        ("", "x"),
        (None, None),
        (["a", 1], ["a", 2]),
        (bytearray(b"b a"), bytearray(b"a b")),
    ],
)
@pytest.mark.parametrize("name", FUZZ_NAMES)
def test_fuzz_fixed_differential(name, left, right) -> None:
    shim_fn = getattr(fuzz, name)
    upstream_fn = getattr(up_fuzz, name)
    if name in {"ratio", "partial_ratio"}:
        assert shim_fn(left, right) == upstream_fn(left, right)
    else:
        assert shim_fn(left, right, full_process=False) == upstream_fn(
            left, right, full_process=False
        )


@requires_upstream
def test_fuzz_random_string_differential() -> None:
    rng = random.Random(42)
    alphabets = ["abc xyz", "ABC,! xyz", "café 東京", "ΟΣİ_"]
    for _ in range(500):
        alphabet = rng.choice(alphabets)
        left = "".join(rng.choice(alphabet) for _ in range(rng.randrange(30)))
        right = "".join(rng.choice(alphabet) for _ in range(rng.randrange(30)))
        for name in FUZZ_NAMES:
            assert getattr(fuzz, name)(left, right) == getattr(up_fuzz, name)(left, right), (
                name,
                left,
                right,
            )


@requires_upstream
def test_fuzz_random_sequence_differential() -> None:
    rng = random.Random(73)
    elements = ["a", "b", " ", "\t", 1, 2, -1, 32, 9, ("token",), None]
    for _ in range(200):
        left = [rng.choice(elements) for _ in range(rng.randrange(12))]
        right = [rng.choice(elements) for _ in range(rng.randrange(12))]
        for name in FUZZ_NAMES:
            shim_fn = getattr(fuzz, name)
            upstream_fn = getattr(up_fuzz, name)
            if name in {"ratio", "partial_ratio"}:
                actual = shim_fn(left, right)
                expected = upstream_fn(left, right)
            else:
                actual = shim_fn(left, right, full_process=False)
                expected = upstream_fn(left, right, full_process=False)
            assert actual == expected, (name, left, right, actual, expected)


@requires_upstream
def test_utils_differential() -> None:
    samples = [
        "Hello, World!",
        "Café—東京",
        "İstanbul",
        "ΟΣ",
        "foo_bar",
        "  multiple   spaces  ",
        "",
        b"caf\xe9",
        123,
    ]
    for value in samples:
        for force_ascii in (False, True):
            try:
                expected = up_utils.full_process(value, force_ascii)
            except Exception as expected_error:
                with pytest.raises(type(expected_error)):
                    utils.full_process(value, force_ascii)
            else:
                assert utils.full_process(value, force_ascii) == expected


@requires_upstream
def test_process_random_differential() -> None:
    rng = random.Random(4)
    for _ in range(75):
        query = "".join(rng.choice("abc XYZ,!") for _ in range(rng.randrange(1, 15)))
        choices = [
            "".join(rng.choice("abc XYZ,!") for _ in range(rng.randrange(15)))
            for _ in range(rng.randrange(20))
        ]
        for name in ("WRatio", "ratio", "partial_ratio", "token_set_ratio"):
            assert process.extractBests(
                query,
                choices,
                scorer=getattr(fuzz, name),
                limit=5,
            ) == up_process.extractBests(
                query,
                choices,
                scorer=getattr(up_fuzz, name),
                limit=5,
            )
