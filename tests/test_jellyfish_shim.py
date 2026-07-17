"""Compatibility tests for ``stride_align.jellyfish``.

Pinned tests run without the third-party package. When Jellyfish is
installed through the ``phonetic-compat`` extra, the differential battery
also verifies the facade against the current upstream implementation.
"""

from __future__ import annotations

import inspect
import random
import string

import pytest

import stride_align.jellyfish as shim

try:
    import jellyfish as upstream
except ImportError:  # pragma: no cover - exercised in minimal installs
    upstream = None


PUBLIC_FUNCTIONS = {
    "damerau_levenshtein_distance",
    "hamming_distance",
    "jaccard_similarity",
    "jaro_similarity",
    "jaro_winkler_similarity",
    "levenshtein_distance",
    "match_rating_codex",
    "match_rating_comparison",
    "metaphone",
    "nysiis",
    "soundex",
}


def test_public_surface() -> None:
    assert set(shim.__all__) == PUBLIC_FUNCTIONS
    assert list(inspect.signature(shim.levenshtein_distance).parameters) == ["a", "b"]
    jaccard_parameters = inspect.signature(shim.jaccard_similarity).parameters
    assert list(jaccard_parameters) == ["a", "b", "ngram_size"]
    assert jaccard_parameters["ngram_size"].default is None


@pytest.mark.parametrize(
    "function,args,expected",
    [
        (shim.levenshtein_distance, ("kitten", "sitting"), 3),
        (shim.damerau_levenshtein_distance, ("ca", "abc"), 2),
        (shim.hamming_distance, ("abcd", "abc"), 1),
        (shim.hamming_distance, ("Saturday", "Sunday"), 7),
        (shim.jaro_similarity, ("martha", "marhta"), 0.9444444444444445),
        (shim.jaro_similarity, ("", ""), 0.0),
        (shim.jaro_winkler_similarity, ("dixon", "dicksonx"), 0.8133333333333332),
    ],
)
def test_distance_and_similarity_examples(function, args, expected) -> None:
    assert function(*args) == pytest.approx(expected, abs=1e-12)


def test_jaro_winkler_long_tolerance() -> None:
    regular = shim.jaro_winkler_similarity("two long strings", "two long stringz")
    long = shim.jaro_winkler_similarity("two long strings", "two long stringz", long_tolerance=True)
    assert regular == pytest.approx(0.975, abs=1e-3)
    assert long == pytest.approx(0.984, abs=1e-3)
    assert long > regular


def test_jaccard_word_set_semantics() -> None:
    assert shim.jaccard_similarity("John Smith", "Smith John") == 1.0
    assert shim.jaccard_similarity("John Smith", "John Jacob Smith") == pytest.approx(2 / 3)
    assert shim.jaccard_similarity("", "") == 0.0


def test_jaccard_uses_non_overlapping_character_chunks() -> None:
    assert shim.jaccard_similarity("night", "nacht", 2) == pytest.approx(0.2)
    assert shim.jaccard_similarity("night", "nacht", 3) == pytest.approx(1 / 3)


def test_jaccard_rejects_zero_chunk_without_upstream_panic() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        shim.jaccard_similarity("abc", "abc", 0)


@pytest.mark.parametrize(
    "function,value,expected",
    [
        (shim.soundex, "Washington", "W252"),
        (shim.soundex, "Çáŕẗéř", "C636"),
        (shim.metaphone, "this is a difficult string", "0S IS A TFKLT STRNK"),
        (shim.metaphone, "Çáŕẗéř", "KRTR"),
        (shim.nysiis, "Catherine", "CATARAN"),
        (shim.nysiis, "ç", "Ç"),
        (shim.match_rating_codex, "Kathrynoglin", "KTHGLN"),
        (shim.match_rating_codex, "Frédéric", "FRÉÉRC"),
    ],
)
def test_phonetic_examples(function, value, expected) -> None:
    assert function(value) == expected


def test_match_rating_comparison_tristate() -> None:
    assert shim.match_rating_comparison("Smith", "Smyth") is True
    assert shim.match_rating_comparison("Michael", "Mike") is False
    assert shim.match_rating_comparison("Tim", "Timothy") is None
    assert shim.match_rating_comparison("invalid!", "invalid") is None


def test_match_rating_codex_rejects_non_alphabetic_input() -> None:
    with pytest.raises(ValueError, match="alphabetical"):
        shim.match_rating_codex("i’m")


@pytest.mark.parametrize("function", [getattr(shim, name) for name in PUBLIC_FUNCTIONS])
def test_string_only_contract(function) -> None:
    count = len(inspect.signature(function).parameters)
    args = (b"abc", b"abd") if count >= 2 else (b"abc",)
    with pytest.raises(TypeError):
        function(*args)


@pytest.mark.parametrize(
    "left,right",
    [
        ("e\u0301", "x"),
        ("👩\u200d💻", "x"),
        ("👨\u200d👩\u200d👧\u200d👦", "👨"),
        ("🇺🇸", "x"),
        ("🇺🇸🇨🇦", "🇺🇸"),
        ("\r\n", "x"),
        ("\u0600\u0601a", "x"),
        ("가", "x"),
        ("क्ष", "क"),
        ("👍🏽", "👍"),
    ],
)
def test_edit_distance_counts_grapheme_clusters(left: str, right: str) -> None:
    # These all contain at least one multi-code-point extended grapheme.
    # The expected values are independently pinned to Jellyfish 1.2.1.
    expected = {
        ("e\u0301", "x"): 1,
        ("👩\u200d💻", "x"): 1,
        ("👨\u200d👩\u200d👧\u200d👦", "👨"): 1,
        ("🇺🇸", "x"): 1,
        ("🇺🇸🇨🇦", "🇺🇸"): 1,
        ("\r\n", "x"): 1,
        ("\u0600\u0601a", "x"): 1,
        ("가", "x"): 1,
        ("क्ष", "क"): 1,
        ("👍🏽", "👍"): 1,
    }[left, right]
    assert shim.levenshtein_distance(left, right) == expected


@pytest.mark.skipif(upstream is None, reason="install jellyfish via the phonetic-compat extra")
@pytest.mark.parametrize(
    "function_name",
    [
        "levenshtein_distance",
        "damerau_levenshtein_distance",
        "hamming_distance",
        "jaro_similarity",
        "jaro_winkler_similarity",
    ],
)
def test_grapheme_family_matches_upstream(function_name: str) -> None:
    pairs = [
        ("e\u0301", "x"),
        ("👩\u200d💻", "x"),
        ("👨\u200d👩\u200d👧\u200d👦", "👨"),
        ("🇺🇸🇨🇦", "🇺🇸"),
        ("\u0600\u0601a", "x"),
        ("가", "x"),
        ("क्ष", "क"),
        ("👍🏽", "👍"),
    ]
    ours = getattr(shim, function_name)
    theirs = getattr(upstream, function_name)
    for left, right in pairs:
        assert ours(left, right) == pytest.approx(theirs(left, right), abs=1e-12)


@pytest.mark.skipif(upstream is None, reason="install jellyfish via the phonetic-compat extra")
def test_random_ascii_surface_matches_upstream() -> None:
    rng = random.Random(0x4A454C4C59)
    alphabet = string.ascii_letters + string.digits + " -'@"
    pairs = [
        (
            "".join(rng.choice(alphabet) for _ in range(rng.randrange(24))),
            "".join(rng.choice(alphabet) for _ in range(rng.randrange(24))),
        )
        for _ in range(300)
    ]

    pair_functions = [
        "levenshtein_distance",
        "damerau_levenshtein_distance",
        "hamming_distance",
        "jaro_similarity",
        "jaro_winkler_similarity",
        "match_rating_comparison",
    ]
    for function_name in pair_functions:
        ours = getattr(shim, function_name)
        theirs = getattr(upstream, function_name)
        for left, right in pairs:
            assert ours(left, right) == pytest.approx(theirs(left, right), abs=1e-12)

    for left, right in pairs:
        assert shim.jaro_winkler_similarity(left, right, True) == pytest.approx(
            upstream.jaro_winkler_similarity(left, right, True), abs=1e-12
        )
        assert shim.jaccard_similarity(left, right) == upstream.jaccard_similarity(left, right)
        for size in (1, 2, 3, 5):
            assert shim.jaccard_similarity(left, right, size) == upstream.jaccard_similarity(
                left, right, size
            )


@pytest.mark.skipif(upstream is None, reason="install jellyfish via the phonetic-compat extra")
def test_random_phonetic_surface_matches_upstream() -> None:
    rng = random.Random(0x50484F4E45)
    alphabet = string.ascii_letters + string.digits + " -'@"
    values = ["".join(rng.choice(alphabet) for _ in range(rng.randrange(24))) for _ in range(500)]
    for function_name in ("soundex", "metaphone", "nysiis", "match_rating_codex"):
        ours = getattr(shim, function_name)
        theirs = getattr(upstream, function_name)
        for value in values:
            try:
                expected = theirs(value)
            except ValueError:
                with pytest.raises(ValueError):
                    ours(value)
            else:
                assert ours(value) == expected
