"""Tests for `*_top_k`, `*_best`, and the unified `extract()` entry point.

The native bindings fold a SIMD `*_scores` call and an O(N)
std::nth_element partition into a single C++ entry point that returns
`list[(target, score, index)]`. Order within the returned list is
unspecified — only the *set* of (target, score, index) tuples is
guaranteed to be the top-k. These tests assert the set, not the order.
"""

from __future__ import annotations

import pytest

from stride_align import (
    Scorer,
    damerau_levenshtein_best,
    damerau_levenshtein_normalized_best,
    damerau_levenshtein_normalized_scores,
    damerau_levenshtein_normalized_top_k,
    damerau_levenshtein_scores,
    damerau_levenshtein_top_k,
    extract,
    extract_best,
    hamming_best,
    hamming_normalized_best,
    hamming_normalized_scores,
    hamming_normalized_top_k,
    hamming_scores,
    hamming_top_k,
    levenshtein_best,
    levenshtein_normalized_best,
    levenshtein_normalized_scores,
    levenshtein_normalized_top_k,
    levenshtein_scores,
    levenshtein_top_k,
    smith_waterman_best,
    smith_waterman_scores,
    smith_waterman_top_k,
)


def _expected_lowest(raw, k):
    """The k smallest scores, as a sorted list (no index info)."""
    return sorted(raw)[:k]


def _expected_highest(raw, k):
    """The k largest scores, as a sorted-descending list (no index info)."""
    return sorted(raw, reverse=True)[:k]


@pytest.fixture
def lev_corpus():
    return ["kitten", "sitting", "kittenish", "kit", "", "biting", "kitten"]


def test_levenshtein_top_k_set_matches_argsort(lev_corpus):
    query = "kitten"
    raw = levenshtein_scores(query, lev_corpus).tolist()
    actual = levenshtein_top_k(query, lev_corpus, k=3)
    actual_scores = sorted(score for _, score, _ in actual)
    assert actual_scores == _expected_lowest(raw, 3)
    # Indices point back at scores that match the raw vector.
    for target, score, idx in actual:
        assert raw[idx] == score
        assert lev_corpus[idx] == target


def test_levenshtein_normalized_top_k_highest(lev_corpus):
    query = "kitten"
    raw = levenshtein_normalized_scores(query, lev_corpus).tolist()
    actual = levenshtein_normalized_top_k(query, lev_corpus, k=3)
    actual_scores = sorted((score for _, score, _ in actual), reverse=True)
    assert actual_scores == pytest.approx(_expected_highest(raw, 3))
    # The exact match should always be in the top-3.
    assert any(score == pytest.approx(1.0) for _, score, _ in actual)


def test_damerau_levenshtein_top_k_transposition():
    # OSA distance 1 (transposition) beats Levenshtein's 2 substitutions.
    query = "ab"
    targets = ["ba", "xy", "ab", "az"]
    ranked = damerau_levenshtein_top_k(query, targets, k=3)
    scores = sorted(s for _, s, _ in ranked)
    # The top-3 set should be {0 (exact), 1 (transposition), 1 (1 sub)}.
    assert scores == [0, 1, 1]


def test_damerau_levenshtein_normalized_top_k_floats():
    query = "house"
    targets = ["mouse", "house", "horse", "hosue"]
    ranked = damerau_levenshtein_normalized_top_k(query, targets, k=2)
    raw = damerau_levenshtein_normalized_scores(query, targets).tolist()
    assert all(isinstance(score, float) for _, score, _ in ranked)
    actual_scores = sorted((score for _, score, _ in ranked), reverse=True)
    assert actual_scores == pytest.approx(_expected_highest(raw, 2))


def test_hamming_top_k_lowest_distance():
    query = b"abcdef"
    targets = [b"abcdef", b"abcxxx", b"xxcdef", b"abxdef"]
    ranked = hamming_top_k(query, targets, k=2)
    raw = hamming_scores(query, targets).tolist()
    actual_scores = sorted(s for _, s, _ in ranked)
    assert actual_scores == _expected_lowest(raw, 2)
    # The exact match (distance 0) must appear in the top-2.
    assert (b"abcdef", 0, 0) in ranked


def test_hamming_normalized_top_k_highest():
    query = b"abcdef"
    targets = [b"abcdef", b"abcxxx", b"xxcdef", b"abxdef"]
    ranked = hamming_normalized_top_k(query, targets, k=3)
    raw = hamming_normalized_scores(query, targets).tolist()
    actual_scores = sorted((s for _, s, _ in ranked), reverse=True)
    assert actual_scores == pytest.approx(_expected_highest(raw, 3))


def test_hamming_top_k_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="equal-length"):
        hamming_top_k(b"abc", [b"ab"], k=1)


def test_smith_waterman_top_k_highest():
    query = "ACGTACGTACGT"
    targets = ["ACGTACGTACGT", "AAAAAAAAAAAA", "ACGTACGTACGA", "ACGTAAAAACGT"]
    raw = smith_waterman_scores(query, targets).tolist()
    ranked = smith_waterman_top_k(query, targets, k=2)
    actual_scores = sorted((s for _, s, _ in ranked), reverse=True)
    assert actual_scores == _expected_highest(raw, 2)
    for _, score, idx in ranked:
        assert score == raw[idx]


@pytest.mark.parametrize(
    "fn",
    [
        levenshtein_top_k,
        levenshtein_normalized_top_k,
        damerau_levenshtein_top_k,
        damerau_levenshtein_normalized_top_k,
        hamming_top_k,
        hamming_normalized_top_k,
    ],
)
def test_top_k_zero_returns_empty(fn):
    assert fn("abc", ["abc", "abd", "abe"], 0) == []


@pytest.mark.parametrize(
    "fn",
    [
        levenshtein_top_k,
        levenshtein_normalized_top_k,
        damerau_levenshtein_top_k,
        damerau_levenshtein_normalized_top_k,
        hamming_top_k,
        hamming_normalized_top_k,
    ],
)
def test_top_k_empty_targets_returns_empty(fn):
    assert fn("abc", [], 5) == []


@pytest.mark.parametrize(
    "fn",
    [
        levenshtein_top_k,
        damerau_levenshtein_top_k,
        hamming_top_k,
    ],
)
def test_top_k_k_greater_than_n_returns_all(fn):
    query = "aaa"
    targets = ["aaa", "aab", "abb", "bbb"]
    ranked = fn(query, targets, 99)
    assert len(ranked) == len(targets)
    # Every input index should appear exactly once.
    assert {idx for _, _, idx in ranked} == set(range(len(targets)))


def test_top_k_indices_point_back_to_input():
    query = "kitten"
    targets = ["sitting", "kitten", "kit", "biting"]
    ranked = levenshtein_top_k(query, targets, k=4)
    for target, _, idx in ranked:
        assert targets[idx] == target


def test_top_k_accepts_generator_input():
    query = "abc"
    targets = (s for s in ["abc", "abd", "xyz"])
    ranked = levenshtein_top_k(query, targets, k=2)
    # The exact match should be in the top-2.
    assert any(score == 0 for _, score, _ in ranked)


def test_top_k_rejects_single_string_targets():
    with pytest.raises(TypeError):
        levenshtein_top_k("abc", "abcabc", k=1)


# --- extract() ----------------------------------------------------


def test_extract_levenshtein_matches_top_k():
    query = "kitten"
    targets = ["kitten", "sitting", "kit"]
    via_extract = extract(query, targets, scorer=Scorer.LEVENSHTEIN, k=2)
    via_direct = levenshtein_top_k(query, targets, k=2)
    assert sorted(s for _, s, _ in via_extract) == sorted(s for _, s, _ in via_direct)


def test_extract_levenshtein_normalized():
    query = "kitten"
    targets = ["kitten", "sitting", "kit"]
    ranked = extract(query, targets, scorer=Scorer.LEVENSHTEIN_NORMALIZED, k=3)
    # The exact match must be in the top-3.
    assert any(score == pytest.approx(1.0) for _, score, _ in ranked)


def test_extract_damerau_levenshtein():
    query = "ab"
    targets = ["ba", "xy", "ab"]
    ranked = extract(query, targets, scorer=Scorer.DAMERAU_LEVENSHTEIN, k=2)
    scores = sorted(s for _, s, _ in ranked)
    assert scores == [0, 1]


def test_extract_hamming():
    query = b"abcd"
    targets = [b"abcd", b"abce", b"xxxx"]
    ranked = extract(query, targets, scorer=Scorer.HAMMING, k=2)
    scores = sorted(s for _, s, _ in ranked)
    assert scores == [0, 1]


def test_extract_hamming_normalized_returns_floats():
    query = b"abcd"
    targets = [b"abcd", b"abce", b"xxxx"]
    ranked = extract(query, targets, scorer=Scorer.HAMMING_NORMALIZED, k=2)
    for _, score, _ in ranked:
        assert isinstance(score, float)
    assert any(score == pytest.approx(1.0) for _, score, _ in ranked)


def test_scorer_intenum_round_trips_through_int():
    # Scorer is a Python IntEnum; integer values are part of the
    # contract because the C++ binding takes a plain int across the
    # boundary.
    assert int(Scorer.LEVENSHTEIN) == 0
    assert int(Scorer.LEVENSHTEIN_NORMALIZED) == 1
    assert int(Scorer.DAMERAU_LEVENSHTEIN) == 2
    assert int(Scorer.DAMERAU_LEVENSHTEIN_NORMALIZED) == 3
    assert int(Scorer.HAMMING) == 4
    assert int(Scorer.HAMMING_NORMALIZED) == 5


# --- *_best ------------------------------------------------------


def test_levenshtein_best_returns_zero_distance_match():
    query = "kitten"
    targets = ["sitting", "kitten", "kit"]
    best = levenshtein_best(query, targets)
    assert best == ("kitten", 0, 1)


def test_levenshtein_best_returns_none_for_empty():
    assert levenshtein_best("abc", []) is None


def test_levenshtein_normalized_best_returns_highest():
    query = "house"
    targets = ["abcde", "house"]
    target, score, idx = levenshtein_normalized_best(query, targets)
    assert target == "house"
    assert score == pytest.approx(1.0)
    assert idx == 1


def test_damerau_levenshtein_best_prefers_transposition():
    # Among {"ab", "ba", "xy"}, "ab" is exact (distance 0); pick that.
    target, score, _ = damerau_levenshtein_best("ab", ["xy", "ab", "ba"])
    assert target == "ab"
    assert score == 0


def test_damerau_levenshtein_normalized_best_returns_float():
    target, score, _ = damerau_levenshtein_normalized_best(
        "house", ["mouse", "house"]
    )
    assert target == "house"
    assert isinstance(score, float)
    assert score == pytest.approx(1.0)


def test_hamming_best_returns_zero():
    target, score, idx = hamming_best(b"abc", [b"abx", b"abc", b"xxx"])
    assert target == b"abc"
    assert score == 0
    assert idx == 1


def test_hamming_normalized_best_returns_float():
    target, score, _ = hamming_normalized_best(b"abc", [b"abx", b"abc"])
    assert target == b"abc"
    assert isinstance(score, float)
    assert score == pytest.approx(1.0)


def test_smith_waterman_best_picks_perfect_match():
    target, score, _ = smith_waterman_best(
        "ACGTACGTACGT",
        ["AAAAAAAAAAAA", "ACGTACGTACGT", "ACGTAAAA"],
    )
    assert target == "ACGTACGTACGT"
    # SW with match=2 over 12 perfect matches: 24.
    assert score == 24


def test_extract_best_routes_via_scorer():
    target, score, _ = extract_best(
        "kitten",
        ["sitting", "kitten"],
        scorer=Scorer.LEVENSHTEIN,
    )
    assert target == "kitten"
    assert score == 0


def test_extract_best_normalized_returns_float():
    target, score, _ = extract_best(
        "house",
        ["mouse", "house"],
        scorer=Scorer.LEVENSHTEIN_NORMALIZED,
    )
    assert target == "house"
    assert isinstance(score, float)
    assert score == pytest.approx(1.0)


def test_extract_best_returns_none_for_empty():
    assert extract_best("abc", [], scorer=Scorer.LEVENSHTEIN) is None
