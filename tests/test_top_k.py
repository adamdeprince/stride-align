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


# ---------------------------------------------------------------------------
# cdist_top_k_per_query: top-k targets per query, yielded as a generator.
# Differs from cdist_top_k (which returns the k highest pairs globally
# across the whole queries x targets matrix).
# ---------------------------------------------------------------------------


from stride_align import (
    cdist_top_k_per_query,
    levenshtein_normalized_score,
    jaro_similarity,
)


def test_cdist_top_k_per_query_yields_per_query():
    queries = ["hello", "world", "pyth"]
    targets = ["hellp", "help", "world", "pythn", "apple", "wrld"]
    results = list(
        cdist_top_k_per_query(
            queries, targets, scorer=Scorer.LEVENSHTEIN_NORMALIZED, k=2
        )
    )
    assert len(results) == len(queries)
    seen_queries = [q for q, _ in results]
    assert seen_queries == queries  # preserves input order


def test_cdist_top_k_per_query_returns_top_by_score():
    queries = ["hello"]
    targets = ["hellp", "help", "world", "wrld", "yo", "helloworld"]
    [(query, top)] = list(
        cdist_top_k_per_query(
            queries, targets, scorer=Scorer.LEVENSHTEIN_NORMALIZED, k=3
        )
    )
    assert query == "hello"
    # Each entry is (score, target). Sorted descending.
    scores = [score for score, _ in top]
    assert scores == sorted(scores, reverse=True)
    # Top-3 must match the true top-3 by score (set comparison).
    expected_top3 = _expected_highest(
        [levenshtein_normalized_score("hello", t) for t in targets], 3
    )
    assert [pytest.approx(s) for s in scores] == [pytest.approx(s) for s in expected_top3]


def test_cdist_top_k_per_query_matches_brute_force():
    """Score-only correctness vs a Python brute-force loop across a
    range of query/target lengths. Ties at the boundary mean the
    *set of targets* picked can vary, but the sorted scores must
    agree exactly with the brute-force result."""
    queries = ["a", "ab", "abc", "fuzzy", "longerquery", "x" * 40]
    targets = [
        "a", "ab", "abc", "abcd",
        "fuzzz", "fuzy", "fuzzyy",
        "longerquery", "shortquery", "x" * 40, "x" * 80,
        "wholly unrelated string", "y" * 25,
    ]
    for query, top in cdist_top_k_per_query(
        queries, targets, scorer=Scorer.LEVENSHTEIN_NORMALIZED, k=5
    ):
        impl_scores = sorted([round(s, 9) for s, _ in top], reverse=True)
        brute_scores = sorted(
            (round(levenshtein_normalized_score(query, t), 9) for t in targets),
            reverse=True,
        )[:5]
        assert impl_scores == brute_scores, (
            f"query={query!r}: {impl_scores} != {brute_scores}"
        )


def test_cdist_top_k_per_query_jaro_scorer():
    queries = ["martha"]
    targets = ["marhta", "marta", "matra", "marble", "wholly different"]
    [(_, top)] = list(
        cdist_top_k_per_query(
            queries, targets, scorer=Scorer.JARO, k=3
        )
    )
    scores = [round(s, 9) for s, _ in top]
    brute = sorted(
        (round(jaro_similarity("martha", t), 9) for t in targets), reverse=True
    )[:3]
    assert scores == brute


def test_cdist_top_k_per_query_accepts_generator_queries():
    targets = ["hellp", "help", "world", "wrld"]

    def gen():
        yield "hello"
        yield "world"

    results = list(
        cdist_top_k_per_query(
            gen(), targets, scorer=Scorer.LEVENSHTEIN_NORMALIZED, k=2
        )
    )
    assert [q for q, _ in results] == ["hello", "world"]


def test_cdist_top_k_per_query_accepts_generator_targets():
    """``targets`` is materialised once and re-iterated per query.
    Passing a generator must work (it's allowed to be slow, just not wrong)."""
    queries = ["hello", "world"]

    def gen():
        yield "hellp"
        yield "help"
        yield "world"

    results = list(
        cdist_top_k_per_query(
            queries, gen(), scorer=Scorer.LEVENSHTEIN_NORMALIZED, k=2
        )
    )
    # Each query sees the same materialised target set.
    assert len(results) == 2
    assert ("hello", [(0.8, "hellp"), (0.6, "help")]) == (
        results[0][0],
        [(round(s, 9), t) for s, t in results[0][1]],
    )


def test_cdist_top_k_per_query_rejects_string_queries():
    with pytest.raises(TypeError, match="queries"):
        # The bad call must raise eagerly, not on first consumption.
        cdist_top_k_per_query(
            "hello", ["targets"], scorer=Scorer.LEVENSHTEIN_NORMALIZED, k=3
        )


def test_cdist_top_k_per_query_rejects_string_targets():
    with pytest.raises(TypeError, match="targets"):
        cdist_top_k_per_query(
            ["hello"], "targets_as_str", scorer=Scorer.LEVENSHTEIN_NORMALIZED, k=3
        )


def test_cdist_top_k_per_query_rejects_distance_scorer():
    """Distance scorers (lower-is-better) aren't supported in this first
    cut — the length bound function is similarity-only and the heap
    invariant flips. Raise a clear TypeError naming the supported set."""
    with pytest.raises(TypeError, match="normalised-similarity"):
        list(
            cdist_top_k_per_query(
                ["hello"], ["world"],
                scorer=Scorer.LEVENSHTEIN, k=2,
            )
        )


def test_cdist_top_k_per_query_k_larger_than_targets():
    queries = ["hello"]
    targets = ["world", "wrld"]
    [(_, top)] = list(
        cdist_top_k_per_query(
            queries, targets, scorer=Scorer.LEVENSHTEIN_NORMALIZED, k=10
        )
    )
    # Only as many entries as there are scorable targets.
    assert len(top) == 2


def test_cdist_top_k_per_query_empty_targets():
    [(_, top)] = list(
        cdist_top_k_per_query(
            ["hello"], [], scorer=Scorer.LEVENSHTEIN_NORMALIZED, k=5
        )
    )
    assert top == []


def test_cdist_top_k_per_query_empty_queries():
    results = list(
        cdist_top_k_per_query(
            [], ["a", "b"], scorer=Scorer.LEVENSHTEIN_NORMALIZED, k=5
        )
    )
    assert results == []


def test_cdist_top_k_per_query_hamming_length_mismatch_skipped():
    """HAMMING_NORMALIZED with unequal-length targets: the length-bound
    function returns 0 for those pairs, so they get pruned before
    the kernel runs (which would otherwise raise ValueError). Only
    equal-length targets enter the heap."""
    queries = ["hello"]
    targets = ["world", "hellp", "longerthing", "h", "wrld"]
    # Equal-length to "hello" (5): "world", "hellp"
    [(_, top)] = list(
        cdist_top_k_per_query(
            queries, targets, scorer=Scorer.HAMMING_NORMALIZED, k=10
        )
    )
    chosen = {t for _, t in top}
    assert chosen == {"world", "hellp"}


def test_cdist_top_k_per_query_pruning_matches_unpruned_scores():
    """The closed-form length bound is provably tight, so enabling
    pruning must never change the SET of top-k SCORES. (The chosen
    targets at tied-score boundaries can differ — that's an arbitrary
    heap tie-break — but the sorted scores must agree exactly.)"""
    import random
    import string

    rng = random.Random(7)
    targets = [
        "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 60)))
        for _ in range(500)
    ]
    queries = ["abc", "hello", "fuzzymatch", "x" * 30]

    for q, unpruned in cdist_top_k_per_query(
        queries, targets, scorer=Scorer.LEVENSHTEIN_NORMALIZED, k=5, pruning=False,
    ):
        [(q2, pruned)] = list(
            cdist_top_k_per_query(
                [q], targets,
                scorer=Scorer.LEVENSHTEIN_NORMALIZED, k=5, pruning=True,
            )
        )
        assert q == q2
        unpruned_scores = sorted([round(s, 9) for s, _ in unpruned], reverse=True)
        pruned_scores = sorted([round(s, 9) for s, _ in pruned], reverse=True)
        assert unpruned_scores == pruned_scores, (
            f"query={q!r}: unpruned {unpruned_scores} != pruned {pruned_scores}"
        )


def test_cdist_top_k_per_query_zero_k_returns_empty_lists():
    queries = ["hello", "world"]
    targets = ["hellp", "help"]
    results = list(
        cdist_top_k_per_query(
            queries, targets, scorer=Scorer.LEVENSHTEIN_NORMALIZED, k=0
        )
    )
    assert [q for q, _ in results] == queries
    assert all(top == [] for _, top in results)
