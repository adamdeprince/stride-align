"""Tests for Jaro and Jaro-Winkler.

Correctness uses rapidfuzz as the oracle — we promise bit-equivalent
values for byte and Unicode inputs (the rounding-down convention for
the transposition count, which is what every common implementation
does).
"""

from __future__ import annotations

import random
import string

import pytest

import stride_align as sa


rapidfuzz = pytest.importorskip("rapidfuzz")
from rapidfuzz.distance import Jaro as _RFJaro  # noqa: E402
from rapidfuzz.distance import JaroWinkler as _RFJW  # noqa: E402


def _rng_strings(n_pairs: int, max_len: int, alphabet: str, *, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    return [
        (
            "".join(rng.choice(alphabet) for _ in range(rng.randint(0, max_len))),
            "".join(rng.choice(alphabet) for _ in range(rng.randint(0, max_len))),
        )
        for _ in range(n_pairs)
    ]


def test_jaro_edge_cases():
    assert sa.jaro_similarity("", "") == 1.0
    assert sa.jaro_similarity("abc", "") == 0.0
    assert sa.jaro_similarity("", "abc") == 0.0
    assert sa.jaro_similarity("abc", "abc") == 1.0
    assert sa.jaro_similarity("abc", "xyz") == 0.0


def test_jaro_known_values_match_rapidfuzz():
    for s1, s2 in [
        ("martha", "marhta"),
        ("dwayne", "duane"),
        ("dixon", "dicksonx"),
        ("jellyfish", "smellyfish"),
        ("ab", "ba"),
        ("abcd", "dcba"),
    ]:
        assert sa.jaro_similarity(s1, s2) == pytest.approx(_RFJaro.similarity(s1, s2))


def test_jaro_random_battery_matches_rapidfuzz():
    cases = _rng_strings(500, max_len=50, alphabet=string.ascii_lowercase, seed=7)
    for s1, s2 in cases:
        ours = sa.jaro_similarity(s1, s2)
        ref = _RFJaro.similarity(s1, s2)
        assert ours == pytest.approx(ref, abs=1e-12), (s1, s2, ours, ref)


def test_jaro_winkler_default_matches_rapidfuzz():
    cases = _rng_strings(500, max_len=50, alphabet=string.ascii_lowercase, seed=11)
    for s1, s2 in cases:
        assert sa.jaro_winkler_similarity(s1, s2) == pytest.approx(
            _RFJW.similarity(s1, s2), abs=1e-12
        )


def test_jaro_winkler_custom_prefix_weight():
    # Custom prefix_weight should still match rapidfuzz when threshold is the same.
    for weight in (0.05, 0.15, 0.2):
        assert sa.jaro_winkler_similarity(
            "martha", "marhta", prefix_weight=weight
        ) == pytest.approx(_RFJW.similarity("martha", "marhta", prefix_weight=weight))


def test_jaro_winkler_threshold_disables_bonus():
    # threshold=0 always applies the bonus; threshold=1 always disables.
    base = sa.jaro_similarity("ab", "ax")
    assert sa.jaro_winkler_similarity("ab", "ax", prefix_threshold=0.0) > base
    assert sa.jaro_winkler_similarity("ab", "ax", prefix_threshold=1.0) == base


def test_jaro_handles_bytes_and_unicode():
    assert sa.jaro_similarity(b"martha", b"marhta") == pytest.approx(
        _RFJaro.similarity("martha", "marhta")
    )
    assert sa.jaro_similarity("café", "cafe") == pytest.approx(
        _RFJaro.similarity("café", "cafe")
    )
    assert sa.jaro_similarity("🎉🎈ab", "🎈🎉ab") == pytest.approx(
        _RFJaro.similarity("🎉🎈ab", "🎈🎉ab")
    )


def test_jaro_similarities_returns_ndarray():
    out = sa.jaro_similarities("kitten", ["kitten", "sitting", "kit"])
    assert out.dtype.name == "float64"
    assert out.tolist() == pytest.approx(
        [_RFJaro.similarity("kitten", t) for t in ("kitten", "sitting", "kit")]
    )


def test_jaro_winkler_similarities_returns_ndarray():
    targets = ["kitten", "sitting", "kit"]
    out = sa.jaro_winkler_similarities("kitten", targets)
    assert out.dtype.name == "float64"
    assert out.tolist() == pytest.approx(
        [_RFJW.similarity("kitten", t) for t in targets]
    )


# --- top_k + best -------------------------------------------------


def test_jaro_top_k_returns_highest():
    targets = ["sitting", "kitten", "kit"]
    raw = sa.jaro_similarities("kitten", targets).tolist()
    ranked = sa.jaro_top_k("kitten", targets, k=2)
    actual_scores = sorted((s for _, s, _ in ranked), reverse=True)
    assert actual_scores == pytest.approx(sorted(raw, reverse=True)[:2])
    # Indices map back to scores.
    for target, score, idx in ranked:
        assert targets[idx] == target
        assert raw[idx] == pytest.approx(score)


def test_jaro_winkler_top_k_with_kwargs():
    targets = ["martha", "marhta", "xyz"]
    ranked = sa.jaro_winkler_top_k("martha", targets, k=2, prefix_weight=0.2)
    # "martha" is exact match, should be in top-2.
    assert any(target == "martha" for target, _, _ in ranked)


def test_jaro_best_finds_top():
    target, score, idx = sa.jaro_best("martha", ["xyz", "marhta", "martha"])
    assert target == "martha"
    assert score == pytest.approx(1.0)
    assert idx == 2


def test_jaro_winkler_best_picks_prefix_match():
    target, score, _ = sa.jaro_winkler_best(
        "martin",
        ["martia", "smartin", "marthax"],
    )
    # The right answer is the highest JW; verify it matches rapidfuzz.
    ref = [
        _RFJW.similarity("martin", t) for t in ("martia", "smartin", "marthax")
    ]
    assert score == pytest.approx(max(ref))


# --- extract / Scorer enum --------------------------------------


def test_extract_jaro():
    targets = ["kitten", "sitting", "kit"]
    ranked = sa.extract("kitten", targets, scorer=sa.Scorer.JARO, k=2)
    assert {t for t, _, _ in ranked} <= set(targets)
    # The exact-match entry should be in the top-2.
    assert any(t == "kitten" and s == pytest.approx(1.0) for t, s, _ in ranked)


def test_extract_jaro_winkler():
    targets = ["martha", "marhta", "xyz"]
    ranked = sa.extract("martha", targets, scorer=sa.Scorer.JARO_WINKLER, k=2)
    assert any(t == "martha" and s == pytest.approx(1.0) for t, s, _ in ranked)


def test_scorer_jaro_values_distinct():
    assert int(sa.Scorer.JARO) == 6
    assert int(sa.Scorer.JARO_WINKLER) == 7
    assert sa.Scorer.JARO != sa.Scorer.JARO_WINKLER


def test_jaro_top_k_zero_returns_empty():
    assert sa.jaro_top_k("abc", ["a", "b"], 0) == []


def test_jaro_top_k_empty_targets_returns_empty():
    assert sa.jaro_top_k("abc", [], 5) == []


def test_jaro_winkler_top_k_handles_generator():
    targets = (s for s in ["martha", "marhta", "xyz"])
    ranked = sa.jaro_winkler_top_k("martha", targets, k=2)
    assert any(t == "martha" for t, _, _ in ranked)
