"""Tests for true (unrestricted) Damerau-Levenshtein.

Distinct from the OSA variant we ship under ``damerau_levenshtein_*``:
in the unrestricted form each character may participate in more than
one edit. Oracle is rapidfuzz.distance.DamerauLevenshtein.
"""

from __future__ import annotations

import random
import string

import pytest

import stride_align as sa


rapidfuzz = pytest.importorskip("rapidfuzz")
from rapidfuzz.distance import DamerauLevenshtein as _RFDL  # noqa: E402
from rapidfuzz.distance import OSA as _RFOSA  # noqa: E402


def _rand(rng, n):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def test_classic_examples():
    assert sa.true_damerau_levenshtein_score("", "") == 0
    assert sa.true_damerau_levenshtein_score("abc", "") == 3
    assert sa.true_damerau_levenshtein_score("", "abc") == 3
    assert sa.true_damerau_levenshtein_score("abc", "abc") == 0
    assert sa.true_damerau_levenshtein_score("kitten", "sitting") == 3


def test_diverges_from_osa_on_overlapping_transposition():
    """The canonical OSA-vs-true-DL diff. ``ca`` -> ``abc`` is 2 in
    the unrestricted form (transpose c,a then insert b) but 3 under
    OSA (each char may touch only one edit)."""
    osa = sa.damerau_levenshtein_score("ca", "abc")
    true_dl = sa.true_damerau_levenshtein_score("ca", "abc")
    assert osa == 3
    assert true_dl == 2
    # Matches the rapidfuzz oracle for both variants.
    assert osa == _RFOSA.distance("ca", "abc")
    assert true_dl == _RFDL.distance("ca", "abc")


def test_normalized_examples():
    assert sa.true_damerau_levenshtein_normalized_score("foo", "foo") == 1.0
    assert sa.true_damerau_levenshtein_normalized_score("", "") == 1.0
    assert sa.true_damerau_levenshtein_normalized_score("abc", "") == 0.0
    # ca -> abc: dist 2, max len 3, sim = 1 - 2/3.
    assert sa.true_damerau_levenshtein_normalized_score("ca", "abc") == \
        pytest.approx(1.0 - 2.0 / 3.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_rapidfuzz_oracle(seed):
    rng = random.Random(seed)
    for _ in range(200):
        a = _rand(rng, rng.randint(0, 40))
        b = _rand(rng, rng.randint(0, 40))
        expected = _RFDL.distance(a, b)
        got = sa.true_damerau_levenshtein_score(a, b)
        assert got == expected, (
            f"true_dl({a!r}, {b!r}) = {got}, expected {expected}"
        )


def test_long_strings():
    rng = random.Random(0xBEEF)
    a = _rand(rng, 150)
    b = _rand(rng, 130)
    assert sa.true_damerau_levenshtein_score(a, b) == _RFDL.distance(a, b)


def test_scores_batch():
    q = "ca"
    targets = ["abc", "ca", "ac", "x"]
    got = list(sa.true_damerau_levenshtein_scores(q, targets))
    expected = [_RFDL.distance(q, t) for t in targets]
    assert got == expected


def test_normalized_scores_batch():
    q = "abc"
    targets = ["abc", "ca", "abcd", ""]
    got = list(sa.true_damerau_levenshtein_normalized_scores(q, targets))
    for g, t in zip(got, targets):
        assert g == pytest.approx(
            _RFDL.normalized_similarity(q, t)
        )


def test_top_k_returns_lowest_distances():
    q = "ca"
    ts = ["abc", "ca", "ac", "xyz", "cab"]
    out = sa.true_damerau_levenshtein_top_k(q, ts, k=3)
    scores = sorted(s for _, s, _ in out)
    expected = sorted(_RFDL.distance(q, t) for t in ts)[:3]
    assert scores == expected


def test_normalized_top_k_returns_highest_similarities():
    q = "ca"
    ts = ["abc", "ca", "ac", "xyz", "cab"]
    out = sa.true_damerau_levenshtein_normalized_top_k(q, ts, k=2)
    scores = sorted((s for _, s, _ in out), reverse=True)
    expected = sorted(
        (_RFDL.normalized_similarity(q, t) for t in ts),
        reverse=True,
    )[:2]
    assert scores == pytest.approx(expected)


def test_best():
    q = "ca"
    ts = ["abc", "ca", "ac"]
    obj, score, idx = sa.true_damerau_levenshtein_best(q, ts)
    assert score == 0
    assert obj == "ca"


def test_cdist_matrix():
    qs = ["ca", "kitten"]
    ts = ["abc", "ca", "sitting"]
    out = sa.cdist(qs, ts, scorer=sa.Scorer.TRUE_DAMERAU_LEVENSHTEIN)
    assert out.shape == (2, 3)
    assert out[0, 0] == 2   # ca -> abc
    assert out[0, 1] == 0
    assert out[1, 2] == 3


def test_cdist_normalized_matches_pairwise():
    qs = ["ca", "kitten"]
    ts = ["abc", "ca", "sitting"]
    matrix = sa.cdist(qs, ts, scorer=sa.Scorer.TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED)
    for i, q in enumerate(qs):
        for j, t in enumerate(ts):
            assert matrix[i, j] == pytest.approx(
                _RFDL.normalized_similarity(q, t)
            )


def test_extract_by_enum():
    out = sa.extract(
        "ca",
        ["abc", "ca", "ac"],
        scorer=sa.Scorer.TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED,
        k=2,
    )
    assert len(out) == 2
    # "ca" itself should be top.
    by_score = sorted(out, key=lambda r: -r[1])
    assert by_score[0][0] == "ca"


def test_cdist_above_threshold():
    qs = ["ca"]
    ts = ["abc", "ca", "ac", "xyz"]
    out = list(sa.cdist_above_threshold(
        qs, ts,
        scorer=sa.Scorer.TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED,
        threshold=0.5,
    ))
    # Threshold 0.5 with max_len=3 ⟹ distance <= 1.5, i.e. <= 1.
    # ca -> ca (dist 0), ca -> ac (dist 1, transposition) → both pass.
    accepted = {t for _, _, t in out}
    assert "ca" in accepted
    assert "ac" in accepted


def test_empty_inputs():
    assert sa.true_damerau_levenshtein_score("", "") == 0
    assert sa.true_damerau_levenshtein_normalized_score("", "") == 1.0
    assert list(sa.true_damerau_levenshtein_scores("abc", [])) == []


def test_unicode_input():
    # Both inputs wider than 1-byte unicode go through the prepared-
    # tokens fallback path.
    a = "café"
    b = "cafe"
    assert sa.true_damerau_levenshtein_score(a, b) == _RFDL.distance(a, b)
