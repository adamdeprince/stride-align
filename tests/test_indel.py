"""Tests for Indel distance — Levenshtein restricted to insertions and
deletions only (no substitutions).

Correctness uses rapidfuzz as the oracle. The math identity
``indel = |a| + |b| - 2 * LCS(a, b)`` gives us a second cross-check.
"""

from __future__ import annotations

import random
import string

import pytest

import stride_align as sa


rapidfuzz = pytest.importorskip("rapidfuzz")
from rapidfuzz.distance import Indel as _RFIndel  # noqa: E402


def _rand(rng, n):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def test_classic_examples():
    # kitten -> sitting requires 1 sub + 1 sub + 1 ins under Levenshtein
    # (3 edits). Under Indel each sub costs 2 (delete + insert), so
    # kitten -> sitting = 2 dels + 2 ins + 1 ins = 5.
    assert sa.indel_score("kitten", "sitting") == 5
    assert sa.indel_score("", "") == 0
    assert sa.indel_score("abc", "") == 3
    assert sa.indel_score("", "abc") == 3
    assert sa.indel_score("abc", "abc") == 0
    # No substitution: "abc" -> "abd" is del 'c' + ins 'd' = 2.
    assert sa.indel_score("abc", "abd") == 2


def test_normalized_examples():
    # Identical -> 1.0.
    assert sa.indel_normalized_score("foo", "foo") == 1.0
    # Disjoint -> 1 - (a+b)/(a+b) = 0.
    assert sa.indel_normalized_score("aaa", "bbb") == 0.0
    # 1 - 2/(3+3) = 0.666...
    assert sa.indel_normalized_score("abc", "abd") == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_rapidfuzz_oracle(seed):
    rng = random.Random(seed)
    for _ in range(200):
        a = _rand(rng, rng.randint(0, 60))
        b = _rand(rng, rng.randint(0, 60))
        expected = _RFIndel.distance(a, b)
        got = sa.indel_score(a, b)
        assert got == expected, (
            f"indel({a!r}, {b!r}) = {got}, expected {expected}"
        )


def test_long_query_falls_back_to_scalar():
    # Bit-parallel SIMD only handles m <= 64; the scalar DP must
    # produce the same result for longer patterns.
    rng = random.Random(0xABCD)
    a = _rand(rng, 100)
    b = _rand(rng, 80)
    expected = _RFIndel.distance(a, b)
    assert sa.indel_score(a, b) == expected


def test_scores_batch():
    a = "kitten"
    targets = ["sitting", "kit", "mitten", ""]
    got = list(sa.indel_scores(a, targets))
    expected = [_RFIndel.distance(a, t) for t in targets]
    assert got == expected


def test_normalized_scores_batch():
    a = "abc"
    targets = ["abc", "abd", "xyz", "abcd"]
    got = list(sa.indel_normalized_scores(a, targets))
    for g, t in zip(got, targets):
        assert g == pytest.approx(_RFIndel.normalized_similarity(a, t))


def test_top_k_returns_lowest_distances():
    q = "kitten"
    ts = ["sitting", "kit", "mitten", "kitten", "foo"]
    out = sa.indel_top_k(q, ts, k=3)
    scores = sorted(s for _, s, _ in out)
    # 0 (kitten), 2 (mitten), 3 (kit)
    assert scores == [0, 2, 3]


def test_normalized_top_k_returns_highest_similarities():
    q = "kitten"
    ts = ["sitting", "kit", "mitten", "kitten", "foo"]
    out = sa.indel_normalized_top_k(q, ts, k=2)
    scores = sorted((s for _, s, _ in out), reverse=True)
    assert scores[0] == 1.0  # kitten itself


def test_best_returns_lowest_distance():
    q = "kitten"
    ts = ["sitting", "kit", "mitten", "foo"]
    obj, score, idx = sa.indel_best(q, ts)
    assert obj == "mitten"
    assert score == 2
    assert idx == 2


def test_empty_inputs():
    assert sa.indel_score("", "") == 0
    assert sa.indel_normalized_score("", "") == 1.0
    assert list(sa.indel_scores("abc", [])) == []
    assert sa.indel_top_k("abc", [], k=5) == []


def test_cdist_matrix():
    qs = ["ab", "cd"]
    ts = ["ab", "cd", "ef"]
    out = sa.cdist(qs, ts, scorer=sa.Scorer.INDEL)
    assert out.shape == (2, 3)
    assert out[0, 0] == 0
    assert out[1, 1] == 0
    assert out[0, 1] == 4  # delete ab, insert cd
    assert out[0, 2] == 4


def test_cdist_normalized_matrix():
    qs = ["ab", "cd"]
    ts = ["ab", "cd", "ef"]
    out = sa.cdist(qs, ts, scorer=sa.Scorer.INDEL_NORMALIZED)
    assert out[0, 0] == 1.0
    assert out[1, 1] == 1.0
    assert out[0, 1] == 0.0
    assert out[0, 2] == 0.0


def test_extract_by_enum():
    out = sa.extract(
        "kitten",
        ["sitting", "kit", "mitten"],
        scorer=sa.Scorer.INDEL_NORMALIZED,
        k=2,
    )
    scores = sorted((s for _, s, _ in out), reverse=True)
    assert scores[0] == pytest.approx(_RFIndel.normalized_similarity(
        "kitten", "mitten"))


def test_lcs_identity():
    """indel(a, b) + 2 * LCS(a, b) == |a| + |b| — verify via a scalar
    LCS computed independently."""
    def lcs_len(a, b):
        m, n = len(a), len(b)
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr
        return prev[n]

    rng = random.Random(7)
    for _ in range(50):
        a = _rand(rng, rng.randint(0, 30))
        b = _rand(rng, rng.randint(0, 30))
        d = sa.indel_score(a, b)
        l = lcs_len(a, b)
        assert d + 2 * l == len(a) + len(b)


# --------------------------------------------------------------------
# score_cutoff kernel-level early-exit
# --------------------------------------------------------------------

class TestIndelScoreCutoff:
    """``indel_score(s1, s2, score_cutoff=k)`` and
    ``indel_normalized_score`` push the cutoff into the bit-parallel
    kernel: when the kernel can prove the final distance will exceed
    ``k`` it bails and returns ``k + 1`` (any value > k carries the
    same 'doesn't qualify' signal). This matches rapidfuzz's
    convention."""

    def test_identity_returns_zero_with_cutoff(self) -> None:
        # cutoff doesn't apply to identical strings — distance = 0 < cutoff.
        assert sa.indel_score("hello", "hello", score_cutoff=5) == 0

    def test_cutoff_above_true_returns_true_distance(self) -> None:
        # cutoff well above the true distance: kernel runs to completion.
        assert sa.indel_score("hello", "hallo", score_cutoff=100) == 2

    def test_cutoff_below_true_returns_cutoff_plus_one_at_minimum(self) -> None:
        # Far-apart inputs with a tight cutoff: kernel bails. Result
        # must be strictly greater than the cutoff.
        result = sa.indel_score("a" * 10, "b" * 10, score_cutoff=3)
        assert result > 3

    def test_cutoff_none_matches_no_cutoff(self) -> None:
        # Passing score_cutoff=None must give the same answer as
        # omitting the kwarg.
        cases = [("hello", "world"), ("abc", "xyz"), ("a" * 30, "b" * 30)]
        for s1, s2 in cases:
            no_cutoff = sa.indel_score(s1, s2)
            with_none = sa.indel_score(s1, s2, score_cutoff=None)
            assert no_cutoff == with_none, (s1, s2)

    def test_normalized_cutoff_returns_zero_below_threshold(self) -> None:
        # normalized score_cutoff is in [0, 1]; below the threshold
        # returns 0.0.
        # 'hello' vs 'world' has indel similarity ~0.2; cutoff=0.5
        # should return 0.0.
        assert sa.indel_normalized_score("hello", "world", score_cutoff=0.5) == 0.0

    def test_normalized_cutoff_above_threshold_returns_true_value(self) -> None:
        # 'hello' vs 'hallo' has indel similarity ~0.8; cutoff=0.5
        # passes through unchanged.
        result = sa.indel_normalized_score("hello", "hallo", score_cutoff=0.5)
        assert result == pytest.approx(0.8, abs=1e-9)

    def test_cutoff_kernel_bails_on_long_disjoint_inputs(self) -> None:
        # Stress test: kernel must bail well before processing all
        # text chars when the inputs share no characters and cutoff
        # is tight.
        result = sa.indel_score("a" * 50, "b" * 50, score_cutoff=5)
        # The true distance is 100 (|a| + |b| - 2*LCS where LCS=0).
        # With cutoff=5 we must return something > 5 and ≤ true.
        assert 5 < result <= 100
