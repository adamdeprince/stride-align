"""Tests for `cdist` — all-pairs distance / similarity matrix.

The C++ implementation is a single Python boundary crossing (per
call) plus optional tqdm callbacks. Per-row dispatch is a C++ switch
keyed off either a Scorer IntEnum or a registered function reference.

Coverage:
* dtype matches scorer (int64 for distances, float64 for similarity).
* matches the scalar reference per-pair (random battery).
* symmetric optimization: identical to non-symmetric when
  queries IS targets.
* function-reference dispatch matches enum dispatch.
* tqdm callback receives a cost-weighted total and per-row updates
  that sum to the total.
"""

from __future__ import annotations

import random
import string

import numpy as np
import pytest

import stride_align as sa
import stride_align._generic as _generic


def _rand_str(rng, n):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def _ref_lev(qs, ts):
    return np.array(
        [[_generic.levenshtein_score(q, t) for t in ts] for q in qs],
        dtype=np.int64,
    )


def _ref_jaro(qs, ts):
    return np.array(
        [[_generic.jaro_similarity(q, t) for t in ts] for q in qs]
    )


def test_cdist_levenshtein_matches_scalar_reference():
    qs = ["kitten", "sitting", "kit"]
    ts = ["kitten", "sitting", "kit", "biting"]
    actual = sa.cdist(qs, ts, scorer=sa.Scorer.LEVENSHTEIN)
    assert actual.dtype == np.int64
    assert actual.shape == (3, 4)
    assert np.array_equal(actual, _ref_lev(qs, ts))


def test_cdist_returns_float_for_similarity_scorers():
    qs = ["kitten", "sitting"]
    ts = ["kitten", "kit"]
    for s in (
        sa.Scorer.LEVENSHTEIN_NORMALIZED,
        sa.Scorer.DAMERAU_LEVENSHTEIN_NORMALIZED,
        sa.Scorer.JARO,
        sa.Scorer.JARO_WINKLER,
    ):
        actual = sa.cdist(qs, ts, scorer=s)
        assert actual.dtype == np.float64, s


def test_cdist_jaro_matches_scalar_reference():
    qs = ["martha", "dwayne", "dixon"]
    ts = ["marhta", "duane", "dicksonx", "no_match"]
    actual = sa.cdist(qs, ts, scorer=sa.Scorer.JARO)
    assert np.allclose(actual, _ref_jaro(qs, ts), atol=1e-12)


def test_cdist_random_battery_lev():
    rng = random.Random(7)
    for _ in range(20):
        n = rng.randint(1, 25)
        m = rng.randint(1, 25)
        qs = [_rand_str(rng, rng.randint(0, 30)) for _ in range(n)]
        ts = [_rand_str(rng, rng.randint(0, 30)) for _ in range(m)]
        actual = sa.cdist(qs, ts, scorer=sa.Scorer.LEVENSHTEIN)
        assert np.array_equal(actual, _ref_lev(qs, ts))


def test_cdist_symmetric_input_is_symmetric_output():
    qs = ["kitten", "sitting", "kit", "biting"]
    actual = sa.cdist(qs, qs, scorer=sa.Scorer.LEVENSHTEIN)
    # Output should be square and symmetric.
    assert actual.shape == (4, 4)
    assert np.array_equal(actual, actual.T)
    # Diagonal is zero for distance scorers.
    assert np.all(np.diag(actual) == 0)
    # Matches non-symmetric path when we pass a copy (different object).
    qs_copy = list(qs)
    asym = sa.cdist(qs, qs_copy, scorer=sa.Scorer.LEVENSHTEIN)
    assert np.array_equal(actual, asym)


def test_cdist_symmetric_diagonal_for_similarity_is_one():
    qs = ["alpha", "beta", "gamma"]
    actual = sa.cdist(qs, qs, scorer=sa.Scorer.JARO)
    assert np.allclose(np.diag(actual), 1.0)


def test_cdist_function_reference_dispatch_matches_enum():
    qs = ["abc", "xyz"]
    ts = ["abc", "xyz", "abx"]
    via_enum = sa.cdist(qs, ts, scorer=sa.Scorer.LEVENSHTEIN)
    via_fnref = sa.cdist(qs, ts, scorer=sa.levenshtein_scores)
    assert np.array_equal(via_enum, via_fnref)


def test_cdist_function_reference_dispatch_jaro():
    qs = ["martha"]
    ts = ["marhta"]
    via_enum = sa.cdist(qs, ts, scorer=sa.Scorer.JARO)
    via_fnref = sa.cdist(qs, ts, scorer=sa.jaro_similarities)
    assert np.array_equal(via_enum, via_fnref)


def test_cdist_unknown_scorer_raises():
    with pytest.raises(ValueError, match="cdist scorer"):
        sa.cdist(["a"], ["b"], scorer=lambda *a, **k: None)


def test_cdist_tqdm_total_and_updates_sum_to_total():
    calls = []

    class FakeTqdm:
        def __init__(self, total=None):
            calls.append(("init", total))
            self.total = total

        def update(self, n):
            calls.append(("update", n))

        def close(self):
            calls.append(("close",))

    qs = ["kitten", "sitting", "kit"]
    ts = ["kitten", "sitting", "kit", "biting"]
    sa.cdist(qs, ts, scorer=sa.Scorer.LEVENSHTEIN, tqdm=FakeTqdm)

    # init -> update*N -> close
    assert calls[0][0] == "init"
    assert calls[-1] == ("close",)
    total = calls[0][1]
    update_sum = sum(c[1] for c in calls if c[0] == "update")
    assert update_sum == total


def test_cdist_tqdm_symmetric_updates_shrink():
    """In symmetric mode the upper-triangle rows shrink — each row's
    update should be smaller than the previous row's. The total still
    matches the sum of updates."""
    calls = []

    class FakeTqdm:
        def __init__(self, total=None):
            self.total = total

        def update(self, n):
            calls.append(n)

        def close(self):
            pass

    qs = ["aa", "bb", "cc", "dd", "ee"]
    sa.cdist(qs, qs, scorer=sa.Scorer.LEVENSHTEIN, tqdm=FakeTqdm)
    # The non-zero updates correspond to rows 0..3 (row 4 has no upper-
    # triangle work). The sequence must be monotonically decreasing.
    nonzero = [n for n in calls if n > 0]
    assert nonzero == sorted(nonzero, reverse=True)
    assert len(nonzero) == 4  # rows 0..3


def test_cdist_hamming_requires_equal_lengths():
    with pytest.raises(ValueError, match="equal-length"):
        sa.cdist(["abc"], ["ab"], scorer=sa.Scorer.HAMMING)


def test_cdist_empty_inputs():
    actual = sa.cdist([], [], scorer=sa.Scorer.LEVENSHTEIN)
    assert actual.shape == (0, 0)


def test_cdist_n_target_long_strings_within_simd_cap():
    # Multi-word query path triggered when q_len > 64.
    qs = ["a" * 100, "b" * 80]
    ts = ["a" * 100, "ab" * 50]
    actual = sa.cdist(qs, ts, scorer=sa.Scorer.LEVENSHTEIN)
    assert actual.dtype == np.int64
    assert np.array_equal(actual, _ref_lev(qs, ts))


def test_cdist_above_simd_cap_raises():
    long_q = "a" * 300
    with pytest.raises(NotImplementedError, match="cdist: query"):
        sa.cdist([long_q], ["a" * 10], scorer=sa.Scorer.LEVENSHTEIN)


def test_cdist_jaro_winkler_uses_prefix_kwargs():
    qs = ["martha"]
    ts = ["marhta"]
    default = sa.cdist(qs, ts, scorer=sa.Scorer.JARO_WINKLER)
    heavier = sa.cdist(
        qs, ts, scorer=sa.Scorer.JARO_WINKLER, prefix_weight=0.2
    )
    # Higher prefix_weight magnifies the bonus when above threshold.
    assert heavier[0, 0] > default[0, 0]
