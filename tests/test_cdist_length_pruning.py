"""Length-difference pruning correctness for ``cdist_above_threshold``
and ``cdist_top_k``.

Pruning short-circuits SIMD work for pairs whose closed-form upper
bound on the normalized similarity is below the active threshold.
These tests verify the optimization is sound (same set of results as
the un-pruned reference, the full ``cdist`` matrix) across every
scorer the pruning helper supports, at multiple thresholds, and on
inputs designed to span the pruning regime (wide length variation).
"""

from __future__ import annotations

import random
import string

import numpy as np
import pytest

import stride_align as sa


def _rand_str(rng, n):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def _varied_lengths(rng, count, lo, hi):
    """Inputs whose lengths span [lo, hi] so length-difference pruning
    actually engages — equal-length inputs would let everything through."""
    return [_rand_str(rng, rng.randint(lo, hi)) for _ in range(count)]


# Pruning is meaningful for these scorers. Hamming requires equal-length
# inputs (no pruning possible). Pure-distance scorers don't go through
# the threshold/top-k variants at all.
PRUNED_SCORERS = [
    sa.Scorer.JARO,
    sa.Scorer.JARO_WINKLER,
    sa.Scorer.LEVENSHTEIN_NORMALIZED,
    sa.Scorer.DAMERAU_LEVENSHTEIN_NORMALIZED,
    sa.Scorer.INDEL_NORMALIZED,
]


@pytest.mark.parametrize("scorer", PRUNED_SCORERS)
@pytest.mark.parametrize("threshold", [0.3, 0.6, 0.8, 0.95])
def test_above_threshold_matches_full_cdist_with_wide_length_spread(
    scorer, threshold
):
    rng = random.Random(scorer.value * 1009 + int(threshold * 100))
    qs = _varied_lengths(rng, 25, 2, 30)
    ts = _varied_lengths(rng, 30, 2, 30)

    full = sa.cdist(qs, ts, scorer=scorer)
    expected = {
        (round(full[i, j], 9), i, j)
        for i in range(len(qs))
        for j in range(len(ts))
        if full[i, j] >= threshold
    }

    actual = set()
    for score, q, t in sa.cdist_above_threshold(
        qs, ts, scorer=scorer, threshold=threshold, cpu_count=2
    ):
        actual.add((round(score, 9), qs.index(q), ts.index(t)))

    assert actual == expected


@pytest.mark.parametrize("scorer", PRUNED_SCORERS)
@pytest.mark.parametrize("k", [1, 5, 25, 100])
def test_top_k_matches_full_cdist_with_wide_length_spread(scorer, k):
    rng = random.Random(scorer.value * 7919 + k)
    qs = _varied_lengths(rng, 30, 2, 30)
    ts = _varied_lengths(rng, 35, 2, 30)

    full = sa.cdist(qs, ts, scorer=scorer).flatten()
    expected_top = sorted(full, reverse=True)[: min(k, full.size)]

    out = sa.cdist_top_k(qs, ts, scorer=scorer, k=k, cpu_count=2)
    actual_scores = sorted((s for s, _, _ in out), reverse=True)

    assert actual_scores == pytest.approx(expected_top)


@pytest.mark.parametrize("scorer", PRUNED_SCORERS)
def test_above_threshold_extreme_length_spread_prunes_safely(scorer):
    # One short query against many long targets — pruning should
    # eliminate almost every pair for the high-threshold cases but
    # results must still match the un-pruned cdist exactly.
    qs = ["abc"]
    ts = ["abcdefghijklmnop" * 4, "ab", "xyz" * 10, "abcde", "abc"]
    threshold = 0.7

    full = sa.cdist(qs, ts, scorer=scorer)
    expected = sorted(
        round(full[0, j], 9)
        for j in range(len(ts))
        if full[0, j] >= threshold
    )

    actual = sorted(
        round(score, 9)
        for score, _, _ in sa.cdist_above_threshold(
            qs, ts, scorer=scorer, threshold=threshold
        )
    )
    assert actual == expected


@pytest.mark.parametrize("scorer", PRUNED_SCORERS)
def test_top_k_with_cpu_count_one_matches_full(scorer):
    # Single-threaded path uses the same prune-threshold snapshot as
    # the multi-threaded path; verify it's still correct without the
    # cross-thread atomic ever being contended.
    rng = random.Random(scorer.value * 31 + 7)
    qs = _varied_lengths(rng, 20, 3, 20)
    ts = _varied_lengths(rng, 25, 3, 20)
    k = 10

    full = sa.cdist(qs, ts, scorer=scorer).flatten()
    expected_top = sorted(full, reverse=True)[:k]

    out = sa.cdist_top_k(qs, ts, scorer=scorer, k=k, cpu_count=1)
    actual_scores = sorted((s for s, _, _ in out), reverse=True)
    assert actual_scores == pytest.approx(expected_top)


def test_top_k_global_bound_does_not_lose_results_under_contention():
    # Stress test: many threads, small k, wide length spread, run
    # multiple times. The atomic bound is monotonic so a stale read
    # never prunes a pair that should have been kept; but under heavy
    # contention any silent off-by-one would show up across repeats.
    rng = random.Random(0xC0FFEE)
    qs = _varied_lengths(rng, 60, 2, 30)
    ts = _varied_lengths(rng, 70, 2, 30)
    k = 15

    full = sa.cdist(qs, ts, scorer=sa.Scorer.JARO_WINKLER).flatten()
    expected_top = sorted(full, reverse=True)[:k]

    for _ in range(5):
        out = sa.cdist_top_k(
            qs, ts, scorer=sa.Scorer.JARO_WINKLER, k=k, cpu_count=2
        )
        actual_scores = sorted((s for s, _, _ in out), reverse=True)
        assert actual_scores == pytest.approx(expected_top)


@pytest.mark.parametrize(
    "scorer",
    [sa.Scorer.LEVENSHTEIN_NORMALIZED, sa.Scorer.DAMERAU_LEVENSHTEIN_NORMALIZED],
)
@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
def test_kernel_cutoff_pushdown_does_not_change_above_threshold_results(
    scorer, threshold
):
    """Pushing the threshold into the SIMD kernel as a per-pair distance
    cutoff lets bailed lanes return cutoff+1 instead of the true
    distance — but only when the true distance exceeds the cutoff
    (i.e. when sim < threshold). The downstream `sim >= threshold`
    filter still has to produce the same accept/reject set as the
    no-cutoff baseline. Verify by comparison against the full cdist
    matrix."""
    rng = random.Random(hash((scorer.value, threshold)) & 0xFFFF)
    qs = _varied_lengths(rng, 25, 2, 50)
    ts = _varied_lengths(rng, 30, 2, 50)

    full = sa.cdist(qs, ts, scorer=scorer)
    expected = {
        (round(full[i, j], 9), i, j)
        for i in range(len(qs))
        for j in range(len(ts))
        if full[i, j] >= threshold
    }

    actual = set()
    for score, q, t in sa.cdist_above_threshold(
        qs, ts, scorer=scorer, threshold=threshold, cpu_count=2,
    ):
        actual.add((round(score, 9), qs.index(q), ts.index(t)))

    assert actual == expected


@pytest.mark.parametrize(
    "scorer",
    [sa.Scorer.LEVENSHTEIN_NORMALIZED, sa.Scorer.DAMERAU_LEVENSHTEIN_NORMALIZED],
)
@pytest.mark.parametrize("k", [1, 5, 25])
def test_kernel_cutoff_pushdown_does_not_change_top_k_results(scorer, k):
    """Same idea for cdist_top_k: the dynamic global heap-min bound
    gets pushed into Lev/OSA kernels as a per-pair cutoff. The top-k
    set must equal the unbounded reference."""
    rng = random.Random(hash((scorer.value, k)) & 0xFFFF)
    qs = _varied_lengths(rng, 30, 2, 50)
    ts = _varied_lengths(rng, 35, 2, 50)

    full = sa.cdist(qs, ts, scorer=scorer).flatten()
    expected_top = sorted(full, reverse=True)[: min(k, full.size)]

    out = sa.cdist_top_k(qs, ts, scorer=scorer, k=k, cpu_count=2)
    actual_scores = sorted((s for s, _, _ in out), reverse=True)
    assert actual_scores == pytest.approx(expected_top)


@pytest.mark.parametrize("n", [16, 50, 100, 200])
@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 1.0])
def test_hamming_cutoff_pushdown_does_not_change_above_threshold_results(
    n, threshold
):
    """Hamming's kernel-level cutoff bails the inner loop once
    accumulated mismatches exceed the cutoff. Verify the bailed
    sentinel maps to sim < threshold so the downstream filter agrees
    with the no-cutoff full cdist."""
    rng = random.Random(hash((n, threshold)) & 0xFFFF)
    qs = [_rand_str(rng, n) for _ in range(20)]
    ts = [_rand_str(rng, n) for _ in range(25)]

    full = sa.cdist(qs, ts, scorer=sa.Scorer.HAMMING_NORMALIZED)
    expected = {
        (round(full[i, j], 9), i, j)
        for i in range(len(qs))
        for j in range(len(ts))
        if full[i, j] >= threshold
    }

    actual = set()
    for score, q, t in sa.cdist_above_threshold(
        qs, ts, scorer=sa.Scorer.HAMMING_NORMALIZED, threshold=threshold,
        cpu_count=2,
    ):
        actual.add((round(score, 9), qs.index(q), ts.index(t)))

    assert actual == expected


@pytest.mark.parametrize("n", [10, 20, 100])
@pytest.mark.parametrize("k", [1, 5, 20])
def test_hamming_cutoff_pushdown_does_not_change_top_k_results(n, k):
    """The cdist_top_k path passes the dynamic heap-min as the cutoff;
    verify top-k results still match the full cdist sort."""
    rng = random.Random(hash((n, k)) & 0xFFFF)
    qs = [_rand_str(rng, n) for _ in range(25)]
    ts = [_rand_str(rng, n) for _ in range(30)]

    full = sa.cdist(qs, ts, scorer=sa.Scorer.HAMMING_NORMALIZED).flatten()
    expected_top = sorted(full, reverse=True)[: min(k, full.size)]

    out = sa.cdist_top_k(
        qs, ts, scorer=sa.Scorer.HAMMING_NORMALIZED, k=k, cpu_count=2,
    )
    actual_scores = sorted((s for s, _, _ in out), reverse=True)
    assert actual_scores == pytest.approx(expected_top)


def test_hamming_cutoff_fp_boundary_thresholds():
    """For Hamming, (1-T)*n is the cutoff math; pick (T, n) pairs that
    land on integer boundaries to expose FP-floor regressions."""
    for n in [10, 20, 50, 100, 200]:
        qs = ["a" * n, "b" * n, "a" * (n // 2) + "b" * (n - n // 2)]
        ts = ["a" * n, "c" * n, "b" * n]
        # T values where (1-T)*n is mathematically integer for these n's:
        # 0.5 * n is integer for even n, 0.75 * n is integer for n%4==0.
        for threshold in [0.5, 0.6, 0.75, 0.8, 0.9]:
            full = sa.cdist(qs, ts, scorer=sa.Scorer.HAMMING_NORMALIZED)
            expected = {
                (round(full[i, j], 9), i, j)
                for i in range(len(qs))
                for j in range(len(ts))
                if full[i, j] >= threshold
            }
            actual = set()
            for score, q, t in sa.cdist_above_threshold(
                qs, ts, scorer=sa.Scorer.HAMMING_NORMALIZED, threshold=threshold,
            ):
                actual.add((round(score, 9), qs.index(q), ts.index(t)))
            assert actual == expected, (
                f"Hamming FP-boundary regression at n={n} threshold={threshold}: "
                f"missing={expected - actual} extra={actual - expected}"
            )


@pytest.mark.parametrize(
    "scorer",
    [sa.Scorer.LEVENSHTEIN_NORMALIZED, sa.Scorer.DAMERAU_LEVENSHTEIN_NORMALIZED],
)
def test_kernel_cutoff_pushdown_fp_boundary_thresholds(scorer):
    """Thresholds that put (1-T)*max(q,t) at an integer boundary are the
    floating-point danger zone for the cutoff helper. Force those
    edge cases by picking thresholds where (1-T)*max is exactly
    integral for likely max values."""
    qs = ["a" * n for n in [5, 10, 20, 25, 30]]
    ts = ["b" * n for n in [5, 10, 20, 25, 30]]

    # 0.6 * 10 = 6.0, 0.6 * 20 = 12.0, etc. — integer boundaries that
    # FP arithmetic may render as 5.9999... or 6.0000....
    for threshold in [0.4, 0.5, 0.6, 0.75, 0.8]:
        full = sa.cdist(qs, ts, scorer=scorer)
        expected = {
            (round(full[i, j], 9), i, j)
            for i in range(len(qs))
            for j in range(len(ts))
            if full[i, j] >= threshold
        }
        actual = set()
        for score, q, t in sa.cdist_above_threshold(
            qs, ts, scorer=scorer, threshold=threshold,
        ):
            actual.add((round(score, 9), qs.index(q), ts.index(t)))
        assert actual == expected, (
            f"FP-boundary regression for scorer={scorer.name} threshold={threshold}: "
            f"missing={expected - actual} extra={actual - expected}"
        )


def test_above_threshold_threshold_one_yields_only_exact_normalized_matches():
    # threshold=1.0 means "only pairs that can hit max_normalized
    # similarity 1.0" — for Lev/JW that requires equal length. The
    # bound is tight enough that all non-equal-length pairs prune.
    qs = ["abc", "abcd", "xyz"]
    ts = ["abc", "abcd", "abcde"]
    out = list(
        sa.cdist_above_threshold(
            qs, ts, scorer=sa.Scorer.LEVENSHTEIN_NORMALIZED, threshold=1.0,
        )
    )
    actual = {(round(s, 12), q, t) for s, q, t in out}
    # Only "abc" == "abc" and "abcd" == "abcd" can reach score 1.0.
    assert actual == {
        (1.0, "abc", "abc"),
        (1.0, "abcd", "abcd"),
    }
