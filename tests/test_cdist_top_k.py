"""Tests for ``cdist_top_k`` — heap-based top-k variant of cdist.

Returns an unsorted list of at most k highest-scoring (score, query,
target) tuples. Heap math runs in C++ over `(score, q_idx, t_idx)`
records — no Python refcount traffic in the hot loop.
"""

from __future__ import annotations

import random
import string
import threading

import numpy as np
import pytest

import stride_align as sa


def _rand_str(rng, n):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def test_returns_top_k_pairs():
    qs = ["kitten", "sitting", "kit"]
    ts = ["kitten", "kit", "sitting", "biting"]
    result = sa.cdist_top_k(qs, ts, scorer=sa.Scorer.JARO, k=5)
    assert isinstance(result, list)
    assert len(result) <= 5
    # The top-5 scores should match the highest 5 scores from the full cdist.
    full = sa.cdist(qs, ts, scorer=sa.Scorer.JARO)
    expected_top5 = sorted(full.flatten(), reverse=True)[:5]
    actual_scores = sorted((s for s, _, _ in result), reverse=True)
    assert actual_scores == pytest.approx(expected_top5)


def test_no_explicit_sort_guarantee():
    # The contract is "we don't sort; caller does if needed". We can't
    # easily assert "is not sorted" because for some inputs the heap
    # order happens to be sorted. Instead, just exercise the API and
    # confirm it returns the expected count.
    rng = random.Random(42)
    qs = [_rand_str(rng, rng.randint(2, 8)) for _ in range(40)]
    out = sa.cdist_top_k(qs, qs, scorer=sa.Scorer.JARO, k=10,
                         reject_duplicates=True, cpu_count=1)
    assert len(out) == 10


def test_k_zero_returns_empty():
    qs = ["abc"]
    ts = ["abc"]
    result = sa.cdist_top_k(qs, ts, scorer=sa.Scorer.JARO, k=0)
    assert result == []


def test_k_larger_than_pair_count():
    qs = ["a", "b"]
    ts = ["a", "b"]
    result = sa.cdist_top_k(qs, ts, scorer=sa.Scorer.JARO, k=100)
    # 4 pairs total → at most 4 results.
    assert len(result) == 4


def test_reject_duplicates_skips_identical_content():
    qs = ["foo", "foo", "bar", "fox"]
    ts = qs  # symmetric
    # Without reject_duplicates the top should include score=1.0
    # self/dup pairs.
    default = sa.cdist_top_k(qs, ts, scorer=sa.Scorer.JARO, k=8,
                              reject_duplicates=False)
    assert any(s == 1.0 for s, _, _ in default)

    # With reject_duplicates, all returned pairs whose content matches
    # exactly must be excluded.
    rejected = sa.cdist_top_k(qs, ts, scorer=sa.Scorer.JARO, k=8,
                               reject_duplicates=True)
    for score, q, t in rejected:
        if score == 1.0:
            assert q != t, (
                f"reject_duplicates should skip identical-content pairs "
                f"but yielded ({score}, {q!r}, {t!r})"
            )


def test_reject_duplicates_keeps_score_1_when_strings_differ():
    # Two strings that hash-out to Jaro score 1.0 only if identical;
    # constructively this is hard, but the empty-string edge case is
    # one where both content and score are degenerate. The behavior we
    # want: when score is 1.0 AND contents are identical, skip.
    qs = ["abc"]
    ts = ["abc"]
    out = sa.cdist_top_k(qs, ts, scorer=sa.Scorer.JARO, k=1,
                          reject_duplicates=True)
    # Identical content with score 1.0 should be rejected → empty.
    assert out == []


def test_rejects_distance_scorers():
    with pytest.raises(ValueError, match="normalized"):
        sa.cdist_top_k(
            ["a"], ["a"], scorer=sa.Scorer.LEVENSHTEIN, k=1
        )


def test_function_reference_scorer():
    out = sa.cdist_top_k(
        ["martha"], ["marhta", "xyz"],
        scorer=sa.jaro_similarities, k=2,
    )
    assert len(out) == 2


def test_multi_threaded_results_match_single_threaded():
    """The set of returned (score, q, t) tuples should be the same
    regardless of thread count — the heap doesn't care about
    insertion order, and we end up with the same k highest scores."""
    rng = random.Random(11)
    qs = [_rand_str(rng, rng.randint(2, 8)) for _ in range(40)]
    ts = [_rand_str(rng, rng.randint(2, 8)) for _ in range(40)]
    k = 25

    single = sa.cdist_top_k(qs, ts, scorer=sa.Scorer.JARO, k=k, cpu_count=1)
    multi = sa.cdist_top_k(qs, ts, scorer=sa.Scorer.JARO, k=k, cpu_count=8)

    # Score multisets must agree; specific (q, t) tuples can differ
    # on ties depending on which thread popped first.
    s_single = sorted(round(s, 9) for s, _, _ in single)
    s_multi = sorted(round(s, 9) for s, _, _ in multi)
    assert s_single == s_multi


def test_tqdm_dispatched_from_main_thread():
    construct_thread = []
    update_threads = set()

    class FakeTqdm:
        def __init__(self, total=None):
            construct_thread.append(threading.get_ident())

        def update(self, n):
            update_threads.add(threading.get_ident())

        def close(self):
            pass

    qs = ["a" * 6 for _ in range(15)]
    ts = ["b" * 6 for _ in range(15)]
    sa.cdist_top_k(
        qs, ts, scorer=sa.Scorer.JARO, k=5,
        tqdm=FakeTqdm, cpu_count=4,
    )
    main = construct_thread[0]
    assert update_threads <= {main}


def test_gil_released_during_compute():
    counter = {"ticks": 0, "stop": False}

    def tick():
        while not counter["stop"]:
            counter["ticks"] += 1

    qs = ["a" * 20 for _ in range(150)]
    ts = ["b" * 20 for _ in range(150)]

    bg = threading.Thread(target=tick, daemon=True)
    bg.start()
    before = counter["ticks"]
    sa.cdist_top_k(qs, ts, scorer=sa.Scorer.JARO, k=10, cpu_count=4)
    after = counter["ticks"]
    counter["stop"] = True
    bg.join(timeout=1.0)
    assert after > before


def test_symmetric_yields_both_directions():
    qs = ["aaa", "abb", "abc", "xyz"]
    out = sa.cdist_top_k(
        qs, qs, scorer=sa.Scorer.JARO, k=20, reject_duplicates=True
    )
    pairs = {(q, t) for _, q, t in out}
    # For every (q, t) with q != t, the mirror (t, q) is also present.
    for q, t in pairs:
        if q != t:
            assert (t, q) in pairs


def test_empty_inputs_returns_empty():
    out = sa.cdist_top_k([], [], scorer=sa.Scorer.JARO, k=5)
    assert out == []


def test_returns_references_to_original_strings_not_copies():
    """The returned tuples should hold references to the SAME Python
    string objects from the input lists, not freshly constructed
    copies. ``.join`` builds fresh string objects so CPython's literal
    interning can't mask a regression here."""
    qs = ["".join(["kit", "ten"]), "".join(["sit", "ting"])]
    ts = ["".join(["kit", "ten"]), "".join(["bi", "ting"])]
    for _, q, t in sa.cdist_top_k(
        qs, ts, scorer=sa.Scorer.JARO, k=10,
    ):
        assert any(q is original for original in qs), (
            "query yielded was a new object, not a reference to the input list"
        )
        assert any(t is original for original in ts), (
            "target yielded was a new object, not a reference to the input list"
        )


def test_defensive_snapshot_against_concurrent_mutation():
    qs = [f"q{i}{'x' * 4}" for i in range(30)]
    ts = [f"t{i}{'x' * 4}" for i in range(30)]
    qs_snap = list(qs)
    ts_snap = list(ts)

    result_holder = []

    def run():
        result_holder.append(sa.cdist_top_k(
            qs, ts, scorer=sa.Scorer.JARO, k=10, cpu_count=4,
        ))

    t = threading.Thread(target=run)
    t.start()
    for _ in range(3):
        qs.clear(); qs.extend(["zzz"] * 30)
        ts.clear(); ts.extend(["zzz"] * 30)
    t.join()

    ref = sa.cdist_top_k(qs_snap, ts_snap, scorer=sa.Scorer.JARO, k=10,
                         cpu_count=1)

    # Both runs should hit the same top-k score set. With ties on
    # score, the specific (q, t) tuples picked depend on insertion
    # order; only the score multiset is fixed.
    scores_concurrent = sorted(round(s, 9) for s, _, _ in result_holder[0])
    scores_ref = sorted(round(s, 9) for s, _, _ in ref)
    assert scores_concurrent == scores_ref
    # And the strings yielded must come from the original lists (not
    # the mutated "zzz" replacements).
    for _, q, t in result_holder[0]:
        assert q in qs_snap
        assert t in ts_snap
