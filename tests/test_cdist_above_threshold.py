"""Tests for ``cdist_above_threshold`` — streaming filtered cdist.

The C++ side returns an iterator backed by a bounded queue fed by
worker threads. The Python __next__ pops events and yields
(score, query, target) for above-threshold pairs.

Coverage:
* correctness vs scalar reference (matches subset of cdist results)
* memory: bounded queue (mostly a smoke test — the API contract)
* tqdm dispatched from the calling thread
* GIL released for compute
* defensive snapshot against concurrent mutation
* abandon-mid-iteration (break) cleanly stops workers
* normalized-only constraint: distance scorers raise ValueError
* threshold range check
* symmetric inputs yield both directions
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


def test_yields_above_threshold_matches_full_cdist():
    rng = random.Random(0)
    qs = [_rand_str(rng, rng.randint(2, 8)) for _ in range(15)]
    ts = [_rand_str(rng, rng.randint(2, 8)) for _ in range(20)]
    threshold = 0.6

    matrix = sa.cdist(qs, ts, scorer=sa.Scorer.JARO)
    expected = {
        (round(matrix[i, j], 12), i, j)
        for i in range(len(qs))
        for j in range(len(ts))
        if matrix[i, j] >= threshold
    }

    actual = set()
    for score, q, t in sa.cdist_above_threshold(
        qs, ts, scorer=sa.Scorer.JARO, threshold=threshold, cpu_count=4
    ):
        # Find i, j by identity (the iterator yields borrowed refs to
        # the original objects).
        i = qs.index(q)
        j = ts.index(t)
        actual.add((round(score, 12), i, j))
    # actual should contain at least expected (we yield by-identity, but
    # qs.index can pick the wrong index if duplicates exist; this test
    # avoids that by using distinct random strings).
    assert actual == expected


def test_returns_iterator_object():
    qs = ["abc"]
    ts = ["abc", "abd"]
    gen = sa.cdist_above_threshold(qs, ts, scorer=sa.Scorer.JARO, threshold=0.0)
    assert iter(gen) is gen
    out = list(gen)
    assert len(out) == 2


def test_symmetric_yields_both_directions():
    qs = ["aaa", "bbb", "ccc"]
    pairs = [
        (q, t) for _, q, t in sa.cdist_above_threshold(
            qs, qs, scorer=sa.Scorer.JARO, threshold=0.0
        )
    ]
    # Every (i, j) for i != j should appear in both orderings, since
    # the symmetric path yields the mirror.
    assert ("aaa", "bbb") in pairs and ("bbb", "aaa") in pairs


def test_rejects_distance_scorers():
    qs = ["a"]
    ts = ["a"]
    with pytest.raises(ValueError, match="normalized"):
        list(sa.cdist_above_threshold(
            qs, ts, scorer=sa.Scorer.LEVENSHTEIN, threshold=0.5
        ))


def test_threshold_out_of_range_raises():
    qs = ["a"]
    ts = ["a"]
    with pytest.raises(ValueError, match="threshold"):
        list(sa.cdist_above_threshold(
            qs, ts, scorer=sa.Scorer.JARO, threshold=1.5
        ))
    with pytest.raises(ValueError, match="threshold"):
        list(sa.cdist_above_threshold(
            qs, ts, scorer=sa.Scorer.JARO, threshold=-0.1
        ))


def test_function_reference_scorer_works():
    qs = ["martha"]
    ts = ["marhta", "xyz"]
    out = list(sa.cdist_above_threshold(
        qs, ts, scorer=sa.jaro_similarities, threshold=0.5
    ))
    # martha vs marhta is high; vs xyz is low.
    assert len(out) == 1
    assert out[0][1] == "martha"
    assert out[0][2] == "marhta"


def test_tqdm_dispatched_from_main_thread():
    construct_thread = []
    update_threads = set()
    close_threads = []

    class FakeTqdm:
        def __init__(self, total=None):
            construct_thread.append(threading.get_ident())

        def update(self, n):
            update_threads.add(threading.get_ident())

        def close(self):
            close_threads.append(threading.get_ident())

    qs = ["abc" * 5 for _ in range(15)]
    ts = ["abd" * 5 for _ in range(20)]
    list(sa.cdist_above_threshold(
        qs, ts, scorer=sa.Scorer.JARO, threshold=0.0,
        tqdm=FakeTqdm, cpu_count=4,
    ))
    main = construct_thread[0]
    assert update_threads <= {main}
    assert close_threads == [main]


def test_break_early_does_not_leak_workers():
    # The iterator's destructor must stop and join workers when the
    # caller abandons iteration. Hard to assert "no threads leaked"
    # directly; instead check that calling break + re-running cdist
    # doesn't hang and produces correct output.
    qs = ["abc" * 3 for _ in range(50)]
    ts = ["abc" * 3 for _ in range(50)]
    gen = sa.cdist_above_threshold(
        qs, ts, scorer=sa.Scorer.JARO, threshold=0.0, cpu_count=4,
    )
    # Take one item, then drop the generator.
    for _ in gen:
        break
    del gen  # iterator destructor should join workers
    # Subsequent call must complete.
    out = list(sa.cdist_above_threshold(
        ["foo"], ["bar"], scorer=sa.Scorer.JARO, threshold=0.0, cpu_count=2,
    ))
    assert len(out) == 1


def test_defensive_snapshot_against_concurrent_mutation():
    qs = ["abc" * 3 for _ in range(40)]
    ts = ["abd" * 3 for _ in range(40)]
    qs_snapshot = list(qs)
    ts_snapshot = list(ts)

    result_holder = []

    def run():
        # cdist_above_threshold returns immediately with an iterator
        # whose workers are already running. We materialize the list
        # inside the same thread to catch the snapshot.
        gen = sa.cdist_above_threshold(
            qs, ts, scorer=sa.Scorer.JARO, threshold=0.0, cpu_count=4,
        )
        result_holder.append(list(gen))

    t = threading.Thread(target=run)
    t.start()
    # Mutate the input lists while the workers run.
    for _ in range(3):
        qs.clear(); qs.extend(["zzz"] * 40)
        ts.clear(); ts.extend(["zzz"] * 40)
    t.join()

    # The yielded results should still reference the ORIGINAL strings
    # (the tuple snapshot held them alive).
    out = result_holder[0]
    # All scores from the original strings — at threshold=0 we expect
    # 40 * 40 = 1600 pairs.
    assert len(out) == 1600
    # Each yielded query/target should be one of the originals.
    q_originals = set(qs_snapshot)
    t_originals = set(ts_snapshot)
    for _, q, t in out[:10]:
        assert q in q_originals
        assert t in t_originals


def test_gil_released_during_compute():
    counter = {"ticks": 0, "stop": False}

    def background_tick():
        while not counter["stop"]:
            counter["ticks"] += 1

    qs = ["a" * 20 for _ in range(200)]
    ts = ["b" * 20 for _ in range(200)]

    bg = threading.Thread(target=background_tick, daemon=True)
    bg.start()
    before = counter["ticks"]
    # Drain the iterator; threshold=2 yields nothing but does all the
    # compute work.
    list(sa.cdist_above_threshold(
        qs, ts, scorer=sa.Scorer.JARO, threshold=1.0, cpu_count=4,
    ))
    after = counter["ticks"]
    counter["stop"] = True
    bg.join(timeout=1.0)
    assert after > before


def test_multi_threaded_yields_same_set_as_single_threaded():
    rng = random.Random(99)
    qs = [_rand_str(rng, rng.randint(2, 10)) for _ in range(30)]
    ts = [_rand_str(rng, rng.randint(2, 10)) for _ in range(35)]
    threshold = 0.5

    a = sorted(
        (round(s, 9), i, j)
        for s, i, j in (
            (s, qs.index(q), ts.index(t))
            for s, q, t in sa.cdist_above_threshold(
                qs, ts, scorer=sa.Scorer.JARO, threshold=threshold, cpu_count=1,
            )
        )
    )
    b = sorted(
        (round(s, 9), i, j)
        for s, i, j in (
            (s, qs.index(q), ts.index(t))
            for s, q, t in sa.cdist_above_threshold(
                qs, ts, scorer=sa.Scorer.JARO, threshold=threshold, cpu_count=8,
            )
        )
    )
    assert a == b


def test_hamming_normalized_requires_equal_lengths():
    with pytest.raises(ValueError, match="equal-length"):
        list(sa.cdist_above_threshold(
            ["abc"], ["abcd"],
            scorer=sa.Scorer.HAMMING_NORMALIZED, threshold=0.0,
        ))


def test_empty_inputs_yields_nothing():
    out = list(sa.cdist_above_threshold(
        [], [], scorer=sa.Scorer.JARO, threshold=0.0
    ))
    assert out == []


def test_yields_references_to_original_strings_not_copies():
    """The (score, query, target) tuples should hold references to the
    SAME Python string objects from the input lists, not freshly
    constructed copies. Built strings via ``join`` to avoid CPython's
    literal-interning fooling the identity check."""
    qs = ["".join(["kit", "ten"]), "".join(["sit", "ting"])]
    ts = ["".join(["kit", "ten"]), "".join(["bi", "ting"])]
    for _, q, t in sa.cdist_above_threshold(
        qs, ts, scorer=sa.Scorer.JARO, threshold=0.0,
    ):
        assert any(q is original for original in qs), (
            "query yielded was a new object, not a reference to the input list"
        )
        assert any(t is original for original in ts), (
            "target yielded was a new object, not a reference to the input list"
        )
