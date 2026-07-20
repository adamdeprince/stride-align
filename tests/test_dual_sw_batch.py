"""Correctness: 1:many SW batch scores match sequential singles.

On AVX-512 hosts with query length 1024 and width 16 the batch path uses the
dual-target deferred exact-fill kernel. Comparing batch vs per-target singles
exercises dual equal-length, unequal lengths, odd counts, and empty targets
without depending on microbench-only strategy flags.

Strategy fallthrough (materialized / bounded) is covered by the native
``stride_align_x86_microbench --verify-dual`` harness on avx10.
"""
from __future__ import annotations

import random

import pytest

import stride_align as sa

MATCH, MISMATCH, GAP = 2, -1, -1


def _english(rng: random.Random, n: int) -> str:
    corpus = (
        "The quick brown fox watches the city wake under a low grey sky. "
        "People cross the station concourse with coffee, folded papers, and quiet plans. "
        "A street musician repeats a careful phrase while buses hiss at the curb. "
        "In the office, someone rewrites a paragraph until the tone is direct and useful. "
        "Human text is uneven: spaces cluster, punctuation interrupts, and words return later. "
    )
    out = [corpus[i % len(corpus)] for i in range(n)]
    for i in range(37, n, 41):
        out[i] = rng.choice("abcdefghijklmnopqrstuvwxyz ")
    return "".join(out)


def _batch_vs_seq(
    query: str,
    targets: list[str],
    *,
    match: int = MATCH,
    mismatch: int = MISMATCH,
    gap: int = GAP,
    width: int | None = 16,
) -> None:
    kwargs: dict = {
        "match_score": match,
        "mismatch_score": mismatch,
        "gap_score": gap,
    }
    if width is not None:
        kwargs["width"] = width

    batch = list(sa.smith_waterman_scores(query, targets, **kwargs))
    seq = [sa.smith_waterman_score(query, t, **kwargs) for t in targets]
    assert batch == seq, f"batch={batch} seq={seq}"


@pytest.mark.parametrize(
    "n_targets,target_len,seed",
    [
        (1, 1024, 1),
        (2, 1024, 2),
        (3, 1024, 3),  # odd leftover
        (7, 1024, 4),
        (8, 1024, 5),
    ],
)
def test_dual_equal_length_batch_matches_seq(
    n_targets: int, target_len: int, seed: int
) -> None:
    rng = random.Random(seed)
    query = _english(rng, 1024)
    targets = [_english(rng, target_len) for _ in range(n_targets)]
    _batch_vs_seq(query, targets)


def test_dual_unequal_lengths_batch_matches_seq() -> None:
    rng = random.Random(42)
    query = _english(rng, 1024)
    lengths = [500, 800, 1024, 1200, 900, 700]
    targets = [_english(rng, n) for n in lengths]
    _batch_vs_seq(query, targets)


def test_dual_empty_and_mixed_targets() -> None:
    rng = random.Random(7)
    query = _english(rng, 1024)
    targets = [
        _english(rng, 1024),
        "",
        _english(rng, 512),
        "",
        _english(rng, 1024),
    ]
    _batch_vs_seq(query, targets)


def test_dual_seed_sweep_equal_pairs() -> None:
    for seed in range(20, 30):
        rng = random.Random(seed)
        query = _english(rng, 1024)
        targets = [_english(rng, 1024) for _ in range(4)]
        _batch_vs_seq(query, targets)


def test_batch_short_query_still_matches_seq() -> None:
    """Non-exact-fill shape: dual falls through, batch must still match."""
    rng = random.Random(99)
    query = _english(rng, 100)
    targets = [_english(rng, n) for n in (100, 80, 120, 100)]
    _batch_vs_seq(query, targets)


def test_batch_alt_scores_match_seq() -> None:
    rng = random.Random(11)
    query = _english(rng, 1024)
    targets = [_english(rng, 1024), _english(rng, 900), _english(rng, 1024)]
    _batch_vs_seq(query, targets, match=8, mismatch=-9, gap=-1)
