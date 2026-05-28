"""Tests for ndarray input support on score-only paths."""

from __future__ import annotations

import numpy as np
import pytest

import stride_align as sa


_SIMD_DTYPES = [
    np.int8, np.uint8,
    np.int16, np.uint16,
    np.int32, np.uint32,
    np.int64, np.uint64,
    np.float16, np.float32, np.float64,
]


@pytest.mark.parametrize("dtype", _SIMD_DTYPES)
def test_smith_waterman_score_matches_list(dtype) -> None:
    q_ints = [1, 2, 3, 4, 5]
    t_ints = [1, 2, 4, 4, 5]
    expected = sa.smith_waterman_score(
        q_ints, t_ints, match_score=2, mismatch_score=-1, gap_score=-1,
    )
    q = np.asarray(q_ints, dtype=dtype)
    t = np.asarray(t_ints, dtype=dtype)
    assert sa.smith_waterman_score(
        q, t, match_score=2, mismatch_score=-1, gap_score=-1,
    ) == expected


@pytest.mark.parametrize("dtype", _SIMD_DTYPES)
def test_needleman_wunsch_score_matches_list(dtype) -> None:
    q_ints = [1, 2, 3]
    t_ints = [1, 4, 3]
    expected = sa.needleman_wunsch_score(
        q_ints, t_ints, match_score=2, mismatch_score=-1, gap_score=-1,
    )
    q = np.asarray(q_ints, dtype=dtype)
    t = np.asarray(t_ints, dtype=dtype)
    assert sa.needleman_wunsch_score(
        q, t, match_score=2, mismatch_score=-1, gap_score=-1,
    ) == expected


def test_object_dtype_rejected() -> None:
    arr = np.array([1, 2, 3], dtype=object)
    with pytest.raises(TypeError, match="dtype is not supported"):
        sa.smith_waterman_score(arr, arr, match_score=2, mismatch_score=-1, gap_score=-1)


def test_non_contiguous_rejected() -> None:
    big = np.arange(20, dtype=np.int32)
    sliced = big[::2]  # non-contiguous
    with pytest.raises(TypeError, match="dtype is not supported"):
        sa.smith_waterman_score(sliced, sliced, match_score=2, mismatch_score=-1, gap_score=-1)


def test_mixed_ndarray_str_rejected() -> None:
    arr = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(TypeError, match="can only align against another ndarray"):
        sa.smith_waterman_score(arr, "abc", match_score=2, mismatch_score=-1, gap_score=-1)
    with pytest.raises(TypeError, match="can only align against another ndarray"):
        sa.smith_waterman_score("abc", arr, match_score=2, mismatch_score=-1, gap_score=-1)


def test_dtype_mismatch_rejected() -> None:
    q = np.array([1, 2, 3], dtype=np.int8)
    t = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(TypeError, match="dtype"):
        sa.smith_waterman_score(q, t, match_score=2, mismatch_score=-1, gap_score=-1)


def test_path_rejects_ndarray() -> None:
    arr = np.array([1, 2, 3], dtype=np.int8)
    with pytest.raises(TypeError, match="score-only path"):
        sa.smith_waterman_path(arr, arr, match_score=2, mismatch_score=-1, gap_score=-1)
    with pytest.raises(TypeError, match="score-only path"):
        sa.needleman_wunsch_path(arr, arr, match_score=2, mismatch_score=-1, gap_score=-1)


def test_wide_uint32_lifts_distinct_symbol_cap() -> None:
    # int32 / uint32 / float32 now route through the wide Farrar kernel
    # (32-bit cells). The striped profile is keyed by the *target*
    # alphabet, so we keep the target tiny and only the query wide —
    # that proves the >256-distinct path runs without allocating a
    # |target|×|query| profile that OOMs memory-constrained hosts.
    for dtype in [np.int32, np.uint32]:
        q = np.arange(70_000, dtype=dtype)
        t = q[:2].copy()
        score = sa.smith_waterman_score(
            q, t, match_score=2, mismatch_score=-1, gap_score=-1,
        )
        # Best local alignment: target's two values match the start of
        # the query contiguously → 2 matches × 2 = 4.
        assert score == 4, f"{dtype}: got {score}"
        # Mirror direction: tiny query vs huge target (also small profile).
        rev = sa.smith_waterman_score(
            t, q, match_score=2, mismatch_score=-1, gap_score=-1,
        )
        assert rev == 4, f"{dtype} rev: got {rev}"


def test_wide_uint64_lifts_distinct_symbol_cap() -> None:
    # int64 / uint64 / float64 route through the wide 64-bit Farrar
    # kernel. Even moderate-size identity alignments work; we keep the
    # array small because the 64-bit DP is slow per cell.
    for dtype in [np.int64, np.uint64]:
        q = np.arange(1000, dtype=dtype)
        t = np.arange(1000, dtype=dtype)
        score = sa.smith_waterman_score(
            q, t, match_score=2, mismatch_score=-1, gap_score=-1,
        )
        assert score == 2000, f"{dtype}: got {score}"


def test_wide_uint16_path_lifts_256_cap() -> None:
    # int16 / uint16 / float16 now route through the wide Farrar kernel
    # (16-bit cells, hash-keyed token→row map), so >256 distinct values
    # work end-to-end instead of raising ValueError.
    for dtype in [np.uint16, np.int16]:
        q = np.arange(500, dtype=dtype)
        t = np.arange(500, dtype=dtype)
        # Identity self-alignment: 500 matches × 2 = 1000.
        score = sa.smith_waterman_score(
            q, t, match_score=2, mismatch_score=-1, gap_score=-1,
        )
        assert score == 1000, f"{dtype}: got {score}"


def test_wide_uint16_matches_list_within_256() -> None:
    # On inputs that fit in the narrow tokeniser's 256-symbol cap, the
    # wide uint16 path produces the same score as the list-based
    # reference (which uses the narrow uint8 path).
    import random
    random.seed(11)
    for _ in range(50):
        q_ints = [random.randint(0, 250) for _ in range(random.randint(1, 60))]
        t_ints = [random.randint(0, 250) for _ in range(random.randint(1, 60))]
        list_score = sa.smith_waterman_score(
            q_ints, t_ints, match_score=2, mismatch_score=-1, gap_score=-1,
        )
        q16 = np.asarray(q_ints, dtype=np.uint16)
        t16 = np.asarray(t_ints, dtype=np.uint16)
        wide_score = sa.smith_waterman_score(
            q16, t16, match_score=2, mismatch_score=-1, gap_score=-1,
        )
        assert list_score == wide_score


def test_under_256_symbols_works_int32() -> None:
    q = np.arange(100, dtype=np.int32)
    t = np.arange(100, dtype=np.int32)
    # Identity self-alignment: all 100 positions match → 100 * 2 = 200.
    assert sa.smith_waterman_score(
        q, t, match_score=2, mismatch_score=-1, gap_score=-1,
    ) == 200


def test_float_bit_pattern_equality() -> None:
    # Floats are compared by bit pattern (bitcast to uint of same width).
    q = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    t = np.array([1.5, 2.5, 7.0, 4.5], dtype=np.float32)
    # Diagonal alignment: 2 + 2 - 1 + 2 = 5.
    assert sa.smith_waterman_score(
        q, t, match_score=2, mismatch_score=-1, gap_score=-1,
    ) == 5


def test_batch_scores_with_ndarrays() -> None:
    query = np.array([1, 2, 3], dtype=np.int32)
    targets = [
        np.array([1, 2, 3], dtype=np.int32),
        np.array([3, 2, 1], dtype=np.int32),
        np.array([1, 1, 1], dtype=np.int32),
    ]
    scores = sa.smith_waterman_scores(
        query, targets, match_score=2, mismatch_score=-1, gap_score=-1,
    )
    assert scores.tolist() == [6, 2, 2]


def test_batch_mixed_target_kinds_rejected() -> None:
    query = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(TypeError, match="ndarray"):
        sa.smith_waterman_scores(
            query, [np.array([1], dtype=np.int32), "ab"],
            match_score=2, mismatch_score=-1, gap_score=-1,
        )


@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64])
def test_batch_scores_wide_ndarray_matches_per_pair(dtype) -> None:
    # Batch (1 query x N targets) with a >256-distinct alphabet must route
    # through the wide kernel and agree with the per-pair wide path.
    query = np.arange(300, dtype=dtype)
    targets = [
        np.arange(300, dtype=dtype),            # identity -> 600
        np.arange(300, dtype=dtype)[::-1].copy(),  # reversed
        np.arange(150, 450, dtype=dtype),       # half-overlap
    ]
    batch = sa.smith_waterman_scores(
        query, targets, match_score=2, mismatch_score=-1, gap_score=-1,
    ).tolist()
    per_pair = [
        sa.smith_waterman_score(query, t, match_score=2, mismatch_score=-1, gap_score=-1)
        for t in targets
    ]
    assert batch == per_pair
    assert batch[0] == 600  # identity self-alignment


@pytest.mark.parametrize("dtype", [np.uint16, np.int32])
def test_batch_needleman_wide_ndarray_matches_per_pair(dtype) -> None:
    query = np.arange(300, dtype=dtype)
    targets = [
        np.arange(300, dtype=dtype),
        np.arange(1, 301, dtype=dtype),
    ]
    batch = sa.needleman_wunsch_scores(
        query, targets, match_score=2, mismatch_score=-1, gap_score=-1,
    ).tolist()
    per_pair = [
        sa.needleman_wunsch_score(query, t, match_score=2, mismatch_score=-1, gap_score=-1)
        for t in targets
    ]
    assert batch == per_pair


def test_batch_wide_ndarray_dtype_mismatch_rejected() -> None:
    query = np.arange(300, dtype=np.uint16)
    with pytest.raises(TypeError, match="dtype"):
        sa.smith_waterman_scores(
            query, [np.arange(300, dtype=np.uint32)],
            match_score=2, mismatch_score=-1, gap_score=-1,
        )


def test_batch_wide_ndarray_mixed_kind_rejected() -> None:
    query = np.arange(300, dtype=np.uint16)
    with pytest.raises(TypeError, match="ndarray"):
        sa.smith_waterman_scores(
            query, [np.arange(300, dtype=np.uint16), "ab"],
            match_score=2, mismatch_score=-1, gap_score=-1,
        )


def test_affine_with_ndarrays() -> None:
    # Affine gaps share prepare_farrar_alignment, so ndarrays work here too.
    q = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    t = np.array([1, 2, 4, 4, 5], dtype=np.int16)
    affine = sa.smith_waterman_score(
        q, t, match_score=2, mismatch_score=-1,
        gap_open_score=-3, gap_extend_score=-1,
    )
    # No gaps in the optimal diagonal alignment; same as linear-gap result.
    linear = sa.smith_waterman_score(
        q, t, match_score=2, mismatch_score=-1, gap_score=-1,
    )
    assert affine == linear
