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


def test_distinct_symbols_capped_at_256_for_int32() -> None:
    # int32/uint32/int64/uint64/float32/float64 still go through the
    # Phase A tokenise-to-uint8 path (the wide-uint16 kernel covers
    # only 2-byte dtypes for now), so the combined-distinct-symbols
    # cap is 256 for those dtypes.
    q = np.arange(300, dtype=np.int32)
    t = np.arange(300, dtype=np.int32)
    with pytest.raises(ValueError, match="256 distinct symbols"):
        sa.smith_waterman_score(q, t, match_score=2, mismatch_score=-1, gap_score=-1)


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
