"""Dynamic Time Warping — phase C.1a scalar reference.

Correctness oracles vs hand-computed DP, edge cases for Sakoe-Chiba
banding + per-dtype distance defaults, error contract for empty /
mixed-dtype / non-ndarray inputs. The SIMD batch kernel (phase
C.1b) cross-checks against these results once it lands.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import stride_align as sa


# ---- correctness vs hand-computed DP -------------------------------------

def _dtw_reference(q: np.ndarray, t: np.ndarray, *, distance: str = "l1",
                   window: int | None = None) -> float:
    """Plain-Python rolling-row DTW for cross-checking."""
    m, n = len(q), len(t)
    radius = max(m, n) if window is None else window
    inf = math.inf
    prev = [inf] * (n + 1)
    curr = [inf] * (n + 1)
    prev[0] = 0.0
    for i in range(1, m + 1):
        curr[0] = inf
        j_lo = max(1, i - radius)
        j_hi = min(n, i + radius)
        for j in range(j_lo, j_hi + 1):
            d = abs(float(q[i-1]) - float(t[j-1]))
            if distance == "l2_squared":
                d = d * d
            m_prev = min(prev[j-1], prev[j], curr[j-1])
            curr[j] = inf if m_prev == inf else d + m_prev
        prev, curr = curr, prev
        curr = [inf] * (n + 1)
        curr[0] = inf
    return prev[n]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtw_identity_zero_float(dtype):
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    assert sa.dtw(q, q) == 0.0


def test_dtw_identity_zero_int16():
    q = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    assert sa.dtw(q, q) == 0.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_dtw_matches_reference_unconstrained(dtype):
    rng = np.random.default_rng(17)
    q = (rng.standard_normal(20) * 10).astype(dtype, copy=False)
    t = (rng.standard_normal(30) * 10).astype(dtype, copy=False)
    actual = sa.dtw(q, t)
    expected = _dtw_reference(q, t, distance="l1" if dtype is np.int16 else "l2_squared")
    assert actual == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtw_l2_squared_default_for_float(dtype):
    q = np.array([0.0, 1.0, 0.0], dtype=dtype)
    t = np.array([0.0, 0.0, 1.0], dtype=dtype)
    # Hand DP, L2-squared: optimal warping is (0,0), (1,0)+ extension, (2,2):
    # Easier to just compare to the explicit kwarg.
    assert sa.dtw(q, t) == sa.dtw(q, t, distance="l2_squared")


def test_dtw_l1_default_for_int16():
    q = np.array([0, 1, 0], dtype=np.int16)
    t = np.array([0, 0, 1], dtype=np.int16)
    assert sa.dtw(q, t) == sa.dtw(q, t, distance="l1")


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_dtw_l1_simple_shift(dtype):
    # Two identical sequences but with one shifted by 1; DTW with L1
    # warps freely, distance is 0.
    q = np.array([0, 1, 2, 3, 4], dtype=dtype)
    t = np.array([0, 0, 1, 2, 3, 4], dtype=dtype)
    assert sa.dtw(q, t, distance="l1") == 0.0


# ---- Sakoe-Chiba band ----------------------------------------------------

def test_dtw_band_none_matches_unconstrained():
    rng = np.random.default_rng(1)
    q = (rng.standard_normal(15) * 5).astype(np.float64)
    t = (rng.standard_normal(15) * 5).astype(np.float64)
    assert sa.dtw(q, t, window=None) == sa.dtw(q, t)


def test_dtw_band_int_radius():
    rng = np.random.default_rng(2)
    q = (rng.standard_normal(20) * 5).astype(np.float64)
    t = (rng.standard_normal(20) * 5).astype(np.float64)
    banded = sa.dtw(q, t, window=2)
    expected = _dtw_reference(q, t, distance="l2_squared", window=2)
    assert banded == pytest.approx(expected, rel=1e-9)
    # Banded distance is >= unconstrained.
    assert banded >= sa.dtw(q, t)


def test_dtw_band_float_fraction():
    rng = np.random.default_rng(3)
    q = (rng.standard_normal(20) * 5).astype(np.float64)
    t = (rng.standard_normal(20) * 5).astype(np.float64)
    # 0.1 of max(20, 20) = 2 (ceil).
    assert sa.dtw(q, t, window=0.1) == sa.dtw(q, t, window=2)
    # 1.0 covers everything → matches unconstrained.
    assert sa.dtw(q, t, window=1.0) == sa.dtw(q, t)


def test_dtw_band_zero_diagonal_only():
    # window=0 forces the diagonal-only path; only works when
    # |q| == |t| and reduces to sum of per-cell distances along the
    # diagonal.
    q = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    t = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    # L2-squared diagonal: 0.25 + 0.25 + 0.25 = 0.75
    assert sa.dtw(q, t, window=0) == pytest.approx(0.75, rel=1e-12)


# ---- batch ---------------------------------------------------------------

def test_dtw_distances_matches_singular():
    rng = np.random.default_rng(11)
    query = (rng.standard_normal(12) * 3).astype(np.float32)
    targets = [(rng.standard_normal(10 + i) * 3).astype(np.float32) for i in range(5)]
    batch = sa.dtw_distances(query, targets)
    singular = [sa.dtw(query, t) for t in targets]
    assert batch.dtype == np.float64
    assert batch.tolist() == pytest.approx(singular, rel=1e-9)


def test_dtw_distances_batch_int16():
    rng = np.random.default_rng(22)
    query = (rng.standard_normal(10) * 100).astype(np.int16)
    targets = [(rng.standard_normal(12) * 100).astype(np.int16) for _ in range(3)]
    out = sa.dtw_distances(query, targets, window=3)
    expected = [_dtw_reference(query, t, distance="l1", window=3) for t in targets]
    assert out.tolist() == pytest.approx(expected, rel=1e-9)


def test_dtw_distances_empty_targets():
    query = np.array([1.0, 2.0], dtype=np.float64)
    out = sa.dtw_distances(query, [])
    assert out.shape == (0,)
    assert out.dtype == np.float64


# ---- error contract ------------------------------------------------------

def test_dtw_rejects_unsupported_dtype():
    q = np.array([1, 2, 3], dtype=np.int32)
    t = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(TypeError, match="float32, float64, or int16"):
        sa.dtw(q, t)


def test_dtw_rejects_mixed_dtype():
    q = np.array([1.0, 2.0], dtype=np.float32)
    t = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(TypeError, match="share dtype"):
        sa.dtw(q, t)


def test_dtw_rejects_non_ndarray():
    with pytest.raises(TypeError):
        sa.dtw([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(TypeError):
        sa.dtw(b"abc", b"abd")


def test_dtw_rejects_empty():
    empty = np.array([], dtype=np.float32)
    nonempty = np.array([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="non-empty"):
        sa.dtw(empty, nonempty)
    with pytest.raises(ValueError, match="non-empty"):
        sa.dtw(nonempty, empty)


def test_dtw_rejects_bad_window():
    q = np.array([1.0, 2.0], dtype=np.float64)
    t = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="non-negative"):
        sa.dtw(q, t, window=-1)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        sa.dtw(q, t, window=0.0)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        sa.dtw(q, t, window=1.5)


def test_dtw_rejects_bad_distance():
    q = np.array([1.0, 2.0], dtype=np.float64)
    t = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="l1.*l2_squared"):
        sa.dtw(q, t, distance="foo")


def test_dtw_distances_rejects_str_targets():
    q = np.array([1.0], dtype=np.float64)
    with pytest.raises(TypeError, match="iterable"):
        sa.dtw_distances(q, "not-a-list")


def test_dtw_distances_rejects_mixed_dtype_targets():
    q = np.array([1.0], dtype=np.float32)
    targets = [np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float64)]
    with pytest.raises(TypeError, match="dtype"):
        sa.dtw_distances(q, targets)
