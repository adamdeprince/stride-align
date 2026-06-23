# Dynamic Time Warping

`dtw_distances` computes the DTW distance from one query
time-series to every target time-series in a single SIMD batch.
Multi-width: `int16`, `float32`, `float64`. Constrained or
unconstrained warping window.

## Function

```python
sa.dtw_distances(
    query,                          # 1D ndarray
    targets,                        # iterable of 1D ndarrays (any length)
    *,
    window: int | float | None = None,    # Sakoe-Chiba band; None = unconstrained
    distance: str | None = None,          # local distance metric
) -> np.ndarray                     # ndarray[float64], one row per target
```

Returned array has shape `(len(targets),)` regardless of
individual target length. The query and all targets share a
single ndarray dtype per call (int16 / float32 / float64); mixing
dtypes raises.

## `window=` parameter

The Sakoe-Chiba band — restricts the warping path to within `w`
cells of the diagonal. Common choices:

| `window` | Effect |
| --- | --- |
| `None` (default) | unconstrained — full O(m·n) DP per target |
| `int >= 0` | absolute band radius |
| `float in (0, 1)` | fractional of the longer sequence |

Constraining the window has two upsides: it reduces work
quadratically when the band is tight, and it prevents the
"pathological warp" failure mode where DTW collapses two
dissimilar series via aggressive bending.

## `distance=` parameter

Local distance function applied at each cell:

| Value | Local metric |
| --- | --- |
| `None` (default) | absolute difference (`|a − b|`) |
| `"squared"` | squared difference (`(a − b)²`) — common for unbounded continuous data |
| `"manhattan"` | alias for `None` |

Per-cell cost is summed along the optimal warping path. The
returned distance is the total path cost; there's no
length-normalisation (caller does that if needed).

## Examples

```python
import numpy as np
import stride_align as sa

# Float series, unconstrained warping
query = np.array([1.0, 2.0, 3.0, 2.5, 1.5], dtype=np.float64)
targets = [
    np.array([1.0, 2.5, 3.0, 2.0, 1.5], dtype=np.float64),
    np.array([0.5, 1.5, 3.5, 2.5], dtype=np.float64),
]
d = sa.dtw_distances(query, targets)

# Int16 series with a 5-cell Sakoe-Chiba band, squared metric
d2 = sa.dtw_distances(
    query.astype(np.int16),
    [t.astype(np.int16) for t in targets],
    window=5,
    distance="squared",
)
```

## When to use DTW vs string distance

DTW is for numeric time-series where the elements are continuous
samples and elastic time alignment matters. For symbolic
sequences (text), use [edit-distance.md](edit-distance.md) or
[alignment.md](alignment.md) instead — they're orders of
magnitude faster on those workloads.
