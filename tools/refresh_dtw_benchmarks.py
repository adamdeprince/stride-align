#!/usr/bin/env python3
"""DTW 1-vs-N benchmarks: stride-align vs dtaidistance / tslearn / dtw-python.

Writes ``benchmarks/<host>-dtw-<DATE>.csv`` with median wall-clock
microseconds per batch (one query vs N targets) for float32 and int16,
with and without a Sakoe-Chiba window, and with an optional score_cutoff
(LB_Keogh path in stride-align).

Usage::

    uv run python tools/refresh_dtw_benchmarks.py
    uv run python tools/refresh_dtw_benchmarks.py --lengths 64,128 --n-targets 200

Optional deps (skipped if missing)::

    pip install dtaidistance tslearn dtw-python
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import stride_align as sa

REPO = Path(__file__).resolve().parent.parent


def _try_import(name: str):
    try:
        return __import__(name)
    except ImportError:
        return None


dtaidistance = _try_import("dtaidistance")
tslearn = _try_import("tslearn")
dtw_python = _try_import("dtw")  # dtw-python package


def _median_us(fn, iterations: int, repeat: int) -> float:
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(iterations):
            fn()
        samples.append((time.perf_counter() - t0) * 1e6 / iterations)
    return float(statistics.median(samples))


def _make_series(rng: np.random.Generator, length: int, dtype):
    if dtype == np.int16:
        return rng.integers(-1000, 1000, size=length, dtype=np.int16)
    return rng.standard_normal(length).astype(dtype, copy=False)


def bench_stride(
    query: np.ndarray,
    targets: list[np.ndarray],
    *,
    window,
    distance,
    score_cutoff,
    iterations: int,
    repeat: int,
) -> float:
    def run():
        return sa.dtw_distances(
            query,
            targets,
            window=window,
            distance=distance,
            score_cutoff=score_cutoff,
        )

    return _median_us(run, iterations, repeat)


def bench_dtaidistance(
    query: np.ndarray,
    targets: list[np.ndarray],
    *,
    window,
    iterations: int,
    repeat: int,
) -> float | None:
    if dtaidistance is None:
        return None
    from dtaidistance import dtw as ddtw

    q = query.astype(np.double, copy=False)
    ts = [t.astype(np.double, copy=False) for t in targets]
    # dtaidistance window is absolute radius (or None).
    w = None if window is None else int(window)

    def run():
        return [ddtw.distance(q, t, window=w) for t in ts]

    return _median_us(run, iterations, repeat)


def bench_tslearn(
    query: np.ndarray,
    targets: list[np.ndarray],
    *,
    window,
    iterations: int,
    repeat: int,
) -> float | None:
    if tslearn is None:
        return None
    from tslearn.metrics import dtw as tdtw

    q = query.astype(np.double, copy=False)
    ts = [t.astype(np.double, copy=False) for t in targets]
    # tslearn sakoe_chiba_radius is absolute.
    r = None if window is None else int(window)

    def run():
        return [
            tdtw(q, t, global_constraint="sakoe_chiba", sakoe_chiba_radius=r)
            if r is not None
            else tdtw(q, t)
            for t in ts
        ]

    return _median_us(run, iterations, repeat)


def bench_dtwpython(
    query: np.ndarray,
    targets: list[np.ndarray],
    *,
    iterations: int,
    repeat: int,
) -> float | None:
    if dtw_python is None:
        return None
    # dtw-python: dtw.dtw(x, y).distance
    q = query.astype(np.double, copy=False)
    ts = [t.astype(np.double, copy=False) for t in targets]

    def run():
        return [dtw_python.dtw(q, t).distance for t in ts]

    return _median_us(run, iterations, repeat)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lengths", default="32,64,128,256")
    ap.add_argument("--n-targets", type=int, default=128)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    host = platform.node().split(".")[0] or "host"
    date = _dt.date.today().isoformat()
    out = args.out or (REPO / "benchmarks" / f"{host}-dtw-{date}.csv")

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    configs = [
        # dtype, distance, window, score_cutoff label
        (np.float32, "l2_squared", None, None, "f32-full"),
        (np.float32, "l2_squared", 8, None, "f32-w8"),
        (np.float32, "l2_squared", 8, 50.0, "f32-w8-cutoff50"),
        (np.int16, "l1", None, None, "i16-full"),
        (np.int16, "l1", 8, None, "i16-w8"),
        (np.int16, "l1", 8, 500.0, "i16-w8-cutoff500"),
    ]

    print(
        f"libs: stride-align ok; "
        f"dtaidistance={'yes' if dtaidistance else 'no'}; "
        f"tslearn={'yes' if tslearn else 'no'}; "
        f"dtw-python={'yes' if dtw_python else 'no'}"
    )

    for length in lengths:
        for dtype, distance, window, cutoff, tag in configs:
            q = _make_series(rng, length, dtype)
            # Mix of equal and near-equal lengths for batch realism.
            targets = []
            for i in range(args.n_targets):
                tl = length if (i % 3) else max(8, length + int(rng.integers(-4, 5)))
                targets.append(_make_series(rng, tl, dtype))

            sa_us = bench_stride(
                q,
                targets,
                window=window,
                distance=distance,
                score_cutoff=cutoff,
                iterations=args.iterations,
                repeat=args.repeat,
            )
            row = {
                "host": host,
                "date": date,
                "length": length,
                "n_targets": args.n_targets,
                "dtype": np.dtype(dtype).name,
                "distance": distance,
                "window": "" if window is None else window,
                "score_cutoff": "" if cutoff is None else cutoff,
                "config": tag,
                "library": "stride_align",
                "us_per_batch": f"{sa_us:.3f}",
            }
            rows.append(row)
            print(f"  L={length} {tag} stride_align {sa_us:.1f} us")

            # Competitors: float64 path only (their APIs).
            if dtype == np.float32 and cutoff is None:
                q64 = q.astype(np.float64)
                t64 = [t.astype(np.float64) for t in targets]
                for name, fn in (
                    (
                        "dtaidistance",
                        lambda: bench_dtaidistance(
                            q64,
                            t64,
                            window=window,
                            iterations=args.iterations,
                            repeat=args.repeat,
                        ),
                    ),
                    (
                        "tslearn",
                        lambda: bench_tslearn(
                            q64,
                            t64,
                            window=window,
                            iterations=args.iterations,
                            repeat=args.repeat,
                        ),
                    ),
                    (
                        "dtw_python",
                        lambda: bench_dtwpython(
                            q64,
                            t64,
                            iterations=args.iterations,
                            repeat=args.repeat,
                        )
                        if window is None
                        else None,
                    ),
                ):
                    try:
                        us = fn()
                    except Exception as exc:  # noqa: BLE001
                        print(f"  L={length} {tag} {name} ERROR {exc}")
                        continue
                    if us is None:
                        continue
                    rows.append(
                        {
                            **{k: row[k] for k in row if k not in ("library", "us_per_batch")},
                            "library": name,
                            "us_per_batch": f"{us:.3f}",
                        }
                    )
                    print(f"  L={length} {tag} {name} {us:.1f} us")

    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "host",
        "date",
        "length",
        "n_targets",
        "dtype",
        "distance",
        "window",
        "score_cutoff",
        "config",
        "library",
        "us_per_batch",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
