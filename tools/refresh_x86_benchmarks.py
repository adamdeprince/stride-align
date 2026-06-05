#!/usr/bin/env python3
"""Refresh the x86 Levenshtein / Damerau-Levenshtein benchmarks.

Reproduces the three CSVs under ``benchmarks/intel-*.csv`` that were
originally generated ad-hoc:

* ``benchmarks/intel-levenshtein-<DATE>.csv`` -- 14 KJV queries vs
  first 1000 KJV verses, total wall-clock seconds per library.
* ``benchmarks/intel-levenshtein-v2-<DATE>.csv`` -- synthetic random
  strings across ``long`` (1 query x 200 targets), ``1v1`` (1 vs 1),
  and ``cutoff`` (1 vs 5000 with cutoff filtering) workloads, in
  microseconds per batch call.
* ``benchmarks/intel-damerau-levenshtein-<DATE>.csv`` -- Damerau-
  Levenshtein (OSA) on similar synthetic workloads.

Usage::

    $ pip install rapidfuzz editdistance
    $ python tools/refresh_x86_benchmarks.py
    $ python tools/refresh_x86_benchmarks.py --bench kjv

Each row is the median of ``--repeat`` measurements (default 5),
each measurement being a tight loop of ``--iterations`` calls so the
wall-clock dominates Python overhead.

Random workloads use ``--seed`` (default 17) for reproducibility.
The KJV pass needs ``demo/kjv.txt`` in the repo root.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import random
import statistics
import string
import sys
import time
from pathlib import Path

import stride_align as sa

try:
    from rapidfuzz.distance import Levenshtein as rf_lev, OSA as rf_osa  # type: ignore[import]
except ImportError:
    rf_lev = None
    rf_osa = None
try:
    import editdistance  # type: ignore[import]
except ImportError:
    editdistance = None


REPO_ROOT = Path(__file__).resolve().parent.parent


# 14 canonical queries used in benchmarks/intel-levenshtein-2026-05-19.csv.
KJV_QUERIES = [
    b"god",
    b"love",
    b"mercy",
    b"blessed",
    b"forgive",
    b"peace",
    b"And God said",
    b"Blessed are the meek",
    b"The earth was without form",
    b"in the beginning was the Word",
    b"a man after mine own heart",
    b"thou shalt not",
    b"the lord is my shepherd",
    b"jesus wept",
]


# ----- timing helpers -----

def _measure(fn, iterations: int, repeat: int) -> float:
    """Return median of ``repeat`` runs, each timing ``iterations`` calls.
    Result is total wall-clock seconds for one ``iterations`` block."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(iterations):
            fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _per_batch_us(fn, iterations: int, repeat: int) -> float:
    """Median wall-clock per single call of ``fn``, in microseconds."""
    block = _measure(fn, iterations, repeat)
    return (block / iterations) * 1e6


# ----- KJV Levenshtein pass -----

def _kjv_lev(args: argparse.Namespace) -> Path:
    kjv = (REPO_ROOT / "demo" / "kjv.txt").read_bytes().splitlines()
    targets = kjv[: args.kjv_n_targets]

    headers = ["query", "query_length", "n_targets", "mode"]
    libs: list[tuple[str, object]] = [("stride_align", "sw")]  # placeholder
    if rf_lev is not None: libs.append(("rapidfuzz", rf_lev.distance))
    if editdistance is not None: libs.append(("editdistance", editdistance.eval))
    headers.extend(f"{name}_s" for name, _ in libs)

    out_path = REPO_ROOT / "benchmarks" / f"intel-levenshtein-{args.date}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        for query in KJV_QUERIES:
            row = [repr(query), len(query), len(targets), "lev"]
            for name, _ in libs:
                if name == "stride_align":
                    fn = lambda q=query, ts=targets: sa.levenshtein_scores(q, ts)
                elif name == "rapidfuzz":
                    fn = lambda q=query, ts=targets: [rf_lev.distance(q, t) for t in ts]
                elif name == "editdistance":
                    fn = lambda q=query, ts=targets: [editdistance.eval(q, t) for t in ts]
                row.append(f"{_measure(fn, args.iterations, args.repeat) / args.iterations:.6f}")
            writer.writerow(row)
            print(",".join(map(str, row)), flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return out_path


# ----- synthetic random helpers -----

def _rand_bytes(rng: random.Random, length: int) -> bytes:
    alphabet = string.ascii_lowercase
    return ("".join(rng.choice(alphabet) for _ in range(length))).encode()


def _v2_workloads() -> list[tuple[str, int, int, int | None]]:
    """(workload, q_len, n_targets, cutoff_or_None) reproducing
    benchmarks/intel-levenshtein-v2-2026-05-19.csv."""
    rows: list[tuple[str, int, int, int | None]] = []
    for q_len in (40, 65, 100, 128, 180, 200):
        rows.append(("long", q_len, 200, None))
    for q_len in (10, 30, 60, 100, 200):
        rows.append(("1v1", q_len, 1, None))
    for q_len, cutoffs in ((10, (None, 2, 5, 10)),
                           (30, (None, 7, 15, 30)),
                           (50, (None, 12, 25, 50))):
        for cutoff in cutoffs:
            rows.append(("cutoff", q_len, 5000, cutoff))
    return rows


def _damerau_workloads() -> list[tuple[str, int, int]]:
    """(workload, q_len, n_targets) reproducing
    benchmarks/intel-damerau-levenshtein-2026-05-19.csv."""
    rows: list[tuple[str, int, int]] = []
    for q_len in (5, 10, 20, 30):
        rows.append(("1-vs-1000 short", q_len, 1000))
    for q_len in (10, 30, 64):
        rows.append(("1-vs-200 medium", q_len, 200))
    for q_len in (10, 30, 60):
        rows.append(("1-vs-1", q_len, 1))
    return rows


# ----- v2 Levenshtein pass -----

def _v2_lev(args: argparse.Namespace) -> Path:
    rng = random.Random(args.seed)
    headers = ["workload", "q_len", "n_targets", "cutoff",
               "stride_align_us"]
    if rf_lev is not None:
        headers.append("rapidfuzz_us")

    out_path = REPO_ROOT / "benchmarks" / f"intel-levenshtein-v2-{args.date}.csv"
    with out_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        for workload, q_len, n_targets, cutoff in _v2_workloads():
            query = _rand_bytes(rng, q_len)
            targets = [_rand_bytes(rng, q_len) for _ in range(n_targets)]
            single = n_targets == 1
            t0 = targets[0] if single else None

            # stride_align — use the singular API when n_targets==1 so we
            # measure the singular hot path, not the batch dispatch around
            # a one-element list (which makes the comparator libs look
            # ~5x faster than they really are).
            if single:
                if cutoff is None:
                    sa_fn = lambda q=query, t=t0: sa.levenshtein_score(q, t)
                else:
                    sa_fn = lambda q=query, t=t0, c=cutoff: sa.levenshtein_score(q, t, score_cutoff=c)
            else:
                if cutoff is None:
                    sa_fn = lambda q=query, ts=targets: sa.levenshtein_scores(q, ts)
                else:
                    sa_fn = lambda q=query, ts=targets, c=cutoff: sa.levenshtein_scores(q, ts, score_cutoff=c)
            sa_us = _per_batch_us(sa_fn, args.iterations, args.repeat)

            row: list[object] = [workload, q_len, n_targets,
                                 "" if cutoff is None else cutoff,
                                 f"{sa_us:.2f}"]
            if rf_lev is not None:
                if single:
                    if cutoff is None:
                        rf_fn = lambda q=query, t=t0: rf_lev.distance(q, t)
                    else:
                        rf_fn = lambda q=query, t=t0, c=cutoff: rf_lev.distance(q, t, score_cutoff=c)
                else:
                    if cutoff is None:
                        rf_fn = lambda q=query, ts=targets: [rf_lev.distance(q, t) for t in ts]
                    else:
                        rf_fn = lambda q=query, ts=targets, c=cutoff: [
                            rf_lev.distance(q, t, score_cutoff=c) for t in ts]
                row.append(f"{_per_batch_us(rf_fn, args.iterations, args.repeat):.2f}")
            writer.writerow(row)
            print(",".join(map(str, row)), flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return out_path


# ----- Damerau (OSA) pass -----

def _damerau(args: argparse.Namespace) -> Path:
    if rf_osa is None:
        print("rapidfuzz not installed -- skipping damerau", file=sys.stderr)
        return Path()

    rng = random.Random(args.seed + 1)  # offset to keep streams distinct
    headers = ["workload", "q_len", "n_targets",
               "stride_align_us", "rapidfuzz_us", "ratio"]
    out_path = REPO_ROOT / "benchmarks" / f"intel-damerau-levenshtein-{args.date}.csv"
    with out_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        for workload, q_len, n_targets in _damerau_workloads():
            query = _rand_bytes(rng, q_len)
            targets = [_rand_bytes(rng, q_len) for _ in range(n_targets)]
            single = n_targets == 1
            t0 = targets[0] if single else None

            # Singular API on 1-vs-1; batch API on multi-target.
            if single:
                sa_fn = lambda q=query, t=t0: sa.damerau_levenshtein_score(q, t)
                rf_fn = lambda q=query, t=t0: rf_osa.distance(q, t)
            else:
                sa_fn = lambda q=query, ts=targets: sa.damerau_levenshtein_scores(q, ts)
                rf_fn = lambda q=query, ts=targets: [rf_osa.distance(q, t) for t in ts]
            sa_us = _per_batch_us(sa_fn, args.iterations, args.repeat)
            rf_us = _per_batch_us(rf_fn, args.iterations, args.repeat)
            ratio = f"{(rf_us / sa_us):.2f}x" if sa_us > 0 else "n/a"

            row = [workload, q_len, n_targets,
                   f"{sa_us:.3f}" if sa_us < 10 else f"{sa_us:.1f}",
                   f"{rf_us:.3f}" if rf_us < 10 else f"{rf_us:.1f}",
                   ratio]
            writer.writerow(row)
            print(",".join(map(str, row)), flush=True)
    print(f"wrote {out_path}", file=sys.stderr)
    return out_path


# ----- entry point -----

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", choices=["all", "kjv", "v2", "damerau"],
                   default="all", help="which CSV to refresh")
    p.add_argument("--date", default=_dt.date.today().isoformat(),
                   help="YYYY-MM-DD tag for output filenames")
    p.add_argument("--seed", type=int, default=17,
                   help="random seed for synthetic workloads")
    p.add_argument("--iterations", type=int, default=20,
                   help="calls per measurement block")
    p.add_argument("--repeat", type=int, default=5,
                   help="number of measurement blocks (median taken)")
    p.add_argument("--kjv-n-targets", type=int, default=1000,
                   help="first N lines of demo/kjv.txt to use as targets")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.bench in ("all", "kjv"):
        _kjv_lev(args)
    if args.bench in ("all", "v2"):
        _v2_lev(args)
    if args.bench in ("all", "damerau"):
        _damerau(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
