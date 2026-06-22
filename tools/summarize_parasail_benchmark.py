#!/usr/bin/env python3
"""Summarize a ``stride_align.benchmark`` CSV as the markdown tables used
in BENCHMARK.md's parasail comparison sections.

Each row is paired with its ``parasail`` counterpart on
``(pass, case, shape, variant, score_width)`` and the speedup is
``cells_per_second / parasail_cells_per_second``. The script then emits,
per non-parasail backend:

* Overall vs parasail (rows / wins / geomean / median / worst / best),
* the score-only and path/CIGAR row breakdowns,
* a per-variant breakdown,
* the five worst rows vs parasail.

Usage::

    python tools/summarize_parasail_benchmark.py benchmark.csv \
        --backends x86_avx10_512 x86_avx2 generic
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict

SCORE_ONLY = ("sw-farrar-score", "sw-score", "nw-score")
PATH_CIGAR = ("sw-path-info", "nw-path-info", "sw-cigar", "nw-cigar")
KEY = ("pass", "case", "shape", "variant", "score_width")


def _load(path):
    with open(path, newline="") as fp:
        return list(csv.DictReader(fp))


def _ratios(rows):
    """Return list of (key_dict, backend, ratio) for every non-parasail row
    that has a matching parasail row."""
    par = {}
    for r in rows:
        if r["backend"] == "parasail":
            par[tuple(r[k] for k in KEY)] = float(r["cells_per_second"])
    out = []
    for r in rows:
        if r["backend"] == "parasail":
            continue
        k = tuple(r[k2] for k2 in KEY)
        if k not in par or par[k] <= 0:
            continue
        out.append((dict(zip(KEY, k)), r["backend"], float(r["cells_per_second"]) / par[k]))
    return out


def _stats(vals):
    vals = list(vals)
    return {
        "rows": len(vals),
        "wins": sum(1 for v in vals if v > 1.0),
        "geomean": statistics.geometric_mean(vals) if vals else 0.0,
        "median": statistics.median(vals) if vals else 0.0,
        "worst": min(vals) if vals else 0.0,
        "best": max(vals) if vals else 0.0,
    }


def _table(title, header, rows):
    print(f"\n{title}\n")
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---:" if i else "---" for i in range(len(header))) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--backends", nargs="+",
                   default=["x86_avx10_512", "x86_avx2", "generic"])
    args = p.parse_args(argv)

    ratios = _ratios(_load(args.csv))
    by_backend = defaultdict(list)
    for kd, backend, ratio in ratios:
        by_backend[backend].append((kd, ratio))

    _table("### Overall vs parasail",
           ["Backend", "Rows", "Wins", "Geomean", "Median", "Worst", "Best"],
           [[f"`{b}`", str(s["rows"]), str(s["wins"]),
             f"{s['geomean']:.3f}x", f"{s['median']:.3f}x",
             f"{s['worst']:.3f}x", f"{s['best']:.3f}x"]
            for b in args.backends if (s := _stats(r for _, r in by_backend[b]))])

    for label, variants in (("Score-only rows", SCORE_ONLY),
                            ("Path/CIGAR rows", PATH_CIGAR)):
        _table(f"{label} ({' + '.join(variants)}):",
               ["Backend", "Rows", "Wins", "Geomean", "Median"],
               [[f"`{b}`", str(s["rows"]), str(s["wins"]),
                 f"{s['geomean']:.3f}x", f"{s['median']:.3f}x"]
                for b in args.backends
                if (s := _stats(r for kd, r in by_backend[b]
                                if kd["variant"] in variants))["rows"]])

    print("\n### By variant")
    for b in args.backends:
        if b == "generic":
            continue
        per_var = defaultdict(list)
        for kd, ratio in by_backend[b]:
            per_var[kd["variant"]].append(ratio)
        ranked = sorted(per_var.items(),
                        key=lambda kv: statistics.geometric_mean(kv[1]),
                        reverse=True)
        _table(f"{b}:",
               ["Variant", "Rows", "Wins", "Geomean", "Median"],
               [[f"`{v}`", str(s["rows"]), str(s["wins"]),
                 f"{s['geomean']:.3f}x", f"{s['median']:.3f}x"]
                for v, vals in ranked if (s := _stats(vals))])

    print("\n### Worst rows vs parasail")
    for b in args.backends:
        if b == "generic":
            continue
        worst = sorted(by_backend[b], key=lambda kr: kr[1])[:5]
        _table(f"{b}:",
               ["Ratio", "Pass", "Case", "Shape", "Variant", "Width"],
               [[f"{ratio:.3f}x", kd["pass"], kd["case"], kd["shape"],
                 f"`{kd['variant']}`", kd["score_width"]]
                for kd, ratio in worst])

    return 0


if __name__ == "__main__":
    sys.exit(main())
