#!/usr/bin/env python3
"""Verify that BENCHMARK.md agrees with its current raw artifacts.

This is intentionally a documentation check, not a benchmark runner. It
recalculates the product-facing summary rows, verifies that the canonical
Intel CSV matches its immutable snapshot, checks the saved rapidfuzz-shim
reference summary, and rejects broken local Markdown links.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "BENCHMARK.md"
INTEL = REPO / "benchmark.csv"
INTEL_SNAPSHOT = REPO / "benchmarks/intel-avx512-parasail-2026-07-18.csv"
INTEL_LEV = REPO / "benchmarks/intel-levenshtein-2026-07-17.csv"
INTEL_LEV_V2 = REPO / "benchmarks/intel-levenshtein-v2-2026-07-17.csv"
INTEL_OSA = REPO / "benchmarks/intel-damerau-levenshtein-2026-07-17.csv"
LOONGSON = REPO / "benchmarks/loongson-native-2026-07-17.csv"

ALIGNMENT_KEY = ("pass", "case", "shape", "variant", "score_width")
SCORE_VARIANTS = {"sw-farrar-score", "sw-score", "nw-score"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def stats(values) -> dict[str, float | int]:
    values = list(values)
    if not values:
        raise ValueError("cannot summarize an empty benchmark slice")
    return {
        "rows": len(values),
        "wins": sum(value > 1.0 for value in values),
        "geomean": statistics.geometric_mean(values),
        "median": statistics.median(values),
        "worst": min(values),
        "best": max(values),
    }


def alignment_ratios(
    path: Path,
    backend: str,
    baseline: str,
    predicate=lambda row: True,
) -> list[float]:
    rows = read_csv(path)
    key = lambda row: tuple(row[field] for field in ALIGNMENT_KEY)
    baseline_times = {
        key(row): float(row["median_seconds"])
        for row in rows
        if row["backend"] == baseline
    }
    return [
        baseline_times[key(row)] / float(row["median_seconds"])
        for row in rows
        if row["backend"] == backend and predicate(row)
    ]


def visible_slice_row(label: str, summary: dict[str, float | int]) -> str:
    return (
        f"| {label} | {summary['rows']} | **{summary['wins']}** | "
        f"**{summary['geomean']:.3f}x** | {summary['median']:.3f}x | "
        f"{summary['worst']:.3f}x | {summary['best']:.3f}x |"
    )


def require(page: str, snippet: str, errors: list[str]) -> None:
    if snippet not in page:
        errors.append(f"missing or stale page content: {snippet}")


def check_current_summary(page: str, errors: list[str]) -> None:
    if INTEL.read_bytes() != INTEL_SNAPSHOT.read_bytes():
        errors.append("benchmark.csv differs from its 2026-07-18 snapshot")

    slices = [
        ("Overall", lambda row: True),
        ("English", lambda row: row["pass"] == "english"),
        ("Chinese", lambda row: row["pass"] == "chinese"),
        ("Path / CIGAR", lambda row: row["variant"] not in SCORE_VARIANTS),
    ]
    for label, predicate in slices:
        summary = stats(alignment_ratios(
            INTEL, "x86_avx512bwvl", "parasail", predicate))
        require(page, visible_slice_row(label, summary), errors)

    kjv = read_csv(INTEL_LEV)
    lev = stats(
        float(row["rapidfuzz_s"]) / float(row["stride_align_s"])
        for row in kjv
    )
    require(page, (
        "| Levenshtein, 1 query × 1,000 KJV targets | Intel AVX-512 | "
        f"rapidfuzz | {lev['rows']} | **{lev['geomean']:.3f}x** | "
        f"{lev['worst']:.3f}x | {lev['best']:.3f}x |"
    ), errors)

    osa_rows = read_csv(INTEL_OSA)
    osa = stats(
        float(row["rapidfuzz_us"]) / float(row["stride_align_us"])
        for row in osa_rows
        if row["workload"] == "1-vs-1000 short"
    )
    require(page, (
        "| OSA/Damerau, short targets | Intel AVX-512 | rapidfuzz OSA | "
        f"{osa['rows']} | **{osa['geomean']:.3f}x** | "
        f"{osa['worst']:.3f}x | {osa['best']:.3f}x |"
    ), errors)

    loongson = stats(alignment_ratios(
        LOONGSON,
        "linux_loongarch64_lasx",
        "generic",
        lambda row: row["variant"] in SCORE_VARIANTS,
    ))
    require(page, (
        "| Score-only alignment | Loongson LASX | stride-align `generic` | "
        f"{loongson['rows']} | **{loongson['geomean']:.3f}x** | "
        f"{loongson['worst']:.3f}x | {loongson['best']:.3f}x |"
    ), errors)

    # Load this artifact even though its longer-pattern rows stay in the
    # detailed section. That keeps the freshness ledger honest.
    if not read_csv(INTEL_LEV_V2):
        errors.append(f"empty artifact: {INTEL_LEV_V2.relative_to(REPO)}")


def check_shim_reference(page: str, errors: list[str]) -> None:
    files = {
        "Mac M4 Max": REPO / "benchmarks/shim-full-mac-2026-06-17.json",
        "avx10 (Intel AWS)": REPO / "benchmarks/shim-full-avx10-2026-06-17.json",
        "Loongson (2026-06-10)": REPO / "benchmarks/shim-full-loongson-2026-06-10.json",
    }
    for host, path in files.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_ratios = [entry["ratio"] for entry in data["timings"].values()]
        speedups = [1.0 / value for value in raw_ratios]
        summary = stats(speedups)
        wins = sum(value <= 1.05 for value in raw_ratios)
        ties = sum(1.05 < value <= 1.15 for value in raw_ratios)
        losses = sum(value > 1.15 for value in raw_ratios)
        require(page, (
            f"| {host} | `{data['backend']}` | {summary['rows']} | "
            f"**{summary['geomean']:.2f}x** | {summary['median']:.2f}x | "
            f"{summary['worst']:.2f}x | {summary['best']:.2f}x | "
            f"{wins}/{ties}/{losses} |"
        ), errors)


def check_local_links(page: str, errors: list[str]) -> None:
    for target in re.findall(r"\]\(([^)]+)\)", page):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        local = (PAGE.parent / unquote(parsed.path)).resolve()
        try:
            local.relative_to(REPO)
        except ValueError:
            errors.append(f"local link escapes the repository: {target}")
            continue
        if not local.exists():
            errors.append(f"broken local link: {target}")


def main() -> int:
    page = PAGE.read_text(encoding="utf-8")
    errors: list[str] = []
    check_current_summary(page, errors)
    check_shim_reference(page, errors)
    check_local_links(page, errors)
    if errors:
        print("benchmark page verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("benchmark page verified against current raw artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
