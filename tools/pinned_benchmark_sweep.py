#!/usr/bin/env python3
"""Run the pinned Python benchmark in separate score/CIGAR/path-info sweeps."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(
    python: Path,
    output: Path,
    cpu: int | None,
    variants: list[str],
    backends: list[str],
    iterations: int,
    warmups: int,
    timing_split: bool,
    shapes: list[str],
) -> None:
    command = [
        str(python),
        "-m",
        "stride_align.benchmark",
        "--backends",
        *backends,
        "--variants",
        *variants,
        "--passes",
        "english",
        "chinese",
        "--shapes",
        *shapes,
        "--scoring-cases",
        "linear",
        "affine",
        "--widths",
        "16",
        "32",
        "--iterations",
        str(iterations),
        "--warmups",
        str(warmups),
        "--format",
        "csv",
    ]
    if timing_split:
        command.append("--timing-split")
    if cpu is not None:
        command = ["taskset", "-c", str(cpu), *command]

    with output.open("w", encoding="utf-8") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.PIPE, text=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/stride-align-pinned"))
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    common_backends = ["generic", "x86_avx2", "x86_avx512bwvl", "parasail"]
    trace_backends = ["generic", "x86_avx2", "x86_avx512bwvl", "parasail"]

    sweeps = [
        (
            "score-only.csv",
            ["sw-farrar-score", "sw-score", "nw-score"],
            common_backends,
            False,
            ["1:1", "1:many"],
        ),
        (
            "cigar.csv",
            ["sw-cigar", "nw-cigar"],
            trace_backends,
            True,
            ["1:1"],
        ),
        (
            "path-info.csv",
            ["sw-path-info", "nw-path-info"],
            trace_backends,
            True,
            ["1:1"],
        ),
    ]
    for filename, variants, backends, timing_split, shapes in sweeps:
        output = args.output_dir / filename
        print(f"writing {output}", file=sys.stderr)
        _run(
            args.python,
            output,
            args.cpu,
            variants,
            backends,
            args.iterations,
            args.warmups,
            timing_split,
            shapes,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
