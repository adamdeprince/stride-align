#!/usr/bin/env python3
"""Run the native x86 microbench as a small regression matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Result:
    backend: str
    shape: str
    benchmark_pass: str
    width: int
    ns_per_target: float
    cells_per_s: float
    score: int

    @property
    def key(self) -> str:
        return f"{self.backend}:{self.shape}:{self.benchmark_pass}:width{self.width}"


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_line(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in line.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key] = value
    return fields


def _run_one(
    binary: Path,
    cpu: int | None,
    backend: str,
    shape: str,
    benchmark_pass: str,
    width: int,
    iterations: int,
    warmups: int,
    targets_per_call: int,
) -> Result:
    command = [
        str(binary),
        "--backend",
        backend,
        "--shape",
        shape,
        "--pass",
        benchmark_pass,
        "--width",
        str(width),
        "--iterations",
        str(iterations),
        "--warmups",
        str(warmups),
        "--many-count",
        str(targets_per_call),
    ]
    if cpu is not None:
        command = ["taskset", "-c", str(cpu), *command]

    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    fields = _parse_line(completed.stdout.strip())
    try:
        return Result(
            backend=fields["backend"],
            shape=fields["shape"],
            benchmark_pass=fields["pass"],
            width=int(fields["width"]),
            ns_per_target=float(fields["ns_per_target"]),
            cells_per_s=float(fields["cells_per_s"]),
            score=int(fields["score"]),
        )
    except KeyError as exc:
        raise RuntimeError(f"microbench output missing {exc.args[0]!r}: {completed.stdout}") from exc


def _load_baseline(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    return {item["key"]: float(item["ns_per_target"]) for item in data["results"]}


def _write_json(path: Path | None, results: list[Result]) -> None:
    if path is None:
        return
    payload = {
        "metric": "ns_per_target",
        "lower_is_better": True,
        "results": [
            {**asdict(result), "key": result.key}
            for result in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _print_table(results: list[Result], baseline: dict[str, float]) -> bool:
    print("backend,shape,pass,width,ns_per_target,cells_per_s,vs_baseline")
    passed = True
    for result in results:
        old = baseline.get(result.key)
        ratio = "" if old is None else f"{old / result.ns_per_target:.4f}"
        print(
            f"{result.backend},{result.shape},{result.benchmark_pass},{result.width},"
            f"{result.ns_per_target:.3f},{result.cells_per_s:.6g},{ratio}"
        )
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("build/perf/stride_align_x86_microbench"),
        help="Path to stride_align_x86_microbench.",
    )
    parser.add_argument("--cpu", type=int, default=None, help="Pin each run with taskset -c CPU.")
    parser.add_argument("--backends", default="avx2,avx512bwvl")
    parser.add_argument("--shapes", default="1:1,1:many")
    parser.add_argument("--passes", default="english,chinese")
    parser.add_argument("--widths", default="16,32")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--targets-per-call", type=int, default=8)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument("--write-json", type=Path, default=None)
    parser.add_argument(
        "--fail-below",
        type=float,
        default=None,
        help="Fail if baseline/current speed ratio is below this value.",
    )
    args = parser.parse_args(argv)

    if not args.binary.exists():
        raise SystemExit(f"microbench binary not found: {args.binary}")

    baseline = _load_baseline(args.baseline_json)
    results: list[Result] = []
    for backend in _split_csv(args.backends):
        for shape in _split_csv(args.shapes):
            for benchmark_pass in _split_csv(args.passes):
                for width_text in _split_csv(args.widths):
                    results.append(
                        _run_one(
                            args.binary,
                            args.cpu,
                            backend,
                            shape,
                            benchmark_pass,
                            int(width_text),
                            args.iterations,
                            args.warmups,
                            args.targets_per_call,
                        )
                    )

    _print_table(results, baseline)
    _write_json(args.write_json, results)

    if args.fail_below is not None and baseline:
        failures = [
            result.key
            for result in results
            if result.key in baseline and baseline[result.key] / result.ns_per_target < args.fail_below
        ]
        if failures:
            print("regressions: " + ", ".join(failures), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
