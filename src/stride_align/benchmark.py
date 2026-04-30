"""Consistent benchmark runner for stride-align backends."""

from __future__ import annotations

import argparse
import importlib
import platform
import random
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from . import available_backends


_BACKEND_MODULES = {
    "generic": "stride_align._generic",
    "x86_sse41": "stride_align._sse41",
    "x86_avx2": "stride_align._avx2",
    "x86_avx512bwvl": "stride_align._avx512bwvl",
    "x86_avx10_256": "stride_align._avx10_256",
    "x86_avx10_512": "stride_align._avx10_512",
    "linux_aarch64_asimd": "stride_align._asimd",
    "linux_aarch64_neon": "stride_align._neon",
    "linux_aarch64_sve": "stride_align._sve",
    "linux_aarch64_sve2": "stride_align._sve2",
    "macos_arm64_neon": "stride_align._macos_arm64_neon",
    "linux_loongarch64_lsx": "stride_align._lsx",
    "linux_loongarch64_lasx": "stride_align._lasx",
    "linux_powerpc64_vsx": "stride_align._vsx",
    "linux_riscv64_rvv": "stride_align._rvv",
}

_SHORT_BACKEND_ALIASES = {
    "sse41": "x86_sse41",
    "avx2": "x86_avx2",
    "avx512bwvl": "x86_avx512bwvl",
    "avx10_256": "x86_avx10_256",
    "avx10_512": "x86_avx10_512",
    "asimd": "linux_aarch64_asimd",
    "neon": "linux_aarch64_neon",
    "sve": "linux_aarch64_sve",
    "sve2": "linux_aarch64_sve2",
    "lsx": "linux_loongarch64_lsx",
    "lasx": "linux_loongarch64_lasx",
    "vsx": "linux_powerpc64_vsx",
    "rvv": "linux_riscv64_rvv",
}

_SIGNED_LIMITS = {
    8: (1 << 7) - 1,
    16: (1 << 15) - 1,
    32: (1 << 31) - 1,
    64: (1 << 63) - 1,
}


class BenchmarkError(Exception):
    """User-facing benchmark configuration error."""


@dataclass(frozen=True, slots=True)
class ResolvedBackend:
    name: str
    module_name: str
    module: Any


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    backend: str
    score_width: int
    score: int
    iterations: int
    warmups: int
    best_seconds: float
    median_seconds: float
    mean_seconds: float
    cells_per_second: float
    speedup_vs_generic: float | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stride-align-benchmark",
        description="Run deterministic stride-align Farrar score benchmarks.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Backends to run. Defaults to generic and CPU-available SIMD backends.",
    )
    parser.add_argument(
        "--widths",
        nargs="+",
        type=int,
        choices=(8, 16, 32, 64),
        default=[16, 32],
        help="Score channel widths to benchmark. Defaults to 16 32.",
    )
    parser.add_argument("--length", type=int, default=1024, help="Default query/target length.")
    parser.add_argument("--query-length", type=int, default=None)
    parser.add_argument("--target-length", type=int, default=None)
    parser.add_argument("--alphabet-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--match-score", "--match", type=int, default=2)
    parser.add_argument("--mismatch-score", "--mismatch", type=int, default=-1)
    parser.add_argument("--gap-score", "--gap", type=int, default=-1)
    parser.add_argument("--format", choices=("table", "csv"), default="table")
    return parser


def _canonical_backend(name: str) -> str:
    return _SHORT_BACKEND_ALIASES.get(name, name)


def _available_backend_names() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name not in seen:
            names.append(name)
            seen.add(name)

    add("generic")
    for record in available_backends():
        if record.available:
            add(record.name)

    return names


def _resolve_backends(names: Sequence[str] | None) -> list[ResolvedBackend]:
    selected = (
        _available_backend_names()
        if names is None
        else [_canonical_backend(name) for name in names]
    )
    available = set(_available_backend_names())
    resolved: list[ResolvedBackend] = []

    for name in selected:
        if name not in _BACKEND_MODULES:
            choices = ", ".join(sorted(_BACKEND_MODULES))
            raise BenchmarkError(f"unknown backend {name!r}; choices: {choices}")
        if name not in available:
            choices = ", ".join(sorted(available))
            raise BenchmarkError(
                f"backend {name!r} is not available on this machine; choices: {choices}"
            )

        module_name = _BACKEND_MODULES[name]
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            if name == "generic":
                module_name = "stride_align._pybackend"
                module = importlib.import_module(module_name)
            else:
                raise BenchmarkError(f"failed to import backend {name}: {exc}") from exc
        resolved.append(ResolvedBackend(name=name, module_name=module_name, module=module))

    return resolved


def _build_inputs(
    query_length: int,
    target_length: int,
    alphabet_size: int,
    seed: int,
) -> tuple[bytes, bytes]:
    if query_length < 0 or target_length < 0:
        raise BenchmarkError("sequence lengths must be non-negative")
    if not 1 <= alphabet_size <= 256:
        raise BenchmarkError("--alphabet-size must be in [1, 256]")

    rng = random.Random(seed)
    query = bytes(rng.randrange(alphabet_size) for _ in range(query_length))
    target = bytes(rng.randrange(alphabet_size) for _ in range(target_length))
    return query, target


def _validate_widths(
    widths: Sequence[int],
    query_length: int,
    target_length: int,
    match_score: int,
    mismatch_score: int,
    gap_score: int,
) -> None:
    score_bound = (query_length + target_length) * max(
        abs(match_score),
        abs(mismatch_score),
        abs(gap_score),
    )
    for width in widths:
        if score_bound > _SIGNED_LIMITS[width]:
            raise BenchmarkError(
                f"score bound {score_bound} exceeds signed {width}-bit capacity "
                f"{_SIGNED_LIMITS[width]}"
            )


def _time_backend(
    backend: ResolvedBackend,
    query: bytes,
    target: bytes,
    width: int,
    iterations: int,
    warmups: int,
    match_score: int,
    mismatch_score: int,
    gap_score: int,
) -> BenchmarkResult:
    function = backend.module.smith_waterman_farrar_score
    score = int(
        function(
            query,
            target,
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_score=gap_score,
            width=width,
        )
    )

    for _ in range(warmups):
        function(
            query,
            target,
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_score=gap_score,
            width=width,
        )

    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        observed_score = int(
            function(
                query,
                target,
                match_score=match_score,
                mismatch_score=mismatch_score,
                gap_score=gap_score,
                width=width,
            )
        )
        elapsed = time.perf_counter() - start
        if observed_score != score:
            raise BenchmarkError(
                f"{backend.name} produced unstable scores for width {width}: "
                f"{score} then {observed_score}"
            )
        timings.append(elapsed)

    median_seconds = statistics.median(timings)
    cells = len(query) * len(target)
    return BenchmarkResult(
        backend=backend.name,
        score_width=width,
        score=score,
        iterations=iterations,
        warmups=warmups,
        best_seconds=min(timings),
        median_seconds=median_seconds,
        mean_seconds=statistics.fmean(timings),
        cells_per_second=cells / median_seconds if median_seconds > 0 else float("inf"),
        speedup_vs_generic=None,
    )


def _with_speedups(results: Sequence[BenchmarkResult]) -> list[BenchmarkResult]:
    generic_medians = {
        result.score_width: result.median_seconds
        for result in results
        if result.backend == "generic"
    }
    output: list[BenchmarkResult] = []
    for result in results:
        generic_median = generic_medians.get(result.score_width)
        speedup = None if generic_median is None else generic_median / result.median_seconds
        output.append(
            BenchmarkResult(
                backend=result.backend,
                score_width=result.score_width,
                score=result.score,
                iterations=result.iterations,
                warmups=result.warmups,
                best_seconds=result.best_seconds,
                median_seconds=result.median_seconds,
                mean_seconds=result.mean_seconds,
                cells_per_second=result.cells_per_second,
                speedup_vs_generic=speedup,
            )
        )
    return output


def _print_csv(results: Sequence[BenchmarkResult]) -> None:
    print(
        "backend,score_width,score,iterations,warmups,best_seconds,"
        "median_seconds,mean_seconds,cells_per_second,speedup_vs_generic"
    )
    for result in results:
        speedup = "" if result.speedup_vs_generic is None else f"{result.speedup_vs_generic:.6g}"
        print(
            f"{result.backend},{result.score_width},{result.score},{result.iterations},"
            f"{result.warmups},{result.best_seconds:.9g},{result.median_seconds:.9g},"
            f"{result.mean_seconds:.9g},{result.cells_per_second:.9g},{speedup}"
        )


def _print_table(
    results: Sequence[BenchmarkResult],
    query_length: int,
    target_length: int,
    alphabet_size: int,
    seed: int,
) -> None:
    print("# stride-align-benchmark")
    print(f"# python={platform.python_version()} platform={platform.platform()}")
    print(
        f"# query_length={query_length} target_length={target_length} "
        f"alphabet_size={alphabet_size} seed={seed}"
    )
    headers = (
        "backend",
        "width",
        "score",
        "median_s",
        "best_s",
        "cells/s",
        "x_generic",
    )
    rows = []
    for result in results:
        speedup = "" if result.speedup_vs_generic is None else f"{result.speedup_vs_generic:.3f}"
        rows.append(
            (
                result.backend,
                str(result.score_width),
                str(result.score),
                f"{result.median_seconds:.6g}",
                f"{result.best_seconds:.6g}",
                f"{result.cells_per_second:.6g}",
                speedup,
            )
        )

    widths = [max(len(row[index]) for row in (headers, *rows)) for index in range(len(headers))]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _run(args: argparse.Namespace) -> list[BenchmarkResult]:
    if args.iterations < 1:
        raise BenchmarkError("--iterations must be at least 1")
    if args.warmups < 0:
        raise BenchmarkError("--warmups must be non-negative")

    query_length = args.length if args.query_length is None else args.query_length
    target_length = args.length if args.target_length is None else args.target_length
    _validate_widths(
        args.widths,
        query_length,
        target_length,
        args.match_score,
        args.mismatch_score,
        args.gap_score,
    )
    query, target = _build_inputs(query_length, target_length, args.alphabet_size, args.seed)
    backends = _resolve_backends(args.backends)

    results: list[BenchmarkResult] = []
    for width in args.widths:
        expected_score: int | None = None
        for backend in backends:
            result = _time_backend(
                backend,
                query,
                target,
                width,
                args.iterations,
                args.warmups,
                args.match_score,
                args.mismatch_score,
                args.gap_score,
            )
            if expected_score is None:
                expected_score = result.score
            elif result.score != expected_score:
                raise BenchmarkError(
                    f"{backend.name} width {width} returned {result.score}, "
                    f"expected {expected_score}"
                )
            results.append(result)

    return _with_speedups(results)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    try:
        results = _run(args)
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    query_length = args.length if args.query_length is None else args.query_length
    target_length = args.length if args.target_length is None else args.target_length
    if args.format == "csv":
        _print_csv(results)
    else:
        _print_table(results, query_length, target_length, args.alphabet_size, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
