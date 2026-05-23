"""Command line file comparison for stride-align."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import available_backends

InputData = str | bytes
InputSymbol = str | int

_SIGNED_LIMITS = {
    8: (1 << 7) - 1,
    16: (1 << 15) - 1,
    32: (1 << 31) - 1,
    64: (1 << 63) - 1,
}

_UNSIGNED_LIMITS = {
    8: (1 << 8) - 1,
    16: (1 << 16) - 1,
    32: (1 << 32) - 1,
    64: (1 << 64) - 1,
}

_BACKEND_MODULES = {
    "generic": "stride_align._generic",
    "swar": "stride_align._swar",
    "x86_sse41": "stride_align._sse41",
    "x86_avx2": "stride_align._avx2",
    "x86_avx512bwvl": "stride_align._avx512bwvl",
    "x86_avx10_256": "stride_align._avx10_256",
    "x86_avx10_512": "stride_align._avx10_512",
    "linux_aarch64_neon": "stride_align._neon",
    "linux_aarch64_sve": "stride_align._sve",
    "linux_aarch64_sve2": "stride_align._sve2",
    "macos_arm64_neon": "stride_align._macos_arm64_neon",
    "linux_loongarch64_lsx": "stride_align._lsx",
    "linux_loongarch64_lasx": "stride_align._lasx",
    "linux_powerpc64_vsx": "stride_align._vsx",
    "solaris_sparc_vis3": "stride_align._vis3",
    "linux_riscv64_rvv": "stride_align._rvv",
}

_SHORT_BACKEND_ALIASES = {
    "swar64": "swar",
    "sse41": "x86_sse41",
    "avx2": "x86_avx2",
    "avx512bwvl": "x86_avx512bwvl",
    "avx10_256": "x86_avx10_256",
    "avx10_512": "x86_avx10_512",
    "asimd": "linux_aarch64_neon",
    "neon": "linux_aarch64_neon",
    "sve": "linux_aarch64_sve",
    "sve2": "linux_aarch64_sve2",
    "lsx": "linux_loongarch64_lsx",
    "lasx": "linux_loongarch64_lasx",
    "vsx": "linux_powerpc64_vsx",
    "vis3": "solaris_sparc_vis3",
    "rvv": "linux_riscv64_rvv",
}


class CommandError(Exception):
    """User-facing command line error."""


@dataclass(frozen=True, slots=True)
class DecodedInput:
    path: str
    data: InputData
    length: int
    max_codepoint: int
    symbols: frozenset[InputSymbol]


@dataclass(frozen=True, slots=True)
class ResolvedBackend:
    name: str
    module_name: str
    module: Any


@dataclass(frozen=True, slots=True)
class CompareResult:
    normalized_score: float
    elapsed_seconds: float | None
    iterations: int
    backend_name: str
    backend_module: str
    kernel_backend: str
    effective_width: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stride-align-file-compare",
        description="Compare two files with stride-align and print a normalized score in [0, 1].",
    )
    parser.add_argument("query_file")
    parser.add_argument("target_file")
    parser.add_argument(
        "--farrar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the Smith-Waterman Farrar score kernel. Use --no-farrar to disable.",
    )
    parser.add_argument("--encoding", choices=("ascii", "utf8", "binary"), default="utf8")
    parser.add_argument(
        "--mode",
        choices=("smith-waterman", "needleman-wunsch", "sw", "nw"),
        default="smith-waterman",
    )
    parser.add_argument("--match-score", "--match", type=int, default=2)
    parser.add_argument("--mismatch-score", "--mismatch", type=int, default=-1)
    parser.add_argument(
        "--gap-open-score",
        "--gap-open",
        "--gap-score",
        "--gap",
        type=int,
        default=-1,
    )
    parser.add_argument("--gap-extend-score", "--gap-extend", type=int, default=None)
    parser.add_argument("--score-width", type=int, choices=(8, 16, 32, 64), default=None)
    parser.add_argument("--token-width", type=int, choices=(8, 16, 32, 64), default=None)
    parser.add_argument(
        "--simd",
        default=None,
        help="Backend name such as generic or x86_avx2.",
    )
    parser.add_argument("--benchmark", action="store_true", help="Print alignment time to stdout.")
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=1,
        help="Run the selected alignment this many times when benchmarking.",
    )
    return parser


def _mode_name(mode: str) -> str:
    if mode == "sw":
        return "smith-waterman"
    if mode == "nw":
        return "needleman-wunsch"
    return mode


def _read_input_once(path_text: str, encoding: str) -> DecodedInput:
    path = Path(path_text)
    try:
        with path.open("rb") as stream:
            raw_bytes = stream.read()
    except OSError as exc:
        raise CommandError(f"failed to read {path}: {exc}") from exc

    if encoding == "binary":
        return DecodedInput(
            path=path_text,
            data=raw_bytes,
            length=len(raw_bytes),
            max_codepoint=max(raw_bytes, default=0),
            symbols=frozenset(raw_bytes),
        )

    codec = "utf-8" if encoding == "utf8" else "ascii"
    try:
        text = raw_bytes.decode(codec)
    except UnicodeDecodeError as exc:
        raise CommandError(
            f"{path} is not valid {encoding} at byte offset {exc.start}: {exc.reason}"
        ) from exc

    maximum = 0
    symbols: set[str] = set()
    for character in text:
        maximum = max(maximum, ord(character))
        symbols.add(character)

    return DecodedInput(
        path=path_text,
        data=text,
        length=len(text),
        max_codepoint=maximum,
        symbols=frozenset(symbols),
    )


def _required_token_width(maximum_codepoint: int) -> int:
    for width, limit in _UNSIGNED_LIMITS.items():
        if maximum_codepoint <= limit:
            return width
    raise CommandError("input contains a token that does not fit into a 64-bit token width")


def _validate_token_width(
    query: DecodedInput,
    target: DecodedInput,
    token_width: int | None,
    farrar: bool,
) -> int:
    if farrar:
        if token_width is not None and token_width != 8:
            raise CommandError("Farrar uses an 8-bit token channel; --token-width must be 8")
        symbol_count = len(query.symbols | target.symbols)
        if symbol_count > 256:
            raise CommandError(
                f"Farrar 8-bit token channel supports at most 256 symbols; input has {symbol_count}"
            )
        return 8

    maximum_codepoint = max(query.max_codepoint, target.max_codepoint)
    required_width = _required_token_width(maximum_codepoint)
    if token_width is None:
        return required_width
    if token_width < required_width:
        raise CommandError(
            f"input contains U+{maximum_codepoint:04X}, which does not fit into "
            f"a {token_width}-bit token width"
        )
    return token_width


def _required_score_width(score_bound: int) -> int:
    for width, limit in _SIGNED_LIMITS.items():
        if score_bound <= limit:
            return width
    raise CommandError("maximum possible score magnitude exceeds signed 64-bit capacity")


def _validate_score_width(
    query_length: int,
    target_length: int,
    score_width: int | None,
    match_score: int,
    mismatch_score: int,
    gap_open_score: int,
    gap_extend_score: int,
) -> int:
    step_bound = max(
        abs(match_score),
        abs(mismatch_score),
        abs(gap_open_score),
        abs(gap_extend_score),
    )
    score_bound = (query_length + target_length) * step_bound
    required_width = _required_score_width(score_bound)
    if score_width is None:
        return required_width
    if score_bound > _SIGNED_LIMITS[score_width]:
        raise CommandError(
            f"maximum possible score magnitude {score_bound} exceeds signed {score_width}-bit "
            f"capacity {_SIGNED_LIMITS[score_width]}"
        )
    return score_width


def _available_simd_names() -> set[str]:
    names = {"generic"}
    for record in available_backends():
        if record.available:
            names.add(record.name)
    return names


def _resolve_backend(simd_name: str | None) -> ResolvedBackend:
    if simd_name is None:
        return ResolvedBackend(
            name="auto",
            module_name="stride_align",
            module=importlib.import_module("stride_align"),
        )

    canonical_name = _SHORT_BACKEND_ALIASES.get(simd_name, simd_name)
    available_names = _available_simd_names()
    if canonical_name not in available_names:
        choices = ", ".join(sorted(available_names))
        raise CommandError(f"--simd {simd_name!r} is not available on this CPU; choices: {choices}")

    module_name = _BACKEND_MODULES.get(canonical_name)
    if module_name is None:
        raise CommandError(f"no Python module is registered for --simd {simd_name!r}")

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        if canonical_name == "generic":
            module_name = "stride_align._pybackend"
            module = importlib.import_module(module_name)
        else:
            raise CommandError(f"failed to import backend {canonical_name}: {exc}") from exc

    return ResolvedBackend(name=canonical_name, module_name=module_name, module=module)


def _select_alignment_function(
    backend: Any,
    mode: str,
    farrar: bool,
    score_width: int,
    token_width: int,
):
    if farrar:
        return backend.smith_waterman_farrar_score, score_width
    if mode == "smith-waterman":
        return backend.smith_waterman_score, max(score_width, token_width)
    return backend.needleman_wunsch_score, max(score_width, token_width)


def _raw_alignment_score(
    function: Any,
    query: InputData,
    target: InputData,
    width: int,
    match_score: int,
    mismatch_score: int,
    gap_open_score: int,
    gap_extend_score: int,
) -> int:
    return int(
        function(
            query,
            target,
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_score=gap_open_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
            width=width,
        )
    )


def _normalize_score(
    raw_score: int,
    query_length: int,
    target_length: int,
    mode: str,
    match_score: int,
) -> float:
    if match_score <= 0:
        raise CommandError("--match-score must be positive to normalize into [0, 1]")
    if query_length == 0 and target_length == 0:
        return 1.0

    if mode == "smith-waterman":
        denominator = min(query_length, target_length) * match_score
    else:
        denominator = max(query_length, target_length) * match_score

    if denominator <= 0:
        return 0.0
    return min(1.0, max(0.0, raw_score / denominator))


def _run(args: argparse.Namespace) -> tuple[float, float | None]:
    mode = _mode_name(args.mode)
    if args.farrar and mode != "smith-waterman":
        raise CommandError("Farrar optimization is only available for Smith-Waterman")
    gap_extend_score = (
        args.gap_open_score if args.gap_extend_score is None else args.gap_extend_score
    )

    query = _read_input_once(args.query_file, args.encoding)
    target = _read_input_once(args.target_file, args.encoding)
    token_width = _validate_token_width(query, target, args.token_width, args.farrar)
    score_width = _validate_score_width(
        query.length,
        target.length,
        args.score_width,
        args.match_score,
        args.mismatch_score,
        args.gap_open_score,
        gap_extend_score,
    )

    if args.benchmark_iterations < 1:
        raise CommandError("--benchmark-iterations must be at least 1")

    backend = _resolve_backend(args.simd)
    function, effective_width = _select_alignment_function(
        backend.module,
        mode,
        args.farrar,
        score_width,
        token_width,
    )

    start_time = time.perf_counter()
    raw_score = 0
    iterations = args.benchmark_iterations if args.benchmark else 1
    for _ in range(iterations):
        raw_score = _raw_alignment_score(
            function,
            query.data,
            target.data,
            effective_width,
            args.match_score,
            args.mismatch_score,
            args.gap_open_score,
            gap_extend_score,
        )
    elapsed = time.perf_counter() - start_time
    normalized = _normalize_score(raw_score, query.length, target.length, mode, args.match_score)
    return CompareResult(
        normalized_score=normalized,
        elapsed_seconds=elapsed if args.benchmark else None,
        iterations=iterations,
        backend_name=backend.name,
        backend_module=backend.module_name,
        kernel_backend=backend.name,
        effective_width=effective_width,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    try:
        result = _run(args)
    except CommandError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{result.normalized_score:.17g}")
    if result.elapsed_seconds is not None:
        print(f"backend={result.backend_name}")
        print(f"backend_module={result.backend_module}")
        print(f"kernel_backend={result.kernel_backend}")
        print(f"effective_width={result.effective_width}")
        print(f"iterations={result.iterations}")
        print(f"time_seconds={result.elapsed_seconds:.9f}")
        print(f"time_seconds_per_iteration={result.elapsed_seconds / result.iterations:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
