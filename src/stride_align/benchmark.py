"""Consistent benchmark runner for stride-align backends."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import platform
import random
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from . import available_backends

_PARASAIL_SAFE_ALPHABET = "".join(chr(codepoint) for codepoint in range(1, 256))

_ENGLISH_CORPUS = (
    "The quick brown fox watches the city wake under a low grey sky. "
    "People cross the station concourse with coffee, folded papers, and quiet plans. "
    "A street musician repeats a careful phrase while buses hiss at the curb. "
    "In the office, someone rewrites a paragraph until the tone is direct and useful. "
    "Human text is uneven: spaces cluster, punctuation interrupts, and words return later. "
)

_CHINESE_CORPUS = (
    "清晨的城市慢慢醒来，街边的早餐摊冒着热气，行人带着各自的计划穿过路口。"
    "办公室里有人反复修改一段文字，希望语气更准确，结构更清楚，也更容易阅读。"
    "人类语言并不均匀，常用字会不断出现，标点和短句让节奏发生变化。"
    "傍晚的地铁里很安静，窗外的灯光一站一站向后退去。"
)

_ENGLISH_REPLACEMENTS = " etaoinshrdlucmfwypvbgkqjxz,.;:-"
_CHINESE_REPLACEMENTS = "的一是在不了有人和国中大为上个情文清城市语言结构阅读计划"

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
    "parasail": "parasail",
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

_ALL_VARIANTS = (
    "sw-farrar-score",
    "sw-score",
    "nw-score",
    "sw-path-info",
    "nw-path-info",
    "sw-path",
    "nw-path",
)

_VARIANT_ALIASES = {
    "farrar": "sw-farrar-score",
    "farrar-score": "sw-farrar-score",
    "smith-waterman-score": "sw-score",
    "needleman-wunsch-score": "nw-score",
    "smith-waterman-path-info": "sw-path-info",
    "needleman-wunsch-path-info": "nw-path-info",
    "smith-waterman-path": "sw-path",
    "needleman-wunsch-path": "nw-path",
}

_ALL_PASSES = ("english", "chinese", "random-bytes")
_DEFAULT_PASSES = ("english", "chinese")
_PASS_ALIASES = {
    "text": "all-text",
    "texts": "all-text",
    "random": "random-bytes",
    "bytes": "random-bytes",
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
    pass_name: str
    backend: str
    variant: str
    score_width: int
    score: int
    iterations: int
    warmups: int
    best_seconds: float
    median_seconds: float
    mean_seconds: float
    cells_per_second: float
    speedup_vs_generic: float | None


@dataclass(frozen=True, slots=True)
class BenchmarkPass:
    name: str
    query: object
    target: object


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stride-align-benchmark",
        description="Run deterministic stride-align alignment benchmarks.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help=(
            "Backends to run. Defaults to all backends available on this machine. "
            "Use 'all' or 'available' to request the same discovered set explicitly. "
            "Adds parasail automatically when importable."
        ),
    )
    parser.add_argument(
        "--widths",
        nargs="+",
        type=int,
        choices=(8, 16, 32, 64),
        default=[16, 32],
        help="Score channel widths to benchmark. Defaults to 16 32.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["sw-farrar-score"],
        help=(
            "Workloads to benchmark. Choices: sw-farrar-score, sw-score, nw-score, "
            "sw-path-info, nw-path-info, sw-path, nw-path, or all. "
            "Defaults to sw-farrar-score."
        ),
    )
    parser.add_argument("--length", type=int, default=1024, help="Default query/target length.")
    parser.add_argument("--query-length", type=int, default=None)
    parser.add_argument("--target-length", type=int, default=None)
    parser.add_argument(
        "--passes",
        nargs="+",
        default=list(_DEFAULT_PASSES),
        help=(
            "Benchmark passes to run. Choices: english, chinese, random-bytes, "
            "all-text, all. Defaults to english chinese."
        ),
    )
    parser.add_argument(
        "--alphabet-size",
        type=int,
        default=96,
        help="Distinct byte values for the random-bytes pass. Defaults to 96.",
    )
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
    for record in available_backends():
        if record.available and record.name in _BACKEND_MODULES and record.name not in names:
            names.append(record.name)
    if importlib.util.find_spec("parasail") is not None and "parasail" not in names:
        names.append("parasail")

    return names


def _selected_backend_names(names: Sequence[str] | None) -> list[str]:
    if names is None:
        return _available_backend_names()

    selected: list[str] = []
    for name in names:
        canonical_name = _canonical_backend(name)
        if canonical_name in {"all", "available"}:
            for available_name in _available_backend_names():
                if available_name not in selected:
                    selected.append(available_name)
            continue
        if canonical_name not in selected:
            selected.append(canonical_name)
    return selected


def _selected_variants(variants: Sequence[str]) -> list[str]:
    selected: list[str] = []
    for variant in variants:
        canonical_variant = _VARIANT_ALIASES.get(variant, variant)
        if canonical_variant == "all":
            for available_variant in _ALL_VARIANTS:
                if available_variant not in selected:
                    selected.append(available_variant)
            continue
        if canonical_variant not in _ALL_VARIANTS:
            choices = ", ".join((*_ALL_VARIANTS, "all"))
            raise BenchmarkError(f"unknown benchmark variant {variant!r}; choices: {choices}")
        if canonical_variant not in selected:
            selected.append(canonical_variant)
    return selected


def _selected_passes(passes: Sequence[str]) -> list[str]:
    selected: list[str] = []
    for benchmark_pass in passes:
        canonical_pass = _PASS_ALIASES.get(benchmark_pass, benchmark_pass)
        if canonical_pass == "all-text":
            candidates = _DEFAULT_PASSES
        elif canonical_pass == "all":
            candidates = _ALL_PASSES
        elif canonical_pass in _ALL_PASSES:
            candidates = (canonical_pass,)
        else:
            choices = ", ".join((*_ALL_PASSES, "all-text", "all"))
            raise BenchmarkError(f"unknown benchmark pass {benchmark_pass!r}; choices: {choices}")

        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)
    return selected


def _resolve_backends(names: Sequence[str] | None) -> list[ResolvedBackend]:
    selected = _selected_backend_names(names)
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
            imported_module = importlib.import_module(module_name)
            module = (
                _ParasailBenchmarkBackend(imported_module)
                if name == "parasail"
                else imported_module
            )
        except ImportError as exc:
            if name == "generic":
                module_name = "stride_align._pybackend"
                module = importlib.import_module(module_name)
            else:
                raise BenchmarkError(f"failed to import backend {name}: {exc}") from exc
        resolved.append(ResolvedBackend(name=name, module_name=module_name, module=module))

    return resolved


class _ParasailBenchmarkBackend:
    def __init__(self, parasail_module: Any) -> None:
        self._parasail = parasail_module
        self._prepared_cache: dict[
            tuple[tuple[object, ...], tuple[object, ...], int, int],
            tuple[str, str, Any],
        ] = {}

    def smith_waterman_farrar_score(
        self,
        query: bytes,
        target: bytes,
        *,
        match_score: int,
        mismatch_score: int,
        gap_score: int,
        width: int,
    ) -> int:
        return self._run(
            "sw-farrar-score",
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            width,
        )

    def smith_waterman_score(
        self,
        query: bytes,
        target: bytes,
        *,
        match_score: int,
        mismatch_score: int,
        gap_score: int,
        width: int,
    ) -> int:
        return self._run("sw-score", query, target, match_score, mismatch_score, gap_score, width)

    def needleman_wunsch_score(
        self,
        query: bytes,
        target: bytes,
        *,
        match_score: int,
        mismatch_score: int,
        gap_score: int,
        width: int,
    ) -> int:
        return self._run("nw-score", query, target, match_score, mismatch_score, gap_score, width)

    def smith_waterman_path_info(
        self,
        query: bytes,
        target: bytes,
        *,
        match_score: int,
        mismatch_score: int,
        gap_score: int,
        width: int,
    ):
        return self._run(
            "sw-path-info",
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            width,
        )

    def needleman_wunsch_path_info(
        self,
        query: bytes,
        target: bytes,
        *,
        match_score: int,
        mismatch_score: int,
        gap_score: int,
        width: int,
    ):
        return self._run(
            "nw-path-info",
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            width,
        )

    def smith_waterman_path(
        self,
        query: bytes,
        target: bytes,
        *,
        match_score: int,
        mismatch_score: int,
        gap_score: int,
        width: int,
    ):
        return self._run("sw-path", query, target, match_score, mismatch_score, gap_score, width)

    def needleman_wunsch_path(
        self,
        query: bytes,
        target: bytes,
        *,
        match_score: int,
        mismatch_score: int,
        gap_score: int,
        width: int,
    ):
        return self._run("nw-path", query, target, match_score, mismatch_score, gap_score, width)

    def _prepare_smith_waterman_farrar_score(
        self,
        query: object,
        target: object,
        *,
        match_score: int,
        mismatch_score: int,
        gap_score: int,
        width: int,
    ):
        if gap_score >= 0:
            raise BenchmarkError("parasail benchmark adapter requires a non-positive gap score")
        query_text, target_text, matrix = self._prepare_inputs(
            query,
            target,
            match_score,
            mismatch_score,
        )
        profile_create = getattr(self._parasail, f"profile_create_{width}", None)
        profile_function = getattr(self._parasail, f"sw_striped_profile_{width}", None)
        if profile_create is None or profile_function is None:
            raise BenchmarkError(
                f"parasail does not expose a prepared striped SW function for width {width}"
            )
        return (
            profile_function,
            profile_create(query_text, matrix),
            target_text,
            -gap_score,
        )

    def _smith_waterman_farrar_score_prepared(self, prepared):
        profile_function, profile, target_text, gap_penalty = prepared
        return profile_function(profile, target_text, gap_penalty, gap_penalty)

    def _symbols(self, value: object) -> tuple[object, ...]:
        if isinstance(value, bytes):
            return tuple(value)
        if isinstance(value, str):
            return tuple(value)
        return tuple(value)  # type: ignore[arg-type]

    def _prepare_inputs(
        self,
        query: object,
        target: object,
        match_score: int,
        mismatch_score: int,
    ) -> tuple[str, str, Any]:
        query_symbols = self._symbols(query)
        target_symbols = self._symbols(target)
        key = (query_symbols, target_symbols, match_score, mismatch_score)
        cached = self._prepared_cache.get(key)
        if cached is not None:
            return cached

        symbols: list[object] = []
        translation: dict[object, str] = {}
        for symbol in (*query_symbols, *target_symbols):
            if symbol in translation:
                continue
            if len(symbols) >= len(_PARASAIL_SAFE_ALPHABET):
                raise BenchmarkError(
                    "parasail benchmark input has too many distinct symbols for the "
                    f"safe adapter alphabet: {len(symbols) + 1} > "
                    f"{len(_PARASAIL_SAFE_ALPHABET)}"
                )
            translation[symbol] = _PARASAIL_SAFE_ALPHABET[len(symbols)]
            symbols.append(symbol)

        if len(symbols) > len(_PARASAIL_SAFE_ALPHABET):
            raise BenchmarkError(
                "parasail benchmark input has too many distinct symbols for the "
                f"safe adapter alphabet: {len(symbols)} > {len(_PARASAIL_SAFE_ALPHABET)}"
            )

        query_text = "".join(translation[symbol] for symbol in query_symbols)
        target_text = "".join(translation[symbol] for symbol in target_symbols)
        alphabet = _PARASAIL_SAFE_ALPHABET[: len(symbols)]
        matrix = self._parasail.matrix_create(alphabet, match_score, mismatch_score)
        prepared = (query_text, target_text, matrix)
        self._prepared_cache[key] = prepared
        return prepared

    def _function_names(self, variant: str, width: int) -> tuple[str, ...]:
        if variant == "sw-farrar-score":
            return (f"sw_striped_{width}",)
        if variant == "sw-score":
            return (f"sw_scan_{width}", f"sw_striped_{width}")
        if variant == "nw-score":
            return (f"nw_scan_{width}", f"nw_striped_{width}")
        if variant in {"sw-path-info", "sw-path"}:
            return (f"sw_trace_striped_{width}", f"sw_trace_scan_{width}")
        if variant in {"nw-path-info", "nw-path"}:
            return (f"nw_trace_striped_{width}", f"nw_trace_scan_{width}")
        raise AssertionError(f"unhandled parasail benchmark variant {variant!r}")

    def _function(self, variant: str, width: int):
        for name in self._function_names(variant, width):
            function = getattr(self._parasail, name, None)
            if function is not None:
                return function
        choices = ", ".join(self._function_names(variant, width))
        raise BenchmarkError(
            f"parasail does not expose a function for {variant} width {width}: {choices}"
        )

    def _run(
        self,
        variant: str,
        query: bytes,
        target: bytes,
        match_score: int,
        mismatch_score: int,
        gap_score: int,
        width: int,
    ):
        if gap_score >= 0:
            raise BenchmarkError("parasail benchmark adapter requires a non-positive gap score")
        query_text, target_text, matrix = self._prepare_inputs(
            query,
            target,
            match_score,
            mismatch_score,
        )
        gap_penalty = -gap_score
        result = self._function(variant, width)(
            query_text,
            target_text,
            gap_penalty,
            gap_penalty,
            matrix,
        )
        if variant.endswith("path") or variant.endswith("path-info"):
            getattr(result, "cigar", None)
        return result


def _function_for_variant(module: Any, variant: str):
    match variant:
        case "sw-farrar-score":
            return module.smith_waterman_farrar_score
        case "sw-score":
            return module.smith_waterman_score
        case "nw-score":
            return module.needleman_wunsch_score
        case "sw-path-info":
            return module.smith_waterman_path_info
        case "nw-path-info":
            return module.needleman_wunsch_path_info
        case "sw-path":
            return module.smith_waterman_path
        case "nw-path":
            return module.needleman_wunsch_path
    raise AssertionError(f"unhandled benchmark variant {variant!r}")


def _result_score(result: Any) -> int:
    if isinstance(result, int):
        return result
    return int(result.score)


def _repeat_window(corpus: str, length: int, offset: int) -> str:
    if length == 0:
        return ""
    repeats = (length + offset) // len(corpus) + 2
    return (corpus * repeats)[offset : offset + length]


def _mutate_text(value: str, replacements: str, seed: int) -> str:
    if not value:
        return value

    rng = random.Random(seed)
    characters = list(value)
    stride = 37 + seed % 17
    for index in range(stride - 1, len(characters), stride):
        characters[index] = rng.choice(replacements)
    return "".join(characters)


def _build_text_pass(
    name: str,
    query_length: int,
    target_length: int,
    seed: int,
) -> BenchmarkPass:
    if name == "english":
        corpus = _ENGLISH_CORPUS
        replacements = _ENGLISH_REPLACEMENTS
    elif name == "chinese":
        corpus = _CHINESE_CORPUS
        replacements = _CHINESE_REPLACEMENTS
    else:
        raise AssertionError(f"unhandled text benchmark pass {name!r}")

    offset = (seed * 29) % len(corpus)
    query = _repeat_window(corpus, query_length, offset)
    target = _repeat_window(corpus, target_length, offset)
    target = _mutate_text(target, replacements, seed + len(name))
    return BenchmarkPass(name=name, query=query, target=target)


def _build_random_bytes_pass(
    query_length: int,
    target_length: int,
    alphabet_size: int,
    seed: int,
) -> BenchmarkPass:
    if query_length < 0 or target_length < 0:
        raise BenchmarkError("sequence lengths must be non-negative")
    if not 1 <= alphabet_size <= 256:
        raise BenchmarkError("--alphabet-size must be in [1, 256]")

    rng = random.Random(seed)
    query = bytes(rng.randrange(alphabet_size) for _ in range(query_length))
    target = bytes(rng.randrange(alphabet_size) for _ in range(target_length))
    return BenchmarkPass(name="random-bytes", query=query, target=target)


def _build_benchmark_passes(
    pass_names: Sequence[str],
    query_length: int,
    target_length: int,
    alphabet_size: int,
    seed: int,
) -> list[BenchmarkPass]:
    if query_length < 0 or target_length < 0:
        raise BenchmarkError("sequence lengths must be non-negative")

    benchmark_passes: list[BenchmarkPass] = []
    for index, pass_name in enumerate(pass_names):
        pass_seed = seed + index * 1009
        if pass_name in {"english", "chinese"}:
            benchmark_passes.append(
                _build_text_pass(pass_name, query_length, target_length, pass_seed)
            )
        elif pass_name == "random-bytes":
            benchmark_passes.append(
                _build_random_bytes_pass(
                    query_length,
                    target_length,
                    alphabet_size,
                    pass_seed,
                )
            )
        else:
            raise AssertionError(f"unhandled benchmark pass {pass_name!r}")
    return benchmark_passes


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
    pass_name: str,
    variant: str,
    query: object,
    target: object,
    width: int,
    iterations: int,
    warmups: int,
    match_score: int,
    mismatch_score: int,
    gap_score: int,
) -> BenchmarkResult:
    function = _function_for_variant(backend.module, variant)

    if (
        variant == "sw-farrar-score"
        and hasattr(backend.module, "_prepare_smith_waterman_farrar_score")
        and hasattr(backend.module, "_smith_waterman_farrar_score_prepared")
    ):
        prepared = backend.module._prepare_smith_waterman_farrar_score(
            query,
            target,
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_score=gap_score,
            width=width,
        )

        def run_once() -> Any:
            return backend.module._smith_waterman_farrar_score_prepared(prepared)

    else:

        def run_once() -> Any:
            return function(
                query,
                target,
                match_score=match_score,
                mismatch_score=mismatch_score,
                gap_score=gap_score,
                width=width,
            )

    score = _result_score(run_once())

    for _ in range(warmups):
        run_once()

    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        observed_score = _result_score(run_once())
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
        pass_name=pass_name,
        backend=backend.name,
        variant=variant,
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
        (result.pass_name, result.variant, result.score_width): result.median_seconds
        for result in results
        if result.backend == "generic"
    }
    output: list[BenchmarkResult] = []
    for result in results:
        generic_median = generic_medians.get(
            (result.pass_name, result.variant, result.score_width)
        )
        speedup = None if generic_median is None else generic_median / result.median_seconds
        output.append(
            BenchmarkResult(
                pass_name=result.pass_name,
                backend=result.backend,
                variant=result.variant,
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
        "pass,backend,variant,score_width,score,iterations,warmups,best_seconds,"
        "median_seconds,mean_seconds,cells_per_second,speedup_vs_generic"
    )
    for result in results:
        speedup = "" if result.speedup_vs_generic is None else f"{result.speedup_vs_generic:.6g}"
        print(
            f"{result.pass_name},{result.backend},{result.variant},"
            f"{result.score_width},{result.score},"
            f"{result.iterations},"
            f"{result.warmups},{result.best_seconds:.9g},{result.median_seconds:.9g},"
            f"{result.mean_seconds:.9g},{result.cells_per_second:.9g},{speedup}"
        )


def _print_table(
    results: Sequence[BenchmarkResult],
    query_length: int,
    target_length: int,
    pass_names: Sequence[str],
    seed: int,
) -> None:
    print("# stride-align-benchmark")
    print(f"# python={platform.python_version()} platform={platform.platform()}")
    print(
        f"# query_length={query_length} target_length={target_length} "
        f"passes={','.join(pass_names)} seed={seed}"
    )
    headers = (
        "pass",
        "backend",
        "variant",
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
                result.pass_name,
                result.backend,
                result.variant,
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
    pass_names = _selected_passes(args.passes)
    benchmark_passes = _build_benchmark_passes(
        pass_names,
        query_length,
        target_length,
        args.alphabet_size,
        args.seed,
    )
    backends = _resolve_backends(args.backends)
    variants = _selected_variants(args.variants)

    results: list[BenchmarkResult] = []
    for benchmark_pass in benchmark_passes:
        for variant in variants:
            for width in args.widths:
                expected_score: int | None = None
                for backend in backends:
                    result = _time_backend(
                        backend,
                        benchmark_pass.name,
                        variant,
                        benchmark_pass.query,
                        benchmark_pass.target,
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
                            f"{backend.name} {benchmark_pass.name} {variant} width {width} "
                            f"returned {result.score}, expected {expected_score}"
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
        _print_table(
            results,
            query_length,
            target_length,
            _selected_passes(args.passes),
            args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
