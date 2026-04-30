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
    "swar": "stride_align._swar",
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
    "swar64": "swar",
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

_SCORE_ONLY_VARIANTS = (
    "sw-farrar-score",
    "sw-score",
    "nw-score",
)
_PATH_INFO_VARIANTS = (
    "sw-path-info",
    "nw-path-info",
)
_FULL_PATH_VARIANTS = (
    "sw-path",
    "nw-path",
)
_ALL_VARIANTS = (*_SCORE_ONLY_VARIANTS, *_PATH_INFO_VARIANTS, *_FULL_PATH_VARIANTS)
_DEFAULT_VARIANTS = (*_SCORE_ONLY_VARIANTS, *_PATH_INFO_VARIANTS)

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
_VARIANT_GROUPS = {
    "all": _ALL_VARIANTS,
    "score": _SCORE_ONLY_VARIANTS,
    "scores": _SCORE_ONLY_VARIANTS,
    "score-only": _SCORE_ONLY_VARIANTS,
    "score-only-generators": _SCORE_ONLY_VARIANTS,
    "path": _PATH_INFO_VARIANTS,
    "paths": _PATH_INFO_VARIANTS,
    "path-info": _PATH_INFO_VARIANTS,
    "path-generators": _PATH_INFO_VARIANTS,
    "full-path": _FULL_PATH_VARIANTS,
    "full-paths": _FULL_PATH_VARIANTS,
}

_ALL_SCORING_CASES = ("linear", "affine")
_DEFAULT_SCORING_CASES = ("linear", "affine")
_SCORING_CASE_ALIASES = {
    "all": "all",
    "both": "all",
    "linear-gap": "linear",
    "linear-gaps": "linear",
    "affine-gap": "affine",
    "affine-gaps": "affine",
}

_ALL_PASSES = ("english-short", "english", "chinese", "random-bytes")
_DEFAULT_PASSES = ("english-short", "english", "chinese")
_DEFAULT_WIDTHS_BY_PASS = {
    "english-short": (8,),
    "english": (16, 32),
    "chinese": (16, 32),
    "random-bytes": (16, 32),
}
_PASS_ALIASES = {
    "short-english": "english-short",
    "short": "english-short",
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
    case_name: str
    backend: str
    variant: str
    generator: str
    output: str
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


@dataclass(frozen=True, slots=True)
class ScoringCase:
    name: str
    match_score: int
    mismatch_score: int
    gap_open_score: int
    gap_extend_score: int

    @property
    def is_linear(self) -> bool:
        return self.gap_open_score == self.gap_extend_score


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
        default=None,
        help=(
            "Score channel widths to benchmark. Defaults are pass-specific: "
            "english-short uses 8, text passes use 16 32."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(_DEFAULT_VARIANTS),
        help=(
            "Workloads to benchmark. Choices: sw-farrar-score, sw-score, nw-score, "
            "sw-path-info, nw-path-info, sw-path, nw-path, or groups score-only, "
            "path, full-path, all. Defaults to score-only plus path-info generators."
        ),
    )
    parser.add_argument(
        "--scoring-cases",
        nargs="+",
        default=list(_DEFAULT_SCORING_CASES),
        help=(
            "Scoring cases to benchmark. Choices: linear, affine, all. "
            "Unequal affine gaps use native fallback unless a backend has a specialized path. "
            "Defaults to linear affine."
        ),
    )
    parser.add_argument("--length", type=int, default=1024, help="Default query/target length.")
    parser.add_argument(
        "--short-length",
        type=int,
        default=31,
        help="Query/target length for the english-short 8-bit pass. Defaults to 31.",
    )
    parser.add_argument("--query-length", type=int, default=None)
    parser.add_argument("--target-length", type=int, default=None)
    parser.add_argument(
        "--passes",
        nargs="+",
        default=list(_DEFAULT_PASSES),
        help=(
            "Benchmark passes to run. Choices: english, chinese, random-bytes, "
            "english-short, all-text, all. Defaults to english-short english chinese."
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
    parser.add_argument("--affine-gap-open-score", "--affine-gap-open", type=int, default=-2)
    parser.add_argument("--affine-gap-extend-score", "--affine-gap-extend", type=int, default=-1)
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
        variant_group = _VARIANT_GROUPS.get(canonical_variant)
        if variant_group is not None:
            for available_variant in variant_group:
                if available_variant not in selected:
                    selected.append(available_variant)
            continue
        if canonical_variant not in _ALL_VARIANTS:
            choices = ", ".join((*_ALL_VARIANTS, *_VARIANT_GROUPS))
            raise BenchmarkError(f"unknown benchmark variant {variant!r}; choices: {choices}")
        if canonical_variant not in selected:
            selected.append(canonical_variant)
    return selected


def _selected_scoring_case_names(cases: Sequence[str]) -> list[str]:
    selected: list[str] = []
    for scoring_case in cases:
        canonical_case = _SCORING_CASE_ALIASES.get(scoring_case, scoring_case)
        if canonical_case == "all":
            for available_case in _ALL_SCORING_CASES:
                if available_case not in selected:
                    selected.append(available_case)
            continue
        if canonical_case not in _ALL_SCORING_CASES:
            choices = ", ".join((*_ALL_SCORING_CASES, "all"))
            raise BenchmarkError(
                f"unknown scoring case {scoring_case!r}; choices: {choices}"
            )
        if canonical_case not in selected:
            selected.append(canonical_case)
    return selected


def _build_scoring_cases(args: argparse.Namespace) -> list[ScoringCase]:
    cases: list[ScoringCase] = []
    for case_name in _selected_scoring_case_names(args.scoring_cases):
        if case_name == "linear":
            cases.append(
                ScoringCase(
                    name="linear",
                    match_score=args.match_score,
                    mismatch_score=args.mismatch_score,
                    gap_open_score=args.gap_score,
                    gap_extend_score=args.gap_score,
                )
            )
            continue
        if case_name == "affine":
            cases.append(
                ScoringCase(
                    name="affine",
                    match_score=args.match_score,
                    mismatch_score=args.mismatch_score,
                    gap_open_score=args.affine_gap_open_score,
                    gap_extend_score=args.affine_gap_extend_score,
                )
            )
            continue
        raise AssertionError(f"unhandled scoring case {case_name!r}")
    return cases


def _selected_passes(passes: Sequence[str]) -> list[str]:
    selected: list[str] = []
    for benchmark_pass in passes:
        canonical_pass = _PASS_ALIASES.get(benchmark_pass, benchmark_pass)
        if canonical_pass == "all-text":
            candidates = ("english", "chinese")
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
        gap_open_score: int | None = None,
        gap_extend_score: int | None = None,
    ) -> int:
        return self._run(
            "sw-farrar-score",
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            width,
            gap_open_score,
            gap_extend_score,
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
        gap_open_score: int | None = None,
        gap_extend_score: int | None = None,
    ) -> int:
        return self._run(
            "sw-score",
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            width,
            gap_open_score,
            gap_extend_score,
        )

    def needleman_wunsch_score(
        self,
        query: bytes,
        target: bytes,
        *,
        match_score: int,
        mismatch_score: int,
        gap_score: int,
        width: int,
        gap_open_score: int | None = None,
        gap_extend_score: int | None = None,
    ) -> int:
        return self._run(
            "nw-score",
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            width,
            gap_open_score,
            gap_extend_score,
        )

    def smith_waterman_path_info(
        self,
        query: bytes,
        target: bytes,
        *,
        match_score: int,
        mismatch_score: int,
        gap_score: int,
        width: int,
        gap_open_score: int | None = None,
        gap_extend_score: int | None = None,
    ):
        return self._run(
            "sw-path-info",
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            width,
            gap_open_score,
            gap_extend_score,
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
        gap_open_score: int | None = None,
        gap_extend_score: int | None = None,
    ):
        return self._run(
            "nw-path-info",
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            width,
            gap_open_score,
            gap_extend_score,
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
        gap_open_score: int | None = None,
        gap_extend_score: int | None = None,
    ):
        return self._run(
            "sw-path",
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            width,
            gap_open_score,
            gap_extend_score,
        )

    def needleman_wunsch_path(
        self,
        query: bytes,
        target: bytes,
        *,
        match_score: int,
        mismatch_score: int,
        gap_score: int,
        width: int,
        gap_open_score: int | None = None,
        gap_extend_score: int | None = None,
    ):
        return self._run(
            "nw-path",
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            width,
            gap_open_score,
            gap_extend_score,
        )

    def _prepare_smith_waterman_farrar_score(
        self,
        query: object,
        target: object,
        *,
        match_score: int,
        mismatch_score: int,
        gap_score: int,
        width: int,
        gap_open_score: int | None = None,
        gap_extend_score: int | None = None,
    ):
        return self._prepare_profile_score(
            "sw-farrar-score",
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            width,
            gap_open_score,
            gap_extend_score,
        )

    def _smith_waterman_farrar_score_prepared(self, prepared):
        return self._score_prepared_profile(prepared)

    def _prepare_smith_waterman_score(
        self,
        query: object,
        target: object,
        *,
        match_score: int,
        mismatch_score: int,
        gap_score: int,
        width: int,
        gap_open_score: int | None = None,
        gap_extend_score: int | None = None,
    ):
        return self._prepare_profile_score(
            "sw-score",
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            width,
            gap_open_score,
            gap_extend_score,
        )

    def _smith_waterman_score_prepared(self, prepared):
        return self._score_prepared_profile(prepared)

    def _prepare_needleman_wunsch_score(
        self,
        query: object,
        target: object,
        *,
        match_score: int,
        mismatch_score: int,
        gap_score: int,
        width: int,
        gap_open_score: int | None = None,
        gap_extend_score: int | None = None,
    ):
        return self._prepare_profile_score(
            "nw-score",
            query,
            target,
            match_score,
            mismatch_score,
            gap_score,
            width,
            gap_open_score,
            gap_extend_score,
        )

    def _needleman_wunsch_score_prepared(self, prepared):
        return self._score_prepared_profile(prepared)

    def _prepare_smith_waterman_affine_score(self, *args, **kwargs):
        return self._prepare_smith_waterman_score(*args, **kwargs)

    def _smith_waterman_affine_score_prepared(self, prepared):
        return self._score_prepared_profile(prepared)

    def _prepare_smith_waterman_affine_farrar_score(self, *args, **kwargs):
        return self._prepare_smith_waterman_farrar_score(*args, **kwargs)

    def _smith_waterman_affine_farrar_score_prepared(self, prepared):
        return self._score_prepared_profile(prepared)

    def _prepare_needleman_wunsch_affine_score(self, *args, **kwargs):
        return self._prepare_needleman_wunsch_score(*args, **kwargs)

    def _needleman_wunsch_affine_score_prepared(self, prepared):
        return self._score_prepared_profile(prepared)

    def _prepare_profile_score(
        self,
        variant: str,
        query: object,
        target: object,
        match_score: int,
        mismatch_score: int,
        gap_score: int,
        width: int,
        gap_open_score: int | None,
        gap_extend_score: int | None,
    ):
        gap_open, gap_extend = self._gap_penalties(
            gap_score,
            gap_open_score,
            gap_extend_score,
        )
        if gap_open >= 0 or gap_extend >= 0:
            raise BenchmarkError("parasail benchmark adapter requires a non-positive gap score")
        query_text, target_text, matrix = self._prepare_inputs(
            query,
            target,
            match_score,
            mismatch_score,
        )
        profile_create = getattr(self._parasail, f"profile_create_{width}", None)
        profile_function = self._profile_function(variant, width)
        if profile_create is None or profile_function is None:
            raise BenchmarkError(
                f"parasail does not expose a prepared profile function for {variant} "
                f"width {width}"
            )
        return (
            profile_function,
            profile_create(query_text, matrix),
            target_text,
            -gap_open,
            -gap_extend,
        )

    def _score_prepared_profile(self, prepared):
        profile_function, profile, target_text, gap_open_penalty, gap_extend_penalty = prepared
        return profile_function(profile, target_text, gap_open_penalty, gap_extend_penalty)

    def _gap_penalties(
        self,
        gap_score: int,
        gap_open_score: int | None,
        gap_extend_score: int | None,
    ) -> tuple[int, int]:
        gap_open = gap_score if gap_open_score is None else gap_open_score
        gap_extend = gap_score if gap_extend_score is None else gap_extend_score
        return gap_open, gap_extend

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

    def _profile_function_names(self, variant: str, width: int) -> tuple[str, ...]:
        if variant == "sw-farrar-score":
            return (f"sw_striped_profile_{width}",)
        if variant == "sw-score":
            return (f"sw_scan_profile_{width}", f"sw_striped_profile_{width}")
        if variant == "nw-score":
            return (f"nw_scan_profile_{width}", f"nw_striped_profile_{width}")
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

    def _profile_function(self, variant: str, width: int):
        for name in self._profile_function_names(variant, width):
            function = getattr(self._parasail, name, None)
            if function is not None:
                return function
        choices = ", ".join(self._profile_function_names(variant, width))
        raise BenchmarkError(
            f"parasail does not expose a prepared profile function for {variant} "
            f"width {width}: {choices}"
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
        gap_open_score: int | None = None,
        gap_extend_score: int | None = None,
    ):
        gap_open, gap_extend = self._gap_penalties(
            gap_score,
            gap_open_score,
            gap_extend_score,
        )
        if gap_open >= 0 or gap_extend >= 0:
            raise BenchmarkError("parasail benchmark adapter requires a non-positive gap score")
        query_text, target_text, matrix = self._prepare_inputs(
            query,
            target,
            match_score,
            mismatch_score,
        )
        gap_open_penalty = -gap_open
        gap_extend_penalty = -gap_extend
        result = self._function(variant, width)(
            query_text,
            target_text,
            gap_open_penalty,
            gap_extend_penalty,
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


def _generator_for_variant(variant: str) -> str:
    if variant.endswith("-score"):
        return "score-only"
    if variant.endswith("-path-info") or variant.endswith("-path"):
        return "path"
    raise AssertionError(f"unhandled benchmark variant {variant!r}")


def _output_for_variant(backend_name: str, variant: str) -> str:
    if variant.endswith("-score"):
        return "score"
    if backend_name == "parasail" and (
        variant.endswith("-path-info") or variant.endswith("-path")
    ):
        return "trace-cigar"
    if variant.endswith("-path-info"):
        return "path-info"
    if variant.endswith("-path"):
        return "full-path"
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
    if name in {"english", "english-short"}:
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
    short_length: int,
    alphabet_size: int,
    seed: int,
) -> list[BenchmarkPass]:
    if query_length < 0 or target_length < 0 or short_length < 0:
        raise BenchmarkError("sequence lengths must be non-negative")

    benchmark_passes: list[BenchmarkPass] = []
    for index, pass_name in enumerate(pass_names):
        pass_seed = seed + index * 1009
        if pass_name == "english-short":
            benchmark_passes.append(
                _build_text_pass(pass_name, short_length, short_length, pass_seed)
            )
        elif pass_name in {"english", "chinese"}:
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


def _validate_width(
    width: int,
    pass_name: str,
    query_length: int,
    target_length: int,
    scoring_case: ScoringCase,
) -> None:
    if width == 8 and pass_name == "chinese":
        raise BenchmarkError("8-bit benchmark widths are limited to English/byte passes")

    score_bound = (query_length + target_length) * max(
        abs(scoring_case.match_score),
        abs(scoring_case.mismatch_score),
        abs(scoring_case.gap_open_score),
        abs(scoring_case.gap_extend_score),
    )
    if score_bound > _SIGNED_LIMITS[width]:
        raise BenchmarkError(
            f"{pass_name}/{scoring_case.name} score bound {score_bound} exceeds signed "
            f"{width}-bit capacity {_SIGNED_LIMITS[width]}"
        )


def _widths_for_pass(
    benchmark_pass: BenchmarkPass,
    selected_widths: Sequence[int] | None,
) -> tuple[int, ...]:
    if selected_widths is not None:
        return tuple(selected_widths)
    return _DEFAULT_WIDTHS_BY_PASS[benchmark_pass.name]


def _backend_supports_scoring_case(backend: ResolvedBackend, scoring_case: ScoringCase) -> bool:
    return scoring_case.is_linear or backend.name in _BACKEND_MODULES


def _time_backend(
    backend: ResolvedBackend,
    pass_name: str,
    case_name: str,
    variant: str,
    query: object,
    target: object,
    width: int,
    iterations: int,
    warmups: int,
    match_score: int,
    mismatch_score: int,
    gap_open_score: int,
    gap_extend_score: int,
) -> BenchmarkResult:
    function = _function_for_variant(backend.module, variant)
    linear_gap_score = gap_open_score
    extra_gap_kwargs = {
        "gap_open_score": gap_open_score,
        "gap_extend_score": gap_extend_score,
    }
    is_affine = gap_open_score != gap_extend_score

    prepared_names: tuple[str, str] | None = None
    if variant in {"sw-farrar-score", "sw-score", "nw-score"}:
        affine_prepared_names = {
            "sw-farrar-score": (
                "_prepare_smith_waterman_affine_farrar_score",
                "_smith_waterman_affine_farrar_score_prepared",
            ),
            "sw-score": (
                "_prepare_smith_waterman_affine_score",
                "_smith_waterman_affine_score_prepared",
            ),
            "nw-score": (
                "_prepare_needleman_wunsch_affine_score",
                "_needleman_wunsch_affine_score_prepared",
            ),
        }
        linear_prepared_names = {
            "sw-farrar-score": (
                "_prepare_smith_waterman_farrar_score",
                "_smith_waterman_farrar_score_prepared",
            ),
            "sw-score": (
                "_prepare_smith_waterman_score",
                "_smith_waterman_score_prepared",
            ),
            "nw-score": (
                "_prepare_needleman_wunsch_score",
                "_needleman_wunsch_score_prepared",
            ),
        }
        prepared_names = (
            affine_prepared_names[variant] if is_affine else linear_prepared_names[variant]
        )

    if prepared_names is not None:
        prepare_name, prepared_score_name = prepared_names
        if hasattr(backend.module, prepare_name) and hasattr(backend.module, prepared_score_name):
            prepared = getattr(backend.module, prepare_name)(
                query,
                target,
                match_score=match_score,
                mismatch_score=mismatch_score,
                gap_score=linear_gap_score,
                width=width,
                **extra_gap_kwargs,
            )
            score_prepared = getattr(backend.module, prepared_score_name)

            def run_once() -> Any:
                return score_prepared(prepared)

        else:

            def run_once() -> Any:
                return function(
                    query,
                    target,
                    match_score=match_score,
                    mismatch_score=mismatch_score,
                    gap_score=linear_gap_score,
                    width=width,
                    **extra_gap_kwargs,
                )

    else:

        def run_once() -> Any:
            return function(
                query,
                target,
                match_score=match_score,
                mismatch_score=mismatch_score,
                gap_score=linear_gap_score,
                width=width,
                **extra_gap_kwargs,
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
        case_name=case_name,
        backend=backend.name,
        variant=variant,
        generator=_generator_for_variant(variant),
        output=_output_for_variant(backend.name, variant),
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
        (
            result.pass_name,
            result.case_name,
            result.variant,
            result.score_width,
        ): result.median_seconds
        for result in results
        if result.backend == "generic"
    }
    output: list[BenchmarkResult] = []
    for result in results:
        generic_median = generic_medians.get(
            (result.pass_name, result.case_name, result.variant, result.score_width)
        )
        speedup = None if generic_median is None else generic_median / result.median_seconds
        output.append(
            BenchmarkResult(
                pass_name=result.pass_name,
                case_name=result.case_name,
                backend=result.backend,
                variant=result.variant,
                generator=result.generator,
                output=result.output,
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
        "pass,case,backend,variant,generator,output,score_width,score,iterations,"
        "warmups,best_seconds,median_seconds,mean_seconds,cells_per_second,"
        "speedup_vs_generic"
    )
    for result in results:
        speedup = "" if result.speedup_vs_generic is None else f"{result.speedup_vs_generic:.6g}"
        print(
            f"{result.pass_name},{result.case_name},{result.backend},{result.variant},"
            f"{result.generator},{result.output},{result.score_width},{result.score},"
            f"{result.iterations},"
            f"{result.warmups},{result.best_seconds:.9g},{result.median_seconds:.9g},"
            f"{result.mean_seconds:.9g},{result.cells_per_second:.9g},{speedup}"
        )


def _print_table(
    results: Sequence[BenchmarkResult],
    query_length: int,
    target_length: int,
    short_length: int,
    pass_names: Sequence[str],
    scoring_case_names: Sequence[str],
    seed: int,
) -> None:
    print("# stride-align-benchmark")
    print(f"# python={platform.python_version()} platform={platform.platform()}")
    print(
        f"# query_length={query_length} target_length={target_length} "
        f"short_length={short_length} passes={','.join(pass_names)} "
        f"cases={','.join(scoring_case_names)} seed={seed}"
    )
    print(
        "# generator distinguishes score-only from path-producing calls; "
        "output distinguishes native path materialization from parasail trace/cigar."
    )
    headers = (
        "pass",
        "case",
        "backend",
        "variant",
        "generator",
        "output",
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
                result.case_name,
                result.backend,
                result.variant,
                result.generator,
                result.output,
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
    pass_names = _selected_passes(args.passes)
    benchmark_passes = _build_benchmark_passes(
        pass_names,
        query_length,
        target_length,
        args.short_length,
        args.alphabet_size,
        args.seed,
    )
    backends = _resolve_backends(args.backends)
    variants = _selected_variants(args.variants)
    scoring_cases = _build_scoring_cases(args)

    results: list[BenchmarkResult] = []
    for benchmark_pass in benchmark_passes:
        for scoring_case in scoring_cases:
            for variant in variants:
                for width in _widths_for_pass(benchmark_pass, args.widths):
                    _validate_width(
                        width,
                        benchmark_pass.name,
                        len(benchmark_pass.query),
                        len(benchmark_pass.target),
                        scoring_case,
                    )
                    expected_score: int | None = None
                    for backend in backends:
                        if not _backend_supports_scoring_case(backend, scoring_case):
                            continue
                        result = _time_backend(
                            backend,
                            benchmark_pass.name,
                            scoring_case.name,
                            variant,
                            benchmark_pass.query,
                            benchmark_pass.target,
                            width,
                            args.iterations,
                            args.warmups,
                            scoring_case.match_score,
                            scoring_case.mismatch_score,
                            scoring_case.gap_open_score,
                            scoring_case.gap_extend_score,
                        )
                        if expected_score is None:
                            expected_score = result.score
                        elif result.score != expected_score:
                            raise BenchmarkError(
                                f"{backend.name} {benchmark_pass.name} {scoring_case.name} "
                                f"{variant} width {width} returned {result.score}, "
                                f"expected {expected_score}"
                            )
                        results.append(result)

    if not results:
        raise BenchmarkError("no compatible benchmark combinations were selected")

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
            args.short_length,
            _selected_passes(args.passes),
            _selected_scoring_case_names(args.scoring_cases),
            args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
