"""Python bootstrap and dynamic dispatcher for alignment backends."""

from __future__ import annotations

import enum
import importlib
import os
import warnings
from types import ModuleType
from typing import Any

import numpy as np

from ._cpu import (
    BackendKind,
    BackendRecord,
    available_backends,
    backend_is_available,
    detect_best_backend,
)


def _backend_module_name(kind: BackendKind) -> str:
    match kind:
        case BackendKind.GENERIC:
            return "stride_align._generic"
        case BackendKind.SWAR:
            return "stride_align._swar"
        case BackendKind.X86_SSE41:
            return "stride_align._sse41"
        case BackendKind.X86_AVX2:
            return "stride_align._avx2"
        case BackendKind.X86_AVX512BWVL:
            return "stride_align._avx512bwvl"
        case BackendKind.X86_AVX10_256:
            return "stride_align._avx10_256"
        case BackendKind.X86_AVX10_512:
            return "stride_align._avx10_512"
        case BackendKind.LINUX_AARCH64_NEON:
            return "stride_align._neon"
        case BackendKind.LINUX_AARCH64_SVE:
            return "stride_align._sve"
        case BackendKind.LINUX_AARCH64_SVE2:
            return "stride_align._sve2"
        case BackendKind.MACOS_ARM64_NEON:
            return "stride_align._macos_arm64_neon"
        case BackendKind.LINUX_LOONGARCH64_LSX:
            return "stride_align._lsx"
        case BackendKind.LINUX_LOONGARCH64_LASX:
            return "stride_align._lasx"
        case BackendKind.LINUX_POWERPC64_VSX:
            return "stride_align._vsx"
        case BackendKind.LINUX_RISCV64_RVV:
            return "stride_align._rvv"
        case _:
            return "stride_align._generic"


def _import_module_suppressing_duplicate_type_warnings(module_name: str) -> ModuleType:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"nanobind: type "
                r"'(Alignment(Result|Path)|_PreparedAffineCigar|_PreparedScoreBatch|"
                r"_PreparedAffineScoreBatch)' was already registered!"
            ),
            category=RuntimeWarning,
        )
        return importlib.import_module(module_name)


def _import_backend(kind: BackendKind) -> ModuleType:
    return _import_module_suppressing_duplicate_type_warnings(_backend_module_name(kind))


def _load_backend_modules() -> tuple[dict[BackendKind, ModuleType], dict[BackendKind, ModuleType]]:
    modules: dict[BackendKind, ModuleType] = {}
    available_modules: dict[BackendKind, ModuleType] = {}

    try:
        generic_module = _import_backend(BackendKind.GENERIC)
    except ImportError:
        generic_module = importlib.import_module("stride_align._pybackend")
    modules[BackendKind.GENERIC] = generic_module
    available_modules[BackendKind.GENERIC] = generic_module

    for record in available_backends():
        if record.kind is BackendKind.GENERIC or not record.compiled:
            continue
        try:
            module = _import_backend(record.kind)
        except ImportError:
            continue
        modules[record.kind] = module
        if record.available:
            available_modules[record.kind] = module

    return modules, available_modules


_BACKENDS, _AVAILABLE_BACKENDS = _load_backend_modules()
_GENERIC_BACKEND = _BACKENDS[BackendKind.GENERIC]

AlignmentPath = _GENERIC_BACKEND.AlignmentPath
AlignmentResult = _GENERIC_BACKEND.AlignmentResult

_REAL_SIMD_WIDE_PRIORITY = (
    BackendKind.X86_AVX10_512,
    BackendKind.X86_AVX512BWVL,
    BackendKind.X86_AVX10_256,
    BackendKind.X86_AVX2,
    BackendKind.X86_SSE41,
    BackendKind.LINUX_AARCH64_SVE2,
    BackendKind.LINUX_AARCH64_SVE,
    BackendKind.LINUX_RISCV64_RVV,
    BackendKind.LINUX_LOONGARCH64_LASX,
    BackendKind.LINUX_LOONGARCH64_LSX,
    BackendKind.LINUX_AARCH64_NEON,
    BackendKind.MACOS_ARM64_NEON,
    BackendKind.LINUX_POWERPC64_VSX,
)

_REAL_SIMD_NARROW_PRIORITY = (
    BackendKind.X86_SSE41,
    BackendKind.X86_AVX2,
    BackendKind.X86_AVX10_256,
    BackendKind.X86_AVX512BWVL,
    BackendKind.X86_AVX10_512,
    BackendKind.LINUX_AARCH64_NEON,
    BackendKind.MACOS_ARM64_NEON,
    BackendKind.LINUX_AARCH64_SVE2,
    BackendKind.LINUX_AARCH64_SVE,
    BackendKind.LINUX_LOONGARCH64_LSX,
    BackendKind.LINUX_LOONGARCH64_LASX,
    BackendKind.LINUX_POWERPC64_VSX,
    BackendKind.LINUX_RISCV64_RVV,
)

_SHORT_LINEAR_FARRAR_PRIORITY = (
    BackendKind.X86_AVX2,
    BackendKind.X86_SSE41,
    BackendKind.SWAR,
    BackendKind.X86_AVX10_256,
    BackendKind.X86_AVX512BWVL,
    BackendKind.X86_AVX10_512,
    BackendKind.LINUX_AARCH64_NEON,
    BackendKind.MACOS_ARM64_NEON,
    BackendKind.LINUX_AARCH64_SVE2,
    BackendKind.LINUX_AARCH64_SVE,
    BackendKind.LINUX_LOONGARCH64_LSX,
    BackendKind.LINUX_LOONGARCH64_LASX,
    BackendKind.LINUX_POWERPC64_VSX,
    BackendKind.LINUX_RISCV64_RVV,
)

_VALID_WIDTHS = {0, 8, 16, 32, 64}


def _resolve_gap_scores(
    gap_score: int,
    gap_open_score: int | None,
    gap_extend_score: int | None,
) -> tuple[int, int]:
    gap_open = gap_score if gap_open_score is None else gap_open_score
    gap_extend = gap_open if gap_extend_score is None else gap_extend_score
    return int(gap_open), int(gap_extend)


def _normalize_alignment_score(
    raw_score: int,
    query_length: int,
    target_length: int,
    *,
    is_local: bool,
    match_score: int,
) -> float:
    # Length-normalized similarity in [0, 1]. SW divides by the shorter
    # length (the strict upper bound for any local alignment); NW divides
    # by the longer length (the upper bound when one string fully embeds
    # in the other).
    if match_score <= 0:
        raise ValueError(
            "match_score must be positive to compute a normalized score"
        )
    if query_length == 0 and target_length == 0:
        return 1.0
    denominator = (
        min(query_length, target_length)
        if is_local
        else max(query_length, target_length)
    ) * match_score
    if denominator <= 0:
        return 0.0
    return min(1.0, max(0.0, raw_score / denominator))


def _normalize_alignment_scores(
    raw_scores: np.ndarray,
    query_length: int,
    target_lengths: np.ndarray,
    *,
    is_local: bool,
    match_score: int,
) -> np.ndarray:
    # Vectorized form of _normalize_alignment_score over a batch.
    if match_score <= 0:
        raise ValueError(
            "match_score must be positive to compute a normalized score"
        )
    if raw_scores.size == 0:
        return np.empty(0, dtype=np.float64)
    if is_local:
        denoms = np.minimum(query_length, target_lengths) * match_score
    else:
        denoms = np.maximum(query_length, target_lengths) * match_score
    out = np.zeros(raw_scores.shape, dtype=np.float64)
    np.divide(
        raw_scores.astype(np.float64, copy=False),
        denoms,
        out=out,
        where=denoms > 0,
    )
    np.clip(out, 0.0, 1.0, out=out)
    if query_length == 0:
        out = np.where(target_lengths == 0, 1.0, out)
    return out


def _forced_width(width: int | None) -> int:
    if width is None:
        return 0
    if width not in _VALID_WIDTHS:
        raise ValueError("width must be None, 0, 8, 16, 32, or 64")
    return int(width)


def _length(value: object) -> int | None:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def _profile_traceback_compatible(query: object, target: object) -> bool:
    if isinstance(query, bytes) and isinstance(target, bytes):
        return True
    if isinstance(query, str) and isinstance(target, str):
        return len(set(query) | set(target)) <= 256
    if isinstance(query, (bytes, str)) or isinstance(target, (bytes, str)):
        return False
    try:
        return len(set(query) | set(target)) <= 256  # type: ignore[arg-type]
    except TypeError:
        return False


def _score_width(
    query_length: int,
    target_length: int,
    *,
    match_score: int,
    mismatch_score: int,
    gap_open_score: int,
    gap_extend_score: int,
    width: int,
) -> int:
    if width:
        return width

    score_bound = (query_length + target_length) * max(
        abs(int(match_score)),
        abs(int(mismatch_score)),
        abs(int(gap_open_score)),
        abs(int(gap_extend_score)),
    )
    if score_bound <= (1 << 7) - 1:
        return 8
    if score_bound <= (1 << 15) - 1:
        return 16
    if score_bound <= (1 << 31) - 1:
        return 32
    return 64


def _first_available(kinds: tuple[BackendKind, ...]) -> ModuleType | None:
    for kind in kinds:
        backend = _AVAILABLE_BACKENDS.get(kind)
        if backend is not None:
            return backend
    return None


def _select_backend(
    *,
    variant: str,
    query: object,
    target: object,
    match_score: int,
    mismatch_score: int,
    gap_open_score: int,
    gap_extend_score: int,
    width: int | None,
) -> ModuleType:
    forced_width = _forced_width(width)
    query_length = _length(query)
    target_length = _length(target)
    if query_length is None or target_length is None:
        return _GENERIC_BACKEND

    cells = query_length * target_length
    affine = gap_open_score != gap_extend_score
    score_width = _score_width(
        query_length,
        target_length,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=forced_width,
    )

    if cells == 0:
        return _GENERIC_BACKEND

    if not affine and variant in {"sw-score", "nw-score"}:
        if cells <= 4096 and score_width == 8:
            priority = (
                _SHORT_LINEAR_FARRAR_PRIORITY
                if variant == "sw-score"
                else _REAL_SIMD_NARROW_PRIORITY
            )
            return _first_available(priority) or _GENERIC_BACKEND
        return _first_available(_REAL_SIMD_WIDE_PRIORITY) or _GENERIC_BACKEND

    if variant in {
        "sw-path",
        "nw-path",
        "sw-path-info",
        "nw-path-info",
        "sw-cigar",
        "nw-cigar",
    }:
        if not _profile_traceback_compatible(query, target):
            return _GENERIC_BACKEND
        priority = (
            _REAL_SIMD_NARROW_PRIORITY
            if cells <= 4096 and score_width == 8
            else _REAL_SIMD_WIDE_PRIORITY
        )
        return _first_available(priority) or _GENERIC_BACKEND

    if affine and variant in {"sw-score", "nw-score"}:
        if not _profile_traceback_compatible(query, target):
            return _GENERIC_BACKEND
        priority = (
            _REAL_SIMD_NARROW_PRIORITY
            if cells <= 4096 and score_width == 8
            else _REAL_SIMD_WIDE_PRIORITY
        )
        return _first_available(priority) or _GENERIC_BACKEND

    if variant != "sw-farrar-score":
        return _GENERIC_BACKEND

    if affine and cells <= 4096 and score_width == 8:
        return _first_available(_REAL_SIMD_NARROW_PRIORITY) or _GENERIC_BACKEND

    if not affine and cells <= 4096 and score_width == 8:
        return _first_available(_SHORT_LINEAR_FARRAR_PRIORITY) or _GENERIC_BACKEND

    if not affine and cells <= 4096:
        return _first_available(_REAL_SIMD_WIDE_PRIORITY) or _GENERIC_BACKEND

    return _first_available(_REAL_SIMD_WIDE_PRIORITY) or _GENERIC_BACKEND


def _dispatch(
    function_name: str,
    variant: str,
    query: object,
    target: object,
    *,
    match_score: int,
    mismatch_score: int,
    gap_score: int,
    gap_open_score: int | None,
    gap_extend_score: int | None,
    width: int | None,
) -> Any:
    gap_open, gap_extend = _resolve_gap_scores(gap_score, gap_open_score, gap_extend_score)
    backend = _select_backend(
        variant=variant,
        query=query,
        target=target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_open_score=gap_open,
        gap_extend_score=gap_extend,
        width=width,
    )
    return getattr(backend, function_name)(
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


_SCORE_VARIANT_ALIASES = {
    "smith_waterman": "sw-score",
    "smith-waterman": "sw-score",
    "sw": "sw-score",
    "local": "sw-score",
    "sw-score": "sw-score",
    "needleman_wunsch": "nw-score",
    "needleman-wunsch": "nw-score",
    "nw": "nw-score",
    "global": "nw-score",
    "nw-score": "nw-score",
    "smith_waterman_farrar": "sw-farrar-score",
    "smith-waterman-farrar": "sw-farrar-score",
    "farrar": "sw-farrar-score",
    "sw-farrar-score": "sw-farrar-score",
}

_SCORE_MANY_FUNCTIONS = {
    "sw-score": ("smith_waterman_scores", "smith_waterman_score"),
    "nw-score": ("needleman_wunsch_scores", "needleman_wunsch_score"),
    "sw-farrar-score": ("smith_waterman_farrar_scores", "smith_waterman_farrar_score"),
}


def _canonical_score_variant(variant: str) -> str:
    canonical = _SCORE_VARIANT_ALIASES.get(variant)
    if canonical is None:
        choices = ", ".join(sorted(_SCORE_VARIANT_ALIASES))
        raise ValueError(f"unknown score variant {variant!r}; choices: {choices}")
    return canonical


def _materialize_targets(targets: object) -> tuple[object, ...]:
    if isinstance(targets, (str, bytes)):
        raise TypeError("targets must be an iterable of target sequences, not a single str/bytes")
    try:
        return tuple(targets)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("targets must be an iterable of target sequences") from exc


def _representative_target(targets: tuple[object, ...]) -> object:
    def target_length(target: object) -> int:
        length = _length(target)
        return -1 if length is None else length

    return max(targets, key=target_length)


def _dispatch_many(
    variant: str,
    query: object,
    targets: object,
    *,
    match_score: int,
    mismatch_score: int,
    gap_score: int,
    gap_open_score: int | None,
    gap_extend_score: int | None,
    width: int | None,
) -> np.ndarray:
    target_tuple = _materialize_targets(targets)
    if not target_tuple:
        return np.empty(0, dtype=np.int64)

    canonical_variant = _canonical_score_variant(variant)
    gap_open, gap_extend = _resolve_gap_scores(gap_score, gap_open_score, gap_extend_score)
    backend = _select_backend(
        variant=canonical_variant,
        query=query,
        target=_representative_target(target_tuple),
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_open_score=gap_open,
        gap_extend_score=gap_extend,
        width=width,
    )
    many_name, scalar_name = _SCORE_MANY_FUNCTIONS[canonical_variant]
    many_function = getattr(backend, many_name, None)
    if many_function is not None:
        result = many_function(
            query,
            target_tuple,
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
            width=width,
        )
        return np.asarray(result, dtype=np.int64)

    scalar_function = getattr(backend, scalar_name)
    return np.fromiter(
        (
            scalar_function(
                query,
                target,
                match_score=match_score,
                mismatch_score=mismatch_score,
                gap_score=gap_score,
                gap_open_score=gap_open_score,
                gap_extend_score=gap_extend_score,
                width=width,
            )
            for target in target_tuple
        ),
        dtype=np.int64,
        count=len(target_tuple),
    )


class Scores:
    """Score facade for comparing one query against many targets."""

    __slots__ = (
        "query",
        "variant",
        "match_score",
        "mismatch_score",
        "gap_score",
        "gap_open_score",
        "gap_extend_score",
        "width",
    )

    def __init__(
        self,
        query: object,
        *,
        variant: str = "smith_waterman",
        match_score: int = 2,
        mismatch_score: int = -1,
        gap_score: int = -1,
        gap_open_score: int | None = None,
        gap_extend_score: int | None = None,
        width: int | None = None,
    ) -> None:
        self.query = query
        self.variant = _canonical_score_variant(variant)
        self.match_score = int(match_score)
        self.mismatch_score = int(mismatch_score)
        self.gap_score = int(gap_score)
        self.gap_open_score = None if gap_open_score is None else int(gap_open_score)
        self.gap_extend_score = None if gap_extend_score is None else int(gap_extend_score)
        self.width = None if width is None else _forced_width(width)

    def compare(self, targets: object) -> np.ndarray:
        return _dispatch_many(
            self.variant,
            self.query,
            targets,
            match_score=self.match_score,
            mismatch_score=self.mismatch_score,
            gap_score=self.gap_score,
            gap_open_score=self.gap_open_score,
            gap_extend_score=self.gap_extend_score,
            width=self.width,
        )


def smith_waterman_scores(
    query: object,
    targets: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    return _dispatch_many(
        "sw-score",
        query,
        targets,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


def smith_waterman_normalized_scores(
    query: object,
    targets: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    target_tuple = _materialize_targets(targets)
    raw_scores = _dispatch_many(
        "sw-score",
        query,
        target_tuple,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )
    target_lengths = np.fromiter(
        (len(t) for t in target_tuple),  # type: ignore[arg-type]
        dtype=np.int64,
        count=len(target_tuple),
    )
    return _normalize_alignment_scores(
        raw_scores,
        len(query),  # type: ignore[arg-type]
        target_lengths,
        is_local=True,
        match_score=match_score,
    )


def needleman_wunsch_scores(
    query: object,
    targets: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    return _dispatch_many(
        "nw-score",
        query,
        targets,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


def needleman_wunsch_normalized_scores(
    query: object,
    targets: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    target_tuple = _materialize_targets(targets)
    raw_scores = _dispatch_many(
        "nw-score",
        query,
        target_tuple,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )
    target_lengths = np.fromiter(
        (len(t) for t in target_tuple),  # type: ignore[arg-type]
        dtype=np.int64,
        count=len(target_tuple),
    )
    return _normalize_alignment_scores(
        raw_scores,
        len(query),  # type: ignore[arg-type]
        target_lengths,
        is_local=False,
        match_score=match_score,
    )


def smith_waterman_farrar_scores(
    query: object,
    targets: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    return _dispatch_many(
        "sw-farrar-score",
        query,
        targets,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


def smith_waterman_farrar_normalized_scores(
    query: object,
    targets: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    target_tuple = _materialize_targets(targets)
    raw_scores = _dispatch_many(
        "sw-farrar-score",
        query,
        target_tuple,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )
    target_lengths = np.fromiter(
        (len(t) for t in target_tuple),  # type: ignore[arg-type]
        dtype=np.int64,
        count=len(target_tuple),
    )
    return _normalize_alignment_scores(
        raw_scores,
        len(query),  # type: ignore[arg-type]
        target_lengths,
        is_local=True,
        match_score=match_score,
    )


def smith_waterman_score(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> int:
    return _dispatch(
        "smith_waterman_score",
        "sw-score",
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


def smith_waterman_normalized_score(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> float:
    raw = smith_waterman_score(
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )
    return _normalize_alignment_score(
        int(raw),
        len(query),  # type: ignore[arg-type]
        len(target),  # type: ignore[arg-type]
        is_local=True,
        match_score=match_score,
    )


def smith_waterman_path(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> AlignmentResult:
    return _dispatch(
        "smith_waterman_path",
        "sw-path",
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


def smith_waterman_path_info(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> AlignmentPath:
    return _dispatch(
        "smith_waterman_path_info",
        "sw-path-info",
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


def smith_waterman_cigar(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> str:
    return _dispatch(
        "smith_waterman_cigar",
        "sw-cigar",
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


def smith_waterman_trace_cigar(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> str:
    return _dispatch(
        "smith_waterman_trace_cigar",
        "sw-cigar",
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


def smith_waterman_trade_cigar(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> str:
    return _dispatch(
        "smith_waterman_trade_cigar",
        "sw-cigar",
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


def smith_waterman_farrar_score(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> int:
    return _dispatch(
        "smith_waterman_farrar_score",
        "sw-farrar-score",
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


def smith_waterman_farrar_normalized_score(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> float:
    raw = smith_waterman_farrar_score(
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )
    return _normalize_alignment_score(
        int(raw),
        len(query),  # type: ignore[arg-type]
        len(target),  # type: ignore[arg-type]
        is_local=True,
        match_score=match_score,
    )


def needleman_wunsch_score(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> int:
    return _dispatch(
        "needleman_wunsch_score",
        "nw-score",
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


def needleman_wunsch_normalized_score(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> float:
    raw = needleman_wunsch_score(
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )
    return _normalize_alignment_score(
        int(raw),
        len(query),  # type: ignore[arg-type]
        len(target),  # type: ignore[arg-type]
        is_local=False,
        match_score=match_score,
    )


def needleman_wunsch_path(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> AlignmentResult:
    return _dispatch(
        "needleman_wunsch_path",
        "nw-path",
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


def needleman_wunsch_path_info(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> AlignmentPath:
    return _dispatch(
        "needleman_wunsch_path_info",
        "nw-path-info",
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


def needleman_wunsch_cigar(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> str:
    return _dispatch(
        "needleman_wunsch_cigar",
        "nw-cigar",
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


def needleman_wunsch_trace_cigar(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> str:
    return _dispatch(
        "needleman_wunsch_trace_cigar",
        "nw-cigar",
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


def needleman_wunsch_trade_cigar(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> str:
    return _dispatch(
        "needleman_wunsch_trade_cigar",
        "nw-cigar",
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


# Levenshtein has no per-call scoring parameters, so the best backend is
# pinned once at module import instead of being rediscovered on every
# call. _LEVENSHTEIN_BACKEND holds the same module object the public
# functions used to look up via _first_available; if the available
# backend set changes at runtime (rare; mostly for tests that monkey-
# patch _AVAILABLE_BACKENDS), call _refresh_levenshtein_backend().
_LEVENSHTEIN_BACKEND = (
    _first_available(_REAL_SIMD_WIDE_PRIORITY) or _GENERIC_BACKEND
)


def _refresh_levenshtein_backend() -> None:
    global _LEVENSHTEIN_BACKEND
    _LEVENSHTEIN_BACKEND = (
        _first_available(_REAL_SIMD_WIDE_PRIORITY) or _GENERIC_BACKEND
    )


def levenshtein_score(
    query: object,
    target: object,
    score_cutoff: int | None = None,
) -> int:
    """Levenshtein edit distance between two sequences (lower is more similar).

    If ``score_cutoff`` is set, the kernel returns as soon as it can prove the
    final distance must exceed it; in that case the returned score is clamped
    to ``score_cutoff + 1`` (the rapidfuzz convention).
    """
    return int(_LEVENSHTEIN_BACKEND.levenshtein_score(query, target, score_cutoff))


def levenshtein_normalized_score(
    query: object,
    target: object,
    score_cutoff: int | None = None,
) -> float:
    """Length-normalized Levenshtein similarity in [0, 1] (1 = identical)."""
    return float(
        _LEVENSHTEIN_BACKEND.levenshtein_normalized_score(query, target, score_cutoff)
    )


def levenshtein_scores(
    query: object,
    targets: object,
    score_cutoff: int | None = None,
) -> np.ndarray:
    """Distance from query to every target, returned as a numpy ndarray[int64].

    If ``score_cutoff`` is set, each lane bails early once its score is
    provably above the cutoff; bailed-out entries appear as
    ``score_cutoff + 1`` in the result.
    """
    # The native binding accepts any sequence (PySequence_Fast handles
    # lists and tuples natively), so we skip the tuple() materialization
    # when the input is already one. The native call returns int64
    # ndarray, so np.asarray on top is a no-op we can drop entirely.
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    return _LEVENSHTEIN_BACKEND.levenshtein_scores(query, targets, score_cutoff)


def levenshtein_normalized_scores(
    query: object,
    targets: object,
    score_cutoff: int | None = None,
) -> np.ndarray:
    """Normalized similarity to every target, returned as a numpy ndarray[float64]."""
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    return _LEVENSHTEIN_BACKEND.levenshtein_normalized_scores(
        query, targets, score_cutoff
    )


def damerau_levenshtein_score(query: object, target: object) -> int:
    """Optimal String Alignment (OSA) distance between two sequences.

    Like Levenshtein but adjacent transpositions ("ab" → "ba") cost 1
    instead of 2 substitutions. Restricted in the OSA sense: each
    character can participate in at most one edit operation, so a
    transposition can't be combined with another edit on the same chars.
    This is what rapidfuzz exposes as ``OSA.distance``.
    """
    return int(_LEVENSHTEIN_BACKEND.damerau_levenshtein_score(query, target))


def damerau_levenshtein_normalized_score(query: object, target: object) -> float:
    """Length-normalized OSA similarity in [0, 1] (1 = identical)."""
    return float(_LEVENSHTEIN_BACKEND.damerau_levenshtein_normalized_score(query, target))


def damerau_levenshtein_scores(query: object, targets: object) -> np.ndarray:
    """OSA distance from query to every target, returned as ndarray[int64]."""
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    return _LEVENSHTEIN_BACKEND.damerau_levenshtein_scores(query, targets)


def damerau_levenshtein_normalized_scores(query: object, targets: object) -> np.ndarray:
    """Normalized OSA similarity per target, returned as ndarray[float64]."""
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    return _LEVENSHTEIN_BACKEND.damerau_levenshtein_normalized_scores(query, targets)


def hamming_score(query: object, target: object) -> int:
    """Hamming distance between two equal-length sequences.

    Raises ``ValueError`` when ``len(query) != len(target)``. Pad inputs
    yourself if you want length-tolerant behavior.
    """
    return int(_LEVENSHTEIN_BACKEND.hamming_score(query, target))


def hamming_normalized_score(query: object, target: object) -> float:
    """Hamming similarity in [0, 1] (1 = identical). Raises ValueError on length mismatch."""
    return float(_LEVENSHTEIN_BACKEND.hamming_normalized_score(query, target))


def hamming_scores(query: object, targets: object) -> np.ndarray:
    """Hamming distance from query to every target, returned as ndarray[int64].

    Every target must have the same length as ``query``; otherwise
    ``ValueError`` is raised (the call processes targets in order and
    raises at the first mismatch).
    """
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    return _LEVENSHTEIN_BACKEND.hamming_scores(query, targets)


def hamming_normalized_scores(query: object, targets: object) -> np.ndarray:
    """Hamming similarity per target, returned as ndarray[float64]."""
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    return _LEVENSHTEIN_BACKEND.hamming_normalized_scores(query, targets)


# Jaro / Jaro-Winkler. Both are similarity in [0, 1]; there is no
# matching distance form. Defaults match rapidfuzz: prefix_weight=0.1,
# prefix_threshold=0.7 (no Winkler bonus below 0.7 base similarity),
# prefix_cap=4 characters of common prefix.


def jaro_similarity(query: object, target: object) -> float:
    """Jaro similarity in [0, 1] (1 = identical, 0 = no matches in window)."""
    return float(_LEVENSHTEIN_BACKEND.jaro_similarity(query, target))


def jaro_winkler_similarity(
    query: object,
    target: object,
    *,
    prefix_weight: float = 0.1,
    prefix_threshold: float = 0.7,
    prefix_cap: int = 4,
) -> float:
    """Jaro-Winkler similarity in [0, 1].

    Adds ``L * prefix_weight * (1 - jaro)`` to the base Jaro score
    when the base score is at least ``prefix_threshold``; ``L`` is the
    common-prefix length, capped at ``prefix_cap`` characters.
    Defaults match ``rapidfuzz.distance.JaroWinkler``.
    """
    return float(
        _LEVENSHTEIN_BACKEND.jaro_winkler_similarity(
            query,
            target,
            prefix_weight=prefix_weight,
            prefix_threshold=prefix_threshold,
            prefix_cap=prefix_cap,
        )
    )


def jaro_similarities(query: object, targets: object) -> np.ndarray:
    """Jaro similarity per target, returned as ndarray[float64]."""
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    return _LEVENSHTEIN_BACKEND.jaro_similarities(query, targets)


def jaro_winkler_similarities(
    query: object,
    targets: object,
    *,
    prefix_weight: float = 0.1,
    prefix_threshold: float = 0.7,
    prefix_cap: int = 4,
) -> np.ndarray:
    """Jaro-Winkler similarity per target, returned as ndarray[float64]."""
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    return _LEVENSHTEIN_BACKEND.jaro_winkler_similarities(
        query,
        targets,
        prefix_weight=prefix_weight,
        prefix_threshold=prefix_threshold,
        prefix_cap=prefix_cap,
    )


# k-best (`*_top_k` and the unified `extract`). The native bindings fold
# the SIMD scoring kernel and an O(N) std::nth_element partition into a
# single C++ call, returning list[(target, score, index)] without a
# numpy round-trip. Direction is baked into the function name:
# `*_top_k` returns the k *smallest* distances; `*_normalized_top_k`
# returns the k *largest* similarities. Order *within* the returned
# list is unspecified — only that it contains the top-k. Sort it
# yourself if you need it ordered. extract() dispatches by Scorer.


class Scorer(enum.IntEnum):
    """Algorithm identifier for ``extract()``.

    Integer values are part of the user-visible contract and must stay
    in sync with the ``Scorer`` enum class in ``src/cpp/topk.hpp``.
    """

    LEVENSHTEIN = 0
    LEVENSHTEIN_NORMALIZED = 1
    DAMERAU_LEVENSHTEIN = 2
    DAMERAU_LEVENSHTEIN_NORMALIZED = 3
    HAMMING = 4
    HAMMING_NORMALIZED = 5
    JARO = 6
    JARO_WINKLER = 7


def _coerce_targets(targets: object) -> object:
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        return tuple(targets)
    return targets


def levenshtein_top_k(
    query: object,
    targets: object,
    k: int = 5,
) -> list[tuple[object, int, int]]:
    """Top-k targets by lowest Levenshtein distance (ascending).

    Returns ``list[(target, score, index)]`` of at most ``k`` entries.
    Each tuple gives the target object, its distance, and its position
    in the input sequence. Distances are int64.
    """
    return _LEVENSHTEIN_BACKEND.levenshtein_top_k(query, _coerce_targets(targets), k)


def levenshtein_normalized_top_k(
    query: object,
    targets: object,
    k: int = 5,
) -> list[tuple[object, float, int]]:
    """Top-k targets by highest Levenshtein similarity (descending)."""
    return _LEVENSHTEIN_BACKEND.levenshtein_normalized_top_k(
        query, _coerce_targets(targets), k
    )


def damerau_levenshtein_top_k(
    query: object,
    targets: object,
    k: int = 5,
) -> list[tuple[object, int, int]]:
    """Top-k targets by lowest OSA (Damerau-Levenshtein) distance."""
    return _LEVENSHTEIN_BACKEND.damerau_levenshtein_top_k(
        query, _coerce_targets(targets), k
    )


def damerau_levenshtein_normalized_top_k(
    query: object,
    targets: object,
    k: int = 5,
) -> list[tuple[object, float, int]]:
    """Top-k targets by highest OSA similarity (descending)."""
    return _LEVENSHTEIN_BACKEND.damerau_levenshtein_normalized_top_k(
        query, _coerce_targets(targets), k
    )


def hamming_top_k(
    query: object,
    targets: object,
    k: int = 5,
) -> list[tuple[object, int, int]]:
    """Top-k targets by lowest Hamming distance.

    Raises ``ValueError`` if any target's length differs from the
    query's; pad inputs yourself if you want length-tolerant behavior.
    """
    return _LEVENSHTEIN_BACKEND.hamming_top_k(query, _coerce_targets(targets), k)


def hamming_normalized_top_k(
    query: object,
    targets: object,
    k: int = 5,
) -> list[tuple[object, float, int]]:
    """Top-k targets by highest Hamming similarity (descending)."""
    return _LEVENSHTEIN_BACKEND.hamming_normalized_top_k(
        query, _coerce_targets(targets), k
    )


def jaro_top_k(
    query: object,
    targets: object,
    k: int = 5,
) -> list[tuple[object, float, int]]:
    """Top-k targets by highest Jaro similarity."""
    return _LEVENSHTEIN_BACKEND.jaro_top_k(query, _coerce_targets(targets), k)


def jaro_winkler_top_k(
    query: object,
    targets: object,
    *,
    k: int = 5,
    prefix_weight: float = 0.1,
    prefix_threshold: float = 0.7,
    prefix_cap: int = 4,
) -> list[tuple[object, float, int]]:
    """Top-k targets by highest Jaro-Winkler similarity."""
    return _LEVENSHTEIN_BACKEND.jaro_winkler_top_k(
        query,
        _coerce_targets(targets),
        k=k,
        prefix_weight=prefix_weight,
        prefix_threshold=prefix_threshold,
        prefix_cap=prefix_cap,
    )


def smith_waterman_top_k(
    query: object,
    targets: object,
    *,
    k: int = 5,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> list[tuple[object, int, int]]:
    """Top-k targets by highest Smith-Waterman score (descending).

    Same scoring parameters as ``smith_waterman_scores``. Backend
    selection is per-call (the same logic that ``smith_waterman_scores``
    uses), so vector-width choice tracks the score range.
    """
    target_tuple = _materialize_targets(targets)
    if not target_tuple:
        return []
    gap_open, gap_extend = _resolve_gap_scores(gap_score, gap_open_score, gap_extend_score)
    backend = _select_backend(
        variant="sw-score",
        query=query,
        target=_representative_target(target_tuple),
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_open_score=gap_open,
        gap_extend_score=gap_extend,
        width=width,
    )
    return backend.smith_waterman_top_k(
        query,
        target_tuple,
        k=k,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    )


def extract(
    query: object,
    targets: object,
    *,
    scorer: "Scorer",
    k: int = 5,
) -> list[tuple[object, int | float, int]]:
    """Top-k targets ranked by ``scorer`` (a ``Scorer`` enum value).

    Distance-based scorers (``LEVENSHTEIN``, ``DAMERAU_LEVENSHTEIN``,
    ``HAMMING``) return the k smallest distances; normalized scorers
    (``*_NORMALIZED``) return the k largest similarities. Order within
    the returned list is unspecified.

    Smith-Waterman is not in the enum — its score depends on scoring
    parameters. Call ``smith_waterman_top_k`` directly for that.
    """
    return _LEVENSHTEIN_BACKEND.extract(
        query, _coerce_targets(targets), scorer=int(scorer), k=k
    )


# `_best` is a convenience: top-k with k=1, returning the single match
# directly (or None if targets is empty). Implemented as a thin wrapper
# over `*_top_k(k=1)` so all the SIMD + selection logic stays in one
# place; the per-call cost difference vs a dedicated min/max kernel is
# in the noise (nth_element with k=1 is essentially std::min_element).


def _first_or_none(result: list) -> object | None:
    return result[0] if result else None


def levenshtein_best(
    query: object,
    targets: object,
) -> tuple[object, int, int] | None:
    """Single target with the lowest Levenshtein distance.

    Returns ``(target, score, index)`` or ``None`` for empty targets.
    """
    return _first_or_none(levenshtein_top_k(query, targets, 1))


def levenshtein_normalized_best(
    query: object,
    targets: object,
) -> tuple[object, float, int] | None:
    """Single target with the highest Levenshtein similarity."""
    return _first_or_none(levenshtein_normalized_top_k(query, targets, 1))


def damerau_levenshtein_best(
    query: object,
    targets: object,
) -> tuple[object, int, int] | None:
    """Single target with the lowest OSA distance."""
    return _first_or_none(damerau_levenshtein_top_k(query, targets, 1))


def damerau_levenshtein_normalized_best(
    query: object,
    targets: object,
) -> tuple[object, float, int] | None:
    """Single target with the highest OSA similarity."""
    return _first_or_none(damerau_levenshtein_normalized_top_k(query, targets, 1))


def hamming_best(
    query: object,
    targets: object,
) -> tuple[object, int, int] | None:
    """Single target with the lowest Hamming distance.

    Raises ``ValueError`` if any target's length differs from the
    query's; pad inputs yourself if you want length-tolerant behavior.
    """
    return _first_or_none(hamming_top_k(query, targets, 1))


def hamming_normalized_best(
    query: object,
    targets: object,
) -> tuple[object, float, int] | None:
    """Single target with the highest Hamming similarity."""
    return _first_or_none(hamming_normalized_top_k(query, targets, 1))


def jaro_best(
    query: object,
    targets: object,
) -> tuple[object, float, int] | None:
    """Single target with the highest Jaro similarity."""
    return _first_or_none(jaro_top_k(query, targets, 1))


def jaro_winkler_best(
    query: object,
    targets: object,
    *,
    prefix_weight: float = 0.1,
    prefix_threshold: float = 0.7,
    prefix_cap: int = 4,
) -> tuple[object, float, int] | None:
    """Single target with the highest Jaro-Winkler similarity."""
    return _first_or_none(
        jaro_winkler_top_k(
            query,
            targets,
            k=1,
            prefix_weight=prefix_weight,
            prefix_threshold=prefix_threshold,
            prefix_cap=prefix_cap,
        )
    )


def smith_waterman_best(
    query: object,
    targets: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> tuple[object, int, int] | None:
    """Single target with the highest Smith-Waterman score."""
    return _first_or_none(
        smith_waterman_top_k(
            query,
            targets,
            k=1,
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
            width=width,
        )
    )


def extract_best(
    query: object,
    targets: object,
    *,
    scorer: "Scorer",
) -> tuple[object, int | float, int] | None:
    """Single best target by ``scorer`` (a ``Scorer`` enum value)."""
    return _first_or_none(extract(query, targets, scorer=scorer, k=1))


# cdist: all-pairs distance / similarity matrix. Everything down to
# the per-row work happens inside C++ — the only Python boundary
# crossings are the initial call and the optional tqdm progress
# callback. Symmetric inputs (queries is targets) compute only the
# upper triangle; the rest is mirrored.
#
# The scorer can be:
#   * A ``Scorer`` enum value (e.g. ``Scorer.LEVENSHTEIN``).
#   * Any of the top-level scoring functions in this module
#     (e.g. ``stride_align.levenshtein_scores``,
#     ``stride_align.jaro_similarities``). The dispatch happens in
#     C++ via a PyObject* -> Scorer table populated at import time.


def cdist(
    queries: object,
    targets: object,
    *,
    scorer: "Scorer | object",
    tqdm: object = None,
    cpu_count: int = 0,
    prefix_weight: float = 0.1,
    prefix_threshold: float = 0.7,
    prefix_cap: int = 4,
) -> np.ndarray:
    """All-pairs distance / similarity matrix.

    ``queries`` and ``targets`` are sequences of strings/bytes. The
    return is a 2-D ``ndarray`` of shape ``(len(queries), len(targets))``
    with ``int64`` cells for distance scorers and ``float64`` for
    similarity scorers.

    ``scorer`` accepts a ``Scorer`` enum value or any of the top-level
    scoring functions in this module (``stride_align.levenshtein_scores``
    etc.) — the dispatch table is registered at import time.

    ``tqdm`` is an optional callable that constructs a ``tqdm``-style
    progress bar. cdist calls ``tqdm(total=N)`` with an estimated work
    total (in length-product units), then ``bar.update(n)`` after each
    query row with ``n = sum_j q_len * t_len`` for that row. Updates
    are dispatched from the main thread; workers never touch the bar.
    For symmetric inputs (``queries is targets``), rows shrink as
    ``i`` grows; the cost-weighted updates make the bar advance
    smoothly in wall-clock time.

    ``cpu_count`` controls the worker thread count. ``0`` (default)
    means ``os.cpu_count()``. ``1`` forces single-threaded mode. The
    GIL is released for the SIMD compute either way.

    ``prefix_weight`` / ``prefix_threshold`` / ``prefix_cap`` are
    Jaro-Winkler hyperparameters used only when the scorer is
    Jaro-Winkler.

    The input lists are tuple-snapshotted on entry, so it is safe for
    another Python thread to mutate the original lists while cdist is
    running.
    """
    resolved_cpu_count = cpu_count
    if resolved_cpu_count <= 0:
        resolved_cpu_count = os.cpu_count() or 1
    return _LEVENSHTEIN_BACKEND.cdist(
        queries, targets,
        scorer=scorer,
        tqdm=tqdm,
        cpu_count=int(resolved_cpu_count),
        prefix_weight=prefix_weight,
        prefix_threshold=prefix_threshold,
        prefix_cap=prefix_cap,
    )


# Register each top-level scoring function with the C++ scorer table
# so users can pass `cdist(..., scorer=stride_align.levenshtein_scores)`
# without going through the Scorer enum. The C++ side already
# registered its own bound functions in the same table; this loop
# adds the Python-layer wrappers as aliases.
def _register_top_level_scorers() -> None:
    pairs = [
        (levenshtein_scores, Scorer.LEVENSHTEIN),
        (levenshtein_normalized_scores, Scorer.LEVENSHTEIN_NORMALIZED),
        (damerau_levenshtein_scores, Scorer.DAMERAU_LEVENSHTEIN),
        (damerau_levenshtein_normalized_scores, Scorer.DAMERAU_LEVENSHTEIN_NORMALIZED),
        (hamming_scores, Scorer.HAMMING),
        (hamming_normalized_scores, Scorer.HAMMING_NORMALIZED),
        (jaro_similarities, Scorer.JARO),
        (jaro_winkler_similarities, Scorer.JARO_WINKLER),
    ]
    register = getattr(_LEVENSHTEIN_BACKEND, "_register_scorer", None)
    if register is None:
        return
    for fn, s in pairs:
        register(fn, int(s))


_register_top_level_scorers()


__all__ = [
    "AlignmentPath",
    "AlignmentResult",
    "BackendKind",
    "BackendRecord",
    "Scorer",
    "Scores",
    "available_backends",
    "backend_is_available",
    "cdist",
    "damerau_levenshtein_best",
    "damerau_levenshtein_normalized_best",
    "damerau_levenshtein_normalized_score",
    "damerau_levenshtein_normalized_scores",
    "damerau_levenshtein_normalized_top_k",
    "damerau_levenshtein_score",
    "damerau_levenshtein_scores",
    "damerau_levenshtein_top_k",
    "detect_best_backend",
    "extract",
    "extract_best",
    "hamming_best",
    "hamming_normalized_best",
    "hamming_normalized_score",
    "hamming_normalized_scores",
    "hamming_normalized_top_k",
    "hamming_score",
    "hamming_scores",
    "hamming_top_k",
    "jaro_best",
    "jaro_similarities",
    "jaro_similarity",
    "jaro_top_k",
    "jaro_winkler_best",
    "jaro_winkler_similarities",
    "jaro_winkler_similarity",
    "jaro_winkler_top_k",
    "levenshtein_best",
    "levenshtein_normalized_best",
    "levenshtein_normalized_score",
    "levenshtein_normalized_scores",
    "levenshtein_normalized_top_k",
    "levenshtein_score",
    "levenshtein_scores",
    "levenshtein_top_k",
    "needleman_wunsch_cigar",
    "needleman_wunsch_normalized_score",
    "needleman_wunsch_normalized_scores",
    "needleman_wunsch_path",
    "needleman_wunsch_path_info",
    "needleman_wunsch_score",
    "needleman_wunsch_scores",
    "needleman_wunsch_trace_cigar",
    "needleman_wunsch_trade_cigar",
    "smith_waterman_cigar",
    "smith_waterman_farrar_normalized_score",
    "smith_waterman_farrar_normalized_scores",
    "smith_waterman_farrar_score",
    "smith_waterman_farrar_scores",
    "smith_waterman_normalized_score",
    "smith_waterman_normalized_scores",
    "smith_waterman_path",
    "smith_waterman_path_info",
    "smith_waterman_score",
    "smith_waterman_best",
    "smith_waterman_scores",
    "smith_waterman_top_k",
    "smith_waterman_trace_cigar",
    "smith_waterman_trade_cigar",
]
