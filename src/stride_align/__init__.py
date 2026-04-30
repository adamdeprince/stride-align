"""Python bootstrap and dynamic dispatcher for alignment backends."""

from __future__ import annotations

import importlib
import warnings
from types import ModuleType
from typing import Any

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
        case BackendKind.LINUX_AARCH64_ASIMD:
            return "stride_align._asimd"
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
            message=r"nanobind: type 'Alignment(Result|Path)' was already registered!",
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
    BackendKind.LINUX_AARCH64_ASIMD,
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
    BackendKind.LINUX_AARCH64_ASIMD,
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
    BackendKind.LINUX_AARCH64_ASIMD,
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

    if variant != "sw-farrar-score" or cells == 0:
        return _GENERIC_BACKEND

    # Current benchmarks show the anti-diagonal SIMD kernels lose to generic for
    # non-Farrar work. Use SIMD only for the striped Farrar score path, where
    # width and setup overhead determine whether narrow or wide vectors win.
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


__all__ = [
    "AlignmentPath",
    "AlignmentResult",
    "BackendKind",
    "BackendRecord",
    "available_backends",
    "backend_is_available",
    "detect_best_backend",
    "needleman_wunsch_path",
    "needleman_wunsch_path_info",
    "needleman_wunsch_score",
    "smith_waterman_farrar_score",
    "smith_waterman_path",
    "smith_waterman_path_info",
    "smith_waterman_score",
]
