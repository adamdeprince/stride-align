"""Python bootstrap for processor-specialized alignment backends."""

from __future__ import annotations

from importlib import import_module, machinery, util
from pathlib import Path
from types import ModuleType
import sys

from ._cpu import BackendKind, BackendRecord, available_backends, backend_is_available, detect_best_backend

_BACKEND_MODULES = {
    BackendKind.GENERIC: "stride_align._generic",
    BackendKind.X86_SSE41: "stride_align._sse41",
    BackendKind.X86_AVX2: "stride_align._avx2",
    BackendKind.X86_AVX512BWVL: "stride_align._avx512bwvl",
    BackendKind.X86_AVX10_256: "stride_align._avx10_256",
    BackendKind.X86_AVX10_512: "stride_align._avx10_512",
    BackendKind.LINUX_AARCH64_ASIMD: "stride_align._asimd",
    BackendKind.LINUX_AARCH64_NEON: "stride_align._neon",
    BackendKind.LINUX_AARCH64_SVE: "stride_align._sve",
    BackendKind.LINUX_AARCH64_SVE2: "stride_align._sve2",
    BackendKind.MACOS_ARM64_NEON: "stride_align._macos_arm64_neon",
    BackendKind.LINUX_LOONGARCH64_LSX: "stride_align._lsx",
    BackendKind.LINUX_LOONGARCH64_LASX: "stride_align._lasx",
    BackendKind.LINUX_POWERPC64_VSX: "stride_align._vsx",
    BackendKind.LINUX_RISCV64_RVV: "stride_align._rvv",
}

_backend_module: ModuleType | None = None


def _load_local_extension(module_name: str) -> ModuleType | None:
    package_dir = Path(__file__).resolve().parent
    module_basename = module_name.rsplit(".", 1)[-1]

    for suffix in machinery.EXTENSION_SUFFIXES:
        candidate = package_dir / f"{module_basename}{suffix}"
        if not candidate.exists():
            continue

        spec = util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            continue

        module = util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    return None


def _load_backend_module() -> ModuleType:
    global _backend_module

    if _backend_module is not None:
        return _backend_module

    backend_kind = detect_best_backend()
    requested_module = _BACKEND_MODULES.get(backend_kind, "stride_align._generic")

    for module_name in (requested_module, "stride_align._generic"):
        try:
            _backend_module = _load_local_extension(module_name) or import_module(module_name)
            return _backend_module
        except ImportError:
            continue

    _backend_module = import_module("stride_align._pybackend")
    return _backend_module


def __getattr__(name: str):
    if name == "AlignmentResult":
        return getattr(_load_backend_module(), name)
    raise AttributeError(name)


def smith_waterman_score(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    width: int | None = None,
) -> int:
    return _load_backend_module().smith_waterman_score(
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        width=0 if width is None else width,
    )


def smith_waterman_path(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    width: int | None = None,
):
    return _load_backend_module().smith_waterman_path(
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        width=0 if width is None else width,
    )


def needleman_wunsch_score(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    width: int | None = None,
) -> int:
    return _load_backend_module().needleman_wunsch_score(
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        width=0 if width is None else width,
    )


def needleman_wunsch_path(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    width: int | None = None,
):
    return _load_backend_module().needleman_wunsch_path(
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        width=0 if width is None else width,
    )


__all__ = [
    "AlignmentResult",
    "BackendKind",
    "BackendRecord",
    "available_backends",
    "backend_is_available",
    "detect_best_backend",
    "needleman_wunsch_path",
    "needleman_wunsch_score",
    "smith_waterman_path",
    "smith_waterman_score",
]
