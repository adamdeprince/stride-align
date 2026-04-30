"""Python bootstrap for processor-specialized alignment backends."""

from __future__ import annotations

from ._cpu import (
    BackendKind,
    BackendRecord,
    available_backends,
    backend_is_available,
    detect_best_backend,
)


def _import_backend(kind: BackendKind):
    match kind:
        case BackendKind.GENERIC:
            from ._generic import (
                AlignmentPath,
                AlignmentResult,
                needleman_wunsch_path,
                needleman_wunsch_path_info,
                needleman_wunsch_score,
                smith_waterman_farrar_score,
                smith_waterman_path,
                smith_waterman_path_info,
                smith_waterman_score,
            )
        case BackendKind.X86_SSE41:
            from ._sse41 import (
                AlignmentPath,
                AlignmentResult,
                needleman_wunsch_path,
                needleman_wunsch_path_info,
                needleman_wunsch_score,
                smith_waterman_farrar_score,
                smith_waterman_path,
                smith_waterman_path_info,
                smith_waterman_score,
            )
        case BackendKind.X86_AVX2:
            from ._avx2 import (
                AlignmentPath,
                AlignmentResult,
                needleman_wunsch_path,
                needleman_wunsch_path_info,
                needleman_wunsch_score,
                smith_waterman_farrar_score,
                smith_waterman_path,
                smith_waterman_path_info,
                smith_waterman_score,
            )
        case BackendKind.X86_AVX512BWVL:
            from ._avx512bwvl import (
                AlignmentPath,
                AlignmentResult,
                needleman_wunsch_path,
                needleman_wunsch_path_info,
                needleman_wunsch_score,
                smith_waterman_farrar_score,
                smith_waterman_path,
                smith_waterman_path_info,
                smith_waterman_score,
            )
        case BackendKind.X86_AVX10_256:
            from ._avx10_256 import (
                AlignmentPath,
                AlignmentResult,
                needleman_wunsch_path,
                needleman_wunsch_path_info,
                needleman_wunsch_score,
                smith_waterman_farrar_score,
                smith_waterman_path,
                smith_waterman_path_info,
                smith_waterman_score,
            )
        case BackendKind.X86_AVX10_512:
            from ._avx10_512 import (
                AlignmentPath,
                AlignmentResult,
                needleman_wunsch_path,
                needleman_wunsch_path_info,
                needleman_wunsch_score,
                smith_waterman_farrar_score,
                smith_waterman_path,
                smith_waterman_path_info,
                smith_waterman_score,
            )
        case BackendKind.LINUX_AARCH64_ASIMD:
            from ._asimd import (
                AlignmentPath,
                AlignmentResult,
                needleman_wunsch_path,
                needleman_wunsch_path_info,
                needleman_wunsch_score,
                smith_waterman_farrar_score,
                smith_waterman_path,
                smith_waterman_path_info,
                smith_waterman_score,
            )
        case BackendKind.LINUX_AARCH64_NEON:
            from ._neon import (
                AlignmentPath,
                AlignmentResult,
                needleman_wunsch_path,
                needleman_wunsch_path_info,
                needleman_wunsch_score,
                smith_waterman_farrar_score,
                smith_waterman_path,
                smith_waterman_path_info,
                smith_waterman_score,
            )
        case BackendKind.LINUX_AARCH64_SVE:
            from ._sve import (
                AlignmentPath,
                AlignmentResult,
                needleman_wunsch_path,
                needleman_wunsch_path_info,
                needleman_wunsch_score,
                smith_waterman_farrar_score,
                smith_waterman_path,
                smith_waterman_path_info,
                smith_waterman_score,
            )
        case BackendKind.LINUX_AARCH64_SVE2:
            from ._sve2 import (
                AlignmentPath,
                AlignmentResult,
                needleman_wunsch_path,
                needleman_wunsch_path_info,
                needleman_wunsch_score,
                smith_waterman_farrar_score,
                smith_waterman_path,
                smith_waterman_path_info,
                smith_waterman_score,
            )
        case BackendKind.MACOS_ARM64_NEON:
            from ._macos_arm64_neon import (
                AlignmentPath,
                AlignmentResult,
                needleman_wunsch_path,
                needleman_wunsch_path_info,
                needleman_wunsch_score,
                smith_waterman_farrar_score,
                smith_waterman_path,
                smith_waterman_path_info,
                smith_waterman_score,
            )
        case BackendKind.LINUX_LOONGARCH64_LSX:
            from ._lsx import (
                AlignmentPath,
                AlignmentResult,
                needleman_wunsch_path,
                needleman_wunsch_path_info,
                needleman_wunsch_score,
                smith_waterman_farrar_score,
                smith_waterman_path,
                smith_waterman_path_info,
                smith_waterman_score,
            )
        case BackendKind.LINUX_LOONGARCH64_LASX:
            from ._lasx import (
                AlignmentPath,
                AlignmentResult,
                needleman_wunsch_path,
                needleman_wunsch_path_info,
                needleman_wunsch_score,
                smith_waterman_farrar_score,
                smith_waterman_path,
                smith_waterman_path_info,
                smith_waterman_score,
            )
        case BackendKind.LINUX_POWERPC64_VSX:
            from ._vsx import (
                AlignmentPath,
                AlignmentResult,
                needleman_wunsch_path,
                needleman_wunsch_path_info,
                needleman_wunsch_score,
                smith_waterman_farrar_score,
                smith_waterman_path,
                smith_waterman_path_info,
                smith_waterman_score,
            )
        case BackendKind.LINUX_RISCV64_RVV:
            from ._rvv import (
                AlignmentPath,
                AlignmentResult,
                needleman_wunsch_path,
                needleman_wunsch_path_info,
                needleman_wunsch_score,
                smith_waterman_farrar_score,
                smith_waterman_path,
                smith_waterman_path_info,
                smith_waterman_score,
            )
        case _:
            from ._generic import (
                AlignmentPath,
                AlignmentResult,
                needleman_wunsch_path,
                needleman_wunsch_path_info,
                needleman_wunsch_score,
                smith_waterman_farrar_score,
                smith_waterman_path,
                smith_waterman_path_info,
                smith_waterman_score,
            )

    return (
        AlignmentPath,
        AlignmentResult,
        smith_waterman_score,
        smith_waterman_path,
        smith_waterman_path_info,
        smith_waterman_farrar_score,
        needleman_wunsch_score,
        needleman_wunsch_path,
        needleman_wunsch_path_info,
    )


def _bind_backend_api() -> tuple[
    type,
    type,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
]:
    for kind in (detect_best_backend(), BackendKind.GENERIC):
        try:
            return _import_backend(kind)
        except ImportError:
            continue

    from ._pybackend import (
        AlignmentPath,
        AlignmentResult,
        needleman_wunsch_path,
        needleman_wunsch_path_info,
        needleman_wunsch_score,
        smith_waterman_farrar_score,
        smith_waterman_path,
        smith_waterman_path_info,
        smith_waterman_score,
    )

    return (
        AlignmentPath,
        AlignmentResult,
        smith_waterman_score,
        smith_waterman_path,
        smith_waterman_path_info,
        smith_waterman_farrar_score,
        needleman_wunsch_score,
        needleman_wunsch_path,
        needleman_wunsch_path_info,
    )


(
    AlignmentPath,
    AlignmentResult,
    smith_waterman_score,
    smith_waterman_path,
    smith_waterman_path_info,
    smith_waterman_farrar_score,
    needleman_wunsch_score,
    needleman_wunsch_path,
    needleman_wunsch_path_info,
) = _bind_backend_api()


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
