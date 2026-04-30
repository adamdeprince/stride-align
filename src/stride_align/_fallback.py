"""Python fallback for backend detection when the native module is unavailable."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class BackendKind(Enum):
    GENERIC = auto()
    SWAR = auto()
    X86_SSE41 = auto()
    X86_AVX2 = auto()
    X86_AVX512BWVL = auto()
    X86_AVX10_256 = auto()
    X86_AVX10_512 = auto()
    LINUX_AARCH64_ASIMD = auto()
    LINUX_AARCH64_NEON = auto()
    LINUX_AARCH64_SVE = auto()
    LINUX_AARCH64_SVE2 = auto()
    MACOS_ARM64_NEON = auto()
    LINUX_LOONGARCH64_LSX = auto()
    LINUX_LOONGARCH64_LASX = auto()
    LINUX_POWERPC64_VSX = auto()
    LINUX_RISCV64_RVV = auto()


@dataclass(frozen=True, slots=True)
class BackendRecord:
    kind: BackendKind
    name: str
    compiled: bool
    available: bool


def detect_best_backend() -> BackendKind:
    return BackendKind.GENERIC


def backend_is_available(kind: BackendKind) -> bool:
    return kind is BackendKind.GENERIC


def available_backends() -> list[BackendRecord]:
    return [
        BackendRecord(
            kind=kind,
            name=kind.name.lower(),
            compiled=kind is BackendKind.GENERIC,
            available=kind is BackendKind.GENERIC,
        )
        for kind in BackendKind
    ]
