"""Python fallback shim for the native CPU-detection module."""

from __future__ import annotations

from ._fallback import (
    BackendKind,
    BackendRecord,
    available_backends,
    backend_is_available,
    detect_best_backend,
)

__all__ = [
    "BackendKind",
    "BackendRecord",
    "available_backends",
    "backend_is_available",
    "detect_best_backend",
]
