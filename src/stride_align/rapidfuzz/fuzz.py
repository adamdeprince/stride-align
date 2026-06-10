"""``rapidfuzz.fuzz`` shim — token-ratio family scaled to ``[0, 100]``.

Direct re-exports of the C++ ``_shim_fuzz_*`` bindings: each one
handles the rapidfuzz contract (optional ``processor`` callable
applied to both inputs, then the kernel, then ``× 100``, then the
``score_cutoff`` clamp) entirely in C++ so the Python boundary is
crossed exactly once per call. See ``src/cpp/fuzz_shim_dispatch.hpp``
for the implementations.
"""

from __future__ import annotations

from stride_align import _LEVENSHTEIN_BACKEND as _backend

ratio                     = _backend._shim_fuzz_ratio
partial_ratio             = _backend._shim_fuzz_partial_ratio
token_sort_ratio          = _backend._shim_fuzz_token_sort_ratio
token_set_ratio           = _backend._shim_fuzz_token_set_ratio
partial_token_sort_ratio  = _backend._shim_fuzz_partial_token_sort_ratio
partial_token_set_ratio   = _backend._shim_fuzz_partial_token_set_ratio
token_ratio               = _backend._shim_fuzz_token_ratio
partial_token_ratio       = _backend._shim_fuzz_partial_token_ratio
WRatio                    = _backend._shim_fuzz_WRatio  # noqa: N816
QRatio                    = _backend._shim_fuzz_QRatio  # noqa: N816

__all__ = [
    "ratio",
    "partial_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "partial_token_sort_ratio",
    "partial_token_set_ratio",
    "token_ratio",
    "partial_token_ratio",
    "WRatio",
    "QRatio",
]
