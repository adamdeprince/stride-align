"""Python bootstrap and dynamic dispatcher for alignment backends."""

from __future__ import annotations

import enum
import importlib
import os
import warnings
from types import ModuleType
from collections.abc import Iterator
from typing import Any

import numpy as np

from ._cpu import (
    BackendKind,
    BackendRecord,
    available_backends,
    backend_is_available,
    detect_best_backend,
)


_BACKEND_MODULE_NAMES = {
    BackendKind.GENERIC: "stride_align._generic",
    BackendKind.SWAR: "stride_align._swar",
    BackendKind.X86_SSE41: "stride_align._sse41",
    BackendKind.X86_AVX2: "stride_align._avx2",
    BackendKind.X86_AVX512BWVL: "stride_align._avx512bwvl",
    BackendKind.X86_AVX10_256: "stride_align._avx10_256",
    BackendKind.X86_AVX10_512: "stride_align._avx10_512",
    BackendKind.LINUX_AARCH64_NEON: "stride_align._neon",
    BackendKind.LINUX_AARCH64_SVE: "stride_align._sve",
    BackendKind.LINUX_AARCH64_SVE2: "stride_align._sve2",
    BackendKind.MACOS_ARM64_NEON: "stride_align._macos_arm64_neon",
    BackendKind.LINUX_LOONGARCH64_LSX: "stride_align._lsx",
    BackendKind.LINUX_LOONGARCH64_LASX: "stride_align._lasx",
    BackendKind.LINUX_POWERPC64_VSX: "stride_align._vsx",
    BackendKind.LINUX_RISCV64_RVV: "stride_align._rvv",
}


def _backend_module_name(kind: BackendKind) -> str:
    return _BACKEND_MODULE_NAMES.get(kind, "stride_align._generic")


def _import_module_suppressing_duplicate_type_warnings(module_name: str) -> ModuleType:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"nanobind: type "
                r"'(Alignment(Result|Path)|_PreparedAffineCigar|_PreparedScoreBatch|"
                r"_PreparedAffineScoreBatch|_ThresholdIterator)' was already registered!"
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

# Wide-token inputs (ndarray itemsize >= 2, non-UCS-1 unicode with large
# alphabets) route through the Phase B wide Farrar kernels, which only
# the fixed-kernel SIMD backends implement. SWAR is pure C scalar and
# also lacks wide wiring, so it's excluded too. SVE/SVE2/RVV use the
# scalable kernel and don't have wide_* templates yet.
_FIXED_KERNEL_WIDE_PRIORITY = (
    BackendKind.X86_AVX10_512,
    BackendKind.X86_AVX512BWVL,
    BackendKind.X86_AVX10_256,
    BackendKind.X86_AVX2,
    BackendKind.X86_SSE41,
    BackendKind.LINUX_AARCH64_NEON,
    BackendKind.MACOS_ARM64_NEON,
    BackendKind.LINUX_LOONGARCH64_LASX,
    BackendKind.LINUX_LOONGARCH64_LSX,
    BackendKind.LINUX_POWERPC64_VSX,
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


def _input_needs_wide_tokens(query: object, target: object) -> bool:
    """True if the inputs need a fixed-kernel SIMD backend rather than
    SWAR / scalable-kernel (SVE/SVE2/RVV) / generic. Two triggers:

    1. **Any ndarray** — SWAR and the scalable-kernel backends reject
       ndarrays in their `prepare_alignment` path, and generic falls
       back to the pure-Python loop; only the fixed-kernel SIMD
       backends accept ndarrays end-to-end (including int8/uint8 via
       memcpy and the wider dtypes via the Phase B wide kernels).
    2. **Potentially wide unicode** — strings with any codepoint above
       U+00FF could exceed the 256-distinct cap, which the narrow
       byte tokeniser rejects. We use a bounded scan and fall back to
       a conservative "yes" past 256 chars."""
    for obj in (query, target):
        if hasattr(obj, "dtype") and hasattr(obj, "shape"):
            return True
        if isinstance(obj, str) and obj and not obj.isascii():
            for index, ch in enumerate(obj):
                if index >= 256:
                    return True
                if ord(ch) > 0xFF:
                    return True
    return False


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

    # Wide-token inputs (ndarray itemsize >= 2, large-alphabet unicode)
    # need a fixed-kernel SIMD backend regardless of cells/score_width —
    # SWAR and scalable-kernel backends (SVE/SVE2/RVV) lack wide_*
    # templates and would raise on >256 distinct symbols. Applies to all
    # score-only variants (linear, affine, farrar-score).
    if variant in {"sw-score", "nw-score", "sw-farrar-score"} and _input_needs_wide_tokens(
        query, target
    ):
        return _first_available(_FIXED_KERNEL_WIDE_PRIORITY) or _GENERIC_BACKEND

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


_UNSET: Any = object()


# Sentinel kept only for tests that exercise the mutex-check helper
# directly. The public match_score / mismatch_score defaults are now
# real ints (2 / -1) so the no-matrix hot path doesn't pay for an
# `is _UNSET` resolution per call. The matrix branch uses
# `_matrix_kwargs_clean` for diagnostics, which compares against the
# documented defaults; this means the user can't pass an
# explicit-but-equal-to-default `match_score=2` together with
# `matrix=`, but that's a normal Python convention — defaults are
# indistinguishable from explicit values that happen to equal them.
_MATCH_DEFAULT = 2
_MISMATCH_DEFAULT = -1


def _matrix_kwargs_clean(
    *,
    match_score: Any,
    mismatch_score: Any,
    width: int | None,
) -> None:
    """Validate the kwargs that are NOT supplied alongside matrix=.

    Affine gap kwargs (gap_open_score / gap_extend_score) are allowed in
    matrix mode and resolved separately by the caller; this guard only
    rejects the mutually-exclusive match_score / mismatch_score path and
    the width override.
    """
    if match_score != _MATCH_DEFAULT or mismatch_score != _MISMATCH_DEFAULT:
        raise TypeError(
            "matrix= is mutually exclusive with match_score=/mismatch_score=; "
            "pass either a SubstitutionMatrix or scalar match/mismatch, not both."
        )
    if width is not None:
        raise NotImplementedError(
            "matrix-mode alignment does not yet honour width=; cell width is "
            "picked automatically from the matrix score bound."
        )


def _matrix_gap_route(
    *,
    gap_score: int,
    gap_open_score: int | None,
    gap_extend_score: int | None,
) -> tuple[bool, int, int]:
    """Return (is_affine, resolved_open, resolved_extend).

    When gap_open_score and gap_extend_score are both None — or both
    explicitly equal — we route through the linear-gap kernel. Otherwise
    the affine kernel.
    """
    open_score = gap_score if gap_open_score is None else gap_open_score
    extend_score = open_score if gap_extend_score is None else gap_extend_score
    is_affine = open_score != extend_score
    return is_affine, int(open_score), int(extend_score)


def _reject_wide_ndarray_for_matrix(*inputs: object) -> None:
    """Matrix-mode kernels read each input element as a uint8 index into
    the substitution matrix's alphabet. A wider-dtype ndarray (int16,
    int32, float64, etc.) would silently misinterpret its multi-byte
    representation as multiple indices, producing wrong scores. Catch
    that here with a clear TypeError; the user can pre-cast to uint8 if
    their values fit in [0, 255]."""
    for obj in inputs:
        if obj is None:
            continue
        if hasattr(obj, "dtype") and hasattr(obj, "shape"):
            itemsize = getattr(obj.dtype, "itemsize", 0)
            if itemsize > 1:
                raise TypeError(
                    "matrix= mode requires uint8/int8 ndarrays (each "
                    "value is an index into the matrix alphabet); got "
                    f"dtype with itemsize {itemsize}. Cast to np.uint8 "
                    "if your values fit in [0, 255]."
                )


def _validate_matrix(matrix: Any) -> Any:
    from .matrices import SubstitutionMatrix

    if not isinstance(matrix, SubstitutionMatrix):
        raise TypeError(
            f"matrix= must be a stride_align.matrices.SubstitutionMatrix "
            f"(got {type(matrix).__name__})"
        )
    return matrix


def _dispatch_matrix(
    function_name: str,
    query: object,
    target: object,
    matrix: Any,
    gap_score: int,
) -> int:
    matrix = _validate_matrix(matrix)
    _reject_wide_ndarray_for_matrix(query, target)
    q_bytes = matrix.encode(query) if isinstance(query, str) else bytes(query)
    t_bytes = matrix.encode(target) if isinstance(target, str) else bytes(target)
    # _LEVENSHTEIN_BACKEND is the best available SIMD backend; the C++
    # binding's `if constexpr (requires { ... })` check falls back to the
    # scalar generic kernel for backends that don't yet ship a matrix
    # path.
    return getattr(_LEVENSHTEIN_BACKEND, function_name)(
        q_bytes,
        t_bytes,
        matrix.matrix.tobytes(),
        matrix.stride,
        gap_score,
    )


def _dispatch_matrix_affine(
    function_name: str,
    query: object,
    target: object,
    matrix: Any,
    gap_open_score: int,
    gap_extend_score: int,
) -> int:
    matrix = _validate_matrix(matrix)
    _reject_wide_ndarray_for_matrix(query, target)
    q_bytes = matrix.encode(query) if isinstance(query, str) else bytes(query)
    t_bytes = matrix.encode(target) if isinstance(target, str) else bytes(target)
    return getattr(_LEVENSHTEIN_BACKEND, function_name)(
        q_bytes,
        t_bytes,
        matrix.matrix.tobytes(),
        matrix.stride,
        gap_open_score,
        gap_extend_score,
    )


def _dispatch_matrix_many(
    function_name: str,
    query: object,
    targets: object,
    matrix: Any,
    gap_score: int,
) -> np.ndarray:
    matrix = _validate_matrix(matrix)
    target_tuple = _materialize_targets(targets)
    if not target_tuple:
        return np.empty(0, dtype=np.int64)
    _reject_wide_ndarray_for_matrix(query, *target_tuple)
    q_bytes = matrix.encode(query) if isinstance(query, str) else bytes(query)
    encoded_targets = tuple(
        matrix.encode(t) if isinstance(t, str) else bytes(t) for t in target_tuple
    )
    scores = getattr(_LEVENSHTEIN_BACKEND, function_name)(
        q_bytes,
        encoded_targets,
        matrix.matrix.tobytes(),
        matrix.stride,
        gap_score,
    )
    return np.asarray(scores, dtype=np.int64)


def _dispatch_matrix_path_info(
    query: object,
    target: object,
    matrix: Any,
    *,
    local: bool,
    gap_score: int,
    gap_open_score: int | None,
    gap_extend_score: int | None,
) -> AlignmentPath:
    matrix = _validate_matrix(matrix)
    _reject_wide_ndarray_for_matrix(query, target)
    q_bytes = matrix.encode(query) if isinstance(query, str) else bytes(query)
    t_bytes = matrix.encode(target) if isinstance(target, str) else bytes(target)
    is_affine, open_s, extend_s = _matrix_gap_route(
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
    )
    if is_affine:
        fn = ("smith_waterman_affine_path_info_matrix"
              if local else "needleman_wunsch_affine_path_info_matrix")
        return getattr(_LEVENSHTEIN_BACKEND, fn)(
            q_bytes, t_bytes, matrix.matrix.tobytes(), matrix.stride,
            open_s, extend_s,
        )
    fn = ("smith_waterman_path_info_matrix"
          if local else "needleman_wunsch_path_info_matrix")
    return getattr(_LEVENSHTEIN_BACKEND, fn)(
        q_bytes, t_bytes, matrix.matrix.tobytes(), matrix.stride, open_s,
    )


def _materialize_alignment_strings(
    query: object,
    target: object,
    path: AlignmentPath,
) -> tuple[object, object]:
    """Rebuild aligned_query / aligned_target from the path operations.

    Operation alphabet (CIGAR-extended): `=` match, `X` mismatch, `D`
    gap-in-target (query advances), `I` gap-in-query (target advances).
    For str inputs the gap fill character is `-`; for bytes inputs it is
    the byte ``b"-"``.
    """
    is_str = isinstance(query, str) and isinstance(target, str)
    is_bytes = isinstance(query, (bytes, bytearray)) and isinstance(target, (bytes, bytearray))
    if not (is_str or is_bytes):
        # Sequence/object inputs are not yet materialised for matrix mode.
        return None, None

    if is_str:
        gap = "-"
        join = "".join
    else:
        gap = b"-"
        join = b"".join

    q_iter = iter(query[path.query_start:path.query_end])
    t_iter = iter(target[path.target_start:path.target_end])
    aligned_q: list = []
    aligned_t: list = []
    for op in path.operations:
        if op in ("=", "X"):
            aligned_q.append(next(q_iter))
            aligned_t.append(next(t_iter))
        elif op == "D":
            aligned_q.append(next(q_iter))
            aligned_t.append(gap)
        elif op == "I":
            aligned_q.append(gap)
            aligned_t.append(next(t_iter))
    if is_str:
        return join(aligned_q), join(aligned_t)
    # bytes/bytearray: items are ints under iteration; rebuild via bytes(...).
    return bytes(aligned_q), bytes(aligned_t)


def _dispatch_matrix_path(
    query: object,
    target: object,
    matrix: Any,
    *,
    local: bool,
    gap_score: int,
    gap_open_score: int | None,
    gap_extend_score: int | None,
) -> AlignmentResult:
    path = _dispatch_matrix_path_info(
        query, target, matrix,
        local=local,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
    )
    aligned_q, aligned_t = _materialize_alignment_strings(query, target, path)
    if aligned_q is None:
        aligned_q = "" if isinstance(query, str) else b""
    if aligned_t is None:
        aligned_t = "" if isinstance(target, str) else b""
    return AlignmentResult(
        score=path.score,
        query_start=path.query_start,
        query_end=path.query_end,
        target_start=path.target_start,
        target_end=path.target_end,
        aligned_query=aligned_q,
        aligned_target=aligned_t,
        operations=path.operations,
    )


def _dispatch_matrix_affine_many(
    function_name: str,
    query: object,
    targets: object,
    matrix: Any,
    gap_open_score: int,
    gap_extend_score: int,
) -> np.ndarray:
    matrix = _validate_matrix(matrix)
    target_tuple = _materialize_targets(targets)
    if not target_tuple:
        return np.empty(0, dtype=np.int64)
    _reject_wide_ndarray_for_matrix(query, *target_tuple)
    q_bytes = matrix.encode(query) if isinstance(query, str) else bytes(query)
    encoded_targets = tuple(
        matrix.encode(t) if isinstance(t, str) else bytes(t) for t in target_tuple
    )
    scores = getattr(_LEVENSHTEIN_BACKEND, function_name)(
        q_bytes,
        encoded_targets,
        matrix.matrix.tobytes(),
        matrix.stride,
        gap_open_score,
        gap_extend_score,
    )
    return np.asarray(scores, dtype=np.int64)


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

    # Batch-of-1 short-circuit. The C++ batch dispatch pays
    # list-to-vector conversion and a result-vector-to-ndarray wrap
    # for every call; for a single target that overhead dominates,
    # so route through the singular path instead. Same final result,
    # ~3x lower per-call latency on short-query workloads.
    if len(target_tuple) == 1:
        scalar_public_name = _SCORE_MANY_FUNCTIONS[canonical_variant][1]
        scalar_public = globals()[scalar_public_name]
        score = scalar_public(
            query,
            target_tuple[0],
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
            width=width,
        )
        return np.asarray([int(score)], dtype=np.int64)

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
    matrix: Any = None,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    if matrix is not None:
        _matrix_kwargs_clean(
            match_score=match_score,
            mismatch_score=mismatch_score,
            width=width,
        )
        is_affine, open_s, extend_s = _matrix_gap_route(
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
        )
        if is_affine:
            return _dispatch_matrix_affine_many(
                "smith_waterman_affine_scores_matrix",
                query, targets, matrix, open_s, extend_s,
            )
        return _dispatch_matrix_many(
            "smith_waterman_scores_matrix", query, targets, matrix, open_s,
        )
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
    matrix: Any = None,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    if matrix is not None:
        _matrix_kwargs_clean(
            match_score=match_score,
            mismatch_score=mismatch_score,
            width=width,
        )
        is_affine, open_s, extend_s = _matrix_gap_route(
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
        )
        if is_affine:
            return _dispatch_matrix_affine_many(
                "needleman_wunsch_affine_scores_matrix",
                query, targets, matrix, open_s, extend_s,
            )
        return _dispatch_matrix_many(
            "needleman_wunsch_scores_matrix", query, targets, matrix, open_s,
        )
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
    matrix: Any = None,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> int:
    if matrix is not None:
        _matrix_kwargs_clean(
            match_score=match_score,
            mismatch_score=mismatch_score,
            width=width,
        )
        is_affine, open_s, extend_s = _matrix_gap_route(
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
        )
        if is_affine:
            return _dispatch_matrix_affine(
                "smith_waterman_affine_score_matrix",
                query, target, matrix, open_s, extend_s,
            )
        return _dispatch_matrix(
            "smith_waterman_score_matrix", query, target, matrix, open_s,
        )
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
    matrix: Any = None,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> AlignmentResult:
    if matrix is not None:
        _matrix_kwargs_clean(
            match_score=match_score, mismatch_score=mismatch_score, width=width,
        )
        return _dispatch_matrix_path(
            query, target, matrix, local=True,
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
        )
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
    matrix: Any = None,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> AlignmentPath:
    if matrix is not None:
        _matrix_kwargs_clean(
            match_score=match_score, mismatch_score=mismatch_score, width=width,
        )
        return _dispatch_matrix_path_info(
            query, target, matrix, local=True,
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
        )
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
    matrix: Any = None,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> str:
    if matrix is not None:
        _matrix_kwargs_clean(
            match_score=match_score, mismatch_score=mismatch_score, width=width,
        )
        return _dispatch_matrix_path_info(
            query, target, matrix, local=True,
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
        ).cigar
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
    matrix: Any = None,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> int:
    if matrix is not None:
        _matrix_kwargs_clean(
            match_score=match_score,
            mismatch_score=mismatch_score,
            width=width,
        )
        is_affine, open_s, extend_s = _matrix_gap_route(
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
        )
        if is_affine:
            return _dispatch_matrix_affine(
                "needleman_wunsch_affine_score_matrix",
                query, target, matrix, open_s, extend_s,
            )
        return _dispatch_matrix(
            "needleman_wunsch_score_matrix", query, target, matrix, open_s,
        )
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
    matrix: Any = None,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> AlignmentResult:
    if matrix is not None:
        _matrix_kwargs_clean(
            match_score=match_score, mismatch_score=mismatch_score, width=width,
        )
        return _dispatch_matrix_path(
            query, target, matrix, local=False,
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
        )
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
    matrix: Any = None,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> AlignmentPath:
    if matrix is not None:
        _matrix_kwargs_clean(
            match_score=match_score, mismatch_score=mismatch_score, width=width,
        )
        return _dispatch_matrix_path_info(
            query, target, matrix, local=False,
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
        )
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
    matrix: Any = None,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> str:
    if matrix is not None:
        _matrix_kwargs_clean(
            match_score=match_score, mismatch_score=mismatch_score, width=width,
        )
        return _dispatch_matrix_path_info(
            query, target, matrix, local=False,
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
        ).cigar
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


# Scalar Levenshtein entrypoints — the binding returns Python int /
# float already, so the old ``int(...)`` / ``float(...)`` casts were
# no-ops and the wrappers were pure overhead. Docstrings live on the
# C++ binding side.
levenshtein_score = _LEVENSHTEIN_BACKEND.levenshtein_score
levenshtein_normalized_score = _LEVENSHTEIN_BACKEND.levenshtein_normalized_score


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
    if len(targets) == 1:
        # Batch-of-1 short-circuit: the singular kernel skips the
        # list-to-vector conversion and the vector-to-ndarray result
        # wrap that the batch entry pays per call.
        return np.asarray(
            [int(_LEVENSHTEIN_BACKEND.levenshtein_score(query, targets[0], score_cutoff))],
            dtype=np.int64,
        )
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
    if len(targets) == 1:
        return np.asarray(
            [float(_LEVENSHTEIN_BACKEND.levenshtein_normalized_score(
                query, targets[0], score_cutoff))],
            dtype=np.float64,
        )
    return _LEVENSHTEIN_BACKEND.levenshtein_normalized_scores(
        query, targets, score_cutoff
    )


class LevenshteinScorer:
    """Persistent scorer for one fixed query against a stream of targets.

    Caches a pre-built Myers pattern_match (PEQ) table for the query
    on construction; each ``.distance(target)`` call then skips the
    per-call PEQ rebuild — typically 20–35% of the kernel's wall
    clock on short queries — as long as both query and target are
    bytes or 1-byte (Latin-1) unicode. For wider tokens we fall back
    to the per-call dispatch so the API still works; the bytes path
    is the streaming-against-fixed-query hot path the class is named
    for.

    >>> scorer = LevenshteinScorer(b"god")
    >>> scorer.distance(b"good")
    1
    >>> scorer.distances([b"good", b"god", b"god!"]).tolist()
    [1, 0, 1]
    """

    __slots__ = ("_query", "_backend", "_prepared")

    def __init__(self, query: object) -> None:
        self._query = query
        # Cache the resolved backend so .distance() can hit it without
        # any attribute walk per call.
        self._backend = _LEVENSHTEIN_BACKEND
        # Try the bytes-only prepared C++ kernel. Returns None when
        # the query isn't byte-compatible (UCS-2/4 strings, sequences
        # of ints, ndarrays); in that case .distance() falls through
        # to the per-call dispatch.
        prepare = getattr(self._backend, "_prepare_levenshtein_score", None)
        self._prepared = prepare(query) if prepare is not None else None

    @property
    def query(self) -> object:
        return self._query

    @property
    def has_prepared(self) -> bool:
        """True if the bytes-fast-path PEQ was pre-built for the query."""
        return self._prepared is not None

    def distance(self, target: object, score_cutoff: int | None = None) -> int:
        """Levenshtein distance from the fixed query to ``target``."""
        if self._prepared is not None:
            score = self._backend._levenshtein_score_prepared(
                self._prepared, target, score_cutoff)
            if score is not None:
                return int(score)
            # target wasn't byte-compatible; fall through.
        return int(self._backend.levenshtein_score(self._query, target, score_cutoff))

    def normalized_distance(
        self, target: object, score_cutoff: int | None = None,
    ) -> float:
        """Normalized similarity in ``[0, 1]`` (1 = identical)."""
        return float(self._backend.levenshtein_normalized_score(
            self._query, target, score_cutoff))

    def distances(
        self, targets: object, score_cutoff: int | None = None,
    ) -> np.ndarray:
        """Distances from the fixed query to every target (``ndarray[int64]``)."""
        if isinstance(targets, (str, bytes)):
            raise TypeError(
                "targets must be an iterable of target sequences, not a single str/bytes"
            )
        if not isinstance(targets, (list, tuple)):
            targets = tuple(targets)
        if len(targets) == 1:
            return np.asarray(
                [self.distance(targets[0], score_cutoff)], dtype=np.int64,
            )
        return self._backend.levenshtein_scores(self._query, targets, score_cutoff)

    def normalized_distances(
        self, targets: object, score_cutoff: int | None = None,
    ) -> np.ndarray:
        """Normalized similarities to every target (``ndarray[float64]``)."""
        if isinstance(targets, (str, bytes)):
            raise TypeError(
                "targets must be an iterable of target sequences, not a single str/bytes"
            )
        if not isinstance(targets, (list, tuple)):
            targets = tuple(targets)
        if len(targets) == 1:
            return np.asarray(
                [float(self._backend.levenshtein_normalized_score(
                    self._query, targets[0], score_cutoff))],
                dtype=np.float64,
            )
        return self._backend.levenshtein_normalized_scores(
            self._query, targets, score_cutoff)


# Scalar OSA (damerau_levenshtein_*) and Hamming entrypoints —
# direct re-exports for the same reason as Levenshtein: the binding
# already returns the right Python scalar type and carries the
# docstring on the C++ side.
damerau_levenshtein_score = _LEVENSHTEIN_BACKEND.damerau_levenshtein_score
damerau_levenshtein_normalized_score = (
    _LEVENSHTEIN_BACKEND.damerau_levenshtein_normalized_score
)


def damerau_levenshtein_scores(query: object, targets: object) -> np.ndarray:
    """OSA distance from query to every target, returned as ndarray[int64]."""
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    if len(targets) == 1:
        return np.asarray(
            [int(_LEVENSHTEIN_BACKEND.damerau_levenshtein_score(query, targets[0]))],
            dtype=np.int64,
        )
    return _LEVENSHTEIN_BACKEND.damerau_levenshtein_scores(query, targets)


def damerau_levenshtein_normalized_scores(query: object, targets: object) -> np.ndarray:
    """Normalized OSA similarity per target, returned as ndarray[float64]."""
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    if len(targets) == 1:
        return np.asarray(
            [float(_LEVENSHTEIN_BACKEND.damerau_levenshtein_normalized_score(
                query, targets[0]))],
            dtype=np.float64,
        )
    return _LEVENSHTEIN_BACKEND.damerau_levenshtein_normalized_scores(query, targets)


# ---- Dynamic Time Warping ------------------------------------------------
#
# DTW lives on the existing _LEVENSHTEIN_BACKEND module for the same
# reason the byte-distance kernels do: it's one PYI symbol per backend
# and lives next to the other numeric / distance APIs. Phase C.1a is the
# scalar reference; phase C.1b plugs SIMD into the same C++ dispatch
# without touching this Python surface.

dtw = _LEVENSHTEIN_BACKEND.dtw


def dtw_distances(
    query: object,
    targets: object,
    *,
    window: int | float | None = None,
    distance: str | None = None,
) -> np.ndarray:
    """DTW distance from one query to every target, returned as ``ndarray[float64]``.

    Arguments mirror :func:`dtw`. ``targets`` is an iterable of
    ndarrays sharing dtype with ``query``. Empty queries or empty
    targets raise ``ValueError``.
    """
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target ndarrays, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    return _LEVENSHTEIN_BACKEND.dtw_distances(
        query, targets, window=window, distance=distance,
    )


hamming_score = _LEVENSHTEIN_BACKEND.hamming_score
hamming_normalized_score = _LEVENSHTEIN_BACKEND.hamming_normalized_score


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


# Indel distance — Levenshtein restricted to insertions/deletions, no
# substitutions. Equivalent to |a| + |b| - 2 * LCS(a, b). Useful when
# substitution cost should be modeled as two edits (delete + insert)
# rather than one. Normalized form divides by (|a| + |b|) so the
# similarity is always in [0, 1].


indel_score = _LEVENSHTEIN_BACKEND.indel_score
indel_normalized_score = _LEVENSHTEIN_BACKEND.indel_normalized_score


def indel_scores(query: object, targets: object) -> np.ndarray:
    """Indel distance from query to every target, returned as ndarray[int64]."""
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    return _LEVENSHTEIN_BACKEND.indel_scores(query, targets)


def indel_normalized_scores(query: object, targets: object) -> np.ndarray:
    """Indel similarity per target, returned as ndarray[float64]."""
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    return _LEVENSHTEIN_BACKEND.indel_normalized_scores(query, targets)


# Longest Common Subsequence / Substring. ``lcs_length`` is the
# subsequence form (characters need not be contiguous); the substring
# pair is the contiguous form. ``lcs_substring`` returns the substring
# itself (``str`` or ``bytes`` matching the input type).
lcs_length = _LEVENSHTEIN_BACKEND.lcs_length
lcs_substring_length = _LEVENSHTEIN_BACKEND.lcs_substring_length
lcs_substring = _LEVENSHTEIN_BACKEND.lcs_substring


# Ratcliff-Obershelp similarity — Python's ``difflib.SequenceMatcher``
# algorithm, with no autojunk heuristic so the result is the
# autojunk-off form (which matches the autojunk-on form on inputs
# without any character that appears very frequently).
ratcliff_obershelp_similarity = (
    _LEVENSHTEIN_BACKEND.ratcliff_obershelp_similarity
)


def ratcliff_obershelp_similarities(
    query: object, targets: object
) -> np.ndarray:
    """Ratcliff-Obershelp similarity per target as ndarray[float64]."""
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    return _LEVENSHTEIN_BACKEND.ratcliff_obershelp_similarities(
        query, targets,
    )


# N-gram set similarity (Phase D.1). Four metrics over character
# n-gram multisets — Jaccard, Sorensen-Dice, Cosine, Overlap — sharing
# the same multiset builder and one-pass comparison. ``n`` defaults to
# 2 (character bigrams).
jaccard = _LEVENSHTEIN_BACKEND.jaccard
dice = _LEVENSHTEIN_BACKEND.dice
cosine = _LEVENSHTEIN_BACKEND.cosine
overlap = _LEVENSHTEIN_BACKEND.overlap


def _ngram_similarities(backend_fn_name: str, query: object, targets: object,
                         *, n: int) -> np.ndarray:
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    return getattr(_LEVENSHTEIN_BACKEND, backend_fn_name)(query, targets, n=int(n))


def jaccard_similarities(
    query: object, targets: object, *, n: int = 2,
) -> np.ndarray:
    """Jaccard similarity per target as ndarray[float64]."""
    return _ngram_similarities("jaccard_similarities", query, targets, n=n)


def dice_similarities(
    query: object, targets: object, *, n: int = 2,
) -> np.ndarray:
    """Sorensen-Dice similarity per target as ndarray[float64]."""
    return _ngram_similarities("dice_similarities", query, targets, n=n)


def cosine_similarities(
    query: object, targets: object, *, n: int = 2,
) -> np.ndarray:
    """Cosine similarity per target as ndarray[float64]."""
    return _ngram_similarities("cosine_similarities", query, targets, n=n)


def overlap_similarities(
    query: object, targets: object, *, n: int = 2,
) -> np.ndarray:
    """Overlap coefficient per target as ndarray[float64]."""
    return _ngram_similarities("overlap_similarities", query, targets, n=n)


# Token-ratio family (Phase D.3) — rapidfuzz ``fuzz.*`` parity. Pure-
# Python composition over ``sa.indel_normalized_score`` (which is
# algebraically ``rapidfuzz.fuzz.ratio / 100``). Returned values are
# in [0, 1] — multiply by 100 if you need rapidfuzz's [0, 100] scale.
from . import _fuzz as _fuzz_mod  # noqa: E402

token_sort_ratio          = _fuzz_mod.token_sort_ratio
token_set_ratio           = _fuzz_mod.token_set_ratio
partial_ratio             = _fuzz_mod.partial_ratio
partial_token_sort_ratio  = _fuzz_mod.partial_token_sort_ratio
partial_token_set_ratio   = _fuzz_mod.partial_token_set_ratio
WRatio                    = _fuzz_mod.WRatio

# Monge-Elkan (Phase D.6) — directional multi-token hybrid for record
# linkage. ``inner=`` selects the per-token similarity from
# ``"jaro"``, ``"jaro_winkler"``, ``"levenshtein_ratio"``,
# ``"indel_ratio"``, or any callable.
from . import _monge_elkan as _monge_elkan_mod  # noqa: E402

monge_elkan = _monge_elkan_mod.monge_elkan


# True Damerau-Levenshtein — unrestricted form. Each character may
# participate in multiple edits, unlike the OSA variant we ship under
# ``damerau_levenshtein_*`` (which restricts each character to at most
# one). Almost always equal to OSA on real-world inputs; only differs
# when overlapping transpositions are the cheapest path. No bit-parallel
# kernel yet; scalar DP per pair.


true_damerau_levenshtein_score = (
    _LEVENSHTEIN_BACKEND.true_damerau_levenshtein_score
)
true_damerau_levenshtein_normalized_score = (
    _LEVENSHTEIN_BACKEND.true_damerau_levenshtein_normalized_score
)


def true_damerau_levenshtein_scores(
    query: object, targets: object
) -> np.ndarray:
    """True Damerau-Levenshtein distance from query to every target."""
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    return _LEVENSHTEIN_BACKEND.true_damerau_levenshtein_scores(query, targets)


def true_damerau_levenshtein_normalized_scores(
    query: object, targets: object
) -> np.ndarray:
    """True Damerau-Levenshtein similarity per target."""
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    if not isinstance(targets, (list, tuple)):
        targets = tuple(targets)
    return _LEVENSHTEIN_BACKEND.true_damerau_levenshtein_normalized_scores(
        query, targets
    )


# ---- Phonetic encoders -----------------------------------------------------
#
# These return an encoded string, not a distance. Compare two encoded
# strings for equality (or pass through a string-distance metric) to decide
# similarity. Non-ASCII codepoints are skipped — Soundex is an ASCII
# algorithm by design. Callers wanting accent folding pre-normalise with
# ``unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()``.


# Phonetic encoders bind their full docstrings on the C++ side and
# are re-exported here directly to skip a Python wrapper frame on
# every call — these are the kind of millions-of-calls-per-second
# fuzzy-match helpers where one extra frame matters.
soundex = _LEVENSHTEIN_BACKEND.soundex
soundex_equal = _LEVENSHTEIN_BACKEND.soundex_equal


class MetaphoneVariant(enum.IntEnum):
    """Selects which Metaphone rule family :func:`metaphone` follows."""

    PHILIPS = 0
    """Lawrence Philips' 1990 published spec (Apache Commons Codec branch). Default."""

    JELLYFISH = 1
    """Matches the popular jellyfish Python library's interpretation.

    Differs from the Philips spec on ``SCH`` (Philips: ``SK``; jellyfish: ``SX``)
    and ``GH`` at end of word (Philips: silent; jellyfish: ``KH``).
    """


# MetaphoneVariant / DoubleMetaphoneVariant are IntEnums; nanobind
# accepts them as ints directly (IntEnum IS-A int), so we re-export
# the bindings without a wrapper.
metaphone = _LEVENSHTEIN_BACKEND.metaphone
metaphone_equal = _LEVENSHTEIN_BACKEND.metaphone_equal
nysiis = _LEVENSHTEIN_BACKEND.nysiis
nysiis_equal = _LEVENSHTEIN_BACKEND.nysiis_equal
match_rating_codex = _LEVENSHTEIN_BACKEND.match_rating_codex
match_rating_compare = _LEVENSHTEIN_BACKEND.match_rating_compare
caverphone = _LEVENSHTEIN_BACKEND.caverphone
cologne_phonetic = _LEVENSHTEIN_BACKEND.cologne_phonetic
daitch_mokotoff = _LEVENSHTEIN_BACKEND.daitch_mokotoff

# Beider-Morse Phonetic Matching is compiled only into the ``_generic``
# backend to avoid duplicating the 280 KB rule set across every SIMD
# ``.so``. The wrapper in :mod:`stride_align._bmpm` lazy-loads the rule
# files via :mod:`importlib.resources` on the first call.
from . import _bmpm as _bmpm_mod  # noqa: E402

_bmpm_mod._set_backend(_BACKENDS[BackendKind.GENERIC])
beider_morse = _bmpm_mod.beider_morse
BmpmRuleType = _bmpm_mod.BmpmRuleType


class DoubleMetaphoneVariant(enum.IntEnum):
    """Selects which port :func:`double_metaphone` follows.

    The two values are not equal-weight algorithmic variations.
    :attr:`COMMONS` is the faithful Lawrence Philips port (Apache
    Commons Codec and the ``doublemetaphone`` PyPI package, which
    mirrors it). :attr:`PYTHON` reproduces a known bug in the
    ``metaphone`` PyPI package: a missing ``else`` in its
    ``process_g`` GH branch lets the previous character's emission
    leak into the next iteration, so names like "Hugh" and "High"
    come out ``HH`` instead of ``H``. Use :attr:`PYTHON` only when
    you need byte-for-byte agreement with that library.
    """

    COMMONS = 0
    """Apache Commons Codec semantics. Correct per the Philips spec. Default."""

    PYTHON = 1
    """``metaphone`` PyPI package bug-compat. ``"Hugh"`` -> ``("HH", "")``."""


double_metaphone = _LEVENSHTEIN_BACKEND.double_metaphone


# Jaro / Jaro-Winkler. Both are similarity in [0, 1]; there is no
# matching distance form. Defaults match rapidfuzz: prefix_weight=0.1,
# prefix_threshold=0.7 (no Winkler bonus below 0.7 base similarity),
# prefix_cap=4 characters of common prefix.
#
# The scalar pair-wise functions are re-exports of the C++ bindings —
# nanobind already returns Python floats, so the old ``float(...)``
# casts were no-ops, and the bindings carry their own docstrings.
# The batch variants keep a thin Python wrapper because they normalise
# ``targets`` into a list/tuple before the binding call (the C++ side
# expects a real sequence, not just any iterable).

jaro_similarity = _LEVENSHTEIN_BACKEND.jaro_similarity
jaro_winkler_similarity = _LEVENSHTEIN_BACKEND.jaro_winkler_similarity


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
    INDEL = 8
    INDEL_NORMALIZED = 9
    TRUE_DAMERAU_LEVENSHTEIN = 10
    TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED = 11


# Top-k re-exports. The C++ bindings now do the str/bytes rejection
# themselves (see ``reject_str_or_bytes_targets`` in preprocess.hpp)
# and PySequence_Fast materialises any iterable on the C++ side, so
# the old ``_coerce_targets`` Python helper was pure overhead.
levenshtein_top_k = _LEVENSHTEIN_BACKEND.levenshtein_top_k
levenshtein_normalized_top_k = _LEVENSHTEIN_BACKEND.levenshtein_normalized_top_k
damerau_levenshtein_top_k = _LEVENSHTEIN_BACKEND.damerau_levenshtein_top_k
damerau_levenshtein_normalized_top_k = (
    _LEVENSHTEIN_BACKEND.damerau_levenshtein_normalized_top_k
)
hamming_top_k = _LEVENSHTEIN_BACKEND.hamming_top_k
hamming_normalized_top_k = _LEVENSHTEIN_BACKEND.hamming_normalized_top_k
indel_top_k = _LEVENSHTEIN_BACKEND.indel_top_k
indel_normalized_top_k = _LEVENSHTEIN_BACKEND.indel_normalized_top_k
true_damerau_levenshtein_top_k = _LEVENSHTEIN_BACKEND.true_damerau_levenshtein_top_k
true_damerau_levenshtein_normalized_top_k = (
    _LEVENSHTEIN_BACKEND.true_damerau_levenshtein_normalized_top_k
)
jaro_top_k = _LEVENSHTEIN_BACKEND.jaro_top_k


jaro_winkler_top_k = _LEVENSHTEIN_BACKEND.jaro_winkler_top_k


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


# ``extract`` is a direct re-export — the C++ binding does str/bytes
# rejection itself, accepts any iterable via PySequence_Fast, and
# accepts the Scorer IntEnum directly (it's an int).
extract = _LEVENSHTEIN_BACKEND.extract


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


def indel_best(
    query: object,
    targets: object,
) -> tuple[object, int, int] | None:
    """Single target with the lowest Indel distance."""
    return _first_or_none(indel_top_k(query, targets, 1))


def indel_normalized_best(
    query: object,
    targets: object,
) -> tuple[object, float, int] | None:
    """Single target with the highest Indel similarity."""
    return _first_or_none(indel_normalized_top_k(query, targets, 1))


def true_damerau_levenshtein_best(
    query: object,
    targets: object,
) -> tuple[object, int, int] | None:
    """Single target with the lowest true Damerau-Levenshtein distance."""
    return _first_or_none(true_damerau_levenshtein_top_k(query, targets, 1))


def true_damerau_levenshtein_normalized_best(
    query: object,
    targets: object,
) -> tuple[object, float, int] | None:
    """Single target with the highest true Damerau-Levenshtein similarity."""
    return _first_or_none(
        true_damerau_levenshtein_normalized_top_k(query, targets, 1)
    )


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


_MATRIX_CDIST_LOCAL = ("sw", "smith_waterman", "smith-waterman", "local")
_MATRIX_CDIST_GLOBAL = ("nw", "needleman_wunsch", "needleman-wunsch", "global")


def _resolve_matrix_cdist_mode(scorer: Any) -> bool:
    """Return True for local (SW), False for global (NW).

    Accepts either a string mode label or one of the module-level
    score / scores functions for SW or NW.
    """
    if isinstance(scorer, str):
        label = scorer.lower()
        if label in _MATRIX_CDIST_LOCAL:
            return True
        if label in _MATRIX_CDIST_GLOBAL:
            return False
        raise ValueError(
            f"matrix-mode cdist scorer must be 'sw'/'nw' (or smith_waterman_score / "
            f"needleman_wunsch_score / *_scores); got {scorer!r}"
        )
    if scorer in (smith_waterman_score, smith_waterman_scores):
        return True
    if scorer in (needleman_wunsch_score, needleman_wunsch_scores):
        return False
    raise ValueError(
        "matrix-mode cdist scorer must be one of: 'sw', 'nw', "
        "smith_waterman_score, smith_waterman_scores, "
        "needleman_wunsch_score, needleman_wunsch_scores"
    )


def _cdist_matrix(
    queries: object,
    targets: object,
    *,
    matrix: Any,
    scorer: Any,
    gap_score: int,
    gap_open_score: int | None,
    gap_extend_score: int | None,
) -> np.ndarray:
    local = _resolve_matrix_cdist_mode(scorer)
    matrix = _validate_matrix(matrix)
    is_affine, open_s, extend_s = _matrix_gap_route(
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
    )
    query_tuple = _materialize_targets(queries)
    target_tuple = _materialize_targets(targets)
    if not query_tuple:
        return np.empty((0, len(target_tuple)), dtype=np.int64)
    if not target_tuple:
        return np.empty((len(query_tuple), 0), dtype=np.int64)

    if is_affine:
        batch_fn = "smith_waterman_affine_scores_matrix" if local \
            else "needleman_wunsch_affine_scores_matrix"
        dispatcher = _dispatch_matrix_affine_many
        result = np.empty((len(query_tuple), len(target_tuple)), dtype=np.int64)
        for i, q in enumerate(query_tuple):
            row = dispatcher(batch_fn, q, target_tuple, matrix, open_s, extend_s)
            result[i, :] = row
        return result

    batch_fn = "smith_waterman_scores_matrix" if local else "needleman_wunsch_scores_matrix"
    dispatcher = _dispatch_matrix_many
    result = np.empty((len(query_tuple), len(target_tuple)), dtype=np.int64)
    for i, q in enumerate(query_tuple):
        row = dispatcher(batch_fn, q, target_tuple, matrix, open_s)
        result[i, :] = row
    return result


def cdist(
    queries: object,
    targets: object,
    *,
    scorer: "Scorer | object" = None,
    matrix: Any = None,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
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
    if matrix is not None:
        return _cdist_matrix(
            queries, targets,
            matrix=matrix, scorer=scorer if scorer is not None else "sw",
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
        )
    if scorer is None:
        raise TypeError("cdist requires either scorer= or matrix=")
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


def cdist_above_threshold(
    queries: object,
    targets: object,
    *,
    scorer: "Scorer | object" = None,
    threshold: float,
    matrix: Any = None,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    tqdm: object = None,
    cpu_count: int = 0,
    prefix_weight: float = 0.1,
    prefix_threshold: float = 0.7,
    prefix_cap: int = 4,
):
    """Streaming filtered cdist: yields ``(score, query, target)`` for
    every pair whose normalized similarity is at least ``threshold``.

    Memory is O(``len(queries) + len(targets)``) regardless of how many
    pairs match — workers feed a bounded queue and the caller drains it
    lazily. The returned object is a Python iterator; iterate it with a
    for-loop, take a slice via ``itertools.islice``, or convert to a
    list (eager, beware of memory if matches are dense).

    Requires a normalized-similarity scorer (one whose values lie in
    ``[0, 1]``): ``Scorer.LEVENSHTEIN_NORMALIZED``,
    ``Scorer.DAMERAU_LEVENSHTEIN_NORMALIZED``,
    ``Scorer.HAMMING_NORMALIZED``, ``Scorer.JARO``, or
    ``Scorer.JARO_WINKLER`` (or the equivalent module-level scoring
    function).

    Like ``cdist``: takes a tuple-snapshot of the inputs at entry,
    spawns ``cpu_count`` workers (0 = ``os.cpu_count()``), releases the
    GIL for the compute, and dispatches optional tqdm updates from the
    main thread. Symmetric inputs (``queries is targets``) yield each
    matching pair in both orderings.

    Abandoning the iterator mid-loop (``break``) is safe — its
    destructor signals the workers to stop and joins them.

    Matrix mode (``matrix=``): the threshold is interpreted as a raw
    integer score (not a [0, 1] normalized similarity), and the call
    runs single-threaded in Python on top of the batch matrix kernel.
    """
    if matrix is not None:
        return _cdist_above_threshold_matrix(
            queries, targets,
            matrix=matrix, scorer=scorer if scorer is not None else "sw",
            threshold=int(threshold),
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
        )
    if scorer is None:
        raise TypeError("cdist_above_threshold requires either scorer= or matrix=")
    resolved_cpu_count = cpu_count
    if resolved_cpu_count <= 0:
        resolved_cpu_count = os.cpu_count() or 1
    return _LEVENSHTEIN_BACKEND.cdist_above_threshold(
        queries, targets,
        scorer=scorer,
        threshold=float(threshold),
        tqdm=tqdm,
        cpu_count=int(resolved_cpu_count),
        prefix_weight=prefix_weight,
        prefix_threshold=prefix_threshold,
        prefix_cap=prefix_cap,
    )


def _cdist_above_threshold_matrix(
    queries: object,
    targets: object,
    *,
    matrix: Any,
    scorer: Any,
    threshold: int,
    gap_score: int,
    gap_open_score: int | None,
    gap_extend_score: int | None,
):
    """Iterator over (score, query, target) for matrix-mode cdist."""
    scores = _cdist_matrix(
        queries, targets,
        matrix=matrix, scorer=scorer,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
    )
    query_tuple = _materialize_targets(queries)
    target_tuple = _materialize_targets(targets)
    return _cdist_threshold_iter(scores, query_tuple, target_tuple, threshold)


def _cdist_threshold_iter(
    scores: np.ndarray,
    query_tuple: tuple,
    target_tuple: tuple,
    threshold: int,
):
    for i, q in enumerate(query_tuple):
        row = scores[i]
        for j in np.flatnonzero(row >= threshold):
            yield int(row[j]), q, target_tuple[j]


def cdist_top_k(
    queries: object,
    targets: object,
    *,
    scorer: "Scorer | object" = None,
    k: int,
    matrix: Any = None,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    tqdm: object = None,
    cpu_count: int = 0,
    reject_duplicates: bool = False,
    prefix_weight: float = 0.1,
    prefix_threshold: float = 0.7,
    prefix_cap: int = 4,
) -> list[tuple[float, object, object]]:
    """Return the ``k`` highest-scoring ``(query, target)`` pairs as
    an unsorted list of ``(score, query, target)`` tuples.

    Variant of ``cdist_above_threshold`` that uses a heap instead of a
    streaming queue. Each worker maintains a private top-k heap of
    ``(score, q_index, t_index)`` records (no Python-object refcount
    traffic during the hot loop); after all workers finish, the
    per-thread heaps are merged into a final top-k heap, and only
    then is each record materialized as a ``(score, query_string,
    target_string)`` tuple.

    The returned list is NOT sorted — heap order, not score order.
    The caller pays for sorting only if needed.

    Memory is O(``cpu_count`` * ``k`` + ``len(queries) + len(targets)``).

    Requires a normalized-similarity scorer.

    ``reject_duplicates`` (default False): when True and the computed
    score is exactly 1.0, the underlying byte buffers are compared;
    if they match, the pair is skipped (and so is its symmetric
    mirror). Useful for fuzzy-match workflows that want to surface
    near-duplicates while ignoring exact duplicates.

    Matrix mode (``matrix=``): scores are raw integers (no [0, 1]
    normalization), reject_duplicates is unsupported, and the call
    runs single-threaded in Python on top of the batch matrix kernel.
    """
    if matrix is not None:
        if reject_duplicates:
            raise NotImplementedError(
                "reject_duplicates is not supported for matrix-mode cdist_top_k"
            )
        return _cdist_top_k_matrix(
            queries, targets,
            matrix=matrix, scorer=scorer if scorer is not None else "sw",
            k=int(k),
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
        )
    if scorer is None:
        raise TypeError("cdist_top_k requires either scorer= or matrix=")
    resolved_cpu_count = cpu_count
    if resolved_cpu_count <= 0:
        resolved_cpu_count = os.cpu_count() or 1
    return _LEVENSHTEIN_BACKEND.cdist_top_k(
        queries, targets,
        scorer=scorer,
        k=int(k),
        tqdm=tqdm,
        cpu_count=int(resolved_cpu_count),
        reject_duplicates=bool(reject_duplicates),
        prefix_weight=prefix_weight,
        prefix_threshold=prefix_threshold,
        prefix_cap=prefix_cap,
    )


def cdist_top_k_per_query(
    queries: object,
    targets: object,
    *,
    scorer: "Scorer",
    k: int = 5,
    pruning: bool = False,
    cpu_count: int = 0,
    prefix_weight: float = 0.1,
    prefix_threshold: float = 0.7,
    prefix_cap: int = 4,
) -> Iterator[tuple[object, list[tuple[float, object]]]]:
    """Yield ``(query, [(score, target), ...])`` for each query.

    The k highest-similarity targets *per query*. Differs from
    :func:`cdist_top_k` (which returns the k highest pairs globally
    across the whole query × target matrix) by keeping a separate
    top-k heap per query.

    Two execution paths:

    * **Threaded** (``cpu_count != 1``): runs a worker pool over the
      queries — same threading model as :func:`cdist_top_k` — with the
      SIMD batch scoring kernel processing each query's whole row
      against all targets. Requires byte-compatible inputs (bytes or
      1-byte unicode). On wider unicode inputs the binding raises
      ``NotImplementedError`` and we silently fall back to the
      single-threaded path. ``cpu_count = 0`` auto-detects via
      ``os.cpu_count() or 1``; ``cpu_count = 1`` opts out of threading.

    * **Single-threaded** (``cpu_count == 1`` or threaded fallback):
      per-pair dispatch through the same scoring kernels the singular
      ``*_normalized_score`` functions use. Supports arbitrary input
      kinds (bytes, 1-byte and wider unicode). Honours
      ``pruning=True`` for adaptive heap-min cutoff — as targets are
      scored, the worst-in-heap score is tracked and targets whose
      closed-form upper bound on similarity can't beat it are skipped
      before the scoring kernel runs.

    The threaded path always applies the closed-form length-bound
    prune (``max_normalized_similarity == 0`` → drop), even if
    ``pruning=False``; that's a correctness gate, not an opt-in.

    ``targets`` is materialised into a list once and re-iterated for
    every query — generator inputs become a stored list.

    ``scorer`` must be a normalised-similarity scorer:
    ``LEVENSHTEIN_NORMALIZED``, ``DAMERAU_LEVENSHTEIN_NORMALIZED``,
    ``HAMMING_NORMALIZED``, ``INDEL_NORMALIZED``,
    ``TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED``, ``JARO``, or
    ``JARO_WINKLER``. Distance scorers aren't supported here — use
    :func:`cdist_top_k` or invert the score manually.

    Each emitted list is sorted descending by score.
    """
    if isinstance(queries, (str, bytes)):
        raise TypeError(
            "queries must be an iterable of queries, not a single str/bytes"
        )
    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "targets must be an iterable of target sequences, not a single str/bytes"
        )
    # Scorer must be one of the normalised-similarity scorers. Both
    # the threaded and single-threaded paths share this constraint;
    # check once here so the message is identical regardless of which
    # path we'd dispatch to.
    _supported_normalized = {
        int(Scorer.LEVENSHTEIN_NORMALIZED),
        int(Scorer.DAMERAU_LEVENSHTEIN_NORMALIZED),
        int(Scorer.HAMMING_NORMALIZED),
        int(Scorer.INDEL_NORMALIZED),
        int(Scorer.TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED),
        int(Scorer.JARO),
        int(Scorer.JARO_WINKLER),
    }
    if int(scorer) not in _supported_normalized:
        raise TypeError(
            "cdist_top_k_per_query: scorer must be a normalised-similarity "
            "scorer (LEVENSHTEIN_NORMALIZED, DAMERAU_LEVENSHTEIN_NORMALIZED, "
            "HAMMING_NORMALIZED, INDEL_NORMALIZED, "
            "TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED, JARO, or JARO_WINKLER)"
        )
    # Materialise once. We iterate it per query, so generator inputs
    # have to become a real sequence here.
    if not isinstance(targets, (list, tuple)):
        targets = list(targets)
    scorer_int = int(scorer)
    k_int = int(k)
    prefix_weight_f = float(prefix_weight)
    prefix_threshold_f = float(prefix_threshold)
    prefix_cap_i = int(prefix_cap)
    pruning_b = bool(pruning)
    resolved_cpu_count = int(cpu_count)
    if resolved_cpu_count <= 0:
        resolved_cpu_count = os.cpu_count() or 1

    # Threaded path: materialise queries too (it iterates them), call
    # the C++ binding, then yield the results. If the binding raises
    # NotImplementedError (wide unicode) we drop into the per-pair
    # path below.
    if resolved_cpu_count > 1:
        if not isinstance(queries, (list, tuple)):
            queries = list(queries)
        try:
            results = _LEVENSHTEIN_BACKEND.cdist_top_k_per_query_threaded(
                queries, targets,
                scorer=scorer_int, k=k_int, pruning=pruning_b,
                cpu_count=resolved_cpu_count,
                prefix_weight=prefix_weight_f,
                prefix_threshold=prefix_threshold_f,
                prefix_cap=prefix_cap_i,
            )
        except NotImplementedError:
            # Wide-unicode input: fall through to single-threaded.
            results = None
        if results is not None:
            return iter(results)

    # Single-threaded path (default for cpu_count == 1, fallback
    # otherwise). Yields lazily, supports arbitrary input kinds, and
    # honours the adaptive heap-min cutoff when ``pruning=True``.
    one_query = _LEVENSHTEIN_BACKEND.cdist_top_k_one_query
    # Generator body lives in a nested function so the str/bytes
    # rejections above fire eagerly at call time rather than on first
    # consumption.
    def _iter() -> Iterator[tuple[object, list[tuple[float, object]]]]:
        for query in queries:
            result = one_query(
                query, targets,
                scorer=scorer_int, k=k_int,
                prefix_weight=prefix_weight_f,
                prefix_threshold=prefix_threshold_f,
                prefix_cap=prefix_cap_i,
                pruning=pruning_b,
            )
            yield query, result

    return _iter()


def _cdist_top_k_matrix(
    queries: object,
    targets: object,
    *,
    matrix: Any,
    scorer: Any,
    k: int,
    gap_score: int,
    gap_open_score: int | None,
    gap_extend_score: int | None,
) -> list[tuple[int, object, object]]:
    scores = _cdist_matrix(
        queries, targets,
        matrix=matrix, scorer=scorer,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
    )
    query_tuple = _materialize_targets(queries)
    target_tuple = _materialize_targets(targets)
    if scores.size == 0 or k <= 0:
        return []
    # Top-k by partial sort over the flattened (Q × T) score grid.
    flat = scores.ravel()
    take = min(k, flat.size)
    indices = np.argpartition(-flat, take - 1)[:take]
    result: list[tuple[int, object, object]] = []
    ncols = scores.shape[1]
    for flat_idx in indices:
        i, j = int(flat_idx) // ncols, int(flat_idx) % ncols
        result.append((int(flat[flat_idx]), query_tuple[i], target_tuple[j]))
    return result


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
        (indel_scores, Scorer.INDEL),
        (indel_normalized_scores, Scorer.INDEL_NORMALIZED),
        (true_damerau_levenshtein_scores, Scorer.TRUE_DAMERAU_LEVENSHTEIN),
        (true_damerau_levenshtein_normalized_scores,
         Scorer.TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED),
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
    "cdist_above_threshold",
    "cdist_top_k",
    "cdist_top_k_per_query",
    "damerau_levenshtein_best",
    "damerau_levenshtein_normalized_best",
    "damerau_levenshtein_normalized_score",
    "damerau_levenshtein_normalized_scores",
    "damerau_levenshtein_normalized_top_k",
    "damerau_levenshtein_score",
    "damerau_levenshtein_scores",
    "damerau_levenshtein_top_k",
    "detect_best_backend",
    "dtw",
    "dtw_distances",
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
    "indel_best",
    "indel_normalized_best",
    "indel_normalized_score",
    "indel_normalized_scores",
    "indel_normalized_top_k",
    "indel_score",
    "indel_scores",
    "indel_top_k",
    "jaro_best",
    "jaro_similarities",
    "jaro_similarity",
    "jaro_top_k",
    "jaro_winkler_best",
    "jaro_winkler_similarities",
    "jaro_winkler_similarity",
    "jaro_winkler_top_k",
    "cosine",
    "cosine_similarities",
    "dice",
    "dice_similarities",
    "jaccard",
    "jaccard_similarities",
    "lcs_length",
    "lcs_substring",
    "lcs_substring_length",
    "monge_elkan",
    "overlap",
    "overlap_similarities",
    "partial_ratio",
    "partial_token_set_ratio",
    "partial_token_sort_ratio",
    "ratcliff_obershelp_similarities",
    "ratcliff_obershelp_similarity",
    "token_set_ratio",
    "token_sort_ratio",
    "WRatio",
    "LevenshteinScorer",
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
    "beider_morse",
    "BmpmRuleType",
    "caverphone",
    "cologne_phonetic",
    "daitch_mokotoff",
    "double_metaphone",
    "DoubleMetaphoneVariant",
    "match_rating_codex",
    "match_rating_compare",
    "metaphone",
    "metaphone_equal",
    "MetaphoneVariant",
    "nysiis",
    "nysiis_equal",
    "soundex",
    "soundex_equal",
    "true_damerau_levenshtein_best",
    "true_damerau_levenshtein_normalized_best",
    "true_damerau_levenshtein_normalized_score",
    "true_damerau_levenshtein_normalized_scores",
    "true_damerau_levenshtein_normalized_top_k",
    "true_damerau_levenshtein_score",
    "true_damerau_levenshtein_scores",
    "true_damerau_levenshtein_top_k",
]
