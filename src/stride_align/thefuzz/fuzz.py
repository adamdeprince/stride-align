"""Integer-scored work-alikes for ``thefuzz.fuzz`` 0.22.1."""

from __future__ import annotations

import sys
from array import array
from math import isnan

import stride_align as _sa
from stride_align.rapidfuzz import fuzz as _rf_fuzz
from stride_align.thefuzz import utils


def _is_none(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and isnan(value):
        return True
    pandas = sys.modules.get("pandas")
    return pandas is not None and value is getattr(pandas, "NA", None)


def _is_native_text(value) -> bool:
    return isinstance(value, (str, bytes))


def _raw_ratio(s1, s2) -> float:
    if _is_none(s1) or _is_none(s2):
        return 0.0
    if _is_native_text(s1) and _is_native_text(s2):
        return float(_rf_fuzz.ratio(s1, s2))
    return float(_sa.indel_normalized_score(_compat_sequence(s1), _compat_sequence(s2))) * 100.0


def _raw_partial_ratio(s1, s2) -> float:
    if _is_none(s1) or _is_none(s2):
        return 0.0
    if _is_native_text(s1) and _is_native_text(s2):
        return float(_sa._partial_ratio_kernel(s1, s2)) * 100.0
    return float(_sa._partial_ratio_kernel(_compat_sequence(s1), _compat_sequence(s2))) * 100.0


def _compat_sequence(sequence):
    """Normalize generic sequence elements for TheFuzz compatibility.

    Single-character strings retain their code point so text and generic
    character sequences interoperate. Other hashable objects use their hash
    value, matching TheFuzz's RapidFuzz-backed sequence convention. The
    resulting integers then enter stride-align's shared compact-token path.
    """
    mask = (1 << 64) - 1
    if isinstance(sequence, str):
        return [ord(char) for char in sequence]
    if isinstance(sequence, bytes):
        return list(sequence)
    if isinstance(sequence, array) and sequence.typecode in ("u", "w"):
        return [ord(char) for char in sequence]

    encoded = []
    # Indexing, rather than generic iteration, retains RapidFuzz's public
    # requirement that inputs are sequences (sets and generators are not).
    for index in range(len(sequence)):
        element = sequence[index]
        if isinstance(element, str) and len(element) == 1:
            value = ord(element)
        elif isinstance(element, int) and element == -1:
            value = -1
        else:
            value = hash(element)
        encoded.append(value & mask)
    return encoded


def _is_space_token(token: int) -> bool:
    try:
        return chr(token).isspace()
    except (OverflowError, ValueError):
        return False


def _split_sequence(sequence) -> list[tuple[int, ...]]:
    parts: list[list[int]] = [[]]
    for token in _compat_sequence(sequence):
        if _is_space_token(token):
            parts.append([])
        else:
            parts[-1].append(token)
    return [tuple(part) for part in parts if part]


def _join_parts(parts) -> list[int]:
    joined: list[int] = []
    for index, part in enumerate(parts):
        if index:
            joined.append(ord(" "))
        joined.extend(part)
    return joined


def _raw_token_sort_ratio(s1, s2) -> float:
    if _is_none(s1) or _is_none(s2):
        return 0.0
    if _is_native_text(s1) and _is_native_text(s2):
        return float(_rf_fuzz.token_sort_ratio(s1, s2))
    left = _join_parts(sorted(_split_sequence(s1)))
    right = _join_parts(sorted(_split_sequence(s2)))
    return _raw_ratio(left, right)


def _token_set_candidates(s1, s2):
    left = set(_split_sequence(s1))
    right = set(_split_sequence(s2))
    if not left or not right:
        return None
    intersection = sorted(left & right)
    left_only = sorted(left - right)
    right_only = sorted(right - left)
    common = _join_parts(intersection)
    common_left = _join_parts([*intersection, *left_only])
    common_right = _join_parts([*intersection, *right_only])
    return common, common_left, common_right


def _raw_token_set_ratio(s1, s2) -> float:
    if _is_none(s1) or _is_none(s2):
        return 0.0
    candidates = _token_set_candidates(s1, s2)
    if candidates is None:
        return 0.0
    common, common_left, common_right = candidates

    def score(left, right):
        total = len(left) + len(right)
        if not total:
            return 100.0
        distance = int(_sa.indel_score(left, right))
        # Operation order is observable when TheFuzz ranks the unrounded
        # RapidFuzz scores and rounds only after selection.
        return 100.0 - 100.0 * distance / total

    return max(
        score(common, common_left),
        score(common, common_right),
        score(common_left, common_right),
    )


def _raw_partial_token_sort_ratio(s1, s2) -> float:
    if _is_none(s1) or _is_none(s2):
        return 0.0
    if _is_native_text(s1) and _is_native_text(s2):
        return float(_rf_fuzz.partial_token_sort_ratio(s1, s2))
    left = _join_parts(sorted(_split_sequence(s1)))
    right = _join_parts(sorted(_split_sequence(s2)))
    return _raw_partial_ratio(left, right)


def _raw_partial_token_set_ratio(s1, s2) -> float:
    if _is_none(s1) or _is_none(s2):
        return 0.0
    if _is_native_text(s1) and _is_native_text(s2):
        return float(_rf_fuzz.partial_token_set_ratio(s1, s2))
    candidates = _token_set_candidates(s1, s2)
    if candidates is None:
        return 0.0
    common, common_left, common_right = candidates
    return max(
        _raw_partial_ratio(common, common_left),
        _raw_partial_ratio(common, common_right),
        _raw_partial_ratio(common_left, common_right),
    )


def _raw_qratio(s1, s2) -> float:
    if _is_none(s1) or _is_none(s2) or not s1 or not s2:
        return 0.0
    return _raw_ratio(s1, s2)


def _raw_wratio(s1, s2) -> float:
    if _is_none(s1) or _is_none(s2) or not s1 or not s2:
        return 0.0

    base = _raw_ratio(s1, s2)
    length_ratio = max(len(s1), len(s2)) / min(len(s1), len(s2))
    if length_ratio < 1.5:
        return max(
            base,
            _raw_token_sort_ratio(s1, s2) * 0.95,
            _raw_token_set_ratio(s1, s2) * 0.95,
        )

    partial_scale = 0.9 if length_ratio <= 8.0 else 0.6
    return max(
        base,
        _raw_partial_ratio(s1, s2) * partial_scale,
        _raw_partial_token_sort_ratio(s1, s2) * 0.95 * partial_scale,
        _raw_partial_token_set_ratio(s1, s2) * 0.95 * partial_scale,
    )


def _rounded_score(scorer, s1, s2, force_ascii, full_process) -> int:
    if full_process:
        if s1 is None or s2 is None:
            return 0
        s1 = utils.full_process(s1, force_ascii=force_ascii)
        s2 = utils.full_process(s2, force_ascii=force_ascii)
    return int(round(scorer(s1, s2)))


def ratio(s1, s2):
    return _rounded_score(_raw_ratio, s1, s2, False, False)


def partial_ratio(s1, s2):
    return _rounded_score(_raw_partial_ratio, s1, s2, False, False)


def token_sort_ratio(s1, s2, force_ascii=True, full_process=True):
    return _rounded_score(_raw_token_sort_ratio, s1, s2, force_ascii, full_process)


def partial_token_sort_ratio(s1, s2, force_ascii=True, full_process=True):
    return _rounded_score(_raw_partial_token_sort_ratio, s1, s2, force_ascii, full_process)


def token_set_ratio(s1, s2, force_ascii=True, full_process=True):
    return _rounded_score(_raw_token_set_ratio, s1, s2, force_ascii, full_process)


def partial_token_set_ratio(s1, s2, force_ascii=True, full_process=True):
    return _rounded_score(_raw_partial_token_set_ratio, s1, s2, force_ascii, full_process)


def QRatio(s1, s2, force_ascii=True, full_process=True):  # noqa: N802
    return _rounded_score(_raw_qratio, s1, s2, force_ascii, full_process)


def UQRatio(s1, s2, full_process=True):  # noqa: N802
    return QRatio(s1, s2, force_ascii=False, full_process=full_process)


def WRatio(s1, s2, force_ascii=True, full_process=True):  # noqa: N802
    return _rounded_score(_raw_wratio, s1, s2, force_ascii, full_process)


def UWRatio(s1, s2, full_process=True):  # noqa: N802
    return WRatio(s1, s2, force_ascii=False, full_process=full_process)


_RAW_SCORERS = {
    ratio: _raw_ratio,
    partial_ratio: _raw_partial_ratio,
    token_sort_ratio: _raw_token_sort_ratio,
    partial_token_sort_ratio: _raw_partial_token_sort_ratio,
    token_set_ratio: _raw_token_set_ratio,
    partial_token_set_ratio: _raw_partial_token_set_ratio,
    QRatio: _raw_qratio,
    UQRatio: _raw_qratio,
    WRatio: _raw_wratio,
    UWRatio: _raw_wratio,
}

__all__ = [
    "ratio",
    "partial_ratio",
    "token_sort_ratio",
    "partial_token_sort_ratio",
    "token_set_ratio",
    "partial_token_set_ratio",
    "QRatio",
    "UQRatio",
    "WRatio",
    "UWRatio",
]
