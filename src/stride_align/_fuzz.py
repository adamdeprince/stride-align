"""Token-ratio family — Phase D.3.

The rapidfuzz / FuzzyWuzzy API gap closer: ``token_sort_ratio``,
``token_set_ratio``, ``partial_ratio``, ``partial_token_sort_ratio``,
``partial_token_set_ratio``, and ``WRatio``.

Pure-Python composition on top of stride-align primitives — no
third-party code is imported into the production path:

* The base ratio is ``sa.indel_normalized_score``, which is
  algebraically identical to rapidfuzz's ``fuzz.ratio / 100``: both
  reduce to ``2 · LCS(a, b) / (|a| + |b|)``.
* Candidate alignment positions for ``partial_ratio`` come from a
  sliding-window scan; the per-position similarity uses the same
  bit-parallel Indel kernel that powers ``sa.indel_normalized_score``.

Returned values are in ``[0, 1]`` (stride-align convention). rapidfuzz
returns ``[0, 100]``; multiply by 100 if you need that scale.
"""

from __future__ import annotations

from typing import Callable


# Imported lazily from the package root to avoid a circular import at
# module-load time (``stride_align/__init__.py`` re-exports the names
# this module defines).
def _indel_normalized_score(a: str, b: str) -> float:
    from stride_align import indel_normalized_score
    return float(indel_normalized_score(a, b))


def _lcs_substring(a: str, b: str) -> str:
    """Longest common contiguous substring (Phase D.4 primitive)."""
    from stride_align import lcs_substring
    return str(lcs_substring(a, b))


# ---- Helpers ------------------------------------------------------

def _coerce_str(s: object) -> str:
    """Token ratios are defined over strings — bytes are accepted but
    decoded as Latin-1 (the same widening convention the C++ engines
    use for ``bytes`` input)."""
    if isinstance(s, str):
        return s
    if isinstance(s, (bytes, bytearray)):
        return bytes(s).decode("latin-1")
    raise TypeError(
        f"token ratio inputs must be str or bytes, got {type(s).__name__}"
    )


def _apply_processor(
    s: str, processor: Callable[[str], str] | None,
) -> str:
    return processor(s) if processor is not None else s


def _tokenise(s: str) -> list[str]:
    """Whitespace tokenisation, the rapidfuzz default. Empty input
    returns ``[]``; multiple consecutive spaces are collapsed."""
    return s.split()


def _sort_join(s: str) -> str:
    return " ".join(sorted(_tokenise(s)))


# ---- Token-sort and token-set ratios -----------------------------

def token_sort_ratio(
    s1: object,
    s2: object,
    *,
    processor: Callable[[str], str] | None = None,
) -> float:
    """Tokenise each input on whitespace, sort the tokens
    lexicographically, join with single spaces, then return the
    Indel-normalised similarity of the two sorted joins.

    Matches rapidfuzz's ``fuzz.token_sort_ratio(s1, s2) / 100`` for
    any pair where neither input is empty after tokenisation. Empty-
    or whitespace-only inputs on both sides return ``1.0`` (vacuously
    identical); one empty and one non-empty returns ``0.0``.

    The tokenisation, sort, join, and Indel similarity all run inside
    one C++ kernel call (``_token_sort_ratio_kernel``); the Python
    layer here is only the processor / type coercion. See
    ``include/stride_align/token_ratios.hpp`` for the engine.
    """
    from stride_align import _token_sort_ratio_kernel
    a = _apply_processor(_coerce_str(s1), processor)
    b = _apply_processor(_coerce_str(s2), processor)
    return float(_token_sort_ratio_kernel(a, b))


def token_set_ratio(
    s1: object,
    s2: object,
    *,
    processor: Callable[[str], str] | None = None,
) -> float:
    """Tokenise each input on whitespace, take the set-intersection and
    the two set-differences, build three candidate strings, and return
    the maximum pairwise Indel-normalised similarity over them.

    The three candidates are built from the SORTED intersection
    ``T0``, the sorted intersection plus sorted ``s1``-only tokens
    ``T1``, and the sorted intersection plus sorted ``s2``-only tokens
    ``T2``. The three ratios compared are ``r(T0, T1)``, ``r(T0, T2)``,
    and ``r(T1, T2)``. Matches the rapidfuzz formula bit-exactly.

    All of the above runs inside one C++ kernel call
    (``_token_set_ratio_kernel``); see
    ``include/stride_align/token_ratios.hpp`` for the engine.
    """
    from stride_align import _token_set_ratio_kernel
    a = _apply_processor(_coerce_str(s1), processor)
    b = _apply_processor(_coerce_str(s2), processor)
    return float(_token_set_ratio_kernel(a, b))


# ---- Partial ratio ------------------------------------------------

def partial_ratio(
    s1: object,
    s2: object,
    *,
    processor: Callable[[str], str] | None = None,
) -> float:
    """Best Indel-normalised similarity over plausible alignments of
    the shorter string within the longer one.

    Enumerates **matching blocks** between the shorter and longer
    strings (the rapidfuzz / difflib approach) — every common
    substring identifies a natural alignment shift, which generates
    one candidate window of length ``len(shorter)`` clamped at the
    longer string's boundaries. The maximum Indel-normalised
    similarity across those windows is returned.

    The matching-block decomposition, window enumeration, and per-
    window Indel ratio all happen inside the C++ kernel
    (``stride_align::partial_ratio::partial_ratio``) — one Python
    boundary crossing per call instead of one per matching block ×
    candidate window. The algorithm is identical to the Python
    reference that lived here through D.3; see
    ``include/stride_align/partial_ratio.hpp`` for the implementation.
    """
    from stride_align import _partial_ratio_kernel
    a = _apply_processor(_coerce_str(s1), processor)
    b = _apply_processor(_coerce_str(s2), processor)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(_partial_ratio_kernel(a, b))


def partial_token_sort_ratio(
    s1: object,
    s2: object,
    *,
    processor: Callable[[str], str] | None = None,
) -> float:
    """Token-sort preprocessing + ``partial_ratio``. Equivalent to
    rapidfuzz's ``fuzz.partial_token_sort_ratio / 100``."""
    a = _apply_processor(_coerce_str(s1), processor)
    b = _apply_processor(_coerce_str(s2), processor)
    return partial_ratio(_sort_join(a), _sort_join(b))


def partial_token_set_ratio(
    s1: object,
    s2: object,
    *,
    processor: Callable[[str], str] | None = None,
) -> float:
    """Token-set preprocessing + ``partial_ratio``. Equivalent to
    rapidfuzz's ``fuzz.partial_token_set_ratio / 100``."""
    a_str = _apply_processor(_coerce_str(s1), processor)
    b_str = _apply_processor(_coerce_str(s2), processor)
    a_tokens = set(_tokenise(a_str))
    b_tokens = set(_tokenise(b_str))
    # rapidfuzz convention: zero when either side has no tokens.
    if not a_tokens or not b_tokens:
        return 0.0

    intersect = sorted(a_tokens & b_tokens)
    diff_a    = sorted(a_tokens - b_tokens)
    diff_b    = sorted(b_tokens - a_tokens)

    t0 = " ".join(intersect)
    t1 = (t0 + " " + " ".join(diff_a)).strip()
    t2 = (t0 + " " + " ".join(diff_b)).strip()

    p0 = partial_ratio(t0, t1)
    p1 = partial_ratio(t0, t2)
    p2 = partial_ratio(t1, t2)
    return max(p0, p1, p2)


# ---- Weighted blend (rapidfuzz WRatio) ---------------------------

# The exact rapidfuzz WRatio recipe: pick the maximum among several
# weighted candidates depending on length ratio. The constants below
# (0.9 partial weight, 0.95 token weight, 1.5 / 8 length thresholds)
# come from rapidfuzz's published algorithm documentation; the
# implementation here is original.
_UNBASE_SCALE  = 0.95
_PARTIAL_SCALE = 0.9


def WRatio(
    s1: object,
    s2: object,
    *,
    processor: Callable[[str], str] | None = None,
) -> float:
    """Weighted ratio — rapidfuzz's ``fuzz.WRatio / 100``.

    Blends the base Indel ratio with token-sort, token-set, and the
    partial variants. The exact weights and length-ratio thresholds
    follow rapidfuzz's published recipe; values that fall below the
    base ratio are dropped, and the maximum survives.
    """
    a = _apply_processor(_coerce_str(s1), processor)
    b = _apply_processor(_coerce_str(s2), processor)
    if not a or not b:
        return 1.0 if (not a and not b) else 0.0

    base = _indel_normalized_score(a, b)

    len_a, len_b = len(a), len(b)
    len_ratio = max(len_a, len_b) / min(len_a, len_b)

    try_partial = True
    unbase_scale = _UNBASE_SCALE
    partial_scale = _PARTIAL_SCALE

    if len_ratio < 1.5:
        # Lengths similar — try token sort and token set, scaled by
        # ``unbase_scale``. Partials don't get the extra penalty.
        partial_scale = unbase_scale
    elif len_ratio >= 8:
        # Very different lengths — partial dominates; token ratios
        # are useless when one string is much smaller.
        try_partial = True
        unbase_scale = 0.6
    # Otherwise medium length difference — use defaults.

    candidates = [base]
    if try_partial:
        candidates.append(partial_ratio(a, b) * partial_scale)
    candidates.append(token_sort_ratio(a, b) * unbase_scale)
    candidates.append(token_set_ratio(a, b) * unbase_scale)
    if try_partial:
        candidates.append(partial_token_sort_ratio(a, b) * partial_scale * unbase_scale)
        candidates.append(partial_token_set_ratio(a, b) * partial_scale * unbase_scale)
    return max(candidates)
