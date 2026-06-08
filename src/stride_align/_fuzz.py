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
    """
    a = _apply_processor(_coerce_str(s1), processor)
    b = _apply_processor(_coerce_str(s2), processor)
    return _indel_normalized_score(_sort_join(a), _sort_join(b))


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
    """
    a_str = _apply_processor(_coerce_str(s1), processor)
    b_str = _apply_processor(_coerce_str(s2), processor)
    a_tokens = set(_tokenise(a_str))
    b_tokens = set(_tokenise(b_str))
    # rapidfuzz convention: token_set_ratio returns 0 when either side
    # has no tokens. The set-difference algebra below would otherwise
    # treat ``(∅, ∅)`` as vacuously identical and report 1.0.
    if not a_tokens or not b_tokens:
        return 0.0

    intersect = sorted(a_tokens & b_tokens)
    diff_a    = sorted(a_tokens - b_tokens)
    diff_b    = sorted(b_tokens - a_tokens)

    t0 = " ".join(intersect)
    t1 = (t0 + " " + " ".join(diff_a)).strip()
    t2 = (t0 + " " + " ".join(diff_b)).strip()

    r0 = _indel_normalized_score(t0, t1)
    r1 = _indel_normalized_score(t0, t2)
    r2 = _indel_normalized_score(t1, t2)
    return max(r0, r1, r2)


# ---- Partial ratio ------------------------------------------------

def _matching_blocks(short: str, long: str) -> list[tuple[int, int, int]]:
    """Recursive longest-common-substring decomposition à la difflib's
    ``SequenceMatcher.get_matching_blocks`` — built on stride-align's
    own ``lcs_substring`` primitive (Phase D.4), not on the upstream
    library. Returns a list of ``(short_pos, long_pos, length)``
    tuples in ascending order of ``short_pos``."""
    blocks: list[tuple[int, int, int]] = []

    # Iterative work stack to avoid hitting Python's recursion limit
    # on long inputs with many small matching blocks.
    work: list[tuple[int, int, int, int]] = [(0, len(short), 0, len(long))]
    while work:
        s_lo, s_hi, l_lo, l_hi = work.pop()
        if s_lo >= s_hi or l_lo >= l_hi:
            continue
        sub_s = short[s_lo:s_hi]
        sub_l = long[l_lo:l_hi]
        lcs = _lcs_substring(sub_s, sub_l)
        if not lcs:
            continue
        s_pos = sub_s.find(lcs) + s_lo
        l_pos = sub_l.find(lcs) + l_lo
        k = len(lcs)
        # Recurse on the right side first so the left side pops next
        # — keeps the final block list sorted by short_pos.
        work.append((s_pos + k, s_hi, l_pos + k, l_hi))
        blocks.append((s_pos, l_pos, k))
        work.append((s_lo, s_pos, l_lo, l_pos))

    blocks.sort(key=lambda b: b[0])
    return blocks


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

    The matching-block decomposition uses stride-align's own
    ``lcs_substring`` primitive (Phase D.4) recursively; no
    third-party code is imported into the production path. Falls
    back to a sliding-window scan only when no matching block exists
    (the inputs share no character).
    """
    a = _apply_processor(_coerce_str(s1), processor)
    b = _apply_processor(_coerce_str(s2), processor)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    n = len(short)
    m = len(long)

    blocks = _matching_blocks(short, long)
    if not blocks:
        return 0.0

    best = 0.0
    seen_windows: set[tuple[int, int]] = set()

    def try_window(start: int, end: int) -> bool:
        nonlocal best
        if end - start <= 0 or start < 0 or end > m:
            return False
        key = (start, end)
        if key in seen_windows:
            return False
        seen_windows.add(key)
        score = _indel_normalized_score(short, long[start:end])
        if score > best:
            best = score
        return best >= 1.0

    # Every block contributes two windows of nominal length
    # ``len(short)``, each clamped at long's boundaries:
    #
    #   * "block at LEFT edge" — natural shift placing the block at
    #     the same offset in long as in short. Window starts at
    #     ``l_pos - s_pos`` clamped at 0; length is ``len(short)``
    #     clamped at long's right edge.
    #   * "block at RIGHT edge" — alignment placing the block's end
    #     at ``l_pos + k`` with the window ending there. When the
    #     window would extend before long's start, it shortens —
    #     yielding the block-region-only window in the case where
    #     the block sits at the start of long.
    #
    # The shortened windows from right-edge clamping are exactly the
    # rapidfuzz "partial-ratio sweet spot" — e.g. ``'color'`` vs
    # ``'colour'`` finds the length-4 ``'colo'`` window because the
    # right-edge alignment for the matching block would end at
    # position 4 with start clamped to 0. Restricting to block-anchored
    # candidates (no bare sliding window, no standalone block region)
    # means the shim never overshoots rapidfuzz on the cases where
    # rapidfuzz's own algorithm misses an optimal sliding-window
    # position outside any block.
    for s_pos, l_pos, k in blocks:
        left_shift = l_pos - s_pos
        if try_window(max(0, left_shift), min(m, left_shift + n)):
            return 1.0
        right_end = l_pos + k
        if try_window(max(0, right_end - n), min(m, right_end)):
            return 1.0
        # Block-as-window: only when the block is at least 4 chars.
        # Below that threshold the block is a sparse anchor and its
        # standalone window can overshoot rapidfuzz on inputs with
        # many small spurious matches (e.g. ``' cf'`` against
        # arbitrary text finds a 2-char ``' c'`` block whose
        # block-itself ratio 4/5 = 0.8 beats rapidfuzz's 2/3 = 0.667
        # from a length-3 window). At ≥ 4 chars the block is
        # substantial enough that rapidfuzz also considers it
        # (e.g. ``'java language'`` vs ``'python programming
        # language'``: the 9-char ``' language'`` block-itself
        # window gives 0.818, matching upstream).
        if k >= 4 and try_window(l_pos, l_pos + k):
            return 1.0

    return best


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
