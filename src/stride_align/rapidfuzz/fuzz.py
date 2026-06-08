"""``rapidfuzz.fuzz`` shim — token-ratio family scaled to ``[0, 100]``.

stride-align's D.3 ships the same algorithm family under the main
namespace returning ``[0, 1]``. This module wraps those functions
with ``× 100`` plus rapidfuzz's ``processor=`` / ``score_cutoff=``
contract, so user code that imports ``rapidfuzz.fuzz as fuzz`` and
calls ``fuzz.token_sort_ratio(...)`` keeps working unchanged.
"""

from __future__ import annotations

from typing import Callable, Optional

import stride_align as _sa


def _apply_processor(s, processor):
    return s if processor is None else processor(s)


def _clamp(score: float, score_cutoff: Optional[float]) -> float:
    """Apply rapidfuzz's similarity score-cutoff convention: anything
    below the cutoff returns ``0.0``; anything ``>=`` cutoff passes
    through unchanged."""
    if score_cutoff is not None and score < score_cutoff:
        return 0.0
    return score


def ratio(s1, s2, *, processor: Optional[Callable] = None,
          score_cutoff: Optional[float] = None) -> float:
    """Indel-normalised similarity scaled to ``[0, 100]`` — rapidfuzz's
    base ``fuzz.ratio``. Algebraically identical to
    ``stride_align.indel_normalized_score(s1, s2) * 100``.

    When ``score_cutoff`` is set, the cutoff is pushed into the
    bit-parallel Indel kernel (converted from rapidfuzz's ``[0, 100]``
    scale to stride-align's ``[0, 1]`` normalised-similarity scale).
    The kernel bails out of the per-character loop when it can prove
    the final similarity will fall below the cutoff — useful in
    ``extract`` / ``cdist`` workloads where most pairs fail the
    threshold.
    """
    a = _apply_processor(s1, processor)
    b = _apply_processor(s2, processor)
    kernel_cutoff = None if score_cutoff is None else score_cutoff / 100.0
    sim = float(_sa.indel_normalized_score(a, b, score_cutoff=kernel_cutoff))
    return _clamp(sim * 100.0, score_cutoff)


def partial_ratio(s1, s2, *, processor: Optional[Callable] = None,
                  score_cutoff: Optional[float] = None) -> float:
    a = _apply_processor(s1, processor)
    b = _apply_processor(s2, processor)
    return _clamp(float(_sa.partial_ratio(a, b)) * 100.0, score_cutoff)


def token_sort_ratio(s1, s2, *, processor: Optional[Callable] = None,
                     score_cutoff: Optional[float] = None) -> float:
    a = _apply_processor(s1, processor)
    b = _apply_processor(s2, processor)
    return _clamp(float(_sa.token_sort_ratio(a, b)) * 100.0, score_cutoff)


def token_set_ratio(s1, s2, *, processor: Optional[Callable] = None,
                    score_cutoff: Optional[float] = None) -> float:
    a = _apply_processor(s1, processor)
    b = _apply_processor(s2, processor)
    return _clamp(float(_sa.token_set_ratio(a, b)) * 100.0, score_cutoff)


def partial_token_sort_ratio(s1, s2, *, processor: Optional[Callable] = None,
                             score_cutoff: Optional[float] = None) -> float:
    a = _apply_processor(s1, processor)
    b = _apply_processor(s2, processor)
    return _clamp(float(_sa.partial_token_sort_ratio(a, b)) * 100.0, score_cutoff)


def partial_token_set_ratio(s1, s2, *, processor: Optional[Callable] = None,
                            score_cutoff: Optional[float] = None) -> float:
    a = _apply_processor(s1, processor)
    b = _apply_processor(s2, processor)
    return _clamp(float(_sa.partial_token_set_ratio(a, b)) * 100.0, score_cutoff)


def token_ratio(s1, s2, *, processor: Optional[Callable] = None,
                score_cutoff: Optional[float] = None) -> float:
    """``max(token_sort_ratio, token_set_ratio)``. rapidfuzz exposes
    this as the convenience max of the two token methods."""
    a = _apply_processor(s1, processor)
    b = _apply_processor(s2, processor)
    score = max(
        float(_sa.token_sort_ratio(a, b)),
        float(_sa.token_set_ratio(a, b)),
    ) * 100.0
    return _clamp(score, score_cutoff)


def partial_token_ratio(s1, s2, *, processor: Optional[Callable] = None,
                        score_cutoff: Optional[float] = None) -> float:
    """``max(partial_token_sort_ratio, partial_token_set_ratio)``."""
    a = _apply_processor(s1, processor)
    b = _apply_processor(s2, processor)
    score = max(
        float(_sa.partial_token_sort_ratio(a, b)),
        float(_sa.partial_token_set_ratio(a, b)),
    ) * 100.0
    return _clamp(score, score_cutoff)


def WRatio(s1, s2, *, processor: Optional[Callable] = None,
           score_cutoff: Optional[float] = None) -> float:  # noqa: N802
    """rapidfuzz's ``fuzz.WRatio`` — weighted blend of ratios.

    Reimplements rapidfuzz's documented recipe exactly (the
    ``sa.WRatio`` in the stride-align main namespace evaluates the
    partial-ratio branch unconditionally, which diverges from upstream
    when ``len_ratio < 1.5``; the shim version follows the upstream
    rule here for drop-in parity).
    """
    a = _apply_processor(s1, processor)
    b = _apply_processor(s2, processor)
    if not a or not b:
        return _clamp(0.0, score_cutoff)

    base = float(_sa.indel_normalized_score(a, b)) * 100.0
    len_ratio = max(len(a), len(b)) / min(len(a), len(b))

    UNBASE_SCALE = 0.95

    if len_ratio < 1.5:
        # Lengths similar — partial ratios add no signal beyond the
        # plain token ratios; the candidate set is just the three
        # token-based variants scaled by the un-base penalty.
        token_sort = float(_sa.token_sort_ratio(a, b)) * 100.0 * UNBASE_SCALE
        token_set  = float(_sa.token_set_ratio(a, b)) * 100.0 * UNBASE_SCALE
        return _clamp(max(base, token_sort, token_set), score_cutoff)

    # Length-mismatched regime. partial_ratio dominates the candidate
    # set; the non-partial token ratios are dropped because their
    # partial cousins are strictly better proxies here.
    partial_scale = 0.9 if len_ratio < 8.0 else 0.6

    partial             = float(_sa.partial_ratio(a, b)) * 100.0 * partial_scale
    partial_token_sort  = float(_sa.partial_token_sort_ratio(a, b)) * 100.0 * UNBASE_SCALE * partial_scale
    partial_token_set   = float(_sa.partial_token_set_ratio(a, b)) * 100.0 * UNBASE_SCALE * partial_scale
    return _clamp(
        max(base, partial, partial_token_sort, partial_token_set),
        score_cutoff,
    )


def QRatio(s1, s2, *, processor: Optional[Callable] = None,
           score_cutoff: Optional[float] = None) -> float:  # noqa: N802
    """rapidfuzz's "quick ratio". Since rapidfuzz v3.0 this behaves
    like ``fuzz.ratio`` with one exception: comparing two empty
    strings returns ``0`` instead of ``1.0``. Does NOT call
    ``utils.default_process`` automatically — pass
    ``processor=utils.default_process`` explicitly if you want that.
    """
    a = _apply_processor(s1, processor)
    b = _apply_processor(s2, processor)
    if not a and not b:
        return _clamp(0.0, score_cutoff)
    kernel_cutoff = None if score_cutoff is None else score_cutoff / 100.0
    sim = float(_sa.indel_normalized_score(a, b, score_cutoff=kernel_cutoff))
    return _clamp(sim * 100.0, score_cutoff)


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
