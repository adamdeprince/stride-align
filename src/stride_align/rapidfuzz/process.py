"""``rapidfuzz.process`` shim — extract / extractOne / cdist."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np

import stride_align as _sa
from stride_align.rapidfuzz import fuzz as _fuzz


def _materialize_choices(choices) -> Tuple[list, list]:
    """Return ``(items, keys)``. If ``choices`` is a mapping, ``items``
    is the values (the strings to score) and ``keys`` is the matching
    keys; otherwise ``keys`` is the integer index range."""
    if isinstance(choices, dict):
        keys = list(choices.keys())
        items = list(choices.values())
    else:
        items = list(choices)
        keys = list(range(len(items)))
    return items, keys


def _score_one(query, choice, scorer, processor, scorer_kwargs):
    q = query if processor is None else processor(query)
    c = choice if processor is None else processor(choice)
    return scorer(q, c, **(scorer_kwargs or {}))


def extract(
    query,
    choices,
    *,
    scorer: Callable = _fuzz.WRatio,
    processor: Optional[Callable] = None,
    limit: Optional[int] = 5,
    score_cutoff: Optional[float] = None,
    score_hint=None,
    scorer_kwargs: Optional[dict] = None,
) -> List[Tuple]:
    """Return up to ``limit`` ``(choice, score, key)`` tuples sorted
    by score (descending for similarity scorers, ascending for
    distance scorers — matches rapidfuzz's behaviour by inferring
    direction from the score field name of the scorer module).

    ``choices`` may be a sequence or a mapping; with a mapping, the
    third tuple element is the matching key instead of an integer
    index.
    """
    items, keys = _materialize_choices(choices)
    higher_is_better = _scorer_is_similarity(scorer)
    results: List[Tuple] = []
    for item, key in zip(items, keys):
        if item is None:
            continue
        score = _score_one(query, item, scorer, processor, scorer_kwargs)
        if score_cutoff is not None:
            if higher_is_better and score < score_cutoff:
                continue
            if not higher_is_better and score > score_cutoff:
                continue
        results.append((item, score, key))
    results.sort(key=lambda r: r[1], reverse=higher_is_better)
    if limit is not None:
        results = results[: int(limit)]
    return results


def extractOne(  # noqa: N802
    query,
    choices,
    *,
    scorer: Callable = _fuzz.WRatio,
    processor: Optional[Callable] = None,
    score_cutoff: Optional[float] = None,
    score_hint=None,
    scorer_kwargs: Optional[dict] = None,
) -> Optional[Tuple]:
    """Return the single best ``(choice, score, key)`` tuple or
    ``None`` if no choice clears ``score_cutoff``."""
    top = extract(
        query, choices,
        scorer=scorer, processor=processor, limit=1,
        score_cutoff=score_cutoff, score_hint=score_hint,
        scorer_kwargs=scorer_kwargs,
    )
    return top[0] if top else None


def extract_iter(
    query,
    choices,
    *,
    scorer: Callable = _fuzz.WRatio,
    processor: Optional[Callable] = None,
    score_cutoff: Optional[float] = None,
    score_hint=None,
    scorer_kwargs: Optional[dict] = None,
):
    """Streaming version of ``extract`` — yields every ``(choice,
    score, key)`` tuple that clears ``score_cutoff``, in input order
    (no sorting)."""
    items, keys = _materialize_choices(choices)
    higher_is_better = _scorer_is_similarity(scorer)
    for item, key in zip(items, keys):
        if item is None:
            continue
        score = _score_one(query, item, scorer, processor, scorer_kwargs)
        if score_cutoff is not None:
            if higher_is_better and score < score_cutoff:
                continue
            if not higher_is_better and score > score_cutoff:
                continue
        yield (item, score, key)


def cdist(
    queries,
    choices,
    *,
    scorer: Callable = _fuzz.ratio,
    processor: Optional[Callable] = None,
    score_cutoff: Optional[float] = None,
    score_hint=None,
    score_multiplier: float = 1,
    dtype=None,
    workers: int = 1,
    **kwargs,
) -> np.ndarray:
    """All-pairs ``(len(queries), len(choices))`` score matrix.

    For now this runs a Python loop calling ``scorer`` per pair; the
    fast-path that routes to ``sa.cdist`` for built-in shim scorers
    is a follow-up. Functionally equivalent to upstream for arbitrary
    callable scorers.
    """
    queries_list = list(queries)
    choices_list = list(choices)
    out_dtype = dtype or np.float64
    if not queries_list or not choices_list:
        return np.empty((len(queries_list), len(choices_list)), dtype=out_dtype)

    result = np.empty((len(queries_list), len(choices_list)), dtype=out_dtype)
    for i, q in enumerate(queries_list):
        qp = q if processor is None else processor(q)
        for j, c in enumerate(choices_list):
            cp = c if processor is None else processor(c)
            score = scorer(qp, cp) * score_multiplier
            if score_cutoff is not None and score < score_cutoff:
                score = 0
            result[i, j] = score
    return result


def _scorer_is_similarity(scorer: Callable) -> bool:
    """Infer whether a scorer is similarity (higher-is-better) or
    distance (lower-is-better) from its source module + name."""
    module = getattr(scorer, "__module__", "") or ""
    name = getattr(scorer, "__name__", "") or ""
    if module.endswith(".fuzz") or "fuzz" in module:
        return True
    if "similarity" in name or "ratio" in name.lower():
        return True
    if "distance" in name:
        return False
    return True


__all__ = ["extract", "extractOne", "extract_iter", "cdist"]
