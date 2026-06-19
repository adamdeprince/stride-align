"""``rapidfuzz.process`` shim — extract / extractOne / cdist."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np

import stride_align as _sa
from stride_align.rapidfuzz import distance as _distance
from stride_align.rapidfuzz import fuzz as _fuzz


# Map shim scorers -> (sa.cdist Scorer enum, sa.*_top_k function,
#                       score scale factor, higher_is_better).
# The shim scorers return values that need scaling/inversion to match
# what sa.cdist / sa.*_top_k produces for the corresponding Scorer
# enum. For rapidfuzz-style similarity (returned in [0, 100]) the
# scale factor applies to sa.cdist's [0, 1] normalized output.
# Distance scorers return integer counts that sa.cdist already
# produces directly.
_FAST_PATH_SCORERS: dict[Callable, Tuple["_sa.Scorer", Callable, float, bool]] = {
    # fuzz.* family — all similarity, return * 100
    _fuzz.ratio:            (_sa.Scorer.INDEL_NORMALIZED,
                             _sa.indel_normalized_top_k, 100.0, True),
    _fuzz.QRatio:           (_sa.Scorer.INDEL_NORMALIZED,
                             _sa.indel_normalized_top_k, 100.0, True),
    # distance.X.similarity / .distance / .normalized_*
    _distance.Levenshtein.distance:
        (_sa.Scorer.LEVENSHTEIN, _sa.levenshtein_top_k, 1.0, False),
    _distance.Levenshtein.normalized_similarity:
        (_sa.Scorer.LEVENSHTEIN_NORMALIZED, _sa.levenshtein_normalized_top_k, 1.0, True),
    _distance.Indel.distance:
        (_sa.Scorer.INDEL, _sa.indel_top_k, 1.0, False),
    _distance.Indel.normalized_similarity:
        (_sa.Scorer.INDEL_NORMALIZED, _sa.indel_normalized_top_k, 1.0, True),
    _distance.Hamming.distance:
        (_sa.Scorer.HAMMING, _sa.hamming_top_k, 1.0, False),
    _distance.Hamming.normalized_similarity:
        (_sa.Scorer.HAMMING_NORMALIZED, _sa.hamming_normalized_top_k, 1.0, True),
    _distance.Jaro.similarity:
        (_sa.Scorer.JARO, _sa.jaro_top_k, 1.0, True),
    _distance.JaroWinkler.similarity:
        (_sa.Scorer.JARO_WINKLER, _sa.jaro_winkler_top_k, 1.0, True),
    _distance.DamerauLevenshtein.distance:
        (_sa.Scorer.TRUE_DAMERAU_LEVENSHTEIN, _sa.true_damerau_levenshtein_top_k, 1.0, False),
    _distance.DamerauLevenshtein.normalized_similarity:
        (_sa.Scorer.TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED,
         _sa.true_damerau_levenshtein_normalized_top_k, 1.0, True),
    _distance.OSA.distance:
        (_sa.Scorer.DAMERAU_LEVENSHTEIN, _sa.damerau_levenshtein_top_k, 1.0, False),
    _distance.OSA.normalized_similarity:
        (_sa.Scorer.DAMERAU_LEVENSHTEIN_NORMALIZED,
         _sa.damerau_levenshtein_normalized_top_k, 1.0, True),
}


def _resolve_fast_path(scorer: Callable):
    """Return the (Scorer, top_k_fn, scale, higher_is_better) routing
    tuple for ``scorer`` if it's a recognised shim entry point, else
    ``None``."""
    return _FAST_PATH_SCORERS.get(scorer)


def _sort_join(s: str) -> str:
    """Whitespace-tokenise, lexicographically sort, single-space join —
    the token_sort preprocessing rapidfuzz applies before its Indel
    ratio. ASCII sort order matches the C++ kernel's byte sort."""
    return " ".join(sorted(s.split()))


# Preprocess-then-fast-path scorers. ``token_sort_ratio(a, b)`` is
# exactly ``indel_normalized(sort_join(a), sort_join(b)) * 100`` — a
# per-string preprocessing followed by the same Indel ratio the SIMD
# flip already accelerates. In a cdist we sort-join each input ONCE
# (O(N+M)) and run one batched INDEL_NORMALIZED matrix, instead of the
# per-pair slow loop that re-sorts both strings on every one of the
# N*M cells. Maps scorer -> (preprocess_fn, sa.Scorer, scale).
_PREPROCESS_FAST_PATH: dict[Callable, Tuple[Callable, "_sa.Scorer", float]] = {
    _fuzz.token_sort_ratio: (_sort_join, _sa.Scorer.INDEL_NORMALIZED, 100.0),
}


def _lcsseq_post(indel, la, lb, kind: str):
    """LCSseq scorers from the INDEL distance matrix + input lengths.
    ``LCS = (|a| + |b| - indel) / 2`` (Indel = |a|+|b|-2·LCS), then:
    similarity = LCS; distance = max(|a|,|b|) - LCS; normalized_distance
    = distance / max(|a|,|b|) (0 when both empty); normalized_similarity
    = 1 - normalized_distance. ``la``/``lb`` broadcast as (N,1)/(1,M)."""
    lcs = (la + lb - indel) / 2.0
    if kind == "similarity":
        return lcs
    mx = np.maximum(la, lb)
    dist = mx - lcs
    if kind == "distance":
        return dist
    # 0/0 (both inputs empty) -> 0.0 without evaluating the divide.
    nd = np.divide(dist, mx, out=np.zeros_like(dist), where=mx > 0)
    if kind == "normalized_distance":
        return nd
    return 1.0 - nd  # normalized_similarity


# Post-process-then-fast-path scorers: derivable from the INDEL flip
# matrix + input lengths via one vectorised numpy pass, instead of the
# per-pair Python loop. Maps scorer -> (kind, default dtype).
_POSTPROCESS_FAST_PATH: dict[Callable, Tuple[str, object]] = {
    _distance.LCSseq.distance:              ("distance", np.uint32),
    _distance.LCSseq.similarity:            ("similarity", np.uint32),
    _distance.LCSseq.normalized_distance:   ("normalized_distance", np.float32),
    _distance.LCSseq.normalized_similarity: ("normalized_similarity", np.float32),
}


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
    distance scorers).

    ``choices`` may be a sequence or a mapping; with a mapping, the
    third tuple element is the matching key instead of an integer
    index.

    Built-in shim scorers dispatch through ``sa.*_top_k`` — SIMD
    kernels with the length-pruning + adaptive-bound optimisations
    already shipped in stride-align. Arbitrary callable scorers fall
    back to a Python loop.
    """
    items, keys = _materialize_choices(choices)
    if not items:
        return []
    higher_is_better = _scorer_is_similarity(scorer)

    # Fast path: dispatch to sa.*_top_k for recognised shim scorers.
    fast = _resolve_fast_path(scorer)
    if fast is not None and not scorer_kwargs:
        _scorer_enum, top_k_fn, scale, fast_higher_is_better = fast
        # Pre-apply processor since sa.*_top_k has no processor kwarg.
        processed_query = query if processor is None else processor(query)
        processed_items = (items if processor is None
                           else [processor(c) for c in items])
        # rapidfuzz default limit is 5; sa top_k accepts k=None as
        # "all matches".
        if limit is None:
            k = len(processed_items)
        else:
            k = max(1, min(int(limit), len(processed_items)))
        raw = top_k_fn(processed_query, processed_items, k=k)
        # raw is [(target_str, score, target_index), ...] from sa
        # (matches the stride-align convention). Map back to the
        # original ``items`` / ``keys`` via the index, scale the
        # score for rapidfuzz's [0, 100] similarity convention,
        # apply the cutoff, then sort.
        results: List[Tuple] = []
        for _target_str, sa_score, idx in raw:
            choice = items[idx]
            key = keys[idx]
            score = float(sa_score) * scale
            if score_cutoff is not None:
                if fast_higher_is_better and score < score_cutoff:
                    continue
                if not fast_higher_is_better and score > score_cutoff:
                    continue
            results.append((choice, score, key))
        results.sort(key=lambda r: r[1], reverse=fast_higher_is_better)
        return results

    # Slow path: callable scorer not in the registry (e.g. fuzz.WRatio,
    # fuzz.partial_ratio, or arbitrary user callables). Per-pair
    # Python loop matches upstream's behaviour for these scorers.
    results = []
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
    workers: int = -1,
    **kwargs,
) -> np.ndarray:
    """All-pairs ``(len(queries), len(choices))`` score matrix.

    Built-in shim scorers (``fuzz.ratio``, ``distance.Levenshtein.distance``,
    ``distance.Jaro.similarity`` etc.) dispatch through ``sa.cdist``
    with the multi-threaded C++ kernel; arbitrary callable scorers
    fall back to a Python loop calling ``scorer`` per pair.

    ``workers`` defaults to ``-1`` (all cores) rather than upstream
    rapidfuzz's ``1`` — cdist is an embarrassingly parallel all-pairs
    matrix and the native kernel already gates parallelism by matrix
    size, so small matrices stay single-threaded automatically while
    large ones use every core. Results are identical regardless of the
    worker count. Pass ``workers=1`` to force single-threaded.
    ``workers <= 0`` maps to ``sa.cdist``'s ``cpu_count=0`` (all
    cores); a positive ``workers`` caps the thread count.
    """
    queries_list = list(queries)
    choices_list = list(choices)
    out_dtype = dtype
    if not queries_list or not choices_list:
        return np.empty(
            (len(queries_list), len(choices_list)),
            dtype=out_dtype or np.float64,
        )

    # Pre-apply the processor once per input — sa.cdist doesn't have a
    # processor kwarg, but the equivalent is "preprocess then cdist".
    if processor is not None:
        queries_list = [processor(q) for q in queries_list]
        choices_list = [processor(c) for c in choices_list]

    fast_path = _resolve_fast_path(scorer)
    if fast_path is not None:
        sa_scorer, _top_k_fn, scale, higher_is_better = fast_path
        # workers <= 0 -> cpu_count=0 (all cores); positive -> cap.
        cpu_count = 0 if int(workers) <= 0 else int(workers)
        sa_result = _sa.cdist(
            queries_list, choices_list,
            scorer=sa_scorer,
            cpu_count=cpu_count,
        )
        # rapidfuzz's default dtypes: float32 for similarity scorers,
        # uint32 for distance scorers. Match those so downstream code
        # that introspects ``result.dtype`` keeps working.
        if out_dtype is None:
            out_dtype = np.float32 if higher_is_better else np.uint32
        # Fast common case (distance scorers with no scaling/cutoff):
        # convert the native result straight to the target dtype,
        # skipping the float64 round-trip. The intermediate copy
        # otherwise dominates for cheap scorers like Indel on small
        # matrices.
        if scale == 1.0 and score_multiplier == 1 and score_cutoff is None:
            return sa_result.astype(out_dtype, copy=False)
        # Apply rapidfuzz's score scaling ([0, 1] -> [0, 100] for the
        # similarity scorers) and the user-supplied ``score_multiplier``.
        scaled = sa_result.astype(np.float64)
        if scale != 1.0:
            scaled = scaled * scale
        if score_multiplier != 1:
            scaled = scaled * score_multiplier
        if score_cutoff is not None:
            scaled = np.where(scaled >= score_cutoff, scaled, 0)
        return scaled.astype(out_dtype)

    # Preprocess-then-fast-path (e.g. token_sort_ratio): apply the
    # per-string preprocessing once, then route through the batched
    # SIMD scorer instead of the per-pair Python loop below.
    preprocess_fp = _PREPROCESS_FAST_PATH.get(scorer)
    if preprocess_fp is not None and not kwargs:
        pre_fn, sa_scorer, scale = preprocess_fp
        q_pre = [pre_fn(q) for q in queries_list]
        c_pre = [pre_fn(c) for c in choices_list]
        cpu_count = 0 if int(workers) <= 0 else int(workers)
        sa_result = _sa.cdist(
            q_pre, c_pre, scorer=sa_scorer, cpu_count=cpu_count,
        )
        if out_dtype is None:
            out_dtype = np.float32
        scaled = sa_result.astype(np.float64) * scale
        if score_multiplier != 1:
            scaled = scaled * score_multiplier
        if score_cutoff is not None:
            scaled = np.where(scaled >= score_cutoff, scaled, 0)
        return scaled.astype(out_dtype)

    # Post-process-then-fast-path (e.g. LCSseq.*): one batched INDEL
    # matrix, then a vectorised numpy transform using the input lengths
    # instead of the per-pair Python loop.
    postprocess_fp = _POSTPROCESS_FAST_PATH.get(scorer)
    if postprocess_fp is not None and not kwargs:
        kind, default_dtype = postprocess_fp
        cpu_count = 0 if int(workers) <= 0 else int(workers)
        indel = _sa.cdist(
            queries_list, choices_list,
            scorer=_sa.Scorer.INDEL, cpu_count=cpu_count,
        ).astype(np.float64)
        la = np.fromiter((len(q) for q in queries_list),
                         dtype=np.float64, count=len(queries_list))[:, None]
        lb = np.fromiter((len(c) for c in choices_list),
                         dtype=np.float64, count=len(choices_list))[None, :]
        out = _lcsseq_post(indel, la, lb, kind)
        if score_multiplier != 1:
            out = out * score_multiplier
        if score_cutoff is not None:
            out = np.where(out >= score_cutoff, out, 0)
        return out.astype(out_dtype if out_dtype is not None else default_dtype)

    # Slow path: arbitrary callable scorer, Python loop.
    out_dtype = out_dtype or np.float64
    result = np.empty((len(queries_list), len(choices_list)), dtype=out_dtype)
    for i, q in enumerate(queries_list):
        for j, c in enumerate(choices_list):
            score = scorer(q, c) * score_multiplier
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
