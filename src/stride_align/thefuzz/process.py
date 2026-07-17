"""Selection helpers compatible with ``thefuzz.process`` 0.22.1."""

from __future__ import annotations

import logging
from functools import partial

from stride_align.thefuzz import fuzz, utils

_logger = logging.getLogger(__name__)

default_scorer = fuzz.WRatio
default_processor = utils.full_process

_DEFAULT_PROCESS_SCORERS = {
    fuzz.WRatio,
    fuzz.QRatio,
    fuzz.token_set_ratio,
    fuzz.token_sort_ratio,
    fuzz.partial_token_set_ratio,
    fuzz.partial_token_sort_ratio,
    fuzz.UWRatio,
    fuzz.UQRatio,
}


def _get_processor(processor, scorer):
    if scorer not in _DEFAULT_PROCESS_SCORERS:
        return processor

    force_ascii = scorer not in (fuzz.UWRatio, fuzz.UQRatio)
    pre_processor = partial(utils.full_process, force_ascii=force_ascii)
    if not processor or processor == utils.full_process:
        return pre_processor

    def wrapper(value):
        return pre_processor(processor(value))

    return wrapper


def _preprocess_query(query, processor):
    processed_query = processor(query) if processor else query
    if len(processed_query) == 0:
        _logger.warning(
            "Applied processor reduces input query to empty string, "
            "all comparisons will have score 0. [Query: '%s']",
            query,
        )
    return processed_query


def _validate_cutoff(score_cutoff, lowered: bool):
    if score_cutoff is None:
        return 0
    if lowered:
        try:
            outside = score_cutoff < 0 or score_cutoff > 100
        except TypeError:
            raise TypeError(f"must be real number, not {type(score_cutoff).__name__}") from None
        if outside:
            raise TypeError("score_cutoff has to be in the range of 0.0 - 100.0")
    return score_cutoff


def _choice_iter(choices):
    if hasattr(choices, "items"):
        return True, iter(choices.items())
    return False, enumerate(choices)


def _prepared_matches(query, choices, processor, scorer, score_cutoff):
    lowered = scorer in fuzz._RAW_SCORERS
    cutoff = _validate_cutoff(score_cutoff, lowered)
    query = _preprocess_query(query, processor)
    effective_processor = _get_processor(processor, scorer)
    if effective_processor is not None:
        query = effective_processor(query)

    raw_scorer = fuzz._RAW_SCORERS.get(scorer, scorer)
    is_mapping, choices_iter = _choice_iter(choices)
    for key, choice in choices_iter:
        if fuzz._is_none(choice):
            continue
        processed_choice = (
            effective_processor(choice) if effective_processor is not None else choice
        )
        raw_score = raw_scorer(query, processed_choice)
        if raw_score < cutoff:
            continue
        score = int(round(raw_score)) if lowered else raw_score
        yield is_mapping, choice, score, key, raw_score


def extractWithoutOrder(  # noqa: N802
    query,
    choices,
    processor=default_processor,
    scorer=default_scorer,
    score_cutoff=0,
):
    """Yield qualifying matches in input order."""
    for is_mapping, choice, score, key, _raw_score in _prepared_matches(
        query, choices, processor, scorer, score_cutoff
    ):
        yield (choice, score, key) if is_mapping else (choice, score)


def extract(
    query,
    choices,
    processor=default_processor,
    scorer=default_scorer,
    limit=5,
):
    return extractBests(
        query,
        choices,
        processor=processor,
        scorer=scorer,
        limit=limit,
    )


def _coerce_limit(limit):
    if limit is None:
        return None
    if isinstance(limit, (str, bytes)):
        raise TypeError("an integer is required")
    value = int(limit)
    if value < 0:
        # This is the exception exposed by TheFuzz 0.22.1's RapidFuzz
        # backend for a negative result-vector size.
        raise RuntimeError("vector")
    return value


def extractBests(  # noqa: N802
    query,
    choices,
    processor=default_processor,
    scorer=default_scorer,
    score_cutoff=0,
    limit=5,
):
    """Return the highest-scoring matches, preserving input order on ties."""
    limit = _coerce_limit(limit)
    matches = list(_prepared_matches(query, choices, processor, scorer, score_cutoff))
    matches.sort(key=lambda match: match[4], reverse=True)
    if limit is not None:
        matches = matches[:limit]
    return [
        (choice, score, key) if is_mapping else (choice, score)
        for is_mapping, choice, score, key, _raw_score in matches
    ]


def extractOne(  # noqa: N802
    query,
    choices,
    processor=default_processor,
    scorer=default_scorer,
    score_cutoff=0,
):
    """Return the first highest-scoring match, or ``None``."""
    best = None
    for match in _prepared_matches(query, choices, processor, scorer, score_cutoff):
        if best is None or match[4] > best[4]:
            best = match
        if match[4] == 100:
            break
    if best is None:
        return None
    is_mapping, choice, score, key, _raw_score = best
    return (choice, score, key) if is_mapping else (choice, score)


def dedupe(contains_dupes, threshold=70, scorer=fuzz.token_set_ratio):
    """Collapse fuzzy duplicates, retaining the longest representative."""
    deduped = set()
    for item in contains_dupes:
        matches = extractBests(
            item,
            contains_dupes,
            scorer=scorer,
            score_cutoff=threshold,
            limit=None,
        )
        deduped.add(max(matches, key=lambda match: (len(match[0]), match[0]))[0])
    return list(deduped) if len(deduped) != len(contains_dupes) else contains_dupes


__all__ = [
    "default_scorer",
    "default_processor",
    "extractWithoutOrder",
    "extract",
    "extractBests",
    "extractOne",
    "dedupe",
]
