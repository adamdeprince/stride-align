"""Monge-Elkan multi-token hybrid similarity — Phase D.6.

For each token in ``s1``, find the best-matching token in ``s2`` by
an inner similarity, then average across ``s1``'s tokens:

    ME(s1, s2) = (1 / |T_a|) · Σ_{t ∈ T_a} max_{u ∈ T_b} sim(t, u)

Asymmetric by definition: ``ME(s1, s2) ≠ ME(s2, s1)`` in general.
The classic record-linkage form is the directional one above (Monge
& Elkan 1996); the ``symmetric=True`` variant averages both
directions when an order-independent score is wanted.

Pure-Python composition on top of stride-align's inner-similarity
primitives — no third-party code is imported into the production
path.
"""

from __future__ import annotations

from typing import Callable, Union


# Lazy import to avoid circular load: ``stride_align/__init__.py``
# re-exports the names this module defines.
def _resolve_inner(
    inner: Union[str, Callable[[str, str], float]],
) -> Callable[[str, str], float]:
    if callable(inner):
        return inner
    if not isinstance(inner, str):
        raise TypeError(
            "inner must be a callable or one of "
            "'jaro', 'jaro_winkler', 'levenshtein_ratio', 'indel_ratio'; "
            f"got {type(inner).__name__}"
        )
    import stride_align as sa
    mapping = {
        "jaro":               sa.jaro_similarity,
        "jaro_winkler":       sa.jaro_winkler_similarity,
        "levenshtein_ratio":  sa.levenshtein_normalized_score,
        "indel_ratio":        sa.indel_normalized_score,
    }
    try:
        return mapping[inner]
    except KeyError:
        raise ValueError(
            f"unknown inner similarity {inner!r}; expected one of "
            f"{sorted(mapping)} or a callable"
        ) from None


def _coerce_str(s: object) -> str:
    if isinstance(s, str):
        return s
    if isinstance(s, (bytes, bytearray)):
        return bytes(s).decode("latin-1")
    raise TypeError(
        f"monge_elkan inputs must be str or bytes, got {type(s).__name__}"
    )


def _tokenise(s: str) -> list[str]:
    return s.split()


def _directional(
    tokens_a: list[str],
    tokens_b: list[str],
    inner_fn: Callable[[str, str], float],
) -> float:
    # Caller has already handled the all-empty / one-empty edge cases.
    total = 0.0
    for t in tokens_a:
        best = 0.0
        for u in tokens_b:
            score = inner_fn(t, u)
            if score > best:
                best = score
                if best >= 1.0:
                    break
        total += best
    return total / len(tokens_a)


def monge_elkan(
    s1: object,
    s2: object,
    *,
    inner: Union[str, Callable[[str, str], float]] = "jaro",
    processor: Callable[[str], str] | None = None,
    symmetric: bool = False,
) -> float:
    """Monge-Elkan multi-token similarity.

    Tokenises both inputs on whitespace, then for each token in ``s1``
    picks the best-matching token in ``s2`` under the inner similarity
    and averages across ``s1``'s tokens.

    Parameters
    ----------
    s1, s2 : str | bytes
        Strings to compare. ``bytes`` is widened as Latin-1.
    inner : str | callable, default ``"jaro"``
        Inner per-token similarity. One of ``"jaro"``,
        ``"jaro_winkler"``, ``"levenshtein_ratio"``, ``"indel_ratio"``,
        or any ``Callable[[str, str], float]`` returning a value in
        ``[0, 1]``.
    processor : callable, optional
        Applied to both inputs before tokenisation (e.g.
        ``processor=str.lower`` for case-insensitive matching).
    symmetric : bool, default ``False``
        If ``True``, return the average of the two directional scores
        ``(ME(s1, s2) + ME(s2, s1)) / 2``. Useful when an order-
        independent score is wanted.

    Returns
    -------
    float
        Score in ``[0, 1]``. ``1.0`` when both inputs have no tokens
        (vacuously identical); ``0.0`` when exactly one side has no
        tokens.
    """
    inner_fn = _resolve_inner(inner)
    a = _coerce_str(s1)
    b = _coerce_str(s2)
    if processor is not None:
        a = processor(a)
        b = processor(b)

    tokens_a = _tokenise(a)
    tokens_b = _tokenise(b)

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    forward = _directional(tokens_a, tokens_b, inner_fn)
    if not symmetric:
        return forward
    backward = _directional(tokens_b, tokens_a, inner_fn)
    return 0.5 * (forward + backward)
