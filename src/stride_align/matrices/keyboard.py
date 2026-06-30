"""Keyboard typo / confusion substitution matrices.

These are example :class:`~stride_align.matrices.SubstitutionMatrix` objects
built from real human typing-error data — the Aalto University "136 Million
Keystrokes" dataset, used with the authors' permission (see ``NOTICE`` and
``docs/keyboard-matrix-external-sources.md``). They score how plausible it is
that one character was typed when another was meant, so an aligner can treat a
likely slip (``teh`` for ``the``) as a near-match instead of a flat mismatch.

Orientation — the query is the MISSPELLED side
===============================================

Every matrix here is oriented ``m[a][b]`` with **a = the query character and
b = the target character** — the same convention as every stride_align call,
where ``align(query, target)`` looks up ``m[query[i]][target[j]]``.

Concretely, for these keyboard matrices:

* **a (the query / row) is the character that was actually typed — the
  mistake.**
* **b (the target / column) is the character that was intended — the
  correction.**

The score is high when "typing ``a`` while meaning ``b``" is a common,
plausible slip. So feed the **misspelled / user-entered** string as the
``query`` and the **canonical / dictionary** string as the ``target``::

    import stride_align as sa
    from stride_align.matrices import keyboard

    # query = what the user typed (may contain typos); target = candidate word
    sa.smith_waterman_score("teh", "the", matrix=keyboard.qwerty)

Reversing the orientation with ``.T``
=====================================

These matrices are **asymmetric**: typing ``a`` when you meant ``b`` is not
equally likely as the reverse, so orientation matters (unlike the symmetric
BLOSUM/PAM matrices, where it never does). If your pipeline is the other way
round — the ``query`` is the *correct* string and the ``target`` is the
misspelled one — flip the matrix with NumPy's ``.T`` (transpose), which swaps
the axes to ``m[intended][misspelled]``::

    import numpy as np
    from stride_align.matrices import SubstitutionMatrix, keyboard

    fwd = keyboard.qwerty                            # m[misspelled][intended]
    rev = SubstitutionMatrix(
        name=fwd.name + ".T",
        alphabet=fwd.alphabet,
        matrix=np.ascontiguousarray(fwd.matrix.T),   # <-- .T flips the roles
        gap_score=fwd.gap_score,
        wildcard=fwd.wildcard,
    )

``fwd.matrix.T`` is the transpose; ``np.ascontiguousarray`` re-packs it into
the C-contiguous int8 buffer the kernel requires. (The loaders below also take
a ``transpose=True`` shortcut that does exactly this.)

Building your own
=================

:func:`from_confusion_counts` turns a ``{(typed, intended): count}`` mapping
into a log-odds matrix — see its docstring for the scoring. :func:`from_npy`
loads a matrix written by ``tools/build_keyboard_matrices.py``. The bundled
``keyboard.qwerty`` matrix is loaded lazily on first access; :func:`available`
lists which ones ship.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stride_align.matrices import SubstitutionMatrix

# Full single-byte ASCII alphabet; index == ord(char). chr(127) (DEL) is the
# wildcard slot that absorbs anything outside the alphabet — same convention
# as stride_align.matrices.ascii_text.
ASCII_ALPHABET = "".join(chr(c) for c in range(128))
ASCII_WILDCARD = chr(127)

# Bundled matrices live alongside this module (curated, published in-tree
# under Apache-2.0 with the dataset authors' permission). The directory may be
# absent until the matrices have been generated.
_DATA_DIR = Path(__file__).resolve().parent / "keyboard_data"


def _log_odds_grid(counts, *, scale=2.0, match_margin=4, floor=None):
    """Core scoring: an (N, N) count matrix -> int8 log-odds grid.

    ``score[a][b] = scale * log2( observed / expected )``
                  ``= scale * log2( N * C[a][b] / (R[a] * K[b]) )``

    where ``observed = C[a][b] / N``, ``expected = (R[a]/N)*(K[b]/N)``,
    ``R`` / ``K`` are the row / column margins and ``N`` the grand total.
    Pairs more common than chance score positive; rarer than chance, negative;
    never-observed pairs are floored. The identity diagonal is set above the
    strongest off-diagonal substitution so an exact match always wins.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.shape[0]
    total = counts.sum()
    if total == 0:
        grid = np.full((n, n), -1, dtype=np.int8)
        np.fill_diagonal(grid, 1)
        return grid

    row = counts.sum(axis=1, keepdims=True)   # R[a]: a as the typed key
    col = counts.sum(axis=0, keepdims=True)   # K[b]: b as the intended key
    with np.errstate(divide="ignore", invalid="ignore"):
        score = scale * np.log2((counts * total) / (row * col))

    finite = np.isfinite(score)
    if floor is None:
        floor = float(np.floor(score[finite].min())) if finite.any() else -1.0
    score[~finite] = floor

    grid = np.clip(np.rint(score), -128, 127).astype(np.int8)

    # Identity diagonal: correction data never has typed == intended, so set
    # it above the strongest substitution (exact match always beats any typo).
    np.fill_diagonal(grid, -128)              # exclude the diagonal from max()
    best_sub = int(grid.max())
    np.fill_diagonal(grid, min(127, max(best_sub, 0) + match_margin))
    return grid


def from_confusion_counts(
    counts,
    *,
    alphabet=ASCII_ALPHABET,
    name="KEYBOARD",
    scale=2.0,
    match_margin=4,
    floor=None,
    wildcard=None,
    gap_score=-1,
    gap_open=None,
    gap_extend=None,
):
    """Build a log-odds ``SubstitutionMatrix`` from confusion counts.

    ``counts`` is either a mapping ``{(typed, intended): count}`` (single-char
    keys) or an already-shaped ``(N, N)`` count array. Per the orientation
    above, the first element of each key is the **typed/mistake** char (row,
    query) and the second is the **intended/correction** char (column,
    target).

    See :func:`_log_odds_grid` for the scoring. ``scale`` is the log-odds
    multiplier (2.0 = BLOSUM-style half-bit units), ``match_margin`` how far
    the identity diagonal sits above the best substitution, and ``floor`` the
    score for never-observed pairs (``None`` = the most-negative observed
    score).
    """
    n = len(alphabet)
    if hasattr(counts, "shape"):
        grid_counts = np.asarray(counts, dtype=np.float64)
        if grid_counts.shape != (n, n):
            raise ValueError(
                f"counts array shape {grid_counts.shape} != ({n}, {n}) for the alphabet"
            )
    else:
        index = {ch: i for i, ch in enumerate(alphabet)}
        grid_counts = np.zeros((n, n), dtype=np.float64)
        for (typed, intended), c in counts.items():
            ia, ib = index.get(typed), index.get(intended)
            if ia is not None and ib is not None:
                grid_counts[ia, ib] = c

    grid = _log_odds_grid(grid_counts, scale=scale, match_margin=match_margin, floor=floor)
    return SubstitutionMatrix(
        name=name,
        alphabet=alphabet,
        matrix=grid,
        gap_score=gap_score,
        wildcard=wildcard or alphabet[-1],
        gap_open=gap_open,
        gap_extend=gap_extend,
    )


def from_npy(
    path,
    *,
    name=None,
    alphabet=ASCII_ALPHABET,
    wildcard=None,
    transpose=False,
    gap_score=-1,
    gap_open=None,
    gap_extend=None,
):
    """Load an int8 ``(N, N)`` ``.npy`` matrix into a ``SubstitutionMatrix``.

    Set ``transpose=True`` to flip the orientation (``.T``) so the query
    becomes the *intended* string and the target the *misspelled* one — see
    the module docstring.
    """
    path = Path(path)
    grid = np.load(path).astype(np.int8, copy=False)
    if transpose:
        grid = grid.T
    grid = np.ascontiguousarray(grid)
    if grid.shape != (len(alphabet), len(alphabet)):
        raise ValueError(
            f"{path.name}: shape {grid.shape} != ({len(alphabet)}, {len(alphabet)})"
        )
    return SubstitutionMatrix(
        name=name or path.stem,
        alphabet=alphabet,
        matrix=grid,
        gap_score=gap_score,
        wildcard=wildcard or alphabet[-1],
        gap_open=gap_open,
        gap_extend=gap_extend,
    )


def available():
    """Names of the bundled keyboard matrices that ship in this install."""
    if not _DATA_DIR.is_dir():
        return []
    return sorted(p.stem for p in _DATA_DIR.glob("*.npy"))


def __getattr__(name):
    # Lazily expose each bundled matrix as a module attribute, e.g.
    # ``keyboard.qwerty`` -> keyboard_data/qwerty.npy.
    path = _DATA_DIR / f"{name}.npy"
    if path.is_file():
        matrix = from_npy(path, name=f"keyboard:{name}")
        globals()[name] = matrix          # cache so it isn't reloaded
        return matrix
    raise AttributeError(
        f"module {__name__!r} has no bundled matrix {name!r}; "
        f"available: {available() or 'none — run tools/build_keyboard_matrices.py'}"
    )


def __dir__():
    return sorted({*globals(), *available()})


__all__ = [
    "ASCII_ALPHABET",
    "ASCII_WILDCARD",
    "from_confusion_counts",
    "from_npy",
    "available",
]
