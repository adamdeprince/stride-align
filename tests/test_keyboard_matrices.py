"""Tests for the keyboard typo/confusion matrices (stride_align.matrices.keyboard).

These exercise the builder, the query=misspelled orientation, the `.T`
reversal, and the loader with synthetic data, so they do not depend on the
bundled matrices being generated.
"""

from __future__ import annotations

import numpy as np
import pytest

import stride_align as sa
from stride_align.matrices import SubstitutionMatrix, keyboard


def _cell(m: SubstitutionMatrix, a: str, b: str) -> int:
    return int(m.matrix[ord(a), ord(b)])


def test_ascii_alphabet_and_wildcard() -> None:
    assert len(keyboard.ASCII_ALPHABET) == 128
    assert keyboard.ASCII_ALPHABET == "".join(chr(c) for c in range(128))
    assert keyboard.ASCII_WILDCARD == chr(127)


def test_from_confusion_counts_builds_valid_matrix() -> None:
    counts = {("x", "y"): 90, ("y", "x"): 10, ("a", "b"): 50, ("e", "r"): 30}
    m = keyboard.from_confusion_counts(counts, name="synthetic")
    assert isinstance(m, SubstitutionMatrix)
    assert m.alphabet == keyboard.ASCII_ALPHABET
    assert m.matrix.shape == (128, 128)
    assert m.matrix.dtype == np.int8
    assert m.matrix.flags["C_CONTIGUOUS"]
    assert m.wildcard == chr(127)
    assert m.name == "synthetic"


def test_orientation_is_asymmetric_typed_then_intended() -> None:
    # Keys are (typed, intended); 'x'->'y' is common in only one direction.
    counts = {("x", "y"): 90, ("y", "x"): 10, ("a", "b"): 40, ("b", "a"): 40}
    m = keyboard.from_confusion_counts(counts)
    # m[typed][intended] is direction-sensitive: the two orderings differ.
    assert _cell(m, "x", "y") != _cell(m, "y", "x")


def test_identity_diagonal_dominates() -> None:
    counts = {("x", "y"): 90, ("y", "x"): 50, ("a", "b"): 40, ("e", "r"): 30}
    m = keyboard.from_confusion_counts(counts, match_margin=4)
    off_max = int(m.matrix[~np.eye(128, dtype=bool)].max())
    diag = np.diag(m.matrix)
    assert (diag == diag[0]).all()        # uniform identity diagonal
    assert int(diag[0]) > off_max          # an exact match beats any substitution


def test_transpose_reverses_orientation() -> None:
    counts = {("x", "y"): 90, ("y", "x"): 10, ("a", "b"): 40}
    fwd = keyboard.from_confusion_counts(counts, name="fwd")
    rev = SubstitutionMatrix(
        name="rev",
        alphabet=fwd.alphabet,
        matrix=np.ascontiguousarray(fwd.matrix.T),
        gap_score=fwd.gap_score,
        wildcard=fwd.wildcard,
    )
    # Transpose swaps query<->target: rev[b][a] == fwd[a][b].
    assert _cell(rev, "y", "x") == _cell(fwd, "x", "y")
    assert _cell(rev, "x", "y") == _cell(fwd, "y", "x")
    assert np.array_equal(rev.matrix, fwd.matrix.T)


def test_from_confusion_counts_accepts_array() -> None:
    grid = np.zeros((128, 128), dtype=np.float64)
    grid[ord("x"), ord("y")] = 90
    grid[ord("y"), ord("x")] = 10
    m = keyboard.from_confusion_counts(grid)
    assert m.matrix.shape == (128, 128)
    assert m.matrix.dtype == np.int8


def test_from_confusion_counts_rejects_wrong_array_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        keyboard.from_confusion_counts(np.zeros((10, 10)))


def test_from_npy_round_trip(tmp_path) -> None:
    counts = {("x", "y"): 90, ("a", "b"): 40, ("e", "r"): 30}
    m = keyboard.from_confusion_counts(counts)
    path = tmp_path / "synthetic.npy"
    np.save(path, m.matrix)
    loaded = keyboard.from_npy(path)
    assert loaded.name == "synthetic"          # default name == file stem
    assert np.array_equal(loaded.matrix, m.matrix)
    assert loaded.matrix.flags["C_CONTIGUOUS"]
    assert loaded.matrix.dtype == np.int8


def test_from_npy_transpose(tmp_path) -> None:
    counts = {("x", "y"): 90, ("y", "x"): 10}
    m = keyboard.from_confusion_counts(counts)
    path = tmp_path / "m.npy"
    np.save(path, m.matrix)
    t = keyboard.from_npy(path, transpose=True)
    assert np.array_equal(t.matrix, m.matrix.T)
    assert t.matrix.flags["C_CONTIGUOUS"]


def test_lazy_unknown_matrix_raises() -> None:
    with pytest.raises(AttributeError, match="no bundled matrix"):
        getattr(keyboard, "nonexistent_layout_xyz")


def test_available_returns_list() -> None:
    assert isinstance(keyboard.available(), list)


def test_matrix_flows_through_kernel() -> None:
    counts = {("t", "h"): 50, ("h", "t"): 10, ("e", "r"): 30}
    m = keyboard.from_confusion_counts(counts)
    # Identical strings: NW score == sum of the (uniform) diagonal matches.
    score = sa.needleman_wunsch_score("hello", "hello", matrix=m)
    assert score == int(np.diag(m.matrix)[0]) * len("hello")


# --- Bundled matrices (qwerty ships in-tree). --------------------------------
# Guarded so a stripped build without the data files skips rather than fails.


def test_bundled_matrices_present_and_valid() -> None:
    if "qwerty" not in keyboard.available():
        pytest.skip("bundled keyboard matrices not present in this build")
    for name in ("qwerty",):
        m = getattr(keyboard, name)
        assert isinstance(m, SubstitutionMatrix)
        assert m.matrix.shape == (128, 128)
        assert m.matrix.dtype == np.int8
        assert m.matrix.flags["C_CONTIGUOUS"]
        off_max = int(m.matrix[~np.eye(128, dtype=bool)].max())
        assert int(np.diag(m.matrix)[0]) > off_max     # identity dominates


def test_bundled_qwerty_captures_keyboard_adjacency() -> None:
    if "qwerty" not in keyboard.available():
        pytest.skip("bundled qwerty matrix not present")
    q = keyboard.qwerty

    def cell(a: str, b: str) -> int:
        return int(q.matrix[ord(a), ord(b)])

    # Physically adjacent keys are plausible typos and score above a far-apart
    # pair (which sits at/near the unobserved floor).
    assert cell("n", "m") > cell("z", "m")
    assert cell("r", "t") > cell("z", "m")
