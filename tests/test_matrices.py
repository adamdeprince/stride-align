"""Tests for substitution-matrix-mode SW / NW alignment."""

from __future__ import annotations

import numpy as np
import pytest

import stride_align as sa
from stride_align.matrices import SubstitutionMatrix, blosum62


def _diag_sum(seq: str, matrix: SubstitutionMatrix) -> int:
    return sum(
        int(matrix.matrix[matrix.alphabet.index(c), matrix.alphabet.index(c)])
        for c in seq
    )


def test_blosum62_metadata() -> None:
    assert blosum62.name == "BLOSUM62"
    assert blosum62.stride == 24
    assert blosum62.matrix.shape == (24, 24)
    assert blosum62.matrix.dtype == np.int8
    # Symmetry: BLOSUM is a symmetric matrix.
    assert np.array_equal(blosum62.matrix, blosum62.matrix.T)
    # A spot-check against the canonical NCBI values.
    a_idx = blosum62.alphabet.index("A")
    r_idx = blosum62.alphabet.index("R")
    w_idx = blosum62.alphabet.index("W")
    assert int(blosum62.matrix[a_idx, a_idx]) == 4
    assert int(blosum62.matrix[w_idx, w_idx]) == 11
    assert int(blosum62.matrix[a_idx, r_idx]) == -1


def test_encode_roundtrip() -> None:
    enc = blosum62.encode("ACDEFGHIKLMNPQRSTVWY")
    expected_indices = [blosum62.alphabet.index(c) for c in "ACDEFGHIKLMNPQRSTVWY"]
    assert list(enc) == expected_indices


def test_encode_lowercase_uppercased() -> None:
    assert blosum62.encode("acdef") == blosum62.encode("ACDEF")


def test_encode_unknown_folds_to_wildcard() -> None:
    wildcard_idx = blosum62.alphabet.index("X")
    assert blosum62.encode("J?@") == bytes([wildcard_idx] * 3)


def test_encode_empty() -> None:
    assert blosum62.encode("") == b""


def test_self_alignment_sw_equals_diagonal_sum() -> None:
    peptide = "ACDEFGHIKLMNPQRSTVWY"
    expected = _diag_sum(peptide, blosum62)
    assert sa.smith_waterman_score(peptide, peptide, matrix=blosum62, gap_score=-4) == expected


def test_self_alignment_nw_equals_diagonal_sum() -> None:
    peptide = "ACDEFGHIKLMNPQRSTVWY"
    expected = _diag_sum(peptide, blosum62)
    assert sa.needleman_wunsch_score(peptide, peptide, matrix=blosum62, gap_score=-4) == expected


def test_short_pair_known_score() -> None:
    # H:H=8, E:E=5 — diagonal alignment with no gaps.
    assert sa.smith_waterman_score("HE", "HE", matrix=blosum62, gap_score=-4) == 13


def test_matrix_mutually_exclusive_with_match_mismatch() -> None:
    with pytest.raises(TypeError, match="mutually exclusive"):
        sa.smith_waterman_score("HE", "HE", matrix=blosum62, match_score=2)
    with pytest.raises(TypeError, match="mutually exclusive"):
        sa.smith_waterman_score("HE", "HE", matrix=blosum62, mismatch_score=-1)
    with pytest.raises(TypeError, match="mutually exclusive"):
        sa.needleman_wunsch_score("HE", "HE", matrix=blosum62, match_score=2)


def test_matrix_rejects_affine_gaps() -> None:
    with pytest.raises(NotImplementedError, match="linear gaps"):
        sa.smith_waterman_score("HE", "HE", matrix=blosum62, gap_open_score=-10, gap_extend_score=-1)


def test_matrix_rejects_bad_type() -> None:
    with pytest.raises(TypeError, match="SubstitutionMatrix"):
        sa.smith_waterman_score("HE", "HE", matrix=np.zeros((4, 4), dtype=np.int8))


def test_match_mismatch_back_compat_untouched() -> None:
    # Existing 5-arg callers must produce the same scores as before the refactor.
    assert sa.smith_waterman_score("HELLO", "HELLO", match_score=2, mismatch_score=-1, gap_score=-1) == 10
    assert sa.smith_waterman_score("HELLO", "WORLD") == 2  # L:L only
    assert sa.needleman_wunsch_score("HELLO", "HELLO") == 10


def test_substitution_matrix_score_helper() -> None:
    # The convenience .score() helper delegates to SW with this matrix's gap_score.
    assert blosum62.score("HE", "HE") == 13


def test_custom_matrix_two_letter_alphabet() -> None:
    # Build a 2-letter alphabet matrix equivalent to match=3 / mismatch=-2,
    # then check the matrix path agrees with the match/mismatch path.
    custom = SubstitutionMatrix(
        name="ABonly",
        alphabet="AB",
        matrix=np.array([[3, -2], [-2, 3]], dtype=np.int8),
        gap_score=-1,
        wildcard="A",
    )
    q, t = "AABB", "ABAB"
    matrix_score = sa.smith_waterman_score(q, t, matrix=custom, gap_score=-1)
    scalar_score = sa.smith_waterman_score(q, t, match_score=3, mismatch_score=-2, gap_score=-1)
    assert matrix_score == scalar_score


def test_substitution_matrix_validation_dtype() -> None:
    with pytest.raises(TypeError, match="int8"):
        SubstitutionMatrix(
            name="bad",
            alphabet="AB",
            matrix=np.zeros((2, 2), dtype=np.int32),
            gap_score=-1,
        )


def test_substitution_matrix_validation_shape() -> None:
    with pytest.raises(ValueError, match="alphabet size"):
        SubstitutionMatrix(
            name="bad",
            alphabet="ABC",
            matrix=np.zeros((2, 2), dtype=np.int8),
            gap_score=-1,
        )


def test_substitution_matrix_validation_wildcard_in_alphabet() -> None:
    with pytest.raises(ValueError, match="wildcard"):
        SubstitutionMatrix(
            name="bad",
            alphabet="AB",
            matrix=np.zeros((2, 2), dtype=np.int8),
            gap_score=-1,
            wildcard="X",
        )
