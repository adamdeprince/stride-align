"""Tests for substitution-matrix-mode SW / NW alignment."""

from __future__ import annotations

import numpy as np
import pytest

import stride_align as sa
from stride_align.matrices import (
    SubstitutionMatrix,
    blosum45,
    blosum50,
    blosum62,
    blosum80,
    blosum90,
    pam30,
    pam70,
    pam250,
)


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
    # The mutex check fires when match_score / mismatch_score differ from
    # their built-in defaults (2 / -1). Passing an explicit value equal
    # to the default is indistinguishable from leaving them out (normal
    # Python kwarg semantics), and is silently allowed alongside matrix=.
    with pytest.raises(TypeError, match="mutually exclusive"):
        sa.smith_waterman_score("HE", "HE", matrix=blosum62, match_score=5)
    with pytest.raises(TypeError, match="mutually exclusive"):
        sa.smith_waterman_score("HE", "HE", matrix=blosum62, mismatch_score=-3)
    with pytest.raises(TypeError, match="mutually exclusive"):
        sa.needleman_wunsch_score("HE", "HE", matrix=blosum62, match_score=5)


def test_matrix_affine_gaps_supported() -> None:
    # Affine gaps in matrix mode used to raise NotImplementedError. They now
    # route to the affine matrix kernel — verify it runs and produces the
    # standard NCBI BLOSUM62 affine score for a known peptide pair.
    # H:H=8, E:E=5, no gaps inserted → 13 (same as linear case).
    score = sa.smith_waterman_score(
        "HE", "HE", matrix=blosum62, gap_open_score=-11, gap_extend_score=-1
    )
    assert score == 13


def test_matrix_affine_gaps_distinguish_open_extend() -> None:
    # Pick a pair where the optimal alignment contains a gap so the
    # difference between gap_open and gap_extend actually matters.
    # HEAGAW vs HEW — best SW alignment inserts a 3-AA gap in target.
    # With open=-100 extend=-1 the gap penalty is much higher than
    # with open=-1 extend=-1 (= linear -1).
    q, t = "HEAGAW", "HEW"
    cheap_gap = sa.smith_waterman_score(
        q, t, matrix=blosum62, gap_open_score=-1, gap_extend_score=-1
    )
    expensive_open = sa.smith_waterman_score(
        q, t, matrix=blosum62, gap_open_score=-100, gap_extend_score=-1
    )
    # Cheap-gap config must score strictly higher than expensive-open
    # config (it can use the gap, expensive-open can't afford to).
    assert cheap_gap > expensive_open


def test_all_builtin_matrices_have_canonical_shape() -> None:
    # Every shipped BLOSUM / PAM matrix uses the NCBI 24-letter protein
    # alphabet and an int8 symmetric matrix.
    import numpy as np
    matrices = [blosum45, blosum50, blosum62, blosum80, blosum90, pam30, pam70, pam250]
    for m in matrices:
        assert m.alphabet == "ARNDCQEGHILKMFPSTWYVBZX*"
        assert m.matrix.shape == (24, 24)
        assert m.matrix.dtype == np.int8
        assert np.array_equal(m.matrix, m.matrix.T), f"{m.name} not symmetric"
        assert m.gap_open is not None and m.gap_open < 0
        assert m.gap_extend is not None and m.gap_extend < 0


def test_pam250_known_diagonal() -> None:
    # NCBI PAM250 reference values.
    assert int(pam250.matrix[pam250.alphabet.index("W"), pam250.alphabet.index("W")]) == 17
    assert int(pam250.matrix[pam250.alphabet.index("C"), pam250.alphabet.index("C")]) == 12


def test_blosum90_known_diagonal() -> None:
    # NCBI BLOSUM90 reference values.
    assert int(blosum90.matrix[blosum90.alphabet.index("W"), blosum90.alphabet.index("W")]) == 11
    assert int(blosum90.matrix[blosum90.alphabet.index("H"), blosum90.alphabet.index("H")]) == 8


def test_from_ncbi_text_round_trip() -> None:
    # Parse a tiny NCBI-style matrix.
    text = """
    # comment line
       A  B  C
    A  4 -1 -2
    B -1  5  0
    C -2  0  9
    """
    m = SubstitutionMatrix.from_ncbi_text(
        text, name="tiny", gap_open=-5, gap_extend=-1, wildcard="C"
    )
    assert m.alphabet == "ABC"
    assert m.matrix.shape == (3, 3)
    assert int(m.matrix[0, 1]) == -1
    assert int(m.matrix[2, 2]) == 9
    assert m.gap_open == -5 and m.gap_extend == -1


def test_from_ncbi_text_rejects_row_header_mismatch() -> None:
    text = """
       A  B
    A  1  2
    X  3  4
    """
    with pytest.raises(ValueError, match="row labels do not match"):
        SubstitutionMatrix.from_ncbi_text(text)


def test_cdist_matrix_local_global() -> None:
    queries = ["HEAGAWGHEE", "PAWHEAE"]
    targets = ["HE", "AWGHE", "PAWHEAE"]
    sw = sa.cdist(queries, targets, matrix=blosum62,
                  gap_open_score=-11, gap_extend_score=-1, scorer="sw")
    nw = sa.cdist(queries, targets, matrix=blosum62,
                  gap_open_score=-11, gap_extend_score=-1, scorer="nw")
    assert sw.shape == (2, 3)
    assert sw.dtype == np.int64
    # Spot-check the symmetric case: aligning PAWHEAE to itself.
    assert sw[1, 2] == sa.smith_waterman_score(
        "PAWHEAE", "PAWHEAE", matrix=blosum62,
        gap_open_score=-11, gap_extend_score=-1
    )
    assert nw[1, 2] == sa.needleman_wunsch_score(
        "PAWHEAE", "PAWHEAE", matrix=blosum62,
        gap_open_score=-11, gap_extend_score=-1
    )


def test_cdist_matrix_passes_function_reference_scorer() -> None:
    queries = ["HE", "WW"]
    targets = ["HE", "WW"]
    sw = sa.cdist(queries, targets, matrix=blosum62, gap_score=-4,
                  scorer=sa.smith_waterman_scores)
    nw = sa.cdist(queries, targets, matrix=blosum62, gap_score=-4,
                  scorer=sa.needleman_wunsch_scores)
    assert sw.shape == (2, 2)
    assert nw.shape == (2, 2)


def test_cdist_top_k_matrix() -> None:
    queries = ["HEAGAWGHEE", "PAWHEAE", "MEEPS", "WW"]
    targets = ["HE", "AWGHE", "PAWHEAE", "XQQQ"]
    top = sa.cdist_top_k(
        queries, targets, matrix=blosum62,
        gap_open_score=-11, gap_extend_score=-1,
        scorer="sw", k=3,
    )
    assert len(top) == 3
    # All entries are (score, query, target) tuples.
    for score, q, t in top:
        assert isinstance(score, int)
        assert q in queries
        assert t in targets
    # The best pair should be PAWHEAE × PAWHEAE.
    best = max(top, key=lambda x: x[0])
    assert best[1] == "PAWHEAE" and best[2] == "PAWHEAE"


def test_cdist_above_threshold_matrix() -> None:
    queries = ["HEAGAWGHEE", "PAWHEAE"]
    targets = ["AWGHE", "PAWHEAE", "XXX"]
    matches = list(sa.cdist_above_threshold(
        queries, targets, matrix=blosum62,
        gap_open_score=-11, gap_extend_score=-1,
        scorer="sw", threshold=15,
    ))
    # Every score must be >= threshold.
    for score, q, t in matches:
        assert score >= 15
    # The self-match PAWHEAE × PAWHEAE clearly clears 15.
    assert any(q == "PAWHEAE" and t == "PAWHEAE" for _, q, t in matches)


def test_cdist_matrix_rejects_unknown_scorer() -> None:
    with pytest.raises(ValueError, match="matrix-mode cdist scorer"):
        sa.cdist(["A"], ["A"], matrix=blosum62, scorer="hamming")


def test_matrix_path_returns_aligned_strings() -> None:
    # Trivial pair where the optimal SW alignment has no gaps.
    r = sa.smith_waterman_path("HE", "HE", matrix=blosum62,
                               gap_open_score=-11, gap_extend_score=-1)
    assert r.score == 13
    assert r.aligned_query == "HE"
    assert r.aligned_target == "HE"
    assert r.operations == "=="
    assert r.query_start == 0 and r.query_end == 2


def test_matrix_path_with_gaps() -> None:
    # A pair where SW inserts a gap. We don't assert which optimal path
    # the kernel picks (multiple may share the optimum); we assert the
    # reconstructed aligned strings are consistent with the operations.
    r = sa.smith_waterman_path("HEAGAWGHEE", "PAWHEAE", matrix=blosum62,
                               gap_open_score=-11, gap_extend_score=-1)
    # Reconstruction consistency: every '=' position has matching chars,
    # every 'X' has differing chars, every 'D'/'I' has a gap on the
    # matching side.
    for q_c, t_c, op in zip(r.aligned_query, r.aligned_target, r.operations):
        if op == "=":
            assert q_c == t_c
        elif op == "X":
            assert q_c != t_c and q_c != "-" and t_c != "-"
        elif op == "D":
            assert t_c == "-"
        elif op == "I":
            assert q_c == "-"


def test_matrix_path_info_returns_cigar() -> None:
    p = sa.smith_waterman_path_info("HE", "HE", matrix=blosum62,
                                    gap_open_score=-11, gap_extend_score=-1)
    assert p.score == 13
    assert p.cigar == "2="
    assert p.matches == 2
    assert p.aligned_length == 2


def test_matrix_cigar_shortcut() -> None:
    cigar = sa.smith_waterman_cigar("HE", "HE", matrix=blosum62,
                                    gap_open_score=-11, gap_extend_score=-1)
    assert cigar == "2="


def test_matrix_nw_path_full_length() -> None:
    # NW always traverses both sequences end-to-end, so the operations
    # string covers the full pair.
    r = sa.needleman_wunsch_path("HE", "HEW", matrix=blosum62,
                                 gap_open_score=-11, gap_extend_score=-1)
    assert r.query_start == 0 and r.query_end == 2
    assert r.target_start == 0 and r.target_end == 3
    # Aligned strings cover the full target (3) with the query padded.
    assert len(r.aligned_target) == 3
    assert len(r.aligned_query) == 3
    assert "-" in r.aligned_query


def test_matrix_path_bytes_input() -> None:
    qb = blosum62.encode("HE")
    tb = blosum62.encode("HE")
    r = sa.smith_waterman_path(qb, tb, matrix=blosum62,
                               gap_open_score=-11, gap_extend_score=-1)
    assert r.score == 13
    # Bytes in → bytes out.
    assert isinstance(r.aligned_query, (bytes, bytearray))


def test_matrix_affine_batch_matches_single() -> None:
    q = "HEAGAWGHEE"
    targets = ["PAWHEAE", "HEAGAWGHEE", "WW", ""]
    single = [
        sa.smith_waterman_score(q, t, matrix=blosum62, gap_open_score=-11, gap_extend_score=-1)
        for t in targets
    ]
    batch = sa.smith_waterman_scores(
        q, targets, matrix=blosum62, gap_open_score=-11, gap_extend_score=-1
    )
    assert list(batch) == single


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


# Matrix-mode kernels read each input element as a uint8 index into the
# matrix alphabet. uint8/int8 ndarrays are bytes-compatible and work
# directly; wider dtypes (int16/int32/float64/...) would silently
# misinterpret their multi-byte representation as multiple indices.
# Reject those at the dispatcher boundary so users get a clear error.

def test_matrix_uint8_ndarray_accepted() -> None:
    arr = np.array([0, 1, 2, 3], dtype=np.uint8)
    matrix = SubstitutionMatrix(
        name="tiny",
        alphabet="ABCD",
        matrix=np.eye(4, dtype=np.int8) * 2 - np.ones((4, 4), dtype=np.int8),
        gap_score=-1,
        wildcard="A",
    )
    score = sa.smith_waterman_score(
        arr, arr, matrix=matrix, gap_score=-1,
    )
    # Identity self-alignment: diagonal entries are 1 (eye*2 - ones gives
    # 1 on the diagonal, -1 off-diagonal); 4 matches * 1 = 4.
    assert score == 4


@pytest.mark.parametrize("dtype", [np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64])
def test_matrix_wide_ndarray_rejected(dtype) -> None:
    arr = np.array([0, 1, 2, 3], dtype=dtype)
    matrix = SubstitutionMatrix(
        name="tiny",
        alphabet="ABCD",
        matrix=np.eye(4, dtype=np.int8),
        gap_score=-1,
        wildcard="A",
    )
    with pytest.raises(TypeError, match="uint8/int8"):
        sa.smith_waterman_score(arr, arr, matrix=matrix, gap_score=-1)


def test_matrix_wide_ndarray_rejected_in_batch() -> None:
    matrix = SubstitutionMatrix(
        name="tiny",
        alphabet="ABCD",
        matrix=np.eye(4, dtype=np.int8),
        gap_score=-1,
        wildcard="A",
    )
    query = np.array([0, 1, 2, 3], dtype=np.uint8)
    targets = [
        np.array([0, 1, 2, 3], dtype=np.uint8),
        np.array([0, 1, 2, 3], dtype=np.int32),  # wide; should reject.
    ]
    with pytest.raises(TypeError, match="uint8/int8"):
        sa.smith_waterman_scores(
            query, targets, matrix=matrix, gap_score=-1,
        )


# --------------------------------------------------------------------
# Matrix-mode cell-width selection (channel width)
#
# The matrix-mode kernel picks its cell type (int8 / int16 / int32 /
# int64) from ``query+target × max(|matrix entry|, |gap|)``, mirroring
# the match/mismatch path's ``step_limit × path_length`` calculation
# in ``preprocess.hpp::compute_score_bound``. ``SubstitutionMatrix``
# caches the matrix max-abs once at construction so the selector
# doesn't have to scan the matrix every call (the C++ side still does
# its own bound calculation per batch, but the cached Python attribute
# lets Python-side decision-making and downstream code skip the scan).
# --------------------------------------------------------------------


def test_max_abs_cached_on_construction() -> None:
    # The standard NCBI matrices have known max-abs values: BLOSUM62
    # entries range over [-4, 11] so max-abs is 11.
    assert blosum62.max_abs == 11
    # BLOSUM45 has wider range (entries up to +15).
    assert blosum45.max_abs == 15
    # PAM250 entries range over [-8, +17] -> 17.
    assert pam250.max_abs == 17


def test_max_abs_recomputed_per_distinct_matrix() -> None:
    # Two matrices with different entries get different cached values.
    big = SubstitutionMatrix(
        name="big", alphabet="ACGTX",
        matrix=np.array([
            [100, -50, -50, -50, -50],
            [-50, 100, -50, -50, -50],
            [-50, -50, 100, -50, -50],
            [-50, -50, -50, 100, -50],
            [-50, -50, -50, -50,   0],
        ], dtype=np.int8),
        gap_score=-10,
    )
    small = SubstitutionMatrix(
        name="small", alphabet="ACGTX",
        matrix=np.eye(5, dtype=np.int8),
        gap_score=-1,
    )
    assert big.max_abs == 100
    assert small.max_abs == 1


def test_max_abs_includes_negative_entries() -> None:
    # |min| > |max| should still flow through.
    m = SubstitutionMatrix(
        name="neg-heavy", alphabet="AX",
        matrix=np.array([[1, -127], [-127, 0]], dtype=np.int8),
        gap_score=-1,
    )
    assert m.max_abs == 127


def test_score_step_limit_defaults_to_matrix_own_gaps() -> None:
    # No args -> uses self.gap_score (and gap_open / gap_extend when
    # present on the matrix). BLOSUM62 has gap_open=-11 / gap_extend=-1.
    assert blosum62.score_step_limit() == max(blosum62.max_abs, 11, 1)


def test_score_step_limit_with_linear_gap_kwarg() -> None:
    # max(max_abs=11, |gap|=20) == 20.
    assert blosum62.score_step_limit(gap_score=-20) == 20
    # max(11, |gap|=4) == 11.
    assert blosum62.score_step_limit(gap_score=-4) == 11


def test_score_step_limit_with_affine_kwargs() -> None:
    # max(max_abs=11, |open|=50, |extend|=2) == 50.
    assert blosum62.score_step_limit(gap_open=-50, gap_extend=-2) == 50


def test_score_step_limit_predicts_kernel_score_bound() -> None:
    # The step_limit × (|q| + |t|) gives the absolute-score upper
    # bound the cell-width selector compares against int8.max=127,
    # int16.max=32767, int32.max=2^31-1.
    m = SubstitutionMatrix(
        name="big-self", alphabet="ACGTX",
        matrix=np.diag([100, 100, 100, 100, 0]).astype(np.int8),
        gap_score=-10,
    )
    # A*5 vs A*5: step_limit=100, |q|+|t|=10, bound=1000. int8 won't
    # hold 1000 (max 127) but int16 will (max 32767). The kernel must
    # use int16-wide cells; correctness check (matches diagonal sum).
    assert sa.smith_waterman_score("AAAAA", "AAAAA", matrix=m) == 5 * 100
    # A*400 vs A*400: bound = 100 * 800 = 80000. int16 won't hold
    # (max 32767); must use int32. Correctness still pinned to the
    # diagonal sum.
    assert sa.smith_waterman_score("A" * 400, "A" * 400, matrix=m) == 400 * 100


def test_score_step_limit_affine_pinned_correctness() -> None:
    # Same logic for affine: max_abs=100 driving the step_limit
    # produces the right int16 / int32 cell selection. NW global
    # alignment on identical inputs equals diagonal sum.
    m = SubstitutionMatrix(
        name="big-affine", alphabet="ACGTX",
        matrix=np.diag([100, 100, 100, 100, 0]).astype(np.int8),
        gap_score=-10,
    )
    assert sa.needleman_wunsch_score(
        "A" * 400, "A" * 400, matrix=m,
        gap_open_score=-15, gap_extend_score=-3,
    ) == 400 * 100


def test_max_abs_appears_in_repr() -> None:
    # The cached value is a dataclass field so it surfaces in repr —
    # makes the cached step_limit visible when debugging.
    m = SubstitutionMatrix(
        name="t", alphabet="AX",
        matrix=np.array([[5, -3], [-3, 5]], dtype=np.int8),
        gap_score=-1,
    )
    assert "max_abs=5" in repr(m)


# --------------------------------------------------------------------
# Matrix-mode cached bytes — per-call .tobytes() elimination
#
# The matrix-mode dispatchers in stride_align/__init__.py used to call
# matrix.matrix.tobytes() once per Python entry, allocating a fresh
# bytes object each time. For 24x24 BLOSUM62 that's a 576-byte malloc;
# for the planned 128x128 text matrices it's a 16 KB malloc whose
# allocation cost adds up under cdist over large corpora. Cache the
# bytes once at construction and let the dispatchers reuse the same
# buffer for every call against this matrix.
# --------------------------------------------------------------------


def test_matrix_bytes_cached_once() -> None:
    # Two reads return the same Python object (no fresh allocation).
    m = SubstitutionMatrix(
        name="t", alphabet="AX",
        matrix=np.array([[5, -3], [-3, 5]], dtype=np.int8),
        gap_score=-1,
    )
    assert m.matrix_bytes is m.matrix_bytes


def test_matrix_bytes_matches_tobytes() -> None:
    # The cached bytes equal a fresh tobytes() of the underlying array.
    assert blosum62.matrix_bytes == blosum62.matrix.tobytes()


def test_matrix_bytes_size_matches_matrix() -> None:
    # row-major int8: len(matrix_bytes) == stride * stride.
    assert len(blosum62.matrix_bytes) == blosum62.stride * blosum62.stride
    assert len(blosum62.matrix_bytes) == blosum62.matrix.nbytes


def test_matrix_bytes_independent_per_matrix() -> None:
    # Two distinct matrices have distinct cached byte buffers.
    m1 = SubstitutionMatrix(
        name="a", alphabet="AX",
        matrix=np.array([[1, 0], [0, 1]], dtype=np.int8), gap_score=-1,
    )
    m2 = SubstitutionMatrix(
        name="b", alphabet="AX",
        matrix=np.array([[2, 0], [0, 2]], dtype=np.int8), gap_score=-1,
    )
    assert m1.matrix_bytes is not m2.matrix_bytes
    assert m1.matrix_bytes != m2.matrix_bytes


def test_matrix_bytes_round_trip_through_dispatcher() -> None:
    # Score-only matrix path consumes the cached bytes. Result must
    # match the analytical answer (5 matches on the diagonal = 5*5).
    m = SubstitutionMatrix(
        name="t", alphabet="ACGTX",
        matrix=np.diag([5, 5, 5, 5, 0]).astype(np.int8),
        gap_score=-1,
    )
    assert sa.smith_waterman_score("AAAAA", "AAAAA", matrix=m) == 5 * 5
