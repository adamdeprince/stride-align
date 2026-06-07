"""``sa.cdist`` with Smith-Waterman and Needleman-Wunsch scorers.

SW and NW dispatch through a Python-level row loop in ``cdist`` —
the C++ cdist machinery doesn't thread the SW / NW scoring parameters
(match_score, mismatch_score, gap_open_score, gap_extend_score,
width) through its per-row dispatch. The per-row kernels themselves
(``smith_waterman_scores``, ``needleman_wunsch_scores``, and their
``*_normalized_scores`` siblings) are the same SIMD Farrar / generic
kernels exercised in ``tests/test_alignment_unicode.py`` and
``tests/test_unicode_wide.py``; this file covers the cdist wrapping:
matrix shape, dtype, threading consistency, scorer-arg resolution,
parameter forwarding, and Unicode.
"""

from __future__ import annotations

import numpy as np
import pytest

import stride_align as sa


# ---------------------------------------------------------------- shape / dtype

class TestShapeAndDtype:
    """``cdist`` returns ``int64`` for raw-score scorers and
    ``float64`` for normalised scorers."""

    def test_sw_raw_returns_int64_matrix(self) -> None:
        result = sa.cdist(["hello", "world"], ["hallo", "word", "help"],
                          scorer=sa.Scorer.SMITH_WATERMAN)
        assert result.shape == (2, 3)
        assert result.dtype == np.int64

    def test_sw_normalized_returns_float64_matrix(self) -> None:
        result = sa.cdist(["hello", "world"], ["hallo", "word", "help"],
                          scorer=sa.Scorer.SMITH_WATERMAN_NORMALIZED)
        assert result.shape == (2, 3)
        assert result.dtype == np.float64

    def test_nw_raw_returns_int64_can_be_negative(self) -> None:
        result = sa.cdist(["abc"], ["xyz"], scorer=sa.Scorer.NEEDLEMAN_WUNSCH)
        assert result.dtype == np.int64
        # NW with no overlap is -|a|-|b|.
        assert int(result[0, 0]) < 0

    def test_nw_normalized_returns_float64_in_unit_interval(self) -> None:
        result = sa.cdist(["abc", "def"], ["abc", "abd", "xyz"],
                          scorer=sa.Scorer.NEEDLEMAN_WUNSCH_NORMALIZED)
        assert result.dtype == np.float64
        assert ((result >= 0.0) & (result <= 1.0)).all()


# ---------------------------------------------------------------- correctness

class TestRowConsistency:
    """Every cdist row equals the per-row ``*_scores`` direct call."""

    @pytest.mark.parametrize("scorer, per_row_fn", [
        (sa.Scorer.SMITH_WATERMAN,            sa.smith_waterman_scores),
        (sa.Scorer.SMITH_WATERMAN_NORMALIZED, sa.smith_waterman_normalized_scores),
        (sa.Scorer.NEEDLEMAN_WUNSCH,          sa.needleman_wunsch_scores),
        (sa.Scorer.NEEDLEMAN_WUNSCH_NORMALIZED, sa.needleman_wunsch_normalized_scores),
    ], ids=lambda x: getattr(x, "name", x.__name__ if callable(x) else str(x)))
    def test_each_row_matches_direct_call(self, scorer, per_row_fn) -> None:
        queries = ["hello", "world", "kitten"]
        targets = ["hallo", "word", "sitting", "help"]
        matrix = sa.cdist(queries, targets, scorer=scorer)
        for i, q in enumerate(queries):
            expected = per_row_fn(q, targets)
            assert np.array_equal(matrix[i], expected), f"row {i} mismatch"

    def test_diagonal_self_similarity_is_one_for_normalized_sw(self) -> None:
        seq = ["hello", "world", "你好"]
        matrix = sa.cdist(seq, seq, scorer=sa.Scorer.SMITH_WATERMAN_NORMALIZED)
        assert np.allclose(np.diag(matrix), 1.0)

    def test_smith_waterman_is_symmetric(self) -> None:
        # SW is commutative in (a, b).
        seq = ["hello", "hallo", "world", "wirld"]
        matrix = sa.cdist(seq, seq, scorer=sa.Scorer.SMITH_WATERMAN_NORMALIZED)
        assert np.allclose(matrix, matrix.T)


# ---------------------------------------------------------------- scorer arg

class TestScorerArgumentResolution:
    """The ``scorer=`` argument accepts the enum value, the module-
    level ``*_scores`` function, and the Farrar variant."""

    @pytest.mark.parametrize("scorer", [
        sa.Scorer.SMITH_WATERMAN,
        sa.smith_waterman_scores,
        sa.smith_waterman_farrar_scores,
    ], ids=["enum", "scores_fn", "farrar_scores_fn"])
    def test_smith_waterman_scorer_arg_equivalents(self, scorer) -> None:
        result = sa.cdist(["hello"], ["hallo"], scorer=scorer)
        assert result.dtype == np.int64
        # 4 matches + 1 mismatch = 4*2 + (-1) = 7
        assert int(result[0, 0]) == 7

    @pytest.mark.parametrize("scorer", [
        sa.Scorer.SMITH_WATERMAN_NORMALIZED,
        sa.smith_waterman_normalized_scores,
        sa.smith_waterman_farrar_normalized_scores,
    ], ids=["enum", "normalized_fn", "farrar_normalized_fn"])
    def test_smith_waterman_normalized_scorer_arg_equivalents(self, scorer) -> None:
        result = sa.cdist(["hello"], ["hallo"], scorer=scorer)
        assert result.dtype == np.float64

    def test_needleman_wunsch_enum_and_function(self) -> None:
        r_enum = sa.cdist(["abc"], ["abd"], scorer=sa.Scorer.NEEDLEMAN_WUNSCH)
        r_fn   = sa.cdist(["abc"], ["abd"], scorer=sa.needleman_wunsch_scores)
        assert np.array_equal(r_enum, r_fn)


# ---------------------------------------------------------------- parameters

class TestScoringParameters:
    """``match_score`` / ``mismatch_score`` / ``gap_score`` /
    ``gap_open_score`` / ``gap_extend_score`` / ``width`` reach the
    per-row kernel."""

    def test_custom_match_mismatch_changes_score(self) -> None:
        # 'AGCT' vs 'AGGT' under match=5, mismatch=-3, gap=-1.
        # Direct alignment AGCT/AGGT scores 5+5-3+5 = 12, but SW
        # prefers the gapped alignment AG__T / AG_GT (3 matches +
        # 2 single gaps = 5+5-1-1+5 = 13) because two single gaps
        # cost less than one mismatch. The kernel returns the true
        # optimum — checking that scoring parameters reach it.
        result = sa.cdist(["AGCT"], ["AGGT"], scorer=sa.Scorer.SMITH_WATERMAN,
                          match_score=5, mismatch_score=-3, gap_score=-1)
        assert int(result[0, 0]) == 13
        # Default parameters (match=2, mismatch=-1, gap=-1): direct
        # alignment scores 2+2-1+2 = 5, and the gapped alternative is
        # also 2+2-1-1+2 = 4, so the kernel returns 5.
        default = sa.cdist(["AGCT"], ["AGGT"], scorer=sa.Scorer.SMITH_WATERMAN)
        assert int(default[0, 0]) == 5
        # Sanity: the score with custom params is strictly different
        # — proves the kwargs are reaching the kernel.
        assert int(result[0, 0]) != int(default[0, 0])

    def test_affine_gap_routes_through(self) -> None:
        # No crash and a positive SW score for clearly-similar inputs.
        r = sa.cdist(["AAAACCCCGGGG"], ["AAAACCCCGGGG"],
                     scorer=sa.Scorer.SMITH_WATERMAN,
                     gap_open_score=-5, gap_extend_score=-1)
        # Self-similarity: 12 matches * 2 = 24.
        assert int(r[0, 0]) == 24

    def test_width_kwarg_accepted(self) -> None:
        r = sa.cdist(["hello"], ["hallo"], scorer=sa.Scorer.SMITH_WATERMAN, width=32)
        # Forced width=32 must give the same answer as default.
        r_default = sa.cdist(["hello"], ["hallo"], scorer=sa.Scorer.SMITH_WATERMAN)
        assert int(r[0, 0]) == int(r_default[0, 0])


# ---------------------------------------------------------------- threading

class TestThreading:
    """Multi-threaded cdist matches single-threaded cdist."""

    @pytest.mark.parametrize("scorer", [
        sa.Scorer.SMITH_WATERMAN,
        sa.Scorer.SMITH_WATERMAN_NORMALIZED,
        sa.Scorer.NEEDLEMAN_WUNSCH,
        sa.Scorer.NEEDLEMAN_WUNSCH_NORMALIZED,
    ], ids=lambda s: s.name)
    def test_cpu_count_2_matches_cpu_count_1(self, scorer) -> None:
        # Per memory rule: tests cap cpu_count at 2 so they don't
        # steal cores from concurrent benchmarks.
        queries = ["hello", "world", "kitten", "你好"] * 5
        targets = ["hallo", "word", "sitting", "你好啊"] * 5
        r1 = sa.cdist(queries, targets, scorer=scorer, cpu_count=1)
        r2 = sa.cdist(queries, targets, scorer=scorer, cpu_count=2)
        assert np.array_equal(r1, r2)


# ---------------------------------------------------------------- tqdm

class TestTqdmProgress:
    """The optional ``tqdm`` factory receives one update per query
    row."""

    def test_tqdm_factory_called_with_total_and_per_row_updates(self) -> None:
        captured = {}

        class FakeBar:
            def __init__(self, total):
                captured["total"] = total
                self.updates = []
            def update(self, n):
                self.updates.append(n)
            def close(self):
                captured["closed"] = True

        bar = None
        def factory(total):
            nonlocal bar
            bar = FakeBar(total)
            return bar

        sa.cdist(["a", "b", "c"], ["x", "y"],
                 scorer=sa.Scorer.SMITH_WATERMAN, tqdm=factory, cpu_count=1)
        assert captured["total"] == 3
        assert sum(bar.updates) == 3
        assert captured.get("closed") is True


# ---------------------------------------------------------------- edge cases

class TestEdgeCases:
    """Empty inputs, bytes inputs, mixed-kind targets."""

    def test_empty_queries_returns_zero_row_matrix(self) -> None:
        r = sa.cdist([], ["a", "b"], scorer=sa.Scorer.SMITH_WATERMAN)
        assert r.shape == (0, 2)

    def test_empty_targets_returns_zero_col_matrix(self) -> None:
        r = sa.cdist(["a"], [], scorer=sa.Scorer.SMITH_WATERMAN)
        assert r.shape == (1, 0)

    def test_both_empty(self) -> None:
        r = sa.cdist([], [], scorer=sa.Scorer.SMITH_WATERMAN_NORMALIZED)
        assert r.shape == (0, 0)
        assert r.dtype == np.float64

    def test_bytes_inputs_accepted(self) -> None:
        r = sa.cdist([b"hello", b"world"], [b"hallo", b"word"],
                     scorer=sa.Scorer.SMITH_WATERMAN)
        assert r.shape == (2, 2)
        assert int(r[0, 0]) == 7  # 'hello' vs 'hallo'


# ---------------------------------------------------------------- Unicode

class TestUnicode:
    """Every Unicode storage kind reaches the row kernel correctly."""

    @pytest.mark.parametrize("queries, targets", [
        (["Müller"],         ["Mueller"]),                # 1B Latin-1
        (["你好", "世界"],     ["你好啊", "再见"]),           # 2B CJK
        (["hi 👋"],          ["ho 👋", "🌸"]),             # 4B emoji
        (["你好", "hi 👋"],   ["你好啊", "hi 🌸"]),          # mixed 2B + 4B
    ], ids=["latin1", "cjk-2b", "emoji-4b", "mixed-2b-4b"])
    def test_unicode_inputs_round_trip(self, queries, targets) -> None:
        # No exception, right shape, every row matches direct call.
        matrix = sa.cdist(queries, targets, scorer=sa.Scorer.SMITH_WATERMAN_NORMALIZED)
        assert matrix.shape == (len(queries), len(targets))
        for i, q in enumerate(queries):
            assert np.array_equal(
                matrix[i], sa.smith_waterman_normalized_scores(q, targets),
            )

    def test_cjk_self_similarity_is_one(self) -> None:
        seq = ["你好", "世界", "你好世界"]
        matrix = sa.cdist(seq, seq, scorer=sa.Scorer.SMITH_WATERMAN_NORMALIZED)
        assert np.allclose(np.diag(matrix), 1.0)


# ---------------------------------------------------------------- contract

class TestEnumContract:
    """New ``Scorer`` integer IDs are 12-15 (the next four slots after
    the existing 0-11) and have user-visible names."""

    def test_new_enum_values_are_12_through_15(self) -> None:
        assert int(sa.Scorer.SMITH_WATERMAN) == 12
        assert int(sa.Scorer.SMITH_WATERMAN_NORMALIZED) == 13
        assert int(sa.Scorer.NEEDLEMAN_WUNSCH) == 14
        assert int(sa.Scorer.NEEDLEMAN_WUNSCH_NORMALIZED) == 15

    def test_enum_includes_all_existing_plus_four(self) -> None:
        # If this fails after adding a Scorer entry, also update the
        # ``Scorer`` enum in src/cpp/topk.hpp to match.
        assert len(list(sa.Scorer)) == 16
