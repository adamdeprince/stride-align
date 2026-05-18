import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stride_align import (
    AlignmentPath,
    AlignmentResult,
    BackendKind,
    Scores,
    available_backends,
    backend_is_available,
    detect_best_backend,
    levenshtein_normalized_score,
    levenshtein_normalized_scores,
    levenshtein_score,
    levenshtein_scores,
    needleman_wunsch_cigar,
    needleman_wunsch_normalized_score,
    needleman_wunsch_normalized_scores,
    needleman_wunsch_path,
    needleman_wunsch_path_info,
    needleman_wunsch_score,
    needleman_wunsch_scores,
    needleman_wunsch_trace_cigar,
    needleman_wunsch_trade_cigar,
    smith_waterman_cigar,
    smith_waterman_farrar_normalized_score,
    smith_waterman_farrar_normalized_scores,
    smith_waterman_farrar_score,
    smith_waterman_farrar_scores,
    smith_waterman_normalized_score,
    smith_waterman_normalized_scores,
    smith_waterman_path,
    smith_waterman_path_info,
    smith_waterman_score,
    smith_waterman_scores,
    smith_waterman_trace_cigar,
    smith_waterman_trade_cigar,
)


def test_detect_best_backend_returns_enum() -> None:
    backend = detect_best_backend()
    assert isinstance(backend, BackendKind)


def test_available_backends_includes_generic() -> None:
    backends = available_backends()
    assert any(record.name == "generic" and record.available for record in backends)


def test_swar_backend_is_packaged_when_available() -> None:
    if backend_is_available(BackendKind.SWAR):
        assert importlib.util.find_spec("stride_align._swar") is not None


def test_swar_backend_is_not_auto_selected() -> None:
    assert detect_best_backend() is not BackendKind.SWAR


def test_backend_is_available_for_detected_backend() -> None:
    assert backend_is_available(detect_best_backend())


def test_scores_batch_api_matches_scalar_scores() -> None:
    query = "foo"
    targets = ["foo", "bar", "food"]

    sw = Scores(query).compare(targets)
    nw = Scores(query, variant="needleman_wunsch").compare(targets)
    farrar = Scores(query, variant="farrar").compare(targets)
    assert isinstance(sw, np.ndarray) and sw.dtype == np.int64
    assert sw.tolist() == [smith_waterman_score(query, target) for target in targets]
    assert nw.tolist() == [needleman_wunsch_score(query, target) for target in targets]
    assert farrar.tolist() == [smith_waterman_farrar_score(query, target) for target in targets]


def test_public_plural_score_functions_match_scalar_scores() -> None:
    query = "foo"
    targets = ["foo", "bar", "food"]

    sw = smith_waterman_scores(query, targets)
    nw = needleman_wunsch_scores(query, targets)
    farrar = smith_waterman_farrar_scores(query, targets)
    assert isinstance(sw, np.ndarray) and sw.dtype == np.int64
    assert isinstance(nw, np.ndarray) and nw.dtype == np.int64
    assert isinstance(farrar, np.ndarray) and farrar.dtype == np.int64
    assert sw.tolist() == [smith_waterman_score(query, target) for target in targets]
    assert nw.tolist() == [needleman_wunsch_score(query, target) for target in targets]
    assert farrar.tolist() == [smith_waterman_farrar_score(query, target) for target in targets]


def test_smith_waterman_normalized_score_perfect_match_is_one() -> None:
    assert smith_waterman_normalized_score("hello", "hello") == 1.0
    assert smith_waterman_normalized_score("hello", "say hello world") == 1.0


def test_smith_waterman_normalized_score_divides_by_shorter_length() -> None:
    # 3 matches at match_score=2 over a query of length 3: 6 / (3*2) == 1.0.
    assert smith_waterman_normalized_score("abc", "abcxyz", match_score=2) == 1.0
    # Half the query matches: 2*2 / (3*2) ≈ 0.6667.
    assert smith_waterman_normalized_score(
        "abc", "abXXX", match_score=2
    ) == pytest.approx(4 / 6)


def test_needleman_wunsch_normalized_score_divides_by_longer_length() -> None:
    # Identical strings: match_score=2, length 3 → 6 / (3*2) == 1.0.
    assert needleman_wunsch_normalized_score("abc", "abc", match_score=2) == 1.0
    # 3 matches + 3 trailing gaps: score = 6 + 3*(-1) = 3; denom = 6*2 = 12.
    assert needleman_wunsch_normalized_score(
        "abc", "abcxyz", match_score=2, gap_score=-1
    ) == pytest.approx(3 / 12)


def test_normalized_scores_match_per_pair_normalization() -> None:
    query = "foo"
    targets = ["foo", "bar", "food", "foobar"]

    sw_expected = [
        smith_waterman_normalized_score(query, target) for target in targets
    ]
    nw_expected = [
        needleman_wunsch_normalized_score(query, target) for target in targets
    ]
    farrar_expected = [
        smith_waterman_farrar_normalized_score(query, target) for target in targets
    ]

    sw_result = smith_waterman_normalized_scores(query, targets)
    nw_result = needleman_wunsch_normalized_scores(query, targets)
    farrar_result = smith_waterman_farrar_normalized_scores(query, targets)
    assert isinstance(sw_result, np.ndarray) and sw_result.dtype == np.float64
    np.testing.assert_allclose(sw_result, sw_expected)
    np.testing.assert_allclose(nw_result, nw_expected)
    np.testing.assert_allclose(farrar_result, farrar_expected)


def test_normalized_score_rejects_non_positive_match_score() -> None:
    with pytest.raises(ValueError, match="match_score must be positive"):
        smith_waterman_normalized_score("abc", "abc", match_score=0)
    with pytest.raises(ValueError, match="match_score must be positive"):
        needleman_wunsch_normalized_score("abc", "abc", match_score=-1)
    with pytest.raises(ValueError, match="match_score must be positive"):
        smith_waterman_farrar_normalized_scores("abc", ["abc"], match_score=0)


def test_normalized_score_clamps_negative_scores_to_zero() -> None:
    # Two disjoint short strings under NW: score is negative, but clamps to 0.
    assert needleman_wunsch_normalized_score(
        "aaa", "bbbbbb", match_score=2, mismatch_score=-5, gap_score=-5
    ) == 0.0


def test_normalized_score_handles_empty_inputs() -> None:
    assert smith_waterman_normalized_score("", "") == 1.0
    assert needleman_wunsch_normalized_score("", "") == 1.0
    # One empty side: SW denominator is min(0, n)*match == 0 → returns 0.0.
    assert smith_waterman_normalized_score("", "abc") == 0.0


def test_smith_waterman_farrar_normalized_score_matches_standard_normalized() -> None:
    # Farrar SW must produce the same final score as standard SW, so the
    # normalized variants should agree row-for-row.
    query = "needle"
    targets = ["needle", "n33dle", "haystack"]
    assert [smith_waterman_farrar_normalized_score(query, t) for t in targets] == [
        smith_waterman_normalized_score(query, t) for t in targets
    ]


def test_levenshtein_score_classic_examples() -> None:
    # The classic Levenshtein illustration: kitten -> sitting is 3 edits
    # (k->s, e->i, +g).
    assert levenshtein_score("kitten", "sitting") == 3
    assert levenshtein_score("", "") == 0
    assert levenshtein_score("abc", "") == 3
    assert levenshtein_score("", "abc") == 3
    assert levenshtein_score("abc", "abc") == 0
    assert levenshtein_score("flaw", "lawn") == 2


def test_levenshtein_normalized_score_bounds() -> None:
    assert levenshtein_normalized_score("foo", "foo") == 1.0
    assert levenshtein_normalized_score("", "") == 1.0
    assert levenshtein_normalized_score("abc", "xyz") == 0.0
    # 1 of 6 longer-length is an edit, so similarity is 5/6.
    assert levenshtein_normalized_score("foobar", "foobaz") == pytest.approx(5 / 6)


def test_levenshtein_scores_returns_int64_ndarray() -> None:
    result = levenshtein_scores("kitten", ["kitten", "sitting", "kit", ""])
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64
    assert result.tolist() == [0, 3, 3, 6]


def test_levenshtein_normalized_scores_returns_float64_ndarray() -> None:
    result = levenshtein_normalized_scores("foo", ["foo", "bar", "food"])
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.tolist() == pytest.approx([1.0, 0.0, 0.75])


def test_levenshtein_score_bytes_inputs() -> None:
    assert levenshtein_score(b"hello", b"hallo") == 1
    assert levenshtein_score(b"", b"abc") == 3


def test_levenshtein_score_long_pattern_uses_multi_word_myers() -> None:
    # Pattern length > 64 forces the Hyyrö multi-word Myers' path. Use a
    # 100-character query and check a known one-edit edit distance.
    query = "abcdefghij" * 10
    target = "xbcdefghij" + "abcdefghij" * 9
    assert len(query) == 100 and len(target) == 100
    assert levenshtein_score(query, target) == 1


def test_levenshtein_unicode_str_inputs() -> None:
    # Wide-codepoint inputs exercise the hashmap-PEQ generic path.
    assert levenshtein_score("café", "cafe") == 1
    assert levenshtein_score("🎉🎈", "🎉🎈") == 0
    assert levenshtein_score("🎉🎈", "🎈🎉") == 2


@pytest.mark.parametrize("backend_module,kind", [
    pytest.param("stride_align._sse41", BackendKind.X86_SSE41, id="sse41"),
    pytest.param("stride_align._avx2", BackendKind.X86_AVX2, id="avx2"),
    pytest.param("stride_align._avx512bwvl", BackendKind.X86_AVX512BWVL, id="avx512bwvl"),
])
def test_levenshtein_x86_simd_matches_scalar(backend_module: str, kind: BackendKind) -> None:
    if not backend_is_available(kind):
        pytest.skip(f"{kind.name} backend not available on this host")
    backend = importlib.import_module(backend_module)
    generic = importlib.import_module("stride_align._generic")

    rng = np.random.default_rng(0)
    alphabet = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz", dtype=np.uint8)
    for q_len in (3, 6, 16, 32, 50, 64):
        query = bytes(rng.choice(alphabet, size=q_len))
        targets = [bytes(rng.choice(alphabet, size=int(rng.integers(1, 80)))) for _ in range(50)]
        scalar = generic.levenshtein_scores(query, targets)
        simd = backend.levenshtein_scores(query, targets)
        np.testing.assert_array_equal(simd, scalar)


def test_scores_batch_api_rejects_single_string_target_collection() -> None:
    with pytest.raises(TypeError, match="targets must be an iterable"):
        Scores("foo").compare("bar")


def test_public_dispatch_uses_profile_simd_for_long_linear_scores(monkeypatch) -> None:
    import stride_align

    class FakeBackend:
        def __init__(self, name: str) -> None:
            self.name = name

        def smith_waterman_score(self, *args, **kwargs) -> str:
            return self.name

        def needleman_wunsch_score(self, *args, **kwargs) -> str:
            return self.name

    generic = FakeBackend("generic")
    avx2 = FakeBackend("avx2")
    monkeypatch.setattr(stride_align, "_GENERIC_BACKEND", generic)
    monkeypatch.setattr(
        stride_align,
        "_AVAILABLE_BACKENDS",
        {BackendKind.GENERIC: generic, BackendKind.X86_AVX2: avx2},
    )

    assert stride_align.smith_waterman_score("A" * 128, "A" * 128) == "avx2"
    assert stride_align.needleman_wunsch_score("A" * 128, "A" * 128) == "avx2"


def test_public_dispatch_uses_wide_simd_for_long_farrar(monkeypatch) -> None:
    import stride_align

    class FakeBackend:
        def __init__(self, name: str) -> None:
            self.name = name

        def smith_waterman_farrar_score(self, *args, **kwargs) -> str:
            return self.name

    generic = FakeBackend("generic")
    avx2 = FakeBackend("avx2")
    monkeypatch.setattr(stride_align, "_GENERIC_BACKEND", generic)
    monkeypatch.setattr(
        stride_align,
        "_AVAILABLE_BACKENDS",
        {BackendKind.GENERIC: generic, BackendKind.X86_AVX2: avx2},
    )

    assert stride_align.smith_waterman_farrar_score("A" * 128, "A" * 128) == "avx2"


def test_public_dispatch_allows_swar_for_short_linear_farrar(monkeypatch) -> None:
    import stride_align

    class FakeBackend:
        def __init__(self, name: str) -> None:
            self.name = name

        def smith_waterman_farrar_score(self, *args, **kwargs) -> str:
            return self.name

    generic = FakeBackend("generic")
    swar = FakeBackend("swar")
    monkeypatch.setattr(stride_align, "_GENERIC_BACKEND", generic)
    monkeypatch.setattr(
        stride_align,
        "_AVAILABLE_BACKENDS",
        {BackendKind.GENERIC: generic, BackendKind.SWAR: swar},
    )

    assert stride_align.smith_waterman_farrar_score("A" * 31, "A" * 31, width=8) == "swar"


def test_public_dispatch_prefers_narrow_simd_for_short_affine_farrar(monkeypatch) -> None:
    import stride_align

    class FakeBackend:
        def __init__(self, name: str) -> None:
            self.name = name

        def smith_waterman_farrar_score(self, *args, **kwargs) -> str:
            return self.name

    generic = FakeBackend("generic")
    sse41 = FakeBackend("sse41")
    avx2 = FakeBackend("avx2")
    monkeypatch.setattr(stride_align, "_GENERIC_BACKEND", generic)
    monkeypatch.setattr(
        stride_align,
        "_AVAILABLE_BACKENDS",
        {
            BackendKind.GENERIC: generic,
            BackendKind.X86_SSE41: sse41,
            BackendKind.X86_AVX2: avx2,
        },
    )

    assert (
        stride_align.smith_waterman_farrar_score(
            "A" * 31,
            "A" * 31,
            gap_open_score=-2,
            gap_extend_score=-1,
            width=8,
        )
        == "sse41"
    )


def test_needleman_wunsch_score_on_strings() -> None:
    assert needleman_wunsch_score("ACGT", "ACCT") == 5


def test_needleman_wunsch_path_on_strings() -> None:
    result = needleman_wunsch_path("ACGT", "ACCT")

    assert isinstance(result, AlignmentResult)
    assert result.score == 5
    assert result.query_start == 0
    assert result.query_end == 4
    assert result.target_start == 0
    assert result.target_end == 4
    assert result.aligned_query == "ACGT"
    assert result.aligned_target == "ACCT"
    assert result.operations == "==X="


def test_needleman_wunsch_path_info_on_strings() -> None:
    result = needleman_wunsch_path_info("ACGT", "ACCT")

    assert isinstance(result, AlignmentPath)
    assert result.score == 5
    assert result.query_start == 0
    assert result.query_end == 4
    assert result.target_start == 0
    assert result.target_end == 4
    assert result.operations == "==X="
    assert result.cigar == "2=1X1="
    assert result.matches == 3
    assert result.mismatches == 1
    assert result.insertions == 0
    assert result.deletions == 0
    assert result.aligned_length == 4
    assert not hasattr(result, "aligned_query")


def test_needleman_wunsch_cigar_apis_return_cigar_only() -> None:
    expected = needleman_wunsch_path_info("ACGT", "ACCT").cigar

    assert needleman_wunsch_cigar("ACGT", "ACCT") == expected
    assert needleman_wunsch_trace_cigar("ACGT", "ACCT") == expected
    assert needleman_wunsch_trade_cigar("ACGT", "ACCT") == expected


def test_string_fast_path_handles_wide_unicode() -> None:
    result = needleman_wunsch_path("A🙂", "A🙂")

    assert result.score == 4
    assert result.aligned_query == "A🙂"
    assert result.aligned_target == "A🙂"
    assert result.operations == "=="


def test_smith_waterman_score_on_strings() -> None:
    assert smith_waterman_score("ACCGT", "CCG") == 6


def test_smith_waterman_farrar_score_matches_standard_score() -> None:
    assert smith_waterman_farrar_score(b"GGCCTT", b"CGGTTAT") == smith_waterman_score(
        b"GGCCTT",
        b"CGGTTAT",
    )


def test_smith_waterman_farrar_compacts_wide_unicode_to_byte_tokens() -> None:
    assert smith_waterman_farrar_score("🙂🙃🙂", "🙃🙂", width=8) == 4


def test_score_fast_paths_for_zero_gap_lcs_case() -> None:
    query = "ABCBDAB"
    target = "BDCABA"

    assert (
        smith_waterman_score(query, target, match_score=2, mismatch_score=-1, gap_score=0)
        == 8
    )
    assert (
        needleman_wunsch_score(query, target, match_score=2, mismatch_score=-1, gap_score=0)
        == 8
    )
    assert (
        smith_waterman_farrar_score(query, target, match_score=2, mismatch_score=-1, gap_score=0)
        == 8
    )


def test_score_fast_paths_for_token_independent_scoring() -> None:
    assert needleman_wunsch_score("AB", "XYZ", match_score=1, mismatch_score=1, gap_score=-1) == 1
    assert smith_waterman_score("AA", "BBB", match_score=1, mismatch_score=1, gap_score=1) == 4
    assert (
        smith_waterman_farrar_score("AA", "BBB", match_score=1, mismatch_score=1, gap_score=1)
        == 4
    )


def test_affine_gap_scores_are_supported_by_public_api() -> None:
    query = "AAABBB"
    target = "AAACCCBBB"
    kwargs = {
        "match_score": 2,
        "mismatch_score": -1,
        "gap_score": -1,
        "gap_open_score": -3,
        "gap_extend_score": -1,
    }

    assert smith_waterman_score(query, target, **kwargs) == 7
    assert smith_waterman_farrar_score(query, target, **kwargs) == 7
    assert needleman_wunsch_score(query, target, **kwargs) == 7

    result = smith_waterman_path(query, target, **kwargs)

    assert result.score == 7
    assert result.aligned_query == "AAA---BBB"
    assert result.aligned_target == "AAACCCBBB"
    assert result.operations == "===III==="
    assert smith_waterman_cigar(query, target, **kwargs) == "3=3I3="
    assert smith_waterman_trace_cigar(query, target, **kwargs) == "3=3I3="
    assert smith_waterman_trade_cigar(query, target, **kwargs) == "3=3I3="
    assert needleman_wunsch_cigar(query, target, **kwargs) == "3=3I3="


def test_smith_waterman_path_on_bytes_returns_bytes() -> None:
    result = smith_waterman_path(b"ACCGT", b"CCG")

    assert isinstance(result, AlignmentResult)
    assert result.score == 6
    assert result.query_start == 1
    assert result.query_end == 4
    assert result.target_start == 0
    assert result.target_end == 3
    assert result.aligned_query == b"CCG"
    assert result.aligned_target == b"CCG"
    assert result.operations == "==="


def test_smith_waterman_path_info_on_bytes() -> None:
    result = smith_waterman_path_info(b"ACCGT", b"CCG")

    assert isinstance(result, AlignmentPath)
    assert result.score == 6
    assert result.query_start == 1
    assert result.query_end == 4
    assert result.target_start == 0
    assert result.target_end == 3
    assert result.operations == "==="
    assert result.cigar == "3="
    assert result.matches == 3
    assert result.mismatches == 0
    assert result.insertions == 0
    assert result.deletions == 0
    assert result.aligned_length == 3


def test_smith_waterman_cigar_apis_return_cigar_only() -> None:
    expected = smith_waterman_path_info(b"ACCGT", b"CCG").cigar

    assert smith_waterman_cigar(b"ACCGT", b"CCG") == expected
    assert smith_waterman_trace_cigar(b"ACCGT", b"CCG") == expected
    assert smith_waterman_trade_cigar(b"ACCGT", b"CCG") == expected


def _linear_cigar_score(
    cigar: str,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
) -> int:
    score = 0
    count = 0
    for character in cigar:
        if character.isdigit():
            count = count * 10 + int(character)
            continue
        if character == "=":
            score += count * match_score
        elif character == "X":
            score += count * mismatch_score
        elif character in {"I", "D"}:
            score += count * gap_score
        else:
            raise AssertionError(f"unexpected CIGAR operation {character!r}")
        count = 0
    assert count == 0
    return score


def test_smith_waterman_trade_cigar_trace_free_score_matches_score() -> None:
    cases = [
        (
            "the city wakes under a grey sky while people cross the station concourse",
            "the city wakes under a blue sky while people cross a quiet station concourse",
        ),
        (
            "清晨的城市慢慢醒来，行人穿过路口，语言结构也慢慢清楚",
            "清晨的城市慢慢亮起来，行人穿过路口，文字结构也慢慢清楚",
        ),
    ]

    for query, target in cases:
        cigar = smith_waterman_trade_cigar(query, target, width=16)
        assert _linear_cigar_score(cigar) == smith_waterman_score(query, target, width=16)


def test_direct_bytes_and_str_pair_raises_type_error() -> None:
    with pytest.raises(TypeError, match="bytes and str inputs"):
        needleman_wunsch_score(b"ABC", "ABC")


def test_sequence_inputs_are_serialized_and_return_tuples() -> None:
    left = (frozenset({1}), frozenset({2}), frozenset({3}))
    right = (frozenset({2}), frozenset({3}))

    result = smith_waterman_path(left, right)

    assert result.score == 4
    assert result.query_start == 1
    assert result.query_end == 3
    assert result.target_start == 0
    assert result.target_end == 2
    assert result.aligned_query == (frozenset({2}), frozenset({3}))
    assert result.aligned_target == (frozenset({2}), frozenset({3}))
    assert result.operations == "=="


def test_mixed_sequence_and_bytes_treats_bytes_as_sequence() -> None:
    result = needleman_wunsch_path([1, 2], b"\x01\x02")

    assert result.score == 4
    assert result.aligned_query == (1, 2)
    assert result.aligned_target == (1, 2)
    assert result.operations == "=="


def test_width_parameter_can_force_a_wider_kernel() -> None:
    result = smith_waterman_path("ACCGT", "CCG", width=64)

    assert result.score == 6
    assert result.aligned_query == "CCG"
    assert result.aligned_target == "CCG"
    assert result.operations == "==="


def test_width_parameter_rejects_narrower_kernel() -> None:
    with pytest.raises(ValueError, match="narrower"):
        needleman_wunsch_score("🙂", "🙂", width=8)


def test_width_parameter_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="None, 0, 8, 16, 32, or 64"):
        smith_waterman_score("AC", "AC", width=7)

    with pytest.raises(ValueError, match="None, 0, 8, 16, 32, or 64"):
        smith_waterman_farrar_score("AC", "AC", width=7)


def test_width_parameter_accepts_none_and_zero() -> None:
    assert smith_waterman_score("ACCGT", "CCG", width=None) == 6
    assert smith_waterman_score("ACCGT", "CCG", width=0) == 6


def test_file_compare_cli_reports_normalized_score(tmp_path, capsys) -> None:
    from stride_align.file_compare import main

    query = tmp_path / "query.txt"
    target = tmp_path / "target.txt"
    query.write_text("ACGT", encoding="utf-8")
    target.write_text("AGT", encoding="utf-8")

    exit_code = main([str(query), str(target), "--mode", "sw", "--simd", "generic"])

    assert exit_code == 0
    assert 0.0 <= float(capsys.readouterr().out.strip()) <= 1.0


def test_file_compare_cli_benchmark_reports_selected_backend(tmp_path, capsys) -> None:
    from stride_align.file_compare import main

    query = tmp_path / "query.txt"
    target = tmp_path / "target.txt"
    query.write_text("ACGT", encoding="utf-8")
    target.write_text("AGT", encoding="utf-8")

    exit_code = main(
        [
            str(query),
            str(target),
            "--simd",
            "generic",
            "--benchmark",
            "--benchmark-iterations",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "backend=generic" in captured.out
    assert "kernel_backend=generic" in captured.out
    assert "effective_width=8" in captured.out
    assert "iterations=2" in captured.out


def test_file_compare_cli_binary_encoding_accepts_raw_bytes(tmp_path, capsys) -> None:
    from stride_align.file_compare import main

    query = tmp_path / "query.bin"
    target = tmp_path / "target.bin"
    payload = b"\x7fELF\x00\xff\x80A"
    query.write_bytes(payload)
    target.write_bytes(payload)

    exit_code = main(
        [
            str(query),
            str(target),
            "--encoding",
            "binary",
            "--token-width",
            "8",
            "--simd",
            "generic",
        ]
    )

    assert exit_code == 0
    assert float(capsys.readouterr().out.strip()) == 1.0


def test_file_compare_cli_defaults_to_auto_backend(tmp_path, capsys) -> None:
    from stride_align.file_compare import main

    query = tmp_path / "query.txt"
    target = tmp_path / "target.txt"
    query.write_text("ACGT", encoding="utf-8")
    target.write_text("AGT", encoding="utf-8")

    exit_code = main([str(query), str(target), "--benchmark"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "backend=auto" in captured.out
    assert "kernel_backend=auto" in captured.out


def test_benchmark_cli_defaults_include_short_8_bit_english(capsys) -> None:
    from stride_align.benchmark import main

    exit_code = main(
        [
            "--backends",
            "generic",
            "--length",
            "16",
            "--iterations",
            "1",
            "--warmups",
            "0",
            "--format",
            "csv",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.splitlines()[0].startswith(
        "pass,case,shape,backend,variant,generator,output,score_width"
    )
    assert "\nenglish-short,linear,1:1,generic,sw-farrar-score,score-only,score,8," in captured.out
    assert "\nenglish-short,linear,1:1,generic,sw-score,score-only,score,8," in captured.out
    assert "\nenglish-short,linear,1:1,generic,nw-score,score-only,score,8," in captured.out
    assert "\nenglish-short,linear,1:1,generic,sw-path-info,path,path-info,8," in captured.out
    assert "\nenglish-short,linear,1:1,generic,nw-path-info,path,path-info,8," in captured.out
    assert "\nenglish-short,linear,1:many,generic,sw-score,score-only,score,8," in captured.out
    assert "\nenglish,linear,1:1,generic,sw-farrar-score,score-only,score,16," in captured.out
    assert "\nchinese,linear,1:1,generic,sw-farrar-score,score-only,score,32," in captured.out


def test_benchmark_cli_supports_path_info_variants(capsys) -> None:
    from stride_align.benchmark import main

    exit_code = main(
        [
            "--backends",
            "generic",
            "--variants",
            "sw-path-info",
            "nw-path-info",
            "--passes",
            "english",
            "--widths",
            "16",
            "--length",
            "16",
            "--iterations",
            "1",
            "--warmups",
            "0",
            "--format",
            "csv",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "\nenglish,linear,1:1,generic,sw-path-info,path,path-info,16," in captured.out
    assert "\nenglish,linear,1:1,generic,nw-path-info,path,path-info,16," in captured.out


def test_benchmark_cli_supports_path_trace_timing_split(capsys) -> None:
    from stride_align.benchmark import main

    exit_code = main(
        [
            "--backends",
            "generic",
            "--variants",
            "sw-cigar",
            "--passes",
            "english-short",
            "--shapes",
            "1:1",
            "--scoring-cases",
            "linear",
            "--widths",
            "8",
            "--short-length",
            "8",
            "--iterations",
            "1",
            "--warmups",
            "0",
            "--timing-split",
            "--format",
            "csv",
        ]
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    header = lines[0].split(",")
    row = lines[1].split(",")
    score_baseline_index = header.index("score_baseline_s")
    trace_over_index = header.index("path_trace_over_score_s")
    path_info_index = header.index("path_info_baseline_s")
    materialize_index = header.index("materialize_over_path_info_s")

    assert exit_code == 0
    assert row[4] == "sw-cigar"
    assert float(row[score_baseline_index]) > 0.0
    assert row[trace_over_index] != ""
    assert row[path_info_index] == ""
    assert row[materialize_index] == ""


def test_benchmark_cli_reports_affine_cigar_preprocess_trace_split(capsys) -> None:
    from stride_align.benchmark import main

    exit_code = main(
        [
            "--backends",
            "generic",
            "--variants",
            "sw-cigar",
            "--passes",
            "english-short",
            "--shapes",
            "1:1",
            "--scoring-cases",
            "affine",
            "--widths",
            "8",
            "--short-length",
            "8",
            "--iterations",
            "1",
            "--warmups",
            "0",
            "--timing-split",
            "--format",
            "csv",
        ]
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    header = lines[0].split(",")
    row = lines[1].split(",")
    preprocess_index = header.index("preprocess_s")
    dp_trace_index = header.index("dp_trace_s")

    assert exit_code == 0
    assert row[4] == "sw-cigar"
    assert row[preprocess_index] != ""
    assert float(row[dp_trace_index]) > 0.0


def test_benchmark_cli_supports_one_to_many_shape(capsys) -> None:
    from stride_align.benchmark import main

    exit_code = main(
        [
            "--backends",
            "generic",
            "--variants",
            "score",
            "--passes",
            "english",
            "--shapes",
            "1:many",
            "--many-count",
            "3",
            "--widths",
            "16",
            "--length",
            "16",
            "--iterations",
            "1",
            "--warmups",
            "0",
            "--format",
            "csv",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "\nenglish,linear,1:many,generic,sw-score,score-only,score,16," in captured.out
    assert "\nenglish,linear,1:many,generic,nw-score,score-only,score,16," in captured.out
    assert ",sw-path-info," not in captured.out


def test_benchmark_labels_score_and_path_outputs() -> None:
    from stride_align import benchmark

    assert benchmark._generator_for_variant("sw-score") == "score-only"
    assert benchmark._generator_for_variant("sw-path-info") == "path"
    assert benchmark._output_for_variant("generic", "sw-path-info") == "path-info"
    assert benchmark._output_for_variant("generic", "sw-path") == "full-path"
    assert benchmark._output_for_variant("parasail", "sw-path-info") == "trace-cigar"


def test_benchmark_variant_groups_expand_generators() -> None:
    from stride_align import benchmark

    assert benchmark._selected_variants(["score-only"]) == [
        "sw-farrar-score",
        "sw-score",
        "nw-score",
    ]
    assert benchmark._selected_variants(["path"]) == ["sw-path-info", "nw-path-info"]


def test_benchmark_cli_defaults_to_current_machine_backends(capsys) -> None:
    from stride_align import benchmark

    expected_backends = set(benchmark._available_backend_names())
    exit_code = benchmark.main(
        [
            "--widths",
            "16",
            "--length",
            "16",
            "--iterations",
            "1",
            "--warmups",
            "0",
            "--format",
            "csv",
        ]
    )

    captured = capsys.readouterr()
    observed_backends = {
        line.split(",")[3] for line in captured.out.splitlines()[1:] if line
    }
    assert exit_code == 0
    assert observed_backends == expected_backends


def test_benchmark_available_backend_names_are_cpu_discovered(monkeypatch) -> None:
    from stride_align import benchmark

    monkeypatch.setattr(
        benchmark,
        "available_backends",
        lambda: [
            SimpleNamespace(name="generic", available=True),
            SimpleNamespace(name="swar", available=True),
            SimpleNamespace(name="x86_sse41", available=False),
            SimpleNamespace(name="x86_avx2", available=True),
        ],
    )
    monkeypatch.setattr(benchmark.importlib.util, "find_spec", lambda name: None)

    assert benchmark._available_backend_names() == ["generic", "swar", "x86_avx2"]
    assert benchmark._selected_backend_names(["available"]) == ["generic", "swar", "x86_avx2"]


def test_benchmark_adds_parasail_when_importable(monkeypatch) -> None:
    from stride_align import benchmark

    monkeypatch.setattr(
        benchmark,
        "available_backends",
        lambda: [SimpleNamespace(name="generic", available=True)],
    )
    monkeypatch.setattr(
        benchmark.importlib.util,
        "find_spec",
        lambda name: object() if name == "parasail" else None,
    )

    assert benchmark._available_backend_names() == ["generic", "parasail"]


@pytest.mark.parametrize(
    ("variant", "prepare_name", "prepared_name", "direct_name"),
    [
        (
            "sw-farrar-score",
            "_prepare_smith_waterman_affine_farrar_score",
            "_smith_waterman_affine_farrar_score_prepared",
            "smith_waterman_farrar_score",
        ),
        (
            "sw-score",
            "_prepare_smith_waterman_affine_score",
            "_smith_waterman_affine_score_prepared",
            "smith_waterman_score",
        ),
        (
            "nw-score",
            "_prepare_needleman_wunsch_affine_score",
            "_needleman_wunsch_affine_score_prepared",
            "needleman_wunsch_score",
        ),
    ],
)
def test_benchmark_uses_prepared_affine_profiles_when_available(
    variant: str,
    prepare_name: str,
    prepared_name: str,
    direct_name: str,
) -> None:
    from stride_align import benchmark

    class FakeModule:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def __getattr__(self, name: str):
            if name == prepare_name:

                def prepare(*args, **kwargs):
                    assert kwargs["gap_open_score"] == -2
                    assert kwargs["gap_extend_score"] == -1
                    self.calls.append("prepare")
                    return object()

                return prepare
            if name == prepared_name:

                def run_prepared(prepared):
                    self.calls.append("prepared")
                    return 37

                return run_prepared
            if name == direct_name:

                def run_direct(*args, **kwargs):
                    raise AssertionError("direct affine benchmark path should not be used")

                return run_direct
            raise AttributeError(name)

    fake = FakeModule()
    result = benchmark._time_backend(
        benchmark.ResolvedBackend("x86_avx2", "fake", fake),
        "english",
        "affine",
        variant,
        "AAAA",
        "AAAT",
        16,
        2,
        1,
        2,
        -1,
        -2,
        -1,
    )

    assert result.score == 37
    assert fake.calls == ["prepare", "prepared", "prepared", "prepared", "prepared"]


def test_benchmark_uses_prepared_one_to_many_profiles_when_available() -> None:
    from stride_align import benchmark

    class FakeModule:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def _prepare_smith_waterman_farrar_scores(self, *args, **kwargs):
            assert kwargs["gap_score"] == -1
            self.calls.append("prepare")
            return object()

        def _smith_waterman_farrar_scores_prepared(self, prepared):
            self.calls.append("prepared")
            return [10, 11, 12]

        def smith_waterman_farrar_scores(self, *args, **kwargs):
            self.calls.append("direct")
            return [10, 11, 12]

    fake = FakeModule()
    result = benchmark._time_backend(
        benchmark.ResolvedBackend("x86_avx2", "fake", fake),
        "english",
        "linear",
        "sw-farrar-score",
        "AAAA",
        ("AAAT", "AATA", "ATAA"),
        16,
        2,
        0,
        2,
        -1,
        -1,
        -1,
        shape="1:many",
        timing_split=True,
    )

    assert result.score == 33
    assert result.preprocess_seconds is not None
    assert result.dp_trace_seconds is not None
    assert fake.calls.count("prepare") == 1
    assert fake.calls.count("prepared") == 3
    assert fake.calls.count("direct") == 2


def test_benchmark_parasail_adapter_uses_safe_translated_inputs() -> None:
    from stride_align import benchmark

    class FakeParasailResult:
        score = 7

        def __init__(self) -> None:
            self.cigar_accessed = False

        @property
        def cigar(self) -> str:
            self.cigar_accessed = True
            return "1M"

    class FakeParasail:
        def __init__(self) -> None:
            self.calls = []
            self.last_result: FakeParasailResult | None = None

        def matrix_create(self, alphabet: bytes, match_score: int, mismatch_score: int):
            return (alphabet, match_score, mismatch_score)

        def sw_trace_striped_16(
            self,
            query: bytes,
            target: bytes,
            gap_open: int,
            gap_extend: int,
            matrix,
        ) -> FakeParasailResult:
            assert b"\x00" not in query
            assert b"\x00" not in target
            assert b"\\" not in query
            assert b"\\" not in target
            assert gap_open == 1
            assert gap_extend == 1
            self.calls.append((query, target, matrix))
            self.last_result = FakeParasailResult()
            return self.last_result

    fake = FakeParasail()
    adapter = benchmark._ParasailBenchmarkBackend(fake)

    result = adapter.smith_waterman_path_info(
        b"\x00\x01",
        b"\x01",
        match_score=2,
        mismatch_score=-1,
        gap_score=-1,
        width=16,
    )

    assert result.score == 7
    assert fake.calls
    assert fake.last_result is not None
    assert fake.last_result.cigar_accessed


def test_pyproject_does_not_depend_on_parasail() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "parasail" not in pyproject


def test_file_compare_cli_validates_score_width_after_decoding(tmp_path, capsys) -> None:
    from stride_align.file_compare import main

    query = tmp_path / "query.txt"
    target = tmp_path / "target.txt"
    query.write_text("A" * 100, encoding="utf-8")
    target.write_text("A" * 100, encoding="utf-8")

    exit_code = main(
        [
            str(query),
            str(target),
            "--score-width",
            "8",
            "--match-score",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "exceeds signed 8-bit capacity" in captured.err


def test_file_compare_cli_rejects_farrar_with_wide_token_channel(tmp_path, capsys) -> None:
    from stride_align.file_compare import main

    query = tmp_path / "query.txt"
    target = tmp_path / "target.txt"
    query.write_text("ACGT", encoding="utf-8")
    target.write_text("AGT", encoding="utf-8")

    exit_code = main([str(query), str(target), "--farrar", "--token-width", "16"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Farrar uses an 8-bit token channel" in captured.err


def test_file_compare_cli_accepts_affine_gap_scores(tmp_path, capsys) -> None:
    from stride_align.file_compare import main

    query = tmp_path / "query.txt"
    target = tmp_path / "target.txt"
    query.write_text("AAABBB", encoding="utf-8")
    target.write_text("AAACCCBBB", encoding="utf-8")

    exit_code = main(
        [
            str(query),
            str(target),
            "--gap-open-score",
            "-3",
            "--gap-extend-score",
            "-1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert float(captured.out.strip()) == pytest.approx(7 / 12)


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_generic_backend_supports_all_kernel_widths(width: int) -> None:
    generic = pytest.importorskip("stride_align._generic")

    assert generic.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert generic.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = generic.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = generic.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "==="
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "==X="


@pytest.mark.skipif(
    not backend_is_available(BackendKind.X86_AVX2),
    reason="AVX2 backend not available on this host",
)
def test_direct_avx2_prepared_batch_scores_match_direct_scores() -> None:
    avx2 = pytest.importorskip("stride_align._avx2")
    query = "The quick brown fox watches the city wake." * 4
    targets = [
        "The quick brown fox watches the city wake." * 4,
        "The quick brown fox watches the city work." * 4,
        "People cross the station concourse." * 5,
    ]
    affine_kwargs = {
        "match_score": 2,
        "mismatch_score": -1,
        "gap_score": -1,
        "gap_open_score": -3,
        "gap_extend_score": -1,
        "width": 16,
    }

    prepared = avx2._prepare_smith_waterman_farrar_scores(
        query,
        targets,
        match_score=2,
        mismatch_score=-1,
        gap_score=-1,
        width=16,
    )
    np.testing.assert_array_equal(
        avx2._smith_waterman_farrar_scores_prepared(prepared),
        avx2.smith_waterman_farrar_scores(
            query,
            targets,
            match_score=2,
            mismatch_score=-1,
            gap_score=-1,
            width=16,
        ),
    )

    prepared_affine = avx2._prepare_needleman_wunsch_affine_scores(
        query,
        targets,
        **affine_kwargs,
    )
    np.testing.assert_array_equal(
        avx2._needleman_wunsch_affine_scores_prepared(prepared_affine),
        avx2.needleman_wunsch_scores(query, targets, **affine_kwargs),
    )


@pytest.mark.skipif(
    not backend_is_available(BackendKind.SWAR),
    reason="SWAR backend not available on this host",
)
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_swar_backend_supports_all_kernel_widths(width: int) -> None:
    swar = pytest.importorskip("stride_align._swar")

    assert swar.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert swar.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = swar.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = swar.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "==="
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "==X="


FARRAR_BACKENDS = [
    ("stride_align._generic", None),
    ("stride_align._swar", BackendKind.SWAR),
    ("stride_align._sse41", BackendKind.X86_SSE41),
    ("stride_align._avx2", BackendKind.X86_AVX2),
    ("stride_align._avx512bwvl", BackendKind.X86_AVX512BWVL),
    ("stride_align._avx10_256", BackendKind.X86_AVX10_256),
    ("stride_align._avx10_512", BackendKind.X86_AVX10_512),
    ("stride_align._neon", BackendKind.LINUX_AARCH64_NEON),
    ("stride_align._sve", BackendKind.LINUX_AARCH64_SVE),
    ("stride_align._sve2", BackendKind.LINUX_AARCH64_SVE2),
    ("stride_align._macos_arm64_neon", BackendKind.MACOS_ARM64_NEON),
    ("stride_align._lsx", BackendKind.LINUX_LOONGARCH64_LSX),
    ("stride_align._lasx", BackendKind.LINUX_LOONGARCH64_LASX),
    ("stride_align._vsx", BackendKind.LINUX_POWERPC64_VSX),
    ("stride_align._rvv", BackendKind.LINUX_RISCV64_RVV),
]


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("module_name, backend_kind", FARRAR_BACKENDS)
def test_direct_backends_linear_scores_match_generic_across_stripes(
    module_name: str,
    backend_kind: BackendKind | None,
    width: int,
) -> None:
    if backend_kind is not None and not backend_is_available(backend_kind):
        pytest.skip(f"{backend_kind.name} not available on this host")

    generic = pytest.importorskip("stride_align._generic")
    module = pytest.importorskip(module_name)
    query = "ABCD" * 9 + "A"
    target = "ACBD" * 7 + "D"
    kwargs = {
        "match_score": 1,
        "mismatch_score": -1,
        "gap_score": -1,
        "width": width,
    }

    assert module.smith_waterman_score(query, target, **kwargs) == generic.smith_waterman_score(
        query,
        target,
        **kwargs,
    )
    assert module.needleman_wunsch_score(query, target, **kwargs) == generic.needleman_wunsch_score(
        query,
        target,
        **kwargs,
    )


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("module_name, backend_kind", FARRAR_BACKENDS)
def test_direct_backends_support_farrar_score_widths(
    module_name: str,
    backend_kind: BackendKind | None,
    width: int,
) -> None:
    if backend_kind is not None and not backend_is_available(backend_kind):
        pytest.skip(f"{backend_kind.name} not available on this host")

    module = pytest.importorskip(module_name)
    query = b"GGCCTT"
    target = b"CGGTTAT"

    expected = module.smith_waterman_score(query, target)

    assert module.smith_waterman_farrar_score(query, target, width=width) == expected


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("module_name, backend_kind", FARRAR_BACKENDS)
def test_direct_backends_accept_affine_gap_scores(
    module_name: str,
    backend_kind: BackendKind | None,
    width: int,
) -> None:
    if backend_kind is not None and not backend_is_available(backend_kind):
        pytest.skip(f"{backend_kind.name} not available on this host")

    module = pytest.importorskip(module_name)
    kwargs = {
        "match_score": 2,
        "mismatch_score": -1,
        "gap_score": -1,
        "gap_open_score": -3,
        "gap_extend_score": -1,
    }

    assert module.smith_waterman_score("AAABBB", "AAACCCBBB", width=width, **kwargs) == 7
    assert module.smith_waterman_farrar_score("AAABBB", "AAACCCBBB", width=width, **kwargs) == 7
    assert module.needleman_wunsch_score("AAABBB", "AAACCCBBB", width=width, **kwargs) == 7
    assert (
        module.smith_waterman_path("AAABBB", "AAACCCBBB", width=width, **kwargs).operations
        == "===III==="
    )
    assert module.smith_waterman_cigar("AAABBB", "AAACCCBBB", width=width, **kwargs) == "3=3I3="
    assert module.smith_waterman_trace_cigar("AAABBB", "AAACCCBBB", width=width, **kwargs) == "3=3I3="
    assert module.smith_waterman_trade_cigar("AAABBB", "AAACCCBBB", width=width, **kwargs) == "3=3I3="


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("module_name, backend_kind", FARRAR_BACKENDS)
def test_direct_backends_prepared_affine_profiles_match_direct_scores(
    module_name: str,
    backend_kind: BackendKind | None,
    width: int,
) -> None:
    if backend_kind is not None and not backend_is_available(backend_kind):
        pytest.skip(f"{backend_kind.name} not available on this host")

    module = pytest.importorskip(module_name)
    if not hasattr(module, "_prepare_smith_waterman_affine_score"):
        pytest.skip(f"{module_name} does not expose prepared affine profiles")

    query = "ABCAABBC"
    target = "AACBBAC"
    kwargs = {
        "match_score": 2,
        "mismatch_score": -1,
        "gap_score": -1,
        "gap_open_score": -3,
        "gap_extend_score": -1,
        "width": width,
    }

    sw_prepared = module._prepare_smith_waterman_affine_score(query, target, **kwargs)
    sw_farrar_prepared = module._prepare_smith_waterman_affine_farrar_score(query, target, **kwargs)
    nw_prepared = module._prepare_needleman_wunsch_affine_score(query, target, **kwargs)

    assert module._smith_waterman_affine_score_prepared(sw_prepared) == module.smith_waterman_score(
        query,
        target,
        **kwargs,
    )
    assert module._smith_waterman_affine_farrar_score_prepared(
        sw_farrar_prepared
    ) == module.smith_waterman_farrar_score(query, target, **kwargs)
    assert module._needleman_wunsch_affine_score_prepared(
        nw_prepared
    ) == module.needleman_wunsch_score(query, target, **kwargs)


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("module_name, backend_kind", FARRAR_BACKENDS)
def test_direct_backends_prepared_affine_cigars_match_direct_cigars(
    module_name: str,
    backend_kind: BackendKind | None,
    width: int,
) -> None:
    if backend_kind is not None and not backend_is_available(backend_kind):
        pytest.skip(f"{backend_kind.name} not available on this host")

    module = pytest.importorskip(module_name)
    if not hasattr(module, "_prepare_smith_waterman_affine_cigar"):
        pytest.skip(f"{module_name} does not expose prepared affine CIGAR profiles")

    query = "ABCAABBC"
    target = "AACBBAC"
    kwargs = {
        "match_score": 2,
        "mismatch_score": -1,
        "gap_score": -1,
        "gap_open_score": -3,
        "gap_extend_score": -1,
        "width": width,
    }

    sw_prepared = module._prepare_smith_waterman_affine_cigar(query, target, **kwargs)
    nw_prepared = module._prepare_needleman_wunsch_affine_cigar(query, target, **kwargs)

    assert module._smith_waterman_affine_cigar_prepared(
        sw_prepared
    ) == module.smith_waterman_cigar(query, target, **kwargs)
    assert module._needleman_wunsch_affine_cigar_prepared(
        nw_prepared
    ) == module.needleman_wunsch_cigar(query, target, **kwargs)


def test_generic_affine_cigar_banded_path_matches_python_backend() -> None:
    generic = pytest.importorskip("stride_align._generic")
    from stride_align import _pybackend

    query = ("thequickbrownfoxjumpsoverthelazydog" * 5)[:160]
    target = query[:47] + "zz" + query[47:91] + "q" + query[92:128] + query[130:]
    kwargs = {
        "match_score": 2,
        "mismatch_score": -1,
        "gap_score": -1,
        "gap_open_score": -3,
        "gap_extend_score": -1,
        "width": 16,
    }

    assert generic.smith_waterman_cigar(query, target, **kwargs) == _pybackend.smith_waterman_cigar(
        query,
        target,
        **kwargs,
    )
    assert generic.needleman_wunsch_cigar(
        query,
        target,
        **kwargs,
    ) == _pybackend.needleman_wunsch_cigar(query, target, **kwargs)


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("module_name, backend_kind", FARRAR_BACKENDS)
def test_direct_backends_profile_traceback_matches_generic(
    module_name: str,
    backend_kind: BackendKind | None,
    width: int,
) -> None:
    if backend_kind is not None and not backend_is_available(backend_kind):
        pytest.skip(f"{backend_kind.name} not available on this host")

    generic = pytest.importorskip("stride_align._generic")
    module = pytest.importorskip(module_name)
    query = "ABCAABBC"
    target = "AACBBAC"

    linear_kwargs = {
        "match_score": 2,
        "mismatch_score": -1,
        "gap_score": -1,
        "width": width,
    }
    affine_kwargs = {
        "match_score": 2,
        "mismatch_score": -1,
        "gap_score": -1,
        "gap_open_score": -3,
        "gap_extend_score": -1,
        "width": width,
    }

    for method_name, kwargs in (
        ("smith_waterman_path", linear_kwargs),
        ("needleman_wunsch_path", linear_kwargs),
        ("smith_waterman_path", affine_kwargs),
        ("needleman_wunsch_path", affine_kwargs),
    ):
        expected = getattr(generic, method_name)(query, target, **kwargs)
        observed = getattr(module, method_name)(query, target, **kwargs)
        observed_info = getattr(module, f"{method_name}_info")(query, target, **kwargs)

        assert observed.score == expected.score
        assert observed.operations == expected.operations
        assert observed.query_start == expected.query_start
        assert observed.target_start == expected.target_start
        assert observed_info.score == expected.score
        assert observed_info.operations == expected.operations
        assert getattr(module, method_name.replace("_path", "_cigar"))(
            query,
            target,
            **kwargs,
        ) == observed_info.cigar


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("module_name, backend_kind", FARRAR_BACKENDS)
def test_direct_backends_affine_farrar_matches_score_across_stripes(
    module_name: str,
    backend_kind: BackendKind | None,
    width: int,
) -> None:
    if backend_kind is not None and not backend_is_available(backend_kind):
        pytest.skip(f"{backend_kind.name} not available on this host")

    module = pytest.importorskip(module_name)
    kwargs = {
        "match_score": 1,
        "mismatch_score": -1,
        "gap_score": -2,
        "gap_open_score": -2,
        "gap_extend_score": -1,
    }
    query = "A" * 8 + "CCC" + "B" * 8
    target = "A" * 8 + "B" * 8

    expected = module.smith_waterman_score(query, target, width=width, **kwargs)

    assert module.smith_waterman_farrar_score(query, target, width=width, **kwargs) == expected


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("module_name, backend_kind", FARRAR_BACKENDS)
def test_direct_backends_affine_traceback_lazy_f_cross_lane_regression(
    module_name: str,
    backend_kind: BackendKind | None,
    width: int,
) -> None:
    if backend_kind is not None and not backend_is_available(backend_kind):
        pytest.skip(f"{backend_kind.name} not available on this host")

    generic = pytest.importorskip("stride_align._generic")
    module = pytest.importorskip(module_name)
    query = "TATCAGTGCGTACTT"
    target = "CCCAATCTTTATGATGACTTTCCTCCCGG"
    kwargs = {
        "match_score": 2,
        "mismatch_score": -1,
        "gap_score": -1,
        "gap_open_score": -3,
        "gap_extend_score": -1,
        "width": width,
    }

    expected = generic.smith_waterman_path_info(query, target, **kwargs)
    observed = module.smith_waterman_path_info(query, target, **kwargs)

    assert observed.score == expected.score
    assert observed.operations == expected.operations
    assert module.smith_waterman_score(query, target, **kwargs) == expected.score


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_sse41_backend_supports_all_kernel_widths(width: int) -> None:
    sse41 = pytest.importorskip("stride_align._sse41")

    assert sse41.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert sse41.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = sse41.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = sse41.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "==="
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "==X="


@pytest.mark.skipif(
    not backend_is_available(BackendKind.X86_AVX2),
    reason="AVX2 not available on this host",
)
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_avx2_backend_supports_all_kernel_widths(width: int) -> None:
    avx2 = pytest.importorskip("stride_align._avx2")

    assert avx2.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert avx2.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = avx2.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = avx2.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "==="
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "==X="


@pytest.mark.skipif(
    not backend_is_available(BackendKind.X86_AVX512BWVL),
    reason="AVX-512BWVL not available on this host",
)
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_avx512bwvl_backend_supports_all_kernel_widths(width: int) -> None:
    avx512 = pytest.importorskip("stride_align._avx512bwvl")

    assert avx512.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert avx512.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = avx512.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = avx512.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "==="
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "==X="


@pytest.mark.skipif(
    backend_is_available(BackendKind.X86_AVX512BWVL),
    reason="AVX-512BWVL available on this host",
)
def test_direct_avx512bwvl_backend_raises_runtime_error_when_unavailable() -> None:
    avx512 = pytest.importorskip("stride_align._avx512bwvl")

    with pytest.raises(RuntimeError, match="not available on this machine"):
        avx512.smith_waterman_score("ACCGT", "CCG")


@pytest.mark.skipif(
    not backend_is_available(BackendKind.X86_AVX10_256),
    reason="AVX10.1-256 not available on this host",
)
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_avx10_256_backend_supports_all_kernel_widths(width: int) -> None:
    avx10_256 = pytest.importorskip("stride_align._avx10_256")

    assert avx10_256.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert avx10_256.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = avx10_256.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = avx10_256.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "==="
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "==X="


@pytest.mark.skipif(
    backend_is_available(BackendKind.X86_AVX10_256),
    reason="AVX10.1-256 available on this host",
)
def test_direct_avx10_256_backend_raises_runtime_error_when_unavailable() -> None:
    avx10_256 = pytest.importorskip("stride_align._avx10_256")

    with pytest.raises(RuntimeError, match="not available on this machine"):
        avx10_256.smith_waterman_score("ACCGT", "CCG")


@pytest.mark.skipif(
    not backend_is_available(BackendKind.X86_AVX10_512),
    reason="AVX10.1-512 not available on this host",
)
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_avx10_512_backend_supports_all_kernel_widths(width: int) -> None:
    avx10_512 = pytest.importorskip("stride_align._avx10_512")

    assert avx10_512.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert avx10_512.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = avx10_512.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = avx10_512.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "==="
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "==X="


@pytest.mark.skipif(
    backend_is_available(BackendKind.X86_AVX10_512),
    reason="AVX10.1-512 available on this host",
)
def test_direct_avx10_512_backend_raises_runtime_error_when_unavailable() -> None:
    avx10_512 = pytest.importorskip("stride_align._avx10_512")

    with pytest.raises(RuntimeError, match="not available on this machine"):
        avx10_512.smith_waterman_score("ACCGT", "CCG")


@pytest.mark.skipif(
    not backend_is_available(BackendKind.MACOS_ARM64_NEON),
    reason="macOS arm64 NEON not available on this host",
)
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_macos_arm64_neon_backend_supports_all_kernel_widths(width: int) -> None:
    macos_neon = pytest.importorskip("stride_align._macos_arm64_neon")

    assert macos_neon.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert macos_neon.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = macos_neon.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = macos_neon.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "==="
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "==X="


@pytest.mark.skipif(
    not backend_is_available(BackendKind.LINUX_AARCH64_NEON),
    reason="Linux AArch64 NEON not available on this host",
)
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_neon_backend_supports_all_kernel_widths(width: int) -> None:
    neon = pytest.importorskip("stride_align._neon")

    assert neon.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert neon.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = neon.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = neon.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "==="
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "==X="


@pytest.mark.skipif(
    not backend_is_available(BackendKind.LINUX_AARCH64_SVE),
    reason="Linux AArch64 SVE not available on this host",
)
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_sve_backend_supports_all_kernel_widths(width: int) -> None:
    sve = pytest.importorskip("stride_align._sve")

    assert sve.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert sve.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = sve.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = sve.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "==="
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "==X="


@pytest.mark.skipif(
    not backend_is_available(BackendKind.LINUX_AARCH64_SVE2),
    reason="Linux AArch64 SVE2 not available on this host",
)
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_sve2_backend_supports_all_kernel_widths(width: int) -> None:
    sve2 = pytest.importorskip("stride_align._sve2")

    assert sve2.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert sve2.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = sve2.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = sve2.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "==="
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "==X="


@pytest.mark.skipif(
    not backend_is_available(BackendKind.LINUX_LOONGARCH64_LSX),
    reason="Linux LoongArch64 LSX not available on this host",
)
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_lsx_backend_supports_all_kernel_widths(width: int) -> None:
    lsx = pytest.importorskip("stride_align._lsx")

    assert lsx.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert lsx.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = lsx.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = lsx.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "==="
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "==X="


@pytest.mark.skipif(
    not backend_is_available(BackendKind.LINUX_LOONGARCH64_LASX),
    reason="Linux LoongArch64 LASX not available on this host",
)
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_lasx_backend_supports_all_kernel_widths(width: int) -> None:
    lasx = pytest.importorskip("stride_align._lasx")

    assert lasx.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert lasx.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = lasx.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = lasx.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "==="
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "==X="


@pytest.mark.skipif(
    not backend_is_available(BackendKind.LINUX_POWERPC64_VSX),
    reason="Linux PowerPC64 VSX not available on this host",
)
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_vsx_backend_supports_all_kernel_widths(width: int) -> None:
    vsx = pytest.importorskip("stride_align._vsx")

    assert vsx.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert vsx.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = vsx.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = vsx.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "==="
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "==X="


@pytest.mark.skipif(
    not backend_is_available(BackendKind.LINUX_RISCV64_RVV),
    reason="Linux RISC-V RVV not available on this host",
)
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_rvv_backend_supports_all_kernel_widths(width: int) -> None:
    rvv = pytest.importorskip("stride_align._rvv")

    assert rvv.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert rvv.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = rvv.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = rvv.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "==="
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "==X="


def test_zero_score_local_alignment_returns_empty_path() -> None:
    result = smith_waterman_path("AAAA", "TTTT", mismatch_score=-3, gap_score=-2)

    assert result.score == 0
    assert result.aligned_query == ""
    assert result.aligned_target == ""
    assert result.operations == ""
