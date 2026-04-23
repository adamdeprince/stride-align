import pytest

from stride_align import (
    AlignmentResult,
    BackendKind,
    available_backends,
    backend_is_available,
    detect_best_backend,
    needleman_wunsch_path,
    needleman_wunsch_score,
    smith_waterman_path,
    smith_waterman_score,
)


def test_detect_best_backend_returns_enum() -> None:
    backend = detect_best_backend()
    assert isinstance(backend, BackendKind)


def test_available_backends_includes_generic() -> None:
    backends = available_backends()
    assert any(record.name == "generic" and record.available for record in backends)


def test_backend_is_available_for_detected_backend() -> None:
    assert backend_is_available(detect_best_backend())


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
    assert result.operations == "MMXM"


def test_string_fast_path_handles_wide_unicode() -> None:
    result = needleman_wunsch_path("A🙂", "A🙂")

    assert result.score == 4
    assert result.aligned_query == "A🙂"
    assert result.aligned_target == "A🙂"
    assert result.operations == "MM"


def test_smith_waterman_score_on_strings() -> None:
    assert smith_waterman_score("ACCGT", "CCG") == 6


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
    assert result.operations == "MMM"


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
    assert result.operations == "MM"


def test_mixed_sequence_and_bytes_treats_bytes_as_sequence() -> None:
    result = needleman_wunsch_path([1, 2], b"\x01\x02")

    assert result.score == 4
    assert result.aligned_query == (1, 2)
    assert result.aligned_target == (1, 2)
    assert result.operations == "MM"


def test_width_parameter_can_force_a_wider_kernel() -> None:
    result = smith_waterman_path("ACCGT", "CCG", width=64)

    assert result.score == 6
    assert result.aligned_query == "CCG"
    assert result.aligned_target == "CCG"
    assert result.operations == "MMM"


def test_width_parameter_rejects_narrower_kernel() -> None:
    with pytest.raises(ValueError, match="narrower"):
        needleman_wunsch_score("🙂", "🙂", width=8)


def test_width_parameter_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="8, 16, 32, or 64"):
        smith_waterman_score("AC", "AC", width=24)


def test_zero_score_local_alignment_returns_empty_path() -> None:
    result = smith_waterman_path("AAAA", "TTTT", mismatch_score=-3, gap_score=-2)

    assert result.score == 0
    assert result.aligned_query == ""
    assert result.aligned_target == ""
    assert result.operations == ""
