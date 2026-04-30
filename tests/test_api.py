import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from stride_align import (
    AlignmentPath,
    AlignmentResult,
    BackendKind,
    available_backends,
    backend_is_available,
    detect_best_backend,
    needleman_wunsch_path,
    needleman_wunsch_path_info,
    needleman_wunsch_score,
    smith_waterman_farrar_score,
    smith_waterman_path,
    smith_waterman_path_info,
    smith_waterman_score,
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
    assert result.operations == "MMXM"


def test_needleman_wunsch_path_info_on_strings() -> None:
    result = needleman_wunsch_path_info("ACGT", "ACCT")

    assert isinstance(result, AlignmentPath)
    assert result.score == 5
    assert result.query_start == 0
    assert result.query_end == 4
    assert result.target_start == 0
    assert result.target_end == 4
    assert result.operations == "MMXM"
    assert result.cigar == "2M1X1M"
    assert result.matches == 3
    assert result.mismatches == 1
    assert result.insertions == 0
    assert result.deletions == 0
    assert result.aligned_length == 4
    assert not hasattr(result, "aligned_query")


def test_string_fast_path_handles_wide_unicode() -> None:
    result = needleman_wunsch_path("A🙂", "A🙂")

    assert result.score == 4
    assert result.aligned_query == "A🙂"
    assert result.aligned_target == "A🙂"
    assert result.operations == "MM"


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
    assert result.operations == "MMMIIIMMM"


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


def test_smith_waterman_path_info_on_bytes() -> None:
    result = smith_waterman_path_info(b"ACCGT", b"CCG")

    assert isinstance(result, AlignmentPath)
    assert result.score == 6
    assert result.query_start == 1
    assert result.query_end == 4
    assert result.target_start == 0
    assert result.target_end == 3
    assert result.operations == "MMM"
    assert result.cigar == "3M"
    assert result.matches == 3
    assert result.mismatches == 0
    assert result.insertions == 0
    assert result.deletions == 0
    assert result.aligned_length == 3


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
        "pass,case,backend,variant,generator,output,score_width"
    )
    assert "\nenglish-short,linear,generic,sw-farrar-score,score-only,score,8," in captured.out
    assert "\nenglish-short,linear,generic,sw-score,score-only,score,8," in captured.out
    assert "\nenglish-short,linear,generic,nw-score,score-only,score,8," in captured.out
    assert "\nenglish-short,linear,generic,sw-path-info,path,path-info,8," in captured.out
    assert "\nenglish-short,linear,generic,nw-path-info,path,path-info,8," in captured.out
    assert "\nenglish,linear,generic,sw-farrar-score,score-only,score,16," in captured.out
    assert "\nchinese,linear,generic,sw-farrar-score,score-only,score,32," in captured.out


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
    assert "\nenglish,linear,generic,sw-path-info,path,path-info,16," in captured.out
    assert "\nenglish,linear,generic,nw-path-info,path,path-info,16," in captured.out


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
        line.split(",")[2] for line in captured.out.splitlines()[1:] if line
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

        def matrix_create(self, alphabet: str, match_score: int, mismatch_score: int):
            return (alphabet, match_score, mismatch_score)

        def sw_trace_striped_16(
            self,
            query: str,
            target: str,
            gap_open: int,
            gap_extend: int,
            matrix,
        ) -> FakeParasailResult:
            assert "\x00" not in query
            assert "\x00" not in target
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
    assert sw_result.operations == "MMM"
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "MMXM"


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
    assert sw_result.operations == "MMM"
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "MMXM"


FARRAR_BACKENDS = [
    ("stride_align._generic", None),
    ("stride_align._swar", BackendKind.SWAR),
    ("stride_align._sse41", BackendKind.X86_SSE41),
    ("stride_align._avx2", BackendKind.X86_AVX2),
    ("stride_align._avx512bwvl", BackendKind.X86_AVX512BWVL),
    ("stride_align._avx10_256", BackendKind.X86_AVX10_256),
    ("stride_align._avx10_512", BackendKind.X86_AVX10_512),
    ("stride_align._asimd", BackendKind.LINUX_AARCH64_ASIMD),
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
        == "MMMIIIMMM"
    )


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


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_sse41_backend_supports_all_kernel_widths(width: int) -> None:
    sse41 = pytest.importorskip("stride_align._sse41")

    assert sse41.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert sse41.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = sse41.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = sse41.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "MMM"
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "MMXM"


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
    assert sw_result.operations == "MMM"
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "MMXM"


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
    assert sw_result.operations == "MMM"
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "MMXM"


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
    assert sw_result.operations == "MMM"
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "MMXM"


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
    assert sw_result.operations == "MMM"
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "MMXM"


@pytest.mark.skipif(
    backend_is_available(BackendKind.X86_AVX10_512),
    reason="AVX10.1-512 available on this host",
)
def test_direct_avx10_512_backend_raises_runtime_error_when_unavailable() -> None:
    avx10_512 = pytest.importorskip("stride_align._avx10_512")

    with pytest.raises(RuntimeError, match="not available on this machine"):
        avx10_512.smith_waterman_score("ACCGT", "CCG")


@pytest.mark.skipif(
    not backend_is_available(BackendKind.LINUX_AARCH64_ASIMD),
    reason="Linux AArch64 ASIMD not available on this host",
)
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_asimd_backend_supports_all_kernel_widths(width: int) -> None:
    asimd = pytest.importorskip("stride_align._asimd")

    assert asimd.smith_waterman_score("ACCGT", "CCG", width=width) == 6
    assert asimd.needleman_wunsch_score("ACGT", "ACCT", width=width) == 5

    sw_result = asimd.smith_waterman_path("ACCGT", "CCG", width=width)
    nw_result = asimd.needleman_wunsch_path("ACGT", "ACCT", width=width)

    assert sw_result.aligned_query == "CCG"
    assert sw_result.aligned_target == "CCG"
    assert sw_result.operations == "MMM"
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "MMXM"


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
    assert sw_result.operations == "MMM"
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "MMXM"


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
    assert sw_result.operations == "MMM"
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "MMXM"


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
    assert sw_result.operations == "MMM"
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "MMXM"


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
    assert sw_result.operations == "MMM"
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "MMXM"


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
    assert sw_result.operations == "MMM"
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "MMXM"


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
    assert sw_result.operations == "MMM"
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "MMXM"


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
    assert sw_result.operations == "MMM"
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "MMXM"


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
    assert sw_result.operations == "MMM"
    assert nw_result.aligned_query == "ACGT"
    assert nw_result.aligned_target == "ACCT"
    assert nw_result.operations == "MMXM"


def test_zero_score_local_alignment_returns_empty_path() -> None:
    result = smith_waterman_path("AAAA", "TTTT", mismatch_score=-3, gap_score=-2)

    assert result.score == 0
    assert result.aligned_query == ""
    assert result.aligned_target == ""
    assert result.operations == ""
