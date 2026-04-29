import pytest

from stride_align import (
    AlignmentResult,
    BackendKind,
    available_backends,
    backend_is_available,
    detect_best_backend,
    needleman_wunsch_path,
    needleman_wunsch_score,
    smith_waterman_farrar_score,
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


def test_smith_waterman_farrar_score_matches_standard_score() -> None:
    assert smith_waterman_farrar_score(b"GGCCTT", b"CGGTTAT") == smith_waterman_score(
        b"GGCCTT",
        b"CGGTTAT",
    )


def test_smith_waterman_farrar_compacts_wide_unicode_to_byte_tokens() -> None:
    assert smith_waterman_farrar_score("🙂🙃🙂", "🙃🙂", width=8) == 4


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
    with pytest.raises(ValueError, match="None, 0, 8, 16, 32, or 64"):
        smith_waterman_score("AC", "AC", width=7)

    with pytest.raises(ValueError, match="None, 0, 8, 16, 32, or 64"):
        smith_waterman_farrar_score("AC", "AC", width=7)


def test_width_parameter_accepts_none_and_zero() -> None:
    assert smith_waterman_score("ACCGT", "CCG", width=None) == 6
    assert smith_waterman_score("ACCGT", "CCG", width=0) == 6


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


FARRAR_BACKENDS = [
    ("stride_align._generic", None),
    ("stride_align._swar", None),
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


@pytest.mark.parametrize("width", [8, 16, 32])
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


def test_direct_swar_backend_uses_generic_for_64_bit_width() -> None:
    swar = pytest.importorskip("stride_align._swar")

    assert swar.smith_waterman_score("ACCGT", "CCG", width=64) == 6

    result = swar.needleman_wunsch_path("ACGT", "ACCT", width=64)

    assert result.aligned_query == "ACGT"
    assert result.aligned_target == "ACCT"
    assert result.operations == "MMXM"


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
