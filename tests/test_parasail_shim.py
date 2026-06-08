"""parasail-compatible shim — ``import stride_align.parasail as parasail``.

Pins the behavioural contract for the most common parasail entry
points: scores match upstream, attribute shapes match upstream, the
gap convention is BLAST (``cost(N) = open + (N-1)*extend``, positive
values), pre-built matrices are wired up, and the SIMD-suffix variants
resolve via ``__getattr__`` to the underlying core entry.

Marked ``skipif`` so the suite still runs on machines without the
upstream ``parasail`` PyPI package installed — but when it's there,
we cross-check against it.
"""

from __future__ import annotations

import pytest

import stride_align.parasail as shim

# Real parasail is in the ``bench`` extra; treat as optional.
upstream = pytest.importorskip("parasail")


# --------------------------------------------------------------------
# Score parity with upstream
# --------------------------------------------------------------------

class TestScoreParity:
    """Scores match parasail bit-for-bit on the cases where SW / NW
    have a unique optimum. The BLOSUM62 ``HEAGAWGHEE``/``PAWHEAE``
    case has multiple optima with score 17 — both implementations
    agree on the score but may report different alignments."""

    def test_nw_dna_clean_match(self) -> None:
        m_up = upstream.matrix_create("ACGT", 2, -1)
        m_sh = shim.matrix_create("ACGT", 2, -1)
        r_up = upstream.nw("ACGTAC", "ACGTAC", 5, 2, m_up)
        r_sh = shim.nw("ACGTAC", "ACGTAC", 5, 2, m_sh)
        assert r_up.score == r_sh.score == 12

    def test_nw_dna_one_mismatch(self) -> None:
        m_up = upstream.matrix_create("ACGT", 2, -1)
        m_sh = shim.matrix_create("ACGT", 2, -1)
        r_up = upstream.nw("ACGTAC", "ACATAC", 5, 2, m_up)
        r_sh = shim.nw("ACGTAC", "ACATAC", 5, 2, m_sh)
        assert r_up.score == r_sh.score

    def test_nw_dna_with_gap(self) -> None:
        m_up = upstream.matrix_create("ACGT", 2, -1)
        m_sh = shim.matrix_create("ACGT", 2, -1)
        # A gap is needed; verifies BLAST gap convention.
        r_up = upstream.nw("ACGT", "ACGTAC", 5, 2, m_up)
        r_sh = shim.nw("ACGT", "ACGTAC", 5, 2, m_sh)
        assert r_up.score == r_sh.score

    def test_sw_dna_finds_substring(self) -> None:
        m_up = upstream.matrix_create("ACGT", 2, -1)
        m_sh = shim.matrix_create("ACGT", 2, -1)
        r_up = upstream.sw("ACGTAC", "XXACGTACXX", 5, 2, m_up)
        r_sh = shim.sw("ACGTAC", "XXACGTACXX", 5, 2, m_sh)
        assert r_up.score == r_sh.score == 12

    @pytest.mark.parametrize("open_p, extend_p", [(5, 2), (10, 1), (1, 1), (8, 4)])
    def test_nw_gap_open_extend_translate_correctly(self, open_p, extend_p) -> None:
        # Same call with a range of gap parameters — pins the
        # negate-at-the-boundary translation from positive parasail
        # to negative stride-align.
        m_up = upstream.matrix_create("AB", 2, -1)
        m_sh = shim.matrix_create("AB", 2, -1)
        r_up = upstream.nw("AABB", "AABBAA", open_p, extend_p, m_up)
        r_sh = shim.nw("AABB", "AABBAA", open_p, extend_p, m_sh)
        assert r_up.score == r_sh.score, (open_p, extend_p, r_up.score, r_sh.score)


# --------------------------------------------------------------------
# Result shape
# --------------------------------------------------------------------

class TestResultShape:
    """The shim's ``Result``, ``Cigar``, ``Traceback`` classes expose
    every attribute parasail user code is likely to read."""

    def test_score_only_result_has_end_query_end_ref(self) -> None:
        m = shim.matrix_create("ACGT", 2, -1)
        r = shim.sw("ACGT", "ACGT", 5, 2, m)
        assert r.score == 8
        # Inclusive last index — parasail convention.
        assert r.end_query == 3
        assert r.end_ref == 3
        # Score-only does not populate cigar / traceback / stats.
        assert r.cigar is None
        assert r.traceback is None
        assert r.matches is None

    def test_trace_result_populates_cigar_and_traceback(self) -> None:
        m = shim.matrix_create("ACGT", 2, -1)
        r = shim.sw_trace("ACGT", "ACGT", 5, 2, m)
        assert r.cigar is not None
        assert r.cigar.decode == b"4="
        assert r.cigar.beg_query == 0
        assert r.cigar.beg_ref == 0
        assert r.cigar.len == 2  # len(b'4=') == 2 -- parasail-compat property
        assert r.traceback is not None
        assert r.traceback.query == "ACGT"
        assert r.traceback.ref == "ACGT"
        assert r.traceback.comp == "||||"

    def test_trace_cigar_with_mismatch_and_gap(self) -> None:
        m = shim.matrix_create("ACGT", 2, -1)
        r = shim.nw_trace("ACGTAC", "ACATAC", 5, 2, m)
        # One mismatch in the middle (G vs A) → CIGAR is '2=1X3='.
        assert r.cigar.decode == b"2=1X3="
        assert r.traceback.comp == "||.|||"

    def test_stats_result_has_matches_length_similar(self) -> None:
        m = shim.matrix_create("ACGT", 2, -1)
        r = shim.sw_stats("ACGT", "ACGT", 5, 2, m)
        assert r.matches == 4
        assert r.length == 4
        # No mismatches with a positive-only score for AT-vs-AT.
        assert r.similar == 4


# --------------------------------------------------------------------
# Stats parity with upstream
# --------------------------------------------------------------------

class TestStatsParity:
    """``_stats`` parity for cases where the optimal alignment is
    unique (so the alignment-derived counts can be cross-checked)."""

    def test_sw_stats_clean_match(self) -> None:
        m_up = upstream.matrix_create("ACGT", 2, -1)
        m_sh = shim.matrix_create("ACGT", 2, -1)
        r_up = upstream.sw_stats("ACGTAC", "ACGTAC", 5, 2, m_up)
        r_sh = shim.sw_stats("ACGTAC", "ACGTAC", 5, 2, m_sh)
        assert r_up.score == r_sh.score
        assert r_up.matches == r_sh.matches
        assert r_up.length == r_sh.length
        assert r_up.similar == r_sh.similar


# --------------------------------------------------------------------
# Pre-built matrices
# --------------------------------------------------------------------

class TestPrebuiltMatrices:
    """The pre-built ``blosum*`` and ``pam*`` matrices share the
    parasail.Matrix shape (size, name, mapper, min/max)."""

    @pytest.mark.parametrize("name", [
        "blosum45", "blosum50", "blosum62", "blosum80", "blosum90",
        "pam30", "pam70", "pam250",
    ])
    def test_matrix_attributes_present(self, name) -> None:
        m = getattr(shim, name)
        assert isinstance(m, shim.Matrix)
        assert m.size > 0
        assert m.matrix.shape == (m.size, m.size)
        assert m.mapper.shape == (256,)
        assert m.name == name.encode("ascii")

    def test_blosum62_max_abs_matches_upstream(self) -> None:
        # Pre-built matrices should give identical max/min to upstream.
        up = upstream.blosum62
        sh = shim.blosum62
        assert up.max == sh.max
        assert up.min == sh.min
        assert up.size == sh.size

    def test_blosum62_sw_score_matches_upstream(self) -> None:
        # Score parity on a known peptide pair.
        r_up = upstream.sw("HEAGAW", "HEAGAW", 11, 1, upstream.blosum62)
        r_sh = shim.sw("HEAGAW", "HEAGAW", 11, 1, shim.blosum62)
        assert r_up.score == r_sh.score


# --------------------------------------------------------------------
# matrix_create
# --------------------------------------------------------------------

class TestMatrixCreate:
    def test_size_matches_upstream(self) -> None:
        # Upstream adds a wildcard slot too.
        assert shim.matrix_create("ACGT", 2, -1).size == upstream.matrix_create("ACGT", 2, -1).size

    def test_set_value(self) -> None:
        m = shim.matrix_create("ACGT", 2, -1)
        m.set_value(0, 1, 5)
        assert int(m.matrix[0, 1]) == 5

    def test_copy_returns_independent(self) -> None:
        m = shim.matrix_create("ACGT", 2, -1)
        c = m.copy()
        c.set_value(0, 0, 99)
        assert int(c.matrix[0, 0]) == 99
        assert int(m.matrix[0, 0]) == 2  # original unchanged

    def test_case_sensitive_true_maps_lowercase_to_wildcard(self) -> None:
        m = shim.matrix_create("ACGT", 2, -1, case_sensitive=True)
        wildcard_idx = m.size - 1
        assert int(m.mapper[ord("a")]) == wildcard_idx

    def test_case_sensitive_false_folds_lowercase(self) -> None:
        m = shim.matrix_create("ACGT", 2, -1, case_sensitive=False)
        # 'a' and 'A' map to the same alphabet index.
        assert int(m.mapper[ord("a")]) == int(m.mapper[ord("A")])


# --------------------------------------------------------------------
# Kernel-suffix alias dispatch
# --------------------------------------------------------------------

class TestKernelAliases:
    """The 2000+ parasail kernel-suffix variants
    (``sw_striped_avx2_16`` etc.) all alias to the matching core
    entry — stride-align picks the kernel internally."""

    @pytest.mark.parametrize("name, expected_base", [
        ("sw_striped_avx2_16", "sw"),
        ("sw_scan_64",         "sw"),
        ("sw_diag_sat",        "sw"),
        ("nw_striped_sse41_8", "nw"),
        ("nw_scan",            "nw"),
        ("sw_trace_striped_avx2_16",   "sw_trace"),
        ("nw_trace_diag_sat",          "nw_trace"),
        ("sw_stats_scan_profile_8",    "sw_stats"),
        ("sg_striped_avx2_16",         "sg"),
    ])
    def test_kernel_suffix_aliases_to_core(self, name, expected_base) -> None:
        fn = getattr(shim, name)
        assert fn.__name__ == expected_base

    def test_unknown_attribute_raises(self) -> None:
        with pytest.raises(AttributeError):
            shim.this_is_not_a_parasail_entry_point

    def test_aliased_call_returns_same_score(self) -> None:
        m = shim.matrix_create("ACGT", 2, -1)
        r_core    = shim.sw("ACGT", "ACGT", 5, 2, m)
        r_aliased = shim.sw_striped_avx2_16("ACGT", "ACGT", 5, 2, m)
        assert r_core.score == r_aliased.score


# --------------------------------------------------------------------
# Capability probes
# --------------------------------------------------------------------

class TestCapabilityProbes:
    """``can_use_*`` reflects what the loaded stride-align backend
    actually supports, and matches the upstream value where both are
    derivable from the same hardware."""

    def test_capability_set_matches_upstream(self) -> None:
        for fn_name in ["can_use_sse2", "can_use_sse41", "can_use_avx2",
                        "can_use_altivec", "can_use_neon"]:
            assert getattr(shim, fn_name)() == getattr(upstream, fn_name)(), fn_name


# --------------------------------------------------------------------
# Drop-in import contract
# --------------------------------------------------------------------

class TestDropInImport:
    """Spot-check the import-line replacement: a snippet of real
    parasail user code runs unchanged after substituting the import."""

    def test_protein_alignment_smoke(self) -> None:
        # Equivalent of: import parasail; parasail.sw_trace(...).
        # Use the shim through the recommended import alias.
        import stride_align.parasail as parasail
        r = parasail.sw_trace("HEAGAW", "PAWHE", 11, 1, parasail.blosum62)
        assert r.score >= 0
        assert isinstance(r.cigar.decode, bytes)
        assert isinstance(r.traceback.query, str)
        assert isinstance(r.traceback.ref, str)
        assert isinstance(r.traceback.comp, str)

    def test_dna_alignment_smoke(self) -> None:
        import stride_align.parasail as parasail
        m = parasail.matrix_create("ACGT", 2, -1)
        r = parasail.nw_trace("ACGT", "ACAT", 5, 2, m)
        assert r.score is not None
        assert r.cigar.decode == b"2=1X1="
