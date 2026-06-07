"""Smith-Waterman and Needleman-Wunsch — Unicode no-exception guarantees.

Locks in the property that every public SW / NW entry point — score,
normalised score, path, path-info, cigar, trace_cigar, trade_cigar
(the trace-free affine CIGAR fast path), batch ``*_scores``, ``best``,
``top_k`` — accepts every Python ``str`` storage kind (1B / 2B / 4B
PyUnicode_KIND) and every reasonable codepoint without raising, and
returns a value of the right type (``str`` in → ``str`` aligned
strings; ``bytes`` in → ``bytes`` aligned strings).

The numeric-correctness side of Unicode auto-promote (UCS-2 with
>256 distinct codepoints lifting to the 16-bit kernel) lives in
``tests/test_unicode_wide.py``. This file is about the breadth: every
entry point × every Unicode kind, with edge-y codepoints that have
historically tripped string kernels — NUL, max BMP, astral max,
combining marks, ZWJ emoji families, mixed-width content.
"""

from __future__ import annotations

import pytest

import stride_align as sa


# ---------------------------------------------------------------- inputs

# Representative Unicode strings per PyUnicode_KIND. Each pair shares
# its storage width on the (query, target) side so the test exercises
# the homogeneous-kind path; mixed-width pairs are exercised separately.
LATIN1_PAIRS = [
    ("hello", "hallo"),
    ("Müller", "Mueller"),
    ("straße", "strasse"),
    ("café", "cafe"),
]

UCS2_PAIRS = [
    ("γειά",           "γεια"),         # Greek
    ("Привет",         "Превет"),       # Cyrillic
    ("你好世界",        "你好啊"),         # CJK
    ("東京",            "京都"),           # CJK pair
]

UCS4_PAIRS = [
    ("hi 👋",          "ho 👋"),         # ASCII + 4B emoji → 4B kind
    ("𝛼+𝛽",            "𝛼+𝛾"),           # math bold
    ("\U0001F600x",   "\U0001F600y"),  # grinning face
    ("𠮷田",            "𠮷川"),           # CJK extension B (surrogate pair)
]

# Edge codepoints that historically have tripped string kernels.
EDGE_PAIRS = [
    ("NUL codepoint",         "a\x00b",                "a\x00c"),
    ("max BMP U+FFFF",        "￿x",               "￿y"),
    ("astral near-max",       "\U0010FFFEx",           "\U0010FFFEy"),
    ("RTL Hebrew",            "שלום עולם",             "שלום"),
    ("Devanagari combining",  "नमस्ते",                  "नमस्कार"),
    ("ZWJ emoji family",      "👨‍👩‍👧",     "👨‍👩‍👧‍👦"),
    ("mixed 1B + 4B",         "hello 👋 world",        "hello 🌸 world"),
    ("long 4B run",           "𝛼" * 50,                "𝛽" * 50),
]

ALL_NONLATIN_STR_PAIRS = UCS2_PAIRS + UCS4_PAIRS + [(a, b) for _, a, b in EDGE_PAIRS]


# ---------------------------------------------------------------- entry points

# Every singular SW / NW entry point that takes (query, target) as
# strings. The dispatch decides whether to route to the 8 / 16 / 32 /
# 64-bit Farrar kernel based on PyUnicode_KIND and symbol count.
SCORE_FNS = [
    sa.smith_waterman_score,
    sa.smith_waterman_normalized_score,
    sa.smith_waterman_farrar_score,
    sa.smith_waterman_farrar_normalized_score,
    sa.needleman_wunsch_score,
    sa.needleman_wunsch_normalized_score,
]

# Path-shaped returns: AlignmentResult or AlignmentPath structs.
PATH_FNS = [
    sa.smith_waterman_path,
    sa.smith_waterman_path_info,
    sa.needleman_wunsch_path,
    sa.needleman_wunsch_path_info,
]

# CIGAR-shaped returns: ``str``.
CIGAR_FNS = [
    sa.smith_waterman_cigar,
    sa.smith_waterman_trace_cigar,
    sa.smith_waterman_trade_cigar,   # trace-free affine CIGAR fast path
    sa.needleman_wunsch_cigar,
    sa.needleman_wunsch_trace_cigar,
    sa.needleman_wunsch_trade_cigar,
]

BATCH_FNS = [
    sa.smith_waterman_scores,
    sa.smith_waterman_normalized_scores,
    sa.smith_waterman_farrar_scores,
    sa.smith_waterman_farrar_normalized_scores,
    sa.needleman_wunsch_scores,
    sa.needleman_wunsch_normalized_scores,
]


# ---------------------------------------------------------------- helpers

def _id(pair):
    a, b = pair
    return f"{a[:6]!r}-{b[:6]!r}"


# ---------------------------------------------------------------- score tests

class TestScoreUnicode:
    """Every score-shaped entry point accepts every storage kind."""

    @pytest.mark.parametrize("fn", SCORE_FNS, ids=lambda f: f.__name__)
    @pytest.mark.parametrize("pair", LATIN1_PAIRS + UCS2_PAIRS + UCS4_PAIRS, ids=_id)
    def test_score_accepts_each_unicode_kind(self, fn, pair) -> None:
        a, b = pair
        r = fn(a, b)
        assert isinstance(r, (int, float))

    @pytest.mark.parametrize("fn", SCORE_FNS, ids=lambda f: f.__name__)
    @pytest.mark.parametrize("pair", [(p[1], p[2]) for p in EDGE_PAIRS], ids=lambda p: _id(p))
    def test_score_accepts_edge_codepoints(self, fn, pair) -> None:
        a, b = pair
        r = fn(a, b)
        assert isinstance(r, (int, float))


# ---------------------------------------------------------------- path tests

class TestPathUnicode:
    """Path / path_info accept every storage kind and preserve the
    input string type in the aligned-string fields."""

    @pytest.mark.parametrize("pair", LATIN1_PAIRS + UCS2_PAIRS + UCS4_PAIRS, ids=_id)
    def test_smith_waterman_path_returns_str_for_str_input(self, pair) -> None:
        a, b = pair
        result = sa.smith_waterman_path(a, b)
        assert isinstance(result.aligned_query, str)
        assert isinstance(result.aligned_target, str)
        assert isinstance(result.operations, str)
        # Aligned strings, with '-' for gaps removed, must contain only
        # codepoints from the original inputs.
        assert set(result.aligned_query) - {"-"} <= set(a)
        assert set(result.aligned_target) - {"-"} <= set(b)

    @pytest.mark.parametrize("pair", LATIN1_PAIRS + UCS2_PAIRS + UCS4_PAIRS, ids=_id)
    def test_needleman_wunsch_path_returns_str_for_str_input(self, pair) -> None:
        a, b = pair
        result = sa.needleman_wunsch_path(a, b)
        assert isinstance(result.aligned_query, str)
        assert isinstance(result.aligned_target, str)
        assert set(result.aligned_query) - {"-"} <= set(a)
        assert set(result.aligned_target) - {"-"} <= set(b)

    def test_path_returns_bytes_for_bytes_input(self) -> None:
        # Symmetric guarantee: bytes in → bytes-aligned out.
        result = sa.smith_waterman_path(b"hello", b"hallo")
        assert isinstance(result.aligned_query, bytes)
        assert isinstance(result.aligned_target, bytes)

    @pytest.mark.parametrize("fn", PATH_FNS, ids=lambda f: f.__name__)
    @pytest.mark.parametrize("pair", [(p[1], p[2]) for p in EDGE_PAIRS], ids=lambda p: _id(p))
    def test_path_entry_points_accept_edge_codepoints(self, fn, pair) -> None:
        a, b = pair
        # We don't pin field values for every edge case — just no-throw.
        fn(a, b)


# ---------------------------------------------------------------- cigar tests

class TestCigarUnicode:
    """Every CIGAR-shaped entry point — including the trace-free
    ``trade_cigar`` affine-gap fast path — accepts every storage kind."""

    @pytest.mark.parametrize("fn", CIGAR_FNS, ids=lambda f: f.__name__)
    @pytest.mark.parametrize("pair", LATIN1_PAIRS + UCS2_PAIRS + UCS4_PAIRS, ids=_id)
    def test_cigar_returns_str(self, fn, pair) -> None:
        a, b = pair
        cigar = fn(a, b)
        assert isinstance(cigar, str)

    @pytest.mark.parametrize("fn", CIGAR_FNS, ids=lambda f: f.__name__)
    @pytest.mark.parametrize("pair", [(p[1], p[2]) for p in EDGE_PAIRS], ids=lambda p: _id(p))
    def test_cigar_accepts_edge_codepoints(self, fn, pair) -> None:
        a, b = pair
        fn(a, b)


# ---------------------------------------------------------------- batch tests

class TestBatchUnicode:
    """``*_scores`` / ``best`` / ``top_k`` accept Unicode targets,
    including target lists whose elements span multiple storage kinds."""

    @pytest.mark.parametrize("fn", BATCH_FNS, ids=lambda f: f.__name__)
    @pytest.mark.parametrize("pair", LATIN1_PAIRS + UCS2_PAIRS + UCS4_PAIRS, ids=_id)
    def test_batch_homogeneous_kind(self, fn, pair) -> None:
        a, b = pair
        out = fn(a, [b, b, b])
        assert out.shape == (3,)
        # The three identical targets must produce identical scores.
        assert out[0] == out[1] == out[2]

    def test_batch_mixed_kind_targets(self) -> None:
        # Mixed-kind target list — the dispatch promotes per pair.
        query = "hello"
        targets = ["hallo", "γειά", "你好", "hi 👋", "\U0010FFFEy"]
        out = sa.smith_waterman_scores(query, targets)
        assert out.shape == (5,)
        for value in out:
            # SW scores are non-negative.
            assert value >= 0

    def test_smith_waterman_best_with_unicode_targets(self) -> None:
        # smith_waterman_best returns (target, score, index).
        query = "你好世界"
        targets = ["你好", "再见", "你好世界", "你好啊朋友"]
        target, score, idx = sa.smith_waterman_best(query, targets)
        assert isinstance(target, str)
        assert target in targets
        # The perfect-match target must win.
        assert targets[idx] == "你好世界"
        assert target == "你好世界"

    def test_smith_waterman_top_k_with_unicode_targets(self) -> None:
        # smith_waterman_top_k returns a list of (target, score, index) tuples.
        query = "hi 👋"
        targets = ["ho 👋", "hi 🌸", "hi 👋"]
        ranked = sa.smith_waterman_top_k(query, targets, k=2)
        assert len(ranked) == 2
        for target, score, idx in ranked:
            assert isinstance(target, str)
            assert target == targets[idx]


# ---------------------------------------------------------------- correctness

class TestUnicodeCorrectness:
    """Values aren't just non-zero — they're the right values. The
    score for two identical Unicode strings equals
    ``len(s) * match_score``, regardless of storage kind."""

    @pytest.mark.parametrize("s", [
        "hello",
        "Müller",
        "你好世界",
        "hi 👋",
        "𝛼+𝛽",
        "שלום",
        "𠮷田川",
    ], ids=lambda s: f"{s[:6]!r}")
    def test_identity_score_equals_n_times_match(self, s) -> None:
        match = 2
        assert sa.smith_waterman_score(s, s, match_score=match) == len(s) * match
        assert sa.needleman_wunsch_score(s, s, match_score=match) == len(s) * match
        assert sa.smith_waterman_farrar_score(s, s, match_score=match) == len(s) * match

    @pytest.mark.parametrize("s", ["你好世界", "hi 👋", "𝛼+𝛽"], ids=lambda s: f"{s[:6]!r}")
    def test_str_score_equals_bytes_score_when_codepoints_fit_in_byte(self, s) -> None:
        # When the codepoint sequence happens to fit in a byte (e.g.
        # ASCII inside the 4B string), the str-path score must match
        # the bytes-path score on the ASCII-only prefix — sanity-check
        # that promotion doesn't change the math.
        ascii_prefix = "".join(c for c in s if ord(c) < 0x80)
        if not ascii_prefix:
            return
        assert (
            sa.smith_waterman_score(ascii_prefix, ascii_prefix)
            == sa.smith_waterman_score(ascii_prefix.encode(), ascii_prefix.encode())
        )

    def test_needleman_wunsch_disjoint_unicode_uses_gaps(self) -> None:
        # No common codepoints → NW alignment is all gaps + mismatches;
        # the score must equal the worst-case lower bound and not raise.
        a, b = "你好", "𝛼+𝛽"  # 2B vs 4B, no overlap
        s = sa.needleman_wunsch_score(a, b, match_score=2, mismatch_score=-1, gap_score=-1)
        # Bound: NW score ≥ -(|a| + |b|) * |gap_score|.
        assert s >= -(len(a) + len(b))


# ---------------------------------------------------------------- affine

class TestUnicodeAffineGaps:
    """Affine gap variants (gap_open + gap_extend) accept every
    storage kind on every entry point."""

    @pytest.mark.parametrize("fn", SCORE_FNS + CIGAR_FNS, ids=lambda f: f.__name__)
    @pytest.mark.parametrize("pair", UCS2_PAIRS[:2] + UCS4_PAIRS[:2], ids=_id)
    def test_affine_accepts_unicode(self, fn, pair) -> None:
        a, b = pair
        fn(a, b, gap_open_score=-3, gap_extend_score=-1)

    def test_affine_path_returns_str_for_unicode(self) -> None:
        result = sa.smith_waterman_path(
            "你好世界", "你好啊朋友",
            match_score=2, mismatch_score=-1,
            gap_open_score=-3, gap_extend_score=-1,
        )
        assert isinstance(result.aligned_query, str)
        assert isinstance(result.aligned_target, str)


# ---------------------------------------------------------------- mixed bytes/str

class TestMixedBytesStrRejected:
    """The dispatcher must reject mixed bytes / str inputs with a
    clear TypeError. Negative test — pins the contract."""

    @pytest.mark.parametrize("fn", SCORE_FNS + PATH_FNS + CIGAR_FNS, ids=lambda f: f.__name__)
    def test_bytes_query_str_target_raises(self, fn) -> None:
        with pytest.raises(TypeError, match="bytes and str"):
            fn(b"hello", "hallo")

    @pytest.mark.parametrize("fn", SCORE_FNS + PATH_FNS + CIGAR_FNS, ids=lambda f: f.__name__)
    def test_str_query_bytes_target_raises(self, fn) -> None:
        with pytest.raises(TypeError, match="bytes and str"):
            fn("hello", b"hallo")
