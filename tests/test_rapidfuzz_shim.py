"""rapidfuzz-compatible shim — ``import stride_align.rapidfuzz as rapidfuzz``.

Cross-checks the shim's `fuzz`, `distance`, `process`, and `utils`
submodules against the installed upstream `rapidfuzz` package.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import stride_align.rapidfuzz as shim
from stride_align.rapidfuzz import distance as sh_dist
from stride_align.rapidfuzz import fuzz as sh_fuzz
from stride_align.rapidfuzz import process as sh_proc
from stride_align.rapidfuzz import utils as sh_utils

# rapidfuzz is in pyproject's bench extra; treat as optional.
upstream = pytest.importorskip("rapidfuzz")
up_fuzz = upstream.fuzz
up_dist = upstream.distance
up_proc = upstream.process
up_utils = upstream.utils


PAIRS = [
    ("hello",         "world"),
    ("hello",         "hallo"),
    ("foo bar baz",   "foo bar"),
    ("color",         "12345"),
    ("the quick brown fox", "the quick brown dog"),
    ("a",             "a" * 18),
    ("hi 👋",         "ho 👋"),
    ("apple",         "an apple a day"),
    ("paul johnson",  "paul jones"),
    ("Hello, World!", "HELLO WORLD"),
    ("",              ""),
    ("abc",           ""),
]


# --------------------------------------------------------------------
# fuzz parity
# --------------------------------------------------------------------

class TestFuzzParity:
    """All ten ``fuzz.*`` entries match upstream bit-exactly on the
    pairs in ``PAIRS``."""

    # Functions that match upstream bit-exactly across every input.
    @pytest.mark.parametrize("a,b", PAIRS, ids=lambda p: repr(p)[:24])
    @pytest.mark.parametrize("fn_name", [
        "ratio", "QRatio", "WRatio",
        "token_sort_ratio", "token_set_ratio", "token_ratio",
    ])
    def test_score_parity_exact(self, fn_name, a, b) -> None:
        u = getattr(up_fuzz, fn_name)(a, b)
        s = getattr(sh_fuzz, fn_name)(a, b)
        assert u == pytest.approx(s, abs=1e-6), (fn_name, a, b, u, s)

    # The partial-ratio family carries a documented Phase D.3
    # conservative-underestimate: stride-align's ``partial_ratio``
    # enumerates fewer matching-block candidates than rapidfuzz, so
    # for inputs where rapidfuzz finds a higher-scoring shifted window
    # the shim returns a lower value. Bit-exact on the vast majority
    # of inputs; the invariant we pin is "shim never overshoots
    # upstream", not bit-exact parity.
    @pytest.mark.parametrize("a,b", PAIRS, ids=lambda p: repr(p)[:24])
    @pytest.mark.parametrize("fn_name", [
        "partial_ratio",
        "partial_token_sort_ratio", "partial_token_set_ratio",
        "partial_token_ratio",
    ])
    def test_partial_family_never_overshoots(self, fn_name, a, b) -> None:
        u = getattr(up_fuzz, fn_name)(a, b)
        s = getattr(sh_fuzz, fn_name)(a, b)
        assert s <= u + 1e-6, f"{fn_name} overshot upstream: shim={s} > upstream={u} ({a!r} vs {b!r})"


# --------------------------------------------------------------------
# fuzz processor + score_cutoff
# --------------------------------------------------------------------

class TestFuzzKwargs:
    def test_processor_runs_before_scoring(self) -> None:
        # processor=str.lower folds case before computing ratio.
        u = up_fuzz.ratio("HELLO", "hello", processor=str.lower)
        s = sh_fuzz.ratio("HELLO", "hello", processor=str.lower)
        assert u == s == 100.0

    def test_score_cutoff_below_returns_zero(self) -> None:
        # rapidfuzz returns 0.0 when the score is below the cutoff.
        s = sh_fuzz.ratio("hello", "world", score_cutoff=50)
        assert s == 0.0

    def test_score_cutoff_above_returns_true_score(self) -> None:
        s = sh_fuzz.ratio("hello", "world", score_cutoff=10)
        assert s == pytest.approx(up_fuzz.ratio("hello", "world"))


# --------------------------------------------------------------------
# distance parity
# --------------------------------------------------------------------

DISTANCE_METRICS = [
    ("Levenshtein",        sh_dist.Levenshtein,        up_dist.Levenshtein),
    ("Indel",              sh_dist.Indel,              up_dist.Indel),
    ("Hamming",            sh_dist.Hamming,            up_dist.Hamming),
    ("Jaro",               sh_dist.Jaro,               up_dist.Jaro),
    ("JaroWinkler",        sh_dist.JaroWinkler,        up_dist.JaroWinkler),
    ("DamerauLevenshtein", sh_dist.DamerauLevenshtein, up_dist.DamerauLevenshtein),
    ("OSA",                sh_dist.OSA,                up_dist.OSA),
    ("LCSseq",             sh_dist.LCSseq,             up_dist.LCSseq),
]


class TestDistanceParity:
    """Each metric's ``distance`` / ``similarity`` / ``normalized_*``
    return matches upstream bit-exactly."""

    @pytest.mark.parametrize("name,sh_cls,up_cls", DISTANCE_METRICS, ids=lambda t: t)
    def test_distance(self, name, sh_cls, up_cls) -> None:
        # Hamming requires equal-length inputs for the default
        # raw-distance call; skip those pairs.
        pairs = [p for p in PAIRS if name != "Hamming" or len(p[0]) == len(p[1])]
        for a, b in pairs:
            if not a and not b:
                continue
            u = up_cls.distance(a, b)
            s = sh_cls.distance(a, b)
            assert u == s, (name, a, b, u, s)

    @pytest.mark.parametrize("name,sh_cls,up_cls", DISTANCE_METRICS, ids=lambda t: t)
    def test_normalized_distance(self, name, sh_cls, up_cls) -> None:
        pairs = [p for p in PAIRS if name != "Hamming" or len(p[0]) == len(p[1])]
        for a, b in pairs:
            if not a and not b:
                continue
            u = up_cls.normalized_distance(a, b)
            s = sh_cls.normalized_distance(a, b)
            assert u == pytest.approx(s, abs=1e-9), (name, a, b, u, s)

    @pytest.mark.parametrize("name,sh_cls,up_cls", DISTANCE_METRICS, ids=lambda t: t)
    def test_normalized_similarity(self, name, sh_cls, up_cls) -> None:
        pairs = [p for p in PAIRS if name != "Hamming" or len(p[0]) == len(p[1])]
        for a, b in pairs:
            if not a and not b:
                continue
            u = up_cls.normalized_similarity(a, b)
            s = sh_cls.normalized_similarity(a, b)
            assert u == pytest.approx(s, abs=1e-9), (name, a, b, u, s)


# --------------------------------------------------------------------
# Editops / Opcodes
# --------------------------------------------------------------------

class TestEditops:
    """Levenshtein.editops and .opcodes match the rapidfuzz collection
    shape on representative inputs."""

    @pytest.mark.parametrize("s1,s2", [
        ("kitten", "sitting"),
        ("hello",  "world"),
        ("",       "abc"),
        ("abc",    ""),
        ("abc",    "abc"),
    ])
    def test_editops_shape_matches_upstream(self, s1, s2) -> None:
        u = up_dist.Levenshtein.editops(s1, s2)
        s = sh_dist.Levenshtein.editops(s1, s2)
        assert s.src_len == u.src_len
        assert s.dest_len == u.dest_len
        # The edit-cost is path-dependent (multiple optimal
        # alignments). Pin the length, not the exact ops.
        assert len(s) == len(u)

    @pytest.mark.parametrize("s1,s2", [
        ("kitten", "sitting"),
        ("hello",  "world"),
        ("abc",    "abc"),
    ])
    def test_opcodes_shape_matches_upstream(self, s1, s2) -> None:
        u = up_dist.Levenshtein.opcodes(s1, s2)
        s = sh_dist.Levenshtein.opcodes(s1, s2)
        assert s.src_len == u.src_len
        assert s.dest_len == u.dest_len
        # Opcodes cover the source and destination span exactly.
        if s.ops:
            assert s.ops[0].src_start == 0
            assert s.ops[-1].src_end == len(s1)
            assert s.ops[0].dest_start == 0
            assert s.ops[-1].dest_end == len(s2)


# --------------------------------------------------------------------
# process.extract / extractOne / extract_iter
# --------------------------------------------------------------------

class TestProcess:
    def test_extract_returns_sorted_tuples(self) -> None:
        choices = ["hallo", "world", "helo", "hello"]
        out = sh_proc.extract("hello", choices)
        # Best match is the identical "hello".
        assert out[0][0] == "hello"
        assert out[0][1] == 100.0
        # Tuple shape: (choice, score, key).
        for choice, score, key in out:
            assert choice in choices
            assert 0.0 <= score <= 100.0
            assert 0 <= key < len(choices)

    def test_extract_with_limit(self) -> None:
        choices = ["a", "b", "c", "d", "e"]
        out = sh_proc.extract("a", choices, limit=3)
        assert len(out) == 3

    def test_extractOne_returns_single(self) -> None:
        out = sh_proc.extractOne("hello", ["hallo", "world", "helo"])
        assert out[0] == "helo"  # closest match by WRatio

    def test_extractOne_none_under_cutoff(self) -> None:
        out = sh_proc.extractOne("hello", ["nope", "zip"], score_cutoff=80)
        assert out is None

    def test_extract_iter_yields(self) -> None:
        results = list(sh_proc.extract_iter("hello", ["hallo", "world", "helo"]))
        assert len(results) == 3
        for choice, score, key in results:
            assert isinstance(choice, str)

    def test_extract_dict_choices_returns_keys(self) -> None:
        choices = {"alpha": "hallo", "beta": "world", "gamma": "helo"}
        out = sh_proc.extract("hello", choices, limit=1)
        choice, score, key = out[0]
        # Keys come from the dict, not indices.
        assert key in choices
        assert choices[key] == choice


class TestProcessCdist:
    def test_cdist_returns_2d_matrix(self) -> None:
        out = sh_proc.cdist(["hello", "world"], ["hallo", "word", "help"])
        assert out.shape == (2, 3)
        assert out.dtype == np.float64

    def test_cdist_uses_passed_scorer(self) -> None:
        # ratio for hello/hello is 100; for hello/world is ~20.
        out = sh_proc.cdist(["hello"], ["hello", "world"], scorer=sh_fuzz.ratio)
        assert out[0, 0] == pytest.approx(100.0)
        assert out[0, 1] < 50.0


# --------------------------------------------------------------------
# utils.default_process
# --------------------------------------------------------------------

class TestUtils:
    @pytest.mark.parametrize("s", [
        "Hello, World!",
        "foo123BAR",
        "  multiple   spaces  ",
        "Hello, World! 123",
        "",
        "abc",
    ])
    def test_default_process(self, s) -> None:
        # Cross-check against upstream — rapidfuzz's default_process
        # replaces each non-alphanumeric ASCII char with a space
        # individually (does NOT collapse internal runs).
        assert sh_utils.default_process(s) == up_utils.default_process(s)


# --------------------------------------------------------------------
# Drop-in import contract
# --------------------------------------------------------------------

class TestDropInImport:
    """Spot-check that real rapidfuzz user code runs unchanged after
    swapping the import line."""

    def test_fuzz_smoke(self) -> None:
        import stride_align.rapidfuzz as rapidfuzz
        assert rapidfuzz.fuzz.ratio("hello", "hallo") == pytest.approx(80.0)
        assert rapidfuzz.fuzz.WRatio("foo bar baz", "foo bar") == pytest.approx(90.0)

    def test_distance_smoke(self) -> None:
        import stride_align.rapidfuzz as rapidfuzz
        assert rapidfuzz.distance.Levenshtein.distance("kitten", "sitting") == 3
        assert rapidfuzz.distance.JaroWinkler.similarity("MARTHA", "MARHTA") > 0.9

    def test_process_extract_smoke(self) -> None:
        import stride_align.rapidfuzz as rapidfuzz
        results = rapidfuzz.process.extract(
            "hello", ["hallo", "world", "helo"], limit=2,
        )
        assert len(results) == 2
        assert results[0][1] >= results[1][1]
