from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import stride_align as sa
from stride_align import matrices

pytestmark = pytest.mark.duckdb


def sql(connection: Any, statement: str, parameters: list[Any] | None = None) -> Any:
    return connection.execute(statement, parameters or []).fetchone()[0]


SCORER_FAMILIES = (
    ("levenshtein", "levenshtein_scores", sa.levenshtein_score, False),
    (
        "levenshtein_normalized",
        "levenshtein_normalized_scores",
        sa.levenshtein_normalized_score,
        True,
    ),
    ("damerau_levenshtein", "damerau_levenshtein_scores",
     sa.damerau_levenshtein_score, False),
    (
        "damerau_levenshtein_normalized",
        "damerau_levenshtein_normalized_scores",
        sa.damerau_levenshtein_normalized_score,
        True,
    ),
    ("true_damerau_levenshtein", "true_damerau_levenshtein_scores",
     sa.true_damerau_levenshtein_score, False),
    ("true_damerau_levenshtein_normalized",
     "true_damerau_levenshtein_normalized_scores",
     sa.true_damerau_levenshtein_normalized_score, True),
    ("indel", "indel_scores", sa.indel_score, False),
    ("indel_normalized", "indel_normalized_scores",
     sa.indel_normalized_score, True),
    ("hamming", "hamming_scores", sa.hamming_score, False),
    ("hamming_normalized", "hamming_normalized_scores",
     sa.hamming_normalized_score, True),
    ("jaro", "jaro_similarities", sa.jaro_similarity, True),
    ("jaro_winkler", "jaro_winkler_similarities",
     sa.jaro_winkler_similarity, True),
    ("smith_waterman", "smith_waterman_scores",
     sa.smith_waterman_score, True),
)


@pytest.mark.parametrize(
    ("stem", "scores_name", "python_score", "higher_is_better"),
    SCORER_FAMILIES,
)
def test_list_scores_and_top_k_match_python(
    duckdb_extension: Any,
    stem: str,
    scores_name: str,
    python_score: Callable[[str, str], int | float],
    higher_is_better: bool,
) -> None:
    query = "kitten" if stem != "hamming" and stem != "hamming_normalized" else "abcdef"
    targets = (
        ["kitten", "sitting", "bitten", "", "你好"]
        if "hamming" not in stem
        else ["abcdef", "abcxef", "xxxxxx", "abxdef"]
    )
    connection = duckdb_extension.connection
    actual = sql(connection, f"SELECT stride_{scores_name}(?, ?)", [query, targets])
    # Use the scalar API as the cross-language oracle. Some Python batch
    # backends intentionally reject or fall back on mixed-width Unicode,
    # while DuckDB natively accepts a heterogeneous UTF-8 list.
    expected = [python_score(query, target) for target in targets]
    assert actual == pytest.approx(expected, abs=1e-12)

    actual_ranked = sql(
        connection, f"SELECT stride_{stem}_top_k(?, ?, 3)", [query, targets]
    )
    # Python deliberately leaves order inside a top-k result unspecified.
    # DuckDB returns it deterministically, so compare the selected score
    # multiset and then validate every target/index association.
    expected_scores = sorted(expected, reverse=higher_is_better)[:3]
    actual_scores = sorted(
        (item["score"] for item in actual_ranked), reverse=higher_is_better
    )
    assert actual_scores == pytest.approx(expected_scores, abs=1e-12)
    for item in actual_ranked:
        assert targets[item["index"]] == item["target"]
        assert item["score"] == pytest.approx(expected[item["index"]], abs=1e-12)


@pytest.mark.parametrize(
    ("scorer", "python_scorer"),
    [
        ("levenshtein", sa.Scorer.LEVENSHTEIN),
        ("levenshtein_normalized", sa.Scorer.LEVENSHTEIN_NORMALIZED),
        ("damerau_levenshtein", sa.Scorer.DAMERAU_LEVENSHTEIN),
        ("damerau_levenshtein_normalized", sa.Scorer.DAMERAU_LEVENSHTEIN_NORMALIZED),
        ("hamming", sa.Scorer.HAMMING),
        ("hamming_normalized", sa.Scorer.HAMMING_NORMALIZED),
        ("jaro", sa.Scorer.JARO),
        ("jaro_winkler", sa.Scorer.JARO_WINKLER),
        ("indel", sa.Scorer.INDEL),
        ("indel_normalized", sa.Scorer.INDEL_NORMALIZED),
        ("true_damerau_levenshtein", sa.Scorer.TRUE_DAMERAU_LEVENSHTEIN),
        (
            "true_damerau_levenshtein_normalized",
            sa.Scorer.TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED,
        ),
        ("smith_waterman", sa.Scorer.SMITH_WATERMAN),
        ("smith_waterman_normalized", sa.Scorer.SMITH_WATERMAN_NORMALIZED),
        ("needleman_wunsch", sa.Scorer.NEEDLEMAN_WUNSCH),
        ("needleman_wunsch_normalized", sa.Scorer.NEEDLEMAN_WUNSCH_NORMALIZED),
    ],
)
def test_cdist_matches_python(
    duckdb_extension: Any,
    scorer: str,
    python_scorer: sa.Scorer,
) -> None:
    queries = ["hello", "world", "你好"]
    targets = ["hallo", "word", "你好啊"]
    if "hamming" in scorer:
        queries, targets = ["hello", "world"], ["hallo", "wordx"]
    actual = sql(
        duckdb_extension.connection,
        "SELECT stride_cdist(?, ?, ?)",
        [queries, targets, scorer],
    )
    scalar = {
        sa.Scorer.LEVENSHTEIN: sa.levenshtein_score,
        sa.Scorer.LEVENSHTEIN_NORMALIZED: sa.levenshtein_normalized_score,
        sa.Scorer.DAMERAU_LEVENSHTEIN: sa.damerau_levenshtein_score,
        sa.Scorer.DAMERAU_LEVENSHTEIN_NORMALIZED:
            sa.damerau_levenshtein_normalized_score,
        sa.Scorer.HAMMING: sa.hamming_score,
        sa.Scorer.HAMMING_NORMALIZED: sa.hamming_normalized_score,
        sa.Scorer.JARO: sa.jaro_similarity,
        sa.Scorer.JARO_WINKLER: sa.jaro_winkler_similarity,
        sa.Scorer.INDEL: sa.indel_score,
        sa.Scorer.INDEL_NORMALIZED: sa.indel_normalized_score,
        sa.Scorer.TRUE_DAMERAU_LEVENSHTEIN:
            sa.true_damerau_levenshtein_score,
        sa.Scorer.TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED:
            sa.true_damerau_levenshtein_normalized_score,
        sa.Scorer.SMITH_WATERMAN: sa.smith_waterman_score,
        sa.Scorer.SMITH_WATERMAN_NORMALIZED: sa.smith_waterman_normalized_score,
        sa.Scorer.NEEDLEMAN_WUNSCH: sa.needleman_wunsch_score,
        sa.Scorer.NEEDLEMAN_WUNSCH_NORMALIZED:
            sa.needleman_wunsch_normalized_score,
    }[python_scorer]
    expected = [[scalar(query, target) for target in targets] for query in queries]
    assert np.asarray(actual) == pytest.approx(np.asarray(expected), abs=1e-12)


def test_threshold_and_global_top_k_match_full_cdist(duckdb_extension: Any) -> None:
    connection = duckdb_extension.connection
    queries = ["kitten", "sitting", "kit"]
    targets = ["kitten", "kit", "sitting", "biting"]
    full = np.asarray(sa.cdist(queries, targets, scorer=sa.Scorer.JARO))
    threshold = 0.75
    actual = sql(
        connection,
        "SELECT stride_cdist_above_threshold(?, ?, 'jaro', ?)",
        [queries, targets, threshold],
    )
    assert {
        (item["query_index"], item["target_index"], item["score"]) for item in actual
    } == {
        (i, j, full[i, j])
        for i in range(len(queries))
        for j in range(len(targets))
        if full[i, j] >= threshold
    }

    ranked = sql(
        connection,
        "SELECT stride_cdist_top_k(?, ?, 'jaro', 5)",
        [queries, targets],
    )
    assert [item["score"] for item in ranked] == pytest.approx(
        sorted(full.ravel(), reverse=True)[:5], abs=1e-12
    )


@pytest.mark.parametrize(
    "scorer",
    ["levenshtein_normalized", "indel_normalized"],
)
def test_threshold_cutoff_keeps_exact_floating_point_boundary(
    duckdb_extension: Any,
    scorer: str,
) -> None:
    # (1 - 0.8) * 10 is mathematically two but is slightly below two as
    # a binary double. The cutoff must remain inclusive at that boundary.
    actual = sql(
        duckdb_extension.connection,
        "SELECT stride_cdist_above_threshold(?, ?, ?, 0.8)",
        [["aaaaaaaaaa"], ["aaaaaaaabb", "bbbbbbbbbb"], scorer],
    )
    assert [(item["target"], item["score"]) for item in actual] == [
        ("aaaaaaaabb", pytest.approx(0.8, abs=1e-12)),
    ]
    singular = f"stride_{scorer}_score"
    assert sql(
        duckdb_extension.connection,
        f"SELECT {singular}('aaaaaaaaaa', 'aaaaaaaabb', 0.8)",
    ) == pytest.approx(0.8, abs=1e-12)


def test_per_query_top_k_skips_bad_hamming_lengths(duckdb_extension: Any) -> None:
    actual = sql(
        duckdb_extension.connection,
        "SELECT stride_cdist_top_k_per_query(?, ?, 'hamming_normalized', 3, true)",
        [["abc", "wxyz"], ["abc", "abd", "wxyz", "bad-length"]],
    )
    assert [[match["target"] for match in row["matches"]] for row in actual] == [
        ["abc", "abd"],
        ["wxyz"],
    ]


@pytest.mark.parametrize(
    ("name", "python_function"),
    [
        ("jaccard", sa.jaccard),
        ("dice", sa.dice),
        ("cosine", sa.cosine),
        ("overlap", sa.overlap),
    ],
)
def test_ngram_functions_match_python(
    duckdb_extension: Any,
    name: str,
    python_function: Callable[..., float],
) -> None:
    actual = sql(
        duckdb_extension.connection,
        f"SELECT stride_{name}('ABCBDAB', 'BDCAB', 2)",
    )
    assert actual == pytest.approx(python_function("ABCBDAB", "BDCAB", n=2), abs=1e-12)
    targets = ["BDCAB", "ABCBDAB", "XYZ"]
    actual_many = sql(
        duckdb_extension.connection,
        f"SELECT stride_{name}_similarities(?, ?, 2)",
        ["ABCBDAB", targets],
    )
    assert actual_many == pytest.approx(
        [python_function("ABCBDAB", target, n=2) for target in targets],
        abs=1e-12,
    )


def test_lcs_ratcliff_token_and_monge_elkan_match_python(duckdb_extension: Any) -> None:
    connection = duckdb_extension.connection
    assert sql(connection, "SELECT stride_lcs_length('ABCBDAB', 'BDCAB')") == 4
    assert sql(connection, "SELECT stride_lcs_substring('Müller', 'Mueller')") == "ller"
    for name, function in (
        ("ratcliff_obershelp_similarity", sa.ratcliff_obershelp_similarity),
        ("partial_ratio", sa.partial_ratio),
        ("token_sort_ratio", sa.token_sort_ratio),
        ("token_set_ratio", sa.token_set_ratio),
        ("partial_token_sort_ratio", sa.partial_token_sort_ratio),
        ("partial_token_set_ratio", sa.partial_token_set_ratio),
        ("wratio", sa.WRatio),
    ):
        actual = sql(connection, f"SELECT stride_{name}(?, ?)", ["new york mets", "york mets new"])
        assert actual == pytest.approx(function("new york mets", "york mets new"), abs=1e-12)
    ratcliff_targets = ["york mets new", "new york yankees", ""]
    assert sql(
        connection,
        "SELECT stride_ratcliff_obershelp_similarities(?, ?)",
        ["new york mets", ratcliff_targets],
    ) == pytest.approx(
        [
            sa.ratcliff_obershelp_similarity("new york mets", target)
            for target in ratcliff_targets
        ],
        abs=1e-12,
    )
    fuzz_targets = ["york mets new", "new york yankees", ""]
    for singular, plural, function in (
        ("partial_ratio", "partial_ratios", sa.partial_ratio),
        ("token_sort_ratio", "token_sort_ratios", sa.token_sort_ratio),
        ("token_set_ratio", "token_set_ratios", sa.token_set_ratio),
        (
            "partial_token_sort_ratio", "partial_token_sort_ratios",
            sa.partial_token_sort_ratio,
        ),
        (
            "partial_token_set_ratio", "partial_token_set_ratios",
            sa.partial_token_set_ratio,
        ),
        ("wratio", "wratios", sa.WRatio),
    ):
        actual = sql(
            connection, f"SELECT stride_{plural}(?, ?)",
            ["new york mets", fuzz_targets],
        )
        expected = [function("new york mets", target) for target in fuzz_targets]
        assert actual == pytest.approx(expected, abs=1e-12), singular
    assert sql(
        connection,
        "SELECT stride_wratio('new york mets', 'new york yankees')",
    ) == pytest.approx(sa.WRatio("new york mets", "new york yankees"), abs=1e-12)
    actual = sql(
        connection,
        "SELECT stride_monge_elkan('hello world', 'hallo world', 'jaro_winkler', true)",
    )
    assert actual == pytest.approx(
        sa.monge_elkan("hello world", "hallo world", inner="jaro_winkler", symmetric=True),
        abs=1e-12,
    )


def test_phonetic_catalog_matches_python(duckdb_extension: Any) -> None:
    connection = duckdb_extension.connection
    assert sql(connection, "SELECT stride_soundex('Robert')") == sa.soundex("Robert")
    assert sql(connection, "SELECT stride_metaphone('Hugh', 1)") == sa.metaphone(
        "Hugh", variant=sa.MetaphoneVariant.JELLYFISH
    )
    assert sql(connection, "SELECT stride_nysiis('Watkins')") == sa.nysiis("Watkins")
    assert sql(connection, "SELECT stride_match_rating_codex('Christopher')") == sa.match_rating_codex(
        "Christopher"
    )
    assert sql(connection, "SELECT stride_caverphone('Stevenson')") == sa.caverphone("Stevenson")
    assert sql(connection, "SELECT stride_cologne_phonetic('Müller')") == sa.cologne_phonetic("Müller")
    assert sql(connection, "SELECT stride_daitch_mokotoff('Schwarz')") == sa.daitch_mokotoff("Schwarz")
    assert sql(connection, "SELECT stride_double_metaphone('Smith')") == {
        "primary": sa.double_metaphone("Smith")[0],
        "alternate": sa.double_metaphone("Smith")[1],
    }
    assert sql(connection, "SELECT stride_beider_morse('Renault', 1, true, 10)") == sa.beider_morse(
        "Renault", rule_type=sa.BmpmRuleType.EXACT, max_phonemes=10
    )


def test_dtw_and_alignment_paths_match_python(duckdb_extension: Any) -> None:
    connection = duckdb_extension.connection
    query = [0.0, 1.0, 2.0, 3.0]
    targets = [[0.0, 1.0, 2.0, 3.0], [0.0, 1.5, 2.5, 4.0]]
    actual = sql(connection, "SELECT stride_dtw_distances(?, ?)", [query, targets])
    expected = sa.dtw_distances(np.asarray(query), [np.asarray(value) for value in targets])
    assert actual == pytest.approx(expected, abs=1e-12)

    path = sql(connection, "SELECT stride_smith_waterman_path('ACCGT', 'ACG')")
    expected_path = sa.smith_waterman_path_info("ACCGT", "ACG")
    assert path["score"] == expected_path.score
    assert path["operations"] == expected_path.operations
    assert path["cigar"] == expected_path.cigar
    assert path["aligned_query"] == "ACCG"
    assert path["aligned_target"] == "A-CG"


@pytest.mark.parametrize(
    "matrix_name",
    ["blosum30", "blosum45", "blosum62", "blosum90", "pam10", "pam250", "pam500", "nuc44"],
)
def test_builtin_matrix_scores_match_python(
    duckdb_extension: Any,
    matrix_name: str,
) -> None:
    matrix = getattr(matrices, matrix_name)
    query, target = ("HEAGAWGHEE", "PAWHEAE") if matrix_name != "nuc44" else ("ACGT", "AGGT")
    actual = sql(
        duckdb_extension.connection,
        "SELECT stride_smith_waterman_matrix_score(?, ?, ?, -4)",
        [query, target, matrix_name],
    )
    expected = sa.smith_waterman_score(query, target, matrix=matrix, gap_score=-4)
    assert actual == expected


def test_matrix_catalog_and_keyboard_are_embedded(duckdb_extension: Any) -> None:
    connection = duckdb_extension.connection
    available = sql(connection, "SELECT stride_matrix_available()")
    assert {"blosum62", "pam500", "nuc44", "dna_match", "ascii_text", "keyboard:qwerty"} <= set(
        available
    )
    assert "qwerty" in sql(connection, "SELECT stride_keyboard_available()")
    assert sql(connection, "SELECT stride_matrix_encode('blosum62', 'HE')") == list(
        matrices.blosum62.encode("HE")
    )
    actual = sql(
        connection,
        "SELECT stride_smith_waterman_matrix_score('teh', 'the', 'keyboard:qwerty')",
    )
    expected = sa.smith_waterman_score(
        "teh", "the", matrix=matrices.keyboard.qwerty,
        gap_open_score=matrices.keyboard.qwerty.gap_score,
        gap_extend_score=matrices.keyboard.qwerty.gap_score,
    )
    assert actual == expected


def test_parameterized_alignment_and_jaro_winkler_match_python(duckdb_extension: Any) -> None:
    connection = duckdb_extension.connection
    actual = sql(
        connection,
        "SELECT stride_smith_waterman_score('AGCT', 'AGGT', 5, -3, -5, -1)",
    )
    assert actual == sa.smith_waterman_score(
        "AGCT", "AGGT", match_score=5, mismatch_score=-3,
        gap_open_score=-5, gap_extend_score=-1,
    )
    actual = sql(
        connection,
        "SELECT stride_jaro_winkler_similarity('MARTHA', 'MARHTA', 0.2, 0.6, 3)",
    )
    assert actual == pytest.approx(
        sa.jaro_winkler_similarity(
            "MARTHA", "MARHTA", prefix_weight=0.2,
            prefix_threshold=0.6, prefix_cap=3,
        ),
        abs=1e-12,
    )


def test_optional_argument_forms_match_python(duckdb_extension: Any) -> None:
    connection = duckdb_extension.connection
    query = "ACGT"
    targets = ["ACGT", "AGGT", "TTTT"]

    expected_scores = sa.smith_waterman_scores(
        query, targets, match_score=5, mismatch_score=-3, gap_score=-2
    )
    assert sql(
        connection,
        "SELECT stride_smith_waterman_scores(?, ?, 5, -3, -2)",
        [query, targets],
    ) == expected_scores.tolist()
    path = sql(
        connection,
        "SELECT stride_smith_waterman_path(?, ?, 5, -3, -2)",
        [query, targets[1]],
    )
    expected_path = sa.smith_waterman_path_info(
        query, targets[1], match_score=5, mismatch_score=-3, gap_score=-2
    )
    assert (path["score"], path["cigar"]) == (
        expected_path.score, expected_path.cigar
    )

    expected_cdist = np.asarray(
        [[
            sa.smith_waterman_score(
                left, right, match_score=5, mismatch_score=-3, gap_score=-2
            )
            for right in targets
        ] for left in [query, "AAAA"]]
    )
    actual_cdist = sql(
        connection,
        "SELECT stride_cdist(?, ?, 'smith_waterman', 5, -3, -2)",
        [[query, "AAAA"], targets],
    )
    assert np.asarray(actual_cdist) == pytest.approx(expected_cdist)

    float_query = [0.0, 1.0, 2.0, 3.0]
    float_target = [0.0, 1.5, 2.5, 4.0]
    assert sql(
        connection, "SELECT stride_dtw(?, ?, 2)",
        [float_query, float_target],
    ) == pytest.approx(
        sa.dtw(np.asarray(float_query), np.asarray(float_target), window=2)
    )
    assert sql(
        connection, "SELECT stride_dtw_distances(?, ?, 2, 'l1')",
        [float_query, [float_target]],
    ) == pytest.approx(
        sa.dtw_distances(
            np.asarray(float_query), [np.asarray(float_target)],
            window=2, distance="l1",
        )
    )

    assert sql(
        connection, "SELECT stride_daitch_mokotoff('GERSCHFELD', false)"
    ) == sa.daitch_mokotoff("GERSCHFELD", branching=False)
    assert sql(
        connection, "SELECT stride_beider_morse('Renault', 1, false)"
    ) == sa.beider_morse(
        "Renault", rule_type=sa.BmpmRuleType.EXACT, concat=False
    )


def test_distance_cutoffs_and_zero_top_k_match_python(duckdb_extension: Any) -> None:
    connection = duckdb_extension.connection
    targets = ["sitting", "kitten", "completely different"]
    assert sql(
        connection,
        "SELECT stride_levenshtein_scores('kitten', ?, 2)",
        [targets],
    ) == sa.levenshtein_scores("kitten", targets, score_cutoff=2).tolist()
    assert sql(
        connection,
        "SELECT stride_indel_normalized_scores('kitten', ?, 0.8)",
        [targets],
    ) == pytest.approx(
        [
            score if score >= 0.8 else 0.0
            for score in sa.indel_normalized_scores("kitten", targets)
        ],
        abs=1e-12,
    )
    assert sql(
        connection,
        "SELECT stride_levenshtein_top_k('kitten', ?, 0)",
        [targets],
    ) == []
    assert sql(
        connection,
        "SELECT stride_cdist_top_k(['kitten'], ?, 'jaro', 0)",
        [targets],
    ) == []


def test_cdist_option_overloads_reach_optimized_paths(duckdb_extension: Any) -> None:
    connection = duckdb_extension.connection
    queries = ["MARTHA", "DWAYNE"]
    targets = ["MARHTA", "DUANE", "MARTIN"]
    expected = np.asarray(
        [[
            sa.jaro_winkler_similarity(
                query, target, prefix_weight=0.2,
                prefix_threshold=0.6, prefix_cap=3,
            )
            for target in targets
        ] for query in queries]
    )
    actual = sql(
        connection,
        "SELECT stride_cdist_top_k(?, ?, 'jaro_winkler', 3, false, 0.2, 0.6, 3)",
        [queries, targets],
    )
    assert [item["score"] for item in actual] == pytest.approx(
        sorted(expected.ravel(), reverse=True)[:3], abs=1e-12
    )
    per_query = sql(
        connection,
        "SELECT stride_cdist_top_k_per_query(?, ?, 'jaro_winkler', 2, true, 0.2, 0.6, 3)",
        [queries, targets],
    )
    for query_index, row in enumerate(per_query):
        assert [item["score"] for item in row["matches"]] == pytest.approx(
            sorted(expected[query_index], reverse=True)[:2], abs=1e-12
        )


def test_custom_matrix_values_round_trip_and_score(duckdb_extension: Any) -> None:
    connection = duckdb_extension.connection
    expression = (
        "stride_substitution_matrix(" 
        "'DNA_CUSTOM', 'ACN', "
        "[[5::BIGINT,-4::BIGINT,-4::BIGINT],"
        "[-4::BIGINT,5::BIGINT,-4::BIGINT],"
        "[-4::BIGINT,-4::BIGINT,-4::BIGINT]], -5, 'N')"
    )
    info = sql(connection, f"SELECT stride_matrix_info({expression})")
    assert info == {
        "name": "DNA_CUSTOM", "alphabet": "ACN", "stride": 3,
        "wildcard_index": 2, "gap_score": -5,
        "gap_open": None, "gap_extend": None,
    }
    assert sql(
        connection,
        f"SELECT stride_substitution_matrix_score({expression}, 'AC', 'AC')",
    ) == 10
    assert sql(
        connection,
        f"SELECT stride_matrix_info(stride_matrix_transpose({expression})).name",
    ) == "DNA_CUSTOM.T"


def test_matrix_constructors_handle_defaults_and_null_affine_gaps(
    duckdb_extension: Any,
) -> None:
    connection = duckdb_extension.connection
    identity = sql(
        connection,
        "SELECT stride_identity_matrix('ACGT', 5, -4, 'N', 'DNA', -5, NULL, NULL)",
    )
    assert identity["name"] == "DNA"
    assert identity["alphabet"] == "ACGTN"
    assert identity["gap_open"] is None
    assert identity["gap_extend"] is None
    assert sql(
        connection,
        "SELECT stride_matrix_info(stride_ascii_matrix()).stride",
    ) == 128
    text = "A X\nA 2 -1\nX -1 -1\n"
    assert sql(
        connection,
        "SELECT stride_matrix_info(stride_matrix_from_ncbi_text(?)).alphabet",
        [text],
    ) == "AX"


def test_keyboard_matrix_builders_match_python(duckdb_extension: Any) -> None:
    connection = duckdb_extension.connection
    counts = [[0.0, 4.0], [2.0, 0.0]]
    python_matrix = matrices.keyboard.from_confusion_counts(
        np.asarray(counts), alphabet="A*", name="KEY", wildcard="*"
    )
    expression = (
        "stride_keyboard_from_confusion_counts(" 
        "?::DOUBLE[][], 'A*', 'KEY', 2.0, 4, NULL, '*', -1, NULL, NULL)"
    )
    info = sql(connection, f"SELECT stride_matrix_info({expression})", [counts])
    assert info["name"] == python_matrix.name
    assert info["alphabet"] == python_matrix.alphabet
    assert info["wildcard_index"] == python_matrix.alphabet.index(
        python_matrix.wildcard
    )
    actual = sql(
        connection,
        f"SELECT stride_smith_waterman_matrix_score('AA', 'A*', {expression})",
        [counts],
    )
    expected = sa.smith_waterman_score(
        "AA", "A*", matrix=python_matrix, gap_score=-1
    )
    assert actual == expected

    npy_path = (
        Path(matrices.keyboard.__file__).with_name("keyboard_data") / "qwerty.npy"
    )
    npy = npy_path.read_bytes()
    alphabet = matrices.keyboard.ASCII_ALPHABET
    wildcard = matrices.keyboard.ASCII_WILDCARD
    assert sql(
        connection,
        "SELECT stride_matrix_info("
        "stride_keyboard_from_npy(?, 'QWERTY', ?, ?, false, -1, NULL, NULL)"
        ").stride",
        [npy, alphabet, wildcard],
    ) == 128


def test_matrix_threshold_and_top_k_skip_materializing_cdist(
    duckdb_extension: Any,
) -> None:
    connection = duckdb_extension.connection
    queries = ["ACGT", "AAAA"]
    targets = ["ACGT", "AGGT", "TTTT"]
    full = np.asarray(
        sql(
            connection,
            "SELECT stride_cdist_matrix_local(?, ?, 'dna_match')",
            [queries, targets],
        )
    )
    ranked = sql(
        connection,
        "SELECT stride_cdist_matrix_top_k(?, ?, 'dna_match', true, 3)",
        [queries, targets],
    )
    assert [item["score"] for item in ranked] == sorted(
        full.ravel(), reverse=True
    )[:3]
    threshold = 5
    above = sql(
        connection,
        "SELECT stride_cdist_matrix_above_threshold(?, ?, 'dna_match', true, ?)",
        [queries, targets, threshold],
    )
    assert {
        (item["query_index"], item["target_index"], item["score"])
        for item in above
    } == {
        (query_index, target_index, full[query_index, target_index])
        for query_index in range(len(queries))
        for target_index in range(len(targets))
        if full[query_index, target_index] >= threshold
    }


def test_dtw_smallint_uses_python_integer_default(duckdb_extension: Any) -> None:
    connection = duckdb_extension.connection
    query = np.asarray([0, 1, 0], dtype=np.int16)
    target = np.asarray([0, 0, 1], dtype=np.int16)
    actual = sql(
        connection,
        "SELECT stride_dtw(?::SMALLINT[], ?::SMALLINT[])",
        [query.tolist(), target.tolist()],
    )
    assert actual == sa.dtw(query, target)
