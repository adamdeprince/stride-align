from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

import stride_align as sa

pytestmark = pytest.mark.duckdb

PythonScorer = Callable[[str, str], int | float]


@dataclass(frozen=True)
class Scorer:
    sql_name: str
    python_scorer: PythonScorer
    floating: bool = False
    equal_length_only: bool = False


def _smith_waterman_affine(left: str, right: str) -> int:
    return sa.smith_waterman_score(
        left,
        right,
        gap_open_score=-2,
        gap_extend_score=-1,
    )


def _needleman_wunsch_affine(left: str, right: str) -> int:
    return sa.needleman_wunsch_score(
        left,
        right,
        gap_open_score=-2,
        gap_extend_score=-1,
    )


SCORERS = (
    Scorer("stride_levenshtein", sa.levenshtein_score),
    Scorer("stride_levenshtein_similarity", sa.levenshtein_normalized_score, True),
    Scorer("stride_osa", sa.damerau_levenshtein_score),
    Scorer("stride_osa_similarity", sa.damerau_levenshtein_normalized_score, True),
    Scorer("stride_true_damerau_levenshtein", sa.true_damerau_levenshtein_score),
    Scorer(
        "stride_true_damerau_levenshtein_similarity",
        sa.true_damerau_levenshtein_normalized_score,
        True,
    ),
    Scorer("stride_indel", sa.indel_score),
    Scorer("stride_indel_similarity", sa.indel_normalized_score, True),
    Scorer("stride_hamming", sa.hamming_score, equal_length_only=True),
    Scorer(
        "stride_hamming_similarity",
        sa.hamming_normalized_score,
        floating=True,
        equal_length_only=True,
    ),
    Scorer("stride_jaro", sa.jaro_similarity, True),
    Scorer("stride_jaro_winkler", sa.jaro_winkler_similarity, True),
    Scorer("stride_smith_waterman", sa.smith_waterman_score),
    Scorer("stride_needleman_wunsch", sa.needleman_wunsch_score),
    Scorer("stride_smith_waterman_affine", _smith_waterman_affine),
    Scorer("stride_needleman_wunsch_affine", _needleman_wunsch_affine),
)

VARIABLE_LENGTH_SCORERS = tuple(scorer for scorer in SCORERS if not scorer.equal_length_only)
HAMMING_SCORERS = tuple(scorer for scorer in SCORERS if scorer.equal_length_only)

PARITY_CASES = (
    pytest.param("", "", id="both-empty"),
    pytest.param("abc", "", id="empty-right"),
    pytest.param("", "abc", id="empty-left"),
    pytest.param("abc", "abc", id="identity"),
    pytest.param("kitten", "sitting", id="classic-edit"),
    pytest.param("flaw", "lawn", id="insert-delete"),
    pytest.param("ca", "abc", id="true-damerau-vs-osa"),
    pytest.param("MARTHA", "MARHTA", id="jaro-transposition"),
    pytest.param("ACCGT", "ACG", id="alignment-gap"),
    pytest.param("Müller", "Mueller", id="latin1"),
    pytest.param("γειά", "γεια", id="greek-ucs2"),
    pytest.param("你好世界", "你好啊", id="cjk-ucs2"),
    pytest.param("hi 👋", "ho 👋", id="emoji-ucs4"),
    pytest.param("𝛼+𝛽", "𝛼+𝛾", id="math-ucs4"),
    pytest.param("a\x00b", "a\x00c", id="embedded-nul"),
    pytest.param("\uffffx", "\uffffy", id="max-bmp"),
    pytest.param("\U0010fffex", "\U0010fffey", id="astral-near-max"),
    pytest.param("नमस्ते", "नमस्कार", id="combining-marks"),
    pytest.param("👨‍👩‍👧", "👨‍👩‍👧‍👦", id="zwj-sequence"),
    pytest.param(
        "abcdefghij" * 10,
        "xbcdefghij" + "abcdefghij" * 9,
        id="long-ascii",
    ),
    pytest.param("你好" * 40, "你号" * 40, id="long-packed-unicode"),
)

HAMMING_CASES = (
    pytest.param("", "", id="both-empty"),
    pytest.param("abc", "abc", id="identity"),
    pytest.param("karolin", "kathrin", id="classic"),
    pytest.param("café", "cafe", id="latin1"),
    pytest.param("你好", "你号", id="ucs2"),
    pytest.param("🎉🎈ab", "🎈🎉ab", id="ucs4"),
    pytest.param("a\x00b", "a\x00c", id="embedded-nul"),
    pytest.param("a" * 127 + "b", "a" * 127 + "c", id="long-ascii"),
)


def _sql_scalar(connection: Any, scorer: Scorer, left: str, right: str) -> int | float:
    return connection.execute(
        f"SELECT {scorer.sql_name}(?, ?)",
        [left, right],
    ).fetchone()[0]


def _assert_matches(scorer: Scorer, actual: int | float, expected: int | float) -> None:
    if scorer.floating:
        assert actual == pytest.approx(expected, abs=1e-12)
    else:
        assert actual == expected


@pytest.mark.parametrize("scorer", VARIABLE_LENGTH_SCORERS, ids=lambda value: value.sql_name)
@pytest.mark.parametrize(("left", "right"), PARITY_CASES)
def test_duckdb_scalar_matches_python(
    duckdb_extension: Any,
    scorer: Scorer,
    left: str,
    right: str,
) -> None:
    expected = scorer.python_scorer(left, right)
    actual = _sql_scalar(duckdb_extension.connection, scorer, left, right)
    _assert_matches(scorer, actual, expected)


@pytest.mark.parametrize("scorer", HAMMING_SCORERS, ids=lambda value: value.sql_name)
@pytest.mark.parametrize(("left", "right"), HAMMING_CASES)
def test_duckdb_hamming_matches_python(
    duckdb_extension: Any,
    scorer: Scorer,
    left: str,
    right: str,
) -> None:
    expected = scorer.python_scorer(left, right)
    actual = _sql_scalar(duckdb_extension.connection, scorer, left, right)
    _assert_matches(scorer, actual, expected)


def _build_vector_rows() -> tuple[tuple[int, str, str], ...]:
    rows: list[tuple[int, str, str]] = []
    for index in range(2057):
        branch = index % 5
        if branch == 0:
            left = f"{index:08x}"
            replacement = "0" if left[-1] != "0" else "1"
            right = left[:-1] + replacement
        elif branch == 1:
            left, right = f"你{index % 10}好界", f"你{index % 10}号界"
        elif branch == 2:
            left, right = f"👋{index % 10}ab", f"🌸{index % 10}ab"
        elif branch == 3:
            left, right = f"𝛼+{index % 10}", f"𝛽+{index % 10}"
        else:
            left, right = f"a\x00{index % 10}", f"b\x00{index % 10}"
        rows.append((index, left, right))
    return tuple(rows)


@pytest.fixture(scope="module")
def duckdb_vector_rows(duckdb_extension: Any) -> tuple[tuple[int, str, str], ...]:
    rows = _build_vector_rows()
    connection = duckdb_extension.connection
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE stride_align_vector_cases "
        "(case_id INTEGER, left_text VARCHAR, right_text VARCHAR)"
    )
    connection.executemany(
        "INSERT INTO stride_align_vector_cases VALUES (?, ?, ?)",
        rows,
    )
    return rows


@pytest.mark.parametrize("scorer", SCORERS, ids=lambda value: value.sql_name)
def test_duckdb_vectorized_chunk_matches_python(
    duckdb_extension: Any,
    duckdb_vector_rows: tuple[tuple[int, str, str], ...],
    scorer: Scorer,
) -> None:
    actual = [
        row[0]
        for row in duckdb_extension.connection.execute(
            f"SELECT {scorer.sql_name}(left_text, right_text) "
            "FROM stride_align_vector_cases ORDER BY case_id"
        ).fetchall()
    ]
    expected = [scorer.python_scorer(left, right) for _, left, right in duckdb_vector_rows]
    if scorer.floating:
        assert actual == pytest.approx(expected, abs=1e-12)
    else:
        assert actual == expected


@pytest.mark.parametrize("scorer", SCORERS, ids=lambda value: value.sql_name)
def test_duckdb_functions_propagate_null(duckdb_extension: Any, scorer: Scorer) -> None:
    result = duckdb_extension.connection.execute(
        f"SELECT {scorer.sql_name}(NULL, 'x'), "
        f"{scorer.sql_name}('x', NULL), "
        f"{scorer.sql_name}(NULL, NULL)"
    ).fetchone()
    assert result == (None, None, None)


@pytest.mark.parametrize("scorer", HAMMING_SCORERS, ids=lambda value: value.sql_name)
def test_duckdb_hamming_rejects_unequal_codepoint_lengths(
    duckdb_extension: Any,
    scorer: Scorer,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    with pytest.raises(duckdb.InvalidInputException, match="equal-length strings"):
        _sql_scalar(duckdb_extension.connection, scorer, "short", "longer")


def test_duckdb_extension_catalog_is_exact(duckdb_extension: Any) -> None:
    registered = {
        row[0]
        for row in duckdb_extension.connection.execute(
            "SELECT DISTINCT function_name FROM duckdb_functions() "
            "WHERE function_name LIKE 'stride_%'"
        ).fetchall()
    }
    assert registered == {scorer.sql_name for scorer in SCORERS} | {"stride_align_simd_level"}


def test_duckdb_reports_loaded_artifact(duckdb_extension: Any) -> None:
    assert duckdb_extension.extension_path.is_file()
    assert duckdb_extension.duckdb_version == "1.5.5"
    assert duckdb_extension.simd_level in {
        "generic",
        "native",
        "avx2",
        "avx512bwvl",
        "neon",
        "la464_lsx",
        "la464_lasx",
        "la664_lasx",
    }


@pytest.mark.parametrize(
    "symbol_count",
    [
        pytest.param(200, id="packed-u8"),
        pytest.param(300, id="packed-u16"),
        pytest.param(66_000, id="packed-u32"),
    ],
)
def test_duckdb_long_unicode_uses_narrowest_lossless_tokens(
    duckdb_extension: Any,
    symbol_count: int,
) -> None:
    valid_codepoints = (
        list(range(0x1000, 0xD800))
        + list(range(0xE000, 0x10000))
        + list(range(0x10000, 0x10000 + 6_608))
    )
    text = "".join(chr(codepoint) for codepoint in valid_codepoints[:symbol_count])
    target = text if symbol_count < 66_000 else text[:2]
    expected = sa.levenshtein_score(text, target)
    actual = duckdb_extension.connection.execute(
        "SELECT stride_levenshtein(?, ?)",
        [text, target],
    ).fetchone()[0]
    assert actual == expected
