from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Protocol

import pytest

pytestmark = pytest.mark.postgres

ROOT = Path(__file__).resolve().parents[2]


class PostgresServer(Protocol):
    major: int

    def command(
        self,
        sql: str,
        *,
        database: str = "postgres",
        tuples_only: bool = True,
    ) -> str: ...

    def create_database(self, name: str, encoding: str) -> None: ...


def result_json(server: PostgresServer, expression: str, *, database: str = "postgres"):
    return json.loads(server.command(f"SELECT to_jsonb({expression});", database=database))


def test_catalog_matches_the_duckdb_native_surface(postgres_server: PostgresServer) -> None:
    expected = sorted(
        set(
            re.findall(
                r"`(stride_[a-z0-9_]+)",
                (ROOT / "bindings" / "duckdb" / "FUNCTIONS.md").read_text(),
            )
        )
    )
    actual = json.loads(
        postgres_server.command(
            """
            SELECT json_agg(proname ORDER BY proname)
            FROM (
              SELECT DISTINCT p.proname
              FROM pg_proc AS p
              JOIN pg_namespace AS n ON n.oid = p.pronamespace
              WHERE n.nspname = 'public' AND p.proname LIKE 'stride_%'
            ) AS functions;
            """
        )
    )
    assert actual == expected
    assert (
        postgres_server.command(
            "SELECT extversion FROM pg_extension WHERE extname = 'stride_align';"
        )
        == "0.6.0"
    )


def test_pair_batch_rank_and_path_operations(postgres_server: PostgresServer) -> None:
    actual = result_json(
        postgres_server,
        """
        jsonb_build_object(
          'distance', stride_levenshtein('kitten', 'sitting'),
          'unicode', stride_levenshtein('😀x', '😀y'),
          'scores', to_jsonb(stride_levenshtein_scores(
              'kitten', ARRAY['kitten', 'sitting', NULL, 'bitten'])),
          'cdist', to_jsonb(stride_cdist(
              ARRAY['kitten', 'sitting'], ARRAY['kitten', 'bitten'], 'jaro')),
          'top_k', stride_cdist_top_k(
              ARRAY['kitten', 'sitting'], ARRAY['kitten', 'bitten'], 'jaro', 2),
          'path', stride_smith_waterman_path('ACCGT', 'ACG')
        )
        """,
    )
    assert actual["distance"] == 3
    assert actual["unicode"] == 1
    assert actual["scores"] == [0, 3, None, 1]
    assert len(actual["cdist"]) == 2
    assert len(actual["top_k"]) == 2
    assert actual["path"]["aligned_query"] == "ACCG"
    assert actual["path"]["aligned_target"] == "A-CG"


def test_matrix_catalog_custom_values_and_vector_forms(
    postgres_server: PostgresServer,
) -> None:
    actual = result_json(
        postgres_server,
        """
        jsonb_build_object(
          'available', to_jsonb(stride_matrix_available()),
          'keyboard', to_jsonb(stride_keyboard_available()),
          'encoded', to_jsonb(stride_matrix_encode('blosum62', 'HE')),
          'protein_score', stride_smith_waterman_matrix_score(
              'HEAGAWGHEE', 'PAWHEAE', 'blosum62', -4),
          'scores', to_jsonb(stride_smith_waterman_matrix_scores(
              'ACGT', ARRAY['ACGT', 'AGGT', NULL], 'dna_match')),
          'cdist', to_jsonb(stride_cdist_matrix_local(
              ARRAY['ACGT', 'AAAA'], ARRAY['ACGT', 'AGGT', 'TTTT'], 'dna_match')),
          'top_k', stride_cdist_matrix_top_k(
              ARRAY['ACGT', 'AAAA'], ARRAY['ACGT', 'AGGT', 'TTTT'],
              'dna_match', true, 3),
          'path', stride_smith_waterman_matrix_path('ACCGT', 'ACG', 'dna_match')
        )
        """,
    )
    assert {"blosum62", "pam500", "nuc44", "dna_match", "keyboard:qwerty"} <= set(
        actual["available"]
    )
    assert "qwerty" in actual["keyboard"]
    assert actual["encoded"] == [8, 6]
    assert actual["protein_score"] == 25
    assert actual["scores"] == [20, 13, None]
    assert actual["cdist"] == [[20, 13, 5], [5, 5, 0]]
    assert [match["score"] for match in actual["top_k"]] == [20, 13, 5]
    assert actual["path"]["cigar"] == "1=1D2="

    custom = result_json(
        postgres_server,
        """
        (WITH matrix_value AS (
          SELECT stride_substitution_matrix(
            'DNA_CUSTOM', 'ACN',
            ARRAY[[5,-4,-4],[-4,5,-4],[-4,-4,-4]]::integer[],
            -5, 'N') AS value
        )
        SELECT jsonb_build_object(
          'info', stride_matrix_info(value),
          'score', stride_substitution_matrix_score(value, 'AC', 'AC'),
          'transpose', stride_matrix_info(stride_matrix_transpose(value))->>'name'
        ) FROM matrix_value)
        """,
    )
    assert custom["info"]["alphabet"] == "ACN"
    assert custom["info"]["wildcard_index"] == 2
    assert custom["score"] == 10
    assert custom["transpose"] == "DNA_CUSTOM.T"


def _npy_int8_2x2(values: bytes) -> bytes:
    header = b"{'descr': '|i1', 'fortran_order': False, 'shape': (2, 2), }"
    prefix_size = 10
    padding = (16 - ((prefix_size + len(header) + 1) % 16)) % 16
    header += b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + values


def _public_smoke_expressions() -> dict[str, str]:
    expressions = {
        "stride_align_simd_level": "stride_align_simd_level()",
        "stride_available_backends": "stride_available_backends()",
        "stride_backend_is_available": "stride_backend_is_available('native')",
        "stride_detect_best_backend": "stride_detect_best_backend()",
    }

    pair_names = (
        "levenshtein",
        "levenshtein_score",
        "levenshtein_similarity",
        "levenshtein_normalized_score",
        "osa",
        "osa_similarity",
        "damerau_levenshtein_score",
        "damerau_levenshtein_normalized_score",
        "true_damerau_levenshtein",
        "true_damerau_levenshtein_score",
        "true_damerau_levenshtein_similarity",
        "true_damerau_levenshtein_normalized_score",
        "indel",
        "indel_score",
        "indel_similarity",
        "indel_normalized_score",
        "hamming",
        "hamming_score",
        "hamming_similarity",
        "hamming_normalized_score",
        "jaro",
        "jaro_similarity",
        "jaro_winkler",
        "jaro_winkler_similarity",
    )
    for name in pair_names:
        arguments = "'abcdef', 'abcxef'" if "hamming" in name else "'kitten', 'sitting'"
        expressions[f"stride_{name}"] = f"stride_{name}({arguments})"

    alignment_names = (
        "smith_waterman",
        "smith_waterman_affine",
        "smith_waterman_score",
        "smith_waterman_normalized_score",
        "smith_waterman_farrar_score",
        "smith_waterman_farrar_normalized_score",
        "needleman_wunsch",
        "needleman_wunsch_affine",
        "needleman_wunsch_score",
        "needleman_wunsch_normalized_score",
    )
    for name in alignment_names:
        expressions[f"stride_{name}"] = f"stride_{name}('ACCGT', 'ACG')"

    vector_names = (
        "levenshtein_scores",
        "levenshtein_normalized_scores",
        "damerau_levenshtein_scores",
        "damerau_levenshtein_normalized_scores",
        "true_damerau_levenshtein_scores",
        "true_damerau_levenshtein_normalized_scores",
        "indel_scores",
        "indel_normalized_scores",
        "hamming_scores",
        "hamming_normalized_scores",
        "jaro_similarities",
        "jaro_winkler_similarities",
        "smith_waterman_scores",
        "smith_waterman_normalized_scores",
        "smith_waterman_farrar_scores",
        "smith_waterman_farrar_normalized_scores",
        "needleman_wunsch_scores",
        "needleman_wunsch_normalized_scores",
    )
    for name in vector_names:
        if "hamming" in name:
            query, targets = "'abcdef'", "ARRAY['abcdef', 'abcxef']"
        else:
            query, targets = "'kitten'", "ARRAY['kitten', 'sitting']"
        expressions[f"stride_{name}"] = f"stride_{name}({query}, {targets})"
    expressions["stride_scores"] = (
        "stride_scores('kitten', ARRAY['kitten', 'sitting'], 'levenshtein')"
    )

    ranking_stems = (
        "levenshtein",
        "levenshtein_normalized",
        "damerau_levenshtein",
        "damerau_levenshtein_normalized",
        "true_damerau_levenshtein",
        "true_damerau_levenshtein_normalized",
        "indel",
        "indel_normalized",
        "hamming",
        "hamming_normalized",
        "jaro",
        "jaro_winkler",
        "smith_waterman",
    )
    for stem in ranking_stems:
        if "hamming" in stem:
            query, targets = "'abcdef'", "ARRAY['abcdef', 'abcxef']"
        else:
            query, targets = "'kitten'", "ARRAY['kitten', 'sitting']"
        expressions[f"stride_{stem}_top_k"] = f"stride_{stem}_top_k({query}, {targets}, 1)"
        expressions[f"stride_{stem}_best"] = f"stride_{stem}_best({query}, {targets})"
    expressions["stride_extract"] = (
        "stride_extract('kitten', ARRAY['kitten', 'sitting'], 'jaro', 1)"
    )
    expressions["stride_extract_best"] = (
        "stride_extract_best('kitten', ARRAY['kitten', 'sitting'], 'jaro')"
    )

    expressions.update(
        {
            "stride_cdist": ("stride_cdist(ARRAY['kitten'], ARRAY['sitting'], 'levenshtein')"),
            "stride_cdist_above_threshold": (
                "stride_cdist_above_threshold(ARRAY['kitten'], ARRAY['sitting'], 'jaro', 0.5)"
            ),
            "stride_cdist_top_k": (
                "stride_cdist_top_k(ARRAY['kitten'], ARRAY['sitting'], 'jaro', 1)"
            ),
            "stride_cdist_top_k_per_query": (
                "stride_cdist_top_k_per_query(ARRAY['kitten'], ARRAY['sitting'], 'jaro', 1)"
            ),
            "stride_lcs_length": "stride_lcs_length('ABCBDAB', 'BDCAB')",
            "stride_lcs_substring_length": ("stride_lcs_substring_length('ABCBDAB', 'BDCAB')"),
            "stride_lcs_substring": "stride_lcs_substring('ABCBDAB', 'BDCAB')",
            "stride_monge_elkan": (
                "stride_monge_elkan('hello world', 'hallo world', 'jaro', true)"
            ),
        }
    )

    for name in ("jaccard", "dice", "cosine", "overlap"):
        expressions[f"stride_{name}"] = f"stride_{name}('ABCBDAB', 'BDCAB', 2)"
        expressions[f"stride_{name}_similarities"] = (
            f"stride_{name}_similarities('ABCBDAB', ARRAY['BDCAB'], 2)"
        )
    similarity_names = (
        "ratcliff_obershelp_similarity",
        "partial_ratio",
        "token_sort_ratio",
        "token_set_ratio",
        "partial_token_sort_ratio",
        "partial_token_set_ratio",
        "wratio",
    )
    for name in similarity_names:
        expressions[f"stride_{name}"] = f"stride_{name}('new york mets', 'york mets new')"
    plural_names = (
        "ratcliff_obershelp_similarities",
        "partial_ratios",
        "token_sort_ratios",
        "token_set_ratios",
        "partial_token_sort_ratios",
        "partial_token_set_ratios",
        "wratios",
    )
    for name in plural_names:
        expressions[f"stride_{name}"] = f"stride_{name}('new york mets', ARRAY['york mets new'])"

    expressions.update(
        {
            "stride_soundex": "stride_soundex('Robert')",
            "stride_soundex_equal": "stride_soundex_equal('Robert', 'Rupert')",
            "stride_metaphone": "stride_metaphone('Schmidt')",
            "stride_metaphone_equal": "stride_metaphone_equal('Smith', 'Smyth')",
            "stride_nysiis": "stride_nysiis('Macdonald')",
            "stride_nysiis_equal": "stride_nysiis_equal('Smith', 'Smyth')",
            "stride_match_rating_codex": "stride_match_rating_codex('Smith')",
            "stride_match_rating_compare": "stride_match_rating_compare('Smith', 'Smyth')",
            "stride_caverphone": "stride_caverphone('Stevenson')",
            "stride_cologne_phonetic": "stride_cologne_phonetic('Müller')",
            "stride_daitch_mokotoff": "stride_daitch_mokotoff('Schneider')",
            "stride_double_metaphone": "stride_double_metaphone('Schmidt')",
            "stride_beider_morse": "stride_beider_morse('Schneider')",
            "stride_dtw": (
                "stride_dtw(ARRAY[0.0,1.0,2.0]::double precision[], "
                "ARRAY[0.0,1.0,2.0]::double precision[])"
            ),
            "stride_dtw_distances": (
                "stride_dtw_distances(ARRAY[0.0,1.0]::double precision[], "
                "ARRAY[[0.0,1.0],[1.0,0.0]]::double precision[])"
            ),
        }
    )

    for stem in ("smith_waterman", "needleman_wunsch"):
        for suffix in ("path", "path_info", "cigar", "trace_cigar", "trade_cigar"):
            name = f"stride_{stem}_{suffix}"
            expressions[name] = f"{name}('ACCGT', 'ACG')"

    custom_matrix = (
        "stride_substitution_matrix('DNA_CUSTOM', 'ACN', "
        "ARRAY[[5,-4,-4],[-4,5,-4],[-4,-4,-4]]::integer[], -5, 'N')"
    )
    ncbi = "E'A X\\nA 2 -1\\nX -1 -1\\n'"
    npy = _npy_int8_2x2(bytes([5, 0xFC, 0xFC, 5])).hex()
    expressions.update(
        {
            "stride_substitution_matrix": custom_matrix,
            "stride_ascii_matrix": "stride_ascii_matrix()",
            "stride_identity_matrix": "stride_identity_matrix('ACGT', 5, -4, 'N')",
            "stride_matrix_available": "stride_matrix_available()",
            "stride_matrix_info": "stride_matrix_info('blosum62')",
            "stride_matrix_encode": "stride_matrix_encode('blosum62', 'HE')",
            "stride_matrix_transpose": "stride_matrix_transpose('dna_match')",
            "stride_matrix_from_ncbi_text": f"stride_matrix_from_ncbi_text({ncbi})",
            "stride_matrix_score_step_limit": "stride_matrix_score_step_limit('blosum62')",
            "stride_substitution_matrix_score": (
                "stride_substitution_matrix_score('dna_match', 'AC', 'AC')"
            ),
            "stride_keyboard_available": "stride_keyboard_available()",
            "stride_keyboard_from_confusion_counts": (
                "stride_keyboard_from_confusion_counts("
                "ARRAY[[0.0,4.0],[2.0,0.0]]::double precision[], 'A*')"
            ),
            "stride_keyboard_from_npy": (
                f"stride_keyboard_from_npy(decode('{npy}', 'hex'), 'KEY', 'A*', '*')"
            ),
        }
    )
    for stem in ("smith_waterman", "needleman_wunsch"):
        expressions[f"stride_{stem}_matrix_score"] = (
            f"stride_{stem}_matrix_score('ACGT', 'AGGT', 'dna_match')"
        )
        expressions[f"stride_{stem}_matrix_scores"] = (
            f"stride_{stem}_matrix_scores('ACGT', ARRAY['ACGT','AGGT'], 'dna_match')"
        )
        expressions[f"stride_{stem}_matrix_path"] = (
            f"stride_{stem}_matrix_path('ACGT', 'AGGT', 'dna_match')"
        )
        expressions[f"stride_{stem}_matrix_path_info"] = (
            f"stride_{stem}_matrix_path_info('ACGT', 'AGGT', 'dna_match')"
        )
        expressions[f"stride_{stem}_matrix_cigar"] = (
            f"stride_{stem}_matrix_cigar('ACGT', 'AGGT', 'dna_match')"
        )
    expressions.update(
        {
            "stride_cdist_matrix_local": (
                "stride_cdist_matrix_local(ARRAY['ACGT'], ARRAY['AGGT'], 'dna_match')"
            ),
            "stride_cdist_matrix_global": (
                "stride_cdist_matrix_global(ARRAY['ACGT'], ARRAY['AGGT'], 'dna_match')"
            ),
            "stride_cdist_matrix_above_threshold": (
                "stride_cdist_matrix_above_threshold("
                "ARRAY['ACGT'], ARRAY['AGGT'], 'dna_match', true, 1)"
            ),
            "stride_cdist_matrix_top_k": (
                "stride_cdist_matrix_top_k(ARRAY['ACGT'], ARRAY['AGGT'], 'dna_match', true, 1)"
            ),
        }
    )
    return expressions


def test_every_public_endpoint_executes(postgres_server: PostgresServer) -> None:
    expected = set(
        re.findall(
            r"`(stride_[a-z0-9_]+)",
            (ROOT / "bindings" / "duckdb" / "FUNCTIONS.md").read_text(),
        )
    )
    expressions = _public_smoke_expressions()
    assert set(expressions) == expected
    statements = [
        f"SELECT '{name}'::text, ({expression}) IS NOT NULL"
        for name, expression in sorted(expressions.items())
    ]
    output = postgres_server.command(";\n".join(statements) + ";")
    rows = [line.split("|", 1) for line in output.splitlines()]
    assert len(rows) == len(expected)
    assert all(result == "t" for _, result in rows)


def test_matrix_builders_and_beider_morse(postgres_server: PostgresServer) -> None:
    npy = _npy_int8_2x2(bytes([5, 0xFC, 0xFC, 5]))
    actual = result_json(
        postgres_server,
        f"""
        jsonb_build_object(
          'identity', stride_matrix_info(stride_identity_matrix(
              'ACGT', 5, -4, 'N', 'DNA', -5, NULL, NULL)),
          'ascii_stride', stride_matrix_info(stride_ascii_matrix())->>'stride',
          'ncbi', stride_matrix_info(stride_matrix_from_ncbi_text(
              E'A X\\nA 2 -1\\nX -1 -1\\n'))->>'alphabet',
          'counts', stride_matrix_info(stride_keyboard_from_confusion_counts(
              ARRAY[[0.0,4.0],[2.0,0.0]]::double precision[],
              'A*', 'KEY', 2.0, 4, NULL, '*', -1, NULL, NULL)),
          'npy_stride', stride_matrix_info(stride_keyboard_from_npy(
              decode('{npy.hex()}', 'hex'), 'KEY_NPY', 'A*', '*'))->>'stride',
          'bmpm', stride_beider_morse('Schneider')
        )
        """,
    )
    assert actual["identity"]["alphabet"] == "ACGTN"
    assert actual["identity"]["gap_open"] is None
    assert actual["ascii_stride"] == "128"
    assert actual["ncbi"] == "AX"
    assert actual["counts"]["name"] == "KEY"
    assert actual["npy_stride"] == "2"
    assert actual["bmpm"]


SERVER_ENCODINGS = (
    "SQL_ASCII",
    "UTF8",
    "EUC_JP",
    "EUC_CN",
    "EUC_KR",
    "EUC_TW",
    "EUC_JIS_2004",
    "LATIN1",
    "LATIN2",
    "LATIN3",
    "LATIN4",
    "LATIN5",
    "LATIN6",
    "LATIN7",
    "LATIN8",
    "LATIN9",
    "LATIN10",
    "WIN1256",
    "WIN1258",
    "WIN866",
    "WIN874",
    "KOI8R",
    "KOI8U",
    "WIN1251",
    "WIN1252",
    "ISO_8859_5",
    "ISO_8859_6",
    "ISO_8859_7",
    "ISO_8859_8",
    "WIN1250",
    "WIN1253",
    "WIN1254",
    "WIN1255",
    "WIN1257",
    "MULE_INTERNAL",
)


def test_every_server_encoding_loads_without_transcoding(
    postgres_server: PostgresServer,
) -> None:
    if postgres_server.major != 18:
        pytest.skip("the full server-encoding sweep runs once, on PostgreSQL 18")
    for index, encoding in enumerate(SERVER_ENCODINGS):
        database = f"stride_enc_{index}"
        postgres_server.create_database(database, encoding)
        info = result_json(postgres_server, "_stride_encoding_info()", database=database)
        assert info["name"] == encoding
        assert info["transcodes"] is False
        assert (
            postgres_server.command(
                "SELECT stride_levenshtein('plain', 'plane');", database=database
            )
            == "2"
        )


def test_native_width_and_cardinality_preparation(
    postgres_server: PostgresServer,
) -> None:
    if postgres_server.major != 18:
        pytest.skip("the native-encoding preparation probes run once, on PostgreSQL 18")

    # Single-byte high-bit characters stay borrowed u8 values.
    postgres_server.create_database("stride_latin1", "LATIN1")
    latin = result_json(
        postgres_server,
        """
        _stride_prepare_info(
          convert_from(decode('636166e9', 'hex'), 'LATIN1'),
          convert_from(decode('63616665', 'hex'), 'LATIN1'))
        """,
        database="stride_latin1",
    )
    assert latin == {"width": "u8", "borrowed": True, "packed": False, "ascii": False}

    # A short EUC-JP pair uses direct 16-bit native keys; a long pair amortizes
    # the Swiss table and packs its tiny alphabet down to u8.
    postgres_server.create_database("stride_eucjp", "EUC_JP")
    short_euc = result_json(
        postgres_server,
        """
        _stride_prepare_info(
          convert_from(decode('c6fccbdc', 'hex'), 'EUC_JP'),
          convert_from(decode('c6fc', 'hex'), 'EUC_JP'))
        """,
        database="stride_eucjp",
    )
    long_euc = result_json(
        postgres_server,
        """
        _stride_prepare_info(
          repeat(convert_from(decode('c6fc', 'hex'), 'EUC_JP'), 40),
          repeat(convert_from(decode('cbdc', 'hex'), 'EUC_JP'), 40))
        """,
        database="stride_eucjp",
    )
    assert short_euc["width"] == "u16" and short_euc["packed"] is False
    assert long_euc["width"] == "u8" and long_euc["packed"] is True

    # UTF-8 is likewise read in place: short supplementary characters use
    # u32, while a long low-cardinality pair packs to u8.
    short_utf8 = result_json(postgres_server, "_stride_prepare_info('😀x', '😀y')")
    long_utf8 = result_json(
        postgres_server,
        "_stride_prepare_info(repeat('😀', 40), repeat('界', 40))",
    )
    assert short_utf8["width"] == "u32" and short_utf8["packed"] is False
    assert long_utf8["width"] == "u8" and long_utf8["packed"] is True
