from __future__ import annotations

from typing import Protocol

import pytest

pytestmark = pytest.mark.memgraph


class MemgraphServer(Protocol):
    def query(self, cypher: str) -> list[list[object]]: ...

    def query_fails(self, cypher: str) -> str: ...


def one(server: MemgraphServer, expression: str):
    rows = server.query(f"RETURN {expression} AS value")
    assert len(rows) == 1 and len(rows[0]) == 1
    return rows[0][0]


def test_pairwise_scalar_surface(memgraph_server: MemgraphServer) -> None:
    assert one(memgraph_server, 'stride_align.levenshtein("kitten", "sitting")') == 3
    assert one(memgraph_server, 'stride_align.levenshtein("😀x", "😀y")') == 1
    assert one(memgraph_server, 'stride_align.osa("CA", "AC")') == 1
    assert one(memgraph_server, 'stride_align.true_damerau_levenshtein("CA", "ABC")') == 2
    assert one(memgraph_server, 'stride_align.indel("kitten", "sitting")') == 5
    assert one(memgraph_server, 'stride_align.hamming("abc", "axc")') == 1
    assert one(memgraph_server, 'stride_align.jaro("MARTHA", "MARHTA")') == pytest.approx(
        0.9444444444
    )
    assert one(
        memgraph_server,
        'stride_align.jaro_winkler("MARTHA", "MARHTA")',
    ) == pytest.approx(0.9611111111)
    assert one(
        memgraph_server,
        'stride_align.levenshtein_score("kitten", "sitting", 2)',
    ) == 3
    assert one(
        memgraph_server,
        'stride_align.levenshtein_normalized_score("kitten", "sitting", 0.9)',
    ) == 0.0


def test_alignment_scalar_and_path_surface(memgraph_server: MemgraphServer) -> None:
    assert one(
        memgraph_server,
        'stride_align.smith_waterman_score("ACCGT", "ACG")',
    ) == 5
    assert one(
        memgraph_server,
        'stride_align.needleman_wunsch_score("ACCGT", "ACG")',
    ) == 4
    assert one(
        memgraph_server,
        'stride_align.score("kitten", "sitting", "levenshtein")',
    ) == 3.0
    rows = memgraph_server.query(
        'WITH stride_align.smith_waterman_path("ACCGT", "ACG") AS path '
        "RETURN path.cigar, path.aligned_query, path.aligned_target"
    )
    assert rows == [["1=1D2=", "ACCG", "A-CG"]]
    assert one(
        memgraph_server,
        'stride_align.needleman_wunsch_cigar("ACCGT", "ACG")',
    ) == "1=1D2=1D"


def test_text_phonetic_dtw_and_matrix_scalars(memgraph_server: MemgraphServer) -> None:
    assert one(
        memgraph_server,
        'stride_align.lcs_substring("xab😀c", "zab😀q")',
    ) == "ab😀"
    assert one(memgraph_server, 'stride_align.lcs_length("ABCBDAB", "BDCABA")') == 4
    assert one(memgraph_server, 'stride_align.jaccard("night", "nacht")') == pytest.approx(
        1 / 7
    )
    assert one(
        memgraph_server,
        'stride_align.token_set_ratio("new york mets", "new york")',
    ) == 1.0
    assert one(memgraph_server, 'stride_align.soundex("Robert")') == "R163"
    assert one(memgraph_server, 'stride_align.metaphone("Smith")') == "SM0"
    rows = memgraph_server.query(
        'WITH stride_align.double_metaphone("Smith") AS code '
        "RETURN code.primary, code.alternate"
    )
    assert rows == [["SM0", "XMT"]]
    assert one(memgraph_server, "stride_align.dtw([1, 2, 3], [1.0, 2.0, 4.0])") == 1.0
    assert one(memgraph_server, 'stride_align.beider_morse("Schneider")')
    assert one(
        memgraph_server,
        'stride_align.smith_waterman_matrix_score('
        '"HEAGAWGHEE", "PAWHEAE", "blosum62", -4)',
    ) == 25
    rows = memgraph_server.query(
        'WITH stride_align.matrix_info("blosum62") AS matrix '
        "RETURN matrix.name, matrix.stride, matrix.wildcard_index"
    )
    assert rows == [["BLOSUM62", 24, 22]]


def test_scalar_functions_compose_over_cypher_rows(memgraph_server: MemgraphServer) -> None:
    rows = memgraph_server.query(
        "UNWIND ["
        '{query: "kitten", target: "sitting"}, '
        '{query: "book", target: "back"}, '
        '{query: "😀x", target: "😀y"}'
        "] AS pair "
        "RETURN stride_align.levenshtein(pair.query, pair.target) AS score"
    )
    assert rows == [[3], [2], [1]]


def test_cdist_streams_row_major_across_internal_chunks(
    memgraph_server: MemgraphServer,
) -> None:
    queries = [f"q{index}" for index in range(17)]
    targets = [f"q{index}" for index in range(17)]
    query_literal = "[" + ",".join(f'"{value}"' for value in queries) + "]"
    target_literal = "[" + ",".join(f'"{value}"' for value in targets) + "]"
    rows = memgraph_server.query(
        f"CALL stride_align.cdist({query_literal}, {target_literal}, \"levenshtein\") "
        "YIELD query_index, target_index, score "
        "RETURN query_index, target_index, score"
    )
    assert len(rows) == 289  # crosses the default 256-row callback chunk
    assert [(row[0], row[1]) for row in rows] == [
        (query_index, target_index)
        for query_index in range(17)
        for target_index in range(17)
    ]
    assert rows[0] == [0, 0, 0.0]
    assert rows[-1] == [16, 16, 0.0]


def test_cdist_limit_cleanup_does_not_leak_stream_state(
    memgraph_server: MemgraphServer,
) -> None:
    limited = memgraph_server.query(
        'CALL stride_align.cdist(["a", "b"], ["a", "b"], "levenshtein") '
        "YIELD query_index, target_index, score "
        "RETURN query_index, target_index, score LIMIT 1"
    )
    assert limited == [[0, 0, 0.0]]
    second = memgraph_server.query(
        'CALL stride_align.cdist(["x"], ["x", "y"], "levenshtein") '
        "YIELD query_index, target_index, score "
        "RETURN query_index, target_index, score"
    )
    assert second == [[0, 0, 0.0], [0, 1, 1.0]]


def test_vector_and_ranking_functions_are_not_registered(
    memgraph_server: MemgraphServer,
) -> None:
    error = memgraph_server.query_fails(
        'RETURN stride_align.levenshtein_scores("a", ["a", "b"])'
    )
    assert "levenshtein_scores" in error
    error = memgraph_server.query_fails(
        'RETURN stride_align.levenshtein_top_k("a", ["a", "b"], 1)'
    )
    assert "levenshtein_top_k" in error
