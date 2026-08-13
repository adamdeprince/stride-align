from __future__ import annotations

from pathlib import Path
from typing import Any

from demo.duckdb_vectorized_lookup import (
    load_bible,
    load_dictionary,
    nearest_bible_match,
    spellcheck_line,
)


def test_bible_lookup_crosses_duckdb_data_chunks(
    duckdb_extension: Any,
    tmp_path: Path,
) -> None:
    # DuckDB's standard vector is smaller than this corpus. Put the only exact
    # match near the end so a scalar implementation that mishandles later
    # DataChunks cannot accidentally pass on the first chunk.
    lines = ["KJV", "test corpus"]
    lines.extend(
        f"Example 1:{index}\tfiller verse number {index}"
        for index in range(5_000)
    )
    lines.append("Example 2:1\tthe exact verse we need")
    corpus = tmp_path / "verses.txt"
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert load_bible(duckdb_extension.connection, corpus) == 5_001
    match = nearest_bible_match(
        duckdb_extension.connection,
        "the exact verse we need",
    )
    assert match.reference == "Example 2:1"
    assert match.verse == "the exact verse we need"
    assert match.score == 1.0


def test_spellchecker_scores_all_tokens_and_dictionary_rows_in_sql(
    duckdb_extension: Any,
    tmp_path: Path,
) -> None:
    words = ["this", "is", "a", "demonstration", "of", "spell", "checker"]
    words.extend(f"unrelatedword{index}" for index in range(5_000))
    dictionary = tmp_path / "words"
    dictionary.write_text("\n".join(words) + "\n", encoding="utf-8")

    assert load_dictionary(duckdb_extension.connection, dictionary) == len(words)
    assert spellcheck_line(
        duckdb_extension.connection,
        "this is a demonstrtion of a spell checker",
    ) == "this is a demonstration of a spell checker"


def test_spellchecker_preserves_token_order_and_handles_empty_input(
    duckdb_extension: Any,
    tmp_path: Path,
) -> None:
    dictionary = tmp_path / "words"
    dictionary.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    load_dictionary(duckdb_extension.connection, dictionary)

    assert spellcheck_line(duckdb_extension.connection, "gama alfa beta") == "gamma alpha beta"
    assert spellcheck_line(duckdb_extension.connection, "") == ""
