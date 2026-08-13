#!/usr/bin/env python3
"""Bible lookup and spell checking through DuckDB and stride-align batches.

Python owns the terminal and file discovery. Candidate strings stay in DuckDB
tables, and SQL passes complete candidate vectors to stride-align's one-to-many
functions. No Python loop or row-at-a-time scorer processes the candidates.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BIBLE_URL = "https://openbible.com/textfiles/kjv.txt"
DEFAULT_BIBLE = Path(__file__).with_name("kjv.txt")
DICTIONARY_PATHS = (
    Path("/usr/share/dict/words"),
    Path("/usr/dict/words"),
    Path("/var/lib/dict/words"),
    Path("/etc/dictionaries-common/words"),
)

# DuckDB collects the table columns into consistently ordered LIST vectors.
# ``stride_extract_best`` owns the optimized batch scoring and top-one reduction.
# Its zero-based index selects the corresponding original reference and verse
# from the two parallel DuckDB lists.
BIBLE_SEARCH_SQL = """
WITH corpus AS (
    SELECT
        list(reference ORDER BY reference, verse) AS reference_values,
        list(verse ORDER BY reference, verse) AS verse_values,
        list(lower(verse) ORDER BY reference, verse) AS normalized_verses
    FROM demo_bible
), matched AS (
    SELECT
        reference_values,
        verse_values,
        stride_extract_best(
            lower(?),
            normalized_verses,
            'needleman_wunsch_normalized'
        ) AS best_match
    FROM corpus
)
SELECT
    reference_values[cast(best_match.index AS BIGINT) + 1] AS reference,
    verse_values[cast(best_match.index AS BIGINT) + 1] AS verse,
    best_match.score AS score
FROM matched
"""

# One SQL statement collects the dictionary into a candidate LIST, expands the
# input tokens, and applies stride-align's batch top-one operation to each
# token. Python never loops over tokens or words.
SPELLCHECK_SQL = """
WITH dictionary AS (
    SELECT list(word ORDER BY word) AS words
    FROM demo_dictionary
), input_tokens AS (
    SELECT position, token
    FROM unnest(string_split(lower(?), ' '))
         WITH ORDINALITY AS tokens(token, position)
    WHERE token <> ''
),
matched AS (
    SELECT
        position,
        stride_extract_best(
            token,
            words,
            'needleman_wunsch_normalized'
        ) AS best_match
    FROM input_tokens
    CROSS JOIN dictionary
)
SELECT coalesce(string_agg(best_match.target, ' ' ORDER BY position), '')
FROM matched
"""


@dataclass(frozen=True)
class BibleMatch:
    reference: str
    verse: str
    score: float


def _extension_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DuckDB extension does not exist: {path}")
    return path


def connect(extension: str | os.PathLike[str]) -> Any:
    """Open DuckDB and sideload the requested stride-align package."""
    try:
        import duckdb
    except ImportError as error:
        raise RuntimeError("install the matching client with: pip install duckdb==1.5.5") from error

    extension_path = _extension_path(extension)
    connection = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    escaped = str(extension_path).replace("'", "''")
    connection.execute(f"LOAD '{escaped}'")
    return connection


def ensure_bible(path: Path) -> Path:
    """Download the public-domain KJV corpus only when it is not cached."""
    path = path.expanduser().resolve()
    if path.is_file():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        BIBLE_URL,
        headers={"User-Agent": "stride-align DuckDB vectorization demo"},
    )
    with urllib.request.urlopen(request) as response:  # noqa: S310 - fixed HTTPS URL
        text = response.read().decode("utf-8-sig")
    path.write_text(text, encoding="utf-8")
    return path


def load_bible(connection: Any, path: Path) -> int:
    """Load the tab-separated corpus into a two-column DuckDB table."""
    connection.execute(
        """
        CREATE OR REPLACE TABLE demo_bible AS
        SELECT trim(reference) AS reference, trim(verse) AS verse
        FROM read_csv(
            ?,
            delim = '\t',
            header = false,
            skip = 2,
            columns = {'reference': 'VARCHAR', 'verse': 'VARCHAR'},
            quote = '',
            escape = ''
        )
        WHERE reference IS NOT NULL
          AND verse IS NOT NULL
          AND trim(verse) <> ''
        """,
        [str(path.expanduser().resolve())],
    )
    return int(connection.execute("SELECT count(*) FROM demo_bible").fetchone()[0])


def nearest_bible_match(connection: Any, query: str) -> BibleMatch:
    """Return the best verse through stride-align's one-to-many best function."""
    row = connection.execute(BIBLE_SEARCH_SQL, [query]).fetchone()
    if row is None:
        raise ValueError("the Bible table is empty")
    return BibleMatch(reference=str(row[0]), verse=str(row[1]), score=float(row[2]))


def find_dictionary(explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"dictionary does not exist: {path}")
        return path
    for candidate in DICTIONARY_PATHS:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("no system word list found; pass --dictionary /path/to/words")


def load_dictionary(connection: Any, path: Path) -> int:
    """Load a one-word-per-line dictionary into DuckDB."""
    connection.execute(
        """
        CREATE OR REPLACE TABLE demo_dictionary AS
        SELECT DISTINCT lower(trim(word)) AS word
        FROM read_csv(
            ?,
            delim = '\x1f',
            header = false,
            columns = {'word': 'VARCHAR'},
            quote = '',
            escape = ''
        )
        WHERE word IS NOT NULL AND trim(word) <> ''
        """,
        [str(path.expanduser().resolve())],
    )
    count = int(connection.execute("SELECT count(*) FROM demo_dictionary").fetchone()[0])
    if count == 0:
        raise ValueError(f"dictionary is empty: {path}")
    return count


def spellcheck_line(connection: Any, line: str) -> str:
    """Correct every token with one SQL query and one batch call per token."""
    return str(connection.execute(SPELLCHECK_SQL, [line]).fetchone()[0])


def run_bible(connection: Any, corpus: Path, query: str | None) -> None:
    corpus = ensure_bible(corpus)
    count = load_bible(connection, corpus)
    print(
        f"Loaded {count:,} verses; stride-align scores one DuckDB LIST batch.",
        file=sys.stderr,
    )

    def search(text: str) -> None:
        started = time.perf_counter()
        match = nearest_bible_match(connection, text)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        print(f"Score: {match.score}")
        print(f"{match.reference}\t{match.verse}")
        print(f"Search time: {elapsed_ms:.2f}ms")

    if query is not None:
        search(query)
        return

    while text := input("Enter a snippet to match. Press enter to end.\n"):
        search(text)
        print()


def run_spellcheck(connection: Any, dictionary: Path | None, text: str | None) -> None:
    dictionary_path = find_dictionary(dictionary)
    count = load_dictionary(connection, dictionary_path)
    print(
        f"Loaded {count:,} words; each input line is one vectorized DuckDB query.",
        file=sys.stderr,
    )

    if text is not None:
        print(spellcheck_line(connection, text))
        return

    for line in sys.stdin:
        print(spellcheck_line(connection, line.strip()), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--extension",
        default=os.environ.get("STRIDE_ALIGN_DUCKDB_EXTENSION"),
        help=("path to stride_align.0.6.0.duckdb_extension (or set STRIDE_ALIGN_DUCKDB_EXTENSION)"),
    )
    commands = result.add_subparsers(dest="command", required=True)

    bible = commands.add_parser("bible", help="find the nearest KJV verse")
    bible.add_argument("query", nargs="?", help="one query; omit for an interactive prompt")
    bible.add_argument("--corpus", type=Path, default=DEFAULT_BIBLE)

    spell = commands.add_parser("spell", help="correct input with a system word list")
    spell.add_argument("text", nargs="?", help="one line; omit to read lines from stdin")
    spell.add_argument("--dictionary", type=Path)
    return result


def main() -> int:
    arguments = parser().parse_args()
    if not arguments.extension:
        print(
            "pass --extension /path/to/stride_align.0.6.0.duckdb_extension "
            "or set STRIDE_ALIGN_DUCKDB_EXTENSION",
            file=sys.stderr,
        )
        return 2

    connection = connect(arguments.extension)
    try:
        if arguments.command == "bible":
            run_bible(connection, arguments.corpus, arguments.query)
        else:
            run_spellcheck(connection, arguments.dictionary, arguments.text)
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
