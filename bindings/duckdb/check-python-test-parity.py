#!/usr/bin/env python3
"""Verify that every non-compatibility Python suite has a DuckDB port."""

from __future__ import annotations

import ast
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "bindings/duckdb/python-test-parity.tsv"
EXCLUDED_COMPATIBILITY_SUITES = {
    Path("tests/test_jellyfish_shim.py"),
    Path("tests/test_parasail_shim.py"),
    Path("tests/test_rapidfuzz_shim.py"),
    Path("tests/test_thefuzz_shim.py"),
}


def test_function_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in ast.walk(tree)
    )


def fail(message: str) -> None:
    raise SystemExit(f"DuckDB/Python test parity check failed: {message}")


def main() -> None:
    discovered = {
        path.relative_to(ROOT)
        for path in (ROOT / "tests").glob("test_*.py")
    }
    missing_exclusions = EXCLUDED_COMPATIBILITY_SUITES - discovered
    if missing_exclusions:
        fail(
            "declared compatibility suites no longer exist: "
            + ", ".join(map(str, sorted(missing_exclusions)))
        )

    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    expected_fields = {
        "python_test", "test_functions", "duckdb_tests", "adaptation"
    }
    if not rows or set(rows[0]) != expected_fields:
        fail(f"{MANIFEST.relative_to(ROOT)} has an invalid header")

    manifest_paths = [Path(row["python_test"]) for row in rows]
    duplicates = sorted({
        path for path in manifest_paths if manifest_paths.count(path) > 1
    })
    if duplicates:
        fail("duplicate manifest rows: " + ", ".join(map(str, duplicates)))

    in_scope = discovered - EXCLUDED_COMPATIBILITY_SUITES
    recorded = set(manifest_paths)
    if missing := sorted(in_scope - recorded):
        fail("Python suites without DuckDB ports: " + ", ".join(map(str, missing)))
    if extra := sorted(recorded - in_scope):
        fail("stale or excluded manifest entries: " + ", ".join(map(str, extra)))

    total_functions = 0
    for row in rows:
        python_test = ROOT / row["python_test"]
        duckdb_tests = [
            ROOT / value for value in row["duckdb_tests"].split(";") if value
        ]
        if not duckdb_tests:
            fail(f"no DuckDB test recorded for {python_test.relative_to(ROOT)}")
        for duckdb_test in duckdb_tests:
            if not duckdb_test.is_file():
                fail(f"missing DuckDB test {duckdb_test.relative_to(ROOT)}")
        if not row["adaptation"].strip():
            fail(f"missing adaptation note for {python_test.relative_to(ROOT)}")
        actual_count = test_function_count(python_test)
        try:
            recorded_count = int(row["test_functions"])
        except ValueError:
            fail(f"invalid test count for {python_test.relative_to(ROOT)}")
        if actual_count != recorded_count:
            fail(
                f"{python_test.relative_to(ROOT)} now has {actual_count} test "
                f"functions; manifest records {recorded_count}"
            )
        total_functions += actual_count

    excluded_functions = sum(
        test_function_count(ROOT / path)
        for path in EXCLUDED_COMPATIBILITY_SUITES
    )
    print(
        f"DuckDB test parity: {len(rows)} Python modules / "
        f"{total_functions} test functions; excluded exactly "
        f"{len(EXCLUDED_COMPATIBILITY_SUITES)} compatibility suites "
        f"({excluded_functions} test functions)."
    )


if __name__ == "__main__":
    main()
