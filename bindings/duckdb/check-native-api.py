#!/usr/bin/env python3
"""Verify that the complete native Python/R API has a DuckDB mapping."""

from __future__ import annotations

import ast
import csv
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "bindings/duckdb/native-api.tsv"
EXCLUDED_COMPATIBILITY_MODULES = {"rapidfuzz", "parasail", "jellyfish", "thefuzz"}


def fail(message: str) -> None:
    raise SystemExit(f"DuckDB native API check failed: {message}")


def python_exports() -> set[str]:
    tree = ast.parse((ROOT / "src/stride_align/__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return {
                        value.value
                        for value in node.value.elts
                        if isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                    }
    fail("src/stride_align/__init__.py has no static __all__")
    return set()


def matrix_exports() -> set[str]:
    """Read matrices.__all__ without importing numpy or a native extension."""
    source = (ROOT / "src/stride_align/matrices/__init__.py").read_text()
    tree = ast.parse(source)
    pam_levels: tuple[int, ...] = ()
    all_node: ast.List | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if "_PAM_LEVELS" in names:
            # The source deliberately spells this as tuple(range(start, stop, step)).
            call = node.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "tuple"
                and len(call.args) == 1
                and isinstance(call.args[0], ast.Call)
                and isinstance(call.args[0].func, ast.Name)
                and call.args[0].func.id == "range"
            ):
                values = [ast.literal_eval(argument) for argument in call.args[0].args]
                pam_levels = tuple(range(*values))
        if "__all__" in names and isinstance(node.value, ast.List):
            all_node = node.value
    if all_node is None or not pam_levels:
        fail("could not statically read matrices.__all__ and _PAM_LEVELS")

    output: set[str] = set()
    for element in all_node.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            output.add(element.value)
        elif isinstance(element, ast.Starred):
            # The sole dynamic portion is *(f"pam{_level}" for _level in
            # _PAM_LEVELS). Its values were resolved above.
            output.update(f"pam{level}" for level in pam_levels)
        else:
            fail("matrices.__all__ contains an unsupported dynamic expression")
    return output


def r_exports() -> set[str]:
    """Read every public R symbol from the package namespace."""
    namespace = (ROOT / "bindings/r/stridealign/NAMESPACE").read_text()
    return {
        match.group(1)
        for match in re.finditer(r"^export\(([^)]+)\)$", namespace, re.MULTILINE)
    }


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows or set(rows[0]) != {"source_api", "duckdb_api", "adaptation"}:
        fail("native-api.tsv has an invalid header")

    mapped: set[str] = set()
    sql_names: set[str] = set()
    for row in rows:
        mapped.update(filter(None, row["source_api"].split(";")))
        sql_names.update(filter(None, row["duckdb_api"].split(";")))
        if not row["adaptation"].strip():
            fail(f"missing adaptation note for {row['source_api']}")

    expected = python_exports() | matrix_exports() | r_exports()
    if missing := sorted(expected - mapped):
        fail("unmapped native exports: " + ", ".join(missing))
    if extra := sorted(mapped - expected):
        fail("stale mapped exports: " + ", ".join(extra))

    extension = os.environ.get("STRIDE_ALIGN_DUCKDB_EXTENSION")
    if extension:
        import duckdb

        connection = duckdb.connect(config={"allow_unsigned_extensions": "true"})
        escaped = str(Path(extension).resolve()).replace("'", "''")
        connection.execute(f"LOAD '{escaped}'")
        registered = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT function_name FROM duckdb_functions() "
                "WHERE function_name LIKE 'stride_%'"
            ).fetchall()
        }
        if missing := sorted(sql_names - registered):
            fail("manifest references unregistered SQL functions: " + ", ".join(missing))
        if any(
            marker in name
            for marker in EXCLUDED_COMPATIBILITY_MODULES
            for name in registered
        ):
            fail("a compatibility facade was registered")

    print(
        f"DuckDB native API: {len(expected)} Python/R exports mapped to "
        f"{len(sql_names)} SQL functions; compatibility facades excluded."
    )


if __name__ == "__main__":
    main()
