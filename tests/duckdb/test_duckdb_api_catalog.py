from __future__ import annotations

from typing import Any

import pytest

from tools.api_catalog import (
    EXAMPLES,
    VECTORIZATION_EXAMPLES,
    ApiExample,
    VectorizationExample,
)


@pytest.mark.parametrize(
    "example",
    EXAMPLES + VECTORIZATION_EXAMPLES,
    ids=lambda example: example.slug,
)
def test_duckdb_api_catalog_example_executes(
    duckdb_extension: Any,
    example: ApiExample | VectorizationExample,
) -> None:
    statements = [statement.strip() for statement in example.duckdb.split(";")]
    for statement in filter(None, statements):
        duckdb_extension.connection.execute(statement)
