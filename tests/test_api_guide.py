from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from tools.api_catalog import (
    EXAMPLES,
    VECTORIZATION_EXAMPLES,
    render_api_catalog,
    render_vectorization_guide,
)

TARGETS = {"python", "r", "duckdb"}
ROOT = Path(__file__).resolve().parents[1]


class ApiCatalogParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.current_example: str | None = None
        self.ids: set[str] = set()
        self.selects: dict[str, int] = {}
        self.panels: dict[str, set[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        element_id = values.get("id")
        if element_id:
            assert element_id not in self.ids, f"duplicate API guide id: {element_id}"
            self.ids.add(element_id)

        classes = set((values.get("class") or "").split())
        if tag == "section" and "api-example" in classes:
            assert element_id
            self.current_example = element_id
        if tag == "select" and "data-language-select" in values:
            assert self.current_example
            self.selects[self.current_example] = self.selects.get(self.current_example, 0) + 1
        if values.get("data-language-panel"):
            assert self.current_example
            self.panels.setdefault(self.current_example, set()).add(
                str(values["data-language-panel"])
            )

    def handle_endtag(self, tag: str) -> None:
        if tag == "section":
            self.current_example = None


def test_api_catalog_has_a_synchronized_selector_for_every_example() -> None:
    parser = ApiCatalogParser()
    parser.feed(render_api_catalog())
    parser.feed(render_vectorization_guide())
    examples = EXAMPLES + VECTORIZATION_EXAMPLES
    slugs = {example.slug for example in examples}

    assert len(slugs) == len(examples)
    assert parser.selects == {slug: 1 for slug in slugs}
    assert parser.panels == {slug: TARGETS for slug in slugs}


def test_python_api_catalog_examples_execute() -> None:
    for example in EXAMPLES + VECTORIZATION_EXAMPLES:
        namespace: dict[str, object] = {}
        exec("import stride_align as sa\n" + example.python, namespace)  # noqa: S102


def test_r_api_catalog_functions_are_exported() -> None:
    namespace = (ROOT / "bindings/r/stridealign/NAMESPACE").read_text(encoding="utf-8")
    exported = set(re.findall(r"^export\(([^)]+)\)$", namespace, flags=re.MULTILINE))
    base_functions = {
        "all",
        "c",
        "dim",
        "identical",
        "integer",
        "length",
        "library",
        "nrow",
        "stopifnot",
        "vapply",
    }
    referenced = {
        name
        for example in EXAMPLES + VECTORIZATION_EXAMPLES
        for name in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\s*\(", example.r)
        if name not in base_functions
    }

    assert referenced <= exported


def test_vectorization_guide_distinguishes_global_and_per_query_top_k() -> None:
    guide = render_vectorization_guide()
    source = (ROOT / "docs/api/README.md").read_text(encoding="utf-8")

    assert "at most K total" in guide
    assert "Q groups, each with at most K matches" in guide
    assert "across the whole grid" in source
    assert "3 × Q" in source
    assert "DuckDB's vectorized chunk machinery" in source


def test_detailed_cdist_shapes_match_the_python_api() -> None:
    source = (ROOT / "docs/api/cdist.md").read_text(encoding="utf-8")

    assert "iterator of `(score, query, target)`" in source
    assert "`(query, list[(score, target)])`" in source
    assert "up to `5 × Q` records overall" in source
    assert "target_index, score" not in source
