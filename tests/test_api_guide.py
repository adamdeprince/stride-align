from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from tools.api_catalog import (
    EXAMPLES,
    MEMGRAPH_EXAMPLE_BODIES,
    MEMGRAPH_VECTORIZATION_BODIES,
    POSTGRES_SQL_OVERRIDES,
    POSTGRES_VECTORIZATION_SQL,
    VECTORIZATION_EXAMPLES,
    ZH_EXAMPLE_TEXT,
    ZH_VECTORIZATION_TEXT,
    _duckdb_python,
    _memgraph_python,
    _memgraph_scalar_body,
    _postgres_python,
    render_api_catalog,
    render_vectorization_guide,
)

TARGETS = {"python", "r", "duckdb", "postgres", "memgraph"}
ROOT = Path(__file__).resolve().parents[1]
CHINESE_API_SOURCE = ROOT / "docs" / "api" / "README.zh-CN.md"
CHINESE_API = ROOT / "html" / "docs" / "api" / "index.zh-CN.html"


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
    examples = EXAMPLES + VECTORIZATION_EXAMPLES
    slugs = {example.slug for example in examples}

    assert len(slugs) == len(examples)
    for locale in ("en", "zh-CN"):
        parser = ApiCatalogParser()
        parser.feed(render_api_catalog(locale))
        parser.feed(render_vectorization_guide(locale))

        assert parser.selects == {slug: 1 for slug in slugs}
        assert parser.panels == {slug: TARGETS for slug in slugs}


def test_chinese_api_catalog_covers_every_card_and_uses_chinese_terms() -> None:
    assert set(ZH_EXAMPLE_TEXT) == {example.slug for example in EXAMPLES}
    assert set(ZH_VECTORIZATION_TEXT) == {example.slug for example in VECTORIZATION_EXAMPLES}

    rendered = render_vectorization_guide("zh-CN") + render_api_catalog("zh-CN")
    source = CHINESE_API_SOURCE.read_text(encoding="utf-8")

    assert ">语言<" in rendered
    assert "向量化形状" in rendered
    assert "全局前 k 项" in rendered
    assert "每个查询各自的前 k 项" in rendered
    assert "详细语义（英文）" in rendered
    assert ">Language<" not in rendered
    assert "整个网格全局" in source
    assert "3 × Q" in source
    assert "逐步产生配对记录" in source


def test_generated_chinese_api_page_has_translation_navigation() -> None:
    page = CHINESE_API.read_text(encoding="utf-8")

    assert '<html lang="zh-CN"' in page
    assert 'hreflang="en" href="index.html"' in page
    assert 'hreflang="zh-CN" href="index.zh-CN.html" class="active"' in page
    assert page.count("data-language-select") == len(EXAMPLES + VECTORIZATION_EXAMPLES)
    assert "<!-- stride-vectorization-guide -->" not in page
    assert "<!-- stride-api-catalog -->" not in page


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


def test_database_api_catalog_examples_are_python_orchestrated_and_compile() -> None:
    for example in EXAMPLES:
        compile(_duckdb_python(example.duckdb), f"{example.slug}-duckdb.py", "exec")
        compile(
            _postgres_python(POSTGRES_SQL_OVERRIDES.get(example.slug, example.duckdb)),
            f"{example.slug}-postgres.py",
            "exec",
        )
        memgraph_body = (
            MEMGRAPH_EXAMPLE_BODIES[example.slug]
            if example.slug in MEMGRAPH_EXAMPLE_BODIES
            else _memgraph_scalar_body(example.duckdb)
        )
        compile(
            _memgraph_python(memgraph_body),
            f"{example.slug}-memgraph.py",
            "exec",
        )

    for example in VECTORIZATION_EXAMPLES:
        compile(_duckdb_python(example.duckdb), f"{example.slug}-duckdb.py", "exec")
        compile(
            _postgres_python(POSTGRES_VECTORIZATION_SQL[example.slug]),
            f"{example.slug}-postgres.py",
            "exec",
        )
        compile(
            _memgraph_python(MEMGRAPH_VECTORIZATION_BODIES[example.slug]),
            f"{example.slug}-memgraph.py",
            "exec",
        )


def test_vectorization_guide_distinguishes_global_and_per_query_top_k() -> None:
    guide = render_vectorization_guide()
    source = (ROOT / "docs/api/README.md").read_text(encoding="utf-8")

    assert "at most K total" in guide
    assert "Q groups, each with at most K matches" in guide
    assert "across the whole grid" in source
    assert "3 × Q" in source
    assert "DuckDB's vectorized chunk machinery" in source
    assert "PostgreSQL" in source
    assert "Q·T" in source
    assert "CALL stride_align.cdist" in source
    assert "continues to produce pair rows incrementally" in source


def test_detailed_cdist_shapes_match_the_python_api() -> None:
    source = (ROOT / "docs/api/cdist.md").read_text(encoding="utf-8")

    assert "iterator of `(score, query, target)`" in source
    assert "`(query, list[(score, target)])`" in source
    assert "up to `5 × Q` records overall" in source
    assert "target_index, score" not in source
