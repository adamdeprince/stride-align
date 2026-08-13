from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOMEPAGE = ROOT / "html" / "index.html"
TARGETS = {"python", "r", "duckdb"}
WORKFLOW_STEPS = {"install", "first-score", "bible-search", "spell-checker"}


class HomepageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.current_section: str | None = None
        self.language_selects = 0
        self.selects_by_section: dict[str, int] = {}
        self.option_values: set[str] = set()
        self.panels_by_section: dict[str, set[str]] = {}
        self.copy_source_ids: set[str] = set()
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        element_id = values.get("id")
        if element_id:
            assert element_id not in self.ids, f"duplicate homepage id: {element_id}"
            self.ids.add(element_id)

        if tag == "section" and element_id in WORKFLOW_STEPS:
            self.current_section = element_id
        if tag == "select" and "data-language-select" in values:
            self.language_selects += 1
            if self.current_section:
                self.selects_by_section[self.current_section] = (
                    self.selects_by_section.get(self.current_section, 0) + 1
                )
        if tag == "option" and values.get("value"):
            self.option_values.add(str(values["value"]))
        if values.get("data-language-panel") and self.current_section:
            self.panels_by_section.setdefault(self.current_section, set()).add(
                str(values["data-language-panel"])
            )
        if values.get("data-copy-source"):
            self.copy_source_ids.add(str(values["data-copy-source"]))

    def handle_endtag(self, tag: str) -> None:
        if tag == "section" and self.current_section:
            self.current_section = None


def test_homepage_walkthrough_is_complete_for_every_interface() -> None:
    parser = HomepageParser()
    parser.feed(HOMEPAGE.read_text(encoding="utf-8"))

    assert parser.language_selects == len(WORKFLOW_STEPS)
    assert parser.selects_by_section == {step: 1 for step in WORKFLOW_STEPS}
    assert parser.option_values == TARGETS
    assert parser.panels_by_section == {step: TARGETS for step in WORKFLOW_STEPS}
    assert parser.copy_source_ids <= parser.ids


def test_homepage_benchmark_numbers_name_their_artifacts() -> None:
    homepage = HOMEPAGE.read_text(encoding="utf-8")

    for measurement in ("1.682×", "1.883×", "1.548×", "3.749×"):
        assert measurement in homepage
    assert "benchmarks/intel-avx512-parasail-2026-07-18.csv" in homepage
    assert "benchmarks/intel-levenshtein-2026-07-17.csv" in homepage


def test_homepage_corpus_example_is_complete_and_contextualized() -> None:
    homepage = HOMEPAGE.read_text(encoding="utf-8")

    assert "Fuzzy-match a paraphrase against a corpus." in homepage
    assert "I apologize to anyone who finds the use of Christian scripture" in homepage
    assert "I am not using religious texts to make a religious statement" in homepage
    assert "Observed top-ranked record" in homepage
    assert "For this shall the earth mourn" not in homepage

    duckdb_example = homepage.split('id="bible-duckdb-code"', 1)[1].split("</code>", 1)[0]
    assert "if not corpus.exists()" in duckdb_example
    assert 'urlretrieve("https://openbible.com/textfiles/kjv.txt", corpus)' in duckdb_example
    assert "CREATE OR REPLACE TABLE demo_bible" in duckdb_example
    assert "list(lower(verse) ORDER BY reference, verse)" in duckdb_example
    assert "stride_extract_best" in duckdb_example
    assert "'needleman_wunsch_normalized'" in duckdb_example
    assert "cast(best_match.index AS BIGINT) + 1" in duckdb_example
    assert "stride_needleman_wunsch_normalized_score(" not in duckdb_example


def test_homepage_duckdb_spellchecker_is_a_complete_batch_script() -> None:
    homepage = HOMEPAGE.read_text(encoding="utf-8")
    duckdb_example = (
        homepage.split('id="spell-duckdb-code"', 1)[1]
        .split(">", 1)[1]
        .split("</code>", 1)[0]
    )

    assert "import duckdb" in duckdb_example
    assert "sys.stdin.readline()" in duckdb_example
    assert "STRIDE_ALIGN_DUCKDB_EXTENSION" in duckdb_example
    assert "CREATE OR REPLACE TABLE demo_dictionary" in duckdb_example
    assert "list(word ORDER BY word)" in duckdb_example
    assert "stride_extract_best" in duckdb_example
    assert "stride_needleman_wunsch_normalized_score(" not in duckdb_example
    compile(unescape(duckdb_example), "spell_duckdb.py", "exec")


def test_homepage_has_one_cross_language_api_destination() -> None:
    homepage = HOMEPAGE.read_text(encoding="utf-8")
    reference = homepage.split('class="research-next"', 1)[1].split("</section>", 1)[0]

    assert reference.count("<a ") == 1
    assert 'href="docs/api/index.html"' in reference
    assert "Every algorithm in Python, R, and DuckDB" in reference
