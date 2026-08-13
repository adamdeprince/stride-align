from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOMEPAGE = ROOT / "html" / "index.html"
CHINESE_HOMEPAGE_SOURCE = ROOT / "website" / "index.zh-CN.html"
CHINESE_HOMEPAGE = ROOT / "html" / "index.zh-CN.html"
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
    for homepage in (HOMEPAGE, CHINESE_HOMEPAGE_SOURCE):
        parser = HomepageParser()
        parser.feed(homepage.read_text(encoding="utf-8"))

        assert parser.language_selects == len(WORKFLOW_STEPS), homepage
        assert parser.selects_by_section == {step: 1 for step in WORKFLOW_STEPS}, homepage
        assert parser.option_values == TARGETS, homepage
        assert parser.panels_by_section == {step: TARGETS for step in WORKFLOW_STEPS}, homepage
        assert parser.copy_source_ids <= parser.ids, homepage


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
        homepage.split('id="spell-duckdb-code"', 1)[1].split(">", 1)[1].split("</code>", 1)[0]
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


def test_duckdb_download_is_prominent_in_both_homepages() -> None:
    english = HOMEPAGE.read_text(encoding="utf-8")
    chinese = CHINESE_HOMEPAGE_SOURCE.read_text(encoding="utf-8")

    assert 'class="duckdb-download-link"' in english
    assert "https://distribution.goblinreactor.com/stride-align/duckdb/" in english
    assert 'class="duckdb-download-link"' in chinese
    assert "https://distribution.goblinreactor.com/stride-align/duckdb/index.zh-CN.html" in chinese


def test_loongarch_python_install_is_prominent_and_bilingual() -> None:
    english = HOMEPAGE.read_text(encoding="utf-8")
    chinese = CHINESE_HOMEPAGE_SOURCE.read_text(encoding="utf-8")

    release = "https://github.com/adamdeprince/stride-align/releases/tag/v0.6.0"
    assert release in english
    assert "PyPI does not index" in english
    assert "README.html#loongarch-installation" in english
    assert release in chinese
    assert "PyPI 不索引" in chinese
    assert "README.zh-CN.html#loongarch" in chinese


def _code_block(homepage: str, element_id: str) -> str:
    return unescape(homepage.split(f'<code id="{element_id}">', 1)[1].split("</code>", 1)[0])


def test_chinese_homepage_matches_the_current_scientific_story() -> None:
    english = HOMEPAGE.read_text(encoding="utf-8")
    chinese = CHINESE_HOMEPAGE_SOURCE.read_text(encoding="utf-8")

    assert '<body class="scientific-home">' in chinese
    assert "统一码（Unicode）字符串比较与序列比对" in chinese
    assert "中日韩文字与表情符号" in chinese
    assert "前 k 项筛选" in chinese
    assert "性能结论均附基线与工作负载" in chinese
    assert "如果在这里使用基督教经文让任何人感到被排斥或冒犯" in chinese
    assert "查看 Python、R 与 DuckDB 中的每一种算法" in chinese
    assert '<script src="language-switch.js"></script>' in chinese
    assert "site-header" not in chinese
    assert "兼容层" not in chinese
    assert "与 emoji" not in chinese
    assert "top-one" not in chinese

    code_ids = (
        "trivial-python-code",
        "trivial-r-code",
        "trivial-duckdb-code",
        "bible-python-code",
        "bible-r-code",
        "bible-duckdb-code",
        "spell-python-code",
        "spell-r-code",
        "spell-duckdb-code",
    )
    for element_id in code_ids:
        assert _code_block(chinese, element_id) == _code_block(english, element_id)


def test_generated_chinese_homepage_matches_its_source() -> None:
    assert CHINESE_HOMEPAGE.read_bytes() == CHINESE_HOMEPAGE_SOURCE.read_bytes()
