#!/usr/bin/env python3
"""Build the stride-align documentation alongside its tracked homepage.

``html/index.html`` is hand-authored and must never be overwritten here. The
localized homepage and shared assets are copied from ``website/``. Every
Markdown file (including the root README language matrix and ``docs/``) is
converted to themed long-form HTML alongside them.
"""

import html as _html
import posixpath
import re
import shutil
from pathlib import Path, PurePosixPath

import markdown
from api_catalog import render_api_catalog, render_vectorization_guide

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "html"
WEBSITE = REPO / "website"
OUT.mkdir(exist_ok=True)

# Fallback documentation theme for source archives that omit website/docs.css.
# Normal builds publish the shared Goblin-family theme from website/docs.css.
STYLE = """\
:root {
  --bg: #f1f5f9;
  --surface: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --rule: #e2e8f0;
  --rule-strong: #cbd5e1;
  --accent: #4f46e5;
  --accent-soft: #eef2ff;
  --code-bg: #f1f5f9;
  --pre-bg: #0f172a;
  --pre-text: #e2e8f0;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif;
  --mono: "IBM Plex Mono", "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
}

* { box-sizing: border-box; }
html { background: var(--bg); }

body {
  margin: 0;
  color: var(--text);
  background: var(--bg);
  font-family: var(--sans);
  font-size: 16px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

a { color: var(--accent); text-decoration: none; }
a:visited { color: var(--accent); }
a:hover { text-decoration: underline; }

.page {
  width: 880px;
  max-width: calc(100% - 32px);
  margin: 40px auto;
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 12px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, .04), 0 10px 30px rgba(15, 23, 42, .06);
  overflow: hidden;
}

.masthead {
  padding: 36px 40px 28px;
  background: var(--surface);
  border-bottom: 1px solid var(--rule);
}
.masthead h1 {
  margin: 0;
  font-size: clamp(30px, 5vw, 44px);
  line-height: 1.05;
  letter-spacing: -.03em;
  font-weight: 700;
  color: var(--text);
}
.masthead .tagline { margin: 10px 0 0; font-size: 16px; color: var(--muted); }

.nav {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 20px;
  padding: 12px 40px;
  background: #f8fafc;
  border-bottom: 1px solid var(--rule);
  font-size: 13px;
}
.nav .home, .nav .github, .nav .api {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .06em;
  font-weight: 600;
}
.nav .home:hover, .nav .github:hover, .nav .api:hover { color: var(--accent); text-decoration: none; }
.lang-switch { display: flex; gap: 14px; margin-left: auto; }
[dir="rtl"] .lang-switch { margin-left: 0; margin-right: auto; }
.lang-switch a { color: var(--muted); white-space: nowrap; }
.lang-switch a.active { color: var(--text); font-weight: 700; }

.content { padding: 32px 40px 44px; background: var(--surface); }
.content > :first-child { margin-top: 0; }

h1, h2, h3, h4, h5, h6 {
  color: var(--text);
  letter-spacing: -.02em;
  font-weight: 700;
  line-height: 1.25;
}
.content h1 { margin: 40px 0 14px; padding-bottom: 8px; border-bottom: 1px solid var(--rule); font-size: 30px; }
.content h2 { margin: 34px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--rule); font-size: 23px; }
.content h3 { margin: 26px 0 8px; font-size: 19px; }
.content h4 { margin: 20px 0 6px; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }

p { margin: 0 0 14px; }
ul, ol { margin: 8px 0 16px; padding-left: 26px; }
[dir="rtl"] ul, [dir="rtl"] ol { padding-left: 0; padding-right: 26px; }
li { margin: 5px 0; }
li::marker { color: var(--muted); }

code {
  background: var(--code-bg);
  color: #0f172a;
  padding: .12em .4em;
  border-radius: 5px;
  font-family: var(--mono);
  font-size: .88em;
}
/* code that is itself a link should read as a link (accent + soft tint) */
a code { color: var(--accent); background: var(--accent-soft); }
a:hover code { text-decoration: underline; }

pre {
  margin: 16px 0 20px;
  padding: 16px 18px;
  background: var(--pre-bg);
  color: var(--pre-text);
  border-radius: 10px;
  overflow-x: auto;
  font-family: var(--mono);
  font-size: 13.5px;
  line-height: 1.55;
}
pre code { background: transparent; color: inherit; padding: 0; border-radius: 0; font-size: inherit; }
[dir="rtl"] code, [dir="rtl"] pre { direction: ltr; unicode-bidi: embed; text-align: start; }

blockquote {
  margin: 16px 0;
  padding: 4px 18px;
  border-left: 3px solid var(--accent);
  background: var(--accent-soft);
  color: var(--muted);
  border-radius: 0 6px 6px 0;
}
[dir="rtl"] blockquote { border-left: 0; border-right: 3px solid var(--accent); border-radius: 6px 0 0 6px; }

hr { margin: 32px 0; border: 0; border-top: 1px solid var(--rule); }

table {
  width: 100%;
  margin: 18px 0 24px;
  border-collapse: collapse;
  font-size: 14px;
  border: 1px solid var(--rule);
  border-radius: 8px;
  overflow: hidden;
}
th, td { padding: 9px 12px; border-bottom: 1px solid var(--rule); text-align: start; vertical-align: top; }
th { background: #f8fafc; color: var(--text); font-weight: 600; border-bottom: 1px solid var(--rule-strong); }
tbody tr:hover { background: #f8fafc; }
tr:last-child td { border-bottom: 0; }
td[align="right"], th[align="right"] { text-align: end; }
td[align="center"], th[align="center"] { text-align: center; }

img { max-width: 100%; height: auto; border-radius: 8px; }
strong, b { color: var(--text); font-weight: 600; }

.footer {
  padding: 18px 40px;
  color: var(--muted);
  background: #f8fafc;
  border-top: 1px solid var(--rule);
  font-size: 13px;
  text-align: center;
}
.footer a { color: var(--muted); }
.footer a:hover { color: var(--accent); }

@media (max-width: 720px) {
  .page { margin: 16px auto; max-width: calc(100% - 20px); }
  .masthead { padding: 26px 22px 20px; }
  .nav, .content, .footer { padding-left: 22px; padding-right: 22px; }
  .content { padding-top: 24px; }
  .content h1 { font-size: 26px; }
  .content h2 { font-size: 20px; }
  pre { font-size: 13px; }
  table { font-size: 13px; }
}
"""

# Add a row per README translation that exists in the repo root.
LANGS = [
    ("", "en", "English", "ltr"),
    ("zh-CN", "zh-CN", "简体中文", "ltr"),
]


def readme_filename(suffix: str) -> str:
    return "README.md" if not suffix else f"README.{suffix}.md"


def html_filename(suffix: str) -> str:
    return "README.html" if not suffix else f"README.{suffix}.html"


def built_markdown_relpath(md_relpath: PurePosixPath) -> PurePosixPath:
    """Return the output path for a Markdown source path.

    Root README translations are long-form pages named ``README*.html``.
    Nested README files remain directory indexes, and every other Markdown
    filename keeps its stem.
    """
    if md_relpath.parent == PurePosixPath("."):
        for suffix, *_ in LANGS:
            if md_relpath.name == readme_filename(suffix):
                return PurePosixPath(html_filename(suffix))
    if md_relpath.name == "README.md":
        return md_relpath.parent / "index.html"
    return md_relpath.with_suffix(".html")


def built_link_target(target: str, *, source_relpath: PurePosixPath) -> str:
    """Map a repo-relative Markdown link to the corresponding built page."""
    base, hashsep, anchor = target.partition("#")
    if not base.endswith(".md") or "://" in base:
        return target

    source_dir = source_relpath.parent.as_posix()
    resolved_source = PurePosixPath(posixpath.normpath(posixpath.join(source_dir, base)))
    resolved_output = built_markdown_relpath(resolved_source)
    source_output_dir = built_markdown_relpath(source_relpath).parent.as_posix()
    relative_output = posixpath.relpath(resolved_output.as_posix(), start=source_output_dir)
    return relative_output + (hashsep + anchor if hashsep else "")


def render_md_to_html(md_text: str) -> str:
    extensions = [
        "fenced_code",
        "tables",
        "sane_lists",
        "toc",
        "attr_list",
    ]
    md = markdown.Markdown(extensions=extensions, output_format="html5")
    body = md.convert(md_text)
    return body


def rewrite_local_links(html_body: str, *, source_relpath: PurePosixPath) -> str:
    # Rewrite .md links to the built .html so converted pages cross-link
    # correctly. The root README maps to README.html, directory READMEs map
    # to index.html, and BENCHMARK.md maps to BENCHMARK.html.
    def replace(match: re.Match) -> str:
        target = built_link_target(match.group(2), source_relpath=source_relpath)
        return f'{match.group(1)}{target}"'

    html_body = re.sub(r'(href=")([^"]+\.md(?:#[^"]*)?)"', replace, html_body)

    # Drop a trailing .md/.html (keeping any #anchor) from filename-like link
    # TEXT, so rendered links read "cdist" rather than "cdist.md". Descriptive
    # text (anything containing whitespace) is left untouched.
    def strip_ext(m: re.Match) -> str:
        text = re.sub(
            r"^(\S+?)\.(?:md|html)(#\S*)?$", lambda t: t.group(1) + (t.group(2) or ""), m.group(2)
        )
        return m.group(1) + text + m.group(3)

    return re.sub(r"(<a\b[^>]*>)([^<]+)(</a>)", strip_ext, html_body)


def rewrite_llms_md_links(text: str) -> str:
    """Rewrite Markdown link targets in the llms*.txt files from their
    repo-relative .md sources to the .html pages this build produces, so the
    copies served on the website resolve. Root README*.md maps to
    README*.html, nested README.md maps to index.html, and everything else
    maps from .md to .html. The repo
    copies keep their .md links (correct for GitHub / source consumption)."""
    return re.sub(
        r"(\]\()([^)\s]+)(\))",
        lambda m: (
            m.group(1)
            + built_link_target(m.group(2), source_relpath=PurePosixPath("llms.txt"))
            + m.group(3)
        ),
        text,
    )


def strip_top_h1_and_lang_line(html_body: str, *, suffix: str) -> tuple[str, str | None]:
    # Pull out the first <h1>...</h1> as title, and the immediately-following
    # language switcher paragraph (which we replace with our own nav).
    title = None
    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", html_body, flags=re.S)
    if h1_match:
        title = re.sub(r"<.*?>", "", h1_match.group(1)).strip()
        html_body = html_body[: h1_match.start()] + html_body[h1_match.end() :]

    # Drop the "Languages:" line markup whether translated or not.
    # The README translations have a single <p> with many language links.
    p_match = re.match(
        r"\s*<p>\s*<strong>[^<]+</strong>\s*(?:<a[^>]+>[^<]+</a>\s*(?:·|<br\s*/?>|\s)*)+</p>",
        html_body,
    )
    if p_match:
        html_body = html_body[p_match.end() :]

    return html_body.lstrip(), title


def render_lang_switch(active_suffix: str) -> str:
    if len(LANGS) <= 1:
        return ""
    parts = []
    for suffix, code, name, _dir in LANGS:
        href = html_filename(suffix)
        active = suffix == active_suffix
        cls = ' class="active"' if active else ""
        parts.append(f'<a hreflang="{code}" href="{href}"{cls}>{name}</a>')
    return (
        '<div class="lang-switch" aria-label="Translations">\n  ' + "\n  ".join(parts) + "\n</div>"
    )


def template(
    *,
    title: str,
    lang: str,
    direction: str,
    masthead_title: str,
    tagline: str,
    nav_inner: str,
    content: str,
    home_link: str = "",
    asset_prefix: str = "",
    page_class: str = "",
    description: str = "SIMD-accelerated fuzzy string matching, sequence alignment, phonetics, and DTW for Python.",
) -> str:
    page_classes = "page" + (f" {_html.escape(page_class)}" if page_class else "")
    return f"""<!doctype html>
<html lang="{lang}" dir="{direction}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<meta name="description" content="{_html.escape(description)}">
<link rel="icon" type="image/png" href="{asset_prefix}goblin.png">
<link rel="apple-touch-icon" href="{asset_prefix}goblin.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&amp;family=IBM+Plex+Sans:wght@400;500;600&amp;display=swap" rel="stylesheet">
<link rel="stylesheet" href="{asset_prefix}style.css">
</head>
<body>
<div class="{page_classes}">
<header class="masthead">
<a class="masthead-product" href="https://goblinreactor.com" rel="noopener"><span class="docs-mascot" aria-hidden="true"><img src="{asset_prefix}goblin.png" alt=""></span><span>A Goblin Reactor product</span><b>↗</b></a>
<h1>{_html.escape(masthead_title)}</h1>
<p class="tagline">{tagline}</p>
</header>
<nav class="nav">
{home_link}<a class="api" href="{asset_prefix}docs/api/index.html">API</a><a class="github" href="https://github.com/adamdeprince/stride-align" rel="noopener">Source Code</a>
{nav_inner}
</nav>
<main class="content">
{content}
</main>
<footer class="footer">
stride-align · open source on
<a href="https://github.com/adamdeprince/stride-align">GitHub</a>
</footer>
</div>
<script src="{asset_prefix}language-switch.js"></script>
</body>
</html>
"""


def build_readme(suffix: str) -> None:
    md_path = REPO / readme_filename(suffix)
    if not md_path.exists():
        print(f"skip (no file): {md_path}")
        return
    md_text = md_path.read_text(encoding="utf-8")
    body = render_md_to_html(md_text)
    body = rewrite_local_links(body, source_relpath=PurePosixPath(readme_filename(suffix)))
    body, page_h1 = strip_top_h1_and_lang_line(body, suffix=suffix)

    lang = next(c for s, c, _n, _d in LANGS if s == suffix)
    direction = next(d for s, _c, _n, d in LANGS if s == suffix)
    name = next(n for s, _c, n, _d in LANGS if s == suffix)

    nav = render_lang_switch(suffix)
    tagline = (
        "Smith-Waterman &amp; Needleman-Wunsch with a fast SIMD backend"
        if suffix == ""
        else f"Smith-Waterman &amp; Needleman-Wunsch · {name}"
    )

    out_html = template(
        title=f"{page_h1 or 'stride-align'} — {name}" if suffix else (page_h1 or "stride-align"),
        lang=lang,
        direction=direction,
        masthead_title=page_h1 or "stride-align",
        tagline=tagline,
        nav_inner=nav,
        content=body.rstrip(),
        home_link='<a class="home" href="index.html">Home</a>',
    )
    out_path = OUT / html_filename(suffix)
    out_path.write_text(out_html, encoding="utf-8")
    print(f"wrote {out_path}")


def build_benchmark() -> None:
    md_path = REPO / "BENCHMARK.md"
    md_text = md_path.read_text(encoding="utf-8")
    body = render_md_to_html(md_text)
    body = rewrite_local_links(body, source_relpath=PurePosixPath("BENCHMARK.md"))
    body, page_h1 = strip_top_h1_and_lang_line(body, suffix="")

    out_html = template(
        title="Benchmark Summary — stride-align",
        lang="en",
        direction="ltr",
        masthead_title=page_h1 or "Benchmark Summary",
        tagline="Dated measurements, named baselines, raw receipts",
        nav_inner="",
        content=body.rstrip(),
        home_link='<a class="home" href="index.html">Home</a>',
        page_class="benchmark-page",
        description="Reproducible stride-align benchmarks with current Intel AVX-512 results, architecture references, methodology, and raw data.",
    )
    out_path = OUT / "BENCHMARK.html"
    out_path.write_text(out_html, encoding="utf-8")
    print(f"wrote {out_path}")


def build_generic(md_relpath: Path) -> None:
    """Convert any markdown file under REPO to a matching .html under OUT,
    preserving subdirectories. Used for everything that isn't the README
    matrix or the BENCHMARK page (those have bespoke builders above)."""
    md_path = REPO / md_relpath
    md_text = md_path.read_text(encoding="utf-8")
    is_api_index = md_relpath == Path("docs/api/README.md")
    if is_api_index:
        replacements = {
            "<!-- stride-vectorization-guide -->": render_vectorization_guide(),
            "<!-- stride-api-catalog -->": render_api_catalog(),
        }
        for marker, replacement in replacements.items():
            if marker not in md_text:
                raise ValueError(f"missing {marker} in {md_path}")
            md_text = md_text.replace(marker, replacement)
    body = render_md_to_html(md_text)
    md_posix = PurePosixPath(md_relpath.as_posix())
    body = rewrite_local_links(body, source_relpath=md_posix)
    body, page_h1 = strip_top_h1_and_lang_line(body, suffix="")

    # Compute the page's relative depth so the `home` link can point back
    # to index.html with the correct number of `../` segments.
    depth = len(md_relpath.parent.parts)
    up = "../" * depth
    home_link = f'<a class="home" href="{up}index.html">Home</a>'

    # `lang` is en by default; if the filename matches a translation
    # suffix in LANGS, switch to that lang code.
    name = md_path.stem  # e.g. "loongson-vs-tiger-lake-cdist-2026-05-24" or "README.zh-CN"
    lang = "en"
    direction = "ltr"
    for suffix, code, _label, lang_dir in LANGS:
        if suffix and name.endswith(f".{suffix}"):
            lang = code
            direction = lang_dir
            break

    title_base = page_h1 or md_path.stem
    out_html = template(
        title=f"{title_base} — stride-align",
        lang=lang,
        direction=direction,
        masthead_title=page_h1 or md_path.stem,
        tagline=(
            "One native algorithm surface for Python, R, and DuckDB"
            if is_api_index
            else "stride-align documentation"
        ),
        nav_inner="",
        content=body.rstrip(),
        home_link=home_link,
        asset_prefix=up,
        page_class="api-page" if is_api_index else "",
        description=(
            "Cross-language stride-align API examples for Python, R, and DuckDB."
            if is_api_index
            else "SIMD-accelerated fuzzy string matching, sequence alignment, phonetics, and DTW for Python."
        ),
    )
    out_path = OUT / Path(built_markdown_relpath(md_posix).as_posix())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_html, encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    # The English product homepage is a tracked, hand-authored file. The build
    # must fail if it is missing and must never replace it with generated copy.
    homepage = OUT / "index.html"
    if not homepage.exists():
        raise FileNotFoundError(f"missing tracked homepage: {homepage}")
    print(f"preserved tracked homepage {homepage}")

    # Localized content and shared assets remain source-controlled outside the
    # otherwise-generated html/ tree and are refreshed on every build.
    for name in (
        "index.zh-CN.html",
        "home.css",
        "goblin.png",
        "language-switch.js",
    ):
        source = WEBSITE / name
        if not source.exists():
            raise FileNotFoundError(f"missing homepage source: {source}")
        shutil.copyfile(source, OUT / name)
        print(f"copied {source} -> {OUT / name}")

    docs_style = WEBSITE / "docs.css"
    if docs_style.exists():
        shutil.copyfile(docs_style, OUT / "style.css")
        print(f"copied {docs_style} -> {OUT / 'style.css'}")
    else:
        (OUT / "style.css").write_text(STYLE, encoding="utf-8")
        print(f"wrote fallback {OUT / 'style.css'}")
    # README language matrix → README*.html
    for suffix, *_ in LANGS:
        build_readme(suffix)
    # BENCHMARK has its own bespoke builder (different tagline, home link).
    build_benchmark()

    # Everything else: walk the repo for *.md, skip the ones the bespoke
    # builders already produced (README*.md, BENCHMARK.md) and ignore the
    # vendored / staging trees.
    handled_root = {
        REPO / "BENCHMARK.md",
        *(REPO / readme_filename(suffix) for suffix, *_ in LANGS),
    }
    skip_dirs = {
        "html",
        "build",
        "dist",
        "wheelhouse",
        ".venv",
        ".git",
        ".pytest_cache",
        ".wrangler",
        "__pycache__",
        "node_modules",
    }
    for md in sorted(REPO.rglob("*.md")):
        if md in handled_root:
            continue
        if any(
            part in skip_dirs or part.startswith("wheelhouse_")
            for part in md.relative_to(REPO).parts
        ):
            continue
        rel = md.relative_to(REPO)
        build_generic(rel)

    # Publish the raw benchmark ledger and dated artifacts referenced by the
    # generated benchmark documentation. These are small evidence files, not
    # build intermediates.
    benchmark_sources = [REPO / "benchmark.csv"]
    benchmark_dir = REPO / "benchmarks"
    for pattern in ("*.csv", "*.txt", "*.json"):
        benchmark_sources.extend(sorted(benchmark_dir.glob(pattern)))
    for src in benchmark_sources:
        if not src.exists():
            continue
        rel = src.relative_to(REPO)
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"copied benchmark data {rel}")

    # Runnable tutorial sources linked by the generated README pages. The
    # corpus and local caches stay out of the website; the demo downloads its
    # public-domain input when a checkout does not already contain it.
    for src in sorted((REPO / "demo").glob("*.py")):
        rel = src.relative_to(REPO)
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"copied runnable demo {rel}")

    # Copy the LLM context files verbatim into the site so they are served
    # (and discoverable) at stride-align.com/llms.txt and /llms-full.txt.
    # They are plain text, not markdown, so they bypass the HTML builders.
    for name in ("llms.txt", "llms-full.txt"):
        src = REPO / name
        if src.exists():
            (OUT / name).write_text(
                rewrite_llms_md_links(src.read_text(encoding="utf-8")), encoding="utf-8"
            )
            print(f"copied {OUT / name} (.md links rewritten to .html)")
        else:
            print(f"WARNING: {src} not found; not copied into html/")


if __name__ == "__main__":
    main()
