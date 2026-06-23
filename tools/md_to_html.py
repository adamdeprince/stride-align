#!/usr/bin/env python3
"""Convert every stride-align *.md file (including docs/) to themed HTML
in the html/ tree. Preserves docs/ subdirectory layout. Handles the
README language matrix via the LANGS table; everything else is rendered
through a single generic builder."""

import html as _html
import re
import shutil
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "html"
OUT.mkdir(exist_ok=True)

# Add a row per README translation that exists in the repo root.
LANGS = [
    ("",        "en",     "English",              "ltr"),
    ("zh-CN",   "zh-CN",  "简体中文",             "ltr"),
]


def readme_filename(suffix: str) -> str:
    return "README.md" if not suffix else f"README.{suffix}.md"


def html_filename(suffix: str) -> str:
    return "index.html" if not suffix else f"index.{suffix}.html"


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


def rewrite_local_links(html_body: str, *, is_readme: bool) -> str:
    # Rewrite .md links to .html so the converted pages cross-link correctly.
    # README*.md -> README*.html, BENCHMARK.md -> BENCHMARK.html.
    def replace(match: re.Match) -> str:
        prefix = match.group(1)
        target = match.group(2)
        # Strip any anchor / query
        anchor = ""
        if "#" in target:
            target, anchor = target.split("#", 1)
            anchor = "#" + anchor
        if target.endswith(".md"):
            target = target[:-3] + ".html"
        return f'{prefix}{target}{anchor}"'

    return re.sub(r'(href=")([^"]+\.md(?:#[^"]*)?)"', replace, html_body)


def strip_top_h1_and_lang_line(html_body: str, *, suffix: str) -> tuple[str, str | None]:
    # Pull out the first <h1>...</h1> as title, and the immediately-following
    # language switcher paragraph (which we replace with our own nav).
    title = None
    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", html_body, flags=re.S)
    if h1_match:
        title = re.sub(r"<.*?>", "", h1_match.group(1)).strip()
        html_body = html_body[: h1_match.start()] + html_body[h1_match.end():]

    # Drop the "Languages:" line markup whether translated or not.
    # The README translations have a single <p> with many language links.
    p_match = re.match(
        r"\s*<p>\s*<strong>[^<]+</strong>\s*(?:<a[^>]+>[^<]+</a>\s*(?:·|<br\s*/?>|\s)*)+</p>",
        html_body,
    )
    if p_match:
        html_body = html_body[p_match.end():]

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
    return '<div class="lang-switch" aria-label="Translations">\n  ' + "\n  ".join(parts) + "\n</div>"


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
) -> str:
    return f"""<!doctype html>
<html lang="{lang}" dir="{direction}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<meta name="description" content="stride-align — Smith-Waterman and Needleman-Wunsch with a fast SIMD backend.">
<link rel="icon" href="favicon.ico" sizes="any">
<link rel="icon" type="image/png" href="favicon-32.png" sizes="32x32">
<link rel="icon" type="image/png" href="favicon-16.png" sizes="16x16">
<link rel="apple-touch-icon" href="apple-touch-icon.png" sizes="180x180">
<link rel="manifest" href="site.webmanifest">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css">
</head>
<body>
<div class="page">
<header class="masthead">
<h1>{_html.escape(masthead_title)}</h1>
<p class="tagline">{tagline}</p>
</header>
<nav class="nav">
{home_link}<a class="github" href="https://github.com/adamdeprince/stride-align" rel="noopener">Source Code</a>
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
    body = rewrite_local_links(body, is_readme=True)
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
        home_link="",
    )
    out_path = OUT / html_filename(suffix)
    out_path.write_text(out_html, encoding="utf-8")
    print(f"wrote {out_path}")


def build_benchmark() -> None:
    md_path = REPO / "BENCHMARK.md"
    md_text = md_path.read_text(encoding="utf-8")
    body = render_md_to_html(md_text)
    body = rewrite_local_links(body, is_readme=False)
    body, page_h1 = strip_top_h1_and_lang_line(body, suffix="")

    out_html = template(
        title="Benchmark Summary — stride-align",
        lang="en",
        direction="ltr",
        masthead_title=page_h1 or "Benchmark Summary",
        tagline="Cross-architecture performance: Intel x86, ARM, Loongson, Power",
        nav_inner="",
        content=body.rstrip(),
        home_link='<a class="home" href="index.html">README</a>',
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
    body = render_md_to_html(md_text)
    body = rewrite_local_links(body, is_readme=False)
    body, page_h1 = strip_top_h1_and_lang_line(body, suffix="")

    # Compute the page's relative depth so the `home` link can point back
    # to index.html with the correct number of `../` segments.
    depth = len(md_relpath.parent.parts)
    up = "../" * depth
    home_link = f'<a class="home" href="{up}index.html">README</a>'

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
        tagline="stride-align documentation",
        nav_inner="",
        content=body.rstrip(),
        home_link=home_link,
    )
    out_path = OUT / md_relpath.with_suffix(".html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_html, encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    # README language matrix → index*.html
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
    skip_dirs = {"html", "build", "dist", "wheelhouse", ".venv", ".git",
                 ".pytest_cache", "__pycache__", "node_modules"}
    for md in sorted(REPO.rglob("*.md")):
        if md in handled_root:
            continue
        if any(part in skip_dirs or part.startswith("wheelhouse_") for part in md.relative_to(REPO).parts):
            continue
        rel = md.relative_to(REPO)
        build_generic(rel)

    # Copy the LLM context files verbatim into the site so they are served
    # (and discoverable) at stride-align.com/llms.txt and /llms-full.txt.
    # They are plain text, not markdown, so they bypass the HTML builders.
    for name in ("llms.txt", "llms-full.txt"):
        src = REPO / name
        if src.exists():
            shutil.copy2(src, OUT / name)
            print(f"copied {OUT / name}")
        else:
            print(f"WARNING: {src} not found; not copied into html/")


if __name__ == "__main__":
    main()
