#!/usr/bin/env python3
"""Convert stride-align README*.md and BENCHMARK.md to themed HTML."""

import html as _html
import re
from pathlib import Path

import markdown

REPO = Path("/home/adam/dev/stride-align")
OUT = REPO / "html"
OUT.mkdir(exist_ok=True)

# Languages: filename suffix -> (lang code, display name, dir)
LANGS = [
    ("",        "en",     "English",              "ltr"),
    ("zh-CN",   "zh-CN",  "简体中文",              "ltr"),
    ("zh-TW",   "zh-TW",  "繁體中文",              "ltr"),
    ("ja",      "ja",     "日本語",                "ltr"),
    ("de",      "de",     "Deutsch",              "ltr"),
    ("ko",      "ko",     "한국어",                "ltr"),
    ("fr",      "fr",     "Français",             "ltr"),
    ("es",      "es",     "Español",              "ltr"),
    ("pt-BR",   "pt-BR",  "Português do Brasil",  "ltr"),
    ("ru",      "ru",     "Русский",              "ltr"),
    ("vi",      "vi",     "Tiếng Việt",           "ltr"),
    ("id",      "id",     "Bahasa Indonesia",     "ltr"),
    ("hi",      "hi",     "हिन्दी",                "ltr"),
    ("ar",      "ar",     "العربية",              "rtl"),
    ("tr",      "tr",     "Türkçe",               "ltr"),
    ("pl",      "pl",     "Polski",               "ltr"),
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
    home_label: str,
    home_href: str,
) -> str:
    return f"""<!doctype html>
<html lang="{lang}" dir="{direction}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<meta name="description" content="stride-align — Smith-Waterman and Needleman-Wunsch with a fast SIMD backend.">
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
<a class="home" href="{home_href}">{home_label}</a>
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
        home_label={
            "en": "README", "ja": "README", "de": "README", "fr": "README",
            "es": "README", "ko": "README", "zh-CN": "README", "zh-TW": "README",
            "pt-BR": "README", "ru": "README", "vi": "README", "id": "README",
            "hi": "README", "ar": "README", "tr": "README", "pl": "README",
        }[lang],
        home_href="README.html",
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
        home_label="README",
        home_href="README.html",
    )
    out_path = OUT / "BENCHMARK.html"
    out_path.write_text(out_html, encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    for suffix, *_ in LANGS:
        build_readme(suffix)
    build_benchmark()


if __name__ == "__main__":
    main()
