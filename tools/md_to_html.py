#!/usr/bin/env python3
"""Convert every stride-align *.md file (including docs/) to themed HTML
in the html/ tree. Preserves docs/ subdirectory layout. Handles the
README language matrix via the LANGS table; everything else is rendered
through a single generic builder."""

import html as _html
import re
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "html"
OUT.mkdir(exist_ok=True)

# Site theme ("Modern light"). Written verbatim to html/style.css by main()
# so the whole site regenerates from this one tracked script (html/ itself
# is git-ignored). Edit here to retheme.
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
    ("",        "en",     "English",              "ltr"),
    ("zh-CN",   "zh-CN",  "简体中文",             "ltr"),
]


def readme_filename(suffix: str) -> str:
    return "README.md" if not suffix else f"README.{suffix}.md"


def html_filename(suffix: str) -> str:
    return "index.html" if not suffix else f"index.{suffix}.html"


def built_html_name(filename: str) -> str:
    """The built .html name for a single source filename: a directory-index
    README[.lang].md -> index[.lang].html, otherwise X.md -> X.html."""
    for suffix, *_ in LANGS:
        if filename == readme_filename(suffix):
            return html_filename(suffix)
    if filename.endswith(".md"):
        return filename[:-3] + ".html"
    return filename


def built_link_target(target: str) -> str:
    """Map a link target (possibly nested, possibly anchored) to its built
    page, applying built_html_name to the last path component. Non-.md
    targets (.txt, external URLs) pass through unchanged."""
    base, hashsep, anchor = target.partition("#")
    head, slash, name = base.rpartition("/")
    return head + slash + built_html_name(name) + (hashsep + anchor if hashsep else "")


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
    # Rewrite .md links to the built .html so converted pages cross-link
    # correctly. Directory READMEs map to index.html (README.md -> index.html,
    # docs/api/README.md -> docs/api/index.html); BENCHMARK.md -> BENCHMARK.html.
    def replace(match: re.Match) -> str:
        return f'{match.group(1)}{built_link_target(match.group(2))}"'

    html_body = re.sub(r'(href=")([^"]+\.md(?:#[^"]*)?)"', replace, html_body)

    # Drop a trailing .md/.html (keeping any #anchor) from filename-like link
    # TEXT, so rendered links read "cdist" rather than "cdist.md". Descriptive
    # text (anything containing whitespace) is left untouched.
    def strip_ext(m: re.Match) -> str:
        text = re.sub(r"^(\S+?)\.(?:md|html)(#\S*)?$",
                      lambda t: t.group(1) + (t.group(2) or ""),
                      m.group(2))
        return m.group(1) + text + m.group(3)

    return re.sub(r"(<a\b[^>]*>)([^<]+)(</a>)", strip_ext, html_body)


def rewrite_llms_md_links(text: str) -> str:
    """Rewrite Markdown link targets in the llms*.txt files from their
    repo-relative .md sources to the .html pages this build produces, so the
    copies served on the website resolve. README*.md maps to index*.html
    (the README's built filename), everything else .md -> .html. The repo
    copies keep their .md links (correct for GitHub / source consumption)."""
    return re.sub(r"(\]\()([^)\s]+)(\))",
                  lambda m: m.group(1) + built_link_target(m.group(2)) + m.group(3),
                  text)


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
    asset_prefix: str = "",
) -> str:
    return f"""<!doctype html>
<html lang="{lang}" dir="{direction}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<meta name="description" content="SIMD-accelerated fuzzy string matching, sequence alignment, phonetics, and DTW for Python — a faster drop-in for rapidfuzz and parasail.">
<link rel="icon" href="{asset_prefix}favicon.ico" sizes="any">
<link rel="icon" type="image/png" href="{asset_prefix}favicon-32.png" sizes="32x32">
<link rel="icon" type="image/png" href="{asset_prefix}favicon-16.png" sizes="16x16">
<link rel="apple-touch-icon" href="{asset_prefix}apple-touch-icon.png" sizes="180x180">
<link rel="manifest" href="{asset_prefix}site.webmanifest">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{asset_prefix}style.css">
</head>
<body>
<div class="page">
<header class="masthead">
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
        asset_prefix=up,
    )
    out_path = OUT / md_relpath.parent / built_html_name(md_relpath.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_html, encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    # Site stylesheet (theme lives in STYLE above) — written into html/.
    (OUT / "style.css").write_text(STYLE, encoding="utf-8")
    print(f"wrote {OUT / 'style.css'}")
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
            (OUT / name).write_text(
                rewrite_llms_md_links(src.read_text(encoding="utf-8")),
                encoding="utf-8")
            print(f"copied {OUT / name} (.md links rewritten to .html)")
        else:
            print(f"WARNING: {src} not found; not copied into html/")


if __name__ == "__main__":
    main()
