#!/usr/bin/env python3
# ruff: noqa: E501
"""Generate the Go fallback download repository from built artifact records."""

from __future__ import annotations

import argparse
import json
import tomllib
from datetime import date
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BINDING = ROOT / "bindings" / "go"
DIST = ROOT / "dist" / "go"
BASE_URL = "https://distribution.goblinreactor.com/stride-align/go/"


TARGETS = (
    {
        "platform": "darwin_arm64",
        "profile": "neon",
        "platform_en": "macOS arm64 13+",
        "platform_zh": "macOS arm64 13+",
        "cpu_en": "NEON / Apple M1 baseline",
        "cpu_zh": "NEON / Apple M1 基线",
    },
    {
        "platform": "linux_amd64",
        "profile": "generic",
        "platform_en": "Linux x86-64",
        "platform_zh": "Linux x86-64",
        "cpu_en": "generic / SSE2 baseline",
        "cpu_zh": "generic / SSE2 基线",
    },
    {
        "platform": "linux_amd64",
        "profile": "avx2",
        "platform_en": "Linux x86-64",
        "platform_zh": "Linux x86-64",
        "cpu_en": "AVX2 / x86-64-v3",
        "cpu_zh": "AVX2 / x86-64-v3",
    },
    {
        "platform": "linux_amd64",
        "profile": "avx512bwvl",
        "platform_en": "Linux x86-64",
        "platform_zh": "Linux x86-64",
        "cpu_en": "AVX-512 F/BW/VL/DQ / x86-64-v4",
        "cpu_zh": "AVX-512 F/BW/VL/DQ / x86-64-v4",
    },
    {
        "platform": "linux_arm64",
        "profile": "neon",
        "platform_en": "Linux arm64",
        "platform_zh": "Linux arm64",
        "cpu_en": "Armv8-A / NEON baseline",
        "cpu_zh": "Armv8-A / NEON 基线",
    },
    {
        "platform": "linux_ppc64le",
        "profile": "power8_vsx",
        "platform_en": "Linux ppc64le",
        "platform_zh": "Linux ppc64le",
        "cpu_en": "POWER8+ / VSX",
        "cpu_zh": "POWER8+ / VSX",
    },
    {
        "platform": "linux_loong64",
        "profile": "la464_lsx",
        "platform_en": "Linux LoongArch64",
        "platform_zh": "Linux 龙架构 64 位",
        "cpu_en": "LA464 / LSX",
        "cpu_zh": "LA464 / LSX",
    },
    {
        "platform": "linux_loong64",
        "profile": "la464_lasx",
        "platform_en": "Linux LoongArch64",
        "platform_zh": "Linux 龙架构 64 位",
        "cpu_en": "LA464 / LASX",
        "cpu_zh": "LA464 / LASX",
    },
    {
        "platform": "linux_loong64",
        "profile": "la664_lasx",
        "platform_en": "Linux LoongArch64",
        "platform_zh": "Linux 龙架构 64 位",
        "cpu_en": "LA664 / LASX",
        "cpu_zh": "LA664 / LASX",
    },
)


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        return str(tomllib.load(project_file)["project"]["version"])


def load_records(version: str, runtime_tested: set[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted((DIST / f"v{version}").glob("*/*/artifact-record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = f"{record['platform']}/{record['profile']}"
        artifact = DIST / str(record["path"])
        if not artifact.is_file():
            raise SystemExit(f"artifact record points to missing file: {artifact}")
        if key in runtime_tested:
            record["validation"] = "runtime-tested"
        records.append(record)
    return records


def styles() -> str:
    return """
    :root { --ink:#15231b; --muted:#617068; --green:#21493a; --blue:#315f78; --paper:#fff; --soft:#f2f6f3; --rule:#ced8d1; font-family:"IBM Plex Sans",system-ui,sans-serif; }
    * { box-sizing:border-box; }
    html { color:var(--ink); background:#edf3ef; background-image:linear-gradient(rgba(33,73,58,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(33,73,58,.035) 1px,transparent 1px); background-size:40px 40px; }
    body { width:min(1180px,calc(100% - 36px)); margin:24px auto; background:var(--paper); border:1px solid var(--rule); }
    a { color:var(--blue); }
    .topbar { min-height:56px; display:flex; align-items:center; justify-content:space-between; padding:0 28px; border-bottom:1px solid var(--rule); background:var(--soft); font:600 10px/1 "IBM Plex Mono",monospace; letter-spacing:.07em; text-transform:uppercase; }
    .topbar strong { color:var(--green); }
    .topbar nav { display:flex; align-self:stretch; }
    .topbar a { display:flex; align-items:center; padding:0 16px; border-left:1px solid var(--rule); color:var(--green); text-decoration:none; }
    .topbar a:hover { color:#fff; background:var(--green); }
    header { padding:58px 54px 50px; border-bottom:1px solid var(--rule); }
    .eyebrow { margin:0 0 18px; color:var(--green); font:600 9px/1 "IBM Plex Mono",monospace; letter-spacing:.09em; text-transform:uppercase; }
    h1 { max-width:900px; margin:0; font-size:clamp(42px,7vw,76px); font-weight:500; line-height:.98; letter-spacing:-.045em; }
    header p { max-width:820px; margin:24px 0 0; color:var(--muted); font-size:18px; line-height:1.65; }
    .primary-cta { display:inline-flex; align-items:center; min-height:50px; margin-top:28px; padding:0 20px; color:#fff; background:var(--green); border:1px solid var(--green); font:600 10px/1 "IBM Plex Mono",monospace; letter-spacing:.065em; text-decoration:none; text-transform:uppercase; }
    .primary-cta:hover { color:var(--green); background:#fff; }
    main { padding:54px; }
    .section-heading { display:grid; grid-template-columns:minmax(0,1fr) minmax(280px,.65fr); gap:42px; align-items:end; }
    h2 { margin:0; font-size:clamp(28px,4vw,42px); font-weight:500; letter-spacing:-.035em; }
    .section-heading p,.note { margin:0; color:var(--muted); line-height:1.65; }
    .table-wrap { overflow-x:auto; margin-top:30px; border:1px solid var(--rule); }
    table { width:100%; min-width:930px; border-collapse:collapse; font-size:14px; }
    th,td { padding:16px 15px; border-bottom:1px solid var(--rule); text-align:left; vertical-align:middle; }
    th { color:var(--muted); background:var(--soft); font:600 9px/1.2 "IBM Plex Mono",monospace; letter-spacing:.065em; text-transform:uppercase; }
    tr:last-child td { border-bottom:0; }
    tbody tr:nth-child(even) { background:#f9fbfa; }
    .download { display:inline-flex; align-items:center; min-height:36px; padding:0 13px; color:#fff; background:var(--green); border:1px solid var(--green); font:600 9px/1 "IBM Plex Mono",monospace; letter-spacing:.055em; text-decoration:none; text-transform:uppercase; }
    .download:hover { color:var(--green); background:#fff; }
    .checksum { margin-left:9px; font:500 10px/1 "IBM Plex Mono",monospace; }
    .status { color:var(--green); white-space:nowrap; }
    .pending { color:var(--muted); font:600 9px/1 "IBM Plex Mono",monospace; letter-spacing:.06em; text-transform:uppercase; }
    .notes { display:grid; grid-template-columns:1fr 1fr; gap:28px; margin-top:28px; }
    .note { padding:18px 20px; border-left:3px solid var(--green); background:var(--soft); font-size:14px; }
    code,pre { font-family:"IBM Plex Mono",monospace; }
    code { color:var(--green); }
    h3 { margin:54px 0 14px; font-size:24px; font-weight:500; }
    pre { margin:0; padding:22px; overflow:auto; color:#e4ece7; background:#14221a; font-size:13px; line-height:1.6; }
    footer { display:flex; flex-wrap:wrap; justify-content:space-between; gap:12px; padding:22px 28px; color:#aebbb3; background:#14221a; font:500 10px/1.4 "IBM Plex Mono",monospace; }
    footer a { color:#dbe6df; }
    @media (max-width:760px) { body{width:100%;margin:0;border:0}.topbar{padding:0 16px}.topbar strong{font-size:9px}.topbar a{padding:0 10px}header,main{padding:38px 22px}.section-heading,.notes{grid-template-columns:1fr}.section-heading{gap:14px}header p{font-size:16px}.table-wrap{margin-right:-22px;margin-left:-22px;border-right:0;border-left:0} }
    """.strip()


def table_rows(records: list[dict[str, object]], locale: str) -> str:
    by_key = {(record["platform"], record["profile"]): record for record in records}
    rows = []
    for target in TARGETS:
        record = by_key.get((target["platform"], target["profile"]))
        platform = target["platform_zh" if locale == "zh-CN" else "platform_en"]
        cpu = target["cpu_zh" if locale == "zh-CN" else "cpu_en"]
        if record is None:
            pending = "待发布" if locale == "zh-CN" else "Pending"
            rows.append(
                f'<tr class="pending-row"><td>{platform}</td><td>{cpu}</td>'
                f'<td><span class="pending">{pending}</span></td><td>—</td></tr>'
            )
            continue
        path = escape(str(record["path"]))
        label = "下载" if locale == "zh-CN" else "Download"
        validation = (
            "运行测试" if record["validation"] == "runtime-tested" else "构建测试"
        ) if locale == "zh-CN" else (
            "Runtime tested" if record["validation"] == "runtime-tested" else "Build tested"
        )
        rows.append(
            f"<tr><td>{platform}</td><td>{cpu}</td><td>"
            f'<a class="download" href="{path}">{label}</a>'
            f'<a class="checksum" href="{path}.sha256">SHA-256</a>'
            f'</td><td class="status">{validation}</td></tr>'
        )
    return "\n            ".join(rows)


def page(records: list[dict[str, object]], version: str, locale: str) -> str:
    chinese = locale == "zh-CN"
    other = "index.html" if chinese else "index.zh-CN.html"
    language_link = "English" if chinese else "简体中文"
    title = "Go 预编译后端 — stride-align" if chinese else "Go precompiled backends — stride-align"
    description = (
        f"下载 stride-align {version} 的 Go 预编译 C++23 后端。"
        if chinese
        else f"Download precompiled C++23 backends for the stride-align {version} Go package."
    )
    eyebrow = (
        f"版本 {version} · Go 1.22+ · C++23 二进制回退"
        if chinese
        else f"Release {version} · Go 1.22+ · C++23 binary fallback"
    )
    headline = "保留 Go API，省去 C++23 编译。" if chinese else "Keep the Go API. Skip the C++23 compile."
    lede = (
        "常规安装仍由 Go 下载源代码并通过 cgo 构建。这些静态后端只为无法使用合适 C++23 编译器的系统提供回退；Go 软件包、API 与结果均不改变。"
        if chinese
        else "The normal installation still lets Go download the source and build it through cgo. These static backends are a fallback for systems without a suitable C++23 compiler; the Go package, API, and results do not change."
    )
    packages = "可用下载" if chinese else "Available downloads"
    package_note = (
        "安装器会选择保守配置。只有在部署处理器明确支持时才选择更宽的 SIMD 配置。"
        if chinese
        else "The installer chooses a conservative profile. Select a wider SIMD profile only when the deployment CPU explicitly supports it."
    )
    heads = ("平台", "CPU 软件包", "文件", "验证") if chinese else ("Platform", "CPU package", "Artifact", "Validation")
    source_note = (
        "首选源码安装：它由 Go 模块校验和保护，并在应用程序的实际目标上编译。"
        if chinese
        else "Prefer the source installation: it is covered by Go module checksums and compiles on the application's actual target."
    )
    fallback_note = (
        "预编译模式仍需要 cgo、C 编译器与 pkg-config，但不会调用 C++ 编译器。Linux 软件包包含静态 C++ 运行库并以 glibc 2.28 为基线；安装器会在解压前验证 SHA-256。"
        if chinese
        else "Precompiled mode still needs cgo, a C compiler, and pkg-config, but never invokes a C++ compiler. Linux packages include the static C++ runtime and use a glibc 2.28 baseline; the installer verifies SHA-256 before extraction."
    )
    install_title = "从源代码安装（推荐）" if chinese else "Install from source (recommended)"
    fallback_title = "安装预编译回退" if chinese else "Install the precompiled fallback"
    installer_comment = (
        "# 安装器会输出所需的 PKG_CONFIG_PATH 与 go build 命令"
        if chinese
        else "# The installer prints the required PKG_CONFIG_PATH and go build command."
    )
    bottom_note = (
        "Intel 用户可传入 --profile avx2 或 --profile avx512bwvl。Linux arm64、POWER 与龙架构软件包仍待发布；这些平台目前请使用源码安装。软件包清单与全部校验和也可单独下载。"
        if chinese
        else "Intel users may pass --profile avx2 or --profile avx512bwvl. Linux arm64, POWER, and LoongArch packages remain pending; use the source installation on those platforms for now. The machine-readable manifest and complete checksum list are also available."
    )
    guide = "指南" if chinese else "Guide"
    api = "编程接口" if chinese else "API"
    view = "查看下载 ↓" if chinese else "View downloads ↓"
    source_code = "源代码" if chinese else "Source code"
    manifest_label = "软件包清单" if chinese else "package manifest"
    checksums_label = "全部校验和" if chinese else "all checksums"
    return f"""<!doctype html>
<html lang="{locale}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <meta name="description" content="{description}">
  <meta name="theme-color" content="#21493a">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&amp;family=IBM+Plex+Sans:wght@400;500;600&amp;display=swap" rel="stylesheet">
  <style>
{styles()}
  </style>
</head>
<body>
  <div class="topbar">
    <strong>stride-align / Go</strong>
    <nav aria-label="{'页面导航' if chinese else 'Page navigation'}">
      <a href="https://stride-align.com/bindings/go/">{guide}</a>
      <a href="https://stride-align.com/docs/api/{'index.zh-CN.html' if chinese else 'index.html'}?language=go">{api}</a>
      <a href="{other}" hreflang="{'en' if chinese else 'zh-CN'}" lang="{'en' if chinese else 'zh-CN'}">{language_link}</a>
    </nav>
  </div>
  <header>
    <p class="eyebrow">{eyebrow}</p>
    <h1>{headline}</h1>
    <p>{lede}</p>
    <a class="primary-cta" href="#packages">{view}</a>
  </header>
  <main>
    <section id="packages" aria-labelledby="packages-title">
      <div class="section-heading">
        <h2 id="packages-title">{packages}</h2>
        <p>{package_note}</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>{heads[0]}</th><th>{heads[1]}</th><th>{heads[2]}</th><th>{heads[3]}</th></tr></thead>
          <tbody>
            {table_rows(records, locale)}
          </tbody>
        </table>
      </div>
      <div class="notes">
        <p class="note">{source_note}</p>
        <p class="note">{fallback_note}</p>
      </div>
    </section>

    <h3>{install_title}</h3>
    <pre><code>go get github.com/adamdeprince/stride-align/bindings/go@main</code></pre>

    <h3>{fallback_title}</h3>
    <pre><code>go run github.com/adamdeprince/stride-align/bindings/go/cmd/stridealign-prebuilt@main

{installer_comment}</code></pre>
    <p class="note" style="margin-top:20px">{bottom_note} <a href="manifest.json">{manifest_label}</a> · <a href="SHA256SUMS">{checksums_label}</a>.</p>
  </main>
  <footer><span>stride-align {version} · Go 1.22+ · Apache-2.0</span><a href="https://github.com/adamdeprince/stride-align">{source_code} ↗</a></footer>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-tested",
        action="append",
        default=[],
        metavar="PLATFORM/PROFILE",
        help="mark a built target as runtime-tested; may be repeated",
    )
    args = parser.parse_args()

    version = project_version()
    records = load_records(version, set(args.runtime_tested))
    expected = {(target["platform"], target["profile"]) for target in TARGETS}
    actual = {(record["platform"], record["profile"]) for record in records}
    unexpected = actual - expected
    if unexpected:
        raise SystemExit(f"unexpected Go artifacts: {sorted(unexpected)}")
    if any(record.get("build_number") != 1 for record in records):
        raise SystemExit("all Go artifacts must use packaging build number 1")

    manifest = {
        "schema_version": 1,
        "project": "stride-align",
        "project_version": version,
        "build_number": 1,
        "source_standard": "C++23",
        "linux_abi": "manylinux_2_28",
        "base_url": BASE_URL,
        "published_at": date.today().isoformat(),
        "artifacts": records,
        "pending": [
            {"platform": target["platform"], "profile": target["profile"]}
            for target in TARGETS
            if (target["platform"], target["profile"]) not in actual
        ],
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    checksums = "".join(f"{record['sha256']}  {record['path']}\n" for record in records)
    english = page(records, version, "en")
    chinese = page(records, version, "zh-CN")

    outputs = {
        BINDING / "repository-manifest.json": manifest_text,
        BINDING / "repository-SHA256SUMS": checksums,
        BINDING / "repository-index.html": english,
        BINDING / "repository-index.zh-CN.html": chinese,
        DIST / "manifest.json": manifest_text,
        DIST / "SHA256SUMS": checksums,
        DIST / "index.html": english,
        DIST / "index.zh-CN.html": chinese,
    }
    for path, contents in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    for record in records:
        artifact = DIST / str(record["path"])
        checksum = artifact.with_suffix(artifact.suffix + ".sha256")
        if not checksum.is_file():
            checksum.write_text(
                f"{record['sha256']}  {artifact.name}\n",
                encoding="utf-8",
            )
    print(f"generated Go repository metadata for {len(records)} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
