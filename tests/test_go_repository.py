from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GO_BINDING = ROOT / "bindings" / "go"
DOWNLOAD_BASE = "https://distribution.goblinreactor.com/stride-align/go/"
EXPECTED_TARGETS = {
    ("darwin_arm64", "neon"),
    ("linux_amd64", "generic"),
    ("linux_amd64", "avx2"),
    ("linux_amd64", "avx512bwvl"),
    ("linux_arm64", "neon"),
    ("linux_ppc64le", "power8_vsx"),
    ("linux_loong64", "la464_lsx"),
    ("linux_loong64", "la464_lasx"),
    ("linux_loong64", "la664_lasx"),
}


def test_go_repository_matches_lockstep_version_and_target_matrix() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        project_version = str(tomllib.load(project_file)["project"]["version"])

    manifest = json.loads(
        (GO_BINDING / "repository-manifest.json").read_text(encoding="utf-8")
    )
    artifacts = manifest["artifacts"]
    available = {(artifact["platform"], artifact["profile"]) for artifact in artifacts}
    pending = {(target["platform"], target["profile"]) for target in manifest["pending"]}
    artifact_hashes = {artifact["path"]: artifact["sha256"] for artifact in artifacts}
    checksum_hashes = {
        path: digest
        for digest, path in (
            line.split(maxsplit=1)
            for line in (GO_BINDING / "repository-SHA256SUMS")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }

    assert manifest["project"] == "stride-align"
    assert manifest["project_version"] == project_version
    assert manifest["build_number"] == 1
    assert manifest["source_standard"] == "C++23"
    assert manifest["linux_abi"] == "manylinux_2_28"
    assert manifest["base_url"] == DOWNLOAD_BASE
    assert available.isdisjoint(pending)
    assert available | pending == EXPECTED_TARGETS
    assert checksum_hashes == artifact_hashes
    assert all(path.startswith(f"v{project_version}/") for path in artifact_hashes)
    assert all(artifact["build_number"] == 1 for artifact in artifacts)
    assert all(f"-{project_version}-1-" in Path(path).name for path in artifact_hashes)


def test_go_repository_is_bilingual_and_documents_source_first_installation() -> None:
    manifest = json.loads(
        (GO_BINDING / "repository-manifest.json").read_text(encoding="utf-8")
    )
    english = (GO_BINDING / "repository-index.html").read_text(encoding="utf-8")
    chinese = (GO_BINDING / "repository-index.zh-CN.html").read_text(encoding="utf-8")

    for page in (english, chinese):
        assert all(artifact["path"] in page for artifact in manifest["artifacts"])
        assert "go get github.com/adamdeprince/stride-align/bindings/go@main" in page
        assert "cmd/stridealign-prebuilt@main" in page
        assert "C++23" in page
    assert 'href="index.zh-CN.html"' in english
    assert 'href="index.html"' in chinese
    assert "Keep the Go API" in english
    assert "保留 Go API" in chinese


def test_go_source_and_prebuilt_modes_have_separate_native_builds() -> None:
    native = (GO_BINDING / "native.go").read_text(encoding="utf-8")
    prebuilt = (GO_BINDING / "prebuilt.go").read_text(encoding="utf-8")
    bridge = (GO_BINDING / "bridge.cpp").read_text(encoding="utf-8")
    beider_morse = (GO_BINDING / "beider_morse.cpp").read_text(encoding="utf-8")

    assert "-std=c++23" in native
    assert "!stridealign_prebuilt CXXFLAGS" in native
    assert "pkg-config: stride-align-go" in prebuilt
    assert bridge.startswith("//go:build !stridealign_prebuilt\n")
    assert beider_morse.startswith("//go:build !stridealign_prebuilt\n")
