from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEMGRAPH_BINDING = ROOT / "bindings" / "memgraph"
DOWNLOAD_BASE = "https://distribution.goblinreactor.com/stride-align/memgraph/"


def test_memgraph_repository_metadata_matches_lockstep_version() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        project_version = str(tomllib.load(project_file)["project"]["version"])

    manifest = json.loads(
        (MEMGRAPH_BINDING / "repository-manifest.json").read_text(encoding="utf-8")
    )
    artifacts = manifest["artifacts"]
    artifact_paths = {artifact["path"] for artifact in artifacts}
    artifact_hashes = {artifact["path"]: artifact["sha256"] for artifact in artifacts}
    checksum_hashes = {
        path: digest
        for digest, path in (
            line.split(maxsplit=1)
            for line in (MEMGRAPH_BINDING / "repository-SHA256SUMS")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }

    assert manifest["project"] == "stride-align"
    assert manifest["project_version"] == project_version
    assert manifest["memgraph_version"] == "3.10.1"
    assert manifest["linux_abi"] == "manylinux_2_28"
    assert manifest["base_url"] == DOWNLOAD_BASE
    assert checksum_hashes == artifact_hashes
    assert all(path.startswith("v3.10.1/") for path in artifact_paths)
    assert all(Path(path).name == f"stride_align-{project_version}.so" for path in artifact_paths)


def test_memgraph_repository_lists_bilingual_mac_and_x86_packages() -> None:
    manifest = json.loads(
        (MEMGRAPH_BINDING / "repository-manifest.json").read_text(encoding="utf-8")
    )
    artifacts = manifest["artifacts"]
    indexes = (
        (MEMGRAPH_BINDING / "repository-index.html").read_text(encoding="utf-8"),
        (MEMGRAPH_BINDING / "repository-index.zh-CN.html").read_text(encoding="utf-8"),
    )

    assert {(artifact["platform"], artifact["profile"]) for artifact in artifacts} == {
        ("osx_arm64", "neon"),
        ("linux_amd64", "generic"),
        ("linux_amd64", "avx2"),
    }
    assert {
        artifact["validation"] for artifact in artifacts if artifact["platform"] == "linux_amd64"
    } == {"runtime-tested"}
    for index in indexes:
        assert all(artifact["path"] in index for artifact in artifacts)
        assert index.count('class="download"') == 3
        assert "stride_align.so" in index
        assert "manylinux_2_28" in index
        assert "https://stride-align.com/bindings/memgraph/" in index
        assert "README.html" not in index
    assert 'href="index.zh-CN.html"' in indexes[0]
    assert 'href="index.html"' in indexes[1]
