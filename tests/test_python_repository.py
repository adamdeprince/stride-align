from __future__ import annotations

import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_REPOSITORY = ROOT / "bindings" / "python"
DOWNLOAD_BASE = "https://distribution.goblinreactor.com/stride-align/python/"


def test_python_repository_matches_project_version_and_checksums() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        project_version = str(tomllib.load(project_file)["project"]["version"])

    manifest = json.loads(
        (PYTHON_REPOSITORY / "repository-manifest.json").read_text(encoding="utf-8")
    )
    artifacts = manifest["artifacts"]
    artifact_paths = {artifact["path"] for artifact in artifacts}
    artifact_hashes = {artifact["path"]: artifact["sha256"] for artifact in artifacts}

    checksum_hashes = {
        path: digest
        for digest, path in (
            line.split(maxsplit=1)
            for line in (PYTHON_REPOSITORY / "repository-SHA256SUMS")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }

    assert manifest["project"] == "stride-align"
    assert manifest["project_version"] == project_version
    assert manifest["base_url"] == DOWNLOAD_BASE
    assert len(artifacts) == 9
    assert checksum_hashes == artifact_hashes
    assert all(path.startswith(f"v{project_version}/") for path in artifact_paths)
    assert all(f"-{project_version}-" in Path(path).name for path in artifact_paths)


def test_python_repository_lists_power_and_both_loongarch_worlds() -> None:
    manifest = json.loads(
        (PYTHON_REPOSITORY / "repository-manifest.json").read_text(encoding="utf-8")
    )
    artifacts = manifest["artifacts"]
    indexes = (
        (PYTHON_REPOSITORY / "repository-index.html").read_text(encoding="utf-8"),
        (PYTHON_REPOSITORY / "repository-index.zh-CN.html").read_text(encoding="utf-8"),
    )

    power = [artifact for artifact in artifacts if artifact["platform"] == "linux_ppc64le"]
    loongarch = [
        artifact for artifact in artifacts if artifact["platform"] == "linux_loongarch64"
    ]

    assert {artifact["python"] for artifact in power} == {"cp312", "cp313", "cp314"}
    assert {artifact["world"] for artifact in loongarch} == {"oldworld", "newworld"}
    assert len(loongarch) == 6
    assert all(artifact["validation"] == "runtime-tested" for artifact in artifacts)
    for index in indexes:
        assert all(artifact["path"] in index for artifact in artifacts)
        assert index.count('class="download"') == 9
    assert 'href="index.zh-CN.html"' in indexes[0]
    assert 'href="index.html"' in indexes[1]


def test_installation_docs_use_only_the_self_hosted_architecture_wheels() -> None:
    documentation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "README.md",
            ROOT / "README.zh-CN.md",
            ROOT / "html" / "index.html",
            ROOT / "website" / "index.zh-CN.html",
        )
    )

    assert DOWNLOAD_BASE in documentation
    assert "github.com/adamdeprince/stride-align/releases/download" not in documentation
    assert "stride-align.com/wheels/" not in documentation
