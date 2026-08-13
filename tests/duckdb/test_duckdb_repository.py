from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DUCKDB_BINDING = ROOT / "bindings" / "duckdb"


def test_duckdb_repository_metadata_matches_project_version() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        project_version = str(tomllib.load(project_file)["project"]["version"])

    manifest = json.loads((DUCKDB_BINDING / "repository-manifest.json").read_text(encoding="utf-8"))
    expected_filename = f"stride_align.{project_version}.duckdb_extension"
    artifact_paths = {artifact["path"] for artifact in manifest["artifacts"]}

    assert manifest["project_version"] == project_version
    assert artifact_paths
    assert all(Path(path).name == expected_filename for path in artifact_paths)

    checksum_paths = {
        line.split(maxsplit=1)[1]
        for line in (DUCKDB_BINDING / "repository-SHA256SUMS")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    assert checksum_paths == artifact_paths

    repository_index = (DUCKDB_BINDING / "repository-index.html").read_text(encoding="utf-8")
    homepage = (ROOT / "html" / "index.html").read_text(encoding="utf-8")
    assert manifest["project"] == "stride-align"
    assert expected_filename in repository_index
    assert expected_filename in homepage
