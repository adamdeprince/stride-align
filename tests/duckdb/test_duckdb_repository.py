from __future__ import annotations

import json
import re
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

    repository_indexes = (
        (DUCKDB_BINDING / "repository-index.html").read_text(encoding="utf-8"),
        (DUCKDB_BINDING / "repository-index.zh-CN.html").read_text(encoding="utf-8"),
    )
    homepage = (ROOT / "html" / "index.html").read_text(encoding="utf-8")
    assert manifest["project"] == "stride-align"
    for repository_index in repository_indexes:
        assert expected_filename in repository_index
        assert all(path in repository_index for path in artifact_paths)
    assert expected_filename in homepage


def test_planned_power_and_loongson_packages_are_pending_not_downloads() -> None:
    indexes = {
        "en": (DUCKDB_BINDING / "repository-index.html").read_text(encoding="utf-8"),
        "zh-CN": (DUCKDB_BINDING / "repository-index.zh-CN.html").read_text(encoding="utf-8"),
    }
    pending_profiles = (
        "POWER8+ / VSX",
        "LA464 / LSX",
        "LA464 / LASX",
        "LA664 / LASX",
    )

    for language, page in indexes.items():
        pending_rows = re.findall(r'<tr class="pending-row">(.*?)</tr>', page, flags=re.DOTALL)

        assert len(pending_rows) == len(pending_profiles)
        assert all(any(profile in row for row in pending_rows) for profile in pending_profiles)
        assert all("<a " not in row for row in pending_rows)
        assert page.count('class="download"') == 4
        assert f'<html lang="{language}">' in page

    assert indexes["en"].count('<span class="pending">Pending</span>') == 4
    assert indexes["zh-CN"].count('<span class="pending">待发布</span>') == 4
    assert 'href="index.zh-CN.html"' in indexes["en"]
    assert 'href="index.html"' in indexes["zh-CN"]
