from __future__ import annotations

import os
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
with (_REPOSITORY_ROOT / "pyproject.toml").open("rb") as _project_file:
    _PROJECT_VERSION = str(tomllib.load(_project_file)["project"]["version"])

_EXTENSION_FILENAME = f"stride_align.{_PROJECT_VERSION}.duckdb_extension"


@dataclass(frozen=True)
class LoadedDuckDB:
    connection: Any
    extension_path: Path
    duckdb_version: str
    platform: str
    simd_level: str


def _explicit_extension_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _REPOSITORY_ROOT / path
    path = path.resolve()
    if not path.is_file():
        pytest.fail(
            f"STRIDE_ALIGN_DUCKDB_EXTENSION does not name a file: {path}",
            pytrace=False,
        )
    return path


def _discover_extension(duckdb_version: str, platform: str) -> Path:
    explicit = os.environ.get("STRIDE_ALIGN_DUCKDB_EXTENSION")
    if explicit:
        return _explicit_extension_path(explicit)

    version_tag = duckdb_version if duckdb_version.startswith("v") else f"v{duckdb_version}"
    package_root = _REPOSITORY_ROOT / "dist" / "duckdb" / version_tag / platform
    requested_simd = os.environ.get("STRIDE_ALIGN_DUCKDB_SIMD")
    if requested_simd:
        requested = package_root / requested_simd / _EXTENSION_FILENAME
        if not requested.is_file():
            pytest.fail(
                f"STRIDE_ALIGN_DUCKDB_SIMD selected a missing extension artifact: {requested}",
                pytrace=False,
            )
        return requested.resolve()

    candidates = sorted(package_root.glob(f"*/{_EXTENSION_FILENAME}"))
    if not candidates:
        pytest.skip(
            "no sideloaded stride-align extension for "
            f"DuckDB {duckdb_version} on {platform}; expected one below "
            f"{package_root}, or set STRIDE_ALIGN_DUCKDB_EXTENSION"
        )

    # Never guess an aggressive x86 ISA. Generic is universally safe, NEON is
    # the AArch64 baseline, and a native artifact was built on this machine.
    preferences = ["generic"]
    normalized_platform = platform.lower()
    if "arm64" in normalized_platform or "aarch64" in normalized_platform:
        preferences.append("neon")
    preferences.append("native")
    by_profile = {candidate.parent.name: candidate for candidate in candidates}
    for profile in preferences:
        if profile in by_profile:
            return by_profile[profile].resolve()

    profiles = ", ".join(candidate.parent.name for candidate in candidates)
    pytest.skip(
        "only CPU-specific DuckDB extension artifacts were found "
        f"({profiles}); set STRIDE_ALIGN_DUCKDB_SIMD after choosing a compatible profile"
    )


@pytest.fixture(scope="session")
def duckdb_extension() -> Iterator[LoadedDuckDB]:
    duckdb = pytest.importorskip(
        "duckdb",
        reason="install the pinned DuckDB test dependency with the project's dev extra",
    )
    connection = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    try:
        platform = str(connection.execute("PRAGMA platform").fetchone()[0])
        extension_path = _discover_extension(duckdb.__version__, platform)
        escaped_path = str(extension_path).replace("'", "''")
        try:
            # DuckDB does not accept a parameter placeholder in LOAD, so quote
            # the already-resolved local path as a SQL string literal.
            connection.execute(f"LOAD '{escaped_path}'")
            simd_level = str(connection.execute("SELECT stride_align_simd_level()").fetchone()[0])
        except Exception as error:
            pytest.fail(
                f"failed to sideload {extension_path} into DuckDB "
                f"{duckdb.__version__} ({platform}): {error}",
                pytrace=False,
            )

        yield LoadedDuckDB(
            connection=connection,
            extension_path=extension_path,
            duckdb_version=duckdb.__version__,
            platform=platform,
            simd_level=simd_level,
        )
    finally:
        connection.close()
