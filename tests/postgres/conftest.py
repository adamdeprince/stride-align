from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
POSTGRES_BINDING = ROOT / "bindings" / "postgres"


def _postgres_major(pg_config: Path) -> int | None:
    try:
        output = subprocess.check_output(
            [str(pg_config), "--version"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"PostgreSQL (\d+)", output)
    return int(match.group(1)) if match else None


def _candidate_configs() -> dict[int, Path]:
    candidates: list[Path] = []
    configured = os.environ.get("STRIDE_ALIGN_PG_CONFIGS")
    if configured:
        candidates.extend(Path(value) for value in configured.split(os.pathsep) if value)
    else:
        for major in (16, 17, 18):
            candidates.extend(
                [
                    Path(f"/opt/homebrew/opt/postgresql@{major}/bin/pg_config"),
                    Path(f"/usr/local/opt/postgresql@{major}/bin/pg_config"),
                    Path(f"/usr/lib/postgresql/{major}/bin/pg_config"),
                ]
            )
        found_on_path = shutil.which("pg_config")
        if found_on_path:
            candidates.append(Path(found_on_path))

    output: dict[int, Path] = {}
    for candidate in candidates:
        if not candidate.is_file():
            continue
        major = _postgres_major(candidate)
        if major in (16, 17, 18):
            output.setdefault(major, candidate.resolve())
    return output


PG_CONFIGS = _candidate_configs()


@dataclass
class PostgresServer:
    major: int
    pg_config: Path
    data_directory: Path
    socket_directory: Path
    port: int

    @property
    def bin_directory(self) -> Path:
        return self.pg_config.parent

    def command(
        self,
        sql: str,
        *,
        database: str = "postgres",
        tuples_only: bool = True,
    ) -> str:
        command = [
            str(self.bin_directory / "psql"),
            "-X",
            "-v",
            "ON_ERROR_STOP=1",
            "-h",
            str(self.socket_directory),
            "-p",
            str(self.port),
            "-d",
            database,
        ]
        if tuples_only:
            command.extend(["-A", "-t"])
        completed = subprocess.run(
            command,
            input=sql,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise AssertionError(f"PostgreSQL {self.major} SQL failed:\n{sql}\n{completed.stderr}")
        return completed.stdout.strip()

    def create_database(self, name: str, encoding: str) -> None:
        createdb = self.bin_directory / "createdb"
        completed = subprocess.run(
            [
                str(createdb),
                "-h",
                str(self.socket_directory),
                "-p",
                str(self.port),
                "-T",
                "template0",
                "--locale=C",
                "-E",
                encoding,
                name,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise AssertionError(
                f"could not create PostgreSQL {self.major} {encoding} database: {completed.stderr}"
            )
        self.command("CREATE EXTENSION stride_align;", database=name)


def _run(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\n{completed.stdout}\n{completed.stderr}"
        )


@pytest.fixture(
    scope="module",
    params=sorted(PG_CONFIGS.items()),
    ids=lambda item: f"pg{item[0]}",
)
def postgres_server(request: pytest.FixtureRequest) -> Iterator[PostgresServer]:
    if not PG_CONFIGS:
        pytest.skip(
            "PostgreSQL 16, 17, or 18 development tools were not found; set STRIDE_ALIGN_PG_CONFIGS"
        )
    major, pg_config = request.param

    _run([sys.executable, "generate_sql.py"], cwd=POSTGRES_BINDING)
    _run(["make", "clean", f"PG_CONFIG={pg_config}"], cwd=POSTGRES_BINDING)
    _run(
        [
            "make",
            f"PG_CONFIG={pg_config}",
            "STRIDE_ALIGN_POSTGRES_SIMD=native",
            f"-j{max(1, min(os.cpu_count() or 1, 8))}",
        ],
        cwd=POSTGRES_BINDING,
    )
    try:
        _run(
            ["make", f"PG_CONFIG={pg_config}", "install"],
            cwd=POSTGRES_BINDING,
        )
    except RuntimeError as error:
        pytest.skip(f"PostgreSQL {major} extension install directory is not writable: {error}")

    data_directory = Path(tempfile.mkdtemp(prefix=f"stride-pg{major}-", dir="/tmp"))
    socket_directory = Path(tempfile.mkdtemp(prefix=f"stride-pg{major}-sock-", dir="/tmp"))
    log_path = data_directory / "server.log"
    bin_directory = pg_config.parent
    _run(
        [
            str(bin_directory / "initdb"),
            "-D",
            str(data_directory),
            "-A",
            "trust",
            "--no-locale",
            "-E",
            "UTF8",
        ],
        cwd=ROOT,
    )
    port = 55800 + major
    pg_ctl = bin_directory / "pg_ctl"
    _run(
        [
            str(pg_ctl),
            "-D",
            str(data_directory),
            "-l",
            str(log_path),
            "-o",
            f"-p {port} -k {socket_directory} -c listen_addresses=''",
            "-w",
            "start",
        ],
        cwd=ROOT,
    )
    server = PostgresServer(major, pg_config, data_directory, socket_directory, port)
    try:
        server.command("CREATE EXTENSION stride_align;")
        yield server
    finally:
        subprocess.run(
            [str(pg_ctl), "-D", str(data_directory), "-m", "fast", "-w", "stop"],
            text=True,
            capture_output=True,
            check=False,
        )
        shutil.rmtree(data_directory, ignore_errors=True)
        shutil.rmtree(socket_directory, ignore_errors=True)
