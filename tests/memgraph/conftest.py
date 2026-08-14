from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MEMGRAPH_BINDING = ROOT / "bindings" / "memgraph"
DEFAULT_IMAGE = "docker.io/memgraph/memgraph:3.10.1"


def _runtime() -> str | None:
    configured = os.environ.get("STRIDE_ALIGN_MEMGRAPH_CONTAINER_RUNTIME")
    if configured:
        return shutil.which(configured) or configured
    return shutil.which("podman") or shutil.which("docker")


def _run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _cypher_scalar(value: str):
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith('"') and value.endswith('"'):
        return json.loads(value)
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


@dataclass
class MemgraphServer:
    runtime: str
    container: str

    def query(self, cypher: str) -> list[list[object]]:
        completed = _run(
            [
                self.runtime,
                "exec",
                "-i",
                self.container,
                "mgconsole",
                "--host",
                "127.0.0.1",
                "--port",
                "7687",
                "--output-format=csv",
            ],
            input_text=cypher.rstrip(";\n") + ";\n",
        )
        rows = list(csv.reader(completed.stdout.splitlines()))
        if not rows:
            raise AssertionError(f"Memgraph returned no CSV header for: {cypher}")
        return [[_cypher_scalar(value) for value in row] for row in rows[1:]]

    def query_fails(self, cypher: str) -> str:
        completed = _run(
            [
                self.runtime,
                "exec",
                "-i",
                self.container,
                "mgconsole",
                "--host",
                "127.0.0.1",
                "--port",
                "7687",
                "--output-format=csv",
            ],
            input_text=cypher.rstrip(";\n") + ";\n",
            check=False,
        )
        assert completed.returncode != 0, completed.stdout
        return completed.stderr + completed.stdout


@pytest.fixture(scope="session")
def memgraph_server() -> Iterator[MemgraphServer]:
    if platform.system() != "Linux":
        pytest.skip("Memgraph native query modules are Linux shared libraries")
    runtime = _runtime()
    if runtime is None:
        pytest.skip("podman or docker is required for Memgraph integration tests")
    for tool in ("cmake", "c++"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} is required for Memgraph integration tests")

    image = os.environ.get("STRIDE_ALIGN_MEMGRAPH_IMAGE", DEFAULT_IMAGE)
    inspected = _run([runtime, "image", "inspect", image], check=False)
    if inspected.returncode:
        _run([runtime, "pull", image])

    temporary = Path(tempfile.mkdtemp(prefix="stride-memgraph-"))
    container = "stride-align-test-" + uuid.uuid4().hex[:12]
    header_container = ""
    try:
        header_container = _run([runtime, "create", image]).stdout.strip()
        api_dir = temporary / "memgraph-api"
        api_dir.mkdir()
        _run(
            [
                runtime,
                "cp",
                f"{header_container}:/usr/include/memgraph/mg_procedure.h",
                str(api_dir / "mg_procedure.h"),
            ]
        )
        _run([runtime, "rm", header_container])
        header_container = ""

        build_dir = temporary / "build"
        configure = [
            "cmake",
            "-S",
            str(MEMGRAPH_BINDING),
            "-B",
            str(build_dir),
            "-DSTRIDE_ALIGN_MEMGRAPH_SIMD=native",
            f"-DSTRIDE_ALIGN_MEMGRAPH_API_DIR={api_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        if shutil.which("ninja"):
            configure.extend(["-G", "Ninja"])
        _run(configure)
        _run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "-j",
                str(max(1, min(os.cpu_count() or 1, 8))),
            ]
        )
        module = build_dir / "stride_align.so"
        assert module.is_file()

        _run(
            [
                runtime,
                "run",
                "--rm",
                "-d",
                "--name",
                container,
                "-v",
                f"{module}:/usr/lib/memgraph/query_modules/stride_align.so:ro",
                image,
            ]
        )
        server = MemgraphServer(runtime, container)
        deadline = time.monotonic() + 30.0
        while True:
            ready = _run(
                [
                    runtime,
                    "exec",
                    "-i",
                    container,
                    "mgconsole",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "7687",
                ],
                input_text="RETURN 1;\n",
                check=False,
            )
            if ready.returncode == 0:
                break
            if time.monotonic() >= deadline:
                logs = _run([runtime, "logs", container], check=False)
                raise RuntimeError(
                    f"Memgraph did not become ready:\n{logs.stdout}\n{logs.stderr}"
                )
            time.sleep(0.25)
        yield server
    finally:
        if header_container:
            _run([runtime, "rm", header_container], check=False)
        _run([runtime, "rm", "-f", container], check=False)
        shutil.rmtree(temporary, ignore_errors=True)
