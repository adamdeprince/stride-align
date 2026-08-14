#!/usr/bin/env python3
"""Build one relocatable precompiled backend for the Go package.

The normal Go package builds the shared C++ source through cgo.  These archives
are an explicit fallback for systems whose C++ compiler cannot compile C++23.
They deliberately contain only the native bridge: the public Go API continues
to come from the versioned Go module.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GO_BINDING = ROOT / "bindings" / "go"


@dataclass(frozen=True)
class Target:
    system: str
    machine: tuple[str, ...]
    platform: str
    profile: str
    flags: tuple[str, ...]
    runtime_libraries: tuple[str, ...]
    minimum_os: str | None = None

    @property
    def key(self) -> str:
        return f"{self.platform}/{self.profile}"


TARGETS = {
    target.key: target
    for target in (
        Target(
            "darwin",
            ("arm64", "aarch64"),
            "darwin_arm64",
            "neon",
            ("-DSTRIDE_ALIGN_GO_TARGET_NEON=1", "-mcpu=apple-m1", "-mmacosx-version-min=13.0"),
            ("-lc++",),
            "macOS 13.0",
        ),
        Target(
            "linux",
            ("x86_64", "amd64"),
            "linux_amd64",
            "generic",
            (
                "-DSTRIDE_ALIGN_GO_TARGET_GENERIC=1",
                "-march=x86-64",
                "-mtune=generic",
                "-msse2",
                "-mno-avx",
                "-mno-avx2",
                "-mno-avx512f",
            ),
            ("-lstdc++", "-lm"),
        ),
        Target(
            "linux",
            ("x86_64", "amd64"),
            "linux_amd64",
            "avx2",
            ("-DSTRIDE_ALIGN_GO_TARGET_AVX2=1", "-march=x86-64-v3", "-mtune=generic"),
            ("-lstdc++", "-lm"),
        ),
        Target(
            "linux",
            ("x86_64", "amd64"),
            "linux_amd64",
            "avx512bwvl",
            (
                "-DSTRIDE_ALIGN_GO_TARGET_AVX512BWVL=1",
                "-march=x86-64-v4",
                "-mtune=generic",
            ),
            ("-lstdc++", "-lm"),
        ),
        Target(
            "linux",
            ("aarch64", "arm64"),
            "linux_arm64",
            "neon",
            (
                "-DSTRIDE_ALIGN_GO_TARGET_NEON=1",
                "-march=armv8-a+simd",
                "-mtune=generic",
            ),
            ("-lstdc++", "-lm"),
        ),
        Target(
            "linux",
            ("ppc64le",),
            "linux_ppc64le",
            "power8_vsx",
            (
                "-DSTRIDE_ALIGN_GO_TARGET_POWER8_VSX=1",
                "-mcpu=power8",
                "-mtune=power8",
                "-mvsx",
            ),
            ("-lstdc++", "-lm"),
        ),
        Target(
            "linux",
            ("loongarch64", "loong64"),
            "linux_loong64",
            "la464_lsx",
            (
                "-DSTRIDE_ALIGN_GO_TARGET_LSX=1",
                "-DSTRIDE_ALIGN_GO_TARGET_LA464=1",
                "-march=la464",
                "-mtune=la464",
                "-mlsx",
                "-mno-lasx",
            ),
            ("-lstdc++", "-lm"),
        ),
        Target(
            "linux",
            ("loongarch64", "loong64"),
            "linux_loong64",
            "la464_lasx",
            (
                "-DSTRIDE_ALIGN_GO_TARGET_LASX=1",
                "-DSTRIDE_ALIGN_GO_TARGET_LA464=1",
                "-march=la464",
                "-mtune=la464",
                "-mlsx",
                "-mlasx",
            ),
            ("-lstdc++", "-lm"),
        ),
        Target(
            "linux",
            ("loongarch64", "loong64"),
            "linux_loong64",
            "la664_lasx",
            (
                "-DSTRIDE_ALIGN_GO_TARGET_LASX=1",
                "-DSTRIDE_ALIGN_GO_TARGET_LA664=1",
                "-march=la664",
                "-mtune=la664",
                "-mlsx",
                "-mlasx",
            ),
            ("-lstdc++", "-lm"),
        ),
    )
}


def project_version() -> str:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', project, re.MULTILINE)
    if match is None:
        raise SystemExit("could not read the stride-align version from pyproject.toml")
    return match.group(1)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def compiler_identity(compiler: str) -> str:
    result = subprocess.run(
        [compiler, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()[0]


def verify_host(target: Target) -> None:
    host_system = platform.system().lower()
    host_machine = platform.machine().lower()
    if host_system != target.system or host_machine not in target.machine:
        expected = f"{target.system}/{' or '.join(target.machine)}"
        raise SystemExit(
            f"{target.key} must be built on {expected}; this host is "
            f"{host_system}/{host_machine}"
        )


def write_pkg_config(
    path: Path,
    target: Target,
    version: str,
    runtime_libraries: tuple[str, ...],
) -> None:
    libraries = " ".join(runtime_libraries)
    path.write_text(
        "\n".join(
            (
                "prefix=${pcfiledir}/../..",
                "libdir=${prefix}/lib",
                "",
                "Name: stride-align-go",
                "Description: Precompiled stride-align C++23 backend for Go",
                f"Version: {version}",
                f"Libs: -L${{libdir}} -lstride_align_go {libraries}",
                "Cflags:",
                "",
            )
        ),
        encoding="utf-8",
    )


def compiler_file(compiler: str, name: str) -> Path:
    result = subprocess.run(
        [compiler, f"-print-file-name={name}"],
        check=True,
        capture_output=True,
        text=True,
    )
    path = Path(result.stdout.strip())
    if path.name == name and not path.is_absolute():
        raise SystemExit(f"{compiler} could not locate {name}")
    if not path.is_file():
        raise SystemExit(f"{compiler} reported a missing {name}: {path}")
    return path.resolve()


def bundle_linux_runtime(compiler: str, staging: Path) -> tuple[str, ...]:
    libdir = staging / "lib"
    for library_name in ("libstdc++.a", "libgcc_eh.a"):
        shutil.copy2(compiler_file(compiler, library_name), libdir / library_name)

    license_source = Path("/usr/share/licenses/libgcc")
    if not license_source.is_dir():
        raise SystemExit(
            "static C++ runtime requested, but /usr/share/licenses/libgcc is missing; "
            "build in the documented manylinux container or supply the runtime licenses"
        )
    license_destination = staging / "licenses" / "gcc-runtime"
    license_destination.mkdir(parents=True)
    copied = 0
    for source in sorted(license_source.glob("COPYING*")):
        if source.is_file():
            shutil.copy2(source, license_destination / source.name)
            copied += 1
    if copied == 0:
        raise SystemExit(f"no GCC runtime licenses found under {license_source}")
    return ("-lstdc++", "-lgcc_eh", "-lm", "-ldl", "-lpthread")


def deterministic_tar(source: Path, output: Path, root_name: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in sorted(source.rglob("*")):
                    relative = path.relative_to(source)
                    info = archive.gettarinfo(str(path), arcname=str(Path(root_name) / relative))
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    info.mtime = 0
                    if path.is_file():
                        with path.open("rb") as source_file:
                            archive.addfile(info, source_file)
                    else:
                        archive.addfile(info)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=sorted(TARGETS))
    parser.add_argument(
        "--build-number",
        type=int,
        default=1,
        help="artifact packaging revision within the lockstep project version",
    )
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--ar", default=os.environ.get("AR", "ar"))
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "go")
    parser.add_argument(
        "--linux-abi",
        help="document the Linux ABI baseline used by the build environment",
    )
    parser.add_argument(
        "--system-cxx-runtime",
        action="store_true",
        help="link the consumer's libstdc++ instead of bundling it (Linux only)",
    )
    parser.add_argument(
        "--skip-host-check",
        action="store_true",
        help="allow a configured cross-compiler to build a non-host target",
    )
    args = parser.parse_args()

    target = TARGETS[args.target]
    if args.build_number < 1:
        raise SystemExit("--build-number must be at least 1")
    if not args.skip_host_check:
        verify_host(target)
    compiler = shutil.which(args.cxx) or args.cxx
    archiver = shutil.which(args.ar) or args.ar
    version = project_version()
    relative_dir = Path(f"v{version}") / target.platform / target.profile
    destination = args.output.resolve() / relative_dir
    destination.mkdir(parents=True, exist_ok=True)
    artifact_name = (
        f"stride-align-go-{version}-{args.build_number}-"
        f"{target.platform}-{target.profile}.tar.gz"
    )
    artifact_path = destination / artifact_name

    common_flags = (
        "-std=c++23",
        "-O3",
        "-DNDEBUG",
        "-fPIC",
        "-fvisibility=hidden",
        f"-I{ROOT / 'include'}",
        f"-I{GO_BINDING}",
    )
    compiler_name = compiler_identity(compiler)

    build_root = args.output.resolve() / ".build"
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{target.profile}-", dir=build_root) as temporary:
        work = Path(temporary)
        objects: list[Path] = []
        for source_name in ("bridge.cpp", "beider_morse.cpp"):
            source = GO_BINDING / source_name
            output = work / f"{source.stem}.o"
            run([compiler, *common_flags, *target.flags, "-c", str(source), "-o", str(output)])
            objects.append(output)

        staging = work / "package"
        libdir = staging / "lib"
        pcdir = libdir / "pkgconfig"
        libdir.mkdir(parents=True)
        pcdir.mkdir(parents=True)
        library = libdir / "libstride_align_go.a"
        run([archiver, "rcs", str(library), *(str(item) for item in objects)])
        runtime_libraries = target.runtime_libraries
        runtime_mode = "system"
        if target.system == "linux" and not args.system_cxx_runtime:
            runtime_libraries = bundle_linux_runtime(compiler, staging)
            runtime_mode = "bundled-static"
        write_pkg_config(
            pcdir / "stride-align-go.pc",
            target,
            version,
            runtime_libraries,
        )
        shutil.copy2(ROOT / "LICENSE", staging / "LICENSE")
        shutil.copy2(ROOT / "NOTICE", staging / "NOTICE")

        metadata = {
            "schema_version": 1,
            "project": "stride-align",
            "project_version": version,
            "build_number": args.build_number,
            "platform": target.platform,
            "profile": target.profile,
            "compiler": compiler_name,
            "compiler_flags": [*common_flags[:5], *target.flags],
            "minimum_os": target.minimum_os,
            "linux_abi": args.linux_abi,
            "link_libraries": list(runtime_libraries),
            "cxx_runtime": runtime_mode,
        }
        (staging / "ARTIFACT.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        root_name = artifact_name[: -len(".tar.gz")]
        deterministic_tar(staging, artifact_path, root_name)

    digest = sha256(artifact_path)
    checksum_path = artifact_path.with_suffix(artifact_path.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {artifact_name}\n", encoding="utf-8")
    record = {
        "platform": target.platform,
        "profile": target.profile,
        "compiler": compiler_name,
        "build_number": args.build_number,
        "compiler_target": " ".join(target.flags),
        "cxx_runtime": runtime_mode,
        "linux_abi": args.linux_abi,
        "validation": "build-tested",
        "path": str(relative_dir / artifact_name),
        "sha256": digest,
        "size": artifact_path.stat().st_size,
    }
    (destination / "artifact-record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
