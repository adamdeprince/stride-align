#!/usr/bin/env python3
"""Generate build-local C++ data for DuckDB's self-contained extension.

The output is intentionally not tracked.  It is derived from the same Python
matrix objects and BMPM text resources used by the Python and R adapters, so
the three packages cannot silently drift to different catalogs.
"""

from __future__ import annotations

import argparse
import ast
import re
import struct
from pathlib import Path


def cpp_bytes(data: bytes) -> str:
    return "{" + ",".join(str(value) for value in data) + "}"


def cpp_string(value: str) -> str:
    return (
        '"'
        + value.encode("unicode_escape").decode("ascii")
        .replace('"', '\\"')
        + '"'
    )


def signed(data: bytes) -> list[int]:
    return [value if value < 128 else value - 256 for value in data]


def parse_ncbi(path: Path) -> tuple[str, list[int]]:
    header: list[str] | None = None
    rows: list[list[int]] = []
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if header is None:
            header = tokens
            continue
        labels.append(tokens[0])
        rows.append([int(value) for value in tokens[1:]])
    if header is None or labels != header or any(len(row) != len(header) for row in rows):
        raise ValueError(f"invalid NCBI matrix: {path}")
    return "".join(header), [value for row in rows for value in row]


def gap_defaults(name: str) -> tuple[int, int, int]:
    if name.startswith("BLOSUM"):
        level = int(name.removeprefix("BLOSUM"))
        if level >= 80:
            return -5, -10, -1
        if level >= 62:
            return -5, -11, -1
        if level >= 50:
            return -6, -13, -2
        return -6, -14, -2
    if name.startswith("PAM"):
        level = int(name.removeprefix("PAM"))
        if level <= 30:
            return -5, -9, -1
        if level <= 120:
            return -6, -10, -1
        if level <= 200:
            return -7, -12, -1
        return -8, -14, -2
    return -5, -10, -2


def hand_written_matrices(source: str) -> dict[str, list[int]]:
    output: dict[str, list[int]] = {}
    for name in ("BLOSUM45", "BLOSUM50", "BLOSUM62", "BLOSUM80", "BLOSUM90",
                 "PAM30", "PAM70", "PAM250"):
        marker = f"_{name}_VALUES"
        start = source.index(marker)
        start = source.index("[", start)
        depth = 0
        end = start
        for end in range(start, len(source)):
            if source[end] == "[":
                depth += 1
            elif source[end] == "]":
                depth -= 1
                if depth == 0:
                    break
        literal = re.sub(r"#.*", "", source[start : end + 1])
        rows = ast.literal_eval(literal)
        output[name] = [value for row in rows for value in row]
    return output


def load_npy_int8(path: Path) -> bytes:
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"invalid NPY file: {path}")
        major, minor = stream.read(2)
        if (major, minor) == (1, 0):
            header_size = struct.unpack("<H", stream.read(2))[0]
        else:
            header_size = struct.unpack("<I", stream.read(4))[0]
        header = ast.literal_eval(stream.read(header_size).decode("latin-1"))
        if header["descr"] not in ("|i1", "<i1") or header["fortran_order"]:
            raise ValueError(f"expected a C-contiguous int8 NPY file: {path}")
        size = 1
        for dimension in header["shape"]:
            size *= dimension
        data = stream.read()
        if len(data) != size:
            raise ValueError(f"truncated NPY file: {path}")
        return data


def matrix_records(source_root: Path):
    module_path = source_root / "src" / "stride_align" / "matrices" / "__init__.py"
    handwritten = hand_written_matrices(module_path.read_text(encoding="utf-8"))
    protein_alphabet = "ARNDCQEGHILKMFPSTWYVBZX*"
    data_root = source_root / "src" / "stride_align" / "matrix_data"
    names = [f"BLOSUM{value}" for value in (30, 35, 40, 45, 50, 55, 60, 62, 65, 70, 75, 80, 85, 90, 100)]
    names += [f"PAM{value}" for value in range(10, 501, 10)]
    for name in names:
        if name in handwritten:
            alphabet, values = protein_alphabet, handwritten[name]
        else:
            alphabet, values = parse_ncbi(data_root / name)
        gap, gap_open, gap_extend = gap_defaults(name)
        # Preserve the public hand-written objects' historical linear gaps.
        gap = {"BLOSUM45": -5, "BLOSUM50": -5, "BLOSUM62": -4,
               "BLOSUM80": -8, "BLOSUM90": -6,
               "PAM30": -9, "PAM70": -10, "PAM250": -8}.get(name, gap)
        yield name.lower(), name, alphabet, "X", gap, gap_open, gap_extend, values
    alphabet, values = parse_ncbi(data_root / "NUC.4.4")
    yield "nuc44", "NUC.4.4", alphabet, "N", -5, -10, -2, values
    dna = [5 if row == column else -4 for row in range(5) for column in range(5)]
    wildcard = 4
    for index in range(5):
        dna[wildcard * 5 + index] = -4
        dna[index * 5 + wildcard] = -4
    yield "dna_match", "DNA_MATCH", "ACGTN", "N", -5, 0, 0, dna
    ascii_values = [1 if row == column else -1 for row in range(128) for column in range(128)]
    for index in range(128):
        ascii_values[127 * 128 + index] = -1
        ascii_values[index * 128 + 127] = -1
    yield "ascii_text", "ASCII", "".join(chr(value) for value in range(128)), chr(127), -1, 0, 0, ascii_values


def matrix_line(record) -> str:
    sql_name, name, alphabet, wildcard, gap, gap_open, gap_extend, values = record
    return "{" + ",".join(
        [cpp_string(sql_name), cpp_string(name), cpp_string(alphabet),
         cpp_string(wildcard), str(gap), str(gap_open), str(gap_extend),
         "true" if gap_open != 0 or gap_extend != 0 else "false",
         "{" + ",".join(str(value) for value in values) + "}"]
    ) + "},"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    lines = [
        "#pragma once",
        "#include <cstdint>",
        "#include <string>",
        "#include <unordered_map>",
        "#include <vector>",
        "namespace duckdb::stride_align_embedded {",
        "struct MatrixRecord {",
        "  const char* sql_name; const char* name; const char* alphabet;",
        "  const char* wildcard; std::int64_t gap_score;",
        "  std::int64_t gap_open; std::int64_t gap_extend;",
        "  bool has_affine; std::vector<std::int8_t> values;",
        "};",
        "inline std::vector<MatrixRecord> matrices() { return {",
    ]
    for record in matrix_records(source_root):
        lines.append(matrix_line(record))
    keyboard_path = (
        source_root / "src" / "stride_align" / "matrices" /
        "keyboard_data" / "qwerty.npy"
    )
    keyboard_values = signed(load_npy_int8(keyboard_path))
    ascii_alphabet = "".join(chr(value) for value in range(128))
    lines.append(matrix_line((
        "keyboard:qwerty", "keyboard:qwerty", ascii_alphabet, chr(127),
        -1, 0, 0, keyboard_values,
    )))
    lines.extend(
        [
            "}; }",
            "inline std::unordered_map<std::string, std::string> bmpm_resources() {",
            "  return {",
        ]
    )
    resource_root = source_root / "src" / "stride_align" / "bmpm_data"
    for path in sorted(resource_root.glob("*.txt")):
        data = path.read_bytes()
        symbol = "bmpm_" + re.sub(r"[^a-zA-Z0-9_]", "_", path.stem)
        lines.append(
            "[](){ static const unsigned char " + symbol + "[]=" +
            cpp_bytes(data) + "; return std::pair<std::string,std::string>{" +
            cpp_string(path.stem) + ",std::string(reinterpret_cast<const char*>(" +
            symbol + "),sizeof(" + symbol + "))}; }(),"
        )
    lines.extend(["}; }", "}  // namespace duckdb::stride_align_embedded", ""])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
