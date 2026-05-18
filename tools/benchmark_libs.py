"""Time stride-align against other SW / NW / Levenshtein Python libraries.

For each line on stdin, run the line as a query against every line in
--input-file using each available library and emit a CSV row of timings.

  --smith-waterman    : compares stride-align against the SW-capable libs
  --needleman-wunsch  : compares stride-align against the NW-capable libs
  --levenshtein       : compares stride-align against stringzilla

Libraries are imported lazily. If a library is not installed the script
prints a one-line `pip install ...` hint to stderr and skips that column
for the rest of the run.

Usage:
    cat queries.txt | tools/benchmark_libs.py \
        --input-file kjv.txt --needleman-wunsch > timings.csv
    cat words.txt   | tools/benchmark_libs.py \
        --input-file /usr/share/dict/words --levenshtein > lev-timings.csv

The script does not modify pyproject.toml. Install whichever libraries
you want included via pip.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from typing import Callable, Sequence


# Each entry: (name, pip_hint, supported_modes).
LIBRARIES = [
    ("stride_align",  "pip install stride-align",  {"sw", "nw", "lev"}),
    ("parasail",      "pip install parasail",      {"sw", "nw"}),
    ("ssw",           "pip install ssw-py",        {"sw"}),
    ("pyssw",         "pip install pyssw",         {"sw"}),
    ("string2string", "pip install string2string", {"sw", "nw"}),
    ("textdistance",  "pip install textdistance",  {"sw", "nw"}),
    ("swalign",       "pip install swalign",       {"sw"}),
    ("minineedle",    "pip install minineedle",    {"sw", "nw"}),
    ("pyalign",       "pip install pyalign",       {"sw", "nw"}),
    ("stringzilla",   "pip install stringzilla",   {"lev"}),
]


def _try_import(name: str, install_hint: str):
    try:
        return __import__(name)
    except ImportError as exc:
        print(
            f"# {name}: import failed ({exc.__class__.__name__}: {exc}).\n"
            f"#   install with: {install_hint}",
            file=sys.stderr,
        )
        return None


MODULES = {name: _try_import(name, hint) for name, hint, _modes in LIBRARIES}


def _time(fn: Callable[[], None]) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


# Each adapter takes (query, targets, mode) and returns elapsed seconds, or
# None if the library does not support the requested mode at runtime.
def run_stride_align(query: str, targets: Sequence[str], mode: str) -> float | None:
    sa = MODULES["stride_align"]
    if mode == "sw":
        return _time(lambda: sa.smith_waterman_normalized_scores(query, targets))
    if mode == "nw":
        return _time(lambda: sa.needleman_wunsch_normalized_scores(query, targets))
    return _time(lambda: sa.levenshtein_scores(query, targets))


def run_parasail(query: str, targets: Sequence[str], mode: str) -> float | None:
    parasail = MODULES["parasail"]
    matrix = parasail.matrix_create("ACDEFGHIKLMNPQRSTVWY", 2, -1)
    align_fn = parasail.sw_striped_16 if mode == "sw" else parasail.nw_striped_16
    gap_open, gap_extend = 1, 1
    return _time(lambda: [align_fn(query, t, gap_open, gap_extend, matrix) for t in targets])


def run_ssw(query: str, targets: Sequence[str], mode: str) -> float | None:
    if mode != "sw":
        return None
    ssw = MODULES["ssw"]
    Aligner = ssw.Aligner
    aligner = Aligner(query, match=2, mismatch=-1, gap_open=1, gap_extend=1)
    return _time(lambda: [aligner.align(t) for t in targets])


def run_pyssw(query: str, targets: Sequence[str], mode: str) -> float | None:
    if mode != "sw":
        return None
    pyssw = MODULES["pyssw"]
    if hasattr(pyssw, "Aligner"):
        aligner = pyssw.Aligner(query)
        return _time(lambda: [aligner.align(t) for t in targets])
    if hasattr(pyssw, "ssw_align"):
        return _time(lambda: [pyssw.ssw_align(query, t) for t in targets])
    print("# pyssw: no Aligner / ssw_align API found", file=sys.stderr)
    return None


def run_string2string(query: str, targets: Sequence[str], mode: str) -> float | None:
    from string2string.alignment import NeedlemanWunsch, SmithWaterman  # type: ignore
    aligner = SmithWaterman() if mode == "sw" else NeedlemanWunsch()
    return _time(lambda: [aligner.get_alignment(query, t) for t in targets])


def run_textdistance(query: str, targets: Sequence[str], mode: str) -> float | None:
    td = MODULES["textdistance"]
    algo = td.smith_waterman if mode == "sw" else td.needleman_wunsch
    return _time(lambda: [algo(query, t) for t in targets])


def run_swalign(query: str, targets: Sequence[str], mode: str) -> float | None:
    if mode != "sw":
        return None
    sw_lib = MODULES["swalign"]
    matrix = sw_lib.NucleotideScoringMatrix(match=2, mismatch=-1)
    sw = sw_lib.LocalAlignment(matrix, gap_penalty=-1, gap_extension_penalty=-1)
    return _time(lambda: [sw.align(query, t) for t in targets])


def run_minineedle(query: str, targets: Sequence[str], mode: str) -> float | None:
    if mode == "sw":
        from minineedle.smith import SmithWaterman  # type: ignore
        Cls = SmithWaterman
    else:
        from minineedle.needle import NeedlemanWunsch  # type: ignore
        Cls = NeedlemanWunsch

    def one(t: str) -> None:
        a = Cls(list(query), list(t))
        a.align()
        a.get_score()

    return _time(lambda: [one(t) for t in targets])


def run_pyalign(query: str, targets: Sequence[str], mode: str) -> float | None:
    pa = MODULES["pyalign"]
    if mode == "sw":
        solver = getattr(pa, "local_alignment", None) or getattr(pa, "smith_waterman", None)
    else:
        solver = getattr(pa, "global_alignment", None) or getattr(pa, "needleman_wunsch", None)
    if solver is None:
        print("# pyalign: no global/local_alignment entry point found", file=sys.stderr)
        return None
    return _time(lambda: [solver(query, t) for t in targets])


def run_stringzilla(query: str, targets: Sequence[str], mode: str) -> float | None:
    if mode != "lev":
        return None
    sz = MODULES["stringzilla"]
    # API has shifted across stringzilla versions. Try the documented
    # module-level helpers first, fall back to per-Str-instance methods.
    if hasattr(sz, "edit_distance"):
        return _time(lambda: [sz.edit_distance(query, t) for t in targets])
    if hasattr(sz, "levenshtein_distance"):
        return _time(lambda: [sz.levenshtein_distance(query, t) for t in targets])
    Str = getattr(sz, "Str", None)
    if Str is not None:
        q = Str(query)
        if hasattr(q, "edit_distance"):
            return _time(lambda: [q.edit_distance(Str(t)) for t in targets])
        if hasattr(q, "levenshtein_distance"):
            return _time(lambda: [q.levenshtein_distance(Str(t)) for t in targets])
    print("# stringzilla: no edit_distance / levenshtein_distance API found", file=sys.stderr)
    return None


ADAPTERS = {
    "stride_align":  run_stride_align,
    "parasail":      run_parasail,
    "ssw":           run_ssw,
    "pyssw":         run_pyssw,
    "string2string": run_string2string,
    "textdistance":  run_textdistance,
    "swalign":       run_swalign,
    "minineedle":    run_minineedle,
    "pyalign":       run_pyalign,
    "stringzilla":   run_stringzilla,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time SW / NW / Levenshtein libraries on a corpus loaded from a file.",
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="path to a file whose lines become the target corpus.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smith-waterman", action="store_true", help="run Smith-Waterman.")
    mode.add_argument("--needleman-wunsch", action="store_true", help="run Needleman-Wunsch.")
    mode.add_argument(
        "--levenshtein",
        action="store_true",
        help="run Levenshtein (compared against stringzilla).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smith_waterman:
        mode = "sw"
    elif args.needleman_wunsch:
        mode = "nw"
    else:
        mode = "lev"

    # Read corpus and queries as bytes — parasail's bindings hold the
    # input via latin-1 and can't accept arbitrary unicode. stride-align,
    # stringzilla, and the rest accept bytes too, so this gives every
    # library the same input and lets the user point at unicode-rich
    # corpora (e.g. demo/kjv.txt with its smart quotes).
    with open(args.input_file, "rb") as fh:
        targets = [line.rstrip(b"\n") for line in fh]

    # Drop libraries that failed to import or don't support this mode.
    active = [
        name for name, _hint, modes in LIBRARIES
        if MODULES[name] is not None and mode in modes
    ]

    writer = csv.writer(sys.stdout)
    writer.writerow(
        ["query", "query_length", "n_targets", "mode"] + [f"{name}_s" for name in active]
    )

    for raw_line in sys.stdin.buffer:
        query = raw_line.rstrip(b"\n")
        if not query:
            continue

        timings: list[str] = []
        for name in active:
            try:
                seconds = ADAPTERS[name](query, targets, mode)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"# {name}: runtime error on query of len {len(query)}: "
                    f"{exc.__class__.__name__}: {exc}",
                    file=sys.stderr,
                )
                seconds = None
            timings.append(f"{seconds:.6f}" if seconds is not None else "")

        writer.writerow([query, len(query), len(targets), mode] + timings)
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
