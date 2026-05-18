"""Compare stride-align's raw *_scores against a reference library.

For each line on stdin, run the line as a query against every line in
--input-file, then check stride-align's integer scores against:

  --smith-waterman    : parasail.sw_*
  --needleman-wunsch  : parasail.nw_*
  --levenshtein       : Levenshtein.distance (the python-Levenshtein package)

Edit weights for the SW / NW comparison are pinned so the two libraries
produce comparable integer scores:

  match    = +2
  mismatch = -1
  gap_open = gap_extend = 1   (penalty subtracted by parasail; the
                               equivalent gap_score on the stride-align
                               side is -1)

The script tests against `stride_align.*_scores` (the integer variants,
not `*_normalized_scores`).

CSV columns per query line:

  query, query_length, n_targets, mode, reference, n_correct,
  n_mismatched, first_mismatch_index, ours_at_mismatch, ref_at_mismatch

If a reference library is missing, the script prints a one-line
`pip install ...` hint to stderr and exits non-zero before consuming
stdin.
"""

from __future__ import annotations

import argparse
import csv
import sys
from typing import Iterable, Sequence

import numpy as np


MATCH_SCORE = 2
MISMATCH_SCORE = -1
GAP_SCORE = -1            # stride-align side: negative bonus.
PARASAIL_GAP_PENALTY = 1  # parasail side: positive penalty.


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


def _import_stride_align():
    sa = _try_import("stride_align", "pip install stride-align")
    if sa is None:
        sys.exit(1)
    return sa


def _import_parasail():
    parasail = _try_import("parasail", "pip install parasail")
    if parasail is None:
        sys.exit(1)
    return parasail


def _import_levenshtein():
    lev = _try_import("Levenshtein", "pip install Levenshtein")
    if lev is None:
        sys.exit(1)
    return lev


def _build_full_byte_matrix(parasail):
    """Build a parasail substitution matrix over a 255-byte alphabet.

    parasail's matrix_create silently produces an all-zero matrix when
    the alphabet has 256 entries (its internal index reserves one slot
    for the "unknown character" sentinel). We use bytes 1..255 — the
    NUL byte is skipped, which is fine for line-oriented text input.
    Any character in the input therefore has a defined +2 / -1 score
    against any other character, matching stride-align's substitution
    model.
    """
    alphabet = bytes(range(1, 256)).decode("latin-1")
    return parasail.matrix_create(alphabet, MATCH_SCORE, MISMATCH_SCORE)


def _parasail_align_fn(parasail, mode: str):
    # The striped 16-bit variants can disagree with the reference scalar /
    # diagonal SW path on inputs that fit cleanly inside int16 (observed
    # 18 vs 16 on 'blessed are the' / 'and god blessed the' with this
    # very matrix). The scan variants run the same DP recurrence as the
    # textbook algorithm and matched the scalar `sw` and `sw_diag_16`
    # results in spot checks, so they are the appropriate reference for
    # correctness work.
    return parasail.sw_scan_16 if mode == "sw" else parasail.nw_scan_16


def _parasail_scores(parasail, matrix, mode: str, query: str, targets: Sequence[str]) -> np.ndarray:
    align = _parasail_align_fn(parasail, mode)
    out = np.empty(len(targets), dtype=np.int64)
    for i, t in enumerate(targets):
        r = align(query, t, PARASAIL_GAP_PENALTY, PARASAIL_GAP_PENALTY, matrix)
        out[i] = int(r.score)
    return out


def _stride_align_scores(sa, mode: str, query: str, targets: Sequence[str]) -> np.ndarray:
    if mode == "sw":
        return sa.smith_waterman_scores(
            query,
            targets,
            match_score=MATCH_SCORE,
            mismatch_score=MISMATCH_SCORE,
            gap_score=GAP_SCORE,
        )
    return sa.needleman_wunsch_scores(
        query,
        targets,
        match_score=MATCH_SCORE,
        mismatch_score=MISMATCH_SCORE,
        gap_score=GAP_SCORE,
    )


def _levenshtein_scores(lev, query, targets: Sequence) -> np.ndarray:
    return np.fromiter(
        (lev.distance(query, t) for t in targets),
        dtype=np.int64,
        count=len(targets),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare stride-align's integer *_scores output against a reference library.",
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="path to a file whose lines become the target corpus.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smith-waterman", action="store_true",
                      help="compare smith_waterman_scores against parasail.sw_striped_16.")
    mode.add_argument("--needleman-wunsch", action="store_true",
                      help="compare needleman_wunsch_scores against parasail.nw_striped_16.")
    mode.add_argument("--levenshtein", action="store_true",
                      help="compare levenshtein_scores against Levenshtein.distance.")
    parser.add_argument(
        "--max-diff-rows",
        type=int,
        default=5,
        help="how many mismatched rows to print to stderr per query (default 5).",
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

    sa = _import_stride_align()

    parasail = None
    matrix = None
    lev_lib = None
    reference_name = ""
    if mode in {"sw", "nw"}:
        parasail = _import_parasail()
        matrix = _build_full_byte_matrix(parasail)
        reference_name = "parasail"
    else:
        lev_lib = _import_levenshtein()
        reference_name = "Levenshtein"

    # We operate at the byte level: stride-align and parasail's matrix
    # both index by byte value, so encoding str inputs to UTF-8 bytes
    # gives a consistent definition of "a character" for the comparison.
    # This loses unicode-codepoint semantics but the alternative is that
    # parasail (latin-1 only at the binding layer) refuses non-latin-1
    # input outright.
    #
    # Additional wrinkle for SW/NW: parasail.matrix_create silently folds
    # uppercase A-Z onto lowercase a-z in its internal mapper, regardless
    # of the alphabet string we pass in (this is baked into the C library
    # and the mapper is read-only at the Python layer). To make
    # stride-align's byte-sensitive scoring comparable, we lowercase
    # everything in SW/NW mode so neither side sees a case difference.
    # Levenshtein mode (stringzilla reference) is case-sensitive on both
    # sides, so we keep the bytes as-is.
    case_fold = mode in {"sw", "nw"}

    with open(args.input_file, "rb") as fh:
        targets = [line.rstrip(b"\n") for line in fh]
    if case_fold:
        targets = [t.lower() for t in targets]

    writer = csv.writer(sys.stdout)
    writer.writerow([
        "query", "query_length", "n_targets", "mode", "reference",
        "n_correct", "n_mismatched",
        "first_mismatch_index", "ours_at_mismatch", "ref_at_mismatch",
    ])

    total_mismatches = 0
    total_rows = 0

    for raw_line in sys.stdin.buffer:
        query = raw_line.rstrip(b"\n")
        if not query:
            continue
        if case_fold:
            query = query.lower()

        if mode in {"sw", "nw"}:
            ours = _stride_align_scores(sa, mode, query, targets)
            ref = _parasail_scores(parasail, matrix, mode, query, targets)
        else:
            ours = sa.levenshtein_scores(query, targets)
            ref = _levenshtein_scores(lev_lib, query, targets)

        diff_idx = np.where(ours != ref)[0]
        n_mismatch = int(diff_idx.size)
        n_correct = int(len(targets) - n_mismatch)
        first_idx = int(diff_idx[0]) if n_mismatch else -1
        ours_at = int(ours[first_idx]) if first_idx >= 0 else 0
        ref_at = int(ref[first_idx]) if first_idx >= 0 else 0

        writer.writerow([
            query, len(query), len(targets), mode, reference_name,
            n_correct, n_mismatch, first_idx, ours_at, ref_at,
        ])
        sys.stdout.flush()

        total_rows += 1
        total_mismatches += n_mismatch

        if n_mismatch and args.max_diff_rows > 0:
            shown = 0
            for idx in diff_idx:
                if shown >= args.max_diff_rows:
                    break
                shown += 1
                print(
                    f"# mismatch query[:32]={query[:32]!r} target[{idx}][:32]="
                    f"{targets[int(idx)][:32]!r}  ours={int(ours[idx])} "
                    f"{reference_name}={int(ref[idx])}",
                    file=sys.stderr,
                )

    print(
        f"# summary: rows={total_rows} mismatches={total_mismatches} "
        f"mode={mode} reference={reference_name}",
        file=sys.stderr,
    )
    return 1 if total_mismatches > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
