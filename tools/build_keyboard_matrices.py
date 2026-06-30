#!/usr/bin/env python3
"""Build keyboard typo / confusion substitution matrices from the Aalto
"136 Million Keystrokes" dataset.

Data source
===========

This script derives substitution matrices from:

    Vivek Dhakal, Anna Maria Feit, Per Ola Kristensson, Antti Oulasvirta.
    "Observations on Typing from 136 Million Keystrokes."
    Proceedings of the 2018 CHI Conference on Human Factors in Computing
    Systems (CHI '18), 2018.  https://doi.org/10.1145/3173574.3174220
    Dataset: https://userinterfaces.aalto.fi/136Mkeystrokes/

The dataset records ~136 million keystrokes from ~168 000 volunteers,
including raw keystroke logs with timestamps and key/letter columns. Its
public terms allow research / non-commercial use with attribution; the
dataset authors additionally granted stride-align explicit permission to
build and publish a small set of derived scoring matrices under its
Apache-2.0 licence. Only the derived matrices are published — the raw
keystroke data is never redistributed. See NOTICE and
``docs/keyboard-matrix-external-sources.md``.

What we extract
===============

We scan each participant's keystroke log for *self-corrections*: the pattern
"typed X, pressed Backspace, typed Y" (X != Y). Each such event is a real,
human confusion of X for Y. We tally those events into per-keyboard-layout
``Counter`` objects keyed by ``(mistake, correction)`` pairs, then turn each
layout's counts into a 128x128 ASCII substitution matrix.

Matrix orientation (IMPORTANT)
==============================

``m[a][b]`` is indexed **a = query char (row), b = target char (column)** —
the score for "the query has ``a`` where the target has ``b``". A correction
event is "typed ``a`` (the mistake) but meant ``b`` (the correction)", so the
mistake is the query/row and the correction is the target/column. Feed the
*typed* (possibly mistyped) string as the query and the candidate/correct
string as the target. (See ``docs/api/matrices.md`` and the
``stride_align.matrices.keyboard`` docstring.)

Scoring: log-odds, not raw counts
==================================

The naive approach — score = ``log2(count)`` — is *all positive* and so the
aligner never prefers a gap or penalises an implausible pairing; it just
matches everything. Instead we use a **log-odds** score, the same
construction BLOSUM/PAM use:

    score(a, b) = scale * log2( observed / expected )
                = scale * log2( N * C[a][b] / (R[a] * K[b]) )

      observed = C[a][b] / N            normalised counts (your data)
      expected = (R[a]/N) * (K[b]/N)    chance, from the margins
        C[a][b] = count of "typed a, meant b"
        R[a]    = row total  (a as the typed key)
        K[b]    = column total (b as the intended key)
        N       = grand total of all corrections for this layout

Pairs more common than chance score positive (plausible typo -> reward);
rarer than chance score negative; never-observed pairs are floored to a
penalty. Dividing by ``expected`` corrects for raw key frequency, so a key
that shows up in many corrections merely because it is *common* does not get
an inflated score.

The identity diagonal (exact match, a == b) carries no signal from this data
(you do not backspace a key to retype the same key), so it is set explicitly
to a value above the strongest off-diagonal substitution: an exact match
always outscores any typo, while plausible typos still score positive.

Usage
=====

    python tools/build_keyboard_matrices.py path/to/Keystrokes.zip
    python tools/build_keyboard_matrices.py Keystrokes.zip --out DIR --limit 500

Each matrix is written as ``<layout>.npy`` (int8, indexed by ``ord(char)``).
The raw dataset and this script's full multi-layout output are not committed
to the repo; only the small curated set of published matrices ships with
stride-align (under Apache-2.0, with the dataset authors' permission). By
default the files land next to the input zip — pass ``--out`` to choose
another location — then load the curated ones into
``stride_align.matrices.keyboard``.
"""

from __future__ import annotations

import argparse
import csv
import io
import math  # noqa: F401  (kept available for ad-hoc analysis; see build_log_odds_matrix)
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:  # progress bar is nice-to-have, not required
    from tqdm import tqdm
except ImportError:  # pragma: no cover - trivial fallback
    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else iter(())


# ---------------------------------------------------------------------------
# Parsing the (malformed-in-places) raw keystroke logs.
#
# repair_file() and count_pairs() are kept verbatim from the original
# extraction script: the dataset is tab-delimited but some rows have the
# wrong tab count around special keys (ENTER, DIGIT7, CAPSLOCK, HOME), and
# the fix-ups below were worked out empirically against the real data. Do
# not "tidy" them without re-validating against Keystrokes.zip.
# ---------------------------------------------------------------------------


def repair_file(f_in):
    """Normalise rows with an off-by-one tab count back to the 9-field shape.

    The canonical row has 8 tabs (9 fields). Rows with 9 tabs encode a
    special scancode (DIGIT7/CAPSLOCK/HOME) shifted by one; rows with 7 tabs
    are a truncated ENTER that has to be merged with the following line.
    """
    out = io.StringIO()
    hold = []
    for line in f_in:
        if hold:
            assert(line.count('\t') == 1)
            hold.append(line.strip())
            out.write('\t'.join(hold))
            out.write('\n')
            hold = []

        if line.count('\t') == 8:
            out.write(line)
        else:
            if line.count('\t') == 9:
                scancode = line.strip().split()[-1]
                line = line.strip().split('\t')
                line[-2:] = line[-1:]
                line[-2] = {'73': 'DIGIT7', '20': 'CAPSLOCK', '71': 'HOME'}.get(scancode, '')
                if line[-1] == '73':
                    line[-2] = 'DIGIT7'
                elif line[-1] == '20':
                    line[-2] = 'CAPSLOCK'
                elif line[-1] == '71':
                    line[-2] = 'HOME'
                out.write('\t'.join(line))
                out.write('\n')
            elif line.count('\t') == 7:
                line = line.split('\t')
                assert(line[-1] == '\n')
                line[-1] = 'ENTER'
                hold[:] = line

    return io.StringIO(out.getvalue())


def count_pairs(f_in):
    """Yield ``(mistake, correction)`` char pairs from one repaired log.

    Detects the "typed X, Backspace, typed Y" self-correction pattern over a
    sliding window of single-character keypresses (SPACE-bounded, ASCII only).
    ``mistake`` = the char that was deleted; ``correction`` = what replaced it.
    """
    characters = []
    for line in f_in:

        line = line.rstrip('\n')
        scancode = line.split()[-1]
        if scancode == '32':

            character = ' '
        else:
            character = line.split('\t')[-2]
        if len(character) == 1 or character == 'BKSP':
            if character == ' ':
                character = 'SPACE'
            characters.append(character)
            if len(characters) < 3: continue
            if characters[-2] == 'BKSP' and characters[-3] != 'BKSP' and characters[-1] != 'BKSP':
                if 'SPACE' not in characters[-3:]:

                    a = characters[-3]
                    b = characters[-1]

                    if ord(a) < 128 and ord(b) < 128:
                        yield (a, b)


# ---------------------------------------------------------------------------
# Collecting counts across the whole dataset.
# ---------------------------------------------------------------------------


def collect_counters(zip_path, *, limit=None, skip_pids=(3,)):
    """Walk the dataset zip and return ``{layout: Counter{(mistake, correction): n}}``.

    ``limit`` caps the number of participants processed (handy for a quick
    smoke test). Participants in ``skip_pids`` are skipped (pid 3 has a
    known-bad log). Individual unreadable participant files are skipped and
    tallied rather than aborting the whole run.
    """
    counters = defaultdict(Counter)
    z = zipfile.ZipFile(zip_path)

    meta = 'Keystrokes/files/metadata_participants.txt'
    with z.open(meta, mode='r') as fh:
        total = sum(1 for _ in fh)

    rows = csv.DictReader(
        io.TextIOWrapper(z.open(meta, mode='r'), encoding='utf-8'),
        delimiter='\t',
    )

    processed = 0
    failures = 0
    for row in tqdm(rows, total=total):
        pid = int(row['PARTICIPANT_ID'])
        layout = row['LAYOUT']
        if pid in skip_pids:
            continue
        try:
            with z.open(f'Keystrokes/files/{pid}_keystrokes.txt', mode='r') as raw:
                counters[layout].update(
                    count_pairs(repair_file(io.TextIOWrapper(raw, encoding='latin-1')))
                )
        except (KeyError, AssertionError, UnicodeError, IndexError):
            failures += 1
            continue
        processed += 1
        if limit is not None and processed >= limit:
            break

    if failures:
        print(f"  (skipped {failures} unreadable/malformed participant files)", file=sys.stderr)
    return counters


# ---------------------------------------------------------------------------
# Counts -> int8 log-odds substitution matrix.
# ---------------------------------------------------------------------------

ASCII = 128


def build_log_odds_matrix(counter, *, scale=2.0, match_margin=4, floor=None):
    """Turn a ``Counter{(mistake, correction): count}`` into a 128x128 int8 matrix.

    See the module docstring for the orientation and the log-odds formula.
    ``scale`` is the multiplier on ``log2(observed/expected)`` (2.0 gives
    BLOSUM-style half-bit units). ``match_margin`` is how far the identity
    diagonal sits above the strongest substitution. ``floor`` is the score for
    never-observed pairs; ``None`` uses the most-negative observed score so
    "never confused" is treated as at least as unlikely as the rarest
    observed confusion.

    Returns the ``int8`` ndarray (row/col index == ``ord(char)``).
    """
    # Delegate to the library's canonical scorer so the matrices this script
    # writes are bit-identical to what stride_align.matrices.keyboard builds
    # at runtime (single source of truth for the log-odds + identity logic).
    from stride_align.matrices.keyboard import ASCII_ALPHABET, from_confusion_counts

    return from_confusion_counts(
        dict(counter),
        alphabet=ASCII_ALPHABET,
        scale=scale,
        match_margin=match_margin,
        floor=floor,
    ).matrix


def _summarise(layout, counter, grid):
    off = grid[~np.eye(ASCII, dtype=bool)]   # mask the diagonal out of the stats
    return (
        f"{layout:14s} pairs={len(counter):5d} events={sum(counter.values()):8d} "
        f"off-diag[{int(off.min()):4d},{int(off.max()):4d}] match={int(grid[0, 0]):4d}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zip", type=Path, help="path to Keystrokes.zip")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="output dir for <layout>.npy (default: <zip-dir>/keyboard_matrices/, never the repo)",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N participants (smoke test)")
    parser.add_argument("--min-events", type=int, default=1000,
                        help="skip layouts with fewer than this many correction events")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="log-odds multiplier (2.0 = half-bit units, BLOSUM-style)")
    parser.add_argument("--match-margin", type=int, default=4,
                        help="how far the identity diagonal sits above the best substitution")
    args = parser.parse_args(argv)

    out = args.out
    if out is None:
        # Derived data stays OUT of the repo; default next to the input zip.
        out = args.zip.resolve().parent / "keyboard_matrices"
    out.mkdir(parents=True, exist_ok=True)

    counters = collect_counters(args.zip, limit=args.limit)

    # "generic" = every layout's confusions summed into a single matrix (the
    # layout-agnostic default; ships alongside the per-layout matrices).
    if counters:
        generic = Counter()
        for counter in counters.values():
            generic.update(counter)
        counters = {"generic": generic, **counters}

    written = 0
    for layout, counter in sorted(counters.items()):
        events = sum(counter.values())
        if events < args.min_events:
            print(f"  skip {layout!r}: only {events} events (< {args.min_events})")
            continue
        grid = build_log_odds_matrix(
            counter, scale=args.scale, match_margin=args.match_margin,
        )
        safe = "".join(c if c.isalnum() else "_" for c in layout) or "unknown"
        path = out / f"{safe}.npy"
        np.save(path, grid)
        written += 1
        print(_summarise(layout, counter, grid), "->", path.name)

    print(f"\nwrote {written} layout matrix file(s) to {out}")


if __name__ == "__main__":
    main()
