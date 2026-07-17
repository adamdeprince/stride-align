#!/usr/bin/env python3
"""Independent cross-check for the bounded lazy-F counter-examples.

Background: see docs/known-issue-bounded-lazy-f-scan.md. The incumbent striped
Smith-Waterman kernel had a "bounded" correction sweep that could stop early and
report a local alignment score that was too low. This script re-scores the
recorded counter-examples with two implementations that share no code with
stride-align, to prove that the *low* ("bounded") value is the wrong one:

  1. a pure-Python textbook Smith-Waterman -- the literal O(n*m) recurrence, no
     striping, no SIMD, nothing to get subtly wrong;
  2. Biopython's PairwiseAligner -- a widely used, independently written library
     that does not use Farrar's striped method at all (optional; skipped if
     biopython is not installed).

Both must equal the `materialized` value (stride-align's full, non-shortcut
sweep) and differ from the `bounded` value.

Scoring is the linear-gap local model recorded in the data file's header:
  match = +8, mismatch = -9, gap = -1 per residue, local (zero floor).

Usage:
    pip install biopython            # for the third-party cross-check
    python tools/verify_bounded_lazy_f.py [docs/bounded-lazy-f-counterexamples.txt]
"""
import re
import sys

MATCH, MISMATCH, GAP = 8, -9, -1
DEFAULT_DATA = "docs/bounded-lazy-f-counterexamples.txt"


def parse(path):
    exs, lines, i = [], open(path).read().splitlines(), 0
    while i < len(lines):
        m = re.match(
            r"\*\*\* DISAGREEMENT #(\d+) bounded=(-?\d+) materialized=(-?\d+) scalar=(-?\d+)",
            lines[i],
        )
        if m and i + 2 < len(lines):
            q = list(map(int, lines[i + 1].split(":", 1)[1].split()))
            t = list(map(int, lines[i + 2].split(":", 1)[1].split()))
            exs.append(
                dict(idx=int(m[1]), bounded=int(m[2]), materialized=int(m[3]),
                     scalar=int(m[4]), q=q, t=t)
            )
            i += 3
        else:
            i += 1
    return exs


def sw_textbook(q, t):
    """Literal Smith-Waterman, linear gap. No cleverness -- this is the def."""
    m = len(t)
    prev = [0] * (m + 1)
    best = 0
    for qi in q:
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            s = MATCH if qi == t[j - 1] else MISMATCH
            h = prev[j - 1] + s
            up = prev[j] + GAP
            if up > h:
                h = up
            left = cur[j - 1] + GAP
            if left > h:
                h = left
            if h < 0:
                h = 0
            cur[j] = h
            if h > best:
                best = h
        prev = cur
    return best


def sw_biopython(q, t):
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = MATCH
    aligner.mismatch_score = MISMATCH
    aligner.open_gap_score = GAP
    aligner.extend_gap_score = GAP
    to_str = lambda seq: "".join(chr(64 + x) for x in seq)  # 1..26 -> A..Z
    return int(aligner.score(to_str(q), to_str(t)))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
    exs = parse(path)
    print(f"Loaded {len(exs)} counter-examples from {path}")
    print(f"Scoring: match={MATCH} mismatch={MISMATCH} gap={GAP} (linear, local)\n")
    try:
        import Bio  # noqa: F401
        have_bio = True
    except ImportError:
        have_bio = False
        print("(biopython not installed -- skipping the third-party cross-check; "
              "the textbook DP still runs)\n")

    hdr = (f"{'#':>2} {'len':>5} {'bounded':>8} {'materialized':>13} "
           f"{'textbook-DP':>12} {'biopython':>10}  verdict")
    print(hdr)
    print("-" * len(hdr))
    all_ok = True
    for e in exs:
        dp = sw_textbook(e["q"], e["t"])
        bio = sw_biopython(e["q"], e["t"]) if have_bio else None
        refs = [dp] + ([bio] if bio is not None else [])
        ok = all(r == e["materialized"] for r in refs) and all(r != e["bounded"] for r in refs)
        all_ok &= ok
        biostr = f"{bio:>10}" if bio is not None else f"{'(n/a)':>10}"
        verdict = "bounded WRONG (refs=materialized)" if ok else "!! UNEXPECTED !!"
        print(f"{e['idx']:>2} {len(e['q']):>5} {e['bounded']:>8} "
              f"{e['materialized']:>13} {dp:>12} {biostr}  {verdict}")

    print()
    if all_ok:
        print("RESULT: every independent reference matches `materialized` and rejects "
              "`bounded`.\n        The bounded lazy-F shortcut under-reported the score.")
    else:
        print("RESULT: references did NOT uniformly confirm -- investigate.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
