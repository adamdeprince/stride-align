"""Regression test for the bounded lazy-F Smith-Waterman shortcut bug.

Background: docs/known-issue-bounded-lazy-f-scan.md.

The pre-fix striped Farrar kernel had a "bounded" lazy-F correction that stopped
its correction sweep early and, on certain structured inputs, returned a LOCAL
Smith-Waterman score that was too LOW. The five length-1024 inputs exercised here
each triggered it (the AVX2 exact-fill path in particular). The corrected kernel
must return the true, higher score.

The expected values are not self-referential: independent implementations that
share no code with stride-align -- a from-scratch textbook DP and Biopython's
PairwiseAligner -- confirm every `correct` value below and reject every
`bounded` value. Reproduce with ``python tools/verify_bounded_lazy_f.py``.

Scoring is the linear-gap local model that exposed the bug: match +8,
mismatch -9, gap -1 per residue.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import stride_align as sa

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "docs" / "bounded-lazy-f-counterexamples.txt"
# When run from the source tree the fixture must exist; only tolerate its
# absence for out-of-tree (installed-wheel) runs, so a moved/renamed fixture
# fails loudly instead of silently skipping the regression forever.
_IN_SOURCE_TREE = (_REPO / "pyproject.toml").exists()

# (example index, correct local-SW score, historical too-low "bounded" score)
_CASES = [
    (1, 3320, 3310),
    (2, 4907, 4897),
    (3, 2628, 2601),
    (4, 3010, 2986),
    (5, 4480, 4463),
]

MATCH, MISMATCH, GAP = 8, -9, -1


def _load_sequences() -> dict[int, tuple[str, str]]:
    """Parse the counter-example inputs, mapping integer tokens 1..7 to A..G.

    Only character equality affects the score, so the letter mapping preserves
    the exact alignment problem recorded in the data file.
    """
    if not _DATA.exists():
        return {}
    lines = _DATA.read_text().splitlines()
    seqs: dict[int, tuple[str, str]] = {}
    i = 0
    while i < len(lines):
        header = re.match(r"\*\*\* DISAGREEMENT #(\d+) ", lines[i])
        if header and i + 2 < len(lines):
            to_str = lambda spec: "".join(
                chr(64 + int(tok)) for tok in spec.split(":", 1)[1].split()
            )
            seqs[int(header.group(1))] = (to_str(lines[i + 1]), to_str(lines[i + 2]))
            i += 3
        else:
            i += 1
    return seqs


_SEQUENCES = _load_sequences()


@pytest.mark.parametrize("index, correct, bounded", _CASES)
def test_bounded_lazy_f_returns_true_score(index: int, correct: int, bounded: int) -> None:
    """The fixed kernel returns the true score, never the too-low bounded one."""
    if index not in _SEQUENCES:
        if _IN_SOURCE_TREE:
            pytest.fail(
                f"counter-example fixture missing from source tree: {_DATA}. "
                "The bounded lazy-F regression cannot run -- do not skip silently."
            )
        pytest.skip(f"counter-example data unavailable ({_DATA} not present)")

    query, target = _SEQUENCES[index]

    score = sa.smith_waterman_score(
        query, target, match_score=MATCH, mismatch_score=MISMATCH, gap_score=GAP
    )
    assert score == correct, (
        f"example {index}: expected the true score {correct}, got {score}"
        + (f" (regressed to the buggy bounded value {bounded})" if score == bounded else "")
    )

    # Same input through the explicit Farrar entry point (the path that had the bug).
    farrar = sa.smith_waterman_farrar_score(
        query, target, match_score=MATCH, mismatch_score=MISMATCH, gap_score=GAP
    )
    assert farrar == correct, f"example {index}: farrar score {farrar} != true score {correct}"
