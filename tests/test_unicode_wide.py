"""Unicode auto-promote (Phase B): UCS-2 / UCS-4 strings with >256 distinct
codepoints route to the 16-bit Farrar kernel instead of raising the
narrow tokeniser's `ValueError("...256 distinct symbols")`.

UCS-1 strings (Latin-1) can have at most 256 distinct codepoints by
construction, so they continue to use the fast 8-bit Farrar kernel
unchanged.
"""

from __future__ import annotations

import random

import pytest

import stride_align as sa


def _build_chinese(length: int, alphabet_size: int, seed: int) -> str:
    chars = [chr(0x4E00 + i) for i in range(alphabet_size)]
    rng = random.Random(seed)
    return "".join(rng.choice(chars) for _ in range(length))


def test_ucs1_unchanged() -> None:
    # UCS-1 (Latin-1) — narrow path always (≤256 distinct guaranteed).
    assert sa.smith_waterman_score(
        "hello", "world", match_score=2, mismatch_score=-1, gap_score=-1,
    ) == 2
    # 'café' is a UCS-1 string (all codepoints ≤ U+00FF).
    assert sa.smith_waterman_score(
        "café", "café", match_score=2, mismatch_score=-1, gap_score=-1,
    ) == 8


def test_ucs2_small_alphabet_unchanged() -> None:
    # UCS-2 (BMP) but ≤256 distinct codepoints — pre-scan keeps it
    # on the narrow path.
    q = "你好世界"
    t = "你好啊朋友"
    # 你好 prefix matches, score 4.
    assert sa.smith_waterman_score(
        q, t, match_score=2, mismatch_score=-1, gap_score=-1,
    ) == 4


def test_ucs2_large_alphabet_lifts_256_cap() -> None:
    # 300 distinct CJK codepoints in query — previously raised
    # ValueError; now runs on the 16-bit wide kernel.
    chars = [chr(0x4E00 + i) for i in range(300)]
    q = "".join(chars)
    t = "".join(chars)
    # Identity self-alignment: 300 matches × 2 = 600.
    assert sa.smith_waterman_score(
        q, t, match_score=2, mismatch_score=-1, gap_score=-1,
    ) == 600


def test_ucs2_large_alphabet_random() -> None:
    # 100-char paragraph from a 300-symbol CJK alphabet, two independent
    # samples. Verifies the kernel runs and produces a non-negative
    # score (we don't pin exact value — many co-optimal alignments).
    q = _build_chinese(100, 300, seed=11)
    t = _build_chinese(100, 300, seed=12)
    score = sa.smith_waterman_score(
        q, t, match_score=2, mismatch_score=-1, gap_score=-1,
    )
    assert score >= 0
    # Self-alignment of the same q must equal len(q) * match.
    assert sa.smith_waterman_score(
        q, q, match_score=2, mismatch_score=-1, gap_score=-1,
    ) == 200


def test_ucs4_alphabet_above_uint16_cap_routes_to_uint32() -> None:
    # Above 65 535 distinct codepoints we no longer raise — the
    # auto-promote chain falls through to the 32-bit Farrar kernel
    # feeding raw UCS-4 codepoints as uint32 tokens.
    #
    # The striped profile is keyed by the *target* alphabet, so a huge
    # query alphabet against a tiny target stays cheap (profile is
    # |target distinct| x |query|, not |query| x |query|). A 66 000-char
    # query against a 2-char target exercises the >65 535 distinct
    # routing without the quadratic O(|q|x|q|) blowup.
    chars = [chr(0x4E00 + i) for i in range(60000)] + [chr(0x20000 + i) for i in range(6000)]
    q = "".join(chars)              # 66 000 distinct codepoints
    t = q[:2]                       # first two distinct codepoints
    # Combined alphabet is 66 000 (>65 535) so this routes to uint32.
    # Best local alignment: the 2-char target matches the start of the
    # query contiguously -> 2 matches x 2 = 4.
    score = sa.smith_waterman_score(q, t, match_score=2, mismatch_score=-1, gap_score=-1)
    assert score == 4

    # And the mirror: a tiny query against a 66 000-distinct target.
    # The profile is keyed by the target alphabet (66 000 rows x one
    # query segment ~= 4 MB), and the DP scans 66 000 target columns,
    # so this is cheap in both memory and time. Best local alignment is
    # the 2-char query matching the target's first two codepoints -> 4.
    score_rev = sa.smith_waterman_score(t, q, match_score=2, mismatch_score=-1, gap_score=-1)
    assert score_rev == 4


def test_mixed_ucs1_ucs2_with_large_target_promotes() -> None:
    # UCS-1 query, UCS-2 target with >256 distinct → wide path.
    q = "abc"
    t_chars = [chr(0x4E00 + i) for i in range(300)]
    t = "".join(t_chars)
    # No common codepoints → score 0.
    assert sa.smith_waterman_score(
        q, t, match_score=2, mismatch_score=-1, gap_score=-1,
    ) == 0
