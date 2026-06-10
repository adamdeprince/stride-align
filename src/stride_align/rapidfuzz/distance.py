"""``rapidfuzz.distance`` shim — metric namespaces.

Each metric is a class with four classmethods (``distance``,
``normalized_distance``, ``similarity``, ``normalized_similarity``)
plus ``editops`` / ``opcodes`` for the alignment-capable ones
(``Levenshtein``, ``Indel``).

The collection types ``Editops``, ``Editop``, ``Opcodes``, ``Opcode``,
``MatchingBlock``, ``ScoreAlignment`` match rapidfuzz's shapes so
downstream code that introspects them keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import stride_align as _sa


# --------------------------------------------------------------------
# Collection types
# --------------------------------------------------------------------

@dataclass
class Editop:
    """One edit operation. ``tag`` ∈ {'insert', 'delete', 'replace'}.
    ``src_pos`` is the position in the source where the op applies;
    ``dest_pos`` is the position in the destination."""

    tag: str
    src_pos: int
    dest_pos: int


@dataclass
class Opcode:
    """A run of one operation, like difflib's get_opcodes. ``tag`` ∈
    {'equal', 'insert', 'delete', 'replace'}."""

    tag: str
    src_start: int
    src_end: int
    dest_start: int
    dest_end: int


@dataclass
class Editops:
    """List of ``Editop`` plus the source / destination lengths."""

    ops: List[Editop] = field(default_factory=list)
    src_len: int = 0
    dest_len: int = 0

    def __iter__(self):
        return iter(self.ops)

    def __len__(self):
        return len(self.ops)

    def __getitem__(self, i):
        return self.ops[i]

    def __repr__(self):
        return f"Editops({self.ops!r}, src_len={self.src_len}, dest_len={self.dest_len})"


@dataclass
class Opcodes:
    ops: List[Opcode] = field(default_factory=list)
    src_len: int = 0
    dest_len: int = 0

    def __iter__(self):
        return iter(self.ops)

    def __len__(self):
        return len(self.ops)

    def __getitem__(self, i):
        return self.ops[i]

    def __repr__(self):
        return f"Opcodes({self.ops!r}, src_len={self.src_len}, dest_len={self.dest_len})"


@dataclass
class MatchingBlock:
    a: int
    b: int
    size: int


@dataclass
class ScoreAlignment:
    score: float
    src_start: int
    src_end: int
    dest_start: int
    dest_end: int


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _apply(s, processor):
    return s if processor is None else processor(s)


def _len(s) -> int:
    return len(s) if s is not None else 0


def _max_distance_indel(s1, s2) -> int:
    return _len(s1) + _len(s2)


def _max_distance_lev(s1, s2) -> int:
    return max(_len(s1), _len(s2))


def _distance_to_similarity(distance: int, max_distance: int) -> int:
    return max_distance - distance


def _normalized(distance: int, max_distance: int) -> float:
    if max_distance == 0:
        return 0.0
    return distance / max_distance


def _operations_to_editops(operations: str, src_start: int = 0,
                            dest_start: int = 0) -> List[Editop]:
    """Convert stride-align's per-cell ``operations`` string
    (``=`` match, ``X`` mismatch, ``I`` insert, ``D`` delete) into the
    list of edit operations rapidfuzz's editops returns: every match
    is skipped, every other op is an ``Editop`` with its source /
    destination position."""
    out: List[Editop] = []
    src = src_start
    dest = dest_start
    for op in operations:
        if op == "=":
            src += 1
            dest += 1
        elif op == "X":
            out.append(Editop(tag="replace", src_pos=src, dest_pos=dest))
            src += 1
            dest += 1
        elif op == "I":
            # rapidfuzz: 'insert' means a char appears in dest that's
            # not in src — stride-align's 'I' has the same meaning.
            out.append(Editop(tag="insert", src_pos=src, dest_pos=dest))
            dest += 1
        elif op == "D":
            out.append(Editop(tag="delete", src_pos=src, dest_pos=dest))
            src += 1
    return out


def _operations_to_opcodes(operations: str, src_len: int, dest_len: int) -> List[Opcode]:
    """Convert per-cell operations into run-length-encoded opcodes
    (difflib-style: equal / replace / insert / delete)."""
    if not operations:
        return [Opcode(tag="equal", src_start=0, src_end=src_len,
                       dest_start=0, dest_end=dest_len)] if src_len == dest_len else []
    out: List[Opcode] = []
    src = 0
    dest = 0
    i = 0
    while i < len(operations):
        op = operations[i]
        j = i
        while j < len(operations) and operations[j] == op:
            j += 1
        run = j - i
        if op == "=":
            out.append(Opcode(tag="equal", src_start=src, src_end=src + run,
                              dest_start=dest, dest_end=dest + run))
            src += run; dest += run
        elif op == "X":
            out.append(Opcode(tag="replace", src_start=src, src_end=src + run,
                              dest_start=dest, dest_end=dest + run))
            src += run; dest += run
        elif op == "I":
            out.append(Opcode(tag="insert", src_start=src, src_end=src,
                              dest_start=dest, dest_end=dest + run))
            dest += run
        elif op == "D":
            out.append(Opcode(tag="delete", src_start=src, src_end=src + run,
                              dest_start=dest, dest_end=dest))
            src += run
        i = j
    return out


def _coerce_score_cutoff_distance(value: int, score_cutoff: Optional[int]) -> int:
    """Distance score-cutoff: rapidfuzz returns ``score_cutoff + 1``
    when the true distance exceeds the cutoff. stride-align kernels
    don't currently support push-down cutoff for these metrics, so we
    return the true value (matches the upstream behaviour observed
    empirically — see tests/test_rapidfuzz_shim.py)."""
    return value


# --------------------------------------------------------------------
# Levenshtein
# --------------------------------------------------------------------

class _Levenshtein:
    """Mirrors ``rapidfuzz.distance.Levenshtein``.

    The four hot methods bind directly to the C++ dispatchers; custom
    ``weights`` aren't supported (the bit-parallel Myers kernel
    assumes unit insert/delete/substitute), so non-default weights
    will silently produce the default-weights result. Callers that
    rely on non-default weights should use ``rapidfuzz.distance``
    directly.
    """

    distance              = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_Levenshtein_distance)
    similarity            = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_Levenshtein_similarity)
    normalized_distance   = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_Levenshtein_normalized_distance)
    normalized_similarity = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_Levenshtein_normalized_similarity)

    @staticmethod
    def editops(s1, s2, *, processor: Optional[Callable] = None,
                score_hint=None) -> Editops:
        a = _apply(s1, processor); b = _apply(s2, processor)
        # Use Needleman-Wunsch with unit weights to get the alignment.
        path = _sa.needleman_wunsch_path(a, b, match_score=0, mismatch_score=-1, gap_score=-1)
        ops = _operations_to_editops(path.operations)
        return Editops(ops=ops, src_len=len(a), dest_len=len(b))

    @staticmethod
    def opcodes(s1, s2, *, processor: Optional[Callable] = None,
                score_hint=None) -> Opcodes:
        a = _apply(s1, processor); b = _apply(s2, processor)
        path = _sa.needleman_wunsch_path(a, b, match_score=0, mismatch_score=-1, gap_score=-1)
        ops = _operations_to_opcodes(path.operations, len(a), len(b))
        return Opcodes(ops=ops, src_len=len(a), dest_len=len(b))


# --------------------------------------------------------------------
# Indel
# --------------------------------------------------------------------

class _Indel:
    """Indel distance: Levenshtein restricted to insert/delete (no
    substitute). Equivalent to ``|s1| + |s2| - 2 * LCS(s1, s2)``.

    The four hot methods (``distance``, ``similarity``,
    ``normalized_distance``, ``normalized_similarity``) re-export the
    C++ ``_shim_dist_Indel_*`` dispatchers directly. Each handles
    processor / score_cutoff / score translation in one C++ call.
    """

    distance              = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_Indel_distance)
    similarity            = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_Indel_similarity)
    normalized_distance   = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_Indel_normalized_distance)
    normalized_similarity = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_Indel_normalized_similarity)


# --------------------------------------------------------------------
# Hamming
# --------------------------------------------------------------------

class _Hamming:
    """Hamming distance with rapidfuzz ``pad=True`` semantics (padded
    with mismatches) by default. ``pad`` and the length-mismatch
    ValueError both live in the C++ dispatcher, so the Python class
    is a direct re-export — same per-call overhead profile as
    ``_Indel``."""

    distance              = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_Hamming_distance)
    similarity            = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_Hamming_similarity)
    normalized_distance   = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_Hamming_normalized_distance)
    normalized_similarity = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_Hamming_normalized_similarity)


# --------------------------------------------------------------------
# Jaro / JaroWinkler
# --------------------------------------------------------------------

class _Jaro:
    distance              = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_Jaro_distance)
    similarity            = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_Jaro_similarity)
    normalized_distance   = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_Jaro_normalized_distance)
    normalized_similarity = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_Jaro_normalized_similarity)


class _JaroWinkler:
    distance              = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_JaroWinkler_distance)
    similarity            = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_JaroWinkler_similarity)
    normalized_distance   = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_JaroWinkler_normalized_distance)
    normalized_similarity = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_JaroWinkler_normalized_similarity)


# --------------------------------------------------------------------
# DamerauLevenshtein / OSA
# --------------------------------------------------------------------

class _DamerauLevenshtein:
    """True Damerau-Levenshtein (unrestricted transpositions)."""

    distance              = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_DamerauLevenshtein_distance)
    similarity            = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_DamerauLevenshtein_similarity)
    normalized_distance   = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_DamerauLevenshtein_normalized_distance)
    normalized_similarity = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_DamerauLevenshtein_normalized_similarity)


class _OSA:
    """Optimal String Alignment (restricted Damerau-Levenshtein)."""

    distance              = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_OSA_distance)
    similarity            = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_OSA_similarity)
    normalized_distance   = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_OSA_normalized_distance)
    normalized_similarity = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_OSA_normalized_similarity)


# --------------------------------------------------------------------
# LCSseq (longest common subsequence)
# --------------------------------------------------------------------

class _LCSseq:
    distance              = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_LCSseq_distance)
    similarity            = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_LCSseq_similarity)
    normalized_distance   = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_LCSseq_normalized_distance)
    normalized_similarity = staticmethod(_sa._LEVENSHTEIN_BACKEND._shim_dist_LCSseq_normalized_similarity)


# --------------------------------------------------------------------
# Public namespace
# --------------------------------------------------------------------

Levenshtein = _Levenshtein
Indel = _Indel
Hamming = _Hamming
Jaro = _Jaro
JaroWinkler = _JaroWinkler
DamerauLevenshtein = _DamerauLevenshtein
OSA = _OSA
LCSseq = _LCSseq


__all__ = [
    # Classes
    "Levenshtein", "Indel", "Hamming", "Jaro", "JaroWinkler",
    "DamerauLevenshtein", "OSA", "LCSseq",
    # Collection types
    "Editop", "Editops", "Opcode", "Opcodes",
    "MatchingBlock", "ScoreAlignment",
]
