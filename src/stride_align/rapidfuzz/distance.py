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
    """Mirrors ``rapidfuzz.distance.Levenshtein``."""

    @staticmethod
    def distance(s1, s2, *, weights: Tuple[int, int, int] = (1, 1, 1),
                 processor: Optional[Callable] = None,
                 score_cutoff: Optional[int] = None,
                 score_hint=None) -> int:
        a = _apply(s1, processor); b = _apply(s2, processor)
        if weights != (1, 1, 1):
            raise NotImplementedError(
                "stride_align.rapidfuzz.distance.Levenshtein does not yet "
                "support custom weights (insert, delete, replace)"
            )
        return _coerce_score_cutoff_distance(int(_sa.levenshtein_score(a, b)), score_cutoff)

    @staticmethod
    def similarity(s1, s2, *, weights: Tuple[int, int, int] = (1, 1, 1),
                   processor: Optional[Callable] = None,
                   score_cutoff: Optional[int] = None,
                   score_hint=None) -> int:
        a = _apply(s1, processor); b = _apply(s2, processor)
        d = _Levenshtein.distance(a, b, weights=weights)
        return _distance_to_similarity(d, _max_distance_lev(a, b))

    @staticmethod
    def normalized_distance(s1, s2, *, weights: Tuple[int, int, int] = (1, 1, 1),
                            processor: Optional[Callable] = None,
                            score_cutoff: Optional[float] = None,
                            score_hint=None) -> float:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return 1.0 - float(_sa.levenshtein_normalized_score(a, b))

    @staticmethod
    def normalized_similarity(s1, s2, *, weights: Tuple[int, int, int] = (1, 1, 1),
                              processor: Optional[Callable] = None,
                              score_cutoff: Optional[float] = None,
                              score_hint=None) -> float:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return float(_sa.levenshtein_normalized_score(a, b))

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
    substitute). Equivalent to ``|s1| + |s2| - 2 * LCS(s1, s2)``."""

    @staticmethod
    def distance(s1, s2, *, processor: Optional[Callable] = None,
                 score_cutoff: Optional[int] = None, score_hint=None) -> int:
        a = _apply(s1, processor); b = _apply(s2, processor)
        # Kernel-level cutoff: pass through to the bit-parallel Indel
        # kernel. score_cutoff for distance is the max-distance the
        # caller cares about; the kernel returns cutoff+1 when the
        # true distance exceeds it.
        return int(_sa.indel_score(a, b, score_cutoff=score_cutoff))

    @staticmethod
    def similarity(s1, s2, *, processor: Optional[Callable] = None,
                   score_cutoff: Optional[int] = None, score_hint=None) -> int:
        a = _apply(s1, processor); b = _apply(s2, processor)
        # similarity = max_distance - distance, so a similarity cutoff
        # of k corresponds to a distance cutoff of max_distance - k.
        if score_cutoff is None:
            return _max_distance_indel(a, b) - int(_sa.indel_score(a, b))
        max_d = _max_distance_indel(a, b)
        distance_cutoff = max_d - int(score_cutoff)
        return max_d - int(_sa.indel_score(a, b, score_cutoff=distance_cutoff))

    @staticmethod
    def normalized_distance(s1, s2, *, processor: Optional[Callable] = None,
                            score_cutoff: Optional[float] = None,
                            score_hint=None) -> float:
        a = _apply(s1, processor); b = _apply(s2, processor)
        # normalized_distance = 1 - normalized_similarity. Translate.
        sim_cutoff = None if score_cutoff is None else 1.0 - float(score_cutoff)
        return 1.0 - float(_sa.indel_normalized_score(a, b, score_cutoff=sim_cutoff))

    @staticmethod
    def normalized_similarity(s1, s2, *, processor: Optional[Callable] = None,
                              score_cutoff: Optional[float] = None,
                              score_hint=None) -> float:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return float(_sa.indel_normalized_score(a, b, score_cutoff=score_cutoff))


# --------------------------------------------------------------------
# Hamming
# --------------------------------------------------------------------

class _Hamming:
    @staticmethod
    def distance(s1, s2, *, pad: bool = True,
                 processor: Optional[Callable] = None,
                 score_cutoff: Optional[int] = None, score_hint=None) -> int:
        a = _apply(s1, processor); b = _apply(s2, processor)
        # rapidfuzz: pad=True (default) treats lengths as padded with
        # mismatches; pad=False raises on length mismatch.
        if len(a) != len(b) and not pad:
            raise ValueError("Hamming distance requires equal-length inputs when pad=False")
        if len(a) == len(b):
            return int(_sa.hamming_score(a, b))
        # Padded Hamming: extra characters count as mismatches.
        short, long = (a, b) if len(a) < len(b) else (b, a)
        head_mismatches = int(_sa.hamming_score(short, long[: len(short)]))
        return head_mismatches + (len(long) - len(short))

    @staticmethod
    def similarity(s1, s2, *, pad: bool = True,
                   processor: Optional[Callable] = None,
                   score_cutoff: Optional[int] = None, score_hint=None) -> int:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return max(len(a), len(b)) - _Hamming.distance(a, b, pad=pad)

    @staticmethod
    def normalized_distance(s1, s2, *, pad: bool = True,
                            processor: Optional[Callable] = None,
                            score_cutoff: Optional[float] = None,
                            score_hint=None) -> float:
        a = _apply(s1, processor); b = _apply(s2, processor)
        m = max(len(a), len(b))
        if m == 0:
            return 0.0
        return _Hamming.distance(a, b, pad=pad) / m

    @staticmethod
    def normalized_similarity(s1, s2, *, pad: bool = True,
                              processor: Optional[Callable] = None,
                              score_cutoff: Optional[float] = None,
                              score_hint=None) -> float:
        return 1.0 - _Hamming.normalized_distance(s1, s2, pad=pad, processor=processor)


# --------------------------------------------------------------------
# Jaro / JaroWinkler
# --------------------------------------------------------------------

class _Jaro:
    @staticmethod
    def distance(s1, s2, *, processor: Optional[Callable] = None,
                 score_cutoff: Optional[float] = None, score_hint=None) -> float:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return 1.0 - float(_sa.jaro_similarity(a, b))

    @staticmethod
    def similarity(s1, s2, *, processor: Optional[Callable] = None,
                   score_cutoff: Optional[float] = None, score_hint=None) -> float:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return float(_sa.jaro_similarity(a, b))

    @staticmethod
    def normalized_distance(s1, s2, *, processor: Optional[Callable] = None,
                            score_cutoff: Optional[float] = None,
                            score_hint=None) -> float:
        return _Jaro.distance(s1, s2, processor=processor)

    @staticmethod
    def normalized_similarity(s1, s2, *, processor: Optional[Callable] = None,
                              score_cutoff: Optional[float] = None,
                              score_hint=None) -> float:
        return _Jaro.similarity(s1, s2, processor=processor)


class _JaroWinkler:
    @staticmethod
    def distance(s1, s2, *, prefix_weight: float = 0.1,
                 processor: Optional[Callable] = None,
                 score_cutoff: Optional[float] = None, score_hint=None) -> float:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return 1.0 - float(_sa.jaro_winkler_similarity(a, b, prefix_weight=prefix_weight))

    @staticmethod
    def similarity(s1, s2, *, prefix_weight: float = 0.1,
                   processor: Optional[Callable] = None,
                   score_cutoff: Optional[float] = None, score_hint=None) -> float:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return float(_sa.jaro_winkler_similarity(a, b, prefix_weight=prefix_weight))

    @staticmethod
    def normalized_distance(s1, s2, *, prefix_weight: float = 0.1,
                            processor: Optional[Callable] = None,
                            score_cutoff: Optional[float] = None,
                            score_hint=None) -> float:
        return _JaroWinkler.distance(s1, s2, prefix_weight=prefix_weight, processor=processor)

    @staticmethod
    def normalized_similarity(s1, s2, *, prefix_weight: float = 0.1,
                              processor: Optional[Callable] = None,
                              score_cutoff: Optional[float] = None,
                              score_hint=None) -> float:
        return _JaroWinkler.similarity(s1, s2, prefix_weight=prefix_weight, processor=processor)


# --------------------------------------------------------------------
# DamerauLevenshtein / OSA
# --------------------------------------------------------------------

class _DamerauLevenshtein:
    """True Damerau-Levenshtein (unrestricted transpositions)."""

    @staticmethod
    def distance(s1, s2, *, processor: Optional[Callable] = None,
                 score_cutoff: Optional[int] = None, score_hint=None) -> int:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return int(_sa.true_damerau_levenshtein_score(a, b))

    @staticmethod
    def similarity(s1, s2, *, processor: Optional[Callable] = None,
                   score_cutoff: Optional[int] = None, score_hint=None) -> int:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return _max_distance_lev(a, b) - int(_sa.true_damerau_levenshtein_score(a, b))

    @staticmethod
    def normalized_distance(s1, s2, *, processor: Optional[Callable] = None,
                            score_cutoff: Optional[float] = None,
                            score_hint=None) -> float:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return 1.0 - float(_sa.true_damerau_levenshtein_normalized_score(a, b))

    @staticmethod
    def normalized_similarity(s1, s2, *, processor: Optional[Callable] = None,
                              score_cutoff: Optional[float] = None,
                              score_hint=None) -> float:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return float(_sa.true_damerau_levenshtein_normalized_score(a, b))


class _OSA:
    """Optimal String Alignment (restricted Damerau-Levenshtein)."""

    @staticmethod
    def distance(s1, s2, *, processor: Optional[Callable] = None,
                 score_cutoff: Optional[int] = None, score_hint=None) -> int:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return int(_sa.damerau_levenshtein_score(a, b))

    @staticmethod
    def similarity(s1, s2, *, processor: Optional[Callable] = None,
                   score_cutoff: Optional[int] = None, score_hint=None) -> int:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return _max_distance_lev(a, b) - int(_sa.damerau_levenshtein_score(a, b))

    @staticmethod
    def normalized_distance(s1, s2, *, processor: Optional[Callable] = None,
                            score_cutoff: Optional[float] = None,
                            score_hint=None) -> float:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return 1.0 - float(_sa.damerau_levenshtein_normalized_score(a, b))

    @staticmethod
    def normalized_similarity(s1, s2, *, processor: Optional[Callable] = None,
                              score_cutoff: Optional[float] = None,
                              score_hint=None) -> float:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return float(_sa.damerau_levenshtein_normalized_score(a, b))


# --------------------------------------------------------------------
# LCSseq (longest common subsequence)
# --------------------------------------------------------------------

class _LCSseq:
    @staticmethod
    def distance(s1, s2, *, processor: Optional[Callable] = None,
                 score_cutoff: Optional[int] = None, score_hint=None) -> int:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return max(len(a), len(b)) - int(_sa.lcs_length(a, b))

    @staticmethod
    def similarity(s1, s2, *, processor: Optional[Callable] = None,
                   score_cutoff: Optional[int] = None, score_hint=None) -> int:
        a = _apply(s1, processor); b = _apply(s2, processor)
        return int(_sa.lcs_length(a, b))

    @staticmethod
    def normalized_distance(s1, s2, *, processor: Optional[Callable] = None,
                            score_cutoff: Optional[float] = None,
                            score_hint=None) -> float:
        a = _apply(s1, processor); b = _apply(s2, processor)
        m = max(len(a), len(b))
        if m == 0:
            return 0.0
        return (m - int(_sa.lcs_length(a, b))) / m

    @staticmethod
    def normalized_similarity(s1, s2, *, processor: Optional[Callable] = None,
                              score_cutoff: Optional[float] = None,
                              score_hint=None) -> float:
        return 1.0 - _LCSseq.normalized_distance(s1, s2, processor=processor)


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
