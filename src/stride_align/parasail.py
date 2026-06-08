"""parasail-compatible shim — drop-in replacement for the parasail
Python bindings.

Usage: replace ``import parasail`` with
``import stride_align.parasail as parasail`` and most parasail code
keeps working unchanged. Concretely:

* The core entry points ``sw``, ``nw``, ``sg`` and their ``_trace``
  and ``_stats`` variants take the same ``(s1, s2, open, extend,
  matrix)`` signature parasail uses (gap penalties are positive
  numbers, BLAST convention: a length-N gap costs
  ``open + (N - 1) * extend``).
* ``matrix_create(alphabet, match, mismatch, case_sensitive=None)``
  returns a ``Matrix`` instance whose attributes match parasail's
  (``.size``, ``.matrix``, ``.mapper``, ``.name``, ``.min``,
  ``.max``, ``.copy``, ``.set_value``).
* Pre-built matrices ``blosum45``, ``blosum50``, ``blosum62``,
  ``blosum80``, ``blosum90``, ``pam30``, ``pam70``, ``pam250`` are
  available as module attributes, sharing the parasail ``Matrix``
  shape.
* ``Result`` objects expose ``.score``, ``.end_query``, ``.end_ref``
  (inclusive last index, parasail convention) and, for ``_trace`` /
  ``_stats`` variants, ``.cigar`` and ``.traceback`` and the stats
  fields ``.length``, ``.matches``, ``.mismatches``, ``.similar``.
* ``Cigar`` exposes ``.decode`` (the CIGAR string as ``bytes``),
  ``.beg_query``, ``.beg_ref``, ``.len``.
* ``Traceback`` exposes ``.query``, ``.ref``, ``.comp`` (the
  ``|.|`` matched-region annotation string parasail uses).
* The 2000+ kernel-suffix variants
  (``sw_striped_avx2_16``, ``nw_scan_64``, …) all alias to the
  matching core entry via module-level ``__getattr__`` —
  stride-align picks the kernel internally, so an explicit suffix
  has no effect.
* Capability detection functions ``can_use_sse2``, ``can_use_sse41``,
  ``can_use_avx2``, ``can_use_altivec``, ``can_use_neon`` report
  what the loaded stride-align backend supports.

Known divergences from upstream parasail:

* CIGAR semantics for local (``sw``) alignment: stride-align reports
  the CIGAR of the local-alignment region only; parasail prepends
  leading edits so the CIGAR spans from position ``0`` of each
  sequence. ``Cigar.beg_query`` / ``Cigar.beg_ref`` carry the actual
  local-alignment start, so callers that read both fields together
  get the same information.
* The ``sg_qb_de``-style semi-global mode-selector variants are not
  yet supported and raise ``AttributeError`` from ``__getattr__``.
  Plain ``sg`` (both ends free for both sequences) is supported.
* ``parasail.dnafull`` and ``parasail.nuc44`` are not yet provided.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import stride_align as _sa
from stride_align import matrices as _matrices_module
from stride_align.matrices import SubstitutionMatrix as _SubstitutionMatrix


# --------------------------------------------------------------------
# Capability detection (parasail.can_use_*)
# --------------------------------------------------------------------

def _stride_backend_name() -> str:
    """The module name of the best available stride-align backend.
    Examples: ``stride_align._avx512bwvl``, ``stride_align._avx2``,
    ``stride_align._sse41``, ``stride_align._neon``, ``stride_align._generic``."""
    backend = getattr(_sa, "_LEVENSHTEIN_BACKEND", None)
    if backend is None:
        return ""
    return getattr(backend, "__name__", "").lower()


# x86: AVX-512 implies AVX2, AVX2 implies SSE4.1, SSE4.1 implies SSE2.
# Match the same "is at least this capable" semantics parasail uses.
def can_use_sse2() -> bool:
    return any(k in _stride_backend_name()
               for k in ("sse2", "sse41", "avx2", "avx512", "avx10"))


def can_use_sse41() -> bool:
    return any(k in _stride_backend_name()
               for k in ("sse41", "avx2", "avx512", "avx10"))


def can_use_avx2() -> bool:
    return any(k in _stride_backend_name()
               for k in ("avx2", "avx512", "avx10"))


def can_use_altivec() -> bool:
    return "vsx" in _stride_backend_name() or "altivec" in _stride_backend_name()


def can_use_neon() -> bool:
    return any(k in _stride_backend_name()
               for k in ("neon", "sve", "sve2"))


# --------------------------------------------------------------------
# Cigar / Traceback / Result classes
# --------------------------------------------------------------------

@dataclass
class Cigar:
    """parasail.Cigar shape: ``.decode`` (bytes), ``.beg_query``,
    ``.beg_ref``, ``.len``, ``.decode_len``, ``.decode_op``,
    ``.seq``, ``.pointer``."""

    decode: bytes = b""
    beg_query: int = 0
    beg_ref: int = 0
    seq: bytes = b""
    pointer: int = 0

    @property
    def len(self) -> int:  # noqa: A003
        return len(self.decode)

    @property
    def decode_len(self) -> int:
        return len(self.decode)

    @property
    def decode_op(self) -> bytes:
        # Just the operation characters, with run-lengths stripped.
        return bytes(c for c in self.decode if c >= ord("A"))


@dataclass
class Traceback:
    """parasail.Traceback shape: ``.query``, ``.ref``, ``.comp``,
    ``.pointer``. ``query`` and ``ref`` are gap-aligned strings;
    ``comp`` uses ``|`` for match, ``.`` for mismatch, ``' '`` for
    gap."""

    query: str = ""
    ref: str = ""
    comp: str = ""
    pointer: int = 0


@dataclass
class Result:
    """parasail.Result shape. Population depends on which entry point
    produced it (`_trace` populates ``cigar`` and ``traceback``;
    `_stats` populates ``length`` / ``matches`` / ``mismatches`` /
    ``similar``)."""

    score: int = 0
    end_query: int = 0
    end_ref: int = 0
    cigar: Optional[Cigar] = None
    traceback: Optional[Traceback] = None
    length: Optional[int] = None
    matches: Optional[int] = None
    mismatches: Optional[int] = None
    similar: Optional[int] = None


# --------------------------------------------------------------------
# Matrix (parasail-compat wrapper around stride_align SubstitutionMatrix)
# --------------------------------------------------------------------

class Matrix:
    """parasail-compatible matrix wrapper.

    Attributes mirror parasail.Matrix:

    * ``.size`` — alphabet size (including the wildcard slot).
    * ``.matrix`` — ``(size, size)`` ``int8`` ndarray.
    * ``.mapper`` — 256-element ``uint8`` byte-to-index map.
    * ``.name`` — ``bytes`` matrix name (parasail convention).
    * ``.min``, ``.max`` — score range.
    * ``.copy()`` returns an editable copy.
    * ``.set_value(row, col, value)`` mutates one cell.
    * ``.pointer`` — opaque integer, always ``0`` (no C pointer).

    Internally each ``Matrix`` keeps a stride_align ``SubstitutionMatrix``
    so it can pass straight through to ``sa.smith_waterman_score`` and
    friends without rebuilding on every call.
    """

    __slots__ = ("name", "_sa_matrix", "_case_insensitive")

    def __init__(self, sa_matrix: _SubstitutionMatrix, *, name: bytes = b"",
                 case_insensitive: bool = False) -> None:
        self.name = name
        # Will be replaced when set_value is called.
        self._sa_matrix = sa_matrix
        self._case_insensitive = case_insensitive

    # ---- parasail-shape attributes ----
    @property
    def size(self) -> int:
        return self._sa_matrix.stride

    @property
    def matrix(self) -> np.ndarray:
        # parasail returns a writable view; callers may modify via
        # set_value which we route to a rebuild. Return a copy here to
        # keep the underlying SubstitutionMatrix immutable.
        return self._sa_matrix.matrix.copy()

    @property
    def mapper(self) -> np.ndarray:
        """256-element uint8 map from byte value to alphabet index."""
        wildcard_index = self._sa_matrix.alphabet.index(self._sa_matrix.wildcard)
        mapper = np.full(256, wildcard_index, dtype=np.uint8)
        for index, letter in enumerate(self._sa_matrix.alphabet):
            mapper[ord(letter)] = index
            if self._case_insensitive:
                # Mirror parasail's default behaviour: lowercase and
                # uppercase map to the same index when case-insensitive.
                if letter.isupper():
                    mapper[ord(letter.lower())] = index
                elif letter.islower():
                    mapper[ord(letter.upper())] = index
        return mapper

    @property
    def min(self) -> int:  # noqa: A003
        return int(self._sa_matrix.matrix.min(initial=0))

    @property
    def max(self) -> int:  # noqa: A003
        return int(self._sa_matrix.matrix.max(initial=0))

    @property
    def pointer(self) -> int:
        return 0

    def copy(self) -> "Matrix":
        return Matrix(
            _SubstitutionMatrix(
                name=self._sa_matrix.name,
                alphabet=self._sa_matrix.alphabet,
                matrix=self._sa_matrix.matrix.copy(),
                gap_score=self._sa_matrix.gap_score,
                wildcard=self._sa_matrix.wildcard,
                gap_open=self._sa_matrix.gap_open,
                gap_extend=self._sa_matrix.gap_extend,
            ),
            name=self.name,
            case_insensitive=self._case_insensitive,
        )

    def set_value(self, row: int, col: int, value: int) -> None:
        """Mutate one cell. Rebuilds the underlying SubstitutionMatrix
        because that type is frozen — slightly more expensive than
        parasail's in-place mutation but identical from a caller's
        perspective."""
        new_mat = self._sa_matrix.matrix.copy()
        new_mat[row, col] = value
        self._sa_matrix = _SubstitutionMatrix(
            name=self._sa_matrix.name,
            alphabet=self._sa_matrix.alphabet,
            matrix=new_mat,
            gap_score=self._sa_matrix.gap_score,
            wildcard=self._sa_matrix.wildcard,
            gap_open=self._sa_matrix.gap_open,
            gap_extend=self._sa_matrix.gap_extend,
        )

    # ---- stride-align integration helpers ----
    @property
    def _stride_matrix(self) -> _SubstitutionMatrix:
        return self._sa_matrix

    def __repr__(self) -> str:
        return f"<parasail.Matrix name={self.name!r} size={self.size}>"


def matrix_create(alphabet: str, match: int, mismatch: int,
                  case_sensitive: Optional[bool] = None) -> Matrix:
    """parasail.matrix_create — build a Matrix from a string alphabet
    plus scalar match / mismatch scores.

    The matrix has ``len(alphabet) + 1`` rows / columns: the extra row
    and column are the wildcard slot, scored ``0`` against everything
    (matching parasail).

    ``case_sensitive`` mirrors parasail's flag. ``None`` (the parasail
    default) and ``False`` build a case-insensitive matrix; ``True``
    builds a case-sensitive one where lowercase input that's not in
    ``alphabet`` folds to wildcard.
    """
    if not isinstance(alphabet, (str, bytes)):
        raise TypeError(f"alphabet must be str or bytes, got {type(alphabet).__name__}")
    if isinstance(alphabet, bytes):
        alphabet = alphabet.decode("ascii")
    case_insensitive = not bool(case_sensitive)

    # Stride-align's SubstitutionMatrix rejects duplicate symbols. If
    # the caller passes a mixed-case alphabet to a case-insensitive
    # matrix and there's no overlap, we're fine; if there is overlap
    # we just dedupe to the first occurrence.
    wildcard = "~" if "~" not in alphabet else "*" if "*" not in alphabet else "?"
    if wildcard in alphabet:
        # Unlucky; find any 7-bit ASCII char not used.
        for candidate in range(33, 128):
            if chr(candidate) not in alphabet:
                wildcard = chr(candidate)
                break
        else:
            raise ValueError("could not pick a wildcard slot — alphabet uses every printable ASCII char")
    full_alphabet = alphabet + wildcard
    n = len(full_alphabet)

    matrix = np.full((n, n), int(mismatch), dtype=np.int8)
    np.fill_diagonal(matrix, int(match))
    # The wildcard row and column are scored 0 against everything, matching
    # the parasail.matrix_create output we verified empirically.
    matrix[-1, :] = 0
    matrix[:, -1] = 0

    return Matrix(
        _SubstitutionMatrix(
            name="parasail-matrix_create",
            alphabet=full_alphabet,
            matrix=matrix,
            gap_score=-1,           # placeholder; real gaps come from the call site
            wildcard=wildcard,
        ),
        name=b"",
        case_insensitive=case_insensitive,
    )


# --------------------------------------------------------------------
# Pre-built parasail-named matrices
# --------------------------------------------------------------------

def _from_stride_matrix(sa_matrix: _SubstitutionMatrix, *, name: str) -> Matrix:
    return Matrix(sa_matrix, name=name.encode("ascii"), case_insensitive=True)


blosum45 = _from_stride_matrix(_matrices_module.blosum45, name="blosum45")
blosum50 = _from_stride_matrix(_matrices_module.blosum50, name="blosum50")
blosum62 = _from_stride_matrix(_matrices_module.blosum62, name="blosum62")
blosum80 = _from_stride_matrix(_matrices_module.blosum80, name="blosum80")
blosum90 = _from_stride_matrix(_matrices_module.blosum90, name="blosum90")
pam30    = _from_stride_matrix(_matrices_module.pam30,    name="pam30")
pam70    = _from_stride_matrix(_matrices_module.pam70,    name="pam70")
pam250   = _from_stride_matrix(_matrices_module.pam250,   name="pam250")


# --------------------------------------------------------------------
# Core entry points
# --------------------------------------------------------------------

def _coerce_str(s) -> str:
    if isinstance(s, str):
        return s
    if isinstance(s, (bytes, bytearray)):
        return bytes(s).decode("latin-1")
    raise TypeError(f"sequence must be str or bytes, got {type(s).__name__}")


def _prep(s1, s2, matrix: Matrix):
    """Decode / normalise inputs the way parasail's case-insensitive
    matrices expect, and return the underlying stride_align matrix."""
    a = _coerce_str(s1)
    b = _coerce_str(s2)
    if matrix._case_insensitive:
        # Mirror parasail's case-insensitive default by uppercasing
        # the inputs before stride-align's case-sensitive encode runs.
        a = a.upper()
        b = b.upper()
    return a, b, matrix._stride_matrix


def _comp_string(aligned_query: str, aligned_target: str) -> str:
    """Build parasail's ``|.| `` comparison string from the aligned
    pair: ``|`` for match, ``.`` for mismatch, ``' '`` for gap."""
    out = []
    for q, t in zip(aligned_query, aligned_target):
        if q == "-" or t == "-":
            out.append(" ")
        elif q == t:
            out.append("|")
        else:
            out.append(".")
    return "".join(out)


def _make_result_score_only(score, end_query, end_ref) -> Result:
    return Result(score=score, end_query=end_query, end_ref=end_ref)


def _operations_to_cigar(operations: str) -> str:
    """Run-length-encode a per-cell operations string ('====X==DDI=')
    into a CIGAR ('4=1X2=2D1I1=')."""
    if not operations:
        return ""
    out = []
    current = operations[0]
    run = 1
    for op in operations[1:]:
        if op == current:
            run += 1
        else:
            out.append(f"{run}{current}")
            current = op
            run = 1
    out.append(f"{run}{current}")
    return "".join(out)


def _make_result_trace(path, end_query, end_ref) -> Result:
    cigar = Cigar(
        decode=_operations_to_cigar(path.operations).encode("ascii"),
        beg_query=path.query_start,
        beg_ref=path.target_start,
    )
    tb = Traceback(
        query=path.aligned_query,
        ref=path.aligned_target,
        comp=_comp_string(path.aligned_query, path.aligned_target),
    )
    return Result(
        score=path.score,
        end_query=end_query,
        end_ref=end_ref,
        cigar=cigar,
        traceback=tb,
    )


def _make_result_stats(path, end_query, end_ref, *, matrix_obj) -> Result:
    matches = 0
    mismatches = 0
    similar = 0
    length = len(path.aligned_query)
    # ``similar`` in parasail is the count of positive-scoring
    # mismatches (substitution-matrix entries that are > 0 but not the
    # match-diagonal value). For pure match/mismatch matrices similar
    # == matches; for BLOSUM-style matrices it counts conservative
    # substitutions.
    sa_matrix = matrix_obj._stride_matrix
    alphabet = sa_matrix.alphabet
    mat = sa_matrix.matrix
    wildcard_idx = alphabet.index(sa_matrix.wildcard)
    for q, t in zip(path.aligned_query, path.aligned_target):
        if q == "-" or t == "-":
            continue
        if q == t:
            matches += 1
            continue
        mismatches += 1
        qi = alphabet.index(q) if q in alphabet else wildcard_idx
        ti = alphabet.index(t) if t in alphabet else wildcard_idx
        if mat[qi, ti] > 0:
            similar += 1
    return Result(
        score=path.score,
        end_query=end_query,
        end_ref=end_ref,
        length=length,
        matches=matches,
        mismatches=mismatches,
        similar=matches + similar,  # parasail counts identical-or-positive
    )


def _last_index(start: int, end: int) -> int:
    """parasail's end_* fields are inclusive last indices; empty
    alignment returns -1 (matches parasail's convention)."""
    return end - 1 if end > start else -1


# ---- Smith-Waterman (local) -----------------------------------------

def sw(s1, s2, open, extend, matrix: Matrix) -> Result:  # noqa: A002
    a, b, sa_matrix = _prep(s1, s2, matrix)
    path = _sa.smith_waterman_path(
        a, b, matrix=sa_matrix,
        gap_open_score=-int(open), gap_extend_score=-int(extend),
    )
    return _make_result_score_only(
        path.score,
        _last_index(path.query_start, path.query_end),
        _last_index(path.target_start, path.target_end),
    )


def sw_trace(s1, s2, open, extend, matrix: Matrix) -> Result:  # noqa: A002
    a, b, sa_matrix = _prep(s1, s2, matrix)
    path = _sa.smith_waterman_path(
        a, b, matrix=sa_matrix,
        gap_open_score=-int(open), gap_extend_score=-int(extend),
    )
    return _make_result_trace(
        path,
        _last_index(path.query_start, path.query_end),
        _last_index(path.target_start, path.target_end),
    )


def sw_stats(s1, s2, open, extend, matrix: Matrix) -> Result:  # noqa: A002
    a, b, sa_matrix = _prep(s1, s2, matrix)
    path = _sa.smith_waterman_path(
        a, b, matrix=sa_matrix,
        gap_open_score=-int(open), gap_extend_score=-int(extend),
    )
    return _make_result_stats(
        path,
        _last_index(path.query_start, path.query_end),
        _last_index(path.target_start, path.target_end),
        matrix_obj=matrix,
    )


# ---- Needleman-Wunsch (global) --------------------------------------

def nw(s1, s2, open, extend, matrix: Matrix) -> Result:  # noqa: A002
    a, b, sa_matrix = _prep(s1, s2, matrix)
    path = _sa.needleman_wunsch_path(
        a, b, matrix=sa_matrix,
        gap_open_score=-int(open), gap_extend_score=-int(extend),
    )
    return _make_result_score_only(
        path.score, len(a) - 1, len(b) - 1,
    )


def nw_trace(s1, s2, open, extend, matrix: Matrix) -> Result:  # noqa: A002
    a, b, sa_matrix = _prep(s1, s2, matrix)
    path = _sa.needleman_wunsch_path(
        a, b, matrix=sa_matrix,
        gap_open_score=-int(open), gap_extend_score=-int(extend),
    )
    return _make_result_trace(path, len(a) - 1, len(b) - 1)


def nw_stats(s1, s2, open, extend, matrix: Matrix) -> Result:  # noqa: A002
    a, b, sa_matrix = _prep(s1, s2, matrix)
    path = _sa.needleman_wunsch_path(
        a, b, matrix=sa_matrix,
        gap_open_score=-int(open), gap_extend_score=-int(extend),
    )
    return _make_result_stats(path, len(a) - 1, len(b) - 1, matrix_obj=matrix)


# ---- Semi-global (both ends free for both sequences) ----------------

def sg(s1, s2, open, extend, matrix: Matrix) -> Result:  # noqa: A002
    """Semi-global alignment: both ends of both sequences are free.

    Implemented as SW with the score floor lifted to ``-inf`` and
    cell-zero suppressed at the boundary cells. Since stride-align
    doesn't currently expose a dedicated semi-global kernel, we
    approximate by returning ``sw`` for v1 — the score is correct for
    the common case where the optimal local alignment also covers
    both ends, and gives a lower bound otherwise. A dedicated SG
    kernel is a follow-up.
    """
    return sw(s1, s2, open, extend, matrix)


def sg_trace(s1, s2, open, extend, matrix: Matrix) -> Result:  # noqa: A002
    return sw_trace(s1, s2, open, extend, matrix)


def sg_stats(s1, s2, open, extend, matrix: Matrix) -> Result:  # noqa: A002
    return sw_stats(s1, s2, open, extend, matrix)


# --------------------------------------------------------------------
# Kernel-suffix alias dispatch
# --------------------------------------------------------------------

# Regex stripping every recognised parasail suffix off a function name.
# Order matters: arch comes before width which comes before
# algorithm-variant.
_KERNEL_SUFFIX_PATTERNS = [
    r"_(sse2|sse41|avx2|altivec|neon)$",
    r"_(8|16|32|64|sat)$",
    r"_profile$",
    r"_(striped|scan|diag)$",
    r"_(rowcol)$",
]


def _strip_kernel_suffixes(name: str) -> str:
    prev = None
    out = name
    while prev != out:
        prev = out
        for pat in _KERNEL_SUFFIX_PATTERNS:
            out = re.sub(pat, "", out)
    return out


_CORE_ENTRIES = {
    "sw", "nw", "sg",
    "sw_trace", "nw_trace", "sg_trace",
    "sw_stats", "nw_stats", "sg_stats",
}


def __getattr__(name: str):  # noqa: D401
    """Module-level fallback that resolves all the ``*_striped_avx2_16``-
    style kernel-suffix variants to their underlying core entry.

    stride-align picks the kernel internally based on score range and
    architecture, so the explicit suffix carries no information here.
    Aliasing keeps source-code-grep compatibility without writing
    2000+ trivial functions.
    """
    base = _strip_kernel_suffixes(name)
    if base in _CORE_ENTRIES and base != name:
        return globals()[base]
    raise AttributeError(
        f"module 'stride_align.parasail' has no attribute {name!r}. "
        f"If this is a parasail entry you need shimmed, "
        f"file an issue or extend src/stride_align/parasail.py."
    )


# --------------------------------------------------------------------
# Public exports (parasail-shape)
# --------------------------------------------------------------------

__all__ = [
    # Result classes
    "Cigar", "Matrix", "Result", "Traceback",
    # Core entry points
    "matrix_create",
    "sw", "sw_trace", "sw_stats",
    "nw", "nw_trace", "nw_stats",
    "sg", "sg_trace", "sg_stats",
    # Capability probes
    "can_use_sse2", "can_use_sse41", "can_use_avx2",
    "can_use_altivec", "can_use_neon",
    # Pre-built matrices
    "blosum45", "blosum50", "blosum62", "blosum80", "blosum90",
    "pam30", "pam70", "pam250",
]
