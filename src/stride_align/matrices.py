"""Substitution matrices for protein / nucleotide alignment.

A `SubstitutionMatrix` bundles three things the matrix-mode kernels need:

  * `alphabet`   — the canonical ordering of tokens (e.g. amino-acid one-letter codes).
  * `matrix`     — a contiguous row-major `(N, N)` int8 buffer of substitution scores.
  * `gap_score`  — the default linear gap penalty associated with this matrix.

`SubstitutionMatrix.encode(seq)` converts a human-readable string into a `bytes`
of alphabet indices ready to feed the kernel. Unknown letters fold to the
alphabet's wildcard slot (`X` for the standard protein matrices); a `gap=` kwarg
lets callers specify the wildcard explicitly.

The kernels take the encoded indices, not the original strings — matrix-mode
alignment is bit-cheaper because no per-cell letter comparison happens. The cost
of the `encode` step is amortised over the whole alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubstitutionMatrix:
    name: str
    alphabet: str
    matrix: np.ndarray
    gap_score: int
    wildcard: str = "X"

    def __post_init__(self) -> None:
        if self.matrix.dtype != np.int8:
            raise TypeError(
                f"SubstitutionMatrix.matrix must be int8; got {self.matrix.dtype}"
            )
        n = len(self.alphabet)
        if self.matrix.shape != (n, n):
            raise ValueError(
                f"matrix shape {self.matrix.shape} does not match "
                f"alphabet size {n}"
            )
        if not self.matrix.flags["C_CONTIGUOUS"]:
            raise ValueError("matrix must be C-contiguous")
        if self.wildcard not in self.alphabet:
            raise ValueError(
                f"wildcard {self.wildcard!r} is not in alphabet {self.alphabet!r}"
            )
        if len(set(self.alphabet)) != n:
            raise ValueError(f"alphabet {self.alphabet!r} has duplicate symbols")

    @property
    def stride(self) -> int:
        return len(self.alphabet)

    def encode(self, sequence: str) -> bytes:
        """Map `sequence` to a `bytes` of alphabet indices.

        Letters are uppercased before lookup; anything not in `alphabet`
        becomes the wildcard index. Empty input yields empty bytes.
        """
        if not isinstance(sequence, str):
            raise TypeError(
                f"sequence must be str (got {type(sequence).__name__}); "
                "matrix-mode alignment encodes string sequences into "
                "alphabet indices before scoring."
            )
        if not sequence:
            return b""
        table = self._encode_table
        upper = sequence.upper().encode("ascii", errors="replace")
        return upper.translate(table)

    @property
    def _encode_table(self) -> bytes:
        cached = self.__dict__.get("__encode_table_cache")
        if cached is not None:
            return cached
        wildcard_index = self.alphabet.index(self.wildcard)
        table = bytearray([wildcard_index]) * 256
        for index, letter in enumerate(self.alphabet):
            table[ord(letter)] = index
            table[ord(letter.lower())] = index
        result = bytes(table)
        object.__setattr__(self, "__encode_table_cache", result)
        return result

    def score(self, query: str, target: str) -> int:
        """Score a single (query, target) pair under this matrix.

        Convenience helper that hides the encode + dispatch dance. For
        production code, prefer `stride_align.smith_waterman_score(..., matrix=m)`
        directly so you can pick local vs global yourself.
        """
        # Imported lazily to avoid a circular import: stride_align/__init__.py
        # itself imports this module.
        from stride_align import smith_waterman_score

        return smith_waterman_score(query, target, matrix=self, gap_score=self.gap_score)


# Standard 24-letter protein alphabet, NCBI ordering. Includes the three
# ambiguity codes (B = D|N, Z = E|Q, X = any) and the stop/translation
# terminator (*). This is the same layout the NCBI BLAST distribution uses.
_BLOSUM62_ALPHABET = "ARNDCQEGHILKMFPSTWYVBZX*"

# Row-major BLOSUM62 score matrix, NCBI values (matches BLAST's BLOSUM62).
# Each row corresponds to a letter in _BLOSUM62_ALPHABET; values are int8.
# fmt: off
_BLOSUM62_VALUES = np.array([
    # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   B   Z   X   *
    [  4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4],  # A
    [ -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4],  # R
    [ -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4],  # N
    [ -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4],  # D
    [  0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4],  # C
    [ -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4],  # Q
    [ -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4],  # E
    [  0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4],  # G
    [ -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4],  # H
    [ -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4],  # I
    [ -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4],  # L
    [ -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4],  # K
    [ -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4],  # M
    [ -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4],  # F
    [ -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4],  # P
    [  1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4],  # S
    [  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4],  # T
    [ -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4],  # W
    [ -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4],  # Y
    [  0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4],  # V
    [ -2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4],  # B
    [ -1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4],  # Z
    [  0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4],  # X
    [ -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1],  # *
], dtype=np.int8)
# fmt: on


blosum62 = SubstitutionMatrix(
    name="BLOSUM62",
    alphabet=_BLOSUM62_ALPHABET,
    matrix=_BLOSUM62_VALUES,
    gap_score=-4,
    wildcard="X",
)


__all__ = ["SubstitutionMatrix", "blosum62"]
