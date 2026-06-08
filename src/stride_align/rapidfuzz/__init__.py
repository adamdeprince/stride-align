"""rapidfuzz-compatible shim — drop-in replacement for the rapidfuzz
Python package.

Usage: replace ``import rapidfuzz`` with
``import stride_align.rapidfuzz as rapidfuzz`` and most rapidfuzz code
keeps working unchanged. Submodules mirror upstream:

* ``stride_align.rapidfuzz.fuzz`` — token-ratio family
  (``ratio``, ``partial_ratio``, ``token_sort_ratio``,
  ``token_set_ratio``, ``partial_token_sort_ratio``,
  ``partial_token_set_ratio``, ``token_ratio``,
  ``partial_token_ratio``, ``WRatio``, ``QRatio``). All return values
  in ``[0, 100]`` matching rapidfuzz's convention (stride-align's
  own functions live in the main namespace and return ``[0, 1]``).

* ``stride_align.rapidfuzz.distance`` — ``Levenshtein``, ``Indel``,
  ``Jaro``, ``JaroWinkler``, ``Hamming``, ``DamerauLevenshtein``,
  ``OSA``, ``LCSseq`` plus ``Editop`` / ``Editops`` / ``Opcode`` /
  ``Opcodes`` / ``MatchingBlock`` / ``ScoreAlignment`` collection
  classes. Each metric exposes ``distance`` / ``normalized_distance``
  / ``similarity`` / ``normalized_similarity``; ``Levenshtein`` and
  ``Indel`` add ``editops`` and ``opcodes``.

* ``stride_align.rapidfuzz.process`` — ``extract``, ``extractOne``,
  ``extract_iter``, ``cdist`` with the same kwargs as upstream
  (``scorer=``, ``processor=``, ``score_cutoff=``, ``limit=``,
  ``workers=``).

* ``stride_align.rapidfuzz.utils`` — ``default_process`` (lowercase,
  strip non-alphanumeric, collapse whitespace).

Known divergences from upstream rapidfuzz:

* ``score_hint=``, ``scorer_kwargs=``, ``score_multiplier=`` and
  ``dtype=`` kwargs are accepted but ignored; stride-align picks
  the kernel internally and always returns ``float64`` /
  ``int64``.
* ``fuzz.partial_ratio_alignment`` is not yet implemented (returns
  ``ScoreAlignment(score, src_start, src_end, dest_start, dest_end)``
  upstream; stride-align would derive this from the alignment path —
  follow-up).
* ``Indel.editops`` / ``Indel.opcodes`` are not yet implemented.
* ``process.extract`` with a custom callable ``scorer`` runs a
  Python loop rather than dispatching to a C++ kernel; the built-in
  scorers route to the stride-align fast path.
"""

from __future__ import annotations

from stride_align.rapidfuzz import distance, fuzz, process, utils

__all__ = ["distance", "fuzz", "process", "utils"]
