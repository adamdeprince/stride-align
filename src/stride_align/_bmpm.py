"""Beider-Morse Phonetic Matching wrapper.

Public entry point is :func:`beider_morse`. The first call lazily reads the
rule files shipped under :mod:`stride_align.bmpm_data` via
:mod:`importlib.resources` and registers them with the C++ engine living in
the ``_generic`` backend module. Subsequent calls go straight to the engine.

GENERIC name-type only — Ashkenazi and Sephardic rule sets are not shipped.
"""

from __future__ import annotations

import enum
import threading
from importlib.resources import files


class BmpmRuleType(enum.IntEnum):
    """Selects the BMPM rule family used by :func:`beider_morse`.

    Integer values match the upstream Apache Commons Codec ``RuleType``
    ordering (APPROX = 0, EXACT = 1); they cross the C++ boundary as
    plain ints via :class:`enum.IntEnum`.
    """

    APPROX = 0
    """Broader phonetic spread. Default."""

    EXACT = 1
    """Tighter phonetic spread — fewer alternative pronunciations."""


_REGISTER_LOCK = threading.Lock()
_REGISTERED = False
_BACKEND = None  # populated by stride_align.__init__ at import time


def _set_backend(backend) -> None:
    """Hook used by :mod:`stride_align.__init__` to inject the generic backend."""
    global _BACKEND
    _BACKEND = backend


def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    with _REGISTER_LOCK:
        if _REGISTERED:
            return
        if _BACKEND is None:
            raise RuntimeError(
                "stride_align BMPM backend not initialised — the generic "
                "backend module did not load"
            )
        if not hasattr(_BACKEND, "_bmpm_register_resources"):
            raise RuntimeError(
                "stride_align was built without BMPM support — the C++ "
                "extension was compiled without STRIDE_ALIGN_BMPM_BACKEND"
            )
        bmpm_root = files("stride_align") / "bmpm_data"
        resources: dict[str, str] = {}
        for entry in bmpm_root.iterdir():
            name = entry.name
            if not name.endswith(".txt"):
                continue
            resources[name[: -len(".txt")]] = entry.read_text(encoding="utf-8")
        if not resources:
            raise RuntimeError(
                "stride_align BMPM rule files are missing under "
                "stride_align/bmpm_data/ — the wheel is incomplete"
            )
        _BACKEND._bmpm_register_resources(resources)
        _REGISTERED = True


def beider_morse(
    s: object,
    *,
    rule_type: BmpmRuleType | int = BmpmRuleType.APPROX,
    concat: bool = True,
    max_phonemes: int = 20,
) -> str:
    """Beider-Morse Phonetic Matching encoding (Beider & Morse, 2008).

    Returns a ``|``-separated string of phonetic codes capturing plausible
    pronunciations of ``s`` across European languages. An exact-match search
    against the BMPM codes of a candidate list surfaces names spelled
    differently but pronounced similarly.

    Parameters
    ----------
    s : str or bytes
        Input name or word. ``str`` is re-encoded as UTF-8; ``bytes`` is
        passed through (caller is responsible for the encoding).
    rule_type : BmpmRuleType or int, keyword-only
        :attr:`BmpmRuleType.APPROX` (default) for the broader phonetic
        spread; :attr:`BmpmRuleType.EXACT` for a tighter one.
    concat : bool, keyword-only
        When ``True`` (default), multi-word names encode jointly. When
        ``False``, each word encodes separately and per-word codes are
        joined with ``-``.
    max_phonemes : int, keyword-only
        Cap on the PhonemeBuilder set size; default 20 matches upstream
        Apache Commons Codec.

    Notes
    -----
    Stride-align ships the **GENERIC** name-type rules only — the broad
    general-purpose rule set. The Ashkenazi and Sephardic rule sets from
    the upstream Apache Commons Codec distribution are not included.
    """
    _ensure_registered()
    return _BACKEND.beider_morse(
        s,
        rule_type=int(rule_type),
        concat=concat,
        max_phonemes=int(max_phonemes),
    )
