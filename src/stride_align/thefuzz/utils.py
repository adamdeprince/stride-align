"""Legacy text preprocessing compatible with ``thefuzz.utils``."""

from __future__ import annotations

import re

# TheFuzz intentionally removes only Latin-1 non-ASCII code points here,
# rather than every non-ASCII character. Keep the table public: upstream's
# module exposes it to star imports and some callers inspect it directly.
translation_table = {i: None for i in range(128, 256)}

_ASCII_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")


def ascii_only(s):
    """Delete characters with code points 128 through 255."""
    return s.translate(translation_table)


def _default_process(sentence) -> str:
    """RapidFuzz-style alphanumeric cleanup without a runtime dependency.

    The native RapidFuzz processor uses Unicode *simple* lowercasing one
    code point at a time. Taking the first character of Python's per-codepoint
    ``lower`` result preserves that behavior for special-casing expansions
    such as ``\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}`` and avoids Python's
    context-sensitive final-sigma transformation.
    """
    if isinstance(sentence, bytes):
        sentence = sentence.decode("latin-1")
    elif not isinstance(sentence, str):
        raise TypeError("sentence must be a String")

    if sentence.isascii():
        return _ASCII_NON_ALNUM.sub(" ", sentence).lower().strip()

    output: list[str] = []
    for char in sentence:
        if char.isalnum():
            lowered = char.lower()
            output.append(lowered[0])
        else:
            output.append(" ")
    return "".join(output).strip()


def full_process(s, force_ascii=False):
    """Keep letters/numbers, lowercase, and trim surrounding whitespace.

    With ``force_ascii=True``, TheFuzz first coerces the input with ``str``
    and deletes code points 128 through 255. This deliberately preserves its
    historical behavior (for example CJK characters are not deleted).
    """
    if force_ascii:
        s = ascii_only(str(s))
    return _default_process(s)


__all__ = ["ascii_only", "full_process", "translation_table"]
