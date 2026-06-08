"""``rapidfuzz.utils`` shim — preprocessing helpers."""

from __future__ import annotations

import re

# Matches a single non-alphanumeric ASCII character. rapidfuzz's
# default_process replaces every such character with a space
# individually (NOT collapsing runs), then lowercases and strips
# leading/trailing whitespace.
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")


def default_process(sentence: str) -> str:
    """Replace each non-alphanumeric ASCII char with a single space,
    lowercase, and strip leading/trailing whitespace. The standard
    preprocessor rapidfuzz applies when
    ``processor=utils.default_process`` is passed to a scoring
    function.

    Note: rapidfuzz does NOT collapse internal whitespace runs —
    ``'Hello, World!'`` becomes ``'hello  world'`` (two spaces). The
    shim matches this behaviour exactly.
    """
    if not isinstance(sentence, str):
        # rapidfuzz coerces bytes via latin-1; mirror that.
        if isinstance(sentence, (bytes, bytearray)):
            sentence = bytes(sentence).decode("latin-1")
        else:
            raise TypeError(
                f"default_process expects str or bytes, got {type(sentence).__name__}"
            )
    return _NON_ALNUM.sub(" ", sentence).lower().strip()


__all__ = ["default_process"]
