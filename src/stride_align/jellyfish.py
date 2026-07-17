"""Jellyfish-compatible facade backed by :mod:`stride_align`.

Use this module as a migration shim::

    import stride_align.jellyfish as jellyfish

The distance and Jaro families dispatch to stride-align's native kernels.
Jellyfish measures those algorithms over Unicode grapheme clusters rather
than Python code points, so strings containing combining sequences, emoji
ZWJ sequences, or regional-indicator pairs are segmented before being sent
through stride-align's general hashable-token path.

The phonetic compatibility helpers reproduce Jellyfish's rule variants.
Those variants intentionally differ from stride-align's primary phonetic
API on punctuation, non-ASCII input, and several historical Metaphone,
NYSIIS, and Match Rating edge cases.

The compatibility rule variants are adapted from Jellyfish 1.2.1
(Copyright 2015 James Turk), distributed under the MIT License. The full
license notice is retained in this project's ``NOTICE`` file.
"""

from __future__ import annotations

import operator
import unicodedata
from collections.abc import Sequence

import stride_align as _sa

type _Tokens = str | tuple[str, ...]


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"argument {name!r}: {type(value).__name__!r} object cannot be converted to str"
        )
    return value


# ---------------------------------------------------------------------------
# Unicode grapheme handling
# ---------------------------------------------------------------------------

_ZWJ = "\u200d"
_ZWNJ = "\u200c"


def _is_regional_indicator(character: str) -> bool:
    return "\U0001f1e6" <= character <= "\U0001f1ff"


def _is_emoji_modifier(character: str) -> bool:
    return "\U0001f3fb" <= character <= "\U0001f3ff"


def _is_variation_selector(character: str) -> bool:
    codepoint = ord(character)
    return 0xFE00 <= codepoint <= 0xFE0F or 0xE0100 <= codepoint <= 0xE01EF


def _is_tag_character(character: str) -> bool:
    return 0xE0020 <= ord(character) <= 0xE007F


def _is_extend(character: str) -> bool:
    return (
        unicodedata.category(character) in {"Mn", "Me"}
        or character == _ZWNJ
        or _is_emoji_modifier(character)
        or _is_variation_selector(character)
        or _is_tag_character(character)
    )


def _is_spacing_mark(character: str) -> bool:
    return unicodedata.category(character) == "Mc"


def _is_control(character: str) -> bool:
    if character in {_ZWJ, _ZWNJ} or _is_prepend(character):
        return False
    return unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}


def _is_prepend(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x0600 <= codepoint <= 0x0605
        or codepoint == 0x06DD
        or codepoint == 0x070F
        or 0x0890 <= codepoint <= 0x0891
        or codepoint == 0x08E2
        or codepoint == 0x0D4E
        or codepoint == 0x110BD
        or codepoint == 0x110CD
        or 0x111C2 <= codepoint <= 0x111C3
        or codepoint == 0x1193F
        or codepoint == 0x11941
        or codepoint == 0x11A3A
        or 0x11A84 <= codepoint <= 0x11A89
        or codepoint == 0x11D46
    )


def _hangul_class(character: str) -> str | None:
    codepoint = ord(character)
    if 0x1100 <= codepoint <= 0x115F or 0xA960 <= codepoint <= 0xA97C:
        return "L"
    if 0x1160 <= codepoint <= 0x11A7 or 0xD7B0 <= codepoint <= 0xD7C6:
        return "V"
    if 0x11A8 <= codepoint <= 0x11FF or 0xD7CB <= codepoint <= 0xD7FB:
        return "T"
    if 0xAC00 <= codepoint <= 0xD7A3:
        return "LV" if (codepoint - 0xAC00) % 28 == 0 else "LVT"
    return None


def _is_extended_pictographic(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x2300 <= codepoint <= 0x23FF
        or 0x2600 <= codepoint <= 0x27BF
        or 0x2B00 <= codepoint <= 0x2BFF
        or 0x1F000 <= codepoint <= 0x1FAFF
    )


def _is_virama(character: str) -> bool:
    name = unicodedata.name(character, "")
    return "VIRAMA" in name or "HALANT" in name or "COENG" in name


def _links_after_virama(previous: str, current: str, following: str) -> bool:
    if not _is_virama(previous):
        return False
    virama_name = unicodedata.name(previous, "")
    following_name = unicodedata.name(following, "")
    if not virama_name or not following_name:
        return False
    script = virama_name.split(maxsplit=1)[0]
    if not following_name.startswith(f"{script} ") or "LETTER" not in following_name:
        return False
    return any(
        item.isalpha() and unicodedata.name(item, "").startswith(f"{script} ")
        for item in current[:-1]
    )


def _needs_grapheme_segmentation(value: str) -> bool:
    if value.isascii():
        return "\r\n" in value
    if "\r\n" in value:
        return True
    for character in value:
        if (
            _is_extend(character)
            or _is_spacing_mark(character)
            or character == _ZWJ
            or _is_regional_indicator(character)
            or _is_prepend(character)
            or _hangul_class(character) is not None
            or _is_virama(character)
        ):
            return True
    return False


def _graphemes(value: str) -> tuple[str, ...]:
    """Split the cluster forms Jellyfish treats as single symbols.

    This follows the extended-grapheme rules needed by current Python
    text in practice: combining/spacing marks, Hangul composition,
    emoji modifiers and ZWJ sequences, regional-indicator pairs,
    prepend characters, CRLF, and virama-linked conjuncts.
    """
    if not value:
        return ()

    clusters: list[str] = []
    current = value[0]
    previous = value[0]
    regional_run = 1 if _is_regional_indicator(previous) else 0

    for character in value[1:]:
        previous_hangul = _hangul_class(previous)
        current_hangul = _hangul_class(character)
        join = False

        if previous == "\r" and character == "\n":
            join = True
        elif _is_prepend(previous) and not _is_control(character):
            join = True
        elif _is_control(previous) or _is_control(character):
            join = False
        elif previous_hangul == "L" and current_hangul in {"L", "V", "LV", "LVT"}:
            join = True
        elif previous_hangul in {"LV", "V"} and current_hangul in {"V", "T"}:
            join = True
        elif previous_hangul in {"LVT", "T"} and current_hangul == "T":
            join = True
        elif _is_extend(character) or _is_spacing_mark(character) or character == _ZWJ:
            join = True
        elif previous == _ZWJ and _is_extended_pictographic(character):
            join = True
        elif _links_after_virama(previous, current, character):
            join = True
        elif _is_regional_indicator(previous) and _is_regional_indicator(character):
            join = regional_run % 2 == 1

        if join:
            current += character
        else:
            clusters.append(current)
            current = character

        if _is_regional_indicator(character):
            regional_run = regional_run + 1 if _is_regional_indicator(previous) else 1
        else:
            regional_run = 0
        previous = character

    clusters.append(current)
    return tuple(clusters)


def _tokenize_pair(a: str, b: str) -> tuple[_Tokens, _Tokens]:
    if not (_needs_grapheme_segmentation(a) or _needs_grapheme_segmentation(b)):
        return a, b
    return _graphemes(a), _graphemes(b)


# ---------------------------------------------------------------------------
# Distance and similarity functions
# ---------------------------------------------------------------------------


def levenshtein_distance(a: str, b: str) -> int:
    """Calculate the unit-cost Levenshtein distance between two strings."""
    left = _require_str(a, "a")
    right = _require_str(b, "b")
    left_tokens, right_tokens = _tokenize_pair(left, right)
    return int(_sa.levenshtein_score(left_tokens, right_tokens))


def damerau_levenshtein_distance(a: str, b: str) -> int:
    """Calculate unrestricted Damerau-Levenshtein distance."""
    left = _require_str(a, "a")
    right = _require_str(b, "b")
    left_tokens, right_tokens = _tokenize_pair(left, right)
    return int(_sa.true_damerau_levenshtein_score(left_tokens, right_tokens))


def hamming_distance(a: str, b: str) -> int:
    """Count positional differences, padding the shorter string with mismatches."""
    left = _require_str(a, "a")
    right = _require_str(b, "b")
    left_tokens, right_tokens = _tokenize_pair(left, right)
    common_length = min(len(left_tokens), len(right_tokens))
    distance = abs(len(left_tokens) - len(right_tokens))
    if common_length:
        distance += int(
            _sa.hamming_score(
                left_tokens[:common_length],
                right_tokens[:common_length],
            )
        )
    return distance


def jaccard_similarity(a: str, b: str, ngram_size: int | None = None) -> float:
    """Calculate Jellyfish's set Jaccard similarity.

    With ``ngram_size=None``, tokens are whitespace-separated words.
    With an integer size, tokens are non-overlapping character chunks;
    a final short chunk is retained.
    """
    left = _require_str(a, "a")
    right = _require_str(b, "b")

    if ngram_size is None:
        left_grams = set(left.split())
        right_grams = set(right.split())
    else:
        try:
            size = operator.index(ngram_size)
        except TypeError:
            raise TypeError(
                f"argument 'ngram_size': {type(ngram_size).__name__!r} object "
                "cannot be interpreted as an integer"
            ) from None
        if size < 0:
            raise OverflowError("can't convert negative int to unsigned")
        if size == 0:
            raise ValueError("ngram_size must be greater than zero")
        left_grams = {left[index : index + size] for index in range(0, len(left), size)}
        right_grams = {right[index : index + size] for index in range(0, len(right), size)}

    union = left_grams | right_grams
    if not union:
        return 0.0
    return len(left_grams & right_grams) / len(union)


def jaro_similarity(a: str, b: str) -> float:
    """Calculate Jaro similarity on Jellyfish grapheme tokens."""
    left = _require_str(a, "a")
    right = _require_str(b, "b")
    if not left or not right:
        return 0.0
    left_tokens, right_tokens = _tokenize_pair(left, right)
    return float(_sa.jaro_similarity(left_tokens, right_tokens))


def _common_prefix_length(left: Sequence[str] | str, right: Sequence[str] | str) -> int:
    length = 0
    for left_item, right_item in zip(left[:4], right[:4], strict=False):
        if left_item != right_item:
            break
        length += 1
    return length


def _jaro_common_count(left: Sequence[str] | str, right: Sequence[str] | str) -> int:
    search_range = max(len(left), len(right)) // 2
    search_range = max(0, search_range - 1)
    right_flags = [False] * len(right)
    common = 0
    for index, item in enumerate(left):
        low = max(0, index - search_range)
        high = min(index + search_range, len(right) - 1)
        for target_index in range(low, high + 1):
            if not right_flags[target_index] and right[target_index] == item:
                right_flags[target_index] = True
                common += 1
                break
    return common


def jaro_winkler_similarity(
    a: str,
    b: str,
    long_tolerance: bool | None = None,
) -> float:
    """Calculate Jaro-Winkler similarity, including Jellyfish long tolerance."""
    left = _require_str(a, "a")
    right = _require_str(b, "b")
    if long_tolerance is not None and not isinstance(long_tolerance, bool):
        raise TypeError(
            f"argument 'long_tolerance': {type(long_tolerance).__name__!r} "
            "object cannot be converted to bool"
        )
    if not left or not right:
        return 0.0

    left_tokens, right_tokens = _tokenize_pair(left, right)
    score = float(_sa.jaro_similarity(left_tokens, right_tokens))
    if score <= 0.7:
        return score

    prefix_length = _common_prefix_length(left_tokens, right_tokens)
    if prefix_length:
        score += prefix_length * 0.1 * (1.0 - score)

    if long_tolerance:
        common = _jaro_common_count(left_tokens, right_tokens)
        minimum_length = min(len(left_tokens), len(right_tokens))
        if (
            minimum_length > 4
            and common > prefix_length + 1
            and 2 * common >= minimum_length + prefix_length
        ):
            score += (1.0 - score) * (
                (common - prefix_length - 1)
                / (len(left_tokens) + len(right_tokens) - 2 * prefix_length + 2)
            )
    return score


# ---------------------------------------------------------------------------
# Jellyfish phonetic variants
# ---------------------------------------------------------------------------

_SOUNDEX_REPLACEMENTS = {
    **dict.fromkeys("BFPV", "1"),
    **dict.fromkeys("CGJKQSXZ", "2"),
    **dict.fromkeys("DT", "3"),
    "L": "4",
    **dict.fromkeys("MN", "5"),
    "R": "6",
}


def soundex(a: str) -> str:
    """Calculate Jellyfish's four-character Soundex code."""
    value = _require_str(a, "a")
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value.upper())
    result = [normalized[0]]
    previous_code = _SOUNDEX_REPLACEMENTS.get(normalized[0])
    for character in normalized[1:]:
        code = _SOUNDEX_REPLACEMENTS.get(character)
        if code is not None:
            if code != previous_code:
                result.append(code)
                if len(result) == 4:
                    break
            previous_code = code
        elif character not in "HW":
            previous_code = None
    result.extend("0" for _ in range(4 - len(result)))
    return "".join(result)


def _is_vowel(character: str | None) -> bool:
    return character is not None and character in "AEIOU"


def metaphone(a: str) -> str:
    """Calculate Jellyfish's Metaphone encoding."""
    value = _require_str(a, "a")
    if not value:
        return ""

    upper = value.upper()
    characters = list(unicodedata.normalize("NFKD", upper))
    if upper.startswith(("KN", "GN", "PN", "WR", "AE")):
        characters.pop(0)

    result: list[str] = []
    index = 0
    while index < len(characters):
        character = characters[index]
        previous = characters[index - 1] if index else None
        following = characters[index + 1] if index + 1 < len(characters) else "*"
        after_following = characters[index + 2] if index + 2 < len(characters) else "*"

        if character == following and character != "C":
            index += 1
            continue

        if _is_vowel(character):
            if index == 0 or previous == " ":
                result.append(character)
        elif character == "B":
            if (index == 0 or previous != "M") or following != "*":
                result.append("B")
        elif character == "C":
            if (following == "I" and after_following == "A") or following == "H":
                result.append("X")
                index += 1
            elif following in "IEY":
                result.append("S")
                index += 1
            else:
                result.append("K")
        elif character == "D":
            if following == "G" and after_following in "IEY":
                result.append("J")
                index += 2
            else:
                result.append("T")
        elif character in "FJLMNR":
            result.append(character)
        elif character == "G":
            if following in "IEY":
                result.append("J")
            elif (
                following == "H" and after_following != "*" and not _is_vowel(after_following)
            ) or (following == "N" and after_following == "*"):
                index += 1
            else:
                result.append("K")
        elif character == "H":
            if index == 0 or _is_vowel(following) or not _is_vowel(previous):
                result.append("H")
        elif character == "K":
            if index == 0 or previous != "C":
                result.append("K")
        elif character == "P":
            if following == "H":
                result.append("F")
                index += 1
            else:
                result.append("P")
        elif character == "Q":
            result.append("K")
        elif character == "S":
            if following == "H":
                result.append("X")
                index += 1
            elif following == "I" and after_following in "OA":
                result.append("X")
                index += 2
            else:
                result.append("S")
        elif character == "T":
            if following == "I" and after_following in "OA":
                result.append("X")
            elif following == "H":
                result.append("0")
                index += 1
            elif following != "C" or after_following != "H":
                result.append("T")
        elif character == "V":
            result.append("F")
        elif character == "W":
            if index == 0 and following == "H":
                index += 1
                result.append("W")
            elif _is_vowel(following):
                result.append("W")
        elif character == "X":
            if index == 0:
                if following == "H" or (following == "I" and after_following in "OA"):
                    result.append("X")
                else:
                    result.append("S")
            else:
                result.extend(("K", "S"))
        elif character == "Y":
            if _is_vowel(following):
                result.append("Y")
        elif character == "Z":
            result.append("S")
        elif character == " " and result and result[-1] != " ":
            result.append(" ")

        index += 1

    return "".join(result)


def nysiis(a: str) -> str:
    """Calculate Jellyfish's NYSIIS encoding."""
    value = _require_str(a, "a")
    if not value:
        return ""

    upper = value.upper()
    characters = list(_graphemes(upper))
    if upper.startswith("MAC"):
        characters[1] = "C"
    elif upper.startswith("KN"):
        characters.pop(0)
    elif upper.startswith("K"):
        characters[0] = "C"
    elif upper.startswith(("PH", "PF")):
        characters[0:2] = ["F", "F"]
    elif upper.startswith("SCH"):
        characters[0:3] = ["S", "S", "S"]

    if upper.endswith(("IE", "EE")):
        characters[-2:] = ["Y"]
    elif upper.endswith(("DT", "RT", "RD", "NT", "ND")):
        characters[-2:] = ["D"]

    key = [characters[0]]
    index = 1
    while index < len(characters):
        character = characters[index]
        translated: list[str]
        if character == "E" and index + 1 < len(characters) and characters[index + 1] == "V":
            translated = ["A", "F"]
            index += 1
        elif _is_vowel(character):
            translated = ["A"]
        elif character == "Q":
            translated = ["G"]
        elif character == "Z":
            translated = ["S"]
        elif character == "M":
            translated = ["N"]
        elif character == "K":
            translated = (
                ["N"] if index + 1 < len(characters) and characters[index + 1] == "N" else ["C"]
            )
        elif (
            character == "S"
            and index + 2 < len(characters)
            and characters[index + 1 : index + 3] == ["C", "H"]
        ):
            translated = ["S", "S"]
            index += 2
        elif character == "P" and index + 1 < len(characters) and characters[index + 1] == "H":
            translated = ["F"]
            index += 1
        elif character == "H" and (
            not _is_vowel(characters[index - 1])
            or (index + 1 < len(characters) and not _is_vowel(characters[index + 1]))
            or index + 1 == len(characters)
        ):
            translated = ["A"] if _is_vowel(characters[index - 1]) else [characters[index - 1]]
        elif character == "W" and _is_vowel(characters[index - 1]):
            translated = [characters[index - 1]]
        else:
            translated = [character]

        if translated and translated[-1] != key[-1]:
            key.extend(translated)
        index += 1

    if len(key) > 1 and key[-1] == "S":
        key.pop()
    if key[-2:] == ["A", "Y"]:
        key[-2:] = ["Y"]
    if len(key) > 1 and key[-1] == "A":
        key.pop()
    return "".join(key)


def _match_rating_codex(value: str) -> str:
    upper = value.upper()
    if not all(character.isalpha() or character == " " for character in upper):
        raise ValueError("Strings must only contain alphabetical characters")

    characters = _graphemes(upper)
    codex: list[str] = []
    previous = "~tmp~"
    for index, character in enumerate(characters):
        vowel = character in "AEIOU"
        if (character != " " and index == 0 and vowel) or (not vowel and character != previous):
            codex.append(character)
        previous = character

    result = "".join(codex)
    if len(result.encode("utf-8")) > 6:
        codepoints = list(result)
        result = "".join(codepoints[:3] + codepoints[-3:])
    return result


def match_rating_codex(a: str) -> str:
    """Calculate Jellyfish's Match Rating Approach codex."""
    return _match_rating_codex(_require_str(a, "a"))


def match_rating_comparison(a: str, b: str) -> bool | None:
    """Compare two strings with Match Rating Approach.

    Jellyfish returns ``None`` when either input is invalid or when the
    codex lengths differ by at least three bytes.
    """
    left = _require_str(a, "a")
    right = _require_str(b, "b")
    try:
        left_codex = _match_rating_codex(left)
        right_codex = _match_rating_codex(right)
    except ValueError:
        return None

    if len(left_codex.encode("utf-8")) > len(right_codex.encode("utf-8")):
        longer, shorter = left_codex, right_codex
    else:
        longer, shorter = right_codex, left_codex

    longer_length = len(longer.encode("utf-8"))
    shorter_length = len(shorter.encode("utf-8"))
    if longer_length - shorter_length >= 3:
        return None

    left_remainder: list[str] = []
    right_remainder: list[str] = []
    left_chars = list(longer)
    right_chars = list(shorter)
    shared = min(len(left_chars), len(right_chars))
    for index in range(shared):
        if left_chars[index] != right_chars[index]:
            left_remainder.append(left_chars[index])
            right_remainder.append(right_chars[index])
    left_remainder.extend(left_chars[shared:])
    right_remainder.extend(right_chars[shared:])

    unmatched_left = 0
    unmatched_right = 0
    shared_remainder = min(len(left_remainder), len(right_remainder))
    for index in range(1, shared_remainder + 1):
        if left_remainder[-index] != right_remainder[-index]:
            unmatched_left += 1
            unmatched_right += 1
    unmatched_left += len(left_remainder) - shared_remainder
    unmatched_right += len(right_remainder) - shared_remainder

    score = 6 - max(unmatched_left, unmatched_right)
    length_sum = longer_length + shorter_length
    if length_sum <= 4:
        minimum = 5
    elif length_sum <= 7:
        minimum = 4
    elif length_sum <= 11:
        minimum = 3
    else:
        minimum = 2
    return score >= minimum


__all__ = [
    "damerau_levenshtein_distance",
    "hamming_distance",
    "jaccard_similarity",
    "jaro_similarity",
    "jaro_winkler_similarity",
    "levenshtein_distance",
    "match_rating_codex",
    "match_rating_comparison",
    "metaphone",
    "nysiis",
    "soundex",
]
