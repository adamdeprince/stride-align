# Phonetic encoders

Hand-vectorised implementations of the standard phonetic codecs —
**Soundex**, **Metaphone** (two variants), **Double Metaphone**
(two variants), **NYSIIS**, **Match Rating Approach (MRA)**,
**Caverphone 2**, **Cologne Phonetic**, **Daitch-Mokotoff
Soundex**, and **Beider-Morse Phonetic Matching**. Bit-for-bit
agreement with the reference implementation in each case; the
variant kwarg picks which spec where multiple ports diverge.

## Encoder / matcher pairs

Each encoder family ships in two flavours: an `encode(name)` that
returns the phonetic key as a string, and an
`equal(a, b)` (or `compare(a, b)` for MRA) that returns whether
two names encode to the same key, with the engine avoiding the
allocation. Use the second when you only care about the match.

| Family | Encoder | Matcher |
| --- | --- | --- |
| Soundex | `soundex` | `soundex_equal` |
| Metaphone | `metaphone` | `metaphone_equal` |
| Double Metaphone | `double_metaphone` | (use Python `==` on the two-element output) |
| NYSIIS | `nysiis` | `nysiis_equal` |
| MRA | `match_rating_codex` | `match_rating_compare` (returns the MRA *similarity*, not just equality) |
| Caverphone 2 | `caverphone` | (use Python `==` on `caverphone` outputs) |
| Cologne Phonetic | `cologne_phonetic` | (use `==`) |
| Daitch-Mokotoff | `daitch_mokotoff` | (use `==` on the multi-code output set) |
| Beider-Morse | `beider_morse` | (use `==` on the multi-code output set) |

## Variant-selecting enums

### `MetaphoneVariant`

```python
from stride_align import metaphone, MetaphoneVariant

metaphone("school", MetaphoneVariant.PHILIPS)    # 'SXKL'  (Apache Commons branch)
metaphone("school", MetaphoneVariant.JELLYFISH)  # 'SXKL'  (jellyfish branch)
metaphone("rough",  MetaphoneVariant.PHILIPS)    # 'RF'    (Philips: GH silent)
metaphone("rough",  MetaphoneVariant.JELLYFISH)  # 'RFKH'  (jellyfish: GH → KH at end)
```

- `PHILIPS` (default) — Lawrence Philips' 1990 spec, matches Apache
  Commons Codec.
- `JELLYFISH` — matches the `jellyfish` PyPI package (diverges on
  `SCH` and word-final `GH`).

### `DoubleMetaphoneVariant`

```python
from stride_align import double_metaphone, DoubleMetaphoneVariant

double_metaphone("Hugh", DoubleMetaphoneVariant.COMMONS)  # ('H',  '')
double_metaphone("Hugh", DoubleMetaphoneVariant.PYTHON)   # ('HH', '')  — bug-for-bug
```

- `COMMONS` (default) — Apache Commons Codec / `doublemetaphone`
  PyPI package. Correct per the Philips spec.
- `PYTHON` — bug-for-bug compatibility with the `metaphone` PyPI
  package, which has a missing `else` in its GH branch that lets
  the previous character's emission leak.

## Examples

```python
import stride_align as sa

sa.soundex("Robert")             # 'R163'
sa.soundex_equal("Robert", "Rupert")  # True

sa.nysiis("Knuth")               # 'NAT'
sa.metaphone("Schwarzenegger")   # 'SWRSNKR'
sa.caverphone("Catherine")       # 'KTRN111111'
sa.cologne_phonetic("Müller")    # '657'   (Latin-1 / Unicode-aware)

# Daitch-Mokotoff and Beider-Morse return MULTIPLE codes for a name
# (one per pronunciation branch). Compare via set intersection:
codes_a = set(sa.daitch_mokotoff("Goldberg").split())
codes_b = set(sa.daitch_mokotoff("Goldburg").split())
match = bool(codes_a & codes_b)

# Beider-Morse with rule-family selection
from stride_align import beider_morse, BmpmRuleType
codes = beider_morse("Müller", rule_type=BmpmRuleType.GENERIC)
```

## Unicode behaviour

Soundex and NYSIIS are pure-ASCII algorithms by definition;
non-ASCII codepoints are skipped. Metaphone, Double Metaphone,
Caverphone, MRA, and Daitch-Mokotoff follow their reference
implementations on the ASCII subset.

The encoders that **were designed for non-ASCII text** —
**Cologne Phonetic** (German) and **Beider-Morse** (Jewish names,
many rule families) — accept the full Unicode codepoint range
through the UCS-1 / UCS-2 / UCS-4 zero-copy bridge and handle
diacritics natively. No `.encode('utf-8')` round-trip needed.

## When to use which

| Workload | Recommended |
| --- | --- |
| US names, fast hash bucket | Soundex |
| English-language record linkage | Metaphone or Double Metaphone |
| Mixed-locale name matching | Double Metaphone (better at silent letters) |
| Census-style name lookup | NYSIIS |
| Personalisation / typing-error tolerance | MRA |
| German names | Cologne Phonetic |
| Jewish / Slavic / mixed-European genealogy | Beider-Morse Phonetic Matching |
| Pre-filter for a slower edit-distance pass | any of the above followed by `levenshtein_normalized_best` |
