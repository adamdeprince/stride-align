# jellyfish compatibility shim

`stride_align.jellyfish` is a work-alike facade for the public
`jellyfish` Python API. Replace the import and keep the function calls:

```python
# import jellyfish
import stride_align.jellyfish as jellyfish

jellyfish.levenshtein_distance("kitten", "sitting")  # 3
jellyfish.soundex("Washington")                       # 'W252'
```

The facade tracks Jellyfish 1.2.1 and exposes its eleven public
functions:

| Jellyfish function | Implementation |
| --- | --- |
| `levenshtein_distance(a, b)` | stride-align native Levenshtein kernel |
| `damerau_levenshtein_distance(a, b)` | native unrestricted Damerau-Levenshtein kernel |
| `hamming_distance(a, b)` | native Hamming kernel plus unequal-length padding |
| `jaro_similarity(a, b)` | native Jaro kernel |
| `jaro_winkler_similarity(a, b, long_tolerance=None)` | native Jaro plus Jellyfish prefix/long-string adjustment |
| `jaccard_similarity(a, b, ngram_size=None)` | Jellyfish-compatible set and chunk adapter |
| `soundex(a)` | Jellyfish phonetic rule variant |
| `metaphone(a)` | Jellyfish phonetic rule variant |
| `nysiis(a)` | Jellyfish phonetic rule variant |
| `match_rating_codex(a)` | Jellyfish Match Rating codex |
| `match_rating_comparison(a, b)` | Jellyfish tri-state comparison (`True`, `False`, or `None`) |

## Compatibility details

Like upstream Jellyfish, every function accepts `str` only. The
broader native stride-align API also accepts bytes, NumPy arrays, and
arbitrary hashable sequences, but the facade deliberately retains
Jellyfish's narrower type contract.

Jellyfish counts extended Unicode grapheme clusters in its edit,
Hamming, and Jaro algorithms. ASCII and ordinary one-code-point text
stay on stride-align's zero-copy string path. Combining sequences,
emoji ZWJ sequences, emoji modifiers, regional-indicator flags,
Hangul composition, and Indic conjuncts are segmented and sent through
the native general-sequence tokenizer. That tokenizer assigns compact
integer symbols with the same shared hash table used by alignment over
arbitrary Python objects.

Jellyfish's Jaccard operation is not the same as native
`stride_align.jaccard`:

- With no `ngram_size`, Jellyfish compares sets of whitespace-separated
  words.
- With `ngram_size=n`, it compares sets of **non-overlapping** character
  chunks and retains a final short chunk.
- Two empty token sets score `0.0`, not `1.0`.

The facade reproduces those conventions rather than routing to
stride-align's overlapping character-n-gram multiset scorer.

## Known limit

`jaccard_similarity(..., ngram_size=0)` raises a regular `ValueError`.
Jellyfish 1.2.1 lets Rust's zero-sized `chunks()` panic escape as a
`PanicException`; reproducing a runtime panic would make the shim less
safe without helping normal callers.

The in-tree grapheme segmenter covers the established cluster families
listed above. A future Unicode release could introduce a new grapheme
property or script-specific conjunct rule before Python's Unicode tables
and this compatibility layer are updated.

## Native equivalents

New code can use the richer native families when Jellyfish's exact API
shape is not required:

- [edit-distance.md](edit-distance.md) — batched scores, normalized
  scores, best-match and top-k operations
- [similarity.md](similarity.md) — batched Jaro/Jaro-Winkler and
  overlapping n-gram similarity
- [phonetic.md](phonetic.md) — explicit phonetic variants and additional
  encoders
- [thefuzz-shim.md](thefuzz-shim.md) — integer fuzzy scorers and
  extraction helpers for TheFuzz codebases
