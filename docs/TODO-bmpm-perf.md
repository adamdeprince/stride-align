# BMPM perf roadmap

The shipped Beider-Morse engine is correctness-first. The hot path
allocates more than it strictly needs to, and the per-position rule
lookup is a sorted bucket scan rather than a trie walk. Both have
known cheaper alternatives that this document tracks.

## Landed

* **Per-call regex slice elision.** ``ContextPred::match`` for the
  regex fallback used to construct a ``std::string`` from
  ``raw_byte_input.substr(...)`` per call. It now uses the iterator-
  range overload of ``std::regex_search(begin, end, regex)`` and
  performs no heap traffic on the predicate match.
* **Hoisted scratch buffers in the final-rules loop.**
  ``apply_final`` reuses one ``PhonemeArena``, one ``CodepointVec``,
  one ``std::string``, and one ``std::vector<std::size_t>`` across
  every phoneme in the live set. The codepoint → byte-offset map is
  computed without allocating a temporary string per codepoint.

## Outstanding

* **Aho-Corasick over the per-language rule pattern set.** The
  current ``RuleBuckets`` hashes rules by their first codepoint and
  iterates the bucket in descending-pattern-length order. A trie
  with failure links (or even a plain longest-prefix trie, since
  BMPM only needs the longest match at the current position) drops
  the per-position cost from ``O(B · P)`` (bucket size × avg pattern
  length) to ``O(L)`` (longest pattern length). Profile data shows
  ``B`` is typically 5–30 and ``P`` is 1–3 codepoints; the win is
  bounded but real.

* **Arena-bumping ``PhonemeBuilder``.** The arena is a
  ``std::vector<Codepoint>`` plus a small ``std::vector<ArenaPhoneme>``;
  growth amortises after the first encode, but each new arena starts
  empty. A bump-pointer arena pre-sized to ``max_phonemes · L_max``
  would eliminate the realloc tail of the first warm calls and
  consolidate every allocation into one buffer per thread.

* **Pre-decoded LangSet bit tables for ``[lang1+lang2+...]`` literals.**
  ``parse_lang_set`` does a per-call linear scan and per-name
  ``unordered_map::find``. The 18 GENERIC languages would fit in an
  ``std::array<uint8_t, 256>`` keyed by the first byte of the name.

* **Compile-time predicate dispatch.** ``ContextPred::match`` is a
  ``switch`` over ten kinds. Generating one templated specialisation
  per kind and dispatching via a function pointer at parse time
  would remove the switch from the hot path.
