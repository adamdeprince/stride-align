# BMPM perf roadmap

The shipped Beider-Morse engine is correctness-first. This document
tracks the data-structure and allocation optimisations that have
landed and the ones still outstanding.

## Landed

* **Per-call regex slice elision.** ``ContextPred::match`` for the
  regex fallback uses the iterator-range overload of
  ``std::regex_search(begin, end, regex)`` instead of constructing a
  ``std::string`` from ``raw_byte_input.substr(...)`` per call.

* **Hoisted scratch buffers in the final-rules loop.**
  ``apply_final`` reuses one ``PhonemeArena``, one ``CodepointVec``,
  one ``std::string``, and one ``std::vector<std::size_t>`` across
  every phoneme in the live set. The codepoint → byte-offset map is
  computed without allocating a temporary string per codepoint.

* **Aho-Corasick trie over the per-(role × language) rule pattern set.**
  ``RuleTrie`` replaces the old ``RuleBuckets`` (first-codepoint hash
  bucket plus a per-bucket descending-pattern-length scan).
  ``AhoNode`` holds ``unordered_map<Codepoint, uint32_t>`` children,
  a failure link populated by a single BFS sweep, and the
  ``rule_ids`` list of rules whose pattern ends at that node in
  insertion order. ``find_rule`` walks from root for ``input[pos..]``,
  collects terminals grouped by depth, and iterates descending depth
  with the file-order tiebreak preserved. Failure links are
  available for callers that want classic AC continuous-scan
  semantics; BMPM's per-position-then-skip-ahead loop walks from
  root each step and does not currently traverse them.

* **Bump-pointer ``PhonemeArena``.** ``apply`` and ``append_literal``
  now compute the worst-case combined buffer growth up-front and
  ``reserve`` it once before the per-rule-fire self-insert loop.
  The old ``CodepointVec left_copy`` allocation that existed to
  dodge mid-loop ``std::vector`` reallocation goes away — after the
  pre-reserve the source iterators remain valid for the whole loop.

* **Codepoint-end-to-end engine; no UTF-8 round-trip on the input
  side.** ``dispatch_beider_morse`` reads codepoints straight out of
  ``PyUnicode_DATA`` (widening 1/2/4-byte storage into
  ``std::vector<Codepoint>``) and hands them to the engine. The
  public ``beider_morse`` entry takes ``const std::vector<Codepoint>&``.
  ``encode_one_word`` and ``apply_final`` no longer build the
  ``lowered_bytes`` UTF-8 buffer or the parallel ``cp_to_byte``
  offset table. ``ContextPred::match`` runs in codepoint space; the
  ``kRegex`` fallback lazy-encodes its slice to UTF-8 only when a
  regex predicate is about to fire (a small minority of context
  predicates after classification). ``Lang::guess`` accepts a
  codepoint vector and encodes once for its UTF-8-pattern regex
  search.

## Outstanding

The AC trie + bump arena changes plus the codepoint-end-to-end
rewrite preserve correctness exactly and remove the input-side and
per-phoneme UTF-8 traffic. The end-to-end ``sa.beider_morse(name)``
wall clock on a 14-name mix moved from ~184 µs/call to ~177 µs/call
— small because most of the remaining cost is per-phoneme allocation
in the final-rules pass that no per-cycle micro-fix touches.

* **Pre-decoded LangSet bit tables for ``[lang1+lang2+...]`` literals.**
  ``parse_lang_set`` does a per-call linear scan plus per-name
  ``unordered_map::find``. The 18 GENERIC languages would fit in an
  ``std::array<uint8_t, 256>`` keyed by the first byte of the name.

* **Compile-time predicate dispatch.** ``ContextPred::match`` is a
  ``switch`` over ten kinds. Generating one templated specialisation
  per kind and dispatching via a function pointer at parse time
  would remove the switch from the hot path.

* **AC failure-link use in continuous-scan mode.** A future
  ``find_all_rules_in_range`` entry point could exploit the failure
  links to scan an input span without restarting at root per
  position. Not used by ``encode_one_word`` today.
