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

## Outstanding

The AC trie + bump arena changes preserve correctness exactly but did
not noticeably move the end-to-end ``sa.beider_morse(name)`` wall
clock on a 14-name mix (the previous ~184 µs/call held within
measurement noise). Profiling shows the remaining time is dominated
by allocations and UTF-8 decoding inside ``apply_final``, not by the
inner-loop pattern lookup the AC change targets. The list below ranks
the remaining wins by expected impact.

* **Strip the per-phoneme ``decode_utf8`` round-trip.** The final-
  rules pass takes each live phoneme's codepoint vector, re-encodes
  it to UTF-8 only to hand the byte view to the regex-fallback
  predicate, then decodes nothing back. The regex fallback is the
  only consumer of the UTF-8 form; classifying the predicate at
  parse time and skipping the byte buffer entirely when no regex
  fires would save the encode + the ``sub_cp_to_byte`` rebuild for
  most phonemes.

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
