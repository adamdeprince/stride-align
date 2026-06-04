#pragma once

// Beider-Morse Phonetic Matching (BMPM) — Alexander Beider & Stephen P.
// Morse, 2008.
//
// Transforms a word (typically a family name) into one or more phonetic
// codes by recognising letter-cluster patterns and emitting alternative
// pronunciations conditional on plausible source languages. The output
// is a ``|``-separated string of distinct codes; an exact-match search
// against the BMPM codes of a candidate list will surface names spelled
// differently but pronounced similarly.
//
// Stride-align ships the **GENERIC** ``NameType`` only — the broad
// "general-purpose name or word" rule set. The ``ASHKENAZI`` and
// ``SEPHARDIC`` name types from the upstream Apache Commons Codec
// distribution are not included.
//
// Pipeline (per Beider-Morse, ported from the algorithm description in
// the upstream Apache Commons Codec 1.18 BMPM classes):
//
//   1. Lower-case input (ASCII-only fold), replace ``-`` with space,
//      trim.
//   2. Handle generic name prefixes (``d'``, ``van``, ``von``, ``de``,
//      etc.): for each recognised prefix, emit ``(encoded_remainder)-
//      (encoded_combined)`` and stop.
//   3. Decode UTF-8 to a codepoint stream.
//   4. Walk the input. At each position, the rule set's Aho-Corasick
//      trie finds the longest-prefix-matching pattern whose left and
//      right context predicates also fire. Apply that rule's
//      phoneme expression to the running PhonemeBuilder set, advance
//      past the pattern. If no rule matches, drop the character.
//   5. Apply ``common`` final rules then language-specific final rules
//      to convert language-conditional phonemes into a language-
//      independent representation.
//   6. Emit ``|``-joined unique phonemes.
//
// Performance:
//   * One Aho-Corasick trie per ``(RuleType, language)`` shares a single
//     scan over each position.
//   * Left/right context predicates are pre-classified at static-init
//     into ``ContextPredKind`` (empty, anchor, single char-class,
//     literal prefix/suffix, anchored variants, regex-fallback).
//     ``std::regex`` is only used for the few patterns the classifier
//     rejects.
//   * A per-call bump arena hosts the PhonemeBuilder set so the hot
//     loop does not call ``malloc`` / ``free``.
//   * The rule trie + final-rule maps + Lang regexes are parsed once
//     into a single immutable ``BmpmTables`` singleton on the first
//     ``beider_morse()`` call and reused thereafter (``std::call_once``
//     -guarded).
//
// Source attribution:
//   * Beider, A. & Morse, S.P. "Phonetic Matching: A Better Soundex"
//     (2008). https://stevemorse.org/phonetics/bmpm.htm
//   * Apache Commons Codec ``org.apache.commons.codec.language.bm.*``
//     (Apache 2.0) for the algorithm structure and the rule data
//     files. The vendored upstream rule files live in
//     ``src/stride_align/bmpm_data/`` with their original ASF headers
//     intact.
//
// The C++ port and the Aho-Corasick / arena / predicate-classifier
// machinery are original.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

namespace stride_align::phonetic {

enum class BmpmRuleType : int {
  kApprox = 0,
  kExact  = 1,
};

// Register the Apache Commons Codec BMPM resource files with the engine.
// ``resources`` maps the basename (e.g. ``"gen_rules_english"``, no
// ``.txt`` suffix) to the file's textual content. Safe to call multiple
// times — the first registration wins and the engine is built lazily on
// the first ``beider_morse()`` call after registration.
//
// Throws ``std::runtime_error`` if a required resource is missing or
// malformed.
void bmpm_register_resources(
    const std::unordered_map<std::string, std::string>& resources);

// Encode ``input`` into one or more BMPM phonetic codes joined with
// ``|``. Returns an empty string when the engine has not been
// registered or when ``input`` contains no encodable characters.
//
// ``rule_type`` picks the rule family: ``kApprox`` produces a broader
// phonetic spread; ``kExact`` a tighter one.
//
// ``concat`` controls multi-word names: when true, the encoded form is
// produced for the concatenated whole; when false, each word is
// encoded separately and the per-word codes are joined with ``-``.
//
// ``max_phonemes`` caps the PhonemeBuilder set size (default 20,
// matching the upstream Commons Codec default).
std::string beider_morse(
    std::string_view input,
    BmpmRuleType rule_type = BmpmRuleType::kApprox,
    bool concat = true,
    std::size_t max_phonemes = 20);

}  // namespace stride_align::phonetic
