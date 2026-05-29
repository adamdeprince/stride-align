#pragma once

// NYSIIS — New York State Identification and Intelligence System
// phonetic encoder (Taft, 1970).
//
// Developed for cross-referencing person records, NYSIIS tends to
// produce more discriminative codes than Soundex for English-
// language names. Returns a code up to 6 characters (the classic
// length); does not truncate when the input is shorter.
//
// Algorithm steps (canonical formulation):
//
//   1. Translate the first letters:
//        MAC -> MCC,  KN -> NN,  K -> C,  PH -> FF,  PF -> FF,
//        SCH -> SSS
//   2. Translate the last letters:
//        EE -> Y,  IE -> Y,  DT/RT/RD/NT/ND -> D
//   3. First character of the key is the first character of the
//      transformed name.
//   4. Walk the remaining characters; for each:
//        EV -> AF
//        any vowel (A/E/I/O/U) -> A
//        Q -> G,  Z -> S,  M -> N
//        KN -> N,  K -> C
//        SCH -> SSS,  PH -> FF
//        H -> if neighbour is non-vowel, replace with the previous
//             key character (effectively "skip")
//        W -> if previous is a vowel, replace with previous letter
//   5. Skip consecutive duplicate letters in the key.
//   6. If the key ends with S, drop it.
//   7. If the key ends with AY, replace with Y.
//   8. If the key ends with A, drop it.
//   9. Truncate to length 6.
//
// Sources:
//   * Taft, R. L. "Name Search Techniques." New York State
//     Identification and Intelligence System, 1970.
//   * Apache Commons Codec Nysiis class.
//   * https://en.wikipedia.org/wiki/New_York_State_Identification_and_Intelligence_System

#include <cstddef>
#include <string>
#include <string_view>

namespace stride_align::phonetic {

namespace nysiis_detail {

inline constexpr bool is_vowel(char c) noexcept {
  return c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U';
}

inline constexpr char to_upper_ascii(char c) noexcept {
  return (c >= 'a' && c <= 'z') ? static_cast<char>(c - ('a' - 'A')) : c;
}

inline constexpr bool is_upper_alpha(char c) noexcept {
  return c >= 'A' && c <= 'Z';
}

inline bool starts_with(std::string_view s, std::string_view prefix) noexcept {
  return s.size() >= prefix.size() &&
         s.compare(0, prefix.size(), prefix) == 0;
}

inline bool ends_with(std::string_view s, std::string_view suffix) noexcept {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace nysiis_detail

inline std::string nysiis(std::string_view input) {
  using namespace nysiis_detail;

  // Pre-pass: keep ASCII letters only, upper-cased. No duplicate
  // collapse here — NYSIIS handles that at the key-building stage.
  std::string w;
  w.reserve(input.size());
  for (char c : input) {
    const char uc = to_upper_ascii(c);
    if (is_upper_alpha(uc)) w.push_back(uc);
  }
  if (w.empty()) return {};

  // Step 1: first-letter translations.
  if (starts_with(w, "MAC"))      w.replace(0, 3, "MCC");
  else if (starts_with(w, "KN"))  w.replace(0, 2, "NN");
  else if (w.front() == 'K')      w.replace(0, 1, "C");
  else if (starts_with(w, "PH"))  w.replace(0, 2, "FF");
  else if (starts_with(w, "PF"))  w.replace(0, 2, "FF");
  else if (starts_with(w, "SCH")) w.replace(0, 3, "SSS");

  // Step 2: last-letter translations.
  if (ends_with(w, "EE") || ends_with(w, "IE")) {
    w.replace(w.size() - 2, 2, "Y");
  } else if (ends_with(w, "DT") || ends_with(w, "RT") ||
             ends_with(w, "RD") || ends_with(w, "NT") ||
             ends_with(w, "ND")) {
    w.replace(w.size() - 2, 2, "D");
  }

  // Step 3: key starts with the first character.
  std::string key;
  key.reserve(w.size());
  key.push_back(w.front());

  // Steps 4-5: walk remaining characters, apply translations, skip
  // consecutive duplicates in the key. Multi-letter transforms
  // (EV→AF, KN→N, SCH→SSS, PH→FF) consume more than one input
  // letter; we track that explicitly so the input cursor advances
  // past every consumed letter rather than re-scanning them.
  const std::size_t n = w.size();
  for (std::size_t i = 1; i < n; ++i) {
    const char c = w[i];
    const char prev_in_word = w[i - 1];
    const char nx = (i + 1 < n) ? w[i + 1] : '\0';

    std::string translated;  // 0-3 chars
    std::size_t consumed = 1;

    if (c == 'E' && nx == 'V') {
      translated = "AF";
      consumed = 2;
    } else if (is_vowel(c)) {
      translated = "A";
    } else if (c == 'Q') {
      translated = "G";
    } else if (c == 'Z') {
      translated = "S";
    } else if (c == 'M') {
      translated = "N";
    } else if (c == 'K' && nx == 'N') {
      translated = "N";
      consumed = 2;
    } else if (c == 'K') {
      translated = "C";
    } else if (c == 'S' && nx == 'C' && i + 2 < n && w[i + 2] == 'H') {
      translated = "SSS";
      consumed = 3;
    } else if (c == 'P' && nx == 'H') {
      translated = "FF";
      consumed = 2;
    } else if (c == 'H') {
      // Modern NYSIIS interpretation (matches jellyfish): both
      // neighbours vowel → keep H; both non-vowel (or end of word)
      // → replace with previous letter (effectively silent via
      // key-dedup); one of each → drop H entirely.
      const bool prev_v = is_vowel(prev_in_word);
      const bool next_v = (nx != '\0') && is_vowel(nx);
      if (prev_v && next_v) {
        translated.push_back('H');
      } else if (!prev_v && !next_v) {
        translated.push_back(prev_in_word);
      }
      // else: one-of-each, drop H (translated stays empty).
    } else if (c == 'W') {
      // If previous is a vowel, replace with the previous letter
      // (which then dedupes against the previous key character).
      if (is_vowel(prev_in_word)) {
        translated.push_back(prev_in_word);
      } else {
        translated.push_back('W');
      }
    } else {
      translated.push_back(c);
    }

    // Step 5: append to key, skipping consecutive duplicates of the
    // last key character. Internal duplicates inside ``translated``
    // (e.g. the SSS in SCH→SSS) also collapse — both the leading
    // SCH→SSS path and the mid-word case do the right thing.
    for (char tc : translated) {
      if (!key.empty() && key.back() == tc) continue;
      key.push_back(tc);
    }

    // Advance i by (consumed - 1) since the for-loop ++i covers
    // the first letter on its own.
    i += consumed - 1;
  }

  // Steps 6-8: trailing-letter trim.
  if (!key.empty() && key.back() == 'S') key.pop_back();
  if (key.size() >= 2 && key[key.size() - 2] == 'A' && key.back() == 'Y') {
    key.replace(key.size() - 2, 2, "Y");
  }
  if (!key.empty() && key.back() == 'A') key.pop_back();

  // Truncation policy: the classical Taft 1970 paper truncates to
  // six characters, but modern reference implementations (notably
  // jellyfish, which is the de facto Python reference) don't. We
  // follow the modern convention so cross-checks against jellyfish
  // are bit-exact; callers wanting the classical six-character
  // form can just ``[:6]`` the result themselves.
  return key;
}

}  // namespace stride_align::phonetic
