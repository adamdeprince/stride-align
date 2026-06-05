#pragma once

// Cologne Phonetic (Kölner Phonetik) — Hans Joachim Postel, 1969.
//
// German-language phonetic encoder. Maps letters / letter-pairs to
// digits 0-8 with context-sensitive rules for C, X, D, T, P; then
// collapses adjacent duplicate digits; then drops all '0' digits
// except a leading one. Result is a digit string of variable length.
//
// Designed for German names but widely applied to general European
// surnames. Distinguishes pairs like "Müller" / "Maier" that
// Soundex / Metaphone (English-oriented) tend to collapse the same.
//
// Sources:
//   * Postel, H.-J. "Die Kölner Phonetik" (1969).
//   * https://en.wikipedia.org/wiki/Cologne_phonetics
//   * https://de.wikipedia.org/wiki/K%C3%B6lner_Phonetik
//   * Apache Commons Codec ColognePhonetic class.
//
// Cross-checked against the Apache Commons Codec reference vectors
// and the Wikipedia examples (Wikipedia → 3412, Breschnew → 17863,
// Müller → 657).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace stride_align::phonetic {

// The engine runs in codepoint space. Python ``str`` storage is
// fixed-width per string (``PyUnicode_KIND`` is 1/2/4 bytes per
// codepoint, ``PyUnicode_DATA`` is the raw codepoint array); the
// dispatch wrapper widens that storage straight into
// ``std::vector<Codepoint>`` without UTF-8 round-tripping.
using Codepoint = std::uint32_t;

namespace cologne_detail {

inline constexpr char to_upper_ascii(char c) noexcept {
  return (c >= 'a' && c <= 'z') ? static_cast<char>(c - ('a' - 'A')) : c;
}

inline constexpr bool is_upper_alpha(char c) noexcept {
  return c >= 'A' && c <= 'Z';
}

// Apache Commons Codec preprocesses the German umlauts and ß into
// their Latin-letter equivalents before running the digit table; we
// follow that so callers don't have to NFKD-fold themselves. The
// fold is now keyed on the codepoint (the natural Unicode value)
// rather than on UTF-8 byte pairs:
//   Ä U+00C4 / ä U+00E4 -> A
//   Ö U+00D6 / ö U+00F6 -> O
//   Ü U+00DC / ü U+00FC -> U
//   ß U+00DF            -> SS
inline std::string preprocess_to_ascii_letters(
    const std::vector<Codepoint>& input) {
  std::string out;
  out.reserve(input.size());
  for (const auto cp : input) {
    switch (cp) {
      case 0x00C4: case 0x00E4: out.push_back('A');     continue;  // Ä / ä
      case 0x00D6: case 0x00F6: out.push_back('O');     continue;  // Ö / ö
      case 0x00DC: case 0x00FC: out.push_back('U');     continue;  // Ü / ü
      case 0x00DF:              out.append("SS");       continue;  // ß
      default: break;
    }
    if (cp >= 128U) continue;  // any other non-ASCII codepoint is dropped
    const char uc = to_upper_ascii(static_cast<char>(cp));
    if (is_upper_alpha(uc)) out.push_back(uc);
  }
  return out;
}

}  // namespace cologne_detail

inline std::string cologne_phonetic(const std::vector<Codepoint>& input) {
  using namespace cologne_detail;

  const std::string w = preprocess_to_ascii_letters(input);
  if (w.empty()) return {};

  const std::size_t n = w.size();
  auto at = [&](std::size_t i) noexcept -> char {
    return i < n ? w[i] : '\0';
  };

  // Step 1: emit digits per rule. H is silent (no digit), but it
  // still counts as the previous letter for context. We track the
  // previous SOURCE letter (not the previous emitted digit) so the
  // "after S/Z" check for C looks at the right thing.
  std::string codes;
  codes.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const char ch = w[i];
    const char next = at(i + 1);
    const char prev = (i > 0) ? w[i - 1] : '\0';

    switch (ch) {
      case 'A': case 'E': case 'I': case 'J': case 'O': case 'U': case 'Y':
        codes.push_back('0');
        break;
      case 'H':
        // Silent — emit no digit. Postel's original rule: H is a
        // "modifier" letter; it can appear in context lookups but
        // never as an output digit on its own.
        break;
      case 'B':
        codes.push_back('1');
        break;
      case 'P':
        // P -> 3 before H ("Philip" -> 3...); P -> 1 elsewhere.
        codes.push_back(next == 'H' ? '3' : '1');
        break;
      case 'D': case 'T':
        // D/T -> 8 before C/S/Z (the affricate); -> 2 elsewhere.
        codes.push_back(
            (next == 'C' || next == 'S' || next == 'Z') ? '8' : '2');
        break;
      case 'F': case 'V': case 'W':
        codes.push_back('3');
        break;
      case 'G': case 'K': case 'Q':
        codes.push_back('4');
        break;
      case 'C': {
        // C is the most context-sensitive letter. Rules:
        //   * Word-initial: -> 4 before A, H, K, L, O, Q, R, U, X;
        //                    -> 8 otherwise.
        //   * Non-initial:  -> 4 before A, H, K, O, Q, U, X
        //                      and NOT immediately after S or Z;
        //                    -> 8 otherwise.
        // The L / R exception only applies at start of word.
        const bool before_hard_initial =
            next == 'A' || next == 'H' || next == 'K' || next == 'L' ||
            next == 'O' || next == 'Q' || next == 'R' || next == 'U' ||
            next == 'X';
        const bool before_hard_inner =
            next == 'A' || next == 'H' || next == 'K' || next == 'O' ||
            next == 'Q' || next == 'U' || next == 'X';
        if (i == 0) {
          codes.push_back(before_hard_initial ? '4' : '8');
        } else if (before_hard_inner && prev != 'S' && prev != 'Z') {
          codes.push_back('4');
        } else {
          codes.push_back('8');
        }
        break;
      }
      case 'X':
        // X -> 8 after C / K / Q (since the C/K/Q already emitted the
        // /k/ part); X -> 48 elsewhere (representing /ks/).
        if (prev == 'C' || prev == 'K' || prev == 'Q') {
          codes.push_back('8');
        } else {
          codes.push_back('4');
          codes.push_back('8');
        }
        break;
      case 'L':
        codes.push_back('5');
        break;
      case 'M': case 'N':
        codes.push_back('6');
        break;
      case 'R':
        codes.push_back('7');
        break;
      case 'S': case 'Z':
        codes.push_back('8');
        break;
      default:
        // Already filtered to A-Z in preprocess_to_ascii_letters.
        break;
    }
  }

  // Step 2: collapse adjacent duplicate digits.
  std::string collapsed;
  collapsed.reserve(codes.size());
  for (char c : codes) {
    if (collapsed.empty() || collapsed.back() != c) {
      collapsed.push_back(c);
    }
  }

  // Step 3: drop all '0' digits except a leading one (when present).
  // "Mayer" -> 607 -> after dropping internal zeros -> 67 (no leading
  // 0 to preserve). "Aaron" -> 00706 -> collapse -> 0706 -> keep the
  // leading 0, drop the rest -> 076.
  std::string result;
  result.reserve(collapsed.size());
  for (std::size_t i = 0; i < collapsed.size(); ++i) {
    if (i == 0 || collapsed[i] != '0') {
      result.push_back(collapsed[i]);
    }
  }
  return result;
}

}  // namespace stride_align::phonetic
