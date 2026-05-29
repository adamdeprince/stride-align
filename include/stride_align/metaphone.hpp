#pragma once

// Metaphone (Lawrence Philips, 1990).
//
// Transforms a word into a phonetic code representing how it sounds.
// Unlike Soundex (which keeps the leading letter verbatim and emits
// digits), Metaphone transforms the entire word into letters,
// keeping the leading vowel only if the word starts with one.
//
// Code letters used: ``A B F H J K L M N P R S T W X Y 0`` (zero is
// the "theta" sound from ``TH``).
//
// Reference implementation: this follows the **published Philips
// 1990 spec** as faithfully reproduced in Apache Commons Codec's
// Metaphone class. Some popular Python implementations (notably
// jellyfish) deviate from the spec on a couple of rules — the
// most visible being CH-after-S (spec: K, jellyfish: X) and GH
// at end of word (spec: silent, jellyfish: keeps as "KH"). We
// follow the spec.
//
// A standing caveat from Michael Kuhn's 1991 C port (and quoted
// by the aspell maintainer): when Kuhn transcribed the algorithm
// from the printed BASIC listing in the 1990 article, he noted
// "there were discrepancies between the BASIC code and the verbal
// description. The discrepancies look like they could have been
// caused by typing errors in the article." So "the published
// spec" is itself a moving target — the verbal description and
// the printed code disagree in places. We track the verbal
// description / Apache Commons Codec branch of the family,
// which is the most widely cited.
//
// Sources:
//   * Philips, L. "Hanging on the Metaphone." Computer Language
//     Magazine 7(12), December 1990, pp. 39-44.
//   * Apache Commons Codec Metaphone:
//     https://commons.apache.org/proper/commons-codec/apidocs/src-html/org/apache/commons/codec/language/Metaphone.html
//   * Lawrence Philips' Metaphone overview (with Kuhn's caveat):
//     http://aspell.net/metaphone/

#include <cstddef>
#include <string>
#include <string_view>

namespace stride_align::phonetic {

namespace metaphone_detail {

inline constexpr bool is_vowel(char c) noexcept {
  return c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U';
}

inline constexpr char to_upper_ascii(char c) noexcept {
  return (c >= 'a' && c <= 'z') ? static_cast<char>(c - ('a' - 'A')) : c;
}

inline constexpr bool is_upper_alpha(char c) noexcept {
  return c >= 'A' && c <= 'Z';
}

}  // namespace metaphone_detail

inline std::string metaphone(std::string_view input) {
  using namespace metaphone_detail;

  // Pre-pass: keep ASCII letters only, upper-cased, with
  // adjacent-duplicate collapse (except C — Metaphone deliberately
  // preserves doubled C for the rule that handles "ACCE" etc.).
  // Collapsing before the rule pass lets patterns like "TTH" fall
  // out naturally as "TH" → "0" instead of being split into "T"
  // and an orphan "H".
  std::string w;
  w.reserve(input.size());
  for (char c : input) {
    const char uc = to_upper_ascii(c);
    if (!is_upper_alpha(uc)) continue;
    if (!w.empty() && w.back() == uc && uc != 'C') continue;
    w.push_back(uc);
  }
  if (w.empty()) return {};

  const std::size_t n = w.size();
  std::string out;
  out.reserve(n);

  auto get = [&](std::size_t pos) -> char {
    return pos < n ? w[pos] : '\0';
  };

  // Initial-letter pair simplifications.
  std::size_t i = 0;
  if (n >= 2) {
    const char a = w[0], b = w[1];
    if ((a == 'A' && b == 'E') ||
        (a == 'G' && b == 'N') ||
        (a == 'K' && b == 'N') ||
        (a == 'P' && b == 'N') ||
        (a == 'W' && b == 'R')) {
      i = 1;  // drop first letter
    } else if (a == 'W' && b == 'H') {
      out.push_back('W');
      i = 2;
    } else if (a == 'X') {
      out.push_back('S');
      i = 1;
    }
  }

  for (; i < n; ++i) {
    const char c = w[i];
    const char prev = i > 0 ? w[i - 1] : '\0';
    const char nx = get(i + 1);
    const char nx2 = get(i + 2);

    switch (c) {
      case 'A': case 'E': case 'I': case 'O': case 'U':
        if (out.empty()) out.push_back(c);
        break;
      case 'B':
        // Drop B at end of word after M (e.g. "lamb").
        if (i + 1 == n && prev == 'M') break;
        out.push_back('B');
        break;
      case 'C':
        if (nx == 'I' && nx2 == 'A') {
          out.push_back('X');
        } else if (nx == 'H') {
          // SCH → SK; CH otherwise → X (per Philips 1990 / Apache
          // Commons Codec — jellyfish differs here).
          out.push_back(prev == 'S' ? 'K' : 'X');
          ++i;  // consume H
        } else if (nx == 'I' || nx == 'E' || nx == 'Y') {
          // SCI / SCE / SCY: drop the C.
          if (prev == 'S') break;
          out.push_back('S');
        } else {
          out.push_back('K');
        }
        break;
      case 'D':
        if (nx == 'G' && (nx2 == 'E' || nx2 == 'Y' || nx2 == 'I')) {
          out.push_back('J');
          ++i;  // consume G
        } else {
          out.push_back('T');
        }
        break;
      case 'F':
        out.push_back('F');
        break;
      case 'G':
        if (nx == 'H') {
          // Philips 1990: GH is silent at end of word OR when H is
          // followed by a consonant. GH followed by a vowel falls
          // through — G emits K, H is processed by the next
          // iteration's H rule.
          const bool h_at_end = (i + 1 == n - 1);
          const bool h_before_vowel = (i + 2 < n) && is_vowel(nx2);
          if (h_at_end || !h_before_vowel) {
            ++i;  // drop both
            break;
          }
          // Fall through: emit K for G, leave H for next iteration.
          out.push_back('K');
        } else if (nx == 'N') {
          // GN at end of word, or GNED at end → drop the G.
          if (i + 1 == n - 1) {
            // GN at end: drop both
            ++i;
          } else if (i + 3 < n && nx2 == 'E' && w[i + 3] == 'D' &&
                     i + 3 == n - 1) {
            // GNED at end of word: drop the G.
            // Fall through to emit nothing for G, the rest is handled normally.
          } else {
            out.push_back('K');
          }
        } else if (nx == 'I' || nx == 'E' || nx == 'Y') {
          if (prev == 'G') break;  // GG case
          out.push_back('J');
        } else {
          out.push_back('K');
        }
        break;
      case 'H':
        // Drop H after a vowel and not before a vowel.
        if (prev != '\0' && is_vowel(prev) && !is_vowel(nx)) break;
        out.push_back('H');
        break;
      case 'J':
        out.push_back('J');
        break;
      case 'K':
        // CK collapses to K — drop K when preceded by C.
        if (prev == 'C') break;
        out.push_back('K');
        break;
      case 'L':
        out.push_back('L');
        break;
      case 'M':
        out.push_back('M');
        break;
      case 'N':
        out.push_back('N');
        break;
      case 'P':
        if (nx == 'H') {
          out.push_back('F');
          ++i;
        } else {
          out.push_back('P');
        }
        break;
      case 'Q':
        out.push_back('K');
        break;
      case 'R':
        out.push_back('R');
        break;
      case 'S':
        if (nx == 'H') {
          out.push_back('X');
          ++i;
        } else if (nx == 'I' && (nx2 == 'A' || nx2 == 'O')) {
          out.push_back('X');
        } else {
          out.push_back('S');
        }
        break;
      case 'T':
        if (nx == 'H') {
          out.push_back('0');  // theta
          ++i;
        } else if (nx == 'I' && (nx2 == 'A' || nx2 == 'O')) {
          out.push_back('X');
        } else if (nx == 'C' && nx2 == 'H') {
          // T is silent before CH (e.g. "match").
          break;
        } else {
          out.push_back('T');
        }
        break;
      case 'V':
        out.push_back('F');
        break;
      case 'W':
        // W emits only when followed by a vowel (e.g. "Williamson").
        if (is_vowel(nx)) out.push_back('W');
        break;
      case 'X':
        out.push_back('K');
        out.push_back('S');
        break;
      case 'Y':
        // Y emits only when followed by a vowel.
        if (is_vowel(nx)) out.push_back('Y');
        break;
      case 'Z':
        out.push_back('S');
        break;
    }
  }

  return out;
}

}  // namespace stride_align::phonetic
