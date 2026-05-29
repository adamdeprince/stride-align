#pragma once

// Match Rating Approach (Moore, Western Airlines, 1977).
//
// Two operations:
//
//   1. ``match_rating_codex(name)`` — encode a name into a short
//      phonetic code (typically 3-6 ASCII letters).
//   2. ``match_rating_compare(name1, name2)`` — return ``true`` if
//      the pair is similar enough per MRA's matching rules.
//
// MRA pairs an encoder and a rule-based comparator rather than
// being a one-shot distance. Two names produce the same codex
// (and so trivially match) when their consonant skeleton is the
// same; the comparator also tolerates one of the two having a
// few more characters by enforcing a length-dependent minimum
// "rating".
//
// Codex rules:
//   * Drop non-ASCII letters; upper-case the rest.
//   * Drop vowels (AEIOU) **except** when the vowel is the first
//     letter of the name (Y is treated as a consonant).
//   * Collapse consecutive duplicate letters.
//   * If the result is longer than 6 characters, keep the first 3
//     and the last 3 (e.g. ``ROBERTO`` -> ``RBRT`` -> 4 chars,
//     unchanged; ``CHRISTOPHER`` -> ``CHRSTPHR`` -> ``CHRPHR``).
//
// Comparison rules:
//   * Compute the codex of both inputs. If they differ in length
//     by more than 3 letters, return ``false``.
//   * Combined length determines the minimum rating:
//         total <= 4  -> need at least 5
//         5 <= total <= 7  -> need at least 4
//         8 <= total <= 11 -> need at least 3
//         total >= 12 -> need at least 2
//   * Strip identical characters position-wise left-to-right.
//   * Strip identical characters position-wise right-to-left
//     (on the reversed remainder).
//   * Rating = 6 - (length of the longer remainder after both
//     stripping passes).
//   * Match iff rating >= minimum.
//
// Sources:
//   * Moore, G. (Western Airlines), 1977 — original MRA paper.
//   * Apache Commons Codec MatchRatingApproachEncoder.
//   * https://en.wikipedia.org/wiki/Match_rating_approach
//
// Cross-checked against ``jellyfish.match_rating_codex`` and
// ``jellyfish.match_rating_comparison``.

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>

namespace stride_align::phonetic {

namespace mra_detail {

inline constexpr bool is_vowel(char c) noexcept {
  return c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U';
}

inline constexpr char to_upper_ascii(char c) noexcept {
  return (c >= 'a' && c <= 'z') ? static_cast<char>(c - ('a' - 'A')) : c;
}

inline constexpr bool is_upper_alpha(char c) noexcept {
  return c >= 'A' && c <= 'Z';
}

}  // namespace mra_detail

inline std::string match_rating_codex(std::string_view input) {
  using namespace mra_detail;

  // Pre-pass: keep ASCII letters, upper-case.
  std::string w;
  w.reserve(input.size());
  for (char c : input) {
    const char uc = to_upper_ascii(c);
    if (is_upper_alpha(uc)) w.push_back(uc);
  }
  if (w.empty()) return {};

  // Step 1: drop vowels except the first letter.
  std::string consonants;
  consonants.reserve(w.size());
  consonants.push_back(w.front());
  for (std::size_t i = 1; i < w.size(); ++i) {
    if (!is_vowel(w[i])) consonants.push_back(w[i]);
  }

  // Step 2: collapse consecutive duplicates.
  std::string deduped;
  deduped.reserve(consonants.size());
  for (char c : consonants) {
    if (!deduped.empty() && deduped.back() == c) continue;
    deduped.push_back(c);
  }

  // Step 3: if longer than 6, keep first 3 + last 3.
  if (deduped.size() > 6) {
    deduped = deduped.substr(0, 3) + deduped.substr(deduped.size() - 3, 3);
  }
  return deduped;
}

namespace mra_detail {

// Strip identical characters position-wise, left-to-right, from
// the heads of ``a`` and ``b`` in place. Returns nothing — both
// are shortened.
inline void strip_matching_left_to_right(std::string& a, std::string& b) {
  std::string a2, b2;
  a2.reserve(a.size());
  b2.reserve(b.size());
  std::size_t i = 0;
  const std::size_t shared = std::min(a.size(), b.size());
  for (; i < shared; ++i) {
    if (a[i] == b[i]) continue;
    a2.push_back(a[i]);
    b2.push_back(b[i]);
  }
  // Tail of the longer string (if any) is preserved.
  for (; i < a.size(); ++i) a2.push_back(a[i]);
  for (std::size_t j = shared; j < b.size(); ++j) b2.push_back(b[j]);
  a.swap(a2);
  b.swap(b2);
}

}  // namespace mra_detail

inline bool match_rating_compare(
    std::string_view a, std::string_view b) {
  std::string ca = match_rating_codex(a);
  std::string cb = match_rating_codex(b);

  // Empty codex on either side: no meaningful comparison.
  if (ca.empty() || cb.empty()) return false;

  // Length-difference gate.
  const std::size_t la = ca.size();
  const std::size_t lb = cb.size();
  const std::size_t diff = la > lb ? la - lb : lb - la;
  if (diff > 3) return false;

  // Length-sum -> minimum rating threshold.
  const std::size_t total = la + lb;
  int min_rating;
  if (total <= 4) min_rating = 5;
  else if (total <= 7) min_rating = 4;
  else if (total <= 11) min_rating = 3;
  else min_rating = 2;

  // Strip identical characters position-wise left-to-right.
  mra_detail::strip_matching_left_to_right(ca, cb);
  // Then right-to-left: reverse, strip, reverse back.
  std::reverse(ca.begin(), ca.end());
  std::reverse(cb.begin(), cb.end());
  mra_detail::strip_matching_left_to_right(ca, cb);
  // (No need to reverse back — only sizes matter from here.)

  const int rating = 6 - static_cast<int>(std::max(ca.size(), cb.size()));
  return rating >= min_rating;
}

}  // namespace stride_align::phonetic
