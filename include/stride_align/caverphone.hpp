#pragma once

// Caverphone 2.0 (David Hood, University of Otago, 2004).
//
// Phonetic encoder originally designed to match names from late-19th-
// century New Zealand electoral rolls; produces a fixed-length
// 10-character code right-padded with the digit ``1``. Caverphone 2
// supersedes the original 1.0 algorithm; we ship v2 only.
//
// The algorithm is a cascade of ~40 substitutions applied in a
// specific order. Order matters — most rules depend on the
// previous rules having already fired. The reference is Hood's
// "Caverphone Revisited" paper. We follow the rule list as
// reproduced in Apache Commons Codec Caverphone2.
//
// Sources:
//   * Hood, D. "Caverphone Revisited." Technical Report, University
//     of Otago, 2004.
//   * Apache Commons Codec Caverphone2 class.
//   * https://en.wikipedia.org/wiki/Caverphone

#include <cstddef>
#include <string>
#include <string_view>

namespace stride_align::phonetic {

namespace caverphone_detail {

inline constexpr char to_lower_ascii(char c) noexcept {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + ('a' - 'A')) : c;
}

inline constexpr bool is_lower_alpha(char c) noexcept {
  return c >= 'a' && c <= 'z';
}

// In-place find/replace-all of a single substring.
inline void replace_all(std::string& s,
                        std::string_view from,
                        std::string_view to) {
  if (from.empty()) return;
  std::size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
}

inline bool starts_with(std::string_view s, std::string_view prefix) noexcept {
  return s.size() >= prefix.size() &&
         s.compare(0, prefix.size(), prefix) == 0;
}

inline bool ends_with(std::string_view s, std::string_view suffix) noexcept {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace caverphone_detail

inline std::string caverphone(std::string_view input) {
  using namespace caverphone_detail;

  // Step 0: lower-case ASCII only.
  std::string w;
  w.reserve(input.size());
  for (char c : input) {
    const char lc = to_lower_ascii(c);
    if (is_lower_alpha(lc)) w.push_back(lc);
  }
  if (w.empty()) return std::string(10, '1');

  // Initial "-ough" word replacements: only if the entire word starts
  // with the listed prefix.
  if (starts_with(w, "cough"))  w.replace(0, 5, "cou2f");
  else if (starts_with(w, "rough")) w.replace(0, 5, "rou2f");
  else if (starts_with(w, "tough")) w.replace(0, 5, "tou2f");
  else if (starts_with(w, "enough")) w.replace(0, 6, "enou2f");
  else if (starts_with(w, "trough")) w.replace(0, 6, "trou2f");

  // Initial gn -> 2n.
  if (starts_with(w, "gn")) w.replace(0, 2, "2n");
  // Final mb -> m2.
  if (ends_with(w, "mb")) w.replace(w.size() - 2, 2, "m2");

  // Plain substring substitutions (order-sensitive — each rule
  // assumes the preceding ones have already fired).
  replace_all(w, "cq", "2q");
  replace_all(w, "ci", "si");
  replace_all(w, "ce", "se");
  replace_all(w, "cy", "sy");
  replace_all(w, "tch", "2ch");
  replace_all(w, "c", "k");
  replace_all(w, "q", "k");
  replace_all(w, "x", "k");
  replace_all(w, "v", "f");
  replace_all(w, "dg", "2g");
  replace_all(w, "tio", "sio");
  replace_all(w, "tia", "sia");
  replace_all(w, "d", "t");
  replace_all(w, "ph", "fh");
  replace_all(w, "b", "p");
  replace_all(w, "sh", "s2");
  replace_all(w, "z", "s");

  // Initial vowel -> A; remaining vowels -> 3.
  if (!w.empty()) {
    const char c0 = w.front();
    if (c0 == 'a' || c0 == 'e' || c0 == 'i' || c0 == 'o' || c0 == 'u') {
      w[0] = 'A';
    }
  }
  for (char& c : w) {
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') c = '3';
  }

  // GH context.
  replace_all(w, "3gh3", "3kh3");
  replace_all(w, "gh", "22");
  replace_all(w, "g", "k");

  // Consonant-run collapse: each rule maps consecutive same-letter
  // sequences to a single upper-case marker, then back to lower.
  // Order matters; we follow the published rule order.
  auto collapse_runs = [&](char ch, char marker) {
    // Replace runs of 2+ instances of ``ch`` with a single ``marker``.
    std::string out;
    out.reserve(w.size());
    std::size_t i = 0;
    while (i < w.size()) {
      if (w[i] == ch) {
        std::size_t j = i;
        while (j < w.size() && w[j] == ch) ++j;
        out.push_back(j - i >= 2 ? marker : ch);
        i = j;
      } else {
        out.push_back(w[i]);
        ++i;
      }
    }
    w.swap(out);
    for (char& c : w) {
      if (c == marker) c = ch;
    }
  };
  collapse_runs('s', 'S');
  collapse_runs('t', 'T');
  collapse_runs('p', 'P');
  collapse_runs('k', 'K');
  collapse_runs('f', 'F');
  collapse_runs('m', 'M');
  collapse_runs('n', 'N');

  // W rules.
  replace_all(w, "w3", "W3");
  replace_all(w, "wh3", "Wh3");
  if (ends_with(w, "w")) w.back() = '3';
  replace_all(w, "w", "2");

  // H rules.
  if (!w.empty() && w.front() == 'h') w[0] = 'A';
  replace_all(w, "h", "2");

  // R rules.
  replace_all(w, "r3", "R3");
  if (ends_with(w, "r")) w.back() = '3';
  replace_all(w, "r", "2");

  // L rules. The liquid L is preserved only when it sits between
  // two vowel markers — "Cailean" (k33L33n) keeps L because both
  // neighbours are 3; "hold" (A3lt) and "able" (Apl3) drop L
  // because at least one neighbour is a consonant.
  replace_all(w, "3l3", "3L3");
  if (ends_with(w, "l")) w.back() = '3';
  replace_all(w, "l", "2");

  // J / Y rules.
  replace_all(w, "j", "y");
  if (starts_with(w, "y3")) w.replace(0, 2, "Y3");
  if (!w.empty() && w.front() == 'y') w[0] = 'A';
  replace_all(w, "y", "3");

  // Strip 2s first, then handle the "trailing 3 → A" rule, then
  // strip the remaining 3s. Order matters — the trailing 3 needs to
  // survive past the 2-strip to be picked up here.
  {
    std::string out;
    out.reserve(w.size());
    for (char c : w) {
      if (c != '2') out.push_back(c);
    }
    w.swap(out);
  }
  // Trailing 3 → A (gives names like "Peter" their final "A").
  if (!w.empty() && w.back() == '3') w.back() = 'A';
  // Strip remaining 3s.
  {
    std::string out;
    out.reserve(w.size());
    for (char c : w) {
      if (c != '3') out.push_back(c);
    }
    w.swap(out);
  }

  // Lower-case the marker letters back to plain ASCII.
  for (char& c : w) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
  }

  // Pad with '1' to length 10, or truncate to 10.
  if (w.size() >= 10) w.resize(10);
  else                w.append(10 - w.size(), '1');
  // Upper-case for the final code (Apache Commons Codec convention).
  for (char& c : w) {
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - ('a' - 'A'));
  }
  return w;
}

}  // namespace stride_align::phonetic
