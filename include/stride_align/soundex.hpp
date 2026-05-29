#pragma once

// American Soundex (Russell & Odell, 1918; codified by the US Census).
//
// Standard rules:
//   1. First letter is kept verbatim (upper-cased) as the leading
//      character of the four-character output.
//   2. Remaining letters are mapped to digit codes:
//        A E I O U Y  -> 0 (vowels — break adjacency, don't emit)
//        H W          -> transparent (don't emit, don't break
//                        adjacency; H/W between two same-coded
//                        letters lets them still collapse)
//        B F P V      -> 1
//        C G J K Q S X Z -> 2
//        D T          -> 3
//        L            -> 4
//        M N          -> 5
//        R            -> 6
//   3. Adjacent letters mapping to the same digit are collapsed.
//   4. Vowels (including Y) break adjacency. H and W do not.
//   5. Output is padded with '0' to length 4, or truncated to 4.
//
// Non-ASCII codepoints in str inputs are skipped — Soundex is an
// ASCII algorithm by construction. Callers wanting accent-folding
// pre-normalise with ``unicodedata.normalize('NFKD',
// s).encode('ascii', 'ignore').decode()`` (matches jellyfish's
// behaviour) before calling.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace stride_align::phonetic {

namespace soundex_detail {

// Codes for A..Z; '0' for vowels (AEIOUY) and H/W. The collapse
// rule treats H/W specially (transparent); this table just gives
// the digit value, the loop handles the transparency.
//                                     A    B    C    D    E    F
constexpr char kSoundexCode[26] = {  '0', '1', '2', '3', '0', '1',
//                                     G    H    I    J    K    L
                                     '2', '0', '0', '2', '2', '4',
//                                     M    N    O    P    Q    R
                                     '5', '5', '0', '1', '2', '6',
//                                     S    T    U    V    W    X
                                     '2', '3', '0', '1', '0', '2',
//                                     Y    Z
                                     '0', '2'};

inline constexpr char to_upper_ascii(char c) noexcept {
  return (c >= 'a' && c <= 'z') ? static_cast<char>(c - ('a' - 'A')) : c;
}

inline constexpr bool is_upper_alpha(char c) noexcept {
  return c >= 'A' && c <= 'Z';
}

}  // namespace soundex_detail

// Compute the Soundex encoding of ``input`` (interpreted as a
// sequence of ASCII bytes — non-letters and non-ASCII bytes are
// skipped). Returns a 4-character string when ``input`` contains
// at least one ASCII letter, else an empty string.
inline std::string soundex(std::string_view input) {
  using namespace soundex_detail;

  std::string out;

  // Find first ASCII letter.
  std::size_t i = 0;
  for (; i < input.size(); ++i) {
    if (is_upper_alpha(to_upper_ascii(input[i]))) break;
  }
  if (i >= input.size()) return out;

  out.reserve(4);
  const char first = to_upper_ascii(input[i]);
  out.push_back(first);
  char last_code = kSoundexCode[first - 'A'];

  for (++i; i < input.size() && out.size() < 4; ++i) {
    const char c = to_upper_ascii(input[i]);
    if (!is_upper_alpha(c)) continue;

    // H and W are transparent: don't emit, don't break adjacency.
    if (c == 'H' || c == 'W') continue;

    const char code = kSoundexCode[c - 'A'];

    if (code == '0') {
      // True vowel (AEIOUY): breaks adjacency for the collapse
      // rule but emits nothing.
      last_code = '0';
      continue;
    }

    if (code != last_code) out.push_back(code);
    last_code = code;
  }

  // Pad to length 4.
  while (out.size() < 4) out.push_back('0');
  return out;
}

}  // namespace stride_align::phonetic
