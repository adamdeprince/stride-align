#pragma once

// Daitch-Mokotoff Soundex (Daitch & Mokotoff, 1985).
//
// A refinement of American Soundex tuned for Slavic and Yiddish
// surnames: 6-digit output (vs Soundex's letter-and-three-digits),
// the leading letter is encoded (not preserved verbatim), and rules
// fire on multi-character clusters (``sch``, ``tsch``, ``schtsch``,
// ``rz``, ``cz``) before any single-letter fallback. Several rules
// emit branched alternatives separated by ``|``; the returned string
// is a ``|``-joined set of code candidates.
//
// Public API:
//   * ``daitch_mokotoff(s)`` — branching enabled, returns
//     ``code1|code2|...``.
//   * ``daitch_mokotoff(s, branching=false)`` — single-code form.
//   * Optional ``folding=true`` (default) applies the ASCII fold table
//     so common accented characters (è/é/ê/ë/ï/ö/ü/ß ...) collapse to
//     their ASCII bases before encoding.
//
// Source attribution:
//   * Daitch, R. & Mokotoff, G. (1985). Original publication via
//     Avotaynu Inc.
//   * Apache Commons Codec ``DaitchMokotoffSoundex`` (Apache 2.0) and
//     its companion resource file ``dmrules.txt`` for the rule table
//     and algorithm. The rule data is vendored verbatim under
//     ``docs/upstream/apache-commons-codec-dmrules.txt`` with its
//     original ASF header intact and embedded below as the
//     ``kRules`` raw string literal.
//   * https://www.avotaynu.com/soundex.htm
//   * https://en.wikipedia.org/wiki/Daitch-Mokotoff_Soundex
//
// Cross-checked against the canonical ``DaitchMokotoffSoundexTest``
// vectors (Apache Commons Codec).

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace stride_align::phonetic {

namespace daitch_mokotoff_detail {

constexpr std::size_t kMaxLength = 6;

// Rule data — vendored verbatim from
// ``docs/upstream/apache-commons-codec-dmrules.txt`` (Apache 2.0,
// Apache Commons Codec). The ASF header from that file is reproduced
// inside the raw string so the derivative-work attribution survives
// even if this single header is read in isolation.
inline constexpr std::string_view kRules = R"DM(
/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Vowels
"a" "0" "" ""
"e" "0" "" ""
"i" "0" "" ""
"o" "0" "" ""
"u" "0" "" ""

// Consonants
"b" "7" "7" "7"
"d" "3" "3" "3"
"f" "7" "7" "7"
"g" "5" "5" "5"
"h" "5" "5" ""
"k" "5" "5" "5"
"l" "8" "8" "8"
"m" "6" "6" "6"
"n" "6" "6" "6"
"p" "7" "7" "7"
"q" "5" "5" "5"
"r" "9" "9" "9"
"s" "4" "4" "4"
"t" "3" "3" "3"
"v" "7" "7" "7"
"w" "7" "7" "7"
"x" "5" "54" "54"
"y" "1" "" ""
"z" "4" "4" "4"

"ţ" "3|4" "3|4" "3|4"
"ț" "3|4" "3|4" "3|4"

"ę" "" "" "|6"
"ą" "" "" "|6"

"schtsch" "2" "4" "4"
"schtsh" "2" "4" "4"
"schtch" "2" "4" "4"
"shtch" "2" "4" "4"
"shtsh" "2" "4" "4"
"stsch" "2" "4" "4"
"ttsch" "4" "4" "4"
"zhdzh" "2" "4" "4"
"shch" "2" "4" "4"
"scht" "2" "43" "43"
"schd" "2" "43" "43"
"stch" "2" "4" "4"
"strz" "2" "4" "4"
"strs" "2" "4" "4"
"stsh" "2" "4" "4"
"szcz" "2" "4" "4"
"szcs" "2" "4" "4"
"ttch" "4" "4" "4"
"tsch" "4" "4" "4"
"ttsz" "4" "4" "4"
"zdzh" "2" "4" "4"
"zsch" "4" "4" "4"
"chs" "5" "54" "54"
"csz" "4" "4" "4"
"czs" "4" "4" "4"
"drz" "4" "4" "4"
"drs" "4" "4" "4"
"dsh" "4" "4" "4"
"dsz" "4" "4" "4"
"dzh" "4" "4" "4"
"dzs" "4" "4" "4"
"sch" "4" "4" "4"
"sht" "2" "43" "43"
"szt" "2" "43" "43"
"shd" "2" "43" "43"
"szd" "2" "43" "43"
"tch" "4" "4" "4"
"trz" "4" "4" "4"
"trs" "4" "4" "4"
"tsh" "4" "4" "4"
"tts" "4" "4" "4"
"ttz" "4" "4" "4"
"tzs" "4" "4" "4"
"tsz" "4" "4" "4"
"zdz" "2" "4" "4"
"zhd" "2" "43" "43"
"zsh" "4" "4" "4"
"ai" "0" "1" ""
"aj" "0" "1" ""
"ay" "0" "1" ""
"au" "0" "7" ""
"cz" "4" "4" "4"
"cs" "4" "4" "4"
"ds" "4" "4" "4"
"dz" "4" "4" "4"
"dt" "3" "3" "3"
"ei" "0" "1" ""
"ej" "0" "1" ""
"ey" "0" "1" ""
"eu" "1" "1" ""
"fb" "7" "7" "7"
"ia" "1" "" ""
"ie" "1" "" ""
"io" "1" "" ""
"iu" "1" "" ""
"ks" "5" "54" "54"
"kh" "5" "5" "5"
"mn" "66" "66" "66"
"nm" "66" "66" "66"
"oi" "0" "1" ""
"oj" "0" "1" ""
"oy" "0" "1" ""
"pf" "7" "7" "7"
"ph" "7" "7" "7"
"sh" "4" "4" "4"
"sc" "2" "4" "4"
"st" "2" "43" "43"
"sd" "2" "43" "43"
"sz" "4" "4" "4"
"th" "3" "3" "3"
"ts" "4" "4" "4"
"tc" "4" "4" "4"
"tz" "4" "4" "4"
"ui" "0" "1" ""
"uj" "0" "1" ""
"uy" "0" "1" ""
"ue" "0" "1" ""
"zd" "2" "43" "43"
"zh" "4" "4" "4"
"zs" "4" "4" "4"

"c" "4|5" "4|5" "4|5"
"ch" "4|5" "4|5" "4|5"
"ck" "5|45" "5|45" "5|45"
"rs" "4|94" "4|94" "4|94"
"rz" "4|94" "4|94" "4|94"
"j" "1|4" "|4" "|4"

ß=s
à=a
á=a
â=a
ã=a
ä=a
å=a
æ=a
ç=c
è=e
é=e
ê=e
ë=e
ì=i
í=i
î=i
ï=i
ð=d
ñ=n
ò=o
ó=o
ô=o
õ=o
ö=o
ø=o
ù=u
ú=u
û=u
ý=y
þ=b
ÿ=y
ć=c
ł=l
ś=s
ż=z
ź=z
)DM";

// One rule row. ``pattern`` is a UTF-8 byte string (1–7 bytes for the
// Romanian / Polish entries). ``r_start`` / ``r_vowel`` / ``r_other``
// hold ``|``-separated alternative replacements; an empty alternative
// is a valid value (silent in that context).
struct Rule {
  std::string pattern;
  std::vector<std::string> r_start;
  std::vector<std::string> r_vowel;
  std::vector<std::string> r_other;
};

struct CompiledTables {
  // Rules grouped by the first byte of their pattern; within each
  // bucket sorted by descending pattern length so longest-match wins.
  std::array<std::vector<Rule>, 256> by_first_byte;
  // ASCII folding: maps a single codepoint (encoded UTF-8) to a single
  // ASCII byte. Codepoint encoded as UTF-8 string for direct comparison.
  std::unordered_map<std::string, char> foldings;
};

inline std::vector<std::string> split_alts(std::string_view s) {
  std::vector<std::string> out;
  std::size_t start = 0;
  for (std::size_t i = 0; i <= s.size(); ++i) {
    if (i == s.size() || s[i] == '|') {
      out.emplace_back(s.substr(start, i - start));
      start = i + 1;
    }
  }
  return out;
}

inline std::string_view trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r')) {
    s.remove_prefix(1);
  }
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) {
    s.remove_suffix(1);
  }
  return s;
}

// Tokenise a rule line: whitespace-split, then strip exactly one
// leading and one trailing ``"`` per token (mirrors Apache Commons
// Codec ``DaitchMokotoffSoundex.stripQuotes``).
inline std::vector<std::string> tokenise(std::string_view line) {
  std::vector<std::string> out;
  std::size_t i = 0;
  while (i < line.size()) {
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
    if (i >= line.size()) break;
    const std::size_t start = i;
    while (i < line.size() && line[i] != ' ' && line[i] != '\t') ++i;
    std::string tok(line.substr(start, i - start));
    if (!tok.empty() && tok.front() == '"') tok.erase(0, 1);
    if (!tok.empty() && tok.back() == '"') tok.pop_back();
    out.push_back(std::move(tok));
  }
  return out;
}

inline CompiledTables parse_rules() {
  CompiledTables t;
  std::string_view src = kRules;
  bool in_block = false;
  std::size_t i = 0;
  while (i < src.size()) {
    std::size_t end = src.find('\n', i);
    if (end == std::string_view::npos) end = src.size();
    std::string_view line = src.substr(i, end - i);
    i = end + 1;

    if (in_block) {
      if (line.find("*/") != std::string_view::npos) in_block = false;
      continue;
    }
    auto trimmed = trim(line);
    if (trimmed.size() >= 2 && trimmed[0] == '/' && trimmed[1] == '*') {
      // single- or multi-line block comment
      if (trimmed.find("*/") == std::string_view::npos) in_block = true;
      continue;
    }
    // strip ``// ...`` end-of-line comment
    auto cmt = trimmed.find("//");
    if (cmt != std::string_view::npos) trimmed = trimmed.substr(0, cmt);
    trimmed = trim(trimmed);
    if (trimmed.empty()) continue;

    if (trimmed.find('=') != std::string_view::npos) {
      // folding: ``<accented>=<ascii>``
      const auto eq = trimmed.find('=');
      std::string from(trimmed.substr(0, eq));
      std::string to(trimmed.substr(eq + 1));
      if (!from.empty() && !to.empty() && to.size() == 1) {
        t.foldings.emplace(std::move(from), to.front());
      }
      continue;
    }

    const auto parts = tokenise(trimmed);
    if (parts.size() != 4) continue;
    Rule r;
    r.pattern = parts[0];
    r.r_start = split_alts(parts[1]);
    r.r_vowel = split_alts(parts[2]);
    r.r_other = split_alts(parts[3]);
    if (r.pattern.empty()) continue;
    const auto first = static_cast<unsigned char>(r.pattern.front());
    t.by_first_byte[first].push_back(std::move(r));
  }
  for (auto& bucket : t.by_first_byte) {
    std::stable_sort(bucket.begin(), bucket.end(),
                     [](const Rule& a, const Rule& b) {
                       return a.pattern.size() > b.pattern.size();
                     });
  }
  return t;
}

inline const CompiledTables& tables() {
  static const CompiledTables t = parse_rules();
  return t;
}

// Lower-case ASCII fold, mirroring Apache Commons Codec
// ``Character.toLowerCase`` for the ASCII range.
inline char to_lower_ascii(char c) {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - ('A' - 'A') + 32) : c;
}

inline bool is_ascii_letter(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// Decode the next UTF-8 codepoint at ``pos``. Returns the codepoint
// and updates ``consumed`` with the byte count; on malformed input
// consumes one byte and returns the raw byte value (so non-letter
// junk is silently dropped by the caller's letter check).
inline std::uint32_t decode_one(std::string_view s, std::size_t pos,
                                 std::size_t& consumed) {
  const auto b0 = static_cast<std::uint8_t>(s[pos]);
  if (b0 < 0x80) { consumed = 1; return b0; }
  if ((b0 & 0xE0) == 0xC0 && pos + 1 < s.size()) {
    consumed = 2;
    return ((b0 & 0x1F) << 6) |
           (static_cast<std::uint8_t>(s[pos + 1]) & 0x3F);
  }
  if ((b0 & 0xF0) == 0xE0 && pos + 2 < s.size()) {
    consumed = 3;
    return ((b0 & 0x0F) << 12) |
           ((static_cast<std::uint8_t>(s[pos + 1]) & 0x3F) << 6) |
           (static_cast<std::uint8_t>(s[pos + 2]) & 0x3F);
  }
  if ((b0 & 0xF8) == 0xF0 && pos + 3 < s.size()) {
    consumed = 4;
    return ((b0 & 0x07) << 18) |
           ((static_cast<std::uint8_t>(s[pos + 1]) & 0x3F) << 12) |
           ((static_cast<std::uint8_t>(s[pos + 2]) & 0x3F) << 6) |
           (static_cast<std::uint8_t>(s[pos + 3]) & 0x3F);
  }
  consumed = 1;
  return b0;
}

inline bool is_letter_cp(std::uint32_t cp) {
  // ASCII letters first; then a permissive sweep over the codepoints
  // the upstream rule / folding files care about. Anything not in
  // either set is treated as non-letter and dropped during cleanup —
  // matches Apache Commons Codec's ``Character.isLetter`` for the
  // surnames this encoder targets.
  if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
  return cp >= 0x00C0;  // Latin Supplement and beyond
}

inline bool is_whitespace_cp(std::uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r';
}

// To-lower for Latin-1 / Latin-Extended-A characters that the
// folding table keys are lower-case for already. Returns the new
// codepoint.
inline std::uint32_t lower_cp(std::uint32_t cp) {
  if (cp >= 'A' && cp <= 'Z') return cp + 32;
  if (cp >= 0x00C0 && cp <= 0x00DE && cp != 0x00D7) return cp + 32;
  // Latin-Extended-A even codepoints upper, odd lower (mostly).
  if (cp >= 0x0100 && cp <= 0x017F && (cp & 1) == 0) return cp + 1;
  return cp;
}

inline std::string cleanup(std::string_view input, bool folding) {
  const auto& t = tables();
  std::string out;
  out.reserve(input.size());
  for (std::size_t i = 0; i < input.size();) {
    std::size_t consumed = 0;
    std::uint32_t cp = decode_one(input, i, consumed);
    if (is_whitespace_cp(cp) || !is_letter_cp(cp)) {
      i += consumed;
      continue;
    }
    cp = lower_cp(cp);
    // Encode lowered codepoint as UTF-8 for folding-table lookup
    // and for the output string.
    char buf[4];
    int n = 0;
    if (cp < 0x80) {
      buf[n++] = static_cast<char>(cp);
    } else if (cp < 0x800) {
      buf[n++] = static_cast<char>(0xC0 | (cp >> 6));
      buf[n++] = static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      buf[n++] = static_cast<char>(0xE0 | (cp >> 12));
      buf[n++] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      buf[n++] = static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      buf[n++] = static_cast<char>(0xF0 | (cp >> 18));
      buf[n++] = static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      buf[n++] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      buf[n++] = static_cast<char>(0x80 | (cp & 0x3F));
    }
    std::string key(buf, n);
    if (folding) {
      auto it = t.foldings.find(key);
      if (it != t.foldings.end()) {
        out.push_back(it->second);
        i += consumed;
        continue;
      }
    }
    out.append(key);
    i += consumed;
  }
  return out;
}

// Returns true if the byte at ``pos`` in ``s`` is an ASCII vowel.
// The encoder peels accents via folding before rule application, so
// the only vowels that reach the next-char check are ASCII.
inline bool is_ascii_vowel(char c) {
  return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}

struct Branch {
  std::string text;
  std::string last_replacement;
};

inline void process_replacement(Branch& b, const std::string& replacement,
                                 bool force_append) {
  const bool append = b.last_replacement.empty() ||
                      !b.last_replacement.ends_with(replacement) ||
                      force_append;
  if (append && b.text.size() < kMaxLength) {
    b.text.append(replacement);
    if (b.text.size() > kMaxLength) b.text.resize(kMaxLength);
  }
  b.last_replacement = replacement;
}

inline std::string daitch_mokotoff_impl(std::string_view input,
                                         bool branching, bool folding) {
  const auto& t = tables();
  const std::string cleaned = cleanup(input, folding);
  std::vector<Branch> current;
  current.push_back({});  // single empty branch

  char last_char = '\0';
  std::size_t pos = 0;
  while (pos < cleaned.size()) {
    const auto first = static_cast<unsigned char>(cleaned[pos]);
    const auto& bucket = t.by_first_byte[first];
    if (bucket.empty()) {
      // Stay on this position one byte; matches Apache Commons Codec's
      // behaviour of advancing one char on no-rule (it then re-loops
      // — net effect is the same as skip).
      ++pos;
      continue;
    }
    bool matched = false;
    for (const auto& rule : bucket) {
      if (rule.pattern.size() > cleaned.size() - pos) continue;
      if (std::memcmp(cleaned.data() + pos, rule.pattern.data(),
                       rule.pattern.size()) != 0) {
        continue;
      }
      // Pick the replacement column.
      const std::size_t next_index = pos + rule.pattern.size();
      const bool at_start = (last_char == '\0');
      const std::vector<std::string>* reps;
      if (at_start) {
        reps = &rule.r_start;
      } else if (next_index < cleaned.size() &&
                 is_ascii_vowel(cleaned[next_index])) {
        reps = &rule.r_vowel;
      } else {
        reps = &rule.r_other;
      }

      // ``mn`` / ``nm`` always emit both digits, even when adjacent
      // to a duplicate emission (Apache Commons Codec rule).
      const char this_char = cleaned[pos];
      const bool force = (last_char == 'm' && this_char == 'n') ||
                         (last_char == 'n' && this_char == 'm');

      const bool branching_required = branching && reps->size() > 1;
      std::vector<Branch> next;
      if (branching) next.reserve(current.size() * reps->size());
      for (auto& b : current) {
        for (const auto& rep : *reps) {
          if (branching_required) {
            Branch nb = b;
            process_replacement(nb, rep, force);
            next.push_back(std::move(nb));
          } else {
            process_replacement(b, rep, force);
            if (!branching) break;
            next.push_back(b);
          }
        }
      }
      if (branching) {
        // Dedupe by current text (LinkedHashSet semantics in Java).
        std::vector<Branch> uniq;
        uniq.reserve(next.size());
        for (auto& nb : next) {
          bool found = false;
          for (const auto& u : uniq) if (u.text == nb.text) { found = true; break; }
          if (!found) uniq.push_back(std::move(nb));
        }
        current.swap(uniq);
      }
      pos += rule.pattern.size();
      last_char = this_char;
      matched = true;
      break;
    }
    if (!matched) {
      ++pos;
    }
  }

  // Pad each branch with '0' to length kMaxLength and join with '|'.
  std::string out;
  for (std::size_t k = 0; k < current.size(); ++k) {
    while (current[k].text.size() < kMaxLength) current[k].text.push_back('0');
    if (k != 0) out.push_back('|');
    out.append(current[k].text);
  }
  return out;
}

}  // namespace daitch_mokotoff_detail

// Encode ``input`` as a Daitch-Mokotoff Soundex set. When
// ``branching`` is true (default) the result is a ``|``-separated
// list of distinct 6-digit codes; otherwise a single 6-digit code.
// ``folding`` (default true) applies the ASCII fold table for
// accented characters before encoding.
inline std::string daitch_mokotoff(std::string_view input,
                                    bool branching = true,
                                    bool folding = true) {
  return daitch_mokotoff_detail::daitch_mokotoff_impl(
      input, branching, folding);
}

}  // namespace stride_align::phonetic
