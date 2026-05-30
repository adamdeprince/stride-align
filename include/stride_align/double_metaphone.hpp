#pragma once

// Double Metaphone (Lawrence Philips, 2000).
//
// Produces two phonetic codes — a *primary* and an *alternate* —
// per input name. The alternate captures plausible non-English
// pronunciations (Italian, Spanish, German, Slavic, Greek
// patterns) and is empty (matches primary) for names that have
// only one reasonable encoding.
//
// Returns ``DoubleMetaphoneResult{primary, alternate}`` where
// ``alternate`` is an empty string when no second pronunciation
// applies. Maximum output length defaults to 4 (the original
// Philips spec), set via a kwarg on the public API.
//
// Reference: Lawrence Philips, "The Double Metaphone Search
// Algorithm", C/C++ Users Journal, June 2000. We mirror the
// algorithmic structure of Apache Commons Codec's DoubleMetaphone
// class (which is the de-facto open-source reference) — Slavo-
// Germanic and silent-start detection, the heavy letter handlers
// (C, G, S, T), and the per-character cursor-advance pattern.
//
// Cross-checked against the ``metaphone`` and ``doublemetaphone``
// Python packages. They agree on every name we tested except one
// pattern: GH preceded by a vowel at position 1 or 2 from the start
// (e.g. "Hugh", "High"). Apache Commons Codec (and the
// ``doublemetaphone`` PyPI port that mirrors it) say GH is silent
// here, giving "Hugh" -> "H". The ``metaphone`` PyPI package leaks
// the previous character's emission through a missing else clause,
// giving "Hugh" -> "HH". We treat that as a documented variation
// rather than picking a winner — ``DoubleMetaphoneVariant`` selects.

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

namespace stride_align::phonetic {

// Selects which published behaviour to mimic. The two values are
// NOT equal-weight algorithmic variations — ``kCommons`` is the
// faithful Lawrence Philips port (Apache Commons Codec, and the
// ``doublemetaphone`` PyPI package that mirrors it). ``kPython``
// reproduces a known bug in the ``metaphone`` PyPI package: a
// missing ``else`` in its ``process_g`` GH branch lets the previous
// character's ``self.next`` tuple leak into the next iteration, so
// names like "Hugh" and "High" come out "HH" instead of "H". We
// expose it as an opt-in for callers cross-checking against that
// specific library; if you don't have that constraint, stay on
// ``kCommons``.
enum class DoubleMetaphoneVariant : int {
  // Apache Commons Codec semantics. "Hugh" -> ("H", "").
  // Recommended default; correct per the published spec.
  kCommons = 0,
  // ``metaphone`` PyPI package bug-compat. "Hugh" -> ("HH", "").
  // Opt in only when you need byte-for-byte agreement with that
  // library; the divergent code paths are documented bugs, not
  // legitimate algorithmic alternatives.
  kPython  = 1,
};

struct DoubleMetaphoneResult {
  std::string primary;
  std::string alternate;
};

namespace double_metaphone_detail {

inline constexpr char to_upper_ascii(char c) noexcept {
  return (c >= 'a' && c <= 'z') ? static_cast<char>(c - ('a' - 'A')) : c;
}

inline constexpr bool is_upper_alpha(char c) noexcept {
  return c >= 'A' && c <= 'Z';
}

inline bool is_vowel(const std::string& w, std::size_t pos) noexcept {
  if (pos >= w.size()) return false;
  const char c = w[pos];
  return c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' || c == 'Y';
}

// Test whether ``w`` from position ``pos`` for ``len`` characters
// matches any of the strings in ``cands``. Returns false if the
// span runs off either end of ``w``. ``pos`` is signed so call
// sites passing ``pos - N`` (where ``pos`` is ``size_t``) don't
// trigger out-of-range exceptions on underflow — a wrapped-around
// size_t casts to a negative ptrdiff_t and gets rejected here.
inline bool contains_at(const std::string& w,
                        std::ptrdiff_t pos,
                        std::size_t len,
                        std::initializer_list<std::string_view> cands) noexcept {
  if (pos < 0) return false;
  const std::size_t upos = static_cast<std::size_t>(pos);
  if (upos + len > w.size()) return false;
  for (const auto& c : cands) {
    if (c.size() != len) continue;
    if (w.compare(upos, len, c.data(), c.size()) == 0) return true;
  }
  return false;
}

// Slavo-Germanic flag: presence of W, K, CZ, or WITZ anywhere.
inline bool is_slavo_germanic(const std::string& w) noexcept {
  return w.find('W') != std::string::npos
      || w.find('K') != std::string::npos
      || w.find("CZ") != std::string::npos
      || w.find("WITZ") != std::string::npos;
}

inline bool is_silent_start(const std::string& w) noexcept {
  return contains_at(w, 0, 2, {"GN", "KN", "PN", "WR", "PS"});
}

// Append to both primary and alternate (the common case).
inline void append(DoubleMetaphoneResult& r, char c) {
  r.primary.push_back(c);
  r.alternate.push_back(c);
}

inline void append(DoubleMetaphoneResult& r, std::string_view s) {
  r.primary.append(s);
  r.alternate.append(s);
}

// Append different characters to primary and alternate.
inline void append(DoubleMetaphoneResult& r, char p, char a) {
  r.primary.push_back(p);
  r.alternate.push_back(a);
}

inline void append(DoubleMetaphoneResult& r,
                   std::string_view p, std::string_view a) {
  r.primary.append(p);
  r.alternate.append(a);
}

inline bool full(const DoubleMetaphoneResult& r, std::size_t max_len) {
  return r.primary.size() >= max_len && r.alternate.size() >= max_len;
}

// ---- per-letter handlers --------------------------------------------------

inline std::size_t handle_AEIOUY(
    DoubleMetaphoneResult& r, std::size_t pos) {
  if (pos == 0) append(r, 'A');
  return pos + 1;
}

inline std::size_t handle_B(
    DoubleMetaphoneResult& r, const std::string& w, std::size_t pos) {
  append(r, 'P');
  return (pos + 1 < w.size() && w[pos + 1] == 'B') ? pos + 2 : pos + 1;
}

inline std::size_t handle_C(
    DoubleMetaphoneResult& r,
    const std::string& w,
    std::size_t pos,
    bool slavo_germanic) {
  const std::size_t n = w.size();
  auto at = [&](std::size_t p) -> char {
    return p < n ? w[p] : '\0';
  };

  // Various Germanic, Greek, otherwise contexts.
  if (pos > 1
      && !is_vowel(w, pos - 2)
      && contains_at(w, pos - 1, 3, {"ACH"})
      && at(pos + 2) != 'I'
      && (at(pos + 2) != 'E'
          || contains_at(w, pos - 2, 6, {"BACHER", "MACHER"}))) {
    append(r, 'K');
    return pos + 2;
  }
  // Special case for Caesar.
  if (pos == 0 && contains_at(w, 0, 6, {"CAESAR"})) {
    append(r, 'S');
    return pos + 2;
  }
  if (contains_at(w, pos, 4, {"CHIA"})) {
    append(r, 'K');
    return pos + 2;
  }
  if (contains_at(w, pos, 2, {"CH"})) {
    if (pos > 0 && contains_at(w, pos, 4, {"CHAE"})) {
      append(r, 'K', 'X');
      return pos + 2;
    }
    // Greek roots, e.g. Chemistry, Chorus.
    if (pos == 0
        && (contains_at(w, 1, 5, {"HARAC", "HARIS"})
            || contains_at(w, 1, 3, {"HOR", "HYM", "HIA", "HEM"}))
        && !contains_at(w, 0, 5, {"CHORE"})) {
      append(r, 'K');
      return pos + 2;
    }
    // Germanic, Greek, or otherwise.
    if (contains_at(w, 0, 4, {"VAN ", "VON "})
        || contains_at(w, 0, 3, {"SCH"})
        || contains_at(w, pos - 2, 6, {"ORCHES", "ARCHIT", "ORCHID"})
        || contains_at(w, pos + 2, 1, {"T", "S"})
        || ((pos == 0 || contains_at(w, pos - 1, 1, {"A", "O", "U", "E"}))
            && contains_at(w, pos + 2, 1, {
                "L", "R", "N", "M", "B", "H", "F", "V", "W", " "}))) {
      append(r, 'K');
    } else {
      if (pos > 0) {
        if (contains_at(w, 0, 2, {"MC"})) {
          append(r, 'K');
        } else {
          append(r, 'X', 'K');
        }
      } else {
        append(r, 'X');
      }
    }
    return pos + 2;
  }
  // Italian CZ.
  if (contains_at(w, pos, 2, {"CZ"})
      && !contains_at(w, pos - 2, 4, {"WICZ"})) {
    append(r, 'S', 'X');
    return pos + 2;
  }
  // Italian CIO/CIE/CIA.
  if (contains_at(w, pos + 1, 3, {"CIA"})) {
    append(r, 'X');
    return pos + 3;
  }
  // Double CC.
  if (contains_at(w, pos, 2, {"CC"})
      && !(pos == 1 && at(0) == 'M')) {
    // ccI, ccE, ccH but not CCE/CCI in coup etc.
    if (contains_at(w, pos + 2, 1, {"I", "E", "H"})
        && !contains_at(w, pos + 2, 2, {"HU"})) {
      // Accident, Accede, Succeed
      if ((pos == 1 && at(pos - 1) == 'A')
          || contains_at(w, pos - 1, 5, {"UCCEE", "UCCES"})) {
        append(r, "KS");
      } else {
        append(r, 'X');
      }
      return pos + 3;
    }
    // Bacci, Bertucci, etc.
    append(r, 'K');
    return pos + 2;
  }
  if (contains_at(w, pos, 2, {"CK", "CG", "CQ"})) {
    append(r, 'K');
    return pos + 2;
  }
  if (contains_at(w, pos, 2, {"CI", "CE", "CY"})) {
    // Italian vs. English.
    if (contains_at(w, pos, 3, {"CIO", "CIE", "CIA"})) {
      append(r, 'S', 'X');
    } else {
      append(r, 'S');
    }
    return pos + 2;
  }
  append(r, 'K');
  // Mac+something - skip the second character.
  if (contains_at(w, pos + 1, 2, {" C", " Q", " G"})) {
    return pos + 3;
  }
  if (contains_at(w, pos + 1, 1, {"C", "K", "Q"})
      && !contains_at(w, pos + 1, 2, {"CE", "CI"})) {
    return pos + 2;
  }
  return pos + 1;
}

inline std::size_t handle_D(
    DoubleMetaphoneResult& r, const std::string& w, std::size_t pos) {
  if (contains_at(w, pos, 2, {"DG"})) {
    if (contains_at(w, pos + 2, 1, {"I", "E", "Y"})) {
      append(r, 'J');
      return pos + 3;
    }
    append(r, "TK");
    return pos + 2;
  }
  if (contains_at(w, pos, 2, {"DT", "DD"})) {
    append(r, 'T');
    return pos + 2;
  }
  append(r, 'T');
  return pos + 1;
}

inline std::size_t handle_G(
    DoubleMetaphoneResult& r,
    const std::string& w,
    std::size_t pos,
    bool slavo_germanic,
    DoubleMetaphoneVariant variant) {
  const std::size_t n = w.size();
  auto at = [&](std::size_t p) -> char {
    return p < n ? w[p] : '\0';
  };
  if (at(pos + 1) == 'H') {
    if (pos > 0 && !is_vowel(w, pos - 1)) {
      append(r, 'K');
      return pos + 2;
    }
    if (pos == 0) {
      if (at(pos + 2) == 'I') {
        append(r, 'J');
      } else {
        append(r, 'K');
      }
      return pos + 2;
    }
    // metaphone-py compat: at this point ``pos > 0`` and the char
    // before G is a vowel. The PyPI ``metaphone`` package has an
    // elif chain whose ``position < start_index + 3`` clause
    // shortcircuits Parker's rule and, when ``pos`` is 1 or 2,
    // leaves ``self.next`` at the previous letter's tuple — so the
    // previous emit fires again. We replicate the symptom directly
    // here: re-append whatever was last emitted. Apache Commons
    // falls through to Parker (the next block) instead.
    if (variant == DoubleMetaphoneVariant::kPython && pos < 3) {
      if (!r.primary.empty())  r.primary.push_back(r.primary.back());
      if (!r.alternate.empty())r.alternate.push_back(r.alternate.back());
      return pos + 2;
    }
    // Parker's rules — silent in many contexts.
    if ((pos > 1 && contains_at(w, pos - 2, 1, {"B", "H", "D"}))
        || (pos > 2 && contains_at(w, pos - 3, 1, {"B", "H", "D"}))
        || (pos > 3 && contains_at(w, pos - 4, 1, {"B", "H"}))) {
      return pos + 2;
    }
    // E.g. "laugh", "cough"
    if (pos > 2 && at(pos - 1) == 'U'
        && contains_at(w, pos - 3, 1, {"C", "G", "L", "R", "T"})) {
      append(r, 'F');
      return pos + 2;
    }
    if (pos > 0 && at(pos - 1) != 'I') {
      append(r, 'K');
    }
    return pos + 2;
  }
  if (at(pos + 1) == 'N') {
    if (pos == 1 && is_vowel(w, 0) && !slavo_germanic) {
      append(r, "KN", "N");
      return pos + 2;
    }
    if (!contains_at(w, pos + 2, 2, {"EY"})
        && at(pos + 1) != 'Y' && !slavo_germanic) {
      append(r, "N", "KN");
      return pos + 2;
    }
    append(r, "KN");
    return pos + 2;
  }
  if (contains_at(w, pos + 1, 2, {"LI"}) && !slavo_germanic) {
    append(r, "KL", "L");
    return pos + 2;
  }
  if (pos == 0
      && (at(pos + 1) == 'Y'
          || contains_at(w, pos + 1, 2, {
              "ES", "EP", "EB", "EL", "EY", "IB", "IL", "IN", "IE", "EI", "ER"}))) {
    append(r, 'K', 'J');
    return pos + 2;
  }
  if ((contains_at(w, pos + 1, 2, {"ER"})
       || at(pos + 1) == 'Y')
      && !contains_at(w, 0, 6, {"DANGER", "RANGER", "MANGER"})
      && !contains_at(w, pos - 1, 1, {"E", "I"})
      && !contains_at(w, pos - 1, 3, {"RGY", "OGY"})) {
    append(r, 'K', 'J');
    return pos + 2;
  }
  if (contains_at(w, pos + 1, 1, {"E", "I", "Y"})
      || contains_at(w, pos - 1, 4, {"AGGI", "OGGI"})) {
    if (contains_at(w, 0, 4, {"VAN ", "VON "})
        || contains_at(w, 0, 3, {"SCH"})
        || contains_at(w, pos + 1, 2, {"ET"})) {
      append(r, 'K');
    } else if (contains_at(w, pos + 1, 3, {"IER"})) {
      append(r, 'J');
    } else {
      append(r, 'J', 'K');
    }
    return pos + 2;
  }
  if (at(pos + 1) == 'G') {
    append(r, 'K');
    return pos + 2;
  }
  append(r, 'K');
  return pos + 1;
}

inline std::size_t handle_H(
    DoubleMetaphoneResult& r, const std::string& w, std::size_t pos) {
  if ((pos == 0 || is_vowel(w, pos - 1)) && is_vowel(w, pos + 1)) {
    append(r, 'H');
    return pos + 2;
  }
  return pos + 1;
}

inline std::size_t handle_J(
    DoubleMetaphoneResult& r,
    const std::string& w,
    std::size_t pos,
    bool slavo_germanic) {
  const std::size_t n = w.size();
  if (contains_at(w, pos, 4, {"JOSE"}) || contains_at(w, 0, 4, {"SAN "})) {
    if ((pos == 0 && pos + 4 < n && w[pos + 4] == ' ')
        || contains_at(w, 0, 4, {"SAN "})) {
      append(r, 'H');
    } else {
      append(r, 'J', 'H');
    }
    return pos + 1;
  }
  if (pos == 0 && !contains_at(w, pos, 4, {"JOSE"})) {
    append(r, 'J', 'A');
  } else if (is_vowel(w, pos - 1) && !slavo_germanic
             && (pos + 1 < n && (w[pos + 1] == 'A' || w[pos + 1] == 'O'))) {
    append(r, 'J', 'H');
  } else if (pos + 1 == n) {
    append(r, 'J', ' ');
  } else if (pos + 1 < n
             && !contains_at(w, pos + 1, 1, {"L", "T", "K", "S", "N", "M", "B", "Z"})
             && !(pos > 0
                  && contains_at(w, pos - 1, 1, {"S", "K", "L"}))) {
    append(r, 'J');
  }
  return (pos + 1 < n && w[pos + 1] == 'J') ? pos + 2 : pos + 1;
}

inline std::size_t handle_L(
    DoubleMetaphoneResult& r, const std::string& w, std::size_t pos) {
  const std::size_t n = w.size();
  if (pos + 1 < n && w[pos + 1] == 'L') {
    // Spanish/Italian "ll": appended differently between primary
    // and alternate when at certain word-final positions.
    if ((pos == n - 3
         && contains_at(w, pos - 1, 4, {"ILLO", "ILLA", "ALLE"}))
        || ((contains_at(w, n - 2, 2, {"AS", "OS"}) || (n >= 1 && (w[n - 1] == 'A' || w[n - 1] == 'O')))
            && contains_at(w, pos - 1, 4, {"ALLE"}))) {
      r.primary.push_back('L');
      // alternate gets nothing (silent in the Spanish reading).
      return pos + 2;
    }
    append(r, 'L');
    return pos + 2;
  }
  append(r, 'L');
  return pos + 1;
}

inline std::size_t handle_M(
    DoubleMetaphoneResult& r, const std::string& w, std::size_t pos) {
  const std::size_t n = w.size();
  append(r, 'M');
  if ((pos + 1 < n && w[pos + 1] == 'M')
      || (pos > 0 && contains_at(w, pos - 1, 3, {"UMB"})
          && (pos + 1 == n
              || contains_at(w, pos + 2, 2, {"ER"})))) {
    return pos + 2;
  }
  return pos + 1;
}

inline std::size_t handle_N(
    DoubleMetaphoneResult& r, const std::string& w, std::size_t pos) {
  append(r, 'N');
  return (pos + 1 < w.size() && w[pos + 1] == 'N') ? pos + 2 : pos + 1;
}

inline std::size_t handle_P(
    DoubleMetaphoneResult& r, const std::string& w, std::size_t pos) {
  if (pos + 1 < w.size() && w[pos + 1] == 'H') {
    append(r, 'F');
    return pos + 2;
  }
  append(r, 'P');
  return (pos + 1 < w.size() && (w[pos + 1] == 'P' || w[pos + 1] == 'B')) ? pos + 2 : pos + 1;
}

inline std::size_t handle_Q(
    DoubleMetaphoneResult& r, const std::string& w, std::size_t pos) {
  append(r, 'K');
  return (pos + 1 < w.size() && w[pos + 1] == 'Q') ? pos + 2 : pos + 1;
}

inline std::size_t handle_R(
    DoubleMetaphoneResult& r,
    const std::string& w,
    std::size_t pos,
    bool slavo_germanic) {
  const std::size_t n = w.size();
  // French -IER: primary appends nothing, alternate appends R.
  if (pos + 1 == n && !slavo_germanic
      && contains_at(w, pos - 2, 2, {"IE"})
      && !contains_at(w, pos - 4, 2, {"ME", "MA"})) {
    r.alternate.push_back('R');
    return pos + 1;
  }
  append(r, 'R');
  return (pos + 1 < n && w[pos + 1] == 'R') ? pos + 2 : pos + 1;
}

inline std::size_t handle_S(
    DoubleMetaphoneResult& r,
    const std::string& w,
    std::size_t pos,
    bool slavo_germanic) {
  const std::size_t n = w.size();
  if (pos > 0 && contains_at(w, pos - 1, 3, {"ISL", "YSL"})) {
    return pos + 1;
  }
  if (pos == 0 && contains_at(w, 0, 5, {"SUGAR"})) {
    append(r, 'X', 'S');
    return pos + 1;
  }
  if (contains_at(w, pos, 2, {"SH"})) {
    if (contains_at(w, pos + 1, 4, {"HEIM", "HOEK", "HOLM", "HOLZ"})) {
      append(r, 'S');
    } else {
      append(r, 'X');
    }
    return pos + 2;
  }
  if (contains_at(w, pos, 3, {"SIO", "SIA"})
      || contains_at(w, pos, 4, {"SIAN"})) {
    if (!slavo_germanic) append(r, 'S', 'X');
    else append(r, 'S');
    return pos + 3;
  }
  if ((pos == 0
       && contains_at(w, pos + 1, 1, {"M", "N", "L", "W"}))
      || contains_at(w, pos + 1, 1, {"Z"})) {
    append(r, 'S', 'X');
    if (pos + 1 < n && w[pos + 1] == 'Z') return pos + 2;
    return pos + 1;
  }
  if (contains_at(w, pos, 2, {"SC"})) {
    if (pos + 2 < n && w[pos + 2] == 'H') {
      // Dutch origin: schermerhorn, schenker.
      if (contains_at(w, pos + 3, 2, {
          "OO", "ER", "EN", "UY", "ED", "EM"})) {
        if (contains_at(w, pos + 3, 2, {"ER", "EN"})) {
          append(r, "X", "SK");
        } else {
          append(r, "SK");
        }
      } else {
        if (pos == 0 && !is_vowel(w, 3) && w.size() > 3 && w[3] != 'W') {
          append(r, 'X', 'S');
        } else {
          append(r, 'X');
        }
      }
      return pos + 3;
    }
    if (contains_at(w, pos + 2, 1, {"I", "E", "Y"})) {
      append(r, 'S');
      return pos + 3;
    }
    append(r, "SK");
    return pos + 3;
  }
  // French: silent S at end of word like "Debois"/"des".
  if (pos + 1 == n && contains_at(w, pos - 2, 2, {"AI", "OI"})) {
    r.alternate.push_back('S');
    return pos + 1;
  }
  append(r, 'S');
  return (pos + 1 < n && (w[pos + 1] == 'S' || w[pos + 1] == 'Z')) ? pos + 2 : pos + 1;
}

inline std::size_t handle_T(
    DoubleMetaphoneResult& r, const std::string& w, std::size_t pos) {
  const std::size_t n = w.size();
  if (contains_at(w, pos, 4, {"TION"})
      || contains_at(w, pos, 3, {"TIA", "TCH"})) {
    append(r, 'X');
    return pos + 3;
  }
  if (contains_at(w, pos, 2, {"TH"})
      || contains_at(w, pos, 3, {"TTH"})) {
    // Special case: Thomas, Thames.
    if (contains_at(w, pos + 2, 2, {"OM", "AM"})
        || contains_at(w, 0, 4, {"VAN ", "VON "})
        || contains_at(w, 0, 3, {"SCH"})) {
      append(r, 'T');
    } else {
      append(r, '0', 'T');
    }
    return pos + 2;
  }
  append(r, 'T');
  return (pos + 1 < n && (w[pos + 1] == 'T' || w[pos + 1] == 'D')) ? pos + 2 : pos + 1;
}

inline std::size_t handle_V(
    DoubleMetaphoneResult& r, const std::string& w, std::size_t pos) {
  append(r, 'F');
  return (pos + 1 < w.size() && w[pos + 1] == 'V') ? pos + 2 : pos + 1;
}

inline std::size_t handle_W(
    DoubleMetaphoneResult& r, const std::string& w, std::size_t pos) {
  const std::size_t n = w.size();
  // WR — like R.
  if (contains_at(w, pos, 2, {"WR"})) {
    append(r, 'R');
    return pos + 2;
  }
  if (pos == 0
      && (is_vowel(w, pos + 1) || contains_at(w, pos, 2, {"WH"}))) {
    if (is_vowel(w, pos + 1)) {
      append(r, 'A', 'F');
    } else {
      append(r, 'A');
    }
    return pos + 1;
  }
  if ((pos + 1 == n && is_vowel(w, pos - 1))
      || contains_at(w, pos - 1, 5, {"EWSKI", "EWSKY", "OWSKI", "OWSKY"})
      || contains_at(w, 0, 3, {"SCH"})) {
    r.alternate.push_back('F');
    return pos + 1;
  }
  if (contains_at(w, pos, 4, {"WICZ", "WITZ"})) {
    append(r, "TS", "FX");
    return pos + 4;
  }
  return pos + 1;
}

inline std::size_t handle_X(
    DoubleMetaphoneResult& r, const std::string& w, std::size_t pos) {
  if (pos == 0) {
    append(r, 'S');
    return pos + 1;
  }
  const std::size_t n = w.size();
  if (!(pos + 1 == n
        && (contains_at(w, pos - 3, 3, {"IAU", "EAU"})
            || contains_at(w, pos - 2, 2, {"AU", "OU"})))) {
    append(r, "KS");
  }
  return (pos + 1 < n && (w[pos + 1] == 'C' || w[pos + 1] == 'X')) ? pos + 2 : pos + 1;
}

inline std::size_t handle_Z(
    DoubleMetaphoneResult& r,
    const std::string& w,
    std::size_t pos,
    bool slavo_germanic) {
  if (pos + 1 < w.size() && w[pos + 1] == 'H') {
    append(r, 'J');
    return pos + 2;
  }
  // Apache Commons Codec emits the split "S"/"TS" only when the
  // preceding letter is NOT T — names like "Schwartz" (TZ ending) get
  // a single S so the alternate isn't doubled. We had the predicate
  // inverted in an earlier draft; see issue tested by Schwartz.
  if (contains_at(w, pos + 1, 2, {"ZO", "ZI", "ZA"})
      || (slavo_germanic && pos > 0
          && !contains_at(w, pos - 1, 1, {"T"}))) {
    append(r, std::string_view("S"), std::string_view("TS"));
  } else {
    append(r, 'S');
  }
  return (pos + 1 < w.size() && w[pos + 1] == 'Z') ? pos + 2 : pos + 1;
}

}  // namespace double_metaphone_detail

// max_length defaults to a generous bound rather than the
// historical 4 — modern reference implementations (the ``metaphone``
// Python package, Apache Commons Codec when configured) emit the
// full code and let the caller slice. Pass a smaller value if the
// classical 4-char Philips form is needed.
inline DoubleMetaphoneResult double_metaphone(
    std::string_view input,
    std::size_t max_length = 64,
    DoubleMetaphoneVariant variant = DoubleMetaphoneVariant::kCommons) {
  using namespace double_metaphone_detail;

  DoubleMetaphoneResult r;
  std::string w;
  w.reserve(input.size());
  for (char c : input) {
    const char uc = to_upper_ascii(c);
    if (is_upper_alpha(uc)) w.push_back(uc);
  }
  if (w.empty()) return r;

  const bool slavo = is_slavo_germanic(w);
  std::size_t pos = is_silent_start(w) ? 1 : 0;

  if (w.front() == 'X') {
    append(r, 'S');
    pos = 1;
  }

  while (pos < w.size() && !full(r, max_length)) {
    const char c = w[pos];
    switch (c) {
      case 'A': case 'E': case 'I': case 'O': case 'U': case 'Y':
        pos = handle_AEIOUY(r, pos);
        break;
      case 'B':
        pos = handle_B(r, w, pos);
        break;
      case 'C':
        pos = handle_C(r, w, pos, slavo);
        break;
      case 'D':
        pos = handle_D(r, w, pos);
        break;
      case 'F':
        append(r, 'F');
        pos += (pos + 1 < w.size() && w[pos + 1] == 'F') ? 2 : 1;
        break;
      case 'G':
        pos = handle_G(r, w, pos, slavo, variant);
        break;
      case 'H':
        pos = handle_H(r, w, pos);
        break;
      case 'J':
        pos = handle_J(r, w, pos, slavo);
        break;
      case 'K':
        append(r, 'K');
        pos += (pos + 1 < w.size() && w[pos + 1] == 'K') ? 2 : 1;
        break;
      case 'L':
        pos = handle_L(r, w, pos);
        break;
      case 'M':
        pos = handle_M(r, w, pos);
        break;
      case 'N':
        pos = handle_N(r, w, pos);
        break;
      case 'P':
        pos = handle_P(r, w, pos);
        break;
      case 'Q':
        pos = handle_Q(r, w, pos);
        break;
      case 'R':
        pos = handle_R(r, w, pos, slavo);
        break;
      case 'S':
        pos = handle_S(r, w, pos, slavo);
        break;
      case 'T':
        pos = handle_T(r, w, pos);
        break;
      case 'V':
        pos = handle_V(r, w, pos);
        break;
      case 'W':
        pos = handle_W(r, w, pos);
        break;
      case 'X':
        pos = handle_X(r, w, pos);
        break;
      case 'Z':
        pos = handle_Z(r, w, pos, slavo);
        break;
      default:
        ++pos;
        break;
    }
  }

  if (r.primary.size() > max_length) r.primary.resize(max_length);
  if (r.alternate.size() > max_length) r.alternate.resize(max_length);
  // The Apache Commons reference always emits both codes even when
  // identical. The Python ``metaphone`` package — our cross-check
  // oracle — emits an empty alternate when no two-arg ``append``
  // ever diverged it from primary. Match that here so callers can
  // cheaply test ``if alt:`` for "did the name have a second
  // pronunciation".
  if (r.primary == r.alternate) r.alternate.clear();
  return r;
}

}  // namespace stride_align::phonetic
