// Beider-Morse Phonetic Matching engine — implementation.
//
// Linked only into the ``_generic`` backend module (see CMakeLists.txt
// ``STRIDE_ALIGN_BMPM_BACKEND`` definition). Other backend modules
// expose ``beider_morse`` through the same Python-side re-export.
//
// See ``include/stride_align/beider_morse.hpp`` for the public API and
// the algorithm summary; the source-attribution block in that header
// covers the Apache Commons Codec lineage. The Aho-Corasick trie, the
// context-predicate classifier, the bump arena, and the parser are
// original C++.

#include "stride_align/beider_morse.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace stride_align::phonetic {

namespace {

// ----- UTF-8 / codepoint utilities ----------------------------------------

using Codepoint = std::uint32_t;
using CodepointVec = std::vector<Codepoint>;

// Decode a UTF-8 byte string into codepoints. Malformed sequences are
// replaced with U+FFFD so the rest of the engine never sees invalid
// bytes; BMPM rule files are valid UTF-8 by construction so this only
// matters for user input.
CodepointVec decode_utf8(std::string_view s) {
  CodepointVec out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size();) {
    const auto b0 = static_cast<std::uint8_t>(s[i]);
    Codepoint cp = 0xFFFD;
    std::size_t consumed = 1;
    if (b0 < 0x80) {
      cp = b0;
    } else if ((b0 & 0xE0) == 0xC0 && i + 1 < s.size()) {
      const auto b1 = static_cast<std::uint8_t>(s[i + 1]);
      if ((b1 & 0xC0) == 0x80) {
        cp = ((b0 & 0x1F) << 6) | (b1 & 0x3F);
        consumed = 2;
      }
    } else if ((b0 & 0xF0) == 0xE0 && i + 2 < s.size()) {
      const auto b1 = static_cast<std::uint8_t>(s[i + 1]);
      const auto b2 = static_cast<std::uint8_t>(s[i + 2]);
      if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80) {
        cp = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
        consumed = 3;
      }
    } else if ((b0 & 0xF8) == 0xF0 && i + 3 < s.size()) {
      const auto b1 = static_cast<std::uint8_t>(s[i + 1]);
      const auto b2 = static_cast<std::uint8_t>(s[i + 2]);
      const auto b3 = static_cast<std::uint8_t>(s[i + 3]);
      if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80 && (b3 & 0xC0) == 0x80) {
        cp = ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) |
             ((b2 & 0x3F) << 6) | (b3 & 0x3F);
        consumed = 4;
      }
    }
    out.push_back(cp);
    i += consumed;
  }
  return out;
}

// Append a codepoint to ``out`` as UTF-8.
void encode_utf8(std::string& out, Codepoint cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

std::string codepoints_to_utf8(const CodepointVec& v) {
  std::string out;
  out.reserve(v.size());
  for (const auto cp : v) encode_utf8(out, cp);
  return out;
}

// ASCII-only lower-case fold, mirroring Java's ``Locale.ENGLISH``.
// Non-ASCII codepoints pass through unchanged.
Codepoint to_lower_ascii(Codepoint cp) {
  return (cp >= 'A' && cp <= 'Z') ? cp + ('a' - 'A') : cp;
}

void to_lower_in_place(CodepointVec& v) {
  for (auto& cp : v) cp = to_lower_ascii(cp);
}

// ----- Language set --------------------------------------------------------

// GENERIC name type has 19 entries in ``gen_languages.txt`` (including
// ``any``). We assign bit positions in load order; ``any`` is not a
// real language but a sentinel — modelled as ``is_any`` flag.
struct LangRegistry {
  std::vector<std::string> names;          // index -> name (excluding "any")
  std::unordered_map<std::string, int> index_of;  // name -> bit index
  bool finalised = false;

  int add(const std::string& name) {
    if (name == "any") return -1;  // sentinel
    auto it = index_of.find(name);
    if (it != index_of.end()) return it->second;
    const int idx = static_cast<int>(names.size());
    names.push_back(name);
    index_of.emplace(name, idx);
    return idx;
  }
};

struct LangSet {
  std::uint32_t bits = 0;
  bool any = false;

  bool empty() const noexcept { return !any && bits == 0; }
  bool is_any() const noexcept { return any; }
  bool singleton() const noexcept {
    return !any && bits != 0 && (bits & (bits - 1)) == 0;
  }
  int first_index() const noexcept {
    if (bits == 0) return -1;
    return __builtin_ctz(bits);
  }

  static LangSet any_set() { return {0, true}; }
  static LangSet none() { return {0, false}; }

  bool operator==(const LangSet& other) const noexcept {
    return bits == other.bits && any == other.any;
  }

  LangSet restrict_to(const LangSet& other) const {
    if (empty() || other.empty()) return none();
    if (any) return other;
    if (other.any) return *this;
    LangSet r{bits & other.bits, false};
    if (r.bits == 0) return none();
    return r;
  }

  LangSet merge_with(const LangSet& other) const {
    if (empty()) return other;
    if (other.empty()) return *this;
    if (any || other.any) return any_set();
    return {bits | other.bits, false};
  }
};

struct LangSetHash {
  std::size_t operator()(const LangSet& s) const noexcept {
    return (static_cast<std::size_t>(s.bits) << 1) | (s.any ? 1 : 0);
  }
};

// Parse a "+"-separated language string ("english+french") into a LangSet.
LangSet parse_lang_set(std::string_view spec, const LangRegistry& reg) {
  LangSet out = LangSet::none();
  std::size_t start = 0;
  while (start <= spec.size()) {
    std::size_t end = spec.find('+', start);
    if (end == std::string_view::npos) end = spec.size();
    std::string name(spec.substr(start, end - start));
    if (name == "any") {
      out = out.merge_with(LangSet::any_set());
    } else {
      auto it = reg.index_of.find(name);
      if (it != reg.index_of.end()) {
        out.bits |= (1u << it->second);
      }
    }
    if (end >= spec.size()) break;
    start = end + 1;
  }
  return out;
}

// ----- Phoneme + PhonemeExpr (codepoint-based) ----------------------------

struct Phoneme {
  CodepointVec text;
  LangSet languages;
};

struct PhonemeExpr {
  // A rule's right-hand side is either a single phoneme or a list of
  // alternatives. We store the alternatives flat; ``size() == 1`` is
  // the single-phoneme case.
  std::vector<Phoneme> phonemes;
};

// ----- Resource file parsing (comment-strip + #include resolve) -----------

// Token split on whitespace, respecting double-quoted strings.
struct TokenError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Whitespace-split then strip exactly one leading and one trailing
// double quote per token. Matches Java's ``ResourceConstants.SPACES.split``
// followed by ``Rule.stripQuotes`` — neither side handles ``\"`` escapes,
// which is fine because the upstream rule files don't rely on them
// (the one ``"\""`` rule in ``gen_rules_russian.txt`` ends up with
// pattern ``\"`` in both ports, never matching real input).
std::vector<std::string> tokenise_quoted(std::string_view line) {
  std::vector<std::string> out;
  std::size_t i = 0;
  while (i < line.size()) {
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
    if (i >= line.size()) break;
    const std::size_t start = i;
    while (i < line.size() && line[i] != ' ' && line[i] != '\t') ++i;
    std::string token(line.substr(start, i - start));
    if (!token.empty() && token.front() == '"') token.erase(0, 1);
    if (!token.empty() && token.back() == '"') token.pop_back();
    out.push_back(std::move(token));
  }
  return out;
}

// Strip block comments (``/* ... */``) and end-of-line ``// ...`` comments,
// returning the cleaned source as a single string. The vendored rule files
// have only one block comment (the ASF header at the top), but a handful
// have inline blocks too.
std::string strip_comments(std::string_view src) {
  std::string out;
  out.reserve(src.size());
  std::size_t i = 0;
  while (i < src.size()) {
    if (i + 1 < src.size() && src[i] == '/' && src[i + 1] == '*') {
      const std::size_t close = src.find("*/", i + 2);
      if (close == std::string_view::npos) break;
      i = close + 2;
      continue;
    }
    if (i + 1 < src.size() && src[i] == '/' && src[i + 1] == '/') {
      while (i < src.size() && src[i] != '\n') ++i;
      continue;
    }
    out.push_back(src[i]);
    ++i;
  }
  return out;
}

// Split into trimmed, non-empty lines.
std::vector<std::string> split_lines(std::string_view src) {
  std::vector<std::string> out;
  std::size_t i = 0;
  while (i < src.size()) {
    std::size_t end = src.find('\n', i);
    if (end == std::string_view::npos) end = src.size();
    std::string_view line = src.substr(i, end - i);
    while (!line.empty() && (line.front() == ' ' || line.front() == '\t' || line.front() == '\r')) {
      line.remove_prefix(1);
    }
    while (!line.empty() && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r')) {
      line.remove_suffix(1);
    }
    if (!line.empty()) out.emplace_back(line);
    i = end + 1;
  }
  return out;
}

// ----- Context predicate classification -----------------------------------

// Pre-classified left-context / right-context predicate. Built at static
// init; matched in the hot loop without touching ``std::regex`` for the
// common cases that profile data showed cover the rule files.
struct ContextPred {
  enum class Kind : std::uint8_t {
    kAlways,           // empty
    kStartAnchor,      // lcontext only: matches iff pos == 0
    kEndAnchor,        // rcontext only: matches iff pos+plen == input.size()
    kCharClass,        // single codepoint at the adjacent position in set
    kNegCharClass,     // single codepoint at the adjacent position NOT in set
    kAnchoredCharClass,    // start-anchored (l) or end-anchored (r) + class
    kAnchoredNegCharClass,
    kLiteralEnd,       // lcontext: input ends with literal
    kLiteralStart,     // rcontext: input starts at pos+plen with literal
    kAnchoredLiteral,  // exact-match of input slice to literal
    kRegex,            // fallback
  };

  Kind kind = Kind::kAlways;
  bool is_left = false;          // lcontext (vs rcontext)
  std::unordered_set<Codepoint> char_set;
  CodepointVec literal;
  std::shared_ptr<std::regex> regex;

  // Match against ``input[0..pos)`` (lcontext) or ``input[pos..)``
  // (rcontext, where ``pos`` is already advanced past the matched
  // pattern).
  bool match(const CodepointVec& input,
             std::size_t pos,
             std::string_view raw_byte_input,
             std::size_t raw_byte_pos) const;
};

// Parse a ``[...]`` char-class body (after the ``[``, before the ``]``).
// Returns the codepoint set; ``negated`` is set if the body starts with
// ``^``.
std::unordered_set<Codepoint> parse_char_class(std::string_view body, bool& negated) {
  negated = false;
  if (!body.empty() && body.front() == '^') {
    negated = true;
    body.remove_prefix(1);
  }
  const CodepointVec cps = decode_utf8(body);
  return std::unordered_set<Codepoint>(cps.begin(), cps.end());
}

// Returns true if the regex-like fragment contains any character that
// is meaningful in a Java/PCRE regex (other than the constructs we
// already classify — ``^``, ``$``, single ``[...]`` at start/end).
bool needs_regex_fallback(std::string_view body) {
  // The classifier handles: empty, ^, $, single literal, [class],
  // [class]$, ^[class], ^[class]$, literal$, ^literal, ^literal$.
  // Anything else (alternation '|', '(', '*', '+', '?', escapes,
  // multiple char classes) goes to std::regex.
  bool seen_open = false;
  bool seen_close_at_end = false;
  for (std::size_t i = 0; i < body.size(); ++i) {
    const char c = body[i];
    if (c == '[') {
      if (seen_open) return true;
      seen_open = true;
      continue;
    }
    if (c == ']') {
      if (i != body.size() - 1 && !(i == body.size() - 2 && body.back() == '$')) {
        return true;
      }
      seen_close_at_end = true;
      continue;
    }
    if (c == '(' || c == '|' || c == '*' || c == '+' || c == '?' ||
        c == '\\' || c == '.') {
      return true;
    }
  }
  (void)seen_close_at_end;
  return false;
}

ContextPred classify_context(std::string_view raw, bool is_left) {
  ContextPred p;
  p.is_left = is_left;

  if (raw.empty()) {
    p.kind = ContextPred::Kind::kAlways;
    return p;
  }
  if (raw == "^") {
    p.kind = is_left ? ContextPred::Kind::kStartAnchor
                     : ContextPred::Kind::kAlways;
    return p;
  }
  if (raw == "$") {
    p.kind = is_left ? ContextPred::Kind::kAlways
                     : ContextPred::Kind::kEndAnchor;
    return p;
  }

  const bool start_anchor = !raw.empty() && raw.front() == '^';
  const bool end_anchor = !raw.empty() && raw.back() == '$';
  std::string_view body = raw;
  if (start_anchor) body.remove_prefix(1);
  if (end_anchor) body.remove_suffix(1);

  if (needs_regex_fallback(body)) {
    p.kind = ContextPred::Kind::kRegex;
    // Java wraps with anchors:
    //   lcontext -> ``raw + "$"`` against ``input[0..pos)``
    //   rcontext -> ``"^" + raw`` against ``input[pos..)``
    // We mirror that on the byte-level input slice.
    std::string compiled;
    if (is_left) {
      compiled.append(raw);
      compiled.append("$");
    } else {
      compiled.append("^");
      compiled.append(raw);
    }
    try {
      p.regex = std::make_shared<std::regex>(
          compiled, std::regex::ECMAScript | std::regex::optimize);
    } catch (const std::regex_error&) {
      p.kind = ContextPred::Kind::kAlways;  // safest fallback
    }
    return p;
  }

  // Single ``[...]`` body, possibly anchored on one side.
  if (!body.empty() && body.front() == '[' && body.back() == ']') {
    bool negated = false;
    p.char_set = parse_char_class(body.substr(1, body.size() - 2), negated);
    if (start_anchor || end_anchor) {
      p.kind = negated ? ContextPred::Kind::kAnchoredNegCharClass
                       : ContextPred::Kind::kAnchoredCharClass;
    } else {
      p.kind = negated ? ContextPred::Kind::kNegCharClass
                       : ContextPred::Kind::kCharClass;
    }
    return p;
  }

  // Pure literal, possibly anchored.
  p.literal = decode_utf8(body);
  if (start_anchor && end_anchor) {
    p.kind = ContextPred::Kind::kAnchoredLiteral;
  } else if (is_left) {
    p.kind = end_anchor ? ContextPred::Kind::kLiteralEnd
                        : ContextPred::Kind::kLiteralEnd;
  } else {
    p.kind = start_anchor ? ContextPred::Kind::kLiteralStart
                          : ContextPred::Kind::kLiteralStart;
  }
  return p;
}

bool ContextPred::match(const CodepointVec& input,
                        std::size_t pos,
                        std::string_view raw_byte_input,
                        std::size_t raw_byte_pos) const {
  switch (kind) {
    case Kind::kAlways:
      return true;
    case Kind::kStartAnchor:
      return is_left ? (pos == 0) : true;
    case Kind::kEndAnchor:
      return !is_left && pos == input.size();
    case Kind::kCharClass: {
      if (is_left) {
        if (pos == 0) return false;
        return char_set.count(input[pos - 1]) > 0;
      }
      if (pos >= input.size()) return false;
      return char_set.count(input[pos]) > 0;
    }
    case Kind::kNegCharClass: {
      if (is_left) {
        if (pos == 0) return false;
        return char_set.count(input[pos - 1]) == 0;
      }
      if (pos >= input.size()) return false;
      return char_set.count(input[pos]) == 0;
    }
    case Kind::kAnchoredCharClass: {
      // ``^[abc]`` (left) -> input is exactly one cp from set, i.e. pos == 1.
      // ``[abc]$`` (right) -> input has exactly one remaining cp from set.
      if (is_left) {
        return pos == 1 && char_set.count(input[0]) > 0;
      }
      return pos + 1 == input.size() && char_set.count(input[pos]) > 0;
    }
    case Kind::kAnchoredNegCharClass: {
      if (is_left) {
        return pos == 1 && char_set.count(input[0]) == 0;
      }
      return pos + 1 == input.size() && char_set.count(input[pos]) == 0;
    }
    case Kind::kLiteralEnd: {
      // lcontext: input[0..pos) ends with literal
      if (literal.size() > pos) return false;
      for (std::size_t k = 0; k < literal.size(); ++k) {
        if (input[pos - literal.size() + k] != literal[k]) return false;
      }
      return true;
    }
    case Kind::kLiteralStart: {
      // rcontext: input[pos..) starts with literal
      if (pos + literal.size() > input.size()) return false;
      for (std::size_t k = 0; k < literal.size(); ++k) {
        if (input[pos + k] != literal[k]) return false;
      }
      return true;
    }
    case Kind::kAnchoredLiteral: {
      if (is_left) {
        return pos == literal.size() &&
               std::equal(literal.begin(), literal.end(), input.begin());
      }
      return pos + literal.size() == input.size() &&
             std::equal(literal.begin(), literal.end(), input.begin() + pos);
    }
    case Kind::kRegex: {
      if (!regex) return false;
      std::string slice = is_left
          ? std::string(raw_byte_input.substr(0, raw_byte_pos))
          : std::string(raw_byte_input.substr(raw_byte_pos));
      return std::regex_search(slice, *regex);
    }
  }
  return false;
}

// ----- Rule ----------------------------------------------------------------

struct Rule {
  CodepointVec pattern;            // already lowered / decoded
  ContextPred lcontext;
  ContextPred rcontext;
  PhonemeExpr phoneme_expr;
};

// ----- PhonemeExpr parsing -------------------------------------------------

// Parse one phoneme token (possibly with ``[langs]`` suffix) into a Phoneme.
Phoneme parse_phoneme_token(std::string_view tok, const LangRegistry& reg) {
  // ``ph[lang1+lang2]`` -> text=ph, langs={lang1, lang2}.
  // No ``[`` -> any language.
  const auto open = tok.find('[');
  if (open == std::string_view::npos) {
    return {decode_utf8(tok), LangSet::any_set()};
  }
  if (tok.back() != ']') {
    return {decode_utf8(tok), LangSet::any_set()};
  }
  const auto before = tok.substr(0, open);
  const auto inner = tok.substr(open + 1, tok.size() - open - 2);
  return {decode_utf8(before), parse_lang_set(inner, reg)};
}

// Parse a phoneme expression: either a single ``"x"`` token or a
// parenthesised list ``"(a|b|c)"``. Mirrors Java
// ``Rule.parsePhonemeExpr`` including the trailing-pipe -> empty
// phoneme edge case (Java only checks leading/trailing pipe with
// ``length() != 0`` on the surviving side; we replicate that exactly).
PhonemeExpr parse_phoneme_expr(std::string_view src, const LangRegistry& reg) {
  PhonemeExpr expr;
  if (!src.empty() && src.front() == '(') {
    if (src.back() != ')') {
      throw std::runtime_error(
          std::string("phoneme expression missing ')': ") + std::string(src));
    }
    const auto body = src.substr(1, src.size() - 2);
    std::vector<std::string_view> parts;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= body.size(); ++i) {
      if (i == body.size() || body[i] == '|') {
        parts.push_back(body.substr(start, i - start));
        start = i + 1;
      }
    }
    expr.phonemes.reserve(parts.size());
    for (auto p : parts) expr.phonemes.push_back(parse_phoneme_token(p, reg));
    const bool first_nonempty_with_leading_pipe =
        parts.size() > 1 && !parts.front().empty() && body.front() == '|';
    const bool last_nonempty_with_trailing_pipe =
        !parts.empty() && !parts.back().empty() && body.back() == '|';
    if (first_nonempty_with_leading_pipe || last_nonempty_with_trailing_pipe) {
      expr.phonemes.push_back({CodepointVec(), LangSet::any_set()});
    }
    return expr;
  }
  expr.phonemes.push_back(parse_phoneme_token(src, reg));
  return expr;
}

// ----- Aho-Corasick trie ---------------------------------------------------

// One trie per rule list. Each rule sits at a terminal node; multiple
// rules with the same pattern chain into a per-node list. At lookup we
// walk the trie at the input position taking the LONGEST match whose
// context predicates also fire.
struct AhoNode {
  std::unordered_map<Codepoint, std::uint32_t> next;  // transitions
  std::uint32_t fail = 0;                              // failure link
  std::vector<std::uint32_t> rule_ids;                 // rules ending here
};

struct AhoTrie {
  std::vector<AhoNode> nodes{1};      // node 0 is the root
  std::vector<Rule> rules;

  void add_rule(Rule rule) {
    std::uint32_t cur = 0;
    for (const auto cp : rule.pattern) {
      auto it = nodes[cur].next.find(cp);
      if (it == nodes[cur].next.end()) {
        nodes.push_back({});
        const auto new_idx = static_cast<std::uint32_t>(nodes.size() - 1);
        nodes[cur].next.emplace(cp, new_idx);
        cur = new_idx;
      } else {
        cur = it->second;
      }
    }
    const auto rule_id = static_cast<std::uint32_t>(rules.size());
    nodes[cur].rule_ids.push_back(rule_id);
    rules.push_back(std::move(rule));
  }

  // BFS to populate fail links. After this, ``next`` is patched into a
  // goto-with-failure form so the lookup loop can move forward without
  // walking failure links explicitly.
  void build_failures() {
    std::vector<std::uint32_t> queue;
    for (auto& [_, child] : nodes[0].next) {
      nodes[child].fail = 0;
      queue.push_back(child);
    }
    for (std::size_t qi = 0; qi < queue.size(); ++qi) {
      const auto u = queue[qi];
      for (auto& [cp, v] : nodes[u].next) {
        std::uint32_t f = nodes[u].fail;
        while (f != 0 && nodes[f].next.find(cp) == nodes[f].next.end()) {
          f = nodes[f].fail;
        }
        auto it = nodes[f].next.find(cp);
        nodes[v].fail = (it != nodes[f].next.end() && it->second != v)
                            ? it->second
                            : 0;
        queue.push_back(v);
      }
    }
  }
};

// ----- Resource parser: ``parseRules`` equivalent -------------------------

struct ParsedRules {
  // pattern length, rule  — we process longest-first when matching at a
  // position to pick the best match.
  std::vector<Rule> rules;
};

class ResourceMap {
 public:
  explicit ResourceMap(const std::unordered_map<std::string, std::string>& src)
      : src_(src) {}

  // Returns the raw content for ``name``, or throws.
  const std::string& get(const std::string& name) const {
    auto it = src_.find(name);
    if (it == src_.end()) {
      throw std::runtime_error("BMPM resource missing: " + name);
    }
    return it->second;
  }

  bool has(const std::string& name) const { return src_.find(name) != src_.end(); }

 private:
  const std::unordered_map<std::string, std::string>& src_;
};

ParsedRules parse_rules(const ResourceMap& resources,
                        const std::string& name,
                        const LangRegistry& reg,
                        std::unordered_set<std::string>& visited) {
  ParsedRules out;
  if (visited.count(name)) return out;
  visited.insert(name);

  const std::string& raw = resources.get(name);
  const std::string stripped = strip_comments(raw);
  for (const auto& line : split_lines(stripped)) {
    static constexpr std::string_view kInclude = "#include";
    if (line.rfind(std::string(kInclude), 0) == 0) {
      auto rest = std::string_view(line).substr(kInclude.size());
      while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t')) {
        rest.remove_prefix(1);
      }
      auto included = parse_rules(resources, std::string(rest), reg, visited);
      out.rules.insert(out.rules.end(),
                       std::make_move_iterator(included.rules.begin()),
                       std::make_move_iterator(included.rules.end()));
      continue;
    }
    std::vector<std::string> parts;
    try {
      parts = tokenise_quoted(line);
    } catch (const TokenError& e) {
      throw std::runtime_error(std::string("BMPM rule parse error in ") +
                               name + ": " + e.what());
    }
    if (parts.size() != 4) {
      throw std::runtime_error("BMPM rule must have 4 quoted fields in " +
                               name + ": " + line);
    }
    Rule r;
    // Rule patterns are stored case-sensitive: the main-rules files use
    // lowercase patterns matching the lowercased user input, but the
    // approx/exact final-rules files use uppercase BMPM phoneme markers
    // (``"O" "" "" "o"``, ``"E" "" "" "e"``, etc.). Lower-casing here
    // would silently nullify ~260 normalisation rules across the
    // gen_approx_* and gen_exact_* files.
    r.pattern = decode_utf8(parts[0]);
    r.lcontext = classify_context(parts[1], /*is_left=*/true);
    r.rcontext = classify_context(parts[2], /*is_left=*/false);
    r.phoneme_expr = parse_phoneme_expr(parts[3], reg);
    out.rules.push_back(std::move(r));
  }
  return out;
}

ParsedRules parse_rules(const ResourceMap& resources,
                        const std::string& name,
                        const LangRegistry& reg) {
  std::unordered_set<std::string> visited;
  return parse_rules(resources, name, reg, visited);
}

// ----- Lang language-guesser ----------------------------------------------

struct LangRule {
  std::regex pattern;
  std::uint32_t lang_bits = 0;
  bool langs_any = false;
  bool accept_on_match = true;
};

struct Lang {
  std::vector<LangRule> rules;
  std::uint32_t all_bits = 0;  // all real languages in the registry

  LangSet guess(std::string_view input_utf8) const {
    // Match against the lowercased UTF-8 input. Java does the same.
    std::string lowered(input_utf8);
    for (auto& c : lowered) {
      if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - ('A' - 'a'));
    }
    std::uint32_t live = all_bits;
    for (const auto& r : rules) {
      if (std::regex_search(lowered, r.pattern)) {
        if (r.accept_on_match) {
          live &= r.lang_bits;
        } else {
          live &= ~r.lang_bits;
        }
      }
    }
    if (live == 0) return LangSet::any_set();
    return LangSet{live, false};
  }
};

Lang parse_lang(const ResourceMap& resources,
                const std::string& name,
                const LangRegistry& reg) {
  Lang out;
  out.all_bits = (reg.names.empty() ? 0u : ((1u << reg.names.size()) - 1));
  const std::string& raw = resources.get(name);
  const std::string stripped = strip_comments(raw);
  for (const auto& line : split_lines(stripped)) {
    // Three whitespace-separated fields. Java uses ``\s+``.
    std::vector<std::string_view> tokens;
    std::size_t i = 0;
    while (i < line.size()) {
      while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
      if (i >= line.size()) break;
      const std::size_t start = i;
      while (i < line.size() && line[i] != ' ' && line[i] != '\t') ++i;
      tokens.push_back(std::string_view(line).substr(start, i - start));
    }
    if (tokens.size() != 3) continue;
    LangRule r;
    try {
      r.pattern = std::regex(std::string(tokens[0]),
                             std::regex::ECMAScript | std::regex::optimize);
    } catch (const std::regex_error&) {
      continue;
    }
    LangSet langs = parse_lang_set(tokens[1], reg);
    r.lang_bits = langs.bits;
    r.langs_any = langs.any;
    r.accept_on_match = (tokens[2] == "true");
    out.rules.push_back(std::move(r));
  }
  return out;
}

// ----- Final rules: lookup map (language -> rule list) --------------------

struct FinalRules {
  std::unordered_map<std::string, std::vector<Rule>> by_language;
};

// ----- Compiled tables singleton ------------------------------------------

struct BmpmTables {
  LangRegistry registry;
  std::unordered_map<std::string, std::vector<Rule>> main_rules;  // lang -> rules
  std::unordered_map<std::string, std::vector<Rule>> approx_rules;
  std::unordered_map<std::string, std::vector<Rule>> exact_rules;
  Lang lang;
};

std::unique_ptr<BmpmTables> g_tables;
std::once_flag g_tables_once;
std::mutex g_pending_mutex;
std::unique_ptr<std::unordered_map<std::string, std::string>> g_pending_resources;

void load_registry(const ResourceMap& res, LangRegistry& reg) {
  const std::string& raw = res.get("gen_languages");
  const std::string stripped = strip_comments(raw);
  for (const auto& line : split_lines(stripped)) {
    reg.add(line);
  }
  reg.finalised = true;
}

std::vector<Rule> rules_for(const ResourceMap& res,
                            const std::string& prefix,
                            const std::string& lang,
                            const LangRegistry& reg) {
  return parse_rules(res, prefix + "_" + lang, reg).rules;
}

std::unique_ptr<BmpmTables> build_tables(
    const std::unordered_map<std::string, std::string>& src) {
  auto tables = std::make_unique<BmpmTables>();
  ResourceMap res(src);
  load_registry(res, tables->registry);

  const auto& reg = tables->registry;
  std::vector<std::string> languages_to_load = reg.names;
  languages_to_load.push_back("any");

  for (const auto& lang : languages_to_load) {
    tables->main_rules[lang] = rules_for(res, "gen_rules", lang, reg);
  }
  for (const auto& lang : languages_to_load) {
    tables->approx_rules[lang] = rules_for(res, "gen_approx", lang, reg);
    tables->exact_rules[lang]  = rules_for(res, "gen_exact",  lang, reg);
  }
  if (res.has("gen_approx_common")) {
    tables->approx_rules["common"] =
        parse_rules(res, "gen_approx_common", reg).rules;
  }
  if (res.has("gen_exact_common")) {
    tables->exact_rules["common"] =
        parse_rules(res, "gen_exact_common", reg).rules;
  }

  tables->lang = parse_lang(res, "gen_lang", reg);
  return tables;
}

void init_tables_if_needed() {
  std::call_once(g_tables_once, []() {
    std::unique_ptr<std::unordered_map<std::string, std::string>> pending;
    {
      std::lock_guard<std::mutex> lock(g_pending_mutex);
      pending = std::move(g_pending_resources);
    }
    if (!pending) {
      throw std::runtime_error(
          "BMPM resources have not been registered — call "
          "bmpm_register_resources() first");
    }
    g_tables = build_tables(*pending);
  });
}

// ----- PhonemeBuilder: arena-allocated set --------------------------------

// Per-call bump arena. Phoneme texts get appended into ``buffer``; the
// vector of (offset, length, langset) tuples is the live phoneme set.
// We do not deduplicate by text identity for the duration of the encode
// loop — duplicates inside a single ``apply`` are kept because their
// LangSets differ; at the end we collapse them with a LangSet merge.
struct ArenaPhoneme {
  std::uint32_t offset;
  std::uint32_t length;
  LangSet langs;
};

struct PhonemeArena {
  std::vector<Codepoint> buffer;
  std::vector<ArenaPhoneme> phonemes;
  std::vector<ArenaPhoneme> scratch;

  void reset() {
    buffer.clear();
    phonemes.clear();
    scratch.clear();
  }

  std::uint32_t append(const CodepointVec& text) {
    const auto off = static_cast<std::uint32_t>(buffer.size());
    buffer.insert(buffer.end(), text.begin(), text.end());
    return off;
  }

  void start_with_languages(const LangSet& langs) {
    phonemes.clear();
    phonemes.push_back({0, 0, langs});
  }

  // Apply a phoneme expression to every live phoneme: Cartesian product
  // restricted by language-set intersection. Java's behaviour exactly.
  void apply(const PhonemeExpr& expr, std::size_t max_phonemes) {
    scratch.clear();
    scratch.reserve(std::min(phonemes.size() * expr.phonemes.size(),
                             max_phonemes));
    for (const auto& left : phonemes) {
      for (const auto& right : expr.phonemes) {
        const LangSet joined = left.langs.restrict_to(right.languages);
        if (joined.empty()) continue;
        ArenaPhoneme combined;
        combined.langs = joined;
        if (right.text.empty()) {
          combined.offset = left.offset;
          combined.length = left.length;
        } else {
          // Copy the left slice out FIRST: ``buffer.insert(buffer.end(),
          // buffer.begin()+x, buffer.begin()+y)`` is UB when the inserts
          // trigger a reallocation — the source iterators then refer to
          // freed storage.
          const CodepointVec left_copy(
              buffer.begin() + left.offset,
              buffer.begin() + left.offset + left.length);
          combined.offset = static_cast<std::uint32_t>(buffer.size());
          buffer.insert(buffer.end(), left_copy.begin(), left_copy.end());
          buffer.insert(buffer.end(), right.text.begin(), right.text.end());
          combined.length = left.length + static_cast<std::uint32_t>(right.text.size());
        }
        if (scratch.size() < max_phonemes) {
          scratch.push_back(combined);
        } else {
          goto done;
        }
      }
    }
  done:
    phonemes.swap(scratch);
  }

  void append_literal(const CodepointVec& tail) {
    if (tail.empty()) return;
    for (auto& ph : phonemes) {
      // Self-insert UB guard — same as ``apply``.
      const CodepointVec slice(buffer.begin() + ph.offset,
                                buffer.begin() + ph.offset + ph.length);
      const auto new_off = static_cast<std::uint32_t>(buffer.size());
      buffer.insert(buffer.end(), slice.begin(), slice.end());
      buffer.insert(buffer.end(), tail.begin(), tail.end());
      ph.offset = new_off;
      ph.length += static_cast<std::uint32_t>(tail.size());
    }
  }
};

// Render the arena's live phoneme set to a ``|``-joined UTF-8 string.
// Deduplicates by phoneme text (merging language sets of duplicates,
// matching Java's ``applyFinalRules`` merge step) and sorts the result
// by phoneme text — Java does this implicitly because
// ``applyFinalRules`` stores phonemes in a ``TreeMap<Phoneme, Phoneme,
// Phoneme.COMPARATOR>`` whose ``keySet()`` iterates in sorted order.
// Sorting on the C++ side gives byte-identical output to the upstream.
std::string render_arena(const PhonemeArena& a) {
  struct Group { CodepointVec text; LangSet langs; };
  std::vector<Group> groups;
  groups.reserve(a.phonemes.size());
  for (const auto& ph : a.phonemes) {
    CodepointVec text(a.buffer.begin() + ph.offset,
                      a.buffer.begin() + ph.offset + ph.length);
    bool merged = false;
    for (auto& g : groups) {
      if (g.text == text) {
        g.langs = g.langs.merge_with(ph.langs);
        merged = true;
        break;
      }
    }
    if (!merged) groups.push_back({std::move(text), ph.langs});
  }
  std::sort(groups.begin(), groups.end(),
            [](const Group& a, const Group& b) { return a.text < b.text; });
  std::string out;
  for (std::size_t i = 0; i < groups.size(); ++i) {
    if (i != 0) out.push_back('|');
    out.append(codepoints_to_utf8(groups[i].text));
  }
  return out;
}

// ----- Rule lookup at a position ------------------------------------------

// Linear scan with first-codepoint hash bucketing. Aho-Corasick was the
// initial plan but a per-position one-shot lookup against ~200 rules
// where the typical match is a 1-2 cp pattern is dominated by the
// context-predicate check, not the trie walk. We still bucket by first
// codepoint and pick the longest-pattern winner.
struct RuleBuckets {
  std::unordered_map<Codepoint, std::vector<std::uint32_t>> by_first;
  std::vector<Rule> rules;
};

RuleBuckets bucketise(std::vector<Rule>&& rules) {
  RuleBuckets b;
  b.rules = std::move(rules);
  // Sort within each bucket by descending pattern length, then by
  // ascending insertion order — so when two rules share both the
  // first codepoint and the pattern length, the upstream file-order
  // tiebreak wins. ``Rule.parseRules`` in Apache Commons Codec relies
  // on this for the GENERIC ``"v" "^" "" "(v|f[german]|b[spanish])"``
  // rule to fire before the unconditional ``"v" "" "" "V"`` fallback
  // at the start of a word; without the file-order tiebreak, the
  // language-restricted alternation never reaches the encode loop.
  std::vector<std::pair<Codepoint, std::uint32_t>> seq;
  seq.reserve(b.rules.size());
  for (std::uint32_t i = 0; i < b.rules.size(); ++i) {
    if (b.rules[i].pattern.empty()) continue;
    seq.emplace_back(b.rules[i].pattern.front(), i);
  }
  std::stable_sort(seq.begin(), seq.end(),
                   [&](const auto& a, const auto& c) {
                     if (a.first != c.first) return a.first < c.first;
                     return b.rules[a.second].pattern.size() >
                            b.rules[c.second].pattern.size();
                   });
  for (auto& [cp, idx] : seq) b.by_first[cp].push_back(idx);
  return b;
}

// Returns (matched, pattern_length, phoneme_expr_index_in_rules) for the
// best rule at position ``pos``. ``pattern_length`` is 0 if no rule
// matched.
struct RuleHit {
  bool found;
  std::size_t pattern_length;
  const PhonemeExpr* expr;
};

RuleHit find_rule(const RuleBuckets& buckets,
                  const CodepointVec& input,
                  std::size_t pos,
                  std::string_view raw_byte_input,
                  const std::vector<std::size_t>& cp_to_byte) {
  if (pos >= input.size()) return {false, 0, nullptr};
  auto it = buckets.by_first.find(input[pos]);
  if (it == buckets.by_first.end()) return {false, 0, nullptr};
  for (auto rule_idx : it->second) {
    const Rule& r = buckets.rules[rule_idx];
    const std::size_t plen = r.pattern.size();
    if (pos + plen > input.size()) continue;
    bool eq = true;
    for (std::size_t k = 0; k < plen; ++k) {
      if (input[pos + k] != r.pattern[k]) { eq = false; break; }
    }
    if (!eq) continue;
    // lcontext applies to ``input[0..pos)``: byte slice is
    // ``raw_byte_input[0..cp_to_byte[pos])``.
    if (!r.lcontext.match(input, pos, raw_byte_input, cp_to_byte[pos])) continue;
    // rcontext applies to ``input[pos+plen..)``: byte slice starts at
    // ``cp_to_byte[pos+plen]``.
    if (!r.rcontext.match(input, pos + plen, raw_byte_input,
                           cp_to_byte[pos + plen])) {
      continue;
    }
    return {true, plen, &r.phoneme_expr};
  }
  return {false, 0, nullptr};
}

// ----- Per-thread arena ---------------------------------------------------

PhonemeArena& thread_arena() {
  thread_local PhonemeArena a;
  return a;
}

// Live RuleBuckets caches built lazily from BmpmTables.
struct BucketCache {
  std::unordered_map<std::string, RuleBuckets> main;
  std::unordered_map<std::string, RuleBuckets> approx;
  std::unordered_map<std::string, RuleBuckets> exact;
  std::once_flag once;
};

BucketCache& buckets_cache() {
  static BucketCache c;
  return c;
}

void init_buckets_if_needed() {
  auto& c = buckets_cache();
  std::call_once(c.once, []() {
    auto& tables = *g_tables;
    auto& cache = buckets_cache();
    for (auto& [lang, rules] : tables.main_rules) {
      std::vector<Rule> copy = rules;
      cache.main.emplace(lang, bucketise(std::move(copy)));
    }
    for (auto& [lang, rules] : tables.approx_rules) {
      std::vector<Rule> copy = rules;
      cache.approx.emplace(lang, bucketise(std::move(copy)));
    }
    for (auto& [lang, rules] : tables.exact_rules) {
      std::vector<Rule> copy = rules;
      cache.exact.emplace(lang, bucketise(std::move(copy)));
    }
  });
}

// ----- Prefix handling ----------------------------------------------------

// GENERIC prefixes from upstream PhoneticEngine.NAME_PREFIXES (GENERIC).
static const std::vector<std::string>& generic_prefixes() {
  static const std::vector<std::string> p = {
      "da", "dal", "de", "del", "dela", "de la", "della",
      "des", "di", "do", "dos", "du", "van", "von",
  };
  return p;
}

// ----- Encoding -----------------------------------------------------------

// Forward decl for d'/prefix recursion.
std::string encode_impl(const std::string& input,
                        BmpmRuleType rule_type,
                        bool concat,
                        std::size_t max_phonemes,
                        const LangSet& languages);

std::string encode_one_word(std::string_view input,
                            BmpmRuleType rule_type,
                            std::size_t max_phonemes,
                            const LangSet& languages) {
  auto& arena = thread_arena();
  arena.reset();
  arena.start_with_languages(languages);

  CodepointVec cps = decode_utf8(input);
  to_lower_in_place(cps);

  // We need byte offsets aligned to codepoint indices for the regex
  // fallback. Build a lookup of cp_index -> byte_offset for the
  // pre-lowered (codepoint) form re-encoded.
  std::string lowered_bytes = codepoints_to_utf8(cps);
  std::vector<std::size_t> cp_to_byte(cps.size() + 1, 0);
  {
    std::size_t b = 0;
    for (std::size_t i = 0; i < cps.size(); ++i) {
      cp_to_byte[i] = b;
      std::string tmp;
      encode_utf8(tmp, cps[i]);
      b += tmp.size();
    }
    cp_to_byte[cps.size()] = b;
  }

  const auto& cache = buckets_cache();
  auto bucket_for = [&](const std::string& lang) -> const RuleBuckets* {
    auto it = cache.main.find(lang);
    if (it == cache.main.end()) return nullptr;
    return &it->second;
  };

  // Pick the rule set: singleton language uses its own; otherwise "any".
  const RuleBuckets* main_b = nullptr;
  if (languages.singleton()) {
    const int idx = languages.first_index();
    if (idx >= 0 && idx < static_cast<int>(g_tables->registry.names.size())) {
      main_b = bucket_for(g_tables->registry.names[idx]);
    }
  }
  if (main_b == nullptr) main_b = bucket_for("any");
  if (main_b == nullptr) return {};

  std::size_t i = 0;
  while (i < cps.size()) {
    auto hit = find_rule(*main_b, cps, i, lowered_bytes, cp_to_byte);
    if (!hit.found) {
      ++i;
      continue;
    }
    arena.apply(*hit.expr, max_phonemes);
    i += hit.pattern_length;
  }

  // Final rules: apply common then language-specific. Each final-rule
  // pass takes each live phoneme's TEXT as the new input and re-encodes
  // it through the final-rule bucket set. Phonemes with empty language
  // sets drop. Within each pass, we rebuild the live set.
  auto apply_final = [&](const std::string& lang_label) {
    const auto& bucket_map = (rule_type == BmpmRuleType::kApprox)
                                  ? cache.approx
                                  : cache.exact;
    auto it = bucket_map.find(lang_label);
    if (it == bucket_map.end()) return;
    const RuleBuckets& b = it->second;
    if (b.rules.empty() && b.by_first.empty()) return;

    PhonemeArena next_arena;
    next_arena.reset();
    for (const auto& ph : arena.phonemes) {
      CodepointVec sub(arena.buffer.begin() + ph.offset,
                       arena.buffer.begin() + ph.offset + ph.length);
      std::string sub_bytes = codepoints_to_utf8(sub);
      std::vector<std::size_t> sub_cp_to_byte(sub.size() + 1, 0);
      {
        std::size_t bb = 0;
        for (std::size_t k = 0; k < sub.size(); ++k) {
          sub_cp_to_byte[k] = bb;
          std::string tmp;
          encode_utf8(tmp, sub[k]);
          bb += tmp.size();
        }
        sub_cp_to_byte[sub.size()] = bb;
      }

      PhonemeArena local;
      local.reset();
      local.start_with_languages(ph.langs);
      std::size_t k = 0;
      while (k < sub.size()) {
        auto hit = find_rule(b, sub, k, sub_bytes, sub_cp_to_byte);
        if (hit.found) {
          local.apply(*hit.expr, max_phonemes);
          k += hit.pattern_length;
        } else {
          CodepointVec one{sub[k]};
          local.append_literal(one);
          ++k;
        }
      }
      // Merge ``local.phonemes`` into ``next_arena.phonemes`` by text+langset.
      for (const auto& p : local.phonemes) {
        ArenaPhoneme moved;
        moved.langs = p.langs;
        moved.offset = static_cast<std::uint32_t>(next_arena.buffer.size());
        moved.length = p.length;
        next_arena.buffer.insert(next_arena.buffer.end(),
                                 local.buffer.begin() + p.offset,
                                 local.buffer.begin() + p.offset + p.length);
        next_arena.phonemes.push_back(moved);
      }
    }
    arena.buffer.swap(next_arena.buffer);
    arena.phonemes.swap(next_arena.phonemes);
  };

  apply_final("common");
  if (languages.singleton()) {
    const int idx = languages.first_index();
    if (idx >= 0 && idx < static_cast<int>(g_tables->registry.names.size())) {
      apply_final(g_tables->registry.names[idx]);
    }
  } else {
    apply_final("any");
  }

  return render_arena(arena);
}

std::string encode_impl(const std::string& input,
                        BmpmRuleType rule_type,
                        bool concat,
                        std::size_t max_phonemes,
                        const LangSet& languages) {
  // Generic d' prefix handling.
  if (input.size() >= 2 && input[0] == 'd' && input[1] == '\'') {
    const std::string remainder = input.substr(2);
    const std::string combined = "d" + remainder;
    return "(" + encode_impl(remainder, rule_type, concat, max_phonemes,
                              g_tables->lang.guess(remainder)) +
           ")-(" + encode_impl(combined, rule_type, concat, max_phonemes,
                                g_tables->lang.guess(combined)) +
           ")";
  }
  for (const auto& prefix : generic_prefixes()) {
    const std::string with_space = prefix + " ";
    if (input.size() >= with_space.size() &&
        input.compare(0, with_space.size(), with_space) == 0) {
      const std::string remainder = input.substr(with_space.size());
      const std::string combined = prefix + remainder;
      return "(" + encode_impl(remainder, rule_type, concat, max_phonemes,
                                g_tables->lang.guess(remainder)) +
             ")-(" + encode_impl(combined, rule_type, concat, max_phonemes,
                                  g_tables->lang.guess(combined)) +
             ")";
    }
  }

  // Split on whitespace; GENERIC does not strip prefix words.
  std::vector<std::string_view> words;
  std::size_t s = 0;
  for (std::size_t k = 0; k <= input.size(); ++k) {
    if (k == input.size() || input[k] == ' ') {
      if (k > s) words.emplace_back(std::string_view(input).substr(s, k - s));
      s = k + 1;
    }
  }
  if (words.empty()) return {};

  if (concat) {
    std::string joined;
    for (std::size_t k = 0; k < words.size(); ++k) {
      if (k != 0) joined.push_back(' ');
      joined.append(words[k]);
    }
    return encode_one_word(joined, rule_type, max_phonemes, languages);
  }

  if (words.size() == 1) {
    return encode_one_word(words[0], rule_type, max_phonemes, languages);
  }
  std::string out;
  for (std::size_t k = 0; k < words.size(); ++k) {
    if (k != 0) out.push_back('-');
    out.append(encode_one_word(words[k], rule_type, max_phonemes,
                                g_tables->lang.guess(words[k])));
  }
  return out;
}

}  // namespace

// ----- Public API ---------------------------------------------------------

void bmpm_register_resources(
    const std::unordered_map<std::string, std::string>& resources) {
  {
    std::lock_guard<std::mutex> lock(g_pending_mutex);
    if (!g_pending_resources) {
      g_pending_resources =
          std::make_unique<std::unordered_map<std::string, std::string>>(
              resources);
    }
  }
  // Trigger the call_once so subsequent calls don't race the first
  // encode call.
  init_tables_if_needed();
  init_buckets_if_needed();
}

std::string beider_morse(std::string_view input,
                         BmpmRuleType rule_type,
                         bool concat,
                         std::size_t max_phonemes) {
  init_tables_if_needed();
  init_buckets_if_needed();
  if (input.empty()) return {};

  // Normalise: lowercase ASCII, ``-`` -> ``' '``, trim.
  std::string norm(input);
  for (auto& c : norm) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - ('A' - 'a'));
    else if (c == '-') c = ' ';
  }
  std::size_t l = 0, r = norm.size();
  while (l < r && norm[l] == ' ') ++l;
  while (r > l && norm[r - 1] == ' ') --r;
  norm = norm.substr(l, r - l);
  if (norm.empty()) return {};

  const LangSet languages = g_tables->lang.guess(norm);
  return encode_impl(norm, rule_type, concat, max_phonemes, languages);
}

}  // namespace stride_align::phonetic
