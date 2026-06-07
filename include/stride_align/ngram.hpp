#pragma once

// N-gram set similarity — Phase D.1.
//
// Four metrics over the multiset of character n-grams of each input:
//
//   * **Jaccard.**   J(A, B) = |A ∩ B| / |A ∪ B|
//   * **Sørensen-Dice.** D(A, B) = 2·|A ∩ B| / (|A| + |B|)
//   * **Cosine.**    cos(A, B) = ⟨A, B⟩ / (‖A‖·‖B‖) over the
//                                 multiset frequency vectors.
//   * **Overlap.**   O(A, B) = |A ∩ B| / min(|A|, |B|)
//
// All four are computed over MULTISETS (each n-gram counted with
// multiplicity). For ``a = "abab"`` and ``n = 2`` the multiset is
// ``{"ab": 2, "ba": 1}``; for ``a = "aaaa"`` and ``n = 2`` it is
// ``{"aa": 3}``. Set-form behaviour falls out for inputs whose n-grams
// are all distinct.
//
// Identity / edge cases:
//
//   * Two empty inputs (or both inputs shorter than ``n``) give 1.0
//     — vacuously identical, matches the rapidfuzz / scikit-learn
//     convention.
//   * One empty and one non-empty gives 0.0.
//   * ``n == 0`` is rejected at the dispatch boundary.
//
// Engine works in codepoint space throughout. The dispatch wrapper
// widens Python ``str`` storage straight out of ``PyUnicode_DATA``
// into ``std::vector<Codepoint>`` — same convention as the other
// stride-align codepoint-engine entry points.
//
// Source: standard textbook formulas. The C++ here is original.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace stride_align::ngram {

using Codepoint = std::uint32_t;

// One key per n-gram: ``n`` consecutive codepoints packed as a binary
// ``std::string`` (``n * 4`` bytes). For default ``n = 2`` and typical
// ``n = 3 / 4`` the key fits in libstdc++'s small-string-optimisation
// buffer (15 bytes inline) and the multiset insert allocates only the
// hash-map node, not the key.
using NGramKey = std::string;
using NGramMultiset = std::unordered_map<NGramKey, std::uint32_t>;

// Construct the n-gram multiset of ``input``. When ``input.size() < n``
// the multiset is empty (the "shorter than n" edge case folds into the
// empty-input branch of every metric formula).
inline NGramMultiset build_multiset(const std::vector<Codepoint>& input,
                                     std::size_t n) {
  NGramMultiset m;
  if (n == 0 || input.size() < n) return m;
  const std::size_t count = input.size() - n + 1;
  m.reserve(count);
  NGramKey key(n * sizeof(Codepoint), '\0');
  for (std::size_t i = 0; i < count; ++i) {
    std::memcpy(key.data(), input.data() + i, n * sizeof(Codepoint));
    ++m[key];
  }
  return m;
}

// Aggregated stats from a one-pass comparison of two multisets. Enough
// information here to compute all four metrics without re-walking the
// inputs.
struct ComparisonStats {
  std::size_t size_a = 0;       // sum of count_a (i.e. |A| with multiplicity)
  std::size_t size_b = 0;
  std::size_t intersection = 0; // sum over keys of ``min(count_a, count_b)``
  std::uint64_t dot_product = 0;// sum over keys of ``count_a · count_b``
  std::uint64_t norm_a_sq = 0;  // sum over keys of ``count_a²``
  std::uint64_t norm_b_sq = 0;
};

inline ComparisonStats compare(const NGramMultiset& a, const NGramMultiset& b) {
  ComparisonStats s;
  // One pass over ``a`` covers size_a, norm_a_sq, intersection, dot.
  // A separate pass over ``b`` picks up size_b and norm_b_sq; keys
  // present in both sets are visited twice (once from each side) but
  // contribute only to the corresponding side's totals.
  for (const auto& [key, count_a] : a) {
    s.size_a += count_a;
    const auto ca = static_cast<std::uint64_t>(count_a);
    s.norm_a_sq += ca * ca;
    auto it = b.find(key);
    if (it != b.end()) {
      const auto count_b = it->second;
      s.intersection += std::min(count_a, count_b);
      s.dot_product += ca * static_cast<std::uint64_t>(count_b);
    }
  }
  for (const auto& [_, count_b] : b) {
    s.size_b += count_b;
    const auto cb = static_cast<std::uint64_t>(count_b);
    s.norm_b_sq += cb * cb;
  }
  return s;
}

// Individual metric formulas. All four follow the same identity
// convention: two empty inputs → 1.0, one empty → 0.0.

inline double jaccard_from_stats(const ComparisonStats& s) {
  if (s.size_a == 0 && s.size_b == 0) return 1.0;
  const std::size_t union_size = s.size_a + s.size_b - s.intersection;
  if (union_size == 0) return 0.0;
  return static_cast<double>(s.intersection)
       / static_cast<double>(union_size);
}

inline double dice_from_stats(const ComparisonStats& s) {
  if (s.size_a == 0 && s.size_b == 0) return 1.0;
  const std::size_t total = s.size_a + s.size_b;
  if (total == 0) return 0.0;
  return 2.0 * static_cast<double>(s.intersection)
       / static_cast<double>(total);
}

inline double cosine_from_stats(const ComparisonStats& s) {
  if (s.size_a == 0 && s.size_b == 0) return 1.0;
  if (s.norm_a_sq == 0 || s.norm_b_sq == 0) return 0.0;
  const double denom = std::sqrt(static_cast<double>(s.norm_a_sq)
                                 * static_cast<double>(s.norm_b_sq));
  return static_cast<double>(s.dot_product) / denom;
}

inline double overlap_from_stats(const ComparisonStats& s) {
  if (s.size_a == 0 && s.size_b == 0) return 1.0;
  const std::size_t min_size = std::min(s.size_a, s.size_b);
  if (min_size == 0) return 0.0;
  return static_cast<double>(s.intersection)
       / static_cast<double>(min_size);
}

// Convenience scalar entry points. ``n`` defaults to 2 (character
// bigrams), the rapidfuzz / scikit-learn default for character-n-gram
// similarities.

inline double jaccard(const std::vector<Codepoint>& a,
                       const std::vector<Codepoint>& b,
                       std::size_t n = 2) {
  return jaccard_from_stats(compare(build_multiset(a, n),
                                     build_multiset(b, n)));
}

inline double dice(const std::vector<Codepoint>& a,
                    const std::vector<Codepoint>& b,
                    std::size_t n = 2) {
  return dice_from_stats(compare(build_multiset(a, n),
                                  build_multiset(b, n)));
}

inline double cosine(const std::vector<Codepoint>& a,
                      const std::vector<Codepoint>& b,
                      std::size_t n = 2) {
  return cosine_from_stats(compare(build_multiset(a, n),
                                    build_multiset(b, n)));
}

inline double overlap(const std::vector<Codepoint>& a,
                       const std::vector<Codepoint>& b,
                       std::size_t n = 2) {
  return overlap_from_stats(compare(build_multiset(a, n),
                                     build_multiset(b, n)));
}

}  // namespace stride_align::ngram
