#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "stride_align/core.hpp"
#include "stride_align/batch.hpp"
#include "stride_align/encoded.hpp"
#include "stride_align/wratio.hpp"

namespace {

void append_utf8(std::string& output, std::uint32_t codepoint) {
  if (codepoint <= 0x7fU) {
    output.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7ffU) {
    output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
  } else if (codepoint <= 0xffffU) {
    output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
    output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
  } else {
    output.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
    output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
    output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
  }
}

bool close(double lhs, double rhs) {
  return std::abs(lhs - rhs) < 1e-12;
}

std::string encode(std::span<const std::uint32_t> codepoints) {
  std::string output;
  for (const auto codepoint : codepoints) append_utf8(output, codepoint);
  return output;
}

}  // namespace

int main() {
  namespace core = stride_align::core;
  namespace utf8 = stride_align::utf8;

  const std::string kitten = "kitten";
  const std::string sitting = "sitting";
  const auto ascii = utf8::prepare_pair(kitten, sitting);
  assert(ascii.borrowed_ascii);
  assert(!ascii.packed);
  assert(ascii.width == utf8::TokenWidth::u8);
  assert(std::get<std::span<const std::uint8_t>>(ascii.query).data() ==
         reinterpret_cast<const std::uint8_t*>(kitten.data()));
  assert(core::levenshtein_distance(ascii) == 3U);
  assert(close(core::levenshtein_similarity(ascii), 1.0 - 3.0 / 7.0));
  assert(core::indel_distance(ascii) == 5U);

  const auto transposed = utf8::prepare_pair("CA", "AC");
  assert(core::osa_distance(transposed) == 1U);
  assert(core::true_damerau_levenshtein_distance(transposed) == 1U);

  const auto hamming = utf8::prepare_pair("karolin", "kathrin");
  assert(core::hamming_distance(hamming) == 3U);

  const auto jaro = utf8::prepare_pair("MARTHA", "MARHTA");
  assert(close(core::jaro_similarity(jaro), 0.9444444444444445));
  assert(close(core::jaro_winkler_similarity(jaro), 0.9611111111111111));

  const auto cjk = utf8::prepare_pair("你好", "你们好");
  assert(!cjk.borrowed_ascii);
  assert(!cjk.packed);
  assert(cjk.width == utf8::TokenWidth::u16);
  assert(cjk.query_size() == 2U);
  assert(cjk.target_size() == 3U);
  assert(core::levenshtein_distance(cjk) == 1U);
  assert(core::needleman_wunsch_score(cjk) == 3);

  const auto emoji = utf8::prepare_pair("a😀", "a😃");
  assert(emoji.width == utf8::TokenWidth::u32);
  assert(core::levenshtein_distance(emoji) == 1U);
  assert(core::hamming_distance(emoji) == 1U);

  // Streaming host adapters own promoted codepoint buffers across callback
  // boundaries. PreparedPair can borrow those stable UCS-2/UCS-4 spans so a
  // streamed cdist row does not copy either string for every pair.
  const std::vector<std::uint16_t> borrowed_u16_query{0x4f60U, 0x597dU};
  const std::vector<std::uint16_t> borrowed_u16_target{
      0x4f60U, 0x4eecU, 0x597dU};
  utf8::PreparedPair borrowed_u16;
  borrowed_u16.width = utf8::TokenWidth::u16;
  borrowed_u16.borrowed_ascii = false;
  borrowed_u16.query = std::span<const std::uint16_t>(borrowed_u16_query);
  borrowed_u16.target = std::span<const std::uint16_t>(borrowed_u16_target);
  assert(core::levenshtein_distance(borrowed_u16) == 1U);

  const std::vector<std::uint32_t> borrowed_u32_query{U'a', 0x1f600U};
  const std::vector<std::uint32_t> borrowed_u32_target{U'a', 0x1f603U};
  utf8::PreparedPair borrowed_u32;
  borrowed_u32.width = utf8::TokenWidth::u32;
  borrowed_u32.borrowed_ascii = false;
  borrowed_u32.query = std::span<const std::uint32_t>(borrowed_u32_query);
  borrowed_u32.target = std::span<const std::uint32_t>(borrowed_u32_target);
  assert(core::levenshtein_distance(borrowed_u32) == 1U);
  assert(core::hamming_distance(borrowed_u32) == 1U);

  std::string long_query;
  std::string long_target;
  for (std::size_t index = 0; index < 80U; ++index) {
    long_query += (index % 2U == 0U) ? "你" : "好";
    long_target += (index % 3U == 0U) ? "你" : "好";
  }
  const auto packed = utf8::prepare_pair(long_query, long_target);
  assert(packed.packed);
  assert(packed.width == utf8::TokenWidth::u8);
  auto fixed_options = utf8::PreparationOptions{};
  fixed_options.non_ascii = utf8::NonAsciiPolicy::fixed_width;
  const auto fixed = utf8::prepare_pair(long_query, long_target, fixed_options);
  assert(!fixed.packed);
  assert(fixed.width == utf8::TokenWidth::u16);
  assert(core::levenshtein_distance(packed) == core::levenshtein_distance(fixed));
  assert(core::smith_waterman_score(packed) == core::smith_waterman_score(fixed));

  std::string many_symbols;
  for (std::uint32_t codepoint = 0x100U; codepoint < 0x100U + 300U; ++codepoint) {
    append_utf8(many_symbols, codepoint);
  }
  const auto packed_u16 = utf8::prepare_pair(many_symbols, many_symbols);
  assert(packed_u16.packed);
  assert(packed_u16.width == utf8::TokenWidth::u16);
  assert(core::levenshtein_distance(packed_u16) == 0U);

  std::string u32_alphabet;
  std::uint32_t next_codepoint = 0x100U;
  for (std::size_t index = 0; index < 66000U; ++index) {
    if (next_codepoint == 0xd800U) next_codepoint = 0xe000U;
    append_utf8(u32_alphabet, next_codepoint++);
  }
  const auto packed_u32 = utf8::prepare_pair(u32_alphabet, u32_alphabet);
  assert(packed_u32.packed);
  assert(packed_u32.width == utf8::TokenWidth::u32);
  assert(packed_u32.query_size() == 66000U);

  const auto stream_ascii = utf8::prepare_streaming("abc");
  const auto stream_emoji = utf8::prepare_streaming("😀");
  assert(stream_ascii.size() == 3U && stream_ascii[0] == U'a');
  assert(stream_emoji.size() == 1U && stream_emoji[0] == 0x1f600U);

  auto stream_options = utf8::PreparationOptions{};
  stream_options.mode = utf8::PreparationMode::streaming;
  const auto stream_pair = utf8::prepare_pair("abc", "def", stream_options);
  assert(stream_pair.width == utf8::TokenWidth::u32);
  assert(!stream_pair.borrowed_ascii && !stream_pair.packed);

  std::string ascii_boundary(257U, 'a');
  assert(utf8::is_ascii(ascii_boundary));
  ascii_boundary[128] = static_cast<char>(0x80U);
  assert(!utf8::is_ascii(ascii_boundary));

  namespace encoded = stride_align::encoded;
  const auto mock_multibyte_width = [](std::string_view remaining) {
    return static_cast<unsigned char>(remaining.front()) < 0x80U ? 1U : 2U;
  };
  const encoded::EncodingProfile single_byte{1U, 1U};
  const std::string latin_left("caf\xe9", 4U);
  const std::string latin_right("cafe", 4U);
  const auto latin_pair = encoded::prepare_pair(
      latin_left, latin_right, single_byte, mock_multibyte_width);
  assert(latin_pair.width == utf8::TokenWidth::u8);
  assert(latin_pair.borrowed_ascii && !latin_pair.packed);

  const auto fixed_u16 = encoded::prepare_pair(
      "ABCD", "ABEF", encoded::EncodingProfile{2U, 2U},
      mock_multibyte_width);
  assert(fixed_u16.width == utf8::TokenWidth::u16);
  assert(!fixed_u16.borrowed_ascii && !fixed_u16.packed);
  const auto fixed_u32 = encoded::prepare_pair(
      "ABCDEFGH", "ABCDIJKL", encoded::EncodingProfile{4U, 4U},
      mock_multibyte_width);
  assert(fixed_u32.width == utf8::TokenWidth::u32);
  assert(!fixed_u32.borrowed_ascii && !fixed_u32.packed);

  const encoded::EncodingProfile variable_width{0U, 2U};
  const std::string euc_left("\xc6\xfc\xcb\xdc", 4U);
  const std::string euc_right("\xc6\xfc", 2U);
  const auto short_native = encoded::prepare_pair(
      euc_left, euc_right, variable_width, mock_multibyte_width);
  assert(short_native.width == utf8::TokenWidth::u16);
  assert(!short_native.borrowed_ascii && !short_native.packed);
  std::string long_native_left;
  std::string long_native_right;
  for (std::size_t index = 0; index < 40U; ++index) {
    long_native_left += euc_left.substr(0U, 2U);
    long_native_right += euc_left.substr(2U, 2U);
  }
  const auto packed_native = encoded::prepare_pair(
      long_native_left, long_native_right,
      variable_width, mock_multibyte_width);
  assert(packed_native.width == utf8::TokenWidth::u8);
  assert(!packed_native.borrowed_ascii && packed_native.packed);

  bool rejected = false;
  try {
    (void)utf8::prepare_pair(std::string_view("\xf0\x28\x8c\x28", 4U), "x");
  } catch (const utf8::InvalidUtf8& error) {
    rejected = error.offset() == 1U;
  }
  assert(rejected);

  const auto affine = utf8::prepare_pair("ACCGT", "ACG");
  assert(core::smith_waterman_affine_score(affine) == 4);
  assert(core::needleman_wunsch_affine_score(affine) == 2);

  // Force the same randomized Unicode pairs through packed u8/u16 tokens and
  // fixed-width UCS-2/UCS-4 tokens. Every equality-based scorer must agree.
  std::mt19937 generator(0x5a17U);
  constexpr std::array<std::uint32_t, 8> alphabet{
      U'a', U'b', U'c', 0x00e9U, 0x4f60U, 0x597dU, 0x1f600U, 0x1f603U};
  auto packed_options = utf8::PreparationOptions{};
  packed_options.pack_threshold = 0U;
  for (std::size_t iteration = 0; iteration < 250U; ++iteration) {
    const std::size_t query_length = generator() % 18U;
    const std::size_t target_length = generator() % 18U;
    std::vector<std::uint32_t> query_points(query_length);
    std::vector<std::uint32_t> target_points(target_length);
    for (auto& value : query_points) value = alphabet[generator() % alphabet.size()];
    for (auto& value : target_points) value = alphabet[generator() % alphabet.size()];
    if (!query_points.empty()) {
      query_points[0] = 0x1f600U;
    } else if (!target_points.empty()) {
      target_points[0] = 0x1f600U;
    }

    const std::string query_text = encode(query_points);
    const std::string target_text = encode(target_points);
    const auto dense = utf8::prepare_pair(query_text, target_text, packed_options);
    const auto raw = utf8::prepare_pair(query_text, target_text, fixed_options);

    assert(core::levenshtein_distance(dense) == core::levenshtein_distance(raw));
    assert(core::osa_distance(dense) == core::osa_distance(raw));
    assert(core::true_damerau_levenshtein_distance(dense) ==
           core::true_damerau_levenshtein_distance(raw));
    assert(core::indel_distance(dense) == core::indel_distance(raw));
    assert(close(core::jaro_similarity(dense), core::jaro_similarity(raw)));
    assert(close(
        core::jaro_winkler_similarity(dense),
        core::jaro_winkler_similarity(raw)));
    assert(core::smith_waterman_score(dense) == core::smith_waterman_score(raw));
    assert(core::needleman_wunsch_score(dense) ==
           core::needleman_wunsch_score(raw));
    assert(core::smith_waterman_affine_score(dense) ==
           core::smith_waterman_affine_score(raw));
    assert(core::needleman_wunsch_affine_score(dense) ==
           core::needleman_wunsch_affine_score(raw));
    if (query_length == target_length) {
      assert(core::hamming_distance(dense) == core::hamming_distance(raw));
    }
  }

  const auto check_wavefront = [&generator, &alphabet](
      std::size_t query_length,
      std::size_t target_length,
      stride_align::Score match_score,
      stride_align::Score mismatch_score,
      stride_align::Score gap_score,
      stride_align::Score gap_open_score,
      stride_align::Score gap_extend_score) {
    std::vector<std::uint32_t> query(query_length);
    std::vector<std::uint32_t> target(target_length);
    for (auto& value : query) value = alphabet[generator() % alphabet.size()];
    for (auto& value : target) value = alphabet[generator() % alphabet.size()];
    const auto query_span = std::span<const std::uint32_t>(query);
    const auto target_span = std::span<const std::uint32_t>(target);
    namespace pairwise = stride_align::pairwise_alignment;

    assert((pairwise::linear_score<std::uint32_t, true>(
                query_span, target_span,
                match_score, mismatch_score, gap_score) ==
            pairwise::linear_score_scalar<std::uint32_t, true>(
                query_span, target_span,
                match_score, mismatch_score, gap_score)));
    assert((pairwise::linear_score<std::uint32_t, false>(
                query_span, target_span,
                match_score, mismatch_score, gap_score) ==
            pairwise::linear_score_scalar<std::uint32_t, false>(
                query_span, target_span,
                match_score, mismatch_score, gap_score)));
    assert((pairwise::affine_score<std::uint32_t, true>(
                query_span, target_span,
                match_score, mismatch_score, gap_open_score, gap_extend_score) ==
            pairwise::affine_score_scalar<std::uint32_t, true>(
                query_span, target_span,
                match_score, mismatch_score, gap_open_score, gap_extend_score)));
    assert((pairwise::affine_score<std::uint32_t, false>(
                query_span, target_span,
                match_score, mismatch_score, gap_open_score, gap_extend_score) ==
            pairwise::affine_score_scalar<std::uint32_t, false>(
                query_span, target_span,
                match_score, mismatch_score, gap_open_score, gap_extend_score)));
  };

  for (std::size_t iteration = 0; iteration < 40U; ++iteration) {
    // These cases force the SIMD wavefront through 8-, 16-, 32-, and 64-bit
    // score cells while the rolling-row implementation remains the oracle.
    check_wavefront(32U, 32U, 2, -1, -1, -2, -1);
    check_wavefront(83U, 71U, 3, -4, -2, -5, -1);
    check_wavefront(48U, 50U, 1000, -900, -700, -1100, -300);
    check_wavefront(
        33U, 34U, 100000000, -90000000, -70000000,
        -110000000, -30000000);
  }

  // Batch selection is part of the host-neutral core: language adapters must
  // not rebuild cdist, filtering, or ranking out of scalar host calls.
  namespace batch = stride_align::batch;
  const batch::Text batch_query("kitten");
  const std::vector<std::optional<batch::Text>> batch_targets = {
      batch::Text("sitting"), batch::Text("kitten"), batch::Text("bitten"),
      batch::Text("a much longer string")};
  const auto ranked = batch::top_k(
      batch_query, batch_targets, batch::Scorer::levenshtein, 2U);
  assert(ranked.size() == 2U);
  assert(ranked[0].index == 1U && ranked[0].score == 0.0);
  assert(ranked[1].index == 2U && ranked[1].score == 1.0);
  assert(batch::top_k(
      batch_query, batch_targets, batch::Scorer::levenshtein, 0U).empty());
  assert(close(
      batch::maximum_similarity(batch::Scorer::jaro, 3U, 30U),
      0.7));
  assert(batch::maximum_similarity(
      batch::Scorer::jaro_winkler, 3U, 30U) >= 0.7);

  const std::vector<std::optional<batch::Text>> batch_queries = {
      batch::Text("kitten"), batch::Text("bitten")};
  const auto all_pairs = batch::cdist(
      batch_queries, batch_targets, batch::Scorer::levenshtein_normalized);
  assert(all_pairs.size() == 2U && all_pairs[0].size() == 4U);
  const auto filtered = batch::cdist_above_threshold(
      batch_queries, batch_targets,
      batch::Scorer::levenshtein_normalized, 0.8);
  for (const auto& match : filtered) {
    assert(match.score >= 0.8);
    assert(close(match.score, *all_pairs[match.query_index][match.target_index]));
  }
  const auto global = batch::cdist_top_k(
      batch_queries, batch_targets,
      batch::Scorer::levenshtein_normalized, 3U, false);
  assert(global.size() == 3U);
  assert(global[0].score >= global[1].score &&
         global[1].score >= global[2].score);
  const auto per_query = batch::cdist_top_k_per_query(
      batch_queries, batch_targets,
      batch::Scorer::levenshtein_normalized, 2U);
  assert(per_query.size() == 2U);
  assert(per_query[0].size() == 2U && per_query[1].size() == 2U);

  const std::vector<std::uint32_t> wratio_left{
      U'n', U'e', U'w', U' ', U'y', U'o', U'r', U'k', U' ', U'm', U'e', U't', U's'};
  const std::vector<std::uint32_t> wratio_right{
      U'n', U'e', U'w', U' ', U'y', U'o', U'r', U'k', U' ', U'y', U'a', U'n', U'k', U'e', U'e', U's'};
  assert(close(
      stride_align::wratio::native_wratio(wratio_left, wratio_right),
      0.9025));

  return 0;
}
