source("helpers.R")

build_chinese <- function(length, alphabet_size, seed) {
  set.seed(seed)
  alphabet <- intToUtf8(0x4e00 + 0:(alphabet_size - 1L), multiple = TRUE)
  paste0(sample(alphabet, length, replace = TRUE), collapse = "")
}

stopifnot(
  smith_waterman_score("hello", "world") == 2,
  smith_waterman_score("café", "café") == 8,
  smith_waterman_score("你好世界", "你好啊朋友") == 4
)
alphabet <- intToUtf8(0x4e00 + 0:299, multiple = TRUE)
wide <- paste0(alphabet, collapse = "")
stopifnot(smith_waterman_score(wide, wide) == 600)
query <- build_chinese(100, 300, 11)
target <- build_chinese(100, 300, 12)
stopifnot(smith_waterman_score(query, target) >= 0, smith_waterman_score(query, query) == 200)

# Python can retain surrogate code points in its internal UCS4 form; UTF-8 and
# R cannot. Use the same cardinality with 66,000 valid Unicode scalar values.
very_wide <- intToUtf8(c(1:0xd7ff, 0xe000 + 0:10704))
short <- intToUtf8(utf8ToInt(very_wide)[1:2])
stopifnot(
  nchar(very_wide, type = "chars") == 66000L,
  smith_waterman_score(very_wide, short) == 4,
  smith_waterman_score(short, very_wide) == 4
)

targets <- c(wide, paste0(rev(alphabet), collapse = ""), build_chinese(120, 300, 7), "abc")
batch <- smith_waterman_scores(wide, targets)
expected <- smith_waterman_score(wide, targets)
stopifnot(identical(batch, expected), batch[[1L]] == 600, batch[[4L]] == 0)
small_targets <- c("你好啊朋友", "世界你好", "abc")
stopifnot(identical(
  smith_waterman_scores("你好世界", small_targets),
  smith_waterman_score("你好世界", small_targets)
))
stopifnot(smith_waterman_score("abc", wide) == 0)
affine <- smith_waterman_score(
  wide, wide, gap_open_score = -3, gap_extend_score = -1
)
stopifnot(affine == smith_waterman_score(wide, wide), affine == 600)
affine_targets <- c(wide, paste0(rev(alphabet), collapse = ""), "abc")
stopifnot(identical(
  smith_waterman_scores(
    wide, affine_targets, gap_open_score = -3, gap_extend_score = -1
  ),
  smith_waterman_score(
    wide, affine_targets, gap_open_score = -3, gap_extend_score = -1
  )
))

# Python cases: test_ucs1_unchanged, test_ucs2_small_alphabet_unchanged,
# test_ucs2_large_alphabet_lifts_256_cap, test_ucs2_large_alphabet_random,
# test_ucs4_alphabet_above_uint16_cap_routes_to_uint32,
# test_batch_unicode_wide_matches_per_pair,
# test_batch_unicode_small_alphabet_matches_per_pair,
# test_mixed_ucs1_ucs2_with_large_target_promotes,
# test_affine_unicode_wide_matches_linear_on_identity,
# test_batch_affine_unicode_wide_matches_per_pair.
