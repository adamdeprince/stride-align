source("helpers.R")

indel_ratio <- function(left, right) {
  if (!nzchar(left) && !nzchar(right)) return(1)
  2 * lcs_length(left, right) / (nchar(left) + nchar(right))
}

token_sort_cases <- list(
  list("fuzzy wuzzy", "wuzzy fuzzy", 1),
  list("hello world", "world hello", 1),
  list("the cat sat", "sat cat the", 1),
  list("foo bar", "foo bar baz", indel_ratio("bar foo", "bar baz foo")),
  list("kitten", "kitten", 1),
  list("abc def", "xyz uvw", indel_ratio("abc def", "uvw xyz")),
  list("", "", 1), list("   ", "   ", 1)
)
for (case in token_sort_cases) close_to(
  token_sort_ratio(case[[1L]], case[[2L]]), case[[3L]]
)
stopifnot(
  token_sort_ratio("abc", "") == 0,
  token_sort_ratio("", "abc") == 0,
  token_set_ratio("foo bar", "foo bar baz") == 1,
  token_set_ratio("apple", "an apple a day") == 1,
  token_set_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear") == 1,
  token_set_ratio("", "") == 0,
  token_set_ratio("   ", "   ") == 0,
  token_set_ratio("abc", "") == 0,
  token_set_ratio("", "abc def") == 0
)
close_to(token_set_ratio("abc def", "uvw xyz"), indel_ratio("abc def", "uvw xyz"))

stopifnot(
  partial_ratio("apple", "an apple a day") == 1,
  partial_ratio("foo bar", "foo bar baz") == 1,
  partial_ratio("kitten", "kitten") == 1,
  partial_ratio("", "") == 1,
  partial_ratio("abc", "") == 0,
  partial_ratio("", "abc") == 0
)
close_to(
  partial_ratio("java language", "python programming language"),
  18 / 22
)
close_to(partial_ratio("hello world", "world hello"), 0.625)
for (pair in list(
  c("apple", "an apple a day"),
  c("java language", "python programming language"),
  c("hello world", "world hello")
)) close_to(
  partial_ratio(pair[[1L]], pair[[2L]]),
  partial_ratio(pair[[2L]], pair[[1L]])
)
stopifnot(
  partial_token_sort_ratio("foo bar", "bar foo") == 1,
  partial_token_sort_ratio("hello world", "world hello") == 1,
  partial_token_set_ratio("foo bar", "foo bar baz") == 1,
  partial_token_set_ratio("apple", "an apple a day") == 1,
  partial_token_set_ratio("", "") == 0,
  partial_token_set_ratio("abc", "") == 0,
  WRatio("kitten", "kitten") == 1,
  WRatio("the quick brown fox", "the quick brown fox") == 1,
  WRatio("", "") == 1,
  WRatio("abc", "") == 0,
  WRatio("", "abc") == 0
)

set.seed(7)
alphabet <- strsplit("abc def ghi", "")[[1L]]
for (iteration in seq_len(40L)) {
  left <- trimws(random_string(sample(0:19, 1L), alphabet))
  right <- trimws(random_string(sample(0:19, 1L), alphabet))
  score <- WRatio(left, right)
  stopifnot(score >= 0, score <= 1)
}

parity_cases <- list(
  list("kitten", "sitting", c(0.6153846153846154, 0.6153846153846154, 2 / 3, 2 / 3, 2 / 3)),
  list("hello world", "world hello", c(1, 1, 0.625, 1, 1)),
  list("foo bar", "foo bar baz", c(7 / 9, 1, 1, 6 / 7, 1)),
  list("apple", "an apple a day", c(10 / 19, 1, 1, 1, 1)),
  list("python programming language", "java language", c(0.4, 16 / 21, 9 / 11, 16 / 21, 1)),
  list("fuzzy wuzzy", "wuzzy fuzzy bear", c(22 / 27, 1, 9 / 11, 1, 1)),
  list("a quick brown fox", "fox", c(0.3, 1, 1, 1, 1)),
  list("a quick brown fox", "quick", c(5 / 11, 1, 1, 1, 1)),
  list("the cat in the hat", "cat hat", c(0.56, 1, 8 / 11, 1, 1))
)
functions <- list(
  token_sort_ratio, token_set_ratio, partial_ratio,
  partial_token_sort_ratio, partial_token_set_ratio
)
for (case in parity_cases) for (index in seq_along(functions)) {
  close_to(functions[[index]](case[[1L]], case[[2L]]), case[[3L]][[index]], 1e-9)
}

# These deliberately pin the documented conservative deviation from the
# RapidFuzz oracle without importing its compatibility package.
for (case in list(
  list("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear", 0.9302325581395349),
  list("the quick brown fox", "the quick brown dog", 0.9189189189189189)
)) {
  ours <- partial_ratio(case[[1L]], case[[2L]])
  stopifnot(ours <= case[[3L]] + 1e-9, case[[3L]] - ours < 0.05)
}

bytes_left <- "hello world"
bytes_right <- "world hello"
Encoding(bytes_left) <- "bytes"
Encoding(bytes_right) <- "bytes"
stopifnot(token_sort_ratio(bytes_left, bytes_right) == 1)
stopifnot(
  token_sort_ratio("FOO BAR", "bar foo", processor = tolower) == 1,
  token_set_ratio("Foo Bar", "BAR FOO", processor = tolower) == 1
)
expect_error(token_sort_ratio(42, "hello"), "character vectors")
expect_error(partial_ratio("hello", NULL), "character vectors")

# Python cases: test_token_sort_ratio, test_token_sort_ratio_one_empty,
# test_token_set_ratio_full_overlap_returns_one, test_token_set_ratio_pinned_value,
# test_token_set_ratio_disjoint_alphabets, test_token_set_ratio_both_empty_zero,
# test_token_set_ratio_one_empty_zero, test_partial_ratio_substring_inside_longer,
# test_partial_ratio_short_buried_in_long_via_lcs_window,
# test_partial_ratio_hello_world_world_hello, test_partial_ratio_identity,
# test_partial_ratio_empty, test_partial_ratio_symmetric,
# test_partial_token_sort_ratio_reorderings_score_one,
# test_partial_token_set_ratio_subset_returns_one,
# test_partial_token_set_ratio_empty_zero, test_wratio_identity,
# test_wratio_empty, test_wratio_unit_interval,
# test_token_sort_ratio_matches_rapidfuzz, test_token_set_ratio_matches_rapidfuzz,
# test_partial_ratio_matches_rapidfuzz,
# test_partial_token_sort_ratio_matches_rapidfuzz,
# test_partial_token_set_ratio_matches_rapidfuzz,
# test_partial_ratio_conservative_vs_rapidfuzz,
# test_token_sort_ratio_accepts_bytes, test_partial_ratio_accepts_bytes,
# test_processor_preprocesses_inputs, test_non_str_non_bytes_raises_type_error.
