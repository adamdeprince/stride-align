source("helpers.R")

stopifnot(
  monge_elkan("paul johnson", "paul johnson") == 1,
  monge_elkan("hello", "hello") == 1,
  monge_elkan("", "") == 1,
  monge_elkan("   ", "   ") == 1,
  monge_elkan("abc", "") == 0,
  monge_elkan("", "abc") == 0,
  monge_elkan("foo bar", "   ") == 0,
  monge_elkan("foo bar", "bar foo") == 1,
  monge_elkan("the quick brown fox", "fox brown quick the") == 1,
  monge_elkan("paul", "paul johnson") == 1,
  monge_elkan("paul johnson", "paul") == 0.5
)
forward <- monge_elkan("paul", "paul johnson")
backward <- monge_elkan("paul johnson", "paul")
close_to(monge_elkan("paul", "paul johnson", symmetric = TRUE), (forward + backward) / 2)
close_to(
  monge_elkan("paul", "paul johnson", symmetric = TRUE),
  monge_elkan("paul johnson", "paul", symmetric = TRUE)
)

expected_jaro <- (
  max(jaro_similarity("hello", c("hallo", "world"))) + 1
) / 2
close_to(monge_elkan("hello world", "hallo world"), expected_jaro)
jaro_value <- monge_elkan("hello world", "hallo world", inner = "jaro")
winkler_value <- monge_elkan("hello world", "hallo world", inner = "jaro_winkler")
stopifnot(winkler_value >= jaro_value)
close_to(winkler_value, (
  max(jaro_winkler_similarity("hello", c("hallo", "world"))) + 1
) / 2)
close_to(monge_elkan(
  "hello world", "hallo world", inner = "levenshtein_ratio"
), (max(levenshtein_normalized_score("hello", c("hallo", "world"))) + 1) / 2)
close_to(monge_elkan(
  "hello world", "hallo world", inner = "indel_ratio"
), (max(indel_normalized_score("hello", c("hallo", "world"))) + 1) / 2)
close_to(monge_elkan(
  "a b c", "a c d", inner = function(left, right) as.numeric(left == right)
), 2 / 3)
expect_error(monge_elkan("a", "b", inner = "not-a-real-metric"), "unknown inner similarity")
expect_error(monge_elkan("a", "b", inner = 42), "inner must be a function")
stopifnot(
  monge_elkan("PAUL", "paul", processor = tolower) == 1,
  monge_elkan("PAUL JOHNSON", "paul JOHNSON", processor = tolower) == 1
)
bytes <- "foo bar"
Encoding(bytes) <- "bytes"
stopifnot(monge_elkan(bytes, "bar foo") == 1)
expect_error(monge_elkan(42, "hello"), "query must be one character string")

set.seed(99)
alphabet <- strsplit("abcdef ghi jklmn", "")[[1L]]
for (iteration in seq_len(50L)) {
  left <- trimws(random_string(sample(0:29, 1L), alphabet))
  right <- trimws(random_string(sample(0:29, 1L), alphabet))
  for (inner in c("jaro", "jaro_winkler", "levenshtein_ratio", "indel_ratio")) {
    score <- monge_elkan(left, right, inner = inner)
    stopifnot(score >= 0, score <= 1)
  }
}
expected <- (1 + max(
  jaro_similarity("johnson", "paul"),
  jaro_similarity("johnson", "jones")
)) / 2
close_to(monge_elkan("paul johnson", "paul jones"), expected)
stopifnot(
  monge_elkan("a a a", "a") == 1,
  monge_elkan("a", "a a a") == 1
)

# Python cases: test_identity, test_both_empty_returns_one,
# test_one_empty_returns_zero, test_token_reordering_scores_one,
# test_asymmetric_when_token_counts_differ,
# test_symmetric_kwarg_averages_directions, test_inner_jaro_default,
# test_inner_jaro_winkler_distinct_from_jaro,
# test_inner_levenshtein_ratio, test_inner_indel_ratio,
# test_inner_callable, test_inner_unknown_string_raises,
# test_inner_wrong_type_raises, test_processor_preprocesses_inputs,
# test_bytes_input_widens_as_latin1, test_non_str_non_bytes_raises,
# test_result_in_unit_interval, test_paper_style_partial_match,
# test_paper_style_repeated_best_match.
