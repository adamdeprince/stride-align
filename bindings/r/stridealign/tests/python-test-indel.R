source("helpers.R")

stopifnot(
  indel_score("kitten", "sitting") == 5,
  indel_score("", "") == 0,
  indel_score("abc", "") == 3,
  indel_score("", "abc") == 3,
  indel_score("abc", "abc") == 0,
  indel_score("abc", "abd") == 2
)
close_to(indel_normalized_score("foo", "foo"), 1)
close_to(indel_normalized_score("aaa", "bbb"), 0)
close_to(indel_normalized_score("abc", "abd"), 2 / 3)

# The independent LCS identity is the oracle used for the random and long
# batteries; it covers both the SIMD-sized and scalar fallback lengths.
for (seed in 0:4) {
  set.seed(seed + 100L)
  for (iteration in seq_len(200L)) {
    left <- random_string(sample(0:60, 1L))
    right <- random_string(sample(0:60, 1L))
    expected <- nchar(left) + nchar(right) - 2 * reference_lcs_length(left, right)
    stopifnot(indel_score(left, right) == expected)
  }
}
set.seed(0xABCD)
left <- random_string(100L)
right <- random_string(80L)
stopifnot(
  indel_score(left, right) ==
    nchar(left) + nchar(right) - 2 * reference_lcs_length(left, right)
)

query <- "kitten"
targets <- c("sitting", "kit", "mitten", "")
expected <- vapply(targets, function(target) indel_score(query, target), numeric(1))
close_to(indel_scores(query, targets), expected)
normalized_targets <- c("abc", "abd", "xyz", "abcd")
close_to(
  indel_normalized_scores("abc", normalized_targets),
  indel_normalized_score("abc", normalized_targets)
)

ranked <- indel_top_k(
  "kitten", c("sitting", "kit", "mitten", "kitten", "foo"), k = 3
)
stopifnot(identical(sort(ranked$score), c(0, 2, 3)))
normalized_ranked <- indel_normalized_top_k(
  "kitten", c("sitting", "kit", "mitten", "kitten", "foo"), k = 2
)
stopifnot(max(normalized_ranked$score) == 1)
best <- indel_best("kitten", c("sitting", "kit", "mitten", "foo"))
stopifnot(best$target == "mitten", best$score == 2, best$index == 3L)
stopifnot(
  length(indel_scores("abc", character())) == 0L,
  nrow(indel_top_k("abc", character(), k = 5)) == 0L
)

dense <- cdist(c("ab", "cd"), c("ab", "cd", "ef"), scorer = Scorer$INDEL)
stopifnot(
  identical(dim(dense), c(2L, 3L)), dense[1L, 1L] == 0,
  dense[2L, 2L] == 0, dense[1L, 2L] == 4, dense[1L, 3L] == 4
)
normalized <- cdist(
  c("ab", "cd"), c("ab", "cd", "ef"),
  scorer = Scorer$INDEL_NORMALIZED
)
stopifnot(
  normalized[1L, 1L] == 1, normalized[2L, 2L] == 1,
  normalized[1L, 2L] == 0, normalized[1L, 3L] == 0
)
extracted <- extract(
  "kitten", c("sitting", "kit", "mitten"),
  scorer = Scorer$INDEL_NORMALIZED, k = 2
)
stopifnot(max(extracted$score) == indel_normalized_score("kitten", "mitten"))

stopifnot(
  indel_score("hello", "hello", score_cutoff = 5) == 0,
  indel_score("hello", "hallo", score_cutoff = 100) == 2,
  indel_score(strrep("a", 10), strrep("b", 10), score_cutoff = 3) > 3,
  indel_score("hello", "world", score_cutoff = NULL) ==
    indel_score("hello", "world"),
  indel_normalized_score("hello", "world", score_cutoff = 0.5) == 0
)
close_to(indel_normalized_score("hello", "hallo", score_cutoff = 0.5), 0.8)
bounded <- indel_score(strrep("a", 50), strrep("b", 50), score_cutoff = 5)
stopifnot(bounded > 5, bounded <= 100)

# Python cases: test_classic_examples, test_normalized_examples,
# test_matches_rapidfuzz_oracle, test_long_query_falls_back_to_scalar,
# test_scores_batch, test_normalized_scores_batch,
# test_top_k_returns_lowest_distances,
# test_normalized_top_k_returns_highest_similarities,
# test_best_returns_lowest_distance, test_empty_inputs, test_cdist_matrix,
# test_cdist_normalized_matrix, test_extract_by_enum, test_lcs_identity,
# TestIndelScoreCutoff.test_identity_returns_zero_with_cutoff,
# TestIndelScoreCutoff.test_cutoff_above_true_returns_true_distance,
# TestIndelScoreCutoff.test_cutoff_below_true_returns_cutoff_plus_one_at_minimum,
# TestIndelScoreCutoff.test_cutoff_none_matches_no_cutoff,
# TestIndelScoreCutoff.test_normalized_cutoff_returns_zero_below_threshold,
# TestIndelScoreCutoff.test_normalized_cutoff_above_threshold_returns_true_value,
# TestIndelScoreCutoff.test_cutoff_kernel_bails_on_long_disjoint_inputs.
