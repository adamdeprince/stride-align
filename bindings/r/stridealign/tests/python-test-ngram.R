source("helpers.R")

families <- list(jaccard, dice, cosine, overlap)
expected <- c(0.25, 0.4, 3 / sqrt(32), 0.5)
for (index in seq_along(families)) {
  close_to(families[[index]]("ABCBDAB", "BDCAB"), expected[[index]])
}
close_to(jaccard("hello", "help", n = 3), 0.25)
for (function_ in families) {
  stopifnot(
    function_("aaaa", "aaaa") == 1,
    function_("hello", "hello") == 1,
    function_("", "") == 1,
    function_("abc", "") == 0,
    function_("", "abc") == 0,
    function_("abc", "xyz") == 0,
    function_("ab", "cd", n = 5) == 1,
    function_("ab", "abcdef", n = 5) == 0
  )
}
close_to(jaccard("aaaa", "aaa"), 2 / 3)
close_to(dice("aaaa", "aaa"), 0.8)
close_to(cosine("aaaa", "aaa"), 1)
close_to(overlap("aaaa", "aaa"), 1)

pairs <- list(
  c("ABCBDAB", "BDCAB"), c("hello", "world"),
  c("aaaa", "aaa"), c("kitten", "sitting")
)
for (function_ in families) for (pair in pairs) {
  close_to(function_(pair[[1L]], pair[[2L]]), function_(pair[[2L]], pair[[1L]]))
}
set.seed(31)
for (iteration in seq_len(60L)) {
  alphabet <- sample(c("abc", "abcdef"), 1L)
  left <- random_string(sample(0:29, 1L), strsplit(alphabet, "")[[1L]])
  right <- random_string(sample(0:29, 1L), strsplit(alphabet, "")[[1L]])
  for (function_ in families) {
    score <- function_(left, right)
    stopifnot(score >= 0, score <= 1)
  }
}
for (pair in pairs) {
  j <- jaccard(pair[[1L]], pair[[2L]])
  close_to(dice(pair[[1L]], pair[[2L]]), 2 * j / (1 + j))
}
for (function_ in families) expect_error(function_("hello", "world", n = 0), "n must be positive")

ascii_left <- "hello"
ascii_right <- "world"
Encoding(ascii_left) <- "bytes"
Encoding(ascii_right) <- "bytes"
for (function_ in families) {
  close_to(function_(ascii_left, ascii_right), function_("hello", "world"))
}
close_to(jaccard("Müller", "Mueller"), 0.375)

batch_families <- list(
  list(jaccard, jaccard_similarities), list(dice, dice_similarities),
  list(cosine, cosine_similarities), list(overlap, overlap_similarities)
)
targets <- c("sitting", "kitten", "kit", "biting")
for (family in batch_families) {
  batch <- family[[2L]]("kitten", targets)
  expected <- family[[1L]]("kitten", targets)
  close_to(batch, expected)
  stopifnot(is.double(batch), length(batch) == length(targets))
  # A scalar character value is already a one-element vector in R, so the
  # Python ambiguity check becomes a one-target batch assertion.
  stopifnot(length(family[[2L]]("query", "single")) == 1L)
}
close_to(jaccard_similarities("hello", c("hello", "help"), n = 3), c(1, 0.25))

# Python's generator acceptance is represented by R's ordinary lazy promise
# feeding a materialized character vector into the same one-to-many contract.
generated <- local({ values <- c("sitting", "kit"); values })
for (family in batch_families) stopifnot(length(family[[2L]]("kitten", generated)) == 2L)

# Python cases: test_canonical_abcbdab_bdcab, test_hello_vs_help_n3,
# test_aaaa_vs_aaaa_n2_multiset_semantics,
# test_aaaa_vs_aaa_n2_partial_multiset_overlap,
# test_identical_inputs_score_one, test_both_empty_score_one,
# test_one_empty_scores_zero, test_disjoint_alphabets_score_zero,
# test_shorter_than_n_treated_as_empty, test_symmetric,
# test_result_in_unit_interval, test_dice_jaccard_relation,
# test_n_zero_rejected, test_bytes_input_matches_str_on_ascii,
# test_non_ascii_codepoints_first_class, test_batch_matches_per_pair,
# test_batch_returns_float64_ndarray, test_batch_rejects_str_or_bytes_targets,
# test_batch_accepts_generator, test_batch_n_kwarg_propagates.
