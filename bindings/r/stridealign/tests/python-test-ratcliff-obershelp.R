source("helpers.R")

pairs <- list(
  list("", "", 1), list("a", "", 0), list("", "b", 0),
  list("abc", "abc", 1), list("abc", "xyz", 0),
  list("kitten", "sitting", 0.6153846153846154),
  list("hello world", "world hello", 0.45454545454545453),
  list("ABCBDAB", "BDCAB", 1 / 3),
  list("Müller", "Mueller", 0.7692307692307693),
  list("difflib", "diff_lib", 0.9333333333333333),
  list("abcdef", "abcgef", 5 / 6),
  list("the quick brown fox", "the quick brown dog", 17 / 19),
  list("aaaabbbb", "bbbbaaaa", 0.5),
  list("aaaaaaaa", "aaaaaaaab", 16 / 17),
  list("prefix", "prefixmore", 0.75),
  list("morerunning", "running", 7 / 9)
)
for (pair in pairs) close_to(
  ratcliff_obershelp_similarity(pair[[1L]], pair[[2L]]), pair[[3L]]
)
stopifnot(
  ratcliff_obershelp_similarity("hello", "hello") == 1,
  ratcliff_obershelp_similarity("abc", "xyz") == 0,
  ratcliff_obershelp_similarity("xyz", "abc") == 0,
  ratcliff_obershelp_similarity("abc", "") == 0,
  ratcliff_obershelp_similarity("", "abc") == 0
)
forward <- ratcliff_obershelp_similarity("ABCBDAB", "BDCAB")
backward <- ratcliff_obershelp_similarity("BDCAB", "ABCBDAB")
stopifnot(forward != backward)
close_to(forward, 1 / 3)

bytes_left <- "kitten"
bytes_right <- "sitting"
Encoding(bytes_left) <- "bytes"
Encoding(bytes_right) <- "bytes"
close_to(
  ratcliff_obershelp_similarity(bytes_left, bytes_right),
  ratcliff_obershelp_similarity("kitten", "sitting")
)
set.seed(17)
for (iteration in seq_len(50L)) {
  left <- random_string(sample(0:29, 1L), letters[1:5])
  right <- random_string(sample(0:29, 1L), letters[1:5])
  score <- ratcliff_obershelp_similarity(left, right)
  stopifnot(score >= 0, score <= 1)
}
targets <- c("sitting", "kitten", "kit", "biting")
batch <- ratcliff_obershelp_similarities("kitten", targets)
close_to(batch, ratcliff_obershelp_similarity("kitten", targets))
stopifnot(is.double(batch), length(batch) == 4L)
stopifnot(length(ratcliff_obershelp_similarities("query", "single")) == 1L)
generated <- local({ values <- c("sitting", "kit"); values })
stopifnot(length(ratcliff_obershelp_similarities("kitten", generated)) == 2L)

# Python cases: test_matches_difflib_autojunk_false,
# test_identical_inputs_score_one, test_no_overlap_scores_zero,
# test_one_side_empty_scores_zero, test_not_strictly_symmetric,
# test_bytes_input_accepted, test_result_is_in_unit_interval,
# test_similarities_batch_matches_per_pair,
# test_similarities_returns_float64_ndarray,
# test_similarities_rejects_str_or_bytes_targets,
# test_similarities_accepts_generator.
