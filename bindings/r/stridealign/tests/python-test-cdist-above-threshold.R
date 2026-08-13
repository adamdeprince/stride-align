source("helpers.R")

set.seed(0)
queries <- unique(vapply(1:20, function(index) random_string(sample(2:8, 1L)), character(1)))[1:15]
targets <- unique(vapply(1:30, function(index) random_string(sample(2:8, 1L)), character(1)))[1:20]
threshold <- 0.6
dense <- cdist(queries, targets, scorer = Scorer$JARO)
expected <- which(dense >= threshold, arr.ind = TRUE)
actual <- cdist_above_threshold(
  queries, targets, scorer = Scorer$JARO, threshold = threshold, cpu_count = 2
)
stopifnot(
  is.data.frame(actual), nrow(actual) == nrow(expected),
  all(actual$score >= threshold)
)
for (index in seq_len(nrow(actual))) stopifnot(
  actual$score[[index]] == dense[actual$query_index[[index]], actual$target_index[[index]]]
)

# Python exposes a streaming iterator; the idiomatic R counterpart is a
# materialized data frame with stable one-based source indices.
all_pairs <- cdist_above_threshold("abc", c("abc", "abd"), scorer = Scorer$JARO, threshold = 0)
stopifnot(is.data.frame(all_pairs), nrow(all_pairs) == 2L)
values <- c("aaa", "bbb", "ccc")
pairs <- cdist_above_threshold(values, values, scorer = Scorer$JARO, threshold = 0)
stopifnot(
  any(pairs$query == "aaa" & pairs$target == "bbb"),
  any(pairs$query == "bbb" & pairs$target == "aaa")
)
expect_error(cdist_above_threshold("a", "a", scorer = Scorer$LEVENSHTEIN, threshold = 0.5), "normalized")
expect_error(cdist_above_threshold("a", "a", scorer = Scorer$JARO, threshold = 1.5), "threshold")
expect_error(cdist_above_threshold("a", "a", scorer = Scorer$JARO, threshold = -0.1), "threshold")
result <- cdist_above_threshold(
  "martha", c("marhta", "xyz"), scorer = jaro_similarities, threshold = 0.5
)
stopifnot(nrow(result) == 1L, result$query[[1L]] == "martha", result$target[[1L]] == "marhta")

progress_state <- new.env(parent = emptyenv())
progress_state$updates <- 0L
progress_state$closed <- FALSE
factory <- function(total) list(
  update = function(amount) progress_state$updates <- progress_state$updates + amount,
  close = function() progress_state$closed <- TRUE
)
invisible(cdist_above_threshold(
  rep("abcabc", 15), rep("abdabc", 20), scorer = Scorer$JARO,
  threshold = 0, tqdm = factory, cpu_count = 2
))
stopifnot(progress_state$updates == 15L, progress_state$closed)

# No background workers are created while R holds STRSXP; early-break,
# mutation, and GIL tests become repeatability/copy-on-write checks.
first <- cdist_above_threshold(rep("abc", 50), rep("abc", 50), scorer = Scorer$JARO, threshold = 0)
second <- cdist_above_threshold("foo", "bar", scorer = Scorer$JARO, threshold = 0)
stopifnot(nrow(first) == 2500L, nrow(second) == 1L)
queries <- rep("abcabcabc", 40)
targets <- rep("abdabdabd", 40)
snapshot <- cdist_above_threshold(queries, targets, scorer = Scorer$JARO, threshold = 0)
queries[] <- "zzz"
targets[] <- "zzz"
stopifnot(all(snapshot$query == "abcabcabc"), all(snapshot$target == "abdabdabd"))

set.seed(99)
queries <- vapply(1:30, function(index) random_string(sample(2:10, 1L)), character(1))
targets <- vapply(1:35, function(index) random_string(sample(2:10, 1L)), character(1))
stopifnot(identical(
  cdist_above_threshold(queries, targets, scorer = Scorer$JARO, threshold = 0.5, cpu_count = 1),
  cdist_above_threshold(queries, targets, scorer = Scorer$JARO, threshold = 0.5, cpu_count = 2)
))
expect_error(
  cdist_above_threshold("abc", "abcd", scorer = Scorer$HAMMING_NORMALIZED, threshold = 0),
  "equal-length"
)
stopifnot(nrow(cdist_above_threshold(character(), character(), scorer = Scorer$JARO, threshold = 0)) == 0L)
original_queries <- c("kitten", "sitting")
original_targets <- c("kitten", "biting")
references <- cdist_above_threshold(original_queries, original_targets, scorer = Scorer$JARO, threshold = 0)
stopifnot(all(references$query %in% original_queries), all(references$target %in% original_targets))

# Python cases: all 14 tests in tests/test_cdist_above_threshold.py.
