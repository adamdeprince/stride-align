source("helpers.R")

queries <- c("kitten", "sitting", "kit")
targets <- c("kitten", "kit", "sitting", "biting")
result <- cdist_top_k(queries, targets, scorer = Scorer$JARO, k = 5)
expected <- head(sort(as.vector(cdist(queries, targets, scorer = Scorer$JARO)), decreasing = TRUE), 5)
close_to(sort(result$score, decreasing = TRUE), expected)
stopifnot(is.data.frame(result), nrow(result) <= 5L)

set.seed(42)
queries <- vapply(1:40, function(index) random_string(sample(2:8, 1L)), character(1))
stopifnot(nrow(cdist_top_k(
  queries, queries, scorer = Scorer$JARO, k = 10,
  reject_duplicates = TRUE, cpu_count = 1
)) == 10L)
stopifnot(
  nrow(cdist_top_k("abc", "abc", scorer = Scorer$JARO, k = 0)) == 0L,
  nrow(cdist_top_k(c("a", "b"), c("a", "b"), scorer = Scorer$JARO, k = 100)) == 4L
)
values <- c("foo", "foo", "bar", "fox")
default <- cdist_top_k(values, values, scorer = Scorer$JARO, k = 8)
rejected <- cdist_top_k(values, values, scorer = Scorer$JARO, k = 8, reject_duplicates = TRUE)
stopifnot(any(default$score == 1), all(rejected$query != rejected$target))
stopifnot(nrow(cdist_top_k("abc", "abc", scorer = Scorer$JARO, k = 1, reject_duplicates = TRUE)) == 0L)
expect_error(cdist_top_k("a", "a", scorer = Scorer$LEVENSHTEIN, k = 1), "normalized")
stopifnot(nrow(cdist_top_k(
  "martha", c("marhta", "xyz"), scorer = jaro_similarities, k = 2
)) == 2L)

set.seed(11)
queries <- vapply(1:40, function(index) random_string(sample(2:8, 1L)), character(1))
targets <- vapply(1:40, function(index) random_string(sample(2:8, 1L)), character(1))
single <- cdist_top_k(queries, targets, scorer = Scorer$JARO, k = 25, cpu_count = 1)
multiple <- cdist_top_k(queries, targets, scorer = Scorer$JARO, k = 25, cpu_count = 2)
close_to(sort(single$score), sort(multiple$score))

progress_state <- new.env(parent = emptyenv())
progress_state$updates <- 0L
factory <- function(total) list(
  update = function(amount) progress_state$updates <- progress_state$updates + amount,
  close = function() invisible(NULL)
)
invisible(cdist_top_k(
  rep(strrep("a", 6), 15), rep(strrep("b", 6), 15),
  scorer = Scorer$JARO, k = 5, tqdm = factory, cpu_count = 2
))
stopifnot(progress_state$updates == 15L)

# R does not spawn workers while it owns STRSXP. CPU-count, GIL, and concurrent
# mutation cases map to deterministic single-thread and copy-on-write checks.
values <- c("aaa", "abb", "abc", "xyz")
mirrors <- cdist_top_k(values, values, scorer = Scorer$JARO, k = 20, reject_duplicates = TRUE)
for (index in seq_len(nrow(mirrors))) stopifnot(any(
  mirrors$query == mirrors$target[[index]] & mirrors$target == mirrors$query[[index]]
))
stopifnot(nrow(cdist_top_k(character(), character(), scorer = Scorer$JARO, k = 5)) == 0L)
original_queries <- c("kitten", "sitting")
original_targets <- c("kitten", "biting")
references <- cdist_top_k(original_queries, original_targets, scorer = Scorer$JARO, k = 10)
stopifnot(all(references$query %in% original_queries), all(references$target %in% original_targets))
queries_snapshot <- paste0("q", 1:30, "xxxx")
targets_snapshot <- paste0("t", 1:30, "xxxx")
before <- cdist_top_k(queries_snapshot, targets_snapshot, scorer = Scorer$JARO, k = 10)
queries_snapshot[] <- "zzz"
targets_snapshot[] <- "zzz"
stopifnot(!any(before$query == "zzz"), !any(before$target == "zzz"))

# Python cases: all 15 tests in tests/test_cdist_top_k.py.
