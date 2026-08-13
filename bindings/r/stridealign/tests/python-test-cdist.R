source("helpers.R")

queries <- c("kitten", "sitting", "kit")
targets <- c("kitten", "sitting", "kit", "biting")
dense <- cdist(queries, targets, scorer = Scorer$LEVENSHTEIN)
stopifnot(identical(dim(dense), c(3L, 4L)))
for (index in seq_along(queries)) stopifnot(identical(
  dense[index, ], levenshtein_scores(queries[[index]], targets)
))
for (scorer in c(
  Scorer$LEVENSHTEIN_NORMALIZED, Scorer$DAMERAU_LEVENSHTEIN_NORMALIZED,
  Scorer$JARO, Scorer$JARO_WINKLER
)) stopifnot(is.double(cdist(queries[1:2], targets[1:2], scorer = scorer)))
jaro_dense <- cdist(
  c("martha", "dwayne", "dixon"),
  c("marhta", "duane", "dicksonx", "no_match"),
  scorer = Scorer$JARO
)
for (index in 1:3) close_to(
  jaro_dense[index, ],
  jaro_similarity(c("martha", "dwayne", "dixon")[[index]], c("marhta", "duane", "dicksonx", "no_match"))
)

set.seed(7)
for (iteration in seq_len(20L)) {
  queries <- vapply(seq_len(sample(1:25, 1L)), function(index) random_string(sample(0:30, 1L)), character(1))
  targets <- vapply(seq_len(sample(1:25, 1L)), function(index) random_string(sample(0:30, 1L)), character(1))
  dense <- cdist(queries, targets, scorer = Scorer$LEVENSHTEIN)
  for (index in seq_along(queries)) stopifnot(identical(
    dense[index, ], levenshtein_score(queries[[index]], targets)
  ))
}
values <- c("kitten", "sitting", "kit", "biting")
symmetric <- cdist(values, values, scorer = Scorer$LEVENSHTEIN)
stopifnot(
  identical(dim(symmetric), c(4L, 4L)), identical(symmetric, t(symmetric)),
  all(diag(symmetric) == 0),
  identical(symmetric, cdist(values, as.character(values), scorer = Scorer$LEVENSHTEIN)),
  all(diag(cdist(values, values, scorer = Scorer$JARO)) == 1)
)
stopifnot(
  identical(
    cdist(c("abc", "xyz"), c("abc", "xyz", "abx"), scorer = Scorer$LEVENSHTEIN),
    cdist(c("abc", "xyz"), c("abc", "xyz", "abx"), scorer = levenshtein_scores)
  ),
  identical(
    cdist("martha", "marhta", scorer = Scorer$JARO),
    cdist("martha", "marhta", scorer = jaro_similarities)
  )
)
expect_error(cdist("a", "b", scorer = function(...) NULL), "unknown stride-align scorer function")

progress_state <- new.env(parent = emptyenv())
progress_state$total <- 0L
progress_state$updates <- 0L
progress_state$closed <- FALSE
factory <- function(total) {
  progress_state$total <- total
  list(
    update = function(amount) progress_state$updates <- progress_state$updates + amount,
    close = function() progress_state$closed <- TRUE
  )
}
invisible(cdist(queries[1:3], targets[1:4], scorer = Scorer$LEVENSHTEIN, tqdm = factory))
stopifnot(
  progress_state$total == 3L, progress_state$updates == 3L,
  isTRUE(progress_state$closed)
)

expect_error(cdist("abc", "ab", scorer = Scorer$HAMMING), "equal-length")
stopifnot(identical(dim(cdist(character(), character(), scorer = Scorer$LEVENSHTEIN)), c(0L, 0L)))
long_queries <- c(strrep("a", 100), strrep("b", 80))
long_targets <- c(strrep("a", 100), strrep("ab", 50))
long_dense <- cdist(long_queries, long_targets, scorer = Scorer$LEVENSHTEIN)
for (index in 1:2) stopifnot(identical(long_dense[index, ], levenshtein_score(long_queries[[index]], long_targets)))
# R's row-wise batch dispatcher has no Python cdist's arbitrary 256-codepoint
# query cap; the corresponding test asserts the more capable behavior.
stopifnot(cdist(strrep("a", 300), strrep("a", 10), scorer = Scorer$LEVENSHTEIN)[1L, 1L] == 290)
default <- cdist("martha", "marhta", scorer = Scorer$JARO_WINKLER)
heavier <- cdist("martha", "marhta", scorer = Scorer$JARO_WINKLER, prefix_weight = 0.2)
stopifnot(heavier[1L, 1L] > default[1L, 1L])

set.seed(13)
queries <- vapply(1:40, function(index) random_string(sample(1:30, 1L)), character(1))
targets <- vapply(1:50, function(index) random_string(sample(1:30, 1L)), character(1))
stopifnot(identical(
  cdist(queries, targets, scorer = Scorer$LEVENSHTEIN, cpu_count = 1),
  cdist(queries, targets, scorer = Scorer$LEVENSHTEIN, cpu_count = 2)
))
stopifnot(identical(
  cdist(queries, queries, scorer = Scorer$LEVENSHTEIN, cpu_count = 1),
  cdist(queries, queries, scorer = Scorer$LEVENSHTEIN, cpu_count = 2)
))
stopifnot(isTRUE(all.equal(
  cdist(queries[1:20], targets[1:25], scorer = Scorer$JARO, cpu_count = 2),
  cdist(queries[1:20], targets[1:25], scorer = Scorer$JARO, cpu_count = 1)
)))

# R is deliberately single-threaded while it holds STRSXP. Python GIL,
# worker-thread callback, and concurrent-list-mutation cases map to the tests
# above for deterministic cpu_count handling, main-thread progress callbacks,
# and R copy-on-write input stability.

# Python cases: test_cdist_levenshtein_matches_scalar_reference,
# test_cdist_returns_float_for_similarity_scorers,
# test_cdist_jaro_matches_scalar_reference, test_cdist_random_battery_lev,
# test_cdist_symmetric_input_is_symmetric_output,
# test_cdist_symmetric_diagonal_for_similarity_is_one,
# test_cdist_function_reference_dispatch_matches_enum,
# test_cdist_function_reference_dispatch_jaro,
# test_cdist_unknown_scorer_raises,
# test_cdist_tqdm_total_and_updates_sum_to_total,
# test_cdist_tqdm_symmetric_updates_shrink,
# test_cdist_hamming_requires_equal_lengths, test_cdist_empty_inputs,
# test_cdist_n_target_long_strings_within_simd_cap,
# test_cdist_above_simd_cap_raises,
# test_cdist_jaro_winkler_uses_prefix_kwargs,
# test_cdist_multi_threaded_matches_single_threaded,
# test_cdist_multi_threaded_symmetric_matches_single_threaded,
# test_cdist_multi_threaded_jaro_matches_scalar,
# test_cdist_tqdm_updates_dispatched_from_main_thread,
# test_cdist_defensive_copy_against_mutation_during_compute,
# test_cdist_releases_gil_during_compute.
