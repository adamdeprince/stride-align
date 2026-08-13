source("helpers.R")

varied_lengths <- function(count, lower, upper, seed) {
  set.seed(seed)
  vapply(seq_len(count), function(index) {
    length <- lower + sample.int(upper - lower + 1L, 1L) - 1L
    random_string(length)
  }, character(1))
}
pruned_scorers <- c(
  Scorer$JARO, Scorer$JARO_WINKLER,
  Scorer$LEVENSHTEIN_NORMALIZED,
  Scorer$DAMERAU_LEVENSHTEIN_NORMALIZED,
  Scorer$INDEL_NORMALIZED
)
check_filtered <- function(queries, targets, scorer, threshold) {
  dense <- cdist(queries, targets, scorer = scorer)
  expected <- which(dense >= threshold, arr.ind = TRUE)
  actual <- cdist_above_threshold(
    queries, targets, scorer = scorer, threshold = threshold, cpu_count = 2
  )
  expected_keys <- if (nrow(expected)) paste(expected[, 1L], expected[, 2L]) else character()
  actual_keys <- paste(actual$query_index, actual$target_index)
  stopifnot(setequal(expected_keys, actual_keys))
  for (index in seq_len(nrow(actual))) close_to(
    actual$score[[index]], dense[actual$query_index[[index]], actual$target_index[[index]]], 1e-9
  )
}
check_top <- function(queries, targets, scorer, k, cpu_count = 2) {
  dense <- cdist(queries, targets, scorer = scorer)
  expected <- head(sort(as.vector(dense), decreasing = TRUE), min(k, length(dense)))
  actual <- cdist_top_k(
    queries, targets, scorer = scorer, k = k, cpu_count = cpu_count
  )
  close_to(sort(actual$score, decreasing = TRUE), expected, 1e-9)
}

for (scorer in pruned_scorers) for (threshold in c(0.3, 0.6, 0.8, 0.95)) {
  seed <- scorer * 1009L + as.integer(threshold * 100)
  check_filtered(
    varied_lengths(25, 2, 30, seed),
    varied_lengths(30, 2, 30, seed + 1L), scorer, threshold
  )
}
for (scorer in pruned_scorers) for (k in c(1L, 5L, 25L, 100L)) {
  seed <- scorer * 7919L + k
  check_top(
    varied_lengths(30, 2, 30, seed),
    varied_lengths(35, 2, 30, seed + 1L), scorer, k
  )
}
for (scorer in pruned_scorers) {
  check_filtered(
    "abc", c(strrep("abcdefghijklmnop", 4), "ab", strrep("xyz", 10), "abcde", "abc"),
    scorer, 0.7
  )
  seed <- scorer * 31L + 7L
  check_top(
    varied_lengths(20, 3, 20, seed), varied_lengths(25, 3, 20, seed + 1L),
    scorer, 10L, cpu_count = 1
  )
}

queries <- varied_lengths(60, 2, 30, 0xC0FFEE)
targets <- varied_lengths(70, 2, 30, 0xC0FFEF)
for (iteration in 1:5) check_top(queries, targets, Scorer$JARO_WINKLER, 15L)

for (scorer in c(Scorer$LEVENSHTEIN_NORMALIZED, Scorer$DAMERAU_LEVENSHTEIN_NORMALIZED)) {
  for (threshold in c(0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1)) {
    seed <- as.integer((scorer + 1L) * 10000L + threshold * 1000)
    check_filtered(
      varied_lengths(25, 2, 50, seed), varied_lengths(30, 2, 50, seed + 1L),
      scorer, threshold
    )
  }
  for (k in c(1L, 5L, 25L)) {
    seed <- scorer * 101L + k
    check_top(
      varied_lengths(30, 2, 50, seed), varied_lengths(35, 2, 50, seed + 1L),
      scorer, k
    )
  }
}

for (n in c(16L, 50L, 100L, 200L)) for (threshold in c(0, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 1)) {
  queries <- varied_lengths(20, n, n, n + as.integer(threshold * 100))
  targets <- varied_lengths(25, n, n, n + as.integer(threshold * 100) + 1L)
  check_filtered(queries, targets, Scorer$HAMMING_NORMALIZED, threshold)
}
for (n in c(10L, 20L, 100L)) for (k in c(1L, 5L, 20L)) {
  check_top(
    varied_lengths(25, n, n, n + k), varied_lengths(30, n, n, n + k + 1L),
    Scorer$HAMMING_NORMALIZED, k
  )
}
for (n in c(10L, 20L, 50L, 100L, 200L)) {
  queries <- c(strrep("a", n), strrep("b", n), paste0(strrep("a", n %/% 2), strrep("b", n - n %/% 2)))
  targets <- c(strrep("a", n), strrep("c", n), strrep("b", n))
  for (threshold in c(0.5, 0.6, 0.75, 0.8, 0.9)) check_filtered(
    queries, targets, Scorer$HAMMING_NORMALIZED, threshold
  )
}
queries <- vapply(c(5, 10, 20, 25, 30), function(n) strrep("a", n), character(1))
targets <- vapply(c(5, 10, 20, 25, 30), function(n) strrep("b", n), character(1))
for (scorer in c(Scorer$LEVENSHTEIN_NORMALIZED, Scorer$DAMERAU_LEVENSHTEIN_NORMALIZED)) {
  for (threshold in c(0.4, 0.5, 0.6, 0.75, 0.8)) check_filtered(queries, targets, scorer, threshold)
}
exact <- cdist_above_threshold(
  c("abc", "abcd", "xyz"), c("abc", "abcd", "abcde"),
  scorer = Scorer$LEVENSHTEIN_NORMALIZED, threshold = 1
)
stopifnot(
  nrow(exact) == 2L,
  setequal(paste(exact$query, exact$target), c("abc abc", "abcd abcd"))
)

# Python cases: all 12 tests in tests/test_cdist_length_pruning.py,
# including every original scorer/threshold/k parameter expansion.
