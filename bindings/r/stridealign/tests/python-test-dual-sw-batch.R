source("helpers.R")

english_text <- function(length, seed) {
  corpus <- paste0(
    "The quick brown fox watches the city wake under a low grey sky. ",
    "People cross the station concourse with coffee, folded papers, and quiet plans. ",
    "A street musician repeats a careful phrase while buses hiss at the curb. ",
    "In the office, someone rewrites a paragraph until the tone is direct and useful. ",
    "Human text is uneven: spaces cluster, punctuation interrupts, and words return later. "
  )
  points <- rep(strsplit(corpus, "", fixed = TRUE)[[1L]], length.out = length)
  set.seed(seed)
  positions <- seq(37L, length, by = 41L)
  if (length(positions)) points[positions] <- sample(c(letters, " "), length(positions), replace = TRUE)
  paste0(points, collapse = "")
}

batch_vs_scalar <- function(
  query,
  targets,
  match_score = 2,
  mismatch_score = -1,
  gap_score = -1,
  width = 16
) {
  batch <- smith_waterman_scores(
    query, targets, match_score, mismatch_score, gap_score, width = width
  )
  scalar <- smith_waterman_score(
    query, targets, match_score, mismatch_score, gap_score, width = width
  )
  stopifnot(identical(batch, scalar))
}

for (case in list(c(1, 1024, 1), c(2, 1024, 2), c(3, 1024, 3), c(7, 1024, 4), c(8, 1024, 5))) {
  query <- english_text(1024, case[[3L]])
  targets <- vapply(seq_len(case[[1L]]), function(index) {
    english_text(case[[2L]], case[[3L]] + index)
  }, character(1))
  batch_vs_scalar(query, targets)
}
query <- english_text(1024, 42)
batch_vs_scalar(query, vapply(c(500, 800, 1024, 1200, 900, 700), function(n) english_text(n, n), character(1)))
batch_vs_scalar(query, c(english_text(1024, 8), "", english_text(512, 9), "", english_text(1024, 10)))
for (seed in 20:29) {
  query <- english_text(1024, seed)
  batch_vs_scalar(query, vapply(1:4, function(index) english_text(1024, seed + index), character(1)))
}
batch_vs_scalar(
  english_text(100, 99),
  vapply(c(100, 80, 120, 100), function(n) english_text(n, n + 99), character(1))
)
query <- english_text(1024, 11)
targets <- c(english_text(1024, 12), english_text(900, 13), english_text(1024, 14))
batch_vs_scalar(query, targets, match_score = 8, mismatch_score = -9)

# Python cases: test_dual_equal_length_batch_matches_seq,
# test_dual_unequal_lengths_batch_matches_seq,
# test_dual_empty_and_mixed_targets, test_dual_seed_sweep_equal_pairs,
# test_batch_short_query_still_matches_seq, test_batch_alt_scores_match_seq.
