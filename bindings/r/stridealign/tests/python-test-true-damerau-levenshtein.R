source("helpers.R")

stopifnot(
  true_damerau_levenshtein_score("", "") == 0,
  true_damerau_levenshtein_score("abc", "") == 3,
  true_damerau_levenshtein_score("", "abc") == 3,
  true_damerau_levenshtein_score("abc", "abc") == 0,
  true_damerau_levenshtein_score("kitten", "sitting") == 3,
  damerau_levenshtein_score("ca", "abc") == 3,
  true_damerau_levenshtein_score("ca", "abc") == 2
)
close_to(true_damerau_levenshtein_normalized_score("foo", "foo"), 1)
close_to(true_damerau_levenshtein_normalized_score("", ""), 1)
close_to(true_damerau_levenshtein_normalized_score("abc", ""), 0)
close_to(true_damerau_levenshtein_normalized_score("ca", "abc"), 1 / 3)

for (seed in 0:4) {
  set.seed(seed + 500L)
  for (iteration in seq_len(200L)) {
    left <- random_string(sample(0:40, 1L))
    right <- random_string(sample(0:40, 1L))
    stopifnot(
      true_damerau_levenshtein_score(left, right) ==
        reference_true_damerau(left, right)
    )
  }
}
set.seed(0xBEEF)
left <- random_string(150L)
right <- random_string(130L)
stopifnot(
  true_damerau_levenshtein_score(left, right) ==
    reference_true_damerau(left, right)
)

query <- "ca"
targets <- c("abc", "ca", "ac", "x")
stopifnot(identical(
  true_damerau_levenshtein_scores(query, targets),
  true_damerau_levenshtein_score(query, targets)
))
close_to(
  true_damerau_levenshtein_normalized_scores(
    "abc", c("abc", "ca", "abcd", "")
  ),
  true_damerau_levenshtein_normalized_score(
    "abc", c("abc", "ca", "abcd", "")
  )
)
ranked <- true_damerau_levenshtein_top_k(
  query, c("abc", "ca", "ac", "xyz", "cab"), 3
)
stopifnot(identical(sort(ranked$score), c(0, 1, 1)))
normalized_ranked <- true_damerau_levenshtein_normalized_top_k(
  query, c("abc", "ca", "ac", "xyz", "cab"), 2
)
stopifnot(identical(normalized_ranked$target[[1L]], "ca"))
stopifnot(true_damerau_levenshtein_best(query, c("abc", "ca", "ac"))$score == 0)

dense <- cdist(
  c("ca", "kitten"), c("abc", "ca", "sitting"),
  scorer = Scorer$TRUE_DAMERAU_LEVENSHTEIN
)
stopifnot(dense[1L, 1L] == 2, dense[1L, 2L] == 0, dense[2L, 3L] == 3)
similarity <- cdist(
  c("ca", "kitten"), c("abc", "ca", "sitting"),
  scorer = Scorer$TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED
)
for (i in 1:2) for (j in 1:3) close_to(
  similarity[i, j],
  true_damerau_levenshtein_normalized_score(
    c("ca", "kitten")[[i]], c("abc", "ca", "sitting")[[j]]
  )
)
stopifnot(extract(
  "ca", c("abc", "ca", "ac"),
  scorer = Scorer$TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED, k = 2
)$target[[1L]] == "ca")
filtered <- cdist_above_threshold(
  "ca", c("abc", "ca", "ac", "xyz"),
  scorer = Scorer$TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED,
  threshold = 0.5
)
stopifnot(all(c("ca", "ac") %in% filtered$target))
stopifnot(
  length(true_damerau_levenshtein_scores("abc", character())) == 0L,
  true_damerau_levenshtein_score("café", "cafe") == 1
)

# Python cases: test_classic_examples,
# test_diverges_from_osa_on_overlapping_transposition,
# test_normalized_examples, test_matches_rapidfuzz_oracle,
# test_long_strings, test_scores_batch, test_normalized_scores_batch,
# test_top_k_returns_lowest_distances,
# test_normalized_top_k_returns_highest_similarities, test_best,
# test_cdist_matrix, test_cdist_normalized_matches_pairwise,
# test_extract_by_enum, test_cdist_above_threshold, test_empty_inputs,
# test_unicode_input.
