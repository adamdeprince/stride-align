source("helpers.R")

stopifnot(
  jaro_similarity("", "") == 1,
  jaro_similarity("abc", "") == 0,
  jaro_similarity("", "abc") == 0,
  jaro_similarity("abc", "abc") == 1,
  jaro_similarity("abc", "xyz") == 0
)
known <- list(
  c("martha", "marhta"), c("dwayne", "duane"),
  c("dixon", "dicksonx"), c("jellyfish", "smellyfish"),
  c("ab", "ba"), c("abcd", "dcba")
)
for (pair in known) close_to(
  jaro_similarity(pair[[1L]], pair[[2L]]),
  reference_jaro(pair[[1L]], pair[[2L]])
)
for (seed in c(7L, 11L)) {
  set.seed(seed)
  for (iteration in seq_len(500L)) {
    left <- random_string(sample(0:50, 1L))
    right <- random_string(sample(0:50, 1L))
    close_to(jaro_similarity(left, right), reference_jaro(left, right))
    close_to(
      jaro_winkler_similarity(left, right),
      reference_jaro_winkler(left, right)
    )
  }
}
for (weight in c(0.05, 0.15, 0.2)) close_to(
  jaro_winkler_similarity("martha", "marhta", prefix_weight = weight),
  reference_jaro_winkler("martha", "marhta", prefix_weight = weight)
)
base <- jaro_similarity("ab", "ax")
stopifnot(
  jaro_winkler_similarity("ab", "ax", prefix_threshold = 0) > base,
  jaro_winkler_similarity("ab", "ax", prefix_threshold = 1) == base
)
for (pair in list(c("café", "cafe"), c("🎉🎈ab", "🎈🎉ab"))) {
  close_to(jaro_similarity(pair[[1L]], pair[[2L]]), reference_jaro(pair[[1L]], pair[[2L]]))
}
bytes_left <- "martha"
bytes_right <- "marhta"
Encoding(bytes_left) <- "bytes"
Encoding(bytes_right) <- "bytes"
close_to(jaro_similarity(bytes_left, bytes_right), reference_jaro("martha", "marhta"))

targets <- c("kitten", "sitting", "kit")
batch <- jaro_similarities("kitten", targets)
close_to(batch, vapply(targets, function(target) reference_jaro("kitten", target), numeric(1)))
stopifnot(is.double(batch), length(batch) == 3L)
winkler_batch <- jaro_winkler_similarities("kitten", targets)
close_to(winkler_batch, vapply(targets, function(target) reference_jaro_winkler("kitten", target), numeric(1)))

ranked <- jaro_top_k("kitten", c("sitting", "kitten", "kit"), k = 2)
stopifnot(identical(ranked$target[[1L]], "kitten"), nrow(ranked) == 2L)
ranked <- jaro_winkler_top_k("martha", c("martha", "marhta", "xyz"), k = 2, prefix_weight = 0.2)
stopifnot("martha" %in% ranked$target)
best <- jaro_best("martha", c("xyz", "marhta", "martha"))
stopifnot(best$target == "martha", best$score == 1, best$index == 3L)
jw_best <- jaro_winkler_best("martin", c("martia", "smartin", "marthax"))
expected <- max(jaro_winkler_similarity("martin", c("martia", "smartin", "marthax")))
close_to(jw_best$score, expected)
stopifnot(
  "kitten" %in% extract(
    "kitten", targets, scorer = Scorer$JARO, k = 2
  )$target,
  "martha" %in% extract(
    "martha", c("martha", "marhta", "xyz"),
    scorer = Scorer$JARO_WINKLER, k = 2
  )$target,
  Scorer$JARO == 6L,
  Scorer$JARO_WINKLER == 7L,
  Scorer$JARO != Scorer$JARO_WINKLER,
  nrow(jaro_top_k("abc", c("a", "b"), 0)) == 0L,
  nrow(jaro_top_k("abc", character(), 5)) == 0L
)
generated <- local({ values <- c("martha", "marhta", "xyz"); values })
stopifnot("martha" %in% jaro_winkler_top_k("martha", generated, k = 2)$target)

# Python cases: test_jaro_edge_cases, test_jaro_known_values_match_rapidfuzz,
# test_jaro_random_battery_matches_rapidfuzz,
# test_jaro_winkler_default_matches_rapidfuzz,
# test_jaro_winkler_custom_prefix_weight,
# test_jaro_winkler_threshold_disables_bonus,
# test_jaro_handles_bytes_and_unicode,
# test_jaro_similarities_returns_ndarray,
# test_jaro_winkler_similarities_returns_ndarray,
# test_jaro_top_k_returns_highest, test_jaro_winkler_top_k_with_kwargs,
# test_jaro_best_finds_top, test_jaro_winkler_best_picks_prefix_match,
# test_extract_jaro, test_extract_jaro_winkler,
# test_scorer_jaro_values_distinct, test_jaro_top_k_zero_returns_empty,
# test_jaro_top_k_empty_targets_returns_empty,
# test_jaro_winkler_top_k_handles_generator.
