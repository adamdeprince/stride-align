source("helpers.R")

corpus <- c("kitten", "sitting", "kittenish", "kit", "", "biting", "kitten")
raw <- levenshtein_scores("kitten", corpus)
ranked <- levenshtein_top_k("kitten", corpus, k = 3)
stopifnot(
  identical(sort(ranked$score), head(sort(raw), 3L)),
  all(raw[ranked$index] == ranked$score),
  all(corpus[ranked$index] == ranked$target)
)
raw <- levenshtein_normalized_scores("kitten", corpus)
ranked <- levenshtein_normalized_top_k("kitten", corpus, k = 3)
close_to(sort(ranked$score, decreasing = TRUE), head(sort(raw, decreasing = TRUE), 3L))
stopifnot(any(ranked$score == 1))
stopifnot(identical(
  sort(damerau_levenshtein_top_k("ab", c("ba", "xy", "ab", "az"), 3)$score),
  c(0, 1, 1)
))
targets <- c("mouse", "house", "horse", "hosue")
ranked <- damerau_levenshtein_normalized_top_k("house", targets, 2)
close_to(
  sort(ranked$score, decreasing = TRUE),
  head(sort(damerau_levenshtein_normalized_scores("house", targets), decreasing = TRUE), 2L)
)

bytes_query <- "abcdef"
bytes_targets <- c("abcdef", "abcxxx", "xxcdef", "abxdef")
Encoding(bytes_query) <- "bytes"
Encoding(bytes_targets) <- "bytes"
ranked <- hamming_top_k(bytes_query, bytes_targets, 2)
stopifnot(
  identical(sort(ranked$score), head(sort(hamming_scores(bytes_query, bytes_targets)), 2L)),
  ranked$score[[1L]] == 0
)
ranked <- hamming_normalized_top_k(bytes_query, bytes_targets, 3)
close_to(
  sort(ranked$score, decreasing = TRUE),
  head(sort(hamming_normalized_scores(bytes_query, bytes_targets), decreasing = TRUE), 3L)
)
expect_error(hamming_top_k("abc", "ab", 1), "equal-length")

query <- "ACGTACGTACGT"
targets <- c("ACGTACGTACGT", "AAAAAAAAAAAA", "ACGTACGTACGA", "ACGTAAAAACGT")
ranked <- smith_waterman_top_k(query, targets, 2)
stopifnot(identical(
  sort(ranked$score, decreasing = TRUE),
  head(sort(smith_waterman_scores(query, targets), decreasing = TRUE), 2L)
))

top_functions <- list(
  levenshtein_top_k, levenshtein_normalized_top_k,
  damerau_levenshtein_top_k, damerau_levenshtein_normalized_top_k,
  hamming_top_k, hamming_normalized_top_k
)
for (function_ in top_functions) stopifnot(
  nrow(function_("abc", c("abc", "abd", "abe"), 0)) == 0L,
  nrow(function_("abc", character(), 5)) == 0L
)
for (function_ in list(levenshtein_top_k, damerau_levenshtein_top_k, hamming_top_k)) {
  targets <- c("aaa", "aab", "abb", "bbb")
  ranked <- function_("aaa", targets, 99)
  stopifnot(nrow(ranked) == length(targets), setequal(ranked$index, seq_along(targets)))
}
targets <- c("sitting", "kitten", "kit", "biting")
ranked <- levenshtein_top_k("kitten", targets, 4)
stopifnot(all(targets[ranked$index] == ranked$target))
# A scalar character is already a one-element collection in R.
stopifnot(nrow(levenshtein_top_k("abc", "abcabc", 1)) == 1L)

stopifnot(
  identical(
    sort(extract("kitten", c("kitten", "sitting", "kit"), Scorer$LEVENSHTEIN, 2)$score),
    sort(levenshtein_top_k("kitten", c("kitten", "sitting", "kit"), 2)$score)
  ),
  any(extract(
    "kitten", c("kitten", "sitting", "kit"),
    Scorer$LEVENSHTEIN_NORMALIZED, 3
  )$score == 1),
  identical(sort(extract("ab", c("ba", "xy", "ab"), Scorer$DAMERAU_LEVENSHTEIN, 2)$score), c(0, 1)),
  identical(sort(extract("abcd", c("abcd", "abce", "xxxx"), Scorer$HAMMING, 2)$score), c(0, 1)),
  any(extract("abcd", c("abcd", "abce", "xxxx"), Scorer$HAMMING_NORMALIZED, 2)$score == 1),
  identical(unname(unlist(Scorer[1:6])), 0:5)
)

best <- levenshtein_best("kitten", c("sitting", "kitten", "kit"))
stopifnot(best$target == "kitten", best$score == 0, best$index == 2L)
stopifnot(is.null(levenshtein_best("abc", character())))
best <- levenshtein_normalized_best("house", c("abcde", "house"))
stopifnot(best$target == "house", best$score == 1, best$index == 2L)
best <- damerau_levenshtein_best("ab", c("xy", "ab", "ba"))
stopifnot(best$target == "ab", best$score == 0)
best <- damerau_levenshtein_normalized_best("house", c("mouse", "house"))
stopifnot(best$target == "house", best$score == 1)
best <- hamming_best("abc", c("abx", "abc", "xxx"))
stopifnot(best$target == "abc", best$score == 0, best$index == 2L)
best <- hamming_normalized_best("abc", c("abx", "abc"))
stopifnot(best$target == "abc", best$score == 1)
best <- smith_waterman_best(
  "ACGTACGTACGT", c("AAAAAAAAAAAA", "ACGTACGTACGT", "ACGTAAAA")
)
stopifnot(best$target == "ACGTACGTACGT", best$score == 24)
best <- extract_best("kitten", c("sitting", "kitten"), Scorer$LEVENSHTEIN)
stopifnot(best$target == "kitten", best$score == 0)
best <- extract_best("house", c("mouse", "house"), Scorer$LEVENSHTEIN_NORMALIZED)
stopifnot(best$target == "house", best$score == 1)
stopifnot(is.null(extract_best("abc", character(), Scorer$LEVENSHTEIN)))

queries <- c("hello", "world", "pyth")
targets <- c("hellp", "help", "world", "pythn", "apple", "wrld")
per_query <- cdist_top_k_per_query(
  queries, targets, scorer = Scorer$LEVENSHTEIN_NORMALIZED, k = 2
)
stopifnot(length(per_query) == length(queries), identical(names(per_query), queries))
one <- cdist_top_k_per_query(
  "hello", c("hellp", "help", "world", "wrld", "yo", "helloworld"),
  scorer = Scorer$LEVENSHTEIN_NORMALIZED, k = 3
)[[1L]]
stopifnot(identical(one$score, sort(one$score, decreasing = TRUE)))
close_to(
  one$score,
  head(sort(levenshtein_normalized_score(
    "hello", c("hellp", "help", "world", "wrld", "yo", "helloworld")
  ), decreasing = TRUE), 3L)
)

queries <- c("a", "ab", "abc", "fuzzy", "longerquery", strrep("x", 40))
targets <- c(
  "a", "ab", "abc", "abcd", "fuzzz", "fuzy", "fuzzyy", "longerquery",
  "shortquery", strrep("x", 40), strrep("x", 80),
  "wholly unrelated string", strrep("y", 25)
)
results <- cdist_top_k_per_query(queries, targets, Scorer$LEVENSHTEIN_NORMALIZED, 5)
for (index in seq_along(queries)) close_to(
  sort(results[[index]]$score, decreasing = TRUE),
  head(sort(levenshtein_normalized_score(queries[[index]], targets), decreasing = TRUE), 5L),
  1e-9
)
jaro_result <- cdist_top_k_per_query(
  "martha", c("marhta", "marta", "matra", "marble", "wholly different"),
  Scorer$JARO, 3
)[[1L]]
close_to(
  jaro_result$score,
  head(sort(jaro_similarity("martha", c("marhta", "marta", "matra", "marble", "wholly different")), decreasing = TRUE), 3L),
  1e-9
)

# R promises/vectors replace Python generators. Both query and target vectors
# are materialized once and preserve order.
generated_queries <- local(c("hello", "world"))
generated_targets <- local(c("hellp", "help", "world"))
generated <- cdist_top_k_per_query(
  generated_queries, generated_targets, Scorer$LEVENSHTEIN_NORMALIZED, 2
)
stopifnot(identical(names(generated), generated_queries), length(generated) == 2L)
stopifnot(nrow(cdist_top_k_per_query(
  "hello", "targets", Scorer$LEVENSHTEIN_NORMALIZED, 3
)[[1L]]) == 1L)
expect_error(
  cdist_top_k_per_query("hello", "world", Scorer$LEVENSHTEIN, 2),
  "normalized or similarity"
)
stopifnot(
  nrow(cdist_top_k_per_query("hello", c("world", "wrld"), Scorer$LEVENSHTEIN_NORMALIZED, 10)[[1L]]) == 2L,
  nrow(cdist_top_k_per_query("hello", character(), Scorer$LEVENSHTEIN_NORMALIZED, 5)[[1L]]) == 0L,
  length(cdist_top_k_per_query(character(), c("a", "b"), Scorer$LEVENSHTEIN_NORMALIZED, 5)) == 0L
)
hamming <- cdist_top_k_per_query(
  "hello", c("world", "hellp", "longerthing", "h", "wrld"),
  Scorer$HAMMING_NORMALIZED, 10
)[[1L]]
stopifnot(setequal(hamming$target, c("world", "hellp")), setequal(hamming$index, c(1L, 2L)))

set.seed(7)
targets <- vapply(1:500, function(index) random_string(sample(2:60, 1L)), character(1))
queries <- c("abc", "hello", "fuzzymatch", strrep("x", 30))
unpruned <- cdist_top_k_per_query(
  queries, targets, Scorer$LEVENSHTEIN_NORMALIZED, 5, pruning = FALSE
)
pruned <- cdist_top_k_per_query(
  queries, targets, Scorer$LEVENSHTEIN_NORMALIZED, 5, pruning = TRUE
)
for (index in seq_along(queries)) close_to(
  sort(unpruned[[index]]$score), sort(pruned[[index]]$score), 1e-9
)
zero <- cdist_top_k_per_query(
  c("hello", "world"), c("hellp", "help"),
  Scorer$LEVENSHTEIN_NORMALIZED, 0
)
stopifnot(identical(names(zero), c("hello", "world")), all(vapply(zero, nrow, integer(1)) == 0L))

set.seed(11)
targets <- vapply(1:400, function(index) random_string(sample(3:50, 1L)), character(1))
queries <- c("hello", "goodbye", "fuzzy", "pythonlibrary")
single <- cdist_top_k_per_query(queries, targets, Scorer$LEVENSHTEIN_NORMALIZED, 5, cpu_count = 1)
threaded <- cdist_top_k_per_query(queries, targets, Scorer$LEVENSHTEIN_NORMALIZED, 5, cpu_count = 2)
for (index in seq_along(queries)) close_to(single[[index]]$score, threaded[[index]]$score)
ordered <- cdist_top_k_per_query(
  c("alpha", "bravo", "charlie", "delta", "echo"),
  c("alpa", "bravs", "charli", "delt", "ech"),
  Scorer$LEVENSHTEIN_NORMALIZED, 1, cpu_count = 4
)
stopifnot(identical(names(ordered), c("alpha", "bravo", "charlie", "delta", "echo")))
wide <- cdist_top_k_per_query(
  "你好", c("你好", "你好世界", "世界你好"),
  Scorer$LEVENSHTEIN_NORMALIZED, 3, cpu_count = 4
)[[1L]]
stopifnot(wide$score[[1L]] == 1, all(wide$score[-1L] < 1))
automatic <- cdist_top_k_per_query(
  c("hello", "world"), c("hellp", "wrld", "help"),
  Scorer$LEVENSHTEIN_NORMALIZED, 2, cpu_count = 0
)
stopifnot(length(automatic) == 2L, identical(names(automatic), c("hello", "world")))

# Python cases: all 50 tests in tests/test_top_k.py. Zero-based Python indices
# are translated to R's one-based indices; generators become R vectors; the
# generator-of-pairs result becomes a named list of data frames.
