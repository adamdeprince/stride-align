source("helpers.R")

path <- system.file(
  "testdata", "bounded-lazy-f-counterexamples.txt", package = "stridealign"
)
stopifnot(nzchar(path), file.exists(path))
lines <- readLines(path, warn = FALSE)
headers <- grep("^\\*\\*\\* DISAGREEMENT #[0-9]+ ", lines)
stopifnot(length(headers) == 5L)
expected <- c(3320, 4907, 2628, 3010, 4480)
historical <- c(3310, 4897, 2601, 2986, 4463)
for (index in seq_along(headers)) {
  tokens_to_string <- function(line) {
    tokens <- as.integer(strsplit(sub("^[^:]+:", "", line), "[[:space:]]+")[[1L]])
    tokens <- tokens[!is.na(tokens)]
    intToUtf8(64L + tokens)
  }
  query <- tokens_to_string(lines[[headers[[index]] + 1L]])
  target <- tokens_to_string(lines[[headers[[index]] + 2L]])
  score <- smith_waterman_score(
    query, target, match_score = 8, mismatch_score = -9, gap_score = -1
  )
  farrar <- smith_waterman_farrar_score(
    query, target, match_score = 8, mismatch_score = -9, gap_score = -1
  )
  stopifnot(
    score == expected[[index]], farrar == expected[[index]],
    score != historical[[index]]
  )
}

# Python case: test_bounded_lazy_f_returns_true_score.
