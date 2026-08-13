source("helpers.R")

scorers <- list(
  list(Scorer$SMITH_WATERMAN, smith_waterman_scores),
  list(Scorer$SMITH_WATERMAN_NORMALIZED, smith_waterman_normalized_scores),
  list(Scorer$NEEDLEMAN_WUNSCH, needleman_wunsch_scores),
  list(Scorer$NEEDLEMAN_WUNSCH_NORMALIZED, needleman_wunsch_normalized_scores)
)
raw_sw <- cdist(c("hello", "world"), c("hallo", "word", "help"), scorer = Scorer$SMITH_WATERMAN)
normalized_sw <- cdist(c("hello", "world"), c("hallo", "word", "help"), scorer = Scorer$SMITH_WATERMAN_NORMALIZED)
raw_nw <- cdist("abc", "xyz", scorer = Scorer$NEEDLEMAN_WUNSCH)
normalized_nw <- cdist(c("abc", "def"), c("abc", "abd", "xyz"), scorer = Scorer$NEEDLEMAN_WUNSCH_NORMALIZED)
stopifnot(
  identical(dim(raw_sw), c(2L, 3L)), is.double(raw_sw),
  is.double(normalized_sw), raw_nw[1L, 1L] < 0,
  all(normalized_nw >= 0 & normalized_nw <= 1)
)
queries <- c("hello", "world", "kitten")
targets <- c("hallo", "word", "sitting", "help")
for (entry in scorers) {
  dense <- cdist(queries, targets, scorer = entry[[1L]])
  for (index in seq_along(queries)) stopifnot(identical(
    dense[index, ], entry[[2L]](queries[[index]], targets)
  ))
}
values <- c("hello", "world", "你好")
stopifnot(all(diag(cdist(values, values, scorer = Scorer$SMITH_WATERMAN_NORMALIZED)) == 1))
values <- c("hello", "hallo", "world", "wirld")
dense <- cdist(values, values, scorer = Scorer$SMITH_WATERMAN_NORMALIZED)
stopifnot(isTRUE(all.equal(dense, t(dense))))

for (scorer in list(
  Scorer$SMITH_WATERMAN, smith_waterman_scores,
  smith_waterman_farrar_scores
)) stopifnot(cdist("hello", "hallo", scorer = scorer)[1L, 1L] == 7)
for (scorer in list(
  Scorer$SMITH_WATERMAN_NORMALIZED, smith_waterman_normalized_scores,
  smith_waterman_farrar_normalized_scores
)) stopifnot(is.double(cdist("hello", "hallo", scorer = scorer)))
stopifnot(identical(
  cdist("abc", "abd", scorer = Scorer$NEEDLEMAN_WUNSCH),
  cdist("abc", "abd", scorer = needleman_wunsch_scores)
))
custom <- cdist(
  "AGCT", "AGGT", scorer = Scorer$SMITH_WATERMAN,
  match_score = 5, mismatch_score = -3, gap_score = -1
)
stopifnot(custom[1L, 1L] == 13, cdist("AGCT", "AGGT", scorer = Scorer$SMITH_WATERMAN)[1L, 1L] == 5)
stopifnot(
  cdist(
    "AAAACCCCGGGG", "AAAACCCCGGGG", scorer = Scorer$SMITH_WATERMAN,
    gap_open_score = -5, gap_extend_score = -1
  )[1L, 1L] == 24,
  cdist("hello", "hallo", scorer = Scorer$SMITH_WATERMAN, width = 32)[1L, 1L] == 7
)
for (entry in scorers) stopifnot(identical(
  cdist(rep(queries, 5), rep(targets, 5), scorer = entry[[1L]], cpu_count = 1),
  cdist(rep(queries, 5), rep(targets, 5), scorer = entry[[1L]], cpu_count = 2)
))

stopifnot(
  identical(dim(cdist(character(), c("a", "b"), scorer = Scorer$SMITH_WATERMAN)), c(0L, 2L)),
  identical(dim(cdist("a", character(), scorer = Scorer$SMITH_WATERMAN)), c(1L, 0L)),
  identical(dim(cdist(character(), character(), scorer = Scorer$SMITH_WATERMAN_NORMALIZED)), c(0L, 0L))
)
bytes_queries <- c("hello", "world")
bytes_targets <- c("hallo", "word")
Encoding(bytes_queries) <- "bytes"
Encoding(bytes_targets) <- "bytes"
stopifnot(cdist(bytes_queries, bytes_targets, scorer = Scorer$SMITH_WATERMAN)[1L, 1L] == 7)
unicode_cases <- list(
  list(c("Müller"), c("Mueller")),
  list(c("你好", "世界"), c("你好啊", "再见")),
  list(c("hi 👋"), c("ho 👋", "🌸")),
  list(c("你好", "hi 👋"), c("你好啊", "hi 🌸"))
)
for (case in unicode_cases) {
  dense <- cdist(case[[1L]], case[[2L]], scorer = Scorer$SMITH_WATERMAN_NORMALIZED)
  stopifnot(identical(dim(dense), c(length(case[[1L]]), length(case[[2L]]))))
  for (index in seq_along(case[[1L]])) stopifnot(identical(
    dense[index, ], smith_waterman_normalized_scores(case[[1L]][[index]], case[[2L]])
  ))
}
stopifnot(all(diag(cdist(c("你好", "世界", "你好世界"), c("你好", "世界", "你好世界"), scorer = Scorer$SMITH_WATERMAN_NORMALIZED)) == 1))
stopifnot(
  Scorer$SMITH_WATERMAN == 12L,
  Scorer$SMITH_WATERMAN_NORMALIZED == 13L,
  Scorer$NEEDLEMAN_WUNSCH == 14L,
  Scorer$NEEDLEMAN_WUNSCH_NORMALIZED == 15L,
  length(Scorer) == 16L
)

# The Python tqdm and worker-thread tests are covered by python-test-cdist.R;
# R executes callbacks on its only interpreter thread.
# Python cases: all 23 methods in tests/test_cdist_sw_nw.py.
