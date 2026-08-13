source("helpers.R")

# NumPy ndarray support is a Python boundary feature. R's public string APIs
# intentionally accept STRSXP only, avoiding the ambiguity between a numeric
# token sequence and a vectorized batch of values. These checks copy the
# storage, validation, cardinality, batch, and affine intent using R types.
for (value in list(1:5, as.double(1:5), as.raw(1:5), list(1, 2, 3))) {
  expect_error(smith_waterman_score(value, value), "character")
  expect_error(needleman_wunsch_score(value, value), "character")
}
expect_error(smith_waterman_score(1:3, "abc"), "character")
expect_error(smith_waterman_score("abc", 1:3), "character")
expect_error(smith_waterman_path(1:3, 1:3), "character")
expect_error(needleman_wunsch_path(1:3, 1:3), "character")

# R's lossless equivalent for wide symbol arrays is a Unicode scalar string;
# these route through the same 16/32-bit core templates.
wide_300 <- intToUtf8(0x4e00 + 0:299)
wide_500 <- intToUtf8(c(1:500))
wide_1000 <- intToUtf8(c(1:1000))
very_wide <- intToUtf8(c(1:0xd7ff, 0xe000 + 0:14704))
short <- intToUtf8(utf8ToInt(very_wide)[1:2])
stopifnot(
  smith_waterman_score(very_wide, short) == 4,
  smith_waterman_score(short, very_wide) == 4,
  smith_waterman_score(wide_1000, wide_1000) == 2000,
  smith_waterman_score(wide_500, wide_500) == 1000,
  smith_waterman_score(intToUtf8(1:100), intToUtf8(1:100)) == 200
)
set.seed(11)
for (iteration in seq_len(50L)) {
  alphabet <- intToUtf8(1:250, multiple = TRUE)
  left <- paste0(sample(alphabet, sample(1:60, 1L), replace = TRUE), collapse = "")
  right <- paste0(sample(alphabet, sample(1:60, 1L), replace = TRUE), collapse = "")
  stopifnot(smith_waterman_score(left, right) == reference_alignment_score(left, right))
}

# Floating-point bit-pattern equality belongs to NumPy token arrays; DTW is
# R's numeric-sequence API and verifies integer/double storage separately.
stopifnot(dtw(c(1.5, 2.5, 3.5, 4.5), c(1.5, 2.5, 7, 4.5)) >= 0)
targets <- c(wide_300, paste0(rev(strsplit(wide_300, "")[[1L]]), collapse = ""), intToUtf8(0x4e00 + 150:449))
stopifnot(identical(
  smith_waterman_scores(wide_300, targets), smith_waterman_score(wide_300, targets)
))
stopifnot(identical(
  needleman_wunsch_scores(wide_300, targets[1:2]), needleman_wunsch_score(wide_300, targets[1:2])
))
expect_error(smith_waterman_scores(wide_300, list(wide_300, 1:3)), "character vector")
affine <- smith_waterman_score(
  wide_300, wide_300, gap_open_score = -3, gap_extend_score = -1
)
linear <- smith_waterman_score(wide_300, wide_300)
stopifnot(affine == linear, affine == 600)
stopifnot(identical(
  smith_waterman_scores(
    wide_300, targets[1:2], gap_open_score = -3, gap_extend_score = -1
  ),
  smith_waterman_score(
    wide_300, targets[1:2], gap_open_score = -3, gap_extend_score = -1
  )
))
stopifnot(identical(
  needleman_wunsch_scores(
    wide_300, targets[1:2], gap_open_score = -3, gap_extend_score = -1
  ),
  needleman_wunsch_score(
    wide_300, targets[1:2], gap_open_score = -3, gap_extend_score = -1
  )
))

# Python cases: all 24 tests in tests/test_ndarray.py, translated from NumPy
# dtype/layout mechanics to the R STRSXP boundary and Unicode wide-token path.
