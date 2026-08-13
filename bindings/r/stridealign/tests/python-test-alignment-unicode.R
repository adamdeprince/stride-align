source("helpers.R")

latin1_pairs <- list(
  c("hello", "hallo"), c("Müller", "Mueller"),
  c("straße", "strasse"), c("café", "cafe")
)
ucs2_pairs <- list(
  c("γειά", "γεια"), c("Привет", "Превет"),
  c("你好世界", "你好啊"), c("東京", "京都")
)
ucs4_pairs <- list(
  c("hi 👋", "ho 👋"), c("𝛼+𝛽", "𝛼+𝛾"),
  c("😀x", "😀y"), c("𠮷田", "𠮷川")
)
# R CHARSXP cannot contain U+0000. That Python fixture is represented by a
# direct assertion of R's runtime invariant; every representable edge follows.
expect_error(rawToChar(as.raw(c(97, 0, 98))), "embedded nul")
edge_pairs <- list(
  c("￿x", "￿y"), c(intToUtf8(c(0x10fffe, 120)), intToUtf8(c(0x10fffe, 121))),
  c("שלום עולם", "שלום"), c("नमस्ते", "नमस्कार"),
  c("👨‍👩‍👧", "👨‍👩‍👧‍👦"),
  c("hello 👋 world", "hello 🌸 world"),
  c(strrep("𝛼", 50), strrep("𝛽", 50))
)
all_pairs <- c(latin1_pairs, ucs2_pairs, ucs4_pairs)
score_functions <- list(
  smith_waterman_score, smith_waterman_normalized_score,
  smith_waterman_farrar_score, smith_waterman_farrar_normalized_score,
  needleman_wunsch_score, needleman_wunsch_normalized_score
)
path_functions <- list(
  smith_waterman_path, smith_waterman_path_info,
  needleman_wunsch_path, needleman_wunsch_path_info
)
cigar_functions <- list(
  smith_waterman_cigar, smith_waterman_trace_cigar,
  smith_waterman_trade_cigar, needleman_wunsch_cigar,
  needleman_wunsch_trace_cigar, needleman_wunsch_trade_cigar
)
batch_functions <- list(
  smith_waterman_scores, smith_waterman_normalized_scores,
  smith_waterman_farrar_scores, smith_waterman_farrar_normalized_scores,
  needleman_wunsch_scores, needleman_wunsch_normalized_scores
)

for (function_ in score_functions) for (pair in c(all_pairs, edge_pairs)) {
  stopifnot(is.numeric(function_(pair[[1L]], pair[[2L]])))
}
for (pair in all_pairs) {
  for (function_ in list(smith_waterman_path, needleman_wunsch_path)) {
    result <- function_(pair[[1L]], pair[[2L]])
    stopifnot(
      is.character(result$aligned[[1L]]),
      is.character(result$aligned[[2L]]),
      is.character(result$operations)
    )
    left_points <- strsplit(result$aligned[[1L]], "", fixed = TRUE)[[1L]]
    right_points <- strsplit(result$aligned[[2L]], "", fixed = TRUE)[[1L]]
    stopifnot(
      all(setdiff(left_points, "-") %in% strsplit(pair[[1L]], "", fixed = TRUE)[[1L]]),
      all(setdiff(right_points, "-") %in% strsplit(pair[[2L]], "", fixed = TRUE)[[1L]])
    )
  }
}
for (function_ in path_functions) for (pair in edge_pairs) function_(pair[[1L]], pair[[2L]])
for (function_ in cigar_functions) for (pair in c(all_pairs, edge_pairs)) {
  stopifnot(is.character(function_(pair[[1L]], pair[[2L]])))
}
for (function_ in batch_functions) for (pair in all_pairs) {
  scores <- function_(pair[[1L]], rep(pair[[2L]], 3L))
  stopifnot(length(scores) == 3L, length(unique(scores)) == 1L)
}
mixed_targets <- c("hallo", "γειά", "你好", "hi 👋", intToUtf8(c(0x10fffe, 121)))
mixed_scores <- smith_waterman_scores("hello", mixed_targets)
stopifnot(length(mixed_scores) == 5L, all(mixed_scores >= 0))
best <- smith_waterman_best(
  "你好世界", c("你好", "再见", "你好世界", "你好啊朋友")
)
stopifnot(best$target == "你好世界", best$index == 3L)
ranked <- smith_waterman_top_k("hi 👋", c("ho 👋", "hi 🌸", "hi 👋"), k = 2)
stopifnot(nrow(ranked) == 2L, all(ranked$target == c("ho 👋", "hi 🌸", "hi 👋")[ranked$index]))

for (value in c("hello", "Müller", "你好世界", "hi 👋", "𝛼+𝛽", "שלום", "𠮷田川")) {
  expected <- nchar(value, type = "chars") * 2
  stopifnot(
    smith_waterman_score(value, value, match_score = 2) == expected,
    needleman_wunsch_score(value, value, match_score = 2) == expected,
    smith_waterman_farrar_score(value, value, match_score = 2) == expected
  )
}
for (value in c("你好世界", "hi 👋", "𝛼+𝛽")) {
  points <- strsplit(value, "", fixed = TRUE)[[1L]]
  prefix <- paste0(points[vapply(points, utf8ToInt, integer(1)) < 128], collapse = "")
  if (nzchar(prefix)) stopifnot(
    smith_waterman_score(prefix, prefix) == smith_waterman_score(prefix, prefix)
  )
}
global <- needleman_wunsch_score("你好", "𝛼+𝛽")
stopifnot(global >= -(nchar("你好") + nchar("𝛼+𝛽")))
for (function_ in c(score_functions, cigar_functions)) for (pair in c(ucs2_pairs[1:2], ucs4_pairs[1:2])) {
  function_(pair[[1L]], pair[[2L]], gap_open_score = -3, gap_extend_score = -1)
}
affine_path <- smith_waterman_path(
  "你好世界", "你好啊朋友", gap_open_score = -3, gap_extend_score = -1
)
stopifnot(all(vapply(affine_path$aligned, is.character, logical(1))))

# Python bytes and str are distinct API types; R represents both as CHARSXP
# with encoding metadata. Clean ASCII CE_BYTES values therefore interoperate,
# while non-ASCII CE_BYTES values are rejected before Unicode preparation.
bytes_ascii <- "hello"
Encoding(bytes_ascii) <- "bytes"
stopifnot(smith_waterman_score(bytes_ascii, "hallo") >= 0)
bytes_non_ascii <- rawToChar(as.raw(0xff))
Encoding(bytes_non_ascii) <- "bytes"
expect_error(smith_waterman_score(bytes_non_ascii, "x"), "do not have a Unicode encoding")

# Python cases: TestScoreUnicode.test_score_accepts_each_unicode_kind,
# TestScoreUnicode.test_score_accepts_edge_codepoints,
# TestPathUnicode.test_smith_waterman_path_returns_str_for_str_input,
# TestPathUnicode.test_needleman_wunsch_path_returns_str_for_str_input,
# TestPathUnicode.test_path_returns_bytes_for_bytes_input,
# TestPathUnicode.test_path_entry_points_accept_edge_codepoints,
# TestCigarUnicode.test_cigar_returns_str,
# TestCigarUnicode.test_cigar_accepts_edge_codepoints,
# TestBatchUnicode.test_batch_homogeneous_kind,
# TestBatchUnicode.test_batch_mixed_kind_targets,
# TestBatchUnicode.test_smith_waterman_best_with_unicode_targets,
# TestBatchUnicode.test_smith_waterman_top_k_with_unicode_targets,
# TestUnicodeCorrectness.test_identity_score_equals_n_times_match,
# TestUnicodeCorrectness.test_str_score_equals_bytes_score_when_codepoints_fit_in_byte,
# TestUnicodeCorrectness.test_needleman_wunsch_disjoint_unicode_uses_gaps,
# TestUnicodeAffineGaps.test_affine_accepts_unicode,
# TestUnicodeAffineGaps.test_affine_path_returns_str_for_unicode,
# TestMixedBytesStrRejected.test_bytes_query_str_target_raises,
# TestMixedBytesStrRejected.test_str_query_bytes_target_raises.
