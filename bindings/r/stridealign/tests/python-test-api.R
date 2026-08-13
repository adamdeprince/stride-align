source("helpers.R")

backend <- detect_best_backend()
records <- available_backends()
stopifnot(
  is.character(backend), length(backend) == 1L,
  any(vapply(records, function(record) record$name == "generic" && record$available, logical(1))),
  backend != BackendKind$SWAR,
  backend_is_available(backend)
)
if (backend_is_available(BackendKind$SWAR)) stopifnot("swar" %in% stride_available_backends())

query <- "foo"
targets <- c("foo", "bar", "food")
stopifnot(
  identical(Scores(query)$compare(targets), smith_waterman_score(query, targets)),
  identical(Scores(query, "needleman_wunsch")$compare(targets), needleman_wunsch_score(query, targets)),
  identical(Scores(query, "farrar")$compare(targets), smith_waterman_farrar_score(query, targets)),
  identical(smith_waterman_scores(query, targets), smith_waterman_score(query, targets)),
  identical(needleman_wunsch_scores(query, targets), needleman_wunsch_score(query, targets)),
  identical(smith_waterman_farrar_scores(query, targets), smith_waterman_farrar_score(query, targets))
)
stopifnot(
  smith_waterman_normalized_score("hello", "hello") == 1,
  smith_waterman_normalized_score("hello", "say hello world") == 1,
  smith_waterman_normalized_score("abc", "abcxyz", match_score = 2) == 1
)
close_to(smith_waterman_normalized_score("abc", "abXXX"), 4 / 6)
close_to(needleman_wunsch_normalized_score("abc", "abcxyz"), 3 / 12)
for (family in list(
  list(smith_waterman_normalized_score, smith_waterman_normalized_scores),
  list(needleman_wunsch_normalized_score, needleman_wunsch_normalized_scores),
  list(smith_waterman_farrar_normalized_score, smith_waterman_farrar_normalized_scores)
)) close_to(family[[2L]](query, targets), family[[1L]](query, targets))
expect_error(smith_waterman_normalized_score("abc", "abc", match_score = 0), "match_score must be positive")
expect_error(needleman_wunsch_normalized_score("abc", "abc", match_score = -1), "match_score must be positive")
expect_error(smith_waterman_farrar_normalized_scores("abc", "abc", match_score = 0), "match_score must be positive")
stopifnot(
  needleman_wunsch_normalized_score("aaa", "bbbbbb", mismatch_score = -5, gap_score = -5) == 0,
  smith_waterman_normalized_score("", "") == 1,
  needleman_wunsch_normalized_score("", "") == 1,
  smith_waterman_normalized_score("", "abc") == 0
)
close_to(
  smith_waterman_farrar_normalized_score("needle", c("needle", "n33dle", "haystack")),
  smith_waterman_normalized_score("needle", c("needle", "n33dle", "haystack"))
)

stopifnot(
  levenshtein_score("kitten", "sitting") == 3,
  levenshtein_score("", "") == 0,
  levenshtein_score("abc", "") == 3,
  levenshtein_score("", "abc") == 3,
  levenshtein_score("abc", "abc") == 0,
  levenshtein_score("flaw", "lawn") == 2,
  levenshtein_normalized_score("foo", "foo") == 1,
  levenshtein_normalized_score("", "") == 1,
  levenshtein_normalized_score("abc", "xyz") == 0
)
close_to(levenshtein_normalized_score("foobar", "foobaz"), 5 / 6)
stopifnot(
  identical(levenshtein_scores("kitten", c("kitten", "sitting", "kit", "")), c(0, 3, 3, 6)),
  is.double(levenshtein_normalized_scores("foo", c("foo", "bar", "food")))
)
close_to(levenshtein_normalized_scores("foo", c("foo", "bar", "food")), c(1, 0, 0.75))
bytes_left <- "hello"
bytes_right <- "hallo"
Encoding(bytes_left) <- "bytes"
Encoding(bytes_right) <- "bytes"
stopifnot(levenshtein_score(bytes_left, bytes_right) == 1)
long_query <- strrep("abcdefghij", 10)
long_target <- paste0("xbcdefghij", strrep("abcdefghij", 9))
stopifnot(levenshtein_score(long_query, long_target) == 1)
stopifnot(
  levenshtein_score("café", "cafe") == 1,
  levenshtein_score("🎉🎈", "🎉🎈") == 0,
  levenshtein_score("🎉🎈", "🎈🎉") == 2
)

stopifnot(needleman_wunsch_score("ACGT", "ACCT") == 5)
nw <- needleman_wunsch_path("ACGT", "ACCT")
stopifnot(
  inherits(nw, "stride_alignment_result"), nw$score == 5,
  nw$query_start == 0, nw$query_end == 4, nw$target_start == 0,
  nw$target_end == 4, identical(unname(nw$aligned), c("ACGT", "ACCT")),
  nw$operations == "==X="
)
nw_info <- needleman_wunsch_path_info("ACGT", "ACCT")
stopifnot(
  inherits(nw_info, "stride_alignment_path"), nw_info$cigar == "2=1X1=",
  nw_info$matches == 3, nw_info$mismatches == 1,
  nw_info$insertions == 0, nw_info$deletions == 0,
  nw_info$aligned_length == 4,
  is.null(nw_info[["aligned", exact = TRUE]])
)
stopifnot(
  needleman_wunsch_cigar("ACGT", "ACCT") == nw_info$cigar,
  needleman_wunsch_trace_cigar("ACGT", "ACCT") == nw_info$cigar,
  needleman_wunsch_trade_cigar("ACGT", "ACCT") == nw_info$cigar,
  needleman_wunsch_path("A🙂", "A🙂")$score == 4,
  smith_waterman_score("ACCGT", "CCG") == 6,
  smith_waterman_farrar_score("GGCCTT", "CGGTTAT") == smith_waterman_score("GGCCTT", "CGGTTAT"),
  smith_waterman_farrar_score("🙂🙃🙂", "🙃🙂", width = 8) == 4
)
for (function_ in list(smith_waterman_score, needleman_wunsch_score, smith_waterman_farrar_score)) stopifnot(
  function_("ABCBDAB", "BDCABA", match_score = 2, mismatch_score = -1, gap_score = 0) == 8
)
stopifnot(
  needleman_wunsch_score("AB", "XYZ", match_score = 1, mismatch_score = 1, gap_score = -1) == 1,
  smith_waterman_score("AA", "BBB", match_score = 1, mismatch_score = 1, gap_score = 1) == 4,
  smith_waterman_farrar_score("AA", "BBB", match_score = 1, mismatch_score = 1, gap_score = 1) == 4
)
affine_arguments <- list(
  match_score = 2, mismatch_score = -1, gap_score = -1,
  gap_open_score = -3, gap_extend_score = -1
)
for (function_ in list(smith_waterman_score, smith_waterman_farrar_score, needleman_wunsch_score)) stopifnot(
  do.call(function_, c(list("AAABBB", "AAACCCBBB"), affine_arguments)) == 7
)
affine <- do.call(smith_waterman_path, c(list("AAABBB", "AAACCCBBB"), affine_arguments))
stopifnot(
  affine$score == 7, identical(unname(affine$aligned), c("AAA---BBB", "AAACCCBBB")),
  affine$operations == "===III===", affine$cigar == "3=3I3="
)

sw <- smith_waterman_path("ACCGT", "CCG")
sw_info <- smith_waterman_path_info("ACCGT", "CCG")
stopifnot(
  sw$score == 6, sw$query_start == 1, sw$query_end == 4,
  sw$target_start == 0, sw$target_end == 3,
  identical(unname(sw$aligned), c("CCG", "CCG")), sw$operations == "===",
  sw_info$cigar == "3=", sw_info$matches == 3, sw_info$aligned_length == 3,
  smith_waterman_cigar("ACCGT", "CCG") == "3=",
  smith_waterman_trace_cigar("ACCGT", "CCG") == "3=",
  smith_waterman_trade_cigar("ACCGT", "CCG") == "3="
)
stopifnot(
  smith_waterman_path("ACCGT", "CCG", width = 64)$score == 6,
  smith_waterman_score("ACCGT", "CCG", width = NULL) == 6,
  smith_waterman_score("ACCGT", "CCG", width = 0) == 6
)
expect_error(smith_waterman_score("AC", "AC", width = 7), "width must be")
expect_error(smith_waterman_farrar_score("AC", "AC", width = 7), "width must be")

zero <- smith_waterman_path("AAAA", "TTTT", mismatch_score = -3, gap_score = -2)
stopifnot(zero$score == 0, identical(unname(zero$aligned), c("", "")), zero$operations == "")
stopifnot(
  smith_waterman_scores("ACCGT", "CCG")[[1L]] == smith_waterman_score("ACCGT", "CCG"),
  needleman_wunsch_scores("ACGT", "ACCT")[[1L]] == needleman_wunsch_score("ACGT", "ACCT"),
  levenshtein_scores("god", "good")[[1L]] == levenshtein_score("god", "good"),
  levenshtein_normalized_scores("god", "good")[[1L]] == levenshtein_normalized_score("god", "good"),
  damerau_levenshtein_scores("abcd", "acbd")[[1L]] == 1,
  damerau_levenshtein_normalized_scores("abcd", "acbd")[[1L]] ==
    damerau_levenshtein_normalized_score("abcd", "acbd")
)
prepared <- LevenshteinScorer("god")
stopifnot(
  prepared$distance("good") == 1, prepared$distance("god") == 0,
  prepared$distance("abc") == levenshtein_score("god", "abc"),
  prepared$distance("good", score_cutoff = 0) == 1,
  identical(prepared$distances(c("good", "god", "god!")), c(1, 0, 1)),
  prepared$query == "god", prepared$normalized_distance("god") == 1
)
close_to(prepared$normalized_distances(c("god", "good")), c(1, 0.75))

# Python module/load monkeypatches, CLI/benchmark commands, and direct nanobind
# backend-module probes are represented by backend-generic.R,
# backend-invalid.R, package installation checks, and the public-dispatch
# result tests above. Parasail benchmark-adapter cases are compatibility code
# and are intentionally excluded.
# Python cases: all native public-API and dispatch assertions in test_api.py.
