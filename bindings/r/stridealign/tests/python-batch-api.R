library(stridealign)

close_to <- function(actual, expected, tolerance = 1e-12) {
    stopifnot(isTRUE(all.equal(actual, expected, tolerance = tolerance)))
}

expect_error <- function(expression, pattern) {
    message <- tryCatch(
        {
            force(expression)
            NULL
        },
        error = conditionMessage
    )
    stopifnot(!is.null(message), grepl(pattern, message, fixed = TRUE))
}

# Every name in stride_align.__all__ has an R counterpart. Compatibility
# packages remain separate in Python and are deliberately absent in R.
python_api <- c(
    "AlignmentPath", "AlignmentResult", "BackendKind", "BackendRecord", "Scorer",
    "Scores", "available_backends", "backend_is_available", "cdist", "cdist_above_threshold",
    "cdist_top_k", "cdist_top_k_per_query", "damerau_levenshtein_best", "damerau_levenshtein_normalized_best", "damerau_levenshtein_normalized_score",
    "damerau_levenshtein_normalized_scores", "damerau_levenshtein_normalized_top_k", "damerau_levenshtein_score", "damerau_levenshtein_scores", "damerau_levenshtein_top_k",
    "detect_best_backend", "dtw", "dtw_distances", "extract", "extract_best",
    "hamming_best", "hamming_normalized_best", "hamming_normalized_score", "hamming_normalized_scores", "hamming_normalized_top_k",
    "hamming_score", "hamming_scores", "hamming_top_k", "indel_best", "indel_normalized_best",
    "indel_normalized_score", "indel_normalized_scores", "indel_normalized_top_k", "indel_score", "indel_scores",
    "indel_top_k", "jaro_best", "jaro_similarities", "jaro_similarity", "jaro_top_k",
    "jaro_winkler_best", "jaro_winkler_similarities", "jaro_winkler_similarity", "jaro_winkler_top_k", "cosine",
    "cosine_similarities", "dice", "dice_similarities", "jaccard", "jaccard_similarities",
    "lcs_length", "lcs_substring", "lcs_substring_length", "monge_elkan", "overlap",
    "overlap_similarities", "partial_ratio", "partial_token_set_ratio", "partial_token_sort_ratio", "ratcliff_obershelp_similarities",
    "ratcliff_obershelp_similarity", "token_set_ratio", "token_sort_ratio", "WRatio", "LevenshteinScorer",
    "levenshtein_best", "levenshtein_normalized_best", "levenshtein_normalized_score", "levenshtein_normalized_scores", "levenshtein_normalized_top_k",
    "levenshtein_score", "levenshtein_scores", "levenshtein_top_k", "needleman_wunsch_cigar", "needleman_wunsch_normalized_score",
    "needleman_wunsch_normalized_scores", "needleman_wunsch_path", "needleman_wunsch_path_info", "needleman_wunsch_score", "needleman_wunsch_scores",
    "needleman_wunsch_trace_cigar", "needleman_wunsch_trade_cigar", "smith_waterman_cigar", "smith_waterman_farrar_normalized_score", "smith_waterman_farrar_normalized_scores",
    "smith_waterman_farrar_score", "smith_waterman_farrar_scores", "smith_waterman_normalized_score", "smith_waterman_normalized_scores", "smith_waterman_path",
    "smith_waterman_path_info", "smith_waterman_score", "smith_waterman_best", "smith_waterman_scores", "smith_waterman_top_k",
    "smith_waterman_trace_cigar", "smith_waterman_trade_cigar", "beider_morse", "BmpmRuleType", "caverphone",
    "cologne_phonetic", "daitch_mokotoff", "double_metaphone", "DoubleMetaphoneVariant", "match_rating_codex",
    "match_rating_compare", "metaphone", "metaphone_equal", "MetaphoneVariant", "nysiis",
    "nysiis_equal", "soundex", "soundex_equal", "true_damerau_levenshtein_best", "true_damerau_levenshtein_normalized_best",
    "true_damerau_levenshtein_normalized_score", "true_damerau_levenshtein_normalized_scores", "true_damerau_levenshtein_normalized_top_k", "true_damerau_levenshtein_score", "true_damerau_levenshtein_scores",
    "true_damerau_levenshtein_top_k"
)
exports <- getNamespaceExports("stridealign")
stopifnot(
    length(python_api) == 126L,
    length(setdiff(python_api, exports)) == 0L,
    !any(c(
        "rapidfuzz", "parasail", "jellyfish", "thefuzz", "extractOne",
        "extract_iter", "sw_trace_striped_16"
    ) %in% exports)
)

# The complete sixteen-scorer batch catalog agrees with pairwise scoring and
# can be used by cdist through either its public integer ID or name.
catalog <- list(
    list(Scorer$LEVENSHTEIN, levenshtein_score, levenshtein_scores),
    list(Scorer$LEVENSHTEIN_NORMALIZED, levenshtein_normalized_score, levenshtein_normalized_scores),
    list(Scorer$DAMERAU_LEVENSHTEIN, damerau_levenshtein_score, damerau_levenshtein_scores),
    list(Scorer$DAMERAU_LEVENSHTEIN_NORMALIZED, damerau_levenshtein_normalized_score, damerau_levenshtein_normalized_scores),
    list(Scorer$HAMMING, hamming_score, hamming_scores),
    list(Scorer$HAMMING_NORMALIZED, hamming_normalized_score, hamming_normalized_scores),
    list(Scorer$JARO, jaro_similarity, jaro_similarities),
    list(Scorer$JARO_WINKLER, jaro_winkler_similarity, jaro_winkler_similarities),
    list(Scorer$INDEL, indel_score, indel_scores),
    list(Scorer$INDEL_NORMALIZED, indel_normalized_score, indel_normalized_scores),
    list(Scorer$TRUE_DAMERAU_LEVENSHTEIN, true_damerau_levenshtein_score, true_damerau_levenshtein_scores),
    list(Scorer$TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED, true_damerau_levenshtein_normalized_score, true_damerau_levenshtein_normalized_scores),
    list(Scorer$SMITH_WATERMAN, smith_waterman_score, smith_waterman_scores),
    list(Scorer$SMITH_WATERMAN_NORMALIZED, smith_waterman_normalized_score, smith_waterman_normalized_scores),
    list(Scorer$NEEDLEMAN_WUNSCH, needleman_wunsch_score, needleman_wunsch_scores),
    list(Scorer$NEEDLEMAN_WUNSCH_NORMALIZED, needleman_wunsch_normalized_score, needleman_wunsch_normalized_scores)
)
query <- "MARTHA"
targets <- c("MARTHA", "MARHTA", "ARTHUR", intToUtf8(c(77, 196, 82, 84, 72, 65)), NA_character_)
for (entry in catalog) {
    id <- entry[[1L]]
    pair <- entry[[2L]]
    batch <- entry[[3L]]
    expected <- pair(query, targets)
    close_to(batch(query, targets), expected)
    row <- cdist(query, targets, scorer = id)
    stopifnot(identical(dim(row), c(1L, length(targets))))
    close_to(as.numeric(row[1L, ]), expected)
}

queries <- c("MARTHA", "ARTHUR")
targets_no_na <- c("MARTHA", "MARHTA", "ARTHUR")
for (entry in catalog) {
    scores <- cdist(queries, targets_no_na, scorer = entry[[1L]])
    stopifnot(identical(dim(scores), c(2L, 3L)))
    for (index in seq_along(queries)) {
        close_to(scores[index, ], entry[[3L]](queries[[index]], targets_no_na))
    }
}

# R selection results use data frames/lists and R's one-based indices.
choices <- c("sitting", "kitten", "bitten", "mittens")
top_distance <- levenshtein_top_k("kitten", choices, k = 2)
stopifnot(
    identical(top_distance$target, c("kitten", "bitten")),
    identical(top_distance$score, c(0, 1)),
    identical(top_distance$index, c(2L, 3L))
)
top_similarity <- levenshtein_normalized_top_k("kitten", choices, k = 2)
stopifnot(identical(top_similarity$index, c(2L, 3L)))
stopifnot(identical(levenshtein_best("kitten", choices)$index, 2L))
stopifnot(identical(extract_best(
    "kitten", choices, scorer = Scorer$LEVENSHTEIN
)$index, 2L))
stopifnot(identical(extract(
    "kitten", choices, scorer = Scorer$LEVENSHTEIN, k = 2
)$index, c(2L, 3L)))

prepared <- LevenshteinScorer("kitten")
stopifnot(
    identical(prepared$query, "kitten"),
    prepared$distance("sitting") == 3,
    identical(prepared$distances(choices), levenshtein_scores("kitten", choices))
)
close_to(
    prepared$normalized_distances(choices),
    levenshtein_normalized_scores("kitten", choices)
)

# Dense, filtered, global top-k, and per-query top-k cdist forms.
filtered <- cdist_above_threshold(
    "cat", c("cat", "cut", "dog"),
    scorer = Scorer$LEVENSHTEIN_NORMALIZED, threshold = 0.6
)
stopifnot(
    identical(filtered$target, c("cat", "cut")),
    identical(filtered$query_index, c(1L, 1L)),
    identical(filtered$target_index, c(1L, 2L))
)
global <- cdist_top_k(
    c("cat", "dog"), c("cat", "cot", "dog"),
    scorer = Scorer$JARO, k = 2, reject_duplicates = TRUE
)
stopifnot(nrow(global) == 2L, all(global$query != global$target))
per_query <- cdist_top_k_per_query(
    c("cat", "dog"), c("cat", "cot", "dog"),
    scorer = Scorer$LEVENSHTEIN_NORMALIZED, k = 2
)
stopifnot(
    identical(names(per_query), c("cat", "dog")),
    identical(per_query[[1L]]$index, c(1L, 2L)),
    identical(per_query[[2L]]$index[[1L]], 3L)
)

# Substitution-matrix batches and cdist use the same broadcasting contract.
close_to(
    smith_waterman_scores("HE", c("HE", "HH"), matrix = blosum62),
    c(13, 8)
)
matrix_scores <- cdist(c("HE", "HH"), c("HE", "HH"), matrix = blosum62)
close_to(matrix_scores, base::matrix(c(13, 8, 8, 16), 2L, byrow = TRUE))
matrix_filtered <- cdist_above_threshold(
    c("HE", "HH"), c("HE", "HH"), matrix = blosum62, threshold = 13
)
stopifnot(nrow(matrix_filtered) == 2L, all(matrix_filtered$score >= 13))
matrix_top <- cdist_top_k(
    c("HE", "HH"), c("HE", "HH"), matrix = blosum62, k = 2
)
stopifnot(identical(matrix_top$score, c(16, 13)))

expect_error(cdist("a", "a"), "requires either scorer or matrix")
expect_error(
    cdist_above_threshold("a", "a", scorer = Scorer$LEVENSHTEIN, threshold = 0.5),
    "normalized or similarity"
)
expect_error(
    cdist_top_k("a", "a", scorer = Scorer$LEVENSHTEIN, k = 1),
    "normalized or similarity"
)
expect_error(
    cdist_top_k_per_query("a", "a", scorer = Scorer$LEVENSHTEIN),
    "normalized or similarity"
)
