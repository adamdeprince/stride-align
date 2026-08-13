library(stridealign)

close_to <- function(actual, expected, tolerance = 1e-12) {
    stopifnot(isTRUE(all.equal(actual, expected, tolerance = tolerance)))
}

smith_waterman_affine <- function(query, target) {
    stride_smith_waterman_affine(query, target, gap_open_score = -2,
                                 gap_extend_score = -1)
}

needleman_wunsch_affine <- function(query, target) {
    stride_needleman_wunsch_affine(query, target, gap_open_score = -2,
                                   gap_extend_score = -1)
}

# This is the same sixteen-function catalog exercised by the DuckDB binding.
scorers <- list(
    list(stride_levenshtein, FALSE, FALSE),
    list(stride_levenshtein_similarity, TRUE, FALSE),
    list(stride_osa, FALSE, FALSE),
    list(stride_osa_similarity, TRUE, FALSE),
    list(stride_true_damerau_levenshtein, FALSE, FALSE),
    list(stride_true_damerau_levenshtein_similarity, TRUE, FALSE),
    list(stride_indel, FALSE, FALSE),
    list(stride_indel_similarity, TRUE, FALSE),
    list(stride_hamming, FALSE, TRUE),
    list(stride_hamming_similarity, TRUE, TRUE),
    list(stride_jaro, TRUE, FALSE),
    list(stride_jaro_winkler, TRUE, FALSE),
    list(stride_smith_waterman, FALSE, FALSE),
    list(stride_needleman_wunsch, FALSE, FALSE),
    list(smith_waterman_affine, FALSE, FALSE),
    list(needleman_wunsch_affine, FALSE, FALSE)
)

# R character values cannot contain an embedded NUL. Every other DuckDB
# scalar parity case is repeated here, including BMP, astral, combining-mark,
# ZWJ, and long-input paths.
variable_cases <- list(
    c("", ""),
    c("abc", ""),
    c("", "abc"),
    c("abc", "abc"),
    c("kitten", "sitting"),
    c("flaw", "lawn"),
    c("ca", "abc"),
    c("MARTHA", "MARHTA"),
    c("ACCGT", "ACG"),
    c("Müller", "Mueller"),
    c("γειά", "γεια"),
    c("你好世界", "你好啊"),
    c("hi 👋", "ho 👋"),
    c("𝛼+𝛽", "𝛼+𝛾"),
    c(paste0(intToUtf8(0xffff), "x"), paste0(intToUtf8(0xffff), "y")),
    c(paste0(intToUtf8(0x10fffe), "x"), paste0(intToUtf8(0x10fffe), "y")),
    c("नमस्ते", "नमस्कार"),
    c("👨‍👩‍👧", "👨‍👩‍👧‍👦"),
    c(paste0(rep("abcdefghij", 10), collapse = ""),
      paste0("xbcdefghij", paste0(rep("abcdefghij", 9), collapse = ""))),
    c(paste0(rep("你好", 40), collapse = ""),
      paste0(rep("你号", 40), collapse = ""))
)

hamming_cases <- list(
    c("", ""),
    c("abc", "abc"),
    c("karolin", "kathrin"),
    c("café", "cafe"),
    c("你好", "你号"),
    c("🎉🎈ab", "🎈🎉ab"),
    c(paste0(strrep("a", 127), "b"), paste0(strrep("a", 127), "c"))
)

for (entry in scorers) {
    scorer <- entry[[1L]]
    cases <- if (entry[[3L]]) hamming_cases else variable_cases
    left <- vapply(cases, `[[`, character(1), 1L)
    right <- vapply(cases, `[[`, character(1), 2L)
    vector_scores <- scorer(left, right)
    scalar_scores <- vapply(
        seq_along(left),
        function(index) scorer(left[[index]], right[[index]]),
        numeric(1)
    )
    close_to(vector_scores, scalar_scores)
    stopifnot(identical(
        is.na(scorer(c(NA_character_, "x", NA_character_),
                     c("x", NA_character_, NA_character_))),
        rep(TRUE, 3L)
    ))
}

# Cross an R vector boundary larger than DuckDB's standard vector size, using
# the same mixture of ASCII, BMP, astral, and unusual Unicode rows.
row_count <- 2057L
left <- character(row_count)
right <- character(row_count)
for (index in seq_len(row_count)) {
    value <- index - 1L
    branch <- value %% 5L
    if (branch == 0L) {
        left[[index]] <- sprintf("%08x", value)
        replacement <- if (endsWith(left[[index]], "0")) "1" else "0"
        right[[index]] <- paste0(substr(left[[index]], 1L, 7L), replacement)
    } else if (branch == 1L) {
        left[[index]] <- paste0("你", value %% 10L, "好界")
        right[[index]] <- paste0("你", value %% 10L, "号界")
    } else if (branch == 2L) {
        left[[index]] <- paste0("👋", value %% 10L, "ab")
        right[[index]] <- paste0("🌸", value %% 10L, "ab")
    } else if (branch == 3L) {
        left[[index]] <- paste0("𝛼+", value %% 10L)
        right[[index]] <- paste0("𝛽+", value %% 10L)
    } else {
        left[[index]] <- paste0("a", intToUtf8(0xffff), value %% 10L)
        right[[index]] <- paste0("b", intToUtf8(0xffff), value %% 10L)
    }
}
for (entry in scorers) {
    scorer <- entry[[1L]]
    vector_scores <- scorer(left, right)
    scalar_scores <- vapply(
        seq_len(row_count),
        function(index) scorer(left[[index]], right[[index]]),
        numeric(1)
    )
    close_to(vector_scores, scalar_scores)
}

# Exercise all three lossless token widths used by cardinality packing.
valid_codepoints <- c(
    0x1000:0xd7ff,
    0xe000:0xffff,
    0x10000:(0x10000 + 6607L)
)
stopifnot(length(valid_codepoints) == 66000L)
for (symbol_count in c(200L, 300L, 66000L)) {
    text <- intToUtf8(valid_codepoints[seq_len(symbol_count)])
    target <- if (symbol_count < 66000L) text else intToUtf8(valid_codepoints[1:2])
    expected <- if (symbol_count < 66000L) 0 else 65998
    stopifnot(levenshtein_score(text, target) == expected)
}
