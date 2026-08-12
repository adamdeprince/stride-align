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

stopifnot(
    is.character(stride_backend()),
    length(stride_backend()) == 1L,
    stride_backend() %in% stride_available_backends(),
    tail(stride_available_backends(), 1L) == "generic"
)

# Every exported distance and similarity function gets a scalar exercise.
stopifnot(stride_levenshtein("kitten", "sitting") == 3)
close_to(stride_levenshtein_similarity("kitten", "sitting"), 4 / 7)
stopifnot(stride_osa("CA", "AC") == 1)
close_to(stride_osa_similarity("CA", "AC"), 0.5)
stopifnot(stride_true_damerau_levenshtein("CA", "ABC") == 2)
close_to(
    stride_true_damerau_levenshtein_similarity("CA", "ABC"),
    1 / 3
)
stopifnot(stride_indel("abc", "axc") == 2)
close_to(stride_indel_similarity("abc", "axc"), 2 / 3)
stopifnot(stride_hamming("karolin", "kathrin") == 3)
close_to(stride_hamming_similarity("karolin", "kathrin"), 4 / 7)
close_to(stride_jaro("MARTHA", "MARHTA"), 0.9444444444444444)
close_to(stride_jaro_winkler("MARTHA", "MARHTA"), 0.9611111111111111)

# Every exported sequence-alignment function gets default and custom scoring.
stopifnot(stride_smith_waterman("GATTACA", "GATTACA") == 14)
stopifnot(stride_needleman_wunsch("GATTACA", "GATTACA") == 14)
stopifnot(stride_smith_waterman_affine("GATTACA", "GATTACA") == 14)
stopifnot(stride_needleman_wunsch_affine("GATTACA", "GATTACA") == 14)
stopifnot(stride_smith_waterman("abc", "xbc", 3, -2, -2) == 6)
stopifnot(stride_needleman_wunsch("abc", "xbc", 3, -2, -2) == 4)
stopifnot(stride_smith_waterman_affine("abc", "xbc", 3, -2, -4, -1) == 6)
stopifnot(stride_needleman_wunsch_affine("abc", "xbc", 3, -2, -4, -1) == 4)

# R-vector behavior: exact lengths, scalar broadcasting, empty vectors, and NA.
queries <- c("kitten", "flaw", NA_character_, "same")
targets <- c("sitting", "lawn", "anything", "same")
stopifnot(identical(stride_levenshtein(queries, targets), c(3, 2, NA, 0)))
stopifnot(identical(
    stride_levenshtein(c("a", "ab", "abc"), "abc"),
    c(2, 1, 0)
))
stopifnot(identical(
    stride_levenshtein("abc", c("a", "ab", "abc")),
    c(2, 1, 0)
))
stopifnot(identical(stride_levenshtein(character(), "abc"), numeric()))
stopifnot(identical(stride_levenshtein(character(), character()), numeric()))

# Short BMP and astral text, then long low- and medium-cardinality text.
nihao_world <- intToUtf8(c(0x4f60, 0x597d, 0x4e16, 0x754c))
nihao_mortal <- intToUtf8(c(0x4f60, 0x597d, 0x4e16, 0x95f4))
nihao <- intToUtf8(c(0x4f60, 0x597d))
ni_men_hao <- intToUtf8(c(0x4f60, 0x4eec, 0x597d))
grinning <- intToUtf8(0x1f600)
smiley <- intToUtf8(0x1f603)
stopifnot(stride_levenshtein(nihao_world, nihao_mortal) == 1)
stopifnot(stride_levenshtein(paste0("a", grinning), paste0("a", smiley)) == 1)
stopifnot(stride_needleman_wunsch(nihao, ni_men_hao) == 3)

greek <- intToUtf8(0x3b1:0x3b5, multiple = TRUE)
long_u8_query <- paste0(rep(paste0(greek[1:4], collapse = ""), 40), collapse = "")
long_u8_target <- paste0(substr(long_u8_query, 1, nchar(long_u8_query) - 1), greek[5])
stopifnot(stride_levenshtein(long_u8_query, long_u8_target) == 1)

medium_alphabet <- intToUtf8(0x400 + seq_len(300), multiple = TRUE)
long_u16_query <- paste0(medium_alphabet, collapse = "")
long_u16_target <- paste0(c(medium_alphabet[-300], intToUtf8(0x754c)), collapse = "")
stopifnot(stride_levenshtein(long_u16_query, long_u16_target) == 1)

# R may carry Latin-1 or native-marked CHARSXPs as well as UTF-8 ones.
utf8_cafe <- paste0("caf", intToUtf8(0xe9))
latin1_cafe <- iconv(utf8_cafe, from = "UTF-8", to = "latin1")
stopifnot(!is.na(latin1_cafe))
stopifnot(stride_levenshtein(latin1_cafe, utf8_cafe) == 0)

ascii_bytes <- "plain ASCII"
Encoding(ascii_bytes) <- "bytes"
stopifnot(stride_levenshtein(ascii_bytes, "plain ASCII") == 0)

non_ascii_bytes <- rawToChar(as.raw(c(0xc3, 0xa9)))
Encoding(non_ascii_bytes) <- "bytes"
expect_error(
    stride_levenshtein(non_ascii_bytes, intToUtf8(0xe9)),
    "strings marked as bytes"
)

# The binding treats input vectors as immutable for the entire operation.
held <- c("ASCII", nihao, grinning, NA_character_)
held_before <- serialize(held, NULL)
invisible(stride_jaro_winkler(held, "ASCII"))
stopifnot(identical(serialize(held, NULL), held_before))

expect_error(stride_levenshtein(c("a", "b"), c("a", "b", "c")), "equal lengths")
expect_error(stride_levenshtein(1:3, "a"), "character vectors")
expect_error(stride_hamming("ab", "abc"), "equal-length strings")
expect_error(stride_jaro_winkler("a", "a", prefix_cap = -1), "non-negative integer")
expect_error(stride_smith_waterman("a", "a", match_score = 1.5), "integer")
expect_error(stride_smith_waterman("a", "a", gap_score = NA_real_), "finite")
