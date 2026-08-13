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

# LCS, n-gram, Ratcliff-Obershelp, token-ratio, and Monge-Elkan families.
stopifnot(
    lcs_length("ABCBDAB", "BDCAB") == 4,
    lcs_substring_length("alpha beta", "theta beta") == 6,
    identical(lcs_substring("alpha beta", "theta beta"), "a beta"),
    lcs_substring(intToUtf8(0x3b1:0x3b5), intToUtf8(0x3b2:0x3b4)) == intToUtf8(0x3b2:0x3b4)
)
close_to(jaccard("hello", "help", n = 3), 0.25)
close_to(jaccard("aaaa", "aaa"), 2 / 3)
close_to(dice("aaaa", "aaa"), 0.8)
close_to(cosine("aaaa", "aaa"), 1)
close_to(overlap("aaaa", "aaa"), 1)

ngram_targets <- c("help", "hello", "yellow", NA_character_)
for (family in list(
    list(jaccard, jaccard_similarities),
    list(dice, dice_similarities),
    list(cosine, cosine_similarities),
    list(overlap, overlap_similarities)
)) {
    close_to(
        family[[2L]]("hello", ngram_targets, n = 2),
        family[[1L]]("hello", ngram_targets, n = 2)
    )
}
close_to(
    ratcliff_obershelp_similarities("hello", c("hello", "yellow", "xyz")),
    ratcliff_obershelp_similarity("hello", c("hello", "yellow", "xyz"))
)
stopifnot(
    partial_ratio("apple", "an apple a day") == 1,
    token_sort_ratio("hello world", "world hello") == 1,
    token_set_ratio("foo bar", "foo bar baz") == 1,
    partial_token_sort_ratio("new york", "york new city") == 1,
    partial_token_set_ratio("new york", "new york city") == 1,
    WRatio("kitten", "kitten") == 1,
    isTRUE(all.equal(
        WRatio("new york mets", "new york yankees"),
        0.9025,
        tolerance = 1e-12
    )),
    token_sort_ratio("NEW YORK", "new york", processor = tolower) == 1,
    monge_elkan("paul", "paul johnson") == 1,
    monge_elkan("paul johnson", "paul") == 0.5,
    monge_elkan("PAUL JOHNSON", "paul johnson", processor = tolower) == 1,
    monge_elkan("foo bar", "bar foo", inner = function(a, b) as.numeric(a == b)) == 1
)

# Dynamic Time Warping preserves R numeric storage type when choosing the
# Python-compatible default metric, and its list batch matches singular calls.
integer_query <- c(0L, 2L)
integer_target <- c(0L, 0L)
double_query <- as.double(integer_query)
double_target <- as.double(integer_target)
stopifnot(
    dtw(integer_query, integer_target) == 2,
    dtw(double_query, double_target) == 4,
    dtw(integer_query, integer_target) == dtw(integer_query, integer_target, distance = "l1"),
    dtw(double_query, double_target) == dtw(double_query, double_target, distance = "l2_squared")
)
dtw_targets <- list(c(0, 0), c(0, 1), c(0, 2), c(0, 3))
close_to(
    dtw_distances(c(0, 2), dtw_targets, window = 1L),
    vapply(dtw_targets, function(target) dtw(c(0, 2), target, window = 1L), numeric(1))
)
stopifnot(
    length(dtw_distances(c(1, 2), list())) == 0L,
    is.infinite(dtw(c(0, 1, 2), c(10, 11, 12), score_cutoff = 1))
)
expect_error(dtw(1:3, as.double(1:3)), "share numeric storage type")
expect_error(
    dtw_distances(1:3, list(1:3, as.double(1:3))),
    "share the query's numeric storage type"
)

# Alignment score facades, tracebacks, and CIGAR forms.
scores <- Scores("ACCGT")
stopifnot(identical(scores$compare(c("CCG", "ACCGT")), c(6, 10)))
stopifnot(
    smith_waterman_farrar_score("ACCGT", "CCG") == smith_waterman_score("ACCGT", "CCG"),
    identical(
        smith_waterman_farrar_scores("ACCGT", c("CCG", "ACCGT")),
        smith_waterman_scores("ACCGT", c("CCG", "ACCGT"))
    )
)
sw_path <- smith_waterman_path("ACCGT", "CCG")
stopifnot(
    inherits(sw_path, "stride_alignment_result"),
    sw_path$score == 6,
    sw_path$cigar == "3=",
    identical(unname(sw_path$aligned), c("CCG", "CCG")),
    smith_waterman_cigar("ACCGT", "CCG") == "3=",
    smith_waterman_trace_cigar("ACCGT", "CCG") == "3=",
    smith_waterman_trade_cigar("ACCGT", "CCG") == "3="
)
nw_path <- needleman_wunsch_path("HE", "HEW", matrix = blosum62)
stopifnot(
    inherits(nw_path, "stride_alignment_result"),
    nw_path$score == 12,
    needleman_wunsch_cigar("ABC", "ABC") == "3=",
    needleman_wunsch_trace_cigar("ABC", "ABC") == "3=",
    needleman_wunsch_trade_cigar("ABC", "ABC") == "3="
)
stopifnot(inherits(smith_waterman_path_info("ACCGT", "CCG"), "stride_alignment_path"))

# User-defined and built-in substitution matrices.
custom <- identity_matrix("AB", match = 3, mismatch = -2, wildcard = "?")
stopifnot(
    inherits(custom, "stride_substitution_matrix"),
    custom$matrix[1L, 1L] == 3,
    custom$matrix[1L, 2L] == -2,
    custom$matrix[3L, 3L] == -2,
    smith_waterman_score("AB", "AB", matrix = custom) == 6,
    needleman_wunsch_score(c("AB", "AA"), "AB", matrix = custom) == c(6, 1)
)
stopifnot(
    inherits(ascii_matrix(), "stride_substitution_matrix"),
    smith_waterman_score("Hello", "Hello", matrix = ascii_text) == 5,
    smith_waterman_score("HE", "HE", matrix = blosum62, gap_score = -4) == 13,
    needleman_wunsch_score("ACGT", "ACGT", matrix = dna_match) == 20
)
catalog_names <- c(
    paste0("blosum", c(30, 35, 40, 45, 50, 55, 60, 62, 65, 70, 75, 80, 85, 90, 100)),
    paste0("pam", seq(10, 500, 10)),
    "nuc44", "dna_match", "ascii_text"
)
for (name in catalog_names) {
    value <- getExportedValue("stridealign", name)
    stopifnot(
        inherits(value, "stride_substitution_matrix"),
        nrow(value$matrix) == ncol(value$matrix)
    )
}
expect_error(
    smith_waterman_score("A", "A", matrix = custom, match_score = 5),
    "cannot be used with matrix"
)

# Phonetic functions are vectorized over ordinary R character vectors.
stopifnot(
    identical(soundex(c("Robert", "Rupert", NA_character_)), c("R163", "R163", NA_character_)),
    soundex_equal("Robert", "Rupert"),
    metaphone("Robert") == "RBRT",
    metaphone_equal("Catherine", "Kathryn"),
    nysiis("Robert") == "RABAD",
    nysiis_equal("Catherine", "Catharine"),
    match_rating_codex("Aether") == "ATHR",
    match_rating_compare("Robert", "Rupert"),
    caverphone("Stevenson") == "STFNSN1111",
    cologne_phonetic("Müller") == "657",
    nchar(daitch_mokotoff("Washington")) >= 6L
)
metaphones <- double_metaphone(c("Smith", "Robert"))
stopifnot(
    is.data.frame(metaphones),
    identical(metaphones$primary[[1L]], "SM0"),
    identical(metaphones$alternate[[1L]], "XMT")
)
stopifnot(
    beider_morse("") == "",
    nzchar(beider_morse("Schneider")),
    identical(
        beider_morse("Schneider"),
        beider_morse("Schneider", rule_type = BmpmRuleType$APPROX)
    )
)

# Backend inspection mirrors the Python facade using R lists and strings.
records <- available_backends()
stopifnot(
    length(records) >= 1L,
    all(vapply(records, function(record) inherits(record, "stride_backend_record"), logical(1))),
    all(vapply(records, function(record) identical(record$name, record$kind), logical(1))),
    backend_is_available(detect_best_backend()),
    BackendKind$GENERIC == "generic",
    BackendKind$X86_AVX10_512 == "avx10_512"
)
