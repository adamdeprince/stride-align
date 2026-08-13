source("helpers.R")

matrix_cell <- function(matrix, left, right) {
  symbols <- strsplit(matrix$alphabet, "", fixed = TRUE)[[1L]]
  matrix$matrix[match(left, symbols), match(right, symbols)]
}
diagonal_sum <- function(sequence, matrix) sum(vapply(
  strsplit(sequence, "", fixed = TRUE)[[1L]],
  function(symbol) matrix_cell(matrix, symbol, symbol),
  numeric(1)
))

stopifnot(
  blosum62$name == "BLOSUM62", blosum62$stride == 24L,
  identical(dim(blosum62$matrix), c(24L, 24L)),
  is.integer(blosum62$matrix), identical(blosum62$matrix, t(blosum62$matrix)),
  matrix_cell(blosum62, "A", "A") == 4,
  matrix_cell(blosum62, "W", "W") == 11,
  matrix_cell(blosum62, "A", "R") == -1
)
protein <- "ACDEFGHIKLMNPQRSTVWY"
expected_encoding <- match(
  strsplit(protein, "", fixed = TRUE)[[1L]],
  strsplit(blosum62$alphabet, "", fixed = TRUE)[[1L]]
) - 1L
stopifnot(identical(substitution_matrix_encode(blosum62, protein), expected_encoding))
wildcard <- match("X", strsplit(blosum62$alphabet, "", fixed = TRUE)[[1L]]) - 1L
stopifnot(
  identical(substitution_matrix_encode(blosum62, "acdef"), rep(wildcard, 5L)),
  !identical(
    substitution_matrix_encode(blosum62, "ACDEF"),
    substitution_matrix_encode(blosum62, "acdef")
  ),
  identical(
    substitution_matrix_encode(blosum62, toupper("acdef")),
    substitution_matrix_encode(blosum62, "ACDEF")
  ),
  identical(substitution_matrix_encode(blosum62, "J?@"), rep(wildcard, 3L)),
  length(substitution_matrix_encode(blosum62, "")) == 0L
)
text_alphabet <- paste0(c(letters, LETTERS, " ", "?"), collapse = "")
case_matrix <- SubstitutionMatrix(
  "case-sensitive", text_alphabet, diag(length(strsplit(text_alphabet, "")[[1L]])),
  gap_score = -1, wildcard = "?"
)
stopifnot(!identical(
  substitution_matrix_encode(case_matrix, "abc"),
  substitution_matrix_encode(case_matrix, "ABC")
))

expected <- diagonal_sum(protein, blosum62)
stopifnot(
  smith_waterman_score(protein, protein, matrix = blosum62, gap_score = -4) == expected,
  needleman_wunsch_score(protein, protein, matrix = blosum62, gap_score = -4) == expected,
  smith_waterman_score("HE", "HE", matrix = blosum62, gap_score = -4) == 13
)
expect_error(smith_waterman_score("HE", "HE", matrix = blosum62, match_score = 5), "cannot be used with matrix")
expect_error(needleman_wunsch_score("HE", "HE", matrix = blosum62, mismatch_score = -3), "cannot be used with matrix")
stopifnot(smith_waterman_score(
  "HE", "HE", matrix = blosum62, gap_open_score = -11, gap_extend_score = -1
) == 13)
stopifnot(smith_waterman_score(
  "HEAGAW", "HEW", matrix = blosum62, gap_open_score = -1, gap_extend_score = -1
) > smith_waterman_score(
  "HEAGAW", "HEW", matrix = blosum62, gap_open_score = -100, gap_extend_score = -1
))

catalog <- mget(c(
  paste0("blosum", c(30, 35, 40, 45, 50, 55, 60, 62, 65, 70, 75, 80, 85, 90, 100)),
  paste0("pam", seq(10, 500, 10))
), envir = asNamespace("stridealign"))
stopifnot(length(catalog) == 65L)
for (matrix in catalog) stopifnot(
  matrix$alphabet == "ARNDCQEGHILKMFPSTWYVBZX*",
  identical(dim(matrix$matrix), c(24L, 24L)),
  is.integer(matrix$matrix), identical(matrix$matrix, t(matrix$matrix)),
  matrix$wildcard == "X", matrix$gap_score < 0,
  matrix$gap_open < 0, matrix$gap_extend < 0
)
stopifnot(
  matrix_cell(pam250, "W", "W") == 17,
  matrix_cell(pam250, "C", "C") == 12,
  matrix_cell(blosum90, "W", "W") == 11,
  matrix_cell(blosum90, "H", "H") == 8
)

tiny_text <- paste(
  "# comment line", "A B C", "A 4 -1 -2", "B -1 5 0", "C -2 0 9",
  sep = "\n"
)
tiny <- substitution_matrix_from_ncbi_text(
  tiny_text, name = "tiny", wildcard = "C", gap_open = -5, gap_extend = -1
)
stopifnot(
  tiny$alphabet == "ABC", identical(dim(tiny$matrix), c(3L, 3L)),
  tiny$matrix[1L, 2L] == -1, tiny$matrix[3L, 3L] == 9,
  tiny$gap_open == -5, tiny$gap_extend == -1
)
expect_error(substitution_matrix_from_ncbi_text(
  paste("A B", "A 1 2", "X 3 4", sep = "\n"), wildcard = "A"
), "row labels do not match")

queries <- c("HEAGAWGHEE", "PAWHEAE")
targets <- c("HE", "AWGHE", "PAWHEAE")
sw <- cdist(
  queries, targets, matrix = blosum62, scorer = "sw",
  gap_open_score = -11, gap_extend_score = -1
)
nw <- cdist(
  queries, targets, matrix = blosum62, scorer = "nw",
  gap_open_score = -11, gap_extend_score = -1
)
stopifnot(
  identical(dim(sw), c(2L, 3L)), identical(dim(nw), c(2L, 3L)),
  sw[2L, 3L] == smith_waterman_score(
    "PAWHEAE", "PAWHEAE", matrix = blosum62,
    gap_open_score = -11, gap_extend_score = -1
  ),
  nw[2L, 3L] == needleman_wunsch_score(
    "PAWHEAE", "PAWHEAE", matrix = blosum62,
    gap_open_score = -11, gap_extend_score = -1
  )
)
stopifnot(identical(
  cdist("HE", "HE", matrix = blosum62, scorer = smith_waterman_scores),
  cdist("HE", "HE", matrix = blosum62, scorer = Scorer$SMITH_WATERMAN)
))
top <- cdist_top_k(
  c(queries, "MEEPS", "WW"), c(targets, "XQQQ"), matrix = blosum62,
  scorer = "sw", k = 3, gap_open_score = -11, gap_extend_score = -1
)
stopifnot(nrow(top) == 3L, top$query[[1L]] == "PAWHEAE", top$target[[1L]] == "PAWHEAE")
filtered <- cdist_above_threshold(
  queries, c("AWGHE", "PAWHEAE", "XXX"), matrix = blosum62,
  scorer = "sw", threshold = 15,
  gap_open_score = -11, gap_extend_score = -1
)
stopifnot(all(filtered$score >= 15), any(filtered$query == "PAWHEAE" & filtered$target == "PAWHEAE"))
expect_error(cdist("A", "A", matrix = blosum62, scorer = "hamming"), "requires a Smith-Waterman")

path <- smith_waterman_path(
  "HE", "HE", matrix = blosum62, gap_open_score = -11, gap_extend_score = -1
)
stopifnot(
  path$score == 13, identical(unname(path$aligned), c("HE", "HE")),
  path$operations == "==", path$query_start == 0, path$query_end == 2
)
gapped <- smith_waterman_path(
  "HEAGAWGHEE", "PAWHEAE", matrix = blosum62,
  gap_open_score = -11, gap_extend_score = -1
)
left <- strsplit(gapped$aligned[[1L]], "", fixed = TRUE)[[1L]]
right <- strsplit(gapped$aligned[[2L]], "", fixed = TRUE)[[1L]]
operations <- strsplit(gapped$operations, "", fixed = TRUE)[[1L]]
for (index in seq_along(operations)) {
  operation <- operations[[index]]
  if (operation == "=") stopifnot(left[[index]] == right[[index]])
  if (operation == "X") stopifnot(left[[index]] != right[[index]], left[[index]] != "-", right[[index]] != "-")
  if (operation == "D") stopifnot(right[[index]] == "-")
  if (operation == "I") stopifnot(left[[index]] == "-")
}
info <- smith_waterman_path_info("HE", "HE", matrix = blosum62)
stopifnot(info$score == 13, info$cigar == "2=", info$matches == 2, info$aligned_length == 2)
stopifnot(smith_waterman_cigar("HE", "HE", matrix = blosum62) == "2=")
global_path <- needleman_wunsch_path("HE", "HEW", matrix = blosum62)
stopifnot(
  global_path$query_start == 0, global_path$query_end == 2,
  global_path$target_start == 0, global_path$target_end == 3,
  nchar(global_path$aligned[[1L]]) == 3, nchar(global_path$aligned[[2L]]) == 3,
  grepl("-", global_path$aligned[[1L]], fixed = TRUE)
)

matrix_batch_targets <- c("PAWHEAE", "HEAGAWGHEE", "WW", "")
stopifnot(identical(
  smith_waterman_scores(
    "HEAGAWGHEE", matrix_batch_targets, matrix = blosum62,
    gap_open_score = -11, gap_extend_score = -1
  ),
  smith_waterman_score(
    "HEAGAWGHEE", matrix_batch_targets, matrix = blosum62,
    gap_open_score = -11, gap_extend_score = -1
  )
))
expect_error(smith_waterman_score("HE", "HE", matrix = matrix(0, 4, 4)), "SubstitutionMatrix")
stopifnot(
  smith_waterman_score("HELLO", "HELLO") == 10,
  smith_waterman_score("HELLO", "WORLD") == 2,
  needleman_wunsch_score("HELLO", "HELLO") == 10,
  substitution_matrix_score(blosum62, "HE", "HE") == 13,
  blosum62$score("HE", "HE") == 13
)
custom <- SubstitutionMatrix(
  "ABonly", "AB", matrix(c(3, -2, -2, 3), 2L, byrow = TRUE),
  gap_score = -1, wildcard = "A"
)
stopifnot(
  smith_waterman_score("AABB", "ABAB", matrix = custom) ==
    smith_waterman_score("AABB", "ABAB", match_score = 3, mismatch_score = -2)
)
expect_error(SubstitutionMatrix("bad", "ABC", matrix(0, 2, 2), wildcard = "A"), "dimensions")
expect_error(SubstitutionMatrix("bad", "AB", matrix(0, 2, 2), wildcard = "X"), "wildcard")
expect_error(SubstitutionMatrix("bad", "AB", matrix(200, 2, 2), wildcard = "A"), "signed 8-bit")

stopifnot(
  blosum62$max_abs == 11, blosum45$max_abs == 15, pam250$max_abs == 17
)
big <- SubstitutionMatrix(
  "big", "ACGTX", diag(c(100, 100, 100, 100, 0)), -10, "X"
)
small <- SubstitutionMatrix("small", "ACGTX", diag(5), -1, "X")
negative <- SubstitutionMatrix("negative", "AX", matrix(c(1, -127, -127, 0), 2, byrow = TRUE), -1, "X")
stopifnot(
  big$max_abs == 100, small$max_abs == 1, negative$max_abs == 127,
  blosum62$score_step_limit() == 11,
  blosum62$score_step_limit(gap_score = -20) == 20,
  blosum62$score_step_limit(gap_score = -4) == 11,
  blosum62$score_step_limit(gap_open = -50, gap_extend = -2) == 50,
  smith_waterman_score(strrep("A", 5), strrep("A", 5), matrix = big) == 500,
  smith_waterman_score(strrep("A", 400), strrep("A", 400), matrix = big) == 40000,
  needleman_wunsch_score(
    strrep("A", 400), strrep("A", 400), matrix = big,
    gap_open_score = -15, gap_extend_score = -3
  ) == 40000
)
stopifnot(
  identical(blosum62$matrix_bytes, blosum62$matrix_bytes),
  length(blosum62$matrix_bytes) == blosum62$stride^2,
  !identical(
    SubstitutionMatrix("a", "AX", diag(2), -1, "X")$matrix_bytes,
    SubstitutionMatrix("b", "AX", diag(c(2, 2)), -1, "X")$matrix_bytes
  )
)

stopifnot(
  matrix_cell(blosum30, "A", "A") == 4,
  matrix_cell(blosum30, "X", "X") == -1,
  matrix_cell(blosum40, "C", "C") == 16,
  matrix_cell(blosum100, "W", "W") == 17,
  matrix_cell(pam10, "A", "A") == 7,
  matrix_cell(pam20, "A", "A") == 6,
  matrix_cell(pam120, "R", "R") == 6,
  matrix_cell(pam400, "W", "W") == 26,
  matrix_cell(pam500, "W", "W") == 34,
  nuc44$name == "NUC.4.4", nuc44$alphabet == "ATGCSWRYKMBVHDN",
  nuc44$stride == 15, nuc44$wildcard == "N",
  identical(nuc44$matrix, t(nuc44$matrix)),
  matrix_cell(nuc44, "A", "A") == 5,
  matrix_cell(nuc44, "A", "T") == -4,
  matrix_cell(nuc44, "A", "N") == -2,
  nuc44$max_abs == 5
)
identity <- identity_matrix("XYZ", match = 3, mismatch = -2, wildcard = "?")
stopifnot(
  identity$alphabet == "XYZ?", identity$wildcard == "?",
  matrix_cell(identity, "X", "X") == 3,
  matrix_cell(identity, "X", "Y") == -2,
  matrix_cell(identity, "?", "?") == -2,
  identity_matrix("AB*", wildcard = "*")$stride == 3L
)
stopifnot(
  ascii_text$stride == 127L,
  matrix_cell(ascii_text, "a", "a") == 1,
  matrix_cell(ascii_text, "a", "A") == -1,
  matrix_cell(ascii_text, intToUtf8(127), intToUtf8(127)) == -1,
  dna_match$alphabet == "ACGTN", dna_match$wildcard == "N",
  matrix_cell(dna_match, "A", "A") == 5,
  matrix_cell(dna_match, "A", "C") == -4,
  matrix_cell(dna_match, "N", "N") == -4,
  identical(substitution_matrix_encode(dna_match, "ACGTN-x"), c(0L, 1L, 2L, 3L, 4L, 4L, 4L))
)
for (matrix in list(blosum30, pam120)) stopifnot(
  needleman_wunsch_score("MKTAYIAKQR", "MKTAYIAKQR", matrix = matrix) ==
    diagonal_sum("MKTAYIAKQR", matrix)
)
stopifnot(
  needleman_wunsch_score("ACGTACGTAC", "ACGTACGTAC", matrix = nuc44) == diagonal_sum("ACGTACGTAC", nuc44),
  needleman_wunsch_score("Hello, World!", "Hello, World!", matrix = ascii_text) == diagonal_sum("Hello, World!", ascii_text),
  smith_waterman_score("Hello", "Hello", matrix = ascii_text) == 5,
  smith_waterman_score("Hello", "hello", matrix = ascii_text) == 4
)

# Python cases: all 64 tests in tests/test_matrices.py. NumPy dtype/layout
# cases are translated to R's numeric-matrix, signed-int8-range, stable raw
# snapshot, and column-major representation contracts; the NUL row of the
# ASCII matrix is omitted because CHARSXP cannot represent U+0000.
