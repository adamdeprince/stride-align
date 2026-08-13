source("helpers.R")

# R cannot store U+0000 in a CHARSXP, so this alphabet is the representable
# ASCII range 1..127. Code point i maps to row/column i, preserving the useful
# identity-index convention and DEL wildcard.
stopifnot(
  nchar(keyboard$ASCII_ALPHABET, type = "chars") == 127L,
  identical(keyboard$ASCII_CODEPOINTS, 1:127),
  keyboard$ASCII_WILDCARD == intToUtf8(127)
)
counts <- data.frame(
  typed = c("x", "y", "a", "e"),
  intended = c("y", "x", "b", "r"),
  count = c(90, 10, 50, 30),
  stringsAsFactors = FALSE
)
synthetic <- keyboard$from_confusion_counts(counts, name = "synthetic")
stopifnot(
  inherits(synthetic, "stride_substitution_matrix"),
  synthetic$alphabet == keyboard$ASCII_ALPHABET,
  identical(dim(synthetic$matrix), c(127L, 127L)),
  is.integer(synthetic$matrix),
  synthetic$wildcard == intToUtf8(127),
  synthetic$name == "synthetic"
)
symbols <- strsplit(synthetic$alphabet, "", fixed = TRUE)[[1L]]
cell <- function(matrix, typed, intended) {
  matrix$matrix[match(typed, symbols), match(intended, symbols)]
}
stopifnot(cell(synthetic, "x", "y") != cell(synthetic, "y", "x"))
off_diagonal <- synthetic$matrix[row(synthetic$matrix) != col(synthetic$matrix)]
stopifnot(
  length(unique(diag(synthetic$matrix))) == 1L,
  diag(synthetic$matrix)[[1L]] > max(off_diagonal)
)
reversed <- substitution_matrix_transpose(synthetic, name = "rev")
stopifnot(
  cell(reversed, "y", "x") == cell(synthetic, "x", "y"),
  identical(reversed$matrix, t(synthetic$matrix))
)

grid <- matrix(0, 127L, 127L)
grid[utf8ToInt("x"), utf8ToInt("y")] <- 90
grid[utf8ToInt("y"), utf8ToInt("x")] <- 10
from_grid <- keyboard$from_confusion_counts(grid)
stopifnot(identical(dim(from_grid$matrix), c(127L, 127L)))
expect_error(
  keyboard$from_confusion_counts(matrix(0, 10L, 10L)),
  "shape does not match"
)

path <- system.file("keyboard_data", "qwerty.npy", package = "stridealign")
stopifnot(nzchar(path), file.exists(path))
loaded <- keyboard$from_npy(path, name = "qwerty-copy")
stopifnot(
  loaded$name == "qwerty-copy",
  identical(loaded$matrix, keyboard$qwerty$matrix),
  is.integer(loaded$matrix)
)
transposed <- keyboard$from_npy(path, transpose = TRUE)
stopifnot(identical(transposed$matrix, t(loaded$matrix)))
expect_error(get("nonexistent_layout_xyz", envir = keyboard), "not found")
stopifnot(is.character(keyboard$available()), "qwerty" %in% keyboard$available())

self_score <- needleman_wunsch_score("hello", "hello", matrix = synthetic)
stopifnot(self_score == diag(synthetic$matrix)[[1L]] * nchar("hello"))
for (name in keyboard$available()) {
  matrix <- keyboard[[name]]
  off <- matrix$matrix[row(matrix$matrix) != col(matrix$matrix)]
  stopifnot(
    inherits(matrix, "stride_substitution_matrix"),
    identical(dim(matrix$matrix), c(127L, 127L)),
    diag(matrix$matrix)[[1L]] > max(off)
  )
}
qwerty_symbols <- strsplit(keyboard$qwerty$alphabet, "", fixed = TRUE)[[1L]]
qwerty_cell <- function(left, right) keyboard$qwerty$matrix[
  match(left, qwerty_symbols), match(right, qwerty_symbols)
]
stopifnot(
  qwerty_cell("n", "m") > qwerty_cell("z", "m"),
  qwerty_cell("r", "t") > qwerty_cell("z", "m")
)

# Python cases: test_ascii_alphabet_and_wildcard,
# test_from_confusion_counts_builds_valid_matrix,
# test_orientation_is_asymmetric_typed_then_intended,
# test_identity_diagonal_dominates, test_transpose_reverses_orientation,
# test_from_confusion_counts_accepts_array,
# test_from_confusion_counts_rejects_wrong_array_shape,
# test_from_npy_round_trip, test_from_npy_transpose,
# test_lazy_unknown_matrix_raises, test_available_returns_list,
# test_matrix_flows_through_kernel,
# test_bundled_matrices_present_and_valid,
# test_bundled_qwerty_captures_keyboard_adjacency.
