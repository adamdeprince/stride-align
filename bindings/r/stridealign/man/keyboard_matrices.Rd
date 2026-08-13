\name{keyboard_matrices}
\alias{keyboard}
\alias{keyboard_from_confusion_counts}
\alias{keyboard_from_npy}
\alias{keyboard_available}
\title{Keyboard-aware substitution matrices}
\usage{
keyboard_from_confusion_counts(counts,
  alphabet = keyboard$ASCII_ALPHABET, name = "KEYBOARD", scale = 2,
  match_margin = 4, floor = NULL,
  wildcard = substr(alphabet, nchar(alphabet), nchar(alphabet)),
  gap_score = -1, gap_open = NULL, gap_extend = NULL)
keyboard_from_npy(path, name = tools::file_path_sans_ext(basename(path)),
  alphabet = keyboard$ASCII_ALPHABET,
  wildcard = keyboard$ASCII_WILDCARD, transpose = FALSE,
  gap_score = -1, gap_open = NULL, gap_extend = NULL)
keyboard_available()
keyboard
}
\arguments{
\item{counts}{A non-negative numeric square matrix, or a data frame with
\code{typed}, \code{intended}, and \code{count} columns.}
\item{alphabet}{One character string containing each matrix symbol once.}
\item{name}{Matrix name.}
\item{scale}{Log-odds score multiplier.}
\item{match_margin}{Amount by which diagonal matches exceed the best
substitution.}
\item{floor}{Optional score used for unobserved pairs.}
\item{wildcard}{Alphabet symbol used for unknown input code points.}
\item{gap_score}{Suggested linear gap score stored with the matrix.}
\item{gap_open, gap_extend}{Optional suggested affine gap scores.}
\item{path}{Path to a C-order, signed-int8, two-dimensional NumPy
\file{.npy} matrix.}
\item{transpose}{Whether to transpose the loaded grid.}
}
\value{
The constructors return a \code{stride_substitution_matrix}.
\code{keyboard_available()} returns bundled matrix names. \code{keyboard} is
an environment containing the constructors, ASCII constants, availability
function, and bundled matrices such as \code{keyboard$qwerty}.
}
\description{
Build, load, discover, and use keyboard-error substitution matrices with the
alignment APIs.
}
\details{
R character strings cannot contain U+0000, so the R keyboard alphabet covers
the representable ASCII code points 1 through 127. DEL is the wildcard slot.
Bundled 128 by 128 artifacts are narrowed automatically when loaded.
}
