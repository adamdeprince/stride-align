\name{substitution_matrices}
\alias{SubstitutionMatrix}
\alias{substitution_matrix_encode}
\alias{substitution_matrix_score}
\alias{substitution_matrix_score_step_limit}
\alias{substitution_matrix_transpose}
\alias{substitution_matrix_from_ncbi_text}
\alias{identity_matrix}
\alias{ascii_matrix}
\alias{blosum30}
\alias{blosum35}
\alias{blosum40}
\alias{blosum45}
\alias{blosum50}
\alias{blosum55}
\alias{blosum60}
\alias{blosum62}
\alias{blosum65}
\alias{blosum70}
\alias{blosum75}
\alias{blosum80}
\alias{blosum85}
\alias{blosum90}
\alias{blosum100}
\alias{pam10}
\alias{pam20}
\alias{pam30}
\alias{pam40}
\alias{pam50}
\alias{pam60}
\alias{pam70}
\alias{pam80}
\alias{pam90}
\alias{pam100}
\alias{pam110}
\alias{pam120}
\alias{pam130}
\alias{pam140}
\alias{pam150}
\alias{pam160}
\alias{pam170}
\alias{pam180}
\alias{pam190}
\alias{pam200}
\alias{pam210}
\alias{pam220}
\alias{pam230}
\alias{pam240}
\alias{pam250}
\alias{pam260}
\alias{pam270}
\alias{pam280}
\alias{pam290}
\alias{pam300}
\alias{pam310}
\alias{pam320}
\alias{pam330}
\alias{pam340}
\alias{pam350}
\alias{pam360}
\alias{pam370}
\alias{pam380}
\alias{pam390}
\alias{pam400}
\alias{pam410}
\alias{pam420}
\alias{pam430}
\alias{pam440}
\alias{pam450}
\alias{pam460}
\alias{pam470}
\alias{pam480}
\alias{pam490}
\alias{pam500}
\alias{nuc44}
\alias{dna_match}
\alias{ascii_text}
\title{Substitution matrices for sequence alignment}
\usage{
SubstitutionMatrix(name, alphabet, matrix, gap_score = -4,
  wildcard = "X", gap_open = NULL, gap_extend = NULL)
substitution_matrix_encode(substitution_matrix, sequence)
substitution_matrix_score(substitution_matrix, query, target)
substitution_matrix_score_step_limit(substitution_matrix,
  gap_score = NULL, gap_open = NULL, gap_extend = NULL)
substitution_matrix_transpose(substitution_matrix,
  name = paste0(substitution_matrix$name, ".T"))
substitution_matrix_from_ncbi_text(text, name = "<unnamed>",
  gap_score = -4, wildcard = "X", gap_open = NULL,
  gap_extend = NULL)
identity_matrix(alphabet, match = 1, mismatch = 0, wildcard = "*",
  name = "IDENTITY", gap_score = -1, gap_open = NULL,
  gap_extend = NULL)
ascii_matrix(match = 1, mismatch = -1, name = "ASCII",
  gap_score = -1, gap_open = NULL, gap_extend = NULL)
}
\arguments{
\item{name}{Matrix name.}
\item{alphabet}{One character string containing each matrix symbol once.}
\item{matrix}{A square numeric matrix of integer substitution scores.}
\item{substitution_matrix}{A matrix created by \code{SubstitutionMatrix()}.}
\item{sequence}{A character vector to encode as zero-based matrix indices.}
\item{query, target}{A query string and one or more target strings to score.}
\item{text}{An NCBI-format substitution matrix in one character string.}
\item{gap_score}{Suggested linear gap score stored with the matrix.}
\item{wildcard}{Alphabet symbol used for unknown input code points.}
\item{gap_open, gap_extend}{Optional suggested affine gap scores.}
\item{match, mismatch}{Identity-matrix diagonal and off-diagonal scores.}
}
\value{
Constructors and transposition return a \code{stride_substitution_matrix}
list. Encoding returns integer indices, scoring returns numeric scores, and
the step-limit helper returns one non-negative numeric value. Catalog names
such as \code{blosum62}, \code{pam250}, \code{nuc44}, \code{dna_match}, and
\code{ascii_text} are lazily loaded matrix objects.
}
\description{
Constructors and the full built-in BLOSUM, PAM, nucleotide, DNA identity, and
ASCII substitution-matrix catalog used by matrix-mode alignment APIs. Matrix
objects also provide \code{$encode()}, \code{$score()}, and
\code{$score_step_limit()} convenience functions.
}
\details{
R character strings cannot represent an embedded NUL as an alphabet member,
so \code{ascii_matrix()} covers code points 1 through 127.
}
