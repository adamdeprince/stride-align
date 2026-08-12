\name{stride_smith_waterman}
\alias{stride_smith_waterman}
\alias{stride_needleman_wunsch}
\alias{stride_smith_waterman_affine}
\alias{stride_needleman_wunsch_affine}
\title{Sequence-alignment scores}
\usage{
stride_smith_waterman(query, target, match_score = 2,
  mismatch_score = -1, gap_score = -1)
stride_needleman_wunsch(query, target, match_score = 2,
  mismatch_score = -1, gap_score = -1)
stride_smith_waterman_affine(query, target, match_score = 2,
  mismatch_score = -1, gap_open_score = -2, gap_extend_score = -1)
stride_needleman_wunsch_affine(query, target, match_score = 2,
  mismatch_score = -1, gap_open_score = -2, gap_extend_score = -1)
}
\arguments{
\item{query, target}{Character vectors. One length-one input is broadcast.}
\item{match_score}{Score for a matching code point.}
\item{mismatch_score}{Score for a mismatching code point.}
\item{gap_score}{Linear gap score.}
\item{gap_open_score}{Score for opening a gap.}
\item{gap_extend_score}{Score for extending a gap.}
}
\value{A numeric vector of alignment scores.}
\description{
Computes Smith-Waterman local or Needleman-Wunsch global alignment scores with
linear or affine gaps.
}
