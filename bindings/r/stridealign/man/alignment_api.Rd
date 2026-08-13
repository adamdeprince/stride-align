\name{alignment_api}
\alias{AlignmentPath}
\alias{AlignmentResult}
\alias{Scores}
\alias{smith_waterman_score}
\alias{smith_waterman_normalized_score}
\alias{smith_waterman_scores}
\alias{smith_waterman_normalized_scores}
\alias{smith_waterman_farrar_score}
\alias{smith_waterman_farrar_normalized_score}
\alias{smith_waterman_farrar_scores}
\alias{smith_waterman_farrar_normalized_scores}
\alias{smith_waterman_top_k}
\alias{smith_waterman_best}
\alias{needleman_wunsch_score}
\alias{needleman_wunsch_normalized_score}
\alias{needleman_wunsch_scores}
\alias{needleman_wunsch_normalized_scores}
\alias{smith_waterman_path}
\alias{smith_waterman_path_info}
\alias{smith_waterman_cigar}
\alias{smith_waterman_trace_cigar}
\alias{smith_waterman_trade_cigar}
\alias{needleman_wunsch_path}
\alias{needleman_wunsch_path_info}
\alias{needleman_wunsch_cigar}
\alias{needleman_wunsch_trace_cigar}
\alias{needleman_wunsch_trade_cigar}
\title{Canonical sequence-alignment APIs}
\usage{
smith_waterman_score(query, target, match_score = 2,
  mismatch_score = -1, gap_score = -1, gap_open_score = NULL,
  gap_extend_score = NULL, width = NULL, matrix = NULL)
smith_waterman_scores(query, targets, ...)
smith_waterman_path(query, target, match_score = 2,
  mismatch_score = -1, gap_score = -1, gap_open_score = NULL,
  gap_extend_score = NULL, width = NULL, matrix = NULL)

needleman_wunsch_score(query, target, match_score = 2,
  mismatch_score = -1, gap_score = -1, gap_open_score = NULL,
  gap_extend_score = NULL, width = NULL, matrix = NULL)
needleman_wunsch_scores(query, targets, ...)
needleman_wunsch_path(query, target, match_score = 2,
  mismatch_score = -1, gap_score = -1, gap_open_score = NULL,
  gap_extend_score = NULL, width = NULL, matrix = NULL)

Scores(query, variant = c("smith_waterman", "needleman_wunsch",
  "smith_waterman_farrar", "farrar"), match_score = 2, mismatch_score = -1,
  gap_score = -1, gap_open_score = NULL, gap_extend_score = NULL,
  width = NULL)
}
\arguments{
\item{query, target, targets}{Character inputs. The plural form compares one
query with every target.}
\item{match_score, mismatch_score}{Scores for equal and unequal code points.}
\item{gap_score}{Linear gap score.}
\item{gap_open_score, gap_extend_score}{Optional affine gap scores.}
\item{width}{Optional forced lane width: 0, 8, 16, 32, or 64.}
\item{matrix}{Optional substitution matrix.}
\item{variant}{Alignment engine selected by \code{Scores}.}
\item{...}{Alignment scoring arguments.}
}
\value{
Score functions return numeric vectors. Path functions return an
\code{AlignmentResult}; path-info functions return an \code{AlignmentPath};
CIGAR helpers return a character scalar. Coordinates are zero-based half-open
coordinates, matching the Python path objects.
}
\description{
Smith-Waterman local and Needleman-Wunsch global alignment with scalar,
one-to-many, normalized, traceback, CIGAR, top-k, and persistent-query forms.
}
