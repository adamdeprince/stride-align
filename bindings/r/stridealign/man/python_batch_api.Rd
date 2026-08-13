\name{python_batch_api}
\alias{Scorer}
\alias{LevenshteinScorer}
\alias{levenshtein_score}
\alias{levenshtein_normalized_score}
\alias{levenshtein_scores}
\alias{levenshtein_normalized_scores}
\alias{levenshtein_top_k}
\alias{levenshtein_normalized_top_k}
\alias{levenshtein_best}
\alias{levenshtein_normalized_best}
\alias{damerau_levenshtein_score}
\alias{damerau_levenshtein_normalized_score}
\alias{damerau_levenshtein_scores}
\alias{damerau_levenshtein_normalized_scores}
\alias{damerau_levenshtein_top_k}
\alias{damerau_levenshtein_normalized_top_k}
\alias{damerau_levenshtein_best}
\alias{damerau_levenshtein_normalized_best}
\alias{true_damerau_levenshtein_score}
\alias{true_damerau_levenshtein_normalized_score}
\alias{true_damerau_levenshtein_scores}
\alias{true_damerau_levenshtein_normalized_scores}
\alias{true_damerau_levenshtein_top_k}
\alias{true_damerau_levenshtein_normalized_top_k}
\alias{true_damerau_levenshtein_best}
\alias{true_damerau_levenshtein_normalized_best}
\alias{indel_score}
\alias{indel_normalized_score}
\alias{indel_scores}
\alias{indel_normalized_scores}
\alias{indel_top_k}
\alias{indel_normalized_top_k}
\alias{indel_best}
\alias{indel_normalized_best}
\alias{hamming_score}
\alias{hamming_normalized_score}
\alias{hamming_scores}
\alias{hamming_normalized_scores}
\alias{hamming_top_k}
\alias{hamming_normalized_top_k}
\alias{hamming_best}
\alias{hamming_normalized_best}
\alias{jaro_similarity}
\alias{jaro_similarities}
\alias{jaro_top_k}
\alias{jaro_best}
\alias{jaro_winkler_similarity}
\alias{jaro_winkler_similarities}
\alias{jaro_winkler_top_k}
\alias{jaro_winkler_best}
\alias{extract}
\alias{extract_best}
\alias{cdist}
\alias{cdist_above_threshold}
\alias{cdist_top_k}
\alias{cdist_top_k_per_query}
\title{Python-style pair, batch, selection, and cdist APIs}
\usage{
levenshtein_score(query, target, score_cutoff = NULL)
levenshtein_scores(query, targets, score_cutoff = NULL)
levenshtein_top_k(query, targets, k = 5)
levenshtein_best(query, targets)

extract(query, targets, scorer, k = 5, ...)
extract_best(query, targets, scorer, ...)

cdist(queries, targets, scorer = NULL, matrix = NULL,
  match_score = 2, mismatch_score = -1, gap_score = -1,
  gap_open_score = NULL, gap_extend_score = NULL, width = NULL,
  tqdm = NULL, cpu_count = 0, prefix_weight = 0.1,
  prefix_threshold = 0.7, prefix_cap = 4)
cdist_above_threshold(queries, targets, scorer = NULL, threshold,
  matrix = NULL, ...)
cdist_top_k(queries, targets, scorer = NULL, k, matrix = NULL,
  reject_duplicates = FALSE, ...)
cdist_top_k_per_query(queries, targets, scorer, k = 5,
  pruning = FALSE, cpu_count = 0, prefix_weight = 0.1,
  prefix_threshold = 0.7, prefix_cap = 4)

LevenshteinScorer(query)
}
\arguments{
\item{query, target}{Character vectors. Pair functions compare equal-length
vectors elementwise and broadcast a length-one input.}
\item{queries, targets}{Character vectors containing query and target sets.}
\item{score_cutoff}{Optional early-rejection cutoff, with the same returned
score convention as the Python API.}
\item{k}{Number of selected matches.}
\item{scorer}{A value from \code{Scorer}, a scorer name, or a supported batch
function.}
\item{matrix}{An optional \code{SubstitutionMatrix}. Matrix mode defaults to
Smith-Waterman.}
\item{threshold}{Minimum normalized similarity, or minimum raw score in
substitution-matrix mode.}
\item{match_score, mismatch_score, gap_score, gap_open_score,
gap_extend_score, width}{Sequence-alignment scoring controls.}
\item{reject_duplicates}{Whether exact string duplicates are omitted from a
global normalized-similarity top-k result.}
\item{prefix_weight, prefix_threshold, prefix_cap}{Jaro-Winkler controls.}
\item{tqdm, cpu_count, pruning}{Python-parity controls accepted by the R
surface. R dispatch is synchronous and does not use a Python progress object.}
\item{...}{Scorer-specific arguments.}
}
\value{
Pair and one-to-many functions return numeric vectors. \code{cdist()} returns
a numeric matrix. Selection and filtered-cdist functions return data frames;
\code{*_best()} returns a named list or \code{NULL}. All result indices are
one-based R indices. \code{LevenshteinScorer()} returns a list of closures for
a fixed query.
}
\description{
Canonical stride-align APIs corresponding to the package's Python pair,
one-to-many, top-k, best-match, persistent-scorer, and all-pairs operations.
The complete sixteen-value \code{Scorer} catalog is supported by
\code{cdist()}.
}
\details{
The R package exposes stride-align's native API. It intentionally does not
provide the RapidFuzz, Parasail, Jellyfish, or TheFuzz compatibility layers.
}
