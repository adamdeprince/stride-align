\name{string_algorithms}
\alias{lcs_length}
\alias{lcs_substring_length}
\alias{lcs_substring}
\alias{jaccard}
\alias{jaccard_similarities}
\alias{dice}
\alias{dice_similarities}
\alias{cosine}
\alias{cosine_similarities}
\alias{overlap}
\alias{overlap_similarities}
\alias{ratcliff_obershelp_similarity}
\alias{ratcliff_obershelp_similarities}
\alias{partial_ratio}
\alias{token_sort_ratio}
\alias{token_set_ratio}
\alias{partial_token_sort_ratio}
\alias{partial_token_set_ratio}
\alias{WRatio}
\alias{monge_elkan}
\title{Additional native string similarities}
\usage{
lcs_length(query, target)
lcs_substring_length(query, target)
lcs_substring(query, target)

jaccard(query, target, n = 2)
jaccard_similarities(query, targets, n = 2)
dice(query, target, n = 2)
cosine(query, target, n = 2)
overlap(query, target, n = 2)

ratcliff_obershelp_similarity(query, target)
ratcliff_obershelp_similarities(query, targets)

partial_ratio(query, target, processor = NULL)
token_sort_ratio(query, target, processor = NULL)
token_set_ratio(query, target, processor = NULL)
partial_token_sort_ratio(query, target, processor = NULL)
partial_token_set_ratio(query, target, processor = NULL)
WRatio(query, target, processor = NULL)

monge_elkan(query, target, inner = "jaro", processor = NULL,
  symmetric = FALSE)
}
\arguments{
\item{query, target}{Character vectors.}
\item{targets}{A character vector for one-to-many scoring.}
\item{n}{Positive n-gram width.}
\item{processor}{An optional R function applied to each input string first.}
\item{inner}{A Monge-Elkan inner scorer name or R function accepting two
strings and returning one numeric similarity.}
\item{symmetric}{Whether to average both Monge-Elkan directions.}
}
\value{
Numeric vectors, except \code{lcs_substring()}, which returns character
vectors.
}
\description{
Longest-common-subsequence, longest-common-substring, multiset n-gram,
Ratcliff-Obershelp, token-ratio, partial-ratio, weighted-ratio, and
Monge-Elkan algorithms from stride-align's native API.
}
