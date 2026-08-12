\name{stride_levenshtein}
\alias{stride_levenshtein}
\alias{stride_levenshtein_similarity}
\alias{stride_osa}
\alias{stride_osa_similarity}
\alias{stride_true_damerau_levenshtein}
\alias{stride_true_damerau_levenshtein_similarity}
\alias{stride_indel}
\alias{stride_indel_similarity}
\alias{stride_hamming}
\alias{stride_hamming_similarity}
\alias{stride_jaro}
\alias{stride_jaro_winkler}
\title{String distance and similarity}
\usage{
stride_levenshtein(query, target)
stride_levenshtein_similarity(query, target)
stride_osa(query, target)
stride_osa_similarity(query, target)
stride_true_damerau_levenshtein(query, target)
stride_true_damerau_levenshtein_similarity(query, target)
stride_indel(query, target)
stride_indel_similarity(query, target)
stride_hamming(query, target)
stride_hamming_similarity(query, target)
stride_jaro(query, target)
stride_jaro_winkler(query, target, prefix_weight = 0.1,
  prefix_threshold = 0.7, prefix_cap = 4)
}
\arguments{
\item{query, target}{Character vectors. One length-one input is broadcast.}
\item{prefix_weight}{Numeric Jaro-Winkler prefix bonus.}
\item{prefix_threshold}{Minimum Jaro score for the prefix bonus.}
\item{prefix_cap}{Maximum rewarded prefix length.}
}
\value{A numeric vector. Missing inputs produce missing results.}
\description{
Computes Unicode-aware string distances or normalized similarities.
}
