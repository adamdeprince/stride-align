\name{dtw_api}
\alias{dtw}
\alias{dtw_distances}
\title{Dynamic Time Warping distances}
\usage{
dtw(query, target, window = NULL, distance = NULL, score_cutoff = NULL)
dtw_distances(query, targets, window = NULL, distance = NULL,
  score_cutoff = NULL)
}
\arguments{
\item{query, target}{Non-empty integer or double vectors with the same R
storage type.}
\item{targets}{A list of numeric vectors sharing the query's storage type.}
\item{window}{A non-negative integer radius, a fraction in \code{(0, 1)}, or
\code{NULL} for an unconstrained path.}
\item{distance}{\code{"l1"}, \code{"l2_squared"}, or \code{NULL}. The
default is L1 for integer vectors and squared L2 for double vectors.}
\item{score_cutoff}{Optional non-negative cutoff; rejected paths return
positive infinity.}
}
\value{One numeric distance or a numeric vector of distances.}
\description{
Dynamic Time Warping with optional Sakoe-Chiba banding and one-to-many batch
scoring.
}
