source("helpers.R")

for (query in list(as.double(1:5), as.integer(1:5))) stopifnot(dtw(query, query) == 0)
for (kind in c("double", "integer")) {
  set.seed(17)
  query <- rnorm(20) * 10
  target <- rnorm(30) * 10
  if (kind == "integer") {
    query <- as.integer(query)
    target <- as.integer(target)
  }
  metric <- if (kind == "integer") "l1" else "l2_squared"
  close_to(dtw(query, target), reference_dtw(query, target, metric), 1e-6)
}
query <- c(0, 1, 0)
target <- c(0, 0, 1)
stopifnot(dtw(query, target) == dtw(query, target, distance = "l2_squared"))
query_integer <- as.integer(query)
target_integer <- as.integer(target)
stopifnot(dtw(query_integer, target_integer) == dtw(query_integer, target_integer, distance = "l1"))
for (query in list(as.double(0:4), as.integer(0:4))) {
  target <- c(query[[1L]], query)
  stopifnot(dtw(query, target, distance = "l1") == 0)
}

set.seed(1)
query <- rnorm(15) * 5
target <- rnorm(15) * 5
stopifnot(dtw(query, target, window = NULL) == dtw(query, target))
set.seed(2)
query <- rnorm(20) * 5
target <- rnorm(20) * 5
banded <- dtw(query, target, window = 2L)
close_to(banded, reference_dtw(query, target, "l2_squared", 2L), 1e-9)
stopifnot(banded >= dtw(query, target))
set.seed(3)
query <- rnorm(20) * 5
target <- rnorm(20) * 5
stopifnot(
  dtw(query, target, window = 0.1) == dtw(query, target, window = 2L),
  # R has one numeric scalar syntax for 1 and 1.0; whole values are radii.
  # A radius spanning the longer sequence is the unconstrained equivalent.
  dtw(query, target, window = max(length(query), length(target))) == dtw(query, target)
)
close_to(dtw(c(1, 2, 3), c(1.5, 2.5, 3.5), window = 0L), 0.75)

set.seed(11)
query <- as.double(rnorm(12) * 3)
targets <- lapply(0:4, function(index) as.double(rnorm(10 + index) * 3))
batch <- dtw_distances(query, targets)
close_to(batch, vapply(targets, function(target) dtw(query, target), numeric(1)), 1e-9)
stopifnot(is.double(batch))
set.seed(22)
query_integer <- as.integer(rnorm(10) * 100)
targets_integer <- lapply(1:3, function(index) as.integer(rnorm(12) * 100))
close_to(
  dtw_distances(query_integer, targets_integer, window = 3L),
  vapply(targets_integer, function(target) reference_dtw(query_integer, target, "l1", 3L), numeric(1)),
  1e-9
)
stopifnot(is.double(dtw_distances(c(1, 2), list())), length(dtw_distances(c(1, 2), list())) == 0L)

# R has integer and double vectors rather than NumPy's int16/float32/float64
# dtype matrix. Its integer path is the direct equivalent of Python int16.
expect_error(dtw(1:3, as.double(1:3)), "share numeric storage type")
expect_error(dtw("abc", "abd"), "numeric vectors")
expect_error(dtw(numeric(), 1), "non-empty")
expect_error(dtw(1 + 0, numeric()), "non-empty")
expect_error(dtw(c(1, 2), c(1, 2), window = -1), "non-negative")
expect_error(dtw(c(1, 2), c(1, 2), window = 1.5), "fraction in (0, 1)")
expect_error(dtw(c(1, 2), c(1, 2), distance = "foo"), "'l1' or 'l2_squared'")
expect_error(dtw_distances(c(1), "not-a-list"), "list of numeric vectors")
expect_error(dtw_distances(c(1), list(c(1), 1L)), "share the query's numeric storage type")

for (kind in c("double", "integer")) for (metric in c(NA, "l1")) for (window in list(NULL, 5L, 0.25)) {
  set.seed(99)
  query <- rnorm(32) * 8
  targets <- lapply(c(16, 32, 48, 24, 40, 32, 20, 36, 28), function(n) rnorm(n) * 8)
  if (kind == "integer") {
    query <- as.integer(query)
    targets <- lapply(targets, as.integer)
  }
  distance <- if (is.na(metric)) NULL else metric
  batch <- dtw_distances(query, targets, window = window, distance = distance)
  singular <- vapply(targets, function(target) dtw(query, target, window = window, distance = distance), numeric(1))
  close_to(batch, singular, 1e-5)
}
query <- 0:3 + 0
target <- 10:13 + 0
full <- dtw(query, target, distance = "l1")
stopifnot(
  full > 0,
  is.infinite(dtw(query, target, distance = "l1", score_cutoff = full * 0.01))
)
close_to(dtw(query, target, distance = "l1", score_cutoff = full + 1), full)
stopifnot(is.infinite(dtw(rep(0, 10), rep(0, 30), window = 5L, distance = "l1")))
set.seed(3)
query <- rnorm(64)
near <- query + rnorm(64, 0, 0.01)
far <- rnorm(64) * 20
distances <- dtw_distances(
  query, list(near, far), window = 8L,
  distance = "l2_squared", score_cutoff = 1
)
stopifnot(is.finite(distances[[1L]]), is.infinite(distances[[2L]]) || distances[[2L]] > 1)

# Python cases: test_dtw_identity_zero_float, test_dtw_identity_zero_int16,
# test_dtw_matches_reference_unconstrained,
# test_dtw_l2_squared_default_for_float, test_dtw_l1_default_for_int16,
# test_dtw_l1_simple_shift, test_dtw_band_none_matches_unconstrained,
# test_dtw_band_int_radius, test_dtw_band_float_fraction,
# test_dtw_band_zero_diagonal_only, test_dtw_distances_matches_singular,
# test_dtw_distances_batch_int16, test_dtw_distances_empty_targets,
# test_dtw_rejects_unsupported_dtype, test_dtw_rejects_mixed_dtype,
# test_dtw_rejects_non_ndarray, test_dtw_rejects_empty,
# test_dtw_rejects_bad_window, test_dtw_rejects_bad_distance,
# test_dtw_distances_rejects_str_targets,
# test_dtw_distances_rejects_mixed_dtype_targets,
# test_dtw_distances_matches_singular, test_dtw_score_cutoff_returns_inf_when_exceeded,
# test_dtw_band_impossibility_is_inf,
# test_dtw_distances_cutoff_prunes_dissimilar.
