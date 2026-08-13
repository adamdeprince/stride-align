.stride_scalar_query <- function(query) {
  if (!is.character(query) || length(query) != 1L) {
    stop("query must be one character string", call. = FALSE)
  }
  query
}

.stride_targets <- function(targets) {
  if (!is.character(targets)) {
    stop("targets must be a character vector", call. = FALSE)
  }
  targets
}

.stride_non_negative_integer <- function(value, name) {
  if (length(value) != 1L || !is.numeric(value) || is.na(value) ||
      !is.finite(value) || value < 0 || value != trunc(value)) {
    stop(name, " must be one non-negative integer", call. = FALSE)
  }
  as.integer(value)
}

.stride_optional_cutoff <- function(scores, score_cutoff) {
  if (is.null(score_cutoff)) {
    return(scores)
  }
  cutoff <- .stride_non_negative_integer(score_cutoff, "score_cutoff")
  pmin(scores, cutoff + 1, na.rm = FALSE)
}

.stride_optional_similarity_cutoff <- function(scores, score_cutoff) {
  if (is.null(score_cutoff)) return(scores)
  if (length(score_cutoff) != 1L || !is.numeric(score_cutoff) ||
      is.na(score_cutoff) || !is.finite(score_cutoff) ||
      score_cutoff < 0 || score_cutoff > 1) {
    stop("score_cutoff must be between 0 and 1", call. = FALSE)
  }
  ifelse(is.na(scores), NA_real_, ifelse(scores < score_cutoff, 0, scores))
}

.stride_matrix_kwargs_clean <- function(match_score, mismatch_score, width) {
  if (!identical(as.numeric(match_score), 2) ||
      !identical(as.numeric(mismatch_score), -1) || !is.null(width)) {
    stop(
      "match_score, mismatch_score, and width cannot be used with matrix",
      call. = FALSE
    )
  }
}

.stride_resolve_gaps <- function(gap_score, gap_open_score, gap_extend_score) {
  gap_open <- if (is.null(gap_open_score)) gap_score else gap_open_score
  gap_extend <- if (is.null(gap_extend_score)) gap_open else gap_extend_score
  list(open = gap_open, extend = gap_extend)
}

.stride_alignment_score <- function(
  query,
  target,
  local,
  match_score,
  mismatch_score,
  gap_score,
  gap_open_score,
  gap_extend_score
) {
  gaps <- .stride_resolve_gaps(gap_score, gap_open_score, gap_extend_score)
  if (identical(as.numeric(gaps$open), as.numeric(gaps$extend))) {
    if (local) {
      return(stride_smith_waterman(
        query, target, match_score, mismatch_score, gaps$open
      ))
    }
    return(stride_needleman_wunsch(
      query, target, match_score, mismatch_score, gaps$open
    ))
  }
  if (local) {
    return(stride_smith_waterman_affine(
      query, target, match_score, mismatch_score, gaps$open, gaps$extend
    ))
  }
  stride_needleman_wunsch_affine(
    query, target, match_score, mismatch_score, gaps$open, gaps$extend
  )
}

.stride_alignment_normalize <- function(raw, query, target, local, match_score) {
  if (length(match_score) != 1L || !is.numeric(match_score) ||
      is.na(match_score) || !is.finite(match_score) || match_score <= 0) {
    stop("match_score must be positive for normalized scores", call. = FALSE)
  }
  output_size <- length(raw)
  query_values <- if (length(query) == 1L) rep(query, output_size) else query
  target_values <- if (length(target) == 1L) rep(target, output_size) else target
  query_length <- nchar(query_values, type = "chars", allowNA = TRUE)
  target_length <- nchar(target_values, type = "chars", allowNA = TRUE)
  denominator <- if (local) {
    pmin(query_length, target_length) * match_score
  } else {
    pmax(query_length, target_length) * match_score
  }
  result <- numeric(output_size)
  valid <- !is.na(raw) & !is.na(denominator)
  nonzero <- valid & denominator > 0
  result[nonzero] <- raw[nonzero] / denominator[nonzero]
  both_empty <- valid & query_length == 0 & target_length == 0
  result[both_empty] <- 1
  result[!valid] <- NA_real_
  pmax(0, pmin(1, result))
}

# Canonical Python API spellings. Pair functions remain vectorized in R:
# equal-length inputs are processed elementwise and a length-one input is
# broadcast across the other side.
levenshtein_score <- function(query, target, score_cutoff = NULL) {
  .stride_optional_cutoff(stride_levenshtein(query, target), score_cutoff)
}

levenshtein_normalized_score <- function(query, target, score_cutoff = NULL) {
  distance <- levenshtein_score(query, target, score_cutoff)
  query_values <- if (length(query) == 1L) rep(query, length(distance)) else query
  target_values <- if (length(target) == 1L) rep(target, length(distance)) else target
  longest <- pmax(
    nchar(query_values, type = "chars", allowNA = TRUE),
    nchar(target_values, type = "chars", allowNA = TRUE)
  )
  result <- ifelse(longest == 0, 1, 1 - distance / longest)
  result[is.na(distance)] <- NA_real_
  pmax(0, result)
}

damerau_levenshtein_score <- function(query, target) stride_osa(query, target)
damerau_levenshtein_normalized_score <- function(query, target) {
  stride_osa_similarity(query, target)
}
true_damerau_levenshtein_score <- function(query, target) {
  stride_true_damerau_levenshtein(query, target)
}
true_damerau_levenshtein_normalized_score <- function(query, target) {
  stride_true_damerau_levenshtein_similarity(query, target)
}
indel_score <- function(query, target, score_cutoff = NULL) {
  .stride_optional_cutoff(stride_indel(query, target), score_cutoff)
}
indel_normalized_score <- function(query, target, score_cutoff = NULL) {
  .stride_optional_similarity_cutoff(
    stride_indel_similarity(query, target), score_cutoff
  )
}
hamming_score <- function(query, target) stride_hamming(query, target)
hamming_normalized_score <- function(query, target) stride_hamming_similarity(query, target)
jaro_similarity <- function(query, target) stride_jaro(query, target)
jaro_winkler_similarity <- function(
  query,
  target,
  prefix_weight = 0.1,
  prefix_threshold = 0.7,
  prefix_cap = 4
) {
  stride_jaro_winkler(
    query, target, prefix_weight, prefix_threshold, prefix_cap
  )
}

smith_waterman_score <- function(
  query,
  target,
  match_score = 2,
  mismatch_score = -1,
  gap_score = -1,
  gap_open_score = NULL,
  gap_extend_score = NULL,
  width = NULL,
  matrix = NULL
) {
  if (!is.null(matrix)) {
    .stride_matrix_kwargs_clean(match_score, mismatch_score, width)
    return(.stride_matrix_pairwise(
      query, target, matrix, TRUE, gap_score, gap_open_score, gap_extend_score
    ))
  }
  if (!is.null(width) && !width %in% c(0, 8, 16, 32, 64)) {
    stop("width must be NULL, 0, 8, 16, 32, or 64", call. = FALSE)
  }
  .stride_alignment_score(
    query, target, TRUE, match_score, mismatch_score, gap_score,
    gap_open_score, gap_extend_score
  )
}

needleman_wunsch_score <- function(
  query,
  target,
  match_score = 2,
  mismatch_score = -1,
  gap_score = -1,
  gap_open_score = NULL,
  gap_extend_score = NULL,
  width = NULL,
  matrix = NULL
) {
  if (!is.null(matrix)) {
    .stride_matrix_kwargs_clean(match_score, mismatch_score, width)
    return(.stride_matrix_pairwise(
      query, target, matrix, FALSE, gap_score, gap_open_score, gap_extend_score
    ))
  }
  if (!is.null(width) && !width %in% c(0, 8, 16, 32, 64)) {
    stop("width must be NULL, 0, 8, 16, 32, or 64", call. = FALSE)
  }
  .stride_alignment_score(
    query, target, FALSE, match_score, mismatch_score, gap_score,
    gap_open_score, gap_extend_score
  )
}

smith_waterman_normalized_score <- function(query, target, ...) {
  arguments <- list(...)
  match_score <- if (is.null(arguments$match_score)) 2 else arguments$match_score
  raw <- do.call(smith_waterman_score, c(list(query = query, target = target), arguments))
  .stride_alignment_normalize(raw, query, target, TRUE, match_score)
}

needleman_wunsch_normalized_score <- function(query, target, ...) {
  arguments <- list(...)
  match_score <- if (is.null(arguments$match_score)) 2 else arguments$match_score
  raw <- do.call(needleman_wunsch_score, c(list(query = query, target = target), arguments))
  .stride_alignment_normalize(raw, query, target, FALSE, match_score)
}

smith_waterman_farrar_score <- smith_waterman_score
smith_waterman_farrar_normalized_score <- smith_waterman_normalized_score

.stride_one_to_many <- function(query, targets, function_name, ...) {
  query <- .stride_scalar_query(query)
  targets <- .stride_targets(targets)
  function_name(query, targets, ...)
}

levenshtein_scores <- function(query, targets, score_cutoff = NULL) {
  .stride_one_to_many(query, targets, levenshtein_score, score_cutoff)
}
levenshtein_normalized_scores <- function(query, targets, score_cutoff = NULL) {
  .stride_one_to_many(query, targets, levenshtein_normalized_score, score_cutoff)
}
damerau_levenshtein_scores <- function(query, targets) {
  .stride_one_to_many(query, targets, damerau_levenshtein_score)
}
damerau_levenshtein_normalized_scores <- function(query, targets) {
  .stride_one_to_many(query, targets, damerau_levenshtein_normalized_score)
}
true_damerau_levenshtein_scores <- function(query, targets) {
  .stride_one_to_many(query, targets, true_damerau_levenshtein_score)
}
true_damerau_levenshtein_normalized_scores <- function(query, targets) {
  .stride_one_to_many(query, targets, true_damerau_levenshtein_normalized_score)
}
indel_scores <- function(query, targets) {
  .stride_one_to_many(query, targets, indel_score)
}
indel_normalized_scores <- function(query, targets) {
  .stride_one_to_many(query, targets, indel_normalized_score)
}
hamming_scores <- function(query, targets) {
  .stride_one_to_many(query, targets, hamming_score)
}
hamming_normalized_scores <- function(query, targets) {
  .stride_one_to_many(query, targets, hamming_normalized_score)
}
jaro_similarities <- function(query, targets) {
  .stride_one_to_many(query, targets, jaro_similarity)
}
jaro_winkler_similarities <- function(query, targets, ...) {
  .stride_one_to_many(query, targets, jaro_winkler_similarity, ...)
}
smith_waterman_scores <- function(query, targets, ...) {
  .stride_one_to_many(query, targets, smith_waterman_score, ...)
}
smith_waterman_normalized_scores <- function(query, targets, ...) {
  .stride_one_to_many(query, targets, smith_waterman_normalized_score, ...)
}
smith_waterman_farrar_scores <- smith_waterman_scores
smith_waterman_farrar_normalized_scores <- smith_waterman_normalized_scores
needleman_wunsch_scores <- function(query, targets, ...) {
  .stride_one_to_many(query, targets, needleman_wunsch_score, ...)
}
needleman_wunsch_normalized_scores <- function(query, targets, ...) {
  .stride_one_to_many(query, targets, needleman_wunsch_normalized_score, ...)
}

Scorer <- list(
  LEVENSHTEIN = 0L,
  LEVENSHTEIN_NORMALIZED = 1L,
  DAMERAU_LEVENSHTEIN = 2L,
  DAMERAU_LEVENSHTEIN_NORMALIZED = 3L,
  HAMMING = 4L,
  HAMMING_NORMALIZED = 5L,
  JARO = 6L,
  JARO_WINKLER = 7L,
  INDEL = 8L,
  INDEL_NORMALIZED = 9L,
  TRUE_DAMERAU_LEVENSHTEIN = 10L,
  TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED = 11L,
  SMITH_WATERMAN = 12L,
  SMITH_WATERMAN_NORMALIZED = 13L,
  NEEDLEMAN_WUNSCH = 14L,
  NEEDLEMAN_WUNSCH_NORMALIZED = 15L
)

.stride_scorer_table <- function() {
  list(
    list(names = c("levenshtein", "levenshtein_score"), id = 0L,
         batch = levenshtein_scores, higher = FALSE, normalized = FALSE),
    list(names = c("levenshtein_normalized", "levenshtein_normalized_score"), id = 1L,
         batch = levenshtein_normalized_scores, higher = TRUE, normalized = TRUE),
    list(names = c("damerau_levenshtein", "osa", "damerau_levenshtein_score"), id = 2L,
         batch = damerau_levenshtein_scores, higher = FALSE, normalized = FALSE),
    list(names = c("damerau_levenshtein_normalized", "osa_normalized"), id = 3L,
         batch = damerau_levenshtein_normalized_scores, higher = TRUE, normalized = TRUE),
    list(names = c("hamming", "hamming_score"), id = 4L,
         batch = hamming_scores, higher = FALSE, normalized = FALSE),
    list(names = c("hamming_normalized", "hamming_normalized_score"), id = 5L,
         batch = hamming_normalized_scores, higher = TRUE, normalized = TRUE),
    list(names = c("jaro", "jaro_similarity"), id = 6L,
         batch = jaro_similarities, higher = TRUE, normalized = TRUE),
    list(names = c("jaro_winkler", "jaro_winkler_similarity"), id = 7L,
         batch = jaro_winkler_similarities, higher = TRUE, normalized = TRUE),
    list(names = c("indel", "indel_score"), id = 8L,
         batch = indel_scores, higher = FALSE, normalized = FALSE),
    list(names = c("indel_normalized", "indel_normalized_score"), id = 9L,
         batch = indel_normalized_scores, higher = TRUE, normalized = TRUE),
    list(names = c("true_damerau_levenshtein", "true_damerau_levenshtein_score"), id = 10L,
         batch = true_damerau_levenshtein_scores, higher = FALSE, normalized = FALSE),
    list(names = c("true_damerau_levenshtein_normalized"), id = 11L,
         batch = true_damerau_levenshtein_normalized_scores, higher = TRUE, normalized = TRUE),
    list(names = c("sw", "smith_waterman", "smith-waterman", "local", "smith_waterman_score"), id = 12L,
         batch = smith_waterman_scores, higher = TRUE, normalized = FALSE),
    list(names = c("smith_waterman_normalized"), id = 13L,
         batch = smith_waterman_normalized_scores, higher = TRUE, normalized = TRUE),
    list(names = c("nw", "needleman_wunsch", "needleman-wunsch", "global", "needleman_wunsch_score"), id = 14L,
         batch = needleman_wunsch_scores, higher = TRUE, normalized = FALSE),
    list(names = c("needleman_wunsch_normalized"), id = 15L,
         batch = needleman_wunsch_normalized_scores, higher = TRUE, normalized = TRUE)
  )
}

.stride_resolve_scorer <- function(scorer) {
  table <- .stride_scorer_table()
  if (is.function(scorer)) {
    for (entry in table) {
      if (identical(scorer, entry$batch)) return(entry)
    }
    stop("unknown stride-align scorer function", call. = FALSE)
  }
  if (length(scorer) != 1L || is.na(scorer)) {
    stop("scorer must be one scorer name, ID, or batch function", call. = FALSE)
  }
  if (is.numeric(scorer)) {
    id <- as.integer(scorer)
    for (entry in table) if (entry$id == id) return(entry)
  } else if (is.character(scorer)) {
    name <- tolower(scorer)
    for (entry in table) if (name %in% entry$names) return(entry)
  }
  stop("unknown stride-align scorer", call. = FALSE)
}

.stride_ranked <- function(targets, scores, k, higher) {
  k <- .stride_non_negative_integer(k, "k")
  usable <- which(!is.na(scores))
  if (!length(usable) || k == 0L) {
    return(data.frame(
      target = character(), score = numeric(), index = integer(),
      stringsAsFactors = FALSE
    ))
  }
  order_index <- order(if (higher) -scores[usable] else scores[usable], usable)
  selected <- usable[utils::head(order_index, k)]
  data.frame(
    target = targets[selected],
    score = scores[selected],
    index = selected,
    stringsAsFactors = FALSE
  )
}

.stride_top_k <- function(query, targets, scorer, k = 5, ...) {
  query <- .stride_scalar_query(query)
  targets <- .stride_targets(targets)
  entry <- .stride_resolve_scorer(scorer)
  scores <- entry$batch(query, targets, ...)
  .stride_ranked(targets, scores, k, entry$higher)
}

.stride_best <- function(query, targets, scorer, ...) {
  result <- .stride_top_k(query, targets, scorer, 1L, ...)
  if (!nrow(result)) return(NULL)
  list(target = result$target[[1L]], score = result$score[[1L]], index = result$index[[1L]])
}

levenshtein_top_k <- function(query, targets, k = 5) {
  .stride_top_k(query, targets, Scorer$LEVENSHTEIN, k)
}
levenshtein_normalized_top_k <- function(query, targets, k = 5) {
  .stride_top_k(query, targets, Scorer$LEVENSHTEIN_NORMALIZED, k)
}
damerau_levenshtein_top_k <- function(query, targets, k = 5) {
  .stride_top_k(query, targets, Scorer$DAMERAU_LEVENSHTEIN, k)
}
damerau_levenshtein_normalized_top_k <- function(query, targets, k = 5) {
  .stride_top_k(query, targets, Scorer$DAMERAU_LEVENSHTEIN_NORMALIZED, k)
}
true_damerau_levenshtein_top_k <- function(query, targets, k = 5) {
  .stride_top_k(query, targets, Scorer$TRUE_DAMERAU_LEVENSHTEIN, k)
}
true_damerau_levenshtein_normalized_top_k <- function(query, targets, k = 5) {
  .stride_top_k(query, targets, Scorer$TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED, k)
}
indel_top_k <- function(query, targets, k = 5) {
  .stride_top_k(query, targets, Scorer$INDEL, k)
}
indel_normalized_top_k <- function(query, targets, k = 5) {
  .stride_top_k(query, targets, Scorer$INDEL_NORMALIZED, k)
}
hamming_top_k <- function(query, targets, k = 5) {
  .stride_top_k(query, targets, Scorer$HAMMING, k)
}
hamming_normalized_top_k <- function(query, targets, k = 5) {
  .stride_top_k(query, targets, Scorer$HAMMING_NORMALIZED, k)
}
jaro_top_k <- function(query, targets, k = 5) {
  .stride_top_k(query, targets, Scorer$JARO, k)
}
jaro_winkler_top_k <- function(query, targets, k = 5, ...) {
  .stride_top_k(query, targets, Scorer$JARO_WINKLER, k, ...)
}
smith_waterman_top_k <- function(query, targets, k = 5, ...) {
  .stride_top_k(query, targets, Scorer$SMITH_WATERMAN, k, ...)
}

levenshtein_best <- function(query, targets) {
  .stride_best(query, targets, Scorer$LEVENSHTEIN)
}
levenshtein_normalized_best <- function(query, targets) {
  .stride_best(query, targets, Scorer$LEVENSHTEIN_NORMALIZED)
}
damerau_levenshtein_best <- function(query, targets) {
  .stride_best(query, targets, Scorer$DAMERAU_LEVENSHTEIN)
}
damerau_levenshtein_normalized_best <- function(query, targets) {
  .stride_best(query, targets, Scorer$DAMERAU_LEVENSHTEIN_NORMALIZED)
}
true_damerau_levenshtein_best <- function(query, targets) {
  .stride_best(query, targets, Scorer$TRUE_DAMERAU_LEVENSHTEIN)
}
true_damerau_levenshtein_normalized_best <- function(query, targets) {
  .stride_best(query, targets, Scorer$TRUE_DAMERAU_LEVENSHTEIN_NORMALIZED)
}
indel_best <- function(query, targets) .stride_best(query, targets, Scorer$INDEL)
indel_normalized_best <- function(query, targets) {
  .stride_best(query, targets, Scorer$INDEL_NORMALIZED)
}
hamming_best <- function(query, targets) .stride_best(query, targets, Scorer$HAMMING)
hamming_normalized_best <- function(query, targets) {
  .stride_best(query, targets, Scorer$HAMMING_NORMALIZED)
}
jaro_best <- function(query, targets) .stride_best(query, targets, Scorer$JARO)
jaro_winkler_best <- function(query, targets, ...) {
  .stride_best(query, targets, Scorer$JARO_WINKLER, ...)
}
smith_waterman_best <- function(query, targets, ...) {
  .stride_best(query, targets, Scorer$SMITH_WATERMAN, ...)
}

extract <- function(query, targets, scorer, k = 5, ...) {
  .stride_top_k(query, targets, scorer, k, ...)
}
extract_best <- function(query, targets, scorer, ...) {
  .stride_best(query, targets, scorer, ...)
}

cdist <- function(
  queries,
  targets,
  scorer = NULL,
  matrix = NULL,
  match_score = 2,
  mismatch_score = -1,
  gap_score = -1,
  gap_open_score = NULL,
  gap_extend_score = NULL,
  width = NULL,
  tqdm = NULL,
  cpu_count = 0,
  prefix_weight = 0.1,
  prefix_threshold = 0.7,
  prefix_cap = 4
) {
  queries <- .stride_targets(queries)
  targets <- .stride_targets(targets)
  if (is.null(scorer)) {
    if (is.null(matrix)) {
      stop("cdist requires either scorer or matrix", call. = FALSE)
    }
    scorer <- Scorer$SMITH_WATERMAN
  }
  entry <- .stride_resolve_scorer(scorer)
  if (!is.null(matrix) && !entry$id %in% c(
    Scorer$SMITH_WATERMAN, Scorer$NEEDLEMAN_WUNSCH
  )) {
    stop("matrix cdist requires a Smith-Waterman or Needleman-Wunsch scorer", call. = FALSE)
  }
  result <- base::matrix(numeric(length(queries) * length(targets)),
                         nrow = length(queries), ncol = length(targets))
  if (!length(queries) || !length(targets)) return(result)
  progress <- NULL
  if (!is.null(tqdm)) {
    if (!is.function(tqdm)) {
      stop("tqdm must be a progress-bar factory function or NULL", call. = FALSE)
    }
    progress <- tqdm(length(queries))
    if (!is.list(progress) && !is.environment(progress)) {
      stop("tqdm factory must return a list or environment", call. = FALSE)
    }
    if (!is.function(progress$update) || !is.function(progress$close)) {
      stop("tqdm progress object must provide update and close functions", call. = FALSE)
    }
    on.exit(progress$close(), add = TRUE)
  }
  for (index in seq_along(queries)) {
    arguments <- list(query = queries[[index]], targets = targets)
    if (entry$id %in% 12:15) {
      arguments <- c(arguments, list(
        match_score = match_score,
        mismatch_score = mismatch_score,
        gap_score = gap_score,
        gap_open_score = gap_open_score,
        gap_extend_score = gap_extend_score,
        width = width
      ))
    }
    if (!is.null(matrix)) arguments$matrix <- matrix
    if (entry$id == Scorer$JARO_WINKLER) {
      arguments <- c(arguments, list(
        prefix_weight = prefix_weight,
        prefix_threshold = prefix_threshold,
        prefix_cap = prefix_cap
      ))
    }
    result[index, ] <- do.call(entry$batch, arguments)
    if (!is.null(progress)) progress$update(1L)
  }
  result
}

cdist_above_threshold <- function(
  queries,
  targets,
  scorer = NULL,
  threshold,
  matrix = NULL,
  ...
) {
  if (is.null(scorer) && is.null(matrix)) {
    stop("cdist_above_threshold requires either scorer or matrix", call. = FALSE)
  }
  resolved_scorer <- if (is.null(scorer)) Scorer$SMITH_WATERMAN else scorer
  entry <- .stride_resolve_scorer(resolved_scorer)
  if (is.null(matrix) && !entry$normalized) {
    stop("cdist_above_threshold requires a normalized or similarity scorer", call. = FALSE)
  }
  if (length(threshold) != 1L || !is.numeric(threshold) || is.na(threshold) ||
      !is.finite(threshold) || (is.null(matrix) && (threshold < 0 || threshold > 1))) {
    stop(
      if (is.null(matrix)) "threshold must be between 0 and 1" else
        "threshold must be one finite numeric value",
      call. = FALSE
    )
  }
  scores <- cdist(
    queries, targets, scorer = resolved_scorer, matrix = matrix, ...
  )
  positions <- which(!is.na(scores) & scores >= threshold, arr.ind = TRUE)
  if (!nrow(positions)) {
    return(data.frame(
      score = numeric(), query = character(), target = character(),
      query_index = integer(), target_index = integer(),
      stringsAsFactors = FALSE
    ))
  }
  data.frame(
    score = scores[positions],
    query = queries[positions[, 1L]],
    target = targets[positions[, 2L]],
    query_index = positions[, 1L],
    target_index = positions[, 2L],
    stringsAsFactors = FALSE
  )
}

cdist_top_k <- function(
  queries,
  targets,
  scorer = NULL,
  k,
  matrix = NULL,
  reject_duplicates = FALSE,
  ...
) {
  if (is.null(scorer) && is.null(matrix)) {
    stop("cdist_top_k requires either scorer or matrix", call. = FALSE)
  }
  resolved_scorer <- if (is.null(scorer)) Scorer$SMITH_WATERMAN else scorer
  entry <- .stride_resolve_scorer(resolved_scorer)
  if (is.null(matrix) && !entry$normalized) {
    stop("cdist_top_k requires a normalized or similarity scorer", call. = FALSE)
  }
  if (!is.null(matrix) && isTRUE(reject_duplicates)) {
    stop("reject_duplicates is not supported with matrix", call. = FALSE)
  }
  k <- .stride_non_negative_integer(k, "k")
  queries <- .stride_targets(queries)
  targets <- .stride_targets(targets)
  scores <- cdist(
    queries, targets, scorer = resolved_scorer, matrix = matrix, ...
  )
  positions <- which(!is.na(scores), arr.ind = TRUE)
  if (isTRUE(reject_duplicates) && nrow(positions)) {
    keep <- queries[positions[, 1L]] != targets[positions[, 2L]]
    positions <- positions[keep, , drop = FALSE]
  }
  if (!nrow(positions) || k == 0L) {
    return(data.frame(
      score = numeric(), query = character(), target = character(),
      query_index = integer(), target_index = integer(),
      stringsAsFactors = FALSE
    ))
  }
  values <- scores[positions]
  selected <- utils::head(order(-values, positions[, 1L], positions[, 2L]), k)
  positions <- positions[selected, , drop = FALSE]
  data.frame(
    score = values[selected],
    query = queries[positions[, 1L]],
    target = targets[positions[, 2L]],
    query_index = positions[, 1L],
    target_index = positions[, 2L],
    stringsAsFactors = FALSE
  )
}

cdist_top_k_per_query <- function(
  queries,
  targets,
  scorer,
  k = 5,
  pruning = FALSE,
  cpu_count = 0,
  prefix_weight = 0.1,
  prefix_threshold = 0.7,
  prefix_cap = 4
) {
  queries <- .stride_targets(queries)
  targets <- .stride_targets(targets)
  entry <- .stride_resolve_scorer(scorer)
  if (!entry$normalized) {
    stop("cdist_top_k_per_query requires a normalized or similarity scorer", call. = FALSE)
  }
  arguments <- if (entry$id == Scorer$JARO_WINKLER) {
    list(
      prefix_weight = prefix_weight,
      prefix_threshold = prefix_threshold,
      prefix_cap = prefix_cap
    )
  } else {
    list()
  }
  results <- lapply(queries, function(query) {
    if (entry$id == Scorer$HAMMING_NORMALIZED) {
      valid <- which(
        nchar(targets, type = "chars", allowNA = TRUE) ==
          nchar(query, type = "chars", allowNA = TRUE)
      )
      if (!length(valid)) {
        return(.stride_ranked(character(), numeric(), k, TRUE))
      }
      ranked <- .stride_ranked(
        targets[valid], hamming_normalized_scores(query, targets[valid]), k, TRUE
      )
      ranked$index <- valid[ranked$index]
      ranked
    } else {
      do.call(
        .stride_top_k,
        c(list(query = query, targets = targets, scorer = scorer, k = k), arguments)
      )
    }
  })
  names(results) <- queries
  results
}

LevenshteinScorer <- function(query) {
  query <- .stride_scalar_query(query)
  structure(list(
    query = query,
    distance = function(target, score_cutoff = NULL) {
      levenshtein_score(query, target, score_cutoff)
    },
    normalized_distance = function(target, score_cutoff = NULL) {
      levenshtein_normalized_score(query, target, score_cutoff)
    },
    distances = function(targets, score_cutoff = NULL) {
      levenshtein_scores(query, targets, score_cutoff)
    },
    normalized_distances = function(targets, score_cutoff = NULL) {
      levenshtein_normalized_scores(query, targets, score_cutoff)
    }
  ), class = "stride_levenshtein_scorer")
}
