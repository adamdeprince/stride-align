lcs_length <- function(query, target) {
  .stridealign_call("stride_r_lcs_length", query, target)
}

lcs_substring_length <- function(query, target) {
  .stridealign_call("stride_r_lcs_substring_length", query, target)
}

lcs_substring <- function(query, target) {
  .stridealign_call("stride_r_lcs_substring", query, target)
}

jaccard <- function(query, target, n = 2) {
  .stridealign_call("stride_r_jaccard", query, target, n)
}

dice <- function(query, target, n = 2) {
  .stridealign_call("stride_r_dice", query, target, n)
}

cosine <- function(query, target, n = 2) {
  .stridealign_call("stride_r_cosine", query, target, n)
}

overlap <- function(query, target, n = 2) {
  .stridealign_call("stride_r_overlap", query, target, n)
}

jaccard_similarities <- function(query, targets, n = 2) {
  .stride_one_to_many(query, targets, jaccard, n)
}

dice_similarities <- function(query, targets, n = 2) {
  .stride_one_to_many(query, targets, dice, n)
}

cosine_similarities <- function(query, targets, n = 2) {
  .stride_one_to_many(query, targets, cosine, n)
}

overlap_similarities <- function(query, targets, n = 2) {
  .stride_one_to_many(query, targets, overlap, n)
}

ratcliff_obershelp_similarity <- function(query, target) {
  .stridealign_call("stride_r_ratcliff_obershelp", query, target)
}

ratcliff_obershelp_similarities <- function(query, targets) {
  .stride_one_to_many(query, targets, ratcliff_obershelp_similarity)
}

.stride_apply_processor <- function(input, processor) {
  if (is.null(processor)) return(input)
  if (!is.function(processor)) {
    stop("processor must be a function or NULL", call. = FALSE)
  }
  if (!is.character(input)) {
    stop("inputs must be character vectors", call. = FALSE)
  }
  vapply(input, function(value) {
    if (is.na(value)) return(NA_character_)
    result <- processor(value)
    if (!is.character(result) || length(result) != 1L || is.na(result)) {
      stop("processor must return one non-missing character string", call. = FALSE)
    }
    result
  }, character(1), USE.NAMES = FALSE)
}

.stride_processed_pair <- function(query, target, processor) {
  list(
    query = .stride_apply_processor(query, processor),
    target = .stride_apply_processor(target, processor)
  )
}

partial_ratio <- function(query, target, processor = NULL) {
  values <- .stride_processed_pair(query, target, processor)
  .stridealign_call("stride_r_partial_ratio", values$query, values$target)
}

token_sort_ratio <- function(query, target, processor = NULL) {
  values <- .stride_processed_pair(query, target, processor)
  .stridealign_call("stride_r_token_sort_ratio", values$query, values$target)
}

token_set_ratio <- function(query, target, processor = NULL) {
  values <- .stride_processed_pair(query, target, processor)
  .stridealign_call("stride_r_token_set_ratio", values$query, values$target)
}

partial_token_sort_ratio <- function(query, target, processor = NULL) {
  values <- .stride_processed_pair(query, target, processor)
  .stridealign_call(
    "stride_r_partial_token_sort_ratio", values$query, values$target
  )
}

partial_token_set_ratio <- function(query, target, processor = NULL) {
  values <- .stride_processed_pair(query, target, processor)
  .stridealign_call(
    "stride_r_partial_token_set_ratio", values$query, values$target
  )
}

WRatio <- function(query, target, processor = NULL) {
  values <- .stride_processed_pair(query, target, processor)
  result <- .stridealign_call("stride_r_wratio", values$query, values$target)
  query_values <- if (length(values$query) == 1L) {
    rep(values$query, length(result))
  } else {
    values$query
  }
  target_values <- if (length(values$target) == 1L) {
    rep(values$target, length(result))
  } else {
    values$target
  }
  both_empty <- !is.na(query_values) & !is.na(target_values) &
    !nzchar(query_values) & !nzchar(target_values)
  result[both_empty] <- 1
  result
}

.stride_monge_direction <- function(left, right, scorer) {
  mean(vapply(left, function(token) {
    max(vapply(right, function(candidate) {
      score <- scorer(token, candidate)
      if (!is.numeric(score) || length(score) != 1L || is.na(score)) {
        stop("inner must return one non-missing numeric score", call. = FALSE)
      }
      as.numeric(score)
    }, numeric(1)))
  }, numeric(1)))
}

monge_elkan <- function(
  query,
  target,
  inner = "jaro",
  processor = NULL,
  symmetric = FALSE
) {
  query <- .stride_scalar_query(query)
  target <- .stride_scalar_query(target)
  values <- .stride_processed_pair(query, target, processor)
  query <- values$query
  target <- values$target
  scorer <- if (is.function(inner)) {
    inner
  } else {
    if (!is.character(inner) || length(inner) != 1L || is.na(inner)) {
      stop("inner must be a function or one inner-similarity name", call. = FALSE)
    }
    switch(
      inner,
      jaro = jaro_similarity,
      jaro_winkler = jaro_winkler_similarity,
      levenshtein_ratio = levenshtein_normalized_score,
      indel_ratio = indel_normalized_score,
      stop("unknown inner similarity: ", inner, call. = FALSE)
    )
  }
  left <- strsplit(query, "[[:space:]]+", perl = TRUE)[[1L]]
  right <- strsplit(target, "[[:space:]]+", perl = TRUE)[[1L]]
  left <- left[nzchar(left)]
  right <- right[nzchar(right)]
  if (!length(left) && !length(right)) return(1)
  if (!length(left) || !length(right)) return(0)
  forward <- .stride_monge_direction(left, right, scorer)
  if (!isTRUE(symmetric)) return(forward)
  (forward + .stride_monge_direction(right, left, scorer)) / 2
}

MetaphoneVariant <- list(PHILIPS = 0L, JELLYFISH = 1L)
DoubleMetaphoneVariant <- list(COMMONS = 0L, PYTHON = 1L)
BmpmRuleType <- list(APPROX = 0L, EXACT = 1L)

soundex <- function(input) .stridealign_call("stride_r_soundex", input)
soundex_equal <- function(query, target) {
  result <- .stridealign_call("stride_r_soundex_equal", query, target)
  .stride_phonetic_nonempty_equal(result, soundex(query), soundex(target))
}
metaphone <- function(input, variant = MetaphoneVariant$PHILIPS) {
  .stridealign_call("stride_r_metaphone", input, variant)
}
metaphone_equal <- function(query, target, variant = MetaphoneVariant$PHILIPS) {
  result <- .stridealign_call("stride_r_metaphone_equal", query, target, variant)
  .stride_phonetic_nonempty_equal(
    result, metaphone(query, variant), metaphone(target, variant)
  )
}
nysiis <- function(input) .stridealign_call("stride_r_nysiis", input)
nysiis_equal <- function(query, target) {
  result <- .stridealign_call("stride_r_nysiis_equal", query, target)
  .stride_phonetic_nonempty_equal(result, nysiis(query), nysiis(target))
}

.stride_phonetic_nonempty_equal <- function(result, query_code, target_code) {
  size <- length(result)
  if (length(query_code) == 1L) query_code <- rep(query_code, size)
  if (length(target_code) == 1L) target_code <- rep(target_code, size)
  valid <- !is.na(query_code) & !is.na(target_code) &
    nzchar(query_code) & nzchar(target_code)
  result[!is.na(result) & !valid] <- FALSE
  result
}
match_rating_codex <- function(input) {
  .stridealign_call("stride_r_match_rating_codex", input)
}
match_rating_compare <- function(query, target) {
  .stridealign_call("stride_r_match_rating_compare", query, target)
}
caverphone <- function(input) .stridealign_call("stride_r_caverphone", input)
cologne_phonetic <- function(input) {
  .stridealign_call("stride_r_cologne_phonetic", input)
}
daitch_mokotoff <- function(input, branching = TRUE, folding = TRUE) {
  .stridealign_call("stride_r_daitch_mokotoff", input, branching, folding)
}
double_metaphone <- function(
  input,
  max_length = 64,
  variant = DoubleMetaphoneVariant$COMMONS
) {
  result <- .stridealign_call(
    "stride_r_double_metaphone", input, max_length, variant
  )
  data.frame(
    primary = result[[1L]], alternate = result[[2L]],
    stringsAsFactors = FALSE
  )
}

.stride_bmpm_register <- function() {
  if (isTRUE(.stridealign_state$bmpm_registered)) return(invisible(NULL))
  root <- system.file("bmpm_data", package = "stridealign")
  files <- list.files(root, pattern = "[.]txt$", full.names = TRUE)
  if (!length(files)) {
    stop("stride-align BMPM rule resources are missing", call. = FALSE)
  }
  resources <- vapply(files, function(path) {
    readChar(path, file.info(path)$size, useBytes = TRUE)
  }, character(1), USE.NAMES = FALSE)
  names(resources) <- sub("[.]txt$", "", basename(files))
  .stridealign_call("stride_r_bmpm_register", resources)
  .stridealign_state$bmpm_registered <- TRUE
  invisible(NULL)
}

beider_morse <- function(
  input,
  rule_type = BmpmRuleType$APPROX,
  concat = TRUE,
  max_phonemes = 20
) {
  .stride_bmpm_register()
  .stridealign_call(
    "stride_r_beider_morse", input, rule_type, concat, max_phonemes
  )
}

dtw <- function(
  query,
  target,
  window = NULL,
  distance = NULL,
  score_cutoff = NULL
) {
  if (!is.numeric(query) || !is.numeric(target)) {
    stop("query and target must be numeric vectors", call. = FALSE)
  }
  if (typeof(query) != typeof(target)) {
    stop("query and target must share numeric storage type", call. = FALSE)
  }
  if (is.null(distance)) {
    distance <- if (is.integer(query)) "l1" else "l2_squared"
  }
  .stridealign_call(
    "stride_r_dtw", query, target, window, distance, score_cutoff
  )
}

dtw_distances <- function(
  query,
  targets,
  window = NULL,
  distance = NULL,
  score_cutoff = NULL
) {
  if (!is.list(targets)) {
    stop("targets must be a list of numeric vectors", call. = FALSE)
  }
  if (!is.numeric(query)) {
    stop("query must be a numeric vector", call. = FALSE)
  }
  if (length(targets) && any(vapply(
    targets,
    function(target) !is.numeric(target) || typeof(target) != typeof(query),
    logical(1)
  ))) {
    stop("every target must share the query's numeric storage type", call. = FALSE)
  }
  vapply(
    targets,
    function(target) dtw(query, target, window, distance, score_cutoff),
    numeric(1)
  )
}
