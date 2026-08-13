AlignmentPath <- function(
  score = 0,
  query_start = 0,
  query_end = 0,
  target_start = 0,
  target_end = 0,
  operations = "",
  cigar = "",
  matches = 0,
  mismatches = 0,
  insertions = 0,
  deletions = 0,
  aligned_length = nchar(operations)
) {
  structure(list(
    score = score,
    query_start = query_start,
    query_end = query_end,
    target_start = target_start,
    target_end = target_end,
    operations = operations,
    cigar = cigar,
    matches = matches,
    mismatches = mismatches,
    insertions = insertions,
    deletions = deletions,
    aligned_length = aligned_length
  ), class = c("stride_alignment_path", "list"))
}

AlignmentResult <- function(..., aligned_query = "", aligned_target = "") {
  result <- AlignmentPath(...)
  result$aligned <- c(query = aligned_query, target = aligned_target)
  class(result) <- c("stride_alignment_result", "stride_alignment_path", "list")
  result
}

.stride_path <- function(
  query,
  target,
  local,
  match_score,
  mismatch_score,
  gap_score,
  gap_open_score,
  gap_extend_score,
  width
) {
  if (!is.null(width) && !width %in% c(0, 8, 16, 32, 64)) {
    stop("width must be NULL, 0, 8, 16, 32, or 64", call. = FALSE)
  }
  gaps <- .stride_resolve_gaps(gap_score, gap_open_score, gap_extend_score)
  .stridealign_call(
    "stride_r_alignment_path",
    query,
    target,
    local,
    match_score,
    mismatch_score,
    gaps$open,
    gaps$extend
  )
}

.stride_materialize_path <- function(path, query, target) {
  query_points <- strsplit(query, "", fixed = TRUE)[[1L]]
  target_points <- strsplit(target, "", fixed = TRUE)[[1L]]
  operations <- if (nzchar(path$operations)) {
    strsplit(path$operations, "", fixed = TRUE)[[1L]]
  } else {
    character()
  }
  qi <- path$query_start + 1L
  ti <- path$target_start + 1L
  aligned_query <- character()
  aligned_target <- character()
  for (operation in operations) {
    if (operation %in% c("=", "X")) {
      aligned_query <- c(aligned_query, query_points[[qi]])
      aligned_target <- c(aligned_target, target_points[[ti]])
      qi <- qi + 1L
      ti <- ti + 1L
    } else if (operation == "D") {
      aligned_query <- c(aligned_query, query_points[[qi]])
      aligned_target <- c(aligned_target, "-")
      qi <- qi + 1L
    } else {
      aligned_query <- c(aligned_query, "-")
      aligned_target <- c(aligned_target, target_points[[ti]])
      ti <- ti + 1L
    }
  }
  path$aligned <- c(
    query = paste0(aligned_query, collapse = ""),
    target = paste0(aligned_target, collapse = "")
  )
  class(path) <- c("stride_alignment_result", "stride_alignment_path", "list")
  path
}

smith_waterman_path <- function(
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
    path <- .stride_matrix_alignment(
      query, target, matrix, TRUE, gap_score, gap_open_score,
      gap_extend_score, traceback = TRUE
    )
    return(.stride_materialize_path(path, query, target))
  }
  .stride_path(
    query, target, TRUE, match_score, mismatch_score, gap_score,
    gap_open_score, gap_extend_score, width
  )
}

smith_waterman_path_info <- function(...) {
  result <- smith_waterman_path(...)
  result$aligned <- NULL
  class(result) <- c("stride_alignment_path", "list")
  result
}

needleman_wunsch_path <- function(
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
    path <- .stride_matrix_alignment(
      query, target, matrix, FALSE, gap_score, gap_open_score,
      gap_extend_score, traceback = TRUE
    )
    return(.stride_materialize_path(path, query, target))
  }
  .stride_path(
    query, target, FALSE, match_score, mismatch_score, gap_score,
    gap_open_score, gap_extend_score, width
  )
}

needleman_wunsch_path_info <- function(...) {
  result <- needleman_wunsch_path(...)
  result$aligned <- NULL
  class(result) <- c("stride_alignment_path", "list")
  result
}

smith_waterman_cigar <- function(...) smith_waterman_path_info(...)$cigar
smith_waterman_trace_cigar <- smith_waterman_cigar
smith_waterman_trade_cigar <- smith_waterman_cigar
needleman_wunsch_cigar <- function(...) needleman_wunsch_path_info(...)$cigar
needleman_wunsch_trace_cigar <- needleman_wunsch_cigar
needleman_wunsch_trade_cigar <- needleman_wunsch_cigar

Scores <- function(
  query,
  variant = c("smith_waterman", "needleman_wunsch", "smith_waterman_farrar", "farrar"),
  match_score = 2,
  mismatch_score = -1,
  gap_score = -1,
  gap_open_score = NULL,
  gap_extend_score = NULL,
  width = NULL
) {
  query <- .stride_scalar_query(query)
  variant <- match.arg(variant)
  if (variant == "farrar") variant <- "smith_waterman_farrar"
  compare <- switch(
    variant,
    smith_waterman = smith_waterman_scores,
    needleman_wunsch = needleman_wunsch_scores,
    smith_waterman_farrar = smith_waterman_farrar_scores
  )
  structure(list(
    query = query,
    variant = variant,
    compare = function(targets) compare(
      query,
      targets,
      match_score = match_score,
      mismatch_score = mismatch_score,
      gap_score = gap_score,
      gap_open_score = gap_open_score,
      gap_extend_score = gap_extend_score,
      width = width
    )
  ), class = "stride_scores")
}

print.stride_alignment_path <- function(x, ...) {
  cat(
    "<stride_alignment_path score=", x$score,
    " query=[", x$query_start, ",", x$query_end, ")",
    " target=[", x$target_start, ",", x$target_end, ")",
    " cigar=\"", x$cigar, "\">\n",
    sep = ""
  )
  invisible(x)
}

print.stride_alignment_result <- print.stride_alignment_path
