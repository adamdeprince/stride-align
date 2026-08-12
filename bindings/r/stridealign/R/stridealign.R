.stridealign_state <- new.env(parent = emptyenv())

.stridealign_registered_calls <- c(
  "stride_r_levenshtein",
  "stride_r_levenshtein_similarity",
  "stride_r_osa",
  "stride_r_osa_similarity",
  "stride_r_true_damerau_levenshtein",
  "stride_r_true_damerau_levenshtein_similarity",
  "stride_r_indel",
  "stride_r_indel_similarity",
  "stride_r_hamming",
  "stride_r_hamming_similarity",
  "stride_r_jaro",
  "stride_r_jaro_winkler",
  "stride_r_smith_waterman",
  "stride_r_needleman_wunsch",
  "stride_r_smith_waterman_affine",
  "stride_r_needleman_wunsch_affine"
)

.stridealign_symbol <- function(name) {
  symbol <- .stridealign_state$symbols[[name]]
  if (is.null(symbol)) {
    stop("stride-align native routine is not loaded: ", name, call. = FALSE)
  }
  symbol
}

.stridealign_call <- function(name, ...) {
  .Call(.stridealign_symbol(name), ...)
}

.onLoad <- function(libname, pkgname) {
  baseline <- getLoadedDLLs()[["stridealign"]]
  if (is.null(baseline)) {
    stop("stride-align baseline native library was not loaded", call. = FALSE)
  }

  candidate_symbol <- getNativeSymbolInfo(
    "stride_r_backend_candidates",
    PACKAGE = baseline
  )$address
  candidates <- .Call(candidate_symbol)
  selected <- "generic"
  selected_dll <- baseline
  loaded_name <- NULL
  requested <- trimws(Sys.getenv("STRIDE_ALIGN_R_BACKEND", unset = ""))

  if (nzchar(requested) && !requested %in% candidates) {
    stop(
      "requested stride-align backend is unavailable or incompatible: ",
      requested,
      "; compatible packaged backends: ",
      paste(candidates, collapse = ", "),
      call. = FALSE
    )
  }

  if (nzchar(requested)) {
    dispatch_order <- requested
  } else {
    dispatch_order <- candidates
  }

  for (candidate in dispatch_order) {
    if (identical(candidate, "generic")) {
      break
    }
    dll_name <- paste0("stridealign_", candidate)
    candidate_dll <- tryCatch(
      library.dynam(dll_name, pkgname, libname),
      error = function(error) NULL
    )
    if (!is.null(candidate_dll)) {
      selected <- candidate
      selected_dll <- candidate_dll
      loaded_name <- dll_name
      break
    }
    if (nzchar(requested)) {
      stop(
        "failed to load requested stride-align backend: ", requested,
        call. = FALSE
      )
    }
  }

  registered <- getDLLRegisteredRoutines(selected_dll)[[".Call"]]
  missing <- setdiff(.stridealign_registered_calls, names(registered))
  if (length(missing)) {
    if (!is.null(loaded_name)) {
      library.dynam.unload(loaded_name, libname)
    }
    stop(
      "stride-align backend is missing registered routines: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }

  .stridealign_state$backend <- selected
  .stridealign_state$available <- candidates
  .stridealign_state$loaded_name <- loaded_name
  .stridealign_state$symbols <- lapply(
    registered[.stridealign_registered_calls],
    function(symbol) symbol$address
  )
}

.onUnload <- function(libpath) {
  loaded_name <- .stridealign_state$loaded_name
  if (!is.null(loaded_name)) {
    library.dynam.unload(loaded_name, libpath)
  }
}

#' Report the selected native backend
#'
#' @return A single character value naming the SIMD backend selected when the
#'   package namespace was loaded.
#' @export
stride_backend <- function() {
  .stridealign_state$backend
}

#' Report CPU-compatible backends included in the package
#'
#' @return A character vector in dispatch-priority order.
#' @export
stride_available_backends <- function() {
  .stridealign_state$available
}

#' Compute Levenshtein distance
#' @param query,target Character vectors. One input may have length one and is
#'   then broadcast across the other input.
#' @return A numeric vector. Missing input values produce missing results.
#' @export
stride_levenshtein <- function(query, target) {
  .stridealign_call("stride_r_levenshtein", query, target)
}

#' @rdname stride_levenshtein
#' @export
stride_levenshtein_similarity <- function(query, target) {
  .stridealign_call("stride_r_levenshtein_similarity", query, target)
}

#' @rdname stride_levenshtein
#' @export
stride_osa <- function(query, target) {
  .stridealign_call("stride_r_osa", query, target)
}

#' @rdname stride_levenshtein
#' @export
stride_osa_similarity <- function(query, target) {
  .stridealign_call("stride_r_osa_similarity", query, target)
}

#' @rdname stride_levenshtein
#' @export
stride_true_damerau_levenshtein <- function(query, target) {
  .stridealign_call("stride_r_true_damerau_levenshtein", query, target)
}

#' @rdname stride_levenshtein
#' @export
stride_true_damerau_levenshtein_similarity <- function(query, target) {
  .stridealign_call(
    "stride_r_true_damerau_levenshtein_similarity",
    query,
    target
  )
}

#' @rdname stride_levenshtein
#' @export
stride_indel <- function(query, target) {
  .stridealign_call("stride_r_indel", query, target)
}

#' @rdname stride_levenshtein
#' @export
stride_indel_similarity <- function(query, target) {
  .stridealign_call("stride_r_indel_similarity", query, target)
}

#' @rdname stride_levenshtein
#' @export
stride_hamming <- function(query, target) {
  .stridealign_call("stride_r_hamming", query, target)
}

#' @rdname stride_levenshtein
#' @export
stride_hamming_similarity <- function(query, target) {
  .stridealign_call("stride_r_hamming_similarity", query, target)
}

#' @rdname stride_levenshtein
#' @export
stride_jaro <- function(query, target) {
  .stridealign_call("stride_r_jaro", query, target)
}

#' Compute Jaro-Winkler similarity
#' @inheritParams stride_levenshtein
#' @param prefix_weight Numeric prefix bonus, normally `0.1`.
#' @param prefix_threshold Minimum Jaro score before applying the prefix bonus.
#' @param prefix_cap Maximum number of prefix code points to reward.
#' @return A numeric vector.
#' @export
stride_jaro_winkler <- function(
  query,
  target,
  prefix_weight = 0.1,
  prefix_threshold = 0.7,
  prefix_cap = 4
) {
  .stridealign_call(
    "stride_r_jaro_winkler",
    query,
    target,
    prefix_weight,
    prefix_threshold,
    prefix_cap
  )
}

#' Compute linear-gap sequence-alignment scores
#' @inheritParams stride_levenshtein
#' @param match_score Score for matching code points.
#' @param mismatch_score Score for mismatching code points.
#' @param gap_score Linear gap score.
#' @return A numeric vector.
#' @export
stride_smith_waterman <- function(
  query,
  target,
  match_score = 2,
  mismatch_score = -1,
  gap_score = -1
) {
  .stridealign_call(
    "stride_r_smith_waterman",
    query,
    target,
    match_score,
    mismatch_score,
    gap_score
  )
}

#' @rdname stride_smith_waterman
#' @export
stride_needleman_wunsch <- function(
  query,
  target,
  match_score = 2,
  mismatch_score = -1,
  gap_score = -1
) {
  .stridealign_call(
    "stride_r_needleman_wunsch",
    query,
    target,
    match_score,
    mismatch_score,
    gap_score
  )
}

#' Compute affine-gap sequence-alignment scores
#' @inheritParams stride_smith_waterman
#' @param gap_open_score Score for opening a gap.
#' @param gap_extend_score Score for extending a gap.
#' @return A numeric vector.
#' @export
stride_smith_waterman_affine <- function(
  query,
  target,
  match_score = 2,
  mismatch_score = -1,
  gap_open_score = -2,
  gap_extend_score = -1
) {
  .stridealign_call(
    "stride_r_smith_waterman_affine",
    query,
    target,
    match_score,
    mismatch_score,
    gap_open_score,
    gap_extend_score
  )
}

#' @rdname stride_smith_waterman_affine
#' @export
stride_needleman_wunsch_affine <- function(
  query,
  target,
  match_score = 2,
  mismatch_score = -1,
  gap_open_score = -2,
  gap_extend_score = -1
) {
  .stridealign_call(
    "stride_r_needleman_wunsch_affine",
    query,
    target,
    match_score,
    mismatch_score,
    gap_open_score,
    gap_extend_score
  )
}
