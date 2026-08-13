SubstitutionMatrix <- function(
  name,
  alphabet,
  matrix,
  gap_score = -4,
  wildcard = "X",
  gap_open = NULL,
  gap_extend = NULL
) {
  if (!is.character(name) || length(name) != 1L || is.na(name)) {
    stop("name must be one character string", call. = FALSE)
  }
  if (!is.character(alphabet) || length(alphabet) != 1L || is.na(alphabet)) {
    stop("alphabet must be one character string", call. = FALSE)
  }
  symbols <- strsplit(alphabet, "", fixed = TRUE)[[1L]]
  if (anyDuplicated(symbols)) stop("alphabet has duplicate symbols", call. = FALSE)
  if (!is.matrix(matrix) || !is.numeric(matrix) ||
      !identical(dim(matrix), c(length(symbols), length(symbols)))) {
    stop("matrix dimensions must match the alphabet", call. = FALSE)
  }
  if (anyNA(matrix) || any(!is.finite(matrix)) || any(matrix != trunc(matrix))) {
    stop("matrix must contain finite integer scores", call. = FALSE)
  }
  if (any(matrix < -128 | matrix > 127)) {
    stop("matrix scores must fit in signed 8-bit integers", call. = FALSE)
  }
  if (!is.character(wildcard) || length(wildcard) != 1L ||
      nchar(wildcard, type = "chars") != 1L || !wildcard %in% symbols) {
    stop("wildcard must be one symbol in alphabet", call. = FALSE)
  }
  values <- matrix
  storage.mode(values) <- "integer"
  result <- structure(list(
    name = name,
    alphabet = alphabet,
    matrix = values,
    gap_score = gap_score,
    wildcard = wildcard,
    gap_open = gap_open,
    gap_extend = gap_extend,
    stride = length(symbols),
    max_abs = if (length(values)) max(abs(values)) else 0L,
    # R matrices are column-major; transpose before flattening so this is the
    # same stable row-major int8 snapshot consumed by the other bindings.
    matrix_bytes = as.raw(bitwAnd(as.integer(as.vector(t(values))), 255L))
  ), class = "stride_substitution_matrix")
  matrix_snapshot <- result
  result$encode <- function(sequence) {
    substitution_matrix_encode(matrix_snapshot, sequence)
  }
  result$score <- function(query, target) {
    substitution_matrix_score(matrix_snapshot, query, target)
  }
  result$score_step_limit <- function(
    gap_score = NULL,
    gap_open = NULL,
    gap_extend = NULL
  ) {
    substitution_matrix_score_step_limit(
      matrix_snapshot, gap_score, gap_open, gap_extend
    )
  }
  result
}

substitution_matrix_encode <- function(substitution_matrix, sequence) {
  if (!inherits(substitution_matrix, "stride_substitution_matrix")) {
    stop("matrix must be a SubstitutionMatrix", call. = FALSE)
  }
  if (!is.character(sequence)) {
    stop("sequence must be a character vector", call. = FALSE)
  }
  values <- lapply(sequence, function(value) {
    if (is.na(value)) return(NA_integer_)
    .stride_matrix_encode(value, substitution_matrix) - 1L
  })
  if (length(values) == 1L) values[[1L]] else values
}

substitution_matrix_score <- function(substitution_matrix, query, target) {
  if (!inherits(substitution_matrix, "stride_substitution_matrix")) {
    stop("matrix must be a SubstitutionMatrix", call. = FALSE)
  }
  smith_waterman_score(
    query, target, matrix = substitution_matrix,
    gap_score = substitution_matrix$gap_score
  )
}

substitution_matrix_score_step_limit <- function(
  substitution_matrix,
  gap_score = NULL,
  gap_open = NULL,
  gap_extend = NULL
) {
  if (!inherits(substitution_matrix, "stride_substitution_matrix")) {
    stop("matrix must be a SubstitutionMatrix", call. = FALSE)
  }
  if (is.null(gap_score) && is.null(gap_open) && is.null(gap_extend)) {
    gap_score <- substitution_matrix$gap_score
    gap_open <- substitution_matrix$gap_open
    gap_extend <- substitution_matrix$gap_extend
  }
  candidates <- substitution_matrix$max_abs
  for (value in list(gap_score, gap_open, gap_extend)) {
    if (!is.null(value)) candidates <- c(candidates, abs(value))
  }
  max(candidates)
}

substitution_matrix_transpose <- function(
  substitution_matrix,
  name = paste0(substitution_matrix$name, ".T")
) {
  if (!inherits(substitution_matrix, "stride_substitution_matrix")) {
    stop("matrix must be a SubstitutionMatrix", call. = FALSE)
  }
  SubstitutionMatrix(
    name = name,
    alphabet = substitution_matrix$alphabet,
    matrix = t(substitution_matrix$matrix),
    gap_score = substitution_matrix$gap_score,
    wildcard = substitution_matrix$wildcard,
    gap_open = substitution_matrix$gap_open,
    gap_extend = substitution_matrix$gap_extend
  )
}

substitution_matrix_from_ncbi_text <- function(
  text,
  name = "<unnamed>",
  gap_score = -4,
  wildcard = "X",
  gap_open = NULL,
  gap_extend = NULL
) {
  if (!is.character(text) || length(text) != 1L || is.na(text)) {
    stop("text must be one non-missing character string", call. = FALSE)
  }
  lines <- strsplit(text, "\n", fixed = TRUE)[[1L]]
  lines <- trimws(lines)
  lines <- lines[nzchar(lines) & !startsWith(lines, "#")]
  if (length(lines) < 2L) {
    stop("NCBI matrix text has no header or data rows", call. = FALSE)
  }
  header <- strsplit(lines[[1L]], "[[:space:]]+")[[1L]]
  rows <- strsplit(lines[-1L], "[[:space:]]+")
  labels <- vapply(rows, `[[`, character(1), 1L)
  if (!identical(labels, header)) {
    stop("NCBI matrix row labels do not match column header", call. = FALSE)
  }
  values <- lapply(rows, function(row) {
    parsed <- suppressWarnings(as.numeric(row[-1L]))
    if (length(parsed) != length(header)) {
      stop("NCBI matrix row has the wrong number of values", call. = FALSE)
    }
    if (anyNA(parsed)) {
      stop("non-integer score in NCBI matrix row", call. = FALSE)
    }
    parsed
  })
  grid <- t(vapply(values, identity, numeric(length(header))))
  SubstitutionMatrix(
    name = name,
    alphabet = paste0(header, collapse = ""),
    matrix = grid,
    gap_score = gap_score,
    wildcard = wildcard,
    gap_open = gap_open,
    gap_extend = gap_extend
  )
}

identity_matrix <- function(
  alphabet,
  match = 1,
  mismatch = 0,
  wildcard = "*",
  name = "IDENTITY",
  gap_score = -1,
  gap_open = NULL,
  gap_extend = NULL
) {
  symbols <- strsplit(alphabet, "", fixed = TRUE)[[1L]]
  if (!wildcard %in% symbols) symbols <- c(symbols, wildcard)
  values <- matrix(mismatch, length(symbols), length(symbols))
  diag(values) <- match
  wildcard_index <- base::match(wildcard, symbols)
  values[wildcard_index, ] <- mismatch
  values[, wildcard_index] <- mismatch
  SubstitutionMatrix(
    name, paste0(symbols, collapse = ""), values, gap_score, wildcard,
    gap_open, gap_extend
  )
}

ascii_matrix <- function(
  match = 1,
  mismatch = -1,
  name = "ASCII",
  gap_score = -1,
  gap_open = NULL,
  gap_extend = NULL
) {
  # R character strings cannot name U+0000 as a one-character alphabet
  # member, so the R catalog covers the representable ASCII range 1..127.
  symbols <- intToUtf8(1:127, multiple = TRUE)
  values <- matrix(mismatch, 127L, 127L)
  diag(values) <- match
  values[127L, ] <- mismatch
  values[, 127L] <- mismatch
  SubstitutionMatrix(
    name, paste0(symbols, collapse = ""), values, gap_score, symbols[[127L]],
    gap_open, gap_extend
  )
}

.stride_matrix_defaults <- function(name) {
  if (startsWith(name, "BLOSUM")) {
    level <- as.integer(sub("BLOSUM", "", name, fixed = TRUE))
    if (level >= 80) return(c(-5, -10, -1))
    if (level >= 62) return(c(-5, -11, -1))
    if (level >= 50) return(c(-6, -13, -2))
    return(c(-6, -14, -2))
  }
  if (startsWith(name, "PAM")) {
    level <- as.integer(sub("PAM", "", name, fixed = TRUE))
    if (level <= 30) return(c(-5, -9, -1))
    if (level <= 120) return(c(-6, -10, -1))
    if (level <= 200) return(c(-7, -12, -1))
    return(c(-8, -14, -2))
  }
  c(-5, -10, -2)
}

.stride_read_ncbi_matrix <- function(filename, name = filename, wildcard = "X") {
  path <- system.file("matrix_data", filename, package = "stridealign")
  defaults <- .stride_matrix_defaults(name)
  substitution_matrix_from_ncbi_text(
    paste(readLines(path, warn = FALSE), collapse = "\n"),
    name = name,
    gap_score = defaults[[1L]],
    wildcard = wildcard,
    gap_open = defaults[[2L]],
    gap_extend = defaults[[3L]]
  )
}

.stride_matrix_catalog <- new.env(parent = emptyenv())
.stride_matrix_names <- c(
  paste0("blosum", c(30, 35, 40, 45, 50, 55, 60, 62, 65, 70, 75, 80, 85, 90, 100)),
  paste0("pam", seq(10, 500, 10)),
  "nuc44", "dna_match", "ascii_text"
)

.stride_matrix_catalog_load <- function(name) {
  if (exists(name, envir = .stride_matrix_catalog, inherits = FALSE)) {
    return(get(name, envir = .stride_matrix_catalog, inherits = FALSE))
  }
  value <- if (name == "nuc44") {
    .stride_read_ncbi_matrix("NUC.4.4", "NUC.4.4", "N")
  } else if (name == "dna_match") {
    identity_matrix("ACGT", 5, -4, "N", "DNA_MATCH", -5)
  } else if (name == "ascii_text") {
    ascii_matrix(name = "ASCII")
  } else if (startsWith(name, "blosum")) {
    label <- toupper(name)
    .stride_read_ncbi_matrix(label, label)
  } else if (startsWith(name, "pam")) {
    label <- toupper(name)
    .stride_read_ncbi_matrix(label, label)
  } else {
    stop("unknown substitution matrix: ", name, call. = FALSE)
  }
  assign(name, value, envir = .stride_matrix_catalog)
  value
}

for (.stride_name in .stride_matrix_names) {
  local({
    matrix_name <- .stride_name
    delayedAssign(
      matrix_name,
      .stride_matrix_catalog_load(matrix_name),
      assign.env = parent.env(environment()),
      eval.env = environment()
    )
  })
}
rm(.stride_name)

.stride_matrix_encode <- function(input, substitution_matrix) {
  symbols <- strsplit(substitution_matrix$alphabet, "", fixed = TRUE)[[1L]]
  wildcard <- match(substitution_matrix$wildcard, symbols)
  values <- strsplit(input, "", fixed = TRUE)[[1L]]
  encoded <- match(values, symbols)
  encoded[is.na(encoded)] <- wildcard
  encoded
}

.stride_matrix_alignment <- function(
  query,
  target,
  substitution_matrix,
  local,
  gap_score = NULL,
  gap_open_score = NULL,
  gap_extend_score = NULL,
  traceback = FALSE
) {
  query <- .stride_scalar_query(query)
  target <- .stride_scalar_query(target)
  if (!inherits(substitution_matrix, "stride_substitution_matrix")) {
    stop("matrix must be a SubstitutionMatrix", call. = FALSE)
  }
  q <- .stride_matrix_encode(query, substitution_matrix)
  t <- .stride_matrix_encode(target, substitution_matrix)
  linear_gap <- if (is.null(gap_score)) substitution_matrix$gap_score else gap_score
  open <- if (is.null(gap_open_score)) linear_gap else gap_open_score
  extend <- if (is.null(gap_extend_score)) open else gap_extend_score
  rows <- length(q) + 1L
  columns <- length(t) + 1L
  negative_infinity <- -(.Machine$double.xmax / 4)
  h <- matrix(if (local) 0 else negative_infinity, rows, columns)
  vertical <- matrix(negative_infinity, rows, columns)
  horizontal <- matrix(negative_infinity, rows, columns)
  h[1L, 1L] <- 0
  gap_cost <- function(length) if (length == 0L) 0 else open + (length - 1L) * extend
  if (!local) {
    if (rows > 1L) for (row in 2:rows) {
      h[row, 1L] <- gap_cost(row - 1L)
      vertical[row, 1L] <- h[row, 1L]
    }
    if (columns > 1L) for (column in 2:columns) {
      h[1L, column] <- gap_cost(column - 1L)
      horizontal[1L, column] <- h[1L, column]
    }
  }
  best <- c(score = 0, row = 1, column = 1)
  if (rows > 1L && columns > 1L) for (row in 2:rows) for (column in 2:columns) {
    vertical[row, column] <- max(
      h[row - 1L, column] + open,
      vertical[row - 1L, column] + extend
    )
    horizontal[row, column] <- max(
      h[row, column - 1L] + open,
      horizontal[row, column - 1L] + extend
    )
    diagonal <- h[row - 1L, column - 1L] +
      substitution_matrix$matrix[q[[row - 1L]], t[[column - 1L]]]
    h[row, column] <- max(diagonal, vertical[row, column], horizontal[row, column])
    if (local) h[row, column] <- max(0, h[row, column])
    if (local && h[row, column] > best[["score"]]) {
      best <- c(score = h[row, column], row = row, column = column)
    }
  }
  if (!local) best <- c(score = h[rows, columns], row = rows, column = columns)
  if (!traceback) return(unname(best[["score"]]))

  row <- as.integer(best[["row"]])
  column <- as.integer(best[["column"]])
  query_end <- row - 1L
  target_end <- column - 1L
  operations <- character()
  state <- "h"
  while (row > 1L || column > 1L) {
    if (local && state == "h" && h[row, column] <= 0) break
    if (state == "h") {
      if (row > 1L && column > 1L) {
        diagonal <- h[row - 1L, column - 1L] +
          substitution_matrix$matrix[q[[row - 1L]], t[[column - 1L]]]
        if (h[row, column] == diagonal) {
          operations <- c(if (q[[row - 1L]] == t[[column - 1L]]) "=" else "X", operations)
          row <- row - 1L
          column <- column - 1L
          next
        }
      }
      if (row > 1L && h[row, column] == vertical[row, column]) {
        state <- "vertical"
        next
      }
      if (column > 1L && h[row, column] == horizontal[row, column]) {
        state <- "horizontal"
        next
      }
      break
    }
    if (state == "vertical") {
      operations <- c("D", operations)
      continuing <- row > 2L &&
        vertical[row, column] == vertical[row - 1L, column] + extend
      row <- row - 1L
      state <- if (continuing) "vertical" else "h"
    } else {
      operations <- c("I", operations)
      continuing <- column > 2L &&
        horizontal[row, column] == horizontal[row, column - 1L] + extend
      column <- column - 1L
      state <- if (continuing) "horizontal" else "h"
    }
  }
  ops <- paste0(operations, collapse = "")
  if (!length(operations)) {
    cigar <- ""
  } else {
    runs <- rle(operations)
    cigar <- paste0(runs$lengths, runs$values, collapse = "")
  }
  AlignmentPath(
    best[["score"]], row - 1L, query_end, column - 1L, target_end,
    ops, cigar,
    sum(operations == "="), sum(operations == "X"),
    sum(operations == "I"), sum(operations == "D"), length(operations)
  )
}

.stride_matrix_pairwise <- function(
  query,
  target,
  substitution_matrix,
  local,
  gap_score = NULL,
  gap_open_score = NULL,
  gap_extend_score = NULL
) {
  if (!is.character(query) || !is.character(target)) {
    stop("query and target must be character vectors", call. = FALSE)
  }
  query_size <- length(query)
  target_size <- length(target)
  output_size <- if (query_size == target_size) {
    query_size
  } else if (query_size == 1L) {
    target_size
  } else if (target_size == 1L) {
    query_size
  } else {
    stop(
      "query and target must have equal lengths, or one must have length one",
      call. = FALSE
    )
  }
  if (output_size == 0L) return(numeric())
  output <- numeric(output_size)
  for (index in seq_len(output_size)) {
    query_index <- if (query_size == 1L) 1L else index
    target_index <- if (target_size == 1L) 1L else index
    if (is.na(query[[query_index]]) || is.na(target[[target_index]])) {
      output[[index]] <- NA_real_
    } else {
      output[[index]] <- .stride_matrix_alignment(
        query[[query_index]],
        target[[target_index]],
        substitution_matrix,
        local,
        gap_score,
        gap_open_score,
        gap_extend_score
      )
    }
  }
  output
}
