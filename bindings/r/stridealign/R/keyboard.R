# R cannot place U+0000 in a CHARSXP. The keyboard facade therefore uses the
# representable ASCII code points 1..127; matrix index i still equals the ASCII
# code point i, and DEL remains the wildcard slot.
.stride_keyboard_ascii_symbols <- intToUtf8(1:127, multiple = TRUE)
.stride_keyboard_ascii_alphabet <- paste0(
  .stride_keyboard_ascii_symbols,
  collapse = ""
)
.stride_keyboard_ascii_wildcard <- intToUtf8(127)

.stride_keyboard_read_npy <- function(path) {
  connection <- file(path, open = "rb")
  on.exit(close(connection), add = TRUE)
  magic <- readBin(connection, what = "raw", n = 6L)
  if (!identical(magic, as.raw(c(0x93, charToRaw("NUMPY"))))) {
    stop("not a NumPy .npy file", call. = FALSE)
  }
  version <- as.integer(readBin(connection, what = "raw", n = 2L))
  header_size <- if (version[[1L]] == 1L) {
    readBin(
      connection, what = integer(), n = 1L, size = 2L,
      signed = FALSE, endian = "little"
    )
  } else if (version[[1L]] %in% c(2L, 3L)) {
    readBin(
      connection, what = integer(), n = 1L, size = 4L,
      signed = FALSE, endian = "little"
    )
  } else {
    stop("unsupported NumPy .npy version", call. = FALSE)
  }
  header <- rawToChar(readBin(connection, what = "raw", n = header_size))
  if (!grepl("['\"]descr['\"]\\s*:\\s*['\"]\\|i1['\"]", header, perl = TRUE)) {
    stop("keyboard .npy matrix must use signed int8 values", call. = FALSE)
  }
  if (grepl("['\"]fortran_order['\"]\\s*:\\s*True", header, perl = TRUE)) {
    stop("Fortran-order .npy matrices are not supported", call. = FALSE)
  }
  shape_match <- regexec(
    "['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)", header, perl = TRUE
  )
  captured <- regmatches(header, shape_match)[[1L]]
  if (length(captured) != 2L) {
    stop("could not read shape from keyboard .npy file", call. = FALSE)
  }
  dimensions <- as.integer(trimws(strsplit(captured[[2L]], ",")[[1L]]))
  dimensions <- dimensions[!is.na(dimensions)]
  if (length(dimensions) != 2L) {
    stop("keyboard .npy value must be a two-dimensional matrix", call. = FALSE)
  }
  payload <- readBin(connection, what = "raw", n = prod(dimensions))
  if (length(payload) != prod(dimensions)) {
    stop("truncated keyboard .npy payload", call. = FALSE)
  }
  values <- as.integer(payload)
  values[values >= 128L] <- values[values >= 128L] - 256L
  matrix(values, nrow = dimensions[[1L]], ncol = dimensions[[2L]], byrow = TRUE)
}

.stride_keyboard_grid_for_alphabet <- function(grid, alphabet) {
  size <- nchar(alphabet, type = "chars")
  # Bundled Python artifacts include the unrepresentable NUL row/column.
  if (identical(dim(grid), c(128L, 128L)) && size == 127L) {
    grid <- grid[2:128, 2:128, drop = FALSE]
  }
  if (!identical(dim(grid), c(size, size))) {
    stop(
      "keyboard matrix shape does not match the alphabet size",
      call. = FALSE
    )
  }
  grid
}

keyboard_from_npy <- function(
  path,
  name = tools::file_path_sans_ext(basename(path)),
  alphabet = keyboard$ASCII_ALPHABET,
  wildcard = keyboard$ASCII_WILDCARD,
  transpose = FALSE,
  gap_score = -1,
  gap_open = NULL,
  gap_extend = NULL
) {
  if (!is.character(path) || length(path) != 1L || is.na(path)) {
    stop("path must be one non-missing filename", call. = FALSE)
  }
  grid <- .stride_keyboard_grid_for_alphabet(
    .stride_keyboard_read_npy(path), alphabet
  )
  if (isTRUE(transpose)) grid <- t(grid)
  SubstitutionMatrix(
    name = name,
    alphabet = alphabet,
    matrix = grid,
    gap_score = gap_score,
    wildcard = wildcard,
    gap_open = gap_open,
    gap_extend = gap_extend
  )
}

.stride_keyboard_counts_grid <- function(counts, alphabet) {
  symbols <- strsplit(alphabet, "", fixed = TRUE)[[1L]]
  size <- length(symbols)
  if (is.matrix(counts)) {
    if (!is.numeric(counts) || !identical(dim(counts), c(size, size))) {
      stop("counts matrix shape does not match the alphabet", call. = FALSE)
    }
    if (anyNA(counts) || any(!is.finite(counts)) || any(counts < 0)) {
      stop("counts must be finite non-negative numbers", call. = FALSE)
    }
    return(matrix(as.numeric(counts), size, size))
  }
  if (!is.data.frame(counts) ||
      !all(c("typed", "intended", "count") %in% names(counts))) {
    stop(
      "counts must be a numeric matrix or a data frame with typed, intended, and count columns",
      call. = FALSE
    )
  }
  if (!is.character(counts$typed) || !is.character(counts$intended) ||
      !is.numeric(counts$count) || anyNA(counts$count) ||
      any(!is.finite(counts$count)) || any(counts$count < 0)) {
    stop("confusion-count rows are invalid", call. = FALSE)
  }
  grid <- matrix(0, size, size)
  typed <- match(counts$typed, symbols)
  intended <- match(counts$intended, symbols)
  keep <- !is.na(typed) & !is.na(intended)
  if (any(keep)) {
    for (index in which(keep)) {
      grid[typed[[index]], intended[[index]]] <- counts$count[[index]]
    }
  }
  grid
}

keyboard_from_confusion_counts <- function(
  counts,
  alphabet = keyboard$ASCII_ALPHABET,
  name = "KEYBOARD",
  scale = 2,
  match_margin = 4,
  floor = NULL,
  wildcard = substr(alphabet, nchar(alphabet), nchar(alphabet)),
  gap_score = -1,
  gap_open = NULL,
  gap_extend = NULL
) {
  grid_counts <- .stride_keyboard_counts_grid(counts, alphabet)
  size <- nrow(grid_counts)
  total <- sum(grid_counts)
  if (total == 0) {
    grid <- matrix(-1L, size, size)
    diag(grid) <- 1L
  } else {
    row_totals <- rowSums(grid_counts)
    column_totals <- colSums(grid_counts)
    denominator <- outer(row_totals, column_totals)
    scores <- scale * log2((grid_counts * total) / denominator)
    finite <- is.finite(scores)
    resolved_floor <- if (is.null(floor)) {
      if (any(finite)) base::floor(min(scores[finite])) else -1
    } else {
      floor
    }
    scores[!finite] <- resolved_floor
    grid <- matrix(
      pmax(-128, pmin(127, round(scores))),
      nrow = size,
      ncol = size
    )
    diag(grid) <- -128
    best_substitution <- max(grid)
    diag(grid) <- min(127, max(best_substitution, 0) + match_margin)
    storage.mode(grid) <- "integer"
  }
  SubstitutionMatrix(
    name = name,
    alphabet = alphabet,
    matrix = grid,
    gap_score = gap_score,
    wildcard = wildcard,
    gap_open = gap_open,
    gap_extend = gap_extend
  )
}

keyboard_available <- function() {
  directory <- system.file("keyboard_data", package = "stridealign")
  if (!nzchar(directory) || !dir.exists(directory)) return(character())
  sort(tools::file_path_sans_ext(basename(list.files(
    directory, pattern = "[.]npy$", full.names = FALSE
  ))))
}

keyboard <- new.env(parent = emptyenv())
keyboard$ASCII_ALPHABET <- .stride_keyboard_ascii_alphabet
keyboard$ASCII_CODEPOINTS <- 1:127
keyboard$ASCII_WILDCARD <- .stride_keyboard_ascii_wildcard
keyboard$from_confusion_counts <- keyboard_from_confusion_counts
keyboard$from_npy <- keyboard_from_npy
keyboard$available <- keyboard_available

.stride_keyboard_load_bundled <- function() {
  directory <- system.file("keyboard_data", package = "stridealign")
  for (name in keyboard_available()) {
    assign(
      name,
      keyboard_from_npy(
        file.path(directory, paste0(name, ".npy")),
        name = paste0("keyboard:", name)
      ),
      envir = keyboard
    )
  }
  invisible(NULL)
}
