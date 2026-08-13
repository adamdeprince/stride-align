library(stridealign)

close_to <- function(actual, expected, tolerance = 1e-12) {
  stopifnot(isTRUE(all.equal(
    unname(actual), unname(expected), tolerance = tolerance,
    check.attributes = FALSE
  )))
}

expect_error <- function(expression, pattern = NULL, fixed = TRUE) {
  message <- tryCatch(
    {
      force(expression)
      NULL
    },
    error = conditionMessage
  )
  stopifnot(!is.null(message))
  if (!is.null(pattern)) stopifnot(grepl(pattern, message, fixed = fixed))
  invisible(message)
}

random_string <- function(length, alphabet = letters) {
  if (length == 0L) return("")
  paste0(sample(alphabet, length, replace = TRUE), collapse = "")
}

reference_lcs_length <- function(left, right) {
  left <- utf8ToInt(left)
  right <- utf8ToInt(right)
  previous <- integer(length(right) + 1L)
  if (!length(left) || !length(right)) return(0L)
  for (row in seq_along(left)) {
    current <- integer(length(right) + 1L)
    for (column in seq_along(right)) {
      current[[column + 1L]] <- if (left[[row]] == right[[column]]) {
        previous[[column]] + 1L
      } else {
        max(previous[[column + 1L]], current[[column]])
      }
    }
    previous <- current
  }
  previous[[length(previous)]]
}

reference_true_damerau <- function(left, right) {
  left <- utf8ToInt(left)
  right <- utf8ToInt(right)
  m <- length(left)
  n <- length(right)
  maximum <- m + n
  distance <- matrix(0L, m + 2L, n + 2L)
  distance[1L, 1L] <- maximum
  distance[2L:(m + 2L), 1L] <- maximum
  distance[1L, 2L:(n + 2L)] <- maximum
  distance[2L:(m + 2L), 2L] <- 0:m
  distance[2L, 2L:(n + 2L)] <- 0:n
  last <- new.env(hash = TRUE, parent = emptyenv())
  if (m && n) for (i in seq_len(m)) {
    last_match <- 0L
    for (j in seq_len(n)) {
      key <- as.character(right[[j]])
      i1 <- if (exists(key, last, inherits = FALSE)) get(key, last) else 0L
      j1 <- last_match
      cost <- 1L
      if (left[[i]] == right[[j]]) {
        cost <- 0L
        last_match <- j
      }
      distance[i + 2L, j + 2L] <- min(
        distance[i + 1L, j + 1L] + cost,
        distance[i + 2L, j + 1L] + 1L,
        distance[i + 1L, j + 2L] + 1L,
        distance[i1 + 1L, j1 + 1L] +
          (i - i1 - 1L) + 1L + (j - j1 - 1L)
      )
    }
    assign(as.character(left[[i]]), i, last)
  }
  distance[m + 2L, n + 2L]
}

reference_jaro <- function(left, right) {
  left <- utf8ToInt(left)
  right <- utf8ToInt(right)
  if (!length(left) && !length(right)) return(1)
  if (!length(left) || !length(right)) return(0)
  radius <- max(length(left), length(right)) %/% 2L - 1L
  radius <- max(radius, 0L)
  left_match <- rep(FALSE, length(left))
  right_match <- rep(FALSE, length(right))
  matches <- 0L
  for (i in seq_along(left)) {
    lower <- max(1L, i - radius)
    upper <- min(length(right), i + radius)
    if (lower <= upper) for (j in lower:upper) {
      if (!right_match[[j]] && left[[i]] == right[[j]]) {
        left_match[[i]] <- TRUE
        right_match[[j]] <- TRUE
        matches <- matches + 1L
        break
      }
    }
  }
  if (!matches) return(0)
  left_values <- left[left_match]
  right_values <- right[right_match]
  # stride-align and RapidFuzz use the conventional floor when the number of
  # out-of-order matched characters is odd.
  transpositions <- sum(left_values != right_values) %/% 2L
  (matches / length(left) + matches / length(right) +
     (matches - transpositions) / matches) / 3
}

reference_jaro_winkler <- function(
  left,
  right,
  prefix_weight = 0.1,
  prefix_threshold = 0.7,
  prefix_cap = 4L
) {
  score <- reference_jaro(left, right)
  if (score < prefix_threshold) return(score)
  a <- utf8ToInt(left)
  b <- utf8ToInt(right)
  limit <- min(length(a), length(b), prefix_cap)
  prefix <- 0L
  if (limit) for (index in seq_len(limit)) {
    if (a[[index]] != b[[index]]) break
    prefix <- prefix + 1L
  }
  score + prefix * prefix_weight * (1 - score)
}

reference_alignment_score <- function(
  query,
  target,
  local = TRUE,
  match_score = 2,
  mismatch_score = -1,
  gap_score = -1
) {
  query <- utf8ToInt(query)
  target <- utf8ToInt(target)
  rows <- length(query) + 1L
  columns <- length(target) + 1L
  scores <- matrix(0, rows, columns)
  if (!local) {
    scores[, 1L] <- (0:(rows - 1L)) * gap_score
    scores[1L, ] <- (0:(columns - 1L)) * gap_score
  }
  best <- 0
  if (rows > 1L && columns > 1L) for (i in 2:rows) for (j in 2:columns) {
    value <- max(
      scores[i - 1L, j - 1L] +
        if (query[[i - 1L]] == target[[j - 1L]]) match_score else mismatch_score,
      scores[i - 1L, j] + gap_score,
      scores[i, j - 1L] + gap_score
    )
    if (local) value <- max(0, value)
    scores[i, j] <- value
    best <- max(best, value)
  }
  if (local) best else scores[rows, columns]
}

reference_dtw <- function(query, target, metric = c("l1", "l2_squared"), window = NULL) {
  metric <- match.arg(metric)
  n <- length(query)
  m <- length(target)
  if (!n && !m) return(0)
  if (!n || !m) return(Inf)
  radius <- if (is.null(window)) max(n, m) else if (window <= 1) {
    ceiling(max(n, m) * window)
  } else {
    as.integer(window)
  }
  if (abs(n - m) > radius) return(Inf)
  costs <- matrix(Inf, n + 1L, m + 1L)
  costs[1L, 1L] <- 0
  for (i in seq_len(n)) {
    lower <- max(1L, i - radius)
    upper <- min(m, i + radius)
    if (lower <= upper) for (j in lower:upper) {
      delta <- query[[i]] - target[[j]]
      point <- if (metric == "l1") abs(delta) else delta * delta
      costs[i + 1L, j + 1L] <- point + min(
        costs[i, j + 1L], costs[i + 1L, j], costs[i, j]
      )
    }
  }
  costs[n + 1L, m + 1L]
}
