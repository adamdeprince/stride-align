source("helpers.R")

# Mirrors tests/test_lcs.py. R's CE_BYTES character strings replace Python's
# bytes object for the ASCII storage-path checks.
length_cases <- list(
  c("ABCBDAB", "BDCAB", 4), c("AGGTAB", "GXTXAYB", 4),
  c("hello", "hello", 5), c("", "abc", 0), c("abc", "", 0),
  c("", "", 0), c("abc", "xyz", 0), c("aabb", "bbaa", 2),
  c("abc", "axbxc", 3)
)
for (case in length_cases) {
  stopifnot(lcs_length(case[[1L]], case[[2L]]) == as.integer(case[[3L]]))
}
stopifnot(lcs_length("ABCBDAB", "BDCAB") == lcs_length("BDCAB", "ABCBDAB"))
for (pair in list(
  c("kitten", "sitting"), c("hello", "world"),
  c("ABCBDAB", "BDCAB"), c("", "anything")
)) {
  left <- pair[[1L]]
  right <- pair[[2L]]
  stopifnot(
    indel_score(left, right) ==
      nchar(left, type = "chars") + nchar(right, type = "chars") -
        2 * lcs_length(left, right)
  )
}

substring_cases <- list(
  list("ABCBDAB", "BDCAB", 2L, "AB"),
  list("Müller", "Mueller", 4L, "ller"),
  list("hello", "hello", 5L, "hello"), list("abc", "xyz", 0L, ""),
  list("", "abc", 0L, ""), list("abc", "", 0L, ""),
  list("hello world", "world hello", 5L, "hello"),
  list("prefixA", "prefixB", 6L, "prefix"),
  list("Asuffix", "Bsuffix", 6L, "suffix"),
  list("xxABCyy", "zzABCzz", 3L, "ABC")
)
for (case in substring_cases) {
  stopifnot(
    lcs_substring_length(case[[1L]], case[[2L]]) == case[[3L]],
    lcs_substring(case[[1L]], case[[2L]]) == case[[4L]]
  )
}
bytes_left <- "hello world"
bytes_right <- "world hello"
Encoding(bytes_left) <- "bytes"
Encoding(bytes_right) <- "bytes"
stopifnot(lcs_substring(bytes_left, bytes_right) == "hello")
stopifnot(
  lcs_substring("αβγδε", "βγδ") == "βγδ",
  lcs_substring_length("αβγδε", "βγδ") == 3L,
  lcs_substring("abc", "b") == "b",
  lcs_substring_length("abc", "b") == 1L
)
for (pair in list(
  c("ABCBDAB", "BDCAB"), c("hello world", "world hello"),
  c("kitten", "sitting"), c("Müller", "Mueller")
)) {
  subsequence <- lcs_length(pair[[1L]], pair[[2L]])
  substring <- lcs_substring_length(pair[[1L]], pair[[2L]])
  stopifnot(
    is.numeric(subsequence), subsequence >= 0,
    is.numeric(substring), substring >= 0,
    substring <= subsequence
  )
}

# Python cases: test_lcs_length, test_lcs_length_symmetric,
# test_lcs_length_matches_indel_relation, test_lcs_substring,
# test_lcs_substring_bytes_input_returns_bytes,
# test_lcs_substring_str_and_bytes_are_consistent_on_ascii,
# test_lcs_substring_unicode_codepoints, test_lcs_substring_disjoint_returns_empty,
# test_lcs_substring_single_char_match, test_lcs_lengths_are_non_negative_ints,
# test_lcs_substring_length_le_lcs_length.
