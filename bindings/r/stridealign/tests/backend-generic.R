Sys.setenv(STRIDE_ALIGN_R_BACKEND = "generic")
library(stridealign)

stopifnot(
    identical(stride_backend(), "generic"),
    stride_levenshtein("kitten", "sitting") == 3,
    stride_levenshtein(
        intToUtf8(c(0x4f60, 0x597d, 0x4e16, 0x754c)),
        intToUtf8(c(0x4f60, 0x597d, 0x4e16, 0x95f4))
    ) == 1
)
