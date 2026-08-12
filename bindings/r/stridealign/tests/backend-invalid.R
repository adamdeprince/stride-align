Sys.setenv(STRIDE_ALIGN_R_BACKEND = "definitely-not-a-backend")
message <- tryCatch(
    {
        library(stridealign)
        NULL
    },
    error = conditionMessage
)

stopifnot(
    !is.null(message),
    grepl("unavailable or incompatible", message, fixed = TRUE)
)
