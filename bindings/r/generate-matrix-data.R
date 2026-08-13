args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
    stop("usage: generate-matrix-data.R PYTHON_MATRIX_MODULE OUTPUT_DIRECTORY")
}

source_file <- args[[1L]]
output_directory <- args[[2L]]
dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)
lines <- readLines(source_file, warn = FALSE, encoding = "UTF-8")
alphabet <- strsplit("ARNDCQEGHILKMFPSTWYVBZX*", "", fixed = TRUE)[[1L]]

extract_matrix <- function(symbol) {
    start <- grep(paste0("^", symbol, " = (np[.]array|_matrix)[(]"), lines)
    if (length(start) != 1L) stop("could not locate ", symbol)
    size <- length(alphabet)
    rows <- list()
    cursor <- start + 1L
    while (cursor <= length(lines) && length(rows) < size) {
        row <- sub("#.*$", "", lines[[cursor]])
        if (grepl("^[[:space:]]*[[].*[]][,]?[[:space:]]*$", row)) {
            matches <- regmatches(row, gregexpr("-?[0-9]+", row, perl = TRUE))[[1L]]
            values <- as.integer(matches)
            if (length(values) != size) {
                stop(symbol, " row ", length(rows) + 1L, " has ", length(values),
                     " values; expected ", size)
            }
            rows[[length(rows) + 1L]] <- values
        }
        cursor <- cursor + 1L
    }
    if (length(rows) != size) stop(symbol, " has only ", length(rows), " rows")
    do.call(rbind, rows)
}

write_ncbi <- function(values, filename) {
    path <- file.path(output_directory, filename)
    con <- file(path, open = "wt", encoding = "UTF-8")
    on.exit(close(con), add = TRUE)
    writeLines(paste("  ", paste(alphabet, collapse = "  ")), con)
    for (row in seq_along(alphabet)) {
        writeLines(sprintf(
            "%s %s",
            alphabet[[row]],
            paste(sprintf("%3d", values[row, ]), collapse = " ")
        ), con)
    }
}

symbols <- c(
    BLOSUM45 = "_BLOSUM45_VALUES",
    BLOSUM50 = "_BLOSUM50_VALUES",
    BLOSUM62 = "_BLOSUM62_VALUES",
    BLOSUM80 = "_BLOSUM80_VALUES",
    BLOSUM90 = "_BLOSUM90_VALUES",
    PAM30 = "_PAM30_VALUES",
    PAM70 = "_PAM70_VALUES",
    PAM250 = "_PAM250_VALUES"
)
for (filename in names(symbols)) {
    write_ncbi(extract_matrix(symbols[[filename]]), filename)
}
