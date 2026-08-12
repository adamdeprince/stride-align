files <- Sys.glob(paste0("*", SHLIB_EXT))
destination <- file.path(
    Sys.getenv("R_PACKAGE_DIR"),
    paste0("libs", Sys.getenv("R_ARCH"))
)

if (!dir.exists(destination)) {
    dir.create(destination, recursive = TRUE)
}

if (!length(files) || !all(file.copy(files, destination, overwrite = TRUE))) {
    stop("failed to install stride-align native libraries")
}
