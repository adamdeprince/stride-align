# stride-align for R

`stridealign` brings stride-align's Unicode-aware string distances,
similarities, and sequence-alignment scores to ordinary R character vectors.
Calls are vectorized, a length-one argument is broadcast, and missing strings
produce missing scores.

## Build and install

Build the self-contained R source package from the repository root:

```sh
bindings/r/build-package.sh
R CMD INSTALL dist/r/stridealign_0.6.0.tar.gz
```

The package selects the best compatible backend when R loads it. Published
binary packages can contain separate generic, SSE4.1, AVX2, AVX-512, NEON,
SVE, SVE2, LSX, LASX, VSX, or RVV libraries. To pin a packaged backend for a
benchmark, set `STRIDE_ALIGN_R_BACKEND` before loading the package:

```sh
STRIDE_ALIGN_R_BACKEND=generic Rscript benchmark.R
STRIDE_ALIGN_R_BACKEND=avx2 Rscript benchmark.R
```

An unavailable or CPU-incompatible override is rejected during package load.

## Use

```r
library(stridealign)

stride_backend()
stride_levenshtein("kitten", "sitting")
stride_levenshtein_similarity("你好世界", "你好世间")

names <- c("Martha", "Marhta", "Arthur", NA_character_)
stride_jaro_winkler(names, "Martha")

stride_needleman_wunsch(
    c("GATTACA", "GACTATA"),
    "GATTACA",
    match_score = 2,
    mismatch_score = -1,
    gap_score = -1
)
```

The R package exposes stride-align's native two-string algorithms. It does not
reproduce the Python compatibility namespaces.

## Native functions

- Levenshtein, optimal string alignment, unrestricted Damerau-Levenshtein,
  indel, and Hamming distance and normalized similarity
- Jaro and configurable Jaro-Winkler similarity
- Smith-Waterman and Needleman-Wunsch scores with linear or affine gaps

All functions compare Unicode code points, accept character vectors, and
return numeric vectors.
