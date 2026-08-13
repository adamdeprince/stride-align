# stride-align for R

`stridealign` brings stride-align's complete native API to ordinary R objects:
Unicode string distances and similarities, sequence alignment, batch search,
substitution matrices, phonetics, and Dynamic Time Warping. Pair calls are
vectorized, a length-one argument is broadcast, and missing strings produce
missing scores.

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
levenshtein_score("kitten", "sitting")
levenshtein_normalized_score("你好世界", "你好世间")

targets <- c("Martha", "Marhta", "Arthur", NA_character_)
jaro_winkler_similarities("Martha", targets)
jaro_winkler_top_k("Martha", targets, k = 2)

cdist(
    c("Martha", "Arthur"),
    c("Martha", "Marhta", "Arthur"),
    scorer = Scorer$JARO_WINKLER
)

smith_waterman_scores("HE", c("HE", "HH"), matrix = blosum62)
```

Selection results use data frames or named lists and one-based R indices.
`cdist()` returns an ordinary numeric matrix. The canonical names mirror the
Python package, while return containers follow R conventions.

## Native API

- Pair, one-to-many, top-k, best-match, persistent-query, and all-pairs APIs
  for all 16 `Scorer` values
- Levenshtein, OSA, unrestricted Damerau-Levenshtein, indel, Hamming, Jaro,
  and Jaro-Winkler
- Smith-Waterman and Needleman-Wunsch scores, normalized scores, traceback,
  CIGAR output, linear and affine gaps, and substitution matrices
- LCS, n-gram similarities, Ratcliff-Obershelp, token and partial ratios,
  WRatio, and Monge-Elkan
- Soundex, Metaphone, NYSIIS, Match Rating, Caverphone, Cologne,
  Daitch-Mokotoff, Double Metaphone, and Beider-Morse phonetics
- Dynamic Time Warping scalar and one-to-many distances

The package intentionally does not reproduce the RapidFuzz, Parasail,
Jellyfish, or TheFuzz compatibility layers.

## Tests

The R tests mirror every native Python test module. Runtime-specific checks
use R equivalents—for example, ordinary numeric matrices instead of NumPy
arrays and one-based result indices. The parity ledger intentionally excludes
only the four RapidFuzz, Parasail, Jellyfish, and TheFuzz shim suites.

From the repository root, audit that mapping and run the package tests with:

```sh
python3 bindings/r/check-python-test-parity.py
bindings/r/build-package.sh
R CMD check --no-manual --no-build-vignettes dist/r/stridealign_0.6.0.tar.gz
```
