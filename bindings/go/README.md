# stride-align for Go

## Install

You need Go 1.22 or newer, cgo, and a C++23 compiler. Until the next lockstep
stride-align release is tagged, install the development package with:

```sh
go get github.com/adamdeprince/stride-align/bindings/go@main
```

Import it as:

```go
import stridealign "github.com/adamdeprince/stride-align/bindings/go"
```

This is the normal installation: Go downloads the versioned source and cgo
builds it on the application machine. If the local toolchain cannot compile
C++23, use a [precompiled native backend](https://distribution.goblinreactor.com/stride-align/go/)
while keeping the same Go package and API:

```sh
go run github.com/adamdeprince/stride-align/bindings/go/cmd/stridealign-prebuilt@main

# Run the two commands printed by the installer. They set PKG_CONFIG_PATH and
# build with the stridealign_prebuilt tag.
```

The installer chooses the conservative profile for the current platform,
downloads it over HTTPS, verifies its SHA-256 digest against the repository
manifest, and installs it under `~/.stride-align/go`. On amd64, use
`--profile avx2` or `--profile avx512bwvl` when the deployment CPU supports
that level. The current binary repository covers macOS arm64 and Linux x86-64;
Arm Linux, POWER, and LoongArch are listed as pending. The fallback still needs
cgo, a C compiler, and `pkg-config`, but it does not invoke a C++ compiler.
Linux artifacts use a glibc 2.28 baseline and carry their static GCC C++ runtime
and corresponding license texts, so they do not depend on the user's
`libstdc++` version.

Go strings are passed as length-delimited UTF-8, so embedded NUL is supported.
One-to-many and matrix APIs flatten `[]string` once and cross the cgo boundary
once per operation. Returned data is copied into Go-owned memory before native
allocations are released.

## First comparison

```go
package main

import (
    "fmt"
    "log"

    stridealign "github.com/adamdeprince/stride-align/bindings/go"
)

func main() {
    score, err := stridealign.LevenshteinNormalizedScore("naïve", "naive")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("%.3f\n", score)
}
```

## Batch, cdist, and the two top-k shapes

```go
targets := []string{"kitten", "sitting", "bitten", "mittens"}

scores, err := stridealign.LevenshteinScores("kitten", targets)
nearest, err := stridealign.LevenshteinTopK("kitten", targets, 2)

queries := []string{"cat", "dog"}
dictionary := []string{"cat", "cot", "dog"}

// A 2 x 3 dense matrix.
dense, err := stridealign.CDist(
    queries, dictionary, stridealign.ScorerJaro, nil,
)

// The best two cells across the complete 2 x 3 matrix.
global, err := stridealign.CDistTopK(
    queries, dictionary, stridealign.ScorerJaro, 2, false, nil,
)

// Up to two targets for each query: at most 2 x 2 results.
perQuery, err := stridealign.CDistTopKPerQuery(
    queries, dictionary,
    stridealign.ScorerLevenshteinNormalized, 2, nil,
)
```

All batch APIs preserve zero-based source indices. Distance scorers rank lower
values first; similarities rank higher values first. Threshold and cdist top-k
operations require a normalized or similarity scorer because their pruning
bounds have that meaning.

## Alignment, matrices, phonetics, and DTW

```go
path, err := stridealign.SmithWatermanPath("ACCGT", "CCG", nil)
fmt.Println(path.Score, path.CIGAR, path.AlignedQuery, path.AlignedTarget)

blosum62, err := stridealign.BLOSUM62()
proteinScore, err := blosum62.SmithWatermanScore("HE", "HH", nil)

codes, err := stridealign.SoundexAll([]string{"Robert", "Rupert", "Rubin"})

distance, err := stridealign.DTW(
    []float64{1, 2, 3}, []float64{2, 2, 4}, nil,
)
```

The package also includes custom and NCBI-text substitution matrices, the full
BLOSUM/PAM catalog, nucleotide and ASCII matrices, keyboard-confusion matrix
construction and NumPy loading, LCS and n-gram algorithms, fuzzy token ratios,
Monge-Elkan, every R phonetic encoder, alignment tracebacks, and affine gaps.

The matrix and Beider–Morse resources are embedded in the source package. If
those shared catalogs change, regenerate the checked-in header from the
repository root:

```sh
python3 bindings/duckdb/generate_embedded_data.py \
  --source-root . \
  --namespace stride_align_go_embedded \
  --output bindings/go/embedded_data.hpp
```

## SIMD build profiles

The default amd64 build is portable x86-64, and the default arm64 build uses
NEON. Select a machine-specific package at application build time with one tag:

```sh
go build -tags stridealign_native ./cmd/myapp
go build -tags stridealign_avx2 ./cmd/myapp
go build -tags stridealign_avx512bwvl ./cmd/myapp
go build -tags stridealign_power8_vsx ./cmd/myapp
go build -tags stridealign_la464_lsx ./cmd/myapp
go build -tags stridealign_la464_lasx ./cmd/myapp
go build -tags stridealign_la664_lasx ./cmd/myapp
```

The AVX2 package compiles with `-march=x86-64-v3`; AVX-512 uses
`-march=x86-64-v4`; POWER uses `-mcpu=power8`; and each LoongArch profile has
the matching LA464 or LA664 `-march`/`-mtune` flags. `stridealign.Backend()`
reports the selected profile. Do not run a tagged binary on a CPU below that
profile.

The precompiled fallback uses the same profiles. Pass the requested profile to
the installer, then use the exact build tags it prints; the archive and Go
build must describe the same CPU target.

## Test

```sh
go test ./bindings/go
go test -race ./bindings/go
go test -tags stridealign_native ./bindings/go
```

To verify a downloaded backend without allowing Go to invoke a C++ compiler:

```sh
export PKG_CONFIG_PATH="$HOME/.stride-align/go/stride-align-go-0.6.0-1-PLATFORM-PROFILE/lib/pkgconfig"
CXX=false go test -tags stridealign_prebuilt ./bindings/go
```

Replace the final path with the exact value printed by the installer. Add its
profile tag as well when testing AVX2 or AVX-512.
