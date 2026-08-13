#!/bin/sh

set -eu

script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH= cd -- "$script_directory/../.." && pwd)
output_directory=${1:-"$repository_root/dist/r"}

if ! command -v R >/dev/null 2>&1; then
  printf '%s\n' "R is required to build the stridealign package" >&2
  exit 1
fi

stage_parent=$(mktemp -d "${TMPDIR:-/tmp}/stridealign-r-package.XXXXXX")
trap 'rm -rf "$stage_parent"' EXIT HUP INT TERM
stage_package="$stage_parent/stridealign"

cp -R "$script_directory/stridealign" "$stage_package"
mkdir -p "$stage_package/src/vendor"
cp -R "$repository_root/include/stride_align" "$stage_package/src/vendor/stride_align"
cp "$repository_root/src/cpp/beider_morse_impl.cpp" "$stage_package/src/beider_morse_impl.cpp"
mkdir -p "$stage_package/inst"
mkdir -p "$stage_package/inst/bmpm_data"
cp "$repository_root/src/stride_align/bmpm_data/"*.txt "$stage_package/inst/bmpm_data/"
mkdir -p "$stage_package/inst/matrix_data"
cp "$repository_root/src/stride_align/matrix_data/"* "$stage_package/inst/matrix_data/"
mkdir -p "$stage_package/inst/keyboard_data"
cp "$repository_root/src/stride_align/matrices/keyboard_data/"*.npy \
  "$stage_package/inst/keyboard_data/"
mkdir -p "$stage_package/inst/testdata"
cp "$repository_root/docs/bounded-lazy-f-counterexamples.txt" \
  "$stage_package/inst/testdata/bounded-lazy-f-counterexamples.txt"
Rscript "$script_directory/generate-matrix-data.R" \
  "$repository_root/src/stride_align/matrices/__init__.py" \
  "$stage_package/inst/matrix_data"
cp "$repository_root/LICENSE" "$stage_package/inst/LICENSE-APACHE-2.0"
cp "$repository_root/NOTICE" "$stage_package/NOTICE"

mkdir -p "$output_directory"
(
  cd "$stage_parent"
  R CMD build --no-manual --no-build-vignettes stridealign
)

artifact=$(find "$stage_parent" -maxdepth 1 -type f -name 'stridealign_*.tar.gz' -print)
if test -z "$artifact"; then
  printf '%s\n' "R did not produce a stridealign source package" >&2
  exit 1
fi

cp "$artifact" "$output_directory/"
printf '%s\n' "$output_directory/$(basename "$artifact")"
