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
mkdir -p "$stage_package/inst"
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
