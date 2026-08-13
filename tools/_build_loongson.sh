#!/bin/bash
# Build LoongArch64 old-world (GCC 15.2.0) + new-world (GCC 16.1.0)
# wheels for cp312/313/314 from the current sdist. See docs/loongson-build.md.
#
# loongson's DNS to PyPI is flaky, so the build-backend deps are installed
# OFFLINE from /tmp/sa-bw (py3-none-any wheels shipped from the Mac). The
# 1.oldworld / 1.newworld PEP 427 build tag is applied by RENAMING the
# finished wheels (the tag is filename-only; the wheel.build-tag config
# setting proved unreliable here).
# Transfer inputs and outputs with one rsync process at a time and
# --bwlimit=100; the reference box has a 100 KiB/s aggregate network ceiling.
set -uo pipefail
VERSION=${STRIDE_ALIGN_VERSION:-0.6.0}
SDIST=${STRIDE_ALIGN_SDIST:-/tmp/stride_align-$VERSION.tar.gz}
BW=/tmp/sa-bw
CMAKE_BIN=/opt/loongson-cmake-4.3.2/bin
rm -rf ~/wheelhouse/oldworld ~/wheelhouse/newworld
mkdir -p ~/wheelhouse/oldworld ~/wheelhouse/newworld

# One fresh venv per Python version, backend deps installed offline.
for pair in 3.12:3.12.13 3.13:3.13.13 3.14:3.14.4; do
  short=${pair%%:*}; full=${pair##*:}; venv=/tmp/sa-bvenv-$short
  rm -rf "$venv"
  "$HOME/.pyenv/versions/$full/bin/python" -m venv "$venv"
  "$venv/bin/pip" install -q --no-index --find-links "$BW" scikit-build-core nanobind wheel 2>&1 | tail -1
  "$venv/bin/python" -c "import scikit_build_core, nanobind" 2>/dev/null \
    && echo "venv $short deps OK" || echo "venv $short DEPS FAILED"
done

build() {  # $1=world  $2=CC  $3=CXX  $4=extra_path
  for pair in 3.12:3.12.13 3.13:3.13.13 3.14:3.14.4; do
    short=${pair%%:*}; venv=/tmp/sa-bvenv-$short
    build_dir=/tmp/sa-build-$short-$1
    rm -rf "$build_dir"
    echo "--- cp$short $1 ($(date +%H:%M:%S)) ---"
    CC="$2" CXX="$3" PATH="$CMAKE_BIN:$4:$PATH" \
      "$venv/bin/pip" wheel "$SDIST" --no-build-isolation --no-deps \
        -C "build-dir=$build_dir" -w "$HOME/wheelhouse/$1" \
        2>&1 | grep -iE "created|error|fail" | tail -2
  done
}

GCC15=/opt/loongson-gcc-15.2.0; GCC16=/opt/loongson-gcc-16.1.0
build oldworld "$GCC15/bin/gcc"      "$GCC15/bin/g++"      "$GCC15/bin"
build newworld "$GCC16/bin/loongarch64-loongson-linux-gnu-gcc" \
               "$GCC16/bin/loongarch64-loongson-linux-gnu-g++" \
               "$GCC16/bin"

# Insert the PEP 427 build tag into each filename (filename-only; safe).
for world in oldworld newworld; do
  for w in "$HOME"/wheelhouse/$world/stride_align-$VERSION-cp*.whl; do
    [ -e "$w" ] || continue
    b=$(basename "$w")
    mv "$w" "$HOME/wheelhouse/$world/${b/stride_align-$VERSION-/stride_align-$VERSION-1.$world-}"
  done
done

echo "=== loongson wheels (tagged) ==="
ls -1 "$HOME"/wheelhouse/oldworld/*.whl "$HOME"/wheelhouse/newworld/*.whl 2>/dev/null | sed 's#.*/##'
echo "DONE $(date +%H:%M:%S)"
