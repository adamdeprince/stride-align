# New-world Python validation for LoongArch (CPython 3.12 through 3.14)

**Status:** completed 2026-08-13.
**Goal:** build and smoke-test the new-world LoongArch wheel for every
supported Python version (3.12, 3.13, 3.14) even though the installed Kylin
system remains old-world. The validation interpreters use the GCC 16 sysroot;
the redistributable wheels use the target-prefixed compilers directly and do
not retain toolchain RPATHs.

## Why this is needed

LoongArch has two binary worlds — old-world (loader `/lib64/ld.so.1`,
glibc 2.28, what stock Kylin ships) and new-world (loader
`/lib64/ld-linux-loongarch-lp64d.so.1`, glibc ≥ 2.36). A new-world
`.so` cannot be loaded into an old-world Python process because the
dynamic linker matches `libc.so.6` by SONAME from whatever is already
mapped, and only one libc can live in one address space. So:

- Old-world Python + 15.2.0 toolchain → builds + runs the **old-world**
  wheel. Already working today.
- New-world Python + 16.1.0 toolchain (the validation interpreter may use
  `wrappers/`) → runs the **new-world** wheel.

The release check now builds small new-world CPython runtimes for all three
supported ABIs and imports each finished wheel under glibc 2.43. An attempted
import into the host's old-world CPython is not considered validation.

## Plan

### 1. Build CPython 3.12–3.14 with the 16.1.0 wrappers

Native build (the box itself produces new-world binaries), one
prefix per minor version so they coexist with the old-world pyenv
versions. The wrappers in `/opt/loongson-gcc-16.1.0/wrappers/` are
critical: they auto-append the three `-Wl,-rpath,…` flags that point
the resulting `python3.x` binary at the new-world sysroot libs.

Common environment for every version:

```bash
export GCC16=/opt/loongson-gcc-16.1.0
export PATH=/opt/loongson-cmake-4.3.2/bin:$GCC16/wrappers:$GCC16/bin:$PATH
export CC=$GCC16/wrappers/gcc
export CXX=$GCC16/wrappers/g++
export LDFLAGS="-Wl,-rpath,$GCC16/sysroot/lib64 -Wl,-rpath,$GCC16/sysroot/usr/lib64 -Wl,-rpath,$GCC16/loongarch64-loongson-linux-gnu/lib"
```

(The wrappers already add the RPATH at link time, but exporting
`LDFLAGS` belt-and-suspenders is harmless and surfaces the intent in
the configure log.)

Per-version build loop. The Python source layout has stabilised
enough that the same incantation works for 3.12 onward; if a version
fails, drop a per-version note here.

```bash
PREFIX_ROOT=/opt/loongson-python-newworld
mkdir -p ~/src/python-newworld
cd ~/src/python-newworld

for VER in 3.12.7 3.13.13 3.14.0; do
    MAJ_MIN=${VER%.*}
    test -d Python-$VER || (
        curl -L -O https://www.python.org/ftp/python/$VER/Python-$VER.tar.xz
        tar xf Python-$VER.tar.xz
    )
    PREFIX=$PREFIX_ROOT/$MAJ_MIN
    rm -rf build-$VER
    mkdir build-$VER && cd build-$VER
    ../Python-$VER/configure \
        --prefix=$PREFIX \
        --enable-shared \
        --with-ensurepip=install \
        --enable-optimizations=no \
        --with-system-ffi
    make -j$(nproc)
    sudo make install
    cd ..
done
```

Pin the exact 3.x.y you want — the list above is "latest stable in
2026-05" but bump as needed.

Verify each interpreter is new-world:

```bash
for V in 3.12 3.13 3.14; do
    p=/opt/loongson-python-newworld/$V/bin/python3
    echo "== $V =="
    file $p | grep -o "interpreter [^,]*"
    ldd $p | grep libc.so
done
```

Expected per version: `interpreter /lib64/ld-linux-loongarch-lp64d.so.1`
and `libc.so.6 => /opt/loongson-gcc-16.1.0/sysroot/lib64/libc.so.6`.
If you get the system loader / system libc, the wrappers weren't on
PATH at configure time — wipe `build-$VER` and start over.

### 2. Per-version new-world venv + numpy

```bash
for V in 3.12 3.13 3.14; do
    /opt/loongson-python-newworld/$V/bin/python3 -m venv \
        ~/venvs/stride-align-newworld-$V
    source ~/venvs/stride-align-newworld-$V/bin/activate
    pip install --upgrade pip cython meson meson-python ninja build wheel
    # Build numpy from source with the same wrappers so its RPATH
    # matches Python's. --no-binary avoids the manylinux wheel
    # (which would be old-world or x86 anyway).
    CC=$GCC16/wrappers/gcc CXX=$GCC16/wrappers/g++ \
        pip install --no-binary numpy numpy
    pip install pytest
    deactivate
done
```

### 3. Build the new-world stride-align wheels

```bash
cd ~/dev/stride-align
export NEW_CC=$GCC16/bin/loongarch64-loongson-linux-gnu-gcc
export NEW_CXX=$GCC16/bin/loongarch64-loongson-linux-gnu-g++
for V in 3.12 3.13 3.14; do
    source ~/venvs/stride-align-newworld-$V/bin/activate
    CC=$NEW_CC CXX=$NEW_CXX \
        pip wheel . --no-build-isolation --no-deps \
        -C wheel.build_tag=1.newworld \
        -w wheelhouse/newworld
    deactivate
done
```

This produces:

```
wheelhouse/newworld/stride_align-0.6.0-1.newworld-cp312-cp312-linux_loongarch64.whl
wheelhouse/newworld/stride_align-0.6.0-1.newworld-cp313-cp313-linux_loongarch64.whl
wheelhouse/newworld/stride_align-0.6.0-1.newworld-cp314-cp314-linux_loongarch64.whl
```

### 4. Smoke test each new-world wheel

```bash
for V in 3.12 3.13 3.14; do
    PY=cp$(echo $V | tr -d .)
    source ~/venvs/stride-align-newworld-$V/bin/activate
    pip install --force-reinstall \
        wheelhouse/newworld/stride_align-0.6.0-1.newworld-${PY}-${PY}-linux_loongarch64.whl
    python -m pytest tests/test_ndarray.py tests/test_unicode_wide.py -q
    deactivate
done
```

If a version blows up, capture the traceback alongside the failing
test_*.py path and Python version in a follow-up.

### 5. Mirror the old-world matrix

The same loop with `GCC15=/opt/loongson-gcc-15.2.0`,
`CC=$GCC15/bin/gcc`, no wrappers, no LDFLAGS, and venvs under
`~/venvs/stride-align-oldworld-$V` produces the old-world wheels.
The dev box's existing pyenv'd Python 3.13 already covers 3.13
old-world; build the rest only when shipping a release that
promises older-Python support.

## Pitfalls a future AI should know

- **Use wrappers only for the validation interpreter.** Build distributable
  wheels with `bin/loongarch64-loongson-linux-gnu-gcc` and
  `bin/loongarch64-loongson-linux-gnu-g++`. The wrappers deliberately add
  absolute build-host RPATHs, which must never appear in a wheel.
- **Don't put 16.1's libs on LD_LIBRARY_PATH while running an
  old-world binary.** It pulls the new-world libstdc++ in, which
  DT_NEEDEDs the new-world loader, which the old-world process
  can't substitute for its already-mapped one.
- **One libc per process.** Don't try to load new-world `.so` into
  old-world Python; ldd will show the right RPATH targets but the
  process's first-loaded libc wins by SONAME and you'll see
  `GLIBC_2.38 not found` from libc 2.28.
- **`STRIDE_ALIGN_STATIC_CXX_RUNTIME=ON`** is already the default
  for loongarch64; don't disable it. Static libstdc++ / libgcc is
  what makes the wheel ABI-stable across distros within the same
  world.
- **Loader symlink for new-world boxes:**

      sudo ln -sf /opt/loongson-gcc-16.1.0/sysroot/lib64/ld-linux-loongarch-lp64d.so.1 \
                  /lib64/ld-linux-loongarch-lp64d.so.1

  Add a note to the README install section that new-world wheels
  expect this symlink (most new-world distros ship it via their
  glibc package; only the GCC-16.1-only bootstrap setups need the
  manual symlink).
- **Python EOL.** The project floor is 3.12 (Python 3.9–3.11 were
  dropped). 3.12 itself reaches end-of-life 2028-10 — well after the
  next planned rev — so the matrix should stay 3.12/3.13/3.14 for
  the foreseeable future. Switching to a single abi3 wheel would
  collapse this to one build per world; tracked separately.

## Related

- [docs/loongson-build.md](loongson-build.md) — current dual-toolchain build
  and validation instructions for both worlds.
- [README.md](../README.md) — points users at the right wheel URL
  per their world; URLs above must match the GitHub Release asset
  names produced in §3 and §5.
