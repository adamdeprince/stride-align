# Building stride-align for LoongArch

The Loongson 3A6000 reference box runs Kylin OS with glibc 2.28 — an
"old-world" LoongArch system that uses the historical loader
`/lib64/ld.so.1`. Newer LoongArch distributions ship "new-world"
binaries that use a different loader, `/lib64/ld-linux-loongarch-lp64d.so.1`,
and require glibc ≥ 2.36 ABI symbols.

The two worlds are mutually incompatible at the binary level: a
new-world `.so` won't load on an old-world box (the loader filename
doesn't exist), and an old-world `.so` won't pick up the new-world
symbols an upstream-built numpy needs.

We ship **one stride-align wheel per world**:

| Wheel build tag | Toolchain | Target audience |
| --- | --- | --- |
| `1.oldworld` | GCC 15.2.0 (`/opt/loongson-gcc-15.2.0`) | stock Kylin, older LoongArch distros |
| `1.newworld` | GCC 16.1.0 (`/opt/loongson-gcc-16.1.0`, via `wrappers/`) | recent LoongArch distros with the new loader installed |

Both wheels are byte-for-byte identical at the source level — the
distinction lives entirely in which toolchain links them. The
`STRIDE_ALIGN_STATIC_CXX_RUNTIME` CMake option defaults `ON` for
loongarch64, so libstdc++ / libgcc are statically embedded; only the
glibc / loader ABI separates the two builds.

## Dev environment on the reference box

Both toolchains coexist at `/opt/loongson-gcc-15.2.0` and
`/opt/loongson-gcc-16.1.0`. cmake ≥ 3.26 lives at
`/opt/loongson-cmake-4.3.2` (the system `/usr/bin/cmake` is 3.16,
too old for scikit-build-core).

The one-time **sudo** step the new-world loader needs:

```bash
sudo ln -sf /opt/loongson-gcc-16.1.0/sysroot/lib64/ld-linux-loongarch-lp64d.so.1 \
            /lib64/ld-linux-loongarch-lp64d.so.1
```

This creates a new filename and never touches `/lib64/ld.so.1`, so
existing old-world binaries are unaffected.

## Building the old-world wheel

```bash
ssh loongson '
    export GCC15=/opt/loongson-gcc-15.2.0
    export PATH=/opt/loongson-cmake-4.3.2/bin:$GCC15/bin:$PATH
    export CC=$GCC15/bin/gcc
    export CXX=$GCC15/bin/g++
    cd ~/dev/stride-align
    .venv/bin/pip install -e . --no-build-isolation --no-deps
'
```

The 15.2.0 toolchain produces old-world binaries: PT_INTERP points at
`/lib64/ld.so.1`, libc symbol references stay within glibc 2.28's
range. No loader symlink required at install time on the user box.

If you want to package a wheel rather than an editable install,
replace the `pip install -e . …` line with
`pip wheel . --no-build-isolation --no-deps -w wheelhouse/oldworld`
and tag the resulting file with build tag `1.oldworld` (see
[Wheel build tags](#wheel-build-tags) below).

## Building the new-world wheel

```bash
ssh loongson '
    export GCC16=/opt/loongson-gcc-16.1.0
    export PATH=/opt/loongson-cmake-4.3.2/bin:$GCC16/wrappers:$GCC16/bin:$PATH
    export CC=$GCC16/wrappers/gcc
    export CXX=$GCC16/wrappers/g++
    cd ~/dev/stride-align
    .venv/bin/pip install -e . --no-build-isolation --no-deps
'
```

Note the **wrappers** directory on PATH and as CC/CXX. The wrappers
exec the real `gcc` / `g++` with three extra `-Wl,-rpath,…` flags so
the produced `.so` carries an RPATH pointing at the new-world sysroot
libs (libstdc++, libc, libm, the new loader). Without the wrappers
the .so still builds, but it would need `LD_LIBRARY_PATH` set at run
time — which we don't want users to need.

The user box needs the loader symlink installed (see above) but no
LD_LIBRARY_PATH. The RPATH locates everything else.

## Wheel build tags

`pip` and PyPI accept an optional **build tag** between the
distribution name and the Python tag, format `\d+(\.\w+)*` per
PEP 491. We use it to mark the LoongArch world:

```
stride_align-0.3.0-1.oldworld-cp313-cp313-linux_loongarch64.whl
stride_align-0.3.0-1.newworld-cp313-cp313-linux_loongarch64.whl
```

`pip install <url>` accepts either filename and installs the right
one. The `1.` prefix is a numeric build sequence (PEP 440 build
number); bump it on respins.

To set the build tag at wheel time, pass `-C wheel.build_tag=1.oldworld`
(or `1.newworld`) to `pip wheel` / `pip install`, e.g.:

```bash
.venv/bin/pip wheel . --no-build-isolation --no-deps \
    -C wheel.build_tag=1.oldworld -w wheelhouse/oldworld
```

## How a user picks the right wheel

```bash
# Old-world: stock Kylin / older LoongArch distros. The new loader
# filename is absent.
test ! -e /lib64/ld-linux-loongarch-lp64d.so.1 && echo "old-world"

# New-world: the loader exists (either via distro packaging or the
# sudo symlink step from §Dev environment).
test -e /lib64/ld-linux-loongarch-lp64d.so.1 && echo "new-world"
```

The README will host direct download URLs for the latest stable
release of each world.

## Rebuilding numpy on the dev box (one-time)

If the venv numpy was last built against the wrong world's libstdc++
(e.g. you switched worlds and pytest now fails with
`GLIBCXX_3.4.29 not found` or `ld-linux-loongarch-lp64d.so.1: cannot
open shared object file`), rebuild it with the matching wrappers:

```bash
ssh loongson '
    export GCC16=/opt/loongson-gcc-16.1.0
    export PATH=/opt/loongson-cmake-4.3.2/bin:$GCC16/wrappers:$GCC16/bin:$PATH
    cd ~/dev/stride-align
    .venv/bin/pip install cython
    CC=$GCC16/wrappers/gcc CXX=$GCC16/wrappers/g++ \
        .venv/bin/pip install --no-binary numpy --force-reinstall numpy
'
```

For the old-world venv swap `GCC16` → `GCC15` and drop the wrappers
suffix (`$GCC15/bin/gcc`, `$GCC15/bin/g++`).

## Gotchas

- **Don't mix LD_LIBRARY_PATH across worlds.** Putting the 16.1 lib
  dir on `LD_LIBRARY_PATH` while running an old-world binary makes
  the dynamic linker load the new-world libstdc++, which DT_NEEDEDs
  the new-world loader, which the running process can't satisfy.
  Everything breaks.
- **Use the wrappers, not bare `bin/`, for new-world builds.** The
  bare binaries skip the RPATH and produce .so files that need
  `LD_LIBRARY_PATH` to find their libs at run time. Users won't
  thank you.
- **Static C++ runtime is on by default for loongarch64.** Don't
  disable it unless you're certain the user box has a matching
  libstdc++ in `LD_LIBRARY_PATH` — the wheel becomes ABI-fragile
  otherwise.
- **PyPI doesn't index loongarch wheels.** GitHub Releases is the
  distribution point; the README pins specific URLs per Python
  version per world.
