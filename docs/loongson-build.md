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
| `1.newworld` | GCC 16.1.0 (`/opt/loongson-gcc-16.1.0`) | recent LoongArch distros with the new loader installed |

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

All transfers into or out of the reference box must be serialized and capped
at 100 KiB/s. Use resumable rsync, including for local-LAN transfers:

```bash
rsync -av --partial --append-verify --bwlimit=100 SOURCE loongson:DESTINATION
```

Do not run multiple capped transfers concurrently: the 100 KiB/s limit is the
aggregate ceiling, not a per-process allowance.

The build-only new-world validation interpreters use the GCC 16 sysroot and
loader. That is a test harness, not a migration recipe for an old-world user
system; users select a wheel from the loader embedded in their Python
executable.

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
    export PATH=/opt/loongson-cmake-4.3.2/bin:$GCC16/bin:$PATH
    export CC=$GCC16/bin/loongarch64-loongson-linux-gnu-gcc
    export CXX=$GCC16/bin/loongarch64-loongson-linux-gnu-g++
    cd ~/dev/stride-align
    .venv/bin/pip install -e . --no-build-isolation --no-deps
'
```

Use the toolchain binaries directly. They already target the new-world sysroot,
and stride-align statically links libstdc++ and libgcc into its extension
modules. The old `wrappers/` launchers add an absolute `/opt/loongson-gcc-*`
RPATH; that happens to work on the build box but is not suitable for a
redistributable wheel.

The user box needs the new-world loader (normally supplied by its distribution)
but no build-host RPATH or `LD_LIBRARY_PATH`.

## Wheel build tags

`pip` and PyPI accept an optional **build tag** between the
distribution name and the Python tag, format `\d+(\.\w+)*` per
PEP 491. We use it to mark the LoongArch world:

```
stride_align-0.6.0-1.oldworld-cp313-cp313-linux_loongarch64.whl
stride_align-0.6.0-1.newworld-cp313-cp313-linux_loongarch64.whl
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
open shared object file`), rebuild it with the matching toolchain:

```bash
ssh loongson '
    export GCC16=/opt/loongson-gcc-16.1.0
    export PATH=/opt/loongson-cmake-4.3.2/bin:$GCC16/bin:$PATH
    cd ~/dev/stride-align
    .venv/bin/pip install cython
    CC=$GCC16/bin/loongarch64-loongson-linux-gnu-gcc \
        CXX=$GCC16/bin/loongarch64-loongson-linux-gnu-g++ \
        .venv/bin/pip install --no-binary numpy --force-reinstall numpy
'
```

For the old-world venv swap `GCC16` → `GCC15`; the compiler paths remain
`$GCC15/bin/gcc` and `$GCC15/bin/g++`.

## Gotchas

- **Don't mix LD_LIBRARY_PATH across worlds.** Putting the 16.1 lib
  dir on `LD_LIBRARY_PATH` while running an old-world binary makes
  the dynamic linker load the new-world libstdc++, which DT_NEEDEDs
  the new-world loader, which the running process can't satisfy.
  Everything breaks.
- **Do not ship the toolchain wrappers' RPATH.** Build new-world wheels with
  bare `bin/loongarch64-loongson-linux-gnu-gcc` and
  `bin/loongarch64-loongson-linux-gnu-g++`, then audit every `.so` with
  `readelf -d` to ensure no build-host `RPATH` or `RUNPATH` remains.
- **Static C++ runtime is on by default for loongarch64.** Don't
  disable it unless you're certain the user box has a matching
  libstdc++ in `LD_LIBRARY_PATH` — the wheel becomes ABI-fragile
  otherwise.
- **PyPI doesn't index loongarch wheels.** GitHub Releases is the
  distribution point; the README pins specific URLs per Python
  version per world.
