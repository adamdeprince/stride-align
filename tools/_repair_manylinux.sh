#!/bin/bash
# auditwheel-repair the native linux wheels in ~/wheelhouse into manylinux
# wheels under ~/wheelhouse-manylinux. Run on the matching arch host
# (avx10 = x86_64, graviton5 = aarch64). Bundles libstdc++/libgcc and
# relabels to the manylinux policy the host's glibc supports.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v patchelf   >/dev/null 2>&1 || uv tool install patchelf   2>&1 | tail -1
command -v auditwheel >/dev/null 2>&1 || uv tool install auditwheel 2>&1 | tail -1
echo "patchelf:   $(command -v patchelf || echo MISSING)"
echo "auditwheel: $(auditwheel --version 2>&1)"
rm -rf ~/wheelhouse-manylinux && mkdir -p ~/wheelhouse-manylinux
for w in ~/wheelhouse/*.whl; do
  [ -e "$w" ] || continue
  echo "--- repair $(basename "$w") ---"
  auditwheel repair "$w" -w ~/wheelhouse-manylinux 2>&1 \
    | grep -iE "Repairing|Fixed-up|to be|manylinux|error|cannot|warning" | tail -3
done
echo "=== manylinux wheels ==="
ls -1 ~/wheelhouse-manylinux/*.whl 2>/dev/null | sed 's#.*/##'
echo "DONE"
