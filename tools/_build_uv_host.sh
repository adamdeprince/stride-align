#!/bin/bash
# Provision uv + Python 3.12/3.13/3.14 and build per-version wheels from the
# 0.5.1 sdist at /tmp/stride_align-0.5.1.tar.gz. Used for the x86_64 (avx10)
# and aarch64 (graviton5) release hosts.
set -uo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh >/tmp/uv-install.log 2>&1 \
    || { echo "UV INSTALL FAILED"; tail -8 /tmp/uv-install.log; exit 1; }
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version 2>&1)"
uv python install 3.12 3.13 3.14 2>&1 | tail -3
rm -rf ~/sa-build && mkdir -p ~/sa-build ~/wheelhouse && cd ~/sa-build
tar xzf /tmp/stride_align-0.5.1.tar.gz
for py in 3.12 3.13 3.14; do
  echo "=== build cp$py ($(date +%H:%M:%S)) ==="
  uv build --wheel --python "$py" stride_align-0.5.1 --out-dir ~/wheelhouse 2>&1 \
    | grep -E "Created|error|Error|failed|FAILED" | tail -3
done
echo "ALLWHEELS:"; ls -1 ~/wheelhouse/*.whl 2>/dev/null | sed 's#.*/##'
echo "DONE $(date +%H:%M:%S)"
