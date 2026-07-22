#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
site_python_dir="${TMPDIR:-/tmp}/stride-align-cloudflare-python"

cd "${repo_root}"

# Workers Builds is configured with SKIP_DEPENDENCY_INSTALL=1 so Cloudflare
# does not infer `pip install .` from pyproject.toml and compile stride-align.
# Install only Wrangler and the pure-Python Markdown renderer we actually use.
npm ci
python3 -m pip install \
  --disable-pip-version-check \
  --no-compile \
  --no-deps \
  --upgrade \
  --target "${site_python_dir}" \
  --requirement requirements-site.txt

PYTHONPATH="${site_python_dir}${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 tools/md_to_html.py
