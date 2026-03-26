#!/usr/bin/env bash
# Build sdist + wheel and upload to PyPI using an API token from the environment.
#
# Supported credentials (first match wins):
#   PYPI_TOKEN          → TWINE_USERNAME=__token__ TWINE_PASSWORD=<value>
#   TWINE_PASSWORD      → use as-is (set TWINE_USERNAME=__token__ if using API token)
#
# Usage:
#   export PYPI_TOKEN=pypi-xxxxxxxx
#   ./scripts/publish_pypi.sh
#
# Or dry-run (build only):
#   ./scripts/publish_pypi.sh --no-upload

set -euo pipefail
cd "$(dirname "$0")/.."

UPLOAD=1
if [[ "${1:-}" == "--no-upload" ]]; then
  UPLOAD=0
fi

python3 -m pip install -q build twine

rm -rf dist/
python3 -m build

if [[ "$UPLOAD" -eq 0 ]]; then
  echo "Built: $(ls -1 dist/)"
  exit 0
fi

if [[ -n "${PYPI_TOKEN:-}" ]]; then
  export TWINE_USERNAME=__token__
  export TWINE_PASSWORD="${PYPI_TOKEN}"
elif [[ -n "${TWINE_PASSWORD:-}" ]]; then
  export TWINE_USERNAME="${TWINE_USERNAME:-__token__}"
else
  echo "error: set PYPI_TOKEN or TWINE_PASSWORD for upload" >&2
  exit 1
fi

python3 -m twine upload dist/*
