#!/usr/bin/env bash
# Idempotent environment setup for the gold/silver commodities analysis scripts.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# The default base image may ship Python without the venv/ensurepip module.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends python3-venv
fi

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
