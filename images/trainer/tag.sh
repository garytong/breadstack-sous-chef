#!/usr/bin/env bash
# Prints the trainer image's content tag, t-<12 hex>: sha256 over the path
# and contents of every file the trainer image copies (Dockerfile.dockerignore:
# the lab without tests, data, runs, venvs and caches, plus images/trainer).
# Run from anywhere; CI and `make trainer-image` use it to skip a build whose
# tag already exists.
set -euo pipefail
cd "$(dirname "$0")/../.."
find lab images/trainer \
  \( -name __pycache__ -o -name '*.egg-info' -o -path lab/data -o -path lab/runs -o -path lab/tests \
     -o -path lab/.venv -o -path lab/.venv-train -o -path lab/.cache -o -path lab/unsloth_compiled_cache \
     -o -path lab/.pytest_cache \) -prune -o \( -type f -o -type l \) -print |
  LC_ALL=C sort |
  while IFS= read -r f; do
    if [ -L "$f" ]; then printf '%s -> %s\n' "$f" "$(readlink "$f")"; else printf '%s %s\n' "$f" "$(sha256sum <"$f" | cut -c1-64)"; fi
  done |
  sha256sum | cut -c1-12 | sed 's/^/t-/'
