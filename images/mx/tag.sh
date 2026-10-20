#!/usr/bin/env bash
# Prints the sous-chef-mx image's content tag, m-<12 hex>: sha256 over the path and contents of every file the image
# copies (images/mx/Dockerfile.dockerignore: the lab without tests, data, runs, venvs, caches and env files) plus
# images/mx itself (the Dockerfile pins both bases by digest). A tag names exactly one tree, so an uncommitted
# worktree gets its own tag and a rebuild of the same tree is a no-op (images/mx/cloudbuild.yaml). Like
# images/trainer/tag.sh; run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../.."
find lab images/mx \
  \( -name __pycache__ -o -name .ruff_cache -o -name .pytest_cache -o -name '*.egg-info' -o -name '.env' -o -name '.env.*' -o -path lab/data -o -path lab/runs \
     -o -path lab/tests -o -path lab/.venv -o -path lab/.venv-train -o -path lab/.cache -o -path lab/unsloth_compiled_cache \
     -o -path lab/.pytest_cache -o -path lab/.ruff_cache \) -prune -o \( -type f -o -type l \) -print |
  LC_ALL=C sort |
  while IFS= read -r f; do
    if [ -L "$f" ]; then printf '%s -> %s\n' "$f" "$(readlink "$f")"; else printf '%s %s\n' "$f" "$(sha256sum <"$f" | cut -c1-64)"; fi
  done |
  sha256sum | cut -c1-12 | sed 's/^/m-/'
