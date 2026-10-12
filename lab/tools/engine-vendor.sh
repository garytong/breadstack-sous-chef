#!/usr/bin/env bash
# engine-vendor.sh — vendor the launcher engine of one cookie-cutter commit for the GCP pipeline (DESIGN §6.2,
# docs/DRIFT.md "Engine refresh"). GCP has no Bitbucket access, so the drift and retrain jobs fetch this bundle from
# the pipeline bucket instead of a checkout.
#
#   lab/tools/engine-vendor.sh <cookie-cutter checkout> <sha> [--upload] [--out DIR]
#   make -C lab engine-vendor SHA=<sha> CC=<checkout> [UPLOAD=1]
#
#   1. git archive <sha> launcher/client packages package.json package-lock.json sites/next/package.json
#      (sites/next's package.json only: the workspace list names it, so `npm ci` needs it to match the lockfile)
#   2. npm ci --ignore-scripts for the launcher-client and packages/ui workspaces (no lifecycle scripts run)
#   3. SNAPSHOT_OF = <sha> (the full commit id)
#   4. souschef_lab.enginebundle pack: browser-defaults.json (the dump of the snapshot), engine.json
#      ({sha, fingerprint, node, created_at, browser_defaults_sha256, tar_sha256}), snapshot.tar.zst
#   5. --upload: souschef_lab.enginebundle publish → $PIPELINE_BUCKET/engine/<sha>/… and engine/index.json
#
# Needs git, node ≥ 20 + npm, tar + zstd, the lab venv; --upload needs gcloud with write access to engine/ of the
# pipeline bucket (PIPELINE_BUCKET=gs://breadstack-sous-chef-int-pipeline). Prints the bundle's engine.json.
# bash, not zsh (PIPESTATUS; r4 lesson).
set -euo pipefail

usage() { echo "usage: $0 <cookie-cutter checkout> <sha> [--upload] [--out DIR]" >&2; exit 2; }
[ $# -ge 2 ] || usage
CC=$1 SHA_IN=$2; shift 2
UPLOAD= OUT=
while [ $# -gt 0 ]; do
  case "$1" in
    --upload) UPLOAD=1 ;;
    --out) OUT=$2; shift ;;
    *) usage ;;
  esac
  shift
done
LAB=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-$LAB/.venv/bin/python}
[ -x "$PY" ] || PY=python3
die() { echo "engine-vendor: $*" >&2; exit 2; }

git -C "$CC" rev-parse --git-dir >/dev/null 2>&1 || die "$CC is not a git checkout"
SHA=$(git -C "$CC" rev-parse --verify "$SHA_IN^{commit}") || die "no commit $SHA_IN in $CC (git fetch first?)"
command -v npm >/dev/null || die "npm not on PATH"
command -v zstd >/dev/null || die "zstd not on PATH"

WORK=$(mktemp -d "${TMPDIR:-/tmp}/engine-vendor-XXXXXX")
trap 'rm -rf "$WORK"' EXIT
SNAP=$WORK/snap
OUT=${OUT:-$WORK/bundle}
mkdir -p "$SNAP" "$OUT"

paths=(launcher/client packages package.json package-lock.json)
git -C "$CC" cat-file -e "$SHA:sites/next/package.json" 2>/dev/null && paths+=(sites/next/package.json)
git -C "$CC" archive "$SHA" "${paths[@]}" | tar -x -C "$SNAP"
statuses=("${PIPESTATUS[@]}")
[ "${statuses[0]}" = 0 ] && [ "${statuses[1]}" = 0 ] || die "git archive | tar failed (${statuses[*]})"

( cd "$SNAP" && npm ci --ignore-scripts --no-audit --no-fund --workspace launcher/client --workspace packages/ui ) \
  >"$WORK/npm.log" 2>&1 || { tail -20 "$WORK/npm.log" >&2; die "npm ci failed"; }
echo "$SHA" > "$SNAP/SNAPSHOT_OF"
echo "engine-vendor: $SHA snapshot $(du -sh "$SNAP" | cut -f1)"

( cd "$LAB" && "$PY" -m souschef_lab.enginebundle pack --snapshot "$SNAP" --sha "$SHA" --out "$OUT" )
if [ -n "$UPLOAD" ]; then
  [ -n "${PIPELINE_BUCKET:-}" ] || die "--upload needs PIPELINE_BUCKET=gs://…"
  ( cd "$LAB" && "$PY" -m souschef_lab.enginebundle publish --bucket "$PIPELINE_BUCKET" --dir "$OUT" )
fi
if [ "$OUT" = "$WORK/bundle" ] && [ -z "$UPLOAD" ]; then
  KEEP=$LAB/runs/engine-bundles/$SHA
  mkdir -p "$KEEP" && cp "$OUT"/* "$KEEP"/
  echo "engine-vendor: bundle kept in $KEEP (not uploaded; pass --upload)"
fi
