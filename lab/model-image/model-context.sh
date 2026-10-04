#!/bin/sh
# Prints the directory that `make -C lab model-build` hands the image build
# as its "model" context (lab/model-image/Dockerfile), after checking that
# it holds a Modelfile and the one GGUF its FROM line names, and nothing
# else. Everything in that directory is sent to the builder and lands in
# the image, so no merged safetensors, a second quant, export.json or
# subdirectories may sit beside them.
#
# export.py writes an export (lab/runs/exports/<name>/) with the q4_k_m and
# q8_0 GGUFs, a Modelfile per quant (Modelfile.q4_k_m, Modelfile.q8_0) and
# export.json side by side, so an export directory never passes this check
# as it is: `make model-stage` hard-links the one GGUF and writes the
# rewritten Modelfile into a staging directory (lab/runs/model-image/<name>/),
# and that is what this script takes.
#
#   sh lab/model-image/model-context.sh <staging directory>
set -eu

fail() {
  echo "sous-chef model: $*" >&2
  exit 1
}

dir=${1:-}
[ -n "$dir" ] || fail "usage: model-context.sh <staging directory>"
[ -d "$dir" ] || fail "no staging directory at $dir: make -C lab model-stage writes it (MODEL_STAGE=<dir> for another)"
[ -f "$dir/Modelfile" ] || fail "no Modelfile in $dir: the directory must hold the GGUF and its Modelfile (make -C lab model-stage writes both)"
dir=$(cd "$dir" && pwd -P)

# One FROM, naming a GGUF beside the Modelfile. An ADAPTER line would need
# a second weights file in the build: merge the adapter first (export.py
# does).
[ "$(grep -Ec '^FROM[[:space:]]' "$dir/Modelfile")" -eq 1 ] || fail "$dir/Modelfile must have exactly one FROM line"
! grep -Eq '^ADAPTER[[:space:]]' "$dir/Modelfile" || fail "$dir/Modelfile has an ADAPTER line: merge the adapter into the GGUF (export.py does) so the image holds one weights file"
from=$(sed -n -E 's/^FROM[[:space:]]+//p' "$dir/Modelfile" | sed -E 's/[[:space:]]+$//')
case "$from" in
  ./*.gguf) gguf=${from#./} ;;
  *) fail "$dir/Modelfile must say FROM ./<file>.gguf (the GGUF beside it), not FROM $from" ;;
esac
case "$gguf" in
  */*) fail "$dir/Modelfile's GGUF must sit beside it, not in a subdirectory: FROM $from" ;;
esac
[ -f "$dir/$gguf" ] || fail "$dir/Modelfile names $gguf, which is not there"
# The build context sends a symlink as a link, not the file it points at.
[ ! -L "$dir/$gguf" ] || fail "$dir/$gguf is a symlink: put the file itself there (a hard link will do, as make model-stage makes; keep it under lab/runs/, which git ignores, or outside the repo)"
[ "$(head -c 4 "$dir/$gguf")" = GGUF ] || fail "$dir/$gguf is not a GGUF file"
# The Dockerfile asserts this after `ollama create`; catching it here saves
# the import of a 5 GB file.
grep -Eq '^PARAMETER[[:space:]]+num_ctx[[:space:]]+65536[[:space:]]*$' "$dir/Modelfile" || fail "$dir/Modelfile lacks PARAMETER num_ctx 65536 (make model-stage writes it)"

extra=$(ls -A "$dir" | grep -vxF -e Modelfile -e "$gguf" || true)
[ -z "$extra" ] || fail "$dir holds more than the Modelfile and $gguf, and all of it would go into the image ($(echo "$extra" | paste -sd ' ' -)): move the rest out, or run make -C lab model-stage again (it starts from an empty staging directory)"

size=$(wc -c <"$dir/$gguf")
echo "sous-chef model: model context $dir: Modelfile + $gguf ($(awk -v b="$size" 'BEGIN { printf "%.2f", b / 1e9 }') GB)" >&2
echo "$dir"
