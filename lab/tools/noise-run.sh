#!/usr/bin/env bash
# noise-run.sh EXP [ARM…] — the seed-noise experiment end to end (docs/RETRAIN.md R10; souschef_lab/retrain/noise.py).
#
#   bash lab/tools/noise-run.sh nz-20261015                     # r4@2 and r4@3 against live r4
#   bash lab/tools/noise-run.sh nz-20261015 r4@2 r4@3 r4@3407    # + a same-seed re-run of r4 on the trainer image
#   DRY=1 bash lab/tools/noise-run.sh nz-20261015                # fakes on a local bucket: no network, no spend
#
# Steps, each idempotent (re-run the script after any stop; it continues where the records in
# runs/<arm>/noise.json say it stopped):
#   plan → setup → train --wait (Vertex, the arms in parallel) → build --wait (Cloud Build) → eval (one arm at a
#   time, under state/locks/retrain.json + gpu-eval.json) → analyze (noise/<EXP>/result.json)
#
# A person runs it (it spends: about US$50–60 for two arms; `plan` prints the estimate first and the script asks
# before anything is created). Needs gcloud logged in as someone who may create Vertex jobs and Cloud Builds, update and invoke
# the two GPU services and execute job souschef-eval in breadstack-sous-chef-int; the lab venv; bash, not zsh.
# Env: PIPELINE_BUCKET (default gs://breadstack-sous-chef-int-pipeline), BASE_RUN (default r5's round),
#      TRAINER_IMAGE (a trainer whose pipeline-train.sh honours TRAIN_SEED; default the base round's — checked after
#      training: an arm trained with another seed is stopped as seed_mismatch), RUNG (default h100-flex),
#      LAB_IMAGE (refuse to eval unless job souschef-eval runs it), YES=1 (do not ask).
# Long: training 3–5 h, then about 3 h of eval per arm. Run it in tmux or under nohup; Ctrl-C between polls is safe
# (the eval's locks are released; re-run to re-take them and keep polling the same execution).
set -euo pipefail

EXP=${1:?usage: noise-run.sh EXP [DATA@SEED …]   (EXP = nz-<YYYYMMDD>[-…])}
shift
ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(r4@2 r4@3)
LAB=${SOUS_CHEF_LAB:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${PY:-$LAB/.venv/bin/python}
ARMF=()
for a in "${ARMS[@]}"; do ARMF+=(--arm "$a"); done

if [ -n "${DRY:-}" ]; then
  W=${WORKDIR:-$(mktemp -d)}
  echo "noise-run: DRY on $W/bucket"
  cd "$LAB" && exec "$PY" -m souschef_lab.retrain.noise dry --workdir "$W" --exp "$EXP" "${ARMF[@]}"
fi

export PIPELINE_BUCKET=${PIPELINE_BUCKET:-gs://breadstack-sous-chef-int-pipeline}
BASE_RUN=${BASE_RUN:-rt-20261014-0708-7cac28-1184ff9a}
N() { (cd "$LAB" && "$PY" -m souschef_lab.retrain.noise "$@"); }

N plan --exp "$EXP" "${ARMF[@]}" --base-run "$BASE_RUN" ${RUNG:+--rung "$RUNG"}
if [ -z "${YES:-}" ]; then
  read -r -p "noise-run: create the run folders and start training? [y/N] " ok
  [ "$ok" = y ] || { echo "noise-run: stopped before anything was created"; exit 0; }
fi
N setup --exp "$EXP" "${ARMF[@]}" --base-run "$BASE_RUN" ${TRAINER_IMAGE:+--trainer-image "$TRAINER_IMAGE"}
N train --exp "$EXP" "${ARMF[@]}" --wait ${RUNG:+--rung "$RUNG"}
N build --exp "$EXP" "${ARMF[@]}" --wait
N eval --exp "$EXP" "${ARMF[@]}" ${LAB_IMAGE:+--lab-image "$LAB_IMAGE"}
N analyze --exp "$EXP" "${ARMF[@]}" --compare "$BASE_RUN"
N status --exp "$EXP" "${ARMF[@]}"
