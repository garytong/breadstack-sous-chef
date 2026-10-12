#!/usr/bin/env bash
# The trainer image's entrypoint (Vertex AI custom training, DESIGN §3.5).
#
#   <run id> <full|smoke>   the self-test, then lab/tools/pipeline-train.sh
#                           (pull the dataset and base weights from the
#                           pipeline bucket, SFT, export, stage, push, DONE)
#   selftest [--upload <run id>] [--no-gpu]
#                           the GPU environment check alone
#   shell                   bash (debugging: docker run -it ... shell)
#
# Env from the workflow: PIPELINE_BUCKET, GOOGLE_CLOUD_PROJECT, TRAIN_RUNG,
# TRAIN_ATTEMPT. No secret: the trainer's account reads and writes the
# pipeline bucket only.
#
# bash, not sh: pipeline-train.sh reads PIPESTATUS (the r4 lesson).
set -uo pipefail

# CUDA forward compatibility: torch's cu130 wheels need a 580+ driver. On an
# older host driver, the image's cuda-compat-13-0 libcuda goes first (a
# newer host driver must NOT get it: the compat library would then be the
# older one).
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)
major=${driver%%.*}
if [[ "$major" =~ ^[0-9]+$ ]] && [ "$major" -lt 580 ] && [ -d /usr/local/cuda/compat ]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  echo "trainer: host driver $driver < 580: CUDA forward compatibility on (/usr/local/cuda/compat)"
else
  echo "trainer: host driver ${driver:-unknown}"
fi

case "${1:-}" in
  selftest) shift; exec /opt/lab/.venv-train/bin/python /opt/trainer/selftest.py "$@" ;;
  shell) exec bash ;;
  ""|-h|--help) echo "usage: <run id> <full|smoke> | selftest [--upload <run id>] [--no-gpu] | shell"; exit 2 ;;
esac

run_id=$1
mode=${2:-full}
case "$mode" in full|smoke) ;; *) echo "trainer: mode must be full or smoke, not $mode"; exit 2 ;; esac
[ -n "${PIPELINE_BUCKET:-}" ] || { echo "trainer: PIPELINE_BUCKET is not set"; exit 2; }
# the bucket as gs://name (the workflow may pass the bare name)
case "$PIPELINE_BUCKET" in gs://*) ;; *) PIPELINE_BUCKET=gs://$PIPELINE_BUCKET ;; esac
export PIPELINE_BUCKET=${PIPELINE_BUCKET%/}

# Every attempt proves the GPU stack first (a minute), so an incompatible
# driver fails here — recorded in train/progress.json as a failure of the
# trainer, which the workflow does not retry on another machine — and not
# an hour into the run.
if ! /opt/lab/.venv-train/bin/python /opt/trainer/selftest.py --upload "$run_id"; then
  now=$(date -u +%FT%TZ)
  printf '{"status":"failed","stage":"selftest","error":"the trainer self-test failed (%s/runs/%s/train/selftest.json)","updated_at":"%s","rung":"%s","attempt":"%s"}\n' \
    "$PIPELINE_BUCKET" "$run_id" "$now" "${TRAIN_RUNG:-}" "${TRAIN_ATTEMPT:-}" |
    gcloud storage cp - "$PIPELINE_BUCKET/runs/$run_id/train/progress.json" --content-type=application/json >/dev/null 2>&1 || true
  exit 1
fi

[ -f /opt/lab/tools/pipeline-train.sh ] || { echo "trainer: this image has no lab/tools/pipeline-train.sh"; exit 2; }
exec bash /opt/lab/tools/pipeline-train.sh "$run_id" "$mode"
