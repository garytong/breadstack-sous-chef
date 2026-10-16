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
# TRAIN_ATTEMPT, TRAIN_NONCE (the job's one-off `create` label), TRAIN_IMAGE.
# Optional, read by pipeline-train.sh (unset: one arm, base:3407, as r4/r5):
# TRAIN_ARMS (base:<seed> / warm:<seed>, e.g. base:3407,base:2,warm:3407: each
# trained in turn, the best on validation kept), TRAIN_SEED, TRAIN_SEEDS,
# INIT_ADAPTER, INIT_LR, TRAIN_GEN_CHECK, TRAIN_WINDOW_S, TRAIN_TAIL_S — or
# the same keys (arms, seed, seeds, init_adapter, init_lr, gen_check) in
# runs/<id>/train-config.json, which needs no change to the workflow's job
# spec (pipeline-train.sh's header has the details).
# No secret: the trainer's account reads and writes the pipeline bucket only.
#
# bash, not sh: pipeline-train.sh reads PIPESTATUS (the r4 lesson).
set -uo pipefail

# gcloud storage without parallel composite uploads: the trainer's account has
# no storage.buckets.get for gcloud's compatibility check, and the temporary
# components would be written to gcloud/tmp/… at the bucket root, outside
# runs/<id>/train/ (the only prefix it may write), and never cleaned up.
export CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED=${CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED:-False}

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

# train/progress.json for the workflow's vertex_attempt (pipeline-train.sh
# writes it from here on): progress STATUS ERROR
progress() {
  local body
  body=$(printf '{"status":"%s","step":"selftest","stage":"selftest","error":"%s","updated_at":"%s","rung":"%s","attempt":"%s","mode":"%s","nonce":"%s"}' \
    "$1" "$2" "$(date -u +%FT%TZ)" "${TRAIN_RUNG:-}" "${TRAIN_ATTEMPT:-1}" "$mode" "${TRAIN_NONCE:-}")
  printf '%s\n' "$body" |
    gcloud storage cp - "$PIPELINE_BUCKET/runs/$run_id/train/progress.json" --content-type=application/json --quiet >/dev/null 2>&1 || true
  # a failure also leaves train/failed-<nonce>.json, which a restart of this job never rewrites
  if [ "$1" = failed ] && [ -n "${TRAIN_NONCE:-}" ]; then
    printf '%s\n' "$body" |
      gcloud storage cp - "$PIPELINE_BUCKET/runs/$run_id/train/failed-$TRAIN_NONCE.json" --content-type=application/json --quiet >/dev/null 2>&1 || true
  fi
}

# Vertex restarts a container that exited non-zero, even after it has marked
# the job FAILED (the first SMOKE, 2026-10-14: job 8149263396877893632 ran
# three times; the restarts rewrote progress.json `failed` as `running` and
# `interrupted`, so the workflow retried a deterministic failure as a
# preemption, and a restart still ran beside the next attempt's job). This
# job already failed (train/failed-<nonce>.json, written by this job only):
# exit 1 at once and write nothing. A preemption or a restart after one
# leaves no such file, and resumes from the checkpoints as before.
if [ -n "${TRAIN_NONCE:-}" ] && gcloud storage ls "$PIPELINE_BUCKET/runs/$run_id/train/failed-$TRAIN_NONCE.json" >/dev/null 2>&1; then
  echo "trainer: job ${TRAIN_NONCE} already failed ($PIPELINE_BUCKET/runs/$run_id/train/failed-$TRAIN_NONCE.json): a restart of a failed job, not run again"
  exit 1
fi

# A SIGTERM during the self-test (a Spot preemption, a Vertex restart or
# cancel) is recorded as `interrupted`, like pipeline-train.sh's trap (this
# shell is PID 1 in the container: without a trap the signal is ignored and
# the attempt dies later as `running`). The self-test runs in the background
# and is `wait`ed for, so the trap runs at once.
selftest_pid=
on_signal() {
  trap - TERM INT
  [ -n "$selftest_pid" ] && kill -TERM "$selftest_pid" 2>/dev/null
  progress interrupted "signal during selftest"
  exit 143
}
trap on_signal TERM INT

# Every attempt proves the GPU stack first (a minute), so an incompatible
# driver fails here — recorded in train/progress.json as a failure of the
# trainer, which the workflow does not retry on another machine — and not
# an hour into the run.
/opt/lab/.venv-train/bin/python /opt/trainer/selftest.py --upload "$run_id" &
selftest_pid=$!
wait "$selftest_pid"
rc=$?
selftest_pid=
if [ "$rc" -ne 0 ]; then
  progress failed "the trainer self-test failed ($PIPELINE_BUCKET/runs/$run_id/train/selftest.json)"
  exit 1
fi
trap - TERM INT

[ -f /opt/lab/tools/pipeline-train.sh ] || { echo "trainer: this image has no lab/tools/pipeline-train.sh"; exit 2; }
exec bash /opt/lab/tools/pipeline-train.sh "$run_id" "$mode"
