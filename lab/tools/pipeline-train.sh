#!/usr/bin/env bash
# pipeline-train.sh RUN_ID [full|smoke] — the trainer image's entry on Vertex AI custom training (design §3.5, I6).
# tools/run-r4.sh adapted to the pipeline bucket: everything comes from and goes to gs://…/runs/<RUN_ID>/.
#
#   1  pull runs/<id>/dataset (r4dataset's layout) and base/unsloth-qwen3.5-9b-hf.tar (+ .sha256: sha256sum -c)
#   2  pull runs/<id>/train/checkpoints/ when there (a resume after a Spot preemption / a Vertex restart)
#   3  row checks (blocks, follow-ups, 0 preview replies, 0 drifted blocks) and the dry plan
#   4  the resume smoke (2 updates with a checkpoint, then RESUMED to 4) — first attempt only
#   5  the SFT with --checkpoint-steps 50 --resume; checkpoints pushed every 120 s by a background loop
#      (SMOKE: --max-steps 4 --checkpoint-steps 2, then a resume to 6)
#   6  export q4_k_m with base-modelfiles/qwen3.5-9b.modelfile (export.py --method gguf: the image's prebuilt llama.cpp)
#   7  make model-stage NAME=<id> MODEL_TEMPERATURE=0.3 MODEL_PRESENCE_PENALTY=0 (Modelfile: temperature 0.3,
#      presence_penalty 0, top_k 20, top_p 0.95 from the base, num_ctx 65536)
#   8  push adapter/, export/, stage/ to runs/<id>/train/
#   9  DONE + progress.json — exit 0 only after DONE
#
# Every step rewrites runs/<id>/train/progress.json {status, step, error, attempt, mode, updated_at, last_step, …}
# — the shape the retrain workflow's vertex_attempt reads: status running | failed | interrupted | done. A failure
# records status "failed" with the step in `error`, pushes the log (runs/<id>/train/train.log) and exits 1 (the
# workflow classes it fatal: never retried on another machine). A SIGTERM (a Spot preemption, a Vertex restart or
# cancel) records status "interrupted" FIRST (a Spot VM gets about 30 s before it is switched off: the one small
# object the workflow classes by goes before anything slow), then stops the running step and pushes the
# checkpoints (the workflow classes it preempted: resumed from the checkpoints). During the SFT the 120 s checkpoint
# loop also rewrites progress.json (status running, last_step from loss.jsonl), so a VM that dies before the trap
# finishes still shows the steps trained since the attempt began. Rerunning is safe: a run with DONE exits 0 at once.
# The recipe is r4's (train.py RECIPE: Qwen3.5-9B bf16, fp32 LoRA r16/α16, lr 2e-4, 1 epoch, batch 1 × grad_accum 4,
# adamw_8bit, warmup 3 %, max_seq 32,768). bash, not zsh: the step runner relies on bash's wait and traps.
set -uo pipefail

RUN=${1:?usage: pipeline-train.sh RUN_ID [full|smoke]}
MODE=${2:-full}
case "$MODE" in full|smoke) ;; *) echo "pipeline-train: mode must be full or smoke" >&2; exit 2;; esac
B=${PIPELINE_BUCKET:?PIPELINE_BUCKET (gs://… or the bare bucket name) is required}
case "$B" in gs://*) ;; *) B=gs://$B ;; esac
B=${B%/}
LAB=${SOUS_CHEF_LAB:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$LAB" || exit 2
PY=${TRAINPY:-$LAB/.venv-train/bin/python}
GCLOUD=${GCLOUD:-gcloud}
W=${PIPELINE_WORKDIR:-/tmp/pipeline-train}
export HF_HOME=${HF_HOME:-$W/hf}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NAME=$RUN
R=$B/runs/$RUN/train
DATA=$LAB/data/$RUN
ADAPTER_DIR=runs/adapters/$NAME-sft
CKPT=${CKPT:-50}
BASE_TAR=${BASE_TAR:-unsloth-qwen3.5-9b-hf.tar}
BASE_MODELFILE=base-modelfiles/qwen3.5-9b.modelfile
LOG=$W/train.log
# the workflow sends TRAIN_ATTEMPT (vertex_attempt's try number); CLOUD_ML_TRIAL_ID is a hyperparameter-tuning
# variable a CustomJob never sets
ATTEMPT=${TRAIN_ATTEMPT:-${PIPELINE_ATTEMPT:-1}}
SYNC_EVERY=${PIPELINE_SYNC_EVERY_S:-120}
FLAGS=(--data "$DATA/dataset" --fields-context on --eval-steps "${EVAL_STEPS:-150}" --val-max-rows "${VAL_MAX_ROWS:-150}")
mkdir -p "$W" "$HF_HOME"
CUR_STEP=start
SYNC_PID=

# parallel composite uploads off: the trainer account has no storage.buckets.get for gcloud's check, and the
# components would go to gcloud/tmp/… at the bucket root, outside runs/<id>/train/ (the only prefix it may write)
export CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED=${CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED:-False}
gs() { "$GCLOUD" storage "$@" --quiet; }
gs_exists() { "$GCLOUD" storage ls "$1" >/dev/null 2>&1; }

progress() {  # progress STATUS [ERROR] — the step, the status and what the adapter's run.json / loss.jsonl say so far
  # (a file of its own per caller: the checkpoint loop writes `running` from a subshell while the main shell may
  # be writing `interrupted`)
  local pj="$W/progress.$BASHPID.json"
  rm -f "$pj"
  "$PY" - "$pj" "$CUR_STEP" "$1" "$ATTEMPT" "$MODE" "$ADAPTER_DIR" "${2:-}" "${TRAIN_RUNG:-}" <<'EOF' >/dev/null 2>&1 || true
import json, sys, time
from pathlib import Path
out, step, status, attempt, mode, adir, error, rung = sys.argv[1:9]
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
p = {"status": status, "step": step, "error": error or None, "attempt": attempt, "rung": rung or None, "mode": mode,
     "updated_at": now, "at": now}
ad = Path(adir)
run = ad / "run.json"
if run.is_file():
    try:
        r = json.loads(run.read_text())
        for k in ("updates", "tokens_per_s", "peak_mem_gb", "wall_s", "train_loss"):
            if k in r:
                p[k] = r[k]
        p["updates"] = p.get("updates") or (r.get("plan") or {}).get("updates")
    except ValueError:
        pass
evals = []
for f in (ad / "loss.jsonl",):
    if f.is_file():
        for line in f.read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if "eval_loss" in d:
                evals.append(d["eval_loss"])
            if "step" in d:
                p["last_step"] = d["step"]
if evals:
    p["eval_loss_first"], p["eval_loss_last"] = evals[0], evals[-1]
Path(out).write_text(json.dumps(p, indent=1))
EOF
  if [ ! -s "$pj" ]; then
    # no python (or it failed): the status the workflow reads must still be there
    printf '{"status":"%s","step":"%s","error":"%s","attempt":"%s","updated_at":"%s"}\n' \
      "$1" "$(printf '%s' "$CUR_STEP" | tr -d '"\\')" "$(printf '%s' "${2:-}" | tr -d '"\\')" "$ATTEMPT" \
      "$(date -u +%FT%TZ)" > "$pj"
  fi
  gs cp "$pj" "$R/progress.json" --content-type=application/json >/dev/null 2>&1 || true
  cp -f "$pj" "$W/progress.json" 2>/dev/null; rm -f "$pj"
}

step() { CUR_STEP=$1; printf '%s' "$1" > "$W/step"; echo "== $(date -u +%FT%TZ) $* ==" | tee -a "$LOG"; progress running; }
# run CMD… — the command in the background and `wait`ed for, so a SIGTERM reaches the trap at once (bash runs a
# trap only between foreground commands: a 2-hour SFT would otherwise outlive Vertex's grace period)
CHILD=
run() {
  "$@" <&0 > >(tee -a "$LOG") 2>&1 &
  CHILD=$!
  wait "$CHILD"
  local rc=$?
  CHILD=
  return "$rc"
}
fail() {
  echo "== $(date -u +%FT%TZ) FAILED: $* ==" | tee -a "$LOG"
  stop_sync
  push_checkpoints
  progress failed "$CUR_STEP: $*"
  gs cp "$LOG" "$R/train.log" >/dev/null 2>&1 || true
  exit 1
}
kill_tree() {  # kill_tree PID — TERM the step and everything it started (dataloader workers, a wrapper's children)
  local c
  if command -v pgrep >/dev/null 2>&1; then
    for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done
  fi
  kill -TERM "$1" 2>/dev/null
}
interrupted() {  # SIGTERM / SIGINT: a preemption or a cancel — say so first, then keep what was trained, leave
  trap - TERM INT
  echo "== $(date -u +%FT%TZ) INTERRUPTED (signal) during $CUR_STEP ==" | tee -a "$LOG"
  # the step gets its TERM now (it stops while the status goes up), the loop is stopped so its `running` cannot
  # land after `interrupted`, and the status goes up before the slow part (waiting for the step, the rsync)
  [ -n "$CHILD" ] && kill_tree "$CHILD"
  stop_sync
  progress interrupted "signal during $CUR_STEP"
  [ -n "$CHILD" ] && wait "$CHILD" 2>/dev/null
  push_checkpoints
  gs cp "$LOG" "$R/train.log" >/dev/null 2>&1 || true
  exit 143
}
trap interrupted TERM INT
push_checkpoints() {
  [ -d "$ADAPTER_DIR/checkpoints" ] && gs rsync --recursive "$ADAPTER_DIR/checkpoints" "$R/checkpoints" >/dev/null 2>&1
  return 0
}
sync_loop() {  # every SYNC_EVERY s: the checkpoints, then progress `running` (last_step from loss.jsonl moves on)
  trap - TERM INT
  while sleep "$SYNC_EVERY"; do
    [ -e "$W/.sync-stop" ] && break
    push_checkpoints
    [ -e "$W/.sync-stop" ] && break
    CUR_STEP=$(cat "$W/step" 2>/dev/null || printf '%s' "$CUR_STEP")
    progress running
  done
}
start_sync() { rm -f "$W/.sync-stop"; sync_loop & SYNC_PID=$!; }
stop_sync() {  # the loop and whatever it is running (a gcloud upload), then wait for it: nothing of it lands later
  [ -n "$SYNC_PID" ] || return 0
  touch "$W/.sync-stop"
  kill_tree "$SYNC_PID"
  wait "$SYNC_PID" 2>/dev/null
  SYNC_PID=
}

if gs_exists "$R/DONE"; then echo "pipeline-train: $R/DONE exists — nothing to do"; exit 0; fi

step "kit"; run cat KIT-VERSION 2>/dev/null; run "$PY" --version
step "pull dataset"
mkdir -p "$DATA"
run gs rsync --recursive "$B/runs/$RUN/dataset" "$DATA" || fail "pull dataset"
step "pull base"
if [ ! -f "$HF_HOME/.base-ok" ]; then
  run gs cp "$B/base/$BASE_TAR" "$B/base/$BASE_TAR.sha256" "$W/" || fail "pull base"
  run bash -c 'cd "$1" && sha256sum -c "$2"' _ "$W" "$BASE_TAR.sha256" || fail "base tar sha256"
  run tar -x -C "$HF_HOME" -f "$W/$BASE_TAR" || fail "untar base"
  rm -f "$W/$BASE_TAR"
  touch "$HF_HOME/.base-ok"
fi
step "pull checkpoints"
RESUMED=0
if gs_exists "$R/checkpoints"; then
  mkdir -p "$ADAPTER_DIR/checkpoints"
  run gs rsync --recursive "$R/checkpoints" "$ADAPTER_DIR/checkpoints" || fail "pull checkpoints"
  RESUMED=1
fi

step "rows"
run "$PY" - "$DATA/dataset" "$MODE" <<'EOF' || fail "row checks"
import json, sys
from pathlib import Path
from souschef_lab.r4score import mentions_preview   # the ONE preview rule the scorer and the build use
d, mode = Path(sys.argv[1]), sys.argv[2]
st = json.loads((d / "stats.json").read_text())
follow_all = 0
for split in ("train", "val"):
    rows = [json.loads(l) for l in (d / f"{split}.jsonl").read_text().splitlines() if l.strip()]
    block = sum(1 for r in rows if (r.get("meta") or {}).get("fields_context"))
    follow = sum(1 for r in rows if (r.get("meta") or {}).get("history_assistant_turns"))
    follow_all += follow
    drift = sum(1 for r in rows if (r.get("meta") or {}).get("block_drift"))
    preview = sum(1 for r in rows for m in r["messages"] if m["role"] == "assistant" and mentions_preview(m.get("content")))
    print(f"{split}: {len(rows)} rows, {block} with a candidate-fields block, {follow} follow-ups, "
          f"{preview} replies mentioning the preview, {drift} with block drift")
    assert rows and preview == 0 and drift == 0, "no rows, a reply that mentions the preview, or a drifted block"
assert mode == "smoke" or follow_all > 0, "no follow-up rows"
assert st.get("block_drift_rows") == 0 and st.get("preview_claim_kept") == 0, "stats.json: drift or preview rows kept"
print(json.dumps(st.get("by_family")))
EOF
step "dry plan"; run "$PY" -m souschef_lab.train "${FLAGS[@]}" --name "$NAME-sft" --checkpoint-steps "$CKPT" --dry-run || fail "dry plan"

if [ "$MODE" = full ] && [ "$RESUMED" = 0 ] && [ ! -f "$ADAPTER_DIR/adapter/adapter_config.json" ]; then
  rm -rf "runs/adapters/$NAME-smoke-sft"
  step "resume smoke: 2 updates, checkpoint at 2"
  run "$PY" -m souschef_lab.train "${FLAGS[@]}" --name "$NAME-smoke-sft" --max-steps 2 --checkpoint-steps 2 || fail "smoke"
  step "resume smoke: resume to 4"
  run "$PY" -m souschef_lab.train "${FLAGS[@]}" --name "$NAME-smoke-sft" --max-steps 4 --checkpoint-steps 2 --resume || fail "smoke resume"
  grep -Eq "Resuming from .*$NAME-smoke-sft/checkpoints/checkpoint-2" "$LOG" || fail "the smoke did not resume from its checkpoint"
  rm -rf "runs/adapters/$NAME-smoke-sft"
fi

if [ ! -f "$ADAPTER_DIR/adapter/adapter_config.json" ]; then
  start_sync
  trap '[ -n "$SYNC_PID" ] && kill_tree "$SYNC_PID" 2>/dev/null' EXIT
  if [ "$MODE" = smoke ]; then
    step "SFT (smoke: 4 updates, checkpoint every 2)"
    run "$PY" -m souschef_lab.train "${FLAGS[@]}" --name "$NAME-sft" --max-steps 4 --checkpoint-steps 2 --resume || fail "SFT"
    step "SFT (smoke: resume to 6)"
    rm -rf "$ADAPTER_DIR/adapter"
    run "$PY" -m souschef_lab.train "${FLAGS[@]}" --name "$NAME-sft" --max-steps 6 --checkpoint-steps 2 --resume || fail "SFT resume"
    grep -Eq "Resuming from .*$NAME-sft/checkpoints/checkpoint-" "$LOG" || fail "the smoke SFT did not resume"
  else
    step "SFT ($NAME-sft; resumes from the newest checkpoint)"
    run "$PY" -m souschef_lab.train "${FLAGS[@]}" --name "$NAME-sft" --checkpoint-steps "$CKPT" --resume || fail "SFT"
  fi
  stop_sync
  push_checkpoints
fi

step "export q4_k_m"
run "$PY" -m souschef_lab.export --adapter "$ADAPTER_DIR/adapter" --name "$NAME" --quants q4_k_m \
  --base-modelfile "$BASE_MODELFILE" || fail "export"
step "stage"
run make --no-print-directory model-stage NAME="$NAME" MODEL_STAGE="runs/model-image/$NAME" \
  MODEL_TEMPERATURE=0.3 MODEL_PRESENCE_PENALTY=0 || fail "model-stage"
step "push"
run gs rsync --recursive --exclude='^checkpoints/.*' "$ADAPTER_DIR" "$R/adapter" || fail "push adapter"
run gs rsync --recursive "runs/exports/$NAME" "$R/export" || fail "push export"
run gs rsync --recursive "runs/model-image/$NAME" "$R/stage" || fail "push stage"
CUR_STEP=DONE
progress done
date -u +%FT%TZ > "$W/DONE"
gs cp "$W/DONE" "$R/DONE" || fail "DONE"
gs cp "$LOG" "$R/train.log" >/dev/null 2>&1 || true
echo "== $(date -u +%FT%TZ) DONE =="
exit 0
