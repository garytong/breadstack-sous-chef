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
#
# Arms, seeds and warm starts (r6). Settings come from the environment, else from runs/<id>/train-config.json
# ({"arms": ["base:3407", "base:2", "warm:3407"], "init_adapter": "…", "init_lr": 5e-5, "gen_check": 32}; the trainer
# may read runs/ and base/), else today's defaults — one arm, base:3407 (train.py's seed), no generation check — so a
# round that sets nothing trains exactly as before (train.py pipeline_config is the one resolver):
#   TRAIN_ARMS=base:3407,base:2,warm:3407
#                             train each arm in turn on the same dataset: base:<seed> from the base at the recipe's lr,
#                             warm:<seed> from INIT_ADAPTER at INIT_LR. Each is resumable on its own (checkpoints
#                             under train/checkpoints/<tag>/, each finished arm under train/seeds/<tag>/, run.json
#                             last; tag s<seed> for base, w<seed> for warm), scored on the SAME validation rows, and
#                             the best kept (train.py select) as train/adapter; run.json and progress.json carry
#                             every arm's scores and the arms not trained (selection.skipped). Exclusive with
#                             TRAIN_SEED(S). The 12 h window: an arm that would not finish before TRAIN_WINDOW_S
#                             (default 43200, the workflow's Vertex timeout) less TRAIN_TAIL_S (2700: export, stage,
#                             push) is not started — judged by the longest arm finished so far — once one arm is done.
#   TRAIN_SEED=2              one seed other than 3407 (train.py --seed)
#   TRAIN_SEEDS=3407,2,3      base:3407,base:2,base:3 (every seed warm when INIT_ADAPTER is set: r6's first form).
#                             Several arms score TRAIN_GEN_CHECK first turns (default 32 then); progress.json
#                             last_step counts across the arms.
#   INIT_ADAPTER=…            the warm arms' LoRA adapter: gs://…, runs/… or base/… in this bucket, or a run id
#                             (rt-…: that round's train/adapter/adapter); pulled to $W/init-adapter (train.py --init-adapter)
#   INIT_LR=5e-5              the warm arms' --lr (default 5e-5; only with INIT_ADAPTER; base arms keep 2e-4)
#   TRAIN_GEN_CHECK=N         train.py --gen-check N on each SFT (0 = off; the default with one arm)
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
# TRAIN_NONCE: the job's one-off `create` label (vertex_attempt). progress.json carries it, so the workflow reads only
# what THIS job wrote; a failure also leaves train/failed-<nonce>.json, which nothing overwrites (Vertex restarts a
# container that exited 1, even after it has marked the job FAILED: the restart used to rewrite `failed`)
NONCE=${TRAIN_NONCE:-}
SYNC_EVERY=${PIPELINE_SYNC_EVERY_S:-120}
FLAGS=(--data "$DATA/dataset" --fields-context on --eval-steps "${EVAL_STEPS:-150}" --val-max-rows "${VAL_MAX_ROWS:-150}")
mkdir -p "$W" "$HF_HOME"
CUR_STEP=start
SYNC_PID=
# the training settings (the "settings" step resolves them; these are today's defaults)
SEEDS=(3407)
ARMS=(base:3407)
SEED_FLAG=0
MULTI=0
INIT_ADAPTER_URL=
WARM_LR=  # (not INIT_LR: assigning an inherited variable would change what the resolver reads)
GEN_CHECK=0
SEED_ARGS=()
GEN_ARGS=()
WARM_ARGS=()  # --init-adapter DIR --lr LR: a warm arm's (the "pull init adapter" step sets them)
SKIPPED=()    # ARM=REASON for each planned arm not trained (the A100 trim, the 12 h window): select records them
WINDOW_S=${TRAIN_WINDOW_S:-43200}
TAIL_S=${TRAIN_TAIL_S:-2700}
# where push_checkpoints sends $ADAPTER_DIR/checkpoints (a TRAIN_SEEDS seed: $R/checkpoints/s<seed>), its markers'
# folder, the seed being trained and the updates of the seeds finished before it (progress.json last_step)
CKPT_DEST=$R/checkpoints
PUSH_TAG=main
CUR_SEED=
CUR_ARM=
SEED_OFFSET=0

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
  "$PY" - "$pj" "$CUR_STEP" "$1" "$ATTEMPT" "$MODE" "$ADAPTER_DIR" "${2:-}" "${TRAIN_RUNG:-}" "$NONCE" \
    "$CUR_SEED" "$SEED_OFFSET" "${SEEDS[*]}" "$MULTI" "${ARMS[*]}" "$CUR_ARM" <<'EOF' >/dev/null 2>&1 || true
import json, sys, time
from pathlib import Path
out, step, status, attempt, mode, adir, error, rung, nonce, seed, offset, seeds, multi, arms, arm = sys.argv[1:16]
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
p = {"status": status, "step": step, "error": error or None, "attempt": attempt, "rung": rung or None, "mode": mode,
     "nonce": nonce or None, "updated_at": now, "at": now}
if multi == "1":
    p["seeds_planned"] = [int(s) for s in seeds.split()]
    p["arms_planned"] = arms.split()
    p["seed"] = int(seed) if seed else None
    p["arm"] = arm or None
ad = Path(adir)
run = ad / "run.json"
if run.is_file():
    try:
        r = json.loads(run.read_text())
        for k in ("updates", "tokens_per_s", "peak_mem_gb", "wall_s", "train_loss"):
            if k in r:
                p[k] = r[k]
        p["updates"] = p.get("updates") or (r.get("plan") or {}).get("updates")
        if isinstance(r.get("seeds"), list):  # the selected arm's run: every arm's scores
            p["seeds"] = [{k: s.get(k) for k in ("arm", "seed", "start", "lr", "eval_loss", "gen_match", "gen_n",
                                                 "selected")}
                          for s in r["seeds"] if isinstance(s, dict)]
            p["selection"] = r.get("selection")
            p["seed"] = (r.get("selection") or {}).get("selected_seed")
            p["arm"] = (r.get("selection") or {}).get("selected_arm")
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
                # with TRAIN_SEEDS, counted across the seeds: the workflow reads a rise as progress
                p["last_step"] = d["step"] + int(offset or 0)
if evals:
    p["eval_loss_first"], p["eval_loss_last"] = evals[0], evals[-1]
Path(out).write_text(json.dumps(p, indent=1))
EOF
  if [ ! -s "$pj" ]; then
    # no python (or it failed): the status the workflow reads must still be there
    printf '{"status":"%s","step":"%s","error":"%s","attempt":"%s","nonce":"%s","updated_at":"%s"}\n' \
      "$1" "$(printf '%s' "$CUR_STEP" | tr -d '"\\')" "$(printf '%s' "${2:-}" | tr -d '"\\')" "$ATTEMPT" "$NONCE" \
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
  # sticky: a restart of this job never rewrites it (entrypoint.sh exits at once on it; the workflow reads it first)
  [ -n "$NONCE" ] && gs cp "$W/progress.json" "$R/failed-$NONCE.json" --content-type=application/json >/dev/null 2>&1
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
push_checkpoints() {  # complete checkpoints only, trainer_state.json last (train.py resumes only from a checkpoint
  # that has it): the Trainer writes it last (transformers _save_checkpoint), so a checkpoint still being written
  # waits for the next pass, and one cut off mid-upload (a Spot VM switched off) is never resumed from
  local c n
  [ -d "$ADAPTER_DIR/checkpoints" ] || return 0
  mkdir -p "$W/pushed/$PUSH_TAG"
  for c in "$ADAPTER_DIR"/checkpoints/checkpoint-*; do
    n=${c##*/}
    [ -f "$c/trainer_state.json" ] && [ ! -e "$W/pushed/$PUSH_TAG/$n" ] || continue
    gs rsync --recursive --exclude='^trainer_state\.json$' "$c" "$CKPT_DEST/$n" >/dev/null 2>&1 || continue
    gs cp "$c/trainer_state.json" "$CKPT_DEST/$n/trainer_state.json" >/dev/null 2>&1 && touch "$W/pushed/$PUSH_TAG/$n"
  done
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

step "kit"; echo "trainer image: ${TRAIN_IMAGE:-unknown} (nonce ${NONCE:-none})" | tee -a "$LOG"; run "$PY" --version
step "settings"
CFG_FILE=()
if gs_exists "$B/runs/$RUN/train-config.json"; then
  run gs cp "$B/runs/$RUN/train-config.json" "$W/train-config.json" || fail "pull train-config.json"
  CFG_FILE=("$W/train-config.json")
fi
"$PY" -c 'import sys; from souschef_lab.train import pipeline_config_main; sys.exit(pipeline_config_main(sys.argv[1:]))' \
  ${CFG_FILE[@]+"${CFG_FILE[@]}"} "$B" > "$W/settings.env" 2> "$W/settings.err"
rc=$?
tee -a "$LOG" < "$W/settings.err"
[ "$rc" = 0 ] || fail "$(tail -n 1 "$W/settings.err" | tr -d '"\\')"
while IFS='=' read -r k v; do
  case "$k" in
    SEEDS) read -r -a SEEDS <<< "$v" ;;
    ARMS) read -r -a ARMS <<< "$v" ;;
    SEED_FLAG) SEED_FLAG=$v ;;
    MULTI) MULTI=$v ;;
    INIT_ADAPTER) INIT_ADAPTER_URL=$v ;;
    INIT_LR) WARM_LR=$v ;;
    GEN_CHECK) GEN_CHECK=$v ;;
  esac
done < "$W/settings.env"
arm_kind() { printf '%s' "${1%%:*}"; }
arm_seed() { printf '%s' "${1##*:}"; }
arm_tag() { if [ "$(arm_kind "$1")" = warm ]; then printf 'w%s' "$(arm_seed "$1")"; else printf 's%s' "$(arm_seed "$1")"; fi; }
# An A100 rung trains about 2x slower than an H100: three arms (~7-8.5 h on an H100) would not fit its 12 h
# attempt. There, a several-arm round keeps the arms an earlier attempt already finished (train/seeds/<tag>/) and
# trains no new one; with none finished, only the first arm. Selection and the rest of the flow are unchanged; the
# arms dropped are recorded (selection.skipped).
case "${TRAIN_RUNG:-}" in
  a100*)
    if [ "$MULTI" = 1 ] && [ "$MODE" = full ]; then
      KEEP=()
      for a in "${ARMS[@]}"; do gs_exists "$R/seeds/$(arm_tag "$a")/run.json" && KEEP+=("$a"); done
      [ "${#KEEP[@]}" -gt 0 ] || KEEP=("${ARMS[0]}")
      if [ "${#KEEP[@]}" -lt "${#ARMS[@]}" ]; then
        echo "rung ${TRAIN_RUNG}: arms ${ARMS[*]} -> ${KEEP[*]} (an A100 attempt has no room for more)" | tee -a "$LOG"
        for a in "${ARMS[@]}"; do
          case " ${KEEP[*]} " in *" $a "*) ;; *) SKIPPED+=("$a=A100 rung ${TRAIN_RUNG}: no room in its 12 h attempt") ;; esac
        done
        ARMS=("${KEEP[@]}")
        SEEDS=()
        for a in "${ARMS[@]}"; do SEEDS+=("$(arm_seed "$a")"); done
      fi
    fi
    ;;
esac
ANY_WARM=0
for a in "${ARMS[@]}"; do [ "$(arm_kind "$a")" = warm ] && ANY_WARM=1; done
[ "$SEED_FLAG" = 1 ] && [ "$MULTI" = 0 ] && SEED_ARGS=(--seed "${SEEDS[0]}")
if [ "$GEN_CHECK" -gt 0 ] 2>/dev/null; then
  if [ "$MODE" = smoke ]; then GEN_ARGS=(--gen-check 2); else GEN_ARGS=(--gen-check "$GEN_CHECK"); fi
fi
step "pull dataset"
mkdir -p "$DATA"
run gs rsync --recursive "$B/runs/$RUN/dataset" "$DATA" || fail "pull dataset"
step "pull base"
if [ ! -f "$HF_HOME/.base-ok" ]; then
  run gs cp "$B/base/$BASE_TAR" "$B/base/$BASE_TAR.sha256" "$W/" || fail "pull base"
  run bash -c 'cd "$1" && sha256sum -c "$2"' _ "$W" "$BASE_TAR.sha256" || fail "base tar sha256"
  # The tar holds a Hugging Face hub cache folder (models--unsloth--Qwen3.5-9B/…),
  # which belongs under $HF_HOME/hub: unpacked into $HF_HOME itself, the offline
  # loader finds nothing (2026-10-14: the first SMOKE failed at model load with
  # LocalEntryNotFoundError). A tar that already starts with hub/ goes to $HF_HOME.
  first=$(tar -tf "$W/$BASE_TAR" 2>/dev/null | head -n 1 || true)
  case "$first" in hub/*) dest="$HF_HOME" ;; *) dest="$HF_HOME/hub" ;; esac
  mkdir -p "$dest"
  run tar -x -C "$dest" -f "$W/$BASE_TAR" || fail "untar base"
  rm -f "$W/$BASE_TAR"
  [ -f "$HF_HOME/hub/models--unsloth--Qwen3.5-9B/refs/main" ] || fail "base cache layout: no $HF_HOME/hub/models--unsloth--Qwen3.5-9B/refs/main after unpacking"
  touch "$HF_HOME/.base-ok"
fi
# the export merges into this snapshot folder (export.py local_base_adapter: offline, Unsloth's merge cannot resolve
# the Hub id): proved here, before the SFT, not after it
run "$PY" -c 'import sys; from souschef_lab.export import cached_snapshot; s = cached_snapshot("unsloth/Qwen3.5-9B"); print("base snapshot:", s); sys.exit(0 if s else 1)' \
  || fail "base snapshot: no unsloth/Qwen3.5-9B snapshot with config.json and safetensors under $HF_HOME/hub"
if [ -n "$INIT_ADAPTER_URL" ] && [ "$ANY_WARM" = 1 ]; then
  step "pull init adapter"
  rm -rf "$W/init-adapter"
  mkdir -p "$W/init-adapter"
  run gs rsync --recursive "$INIT_ADAPTER_URL" "$W/init-adapter" || fail "pull init adapter $INIT_ADAPTER_URL"
  INIT_DIR=$W/init-adapter
  # a run's train/adapter folder holds the adapter one level down (adapter/, beside its run.json)
  if [ ! -f "$INIT_DIR/adapter_config.json" ] && [ -f "$INIT_DIR/adapter/adapter_config.json" ]; then INIT_DIR=$INIT_DIR/adapter; fi
  [ -f "$INIT_DIR/adapter_config.json" ] && [ -f "$INIT_DIR/adapter_model.safetensors" ] \
    || fail "init adapter: no adapter_config.json and adapter_model.safetensors under $INIT_ADAPTER_URL"
  WARM_ARGS=(--init-adapter "$INIT_DIR" --lr "$WARM_LR")
fi
# a one-arm round's arm: its seed flag and, for a warm arm, the adapter (as r6's single warm start did)
ONE_ARGS=()
[ "$MULTI" = 0 ] && [ "$(arm_kind "${ARMS[0]}")" = warm ] && ONE_ARGS=(${WARM_ARGS[@]+"${WARM_ARGS[@]}"})
step "pull checkpoints"
RESUMED=0
if [ "$MULTI" = 1 ]; then
  # TRAIN_SEEDS: each seed pulls its own (train/checkpoints/s<seed>/) as its turn comes
  if gs_exists "$R/checkpoints" || gs_exists "$R/seeds"; then RESUMED=1; fi
elif gs_exists "$R/checkpoints"; then
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
step "dry plan"; run "$PY" -m souschef_lab.train "${FLAGS[@]}" ${ONE_ARGS[@]+"${ONE_ARGS[@]}"} \
  ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} ${GEN_ARGS[@]+"${GEN_ARGS[@]}"} --name "$NAME-sft" --checkpoint-steps "$CKPT" \
  --dry-run || fail "dry plan"
if [ "$MULTI" = 1 ] && [ "$ANY_WARM" = 1 ]; then
  # the warm arms' plan too: the adapter's LoRA shape and base are checked before any arm trains
  step "dry plan (warm arms)"; run "$PY" -m souschef_lab.train "${FLAGS[@]}" "${WARM_ARGS[@]}" \
    ${GEN_ARGS[@]+"${GEN_ARGS[@]}"} --name "$NAME-sft" --checkpoint-steps "$CKPT" --dry-run || fail "dry plan (warm arms)"
fi

seed_steps() {  # seed_steps DIR — the updates a finished arm's run.json records (0 when there is none)
  "$PY" -c 'import json, sys
try:
    r = json.load(open(sys.argv[1] + "/run.json"))
except (OSError, ValueError):
    r = {}
print(int(r.get("steps") or r.get("updates") or 0))' "$1" 2>/dev/null || echo 0
}
arm_wall_s() {  # arm_wall_s DIR — about how long a whole arm takes, from its run.json (train.py's minutes, scaled up
  # when that process resumed part-way: resumed_from_step of steps); 0 when unknown
  "$PY" -c 'import json, sys
try:
    r = json.load(open(sys.argv[1] + "/run.json"))
except (OSError, ValueError):
    r = {}
m = r.get("minutes")
w = float(m) * 60 if isinstance(m, (int, float)) else float(r.get("train_seconds") or 0)
steps, done = int(r.get("steps") or 0), int(r.get("resumed_from_step") or 0)
if 0 < done < steps:
    w *= steps / (steps - done)
print(int(w))' "$1" 2>/dev/null || echo 0
}
train_arm() {  # train_arm ARM — one arm (base:<seed> | warm:<seed>) of a several-arm round, resumable on its own;
  # pushed to train/seeds/<tag>/ (s<seed> from the base, w<seed> from the warm-start adapter)
  local a=$1 s tag sname
  s=$(arm_seed "$a"); tag=$(arm_tag "$a"); sname="$NAME-sft-$tag"
  local -a xargs=()
  [ "$(arm_kind "$a")" = warm ] && xargs=("${WARM_ARGS[@]}")
  ADAPTER_DIR=runs/adapters/$sname
  CKPT_DEST=$R/checkpoints/$tag
  PUSH_TAG=$tag
  CUR_SEED=$s
  CUR_ARM=$a
  if [ ! -f "$ADAPTER_DIR/adapter/adapter_config.json" ] || [ ! -f "$ADAPTER_DIR/run.json" ]; then
    step "pull checkpoints (arm $a)"
    if gs_exists "$CKPT_DEST"; then
      mkdir -p "$ADAPTER_DIR/checkpoints"
      run gs rsync --recursive "$CKPT_DEST" "$ADAPTER_DIR/checkpoints" || fail "pull checkpoints (arm $a)"
    fi
    start_sync
    if [ "$MODE" = smoke ]; then
      step "SFT arm $a (smoke: 4 updates, checkpoint every 2)"
      run "$PY" -m souschef_lab.train "${FLAGS[@]}" ${xargs[@]+"${xargs[@]}"} --seed "$s" ${GEN_ARGS[@]+"${GEN_ARGS[@]}"} \
        --name "$sname" --max-steps 4 --checkpoint-steps 2 --resume || fail "SFT arm $a"
      step "SFT arm $a (smoke: resume to 6)"
      rm -rf "$ADAPTER_DIR/adapter"
      run "$PY" -m souschef_lab.train "${FLAGS[@]}" ${xargs[@]+"${xargs[@]}"} --seed "$s" ${GEN_ARGS[@]+"${GEN_ARGS[@]}"} \
        --name "$sname" --max-steps 6 --checkpoint-steps 2 --resume || fail "SFT arm $a resume"
      grep -Eq "Resuming from .*$sname/checkpoints/checkpoint-" "$LOG" || fail "the smoke SFT of arm $a did not resume"
    else
      step "SFT arm $a ($sname; resumes from the newest checkpoint)"
      run "$PY" -m souschef_lab.train "${FLAGS[@]}" ${xargs[@]+"${xargs[@]}"} --seed "$s" ${GEN_ARGS[@]+"${GEN_ARGS[@]}"} \
        --name "$sname" --checkpoint-steps "$CKPT" --resume || fail "SFT arm $a"
    fi
    stop_sync
    push_checkpoints
  fi
  step "push arm $a"
  run gs rsync --recursive --exclude='^(checkpoints/.*|run\.json)$' "$ADAPTER_DIR" "$R/seeds/$tag" || fail "push arm $a"
  run gs cp "$ADAPTER_DIR/run.json" "$R/seeds/$tag/run.json" || fail "push arm $a run.json"
  rm -rf "$ADAPTER_DIR/checkpoints"  # pushed (train/checkpoints/<tag>/); the disk is for the next arm
}

if [ "$MODE" = full ] && [ "$RESUMED" = 0 ] && [ ! -f "$ADAPTER_DIR/adapter/adapter_config.json" ]; then
  # the warm adapter when any arm is warm: its load is proved here, in minutes, before hours of training
  SMOKE_ARGS=()
  [ "$ANY_WARM" = 1 ] && SMOKE_ARGS=("${WARM_ARGS[@]}")
  rm -rf "runs/adapters/$NAME-smoke-sft"
  step "resume smoke: 2 updates, checkpoint at 2"
  run "$PY" -m souschef_lab.train "${FLAGS[@]}" ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} --name "$NAME-smoke-sft" \
    --max-steps 2 --checkpoint-steps 2 || fail "smoke"
  step "resume smoke: resume to 4"
  run "$PY" -m souschef_lab.train "${FLAGS[@]}" ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} --name "$NAME-smoke-sft" \
    --max-steps 4 --checkpoint-steps 2 --resume || fail "smoke resume"
  grep -Eq "Resuming from .*$NAME-smoke-sft/checkpoints/checkpoint-2" "$LOG" || fail "the smoke did not resume from its checkpoint"
  rm -rf "runs/adapters/$NAME-smoke-sft"
fi

if [ "$MULTI" = 1 ] && [ ! -f "$ADAPTER_DIR/adapter/adapter_config.json" ]; then
  trap '[ -n "$SYNC_PID" ] && kill_tree "$SYNC_PID" 2>/dev/null' EXIT
  FINAL_DIR=$ADAPTER_DIR
  ARM_DIRS=()
  LONGEST=0  # the longest finished arm (s): the 12 h window guard's estimate of the next one
  FINISHED=0
  STOPPED=
  # finished in an earlier attempt: pulled first (free, and they tell how long an arm takes)
  for a in "${ARMS[@]}"; do
    t=$(arm_tag "$a"); d="runs/adapters/$NAME-sft-$t"
    if [ ! -f "$d/run.json" ] && gs_exists "$R/seeds/$t/run.json"; then  # run.json goes up last: the arm finished
      CUR_ARM=$a; CUR_SEED=$(arm_seed "$a")
      step "arm $a: trained in an earlier attempt (pull)"
      run gs rsync --recursive "$R/seeds/$t" "$d" || fail "pull arm $a"
    fi
    if [ -f "$d/run.json" ]; then
      FINISHED=$((FINISHED + 1))
      w=$(arm_wall_s "$d"); [ "$w" -gt "$LONGEST" ] 2>/dev/null && LONGEST=$w
    fi
  done
  for a in "${ARMS[@]}"; do
    t=$(arm_tag "$a"); d="runs/adapters/$NAME-sft-$t"
    if [ ! -f "$d/run.json" ]; then
      if [ -n "$STOPPED" ]; then
        SKIPPED+=("$a=$STOPPED"); continue
      fi
      # the 12 h window: once an arm is done, start another only if it would end in time for the export
      if [ "$FINISHED" -gt 0 ] && [ "$LONGEST" -gt 0 ] && [ $((SECONDS + LONGEST + TAIL_S)) -gt "$WINDOW_S" ]; then
        STOPPED="the 12 h window: ${SECONDS}s used, an arm takes about ${LONGEST}s, ${TAIL_S}s kept for the export (TRAIN_WINDOW_S=$WINDOW_S)"
        echo "arm $a not started: $STOPPED" | tee -a "$LOG"
        SKIPPED+=("$a=$STOPPED"); continue
      fi
      started=$SECONDS
      train_arm "$a"
      w=$(arm_wall_s "$d"); [ "$w" -gt 0 ] 2>/dev/null || w=$((SECONDS - started))
      [ "$w" -gt "$LONGEST" ] && LONGEST=$w
      FINISHED=$((FINISHED + 1))
    elif ! gs_exists "$R/seeds/$t/run.json"; then
      train_arm "$a"  # finished on this disk but never pushed (a restarted container): push it
    fi
    SEED_OFFSET=$((SEED_OFFSET + $(seed_steps "$d")))
    ARM_DIRS+=("$d")
  done
  ADAPTER_DIR=$FINAL_DIR
  CKPT_DEST=$R/checkpoints
  PUSH_TAG=main
  CUR_SEED=
  CUR_ARM=
  SKIP_ARGS=()
  for k in ${SKIPPED[@]+"${SKIPPED[@]}"}; do SKIP_ARGS+=(--skipped "$k"); done
  step "select arm (${ARMS[*]})"
  run "$PY" -c 'import sys; from souschef_lab.train import select_main; sys.exit(select_main(sys.argv[1:]))' \
    "${ARM_DIRS[@]}" --into "$ADAPTER_DIR" ${SKIP_ARGS[@]+"${SKIP_ARGS[@]}"} || fail "select arm"
  # progress.json now reads the kept arm's loss.jsonl: last_step stays the total over the arms (never falls back)
  SEED_OFFSET=$((SEED_OFFSET - $(seed_steps "$ADAPTER_DIR")))
elif [ ! -f "$ADAPTER_DIR/adapter/adapter_config.json" ]; then
  start_sync
  trap '[ -n "$SYNC_PID" ] && kill_tree "$SYNC_PID" 2>/dev/null' EXIT
  if [ "$MODE" = smoke ]; then
    step "SFT (smoke: 4 updates, checkpoint every 2)"
    run "$PY" -m souschef_lab.train "${FLAGS[@]}" ${ONE_ARGS[@]+"${ONE_ARGS[@]}"} ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} \
      ${GEN_ARGS[@]+"${GEN_ARGS[@]}"} --name "$NAME-sft" --max-steps 4 --checkpoint-steps 2 --resume || fail "SFT"
    step "SFT (smoke: resume to 6)"
    rm -rf "$ADAPTER_DIR/adapter"
    run "$PY" -m souschef_lab.train "${FLAGS[@]}" ${ONE_ARGS[@]+"${ONE_ARGS[@]}"} ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} \
      ${GEN_ARGS[@]+"${GEN_ARGS[@]}"} --name "$NAME-sft" --max-steps 6 --checkpoint-steps 2 --resume || fail "SFT resume"
    grep -Eq "Resuming from .*$NAME-sft/checkpoints/checkpoint-" "$LOG" || fail "the smoke SFT did not resume"
  else
    step "SFT ($NAME-sft; resumes from the newest checkpoint)"
    run "$PY" -m souschef_lab.train "${FLAGS[@]}" ${ONE_ARGS[@]+"${ONE_ARGS[@]}"} ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} \
      ${GEN_ARGS[@]+"${GEN_ARGS[@]}"} --name "$NAME-sft" --checkpoint-steps "$CKPT" --resume || fail "SFT"
  fi
  stop_sync
  push_checkpoints
fi

step "export q4_k_m"
run "$PY" -m souschef_lab.export --adapter "$ADAPTER_DIR/adapter" --name "$NAME" --quants q4_k_m \
  --base-modelfile "$BASE_MODELFILE" || fail "export"
# Unsloth's intermediates: gguf/ (the merged 16-bit model, ~18 GB, kept when the disk is not tight) and gguf_gguf/
# (the mmproj and Unsloth's own Modelfile). Only <name>-q4_k_m.gguf, Modelfile.q4_k_m and export.json go up (r4's push
# excluded gguf/ too)
rm -rf "runs/exports/$NAME/gguf" "runs/exports/$NAME/gguf_gguf"
step "stage"
run make --no-print-directory model-stage NAME="$NAME" MODEL_STAGE="runs/model-image/$NAME" \
  MODEL_TEMPERATURE=0.3 MODEL_PRESENCE_PENALTY=0 || fail "model-stage"
step "push"
run gs rsync --recursive --exclude='^checkpoints/.*' "$ADAPTER_DIR" "$R/adapter" || fail "push adapter"
run gs rsync --recursive --exclude='^gguf(_gguf)?/.*' "runs/exports/$NAME" "$R/export" || fail "push export"
run gs rsync --recursive "runs/model-image/$NAME" "$R/stage" || fail "push stage"
CUR_STEP=DONE
progress done
date -u +%FT%TZ > "$W/DONE"
gs cp "$W/DONE" "$R/DONE" || fail "DONE"
gs cp "$LOG" "$R/train.log" >/dev/null 2>&1 || true
echo "== $(date -u +%FT%TZ) DONE =="
exit 0
