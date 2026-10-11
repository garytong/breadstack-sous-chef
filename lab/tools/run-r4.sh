#!/usr/bin/env bash
# run-r4.sh — the r4 SFT chain on the training VM (TRAIN.md), as souschef in the lab:
#
#   sudo -iu souschef bash -lc 'cd /work/sous-chef/lab && setsid nohup bash tools/run-r4.sh >/dev/null 2>&1 &'
#   tail -f /work/runs-r4.log            # progress;  sous-chef-train-status
#
# Steps (each logged under a "== <UTC> <step> ==" line, like r3's chain): kit version → data pull → row checks →
# dry plan → smoke (2 updates with a checkpoint, then RESUMED to 4 — proves the preemption path before the long
# run) → the full SFT (resumable: a rerun after a Spot preemption continues from the newest checkpoint) → GGUF
# q4_k_m export with the base Modelfile → push the adapter and the export to the bucket → the log → DONE.
# Rerunning the script is safe: the smoke is skipped once the full run has an adapter, the full run resumes.
#
# The recipe is r3's (train.py RECIPE: unsloth/Qwen3.5-9B bf16, LoRA r16/α16, lr 2e-4, 1 epoch, batch 1 × grad_accum 4,
# adamw_8bit, linear, warmup 3 %, wd 0.01, max_seq 32,768, serving-parity render, thinking off). What differs is the
# data and three run settings (TRAIN.md "What changed and why"): --checkpoint-steps 50 + --resume (a 2–3 h run on
# Spot), --eval-steps 150 and --val-max-rows 150 (r3's 50-step cadence on a ~300-row val set would spend ~40 min).
set -uo pipefail
[ -f /etc/sous-chef-train.env ] && { set -a; . /etc/sous-chef-train.env; set +a; }
cd "${SOUS_CHEF_LAB:-/work/sous-chef/lab}" || exit 1
NAME=${NAME:-r4}
DATA=${DATA:-r4}
LOG=${LOG:-/work/runs-$NAME.log}
PY=.venv-train/bin/python
CKPT=${CKPT:-50}
FLAGS=(--data "data/$DATA/dataset" --fields-context on --eval-steps "${EVAL_STEPS:-150}" --val-max-rows "${VAL_MAX_ROWS:-150}")
BASE_MODELFILE=base-modelfiles/qwen3.5-9b.modelfile

step() { echo "== $(date -u +%FT%TZ) $* ==" | tee -a "$LOG"; }
run() { "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
fail() { step "FAILED: $*"; [ -n "${SOUS_CHEF_TRAIN_BUCKET:-}" ] && gcloud storage cp "$LOG" "$SOUS_CHEF_TRAIN_BUCKET/runs/logs/$NAME.log" >/dev/null 2>&1; exit 1; }

step "kit version"; run cat KIT-VERSION
step "data pull $DATA"; run sous-chef-data-pull "$DATA" || fail "data pull"
step "rows"
run "$PY" - "data/$DATA/dataset" <<'EOF' || fail "row checks"
import json, sys
from pathlib import Path
from souschef_lab.r4score import mentions_preview   # the ONE preview rule the scorer and the build use
d = Path(sys.argv[1])
st = json.loads((d / "stats.json").read_text())
for split in ("train", "val"):
    rows = [json.loads(l) for l in (d / f"{split}.jsonl").read_text().splitlines() if l.strip()]
    block = sum(1 for r in rows if (r.get("meta") or {}).get("fields_context"))
    follow = sum(1 for r in rows if (r.get("meta") or {}).get("history_assistant_turns"))
    drift = sum(1 for r in rows if (r.get("meta") or {}).get("block_drift"))
    preview = sum(1 for r in rows for m in r["messages"] if m["role"] == "assistant" and mentions_preview(m.get("content")))
    print(f"{split}: {len(rows)} rows, {block} with a candidate-fields block, {follow} follow-ups, "
          f"{preview} replies mentioning the preview, {drift} with block drift")
    assert rows and preview == 0 and drift == 0, "no rows, a reply that mentions the preview, or a drifted block"
assert st.get("block_drift_rows") == 0 and st.get("preview_claim_kept") == 0, "stats.json: drift or preview rows kept"
print(json.dumps(st.get("by_family")))
EOF
step "dry plan"; run "$PY" -m souschef_lab.train "${FLAGS[@]}" --name "$NAME-sft" --checkpoint-steps "$CKPT" --dry-run || fail "dry plan"
if [ -z "${SKIP_SMOKE:-}" ] && [ ! -f "runs/adapters/$NAME-sft/adapter/adapter_config.json" ] && [ ! -d "runs/adapters/$NAME-sft/checkpoints" ]; then
  rm -rf "runs/adapters/$NAME-smoke-sft"
  step "smoke: 2 updates, checkpoint at 2"
  run "$PY" -m souschef_lab.train "${FLAGS[@]}" --name "$NAME-smoke-sft" --max-steps 2 --checkpoint-steps 2 || fail "smoke"
  step "smoke: resume to 4"
  run "$PY" -m souschef_lab.train "${FLAGS[@]}" --name "$NAME-smoke-sft" --max-steps 4 --checkpoint-steps 2 --resume || fail "smoke resume"
  grep -Eq "Resuming from .*$NAME-smoke-sft/checkpoints/checkpoint-2" "$LOG" || fail "the smoke did not resume from its checkpoint"
  rm -rf "runs/adapters/$NAME-smoke-sft"
fi
if [ ! -f "runs/adapters/$NAME-sft/adapter/adapter_config.json" ]; then
  step "full SFT ($NAME-sft; resumes from the newest checkpoint after a preemption)"
  run "$PY" -m souschef_lab.train "${FLAGS[@]}" --name "$NAME-sft" --checkpoint-steps "$CKPT" --resume || fail "full SFT"
fi
step "export q4_k_m"
run "$PY" -m souschef_lab.export --adapter "runs/adapters/$NAME-sft/adapter" --name "$NAME" --quants q4_k_m \
  --base-modelfile "$BASE_MODELFILE" || fail "export"
rm -rf "runs/adapters/$NAME-sft/checkpoints"   # trainer state only; the adapter is saved
step "push adapter"; run sous-chef-runs-push "$NAME-sft" || fail "push adapter"
step "push export"; run sous-chef-runs-push "$NAME" || fail "push export"
step "DONE"
[ -n "${SOUS_CHEF_TRAIN_BUCKET:-}" ] && gcloud storage cp "$LOG" "$SOUS_CHEF_TRAIN_BUCKET/runs/logs/$NAME.log" >/dev/null 2>&1
exit 0
