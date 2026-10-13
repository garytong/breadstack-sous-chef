# Retraining blueberry-muffin: the automated round and the playbook

A retrain round turns a big drift ([`docs/DRIFT.md`](DRIFT.md)) into a
candidate model, measures it against the live one, and waits for a person to
Approve or Skip on the thermometer. Every stage runs in GCP by itself
(workflow `souschef-retrain`), and every stage can also be run by hand. This
file is both: for each stage, **Automated** says what the workflow does and
**By hand** is the playbook, for debugging or when the automation is off.

Top-level runbook and switches: [`docs/PIPELINE.md`](PIPELINE.md).
The r4 recipe this generalises: [`docs/LAB.md`](LAB.md) §5b (cases, collector,
scoring, dataset) and §6b (the exam); training on a VM:
[`docs/TRAIN-REMOTE.md`](TRAIN-REMOTE.md).

```bash
export PROJECT=breadstack-sous-chef-int
export REGION=northamerica-northeast2     # the int service, the lab image, the pipeline jobs
export WF_REGION=us-central1              # workflows, Vertex, the pipeline bucket
export GPU_REGION=us-east4                # sous-chef-model, -eval, -cand
export PIPE=gs://breadstack-sous-chef-int-pipeline
export TRAIN=gs://breadstack-sous-chef-int-training
RUN='rt-YYYYMMDD-HHMM-xxxxxxxx'           # the round's id (Slack and the thermometer show it)
W=$PIPE/runs/$RUN
```

**Shell: bash.** Every loop here reads exit codes; zsh has no `PIPESTATUS`.

## The round at a glance

| # | stage | runs on | about | output |
|---|---|---|---|---|
| — | init: lock, resolve images, pin the deployed manifest (`run.json → deployed`; every stage reads it), `retrain_started` | workflow | 1 min | `runs/<id>/run.json` |
| R1 | prepare: rebase every family, author new-field cases | job `souschef-prepare` | 30–60 min | `prepare/{cases.jsonl,report.json,leak-report.json}` |
| R2 | teacher collection, 3 passes | job `souschef-collect` | 1.8–2.2 h | `collect/*` |
| R3 | dataset + parity gate | job `souschef-dataset` | 15 min | `dataset/*`, `model/context.tar.gz` |
| R4 | SFT + export + stage | Vertex custom job | 2.4–2.8 h + any capacity wait | `train/{adapter,export,stage,progress.json,DONE}` |
| R5 | model image | Cloud Build | 15–25 min | `model/image.json`, `model:c-<id>` |
| R6 | candidate deploy, paired eval, report | `sous-chef-model-cand`, job `souschef-eval` | 2–3 h | `eval/*`, `report.json`, `decision.json` |
| R7 | decision wait, then roll out or park | workflow + thermometer | up to 14 days | `rollout.json` or `state/parked/…` |
| — | cleanup: drop checkpoints, unlock | workflow | 1 min | |

Total about 9–12 h from the drift to the decision message. Any failure posts
`stage_failed` with the mention, releases the lock and keeps every artifact.

Workflow input: `{mode: "full" | "smoke", trigger: {drift_run, reason},
dry_rollout: bool, resume_from?: "<stage>", run_id?: "<id>"}`.

---

## R0. First-time setup

The ordered steps (quotas, thermometer, CI, `pipeline_enabled`, secrets,
seeding, smoke, calibration, the schedule) are in
[PIPELINE.md §4](PIPELINE.md#4-turning-it-on-first-time-in-order). This
section holds the two that are retrain-specific.

### R0. Seeding the bucket

Nothing here spends money. A person (or the main session) runs it once after
`pipeline_enabled` is applied. `R4` is the r4 seed folder, copied out of the
session scratch on 2026-10-11 into the lab's git-ignored `lab/data/r4-seed/`
(the case pool, the F0X exam, the revisions and `SHA256SUMS`; `sha256sum -c`
there before seeding), and `WHY` the paraphrase study folder inside it
(merchants' own turns from int: keep it out of git and out of chat).

```bash
LAB=~/projects/breadstack-sous-chef/lab
R4=$LAB/data/r4-seed
WHY=$R4/why
(cd $R4 && sha256sum -c SHA256SUMS)

# 1. The base weights: a bucket-to-bucket copy (same location, nothing through the workstation),
#    then their sha256 (pipeline-train.sh runs sha256sum -c). The hash streams 18 GiB once.
gcloud storage cp $TRAIN/base/unsloth-qwen3.5-9b-hf.tar $PIPE/base/
gcloud storage cat $PIPE/base/unsloth-qwen3.5-9b-hf.tar | sha256sum | sed 's| -$| unsloth-qwen3.5-9b-hf.tar|' \
  | gcloud storage cp - $PIPE/base/unsloth-qwen3.5-9b-hf.tar.sha256

# 2. The r4 case pool: 3,577 cases on bf788701 (3,344 trainable + the 233 F0H holdout), the F0X exam, the revisions
gcloud storage cp $R4/cases-all.jsonl $R4/cases-F0X-exam.jsonl $R4/r4-revisions.json $PIPE/cases/pool/r4/

# 3. The study folder for the hold-out filter (leak.py rules 1–3 read the exam and the real conversations).
#    It holds merchants' own turns from int: the same handling as the training data (US-CENTRAL1).
gcloud storage rsync -r $WHY $PIPE/cases/why/

# 4. r4's dataset, for reference and rebuilds (no pipeline code reads it: stale_training_rows reads cases-all.jsonl;
#    the training bucket's data/ copy expires after 30 days). $LAB/data/r4 holds dataset/ and manifest.json, so this
#    gives cases/pool/r4/dataset/dataset/train.jsonl: the same layout as a round's runs/<id>/dataset/ and an approved
#    model's cases/trained/<candidate>/dataset/
gcloud storage rsync -r $LAB/data/r4 $PIPE/cases/pool/r4/dataset/

# 5. r4's training manifest, by its full hash
gcloud storage cp $LAB/fixtures/exam/manifest-real.json \
  $PIPE/manifests/bf78870182302f821856057d0091d62b09403c5180716c4f37a4830761e18a12.json
```

6. **The pool index** (`cases/pool/index.json`), counting families from the
   case file:

```bash
python3 - $R4/cases-all.jsonl <<'EOF' | gcloud storage cp - $PIPE/cases/pool/index.json
import collections, json, sys
fam = collections.Counter(json.loads(l)["meta"]["family"] for l in open(sys.argv[1]) if l.strip())
print(json.dumps({"files": [{"path": "cases/pool/r4/cases-all.jsonl",
  "manifest": "bf78870182302f821856057d0091d62b09403c5180716c4f37a4830761e18a12",
  "family_counts": dict(sorted(fam.items())), "round": "r4", "trained_in": "r4"}]}, indent=1))
EOF
```

7. **The live model** (`state/model.json`). The digest is what baselines key
   on; read it from the registry:

```bash
DIGEST=$(gcloud artifacts docker images describe $GPU_REGION-docker.pkg.dev/$PROJECT/sous-chef-model/model:r4 \
  --format='value(image_summary.digest)')
TRAINED=2026-10-11T03:18:31Z   # r4's run.json creation time, recorded 2026-10-11 (the bucket copy expires after 30 days)
cat <<EOF | gcloud storage cp - $PIPE/state/model.json
{"tag": "r4", "digest": "$DIGEST",
 "training_manifest": "bf78870182302f821856057d0091d62b09403c5180716c4f37a4830761e18a12",
 "trained_at": "$TRAINED", "run_id": "r4", "dataset": "$PIPE/cases/pool/r4/dataset", "history": []}
EOF
gcloud run services describe sous-chef-model --project $PROJECT --region $GPU_REGION \
  --format='value(spec.template.spec.containers[0].image)'      # must be …/model:r4 (or @$DIGEST)
```

8. **Engine snapshots**: r4's (cookie-cutter `e7e90106`, the snapshot the r4
   ground truth was measured on) and the int launcher's own commit, read
   from the deployed manifest ([DRIFT.md §6](DRIFT.md#6-the-engine-snapshot)):

```bash
CC=~/projects/breadstack-cookie-cutter
export PIPELINE_BUCKET=$PIPE                    # where --upload publishes
bash $LAB/tools/engine-vendor.sh $CC e7e90106 --upload
URL=$(gcloud run services describe sous-chef --project $PROJECT --region $REGION --format='value(status.url)')
# the key goes to curl on stdin (-H @-), never on a command line `ps` shows
SHA=$(gcloud secrets versions access latest --secret sous-chef-pipeline-service-key --project $PROJECT \
  | sed 's/^/Authorization: Bearer /' | curl -s -H @- "$URL/v1/manifests/deployed?line=int" | jq -r .generated_from.sha)
git -C $CC fetch && bash $LAB/tools/engine-vendor.sh $CC $SHA --upload
```

Check: `gcloud storage ls -r $PIPE/ | head -40` shows `base/`, `cases/`,
`engine/`, `manifests/`, `state/`.

### SMOKE

`make pipeline-smoke` (= `gcloud workflows execute souschef-retrain --project $PROJECT --location $WF_REGION --data='{"mode":"smoke"}'`; `execute` returns at once, `run` would block until the decision)

The same stages with tiny inputs (2–6 h with the GPU waits, about US$2 of API,
1 GPU-hour, 30 L4-minutes). It proves every edge before a full round spends US$150:

| stage | in SMOKE |
|---|---|
| prepare | 10 pool cases rebased (≥ 1 per family) + 1 target × 4 authored cases (US$ < 1) |
| collect | those 14, pass 1 only |
| dataset | the parity gate in full; `--val-pct 20` |
| train | first the trainer image's `selftest` (CUDA visible, bf16, `import unsloth`); then the ladder with `--max-steps 4 --checkpoint-steps 2`, a resume to 6, and a full export (proves llama.cpp) |
| build / candidate | real, tag `c-smoke-<id>` |
| eval | 10 rows per set, 1 replicate, both arms |
| decision | the real page and API, badge **SMOKE**, expires after 1 day |
| approve | a **dry rollout**: `testIamPermissions(run.services.update)` on `sous-chef-model` and a `services.get`; `rolled_out` with `dry_run: true` |

Two things are proven here for the first time: that Vertex's GPU driver runs
the cu130 wheels (the trainer image carries `cuda-compat`; the fallback is a
cu129 torch pin for the trainer, render parity unaffected), and whether
Vertex accepts `a3-highgpu-1g` with Flex-start (if not, the ladder starts at
Spot, R4).

Locally, with no network at all: `make -C lab retrain-dry` runs prepare →
collect → dataset → eval on the fake service with the oracle and a fake
author.

**The whole chain, dry** (before a change to the workflows, the jobs or the
thermometer's pipeline routes): `infra/modules/pipeline/tests/e2e_dry.py`
executes the deployed workflow definitions in `tests/wfsim.py` (a Cloud
Workflows interpreter) against fake Google APIs, with the real lab jobs, this
tree's service binary and the thermometer's real handlers, MySQL repo and
notifier (its `TestSousChefDevServe`, posting to a local webhook):

```bash
W=/tmp/e2e
python3 infra/modules/pipeline/tests/e2e_dry.py --work $W setup --bundle <engine-vendor bundle dir> --snapshot <cc snapshot>
for p in phase1 phase2 phase3 weekly phase4 phase5; do python3 infra/modules/pipeline/tests/e2e_dry.py --work $W $p || break; done
```

phase1: first poll, heartbeat, the debounced check calibrating r4 first.
phase2: two new launcher fields → R3 notice → check: retrain → a full round
(Flex-start stockout → Spot, the image build, the paired eval) → Skip: ack,
parked, authored cases kept. phase3: more fields → drift beyond the ack → a
round that batches round A's cases → Approve with the bar override:
production patched, baseline reset, ack gone. weekly: the Monday run on the
new model. phase4: SMOKE + Approve (dry rollout) and a collect that fails
three times (stage_failed with the mention, lock released). Each phase is a
few minutes; it needs Go, node, Postgres server binaries and the
thermometer's throwaway MySQL (its README, 127.0.0.1:13399).

---

## R1. Prepare

**Automated** (`souschef-prepare`, `python -m souschef_lab.pipeline prepare --run-id $RUN --bucket $PIPE --mode full`).

1. **Rebase every family onto the deployed manifest** (`retrain/rebase.py`;
   r4 could only rebase the verbatim family and dropped F1–F6 as
   `other_manifest`). For each pool case:
   - apply the static diff's `renames` everywhere a key appears (fields,
     expected keys and alts, ops, oracle ops, history ops, direction, list ops);
   - drop with a reason when a key, list or container is gone;
   - **F0 (verbatim) relabel**: F0 requests quote the label word for word, so a
     changed label re-renders the request from the generator template
     (`meta.relabelled`); other families keep their merchant phrasing;
   - re-derive every turn's context with `state.context_of` and the deployed
     engine's browser defaults, replaying earlier turns through the engine;
   - re-measure through the engine by score mode (`exact`: diff and
     `alt_exact`, drop on `exact_refused` / `exact_empty`; `keys`: oracle ops
     must apply; `where` / `clarify` / `answer`: unchanged);
   - re-run the hold-out filter (`leak.py` rules 1–3), since relabelled
     requests are new text.

   **A family that loses more than 20 % fails the stage** (`rebase_loss`): a
   person looks before anything is spent.

2. **Author cases for what changed** (`retrain/author.py`) through the
   Anthropic API, as the r4 authors did by hand:
   - Targets: R3-added, renamed and **retyped** fields, added lists and
     containers, and R2 changes whose old cases were dropped. A target is
     skipped when the pool already holds ≥ 6 kept cases writing it (counting
     skipped rounds); a hold-out target counts earlier rounds' F9H cases, so
     it is not authored (and paid for) again after a Skip.
   - Only the prepare job's account (`souschef-author`) reads the Anthropic
     key; collect and dataset run as `souschef-prep` without it.
   - 8 cases per field: 2 verbatim-ish, 2 colloquial (aliases from
     `service/internal/fields/vocab.txt`), 1 short or relative command, 1
     focus case (the section open, no target named), 1 two-turn follow-up, 1
     near-miss or out-of-editor where the neighbourhood invites one. A new
     section, panel or page also gets 6 container cases.
   - Model `claude-opus-5-5`, effort high, JSON-schema output, streaming,
     server-side fallback, prompt caching on the fixed rules, 8 at a time.
   - The rules are r4's: realistic merchant phrasing, no "preview", British
     spellings and typos in a share, no invented UI words. **Exam texts are
     never shown to the author.**
   - Validation, every rejection counted: keys exist on the deployed manifest;
     oracle ops apply through the engine with a non-empty diff; a literal value
     means `exact`, an authored or relative value means `keys`; the
     `preview_claim` rule; the hold-out filter against the exam and the study's
     conversations; dedup against the pool. A rejected case is re-asked once
     with the reason, then dropped.
   - **Hold-out**: targets with `sha1(field_key) % 100 < 20` go to
     `cases/exam-ext/<round>.jsonl` as family **F9H**, never collected or
     trained, and scored in R6.

3. **Accumulate.** The authored file is written to
   `cases/pool/authored/<round>.jsonl` **before** collection, so a Skip keeps
   it; the next round reads every authored file. On Approve the round is
   marked `trained_in` in `cases/pool/index.json`.

Output: `prepare/cases.jsonl` (the r4 schema, families F0–F6, authored F7+,
holdouts F0H and F9H), `report.json` (kept, relabelled, renamed, dropped by
reason, per family; Anthropic spend from `usage`; rejects by reason),
`leak-report.json`.

**By hand.**

```bash
gcloud run jobs execute souschef-prepare --project $PROJECT --region $REGION --wait \
  --args=prepare,--run-id,$RUN,--bucket,$PIPE,--mode,full
gcloud storage cat $W/prepare/report.json | jq '.families, .authoring, .rejects'
```

To change the round's authored cases by hand: edit
`cases/pool/authored/$RUN.jsonl` (and `cases/exam-ext/$RUN.jsonl`) in the
bucket and re-run prepare. Prepare never re-authors a round whose files exist;
it reads them back, so your edits are what the round collects and trains on.

To accept a `rebase_loss` you have read and agree with, re-run prepare with
`--accept-rebase-loss <family>[,<family>…]` (or `all`) added to the job's
args, or resume the round with `"accept_rebase_loss": "F3"` in the workflow's
input (R8). The report records the accepted families under
`rebase.loss_accepted`.
Then resume the round from `collect` (R8).

## R2. Teacher collection

**Automated** (`souschef-collect`). `r4collect run` at its core, driven as
r4's `collect.sh` did it:

| pass | route | cases | site prefix |
|---|---|---|---|
| smoke | `default` | 7, one per family (a follow-up among them) | `lab-<run8>s-` |
| pass 1 | `default` (Claude Sonnet 5.5) | all trainable | `lab-<run8>p1-` |
| pass 2 | `opus` (Claude Opus 5.5) | cases without a usable row | `lab-<run8>p2-` |
| plan 3 | — | assigns each remaining case the better route for its family | |
| pass 3 | `opus`, then `default` | the rest | `lab-<run8>p3<route>-` |

- **Target:** int's own service with the pipeline key, `--build-id lab-<run>`
  (a lab registration by the deployed-manifest rule). `/version` must report
  `fields_context: true` for `default` and `opus`. Name the route on every
  call: on int a turn without `model` goes to blueberry-muffin.
- **The smoke gate comes first:** `r4dataset --render-check` on the 7 rows,
  then `tools/r4_smoke_check.py` must say OK: every row rendered from the
  stored conversation, no block drift, no render error. Block drift means int
  is not serving the retrieval the lab image carries; the stage stops before
  spending more.
- **The teacher build is pinned.** The expected build is the lab image's
  `/opt/sous-chef/BUILD_ID`, which equals the deployed service tag; it is
  checked before every chunk. A mismatch (`teacher_changed`, someone deployed
  mid-round) restarts collection once, from scratch, as `--attempt 2` (new
  site prefix and logs): rows from two builds are never mixed. A second
  change fails the stage.
- **Resume.** Logs are fsynced locally and synced to `collect/` every 60 s and
  at exit; a retried task pulls them back and resumes by case id. Chunks run
  `--max-seconds 420` in a bash loop; exit 3 means "cases left, run again".
- **Do not change the round's `prepare/cases.jsonl` during collection.** The
  logs refer to its ids (r4 rule).
- 6 workers (`COLLECT_WORKERS`; int runs 8 concurrent turns an instance, so 6 leave merchants room instead of 503 busy), task timeout 24 h, 2 retries.

**By hand** (spends Claude on int; about US$135–175 for a full round). The
lab tools need the binaries and service tree of the **deployed** build, or
every row shows block drift. Take them from the lab image rather than your
working tree:

```bash
T=$(gcloud run services describe sous-chef --project $PROJECT --region $REGION \
     --format='value(spec.template.spec.containers[0].image)' | sed 's/.*://')
WORK=~/souschef-$RUN; mkdir -p $WORK && cd $WORK
docker create --name lab-$RUN $REGION-docker.pkg.dev/$PROJECT/sous-chef/sous-chef-lab:$T
docker cp lab-$RUN:/opt/sous-chef ./sous-chef && docker rm lab-$RUN
export SOUS_CHEF_SERVICE_DIR=$PWD/sous-chef/service
export SOUS_CHEF_FIELDS_BIN=$PWD/sous-chef/bin/sous-chef-fields SOUS_CHEF_PREFIX_BIN=$PWD/sous-chef/bin/sous-chef-prefix
gcloud storage cp $W/prepare/cases.jsonl $W/run.json .        # run.json names the deployed manifest hash and engine sha
gcloud storage cp $PIPE/manifests/<deployed hash>.json manifest.json
gcloud storage cp $PIPE/engine/<sha>/snapshot.tar.zst . && mkdir -p cc && tar --zstd -xf snapshot.tar.zst -C cc
export COOKIE_CUTTER_DIR=$PWD/cc ENGINE=$PWD/cc/launcher/client/src/sousChef/ops/headless.mjs
export URL=$(gcloud run services describe sous-chef --project $PROJECT --region $REGION --format='value(status.url)')
read -rs SOUS_CHEF_KEY && export SOUS_CHEF_KEY                # the pipeline key: never on argv, never echoed
PY=~/projects/breadstack-sous-chef/lab/.venv/bin/python; LAB=~/projects/breadstack-sous-chef/lab
P=lab-${RUN:3:8}

chunk() {  # chunk LABEL ROUTE PREFIX extra…  (exit 3 = cases left: run the same command again)
  local label=$1 route=$2 prefix=$3; shift 3
  ( cd $LAB && timeout 560 $PY -m souschef_lab.r4collect run --cases $WORK/cases.jsonl --log $WORK/$label.jsonl \
      --label $label --route $route --manifest $WORK/manifest.json --engine $ENGINE --workers 8 \
      --max-seconds 420 --site-prefix $prefix --url $URL "$@" ) 2>&1 | tee -a $WORK/$label.log | tail -3
  return "${PIPESTATUS[0]}"
}
```

Then, one command at a time (each returns in under 10 minutes):

```bash
curl -fsS $URL/version | jq '.build_id, .route_info.default.fields_context, .route_info.opus.fields_context'   # = $T, true, true
# smoke: 7 cases, one per family (a follow-up for F2; never a holdout), then the gate
$PY - cases.jsonl > smoke.ids <<'EOF2'
import json, sys
seen = {}
for line in open(sys.argv[1]):
    c = json.loads(line); f = c["meta"]["family"]
    if c.get("holdout") or f in seen or (f == "F2" and not c.get("history")):
        continue
    seen[f] = c["id"]
print("\n".join(seen.values()))
EOF2
chunk smoke default ${P}s- --ids-file $WORK/smoke.ids
( cd $LAB && $PY -m souschef_lab.r4dataset --render-check --cases $WORK/cases.jsonl --log $WORK/smoke.jsonl \
    --manifest $WORK/manifest.json > $WORK/smoke-check.json ) && $PY $LAB/tools/r4_smoke_check.py smoke-check.json smoke.ids
chunk pass1 default ${P}p1-                                                       # repeat while it returns 3
chunk pass2 opus ${P}p2- --select unusable --prior $WORK/pass1.jsonl              # repeat while 3
( cd $LAB && $PY -m souschef_lab.r4collect plan3 --cases $WORK/cases.jsonl --log $WORK/pass1.jsonl --log $WORK/pass2.jsonl --out-dir $WORK )
chunk pass3-opus opus ${P}p3opus- --ids-file $WORK/pass3-opus.ids --select unusable --prior $WORK/pass1.jsonl --prior $WORK/pass2.jsonl
chunk pass3-default default ${P}p3default- --ids-file $WORK/pass3-default.ids --select unusable --prior $WORK/pass1.jsonl --prior $WORK/pass2.jsonl
( cd $LAB && $PY -m souschef_lab.r4collect status --cases $WORK/cases.jsonl --log $WORK/pass1.jsonl --log $WORK/pass2.jsonl \
    --log $WORK/pass3-opus.jsonl --log $WORK/pass3-default.jsonl )             # usable per family, flags, US$ so far
gcloud storage cp $WORK/*.jsonl $WORK/*.ids $W/collect/      # hand the logs back to the round, then resume from dataset
```

`DRY=1`-style rehearsal: replace `--url $URL` with `--fake --oracle --fake-blocks`.

## R3. Dataset and the parity gate

**Automated** (`souschef-dataset`). `r4dataset` with the deployed manifest,
every pass's logs, `--section-cap-scope F0,F1` and the study folder for the
final hold-out check. Gates, each of which fails the stage:

1. **Render parity** (`souschef_lab/retrain/parity_gate.py`, the `tests/r4_parity.py`
   pattern): three conversations (a 2-turn follow-up with `lookup_fields`, an
   out-of-editor `report_unsupported`, a single write) through the real
   service binary in the image, with the lab's scripted fake model and the
   harness's Postgres. Every model-call body must equal
   `r4dataset.student_source` byte for byte: system, tools, stable history,
   follow-up block, `thinking: disabled`. If this fails, the rows would teach
   the student a prompt the service never sends.
2. `stats.json`: `block_drift_rows == 0`, `preview_claim_kept == 0`,
   `where_missing == 0`.
3. The row checks of `tools/run-r4.sh`: blocks present, follow-ups > 0, no
   reply mentioning the preview (`r4score.mentions_preview`: preview,
   previews, previewed, previewing).
4. `train --dry-run`: rows, tokens, updates, none over 32,768 tokens.

It also writes `model/context.tar.gz` (the image's `lab/model-image/`) for R5,
so Cloud Build needs no repository access.

Output: `dataset/{dataset/{train,val,rejects}.jsonl, stats.json, dataset-manifest.json, manifest.json}`,
the layout `train.py` and `sous-chef-data-pull` read. r4's yield for scale:
2,595 train / 245 val, 321 follow-ups.

**By hand** (no network; in `$WORK` from R2 with the same exports):

```bash
( cd $LAB && timeout 590 $PY -m souschef_lab.r4dataset --cases $WORK/cases.jsonl \
    --log $WORK/pass1.jsonl --log $WORK/pass2.jsonl --log $WORK/pass3-opus.jsonl --log $WORK/pass3-default.jsonl \
    --manifest $WORK/manifest.json --out $LAB/data/$RUN --why <study folder> --engine $ENGINE \
    --section-cap-scope F0,F1 | tail -60 )
jq '.block_drift_rows, .preview_claim_kept, .where_missing' $LAB/data/$RUN/stats.json        # 0 0 0
( cd $LAB && .venv-train/bin/python -m souschef_lab.train --data data/$RUN/dataset --name $RUN-sft --fields-context on \
    --eval-steps 150 --val-max-rows 150 --checkpoint-steps 50 --dry-run )
```

Parity by hand: `python -m tests.r4_parity --binary $WORK/sous-chef/bin/sous-chef`
with the same `SOUS_CHEF_*` exports and a local Postgres on `:5433`
(`make db-up` at the repo root); `docs/LAB.md` §5b.

## R4. Training

**Automated.** A Vertex AI custom job on the trainer image
`sous-chef-trainer:t-<hash12>` (CUDA 13, the pinned unsloth/torch env, a
prebuilt llama.cpp, the lab), entry `tools/pipeline-train.sh <run_id> <full|smoke>`:

1. pull `dataset/` and the base tar, `sha256sum -c`;
2. pull `train/checkpoints/` if present (a resume after preemption);
3. row checks, then the dry plan;
4. on the first attempt only, the resume smoke (2 updates with a checkpoint,
   resumed to 4: it must log `Resuming from …checkpoint-2`);
5. full SFT, checkpoints synced to the bucket every 2 minutes;
6. export q4_k_m with `base-modelfiles/qwen3.5-9b.modelfile`;
7. `make model-stage NAME=<id> MODEL_TEMPERATURE=0.3 MODEL_PRESENCE_PENALTY=0`;
8. push adapter, export and stage to `train/`;
9. write `DONE`. Every step updates `progress.json` (`status` running |
   failed | interrupted | done, `step`, `error`, `updated_at`, `last_step`,
   loss, eval loss, tokens/s, peak memory). A failing step writes
   `status: failed` (the workflow classes it **fatal**: not retried on another
   machine). A SIGTERM (a Spot preemption, a Vertex restart or cancel) reaches
   the script's trap at once — each step runs in the background and is
   `wait`ed for — which stops the step and its children, pushes the
   checkpoints and writes `status: interrupted` (classed **preempted**:
   resumed from the checkpoints).

**The recipe is r4's; only the data changes.** `unsloth/Qwen3.5-9B`, bf16
base, fp32 LoRA r16/α16, lr 2e-4, 1 epoch, batch 1 × grad_accum 4,
adamw_8bit, warmup 3 %, max_seq 32,768, `--eval-steps 150 --val-max-rows 150
--checkpoint-steps 50 --resume`. One epoch on purpose: a second epoch on
authored paraphrases teaches their wording rather than the mapping.

**The machine ladder.** A rung is left at once on no capacity — a create that
says so, a job that ends for lack of resources, or a job that has **not
started** after `max_pending_s` (1 h; Flex-start: its max wait + 30 min; it is
cancelled) — after its `attempts` preemptions (each a resume from the
checkpoints), and after two failures of unknown cause on it (four in all fail
the stage). A trainer failure (`progress.json` status `failed`) fails the
stage at once. SMOKE caps the Flex-start wait at 30 minutes.

| # | machine | scheduling | regions | notes |
|---|---|---|---|---|
| 1 | `a3-highgpu-1g` (1× H100 80 GB) | Flex-start, max wait 4 h | us-central1 | not preempted once started; preemptible quota (64) |
| 2 | `a3-highgpu-1g` | Spot, up to 3 preemption resumes per region | us-central1 → us-east4 → us-west1 | a preemption loses ≤ 50 updates (≈ 9 min) |
| 3 | `a2-ultragpu-1g` (A100 80 GB) | on demand | us-central1 | needs the quota request ([PIPELINE.md §4.1](PIPELINE.md#41-quotas-a-person-start-first-they-take-a-day-or-more)): 0 today, so the rung ends as `capacity` (at once, or after `max_pending_s`) and the ladder moves on |
| 4 | `a2-highgpu-1g` (A100 40 GB) | on demand | us-central1 | fits (r3 peaked at 21.6 GB), about 2× slower |

**The job's lifecycle.** Vertex needs its service agent before the first
job ([PIPELINE.md §4.4](PIPELINE.md#44-the-two-secrets-first-a-person-targeted-apply-then-no-echo-writes)).
The workflow writes the job's name to `wf/runs/<id>/train/current.json` the
moment the create returns, and removes it once the job has ended. The
create has no request id, so a create whose answer is lost (a 5xx, a
dropped connection) is not simply sent again: each create carries a
one-off `create` label, and the workflow lists the region's jobs by that
label (`customJobs.list`, filter `labels.create=<value>`) and takes the job
it finds before it sends the create once more. A resume
(or a re-run of the stage) finds a job there that is still queued or
running and **adopts** it instead of creating a second one (two jobs would
write the same `progress.json` and checkpoints); a job from a rung no
longer in the ladder is cancelled. A round that fails while a job runs
cancels that job, so it does not keep an H100 billing. An execution
cancelled by hand runs no step after the cancel: its job keeps running
until a `resume_from train` adopts it, or until you cancel it
(`gcloud ai custom-jobs cancel <job> --region <region>`, the job and region
are in `current.json`).
Every cancel (overdue, never started, a stopped round) is **waited out**:
Vertex cancels asynchronously, so the workflow polls every 30 s, up to
15 min, until the job is `SUCCEEDED`, `FAILED`, `CANCELLED` or `EXPIRED`
before it moves to the next rung (a job that still succeeds is taken; one
still not ended after 15 min fails the stage as training `fatal`, its
`current.json` kept, rather than start a second job beside it).
A Flex-start or Spot job that ends without ever starting is `capacity`,
whatever its message.

Disk pd-ssd 500 GB, timeout 12 h, service account `souschef-trainer` (bucket
only, no secrets). Expect 610–650 updates for an r4-sized set, 2.0–2.4 h of
SFT on the H100, eval loss falling across about 4 validations.

**By hand on Vertex** (re-run the stage alone):

```bash
make pipeline-retrain RESUME=train RUN=$RUN    # = gcloud workflows execute souschef-retrain … resume_from train
gcloud storage cat $PIPE/wf/runs/$RUN/train/current.json | jq     # the job the round is on, while one runs
gcloud ai custom-jobs list --project $PROJECT --region us-central1 --filter="displayName~$RUN"
gcloud storage cat $W/train/progress.json | jq
gcloud ai custom-jobs stream-logs <job id> --project $PROJECT --region us-central1
```

**By hand on the training VM** (the pre-pipeline path, still intact;
[TRAIN-REMOTE.md](TRAIN-REMOTE.md) and r4's runbook): copy the round's dataset
to `lab/data/$RUN`, then

```bash
make -C lab train-remote-push DATA=data/$RUN
make -C lab train-vm-up                                     # a person: billed from here
VMSSH="gcloud compute ssh sous-chef-train --project $PROJECT --zone us-central1-a --tunnel-through-iap"
$VMSSH --command "sudo -iu souschef bash -lc 'cd /work/sous-chef/lab && sous-chef-kit-pull && NAME=$RUN DATA=$RUN setsid nohup bash tools/run-r4.sh >/dev/null 2>&1 </dev/null &'"
$VMSSH --command "grep -E '^== |validation loss|Resuming|Done in' /work/runs-$RUN.log | tail -20"
make -C lab train-remote-pull NAME=$RUN-sft && make -C lab train-remote-pull NAME=$RUN
make -C lab train-vm-down                                   # a person
```

After a Spot stop: `train-vm-up` again and re-run the start command; it skips
the smoke and resumes.

## R5. Export, stage and the model image

**Export** happens inside the training job (step 6): unsloth's GGUF writer
with the trainer image's prebuilt llama.cpp, so nothing is cloned or compiled
at run time (`export.py --method merged --llama-cpp /opt/llama.cpp` is the
fallback). `train/stage/` is exactly what `lab/model-image/model-context.sh`
accepts: one GGUF and a `Modelfile` with temperature 0.3, presence_penalty 0,
top_k 20, top_p 0.95, num_ctx 65536.

**The image** (Cloud Build, `infra/modules/pipeline/cloudbuild-model.yaml`,
service account `souschef-builder`, `E2_HIGHCPU_8`, 200 GB, 1 h, worker
release `2026.09` = Docker Engine 29): copy the stage, `model-context.sh`,
`docker build --provenance=false --sbom=false --build-context model=…` with
`gcr.io/cloud-builders/docker:29` (pinned by digest) from
`model/context.tar.gz` (`context.tgz` from lab images before the rename; the
workflow takes either), push `us-east4-docker.pkg.dev/$PROJECT/sous-chef-model/model:c-$RUN`
(tags are immutable), record the digest in `model/image.json`. The default
worker and builder are Docker 20.10, which answers `unknown flag:
--build-context`; without `--provenance=false` buildx on Engine 29 pushes an
index with an attestation manifest and the recorded digest would be the
index's. The workflow gives a build that stays queued or pending (awaiting
approval) 3 h, then cancels it (`build_stuck`). A build that ends
`TIMEOUT`, `INTERNAL_ERROR` or `EXPIRED` is resubmitted once, unless the
`c-<id>` tag already exists: then the push finished and the image under the
tag is used (an immutable tag can never be pushed again). The context
is job-written, so the build spec Terraform writes (`infra/cloudbuild-model.json`)
pins the sha256 of `Dockerfile`, `.dockerignore`, `entrypoint.sh` and
`model-context.sh` from the applied checkout's `lab/model-image/`, and the
build's first step refuses a context that differs. A change to
`lab/model-image/` therefore needs a `terraform apply` before the next round's
build (the lab image carries the same files). The candidate and eval stages
serve only the digest the `c-<id>` tag names.

**By hand**, from a staged directory:

```bash
gcloud storage cp -r $W/train/stage ./stage
make -C lab model-build MODEL_STAGE=$PWD/stage MODEL_TAG=c-$RUN     # CPU only, local docker
make -C lab model-push MODEL_TAG=c-$RUN                             # prints the push commands; run them
```

or re-run the pipeline's own build for the round:
`--data='{"mode":"full","resume_from":"build","run_id":"…"}'` (R8).

## R6. Candidate and the paired eval

**Automated.**

1. `sous-chef-model-cand` (same shape as production: L4, 4 vCPU, 16 Gi,
   concurrency 4, min 0, max 1, only the pipeline accounts may invoke it) is
   patched to `model@<digest>` (the build's digest), waited Ready, and warmed
   with one generate. Without an L4 the stage keeps trying for
   `l4_wait_minutes` (6 h) before it fails; so does the eval stage below.
2. The eval lock is taken and `sous-chef-model-eval` is set to the
   production service's current digest.
3. Job `souschef-eval` starts the harness ([DRIFT.md §5](DRIFT.md#5-where-the-exam-runs))
   with two routes, `current` (eval URL) and `candidate` (candidate URL), and
   runs both arms at once, one per GPU, same rows, same contexts:

| set | contexts | replicates | in the bar? |
|---|---|---|---|
| real-phrasing exam, 6 sets, rebased onto deployed | re-derived | 2 | yes |
| real-phrasing exam, recorded contexts (continuity with `measure-wave2`) | recorded | 1 | reported |
| F0H rebased onto deployed, split `:unseen` / `:seen` against this round's training cases | built | 1 | `:unseen` yes |
| F9H, every round's authored new-key holdout | built | 2 | when n ≥ 10 |
| the new-key probe | built | 2 | when keys were dodged |

**The bar** (`retrain/bar.py`). `tol_s = max(1 row, ⌈2 σ_Δ,s⌉)` from the
current model's baseline. The candidate passes when all hold:

1. on every bar set, candidate − current ≥ −tol_s;
2. a pooled paired McNemar test does not say the candidate is worse (p < 0.05);
3. `out_of_scope` ≥ 11/13 with `where`;
4. F0H `:unseen`: candidate ≥ current − tol;
5. a round triggered by dodged keys: ≥ 75 % of their probes pass and more than
   current; a round triggered by exam drift: the deployed-manifest total
   improves by ≥ σ_Δ;
6. ghost writes ≤ current.

The bar is **advisory**: you can Approve a failing candidate, behind a typed
confirmation. If the launcher changed again during the round (R3 drift
against the round's own manifest), the eval also runs the newest manifest and
the decision carries `stale_round: true` with a Skip recommendation.

**Reading the result.** The Slack table and the decision page show, per set,
n, current, candidate, Δ, the tolerance, McNemar p and its discordant counts
(b, c), with Wilson intervals on the page. Read the **paired** columns, not
overlapping intervals: the same rows were asked of both. `runs/<id>/report.json`
holds everything; `eval/candidate-baseline.json` becomes the drift baseline on
Approve.

**By hand.** Re-run the stage: `--data='{"mode":"full","resume_from":"candidate","run_id":"…"}'`
(re-deploys the candidate) or `"resume_from":"eval"`. The pre-pipeline way is
r4's: serve the candidate on the training VM with `tools/vm-solo.sh serve`,
run `python -m souschef_lab.exam run … --route student --rederive-contexts`
from the workstation through an IAP tunnel, and compare with
`python -m souschef_lab.examcmp` and `python -m souschef_lab.r4collect compare`
(docs/LAB.md §6b).

## R7. Decide, roll out, roll back, skip

**The wait.** After `candidate_ready` (with the mention) the workflow asks the
thermometer for the decision every 30 minutes, and refreshes the retrain
lock every 6 hours (its TTL is 72 h):
`GET /api/ingest/souschef/decisions/<candidate_id>` → `pending | approved |
skipped | expired`. An Approve is therefore acted on within about half an
hour. The window is `decision_ttl_days` (14, also the most allowed); expiry
counts as Skip. The bound is Workflows' 100,000 steps per execution, and
running out is an uncatchable `ResourceLimitError` (no `stage_failed`, the
lock held): the wait costs about 680 steps a day (about 2,160 with the
thermometer down, its retries included), so 15 days take at most about
32,000, leaving the rest for a slow round (measured in `tests/wfsim.py`;
`infra/modules/pipeline/variables.tf` has the arithmetic). The old 10-minute
poll cost 4,608 steps a day. Thermometer errors are retried; after 6 h of errors `stage_failed:
decision api` is posted and the wait continues.

**Only this round's decision counts.** The thermometer answers 409 to a second
`candidate_ready` for a candidate from another notification or run, and the
workflow acts on a decision only when its row names this round's notification
id (`<run>/candidate_ready/1`), run id, SMOKE flag and bar verdict, its
approval comes after its creation, and an approval of a failing bar carries
the override. The round's first ask is recorded in `wf/runs/<id>/decide.json`
(under `wf/`, which only the workflow account writes) and carries the
execution's id (`details.ask`); if that first ask is answered as a duplicate
without that id (someone posted the candidate first), the round stops. A
retried first ask whose first answer was lost carries the id and goes on. Each of these is `stage_failed: decision_mismatch`
with the mention, and nothing is rolled out or parked.

Only the approvers (`souschef_approvers`, default `gary.tong@breadstack.com`)
see the buttons; everyone else sees the page read-only. The first decision
wins; a second click gets "already decided".

**Approve** (and `pipeline_rollout` on; otherwise a dry rollout):

1. take the eval lock and wait until the eval and candidate services report
   no instances (the L4 quota is 3; a revision swap needs a spare; the wait
   goes away at quota 5);
2. check the digest: the `model:c-<id>` tag in the model repository
   (immutable), `model/image.json` and the decision payload's
   `results.candidate.digest` must name the same image — the one the approver
   saw — else `stage_failed: digest_mismatch` and production is untouched;
3. save the current image to `wf/runs/<id>/rollout.json` (a resume at
   `decide` after a finished rollout finds it `done` and never patches
   production again);
4. check the traffic: `sous-chef-model` must send all traffic to its latest
   ready revision (no `traffic` entries, or `LATEST` at 100 % and no pinned
   revision; a 0 % tagged preview is fine). A patch keeps a pinned split as
   it is, so a new revision would serve nothing: a pinned split (a rollback
   by traffic, a person's choice) is refused with `stage_failed:
   traffic_pinned`, production untouched and the approval standing; the
   message gives the `gcloud run services update-traffic … --to-latest`
   command, then `resume_from decide`. A dry rollout reports a pinned split
   in its `rolled_out` summary;
5. patch `sous-chef-model` to `model@<digest>` with annotation
   `souschef/candidate=<id>` (`forceNewRevision` when its last revision had
   failed), wait until the service itself has settled (generation observed,
   not reconciling, the new revision ready; no operation is read: [PIPELINE.md §2](PIPELINE.md#2-moving-parts)), then
   check that `trafficStatuses` sends 100 % to that revision (else
   `stage_failed: traffic_not_moved`), and warm it with one generate;
6. check int's `/version`: blueberry-muffin still points at the same URL (no
   service change is needed);
7. tag the image `live-<run>` (a record of every image that was live;
   adding a tag is allowed under immutable tags);
8. run the bucket's side of the decision as a job (`pipeline decision
   --outcome approved --digest <digest> --by <approver>` on `souschef-eval`,
   `retrain/decide.py`): `eval/candidate-baseline.json` →
   `baselines/<digest>/baseline.json` (calibrated: 2 paired-eval replicates),
   `state/model.json` (the candidate; the previous model into `history`;
   `training_cases` and `dataset` = copies of this round's `prepare/cases.jsonl`
   and `dataset/` under `cases/trained/<candidate>/`, which never expire), every authored
   round this one trained on marked `trained_in` in `cases/pool/index.json`,
   `state/ack.json` deleted; it leaves `result-decision.json`;
9. post `rolled_out`.

**Roll back** (a person):

```bash
make model-rollout TAG='<previous tag or sha256:digest>'   # repo root; rollout.json's "rollback" line names it
# or directly (the second command: `update` keeps a pinned split, so send all traffic to the new revision):
gcloud run services update sous-chef-model --project $PROJECT --region $GPU_REGION \
  --image "$GPU_REGION-docker.pkg.dev/$PROJECT/sous-chef-model/model:<previous tag>"   # or model@sha256:…
gcloud run services update-traffic sous-chef-model --project $PROJECT --region $GPU_REGION --to-latest
```

The previous image is in `wf/runs/<id>/rollout.json` and in `state/model.json →
history`. Drift notices a rollback by itself: it judges the model the
production service serves, from `history`, against its own baseline (still
under `baselines/<digest>/`), and posts a notice
([DRIFT.md "Manual rollbacks"](DRIFT.md#manual-rollbacks)). To make the
rollback permanent, put that history entry back as `state/model.json`. A model
rolled out by hand to a digest the pipeline has never seen needs its
`state/model.json` entry written by hand, then `drift calibrate`.

Terraform no longer owns the model image (`ignore_changes` in
`infra/modules/model-gpu`), so a later `terraform apply` never reverts a
rollout. `gpu_model_image_tag` only matters when the service is recreated:
keep it at a tag that exists.

**Skip** (or expiry): the same job with `--outcome skipped|expired` writes
`state/ack.json` (the trigger's `drift/<run>/ack-material.json`, or for a
round started by hand the newest check's `state/last-ack-material.json`, +
who and when: [DRIFT.md §3.3](DRIFT.md#33-after-a-skip)) and
`state/parked/<candidate>.json` (the candidate; its image stays in the
registry), then `skipped` is posted. A SMOKE round never touches the drift
state: its Skip writes `state/parked/` only, its Approve is a dry rollout. The round's authored cases stay in `cases/pool/authored/`; the next
round rebases and trains on them with whatever has accumulated since. To
roll out a parked candidate later, `make model-rollout TAG=c-<id>`, then
`drift calibrate` (its candidate baseline was measured on an older manifest).

**Cleanup** after either: `train/checkpoints/` is deleted; adapter, export,
stage, logs, eval and report stay (180-day lifecycle on `runs/`); locks are
released.

**Candidate images are never deleted automatically.** The `sous-chef-model`
repository has a cleanup policy for `c-*` images older than 30 days
(keeping the newest 10, `r*` and `live-*`), but its tags are immutable, and
"If a repository has immutable tags enabled, tagged artifacts can't be
deleted" ([cleanup policies](https://docs.cloud.google.com/artifact-registry/docs/repositories/cleanup-policy)):
the policy deletes nothing, and every candidate (about 6.4 GB) stays. A
parked candidate can therefore be rolled out at any time. To prune by hand
(a person; no round may be building or rolling out: `state/locks/retrain.json`
absent), lift the immutability for the deletes and put it back
([repository settings](https://docs.cloud.google.com/artifact-registry/docs/repositories/update-repo-settings);
`terraform apply` also puts it back, `modules/model-gpu` sets it):

```bash
REPO=us-east4-docker.pkg.dev/$PROJECT/sous-chef-model/model
gcloud artifacts docker images list $REPO --include-tags --format='table(version,tags,createTime)'   # pick the c-* to drop; never r*, live-*, or what sous-chef-model or state/model.json names
gcloud artifacts repositories update sous-chef-model --project $PROJECT --location us-east4 --no-immutable-tags
gcloud artifacts docker images delete "$REPO@sha256:<digest>" --delete-tags --project $PROJECT --quiet   # once per image
gcloud artifacts repositories update sous-chef-model --project $PROJECT --location us-east4 --immutable-tags
```

## R8. Debugging a failed run

**Where things are** (`$PIPE`):

```
state/deployed.json  model.json  ack.json  last-ack-material.json  parked/<candidate>.json  locks/{retrain,gpu-eval}.json
      failed-round.json  drift-failing.json  heartbeats/<day>.json
wf/outbox/<notification id>.json   wf/runs/<rt-run>/{decide.json, rollout.json}   the workflow's own (no job writes wf/)
baselines/<digest>/baseline.json
drift/<dr-run>/{result.json, static.json, report.json, exam/*, logs/*}
cases/pool/{index.json, r4/*, authored/<rt-run>.jsonl}   cases/exam-ext/<rt-run>.jsonl   cases/why/
cases/trained/<candidate>/{cases.jsonl, dataset/}    an approved model's training inputs (never expire)
engine/{index.json, <sha>/{snapshot.tar.zst, engine.json}}   manifests/<hash>.json   base/   config/thresholds.json
runs/<rt-run>/run.json                     the round: trigger, images, manifest, engine
              result-<stage>.json          each stage's {status, stage, run_id, notification?, metrics, artifacts[]}
              prepare/  collect/  dataset/  train/{checkpoints/,adapter/,export/,stage/,progress.json,DONE}
              model/{context.tar.gz,image.json}  eval/  report.json  decision.json
```

**First look:**

```bash
gcloud workflows executions list souschef-retrain --project $PROJECT --location $WF_REGION --limit 5
gcloud workflows executions describe <execution id> --workflow souschef-retrain --project $PROJECT --location $WF_REGION \
  --format='value(state,error.payload)'
gcloud storage ls $W/ && gcloud storage cat $W/result-<stage>.json | jq
gcloud run jobs executions list --job souschef-<stage> --project $PROJECT --region $REGION --limit 3
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=souschef-<stage>" --project $PROJECT --limit 50 --freshness 1d
```

Never re-run or start an execution with `--call-log-level=log-all-calls` to
see more: it logs every call's arguments and results, the thermometer key
among them ([PIPELINE.md §7](PIPELINE.md#7-day-two)). The stage's inputs and
outputs are in `runs/<id>/`.

Exit codes of every job: 0 = the stage finished (the outcome is in its result
file), 1 = failure (retried by policy, then `stage_failed`), 2 = configuration
or precondition (no retry: fix and resume).

**Resume from a stage** (artifacts of the earlier stages are reused):

```bash
make pipeline-retrain RESUME=<prepare|collect|dataset|train|build|candidate|eval|decide> RUN=$RUN
# = gcloud workflows execute souschef-retrain --project $PROJECT --location $WF_REGION \
#     --data="{\"resume_from\":\"<stage>\",\"run_id\":\"$RUN\"}"
```

**A failed round blocks automatic rounds.** When a full round fails (after it
took the retrain lock), the workflow writes `state/failed-round.json` (`run_id`,
`stage`, `error_class`, `execution` and the exact `resume` command) and the
`stage_failed` message says so. The command names a stage the resume accepts:
a failed rollout or Skip resumes at `decide` (the decision stands, and a
finished rollout is not patched again). A failure at `images` (no run started)
or at `cleanup` (the rollout or Skip is done) writes no block. While it exists, a `retrain` verdict starts
nothing (the drift message names the round), so a deterministic failure never
costs a full round at every check. Resuming that round, or starting a full
round by hand, deletes it; so does `gcloud storage rm $PIPE/state/failed-round.json`
when you decide to let the round go. SMOKE never writes it.

**A resume never reuses an old decision.** `resume_from` any stage up to
`eval` is refused (before the lock, so it leaves nothing behind) when the
thermometer already has a decision row for the candidate — the re-run's
results would never reach the approver, and an earlier approval would apply
at once. Start a new round instead.

`resume_from: "decide"` re-attaches to an open decision (for example after the
execution was cancelled): the candidate_ready post is re-sent with the same id,
which the thermometer treats as a duplicate, and its expiry is kept. With
`resume_from: "prepare"`, `"accept_rebase_loss": "<families>"` passes
`--accept-rebase-loss` to the prepare job.

**Common failures:**

| failure | meaning | fix |
|---|---|---|
| `rebase_loss` | a family lost > 20 % on rebase | read `prepare/report.json`; accept with `--accept-rebase-loss` or revise (R1) |
| collect smoke-check not OK, block drift | int is not serving the retrieval in the lab image | the lab image or the service tag is wrong; `/version` vs `/opt/sous-chef/BUILD_ID` |
| `teacher_changed` twice | int was redeployed twice during collection | wait for deploys to settle, resume from `collect` |
| parity gate | the rows would not match what the service sends | a service change the lab render does not know; stop, fix the lab, re-run dataset |
| `selftest` on Vertex | driver vs cu130 wheels | the cu129 trainer pin (risk 1 of the design) |
| every ladder rung failed | no capacity anywhere | resume from `train` later, or the VM path (R4) |
| `lab image missing for <T>` / `trainer_image_missing` (stage `images`) | a hand-built service without `make lab-image TAG=<T>` / `make trainer-image TAG=<T>` | build them; no run was started and nothing blocks: the next drift check starts a round (or start one by hand) |
| `engine_stale` / retrain refused | no snapshot for the launcher's commit | `engine-vendor` ([DRIFT.md §6](DRIFT.md#6-the-engine-snapshot)), resume |
| `stale lock` (posted once a day by the drift workflow when a `retrain` verdict meets it) | a round's lock past its expiry: no progress for 72 h | check no execution is active, then `gcloud storage rm $PIPE/state/locks/retrain.json` |
| candidate not Ready | no L4 in us-east4 for `l4_wait_minutes` (6 h) | wait or request quota 5; resume from `candidate` |
| `revision_not_ready` | a patched GPU service's new revision did not become ready (no L4, a bad image); the message has its `terminalCondition` | resume the stage; the next patch to the same image forces a new revision |
| `revision_not_ready` (rollout) | production's new revision did not become ready: the old revision still serves; `rollout.json` is `unconfirmed` and keeps the old image | `resume_from decide` (the next patch forces a new revision); nothing to roll back |
| `operation_timeout` | a patched job or service did not settle (generation observed, not reconciling) within the wait | `gcloud run services describe` / `jobs describe` for its conditions; resume the stage |
| `traffic_pinned` (rollout) | `sous-chef-model` has a pinned traffic split, so a new revision would serve nothing; production untouched, the approval stands | if the latest revision should serve: `gcloud run services update-traffic sous-chef-model --to-latest --region us-east4`, then `resume_from decide` |
| `traffic_not_moved` (rollout) | the new revision is ready but `trafficStatuses` does not send it 100 % | read the service's traffic; production may serve the new image partly; fix the split, then `resume_from decide` (`rollout.json` keeps the old image for a rollback) |
| `rollout_unconfirmed` (rollout) | the patch of `sous-chef-model` was sent but its outcome is unknown (the wait timed out, a read failed): production may already serve the candidate; `rollout.json` is `unconfirmed` and keeps the old image | `gcloud run services describe sous-chef-model --region us-east4` (`trafficStatuses`, `latestReadyRevision`); then `resume_from decide` to finish the rollout, or roll back with the message's `make model-rollout TAG=…` |
| `build_stuck` | the model build stayed queued or pending for 3 h; it was cancelled | `gcloud builds list --region us-east4`; resume from `build` |
| `no_build_context` | `model/context.tar.gz` (or `.tgz`) is missing | resume from `dataset` |
| `build FAILURE` with `unknown flag: --build-context` | the build ran on the legacy worker or builder (Docker 20.10): `infra/cloudbuild-model.json` predates `workerRelease` | `terraform apply` (rewrites the spec), resume from `build` |
| `decision_mismatch` | the decision row is not this round's (another notification, run or SMOKE flag; a duplicate at the first ask; a failing bar approved without the override) | look at the decision page and the thermometer's API keys (only the key id in `souschef_ingest_key_ids` may post); then `resume_from decide`, or start a new round |
| `digest_mismatch` | the candidate tag, the build record and the decision's digest disagree (at rollout), or `model/image.json` is not what the `c-<id>` tag names (at candidate or eval) | do not roll out; read `model/image.json`, `decision.json` and the repository's `c-<id>` tag; at candidate or eval, `resume_from build` |
| `candidate_decided` (at init) | a resume up to `eval` of a decided candidate | `resume_from decide`, or a new round |
| training `fatal` | the trainer itself failed (`progress.json` status `failed`, the step in `error`) | read `train/train.log`; fix; resume from `train` |

Locks are only ever broken by a person, after checking
`gcloud workflows executions list … --filter='state=ACTIVE'` is empty.

## R9. Costs and durations

| stage | duration | cost |
|---|---|---|
| prepare | 30–60 min | Anthropic authoring: a few US$ per round (reported in `prepare/report.json`) |
| collect | 2.2–2.8 h at 6 workers | US$100–175 (*recorded* r4 estimate: pass 1 ≈ US$95–110, pass 2 ≈ US$30–50, pass 3 ≈ US$10–15) |
| dataset | 15 min | — |
| train + export | 2.4–2.8 h + 0–4 h Flex-start wait | about 3 H100-hours (the training VM's Spot rate was US$6.57/h; Vertex bills its own rate per scheduling) |
| image | 15–25 min | Cloud Build minutes; about US$0.90/month per kept tag |
| candidate + paired eval | 2–3 h on two L4s | about US$1.05/h each |
| **round** | **about 9–12 h** | **about US$120–230** |
| SMOKE | 2–6 h (the trainer image pull, the base tar, up to 30 min of Flex-start) | about US$2 of API, 1 H100-hour, 30 L4-minutes (about US$10–15) |

Collection is the big line and recurs every round: every case is re-collected
because the system index and the candidate-fields blocks are specific to the
manifest. Cost is reported per round, never a reason to stop (user decision).

## Lessons from r4 that apply to every round

- **bash, never zsh,** for loops that read exit codes (`PIPESTATUS`).
- **Pin the teacher build.** A deploy mid-collection changes the rules the
  teacher follows; rows from two builds are never mixed.
- **Spot only with resume.** Checkpoints every 50 updates, and the resume
  proven by the smoke before the long run.
- **The hold-out rules** (`leak.py`): Jaccard ≥ 0.6, same key and value at
  ≥ 0.4, containment ≥ 80 % on the same key. Exam texts never reach an author.
- **A family that loses more than 20 % on rebase needs a person.**
- **Read the `:unseen` split.** Designed families write some holdout keys;
  those rows favour the newer model, so the bar reads F0H `:unseen`.
- **`report_unsupported.where`** is required for out-of-editor rows to be kept
  (`where_missing` flags the rest); the bar's out-of-scope rule scores `where`.
- **Name the route** on int; the in-house model is the default route there.
- **Never regenerate a round's case file mid-collection.**
- **Keys from the environment only**, never argv, never echoed.
