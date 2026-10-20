# The Mistral arm (branch `mistral-arm`)

A one-off, isolated comparison: is **`mistralai/Ministral-3-8B-Instruct-2512-BF16`** (revision
`f6fae9795746f63c9be8344932f01275f3c63734`, Apache-2.0) a better Sous Chef student than Qwen3.5-9B? It runs
beside the r6 round (`rt-20261016-2102-3eeb97-eb908674`), and nothing it does can change r6, the deployed
service, the pipeline's images, or any existing bucket path.

The arm has three stages, each with a gate:

1. **Stage 1** (go/no-go): untrained Mistral against untrained Qwen3.5 on the real-phrasing exam.
2. **Stage 2**: one LoRA arm (`base:3407`) on r6's dataset, using r6's recipe.
3. **Stage 3**: paired evals against r4 and the r6 candidate, plus a control pair that must reproduce r6's own eval.

The design rationale is in the build spec (`scratchpad/mistral/SPEC.md`, written 2026-10-16). This file is the
runbook: what was built, the exact commands, the gates, the costs and the cleanup.

## Contents

- [0. Isolation rules](#0-isolation-rules)
- [1. What is in this branch](#1-what-is-in-this-branch)
- [2. Bucket layout](#2-bucket-layout)
- [3. One-time steps a person runs](#3-one-time-steps-a-person-runs)
- [4. The stages](#4-the-stages-main-session-or-a-person-submits-after-each-gates-check)
- [5. Gates and verdicts](#5-gates-and-verdicts)
- [6. Costs](#6-costs-estimates-us-central1-the-billing-report-is-the-truth)
- [7. Cleanup](#7-cleanup-a-person-after-the-decision)
- [8. Findings that changed the spec](#8-findings-that-changed-the-spec)
- [9. Tests](#9-tests)
- [10. The r7 arm](#10-the-r7-arm)

---

## 0. Isolation rules

- **Writes go to one prefix: `gs://breadstack-sous-chef-int-pipeline/experiments/mistral/`.** Every write from
  `lab/tools/mxarm.py` goes through `guard()`, which refuses anything else. That includes `runs/`, `base/`,
  `state/`, another experiment, `experiments/mistral` itself, and any `..` or empty path segment. No pipeline
  code reads `experiments/`.
- **Nothing here expires by itself, so it has a delete-by date.** The bucket's 180-day rule covers only `runs/`,
  `wf/runs/` and `drift/`, and `experiments/mistral/` holds copies of r6's case-derived dataset (`runs/$MX*/dataset/`)
  and the r4 snapshot. **Delete-by: 14 days after the Stage 3 decision (or after a STOP), and never later than
  2027-04-14**, the day r6's own `runs/` folder expires. §7 removes everything, the smoke run and both dataset
  copies included.
- **A job never runs twice.** Every exam, eval and base-gguf body sets `scheduling.disableRetries` and passes its
  `create` nonce as `MX_CREATE`. The runner claims `<dest>/attempts/<create>/started.json` before anything else;
  a container that finds it already there is a Vertex restart and exits 1 writing nothing. Each attempt uploads
  only into its own `attempts/<create>/` folder and then points `<dest>/latest.json` at it, which is what `gate`
  and `report` read. A resubmission (`jobs --attempt N+1`) gets a new nonce and a new folder.
- **Stage 2 needs no change to the trainer's scripts.** The trainer job sets
  `PIPELINE_BUCKET=gs://…/experiments/mistral`. As a result, `entrypoint.sh`, `selftest.py` and
  `pipeline-train.sh` (whose paths are `$PIPELINE_BUCKET/runs/<id>/…` and `$PIPELINE_BUCKET/base/…`) read and
  write only `experiments/mistral/runs/<id>/…` and `experiments/mistral/base/…`. `RUNS_ROOT` and `BASE_ROOT` are
  set to the same values, so a script that reads them agrees.
- **One service account, `souschef-mx`, runs every job.** It can read the bucket and can write only under
  `experiments/mistral/` (§3.1). The pipeline's accounts are never used.
- **Images get new names only:** `sous-chef-trainer-mx` and `sous-chef-mx`.
  - `sous-chef-trainer:*` and `sous-chef-lab:*` are only pulled: the first as the layer cache, by digest; the
    second as the base, by digest.
  - The `sous-chef-model` repository is never touched.
  - `images/trainer/`, `images/lab-runtime/`, `lab/model-image/` and the root `cloudbuild.yaml` are
    byte-identical to main. A test checks this.
- **Compute stays clear of r6:**
  - Region is us-central1.
  - L4 (STANDARD) for the exam and eval.
  - A100-40 **Spot** for training while r6 trains.
  - No H100 and no on-demand A100 until r6's `train/DONE` exists; `jobs` refuses those rungs without
    `--after-r6-done`.
  - No Cloud Run.
  - No lock is taken.
  - Labels are `purpose=sous-chef-mx` plus a `create=mx-<stamp>` nonce, never `sous-chef-pipeline`, so a round
    can never adopt or cancel an MX job.
- **Nothing is submitted by code.** `mxarm.py jobs` prints the bodies. The main session or the user submits them
  after `mxarm.py gates` passes.

## 1. What is in this branch

This list covers the images, the export check, serving and the runners, items (d) to (h). The renderer, the
masking and the training switch, items (a) to (c), are the other half of the branch.

| file | what it is for |
|---|---|
| `images/mx/Dockerfile` (+ `Dockerfile.dockerignore`) | **`sous-chef-mx`**, the exam and eval image. It is built `FROM sous-chef-lab:28491da@sha256:aa450fa3…`, which carries the deployed service binaries and therefore the deployed prompt, plus the lab venv, Postgres and node. It adds `/usr/bin/ollama` and `/usr/lib/ollama` from `ollama/ollama:0.34.4@sha256:8262851b…` (the model image's exact Ollama, which bundles llama.cpp b11081), `libgomp1`, `libopenblas0` and `libvulkan1`, the branch's `lab/`, and the model image's environment. At build it checks that `llama-server` links, that the version is 0.34.4, that both base Modelfiles are present, and that `mxarm.py selftest --no-gpu` passes. Entrypoint: `python /opt/lab/tools/mxarm.py`. |
| `images/mx/cloudbuild.yaml` | Builds and pushes **only** `sous-chef-mx:m-<hash12>` (`images/mx/tag.sh`). An existing tag is not rebuilt. |
| `images/mx/cloudbuild-trainer.yaml` | Item (d). Builds the **unchanged** `images/trainer/Dockerfile` from this worktree as **`sous-chef-trainer-mx:t-<hash12>`**, tagged by `images/trainer/tag.sh`. The cache comes from `sous-chef-trainer@sha256:7f37d23a…`, pulled only. It never pushes `latest` or an alias. No version change is needed: transformers 5.5.0 has `mistral3`/`ministral3`, and the pinned unsloth llama.cpp `b11160-mix-a6922cc` registers `Mistral3ForConditionalGeneration` as arch `mistral3` with the projector kept apart. |
| `lab/tools/mx_gguf.py` | Item (e). A standard-library GGUF reader plus a servability check for each family (details below). It runs in both images, after `base-gguf`, before every `ollama create`, and should run after the training export (§1.1). |
| `lab/tools/mxarm.py` | Items (f), (g) and (h). The subcommands are `base-gguf`, `exam`, `eval`, `gate`, `report`, `jobs`, `gates`, `prep-run` and `selftest` (module docstring). |
| `images/mx/tag.sh` | The `sous-chef-mx` content tag, `m-<hash12>`, over the lab the image copies plus `images/mx`, like `images/trainer/tag.sh`. `cloudbuild.yaml` recomputes it on the uploaded tree, refuses a `_TAG` that differs, and skips a tag that exists. |
| `lab/tools/mxtests/` | 112 offline tests: `test_mx_gguf.py`, `test_mxarm.py`, a fake `ollama` binary, a GGUF writer and a `conftest.py` that restores the scoring-policy env (§9). |

**What `mx_gguf.check --family mistral` requires.** Any failure exits 1.

- `general.architecture` is `mistral3`.
- The shape is 34 blocks, embedding 4096 and 8 KV heads, which is the 8B and not the 14B.
- `add_bos_token` is true, `bos` is 1 and `eos` is 2.
- Ids 1–9, 17, 18 and 32 hold their exact strings (`<s>` … `[ARGS]`) and are each token type CONTROL. This is what
  lets llama-server tokenize the TEMPLATE's `[INST]` and similar markers as one id each.
- There are no `clip.*` keys and no `v.*` or `mm.*` tensors.
- The rope is YaRN with factor 16 and a log multiplier. llama.cpp names it `yarn_log_multiplier`; Ollama names it
  `mscale_all_dim`.
- The `llama_4_scaling` attention temperature is 0.1. llama.cpp stores it as `attention.temperature_scale`;
  Ollama stores it as `rope.scaling_beta`.
- `general.file_type` is 15 (Q4_K_M).

For `qwen` the check is arch `qwen35`, no BOS, text only, and the quant. It is checked against the real r4 GGUF.

### 1.1 How the two halves are wired

- **`lab/base-modelfiles/ministral-3-8b.modelfile`** is generated, never hand-edited:
  `python -m souschef_lab.serving_mistral --print-modelfile > lab/base-modelfiles/ministral-3-8b.modelfile`.
  `tools/mxtests/test_mxarm.py` checks its TEMPLATE is `serving_mistral.TEMPLATE` byte for byte.
- **`lab/parity/main.go` has a template mode**: `-renderer`, `-parser` and `-modelfile <base Modelfile>`. With a
  Modelfile it renders through that TEMPLATE (Ollama 0.34.4's template package) with no thinking descriptor; with
  no flags it is the qwen3.5 renderer as before (`served.json` re-renders byte-identical).
  `python -m souschef_lab.serving_mistral --check` re-renders `fixtures/parity/served-ministral.json` through it.
- **`lab/tools/pipeline-train.sh`** reads `BASE_FAMILY` (`qwen` | `mistral`; anything else exits 2), `RUNS_ROOT`
  and `BASE_ROOT`. Unset, the commands are today's. `mistral` selects the Ministral base tar, the hub-cache folder
  `models--mistralai--Ministral-3-8B-Instruct-2512-BF16`, `train.py --family mistral` on every SFT and dry plan,
  `export.py --base-modelfile base-modelfiles/ministral-3-8b.modelfile --base-ollama-tag ministral-3:8b`, and a
  `gguf check` step (`tools/mx_gguf.py check … --family mistral --quant q4_k_m --sha256`, written to
  `export/gguf-check.json`) after the export and before stage and push. The dry plan gets `MX_TOKENIZER_DIR` = the
  base snapshot, so it prints real token counts. `lab/tests/test_pipeline_train_family.py` runs all of this against
  the stub gcloud. Since the pipeline took the family on ([RETRAIN.md §R11](RETRAIN.md#r11-the-base-family-chocolate-muffin),
  2026-10-19) the family can also come from `train-config.json` `family`, the stage is at temperature 0.15 and is
  checked (`chatfmt check-stage`), and a warm start is allowed from a LoRA on the Ministral base (chocolate-muffin's
  `base/adapters/cm-r7`; `train.py`'s dry plan refuses r4's and r6's Qwen3.5 LoRAs). The arm itself never warm-started.
- **`images/trainer/entrypoint.sh` and `selftest.py` are unchanged.** The `PIPELINE_BUCKET` prefix in §0 already
  sends every trainer path under `experiments/mistral/`, and the job sets `RUNS_ROOT`/`BASE_ROOT` to the same
  paths. `images/trainer/` stays identical to the image r6 is training with.
- **The prompt-size smoke** (§5.1 item 4) is in `mxarm.assert_served`.
- **Tests.** `lab/tools/mxtests/` is not collected by `make test` (`testpaths = ["tests"]`): run it as in §9.

## 2. Bucket layout

`EXP=gs://breadstack-sous-chef-int-pipeline/experiments/mistral`, `MX=mx-20261017-m3-8b`.

```
$EXP/base/mistralai-ministral-3-8b-instruct-2512-bf16-hf.tar(.sha256)   a person uploads it (§3.2)
$EXP/snapshots/r4/                                snapshot-r4: state/{model,deployed}.json, r4's baseline and training
                                                  manifest, snapshot.json (Stage 3's r4 pairs read these, §5.3)
$EXP/runs/$MX/
  plan.json, train-config.json, dataset/          prep-run (a server-side copy of runs/<r6>/dataset/, hashes compared;
                                                  the rows are dataset/dataset/{train,val}.jsonl)
  base-gguf/{latest.json,attempts/<create>/}      base-gguf's attempt claims
  models/{mx-base,qwen-base}/                     base-gguf: <arm>-q4_k_m.gguf, Modelfile, gguf-check.json, base-gguf.json
  models/mx-base-native/                          stage-native (optional diagnostic): mx-base's GGUF, no TEMPLATE/PARSER
  models/r4/                                      a person uploads it (§3.3): r4-q4_k_m.gguf, Modelfile
  exam/<arm>/latest.json                          → exam/<arm>/attempts/<create>/: started.json, summary.json,
                                                  <set>-k<k>.jsonl, served.json, gguf-check.json, ollama.log,
                                                  service.log | failed.json     (arm: mx-base, qwen-base, r4[, mx-base-native])
  gate.json, user-go.json                         the Stage 1 verdict; the user's GO on a JUDGE (user-go)
  train/…                                         pipeline-train.sh, exactly a round's layout (progress.json, checkpoints/,
                                                  adapter/, export/, stage/, DONE, train.log, selftest.json)
  eval/<pair>/latest.json                         → eval/<pair>/attempts/<create>/: eval/ (bar.json, current/ and
                                                  candidate/rows.jsonl, candidate-baseline.json, …), errors.json,
                                                  report.json, decision.json, result-eval.json, info.json,
                                                  gguf-check-*.json, ollama-{current,candidate}.log, service.log
  report/{report.json,report.md}                  the comparison
$EXP/runs/$MX-smoke/                              the Stage 2 smoke run (its own dataset copy and train/)
```

The following are **only read**:

- `base/unsloth-qwen3.5-9b-hf.tar`, the qwen-base weights;
- `runs/<r6>/{run.json,prepare/,result-*.json,model/image.json,train/stage/,train/DONE,eval/bar.json,eval/candidate-baseline.json}`;
- `state/{model,deployed}.json`, `baselines/`, `manifests/`, `config/thresholds.json`, `drift/<run>/`,
  `cases/pool/index.json` and `engine/`.

Everything under `$EXP/` is written by `mxarm.py` or by the person-run steps in §3, and nothing else.

## 3. One-time steps a person runs

Agents may not create IAM, download weights or push images.

Big local files go on a disk path, never `/tmp`: on the workstation `/tmp` is a RAM-backed tmpfs (63 GB, about
31 GB free), and the base tar alone is 17.8 GB. Every person-run upload sets
`CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED=False`, so no temporary composite pieces land at the bucket root.

```bash
P=breadstack-sous-chef-int; B=gs://breadstack-sous-chef-int-pipeline; BN=breadstack-sous-chef-int-pipeline
EXP=$B/experiments/mistral; R6=rt-20261016-2102-3eeb97-eb908674; MX=mx-20261017-m3-8b; REG=us-central1
AR=northamerica-northeast2-docker.pkg.dev/$P/sous-chef; MX_SA=souschef-mx@$P.iam.gserviceaccount.com
REV=f6fae9795746f63c9be8344932f01275f3c63734; TAR=mistralai-ministral-3-8b-instruct-2512-bf16-hf.tar
cd /home/gtong/projects/sous-chef-mistral; MXPY="lab/.venv/bin/python lab/tools/mxarm.py"
W=/home/gtong/mx-work; mkdir -p $W                    # a disk path for the base tar and the r4 staging
export CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED=False
```

In the worktree, `lab/.venv` may be missing. In that case use `/home/gtong/projects/breadstack-sous-chef/lab/.venv/bin/python`:
`mxarm.py` is standard library only for `jobs`, `gates`, `gate`, `report` and `prep-run`.

### 3.1 The `souschef-mx` account

This account writes `experiments/mistral/` only, which is narrower than compute.md §7.1, which also allowed
`runs/mx-*`.

```bash
gcloud iam service-accounts create souschef-mx --project $P \
  --display-name "Sous Chef Mistral arm: Vertex jobs (writes experiments/mistral/ only)"
gcloud projects add-iam-policy-binding $P --member serviceAccount:$MX_SA --role roles/logging.logWriter --condition=None
# read: every object but the workflow's wf/ and the infra/ tree (narrower than "the whole bucket")
gcloud storage buckets add-iam-policy-binding $B --member serviceAccount:$MX_SA --role roles/storage.objectViewer \
  --condition="title=mx-read-not-wf-infra,description=all but wf/ and infra/,expression=!resource.name.startsWith(\"projects/_/buckets/$BN/objects/wf/\") && !resource.name.startsWith(\"projects/_/buckets/$BN/objects/infra/\")"
gcloud storage buckets add-iam-policy-binding $B --member serviceAccount:$MX_SA --role roles/storage.objectAdmin \
  --condition="title=mx-experiments-only,description=experiments/mistral/ only,expression=resource.name.startsWith(\"projects/_/buckets/$BN/objects/experiments/mistral/\")"
```

The person who submits needs `iam.serviceAccounts.actAs` on `souschef-mx`; project Owner has it. The Vertex
custom-code service agent already pulls from the `sous-chef` repository, and a new image name there needs no
binding.

### 3.2 The Mistral base tar

This is about 17.8 GB. It is built on a machine with the Hugging Face CLI, laid out like the Qwen tar.

```bash
hf download mistralai/Ministral-3-8B-Instruct-2512-BF16 --revision $REV \
  --include 'model-*.safetensors' 'model.safetensors.index.json' '*.json' 'tekken.json' 'chat_template.jinja' 'SYSTEM_PROMPT.txt'
# a download BY COMMIT writes no refs/main, and the trainer and base-gguf look the snapshot up through refs/main:
D=${HF_HOME:-$HOME/.cache/huggingface}/hub/models--mistralai--Ministral-3-8B-Instruct-2512-BF16
mkdir -p $D/refs && printf %s $REV > $D/refs/main && ls $D/snapshots/$REV/config.json
# no consolidated.safetensors (a 17.8 GB duplicate the include list leaves out)
make -C lab base-tar BASE_HF_DIR=hub/models--mistralai--Ministral-3-8B-Instruct-2512-BF16 OUT=$W/$TAR
(cd $W && sha256sum $TAR > $TAR.sha256)
gcloud storage cp $W/$TAR $W/$TAR.sha256 $EXP/base/
```

The Qwen base is read in place from `$B/base/unsloth-qwen3.5-9b-hf.tar`. No copy is needed.

### 3.3 r4, the live student, as served

The local export's GGUF is the production blob when its sha256 matches. `export.json` records
`531ccfdde3dc62b76538cb9b5ad2ba6344412d20ffaa12bf9f49c6783c233499`; check that the `model:r4` image holds the same
blob.

```bash
docker run --rm --entrypoint ls us-east4-docker.pkg.dev/$P/sous-chef-model/model:r4 /models/blobs | grep 531ccfdde3dc62b7
cd /home/gtong/projects/breadstack-sous-chef/lab
make model-stage NAME=r4 MODEL_EXPORT=runs/exports/r4 MODEL_STAGE=$W/r4-stage MODEL_TEMPERATURE=0.3 MODEL_PRESENCE_PENALTY=0
# the PARAMETER lines must equal the served model's:
docker run --rm --entrypoint sh us-east4-docker.pkg.dev/$P/sous-chef-model/model:r4 -c \
  'ollama serve >/dev/null 2>&1 & sleep 5; ollama show sous-chef --parameters'
gcloud storage cp $W/r4-stage/r4-q4_k_m.gguf $W/r4-stage/Modelfile $EXP/runs/$MX/models/r4/
```

### 3.3a Freeze r4 as Stage 3's current model (now, while r4 is live)

Stage 3's `r4-mx` and `r4-r6c` pairs need r4's baseline and training manifest. If r6 is approved first,
`state/model.json` names r6, so the arm snapshots r4's state now. It refuses unless the live model is r4
(`sha256:f657e110…`, manifest `bf788701…`), it only reads `state/`, `baselines/` and `manifests/`, and it is
idempotent. It is not a person-only step: the main session may run it.

```bash
$MXPY snapshot-r4                # → $EXP/snapshots/r4/ (+ snapshot.json); `gates --stage 1` and `--stage 3` check it
```

### 3.4 The two images

These run in Cloud Build in `prod-devops-core-project`, from this worktree. Both tags are content hashes of the
tree that is uploaded (`images/trainer/tag.sh` → `t-…`, `images/mx/tag.sh` → `m-…`), so committed or not does not
matter, an edit gets a new tag, and re-submitting the same tree is a no-op. Build after the branch code is final.

```bash
gcloud builds submit --project prod-devops-core-project --config images/mx/cloudbuild-trainer.yaml \
  --ignore-file=images/images.gcloudignore .
gcloud builds submit --project prod-devops-core-project --config images/mx/cloudbuild.yaml \
  --ignore-file=images/images.gcloudignore --substitutions=_TAG=$(bash images/mx/tag.sh) .
TRAINER_MX=$(gcloud artifacts docker images describe $AR/sous-chef-trainer-mx:$(bash images/trainer/tag.sh) --format='value(image_summary.fully_qualified_digest)')
MXIMG=$(gcloud artifacts docker images describe $AR/sous-chef-mx:$(bash images/mx/tag.sh) --format='value(image_summary.fully_qualified_digest)')
# safety: r6's trainer is untouched
gcloud artifacts docker images describe $AR/sous-chef-trainer:latest --format='value(image_summary.digest)'
# must print sha256:7f37d23a02f398ce3f9ff9339d9dfe9b671100c94757be735dd48b770c78a233
```

The uncommitted worktree is uploaded as it is, which is fine: the tags name the content, not a commit.

## 4. The stages

The main session or a person submits each stage after its `gates` check passes.

One helper submits a body. This is how the workflow and `noise.py` submit.

```bash
API=https://$REG-aiplatform.googleapis.com/v1/projects/$P/locations/$REG/customJobs
submit() { curl -sS -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json" \
  "$API" -d @"$1" | tee "${1%.json}.created.json" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("name"), d.get("state"), d.get("error"))'; }
watch_job() { gcloud ai custom-jobs describe "$1" --region $REG --format='value(state,startTime,endTime,error.message)'; }
J=/tmp/mx-jobs; mkdir -p $J
```

### Prep: the base GGUFs

This runs on CPU: about 45–75 min, about US$1–2.

```bash
$MXPY gates --stage prep
$MXPY jobs --stage base-gguf --trainer-image $TRAINER_MX --out-dir $J && submit $J/souschef-mx-basegguf-$MX-1.json
gcloud storage cat $EXP/runs/$MX/models/mx-base/gguf-check.json | python3 -c 'import json,sys; print(json.load(sys.stdin)["ok"])'
```

- **Inside the job**, for each arm it:
  1. pulls the tar and checks its sha256;
  2. untars it into `$HF_HOME/hub` and checks that the snapshot is the pinned revision;
  3. runs the image's pinned `convert_hf_to_gguf.py --outtype f16`, then `llama-quantize Q4_K_M`;
  4. runs the `mx_gguf` check;
  5. writes the Modelfile through `export.student_modelfile`, then the real `make model-stage
     MODEL_TEMPERATURE=0.3 MODEL_PRESENCE_PENALTY=0`, and compares the result with `staged_modelfile()`;
  6. uploads it.
- **A check failure** writes `base-gguf.json` with `failed.reason=gguf` and exits 3. Nothing is staged.
- **If `mx-base` fails conversion or the check**, re-run that one arm with the converter that matches Ollama's
  loader. This needs egress to github.com from the job (§8):
  `$MXPY jobs --stage base-gguf --arm mx-base --converter-ref b11081 --trainer-image $TRAINER_MX --attempt 2 --out-dir $J`.

### Stage 1: the zero-shot exam

This is 3 × 1 L4: about 2–3.5 h, about US$7–14. It does not depend on r6.

```bash
$MXPY gates --stage 1            # both base GGUFs + r4 staged, the r4 snapshot, no MX exam running
$MXPY jobs --stage exam --mx-image $MXIMG --out-dir $J
for a in mx-base qwen-base r4; do submit $J/souschef-mx-exam-$MX-$a-1.json; done
# when all three have finished:
$MXPY gate                       # → $EXP/runs/$MX/gate.json; prints the verdict
```

**The job body.** It runs on `g2-standard-12` with 1 × L4, STANDARD, 200 GB pd-ssd, timeout 6 h,
`disableRetries`, as `souschef-mx`. The replicates are 2 for `mx-base` and `qwen-base` and 1 for `r4`. A failed or
blocked exam is resubmitted with `jobs --stage exam --arm <arm> --attempt 2`, which gets a new nonce and folder.

**Inside the job:**

1. It pulls the model folder and runs the GGUF check.
2. It starts `ollama serve` on 127.0.0.1:11434, with the model store under `/tmp`, and runs `ollama create
   sous-chef -f Modelfile`.
3. It runs the serving asserts (§5.1).
4. It fetches the engine from the bucket, using r6's launcher sha.
5. It starts the harness: the 28491da service, `auth=none`, `prime=off`, `MODEL_CALL_TIMEOUT=180s`, with the
   route named after the arm.
6. It runs all 252 rows of the six sets on the exam's recorded manifest. This is the eval's `<set>:recorded`
   path (`eval_stage.Arms.exam`), scored under the eval's STRICT policy with r6's trigger keys on top. Errors are
   recorded and are not fatal.
7. It writes `summary.json`: per-set and pooled pass counts per replicate, flips, format failures by kind,
   latency p50/p95, the Ollama status counts per endpoint, and the service's model-error lines.
8. It uploads everything. A blocked model writes `failed.json` and exits 3.

**Go/no-go goes to the user.**

- **GO** goes on to Stage 2.
- **JUDGE** always goes to the user, with `default` as the recommendation. The `mx-base-native` diagnostic below is
  optional. If the user says go, record it, which is what `gates --stage 2` accepts for a JUDGE (it is tied to that
  `gate.json`; a re-run gate needs a new decision):
  `$MXPY user-go --by "<who>" --note "<why>"` → `user-go.json`.
- **STOP** is not final until `mx-base-native` has run, because compact HTML-escaped Go JSON is not Mistral's
  native format. `gate.json` then says `confirm`. Re-run `gate` after the diagnostic: a native result that is not
  STOP turns the verdict into JUDGE.
- **BLOCKED** and **INVALID** (fewer than 2 replicates) stop here; look at `failed.json` or the summary.

**The `mx-base-native` diagnostic** (spec §(f)) is the same GGUF with a Modelfile that has no TEMPLATE or PARSER, so
Ollama serves the GGUF's own chat template. About 1 h on 1 L4, about US$2.

```bash
$MXPY stage-native               # models/mx-base → models/mx-base-native (server-side GGUF copy, native Modelfile)
$MXPY jobs --stage exam --arm mx-base-native --mx-image $MXIMG --out-dir $J && submit $J/souschef-mx-exam-$MX-mx-base-native-1.json
$MXPY gate                       # now with the native result
```

It goes through the same serving asserts. The prompt-size check only records the count for it. If Ollama does not
give the template-less model the `tools` capability, or Ministral makes no tool call on the echo smoke, it is
BLOCKED, and that is an answer too: the native path is not servable by this Ollama.

### Stage 2: LoRA SFT on r6's dataset

This uses the existing trainer path (`pipeline-train.sh`) as `souschef-mx`.

```bash
# after r6's result-dataset.json is ok: server-side copy of the dataset (hashes compared), train-config.json, plan.json,
# and the r4 snapshot if it is still missing
$MXPY prep-run --trainer-image $TRAINER_MX --mx-image $MXIMG
# CPU pre-flight (spec §4), on a local copy of $EXP/runs/$MX/dataset/:
#   MX_TOKENIZER_DIR=<the pinned snapshot> python -m souschef_lab.train --family mistral --data <copy>/dataset --dry-run
#   (it stops on rows whose control-token strings move their token spans — train.mask_problem, no tokenizer needed —
#   and warns about the rest that hold one as text)
$MXPY gates --stage 2 --rung a100-40-spot   # r6 dataset ok, copy present, GO (or JUDGE + user-go), r6 train RUNNING
                                            # or DONE, no MX train job; an H100 / on-demand A100 rung needs r6 train/DONE
```

**Status on 2026-10-16 23:40Z:** r6's `result-dataset.json` is `gate_failed` (`new_key_positives`: 25 new or
retyped keys short of 6 positive-write rows). Nothing was trained, and `prep-run` refuses until a dataset is `ok`.
If the round is redone under a new id, pass `--cases-run <that id>` to `prep-run`, `gates`, `exam`, `eval` and
`report`.

**Smoke run, then the full run.** `pipeline-train.sh` treats a run with `train/DONE` as finished, and a smoke run
ends with DONE too. So the smoke runs under its own id, `$MX-smoke`, with its own copy of the dataset. The full run
uses `$MX`. Run `gates --stage 2` (for `$MX`) before each.

```bash
$MXPY prep-run --run $MX-smoke
$MXPY jobs --run $MX-smoke --stage train --mode smoke --trainer-image $TRAINER_MX --out-dir $J
submit $J/souschef-mx-train-$MX-smoke-a100-40-spot-1.json
# the smoke passed: $EXP/runs/$MX-smoke/train/DONE, train/adapter/run.json (tokens/s), train/export/ (GGUF check)
$MXPY jobs --stage train --mode full --trainer-image $TRAINER_MX --attempt 1 --out-dir $J
submit $J/souschef-mx-train-$MX-a100-40-spot-1.json
```

**The body.** It runs on `a2-highgpu-1g` with 1 × A100-40, **SPOT**, 500 GB, timeout 12 h. The environment is:

- `PIPELINE_BUCKET=$EXP`, `RUNS_ROOT=$EXP/runs` and `BASE_ROOT=$EXP/base`;
- `BASE_FAMILY=mistral` and `BASE_TAR=<the Mistral tar>`;
- `TRAIN_ARMS=base:3407` and `TRAIN_GEN_CHECK=32`;
- `TRAIN_RUNG`, `TRAIN_ATTEMPT`, `TRAIN_NONCE` (equal to the `create` label) and `TRAIN_IMAGE`.

**What the smoke run proves:**

1. the Ministral load;
2. the LoRA targets: no `vision_tower` in `adapter_config.json`;
3. the masking guards;
4. the longest row;
5. the export plus the GGUF check;
6. `train_tokens_per_second`, which sets the ETA.

If the full arm's ETA is over about 10 h, cancel. Wait for r6 `train/DONE`, then run
`gates --stage 2 --rung h100-spot` (it fails until DONE exists) and only then
`jobs … --rung h100-spot|h100-flex --after-r6-done`.

**Spot preemption.** Resubmit with `--attempt N+1`. The new nonce means it resumes from `train/checkpoints/`.

**No warm start:** no `INIT_ADAPTER`, and no `warm:` arm.

### Stage 3: paired evals

This is 3 × 2 L4: about 2.5–4 h, about US$18–45.

```bash
$MXPY gates --stage 3            # r4 snapshot, r6 train/DONE, MX train/DONE, r6 eval bar.json + candidate-baseline.json,
                                 # no MX eval running
$MXPY jobs --stage eval --mx-image $MXIMG --out-dir $J
for p in r4-mx r6c-mx r4-r6c; do submit $J/souschef-mx-eval-$MX-$p-1.json; done
gcloud ai custom-jobs list --project $P --region $REG --filter='labels.purpose=sous-chef-mx' --format=json > /tmp/mx-jobs.json
$MXPY report --jobs-json /tmp/mx-jobs.json   # → $EXP/runs/$MX/report/report.{json,md}
```

**The body.** It runs on `g2-standard-24` with 2 × L4, STANDARD, 200 GB, timeout 8 h, `disableRetries`. Pass
`--mode smoke` in the job args for a 10-rows-per-set rehearsal.

**Inside the job:**

1. It pulls both models and runs the GGUF checks.
2. It starts two `ollama serve` processes, on ports 11501 and 11502, with `CUDA_VISIBLE_DEVICES` set to 0 and 1
   and separate model stores. Each gets its own asserts.
3. It builds a **local** store mirror (`mirror_store`) with exactly what `eval_stage.run` reads: `state/`, the
   current model's baseline and manifests, r6's `run.json`, `prepare/` and `result-*.json`, the drift report and
   the thresholds.
   - For `r4-mx` and `r4-r6c` it takes `state/model.json`, r4's baseline and its training manifest from
     `$EXP/snapshots/r4/`. Without a snapshot it refuses unless the live model is still r4, so an r6 rollout can
     never score r4's GGUF against r6's baseline.
   - For `r6c-mx` it patches the local `state/model.json` to the r6 candidate's digest and training manifest, and
     its baseline to r6's `eval/candidate-baseline.json`.
   - For MX as candidate it sets the local `model/image.json` to `mx:<run>`.
   - The pipeline bucket is never written, and a test checks this.
4. It runs `eval_stage.run(local store, r6, …)` with a harness factory that sets `auth=none`, `prime=off` and
   `180s`.
5. It recounts the bar sets from both arms' `rows.jsonl` with error rows as fails (`errors.json`, §5.3), counts
   each server's `/v1/messages` 500s (`info.json` `retries_500`), and uploads the local `runs/<r6>/eval/` and the
   report files to `eval/<pair>/attempts/<create>/`.

## 5. Gates and verdicts

### 5.1 Serving asserts

These run in every exam and eval job before any row. Any failure is **BLOCKED** (exit 3) and never a score.

1. **GGUF.** The `mx_gguf` check passes for the Modelfile's family: `ministral` means `mistral`, `qwen3.5` means
   `qwen`.
2. **`/api/show`.** It must report:
   - the Modelfile's PARSER and RENDERER (`ministral`, none / `qwen3.5`, `qwen3.5`);
   - **the TEMPLATE's sha256**, so a trailing newline after the closing `"""` is caught;
   - `num_ctx 65536`;
   - the `tools` capability;
   - the family's architecture.
3. **Load and GPU.** A load-only `/api/generate` succeeds, and `/api/ps` shows `size_vram == size` (100 % GPU). A
   CPU fallback would make every latency and timeout wrong.
4. **Prompt size.** The first prompt the server sees (the load carried none, so no cached prefix shortens it) is
   `serving_mistral.SMOKE_REQUEST` on `/v1/messages` with `max_tokens 1`. Its `usage.input_tokens` must equal
   `SMOKE_PROMPT_TOKENS[family]` (61 for the committed Mistral TEMPLATE, 279 for RENDERER qwen3.5), or the job is
   BLOCKED with reason `render`. That proves the template renders as the lab's `render_prompt`, adds one BOS and
   reads every control marker as one token. Any other template (the native diagnostic) records the count only.
5. **Decode.** `/api/chat` decodes a few tokens.
6. **Tool smoke.** `/v1/messages` with an `echo` tool at **temperature 0** is recorded in `served.json`.
   - A 404 blocks the job (`decode`).
   - A reply whose text holds `[TOOL_CALLS]` or `[ARGS]` blocks it (`render`): the control tokens never reached
     the parser, so every call in Stage 1 would read as a failure of the model.
   - For Mistral, a 200 without a `tool_use` block blocks it (`render`). Ministral calls `echo` here whenever the
     tools reach it and its `[TOOL_CALLS]` reaches the parser.
   - A 500 from Ollama's MinistralParser (unknown tool or bad JSON) does not block an untrained base, because that
     is exactly what Stage 1 measures. `--strict-smoke` makes it block. The smokes' own 500s are recorded as
     `smoke_500s` and left out of `retries_500`.

### 5.2 The Stage 1 verdict

`mxarm.gate` writes `gate.json`. It uses the rows common to both arms, normally 252, and the row thresholds
scale by n/252.

- s_a is the mean pass count over the 2 replicates.
- F_a is the number of rows whose two replicates disagree.
- Δ = s_mx − s_qw, and σ_Δ = 0.5·√(F_mx + F_qw).
- Δ_fmt is Δ over the rows where mx-base had **no format failure** in either replicate, rescaled to n. A format
  failure is a turn that errored (`model_unavailable` after a parser error, a stream end, a failed history) or
  that stopped at `max_tokens` having written nothing.

| verdict | condition |
|---|---|
| **GO** | Δ ≥ −max(13, 2σ_Δ) |
| **STOP** | Δ < −max(38, 3σ_Δ) **and** Δ_fmt < −25 |
| **JUDGE** | Anything else. `default` is GO when Δ_fmt ≥ Δ/2 (format failures explain at least half the gap), else STOP. The user decides. |
| **BLOCKED** | mx-base cannot be served (`failed.json` reason gguf, create, show, load, gpu, render or decode) |
| **INVALID** | either arm has fewer than 2 replicates: no row can flip, so σ_Δ would read 0 |
| MISSING | a summary is absent |

Two rules sit on top of the table:

- **Hidden retries.** Ollama answers a MinistralParser error with HTTP 500, and the service's SDK retries a 5xx
  twice, each a fresh sample at 0.3 (`service/internal/model/routes.go` `DefaultMaxRetries=2`). Qwen's parser passes
  unknown names through, so the Mistral arm gets extra draws the Qwen arm never does, and `fmt_kind` cannot see a
  recovered one. `gate.json` reports `retries_500` per arm (the `/v1/messages` 500s in Ollama's log, minus the
  smokes'). A GO is turned into **JUDGE** when Δ − (excess 500s per replicate) falls below the GO line, the worst
  case being one rescued row per excess retry.
- **STOP needs the native diagnostic.** Without an `mx-base-native` result a STOP carries `confirm` and is not
  final. With one, a native verdict other than STOP turns STOP into JUDGE (default GO when the native run is GO).

**The r4 anchor** is reported and is not part of the gate. It compares r4's pooled pass count here with r5's
`eval/bar.json` `<set>:recorded` current, which is r4 at **208/252** on Cloud Run with prime on. A gap larger
than max(2, 2·√(0.25·F_qw)) rows is flagged, which means look at the Vertex L4 and prime-off setup first.

### 5.3 The Stage 3 trust gate and verdict

`mxarm.report` applies these.

**Trust.** The control pair `r4-r6c` must reproduce `runs/<r6>/eval/bar.json` on three points:

- the same overall pass or fail;
- every bar set's |Δ_here − Δ_r6| ≤ that set's `tol` in r6;
- the same pooled McNemar direction.

If it does not, both MX verdicts are **untrusted**.

**The verdict on `r6c-mx`** (MX the candidate, the r6 candidate current) and on `r4-mx` comes from the
**non-inferiority rules** of `bar.json` (`sets`, `mcnemar`, `oos_where`, `f0h_unseen`, `ghosts`), recounted with
error rows as fails, plus the pooled McNemar test:

- **The trigger rule is not part of the `r6c-mx` verdict.** It asks the candidate to improve on the current model
  (exam drift: Δ ≥ σ_Δ; dodged keys: more probes pass than current's). r6c already learned that fix, so a Mistral
  model on par with r6c would fail it and read as "worse". It is reported beside the verdict. For `r4-mx` it counts.
- **Error rows count as fails.** `bar.json` leaves a row out of both arms when either errored, and the eval
  tolerates up to max(3, n/5) error rows a set, so 3 of the 4 `out_of_scope_holdout` rows could vanish silently.
  For Mistral an error is the model's answer (`MinistralParser` refuses an unknown tool name or bad JSON → the turn
  is `model_unavailable`), as Stage 1 scores it. `report` recounts each bar set from `eval/{current,candidate}/
  rows.jsonl` (`errors.json`): Δ with errors as fails replaces the bar's Δ in the `sets` rule; the pooled b / c / p
  are recounted the same way; and an **`errors`** rule fails when the candidate's error rows per replicate exceed
  the current model's by more than the set's tol. Error counts per set and arm are in the report.
- **Hidden retries.** Each pair's `info.json` has `retries_500` per server. When MX drew more, a "better" or
  "on par" that the excess could explain (each excess retry rescuing one (row, replicate) pair: b + e, c − e) is
  **inconclusive**, with the counts, so the rows get looked at first.

| verdict | condition |
|---|---|
| **better** | every non-inferiority rule passes and pooled McNemar favours MX (c > b) at p < 0.05 |
| **on par** | every non-inferiority rule passes otherwise |
| **inconclusive: train seed 2** | pooled McNemar p in [0.05, 0.2), **or** only the `sets` rule fails and by at most 4 rows on every failing set |
| **inconclusive: retries** | a better / on par the excess 500s could explain |
| **worse** | anything else |

The spec's literal rule, "within 4 rows on any bar set", would fire on almost every run, because `real` has 14
rows and `out_of_scope_holdout` has 4. It is read here as "the verdict hinges on 4 rows or fewer". Seed 2 means
running Stage 2 again with `TRAIN_ARMS=base:2` under a new run id, about US$10, then Stage 3's two MX pairs again,
about US$12–30.

**The report** (`report.md` and `report.json`) contains:

- the model card;
- the Stage 1 gate;
- training: rows, tokens, updates, tokens/s, peak memory, losses, gen-check;
- the three bar tables;
- the control trust result;
- both verdicts;
- job-runtime cost estimates from `--jobs-json`;
- the error recount and the `/v1/messages` 500s per pair, and the r6c-mx trigger rule beside its verdict;
- the caveats: one seed, no warm start, temperature 0.3, prime off, Vertex L4 instead of Cloud Run, a recipe that
  was not re-tuned, Qwen's `top_k 20` / `top_p 0.95` applied to Mistral, Qwen's parser coercing argument types from
  the schema (the string "5" to 5) where Ministral's does not, and the hidden-retry asymmetry.

## 6. Costs

These are estimates for us-central1. The billing report is the truth.

| step | machine | wall | US$ |
|---|---|---|---|
| Cloud Build: 2 images | E2_HIGHCPU_32 (about $0.064/min) / _8 | 30–60 min | 2–6 (more if the trainer layer cache misses) |
| Prep: base-gguf | n2-highmem-16, about $1.0/h | 45–75 min | 1–2 |
| Stage 1: 3 exams | 3 × g2-standard-12 + L4, about $1.0–1.8/h | 2–3.5 h | 7–14 |
| Stage 2: smoke + full | a2-highgpu-1g A100-40 **Spot**, about $1.55/h | 0.5 h + 5–8 h | 9–14 (on demand $22–35; H100 $20–28, after r6 only) |
| Stage 3: 3 evals | 3 × g2-standard-24 + 2 L4, about $2.3–3.6/h | 2.5–4 h | 18–45 |
| Storage | about 60 GB in `experiments/mistral/` plus about 20 GB of images in Artifact Registry | per month | about 3 |
| Optional: mx-base-native | 1 × g2-standard-12 + L4 | about 1 h | about 2 |
| **Total** (the optional native run excluded) | | | **about 40–85**, most likely 45–60 (+ about 10 + 12–30 if seed 2 is needed) |

`report --jobs-json` turns each job's runtime into an estimate. The rates are the constants in `mxarm.py`
(`MACHINES`, `RUNGS`).

## 7. Cleanup

A person does this after the decision, and **by the delete-by date in §0** (14 days after the decision, never
later than 2027-04-14). No lifecycle rule covers `experiments/`, so nothing expires by itself.

```bash
# 1. nothing of ours still running
for r in us-central1 us-east4 us-west1 us-east1; do gcloud ai custom-jobs list --project $P --region $r \
  --filter='labels.purpose=sous-chef-mx AND state:(JOB_STATE_PENDING OR JOB_STATE_QUEUED OR JOB_STATE_RUNNING)' \
  --format='value(name,displayName,state)'; done          # cancel any with: gcloud ai custom-jobs cancel <name> --region <r>
# 2. save the small results elsewhere first (report/, gate.json, exam and eval summaries), e.g.
gcloud storage cp -r $EXP/runs/$MX/report $EXP/runs/$MX/gate.json /home/gtong/mx-work/keep/
# 3. delete everything under the experiment: weights, GGUFs, checkpoints, the smoke run ($MX-smoke: its dataset
#    copy, checkpoints, a ~5 GB GGUF and the adapter), both copies of r6's dataset, the r4 snapshot, any seed-2 run
gcloud storage ls $EXP/runs/                              # every run id: $MX, $MX-smoke, a seed-2 run …
gcloud storage rm -r $EXP
gcloud storage ls $EXP/ 2>&1 | grep -q 'matched no objects' && echo "experiments/mistral is empty"
# 4. the images (the packages, every tag and digest)
gcloud artifacts packages delete sous-chef-mx --repository=sous-chef --location=northamerica-northeast2 --project $P --quiet
gcloud artifacts packages delete sous-chef-trainer-mx --repository=sous-chef --location=northamerica-northeast2 --project $P --quiet
# 5. the account (remove its bucket bindings first; both are conditional)
gcloud storage buckets remove-iam-policy-binding $B --member serviceAccount:$MX_SA --role roles/storage.objectViewer --all
gcloud storage buckets remove-iam-policy-binding $B --member serviceAccount:$MX_SA --role roles/storage.objectAdmin --all
gcloud projects remove-iam-policy-binding $P --member serviceAccount:$MX_SA --role roles/logging.logWriter
gcloud iam service-accounts delete $MX_SA --project $P
# 6. local copies: the base tar and the r4 staging under $W, and any local copy of the dataset
rm -rf /home/gtong/mx-work/*.tar* /home/gtong/mx-work/r4-stage
# 7. the worktree (the main session decides whether the branch is kept)
git -C /home/gtong/projects/breadstack-sous-chef worktree remove /home/gtong/projects/sous-chef-mistral
```

If the arm stops at Stage 1, step 3 still applies (the base tar, the base GGUFs and the r4 snapshot are there).

## 8. Findings that changed the spec

Each was checked in the Ollama v0.34.4 source in the Go module cache.

1. **"Ollama's own converter" is not a fallback in 0.34.4.**
   - `ollama create` from safetensors stages an MLX-runtime model (`server/create.go` `createSafetensorsModel`
     → `create.Create`, `mlxrunner.ErrRuntimeUnavailable`). It does not write a GGUF that llama-server could
     serve on CUDA, and `--quantize` is refused for GGUF input.
   - The fallback is therefore `base-gguf --converter-ref b11081`. That is upstream llama.cpp's
     `convert_hf_to_gguf.py` at the build Ollama 0.34.4 bundles (`LLAMA_CPP_VERSION` b11081), with the image's
     `llama-quantize`.
2. **The load risk is lower than the spec assumed.**
   - `llama/compat/llama-ollama-compat.cpp` `handle_mistral3` treats a llama.cpp-converted `mistral3` file, with
     the projector in a separate mmproj, as the native format. It only rewrites Ollama-published monolithic
     files, which have `v.*`/`mm.*` tensors or the `rope.scaling_beta`/`mscale_all_dim` names.
   - `llm/llama_server.go` lists `mistral3` among the clip-capable architectures.
   - The first minutes of Stage 1 still prove the load.
3. **Ollama 0.34.4 never logs the rendered prompt.** At debug level it logs only `prompt_len`
   (`llm/llama_server.go:1624`).
   - The spec's assert "the first rendered prompt starts with `[SYSTEM_PROMPT]`" is replaced by the
     `/api/show` template-sha, parser and renderer equality and the prompt-size smoke in §5.1.
   - The bytes that template renders are proven offline by the parity fixture (`served-ministral.json`).
   - `OLLAMA_DEBUG_LOG_REQUESTS` is unset in every server, so request bodies never go to disk.
4. **Parser errors never reach Ollama's log.** `MinistralParser` errors go back in the HTTP response
   (`routes.go:2941`). They are counted from the Ollama access log (non-200 on `/v1/messages`), from the
   service's model-error lines, and from rows that errored.
5. **A Hugging Face download by commit writes no `refs/main`.** The trainer's layout check and
   `cached_snapshot()` need it, so §3.2 writes it.
6. **The trainer's paths.** Setting `PIPELINE_BUCKET` to the experiment prefix replaces the spec's
   `entrypoint.sh` and `selftest.py` edits (§0, §1.1.3).
7. **Review fixes (2026-10-16), all in `mxarm.py` unless named.**
   - The `r6c-mx` verdict no longer counts bar.py's trigger rule (§5.3), and Stage 3 recounts error rows as fails
     with an `errors` rule.
   - Parser-error 500s are retried by the service (hidden extra draws); counted per arm in Stage 1 and Stage 3.
   - r4's state is snapshotted (`snapshot-r4`), so an r6 rollout cannot swap the r4 pairs' baseline.
   - Runner bodies set `disableRetries`, and each runner refuses a restart of its own attempt (§0).
   - `gates --stage 2` checks the dataset at `dataset/dataset/train.jsonl` (the previous `dataset/train.jsonl`
     never existed in a round's layout), takes `--rung`, and accepts a JUDGE only with `user-go.json`.
   - `sous-chef-mx` is tagged by content (`images/mx/tag.sh`), not by the commit.
   - `train.py`'s Mistral mask checks every row (one `<s>`, as many token spans and `</s>` as assistant segments),
     not only row 0, and the dry run stops on rows whose control-token strings move their token spans (the mask
     would stop on them, or train other text than their replies; `train.mask_problem`, which the pipeline's dataset
     gate runs on CPU) and warns about the rest that hold one as text.
   - The tool smoke runs at temperature 0, blocks on call markers in text, and for Mistral needs a tool call.
8. **Placement.** The runner is `lab/tools/mxarm.py`, not `lab/souschef_lab/mxarm.py`. The trainer build file is
   `images/mx/cloudbuild-trainer.yaml`, not `images/trainer/cloudbuild-mx.yaml`, which keeps `images/trainer/`
   byte-identical to r6's.

## 9. Tests

Run from `lab/` with the setup report's environment:

```bash
S=/tmp/claude-1000/-home-gtong-projects-breadstack-cookie-cutter/e7ae1b42-660b-4619-a1b1-b6d81ba37cab/scratchpad
# the r6tools engine snapshot's node_modules is a dangling symlink now: a broken engine scores LOWER rather than
# erroring (test_exam_dry_end_to_end: 2.0 >= 4), so use the r7 snapshot
PYTHONDONTWRITEBYTECODE=1 MX_ENGINE_DIR=$S/r7/cc-snapshot-1d4193e2 COOKIE_CUTTER_DIR=$S/r7/cc-snapshot-1d4193e2 \
SOUS_CHEF_BIN=$S/r6tools/bin/sous-chef SOUS_CHEF_FIELDS_BIN=$S/r6tools/bin/sous-chef-fields SOUS_CHEF_PREFIX_BIN=$S/r6tools/bin/sous-chef-prefix \
  /home/gtong/projects/breadstack-sous-chef/lab/.venv/bin/python -m pytest -rfEs -p no:cacheprovider tools/mxtests
```

The suite covers:

- **The write guard.** 16 refused destinations, plus a check that a real gcloud write is guarded before gcloud
  runs.
- **The GGUF check.** Every failure mode, the Ollama key names, gguf-py-compatible bytes, and the real r4 GGUF.
- **`staged_modelfile`.** It is compared with the real `make model-stage` output for both families. The Mistral
  TEMPLATE survives byte for byte, with no trailing newline.
- **Serving asserts against a fake `ollama`.** These cover ok, CPU fallback, template drift, a 404, a parser 500
  that is strict or not, call markers leaked as text, a Mistral text-only reply (blocked) against a Qwen one
  (recorded), a wrong family or parser, and a failed create.
- **Stage 1 gate boundaries.** GO at exactly −13, the noise-widened GO, STOP versus JUDGE through Δ_fmt, the
  default lean, BLOCKED, MISSING and INVALID (one replicate), hidden retries turning GO into JUDGE, STOP waiting
  for and then turned by the native diagnostic, and the r4 anchor.
- **Trust and verdict tables**, the r6c-mx trigger rule reported but not counted (counted for r4-mx), error rows
  recounted as fails, and excess retries making a win inconclusive.
- **`mirror_store`.** For all three pairs the pipeline bucket's tree hash is unchanged, and r6c is patched only
  in the copy. The r4 pairs refuse a moved live model without a snapshot and use the snapshot with one;
  `snapshot-r4` refuses once the live model is not r4 and is idempotent.
- **Attempts.** A restarted runner exits 1 without writing; readers follow `latest.json`.
- **Job bodies.** `souschef-mx`, the labels, the train environment under `experiments/mistral`, the refused
  r6-competing rungs, `disableRetries` and `MX_CREATE` on every runner body (not the trainer's), and no subprocess
  calls.
- **`gates`.** Only read-only gcloud calls are made; the stage 2 rung check, and a JUDGE accepted only with a
  `user-go.json` for that `gate.json`.
- **`stage-native` and `native_modelfile`.**
- **`base-gguf` end to end** with a fake converter and quantizer. Every write stays under the experiment, and a
  failed check stages nothing.
- **`exam --dry` end to end** on the lab's fake service plus the real engine. 6 rows × 2 replicates; every write
  stays under the experiment.
- **The build files.** They push only the MX names, pin their bases by digest, and leave the existing images
  untouched.
- **r7** (`test_mxarm_r7.py`): the round table and its pairs; `mirror_store` for r7's four pairs with r6c live
  (state and baseline untouched), r7c patched, r4 from the snapshot, r4's recorded reference and r6's current rows
  mirrored and loadable by `eval_stage.load_reference`; a previous candidate no longer live keeping its approved
  baseline; the control's distance table; `r4_floor` in the verdict; the r7 report (labels, report-only r4-mxt, the
  latency section, 500s recounted from the logs); `prep-run --from-run`; the stage 3 gates; the r4-mxt route
  temperature; the GIN fix.
- **Latency** (`test_mx_latency.py`): Go durations, the smokes left out, timing blocks attached to their request,
  per-turn groups, the markdown.

## 10. The r7 arm

**Why a second run.** r6's arm beat r6c on r6's data, but its control pair was untrusted (F9H Δ +33.0 here vs +36.5
in r6, tol 2). r7 (`rt-20261018-0213-b02e95-3914e2e4`) changed the lab: `askrule.py` (ask only when vague), the
`.k3` answer keys (h006, h022, p030, p047), the `r4_floor` bar rule (the candidate against r4's RECORDED
calibration replicates, thresholds `r4_reference`), and its thresholds. MX is retrained on r7's dataset and compared
under r7's scoring in a new run, `mx-20261018-m3-8b-r7`; the r6 run stays as it is and its report still renders
(`report --run mx-20261017-m3-8b`: the round comes from its `plan.json`).

**Rounds** (`ROUNDS` in `mxarm.py`): each round has a label (its candidate is `<label>c`) and the model that was
current in its own eval. r6's round ran against r4; r7's runs against **r6c** (approved 2026-10-17, live).

| pair | current | candidate | role |
|---|---|---|---|
| `r7c-mx` | r7c | MX | the verdict vs r7's candidate (the trigger reported, not counted) |
| `r4-mx` | r4 (snapshot) | MX | the verdict vs r4 (the trigger counts) |
| `r6c-r7c` | r6c (live) | r7c | the **control**: re-runs r7's own eval |
| `r4-mxt` | r4 | MX at 0.15 | **report only**: Mistral's temperature on MX's route, 0.3 on r4's |

**What changed in the runner.**

- Every command takes `--cases-run`; without it the run's `plan.json` decides, else r7. A run is tied to one round
  (`prep-run` refuses another).
- `prep-run --from-run mx-20261017-m3-8b` carries Stage 1 (`gate.json` marked `carried_from`; the untrained bases
  are the same) and `models/r4/`. No exam is re-run: Stage 3 re-scores every arm on today's keys.
- `mirror_store`: the round's current model (r6c) is used exactly as r7's eval read it when it is live (state and
  `baselines/<digest>/baseline.json` untouched); if r7 is approved first, r6c is patched in, with its approved
  baseline. r4's recorded reference (every `r4_reference.baseline` path the bucket has; eval_stage takes the copy
  whose sha256 is `496a5ca0…`) and `runs/<rows_run>/eval/current/rows.jsonl` are mirrored, so `r4_floor`,
  `r4_rerun_floor` and the clear-asks count are measured in every pair as in r7's eval.
- **Baseline pin** (`pin_current_baseline`): the current model's baseline sets every bar set's tolerance
  (`bar.set_row` → `sigma_run`). `souschef-drift-weekly` (Mondays 05:00Z, `weekly` mode) appends an arm-T replicate
  to the LIVE model's baseline once the `gpu-eval` lock is free, i.e. after r7's eval. So the local copy drops every
  replicate whose `at` is after r7's `report.json` `at` (then `baseline.recompute`), whichever current it is, and
  logs the kept `at`s, the dropped ones and the copy's sha256 (`info.json` → `mirror.baseline_pin`). Stage 3's gate
  checks that `report.json` has an `at`; without one the eval refuses.
- The verdict's non-inferiority rules include `r4_floor` (NOT MEASURED passes, as in the bar).
- **Trust** is unchanged as a rule (§5.3, plus: the control's current digest must be the round's
  `report.json` `results.current.digest`, and every counted set's `tol` here must equal r7's — a different tol
  means a different current baseline). The report now shows every set's distance from the round's own
  numbers: Δ there, Δ here, the gap, the tolerance (the round bar's own `tol_s`), the slack, each arm's score
  moved, and 2σ of the gap from both evals' discordant pairs (information only: for r6's F9H it is 13.7 rows
  against a tolerance of 2).
- **Latency** (`tools/mx_latency.py`): per call from each Ollama log (llama-server's prompt-eval + decode time,
  Ollama's request wall time, decode and cold-prefill tokens/s, tokens out) and per turn from `rows.jsonl`
  (latency, calls per turn). Every eval attempt writes `latency.json`; `report` adds the section (recomputed
  from the logs for older attempts) and `mxarm.py latency --run …` prints it.
  Token counts are per tokenizer: on r6's transcripts MX wrote about as many CHARACTERS a call as Qwen (~150)
  in ~32 Tekken tokens against ~57 Qwen tokens. Compare decode time per call (MX ≈ 1.0 s, Qwen ≈ 1.5 s), not
  tokens; 2 workers share 1 slot, so production ratios are an assumption.
- **GIN fix**: the service's SDK calls `/v1/messages?beta=true`; the access-log pattern now allows the query.
  Until now every `retries_500` read 0 (r6's r4-mx candidate drew 3).

**Commands** (the main session; `WT=/home/gtong/projects/sous-chef-mistral-r7`). Each step chains its gate with
`&&`: a failed gate stops the submit. `jobs --stage train --rung h100-*` also refuses itself until
`runs/$RT/train/DONE` exists.

```bash
WT=/home/gtong/projects/sous-chef-mistral-r7; cd $WT
MX=mx-20261018-m3-8b-r7; RT=rt-20261018-0213-b02e95-3914e2e4; P=breadstack-sous-chef-int; REG=us-central1
AR=northamerica-northeast2-docker.pkg.dev/$P/sous-chef; J=/home/gtong/mx-work/jobs/r7; mkdir -p $J
# functions, not strings: zsh does not split "$MXPY"
mxpy() { env PYTHONPATH=$WT/lab /home/gtong/projects/breadstack-sous-chef/lab/.venv/bin/python $WT/lab/tools/mxarm.py "$@"; }
API=https://$REG-aiplatform.googleapis.com/v1/projects/$P/locations/$REG/customJobs
submit() { curl -sS -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json" \
  "$API" -d @"$1" | tee "${1%.json}.created.json" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("name"), d.get("state"), d.get("error"))'; }
# 1. images from this tree. Tool caches would enter the tags and the context (the cloud builds refuse them)
find lab images \( -name .ruff_cache -o -name .pytest_cache \) -type d -prune -exec rm -rf {} +
gcloud builds submit --project prod-devops-core-project --config images/mx/cloudbuild-trainer.yaml \
  --ignore-file=images/images.gcloudignore . &&
gcloud builds submit --project prod-devops-core-project --config images/mx/cloudbuild.yaml \
  --ignore-file=images/images.gcloudignore --substitutions=_TAG=$(bash images/mx/tag.sh) .
TRAINER_MX=$(gcloud artifacts docker images describe $AR/sous-chef-trainer-mx:$(bash images/trainer/tag.sh) --format='value(image_summary.fully_qualified_digest)')
MXIMG=$(gcloud artifacts docker images describe $AR/sous-chef-mx:$(bash images/mx/tag.sh) --format='value(image_summary.fully_qualified_digest)')
echo "$TRAINER_MX"; echo "$MXIMG"   # both …@sha256:<64 hex>, else stop
# 2. the run (r7's result-dataset.json is ok since 2026-10-18 03:30Z); carries Stage 1 and models/r4 from the r6 run
mxpy prep-run --run $MX --cases-run $RT --from-run mx-20261017-m3-8b --trainer-image $TRAINER_MX --mx-image $MXIMG
# 3. train on H100 Spot once r7's train/DONE exists (the gate and `jobs` both fail until then)
mxpy gates --run $MX --cases-run $RT --stage 2 --rung h100-spot &&
mxpy jobs --run $MX --cases-run $RT --stage train --mode full --rung h100-spot --after-round-train-done \
  --trainer-image $TRAINER_MX --out-dir $J &&
submit $J/souschef-mx-train-$MX-h100-spot-1.json      # Spot preempted: the same chain with --attempt N+1
# 4. after r7's eval (eval/bar.json, report.json with its `at`) and MX's train/DONE
mxpy gates --run $MX --cases-run $RT --stage 3 --pair r7c-mx,r4-mx,r6c-r7c,r4-mxt &&
mxpy jobs --run $MX --cases-run $RT --stage eval --pair r7c-mx,r4-mx,r6c-r7c,r4-mxt --mx-image $MXIMG --out-dir $J &&
for p in r7c-mx r4-mx r6c-r7c r4-mxt; do submit $J/souschef-mx-eval-$MX-$p-1.json; done
# 5. the report (after all four eval jobs end)
gcloud ai custom-jobs list --project $P --region $REG --filter='labels.purpose=sous-chef-mx' --format=json > $J/mx-jobs.json &&
mxpy report --run $MX --jobs-json $J/mx-jobs.json
```


## The name, and the control's noise bound (2026-10-18)

- **chocolate-muffin** is this arm's model: Ministral 3 8B Instruct (`mistralai/Ministral-3-8B-Instruct-2512-BF16`,
  Apache-2.0) plus a LoRA on the round's data — the Mistral counterpart of blueberry-muffin (Qwen3.5-9B). Reports
  carry the name (`MODEL_NICKNAME` in `lab/tools/mxarm.py`); job, folder and arm ids keep `mx`.
- **Control trust uses a noise bound**, chosen by the user and fixed in code before r7's control ran: a counted set's
  |Δ here − Δ in the round| must be ≤ max(the set's tol in the round's bar.json, 2σ of the gap between the two
  independent evals) (`NOISE_SIGMAS`). A differing bar PASS/FAIL then counts only when some counted set is outside its
  bound; otherwise it is reported as a note. The strict tol alone had marked r6's control untrusted over a 3.5-row
  F9H gap whose 2σ was about 14 rows.
