# The Sous Chef drift check and retraining pipeline: turning it on and living with it

This is the top-level runbook. It says what the suite does, which switches
control it, and the order in which a person turns it on. The two stage
runbooks hold the detail:

- [`docs/DRIFT.md`](DRIFT.md): how drift is measured, the verdict rules, and
  how to run a check by hand.
- [`docs/RETRAIN.md`](RETRAIN.md): every retrain stage, its automated path,
  its manual path (the playbook), and how to debug a failed run.

Scope: **int only** (project `breadstack-sous-chef-int`). Sous Chef runs only
on int, so prod has no pipeline. The thermometer side (notifications, the
decision page, Slack) lives in `breadstack-thermometer`; its README section
"Sous Chef pipeline" documents that half.

Status (2026-10-11): written with the build and revised after its three
reviews (approval integrity, statistics and state, operations). Nothing below
has been applied or run against GCP. Commands are the shapes the build ships;
every step marked *first run* proves something that is not yet verified.

---

## 1. What it does, in one picture

```
 launcher on int ──registers manifest──▶ sous-chef (int) ──GET /v1/manifests/deployed──┐
                                                                                      │
 Cloud Scheduler ──*/15, weekly, night──▶ workflow souschef-drift ──job souschef-drift┘
                                              │   static diff (2 min) → behavioural exam on
                                              │   sous-chef-model-eval (1–3 h) → verdict
                                              │
                         ok / notice ─────────┼──▶ thermometer ──▶ Slack (weekly-digest channel)
                                              │
              retrain (big enough) + switches ▼
                                   workflow souschef-retrain
   prepare (rebase every family, author new-field cases with Claude) → collect (int's Claude routes)
   → dataset (+ parity gate) → train + export (Vertex, H100) → model image (Cloud Build)
   → candidate on sous-chef-model-cand → paired eval vs current (sous-chef-model-eval)
   → candidate_ready ──▶ thermometer decision page + Slack @-mention
                              │
            Approve ──────────┴────────── Skip / 14 days pass
   candidate image → sous-chef-model         candidate parked; its authored cases stay in the
   baseline reset, rolled_out posted         pool and the next big drift's round batches them
```

- **Timeliness.** A launcher manifest change is seen within 15 minutes; the
  static report follows about 2 minutes later; the behavioural verdict 1.5–3 h
  after a 30-minute debounce. A retrain takes about 9–12 h from the drift to
  the decision message, plus any wait for Vertex capacity. Cost is reported,
  never a brake (user decision, 2026-10-11).
- **Who decides.** Only a person: the pipeline never rolls a model out without
  an Approve on the thermometer by an allowlisted approver (default
  `gary.tong@breadstack.com`).
- **What merchants feel.** Nothing until an Approve. Exams and paired evals
  run on separate GPU services (`sous-chef-model-eval`, `sous-chef-model-cand`)
  and never on the production slot, except the night fallback in
  [DRIFT.md §5](DRIFT.md#5-where-the-exam-runs).

## 2. Moving parts

| part | where | what |
|---|---|---|
| `GET /v1/manifests/deployed` | the sous-chef service (README §1) | which manifest the int launcher really runs: launcher registrations (table `manifest_registrations`, migration 0006) joined with the newest real (non-lab) turn |
| workflow `souschef-drift` | Cloud Workflows, `us-central1` | poll / check / weekly / calibrate; posts notifications; starts a retrain when allowed |
| workflow `souschef-retrain` | Cloud Workflows, `us-central1` | the stages of [RETRAIN.md](RETRAIN.md); waits for the decision; rolls out or parks |
| Cloud Run jobs `souschef-drift`, `souschef-prepare`, `souschef-collect`, `souschef-dataset`, `souschef-eval` | `northamerica-northeast2` | one lab runtime image `sous-chef-lab:<service tag>`: the lab, Python, node, Postgres 16 and the deployed service binaries |
| Vertex custom job | `us-central1` (Spot ladder also `us-east4`, `us-west1`) | the trainer image `sous-chef-trainer:t-<hash>`: SFT, GGUF export, staging |
| Cloud Build | int project | the model image `model:c-<run_id>` from the staged GGUF |
| Cloud Run GPU services | `us-east4` | `sous-chef-model` (production, min = max = 1), `sous-chef-model-eval` and `sous-chef-model-cand` (min 0, max 1) |
| bucket `gs://breadstack-sous-chef-int-pipeline` | `US-CENTRAL1` | all state, baselines, the case pool, the engine snapshots, every run's artifacts ([RETRAIN.md §R8](RETRAIN.md#r8-debugging-a-failed-run)) |
| thermometer | prod, `thermometer.breadstackcrm.com` | stores every notification, posts to the **weekly digest** webhook's channel, hosts `/sous-chef/pipeline` and the decision page |

Slack goes to the channel of the thermometer's weekly digest webhook (secret
`thermometer-slack-digest-webhook-url`). Decision and failure messages start
with `<@U07AFF50XL0>` so the user is notified; both the mention and the kinds
that carry it are thermometer settings (§6).

## 3. Switches

Every switch defaults to off. A plan with all of them off creates nothing.

**sous-chef, `infra/envs/int/terraform.tfvars`:**

| switch | on means |
|---|---|
| `pipeline_enabled` | APIs (`aiplatform`, `workflows`, `workflowexecutions`, `cloudbuild`), the bucket, the service accounts and their IAM, the two pipeline secrets (empty), the five jobs, both workflows, the eval and candidate GPU services (min 0, cost nothing idle), the Artifact Registry cleanup policy. **No schedule.** Everything can be run by hand. |
| `pipeline_schedule` | Cloud Scheduler jobs `souschef-drift-poll` (`*/15 * * * *`), `souschef-drift-weekly` (`0 5 * * 1`) and `souschef-drift-night` (`0 9 * * *`). Times are UTC: Monday 05:00 UTC is Sunday 22:00 PDT (21:00 PST); 09:00 UTC is 02:00 PDT (01:00 PST). |
| `pipeline_retrain_auto` | a `retrain` verdict starts `souschef-retrain` by itself. Off = notify only. It also needs a calibrated baseline for the live model ([DRIFT.md §4](DRIFT.md#4-baselines-noise-and-calibration)). |
| `pipeline_rollout` | an Approve patches `sous-chef-model`. Off = every Approve is a dry rollout: a permission check and a `rolled_out` post marked dry run, production untouched. |
| `pipeline_alert_emails` | **required (at least one) while `pipeline_enabled`**: Cloud Monitoring emails these addresses when a workflow execution fails and when a notification does not reach the thermometer (the log line "not delivered"). It is the one alert path that does not go through the thermometer and Slack. |

**Thermometer, `infra/env/prod.tfvars` in `breadstack-thermometer`:**

| switch | default | meaning |
|---|---|---|
| `souschef_pipeline` | `false` | registers the notification, decision and page routes; off = all of them 404 |
| `souschef_approvers` | `["gary.tong@breadstack.com"]` | who may Approve or Skip |
| `souschef_slack_mention` | `"<@U07AFF50XL0>"` | the mention put on decision and failure posts |
| `souschef_heartbeat_hours` | `36` | the dead man's switch ([§7](#7-day-two)) |
| `souschef_ingest_key_ids` | `[]` | **required while `souschef_pipeline` is on**: the id of the pipeline's Ingest key (shown as `#<id>` on the API keys page). The routes admit that key by id only — names are not unique and any employee can mint a key under any name. |

Slack posting also needs the thermometer's `slack_digest = true`, already on
in prod.

## 4. Turning it on (first time, in order)

Each step says who runs it. **A person** runs every `terraform apply`, every
secret write, every deploy and every command that spends money. Commands
assume:

```bash
export PROJECT=breadstack-sous-chef-int
export REGION=northamerica-northeast2     # the sous-chef service, the lab image, the pipeline jobs
export WF_REGION=us-central1              # the workflows, Vertex, the pipeline bucket
export GPU_REGION=us-east4                # sous-chef-model, -eval, -cand
export PIPE=gs://breadstack-sous-chef-int-pipeline
export TRAIN=gs://breadstack-sous-chef-int-training
gcloud config set account gary.tong@breadstack.com
```

Executions are started with `gcloud workflows execute` (or the root Makefile's
`pipeline-*` targets): it returns the execution name at once. Never
`gcloud workflows run` for a retrain: it blocks until the execution ends,
which is after the decision (up to 15 days).

### 4.1 Quotas (a person; start first, they take a day or more)

| quota | region | today (*measured* 2026-10-10) | ask for | why |
|---|---|---|---|---|
| Cloud Run `NvidiaL4GpuAllocNoZonalRedundancyPerProjectRegion` | us-east4 | 3 | 5 | production + eval + candidate is 3; a revision swap during a rollout needs a 4th instance. With 3 the rollout waits for the other two to scale to zero first ([RETRAIN.md §R7](RETRAIN.md#r7-decide-roll-out-roll-back-skip)). |
| Vertex `custom_model_training_nvidia_a100_80gb_gpus` and its preemptible twin | us-central1 | 0 | 1 each | rung 3 of the training ladder ([RETRAIN.md §R4](RETRAIN.md#r4-training)). |

The H100 needs nothing: 64 preemptible H100 per region are already granted in
us-central1, us-east4 and us-west1. Nothing waits for these requests; the
ladder skips a rung it cannot get.

### 4.2 Thermometer (a person; two deploys, before anything posts)

In `breadstack-thermometer` after the pipeline change is merged:

1. `make gcp-deploy ENV=prod` with `souschef_pipeline` still off. It runs
   migration 0045 (`souschef_notifications`, `souschef_decisions`) and ships
   the API keys page that shows each key's id.
2. On the **API Keys** page mint an **Ingest**-scope key named
   `souschef-pipeline`. Copy the key once (it goes into Secret Manager in
   §4.4) and note its id, shown as `#<id>` beside the name.
3. In `infra/env/prod.tfvars`: `souschef_pipeline = true` and
   `souschef_ingest_key_ids = [<id>]` (the other settings keep their defaults
   unless you want more approvers), then `make gcp-deploy ENV=prod` again.
   A plan with the pipeline on and no key id is refused.
4. Open `https://thermometer.breadstackcrm.com/sous-chef/pipeline`: an empty
   feed, no open decision. The dead man's switch is not armed yet: it arms on
   the first heartbeat (§4.7).

### 4.3 The sous-chef build (by hand: int has no CI trigger)

`cloudbuild.yaml` describes CI, but its trigger was never created in
`prod-devops-core-project` (checked 2026-10-11: only the pantry and wpengine
triggers exist), and the live int build `f79753b-speed-a` was built and
rolled by hand. Int also runs with `MIGRATE_AT_BOOT=off`, so migration 0006
goes through the `sous-chef-migrate` job. Until a trigger exists:

1. Confirm the service source behind the live int build is what `main`
   carries in `service/`: the r4 baseline and the render-parity gate assume
   the prompt that build serves. `f79753b-speed-a` is f79753b plus the speed
   changes later committed as `30d1fb5` (built by hand from that working
   tree), so compare against `30d1fb5`, not f79753b (against f79753b the diff
   is 30d1fb5's 33 files as well). After the commit in step 2,
   `git diff --stat 30d1fb5 HEAD -- service/` should list only this change:
   `internal/api/{deployed,ports,server}.go`, `internal/registry/`,
   `internal/store/{registrations,retention}.go`, migration 0006 and their
   tests. To confirm the live build itself renders what this checkout's lab
   renders, run the lab's parity check against the binary inside the live
   image (it needs node, the cookie-cutter snapshot, the lab Postgres and the
   fields and prefix CLIs: `lab/tests/r4_parity.py`, [LAB.md](LAB.md)):

```bash
PROJECT=breadstack-sous-chef-int REGION=northamerica-northeast2
AR=$REGION-docker.pkg.dev/$PROJECT/sous-chef
LIVE=$(gcloud run services describe sous-chef --project $PROJECT --region $REGION \
  --format='value(spec.template.spec.containers[0].image)')            # …/sous-chef:f79753b-speed-a
docker pull "$LIVE" && C=$(docker create "$LIVE") && docker cp "$C:/sous-chef" /tmp/sous-chef-live && docker rm "$C"
(cd lab && .venv/bin/python -m tests.r4_parity --binary /tmp/sous-chef-live)   # exit 0: every model call byte-equal
```

2. Test, commit the pipeline change, then build and push the service image
   (the tag is the short sha):

```bash
make test                                                     # go vet + go test -race (what CI runs first)
T=$(git rev-parse --short HEAD)
make docker-build BUILD_ID=$T && make docker-push BUILD_ID=$T AR_REPO=$AR
```

3. The pipeline's images for the same tag, built in Cloud Build from this
   checkout, **before** the roll: the drift workflow resolves the lab and
   trainer images from the int service's tag, so a round that starts after
   the roll and before these exist fails at `images`. The trainer is about
   12 GB, up to 2 h the first time (the build's timeout is 3 h); the lab
   image about 20 min. Nothing is keyed on the roll, so build them first:

```bash
make lab-image TAG=$T && make trainer-image TAG=$T
```

4. Migration 0006 (`manifest_registrations`). Run it before the roll; the
   live build never reads the new table:

```bash
gcloud run jobs update sous-chef-migrate --image $AR/sous-chef:$T --project $PROJECT --region $REGION
gcloud run jobs execute sous-chef-migrate --wait --project $PROJECT --region $REGION
gcloud run jobs update sous-chef-retention --image $AR/sous-chef:$T --project $PROJECT --region $REGION
```

5. Roll the service: a tagged revision with no traffic, a look at its
   `/version`, then the switch (Terraform ignores the image, so none of this
   shows up as drift):

```bash
gcloud run deploy sous-chef --image $AR/sous-chef:$T --no-traffic --tag p-$T --project $PROJECT --region $REGION
TURL=$(gcloud run services describe sous-chef --project $PROJECT --region $REGION --format=json \
  | jq -r --arg t "p-$T" '.status.traffic[] | select(.tag == $t) | .url')
curl -s "$TURL/version" | jq -r .build_id                       # must print $T
gcloud run services update-traffic sous-chef --to-latest --project $PROJECT --region $REGION
```

If the CI trigger is created later, merging to `main` does steps 2 to 5 on
its own (migrate job, service, retention job, lab and trainer images).

Check:

```bash
T=$(gcloud run services describe sous-chef --project $PROJECT --region $REGION \
  --format='value(spec.template.spec.containers[0].image)' | sed 's/.*://')   # the deployed service tag
gcloud artifacts docker images list $REGION-docker.pkg.dev/$PROJECT/sous-chef/sous-chef-lab --include-tags | grep -- "$T"
gcloud artifacts docker images list $REGION-docker.pkg.dev/$PROJECT/sous-chef/sous-chef-trainer --include-tags | grep -- "$T"
```

A service built by hand has no lab or trainer image until a person runs
`make lab-image TAG=<T>` and `make trainer-image TAG=<T>` at the root (they
upload with `--ignore-file=images/images.gcloudignore`; the root
`.gcloudignore` ships `service/` only). A run refuses to start on a service
tag without its images (`stage_failed: lab image missing for <T>`, stage
`images`); it blocks nothing, and the next drift check starts the round once
the images exist.

### 4.4 The two secrets first (a person; targeted apply, then no-echo writes)

Cloud Run checks a job's secret references when it creates the job, so the
secrets get their values before the full apply:

```bash
cd infra/envs/int
IMG=$(gcloud run services describe sous-chef --project $PROJECT --region $REGION --format='value(spec.template.spec.containers[0].image)')
# terraform.tfvars: pipeline_enabled = true, pipeline_alert_emails = ["gary.tong@breadstack.com"] (the other switches stay false)
terraform apply -var=image=$IMG \
  -target='module.pipeline[0].google_project_service.pipeline' \
  -target='module.pipeline[0].google_secret_manager_secret.pipeline'
```

Workflows creates its service agent a few minutes after the API is switched
on, and a workflow created before the agent exists fails with `Workflows
service agent does not exist` (int, 2026-10-13: both workflows failed in
§4.5 and a second apply created them). Create the agent now, so §4.5 cannot
race it (idempotent; it prints
`service-<project number>@gcp-sa-workflows.iam.gserviceaccount.com`):

```bash
gcloud beta services identity create --service=workflows.googleapis.com --project $PROJECT
```

Then the values, **in a real terminal** with `PROJECT` set as in §4.3, not
through Claude Code's `!` prefix: each `!` line runs in a fresh shell (the
variables would be empty by the next line) and `read -s` needs a terminal.
(An agent's classifier blocks secret writes, so a person runs these.) Paste
each block whole; each runs in a subshell with `set -euo pipefail`, refuses to
write an empty key or key list, and prints nothing secret.

```bash
# 1. a pipeline-only service key, also appended to the service's own key list (run once)
(
set -euo pipefail
NEW=$(openssl rand -hex 32)
OLD=$(gcloud secrets versions access latest --secret sous-chef-service-keys --project "$PROJECT")
[ -n "$NEW" ] && [ -n "$OLD" ] || { echo "empty key or key list: nothing written" >&2; exit 1; }
case ",$OLD," in *",$NEW,"*) echo "the new key is already listed: nothing written" >&2; exit 1 ;; esac
printf %s "$NEW" | gcloud secrets versions add sous-chef-pipeline-service-key --project "$PROJECT" --data-file=-
printf %s "$OLD,$NEW" | gcloud secrets versions add sous-chef-service-keys --project "$PROJECT" --data-file=-
echo "done: sous-chef-service-keys holds $(printf %s "$OLD,$NEW" | tr ',' '\n' | grep -c .) keys"
)

# 2. the thermometer key from §4.2 (paste it at the prompt, Enter)
(
set -euo pipefail
read -rsp "thermometer key: " TM; echo
[ -n "$TM" ] || { echo "empty: nothing written" >&2; exit 1; }
printf %s "$TM" | gcloud secrets versions add sous-chef-pipeline-thermometer-key --project "$PROJECT" --data-file=-
echo "done"
)
```

Block 1 prints the number of keys now listed: the launcher's plus the
pipeline's (2 on a first run). A second run would append a second pipeline
key: revoke the extra one as in §4.5.

### 4.5 `pipeline_enabled` (a person)

```bash
NONCE=$(date -u +%Y%m%dT%H%M%SZ)       # keep it: every later apply in §4 passes the same one
terraform plan  -var=image=$IMG -var=deploy_nonce=$NONCE
terraform apply -var=image=$IMG -var=deploy_nonce=$NONCE
terraform output run_url               # the int service URL
terraform output pipeline              # bucket, eval_url, cand_url, jobs, workflows, service accounts, secrets
```

The `deploy_nonce` rolls a sous-chef revision so it reads the appended
`SERVICE_KEYS`. **Every later apply** (§4.7's schedule, §4.10's switches)
passes the same `-var=deploy_nonce=$NONCE`: without it `DEPLOY_NONCE` goes
back to `""` and int rolls a new revision each time (harmless, but a needless
roll; `gcloud run services describe sous-chef … --format=json | jq
'.spec.template.spec.containers[0].env'` shows the current one). Read the
plan before applying:

- **Creates** only pipeline resources (seven service accounts, the bucket and
  its conditional grants, the jobs, the workflows, the eval and candidate GPU
  services, the two monitoring policies), plus `extra_invokers` members on
  `sous-chef-model`'s authoritative invoker binding and the model
  repository's cleanup policy.
- **`sous-chef-model` shows no image change.** The model-gpu module ignores
  the image, scaling, annotations and client fields, so the pipeline (and
  `make model-rollout`) own the live image from here on.
- `alert_emails` empty with the pipeline on is refused (Terraform 1.9 or
  later: the check reads another variable).

After the apply, look at the project's default build and compute accounts.
The workflow always names `souschef-builder`, but
`roles/cloudbuild.builds.editor` alone could start a build as the project's
default Cloud Build account:

```bash
gcloud builds get-default-service-account --project $PROJECT
gcloud projects get-iam-policy $PROJECT --flatten=bindings --filter='bindings.role=roles/editor' --format='value(bindings.members)'
```

If the default is the legacy `<number>@cloudbuild.gserviceaccount.com` or the
Compute Engine default `<number>-compute@developer.gserviceaccount.com`, and
that account holds `roles/editor` (int's
`175800752268-compute@developer.gserviceaccount.com` did on 2026-10-11),
remove the Editor grant or set a narrow default build account.

Check the pipeline key works, without printing it (the key reaches curl on
stdin, never on a command line):

```bash
URL=$(terraform output -raw run_url)
gcloud secrets versions access latest --secret sous-chef-pipeline-service-key --project $PROJECT \
  | sed 's/^/Authorization: Bearer /' \
  | curl -s -o /dev/null -w '%{http_code}\n' -H @- "$URL/v1/manifests/deployed?line=int"   # 200
```

Two keys stay live in `SERVICE_KEYS`: the launcher's and the pipeline's. To
revoke the pipeline alone, write a version without it and roll again. Keep
`ingress = INGRESS_TRAFFIC_ALL` on the int service: the jobs call its
`run.app` URL directly.

### 4.6 Seed the bucket (a person or the main session; no spend)

[RETRAIN.md §R0](RETRAIN.md#r0-seeding-the-bucket) has the commands: the base
weights, the r4 case pool, the study folder for the hold-out filter, the r4
dataset, the r4 manifest `bf788701`, `state/model.json` for r4, and two engine
snapshots (r4's `e7e90106` and the one the int launcher was built from).

**Do this soon.** The r4 seed files are on one workstation only
(`lab/data/r4-seed/` and `lab/data/r4/`, git-ignored, copied out of the
session scratch on 2026-10-11), and the training bucket deletes `data/`,
`kit/` and `runs/` 30 days after upload (r4's copies around 2026-11-10).

### 4.7 Selftest, the first poll, the schedule (a person; no model calls)

```bash
# the jobs are created with a placeholder image (Google's sample job, which
# exits 0 whatever it is asked); point the drift job at this tag's lab image
# first, or the selftest checks nothing (each workflow run re-points it anyway)
T=$(gcloud run services describe sous-chef --project $PROJECT --region $REGION \
  --format='value(spec.template.spec.containers[0].image)' | sed 's/.*://')
gcloud run jobs update souschef-drift --image $REGION-docker.pkg.dev/$PROJECT/sous-chef/sous-chef-lab:$T \
  --project $PROJECT --region $REGION
gcloud run jobs execute souschef-drift --project $PROJECT --region $REGION --args=selftest --wait
gcloud workflows execute souschef-drift --project $PROJECT --location $WF_REGION --data='{"mode":"poll"}'
```

The selftest checks node, `pg_ctl`, the three service binaries, `gcloud` and
the `anthropic` import inside the lab image. The poll fetches the deployed
manifest, writes `state/deployed.json` and the static report, and posts a
`drift_notice` and the first heartbeat. **Expected on the first poll** (r4
trained on `bf788701`, the int launcher registers `ccd1afab`; measured with
the real `static_diff`): R1 105 wording changes, R2 94 soft-schema changes,
R3 10 hard changes (6 added fields — `stackedImageGap` and
`stackedImageHeight` on contentBlock, perksCards and perksIcons — and 4
retyped fields: `headingFontSize` on three pages and
`newsletter.ctaFontSize`), 0 stale exam rows, a probe on the 10 keys queued.
Auto-retrain is off and there is no baseline yet, so it only notifies.

**Then at once** `pipeline_schedule = true` + `terraform apply -var=image=$IMG -var=deploy_nonce=$NONCE`.
The first heartbeat armed the thermometer's dead man's switch; with the
schedule off for more than 36 h it would post `heartbeat_missed` (with the
mention) every day. With `pipeline_retrain_auto` and `pipeline_rollout` still
off the schedule only checks and notifies. The first poll leaves a debounced
check about 30 minutes later; with the schedule on, that check finds no
baseline for r4 and calibrates it first (about 3 h more on that run), so
§4.9 is usually already done by the time you reach it.

If the int launcher's commit has no engine snapshot, an `engine_stale` post
(with the mention) gives the exact `engine-vendor` command
([DRIFT.md §6](DRIFT.md#6-the-engine-snapshot)).

### 4.8 SMOKE (a person; 2–6 h with the GPU wait, about US$2 of API, 1 GPU-hour)

```bash
make pipeline-smoke        # = gcloud workflows execute souschef-retrain … --data='{"mode":"smoke"}'
```

SMOKE runs every edge with tiny inputs ([RETRAIN.md §R0](RETRAIN.md#smoke)):
Anthropic authoring, collection on int, the parity gate, the trainer's
`selftest` (*first run*: Vertex's driver against the cu130 wheels), the
training ladder (*first run*: whether Vertex accepts `a3-highgpu-1g` with
Flex-start; SMOKE caps the Flex-start wait at 30 minutes, then moves to Spot),
a resume, the GGUF export, Cloud Build, both GPU services, the bucket's
conditional grants (*first run*), the thermometer and the decision wait. Add
the trainer image pull (about 15 GB, cross-region) and the 18 GiB base tar to
the time. It ends with a `candidate_ready` post badged **SMOKE** with the
mention. Approve it on the thermometer: the rollout is dry (a permission check
and a `services.get`), and `rolled_out` is posted with `dry_run: true`.
Production is untouched.

What SMOKE does **not** prove: the behavioural drift check and its night
fallback, a drift run starting a round, the real patch of `sous-chef-model`,
a real Spot preemption, full-scale memory and timing, `teacher_changed`, the
ack and batching, expiry, dedup, `heartbeat_missed`, the monitoring policies
and the scheduler. The dry end-to-end run
(`infra/modules/pipeline/tests/e2e_dry.py`, phases 1–5) rehearses those
against fakes; the first real round and the first real Approve are their
first run.

### 4.9 Calibrate (a person; about 3 h on the eval L4, no API spend)

```bash
make pipeline-drift MODE=calibrate
```

Three replicates of the real-phrasing exam on the training manifest plus the
F0H holdout, all on the live model image through the deployed service build.
This is the noise baseline every verdict is measured against
([DRIFT.md §4](DRIFT.md#4-baselines-noise-and-calibration)). Until it exists,
auto-retrain stays off whatever the switch says. **Usually already done:** the
debounced check after §4.7's first poll calibrates first by itself when the
schedule is on (look for `baselines/<r4 digest>/baseline.json` with
`calibrated: true`: `gcloud storage ls $PIPE/baselines/`); run this only if it
is missing. Any later run that finds the live digest without a calibrated
baseline calibrates first too.

### 4.10 Auto-retrain, then real rollouts (a person; two applies)

Turn them on one at a time, each with `terraform apply -var=image=$IMG -var=deploy_nonce=$NONCE`
(the same nonce as §4.5, so int does not roll again):

1. `pipeline_retrain_auto = true`: a confirmed `retrain` verdict starts a round
   by itself (never while a failed round is waiting for a person, §7).
2. `pipeline_rollout = true`: an Approve now really rolls the candidate out.

## 5. The Slack messages and what to do

All posts go to the weekly digest channel. Links open the thermometer.

| kind | mention | means | what to do |
|---|---|---|---|
| `drift_notice` | no | drift worth knowing about but below the retrain bar (wording drift, new fields the model still handles, a 2σ dip) | nothing; read it when convenient |
| `drift_detected` / `retrain_started` | no | a confirmed big drift; a round started | nothing; the decision follows in about 10 h |
| `candidate_ready` | **yes** | a candidate was trained and evaluated against the live model; the table compares them set by set, with the bar verdict | open the decision page; **Approve** to roll it out to int now, or **Skip** to park it |
| `rolled_out` | no | the candidate is live on `sous-chef-model`; the drift baseline was reset | nothing; roll back with `make model-rollout TAG=<old>` if needed |
| `skipped` | no | parked by you, or the 14-day decision window ran out | nothing; its cases batch into the next round |
| `stage_failed` | **yes** | a stage failed after its retries; artifacts are kept. A failed **round** also blocks automatic rounds until it is resumed (the message ends with the resume command). A failing **drift poll** posts from its 2nd consecutive failure, at most once a day per failure streak; a check or weekly run posts at once. A stale retrain lock (no progress for 72 h) is reported once a day. | follow the playbook anchor in the message ([RETRAIN.md §R8](RETRAIN.md#r8-debugging-a-failed-run)) |
| `engine_stale` | **yes** | the int launcher's commit has no vendored engine; checks continue on the newest snapshot, retrains refuse to start | run the `engine-vendor` command in the message |
| `heartbeat_missed` | **yes** | no heartbeat for 36 h: the scheduler, a workflow or the project is broken | `gcloud workflows executions list souschef-drift --location us-central1 --limit 5` |
| `weekly_all_clear` | no | the Monday run found nothing (or the week's verdict when it did) | nothing |
| `drift_notice` "recovered" | no | a drift failure streak ended | nothing |
| `drift_notice` "production runs X, not Y" | no (warn) | the production model service serves an earlier model than `state/model.json` names (a manual rollback); drift judges the model actually serving | update `state/model.json` if the rollback is meant to stay ([DRIFT.md §4](DRIFT.md#4-baselines-noise-and-calibration)) |

**Approve or Skip.**
- **Approve** rolls the candidate image onto `sous-chef-model` **by digest** —
  the digest the decision page showed, checked against the immutable
  `model:c-<run>` tag and the build record — makes its paired eval rows the new
  drift baseline, keeps its training cases and dataset under
  `cases/trained/<candidate>/` (they outlive `runs/`), clears any earlier Skip,
  and posts `rolled_out`. When the bar failed, Approve needs the override
  checkbox. The workflow acts only on the decision row its own
  `candidate_ready` created (same notification, run and SMOKE flag); anything
  else is a `stage_failed` and nothing is rolled out.
- **Skip** keeps the live model. The candidate is parked
  (`state/parked/<candidate>.json`), and the cases authored for it stay in the
  pool. The drift check then only fires again on drift **beyond** what you
  skipped — a worse exam on the same rows, new dodged keys, stale rows added
  since, schema changes since — and the next round retrains on everything
  accumulated since. Doing nothing for 14 days (`decision_ttl_days`, at most
  21) counts as Skip.
- A decision page marked `stale_round` means the launcher changed again while
  the round ran; the page recommends Skip, and the next round covers both.

## 6. Changing who is notified or who decides

Thermometer tfvars (`infra/env/prod.tfvars`), then `make gcp-deploy ENV=prod`:
`souschef_slack_mention` (a `<@U…>` user or a `<!subteam^…>` group),
`souschef_approvers` (emails). The kinds that carry the mention are the
thermometer env `SOUSCHEF_MENTION_KINDS` (default
`candidate_ready,stage_failed,heartbeat_missed,engine_stale`).

## 7. Day two

| want | do |
|---|---|
| check now, whatever the schedule | `make pipeline-drift MODE=check` (= `gcloud workflows execute souschef-drift --location $WF_REGION --data='{"mode":"check","force":true}'`) |
| retrain now, whatever the verdict | `make pipeline-retrain` (= `gcloud workflows execute souschef-retrain --location $WF_REGION --data='{"mode":"full","trigger":{"reason":"manual"}}'`). A manual full round also clears a waiting `state/failed-round.json`. |
| a round failed | its `stage_failed` names the stage; `state/failed-round.json` holds the resume command, and no automatic round starts until it is resumed (`make pipeline-retrain RESUME=<stage> RUN=<run id>`) or the file is deleted (`gcloud storage rm $PIPE/state/failed-round.json`) ([RETRAIN.md §R8](RETRAIN.md#r8-debugging-a-failed-run)) |
| the thermometer was down | nothing: undelivered messages wait in `$PIPE/wf/outbox/` and the next poll re-sends them (the backstop email says so). `gcloud storage ls $PIPE/wf/outbox/` shows what waits. A 401, 403 or 404 (the key, `souschef_ingest_key_ids`, `souschef_pipeline` off) also waits there until fixed; only a refusal of the message itself (400, 409, 413, 422) is dropped. |
| see what is running | `gcloud workflows executions list souschef-retrain --location $WF_REGION --limit 5` (and `souschef-drift`) |
| notify only for a while | `pipeline_retrain_auto = false` + apply. Checks, heartbeats and posts continue. |
| stop the checks | `pipeline_schedule = false` + apply. The thermometer then posts `heartbeat_missed` (with the mention) once a day after 36 h; also set `souschef_pipeline = false` there if the stop is long. |
| roll back a model | `make model-rollout TAG=<previous tag or sha256:digest>` at the repo root (the round's `wf/runs/<id>/rollout.json` names it) ([RETRAIN.md §R7](RETRAIN.md#r7-decide-roll-out-roll-back-skip)). Drift then judges the model you rolled back to (from `state/model.json` history) and posts a notice. |
| a new service build (new prompt or retrieval) | nothing for the build itself (CI builds the lab image); run `{"mode":"calibrate"}` afterwards, because the baseline is per model image and the service build moves it ([DRIFT.md §4](DRIFT.md#4-baselines-noise-and-calibration)) |
| a new launcher build | nothing; the next poll sees it. If `engine_stale` posts, vendor the engine. |
| turn it all off | `pipeline_rollout`, `pipeline_retrain_auto`, `pipeline_schedule`, then `pipeline_enabled = false` + apply. Copy out what you want to keep first (`cases/`, `baselines/`, `state/`): the case pool holds every authored round. `sous-chef-model` keeps whatever image it runs. Thermometer: `souschef_pipeline = false`. |

## 8. Costs

Recorded or estimated; cost is not a brake on this suite.

| item | per |
|---|---|
| a drift check on the eval L4 | about 1.5–2.5 h of L4, about US$2–3 |
| the weekly run | about 3–4 h of L4, about US$4 |
| a retrain round | about US$120–230: teacher collection US$100–175 (every case is re-collected, because the system index is manifest-specific; 6 workers by default so merchants keep room on int); authoring under US$1 for a few targets, up to about US$150 at the 400-target cap; H100 Spot or Flex about US$10–35; L4 about US$6 |
| idle | the eval and candidate services scale to zero; the bucket is small apart from the 18 GiB base tar |

## 9. What guards the decision, the alerts and the state

The three reviews of the build (2026-10-11) asked for these; each is in the
code and in the dry end-to-end run.

| guard | how |
|---|---|
| only the pipeline's key posts or reads decisions | the thermometer admits its Ingest key **by id** (`souschef_ingest_key_ids`); a key minted under the same name by anyone else is 403 |
| a decision row is never re-pointed | a second `candidate_ready` for a candidate from another notification or run is 409; the candidate id must be `c-<run>` / `c-smoke-<run>`. The workflow refuses a decision whose notification, run, SMOKE flag or bar verdict is not its own, and a first ask answered as a duplicate (someone posted first) |
| the image approved is the image rolled | rollout by digest; the candidate tag, the build record and the decision's `results.candidate.digest` must agree |
| a resume never reuses an old decision | `resume_from` any stage up to `eval` is refused (before the lock) when the candidate already has a decision row; `resume_from decide` re-attaches |
| the jobs cannot steer the workflows | the bucket's `infra/` (thermometer URL, switches, model repository, build spec), `config/` and `wf/` (the outbox the workflow posts with the pipeline key, each round's `decide.json` and `rollout.json`) are writable by the workflow account (and Terraform) only; the trainer writes `runs/<id>/train/` only; only the prepare job's account reads the Anthropic key |
| a job cannot change what is built or served | the model build checks the job-written context against the model-image hashes Terraform pins in the build spec; the candidate and eval services serve only the digest the immutable `c-<id>` tag names, and the rollout only the digest the approver saw |
| no alert storm | a failing poll alerts from its 2nd consecutive failure and once a day per streak; one bad answer from int is retried in the job |
| no silent drop | undelivered notifications wait in `wf/outbox/` (only the workflow writes `wf/`); a log-based alert emails `pipeline_alert_emails` |
| no runaway spend | a failed round blocks automatic rounds (`state/failed-round.json`); a stale retrain lock is reported, never silently swallowed |
| training that never starts or is preempted | Spot or on-demand jobs that have not started after `max_pending_s` (1 h) move to the next rung; a preemption the trainer's SIGTERM trap records is resumed from the checkpoints on the same rung |
| a Skip stays skipped | rules (a)–(c) compare on the rows the ack kept, (e) counts only stale rows added since, (g) only schema changes since |
| the decision wait fits Workflows | `decision_ttl_days` ≤ 21 (about 3,600 steps a day of a 100,000-step execution) |

