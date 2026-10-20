# chocolate-muffin on int: the runbook

**chocolate-muffin** is Ministral 3 8B Instruct (`mistralai/Ministral-3-8B-Instruct-2512-BF16` @
`f6fae9795746f63c9be8344932f01275f3c63734`, Apache-2.0) plus a LoRA the Mistral arm trained on r7's dataset
(`mx-20261018-m3-8b-r7`, [MISTRAL.md](MISTRAL.md) §10). On r7's own data it beat r7 (blueberry-muffin, the live Qwen3.5-9B
LoRA): verdict **better** (pooled McNemar b 42 / c 65, p 0.033). On the arm's Vertex L4 its turns were 30–40 % faster
(per-turn p50 about 4.7 s against 6.6 s). This file puts it on int in place of r7:

1. It is checked first on a real Cloud Run L4: the same service shape as production, called the way int's
   sous-chef service calls it, against r7 measured the same way on the pipeline's eval service.
2. The switch is configuration only: the route alias and the model image, in an order that never serves the
   Ministral model with Qwen's settings, and the pipeline's base family, so the next rounds train chocolate-muffin.
3. Rolling back to r7 is `make inhouse-rollback` (the model, then the route), then the Terraform mirror and the
   pipeline's state.

The retrain pipeline's side (training chocolate-muffin from r8) is in the same tree: [RETRAIN.md §R11](RETRAIN.md#r11-the-base-family-chocolate-muffin).
[§7](#7-what-this-needs-outside-this-repository) lists what it needs outside this repository.

Every step that changes what int does is marked **[LIVE]**.

```bash
PROJECT=breadstack-sous-chef-int; REGION=northamerica-northeast2; GPU_REGION=us-east4
PIPE=gs://breadstack-sous-chef-int-pipeline
MODEL_REPO=$GPU_REGION-docker.pkg.dev/$PROJECT/sous-chef-model/model
R7_DIGEST=sha256:0f301196deae8108e05c0b94d245c628d252d4da28e6bd05b45d143f59d13575   # blueberry-muffin r7, live
EXPORT=$PIPE/experiments/mistral/runs/mx-20261018-m3-8b-r7/train/export                # the GGUF + Modelfile.q4_k_m
W=/home/gtong/mx-work/cm; mkdir -p $W/tmp                                              # a disk path, never /tmp
MAIN=/home/gtong/projects/breadstack-sous-chef   # the checkout with int's terraform.tfvars, backend and lock (§3)
WT=<a checkout with this change>                 # steps 1–3; from step 4 on, $MAIN once the change is merged there
```

## Contents

- [1. What the service build carries](#1-what-the-service-build-carries-deploying-it-changes-nothing)
- [2. The chocolate-muffin route](#2-the-chocolate-muffin-route)
- [3. Before you start](#3-before-you-start)
- [4. The steps](#4-the-steps)
- [5. Rollback to r7, and the break-glass](#5-rollback-to-r7-and-the-break-glass--live)
- [6. The L4 check in detail](#6-the-l4-check-in-detail-labtoolscm_l4checkpy)
- [7. What this needs outside this repository](#7-what-this-needs-outside-this-repository)

## 1. What the service build carries (deploying it changes nothing)

| piece | what it does | default |
|---|---|---|
| route option `parser_retries` (`on` \| `off`; messages-compatible only) | **`off`**: when Ollama answers a call HTTP 500 because its tool-call parser failed (MinistralParser: `tool 'x' not found`, arguments that are not a JSON object, or an argument number too large for a float64), the call is **not** retried. Today the SDK silently redraws such a call up to twice, at the route's temperature. Instead the call fails once. It is counted per route, and the service logs `souschef: model parser error: turn … call N on route chocolate-muffin (reported as chocolate-muffin), not retried, K on this instance: tool 'x' not found`. The turn then goes to Claude **exactly like any other failed call**: escalation reason `model_error`, the same request, nothing of the failed call in the history. Without a fallback the turn ends `turn.error model_unavailable`. Every other retry (429, 408, other 5xx, connection errors) stays as it is. | `on`: as today |
| `MODEL_ROUTE_ALIASES` (env, JSON) | **the switch**. With `{"blueberry-muffin":"chocolate-muffin"}`, every turn that names blueberry-muffin, and every turn on it as `DEFAULT_ROUTE`, is answered by chocolate-muffin: its settings, its `report_as`, its name in `turn.done`, the usage rows and the pending doc. `DEFAULT_ROUTE` stays `blueberry-muffin`. The launcher's `?sousChefModel=blueberry-muffin` turns follow too, so no Qwen settings ever reach the Mistral server. The answering route must be allowlisted; the alias is one hop with no chains. `{"blueberry-muffin":"default"}` is the break-glass ([§5](#5-rollback-to-r7-and-the-break-glass--live)). | unset: every route answers itself |
| `/version` | `route_info.<route>.parser_retries: "off"`, `route_aliases`, and `parser_errors` (`{route: count on this instance}` for the routes with the option). Each member appears only when there is something to say, so today's `/version` is byte for byte what it was. | |

README §5 and §7 document the three.

**Proof that today's configuration is unchanged:** `service/internal/loop/default_config_test.go`
(`TestDefaultConfig_UnchangedFromToday`). It runs int's live environment (MODEL_ROUTES, DEFAULT_ROUTE, allowlist, timeout,
read from the running revision on 2026-10-19) with the Primer on, through three turns:

1. an op batch that fast-finishes;
2. a parser 500 on every attempt, which is retried and then escalated;
3. a turn on opus.

It compares every request against `testdata/default_config.golden.json` byte for byte: method, path, query, every
header but the SDK's three that name the test machine (OS, arch, Go version), and the body's sha256. The same file holds
the primes, the events, the usage rows and the `souschef:` log lines. That golden file was written by the same test on the
code before this change (main 2742dc7 + the arm). The test also checks the 3 attempts and that no `souschef:` line
appears.

**Terraform changes nothing either until you ask it to.** `gpu_model_mistral_route` (§2) defaults to off and
`model_route_aliases` to `{}`: with both, `MODEL_ROUTES`, the allowlist and the service's env are byte for byte today's
(`infra/envs/int/tests`: `inhouse_routes_as_today`; `infra/modules/sous-chef/tests`: `no_alias_no_env`). The one
difference an apply of this tree shows is the pipeline bucket's `infra/config.json` gaining `"base_family": "qwen"`
(the pipeline's switch, [RETRAIN.md §R11](RETRAIN.md#the-switch)), which changes no round.

## 2. The chocolate-muffin route

`gpu_model_mistral_route = { enabled = true }` in int's `terraform.tfvars` adds this route to `MODEL_ROUTES` beside
blueberry-muffin, and to the allowlist (`infra/envs/int/main.tf`, `local.mistral_route`). It uses the same model service
URL and the same Ollama model name, max_tokens and call timeout as `gpu_model_route`:

```json
"chocolate-muffin": {"kind": "messages-compatible", "url": "<sous-chef-model URL>", "auth": "id-token", "model": "sous-chef",
  "max_tokens": 1024, "stream": false, "temperature": 0.15, "fields_context": "on", "fallback": "default",
  "report_as": "chocolate-muffin", "prime": "off", "fast_finish": "on", "parser_retries": "off"}
```

- **`temperature` 0.15.** Mistral recommends ≤ 0.15 for Ministral 3. In the arm, 0.15 (`r4-mxt`) scored slightly better
  than 0.3.
- **`prime` off.** Prompt-cache priming is calibrated for Qwen3.5's hybrid attention (the 1015-token filler and the
  −1024 checkpoint). A transformer like Ministral reuses the shared prefix without it, and the arm measured everything
  with priming off.
- **`parser_retries` off.** Explained in §1.
- **The rest is blueberry-muffin's:** limits, fast finish, the candidate-fields block, Claude as the in-turn fallback,
  and the 60 s `MODEL_CALL_TIMEOUT`.
- **`report_as`.** This is the label every answer carries: `turn.done` (`route.model`, `models`), the trace, the turn
  row and the thermometer's usage rows. In the drawer, merchants see "In-house model" (the launcher labels by route
  kind) and staff see the name.

One set of settings, pinned in four places that must agree: Terraform (`infra/envs/int/tests`:
`the_chocolate_muffin_route`), the service's `service/internal/config/chocolate_test.go` (loads exactly this route on
top of int's table), the pipeline's drift and eval harness (`lab/souschef_lab/harness.py` `FAMILY_ROUTES["mistral"]`)
and the L4 check's profile (`lab/tools/cm_l4check.py` `PROFILES`). `lab/tests/test_harness.py`
(`test_the_mistral_route_matches_infra_int_and_the_service_test`) and `lab/tools/mxtests/test_cm_l4check.py`
(`test_the_profiles_are_the_harness_family_routes`) fail when one of them moves alone. The same tests pin the route →
family map the drift poll uses (`chatfmt.ROUTE_FAMILIES`) to the two route names.

## 3. Before you start

- **A commit, merged into `$MAIN`, before any Terraform step (step 4b on). This is a hard prerequisite.**
  - Committing needs the user's explicit go-ahead.
  - Every `terraform plan` / `apply` of int runs in `$MAIN/infra/envs/int` and nowhere else. Only that directory has
    int's `terraform.tfvars`, the initialised backend (`.terraform/`) and `.terraform.lock.hcl`; all three are
    gitignored, so no other checkout or worktree has them.
  - **Never create a `terraform.tfvars` in another checkout.** One holding only the new keys plans everything else at
    the variables' defaults: `public_invoker=false`, the internal-LB ingress, `gpu_model_*` off and the pipeline off.
    That cuts the launcher off int and destroys the model service.
  - Steps 1–3 need no commit (any checkout with this change, `$WT`). An uncommitted tree under a tag of its own never
    extends to Terraform.
  - **From the switch on, every int apply must come from a tree that has this change, with `model_route_aliases`
    (and `pipeline_base_family`) in its tfvars.** An apply from an older tree, such as `$MAIN` at 2742dc7, silently
    removes `MODEL_ROUTE_ALIASES` and the chocolate-muffin route. The service still boots, and serves the Ministral
    image with Qwen's settings: priming filler, temperature 0.3, hidden parser redraws, labelled blueberry-muffin.
    The drift poll posts a route/family notice when that lasts an hour ([DRIFT.md "Manual rollbacks"](DRIFT.md#manual-rollbacks)).
- **No retrain round may be in flight.** Check that `gcloud storage cat $PIPE/state/locks/retrain.json` finds nothing
  (nothing there on 2026-10-19).
  - A round in its candidate or eval stage re-points `sous-chef-model-cand`, and step 2 would corrupt that round's eval.
  - A round waiting for a decision would roll its Qwen candidate over chocolate-muffin on an Approve.
  - Keep `pipeline_retrain_auto` off until r8 has been reviewed (it is off on int). Step 5b sets
    `pipeline_base_family = "mistral"`, so every round started after the switch trains chocolate-muffin.
- **The thermometer knows the name.** Rows reported as `chocolate-muffin` must price at $0 and show as an in-house
  model. This is a breadstack-thermometer change and deploy (not there on 2026-10-19):
  - `server/internal/therm/pricing.go` needs `{"chocolate-muffin", ModelPrice{0, 0, 0}}` (`"muffin"` is a prefix entry
    and does not match it).
  - `client/src/lib/models.ts` `OTHER_INHOUSE` needs `"chocolate-muffin": { label: "Chocolate Muffin", app: "Sous-chef" }`.

  Without them the rows are stored but shown unpriced. Do this before step 5.
- **L4 quota.** us-east4 has 3 Cloud Run L4s (production, eval, candidate), and a production revision swap needs a spare.
  `make model-rollout` (step 5, rollback) must wait until `sous-chef-model-cand` and `-eval` have idled, about 15–20 min
  after their last request; the L4 check (step 3) uses both. The pipeline waits 20 min for the same reason. A swap
  that gets no L4 fails, and the old revision keeps serving.
- **The experiment's files.** Step 1 reads `$EXPORT` under `experiments/mistral/`, which [MISTRAL.md §7](MISTRAL.md#7-cleanup)
  deletes after the arm. Build the image (and run RETRAIN.md §R11's copies, for r8) before that cleanup.

## 4. The steps

### Step 1. Build the chocolate-muffin model image (a person: bucket writes under `runs/cm-r7-serve/`, one Cloud Build; no GPU)

The image goes through the path every round's candidate does: `make model-stage`, `lab/model-image/model-context.sh`, the
`Dockerfile`, and Terraform's pinned build spec (`infra/cloudbuild-model.json`) on `souschef-builder`. `lab/model-image/`
is unchanged, so the pins hold. The export's own `train/stage/` was staged at 0.3, so the GGUF is re-staged here from
`export/` at 0.15; `lab/tests/test_base_family.py` runs this staging step on the export's exact Modelfile.

```bash
CM_TAG=cm-r7; R=cm-r7-serve; MX=mx-20261018-m3-8b-r7    # CM_TAG: any unused tag (the repository's tags are immutable)
rm -rf $W/$R; mkdir -p $W/$R/export
export CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED=False
gcloud storage cp $EXPORT/$MX-q4_k_m.gguf $EXPORT/Modelfile.q4_k_m $W/$R/export/
(cd $W/$R/export && echo "9b396018481d7ebc47c433a0d90a4ee41aa9d76a54555a3cce16d9f321473258  $MX-q4_k_m.gguf" | sha256sum -c -)
make -C $WT/lab model-stage MODEL_EXPORT=$W/$R/export MODEL_STAGE=$W/$R/stage MODEL_TEMPERATURE=0.15 MODEL_PRESENCE_PENALTY=0
PYTHONPATH=$WT/lab /home/gtong/projects/breadstack-sous-chef/lab/.venv/bin/python -m souschef_lab.chatfmt \
  check-stage $W/$R/stage --family mistral                          # the trainer's own check: TEMPLATE byte for byte,
                                                                    # PARSER ministral, no RENDERER, 0.15, num_ctx 65536
# the build reads runs/ only (souschef-builder): the stage and r7's model-image context under a runs/ id of its own
gcloud storage cp $W/$R/stage/Modelfile $W/$R/stage/$MX-q4_k_m.gguf $PIPE/runs/$R/train/stage/
gcloud storage cp $PIPE/runs/rt-20261018-0213-b02e95-3914e2e4/model/context.tar.gz $PIPE/runs/$R/model/context.tar.gz
gcloud storage cp $PIPE/infra/cloudbuild-model.json $W/$R/cloudbuild-model.json          # Terraform's spec, with the pins
gcloud builds submit $PIPE/runs/$R/model/context.tar.gz --project $PROJECT --region $GPU_REGION \
  --config $W/$R/cloudbuild-model.json \
  --service-account projects/$PROJECT/serviceAccounts/souschef-builder@$PROJECT.iam.gserviceaccount.com \
  --substitutions=_BUCKET=breadstack-sous-chef-int-pipeline,_RUN_ID=$R,_IMAGE=$MODEL_REPO:$CM_TAG
CM_DIGEST=$(gcloud artifacts docker images describe $MODEL_REPO:$CM_TAG --format='value(image_summary.digest)')
echo "$CM_DIGEST"                                    # sha256:<64 hex>, else stop
```

r7's `model/context.tar.gz` is the current `lab/model-image/`, and the build's first step checks it against the pins. The
image is Ollama 0.34.4 with the model named `sous-chef`; step 3's serving asserts check the rest on a real L4.

### Step 2. Put it on `sous-chef-model-cand` (the pipeline's candidate service; nothing merchants reach)

```bash
gcloud run services update sous-chef-model-cand --project $PROJECT --region $GPU_REGION --image $MODEL_REPO@$CM_DIGEST
gcloud run services update-traffic sous-chef-model-cand --project $PROJECT --region $GPU_REGION --to-latest
gcloud run services describe sous-chef-model-cand --project $PROJECT --region $GPU_REGION \
  --format='value(status.url,spec.template.spec.containers[0].image,status.traffic)'   # …@$CM_DIGEST, 100 % latest
```

The next retrain round re-points this service at its own candidate, so nothing needs undoing afterwards.

### Step 3. The L4 check (deploy-free; two lock objects in the bucket; no load on the live model)

```bash
gcloud run services describe sous-chef-model-eval --project $PROJECT --region $GPU_REGION \
  --format='value(spec.template.spec.containers[0].image)'          # …@$R7_DIGEST (the drift workflow keeps it on
                                                                    # production's digest); otherwise the check is INCOMPLETE
cd $WT/service && go build -o $W/sous-chef ./cmd/sous-chef && go build -o $W/sous-chef-fields ./cmd/sous-chef-fields
cd $WT/lab
L4=$W/l4-$(date -u +%Y%m%d-%H%M)
env -u PYTHONPATH SOUS_CHEF_BIN=$W/sous-chef SOUS_CHEF_FIELDS_BIN=$W/sous-chef-fields TMPDIR=$W/tmp \
  /home/gtong/projects/breadstack-sous-chef/lab/.venv/bin/python tools/cm_l4check.py check \
  --engine bucket --out $L4; echo "exit $?"     # 0 PASS, 1 FAIL, 3 INCOMPLETE; $L4/verdict.json is step 5's VERDICT
```

What `check` does:

- It finds both URLs (`gcloud run services describe`, read-only) and records each service's image and traffic, before
  and after its measure.
- It holds the pipeline's own locks for both measures: `state/locks/retrain.json` (no round starts and re-points the
  candidate service) and `state/locks/gpu-eval.json` (no drift check or round eval shares or re-points the eval
  service). A lock another run holds stops the check (exit 3): try again when that run has finished. The locks are
  deleted at the end; a killed check leaves them to expire after 3 h.
- It measures **r7 on `sous-chef-model-eval`** (the blueberry-muffin route settings) as the reference. The eval service
  serves production's digest on production's shape (L4, 4 CPU, 16 GiB, concurrency 4) with no merchant traffic, so
  no merchant's call queues behind the measure on the one Ollama slot, and the live model gets no extra load.
- It then measures **`sous-chef-model-cand`** (chocolate-muffin) as the candidate, the same way and on the same day.
- It writes `verdict.md` / `verdict.json` beside the two `measure.json`.

**PASS** needs all of the following:

- Both measures are comparable: the same rows, engine, gap and timeout, the same UTC day, ≥ 30 ok turns each, and the
  serving asserts passed.
- They measured what the switch is about: the candidate on the chocolate-muffin profile, the reference on
  blueberry-muffin's, two different services, the reference serving **r7's digest**, and each service serving 100 %
  from its latest revision, unchanged from the start of its measure to the end. Anything else is INCOMPLETE.
- The candidate's **per-turn p50 is no worse than r7's**.
- The candidate drew **0 parser errors**.
- The candidate ended **no more failed turns**, and drew **no more non-parser call errors**, than r7.

Read `verdict.md` for the rest: per-call p50/p90, calls per turn, tool-call validity, other errors and the exam pass count.

About 15–25 min, under US$1 of L4 time on the two pipeline services. A FAIL or INCOMPLETE stops here.
[§6](#6-the-l4-check-in-detail-labtoolscm_l4checkpy) has the details.

### Step 4. Deploy the service build with the route available, default unchanged — [LIVE: new revisions, no behaviour change]

From here on, run everything in `$MAIN`, after this change is committed and merged there (§3, the user's go-ahead).

**a. The image, the pipeline's images for the same tag, then the roll** ([PIPELINE.md §4.3](PIPELINE.md#43-the-sous-chef-build-by-hand-int-has-no-ci-trigger);
this change has no migration). `T` is the service tag: the short sha of the merged commit in `$MAIN`, which is also
what `make`'s default `BUILD_ID` gives there. Never build from an uncommitted tree under `make`'s default: that is
`git rev-parse --short HEAD` (2742dc7), int already runs `sous-chef:2742dc7`, the repository's tags are mutable, and
the push would re-point that tag (and `sous-chef-lab:2742dc7`, which the pipeline resolves from the serving tag) at
other code.

```bash
cd $MAIN && git log --oneline -1 && git status --short   # the merged commit, a clean tree
T=$(git rev-parse --short HEAD); AR=$REGION-docker.pkg.dev/$PROJECT/sous-chef
make test                                             # go vet + go test -race
make docker-build && make docker-push AR_REPO=$AR     # BUILD_ID defaults to $T here
make lab-image TAG=$T && make trainer-image TAG=$T    # BEFORE the roll: the drift workflow takes both from the serving tag,
                                                      # and drift/eval read the live model's family with THIS lab
gcloud run deploy sous-chef --image $AR/sous-chef:$T --no-traffic --tag p-$T --project $PROJECT --region $REGION
TURL=$(gcloud run services describe sous-chef --project $PROJECT --region $REGION --format=json \
  | jq -r --arg t "p-$T" '.status.traffic[] | select(.tag == $t) | .url')
curl -s "$TURL/version" | jq '{build_id, default_route, route_info, route_aliases, parser_errors}'
# build_id $T; route_info exactly as before (no chocolate-muffin, no parser_retries); route_aliases and parser_errors null
gcloud run services update-traffic sous-chef --to-latest --project $PROJECT --region $REGION
gcloud run jobs update sous-chef-retention --image $AR/sous-chef:$T --project $PROJECT --region $REGION   # same binary
gcloud run jobs update sous-chef-migrate --image $AR/sous-chef:$T --project $PROJECT --region $REGION     # no migration to run
```

**b. The route** (Terraform, in `$MAIN/infra/envs/int` only). Plan only after the traffic switch (PIPELINE.md §4.3): a
plan made before it puts the stale traffic back. Pass the live `DEPLOY_NONCE`, or the apply rolls one more revision for
nothing.

```bash
cd $MAIN/infra/envs/int && ls terraform.tfvars .terraform.lock.hcl .terraform >/dev/null   # int's own; else stop (§3)
# terraform.tfvars, beside everything already there: gpu_model_mistral_route = { enabled = true }
#                  (model_route_aliases stays unset / {}, pipeline_base_family unset / "qwen")
IMG=$(gcloud run services describe sous-chef --project $PROJECT --region $REGION --format='value(spec.template.spec.containers[0].image)')
NONCE=$(gcloud run services describe sous-chef --project $PROJECT --region $REGION --format=json \
  | jq -r '.spec.template.spec.containers[0].env[] | select(.name == "DEPLOY_NONCE") | .value')
terraform plan -var=image=$IMG -var=deploy_nonce=$NONCE -out=$W/route.tfplan
# expect: sous-chef's MODEL_ROUTES gains chocolate-muffin and MODEL_ROUTE_ALLOWLIST ",chocolate-muffin";
#         the pipeline bucket's infra/config.json gains "base_family": "qwen"; nothing else (anything else: stop)
terraform apply $W/route.tfplan && rm -f $W/route.tfplan    # the saved plan holds variable values: do not keep it
```

**c. Check it:**

```bash
URL=$(gcloud run services describe sous-chef --project $PROJECT --region $REGION --format='value(status.url)')
curl -s $URL/version | jq '{build_id, default_route, routes, cm: .route_info["chocolate-muffin"], route_aliases, parser_errors}'
# build_id $T; default_route "blueberry-muffin"; cm.parser_retries "off", cm.report_as "chocolate-muffin";
# route_aliases null; parser_errors {"chocolate-muffin": 0}
```

### Step 5. The switch — [LIVE]

Do it in this order: **the route first, then the model.** In between, r7 is answered with chocolate-muffin's settings:
temperature 0.15, no priming (its first calls are slower), labelled chocolate-muffin. That is harmless. The other order
would send Qwen's priming filler and silent redraws to the Mistral server.

**a.** Once `-cand` and `-eval` have idled (§3, quota), with step 3's verdict:

```bash
cd $MAIN && make inhouse-switch DIGEST=$CM_DIGEST VERDICT=$L4/verdict.json
```

`inhouse-switch` refuses to change anything unless:

- the verdict is a **PASS measured on `model@$CM_DIGEST`** (`verdict.json` `summary.candidate.image`);
- `/version` shows chocolate-muffin live with `parser_retries` off and no alias yet;
- the traffic of both `sous-chef` and `sous-chef-model` is on their latest revisions.

Then it runs:

```bash
# the route: the alias, by gcloud (checked in /version before the model moves)
gcloud run services update sous-chef --project $PROJECT --region $REGION \
  --update-env-vars='^@^MODEL_ROUTE_ALIASES={"blueberry-muffin":"chocolate-muffin"}'
# A revision whose alias names a route that is not in MODEL_ROUTES or MODEL_ROUTE_ALLOWLIST refuses to boot
# (configuration error); Cloud Run then keeps the previous revision serving.
make model-rollout TAG=$CM_DIGEST                      # sous-chef-model → model@$CM_DIGEST, 100 % to it
```

If `make model-rollout` fails (no spare L4), `inhouse-switch` reads which revision `sous-chef-model` serves:

- **r7 still serves** (the usual case): it takes the alias away again, so int is back where it was before the switch.
  Let `-cand` and `-eval` idle, then run the target again.
- **chocolate-muffin already serves** (the image rolled, `update-traffic` failed): the alias stays. Finish with
  `make model-rollout TAG=$CM_DIGEST` and go on.
- **It cannot tell**: the alias stays, and the target prints the command that takes it away once you see r7 serving.

**b. The Terraform mirror and the pipeline's base family**, right after (in `$MAIN/infra/envs/int`): add to
`terraform.tfvars`

```hcl
model_route_aliases  = { "blueberry-muffin" = "chocolate-muffin" }
pipeline_base_family = "mistral"
```

then plan and apply as in step 4b (with `IMG` and `NONCE` read again). Expect `MODEL_ROUTE_ALIASES` in sous-chef's env
and `"base_family": "mistral"` in the bucket's `infra/config.json`, nothing else. The apply may roll one more, identical,
revision: gcloud appended the variable, and Terraform orders the env by name.

- Without the alias in tfvars, the next apply takes the alias away again (§3).
- Without `pipeline_base_family`, every round (`make pipeline-retrain` / `pipeline-smoke` send no `base_family`, and
  drift-started rounds read the bucket's config) would train blueberry-muffin against a live chocolate-muffin. Its
  Approve would roll a Qwen image under chocolate-muffin's settings and label, with only a report warning.

Optional, the record every pipeline rollout leaves: `gcloud artifacts docker tags add $MODEL_REPO@$CM_DIGEST $MODEL_REPO:live-cm-r7`.

Then one turn through the launcher on int, with no `?sousChefModel=` or with `?sousChefModel=blueberry-muffin`.
The drawer shows "In-house model". `turn.done.route.name` and the thermometer row say `chocolate-muffin`.

**Watch it:**

- `/version .parser_errors`. This counts per instance and resets on every new revision.
- The service log lines:

  ```bash
  gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="sous-chef" AND textPayload:"souschef: model parser error"' \
    --project $PROJECT --freshness=1d --limit=50 --format='value(timestamp,textPayload)'
  ```

- The thermometer's in-house share and its escalations: `turn.escalated` / `routes [chocolate-muffin, default]`.

**Roll back (§5)** on any of these:

- a run of parser errors;
- an escalation rate clearly above r7's;
- slower turns than the L4 check showed.

### Step 6. `state/model.json` and the drift baseline — [PIPELINE STATE] (within the hour of step 5)

The pipeline still names r7 in `state/model.json`. While production serves a digest the pipeline has never seen, drift's
behavioural modes (check, weekly, calibrate) stop with a configuration error and post `stage_failed`
([DRIFT.md "Manual rollbacks"](DRIFT.md#manual-rollbacks)). The 15-minute poll compares int's in-house route with the
entry's family, and posts a route/family notice once that mismatch has lasted an hour. So write the entry right after
the switch.

The entry carries **`"family": "mistral"`**: drift and the retrain eval ask the live model with its family's route
settings (`harness.FAMILY_ROUTES`: temperature 0.15, `prime` off, `parser_retries` off) and refuse to run when the eval
service's `/api/show` reports another family than the entry says, so an entry without it stops drift instead of
measuring chocolate-muffin with Qwen's settings. They read `state/model.json` and `/api/show`, not `/version
route_aliases`; only the poll's route/family notice reads `/version`. This needs the lab image of step 4a: drift runs the
lab image of the int service's serving tag, and an older one knows no family (it would ask chocolate-muffin with
blueberry-muffin's settings, through a binary that redraws parser errors).

```bash
gcloud storage cat $PIPE/state/model.json > $W/model.json.r7                 # keep r7's entry: the rollback puts it back
# chocolate-muffin trained on r7's dataset: r7's training manifest (3914e2e4…), dataset and training cases
# (cases/trained/c-rt-20261018-0213-b02e95-3914e2e4/…) are its own; r7's entry is appended to the history
# the way retrain/decide.py appends an approved round's predecessor (newest last, at most 20)
jq --arg d "$CM_DIGEST" --arg t "$CM_TAG" '{tag: $t, digest: $d, family: "mistral", training_manifest,
     trained_at: "2026-10-18T18:51:09Z", run_id: "mx-20261018-m3-8b-r7", candidate_id: null, dataset, training_cases,
     approved_by: "<who>",
     history: (((.history // []) + [{tag, digest, training_manifest, trained_at, run_id, training_cases, dataset}])[-20:])}' \
  $W/model.json.r7 > $W/model.json.cm
jq '{tag, digest, family, training_manifest, history: [.history[].tag]}' $W/model.json.cm      # read it before writing
gcloud storage ls $PIPE/state/ack.json 2>/dev/null && gcloud storage mv $PIPE/state/ack.json $W/ack.json.r7
#   (an Approve deletes the ack: it holds the old model's arm-D rows; none on 2026-10-19)
gcloud storage cp $W/model.json.cm $PIPE/state/model.json --content-type=application/json      # a person
make pipeline-drift MODE=calibrate      # → baselines/$CM_DIGEST/baseline.json (about 3 h on the eval L4)
```

The next `check` or `weekly` would calibrate on its own anyway: a digest without a baseline is calibrated first.

Do **not** use the arm's `eval/r7c-mx/…/candidate-baseline.json` as the baseline. It was measured on Vertex at temperature
0.3, not at the route's settings on the int L4.

## 5. Rollback to r7, and the break-glass — [LIVE]

The model goes back first. Let `-cand` and `-eval` idle first (§3). r7's baseline (`baselines/sha256:0f301196…/`)
already exists.

```bash
cd $MAIN && make inhouse-rollback DIGEST=$R7_DIGEST
# = make model-rollout TAG=$R7_DIGEST, then
#   gcloud run services update sous-chef --project $PROJECT --region $REGION --remove-env-vars=MODEL_ROUTE_ALIASES
#   (checked in /version)
```

- **Between the two commands,** r7 runs with chocolate-muffin's settings: temperature 0.15, no priming, labelled
  chocolate-muffin. That is harmless, so roll the model first.
- **If `make model-rollout` fails** (no spare L4), the alias stays, so chocolate-muffin keeps its own settings. Run the
  target again once `-cand` and `-eval` have idled, or use the break-glass below if the model must stop answering now.
- **The pipeline's state**, right after:
  - Put r7's entry back: `gcloud storage cp $W/model.json.r7 $PIPE/state/model.json --content-type=application/json`.
  - Move aside any `state/ack.json` written while chocolate-muffin was live. An ack is not tied to a digest: it holds
    the arm-D rows of the model that was live when it was written. Use
    `gcloud storage mv $PIPE/state/ack.json $W/ack.json.cm`, then put back r7's ack if step 6 moved one aside
    (`gcloud storage cp $W/ack.json.r7 $PIPE/state/ack.json --content-type=application/json`).
  - Until the entry is back, drift judges the rollback by itself: production serves a digest in `history`, judged on its
    own baseline with its own family (qwen), and a notice is posted.
- **Terraform** (in `$MAIN/infra/envs/int`): `model_route_aliases = {}` and `pipeline_base_family = "qwen"` in
  `terraform.tfvars`, then plan and apply as in step 4b. Otherwise the next apply brings the alias back, and new rounds
  keep training chocolate-muffin. Expect `MODEL_ROUTE_ALIASES` gone from sous-chef's env and `"base_family": "qwen"`
  in the bucket's `infra/config.json`, nothing else.
- **The route stays available** (blueberry-muffin's default is untouched). The image stays in the registry under its
  immutable tag, so switching again later is only steps 5 and 6.

**The break-glass (no spare L4 needed).** To take every in-house turn off the model at once, answer blueberry-muffin with
Claude's `default` route. It is allowlisted on int, so the alias passes the service's checks:

```bash
gcloud run services update sous-chef --project $PROJECT --region $REGION \
  --update-env-vars='MODEL_ROUTE_ALIASES={"blueberry-muffin":"default"}'
curl -s $URL/version | jq .route_aliases                 # {"blueberry-muffin":"default"}
```

- Every in-house turn, named or not, is then a Claude Sonnet turn (Claude's price, Claude's label in the thermometer).
  Turns that name `chocolate-muffin` (staff only) still reach the model.
- The drift poll says nothing while it is set: Claude's route has no base family.
- To end it: `make inhouse-rollback DIGEST=$R7_DIGEST` (r7 back, the alias removed), or, to give chocolate-muffin its
  turns back, set the alias to `{"blueberry-muffin":"chocolate-muffin"}` again with the same command
  (`--update-env-vars='^@^MODEL_ROUTE_ALIASES={"blueberry-muffin":"chocolate-muffin"}'`). `make inhouse-switch` refuses
  while any alias is set.
- If it stays past the next `terraform apply`, mirror it: `model_route_aliases = { "blueberry-muffin" = "default" }`
  (`infra/envs/int/tests`: `the_break_glass`).

## 6. The L4 check in detail (`lab/tools/cm_l4check.py`)

**Subcommands:**

- `measure --url --profile chocolate-muffin|blueberry-muffin --out [--service-name]`: measures one service. With
  `--service-name` the service's image and traffic are recorded (`compare` needs them for the reference). It takes no
  lock: use `check`, or make sure nothing else uses the service.
- `compare --candidate --reference [--reference-digest]`: gives the verdict. Exit 0 PASS, 1 FAIL, 3 INCOMPLETE.
- `check`: holds the pipeline's locks, runs both measures, then compare. `--ref-service` (default
  `sous-chef-model-eval`; `--live-service` is the old name), `--cand-service` (default `sous-chef-model-cand`),
  `--bucket` (the pipeline bucket), `--no-lock` (nothing else may use the GPU services).
- `rows`: prints the replayed rows.

**The verdict** (`compare`):

- **INCOMPLETE** when the measures are not comparable (rows, tool version, engine, gap, call timeout, UTC day, ≥ 30 ok
  turns each, serving asserts), or did not measure what the switch is about: the candidate profile is not
  chocolate-muffin or the reference's not blueberry-muffin, both are the same URL, the reference's image is unknown
  (measured by `--ref-url`) or not r7's digest, or a recorded service's traffic was not 100 % on its latest revision or
  changed during its measure.
- **FAIL** when the candidate's per-turn p50 is worse than the reference's (× (1 + `--tolerance`), default 0), it drew a
  parser error, or it ended more failed turns or drew more non-parser call errors than the reference.
- **PASS** otherwise. `verdict.json` names each side's image (`summary.<side>.image`); a candidate measured by
  `--cand-url` has none, so `make inhouse-switch` refuses its verdict.

**One measure:**

1. **Serving asserts, straight at the URL.**
   - `/api/version`.
   - `/api/show`: PARSER `ministral` with the lab's TEMPLATE byte for byte (or `qwen3.5` for r7), architecture
     `mistral3` / `qwen35`, num_ctx 65536, the `tools` capability.
   - A load, then `/api/ps` at 100 % VRAM.
   - The prompt-size smoke: `serving_mistral.SMOKE_REQUEST` must be 61 tokens (Qwen: 279), counting any prefix a warm
     server read from its cache.
   - An echo-tool call at temperature 0 must not draw a parser 500 or leak `[TOOL_CALLS]` as text. Ministral must
     answer with a `tool_use`; a Qwen text reply is only recorded.

   A Qwen image on the candidate fails here, so does a wrong Modelfile.
2. **The replay.** This uses the lab's harness: Postgres plus `$SOUS_CHEF_BIN`, which must be **built from this tree**
   (the chocolate-muffin profile refuses a binary that does not report `parser_retries`).
   - **One route**, the profile's settings exactly as `MODEL_ROUTES` gives them, minus `fallback`: the student alone,
     never Claude.
   - **A local proxy** in front of the URL. It adds `X-Serverless-Authorization: Bearer <ID token>`, which is what the
     service's `auth: id-token` sends and how the pipeline's eval calls the cand service. It records every
     `/v1/messages`.
   - **40 real-phrasing exam rows**, about 60 turns: real 14, followup 6, paraphrase 8, holdout 6, out_of_scope 4,
     out_of_scope_holdout 2.
     - The pick is fixed and seeded, the same for every URL and every day.
     - They run as the eval's `<set>:recorded` arm runs them (`exam.run_row`): history turns first, the engine
       applying the ops, fast finish as the service decides.
     - One turn runs at a time. Each turn starts once the model server has been quiet for `--gap-s` (1.5 s), so r7's
       prime finishes as it would between a merchant's turns.
     - Two warm-up rows come first and are left out of every number.
3. **`measure.json`.** It holds:
   - **per turn**: `POST /v1/turns` to the end of the turn's last stream, ok turns, p50/p90, and the failed turns by
     code;
   - **per call**: each `/v1/messages` at the proxy, primes apart, p50/p90;
   - calls per turn and token counts;
   - **tool-call validity**: the name was among the tools the service sent, and the input passes that tool's
     `input_schema`;
   - **parser errors**: 500s whose message is a tool-call parser's, plus the service's `souschef: model parser error`
     lines;
   - other errors (non-200 calls that are not parser errors), and the exam pass count (information only);
   - with `--service-name`: `service` and `service_after`, the image and traffic before and after.

   `calls.jsonl` holds the per-call records: request shapes, token counts, the tool names called and any error message;
   no conversation text, never the token. `service.log` is beside it.

**Auth** (`--auth`):

- `gcloud` (default): your login. Owner holds `run.routes.invoke` on both services.
- `--impersonate <SA>`: a pipeline account's token with the URL as audience. This needs Token Creator on that account.
- `metadata`: inside a GCP job, as `model/idtoken.go` does.

Tokens are re-minted 5 minutes before they expire and never written anywhere. The locks are written with your gcloud
login (`gcloud storage`).

**Needs:**

- Postgres server binaries (on PATH or `PG_BIN`).
- `$SOUS_CHEF_BIN` and `$SOUS_CHEF_FIELDS_BIN` (step 3 builds both).
- node and the engine: `--engine bucket` takes the pipeline's newest vendored snapshot, read-only. Or pass a
  cookie-cutter snapshot or checkout.
- A short `TMPDIR` (≤ 70 characters). Postgres's socket goes there; a longer one falls back to /tmp.

**Tests:** `lab/tools/mxtests/test_cm_l4check.py` covers:

- the classifier, which is the service's list (it reads `service/internal/model/parser.go`);
- tool validity on the real tools;
- tokens;
- the proxy;
- the serving asserts against a fake Ollama;
- the verdict, including the review's case (31 of 60 turns ok, 29 `model_unavailable`, 87 call errors: FAIL), every
  INCOMPLETE reason and an unknown candidate image;
- `check` holding and releasing the locks, and stopping while a drift check holds `gpu-eval`;
- the row pick;
- the import guard;
- the profiles against the harness's family routes and the drift poll's route → family map (§2);
- with Go and Postgres here, two whole measures through the real harness and a binary built from this tree. A
  MinistralParser 500 on chocolate-muffin is drawn once, logged and fails its turn `model_unavailable`. On
  blueberry-muffin it is redrawn and its turn succeeds, and the verdict is FAIL (a parser error and one failed turn
  more than r7).
- one measure through the real engine.

`mxtests` is not collected by `make test` (`testpaths = ["tests"]`): run it as [MISTRAL.md §9](MISTRAL.md#9-tests) does.

## 7. What this needs outside this repository

In this tree: the Terraform route and switch (`gpu_model_mistral_route`, `model_route_aliases`, their tests), the
pipeline's base family ([RETRAIN.md §R11](RETRAIN.md#r11-the-base-family-chocolate-muffin): drift and the eval serve a
live chocolate-muffin with its route settings, from `state/model.json` `family` and `/api/show`; the drift poll compares
int's in-house route with that family; the retrain workflow's rollout check still finds `route_info["blueberry-muffin"]`,
which stays listed), `make inhouse-switch` / `inhouse-rollback`, and README §5/§7. Outside it:

- **A commit, merged into `$MAIN`** (the user's go-ahead): required before step 4b, because Terraform runs only there
  (§3). It also makes `/version`'s build id name the code it runs, and the lab and trainer images match it.
- **`$MAIN/infra/envs/int/terraform.tfvars`** (gitignored, int's own): `gpu_model_mistral_route` (step 4b), then
  `model_route_aliases` and `pipeline_base_family` (step 5b; back in §5).
- **breadstack-thermometer**: the rate-card and label entries (§3), deployed before step 5.
- **For r8**: RETRAIN.md §R11's copies (the Ministral base tar and `base/adapters/cm-r7` into `base/`) before the
  experiment's cleanup, then its SMOKE and round.
