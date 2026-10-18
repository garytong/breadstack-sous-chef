# Drift: is the in-house model still fit for the launcher it serves?

The launcher changes its editor (fields, labels, lists, sections) with every
build. blueberry-muffin, the in-house model, learned one manifest. This
runbook explains how the pipeline measures the gap, when it calls the gap
"big enough" to retrain, and how to run every check by hand.

Top-level runbook: [`docs/PIPELINE.md`](PIPELINE.md). Retraining:
[`docs/RETRAIN.md`](RETRAIN.md). Code: `lab/souschef_lab/{drift,harness,examrebase,baseline,enginebundle,gcsio,pipeline_common}.py`,
thresholds `lab/pipeline/thresholds.json`.

```bash
export PROJECT=breadstack-sous-chef-int REGION=northamerica-northeast2 WF_REGION=us-central1
export PIPE=gs://breadstack-sous-chef-int-pipeline
```

All shell in this file is **bash**. The lab's loops read `PIPESTATUS`, which
zsh does not have (an r4 lesson): run `bash` first if your shell is zsh.

---

## 1. What is compared

| name | what | where it comes from |
|---|---|---|
| **training** manifest | the one the live model trained on | `state/model.json → training_manifest`; for r4 that is `bf788701…` = `lab/fixtures/exam/manifest-real.json` |
| **deployed** manifest | the one the int launcher runs now | `GET /v1/manifests/deployed?line=int` ([§7](#7-the-deployed-manifest-endpoint)) |
| **ack** manifest | the one you last skipped a candidate on | `state/ack.json` ([§3.3](#33-after-a-skip)) |

### Why wording drift is usually tolerated

The model does not memorise labels. Every request carries the manifest's
labels and help text fresh: the system index (`registry.Compile`) and the
`<candidate_fields>` block are rendered from the deployed manifest on every
turn. A relabelled field is usually still found. What the model cannot
handle from the prompt alone is a **key it never wrote** (a new or renamed
field, list or section) or a key that is gone. That is why the static diff
classes below weigh structure over wording, and why a behavioural exam,
not the diff, decides.

## 2. The static diff (about 2 minutes, no model)

`drift static` compares training (or ack) with deployed. Fields are addressed
as `<container>.<key>` and `<container>.<list>[].<key>`, the same addressing as
`retrain_check.field_index` and the service's `manifest.ComputeDiff`.

| class | changes | effect |
|---|---|---|
| **R0** noise | `source` (file and line), key order | ignored |
| **R1** wording | label, help, description, placeholder, group title | usually tolerated (above); can move which fields retrieval offers, so the exam decides |
| **R2** soft schema | enum options added or removed, default, min/max/step, gates (`when`, visibility), list min/max | values the model writes may now be refused or hidden |
| **R3** hard schema | field added, removed, renamed or retyped; list added, removed or item schema changed; a section, global panel, page or details container added or removed; a new op name | the model has never seen the key, or writes one that is gone; usually needs a retrain |

**Renames.** A removed and an added field in the same container with the same
type, and an identical label, a label token Jaccard ≥ 0.8 or an identical help
text, are reported as `renames: [{from, to, score}]`. The retrain's rebase uses
this map to carry old cases onto the new key.

**Impact columns.**

| column | means |
|---|---|
| `stale_exam_rows` | exam rows whose expected keys are absent from deployed (listed by row id) |
| `stale_training_rows` | rows of the live model's dataset that write an absent key |
| `new_keys` | R3-added and renamed keys with no exam row: the probe targets of §3 |
| `engine_status` | `current` when the deployed manifest's `generated_from.sha` has a vendored engine snapshot; `stale` when not; `unknown` when the manifest names no commit |

Output: `drift/<run>/static.json`.

## 3. The behavioural check

### 3.1 Arms

The exam is **paired**: the same rows, the same model image, two manifests.

| arm | rows | contexts | manifest and engine |
|---|---|---|---|
| **T** (training) | the real-phrasing exam, all six sets, 252 rows; for a model trained on another manifest than the exam's (every model after r4) the rows are first rebased from the exam's manifest onto the training manifest (`drift.exam_on`) | re-derived on the training manifest (`exam.rederive_contexts`), today's browser mirror | training; the training-time engine. This is the baseline arm, measured by calibration and once a week, not on every check |
| **D** (deployed) | the same rows, rebased from the exam's manifest onto deployed (`drift.exam_on` → `examrebase.rebase_exam`: `manifest_hash` replaced, `meta.rebased_from` added) | re-derived with `state.context_of` on deployed | deployed manifest and engine. Rows in `stale_exam_rows` are not asked; they are counted as `stale` |
| **P** (probe) | for each new, renamed **or retyped** key with no exam row, up to 4 generated cases (`cases.py` kinds that fit its type: set a literal, toggle, pick an enum value, reset; update cases for scalar list items) | each row on **its own state**: the vintage state, plus a home row for any section type the key names that it lacks, preset so the key shows (§3.4) | deployed; each case is checked by the engine on its state, then asked twice. A key whose probes pass < 50 % is a **dodged key** |

The **weekly** run also asks the 233 F0H holdout cases, rebased onto both
manifests (for a model after r4, from r4's manifest onto its training manifest
first, through the engine). Scoring is the r4 scoring (`exam.score_turn`, `r4score`). An
`invalid_op` that writes a key absent from the manifest is counted as a
**ghost write**.

### 3.2 Verdicts

Δ = baseline mean − arm D, in points (rows). Stale rows are left out of both.
σ is the noise model of §4. The rules live in `lab/pipeline/thresholds.json`;
an object at `$PIPE/config/thresholds.json` overrides it without a rebuild.

| verdict | fires when any line holds |
|---|---|
| `ok` | nothing below |
| `notice` (post, no retrain) | Δ_total > 2 σ_Δ · any R3 change · `engine_status = stale` · one dodged key |
| **`retrain`** | **(a)** Δ_total > 3 σ_Δ **and** ≥ 4 points · **(b)** on a set with n ≥ 12, Δ_set > 3 σ_Δ,set **and** ≥ 15 points · **(c)** `out_of_scope` + `out_of_scope_holdout` drop ≥ 3 rows **and** > 3 σ_Δ of the two sets combined · **(d)** ≥ 2 dodged keys, or every probed key of a new container dodged · **(e)** `stale_training_rows` ≥ 3 % of the dataset, or `stale_exam_rows` ≥ 5 % of the exam · **(f)** arm D has ≥ 3 more ghost writes than arm T · **(g)** ≥ 90 days since training **and** any R2/R3 change |

**Re-keyed rows.** An exam row whose answer key a product decision changed
gets a new id `<id>.k<rev>` (`exam.AMENDMENTS`, an entry's `_rev`, 2 by
default; `exam.AMENDED_ID_RE` matches any): a baseline scored on the old key
never pairs with it, so it is left out of every Δ until the live model is
re-calibrated (`drift calibrate --fresh-baseline`, or the approved candidate's
eval baseline). Twenty rows today: the 19 font and vague-request rows of
2026-10-15 and p137.k2 ("create a new page for events": create_page since
2026-10-15, or — on the exam's state, which already has an Events page — the
reply that the page exists; RETRAIN.md R2a). p137.k2 stays in the
`out_of_scope` set although it is no longer out of scope: after the
re-calibration a lost p137.k2 counts toward rule (c) like any row of that set.
(The retrain eval's paired bar leaves it out — its report-only `capability`
set, RETRAIN.md R6 — because the current model there was never offered the
tool; drift compares a model with itself, so it stays in. The candidate
baseline records it under `out_of_scope:p137.k2`.)

**The r7 answer key (`.k3`, answer key `ask-rule-r7-2026-10-18`).** Four more
rows were re-keyed before the r7 round (RETRAIN.md "What changed for r7"):
h022.k3 ("add our weedmaps page"), p030.k3 ("put our facebook page in the
footer") and p047.k3 ("change what shows up as the title on google") may be
answered with a question (`ask_ok`: no URL or title is given); h006.k3 ("the
body copy font should be Inter") gives its value, so a write is required:
`bodyFont`, with the footer's body font (`footerBodyFont` deployed,
`footerBodyTextFontFamily` on the exam manifest) allowed but not required.
The live drift baseline (r6's `sha256:7fb1…`, set on Approve) and r4's
`sha256:f657…` hold these rows under their old ids, so drift leaves them out
of every Δ until the next calibration (`drift calibrate --fresh-baseline`, or
the approved r7 candidate's eval baseline). So recalibrate **only on Skip**:
on Approve, r7's candidate baseline (on `3914e2e4`, with the `.k3` rows)
becomes the drift baseline by itself. The retrain eval's `r4_floor` rule
counts them as `excluded_amended` (RETRAIN.md R6). p050 / p053 (image
requests) are deliberately not re-keyed: their answer is `pick_asset`.

**Option lists in the re-derived contexts (2026-10-16).** Arms T and D ask
the exam with contexts re-derived by `state.context_of`, which now mirrors
the launcher's turn context from 1d4193e2 on: `tags` and `brands` (the lab
fixture's lists, by name) whenever the viewed page, a page the request names
or the open section has a product carousel, or the request says tag / brand /
carousel — 222 of the 252 rows, since the exam's vintage home has a
carousel — and categories carry their id as text (`fixtures/context.json`
gives ids since 2026-10-16, as every live context has them). Both arms get
the same bytes, so Δ is fair; a baseline
calibrated before this change was asked without the lists, which is one more
reason the re-calibration after r6 is due. The engine ctx of every replay
(`verify.ctx_of`, also the collector's) now carries the fixture's tag and
brand ids, so a tagSlug / brandSlug write also writes its tagId / brandId,
as in the editor.

**Two replicates, up front.** Every check and weekly run asks arm D twice
(`arm_d_replicates` = 2, about 50 minutes more than one): a rule from (a), (b)
or (c) is judged on both, so it is confirmed in the same run instead of a
second pass later — timeliness over cost. One unlucky sample at temperature
0.3 never starts a round. (d)–(g) are deterministic or already counted twice.

*Measured power* (verifier 2, Monte Carlo through `baseline.compare` and
`evaluate` with the exam's set sizes and measured flip rates): no false
retrain in 400 runs on a K = 3 baseline; a true 15-row drop (6 %) is a
confirmed retrain in about half the checks, a 10-row drop in about a fifth.
Moderate drift is therefore caught by repetition (every check after each
launcher change, every Monday) and by the deterministic rules, not by one
exam.

Next to the noise rule the report prints an exact paired McNemar test between
arm D and the baseline replicate with the same index. It is reported, not a
rule.

### 3.3 After a Skip

A Skip leaves the live model in place, so the same drift would fire again on
the next poll. `state/ack.json` stores what you skipped:
`{at, candidate_id, deployed_hash, static_counts, arm_D_mean, arm_D_sets,
arm_D_rows, dodged_keys, ghost_writes, stale_training_rows, stale_exam_rows}`.
While it exists, every rule counts only drift **beyond** it:

- Δ is measured against the ack's arm-D replicates **on the rows both have**
  (`arm_D_rows`; the rows the ack, the baseline and today's arm D share), so a
  row that goes stale or is rebased after the Skip never shifts Δ (an older
  ack with per-set means only is compared on those);
- R3 changes are counted against `ack.deployed_hash`;
- only dodged keys not already in the ack count;
- (e) counts only the stale rows added since the Skip;
- (g) counts only R2/R3 changes since the ack's manifest.

So a skipped condition never fires again by itself every Monday.

Every report shows both comparisons: against the training manifest and
against the ack. An Approve deletes the ack.

### 3.4 The new-key probe

**What went wrong before r6.** The probe that started r5 (drift run
`dr-20261014-0115-502d48-check`) and the probe that judged r5's candidate
were built by the same code, and neither checked its rows. Every row ran on
one shared state, the vintage home of the exam, while the generator wrote
some rows for sections that home does not have (contentBlock-2/-3/-features,
newsletter). The engine refused those rows whatever the model did: 7 of 17
rows, so `stackedImageHeight` could reach at most 25 % and `ctaFontSize`
0 %. Every passable row passed. r4 probably already handled these keys, and
rule (d) fired on the harness, not the model.

**What the probe does now** (`souschef_lab/probe.py`, `drift.probe_rows`;
the retrain eval's `probe` set uses the same code):

- every row carries the state it was generated on (`site_state`) and runs on
  it, so the section it targets is there and its context matches;
- section types a key names but the state lacks (perksCards, perksIcons,
  newsletter …) get a home row with the settings that make the key visible
  (perks `stackedImage*` need `style: image-text`, `imagePosition: top`);
- a list with scalar items and no add op (gallery `imagesLink[]`) gets
  update cases, so those keys are probed at all;
- a case is dropped when it writes a field its own state hides, when its
  gate is closed, when its target is not on its state, or when the engine
  refuses its ops there (`engine=`, now passed by both callers);
- each probe turn's conversation is stored next to its result
  (`trajectory` in `exam/P-<k>.jsonl` and in the eval's rows), so a 0 % can
  be read instead of guessed at.

`drift/<run>/exam/P-plan.json` (the eval: `eval/info.json` `probe_dropped`)
lists every row asked (key, kind, target, request, expected keys, the
sections of its state) and, per key, the candidates and why any were
dropped. A key with no passable case has coverage 0: it is reported, and
never counted as dodged.

**Re-checking a past probe, offline** (no model, no network; `--engine`
needs node and `COOKIE_CUTTER_DIR`). It rebuilds the rows a past run asked,
pairs them with the stored results, marks a row unpassable when its section
is not on the shared state or the engine refuses its own expected ops, and
scores each key on its passable rows only. Verdicts: `false_alarm` (dodged
as run, not on its passable rows), `unmeasured`, `dodged`, `ok`. It also
lists what the fixed probe would ask:

```bash
D=dr-20261014-0115-502d48-check; R=rt-20261014-0708-7cac28-1184ff9a; T=$(mktemp -d)
gcloud storage cp $PIPE/runs/$R/prepare/manifest.json $PIPE/drift/$D/report.json "$PIPE/drift/$D/exam/P-*.jsonl" $T/
cd lab && .venv/bin/python -m souschef_lab.probe_recheck --manifest $T/manifest.json --report $T/report.json \
  --rows $T/P-1.jsonl --rows $T/P-2.jsonl --engine --json $T/recheck.json
```

To confirm on the model itself, run a check on the same manifest against
live r4 with the fixed lab image
(`gcloud workflows execute souschef-drift … --data='{"mode":"check","force":true}'`,
§8.3): about 2–3 h on the eval L4 (about US$3), no Claude spend. If r4 now
passes ≥ 75 % on these keys, r5's trigger was a false alarm.

**Cost.** The probe asks what it asked before plus rows for keys that had
none (the perks and gallery link keys): up to 4 rows a key, twice, at about
8 s a turn on the L4, so a dozen new keys add about 13 minutes to a check.

## 4. Baselines, noise and calibration

**Baselines are per model image digest:** `baselines/<sha256-digest>/baseline.json`.
They hold arm T's per-row results over K replicates, per-set means and SDs,
row pass rates `row_p`, probe results, and the service build they were
measured through.

**Noise.** Rows flip between runs at temperature 0.3. Per set,
`σ_run = sqrt(K/(K−1) · Σ p_i (1 − p_i))` over the row pass rates of K
replicates (the K/(K−1) factor makes the estimate unbiased; it doubles it at
K = 2, a candidate baseline); with K ≥ 3 the observed SD of per-run pass
counts is used when larger; never below 1 row. Arm D with m replicates
against a K-replicate baseline: `σ_Δ = σ_run · sqrt(1/m + 1/K)`. Before a
baseline exists the prior flip rate is 0.08 a row; the two replicate runs on
record disagree on 13 of 60 rows (about 0.11), which only matters while
auto-retrain is off anyway.

**Calibration.** A digest without a calibrated baseline (r4 at first, or a
model rolled out by hand) gets `drift calibrate`: arm T × 3 replicates plus
F0H × 1, about 3 h on the eval L4. The drift workflow does it by itself: a
`check` or `weekly` whose production digest has no calibrated baseline runs
`calibrate` first (run id `<run>-cal`, its own "baseline calibrated" notice),
then the check, in the same execution and under the same lock. Until a
baseline is calibrated:

- verdicts use a prior flip rate of 0.08 per row (`prior_flip_rate`);
- **auto-retrain is disabled**, whatever `pipeline_retrain_auto` says.

Every weekly run adds one arm-T replicate and keeps the newest 5, so σ follows
the service build as well.

**Seed noise is a different thing.** σ_run is how much **one** model's
answers vary between passes. Drift compares one model on two manifests, so
σ_run is the right noise for every rule here. A retrain compares two
**trainings**, which differ even on the same data and recipe; the retrain
bar's seed-aware tolerance adds σ_seed, measured by the seed-noise experiment
([RETRAIN.md R10](RETRAIN.md#r10-the-seed-noise-experiment)), twice: the
candidate and the live model are both seed draws, so
`tol = max(1, ⌈2·sqrt(σ_Δ² + 2·σ_seed²)⌉)`. If
`config/thresholds.json` carries `sigma_seed`, only the retrain bar reads it;
drift verdicts ignore it. An Approve installs the candidate's own paired-eval
rows (2 replicates on the deployed manifest) as the new model's baseline, so
calibration is not needed after a pipeline rollout.

**Recalibrate** after anything that changes how the same model answers:

- a new service build that changes the prompt, retrieval or route settings;
- a model rolled out by hand (`make model-rollout`), unless it is a rollback to
  a digest that already has a baseline;
- a change of Ollama settings in the model image;
- a change of the scoring policy that scores the exam or F0H (a round's
  `trigger.score_policy` / `font_policy` that turns a relaxation on for the
  eval, whose default is the r4 rules;
  [RETRAIN.md R2a](RETRAIN.md#r2a-scoring-policy-r4score)): the baseline's rows
  were scored under the old rule.

```bash
make pipeline-drift MODE=calibrate   # = gcloud workflows execute souschef-drift … --data='{"mode":"calibrate","force":true}'
```

### Manual rollbacks

Drift judges the model the production service actually serves (its digest,
`--model-digest`): the revision its `trafficStatuses` sends the largest
share to. A revision deployed with `--no-traffic`, or left at 0 % by a pinned
split, is not "serving" however new it is. `make model-rollout` sends all
traffic to the revision it deploys. When that is not `state/model.json`'s model but one in its
`history` (a `make model-rollout` back to an earlier model), the check judges
that model on its own training manifest and baseline and posts a
`drift_notice` ("production runs X, not Y"). Make it permanent by writing the
history entry back as `state/model.json` (keep the history), or roll forward
again. A digest the pipeline has never seen (an image rolled out outside the
pipeline) stops the behavioural modes with a configuration error until a
person writes its `state/model.json` entry (`tag`, `digest`,
`training_manifest`, `trained_at`, and `training_cases` if known). An
approved model's training cases and dataset are kept under
`cases/trained/<candidate>/`, which never expires, so a model can be judged
for as long as it is live.

**Reference numbers, not the baseline.** r4's own eval (student alone, H100,
no fallback): 212/252 on the real-phrasing exam with today's contexts, F0H
220/233, unseen keys 154/164. The baseline is measured on the L4 through the
deployed service build, which is what merchants get, so expect it to differ.

## 5. Where the exam runs

The harness (`harness.py`) is the in-job twin of `vm-solo.sh serve`:

1. Postgres 16 from `initdb` in `$TMPDIR/pg`, on `127.0.0.1:5433`.
2. The **deployed service binary** from the lab image (`/opt/sous-chef/bin/sous-chef`)
   on `:8091` with `MIGRATE_AT_BOOT=on`, a random per-run `SERVICE_KEYS`,
   `BUILD_ID=<build>-harness-<run>`, `DEFAULT_ROUTE=current`,
   `MODEL_ROUTE_ALLOWLIST=current,candidate`, `MODEL_CALL_TIMEOUT=180s`,
   `MAX_CONCURRENT_TURNS=4` and **no Anthropic key**.
3. `MODEL_ROUTES` = the deployed blueberry-muffin route (`infra/envs/int/main.tf`
   `inhouse_route`: messages-compatible, `auth: id-token`, `model: sous-chef`,
   `max_tokens: 1024`, `stream: false`, `temperature: 0.3`, `prime` and
   `fast_finish` on) **minus `fallback`**, once per arm, each pointed at a GPU
   service URL.
4. ID tokens come from the job's metadata server through the service's own
   `model/idtoken.go`.
5. Before asking anything it asserts `/version`: every arm has
   `fields_context: true` and `fallback: null`.

The model is the **live model image** on `sous-chef-model-eval`, whose image the
workflow sets to the production service's digest before each check: the
digest of the revision production **serves** (the largest share in its
`trafficStatuses`), not merely its latest ready revision. The patch is waited
out on the eval service itself (generation observed, not reconciling, the new
revision ready; [PIPELINE.md §2](PIPELINE.md#2-moving-parts)). If the eval
service's last revision failed (no L4), the next check patches it with
`forceNewRevision`, so a stockout costs one check, not every check until the
production digest changes.
`int`'s own service and `sous-chef-model` never change for a check.

**If the eval service cannot get an L4 within 15 minutes**, the check is
deferred to the night window (09:00 UTC) and runs against `sous-chef-model`
itself. Its `OLLAMA_NUM_PARALLEL=1` makes merchants queue behind the exam
there, which is why it waits for the night.

**One exam at a time.** `state/locks/gpu-eval.json` (a GCS object created with
`ifGenerationMatch=0`, carrying `expires_at`) guards the eval service; a paired
eval of a retrain takes it too, and a check that finds it held waits for the
next poll. The **workflow** holds it and runs the drift job with `--no-lock`
(the job takes the lock itself only when you run it by hand without that
flag). The workflow also passes `--model-digest` (the production service's
ready digest, which keys the baseline) and `--retrain-auto` (the switch, which
only changes the message's wording). A job that finds the eval service
unavailable (`status: eval_unavailable`) is deferred to the night window the
same way.

**Durations** (*recorded*: about 8 s per turn on the L4 through the speed build).
Arm D is 252 rows ≈ 290 turns ≈ 40–60 minutes with 2 workers on one GPU.

| trigger | runs | about |
|---|---|---|
| every 15 min | `drift poll`: GET deployed; on a new hash, store `manifests/<hash>.json` and run `drift static` | 40 s; 2 min with static |
| 30 min after the last manifest change | `drift check`: arm D × 2 and the probe | 1.8–3 h |
| Monday 05:00 UTC | `drift weekly`: arm T × 1 (baseline refresh), arm D × 2, probe, F0H on both arms; `weekly_all_clear` or the verdict | 3.5–4.5 h |
| a digest without a baseline | `drift calibrate` | about 3 h |

The poll writes `state/deployed.json` with `pending_check_at`; a poll at or
after that time runs the check. A newer change pushes `pending_check_at` out
again, so the debounce needs no second schedule. Every write of
`state/deployed.json` is conditional on the generation read, so a poll that
overlaps the end of a check never brings back the `pending_check_at` the check
cleared (no duplicate checks).

## 6. The engine snapshot

Ground truth (which writes a request should produce, which contexts the
browser builds) comes from cookie-cutter's engine
(`launcher/client/src/sousChef/ops/headless.mjs`) and its browser defaults.
GCP has no Bitbucket access, so the engine is **vendored**: a tarball of the
launcher client, `packages/` and `node_modules` at one commit, stored at
`$PIPE/engine/<sha>/{snapshot.tar.zst,engine.json}` and listed in
`engine/index.json`.

At start every job picks the snapshot of the deployed manifest's
`generated_from.sha`, checks its sha256, extracts it, dumps the browser
defaults from it (`SOUS_CHEF_BROWSER_DEFAULTS` points at the dump), and checks
the engine fingerprint against `engine.json`.

**Engine refresh.** When the launcher's commit has no snapshot, `engine_stale` is posted with
the mention and the command. Drift checks continue on the newest snapshot;
a retrain refuses to start, because its ground truth would not match the
launcher. To vendor (a person or the main session, from a cookie-cutter
checkout that has the commit; no spend):

```bash
export PIPELINE_BUCKET=$PIPE
bash lab/tools/engine-vendor.sh ~/projects/breadstack-cookie-cutter <sha> --upload
# the same through make:  make -C lab engine-vendor SHA=<sha> CC=~/projects/breadstack-cookie-cutter UPLOAD=1
make -C lab engine-index                  # print engine/index.json
```

The script runs `git archive <sha>` (it never touches your working tree),
`npm ci --ignore-scripts`, records `engine.json` (sha, fingerprint, node,
browser-defaults sha256, tar sha256), and uploads. A follow-up in cookie-cutter's
CI will publish the snapshot on every int deploy; until then this is the one
manual step the suite has.

## 7. The deployed-manifest endpoint

`GET /v1/manifests/deployed?line=int[&body=1][&days=30]`, behind the service
key like every `/v1/*` route (README §1).

**Why it exists.** The `manifests` table is content-addressed and keeps only
`build_line` and `first_seen`. "The newest manifest of the int line" can be a
lab registration: the lab registers with `X-Launcher-Build: lab`, `lab-r4` or
`lab-exam`.

**How it decides.** Two signals:

1. **Registrations.** The launcher registers on every boot, and from migration
   0006 every registration is recorded in `manifest_registrations`
   (`hash, launcher_build, source, first_at, last_at, registrations`), new
   hash or not. A launcher rollback re-registers the older hash, so the
   newest registration follows rollbacks.
2. **Real turns.** The newest turn on a non-lab site
   (`conversations.site_id NOT LIKE 'lab-%'` and `turns.build_id NOT LIKE 'lab%'`).

**Lab or launcher.** A build id (a registration's `X-Launcher-Build` or a
turn's `buildId`) is the lab's when, trimmed, it is empty, `dev` (a local
launcher built without `LAUNCHER_BUILD_ID`) or matches `^lab([-_.:].*)?$`
case-insensitively; everything else is the launcher's. That is why every lab
tool registers with a `lab…` build id and asks with `lab-…` site ids, and the
collector's `--build-id lab-<run>` matters.

`deployed` is the launcher hash with the **newest registration**. Real turns
decide only when the line has no launcher registration at all (hashes from
before migration 0006), newest turn first: a merchant tab opened before a
deploy keeps sending the old hash for hours, so turns would lag a deploy or a
rollback. The endpoint therefore answers from day one and follows deploys
once the launcher has re-registered. The response lists up to 5 candidates;
404 `no_launcher_manifest` means neither signal exists in the window.

```bash
URL=$(gcloud run services describe sous-chef --project $PROJECT --region $REGION --format='value(status.url)')
curl -s -H "Authorization: Bearer $(gcloud secrets versions access latest --secret sous-chef-pipeline-service-key --project $PROJECT)" \
  "$URL/v1/manifests/deployed?line=int" | jq '{hash, launcher_build, source, last_registered_at, last_real_turn_at, candidates}'
```

## 8. Running it by hand

### 8.1 Locally, no network (`DRY=1`)

The in-process fake service with the oracle, as the r4 rehearsals ran:

```bash
cd lab
make drift-dry                                   # poll → static → check on fixtures, fake service, oracle
```

### 8.2 Locally against int (a person; no model spend, but it reads int with the key)

```bash
cd lab
export SOUS_CHEF_URL=$(gcloud run services describe sous-chef --project $PROJECT --region $REGION --format='value(status.url)')
read -rs SOUS_CHEF_KEY && export SOUS_CHEF_KEY      # the pipeline key; never on argv, never echoed
.venv/bin/python -m souschef_lab.pipeline drift poll   --run-id dr-manual-$(date -u +%Y%m%d%H%M) --bucket $PIPE
.venv/bin/python -m souschef_lab.pipeline drift static --run-id dr-manual-… --bucket $PIPE
```

The behavioural check needs a GPU service URL and an identity allowed to
invoke it, so run it in GCP:

### 8.3 In GCP

```bash
# the whole thing, as the schedule would (force = ignore the debounce); execute returns at once
gcloud workflows execute souschef-drift --project $PROJECT --location $WF_REGION --data='{"mode":"check","force":true}'
# one job by hand (the workflow does the eval-service image and the lock for you; this does not)
gcloud run jobs execute souschef-drift --project $PROJECT --region $REGION --wait \
  --args=drift,static,--run-id,dr-manual-$(date -u +%Y%m%d%H%M),--bucket,$PIPE
```

Modes: `poll`, `check`, `weekly`, `calibrate`. Job subcommands: `drift poll`,
`drift static`, `drift check [--confirm]`, `drift weekly`,
`drift calibrate [--replicates 3]`, each with `--run-id`, `--bucket`
(`gs://…`; a bare bucket name is read as `gs://` too, never as a folder),
`--eval-url`, `--model-digest`, `--retrain-auto on|off`, `--no-lock` and
`--smoke`. The workflow posts every entry of the result's `notifications[]`
(ids `<run>/<kind>/<n>`) except `heartbeat`, which it sends itself once a UTC
day with the state tiles' `details.model / deployed / verdict`. Exit codes: 0 = finished (the verdict is in
`drift/<run>/result.json`), 1 = failure (retried, then `stage_failed`), 2 =
configuration or precondition error (no retry).

## 9. Reading a drift report

| where | what |
|---|---|
| Slack | the headline: deployed vs training manifest, exam on today's manifest vs baseline (Δ, σ, McNemar p, confirmed or not), new keys missed, stale rows, wording changes |
| thermometer `/sous-chef/pipeline` | the feed, current model, deployed manifest, last check and verdict, next weekly run |
| `$PIPE/drift/<run>/result.json` | `{verdict, deployed, changed, debounce_until, notification, retrain_reason}` |
| `$PIPE/drift/<run>/static.json` | the classes, renames, impact columns, field lists (500 per class at most) |
| `$PIPE/drift/<run>/report.json` | per-set arms, Δ, σ, McNemar, the rule lines that fired, both comparisons when an ack exists |
| `$PIPE/drift/<run>/exam/*`, `logs/*` | the exam rows as run (probe rows `P-<k>.jsonl` with each turn's `trajectory`), the probe's plan `P-plan.json` (§3.4), the harness and service logs |

```bash
gcloud storage ls $PIPE/drift/ | tail -5
gcloud storage cat $PIPE/drift/<run>/report.json | jq .
gcloud storage cat $PIPE/state/deployed.json | jq
```

## 10. Heartbeat and alerts

| signal | how |
|---|---|
| daily heartbeat | the first poll after 00:00 UTC posts `heartbeat` to the thermometer (stored, never in Slack) |
| `heartbeat_missed` | the thermometer posts it (with the mention, once a day at most) when no heartbeat has arrived for 36 h: a broken scheduler, workflow or project |
| `stage_failed` | a drift run that fails: the stage, error class, a scrubbed message, the execution id, a playbook anchor (with the mention). One failure streak (the same stage and error class in a row, `state/drift-failing.json`) alerts once a day at most; a **poll** alerts from its 2nd consecutive failure (one bad answer from int is also retried inside the job), a check, weekly or calibrate run at once. The first good run after an alerted streak posts a `drift_notice` "recovered" (no mention). |
| stale retrain lock | a `retrain` verdict while `state/locks/retrain.json` is past its `expires_at` (a round with no progress for 72 h): `stage_failed` with the mention, once a day; the lock is never broken automatically |
| failed round | a `retrain` verdict while `state/failed-round.json` exists starts nothing; the message says which round to resume ([RETRAIN.md §R8](RETRAIN.md#r8-debugging-a-failed-run)) |
| the thermometer down | a message it does not take waits in `wf/outbox/` (only the workflow account writes `wf/`); every poll re-sends the outbox first. A 401/403/404 (key or switch) waits too; only 400/409/413/422 (the message itself refused) is dropped, logged |
| backstop | `pipeline_alert_emails` (required): Cloud Monitoring emails on any failed workflow execution and on any "not delivered" log line |

## 11. Lessons that apply here

- **bash, not zsh**, for any loop that reads exit codes (`PIPESTATUS`).
- **Name the route.** On int the in-house model is the *default* route; a
  turn without `model` goes to blueberry-muffin. The teacher routes are
  `default` (Claude Sonnet 5.5) and `opus`; the harness has only `current`
  and `candidate`.
- **Lab traffic must look like lab traffic**: `lab-…` site ids, `lab…` build
  ids. Otherwise the deployed-manifest endpoint counts it as the launcher.
- **Keys** are read from the environment, never on argv, never echoed.
