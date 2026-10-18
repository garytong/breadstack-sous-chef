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
| R2 | carry-forward of earlier rows, then teacher collection of the rest, 3 passes | job `souschef-collect` | about 1 h with the carry (estimate); 1.8–2.2 h without | `collect/*` |
| R3 | dataset + parity gate | job `souschef-dataset` | 15 min | `dataset/*`, `model/context.tar.gz` |
| R4 | SFT + export + stage | Vertex custom job | 2.4–2.8 h + any capacity wait | `train/{adapter,export,stage,progress.json,DONE}` |
| R5 | model image | Cloud Build | 15–25 min | `model/image.json`, `model:c-<id>` |
| R6 | candidate deploy, paired eval, report | `sous-chef-model-cand`, job `souschef-eval` | 2–3 h | `eval/*`, `report.json`, `decision.json` |
| R7 | decision wait, then roll out or park | workflow + thermometer | up to 14 days | `rollout.json` or `state/parked/…` |
| — | cleanup: drop checkpoints, unlock | workflow | 1 min | |

Total about 9–12 h from the drift to the decision message. Any failure posts
`stage_failed` with the mention, releases the lock and keeps every artifact.

**What changed for r7** (pre-registered: answer key `ask-rule-r7-2026-10-18`,
committed before the r7 round starts; nothing below changes after r7's eval is
seen). r6 (`c-rt-20261016-2102-3eeb97-eb908674`) went live on int on
2026-10-17 over a failed bar; it asks a question on clear relative requests
("make our logo smaller": 15/48 relative rows against r4's 4/48). The cause is
the training labels, not the teacher (`rules.go:94` already says to ask only
when a request gives "no value, no direction and no change to make"): r6's
ask rows went from 7 to about 300, many of them first turns that carry a value
or a direction. r6 also failed its new-key gate on 25 keys. No `service/` or
`infra/` change: `rules.go` stays byte-identical, so the carry keeps r6's rows.

| change | where | switch | section |
|---|---|---|---|
| **the ask rule** (`souschef_lab/askrule.py`): a request may be labelled `ask` only when it is VAGUE (no value, no direction, no action; image requests are the action `pick_asset`). An ask-first turn that is not vague becomes a WRITE case (`askfirst1r-…`, mode keys, with the direction when one is clear; on a page with several sections of the target's type the case focuses the source's section, so it is not a "which one?") **only on evidence** (`askrule.write_evidence`: the source oracle's literal is in the turn, a direction on a colour / number field, or a toggle named by its label); a clear turn without evidence is left out (`unsure`: "the headings look too plain" had become `headingFont` in one case and `buttonFont` in another), and one text relabelled to different fields leaves entirely (`conflicting_keys`); a pool ask that is not vague becomes a keys case on the same evidence (else dropped, `unsure`); "too X" / "not X enough" count as a direction only for a size, amount, spacing, speed or strength word, "more / less X" not for a style word ("more modern"), a colour word describing what is there ("the grey text", "looks so white") and reported speech ("people say …") give no value, and "rename … to X" / "should be X" give X even when the site shows a name X starts with (review fixes; every single-turn exam row now classifies as its answer key says, a test); an authored follow-up whose ask turn is not vague is rejected (`ask_turn_not_vague`); the dataset re-checks every kept ask row's rendered text, carried rows included (`ask_rule_violation`) | prepare, dataset | none (always on) | [R1](#r1-prepare), [R3](#r3-dataset-and-the-parity-gate) |
| **ask / write pairs**: each vague ask gets write twins in its split group (`meta.ask_pair`): F8 vague asks two (`contrast1-F8-12xx`: "…, it's too light" / "…, try #hex") and, on a colour / number field, a third in the OTHER direction (`contrast1-F8-13xx`: the logo height had only "a bit smaller" next to the exam's "bigger"), ask-first asks one (`askpair1-…`: both turns as one request). An ask row whose twin has no kept row is dropped (`ask_unpaired`); clarify, which-one and no-title create_page asks are exempt | prepare, dataset | none | [R1](#r1-prepare), [R3](#r3-dataset-and-the-parity-gate) |
| **ask cap**: ask + clarify rows at most 3 % of the rows (r4 0.25 %, r6 7.8 %), floors per ask family (`complaint`, `which_one`, `page_title_missing` all; `text_missing`, `choice_missing` 25 each), over-cap rows dropped as `ask_cap` (their twins stay) | dataset | `"trigger": {"ask_share_max": 0.03}` (0 = off), or `SOUSCHEF_ASK_SHARE_MAX` | [R3](#r3-dataset-and-the-parity-gate) |
| **exam answer key `.k3`**: h022, p030, p047 become ask-allowed; h006 requires `bodyFont` (the footer body font allowed); p050 / p053 stay strict (an image request's answer is `pick_asset`; all 9 such rows would move together, never 2) | eval, drift | none | [R2a](#r2a-scoring-policy-r4score), [DRIFT.md](DRIFT.md) |
| **authoring names the instance and the twin**: a request on a page with several sections of the target's type names the one it means ("the Crafted with Care section") unless the view focuses it (`ambiguous_instance`); a "…2" field's request says "second" and its first twin's does not (`ambiguous_twin`). r6's 25 short keys failed at collect: the teacher rightly asked "which one?" | prepare | none | [R1](#r1-prepare) |
| **the four keys r6 could not answer**: a case of a section type the fixture home lacks takes its legacy twin's place (`hero` for heroBanner, its slides; r6 appended an empty `hero`, both "Hero Banner", and the teacher wrote the real one: `alias_instance` retires those); a write to one slide names it (`ambiguous_row`); the custom description tag is a picker of the case's store tags (`option_lists.tags`, the request names one: `tag_not_in_store` / `tag_not_named`); the Pack filter display is the whole Units / Grams setting, never text (`bad_value`, `bad_seed`; a case writing both kinds counts for it) | prepare, dataset | none | [R1](#r1-prepare) |
| **re-authoring the 25 short keys** (+ `packFilterDisplay.edibles`): listed addresses are targets again (reason `reauthor`, 14 kept positives each: 18 of them have no keep rate of their own or one below 0.65, where 10 broke even), r6's ambiguous drafts for them leave the round (`report.json` `ambiguous_superseded`), and their cases split train / val **per case** (`author:<address>#<case id>`) instead of per address: 9 of the 25 had their whole author group hashed into val, so r6 trained none of their rows (perksIcons.buttonOnImageOffset: 9 kept rows, 0 in train) | prepare, dataset | `"trigger": {"reauthor": [addresses], "positives_required": [addresses]}`, or `SOUSCHEF_REAUTHOR`, `SOUSCHEF_POSITIVES_REQUIRED` (comma lists) | [R1](#r1-prepare), [R3](#r3-dataset-and-the-parity-gate) |
| **r4 floor**: the bar also compares the candidate with r4's recorded calibration replicates (`<set>:vs-r4` rows, rule `r4_floor`: every set within tol and the pooled McNemar not significantly worse; the file is pinned by its own sha256 `496a5ca0…`, the model by its digest `f657…`; with `sigma_seed` measured the reference rows add **1**·σ_seed², r4 being one fixed model); a second, report-only comparison with r4 **re-run** in r6's round (`<set>:vs-r4_rerun`, rule `r4_rerun_floor`: same prompt, scorer and nearly the same manifest, pairs the `.k2` rows); report-only "asks on clear requests" per arm and exam class. How a holdout-only failure is read is fixed in advance: [Reading r4_floor](#reading-r4_floor) | eval | `config/thresholds.json` `r4_reference` (`"bar": false` reports only; `"rerun_bar": true` gates the re-run comparison too; `null` off) | [R6](#r6-candidate-and-the-paired-eval) |
| **launcher labels** (cookie-cutter branch `r7-labels`): in contentBlock, perksCards and perksIcons the second panel's fields read `"Buttons 2 › Button Border Color"`, `"Content 2 › Headline"` (39 labels; the first twin keeps its label) | launcher manifest | none | [Running r7](#running-r7-in-order) |

**What changed for r6** (r5, 2026-10-14, retrained r4's data and lost 3.5 rows
on holdout; the analysis found a training-run reroll read as a loss, a probe
no model could pass, and authoring that produced no positive cases). Each
change below says what a person runs and what it costs:

| change | where | switch | section |
|---|---|---|---|
| authoring targets real section instances, opens gates, rejects hidden writes, keeps the oracle's error text, tops up to 8 positives | prepare | none (always on) | [R1](#r1-prepare) |
| trainable pool cases that write a field the editor hides in their state are dropped after the rebase (`report.json` `hidden_write_dropped`; r5 trained 58) | prepare | on; `"trigger": {"pool_hidden": "keep"}` | [R1](#r1-prepare) |
| trainable cases that expect a bare e-mail address on a link field expect `mailto:` instead (`report.json` `mailto_fixed`; r4-F3-0413) | prepare | none (always on) | [R1](#r1-prepare) |
| carry-forward: an earlier round's accepted row is re-verified and kept instead of re-asking the teacher; a carried row that writes a field whose gate is closed is re-asked | collect | on in full rounds; `"trigger": {"carry": "off"}`, `"carry": "required"` (a carry that cannot run fails the stage instead of asking everything), `"carry_hidden": "keep"` | [R2](#r2-teacher-collection) |
| scorer relaxations (collect: `mailto` one way, `same_control` only for the key the editor writes; `refused_corrected` and `acted_first_turn` off) and the font policy; the **eval scores with the r4 rules** (every relaxation off) | collect, dataset, eval | `"trigger": {"score_policy": {…}, "font_policy": "…"}` | [R2a](#r2a-scoring-policy-r4score) |
| new-key positives gate, the font-pair count, `stats.json` rewritten after every drop | dataset | `"trigger": {"min_new_key_positives": N, "new_key_positives_waive": [addresses], "font_pair": "drop"}`, or `SOUSCHEF_MIN_NEW_KEY_POSITIVES`, `SOUSCHEF_NEW_KEY_POSITIVES_WAIVE`, `SOUSCHEF_FONT_PAIR` | [R3](#r3-dataset-and-the-parity-gate) |
| training arms: from-base seeds **and** a warm start from r4 in the same round, each scored on the same validation rows, the best kept (the user's decision, 2026-10-15); every value and the warm adapter checked by the dataset stage before any GPU; an A100 rung trains no extra arm; an arm that would not end inside the 12 h window is not started | train | `"trigger": {"train": {"arms": ["base:3407", "warm:3407", "base:2"], "init_adapter": "base/adapters/r4-sft"}}` (the dataset stage checks it and writes it to `runs/<id>/train-config.json`), or that file by hand; unset = one arm, `base:3407`, as r4/r5 | [R4](#r4-training) |
| the probe runs each row on its own state, checked by the engine, and keeps its conversations | eval (and drift) | none | [R6](#r6-candidate-and-the-paired-eval), [DRIFT.md §3.4](DRIFT.md#34-the-new-key-probe) |
| a seed-aware bar tolerance, measured by the seed-noise experiment; both models count as seed draws (`2·σ_seed²`) | eval | `config/thresholds.json` `sigma_seed` (`bar.py` reads it; empty by default) | [R6](#r6-candidate-and-the-paired-eval), [R10](#r10-the-seed-noise-experiment) |
| contrast cases for look-alike fields (F8), generated from field metadata, no model call; a case that writes an exam item's field or its look-alike with the exam's value or page is dropped | prepare (every full round) | on; `"trigger": {"contrast": "off"}` | [R1](#r1-prepare) |
| **create_page and dropdown choices** (tools first, 2026-10-15: they train in r6): F8 adds 46 `create_page`-family cases (create, slug, draft, `in_nav`, create + sections, collisions, reserved and non-canonical slugs, a template without custom pages, rename / delete still `pages`, "add a section, not a page", "our LinkedIn page" is a social link) and 36 listing cases (short, over 15, cut at 100, empty, unknown list; carousel and plain asks); scored on the op (`page` mode) and on the reply's names (`list` mode); the 17 pool rows that asked to create a page and were filed as `pages` are relabelled; exam p137 is re-keyed `p137.k2` and scored in the eval's report-only `capability` set (with about 15 built create / list cases), never in the bar; carried tag / brand rows written without their derived id are re-asked | prepare, collect, dataset, eval | contrast on/off as above; `"trigger": {"page_relabel": "off"}`; thresholds `capability_report: false` drops the built report cases | [R1](#r1-prepare), [R2a](#r2a-scoring-policy-r4score), [R6](#r6-candidate-and-the-paired-eval) |
| **tool results and contexts as the live browser sends them** (r6-4 review): a tool result's `sectionId` is the call's own target section (turns.js `toToolResult`), never the engine's new-section id; every lab context's categories carry their id as text (`fixtures/context.json`), as live contexts do | collect, carry, every derived context | none | [R2](#r2-teacher-collection), [DRIFT.md](DRIFT.md) |

Every `trigger` key above goes in the workflow input's `trigger` object (it is
copied into `run.json`, and a resume reuses the stored trigger: retrain.yaml
`context`). **Change a running round's switches in its `run.json`, never on a
job.** Edit `trigger` in `runs/<id>/run.json`, then `resume_from` the stage
that reads it:

```bash
gcloud storage cat $W/run.json | jq '.trigger.new_key_positives_waive = ["<address>"]' > /tmp/run.json
gcloud storage cp /tmp/run.json $W/run.json --content-type=application/json
gcloud workflows execute souschef-retrain --project $PROJECT --location $WF_REGION \
  --data="{\"mode\":\"full\",\"resume_from\":\"dataset\",\"run_id\":\"$RUN\"}"
```

The stages also read environment variables (`SOUSCHEF_SCORE_POLICY`,
`SOUSCHEF_FONT_POLICY`, `SOUSCHEF_FONT_PAIR`,
`SOUSCHEF_MIN_NEW_KEY_POSITIVES`, `SOUSCHEF_NEW_KEY_POSITIVES_WAIVE`,
`SOUSCHEF_ASK_SHARE_MAX`, `SOUSCHEF_REAUTHOR`, `SOUSCHEF_POSITIVES_REQUIRED`,
`COLLECT_CARRY*`, `TRAIN_*`), which win over the trigger. Use them only as a
per-execution override on a stage run by hand (`gcloud run jobs execute …
--update-env-vars=…`). Never set them with `gcloud run jobs update`: the value
would stay on the job template for every later round, and the next
`terraform apply` would silently revert it.

### Running r7, in order

r6 serves int throughout. Nothing here needs a Terraform apply, a workflow
change or a hand-made service deploy. Who does each step is named; "a person"
is a step an agent does not take (a commit, a merge, a bucket write).

0. **Freeze cookie-cutter `master`** from step 2 until the r7 decision. A
   launcher deploy mid-round makes the round stale (`stale_round`, Skip
   recommended) or, without a new engine snapshot, `engine_stale`.
1. **Commit the lab** (a person; the exam `.k3` amendments, the ask rule, the
   ask cap and the r4 floor are pre-registered with this commit). If the
   commit is not dated 2026-10-18, change `exam.ANSWER_KEY_R7` to the commit
   date first. Pushing sous-chef `main` makes CI build `sous-chef-lab`,
   **and a new `sous-chef-trainer`** (`images/trainer/tag.sh` hashes every
   file under `lab/` except the tests, so the r7 lab commit is a new trainer
   tag, the ~15 GB build), and roll the int service (the same `service/`
   code, rebuilt). The Python packages are pinned (`train-constraints.txt`)
   and the layer is cached, so the risk is low; still, before the round
   either check that CI's trainer build passed and its `train-env ok` line
   lists the same versions as r6's, or run a SMOKE (US$10–15, [SMOKE](#smoke)).
2. **Merge the launcher labels** (a person: cookie-cutter `r7-labels`, off
   `bitbucket/master` 26a6356e, pushed to Bitbucket `master`). CI deploys the
   launcher on int; `state/deployed.json` should show the manifest hash
   `3914e2e4…` (only 39 labels change from r6's `eb908674…`).
3. **Vendor the engine snapshot right after the deploy**, or the round refuses
   to start (`engine_stale`). The script's work folder must not be on the
   nearly full `/tmp` tmpfs (the r7 builders hit `Errno 122` there), and the
   checkout must have the merge commit:

   ```bash
   git -C ~/projects/breadstack-cookie-cutter fetch bitbucket
   mkdir -p ~/.cache/engine-vendor-tmp
   TMPDIR=~/.cache/engine-vendor-tmp bash lab/tools/engine-vendor.sh ~/projects/breadstack-cookie-cutter <merge sha> --upload
   ```

   (DRIFT.md §6). No sous-chef fixture re-capture.
4. **Copy r4's baseline to its own place** (a person; a bucket write) before
   the Mistral cleanup deletes `experiments/mistral/`:
   `gcloud storage cp $PIPE/baselines/sha256:f657e11053e3baf801b489421637b8a0aaccf450951046ec4c9853ca0c7287b4/baseline.json $PIPE/baselines/reference/r4/baseline.json`.
   Until then the eval reads the live `baselines/sha256:f657…` copy (the same
   bytes on 2026-10-17). Re-copy r4's adapter (`base/adapters/r4-sft`) before
   about 2026-11-10.
5. **Pre-flight (reads only)**:
   - `state/deployed.json` shows `3914e2e4…`; `engine/index.json` lists the
     merge sha;
   - the r4 reference **file's** sha256 is `496a5ca0084ca3da560b4c9cf4bdb9929a2711492771a2313f11f0203a5d585c`
     (`gcloud storage cat $PIPE/baselines/reference/r4/baseline.json | sha256sum`;
     thresholds `r4_reference.file_sha256` pins it, a copy with other bytes
     is skipped). `f657…` is the **model** digest inside the file (its
     `digest` field), not the file's hash;
   - no bucket override of `r4_reference` or `sigma_seed` in
     `config/thresholds.json` (the bucket file is merged key by key over the
     committed one, so an override would quietly change a pre-registered floor);
   - **the rules are unchanged since r6's teacher build**:
     `git diff --stat 28491da HEAD -- service/` is empty. The carry compares
     `system[0]` of r6's rows with the round's rules (`rules_changed`); r6
     carried nothing for exactly that reason and asked the teacher about all
     4,970 cases (US$177). On 2026-10-17 the rules rendered on `3914e2e4`
     with the current service binary equal r6's dataset rows' (sha256
     `631134813ac4…`). `carry: "required"` only stops the round when the
     carry cannot run, **not** when it carries nothing: watch the collect
     stage's first progress line (`carry: … candidate row(s)`) and cancel the
     round if the live model's source is `skipped: rules_changed`;
   - `state/model.json` is r6; no `state/locks/retrain.json`.
6. **r7 round** (expected about US$90–150 with two arms, 11–15 h; worst case
   about US$250 if the carry carries nothing):

   ```bash
   REAUTHOR='["globalPanels.FooterSection.footerBodyFont","globalPanels.FooterSection.footerLinkFontSize",
     "pages.blog.readMoreColor","pages.product-details.productDetail.customDescriptionTagName",
     "pages.products.packFilterDisplay","pages.products.packFilterDisplay.edibles",
     "sections.contentBlock.buttonBgColor2","sections.contentBlock.buttonBorderColor2",
     "sections.contentBlock.buttonOnImage","sections.contentBlock.buttonOnImageOffset","sections.contentBlock.buttonTextColor2",
     "sections.contentBlock.centerContentSwap","sections.contentBlock.stackedImageFullWidth","sections.contentBlock.stackedImageGap",
     "sections.contentBlock.stackedImageHeight","sections.contentBlock.title2Color","sections.contentBlock.title2FontSize",
     "sections.featuredCategories.subDisplay","sections.hero.contentWidth","sections.hero.slides[].btnSize",
     "sections.perksCards.body2FontSize","sections.perksCards.title2Color","sections.perksIcons.buttonBorderColor2",
     "sections.perksIcons.buttonOnImageOffset","sections.perksIcons.buttonOnImagePosition","sections.perksIcons.stackedImageFullWidth"]'
   REQUIRED=$(echo "$REAUTHOR" | jq -c 'map(select(. != "pages.products.packFilterDisplay.edibles"))')
   gcloud workflows execute souschef-retrain --project $PROJECT --location $WF_REGION --data="{\"mode\":\"full\",
     \"trigger\":{\"reason\":\"r7\",\"carry\":\"required\",\"reauthor\":$REAUTHOR,\"positives_required\":$REQUIRED,
                \"ask_share_max\":0.03,
                \"train\":{\"arms\":[\"base:3407\",\"warm:3407\"],\"init_adapter\":\"base/adapters/r4-sft\"}}}"
   ```

   26 keys are re-authored (r6's 25 short keys plus
   `products.packFilterDisplay.edibles`, whose 8 r6 rows taught free text);
   **25** must reach 6 train rows — all of r6's short keys, the four that an
   earlier draft of this plan waived included (`positives_required` must be
   the list itself: a string is read as one comma-separated list).
   `.edibles` is re-authored only to replace its bad rows; it is not
   required. The four keys' r6 cases were unanswerable, not the teacher's
   miss, and the lab now authors them right:
   - **`hero.contentWidth`, `hero.slides[].btnSize`**: r6's case home held
     the fixture's `heroBanner` plus an empty appended `hero` row, both
     "Hero Banner" in the editor; the teacher wrote the real one. A case of a
     section type the fixture home lacks now gets a legacy site
     (`author.alias_home`: `hero` in heroBanner's place, its 2 slides), r6's
     17 two-hero cases leave the round (`alias_instance`), and a slide write
     names the slide (`ambiguous_row`).
   - **`productDetail.customDescriptionTagName`**: a tag picker. Each case now
     carries its store's tags (`option_lists.tags`), the request names one
     and the oracle writes its exact name (`tag_not_in_store`,
     `tag_not_named`); r6's 10 invented-tag cases leave the round.
   - **`products.packFilterDisplay`**: the whole Units / Grams setting, never
     a label. Writes set both kinds to Grams, resets clear it from Grams on
     both, a case writing both kinds counts for the setting; r6's string
     writes leave the round (`bad_value`, `bad_seed`; 9 + 8 `.edibles`).

   Estimated train positives (simulated on r6's data): tag 10.4,
   pack 10.4, hero width 12.5, slide button 11.6 (low case 6.7 / 6.7 / 8.7 /
   7.8). A key short of 6 stops the dataset stage: read its
   `reject_details` and waive it then (R3) rather than before. The warm arm
   starts from **r4**, not r6 (r6's adapter is the over-asker). `base:2` as a
   third arm adds about 2.2 H100-hours (about US$15–25).
7. **Read the report**: the bar (vs r6) and `r4_floor` (vs r4's recording,
   [Reading r4_floor](#reading-r4_floor)); `r4_rerun_floor` (report only);
   "asks on clear requests" (target: relative about r4's 4/48); the ask_ok
   rows (at least r6's); F9H (at least r6 − tol); `stats.json` `ask` (share
   about 3 %, `violations_kept` 0, `unpaired_kept` 0); `new_key_positives`.
   Decide on the thermometer. **Recalibrate drift only on Skip**: on Approve,
   r7's candidate baseline (with the `.k3` rows, on `3914e2e4`) becomes the
   drift baseline by itself; on Skip, r6's baseline still scores h022, p030,
   p047 and h006 on the old key, so recalibrate it (DRIFT.md, the `.k3` rows).

#### Reading r4_floor

Fixed before the round, so a result cannot change how it is read.
`r4_floor`'s tolerance is eval noise only while thresholds `sigma_seed` is
empty, and a retrain of the same data moves the holdout too: r5, a plain
retrain of r4's data, lost 3.5 holdout rows to the reroll. A no-change
retrain could therefore fail this floor. The rule's detail says "within a
reroll's reach" when every failing set is within tol + 3.5 rows.

- **A regression** (Skip): `r4_floor` fails on more than holdout, or holdout
  falls more than tol + 3.5 rows below r4 (r6: −6.3 against tol 2), or the
  relative "asks on clear requests" is above 8/48 (twice r4's 4/48), or
  `r4_rerun_floor` also fails beyond its tolerance + 3.5.
- **Inconclusive, not a regression**: a holdout-only failure within reach,
  with relative clear asks at or below 8/48, and the lost holdout rows not
  asks on clear requests (the report's per-row clear-asks keys). The decision
  then rests on the live bar against r6, which must pass; Approve is a
  person's call over the failing rule, and the thermometer note says so.
- Once the seed-noise experiment fills `sigma_seed`, the reference rows add
  **1**·σ_seed² (r4 is one fixed model; the live bar adds 2·σ_seed², both
  models being draws), and this reading is replaced by the rule itself.

### Running r6, in order

The r6 code is uncommitted when this is written. Each step names its cost;
nothing here needs a Terraform apply or a workflow change.

1. **Deploy.** Commit and push, so CI builds the new `sous-chef-lab` image
   (every stage's code changed) and `sous-chef-trainer` (the slow, about
   15 GB build; `pipeline-train.sh`, `train.py`, `selftest.py` changed).
   Without the new trainer image, `TRAIN_SEED(S)` and `INIT_ADAPTER` are
   ignored, and a `train-config.json` with `arms` fails the train stage at
   step `settings` (`unknown key`). Cloud Build minutes only. Deploy the
   `sous-chef` service in the same push: its prompt carries the new font
   ("the font" = every site font) and ask-when-vague rules that collect's
   teacher must see, and the live r5 model gets them on deploy. The service
   also adds `create_page`, which the editor's engine must support first
   (the cookie-cutter launcher change, and the lab's vendored engine
   snapshot: prepare measures every create_page case on it, and an engine
   without the op drops the creates and declines). Exam row p137 ("create a
   new page for events") is re-keyed `p137.k2` for it (R2a). The prompt fixtures (`lab/fixtures/prompt/`, `parity/served.json`)
   were re-captured from this service tree on 2026-10-15; re-capture again
   (`tests/capture_real.py --from`, then `make -C lab parity-write`) if
   `rules.go` or the tools change before the push.
2. **Put r4's adapter in the pipeline bucket** (a person; minutes, no
   GPU). The warm arm starts from it, and the dataset stage refuses an
   `init_adapter` that is not there. The procedure is in
   [R4, "r4's adapter in the pipeline bucket"](#r4s-adapter-in-the-pipeline-bucket).
   Do it before about 2026-11-10, when the training bucket's copy expires.
3. **SMOKE round with both kinds of arm** (about US$10–15, 2–6 h):

   ```bash
   gcloud workflows execute souschef-retrain --project $PROJECT --location $WF_REGION --data='{"mode":"smoke",
     "trigger":{"reason":"r6 smoke","train":{"arms":["base:3407","warm:3407"],"init_adapter":"base/adapters/r4-sft"}}}'
   ```

   It proves the generation check, the arm selection and the warm-start load
   of r4's adapter on a real GPU before about US$50 of three-arm training.
4. **Seed-noise experiment, setup → train → build** (R10; about US$47–57 for
   two arms, no Claude spend). Training and builds can overlap the r6
   round's early stages.
5. **Drift re-check with the fixed probe** (about US$3, under 1 h; [DRIFT.md
   §3.4](DRIFT.md#34-the-new-key-probe)): confirms r5's trigger was the
   probe and that the fixed probe passes r4 where it should.
6. **Noise eval and analyze** (about 3 h of L4 an arm, one arm at a time):
   it cannot run while a round holds the retrain lock (about 13–17 h for a
   three-seed round), so run it before step 7 or after the round. Then copy
   `noise/<exp>/result.json` `thresholds_override` into
   `config/thresholds.json` if the decision rule says so.
7. **r6 round** with `gcloud workflows execute … --data` (`make
   pipeline-retrain` cannot pass trigger keys):

   ```bash
   gcloud workflows execute souschef-retrain --project $PROJECT --location $WF_REGION --data='{"mode":"full",
     "trigger":{"reason":"r6","carry":"required",
                "train":{"arms":["base:3407","warm:3407","base:2"],"init_adapter":"base/adapters/r4-sft"}}}'
   ```

   Two from-base seeds and one warm start from r4 (lr 5e-5), each on the
   whole dataset, scored on the same validation rows; the best goes to the
   eval and the decision message lists all three (R4). The warm arm is
   second so that a 12 h window cut (which drops the arms not yet started)
   still leaves one arm of each kind. About **US$165–190** and 13–17 h (R9):
   r6 carries nothing (the rules changed twice since r5's rows, `rules_changed`;
   the contexts changed too), so the teacher is asked every case.
   The font and vague-request decisions are the defaults (R2a): no scoring
   key is needed in the trigger.
8. **Recalibrate drift** after the r6 decision (minutes of L4): `drift
   calibrate --fresh-baseline` on the live model, or let the approved
   candidate's eval baseline replace it. Until then the 20 re-keyed exam
   rows (`.k2` ids: the 19 font and vague rows and p137) are left out of
   drift comparisons.

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
| eval | 10 rows per set, 1 replicate, both arms; the candidate arm's error rows (it is 4 updates from the base) are recorded in `eval/info.json` `tolerated_errors`, not fatal (the current arm's still fail the stage: `eval_errors`) |
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
   - 10–13 cases per field (r6; r5 asked 8): 2 verbatim-ish, 2 colloquial
     (aliases from `service/internal/fields/vocab.txt`), 1 short command, 1
     relative value (`keys`), 1 focus case (the section open, no target
     named), 1 two-turn follow-up, 1 reset (when the field can be reset), 1
     near-miss or out-of-editor where the neighbourhood invites one. A
     **gated** field (`conditional_on` / `hidden_when`, such as
     `contentBlock.stackedImageGap`, shown only while `imagePosition` is `top`)
     adds 2 `prereq` cases (the gate closed in the state; the oracle opens it
     in the same `set_fields`, gate key first) and 1 `gate_only` case (the
     merchant asks for the layout that reveals it). A new section, panel or
     page also gets 6 container cases.
   - **Op targets name a real section instance** of the fixture home
     (`contentBlock-1`), never the bare type: r5 targeted `"contentBlock"`,
     which no state holds, and kept 0 positive cases for
     `contentBlock.stackedImage*`. The instance whose state already opens most
     of the field's gates is preferred.
   - **A field short of 8 kept positive writes** (`MIN_POSITIVE`) after the
     re-ask gets a top-up ask (at most 3 rounds in all).
   - Model `claude-opus-5-5`, effort high, JSON-schema output, streaming,
     server-side fallback, prompt caching on the fixed rules, 8 at a time.
   - The rules are r4's: realistic merchant phrasing, no "preview", British
     spellings and typos in a share, no invented UI words. **Exam texts are
     never shown to the author.**
   - Validation, every rejection counted: keys exist on the deployed manifest;
     oracle ops apply through the engine with a non-empty diff; a literal value
     means `exact`, an authored or relative value means `keys`; **no write lands
     on a field the case's state hides** unless the case says so
     (`expected.expect_hidden`; r5 kept 22 perks rows that taught hidden
     writes); the `preview_claim` rule; the hold-out filter against the exam
     and the study's conversations; dedup against the pool. A rejected case is
     re-asked with the reason, then dropped. The full reason, the engine's
     error text included, is in `prepare/author-report.json` per target
     (`reject_details`) and in the re-ask (r5 kept only the prefix, so the
     gallery `imagesLink[]` failures could not be explained).
   - **Contrast cases (family F8, r6; `souschef_lab/contrast.py`).** Cases for
     the look-alike fields both r4 and r5 confuse, in 7 pairs: `links`
     (Typography link colour and nav size vs the Shop Menu's "Link Color" /
     "Link Font Size"), `fonts` (the site's body font vs one section's body
     copy font), `names` (a nav item's name vs a custom link vs checkout
     order-type text), `grid` (ProductCard `cardsPerRow` vs the products page
     `gridCols`), `footer` (`footerTextColor` vs `footerBodyTextColor`),
     `picker` (`locationPickerTitle` vs its description), `logo`
     (`logoMaxHeight` vs `…Mobile`). Each side is asked as a literal value, a
     relative change, a request that names the other side to rule it out, and
     with its panel open: 153 cases asked from 20 sides, 120 kept on
     `1184ff9a` (after the exam-answer rule below). They are
     **generated, not authored**: the words come from the manifest's labels,
     help and container names (never from the exam; `check_templates`
     enforces it), the values from the field's type and options, and the same
     manifest always gives the same cases. No model call, so no authoring
     cost. Every case is validated like an authored one (the oracle through
     the engine, a write behind a closed gate drops it, `keys` mode for an
     untyped value, generator hold-out keys dropped, the hold-out filter plus
     a margin of Jaccard 0.5 against any exam text, dedup) plus an
     **exam-answer rule**: a case that writes an exam item's field, or a
     look-alike of it (any side of the same pair), with the value that exam
     item states or, for a per-page key (`pageVisibility.<page>.…`,
     `orderTypeText.<page>.…`), for the same page, is dropped
     (`contrast-report.json` `exam_answer_clashes`; 18 on `1184ff9a`, from
     h019, h047, h056, p011, p080, p094). The wording can be far from the
     exam while the answer is the same: "change Shop to 'Menu' in the footer
     only" is Jaccard 0.33 from h019 "rename Shop in the navigation to
     Menu". Collecting them
     costs about US$0.03 a case asked, about US$4, once: their ids
     (`contrast1-F8-nnnn`) do not depend on the manifest, so carry-forward
     (R2) keeps an unchanged one in later rounds. **Prepare builds them in
     every full round** (not SMOKE) on the deployed manifest, after authoring,
     and adds them to `prepare/cases.jsonl` only — they are regenerated, never
     pooled (`prepare/contrast-report.json`; `report.json` `contrast`). Off for
     a round: `"trigger": {"contrast": "off"}`. To look at them by hand:

     ```bash
     cd lab && .venv/bin/python -m souschef_lab.contrast --manifest $WORK/manifest.json --out $WORK/f8.jsonl \
         --why <study folder> --pool $WORK/cases.jsonl --report $WORK/f8-report.json
     ```

     The F0H and F9H hold-outs and the leak check must not drop on a round
     that adds them; the per-key slices of paraphrase and holdout for the
     pairs' keys show whether they helped.
   - **create_page and dropdown choices (F8, r6-4; contrast.py `PAGE_JOBS`,
     `LIST_JOBS`).** Two capability groups ride in F8, built the same way (no
     model call, same validation), ids from `contrast1-F8-1001` (pages) and
     `-1101` (lists), tagged `meta.capability`:
     - `create_page` (46): create with a title only (6), with a canonical slug
       (3), as a draft (3), in the header menu (3), with a section in the same
       reply (4: the slug is passed and the section added on it); collisions
       with a page the site has (4: a custom page the context lists, a
       built-in page) and the same with a section add (3: the batch stop must
       block the add after the refused create); reserved slugs (3: menu,
       sitemap, launcher — the engine suggests another, the right answer asks);
       non-canonical slugs (3: the canonical form of the asked slug); a
       template that can't show custom pages (2, premium template `plantrepublic`); rename /
       delete / duplicate / address still report_unsupported `pages` (4); and
       "add a section, not a page" (4, keys mode: a create_page there is a
       stray write); "our LinkedIn / Leafly / Google Business / X page" with
       its URL is a social link (4, keys mode on `socialLinks.*`; the exam's
       own networks — weedmaps, instagram, facebook, tiktok — are left to it).
       Every create's oracle must apply on the round's engine and
       every decline's attempt must be refused with the expected code
       (`E_SLUG` / `E_TARGET`), so an engine without the op drops those cases
       (`contrast-report.json` `capability_dropped`). A section add on a page
       whose `<page>.sections[].id` is a generator hold-out key is dropped like
       any hold-out write (home and faq are; the jobs use pages that are not).
       No case names exam p137's page (`page_exam_clash`, which also checks the
       pool's relabelled creates).
     - `list_options` (36): "what tags / brands / categories can I use for this
       carousel" with the carousel focused, or a plain "which brands do we
       carry?" with nothing focused (`focus: plain`, 14 of them), and the
       store's list in the case's `option_lists` (tags, brands) or
       `context_extras` (categories), laid over the context fixture when the
       context is derived: short (named in full; 1–15 entries), over 15 (15
       names then "...and N more"; 16–100), cut at the context's 100
       (`tagsTruncated`: N from the total, and the dropdown named; 101–160),
       empty (`[]`: say there are none) and unknown (the member left out:
       never say none; point at the dropdown) — at least 6 of each. A case
       whose derived context does not carry the list that way is dropped. The
       names are made up (two word banks), never a real store's.
     - **Split.** Each capability case is its own split group
       (`meta.group` = `<group>:<case id>`; `meta.cap_group` keeps
       `page:exists`, `list:tags` … for the reports): with one group name per
       behaviour, `h100("page:exists") < 8` sent every plain "page exists"
       case to validation and every listing case to train.
     - Cost: 82 cases at about US$0.03–0.06 a case asked (the strict listing
       and decline cases go to passes 2–3 more often), about **US$3–5** the
       first round; later rounds carry them while they do not change.
   - **Page creates filed as `pages` (prepare `fix_page_creates`).** The 34
     where=`pages` rows of the r4 seed split: 17 ask to rename, duplicate,
     delete, unpublish or move a page and stay; 17 ask to create one
     (r4-F3-0064 "add a new page called Strain Guide", r4-F3-0079,
     r4-F4-0243 …) and are relabelled: mode `page` expecting create_page with
     the title the request names (exact for a quoted, "called X" or "page: X"
     name; else one of the request's words, e.g. "4/20 deals"), `in_nav` when
     the request names the menu, header or navigation; or mode `ask` when it
     names no page (r4-F3-0073 "add a blank page", r4-F4-0250); `in_nav` only
     when the request puts the page IN the menu / header / navigation ("a menu
     page" is a page about a menu); a relabelled create of exam p137's page
     ("an events page") is dropped (`page_creates_fixed.exam_clash`). The changed
     expectation makes the carry re-ask them (about US$0.5); a relabelled
     hold-out gets the id `<id>.k2`. `report.json` `page_creates_fixed`; off:
     `"trigger": {"page_relabel": "off"}`.
   - Cost: the extra styles make authoring about 1.3–1.6× r5's tokens, still a
     few US$ a round (`prepare/report.json`).
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

**Carry-forward (r6; `retrain/carry.py`), before pass 1.** r5 paid US$101.57 to
re-ask the teacher r4's whole pool: the teacher re-worded 97.6 % of the
replies, 172 shared cases changed their writes, and 30 cases dropped out and
24 came in by teacher luck. That churn is not the round's intended change, and
it is a large part of why one round's model differs from the last. Now a case
keeps the row it was last **trained** on (the live model's dataset, from
`state/model.json` `dataset`), or failing that the row a parked round accepted
(`state/parked/`, newest first, at most 2), whenever that row still applies
on the round's manifest. The teacher is asked only for new, changed and
failing cases. A row applies when all of these hold (the first that fails is
the case's reason in `carry-report.json`):

| check | means |
|---|---|
| `rules_changed` | the system prompt's rules are the round's |
| `case_changed` | the case is the one the row was collected for (request, history, kind, target, state, score mode, expected; a rebase rename or relabel changes these) |
| `turns_changed` | the row has the case's number of user turns |
| `block_parity` | the row's candidate-fields block is reproduced from its turn inputs on the source manifest |
| `target_changed` | no field, list or container the row writes or the case expects changed its spec between the manifests |
| `server_result_stale` | no `lookup_fields` / `describe_target` result names a container whose spec changed |
| replay | the row's own calls, replayed through the round's engine and manifest, send the same text and context, get byte-identical tool results, and score usable under r4score on the round's case |
| `ref_written_changed` | (r6-4) the row's stored tool results never wrote a tag / brand slug without its derived id (`<…>.tagSlug` without `<…>.tagId`, the same for brands), and the replay's results do not differ from the stored ones by those ids alone. Since 2026-10-15 the editor hands the engine the store's tags and brands, so such a write now also writes the id: the old row's tool result is no longer what the model reads. Re-asked |
| `hidden_write` | no write to a field whose gate is closed after the op (`r4score.closed_gate_writes`, from the engine's `hidden[].byValue`; a bare `W_HIDDEN` comes with every gated write), unless the case lists it. On by default; `"carry_hidden": "keep"` carries them. On r5's inputs 64 of 2,688 carried rows hit it, most of them cases prepare already drops |
| `excluded` | `meta.no_carry` on the case, or its id in `trigger.carry_exclude` / `COLLECT_CARRY_EXCLUDE` |

A carried row goes into the attempt's logs as `pass0-carry.jsonl` (the
collector's log format, so `status`, `plan3`, `--prior` and the dataset build
read it like any pass). The dataset build re-renders it on the round's
manifest (system turn, tools, the scored turn's candidate-fields block;
`r4dataset.rerender_carried`) and applies every keep rule again; the
teacher's words and every tool result stay byte for byte, and a carried row
wins over a fresh one for the same case.

- **Switch.** On in full rounds, off in SMOKE (a rehearsal must ask the
  teacher). First set wins: `COLLECT_CARRY=on|off|required` on the
  execution, then the round's `trigger.carry` (`"on"` / `"off"` /
  `"required"`) or `trigger.reask_all: true`. **A carry that cannot run**
  (no tools, an engine error, no source) **asks every case**, about US$100
  instead of US$40–50, and only `metrics.carry.why` says so; with
  `"carry": "required"` it fails the collect stage instead
  (`carry_required`), before any teacher call. Use `required` for r6. A
  change to the service's system rules re-asks everything too
  (`rules_changed`), by design.
  The workflow copies its input's `trigger` into `run.json`, so a round
  started by hand turns it off with
  `--data='{"mode":"full","trigger":{"reason":"…","carry":"off"}}'`, with no
  workflow change.
- **Read.** `gcloud storage cat $W/collect/carry-report.json | jq` gives
  counts by reason, family and source, and the teacher spend not asked.
  Target: 85 % or more carried on a round whose manifest changed only a few
  fields.
- **Cost.** Per case asked, about US$0.03 for one that passes in pass 1 and
  about US$0.10 for one that goes through passes 2 and 3 (r5: US$101.57 over
  3,408 asked cases). r6, with r4 and r5 as sources, measured offline on r5's
  inputs: 2,688 of 3,408 carried (79 %); still asked: 258 cases rejected in
  every source (US$28.39, the stable reject core), 451 cases r4's caps cut
  (US$10.02), 11 changed (about US$1); plus the 120 F8 cases (about US$4)
  and the new authored F7 cases (about US$3–8). **That measurement is void
  for r6:** it was taken with r5's rules, and `rules.go` changed after both
  sources were collected (5d28c66 the font / ask-when-vague rules, 160ef91
  create_page and the option lists), so every source is skipped
  `rules_changed` and r6 carries **nothing**; the lab's contexts changed too
  (tags / brands lists, category ids, the browser's `sectionId` echo), which
  would re-ask most rows anyway. `"carry": "required"` does not catch it (the
  carry runs and carries 0). **r6 collect: about US$110–125** (about 3,750
  cases at r5's US$0.030 a case, plus more passes 2–3 on the strict listing
  and decline cases). From r7 (same rules) the carry works again and the
  capped cases are kept in `dataset/capped.jsonl` and carried, so collect
  falls to about US$30–40. The carry's replay is
  CPU on the collect job (about a minute; no model, no API).

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

## R2a. Scoring policy (r4score)

r4score decides which teacher rows are usable (collect, the carry's replay,
the dataset build) and scores the eval's generated holdouts (F0H, F9H). r6
adds relaxations for cases that failed on Sonnet, Opus and Sonnet alike
because the case or the scorer was too strict. Each one only turns a failing
turn into a passing one, so a `verify-3` pass is still a pass
(`RULE_VERSION` unchanged, `POLICY_VERSION` `r6-4`: r6-3 plus the `page` and
`list` modes and exam p137.k2, below); the one that applied is
named in the row's `info` and `relaxed`. **A relaxation may accept only an
answer that is right on the site.** The r6 review found the first versions
accepted wrong ones, so the defaults are narrow:

| switch | collect default | does |
|---|---|---|
| `refused_corrected` | **off** | `exact`: an op the engine refused and a later batch of the same turn re-sent accepted is not `invalid_op` (r4-F1-1127). Off: the training row keeps the refused call and its error, so it would teach sending an invalid op first; the case is re-asked instead (12 rows on r5) |
| `mailto` | on | `exact`: the answer wrote `mailto:x@y` where the case expected the bare `x@y`, on a link field. **One way only**: the site renders these as the href itself (the footer's social e-mail icon is `href={socialLinks.email}`, a gallery link gets `https://` prefixed), so a bare address is a broken link. Prepare also rewrites such expectations to `mailto:` (r4-F3-0413) |
| `same_control` | on | one editor control that writes one of several keys by the section's layout: the content-block "Text Alignment" writes `splitContentAlign` on split, `contentTextAlign` on feature columns (perksCards / perksIcons default there), `textAlign` otherwise (launcher `BasicEditor.jsx` `contentAlignKey`). Only the key **the editor would write in the case's state** is accepted for an expected sibling. r5's 18 rows it first recovered wrote `contentTextAlign` on image-text sections, which changes nothing on the page; they stay rejected |
| `acted_first_turn` | **off** | a follow-up whose vague first turn the teacher acted on instead of asking is excused when those writes are the case's own keys (`keys`) or on its targets (`target`), wrote no closed-gate field, **wrote no text the user never gave** (a text / richtext field or a list item's text: r4-F2-0102/0103 invented a 5-star review), and **every one of them was written again by the scored turn**. Off by default: an excused row still trains "act on a vague request instead of asking", the direction r5 drifted (no_write 39 → 28, wrong_target 59 → 88). Kept off by the user's 2026-10-15 decision (vague requests: ask) |
| `font` | `all` | the answer key of "the font", not a relaxation: see below |

**The eval scores with the r4 rules.** Its base is `STRICT_POLICY` (every
relaxation off): the rules the live model's stored calibration replicates
and earlier evals were scored with. A round's `trigger.score_policy` keys
are laid over it, so turning one on for the eval is a deliberate choice,
and then the calibration replicates need re-scoring ([DRIFT.md
§4](DRIFT.md#4-baselines-noise-and-calibration)). `eval/info.json` and
`result-collect.json` record the policy used.

**Measured on r5's collect logs** (offline, no teacher; 947 rows not usable,
257 cases with no usable row):

| policy | rows recovered | never-usable cases recovered |
|---|---|---|
| collect default | 3 (`mailto`) | 1 (F3) |
| + `acted_first_turn: "keys"` (guarded) | 38 (35 acted, 3 mailto) | 16 (F2 15, F3 1) |
| + `acted_first_turn: "target"` (guarded) | 40 | 17 (F2 16, F3 1) |
| + `refused_corrected: true` | 15 (12 refused, 3 mailto) | 5 (F1 3, F3 1, F4 1) |
| eval (`STRICT_POLICY`) | 0 | 0 |

The first version recovered 78 rows and 34 cases, most of them through the
rules above that accepted wrong answers.

**The font policy (decided 2026-10-15, the user after the demo: "font - change
everything").** A font request that names no part of the site ("i don't like
the font", "use lora", "Roboto everywhere") changes **every** site font to the
same font; a request that names a part ("the heading font", "the buttons")
means only that setting. `r4score.site_font_fields` reads the fonts from the
manifest and the case's state: **required** = the six theme fonts (heading,
body, navigation, buttons, product title, product description) plus any other
shown global font family the site has set (the announcement bar's, on every
template: 7 in practice); **optional** = shown but unset fonts that inherit
anyway (cart, footer); a write to a **hidden** one (the shop-menu fonts outside
the modern menu) is stray. `score_site_fonts` needs every required font written
with the asked value; when the merchant named no font, all the written fonts
must be the same one (mixed values are `wrong_value`). A bare "all" does not
widen a request that names its parts ("set the body and headings all to Lora"
is those two). Since 2026-10-15 the service prompt (`service/internal/prompt/rules.go`,
"Where things live") says the same, so the teacher writes every font in collect.

| `SOUSCHEF_FONT_POLICY` / `trigger.font_policy` | training rows (r4score) | exam |
|---|---|---|
| `all` (**default**, collect, eval and the noise arms) | a generic font request with the authored body + heading answer expects every site font (prepare's `fix_site_fonts` converts the 18 r4-F6 cases; F8 adds `site_fonts` contrast cases) | the 7 "the font" rows (p011, p013, p150 ×2, r02, r04, h059) carry the new key under new ids `<id>.k2` (exam.AMENDMENTS); h059 keeps body + heading required with every font as the alternative |
| `as_authored` | as authored (the r4 / r5 key) | the amended rows back to their old ids and keys: for re-scoring old runs |
| `body_only`, `body_or_both` | r6-2 options, kept for re-scoring | as authored first |

**Vague requests (decided 2026-10-15: "vague requests - ask").** A request too
vague to act on is answered with one short question: `ask` mode (a question in
the reply and nothing written passes; any write, even a refused one, is
`wrote`). Prepare's F8 adds single-turn ask cases (the follow-up family's vague
first turns, `askfirst-1`, and label-built vague requests, `vague-1`, both
through the leak filter). The exam lets 12 value-less rows and p013 be answered
with a question (`ask_ok`) while a write there still passes, so a good direct
action is never marked down. `acted_first_turn` stays off.

**create_page and dropdown choices (decided 2026-10-15; tools first: trained in
r6; `POLICY_VERSION` r6-4).** Two scoring modes:

- `page` (`exam.score_page`): scored on the **op**, not on manifest keys or a
  raw diff (`customPages[N]` is no manifest key, `modified` is the engine's
  clock). A create passes with exactly one accepted create_page whose title
  matches without case or spacing, whose slug (the engine's result, else
  `slugify(slug || title)`) is the canonical slugify of the asked slug or the
  title, with `in_nav` and `status` as asked; every other accepted op must
  write on the new page (a section add in the same reply is the page's own
  write) and the asked sections must be added; a refused op fails it
  (`invalid_op`, unless `refused_corrected`). A request that names no title
  (a relabelled pool row, `title_match: words`: "set up a page with a dosing
  guide") also passes with a question and nothing written — the rules say ask
  for the name. A **decline** passes when nothing is written, no
  report_unsupported, and the reply says why: for a page that exists (a
  create case whose page the site already has becomes this decline,
  `if_exists`) it names the page **and** says it is already there
  (`exam.EXISTS_SAID`: already, exists, existing, "you have a", taken, in use,
  instead — "Sorry, I can't make an events page" names it and explains
  nothing); the rules' other answer, "edit that page instead", passes too
  when the only accepted writes are the asked section adds on the existing
  page (not a same-batch follow-up of a refused create, which the batch stop
  blocks); for a reserved slug it asks a question (the engine suggests
  another, the rules say ask); for a template without custom pages it names
  the template. A refused create_page on a page the context already listed
  is flagged `create_refused` (not usable for training: it would teach
  trying). Every pass is verdict `pass`.
- `list` (`r4score.score_listing`): no write and no report_unsupported, and
  the reply, read against the context the turn was **sent**: every bullet that
  looks like a name is one of the list's names or slugs (`invented_name`);
  a list of up to 15 is named in full (`missing_names`); a longer one names 1–15
  then "...and N more" with N = total − named, the total from
  `<list>Truncated` when the context cut the list (`wrong_more_count`,
  `no_more_count`), and a cut list names the editor's dropdown
  (`no_dropdown_pointer`); an empty list (`[]`) is said to be empty
  (`did_not_say_none`; "no / 0 / none / haven't created any …" count); an
  unknown list (the member left out) is never said to be empty (`said_none`;
  a sentence about not being able to see or load the list is no such claim)
  and the dropdown is named. A name the reply runs on with in prose after one
  of the list's own names ("Staff Picks, New and Moon Rocks") counts as
  invented too. Verdict `answer_ok`.
  The dropdown is required only for a cut or unknown list (the service rules
  ask no more for a list of 16–100).

**The r7 answer key (`.k3`, `exam.ANSWER_KEY_R7`).** An amendment may carry
`_rev`; `amend()` writes `<id>.k<rev>`. h022.k3, p030.k3 and p047.k3 are
`ask_ok` (no URL / title given); h006.k3 requires `bodyFont` and allows the
footer body font (a question is `no_write`; an extra heading font is
`stray_writes`). p050 / p053 stay strict. `font_policy as_authored` restores
the built rows by `meta.superseded.id`, as for `.k2`. DRIFT.md "Re-keyed rows".

**Exam p137 → p137.k2** ("create a new page for events"). Its key was
report_unsupported `pages`; it now expects create_page `{title: "Events"}`
(`exam.AMENDMENTS`, answer key `create-page-2026-10-15`) **unless the site
already has that page**. The exam's vintage state does: `state.build_state`
seeds the custom pages About Us (/about) and Events (/events) from
`fixtures/context.json`, and the rederived contexts list them. So on today's
exam p137.k2 is scored as the decline "exists": nothing written, the reply
says the Events page is already there; a create_page the engine refuses
(E_SLUG, title taken) and then explained also passes; the r5 answer
(report_unsupported) fails. Under font policy `as_authored` it goes back to
p137 and `pages`. It stays in the `out_of_scope` set file, but the eval runs
it in the report-only `capability` set (R6): the live model was never offered
create_page, so pairing it would hand a trained candidate a free row. The
bar's `oos_where` rule is over the 12 rows still out of scope
(`exam.is_out_of_scope`, `oos_keys`) and allows two misses
(`bar.OOS_MISSES`), as r5's 11 of 13 did: 10 of 12 needed.

The dataset stage counts the rows that write both fonts for one request in
`stats.json` `font_pair` (and `font_pair.site_fonts`, the rows whose case
expects every font) and keeps them (`SOUSCHEF_FONT_PAIR=keep`, the default);
`SOUSCHEF_FONT_PAIR=drop` moves them to `rejects.jsonl` (reason
`font_pair_policy`). The exam's own answer key is not changed by either.
A teacher row that also writes the optional footer font is dropped by the
dataset's hold-out rule (`footerBodyTextFontFamily` is a held-out exam key);
the prompt does not name it, so few should.

**Setting a policy for a round.** Put it in the round's trigger:
`"trigger": {"font_policy": "body_only", "score_policy": {"acted_first_turn": "keys"}, "font_pair": "drop"}`.
Collect and eval adopt `font_policy` / `score_policy` (an unknown value stops
the stage before anything is scored), the dataset stage reads `font_pair`.
For a round already running, edit `trigger` in `runs/<id>/run.json` and
resume from `collect` (see "Change a running round's switches" at the top);
the stored trigger then reaches collect, dataset and eval alike, so the
dataset never re-scores what collect kept under another rule. A policy that also changes how the exam or F0H is scored makes
the live model's baseline incomparable: recalibrate it
([DRIFT.md §4](DRIFT.md#4-baselines-noise-and-calibration)). Costs nothing by
itself. Each relaxation recovers cases that would otherwise be re-asked
(about 50–90 cases, mostly F1 and F2, by the r5 analysis).

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
   `where_missing == 0`. `stats.json` is rewritten from the final files after
   every drop (r5's bucket copy still read the pre-drop 2,465 / 2,679 rows; the
   files held 2,455 + 214).
3. The row checks of `tools/run-r4.sh`: blocks present, follow-ups > 0, no
   reply mentioning the preview (`r4score.mentions_preview`: preview,
   previews, previewed, previewing).
4. `train --dry-run`: rows, tokens, updates, none over 32,768 tokens.
   (Not a gate:) `stats.json` `capabilities` counts the kept train / val rows
   and the rejected rows of the `create_page` and `list_options` cases (and the
   relabelled page creates) by group and mode, so a group that collected
   nothing usable shows as 0.
5. **New-key positives** (r6). Every new or retyped key of the round
   (`prepare/report.json` targets; F9H holdout keys apart) needs at least 6
   train rows that **write** it on a field the row's state shows. Rows that
   write it while it is hidden do not count. r5 trained 0 rows for
   `contentBlock.stackedImage*` and `gallery.imagesLink[]`, and nothing said
   so. `stats.json` lists every key's count by kind (set, reset, relative).
   SMOKE is exempt. The floor is `trigger.min_new_key_positives` (default 6;
   0 turns it off). A key a person has looked at and accepts short is waived
   by adding it to `trigger.new_key_positives_waive` in
   `runs/<id>/run.json`, then `resume_from dataset` (the resume reads the
   stored trigger; never set the matching environment variable on the job). A read-only or JSON field is exempt by
   itself. When it fails, read `stats.json` and `prepare/author-report.json`
   (`reject_details`) for why authoring came up short; a re-run of prepare
   for the round reads its authored file back and does not re-author.
   r7: `trigger.positives_required` (or `SOUSCHEF_POSITIVES_REQUIRED`) adds
   addresses to this gate that are not new (the re-authored keys, reason
   `positives_required`).
6. **Ask rule** (r7; `stats.json` `ask`). In order, each moving rows to
   `rejects.jsonl` with its reason: every kept `ask` row's rendered user text
   must be VAGUE under `askrule` (no value, no direction, no action; carried
   rows included) — `ask_rule_violation`; every ask row needs a kept row of
   its write twin (case `meta.ask_pair`; clarify, which-one and no-title
   create_page asks exempt) — `ask_unpaired`; ask + clarify rows at most
   `ask_share_max` (default 0.03, `trigger.ask_share_max`,
   `SOUSCHEF_ASK_SHARE_MAX`; 0 = off) of the rows, chosen by sha1 of the case
   id round-robin over fields after the family floors — `ask_cap` (twins
   stay). `stats.json` `ask` holds `rows`, `share`, `cap` (by family before
   and after), `violations_kept`, `unpaired_kept`, `exempt` and `pairs` (per
   field: ask rows, twin write rows, other relative write rows). The gate:
   `violations_kept == 0` and `unpaired_kept == 0`. `prepare/report.json`
   `ask_rule` has the pool's relabels (`pool`, with `unsure`: clear asks
   with no evidence for a write, dropped), the ask-first split
   (`ask_first`: `kept_vague`, `relabelled`, `twins`, `dropped_rule` —
   `unsure`, `conflicting_keys`, `no_field` —, `twins_dropped`), `vague`
   and `vague_twins` (two per vague case, three on a colour / number
   field); a relabelled case's `meta.ask_rule.evidence` says why it is a
   write (`oracle:…`, `direction:…`, `toggle:…`, `literal:…`); `reauthor` and
   `ambiguous_superseded` (`count`, `by_reason`, `regrouped`) cover the
   re-authored keys. Authoring rejects a draft as `ask_turn_not_vague`,
   `ambiguous_instance`, `ambiguous_twin`, `alias_instance`, `ambiguous_row`,
   `tag_not_in_store`, `tag_not_named`, `bad_value` or `bad_seed`
   (`author-report.json` `rejects`); the last six also supersede earlier
   rounds' drafts of a re-authored key (`ambiguous_superseded.by_reason`).

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

1. pull `dataset/` and the base tar, `sha256sum -c`, unpacked into
   `$HF_HOME/hub` (the tar holds the hub-cache folder
   `models--unsloth--Qwen3.5-9B/`; the job runs with `HF_HUB_OFFLINE=1`, so
   the snapshot at `refs/main` must resolve there before anything trains);
2. pull `train/checkpoints/` if present (a resume after preemption; only a
   checkpoint with `trainer_state.json`, `optimizer.pt`, `scheduler.pt` and
   `adapter_model.safetensors` is resumed from, and a checkpoint is pushed
   only once complete, `trainer_state.json` last);
3. row checks, then the dry plan;
4. on the first attempt only, the resume smoke (2 updates with a checkpoint,
   resumed to 4: it must log `Resuming from …checkpoint-2`);
5. full SFT, checkpoints synced to the bucket every 2 minutes;
6. export q4_k_m with `base-modelfiles/qwen3.5-9b.modelfile`. Unsloth's
   merge looks for the base as a local folder or on the Hub, never in the
   HF cache, so offline the Hub id fails ("could not reach the Hugging Face
   Hub"): export.py merges from a copy of the adapter whose
   `base_model_name_or_path` is the cached snapshot folder (the revision
   trained on; `export.json` `base_snapshot`). Unsloth's intermediates
   (`gguf/`, the merged 16-bit model, and `gguf_gguf/`) are deleted, not
   pushed;
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
   **The job's nonce.** Each job gets `TRAIN_NONCE` (its one-off `create`
   label); `progress.json` carries it and the workflow reads only what this
   job wrote. A failure also writes `train/failed-<nonce>.json`. Vertex
   restarts a container that exited 1, even after it has marked the job
   FAILED (the first SMOKE, 2026-10-14: the restarts rewrote `failed` as
   `running` and `interrupted`, so a deterministic failure was retried as a
   preemption); the entrypoint exits 1 at once, writing nothing, when that
   file exists, and the workflow classes a job with it **fatal** first.

**The recipe is r4's; only the data changes.** `unsloth/Qwen3.5-9B`, bf16
base, fp32 LoRA r16/α16, lr 2e-4, 1 epoch, batch 1 × grad_accum 4,
adamw_8bit, warmup 3 %, max_seq 32,768, `--eval-steps 150 --val-max-rows 150
--checkpoint-steps 50 --resume`. One epoch on purpose: a second epoch on
authored paraphrases teaches their wording rather than the mapping.

**Seeds, seed selection and warm starts (r6).** A round that sets nothing
trains exactly as r4 and r5 did: one seed (3407), from the base. The settings
come from the trainer's environment, else from `runs/<id>/train-config.json`
(the trainer reads `runs/`; no change to the workflow's job spec is needed),
resolved by `train.py pipeline_config`:

| setting | `train-config.json` key | effect |
|---|---|---|
| `TRAIN_ARMS=base:3407,warm:3407,base:2` | `"arms": ["base:3407", "warm:3407", "base:2"]` | each arm in turn in the same job, on the same dataset: `base:<seed>` a fresh LoRA from the base at the recipe's lr (2e-4), `warm:<seed>` from `init_adapter` at `init_lr` (5e-5). Each is resumable on its own (`train/checkpoints/<tag>/`, finished arms under `train/seeds/<tag>/`; tag `s<seed>` from the base, `w<seed>` warm); the best is kept as `train/adapter`. At most 4; exclusive with `seed` / `seeds`; a warm arm needs `init_adapter`, and `init_adapter` needs a warm arm. A bare `base` or `warm` is seed 3407 |
| `TRAIN_SEED=2` | `"seed": 2` | one arm, `base:2` (`train.py --seed`: the LoRA init and the data order) |
| `TRAIN_SEEDS=3407,2,3` | `"seeds": "3407,2,3"` | the arms `base:3407,base:2,base:3` (all `warm:` when `init_adapter` is set: r6's first form) |
| `TRAIN_GEN_CHECK=32` | `"gen_check": 32` | after each SFT, the first call on 32 fixed validation first turns, scored; the default with several arms |
| `INIT_ADAPTER=rt-…` (or `runs/…`, `base/…`, `gs://<pipeline bucket>/runs/…` or `…/base/…`) | `"init_adapter": …` | a warm start: that adapter's LoRA weights before the first update (same rank and alpha). Any other `gs://` path is refused: the trainer's account cannot read it |
| `INIT_LR=5e-5` | `"init_lr": 5e-5` | the warm arms' learning rate (default 5e-5; only with `INIT_ADAPTER`); base arms keep 2e-4 |
| `TRAIN_WINDOW_S=43200`, `TRAIN_TAIL_S=2700` | (env only) | the 12 h window guard below: the job's Vertex timeout, and the time kept for the export, stage and push |

- **Selection never reads the bar's sets.** With several arms, `train.py
  select` keeps the arms within 1 match of the best generation check, then
  the lowest final validation loss; a tie goes to the earlier arm. The
  validation subset no longer follows the seed (`--val-seed` 3407), so every
  arm, from the base or warm, is scored on the same rows and the same 32
  generation-check first turns. Every arm's scores are in
  `train/adapter/run.json` and `progress.json` (`seeds`: one entry per arm
  with `arm`, `start`, `lr`, `eval_loss`, `gen_match`; `selection`:
  `selected_arm`, `rule`, `skipped`), in `eval/info.json`
  `training_arms`, in `decision.json` `results.training.arms`, and as one
  line of the decision message ("Training arms (same validation rows):
  base:3407 loss 0.2311, gen 20/32; … Kept warm:3407."). The bar then runs
  once, on the kept arm.
- **Is the warm arm's validation loss fair?** The split is a hash of each
  case's group (`r4dataset.h100`), so a case r4 trained on that is carried
  into this round lands in `train` again, not in `val`. A case whose id
  changed (a rebase rename, a re-authored case) can still sit in `val`
  with a near-copy in r4's training set and flatter the warm arm a little;
  the bar, which runs on the exam sets, is the check that counts.
- **The 12 h window.** Every arm trains on the same rows, so arms take
  about the same time (2.0–2.4 h each on an H100 for an r4/r5-sized set).
  Before it starts an arm, `pipeline-train.sh` checks the time this attempt
  has used plus the longest finished arm (its `run.json` minutes, scaled up
  when that arm was resumed part-way) plus 45 min for the export against
  12 h; an arm that would not fit is **not started** and is recorded in
  `selection.skipped` (and the decision message). The first arm always
  trains. A later attempt (after a preemption) gets a fresh 12 h and trains
  what is left. The dataset stage writes the plan's estimate to
  `result-dataset.json` `train_window` (`total_h`, `fits`, `fit_arms`;
  `train.window_estimate`: 12 s an update, plus a fixed part per arm and
  for the job) and logs a warning when the arms do not fit.
- **An A100 rung** (about 2× slower) trains no new extra arm: it keeps the
  arms an earlier attempt finished and, when none has, trains only the
  first arm. Put the arm you most want first.
- **To use it for a round**, put the keys in the workflow input's trigger,
  `"trigger": {"train": {"seeds": "3407,2,3"}}`: the dataset stage checks the
  keys and every value with the trainer's own resolver (`train.pipeline_config`:
  seeds, `init_lr`, `gen_check`, an `init_adapter` inside the pipeline
  bucket's `runs/` or `base/`, the only prefixes the trainer can read) and
  that a warm-start adapter exists there; a bad one fails the dataset stage,
  not the GPU. It writes them to
  `runs/<id>/train-config.json` when that file is not there yet
  (`result-dataset.json` `train_config`). For a round already past dataset,
  write the file by hand before `resume_from train`:

  ```bash
  echo '{"arms": ["base:3407", "warm:3407", "base:2"], "init_adapter": "base/adapters/r4-sft"}' |
    gcloud storage cp - $W/train-config.json --content-type=application/json
  ```

  The dataset stage also reads the warm adapter's `adapter_config.json` from
  the bucket and refuses one whose LoRA is not r16/α16 on
  `unsloth/Qwen3.5-9B` (what `train.py` would refuse after loading the base
  on the GPU).

- **Warm starts run beside from-base seeds** (the user's decision,
  2026-10-15: "do both", keep the best on validation). The warm arm trains
  on the round's whole dataset, like the base arms, at lr 5e-5 from r4's
  adapter. Selection picks between them on validation only; the bar decides
  whether the kept one ships. A round that keeps a warm arm is a model two
  steps from the base: keep at least one `base:` arm in every round so
  warm starts cannot keep drifting unchecked.
- **Cost.** Each extra arm adds one SFT (2.0–2.4 H100-hours for an
  r4/r5-sized set, about US$15–20 at the recorded Spot rate) plus a
  generation check of a few minutes; a warm arm costs the same as a base
  arm, because it trains on the same rows. Three arms make the train stage
  about 7–8.5 h instead of 2.4–2.8 h.

### r4's adapter in the pipeline bucket

The warm arm reads `$PIPE/base/adapters/r4-sft/`. `base/` has no lifecycle
rule in the pipeline bucket, so the copy stays; `runs/` and the training
bucket's `runs/adapters/` copy do expire (the training bucket's copy around
**2026-11-10**). A person runs this with an account that can read
`$TRAIN` and write `$PIPE/base/` (the trainer's service account cannot write
`base/`). It is a server-side copy: nothing passes through the workstation.

```bash
export PIPE=gs://breadstack-sous-chef-int-pipeline TRAIN=gs://breadstack-sous-chef-int-training
SRC=$TRAIN/runs/adapters/r4-sft
DST=$PIPE/base/adapters/r4-sft
# 1. what is there: adapter/{adapter_config.json, adapter_model.safetensors, tokenizer…}, run.json, loss.jsonl
#    (and checkpoints/ if the VM pushed them: not needed for a warm start)
gcloud storage ls -r "$SRC/" | grep -v '/checkpoints/'
gcloud storage cat "$SRC/adapter/adapter_config.json" | jq '{peft_type, r, lora_alpha, base_model_name_or_path}'
#    expect LORA, 16, 16, "unsloth/Qwen3.5-9B"
# 2. the copy, bucket to bucket, without the checkpoints
gcloud storage rsync --recursive --exclude='^checkpoints/' "$SRC" "$DST"
# 3. check it: the same bytes as the workstation's copy (lab/runs/adapters/r4-sft, 185 MB)
gcloud storage hash --hex "$DST/adapter/adapter_model.safetensors"
gcloud storage hash --hex lab/runs/adapters/r4-sft/adapter/adapter_model.safetensors   # the md5 lines must match
gcloud storage cat "$DST/run.json" | jq '{name, seed, steps, last_eval_loss}'      # r4-sft, 3407, 601, 0.2287
```

If the training bucket's copy has already expired, upload the
workstation's copy instead (the same layout):
`gcloud storage rsync --recursive --exclude='^checkpoints/' lab/runs/adapters/r4-sft $PIPE/base/adapters/r4-sft`.
The trainer pulls the folder and uses its `adapter/` (with `run.json` beside
it, which the warm arm's `run.json` records as `init_adapter.run`).

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
| `capability`: exam rows re-keyed to a new tool (p137.k2, create_page), taken out of their set | re-derived | 2 | reported only |
| `capability:built`: about 15 create_page / listing cases (`contrast.capability_report_cases`, phrasings the training jobs do not use) | built | 2 | reported only |

**Capability rows (r6-4).** The live model was never offered `create_page`, so
on p137.k2 a candidate trained on it wins a row for free: in a bar set that
row moves the set Δ by +1, McNemar's c by 2 and the exam-drift trigger total.
Exam rows whose key is a new tool (`exam.page_spec`) therefore run in the
report-only `capability` set; the candidate baseline keeps them under their
own set's key (`out_of_scope:p137.k2`), so drift still tracks them. No exam
row creates a page or lists choices, so `capability:built` adds about 15
cases that do (creates, declines, a section on an existing page, listings
short / long / cut / empty / unknown), scored by r4score; a candidate that
over-creates pages or invents names shows in the report table. GPU time only;
`capability_report: false` in `config/thresholds.json` leaves them out.

**The probe (r6).** r5's probe could not be passed: 7 of its 17 rows
targeted sections the shared eval state does not have (contentBlock-2/-3/
-features, newsletter), so the engine refused them whatever the model did,
and the trigger rule had ceilings of 25 % and 0 %. Now every probe row
carries the state it was generated on and runs there; the engine checks each
case on that state before it is asked; section types a key names but the
state lacks (perksCards, perksIcons, newsletter) get a home row preset so the
key shows; gallery `imagesLink[]` gets update cases; and each probe row keeps
its conversation (`trajectory`) next to its result in
`eval/<arm>/rows.jsonl`. What was dropped and why is in `eval/info.json`
`probe_dropped`. Details and the offline re-check of a past probe:
[DRIFT.md §3.4](DRIFT.md#34-the-new-key-probe). It adds no cost beyond the
rows asked (the perks and gallery keys now get rows: up to 4 a key, twice,
at about 8 s a turn).

**The bar** (`retrain/bar.py`). `tol_s = max(1 row, ⌈2 σ_Δ,s⌉)` from the
current model's baseline. The candidate passes when all hold:

1. on every bar set, candidate − current ≥ −tol_s;
2. a pooled paired McNemar test does not say the candidate is worse (p < 0.05);
3. `out_of_scope`: every row but two with `where` (`bar.OOS_MISSES`; r5's
   rule was 11 of 13), over the set's rows that are still out of scope (12
   since p137.k2 moved to `capability`: 10 needed);
4. F0H `:unseen`: candidate ≥ current − tol;
5. a round triggered by dodged keys: ≥ 75 % of their probes pass and more than
   current; a round triggered by exam drift: the deployed-manifest total
   improves by ≥ σ_Δ;
6. ghost writes ≤ current.
7. `r4_floor` (r7; thresholds `r4_reference`): the candidate against r4's
   **recorded** calibration replicates (`baselines/reference/r4/baseline.json`,
   else the live `baselines/sha256:f657…` copy, else the Mistral snapshot's;
   every copy found is hashed; a copy whose file sha256 is not
   `r4_reference.file_sha256` (`496a5ca0…`) is skipped, and a model digest
   other than `f657…` is refused), on paraphrase, holdout, real and followup:
   each set Δ_vs_r4 ≥ −tol (r4's own σ_run; with `sigma_seed` set, plus
   **1**·σ_seed², since r4 is one fixed recorded model and only the
   candidate is a seed draw — 2·σ_seed² would have let r6's −6.3 pass at
   σ_seed 2.5) and the pooled McNemar not significantly worse. A failure is
   read as [Reading r4_floor](#reading-r4_floor) says. Re-keyed rows
   (`.k2`/`.k3`) and rows stale on the round's manifest drop out of the
   pairing and are counted (`results.r4_reference`). No file: the rule passes
   "NOT MEASURED" and `report.json` `warnings` says so. Without this rule r7
   would only be judged against r6, which already lost 6.5 holdout rows.
   The `:vs-r4` rows show r4 in the "current" column. Next to it, report
   only unless `r4_reference.rerun_bar` is true, `r4_rerun_floor`: the same
   comparison with r4 **re-run** in `rows_run`'s round (r6's: r4 as its
   current arm, under r6's prompt, scorer and manifest; `<set>:vs-r4_rerun`
   rows, `results.r4_rerun_reference`). It is the better-matched floor — r4
   scores +2.3 paraphrase and +1.2 holdout above its recording when re-run,
   so `r4_floor` is about 1–2 rows lenient — and it pairs the `.k2` rows too
   (4 rows left out instead of 23). On r6's real eval both fail: holdout −6.3
   vs the recording, −6.5 vs the re-run. Beside it, the
   report-only **asks on clear requests**: per arm (and r4's recorded rows of
   `rows_run`), the rows answered with a question although `askrule` calls
   the request not vague, by exam class (relative, synonym, unnamed,
   selected). Read it first.

**The tolerance counts eval noise only, not training noise.** `σ_Δ,s` is how
much the same weights vary between two passes over the exam. A retrain of the
same data with the same recipe lands on other weights, and nobody had
measured how far that moves a set: r5 retrained r4's data and lost 3.5 rows
on holdout against a tolerance of 2 (σ_Δ 1.0), a loss its pooled McNemar
test (b 32 / c 20, p 0.13) could not tell from a reroll. The proposed
seed-aware tolerance is

    tol_s = max(1, ⌈2 · sqrt(σ_Δ,s² + 2 σ_seed,s²)⌉)

`σ_seed,s` is one trained model's seed spread; the bar compares two trained
models (the candidate and live), each a seed draw, so the variance of their
difference holds it twice.

with `σ_seed,s` measured per set by the seed-noise experiment
([R10](#r10-the-seed-noise-experiment)). The experiment writes it as
`{"sigma_seed": {"<set>": rows, …}}` for `config/thresholds.json`, which
`bar.py` reads: a set with a positive `sigma_seed` gets the wider `tol_s`
(`bar.json` then shows `sigma_seed` and `tol_eval_only` beside `tol`); a set
without one keeps `max(1, ⌈2 σ_Δ,s⌉)`. The committed default
(`lab/pipeline/thresholds.json`) is `{}`, so the bar is unchanged until a
person copies the experiment's `thresholds_override` in. R10's `analyze
--compare <round>` shows what a round's bar would say with it first.

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
              collect/{pass0-carry.jsonl, carry-report.json}   the carried rows (r6)
              train-config.json            seeds / warm start for the trainer (r6; optional)
              train/{seeds/<tag>/, checkpoints/<tag>/}   a several-arm round's per-arm runs (tag s<seed> from the base, w<seed> warm)
              model/{context.tar.gz,image.json}  eval/  report.json  decision.json
runs/<nz-exp>-<data>-s<seed>/              a seed-noise arm (R10): a round's layout + noise.json
noise/<nz-exp>/{plan.json, result.json}    a seed-noise experiment: its arms, σ_seed and the proposed tolerances
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
| dataset gate: new-key positives | a new or retyped key has fewer than 6 train rows that write it visibly (`stats.json` per key) | read `prepare/author-report.json` `reject_details`; fix the authored cases (R1, edit `cases/pool/authored/$RUN.jsonl`) and resume from `prepare`, or waive a key you accept short (`trigger.new_key_positives_waive` in `run.json`, R3) and resume from `dataset` |
| dataset: `trigger.train` | an arm list, seed list, learning rate, generation check or adapter path the trainer would refuse (a `warm:` arm without `init_adapter`, an `init_adapter` without a `warm:` arm, `arms` beside `seeds`, more than 4 arms), an `init_adapter` with no `adapter_config.json` in the bucket, or one that is not r16/α16 on the recipe's base | fix `trigger.train` in `run.json` (R4; upload the adapter first: [r4's adapter](#r4s-adapter-in-the-pipeline-bucket)), resume from `dataset`. Checked here so it never fails on the GPU |
| `arm … not started: the 12 h window` (train log; `selection.skipped`) | the arms trained so far left no room for another before the export | nothing to fix for this round: the best of the finished arms is kept and the decision message names the arm not trained. For a later round, fewer arms, or the arm you most want first |
| collect: `carry_required` | `"carry": "required"` and the carry could not run (`result-collect.json` error) | fix the cause (the lab image's tools, the sources in `state/`), or set `"carry": "on"` to let it ask everything (about US$100), then resume from `collect` |
| `train-config.json: unknown key` / a bad `TRAIN_ARMS` or `TRAIN_SEEDS` (training `fatal` at step `settings`) | the round's arm, seed or warm-start settings do not parse | fix `runs/<id>/train-config.json` (R4), resume from `train` |
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
| `build FAILURE` with `toomanyrequests` | a Docker Hub pull limit: the spec predates the `mirror.gcr.io` base | `terraform apply` (rewrites `infra/cloudbuild-model.json`), resume from `build` |
| `build FAILURE` with `unknown flag: --build-context` | the build ran on the legacy worker or builder (Docker 20.10): `infra/cloudbuild-model.json` predates `workerRelease` | `terraform apply` (rewrites the spec), resume from `build` |
| `decision_mismatch` | the decision row is not this round's (another notification, run or SMOKE flag; a duplicate at the first ask; a failing bar approved without the override) | look at the decision page and the thermometer's API keys (only the key id in `souschef_ingest_key_ids` may post); then `resume_from decide`, or start a new round |
| `digest_mismatch` | the candidate tag, the build record and the decision's digest disagree (at rollout), or `model/image.json` is not what the `c-<id>` tag names (at candidate or eval) | do not roll out; read `model/image.json`, `decision.json` and the repository's `c-<id>` tag; at candidate or eval, `resume_from build` |
| `candidate_decided` (at init) | a resume up to `eval` of a decided candidate | `resume_from decide`, or a new round |
| training `fatal` | the trainer itself failed (`train/failed-<nonce>.json`, or `progress.json` status `failed`; the step in `error`) | read `train/train.log` (the job's own log in Cloud Logging, `resource.type=ml_job`, if a restart overwrote it); fix; resume from `train` |
| `images_changed` (stage `images`) | a resume from `train` … `eval` while the service serves another tag than the round's (`run.json` `images.tag`): train and eval would run other images than the ones that built the dataset | start a new round; or, to resume with a fixed trainer, rebuild it under the round's own tag (`make trainer-image TAG=<run.json images.tag>`) while that tag still serves |

Locks are only ever broken by a person, after checking
`gcloud workflows executions list … --filter='state=ACTIVE'` is empty.

## R9. Costs and durations

| stage | duration | cost |
|---|---|---|
| prepare | 30–60 min | Anthropic authoring: a few US$ per round (reported in `prepare/report.json`) |
| collect | about 1 h with the carry (estimate); 2.2–2.8 h at 6 workers without | **r6 asks every case: about US$110–125** (R2: the rules changed since r4 and r5 were collected, so the carry carries nothing; about 3,750 cases at r5's rate, including about US$3–5 for the 82 create_page / listing cases and the 17 relabelled page creates); about US$30–40 from r7 with the carry; without it US$100–175 (r5 asked 3,408 cases for US$101.57). **r7**: the carry should keep r6's rows (rules unchanged, checked 2026-10-17); the label change re-asks about 363 rows (about US$13), plus the new twins, relabels and re-authored keys: about US$50–100. If nothing carries (`rules_changed`), r6's US$177 for 4,970 cases plus r7's new cases: **about US$200–250** |
| dataset | 15 min | — |
| train + export | 2.4–2.8 h + 0–4 h Flex-start wait; about 2.2 h more per extra arm, base or warm (three arms: 7–8.5 h of H100, inside the 12 h attempt; an A100 rung trains no extra arm) | about 3 H100-hours, plus 2.0–2.4 a seed. **The rate is not verified**: US$6.57/h is the training VM's Spot rate (TRAIN-REMOTE.md); Vertex bills its own Flex-start / Spot rate. Read r5's job `customJobs/5995030053153406976` in the billing report before quoting a budget |
| image | 15–25 min | Cloud Build minutes; about US$0.90/month per kept tag |
| candidate + paired eval | 2–3 h on two L4s | about US$1.05/h each |
| **round** | **about 9–12 h** (about 13–17 h with three seeds) | **r6 with three seeds: about US$165–190** (about US$115–130 Claude — collect plus a few US$ of authoring —, US$46–56 H100 at the recorded rate, US$7 L4); one seed about US$135–150; **r7 with two arms and the carry: about US$90–150; worst case about US$250** if the carry carries nothing (pre-flight step 5 checks the rules) |
| seed-noise experiment (R10, once) | about 10–12 h (training in parallel, evals one at a time) | about US$47–57 for two arms (2.4 H100-hours an arm measured on r5, at the recorded rate; US$11.55 of L4), no Claude spend; each further arm about US$24–29 |
| SMOKE | 2–6 h (the trainer image pull, the base tar, up to 30 min of Flex-start) | about US$2 of API, 1 H100-hour, 30 L4-minutes (about US$10–15) |

Collection was the big line and recurred every round: every case was
re-collected because the system index and the candidate-fields blocks are
specific to the manifest. From r6 the carry re-renders an earlier row on the
round's manifest and re-verifies it instead (R2), so a case is re-asked only
when it is new, changed or fails. Cost is reported per round, never a reason
to stop (user decision).

## R10. The seed-noise experiment

**The question.** How far does a retrain of the **same** data with the
**same** recipe move each bar set? The bar's tolerance counts only how much
one model's answers vary between two passes over the exam (R6). r5 retrained
r4's data (plus 58 new-key rows) with r4's recipe and seed and lost 3.5 rows
on holdout against a tolerance of 2, and nobody had ever trained one dataset
twice, so nobody could say whether that was a real loss. This experiment
trains r4's exact dataset with other seeds, evaluates each one against live
r4 the way a round's candidate is evaluated, and measures `σ_seed` per set.

Code: `lab/souschef_lab/retrain/noise.py` (one command per step, each one
idempotent) and `lab/tools/noise-run.sh` (all of them in order).

**What it does.** An *arm* is `DATA@SEED`: `r4@2` trains
`cases/pool/r4/dataset` with seed 2. Each arm gets a run folder
`runs/<exp>-<data>-s<seed>/` laid out like a round's, so the trainer, the
model build and the eval stage run unchanged:

| step | does | runs on |
|---|---|---|
| `plan` | the arms, the commands and the cost; writes nothing | workstation |
| `setup` | `dataset/` copied from DATA (bucket to bucket); `run.json`, `prepare/manifest.json`, `prepare/cases.jsonl` and `model/context.tar.gz` from a base round (r5's by default: its deployed manifest `1184ff9a` is the eval's manifest, and its trigger keeps the new-key probe); `train-config.json` `{"seed": N}`; `noise/<exp>/plan.json` | workstation (bucket writes) |
| `train` | one Vertex custom job per arm, the job the workflow's `vertex_attempt` sends (the trainer image, args `[<arm>, full]`, its env plus `TRAIN_SEED`, labelled `purpose=sous-chef-noise`), on a rung of `infra/config.json`'s ladder (default `h100-flex`). A Spot preemption, or a job that ends without `DONE`, is resubmitted and resumes from the checkpoints (up to 4 attempts); a trainer failure stops. The ladder is walked as the workflow walks it: a job that **never started** (a Flex-start wait that ran out: no `startTime`, no progress) or a rung that used its own `attempts` moves to the next rung (`h100-flex` → `h100-spot-usc1` …), so a Flex rung without capacity costs one wait, not four. After `DONE` it checks that `train/adapter/run.json` recorded the arm's seed | Vertex, arms in parallel |
| `build` | `gcloud builds submit` of the arm's context with the bucket's `infra/cloudbuild-model.json` (the pinned build) → `model:c-<arm>`; `model/image.json` | Cloud Build |
| `eval` | one arm at a time: takes `state/locks/retrain.json` (no round may re-point the candidate service) and `state/locks/gpu-eval.json` (no drift check shares the eval GPU); checks production still serves `state/model.json`'s digest; points `sous-chef-model-eval` at it and `sous-chef-model-cand` at the arm; warms both; runs job `souschef-eval` (`eval --run-id <arm> …`); releases both locks | the two L4 services, job `souschef-eval` |
| `analyze` | σ_seed per bar set, the proposed tolerances, the thresholds override → `noise/<exp>/result.json` | workstation |

Every arm's progress is in `runs/<arm>/noise.json` (jobs and attempts,
build, eval execution, the digests the eval compared, the eval job's lab
image). `status` prints one line per arm.

**Before you start.**

1. **A trainer image that honours `TRAIN_SEED`**: one built from a tree with
   r6's `pipeline-train.sh` (`make trainer-image TAG=<tag>`). r5's trainer
   (`eac87f9`, the base round's default) ignores it and would train seed
   3407. `train` catches that after training (`seed_mismatch`: the arm is
   stopped and left out of the analysis), but the GPU hours are spent by
   then. Pass the image as `TRAINER_IMAGE` / `--trainer-image`.
2. **The live model is r4** (`state/model.json` tag `r4`) and production
   serves its digest. `eval` refuses otherwise: the arms must share one
   reference.
3. **No round running** (`state/locks/retrain.json` absent). `eval` refuses
   while a round holds it, and stops (exit 2, nothing changed) while a drift
   check holds `gpu-eval`: run it again later. While an arm's eval runs, a
   drift check waits for its next poll, a `retrain` verdict starts nothing
   ("a retrain round is in progress"), and a round started by hand stops at
   init with `retrain_locked`.
4. **One lab image for every arm's eval.** `souschef-eval` runs the image
   the last round pointed it at. Use the r6 lab image (the probe fix) and
   pass `--lab-image` so every arm's eval refuses any other.
5. H100 quota for two jobs at once (preemptible: 64 in us-central1).

**Run it** (a person, in bash, inside tmux: about 10–12 h):

```bash
cd ~/projects/breadstack-sous-chef
export PIPELINE_BUCKET=$PIPE
TRAINER_IMAGE=$REGION-docker.pkg.dev/$PROJECT/sous-chef/sous-chef-trainer:<r6 tag> \
LAB_IMAGE=$REGION-docker.pkg.dev/$PROJECT/sous-chef/sous-chef-lab:<r6 tag> \
  bash lab/tools/noise-run.sh nz-$(date -u +%Y%m%d)            # r4@2 and r4@3; prints the plan and asks first
```

Step by step (the same thing; each command can be re-run after any stop):

```bash
N() { (cd lab && .venv/bin/python -m souschef_lab.retrain.noise "$@"); }
E=nz-20261015; A=(--arm r4@2 --arm r4@3)
N plan    --exp $E "${A[@]}"
N setup   --exp $E "${A[@]}" --trainer-image <r6 trainer image>
N train   --exp $E "${A[@]}" --wait                    # or without --wait, then `N status` now and then
N build   --exp $E "${A[@]}" --wait
N eval    --exp $E "${A[@]}" --lab-image <r6 lab image>
N analyze --exp $E "${A[@]}" --compare rt-20261014-0708-7cac28-1184ff9a
```

Rehearsal with no network and no spend: `DRY=1 bash lab/tools/noise-run.sh
nz-20261015` (or `noise dry --workdir <dir>`), which runs every step on a
local bucket with a fake Vertex, Cloud Build and eval job.

**Further arms.**

- `r4@3407`: r4's own seed, retrained on the Vertex trainer image. Live r4
  was trained on the training VM with `run-r4.sh`, so this arm tells
  "another seed" apart from "another machine and software build".
- r5's data with seed 2 (the 2×2 of data × seed in r5why PLAN §2a):
  `--arm r5=runs/rt-20261014-0708-7cac28-1184ff9a/dataset@2`, then at
  analyze `--extra rt-20261014-0708-7cac28-1184ff9a=r5@3407` (r5's own
  candidate as a second draw of its data; its current arm adds two more
  replicates of live r4).

**Reading the result** (`analyze` prints a table; `--json` the whole
document).

| column | means |
|---|---|
| `n` | the rows every replicate of every model answered |
| `df` | the degrees of freedom of the spread: draws of one dataset − 1, summed over datasets (2 for live r4 + two arms) |
| `σ_eval` | the bar's own `σ_Δ,s` for the set (median over the arms' evals): today's noise model |
| `σ_seed` | `sqrt(max(0, S² − e))`: `S²` is the sample variance of the models' mean set scores within one dataset (live r4 counted as one more draw of r4's data; its replicates pooled over every arm's current arm), `e` the eval noise in those means (each model's unbiased per-replicate variance `K/(K−1) Σ p(1−p)` over K) |
| `(hi)` | the one-sided 90 % upper bound of σ_seed (chi-square on `df`): with df 2 it is about 3× the estimate, so read it as "how large it could plausibly be" |
| `tol now` → `seed` | today's `max(1, ⌈2 σ_eval⌉)` and the seed-aware `max(1, ⌈2 sqrt(σ_eval² + 2 σ_seed²)⌉)` (what `bar.py` computes from `sigma_seed`); the next column uses the upper bound |
| Δ vs live | each arm's paired delta against live r4 in its own eval |

`result.json` also holds `tol_one_draw` (the tolerance if only the candidate
were a seed draw: `sqrt(σ_eval² + σ_seed²)`, for reference), each model's per-replicate
counts, and `compare`: for each `--compare` round, every bar set's delta with
the tolerance it had and the seed-aware one, and whether it passes under
each.

**The decision rule** (r5why PLAN §2a), printed as `holdout_verdict`:

- a reseeded r4 moves holdout by **3 rows or more** against live r4 → r5's
  loss is within what a retrain does on its own: put `thresholds_override`
  into `config/thresholds.json` (`bar.py` reads `sigma_seed`; a person
  writes `config/`; the drift and retrain jobs pick it up at their next run,
  no rebuild);
- every reseeded r4 stays within **±1 row** → the noise explanation is
  weakened: look harder at the data links (the font pair, the lost
  cardsPerRow row, re-collection churn) and the train/serve path;
- in between → inconclusive: read `σ_seed` and its upper bound, and add the
  r6 round's own seeds (a `TRAIN_SEEDS` round gives more draws of one
  dataset) before changing the bar.

**What it costs.** Training: about 3.5 H100-hours an arm (an r4-sized SFT is
2.0–2.4 h, plus the base pull, the resume smoke and the export), US$23 at the
recorded Spot rate of US$6.57/h; Vertex bills its own rate for Flex-start and
Spot, so read the billing report for the real figure. Eval: about 2.75 h an
arm on two L4s (2 × US$1.05/h), about US$5.80. Builds: about US$0.15 each.
**Two arms: about US$57, no Claude or Anthropic spend.** Each model tag
(about 6.4 GB) costs about US$0.90 a month until pruned. Wall time: training
3–5 h in parallel (plus up to 4 h of Flex-start wait), then about 3 h of
eval per arm, one after the other. The eval also runs the newest-manifest
set when the launcher has moved with a hard change since the base round
(three `packFilterDisplay` fields since `1184ff9a`): about 40 minutes more
an arm, reported only.

**Scoring.** Every arm's `run.json` pins `trigger.score_policy` to the r4
rules (`STRICT_POLICY`), the eval's default anyway: replicates are pooled
across the arms, live r4 and `--extra` rounds, which must all be scored
alike.

**Afterwards.**

- The arms' run folders expire with `runs/` (180 days). `noise/<exp>/` has no
  lifecycle: keep it, it is the record behind the bar's `sigma_seed`.
- The `c-nz-…` images stay (immutable tags); prune them as in R7 when the
  result is recorded.
- `sous-chef-model-eval` is left on production's digest (as after any
  check); `sous-chef-model-cand` on the last arm's image (the next round
  re-points it). Both scale to zero.
- Each arm's eval leaves `report.json`, `decision.json` and a
  `result-eval.json` with a `candidate_ready` notification in its own
  folder. Only the workflow posts notifications, so nothing reaches Slack or
  the thermometer, and no decision row is created.
- An interrupted `eval` (Ctrl-C between polls) releases both locks; run it
  again and it takes them back and keeps polling the same execution. A
  killed shell leaves the locks to expire (8 h).

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

From r5 (2026-10-14):

- **A 2–4 row delta can be a reroll.** Two trainings of one dataset differ;
  read a set's delta against its seed-aware tolerance (R6, R10) and the
  pooled McNemar, not the raw number.
- **A probe row must be passable.** Check every generated exam row through
  the engine on the state it runs on, before it can fail a model.
- **Count what a round teaches a new key**: positive, visible writes per key
  (the R3 gate), not cases authored.
- **Re-asking the teacher is not free of change**: it rewords and rerolls.
  Carry what still applies (R2).
- **Keys from the environment only**, never argv, never echoed.
