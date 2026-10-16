"""noise — the seed-noise experiment (docs/RETRAIN.md R10): how far does a RETRAIN of the same data move the bar?

The shipping bar's tolerance (retrain/bar.py) counts only eval-sampling noise: how much the same weights vary
between two passes over the exam. r5 (2026-10-14) retrained r4's data with the same recipe and seed and lost
3.5 rows on holdout against a tolerance of 2; nobody had ever trained one dataset twice, so nobody knew whether a
retrain moves holdout that far on its own. This tool measures it:

  1. train one dataset (r4's, cases/pool/r4/dataset, by default) with other seeds on Vertex, on the trainer image,
     with the env the pipeline sends (pipeline-train.sh, TRAIN_SEED);
  2. build each as a candidate image (the pipeline's model build, model:c-<arm>);
  3. run the paired eval of each against the live model with the eval stage as it is (job souschef-eval);
  4. measure σ_seed per bar set and the seed-aware tolerance

        tol_s = max(1, ⌈2 · sqrt(σ_eval,s² + 2 σ_seed,s²)⌉)

     σ_eval,s = the bar's own σ_Δ,s (two replicates a side); σ_seed,s = the SD of one training draw's true set
     score around its recipe's mean (the live model counted as one more draw of its data). The bar compares two
     draws (candidate and live), hence 2 σ_seed².

An ARM is DATA@SEED: DATA = `r4` (cases/pool/r4/dataset) | a round id (`rt-…`: runs/<id>/dataset) |
LABEL=PREFIX (a bucket folder with r4dataset's layout). Each arm is a run folder runs/<exp>-<label>-s<seed>/
shaped like a round's, so pipeline-train.sh, the model build and the eval stage need no change:

  dataset/                  copied from DATA (bucket to bucket)
  run.json                  the BASE round's (its deployed manifest pins the eval's manifest; its trigger keeps
                            the new-key probe) + {"noise": {...}}; never the workflow's: no execution, no resumes
  prepare/manifest.json     the base round's (the eval reads the deployed manifest from here)
  prepare/cases.jsonl       the base round's (F0H / F9H holdouts and the :seen split)
  model/context.tar.gz      the base round's model-image build context
  train/…                   the trainer's (adapter/run.json records the seed it trained with: checked)
  model/image.json          written here after the build
  eval/…, report.json …     the eval stage's
  noise.json                this tool's record: jobs, build, eval execution, digests, the eval job's lab image

Commands (each idempotent: re-run it to continue where it stopped):

  plan     the arms, every command and the cost; writes nothing
  setup    seed the arms' run folders and noise/<exp>/plan.json
  train    submit (or resubmit after a preemption) each arm's Vertex job; --wait polls until all end
  build    the model image per trained arm → model/image.json; --wait polls
  eval     per arm, one at a time, under state/locks/retrain.json and state/locks/gpu-eval.json: point
           sous-chef-model-eval at production's serving digest and sous-chef-model-cand at the arm's, warm both,
           run job souschef-eval, release
  analyze  σ_seed and the proposed tolerances → noise/<exp>/result.json and the thresholds override
  status   one line per arm
  dry      everything above on a local bucket with fakes (no network, no GPU, no spend)

Nothing here reads a secret or calls a model API. Only `plan`, `status`, `analyze` and `dry` are free.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .common import ConfigError, StageError, Store, open_store, run_prefix

RECIPE_SEED = 3407                      # train.py RECIPE["seed"]: every model so far (r4, r5)
DEFAULT_ARMS = ("r4@2", "r4@3")
R4_DATASET = "cases/pool/r4/dataset"
TEACHER_FREE = 0.0
EXAM_SETS = ("paraphrase", "holdout", "real", "followup", "out_of_scope", "out_of_scope_holdout")
# the sets the bar reads (bar.py): the six exam sets ×2, F0H :unseen ×1, F9H and the probe ×2 when present
BAR_SETS = EXAM_SETS + ("F0H:unseen", "F9H", "probe")
REPORTED_SETS = ("F0H:seen",)
EVAL_LOCK, RETRAIN_LOCK = "gpu-eval", "retrain"
EVAL_LOCK_TTL_S = 8 * 3600
EVAL_TIMEOUT_S = 64800                  # the workflow's eval stage_job timeout
TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
# costs (recorded or published; docs/RETRAIN.md R9 / R10)
H100_RATE = 6.57                        # US$/h, the training VM's Spot rate (recorded); Vertex bills its own rate
H100_HOURS = 3.5                        # an r4-sized SFT is 2.0–2.4 h; + pulls, the resume smoke, the export
L4_RATE = 1.05                          # US$/h per L4 service (RETRAIN R9)
EVAL_HOURS = 2.75                       # 2–3 h on two L4s; + about 40 min when the newest-manifest set runs
BUILD_USD = 0.15                        # E2_HIGHCPU_8, 15–25 min
TAG_USD_MONTH = 0.90                    # one kept model tag (about 6.4 GB)


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- arms -------------------------------------------------------------------------------------------------------------

@dataclass
class Arm:
    exp: str
    label: str          # r4 | d<8> | a given LABEL
    prefix: str         # the dataset folder in the bucket (r4dataset's layout: <prefix>/dataset/train.jsonl)
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.exp}-{self.label}-s{self.seed}"

    @property
    def name(self) -> str:
        return f"{self.label}@{self.seed}"

    @property
    def candidate_id(self) -> str:
        return f"c-{self.run_id}"


def parse_arm(spec: str, exp: str) -> Arm:
    """DATA@SEED → Arm. DATA: r4 | rt-<round id> | LABEL=PREFIX."""
    data, sep, seed = spec.rpartition("@")
    if not sep or not data or not seed.isdigit():
        raise ConfigError(f"arm {spec!r}: DATA@SEED (r4@2, rt-…@2, LABEL=PREFIX@2)")
    if "=" in data:
        label, prefix = data.split("=", 1)
    elif data == "r4":
        label, prefix = "r4", R4_DATASET
    elif data.startswith("rt-"):
        run_prefix(data)
        label, prefix = "d" + data.rsplit("-", 1)[-1][:8], f"runs/{data}/dataset"
    else:
        raise ConfigError(f"arm {spec!r}: DATA is r4, a round id (rt-…) or LABEL=PREFIX")
    if not label.replace("-", "").isalnum() or label != label.lower():
        raise ConfigError(f"arm {spec!r}: the label must be lowercase letters, digits and '-'")
    arm = Arm(exp, label, prefix.strip("/"), int(seed))
    run_prefix(arm.run_id)
    return arm


def check_exp(exp: str) -> str:
    if not exp or not exp.startswith("nz-") or exp != exp.lower():
        raise ConfigError(f"--exp {exp!r}: an experiment id starts with nz- (lowercase), e.g. nz-20261015")
    run_prefix(exp)
    return exp


# --- the cloud: gcloud for real, a fake for dry and the tests -------------------------------------------------------

class Cloud:
    """What the experiment asks of GCP. GcloudCloud runs the gcloud CLI; FakeCloud simulates it on a local bucket."""

    def copy_prefix(self, store: Store, src: str, dst: str) -> None: ...
    def create_custom_job(self, region: str, body: dict) -> str: ...
    def custom_job(self, region: str, name: str) -> dict: ...
    def tag_digest(self, image_tag: str) -> str | None: ...
    def submit_build(self, *, region: str, source: str, config: Path, sa: str, subs: dict) -> str: ...
    def build(self, region: str, build_id: str) -> dict: ...
    def serving_digest(self, svc: dict, repo_image: str) -> str: ...
    def set_image(self, svc: dict, image: str) -> None: ...
    def warm(self, url: str, model: str, minutes: float) -> bool: ...
    def job_image(self, job: str, region: str) -> str | None: ...
    def run_job(self, job: str, region: str, args: list[str], env: dict, timeout_s: int) -> str: ...
    def execution(self, region: str, name: str) -> dict: ...
    def sleep(self, s: float) -> None: ...


Runner = Callable[[list[str], bytes | None], subprocess.CompletedProcess]


def _subprocess(argv: list[str], stdin: bytes | None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, input=stdin, capture_output=True, timeout=900)


class GcloudCloud(Cloud):
    """The real thing, through the gcloud CLI (the lab's convention: no Google client library). Nothing secret goes
    on argv: the only credential used is the caller's own identity token, sent in a header by urllib."""

    def __init__(self, project: str, *, runner: Runner = _subprocess, gcloud: str = "gcloud", log: Callable = print):
        self.project, self.run, self.gcloud, self.log = project, runner, gcloud, log

    def _g(self, *args: str, stdin: bytes | None = None, ok_missing: bool = False) -> str:
        argv = [self.gcloud, *args, f"--project={self.project}", "--quiet"]
        self.log("$ " + " ".join(argv))
        cp = self.run(argv, stdin)
        if cp.returncode != 0:
            err = (cp.stderr or b"").decode("utf-8", "replace").strip().splitlines()
            if ok_missing:
                return ""
            raise StageError(f"gcloud {' '.join(args[:3])}: exit {cp.returncode}: {(err or [''])[-1][:300]}", reason="gcloud")
        return (cp.stdout or b"").decode("utf-8", "replace").strip()

    def copy_prefix(self, store: Store, src: str, dst: str) -> None:
        self._g("storage", "rsync", "--recursive", store.url(src.strip("/")), store.url(dst.strip("/")))

    def create_custom_job(self, region: str, body: dict) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(body["jobSpec"], f)
        labels = ",".join(f"{k}={v}" for k, v in body["labels"].items())
        out = self._g("ai", "custom-jobs", "create", f"--region={region}", f"--display-name={body['displayName']}",
                      f"--config={f.name}", f"--labels={labels}", "--format=value(name)")
        return out.splitlines()[-1].strip() if out else ""

    def custom_job(self, region: str, name: str) -> dict:
        out = self._g("ai", "custom-jobs", "describe", name, f"--region={region}", "--format=json")
        return json.loads(out or "{}")

    def tag_digest(self, image_tag: str) -> str | None:
        out = self._g("artifacts", "docker", "images", "describe", image_tag, "--format=value(image_summary.digest)",
                      ok_missing=True)
        return out or None

    def submit_build(self, *, region: str, source: str, config: Path, sa: str, subs: dict) -> str:
        s = ",".join(f"{k}={v}" for k, v in subs.items())
        return self._g("builds", "submit", source, f"--region={region}", f"--config={config}", f"--service-account={sa}",
                       f"--substitutions={s}", "--async", "--format=value(id)").splitlines()[-1].strip()

    def build(self, region: str, build_id: str) -> dict:
        return json.loads(self._g("builds", "describe", build_id, f"--region={region}", "--format=json") or "{}")

    def serving_digest(self, svc: dict, repo_image: str) -> str:
        """The digest of the revision production SERVES (the largest traffic share), as the workflow's prod_model."""
        doc = json.loads(self._g("run", "services", "describe", svc["name"], f"--region={svc['region']}", "--format=json"))
        traffic = (doc.get("status") or {}).get("traffic") or []
        top = max(traffic, key=lambda t: t.get("percent") or 0) if traffic else {}
        rev = top.get("revisionName") or (doc.get("status") or {}).get("latestReadyRevisionName")
        if not rev:
            raise StageError(f"{svc['name']}: no serving revision", reason="prod_model")
        rd = json.loads(self._g("run", "revisions", "describe", rev, f"--region={svc['region']}", "--format=json"))
        img = (rd.get("status") or {}).get("imageDigest") or ""
        if "@sha256:" not in img:
            raise StageError(f"{rev}: no image digest", reason="prod_model")
        return img.split("@", 1)[1]

    def set_image(self, svc: dict, image: str) -> None:
        self._g("run", "services", "update", svc["name"], f"--region={svc['region']}", f"--image={image}")

    def warm(self, url: str, model: str, minutes: float) -> bool:
        """An empty /api/generate with keep_alive -1 (the workflow's `warm`), retried until `minutes` pass."""
        deadline = time.monotonic() + minutes * 60
        body = json.dumps({"model": model, "prompt": "", "keep_alive": -1, "stream": False}).encode()
        while True:
            token = self._g("auth", "print-identity-token")
            req = urllib.request.Request(url.rstrip("/") + "/api/generate", data=body, method="POST",
                                         headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=900) as r:  # noqa: S310 — a Cloud Run URL from infra/config.json
                    if r.status == 200:
                        return True
            except Exception as e:  # noqa: BLE001 — retried until the deadline
                self.log(f"warm {url}: {type(e).__name__}")
            if time.monotonic() + 30 > deadline:
                return False
            time.sleep(30)

    def job_image(self, job: str, region: str) -> str | None:
        doc = json.loads(self._g("run", "jobs", "describe", job, f"--region={region}", "--format=json") or "{}")
        for path in (("spec", "template", "spec", "template", "spec", "containers"), ("template", "template", "containers")):
            cur: Any = doc
            for p in path:
                cur = (cur or {}).get(p) if isinstance(cur, dict) else None
            if isinstance(cur, list) and cur:
                return cur[0].get("image")
        return None

    def run_job(self, job: str, region: str, args: list[str], env: dict, timeout_s: int) -> str:
        e = ",".join(f"{k}={v}" for k, v in env.items())
        out = self._g("run", "jobs", "execute", job, f"--region={region}", "--async", f"--args={','.join(args)}",
                      f"--update-env-vars={e}", f"--task-timeout={timeout_s}s", "--format=value(metadata.name)")
        return out.splitlines()[-1].strip()

    def execution(self, region: str, name: str) -> dict:
        doc = json.loads(self._g("run", "jobs", "executions", "describe", name, f"--region={region}", "--format=json") or "{}")
        st = doc.get("status") or {}
        conds = {c.get("type"): c for c in st.get("conditions") or []}
        done = conds.get("Completed") or {}
        state = ("succeeded" if (st.get("succeededCount") or 0) >= 1 else
                 "failed" if done.get("status") == "False" or (st.get("failedCount") or 0) >= 1 or (st.get("cancelledCount") or 0) else
                 "running")
        return {"state": state, "message": done.get("message")}

    def sleep(self, s: float) -> None:
        time.sleep(s)


# --- the record ---------------------------------------------------------------------------------------------------

def rec_path(arm: Arm) -> str:
    return f"{run_prefix(arm.run_id)}/noise.json"


def load_rec(store: Store, arm: Arm) -> dict:
    return store.read_json(rec_path(arm)) or {"exp": arm.exp, "arm": arm.name, "run_id": arm.run_id, "seed": arm.seed,
                                               "data": {"label": arm.label, "prefix": arm.prefix},
                                               "train": {"attempts": []}, "build": {}, "eval": {}}


def save_rec(store: Store, arm: Arm, rec: dict) -> None:
    rec["updated_at"] = utc()
    store.write_json(rec_path(arm), rec)


def config(store: Store) -> dict:
    cfg = store.read_json("infra/config.json")
    if not cfg:
        raise ConfigError("infra/config.json is missing from the bucket (terraform apply of infra/envs/int writes it)")
    return cfg


# --- plan ---------------------------------------------------------------------------------------------------------

def costs(n_arms: int, *, h100_rate: float = H100_RATE, h100_hours: float = H100_HOURS, l4_rate: float = L4_RATE,
          eval_hours: float = EVAL_HOURS) -> dict:
    train = n_arms * h100_hours * h100_rate
    ev = n_arms * eval_hours * 2 * l4_rate
    build = n_arms * BUILD_USD
    return {"arms": n_arms, "train_usd": round(train, 2), "eval_usd": round(ev, 2), "build_usd": round(build, 2),
            "total_usd": round(train + ev + build, 2), "claude_usd": TEACHER_FREE,
            "tags_usd_per_month": round(n_arms * TAG_USD_MONTH, 2),
            "h100_hours": round(n_arms * h100_hours, 1), "l4_hours": round(n_arms * eval_hours * 2, 1),
            "wall_h": round(h100_hours + 0.4 + n_arms * eval_hours, 1),
            "assumes": {"h100_rate": h100_rate, "h100_hours_per_arm": h100_hours, "l4_rate": l4_rate,
                        "eval_hours_per_arm": eval_hours}}


def plan_text(arms: list[Arm], base_run: str, cfg: dict | None, *, rung: str, c: dict) -> str:
    cfg = cfg or {}
    b = cfg.get("bucket", "<bucket>")
    proj = cfg.get("project", "<project>")
    lines = [f"seed-noise experiment {arms[0].exp}: {', '.join(a.name for a in arms)} (base round {base_run}, rung {rung})", ""]
    for a in arms:
        lines.append(f"  {a.name:<12} runs/{a.run_id}/  dataset ← {a.prefix}/  model:{a.candidate_id}")
    lines += ["", "what each step runs (the tool runs these; docs/RETRAIN.md R10 has them by hand):",
              f"  setup   gcloud storage rsync --recursive gs://{b}/<DATA>/ gs://{b}/runs/<arm>/dataset/ (+ run.json, prepare/, model/context.tar.gz from the base round)",
              f"  train   gcloud ai custom-jobs create --region=<rung region> --config=<jobSpec: the trainer image, args [<arm>, full], env PIPELINE_BUCKET, TRAIN_SEED, TRAIN_NONCE, …> --project={proj}",
              f"  build   gcloud builds submit gs://{b}/runs/<arm>/model/context.tar.gz --config=<infra/cloudbuild-model.json> --substitutions=_BUCKET={b},_RUN_ID=<arm>,_IMAGE=…/model:c-<arm>",
              "  eval    locks retrain + gpu-eval; gcloud run services update sous-chef-model-eval / -cand --image …@<digest>;",
              "          gcloud run jobs execute souschef-eval --args=eval,--run-id,<arm>,--current-url,…,--candidate-url,…,--mode,full",
              "  analyze σ_seed per bar set and tol_s = max(1, ⌈2·sqrt(σ_eval² + 2σ_seed²)⌉) → noise/<exp>/result.json", "",
              f"cost (estimate): training US${c['train_usd']} ({c['h100_hours']} H100-h at US${c['assumes']['h100_rate']}/h: "
              "the recorded Spot rate; Vertex bills its own rate), evals US$" f"{c['eval_usd']} ({c['l4_hours']} L4-h), builds US${c['build_usd']}, "
              f"Claude US$0 → about US${c['total_usd']}; tags about US${c['tags_usd_per_month']}/month until pruned.",
              f"wall: about {c['wall_h']} h (the arms train in parallel; the evals run one at a time on the one candidate service)."]
    return "\n".join(lines)


# --- setup --------------------------------------------------------------------------------------------------------

def setup(store: Store, cloud: Cloud, arms: list[Arm], *, base_run: str, probe: bool = True,
          trainer_image: str | None = None, rung: str | None = None, live_data: str = "r4",
          log: Callable = print) -> dict:
    base = store.read_json(f"{run_prefix(base_run)}/run.json")
    if not base or not (base.get("deployed") or {}).get("hash"):
        raise ConfigError(f"runs/{base_run}/run.json (with deployed.hash) is missing: --base-run names a finished round")
    for rel in ("prepare/manifest.json", "prepare/cases.jsonl"):
        if not store.exists(f"runs/{base_run}/{rel}"):
            raise ConfigError(f"runs/{base_run}/{rel} is missing")
    ctx = next((r for r in ("model/context.tar.gz", "model/context.tgz") if store.exists(f"runs/{base_run}/{r}")), None)
    if ctx is None:
        raise ConfigError(f"runs/{base_run}/model/context.tar.gz is missing")
    image = trainer_image or (base.get("images") or {}).get("trainer")
    if not image:
        raise ConfigError("no trainer image: --trainer-image, or a base round whose run.json names images.trainer")
    for a in arms:
        if not store.exists(f"{a.prefix}/dataset/train.jsonl"):
            raise ConfigError(f"{a.prefix}/dataset/train.jsonl is missing (arm {a.name})")
    for a in arms:
        pre = run_prefix(a.run_id)
        rec = load_rec(store, a)
        if (rec.get("setup") or {}).get("at"):
            log(f"setup {a.name}: done {rec['setup']['at']}")
            continue
        cloud.copy_prefix(store, a.prefix, f"{pre}/dataset")
        for rel in ("prepare/manifest.json", "prepare/cases.jsonl", ctx):
            store.write_bytes(f"{pre}/{rel}", store.read_bytes(f"runs/{base_run}/{rel}") or b"")
        run = {k: copy.deepcopy(v) for k, v in base.items()
               if k not in ("execution", "execution_url", "resumes", "finished_at", "checkpoints_deleted", "candidate_id",
                            "dry_rollout")}
        run.update({"run_id": a.run_id, "mode": "full", "candidate_id": a.candidate_id,
                    "images": {**(base.get("images") or {}), "trainer": image},
                    "noise": {"exp": a.exp, "arm": a.name, "seed": a.seed, "data": a.prefix, "base_run": base_run}})
        trig = dict(base.get("trigger") or {})
        run["trigger"] = ({**trig, "reason": f"seed-noise {a.exp}"} if probe else {"drift_run": None, "reason": f"seed-noise {a.exp}"})
        # every arm scores F0H / F9H with the r4 rules, whatever the base round or the eval's default: replicates are
        # pooled across arms, the live model and --extra rounds, so one scoring policy for all of them
        from .. import r4score as R4S
        run["trigger"]["score_policy"] = {k: v for k, v in R4S.STRICT_POLICY.items() if k != "font"}
        run["trigger"]["font_policy"] = R4S.STRICT_POLICY["font"]
        store.write_json(f"{pre}/run.json", run)
        # the seed twice over: TRAIN_SEED in the job's env, and train-config.json, which pipeline-train.sh reads when
        # the env does not say (train.py pipeline_config: env over file); adapter/run.json is checked after DONE
        store.write_json(f"{pre}/train-config.json", {"seed": a.seed})
        rec["setup"] = {"at": utc(), "base_run": base_run, "trainer_image": image, "probe": probe,
                        "deployed": run["deployed"]["hash"]}
        save_rec(store, a, rec)
        log(f"setup {a.name}: runs/{a.run_id}/ from {a.prefix} and {base_run}")
    plan = {"exp": arms[0].exp, "arms": [a.name for a in arms], "run_ids": [a.run_id for a in arms],
            "base_run": base_run, "live_data": live_data, "probe": probe, "trainer_image": image, "rung": rung,
            "at": utc()}
    old = store.read_json(f"noise/{arms[0].exp}/plan.json") or {}
    if old.get("arms"):
        plan["arms"] = sorted(set(old["arms"]) | set(plan["arms"]))
        plan["run_ids"] = sorted(set(old.get("run_ids") or []) | set(plan["run_ids"]))
    store.write_json(f"noise/{arms[0].exp}/plan.json", plan)
    return plan


# --- train --------------------------------------------------------------------------------------------------------

def pick_rung(cfg: dict, name: str | None) -> dict:
    ladder = ((cfg.get("trainer") or {}).get("ladder")) or []
    if not ladder:
        raise ConfigError("infra/config.json has no trainer ladder")
    if not name:
        return ladder[0]
    for r in ladder:
        if r.get("name") == name:
            return r
    raise ConfigError(f"--rung {name!r}: one of {', '.join(r['name'] for r in ladder)}")


def next_rung(cfg: dict, name: str | None, attempts: list[dict]) -> dict:
    """The rung for the next attempt, as the workflow's ladder walks it: the first one (or --rung) to begin with; after
    an attempt that NEVER STARTED (a Flex-start wait that ran out: no startTime, no progress written) or once a rung
    has used its own `attempts`, the next rung down the ladder (the last rung is kept). A preempted attempt that did
    start resumes on the same rung. Without this, a Flex rung that never gets capacity would be retried up to
    max_attempts times, each one a full max_wait_s of idle waiting."""
    ladder = ((cfg.get("trainer") or {}).get("ladder")) or []
    if not attempts:
        return pick_rung(cfg, name)
    last = attempts[-1]
    names = [r.get("name") for r in ladder]
    cur = pick_rung(cfg, last.get("rung") or name)
    i = names.index(cur["name"]) if cur.get("name") in names else 0
    used = sum(1 for a in attempts if a.get("rung") == cur["name"])
    if (last.get("never_started") or used >= int(cur.get("attempts") or 1 << 30)) and i + 1 < len(ladder):
        return ladder[i + 1]
    return cur


def job_body(cfg: dict, arm: Arm, rung: dict, *, image: str, attempt: int, nonce: str) -> dict:
    """The CustomJob the workflow's vertex_attempt sends, for one arm, plus TRAIN_SEED; labelled sous-chef-noise
    (never sous-chef-pipeline: a round adopts or cancels only its own jobs)."""
    tr = cfg.get("trainer") or {}
    sched: dict[str, Any] = {"timeout": f"{int(tr.get('timeout_s', 43200))}s", "strategy": rung["strategy"]}
    if rung["strategy"] == "FLEX_START":
        sched["maxWaitDuration"] = f"{int(rung.get('max_wait_s') or 14400)}s"
    env = {"PIPELINE_BUCKET": f"gs://{cfg['bucket']}", "GOOGLE_CLOUD_PROJECT": cfg["project"], "TRAIN_RUNG": rung["name"],
           "TRAIN_ATTEMPT": str(attempt), "TRAIN_NONCE": nonce, "TRAIN_IMAGE": image, "TRAIN_SEED": str(arm.seed)}
    return {"displayName": f"souschef-noise-{arm.run_id}-{rung['name']}-{attempt}",
            "labels": {"purpose": "sous-chef-noise", "run": arm.run_id, "rung": rung["name"], "create": nonce},
            "jobSpec": {"workerPoolSpecs": [{"machineSpec": {"machineType": rung["machine_type"],
                                                              "acceleratorType": rung["accelerator_type"], "acceleratorCount": 1},
                                             "replicaCount": 1,
                                             "diskSpec": {"bootDiskType": "pd-ssd", "bootDiskSizeGb": int(tr.get("disk_gb", 500))},
                                             "containerSpec": {"imageUri": image, "args": [arm.run_id, "full"],
                                                               "env": [{"name": k, "value": v} for k, v in env.items()]}}],
                        "scheduling": sched, "serviceAccount": tr.get("service_account")}}


def trained_seed(store: Store, arm: Arm) -> int | None:
    run = store.read_json(f"{run_prefix(arm.run_id)}/train/adapter/run.json") or {}
    s = run.get("seed", (run.get("recipe") or {}).get("seed"))
    return int(s) if isinstance(s, (int, str)) and str(s).isdigit() else None


def train_step(store: Store, cloud: Cloud, cfg: dict, arm: Arm, *, rung_name: str | None, max_attempts: int,
               log: Callable = print) -> str:
    """One look at an arm's training: submit, leave running, resubmit after a preemption, or close it.
    → done | running | failed | seed_mismatch."""
    pre = run_prefix(arm.run_id)
    rec = load_rec(store, arm)
    if not (rec.get("setup") or {}).get("at"):
        raise ConfigError(f"{arm.name}: run `setup` first")
    tr = rec.setdefault("train", {"attempts": []})
    if tr.get("state") in ("done", "failed", "seed_mismatch"):
        return tr["state"]
    if store.exists(f"{pre}/train/DONE"):
        seed = trained_seed(store, arm)
        tr["seed_trained"] = seed
        tr["state"] = "done" if seed == arm.seed else "seed_mismatch"
        if tr["state"] == "seed_mismatch":
            tr["error"] = (f"the adapter's run.json says seed {seed}, not {arm.seed}: this trainer image's "
                           "pipeline-train.sh ignores TRAIN_SEED (rebuild it: make trainer-image TAG=<tag>, then --trainer-image)")
        save_rec(store, arm, rec)
        log(f"train {arm.name}: {tr['state']}" + (f" ({tr.get('error')})" if tr.get("error") else ""))
        return tr["state"]
    att = tr["attempts"][-1] if tr["attempts"] else None
    if att and att.get("job") and att.get("state") not in TERMINAL:
        job = cloud.custom_job(att["region"], att["job"])
        att["state"] = job.get("state") or att.get("state")
        att["started"] = bool(job.get("startTime")) or att.get("started", False)
        if att["state"] not in TERMINAL:
            save_rec(store, arm, rec)
            return "running"
        att["ended_at"] = utc()
    if att and att.get("state") in TERMINAL:
        if store.exists(f"{pre}/train/DONE"):
            save_rec(store, arm, rec)
            return train_step(store, cloud, cfg, arm, rung_name=rung_name, max_attempts=max_attempts, log=log)
        prog = store.read_json(f"{pre}/train/progress.json") or {}
        failed = store.exists(f"{pre}/train/failed-{att.get('nonce')}.json") or (
            prog.get("nonce") == att.get("nonce") and prog.get("status") == "failed")
        if failed:
            tr["state"], tr["error"] = "failed", f"trainer failed at {prog.get('step')}: {prog.get('error')}"
            save_rec(store, arm, rec)
            log(f"train {arm.name}: failed ({tr['error']}); read runs/{arm.run_id}/train/train.log")
            return "failed"
        if len(tr["attempts"]) >= max_attempts:
            tr["state"], tr["error"] = "failed", f"{len(tr['attempts'])} attempts ended without DONE (last {att['state']})"
            save_rec(store, arm, rec)
            return "failed"
        never = not att.get("started") and prog.get("nonce") != att.get("nonce")
        att["never_started"] = never
        log(f"train {arm.name}: attempt {len(tr['attempts'])} ended {att['state']} "
            f"({'preempted' if not never else 'never started'}); resubmitting, resuming from the checkpoints")
    rung = next_rung(cfg, rung_name, tr["attempts"])
    n = len(tr["attempts"]) + 1
    nonce = f"nz{hashlib.sha1(f'{arm.run_id}-{n}-{time.time()}'.encode()).hexdigest()[:8]}-{int(time.time() * 1000)}"
    image = rec["setup"]["trainer_image"]
    name = cloud.create_custom_job(rung["region"], job_body(cfg, arm, rung, image=image, attempt=n, nonce=nonce))
    tr["attempts"].append({"attempt": n, "job": name, "region": rung["region"], "rung": rung["name"], "nonce": nonce,
                           "state": "JOB_STATE_QUEUED", "submitted_at": utc()})
    tr["state"] = "running"
    save_rec(store, arm, rec)
    log(f"train {arm.name}: submitted {name} ({rung['name']}, attempt {n}, seed {arm.seed})")
    return "running"


def train(store: Store, cloud: Cloud, cfg: dict, arms: list[Arm], *, rung: str | None, wait: bool, poll_s: float = 300,
          max_attempts: int = 4, log: Callable = print) -> dict[str, str]:
    while True:
        st = {a.name: train_step(store, cloud, cfg, a, rung_name=rung, max_attempts=max_attempts, log=log) for a in arms}
        if not wait or all(s != "running" for s in st.values()):
            return st
        cloud.sleep(poll_s)


# --- build --------------------------------------------------------------------------------------------------------

def build_step(store: Store, cloud: Cloud, cfg: dict, arm: Arm, *, spec: Path, log: Callable = print) -> str:
    pre = run_prefix(arm.run_id)
    rec = load_rec(store, arm)
    if (rec.get("train") or {}).get("state") != "done":
        return "waiting"
    bd = rec.setdefault("build", {})
    if bd.get("state") == "done":
        return "done"
    repo = cfg["model_repo"]["image"]
    image = f"{repo}:{arm.candidate_id}"
    region = (cfg.get("build") or {}).get("region", "us-east4")
    if bd.get("build_id") and bd.get("state") == "running":
        b = cloud.build(region, bd["build_id"])
        status = b.get("status")
        if status in ("QUEUED", "PENDING", "WORKING", "STATUS_UNKNOWN", None):
            return "running"
        if status != "SUCCESS":
            bd.update(state="failed", error=f"build {bd['build_id']} {status}")
            save_rec(store, arm, rec)
            return "failed"
    digest = cloud.tag_digest(image)
    if digest:
        bd.update(state="done", image=image, digest=digest, at=utc())
        store.write_json(f"{pre}/model/image.json", {"image": image, "tag": arm.candidate_id, "digest": digest,
                                                     "build_id": bd.get("build_id"), "at": utc(),
                                                     "note": "seed-noise arm (retrain/noise.py)"})
        save_rec(store, arm, rec)
        log(f"build {arm.name}: {image} = {digest[:19]}…")
        return "done"
    if bd.get("state") == "running":
        bd.update(state="failed", error="the build succeeded but the tag has no digest")
        save_rec(store, arm, rec)
        return "failed"
    ctx = next((f"{pre}/{r}" for r in ("model/context.tar.gz", "model/context.tgz") if store.exists(f"{pre}/{r}")), None)
    if not ctx:
        raise ConfigError(f"{pre}/model/context.tar.gz is missing (setup copies it)")
    sa = f"projects/{cfg['project']}/serviceAccounts/{(cfg.get('build') or {}).get('service_account')}"
    bd["build_id"] = cloud.submit_build(region=region, source=store.url(ctx), config=spec, sa=sa,
                                        subs={"_BUCKET": cfg["bucket"], "_RUN_ID": arm.run_id, "_IMAGE": image})
    bd.update(state="running", image=image, submitted_at=utc())
    save_rec(store, arm, rec)
    log(f"build {arm.name}: submitted {bd['build_id']}")
    return "running"


def build_all(store: Store, cloud: Cloud, cfg: dict, arms: list[Arm], *, spec: Path | None, wait: bool,
              poll_s: float = 60, log: Callable = print) -> dict[str, str]:
    tmp = None
    if spec is None:
        tmp = Path(tempfile.mkdtemp()) / "cloudbuild-model.json"
        data = store.read_bytes((cfg.get("build") or {}).get("spec") or "infra/cloudbuild-model.json")
        if not data:
            raise ConfigError("infra/cloudbuild-model.json is missing (terraform writes it; or --build-config)")
        tmp.write_bytes(data)
        spec = tmp
    while True:
        st = {a.name: build_step(store, cloud, cfg, a, spec=spec, log=log) for a in arms}
        if not wait or all(s not in ("running",) for s in st.values()):
            return st
        cloud.sleep(poll_s)


# --- eval ---------------------------------------------------------------------------------------------------------

def eval_arm(store: Store, cloud: Cloud, cfg: dict, arm: Arm, *, poll_s: float = 300, lab_image: str | None = None,
             log: Callable = print) -> str:
    """The paired eval of one arm against the live model, exactly as the workflow's eval stage runs it, under the
    retrain and gpu-eval locks (no round may re-point the candidate service, no drift check share the eval GPU)."""
    from .. import gcsio
    pre = run_prefix(arm.run_id)
    rec = load_rec(store, arm)
    ev = rec.setdefault("eval", {})
    if ev.get("state") == "done":
        return "done"
    if ev.get("state") == "failed":  # run it again from the start: a new execution
        ev.clear()
    if (rec.get("build") or {}).get("state") != "done":
        return "waiting"
    holder = f"noise:{arm.run_id}"
    try:
        store.b.acquire(RETRAIN_LOCK, arm.run_id, EVAL_LOCK_TTL_S, holder=holder)
    except gcsio.LockHeld as e:
        raise ConfigError(f"state/locks/retrain.json is held ({e}): a round is running; eval when it has finished") from None
    try:
        try:
            store.b.acquire(EVAL_LOCK, arm.run_id, EVAL_LOCK_TTL_S, holder=holder)
        except gcsio.LockHeld as e:
            raise ConfigError(f"state/locks/gpu-eval.json is held ({e}): a drift check is running; try again later") from None
        try:
            return _eval_locked(store, cloud, cfg, arm, rec, ev, poll_s=poll_s, lab_image=lab_image, log=log)
        finally:
            store.b.release(EVAL_LOCK, arm.run_id)
    finally:
        store.b.release(RETRAIN_LOCK, arm.run_id)


def _eval_locked(store: Store, cloud: Cloud, cfg: dict, arm: Arm, rec: dict, ev: dict, *, poll_s: float,
                 lab_image: str | None, log: Callable) -> str:
    pre = run_prefix(arm.run_id)
    jobs_region = cfg.get("jobs_region")
    job = (cfg.get("jobs") or {}).get("eval", "souschef-eval")
    model = store.read_json("state/model.json") or {}
    if not ev.get("execution"):
        repo = cfg["model_repo"]["image"]
        prod_digest = cloud.serving_digest(cfg["prod_model"], repo)
        if model.get("digest") and model["digest"] != prod_digest:
            raise ConfigError(f"production serves {prod_digest[:19]}…, not state/model.json's {model['digest'][:19]}…: "
                              "the live model changed; the arms would not share one reference")
        cand = rec["build"]["digest"]
        cloud.set_image(cfg["eval_model"], f"{repo}@{prod_digest}")
        cloud.set_image(cfg["cand_model"], f"{repo}@{cand}")
        mins = float(cfg.get("l4_wait_minutes") or 360)
        if not (cloud.warm(cfg["eval_model"]["url"], cfg.get("ollama_model", "sous-chef"), mins)
                and cloud.warm(cfg["cand_model"]["url"], cfg.get("ollama_model", "sous-chef"), mins)):
            raise StageError("the eval or candidate GPU service did not answer (no L4 in us-east4?); run `eval` again later",
                             reason="no_l4")
        img = cloud.job_image(job, jobs_region)
        if lab_image and img != lab_image:
            raise ConfigError(f"job {job} runs {img}, not --lab-image {lab_image}: "
                              f"gcloud run jobs update {job} --region {jobs_region} --image {lab_image}")
        args = ["eval", "--run-id", arm.run_id, "--current-url", cfg["eval_model"]["url"],
                "--candidate-url", cfg["cand_model"]["url"], "--mode", "full"]
        env = {"PIPELINE_RUN_ID": arm.run_id, "PIPELINE_CANDIDATE_ID": arm.candidate_id,
               "PROD_MODEL_DIGEST": prod_digest, "CAND_MODEL_DIGEST": cand}
        ev.update(execution=cloud.run_job(job, jobs_region, args, env, EVAL_TIMEOUT_S), state="running",
                  current_digest=prod_digest, candidate_digest=cand, lab_image=img, started_at=utc())
        save_rec(store, arm, rec)
        log(f"eval {arm.name}: {ev['execution']} (current {prod_digest[:19]}…, candidate {cand[:19]}…)")
    while True:
        x = cloud.execution(jobs_region, ev["execution"])
        if x["state"] != "running":
            break
        store.b.refresh(RETRAIN_LOCK, arm.run_id, EVAL_LOCK_TTL_S)
        store.b.refresh(EVAL_LOCK, arm.run_id, EVAL_LOCK_TTL_S)
        cloud.sleep(poll_s)
    ok = x["state"] == "succeeded" and store.exists(f"{pre}/eval/bar.json")
    ev.update(state="done" if ok else "failed", finished_at=utc(), message=x.get("message"))
    save_rec(store, arm, rec)
    log(f"eval {arm.name}: {ev['state']}" + ("" if ok else f" ({x.get('message')}); runs/{arm.run_id}/result-eval.json"))
    return ev["state"]


def eval_all(store: Store, cloud: Cloud, cfg: dict, arms: list[Arm], **kw) -> dict[str, str]:
    out = {}
    for a in arms:
        out[a.name] = eval_arm(store, cloud, cfg, a, **kw)
        if out[a.name] == "failed":
            break
    return out


# --- analysis -----------------------------------------------------------------------------------------------------

def set_of(row: dict) -> str:
    s = str(row.get("set") or "")
    if s == "F0H":
        return "F0H:seen" if row.get("seen") else "F0H:unseen"
    return s


def replicates(rows: list[dict]) -> dict[str, list[dict[str, int]]]:
    """eval/<arm>/rows.jsonl → {set: [replicate {row key: 0|1}]}. The eval stage writes no replicate index; a key's
    k-th occurrence within its set is replicate k (arm_job asks every set once per replicate). Error rows are left
    out, as the bar leaves them out."""
    out: dict[str, list[dict[str, int]]] = {}
    seen: dict[tuple[str, str], int] = {}
    for r in rows:
        s = set_of(r)
        key = str(r.get("key") or f"{s}:{r.get('id')}")
        k = seen.get((s, key), 0)
        seen[(s, key)] = k + 1
        reps = out.setdefault(s, [])
        while len(reps) <= k:
            reps.append({})
        if r.get("status") == "ok":
            reps[k][key] = int(r.get("pass") or 0)
    return out


def _row_var(reps: list[dict[str, int]], keys: list[str]) -> float | None:
    """The unbiased variance of one replicate's total over `keys` (baseline.sigma_run's K/(K−1) Σ p(1−p))."""
    k = len(reps)
    if k < 2:
        return None
    tot = 0.0
    for key in keys:
        p = sum(r.get(key, 0) for r in reps) / k
        tot += p * (1 - p)
    return k / (k - 1) * tot


def _gammainc_lower(a: float, x: float) -> float:
    """The regularized lower incomplete gamma P(a, x) (series / continued fraction; Numerical Recipes 6.2)."""
    if x <= 0:
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s = a, 1.0 / a
        d = s
        for _ in range(500):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-12:
                break
        return s * math.exp(-x + a * math.log(x) - gln)
    b, c, d = x + 1 - a, 1e300, 1 / (x + 1 - a)
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        h *= d * c
        if abs(d * c - 1) < 1e-12:
            break
    return 1 - math.exp(-x + a * math.log(x) - gln) * h


def chi2_ppf(q: float, df: int) -> float:
    lo, hi = 0.0, max(10.0, df * 10.0)
    for _ in range(200):
        mid = (lo + hi) / 2
        if _gammainc_lower(df / 2, mid / 2) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def seed_aware_tol(sigma_eval: float, sigma_seed: float, draws: int = 2) -> int:
    """THE bar tolerance (rows), as bar.set_row computes it: max(1, ⌈2·sqrt(σ_eval² + draws·σ_seed²)⌉). σ_seed is ONE
    model's seed spread; the bar compares two trained models (candidate and live), both seed draws, so draws = 2.
    draws = 1 is the one-model figure (reported as tol_one_draw)."""
    return max(1, math.ceil(2 * math.sqrt(sigma_eval ** 2 + draws * sigma_seed ** 2) - 1e-9))


@dataclass
class Model:
    name: str           # live | <label>@<seed>
    data: str           # the dataset label (the group σ_seed is measured within)
    reps: dict[str, list[dict[str, int]]] = field(default_factory=dict)
    evals: list[str] = field(default_factory=list)


def gather(store: Store, arms: list[Arm], *, live_data: str, extra: list[tuple[str, str, int]] = (),
           log: Callable = print) -> tuple[dict[str, Model], dict[str, dict], list[str]]:
    """Every model's replicates over every eval: the arms' candidates, the live model (the current arm of every eval,
    pooled), and extra rounds (`rt-…=DATA@SEED`: a round's candidate as one more draw, its current arm as live)."""
    models: dict[str, Model] = {}
    bars: dict[str, dict] = {}
    warn: list[str] = []
    digests: set[str] = set()
    sources = [(a.run_id, a.name, a.label, a.seed) for a in arms] + [(rid, f"{lab}@{s}", lab, s) for rid, lab, s in extra]
    for run_id, name, label, seed in sources:
        pre = run_prefix(run_id)
        bar = store.read_json(f"{pre}/eval/bar.json")
        if not bar:
            warn.append(f"{name}: no runs/{run_id}/eval/bar.json (not evaluated): left out")
            continue
        rec = store.read_json(f"{pre}/noise.json") or {}
        if rec and (rec.get("train") or {}).get("state") == "seed_mismatch":
            warn.append(f"{name}: trained with seed {(rec.get('train') or {}).get('seed_trained')}, not {seed}: left out")
            continue
        cur_d = ((rec.get("eval") or {}).get("current_digest")
                 or ((store.read_json(f"{pre}/eval/finished.json") or {}).get("current") or {}).get("digest"))
        if cur_d:
            digests.add(cur_d)
        bars[name] = bar
        cand = models.setdefault(name, Model(name, label))
        cand.evals.append(run_id)
        for s, reps in replicates(store.read_jsonl(f"{pre}/eval/candidate/rows.jsonl")).items():
            cand.reps.setdefault(s, []).extend(reps)
        live = models.setdefault("live", Model("live", live_data))
        live.evals.append(run_id)
        for s, reps in replicates(store.read_jsonl(f"{pre}/eval/current/rows.jsonl")).items():
            live.reps.setdefault(s, []).extend(reps)
    if len(digests) > 1:
        warn.append(f"the current arm was {len(digests)} different digests across the evals: 'live' mixes models")
    return models, bars, warn


def analyze_set(s: str, models: dict[str, Model], bars: dict[str, dict], *, q_hi: float = 0.90) -> dict | None:
    have = {n: m for n, m in models.items() if m.reps.get(s)}
    if len(have) < 2:
        return None
    keysets = [set(r) for m in have.values() for r in m.reps[s]]
    keys = sorted(set.intersection(*keysets)) if keysets else []
    if not keys:
        return None
    per: dict[str, dict] = {}
    for n, m in have.items():
        counts = [sum(r.get(k, 0) for k in keys) for r in m.reps[s]]
        per[n] = {"data": m.data, "k": len(counts), "mean": statistics.fmean(counts), "counts": counts,
                  "var_rep": _row_var(m.reps[s], keys)}
    vs = [p["var_rep"] for p in per.values() if p["var_rep"] is not None]
    v_pool = statistics.fmean(vs) if vs else None
    groups: dict[str, list[str]] = {}
    for n, p in per.items():
        groups.setdefault(p["data"], []).append(n)
    ss = df = 0.0
    e_num = 0.0
    by_group = {}
    for g, names in groups.items():
        if len(names) < 2:
            continue
        means = [per[n]["mean"] for n in names]
        s2 = statistics.variance(means)
        e = statistics.fmean([(per[n]["var_rep"] if per[n]["var_rep"] is not None else (v_pool or 0.0)) / per[n]["k"]
                              for n in names])
        by_group[g] = {"models": names, "means": [round(x, 2) for x in means], "s2": round(s2, 3), "eval_part": round(e, 3)}
        ss += (len(names) - 1) * s2
        e_num += (len(names) - 1) * e
        df += len(names) - 1
    if df == 0:
        return None
    s2_pool, e_pool = ss / df, e_num / df
    sig_seed = math.sqrt(max(0.0, s2_pool - e_pool))
    sig_seed_hi = math.sqrt(max(0.0, s2_pool * df / chi2_ppf(1 - q_hi, int(df)) - e_pool))
    # σ_eval: the bar's own σ_Δ for this set (what tol_s is built from today), median over the arms' evals
    bar_sd = [r["sigma_delta"] for b in bars.values() for r in b.get("sets") or []
              if r.get("set") == s and r.get("sigma_delta") is not None]
    sig_eval = statistics.median(bar_sd) if bar_sd else (math.sqrt(v_pool) if v_pool is not None else 1.0)
    deltas = {}
    for name, b in bars.items():
        row = next((r for r in b.get("sets") or [] if r.get("set") == s), None)
        if row:
            deltas[name] = {"delta": row.get("delta"), "b": row.get("b"), "c": row.get("c"), "tol": row.get("tol"),
                            "n": row.get("n")}
    return {"set": s, "n": len(keys), "models": {n: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()}
                                                  for n, p in per.items()},
            "groups": by_group, "df": int(df), "s2": round(s2_pool, 3), "eval_part": round(e_pool, 3),
            "sigma_run": round(math.sqrt(v_pool), 3) if v_pool is not None else None,
            "sigma_eval": round(sig_eval, 3), "sigma_seed": round(sig_seed, 3), "sigma_seed_hi": round(sig_seed_hi, 3),
            "q_hi": q_hi, "tol_now": max(1, math.ceil(2 * sig_eval - 1e-9)),
            "tol_seed": seed_aware_tol(sig_eval, sig_seed), "tol_seed_hi": seed_aware_tol(sig_eval, sig_seed_hi),
            "tol_one_draw": seed_aware_tol(sig_eval, sig_seed, draws=1),
            "paired_vs_live": deltas,
            "max_abs_delta": max((abs(d["delta"]) for d in deltas.values() if d.get("delta") is not None), default=None)}


def analyze(store: Store, arms: list[Arm], *, live_data: str = "r4", extra: list[tuple[str, str, int]] = (),
            compare: list[str] = (), q_hi: float = 0.90, write: bool = True, log: Callable = print) -> dict:
    models, bars, warn = gather(store, arms, live_data=live_data, extra=extra, log=log)
    sets = {}
    for s in BAR_SETS + REPORTED_SETS:
        r = analyze_set(s, models, bars, q_hi=q_hi)
        if r:
            r["bar"] = s in BAR_SETS
            sets[s] = r
    cmp = {}
    for rid in compare:
        b = store.read_json(f"{run_prefix(rid)}/eval/bar.json") or {}
        rows = {}
        for r in b.get("sets") or []:
            a = sets.get(r.get("set"))
            if not a or r.get("delta") is None or not r.get("bar"):
                continue
            d = r["delta"]
            rows[r["set"]] = {"delta": d, "tol_then": r.get("tol"), "tol_seed": a["tol_seed"],
                              "passes_then": d >= -(r.get("tol") or 0), "passes_seed": d >= -a["tol_seed"]}
        cmp[rid] = rows
    ho = sets.get("holdout") or {}
    r4_arms = [n for n, m in models.items() if n != "live" and m.data == live_data]
    ho_moves = [abs((ho.get("paired_vs_live") or {}).get(n, {}).get("delta") or 0) for n in r4_arms]
    verdict = ("no seed arm evaluated" if not ho_moves else
               f"holdout moved {max(ho_moves):g} rows on a retrain of the live model's data: a ≥ 3-row loss can be seed noise; "
               "fold σ_seed into the bar" if max(ho_moves) >= 3 else
               f"holdout stayed within ±{max(ho_moves):g} rows on every retrain: the noise explanation is weakened; look at the "
               "data links (PLAN causes 6–8) and the train/serve path" if max(ho_moves) <= 1 else
               f"holdout moved up to {max(ho_moves):g} rows on a retrain: inconclusive; read σ_seed and its upper bound")
    res = {"exp": arms[0].exp if arms else None, "at": utc(), "live_data": live_data,
           "models": {n: {"data": m.data, "evals": m.evals} for n, m in models.items()},
           "sets": sets, "compare": cmp, "holdout_verdict": verdict, "warnings": warn,
           "thresholds_override": thresholds_override(sets, arms[0].exp if arms else ""),
           "formula": "tol_s = max(1, ceil(2*sqrt(sigma_eval_s^2 + 2*sigma_seed_s^2))) (candidate and live are both seed draws); sigma_seed_s^2 = max(0, S^2 - mean(var_rep/K)) "
                      "over the models of one dataset (S^2 their means' sample variance), pooled over datasets"}
    if write and arms:
        store.write_json(f"noise/{arms[0].exp}/result.json", res)
    return res


def thresholds_override(sets: dict, exp: str) -> dict:
    """What config/thresholds.json would carry once bar.py reads it (the integrator's wiring):
    {"sigma_seed": {set: rows}}: ONE model's seed spread; bar.set_row counts it for both models,
    tol = max(1, ⌈2·sqrt(σ_Δ² + 2·sigma_seed[set]²)⌉)."""
    return {"sigma_seed": {s: r["sigma_seed"] for s, r in sets.items() if r.get("bar")},
            "_sigma_seed_from": f"noise/{exp}/result.json"}


def table(res: dict) -> str:
    lines = [f"{'set':<22}{'n':>5}{'df':>4}{'σ_eval':>8}{'σ_seed':>8}{'(hi)':>7}{'tol now':>9}{'→ seed':>8}{'(hi)':>6}  Δ vs live per arm"]
    for s, r in res["sets"].items():
        ds = ", ".join(f"{n} {d['delta']:+g}" for n, d in r["paired_vs_live"].items() if d.get("delta") is not None)
        lines.append(f"{s + ('' if r.get('bar') else ' (rep.)'):<22}{r['n']:>5}{r['df']:>4}{r['sigma_eval']:>8.2f}"
                     f"{r['sigma_seed']:>8.2f}{r['sigma_seed_hi']:>7.2f}{r['tol_now']:>9}{r['tol_seed']:>8}{r['tol_seed_hi']:>6}  {ds}")
    lines.append(res["holdout_verdict"])
    for rid, rows in (res.get("compare") or {}).items():
        lines.append(f"{rid}: " + "; ".join(f"{s} {v['delta']:+g} (tol {v['tol_then']} → {v['tol_seed']}: "
                                             f"{'pass' if v['passes_then'] else 'fail'} → {'pass' if v['passes_seed'] else 'fail'})"
                                             for s, v in rows.items()))
    lines += [f"warning: {w}" for w in res.get("warnings") or []]
    return "\n".join(lines)


def status(store: Store, arms: list[Arm]) -> list[str]:
    out = []
    for a in arms:
        rec = store.read_json(rec_path(a)) or {}
        tr, bd, ev = rec.get("train") or {}, rec.get("build") or {}, rec.get("eval") or {}
        att = (tr.get("attempts") or [{}])[-1]
        prog = store.read_json(f"runs/{a.run_id}/train/progress.json") or {}
        out.append(f"{a.name:<12} setup {'ok' if rec.get('setup') else '-':<3} train {tr.get('state', '-')}"
                   f" ({len(tr.get('attempts') or [])} att, {att.get('state', '-')}, step {prog.get('step', '-')}, "
                   f"{prog.get('last_step', '-')}/{prog.get('updates', '-')}) build {bd.get('state', '-')} eval {ev.get('state', '-')}")
    return out


# --- the fake cloud (dry, tests) --------------------------------------------------------------------------------------

class FakeCloud(Cloud):
    """GCP on a local bucket: a Vertex job that 'trains' on the second look (writes what pipeline-train.sh leaves),
    a build that pushes on the second look, an eval job that writes eval rows from a synthetic model of the exam.

    outcomes: {arm run id: ["preempt" | "nostart" | "fail" | "ok", …]} per attempt (default ok). honour_seed False = a trainer
    image whose pipeline-train.sh ignores TRAIN_SEED (it trains 3407). seed_sd: how far a seed moves a row's odds."""

    SETS = {"paraphrase": 147, "holdout": 62, "real": 14, "followup": 12, "out_of_scope": 13, "out_of_scope_holdout": 4,
            "F9H": 16, "probe": 17}

    def __init__(self, store: Store, *, outcomes: dict | None = None, honour_seed: bool = True, seed_flip: float = 0.06):
        self.store, self.outcomes, self.honour_seed, self.seed_flip = store, outcomes or {}, honour_seed, seed_flip
        self.jobs: dict[str, dict] = {}
        self.builds: dict[str, dict] = {}
        self.execs: dict[str, dict] = {}
        self.tags: dict[str, str] = {}
        self.images: dict[str, str] = {"sous-chef-model": "sha256:" + "a" * 64}
        self.calls: list[tuple] = []

    def copy_prefix(self, store: Store, src: str, dst: str) -> None:
        self.calls.append(("copy", src, dst))
        s, d = store.b.path(src.strip("/")), store.b.path(dst.strip("/"))
        shutil.copytree(s, d, dirs_exist_ok=True)

    def create_custom_job(self, region: str, body: dict) -> str:
        name = f"projects/0/locations/{region}/customJobs/{len(self.jobs) + 1}"
        env = {e["name"]: e["value"] for e in body["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["env"]}
        run_id = body["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["args"][0]
        n = int(env["TRAIN_ATTEMPT"])
        plan = self.outcomes.get(run_id) or []
        self.jobs[name] = {"state": "JOB_STATE_QUEUED", "env": env, "run_id": run_id, "looks": 0,
                           "outcome": plan[n - 1] if n - 1 < len(plan) else "ok"}
        self.calls.append(("job", region, body))
        return name

    def custom_job(self, region: str, name: str) -> dict:
        j = self.jobs[name]
        j["looks"] += 1
        if j["state"] not in TERMINAL and j["looks"] >= 2:
            pre = f"runs/{j['run_id']}/train"
            out = j["outcome"]
            if out == "ok":
                seed = int(j["env"]["TRAIN_SEED"]) if self.honour_seed else RECIPE_SEED
                self.store.write_json(f"{pre}/adapter/run.json", {"seed": seed, "updates": 601})
                self.store.write_json(f"{pre}/progress.json", {"status": "done", "step": "DONE", "nonce": j["env"]["TRAIN_NONCE"]})
                self.store.write_bytes(f"{pre}/DONE", b"fake\n")
                j["state"] = "JOB_STATE_SUCCEEDED"
            elif out == "nostart":  # a Flex-start wait that ran out: the job never got a machine
                j["state"] = "JOB_STATE_FAILED"
            elif out == "preempt":
                self.store.write_json(f"{pre}/progress.json", {"status": "interrupted", "step": "SFT", "nonce": j["env"]["TRAIN_NONCE"]})
                j["state"] = "JOB_STATE_CANCELLED"
            else:
                self.store.write_json(f"{pre}/progress.json", {"status": "failed", "step": "SFT", "error": "boom",
                                                               "nonce": j["env"]["TRAIN_NONCE"]})
                self.store.write_json(f"{pre}/failed-{j['env']['TRAIN_NONCE']}.json", {"status": "failed"})
                j["state"] = "JOB_STATE_FAILED"
        return {"state": j["state"], "startTime": None if j["outcome"] == "nostart" else ("x" if j["looks"] >= 1 else None)}

    def tag_digest(self, image_tag: str) -> str | None:
        return self.tags.get(image_tag)

    def submit_build(self, *, region: str, source: str, config: Path, sa: str, subs: dict) -> str:
        bid = f"b{len(self.builds) + 1}"
        self.builds[bid] = {"looks": 0, "image": subs["_IMAGE"]}
        self.calls.append(("build", source, subs))
        return bid

    def build(self, region: str, build_id: str) -> dict:
        b = self.builds[build_id]
        b["looks"] += 1
        if b["looks"] >= 2:
            self.tags[b["image"]] = "sha256:" + hashlib.sha256(b["image"].encode()).hexdigest()
            return {"status": "SUCCESS"}
        return {"status": "WORKING"}

    def serving_digest(self, svc: dict, repo_image: str) -> str:
        return self.images[svc["name"]]

    def set_image(self, svc: dict, image: str) -> None:
        self.calls.append(("set_image", svc["name"], image))
        self.images[svc["name"]] = image.split("@", 1)[1]

    def warm(self, url: str, model: str, minutes: float) -> bool:
        self.calls.append(("warm", url))
        return True

    def job_image(self, job: str, region: str) -> str | None:
        return "…/sous-chef-lab:fake"

    def run_job(self, job: str, region: str, args: list[str], env: dict, timeout_s: int) -> str:
        name = f"{job}-x{len(self.execs) + 1}"
        self.execs[name] = {"looks": 0, "run_id": args[args.index("--run-id") + 1], "env": env}
        self.calls.append(("run_job", job, args, env))
        assert self.store.b.read_lock(EVAL_LOCK) and self.store.b.read_lock(RETRAIN_LOCK), "eval without the locks"
        return name

    def execution(self, region: str, name: str) -> dict:
        x = self.execs[name]
        x["looks"] += 1
        if x["looks"] < 2:
            return {"state": "running"}
        self._fake_eval(x["run_id"], x["env"])
        return {"state": "succeeded"}

    def sleep(self, s: float) -> None:
        pass

    # the synthetic exam: per row a base pass rate; a model flips a seed-dependent share of rows (its draw)
    @staticmethod
    def _u(*parts: Any) -> float:
        return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16 ** 12

    def _p(self, model: str, s: str, i: int) -> float:
        u = self._u("row", s, i)
        base = 0.97 if u < 0.80 else 0.5 if u < 0.90 else 0.05
        if self._u("draw", model, s, i) < self.seed_flip:
            base = 1 - base
        return base

    def _rows(self, model: str, run_id: str, arm: str) -> list[dict]:
        rng = random.Random(f"{run_id}|{arm}")
        rows: list[dict] = []
        for k in (1, 2):
            for s, n in self.SETS.items():
                for i in range(n):
                    rows.append({"key": f"{s}:{s[:2]}{i:03d}", "id": f"{s[:2]}{i:03d}", "set": s, "status": "ok",
                                 "pass": int(rng.random() < self._p(model, s, i)), "ghost": 0, "where_ok": 0})
        for i in range(233):
            rows.append({"key": f"F0H:f{i:03d}", "id": f"f{i:03d}", "set": "F0H", "status": "ok", "seen": i < 69,
                         "pass": int(rng.random() < self._p(model, "F0H", i))})
        return rows

    def _fake_eval(self, run_id: str, env: dict) -> None:
        from . import bar as BAR
        noise = (self.store.read_json(f"runs/{run_id}/run.json") or {}).get("noise") or {}
        cand_model = f"{noise.get('data')}@{noise.get('seed')}" if self.honour_seed else f"{noise.get('data')}@{RECIPE_SEED}"
        arms = {"current": self._rows("live", run_id, "current"), "candidate": self._rows(cand_model, run_id, "candidate")}
        res: dict[str, dict[str, list[dict[str, int]]]] = {}
        for a, rows in arms.items():
            self.store.write_jsonl(f"runs/{run_id}/eval/{a}/rows.jsonl", rows)
            res[a] = replicates(rows)
        bar = BAR.evaluate(res, baseline=None, th={"prior_flip_rate": 0.08, "min_sigma_rows": 1.0},
                           ghosts={"current": [0, 0], "candidate": [0, 0]},
                           oos_where={a: [{} for _ in range(2)] for a in arms})
        for r in bar["sets"]:
            r.pop("_keys", None)
        self.store.write_json(f"runs/{run_id}/eval/bar.json", bar)


def seed_fake_bucket(root: Path) -> tuple[Store, str]:
    """A local bucket shaped like int's for `dry` and the tests: infra/config.json, state/model.json, r4's dataset
    in the pool, and a finished base round (run.json, prepare/, model/context.tar.gz)."""
    store = open_store(root)
    base = "rt-20261014-0708-7cac28-1184ff9a"
    store.write_json("infra/config.json", {
        "bucket": "fake-pipeline", "project": "fake-project", "jobs_region": "northamerica-northeast2",
        "jobs": {"eval": "souschef-eval"}, "ollama_model": "sous-chef", "l4_wait_minutes": 1,
        "build": {"region": "us-east4", "service_account": "souschef-builder@fake.iam.gserviceaccount.com",
                  "spec": "infra/cloudbuild-model.json"},
        "model_repo": {"image": "us-east4-docker.pkg.dev/fake-project/sous-chef-model/model"},
        "prod_model": {"name": "sous-chef-model", "region": "us-east4", "url": "https://prod.example"},
        "eval_model": {"name": "sous-chef-model-eval", "region": "us-east4", "url": "https://eval.example"},
        "cand_model": {"name": "sous-chef-model-cand", "region": "us-east4", "url": "https://cand.example"},
        "trainer": {"disk_gb": 500, "timeout_s": 43200, "service_account": "souschef-trainer@fake.iam.gserviceaccount.com",
                    "ladder": [{"name": "h100-flex", "region": "us-central1", "machine_type": "a3-highgpu-1g",
                                "accelerator_type": "NVIDIA_H100_80GB", "strategy": "FLEX_START", "max_wait_s": 14400},
                               {"name": "h100-spot-usc1", "region": "us-central1", "machine_type": "a3-highgpu-1g",
                                "accelerator_type": "NVIDIA_H100_80GB", "strategy": "SPOT"}]}})
    store.write_bytes("infra/cloudbuild-model.json", b'{"steps": []}')
    store.write_json("state/model.json", {"tag": "r4", "digest": "sha256:" + "a" * 64, "training_manifest": "bf78870"})
    for split in ("train", "val"):
        store.write_bytes(f"{R4_DATASET}/dataset/{split}.jsonl", b'{"messages": []}\n')
    store.write_json(f"{R4_DATASET}/dataset/stats.json", {"block_drift_rows": 0, "preview_claim_kept": 0})
    store.write_json(f"runs/{base}/run.json", {"run_id": base, "mode": "full", "execution": "x", "resumes": [],
                                               "deployed": {"hash": "1184ff9a" + "0" * 56},
                                               "trigger": {"drift_run": "dr-20261014-0115-502d48-check", "reason": "drift"},
                                               "images": {"tag": "eac87f9", "trainer": "…/sous-chef-trainer@sha256:" + "b" * 64}})
    store.write_json(f"runs/{base}/prepare/manifest.json", {"fake": True})
    store.write_bytes(f"runs/{base}/prepare/cases.jsonl", b'{"id": "c1"}\n')
    store.write_bytes(f"runs/{base}/model/context.tar.gz", b"fake")
    return store, base


def dry(work: Path, arms_spec: list[str], *, exp: str = "nz-20261015", log: Callable = print, **fake) -> dict:
    """The whole experiment on a local bucket with FakeCloud: setup → train → build → eval → analyze."""
    store, base = seed_fake_bucket(Path(work) / "bucket")
    arms = [parse_arm(s, exp) for s in arms_spec]
    cloud = FakeCloud(store, **fake)
    cfg = config(store)
    setup(store, cloud, arms, base_run=base, log=log)
    tr = train(store, cloud, cfg, arms, rung=None, wait=True, poll_s=0, log=log)
    if any(v != "done" for v in tr.values()):
        log(f"dry: training {tr}")
    build_all(store, cloud, cfg, arms, spec=None, wait=True, poll_s=0, log=log)
    eval_all(store, cloud, cfg, arms, poll_s=0, log=log)
    res = analyze(store, arms, live_data="r4", compare=[], log=log)
    res["_cloud_calls"] = len(cloud.calls)
    return res


# --- CLI ----------------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m souschef_lab.retrain.noise", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser, *, need_exp: bool = True) -> None:
        p.add_argument("--exp", required=need_exp, help="the experiment id, nz-<YYYYMMDD>[-…] (lowercase)")
        p.add_argument("--arm", action="append", default=None, help="DATA@SEED (default r4@2 and r4@3); repeatable")
        p.add_argument("--bucket", default=None, help="gs://… (default $PIPELINE_BUCKET) or a local folder")

    p = sub.add_parser("plan", help="the arms, the commands and the cost; writes nothing")
    common(p, need_exp=False)
    p.add_argument("--base-run", default="rt-20261014-0708-7cac28-1184ff9a")
    p.add_argument("--rung", default=None)
    p.add_argument("--h100-rate", type=float, default=H100_RATE)
    p.add_argument("--h100-hours", type=float, default=H100_HOURS)
    p.add_argument("--offline", action="store_true", help="do not read infra/config.json")
    p = sub.add_parser("setup", help="seed the arms' run folders")
    common(p)
    p.add_argument("--base-run", default="rt-20261014-0708-7cac28-1184ff9a",
                   help="a finished round: its deployed manifest, prepare/ and build context (default r5)")
    p.add_argument("--trainer-image", help="default: the base round's images.trainer (it must honour TRAIN_SEED)")
    p.add_argument("--no-probe", action="store_true", help="no new-key probe in the evals (run.json trigger without a drift run)")
    p.add_argument("--live-data", default="r4", help="the dataset the live model trained on (its group for σ_seed)")
    p = sub.add_parser("train", help="submit / resubmit each arm's Vertex job")
    common(p)
    p.add_argument("--rung", default=None, help="a ladder rung of infra/config.json (default its first, h100-flex)")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--poll-s", type=float, default=300)
    p.add_argument("--max-attempts", type=int, default=4)
    p = sub.add_parser("build", help="the model image per trained arm")
    common(p)
    p.add_argument("--build-config", help="a build spec instead of the bucket's infra/cloudbuild-model.json")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--poll-s", type=float, default=60)
    p = sub.add_parser("eval", help="the paired eval of each built arm against the live model, one at a time")
    common(p)
    p.add_argument("--poll-s", type=float, default=300)
    p.add_argument("--lab-image", help="refuse unless job souschef-eval runs this image (the same for every arm)")
    p = sub.add_parser("analyze", help="σ_seed per bar set and the seed-aware tolerance")
    common(p)
    p.add_argument("--extra", action="append", default=[], help="RUN=DATA@SEED: a finished round as one more draw, "
                   "e.g. rt-20261014-0708-7cac28-1184ff9a=r5@3407 (its own dataset: a group of its own)")
    p.add_argument("--compare", action="append", default=[], help="a round whose bar to re-read with the new tolerance")
    p.add_argument("--live-data", default="r4")
    p.add_argument("--q-hi", type=float, default=0.90, help="the one-sided level of σ_seed's upper bound")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("status")
    common(p)
    p = sub.add_parser("dry", help="the whole experiment with fakes on a local bucket")
    p.add_argument("--workdir", required=True)
    p.add_argument("--arm", action="append", default=None)
    p.add_argument("--exp", default="nz-20261015")
    return ap


def _arms(a) -> list[Arm]:
    exp = check_exp(a.exp)
    return [parse_arm(s, exp) for s in (a.arm or DEFAULT_ARMS)]


def main(argv: list[str] | None = None, *, cloud: Cloud | None = None) -> int:
    import os
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "dry":
            res = dry(Path(a.workdir), a.arm or list(DEFAULT_ARMS), exp=check_exp(a.exp))
            print(table(res))
            return 0
        if a.cmd == "plan":
            exp = check_exp(a.exp or f"nz-{datetime.now(timezone.utc):%Y%m%d}")
            arms = [parse_arm(s, exp) for s in (a.arm or DEFAULT_ARMS)]
            cfg = None if a.offline else config(open_store(a.bucket or os.environ.get("PIPELINE_BUCKET")))
            print(plan_text(arms, a.base_run, cfg, rung=a.rung or "h100-flex",
                            c=costs(len(arms), h100_rate=a.h100_rate, h100_hours=a.h100_hours)))
            return 0
        store = open_store(a.bucket or os.environ.get("PIPELINE_BUCKET"))
        arms = _arms(a)
        if a.cmd == "status":
            print("\n".join(status(store, arms)))
            return 0
        if a.cmd == "analyze":
            extra = []
            for e in a.extra:
                rid, _, spec = e.partition("=")
                lab, _, seed = spec.rpartition("@")
                if not rid or not lab or not seed.isdigit():
                    raise ConfigError(f"--extra {e!r}: RUN=DATA@SEED")
                extra.append((rid, lab, int(seed)))
            res = analyze(store, arms, live_data=a.live_data, extra=extra, compare=a.compare, q_hi=a.q_hi)
            print(json.dumps(res, indent=1) if a.json else table(res))
            return 0
        cfg = config(store)
        cl = cloud or GcloudCloud(cfg["project"])
        if a.cmd == "setup":
            setup(store, cl, arms, base_run=a.base_run, probe=not a.no_probe, trainer_image=a.trainer_image,
                  live_data=a.live_data)
            return 0
        if a.cmd == "train":
            st = train(store, cl, cfg, arms, rung=a.rung, wait=a.wait, poll_s=a.poll_s, max_attempts=a.max_attempts)
            print(json.dumps(st))
            return 0 if all(v in ("done", "running") for v in st.values()) else 1
        if a.cmd == "build":
            st = build_all(store, cl, cfg, arms, spec=Path(a.build_config) if a.build_config else None, wait=a.wait,
                           poll_s=a.poll_s)
            print(json.dumps(st))
            return 0 if all(v != "failed" for v in st.values()) else 1
        if a.cmd == "eval":
            st = eval_all(store, cl, cfg, arms, poll_s=a.poll_s, lab_image=a.lab_image)
            print(json.dumps(st))
            return 0 if all(v != "failed" for v in st.values()) else 1
    except ConfigError as e:
        print(f"noise {a.cmd}: {e}", file=sys.stderr)
        return 2
    except StageError as e:
        print(f"noise {a.cmd}: {e}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
