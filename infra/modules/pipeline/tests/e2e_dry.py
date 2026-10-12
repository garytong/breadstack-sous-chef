#!/usr/bin/env python3
"""e2e_dry — the pipeline's dry end-to-end run: the deployed workflow
definitions, executed by wfsim against fake Google APIs, driving the real lab
jobs, the real sous-chef service and the real thermometer.

    python3 infra/modules/pipeline/tests/e2e_dry.py --work DIR setup
    python3 infra/modules/pipeline/tests/e2e_dry.py --work DIR phase1   # first sight, calibrate, check: ok
    python3 infra/modules/pipeline/tests/e2e_dry.py --work DIR phase2   # new fields → retrain round → Skip
    python3 infra/modules/pipeline/tests/e2e_dry.py --work DIR phase3   # more fields → round batches → Approve
    python3 infra/modules/pipeline/tests/e2e_dry.py --work DIR weekly   # the Monday run on the new model
    python3 infra/modules/pipeline/tests/e2e_dry.py --work DIR phase4   # SMOKE → Approve (dry rollout); a failed stage

Real: every workflow step (workflows/<name>.yaml + lib.yaml as workflows.tf
deploys them), every lab job (python -m souschef_lab.pipeline …, the stage
code, the engine snapshot from the bucket, the harness), the sous-chef
service of this tree (GET /v1/manifests/deployed over launcher
registrations, Postgres), lab/tools/pipeline-train.sh, model-context.sh, and
the thermometer's handlers, MySQL repo and notifier (its
TestSousChefDevServe), posting to a local webhook in place of Slack.

Fake: Cloud Storage (a folder per bucket), Cloud Run (jobs run as local
processes; services and revisions are records), Artifact Registry tags,
Vertex AI (the first rung reports a stockout, the next runs the trainer
script with C's stub trainer), Cloud Build (the context is unpacked and
model-context.sh run; the docker build itself is not), Secret Manager, the
GPU services (the warm-up answers; the exams use the lab's fake models:
DRY), the Anthropic author (FakeAsk), the clock (virtual; sleeps are
instant). Deviations a fake runner adds to job arguments, all listed in
FAKE_ARGS below: --dry for prepare/collect/eval, DRY=1 and --smoke for the
drift exams, --no-caps for a full-mode dataset (the pool is tiny).

Run it with a python3 that has PyYAML (the lab's venv is stdlib-only; the jobs
themselves run on lab/.venv). Needs: Go (builds the service and the thermometer's test binary), node,
Postgres server binaries, the throwaway MySQL of the thermometer's README
(127.0.0.1:13399), the engine bundle and snapshot (--bundle, --snapshot).
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LAB = REPO / "lab"
PY = LAB / ".venv" / "bin" / "python"
WF_DIR = HERE.parent / "workflows"
THERMO = Path(os.environ.get("THERMOMETER_REPO", REPO.parent / "breadstack-thermometer"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LAB))

import wfsim  # noqa: E402

PROJECT = "breadstack-sous-chef-int"
BUCKET = "breadstack-sous-chef-int-pipeline"
SERVICE_TAG = "f79753b-speed-a"
NE2 = "northamerica-northeast2"
GPU = "us-east4"
LAB_REPO = f"{NE2}-docker.pkg.dev/{PROJECT}/sous-chef"
MODEL_REPO = f"{GPU}-docker.pkg.dev/{PROJECT}/sous-chef-model"
PROD_URL = "https://sous-chef-model-x7k2q-uk.a.run.app"
EVAL_URL = "https://sous-chef-model-eval-x7k2q-uk.a.run.app"
CAND_URL = "https://sous-chef-model-cand-x7k2q-uk.a.run.app"
R4_DIGEST = "sha256:" + hashlib.sha256(b"r4 model image").hexdigest()
MENTION = "<@U07AFF50XL0>"
THERM_SECRET = "sous-chef-pipeline-thermometer-key"
MYSQL_DSN = "root:rootpw@tcp(127.0.0.1:13399)/thermtest_souschef_e2e?parseTime=true&charset=utf8mb4&loc=UTC"
LADDER = [  # infra/modules/pipeline/variables.tf train_ladder default (optional max_wait_s filled in, as jsonencode does)
    {"name": "h100-flex", "region": "us-central1", "machine_type": "a3-highgpu-1g", "accelerator_type": "NVIDIA_H100_80GB", "strategy": "FLEX_START", "attempts": 1, "max_wait_s": 14400},
    {"name": "h100-spot-usc1", "region": "us-central1", "machine_type": "a3-highgpu-1g", "accelerator_type": "NVIDIA_H100_80GB", "strategy": "SPOT", "attempts": 4, "max_wait_s": 0},
    {"name": "h100-spot-use4", "region": "us-east4", "machine_type": "a3-highgpu-1g", "accelerator_type": "NVIDIA_H100_80GB", "strategy": "SPOT", "attempts": 4, "max_wait_s": 0},
    {"name": "h100-spot-usw1", "region": "us-west1", "machine_type": "a3-highgpu-1g", "accelerator_type": "NVIDIA_H100_80GB", "strategy": "SPOT", "attempts": 4, "max_wait_s": 0},
    {"name": "a100-80-std", "region": "us-central1", "machine_type": "a2-ultragpu-1g", "accelerator_type": "NVIDIA_A100_80GB", "strategy": "STANDARD", "attempts": 2, "max_wait_s": 0},
    {"name": "a100-40-std", "region": "us-central1", "machine_type": "a2-highgpu-1g", "accelerator_type": "NVIDIA_TESLA_A100", "strategy": "STANDARD", "attempts": 2, "max_wait_s": 0},
]
# what the fake job runner adds (or sets) beyond what the workflow sent
FAKE_ARGS = {"prepare": ["--dry"], "collect": ["--dry"], "eval": ["--dry"], "drift-exam": ["--smoke"], "dataset-full": ["--no-caps"]}


def log(msg: str) -> None:
    print(msg, flush=True)


def now_iso(t: float | None = None) -> str:
    return wfsim.fmt_time(time.time() if t is None else t)


# ═══════════════════════════════════════ the world: state on disk between phases ═══════════════════════════════════════

class World:
    def __init__(self, work: Path):
        self.work = work
        self.gcs = work / "gcs"
        self.bucket = self.gcs / BUCKET
        self.bin = work / "bin"
        self.logs = work / "logs"
        self.state_file = work / "gcp-state.json"
        self.s: dict[str, Any] = json.loads(self.state_file.read_text()) if self.state_file.is_file() else {}

    def save(self) -> None:
        self.state_file.write_text(json.dumps(self.s, indent=1, default=str))

    @property
    def clock_t(self) -> float:
        return float(self.s.get("clock") or time.time())


# ═══════════════════════════════════════ fake Google APIs ═══════════════════════════════════════

class HttpFail(Exception):
    def __init__(self, code: int, body: Any = None):
        self.code, self.body = code, body if body is not None else {"error": {"code": code}}


class FakeGCP:
    """The HTTP router wfsim calls: every Google API the workflows use, the GPU services, int's /version, and the
    thermometer (forwarded to the real one)."""

    def __init__(self, world: World, *, int_url: str, int_key: str, therm_url: str, therm_key: str, snapshot: Path,
                 interp_ref: dict):
        self.w = world
        self.st = world.s
        self.int_url = int_url.rstrip("/")
        self.int_key = int_key
        self.therm_url = therm_url.rstrip("/")
        self.therm_key = therm_key
        self.snapshot = snapshot
        self.interp_ref = interp_ref
        self.events: list[dict] = []
        self.fail_next: dict[str, int] = dict(self.st.get("fail_next") or {})

    # ── entry ──
    def __call__(self, method: str, url: str, query: dict, headers: dict, body: Any, auth: Any) -> tuple[int, Any, dict]:
        try:
            code, out = self.dispatch(method, url, query, headers, body, auth)
        except HttpFail as e:
            code, out = e.code, e.body
        self.events.append({"method": method, "url": url, "code": code})
        return code, out, {"Content-Type": "application/json"}

    def need_oauth(self, auth: Any) -> None:
        if not (isinstance(auth, dict) and auth.get("type") == "OAuth2"):
            raise HttpFail(401, {"error": {"code": 401, "message": "a Google API call without auth: OAuth2"}})

    def dispatch(self, method: str, url: str, query: dict, headers: dict, body: Any, auth: Any) -> tuple[int, Any]:
        u = urllib.parse.urlsplit(url)
        host, path = u.netloc, urllib.parse.unquote(u.path) if False else u.path
        if url.startswith(self.therm_url + "/"):
            return self.thermometer(method, url, query, headers, body)
        if url.startswith(self.int_url + "/"):
            return self.int_service(method, u.path)
        if url.rstrip("/").split("/api/")[0] in (PROD_URL, EVAL_URL, CAND_URL):
            return self.gpu(method, url, body, auth)
        if host == "storage.googleapis.com":
            self.need_oauth(auth)
            return self.storage(method, path, query, body)
        if host == "run.googleapis.com":
            self.need_oauth(auth)
            return self.run(method, path, query, body)
        if host == "artifactregistry.googleapis.com":
            self.need_oauth(auth)
            return self.ar(method, path, query, body)
        if host.endswith("-aiplatform.googleapis.com"):
            self.need_oauth(auth)
            return self.vertex(method, host.split("-aiplatform")[0], path, body)
        if host == "cloudbuild.googleapis.com":
            self.need_oauth(auth)
            return self.cloudbuild(method, path, body)
        if host == "workflowexecutions.googleapis.com":
            self.need_oauth(auth)
            return self.wfexec(method, path, body)
        raise wfsim.SimError(f"no fake for {method} {url}")

    # ── Cloud Storage over a folder ──
    def obj_path(self, bucket: str, name: str) -> Path:
        if bucket != BUCKET:
            raise HttpFail(404, {"error": {"code": 404, "message": f"no bucket {bucket}"}})
        if name.startswith("/") or ".." in name.split("/"):
            raise HttpFail(400)
        return self.w.gcs / bucket / name

    @staticmethod
    def meta(p: Path, name: str) -> dict:
        stt = p.stat()
        return {"kind": "storage#object", "bucket": BUCKET, "name": name, "generation": str(stt.st_mtime_ns),
                "size": str(stt.st_size), "updated": now_iso(stt.st_mtime), "contentType": "application/json"}

    def storage(self, method: str, path: str, query: dict, body: Any) -> tuple[int, Any]:
        m = re.fullmatch(r"/storage/v1/b/([^/]+)/o/(.+)/copyTo/b/([^/]+)/o/(.+)", path)
        if m and method == "POST":
            src = self.obj_path(m.group(1), urllib.parse.unquote(m.group(2)))
            dst = self.obj_path(m.group(3), urllib.parse.unquote(m.group(4)))
            if not src.is_file():
                raise HttpFail(404)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            return 200, self.meta(dst, urllib.parse.unquote(m.group(4)))
        m = re.fullmatch(r"/storage/v1/b/([^/]+)/o/(.+)", path)
        if m:
            name = urllib.parse.unquote(m.group(2))
            p = self.obj_path(m.group(1), name)
            if method == "GET":
                if not p.is_file():
                    raise HttpFail(404, {"error": {"code": 404, "message": f"No such object: {BUCKET}/{name}"}})
                if query.get("alt") == "media":
                    raw = p.read_bytes()
                    try:
                        return 200, json.loads(raw)
                    except ValueError:
                        return 200, raw.decode("utf-8", "replace")
                return 200, self.meta(p, name)
            if method == "DELETE":
                if not p.is_file():
                    raise HttpFail(404)
                g = query.get("ifGenerationMatch")
                if g is not None and str(p.stat().st_mtime_ns) != str(g):
                    raise HttpFail(412)
                p.unlink()
                return 204, None
        m = re.fullmatch(r"/storage/v1/b/([^/]+)/o", path)
        if m and method == "GET":
            prefix = query.get("prefix", "")
            root = self.w.gcs / m.group(1)
            names = sorted(q.relative_to(root).as_posix() for q in root.rglob("*") if q.is_file() and not q.name.startswith(".gcsio"))
            return 200, {"items": [{"name": n} for n in names if n.startswith(prefix)][:1000]}
        m = re.fullmatch(r"/upload/storage/v1/b/([^/]+)/o", path)
        if m and method == "POST":
            if query.get("uploadType") != "media" or not query.get("name"):
                raise HttpFail(400)
            name = query["name"]
            p = self.obj_path(m.group(1), name)
            g = query.get("ifGenerationMatch")
            if g is not None:
                if str(g) == "0" and p.exists():
                    raise HttpFail(412, {"error": {"code": 412, "message": "conditionNotMet"}})
                if str(g) != "0" and (not p.exists() or str(p.stat().st_mtime_ns) != str(g)):
                    raise HttpFail(412, {"error": {"code": 412, "message": "conditionNotMet"}})
            p.parent.mkdir(parents=True, exist_ok=True)
            data = body if isinstance(body, str) else json.dumps(body, indent=1)
            p.write_text(data)
            return 200, self.meta(p, name)
        raise wfsim.SimError(f"storage: no fake for {method} {path}")

    # ── Cloud Run v2 ──
    def run(self, method: str, path: str, query: dict, body: Any) -> tuple[int, Any]:
        p = path[len("/v2/"):]
        jobs, svcs = self.st["jobs"], self.st["services"]
        m = re.fullmatch(r"projects/[^/]+/locations/([^/]+)/jobs/([^/:]+)(:run)?", p)
        if m:
            region, job, run = m.groups()
            if job not in jobs or region != NE2:
                raise HttpFail(404)
            if method == "GET" and not run:
                return 200, copy.deepcopy(jobs[job])
            if method == "PATCH" and not run:
                self._check_same(jobs[job], body, {("template", "template", "containers", 0, "image")}, f"job {job}")
                jobs[job] = body
                return 200, {"name": f"projects/{PROJECT}/locations/{NE2}/operations/{uuid.uuid4().hex[:12]}", "done": True,
                             "response": body}
            if method == "POST" and run:
                return 200, self.run_job(job, body)
        m = re.fullmatch(r"(projects/[^/]+/locations/[^/]+/jobs/([^/]+)/executions/([^/]+))(/tasks)?", p)
        if m and method == "GET":
            ex = self.st["executions"].get(m.group(1))
            if ex is None:
                raise HttpFail(404)
            if m.group(4):
                return 200, {"tasks": [{"name": m.group(1) + "/tasks/0", "lastAttemptResult": {
                    "exitCode": ex["exit_code"], "status": {"code": 0 if ex["exit_code"] == 0 else 2,
                                                            "message": "" if ex["exit_code"] == 0 else f"Task failed with exit code {ex['exit_code']}"}}}]}
            out = {"name": ex["name"], "completionTime": ex["completed"], "conditions": [{"type": "Completed",
                                                                                           "state": "CONDITION_SUCCEEDED" if ex["exit_code"] == 0 else "CONDITION_FAILED"}]}
            out["succeededCount" if ex["exit_code"] == 0 else "failedCount"] = 1
            return 200, out
        m = re.fullmatch(r"projects/[^/]+/locations/([^/]+)/services/([^/:]+)(:testIamPermissions)?", p)
        if m:
            region, svc, tip = m.groups()
            if svc not in svcs or svcs[svc]["region"] != region:
                raise HttpFail(404)
            if tip and method == "POST":
                return 200, {"permissions": [x for x in body.get("permissions", []) if x in ("run.services.update", "run.services.get")]}
            if method == "GET":
                return 200, self._service_view(svc)
            if method == "PATCH":
                cur = self._service_view(svc)
                self._check_same(cur, body, {("template", "containers", 0, "image"), ("template", "annotations")}, f"service {svc}")
                return 200, self.new_revision(svc, body["template"]["containers"][0]["image"],
                                              (body["template"].get("annotations") or {}))
        m = re.fullmatch(r"projects/[^/]+/locations/[^/]+/services/([^/]+)/revisions/([^/]+)", p)
        if m and method == "GET":
            rev = self.st["revisions"].get(p)
            if rev is None:
                raise HttpFail(404)
            return 200, rev
        raise wfsim.SimError(f"run: no fake for {method} {path}")

    def _service_view(self, svc: str) -> dict:
        s = copy.deepcopy(self.st["services"][svc])
        view = {k: v for k, v in s.items() if k != "region"}
        return view

    def _check_same(self, cur: dict, new: dict, allowed: set, what: str) -> None:
        """A PATCH body must be the resource as read with only the allowed fields changed."""
        diffs: list[str] = []

        def walk(a: Any, b: Any, at: tuple) -> None:
            if any(at[:len(x)] == x for x in allowed):
                return
            if isinstance(a, dict) and isinstance(b, dict):
                for k in set(a) | set(b):
                    walk(a.get(k), b.get(k), at + (k,))
            elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
                for i, (x, y) in enumerate(zip(a, b)):
                    walk(x, y, at + (i,))
            elif a != b:
                diffs.append("/".join(map(str, at)))
        walk(cur, new, ())
        if diffs:
            raise wfsim.SimError(f"PATCH {what} changes fields beyond the image: {diffs[:5]}")

    def new_revision(self, svc: str, image: str, annotations: dict) -> dict:
        s = self.st["services"][svc]
        n = int(self.st.setdefault("revno", {}).get(svc, 1)) + 1
        self.st["revno"][svc] = n
        name = f"projects/{PROJECT}/locations/{s['region']}/services/{svc}/revisions/{svc}-{n:05d}-{uuid.uuid4().hex[:3]}"
        self.st["revisions"][name] = {"name": name, "containers": [{"image": image}], "createTime": now_iso(self.interp_now())}
        s["template"]["containers"][0]["image"] = image
        s["template"]["annotations"] = dict(annotations)
        s["latestCreatedRevision"] = s["latestReadyRevision"] = name
        self.st.setdefault("patches", []).append({"service": svc, "image": image, "at": now_iso(self.interp_now()),
                                                  "annotations": annotations})
        return {"name": f"projects/{PROJECT}/locations/{s['region']}/operations/{uuid.uuid4().hex[:12]}", "done": True}

    def interp_now(self) -> float:
        it = self.interp_ref.get("interp")
        return it.clock.now() if it else time.time()

    # ── jobs as local processes ──
    def run_job(self, job: str, body: dict) -> dict:
        ov = (body or {}).get("overrides") or {}
        co = (ov.get("containerOverrides") or [{}])[0]
        args = [str(a) for a in co.get("args") or []]
        env_over = {e["name"]: e["value"] for e in co.get("env") or []}
        tmpl = self.st["jobs"][job]["template"]["template"]
        image = tmpl["containers"][0]["image"]
        want = f"{LAB_REPO}/sous-chef-lab:{SERVICE_TAG}"
        if image != want:
            raise wfsim.SimError(f"job {job} runs {image}, not the lab image {want} (job_set_image missed it)")
        n = len(self.st["executions"]) + 1
        ex_name = f"projects/{PROJECT}/locations/{NE2}/jobs/{job}/executions/{job}-{n:04d}"
        cmd = args[0] if args else "selftest"
        fail = self.fail_next.get(cmd, 0)
        rc, secs = self.exec_lab(job, args, env_over, ex_name, fail_with=1 if fail else None)
        if fail:
            self.fail_next[cmd] = fail - 1
            self.st["fail_next"] = self.fail_next
        self.st["executions"][ex_name] = {"name": ex_name, "job": job, "args": args, "exit_code": rc,
                                          "completed": now_iso(self.interp_now() + secs)}
        self.st.setdefault("job_log", []).append({"execution": ex_name.split("/")[-1], "args": args[:4], "exit": rc,
                                                  "seconds": round(secs, 1)})
        log(f"      job {job} {' '.join(args[:2])} → exit {rc} ({secs:.0f} s)")
        return {"name": f"projects/{PROJECT}/locations/{NE2}/operations/{uuid.uuid4().hex[:12]}",
                "metadata": {"name": ex_name}, "done": False}

    def job_env(self, job: str) -> dict[str, str]:
        """jobs.tf job_env + the job's secret refs, then the fake's local stand-ins."""
        return {"PIPELINE_BUCKET": f"gs://{BUCKET}", "GOOGLE_CLOUD_PROJECT": PROJECT, "SOUS_CHEF_URL": self.int_url,
                "PIPELINE_CONFIG": f"gs://{BUCKET}/infra/config.json", "PROD_MODEL_URL": PROD_URL, "EVAL_MODEL_URL": EVAL_URL,
                "CAND_MODEL_URL": CAND_URL, "PYTHONUNBUFFERED": "1", "SOUS_CHEF_KEY": self.int_key}

    def exec_lab(self, job: str, args: list[str], env_over: dict, ex_name: str, fail_with: int | None) -> tuple[int, float]:
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "TZ", "TMPDIR", "USER")}
        env.update(self.job_env(job))
        env.update(env_over)
        local = str(self.w.bucket)
        # the fake's stand-ins: the bucket is a folder, the GPU services are the lab's fake models (DRY), the
        # engine for the drift exams is the snapshot folder, the service binaries are this tree's
        env = {k: v.replace(f"gs://{BUCKET}", local) for k, v in env.items()}
        args = [a.replace(f"gs://{BUCKET}", local) for a in args]
        exw = self.w.work / "jobs" / ex_name.split("/")[-1]
        exw.mkdir(parents=True, exist_ok=True)
        env.update({"PIPELINE_WORKDIR": str(exw), "SOUS_CHEF_BIN": str(self.w.bin / "sous-chef"),
                    "SOUS_CHEF_FIELDS_BIN": str(self.w.bin / "sous-chef-fields"), "SOUS_CHEF_PREFIX_BIN": str(self.w.bin / "sous-chef-prefix"),
                    "SOUS_CHEF_BUILD_ID_FILE": str(self.w.bin / "BUILD_ID"), "SOUS_CHEF_BUILD_FILE": str(self.w.bin / "BUILD_ID"),
                    "PG_BIN": "/usr/bin", "TMPDIR": str(exw), "THERMOMETER_PUBLIC_URL": "https://thermometer.breadstackcrm.com"})
        cmd = args[0] if args else "selftest"
        extra: list[str] = []
        if cmd in ("prepare", "collect", "eval"):
            extra = FAKE_ARGS[cmd]
        elif cmd == "dataset" and "full" in args:
            extra = FAKE_ARGS["dataset-full"]
        elif cmd == "drift" and len(args) > 1 and args[1] in ("check", "weekly", "calibrate"):
            env["DRY"] = "1"
            env["COOKIE_CUTTER_DIR"] = str(self.snapshot)
            extra = FAKE_ARGS["drift-exam"]
        elif cmd == "drift":
            env["DRY"] = "1"
        # the workflow's virtual clock inside the job (e2e_clock/sitecustomize.py)
        env["E2E_CLOCK_OFFSET"] = str(self.interp_now() - time.time())
        env["PYTHONPATH"] = str(HERE / "e2e_clock")
        argv = [str(PY), "-m", "souschef_lab.pipeline", *args, *extra]
        t0 = time.monotonic()
        logf = self.w.logs / f"{ex_name.split('/')[-1]}.log"
        logf.parent.mkdir(parents=True, exist_ok=True)
        if fail_with is not None:
            logf.write_text(f"(injected failure: exit {fail_with})\n")
            return fail_with, 1.0
        with open(logf, "w") as fh:
            fh.write("$ " + " ".join(argv) + "\n")
            fh.flush()
            r = subprocess.run(argv, cwd=str(LAB), env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=1500)
        if r.returncode == 0 and not os.environ.get("E2E_KEEP_JOB_DIRS"):
            # a Cloud Run job's disk is ephemeral: a finished execution's workdir (an engine
            # snapshot, ~7k files) goes, so /tmp's inodes last the five phases; failures are kept
            shutil.rmtree(exw, ignore_errors=True)
        return r.returncode, time.monotonic() - t0

    # ── Artifact Registry ──
    def ar(self, method: str, path: str, query: dict, body: Any) -> tuple[int, Any]:
        m = re.fullmatch(r"/v1/projects/[^/]+/locations/([^/]+)/repositories/([^/]+)/packages/([^/]+)/tags(?:/([^/]+))?", path)
        if not m:
            raise wfsim.SimError(f"ar: {method} {path}")
        loc, repo, pkg, tag = m.groups()
        pkg, tag = urllib.parse.unquote(pkg), urllib.parse.unquote(tag or "")
        tags = self.st["tags"]
        base = f"projects/{PROJECT}/locations/{loc}/repositories/{repo}/packages/{pkg}"
        if method == "GET" and tag:
            key = f"{loc}/{repo}/{pkg}:{tag}"
            if key not in tags:
                raise HttpFail(404)
            return 200, {"name": f"{base}/tags/{tag}", "version": f"{base}/versions/{tags[key]}"}
        if method == "POST" and not tag:
            tid = query.get("tagId")
            ver = body["version"].split("/versions/")[-1]
            key = f"{loc}/{repo}/{pkg}:{tid}"
            if key in tags:
                raise HttpFail(409)
            tags[key] = ver
            return 200, {"name": f"{base}/tags/{tid}", "version": body["version"]}
        raise wfsim.SimError(f"ar: {method} {path}")

    # ── Vertex AI custom jobs ──
    def vertex(self, method: str, region: str, path: str, body: Any) -> tuple[int, Any]:
        vj = self.st["vertex"]
        m = re.fullmatch(r"/v1/projects/[^/]+/locations/([^/]+)/customJobs", path)
        if m and method == "POST":
            if m.group(1) != region:
                raise HttpFail(400, {"error": {"code": 400, "message": "region mismatch"}})
            pool = body["jobSpec"]["workerPoolSpecs"][0]
            cs = pool["containerSpec"]
            sched = body["jobSpec"].get("scheduling") or {}
            assert pool["machineSpec"]["acceleratorCount"] == 1 and pool["replicaCount"] == 1, pool
            want_img = f"{LAB_REPO}/sous-chef-trainer:{SERVICE_TAG}"
            if cs["imageUri"] != want_img:
                raise wfsim.SimError(f"vertex: trainer image {cs['imageUri']} != {want_img}")
            env = {e["name"]: e["value"] for e in cs.get("env") or []}
            if not str(env.get("PIPELINE_BUCKET", "")).startswith("gs://"):
                raise wfsim.SimError(f"vertex: PIPELINE_BUCKET {env.get('PIPELINE_BUCKET')!r} is not gs://")
            n = len(vj) + 1
            name = f"projects/{PROJECT}/locations/{region}/customJobs/{1000 + n}"
            rec = {"name": name, "displayName": body["displayName"], "state": "JOB_STATE_PENDING", "region": region,
                   "strategy": sched.get("strategy"), "machine": pool["machineSpec"]["machineType"]}
            if sched.get("strategy") == "FLEX_START":
                rec.update(state="JOB_STATE_FAILED", error={"code": 8, "message": "Job failed: the zone does not have enough "
                                                                                   "resources available to fulfill the request (resource pool stockout) within maxWaitDuration."})
            elif self.fail_next.get("vertex-pending", 0) > 0:
                # Spot with no capacity: Vertex queues the job silently (no startTime) until it is cancelled
                self.fail_next["vertex-pending"] -= 1
                self.st["fail_next"] = self.fail_next
                rec["state"] = "JOB_STATE_PENDING"
            elif self.fail_next.get("vertex-preempt", 0) > 0:
                # a preemption: the trainer's SIGTERM trap pushed its checkpoints and wrote status "interrupted";
                # Vertex reports a plain non-zero exit (no "preempt" in the message)
                self.fail_next["vertex-preempt"] -= 1
                self.st["fail_next"] = self.fail_next
                run_id = cs["args"][0]
                prog = self.w.bucket / "runs" / run_id / "train" / "progress.json"
                prog.parent.mkdir(parents=True, exist_ok=True)
                prog.write_text(json.dumps({"status": "interrupted", "step": "SFT", "error": "signal during SFT",
                                            "updated_at": now_iso(self.interp_now()), "last_step": 50}))
                rec.update(state="JOB_STATE_FAILED", error={"code": 3, "message": "The replica workerpool0-0 exited with a non-zero status of 143."})
            else:
                rc, out = self.run_trainer(cs["args"], env)
                rec["state"] = "JOB_STATE_SUCCEEDED" if rc == 0 else "JOB_STATE_FAILED"
                if rc:
                    rec["error"] = {"code": 3, "message": f"The replica workerpool0-0 exited with a non-zero status of {rc}."}
            vj[name] = rec
            log(f"      vertex {rec['displayName']} ({rec['strategy']}) → {rec['state']}")
            return 200, {"name": name, "state": "JOB_STATE_QUEUED"}
        m = re.fullmatch(r"/v1/(projects/[^/]+/locations/[^/]+/customJobs/[^/:]+)(:cancel)?", path)
        if m:
            rec = vj.get(m.group(1))
            if rec is None:
                raise HttpFail(404)
            if m.group(2):
                rec["state"] = "JOB_STATE_CANCELLED"
                return 200, {}
            return 200, {k: v for k, v in rec.items() if k in ("name", "displayName", "state", "error")}
        raise wfsim.SimError(f"vertex: {method} {path}")

    def run_trainer(self, args: list[str], env_in: dict) -> tuple[int, str]:
        """images/trainer/entrypoint.sh minus the GPU self-test: lab/tools/pipeline-train.sh with C's stub trainer."""
        FAKE_GCLOUD, FAKE_TRAINPY = stub_scripts()
        t = self.w.work / "trainer"
        t.mkdir(parents=True, exist_ok=True)
        lab = t / "lab"
        if not lab.exists():
            lab.mkdir()
            for name in ("souschef_lab", "model-image", "base-modelfiles", "Makefile", "tools"):
                (lab / name).symlink_to(LAB / name)
        bindir = t / "bin"
        bindir.mkdir(exist_ok=True)
        g = bindir / "gcloud"
        g.write_text(FAKE_GCLOUD.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
        g.chmod(0o755)
        tp = bindir / "trainpy"
        tp.write_text(FAKE_TRAINPY)
        tp.chmod(0o755)
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG")}
        env.update(env_in)
        run_id, mode = args[0], args[1]
        env.update({"SOUS_CHEF_LAB": str(lab), "TRAINPY": str(tp), "GCLOUD": str(g), "FAKE_GCS": str(self.w.gcs),
                    "REAL_PY": str(PY), "PIPELINE_WORKDIR": str(t / f"w-{run_id}-{uuid.uuid4().hex[:4]}"),
                    "HF_HOME": str(t / "hf"), "STUB_LOG": str(t / "stub.log"), "PYTHONPATH": str(lab)})
        r = subprocess.run(["bash", str(LAB / "tools" / "pipeline-train.sh"), run_id, mode], env=env, cwd=str(t),
                           capture_output=True, text=True, timeout=600)
        (self.w.logs / f"trainer-{run_id}.log").write_text(r.stdout + r.stderr)
        return r.returncode, r.stdout[-2000:]

    # ── Cloud Build ──
    def cloudbuild(self, method: str, path: str, body: Any) -> tuple[int, Any]:
        builds = self.st["builds"]
        m = re.fullmatch(r"/v1/projects/[^/]+/locations/([^/]+)/builds(?:/([^/]+))?", path)
        if not m:
            raise wfsim.SimError(f"cloudbuild: {method} {path}")
        if method == "GET" and m.group(2):
            b = builds.get(m.group(2))
            if b is None:
                raise HttpFail(404)
            return 200, b
        if method != "POST" or m.group(2):
            raise wfsim.SimError(f"cloudbuild: {method} {path}")
        subs = body.get("substitutions") or {}
        for k in ("_BUCKET", "_RUN_ID", "_IMAGE"):
            if not subs.get(k):
                raise wfsim.SimError(f"cloudbuild: substitution {k} missing")
        if not str(body.get("serviceAccount", "")).startswith(f"projects/{PROJECT}/serviceAccounts/"):
            raise wfsim.SimError("cloudbuild: no service account resource name")
        if [s.get("id") for s in body.get("steps") or []] != ["stage", "build"] or body.get("images") != ["$_IMAGE"]:
            raise wfsim.SimError("cloudbuild: the spec is not cloudbuild-model.yaml's")
        src = body["source"]["storageSource"]
        tgz = self.obj_path(src["bucket"], src["object"])
        bid = uuid.uuid4().hex[:16]
        status, detail, digest = "SUCCESS", "", None
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            with tarfile.open(tgz) as tf:
                tf.extractall(ws, filter="data")
            pinm = re.search(r"PINS='([^']*)'", "\n".join(map(str, body["steps"][0].get("args") or [])))
            pinned = [] if not pinm or pinm.group(1).startswith("__MODEL_IMAGE_") else [ln.split("  ", 1) for ln in pinm.group(1).split(";")]
            if not pinned:
                raise wfsim.SimError("cloudbuild: the spec carries no model-image pins (Terraform fills them in)")
            bad = [f for h, f in pinned if not (ws / f).is_file() or hashlib.sha256((ws / f).read_bytes()).hexdigest() != h]
            if not (ws / "model-image" / "Dockerfile").is_file() or not (ws / "model-image" / "model-context.sh").is_file():
                status, detail = "FAILURE", "the source is not a model-image context"
            elif bad:
                status, detail = "FAILURE", f"the model-image context differs from the pinned files: {bad}"
            else:
                stage = ws / "stage"
                stage.mkdir()
                srcstage = self.w.bucket / "runs" / subs["_RUN_ID"] / "train" / "stage"
                for f in srcstage.iterdir():
                    shutil.copyfile(f, stage / f.name)
                r = subprocess.run(["sh", "model-image/model-context.sh", str(stage)], cwd=ws, capture_output=True, text=True)
                (self.w.logs / f"build-{subs['_RUN_ID']}.log").write_text(r.stdout + r.stderr)
                if r.returncode != 0:
                    status, detail = "FAILURE", f"model-context.sh exited {r.returncode}: {r.stderr[-300:]}"
                else:
                    h = hashlib.sha256()
                    for f in sorted(stage.rglob("*")):
                        if f.is_file():
                            h.update(f.name.encode() + f.read_bytes())
                    digest = "sha256:" + h.hexdigest()
                    tag = subs["_IMAGE"].rsplit(":", 1)[1]
                    key = f"{GPU}/sous-chef-model/model:{tag}"
                    if key in self.st["tags"]:
                        status, detail = "FAILURE", f"tag {tag} exists (immutable tags)"
                    else:
                        self.st["tags"][key] = digest
        rec = {"id": bid, "status": status, "statusDetail": detail, "logUrl": f"https://console.cloud.google.com/cloud-build/builds/{bid}"}
        if digest and status == "SUCCESS":
            rec["results"] = {"images": [{"name": subs["_IMAGE"], "digest": digest}]}
        builds[bid] = rec
        log(f"      cloud build {subs['_IMAGE'].rsplit('/', 1)[-1]} → {status} {detail}")
        return 200, {"name": f"projects/{PROJECT}/locations/{GPU}/operations/{bid}", "metadata": {"build": {"id": bid}}}

    # ── Workflow executions (drift → retrain) ──
    def wfexec(self, method: str, path: str, body: Any) -> tuple[int, Any]:
        m = re.fullmatch(r"/v1/projects/[^/]+/locations/([^/]+)/workflows/([^/]+)/executions", path)
        if not m or method != "POST":
            raise wfsim.SimError(f"workflowexecutions: {method} {path}")
        arg = json.loads(body["argument"]) if isinstance(body.get("argument"), str) else None
        if arg is None:
            raise wfsim.SimError("workflowexecutions: argument must be a JSON string")
        name = f"projects/{PROJECT}/locations/{m.group(1)}/workflows/{m.group(2)}/executions/{uuid.uuid4()}"
        self.st.setdefault("started", []).append({"workflow": m.group(2), "argument": arg, "name": name})
        return 200, {"name": name, "state": "ACTIVE"}

    # ── GPU services, int, the thermometer ──
    def gpu(self, method: str, url: str, body: Any, auth: Any) -> tuple[int, Any]:
        base = url.split("/api/")[0]
        if not (isinstance(auth, dict) and auth.get("type") == "OIDC" and auth.get("audience") == base):
            raise HttpFail(403, {"error": "no OIDC token for this service"})
        if method == "POST" and url.endswith("/api/generate"):
            return 200, {"model": (body or {}).get("model"), "done": True}
        raise wfsim.SimError(f"gpu: {method} {url}")

    def int_service(self, method: str, path: str) -> tuple[int, Any]:
        if method == "GET" and path == "/version":
            return 200, {"build_id": SERVICE_TAG, "route_info": {"blueberry-muffin": {"url": PROD_URL, "fields_context": True}}}
        raise wfsim.SimError(f"int: {method} {path}")

    def thermometer(self, method: str, url: str, query: dict, headers: dict, body: Any) -> tuple[int, Any]:
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={k: str(v) for k, v in headers.items()})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw, code = r.read(), r.status
        except urllib.error.HTTPError as e:
            raw, code = e.read(), e.code
        try:
            out = json.loads(raw) if raw else None
        except ValueError:
            out = raw.decode("utf-8", "replace")
        if method == "POST" and isinstance(body, dict):
            self.st.setdefault("posted", []).append({"id": body.get("id"), "kind": body.get("kind"), "code": code,
                                                     "title": body.get("title"),
                                                     "answer": out if code >= 300 else (out or {}).get("slack")})
            if code >= 300:
                log(f"      ! thermometer refused {body.get('id')}: {code} {str(out)[:300]}")
        return code, out


def stub_scripts() -> tuple[str, str]:
    """The stub gcloud and stub trainer of lab/tests/test_pipeline_cli.py (read, not imported: no pytest here)."""
    import ast
    tree = ast.parse((LAB / "tests" / "test_pipeline_cli.py").read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("FAKE_GCLOUD", "FAKE_TRAINPY"):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    return found["FAKE_GCLOUD"], found["FAKE_TRAINPY"]


# ═══════════════════════════════════════ the services this run starts ═══════════════════════════════════════

class Thermometer:
    def __init__(self, world: World):
        self.w = world
        self.dir = world.work / "therm"
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "Thermometer":
        self.dir.mkdir(parents=True, exist_ok=True)
        for f in ("addr", "stop"):
            (self.dir / f).unlink(missing_ok=True)
        env = {**os.environ, "SOUSCHEF_DEVSERVE_DIR": str(self.dir), "THERMOMETER_SOUSCHEF_TEST_DSN": MYSQL_DSN,
               "SOUSCHEF_DEVSERVE_SECONDS": "3000"}
        if self.w.s.get("therm_started"):
            env["SOUSCHEF_DEVSERVE_KEEP"] = "1"
        self.log = open(self.w.logs / "thermometer.log", "a")
        self.proc = subprocess.Popen([str(self.w.bin / "therm.test"), "-test.run", "^TestSousChefDevServe$", "-test.v",
                                      "-test.timeout", "60m"], env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                     cwd=str(THERMO / "server" / "internal" / "api"))
        for _ in range(300):
            if (self.dir / "addr").is_file() and (self.dir / "cookie2").is_file():
                break
            if self.proc.poll() is not None:
                raise RuntimeError(f"the thermometer exited {self.proc.returncode}; see {self.w.logs / 'thermometer.log'}")
            time.sleep(0.1)
        else:
            raise RuntimeError("the thermometer did not start")
        time.sleep(0.2)
        self.url = "http://" + (self.dir / "addr").read_text().strip()
        self.key = (self.dir / "key").read_text().strip()
        self.cookie = (self.dir / "cookie").read_text().strip()
        self.cookie2 = (self.dir / "cookie2").read_text().strip()
        self.w.s["therm_started"] = True
        return self

    def __exit__(self, *exc: Any) -> None:
        (self.dir / "stop").write_text("stop")
        if self.proc:
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()

    def slack(self) -> list[dict]:
        p = self.dir / "slack.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []

    def employee(self, method: str, path: str, body: Any = None, *, cookie: str | None = None) -> tuple[int, Any]:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode() if body is not None else None, method=method,
                                     headers={"Cookie": f"thermometer_session={cookie or self.cookie}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, raw.decode("utf-8", "replace")


def cookie_name() -> str:
    src = (THERMO / "server" / "internal" / "auth").glob("*.go")
    for f in src:
        m = re.search(r'CookieName\s*=\s*"([^"]+)"', f.read_text())
        if m:
            return m.group(1)
    return "thermometer_session"


class IntService:
    """This tree's sous-chef service on a local Postgres (B's harness): GET /v1/manifests/deployed for real."""

    def __init__(self, world: World):
        self.w = world

    def __enter__(self) -> "IntService":
        os.environ["SOUS_CHEF_BIN"] = str(self.w.bin / "sous-chef")
        os.environ["SOUS_CHEF_BUILD_ID_FILE"] = str(self.w.bin / "BUILD_ID")
        os.environ["PG_BIN"] = "/usr/bin"
        from souschef_lab.harness import Harness
        self.h = Harness({"default": "http://127.0.0.1:9"}, run_id="int", workdir=self.w.work / "int-service", auth="none",
                         route_overrides={"prime": "off"}, build_id=SERVICE_TAG)
        self.h.check_version = lambda: None  # no model behind the route: the int service only registers and answers
        self.h.__enter__()
        self.url, self.key = self.h.url, self.h.key
        return self

    def __exit__(self, *exc: Any) -> None:
        self.h.stop()

    def register(self, manifest: dict, launcher_build: str) -> str:
        req = urllib.request.Request(self.url + "/v1/manifests", data=json.dumps(manifest).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                                              "X-Launcher-Build": launcher_build})
        with urllib.request.urlopen(req, timeout=30) as r:
            out = json.loads(r.read())
        return out.get("hash") or out.get("manifestHash") or json.dumps(out)[:80]


# ═══════════════════════════════════════ setup ═══════════════════════════════════════

def manifests() -> dict[str, dict]:
    """M0 = the exam's manifest (r4's training manifest, bf788701); M1 = M0 + two new fields on a panel + a renamed
    field; M2 = M1 + a new list field. M1/M2 name the vendored engine snapshot's commit."""
    from souschef_lab.exam import MANIFEST_FILE
    from souschef_lab.retrain.dry import grown_manifest
    m0 = json.loads(Path(MANIFEST_FILE).read_text())
    m1 = grown_manifest(m0)
    panel = m1["globalPanels"]["ButtonsSection"]["fields"]
    panel["buttonGlowColor"] = {"type": "color", "label": "Button Glow Color", "path": "globalSettings.buttonGlowColor",
                                "help": "A soft glow around every button when hovered."}
    m1["generated_from"] = {"sha": "e7e90106f2b3f43ecc6e7ab56d22ed64bb4187ef", "build_id": "2b3c4d5", "at": "2026-10-11T10:00:00Z"}
    m2 = copy.deepcopy(m1)
    panel2 = m2["globalPanels"]["ButtonsSection"]["fields"]
    panel2["buttonBorderWidth"] = {"type": "text", "label": "Button Border Width", "path": "globalSettings.buttonBorderWidth",
                                   "help": "The width of the border around every button, like 2px."}
    panel2["buttonHoverTextColor"] = {"type": "color", "label": "Button Hover Text Color", "path": "globalSettings.buttonHoverTextColor",
                                      "help": "The text colour of a button under the pointer."}
    m2["generated_from"] = {**m1["generated_from"], "build_id": "3c4d5e6"}
    return {"M0": m0, "M1": m1, "M2": m2}


def setup(w: World, bundle: Path, snapshot: Path) -> None:
    from souschef_lab import enginebundle as EB
    from souschef_lab import gcsio
    from souschef_lab.manifest import Manifest
    from souschef_lab.retrain.dry import generated_pool
    if w.work.exists():
        shutil.rmtree(w.work)
    for d in (w.bucket, w.bin, w.logs):
        d.mkdir(parents=True)
    log("setup: building this tree's service binaries and the thermometer's dev server")
    for name in ("sous-chef", "sous-chef-fields", "sous-chef-prefix"):
        subprocess.run(["go", "build", "-trimpath", "-o", str(w.bin / name), f"./cmd/{name}"], cwd=REPO / "service", check=True)
    (w.bin / "BUILD_ID").write_text(SERVICE_TAG + "\n")
    subprocess.run(["go", "test", "-c", "-o", str(w.bin / "therm.test"), "./server/internal/api/"], cwd=THERMO, check=True)
    # a clean thermometer database per world: its dedup windows run on real time, so a previous world's posts
    # would suppress this one's
    subprocess.run(["docker", "exec", "therm-usage-test", "mysql", "-uroot", "-prootpw", "-e",
                    "DROP DATABASE IF EXISTS thermtest_souschef_e2e; CREATE DATABASE thermtest_souschef_e2e"],
                   check=True, capture_output=True)
    b = gcsio.bucket(str(w.bucket))
    ms = manifests()
    m0 = Manifest(ms["M0"])
    log(f"setup: seeding gs://{BUCKET} (playbook R0): r4 = {m0.hash[:8]}, engine {json.loads((bundle / 'engine.json').read_text())['sha'][:8]}")
    b.write_json(f"manifests/{m0.hash}.json", ms["M0"])
    b.write_json("state/model.json", {"tag": "r4", "digest": R4_DIGEST, "training_manifest": m0.hash,
                                      "trained_at": "2026-10-08T20:00:00Z", "run_id": "r4", "history": []})
    pool = generated_pool(m0, 30, 12)
    b.write_bytes("cases/pool/r4/cases-all.jsonl", "".join(json.dumps(c) + "\n" for c in pool).encode())
    b.write_json("cases/pool/index.json", {"files": [{"path": "cases/pool/r4/cases-all.jsonl", "manifest": m0.hash,
                                                      "round": "r4", "trained_in": "r4"}]})
    EB.publish(b, bundle)
    # the base weights (a stand-in tar) as R0 seeds them
    with tempfile.TemporaryDirectory() as td:
        hub = Path(td) / "hub" / "models--unsloth--Qwen3.5-9B"
        hub.mkdir(parents=True)
        (hub / "config.json").write_text("{}")
        base = w.bucket / "base"
        base.mkdir(parents=True)
        subprocess.run(["tar", "-C", td, "-cf", str(base / "unsloth-qwen3.5-9b-hf.tar"), "hub"], check=True)
        sha = subprocess.run(["sha256sum", "unsloth-qwen3.5-9b-hf.tar"], cwd=base, capture_output=True, text=True, check=True).stdout
        (base / "unsloth-qwen3.5-9b-hf.tar.sha256").write_text(sha)
    # what Terraform writes into the bucket (bucket.tf), the model-image pins filled in
    mi = LAB / "model-image"
    pins = ";".join(f"{hashlib.sha256((mi / f).read_bytes()).hexdigest()}  model-image/{f}"
                    for f in ("Dockerfile", ".dockerignore", "entrypoint.sh", "model-context.sh"))
    b.write_json("infra/cloudbuild-model.json",
                 yaml.safe_load((HERE.parent / "cloudbuild-model.yaml").read_text().replace("__MODEL_IMAGE_SHA256SUMS__", pins)))
    # Cloud Run, Artifact Registry as Terraform and CI leave them
    jobs = {}
    for key, name in (("drift", "souschef-drift"), ("eval", "souschef-eval"), ("prepare", "souschef-prepare"),
                      ("collect", "souschef-collect"), ("dataset", "souschef-dataset")):
        jobs[name] = {"name": f"projects/{PROJECT}/locations/{NE2}/jobs/{name}", "uid": uuid.uuid4().hex, "etag": "e1",
                      "template": {"taskCount": 1, "template": {"maxRetries": 0, "timeout": "3600s",
                                                                "serviceAccount": f"souschef-x@{PROJECT}.iam.gserviceaccount.com",
                                                                "containers": [{"image": "us-docker.pkg.dev/cloudrun/container/job:latest",
                                                                                "args": ["selftest"]}]}}}
    def svc(name: str, region: str, image: str, url: str) -> dict:
        rev = f"projects/{PROJECT}/locations/{region}/services/{name}/revisions/{name}-00001-abc"
        return {"name": f"projects/{PROJECT}/locations/{region}/services/{name}", "region": region, "uri": url, "etag": "e1",
                "template": {"containers": [{"image": image, "resources": {"limits": {"nvidia.com/gpu": "1"}}}],
                             "scaling": {"minInstanceCount": 0, "maxInstanceCount": 1}, "annotations": {}},
                "latestReadyRevision": rev, "latestCreatedRevision": rev}
    services = {"sous-chef": svc("sous-chef", NE2, f"{LAB_REPO}/sous-chef:{SERVICE_TAG}", "https://sous-chef-int.example"),
                "sous-chef-model": svc("sous-chef-model", GPU, f"{MODEL_REPO}/model:r4", PROD_URL),
                "sous-chef-model-eval": svc("sous-chef-model-eval", GPU, f"{MODEL_REPO}/model:r4", EVAL_URL),
                "sous-chef-model-cand": svc("sous-chef-model-cand", GPU, f"{MODEL_REPO}/model:r4", CAND_URL)}
    revisions = {s["latestReadyRevision"]: {"name": s["latestReadyRevision"], "containers": [{"image": s["template"]["containers"][0]["image"]}],
                                             "createTime": "2026-10-09T00:00:00Z"} for s in services.values()}
    tags = {f"{NE2}/sous-chef/sous-chef-lab:{SERVICE_TAG}": "sha256:" + "1" * 64,
            f"{NE2}/sous-chef/sous-chef-trainer:{SERVICE_TAG}": "sha256:" + "2" * 64,
            f"{GPU}/sous-chef-model/model:r4": R4_DIGEST}
    w.s = {"clock": time.time(), "jobs": jobs, "services": services, "revisions": revisions, "tags": tags, "executions": {},
           "vertex": {}, "builds": {}, "started": [], "posted": [], "snapshot": str(snapshot), "phases": {}}
    w.save()
    log(f"setup: done → {w.work}")


def write_config(w: World, int_url: str, therm_url: str, **over: Any) -> dict:
    """infra/modules/pipeline/main.tf local.pipeline_config, as bucket.tf writes it."""
    cfg = {
        "project": PROJECT, "bucket": BUCKET, "workflows_region": "us-central1",
        "workflows": {"drift": "souschef-drift", "retrain": "souschef-retrain"},
        "jobs_region": NE2, "jobs": {"drift": "souschef-drift", "eval": "souschef-eval", "prepare": "souschef-prepare",
                                     "collect": "souschef-collect", "dataset": "souschef-dataset"},
        "int_service": {"name": "sous-chef", "region": NE2, "url": int_url},
        "prod_model": {"name": "sous-chef-model", "region": GPU, "url": PROD_URL},
        "eval_model": {"name": "sous-chef-model-eval", "region": GPU, "url": EVAL_URL},
        "cand_model": {"name": "sous-chef-model-cand", "region": GPU, "url": CAND_URL},
        "lab_repo": {"location": NE2, "repository": "sous-chef", "image": f"{LAB_REPO}/sous-chef-lab", "trainer": f"{LAB_REPO}/sous-chef-trainer"},
        "model_repo": {"location": GPU, "repository": "sous-chef-model", "image": f"{MODEL_REPO}/model"},
        "ollama_model": "sous-chef", "route_name": "blueberry-muffin", "thermometer_url": therm_url,
        "thermometer_secret": THERM_SECRET, "retrain_auto": True, "rollout_enabled": True, "l4_headroom": False,
        "decision_ttl_days": 14, "smoke_ttl_days": 1,
        "build": {"region": GPU, "service_account": f"souschef-builder@{PROJECT}.iam.gserviceaccount.com", "spec": "infra/cloudbuild-model.json"},
        "trainer": {"service_account": f"souschef-trainer@{PROJECT}.iam.gserviceaccount.com", "ladder": LADDER, "disk_gb": 500, "timeout_s": 43200},
        "eval_warm_minutes": 15, "l4_wait_minutes": 360, "lab_image_grace_m": 45, "console_project": PROJECT,
    }
    cfg.update(over)
    (w.bucket / "infra").mkdir(parents=True, exist_ok=True)
    (w.bucket / "infra" / "config.json").write_text(json.dumps(cfg, indent=1))
    return cfg


# ═══════════════════════════════════════ running a workflow ═══════════════════════════════════════

class Run:
    """One phase: the int service, the thermometer, the fake APIs, and workflow executions on one virtual clock."""

    def __init__(self, w: World, *, decide: list[dict] | None = None):
        self.w = w
        self.decide_plan = list(decide or [])
        self.decisions: list[dict] = []
        self.ref: dict[str, Any] = {}
        self.clock = wfsim.VirtualClock(w.clock_t)

    def __enter__(self) -> "Run":
        self.svc = IntService(self.w).__enter__()
        self.therm = Thermometer(self.w).__enter__()
        write_config(self.w, self.svc.url, self.therm.url)
        self.gcp = FakeGCP(self.w, int_url=self.svc.url, int_key=self.svc.key, therm_url=self.therm.url,
                           therm_key=self.therm.key, snapshot=Path(self.w.s["snapshot"]), interp_ref=self.ref)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.w.s["clock"] = self.clock.now()
        self.w.save()
        try:
            self.therm.__exit__()
        finally:
            self.svc.__exit__()

    def secret(self, args: dict) -> str:
        if args.get("secret_id") != THERM_SECRET or args.get("project_id") != PROJECT:
            raise wfsim.WfError({"code": 404, "message": f"secret {args.get('secret_id')} not found", "tags": ["HttpError"]})
        return self.therm.key

    def on_sleep(self, interp: wfsim.Interpreter, seconds: float) -> None:
        """At the decide stage's first wait: an approver decides (after a non-approver is refused)."""
        if not self.decide_plan:
            return
        frames = [e for e in interp.trace[-6:] if e.kind == "sleep"]
        if not frames or not frames[-1].detail.get("where", "").startswith("stage_decide."):
            return
        ready = [p for p in self.w.s.get("posted", []) if p.get("kind") == "candidate_ready" and p.get("code") in (200, 201)]
        if not ready:
            return
        cid = self.ref.get("candidate_id")
        if not cid:
            return
        st, d = self.therm.employee("GET", f"/api/souschef/decisions/{cid}")
        if st != 200 or (d.get("decision") or d).get("state") != "pending":
            return
        plan = self.decide_plan.pop(0)
        # the Slack heads-up must be there, with the mention, before anyone decides
        slack = []
        for _ in range(100):
            slack = [s for s in self.therm.slack() if cid in (s.get("text") or "")]
            if any(MENTION in (s.get("text") or "") for s in slack):
                break
            time.sleep(0.1)
        rec: dict[str, Any] = {"candidate_id": cid, "plan": plan["action"], "slack_with_mention": any(MENTION in (s.get("text") or "") for s in slack),
                               "slack_text": next((s["text"] for s in slack if MENTION in (s.get("text") or "")), None)}
        code2, _ = self.therm.employee("POST", f"/api/souschef/decisions/{cid}/{plan['action']}", {"note": "e2e"}, cookie=self.therm.cookie2)
        rec["non_approver"] = code2
        code, out = self.therm.employee("POST", f"/api/souschef/decisions/{cid}/{plan['action']}", {"note": plan.get("note", "e2e")})
        if code == 422 and plan["action"] == "approve":
            rec["bar_override"] = True
            code, out = self.therm.employee("POST", f"/api/souschef/decisions/{cid}/approve", {"note": "override", "override_bar": True})
        rec["code"] = code
        rec["answer"] = out if code >= 300 else {k: (out.get("decision") or out).get(k) for k in ("state", "decided_by")}
        self.decisions.append(rec)
        self.w.s.setdefault("decisions", []).append(rec)
        log(f"      decision on {cid}: non-approver → {code2}; {self.therm.cookie[:0]}approver {plan['action']} → {code}")

    def workflow(self, name: str, args: dict) -> tuple[Any, wfsim.Interpreter]:
        files = [WF_DIR / f"{name}.yaml", WF_DIR / "lib.yaml"]
        exec_id = str(uuid.uuid4())
        interp = wfsim.Interpreter.from_files(*map(str, files), http=self.gcp, secrets=self.secret, clock=self.clock,
                                              env={"PIPELINE_BUCKET": BUCKET, "PIPELINE_CONFIG": "infra/config.json",
                                                   "GOOGLE_CLOUD_WORKFLOW_EXECUTION_ID": exec_id,
                                                   "GOOGLE_CLOUD_PROJECT_ID": PROJECT, "GOOGLE_CLOUD_LOCATION": "us-central1"},
                                              on_sleep=self.on_sleep, log=lambda s: None)
        self.ref["interp"] = interp
        orig = self.gcp.thermometer

        def watch(method: str, url: str, query: dict, headers: dict, body: Any) -> tuple[int, Any]:
            if isinstance(body, dict) and body.get("kind") == "candidate_ready":
                self.ref["candidate_id"] = body.get("candidate_id")
            return orig(method, url, query, headers, body)
        self.gcp.thermometer = watch  # type: ignore[method-assign]
        log(f"   ▶ {name} {json.dumps(args)}  (virtual {wfsim.fmt_time(self.clock.now())[:16]})")
        t0 = time.monotonic()
        try:
            out = interp.run(args)
            status = "SUCCEEDED"
        except wfsim.WfError as e:
            out, status = e.value, "FAILED"
        finally:
            self.gcp.thermometer = orig  # type: ignore[method-assign]
            self.w.save()
        log(f"   ◀ {name} {status} in {time.monotonic() - t0:.0f} s real, {interp.steps} steps: {json.dumps(out, default=str)[:400]}")
        self.w.s.setdefault("executions_log", []).append({"workflow": name, "args": args, "status": status, "result": out,
                                                         "steps": interp.steps, "virtual_end": wfsim.fmt_time(self.clock.now())})
        return (out, status), interp

    def drift(self, mode: str, **kw: Any) -> tuple[Any, str]:
        (out, status), _ = self.workflow("drift", {"mode": mode, **kw})
        return out, status

    def started_retrains(self) -> list[dict]:
        st = self.w.s.get("started") or []
        todo = [s for s in st if not s.get("ran")]
        for s in todo:
            s["ran"] = True
        return todo


# ═══════════════════════════════════════ checks ═══════════════════════════════════════

class Checks:
    def __init__(self, w: World, phase: str):
        self.w, self.phase = w, phase
        self.rows: list[tuple[bool, str]] = []

    def ok(self, cond: bool, what: str) -> bool:
        self.rows.append((bool(cond), what))
        log(f"   {'✓' if cond else '✗'} {what}")
        return bool(cond)

    def read(self, key: str) -> Any:
        p = self.w.bucket / key
        return json.loads(p.read_text()) if p.is_file() else None

    def done(self) -> int:
        bad = [w for ok, w in self.rows if not ok]
        self.w.s["phases"][self.phase] = {"checks": len(self.rows), "failed": bad}
        self.w.save()
        log(f"== {self.phase}: {len(self.rows) - len(bad)}/{len(self.rows)} checks passed")
        return 1 if bad else 0


def posted_since(w: World, n0: int) -> list[dict]:
    return (w.s.get("posted") or [])[n0:]


# ═══════════════════════════════════════ phases ═══════════════════════════════════════

def phase1(w: World) -> int:
    """First enable: the first poll sees the launcher's manifest (= r4's training manifest), posts the heartbeat;
    the debounced check calibrates the uncalibrated r4 digest first, then judges: ok."""
    from souschef_lab.manifest import Manifest
    c = Checks(w, "phase1")
    ms = manifests()
    with Run(w) as r:
        h0 = r.svc.register(ms["M0"], "1a2b3c4")
        r.svc.register(ms["M0"], "lab-r4")  # a lab registration must never count as deployed
        c.ok(h0.startswith(Manifest(ms["M0"]).hash[:8]), f"int registered M0 {h0[:8]} for launcher 1a2b3c4")
        n0 = len(w.s["posted"])
        out, st = r.drift("poll")
        c.ok(st == "SUCCEEDED", "poll #1 succeeded")
        dep = c.read("state/deployed.json") or {}
        c.ok(dep.get("hash") == Manifest(ms["M0"]).hash and dep.get("launcher_build") == "1a2b3c4",
             "state/deployed.json = M0 from launcher 1a2b3c4 (the lab registration ignored)")
        kinds = [p["kind"] for p in posted_since(w, n0)]
        c.ok("heartbeat" in kinds and kinds.count("heartbeat") == 1, f"one heartbeat posted (posted: {kinds})")
        c.ok("engine_stale" in kinds, "engine_stale posted: M0 names launcher 9cc1cd1, which has no vendored snapshot")
        r.clock.advance(31 * 60)
        n1 = len(w.s["posted"])
        out, st = r.drift("poll")
        c.ok(st == "SUCCEEDED" and isinstance(out, dict) and isinstance(out.get("check"), dict), "poll #2 ran the debounced check")
        chk = (out or {}).get("check") or {}
        c.ok(chk.get("verdict") in ("ok", "notice"), f"check verdict {chk.get('verdict')} on an unchanged manifest")
        base = c.read(f"baselines/{R4_DIGEST}/baseline.json") or {}
        c.ok(base.get("calibrated") is True and len(base.get("replicates") or []) >= 3,
             f"r4's digest calibrated first ({len(base.get('replicates') or [])} replicates)")
        kinds = [p["kind"] for p in posted_since(w, n1)]
        c.ok("heartbeat" not in kinds, "no second heartbeat the same day")
        c.ok(not (w.bucket / "state/locks/gpu-eval.json").exists(), "gpu-eval lock released")
        c.ok(all(p["code"] in (200, 201) for p in w.s["posted"]), "the thermometer accepted every notification")
        c.ok(not r.started_retrains(), "no retrain started")
    return c.done()


def run_retrains(r: Run, c: Checks) -> list[tuple[dict, Any, str]]:
    outs = []
    for s in r.started_retrains():
        c.ok(s["workflow"] == "souschef-retrain" and s["argument"].get("mode") == "full",
             f"the drift workflow started souschef-retrain {json.dumps(s['argument'])[:160]}")
        (out, st), _ = r.workflow("retrain", s["argument"])
        outs.append((s, out, st))
    return outs


def phase2(w: World) -> int:
    """The launcher ships two new fields (+ an engine snapshot exists for its commit): the poll posts the R3 notice;
    the check finds the model dodges them → retrain; the round runs every stage; the approver SKIPS: ack + parked,
    the authored cases stay in the pool."""
    from souschef_lab.manifest import Manifest
    c = Checks(w, "phase2")
    ms = manifests()
    m1 = Manifest(ms["M1"]).hash
    with Run(w, decide=[{"action": "skip", "note": "batch it"}]) as r:
        r.clock.advance(3600)
        r.svc.register(ms["M1"], "2b3c4d5")
        n0 = len(w.s["posted"])
        out, st = r.drift("poll")
        kinds = [p["kind"] for p in posted_since(w, n0)]
        c.ok(st == "SUCCEEDED" and (c.read("state/deployed.json") or {}).get("hash") == m1, "poll saw M1 (launcher 2b3c4d5)")
        c.ok("drift_notice" in kinds, f"the R3 notice was posted ({kinds})")
        r.clock.advance(31 * 60)
        out, st = r.drift("poll")
        chk = (out or {}).get("check") or {}
        c.ok(chk.get("verdict") == "retrain", f"check verdict retrain (got {chk.get('verdict')}; retrain → {chk.get('retrain')})")
        res = c.read(f"drift/{(out or {}).get('run_id', '').replace('-poll', '-check')}/result.json") or {}
        c.ok(res.get("retrain_eligible") is True, f"eligible: calibrated + engine current ({res.get('retrain_blocked_by')})")
        rounds = run_retrains(r, c)
        c.ok(len(rounds) == 1, "one retrain round started")
        if rounds:
            s, out, st = rounds[0]
            rid = (out or {}).get("run_id") if isinstance(out, dict) else None
            c.ok(st == "SUCCEEDED" and isinstance(out, dict) and out.get("outcome") == "skipped", f"round {rid} ended skipped")
            w.s["round_a"] = rid
            check_round(c, w, rid, smoke=False)
            dec = r.decisions[-1] if r.decisions else {}
            c.ok(dec.get("slack_with_mention"), "the candidate_ready Slack post carries <@U07AFF50XL0>")
            c.ok(dec.get("non_approver") == 403, f"a non-approver is refused ({dec.get('non_approver')})")
            c.ok(dec.get("code") == 200, f"the approver's skip was accepted ({dec.get('code')})")
            ack = c.read("state/ack.json") or {}
            c.ok(ack.get("candidate_id") == f"c-{rid}" and ack.get("deployed_hash") == m1 and ack.get("arm_D_mean") is not None,
                 f"state/ack.json from the drift run's ack material ({sorted(ack)[:8]})")
            c.ok(c.read(f"state/parked/c-{rid}.json") is not None, "state/parked/<candidate>.json written")
            idx = c.read("cases/pool/index.json") or {}
            mine = [f for f in idx.get("files") or [] if f.get("round") == rid]
            c.ok(len(mine) == 1 and mine[0].get("trained_in") is None, "the round's authored cases stay in the pool, untrained")
            model = c.read("state/model.json") or {}
            c.ok(model.get("tag") == "r4", "the live model is still r4")
            c.ok(w.s["services"]["sous-chef-model"]["template"]["containers"][0]["image"].endswith("model:r4"), "production untouched")
            c.ok(not (w.bucket / "state/locks/retrain.json").exists(), "retrain lock released")
            kinds = [p["kind"] for p in posted_since(w, n0)]
            c.ok("skipped" in kinds and "retrain_started" in kinds and "candidate_ready" in kinds,
                 f"retrain_started, candidate_ready, skipped posted ({kinds})")
    return c.done()


def check_round(c: Checks, w: World, rid: str | None, *, smoke: bool) -> None:
    if not rid:
        c.ok(False, "no run id")
        return
    pre = f"runs/{rid}"
    run = c.read(f"{pre}/run.json") or {}
    c.ok(run.get("status", "").startswith("done:"), f"run.json status {run.get('status')}")
    stages = sorted((run.get("stages") or {}).keys())
    c.ok(all(s in stages for s in ("prepare", "collect", "dataset", "train", "build", "candidate", "eval", "decide")),
         f"every stage marked done ({stages})")
    c.ok((run.get("deployed") or {}).get("hash") is not None, "run.json pins the deployed manifest")
    for st in ("prepare", "collect", "dataset", "eval"):
        res = c.read(f"{pre}/result-{st}.json") or {}
        c.ok(res.get("status") == "ok", f"result-{st}: {res.get('status')}")
    vj = c.read(f"{pre}/train/vertex.json") or {}
    c.ok(vj.get("rung") == "h100-spot-usc1" and any(a.get("state") == "capacity" for a in vj.get("attempts") or []),
         f"training: Flex-start stockout → {vj.get('rung')} ({[a.get('state') for a in vj.get('attempts') or []]})")
    c.ok((w.bucket / pre / "train" / "DONE").is_file() and any((w.bucket / pre / "train" / "stage").glob("*.gguf")),
         "train/DONE and a staged GGUF")
    img = c.read(f"{pre}/model/image.json") or {}
    c.ok(str(img.get("digest", "")).startswith("sha256:") and str(img.get("image", "")).endswith(":" + ("c-smoke-" if smoke else "c-") + rid),
         f"model image {str(img.get('image')).rsplit('/', 1)[-1]} built")
    cand = w.s["services"]["sous-chef-model-cand"]["template"]["containers"][0]["image"]
    c.ok(cand.endswith("@" + str(img.get("digest"))), "the candidate service runs the candidate digest")
    ev = w.s["services"]["sous-chef-model-eval"]["template"]["containers"][0]["image"]
    c.ok("@sha256:" in ev, f"the eval service runs the production digest ({ev.rsplit('/', 1)[-1][:30]})")
    dec = c.read(f"{pre}/decision.json") or {}
    c.ok(isinstance(dec.get("results"), dict) and "bar_pass" in dec, f"decision.json (bar {dec.get('bar_pass')})")


def phase3(w: World) -> int:
    """The launcher ships more fields: drift beyond the ack → a new round, which BATCHES round A's authored cases;
    the approver APPROVES: production rolls to the candidate, the baseline resets, the ack is gone."""
    from souschef_lab.manifest import Manifest
    c = Checks(w, "phase3")
    ms = manifests()
    m2 = Manifest(ms["M2"]).hash
    ra = w.s.get("round_a")
    with Run(w, decide=[{"action": "approve", "note": "ship it"}]) as r:
        r.clock.advance(2 * 86400)
        r.svc.register(ms["M2"], "3c4d5e6")
        n0 = len(w.s["posted"])
        r.drift("poll")
        r.clock.advance(31 * 60)
        out, st = r.drift("poll")
        chk = (out or {}).get("check") or {}
        c.ok(chk.get("verdict") == "retrain", f"drift beyond the ack: verdict {chk.get('verdict')}")
        rounds = run_retrains(r, c)
        c.ok(len(rounds) == 1, "one retrain round started")
        if rounds:
            s, out, st = rounds[0]
            rid = (out or {}).get("run_id") if isinstance(out, dict) else None
            c.ok(st == "SUCCEEDED" and isinstance(out, dict) and out.get("outcome") == "approved", f"round {rid} ended approved")
            check_round(c, w, rid, smoke=False)
            rep = c.read(f"runs/{rid}/prepare/report.json") or {}
            files = [f.get("path") for f in (rep.get("pool") or {}).get("files") or []]
            c.ok(f"cases/pool/authored/{ra}.jsonl" in files, "round B's pool includes round A's authored cases (the batch)")
            img = c.read(f"runs/{rid}/model/image.json") or {}
            prod = w.s["services"]["sous-chef-model"]
            c.ok(prod["template"]["containers"][0]["image"] == f"{MODEL_REPO}/model@{img.get('digest')}",
                 "sous-chef-model now runs the candidate BY DIGEST (the one the approver saw)")
            c.ok(prod["template"].get("annotations", {}).get("souschef/candidate") == f"c-{rid}", "with the souschef/candidate annotation")
            model = c.read("state/model.json") or {}
            c.ok(model.get("tag") == f"c-{rid}" and model.get("digest") == img.get("digest") and model.get("training_manifest") == m2,
                 f"state/model.json → c-{rid}, trained on M2")
            c.ok((model.get("history") or [{}])[-1].get("tag") == "r4", "r4 is in the model history")
            bl = c.read(f"baselines/{img.get('digest')}/baseline.json") or {}
            c.ok(bl.get("calibrated") is True and bl.get("digest") == img.get("digest"), "the candidate's baseline promoted (drift baseline reset)")
            c.ok(c.read("state/ack.json") is None, "state/ack.json deleted")
            idx = c.read("cases/pool/index.json") or {}
            tr = {f.get("round"): f.get("trained_in") for f in idx.get("files") or []}
            c.ok(tr.get(ra) == f"c-{rid}" and tr.get(rid) == f"c-{rid}", "rounds A and B marked trained_in the new model")
            ro = c.read(f"wf/runs/{rid}/rollout.json") or {}
            c.ok(ro.get("state") == "done" and "model-rollout TAG=r4" in str(ro.get("rollback")), "rollout.json records the rollback command")
            live = [k for k in w.s["tags"] if k.endswith(f"model:live-{rid}")]
            c.ok(bool(live), "the live-<run> tag keeps the image from the cleanup policy")
            kinds = [p["kind"] for p in posted_since(w, n0)]
            c.ok("rolled_out" in kinds, f"rolled_out posted ({kinds})")
    return c.done()


def weekly(w: World) -> int:
    """The Monday run on the new model: arm T on ITS training manifest (the exam rebased onto it), arm D, the probe,
    F0H rebased onto it; an all-clear or a verdict, against the promoted baseline."""
    c = Checks(w, "weekly")
    with Run(w) as r:
        r.clock.advance(3 * 86400)
        n0 = len(w.s["posted"])
        out, st = r.drift("weekly")
        c.ok(st == "SUCCEEDED", f"weekly drift run: {st} {json.dumps(out, default=str)[:300]}")
        c.ok((out or {}).get("verdict") in ("ok", "notice"), f"verdict {(out or {}).get('verdict')} on the model's own manifest")
        model = c.read("state/model.json") or {}
        bl = c.read(f"baselines/{model.get('digest')}/baseline.json") or {}
        c.ok(len(bl.get("replicates") or []) >= 3, f"the weekly arm-T replicate joined the baseline ({len(bl.get('replicates') or [])})")
        kinds = [p["kind"] for p in posted_since(w, n0)]
        c.ok(any(k in ("weekly_all_clear", "drift_notice") for k in kinds), f"the week's message posted ({kinds})")
    return c.done()


def phase4(w: World) -> int:
    """SMOKE (the playbook's first run): every edge, Approve = a dry rollout, drift state untouched; then a round whose
    collect fails twice: stage_failed with the mention, the lock released, artifacts kept."""
    c = Checks(w, "phase4")
    model_before = (w.bucket / "state/model.json").read_text()
    with Run(w, decide=[{"action": "approve", "note": "smoke"}]) as r:
        r.clock.advance(3600)
        n0 = len(w.s["posted"])
        (out, st), _ = r.workflow("retrain", {"mode": "smoke"})
        rid = (out or {}).get("run_id") if isinstance(out, dict) else None
        c.ok(st == "SUCCEEDED" and (out or {}).get("outcome") == "approved", f"SMOKE round {rid} approved")
        check_round(c, w, rid, smoke=True)
        ro = c.read(f"wf/runs/{rid}/rollout.json") or {}
        c.ok(ro.get("dry_run") is True and "run.services.update" in (ro.get("permissions") or []), "dry rollout: permissions read, nothing patched")
        c.ok((w.bucket / "state/model.json").read_text() == model_before, "SMOKE left state/model.json alone")
        c.ok(not any((w.bucket / "cases/pool/authored").glob(f"{rid}*")), "SMOKE authored nothing into the pool")
        kinds = [p["kind"] for p in posted_since(w, n0)]
        c.ok("rolled_out" in kinds, f"rolled_out (dry) posted ({kinds})")
        # a failing stage
        r.gcp.fail_next["collect"] = 3
        n1 = len(w.s["posted"])
        (out, st), _ = r.workflow("retrain", {"mode": "smoke"})
        c.ok(st == "FAILED" and isinstance(out, dict) and out.get("pipeline_stage") == "collect", f"a collect that fails 3× fails the round ({st})")
        fails = [p for p in posted_since(w, n1) if p["kind"] == "stage_failed"]
        c.ok(len(fails) >= 1 and all(p["code"] in (200, 201) for p in fails), f"stage_failed posted ({len(fails)})")
        time.sleep(1.5)
        sl = [s["text"] for s in r.therm.slack() if "failed" in (s.get("text") or "") and "collect" in (s.get("text") or "")]
        c.ok(any(MENTION in t for t in sl), "the stage_failed Slack post carries the mention")
        c.ok(not (w.bucket / "state/locks/retrain.json").exists(), "the retrain lock released after the failure")
    return c.done()


def phase5(w: World) -> int:
    """Resilience (the verifiers' must-change items): a failing poll alerts once per streak (not every 15 min) and
    says when it recovers; a thermometer outage parks messages in the outbox and the next poll delivers them; a
    resume that would re-ask a decided candidate is refused before the lock; Spot that never starts is given up as
    capacity and a preemption (the trainer's trap) is resumed on the same rung; a failed FULL round blocks
    automatic rounds (state/failed-round.json) with the resume command."""
    c = Checks(w, "phase5")
    with Run(w, decide=[{"action": "approve", "note": "smoke"}]) as r:
        # a. a failure streak
        r.clock.advance(3600)
        r.gcp.fail_next["drift"] = 3
        n0 = len(w.s["posted"])
        _, st1 = r.drift("poll")
        f1 = [p for p in posted_since(w, n0) if p["kind"] == "stage_failed"]
        c.ok(st1 == "FAILED" and not f1, f"a first failed poll alerts nobody ({st1}, {len(f1)} posted)")
        r.clock.advance(900)
        _, st2 = r.drift("poll")
        f2 = [p for p in posted_since(w, n0) if p["kind"] == "stage_failed"]
        c.ok(st2 == "FAILED" and len(f2) == 1, f"the second failure in a row posts one stage_failed ({len(f2)})")
        r.clock.advance(900)
        _, st3 = r.drift("poll")
        f3 = [p for p in posted_since(w, n0) if p["kind"] == "stage_failed"]
        c.ok(st3 == "FAILED" and len(f3) == 1, f"the third is quiet: at most one alert a day per streak ({len(f3)})")
        streak = c.read("state/drift-failing.json") or {}
        c.ok(streak.get("count") == 3 and streak.get("alerted_at"), f"state/drift-failing.json counts the streak ({streak.get('count')})")
        r.clock.advance(900)
        _, st4 = r.drift("poll")
        rec = [p for p in posted_since(w, n0) if p["kind"] == "drift_notice" and str(p.get("id", "")).endswith("/recovered")]
        c.ok(st4 == "SUCCEEDED" and len(rec) == 1 and c.read("state/drift-failing.json") is None,
             "the next good poll ends the streak and says so (no mention)")
        sl: list[str] = []
        for _ in range(450):  # the worker's notifier posts on its own pass (bursts, up to ~30 s late here)
            sl = [s["text"] for s in r.therm.slack() if "failed: drift-poll" in (s.get("text") or "")]
            if sl:
                break
            time.sleep(0.2)
        time.sleep(1.0)
        sl = [s["text"] for s in r.therm.slack() if "failed: drift-poll" in (s.get("text") or "")]
        c.ok(len(sl) == 1 and MENTION in sl[0], f"one Slack alert for the streak, with the mention ({len(sl)})")

        # b. the thermometer down: the outbox
        orig = r.gcp.thermometer

        def down(method: str, url: str, query: dict, headers: dict, body: Any) -> tuple[int, Any]:
            if method == "POST":
                raise HttpFail(503, {"error": "thermometer down"})
            return orig(method, url, query, headers, body)
        r.gcp.thermometer = down  # type: ignore[method-assign]
        r.clock.advance(86400)  # a new UTC day: the heartbeat is due
        _, st5 = r.drift("poll")
        parked = sorted(p.name for p in (w.bucket / "wf" / "outbox").glob("*.json")) if (w.bucket / "wf" / "outbox").is_dir() else []
        c.ok(st5 == "SUCCEEDED" and any("heartbeat" in n for n in parked), f"the undelivered heartbeat waits in wf/outbox/ ({parked})")
        r.gcp.thermometer = orig  # type: ignore[method-assign]
        n1 = len(w.s["posted"])
        r.clock.advance(900)
        _, st6 = r.drift("poll")
        left = list((w.bucket / "wf" / "outbox").glob("*.json")) if (w.bucket / "wf" / "outbox").is_dir() else []
        hb = [p for p in posted_since(w, n1) if p["kind"] == "heartbeat" and p["code"] in (200, 201)]
        c.ok(st6 == "SUCCEEDED" and not left and hb, f"the next poll delivers the outbox ({len(left)} left, {len(hb)} heartbeat)")

        # c. a resume that would re-ask a decided candidate
        decided = next((d for d in w.s.get("decisions") or [] if d.get("plan") == "approve" and not str(d.get("candidate_id", "")).startswith("c-smoke-")), None)
        rid = str((decided or {}).get("candidate_id", ""))[2:]
        prod_before = json.dumps(w.s["services"]["sous-chef-model"]["template"])
        (out, st), _ = r.workflow("retrain", {"resume_from": "eval", "run_id": rid})
        c.ok(st == "FAILED" and isinstance(out, dict) and out.get("error_class") == "candidate_decided",
             f"resume_from eval of the approved {rid} is refused ({st} {(out or {}).get('error_class') if isinstance(out, dict) else out})")
        c.ok(json.dumps(w.s["services"]["sous-chef-model"]["template"]) == prod_before and c.read("state/failed-round.json") is None,
             "production untouched, no failed-round left by a refused resume")

        # d. Spot that never starts, then a preemption the trainer's trap reports
        r.gcp.fail_next["vertex-pending"] = 1
        r.gcp.fail_next["vertex-preempt"] = 1
        (out, st), _ = r.workflow("retrain", {"mode": "smoke"})
        srid = (out or {}).get("run_id") if isinstance(out, dict) else None
        vj = c.read(f"runs/{srid}/train/vertex.json") or {}
        att = [(a.get("rung"), a.get("state")) for a in vj.get("attempts") or []]
        c.ok(st == "SUCCEEDED" and ("h100-spot-usc1", "capacity") in att and ("h100-spot-use4", "preempted") in att
             and vj.get("rung") == "h100-spot-use4" and vj.get("preemptions") == 1,
             f"never-started Spot → capacity, the next rung; a preemption resumed there ({att})")

        # e. a FULL round that fails blocks automatic rounds
        r.gcp.fail_next["prepare"] = 2
        (out, st), _ = r.workflow("retrain", {"mode": "full", "trigger": {"drift_run": None, "reason": "e2e"}})
        fr = c.read("state/failed-round.json") or {}
        c.ok(st == "FAILED" and fr.get("stage") == "prepare" and "resume_from" in str(fr.get("resume")),
             f"a failed full round leaves state/failed-round.json with its resume command ({fr.get('stage')})")
        c.ok(not (w.bucket / "state/locks/retrain.json").exists(), "the retrain lock released")
        (w.bucket / "state/failed-round.json").unlink(missing_ok=True)
    return c.done()


def report(w: World) -> int:
    s = w.s
    log(json.dumps({"phases": s.get("phases"), "jobs": len(s.get("job_log") or []), "vertex": len(s.get("vertex") or {}),
                    "builds": len(s.get("builds") or {}), "patches": s.get("patches"),
                    "notifications": [(p["kind"], p["code"], p.get("answer")) for p in s.get("posted") or []]}, indent=1, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--bundle", default=os.environ.get("ENGINE_BUNDLE_DIR"), help="an engine-vendor.sh bundle folder")
    ap.add_argument("--snapshot", default=os.environ.get("COOKIE_CUTTER_DIR"), help="the engine snapshot folder (drift DRY)")
    ap.add_argument("what", choices=("setup", "phase1", "phase2", "phase3", "weekly", "phase4", "phase5", "report"))
    a = ap.parse_args()
    w = World(Path(a.work).resolve())
    if a.what == "setup":
        if not a.bundle or not a.snapshot:
            ap.error("setup needs --bundle and --snapshot")
        setup(w, Path(a.bundle), Path(a.snapshot).resolve())
        return 0
    return {"phase1": phase1, "phase2": phase2, "phase3": phase3, "weekly": weekly, "phase4": phase4, "phase5": phase5,
            "report": report}[a.what](w)


if __name__ == "__main__":
    sys.exit(main())
