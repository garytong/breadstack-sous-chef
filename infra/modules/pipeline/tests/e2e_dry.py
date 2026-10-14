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
    python3 infra/modules/pipeline/tests/e2e_dry.py --work DIR phase5   # resilience: failure streaks, outbox, resume, Spot
    python3 infra/modules/pipeline/tests/e2e_dry.py --work DIR phase6   # the real APIs' hard cases (audit 2026-10-12)

Real: every workflow step (workflows/<name>.yaml + lib.yaml as workflows.tf
deploys them), every lab job (python -m souschef_lab.pipeline …, the stage
code, the engine snapshot from the bucket, the harness), the sous-chef
service of this tree (GET /v1/manifests/deployed over launcher
registrations, Postgres), lab/tools/pipeline-train.sh, model-context.sh, and
the thermometer's handlers, MySQL repo and notifier (its
TestSousChefDevServe), posting to a local webhook in place of Slack.

Fake: Cloud Storage (a folder per bucket), Cloud Run (jobs run as local
processes; services and revisions are records), Artifact Registry tags,
Vertex AI (Flex-start never starts, Spot runs the trainer script with C's
stub trainer), Cloud Build (the context is unpacked and model-context.sh
run; the docker build itself is not), Secret Manager, the
GPU services (the warm-up answers; the exams use the lab's fake models:
DRY), the Anthropic author (FakeAsk), the clock (virtual; sleeps are
instant). Deviations a fake runner adds to job arguments, all listed in
FAKE_ARGS below: --dry for prepare/collect/eval, DRY=1 and --smoke for the
drift exams, --no-caps for a full-mode dataset (the pool is tiny).

The fakes answer as the real APIs do (audit 2026-10-12, docs URLs at the
constants and methods): a Cloud Run PATCH returns a not-done operation the
workflow account may not GET (403) while the job/service reconciles on the
virtual clock (generation / observedGeneration / reconciling /
terminalCondition, revisions by digest, traffic + trafficStatuses, an
identical PATCH is a no-op without forceNewRevision, a revision can fail,
a stale etag is a 409); executions have no completionTime until they end
and omit a zero exitCode; Artifact Registry versions answer without
relatedTags unless view=FULL, carry `latest`, and exist untagged; Cloud
Build queues, works and ends later, fails a docker client or worker that
cannot build the step, pushes at the end into immutable tags; Vertex jobs
queue, pend, run and end on the clock (Flex-start fails without startTime
and with a message no regex knows, Spot preemptions restart inside the job,
cancel passes through CANCELLING); Cloud Storage answers media as bytes
with its Content-Type and X-Goog-Generation, a missing object as the
text "No such object: …", pages listings, and can race a reader
(race()); one-off errors can be injected (inject()). wfsim enforces
Workflows' 100,000-step, 512 KB variable, 256 KB string and 20-level
limits.

Run it with a python3 that has PyYAML (the lab's venv is stdlib-only; the jobs
themselves run on lab/.venv). Needs: Go (builds the service and the thermometer's test binary), node,
Postgres server binaries, the throwaway MySQL of the thermometer's README
(127.0.0.1:13399), the engine bundle and snapshot (--bundle, --snapshot).
"""
from __future__ import annotations

import argparse
import base64
import copy
import datetime
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
LABONLY_TAG = "ci-7a1e2f0"          # a sous-chef tag with a lab image but no trainer image
PREVIEW_TAG = "p-9d8c7b6"           # an int preview deploy (--no-traffic --tag), phase6
UNTAGGED_DIGEST = "sha256:" + "5" * 64  # a sous-chef version whose tags moved on (200, no relatedTags)


def revision_image(image: str, tags: dict) -> str:
    """The image reference a real Cloud Run v2 revision records: by digest
    (<host>/<project>/<repo>/<package>@sha256:…), never by tag (the first real
    poll, 2026-10-13, failed on it). A tag the fake registry does not know yet
    gets a stable digest, as a push would have given it."""
    if "@" in image:
        return image
    ref, _, tag = image.rpartition(":")
    host, project, repo, pkg = ref.split("/", 3)
    key = f"{host.split('-docker.pkg.dev')[0]}/{repo}/{pkg}:{tag}"
    if key not in tags:
        tags[key] = "sha256:" + hashlib.sha256(image.encode()).hexdigest()
    return f"{ref}@{tags[key]}"


# ── what the real APIs do, as the fakes below model it (audit 2026-10-12, scratchpad/audit/PLAN.md) ──
# Cloud Run v2: a PATCH answers a NOT done operation (live, cloudrun audit §1); the workflow account cannot GET
# projects/*/locations/*/operations/* (run.operations.get is only in roles grantable on the job/service, and an
# operation is not under them: https://docs.cloud.google.com/run/docs/reference/iam/roles), so that GET is 403
# unless iam.tf grants run.operations.get at project level. The resource itself shows the rollout: generation
# (a string, int64 in JSON), observedGeneration, reconciling, terminalCondition, latestCreated/ReadyRevision,
# traffic / trafficStatuses (https://docs.cloud.google.com/run/docs/reference/rest/v2/projects.locations.services).
RUN_OUTPUT_ONLY = {"uid", "generation", "observedGeneration", "reconciling", "terminalCondition", "conditions",
                   "latestReadyRevision", "latestCreatedRevision", "trafficStatuses", "uri", "urls", "createTime",
                   "updateTime", "deleteTime", "expireTime", "creator", "lastModifier", "etag", "satisfiesPzs",
                   "latestCreatedExecution", "executionCount", "observedGeneration"}
LATEST_100 = [{"type": "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST", "percent": 100}]
RECONCILE_S = {"job": 15, "cpu": 25, "gpu": 150, "gpu-failed": 420}  # virtual seconds until a PATCH settles
GPU_REVISION_FAILED = ("Revision '{rev}' is not ready and cannot serve traffic. The revision could not get an NVIDIA "
                       "L4 GPU in {region} (e2e fake text: Cloud Run's exact wording for a GPU stockout is not documented).")
# Cloud Build: options.workerRelease picks the worker (https://docs.cloud.google.com/build/docs/release-channels):
# unset = `legacy` (Docker Engine 20.10) until 2027-03-28, then `regular`; 2026.09 = Engine 29, which needs a
# client speaking API >= 1.44, i.e. Docker 25+ (https://docs.docker.com/engine/release-notes/29/).
WORKER_ENGINE = {"legacy": 20, "2026.09": 29, "regular": 29, "stable": 29, "rapid": 29}
# gcr.io/cloud-builders/docker digests → Docker major version (gcloud container images list-tags, read 2026-10-12)
DOCKER_BUILDER_DIGESTS = {"6c9d05f41f8e": 29, "f772ab459889": 20}
# Vertex AI custom jobs: Flex-start running out of maxWaitDuration has no documented message; the fake's is
# deliberately one no keyword regex would match (audit VX-5). A Spot preemption restarts the job inside Vertex up to
# six times, then fails it with a STOCKOUT (https://docs.cloud.google.com/gemini-enterprise-agent-platform/
# machine-learning/training/use-spot-vms); this is the stockout text Vertex prints.
VERTEX_FLEX_GAVE_UP = "Job did not run: the requested worker pool was never provisioned for this job."
VERTEX_STOCKOUT = ("Resources are insufficient in region: {region}. Please try a different region. If you use K80, "
                   "please consider using P100 or V100 instead.")
VERTEX_MAX_RESTARTS = 6
VERTEX_TERMINAL = ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED")


def legacy_default(t: float) -> bool:
    """Whether a build without options.workerRelease still gets the legacy worker at virtual time t."""
    return t < 1806192000.0  # 2027-03-28T00:00:00Z


def docker_major(image: str) -> int | None:
    """The Docker CLI major version of a build step's image, when the reference says (None: unknown)."""
    name, _, digest = image.partition("@")
    if digest:
        v = DOCKER_BUILDER_DIGESTS.get(digest.split(":", 1)[-1][:12])
        if v is not None:
            return v
    last = name.rsplit("/", 1)[-1]
    repo, _, tag = last.partition(":")
    if repo != "docker":
        return None
    if not tag:
        return None if digest else 20  # gcr.io/cloud-builders/docker(:latest) is 20.10.24
    if tag == "latest":
        return 20 if "cloud-builders" in name else None
    m = re.match(r"(\d+)", tag)
    return int(m.group(1)) if m else None


def short_rev(name: str) -> str:
    return name.rsplit("/", 1)[-1]


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

GOOGLE_STATUS = {400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED", 404: "NOT_FOUND",
                 408: "DEADLINE_EXCEEDED", 409: "ABORTED", 412: "FAILED_PRECONDITION", 429: "RESOURCE_EXHAUSTED",
                 500: "INTERNAL", 502: "UNAVAILABLE", 503: "UNAVAILABLE", 504: "DEADLINE_EXCEEDED"}
JSON_CT = "application/json; charset=UTF-8"


def google_error(code: int, message: str | None = None, status: str | None = None) -> dict:
    """A Google API error body (application/json): {"error": {code, message, status}}."""
    return {"error": {"code": code, "message": message or f"HTTP {code}", "status": status or GOOGLE_STATUS.get(code, "UNKNOWN")}}


class HttpFail(Exception):
    def __init__(self, code: int, body: Any = None, headers: dict | None = None):
        self.code = code
        self.body = body if body is not None else google_error(code)
        self.headers = headers or {"Content-Type": JSON_CT}


def iam_grants_run_operations() -> bool:
    """Whether iam.tf gives the workflow account run.operations.get at project level (a custom role holding it,
    bound with google_project_iam_member): only then may it GET a Cloud Run operation (audit finding 1)."""
    tf = HERE.parent / "iam.tf"
    t = tf.read_text() if tf.is_file() else ""
    return bool(re.search(r'google_project_iam_custom_role[\s\S]{0,800}?"run\.operations\.get"', t)) \
        and "google_project_iam_member" in t


class FakeGCP:
    """The HTTP router wfsim calls: every Google API the workflows use, the GPU services, int's /version, and the
    thermometer (forwarded to the real one). Time-dependent state (a Cloud Run rollout, a build, a Vertex job)
    moves on the run's virtual clock: tick() settles whatever is due before each request."""

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
        self.injections: list[dict] = []          # one-off HTTP errors (inject())
        self.races: dict[str, Any] = {}           # object name -> what another writer does right after our read
        self.raw_error_bodies = False             # thermometer error bodies handed over as the raw JSON text
        self.ops_allowed = iam_grants_run_operations()
        for k in ("pending", "ops", "gcs_ct", "image_violations", "vertex_overlap"):
            self.st.setdefault(k, {} if k in ("ops", "gcs_ct") else [])

    # ── entry ──
    def __call__(self, method: str, url: str, query: dict, headers: dict, body: Any, auth: Any) -> tuple[int, Any, dict]:
        u = urllib.parse.urlsplit(url)
        if u.query:  # a query string in the URL is the same request as query: {...}
            query = {**{k: v[-1] for k, v in urllib.parse.parse_qs(u.query, keep_blank_values=True).items()}, **query}
            url = urllib.parse.urlunsplit(u._replace(query=""))
        self.tick()
        inj = self.injected(method, url)
        hdrs: dict = {}
        try:
            if inj is not None and not inj.get("after"):
                raise HttpFail(inj["code"], inj.get("body"))
            res = self.dispatch(method, url, query, headers, body, auth)
            code, out = res[0], res[1]
            hdrs = res[2] if len(res) > 2 else {}
            if inj is not None:  # applied, but the answer is lost: the caller sees the error
                raise HttpFail(inj["code"], inj.get("body"))
        except HttpFail as e:
            code, out, hdrs = e.code, e.body, e.headers
        self.events.append({"method": method, "url": url, "code": code, "injected": inj is not None})
        return code, out, hdrs or {"Content-Type": JSON_CT}

    def inject(self, method: str, url_pattern: str, code: int, *, count: int = 1, after: bool = False,
               body: Any = None) -> None:
        """The next `count` requests matching (method, url regex) fail with `code`; after=True applies the request
        first (a lost response: the retry meets what the first attempt did)."""
        self.injections.append({"method": method, "re": re.compile(url_pattern), "code": code, "count": count,
                                "after": after, "body": body if body is not None else google_error(code), "hits": 0})

    def injected(self, method: str, url: str) -> dict | None:
        for inj in self.injections:
            if inj["count"] > 0 and inj["method"] in (method, "*") and inj["re"].search(url):
                inj["count"] -= 1
                inj["hits"] += 1
                return inj
        return None

    def consume(self, key: str) -> bool:
        if self.fail_next.get(key, 0) > 0:
            self.fail_next[key] -= 1
            self.st["fail_next"] = self.fail_next
            return True
        return False

    def interp_now(self) -> float:
        c = self.interp_ref.get("clock")
        if c is not None:
            return c.now()
        it = self.interp_ref.get("interp")
        return it.clock.now() if it else time.time()

    def schedule(self, ev: dict) -> None:
        self.st["pending"].append(ev)

    def tick(self) -> None:
        """Settle every rollout, build and Vertex job whose time has come."""
        now = self.interp_now()
        due = sorted((e for e in self.st["pending"] if e["at"] <= now), key=lambda e: e["at"])
        if due:
            self.st["pending"] = [e for e in self.st["pending"] if e["at"] > now]
            for ev in due:
                if ev["kind"] == "job":
                    self.settle_job(ev)
                elif ev["kind"] == "svc":
                    self.settle_service(ev)
        self.tick_builds(now)
        self.tick_vertex(now)

    def need_oauth(self, auth: Any) -> None:
        if not (isinstance(auth, dict) and auth.get("type") == "OAuth2"):
            raise HttpFail(401, google_error(401, "a Google API call without auth: OAuth2"))

    def dispatch(self, method: str, url: str, query: dict, headers: dict, body: Any, auth: Any) -> tuple:
        u = urllib.parse.urlsplit(url)
        host, path = u.netloc, u.path
        if url.startswith(self.therm_url + "/"):
            return self.thermometer(method, url, query, headers, body)
        if url.startswith(self.int_url + "/"):
            return self.int_service(method, u.path)
        if url.rstrip("/").split("/api/")[0] in (PROD_URL, EVAL_URL, CAND_URL):
            return self.gpu(method, url, body, auth)
        if host == "storage.googleapis.com":
            self.need_oauth(auth)
            return self.storage(method, path, query, headers, body)
        if host == "run.googleapis.com":
            self.need_oauth(auth)
            return self.run(method, path, query, body)
        if host == "artifactregistry.googleapis.com":
            self.need_oauth(auth)
            return self.ar(method, path, query, body)
        if host.endswith("-aiplatform.googleapis.com"):
            self.need_oauth(auth)
            return self.vertex(method, host.split("-aiplatform")[0], path, body, query)
        if host == "cloudbuild.googleapis.com":
            self.need_oauth(auth)
            return self.cloudbuild(method, path, body)
        if host == "workflowexecutions.googleapis.com":
            self.need_oauth(auth)
            return self.wfexec(method, path, body)
        raise wfsim.SimError(f"no fake for {method} {url}")

    # ── Cloud Storage over a folder ──
    # Real (live GETs 2026-10-12, storage audit): alt=media answers the bytes with the object's Content-Type and
    # X-Goog-Generation; a missing object's media 404 is the text "No such object: <b>/<o>" (text/html), the JSON
    # API's 404 a JSON error; generation/size are strings; an unencoded "/" in the object segment is a 404; a list
    # pages with nextPageToken (maxResults <= 1000) and an empty one is {}; gcloud-written (lab) objects are
    # application/octet-stream, so their media body is bytes in Workflows.
    def obj_path(self, bucket: str, name: str) -> Path:
        if bucket != BUCKET:
            raise HttpFail(404, google_error(404, f"The specified bucket does not exist: {bucket}"))
        if name.startswith("/") or ".." in name.split("/"):
            raise HttpFail(400)
        return self.w.gcs / bucket / name

    def content_type(self, name: str) -> str:
        return self.st["gcs_ct"].get(name, "application/octet-stream")

    def meta(self, p: Path, name: str) -> dict:
        stt = p.stat()
        return {"kind": "storage#object", "bucket": BUCKET, "name": name, "generation": str(stt.st_mtime_ns),
                "metageneration": "1", "size": str(stt.st_size), "updated": now_iso(stt.st_mtime),
                "contentType": self.content_type(name)}

    @staticmethod
    def missing_media(name: str) -> HttpFail:
        return HttpFail(404, f"No such object: {BUCKET}/{name}", {"Content-Type": "text/html; charset=UTF-8"})

    @staticmethod
    def missing_meta(name: str) -> HttpFail:
        msg = f"No such object: {BUCKET}/{name}"
        return HttpFail(404, {"error": {"code": 404, "message": msg, "errors": [{"message": msg, "domain": "global",
                                                                              "reason": "notFound"}]}})

    @staticmethod
    def precondition_failed() -> HttpFail:
        msg = "At least one of the pre-conditions you specified did not hold."
        return HttpFail(412, {"error": {"code": 412, "message": msg, "errors": [{"message": msg, "domain": "global",
                                                                              "reason": "conditionNotMet",
                                                                              "locationType": "header",
                                                                              "location": "If-Match"}]}})

    def race(self, name: str, mutate: Any, *, on: str = "media") -> None:
        """Right after our next read of `name` (on: "media" = the content, "meta" = the metadata, "any"), another
        writer rewrites it (a new generation): mutate(dict) -> dict. Models a lock refreshed by its holder between a
        contender's read and its conditional delete (audit ST-2)."""
        self.races[name] = (mutate, on)

    def after_read(self, name: str, p: Path, kind: str) -> None:
        r = self.races.get(name)
        if r is None or r[1] not in (kind, "any") or not p.is_file():
            return
        mutate = self.races.pop(name)[0]
        old = p.stat().st_mtime_ns
        try:
            data = json.loads(p.read_bytes())
        except ValueError:
            data = {}
        p.write_text(json.dumps(mutate(data), indent=1))
        if p.stat().st_mtime_ns == old:
            os.utime(p, ns=(old + 1000, old + 1000))
        self.st.setdefault("races", []).append({"object": name, "at": now_iso(self.interp_now())})

    def storage(self, method: str, path: str, query: dict, headers: dict, body: Any) -> tuple:
        m = re.fullmatch(r"/storage/v1/b/([^/]+)/o/([^/]+)/copyTo/b/([^/]+)/o/([^/]+)", path)
        if m and method == "POST":
            src_name, dst_name = urllib.parse.unquote(m.group(2)), urllib.parse.unquote(m.group(4))
            src = self.obj_path(m.group(1), src_name)
            dst = self.obj_path(m.group(3), dst_name)
            if not src.is_file():
                raise self.missing_meta(src_name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            if src_name in self.st["gcs_ct"]:
                self.st["gcs_ct"][dst_name] = self.st["gcs_ct"][src_name]
            return 200, self.meta(dst, dst_name)
        m = re.fullmatch(r"/storage/v1/b/([^/]+)/o/(.+)", path)
        if m:
            media = query.get("alt") == "media"
            if "/" in m.group(2):  # an object name must be one percent-encoded segment (live: 404)
                raise self.missing_media(m.group(2)) if media else self.missing_meta(m.group(2))
            name = urllib.parse.unquote(m.group(2))
            p = self.obj_path(m.group(1), name)
            if method == "GET":
                if not p.is_file():
                    raise self.missing_media(name) if media else self.missing_meta(name)
                if media:
                    raw = p.read_bytes()
                    gen = str(p.stat().st_mtime_ns)
                    hdr = {"Content-Type": self.content_type(name), "X-Goog-Generation": gen,
                           "X-Goog-Metageneration": "1", "X-Goog-Stored-Content-Length": str(len(raw)),
                           "Content-Length": str(len(raw))}
                    self.after_read(name, p, "media")
                    return 200, raw, hdr
                out = self.meta(p, name)
                self.after_read(name, p, "meta")
                return 200, out
            if method == "DELETE":
                if not p.is_file():
                    raise self.missing_meta(name)
                g = query.get("ifGenerationMatch")
                if g is not None and str(p.stat().st_mtime_ns) != str(g):
                    raise self.precondition_failed()
                p.unlink()
                self.st["gcs_ct"].pop(name, None)
                return 204, None
        m = re.fullmatch(r"/storage/v1/b/([^/]+)/o", path)
        if m and method == "GET":
            prefix = query.get("prefix", "")
            root = self.w.gcs / m.group(1)
            names = sorted(n for n in (q.relative_to(root).as_posix() for q in root.rglob("*")
                                       if q.is_file() and not q.name.startswith(".gcsio")) if n.startswith(prefix))
            size = max(1, min(int(query.get("maxResults") or 1000), 1000))
            start = int(base64.urlsafe_b64decode(query["pageToken"]).decode()) if query.get("pageToken") else 0
            page = names[start:start + size]
            terse = bool(query.get("fields"))
            out: dict = {} if terse else {"kind": "storage#objects"}
            if page:
                out["items"] = [{"name": n} if terse else {"kind": "storage#object", "bucket": BUCKET, "name": n}
                                for n in page]
            if start + size < len(names):
                out["nextPageToken"] = base64.urlsafe_b64encode(str(start + size).encode()).decode()
            return 200, out
        m = re.fullmatch(r"/upload/storage/v1/b/([^/]+)/o", path)
        if m and method == "POST":
            if query.get("uploadType") != "media" or not query.get("name"):
                raise HttpFail(400)
            name = query["name"]
            p = self.obj_path(m.group(1), name)
            g = query.get("ifGenerationMatch")
            if g is not None:
                if str(g) == "0" and p.exists():
                    raise self.precondition_failed()
                if str(g) != "0" and (not p.exists() or str(p.stat().st_mtime_ns) != str(g)):
                    raise self.precondition_failed()
            p.parent.mkdir(parents=True, exist_ok=True)
            ct = next((v for k, v in (headers or {}).items() if k.lower() == "content-type"), "application/octet-stream")
            old = p.stat().st_mtime_ns if p.exists() else None
            if isinstance(body, (bytes, bytearray)):
                p.write_bytes(bytes(body))
            else:
                p.write_text(body if isinstance(body, str) else json.dumps(body, indent=1))
            if old is not None and p.stat().st_mtime_ns == old:  # a new generation, always
                os.utime(p, ns=(old + 1000, old + 1000))
            self.st["gcs_ct"][name] = ct
            return 200, self.meta(p, name)
        raise wfsim.SimError(f"storage: no fake for {method} {path}")

    # ── Cloud Run v2 ──
    def run(self, method: str, path: str, query: dict, body: Any) -> tuple:
        p = path[len("/v2/"):]
        jobs, svcs = self.st["jobs"], self.st["services"]
        m = re.fullmatch(r"projects/[^/]+/locations/([^/]+)/operations/([^/:]+)", p)
        if m and method == "GET":
            return self.run_operation(p, m.group(1), m.group(2))
        m = re.fullmatch(r"projects/[^/]+/locations/([^/]+)/jobs/([^/:]+)(:run)?", p)
        if m:
            region, job, run = m.groups()
            if job not in jobs or region != NE2:
                raise HttpFail(404, google_error(404, f"Resource '{job}' of kind 'JOB' in region '{region}' in project '{PROJECT}' does not exist."))
            if method == "GET" and not run:
                return 200, copy.deepcopy(jobs[job])
            if method == "PATCH" and not run:
                return 200, self.patch_job(job, body)
            if method == "POST" and run:
                return 200, self.run_job(job, body)
        m = re.fullmatch(r"(projects/[^/]+/locations/[^/]+/jobs/([^/]+)/executions/([^/:]+))(/tasks|:cancel)?", p)
        if m:
            ex = self.st["executions"].get(m.group(1))
            if ex is None:
                raise HttpFail(404)
            if method == "GET" and m.group(4) == "/tasks":
                return 200, {"tasks": [self.task_view(ex)]}
            if method == "GET" and not m.group(4):
                return 200, self.execution_view(ex)
            if method == "POST" and m.group(4) == ":cancel":
                if not self.execution_done(ex):
                    ex["cancelled_at"] = self.interp_now()
                return 200, {"name": f"projects/{PROJECT}/locations/{NE2}/operations/{uuid.uuid4()}",
                             "metadata": {"@type": "type.googleapis.com/google.cloud.run.v2.Execution", **self.execution_view(ex)},
                             "done": False}
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
                return 200, self.patch_service(svc, body, query)
        m = re.fullmatch(r"projects/[^/]+/locations/[^/]+/services/([^/]+)/revisions/([^/]+)", p)
        if m and method == "GET":
            rev = self.st["revisions"].get(p)
            if rev is None:
                raise HttpFail(404)
            return 200, rev
        raise wfsim.SimError(f"run: no fake for {method} {path}")

    def new_op(self, kind: str, target: str, region: str, view: dict) -> dict:
        oid = str(uuid.uuid4())
        name = f"projects/{PROJECT}/locations/{region}/operations/{oid}"
        self.st["ops"][oid] = {"kind": kind, "target": target, "gen": view.get("generation")}
        typ = "Job" if kind == "job" else "Service"
        return {"name": name, "metadata": {"@type": f"type.googleapis.com/google.cloud.run.v2.{typ}", **view}, "done": False}

    def run_operation(self, name: str, region: str, oid: str) -> tuple:
        if not self.ops_allowed:
            self.st.setdefault("ops_denied", 0)
            self.st["ops_denied"] += 1
            raise HttpFail(403, google_error(403, f"Permission 'run.operations.get' denied on resource "
                                                  f"'//run.googleapis.com/{name}' (or resource may not exist)."))
        op = self.st["ops"].get(oid)
        if op is None:
            raise HttpFail(404)
        view = copy.deepcopy(self.st["jobs"][op["target"]]) if op["kind"] == "job" else self._service_view(op["target"])
        typ = "Job" if op["kind"] == "job" else "Service"
        out = {"name": name, "metadata": {"@type": f"type.googleapis.com/google.cloud.run.v2.{typ}", **view}, "done": False}
        if not view.get("reconciling") and int(view.get("observedGeneration") or 0) >= int(op["gen"] or 0):
            out["done"] = True
            tc = view.get("terminalCondition") or {}
            if tc.get("state") == "CONDITION_FAILED":
                out["error"] = {"code": 9, "message": tc.get("message", "")}
            else:
                out["response"] = {"@type": f"type.googleapis.com/google.cloud.run.v2.{typ}", **view}
        return 200, out

    def _service_view(self, svc: str) -> dict:
        s = copy.deepcopy(self.st["services"][svc])
        s.pop("region", None)
        if not s.get("reconciling"):
            s.pop("reconciling", None)  # proto3 JSON omits false
        return s

    def _check_same(self, cur: dict, new: dict, allowed: set, what: str) -> None:
        """A PATCH body must be the resource as read with only the allowed fields changed (output-only fields,
        which the API ignores, may differ: the resource can have moved on since the read)."""
        diffs: list[str] = []

        def walk(a: Any, b: Any, at: tuple) -> None:
            if any(at[:len(x)] == x for x in allowed):
                return
            if len(at) == 1 and at[0] in RUN_OUTPUT_ONLY:
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

    def _etag(self, obj: dict, race_key: str, body: dict, what: str) -> None:
        """Someone else's update between the caller's GET and this PATCH (a race mode) bumps the etag; a stale etag
        in the body is a 409 (audit CR-5)."""
        if self.consume(race_key):
            obj["etag"] = f'"e{uuid.uuid4().hex[:10]}"'
        if body.get("etag") is not None and body.get("etag") != obj.get("etag"):
            raise HttpFail(409, google_error(409, f"{what}: the etag in the request does not match the resource's "
                                                  f"current etag (it was updated concurrently)", "ABORTED"))

    def _start_reconcile(self, obj: dict) -> None:
        obj["generation"] = str(int(obj.get("generation") or 1) + 1)
        obj["reconciling"] = True
        obj["terminalCondition"] = {"type": "Ready", "state": "CONDITION_RECONCILING",
                                    "lastTransitionTime": now_iso(self.interp_now())}
        obj["etag"] = f'"e{uuid.uuid4().hex[:10]}"'
        obj["updateTime"] = now_iso(self.interp_now())

    def patch_job(self, job: str, body: dict) -> dict:
        cur = self.st["jobs"][job]
        self._etag(cur, f"run-etag-race:{job}", body, f"job {job}")
        self._check_same(cur, body, {("template", "template", "containers", 0, "image")}, f"job {job}")
        if body.get("template") != cur.get("template"):
            cur["template"] = copy.deepcopy(body["template"])
            self._start_reconcile(cur)
            self.schedule({"kind": "job", "job": job, "at": self.interp_now() + RECONCILE_S["job"], "gen": cur["generation"]})
        return self.new_op("job", job, NE2, copy.deepcopy(cur))

    def settle_job(self, ev: dict) -> None:
        j = self.st["jobs"][ev["job"]]
        j["observedGeneration"] = ev["gen"] if int(ev["gen"]) >= int(j.get("observedGeneration") or 0) else j["observedGeneration"]
        if j["observedGeneration"] == j["generation"]:
            j["reconciling"] = False
            j["terminalCondition"] = {"type": "Ready", "state": "CONDITION_SUCCEEDED",
                                      "lastTransitionTime": now_iso(self.interp_now())}

    def patch_service(self, svc: str, body: dict, query: dict) -> dict:
        s = self.st["services"][svc]
        self._etag(s, f"run-etag-race:{svc}", body, f"service {svc}")
        self._check_same(self._service_view(svc), body, {("template", "containers", 0, "image"), ("template", "annotations"),
                                                         ("template", "revision"), ("traffic",)}, f"service {svc}")
        force = str(query.get("forceNewRevision", "")).lower() == "true"
        tmpl = copy.deepcopy(body.get("template") or {})
        traffic = copy.deepcopy(body.get("traffic") or LATEST_100)  # empty = 100% to the latest ready revision
        tmpl_changed = tmpl != s["template"]
        if not (tmpl_changed or traffic != s.get("traffic") or force):
            # nothing to do: no new generation, no new revision (services/patch creates one only for a change
            # or with forceNewRevision; audit CR-2)
            self.st.setdefault("noop_patches", []).append({"service": svc, "at": now_iso(self.interp_now())})
            return self.new_op("svc", svc, s["region"], self._service_view(svc))
        want_rev = tmpl.get("revision")
        if want_rev and f"{s['name']}/revisions/{want_rev}" in self.st["revisions"] and (tmpl_changed or force):
            raise HttpFail(409, google_error(409, f"Revision named '{want_rev}' with different configuration already exists.",
                                             "ALREADY_EXISTS"))
        new_rev = tmpl_changed or force
        s["template"] = tmpl
        s["traffic"] = traffic
        self._start_reconcile(s)
        fail = new_rev and self.consume(f"revision-fail:{svc}")
        gpu = "nvidia.com/gpu" in json.dumps(tmpl.get("containers") or [])
        delay = RECONCILE_S["gpu-failed" if fail else ("gpu" if gpu else "cpu")]
        self.schedule({"kind": "svc", "svc": svc, "at": self.interp_now() + delay, "new_rev": new_rev, "fail": fail,
                       "gen": s["generation"]})
        if new_rev:
            self.st.setdefault("patches", []).append({"service": svc, "image": tmpl["containers"][0]["image"],
                                                      "at": now_iso(self.interp_now()), "forced": force and not tmpl_changed,
                                                      "annotations": tmpl.get("annotations") or {}, "fails": fail})
        return self.new_op("svc", svc, s["region"], self._service_view(svc))

    def settle_service(self, ev: dict) -> None:
        svc = ev["svc"]
        s = self.st["services"][svc]
        now = self.interp_now()
        tc = {"type": "Ready", "state": "CONDITION_SUCCEEDED", "lastTransitionTime": now_iso(now)}
        if ev["new_rev"]:
            name = self.add_revision(svc, s["template"], ready=not ev["fail"])
            s["latestCreatedRevision"] = name
            if ev["fail"]:
                tc = {"type": "Ready", "state": "CONDITION_FAILED", "reason": "RevisionFailed", "lastTransitionTime": now_iso(now),
                      "message": GPU_REVISION_FAILED.format(rev=short_rev(name), region=s["region"])}
            else:
                s["latestReadyRevision"] = name
        if ev.get("fail") and ev["gen"] == s["generation"]:
            # a failed reconciliation: "traffic_statuses, observed_generation, and latest_ready_revision will have the
            # state of the last serving revision" (v2 Service `reconciling`); reconciling ends, terminalCondition says why
            s["reconciling"] = False
            s["terminalCondition"] = tc
        else:
            if int(ev["gen"]) >= int(s.get("observedGeneration") or 0):
                s["observedGeneration"] = ev["gen"]
            if s["observedGeneration"] == s["generation"]:
                s["reconciling"] = False
                s["terminalCondition"] = tc
        s["trafficStatuses"] = self.traffic_statuses(s)

    def add_revision(self, svc: str, template: dict, *, ready: bool = True, suffix: str | None = None) -> str:
        s = self.st["services"][svc]
        n = int(self.st.setdefault("revno", {}).get(svc, 1)) + 1
        self.st["revno"][svc] = n
        name = f"{s['name']}/revisions/{svc}-{n:05d}-{suffix or uuid.uuid4().hex[:3]}"
        c0 = copy.deepcopy(template["containers"][0])
        c0["image"] = revision_image(c0["image"], self.st["tags"])
        self.st["revisions"][name] = {
            "name": name, "service": svc, "generation": "1", "containers": [c0], "createTime": now_iso(self.interp_now()),
            "annotations": copy.deepcopy(template.get("annotations") or {}),
            "conditions": [{"type": "Ready", "state": "CONDITION_SUCCEEDED" if ready else "CONDITION_FAILED",
                            "lastTransitionTime": now_iso(self.interp_now())}]}
        return name

    @staticmethod
    def traffic_statuses(s: dict) -> list[dict]:
        """What the service actually routes: the LATEST entry resolved to the latest READY revision."""
        host = urllib.parse.urlsplit(s.get("uri") or "https://x.a.run.app").netloc
        out = []
        for t in s.get("traffic") or LATEST_100:
            e: dict = {"type": t.get("type", "TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION")}
            e["revision"] = short_rev(s["latestReadyRevision"]) if e["type"] == "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST" else t.get("revision")
            if t.get("percent"):
                e["percent"] = t["percent"]
            if t.get("tag"):
                e["tag"] = t["tag"]
                e["uri"] = f"https://{t['tag']}---{host}"
            out.append(e)
        return out

    def serving(self, svc: str) -> list[tuple[str, int]]:
        """(full revision name, percent) of every revision that gets traffic."""
        s = self.st["services"][svc]
        return [(f"{s['name']}/revisions/{e['revision']}", int(e.get("percent") or 0))
                for e in self.traffic_statuses(s) if e.get("percent")]

    def deploy_preview(self, svc: str, image: str, tag: str) -> str:
        """`gcloud run deploy --no-traffic --tag <tag>` as gcloud does it: the template gets the image, a new
        revision becomes latestCreated = latestReady, the LATEST traffic entry is pinned to the revision that was
        serving, and the new one gets only the tag (docs/PIPELINE.md's int rollout, before update-traffic)."""
        s = self.st["services"][svc]
        serving = short_rev(s["latestReadyRevision"])
        s["template"]["containers"][0]["image"] = image
        name = self.add_revision(svc, s["template"])
        s["latestCreatedRevision"] = s["latestReadyRevision"] = name
        s["traffic"] = [{"type": "TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION", "revision": serving, "percent": 100},
                        {"type": "TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION", "revision": short_rev(name), "tag": tag}]
        s["generation"] = s["observedGeneration"] = str(int(s["generation"]) + 1)
        s["trafficStatuses"] = self.traffic_statuses(s)
        return name

    # ── jobs as local processes ──
    def lab_image_ok(self, image: str) -> bool:
        tag_ref = f"{LAB_REPO}/sous-chef-lab:{SERVICE_TAG}"
        digest = self.st["tags"].get(f"{NE2}/sous-chef/sous-chef-lab:{SERVICE_TAG}")
        return image in (tag_ref, f"{LAB_REPO}/sous-chef-lab@{digest}")

    def run_job(self, job: str, body: dict) -> dict:
        ov = (body or {}).get("overrides") or {}
        co = (ov.get("containerOverrides") or [{}])[0]
        args = [str(a) for a in co.get("args") or []]
        env_over = {e["name"]: e["value"] for e in co.get("env") or []}
        tmpl = self.st["jobs"][job]["template"]["template"]
        image = tmpl["containers"][0]["image"]
        if not image.startswith(f"{LAB_REPO}/sous-chef-lab"):
            raise wfsim.SimError(f"job {job} runs {image}, not the lab image (job_set_image missed it)")
        n = len(self.st["executions"]) + 1
        ex_name = f"projects/{PROJECT}/locations/{NE2}/jobs/{job}/executions/{job}-{n:04d}"
        cmd = args[0] if args else "selftest"
        fail = self.fail_next.get(cmd, 0)
        wrong = not self.lab_image_ok(image)
        if wrong:  # it would run, but another build's lab: recorded, and the execution fails so nobody misses it
            self.st["image_violations"].append({"job": job, "image": image, "args": args[:2], "at": now_iso(self.interp_now())})
        rc, secs = self.exec_lab(job, args, env_over, ex_name, fail_with=1 if (fail or wrong) else None)
        if fail:
            self.fail_next[cmd] = fail - 1
            self.st["fail_next"] = self.fail_next
        t0 = self.interp_now()
        self.st["executions"][ex_name] = {"name": ex_name, "job": job, "args": args, "exit_code": rc, "image": image,
                                          "start": t0, "end": t0 + max(secs, 45.0)}
        # the job's reference to its newest execution holds the SHORT name (live job_drift.json)
        self.st["jobs"][job]["latestCreatedExecution"] = {"name": ex_name.rsplit("/", 1)[-1], "createTime": now_iso(t0)}
        self.st["jobs"][job]["executionCount"] = int(self.st["jobs"][job].get("executionCount") or 0) + 1
        self.st.setdefault("job_log", []).append({"execution": ex_name.split("/")[-1], "args": args[:4], "exit": rc,
                                                  "seconds": round(secs, 1)})
        log(f"      job {job} {' '.join(args[:2])} → exit {rc} ({secs:.0f} s)")
        return {"name": f"projects/{PROJECT}/locations/{NE2}/operations/{uuid.uuid4()}",
                "metadata": {"@type": "type.googleapis.com/google.cloud.run.v2.Execution", "name": ex_name,
                             "job": job, "createTime": now_iso(t0)}, "done": False}

    def execution_done(self, ex: dict) -> bool:
        return ex.get("cancelled_at") is not None or self.interp_now() >= ex["end"]

    def execution_view(self, ex: dict) -> dict:
        """An Execution as v2 shows it: no completionTime until it ends; succeededCount / failedCount /
        cancelledCount and a Completed condition after (live: exec_drift.json)."""
        now = self.interp_now()
        out: dict = {"name": ex["name"], "job": ex["job"], "createTime": now_iso(ex["start"]), "taskCount": 1,
                     "startTime": now_iso(ex["start"] + 5), "template": {"containers": [{"image": revision_image(ex["image"], self.st["tags"]),
                                                                                         "args": ex["args"]}]}}
        cancelled = ex.get("cancelled_at")
        if cancelled is not None and cancelled < ex["end"]:
            out.update(completionTime=now_iso(cancelled), cancelledCount=1,
                       conditions=[{"type": "Completed", "state": "CONDITION_FAILED", "reason": "Cancelled",
                                    "message": "Execution was cancelled."}])
        elif now >= ex["end"]:
            ok = ex["exit_code"] == 0
            out["completionTime"] = now_iso(ex["end"])
            out["succeededCount" if ok else "failedCount"] = 1
            out["conditions"] = [{"type": "Completed", "state": "CONDITION_SUCCEEDED" if ok else "CONDITION_FAILED",
                                  "message": "Execution completed successfully." if ok else "Task failed."}]
        else:
            out["runningCount"] = 1
            out["conditions"] = [{"type": "Started", "state": "CONDITION_SUCCEEDED"},
                                 {"type": "Completed", "state": "CONDITION_PENDING"}]
        return out

    def task_view(self, ex: dict) -> dict:
        t: dict = {"name": ex["name"] + "/tasks/" + ex["name"].rsplit("/", 1)[-1] + "-task0", "index": 0}
        if self.interp_now() >= ex["end"] and ex.get("cancelled_at") is None:
            rc = ex["exit_code"]
            # a zero exit code is omitted (live tasks_drift.json: lastAttemptResult {"status": {}})
            t["lastAttemptResult"] = {"status": {}} if rc == 0 else {
                "exitCode": rc, "status": {"code": 2, "message": f"Task {t['name'].rsplit('/', 1)[-1]} failed with exit code {rc}"}}
        return t

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
    # versions/<digest>: 200 for any version in the package, tagged or not; relatedTags only with view=FULL and
    # omitted when empty (repeated fields are), up to 100 tags incl. `latest` (live trainer version: bfb0ac3,
    # latest, t-de71aa0f078b). https://docs.cloud.google.com/artifact-registry/docs/reference/rest/v1/projects.locations.repositories.packages.versions
    def ar(self, method: str, path: str, query: dict, body: Any) -> tuple:
        v = re.fullmatch(r"/v1/projects/[^/]+/locations/([^/]+)/repositories/([^/]+)/packages/([^/]+)/versions/([^/]+)", path)
        if v and method == "GET":
            loc, repo, pkg, ver = (urllib.parse.unquote(x) for x in v.groups())
            base = f"projects/{PROJECT}/locations/{loc}/repositories/{repo}/packages/{pkg}"
            names = [k.rsplit(":", 1)[1] for k, d in self.st["tags"].items() if k.startswith(f"{loc}/{repo}/{pkg}:") and d == ver]
            untagged = ver in (self.st.get("untagged_versions") or {}).get(f"{loc}/{repo}/{pkg}", [])
            if not names and not untagged:
                raise HttpFail(404, google_error(404, f"Requested entity was not found."))
            out = {"name": f"{base}/versions/{ver}", "createTime": "2026-10-09T00:00:00Z", "updateTime": "2026-10-09T00:00:00Z"}
            if query.get("view") == "FULL" and names:
                order = sorted(names, key=lambda n: (n != "latest", n))  # `latest` first: the worst case for [0]
                out["relatedTags"] = [{"name": f"{base}/tags/{n}", "version": f"{base}/versions/{ver}"} for n in order]
            return 200, out
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
                raise HttpFail(404, google_error(404, "Requested entity was not found."))
            return 200, {"name": f"{base}/tags/{tag}", "version": f"{base}/versions/{tags[key]}"}
        if method == "POST" and not tag:
            tid = query.get("tagId")
            ver = body["version"].split("/versions/")[-1]
            key = f"{loc}/{repo}/{pkg}:{tid}"
            if key in tags:
                raise HttpFail(409, google_error(409, f"tag {tid} already exists", "ALREADY_EXISTS"))
            tags[key] = ver
            return 200, {"name": f"{base}/tags/{tid}", "version": body["version"]}
        raise wfsim.SimError(f"ar: {method} {path}")

    # ── Vertex AI custom jobs ──
    # States move on the virtual clock: QUEUED → PENDING → RUNNING (startTime) → a terminal state; the trainer runs
    # when the job's run ends. Flex-start: PENDING until maxWaitDuration (0 = forever, absent = 24 h), then FAILED
    # with no startTime and an undocumented message. Spot: a preemption restarts the job inside Vertex (up to six
    # times, the trainer resumes from its checkpoints); past that, FAILED with the STOCKOUT text and a startTime.
    # cancel → CANCELLING → CANCELLED ("success is not guaranteed": a job that got capacity first SUCCEEDS).
    # Modes (fail_next): vertex-pending (Spot that never starts), vertex-late-success (with it: the cancel loses
    # the race), vertex-preempt = N restarts inside the next job.
    def vertex(self, method: str, region: str, path: str, body: Any, query: dict | None = None) -> tuple:
        vj = self.st["vertex"]
        now = self.interp_now()
        m = re.fullmatch(r"/v1/projects/[^/]+/locations/([^/]+)/customJobs", path)
        if m and method == "GET":
            # customJobs.list: filter `labels.key=value` / `displayName="…"` (AND-joined); an empty page is {}
            # (https://docs.cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.customJobs/list)
            conds = [c.strip() for c in re.split(r"\s+AND\s+", str((query or {}).get("filter") or "")) if c.strip()]
            def hit(rec: dict) -> bool:
                for c in conds:
                    mm = re.fullmatch(r'(labels\.[a-z0-9_-]+|displayName)\s*=\s*"?([^"]*)"?', c)
                    if not mm:
                        raise HttpFail(400, google_error(400, f"Invalid filter: {c}"))
                    k, v = mm.groups()
                    got = rec.get("displayName") if k == "displayName" else (rec.get("labels") or {}).get(k[7:])
                    if got != v:
                        return False
                return True
            jobs = [self.vertex_view(r) for r in vj.values() if r["region"] == m.group(1) and hit(r)]
            return 200, ({"customJobs": jobs} if jobs else {})
        if m and method == "POST":
            if m.group(1) != region:
                raise HttpFail(400, google_error(400, "region mismatch"))
            pool = body["jobSpec"]["workerPoolSpecs"][0]
            cs = pool["containerSpec"]
            sched = body["jobSpec"].get("scheduling") or {}
            assert pool["machineSpec"]["acceleratorCount"] == 1 and pool["replicaCount"] == 1, pool
            trainer_d = self.st["tags"].get(f"{NE2}/sous-chef/sous-chef-trainer:{SERVICE_TAG}")
            want = (f"{LAB_REPO}/sous-chef-trainer:{SERVICE_TAG}", f"{LAB_REPO}/sous-chef-trainer@{trainer_d}")
            if cs["imageUri"] not in want:
                raise wfsim.SimError(f"vertex: trainer image {cs['imageUri']} is not {want[0]} (or its digest)")
            env = {e["name"]: e["value"] for e in cs.get("env") or []}
            if not str(env.get("PIPELINE_BUCKET", "")).startswith("gs://"):
                raise wfsim.SimError(f"vertex: PIPELINE_BUCKET {env.get('PIPELINE_BUCKET')!r} is not gs://")
            run_id = str((cs.get("args") or ["?"])[0])
            # a second live job for the same round (an orphan, a retried create, a cancel not awaited): recorded
            live = [k for k, r in vj.items() if r.get("_run") == run_id and r["state"] not in VERTEX_TERMINAL]
            n = len(vj) + 1
            name = f"projects/{PROJECT}/locations/{region}/customJobs/{1000 + n}"
            if live:
                self.st["vertex_overlap"].append({"new": name, "live": live, "at": now_iso(now)})
            rec = {"name": name, "displayName": body["displayName"], "state": "JOB_STATE_QUEUED", "region": region,
                   "strategy": sched.get("strategy"), "machine": pool["machineSpec"]["machineType"],
                   "labels": body.get("labels") or {}, "createTime": now_iso(now), "updateTime": now_iso(now),
                   "_run": run_id, "_args": cs.get("args") or [], "_env": env, "_t0": now}
            if sched.get("strategy") == "FLEX_START":
                wait = sched.get("maxWaitDuration")
                secs = 86400.0 if wait is None else float(str(wait).rstrip("s") or 0)
                rec["_plan"] = {"kind": "flex", "fail_at": now + secs if secs > 0 else None}
            elif self.consume("vertex-pending"):
                rec["_plan"] = {"kind": "pending", "late": self.consume("vertex-late-success")}
            else:
                restarts = self.fail_next.get("vertex-preempt", 0)
                if restarts:
                    self.fail_next["vertex-preempt"] = 0
                    self.st["fail_next"] = self.fail_next
                rec["_plan"] = {"kind": "run", "start_at": now + 240, "end_at": now + 240 + 1800 + 600 * restarts,
                                "restarts": restarts}
            vj[name] = rec
            log(f"      vertex {rec['displayName']} ({rec['strategy']}) created: {rec['_plan']['kind']}")
            return 200, self.vertex_view(rec)
        m = re.fullmatch(r"/v1/(projects/[^/]+/locations/[^/]+/customJobs/[^/:]+)(:cancel)?", path)
        if m:
            rec = vj.get(m.group(1))
            if rec is None:
                raise HttpFail(404)
            if m.group(2) and method == "POST":
                if rec["state"] not in VERTEX_TERMINAL and rec["state"] != "JOB_STATE_CANCELLING":
                    if rec["_plan"]["kind"] == "pending" and rec["_plan"].get("late"):
                        rec["startTime"] = now_iso(now)  # capacity arrived just before the cancel
                    rec["state"] = "JOB_STATE_CANCELLING"
                    rec["_cancel_at"] = now + 90
                    rec["updateTime"] = now_iso(now)
                return 200, {}
            if method == "GET":
                return 200, self.vertex_view(rec)
        raise wfsim.SimError(f"vertex: {method} {path}")

    @staticmethod
    def vertex_view(rec: dict) -> dict:
        return {k: v for k, v in rec.items() if not k.startswith("_") and k not in ("region", "strategy", "machine")}

    def tick_vertex(self, now: float) -> None:
        for rec in self.st["vertex"].values():
            if rec["state"] in VERTEX_TERMINAL:
                continue
            plan = rec["_plan"]
            if rec["state"] == "JOB_STATE_CANCELLING":
                if now >= rec["_cancel_at"]:
                    if plan["kind"] == "pending" and plan.get("late"):
                        self.finish_vertex(rec, rec["_cancel_at"], run=True)
                    else:
                        rec.update(state="JOB_STATE_CANCELLED", endTime=now_iso(rec["_cancel_at"]),
                                   error={"code": 1, "message": "CANCELED"})
                continue
            if rec["state"] == "JOB_STATE_QUEUED" and now >= rec["_t0"] + 30:
                rec["state"] = "JOB_STATE_PENDING"
            if plan["kind"] == "flex" and plan["fail_at"] is not None and now >= plan["fail_at"]:
                rec.update(state="JOB_STATE_FAILED", endTime=now_iso(plan["fail_at"]),
                           error={"code": 4, "message": VERTEX_FLEX_GAVE_UP})
            elif plan["kind"] == "run":
                if now >= plan["start_at"] and "startTime" not in rec:
                    rec.update(state="JOB_STATE_RUNNING", startTime=now_iso(plan["start_at"]))
                if now >= plan["end_at"]:
                    self.finish_vertex(rec, plan["end_at"], run=plan["restarts"] <= VERTEX_MAX_RESTARTS)
            rec["updateTime"] = now_iso(now)

    def finish_vertex(self, rec: dict, at: float, *, run: bool) -> None:
        rec.setdefault("startTime", now_iso(at))
        rec["endTime"] = now_iso(at)
        if not run:
            # preempted more than six times: the trainer's SIGTERM trap wrote "interrupted" (and pushed its
            # checkpoints) on the last one; Vertex gives up with a STOCKOUT
            prog = self.w.bucket / "runs" / rec["_run"] / "train" / "progress.json"
            prog.parent.mkdir(parents=True, exist_ok=True)
            prog.write_text(json.dumps({"status": "interrupted", "step": "SFT", "error": "signal during SFT",
                                        "updated_at": now_iso(at), "last_step": 50}))
            rec.update(state="JOB_STATE_FAILED", error={"code": 8, "message": VERTEX_STOCKOUT.format(region=rec["region"])})
            log(f"      vertex {rec['displayName']} → FAILED (stockout after {rec['_plan']['restarts']} preemptions)")
            return
        rc, _ = self.run_trainer(rec["_args"], rec["_env"])
        rec["state"] = "JOB_STATE_SUCCEEDED" if rc == 0 else "JOB_STATE_FAILED"
        if rc:
            rec["error"] = {"code": 3, "message": f"The replica workerpool0-0 exited with a non-zero status of {rc}."}
        log(f"      vertex {rec['displayName']} → {rec['state']}")

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
    # A build is QUEUED, then WORKING, then terminal on the virtual clock; images are pushed at its end (immutable
    # tags in sous-chef-model: a second push of a tag fails the build). The worker and the docker client are
    # checked as Cloud Build would run them (WORKER_ENGINE, docker_major above). Mode: build-timeout-after-push =
    # the next build pushes its image, then ends TIMEOUT (audit RB-4).
    def cloudbuild(self, method: str, path: str, body: Any) -> tuple:
        builds = self.st["builds"]
        m = re.fullmatch(r"/v1/projects/[^/]+/locations/([^/]+)/builds(?:/([^/]+))?", path)
        if not m:
            raise wfsim.SimError(f"cloudbuild: {method} {path}")
        if method == "GET" and m.group(2):
            b = builds.get(m.group(2))
            if b is None:
                raise HttpFail(404)
            return 200, {k: v for k, v in b.items() if not k.startswith("_")}
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
        now = self.interp_now()
        release = (body.get("options") or {}).get("workerRelease")
        if release is not None and release not in WORKER_ENGINE:
            raise HttpFail(400, google_error(400, f"invalid value for options.workerRelease: {release!r}"))
        engine = WORKER_ENGINE["legacy" if release is None and legacy_default(now) else (release or "regular")]
        src = body["source"]["storageSource"]
        tgz = self.obj_path(src["bucket"], src["object"])
        if not tgz.is_file():
            raise HttpFail(400, google_error(400, f"source object gs://{src['bucket']}/{src['object']} not found"))
        bid = str(uuid.uuid4())
        status, detail, digest = "SUCCESS", "", None
        problem = self.builder_problem(body["steps"], engine)
        if problem:
            status, detail = "FAILURE", problem
        else:
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
        push = None
        t_done = now + 60
        if status == "SUCCESS":
            tag = subs["_IMAGE"].rsplit(":", 1)[1]
            push = [f"{GPU}/sous-chef-model/model:{tag}", digest]
            t_done = now + 540
            if self.consume("build-timeout-after-push"):
                status, detail, t_done = "TIMEOUT", "Build timeout after the image was pushed (e2e fault)", now + 3600
        rec = {"id": bid, "status": "QUEUED", "createTime": now_iso(now), "timeout": body.get("timeout", "600s"),
               "logUrl": f"https://console.cloud.google.com/cloud-build/builds;region={GPU}/{bid}?project={PROJECT}",
               "_final": status, "_detail": detail, "_push": push, "_t_work": now + 20, "_t_done": t_done,
               "_image": subs["_IMAGE"], "_release": release, "_engine": engine}
        builds[bid] = rec
        log(f"      cloud build {subs['_IMAGE'].rsplit('/', 1)[-1]} submitted (worker {release or 'default'}, Engine {engine})"
            + (f": will end {status} {detail}" if status != "SUCCESS" else ""))
        return 200, {"name": f"projects/{PROJECT}/locations/{GPU}/operations/{bid}",
                     "metadata": {"@type": "type.googleapis.com/google.devtools.cloudbuild.v1.BuildOperationMetadata",
                                  "build": {k: v for k, v in rec.items() if not k.startswith("_")}}, "done": False}

    @staticmethod
    def builder_problem(steps: list, engine: int) -> str | None:
        """Why the docker steps fail on this worker, as Cloud Build would end them (FAILURE + a log line)."""
        for s in steps:
            name = str(s.get("name", ""))
            is_docker = s.get("entrypoint") == "docker" or name.rsplit("/", 1)[-1].split("@")[0].split(":")[0] == "docker"
            if not is_docker:
                continue
            args = [str(a) for a in s.get("args") or []]
            client = docker_major(name)
            if engine >= 29 and client is not None and client < 25:
                return (f"step {s.get('id')}: Error response from daemon: client version 1.{41 if client < 23 else 43} is too old. "
                        "Minimum supported API version is 1.44, please upgrade your client to a newer version")
            if "--build-context" in args:
                if client is not None and client < 23:
                    return f"step {s.get('id')}: unknown flag: --build-context (docker {client} on {name})"
                if engine < 23:
                    return (f"step {s.get('id')}: named build contexts are not supported by the legacy worker's BuildKit "
                            "(Docker Engine 20.10; set options.workerRelease)")
        return None

    def tick_builds(self, now: float) -> None:
        for b in self.st["builds"].values():
            if b["status"] not in ("QUEUED", "WORKING"):
                continue
            if now >= b["_t_done"]:
                status = b["_final"]
                if b["_push"]:
                    key, digest = b["_push"]
                    if key in self.st["tags"]:
                        status, b["_detail"] = "FAILURE", (f"push {b['_image']}: tag {key.rsplit(':', 1)[1]} already exists "
                                                           "and the repository has immutable tags")
                    else:
                        self.st["tags"][key] = digest
                b["status"] = status
                b["finishTime"] = now_iso(b["_t_done"])
                b.setdefault("startTime", now_iso(b["_t_work"]))
                if b["_detail"]:
                    b["statusDetail"] = b["_detail"]
                if status == "SUCCESS":
                    b["results"] = {"images": [{"name": b["_image"], "digest": b["_push"][1]}]}
                log(f"      cloud build {b['_image'].rsplit('/', 1)[-1]} → {status} {b.get('statusDetail', '')}")
            elif now >= b["_t_work"] and b["status"] == "QUEUED":
                b["status"] = "WORKING"
                b["startTime"] = now_iso(b["_t_work"])

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

    def thermometer(self, method: str, url: str, query: dict, headers: dict, body: Any) -> tuple:
        key = next((str(v) for k, v in headers.items() if k.lower() == "x-api-key"), None)
        if key is not None and key != key.strip():
            # the thermometer hashes the header as sent (auth/middleware.go): a key read from a secret version added
            # with `echo` ends in "\n" (audit RT-5) and matches nothing (a raw newline cannot even go into an HTTP
            # header); either way the request is refused
            code, out = 401, {"error": "invalid API key", "code": "UNAUTHORIZED"}
            if method == "POST" and isinstance(body, dict):
                self.st.setdefault("posted", []).append({"id": body.get("id"), "kind": body.get("kind"), "code": code,
                                                         "title": body.get("title"), "answer": out})
            return code, (json.dumps(out) if self.raw_error_bodies else out)
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
        if code >= 400 and self.raw_error_bodies and raw:
            # Workflows documents map conversion for application/json responses, not what HttpError.body holds
            # (syntax/error-types: "response body"); this mode hands the error body over as the raw text
            return code, raw.decode("utf-8", "replace"), {"Content-Type": "application/json"}
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


def gcp_fixture() -> dict:
    # Cloud Run, Artifact Registry as Terraform and CI leave them, in the shapes the live int resources have
    # (scratchpad/audit svc_*.json, job_drift.json: generation strings, terminalCondition, traffic/trafficStatuses)
    jobs = {}
    for key, name in (("drift", "souschef-drift"), ("eval", "souschef-eval"), ("prepare", "souschef-prepare"),
                      ("collect", "souschef-collect"), ("dataset", "souschef-dataset")):
        jobs[name] = {"name": f"projects/{PROJECT}/locations/{NE2}/jobs/{name}", "uid": str(uuid.uuid4()), "etag": '"e1"',
                      "generation": "1", "observedGeneration": "1", "launchStage": "GA",
                      "terminalCondition": {"type": "Ready", "state": "CONDITION_SUCCEEDED"},
                      "template": {"parallelism": 1, "taskCount": 1, "template": {
                          "maxRetries": 0, "timeout": "3600s", "serviceAccount": f"souschef-x@{PROJECT}.iam.gserviceaccount.com",
                          "containers": [{"image": "us-docker.pkg.dev/cloudrun/container/job:latest", "args": ["selftest"],
                                          "resources": {"limits": {"cpu": "4", "memory": "16Gi"}}}],
                          "executionEnvironment": "EXECUTION_ENVIRONMENT_GEN2"}}}

    def svc(name: str, region: str, image: str, url: str, gpu: bool) -> dict:
        rev = f"projects/{PROJECT}/locations/{region}/services/{name}/revisions/{name}-00001-abc"
        limits = {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": "1"} if gpu else {"cpu": "1", "memory": "512Mi"}
        out = {"name": f"projects/{PROJECT}/locations/{region}/services/{name}", "region": region, "uri": url, "etag": '"e1"',
               "uid": str(uuid.uuid4()), "generation": "1", "observedGeneration": "1", "ingress": "INGRESS_TRAFFIC_ALL",
               "template": {"containers": [{"image": image, "resources": {"limits": limits}}],
                            "scaling": {"maxInstanceCount": 1 if gpu else 4}, "annotations": {}},
               "traffic": copy.deepcopy(LATEST_100),
               "terminalCondition": {"type": "Ready", "state": "CONDITION_SUCCEEDED", "lastTransitionTime": "2026-10-09T00:00:00Z"},
               "conditions": [{"type": "RoutesReady", "state": "CONDITION_SUCCEEDED"},
                              {"type": "ConfigurationsReady", "state": "CONDITION_SUCCEEDED"}],
               "latestReadyRevision": rev, "latestCreatedRevision": rev}
        if gpu:
            out["template"]["nodeSelector"] = {"accelerator": "nvidia-l4"}
            out["template"]["gpuZonalRedundancyDisabled"] = True
        out["trafficStatuses"] = FakeGCP.traffic_statuses(out)
        return out
    services = {"sous-chef": svc("sous-chef", NE2, f"{LAB_REPO}/sous-chef:{SERVICE_TAG}", "https://sous-chef-int.example", False),
                "sous-chef-model": svc("sous-chef-model", GPU, f"{MODEL_REPO}/model:r4", PROD_URL, True),
                "sous-chef-model-eval": svc("sous-chef-model-eval", GPU, f"{MODEL_REPO}/model:r4", EVAL_URL, True),
                "sous-chef-model-cand": svc("sous-chef-model-cand", GPU, f"{MODEL_REPO}/model:r4", CAND_URL, True)}
    # the service build's version carries several tags, `latest` among them, and one (a hand-built lab image,
    # `make lab-image TAG=…`) that has a lab image but no trainer: only SERVICE_TAG has both (audit RB-3)
    svc_digest = "sha256:" + "3" * 64
    tags = {f"{NE2}/sous-chef/sous-chef-lab:{SERVICE_TAG}": "sha256:" + "1" * 64,
            f"{NE2}/sous-chef/sous-chef-trainer:{SERVICE_TAG}": "sha256:" + "2" * 64,
            f"{NE2}/sous-chef/sous-chef:{SERVICE_TAG}": svc_digest,
            f"{NE2}/sous-chef/sous-chef:latest": svc_digest,
            f"{NE2}/sous-chef/sous-chef:{LABONLY_TAG}": svc_digest,
            f"{NE2}/sous-chef/sous-chef-lab:{LABONLY_TAG}": "sha256:" + "4" * 64,
            f"{NE2}/sous-chef/sous-chef-lab:latest": "sha256:" + "1" * 64,
            f"{NE2}/sous-chef/sous-chef-trainer:latest": "sha256:" + "2" * 64,
            f"{GPU}/sous-chef-model/model:r4": R4_DIGEST}
    revisions = {}
    for s in services.values():
        c0 = copy.deepcopy(s["template"]["containers"][0])
        c0["image"] = revision_image(c0["image"], tags)
        revisions[s["latestReadyRevision"]] = {"name": s["latestReadyRevision"], "service": s["name"].rsplit("/", 1)[1],
                                               "generation": "1", "containers": [c0], "createTime": "2026-10-09T00:00:00Z",
                                               "conditions": [{"type": "Ready", "state": "CONDITION_SUCCEEDED"}]}
    return {"jobs": jobs, "services": services, "revisions": revisions, "tags": tags,
            "untagged_versions": {f"{NE2}/sous-chef/sous-chef": [UNTAGGED_DIGEST]}}


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
    # the base weights (a stand-in tar) as R0 seeds them, in the real tar's layout (2026-10-14 header walk): the
    # hub-cache folder at the top, refs/main naming the snapshot, the snapshot's files hard links to blobs/
    with tempfile.TemporaryDirectory() as td:
        hub = Path(td) / "models--unsloth--Qwen3.5-9B"
        rev = "005429cee5cb648998cf2b70eebdd83175989c9a"
        (hub / "refs").mkdir(parents=True)
        (hub / "refs" / "main").write_text(rev)
        (hub / "blobs").mkdir()
        (hub / "snapshots" / rev).mkdir(parents=True)
        for name, body in (("config.json", "{}"), ("model.safetensors-00001-of-00004.safetensors", "w")):
            (hub / "blobs" / name).write_text(body)
            os.link(hub / "blobs" / name, hub / "snapshots" / rev / name)
        base = w.bucket / "base"
        base.mkdir(parents=True)
        subprocess.run(["tar", "-C", td, "-cf", str(base / "unsloth-qwen3.5-9b-hf.tar"), hub.name], check=True)
        sha = subprocess.run(["sha256sum", "unsloth-qwen3.5-9b-hf.tar"], cwd=base, capture_output=True, text=True, check=True).stdout
        (base / "unsloth-qwen3.5-9b-hf.tar.sha256").write_text(sha)
    # what Terraform writes into the bucket (bucket.tf), the model-image pins filled in
    mi = LAB / "model-image"
    pins = ";".join(f"{hashlib.sha256((mi / f).read_bytes()).hexdigest()}  model-image/{f}"
                    for f in ("Dockerfile", ".dockerignore", "entrypoint.sh", "model-context.sh"))
    b.write_json("infra/cloudbuild-model.json",
                 yaml.safe_load((HERE.parent / "cloudbuild-model.yaml").read_text().replace("__MODEL_IMAGE_SHA256SUMS__", pins)))
    # Cloud Run, Artifact Registry as Terraform and CI leave them
    fx = gcp_fixture()
    w.s = {"clock": time.time(), **fx, "executions": {}, "vertex": {}, "builds": {}, "started": [], "posted": [],
           "snapshot": str(snapshot), "phases": {}}
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

    def __init__(self, w: World, *, decide: list[dict] | None = None, config: dict | None = None):
        self.w = w
        self.decide_plan = list(decide or [])
        self.decisions: list[dict] = []
        self.clock = wfsim.VirtualClock(w.clock_t)
        self.ref: dict[str, Any] = {"clock": self.clock}
        self.config_over = dict(config or {})
        self.secret_suffix = ""  # e.g. "\n": the secret version as `echo key | gcloud secrets versions add` stores it

    def __enter__(self) -> "Run":
        self.svc = IntService(self.w).__enter__()
        self.therm = Thermometer(self.w).__enter__()
        write_config(self.w, self.svc.url, self.therm.url, **self.config_over)
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
        return self.therm.key + self.secret_suffix

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
        except wfsim.ResourceLimitError as e:  # uncatchable: the execution fails at once, no except ran
            out, status = e.value, "FAILED"
        finally:
            self.gcp.thermometer = orig  # type: ignore[method-assign]
            self.w.save()
        log(f"   ◀ {name} {status} in {time.monotonic() - t0:.0f} s real, {interp.steps} steps, "
            f"{interp.peak_memory // 1024} KB peak variables: {json.dumps(out, default=str)[:400]}")
        self.w.s.setdefault("executions_log", []).append({"workflow": name, "args": args, "status": status, "result": out,
                                                         "steps": interp.steps, "peak_memory": interp.peak_memory,
                                                         "max_depth": interp.max_depth,
                                                         "virtual_end": wfsim.fmt_time(self.clock.now())})
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
            # a real revision records its image by digest, so the rollback names r4's digest (make model-rollout takes either)
            c.ok(ro.get("state") == "done" and any(f"model-rollout TAG={t}" in str(ro.get("rollback")) for t in ("r4", R4_DIGEST)),
                 f"rollout.json records the rollback command ({ro.get('rollback')})")
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

        # d. Spot that never starts, then a preemption on the next rung. A real Spot preemption restarts the job
        # inside Vertex (the trainer resumes from its checkpoints) and the workflow sees one job that succeeds
        # (audit VX-4); more than six restarts are phase6's case.
        r.gcp.fail_next["vertex-pending"] = 1
        r.gcp.fail_next["vertex-preempt"] = 1
        (out, st), _ = r.workflow("retrain", {"mode": "smoke"})
        srid = (out or {}).get("run_id") if isinstance(out, dict) else None
        vj = c.read(f"runs/{srid}/train/vertex.json") or {}
        att = [(a.get("rung"), a.get("state")) for a in vj.get("attempts") or []]
        c.ok(st == "SUCCEEDED" and ("h100-spot-usc1", "capacity") in att and vj.get("rung") == "h100-spot-use4"
             and not any(s in ("unknown", "fatal") for _, s in att),
             f"never-started Spot → capacity, the next rung; a preemption absorbed inside the job there ({att})")
        c.ok(not w.s.get("vertex_overlap"), f"no Vertex job started while another of the round was live ({w.s.get('vertex_overlap')})")

        # e. a FULL round that fails blocks automatic rounds
        r.gcp.fail_next["prepare"] = 2
        (out, st), _ = r.workflow("retrain", {"mode": "full", "trigger": {"drift_run": None, "reason": "e2e"}})
        fr = c.read("state/failed-round.json") or {}
        c.ok(st == "FAILED" and fr.get("stage") == "prepare" and "resume_from" in str(fr.get("resume")),
             f"a failed full round leaves state/failed-round.json with its resume command ({fr.get('stage')})")
        c.ok(not (w.bucket / "state/locks/retrain.json").exists(), "the retrain lock released")
        (w.bucket / "state/failed-round.json").unlink(missing_ok=True)
    return c.done()


def ttl_max() -> int:
    """variables.tf's upper bound for decision_ttl_days (the validation's `<= N`)."""
    t = (HERE.parent / "variables.tf").read_text()
    m = re.search(r'variable "decision_ttl_days"[\s\S]*?decision_ttl_days\s*<=\s*(\d+)', t)
    return int(m.group(1)) if m else 21


def vertex_jobs(w: World, rid: str | None) -> list[dict]:
    return [v for v in (w.s.get("vertex") or {}).values() if rid and v.get("_run") == rid]


def phase6(w: World) -> int:
    """The real APIs as the 2026-10-12 audit found them (scratchpad/audit/PLAN.md), on the runs int makes next:
    a. what phases 1-5 already did against the realistic fakes (no workflow run): nobody waited on a Cloud Run
       operation (403 for the workflow account), every job ran the resolved lab image, no Vertex job overlapped
       another of its round, every model build ran on a worker that can build it;
    b. a drift poll while the int service has a --no-traffic preview revision (latestReady ≠ serving), a digest
       whose tags moved on, two polls in the same minute, one-off 500 / 408 answers;
    c. the thermometer key read from a secret version that ends in a newline;
    d. drift checks: a gpu-eval lock refreshed by its holder between our read and our delete (412), a job PATCH
       racing another update (etag 409), the eval service's newest revision failed (no L4) so the next identical
       PATCH must force a new revision;
    e. a SMOKE round under one-off faults (jobs:run 500, a jobs:run applied but answered 503, an upload 408, a Vertex
       create applied but answered 503)
       whose training then fails (Vertex GETs 500): no orphaned Vertex job;
    f. resume_from train of it with the thermometer's 404 handed over as raw text, a Spot job whose cancel loses
       to capacity: approved;
    g. a SMOKE round nobody decides, with the decision TTL at variables.tf's maximum: under 100,000 steps;
    h. a FULL round: Spot preempted 7× (STOCKOUT), a build that times out after its push, production traffic
       pinned to the old revision on approval;
    i. a 4,000-message outbox (pagination, the 512 KB variable limit);
    j. a launcher manifest with 600 new fields (a large drift notice under the 512 KB limit).
    E2E_PHASE6=a,b,… runs a subset."""
    from souschef_lab.manifest import Manifest
    c = Checks(w, "phase6")
    only = {x.strip() for x in os.environ.get("E2E_PHASE6", "").split(",") if x.strip()}

    def want(k: str) -> bool:
        if only and k not in only:
            return False
        log(f"  ── phase6 {k}")
        return True

    # a. phases 1-5 against the realistic fakes
    if want("a"):
        c.ok(not w.s.get("ops_denied"), f"no workflow GETs a Cloud Run operation (403 for the workflow account: "
                                       f"{w.s.get('ops_denied', 0)} refused)")
        c.ok(not w.s.get("image_violations"), f"every job ran the resolved lab image ({(w.s.get('image_violations') or [])[:2]})")
        c.ok(not w.s.get("vertex_overlap"), f"no Vertex job started while another of its round was live ({(w.s.get('vertex_overlap') or [])[:2]})")
        bad_builds = [b.get("statusDetail") for b in (w.s.get("builds") or {}).values()
                      if b.get("status") == "FAILURE" and re.search(r"build.context|client version|docker", str(b.get("statusDetail")), re.I)]
        c.ok(not bad_builds, f"every model build ran on a worker and client that can build it ({bad_builds[:1]})")
        unsettled = [n for n, s in w.s["services"].items() if s.get("latestReadyRevision") != s.get("latestCreatedRevision")]
        c.ok(not unsettled, f"every GPU service's newest revision is the ready one ({unsettled})")

    if want("b"):
        with Run(w) as r:
            r.clock.advance(3600)
            snap = copy.deepcopy({k: w.s[k] for k in ("services", "revisions")})
            # a digest whose tags have all moved on: the poll must fail loudly as untagged, not on a missing key
            s = w.s["services"]["sous-chef"]
            s["template"]["containers"][0]["image"] = f"{LAB_REPO}/sous-chef@{UNTAGGED_DIGEST}"
            rev = r.gcp.add_revision("sous-chef", s["template"])
            s["latestCreatedRevision"] = s["latestReadyRevision"] = rev
            s["trafficStatuses"] = r.gcp.traffic_statuses(s)
            out, st = r.drift("poll")
            c.ok(st == "FAILED" and isinstance(out, dict) and out.get("error_class") == "untagged_service_image",
                 f"the int service on an untagged digest: the poll fails as untagged_service_image ({st} "
                 f"{(out or {}).get('error_class') if isinstance(out, dict) else out})")
            w.s["services"], w.s["revisions"] = copy.deepcopy(snap["services"]), copy.deepcopy(snap["revisions"])

            # a --no-traffic preview of another build: latestReady = latestCreated = the preview, serving = the old one
            w.s["tags"].update({f"{NE2}/sous-chef/sous-chef:{PREVIEW_TAG}": "sha256:" + "6" * 64,
                                f"{NE2}/sous-chef/sous-chef-lab:{PREVIEW_TAG}": "sha256:" + "7" * 64,
                                f"{NE2}/sous-chef/sous-chef-trainer:{PREVIEW_TAG}": "sha256:" + "8" * 64})
            r.gcp.deploy_preview("sous-chef", f"{LAB_REPO}/sous-chef:{PREVIEW_TAG}", PREVIEW_TAG)
            # one-off transient answers Google documents as retryable (https://docs.cloud.google.com/storage/docs/retry-strategy)
            r.gcp.inject("GET", r"/o/infra%2Fconfig\.json$", 500)
            r.gcp.inject("POST", r"storage\.googleapis\.com/upload/", 408)
            r.gcp.inject("GET", r"run\.googleapis\.com/v2/projects/[^/]+/locations/[^/]+/services/sous-chef$", 500)
            r.gcp.inject("GET", r"artifactregistry\.googleapis\.com/.*/tags/", 502)
            nv = len(w.s["image_violations"])
            out1, st1 = r.drift("poll")
            c.ok(st1 == "SUCCEEDED", f"a poll during an int preview deploy, under one-off 500/408/502 answers, succeeds "
                                     f"({st1} {json.dumps(out1, default=str)[:200]})")
            c.ok(len(w.s["image_violations"]) == nv and r.gcp.lab_image_ok(w.s["jobs"]["souschef-drift"]["template"]["template"]["containers"][0]["image"]),
                 f"the drift job runs the SERVING build's lab image, not the preview's ({w.s['jobs']['souschef-drift']['template']['template']['containers'][0]['image'].rsplit('/', 1)[-1]})")
            hit = {i["re"].pattern: i["hits"] for i in r.gcp.injections}
            c.ok(hit[r"/o/infra%2Fconfig\.json$"] and hit[r"run\.googleapis\.com/v2/projects/[^/]+/locations/[^/]+/services/sous-chef$"],
                 f"(the injected 500s were met: {list(hit.values())})")
            r.gcp.injections.clear()
            r.clock.advance(5)
            out2, st2 = r.drift("poll")
            rid1 = (out1 or {}).get("run_id") if isinstance(out1, dict) else None
            rid2 = (out2 or {}).get("run_id") if isinstance(out2, dict) else None
            c.ok(st2 == "SUCCEEDED" and rid1 and rid2 and rid1 != rid2,
                 f"two polls in the same minute get different run ids ({rid1}, {rid2})")
            # update-traffic --to-latest is a person's call; here the preview is abandoned: back to the old state
            w.s["services"], w.s["revisions"] = snap["services"], snap["revisions"]

    if want("c"):
        with Run(w) as r:
            r.clock.advance(86400)  # a new UTC day: the heartbeat is due
            r.secret_suffix = "\n"  # `echo $KEY | gcloud secrets versions add …`
            n0 = len(w.s["posted"])
            out, st = r.drift("poll")
            hb = [p for p in posted_since(w, n0) if p.get("kind") == "heartbeat"]
            c.ok(st == "SUCCEEDED" and hb and all(p["code"] in (200, 201) for p in hb),
                 f"a secret version ending in a newline still authenticates ({[p['code'] for p in hb]})")
            r.secret_suffix = ""

    if want("d"):
        with Run(w) as r:
            r.clock.advance(3600)
            now = r.clock.now()
            # a job PATCH that races someone else's update: the etag the workflow read is stale (409)
            w.s["jobs"]["souschef-drift"]["template"]["template"]["containers"][0]["image"] = f"{LAB_REPO}/sous-chef-lab:{LABONLY_TAG}"
            r.gcp.fail_next["run-etag-race:souschef-drift"] = 1
            out, st = r.drift("poll")
            img = w.s["jobs"]["souschef-drift"]["template"]["template"]["containers"][0]["image"]
            c.ok(st == "SUCCEEDED" and r.gcp.lab_image_ok(img),
                 f"a job re-pointed while someone else updated it (409 on the stale etag) still runs the lab ({st}, {img.rsplit('/', 1)[-1]})")
            r.clock.advance(1800)
            lock = w.bucket / "state" / "locks" / "gpu-eval.json"
            lock.parent.mkdir(parents=True, exist_ok=True)
            other = {"holder": "0f0f0f0f-other-execution", "run_id": "dr-20261012-0000-check",
                     "expires_at": now_iso(now - 600), "updated_at": now_iso(now - 7200)}
            lock.write_text(json.dumps(other))
            r.gcp.race("state/locks/gpu-eval.json",
                       lambda d: {**d, "expires_at": now_iso(r.clock.now() + 7200), "updated_at": now_iso(r.clock.now())})
            out0, st0 = r.drift("check")
            held = json.loads(lock.read_text()) if lock.is_file() else {}
            c.ok(st0 == "SUCCEEDED", f"a gpu-eval lock refreshed between our read and our delete (412) defers the check, "
                                     f"it does not fail it ({st0} {json.dumps(out0, default=str)[:200]})")
            c.ok(held.get("holder") == other["holder"], "the other execution's refreshed lock survives")
            lock.unlink(missing_ok=True)

            # the eval service's newest revision fails (no L4): ready stays on the old one, the template already
            # names the target; the next check's PATCH is then identical and must force a new revision
            ev = w.s["services"]["sous-chef-model-eval"]
            ev["template"]["containers"][0]["image"] = f"{MODEL_REPO}/model:r4"
            r.gcp.fail_next["revision-fail:sous-chef-model-eval"] = 1
            r.clock.advance(1800)
            out1, st1 = r.drift("check", force=True)
            r.clock.advance(1800)
            r.gcp.tick()  # the failed revision settles even when the workflow stopped watching
            failed = ev["latestReadyRevision"] != ev["latestCreatedRevision"]
            c.ok(failed, f"(set-up) the check's eval revision failed: ready {short_rev(ev['latestReadyRevision'])}, "
                         f"created {short_rev(ev['latestCreatedRevision'])} ({st1})")
            r.clock.advance(1800)
            n_patch = len(w.s.get("patches") or [])
            out2, st2 = r.drift("check", force=True)
            serving = r.gcp.serving("sous-chef-model-eval")
            model = c.read("state/model.json") or {}
            rimg = w.s["revisions"].get(serving[0][0], {}).get("containers", [{}])[0].get("image", "") if serving else ""
            c.ok(st2 == "SUCCEEDED" and isinstance(out2, dict) and out2.get("verdict") is not None,
                 f"the next check gets a new revision and judges ({st2} {(out2 or {}).get('verdict') if isinstance(out2, dict) else out2})")
            c.ok(ev["latestReadyRevision"] == ev["latestCreatedRevision"] and len(w.s.get("patches") or []) > n_patch
                 and rimg.endswith("@" + str(model.get("digest"))),
                 f"the eval service serves the production digest from a new revision ({short_rev(ev['latestReadyRevision'])})")
            c.ok(not (w.bucket / "state/locks/gpu-eval.json").exists(), "gpu-eval lock released")

    rid_e = w.s.get("phase6_e_rid")
    if want("e"):
        with Run(w) as r:
            r.clock.advance(3600)
            r.gcp.inject("POST", r"/jobs/souschef-prepare:run$", 500)                       # not applied
            r.gcp.inject("POST", r"/jobs/souschef-dataset:run$", 503, after=True)            # applied, answer lost
            ds0 = sum(1 for x in w.s["executions"].values() if x["job"] == "souschef-dataset")
            r.gcp.inject("POST", r"storage\.googleapis\.com/upload/.*", 408)                # not applied
            r.gcp.inject("POST", r"aiplatform\.googleapis\.com/.*/customJobs$", 503, after=True)  # applied, answer lost
            r.gcp.inject("GET", r"aiplatform\.googleapis\.com/.*/customJobs/\d+$", 500, count=60)  # then training polls fail
            (out, st), _ = r.workflow("retrain", {"mode": "smoke"})
            rid_e = (out or {}).get("run_id") if isinstance(out, dict) else None
            if not rid_e:
                rid_e = next((x.get("_run") for x in reversed(list((w.s.get("vertex") or {}).values()))), None)
            w.s["phase6_e_rid"] = rid_e
            hits = [(i["re"].pattern[-30:], i["hits"]) for i in r.gcp.injections]
            r.gcp.injections.clear()
            c.ok(st == "FAILED" and isinstance(out, dict) and out.get("pipeline_stage") == "train",
                 f"one-off faults before training are absorbed; failing Vertex polls fail the round at train "
                 f"({st} {(out or {}).get('pipeline_stage') if isinstance(out, dict) else out}; faults met {hits})")
            r.gcp.tick()
            live = [v["name"].rsplit("/", 1)[1] + ":" + v["state"] for v in vertex_jobs(w, rid_e)
                    if v["state"] not in VERTEX_TERMINAL and v["state"] != "JOB_STATE_CANCELLING"]
            c.ok(rid_e and not live, f"the failed round left no live Vertex job behind ({live})")
            ov = [o for o in w.s["vertex_overlap"] if w.s["vertex"].get(o["new"], {}).get("_run") == rid_e]
            c.ok(rid_e and not ov, f"a create applied but answered 503 did not start a second job ({ov[:2]})")
            ds = sum(1 for x in w.s["executions"].values() if x["job"] == "souschef-dataset") - ds0
            c.ok(ds == 1, f"a jobs:run applied but answered 503 started one execution, not two ({ds})")
            c.ok(not (w.bucket / "state/locks/retrain.json").exists(), "the retrain lock released")

    if want("f"):
        with Run(w, decide=[{"action": "approve", "note": "resume"}]) as r:
            r.clock.advance(3600)
            r.gcp.raw_error_bodies = True
            r.gcp.fail_next["vertex-pending"] = 1        # the first Spot rung waits for capacity …
            r.gcp.fail_next["vertex-late-success"] = 1   # … which arrives just as the workflow cancels it
            ov0 = len(w.s["vertex_overlap"])
            before = set(w.s["vertex"])
            (out, st), _ = r.workflow("retrain", {"resume_from": "train", "run_id": rid_e or "rt-missing"})
            r.gcp.raw_error_bodies = False
            c.ok(st == "SUCCEEDED" and isinstance(out, dict) and out.get("outcome") == "approved",
                 f"resume_from train with the decision API's 404 as raw text: not refused, round approved "
                 f"({st} {json.dumps(out, default=str)[:240]})")
            c.ok(len(w.s["vertex_overlap"]) == ov0,
                 f"a cancelled job is seen to its end before the next one starts ({w.s['vertex_overlap'][ov0:]})")
            r.gcp.tick()
            done = [v["name"].rsplit("/", 1)[1] for v in vertex_jobs(w, rid_e) if v["name"] not in before
                    and v["state"] == "JOB_STATE_SUCCEEDED"]
            c.ok(len(done) == 1, f"exactly one job trained the round (the late success is used, not raced) ({done})")

    if want("g"):
        ttl = ttl_max()
        with Run(w, config={"smoke_ttl_days": ttl}) as r:
            r.clock.advance(3600)
            (out, st), interp = r.workflow("retrain", {"mode": "smoke"})
            c.ok(st == "SUCCEEDED", f"a round nobody decides for {ttl} days (variables.tf's maximum) ends by itself "
                                    f"({st} {json.dumps(out, default=str)[:200]})")
            c.ok(interp.steps < 90_000, f"{interp.steps} steps: under 90,000 (Workflows stops an execution at 100,000)")
            c.ok(not (w.bucket / "state/locks/retrain.json").exists(), "the retrain lock released")
            log(f"      decision wait: {interp.steps} steps, peak {interp.peak_memory} bytes of variables")

    if want("h"):
        with Run(w, decide=[{"action": "approve", "note": "pinned traffic"}]) as r:
            r.clock.advance(3600)
            prod = w.s["services"]["sous-chef-model"]
            pinned_rev = short_rev(prod["latestReadyRevision"])
            prod["traffic"] = [{"type": "TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION", "revision": pinned_rev, "percent": 100}]
            prod["trafficStatuses"] = r.gcp.traffic_statuses(prod)
            model_before = (w.bucket / "state/model.json").read_text()
            r.gcp.fail_next["vertex-preempt"] = VERTEX_MAX_RESTARTS + 1
            r.gcp.fail_next["build-timeout-after-push"] = 1
            n0 = len(w.s["posted"])
            (out, st), _ = r.workflow("retrain", {"mode": "full", "trigger": {"drift_run": None, "reason": "e2e phase6"}})
            rid = (out or {}).get("run_id") if isinstance(out, dict) else None
            if not rid:  # a refused rollout fails the execution: its error carries no run id
                rid = next((x.get("_run") for x in reversed(list((w.s.get("vertex") or {}).values()))), None)
            vj = c.read(f"runs/{rid}/train/vertex.json") or {}
            att = [(a.get("rung"), a.get("state")) for a in vj.get("attempts") or []]
            c.ok(("h100-flex", "capacity") in att, f"a Flex-start that never started is capacity, whatever its message ({att})")
            c.ok(att and not any(s in ("unknown", "fatal") for _, s in att) and vj.get("rung"),
                 f"a Spot job preempted past Vertex's six restarts (STOCKOUT) is retried, not counted unknown ({att})")
            img = c.read(f"runs/{rid}/model/image.json") or {}
            tag_d = w.s["tags"].get(f"{GPU}/sous-chef-model/model:c-{rid}")
            c.ok(img.get("digest") and img.get("digest") == tag_d,
                 f"a build that timed out after its push is taken from the pushed tag, not rebuilt into it ({img.get('digest')}, {tag_d})")
            kinds = [p["kind"] for p in posted_since(w, n0)]
            serving = r.gcp.serving("sous-chef-model")
            srv_img = w.s["revisions"].get(serving[0][0], {}).get("containers", [{}])[0].get("image", "") if serving else ""
            c.ok("rolled_out" not in kinds and st == "FAILED" and isinstance(out, dict) and out.get("error_class") == "traffic_pinned",
                 f"with traffic pinned, the rollout is refused as traffic_pinned ({st} {json.dumps(out, default=str)[:200]})")
            c.ok((w.bucket / "state/model.json").read_text() == model_before and len(serving) == 1
                 and srv_img.endswith(str((json.loads(model_before) or {}).get("digest"))),
                 f"production + state/model.json stay as they were ({[(short_rev(n), p) for n, p in serving]})")
            c.ok("stage_failed" in kinds, f"and somebody is told ({kinds})")
            prod["traffic"] = copy.deepcopy(LATEST_100)
            prod["trafficStatuses"] = r.gcp.traffic_statuses(prod)
            c.ok(not (w.bucket / "state/locks/retrain.json").exists(), "the retrain lock released")
            (w.bucket / "state/failed-round.json").unlink(missing_ok=True)

    if want("i"):
        with Run(w) as r:
            r.clock.advance(900)
            box = w.bucket / "wf" / "outbox"
            box.mkdir(parents=True, exist_ok=True)
            t0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
            seeded = []
            for k in range(4000):
                t = t0 + datetime.timedelta(minutes=k)
                rid = f"dr-{t:%Y%m%d-%H%M}-poll"
                payload = {"id": f"{rid}/heartbeat", "kind": "heartbeat", "severity": "info", "run_id": rid,
                           "candidate_id": None, "title": "Sous Chef drift check: heartbeat",
                           "summary": f"souschef-drift polled the deployed manifest ({rid})", "dedup_key": None,
                           "details": {"day": f"{t:%Y-%m-%d}"}, "created_at": now_iso(t.timestamp())}
                f = box / (payload["id"].replace("/", "~") + ".json")
                f.write_text(json.dumps(payload))
                seeded.append(f)
            out, st = r.drift("poll")
            left = len(list(box.glob("*.json")))
            last = (w.s.get("executions_log") or [{}])[-1]
            c.ok(st == "SUCCEEDED", f"a poll over a 4,000-message outbox succeeds ({st} {json.dumps(out, default=str)[:160]}; "
                                    f"peak {last.get('peak_memory')} bytes of variables)")
            c.ok(3950 - 60 <= left < 4000, f"it drains a bounded batch ({4000 - left} sent or dropped)")
            for f in seeded:  # 4,000 inodes on a shared tmpfs
                f.unlink(missing_ok=True)

    if want("j"):
        ms = manifests()
        big = copy.deepcopy(ms["M2"])
        panel = big["globalPanels"]["ButtonsSection"]["fields"]
        for k in range(600):
            panel[f"e2eSetting{k:03d}"] = {"type": "text", "label": f"E2E Setting {k:03d}", "path": f"globalSettings.e2eSetting{k:03d}",
                                           "help": f"An e2e setting number {k}, there to make the drift notice large."}
        big["generated_from"] = {**ms["M2"]["generated_from"], "build_id": "4d5e6f7"}
        with Run(w) as r:
            r.clock.advance(3600)
            r.svc.register(big, "4d5e6f7")
            n0 = len(w.s["posted"])
            out, st = r.drift("poll")
            kinds = [(p["kind"], p["code"]) for p in posted_since(w, n0)]
            last = (w.s.get("executions_log") or [{}])[-1]
            c.ok(st == "SUCCEEDED" and any(k == "drift_notice" for k, _ in kinds),
                 f"600 new fields: the poll posts its notice within Workflows' 512 KB ({st}, {kinds}, peak "
                 f"{last.get('peak_memory')} bytes)")
            r.svc.register(ms["M2"], "5e6f7a8")  # the launcher rolls back
            r.started_retrains()
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
    ap.add_argument("what", choices=("setup", "phase1", "phase2", "phase3", "weekly", "phase4", "phase5", "phase6", "report"))
    a = ap.parse_args()
    w = World(Path(a.work).resolve())
    if a.what == "setup":
        if not a.bundle or not a.snapshot:
            ap.error("setup needs --bundle and --snapshot")
        setup(w, Path(a.bundle), Path(a.snapshot).resolve())
        return 0
    return {"phase1": phase1, "phase2": phase2, "phase3": phase3, "weekly": weekly, "phase4": phase4, "phase5": phase5,
            "phase6": phase6, "report": report}[a.what](w)


if __name__ == "__main__":
    sys.exit(main())
