#!/usr/bin/env python3
"""mxarm — the Mistral arm's one-container runners and job bodies (docs/MISTRAL.md).

    python tools/mxarm.py base-gguf  --run MX [--arms mx-base,qwen-base] [--converter-ref b11081]   trainer-mx image, CPU
    python tools/mxarm.py exam       --run MX --arm mx-base|qwen-base|r4|mx-base-native [--replicates 2]  sous-chef-mx, 1 x L4
    python tools/mxarm.py eval       --run MX --cases-run RT --pair PAIR [--mode full|smoke]        sous-chef-mx, 2 x L4
    python tools/mxarm.py gate       --run MX                                                      CPU, anywhere
    python tools/mxarm.py report     --run MX [--jobs-json jobs.json]                              CPU, anywhere
    python tools/mxarm.py latency    --run MX                                                      CPU, anywhere (prints)
    python tools/mxarm.py jobs       --run MX --stage base-gguf|exam|train|eval …                  prints bodies, never submits
    python tools/mxarm.py gates      --run MX --stage prep|1|2|3 [--rung R]                         read-only pre-submit checks
    python tools/mxarm.py prep-run   --run MX --cases-run RT [--from-run MX0]                       dataset copy + train-config
    python tools/mxarm.py selftest   [--trainer] [--no-gpu]                                        the image's tools
    python tools/mxarm.py snapshot-r4                                                              r4's state → snapshots/r4/
    python tools/mxarm.py user-go    --run MX --by WHO --note WHY                                  the user's GO on a JUDGE
    python tools/mxarm.py stage-native --run MX                                                    mx-base-native diagnostic

Rounds. `--cases-run` names the retrain round whose dataset MX trains on and whose cases, candidate and eval it is
compared on (ROUNDS: r6 = rt-20261016-2102-…, r7 = rt-20261018-0213-…; a command without it takes the run's
plan.json, else DEFAULT_CASES_RUN). The round's candidate is "<label>c" (r6c, r7c). Pairs (current-candidate):
r4-mx, <label>c-mx and the CONTROL <round's current>-<label>c, which must reproduce the round's own eval/bar.json
before any MX verdict is trusted (r6's round: r4-r6c; r7's: r6c-r7c, r6 being live when r7 evaluated). r4-mxt
(MX served at Mistral's temperature, ≤ 0.15) is report only, never part of a verdict.

Where things go. Every object this tool writes is under EXP_ROOT = gs://breadstack-sous-chef-int-pipeline/
experiments/mistral/ and goes through `guard()`, which refuses any other destination (runs/, base/, state/, another
experiment). Everything else in the bucket is only read: the rounds' run folders (prepare/, train/stage/, eval/),
state/, baselines/, manifests/, drift/, engine/ and base/unsloth-qwen3.5-9b-hf.tar. No lock is taken, no workflow or
service is called, nothing is submitted: `jobs` prints the Vertex customJobs bodies a person (or the main session)
submits after `gates` passes.

Serving (docs/MISTRAL.md §(f)). exam and eval run Ollama 0.34.4 (the model image's binary, in the sous-chef-mx
image) one model per server, `ollama create sous-chef -f Modelfile` from the staged GGUF and Modelfile, then assert
before any row runs: the GGUF passes mx_gguf.check; /api/show has the Modelfile's PARSER/RENDERER, the TEMPLATE's
sha256, num_ctx 65536, the tools capability and the family's architecture; the model loads and sits 100 % in VRAM
(/api/ps size_vram == size); the first request on the empty cache, serving_mistral.SMOKE_REQUEST on /v1/messages,
reads usage.input_tokens == SMOKE_PROMPT_TOKENS[family] (the template renders as the lab's render_prompt does, with
ONE BOS and every control marker one token — Ollama never logs the rendered prompt); a short /api/chat decodes. A failure there is BLOCKED (exit 3, exam/<arm>/failed.json),
never a score. The Sous Chef service (the lab image's 28491da binaries, so the deployed prompt) runs in
souschef_lab.harness with auth "none" and prime "off" on every arm.

Attempts. exam, eval and base-gguf write only into <dest>/attempts/<create>/ (MX_CREATE, the body's create label)
and then point <dest>/latest.json at it; a container that finds its attempt already started is a Vertex restart and
exits 1 writing nothing. The bodies also set scheduling.disableRetries.

Exit codes: 0 done, 1 failed (or a refused restart), 2 configuration / precondition, 3 blocked (the model cannot be
served).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib import error as urlerror
from urllib import request as urlrequest

TOOLS_DIR = Path(__file__).resolve().parent
LAB_DIR = TOOLS_DIR.parent
for _p in (str(LAB_DIR), str(TOOLS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mx_gguf  # noqa: E402  (tools/mx_gguf.py, stdlib only)

# --- constants ----------------------------------------------------------------------------------------------------------

PROJECT = "breadstack-sous-chef-int"
BUCKET = "gs://breadstack-sous-chef-int-pipeline"
EXP_ROOT = BUCKET + "/experiments/mistral"
REGION = "us-central1"
AR = "northamerica-northeast2-docker.pkg.dev/breadstack-sous-chef-int/sous-chef"
MX_SA = f"souschef-mx@{PROJECT}.iam.gserviceaccount.com"
DEFAULT_RUN = "mx-20261018-m3-8b-r7"
R6_ARM_RUN = "mx-20261017-m3-8b"  # the r6 arm (Stage 1 GO, models/r4): what an r7 run carries over (prep-run --from-run)
R6 = "rt-20261016-2102-3eeb97-eb908674"
R7 = "rt-20261018-0213-b02e95-3914e2e4"
DEFAULT_CASES_RUN = R7
# The retrain rounds the arm compares against: `label` names the round's candidate in everything people read
# ("<label>c": r6c, r7c); `current` is the model that was CURRENT in that round's own eval — the control pair's
# current (r6's round ran against r4; r7's against r6c, live since its approval on 2026-10-17).
ROUNDS = {R6: {"label": "r6", "current": "r4"},
          R7: {"label": "r7", "current": "r6c"}}
OLLAMA_VERSION = "0.34.4"
MODEL_NAME = "sous-chef"  # harness.ROUTE_DEFAULTS["model"]
MISTRAL_REPO = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
MISTRAL_REV = "f6fae9795746f63c9be8344932f01275f3c63734"
MISTRAL_TAR = "mistralai-ministral-3-8b-instruct-2512-bf16-hf.tar"
QWEN_REPO = "unsloth/Qwen3.5-9B"
QWEN_TAR = "unsloth-qwen3.5-9b-hf.tar"
BASES = {  # base-gguf arms: the untrained bases, converted by the trainer image's pinned llama.cpp
    "mx-base": {"family": "mistral", "repo": MISTRAL_REPO, "revision": MISTRAL_REV, "tar": f"{EXP_ROOT}/base/{MISTRAL_TAR}",
                "modelfile": "base-modelfiles/ministral-3-8b.modelfile", "tag": "ministral-3:8b"},
    "qwen-base": {"family": "qwen", "repo": QWEN_REPO, "revision": None, "tar": f"{BUCKET}/base/{QWEN_TAR}",
                  "modelfile": "base-modelfiles/qwen3.5-9b.modelfile", "tag": "qwen3.5:9b"},
}
STAGE1_ARMS = ("mx-base", "qwen-base", "r4")
GATE_ARMS = ("mx-base", "qwen-base")
NATIVE_ARM = "mx-base-native"  # the optional diagnostic (spec §(f)): mx-base's GGUF, no TEMPLATE/PARSER
EXAM_ARMS = STAGE1_ARMS + (NATIVE_ARM,)
TWO_REPLICATE_ARMS = GATE_ARMS + (NATIVE_ARM,)
# r4, the live model when the arm was built (state/model.json on 2026-10-16). Stage 3's r4 pairs take r4's baseline
# and training manifest from a snapshot of that state (snapshot-r4), never from a state/model.json r6 may have moved.
R4_DIGEST = "sha256:f657e11053e3baf801b489421637b8a0aaccf450951046ec4c9853ca0c7287b4"
R4_MANIFEST = "bf78870182302f821856057d0091d62b09403c5180716c4f37a4830761e18a12"
MX_MODELS = ("mx", "mxt")  # mxt: the same staged MX model, its route at MXT_TEMPERATURE (report only)
MXT_TEMPERATURE = 0.15  # Mistral's recommended sampling for Ministral 3 (≤ 0.15); every other arm runs at 0.3
REPORT_ONLY_PAIRS = {"r4-mxt": ("r4", "mxt")}  # never part of a verdict, never needed for one
EXAM_SETS = ("paraphrase", "holdout", "real", "followup", "out_of_scope", "out_of_scope_holdout")  # eval_stage's
EXAM_N = 252
PARSER_FAMILY = {"ministral": "mistral", "qwen3.5": "qwen"}
QUANT = "q4_k_m"
STAGE_TEMPERATURE, STAGE_PRESENCE = "0.3", "0"
BLOCKING = ("gguf", "create", "show", "load", "gpu", "render", "decode")
# Stage 1 gate (rows of the 252; scaled when fewer rows are common to both arms)
GO_ROWS, STOP_ROWS, STOP_FMT_ROWS = 13, 38, 25
SEED2_ROWS = 4  # Stage 3: a failing set short of its tolerance by at most this many rows → seed 2 first
P_GREY = (0.05, 0.2)
# Vertex machines (compute.md §5, us-central1). USD/h are estimates (static Vertex table + CE Spot + Vertex fee),
# for the report's cost line only — the billing report is the truth.
MACHINES = {
    "base-gguf": {"machineType": "n2-highmem-16", "disk": 300, "timeout": 10800, "usd_h": 1.0},
    "exam": {"machineType": "g2-standard-12", "acceleratorType": "NVIDIA_L4", "acceleratorCount": 1, "disk": 200,
             "timeout": 21600, "usd_h": 1.4},
    "eval": {"machineType": "g2-standard-24", "acceleratorType": "NVIDIA_L4", "acceleratorCount": 2, "disk": 200,
             "timeout": 28800, "usd_h": 2.9},
}
RUNGS = {  # Stage 2 machines; only a100-40-spot may run while the round trains (docs/MISTRAL.md §Submission gates)
    "a100-40-spot": {"machineType": "a2-highgpu-1g", "acceleratorType": "NVIDIA_TESLA_A100", "strategy": "SPOT", "usd_h": 1.55},
    "a100-40-std": {"machineType": "a2-highgpu-1g", "acceleratorType": "NVIDIA_TESLA_A100", "strategy": "STANDARD", "usd_h": 4.425,
                    "after_round_train": True},
    "h100-spot": {"machineType": "a3-highgpu-1g", "acceleratorType": "NVIDIA_H100_80GB", "strategy": "SPOT", "usd_h": 8.0,
                  "after_round_train": True},
    "h100-flex": {"machineType": "a3-highgpu-1g", "acceleratorType": "NVIDIA_H100_80GB", "strategy": "FLEX_START", "usd_h": 8.0,
                  "after_round_train": True},
}
RUN_RE = re.compile(r"mx-[a-z0-9][a-z0-9-]{0,60}")
CASES_RUN_RE = re.compile(r"rt-[0-9a-f-]{8,60}")
DIGEST_RE = re.compile(r"[a-z0-9.-]+-docker\.pkg\.dev/[^\s@]+@sha256:[0-9a-f]{64}")


class MxError(RuntimeError):
    code = 1


class ConfigError(MxError):
    code = 2


class Blocked(MxError):
    """The model cannot be served (exit 3): reason one of BLOCKING."""
    code = 3

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


def say(msg: str) -> None:
    print(f"mxarm {datetime.now(timezone.utc):%H:%M:%S} {msg}", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def check_run(run: str) -> str:
    if not RUN_RE.fullmatch(str(run or "")):
        raise ConfigError(f"run id {run!r}: mx-<lowercase letters, digits, dashes> (one path segment)")
    return run


def check_cases_run(r: str) -> str:
    if not CASES_RUN_RE.fullmatch(str(r or "")):
        raise ConfigError(f"cases run {r!r}: a retrain round id rt-…")
    return r


def run_root(run: str) -> str:
    return f"{EXP_ROOT}/runs/{check_run(run)}"


def round_info(cases_run: str) -> dict:
    """ROUNDS[cases_run] + {"run", "cc"} (cc = "<label>c", the round's candidate as people read it)."""
    r = check_cases_run(cases_run)
    if r not in ROUNDS:
        raise ConfigError(f"cases run {r}: not a round the arm knows ({', '.join(ROUNDS)}); add it to ROUNDS with its "
                          "label and the model that was current in its eval")
    return dict(ROUNDS[r], run=r, cc=ROUNDS[r]["label"] + "c")


def round_of_model(name: str) -> str | None:
    """The round id whose candidate `name` ("r6c", "r7c") is; None for any other model."""
    return next((r for r, i in ROUNDS.items() if i["label"] + "c" == name), None)


def pairs_for(cases_run: str, *, report_only: bool = True) -> dict[str, tuple[str, str]]:
    """{pair: (current, candidate)} of a round: r4-mx, <cc>-mx, the control <current>-<cc> (+ the report-only)."""
    ri = round_info(cases_run)
    cc = ri["cc"]
    out = {"r4-mx": ("r4", "mx"), f"{cc}-mx": (cc, "mx"), control_pair(cases_run): (ri["current"], cc)}
    if report_only:
        out.update(REPORT_ONLY_PAIRS)
    return out


def control_pair(cases_run: str) -> str:
    """The pair that re-runs the round's own eval (its current vs its candidate) here: r4-r6c, r6c-r7c."""
    ri = round_info(cases_run)
    return f"{ri['current']}-{ri['cc']}"


def main_pairs(cases_run: str) -> list[str]:
    """The pairs the verdicts are about: <cc>-mx first, then r4-mx."""
    return [f"{round_info(cases_run)['cc']}-mx", "r4-mx"]


def model_uri(name: str, run: str, cases_run: str | None = None) -> str:
    """Where a served model's staged folder (one GGUF + its Modelfile) lives. A round's candidate ("r6c", "r7c")
    is read from that round's train/stage/ whatever the cases run; mx and mxt are this run's trained model."""
    if name in EXAM_ARMS:
        return f"{run_root(run)}/models/{name}"
    if name in MX_MODELS:
        return f"{run_root(run)}/train/stage"
    rr = round_of_model(name)
    if rr:
        return f"{BUCKET}/runs/{rr}/train/stage"
    raise ConfigError(f"unknown model {name!r}")


def train_done_uri(name: str, run: str) -> str | None:
    """The train/DONE marker a served model needs (None: a staged model with no training here, r4)."""
    if name in MX_MODELS:
        return f"{run_root(run)}/train/DONE"
    rr = round_of_model(name)
    return f"{BUCKET}/runs/{rr}/train/DONE" if rr else None


# --- the bucket: reads anywhere, writes only under EXP_ROOT --------------------------------------------------------------

def guard(dest: str) -> str:
    """`dest` when it is an object or prefix strictly under EXP_ROOT; else ConfigError. No `.`/`..`/empty segment."""
    d = str(dest or "")
    if not d.startswith("gs://"):
        raise ConfigError(f"refusing to write {d!r}: not a gs:// URI under {EXP_ROOT}/")
    body = d[5:].rstrip("/")
    segs = body.split("/")
    if any(s in ("", ".", "..") for s in segs) or "\\" in body:
        raise ConfigError(f"refusing to write {d!r}: an empty, '.' or '..' path segment")
    norm = "gs://" + body
    if not norm.startswith(EXP_ROOT + "/"):
        raise ConfigError(f"refusing to write {d!r}: only {EXP_ROOT}/… may be written by the Mistral arm")
    return norm


def _run(argv: list[str], *, input_: bytes | None = None, timeout: float | None = None,
         env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, input=input_, capture_output=True, timeout=timeout, env=env, check=False)


_NOT_FOUND = re.compile(r"matched no objects|No URLs matched|not found|404", re.IGNORECASE)


class GS:
    """gs:// through `gcloud storage`; with MX_GS_LOCAL=<dir> (tests, a rehearsal) gs://<bucket>/<key> is
    <dir>/<bucket>/<key> on disk. Writes go through guard()."""

    def __init__(self, local: str | Path | None = None, runner: Callable = _run):
        loc = local if local is not None else os.environ.get("MX_GS_LOCAL")
        self.local = Path(loc) if loc else None
        self.runner = runner
        self.writes: list[str] = []

    def path(self, uri: str) -> Path:
        if not uri.startswith("gs://"):
            raise ConfigError(f"not a gs:// URI: {uri}")
        return self.local / uri[5:].rstrip("/")

    def _gcloud(self, *args: str, input_: bytes | None = None, timeout: float = 7200) -> subprocess.CompletedProcess:
        return self.runner(["gcloud", "storage", *args, "--quiet"], input_=input_, timeout=timeout)

    def exists(self, uri: str) -> bool:
        if self.local:
            return self.path(uri).exists()
        return self._gcloud("ls", uri, timeout=300).returncode == 0

    def cat(self, uri: str) -> bytes | None:
        if self.local:
            p = self.path(uri)
            return p.read_bytes() if p.is_file() else None
        r = self._gcloud("cat", uri, timeout=600)
        if r.returncode == 0:
            return r.stdout
        if _NOT_FOUND.search((r.stderr or b"").decode("utf-8", "replace")):
            return None
        raise MxError(f"gcloud storage cat {uri}: exit {r.returncode}")

    def read_json(self, uri: str, default: Any = None) -> Any:
        b = self.cat(uri)
        return default if b is None else json.loads(b.decode("utf-8"))

    def download(self, uri: str, local: Path) -> bool:
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        if self.local:
            p = self.path(uri)
            if not p.is_file():
                return False
            shutil.copyfile(p, local)
            return True
        r = self._gcloud("cp", uri, str(local))
        if r.returncode != 0:
            if _NOT_FOUND.search((r.stderr or b"").decode("utf-8", "replace")):
                return False
            raise MxError(f"gcloud storage cp {uri}: exit {r.returncode}")
        return True

    def rsync_down(self, prefix: str, local_dir: Path) -> int:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        if self.local:
            src = self.path(prefix)
            if not src.is_dir():
                return 0
            n = 0
            for p in src.rglob("*"):
                if p.is_file():
                    dst = local_dir / p.relative_to(src)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, dst)
                    n += 1
            return n
        if not self.exists(prefix.rstrip("/") + "/"):
            return 0
        r = self._gcloud("rsync", "--recursive", prefix.rstrip("/"), str(local_dir))
        if r.returncode != 0:
            raise MxError(f"gcloud storage rsync {prefix}: exit {r.returncode}: {(r.stderr or b'')[-300:]!r}")
        return sum(1 for p in local_dir.rglob("*") if p.is_file())

    def put_bytes(self, data: bytes, dest: str, content_type: str = "application/json") -> str:
        d = guard(dest)
        if self.local:
            p = self.path(d)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        else:
            r = self._gcloud("cp", "-", d, f"--content-type={content_type}", input_=data, timeout=600)
            if r.returncode != 0:
                raise MxError(f"gcloud storage cp - {d}: exit {r.returncode}")
        self.writes.append(d)
        return d

    def put_json(self, obj: Any, dest: str) -> str:
        return self.put_bytes((json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n").encode("utf-8"), dest)

    def put_file(self, local: Path, dest: str) -> str:
        d = guard(dest)
        if self.local:
            p = self.path(d)
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, p)
        else:
            r = self._gcloud("cp", str(local), d)
            if r.returncode != 0:
                raise MxError(f"gcloud storage cp {local} {d}: exit {r.returncode}")
        self.writes.append(d)
        return d

    def put_dir(self, local_dir: Path, dest: str) -> str:
        d = guard(dest)
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            return d
        if self.local:
            for p in local_dir.rglob("*"):
                if p.is_file():
                    self.put_file(p, guard(f"{d}/{p.relative_to(local_dir).as_posix()}"))
            return d
        r = self._gcloud("rsync", "--recursive", str(local_dir), d)  # never --delete-unmatched-destination-objects
        if r.returncode != 0:
            raise MxError(f"gcloud storage rsync {local_dir} {d}: exit {r.returncode}")
        self.writes.append(d + "/")
        return d

    def copy_prefix(self, src: str, dest: str) -> str:
        """Server-side copy of a prefix (read anywhere, write under EXP_ROOT)."""
        d = guard(dest)
        if self.local:
            s = self.path(src)
            for p in s.rglob("*"):
                if p.is_file():
                    t = self.path(guard(f"{d}/{p.relative_to(s).as_posix()}"))
                    t.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, t)
            return d
        r = self._gcloud("rsync", "--recursive", src.rstrip("/"), d)
        if r.returncode != 0:
            raise MxError(f"gcloud storage rsync {src} {d}: exit {r.returncode}")
        return d

    def copy_object(self, src: str, dest: str) -> str:
        """Server-side copy of one object (read anywhere, write under EXP_ROOT)."""
        d = guard(dest)
        if self.local:
            t = self.path(d)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.path(src), t)
        else:
            r = self._gcloud("cp", src, d)
            if r.returncode != 0:
                raise MxError(f"gcloud storage cp {src} {d}: exit {r.returncode}")
        self.writes.append(d)
        return d

    def list(self, prefix: str) -> list[str]:
        if self.local:
            base = self.path(prefix)
            return sorted("gs://" + p.relative_to(self.local).as_posix() for p in base.rglob("*") if p.is_file()) if base.is_dir() else []
        r = self._gcloud("ls", prefix.rstrip("/") + "/**", timeout=600)
        if r.returncode != 0:
            return []
        return [ln.strip() for ln in r.stdout.decode("utf-8").splitlines() if ln.strip().startswith("gs://") and not ln.strip().endswith(":")]


def read_store():
    """The pipeline bucket (or MX_GS_LOCAL's copy) as a retrain Store, for READS (engine, state, r6's folder)."""
    from souschef_lab.retrain.common import open_store
    loc = os.environ.get("MX_GS_LOCAL")
    return open_store(str(Path(loc) / BUCKET[5:]) if loc else BUCKET)


# --- attempts: one job, one folder, never run twice -----------------------------------------------------------------------
#
# Vertex can restart a container that exited non-zero, even after it marked the job FAILED (images/trainer/
# entrypoint.sh: one job ran three times). The bodies set scheduling.disableRetries, and every exam / eval / base-gguf
# job also guards itself: its create nonce (MX_CREATE, the body's `create` label) names an attempt folder,
# <dest>/attempts/<create>/; a container that finds its own started.json there is a restart of a job that already
# ran, and exits 1 at once, writing nothing. Each attempt uploads only into its own folder, then points
# <dest>/latest.json at it; readers (gate, report) follow latest.json (results_uri).

def attempt_id() -> str:
    a = os.environ.get("MX_CREATE") or ""
    if a:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", a):
            raise ConfigError(f"MX_CREATE {a!r}: the body's create label (lowercase letters, digits, dashes)")
        return a
    return "local-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


class Restarted(MxError):
    """This container is a restart of a job that already ran (its attempt folder has started.json)."""
    code = 1


def begin_attempt(gs: GS, dest: str, attempt: str, info: dict) -> str:
    """Claim <dest>/attempts/<attempt>/ (writes started.json); a Restarted when it was claimed before."""
    base = f"{dest}/attempts/{attempt}"
    if gs.exists(f"{base}/started.json"):
        raise Restarted(f"{base}/started.json exists: a restart of job {attempt}, which already ran — not run again "
                        "(resubmit with a new --attempt)")
    gs.put_json({"attempt": attempt, "at": utc_now(), **info}, f"{base}/started.json")
    return base


def publish_attempt(gs: GS, dest: str, attempt: str, rc: int, extra: dict | None = None) -> None:
    status = {0: "ok", 3: "blocked"}.get(rc, "failed")
    gs.put_json({"attempt": attempt, "rc": rc, "status": status, "at": utc_now(), **(extra or {})}, f"{dest}/latest.json")


def results_uri(gs: GS, dest: str, name: str) -> str:
    """<dest>/attempts/<latest>/<name> when <dest>/latest.json names an attempt, else <dest>/<name> (a folder
    written before attempts existed, or by hand)."""
    latest = gs.read_json(f"{dest}/latest.json") or {}
    a = str(latest.get("attempt") or "")
    return f"{dest}/attempts/{a}/{name}" if a else f"{dest}/{name}"


# --- Modelfiles ---------------------------------------------------------------------------------------------------------

def modelfile_value(raw: str) -> str:
    """The value of one Modelfile instruction as Ollama stores it: the text after the keyword, triple quotes off."""
    _kw, _, rest = raw.partition(" ")
    rest = rest.lstrip()
    if rest.startswith('"""') and rest.endswith('"""') and len(rest) >= 6:
        return rest[3:-3]
    if len(rest) >= 2 and rest[0] == rest[-1] == '"':
        return rest[1:-1]
    return rest


def modelfile_facts(text: str) -> dict:
    """{template, parser, renderer, parameters: {name: [values]}, from} of a Modelfile."""
    from souschef_lab.export import parse_modelfile
    out: dict[str, Any] = {"template": None, "parser": None, "renderer": None, "parameters": {}, "from": None}
    for ins in parse_modelfile(text):
        if ins.keyword == "TEMPLATE":
            out["template"] = modelfile_value(ins.raw)
        elif ins.keyword in ("PARSER", "RENDERER"):
            out[ins.keyword.lower()] = modelfile_value(ins.raw).strip()
        elif ins.keyword == "FROM":
            out["from"] = modelfile_value(ins.raw).strip()
        elif ins.keyword == "PARAMETER":
            parts = ins.raw.split(None, 2)
            if len(parts) == 3:
                out["parameters"].setdefault(parts[1], []).append(parts[2].strip())
    return out


def family_of(facts: dict) -> str:
    fam = PARSER_FAMILY.get(str(facts.get("parser") or ""))
    if not fam:
        raise ConfigError(f"Modelfile PARSER {facts.get('parser')!r}: not one of {', '.join(PARSER_FAMILY)}")
    return fam


def sha256_text(s: str | None) -> str | None:
    return None if s is None else hashlib.sha256(s.encode("utf-8")).hexdigest()


def staged_modelfile(base_text: str, gguf_name: str, base_tag: str) -> str:
    """What export.py + `make model-stage MODEL_TEMPERATURE=0.3 MODEL_PRESENCE_PENALTY=0` write for a GGUF (the
    trainer pipeline's Modelfile; base-gguf runs the real `make model-stage`, this is the reference the tests pin)."""
    from souschef_lab.export import student_modelfile
    lines = student_modelfile(base_text, gguf_name, base_tag, 65536).splitlines()
    keep = [ln for ln in lines if not re.match(r"^PARAMETER\s+(num_ctx|temperature|presence_penalty)(\s|$)", ln)]
    keep = [re.sub(r"^FROM\s.*", f"FROM ./{gguf_name}", ln) for ln in keep]
    return "\n".join(keep + [f"PARAMETER temperature {STAGE_TEMPERATURE}", f"PARAMETER presence_penalty {STAGE_PRESENCE}",
                             "PARAMETER num_ctx 65536"]) + "\n"


def model_dir_facts(d: Path) -> tuple[dict, Path]:
    mf = Path(d) / "Modelfile"
    if not mf.is_file():
        raise Blocked("gguf", f"no Modelfile in {d}")
    facts = modelfile_facts(mf.read_text(encoding="utf-8"))
    src = str(facts.get("from") or "")
    if not src.startswith("./") or not src.endswith(".gguf"):
        raise Blocked("gguf", f"Modelfile FROM {src!r}: want ./<file>.gguf beside it")
    g = Path(d) / src[2:]
    if not g.is_file():
        raise Blocked("gguf", f"{g.name} (the Modelfile's FROM) is not in {d}")
    return facts, g


def pull_model(gs: GS, uri: str, dest: Path, *, quant: str | None = QUANT, family: str | None = None) -> dict:
    """Download a staged model folder, check its GGUF for the Modelfile's family (or `family`, for a Modelfile with
    no PARSER: mx-base-native) → {dir, gguf, facts, family, check}."""
    n = gs.rsync_down(uri, dest)
    if n == 0:
        raise ConfigError(f"{uri}/ is empty or missing")
    facts, g = model_dir_facts(dest)
    fam = family if family and not facts.get("parser") else family_of(facts)
    doc = mx_gguf.check_file(g, fam, quant)
    if not doc["ok"]:
        bad = "; ".join(f"{c['name']}: {c['detail']}" for c in doc["checks"] if not c["ok"])
        raise Blocked("gguf", f"{g.name} fails the {fam} GGUF check: {bad}")
    return {"uri": uri, "dir": str(dest), "gguf": g.name, "facts": facts, "family": fam, "check": doc}


# --- Ollama (serving side, §(f)) ----------------------------------------------------------------------------------------

def http_json(url: str, body: Any = None, *, timeout: float = 60.0, method: str | None = None) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urlrequest.Request(url, data=data, method=method or ("POST" if data is not None else "GET"))
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            status = r.status
    except urlerror.HTTPError as e:
        raw, status = e.read(), e.code
    try:
        return status, json.loads(raw.decode("utf-8")) if raw else None
    except ValueError:
        return status, {"raw": raw[:500].decode("utf-8", "replace")}


class Ollama:
    """One `ollama serve` on 127.0.0.1:<port> with its own OLLAMA_MODELS, optionally pinned to one GPU."""

    def __init__(self, port: int, models_dir: Path, log_path: Path, *, gpu: str | None = None, binary: str = "ollama"):
        self.port, self.models_dir, self.log_path, self.gpu, self.binary = port, Path(models_dir), Path(log_path), gpu, binary
        self.proc: subprocess.Popen | None = None
        self._log = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def env(self) -> dict:
        e = dict(os.environ)
        e.update({"OLLAMA_HOST": f"127.0.0.1:{self.port}", "OLLAMA_MODELS": str(self.models_dir)})
        for k, v in (("OLLAMA_NO_CLOUD", "1"), ("OLLAMA_NUM_PARALLEL", "1"), ("OLLAMA_KEEP_ALIVE", "-1"),
                     ("OLLAMA_CONTEXT_LENGTH", "65536")):
            e.setdefault(k, v)
        e.pop("OLLAMA_DEBUG_LOG_REQUESTS", None)  # never write request bodies to disk
        if self.gpu is not None:
            e["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        return e

    def start(self, timeout_s: float = 120.0) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")
        self.proc = subprocess.Popen([self.binary, "serve"], env=self.env(), stdout=self._log, stderr=subprocess.STDOUT)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise Blocked("load", f"ollama serve exited {self.proc.returncode} at start (log {self.log_path})")
            try:
                st, _ = http_json(self.url + "/api/version", timeout=2)
                if st == 200:
                    return
            except (urlerror.URLError, OSError):
                pass
            time.sleep(0.5)
        raise Blocked("load", f"ollama serve did not answer in {timeout_s:.0f}s")

    def create(self, model_dir: Path, timeout_s: float = 1800) -> None:
        r = subprocess.run([self.binary, "create", MODEL_NAME, "-f", "Modelfile"], cwd=str(model_dir), env=self.env(),
                           capture_output=True, timeout=timeout_s, check=False)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or b"")[-600:].decode("utf-8", "replace")
            raise Blocked("create", f"ollama create exited {r.returncode}: {tail}")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.proc = None
        if self._log is not None:
            self._log.close()
            self._log = None


ECHO_TOOL = {"name": "echo", "description": "Echo the given text back.",
             "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}
# a call written out as text: the parser never saw it (the control tokens were stripped or read as text)
LEAK_MARKERS = {"mistral": ("[TOOL_CALLS]", "[ARGS]")}


def smoke_prompt_tokens(facts: dict, family: str) -> int | None:
    """The prompt size SMOKE_REQUEST must have when this Modelfile serves it as the lab renders: the committed TEMPLATE
    (mistral) or RENDERER qwen3.5 (qwen). None for any other template (the mx-base-native diagnostic: GGUF's own)."""
    from souschef_lab import serving_mistral as SM
    if family == "mistral" and facts.get("template") == SM.TEMPLATE:
        return SM.SMOKE_PROMPT_TOKENS["mistral"]
    if family == "qwen" and (facts.get("renderer") or None) == "qwen3.5":
        return SM.SMOKE_PROMPT_TOKENS["qwen"]
    return None


def served_parser_renderer(show: dict) -> tuple:
    """The PARSER and RENDERER a served model has. Ollama 0.34.4's /api/show has no top-level parser / renderer:
    they are only lines of its `modelfile` text (2026-10-17: production r4 shows `RENDERER qwen3.5` / `PARSER qwen3.5`
    there and nothing at the top level, and every exam blocked on "parser None"). A top-level value wins if a later
    Ollama adds one."""
    mf = modelfile_facts(str(show.get("modelfile") or ""))
    return (show.get("parser") or mf.get("parser") or None, show.get("renderer") or mf.get("renderer") or None)


def assert_served(url: str, facts: dict, family: str, *, strict_smoke: bool = False, load_timeout: float = 900) -> dict:
    """The serving asserts (module docstring). Raises Blocked; returns what was seen (served.json)."""
    out: dict[str, Any] = {"url": url, "family": family, "at": utc_now()}
    st, show = http_json(url + "/api/show", {"model": MODEL_NAME}, timeout=120)
    if st != 200 or not isinstance(show, dict):
        raise Blocked("show", f"/api/show answered {st}")
    want_t = facts.get("template")
    got_t = show.get("template")
    params = str(show.get("parameters") or "")
    arch = (show.get("model_info") or {}).get("general.architecture")
    caps = list(show.get("capabilities") or [])
    parser, renderer = served_parser_renderer(show)
    seen = {"parser": parser, "renderer": renderer,
            "template_sha256": sha256_text(got_t), "architecture": arch, "capabilities": caps,
            "num_ctx_65536": bool(re.search(r"(?m)^num_ctx\s+65536\s*$", params))}
    out["show"] = seen
    problems = []
    if seen["parser"] != (facts.get("parser") or None):
        problems.append(f"parser {seen['parser']!r} (Modelfile {facts.get('parser')!r})")
    if seen["renderer"] != (facts.get("renderer") or None):
        problems.append(f"renderer {seen['renderer']!r} (Modelfile {facts.get('renderer')!r})")
    if want_t is not None and got_t != want_t:
        problems.append(f"template sha256 {seen['template_sha256']} != Modelfile's {sha256_text(want_t)}")
    if arch != mx_gguf.ARCH[family]:
        problems.append(f"architecture {arch!r} (want {mx_gguf.ARCH[family]!r})")
    if "tools" not in caps:
        problems.append(f"no tools capability ({caps})")
    if not seen["num_ctx_65536"]:
        problems.append("num_ctx is not 65536")
    if problems:
        raise Blocked("show", "; ".join(problems))
    t0 = time.monotonic()
    st, body = http_json(url + "/api/generate", {"model": MODEL_NAME, "prompt": "", "stream": False}, timeout=load_timeout)
    out["load_s"] = round(time.monotonic() - t0, 1)
    if st != 200:
        raise Blocked("load", f"load-only /api/generate answered {st}: {str(body)[:300]}")
    st, ps = http_json(url + "/api/ps", timeout=30)
    mods = [m for m in (ps or {}).get("models") or [] if str(m.get("name", "")).startswith(MODEL_NAME)]
    if not mods:
        raise Blocked("gpu", "/api/ps lists no loaded sous-chef model")
    size, vram = int(mods[0].get("size") or 0), int(mods[0].get("size_vram") or 0)
    out["ps"] = {"size": size, "size_vram": vram, "context_length": mods[0].get("context_length")}
    if size <= 0 or vram < size:
        raise Blocked("gpu", f"not 100% GPU: size_vram {vram} of {size} bytes (a CPU fallback would make every number wrong)")
    # the first prompt this server sees (the load above carried none, so no cached prefix shortens the count)
    from souschef_lab import serving_mistral as SM
    want_n = smoke_prompt_tokens(facts, family)
    try:
        # one output token: the count is the prompt's, and no reply is long enough for a parser error to hide it
        st, sm = http_json(url + "/v1/messages", dict(SM.SMOKE_REQUEST, model=MODEL_NAME, max_tokens=1, temperature=0,
                                                      stream=False), timeout=300)
    except (urlerror.URLError, OSError) as e:
        raise Blocked("render", f"/v1/messages prompt-size smoke: {e}") from None
    if st == 404:
        raise Blocked("decode", "/v1/messages answered 404 (no messages route: the service cannot reach this model)")
    got_n = ((sm or {}).get("usage") or {}).get("input_tokens") if isinstance(sm, dict) else None
    out["prompt_smoke"] = {"status": st, "input_tokens": got_n, "expected": want_n}
    if want_n is not None and (st != 200 or got_n != want_n):
        raise Blocked("render", f"the smoke prompt is {got_n} tokens (status {st}), want {want_n}: the served template, "
                                "BOS or control-token handling differs from the lab's render")
    t0 = time.monotonic()
    st, chat = http_json(url + "/api/chat", {"model": MODEL_NAME, "stream": False,
                                             "messages": [{"role": "user", "content": "Reply with the word OK."}],
                                             "options": {"num_predict": 8, "temperature": 0}}, timeout=300)
    out["chat"] = {"status": st, "s": round(time.monotonic() - t0, 2),
                   "eval_count": (chat or {}).get("eval_count"), "done_reason": (chat or {}).get("done_reason")}
    if st != 200 or not isinstance((chat or {}).get("message"), dict):
        raise Blocked("decode", f"/api/chat answered {st}: {str(chat)[:300]}")
    try:  # temperature 0: the same answer every time, so a block here is the serving's, not a bad draw
        st, msg = http_json(url + "/v1/messages", {"model": MODEL_NAME, "max_tokens": 128, "temperature": 0, "stream": False,
                                                   "thinking": {"type": "disabled"}, "tools": [ECHO_TOOL],
                                                   "messages": [{"role": "user", "content": "Use the echo tool to echo the word hello."}]},
                            timeout=300)
    except (urlerror.URLError, OSError) as e:
        raise Blocked("decode", f"/v1/messages: {e}") from None
    content = [c for c in (msg or {}).get("content") or [] if isinstance(c, dict)]
    kinds = [c.get("type") for c in content]
    text = "".join(str(c.get("text") or "") for c in content if c.get("type") == "text")
    leaked = [m for m in LEAK_MARKERS.get(family, ()) if m in text]
    out["messages_smoke"] = {"status": st, "content": kinds, "stop_reason": (msg or {}).get("stop_reason"),
                             "error": (msg or {}).get("error") if st != 200 else None, "leaked_markers": leaked}
    # the smoke's own /v1/messages 500s (the prompt-size smoke, this one): not the rows' hidden retries (retries_500)
    out["smoke_500s"] = int(out["prompt_smoke"]["status"] == 500) + int(st == 500)
    if st == 404 or (strict_smoke and (st != 200 or not ({"tool_use", "text"} & set(kinds)))):
        raise Blocked("decode", f"/v1/messages smoke answered {st} with {kinds}")
    if leaked:  # the parser never saw the call: control tokens stripped or detokenized as text — every call would fail
        raise Blocked("render", f"/v1/messages smoke answered text with the call markers {leaked}: the control tokens "
                                "do not reach the parser")
    if family == "mistral" and st == 200 and "tool_use" not in kinds:
        raise Blocked("render", f"/v1/messages smoke at temperature 0 made no tool call ({kinds}): Ministral calls echo "
                                "here when the tools reach it and its [TOOL_CALLS] reaches the parser")
    return out


# --- exam (Stage 1) -----------------------------------------------------------------------------------------------------

def fmt_kind(r: dict) -> str | None:
    """A format failure: the turn errored (model_unavailable after a parser error, stream end, history failed, …),
    or it stopped at max_tokens having written nothing. None = the model answered (right or wrong)."""
    if r.get("status") != "ok":
        return str(r.get("code") or r.get("status") or "error")
    if r.get("code") == "max_tokens" and not r.get("written"):
        return "max_tokens"
    return None


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, max(0, math.ceil(q * len(v)) - 1))], 1)


def summarize_exam(reps: list[dict[str, list[dict]]], *, arm: str) -> dict:
    """reps[k][set] = trimmed rows of replicate k → summary.json's scores (per set and pooled)."""
    k_n = len(reps)
    rows: dict[str, dict] = {}
    for k, by_set in enumerate(reps):
        for s, trs in by_set.items():
            for r in trs:
                e = rows.setdefault(r["key"], {"set": s, "pass": [None] * k_n, "fmt": [None] * k_n})
                e["pass"][k] = int(r.get("pass") or 0)
                e["fmt"][k] = fmt_kind(r)
    lat = [float(r["latency_ms"]) for by_set in reps for trs in by_set.values() for r in trs
           if r.get("status") == "ok" and isinstance(r.get("latency_ms"), (int, float))]
    out_tok = [(r.get("usage") or {}).get("output_tokens") for by_set in reps for trs in by_set.values() for r in trs
               if r.get("status") == "ok" and isinstance(r.get("usage"), dict)]

    def group(keys: list[str]) -> dict:
        full = [k for k in keys if None not in rows[k]["pass"]]
        per = [sum(rows[k]["pass"][i] for k in full) for i in range(k_n)]
        fmt: dict[str, int] = {}
        for k in full:
            for f in rows[k]["fmt"]:
                if f:
                    fmt[f] = fmt.get(f, 0) + 1
        return {"n": len(full), "pass": per, "mean": round(sum(per) / k_n, 2) if k_n else 0.0,
                "flips": sum(1 for k in full if len(set(rows[k]["pass"])) > 1),
                "fmt_rows": sum(1 for k in full if any(rows[k]["fmt"])), "fmt": fmt}

    sets = {}
    for s in sorted({e["set"] for e in rows.values()}, key=lambda x: (EXAM_SETS.index(x.split(":")[0]) if x.split(":")[0] in EXAM_SETS else 99, x)):
        sets[s] = group([k for k, e in rows.items() if e["set"] == s])
    return {"arm": arm, "replicates": k_n, "sets": sets, "pooled": group(list(rows)), "rows": rows,
            "latency_ms": {"n": len(lat), "p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95)},
            "output_tokens": {"n": len([t for t in out_tok if isinstance(t, (int, float))]),
                              "p50": _pct([float(t) for t in out_tok if isinstance(t, (int, float))], 0.5)}}


# Ollama's access log. The service's SDK calls /v1/messages?beta=true: the query must not hide the line (until
# 2026-10-18 it did, and every retries_500 read 0 — r6's r4-mx candidate drew 3 parser-error 500s).
GIN = re.compile(r'\|\s*(\d{3})\s*\|[^|]*\|[^|]*\|\s*(?:POST|GET)\s+"(/v1/messages|/api/chat|/api/generate)(?:\?[^"]*)?"')
SERVICE_MODEL_ERR = re.compile(r"model_unavailable|tool '[^']*' not found|invalid character|unexpected end of JSON")


def log_counts(ollama_logs: Iterable[Path], service_log: Path | None) -> dict:
    """HTTP statuses Ollama answered per endpoint (its access log) and the service's model-error lines."""
    status: dict[str, dict[str, int]] = {}
    for p in ollama_logs:
        if Path(p).is_file():
            for ln in Path(p).read_text(encoding="utf-8", errors="replace").splitlines():
                m = GIN.search(ln)
                if m:
                    d = status.setdefault(m.group(2), {})
                    d[m.group(1)] = d.get(m.group(1), 0) + 1
    errs = 0
    if service_log and Path(service_log).is_file():
        errs = sum(1 for ln in Path(service_log).read_text(encoding="utf-8", errors="replace").splitlines()
                   if SERVICE_MODEL_ERR.search(ln))
    return {"ollama_status": status, "service_model_error_lines": errs}


def retries_500(logs: dict | None, served: dict | None = None) -> int:
    """The /v1/messages 500s the rows drew (Ollama answers a parser error — an unknown tool name, bad JSON — with 500,
    and the service's SDK retries a 5xx twice, each a fresh sample: hidden extra draws), the serving smokes' own 500s
    left out. Qwen's parser passes unknown names through, so these are mostly the Mistral arm's."""
    n = int((((logs or {}).get("ollama_status") or {}).get("/v1/messages") or {}).get("500") or 0)
    return max(0, n - int((served or {}).get("smoke_500s") or 0))


def exam_rows(sets: Iterable[str], limit: int | None = None) -> dict[str, list[dict]]:
    """The real-phrasing rows as the eval's `<set>:recorded` arm runs them: the exam's own manifest and contexts,
    today's answer key and font policy (r4score.policy_of after adopt_trigger_policy). All 252 rows, the create_page
    capability row included (it is the same row for every arm)."""
    from souschef_lab.exam import load_set
    from souschef_lab.r4score import apply_font_policy, policy_of
    font = policy_of()["font"]
    out = {}
    for s in sets:
        rows = [apply_font_policy(r, font)[0] for r in load_set(s)]
        out[s] = rows[:limit] if limit else rows
    return out


def adopt_policy(trigger: dict | None) -> dict:
    from souschef_lab import r4score as R4S
    return R4S.adopt_trigger_policy(trigger, base=R4S.STRICT_POLICY)


def engine_for(cases_run: str, work: Path) -> dict:
    """The browser engine the eval uses (pipeline.ensure_engine: $ENGINE, else the bucket's bundle for r6's sha)."""
    from souschef_lab.pipeline import _want_sha, ensure_engine
    store = read_store()
    return ensure_engine(store, _want_sha(store, cases_run), engine_dir=os.environ.get("MX_ENGINE_DIR") or None, work=work)


def harness_factory(per_route: dict[str, dict] | None = None, **fixed) -> Callable:
    """Harnesses as every arm runs them (auth none, prime off). `per_route` = {route: {field: value}} on top of one
    route only — r4-mxt's candidate temperature; every other route keeps the deployed 0.3."""
    from souschef_lab.harness import Harness, route_spec

    class _Harness(Harness):
        def model_routes(self) -> dict[str, dict]:
            return {name: route_spec(u, name, auth=self.auth, overrides={**self.route_overrides, **((per_route or {}).get(name) or {})})
                    for name, u in self.routes.items()}

    def make(routes: dict, **kw):
        return _Harness(routes, auth="none", route_overrides={"prime": "off"}, model_call_timeout="180s", **fixed, **kw)
    return make


def cmd_exam(a) -> int:
    run, arm = check_run(a.run), a.arm
    if arm not in EXAM_ARMS:
        raise ConfigError(f"--arm {arm}: one of {', '.join(EXAM_ARMS)}")
    gs = GS()
    a.cases_run = resolve_cases_run(gs, run, a.cases_run)
    work = Path(a.workdir or os.environ.get("MX_WORKDIR") or tempfile.mkdtemp(prefix="mx-exam-"))
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    dest = f"{run_root(run)}/exam/{arm}"
    reps_n = a.replicates if a.replicates is not None else (2 if arm in TWO_REPLICATE_ARMS else 1)
    attempt = attempt_id()
    ol: Ollama | None = None
    t0 = time.monotonic()
    info: dict[str, Any] = {"arm": arm, "run": run, "cases_run": a.cases_run, "replicates": reps_n, "started_at": utc_now(),
                            "dry": bool(a.dry), "attempt": attempt}
    base = begin_attempt(gs, dest, attempt, {"arm": arm, "run": run})  # a Restarted: exit 1, nothing written
    rc = 0
    try:
        if a.dry:
            url, served, model = None, {"dry": True}, {"dry": True}
        else:
            model = pull_model(gs, a.model or model_uri(arm, run, a.cases_run), work / "model",
                               family="mistral" if arm == NATIVE_ARM else None)
            (out / "gguf-check.json").write_text(json.dumps(model["check"], indent=1), encoding="utf-8")
            ol = Ollama(11434, work / "ollama-models", out / "ollama.log")
            ol.start()
            ol.create(Path(model["dir"]))
            served = assert_served(ol.url, model["facts"], model["family"], strict_smoke=a.strict_smoke)
            url = ol.url
        (out / "served.json").write_text(json.dumps(served, indent=1), encoding="utf-8")
        info["served"] = served
        info["model"] = {k: model.get(k) for k in ("uri", "gguf", "family")} if not a.dry else {"dry": True}
        rs = read_store()
        trig = (rs.read_json(f"runs/{a.cases_run}/run.json") or {}).get("trigger")
        info["score_policy"] = adopt_policy(trig)
        eng = engine_for(a.cases_run, work)
        info["engine"] = {k: eng.get(k) for k in ("status", "sha", "fingerprint")}
        rows = exam_rows(a.sets.split(","), a.limit)
        reps = run_exam(arm, rows, url=url, engine=eng.get("engine"), reps=reps_n, workers=a.workers, work=work,
                        dry=a.dry, out=out)
        summary = summarize_exam(reps, arm=arm)
        logs = log_counts([out / "ollama.log"], out / "service.log")
        summary.update({"run": run, "at": utc_now(), "wall_s": round(time.monotonic() - t0), "logs": logs,
                        "retries_500": retries_500(logs, served),
                        **{k: info.get(k) for k in ("served", "model", "engine", "score_policy", "dry", "attempt")}})
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        p = summary["pooled"]
        say(f"exam {arm}: {p['pass']} of {p['n']} (flips {p['flips']}, format failures on {p['fmt_rows']} rows)")
    except Blocked as e:
        rc = 3
        (out / "failed.json").write_text(json.dumps({"reason": e.reason, "detail": e.detail, "at": utc_now(), **info},
                                                    indent=1, default=str), encoding="utf-8")
        say(f"exam {arm}: BLOCKED ({e})")
    except BaseException as e:
        rc = getattr(e, "code", 1) if isinstance(e, MxError) else 1
        (out / "failed.json").write_text(json.dumps({"reason": "error", "detail": f"{type(e).__name__}: {e}"[:2000],
                                                     "at": utc_now(), **info}, indent=1, default=str), encoding="utf-8")
        if not isinstance(e, Exception):
            raise
        say(f"exam {arm}: FAILED ({type(e).__name__}: {e})")
    finally:
        if ol is not None:
            ol.stop()
        try:
            gs.put_dir(out, base)
            publish_attempt(gs, dest, attempt, rc, {"arm": arm})
        except MxError as e:
            say(f"exam {arm}: upload to {base} failed: {e}")
            rc = rc or 1
    return rc


def run_exam(arm: str, rows: dict[str, list[dict]], *, url: str | None, engine: str | None, reps: int, workers: int,
             work: Path, dry: bool, out: Path) -> list[dict[str, list[dict]]]:
    from souschef_lab.exam import MANIFEST_FILE
    from souschef_lab.manifest import Manifest
    from souschef_lab.retrain.common import run8, write_jsonl
    from souschef_lab.retrain.eval_stage import SET_CODE, Arms
    exam_m = Manifest(json.loads(Path(MANIFEST_FILE).read_text(encoding="utf-8")))
    route = "candidate" if dry else arm  # DRY: the fake service's scripted route
    harness = None if dry else harness_factory()({arm: url}, run_id=f"mx{run8(arm)}", workdir=work / "harness")
    if harness is not None:
        harness.__enter__()
    try:
        arms = Arms(harness=harness, dry=dry, engine_path=engine, workers=workers, progress=say,
                    tolerate_errors=frozenset({route}))
        tag = run8(arm)
        result: list[dict[str, list[dict]]] = []
        for k in range(1, reps + 1):
            by_set: dict[str, list[dict]] = {}
            for s, rs in rows.items():
                name = f"{s}:recorded"
                tr = arms.exam(route, name, rs, Path(MANIFEST_FILE), exam_m, sample=k,
                               prefix=f"lab-mx{tag}{k}{SET_CODE.get(s, s[:2])}-", script_rows=rs if dry else None,
                               script_m=exam_m if dry else None)
                by_set[name] = tr
                write_jsonl(out / f"{s}-k{k}.jsonl", tr)
            result.append(by_set)
        return result
    finally:
        if harness is not None:
            harness.__exit__(None, None, None)
            if (work / "harness" / "service.log").is_file():
                shutil.copyfile(work / "harness" / "service.log", out / "service.log")


# --- gate (Stage 1 verdict) ---------------------------------------------------------------------------------------------

def _rows(summary: dict) -> dict[str, dict]:
    return {k: v for k, v in (summary.get("rows") or {}).items() if None not in (v.get("pass") or [None])}


def _mean_pass(rows: dict[str, dict], keys: list[str]) -> float:
    if not keys:
        return 0.0
    k_n = len(next(iter(rows.values()))["pass"]) if rows else 1
    return sum(sum(rows[k]["pass"]) for k in keys) / k_n


def _replicates(summary: dict) -> int:
    rows = _rows(summary)
    return len(next(iter(rows.values()))["pass"]) if rows else int(summary.get("replicates") or 0)


def _compare(mx: dict, qw: dict) -> dict:
    """Δ, σ_Δ, Δ_fmt and the thresholds of one Mistral exam against qwen-base's, over their common rows."""
    rm, rq = _rows(mx), _rows(qw)
    keys = sorted(set(rm) & set(rq))
    n = len(keys)
    scale = n / EXAM_N
    s_mx, s_qw = _mean_pass(rm, keys), _mean_pass(rq, keys)
    f_mx = sum(1 for k in keys if len(set(rm[k]["pass"])) > 1)
    f_qw = sum(1 for k in keys if len(set(rq[k]["pass"])) > 1)
    clean = [k for k in keys if not any(rm[k].get("fmt") or [])]
    sigma = 0.5 * math.sqrt(f_mx + f_qw)
    return {"n": n, "scale": scale, "s_mx": s_mx, "s_qw": s_qw, "f_mx": f_mx, "f_qw": f_qw, "delta": s_mx - s_qw,
            "sigma": sigma, "clean": len(clean),
            "d_fmt": ((_mean_pass(rm, clean) - _mean_pass(rq, clean)) * n / len(clean)) if clean else float("-inf"),
            "go_thr": max(GO_ROWS * scale, 2 * sigma), "stop_thr": max(STOP_ROWS * scale, 3 * sigma)}


def _band(c: dict) -> str:
    if c["delta"] >= -c["go_thr"]:
        return "GO"
    if c["delta"] < -c["stop_thr"] and c["d_fmt"] < -STOP_FMT_ROWS * c["scale"]:
        return "STOP"
    return "JUDGE"


def gate(mx: dict | None, qw: dict | None, *, mx_failed: dict | None = None, r4: dict | None = None,
         r4_ref: float | None = None, native: dict | None = None) -> dict:
    """The Stage 1 verdict (docs/MISTRAL.md §5.2) from the two gate arms' summary.json (2 replicates each), and the
    mx-base-native diagnostic's when it ran."""
    doc: dict[str, Any] = {"at": utc_now(), "rule": {"go_rows": GO_ROWS, "stop_rows": STOP_ROWS, "stop_fmt_rows": STOP_FMT_ROWS,
                                                    "n_ref": EXAM_N}}
    if mx_failed and mx_failed.get("reason") in BLOCKING:
        doc.update(verdict="BLOCKED", why=f"mx-base cannot be served: {mx_failed.get('reason')}: {mx_failed.get('detail')}")
        return doc
    if not mx or not qw:
        doc.update(verdict="MISSING", why="exam/mx-base and exam/qwen-base summary.json are both needed")
        return doc
    reps = {"mx-base": _replicates(mx), "qwen-base": _replicates(qw)}
    if min(reps.values()) < 2:  # with one replicate no row can flip: σ_Δ would read 0 and the thresholds go fixed
        doc.update(verdict="INVALID", why=f"replicates {reps}: the gate needs 2 per arm (σ_Δ comes from the flips)",
                   replicates=reps)
        return doc
    c = _compare(mx, qw)
    n, scale, delta, d_fmt, go_thr, stop_thr = c["n"], c["scale"], c["delta"], c["d_fmt"], c["go_thr"], c["stop_thr"]
    if n == 0:
        doc.update(verdict="MISSING", why="no row is common to both arms")
        return doc
    # hidden retries (M3): each /v1/messages 500 the rows drew was retried by the service's SDK — a fresh sample the
    # Qwen arm's parser never asks for. At worst each excess retry rescued one row of one replicate.
    r_mx, r_qw = retries_500(mx.get("logs"), mx.get("served")), retries_500(qw.get("logs"), qw.get("served"))
    excess = max(0, r_mx - r_qw) / reps["mx-base"]
    verdict = _band(c)
    default = None
    if verdict == "GO":
        why = f"Δ {delta:+.1f} ≥ −{go_thr:.1f} rows"
        if delta - excess < -go_thr:
            verdict = "JUDGE"
            d_wc = delta - excess
            default = "GO" if d_fmt >= d_wc / 2 else "STOP"
            why += (f", but mx-base drew {r_mx} parser-error 500s (qwen-base {r_qw}) the service retried: if each excess "
                    f"retry rescued a row, Δ is {d_wc:+.1f} — below GO; default {default}; the user decides")
    elif verdict == "STOP":
        why = (f"Δ {delta:+.1f} < −{stop_thr:.1f} and Δ_fmt {d_fmt:+.1f} < −{STOP_FMT_ROWS * scale:.1f}: behind even on its "
               "well-formed turns")
    else:
        default = "GO" if d_fmt >= delta / 2 else "STOP"
        why = (f"Δ {delta:+.1f} between −{go_thr:.1f} and the STOP rule (Δ_fmt {d_fmt:+.1f}); default {default} "
               f"({'format failures explain at least half the gap' if default == 'GO' else 'they do not'}); "
               "run the mx-base-native diagnostic and let the user decide")
    # mx-base-native: is the gap the compact Go-JSON template's rather than the model's? A STOP is final only when
    # the native rendering is a STOP too (S3).
    if native and _rows(native):
        cn = _compare(native, qw)
        nb = _band(cn) if _replicates(native) >= 2 and cn["n"] else "INVALID"
        doc["native"] = {"verdict": nb, "n": cn["n"], "delta": round(cn["delta"], 2), "delta_fmt": round(cn["d_fmt"], 2)
                         if cn["clean"] else None, "sigma_delta": round(cn["sigma"], 2),
                         "retries_500": retries_500(native.get("logs"), native.get("served"))}
        if verdict == "STOP" and nb != "STOP":
            verdict, default = "JUDGE", "GO" if nb == "GO" else None
            why += (f"; but mx-base-native is {nb} (Δ {cn['delta']:+.1f}): the template, not the model, may cost the "
                    "gap — the user decides")
        elif verdict == "STOP":
            why += f"; mx-base-native is STOP too (Δ {cn['delta']:+.1f})"
    elif verdict == "STOP":
        doc["confirm"] = "run the mx-base-native diagnostic (docs/MISTRAL.md §4) and re-run gate before accepting STOP"
        why += "; not final until mx-base-native has run"
    if default:
        doc["default"] = default
    fmt_mx = (mx.get("pooled") or {}).get("fmt") or {}
    fmt_qw = (qw.get("pooled") or {}).get("fmt") or {}
    doc.update(verdict=verdict, why=why, n=n, s_mx=round(c["s_mx"], 2), s_qw=round(c["s_qw"], 2), delta=round(delta, 2),
               sigma_delta=round(c["sigma"], 2), flips={"mx-base": c["f_mx"], "qwen-base": c["f_qw"]}, clean_rows=c["clean"],
               delta_fmt=None if not c["clean"] else round(d_fmt, 2),
               thresholds={"go": round(go_thr, 2), "stop": round(stop_thr, 2)}, replicates=reps,
               retries_500={"mx-base": r_mx, "qwen-base": r_qw, "excess_rows_per_replicate": round(excess, 2)},
               format_failures={"mx-base": fmt_mx, "qwen-base": fmt_qw},
               sets={s: {"mx-base": (mx.get("sets") or {}).get(s, {}).get("mean"), "qwen-base": (qw.get("sets") or {}).get(s, {}).get("mean"),
                         "n": (mx.get("sets") or {}).get(s, {}).get("n")} for s in (mx.get("sets") or {})},
               latency_ms={"mx-base": mx.get("latency_ms"), "qwen-base": qw.get("latency_ms")},
               logs={"mx-base": mx.get("logs"), "qwen-base": qw.get("logs")})
    if r4:
        p = (r4.get("pooled") or {})
        anchor: dict[str, Any] = {"r4_pass": p.get("mean"), "n": p.get("n"), "retries_500": retries_500(r4.get("logs"), r4.get("served"))}
        if r4_ref is not None and isinstance(p.get("mean"), (int, float)):
            tol = max(2.0, 2 * math.sqrt(0.25 * c["f_qw"]))
            gap = float(p["mean"]) - float(r4_ref)
            anchor.update(reference=r4_ref, gap=round(gap, 2), tol=round(tol, 2), flagged=abs(gap) > tol,
                          note="Vertex L4 + prime off vs the pipeline's recorded-manifest rows (r5's eval, current = r4)")
        doc["r4_anchor"] = anchor
    return doc


def r4_reference(bar: dict | None) -> float | None:
    """r4's pooled pass count on the six `<set>:recorded` rows of a round's eval/bar.json (current = r4)."""
    if not bar:
        return None
    vals = [s.get("current") for s in bar.get("sets") or [] if str(s.get("set", "")).endswith(":recorded")
            and str(s.get("set")).split(":")[0] in EXAM_SETS]
    return round(sum(vals), 2) if vals and all(isinstance(v, (int, float)) for v in vals) else None


def read_exam(gs: GS, root: str, arm: str, name: str) -> dict | None:
    return gs.read_json(results_uri(gs, f"{root}/exam/{arm}", name))


def cmd_gate(a) -> int:
    gs = GS()
    root = run_root(a.run)
    mx = read_exam(gs, root, "mx-base", "summary.json")
    qw = read_exam(gs, root, "qwen-base", "summary.json")
    r4 = read_exam(gs, root, "r4", "summary.json")
    native = read_exam(gs, root, NATIVE_ARM, "summary.json")
    failed = read_exam(gs, root, "mx-base", "failed.json")
    ref = r4_reference(gs.read_json(f"{BUCKET}/runs/{a.r4_ref_run}/eval/bar.json")) if a.r4_ref_run else None
    doc = gate(mx, qw, mx_failed=failed if not mx else None, r4=r4, r4_ref=ref, native=native)
    doc["run"] = a.run
    if not a.no_write:
        gs.put_json(doc, f"{root}/gate.json")
    print(json.dumps({k: doc.get(k) for k in ("verdict", "why", "delta", "sigma_delta", "delta_fmt", "default", "retries_500",
                                              "native", "confirm", "r4_anchor")}, indent=1, default=str))
    return 0


def cmd_user_go(a) -> int:
    """The user's GO on a JUDGE gate (S5 of review 2): user-go.json names the gate.json it answers (its `at`), so a
    re-run gate needs a new decision. Only a JUDGE can be overridden; a STOP or BLOCKED cannot."""
    gs = GS()
    root = run_root(a.run)
    gd = gs.read_json(f"{root}/gate.json") or {}
    if gd.get("verdict") != "JUDGE":
        raise ConfigError(f"gate.json verdict {gd.get('verdict')!r}: only a JUDGE takes the user's GO")
    if not (a.by or "").strip() or not (a.note or "").strip():
        raise ConfigError("--by (who decided) and --note (why) are required")
    doc = {"decision": "GO", "gate_at": gd.get("at"), "gate_verdict": gd.get("verdict"), "gate_default": gd.get("default"),
           "by": a.by.strip(), "note": a.note.strip(), "at": utc_now()}
    gs.put_json(doc, f"{root}/user-go.json")
    print(json.dumps(doc, indent=1))
    return 0


def stage1_go(gd: dict, ug: dict | None) -> tuple[bool, str]:
    if gd.get("verdict") == "GO":
        return True, "gate.json verdict GO"
    if gd.get("verdict") == "JUDGE" and ug and ug.get("decision") == "GO" and ug.get("gate_at") == gd.get("at"):
        return True, f"JUDGE, the user's GO ({ug.get('by')}, {ug.get('at')})"
    return False, (f"gate.json verdict {gd.get('verdict')!r}" + (" (JUDGE: `mxarm user-go` records the user's GO)"
                                                               if gd.get("verdict") == "JUDGE" else ""))


# --- eval (Stage 3) ---------------------------------------------------------------------------------------------------------

SNAP_REL = "experiments/mistral/snapshots/r4"  # bucket-relative (a Store key); snapshot_r4 writes it
SNAP_FILES = ("state/model.json", "state/deployed.json", f"baselines/{R4_DIGEST}/baseline.json",
              f"manifests/{R4_MANIFEST}.json")


def snapshot_r4(gs: GS) -> dict:
    """Freeze r4 as the current model for Stage 3's r4 pairs: state/model.json (must still be r4: R4_DIGEST,
    R4_MANIFEST), state/deployed.json, r4's baseline and training manifest → experiments/mistral/snapshots/r4/ (+
    snapshot.json). Idempotent: an existing complete snapshot of r4 is kept. Run it before r6 can be rolled out."""
    snap = f"{BUCKET}/{SNAP_REL}"
    have = gs.read_json(f"{snap}/snapshot.json")
    if have and have.get("digest") == R4_DIGEST and all(gs.exists(f"{snap}/{f}") for f in SNAP_FILES):
        return dict(have, kept=True)
    model = gs.read_json(f"{BUCKET}/state/model.json") or {}
    if model.get("digest") != R4_DIGEST or model.get("training_manifest") != R4_MANIFEST:
        raise ConfigError(f"state/model.json is {model.get('tag')!r} {model.get('digest')!r}, not r4 ({R4_DIGEST}): the "
                          "live model moved before r4 was snapshotted — restore r4's state files by hand from "
                          "state history before Stage 3")
    for f in SNAP_FILES:
        if not gs.exists(f"{BUCKET}/{f}"):
            raise ConfigError(f"{BUCKET}/{f} is missing")
        gs.copy_object(f"{BUCKET}/{f}", f"{snap}/{f}")
    doc = {"tag": model.get("tag"), "digest": R4_DIGEST, "training_manifest": R4_MANIFEST, "files": list(SNAP_FILES),
           "at": utc_now()}
    gs.put_json(doc, f"{snap}/snapshot.json")
    return doc


def mirror_store(src, dst, cases_run: str, *, current: str, candidate: str, run: str) -> dict:
    """A LOCAL copy of what eval_stage.run reads for the cases run, with the current and candidate models patched
    for the pair. `src`/`dst` are retrain Stores (src: the pipeline bucket, read only; dst: a local folder). Nothing
    is written to `src`.

    current r4     r4's state, baseline and manifest from the snapshot (snapshot_r4), so a rollout cannot swap
                   them; without one, the live state/model.json must still be r4.
    current <L>c   a round's candidate (r6c, r7c): when it is the live model (r6c since 2026-10-17) the live
                   state/model.json and baselines/<digest>/baseline.json are used as they are — what that round's
                   successor evaluated against; otherwise state/model.json is patched to it and its baseline is
                   baselines/<digest>/baseline.json when one exists (an approved model), else the round's
                   eval/candidate-baseline.json (that round's eval must have finished).
    candidate      mx / mxt: the cases run's model/image.json is patched to the arm; the cases run's own
                   candidate (<L>c): its model/image.json as it is.

    Every current's baseline is then PINNED to the cases round's eval (pin_current_baseline): replicates added after
    the round's report.json `at` (a drift weekly arm T on the live model) are dropped and the sha256 is logged, so
    the bar's tolerances here are the ones the round's own eval used.

    r7's bar also reads r4's RECORDED reference (thresholds r4_reference: the first listed baseline path whose
    sha256 is the pinned one; rule r4_floor) and the reference's own rows (r4_reference.rows_run's
    eval/current/rows.jsonl: the re-run floor, report only, and the clear-asks count): both are copied when the
    bucket has them, so every pair is scored as the round's own eval is."""
    pre = f"runs/{check_cases_run(cases_run)}"
    ri = round_info(cases_run)
    log: dict[str, list] = {"copied": [], "missing": [], "patched": []}

    def cp(rel: str, required: bool = True) -> bool:
        b = src.read_bytes(rel)
        if b is None:
            if required:
                raise ConfigError(f"{rel} is missing in the pipeline bucket")
            log["missing"].append(rel)
            return False
        dst.write_bytes(rel, b)
        log["copied"].append(rel)
        return True

    for rel in ("state/model.json", "state/deployed.json", f"{pre}/run.json"):
        cp(rel)
    live = json.loads(dst.read_bytes("state/model.json"))
    if current == "r4":
        snap = src.read_bytes(f"{SNAP_REL}/state/model.json")
        if snap is not None:
            for f in SNAP_FILES[:1] + SNAP_FILES[2:]:  # the live deployed.json stays (it is the site's, not the model's)
                b = src.read_bytes(f"{SNAP_REL}/{f}")
                if b is None:
                    raise ConfigError(f"{SNAP_REL}/{f} is missing: the r4 snapshot is incomplete (snapshot-r4)")
                dst.write_bytes(f, b)
            log["patched"].append(f"state/model.json, r4's baseline and manifest ← {SNAP_REL}/")
        live = json.loads(dst.read_bytes("state/model.json"))
        if live.get("digest") != R4_DIGEST:
            raise ConfigError(f"the current model is {live.get('tag')!r} {live.get('digest')!r}, not r4 ({R4_DIGEST}), and "
                              f"there is no r4 snapshot ({SNAP_REL}/): run `mxarm.py snapshot-r4` while r4 is live")
    cp("config/thresholds.json", required=False)
    cp("cases/pool/index.json", required=False)
    for s in ("prepare", "collect", "dataset"):
        cp(f"{pre}/result-{s}.json", required=False)
    prep = [k for k in src.list(f"{pre}/prepare")]
    if f"{pre}/prepare/manifest.json" not in prep or f"{pre}/prepare/cases.jsonl" not in prep:
        raise ConfigError(f"{pre}/prepare/ lacks manifest.json or cases.jsonl")
    for rel in prep:
        cp(rel)
    run_doc = json.loads(dst.read_bytes(f"{pre}/run.json"))
    drift = (run_doc.get("trigger") or {}).get("drift_run")
    if drift:
        cp(f"drift/{drift}/report.json", required=False)
        cp(f"drift/{drift}/static.json", required=False)
    model = json.loads(dst.read_bytes("state/model.json"))
    if dst.read_bytes(f"manifests/{model['training_manifest']}.json") is None:  # the snapshot's, when it gave one
        cp(f"manifests/{model['training_manifest']}.json", required=False)
    dep = json.loads(dst.read_bytes("state/deployed.json"))
    if dep.get("hash"):
        cp(f"manifests/{dep['hash']}.json", required=False)
    cp(f"{pre}/model/image.json", required=ri["cc"] in (current, candidate))
    cur_round = round_of_model(current)
    if current == "r4":
        if dst.read_bytes(f"baselines/{model['digest']}/baseline.json") is None:
            cp(f"baselines/{model['digest']}/baseline.json")
    elif cur_round:
        rpre = f"runs/{cur_round}"
        img = src.read_json(f"{rpre}/model/image.json") or {}
        if not img.get("digest"):
            raise ConfigError(f"{rpre}/model/image.json is needed for {current} as current (its model image)")
        if live.get("digest") == img["digest"]:
            # the live model: the state and baseline the successor round's own eval read, untouched
            if not cp(f"baselines/{img['digest']}/baseline.json", required=False):
                raise ConfigError(f"baselines/{img['digest']}/baseline.json is missing for the live {current}")
            log["patched"].append(f"none: {current} is the live model (state/model.json and its baseline as they are)")
        else:
            cb_rel = f"baselines/{img['digest']}/baseline.json"
            cb = src.read_json(cb_rel)
            if cb is None:
                cb_rel = f"{rpre}/eval/candidate-baseline.json"
                cb = src.read_json(cb_rel)
            if not cb:
                raise ConfigError(f"baselines/{img['digest']}/baseline.json or {rpre}/eval/candidate-baseline.json is "
                                  f"needed for {current} as current ({cur_round}'s own eval must have finished)")
            hsh = cb.get("training_manifest") or (src.read_json(f"{rpre}/prepare/manifest.json") or {}).get("hash")
            if not hsh:
                raise ConfigError(f"{current}'s baseline names no training_manifest")
            if not dst.read_bytes(f"manifests/{hsh}.json"):
                if not cp(f"manifests/{hsh}.json", required=False):
                    pm = src.read_bytes(f"{rpre}/prepare/manifest.json")
                    if pm is None:
                        raise ConfigError(f"manifests/{hsh}.json and {rpre}/prepare/manifest.json are both missing")
                    dst.write_bytes(f"manifests/{hsh}.json", pm)
                    log["patched"].append(f"manifests/{hsh}.json ← {rpre}/prepare/manifest.json")
            dst.write_json("state/model.json", {"tag": img.get("tag"), "digest": img["digest"], "training_manifest": hsh,
                                                "run_id": cur_round, "mx_note": f"LOCAL patch: {current} as the current model"})
            dst.write_json(f"baselines/{img['digest']}/baseline.json", cb)
            log["patched"] += [f"state/model.json → {current}", f"baselines/{img['digest']}/baseline.json ← {cb_rel}"]
    else:
        raise ConfigError(f"current {current!r}: r4 or a round's candidate ({', '.join(i['label'] + 'c' for i in ROUNDS.values())})")
    if candidate in MX_MODELS:
        dst.write_json(f"{pre}/model/image.json", {"digest": f"{candidate}:{run}", "tag": f"{candidate}-{run}",
                                                   "mx_note": "LOCAL patch: the Mistral arm as the candidate"})
        log["patched"].append(f"{pre}/model/image.json → {candidate}:{run}")
    elif candidate != ri["cc"]:
        raise ConfigError(f"candidate {candidate!r}: mx, mxt or the cases run's own candidate {ri['cc']}")
    log["baseline_pin"] = pin_current_baseline(src, dst, cases_run, log)
    mirror_reference(src, dst, cp, log)
    for rel in [k for k in dst.list(f"{pre}/eval")]:  # never an earlier eval's leftovers
        dst.delete(rel)
    return log


def round_eval_cutoff(src, cases_run: str) -> str:
    """When the cases round's own eval finished: report.json `at` (eval_stage.run's last write), else
    result-eval.json `at`. The current model's baseline is pinned to its replicates up to then — what the round's
    eval read (the drift arm T appends only after the gpu-eval lock is released)."""
    pre = f"runs/{check_cases_run(cases_run)}"
    for rel in (f"{pre}/report.json", f"{pre}/result-eval.json"):
        at = (src.read_json(rel) or {}).get("at")
        if at and _parse_at(at):
            return str(at)
    raise ConfigError(f"{pre}/report.json (or result-eval.json) has no `at`: {round_info(cases_run)['label']}'s own eval "
                      "must have finished — it pins the current model's baseline")


def _parse_at(s: Any) -> datetime | None:
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def pin_baseline(b: dict, cutoff: str) -> tuple[dict, list[dict]]:
    """`b` without the replicates added after `cutoff` (a replicate without a parseable `at` is kept), recomputed
    when any was dropped. Returns (baseline, dropped [{at, run}])."""
    from souschef_lab import baseline as BL
    cut = _parse_at(cutoff)
    if cut is None:
        raise ConfigError(f"baseline pin: cutoff {cutoff!r} is not an ISO time")
    keep, dropped = [], []
    for r in b.get("replicates") or []:
        at = _parse_at(r.get("at")) if r.get("at") else None
        (dropped if at is not None and at > cut else keep).append(r)
    if not dropped:
        return b, []
    out = dict(b, replicates=keep)
    BL.recompute(out)
    out["mx_note"] = f"LOCAL pin: replicates after {cutoff} dropped ({len(dropped)})"
    return out, [{"at": r.get("at"), "run": r.get("run")} for r in dropped]


def pin_current_baseline(src, dst, cases_run: str, log: dict) -> dict:
    """The local copy's current baseline (baselines/<state/model.json digest>/baseline.json) pinned to the round's
    eval cutoff (round_eval_cutoff, pin_baseline): a drift weekly arm-T replicate appended to the live model's baseline
    after the round's eval would change every bar set's tolerance (bar.set_row → sigma_run) here but not there.
    Logs the replicates kept and dropped and the sha256 of the copy used."""
    digest = (json.loads(dst.read_bytes("state/model.json")) or {}).get("digest")
    rel = f"baselines/{digest}/baseline.json"
    raw = dst.read_bytes(rel)
    if raw is None:
        raise ConfigError(f"{rel} is missing in the local copy")
    cutoff = round_eval_cutoff(src, cases_run)
    b, dropped = pin_baseline(json.loads(raw), cutoff)
    if dropped:
        dst.write_json(rel, b)
        raw = dst.read_bytes(rel)
        log["patched"].append(f"{rel}: {len(dropped)} replicate(s) after {cutoff} dropped (pinned to the round's eval)")
    info = {"path": rel, "cutoff": cutoff, "replicates": len(b.get("replicates") or []),
            "ats": [r.get("at") for r in b.get("replicates") or []], "dropped": dropped,
            "sha256": hashlib.sha256(raw).hexdigest()}
    print(f"baseline pin: {rel} — {info['replicates']} replicate(s) up to {cutoff}, {len(dropped)} dropped, "
          f"sha256 {info['sha256'][:16]}…", file=sys.stderr)
    return info


def mirror_reference(src, dst, cp: Callable, log: dict) -> None:
    """thresholds r4_reference's inputs (r7): every listed baseline path the bucket has (eval_stage takes the first
    whose sha256 is the pinned one, and records every copy) and rows_run's eval/current/rows.jsonl. A path the
    local copy already holds (the r4 snapshot's) is kept. Not configured: nothing."""
    from souschef_lab import pipeline_common as PC
    th = PC.load_thresholds(getattr(src, "b", None))
    cfg = th.get("r4_reference")
    if not isinstance(cfg, dict):
        return
    pin = str(cfg.get("file_sha256") or "").lower()
    for rel in [str(p) for p in cfg.get("baseline") or [] if p]:
        if dst.read_bytes(rel) is None:
            cp(rel, required=False)
        b = dst.read_bytes(rel)
        if b is not None and pin and hashlib.sha256(b).hexdigest() != pin:
            log.setdefault("warnings", []).append(f"{rel}: sha256 is not the pinned r4 reference (eval_stage skips it)")
    rr = cfg.get("rows_run")
    if rr:
        cp(f"runs/{check_cases_run(str(rr))}/eval/current/rows.jsonl", required=False)


def check_pair(pair: str, cases_run: str) -> tuple[str, str]:
    pairs = pairs_for(cases_run)
    if pair not in pairs:
        raise ConfigError(f"--pair {pair}: one of {', '.join(pairs)} for {round_info(cases_run)['label']}'s round {cases_run}")
    return pairs[pair]


def check_mxt_temperature(t: float | None) -> float:
    t = MXT_TEMPERATURE if t is None else float(t)
    if not 0 <= t <= MXT_TEMPERATURE:
        raise ConfigError(f"--mxt-temperature {t}: Mistral's range, 0 … {MXT_TEMPERATURE}")
    return t


def cmd_eval(a) -> int:
    run = check_run(a.run)
    gs = GS()
    rr = resolve_cases_run(gs, run, a.cases_run)
    cur_name, cand_name = check_pair(a.pair, rr)
    per_route = {"candidate": {"temperature": check_mxt_temperature(a.mxt_temperature)}} if cand_name == "mxt" else None
    work = Path(a.workdir or os.environ.get("MX_WORKDIR") or tempfile.mkdtemp(prefix="mx-eval-"))
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    dest = f"{run_root(run)}/eval/{a.pair}"
    attempt = attempt_id()
    servers: list[Ollama] = []
    rc = 0
    t0 = time.monotonic()
    info: dict[str, Any] = {"pair": a.pair, "current": cur_name, "candidate": cand_name, "run": run, "cases_run": rr,
                            "mode": a.mode, "started_at": utc_now(), "attempt": attempt, "route_overrides": per_route,
                            "report_only": a.pair in REPORT_ONLY_PAIRS}
    base = begin_attempt(gs, dest, attempt, {"pair": a.pair, "run": run})  # a Restarted: exit 1, nothing written
    try:
        for name in (cur_name, cand_name):
            done = train_done_uri(name, run)
            if done and not gs.exists(done):
                raise ConfigError(f"{done} is missing: {name} has not finished training")
        urls = {}
        for i, (role, name, port) in enumerate((("current", cur_name, 11501), ("candidate", cand_name, 11502))):
            m = pull_model(gs, model_uri(name, run, rr), work / f"model-{role}")
            (out / f"gguf-check-{role}.json").write_text(json.dumps(m["check"], indent=1), encoding="utf-8")
            ol = Ollama(port, work / f"ollama-{role}", out / f"ollama-{role}.log", gpu=str(i))
            servers.append(ol)
            ol.start()
            ol.create(Path(m["dir"]))
            info[f"served_{role}"] = assert_served(ol.url, m["facts"], m["family"], strict_smoke=a.strict_smoke)
            info[f"model_{role}"] = {"name": name, "uri": m["uri"], "gguf": m["gguf"], "family": m["family"]}
            urls[role] = ol.url
        from souschef_lab.retrain import eval_stage
        from souschef_lab.retrain.common import open_store
        local = work / "store"
        dst = open_store(str(local))
        info["mirror"] = mirror_store(read_store(), dst, rr, current=cur_name, candidate=cand_name, run=run)
        st = dst.read_json("state/model.json") or {}
        info["current_state"] = {k: st.get(k) for k in ("tag", "digest", "training_manifest", "run_id")}
        eng = engine_for(rr, work)
        info["engine"] = {k: eng.get(k) for k in ("status", "sha", "fingerprint")}
        res = eval_stage.run(dst, rr, mode=a.mode, current_url=urls["current"], candidate_url=urls["candidate"],
                             workdir=work / "eval", engine_path=eng.get("engine"), workers=a.workers,
                             harness_factory=harness_factory(per_route), progress=say)
        info["result"] = {"status": res.get("status"), "metrics": res.get("metrics")}
        bar = dst.read_json(f"runs/{rr}/eval/bar.json") or {}
        say(f"eval {a.pair}: bar {'PASS' if bar.get('pass') else 'FAIL'}; pooled {bar.get('pooled')}")
        src_eval = local / "runs" / rr / "eval"
        if src_eval.is_dir():
            shutil.copytree(src_eval, out / "eval", dirs_exist_ok=True)
        for f in ("report.json", "decision.json", "result-eval.json"):
            p = local / "runs" / rr / f
            if p.is_file():
                shutil.copyfile(p, out / f)
        svc = work / "eval" / "harness" / "service.log"
        if svc.is_file():
            shutil.copyfile(svc, out / "service.log")
        info["logs"] = {role: log_counts([out / f"ollama-{role}.log"], None) for role in ("current", "candidate")}
        info["logs"]["service"] = log_counts([], out / "service.log")
        info["retries_500"] = {role: retries_500(info["logs"][role], info.get(f"served_{role}")) for role in ("current", "candidate")}
        rows = {role: read_jsonl_file(out / "eval" / role / "rows.jsonl") for role in ("current", "candidate")}
        if bar and rows["current"] and rows["candidate"]:
            acct = error_accounting(rows["current"], rows["candidate"], bar)
            (out / "errors.json").write_text(json.dumps(acct, indent=1), encoding="utf-8")
            info["errors"] = {k: acct[k] for k in ("errors_total", "errors_over_tol", "worse_sets", "pooled")}
        lat = eval_latency(lambda role: _read_text(out / f"ollama-{role}.log"), rows, {"current": cur_name, "candidate": cand_name})
        (out / "latency.json").write_text(json.dumps(lat, indent=1), encoding="utf-8")
        info["latency"] = {role: {"service_ms": ((lat.get(role) or {}).get("calls") or {}).get("service_ms"),
                                  "turn_ms": (((lat.get(role) or {}).get("turns") or {}).get("all") or {}).get("latency_ms")}
                           for role in ("current", "candidate")}
    except Blocked as e:
        rc = 3
        info["failed"] = {"reason": e.reason, "detail": e.detail}
        say(f"eval {a.pair}: BLOCKED ({e})")
    except BaseException as e:
        rc = getattr(e, "code", 1) if isinstance(e, MxError) else 1
        info["failed"] = {"reason": "error", "detail": f"{type(e).__name__}: {e}"[:2000]}
        if not isinstance(e, Exception):
            raise
        say(f"eval {a.pair}: FAILED ({type(e).__name__}: {e})")
    finally:
        for ol in servers:
            ol.stop()
        info["wall_s"] = round(time.monotonic() - t0)
        (out / "info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
        try:
            gs.put_dir(out, base)
            publish_attempt(gs, dest, attempt, rc, {"pair": a.pair})
        except MxError as e:
            say(f"eval {a.pair}: upload to {base} failed: {e}")
            rc = rc or 1
    return rc


def _read_text(p: Path) -> str | None:
    return Path(p).read_text(encoding="utf-8", errors="replace") if Path(p).is_file() else None


def eval_latency(log_of: Callable[[str], str | None], rows: dict[str, list[dict]], names: dict[str, str]) -> dict:
    """mx_latency.pair_latency of one eval attempt: its two Ollama logs (per call) and rows (per turn)."""
    import mx_latency
    return mx_latency.pair_latency({role: log_of(role) for role in ("current", "candidate")}, rows, names)


def read_jsonl_file(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()] if Path(p).is_file() else []


def parse_jsonl(b: bytes | None) -> list[dict]:
    return [json.loads(ln) for ln in (b or b"").decode("utf-8").splitlines() if ln.strip()]


# --- report (Stage 3 verdict) ---------------------------------------------------------------------------------------------

def _direction(b: int, c: int) -> str:
    return "candidate" if c > b else "current" if b > c else "tie"


NOISE_SIGMAS = 2.0  # the control's noise bound: max(set tol, 2σ of the eval-to-eval gap); pre-registered 2026-10-18
MODEL_NICKNAME = "chocolate-muffin"  # the Mistral arm's model (Ministral 3 8B + LoRA), beside blueberry-muffin (Qwen3.5-9B)
TRUST_RULE = ("the control pair re-runs the round's own eval here (same cases, same scorer; Vertex L4 + Ollama instead "
              "of the Cloud Run L4 services, prime off): trusted when the bar's PASS/FAIL matches, every bar set's "
              "|Δ here − Δ in the round| ≤ that set's NOISE BOUND = max(the set's tol in the round's own bar.json, 2σ of the "
              "gap between the two independent evals) — pre-registered 2026-10-18, before the next round's control ran "
              "(the strict tol alone had flagged the first control on F9H, 3.5 rows off with 2σ ≈ 13.7) — a "
              "differing bar PASS/FAIL counts only when some counted set is outside its bound, every counted "
              "set's tol here equals the round's (the same current baseline), the pooled McNemar favours the same arm, "
              "and the control's current model is the round's")


def _gap_sigma(here: dict, there: dict) -> float | None:
    """σ of (Δ here − Δ in the round) from each eval's discordant pairs (Var Δ ≈ (b + c) / m², m = replicates; the
    two evals independent). Information beside the rule, never part of it."""
    try:
        v = 0.0
        for x in (here, there):
            m = min(int(k) for k in (x.get("m") or [1])) or 1
            v += (int(x["b"]) + int(x["c"])) / (m * m)
        return round(math.sqrt(v), 2)
    except (KeyError, TypeError, ValueError):
        return None


def trust(control: dict | None, round_bar: dict | None, *, label: str = "r6", control_current: str | None = None,
          round_current: str | None = None) -> dict:
    """Does the control pair here reproduce the round's own eval/bar.json? TRUST_RULE. Every set both bars have is
    reported with its distance from the round's numbers (Δ, and each arm's score), the counted ones with their
    tolerance and slack, and 2σ of the gap (information only)."""
    if not control or not round_bar:
        return {"trusted": False, "why": f"the control pair's bar or {label}'s own eval/bar.json is missing", "label": label,
                "rule": TRUST_RULE, "sets": []}
    probs, notes = [], []
    bar_differs = bool(control.get("pass")) != bool(round_bar.get("pass"))
    bar_msg = f"bar {'PASS' if control.get('pass') else 'FAIL'} here vs {'PASS' if round_bar.get('pass') else 'FAIL'} in {label}"
    if control_current and round_current and control_current != round_current:
        probs.append(f"the control's current model is {control_current[:19]}…, {label}'s eval ran against {round_current[:19]}…")
    ctrl = {s["set"]: s for s in control.get("sets") or []}
    per = []
    for s in round_bar.get("sets") or []:
        counts = bool(s.get("bar"))
        c = ctrl.get(s["set"])
        if c is None:
            if counts:
                probs.append(f"{s['set']} not measured here")
            per.append({"set": s["set"], "counts": counts, "measured": False})
            continue
        gap = round(float(c["delta"]) - float(s["delta"]), 2)
        tol = float(s.get("tol") or 0)
        sg = _gap_sigma(c, s)
        bound = max(tol, NOISE_SIGMAS * sg) if sg is not None else tol  # the pre-registered noise bound
        ok = abs(gap) <= bound if counts else None
        same_tol = c.get("tol") is not None and s.get("tol") is not None and float(c["tol"]) == tol
        if counts and not same_tol:  # the bar here was sized from a different current baseline than the round's
            probs.append(f"{s['set']} tol {c.get('tol')} here vs {s.get('tol')} in {label} (current baseline differs)")
        per.append({"set": s["set"], "counts": counts, "measured": True, "n": [s.get("n"), c.get("n")],
                    "round_delta": s["delta"], "here_delta": c["delta"], "gap": gap, "tol": s.get("tol"),
                    "tol_here": c.get("tol"), "same_tol": same_tol,
                    "bound": round(bound, 1), "slack": round(bound - abs(gap), 2) if counts else None, "ok": ok,
                    "d_current": round(float(c["current"]) - float(s["current"]), 2),
                    "d_candidate": round(float(c["candidate"]) - float(s["candidate"]), 2),
                    "gap_2sigma": None if sg is None else round(2 * sg, 1),
                    # kept for readers of the r6 report.json
                    "r6_delta" if label == "r6" else f"{label}_delta": s["delta"]})
        if counts and not ok:
            probs.append(f"{s['set']} Δ {c['delta']:+.1f} here vs {s['delta']:+.1f} in {label} (gap {gap:+.1f}, bound {bound:.1f} "
                         f"= max(tol {s['tol']}, 2σ {'?' if sg is None else round(NOISE_SIGMAS * sg, 1)}))")
    if bar_differs:
        if any(x.get("counts") and x.get("measured") and not x.get("ok") for x in per):
            probs.append(bar_msg)
        else:
            notes.append(f"{bar_msg}, with every counted set inside its noise bound (a verdict within noise)")
    pc, pr = control.get("pooled") or {}, round_bar.get("pooled") or {}
    d_here, d_round = _direction(int(pc.get("b", 0)), int(pc.get("c", 0))), _direction(int(pr.get("b", 0)), int(pr.get("c", 0)))
    if d_here != d_round:
        probs.append(f"pooled McNemar favours {d_here} here, {d_round} in {label}")
    return {"trusted": not probs, "why": "; ".join(probs) or f"reproduces {label}'s own eval within noise", "notes": notes, "sets": per,
            "direction": {"here": d_here, label: d_round}, "label": label, "rule": TRUST_RULE,
            "current": {"control": control_current, "round": round_current}}


NONINFERIORITY = ("sets", "mcnemar", "oos_where", "f0h_unseen", "ghosts", "r4_floor")  # bar.py's rules but the trigger
# r4_floor (r7): the candidate against r4's RECORDED answers, gating in the round's own bar (thresholds r4_reference.bar);
# a bar without it (r6's round) or with it NOT MEASURED passes it. r4_rerun_floor stays report only, as in the round.
# The trigger counts unless the current model is the round's own candidate (it already learned the trigger's fix).


def _bar_set_of(r: dict) -> str:
    s = str(r.get("set") or "")
    return ("F0H:seen" if r.get("seen") else "F0H:unseen") if s == "F0H" else s


def _occurrences(rows: list[dict]) -> dict[str, dict[str, list[tuple[int, bool]]]]:
    """{bar set: {row key: [(pass, errored) per replicate]}} of one arm's eval rows.jsonl (eval_stage writes a row
    once per replicate, replicate 1 first; error rows included)."""
    out: dict[str, dict[str, list[tuple[int, bool]]]] = {}
    for r in rows:
        out.setdefault(_bar_set_of(r), {}).setdefault(str(r.get("key")), []).append(
            (int(r.get("pass") or 0), r.get("status") != "ok"))
    return out


def error_accounting(cur_rows: list[dict], cand_rows: list[dict], bar: dict) -> dict:
    """The bar's sets recounted with ERROR rows as fails for both arms (bar.common leaves a row out of both arms when
    either errored, and eval_stage tolerates up to max(3, n//5) error rows a set — for MX a parser error is the
    model's answer, as Stage 1 scores it). Per bar set: error rows per replicate and arm, Δ with errors as fails, the
    candidate's excess errors against the set's tol; pooled b/c/p over (row, replicate) pairs."""
    from souschef_lab.examcmp import mcnemar_p
    oc, ok_ = _occurrences(cur_rows), _occurrences(cand_rows)
    sets, b_all, c_all = [], 0, 0
    for s in bar.get("sets") or []:
        if not s.get("bar"):
            continue
        name = s["set"]
        cur, cand = oc.get(name) or {}, ok_.get(name) or {}
        m_cur = max((len(v) for v in cur.values()), default=0)
        m_cand = max((len(v) for v in cand.values()), default=0)
        if not m_cur or not m_cand:
            sets.append({"set": name, "measured": False})
            continue
        keys = sorted(set(cur) | set(cand))
        pc = sum(p for v in cur.values() for p, _ in v) / m_cur
        pk = sum(p for v in cand.values() for p, _ in v) / m_cand
        ec = sum(e for v in cur.values() for _, e in v) / m_cur
        ek = sum(e for v in cand.values() for _, e in v) / m_cand
        b = c = 0
        for k in range(min(m_cur, m_cand)):
            for key in keys:
                x = cur.get(key, [])[k][0] if k < len(cur.get(key, [])) else 0
                y = cand.get(key, [])[k][0] if k < len(cand.get(key, [])) else 0
                b += int(x == 1 and y == 0)
                c += int(x == 0 and y == 1)
        b_all, c_all = b_all + b, c_all + c
        tol = float(s.get("tol") or 0)
        d = pk - pc
        sets.append({"set": name, "measured": True, "n": len(keys), "tol": s.get("tol"),
                     "errors": {"current": round(ec, 2), "candidate": round(ek, 2)}, "delta_errors_as_fails": round(d, 2),
                     "delta_bar": s.get("delta"), "b": b, "c": c,
                     "worse": d < -tol, "errors_over_tol": ek - ec > tol})
    p = mcnemar_p(b_all, c_all)
    return {"sets": sets, "pooled": {"b": b_all, "c": c_all, "mcnemar_p": round(p, 4)},
            "worse_sets": [x["set"] for x in sets if x.get("worse")],
            "errors_over_tol": [x["set"] for x in sets if x.get("errors_over_tol")],
            "errors_total": {"current": round(sum(x["errors"]["current"] for x in sets if x.get("measured")), 2),
                             "candidate": round(sum(x["errors"]["candidate"] for x in sets if x.get("measured")), 2)},
            "note": "error rows counted as fails for both arms; errors per replicate"}


def verdict(bar: dict | None, *, trusted: bool, counts_trigger: bool = True, acct: dict | None = None,
            retries: dict | None = None, label: str = "r6") -> dict:
    """MX (candidate) against the pair's current model (docs/MISTRAL.md §5.3). On par / better / worse come from the
    non-inferiority rules (NONINFERIORITY, r4_floor included when the bar has it) plus the pooled McNemar —
    recounted with error rows as fails when `acct` (error_accounting) is given, with an `errors` rule on top; the
    trigger rule counts unless `counts_trigger` is False (against the round's own candidate: reported beside it). `retries` = {"current": n, "candidate": n} /v1/messages 500s (hidden SDK retries):
    a better / on-par that the excess retries could explain (each rescuing one (row, replicate) pair) is
    inconclusive."""
    if not bar:
        return {"verdict": "missing", "why": "no bar.json"}
    if not trusted:
        return {"verdict": "untrusted", "why": f"the control pair did not reproduce {label}'s own eval"}
    from souschef_lab.examcmp import mcnemar_p
    rules = {r["name"]: r for r in bar.get("rules") or []}
    failing = [n for n in NONINFERIORITY if n in rules and not rules[n].get("pass")]
    trig = rules.get("trigger") or {}
    if counts_trigger and trig and not trig.get("pass"):
        failing.append("trigger")
    margins: dict[str, float] = {}
    for s in bar.get("sets") or []:
        if s.get("bar") and float(s["delta"]) < -float(s["tol"]):
            margins[s["set"]] = round(-float(s["delta"]) - float(s["tol"]), 2)
    pooled = bar.get("pooled") or {}
    b, c, p = int(pooled.get("b", 0)), int(pooled.get("c", 0)), float(pooled.get("mcnemar_p", 1.0))
    if acct:
        for x in acct.get("sets") or []:
            if x.get("worse"):
                m = round(-float(x["delta_errors_as_fails"]) - float(x["tol"]), 2)
                margins[x["set"]] = max(margins.get(x["set"], m), m)
                if "sets" not in failing:
                    failing.insert(0, "sets")
        if acct.get("errors_over_tol"):
            failing.append("errors")
        ap = acct.get("pooled") or {}
        b, c, p = int(ap.get("b", b)), int(ap.get("c", c)), float(ap.get("mcnemar_p", p))
        if b > c and p < 0.05 and "mcnemar" not in failing:
            failing.append("mcnemar")
    base = {"b": b, "c": c, "p": p, "failing": failing, "pooled_from": "errors as fails" if acct else "bar",
            "trigger": {"pass": trig.get("pass"), "detail": trig.get("detail"), "counted": counts_trigger}}
    if acct:
        base["errors"] = {"total": acct.get("errors_total"), "over_tol": acct.get("errors_over_tol")}
    e = 0
    if retries:
        e = max(0, int(retries.get("candidate") or 0) - int(retries.get("current") or 0))
        base["retries_500"] = dict(retries, excess=e)
    if P_GREY[0] <= p < P_GREY[1]:
        return {"verdict": "inconclusive", "why": f"pooled McNemar p {p:.3f} in [{P_GREY[0]}, {P_GREY[1]}): train seed 2 first", **base}
    if not failing:
        v = "better" if c > b and p < P_GREY[0] else "on_par"
        why = (f"non-inferior on every rule; McNemar favours MX (b {b} / c {c}, p {p:.3f})" if v == "better" else
               f"non-inferior on every rule without a significant difference (b {b} / c {c}, p {p:.3f})")
        if e:
            bw, cw = b + e, max(0, c - e)
            pw = mcnemar_p(bw, cw)
            if (v == "better" and not (cw > bw and pw < P_GREY[0])) or (v == "on_par" and bw > cw and pw < P_GREY[0]):
                return {"verdict": "inconclusive", "why": f"{why} — but MX drew {e} more parser-error 500s the service "
                        f"retried; if each rescued a pair it is b {bw} / c {cw}, p {pw:.3f}: look at the rows first", **base}
        return {"verdict": v, "why": why + ("" if counts_trigger else f"; trigger (reported, not counted): {trig.get('detail')}"),
                **base}
    if set(failing) <= {"sets"} and margins and max(margins.values()) <= SEED2_ROWS:
        short = ", ".join(f"{k} {v}" for k, v in margins.items())
        return {"verdict": "inconclusive", "why": f"fails only on set tolerance, by ≤ {SEED2_ROWS} rows ({short}): "
                "a reroll can flip it — train seed 2 first", **base}
    return {"verdict": "worse", "why": f"fails {', '.join(failing)}", **base}


def job_costs(jobs: list[dict]) -> dict:
    """USD estimates from `gcloud ai custom-jobs list --filter=labels.purpose=sous-chef-mx --format=json`."""
    rows, total = [], 0.0
    for j in jobs or []:
        try:
            st = datetime.fromisoformat(str(j["startTime"]).replace("Z", "+00:00"))
            en = datetime.fromisoformat(str(j.get("endTime") or utc_now()).replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        h = max(0.0, (en - st).total_seconds() / 3600)
        lab = j.get("labels") or {}
        ms = (((j.get("jobSpec") or {}).get("workerPoolSpecs") or [{}])[0].get("machineSpec") or {})
        strat = ((j.get("jobSpec") or {}).get("scheduling") or {}).get("strategy") or "STANDARD"
        rate = next((r["usd_h"] for r in RUNGS.values() if r["machineType"] == ms.get("machineType") and r["strategy"] == strat), None)
        if rate is None:
            rate = next((m["usd_h"] for m in MACHINES.values() if m["machineType"] == ms.get("machineType")), 0.0)
        usd = round(h * rate, 2)
        total += usd
        rows.append({"job": j.get("displayName"), "stage": lab.get("stage") or lab.get("rung"), "machine": ms.get("machineType"),
                     "strategy": strat, "hours": round(h, 2), "usd_est": usd, "state": j.get("state")})
    return {"jobs": rows, "usd_est_total": round(total, 2), "note": "estimates (compute.md rates); the billing report is the truth"}


def build_report(*, run: str, cases_run: str, gate_doc: dict | None, exams: dict, train: dict, evals: dict,
                 round_bar: dict | None, costs: dict | None, base_gguf: dict | None, round_report: dict | None = None,
                 round_train_config: dict | None = None) -> dict:
    ri = round_info(cases_run)
    label, cc, ctrl = ri["label"], ri["cc"], control_pair(cases_run)
    pairs = pairs_for(cases_run)
    ctl_info = (evals.get(ctrl) or {}).get("info") or {}
    tr = trust((evals.get(ctrl) or {}).get("bar"), round_bar, label=label,
               control_current=(ctl_info.get("current_state") or {}).get("digest"),
               round_current=(((round_report or {}).get("results") or {}).get("current") or {}).get("digest"))

    def v(p: str) -> dict:
        e = evals.get(p) or {}
        return verdict(e.get("bar"), trusted=tr["trusted"], counts_trigger=pairs[p][0] != cc, acct=e.get("errors"),
                       retries=e.get("retries_500"), label=label)
    verdicts = {p: v(p) for p in main_pairs(cases_run)}
    report_only = {p: dict(v(p), report_only=True) for p in REPORT_ONLY_PAIRS if (evals.get(p) or {}).get("bar")}
    run_json = train.get("run") or {}
    arms = (round_train_config or {}).get("arms")
    carried = (gate_doc or {}).get("carried_from")
    caveats = [f"one seed (base:3407); {label} kept the best of its arms ({', '.join(arms) if arms else 'see its train-config.json'})",
               f"no warm start for MX (the r4/{label} adapters are Qwen3.5 LoRAs)",
               f"temperature 0.3 on every main arm (Mistral recommends ≤ {MXT_TEMPERATURE}: r4-mxt, report only, runs MX there)",
               "prime off on every arm (calibrated for Qwen3.5's hybrid attention)",
               "Vertex L4 + Ollama 0.34.4 in one container, not the Cloud Run L4 services; latency has 2 eval workers "
               "sharing each arm's one slot (wall includes the queue; service time does not)",
               "recipe (lr 2e-4, r16/α16) tuned on Qwen3.5, not re-tuned",
               "top_k 20 / top_p 0.95 are Qwen3.5's recommended sampling, applied to Mistral too",
               "Qwen's parser coerces argument types from the schema (the string \"5\" → 5); Ministral's does not, "
               "so a mistyped argument is a miss for MX only",
               "parser errors: Ollama answers an unknown tool name or bad JSON from MinistralParser with HTTP 500 "
               "and the service's SDK retries a 5xx twice (fresh samples at 0.3) — hidden extra draws Qwen's "
               "parser never asks for; counted per arm as retries_500 (Stage 1 gate, Stage 3 verdicts)",
               "Stage 3 verdicts count error rows as fails for both arms (bar.json leaves them out of both)"]
    if carried:
        caveats.append(f"Stage 1 carried from {carried} (its exams were scored on the answer keys of that day; r7 moved "
                       "h006/h022/p030/p047 to .k3 — Stage 3 re-scores every arm on today's keys)")
    return {"run": run, "at": utc_now(), "cases_run": cases_run, "round": {"label": label, "candidate": cc, "control_pair": ctrl,
                                                                         "current": ri["current"]},
            "model": {"repo": MISTRAL_REPO, "revision": MISTRAL_REV, "license": "Apache-2.0",
                      "base_gguf": (base_gguf or {}).get("mx-base"), "export": train.get("export")},
            "stage1": gate_doc, "stage1_exams": {a: {k: (exams.get(a) or {}).get(k) for k in ("pooled", "sets", "latency_ms", "logs")}
                                                   for a in STAGE1_ARMS if exams.get(a)},
            "training": {k: run_json.get(k) for k in ("rows", "tokens_estimate", "updates", "minutes", "train_seconds",
                                                      "train_tokens_per_second", "peak_vram_gb", "first_loss", "last_loss",
                                                      "start_eval_loss", "last_eval_loss", "eval_losses", "gen_check", "gpu",
                                                      "versions", "family", "revision")} | {"progress": train.get("progress")},
            "stage3": {"control_pair": ctrl, "control": tr,
                       "pairs": {p: {k: (evals.get(p) or {}).get(k) for k in ("bar", "info", "errors", "retries_500")}
                                 for p in pairs},
                       "verdicts": verdicts, "report_only": report_only},
            "latency": {p: (evals.get(p) or {}).get("latency") for p in pairs if (evals.get(p) or {}).get("latency")},
            "cost": costs, "caveats": caveats}


def _bar_lines(bar: dict | None) -> list[str]:
    if not bar:
        return ["(missing)"]
    lines = ["| set | n | current | candidate | Δ | tol | b/c | p | bar |", "|---|---|---|---|---|---|---|---|---|"]
    for s in bar.get("sets") or []:
        lines.append(f"| {s['set']} | {s['n']} | {s['current']} | {s['candidate']} | {s['delta']:+} | {s['tol']} | "
                     f"{s.get('b')}/{s.get('c')} | {s.get('mcnemar_p')} | {'yes' if s.get('bar') else ''} |")
    pooled = bar.get("pooled") or {}
    lines.append(f"\nBar: **{'PASS' if bar.get('pass') else 'FAIL'}** — pooled b {pooled.get('b')} / c {pooled.get('c')}, "
                 f"p {pooled.get('mcnemar_p')}. Rules: " + "; ".join(f"{r['name']} {'ok' if r['pass'] else 'FAIL'} ({r['detail']})"
                                                                     for r in bar.get("rules") or []))
    return lines


def _num(x: Any, fmt: str = "+.1f") -> str:
    return format(x, fmt) if isinstance(x, (int, float)) else "–"


def _tol_cell(x: dict) -> str:
    if not x.get("counts"):
        return ""
    return str(x.get("tol")) if x.get("same_tol", True) else f"{x.get('tol')} **(here {x.get('tol_here')})**"


def trust_lines(tr: dict, ctrl: str) -> list[str]:
    """The control's distance from the round's own numbers, set by set, with the tolerance that decides trust."""
    label = tr.get("label") or "the round"
    out = [f"Control ({ctrl}) trusted: **{tr.get('trusted')}** — {tr.get('why')}", "",
           f"Rule: {tr.get('rule') or TRUST_RULE}.", "",
           f"| set | counts | Δ in {label} | Δ here | gap | tol | 2σ gap | bound | slack | current here − {label} | candidate here − {label} |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in tr.get("sets") or []:
        if not x.get("measured"):
            out.append(f"| {x['set']} | {'yes' if x.get('counts') else ''} | not measured here | | | | | | | | |")
            continue
        flag = "" if not x.get("counts") else (" ok" if x.get("ok") else " **OUT**")
        out.append(f"| {x['set']} | {'yes' if x.get('counts') else ''} | {_num(x.get('round_delta'))} | {_num(x.get('here_delta'))} | "
                   f"{_num(x.get('gap'))}{flag} | {_tol_cell(x)} | {_num(x.get('gap_2sigma'), '.1f')} | "
                   f"{_num(x.get('bound'), '.1f')} | {_num(x.get('slack'), '.1f') if x.get('counts') else ''} | "
                   f"{_num(x.get('d_current'))} | {_num(x.get('d_candidate'))} |")
    return out + [f"Note: {n}" for n in tr.get("notes") or []] + [""]


def report_md(rep: dict) -> str:
    g = rep.get("stage1") or {}
    s3 = rep.get("stage3") or {}
    rd = rep.get("round") or {"label": "r6", "candidate": "r6c", "control_pair": "r4-r6c"}
    label, cc, ctrl = rd["label"], rd["candidate"], rd["control_pair"]
    verdicts = s3.get("verdicts") or {}
    v = verdicts.get(f"{cc}-mx") or {}
    out = [f"# {MODEL_NICKNAME} (the Mistral arm) `{rep['run']}` — {label}'s round", "",
           f"**Verdict vs the {label} candidate ({cc}): {str(v.get('verdict', 'missing')).upper()}** — {v.get('why', '')}", "",
           f"Cases run `{rep.get('cases_run')}` ({label}); the control pair {ctrl} re-runs {label}'s own eval.", "",
           "## Model", "", f"**{MODEL_NICKNAME}**: `{rep['model']['repo']}` @ `{rep['model']['revision']}` ({rep['model']['license']}) "
           f"+ a LoRA on the round's data — the Mistral counterpart of blueberry-muffin (Qwen3.5-9B).", "",
           "## Stage 1 (untrained bases, 2 replicates of the 252 real-phrasing rows)", ""]
    if g.get("carried_from"):
        out += [f"Carried from `{g['carried_from']}` (its gate.json of {g.get('at')}; the bases are the same).", ""]
    out += [f"Verdict **{g.get('verdict')}**: {g.get('why')}", "",
            f"Δ (mx − qwen) {g.get('delta')} rows, σ_Δ {g.get('sigma_delta')}, Δ_fmt {g.get('delta_fmt')}, "
            f"flips {g.get('flips')}, format failures {g.get('format_failures')}.", "",
            f"/v1/messages 500s the service retried (hidden extra draws): {g.get('retries_500')}.", ""]
    if g.get("native"):
        out += [f"mx-base-native (diagnostic): {g['native']}", ""]
    if g.get("r4_anchor"):
        out += [f"r4 anchor: {g['r4_anchor']}", ""]
    t = rep.get("training") or {}
    out += [f"## Training (one arm, base:3407, {label}'s dataset)", "",
            f"rows {t.get('rows')}; updates {t.get('updates')}; tokens/s {t.get('train_tokens_per_second')}; "
            f"peak {t.get('peak_vram_gb')} GB; loss {t.get('first_loss')} → {t.get('last_loss')}; "
            f"validation {t.get('start_eval_loss')} → {t.get('last_eval_loss')}; gen-check {t.get('gen_check')}", "",
            "## Stage 3 (paired evals, Vertex 2 × L4)", ""]
    out += trust_lines(s3.get("control") or {}, ctrl)
    order = [p for p in [f"{cc}-mx", "r4-mx", ctrl] if p in (s3.get("pairs") or {})]
    for p in order:
        out += pair_lines(p, (s3.get("pairs") or {}).get(p) or {})
    r4v = verdicts.get("r4-mx") or {}
    out += [f"r4 vs MX: **{r4v.get('verdict')}** — {r4v.get('why')}", ""]
    out += [f"{cc}-mx trigger rule (reported, not part of the verdict): {(v.get('trigger') or {}).get('detail')}", ""]
    ro = s3.get("report_only") or {}
    if ro:
        out += ["## Report only (never part of a verdict)", ""]
        for p, vv in ro.items():
            info = ((s3.get("pairs") or {}).get(p) or {}).get("info") or {}
            temp = ((info.get("route_overrides") or {}).get("candidate") or {}).get("temperature")
            out += [f"### {p} — MX at temperature {temp if temp is not None else MXT_TEMPERATURE} vs r4 (0.3)", "",
                    f"Would read: **{vv.get('verdict')}** — {vv.get('why')}", ""]
            out += pair_lines(p, (s3.get("pairs") or {}).get(p) or {}, header=False)
            out += temperature_lines((s3.get("pairs") or {}).get("r4-mx") or {}, (s3.get("pairs") or {}).get(p) or {}, p)
    lat = rep.get("latency") or {}
    if lat:
        import mx_latency
        out += [mx_latency.markdown(lat)]
    if rep.get("cost"):
        out += ["## Cost (estimates)", "", f"US${rep['cost'].get('usd_est_total')} — {rep['cost'].get('note')}", ""]
    out += ["## Caveats", ""] + [f"- {c}" for c in rep.get("caveats") or []]
    return "\n".join(out) + "\n"


def pair_lines(p: str, pp: dict, header: bool = True) -> list[str]:
    out = ([f"### {p}", ""] if header else []) + _bar_lines(pp.get("bar")) + [""]
    acct = pp.get("errors") or {}
    if acct:
        out += [f"Errors as fails: pooled b {(acct.get('pooled') or {}).get('b')} / c {(acct.get('pooled') or {}).get('c')}, "
                f"p {(acct.get('pooled') or {}).get('mcnemar_p')}; error rows per replicate {acct.get('errors_total')}; "
                f"sets worse with errors as fails {acct.get('worse_sets') or 'none'}; candidate errors over tol "
                f"{acct.get('errors_over_tol') or 'none'}.", ""]
        out += ["| set | errors cur | errors cand | Δ errors-as-fails | Δ bar | tol |", "|---|---|---|---|---|---|"]
        out += [f"| {x['set']} | {x['errors']['current']} | {x['errors']['candidate']} | {x['delta_errors_as_fails']:+} | "
                f"{x.get('delta_bar')} | {x.get('tol')} |" for x in acct.get("sets") or [] if x.get("measured")] + [""]
    if pp.get("retries_500") is not None:
        out += [f"/v1/messages 500s retried by the service: {pp['retries_500']}.", ""]
    return out


def temperature_lines(main: dict, low: dict, p: str) -> list[str]:
    """r4-mx beside r4-mxt, set by set: what Mistral's temperature changes (r4 is the current arm of both)."""
    a = {s["set"]: s for s in (main.get("bar") or {}).get("sets") or []}
    b = {s["set"]: s for s in (low.get("bar") or {}).get("sets") or []}
    names = [s for s in a if s in b]
    if not names:
        return []
    out = [f"| set | MX Δ vs r4 at 0.3 (r4-mx) | MX Δ vs r4 low temperature ({p}) | difference |", "|---|---|---|---|"]
    for n in names:
        out.append(f"| {n} | {_num(a[n].get('delta'))} | {_num(b[n].get('delta'))} | "
                   f"{_num(float(b[n]['delta']) - float(a[n]['delta']))} |")
    return out + [""]


def resolve_cases_run(gs: GS, run: str, given: str | None) -> str:
    """--cases-run when given, else the run's plan.json (prep-run wrote it), else DEFAULT_CASES_RUN."""
    if given:
        return check_cases_run(given)
    try:
        plan = gs.read_json(f"{run_root(run)}/plan.json") or {}
    except (MxError, ValueError):
        plan = {}
    return check_cases_run(plan.get("cases_run") or DEFAULT_CASES_RUN)


def read_pair(gs: GS, root: str, p: str, names: tuple[str, str], *, latency: bool = True) -> dict:
    """One pair's results (following latest.json): bar, info, error accounting, retries and latency (the attempt's
    latency.json, else recomputed from its Ollama logs and rows: attempts older than the latency section)."""
    d = f"{root}/eval/{p}"
    bar = gs.read_json(results_uri(gs, d, "eval/bar.json"))
    info = gs.read_json(results_uri(gs, d, "info.json")) or {}
    rows = {role: parse_jsonl(gs.cat(results_uri(gs, d, f"eval/{role}/rows.jsonl"))) for role in ("current", "candidate")}
    acct = (error_accounting(rows["current"], rows["candidate"], bar) if bar and rows["current"] and rows["candidate"]
            else gs.read_json(results_uri(gs, d, "errors.json")))
    # recounted from the logs: attempts before the GIN fix recorded 0 for every arm
    logs: dict[str, str | None] = {}
    if bar:
        for role in ("current", "candidate"):
            b = gs.cat(results_uri(gs, d, f"ollama-{role}.log"))
            logs[role] = None if b is None else b.decode("utf-8", "replace")
    retries = info.get("retries_500")
    if any(logs.values()):
        retries = {role: retries_500(log_counts_text(logs.get(role) or ""), info.get(f"served_{role}"))
                   for role in ("current", "candidate")}
    lat = None
    if latency and bar:
        lat = gs.read_json(results_uri(gs, d, "latency.json"))
        if lat is None and any(logs.values()):
            lat = eval_latency(lambda role: logs.get(role), rows, {"current": names[0], "candidate": names[1]})
    return {"bar": bar, "info": info, "errors": acct, "retries_500": retries, "latency": lat}


def log_counts_text(text: str) -> dict:
    """log_counts' ollama_status of one Ollama log's text."""
    status: dict[str, dict[str, int]] = {}
    for ln in text.splitlines():
        m = GIN.search(ln)
        if m:
            d = status.setdefault(m.group(2), {})
            d[m.group(1)] = d.get(m.group(1), 0) + 1
    return {"ollama_status": status}


def cmd_report(a) -> int:
    gs = GS()
    root = run_root(a.run)
    rr = resolve_cases_run(gs, a.run, a.cases_run)
    exams = {arm: read_exam(gs, root, arm, "summary.json") for arm in STAGE1_ARMS}
    train = {"run": gs.read_json(f"{root}/train/adapter/run.json"), "progress": gs.read_json(f"{root}/train/progress.json"),
             "export": gs.read_json(f"{root}/train/export/export.json")}
    evals = {p: read_pair(gs, root, p, names, latency=not a.no_latency) for p, names in pairs_for(rr).items()}
    jobs = json.loads(Path(a.jobs_json).read_text(encoding="utf-8")) if a.jobs_json else None
    rep = build_report(run=a.run, cases_run=rr, gate_doc=gs.read_json(f"{root}/gate.json"), exams=exams, train=train,
                       evals=evals, round_bar=gs.read_json(f"{BUCKET}/runs/{rr}/eval/bar.json"),
                       round_report=gs.read_json(f"{BUCKET}/runs/{rr}/report.json"),
                       round_train_config=gs.read_json(f"{BUCKET}/runs/{rr}/train-config.json"),
                       costs=job_costs(jobs) if jobs else None,
                       base_gguf={arm: gs.read_json(f"{root}/models/{arm}/gguf-check.json") for arm in BASES})
    md = report_md(rep)
    if not a.no_write:
        gs.put_json(rep, f"{root}/report/report.json")
        gs.put_bytes(md.encode("utf-8"), f"{root}/report/report.md", content_type="text/markdown")
    print(md)
    return 0


def cmd_latency(a) -> int:
    """The latency section of a run's Stage 3 pairs, printed (reads only; --json FILE keeps the numbers locally)."""
    import mx_latency
    gs = GS()
    root = run_root(a.run)
    rr = resolve_cases_run(gs, a.run, a.cases_run)
    lat = {}
    for p, names in pairs_for(rr).items():
        r = read_pair(gs, root, p, names)
        if r.get("latency"):
            lat[p] = r["latency"]
    if a.json:
        Path(a.json).write_text(json.dumps({"run": a.run, "cases_run": rr, "pairs": lat,
                                            "by_model": mx_latency.by_model(lat)}, indent=1), encoding="utf-8")
    print(mx_latency.markdown(lat))
    return 0


# --- base-gguf (prep, CPU, trainer-mx image) ----------------------------------------------------------------------------

def _sha256_ok(tar: Path, sha_file: Path) -> bool:
    want = sha_file.read_text(encoding="utf-8").split()[0].strip().lower()
    return mx_gguf.file_sha256(tar) == want


def untar_base(tar: Path, hf_home: Path) -> Path:
    """As pipeline-train.sh: a hub-cache folder (models--org--name/…) goes under $HF_HOME/hub; a tar starting with
    hub/ goes to $HF_HOME."""
    with tarfile.open(tar) as tf:
        first = next(iter(tf.getnames()), "")
        dest = hf_home if first.startswith("hub/") else hf_home / "hub"
        dest.mkdir(parents=True, exist_ok=True)
        for m in tf.getmembers():
            p = (dest / m.name).resolve()
            if not str(p).startswith(str(dest.resolve())):
                raise ConfigError(f"{tar.name}: member {m.name} escapes the cache folder")
        tf.extractall(dest, filter="data")
    return hf_home / "hub"


def cmd_base_gguf(a) -> int:
    from souschef_lab.export import cached_snapshot, student_modelfile
    run = check_run(a.run)
    gs = GS()
    work = Path(a.workdir or os.environ.get("MX_WORKDIR") or "/work/mx-base-gguf")
    work.mkdir(parents=True, exist_ok=True)
    llama = Path(os.environ.get("UNSLOTH_LLAMA_CPP_PATH") or "/opt/llama.cpp")
    py = sys.executable
    hf_home = Path(os.environ.get("HF_HOME") or work / "hf")
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for arm in arms:
        if arm not in BASES:
            raise ConfigError(f"--arms: {arm} is not one of {', '.join(BASES)}")
    attempt = attempt_id()
    jobdir = f"{run_root(run)}/base-gguf"
    begin_attempt(gs, jobdir, attempt, {"arms": arms, "converter_ref": a.converter_ref})  # a Restarted: exit 1
    rc = 0
    for arm in arms:
        cfg = BASES[arm]
        dest = f"{run_root(run)}/models/{arm}"
        aw = work / arm
        shutil.rmtree(aw, ignore_errors=True)
        (aw / "out").mkdir(parents=True)
        rec: dict[str, Any] = {"arm": arm, "repo": cfg["repo"], "revision": cfg["revision"], "tar": cfg["tar"],
                               "started_at": utc_now(), "converter": None}
        t0 = time.monotonic()
        try:
            base_mf = LAB_DIR / cfg["modelfile"]
            if not base_mf.is_file():
                raise ConfigError(f"{base_mf} is missing (the renderer change lands it)")
            tar = aw / Path(cfg["tar"]).name
            say(f"{arm}: pulling {cfg['tar']}")
            if not gs.download(cfg["tar"], tar) or not gs.download(cfg["tar"] + ".sha256", aw / (tar.name + ".sha256")):
                raise ConfigError(f"{cfg['tar']} (+ .sha256) is missing")
            if not _sha256_ok(tar, aw / (tar.name + ".sha256")):
                raise MxError(f"{tar.name}: sha256 does not match its .sha256")
            hub = untar_base(tar, hf_home)
            tar.unlink()
            snap = cached_snapshot(cfg["repo"], hub)
            if snap is None:
                raise ConfigError(f"no {cfg['repo']} snapshot with config.json + safetensors under {hub}")
            if cfg["revision"] and snap.name != cfg["revision"]:
                raise ConfigError(f"{cfg['repo']} snapshot {snap.name} is not the pinned revision {cfg['revision']}")
            rec["snapshot"] = snap.name
            conv = llama / "convert_hf_to_gguf.py"
            env = dict(os.environ)
            if a.converter_ref:
                conv = fetch_converter(a.converter_ref, work)
                env.pop("NO_LOCAL_GGUF", None)
            meta = llama / "SOUS_CHEF_LLAMA_CPP.json"
            rec["converter"] = {"script": str(conv), "ref": a.converter_ref or
                                (json.loads(meta.read_text()) if meta.is_file() else None)}
            f16 = aw / f"{arm}-f16.gguf"
            q4 = aw / "export" / f"{arm}-{QUANT}.gguf"
            q4.parent.mkdir(parents=True)
            for cmd in ([py, str(conv), str(snap), "--outfile", str(f16), "--outtype", "f16"],
                        [str(llama / "llama-quantize"), str(f16), str(q4), "Q4_K_M"]):
                say(f"{arm}: $ {' '.join(cmd)}")
                r = subprocess.run(cmd, env=env, check=False)
                if r.returncode != 0:
                    raise MxError(f"{Path(cmd[1] if cmd[0] == py else cmd[0]).name} exited {r.returncode}")
            f16.unlink()
            doc = mx_gguf.check_file(q4, cfg["family"], QUANT, sha256=True)
            (aw / "out" / "gguf-check.json").write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
            if not doc["ok"]:
                raise Blocked("gguf", "; ".join(f"{c['name']}: {c['detail']}" for c in doc["checks"] if not c["ok"]))
            (q4.parent / f"Modelfile.{QUANT}").write_text(
                student_modelfile(base_mf.read_text(encoding="utf-8"), q4.name, cfg["tag"], 65536), encoding="utf-8")
            stage = aw / "stage"
            r = subprocess.run(["make", "--no-print-directory", "-C", str(LAB_DIR), "model-stage", f"NAME={arm}",
                                f"MODEL_EXPORT={q4.parent}", f"MODEL_STAGE={stage}", f"MODEL_TEMPERATURE={STAGE_TEMPERATURE}",
                                f"MODEL_PRESENCE_PENALTY={STAGE_PRESENCE}"], check=False)
            if r.returncode != 0:
                raise MxError(f"make model-stage exited {r.returncode}")
            want = staged_modelfile(base_mf.read_text(encoding="utf-8"), q4.name, cfg["tag"])
            got = (stage / "Modelfile").read_text(encoding="utf-8")
            if got != want:
                raise MxError("the staged Modelfile differs from staged_modelfile() (model-stage changed?)")
            rec.update(gguf=q4.name, sha256=doc.get("sha256"), size=doc["size"], minutes=round((time.monotonic() - t0) / 60, 1))
            gs.put_dir(stage, dest)
            say(f"{arm}: staged to {dest}/ ({doc['size'] / 1e9:.2f} GB)")
            shutil.rmtree(snap.parent.parent, ignore_errors=True)  # the HF cache folder of this base: disk for the next
        except MxError as e:
            rc = rc or (e.code if not isinstance(e, Blocked) else 3)
            rec["failed"] = {"reason": getattr(e, "reason", "error"), "detail": str(e)[:2000]}
            say(f"{arm}: FAILED ({e})")
        finally:
            rec["finished_at"] = utc_now()
            (aw / "out" / "base-gguf.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
            try:
                gs.put_dir(aw / "out", dest)
            except MxError as e:
                say(f"{arm}: upload failed: {e}")
                rc = rc or 1
    publish_attempt(gs, jobdir, attempt, rc, {"arms": arms})
    return rc


def native_modelfile(staged: str) -> str:
    """mx-base's staged Modelfile without TEMPLATE and PARSER: Ollama then serves the GGUF's own chat template
    (the spec's mx-base-native diagnostic). FROM and every PARAMETER stay."""
    from souschef_lab.export import parse_modelfile
    keep = [ins.raw for ins in parse_modelfile(staged) if ins.keyword not in ("TEMPLATE", "PARSER", "RENDERER")]
    return "\n".join(keep) + "\n"


def cmd_stage_native(a) -> int:
    """models/mx-base/ → models/mx-base-native/ (server-side GGUF copy + native_modelfile). Run it only for the
    diagnostic (a JUDGE, or before accepting a STOP)."""
    run = check_run(a.run)
    gs = GS()
    src, dst = model_uri("mx-base", run), model_uri(NATIVE_ARM, run)
    text = (gs.cat(f"{src}/Modelfile") or b"").decode("utf-8")
    if not text:
        raise ConfigError(f"{src}/Modelfile is missing: base-gguf first")
    facts = modelfile_facts(text)
    g = str(facts.get("from") or "")
    if not g.startswith("./") or not g.endswith(".gguf"):
        raise ConfigError(f"{src}/Modelfile FROM {g!r}")
    gs.copy_object(f"{src}/{g[2:]}", f"{dst}/{g[2:]}")
    nat = native_modelfile(text)
    gs.put_bytes(nat.encode("utf-8"), f"{dst}/Modelfile", content_type="text/plain")
    say(f"stage-native: {dst}/ (FROM {g}, no TEMPLATE/PARSER)")
    return 0


def fetch_converter(ref: str, work: Path) -> Path:
    """The fallback converter (docs/MISTRAL.md risk 1): upstream llama.cpp's convert_hf_to_gguf.py at `ref`
    (b11081 = the llama.cpp Ollama 0.34.4 loads with). Needs egress to github.com; scripts only, no build."""
    if not re.fullmatch(r"b\d{4,6}", ref):
        raise ConfigError(f"--converter-ref {ref!r}: a llama.cpp build tag b<NNNN>")
    d = work / f"llama.cpp-{ref}"
    if not (d / "convert_hf_to_gguf.py").is_file():
        shutil.rmtree(d, ignore_errors=True)
        r = subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", ref, "https://github.com/ggml-org/llama.cpp", str(d)],
                           check=False)
        if r.returncode != 0:
            raise MxError(f"git clone llama.cpp {ref} failed (egress?)")
    return d / "convert_hf_to_gguf.py"


# --- job bodies (printed, never submitted) -----------------------------------------------------------------------------

def nonce() -> str:
    return "mx-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def check_image(image: str, what: str, placeholders: bool) -> str:
    if placeholders and image.startswith("<"):
        return image
    if not DIGEST_RE.fullmatch(image or ""):
        raise ConfigError(f"{what} {image!r}: pin it by digest (<host>/<path>@sha256:<64 hex>)")
    if "/sous-chef-trainer@" in image or "/sous-chef-lab@" in image or "/sous-chef-model/" in image:
        raise ConfigError(f"{what} {image!r}: the Mistral arm runs only sous-chef-trainer-mx / sous-chef-mx images")
    return image


def job_body(stage: str, *, run: str, image: str, arm: str | None = None, pair: str | None = None, attempt: int = 1,
             create: str | None = None, rung: str | None = None, mode: str = "full", cases_run: str = DEFAULT_CASES_RUN,
             allow_after_round_train: bool = False, converter_ref: str | None = None,
             mxt_temperature: float | None = None) -> dict:
    """One Vertex customJobs resource. Every body: serviceAccount souschef-mx, labels purpose=sous-chef-mx +
    a create nonce (never sous-chef-pipeline: a round adopts or cancels only its own jobs)."""
    run = check_run(run)
    create = create or nonce()
    labels = {"purpose": "sous-chef-mx", "run": run, "stage": stage, "create": create}
    env = {"GOOGLE_CLOUD_PROJECT": PROJECT, "PIPELINE_BUCKET": BUCKET}
    sched: dict[str, Any] = {"strategy": "STANDARD"}
    if stage == "base-gguf":
        m = MACHINES["base-gguf"]
        spec = {"machineSpec": {"machineType": m["machineType"]}}
        cont = {"imageUri": image, "command": ["/opt/lab/.venv-train/bin/python", "/opt/lab/tools/mxarm.py"],
                "args": ["base-gguf", "--run", run, "--arms", arm or "mx-base,qwen-base"]
                + (["--converter-ref", converter_ref] if converter_ref else [])}
        name = f"souschef-mx-basegguf-{run}-{attempt}"
    elif stage == "exam":
        if arm not in EXAM_ARMS:
            raise ConfigError(f"exam --arm {arm}: one of {', '.join(EXAM_ARMS)}")
        m = MACHINES["exam"]
        spec = {"machineSpec": {k: m[k] for k in ("machineType", "acceleratorType", "acceleratorCount")}}
        cont = {"imageUri": image, "args": ["exam", "--run", run, "--arm", arm, "--replicates",
                                            "2" if arm in TWO_REPLICATE_ARMS else "1", "--cases-run", cases_run]}
        labels["arm"] = arm
        name = f"souschef-mx-exam-{run}-{arm}-{attempt}"
    elif stage == "train":
        r = RUNGS.get(rung or "")
        if r is None:
            raise ConfigError(f"train --rung {rung}: one of {', '.join(RUNGS)}")
        if r.get("after_round_train") and not allow_after_round_train:
            lab = ROUNDS.get(cases_run, {}).get("label", cases_run)
            raise ConfigError(f"rung {rung} competes with {lab}'s training ladder: only after {lab}'s train/DONE (pass "
                              "--after-round-train-done once `gates --stage 2 --rung …` shows it)")
        if mode not in ("smoke", "full"):
            raise ConfigError("train --mode smoke|full")
        m = {"disk": 500, "timeout": 43200}
        spec = {"machineSpec": {"machineType": r["machineType"], "acceleratorType": r["acceleratorType"], "acceleratorCount": 1}}
        sched = {"strategy": r["strategy"]}
        if r["strategy"] == "FLEX_START":
            sched["maxWaitDuration"] = "14400s"
        env = {"PIPELINE_BUCKET": EXP_ROOT, "RUNS_ROOT": f"{EXP_ROOT}/runs", "BASE_ROOT": f"{EXP_ROOT}/base",
               "BASE_FAMILY": "mistral", "BASE_TAR": MISTRAL_TAR, "GOOGLE_CLOUD_PROJECT": PROJECT,
               "TRAIN_RUNG": rung, "TRAIN_ATTEMPT": str(attempt), "TRAIN_NONCE": create, "TRAIN_IMAGE": image,
               "TRAIN_ARMS": "base:3407", "TRAIN_GEN_CHECK": "32"}
        cont = {"imageUri": image, "args": [run, mode]}
        labels.update(rung=rung, stage=f"train-{mode}")
        name = f"souschef-mx-train-{run}-{rung}-{attempt}"
    elif stage == "eval":
        _cur, cand = check_pair(str(pair), cases_run)
        m = MACHINES["eval"]
        spec = {"machineSpec": {k: m[k] for k in ("machineType", "acceleratorType", "acceleratorCount")}}
        cont = {"imageUri": image, "args": ["eval", "--run", run, "--pair", pair, "--cases-run", cases_run, "--mode", mode]
                + (["--mxt-temperature", str(check_mxt_temperature(mxt_temperature))] if cand == "mxt" else [])}
        labels["pair"] = pair
        name = f"souschef-mx-eval-{run}-{pair}-{attempt}"
    else:
        raise ConfigError(f"stage {stage!r}: base-gguf, exam, train or eval")
    sched["timeout"] = f"{m['timeout']}s"
    if stage != "train":
        # a failed or blocked runner is final: Vertex must not re-run (and re-bill) it. The trainer keeps r6's
        # scheduling (its entrypoint's failed-<nonce>.json guard; Spot resumes are new attempts). Every runner also
        # refuses a restart itself (begin_attempt: MX_CREATE names its attempt folder)
        sched["disableRetries"] = True
        env["MX_CREATE"] = create
    cont["env"] = [{"name": k, "value": v} for k, v in env.items()]
    pool = {**spec, "replicaCount": "1", "diskSpec": {"bootDiskType": "pd-ssd", "bootDiskSizeGb": m["disk"]}, "containerSpec": cont}
    return {"displayName": name, "labels": labels,
            "jobSpec": {"workerPoolSpecs": [pool], "scheduling": sched, "serviceAccount": MX_SA}}


def cmd_jobs(a) -> int:
    run = check_run(a.run)
    image_for = {"base-gguf": a.trainer_image, "train": a.trainer_image, "exam": a.mx_image, "eval": a.mx_image}
    img = image_for.get(a.stage)
    if not img:
        raise ConfigError(f"--{'trainer' if a.stage in ('base-gguf', 'train') else 'mx'}-image is required for {a.stage}")
    img = check_image(img, "image", a.placeholders)
    rr = check_cases_run(a.cases_run or DEFAULT_CASES_RUN)  # printing reads nothing but the guards below
    if a.stage == "train" and (RUNGS.get(a.rung or "") or {}).get("after_round_train"):
        # enforced here, not only by the flag: no H100 body is written before the round's own train/DONE exists
        done = f"{BUCKET}/runs/{rr}/train/DONE"
        if not GS().exists(done):
            lab = ROUNDS.get(rr, {}).get("label", rr)
            raise ConfigError(f"{done} is missing: rung {a.rung} competes with {lab}'s training ladder until {lab}'s "
                              "train/DONE exists — no job body written")
    pairs = a.pair.split(",") if a.pair else list(pairs_for(rr, report_only=False))
    items = (a.arm.split(",") if a.stage == "exam" else pairs if a.stage == "eval" else [None])
    base_arms = ",".join(x for x in a.arm.split(",") if x in BASES) or "mx-base,qwen-base"
    bodies = []
    for it in items:
        b = job_body(a.stage, run=run, image=img, arm=it if a.stage == "exam" else base_arms if a.stage == "base-gguf" else None,
                     pair=it if a.stage == "eval" else None, attempt=a.attempt, create=a.create, rung=a.rung, mode=a.mode,
                     cases_run=rr, allow_after_round_train=a.after_round_train_done, converter_ref=a.converter_ref,
                     mxt_temperature=a.mxt_temperature)
        bodies.append(b)
    if a.out_dir:
        od = Path(a.out_dir)
        od.mkdir(parents=True, exist_ok=True)
        for b in bodies:
            p = od / f"{b['displayName']}.json"
            p.write_text(json.dumps(b, indent=1) + "\n", encoding="utf-8")
            print(p)
    else:
        print(json.dumps(bodies if len(bodies) > 1 else bodies[0], indent=1))
    return 0


# --- gates (read-only pre-submit checks) --------------------------------------------------------------------------------

def vertex_jobs(runner: Callable, flt: str, regions: Iterable[str] = ("us-central1", "us-east4", "us-west1", "us-east1")) -> list[dict]:
    out = []
    for reg in regions:
        r = runner(["gcloud", "ai", "custom-jobs", "list", f"--project={PROJECT}", f"--region={reg}", f"--filter={flt}",
                    "--format=json"], timeout=300)
        if r.returncode == 0 and r.stdout.strip():
            for j in json.loads(r.stdout.decode("utf-8")):
                j["_region"] = reg
                out.append(j)
    return out


ACTIVE = ("JOB_STATE_QUEUED", "JOB_STATE_PENDING", "JOB_STATE_RUNNING")
STAGE_LABEL = {"prep": "base-gguf", "1": "exam", "2": "train", "3": "eval"}  # the stage label of the jobs each gate guards


def gates(stage: str, *, run: str, cases_run: str, gs: GS, runner: Callable = _run, rung: str | None = None,
          pairs: list[str] | None = None) -> list[dict]:
    ri = round_info(cases_run)
    lab = ri["label"]
    root, rnd = run_root(run), f"{BUCKET}/runs/{cases_run}"
    res: list[dict] = []

    def g(name: str, ok: bool, detail: str) -> None:
        res.append({"gate": name, "ok": bool(ok), "detail": detail})

    mx_active = [j for j in vertex_jobs(runner, "labels.purpose=sous-chef-mx") if j.get("state") in ACTIVE]
    if stage in ("prep", "1"):
        g("mistral base tar", gs.exists(f"{EXP_ROOT}/base/{MISTRAL_TAR}") and gs.exists(f"{EXP_ROOT}/base/{MISTRAL_TAR}.sha256"),
          f"{EXP_ROOT}/base/{MISTRAL_TAR} (+ .sha256)")
        g("qwen base tar (read only)", gs.exists(f"{BUCKET}/base/{QWEN_TAR}"), f"{BUCKET}/base/{QWEN_TAR}")
    if stage == "1":
        snap = gs.read_json(f"{BUCKET}/{SNAP_REL}/snapshot.json") or {}
        g("r4 snapshot", snap.get("digest") == R4_DIGEST, f"{BUCKET}/{SNAP_REL}/snapshot.json (snapshot-r4, while r4 is live)")
        for arm in STAGE1_ARMS:
            g(f"model {arm}", gs.exists(f"{root}/models/{arm}/Modelfile"), f"{root}/models/{arm}/")
    if stage == "2":
        plan = gs.read_json(f"{root}/plan.json") or {}
        g("plan is this round", plan.get("cases_run") == cases_run,
          f"{root}/plan.json cases_run {plan.get('cases_run')!r} (prep-run --cases-run {cases_run})")
        ds = gs.read_json(f"{rnd}/result-dataset.json") or {}
        g(f"{lab} dataset ok", ds.get("status") == "ok", f"result-dataset.json status {ds.get('status')!r}")
        # runs/<id>/dataset/ holds r4dataset's layout: dataset/{train,val}.jsonl under it (pipeline-train.sh --data)
        g("dataset copied", gs.exists(f"{root}/dataset/dataset/train.jsonl") and gs.exists(f"{root}/plan.json"),
          f"{root}/dataset/dataset/train.jsonl + plan.json (prep-run)")
        gd = gs.read_json(f"{root}/gate.json") or {}
        ok, why = stage1_go(gd, gs.read_json(f"{root}/user-go.json"))
        g("stage 1 verdict", ok, why + (f" (carried from {gd['carried_from']})" if gd.get("carried_from") else ""))
        tr = vertex_jobs(runner, f"displayName~souschef-train-{cases_run}")
        states = sorted({str(j.get("state")) for j in tr})
        done = gs.exists(f"{rnd}/train/DONE")
        g(f"{lab} train running or done", "JOB_STATE_RUNNING" in states or done, f"{lab} train jobs {states or 'none'}; DONE {done}")
        if rung is None:
            g("rung named", False, f"--rung: the rung the train body uses (a100-40-spot while {lab} trains)")
        elif rung not in RUNGS:
            g("rung named", False, f"--rung {rung}: one of {', '.join(RUNGS)}")
        elif RUNGS[rung].get("after_round_train"):
            g(f"rung {rung} after {lab}", done, f"{rnd}/train/DONE {'exists' if done else 'is missing'}: {rung} competes with "
              f"{lab}'s ladder until then")
        else:
            g(f"rung {rung}", True, f"Spot A100-40: its own quota, never {lab}'s on-demand fallback")
    busy = [j.get("displayName") for j in mx_active
            if str((j.get("labels") or {}).get("stage") or "").startswith(STAGE_LABEL[stage])]
    g(f"no active MX {STAGE_LABEL[stage]} job", not busy, f"{busy or 'none'} (all active MX jobs: "
      f"{[j.get('displayName') for j in mx_active] or 'none'})")
    if stage == "3":
        allp = pairs_for(cases_run)
        want = pairs or list(pairs_for(cases_run, report_only=False))
        models = {m for p in want for m in allp.get(p, ())}
        unknown = [p for p in want if p not in allp]
        if unknown:
            g("pairs known", False, f"{unknown}: not pairs of {lab}'s round ({', '.join(allp)})")
        if "r4" in models:
            snap = gs.read_json(f"{BUCKET}/{SNAP_REL}/snapshot.json") or {}
            g("r4 snapshot", snap.get("digest") == R4_DIGEST, f"{BUCKET}/{SNAP_REL}/snapshot.json digest {snap.get('digest')!r} "
              "(snapshot-r4: the r4 pairs' baseline and manifest)")
            g("model r4 staged", gs.exists(f"{root}/models/r4/Modelfile"), f"{root}/models/r4/ (prep-run --from-run)")
        seen_done: set[str] = set()
        for m in sorted(models):
            done = train_done_uri(m, run)
            if done and done not in seen_done:  # mx and mxt share one training
                seen_done.add(done)
                g(f"{m} train DONE", gs.exists(done), done)
            if round_of_model(m):
                g(f"{m} staged + image", gs.exists(f"{model_uri(m, run)}/Modelfile") and
                  gs.exists(f"{BUCKET}/runs/{round_of_model(m)}/model/image.json"),
                  f"{model_uri(m, run)}/ and runs/{round_of_model(m)}/model/image.json")
        rep = gs.read_json(f"{rnd}/report.json") or {}
        g(f"{lab} report.json at", bool(rep.get("at")) and _parse_at(rep.get("at")) is not None,
          f"{rnd}/report.json at {rep.get('at')!r}: pins the current model's baseline (later replicates dropped)")
        g(f"{lab} eval finished", gs.exists(f"{rnd}/eval/bar.json") and gs.exists(f"{rnd}/eval/candidate-baseline.json"),
          f"{lab} eval/bar.json (control trust) and eval/candidate-baseline.json ({ri['cc']} as current)")
        cur = ri["current"]
        want_digest = R4_DIGEST if cur == "r4" else (gs.read_json(f"{BUCKET}/runs/{round_of_model(cur)}/model/image.json") or {}).get("digest")
        got = ((rep.get("results") or {}).get("current") or {}).get("digest")
        g(f"{lab}'s eval ran against {cur}", bool(got) and got == want_digest,
          f"{rnd}/report.json results.current.digest {str(got)[:19]}… vs {cur} {str(want_digest)[:19]}… (ROUNDS: the control's current)")
    return res


def cmd_gates(a) -> int:
    gs = GS()
    rows = gates(a.stage, run=a.run, cases_run=resolve_cases_run(gs, a.run, a.cases_run), gs=gs, rung=a.rung,
                 pairs=a.pair.split(",") if a.pair else None)
    for r in rows:
        print(f"{'ok  ' if r['ok'] else 'FAIL'} {r['gate']:<28} {r['detail']}")
    return 0 if all(r["ok"] for r in rows) else 1


# --- prep-run (main session: writes under EXP_ROOT only) ----------------------------------------------------------------

def object_hashes(gs: GS, prefix: str) -> dict[str, str]:
    """{relative name: md5/crc32c} of every object under a prefix (gcloud storage objects list)."""
    if gs.local:
        base = gs.path(prefix)
        return {p.relative_to(base).as_posix(): mx_gguf.file_sha256(p) for p in base.rglob("*") if p.is_file()} if base.is_dir() else {}
    r = gs.runner(["gcloud", "storage", "objects", "list", prefix.rstrip("/") + "/**", "--format=json"], timeout=600)
    if r.returncode != 0:
        raise MxError(f"gcloud storage objects list {prefix}: exit {r.returncode}")
    out = {}
    for o in json.loads(r.stdout.decode("utf-8") or "[]"):
        name = str(o.get("name") or "")
        key = name.split("/dataset/", 1)[-1]
        out[key] = (o.get("md5_hash") or o.get("md5Hash") or o.get("crc32c_hash") or o.get("crc32cHash") or "")
    return out


def carry_over(gs: GS, run: str, from_run: str) -> dict:
    """What a new round's run takes from an earlier arm run (both under EXP_ROOT): Stage 1 — gate.json (marked
    carried_from; a GO on the same untrained bases needs no re-run) and user-go.json — and the staged r4 model
    (models/r4/, the r4 pairs' current). Never overwrites: what the run already has stays."""
    src, dst = run_root(check_run(from_run)), run_root(run)
    if src == dst:
        raise ConfigError("--from-run is this run")
    done: dict[str, Any] = {"from": from_run, "copied": [], "kept": []}
    gd = gs.read_json(f"{src}/gate.json")
    if gd is None:
        raise ConfigError(f"{src}/gate.json is missing: nothing to carry")
    if gs.exists(f"{dst}/gate.json"):
        done["kept"].append("gate.json")
    else:
        gs.put_json(dict(gd, carried_from=from_run, carried_at=utc_now()), f"{dst}/gate.json")
        done["copied"].append("gate.json")
        ug = gs.read_json(f"{src}/user-go.json")
        if ug is not None:
            gs.put_json(dict(ug, carried_from=from_run), f"{dst}/user-go.json")
            done["copied"].append("user-go.json")
    if gs.exists(f"{dst}/models/r4/Modelfile"):
        done["kept"].append("models/r4/")
    elif gs.exists(f"{src}/models/r4/Modelfile"):
        for uri in gs.list(f"{src}/models/r4"):
            gs.copy_object(uri, f"{dst}/models/r4/{uri.rsplit('/models/r4/', 1)[1]}")
        done["copied"].append("models/r4/")
    else:
        raise ConfigError(f"{src}/models/r4/ is missing: the r4 pairs need r4's staged model")
    return done


def cmd_prep_run(a) -> int:
    run, rr = check_run(a.run), check_cases_run(a.cases_run or DEFAULT_CASES_RUN)
    ri = round_info(rr)
    gs = GS()
    src = f"{BUCKET}/runs/{rr}"
    root = run_root(run)
    plan = gs.read_json(f"{root}/plan.json")
    if plan and plan.get("cases_run") not in (None, rr):
        raise ConfigError(f"{root}/plan.json is for {plan.get('cases_run')}: one run, one round (pick a new --run)")
    ds = gs.read_json(f"{src}/result-dataset.json") or {}
    if ds.get("status") != "ok":
        raise ConfigError(f"{src}/result-dataset.json status {ds.get('status')!r}: wait for {ri['label']}'s dataset stage")
    gs.copy_prefix(f"{src}/dataset", f"{root}/dataset")
    h_src, h_dst = object_hashes(gs, f"{src}/dataset"), object_hashes(gs, f"{root}/dataset")
    if not h_src or h_src != h_dst:
        raise MxError(f"the dataset copy differs from {ri['label']}'s ({len(h_src)} vs {len(h_dst)} objects)")
    try:
        snap = snapshot_r4(gs)
        say(f"prep-run: r4 snapshot {'kept' if snap.get('kept') else 'written'} ({BUCKET}/{SNAP_REL}/)")
    except ConfigError as e:
        say(f"prep-run: WARNING no r4 snapshot ({e}); Stage 3's r4 pairs refuse to run without one")
    carried = carry_over(gs, run, a.from_run) if a.from_run else None
    if carried:
        say(f"prep-run: from {a.from_run}: copied {carried['copied'] or 'nothing'}, kept {carried['kept'] or 'nothing'}")
    gs.put_json({"arms": ["base:3407"], "gen_check": 32}, f"{root}/train-config.json")
    gs.put_json({"run": run, "cases_run": rr, "label": ri["label"], "at": utc_now(),
                 "model": {"repo": MISTRAL_REPO, "revision": MISTRAL_REV},
                 "dataset": {"from": f"{src}/dataset", "hashes": h_src, "result": ds.get("metrics")},
                 "round_train_config": gs.read_json(f"{src}/train-config.json"), "carried": carried,
                 "images": {"trainer_mx": a.trainer_image, "mx": a.mx_image}}, f"{root}/plan.json")
    say(f"prep-run: {root}/dataset ({len(h_src)} objects, identical to {ri['label']}'s), train-config.json, plan.json")
    return 0


def cmd_snapshot_r4(a) -> int:
    doc = snapshot_r4(GS())
    print(json.dumps(doc, indent=1))
    return 0


# --- selftest ----------------------------------------------------------------------------------------------------------------

def cmd_selftest(a) -> int:
    checks: dict[str, str] = {}

    def run(name: str, argv: list[str], want: str | None = None) -> None:
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
            text = (r.stdout + r.stderr).strip()
            ok = r.returncode == 0 and (want is None or want in text)
            checks[name] = ("ok " if ok else f"FAIL (exit {r.returncode}) ") + (text.splitlines()[-1][:120] if text else "")
        except (OSError, subprocess.SubprocessError) as e:
            checks[name] = f"FAIL {type(e).__name__}: {e}"[:200]

    for mod in ("souschef_lab.export", "souschef_lab.retrain.common", "souschef_lab.serving_mistral", "mx_latency"):
        try:
            __import__(mod)
            checks[mod] = "ok"
        except Exception as e:  # noqa: BLE001
            checks[mod] = f"FAIL {e}"[:200]
    if a.trainer:
        llama = Path(os.environ.get("UNSLOTH_LLAMA_CPP_PATH") or "/opt/llama.cpp")
        checks["convert_hf_to_gguf.py"] = "ok" if (llama / "convert_hf_to_gguf.py").is_file() else "FAIL missing"
        checks["llama-quantize"] = "ok" if os.access(llama / "llama-quantize", os.X_OK) else "FAIL missing"
        run("make", ["make", "--version"])
        for arm, cfg in BASES.items():
            checks[cfg["modelfile"]] = "ok" if (LAB_DIR / cfg["modelfile"]).is_file() else "FAIL missing"
    else:
        run("ollama", ["ollama", "--version"], want=OLLAMA_VERSION)
        for mod in ("souschef_lab.harness", "souschef_lab.retrain.eval_stage", "souschef_lab.pipeline"):
            try:
                __import__(mod)
                checks[mod] = "ok"
            except Exception as e:  # noqa: BLE001
                checks[mod] = f"FAIL {e}"[:200]
        exe = os.environ.get("SOUS_CHEF_BIN") or "/opt/sous-chef/bin/sous-chef"
        checks["sous-chef"] = "ok" if os.access(exe, os.X_OK) else f"FAIL {exe}"
        run("node", ["node", "--version"])
        run("gcloud", ["gcloud", "--version"])
        if not a.no_gpu:
            run("nvidia-smi", ["nvidia-smi", "-L"])
    for k, v in checks.items():
        print(f"{k:<44} {v}")
    return 0 if all(v.startswith("ok") for v in checks.values()) else 1


# --- CLI ---------------------------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mxarm", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--run", default=DEFAULT_RUN)
        p.add_argument("--cases-run", default=None,
                       help="the retrain round whose dataset, cases, candidate and eval are used (" +
                            ", ".join(i["label"] + " " + r for r, i in ROUNDS.items()) +
                            f"); default: the run's plan.json, else {DEFAULT_CASES_RUN}")
        return p

    b = common(sub.add_parser("base-gguf", help="convert + quantize the untrained bases (CPU; trainer-mx image)"))
    b.add_argument("--arms", default="mx-base,qwen-base")
    b.add_argument("--converter-ref", help="fallback: upstream llama.cpp convert_hf_to_gguf.py at this tag (b11081)")
    b.add_argument("--workdir")
    e = common(sub.add_parser("exam", help="Stage 1: one arm's real-phrasing exam on one L4"))
    e.add_argument("--arm", required=True, choices=EXAM_ARMS)
    e.add_argument("--replicates", type=int)
    e.add_argument("--model", help="a staged model folder (default: the arm's under the run)")
    e.add_argument("--sets", default=",".join(EXAM_SETS))
    e.add_argument("--limit", type=int, help="rows per set (a rehearsal)")
    e.add_argument("--workers", type=int, default=2)
    e.add_argument("--strict-smoke", action="store_true", help="a /v1/messages smoke without tool_use/text is BLOCKED")
    e.add_argument("--dry", action="store_true", help="the lab's fake service, oracle-scripted (no Ollama, no model)")
    e.add_argument("--workdir")
    v = common(sub.add_parser("eval", help="Stage 3: one paired eval on two L4s"))
    v.add_argument("--pair", required=True, help="r4-mx, <label>c-mx, the control (r4-r6c / r6c-r7c) or r4-mxt (report only)")
    v.add_argument("--mxt-temperature", type=float, default=None, help=f"r4-mxt: MX's route temperature (≤ {MXT_TEMPERATURE})")
    v.add_argument("--mode", choices=("full", "smoke"), default="full")
    v.add_argument("--workers", type=int, default=2)
    v.add_argument("--strict-smoke", action="store_true")
    v.add_argument("--workdir")
    g = common(sub.add_parser("gate", help="Stage 1 verdict → gate.json"))
    g.add_argument("--r4-ref-run", default="rt-20261014-0708-7cac28-1184ff9a",
                   help="a round whose eval/bar.json `<set>:recorded` current = r4 (the anchor); '' = none")
    g.add_argument("--no-write", action="store_true")
    r = common(sub.add_parser("report", help="the comparison report → report/report.{json,md}"))
    r.add_argument("--jobs-json", help="gcloud ai custom-jobs list --filter=labels.purpose=sous-chef-mx --format=json > FILE")
    r.add_argument("--no-write", action="store_true")
    r.add_argument("--no-latency", action="store_true", help="skip the latency section (it reads each pair's Ollama logs)")
    la = common(sub.add_parser("latency", help="the Stage 3 latency section, printed (reads only)"))
    la.add_argument("--json", help="also write the numbers to this LOCAL file")
    j = common(sub.add_parser("jobs", help="print Vertex customJobs bodies (never submits)"))
    j.add_argument("--stage", required=True, choices=("base-gguf", "exam", "train", "eval"))
    j.add_argument("--trainer-image", default="")
    j.add_argument("--mx-image", default="")
    j.add_argument("--arm", default="mx-base,qwen-base,r4")
    j.add_argument("--pair", default="", help="comma-separated; default the round's three main pairs (r4-mxt only by name)")
    j.add_argument("--mxt-temperature", type=float, default=None)
    j.add_argument("--rung", default="a100-40-spot", choices=sorted(RUNGS))
    j.add_argument("--mode", default="full", choices=("full", "smoke"))
    j.add_argument("--attempt", type=int, default=1)
    j.add_argument("--create", help="the create-label nonce (default mx-<utc stamp>)")
    j.add_argument("--after-round-train-done", "--after-r6-done", dest="after_round_train_done", action="store_true",
                   help="allow a rung that competes with the round's training (A100 on demand, H100)")
    j.add_argument("--placeholders", action="store_true", help="allow <…> image placeholders (docs)")
    j.add_argument("--converter-ref", help="base-gguf: the fallback converter tag (b11081; needs egress)")
    j.add_argument("--out-dir")
    gt = common(sub.add_parser("gates", help="read-only pre-submit checks"))
    gt.add_argument("--stage", required=True, choices=("prep", "1", "2", "3"))
    gt.add_argument("--rung", choices=sorted(RUNGS), help="stage 2: the train body's rung (H100 / on-demand A100 need the round's train/DONE)")
    gt.add_argument("--pair", default="", help="stage 3: the pairs about to be submitted (default the round's three main pairs)")
    sn = sub.add_parser("snapshot-r4", help="freeze r4's state, baseline and manifest under experiments/mistral/snapshots/r4/")
    sn.set_defaults(run=DEFAULT_RUN)
    ug = common(sub.add_parser("user-go", help="record the user's GO on a JUDGE gate → user-go.json"))
    ug.add_argument("--by", required=True, help="who decided")
    ug.add_argument("--note", required=True, help="why (one line)")
    common(sub.add_parser("stage-native", help="models/mx-base → models/mx-base-native (no TEMPLATE/PARSER)"))
    p = common(sub.add_parser("prep-run", help="copy the round's dataset under the run, write train-config.json + plan.json"))
    p.add_argument("--from-run", default=None, help=f"carry Stage 1 (gate.json) and models/r4/ from an earlier arm run ({R6_ARM_RUN})")
    p.add_argument("--trainer-image", default="")
    p.add_argument("--mx-image", default="")
    s = sub.add_parser("selftest", help="the image's tools")
    s.add_argument("--trainer", action="store_true", help="the trainer-mx image's tools instead of sous-chef-mx's")
    s.add_argument("--no-gpu", action="store_true")
    return ap


COMMANDS = {"base-gguf": cmd_base_gguf, "exam": cmd_exam, "eval": cmd_eval, "gate": cmd_gate, "report": cmd_report,
            "jobs": cmd_jobs, "gates": cmd_gates, "prep-run": cmd_prep_run, "selftest": cmd_selftest,
            "snapshot-r4": cmd_snapshot_r4, "user-go": cmd_user_go, "stage-native": cmd_stage_native, "latency": cmd_latency}


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # Vertex cancel / preemption: run the finally blocks
    a = build_parser().parse_args(argv)
    try:
        return COMMANDS[a.cmd](a)
    except MxError as e:
        print(f"mxarm {a.cmd}: {e}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
