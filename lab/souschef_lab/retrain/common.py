"""Shared plumbing of the retrain stages (design §3, I2/I3): the pipeline bucket,
run files, stage results, notifications and secret scrubbing.

Every stage reads and writes the pipeline bucket by RELATIVE key (`runs/<id>/…`,
`cases/pool/…`, `state/…`, I2) through `Store`, a thin adapter over builder B's
gcsio.Bucket (gs://… through the `gcloud storage` CLI, or a local folder with
the same layout for DRY runs, the tests and a workstation): a missing object
reads as None / False.

Stage result (I3): `runs/<id>/result-<stage>.json` in pipeline_common.result_doc's
shape {status, stage, run_id, at, notification, notifications, metrics,
artifacts, …}. Exit codes: 0 the stage finished (its outcome is the result's
status), 1 a failure the workflow may retry, 2 a configuration / precondition
error (no retry).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

EXIT_OK, EXIT_FAIL, EXIT_CONFIG = 0, 1, 2


class StageError(RuntimeError):
    """A stage failed (exit 1: the workflow retries by policy, then posts stage_failed)."""

    def __init__(self, message: str, *, reason: str = "failed", details: dict | None = None):
        super().__init__(message)
        self.reason = reason
        self.details = details or {}


class ConfigError(StageError):
    """A configuration or precondition error (exit 2: no retry)."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run8(run_id: str) -> str:
    """Eight hex characters naming a run in site ids (`lab-<run8>p1-…`)."""
    return hashlib.sha1(str(run_id).encode("utf-8")).hexdigest()[:8]


def h100(s: str) -> int:
    return int(hashlib.sha1(str(s).encode("utf-8")).hexdigest(), 16) % 100


def redact(text: Any) -> str:
    """A message safe for logs and Slack (pipeline_common.scrub: env secrets, tokens, keys, webhooks)."""
    from .. import pipeline_common as PC
    return PC.scrub(text)


# --- jsonl --------------------------------------------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(p)
    return n


def write_json(path: Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


# --- the store: gcsio.Bucket (builder B's) behind the few calls the stages make ----------------------------------------

class Store:
    """The pipeline bucket by relative key (I2), over gcsio.Bucket: a missing
    object reads as None / False here instead of raising."""

    def __init__(self, bucket):
        self.b = bucket
        self.uri = bucket.url

    def url(self, rel: str) -> str:
        return self.b.uri(rel)

    def read_bytes(self, rel: str) -> bytes | None:
        from .. import gcsio
        try:
            return self.b.read_bytes(rel)
        except gcsio.NotFound:
            return None

    def write_bytes(self, rel: str, data: bytes) -> None:
        self.b.write_bytes(rel, data)

    def create_bytes(self, rel: str, data: bytes) -> bool:
        """Write only when absent (ifGenerationMatch=0). True = written."""
        from .. import gcsio
        try:
            self.b.write_bytes(rel, data, if_generation=0)
            return True
        except gcsio.PreconditionFailed:
            return False

    def exists(self, rel: str) -> bool:
        return self.b.exists(rel)

    def list(self, prefix: str) -> list[str]:
        p = prefix.strip("/")
        return self.b.list(p + "/") if p else self.b.list("")

    def delete(self, rel: str) -> None:
        self.b.delete(rel)

    def download(self, rel: str, local: Path) -> bool:
        from .. import gcsio
        try:
            self.b.download(rel, local)
            return True
        except gcsio.NotFound:
            return False

    def upload(self, local: Path, rel: str) -> None:
        self.b.upload(local, rel)

    def sync_down(self, prefix: str, local_dir: Path) -> int:
        if not self.list(prefix):
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            return 0
        self.b.rsync_down(prefix.strip("/"), local_dir)
        return 1

    def sync_up(self, local_dir: Path, prefix: str) -> int:
        if not Path(local_dir).is_dir():
            return 0
        self.b.rsync_up(local_dir, prefix.strip("/"))
        return 1

    def read_json(self, rel: str, default: Any = None) -> Any:
        b = self.read_bytes(rel)
        return default if b is None else json.loads(b.decode("utf-8"))

    def write_json(self, rel: str, obj: Any) -> None:
        self.b.write_json(rel, obj)  # strict JSON (NaN → null), uploaded as application/json

    def read_jsonl(self, rel: str) -> list[dict]:
        b = self.read_bytes(rel)
        if b is None:
            return []
        return [json.loads(x) for x in b.decode("utf-8").splitlines() if x.strip()]

    def write_jsonl(self, rel: str, rows: Iterable[dict]) -> None:
        from .. import gcsio
        self.write_bytes(rel, b"".join(gcsio.json_bytes(r, indent=None) for r in rows))


def open_store(uri: str | Path | None) -> Store:
    """gs://bucket (gcloud storage) or a local folder with the same layout; default $PIPELINE_BUCKET."""
    from .. import gcsio
    try:
        return Store(gcsio.bucket(str(uri) if uri else None))
    except gcsio.StoreError as e:
        raise ConfigError(str(e)) from None


# --- run files --------------------------------------------------------------------------------------------------------

def run_prefix(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}", str(run_id or "")):
        raise ConfigError(f"bad run id {run_id!r}")
    return f"runs/{run_id}"


def load_run(store: Store, run_id: str) -> dict:
    """runs/<id>/run.json (written by the workflow's init step). Missing keys get
    defaults; a missing file is a run started by hand (the playbook)."""
    run = store.read_json(f"{run_prefix(run_id)}/run.json") or {}
    run.setdefault("run_id", run_id)
    run.setdefault("mode", "full")
    run.setdefault("trigger", {})
    run.setdefault("deployed", {})
    return run


def manifest_doc(store: Store, hsh: str) -> dict:
    doc = store.read_json(f"manifests/{hsh}.json")
    if doc is None:
        raise ConfigError(f"manifests/{hsh}.json is not in the bucket (the drift poll writes every manifest it sees)")
    return doc


def deployed_of(store: Store, run: dict) -> dict:
    """{hash, launcher_build, generated_from} of the manifest the round trains on:
    run.json's `deployed` (pinned at init), else state/deployed.json."""
    d = dict(run.get("deployed") or {})
    if not d.get("hash"):
        d = dict(store.read_json("state/deployed.json") or {})
    if not d.get("hash"):
        raise ConfigError("no deployed manifest: neither run.json `deployed.hash` nor state/deployed.json")
    return d


def model_state(store: Store) -> dict:
    m = store.read_json("state/model.json")
    if not m or not m.get("training_manifest"):
        raise ConfigError("state/model.json (the current model: tag, digest, training_manifest) is missing — playbook R0")
    return m


# --- the base family (souschef_lab/chatfmt.py; docs/RETRAIN.md "The base family") ---------------------------------------
# A round trains one family: trigger.base_family, else the bucket's infra/config.json `base_family` (Terraform's
# pipeline_base_family), else qwen. The dataset stage resolves it once and pins it in runs/<id>/train-config.json
# (`family`, written only for a non-qwen round), which the trainer, the eval and the decision read. A model's family
# rides in state/model.json (`family`, absent = qwen: every model before chocolate-muffin) and its history entries.

TRAIN_CONFIG = "train-config.json"


def round_base_family(store: Store, trigger: dict | None) -> tuple[str, str]:
    """(family, where it came from) for a round: trigger.base_family, else infra/config.json base_family, else the
    default. ConfigError on a name chatfmt does not know."""
    from .. import chatfmt
    val, src = (trigger or {}).get("base_family"), "trigger.base_family"
    if val in (None, ""):
        cfg = store.read_json("infra/config.json")
        val, src = (cfg.get("base_family") if isinstance(cfg, dict) else None), "infra/config.json base_family"
    if val in (None, ""):
        return chatfmt.DEFAULT, "default"
    try:
        return chatfmt.check(val), src
    except ValueError as e:
        raise ConfigError(f"{src}: {e}") from None


def trained_family(store: Store, run_id: str) -> str:
    """The family a round trained: runs/<id>/train-config.json `family` (absent: qwen)."""
    from .. import chatfmt
    cfg = store.read_json(f"{run_prefix(run_id)}/{TRAIN_CONFIG}")
    try:
        return chatfmt.check(cfg.get("family") if isinstance(cfg, dict) else None)
    except ValueError as e:
        raise ConfigError(f"{run_prefix(run_id)}/{TRAIN_CONFIG}: {e}") from None


def model_family(model: dict | None) -> str:
    """A state/model.json entry's family (or a history entry's): `family`, absent = qwen."""
    from .. import chatfmt
    try:
        return chatfmt.check((model or {}).get("family"))
    except ValueError as e:
        raise ConfigError(f"state/model.json `{(model or {}).get('tag')}`: {e}") from None


def save_manifest(doc: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def notification(kind: str, severity: str, title: str, summary: str, details: dict | None = None,
                 dedup_key: str | None = None) -> dict:
    """I4: what a job hands the workflow (pipeline_common.notification; the workflow adds id, run_id,
    candidate_id, created_at and, for candidate_ready, the decision)."""
    from .. import pipeline_common as PC
    return PC.notification(kind, title=title, summary=summary, severity=severity, details=details, dedup_key=dedup_key)


def write_result(store: Store, run_id: str, stage: str, status: str, *, metrics: dict | None = None,
                 artifacts: list[str] | None = None, notification_: dict | None = None, extra: dict | None = None) -> dict:
    """runs/<id>/result-<stage>.json in pipeline_common.result_doc's shape (I3)."""
    from .. import pipeline_common as PC
    res = PC.result_doc(status=status, stage=stage, run_id=run_id, notification=notification_, metrics=metrics,
                        artifacts=artifacts, **(extra or {}))
    store.write_json(f"{run_prefix(run_id)}/result-{stage}.json", res)
    return res


def workdir(run_id: str, stage: str) -> Path:
    base = Path(os.environ.get("PIPELINE_WORKDIR") or tempfile.gettempdir()) / f"souschef-{run_id}" / stage
    base.mkdir(parents=True, exist_ok=True)
    return base
