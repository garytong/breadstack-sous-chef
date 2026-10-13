"""pipeline_common — what the drift runner and the retrain stages share
(docs/DRIFT.md; DESIGN I1, I3, I4).

  deployed_manifest(url, key, line="int", body=True)   GET /v1/manifests/deployed (I1);
                                                       NoLauncherManifest on 404
  get_manifest(url, key, hash)                         GET /v1/manifests/{hash}?body=1
  notification(kind, …)                                an I4 payload the workflow completes and posts
  write_result(bucket, key, …)                         the {status, stage, run_id, notification?, metrics,
                                                       artifacts[]} every job leaves behind (I3)
  load_thresholds(bucket)                              lab/pipeline/thresholds.json, a bucket's
                                                       config/thresholds.json on top
  id_token(audience) / gpu_warm(url, …)                the Cloud Run metadata server's ID token; a GPU
                                                       service answering /api/version and one generate
  scrub(text)                                          secrets and token-bearing URLs out of a message
  post_notification(url, key, payload)                 the thermometer's ingest call — for the playbook by
                                                       hand only: the jobs never hold the thermometer key
                                                       (the workflow posts)

Exit codes of every pipeline command: 0 = the stage finished (the outcome is in
its result JSON), 1 = failure (the workflow retries by policy), 2 = a
configuration or precondition error (no retry).

Standard library only. Keys are read from the environment (SOUS_CHEF_KEY,
ANTHROPIC_API_KEY, THERMOMETER_API_KEY), never from argv, and never printed.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from . import LAB_DIR

EXIT_OK, EXIT_FAIL, EXIT_CONFIG = 0, 1, 2
THRESHOLDS_FILE = LAB_DIR / "pipeline" / "thresholds.json"
THRESHOLDS_OVERRIDE = "config/thresholds.json"
METADATA_URL = "http://metadata.google.internal"
SECRET_ENV = ("SOUS_CHEF_KEY", "ANTHROPIC_API_KEY", "THERMOMETER_API_KEY", "SERVICE_KEYS")
NOTIFICATION_KINDS = ("drift_detected", "drift_notice", "retrain_started", "stage_failed", "candidate_ready", "rolled_out",
                      "skipped", "weekly_all_clear", "heartbeat", "engine_stale")
SEVERITIES = ("info", "warn", "error")


class PipelineError(RuntimeError):
    """A failure the workflow may retry (exit 1)."""


class ConfigError(PipelineError):
    """A configuration or precondition error (exit 2: no retry)."""


class NoLauncherManifest(PipelineError):
    """GET /v1/manifests/deployed answered 404 no_launcher_manifest."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    return (dt or utc_now()).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def run_id(prefix: str, deployed_hash: str | None = None, now: datetime | None = None) -> str:
    """`dr-YYYYMMDD-HHMM-<deployed8>` (drift) / `rt-…` (retrain)."""
    n = (now or utc_now()).astimezone(timezone.utc)
    tail = f"-{str(deployed_hash)[:8]}" if deployed_hash else ""
    return f"{prefix}-{n:%Y%m%d-%H%M}{tail}"


# ── secrets out of messages ──────────────────────────────────────────────────

_TOKENISH = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}")
_QUERY_SECRET = re.compile(r"(?i)([?&](?:key|token|access_token|id_token|sig|signature|api_key)=)[^&\s]+")
_ANTHROPIC = re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}")
_TM_KEY = re.compile(r"\btm_[A-Za-z0-9_-]{8,}")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
_HOOK = re.compile(r"https://hooks\.slack\.com/\S+")


def scrub(text: Any, extra: list[str] | None = None) -> str:
    """The text without any secret the environment holds, bearer tokens,
    JWTs, Anthropic / thermometer keys, Slack webhooks or secret query
    parameters (DESIGN §1.6: a stage failure's message is posted)."""
    s = str(text if text is not None else "")
    secrets = [os.environ.get(k, "") for k in SECRET_ENV] + list(extra or [])
    for v in secrets:
        for part in (v.split(",") if v else []):
            part = part.strip()
            if len(part) >= 6:
                s = s.replace(part, "[redacted]")
    s = _TOKENISH.sub(r"\1[redacted]", s)
    s = _QUERY_SECRET.sub(r"\1[redacted]", s)
    s = _ANTHROPIC.sub("[redacted]", s)
    s = _TM_KEY.sub("[redacted]", s)
    s = _JWT.sub("[redacted]", s)
    s = _HOOK.sub("[redacted webhook]", s)
    return s


# ── HTTP ─────────────────────────────────────────────────────────────────────

def _http(method: str, url: str, *, headers: dict[str, str] | None = None, body: Any = None,
          timeout: float = 30.0) -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(url, data=data, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urlerror.HTTPError as e:
        return e.code, e.read()
    except (urlerror.URLError, TimeoutError, OSError) as e:
        raise PipelineError(scrub(f"{method} {urlparse.urlsplit(url).netloc}{urlparse.urlsplit(url).path}: {e}")) from None


def _json_body(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"raw": raw[:200].decode("utf-8", "replace")}


def service_key() -> str:
    k = os.environ.get("SOUS_CHEF_KEY", "")
    if not k:
        raise ConfigError("SOUS_CHEF_KEY is empty (the job gets it as a secret env var)")
    return k


# Transient answers of the int service (a cold start, a deploy, Cloud Run's front end) retried before a poll fails.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def deployed_manifest(url: str, key: str, line: str = "int", body: bool = True, days: int | None = None,
                      timeout: float = 30.0, attempts: int = 4, backoff_s: float = 5.0) -> dict:
    """I1: the launcher-deployed manifest of `line` (`body` included when asked). A network error or a
    429/5xx is retried `attempts` times with a doubling pause (5, 10, 20 s): one bad answer from int
    never fails a poll (and never pages anyone)."""
    q = {"line": line}
    if body:
        q["body"] = "1"
    if days:
        q["days"] = str(int(days))
    for i in range(max(1, attempts)):
        last = i == max(1, attempts) - 1
        try:
            status, raw = _http("GET", f"{url.rstrip('/')}/v1/manifests/deployed?{urlparse.urlencode(q)}",
                                headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        except PipelineError:
            if last:
                raise
            time.sleep(backoff_s * (2 ** i))
            continue
        if status in RETRY_STATUSES and not last:
            time.sleep(backoff_s * (2 ** i))
            continue
        break
    doc = _json_body(raw)
    if status == 404:
        code = ((doc.get("error") or {}).get("code") if isinstance(doc.get("error"), dict) else doc.get("error")) if isinstance(doc, dict) else None
        if code in (None, "no_launcher_manifest"):
            raise NoLauncherManifest(f"no launcher manifest on line {line} (no registration and no real turn in the window)")
        raise ConfigError(f"GET /v1/manifests/deployed: 404 {code} (a service build without the endpoint?)")
    if status in (401, 403):
        raise ConfigError(f"GET /v1/manifests/deployed: {status} (the pipeline service key is not accepted)")
    if status != 200:
        raise PipelineError(f"GET /v1/manifests/deployed: {status} {scrub(str(doc)[:200])}")
    if not isinstance(doc, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(doc.get("hash") or "")):
        raise PipelineError("GET /v1/manifests/deployed: no hash in the answer")
    if body and not isinstance(doc.get("body"), dict):
        raise PipelineError("GET /v1/manifests/deployed?body=1: no body in the answer")
    return doc


def get_manifest(url: str, key: str, hsh: str, timeout: float = 30.0) -> dict:
    status, raw = _http("GET", f"{url.rstrip('/')}/v1/manifests/{hsh}?body=1",
                        headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
    if status != 200:
        raise PipelineError(f"GET /v1/manifests/{hsh[:12]}…: {status}")
    doc = _json_body(raw)
    return doc.get("body") if isinstance(doc, dict) and isinstance(doc.get("body"), dict) else doc


def id_token(audience: str, timeout: float = 5.0) -> str:
    """The runtime service account's ID token for `audience` (Cloud Run and
    Cloud Run Jobs: the metadata server)."""
    host = os.environ.get("GCE_METADATA_HOST")  # the client libraries' override (a test's fake metadata server)
    base = f"http://{host}" if host else METADATA_URL
    url = f"{base}/computeMetadata/v1/instance/service-accounts/default/identity?audience={urlparse.quote(audience, safe='')}"
    status, raw = _http("GET", url, headers={"Metadata-Flavor": "Google"}, timeout=timeout)
    if status != 200 or not raw:
        raise PipelineError(f"metadata server: no ID token ({status})")
    return raw.decode("ascii").strip()


def gpu_warm(url: str, *, model: str = "sous-chef", wait_s: float = 900.0, every_s: float = 15.0,
             auth: str = "id-token", progress=print) -> dict:
    """Wait until the GPU service answers `/api/version`, then load the model
    with one 1-token generate (vm-solo.sh's warm). {ok, seconds, version} —
    ok False when it did not answer within `wait_s` (L4 capacity: the drift
    workflow then defers to the night window)."""
    t0 = time.monotonic()
    last = ""
    while time.monotonic() - t0 < wait_s:
        try:
            hdr = {"Authorization": f"Bearer {id_token(url)}"} if auth == "id-token" else {}
            status, raw = _http("GET", f"{url.rstrip('/')}/api/version", headers=hdr, timeout=60)
            if status == 200:
                status, _ = _http("POST", f"{url.rstrip('/')}/api/generate", headers=hdr, timeout=600,
                                  body={"model": model, "prompt": "ok", "stream": False, "options": {"num_predict": 1}})
                if status == 200:
                    return {"ok": True, "seconds": round(time.monotonic() - t0, 1), "version": _json_body(raw)}
                last = f"generate {status}"
            else:
                last = f"version {status}"
        except PipelineError as e:
            last = str(e)
        progress(f"gpu_warm: {urlparse.urlsplit(url).netloc} not ready ({last}); {time.monotonic() - t0:.0f}s")
        time.sleep(every_s)
    return {"ok": False, "seconds": round(time.monotonic() - t0, 1), "last": scrub(last)}


def post_notification(url: str, key: str, payload: dict, timeout: float = 30.0) -> dict:
    """POST /api/ingest/souschef/notifications (the playbook's manual path)."""
    status, raw = _http("POST", f"{url.rstrip('/')}/api/ingest/souschef/notifications",
                        headers={"X-API-Key": key}, body=payload, timeout=timeout)
    if status not in (200, 201):
        raise PipelineError(f"thermometer: {status} {scrub(raw[:200].decode('utf-8', 'replace'))}")
    return _json_body(raw)


# ── notifications and results ────────────────────────────────────────────────

def notification(kind: str, *, title: str, summary: str, severity: str = "info", details: dict | None = None,
                 dedup_key: str | None = None) -> dict:
    """A job's half of an I4 payload (the workflow adds id, run_id,
    candidate_id, created_at and, for candidate_ready, decision)."""
    if kind not in NOTIFICATION_KINDS:
        raise ValueError(f"notification kind {kind!r} is not one of {NOTIFICATION_KINDS}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity {severity!r}")
    return {"kind": kind, "severity": severity, "title": scrub(title)[:200], "summary": scrub(summary)[:4000],
            "details": cap_details(details or {}), "dedup_key": dedup_key}


# The thermometer refuses details over 256 KiB, but the workflow is the tighter bound: Workflows holds every
# variable of every live frame within 512 KB, and a notice is copied into ~6 frames on its way to notify_send
# (drift_poll.res, notify_result.res/notes/msg, notify.payload, notify_send.payload). 32 KB keeps that ~200 KB
# with room for the rest of the run; the full report is linked from the notice anyway.
MAX_DETAILS_BYTES = 32_000


def cap_details(details: dict, limit: int = MAX_DETAILS_BYTES) -> dict:
    """details as they are when they fit, else every list cut to its first 20 items (marked), else a stub that
    points at the report: a notification must never be refused for its size."""
    def size(d: Any) -> int:
        return len(json.dumps(d, ensure_ascii=False, default=str).encode("utf-8"))

    if size(details) <= limit:
        return details

    def cut(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: cut(x) for k, x in v.items()}
        if isinstance(v, list):
            return [cut(x) for x in v[:20]] + ([f"… {len(v) - 20} more"] if len(v) > 20 else [])
        return v

    small = cut(details)
    if isinstance(small, dict):
        small["truncated"] = True
    if size(small) <= limit:
        return small
    keep = {k: (v[:500] if isinstance(v, str) else v) for k, v in details.items()
            if isinstance(v, (str, int, float, bool)) or v is None}
    stub = {**keep, "truncated": True, "note": "details too large for a notification; see the run's report"}
    if size(stub) <= limit:
        return stub
    return {"truncated": True, "note": "details too large for a notification; see the run's report"}


def result_doc(*, status: str, stage: str, run_id: str, notification: dict | None = None,
               notifications: list[dict] | None = None, metrics: dict | None = None, artifacts: list[str] | None = None,
               **extra: Any) -> dict:
    notes = list(notifications or ([] if notification is None else [notification]))
    return {"status": status, "stage": stage, "run_id": run_id, "at": iso(),
            "notification": notes[0] if notes else None, "notifications": notes,
            "metrics": metrics or {}, "artifacts": list(artifacts or []), **extra}


def write_result(b, key: str, doc: dict) -> None:
    b.write_json(key, doc)


def failure_doc(stage: str, run_id: str, exc: BaseException) -> dict:
    return result_doc(status="failed", stage=stage, run_id=run_id,
                      error={"class": type(exc).__name__, "message": scrub(str(exc))[:2000]})


# ── thresholds ───────────────────────────────────────────────────────────────

def load_thresholds(b=None, path: Path | None = None) -> dict:
    """The committed thresholds, a bucket's `config/thresholds.json` on top
    (keys beginning with `_` are comments)."""
    base = json.loads(Path(path or THRESHOLDS_FILE).read_text(encoding="utf-8"))
    over = b.read_json(THRESHOLDS_OVERRIDE, None) if b is not None else None
    out = {k: v for k, v in base.items() if not k.startswith("_")}
    if isinstance(over, dict):
        out.update({k: v for k, v in over.items() if not k.startswith("_")})
    return out
