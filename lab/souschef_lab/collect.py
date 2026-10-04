"""Collector — the browser emulator that asks the Sous Chef service each case
and records what the model did (docs/LAB.md §4).

The service decides, the browser applies (README). Here the "browser" is
this module: per case it starts a turn (`POST /v1/turns`), and for every
batch of `tool.call` events it runs the ops through the engine on a running
copy of the case's site state, posts the engine's results back
(`POST /v1/turns/{id}/tool-results`), answers a `turn.suspended`
`pick_asset` with a fixture asset, and loops until `turn.done` or
`turn.error`. One row per sample is appended to the log, fsynced as it
lands, so a run can be stopped and resumed (a sample logged `status:"ok"`
is never asked again; an error is).

Transport: Bearer `SOUS_CHEF_KEY`, `urllib` only. The default is
`Accept: application/json` — the service's envelope `{turnId, events}`
(no heartbeats); `--sse` reads the `event:`/`data:` frames instead and
drops heartbeats. The row is the same either way.

Pattern copied from the thermometer kit's collect.py
(breadstack-thermometer/lab/in-house-model/collect.py: append_row,
answered_ids, run_batch, the best-of-N rounds and the adaptive stop);
nothing is imported from that repo. Dropped on purpose (docs/LAB.md, "Fixed
decisions"): the Claude refusal — Claude is this lab's teacher — the
assists/plan-hint/clock stops and the real-data paths.

Row (a superset of the thermometer log row, so dataset.to_hf copies over):

  {id:"<case id>#<k>", case_id, sample, model, status, http_status, code,
   error, latency_ms, context, text, messages:[Anthropic blocks], tools, ops,
   events (the streams' + one tool.result per posted item, the trace's shape),
   verdict, reasons, written_paths, usage, stop, turn_id, conversation_id,
   conversation_ended, batches, at, meta:{kind, section, template, field_type,
   holdout, manifest_hash, buildId, engine_fingerprint}}

`messages` is a cache: render.messages_from_events(events, context, text)
rebuilds it exactly (dataset.to_hf reads only the rebuild). Every sample is
a fresh conversation: the site's open one is ended first (README §1.1 would
otherwise continue it — see run_case).

Engine contract (engine.py, another module): `engine(state, ops, ctx)`
returns `{ok, results:[{op, ok, error?, warnings, written, sectionId?}],
changes:[...], state}` — headless.mjs's output with `--print-state`. Any
callable of that shape works; the tests use a pure-Python one.

Verifier contract (verify.py): `verifier(case, row)` returns an object or
dict with `verdict` and `reasons` (extra keys are copied into the row when
present: actual_diff, engine_errors, warnings).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from . import render

LAB_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = LAB_DIR / "fixtures" / "manifest.snapshot.json"
DEFAULT_ASSETS = LAB_DIR / "fixtures" / "assets.json"
DEFAULT_CASES = LAB_DIR / "data" / "cases.jsonl"
DEFAULT_LOG = LAB_DIR / "runs" / "collect-log.jsonl"

# README §2: the ten tools in the service's fixed order.
TOOL_NAMES = ("set_fields", "edit_list", "reset", "restore_default", "set_page_layout", "reset_pinned_section",
              "pick_asset", "lookup_fields", "describe_target", "report_unsupported")
BROWSER_OPS = frozenset(TOOL_NAMES[:6])
# Answered by the service itself; the browser never posts a result for them
# (report_unsupported is surfaced as a tool.call for display only).
SERVER_TOOLS = frozenset(("lookup_fields", "describe_target", "report_unsupported"))
PICK_ASSET = "pick_asset"
MAX_BATCHES = 8  # the service's maxCalls per turn (README §4)
# The verifier's golden engine context (docs/LAB.md §3, FACTS 2 §1).
GOLDEN_CTX = {"cdnUrl": "https://cdn.int.breadstack.io", "galleryFolder": "gal", "now": 1700000000000}
DEFAULT_TEMPLATE = "editorial"
DEFAULT_TIMEOUT_S = 300.0
INTERRUPTED = 130
PASS_VERDICTS = frozenset(("pass", "clarify_ok", "unsupported_ok"))
# A sample of a held-out case: never training material (Adaptive "excluded").
EXCLUDED_REASONS = frozenset(("holdout",))


# --- small helpers (thermometer lab/console.py pattern) --------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict]:
    """Every parsable JSON object line of path; nothing for a missing file."""
    out: list[dict] = []
    p = Path(path)
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line after a crash
            if isinstance(row, dict):
                out.append(row)
    return out


def append_row(path: Path, row: dict) -> None:
    """One log line, flushed and synced: a crash loses no answer already
    logged (thermometer collect.append_row)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def runs_log(log: Path) -> Path:
    """Where each start's settings are recorded, beside the log."""
    log = Path(log)
    return log.with_name(f"{log.stem}-runs.jsonl")


def answered_ids(log_path: Path) -> set[str]:
    """The sample ids ("<case>#<k>") logged ok: never asked again."""
    return {str(r.get("id")) for r in read_jsonl(log_path) if r.get("status") == "ok"}


def stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:12], 16)


def manifest_hash(doc: dict) -> str:
    """README §1.1 in Python (cross-checked against Go in the service's tests)."""
    body = {k: v for k, v in doc.items() if k not in ("generated_from", "coverage")}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- the HTTP client -----------------------------------------------------------------------------------------------------

class ServiceError(Exception):
    """A non-2xx answer (or no answer) from the service."""

    def __init__(self, status: int, code: str, message: str, body: dict | None = None):
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message
        self.body = body or {}


class Superseded(ServiceError):
    """409 turn_not_waiting with status superseded: a newer turn owns the
    conversation; the browser drops its results silently (README §4)."""


@dataclass
class Stream:
    """One answer of POST /v1/turns or tool-results: the events of that stream."""

    turn_id: str
    events: list[dict]  # [{event, data}] in order; data carries turnId and seq

    def of(self, name: str) -> list[dict]:
        return [e["data"] for e in self.events if e.get("event") == name]


def parse_sse(text: str) -> list[dict]:
    """`event:`/`data:` frames → [{event, data}]; heartbeats (and frames
    without data) are dropped. Multi-line `data:` is joined with newlines
    as the SSE spec says."""
    out: list[dict] = []
    name = None
    data_lines: list[str] = []
    for raw in text.splitlines() + [""]:
        line = raw.rstrip("\r")
        if line == "":
            if name is not None and data_lines and name != "heartbeat":
                try:
                    out.append({"event": name, "data": json.loads("\n".join(data_lines))})
                except json.JSONDecodeError:
                    out.append({"event": name, "data": {"raw": "\n".join(data_lines)}})
            name, data_lines = None, []
            continue
        if line.startswith(":"):
            continue
        key, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if key == "event":
            name = value
        elif key == "data":
            data_lines.append(value)
    return out


class Client:
    """The service, as the launcher proxy would reach it: Bearer key, JSON
    envelope by default, SSE on request."""

    def __init__(self, base_url: str, key: str, timeout_s: float = DEFAULT_TIMEOUT_S, sse: bool = False,
                 build_id: str = "lab"):
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.timeout_s = timeout_s
        self.sse = sse
        self.build_id = build_id

    def _request(self, method: str, path: str, body: Any = None, accept: str | None = None) -> tuple[int, str, bytes]:
        # render.compact_json: the context object goes on the wire as the very bytes render.context_block puts in
        # the logged user message (the service json.Compact-s the raw request bytes, keeping their key order).
        data = None if body is None else render.compact_json(body).encode("utf-8")
        req = urlrequest.Request(self.base_url + path, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.key}")
        req.add_header("X-Launcher-Build", self.build_id)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if accept:
            req.add_header("Accept", accept)
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.status, resp.headers.get("Content-Type", ""), resp.read()
        except urlerror.HTTPError as e:
            raw = e.read()
            try:
                parsed = json.loads(raw.decode("utf-8")) if raw else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                parsed = {"error": "http_error", "message": raw.decode("utf-8", "replace")[:200]}
            code = str(parsed.get("error") or "http_error")
            msg = str(parsed.get("message") or e.reason)
            if e.code == 409 and parsed.get("status") == "superseded":
                raise Superseded(e.code, code, msg, parsed) from None
            raise ServiceError(e.code, code, msg, parsed) from None
        except urlerror.URLError as e:
            raise ServiceError(0, "unreachable", str(e.reason)) from None
        except TimeoutError as e:
            raise ServiceError(0, "timeout", str(e) or "timed out") from None

    def _json(self, method: str, path: str, body: Any = None) -> tuple[int, dict]:
        status, _, raw = self._request(method, path, body, accept="application/json")
        return status, (json.loads(raw.decode("utf-8")) if raw else {})

    def register_manifest(self, manifest: dict) -> dict:
        """POST /v1/manifests: the server computes the hash; returns its body."""
        _, body = self._json("POST", "/v1/manifests", manifest)
        return body

    def get_manifest(self, hsh: str, body: bool = False) -> dict:
        _, out = self._json("GET", f"/v1/manifests/{hsh}" + ("?body=1" if body else ""))
        return out

    def version(self) -> dict:
        return self._json("GET", "/version")[1]

    def start_turn(self, body: dict) -> Stream:
        return self._turn("POST", "/v1/turns", body)

    def tool_results(self, turn_id: str, body: dict) -> Stream:
        return self._turn("POST", f"/v1/turns/{turn_id}/tool-results", body)

    def feedback(self, body: dict) -> None:
        self._request("POST", "/v1/feedback", body)

    def end_conversation(self, site_id: str) -> bool:
        """DELETE /v1/conversations/{siteId} (README §1.1): supersedes the
        site's live turns and ends its open conversation, so the next turn
        starts fresh. The service answers 204 whether or not one was open;
        a 404/501 (a fake without the route) is reported as False."""
        try:
            status, _, _ = self._request("DELETE", f"/v1/conversations/{urlparse.quote(site_id, safe='')}")
        except ServiceError as e:
            if e.status in (404, 501):
                return False
            raise
        return status in (200, 204)

    def _turn(self, method: str, path: str, body: dict) -> Stream:
        accept = "text/event-stream" if self.sse else "application/json"
        _, ctype, raw = self._request(method, path, body, accept=accept)
        text = raw.decode("utf-8")
        if "text/event-stream" in ctype:
            events = parse_sse(text)
            turn_id = next((str(e["data"].get("turnId")) for e in events if isinstance(e.get("data"), dict)
                            and e["data"].get("turnId")), "")
            return Stream(turn_id, events)
        env = json.loads(text) if text else {}
        return Stream(str(env.get("turnId") or ""), list(env.get("events") or []))


# --- the browser side ------------------------------------------------------------------------------------------------------

Engine = Callable[[dict, list[dict], dict], dict]


class DictEngine:
    """engine.Engine (the subprocess wrapper over headless.mjs) as the plain
    callable this module and verify.py expect: `run(state, ops, ctx) -> dict`
    in headless.mjs's output shape plus `state`. An engine that exits 1 (a
    load error, a timeout) becomes `{ok:false, results:[], changes:[],
    error, exit_code:1}` — every call in the batch is then answered
    E_ENGINE and verify.py reads it as `invalid_op`. `fingerprint` is the
    inner engine's (engine.engine_fingerprint), stamped on every row."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.fingerprint = getattr(inner, "fingerprint", None)

    def run(self, state: dict, ops: list[dict], ctx: dict) -> dict:
        try:
            result = self.inner.run(state, ops, ctx, print_state=True)
        except RuntimeError as e:  # engine.EngineError
            return {"ok": False, "results": [], "changes": [], "error": str(e), "exit_code": 1}
        return result.to_dict() if hasattr(result, "to_dict") else dict(result)

    __call__ = run


def default_engine(script: Path | str | None = None, manifest: Path | str | None = None) -> Engine:
    """The real engine: `engine.Engine(path, manifest_path=…)` wrapped as a
    DictEngine. Raises engine.EngineError when node or headless.mjs is
    missing (the CLI reports it)."""
    from . import engine as mod  # lazy: the tests inject a pure-Python engine
    return DictEngine(mod.Engine(script, manifest_path=manifest))


def engine_fingerprint_of(engine: Any) -> str | None:
    """The engine's fingerprint when it has one (engine.Engine, DictEngine);
    None for a test stand-in."""
    fp = getattr(engine, "fingerprint", None)
    return str(fp) if fp else None


def default_verifier(engine: Any | None = None) -> Callable[[dict, dict], Any]:
    """verify.verify in this module's `(case, row)` order; the row's `events`
    are the trace. `engine` is the DictEngine the collector already runs
    (verify replays on a fresh copy of the case state)."""
    from .verify import verify  # lazy: the tests inject one
    return lambda case, row: verify(row, case, engine)


def ctx_for(case: dict) -> dict:
    """The engine context for a case: the golden ctx plus the case's template
    and viewed page (the host fills a missing target.page from currentPage)."""
    state = case.get("site_state") or {}
    template = ((state.get("template") or {}).get("id")) or (case.get("template")) or DEFAULT_TEMPLATE
    ctx = dict(GOLDEN_CTX)
    ctx["templateId"] = template
    page = (case.get("context") or {}).get("page")
    if page:
        ctx["currentPage"] = page
    return ctx


def user_text(context: dict | None, text: str) -> str:
    """The turn's user message as the service renders it (render.py,
    prompt.ContextBlock): the compacted `<context>` JSON on its own lines,
    then the trimmed merchant text."""
    return render.context_block(context, text)


def load_assets(path: Path | None) -> list[dict]:
    if path is None or not Path(path).exists():
        return []
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    items = doc.get("assets") if isinstance(doc, dict) else doc
    return [a for a in (items or []) if isinstance(a, dict) and a.get("public_url")]


def pick_asset_for(assets: Iterable[dict], suspended: dict, case_id: str) -> dict:
    """The fixture asset that answers a pick_asset: one of the asked type
    (`assetType`, default image), chosen deterministically from the case id
    and tool call id so reruns agree. `{public_url, alt, srcset}`."""
    wanted = suspended.get("assetType") or "image"
    pool = [a for a in assets if (a.get("asset_type") or a.get("type") or "image") == wanted] or list(assets)
    if not pool:
        raise ValueError("no fixture assets to answer pick_asset with (fixtures/assets.json)")
    a = pool[stable_hash(f"{case_id}|{suspended.get('toolCallId')}") % len(pool)]
    out = {"public_url": a["public_url"]}
    for k in ("alt", "srcset"):
        if a.get(k):
            out[k] = a[k]
    return out


def tool_results_for(calls: list[dict], engine_result: dict) -> list[dict]:
    """The tool-results items for one batch: the engine's result per op, in
    the order of the tool calls (one `toolCallId` each, exactly once)."""
    results = list(engine_result.get("results") or [])
    claims = engine_result.get("claims") or {}
    out = []
    for i, call in enumerate(calls):
        r = results[i] if i < len(results) else {"ok": False, "error": {"code": "E_ENGINE", "message": "no result"}}
        item: dict[str, Any] = {"toolCallId": str(call.get("id")), "ok": bool(r.get("ok"))}
        if not item["ok"]:
            err = r.get("error") or {"code": "E_OP", "message": "the engine rejected the op"}
            if isinstance(err, str):
                err = {"code": "E_OP", "message": err}
            item["error"] = {"code": str(err.get("code") or "E_OP"), "message": str(err.get("message") or "")}
            if err.get("key"):
                item["error"]["key"] = err["key"]
        if r.get("warnings"):
            item["warnings"] = [str(w) if isinstance(w, str) else str(w.get("code") or w) for w in r["warnings"]]
        if r.get("written"):
            item["written"] = list(r["written"])
        section_id = r.get("sectionId") or (claims.get(str(call.get("id"))) if isinstance(claims, dict) else None)
        if section_id:
            item["sectionId"] = str(section_id)
        out.append(item)
    return out


def case_meta(case: dict) -> dict:
    """What the dataset and verifier read about a case, logged beside the row."""
    meta = case.get("meta") or {}
    target = case.get("target") or {}
    section = meta.get("section_type") or target.get("sectionId") or target.get("panel") or target.get("kind")
    return {"kind": case.get("kind"), "section": section, "template": case.get("template")
            or ((case.get("site_state") or {}).get("template") or {}).get("id"),
            "field_type": meta.get("field_type"), "holdout": bool(case.get("holdout")),
            "manifest_hash": case.get("manifest_hash"), "target": target}


def family_of(case: dict) -> str:
    """A case's family for the adaptive stop: its section type or panel
    (docs/LAB.md §5: family = section type / panel)."""
    return str(case_meta(case).get("section") or f"id:{case.get('id')}")


def _verdict_fields(v: Any) -> dict:
    """verdict, reasons and the verifier's extras, from an object or a dict."""
    if v is None:
        return {"verdict": None, "reasons": None}
    get = (lambda k: v.get(k)) if isinstance(v, dict) else (lambda k: getattr(v, k, None))
    out = {"verdict": get("verdict"), "reasons": list(get("reasons") or [])}
    for k in ("actual_diff", "engine_errors", "warnings"):
        val = get(k)
        if val is not None:
            out[k] = val
    return out


@dataclass
class TurnRecorder:
    """What one turn accumulates while its streams are read."""

    case: dict
    events: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    ops: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    engine_results: list[dict] = field(default_factory=list)
    assets: list[dict] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    names: dict[str, str] = field(default_factory=dict)  # tool call id → tool name (pick_asset renders differently)
    batches: int = 0

    def assistant(self, text: str, calls: list[dict], suspended: dict | None = None) -> None:
        """The assistant turn: text, one tool_use per tool.call and — on a
        turn.suspended — the pick_asset tool_use the service streamed no
        tool.call for (render.pick_asset_call)."""
        blocks: list[dict] = []
        if text:
            blocks.append({"type": "text", "text": text})
        for c in calls:
            self.names[str(c.get("id"))] = str(c.get("name"))
            blocks.append({"type": "tool_use", "id": str(c.get("id")), "name": str(c.get("name")),
                           "input": c.get("input") if isinstance(c.get("input"), dict) else {}})
        if suspended is not None:
            call = render.pick_asset_call(suspended)
            self.names[call["id"]] = PICK_ASSET
            blocks.append(call)
        if blocks:
            self.messages.append({"role": "assistant", "content": blocks})

    def tool_results(self, results: list[dict]) -> None:
        """The posted items as the service renders them for the model
        (render.tool_results_message — loop.browserResult byte for byte)."""
        self.messages.append(render.tool_results_message(results, self.names))

    def posted(self, results: list[dict], turn_id: str | None) -> None:
        """The posted items as the service's trace records them: one
        ``tool.result`` event per item right after the stream it answers
        (README §4 / trace.go: ``data`` = the item plus ``for_seq`` of the
        call it answers — a ``tool.call``'s seq, or the ``turn.suspended``'s
        for ``pick_asset`` — numbered on from the stream's last seq; the
        service's next stream continues after them, so the gap is real).
        With them, ``render.messages_from_events(row.events)`` rebuilds the
        logged ``messages`` exactly — dataset.to_hf's path."""
        seqs: dict[str, Any] = {}
        last = 0
        for e in self.events:
            d = e.get("data") if isinstance(e.get("data"), dict) else {}
            seq = d.get("seq", e.get("seq"))
            if isinstance(seq, int):
                last = max(last, seq)
            if e.get("event") == "tool.call" and d.get("id") is not None:
                seqs[str(d["id"])] = seq
            elif e.get("event") == "turn.suspended" and d.get("toolCallId") is not None:
                seqs[str(d["toolCallId"])] = seq
        for item in results:
            last += 1
            data = dict(item)
            for_seq = seqs.get(str(item.get("toolCallId")))
            if for_seq is not None:
                data["for_seq"] = for_seq
            data["seq"] = last
            if turn_id:
                data["turnId"] = turn_id
            self.events.append({"event": "tool.result", "data": data})


def run_case(case: dict, client: Client, engine: Engine, *, route: str | None = None, sample: int = 1,
             assets: Iterable[dict] = (), verifier: Callable[[dict, dict], Any] | None = None,
             max_batches: int = MAX_BATCHES, build_id: str | None = None, tools: list | None = None,
             store_id: str = "lab", fresh_conversation: bool = True) -> dict:
    """Ask the service one case once, emulating the browser, and return the
    log row. Never raises for a service answer: errors are rows with
    status "error" (resumable).

    Every sample is a FRESH conversation (LAB.md §4): a turn without
    `conversationId` continues the site's open conversation (README §1.1),
    and the lab's `siteId` is `lab-<case id>` for every sample of a case, so
    the previous sample's (or run's) history would otherwise sit in the
    prompt — measured 2026-10-01 against the real binary: the first model
    request of a re-collected case carried the earlier run's four messages.
    `fresh_conversation` ends the site's conversation first
    (`Client.end_conversation`); the row records whether one was ended."""
    case_id = str(case.get("id"))
    context = render.turn_context(case)  # the object sent, logged (row.context) and rendered — one and the same
    text = str(case.get("request") or "")
    site_id = f"lab-{case_id}"
    build_id = build_id or client.build_id
    body = {"siteId": site_id, "storeId": store_id, "manifestHash": case.get("manifest_hash"), "buildId": build_id,
            "context": context, "text": text}
    if route:
        body["model"] = route
    rec = TurnRecorder(case)
    rec.messages.append(render.user_message(context, text))
    state = copy.deepcopy(case.get("site_state") or {})
    ctx = ctx_for(case)
    row: dict[str, Any] = {"id": f"{case_id}#{sample}", "case_id": case_id, "sample": sample, "model": route or "",
                           "status": "ok", "http_status": 200, "code": None, "error": None, "latency_ms": 0,
                           "turn_id": None, "conversation_id": None, "stop": None, "usage": None}
    t0 = time.monotonic()

    def fail(status: int, code: str, message: str) -> None:
        row.update(status="error", http_status=status, code=code, error=message)

    stream = None
    try:
        if fresh_conversation:
            row["conversation_ended"] = client.end_conversation(site_id)
        stream = client.start_turn(body)
    except ServiceError as e:
        fail(e.status, e.code, e.message)
        stream = None
    while stream is not None:
        row["turn_id"] = row["turn_id"] or stream.turn_id or None
        rec.events.extend(stream.events)
        chunk: list[str] = []
        calls: list[dict] = []
        waiting = suspended = done = err = None
        for ev in stream.events:
            name, data = ev.get("event"), ev.get("data") or {}
            if name == "text.delta":
                chunk.append(str(data.get("text") or ""))
            elif name == "tool.call":
                calls.append(data)
            elif name == "turn.waiting":
                waiting = data
            elif name == "turn.suspended":
                suspended = data
            elif name == "turn.done":
                done = data
            elif name == "turn.error":
                err = data
        joined = "".join(chunk)
        if joined:
            rec.text.append(joined)
        rec.tool_calls.extend(calls)
        rec.assistant(joined, calls, suspended)
        if done is not None:
            row["stop"] = done.get("stop")
            row["usage"] = done.get("usage")
            row["conversation_id"] = done.get("conversationId")
            if done.get("stop") not in (None, "end_turn"):
                row["code"] = str(done.get("stop"))
            break
        if err is not None:
            fail(200, str(err.get("code") or "turn_error"), str(err.get("message") or ""))
            break
        if waiting is None and suspended is None:
            fail(200, "stream_ended", "the stream closed without turn.waiting, turn.suspended, turn.done or turn.error")
            break
        rec.batches += 1
        if rec.batches > max_batches:
            fail(200, "max_batches", f"more than {max_batches} tool batches in one turn")
            break
        browser_calls = [c for c in calls if c.get("name") not in SERVER_TOOLS and c.get("name") != PICK_ASSET]
        results: list[dict] = []
        if browser_calls:
            ops = [{"op": c.get("name"), **(c.get("input") if isinstance(c.get("input"), dict) else {})}
                   for c in browser_calls]
            eng = engine(state, ops, ctx)
            if isinstance(eng.get("state"), dict):
                state = eng["state"]
            items = tool_results_for(browser_calls, eng)
            results.extend(items)
            rec.ops.extend(ops)
            for it in items:
                rec.written.extend(it.get("written") or [])
            rec.engine_results.append({k: v for k, v in eng.items() if k != "state"})
        if suspended is not None:
            asset = pick_asset_for(assets, suspended, case_id)
            rec.assets.append(asset)
            results.append({"toolCallId": str(suspended.get("toolCallId")), "ok": True, "asset": asset})
        if not results:
            fail(200, "nothing_to_post", "turn.waiting with no browser op to answer")
            break
        rec.tool_results(results)
        try:
            stream = client.tool_results(row["turn_id"] or stream.turn_id, {"siteId": site_id, "storeId": store_id,
                                                                            "results": results})
        except Superseded as e:
            fail(e.status, "superseded", e.message)
            break
        except ServiceError as e:
            fail(e.status, e.code, e.message)
            break
        rec.posted(results, row["turn_id"] or stream.turn_id)
    row["latency_ms"] = int((time.monotonic() - t0) * 1000)
    row.update({"context": context, "text": text,  # what was sent: render.turn_inputs reads these back for the rebuild
                "messages": rec.messages, "tools": list(tools) if tools is not None else list(TOOL_NAMES),
                "ops": rec.ops, "events": rec.events, "tool_calls": rec.tool_calls, "written_paths": rec.written,
                "engine_results": rec.engine_results, "picked_assets": rec.assets, "reply": "\n".join(rec.text),
                "batches": rec.batches, "verdict": None, "reasons": None, "at": utc_now(),
                "meta": {**case_meta(case), "buildId": build_id, "engine_fingerprint": engine_fingerprint_of(engine)}})
    if verifier is not None:
        row.update(_verdict_fields(verifier(case, row)))
    return row


# --- the adaptive stop (thermometer lab/bestofn.py Adaptive + question_state, copied) --------------------------------------

@dataclass(frozen=True)
class Adaptive:
    """When best-of-N stops sampling a case before its N samples:

      enough     it has `enough` passing samples (default 2);
      hopeless   none of its first `hopeless_after` samples passed (default
                 ⌈N/2⌉) AND its family (section type / panel) has passed
                 under `family_rate` (1/8) of at least `family_min` (8)
                 finished samples;
      excluded   a sample named a reason about the case itself (holdout).

    Copied from the thermometer kit (lab/bestofn.py); bestofn.py in this
    package may carry the same class — `adaptive_classes()` prefers it."""

    enough: int = 2
    hopeless_after: int | None = None
    family_min: int = 8
    family_rate: float = 1 / 8

    def after(self, n: int) -> int:
        return self.hopeless_after if self.hopeless_after is not None else max(1, math.ceil(n / 2))

    def to_dict(self, n: int) -> dict:
        return {"enough": self.enough, "hopeless_after": self.after(n), "family_min": self.family_min,
                "family_rate": self.family_rate}


def sample_state(outcomes: list[bool], n: int, family: tuple[int, int], rule: Adaptive | None,
                 excluded: bool = False) -> str:
    """"open" (sample again), "done" (N samples), or with a rule: "enough",
    "hopeless", "excluded" (thermometer question_state)."""
    if len(outcomes) >= n:
        return "done"
    if rule is None:
        return "open"
    if excluded:
        return "excluded"
    if sum(outcomes) >= rule.enough:
        return "enough"
    passes, samples = family
    if not any(outcomes) and len(outcomes) >= rule.after(n) and samples >= rule.family_min \
            and passes < rule.family_rate * samples:
        return "hopeless"
    return "open"


def adaptive_classes():
    """(Adaptive, state function): bestofn.py's when it has them, else the
    copies above."""
    try:
        from . import bestofn  # noqa: WPS433
        cls = getattr(bestofn, "Adaptive", None)
        fn = getattr(bestofn, "question_state", None) or getattr(bestofn, "sample_state", None)
        if cls is not None and fn is not None:
            return cls, fn
    except ImportError:
        pass
    return Adaptive, sample_state


# --- running many cases ------------------------------------------------------------------------------------------------------

def run_batch(pool: ThreadPoolExecutor, concurrency: int, stop: threading.Event, errors: list[str], fn, jobs) -> None:
    """fn(*job) per job with at most `concurrency` in flight: a job is
    submitted only when a worker is free, so a stop leaves no queue to drain
    (thermometer collect.run_batch)."""
    todo = iter(jobs)
    running: set[Future] = set()
    while True:
        while len(running) < concurrency and not stop.is_set():
            job = next(todo, None)
            if job is None:
                break
            running.add(pool.submit(fn, *job))
        if not running:
            return
        done, running = wait(running, return_when=FIRST_COMPLETED)
        for f in done:
            exc = f.exception()
            if exc is not None:
                stop.set()
                errors.append(repr(exc))


def passed(row: dict) -> bool | None:
    """Whether a finished row passed (None when it has no verdict)."""
    if row.get("status") != "ok" or row.get("verdict") is None:
        return None
    return row["verdict"] in PASS_VERDICTS


def collect(cases: list[dict], client: Client, engine: Engine, log: Path, *, route: str | None = None, samples: int = 1,
            adaptive: Adaptive | None = None, workers: int = 1, max_cases: int | None = None,
            assets: Iterable[dict] = (), verifier: Callable[[dict, dict], Any] | None = None,
            tools: list | None = None, build_id: str | None = None, progress: Callable[[str], None] | None = None,
            settings: dict | None = None) -> dict:
    """Ask every case `samples` times, in rounds (every open case's first
    sample, then every open case's second …), resumable from `log`; with
    `adaptive`, stop a case early (Adaptive). Returns the counts
    `{asked, ok, failed, passed, states:{open|done|enough|hopeless|excluded: n}}`."""
    if not 1 <= workers <= 4:
        raise ValueError("workers must be 1–4")
    assets = list(assets)
    _, state_fn = adaptive_classes() if adaptive is not None else (None, sample_state)
    log = Path(log)
    rows = read_jsonl(log)
    finished: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        if r.get("status") == "ok" and r.get("sample") is not None:
            finished[str(r.get("case_id") or str(r.get("id")).split("#")[0])][int(r["sample"])] = r
    by_id = {str(c.get("id")): c for c in cases}
    fam: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for cid, samples_of in finished.items():
        c = by_id.get(cid)
        if c is None:
            continue
        for r in samples_of.values():
            p = passed(r)
            if p is not None:
                fam[family_of(c)][0] += p
                fam[family_of(c)][1] += 1

    def outcomes(cid: str) -> list[bool]:
        return [p for r in finished[cid].values() if (p := passed(r)) is not None]

    def excluded(cid: str) -> bool:
        return any(set(r.get("reasons") or []) & EXCLUDED_REASONS for r in finished[cid].values())

    def state_of(c: dict) -> str:
        cid = str(c.get("id"))
        outs = outcomes(cid) if adaptive is not None else [True] * len(finished[cid])
        return state_fn(outs, samples, tuple(fam[family_of(c)]), adaptive, excluded(cid))

    todo = [c for c in cases if state_of(c) == "open"]
    if max_cases is not None:
        todo = todo[:max_cases]
    counts: Counter = Counter(asked=0, ok=0, failed=0, passed=0)
    record = {"at": utc_now(), "cases": len(cases), "open": len(todo), "samples": samples, "route": route,
              "workers": workers, "adaptive": adaptive.to_dict(samples) if adaptive is not None else None,
              "already_logged": sum(len(v) for v in finished.values()), **(settings or {})}
    if not todo:
        if progress:
            progress(f"nothing to do: {len(cases)} cases, {record['already_logged']} samples logged in {log}")
        return {**counts, "states": dict(Counter(state_of(c) for c in cases))}
    append_row(runs_log(log), record)
    lock = threading.Lock()
    stop = threading.Event()
    errors: list[str] = []
    t0 = time.monotonic()

    def one(case: dict, k: int) -> None:
        if stop.is_set():
            return
        row = run_case(case, client, engine, route=route, sample=k, assets=assets, verifier=verifier, tools=tools,
                       build_id=build_id)
        with lock:
            append_row(log, row)
            counts["asked"] += 1
            cid = str(case.get("id"))
            if row["status"] == "ok":
                finished[cid][k] = row
                counts["ok"] += 1
                p = passed(row)
                if p is not None:
                    fam[family_of(case)][0] += p
                    fam[family_of(case)][1] += 1
                    counts["passed"] += p
            else:
                counts["failed"] += 1
            if progress:
                mark = (f"error {row.get('code')}" if row["status"] != "ok" else (row.get("verdict") or "no verdict"))
                progress(f"[{counts['asked']}] {row['id']} {mark} · {row['latency_ms'] / 1000:.1f}s · "
                         f"passed {counts['passed']}/{counts['ok']} · {(time.monotonic() - t0) / 60:.1f} min")

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for k in range(1, samples + 1):
            if stop.is_set():
                break
            jobs = [(c, k) for c in todo if k not in finished[str(c.get("id"))] and state_of(c) == "open"]
            run_batch(pool, workers, stop, errors, one, jobs)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if errors:
        raise RuntimeError(errors[0])
    return {**counts, "states": dict(Counter(state_of(c) for c in todo))}


# --- CLI -------------------------------------------------------------------------------------------------------------------------

def load_cases(path: Path, only: str | None = None, skip_holdout: bool = False) -> list[dict]:
    cases = [c for c in read_jsonl(path) if c.get("id")]
    if only:
        keys = {k.strip() for k in only.split(",") if k.strip()}
        cases = [c for c in cases if c.get("kind") in keys or str(c.get("id")) in keys]
    if skip_holdout:
        cases = [c for c in cases if not c.get("holdout")]
    return cases


def connect(url: str, key: str, manifest_path: Path, cases: list[dict], *, sse: bool, timeout_s: float,
            build_id: str) -> tuple[Client, str]:
    """The client after the manifest is registered (once per process) and
    its hash checked against the cases'."""
    client = Client(url, key, timeout_s=timeout_s, sse=sse, build_id=build_id)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    registered = client.register_manifest(manifest)
    hsh = str(registered.get("hash") or "")
    local = manifest_hash(manifest)
    if hsh != local:
        raise ServiceError(200, "hash_mismatch", f"the service hashed the manifest {hsh[:12]}…, this lab {local[:12]}…")
    mismatched = sorted({str(c.get("manifest_hash")) for c in cases if c.get("manifest_hash") not in (None, hsh)})
    if mismatched:
        raise ServiceError(200, "hash_mismatch", f"{len(mismatched)} case hash(es) differ from the registered manifest "
                                                 f"{hsh[:12]}…: regenerate the cases")
    for c in cases:
        if not c.get("manifest_hash"):
            c["manifest_hash"] = hsh
    return client, hsh


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="souschef-lab collect", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="cases JSONL (default: %(default)s)")
    p.add_argument("--log", "--out", dest="log", type=Path, default=DEFAULT_LOG,
                   help="progress log, appended and resumable (default: %(default)s)")
    p.add_argument("--url", default=os.environ.get("SOUS_CHEF_URL", "http://127.0.0.1:8090"),
                   help="service URL (default: $SOUS_CHEF_URL or %(default)s)")
    p.add_argument("--key", default=os.environ.get("SOUS_CHEF_KEY", ""), help="service key (default: $SOUS_CHEF_KEY)")
    p.add_argument("--route", "--model", dest="route", help="model route name sent as `model` (default: the service's)")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="manifest to register (default: %(default)s)")
    p.add_argument("--assets", type=Path, default=DEFAULT_ASSETS, help="fixture assets for pick_asset (default: %(default)s)")
    p.add_argument("--engine", type=Path, default=os.environ.get("ENGINE"), help="headless.mjs path (default: $ENGINE)")
    p.add_argument("--samples", type=int, default=1, metavar="N", help="best-of-N: ask each case N times in rounds")
    p.add_argument("--adaptive", action="store_true", help="stop a case early: enough passes, hopeless, or held out")
    p.add_argument("--enough", type=int, default=2, help="--adaptive: passing samples that are enough (default: 2)")
    p.add_argument("--hopeless-after", type=int, help="--adaptive: failed samples before a hopeless case stops (default ⌈N/2⌉)")
    p.add_argument("--workers", type=int, default=1, help="cases in flight at once, 1–4 (default: 1)")
    p.add_argument("--max-cases", type=int, help="ask at most this many open cases")
    p.add_argument("--only", help="comma-separated kinds or case ids")
    p.add_argument("--skip-holdout", action="store_true", help="do not ask held-out (exam) cases")
    p.add_argument("--sse", action="store_true", help="read SSE frames instead of the JSON envelope")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds per HTTP call (default: %(default)s)")
    p.add_argument("--build-id", default="lab", help="buildId sent with every turn (default: %(default)s)")
    p.add_argument("--no-verify", action="store_true", help="log rows without a verdict (verify later)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples < 1:
        print("--samples must be at least 1", file=sys.stderr)
        return 2
    if not 1 <= args.workers <= 4:
        print("--workers must be 1–4", file=sys.stderr)
        return 2
    if not args.key:
        print("a service key is required (--key or $SOUS_CHEF_KEY)", file=sys.stderr)
        return 2
    cases = load_cases(args.cases, args.only, args.skip_holdout)
    if not cases:
        print(f"no cases in {args.cases} — run `souschef-lab cases` first", file=sys.stderr)
        return 2
    try:
        client, _ = connect(args.url, args.key, args.manifest, cases, sse=args.sse, timeout_s=args.timeout,
                            build_id=args.build_id)
    except (ServiceError, OSError, json.JSONDecodeError) as e:
        print(f"cannot start: {e}", file=sys.stderr)
        return 1
    try:
        engine = default_engine(args.engine, args.manifest)
    except RuntimeError as e:  # engine.EngineError: no node / no headless.mjs
        print(f"cannot start: {e}", file=sys.stderr)
        return 1
    verifier = None if args.no_verify else default_verifier(engine)
    rule_cls, _ = adaptive_classes()
    rule = rule_cls(enough=args.enough, hopeless_after=args.hopeless_after) if args.adaptive else None
    try:
        counts = collect(cases, client, engine, args.log, route=args.route, samples=args.samples, adaptive=rule,
                         workers=args.workers, max_cases=args.max_cases, assets=load_assets(args.assets),
                         verifier=verifier, build_id=args.build_id, progress=print,
                         settings={"cases_file": str(args.cases), "sse": args.sse, "engine": str(args.engine or "")})
    except KeyboardInterrupt:
        print(f"\nstopped (Ctrl-C) — rows already logged in {args.log}; run again to resume", file=sys.stderr)
        return INTERRUPTED
    except RuntimeError as e:
        print(f"stopped: {e}", file=sys.stderr)
        return 1
    print(f"asked {counts['asked']}: {counts['ok']} answered, {counts['failed']} failed, {counts['passed']} passed. "
          f"Log: {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
