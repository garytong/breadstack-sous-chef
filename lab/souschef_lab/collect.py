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
   conversation_ended, batches, at, candidate_fields?, fields_context,
   fields_context_by, routes, escalated,
   meta:{kind, section, template, field_type, holdout, manifest_hash, buildId,
   engine_fingerprint}}

`messages` is a cache: render.messages_from_events(events, context, text,
candidate_fields) rebuilds it exactly (dataset.to_hf reads only the rebuild).

Candidate fields (`--fields-context auto|on|off`, LAB.md §5): a route with
`fields_context` on (the default for every route kind since 2026-10-10,
README §5) gets a `<candidate_fields>` block in the user turn. The
collector cannot see the prompt, so the SERVICE says: `GET /version` states
each route's `{kind, fields_context, fallback}` (`route_info`, by name, and
`default_route`) and `turn.done` carries the turn's (`route.fields_context`). `auto` (the default) follows /version; `on` /
`off` are refused when /version says otherwise, and a `turn.done` that
contradicts what the collector rendered stops the run (FieldsContextMismatch)
— a row's `messages` are never silently another prompt than the model read.
With the block on, the collector computes the same block with the service's
own code (fields.py; needs go) and logs it as `row.candidate_fields` ("" when
nothing matched). `row.fields_context` records it, `row.fields_context_by`
whether the service's turn.done said so ("service") or only the flag ("flag":
a service older than the statement, or an errored turn).

Hybrid (README §4): on a route with a `fallback` the service may hand the turn
to the fallback route once. The row records it — `routes` (turn.done's list:
the routes that answered, in order; None on a route without a fallback) and
`escalated` (`{from, to, reason}` of the turn.escalated event, else None) —
and `hybrid_summary` reports the student-only pass (labelled with the model
id the service reported for the route — `row.models`, from turn.done — else
the route's name; never a fixed name), the hybrid pass and the escalation
rate. An escalated row is Claude-completed: dataset.py keeps it
as a teacher row, flagged, never as the student's. Every sample is
a fresh conversation: the site's open one is ended first (README §1.1 would
otherwise continue it — see run_case).

Engine contract (engine.py, another module): `engine(state, ops, ctx)`
returns `{ok, results:[{op, ok, error?, warnings, written, sectionId?, slug?}],
changes:[...], state}` — headless.mjs's output with `--print-state`. Any
callable of that shape works; the tests use a pure-Python one. A batch is
applied as the browser applies it (createpage.run_batch): a call aimed at
the page a refused create_page of the same batch would have made is
answered E_TARGET and never reaches the engine; a create's `slug` is posted
with its result (client/turns.js toToolResult).

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

from . import createpage, render
from .state import case_asset_id

LAB_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = LAB_DIR / "fixtures" / "manifest.snapshot.json"
DEFAULT_ASSETS = LAB_DIR / "fixtures" / "assets.json"
DEFAULT_CASES = LAB_DIR / "data" / "cases.jsonl"
DEFAULT_LOG = LAB_DIR / "runs" / "collect-log.jsonl"

# README §2: the eleven tools in the service's fixed order (toolgen.Order; fixtures/prompt/tools.json).
TOOL_NAMES = ("set_fields", "edit_list", "reset", "restore_default", "set_page_layout", "create_page",
              "reset_pinned_section", "pick_asset", "lookup_fields", "describe_target", "report_unsupported")
# Answered by the service itself; the browser never posts a result for them
# (report_unsupported is surfaced as a tool.call for display only).
SERVER_TOOLS = frozenset(("lookup_fields", "describe_target", "report_unsupported"))
PICK_ASSET = "pick_asset"
# The ops the browser applies through its engine (toolgen runsOf = RunsBrowser): every tool that is neither
# answered by the service nor the suspending pick_asset — by name, never by position in TOOL_NAMES.
BROWSER_OPS = frozenset(n for n in TOOL_NAMES if n not in SERVER_TOOLS and n != PICK_ASSET)
MAX_BATCHES = 8  # the service's maxCalls per turn (README §4)
# The verifier's golden engine context (docs/LAB.md §3, FACTS 2 §1).
GOLDEN_CTX = {"cdnUrl": "https://cdn.int.breadstack.io", "galleryFolder": "gal", "now": 1700000000000}
DEFAULT_TEMPLATE = "editorial"
DEFAULT_TIMEOUT_S = 300.0
INTERRUPTED = 130
PASS_VERDICTS = frozenset(("pass", "clarify_ok", "unsupported_ok"))
# verify.FAILING: the reasons that are verdicts (the rest of a row's `reasons` are notes such as turn_error:<code>).
FAILING_REASONS = frozenset(("invalid_op", "no_answer", "wrong_value", "missing_writes", "extra_writes"))
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
    """The engine context for a case — the very ctx the verifier replays with
    (``verify.ctx_of``): the golden ctx, the case's template and viewed page
    (the host fills a missing target.page from currentPage), its locations,
    and the host's ``pageSlugs`` / carousel ``tags`` / ``brands``
    (host/previewApi buildCtx: create_page refuses a renamed built-in slug,
    and a tagSlug / brandSlug write derives tagId / brandId from the lists)."""
    from .verify import ctx_of  # lazy: verify imports nothing from here
    return ctx_of(case)


# --- what the service says about a route (GET /version, turn.done) -----------------------------------------------------

# /version: `routes` stays the list of names; the detail sits beside it — the service's `route_info`
# (name → {kind, fields_context, fallback}, with `default_route`), or a list of {name, kind, fields_context, fallback}.
ROUTE_DETAIL_KEYS = ("route_info", "routes_detail", "route_details", "routesDetail", "routeDetails")
DEFAULT_ROUTE = "default"


class FieldsContextMismatch(RuntimeError):
    """The service says the route's user turn is not what the collector
    rendered (candidate-fields block on one side, none on the other)."""


def on_off(value: Any) -> bool | None:
    """A stated `fields_context` as a bool: true/false, "on"/"off"; None when
    it is not stated (absent, null, or a shape this lab does not know)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("on", "off", "true", "false"):
        return value.strip().lower() in ("on", "true")
    return None


def route_details(version: dict | None) -> dict[str, dict]:
    """name → `{name, kind, fields_context, fallback}` from a /version
    answer; {} when the service states none (a build before the statement).
    A `routes` list of objects is read too."""
    version = version if isinstance(version, dict) else {}
    for key in ROUTE_DETAIL_KEYS + ("routes",):
        items = version.get(key)
        if isinstance(items, dict):  # name → detail
            items = [{"name": k, **v} for k, v in items.items() if isinstance(v, dict)]
        if isinstance(items, list) and items and all(isinstance(x, dict) and x.get("name") for x in items):
            return {str(x["name"]): x for x in items}
    return {}


def route_info(version: dict | None, route: str | None) -> dict | None:
    """The stated detail of `route` (None = the service's default route:
    /version's `default_route` when it names one, else "default", else the
    only route there is). None when the service does not state it."""
    details = route_details(version)
    if not details:
        return None
    if route:
        return details.get(route)
    named = (version or {}).get("default_route")
    if isinstance(named, str) and named in details:
        return details[named]
    if DEFAULT_ROUTE in details:
        return details[DEFAULT_ROUTE]
    return next(iter(details.values())) if len(details) == 1 else None


def done_fields_context(done: dict, requested: str | None = None) -> bool | None:
    """turn.done's statement for the turn's user message: `route.fields_context`
    (the service's `route` = `{name, fields_context, fallback}` of the route
    the turn STARTED on — the block is rendered once, for that route;
    loop.candidateFields), else a top-level `fields_context`: a bool (or
    on/off), or a per-route object — then the requested route's (the first
    of `routes`). None when turn.done does not say."""
    started = done.get("route")
    if isinstance(started, dict) and on_off(started.get("fields_context")) is not None:
        return on_off(started.get("fields_context"))
    value = done.get("fields_context")
    if isinstance(value, dict):
        routes = done.get("routes") if isinstance(done.get("routes"), list) else []
        for name in ([requested] if requested else []) + [str(r) for r in routes[:1]] + [DEFAULT_ROUTE]:
            if name in value:
                return on_off(value[name])
        return on_off(next(iter(value.values()))) if len(value) == 1 else None
    return on_off(value)


def done_models(done: dict) -> dict[str, str]:
    """route name → the model id that route's calls of the turn were REPORTED
    under (the service's usage rows carry the same id; README §5
    `report_as`): turn.done's `route.model` for the route the turn started
    on, and `models` (a hybrid turn: every route it called). {} from a build
    that does not say."""
    out: dict[str, str] = {}
    started = done.get("route")
    if isinstance(started, dict) and started.get("name") and started.get("model"):
        out[str(started["name"])] = str(started["model"])
    if isinstance(done.get("models"), dict):
        out.update({str(k): str(v) for k, v in done["models"].items() if v})
    return out


def user_text(context: dict | None, text: str, candidates: str = "") -> str:
    """The turn's user message as the service renders it (render.py,
    prompt.UserText): the compacted `<context>` JSON on its own lines, the
    candidate-fields block when the route sends one, then the trimmed
    merchant text."""
    return render.context_block(context, text, candidates)


def load_assets(path: Path | None) -> list[dict]:
    if path is None or not Path(path).exists():
        return []
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    items = doc.get("assets") if isinstance(doc, dict) else doc
    return [a for a in (items or []) if isinstance(a, dict) and a.get("public_url")]


def pick_asset_for(assets: Iterable[dict], suspended: dict, case_id: str, asset_id: str | None = None) -> dict:
    """The fixture asset that answers a pick_asset: the asset the case names
    (`asset_id` = the case's `meta.asset_id`, `state.case_asset_id`) when it
    is of the asked type (`assetType`, default image), else one of that type
    chosen from the case id alone — `state.fixture_asset_for`, the same
    function the case generator uses for the expected ops (LAB.md §2/§4), so
    the request text, the expectation and reruns agree. `{public_url, alt,
    srcset}` (empty alt/srcset omitted, as the real picker's api.Asset)."""
    from .state import fixture_asset_for
    a = fixture_asset_for(list(assets), case_id, str(suspended.get("assetType") or "image"), asset_id)
    out = {"public_url": a["public_url"]}
    for k in ("alt", "srcset"):
        if a.get(k):
            out[k] = a[k]
    return out


def call_section_id(call: dict) -> str | None:
    """turns.js toToolResult's `sectionId`: the call input's `target.sectionId || sectionId` (JS truthiness; the
    service's ToolResult.SectionID is a string, so only a non-empty string is ever posted)."""
    inp = call.get("input") if isinstance(call.get("input"), dict) else {}
    tgt = inp.get("target") if isinstance(inp.get("target"), dict) else {}
    for v in (tgt.get("sectionId"), inp.get("sectionId")):
        if v:
            return v if isinstance(v, str) else None
    return None


def tool_results_for(calls: list[dict], engine_result: dict) -> list[dict]:
    """The tool-results items for one batch: the engine's result per op, in
    the order of the tool calls (one `toolCallId` each, exactly once).
    `sectionId` is the browser's echo of the call's OWN target section
    (call_section_id; turns.js toToolResult since sous-chef v2, 8a9b6eee), on
    ok results only — never the engine's `sectionId` / `claims` (a new
    section's id), which the live browser does not post."""
    results = list(engine_result.get("results") or [])
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
        if item["ok"]:
            # The browser's `hidden: [{key, by, byValue}]` (shared contract 4;
            # launcher client/turns.js wireHidden), only when non-empty.
            hidden = render.wire_hidden(r.get("hidden"))
            if hidden:
                item["hidden"] = hidden
        section_id = call_section_id(call) if item["ok"] else None
        if section_id:
            item["sectionId"] = section_id
        if item["ok"] and isinstance(r.get("slug"), str) and r["slug"]:
            # create_page: the new page's final slug (turns.js toToolResult), which the service forwards to the
            # model (api.ToolResult.Slug → loop.browserResult) and later calls of the reply target.
            item["slug"] = r["slug"]
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
             store_id: str = "lab", fresh_conversation: bool = True, fields: Any | None = None,
             site_prefix: str = "lab-") -> dict:
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
    (`Client.end_conversation`); the row records whether one was ended.

    `fields` (a fields.Fields on the registered manifest) says the route
    sends the candidate-fields block: the same block is computed here and
    logged (`row.candidate_fields`), so `messages` are the bytes the model
    read. None = the route sends none (an Anthropic route). The service has
    the last word: a `turn.done` whose `fields_context` says the opposite
    raises FieldsContextMismatch (the one exception this function raises for
    a service answer — such a row would log a prompt the model never read).

    `site_prefix` (default `lab-`; it must start with `lab-`, the lab's own
    site ids) lets two runs of the same rows — a baseline and a candidate
    revision sharing one database — keep separate conversations."""
    if not site_prefix.startswith("lab-"):
        raise ValueError(f"site_prefix {site_prefix!r}: the lab only uses its own site ids (lab-…)")
    case_id = str(case.get("id"))
    context = render.turn_context(case)  # the object sent, logged (row.context) and rendered — one and the same
    text = str(case.get("request") or "")
    site_id = f"{site_prefix}{case_id}"
    build_id = build_id or client.build_id
    body = {"siteId": site_id, "storeId": store_id, "manifestHash": case.get("manifest_hash"), "buildId": build_id,
            "context": context, "text": text}
    if route:
        body["model"] = route
    rec = TurnRecorder(case)
    candidates = fields.block(context, text) if fields is not None else None
    rec.messages.append(render.user_message(context, text, candidates or ""))
    state = copy.deepcopy(case.get("site_state") or {})
    ctx = ctx_for(case)
    row: dict[str, Any] = {"id": f"{case_id}#{sample}", "case_id": case_id, "sample": sample, "model": route or "",
                           "status": "ok", "http_status": 200, "code": None, "error": None, "latency_ms": 0,
                           "turn_id": None, "conversation_id": None, "stop": None, "usage": None,
                           "routes": None, "escalated": None,
                           "fields_context": fields is not None, "fields_context_by": "flag"}
    if candidates is not None:
        row["candidate_fields"] = candidates
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
            elif name == "turn.escalated" and row["escalated"] is None:  # at most once per turn (README §4)
                row["escalated"] = {k: data.get(k) for k in ("from", "to", "reason")}
        joined = "".join(chunk)
        if joined:
            rec.text.append(joined)
        rec.tool_calls.extend(calls)
        rec.assistant(joined, calls, suspended)
        if done is not None:
            row["stop"] = done.get("stop")
            row["usage"] = done.get("usage")
            row["conversation_id"] = done.get("conversationId")
            if isinstance(done.get("routes"), list):  # only on a route with a fallback
                row["routes"] = [str(r) for r in done["routes"]]
            models = done_models(done)
            if models:  # route name → the model id its calls were reported under (README §5 report_as)
                row["models"] = models
            if done.get("stop") not in (None, "end_turn"):
                row["code"] = str(done.get("stop"))
            stated = done_fields_context(done, route)
            if stated is not None:
                if stated != (fields is not None):
                    raise FieldsContextMismatch(
                        f"turn {row['turn_id']} ({row['id']}): the service says fields_context is "
                        f"{'on' if stated else 'off'} for route {route or 'the default'}, but the collector rendered the "
                        f"user turn {'with' if fields is not None else 'without'} the candidate-fields block — run with "
                        f"--fields-context auto (or {'on' if stated else 'off'})")
                row["fields_context_by"] = "service"
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
            # As the browser applies a batch: a call aimed at the page a refused create_page of this batch
            # would have made is stopped (E_TARGET, never applied) — turns.js applyOps (createpage.run_batch).
            eng = createpage.run_batch(lambda batch, base=state: engine(copy.deepcopy(base), batch, ctx), ops)
            if isinstance(eng.get("state"), dict):
                state = eng["state"]
            items = tool_results_for(browser_calls, eng)
            results.extend(items)
            rec.ops.extend(ops)
            for it in items:
                rec.written.extend(it.get("written") or [])
            rec.engine_results.append({k: v for k, v in eng.items() if k != "state"})
        if suspended is not None:
            asset = pick_asset_for(assets, suspended, case_id, case_asset_id(case))
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


# --- hybrid: the student's route with a fallback (README §4) ----------------------------------------------------------------

def escalation_of(row: dict) -> dict | None:
    """`{from, to, reason}` when the turn was handed to the fallback route:
    the row's `escalated`, else its `turn.escalated` event (a trace, or a row
    logged without the field)."""
    esc = row.get("escalated")
    if isinstance(esc, dict) and esc:
        return esc
    for e in row.get("events") or []:
        if isinstance(e, dict) and (e.get("event") or e.get("type")) == "turn.escalated":
            d = e.get("data") or {}
            return {k: d.get(k) for k in ("from", "to", "reason")}
    return None


def routes_of(row: dict) -> list[str]:
    """The routes that answered the turn's model calls, in order: turn.done's
    `routes`, else the row's requested route followed by the fallback it
    escalated to."""
    if isinstance(row.get("routes"), list) and row["routes"]:
        return [str(r) for r in row["routes"]]
    esc = escalation_of(row)
    out = [str(row.get("model"))] if row.get("model") else []
    if esc and esc.get("to"):
        out = [str(esc.get("from") or (out[0] if out else "")), str(esc["to"])]
    return out


def completed_by(row: dict) -> str:
    """The route whose model finished the turn (the fallback once escalated)."""
    esc = escalation_of(row)
    if esc and esc.get("to"):
        return str(esc["to"])
    routes = routes_of(row)
    return routes[-1] if routes else str(row.get("model") or "")


def student_of(rows: Iterable[dict]) -> dict:
    """Who "the student" of a run is, from the rows — never a fixed name:
    `route` = the route the turns were asked on (an escalation's `from`, the
    row's requested route, else the first of turn.done's `routes`), the most
    common one; `model` = the model id the service reported that route's
    calls under (turn.done, `row.models`), None when no row states it."""
    names: Counter = Counter()
    reported: dict[str, Counter] = {}
    for r in rows:
        esc = escalation_of(r) or {}
        routes = r.get("routes") if isinstance(r.get("routes"), list) else []
        name = str(esc.get("from") or r.get("model") or (routes[0] if routes else "") or "")
        models = r.get("models") if isinstance(r.get("models"), dict) else {}
        if not name and len(models) == 1:  # the default route, named only by turn.done
            name = next(iter(models))
        if not name:
            continue
        names[name] += 1
        if models.get(name):
            reported.setdefault(name, Counter())[str(models[name])] += 1
    if not names:
        return {"route": None, "model": None}
    route = names.most_common(1)[0][0]
    return {"route": route, "model": reported[route].most_common(1)[0][0] if route in reported else None}


def student_label(h: dict) -> str:
    """What a summary line calls the student: the model id the service
    reported, else the route's name, else "student"."""
    s = h.get("student") or {}
    return str(s.get("model") or s.get("route") or "student")


def refused_after_escalation(row: dict) -> int:
    """Browser ops refused AFTER the turn escalated — the fallback's own
    refusals (the student's come before the turn.escalated event)."""
    n, seen = 0, False
    for e in row.get("events") or []:
        name = e.get("event") or e.get("type")
        if name == "turn.escalated":
            seen = True
        elif seen and name == "tool.result" and not (e.get("data") or {}).get("ok"):
            n += 1
    return n


def outcome_passed(row: dict) -> bool | None:
    """Whether the turn's RESULT is right, for the hybrid pass.

    The verifier is strict about the path: one refused op makes the row
    `invalid_op` even when the turn then lands the expected diff (a refused
    op applies nothing). On an escalated turn the student's refused op is
    exactly what the fallback is there to correct, so it is forgiven here:
    the row passes when `invalid_op` is its ONLY failing reason and every
    refusal came before the escalation (a refusal of the fallback's own op
    stays a failure, as on a Claude-only route). A row that did not escalate
    is judged as always."""
    p = passed(row)
    if p is None or p or not escalation_of(row):
        return p
    failing = [r for r in row.get("reasons") or [] if r in FAILING_REASONS]
    return failing == ["invalid_op"] and refused_after_escalation(row) == 0


def hybrid_summary(rows: Iterable[dict]) -> dict:
    """The three figures of a run on a route with a fallback, in points
    (0–100) over the SCORED rows (a row with a verdict):

      student_only  rows that never escalated and passed — what the student
                    did by itself (`pass_rate` over all scored rows;
                    `pass_rate_unescalated` over the rows it kept)
      hybrid        what the merchant gets: those, plus the escalated rows the
                    fallback finished correctly (`outcome_passed`);
                    `strict_pass_rate` is the verifier's verdict as is
      escalation_rate  escalated rows over all rows, with the reasons

    `fallback_seen` is False when no row names a fallback (no `routes`, no
    escalation): the route has none and the figures equal the plain pass.
    `student` = `{route, model}`: the route the run was asked on and the
    model id the service reported for it (`student_of`) — the label the
    printed line uses."""
    rows = list(rows)
    scored = [r for r in rows if passed(r) is not None]
    esc = [r for r in rows if escalation_of(r)]
    alone = [r for r in scored if not escalation_of(r)]
    alone_pass = sum(1 for r in alone if passed(r))
    esc_scored = [r for r in scored if escalation_of(r)]
    esc_pass = sum(1 for r in esc_scored if outcome_passed(r))
    strict = sum(1 for r in scored if passed(r))

    def pts(a: int, b: int) -> float | None:
        return round(100.0 * a / b, 1) if b else None

    return {"rows": len(rows), "scored": len(scored),
            "fallback_seen": bool(esc) or any(r.get("routes") for r in rows),
            "escalated": len(esc), "escalation_rate": pts(len(esc), len(rows)),
            "reasons": dict(sorted(Counter(str((escalation_of(r) or {}).get("reason")) for r in esc).items())),
            "student": student_of(rows),
            "student_only": {"n": len(alone), "passed": alone_pass, "pass_rate": pts(alone_pass, len(scored)),
                             "pass_rate_unescalated": pts(alone_pass, len(alone))},
            "escalated_rows": {"scored": len(esc_scored), "passed": esc_pass, "pass_rate": pts(esc_pass, len(esc_scored))},
            "hybrid": {"passed": alone_pass + esc_pass, "pass_rate": pts(alone_pass + esc_pass, len(scored)),
                       "strict_passed": strict, "strict_pass_rate": pts(strict, len(scored))}}


def format_hybrid(h: dict) -> str:
    s, y = h["student_only"], h["hybrid"]
    reasons = ", ".join(f"{k} {v}" for k, v in h["reasons"].items()) or "none"
    return (f"{student_label(h)}-only pass {s['passed']}/{h['scored']} ({s['pass_rate']} pts; {s['pass_rate_unescalated']} of the "
            f"{s['n']} it kept) · hybrid pass {y['passed']}/{h['scored']} ({y['pass_rate']} pts; strict "
            f"{y['strict_pass_rate']}) · escalated {h['escalated']}/{h['rows']} ({h['escalation_rate']}%: {reasons})")


def latest_rows(rows: Iterable[dict]) -> list[dict]:
    """One row per sample id, the LAST logged: an error row that a later
    start asked again (errors are resumable) is not the sample's outcome and
    must not be counted beside its retry."""
    last: dict[str, dict] = {}
    anonymous: list[dict] = []
    for r in rows:
        if r.get("id") is None:
            anonymous.append(r)
        else:
            last.pop(str(r["id"]), None)  # re-insert: keep the log's order of the kept rows
            last[str(r["id"])] = r
    return anonymous + list(last.values())


def collect(cases: list[dict], client: Client, engine: Engine, log: Path, *, route: str | None = None, samples: int = 1,
            adaptive: Adaptive | None = None, workers: int = 1, max_cases: int | None = None,
            assets: Iterable[dict] = (), verifier: Callable[[dict, dict], Any] | None = None,
            tools: list | None = None, build_id: str | None = None, progress: Callable[[str], None] | None = None,
            settings: dict | None = None, fields: Any | None = None) -> dict:
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
    counts: Counter = Counter(asked=0, ok=0, failed=0, passed=0, escalated=0)
    record = {"at": utc_now(), "cases": len(cases), "open": len(todo), "samples": samples, "route": route,
              "fields_context": fields is not None,  # settings.fields_context_by / route_info say who decided (main)
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
                       build_id=build_id, fields=fields)
        with lock:
            append_row(log, row)
            counts["asked"] += 1
            counts["escalated"] += bool(row.get("escalated"))
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
                if row.get("escalated"):
                    mark += f" · escalated→{row['escalated'].get('to')} ({row['escalated'].get('reason')})"
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
    p.add_argument("--fields-context", choices=FIELDS_FLAGS, default="auto",
                   help="whether the route sends the candidate-fields block (fields_context): auto = what the service's "
                        "/version says for the route; on / off = say it yourself (refused when /version says "
                        "otherwise). With the block on, the same block is computed and logged with the row; needs go "
                        "(default: %(default)s)")
    return p


FIELDS_FLAGS = ("auto", "on", "off")


def fields_for(flag: str, manifest: Path) -> Any | None:
    """The fields.Fields of `--fields-context on` (None when off). Raises
    RuntimeError (fields.FieldsError) when the CLI cannot be had."""
    if flag != "on":
        return None
    from . import fields as mod  # lazy: only this flag needs go
    return mod.Fields(manifest)


def resolve_fields(flag: str, manifest: Path, client: Client, route: str | None) -> tuple[Any | None, dict]:
    """`(fields, record)` for a run on `route`: the truth comes from the
    service. /version's detail of the route decides `auto` and vetoes an
    `on`/`off` that contradicts it; a service that states nothing (a build
    before the statement) cannot answer `auto`, and an explicit flag is then
    taken on trust (turn.done still checks every turn when it says).
    `record` = `{fields_context, fields_context_by, route_info}` for the
    run's record. Raises RuntimeError (FieldsContextMismatch, or
    fields.FieldsError when the block is on and the CLI cannot be had)."""
    if flag not in FIELDS_FLAGS:
        raise ValueError(f"fields_context must be one of {FIELDS_FLAGS}")
    try:
        version = client.version()
    except (ServiceError, OSError, json.JSONDecodeError):
        version = {}
    name = route or "the default route"
    info = route_info(version, route)
    details = route_details(version)
    if details and info is None and route:
        raise FieldsContextMismatch(f"the service's /version states its routes ({', '.join(sorted(details))}) and "
                                    f"{name} is not one of them")
    stated = on_off(info.get("fields_context")) if info else None
    if flag == "auto":
        if stated is None:
            raise FieldsContextMismatch(
                f"--fields-context auto: the service's /version does not state fields_context for {name}"
                + (f" (it lists {', '.join(sorted(details))} and does not say which is the default) — name the route "
                   f"with --route, or" if details else " (a build before the statement?) —")
                + " say it yourself: --fields-context on when the route sends the candidate-fields block (the default "
                  "of a messages-compatible route), off when it does not")
        on = stated
    else:
        on = flag == "on"
        if stated is not None and stated != on:
            raise FieldsContextMismatch(
                f"--fields-context {flag}, but the service's /version says fields_context is "
                f"{'on' if stated else 'off'} for {name}: the logged messages would not be what the model reads — "
                f"drop the flag (auto follows the service)")
    record = {"fields_context": on, "fields_context_by": "service" if stated is not None else "flag",
              "route_info": {k: info.get(k) for k in ("name", "kind", "fields_context", "fallback", "report_as")} if info else None}
    return (fields_for("on", manifest) if on else None), record


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
    try:
        fields, fields_record = resolve_fields(args.fields_context, args.manifest, client, args.route)
    except RuntimeError as e:  # FieldsContextMismatch; fields.FieldsError: no go / no CLI
        print(f"cannot start: {e}", file=sys.stderr)
        return 1
    print(f"fields_context {'on' if fields is not None else 'off'} for {args.route or 'the default route'} "
          f"(says {'the service' if fields_record['fields_context_by'] == 'service' else '--fields-context'})")
    verifier = None if args.no_verify else default_verifier(engine)
    rule_cls, _ = adaptive_classes()
    rule = rule_cls(enough=args.enough, hopeless_after=args.hopeless_after) if args.adaptive else None
    try:
        counts = collect(cases, client, engine, args.log, route=args.route, samples=args.samples, adaptive=rule,
                         workers=args.workers, max_cases=args.max_cases, assets=load_assets(args.assets),
                         verifier=verifier, build_id=args.build_id, progress=print, fields=fields,
                         settings={"cases_file": str(args.cases), "sse": args.sse, "engine": str(args.engine or ""),
                                   "fields_context_by": fields_record["fields_context_by"],
                                   "route_info": fields_record["route_info"]})
    except KeyboardInterrupt:
        print(f"\nstopped (Ctrl-C) — rows already logged in {args.log}; run again to resume", file=sys.stderr)
        return INTERRUPTED
    except RuntimeError as e:
        print(f"stopped: {e}", file=sys.stderr)
        return 1
    print(f"asked {counts['asked']}: {counts['ok']} answered, {counts['failed']} failed, {counts['passed']} passed. "
          f"Log: {args.log}")
    hybrid = hybrid_summary(latest_rows(read_jsonl(args.log)))
    if hybrid["fallback_seen"]:  # the whole log, earlier starts included — a retried error counted once, as its retry
        print(f"hybrid ({hybrid['rows']} samples logged): {format_hybrid(hybrid)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
