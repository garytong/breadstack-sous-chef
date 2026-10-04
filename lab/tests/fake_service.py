"""A fake Sous Chef service for the lab's tests (docs/LAB.md §9).

Stands in for the Go service (README §1) on 127.0.0.1:<random port>, with
Python's `http.server` and nothing else — no network, no model, no API
spend. It speaks the parts of the API the collector touches:

  POST /v1/manifests                computes the §1.1 hash (sha256 of the
                                    canonical JSON minus generated_from and
                                    coverage) and returns {hash, fields,
                                    lists, sections, globalPanels, pages,
                                    details, generated_from, diff}; 201 the
                                    first time, 200 after; 413 above the
                                    manifest body limit
  PUT  /v1/manifests/{hash}         the same, 400 hash_mismatch on a miss
  GET  /v1/manifests/{hash}         {hash, generated_from, counts, first_seen,
                                    index}; ?body=1 adds the manifest
  POST /v1/turns                    replays the scripted turn for the body's
                                    siteId ("lab-<case id>") or text, as the
                                    JSON envelope {turnId, events} or as
                                    SSE frames, by the Accept header; 400 on
                                    a missing required field, 409
                                    unknown_manifest, 413 above the body cap
  POST /v1/turns/{id}/tool-results  records the body (the SINK the tests
                                    assert on) and returns the next
                                    scripted stream; 409 turn_not_waiting
                                    (status superseded when the script says
                                    so); a pending pick_asset must be
                                    answered with `asset`
  POST /v1/feedback                 204, recorded
  GET  /healthz /readyz /version    open

A SCRIPT is a list of streams; a stream is a list of (event, data) pairs
without turnId/seq — the server injects both, seq continuing across the
turn's streams as the real service does (README §1.2), leaving one seq per
posted tool result between streams (the real trace records a `tool.result`
event there; the collector fills the gap in its own row). A stream entry equal
to the string "superseded" makes the tool-results call that would have
returned it answer 409 {error: turn_not_waiting, status: superseded}
instead. Scripts are given in Python (`server.scripts[case_id] = [...]`) or
loaded from a directory of `<case id>.json` files (`load_scripts`), the
shape docs/LAB.md gives fixtures/traces/. A `Rotation([script, script, …])`
(JSON: `{"rotate": [script, …]}`) serves a different script per turn of
the same case, round-robin — best-of-N samples that pass once and fail once.

Usage in a test:

    with FakeService(manifest=MANIFEST_DICT, key="k") as svc:
        svc.scripts["case-1"] = [[("text.delta", {"text": "ok"}),
                                  ("tool.call", {"id": "t1", "name": "set_fields", "input": {...}}),
                                  ("turn.waiting", {"toolCallIds": ["t1"]})],
                                 [("text.delta", {"text": "Done."}),
                                  ("turn.done", {"usage": {...}, "stop": "end_turn"})]]
        ... client against svc.url ...
        assert svc.tool_results[0]["body"]["results"][0]["ok"]
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

MANIFEST_BODY_LIMIT = 8 * 1024 * 1024  # README §1: manifest bodies ≤ 8 MiB
MAX_BODY_BYTES = 256 * 1024  # README §1: every other JSON body ≤ 256 KB
TURN_REQUIRED = ("siteId", "storeId", "manifestHash", "buildId")
SUPERSEDED = "superseded"


def manifest_hash(doc: dict) -> str:
    """README §1.1 in Python: sha256 of the canonical JSON (sorted keys, no
    whitespace, non-ASCII kept raw) minus generated_from and coverage. The
    README cross-checks this exact `json.dumps` form against Go."""
    body = {k: v for k, v in doc.items() if k not in ("generated_from", "coverage")}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def manifest_counts(doc: dict) -> dict:
    """The counts the registry answers with. `fields` = container fields plus
    list item fields (that sum is the README's 2,094 on the snapshot);
    `lists` = every container's lists (69)."""
    fields = lists = 0
    containers = []
    for grp in ("sections", "globalPanels", "pages", "details"):
        containers += list((doc.get(grp) or {}).values())
    containers.append(doc.get("branding") or {})
    for c in containers:
        if not isinstance(c, dict):
            continue
        fields += len(c.get("fields") or {})
        for lst in (c.get("lists") or {}).values():
            lists += 1
            fields += len((lst or {}).get("item") or {})
    return {"fields": fields, "lists": lists, "sections": len(doc.get("sections") or {}),
            "globalPanels": len(doc.get("globalPanels") or {}), "pages": len(doc.get("pages") or {}),
            "details": len(doc.get("details") or {})}


class Rotation(list):
    """Several scripts for one case, served round-robin per POST /v1/turns
    (sample 1 gets the first, sample 2 the second, …, wrapping around)."""

    def __init__(self, scripts):
        super().__init__(scripts)
        self.served = 0

    def next_script(self) -> list:
        if not self:
            return default_script()
        script = self[self.served % len(self)]
        self.served += 1
        return script


def _script_from_json(streams) -> list:
    return [s if isinstance(s, str) else [(e["event"], e["data"]) for e in s] for s in streams]


def load_scripts(directory: Path) -> dict[str, list]:
    """Scripts from `<case id>.json` files: each file is a list of streams,
    a stream a list of {"event", "data"} objects (or the string
    "superseded"); or `{"rotate": [streams, streams, …]}` for a Rotation.
    A fixtures/traces-style file (`{"streams": [...], ...}`) is accepted too.
    The key is the file stem unless the document carries `"id"` (case ids
    hold `/`, which a file name cannot) or a fixture's `case.id`."""
    out: dict[str, list] = {}
    for p in sorted(Path(directory).glob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        key = p.stem
        if isinstance(doc, dict):
            key = str(doc.get("id") or (doc.get("case") or {}).get("id") or p.stem)
        if isinstance(doc, dict) and "rotate" in doc:
            out[key] = Rotation([_script_from_json(s) for s in doc["rotate"]])
        elif isinstance(doc, dict) and "streams" in doc:
            out[key] = _script_from_json(doc["streams"])
        else:
            out[key] = _script_from_json(doc)
    return out


def default_script(text: str = "Done.") -> list:
    """A one-stream turn: a reply and turn.done, for case ids without a script."""
    return [[("text.delta", {"text": text}),
             ("turn.done", {"usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 0,
                                      "cache_creation_input_tokens": 0, "calls": 1}, "stop": "end_turn"})]]


class _Turn:
    def __init__(self, turn_id: str, site_id: str, store_id: str, script: list):
        self.id = turn_id
        self.site_id = site_id
        self.store_id = store_id
        self.script = script
        self.next_stream = 0
        self.seq = 0
        self.status = "running"
        self.pending: list[str] = []
        self.suspended_id: str | None = None


class FakeService:
    """The server. `tool_results`, `turn_requests`, `feedback` and
    `manifest_posts` are the sinks tests assert on."""

    def __init__(self, manifest: dict | None = None, key: str = "test-key", heartbeats: int = 0,
                 max_body_bytes: int = MAX_BODY_BYTES, manifest_body_limit: int = MANIFEST_BODY_LIMIT,
                 require_manifest: bool = True, port: int = 0):
        self.key = key
        self.port = port  # 0 = a free port (tests); `make fake-service` passes 8089
        self.heartbeats = heartbeats  # SSE mode: this many heartbeat frames before each stream
        self.max_body_bytes = max_body_bytes
        self.manifest_body_limit = manifest_body_limit
        self.require_manifest = require_manifest
        self.manifests: dict[str, dict] = {}  # hash → {body, counts, first_seen, generated_from}
        self.scripts: dict[str, list] = {}
        self.turns: dict[str, _Turn] = {}
        self.tool_results: list[dict] = []
        self.turn_requests: list[dict] = []
        self.feedback: list[dict] = []
        self.conversation_ends: list[str] = []  # DELETE /v1/conversations/{siteId}, in order
        self.manifest_posts: int = 0
        self.unauthorized: int = 0
        self.lock = threading.Lock()
        if manifest is not None:
            self._register(manifest)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # --- lifecycle -------------------------------------------------------------------------------------------------

    @property
    def url(self) -> str:
        assert self._server is not None, "FakeService is not started"
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "FakeService":
        service = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):  # quiet
                pass

            def do_GET(self):
                service._handle(self, "GET")

            def do_POST(self):
                service._handle(self, "POST")

            def do_PUT(self):
                service._handle(self, "PUT")

            def do_DELETE(self):
                service._handle(self, "DELETE")

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> "FakeService":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # --- scripting helpers -------------------------------------------------------------------------------------------

    def script_for(self, site_id: str, text: str) -> list:
        """The script for a turn: by case id (siteId "lab-<case id>"), by
        siteId, by text, else the default."""
        case_id = site_id[4:] if site_id.startswith("lab-") else site_id
        for key in (case_id, site_id, text):
            if key in self.scripts:
                script = self.scripts[key]
                return script.next_script() if isinstance(script, Rotation) else script
        return default_script()

    def _register(self, doc: dict) -> tuple[str, bool]:
        h = manifest_hash(doc)
        with self.lock:
            created = h not in self.manifests
            if created:
                self.manifests[h] = {"body": doc, "counts": manifest_counts(doc),
                                     "first_seen": datetime.now(timezone.utc).isoformat(),
                                     "generated_from": doc.get("generated_from")}
        return h, created

    # --- HTTP ------------------------------------------------------------------------------------------------------------

    def _handle(self, h: BaseHTTPRequestHandler, method: str) -> None:
        try:
            self._dispatch(h, method)
        except _Reply as r:
            self._json(h, r.status, r.body)
        except Exception as e:  # pragma: no cover — surfaces test bugs as 500s
            self._json(h, 500, {"error": "internal", "message": repr(e)})

    def _dispatch(self, h: BaseHTTPRequestHandler, method: str) -> None:
        u = urlparse(h.path)
        parts = [p for p in u.path.split("/") if p]
        if parts in (["healthz"], ["readyz"]):
            return self._json(h, 200, {"ok": True})
        if parts == ["version"]:
            return self._json(h, 200, {"build_id": "fake", "go": "none", "routes": ["default", "fake"]})
        auth = h.headers.get("Authorization", "")
        if auth != f"Bearer {self.key}":
            self.unauthorized += 1
            raise _Reply(401, {"error": "unauthorized", "message": "bad or missing bearer key"})
        if len(parts) >= 2 and parts[0] == "v1" and parts[1] == "manifests":
            return self._manifests(h, method, parts, u)
        if method == "POST" and parts == ["v1", "turns"]:
            return self._start_turn(h)
        if method == "POST" and len(parts) == 4 and parts[:2] == ["v1", "turns"] and parts[3] == "tool-results":
            return self._tool_results(h, parts[2])
        if method == "DELETE" and len(parts) == 3 and parts[:2] == ["v1", "conversations"]:
            # README §1.1: supersede the site's live turns, end its open conversation; 204 either way
            site_id = unquote(parts[2])
            self.conversation_ends.append(site_id)
            with self.lock:
                for t in self.turns.values():
                    if t.site_id == site_id and t.status in ("waiting_tools", "suspended", "running"):
                        t.status = SUPERSEDED
            h.send_response(204)
            h.send_header("Content-Length", "0")
            h.end_headers()
            return None
        if method == "POST" and parts == ["v1", "feedback"]:
            body = self._read_json(h, self.max_body_bytes)
            self.feedback.append(body)
            h.send_response(204)
            h.send_header("Content-Length", "0")
            h.end_headers()
            return None
        raise _Reply(404, {"error": "not_found", "message": h.path})

    def _manifests(self, h, method: str, parts: list[str], u) -> None:
        if method == "POST" and len(parts) == 2:
            doc = self._read_json(h, self.manifest_body_limit)
            if not isinstance(doc, dict):
                raise _Reply(400, {"error": "bad_request", "message": "manifest must be an object"})
            self.manifest_posts += 1
            hsh, created = self._register(doc)
            return self._json(h, 201 if created else 200, self._registered(hsh))
        if method == "PUT" and len(parts) == 3:
            doc = self._read_json(h, self.manifest_body_limit)
            if manifest_hash(doc) != parts[2]:
                raise _Reply(400, {"error": "hash_mismatch", "message": "body hashes differently", "hash": manifest_hash(doc)})
            self.manifest_posts += 1
            hsh, created = self._register(doc)
            return self._json(h, 201 if created else 200, self._registered(hsh))
        if method == "GET" and len(parts) == 3:
            m = self.manifests.get(parts[2])
            if m is None:
                raise _Reply(404, {"error": "not_found", "message": "unknown manifest"})
            out = {"hash": parts[2], "generated_from": m["generated_from"], "counts": m["counts"],
                   "first_seen": m["first_seen"], "index": "fake index"}
            if parse_qs(u.query).get("body", ["0"])[0] == "1":
                out["body"] = m["body"]
            return self._json(h, 200, out)
        raise _Reply(404, {"error": "not_found", "message": h.path})

    def _registered(self, hsh: str) -> dict:
        m = self.manifests[hsh]
        return {"hash": hsh, **m["counts"], "generated_from": m["generated_from"], "diff": None}

    def _start_turn(self, h) -> None:
        body = self._read_json(h, self.max_body_bytes)
        for k in TURN_REQUIRED:
            if not body.get(k):
                raise _Reply(400, {"error": "bad_request", "message": f"{k} is required"})
        if self.require_manifest and body["manifestHash"] not in self.manifests:
            raise _Reply(409, {"error": "unknown_manifest", "message": "register the manifest first",
                               "hash": body["manifestHash"]})
        self.turn_requests.append(body)
        script = self.script_for(str(body["siteId"]), str(body.get("text", "")))
        turn = _Turn(uuid.uuid4().hex, str(body["siteId"]), str(body["storeId"]), script)
        with self.lock:
            self.turns[turn.id] = turn
        self._stream(h, turn)

    def _tool_results(self, h, turn_id: str) -> None:
        body = self._read_json(h, self.max_body_bytes)
        turn = self.turns.get(turn_id)
        if turn is None:
            raise _Reply(404, {"error": "turn_not_found", "message": "turn not found or expired", "turnId": turn_id})
        if body.get("siteId") not in (None, turn.site_id) or body.get("storeId") not in (None, turn.store_id):
            raise _Reply(404, {"error": "turn_not_found", "message": "turn not found or expired", "turnId": turn_id})
        results = body.get("results")
        if not isinstance(results, list) or not results:
            raise _Reply(400, {"error": "bad_request", "message": "results must not be empty"})
        for i, r in enumerate(results):
            if not r.get("toolCallId"):
                raise _Reply(400, {"error": "bad_request", "message": f"results[{i}].toolCallId is required"})
            if not r.get("ok") and not r.get("error"):
                raise _Reply(400, {"error": "bad_request", "message": f"results[{i}]: ok=false requires error"})
        # The next script entry decides: "superseded" → 409 before anything is recorded as applied.
        nxt = turn.script[turn.next_stream] if turn.next_stream < len(turn.script) else None
        if nxt == SUPERSEDED:
            turn.status = SUPERSEDED
            turn.next_stream += 1
            self.tool_results.append({"turnId": turn_id, "body": body, "answer": 409})
            raise _Reply(409, {"error": "turn_not_waiting", "message": "turn is not waiting for tool results",
                               "turnId": turn_id, "status": SUPERSEDED})
        if turn.status not in ("waiting_tools", "suspended"):
            raise _Reply(409, {"error": "turn_not_waiting", "message": "turn is not waiting for tool results",
                               "turnId": turn_id, "status": turn.status})
        answered = {r["toolCallId"] for r in results}
        missing = [i for i in turn.pending if i not in answered]
        if turn.suspended_id is not None:
            picked = next((r for r in results if r["toolCallId"] == turn.suspended_id), None)
            if picked is None or not isinstance(picked.get("asset"), dict) or not picked["asset"].get("public_url"):
                raise _Reply(400, {"error": "bad_request", "message": "the suspended pick_asset needs an asset"})
        self.tool_results.append({"turnId": turn_id, "body": body, "answer": 200, "missing": missing})
        turn.seq += len(results)  # the real service's trace numbers one tool.result per posted item here (trace.go)
        self._stream(h, turn)

    def _stream(self, h, turn: _Turn) -> None:
        if turn.next_stream >= len(turn.script):
            events: list = [("turn.error", {"code": "script_exhausted", "message": "the fake script has no more streams"})]
        else:
            events = turn.script[turn.next_stream]
            if events == SUPERSEDED:  # a script cannot start with it; treat as an error stream
                events = [("turn.error", {"code": "turn_expired", "message": "superseded"})]
        turn.next_stream += 1
        framed = []
        turn.pending, turn.suspended_id = [], None
        turn.status = "running"
        for name, data in events:
            turn.seq += 1
            payload = dict(data)
            payload["turnId"] = turn.id
            payload["seq"] = turn.seq
            framed.append({"event": name, "data": payload})
            if name == "tool.call":
                turn.pending.append(str(payload.get("id")))
            elif name == "turn.waiting":
                turn.status = "waiting_tools"
            elif name == "turn.suspended":
                turn.status = "suspended"
                turn.suspended_id = str(payload.get("toolCallId"))
                if turn.suspended_id not in turn.pending:
                    turn.pending.append(turn.suspended_id)
            elif name in ("turn.done", "turn.error"):
                turn.status = "done" if name == "turn.done" else "error"
        if "application/json" in (h.headers.get("Accept") or "") and "text/event-stream" not in (h.headers.get("Accept") or ""):
            return self._json(h, 200, {"turnId": turn.id, "events": framed})
        out = []
        for _ in range(self.heartbeats):
            out.append("event: heartbeat\ndata: {}\n\n")
        for f in framed:
            out.append(f"event: {f['event']}\ndata: {json.dumps(f['data'], ensure_ascii=False)}\n\n")
        raw = "".join(out).encode("utf-8")
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("Content-Length", str(len(raw)))
        h.end_headers()
        h.wfile.write(raw)

    # --- plumbing ----------------------------------------------------------------------------------------------------

    def _read_json(self, h, limit: int):
        n = int(h.headers.get("Content-Length") or 0)
        raw = h.rfile.read(n) if n else b""
        if n > limit:
            raise _Reply(413, {"error": "bad_request", "message": f"body exceeds {limit} bytes"})
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise _Reply(400, {"error": "bad_request", "message": "invalid JSON"})

    def _json(self, h, status: int, body) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(raw)))
        h.end_headers()
        h.wfile.write(raw)


class _Reply(Exception):
    def __init__(self, status: int, body: dict):
        super().__init__(status)
        self.status = status
        self.body = body


def main(argv: list[str] | None = None) -> int:
    """`python tests/fake_service.py --manifest m.json [--port 0] [--key k]`:
    run the fake on its own for a manual collector run (make lab-fake-service)."""
    import argparse
    import sys
    import time

    p = argparse.ArgumentParser(description="fake Sous Chef service for the lab")
    p.add_argument("--manifest", type=Path, help="manifest JSON to pre-register")
    p.add_argument("--key", default="test-key")
    p.add_argument("--port", type=int, default=0, help="listen port on 127.0.0.1 (default: a free one, printed)")
    p.add_argument("--scripts", type=Path, help="directory of <case id>.json scripts")
    args = p.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    svc = FakeService(manifest=manifest, key=args.key, port=args.port)
    if args.scripts:
        svc.scripts.update(load_scripts(args.scripts))
    svc.start()
    print(svc.url, flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        svc.stop()
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
