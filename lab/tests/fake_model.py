"""A scripted fake model speaking POST /v1/messages, for running the REAL
Sous Chef service binary with no API spend (docs/LAB.md §9, §10).

Pattern copied from the service's Go helper
`service/internal/model/modeltest/fake.go` (never imported — that package is
internal to the Go module): each request consumes the next STEP of the
script; a step is an HTTP status with a body, or a MESSAGE the fake renders
as a JSON reply or — when the request says `"stream": true` — as the Messages
API SSE sequence (message_start → content_block_* → message_delta →
message_stop) with tool_use inputs split over several `input_json_delta`
parts so a client must accumulate them. Every request is captured (headers
+ parsed body). A route for it in the service's MODEL_ROUTES:

    {"lab-fake":{"kind":"messages-compatible","url":"http://127.0.0.1:8091",
                 "auth":"none","model":"fake","max_tokens":4096,"stream":false}}

Steps as JSON (`--steps file.json`, a list):

    {"message": {"content": [{"type":"text","text":"Setting it."},
                             {"type":"tool_use","id":"t1","name":"set_fields","input":{…}}],
                 "stop_reason": "tool_use", "usage": {"input_tokens": 120, "output_tokens": 30}}}
    {"status": 529, "body": {"type":"error","error":{"type":"overloaded_error","message":"busy"}}}

`stop_reason` defaults to `tool_use` when a tool_use block is present, else
`end_turn`. A script that runs out answers 500 `api_error` ("script
exhausted"), as the Go fake does. Stdlib only.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def fill_defaults(msg: dict) -> dict:
    """modeltest.fillDefaults: id, model and stop_reason."""
    out = dict(msg)
    out.setdefault("id", "msg_fake")
    out.setdefault("model", "fake-model")
    content = out.get("content") or []
    if not out.get("stop_reason"):
        out["stop_reason"] = "tool_use" if any(b.get("type") == "tool_use" for b in content) else "end_turn"
    out["content"] = [block_json(b) for b in content]
    return out


def block_json(b: dict) -> dict:
    t = b.get("type", "text")
    if t == "tool_use":
        inp = b.get("input")
        return {"type": "tool_use", "id": b.get("id", ""), "name": b.get("name", ""), "input": {} if inp is None else inp}
    if t == "thinking":
        return {"type": "thinking", "thinking": b.get("thinking", ""), "signature": "sig"}
    return {"type": "text", "text": b.get("text", "")}


def usage_json(u: dict | None) -> dict:
    u = u or {}
    return {"input_tokens": int(u.get("input_tokens", 0)), "output_tokens": int(u.get("output_tokens", 0)),
            "cache_read_input_tokens": int(u.get("cache_read_input_tokens", 0)),
            "cache_creation_input_tokens": int(u.get("cache_creation_input_tokens", 0))}


def message_json(msg: dict) -> dict:
    """modeltest.messageJSON: the non-streaming reply body."""
    m = fill_defaults(msg)
    stop = m["stop_reason"]
    details = None
    if stop == "refusal":
        details = {"type": "refusal", "category": m.get("stop_category", ""), "explanation": "scripted refusal"}
    return {"id": m["id"], "type": "message", "role": "assistant", "model": m["model"], "content": m["content"],
            "stop_reason": stop, "stop_sequence": None, "stop_details": details, "usage": usage_json(m.get("usage"))}


def split_parts(s: str, n: int) -> list[str]:
    """The input JSON in `n` roughly equal parts (at least one)."""
    if not s:
        return [""]
    n = max(1, min(n, len(s)))
    size = -(-len(s) // n)
    return [s[i:i + size] for i in range(0, len(s), size)]


def sse_events(msg: dict) -> list[tuple[str, dict]]:
    """modeltest.writeSSE as a list of (event, data)."""
    m = fill_defaults(msg)
    start = message_json(m)
    usage = m.get("usage") or {}
    start.update(content=[], stop_reason=None, stop_details=None,
                 usage=usage_json({**usage, "output_tokens": 1}))
    out: list[tuple[str, dict]] = [("message_start", {"type": "message_start", "message": start})]
    for i, b in enumerate(m["content"]):
        if b["type"] == "tool_use":
            out.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}}))
            for part in split_parts(json.dumps(b["input"], separators=(",", ":")), 3):
                out.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                    "delta": {"type": "input_json_delta", "partial_json": part}}))
        elif b["type"] == "thinking":
            out.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                "content_block": {"type": "thinking", "thinking": "", "signature": ""}}))
            out.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                "delta": {"type": "thinking_delta", "thinking": b["thinking"]}}))
            out.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                "delta": {"type": "signature_delta", "signature": "sig"}}))
        else:
            out.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                "content_block": {"type": "text", "text": ""}}))
            for part in split_parts(b["text"], 2):
                out.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                    "delta": {"type": "text_delta", "text": part}}))
        out.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    stop, details = message_json(m)["stop_reason"], message_json(m)["stop_details"]
    out.append(("message_delta", {"type": "message_delta",
                                  "delta": {"stop_reason": stop, "stop_sequence": None, "stop_details": details},
                                  "usage": {"output_tokens": int(usage.get("output_tokens", 0))}}))
    out.append(("message_stop", {"type": "message_stop"}))
    return out


def load_steps(path: str | Path) -> list[dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, list):
        raise ValueError(f"{path}: steps must be a JSON list")
    return doc


class FakeModel:
    """The server. `requests` holds every captured request
    (`{path, headers, body, stream, raw_len}`)."""

    def __init__(self, steps: list[dict] | None = None, *, port: int = 0, capture: str | Path | None = None):
        self.steps: list[dict] = list(steps or [])
        self.next = 0
        self.requests: list[dict] = []
        self.port = port
        self.capture = Path(capture) if capture else None
        self.lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # --- lifecycle ---------------------------------------------------------------------------------------------------

    @property
    def url(self) -> str:
        assert self._server is not None, "not started"
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def append(self, *steps: dict) -> None:
        with self.lock:
            self.steps.extend(steps)

    def start(self) -> "FakeModel":
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # silent
                pass

            def do_POST(self):
                fake._handle(self)

            def do_GET(self):
                body = json.dumps({"ok": True, "steps": len(fake.steps), "served": fake.next}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

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

    def __enter__(self) -> "FakeModel":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # --- HTTP --------------------------------------------------------------------------------------------------------

    def _handle(self, h: BaseHTTPRequestHandler) -> None:
        n = int(h.headers.get("Content-Length") or 0)
        raw = h.rfile.read(n) if n else b""
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            body = {"_unparsable": raw[:200].decode("utf-8", "replace")}
        stream = bool(isinstance(body, dict) and body.get("stream"))
        rec = {"path": h.path, "headers": {k: v for k, v in h.headers.items()}, "body": body, "stream": stream, "raw_len": len(raw)}
        with self.lock:
            self.requests.append(rec)
            step = self.steps[self.next] if self.next < len(self.steps) else None
            if step is not None:
                self.next += 1
        if self.capture:
            with self.capture.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if step is None:
            return self._json(h, 500, {"type": "error", "error": {"type": "api_error", "message": "fake_model: script exhausted"}})
        if step.get("message") is None:
            status = int(step.get("status") or 200)
            body_out = step.get("body")
            return self._json(h, status, body_out if body_out is not None else {}, step.get("headers") or {})
        status = int(step.get("status") or 200)
        if stream:
            return self._sse(h, status, sse_events(step["message"]))
        return self._json(h, status, message_json(step["message"]), step.get("headers") or {})

    @staticmethod
    def _json(h: BaseHTTPRequestHandler, status: int, body, headers: dict | None = None) -> None:
        data = (body if isinstance(body, (bytes, bytearray)) else json.dumps(body, ensure_ascii=False).encode("utf-8"))
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            h.send_header(k, v)
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)

    @staticmethod
    def _sse(h: BaseHTTPRequestHandler, status: int, events: list[tuple[str, dict]]) -> None:
        frames = b"".join(f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8") for name, data in events)
        h.send_response(status)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("Content-Length", str(len(frames)))
        h.end_headers()
        h.wfile.write(frames)
        h.wfile.flush()


def main(argv: list[str] | None = None) -> int:
    """`python -m tests.fake_model --steps steps.json [--port 8091] [--capture req.jsonl]`."""
    import argparse
    import sys
    import time

    p = argparse.ArgumentParser(description="fake /v1/messages model for the lab")
    p.add_argument("--steps", type=Path, help="JSON list of steps (see module docstring)")
    p.add_argument("--port", type=int, default=0, help="listen port on 127.0.0.1 (default: a free one, printed)")
    p.add_argument("--capture", type=Path, help="append every captured request here (jsonl)")
    args = p.parse_args(argv)
    steps = load_steps(args.steps) if args.steps else []
    fake = FakeModel(steps, port=args.port, capture=args.capture).start()
    print(fake.url, flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        fake.stop()
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
