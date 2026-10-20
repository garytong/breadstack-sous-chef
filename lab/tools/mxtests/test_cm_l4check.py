"""tools/cm_l4check.py — chocolate-muffin's L4 check: the parser-error classifier (the same list as the service's
model.IsParserMessage), the tool-call validity check, percentiles, ID tokens (gcloud and the metadata server, cached,
never written), the recording proxy, the serving asserts against a fake Ollama, the verdict, the row pick, the import
guard, and — when Go and Postgres are here — a whole measure through the real harness and a sous-chef binary built
from THIS tree: a MinistralParser 500 on the chocolate-muffin route is drawn once and logged, on blueberry-muffin it
is redrawn. Offline: no GCP, no GPU, no model; every server is a local fake."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
LAB = TOOLS.parent
REPO = LAB.parent
for p in (str(LAB), str(TOOLS)):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import cm_l4check as L  # noqa: E402
from souschef_lab import serving_mistral as SM  # noqa: E402

TOKEN_EXP = 4102444800  # 2100-01-01


def jwt(exp: int | None = TOKEN_EXP, sub: str = "someone") -> str:
    def b64(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")
    claims = {"sub": sub, **({"exp": exp} if exp else {})}
    return f"{b64({'alg': 'RS256'})}.{b64(claims)}.c2ln"


TOKEN = jwt()


# --- the classifier, in step with the service ---------------------------------------------------------------------------

def test_parser_messages():
    for msg, want in {
        "tool 'set_fieldz' not found": True, "tool '' not found": True,
        "invalid character '}' looking for beginning of object key string": True, "unexpected end of JSON input": True,
        "Value looks like object, but can't find closing '}' symbol": True, "XML syntax error on line 1: unexpected EOF": True,
        "json: cannot unmarshal number 1e999 into Go value of type float64": True,
        "json: cannot unmarshal string into Go value of type float64": False,
        "llama runner process has terminated: exit status 2": False, "EOF": False, "": False, "tool not found": False,
    }.items():
        assert L.is_parser_message(msg) is want, msg


@pytest.mark.skipif(not (REPO / "service" / "internal" / "model" / "parser.go").is_file(), reason="no service/ here")
def test_the_list_is_the_services():
    """The Go list (service/internal/model/parser.go) and this one name the same messages."""
    go = (REPO / "service" / "internal" / "model" / "parser.go").read_text(encoding="utf-8")
    block = go[go.index("parserMessagePrefixes = []string{"):go.index("// IsParserMessage")]
    prefixes_go = block[:block.index("parserMessages = map")]
    messages_go = block[block.index("parserMessages = map"):]
    strings = lambda s: {json.loads(m) for m in re.findall(r'"(?:[^"\\]|\\.)*"', s)}  # noqa: E731
    assert strings(prefixes_go) == set(L.PARSER_PREFIXES)
    assert strings(messages_go) == set(L.PARSER_MESSAGES)
    assert "\"tool '\"" in go and "\"' not found\"" in go
    overflow = tuple(json.loads(re.search(rf'{name}\s*=\s*("(?:[^"\\]|\\.)*")', go).group(1))
                     for name in ("numberOverflowPrefix", "numberOverflowSuffix"))
    assert overflow == L.NUMBER_OVERFLOW


def test_error_message_shapes():
    assert L.error_message(b'{"type":"error","error":{"type":"api_error","message":"tool \'x\' not found"}}') == "tool 'x' not found"
    assert L.error_message(b'{"error":"tool \'y\' not found"}') == "tool 'y' not found"
    assert L.error_message(b"<html>bad gateway</html>") == "<html>bad gateway</html>"


# --- numbers -------------------------------------------------------------------------------------------------------------

def test_percentiles_are_nearest_rank():
    assert L.pct([], 0.5) is None
    assert L.pct([5], 0.9) == 5.0
    vals = list(range(1, 11))
    assert L.pct(vals, 0.5) == 5.0 and L.pct(vals, 0.9) == 9.0 and L.pct(vals, 1.0) == 10.0
    assert L.dist([1, 2, None, 3]) == {"n": 3, "p50": 2.0, "p90": 3.0, "mean": 2.0}


# --- tool-call validity --------------------------------------------------------------------------------------------------

TOOLS_JSON = {t["name"]: t["input_schema"] for t in json.loads((LAB / "fixtures" / "prompt" / "tools.json").read_text())}


def test_tool_use_validity_on_the_real_tools():
    ok = {"name": "report_unsupported", "input": {"reason": "Prices come from your product catalogue.", "where": "products"}}
    v = L.tool_use_check(ok, TOOLS_JSON)
    assert v["valid"], v
    assert L.tool_use_check({"name": "set_fieldz", "input": {}}, TOOLS_JSON) == {"name": "set_fieldz", "valid": False, "problem": "unknown_tool"}
    assert L.tool_use_check({"name": "set_fields", "input": "x"}, TOOLS_JSON)["problem"] == "input_not_object"
    missing = L.tool_use_check({"name": "set_fields", "input": {}}, TOOLS_JSON)
    assert not missing["valid"] and "required" in missing["problem"]
    extra = L.tool_use_check({"name": "report_unsupported", "input": {**ok["input"], "colour": "red"}}, TOOLS_JSON)
    assert not extra["valid"] and "not allowed" in extra["problem"]
    enum = L.tool_use_check({"name": "report_unsupported", "input": {**ok["input"], "where": "the moon"}}, TOOLS_JSON)
    assert not enum["valid"] and "enum" in enum["problem"]


def test_schema_subset():
    s = {"type": "object", "required": ["a"], "additionalProperties": False,
         "properties": {"a": {"type": "array", "minItems": 1, "items": {"type": "integer"}},
                        "b": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}
    assert L.schema_problems({"a": [1, 2]}, s) == []
    assert L.schema_problems({"a": [1, True]}, s) == ["$.a[1]: want integer, got bool"]
    assert L.schema_problems({"a": []}, s) == ["$.a: fewer than 1 items"]
    assert L.schema_problems({"a": [1], "b": 3}, s) == ["$.b: matches no anyOf branch"]
    assert L.schema_problems({"a": [1], "c": 1}, s) == ["$.c: not allowed"]


# --- ID tokens -----------------------------------------------------------------------------------------------------------

@pytest.fixture
def fake_gcloud(tmp_path):
    """A gcloud that prints a token and logs its arguments (never the token)."""
    log = tmp_path / "gcloud.args"
    script = tmp_path / "bin" / "gcloud"
    script.parent.mkdir()
    script.write_text(f"#!/bin/sh\necho \"$@\" >> {log}\necho '{TOKEN}'\n", encoding="utf-8")
    script.chmod(0o755)
    return script, log


def test_gcloud_tokens_are_cached_until_five_minutes_before_exp(fake_gcloud):
    script, log = fake_gcloud
    now = [TOKEN_EXP - 3600.0]
    t = L.IDTokens("gcloud", "https://m.run.app", gcloud=str(script), now=lambda: now[0])
    assert t.get() == TOKEN and t.get() == TOKEN and t.minted == 1
    now[0] = TOKEN_EXP - 299
    assert t.get() == TOKEN and t.minted == 2
    assert log.read_text().splitlines() == ["auth print-identity-token"] * 2
    assert t.headers() == {"X-Serverless-Authorization": f"Bearer {TOKEN}"}


def test_gcloud_impersonation_names_the_audience(fake_gcloud):
    script, log = fake_gcloud
    L.IDTokens("gcloud", "https://m.run.app", impersonate="souschef-drift@p.iam.gserviceaccount.com", gcloud=str(script)).get()
    assert log.read_text().split() == ["auth", "print-identity-token", "--impersonate-service-account=souschef-drift@p.iam.gserviceaccount.com",
                                       "--audiences=https://m.run.app", "--include-email"]


def test_a_failing_gcloud_is_an_l4_error(tmp_path):
    bad = tmp_path / "gcloud"
    bad.write_text("#!/bin/sh\necho 'ERROR: not logged in' >&2\nexit 1\n")
    bad.chmod(0o755)
    with pytest.raises(L.L4Error, match="exited 1: ERROR: not logged in"):
        L.IDTokens("gcloud", "https://m", gcloud=str(bad)).get()
    assert L.IDTokens("none", "https://m").headers() == {}


class _Metadata(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        return

    def do_GET(self):
        _Metadata.seen.append((self.path, self.headers.get("Metadata-Flavor")))
        body = jwt(exp=None).encode()
        self.send_response(200 if self.headers.get("Metadata-Flavor") == "Google" else 403)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_metadata_tokens_ask_for_the_url_as_audience():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Metadata)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        now = [1000.0]
        t = L.IDTokens("metadata", "https://sous-chef-model-cand.example.run.app",
                       metadata_url=f"http://127.0.0.1:{srv.server_address[1]}", now=lambda: now[0])
        assert t.get() == jwt(exp=None)
        path, flavor = _Metadata.seen[-1]
        assert flavor == "Google" and path.endswith("identity?audience=https%3A%2F%2Fsous-chef-model-cand.example.run.app")
        now[0] += 1700  # no exp in the token: 30 minutes
        t.get()
        assert t.minted == 1
        now[0] += 200
        t.get()
        assert t.minted == 2
    finally:
        srv.shutdown()
        srv.server_close()


# --- a fake Ollama behind Cloud Run's IAM ----------------------------------------------------------------------------------

def tool_use(name: str, inp: dict, i: int = 1) -> dict:
    return {"type": "tool_use", "id": f"call_{i}", "name": name, "input": inp}


class FakeOllama:
    """Ollama 0.34.4's API as the model image serves it, behind a token check (X-Serverless-Authorization).
    `family` picks /api/show's answer; `script(n, body)` answers the n-th service call (1-based; smokes and primes
    apart) with (status, body dict) — default a text reply. Every /v1/messages body is kept."""

    def __init__(self, family: str = "mistral", *, token: str | None = TOKEN, template: str | None = None,
                 vram: bool = True, script=None, echo: str = "tool_use"):
        self.family, self.token, self.vram, self.script, self.echo = family, token, vram, script, echo
        self.smoke_cached = 0
        self.template = SM.TEMPLATE if template is None and family == "mistral" else template
        self.messages: list[dict] = []
        self.denied = 0
        self.service_calls = 0
        self.lock = threading.Lock()
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                return

            def _send(self, status: int, doc) -> None:
                raw = json.dumps(doc).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _auth(self) -> bool:
                if fake.token and self.headers.get("X-Serverless-Authorization") != f"Bearer {fake.token}":
                    fake.denied += 1
                    self._send(403, {"error": "forbidden"})
                    return False
                return True

            def do_GET(self):
                if not self._auth():
                    return
                if self.path == "/api/version":
                    return self._send(200, {"version": "0.34.4"})
                if self.path == "/api/ps":
                    size = 5_000_000_000
                    return self._send(200, {"models": [{"name": "sous-chef:latest", "size": size,
                                                        "size_vram": size if fake.vram else size // 2, "context_length": 65536}]})
                self._send(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if not self._auth():
                    return
                if self.path == "/api/show":
                    return self._send(200, fake.show())
                if self.path == "/api/generate":
                    return self._send(200, {"done": True})
                if self.path.startswith("/v1/messages"):
                    status, doc = fake.answer(body)
                    return self._send(status, doc)
                self._send(404, {"error": "not found"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def show(self) -> dict:
        if self.family == "mistral":
            mf = "FROM /models/blobs/sha256-x\nPARSER ministral\nPARAMETER temperature 0.3\n"
            return {"modelfile": mf, "template": self.template, "capabilities": ["completion", "tools"],
                    "model_info": {"general.architecture": "mistral3"}, "parameters": "num_ctx 65536\ntemperature 0.3"}
        mf = "FROM /models/blobs/sha256-y\nRENDERER qwen3.5\nPARSER qwen3.5\n"
        return {"modelfile": mf, "template": "{{ .Prompt }}", "capabilities": ["completion", "tools", "thinking"],
                "model_info": {"general.architecture": "qwen35"}, "parameters": "num_ctx 65536\ntemperature 0.3\ntop_k 20"}

    def answer(self, body: dict) -> tuple[int, dict]:
        with self.lock:
            self.messages.append(body)
        msg = lambda content, stop="end_turn", out=7: (200, {  # noqa: E731
            "id": "msg_f", "type": "message", "role": "assistant", "model": "sous-chef", "content": content,
            "stop_reason": stop, "usage": {"input_tokens": 1500, "output_tokens": out, "cache_read_input_tokens": 14000}})
        names = [t.get("name") for t in body.get("tools") or []]
        if body.get("max_tokens") == 1 and body.get("messages") == SM.SMOKE_REQUEST["messages"]:
            n, cached = SM.SMOKE_PROMPT_TOKENS[self.family], self.smoke_cached  # a warm slot may hold [BOS][SYSTEM_PROMPT]
            return 200, {"id": "msg_s", "type": "message", "role": "assistant", "model": "sous-chef", "content": [],
                         "stop_reason": "max_tokens", "usage": {"input_tokens": n - cached, "output_tokens": 1,
                                                                **({"cache_read_input_tokens": cached} if cached else {})}}
        if names == ["echo"]:
            if self.echo == "parser500":
                return 500, {"type": "error", "error": {"type": "api_error", "message": "tool 'echoo' not found"}}
            if self.echo == "leak":
                return msg([{"type": "text", "text": "[TOOL_CALLS]echo[ARGS]{\"text\":\"hello\"}"}])
            if self.echo == "text":
                return msg([{"type": "text", "text": "hello"}])
            return msg([tool_use("echo", {"text": "hello"})], "tool_use")
        if body.get("max_tokens") == 1:  # the Primer's
            return msg([{"type": "text", "text": "x"}], "max_tokens", 1)
        with self.lock:
            self.service_calls += 1
            n = self.service_calls
        if self.script:
            got = self.script(n, body)
            if got is not None:
                return got
        return msg([{"type": "text", "text": "Done — that's in your draft."}])


@pytest.fixture
def ollama():
    made = []

    def make(*a, **kw):
        f = FakeOllama(*a, **kw)
        made.append(f)
        return f
    yield make
    for f in made:
        f.close()


def static_tokens() -> L.IDTokens:
    t = L.IDTokens("none", "x")
    t.mode, t._token, t._until = "static", TOKEN, float("inf")
    return t


# --- the serving asserts ---------------------------------------------------------------------------------------------------

def test_served_checks_pass_on_a_correct_mistral_image(ollama):
    f = ollama("mistral")
    s = L.served_checks(f.url, "mistral", static_tokens())
    assert s["ok"], s["problems"]
    assert s["show"]["parser"] == "ministral" and s["show"]["template_sha256"] == L.sha256_text(SM.TEMPLATE)
    assert s["prompt_smoke"]["input_tokens"] == 61 and s["tool_smoke"]["content"] == ["tool_use"] and s["ollama_version"] == "0.34.4"


def test_the_prompt_size_counts_the_cached_prefix_too(ollama):
    """A warm server answers the smoke with part of the prompt read from its cache: the size is the sum."""
    f = ollama("mistral")
    f.smoke_cached = 2
    s = L.served_checks(f.url, "mistral", static_tokens())
    assert s["ok"], s["problems"]
    assert s["prompt_smoke"]["input_tokens"] == 61 and s["prompt_smoke"]["cached"] == 2


def test_served_checks_pass_on_the_live_qwen_image(ollama):
    s = L.served_checks(ollama("qwen").url, "qwen", static_tokens())
    assert s["ok"], s["problems"]
    assert s["show"]["renderer"] == "qwen3.5" and s["prompt_smoke"]["input_tokens"] == 279


@pytest.mark.parametrize("kw, family, want", [
    ({"template": SM.TEMPLATE + "\n"}, "mistral", "TEMPLATE sha256"),
    ({"vram": False}, "mistral", "not 100% GPU"),
    ({"echo": "parser500"}, "mistral", "the echo-tool smoke answered 500: tool 'echoo' not found"),
    ({"echo": "leak"}, "mistral", "call markers"),
    ({}, "qwen", "PARSER 'ministral'"),  # a Mistral image measured as the live model
])
def test_served_checks_problems(ollama, kw, family, want):
    s = L.served_checks(ollama("mistral", **kw).url, family, static_tokens())
    assert not s["ok"] and any(want in p for p in s["problems"]), s["problems"]


def test_a_qwen_text_reply_to_the_echo_smoke_is_recorded_not_a_problem(ollama):
    f = ollama("qwen")
    f.echo = "text"
    s = L.served_checks(f.url, "qwen", static_tokens())
    assert s["ok"] and s["tool_smoke"]["content"] == ["text"], s
    m = ollama("mistral")
    m.echo = "text"
    s = L.served_checks(m.url, "mistral", static_tokens())
    assert not s["ok"] and "made no tool call" in s["problems"][0]


def test_a_refused_token_says_so(ollama):
    s = L.served_checks(ollama("mistral", token="another").url, "mistral", static_tokens())
    assert not s["ok"] and "refused" in s["problems"][0]


# --- the proxy -------------------------------------------------------------------------------------------------------------

def post(url: str, body: dict, path: str = "/v1/messages?beta=true") -> tuple[int, dict]:
    st, doc, _ = L.http_json(url + path, body)
    return st, doc


def test_the_proxy_adds_the_token_and_records_each_call(ollama):
    def script(n, body):
        if n == 2:
            return 500, {"type": "error", "error": {"type": "api_error", "message": "tool 'set_fieldz' not found"}}
        if n == 3:
            return 200, {"id": "m", "type": "message", "role": "assistant", "model": "sous-chef", "stop_reason": "tool_use",
                         "content": [tool_use("lookup_fields", {"target": {"kind": "global", "panel": "BrandingSection"}}),
                                     tool_use("set_fieldz", {}, 2)], "usage": {"input_tokens": 9, "output_tokens": 30}}
        return None
    f = ollama("mistral", script=script)
    tools = json.loads((LAB / "fixtures" / "prompt" / "tools.json").read_text())
    with L.ModelProxy(f.url, static_tokens()) as px:
        px.tag = "measure:r00#1"
        req = {"model": "sous-chef", "max_tokens": 1024, "temperature": 0.15, "stream": False, "tools": tools,
               "thinking": {"type": "disabled"}, "system": [{"type": "text", "text": "RULES"}],
               "messages": [{"role": "user", "content": "make the headline red"}]}
        assert post(px.url, req)[0] == 200
        st, doc = post(px.url, req)
        assert st == 500 and doc["error"]["message"] == "tool 'set_fieldz' not found"  # passed through unchanged
        assert post(px.url, req)[0] == 200
        assert post(px.url, dict(req, max_tokens=1))[0] == 200  # a prime
        px.tag = "measure:r00#2"
        assert px.wait_idle(0.05, timeout_s=5) < 5
    assert f.denied == 0 and len(f.messages) == 4 and f.messages[0] == req  # the body went out as sent
    c = px.calls
    assert [x["kind"] for x in c] == ["call", "call", "call", "prime"] and {x["turn"] for x in c} == {"measure:r00#1"}
    assert c[0]["status"] == 200 and c[0]["request"]["temperature"] == 0.15 and c[0]["request"]["tools"] == len(tools)
    assert c[0]["request"]["system_chars"] == 5 and c[0]["request"]["thinking"] == "disabled"
    assert c[1]["parser_error"] is True and c[1]["error"] == "tool 'set_fieldz' not found"
    assert c[2]["tool_uses"] == [{"name": "lookup_fields", "valid": True, "problem": None},
                                 {"name": "set_fieldz", "valid": False, "problem": "unknown_tool"}]
    assert all(x["ms"] >= 0 for x in c) and TOKEN not in json.dumps(c)


def test_an_unreachable_server_is_a_recorded_502():
    with L.ModelProxy("http://127.0.0.1:9", None) as px:
        st, doc = post(px.url, {"model": "m", "max_tokens": 5, "messages": []})
    assert st == 502 and "cm_l4check proxy" in doc["error"]["message"]
    assert px.calls[0]["status"] == 502 and px.calls[0]["parser_error"] is False


# --- the verdict -----------------------------------------------------------------------------------------------------------

MODEL_REPO = "us-east4-docker.pkg.dev/p/sous-chef-model/model"
CAND_IMAGE = f"{MODEL_REPO}@sha256:" + "c" * 64
R7_IMAGE = f"{MODEL_REPO}@{L.R7_DIGEST}"
LATEST = [{"latestRevision": True, "percent": 100, "revisionName": "svc-00007-abc"}]


def measured(p50: float, *, parser: int = 0, label: str = "candidate", profile: str = "chocolate-muffin", day: str = "2026-10-19",
             rows: str = "abc", ok: int = 40, served: bool = True, image: str = CAND_IMAGE, failed: int = 0,
             other: dict | None = None, traffic: list | None = None, url: str | None = None) -> dict:
    return {"tool": "cm_l4check", "tool_version": L.TOOL_VERSION, "label": label, "profile": profile, "url": url or f"https://{label}",
            "complete": True, "served": {"ok": served, "problems": [] if served else ["PARSER x"]}, "day": day, "rows_sha256": rows,
            "gap_s": 1.5, "call_timeout": "60s", "engine_fingerprint": "e1",
            "service": {"name": label, "described": True, "image": image, "traffic": LATEST if traffic is None else traffic} if image else {},
            "turns": {"p50": p50, "p90": p50 * 2, "n_ok": ok, "n_all": ok + failed,
                      "errors": {"model_unavailable": failed} if failed else {}, "calls_per_turn_mean": 1.9},
            "calls": {"p50": p50 / 2, "p90": p50, "n_all": 80, "output_tokens": {"p50": 32}},
            "tool_calls": {"valid": 70, "n": 70}, "parser_errors": {"n": parser}, "other_errors": dict(other or {}),
            "exam": {"pass": 30, "rows": 40}, "primes": {"n": 0}}


REF = dict(label="reference", profile="blueberry-muffin", image=R7_IMAGE)


def test_pass_fail_and_incomplete():
    v = L.compare(measured(4700), measured(6600, **REF))
    assert v["verdict"] == "PASS" and [c["ok"] for c in v["checks"]] == [True, True, True, True] and not v["notes"]
    assert v["summary"]["candidate"]["image"] == CAND_IMAGE and v["summary"]["reference"]["image"] == R7_IMAGE
    assert L.compare(measured(6700), measured(6600, **REF))["verdict"] == "FAIL"
    assert L.compare(measured(6700), measured(6600, **REF), tolerance=0.02)["verdict"] == "PASS"
    v = L.compare(measured(4000, parser=1), measured(6600, **REF))
    assert v["verdict"] == "FAIL" and v["checks"][1] == {"check": "0 parser errors", "candidate": 1, "ok": False}
    pinned = [{"percent": 100, "revisionName": "svc-00006-old"}]
    for cand, ref, why in [
            (measured(4000, day="2026-10-18"), {}, "different days"), (measured(4000, rows="xyz"), {}, "different rows"),
            (measured(4000, ok=12), {}, "fewer than 30"), (measured(4000, served=False), {}, "serving asserts failed"),
            ({**measured(4000), "engine_fingerprint": "e2"}, {}, "different engines"),
            ({**measured(4000), "complete": False, "failed": "the binary is too old"}, {}, "too old"),
            # what was measured must be what the switch is about (these were notes before the review)
            (measured(4000, profile="blueberry-muffin"), {}, "not chocolate-muffin"),
            (measured(4000), {"profile": "chocolate-muffin"}, "not blueberry-muffin"),
            (measured(4000, url="https://same"), {"url": "https://same"}, "the same URL"),
            (measured(4000), {"image": f"{MODEL_REPO}@sha256:" + "1" * 64}, "not r7"),
            (measured(4000), {"image": ""}, "image is unknown"),
            (measured(4000), {"traffic": pinned}, "reference service's traffic is not 100 %"),
            (measured(4000, traffic=pinned), {}, "candidate service's traffic is not 100 %"),
            ({**measured(4000), "service_after": {"image": f"{MODEL_REPO}@sha256:" + "d" * 64, "traffic": LATEST}}, {},
             "candidate service changed during the measure")]:
        v = L.compare(cand, measured(6600, **{**REF, **ref}))
        assert v["verdict"] == "INCOMPLETE" and any(why in w for w in v["incomplete"]), (why, v["incomplete"])
    # the r7 check can be skipped explicitly (another live model), never by an unknown image
    assert L.compare(measured(4000), measured(6600, **{**REF, "image": ""}), reference_digest=None)["verdict"] == "PASS"
    # an unchanged service after the measure is fine; an unknown candidate image is a note (make inhouse-switch refuses it)
    assert L.compare({**measured(4000), "service_after": measured(4000)["service"]}, measured(6600, **REF))["verdict"] == "PASS"
    v = L.compare(measured(4000, image=""), measured(6600, **REF))
    assert v["verdict"] == "PASS" and any("candidate's image is unknown" in n for n in v["notes"])
    md = L.verdict_md(L.compare(measured(4700), measured(6600, **REF)))
    assert md.startswith("# chocolate-muffin L4 check: **PASS**") and "| candidate (candidate) | 4700" in md


def test_more_failed_turns_or_call_errors_than_r7_fail():
    """The review's case: 31 of 60 turns ok, 29 model_unavailable, 87 non-parser call errors — fast and parser-clean,
    and it was a PASS. Failed turns and non-parser errors may not exceed the reference's."""
    v = L.compare(measured(4000, ok=31, failed=29, other={"502": 60, "503": 27}), measured(6600, **REF, ok=60))
    assert v["verdict"] == "FAIL"
    assert v["checks"][2] == {"check": "no more failed turns than the reference", "candidate": 29, "reference": 0,
                              "candidate_errors": {"model_unavailable": 29}, "ok": False}
    assert v["checks"][3]["candidate"] == 87 and v["checks"][3]["reference"] == 0 and not v["checks"][3]["ok"]
    assert L.compare(measured(4000, failed=2), measured(6600, **REF, failed=1))["verdict"] == "FAIL"
    assert L.compare(measured(4000, failed=1), measured(6600, **REF, failed=1))["verdict"] == "PASS"
    assert L.compare(measured(4000, other={"503": 2}), measured(6600, **REF, other={"503": 1}))["verdict"] == "FAIL"
    assert L.compare(measured(4000, other={"0": 1}), measured(6600, **REF, other={"503": 1}))["verdict"] == "PASS"


def test_compare_cli_exit_codes(tmp_path):
    (tmp_path / "c.json").write_text(json.dumps(measured(4000, parser=2)))
    (tmp_path / "r.json").write_text(json.dumps(measured(6000, **REF)))
    assert L.main(["compare", "--candidate", str(tmp_path / "c.json"), "--reference", str(tmp_path / "r.json"),
                   "--out", str(tmp_path / "v")]) == L.EXIT_FAIL
    assert json.loads((tmp_path / "v" / "verdict.json").read_text())["verdict"] == "FAIL"
    (tmp_path / "c.json").write_text(json.dumps(measured(4000)))
    assert L.main(["compare", "--candidate", str(tmp_path / "c.json"), "--reference", str(tmp_path / "r.json")]) == L.EXIT_PASS
    (tmp_path / "r.json").write_text(json.dumps(measured(6000, **{**REF, "image": ""})))
    assert L.main(["compare", "--candidate", str(tmp_path / "c.json"), "--reference", str(tmp_path / "r.json")]) == L.EXIT_INCOMPLETE
    assert L.main(["compare", "--candidate", str(tmp_path / "c.json"), "--reference", str(tmp_path / "r.json"),
                   "--reference-digest", ""]) == L.EXIT_PASS


# --- check: the reference on the eval service, under the pipeline's locks ---------------------------------------------------

def fake_measures(bucket: Path, seen: list, cand: dict | None = None):
    """L.measure's stand-in: records the pipeline locks it ran under, writes a measure.json per side."""
    from souschef_lab import gcsio

    def fake(a):
        b = gcsio.bucket(str(bucket))
        seen.append({"profile": a.profile, "service": a.service_name, "url": a.url,
                     "locks": {n: (b.read_lock(n) or {}).get("holder") for n in L.CHECK_LOCKS}})
        doc = (cand or measured(4000)) if a.profile == "chocolate-muffin" else measured(6600, **REF)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "measure.json").write_text(json.dumps({**doc, "url": a.url}))
        return 0
    return fake


def test_check_measures_r7_on_the_eval_service_under_the_pipeline_locks(tmp_path, monkeypatch):
    from souschef_lab import gcsio
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    seen: list = []
    monkeypatch.setattr(L, "measure", fake_measures(bucket, seen))
    a = L.build_parser().parse_args(["check", "--out", str(tmp_path / "v")])
    assert (a.ref_service, a.cand_service, a.bucket, a.no_lock) == ("sous-chef-model-eval", "sous-chef-model-cand", L.BUCKET, False)
    assert L.build_parser().parse_args(["check", "--out", "x", "--live-service", "sous-chef-model"]).ref_service == "sous-chef-model"
    rc = L.main(["check", "--ref-url", "https://eval", "--cand-url", "https://cand", "--bucket", str(bucket),
                 "--out", str(tmp_path / "v")])
    assert rc == L.EXIT_PASS, (tmp_path / "v" / "verdict.md").read_text()
    assert [(s["profile"], s["url"]) for s in seen] == [("blueberry-muffin", "https://eval"), ("chocolate-muffin", "https://cand")]
    for s in seen:  # both measures ran holding retrain + gpu-eval
        assert all(str(h).startswith("cm_l4check:l4-") for h in s["locks"].values()), s
    b = gcsio.bucket(str(bucket))
    assert all(b.read_lock(n) is None for n in L.CHECK_LOCKS)  # given back
    v = json.loads((tmp_path / "v" / "verdict.json").read_text())
    assert v["verdict"] == "PASS" and v["summary"]["candidate"]["image"] == CAND_IMAGE


def test_check_stops_while_a_round_or_a_drift_check_holds_a_lock(tmp_path, monkeypatch):
    from souschef_lab import gcsio
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    b = gcsio.bucket(str(bucket))
    b.acquire("gpu-eval", "dr-20261019-120000-check", 3600)
    monkeypatch.setattr(L, "measure", lambda a: pytest.fail("measured while gpu-eval was held"))
    rc = L.main(["check", "--ref-url", "https://eval", "--cand-url", "https://cand", "--bucket", str(bucket),
                 "--out", str(tmp_path / "v")])
    assert rc == L.EXIT_INCOMPLETE and not (tmp_path / "v" / "verdict.json").exists()
    assert b.read_lock("gpu-eval")["run_id"] == "dr-20261019-120000-check"  # the drift check's, untouched
    assert b.read_lock("retrain") is None  # the one the check took is given back
    # --no-lock: nothing is taken (a person who knows nothing else uses the GPU services)
    seen: list = []
    monkeypatch.setattr(L, "measure", fake_measures(bucket, seen))
    assert L.main(["check", "--ref-url", "https://eval", "--cand-url", "https://cand", "--bucket", str(bucket), "--no-lock",
                   "--out", str(tmp_path / "v2")]) == L.EXIT_PASS
    assert all(s["locks"]["retrain"] is None for s in seen)


def test_a_check_whose_candidate_fails_its_turns_is_a_fail(tmp_path, monkeypatch):
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    monkeypatch.setattr(L, "measure", fake_measures(bucket, [], cand=measured(4000, ok=31, failed=29, other={"502": 87})))
    assert L.main(["check", "--ref-url", "https://eval", "--cand-url", "https://cand", "--bucket", str(bucket),
                   "--out", str(tmp_path / "v")]) == L.EXIT_FAIL


# --- the rows and the import guard -------------------------------------------------------------------------------------------

def test_the_pick_is_fixed_and_disjoint():
    rows, warm = L.pick_rows()
    again, _ = L.pick_rows()
    assert [r["id"] for r in rows] == [r["id"] for r in again] and len(rows) == sum(L.DEFAULT_PICK.values()) == 40
    assert len(warm) == 2 and not {r["id"] for r in warm} & {r["id"] for r in rows}  # ids, not set:id: a followup row
    assert len({L.row_key(r) for r in rows}) == len(rows)                           # shares its paraphrase twin's
    assert all(r.get("set") in L.DEFAULT_PICK for r in rows) and not any(r["id"].startswith("p137") for r in rows)
    assert L.rows_fingerprint(rows) == L.rows_fingerprint(again) != L.rows_fingerprint(rows[1:])
    assert L.parse_pick("real=2,holdout=1") == {"real": 2, "holdout": 1}
    with pytest.raises(ValueError):
        L.parse_pick("nope=1")


@pytest.mark.parametrize("pythonpath", ["worktree", "other-checkout"])
def test_a_subprocess_imports_this_trees_lab(tmp_path, pythonpath):
    """The CLI imports its own lab whatever PYTHONPATH names (an earlier run imported another checkout's)."""
    other = tmp_path / "other-lab" / "souschef_lab"
    other.mkdir(parents=True)
    (other / "__init__.py").write_text("raise ImportError('the other checkout was imported')\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(LAB) if pythonpath == "worktree" else str(other.parent)
    env["TMPDIR"] = str(tmp_path)
    r = subprocess.run([sys.executable, str(TOOLS / "cm_l4check.py"), "rows", "--pick", "real=2", "--warmup", "1"],
                       capture_output=True, text=True, env=env, timeout=120, check=False)
    assert r.returncode == 0, r.stderr[-800:]
    assert "# 2 rows" in r.stdout


# --- a whole measure through the real harness ---------------------------------------------------------------------------------

def _pg_ok() -> bool:
    from souschef_lab import harness as H
    try:
        H.pg_bin_dir()
        return True
    except H.HarnessError:
        return False


SERVICE = REPO / "service"
REAL = bool(shutil.which("go")) and SERVICE.is_dir() and _pg_ok() and os.getuid() != 0


@pytest.fixture(scope="module")
def service_bin(tmp_path_factory):
    """The sous-chef binary built from THIS tree (the chocolate-muffin route needs parser_retries)."""
    out = tmp_path_factory.mktemp("bin") / "sous-chef"
    env = {**os.environ, "CGO_ENABLED": "0"}
    r = subprocess.run(["go", "build", "-o", str(out), "./cmd/sous-chef"], cwd=str(SERVICE), capture_output=True, text=True,
                       env=env, timeout=600, check=False)
    if r.returncode != 0:
        pytest.fail(f"go build: {r.stderr[-1500:]}")
    return out


PICK = "paraphrase=2,followup=1"


def measure_args(tmp_path: Path, url: str, profile: str, service_bin: Path, name: str) -> argparse.Namespace:
    return argparse.Namespace(url=url, profile=profile, label=profile, out=str(tmp_path / name), service_name=None, set=None,
                              ignore_served=False, auth="gcloud", impersonate=None, pick=PICK, warmup=1,
                              limit=None, gap_s=0.2, call_timeout="20s", service_bin=str(service_bin), engine=None,
                              fake_engine=True, keep_work=False)


PARSER_500 = (500, {"type": "error", "error": {"type": "api_error", "message": "tool 'set_fieldz' not found"}})


def last_user_text(body: dict) -> str:
    content = (body.get("messages") or [{}])[-1].get("content")
    if isinstance(content, str):
        return content
    return "".join(str(b.get("text") or "") for b in content or [] if isinstance(b, dict))


def parser_error_once_on(text: str):
    """A script: the first service call whose user message carries `text` is a MinistralParser 500."""
    fired: list[int] = []

    def script(n, body):
        if not fired and text in last_user_text(body):
            fired.append(n)
            return PARSER_500
        return None
    return script


@pytest.mark.skipif(not REAL, reason="needs go, the service/ sources and Postgres server binaries (PG_BIN or PATH)")
def test_a_measure_through_the_real_harness(tmp_path, ollama, fake_gcloud, monkeypatch, service_bin):
    script, _ = fake_gcloud
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")
    rows, _ = L.pick_rows(L.parse_pick(PICK), 1)
    ids = [r["id"] for r in rows]
    # its scored turn (after any history turns) draws the parser error; a followup row and its paraphrase twin share
    # their text, so the target is a row whose id is its own
    target = next(r for r in reversed(rows) if ids.count(r["id"]) == 1)
    assert target["request"] not in json.dumps([h.get("user") for r in rows for h in r.get("history") or []])
    # the candidate: the scored turn of the last row draws a MinistralParser 500
    cand = ollama("mistral", script=parser_error_once_on(target["request"]))
    rc = L.measure(measure_args(tmp_path, cand.url, "chocolate-muffin", service_bin, "cand"))
    doc = json.loads((tmp_path / "cand" / "measure.json").read_text())
    assert rc == 0 and doc["complete"], doc.get("failed")
    assert cand.denied == 0 and doc["served"]["ok"] and doc["tokens_minted"] == 1
    # the request shape the service sent: the chocolate-muffin route as int will run it
    assert doc["route_sent"] == {"kind": "messages-compatible", "auth": "none", "model": "sous-chef", "max_tokens": 1024,
                                 "stream": False, "temperature": 0.15, "prime": "off", "fast_finish": "on",
                                 "fields_context": "on", "parser_retries": "off", "report_as": "chocolate-muffin"}
    assert doc["route_info"]["parser_retries"] == "off"
    service_calls = [m for m in cand.messages if len(m.get("tools") or []) > 1]
    assert {m["temperature"] for m in service_calls} == {0.15} and {m["max_tokens"] for m in service_calls} == {1024}
    assert all(m["thinking"] == {"type": "disabled"} for m in service_calls)
    assert cand.service_calls == len(service_calls)  # no prime reached the server
    # the parser 500 was drawn once (no redraw), counted, logged by the service, and its turn ended model_unavailable
    pe = doc["parser_errors"]
    assert pe["n"] == 1 and pe["service_log_lines"] == 1 and pe["messages"] == ["tool 'set_fieldz' not found"]
    calls = [json.loads(ln) for ln in (tmp_path / "cand" / "calls.jsonl").read_text().splitlines()]
    bad = [c for c in calls if c.get("parser_error")]
    assert len(bad) == 1 and sum(1 for c in calls if c["turn"] == bad[0]["turn"]) == 1
    failed = [t for t in doc["per_turn"] if t["status"] != "ok"]
    assert len(failed) == 1 and failed[0]["code"] == "model_unavailable" and failed[0]["parser_errors"] == 1
    assert failed[0]["row"] == L.row_key(target)
    assert doc["turns"]["n_all"] == sum(1 + len(r.get("history") or []) for r in rows)
    assert doc["turns"]["p50"] is not None and doc["calls"]["n_all"] >= doc["turns"]["n_all"]
    log = (tmp_path / "cand" / "service.log").read_text()
    assert "souschef: model parser error" in log and TOKEN not in log and TOKEN not in json.dumps(doc)

    # the reference: blueberry-muffin's route redraws the same 500 (the SDK's retries) and primes the server
    ref = ollama("qwen", script=parser_error_once_on(target["request"]))
    rc = L.measure(measure_args(tmp_path, ref.url, "blueberry-muffin", service_bin, "ref"))
    rdoc = json.loads((tmp_path / "ref" / "measure.json").read_text())
    assert rc == 0 and rdoc["complete"], rdoc.get("failed")
    assert rdoc["route_sent"]["prime"] == "on" and rdoc["route_sent"]["temperature"] == 0.3 and "parser_retries" not in rdoc["route_sent"]
    assert rdoc["parser_errors"]["n"] == 1 and rdoc["parser_errors"]["service_log_lines"] == 0
    assert all(t["status"] == "ok" for t in rdoc["per_turn"])  # the redraw answered
    rcalls = [json.loads(ln) for ln in (tmp_path / "ref" / "calls.jsonl").read_text().splitlines()]
    redrawn = [c for c in rcalls if c["turn"] == [c for c in rcalls if c.get("parser_error")][0]["turn"] and c["kind"] == "call"]
    assert [c["status"] for c in redrawn] == [500, 200]
    assert rdoc["primes"]["n"] >= 1

    # measured by URL here (a local fake): the reference's image is recorded by hand, as `check` records it by service
    v = L.compare(doc, rdoc, min_turns=1)
    assert v["verdict"] == "INCOMPLETE" and any("image is unknown" in w for w in v["incomplete"]), v
    rdoc["service"] = {"name": "sous-chef-model-eval", "described": True, "image": R7_IMAGE, "traffic": LATEST}
    v = L.compare(doc, rdoc, min_turns=1)
    assert v["verdict"] == "FAIL" and not v["checks"][1]["ok"], v
    # the parser error also ended a turn the reference's redraw saved: one failed turn more than r7
    assert v["checks"][2]["candidate"] == 1 and v["checks"][2]["reference"] == 0 and not v["checks"][2]["ok"], v


def _engine_ok() -> bool:
    from souschef_lab.engine import find_engine
    return bool(shutil.which("node")) and find_engine(None) is not None


def set_store_name(n, body):
    """A turn's first call writes an op (the store name); its continuation (a tool_result last) is left to the
    default text reply — a clean batch fast-finishes without one."""
    last = (body.get("messages") or [{}])[-1]
    content = last.get("content")
    if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
        return None
    return 200, {"id": f"msg_{n}", "type": "message", "role": "assistant", "model": "sous-chef", "stop_reason": "tool_use",
                 "content": [{"type": "text", "text": "Renaming the store."},
                             tool_use("set_fields", {"target": {"kind": "global", "panel": "BrandingSection"},
                                                     "values": {"storeName": "Fresh Cookies"}}, n)],
                 "usage": {"input_tokens": 1500, "output_tokens": 30}}


@pytest.mark.skipif(not (REAL and _engine_ok()), reason="needs go, Postgres, node and the engine ($COOKIE_CUTTER_DIR)")
def test_a_measure_with_the_real_engine(tmp_path, ollama, fake_gcloud, monkeypatch, service_bin):
    """The ops the model writes go through headless.mjs (PersistentEngine), the tool calls are checked against the
    tools the service sent, and the turns end as the service ends them (fast finish after a clean batch)."""
    script, _ = fake_gcloud
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")
    f = ollama("mistral", script=set_store_name)
    a = measure_args(tmp_path, f.url, "chocolate-muffin", service_bin, "real-engine")
    a.pick, a.fake_engine = "real=2", False
    assert L.measure(a) == 0
    doc = json.loads((tmp_path / "real-engine" / "measure.json").read_text())
    assert doc["complete"] and doc["engine"].endswith("headless.mjs") and doc["engine_fingerprint"] not in (None, "fake")
    tc = doc["tool_calls"]
    assert tc["n"] >= 2 and tc["valid"] == tc["n"], tc
    assert all(t["status"] == "ok" for t in doc["per_turn"]) and doc["parser_errors"]["n"] == 0


# --- the profiles, in step with the harness (and through tests/test_harness.py with infra/envs/int and the service) ---

def test_the_profiles_are_the_harness_family_routes():
    """PROFILES is what int's MODEL_ROUTES gives each in-house route: blueberry-muffin is the harness's deployed route
    (qwen), chocolate-muffin the mistral family's (harness.FAMILY_ROUTES, which tests/test_harness.py pins to
    infra/envs/int/main.tf's mistral_route and service/internal/config/chocolate_test.go). fields_context is the
    service's default (on), which the harness asserts in /version instead of sending."""
    from souschef_lab import harness as H
    for prof, fam in (("blueberry-muffin", "qwen"), ("chocolate-muffin", "mistral")):
        spec, p = H.route_spec("u", prof, family=fam), L.PROFILES[prof]
        assert L.FAMILY[prof] == fam
        assert {k: v for k, v in p.items() if k != "fields_context"} == \
            {k: v for k, v in spec.items() if k not in ("kind", "url", "auth", "report_as")}, prof
        assert p["fields_context"] == "on"
    from souschef_lab import chatfmt
    assert L.FAMILY == chatfmt.ROUTE_FAMILIES  # the drift poll's route → family map
