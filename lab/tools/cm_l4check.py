#!/usr/bin/env python3
"""cm_l4check — chocolate-muffin's L4 check (docs/CHOCOLATE-MUFFIN.md): a model service on a real Cloud Run L4,
measured the way int's sous-chef service calls it. Deploy-free: it starts nothing in GCP and writes nothing there;
the only remote calls are to the model URLs it is given, with an ID token, as the pipeline's eval calls them.

    measure  --url URL --profile chocolate-muffin|blueberry-muffin --out DIR   one service → DIR/measure.json
    compare  --candidate A/measure.json --reference B/measure.json [--out DIR]  the verdict (verdict.json, verdict.md)
    check    --out DIR [--ref-service S] [--cand-service S]                     under the pipeline's retrain + gpu-eval
                                                                                 locks: measure the reference (r7 on
                                                                                 sous-chef-model-eval, the blueberry-muffin
                                                                                 route), then the candidate on
                                                                                 sous-chef-model-cand (chocolate-muffin),
                                                                                 then compare
    rows     the exam rows a measure replays

One `measure`:

  1. The serving asserts, straight at the URL: /api/version; /api/show (the family's PARSER and RENDERER — ministral
     with the lab's TEMPLATE byte for byte, or qwen3.5 — the architecture, num_ctx 65536, the tools capability); a
     load and /api/ps at 100 % VRAM; the prompt-size smoke (serving_mistral.SMOKE_REQUEST must render to the lab's
     token count, cached prefix included); the echo-tool smoke at temperature 0 (no parser 500, no call markers leaked
     as text, and — for Ministral — a tool_use).
  2. The replay: the lab's harness (souschef_lab.harness: Postgres + the sous-chef binary, $SOUS_CHEF_BIN — build it
     from THIS tree, the chocolate-muffin profile needs `parser_retries`) with ONE route, the profile's settings
     exactly as int's MODEL_ROUTES gives them (minus `fallback`: the student alone, never Claude), pointed at a local
     proxy that adds `X-Serverless-Authorization: Bearer <ID token>` (what the service's id-token auth sends) and
     records every call. About 40 real-phrasing exam rows (the lab fixtures: real, followup, paraphrase, holdout,
     out_of_scope; a fixed, seeded pick, the same for every URL) run as the eval's `<set>:recorded` arm runs them
     (exam.run_row: history turns, then the scored turn, the engine applying the ops), one turn at a time, each turn
     starting once the model server has been quiet for --gap-s (a merchant's next turn comes after the Primer's
     request, not on top of it). Two warm-up rows first are left out of every number.
  3. measure.json: per call (each /v1/messages the service sent, primes apart) and per turn (POST /v1/turns to the
     end of the turn's last stream, the ok turns) p50 / p90 ms, calls per turn, the tool calls' validity (the name was
     offered and the input passes the tool's input_schema), the parser errors (500s whose message is a tool-call
     parser's: MinistralParser's "tool 'x' not found" or bad arguments JSON, qwen3.5's XML), the other errors, the
     exam pass count (information only), the service's "souschef: model parser error" lines.

`compare` → INCOMPLETE unless the two measures are comparable (same rows, same tool version, engine, gap and call
timeout, the same UTC day, each with at least --min-turns ok turns and its serving asserts passed) and measured what
the switch is about (the candidate on the chocolate-muffin profile, the reference on blueberry-muffin's, two different
URLs, the reference's service serving r7's digest — so it must be measured by service name, its image recorded —, and
every recorded service serving 100 % from its latest revision, unchanged from the start of its measure to the end).
Then PASS when the candidate's per-turn p50 is no worse than the reference's (× (1 + --tolerance), default 0), it drew
0 parser errors, and it ended no more turns failed and drew no more non-parser call errors than the reference; FAIL
otherwise. Exit codes: 0 PASS, 1 FAIL, 2 usage, 3 INCOMPLETE (a measure that could not finish, or a pipeline lock
held by another run, exits 3 too). `verdict.json` names each side's image (`summary.<side>.image`): `make
inhouse-switch VERDICT=…` takes only a PASS whose candidate image is the digest it rolls out.

Auth (--auth): `gcloud` (default: `gcloud auth print-identity-token`, your login — Owner holds run.routes.invoke;
with --impersonate SA the token is minted for that account with the URL as audience), `metadata` (inside a GCP
job: the metadata server, audience = the URL, as model/idtoken.go does), `none` (a local server). Tokens are
cached until 5 minutes before they expire and never written anywhere.

Standard library + the lab. Needs: Postgres server binaries (PG_BIN or PATH), the service binary ($SOUS_CHEF_BIN
or --service-bin), the fields CLI ($SOUS_CHEF_FIELDS_BIN, else `go build` from ../service), node and the engine
(--engine: a cookie-cutter checkout or snapshot, or headless.mjs; else $SOUS_CHEF_ENGINE / $COOKIE_CUTTER_DIR).
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from http import client as httpclient
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

TOOLS_DIR = Path(__file__).resolve().parent
LAB_DIR = TOOLS_DIR.parent
# This tree's lab first, whatever PYTHONPATH says (another checkout's souschef_lab would render other prompts).
while str(LAB_DIR) in sys.path:
    sys.path.remove(str(LAB_DIR))
sys.path.insert(0, str(LAB_DIR))

TOOL_VERSION = 1
MODEL_NAME = "sous-chef"  # the Ollama model inside every model image (lab/model-image)
R7_DIGEST = "sha256:0f301196deae8108e05c0b94d245c628d252d4da28e6bd05b45d143f59d13575"  # blueberry-muffin r7, live 2026-10-19
PROJECT = "breadstack-sous-chef-int"
GPU_REGION = "us-east4"
LIVE_SERVICE = "sous-chef-model"
CAND_SERVICE = "sous-chef-model-cand"
# The reference: the pipeline's eval service. It serves production's digest (the drift workflow points it there
# before every check: r7 on 2026-10-19) on production's shape (L4, 4 CPU, 16 GiB, concurrency 4), and no merchant's
# call shares its one Ollama slot or is slowed by the measure — what production cannot give.
REF_SERVICE = "sous-chef-model-eval"
# `check` holds the pipeline's own locks for both measures (gcsio: state/locks/<name>.json in the bucket): no round
# re-points sous-chef-model-cand, no drift check or round eval shares or re-points sous-chef-model-eval.
CHECK_LOCKS = ("retrain", "gpu-eval")
LOCK_TTL_S = 3 * 3600
METADATA_URL = "http://metadata.google.internal"

# The route settings int's MODEL_ROUTES gives each in-house route (infra/envs/int/main.tf; the service's
# config.Route). The harness adds kind, url, auth and report_as and drops fallback.
COMMON_ROUTE = {"model": MODEL_NAME, "max_tokens": 1024, "stream": False, "fast_finish": "on", "fields_context": "on"}
PROFILES: dict[str, dict] = {
    # live today: r7, Qwen3.5-9B + LoRA (the deployed inhouse_route)
    "blueberry-muffin": {**COMMON_ROUTE, "temperature": 0.3, "prime": "on"},
    # Ministral 3 8B + LoRA: Mistral's sampling, no priming (calibrated for Qwen's hybrid attention), no silent
    # redraw of a MinistralParser 500 (service config.Route.ParserRetries)
    "chocolate-muffin": {**COMMON_ROUTE, "temperature": 0.15, "prime": "off", "parser_retries": "off"},
}
FAMILY = {"blueberry-muffin": "qwen", "chocolate-muffin": "mistral"}
ARCH = {"qwen": "qwen35", "mistral": "mistral3"}
PARSER = {"qwen": "qwen3.5", "mistral": "ministral"}
RENDERER = {"qwen": "qwen3.5", "mistral": None}
CALL_TIMEOUT = "60s"  # int's MODEL_CALL_TIMEOUT (gpu_model_route.call_timeout)

# ~40 rows of the real-phrasing exam (the eval's `<set>:recorded` arm), seeded per set; 2 more warm the server.
DEFAULT_PICK = {"real": 14, "followup": 6, "paraphrase": 8, "holdout": 6, "out_of_scope": 4, "out_of_scope_holdout": 2}
WARMUP_SET, WARMUP_N = "paraphrase", 2
PICK_SEED = "cm-l4check-1"
DEFAULT_GAP_S = 1.5
MIN_OK_TURNS = 30

EXIT_PASS, EXIT_FAIL, EXIT_USAGE, EXIT_INCOMPLETE = 0, 1, 2, 3
SERVICE_PARSER_LINE = "souschef: model parser error"


class L4Error(RuntimeError):
    """The measure cannot go on (exit 3)."""


def say(msg: str) -> None:
    print(f"cm_l4check: {msg}", file=sys.stderr, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def lab_import_check() -> None:
    """souschef_lab must be this tree's (an earlier test run once imported another checkout's)."""
    import souschef_lab
    where = Path(souschef_lab.__file__).resolve()
    if LAB_DIR not in where.parents:
        raise L4Error(f"souschef_lab was imported from {where}, not from {LAB_DIR}: unset PYTHONPATH")


# --- parser errors: the service's model.IsParserMessage, in Python (service/internal/model/parser.go) -------------------

PARSER_PREFIXES = ("invalid character ", "not a valid UTF-8 string", "XML syntax error on line ", "expected element type <",
                   "failed to parse JSON: ")
PARSER_MESSAGES = frozenset({
    "unexpected end of JSON input", "Key path not found", "Unknown value type", "Malformed JSON error",
    "Value is string, but can't find closing '\"' symbol", "Value is array, but can't find closing ']' symbol",
    "Value looks like object, but can't find closing '}' symbol",
    "Value looks like Number/Boolean/None, but can't find its end: ',' or '}' symbol",
    "Value is number, but overflowed while parsing", "Encountered an invalid escape sequence in a string",
    "empty function name"})


# an argument number too large for a float64 (encoding/json into any): the service's numberOverflowPrefix/-Suffix
NUMBER_OVERFLOW = ("json: cannot unmarshal number ", " into Go value of type float64")


def is_parser_message(msg: str) -> bool:
    m = (msg or "").strip()
    if m.startswith("tool '") and m.endswith("' not found"):
        return True
    if m.startswith(NUMBER_OVERFLOW[0]) and m.endswith(NUMBER_OVERFLOW[1]):
        return True
    return m in PARSER_MESSAGES or m.startswith(PARSER_PREFIXES)


def error_message(body: bytes) -> str:
    """The message of an error body: {"error":{"message":…}} (Ollama's /v1/messages), {"error":"…"} (its native API),
    else the text itself, shortened."""
    try:
        doc = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return body[:200].decode("utf-8", "replace")
    err = doc.get("error") if isinstance(doc, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or "")[:300]
    if isinstance(err, str):
        return err[:300]
    return json.dumps(doc)[:200]


# --- stats ---------------------------------------------------------------------------------------------------------------

def pct(values: Iterable[float], q: float) -> float | None:
    """Nearest-rank percentile (as tools/mx_latency.py and mxarm._pct), None for no values."""
    v = sorted(float(x) for x in values if x is not None)
    if not v:
        return None
    return round(v[min(len(v) - 1, max(0, math.ceil(q * len(v)) - 1))], 1)


def dist(values: list[float]) -> dict:
    vals = [float(x) for x in values if x is not None]
    return {"n": len(vals), "p50": pct(vals, 0.5), "p90": pct(vals, 0.9),
            "mean": round(sum(vals) / len(vals), 1) if vals else None}


# --- tool-call validity: the JSON Schema subset the service's tools use ----------------------------------------------------

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _type_ok(value: Any, t: str) -> bool:
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    py = _TYPES.get(t)
    return True if py is None else isinstance(value, py)


def schema_problems(value: Any, schema: Any, path: str = "$") -> list[str]:
    """Violations of `schema` (type, enum, required, properties, additionalProperties, items, minItems, anyOf) —
    what the tools' input_schema uses (fixtures/prompt/tools.json)."""
    if not isinstance(schema, dict):
        return []
    out: list[str] = []
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_type_ok(value, x) for x in types):
            return [f"{path}: want {'|'.join(map(str, types))}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{path}: {json.dumps(value)[:60]} not in enum")
    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        if not any(not schema_problems(value, s, path) for s in schema["anyOf"]):
            out.append(f"{path}: matches no anyOf branch")
    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for k in schema.get("required") or []:
            if k not in value:
                out.append(f"{path}.{k}: required")
        extra = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                out.extend(schema_problems(v, props[k], f"{path}.{k}"))
            elif extra is False:
                out.append(f"{path}.{k}: not allowed")
            elif isinstance(extra, dict):
                out.extend(schema_problems(v, extra, f"{path}.{k}"))
    if isinstance(value, list):
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            out.append(f"{path}: fewer than {schema['minItems']} items")
        if isinstance(schema.get("items"), dict):
            for i, v in enumerate(value):
                out.extend(schema_problems(v, schema["items"], f"{path}[{i}]"))
    return out


def tool_use_check(block: dict, tools: dict[str, dict]) -> dict:
    """{name, valid, problem}: the name was offered in the request and the input passes its input_schema."""
    name = str(block.get("name") or "")
    if name not in tools:
        return {"name": name[:60], "valid": False, "problem": "unknown_tool"}
    inp = block.get("input")
    if not isinstance(inp, dict):
        return {"name": name, "valid": False, "problem": "input_not_object"}
    probs = schema_problems(inp, tools[name])
    return {"name": name, "valid": not probs, "problem": ("schema: " + "; ".join(probs[:3])[:240]) if probs else None}


# --- ID tokens -----------------------------------------------------------------------------------------------------------

def jwt_exp(token: str) -> float | None:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload.encode("ascii"))).get("exp")
        return float(exp) if exp else None
    except (IndexError, ValueError, TypeError, UnicodeDecodeError):
        return None


class IDTokens:
    """The bearer the model service is called with (X-Serverless-Authorization), cached until 5 min before exp."""

    MARGIN_S = 300.0

    def __init__(self, mode: str, audience: str, *, impersonate: str | None = None, gcloud: str = "gcloud",
                 metadata_url: str = METADATA_URL, now: Callable[[], float] = time.time):
        if mode not in ("gcloud", "metadata", "none"):
            raise ValueError(f"auth {mode!r}: gcloud | metadata | none")
        self.mode, self.audience, self.impersonate = mode, audience, impersonate
        self.gcloud, self.metadata_url, self.now = gcloud, metadata_url.rstrip("/"), now
        self._lock = threading.Lock()
        self._token: str | None = None
        self._until = 0.0
        self.minted = 0

    def get(self) -> str | None:
        if self.mode == "none":
            return None
        with self._lock:
            if self._token and self.now() < self._until:
                return self._token
            token = self._mint()
            exp = jwt_exp(token)
            self._token = token
            self._until = (exp - self.MARGIN_S) if exp else self.now() + 1800
            self.minted += 1
            return token

    def _mint(self) -> str:
        if self.mode == "metadata":
            q = urlparse.urlencode({"audience": self.audience})
            req = urlrequest.Request(f"{self.metadata_url}/computeMetadata/v1/instance/service-accounts/default/identity?{q}",
                                     headers={"Metadata-Flavor": "Google"})
            try:
                with urlrequest.urlopen(req, timeout=10) as r:
                    return r.read().decode("utf-8").strip()
            except (urlerror.URLError, OSError) as e:
                raise L4Error(f"metadata server: {e}") from None
        cmd = [self.gcloud, "auth", "print-identity-token"]
        if self.impersonate:
            cmd += [f"--impersonate-service-account={self.impersonate}", f"--audiences={self.audience}", "--include-email"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=False)
        except (OSError, subprocess.SubprocessError) as e:
            raise L4Error(f"{' '.join(cmd[:3])}: {e}") from None
        token = (r.stdout or "").strip()
        if r.returncode != 0 or not token or " " in token:
            raise L4Error(f"{' '.join(cmd[:3])} exited {r.returncode}: {(r.stderr or '').strip()[-300:]}")
        return token

    def headers(self) -> dict[str, str]:
        t = self.get()
        return {"X-Serverless-Authorization": f"Bearer {t}"} if t else {}


def http_json(url: str, body: Any = None, *, tokens: IDTokens | None = None, timeout: float = 60.0,
              method: str | None = None) -> tuple[int, Any, float]:
    """(status, parsed body or {"raw": …}, seconds)."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urlrequest.Request(url, data=data, method=method or ("POST" if data is not None else "GET"))
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for k, v in (tokens.headers() if tokens else {}).items():
        req.add_header(k, v)
    t0 = time.monotonic()
    try:
        with urlrequest.urlopen(req, timeout=timeout) as r:
            raw, status = r.read(), r.status
    except urlerror.HTTPError as e:
        raw, status = e.read(), e.code
    took = time.monotonic() - t0
    try:
        return status, (json.loads(raw.decode("utf-8")) if raw else None), took
    except (ValueError, UnicodeDecodeError):
        return status, {"raw": raw[:300].decode("utf-8", "replace")}, took


# --- serving asserts -----------------------------------------------------------------------------------------------------

ECHO_TOOL = {"name": "echo", "description": "Echo the given text back.",
             "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}
LEAK_MARKERS = {"mistral": ("[TOOL_CALLS]", "[ARGS]"), "qwen": ("<tool_call>", "<function=")}


def _modelfile_line(modelfile: str, keyword: str) -> str | None:
    for ln in (modelfile or "").splitlines():
        parts = ln.strip().split(None, 1)
        if len(parts) == 2 and parts[0].upper() == keyword:
            return parts[1].strip()
    return None


def sha256_text(s: str | None) -> str | None:
    return None if s is None else hashlib.sha256(s.encode("utf-8")).hexdigest()


def served_checks(url: str, family: str, tokens: IDTokens | None, *, load_timeout: float = 900.0) -> dict:
    """The serving asserts (module docstring, step 1): {"ok", "problems", …what was seen}. Never raises on a bad
    answer; an unreachable URL or a refused token is a problem like any other."""
    from souschef_lab import serving_mistral as SM
    url = url.rstrip("/")
    out: dict[str, Any] = {"family": family, "at": utc_now(), "problems": []}
    problems: list[str] = out["problems"]

    def fail(msg: str) -> dict:
        problems.append(msg)
        out["ok"] = False
        return out

    try:
        st, ver, _ = http_json(url + "/api/version", tokens=tokens, timeout=60)
    except (urlerror.URLError, OSError) as e:
        return fail(f"unreachable: {e}")
    if st in (401, 403):
        return fail(f"/api/version answered {st}: the ID token was refused (audience, or run.routes.invoke)")
    if st != 200:
        return fail(f"/api/version answered {st}")
    out["ollama_version"] = (ver or {}).get("version") if isinstance(ver, dict) else None
    st, show, _ = http_json(url + "/api/show", {"model": MODEL_NAME}, tokens=tokens, timeout=120)
    if st != 200 or not isinstance(show, dict):
        return fail(f"/api/show answered {st}")
    modelfile = str(show.get("modelfile") or "")
    parser = show.get("parser") or _modelfile_line(modelfile, "PARSER")
    renderer = show.get("renderer") or _modelfile_line(modelfile, "RENDERER")
    template = show.get("template")
    caps = list(show.get("capabilities") or [])
    arch = (show.get("model_info") or {}).get("general.architecture")
    params = str(show.get("parameters") or "")
    num_ctx = any(ln.split()[:2] == ["num_ctx", "65536"] for ln in params.splitlines())
    out["show"] = {"parser": parser, "renderer": renderer, "template_sha256": sha256_text(template), "architecture": arch,
                   "capabilities": caps, "num_ctx_65536": num_ctx,
                   "temperature": next((ln.split()[1] for ln in params.splitlines() if ln.split()[:1] == ["temperature"]), None)}
    if parser != PARSER[family]:
        problems.append(f"PARSER {parser!r}, want {PARSER[family]!r}")
    if (renderer or None) != RENDERER[family]:
        problems.append(f"RENDERER {renderer!r}, want {RENDERER[family]!r}")
    if family == "mistral" and template != SM.TEMPLATE:
        problems.append(f"TEMPLATE sha256 {sha256_text(template)} is not the lab's serving_mistral.TEMPLATE "
                        f"({sha256_text(SM.TEMPLATE)}): the image's Modelfile differs (a trailing newline?)")
    if arch != ARCH[family]:
        problems.append(f"architecture {arch!r}, want {ARCH[family]!r}")
    if "tools" not in caps:
        problems.append(f"no tools capability ({caps})")
    if not num_ctx:
        problems.append("num_ctx is not 65536")
    st, body, took = http_json(url + "/api/generate", {"model": MODEL_NAME, "prompt": "", "stream": False},
                               tokens=tokens, timeout=load_timeout)
    out["load_s"] = round(took, 1)
    if st != 200:
        return fail(f"load-only /api/generate answered {st}: {str(body)[:200]}")
    st, ps, _ = http_json(url + "/api/ps", tokens=tokens, timeout=30)
    mods = [m for m in (ps or {}).get("models") or [] if str(m.get("name", "")).startswith(MODEL_NAME)] if isinstance(ps, dict) else []
    if not mods:
        problems.append("/api/ps lists no loaded sous-chef model")
    else:
        size, vram = int(mods[0].get("size") or 0), int(mods[0].get("size_vram") or 0)
        out["ps"] = {"size": size, "size_vram": vram, "context_length": mods[0].get("context_length")}
        if size <= 0 or vram < size:
            problems.append(f"not 100% GPU: size_vram {vram} of {size} bytes")
    want_n = SM.SMOKE_PROMPT_TOKENS.get(family)
    st, sm, took = http_json(url + "/v1/messages", dict(SM.SMOKE_REQUEST, model=MODEL_NAME, max_tokens=1, temperature=0,
                                                        stream=False), tokens=tokens, timeout=300)
    # Ollama splits the prompt into input_tokens (evaluated) and cache_read_input_tokens (a prefix the slot still held:
    # anthropic.UsageFromMetrics); a warm server may hold the shared [BOS][SYSTEM_PROMPT], so the size is the sum
    usage = ((sm or {}).get("usage") or {}) if isinstance(sm, dict) else {}
    got_n = None if usage.get("input_tokens") is None else int(usage["input_tokens"]) + int(usage.get("cache_read_input_tokens") or 0)
    out["prompt_smoke"] = {"status": st, "input_tokens": got_n, "cached": usage.get("cache_read_input_tokens"),
                           "expected": want_n, "s": round(took, 2)}
    if st != 200 or got_n != want_n:
        problems.append(f"the smoke prompt is {got_n} tokens (status {st}), want {want_n}: the served template, BOS or "
                        "control-token handling differs from the lab's render")
    st, msg, took = http_json(url + "/v1/messages", {"model": MODEL_NAME, "max_tokens": 128, "temperature": 0, "stream": False,
                                                     "thinking": {"type": "disabled"}, "tools": [ECHO_TOOL],
                                                     "messages": [{"role": "user", "content": "Use the echo tool to echo the word hello."}]},
                              tokens=tokens, timeout=300)
    content = [c for c in ((msg or {}).get("content") or []) if isinstance(c, dict)] if isinstance(msg, dict) else []
    kinds = [c.get("type") for c in content]
    text = "".join(str(c.get("text") or "") for c in content if c.get("type") == "text")
    leaked = [m for m in LEAK_MARKERS.get(family, ()) if m in text]
    err = error_message(json.dumps(msg).encode()) if st != 200 else None
    out["tool_smoke"] = {"status": st, "content": kinds, "stop_reason": (msg or {}).get("stop_reason") if isinstance(msg, dict) else None,
                         "parser_error": bool(st == 500 and err and is_parser_message(err)), "leaked_markers": leaked,
                         "s": round(took, 2)}
    if st != 200:
        problems.append(f"the echo-tool smoke answered {st}: {err}")
    elif leaked:
        problems.append(f"the echo-tool smoke answered text with the call markers {leaked}: the control tokens do not "
                        "reach the parser")
    elif "tool_use" not in kinds and family == "mistral":
        # Ministral calls echo here whenever the tools reach it and its [TOOL_CALLS] reaches the parser; a Qwen
        # model may answer in text, which is only recorded (mxarm.assert_served's rule)
        problems.append(f"the echo-tool smoke made no tool call at temperature 0 ({kinds})")
    out["ok"] = not problems
    return out


# --- the proxy -------------------------------------------------------------------------------------------------------------

HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "trailers",
                 "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding", "authorization",
                 "x-serverless-authorization", "x-api-key"})


class ModelProxy:
    """127.0.0.1:<port> → the model service. Every request goes out as the service wrote it (identity encoding),
    plus the ID token; every POST /v1/messages is recorded: its wall time here, status, kind (a prime is max_tokens
    1), the turn it belongs to (tag), token counts, the tool calls' validity and any error's message."""

    def __init__(self, target: str, tokens: IDTokens | None = None, *, timeout_s: float = 600.0):
        self.target = target.rstrip("/")
        u = urlparse.urlsplit(self.target)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError(f"not an http(s) URL: {target!r}")
        self.scheme, self.host, self.port = u.scheme, u.hostname, u.port
        self.base_path = u.path.rstrip("/")
        self.tokens = tokens
        self.timeout_s = timeout_s
        self.calls: list[dict] = []
        self.tag: str | None = None
        self._lock = threading.Lock()
        self._inflight = 0
        self._last = time.monotonic()
        self._idle = threading.Condition(self._lock)
        self._pool: "queue.LifoQueue[httpclient.HTTPConnection]" = queue.LifoQueue()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> "ModelProxy":
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # the service log has its own lines
                return

            def do_GET(self):
                proxy._forward(self)

            def do_POST(self):
                proxy._forward(self)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        while True:
            try:
                self._pool.get_nowait().close()
            except queue.Empty:
                break

    def __enter__(self) -> "ModelProxy":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _conn(self) -> httpclient.HTTPConnection:
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            cls = httpclient.HTTPSConnection if self.scheme == "https" else httpclient.HTTPConnection
            return cls(self.host, self.port, timeout=self.timeout_s)

    def _upstream(self, method: str, path: str, headers: dict, body: bytes) -> tuple[int, list[tuple[str, str]], bytes]:
        for attempt in (1, 2):  # a pooled connection the server closed meanwhile: once more on a fresh one
            conn = self._conn()
            try:
                conn.request(method, path, body=body if method != "GET" else None, headers=headers)
                r = conn.getresponse()
                data = r.read()
                if r.getheader("connection", "").lower() != "close":
                    self._pool.put(conn)
                else:
                    conn.close()
                return r.status, r.getheaders(), data
            except (httpclient.RemoteDisconnected, ConnectionResetError, BrokenPipeError):
                conn.close()
                if attempt == 2:
                    raise
            except BaseException:
                conn.close()
                raise
        raise AssertionError("unreachable")

    def _forward(self, h: BaseHTTPRequestHandler) -> None:
        n = int(h.headers.get("Content-Length") or 0)
        body = h.rfile.read(n) if n > 0 else b""
        headers = {k: v for k, v in h.headers.items() if k.lower() not in HOP}
        headers["Host"] = self.host if self.port in (None, 80, 443) else f"{self.host}:{self.port}"
        headers["Accept-Encoding"] = "identity"
        record = h.command == "POST" and urlparse.urlsplit(h.path).path.endswith("/v1/messages")
        with self._lock:
            self._inflight += 1
            tag = self.tag
        t0 = time.monotonic()
        status, rheaders, data, exc = 502, [], b"", None
        try:
            if self.tokens is not None:
                headers.update(self.tokens.headers())
            status, rheaders, data = self._upstream(h.command, self.base_path + h.path, headers, body)
        except Exception as e:  # noqa: BLE001 — answered 502, recorded, the service's SDK retries it like a reset
            exc = f"{type(e).__name__}: {e}"[:200]
            data = json.dumps({"type": "error", "error": {"type": "api_error", "message": f"cm_l4check proxy: {exc}"}}).encode()
            rheaders = [("Content-Type", "application/json")]
        ms = (time.monotonic() - t0) * 1000.0
        try:
            if record:
                self._record(tag, body, status, data, ms, exc)
            h.send_response(status)
            for k, v in rheaders:
                if k.lower() not in HOP and k.lower() != "content-encoding":
                    h.send_header(k, v)
            h.send_header("Content-Length", str(len(data)))
            h.end_headers()
            h.wfile.write(data)
        finally:
            with self._idle:
                self._inflight -= 1
                self._last = time.monotonic()
                self._idle.notify_all()

    def _record(self, tag: str | None, body: bytes, status: int, data: bytes, ms: float, exc: str | None) -> None:
        try:
            req = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            req = {}
        tools = {str(t.get("name")): t.get("input_schema") or {} for t in req.get("tools") or [] if isinstance(t, dict)}
        rec: dict[str, Any] = {"turn": tag, "status": status, "ms": round(ms, 1),
                               "kind": "prime" if req.get("max_tokens") == 1 else "call",
                               "request": {"max_tokens": req.get("max_tokens"), "temperature": req.get("temperature"),
                                           "top_p": req.get("top_p"), "stream": req.get("stream"),
                                           "thinking": (req.get("thinking") or {}).get("type") if isinstance(req.get("thinking"), dict) else None,
                                           "messages": len(req.get("messages") or []), "tools": len(tools),
                                           "system_chars": sum(len(str(b.get("text") or "")) for b in req.get("system") or [] if isinstance(b, dict))
                                           if isinstance(req.get("system"), list) else len(str(req.get("system") or ""))}}
        if status == 200:
            try:
                resp = json.loads(data.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                resp = {}
            content = [c for c in resp.get("content") or [] if isinstance(c, dict)]
            rec["stop_reason"] = resp.get("stop_reason")
            u = resp.get("usage") or {}
            rec["usage"] = {k: u.get(k) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens")}
            rec["text_chars"] = sum(len(str(c.get("text") or "")) for c in content if c.get("type") == "text")
            rec["tool_uses"] = [tool_use_check(c, tools) for c in content if c.get("type") == "tool_use"]
        else:
            msg = exc or error_message(data)
            rec["error"] = msg[:200]
            rec["parser_error"] = bool(status == 500 and is_parser_message(msg))
        with self._lock:
            rec["i"] = len(self.calls)
            self.calls.append(rec)

    def wait_idle(self, quiet_s: float, timeout_s: float = 300.0) -> float:
        """Block until nothing is in flight and nothing has been for quiet_s (the Primer's request after a turn,
        then a merchant's pause). Returns the seconds waited."""
        t0 = time.monotonic()
        with self._idle:
            while True:
                waited = time.monotonic() - t0
                if waited >= timeout_s:
                    return waited
                if self._inflight == 0:
                    rest = quiet_s - (time.monotonic() - self._last)
                    if rest <= 0:
                        return waited
                    self._idle.wait(timeout=min(rest, timeout_s - waited))
                else:
                    self._idle.wait(timeout=min(1.0, timeout_s - waited))


# --- the rows --------------------------------------------------------------------------------------------------------------

def _seeded(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: hashlib.sha256(f"{PICK_SEED}:{r.get('id')}".encode()).hexdigest())


def pick_rows(pick: dict[str, int] | None = None, warmup: int = WARMUP_N) -> tuple[list[dict], list[dict]]:
    """(measured rows, warm-up rows): per set the first n of a seeded order (the same for every URL, every day),
    capability rows (a tool the live model was never offered) left out; the warm-up rows are the next ones of
    WARMUP_SET. The eval's font policy is applied, as mxarm.exam_rows does (scoring only)."""
    from souschef_lab.exam import load_set
    from souschef_lab.r4score import apply_font_policy, policy_of
    from souschef_lab.retrain.eval_stage import is_capability_row
    pick = dict(DEFAULT_PICK if pick is None else pick)
    font = policy_of()["font"]
    measured: list[dict] = []
    spare: list[dict] = []
    for s in sorted(set(pick) | {WARMUP_SET}, key=lambda x: list(DEFAULT_PICK).index(x) if x in DEFAULT_PICK else 99):
        rows = _seeded([r for r in load_set(s) if not is_capability_row(r)])
        n = int(pick.get(s, 0))
        if n > len(rows):
            raise L4Error(f"set {s}: {n} rows asked, {len(rows)} there")
        measured.extend(apply_font_policy(r, font)[0] for r in rows[:n])
        if s == WARMUP_SET:
            spare = rows[n:]
    # a followup row shares its id (and text) with a paraphrase row: warm up on neither
    ids = {r.get("id") for r in measured}
    warm = [apply_font_policy(r, font)[0] for r in spare if r.get("id") not in ids][:warmup]
    return measured, warm


def row_key(case: dict) -> str:
    """set:id — a followup row and a paraphrase row may share an id."""
    return f"{case.get('set')}:{case.get('id')}"


def rows_fingerprint(rows: list[dict]) -> str:
    keys = [[r.get("set"), r.get("id"), r.get("request"), [h.get("user") for h in r.get("history") or []],
             r.get("context_ref"), r.get("manifest_hash")] for r in rows]
    return hashlib.sha256(json.dumps(keys, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def parse_pick(s: str | None) -> dict[str, int] | None:
    if not s:
        return None
    out = {}
    for part in s.split(","):
        name, _, n = part.strip().partition("=")
        if name not in DEFAULT_PICK or not n.strip().isdigit():
            raise ValueError(f"--pick {part!r}: <set>=<n>, sets {', '.join(DEFAULT_PICK)}")
        out[name] = int(n)
    return out


# --- the replay ------------------------------------------------------------------------------------------------------------

def turn_clock_client(base_cls):
    """collect.Client that times turns: each POST /v1/turns waits for the model server to be quiet (proxy.wait_idle)
    and then tags the proxy with the turn; the turn ends with its last stream (start or tool-results)."""

    class TurnClock(base_cls):
        def __init__(self, *a, proxy: ModelProxy, gap_s: float, **kw):
            super().__init__(*a, **kw)
            self.proxy, self.gap_s = proxy, gap_s
            self.row_id: str | None = None
            self.phase = "measure"
            self.turns: list[dict] = []

        def begin_row(self, row_id: str, phase: str) -> None:
            self.row_id, self.phase = row_id, phase

        def start_turn(self, body: dict):
            waited = self.proxy.wait_idle(self.gap_s)
            n = sum(1 for t in self.turns if t["row"] == self.row_id and t["phase"] == self.phase)
            key = f"{self.phase}:{self.row_id}#{n + 1}"
            self.proxy.tag = key
            t = {"key": key, "row": self.row_id, "phase": self.phase, "waited_s": round(waited, 2), "t0": time.monotonic(), "t1": None}
            self.turns.append(t)
            try:
                return super().start_turn(body)
            finally:
                t["t1"] = time.monotonic()

        def tool_results(self, turn_id: str, body: dict):
            try:
                return super().tool_results(turn_id, body)
            finally:
                if self.turns:
                    self.turns[-1]["t1"] = time.monotonic()

    return TurnClock


def turn_outcomes(row: dict) -> list[dict]:
    """[{status, code}] of a replayed row's turns in order: its history turns, then the scored turn (absent when a
    history turn failed: run_row stops there)."""
    out = [{"status": h.get("status"), "code": h.get("code")} for h in row.get("history_turns") or []]
    if row.get("code") != "history_failed":
        out.append({"status": row.get("status"), "code": row.get("code")})
    return out


def make_harness(routes: dict[str, str], **kw):
    """souschef_lab.harness.Harness, with Postgres's socket folder under $TMPDIR when that path is short enough (a
    socket path is limited to ~107 bytes, which is why the harness itself uses /tmp)."""
    from souschef_lab import harness as H

    class Harness(H.Harness):
        def start_postgres(self) -> None:
            root = os.environ.get("TMPDIR") or ""
            if not root or len(root) > 70 or not Path(root).is_dir():
                return super().start_postgres()
            real = H.tempfile.mkdtemp

            def mkdtemp(*args, **kwargs):
                if kwargs.get("prefix") == "pgs-":
                    kwargs["dir"] = root
                return real(*args, **kwargs)

            H.tempfile.mkdtemp = mkdtemp
            try:
                return super().start_postgres()
            finally:
                H.tempfile.mkdtemp = real

    return Harness(routes, **kw)


BUCKET = "gs://breadstack-sous-chef-int-pipeline"


def resolve_engine(engine: str | None) -> str | None:
    """--engine as headless.mjs: a file as given; a cookie-cutter checkout or snapshot → its headless.mjs;
    `bucket` / `bucket:<launcher sha>` → the pipeline's vendored snapshot (read-only: engine/<sha>/, newest by
    default — what the eval runs), extracted under $TMPDIR; None = the lab's own lookup ($SOUS_CHEF_ENGINE,
    $COOKIE_CUTTER_DIR, the sibling checkout)."""
    if not engine:
        return None
    if engine == "bucket" or engine.startswith("bucket:"):
        from souschef_lab import enginebundle as EB
        from souschef_lab import gcsio
        want = engine.partition(":")[2] or "newest"
        try:
            b = EB.ensure(gcsio.bucket(os.environ.get("PIPELINE_BUCKET") or BUCKET), want,
                          workdir=Path(os.environ.get("TMPDIR") or tempfile.gettempdir()) / "cm-l4-engine")
        except (EB.BundleError, gcsio.StoreError) as e:
            raise L4Error(f"--engine {engine}: {e}") from None
        return str(b.engine)
    from souschef_lab.engine import ENGINE_REL_PATH
    p = Path(engine)
    if p.is_dir():
        p = p / ENGINE_REL_PATH
    if not p.is_file():
        raise L4Error(f"--engine {engine}: no headless.mjs there")
    return str(p.resolve())


def replay(*, url: str, label: str, profile: dict, tokens: IDTokens | None, rows: list[dict], warm: list[dict],
           work: Path, service_bin: str | None, engine: str | None, fake_engine: bool, gap_s: float,
           call_timeout: str, progress: Callable[[str], None] = say) -> dict:
    """The harness on one route → the proxy → url; every row through exam.run_row. Returns the raw material."""
    from souschef_lab import collect as C
    from souschef_lab import probe as PR
    from souschef_lab.drift import ManifestEngine, home_context
    from souschef_lab.exam import MANIFEST_FILE, run_row, vintage_state
    from souschef_lab.manifest import Manifest
    lab_import_check()
    mpath = Path(MANIFEST_FILE)
    m = Manifest(json.loads(mpath.read_text(encoding="utf-8")), mpath)
    overrides = dict(profile)
    proxy = ModelProxy(url, tokens).start()
    pe = None
    trimmed: list[dict] = []
    try:
        with make_harness({label: proxy.url}, run_id=f"l4{hashlib.sha256(label.encode()).hexdigest()[:6]}",
                      workdir=work / "harness", binary=service_bin, auth="none", route_overrides=overrides,
                      model_call_timeout=call_timeout, max_concurrent_turns=2) as h:
            ri = (h.version.get("route_info") or {}).get(label) or {}
            if profile.get("parser_retries") == "off" and ri.get("parser_retries") != "off":
                raise L4Error(f"the service binary ({h.binary}) does not state parser_retries \"off\" for {label}: a build "
                              "before the option would redraw every parser error — build it from this tree "
                              "(go build -o … ./cmd/sous-chef in service/) and pass --service-bin or SOUS_CHEF_BIN")
            route_sent = {k: v for k, v in h.model_routes()[label].items() if k != "url"}
            client0, _ = C.connect(h.url, h.key, mpath, rows + warm, sse=False, timeout_s=300, build_id="lab-l4check")
            fields, frec = C.resolve_fields("auto", mpath, client0, label)
            if fake_engine:
                eng: Any = ManifestEngine(m)
            else:
                from souschef_lab.engine import PersistentEngine
                pe = PersistentEngine(engine, manifest_path=mpath, size=1)
                eng = C.DictEngine(pe)
            state = vintage_state(mpath, home_context(rows + warm))
            assets = C.load_assets(C.DEFAULT_ASSETS)
            client = turn_clock_client(C.Client)(h.url, h.key, timeout_s=300, build_id="lab-l4check", proxy=proxy, gap_s=gap_s)
            outcomes: dict[str, list[dict]] = {}
            t_start = time.monotonic()
            for phase, batch in (("warmup", warm), ("measure", rows)):
                for i, case in enumerate(batch, 1):
                    client.begin_row(row_key(case), phase)
                    try:
                        row = run_row(case, client, eng, state=PR.run_state(case, state), route=label, sample=1,
                                      assets=assets, fields=fields, history_mode="live",
                                      site_prefix=f"lab-l4{hashlib.sha256(label.encode()).hexdigest()[:4]}{phase[0]}-")
                    except Exception as e:  # noqa: BLE001 — one row's lab failure is recorded, the run goes on
                        row = {"status": "error", "code": "lab_exception", "error": f"{type(e).__name__}: {e}"[:300]}
                    outcomes[f"{phase}:{row_key(case)}"] = turn_outcomes(row)
                    if phase == "measure":
                        trimmed.append({"key": row_key(case), "id": case.get("id"), "set": case.get("set"), "status": row.get("status"),
                                        "code": row.get("code"), "verdict": row.get("verdict"),
                                        "pass": int(row.get("status") == "ok" and row.get("verdict") in ("pass", "unsupported_ok")),
                                        "history_turns": len(row.get("history_turns") or []), "latency_ms": row.get("latency_ms")})
                        progress(f"{label}: {i}/{len(batch)} {row_key(case)} {row.get('status')}"
                                 f"{' ' + str(row.get('code')) if row.get('code') else ''}")
            proxy.wait_idle(gap_s, timeout_s=120)
            log_text = (work / "harness" / "service.log").read_text(encoding="utf-8", errors="replace") \
                if (work / "harness" / "service.log").is_file() else ""
            fp = None
            if not fake_engine:
                from souschef_lab.engine import engine_fingerprint
                fp = (engine_fingerprint(getattr(pe, "path", None) or engine) or {}).get("fingerprint")
            return {"route_sent": route_sent, "route_info": ri, "build_id": h.version.get("build_id"),
                    "service_bin": str(h.binary), "fields": frec, "turns": client.turns, "outcomes": outcomes,
                    "rows": trimmed, "calls": list(proxy.calls), "replay_s": round(time.monotonic() - t_start, 1),
                    "service_log": log_text, "engine": "fake" if fake_engine else str(getattr(pe, "path", engine) or ""),
                    "engine_fingerprint": "fake" if fake_engine else fp}
    finally:
        if pe is not None:
            pe.close()
        proxy.stop()


def summarize(raw: dict) -> dict:
    """measure.json's numbers from replay()'s material (measured rows only; warm-up and primes apart)."""
    turns = [t for t in raw["turns"] if t["phase"] == "measure"]
    by_key: dict[str, list[dict]] = {}
    for c in raw["calls"]:
        by_key.setdefault(str(c.get("turn")), []).append(c)
    per_turn = []
    for t in turns:
        row_turns = [x for x in turns if x["row"] == t["row"]]
        idx = row_turns.index(t)
        oc = (raw["outcomes"].get(f"measure:{t['row']}") or [])
        outcome = oc[idx] if idx < len(oc) else {"status": "error", "code": "unrecorded"}
        calls = [c for c in by_key.get(t["key"], []) if c["kind"] == "call"]
        per_turn.append({"key": t["key"], "row": t["row"], "ms": round(((t["t1"] or t["t0"]) - t["t0"]) * 1000.0, 1),
                         "status": outcome.get("status"), "code": outcome.get("code"), "calls": len(calls),
                         "parser_errors": sum(1 for c in calls if c.get("parser_error")), "waited_s": t["waited_s"]})
    measured_keys = {t["key"] for t in turns}
    calls = [c for c in raw["calls"] if c.get("turn") in measured_keys]
    real = [c for c in calls if c["kind"] == "call"]
    primes = [c for c in calls if c["kind"] == "prime"]
    ok_calls = [c for c in real if c["status"] == 200]
    uses = [u for c in ok_calls for u in c.get("tool_uses") or []]
    problems: dict[str, int] = {}
    for u in uses:
        if not u["valid"]:
            k = "unknown_tool" if u["problem"] == "unknown_tool" else ("input_not_object" if u["problem"] == "input_not_object" else "schema")
            problems[k] = problems.get(k, 0) + 1
    parser = [c for c in raw["calls"] if c.get("parser_error")]
    statuses: dict[str, int] = {}
    for c in real:
        statuses[str(c["status"])] = statuses.get(str(c["status"]), 0) + 1
    ok_turns = [t for t in per_turn if t["status"] == "ok"]
    codes: dict[str, int] = {}
    for t in per_turn:
        if t["status"] != "ok":
            codes[str(t["code"] or t["status"])] = codes.get(str(t["code"] or t["status"]), 0) + 1
    service_lines = [ln for ln in raw.get("service_log", "").splitlines() if SERVICE_PARSER_LINE in ln]
    rows = raw["rows"]
    return {
        "turns": {**dist([t["ms"] for t in ok_turns]), "n_all": len(per_turn), "n_ok": len(ok_turns), "errors": codes,
                  "calls_per_turn_mean": round(sum(t["calls"] for t in per_turn) / len(per_turn), 2) if per_turn else None},
        "calls": {**dist([c["ms"] for c in ok_calls]), "n_all": len(real), "status": statuses,
                  "output_tokens": dist([(c.get("usage") or {}).get("output_tokens") for c in ok_calls]),
                  "input_tokens": dist([(c.get("usage") or {}).get("input_tokens") for c in ok_calls]),
                  "cache_read_tokens": dist([(c.get("usage") or {}).get("cache_read_input_tokens") for c in ok_calls])},
        "primes": {**dist([c["ms"] for c in primes]), "status": sorted({str(c["status"]) for c in primes})},
        "tool_calls": {"n": len(uses), "valid": sum(1 for u in uses if u["valid"]),
                       "valid_rate": round(sum(1 for u in uses if u["valid"]) / len(uses), 4) if uses else None,
                       "invalid": problems,
                       "examples": [u for u in uses if not u["valid"]][:8]},
        "parser_errors": {"n": len(parser), "in_measured_turns": sum(1 for c in calls if c.get("parser_error")),
                          "messages": sorted({c.get("error", "")[:120] for c in parser})[:10],
                          "service_log_lines": len(service_lines)},
        "other_errors": other_errors(real),
        "exam": {"rows": len(rows), "scored_ok": sum(1 for r in rows if r["status"] == "ok"), "pass": sum(r["pass"] for r in rows)},
        "per_turn": per_turn,
    }


def other_errors(calls: list[dict]) -> dict[str, int]:
    """Non-200 calls that are not parser errors, by status (0 = the proxy could not reach the server)."""
    out: dict[str, int] = {}
    for c in calls:
        if c["status"] != 200 and not c.get("parser_error"):
            out[str(c["status"])] = out.get(str(c["status"]), 0) + 1
    return out


def measure(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    label = a.label or a.profile
    profile = dict(PROFILES[a.profile])
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        try:
            profile[k] = json.loads(v)
        except ValueError:
            profile[k] = v  # on / off / a name
    family = FAMILY[a.profile]
    tokens = IDTokens(a.auth, a.url.rstrip("/"), impersonate=a.impersonate)
    doc: dict[str, Any] = {"tool": "cm_l4check", "tool_version": TOOL_VERSION, "label": label, "profile": a.profile,
                           "family": family, "url": a.url.rstrip("/"), "auth": a.auth, "impersonate": a.impersonate,
                           "measured_at": utc_now(), "day": utc_now()[:10], "gap_s": a.gap_s, "call_timeout": a.call_timeout,
                           "complete": False}
    if a.service_name:
        doc["service"] = describe_service(a.service_name)
    rows, warm = pick_rows(parse_pick(a.pick), a.warmup)
    if a.limit:
        rows = rows[:a.limit]
    doc["rows"] = [row_key(r) for r in rows]
    doc["rows_sha256"] = rows_fingerprint(rows)
    doc["warmup_rows"] = [row_key(r) for r in warm]
    rc = EXIT_INCOMPLETE
    work = Path(tempfile.mkdtemp(prefix="cm-l4-", dir=os.environ.get("TMPDIR") or None))
    t0 = time.monotonic()
    try:
        say(f"{label}: serving asserts on {doc['url']}")
        doc["served"] = served_checks(a.url, family, tokens)
        if not doc["served"]["ok"] and not a.ignore_served:
            raise L4Error("serving asserts failed: " + "; ".join(doc["served"]["problems"]))
        say(f"{label}: replaying {len(rows)} rows (+{len(warm)} warm-up) through the harness")
        raw = replay(url=a.url, label=label, profile=profile, tokens=tokens, rows=rows, warm=warm, work=work,
                     service_bin=a.service_bin, engine=resolve_engine(a.engine), fake_engine=a.fake_engine, gap_s=a.gap_s,
                     call_timeout=a.call_timeout)
        doc.update({k: raw[k] for k in ("route_sent", "route_info", "build_id", "fields", "replay_s", "engine",
                                        "engine_fingerprint")})
        doc.update(summarize(raw))
        with open(out / "calls.jsonl", "w", encoding="utf-8") as f:
            for c in raw["calls"]:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
            for r in raw["rows"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (out / "service.log").write_text(raw["service_log"], encoding="utf-8")
        if a.service_name:  # the same image and traffic after the replay: nothing re-pointed the service meanwhile
            doc["service_after"] = describe_service(a.service_name)
        doc["complete"] = True
        rc = EXIT_PASS
    except L4Error as e:
        doc["failed"] = str(e)
        say(f"{label}: INCOMPLETE — {e}")
    except Exception as e:  # noqa: BLE001 — written down, never lost
        doc["failed"] = f"{type(e).__name__}: {e}"[:1000]
        say(f"{label}: INCOMPLETE — {doc['failed']}")
    finally:
        doc["wall_s"] = round(time.monotonic() - t0, 1)
        doc["tokens_minted"] = tokens.minted
        (out / "measure.json").write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
        if a.keep_work:
            say(f"work kept in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    if doc["complete"]:
        t, c = doc["turns"], doc["calls"]
        say(f"{label}: per turn p50 {t['p50']} / p90 {t['p90']} ms ({t['n_ok']}/{t['n_all']} ok), per call p50 {c['p50']} / "
            f"p90 {c['p90']} ms, parser errors {doc['parser_errors']['n']}, tool calls valid "
            f"{doc['tool_calls']['valid']}/{doc['tool_calls']['n']} → {out / 'measure.json'}")
    return rc


def describe_service(name: str, project: str = PROJECT, region: str = GPU_REGION) -> dict:
    """The Cloud Run service's URL and the image digest its traffic goes to (read-only gcloud), for the record."""
    cmd = ["gcloud", "run", "services", "describe", name, "--project", project, "--region", region, "--format=json"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        doc = json.loads(r.stdout) if r.returncode == 0 else {}
    except (OSError, subprocess.SubprocessError, ValueError):
        doc = {}
    if not doc:
        return {"name": name, "described": False}
    img = ((((doc.get("spec") or {}).get("template") or {}).get("spec") or {}).get("containers") or [{}])[0].get("image")
    return {"name": name, "described": True, "url": (doc.get("status") or {}).get("url"), "image": img,
            "traffic": (doc.get("status") or {}).get("traffic")}


# --- compare ---------------------------------------------------------------------------------------------------------------

def serves_latest(svc: dict) -> bool:
    """A described service's traffic is 100 % on its latest revision: the template image it records is what answered."""
    return any(t.get("latestRevision") and t.get("percent") == 100 for t in (svc.get("traffic") or []) if isinstance(t, dict))


def failed_turns(d: dict) -> int:
    """Measured turns that did not end ok (model_unavailable, timeouts, a lab error)."""
    t = d.get("turns") or {}
    return int(t.get("n_all") or 0) - int(t.get("n_ok") or 0)


def other_errors_n(d: dict) -> int:
    """Non-200 model calls that were not parser errors (other_errors)."""
    return sum(int(v or 0) for v in (d.get("other_errors") or {}).values())


def compare(cand: dict, ref: dict, *, tolerance: float = 0.0, min_turns: int = MIN_OK_TURNS,
            reference_digest: str | None = R7_DIGEST) -> dict:
    """The verdict (module docstring): PASS / FAIL / INCOMPLETE, the checks behind it, and notes. INCOMPLETE unless
    the two measures are comparable AND measured what the switch is about: the candidate on the chocolate-muffin
    profile, the reference on the blueberry-muffin one serving `reference_digest` (r7; None skips that check), two
    different services, each described image the one that answered (traffic 100 % on the latest revision, unchanged
    during the measure). FAIL when the candidate's per-turn p50 is worse than the reference's, it drew a parser error,
    or it ended more turns failed, or drew more non-parser call errors, than the reference."""
    why_incomplete: list[str] = []
    for name, d in (("candidate", cand), ("reference", ref)):
        if d.get("tool") != "cm_l4check":
            why_incomplete.append(f"{name}: not a cm_l4check measure.json")
            continue
        if not d.get("complete"):
            why_incomplete.append(f"{name}: did not finish ({d.get('failed') or 'no reason recorded'})")
            continue
        if not (d.get("served") or {}).get("ok"):
            why_incomplete.append(f"{name}: serving asserts failed ({'; '.join((d.get('served') or {}).get('problems') or [])})")
        if int((d.get("turns") or {}).get("n_ok") or 0) < max(1, min_turns):
            why_incomplete.append(f"{name}: {(d.get('turns') or {}).get('n_ok')} ok turns, fewer than {max(1, min_turns)}")
    if not why_incomplete:
        if cand.get("tool_version") != ref.get("tool_version"):
            why_incomplete.append(f"tool versions differ ({cand.get('tool_version')} vs {ref.get('tool_version')})")
        if cand.get("rows_sha256") != ref.get("rows_sha256"):
            why_incomplete.append("the two measures replayed different rows (rows_sha256): measure both the same way")
        if cand.get("day") != ref.get("day"):
            why_incomplete.append(f"measured on different days ({cand.get('day')} vs {ref.get('day')}): re-measure the "
                                  "reference today")
        if cand.get("gap_s") != ref.get("gap_s") or cand.get("call_timeout") != ref.get("call_timeout"):
            why_incomplete.append("the two measures ran with different --gap-s / --call-timeout")
        if cand.get("engine_fingerprint") != ref.get("engine_fingerprint"):
            why_incomplete.append("the two measures ran different engines (engine_fingerprint): pass the same --engine")
    # what was measured must be what the switch is about: the profiles, two services, r7 as the reference, and each
    # recorded image the one that answered
    if cand.get("profile") != "chocolate-muffin":
        why_incomplete.append(f"the candidate ran the {cand.get('profile')} profile, not chocolate-muffin")
    if ref.get("profile") != "blueberry-muffin":
        why_incomplete.append(f"the reference ran the {ref.get('profile')} profile, not blueberry-muffin")
    if cand.get("url") and cand.get("url") == ref.get("url"):
        why_incomplete.append("the candidate and the reference are the same URL")
    ref_svc = ref.get("service") or {}
    ref_img = str(ref_svc.get("image") or "")
    if reference_digest:
        if not ref_img:
            why_incomplete.append("the reference's image is unknown (measured by --ref-url, or `gcloud run services "
                                  "describe` failed): measure it by service name (--ref-service) so the image is recorded")
        elif reference_digest not in ref_img:
            why_incomplete.append(f"the reference service serves {ref_img}, not r7 ({reference_digest[:19]}…)")
    notes: list[str] = []
    for name, d in (("candidate", cand), ("reference", ref)):
        svc, after = d.get("service") or {}, d.get("service_after")
        if svc.get("image") and not serves_latest(svc):
            why_incomplete.append(f"the {name} service's traffic is not 100 % on its latest revision "
                                  f"({svc.get('traffic')}): the image recorded may not be the one that answered")
        if svc.get("image") and isinstance(after, dict) and (after.get("image"), after.get("traffic")) != \
                (svc.get("image"), svc.get("traffic")):
            why_incomplete.append(f"the {name} service changed during the measure ({svc.get('image')} → "
                                  f"{after.get('image')}): something else re-pointed it; measure again")
    if not (cand.get("service") or {}).get("image"):
        notes.append("the candidate's image is unknown (measured by --cand-url): `make inhouse-switch` needs the "
                     "verdict to name it — measure by service name (--cand-service)")
    checks: list[dict] = []
    verdict = "INCOMPLETE"
    if not why_incomplete:
        cp50, rp50 = cand["turns"]["p50"], ref["turns"]["p50"]
        bound = round(rp50 * (1.0 + tolerance), 1)
        checks.append({"check": "per-turn p50 no worse than the reference's", "candidate_ms": cp50, "reference_ms": rp50,
                       "bound_ms": bound, "ok": cp50 <= bound})
        pe = int(cand["parser_errors"]["n"])
        checks.append({"check": "0 parser errors", "candidate": pe, "ok": pe == 0})
        cf, rf = failed_turns(cand), failed_turns(ref)
        checks.append({"check": "no more failed turns than the reference", "candidate": cf, "reference": rf,
                       "candidate_errors": (cand.get("turns") or {}).get("errors"), "ok": cf <= rf})
        co, ro = other_errors_n(cand), other_errors_n(ref)
        checks.append({"check": "no more non-parser call errors than the reference", "candidate": co, "reference": ro,
                       "candidate_errors": cand.get("other_errors"), "ok": co <= ro})
        verdict = "PASS" if all(c["ok"] for c in checks) else "FAIL"
    info = {}
    for name, d in (("candidate", cand), ("reference", ref)):
        if d.get("complete"):
            info[name] = {"label": d.get("label"), "url": d.get("url"), "image": (d.get("service") or {}).get("image"),
                          "turn_p50": d["turns"]["p50"], "turn_p90": d["turns"]["p90"], "turns_ok": d["turns"]["n_ok"],
                          "turns": d["turns"]["n_all"], "turn_errors": d["turns"]["errors"],
                          "call_p50": d["calls"]["p50"], "call_p90": d["calls"]["p90"], "calls": d["calls"]["n_all"],
                          "calls_per_turn": d["turns"]["calls_per_turn_mean"], "output_tokens_p50": d["calls"]["output_tokens"]["p50"],
                          "tool_calls_valid": f"{d['tool_calls']['valid']}/{d['tool_calls']['n']}",
                          "parser_errors": d["parser_errors"]["n"], "other_errors": d.get("other_errors"),
                          "exam_pass": f"{d['exam']['pass']}/{d['exam']['rows']}", "primes": d["primes"]["n"]}
    return {"verdict": verdict, "checks": checks, "incomplete": why_incomplete, "notes": notes, "tolerance": tolerance,
            "min_turns": min_turns, "reference_digest": reference_digest, "at": utc_now(), "summary": info}


def verdict_md(v: dict) -> str:
    lines = [f"# chocolate-muffin L4 check: **{v['verdict']}**", ""]
    for c in v["checks"]:
        lines.append(f"- {'ok' if c['ok'] else 'FAILED'}: {c['check']} — "
                     + ", ".join(f"{k} {c[k]}" for k in c if k not in ("check", "ok")))
    for w in v["incomplete"]:
        lines.append(f"- INCOMPLETE: {w}")
    for n in v["notes"]:
        lines.append(f"- note: {n}")
    s = v.get("summary") or {}
    if s:
        cols = ["turn_p50", "turn_p90", "turns_ok", "call_p50", "call_p90", "calls", "calls_per_turn", "output_tokens_p50",
                "tool_calls_valid", "parser_errors", "turn_errors", "other_errors", "exam_pass", "primes"]
        lines += ["", "| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for name in ("candidate", "reference"):
            if name in s:
                lines.append(f"| {name} ({s[name]['label']}) | " + " | ".join(str(s[name].get(c)) for c in cols) + " |")
        lines.append("")
        lines.append("ms; per turn = POST /v1/turns to the end of the turn's last stream (ok turns); per call = each "
                     "/v1/messages at the proxy (primes apart).")
    return "\n".join(lines) + "\n"


def cmd_compare(a) -> int:
    cand = json.loads(Path(a.candidate).read_text(encoding="utf-8"))
    ref = json.loads(Path(a.reference).read_text(encoding="utf-8"))
    v = compare(cand, ref, tolerance=a.tolerance, min_turns=a.min_turns, reference_digest=a.reference_digest or None)
    md = verdict_md(v)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "verdict.json").write_text(json.dumps(v, indent=1) + "\n", encoding="utf-8")
        (Path(a.out) / "verdict.md").write_text(md, encoding="utf-8")
    print(md, end="")
    return {"PASS": EXIT_PASS, "FAIL": EXIT_FAIL}.get(v["verdict"], EXIT_INCOMPLETE)


def gcloud_url(service: str) -> str:
    d = describe_service(service)
    if not d.get("url"):
        raise L4Error(f"no URL for {service} (gcloud run services describe {service} --region {GPU_REGION}): pass it")
    return str(d["url"])


@contextlib.contextmanager
def pipeline_locks(bucket_url: str, run_id: str, *, names: tuple[str, ...] = CHECK_LOCKS, ttl_s: float = LOCK_TTL_S):
    """The pipeline's own locks (state/locks/<name>.json in its bucket, gcsio's protocol), held for the whole check:
    `retrain` (no round may start and re-point sous-chef-model-cand) and `gpu-eval` (no drift check or round eval
    shares or re-points sous-chef-model-eval). A lock another run holds stops the check (L4Error, exit 3); a killed
    check leaves them to expire after ttl_s (the drift workflow breaks an expired gpu-eval lock; an expired retrain
    lock is taken over by the next acquire)."""
    from souschef_lab import gcsio
    b = gcsio.bucket(bucket_url)
    held: list[str] = []
    try:
        for name in names:
            try:
                b.acquire(name, run_id, ttl_s, holder=f"cm_l4check:{run_id}")
            except gcsio.LockHeld as e:
                raise L4Error(f"{b.lock_key(name)} is held ({e}): a {'retrain round' if name == 'retrain' else 'drift check or round eval'} "
                              "is using the GPU services; check again when it has finished") from None
            held.append(name)
            say(f"holding {b.lock_key(name)} as {run_id} (until {ttl_s / 3600:.0f} h at most)")
        yield b
    finally:
        for name in reversed(held):
            try:
                b.release(name, run_id)
            except Exception as e:  # noqa: BLE001 — a lock left behind expires; never hide the verdict for it
                say(f"could not release {b.lock_key(name)} ({e}); it expires on its own")


def cmd_check(a) -> int:
    out = Path(a.out)
    ref = a.ref_url or gcloud_url(a.ref_service)
    cand = a.cand_url or gcloud_url(a.cand_service)
    common = dict(auth=a.auth, impersonate=a.impersonate, pick=a.pick, warmup=a.warmup, limit=a.limit, gap_s=a.gap_s,
                  call_timeout=a.call_timeout, service_bin=a.service_bin, engine=a.engine, fake_engine=a.fake_engine,
                  ignore_served=False, keep_work=a.keep_work, set=None)
    run_id = "l4-" + utc_now().replace("-", "").replace(":", "").lower()
    with contextlib.nullcontext() if a.no_lock else pipeline_locks(a.bucket, run_id):
        rc_ref = measure(argparse.Namespace(url=ref, profile="blueberry-muffin", label="reference", out=str(out / "reference"),
                                            service_name=a.ref_service if not a.ref_url else None, **common))
        rc_cand = measure(argparse.Namespace(url=cand, profile="chocolate-muffin", label="candidate", out=str(out / "candidate"),
                                             service_name=a.cand_service if not a.cand_url else None, **common))
    say(f"measures: reference rc {rc_ref}, candidate rc {rc_cand}")
    return cmd_compare(argparse.Namespace(candidate=str(out / "candidate" / "measure.json"),
                                          reference=str(out / "reference" / "measure.json"), out=str(out),
                                          tolerance=a.tolerance, min_turns=a.min_turns, reference_digest=a.reference_digest))


def cmd_rows(a) -> int:
    rows, warm = pick_rows(parse_pick(a.pick), a.warmup)
    for r in rows:
        print(f"{row_key(r)}\t{len(r.get('history') or [])} history\t{r.get('request')}")
    print(f"# {len(rows)} rows, {sum(1 + len(r.get('history') or []) for r in rows)} turns; warm-up "
          f"{', '.join(str(r.get('id')) for r in warm)}; rows_sha256 {rows_fingerprint(rows)}")
    return 0


# --- cli ---------------------------------------------------------------------------------------------------------------------

def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--auth", choices=("gcloud", "metadata", "none"), default="gcloud")
    p.add_argument("--impersonate", help="mint the ID token for this service account (gcloud --impersonate-service-account)")
    p.add_argument("--pick", help=f"rows per set, e.g. real=14,paraphrase=8 (default {DEFAULT_PICK})")
    p.add_argument("--warmup", type=int, default=WARMUP_N)
    p.add_argument("--limit", type=int, help="the first N picked rows only (a rehearsal; compare needs both alike)")
    p.add_argument("--gap-s", type=float, default=DEFAULT_GAP_S, help="quiet seconds before each turn")
    p.add_argument("--call-timeout", default=CALL_TIMEOUT, help="the service's MODEL_CALL_TIMEOUT (int: 60s)")
    p.add_argument("--service-bin", help="the sous-chef binary (default $SOUS_CHEF_BIN); build it from this tree")
    p.add_argument("--engine", help="a cookie-cutter checkout/snapshot, headless.mjs, or bucket[:<launcher sha>] (the pipeline's "
                                    "vendored snapshot, read-only); default $SOUS_CHEF_ENGINE / $COOKIE_CUTTER_DIR")
    p.add_argument("--fake-engine", action="store_true", help="the lab's DRY ManifestEngine instead of headless.mjs (tests)")
    p.add_argument("--keep-work", action="store_true", help="keep the harness's work folder (Postgres, logs)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cm_l4check", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="measure one model service")
    m.add_argument("--url", required=True)
    m.add_argument("--profile", choices=sorted(PROFILES), required=True)
    m.add_argument("--label")
    m.add_argument("--out", required=True)
    m.add_argument("--service-name", help="the Cloud Run service behind --url, described read-only for the record")
    m.add_argument("--set", action="append", metavar="KEY=VALUE", help="override one route setting of the profile")
    m.add_argument("--ignore-served", action="store_true", help="replay even when a serving assert failed")
    _common(m)
    c = sub.add_parser("compare", help="the verdict of two measures")
    c.add_argument("--candidate", required=True)
    c.add_argument("--reference", required=True)
    c.add_argument("--out")
    c.add_argument("--tolerance", type=float, default=0.0, help="the candidate's per-turn p50 may exceed the reference's by this fraction")
    c.add_argument("--min-turns", type=int, default=MIN_OK_TURNS)
    c.add_argument("--reference-digest", default=R7_DIGEST, help="the image the reference must serve (default r7's; '' skips)")
    k = sub.add_parser("check", help="measure the reference (r7 on the eval service) and the candidate, then compare")
    k.add_argument("--cand-url", help="the candidate's URL (default: --cand-service's; its image is then not recorded)")
    k.add_argument("--ref-url", "--live-url", dest="ref_url",
                   help="the reference's URL (default: --ref-service's); its image is then unknown and the verdict INCOMPLETE")
    k.add_argument("--cand-service", default=CAND_SERVICE)
    k.add_argument("--ref-service", "--live-service", dest="ref_service", default=REF_SERVICE,
                   help=f"the Cloud Run service measured as the reference (default {REF_SERVICE}: the pipeline's eval "
                        f"service, r7's image and production's shape, no merchant traffic; {LIVE_SERVICE} is production)")
    k.add_argument("--bucket", default=BUCKET, help="the pipeline bucket whose retrain + gpu-eval locks the check holds")
    k.add_argument("--no-lock", action="store_true", help="hold no pipeline lock (nothing else may use the GPU services)")
    k.add_argument("--out", required=True)
    k.add_argument("--tolerance", type=float, default=0.0)
    k.add_argument("--min-turns", type=int, default=MIN_OK_TURNS)
    k.add_argument("--reference-digest", default=R7_DIGEST, help="the image the reference must serve (default r7's; '' skips)")
    _common(k)
    r = sub.add_parser("rows", help="print the rows a measure replays")
    r.add_argument("--pick")
    r.add_argument("--warmup", type=int, default=WARMUP_N)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        lab_import_check()
        if a.cmd == "measure":
            return measure(a)
        if a.cmd == "compare":
            return cmd_compare(a)
        if a.cmd == "check":
            return cmd_check(a)
        return cmd_rows(a)
    except (ValueError, L4Error) as e:
        say(str(e))
        return EXIT_USAGE if isinstance(e, ValueError) else EXIT_INCOMPLETE


if __name__ == "__main__":
    sys.exit(main())
