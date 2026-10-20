#!/usr/bin/env python3
"""A stand-in `ollama` binary for the Mistral arm's serving tests (never a model): `serve` answers the endpoints
mxarm.assert_served calls from what `create` recorded; `create` records the Modelfile; `--version` prints 0.34.4.

FAKE_OLLAMA_MODE: ok (default) | cpu (half the model in VRAM) | badtemplate (the template gains a newline) |
                  msg500 (/v1/messages answers 500, a parser error) | msg404 | nocreate (create exits 1) |
                  badprompt (the prompt-size smoke reads one token more than SMOKE_PROMPT_TOKENS) |
                  leak (the tool smoke answers the call as text: `[TOOL_CALLS]echo[ARGS]{…}`) |
                  notool (the tool smoke answers plain text, no call)
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lab/tools
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # lab

MODE = os.environ.get("FAKE_OLLAMA_MODE", "ok")
STATE = Path(os.environ.get("OLLAMA_MODELS", "/tmp")) / "fake-created.json"


def created() -> dict | None:
    return json.loads(STATE.read_text()) if STATE.is_file() else None


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # the GIN-style access line mxarm.log_counts reads
        sys.stdout.write(f'[GIN] 2026/10/17 - 00:00:00 | {args[1] if len(args) > 1 else 200} |  0.1s |  127.0.0.1 | '
                         f'{self.command}     "{self.path}"\n')
        sys.stdout.flush()

    def send(self, code: int, doc) -> None:
        raw = json.dumps(doc).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/api/version":
            return self.send(200, {"version": "0.34.4"})
        if self.path == "/api/ps":
            c = created()
            if not c:
                return self.send(200, {"models": []})
            size = 6_000_000_000
            return self.send(200, {"models": [{"name": "sous-chef:latest", "size": size,
                                               "size_vram": size // 2 if MODE == "cpu" else size, "context_length": 65536}]})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        c = created()
        if self.path == "/api/show":
            if not c or body.get("model") != "sous-chef":
                return self.send(404, {"error": "model not found"})
            import mxarm
            f = mxarm.modelfile_facts(c["modelfile"])
            arch = {"ministral": "mistral3", "qwen3.5": "qwen35"}.get(f["parser"] or "", "llama")
            params = "\n".join(f"{k:<30} {v}" for k, vs in f["parameters"].items() for v in vs)
            t = f["template"] + ("\n" if MODE == "badtemplate" else "")
            # Ollama 0.34.4's shape: no top-level parser / renderer, only PARSER / RENDERER lines in `modelfile`
            return self.send(200, {"template": t, "modelfile": c["modelfile"],
                                   "parameters": params, "capabilities": ["completion", "tools"],
                                   "model_info": {"general.architecture": arch}})
        if self.path == "/api/generate":
            return self.send(200, {"model": "sous-chef", "response": "", "done": True, "done_reason": "load"})
        if self.path == "/api/chat":
            return self.send(200, {"message": {"role": "assistant", "content": "OK"}, "done": True, "eval_count": 2,
                                   "done_reason": "stop"})
        if self.path == "/v1/messages":
            if MODE == "msg404":
                return self.send(404, {"error": "not found"})
            if MODE == "msg500" and body.get("max_tokens") != 1:  # a parser error needs a reply longer than 1 token
                return self.send(500, {"type": "error", "error": {"type": "api_error", "message": "tool 'ech' not found"}})
            from souschef_lab import serving_mistral as SM
            fam = {"ministral": "mistral"}.get(mxarm_facts(c).get("parser") or "", "qwen") if c else "qwen"
            n = SM.SMOKE_PROMPT_TOKENS[fam] + (1 if MODE == "badprompt" else 0)
            if MODE in ("leak", "notool") and body.get("max_tokens") != 1:
                text = '[TOOL_CALLS]echo[ARGS]{"text":"hello"}' if MODE == "leak" else "hello"
                return self.send(200, {"type": "message", "stop_reason": "end_turn", "usage": {"input_tokens": n, "output_tokens": 9},
                                       "content": [{"type": "text", "text": text}]})
            return self.send(200, {"type": "message", "stop_reason": "tool_use", "usage": {"input_tokens": n, "output_tokens": 9},
                                   "content": [{"type": "tool_use", "name": "echo", "input": {"text": "hello"}}]})
        self.send(404, {"error": "not found"})


def mxarm_facts(c: dict) -> dict:
    import mxarm
    return mxarm.modelfile_facts(c["modelfile"])


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in ("--version", "-v"):
        print("Warning: could not connect to a running Ollama instance\nWarning: client version is 0.34.4")
        return 0
    if cmd == "serve":
        host, port = os.environ["OLLAMA_HOST"].rsplit(":", 1)
        ThreadingHTTPServer((host, int(port)), H).serve_forever()
        return 0
    if cmd == "create":
        if MODE == "nocreate":
            print("Error: invalid model file", file=sys.stderr)
            return 1
        mf = Path(sys.argv[sys.argv.index("-f") + 1])
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps({"name": sys.argv[2], "modelfile": mf.read_text(),
                                     "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}))
        print("success")
        return 0
    print(f"fake ollama: unknown command {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
