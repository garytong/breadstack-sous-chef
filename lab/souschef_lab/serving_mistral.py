"""serving_mistral — a training row rendered exactly as Ollama 0.34.4 renders
the same conversation for a Ministral 3 student (the Mistral arm; the Qwen
counterpart is serving.py, which this module never changes).

Ollama 0.34.4 has a `ministral` PARSER but no `ministral` RENDERER, and the
library's Modelfile template uses the pre-v13 layout, so the arm serves
through a Go TEMPLATE of its own in Mistral's v13 layout (mistral-common
InstructTokenizerV13; the HF chat_template.jinja of
mistralai/Ministral-3-8B-Instruct-2512-BF16 @ f6fae97):

    [SYSTEM_PROMPT]…[/SYSTEM_PROMPT][AVAILABLE_TOOLS][…][/AVAILABLE_TOOLS][INST]…[/INST]
    <reply>[TOOL_CALLS]name[ARGS]{…}[TOOL_CALLS]…</s>[TOOL_RESULTS]…[/TOOL_RESULTS]…

With a TEMPLATE (no RENDERER) Ollama's server does this (file:line at v0.34.4):

1. `anthropic.FromMessagesRequest` converts the /v1/messages body as for
   any model (system blocks joined into one system message, tool_result
   blocks into `tool` messages before the user's own text, tool_use blocks
   into the assistant's ToolCalls).
2. `template.Execute` (`template/template.go:256-284`) first COLLATES the
   messages (`:357-374`): consecutive messages of one role are merged —
   content joined with "\\n\\n" — except tool messages; the merged message
   keeps the FIRST message's tool calls only (Go appends content, never
   ToolCalls). Then the template sees `.Messages` (collated) and `.Tools`.
3. `{{ $.Tools }}` prints `templateTools.String()` = Go's `json.Marshal`
   of `templateTool` (`:375-433`): {"type","function":{"name",
   "description" (always, no omitempty),"parameters":{"type","$defs"?,
   "items"?,"required"? (omitempty),"properties": a Go MAP (sorted keys;
   null when absent)}}}, each property an `api.ToolProperty`
   (serving._property's field order), compact, `<>&` as \\u003c/\\u003e/
   \\u0026. Keys with no struct field (additionalProperties, strict) drop.
4. `{{ .Function.Arguments }}` prints `templateArgs.String()`: "{}" for no
   arguments, else `json.Marshal` of a map[string]any — compact, sorted
   keys, HTML-escaped, numbers as float64 (1000000 stays 1000000).
5. No content is trimmed; no `<s>` is written — llama-server adds BOS
   (GGUF add_bos_token) and the HF tokenizer adds it too, so a literal
   `<s>` in the text would be doubled.
6. `PARSER ministral` (`model/parsers/ministral.go`) reads the reply:
   text up to `[TOOL_CALLS]` (trailing whitespace cut), the name up to
   `[ARGS]`, then one balanced JSON object (findJSONEnd); an unknown name or
   invalid JSON is an error (the service's model call fails: model_error).

The training text (`render_training`) is the whole conversation with every
assistant turn closed by `</s>`. Unlike qwen3.5 there is no think residue:
every served prompt is an exact prefix of the training text, and each
assistant segment (content + calls + `</s>`) is the bytes Ollama re-renders
for that turn in the next request.

`style="python"` writes the same layout with Python's json.dumps (the HF
template's `tojson`), which tests/test_serving_mistral.py holds against
`transformers.apply_chat_template` goldens (fixtures/mistral/hf-golden.json)
— proof that the layout is the model's own. The served and trained form is
always `style="go"`.

The parity fixture (fixtures/parity/served-ministral.json) is Ollama's own
render of the real captures through parity/ in template mode (`go run .
-parser ministral -modelfile base-modelfiles/ministral-3-8b.modelfile`):
`python -m souschef_lab.serving_mistral --write` / `--check`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .serving import (CAPTURES, HARNESS, LAB_DIR, _obj, _property, captures, decode_arguments, first_difference,
                      go_json, go_string, read_jsonl, sha)
from . import serving

FAMILY = "mistral"
PARSER = "ministral"
RENDERER = None
OLLAMA_VERSION = serving.OLLAMA_VERSION  # the model image's Ollama (0.34.4): the same binary serves both families
DESCRIPTION = (f"serving-parity: Ollama {OLLAMA_VERSION} Go TEMPLATE (Mistral v13) + PARSER {PARSER}, "
               "thinking disabled")
BASE_MODELFILE = LAB_DIR / "base-modelfiles" / "ministral-3-8b.modelfile"
FIXTURE = LAB_DIR / "fixtures" / "parity" / "served-ministral.json"
HF_GOLDEN = LAB_DIR / "fixtures" / "mistral" / "hf-golden.json"

# Measured on the pinned tokenizer (mistralai/Ministral-3-8B-Instruct-2512-BF16 @ f6fae97, Tekken v13): every
# marker the render writes is ONE control token; train.served_markers checks each maps to exactly this id.
SPECIAL_IDS = {"<s>": 1, "</s>": 2, "[INST]": 3, "[/INST]": 4, "[AVAILABLE_TOOLS]": 5, "[/AVAILABLE_TOOLS]": 6,
               "[TOOL_RESULTS]": 7, "[/TOOL_RESULTS]": 8, "[TOOL_CALLS]": 9, "[SYSTEM_PROMPT]": 17,
               "[/SYSTEM_PROMPT]": 18, "[ARGS]": 32}
SPECIAL_TOKENS = tuple(SPECIAL_IDS)
BOS, END_OF_TURN = "<s>", "</s>"
SYS_OPEN, SYS_CLOSE = "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]"
TOOLS_OPEN, TOOLS_CLOSE = "[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]"
INST_OPEN, INST_CLOSE = "[INST]", "[/INST]"
RESULTS_OPEN, RESULTS_CLOSE = "[TOOL_RESULTS]", "[/TOOL_RESULTS]"
CALLS, ARGS = "[TOOL_CALLS]", "[ARGS]"

# The Modelfile's TEMPLATE, byte for byte (no newline before the closing quotes: one would end every prompt with
# "\n"). base-modelfiles/ministral-3-8b.modelfile must carry exactly this (tests/test_serving_mistral.py).
TEMPLATE = """{{- $tools := false }}
{{- range $i, $m := .Messages }}
{{- if eq $m.Role "system" }}[SYSTEM_PROMPT]{{ $m.Content }}[/SYSTEM_PROMPT]
{{- else if eq $m.Role "user" }}
{{- if and $.Tools (not $tools) }}[AVAILABLE_TOOLS]{{ $.Tools }}[/AVAILABLE_TOOLS]{{ $tools = true }}{{ end }}[INST]{{ $m.Content }}[/INST]
{{- else if eq $m.Role "assistant" }}{{ $m.Content }}
{{- range $m.ToolCalls }}[TOOL_CALLS]{{ .Function.Name }}[ARGS]{{ .Function.Arguments }}{{ end }}
{{- if or $m.ToolCalls (ne (len (slice $.Messages $i)) 1) }}</s>{{ end }}
{{- else if eq $m.Role "tool" }}[TOOL_RESULTS]{{ $m.Content }}[/TOOL_RESULTS]
{{- end }}
{{- end }}"""
# top_k / top_p as the Qwen base Modelfile (the samplers differ between arms only by the model); temperature 0.15 is
# Mistral's recommendation, kept for the record — `make model-stage MODEL_TEMPERATURE=0.3` writes the route's 0.3.
PARAMETERS = (("temperature", "0.15"), ("top_k", "20"), ("top_p", "0.95"))


def modelfile_text() -> str:
    """The base Modelfile's content (base-modelfiles/ministral-3-8b.modelfile): TEMPLATE + PARSER + PARAMETERs."""
    head = ("# Ministral 3 8B Instruct 2512, Mistral v13 layout (mistral-common InstructTokenizerV13, HF "
            "chat_template.jinja\n# @f6fae97): system first; tools ONCE before the first [INST]; [TOOL_CALLS] before "
            "EACH call; </s> after every\n# assistant turn except an open prefill. No <s> (llama-server adds BOS). "
            "Tools/args are Go json.Marshal (compact,\n# sorted keys, HTML-escaped): souschef_lab/serving_mistral.py "
            "replicates these bytes; fixtures/parity/served-ministral.json.\n")
    return (head + 'TEMPLATE """' + TEMPLATE + '"""\n' + f"PARSER {PARSER}\n"
            + "".join(f"PARAMETER {k} {v}\n" for k, v in PARAMETERS))


def template_of(modelfile: str) -> str | None:
    """The TEMPLATE value of a Modelfile's text, as Ollama reads it (triple quotes: verbatim between them)."""
    from .export import parse_modelfile  # export imports nothing heavy at module level
    for ins in parse_modelfile(modelfile):
        if ins.keyword == "TEMPLATE":
            v = ins.raw[len("TEMPLATE"):].strip()
            if v.startswith('"""') and v.endswith('"""') and len(v) >= 6:
                return v[3:-3]
            return v.strip('"')
    return None


def template_sha256(template: str = TEMPLATE) -> str:
    return sha(template)


# --- Ollama's template.collate ------------------------------------------------------------------------------------------

def collate(messages: list[dict], style: str = "go") -> list[dict]:
    """Consecutive messages of one role merged (content joined by "\\n\\n"), tool messages never. go
    (template.go:357-374): system messages merge too, and the merged message keeps the FIRST one's tool calls (Go
    appends content only). python (the HF template): system messages never merge; tool calls are concatenated."""
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if out and out[-1].get("role") == role and role != "tool" and (style == "go" or role != "system"):
            prev = out[-1]
            prev["content"] = str(prev.get("content") or "") + "\n\n" + str(m.get("content") or "")
            if style == "python" and m.get("tool_calls"):
                prev["tool_calls"] = list(prev.get("tool_calls") or []) + list(m["tool_calls"])
        else:
            out.append(dict(m))
    return out


def merged_assistants(messages: list[dict]) -> int:
    """How many assistant messages Ollama's collate merges into the one before them."""
    return sum(1 for a, b in zip(messages, messages[1:]) if a.get("role") == b.get("role") == "assistant")


# --- JSON as the template prints it ------------------------------------------------------------------------------------

def _template_tool(tool: dict) -> str:
    """One api.Tool as templateTool marshals (template.go:409-433): description always present; parameters
    {type, $defs?, items?, required?, properties: a Go map → sorted keys, null when absent}."""
    fn = tool.get("function") or {}
    s = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
    par = [("type", go_string(s.get("type") if isinstance(s.get("type"), str) else ""))]
    if s.get("$defs") is not None:
        par.append(("$defs", go_json(s["$defs"])))
    if s.get("items") is not None:
        par.append(("items", go_json(s["items"])))
    if s.get("required"):
        par.append(("required", go_json(s["required"])))
    props = s.get("properties")
    par.append(("properties", _obj([(k, _property(props[k])) for k in sorted(props)]) if isinstance(props, dict)
                else "null"))
    f = [("name", go_string(str(fn.get("name") or ""))), ("description", go_string(str(fn.get("description") or ""))),
         ("parameters", _obj(par))]
    return _obj([("type", go_string("function")), ("function", _obj(f))])


def tools_json(tools: list[dict] | None, style: str = "go") -> str:
    """The [AVAILABLE_TOOLS] body: go = templateTools.String(); python = json.dumps (the HF template's tojson)."""
    if style == "python":
        return json.dumps(list(tools or []), ensure_ascii=False)
    return "[" + ",".join(_template_tool(t) for t in tools or []) + "]"


def args_json(arguments: Any, style: str = "go") -> str:
    """A call's arguments after [ARGS]: go = templateArgs.String() (compact, sorted, HTML-escaped; {} when empty);
    python = the HF template (a string verbatim, "" → {}, anything else json.dumps)."""
    if style == "python":
        if isinstance(arguments, str):
            return arguments if arguments != "" else "{}"
        return json.dumps(arguments, ensure_ascii=False)
    return go_json(decode_arguments(arguments))


# --- The render ----------------------------------------------------------------------------------------------------------

def _check_style(style: str) -> None:
    if style not in ("go", "python"):
        raise ValueError(f"style {style!r}: go or python")


def render_conversation(messages: list[dict], tools: list[dict] | None, *, style: str = "go", close_last: bool = True,
                        bos: bool = False, history_turns: int = 0) -> tuple[str, list[tuple[int, int]]]:
    """(text, spans): the TEMPLATE applied to the collated messages; spans are [a, b) character ranges of each
    assistant segment (content, calls, and its </s> when written). close_last=False is the served form: a trailing
    assistant message without calls is an open prefill (no </s>). bos writes "<s>" first (only the HF golden: the
    served and trained text never holds it). history_turns changes no byte (no think residue)."""
    _check_style(style)
    del history_turns
    msgs = collate(messages, style)
    tj = tools_json(tools, style) if tools else ""
    sb: list[str] = [BOS] if bos else []
    n = len(sb[0]) if bos else 0
    spans: list[tuple[int, int]] = []
    tools_done = False

    def put(s: str) -> None:
        nonlocal n
        sb.append(s)
        n += len(s)

    for i, m in enumerate(msgs):
        role = m.get("role")
        content = m.get("content")
        content = content if isinstance(content, str) else ("" if content is None else str(content))
        if role == "system":
            put(SYS_OPEN + content + SYS_CLOSE)
        elif role == "user":
            if tools and not tools_done:
                put(TOOLS_OPEN + tj + TOOLS_CLOSE)
                tools_done = True
            put(INST_OPEN + content + INST_CLOSE)
        elif role == "assistant":
            a = n
            put(content)
            calls = m.get("tool_calls") or []
            for c in calls:
                fn = c.get("function") or {}
                put(CALLS + str(fn.get("name") or "") + ARGS + args_json(fn.get("arguments"), style))
            if calls or i != len(msgs) - 1 or close_last:
                put(END_OF_TURN)
            spans.append((a, n))
        elif role == "tool":
            put(RESULTS_OPEN + content + RESULTS_CLOSE)
    return "".join(sb), spans


def render_prompt(messages: list[dict], tools: list[dict] | None) -> str:
    """The prompt Ollama renders for the NEXT model call of this conversation (a trailing assistant message is an
    open prefill) — checked byte for byte against the parity fixture."""
    return render_conversation(messages, tools, close_last=False)[0]


def render_training(messages: list[dict], tools: list[dict] | None, history_turns: int = 0) -> str:
    """A row's training text: every assistant turn closed by </s>. A follow-up's history renders the same bytes."""
    return render_conversation(messages, tools, close_last=True, history_turns=history_turns)[0]


def assistant_segments(messages: list[dict], tools: list[dict] | None) -> list[str]:
    """The trained segments of the training text, in order (one per collated assistant message)."""
    text, spans = render_conversation(messages, tools, close_last=True)
    return [text[a:b] for a, b in spans]


def control_ids(text: str, *, other: int = -1) -> tuple[list[int], list[tuple[int, int]]]:
    """`text` as the pinned tokenizer reads it with add_special_tokens=True, as far as the control tokens go: BOS
    first, then every control-token string (SPECIAL_TOKENS) as its id (SPECIAL_IDS) wherever it stands — the tokenizer
    matches them in the raw text, so a merchant's literal "<s>" or "</s>" IS the control token — and one `other` id for
    each run of anything else; with each id's [start, end) characters in `text` (BOS: (0, 0)). trainkit.mistral_spans
    finds the same spans in these ids as in the real ones (tests/test_mistral_masking.py checks it against the
    tokenizer), which lets the dataset stage catch, on CPU, a row the training mask would stop on, or would train on
    other text than its assistant segments, on the GPU (train.mask_problem)."""
    import re
    pat = re.compile("|".join(re.escape(t) for t in sorted(SPECIAL_TOKENS, key=len, reverse=True)))
    ids, offs, pos = [SPECIAL_IDS[BOS]], [(0, 0)], 0
    for m in pat.finditer(text):
        if m.start() > pos:
            ids.append(other)
            offs.append((pos, m.start()))
        ids.append(SPECIAL_IDS[m.group(0)])
        offs.append((m.start(), m.end()))
        pos = m.end()
    if pos < len(text):
        ids.append(other)
        offs.append((pos, len(text)))
    return ids, offs


_NOT_A_REPLY = (RESULTS_OPEN, INST_OPEN, TOOLS_OPEN, SYS_OPEN)


def generation_points(text: str) -> list[int]:
    """Offsets where a model call's completion begins: after each [/INST], and after each [/TOOL_RESULTS], unless the
    next bytes open another result, user turn, tools block or system turn (more context, not a reply)."""
    out = []
    for marker in (INST_CLOSE, RESULTS_CLOSE):
        pos = 0
        while (a := text.find(marker, pos)) >= 0:
            b = a + len(marker)
            if not text.startswith(_NOT_A_REPLY, b):
                out.append(b)
            pos = b
    return sorted(out)


# --- Reading a reply back (Ollama's MinistralParser) -----------------------------------------------------------------------

def find_json_end(s: str) -> int:
    """model/parsers/ministral.go findJSONEnd: the index of the bracket closing the first balanced JSON value, -1
    while it is incomplete."""
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return i
    return -1


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """The tool calls in a rendered (or generated) assistant segment, as the ministral parser reads them: after each
    [TOOL_CALLS] the name up to [ARGS], then one balanced JSON object; (name, arguments) with the object decoded (key
    order does not count). Stops at the first call the parser would refuse (incomplete or not an object)."""
    calls: list[tuple[str, dict]] = []
    rest = text or ""
    while (k := rest.find(CALLS)) >= 0:
        rest = rest[k + len(CALLS):]
        j = rest.find(ARGS)
        if j < 0:
            break
        name, rest = rest[:j], rest[j + len(ARGS):]
        end = find_json_end(rest)
        if end < 0:
            break
        try:
            args = json.loads(rest[:end + 1])
        except ValueError:
            break
        if not isinstance(args, dict):
            break
        calls.append((name, args))
        rest = rest[end + 1:]
    return calls


# --- Rows the served form cannot carry -------------------------------------------------------------------------------------

_FLOAT64_EXACT = 2 ** 53


def _not_float64(v: Any) -> str:
    """Why a decoded argument value would not survive Go's float64 (Ollama decodes arguments into `any`), "" if it
    would."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return ""
    if isinstance(v, int):
        return f"{v} is beyond float64's exact integers" if abs(v) > _FLOAT64_EXACT else ""
    if isinstance(v, float):
        return "" if math.isfinite(v) else f"{v} is not a finite number"
    if isinstance(v, list):
        return next((w for w in map(_not_float64, v) if w), "")
    if isinstance(v, dict):
        return next((w for w in map(_not_float64, v.values()) if w), "")
    return ""


def unservable_mistral(messages: list[dict]) -> list[str]:
    """The Mistral rule alone: an argument Go's float64 changes (an integer beyond 2^53, NaN/Infinity), and
    consecutive assistant messages whose second one carries tool calls (Ollama's collate keeps the first message's
    calls only, so the served history would not hold them)."""
    out = []
    for m in messages:
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            for k, v in decode_arguments(fn.get("arguments")).items():
                if why := _not_float64(v):
                    out.append(f"{fn.get('name', '?')}.{k}: {why}")
    for a, b in zip(messages, messages[1:]):
        if a.get("role") == b.get("role") == "assistant" and b.get("tool_calls"):
            out.append("consecutive assistant messages: Ollama's collate drops the second one's tool calls")
    return out


def unservable(messages: list[dict]) -> list[str]:
    """serving.unservable (the qwen3.5 echo rule) ∪ the Mistral rule: the union keeps the Mistral arm's trained rows
    the same set as the Qwen arms' (the comparison is the model, not the rows). train.py reports both counts."""
    return serving.unservable(messages) + unservable_mistral(messages)


# --- The served-prompt smoke: one /v1/messages request whose prompt size proves the served render ---------------------------
# Ollama 0.34.4 logs no rendered prompt. Its answer's usage.input_tokens (the prompt it evaluated) does show it: on a
# FRESH server (nothing cached: make it the first request after `ollama create`) the smoke below must cost exactly
# SMOKE_PROMPT_TOKENS[family] — the template rendered as render_prompt does, ONE BOS, every control marker one token.
# Measured offline 2026-10-16 on the pinned tokenizers (mistral: render_prompt + BOS, Ministral-3-8B-Instruct-2512-BF16
# @ f6fae97; qwen: serving.render_prompt, unsloth/Qwen3.5-9B, no BOS); tests/test_serving_mistral.py re-measures when
# the tokenizers are at hand. The native diagnostic arm (the GGUF's own template) has no expected count.

SMOKE_TOOL = {"name": "echo", "description": "Echo the text back.",
              "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}
SMOKE_REQUEST = {"model": "sous-chef", "max_tokens": 64, "system": "You are a test.", "tools": [SMOKE_TOOL],
                 "messages": [{"role": "user", "content": "Call echo with the text hi."}],
                 "thinking": {"type": "disabled"}}
SMOKE_PROMPT_TOKENS = {"mistral": 61, "qwen": 279}


def smoke_conversation() -> tuple[list[dict], list[dict]]:
    """SMOKE_REQUEST as a row's (messages, tools) — what Ollama's Anthropic conversion makes of it."""
    tool = {"type": "function", "function": {"name": SMOKE_TOOL["name"], "description": SMOKE_TOOL["description"],
                                             "parameters": SMOKE_TOOL["input_schema"]}}
    return ([{"role": "system", "content": SMOKE_REQUEST["system"]},
             {"role": "user", "content": SMOKE_REQUEST["messages"][0]["content"]}], [tool])


# --- The parity fixture: Ollama's own render of the real captures (template mode) --------------------------------------------

def run_harness(bodies: list[dict], harness: Path = HARNESS, modelfile: Path = BASE_MODELFILE) -> list[dict]:
    """Ollama 0.34.4's prompt for each /v1/messages body, through parity/ in template mode (`go run . -parser
    ministral -modelfile <base Modelfile>`). One dict per body: prompt, think, messages, tools."""
    env = dict(os.environ)
    env.setdefault("GOTOOLCHAIN", "auto")
    stdin = "".join(json.dumps(b, ensure_ascii=False) + "\n" for b in bodies)
    cmd = ["go", "run", ".", "-parser", PARSER, "-modelfile", str(Path(modelfile).resolve())]
    p = subprocess.run(cmd, cwd=harness, input=stdin, capture_output=True, text=True, env=env, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"parity/ (template mode) failed:\n{p.stderr.strip()}")
    out = [json.loads(x) for x in p.stdout.splitlines() if x.strip()]
    if len(out) != len(bodies):
        raise RuntimeError(f"parity/ rendered {len(out)} prompts for {len(bodies)} requests")
    return out


def served_fixture(root: Path = CAPTURES, harness: Path = HARNESS, modelfile: Path = BASE_MODELFILE) -> dict:
    """The fixture's content (serving.served_fixture's shape): per capture, Ollama's prompt for each request."""
    from .export import parse_modelfile
    text = Path(modelfile).read_text(encoding="utf-8")
    tmpl = template_of(text)
    if tmpl is None:
        raise RuntimeError(f"{modelfile}: no TEMPLATE")
    parsers = [ins.raw.split(None, 1)[1].strip() for ins in parse_modelfile(text) if ins.keyword == "PARSER"]
    entries = []
    for where in captures(root):
        reqs = read_jsonl(where / "model-requests.jsonl")
        entries.append((where.name, run_harness([r["body"] for r in reqs], harness, modelfile)))
    prompts = [r["prompt"] for _, rendered in entries for r in rendered]
    prefix = os.path.commonprefix(prompts) if prompts else ""
    thinks = sorted({str(r.get("think")) for _, rendered in entries for r in rendered})
    return {
        "what": "Ollama's prompt for every model request of fixtures/prompt/real/*/model-requests.jsonl with the "
                "Mistral arm's base Modelfile (python -m souschef_lab.serving_mistral --write, through parity/ in "
                "template mode)",
        "ollama": OLLAMA_VERSION, "renderer": RENDERER, "parser": parsers[0] if parsers else None, "think": thinks,
        "template_sha256": template_sha256(tmpl), "prefix": prefix,
        "captures": [{"capture": name,
                      "requests": [{"chars": len(r["prompt"]), "sha256": sha(r["prompt"]), "messages": r["messages"],
                                    "tools": r.get("tools"), "tail": r["prompt"][len(prefix):]} for r in rendered]}
                     for name, rendered in entries],
    }


def comparable(fixture: dict) -> list:
    return [fixture["ollama"], fixture["renderer"], fixture["parser"], fixture["think"], fixture["template_sha256"],
            fixture["prefix"],
            [(c["capture"], [(r["chars"], r["sha256"], r["tail"]) for r in c["requests"]]) for c in fixture["captures"]]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Record (or check) Ollama's own render of the real captures with the "
                                             "Mistral arm's TEMPLATE — the parity fixture tests/test_serving_mistral.py "
                                             "holds serving_mistral to.")
    ap.add_argument("--write", action="store_true", help=f"write {FIXTURE.relative_to(LAB_DIR)} (else only check it)")
    ap.add_argument("--check", action="store_true", help="check the fixture (the default)")
    ap.add_argument("--captures", type=Path, default=CAPTURES)
    ap.add_argument("--modelfile", type=Path, default=BASE_MODELFILE)
    ap.add_argument("--harness", type=Path, default=HARNESS, help="the parity Go module (default: lab/parity)")
    ap.add_argument("--print-modelfile", action="store_true",
                    help="print the base Modelfile this module expects (base-modelfiles/ministral-3-8b.modelfile)")
    args = ap.parse_args(argv)
    if args.print_modelfile:
        sys.stdout.write(modelfile_text())
        return 0
    if not args.modelfile.is_file():
        print(f"error: no base Modelfile at {args.modelfile}", file=sys.stderr)
        return 1
    if template_of(args.modelfile.read_text(encoding="utf-8")) != TEMPLATE:
        print(f"error: {args.modelfile}'s TEMPLATE is not serving_mistral.TEMPLATE", file=sys.stderr)
        return 1
    fixture = served_fixture(args.captures, args.harness, args.modelfile)
    calls = sum(len(c["requests"]) for c in fixture["captures"])
    print(f"parity/ (template mode): {len(fixture['captures'])} captures, {calls} requests rendered as Ollama "
          f"{OLLAMA_VERSION} renders them with PARSER {fixture['parser']} (think {fixture['think']})")
    if args.write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {FIXTURE.relative_to(LAB_DIR)}")
        return 0
    try:
        old = json.loads(FIXTURE.read_text(encoding="utf-8"))
    except OSError:
        print(f"error: no fixture at {FIXTURE.relative_to(LAB_DIR)}: run with --write first", file=sys.stderr)
        return 1
    if comparable(old) != comparable(fixture):
        print("error: the fixture is not what the captures give now — regenerate it with --write", file=sys.stderr)
        return 1
    print("The fixture is what the captures give.")
    return 0


__all__ = ["FAMILY", "PARSER", "RENDERER", "DESCRIPTION", "SPECIAL_IDS", "END_OF_TURN", "TEMPLATE", "collate",
           "tools_json", "args_json", "render_conversation", "render_prompt", "render_training", "generation_points",
           "parse_tool_calls", "unservable", "unservable_mistral", "first_difference"]


if __name__ == "__main__":
    sys.exit(main())
