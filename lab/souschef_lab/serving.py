"""serving — a training row rendered exactly as Ollama renders the same
conversation for the served student (docs/LAB.md §5 "Serving parity").

The service talks to the student through Ollama's Anthropic-compatible
/v1/messages; the export's Modelfile names `RENDERER qwen3.5` / `PARSER
qwen3.5` (export.py), so Ollama's Go renderer writes the prompt — never the
Modelfile's `TEMPLATE {{ .Prompt }}` and never the Hugging Face chat template
train.py used to render with. The two write the same layout but not the
same bytes, and a LoRA only shows a habit when served if the tokens it
trained on are the tokens Ollama puts in front of it. Measured on the four
real captures (fixtures/prompt/real, Ollama 0.34.4's own code via parity/):
0 of 10 tool definitions were byte-identical, every echoed tool call
differed, ~6–7k of a row's ~11.7k prompt tokens.

What Ollama 0.34.4 does, and this module replicates (file:line at the
v0.34.4 tag):

1. **Tool definitions.** `anthropic/anthropic.go:665-681 convertTool`
   unmarshals each `input_schema` into `api.ToolFunctionParameters`
   (`api/types.go:472-478`: type, $defs, items, required, properties) whose
   properties are `api.ToolProperty` (`api/types.go:417-425`: anyOf, type,
   items, description, enum, properties, required). Keys the structs have
   no field for — `additionalProperties`, `minItems`, the top-level
   `strict` — are dropped; `required: []` and an empty description are
   `omitempty`; a `type` with one member is a string, with several an array
   (`api/types.go:332-339`); `properties` keeps the schema's order
   (`ToolPropertiesMap`, an ordered map, `api/types.go:353-415`) and is
   `null` when absent. The renderer then writes the `api.Tool` with
   `marshalWithSpaces` (`model/renderers/qwen35.go:236-240`,
   `json.go:8-14`): Go's `json.Marshal` — struct field order, `<`, `>` and
   `&` as `\\u003c`/`\\u003e`/`\\u0026`, nested `any` values (items, enum,
   $defs) as maps with SORTED keys — then one space after every `:` and `,`
   outside strings (`json.go:16-48`). The HF template's `tool | tojson`
   keeps the dict's own order, every key, and the characters as they are.
2. **Echoed tool calls.** The service re-sends each earlier reply from the
   stored turn, whose `tool_use.input` Go re-marshalled from a map (keys
   SORTED on the wire: fixtures/prompt/real, e.g. `edit_list`
   `action,item,list,target` where the model wrote `target,list,action,item`).
   `anthropic.go:485-503` keeps that order (`api.ToolCallFunctionArguments`,
   an ordered map), and the renderer writes each parameter with
   `formatToolCallArgument` (`model/renderers/qwen3coder.go:200-222`): a
   string as it is, nil as `null`, a map or slice as Go's COMPACT
   `json.Marshal` (sorted keys, HTML-escaped), anything else with `fmt %v`
   (a number, which Go decoded as float64: `1000000` → `1e+06`). The HF
   template writes `tojson` with spaces, in the row's order, and Python's
   `None`/`True` for scalars.
3. **Thinking.** The service sends `{"thinking":{"type":"disabled"}}`
   (service/internal/model/client.go plainRequest), which
   `anthropic.go:410-412` maps to `think=false`; `server/routes.go:2704-2706`
   resolves it (`renderers.ResolveThinking`), and the qwen3.5 renderer
   (`qwen35.go:299-311`, `350-357`) then writes NO think block on the
   assistant turns it echoes and ends the prompt with the EMPTY block
   `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n` (`emitEmptyThinkOnNoThink`,
   `renderer.go:71-72`). Stored thinking blocks are dropped from the
   history by the service (`dropThinking`), so none is ever rendered.
4. **Everything else** — the system blocks joined with nothing between them
   (`anthropic.go:335-348`), `TrimSpace` on every message's content
   (`qwen35.go:288`), the user turn, `<tool_response>` turns merged into one
   user turn (`qwen35.go:336-343`), `\\n\\n` between an assistant's text and
   its first call and `\\n` between calls (`qwen35.go:313-321`) — the HF
   template already wrote the same bytes; this module writes them too, so
   no template is involved at all.

**The training text** (`render_training`) is the conversation with EVERY
assistant turn opened by the empty think block: each assistant turn is the
generation point of its own model call, and the span train.py trains on
(from `<|im_start|>assistant\\n` to the next `<|im_start|>user\\n`) then
begins with the same bytes the served generation prompt ends with. The one
thing a single sequence cannot have both ways: when Ollama renders call k,
the k−1 earlier assistant turns carry no think block (point 3), while the
training text has one on each — 4 tokens per earlier turn, ~11k tokens
back; tests/test_serving.py measures exactly that residue and nothing else
(`strip_history_think`). `render_prompt` is the served form itself — the
prompt Ollama renders for the NEXT call of a conversation — and is what
the parity fixture is checked against byte for byte; dpo.py renders a
pair's prompt with it, so a DPO prompt is exactly the served bytes.

The parity fixture (`fixtures/parity/served.json`) is Ollama's own render
of the four real captures' requests, written by `python -m
souschef_lab.serving --write` through parity/ (a Go module pinning
github.com/ollama/ollama v0.34.4; `--check` re-runs it and diffs). Re-run
`--write` whenever the model image's Ollama version changes (lab/model-image),
the renderer or parser named in the export's Modelfile changes, or the
service changes how it builds a request — and rebuild the datasets'
`over_length` counts with it (dataset.py counts tokens on this render).

The Go-format helpers (go_string, go_json_number, go_fmt_v, go_json,
tool_argument, served_arguments, unservable, add_json_spaces, the api.Tool
writers) are the thermometer's lab/serving.py (FOCACCIA-PLAN K8), copied;
the renderer itself is new here: the thermometer renders through Qwen3.8's
template with placeholders, Sous Chef renders the whole prompt in Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

# The export's renderer and parser (export.py reads them from `ollama show
# qwen3.5:9b --modelfile`; base-modelfiles/qwen3.5-9b.modelfile) and the
# Ollama the model image serves (lab/model-image/Dockerfile) — the version
# parity/go.mod pins and the fixture was recorded on.
RENDERER = "qwen3.5"
PARSER = "qwen3.5"
OLLAMA_VERSION = "0.34.4"
DESCRIPTION = f"serving-parity: Ollama {OLLAMA_VERSION} {RENDERER} renderer, thinking disabled"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
# The markers the render writes that the base's tokenizer must hold as single tokens (train.served_markers).
SPECIAL_TOKENS = (IM_START, IM_END, "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>",
                  "</tool_response>")
EMPTY_THINK = "<think>\n\n</think>\n\n"
GENERATION_PROMPT = IM_START + "assistant\n" + EMPTY_THINK  # qwen35.go:350-357 with think=false
TOOLS_HEAD = IM_START + "system\n# Tools\n\nYou have access to the following functions:\n\n<tools>"
# qwen35.go:19-43 qwen35ToolPostamble, verbatim.
TOOL_POSTAMBLE = """
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""

LAB_DIR = Path(__file__).resolve().parent.parent
HARNESS = LAB_DIR / "parity"
FIXTURE = LAB_DIR / "fixtures" / "parity" / "served.json"
CAPTURES = LAB_DIR / "fixtures" / "prompt" / "real"


# --- Go's strings.TrimSpace ---------------------------------------------------------------------------------------------

# unicode.IsSpace: Go's White_Space set. Python's str.strip() also trims U+001C–U+001F; Go does not.
_GO_SPACE = frozenset("\t\n\v\f\r \u0085            "
                      "     　")


def go_trim_space(s: str) -> str:
    a, b = 0, len(s)
    while a < b and s[a] in _GO_SPACE:
        a += 1
    while b > a and s[b - 1] in _GO_SPACE:
        b -= 1
    return s[a:b]


# --- Go's encoding/json and fmt, for the values Ollama holds as `any` -----------------------------------------------

_ESCAPES = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f",
            "<": "\\u003c", ">": "\\u003e", "&": "\\u0026", " ": "\\u2028", " ": "\\u2029"}


def go_string(s: str) -> str:
    """A string as Go's json.Marshal writes it (HTML escaping on, as it is
    by default): <, >, &, U+2028 and U+2029 as \\u escapes, the other
    control characters as \\u00xx, everything else as UTF-8 — a lone
    surrogate (invalid UTF-8 in Go) as U+FFFD."""
    out = ['"']
    for ch in s:
        e = _ESCAPES.get(ch)
        if e is not None:
            out.append(e)
        elif ch < " ":
            out.append(f"\\u{ord(ch):04x}")
        elif "\ud800" <= ch <= "\udfff":
            out.append("�")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _digits(x: float) -> tuple[str, str, int]:
    """(sign, shortest round-trip digits, decimal exponent of the first
    digit) of a float64 — the digits Go's strconv picks with precision -1."""
    sign, digits, exp = Decimal(repr(x)).as_tuple()
    ds = "".join(map(str, digits)).rstrip("0") or "0"
    lead = len(digits) + exp - 1  # exponent of the first digit
    if ds == "0":
        lead = 0
    return ("-" if sign else ""), ds, lead


def _fixed(sign: str, ds: str, lead: int) -> str:
    if lead >= 0:
        whole = ds[:lead + 1].ljust(lead + 1, "0")
        frac = ds[lead + 1:]
    else:
        whole, frac = "0", "0" * (-lead - 1) + ds
    return sign + whole + ("." + frac if frac else "")


def _exp(sign: str, ds: str, lead: int) -> str:
    mant = ds[0] + ("." + ds[1:] if len(ds) > 1 else "")
    return f"{sign}{mant}e{'-' if lead < 0 else '+'}{abs(lead):02d}"


def go_json_number(x: float) -> str:
    """A float64 as Go's json.Marshal writes it: fixed notation, exponent
    form below 1e-6 or from 1e21 (with a one-digit negative exponent's
    leading zero dropped: 1e-7, not 1e-07)."""
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"Go's JSON cannot write {x}")
    if x == 0:
        return "-0" if math.copysign(1, x) < 0 else "0"
    sign, ds, lead = _digits(x)
    if abs(x) < 1e-6 or abs(x) >= 1e21:
        s = _exp(sign, ds, lead)
        head, _, tail = s.partition("e")
        if tail.startswith("-0"):
            tail = "-" + tail[2:]
        return head + "e" + tail
    return _fixed(sign, ds, lead)


def go_fmt_v(x: float) -> str:
    """A float64 as Go's fmt %v prints it (strconv 'g', shortest): exponent
    form from an exponent of 6 or below -4 — 1000000 → 1e+06."""
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "+Inf" if x > 0 else "-Inf"
    if x == 0:
        return "-0" if math.copysign(1, x) < 0 else "0"
    sign, ds, lead = _digits(x)
    if lead < -4 or lead >= 6:
        return _exp(sign, ds, lead)
    return _fixed(sign, ds, lead)


def go_json(v: Any) -> str:
    """Any JSON value as Go's json.Marshal writes the `any` Ollama decoded
    it into: compact, objects (Go maps) with sorted keys, numbers float64."""
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, (int, float)):
        return go_json_number(v)
    if isinstance(v, str):
        return go_string(v)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(go_json(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(go_string(str(k)) + ":" + go_json(v[k]) for k in sorted(v, key=str)) + "}"
    raise TypeError(f"not a JSON value: {type(v).__name__}")


# --- Tool-call arguments as the renderer echoes them ------------------------------------------------------------------

def tool_argument(v: Any) -> str:
    """One tool-call argument as Ollama's renderer writes it inside
    <parameter=…> (qwen3coder.go formatToolCallArgument): a string as it
    is, a list or object as Go's compact JSON, null as "null", a number
    with %v, a bool as true/false."""
    if isinstance(v, str):
        return v
    if v is None:
        return "null"
    if v is True or v is False:
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return go_fmt_v(v)
    return go_json(v)


def decode_arguments(arguments: Any) -> dict:
    """A call's arguments as an object: a JSON string is decoded first;
    anything but an object is {} (the service stores {} for it too)."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            arguments = {}
    return arguments if isinstance(arguments, dict) else {}


def served_arguments(arguments: Any) -> dict[str, str]:
    """A call's arguments as the served prompt holds them: in the order the
    service sends them back (SORTED — the stored tool_use.input is Go's
    re-marshal of a map; fixtures/prompt/real), each already written as
    Ollama writes it."""
    args = decode_arguments(arguments)
    return {k: tool_argument(args[k]) for k in sorted(args)}


def _unservable_value(v: Any) -> str:
    """Why an argument's served text would not read back as the same value
    through the tag's parser (qwen3.5 → Qwen3CoderParser.parseValue), "" if
    it would: a number Ollama prints in exponent form (%v: 1000000 → "1e+06",
    which an integer parameter reads back as the string "1e+06"), or a
    string the parser changes (one leading or trailing newline is cut; "null"
    in any case reads as null)."""
    if isinstance(v, bool) or v is None:
        return ""
    if isinstance(v, (int, float)):
        return f"{v!r} is served as {go_fmt_v(v)}" if go_fmt_v(v) != go_json_number(v) else ""
    if isinstance(v, str):
        if v.startswith("\n") or v.endswith("\n"):
            return "a string starting or ending with a newline"
        if v.lower() == "null":
            return 'the string "null"'
    return ""


def unservable(messages: list[dict]) -> list[str]:
    """The tool calls of a row whose served form would teach a call the
    parser reads back differently (the thermometer's K8 echo check). A row
    with one is dropped and counted, never rendered another way: the served
    form is what later turns see, so rendering the call as the model first
    wrote it would break parity instead."""
    out = []
    for m in messages:
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            for k, v in decode_arguments(fn.get("arguments")).items():
                if why := _unservable_value(v):
                    out.append(f"{fn.get('name', '?')}.{k}: {why}")
    return out


# --- Ollama's api.Tool, as marshalWithSpaces writes it ------------------------------------------------------------------

def _obj(pairs: list[tuple[str, str]]) -> str:
    return "{" + ",".join(go_string(k) + ":" + v for k, v in pairs) + "}"


def _property(p: Any) -> str:
    """One api.ToolProperty (api/types.go:417-425): anyOf, type, items,
    description, enum, properties, required — each left out when empty,
    anything else dropped."""
    if not isinstance(p, dict):
        p = {}
    out = []
    if p.get("anyOf"):
        out.append(("anyOf", "[" + ",".join(_property(x) for x in p["anyOf"]) + "]"))
    t = p.get("type")
    types = [t] if isinstance(t, str) else list(t or [])
    if len(types) == 1:
        out.append(("type", go_string(types[0])))
    elif types:
        out.append(("type", go_json(types)))
    if p.get("items") is not None:
        out.append(("items", go_json(p["items"])))
    if p.get("description"):
        out.append(("description", go_string(p["description"])))
    if p.get("enum"):
        out.append(("enum", go_json(p["enum"])))
    if isinstance(p.get("properties"), dict):
        out.append(("properties", _obj([(k, _property(v)) for k, v in p["properties"].items()])))
    if p.get("required"):
        out.append(("required", go_json(p["required"])))
    return _obj(out)


def _parameters(s: Any) -> str:
    """api.ToolFunctionParameters (api/types.go:472-478): type, $defs,
    items, required, properties (in the schema's own order; null when
    absent)."""
    if not isinstance(s, dict):
        s = {}
    out = [("type", go_string(s.get("type") if isinstance(s.get("type"), str) else ""))]
    if s.get("$defs") is not None:
        out.append(("$defs", go_json(s["$defs"])))
    if s.get("items") is not None:
        out.append(("items", go_json(s["items"])))
    if s.get("required"):
        out.append(("required", go_json(s["required"])))
    props = s.get("properties")
    fields = _obj([(k, _property(v)) for k, v in props.items()]) if isinstance(props, dict) else "null"
    out.append(("properties", fields))
    return _obj(out)


def add_json_spaces(b: str) -> str:
    """Ollama's addJSONSpaces (model/renderers/json.go:16-48): a space after
    every ':' and ',' outside strings."""
    out, in_str, esc = [], False, False
    for c in b:
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            out.append(c)
        elif c in ":,":
            out.append(c + " ")
        else:
            out.append(c)
    return "".join(out)


def ollama_tool_json(tool: dict) -> str:
    """A row's tool definition ({type, function: {name, description,
    parameters}}, dataset.hf_tools) as the qwen3.5 renderer writes it into
    the <tools> block (qwen35.go:236-240 marshalWithSpaces of the api.Tool
    convertTool built from the Anthropic tool)."""
    fn = tool.get("function") or {}
    parts = [("name", go_string(str(fn.get("name") or "")))]
    if fn.get("description"):
        parts.append(("description", go_string(str(fn["description"]))))
    parts.append(("parameters", _parameters(fn.get("parameters"))))
    return add_json_spaces(_obj([("type", go_string("function")), ("function", _obj(parts))]))


def tools_block(tools: Iterable[dict], system: str) -> str:
    """The one system turn the renderer writes when tools are present
    (qwen35.go:230-251): the <tools> block, the postamble, then the trimmed
    system text after a blank line."""
    sb = [TOOLS_HEAD]
    for t in tools:
        sb.append("\n" + ollama_tool_json(t))
    sb.append(TOOL_POSTAMBLE)
    system = go_trim_space(system)
    if system:
        sb.append("\n\n" + system)
    sb.append(IM_END + "\n")
    return "".join(sb)


# --- The qwen3.5 renderer (model/renderers/qwen35.go Render, think=false, no images) ---------------------------------

def _split_reasoning(content: str, thinking: str, render_think: bool) -> tuple[str, str]:
    """splitQwen35ReasoningContent (qwen35.go:92-112) for the qwen3.5
    variant (extractTaggedReasoning on): a message's own thinking when the
    block is rendered, else whatever a literal </think> in the content
    closes."""
    if render_think and thinking:
        return go_trim_space(thinking), content
    reasoning = ""
    idx = content.find("</think>")
    if idx != -1:
        before = content[:idx]
        open_ = before.rfind("<think>")
        reasoning = before[open_ + len("<think>"):] if open_ != -1 else before
        content = content[idx + len("</think>"):].lstrip("\n")
    return go_trim_space(reasoning), content


def _is_tool_response(content: str) -> bool:
    c = go_trim_space(content)
    return c.startswith("<tool_response>") and c.endswith("</tool_response>")


def render_conversation(messages: list[dict], tools: list[dict] | None, *, think_turns: str = "none",
                        generation_prompt: bool = False, prefill: bool = False) -> str:
    """The qwen3.5 renderer's prompt for a row's messages (dataset.to_hf
    shape: system, user, assistant{content, tool_calls}, tool{content}) and
    hf-form tools, with thinking disabled.

    think_turns says which assistant turns open with the empty think block:
    "none" — as Ollama renders a history (the served form); "all" — the
    training text, every assistant turn being the generation point of its
    own call; "last" — only the final one (a DPO completion, which begins
    where the served generation prompt ends). generation_prompt appends
    `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n` after a conversation
    that does not end in an assistant turn. prefill: a conversation ending
    in an assistant turn is a prefill at serve (qwen35.go:291, 333: no
    <|im_end|> after it); the training text and a DPO completion close it."""
    if think_turns not in ("none", "all", "last"):
        raise ValueError(f"think_turns {think_turns!r}: none, all or last")
    msgs = [dict(m) for m in messages]
    tools = list(tools or [])
    sb: list[str] = []
    system = msgs[0] if msgs and msgs[0].get("role") == "system" else None
    if tools:
        sb.append(tools_block(tools, str((system or {}).get("content") or "")))
    elif system is not None:  # qwen35.go:252-264: the qwen3.5 variant writes the turn even when empty
        sb.append(IM_START + "system\n" + go_trim_space(str(system.get("content") or "")) + IM_END + "\n")

    last_assistant = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)
    for i, m in enumerate(msgs):
        role = str(m.get("role") or "")
        content = go_trim_space(str(m.get("content") or ""))
        last = i == len(msgs) - 1
        is_prefill = prefill and last and role == "assistant"
        if role == "user" or (role == "system" and i != 0):
            sb.append(IM_START + role + "\n" + content + IM_END + "\n")
        elif role == "assistant":
            render_think = think_turns == "all" or (think_turns == "last" and i == last_assistant)
            reasoning, content = _split_reasoning(content, str(m.get("reasoning_content") or ""), render_think)
            if render_think:
                sb.append(IM_START + "assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content)
            else:
                sb.append(IM_START + "assistant\n" + content)
            for j, call in enumerate(m.get("tool_calls") or []):
                fn = call.get("function") or {}
                if j == 0:
                    if go_trim_space(content):
                        sb.append("\n\n")
                else:
                    sb.append("\n")
                sb.append("<tool_call>\n<function=" + str(fn.get("name") or "") + ">\n")
                for name, value in served_arguments(fn.get("arguments")).items():
                    sb.append("<parameter=" + name + ">\n" + value + "\n</parameter>\n")
                sb.append("</function>\n</tool_call>")
            if not is_prefill:
                sb.append(IM_END + "\n")
        elif role == "tool":
            if i == 0 or msgs[i - 1].get("role") != "tool":
                sb.append(IM_START + "user")
            sb.append("\n<tool_response>\n" + content + "\n</tool_response>")
            if last or msgs[i + 1].get("role") != "tool":
                sb.append(IM_END + "\n")
        if last and not is_prefill and generation_prompt:
            sb.append(GENERATION_PROMPT)
    return "".join(sb)


def render_prompt(messages: list[dict], tools: list[dict] | None) -> str:
    """The prompt Ollama renders for the NEXT model call of this
    conversation — the served form, checked byte for byte against the
    parity fixture."""
    return render_conversation(messages, tools, think_turns="none", generation_prompt=True, prefill=True)


def render_training(messages: list[dict], tools: list[dict] | None) -> str:
    """A row's training text (train.py): every assistant turn opened with
    the empty think block the served generation prompt ends with, ending
    after the final reply's <|im_end|>."""
    return render_conversation(messages, tools, think_turns="all", generation_prompt=False)


def strip_history_think(text: str, keep_last: bool = True) -> str:
    """text with the empty think block removed from every assistant turn
    but the last (keep_last) — the training text of a conversation prefix
    in the served form, for the parity test."""
    marker = IM_START + "assistant\n" + EMPTY_THINK
    n = text.count(marker) - (1 if keep_last else 0)
    return text.replace(marker, IM_START + "assistant\n", max(n, 0))


def generation_points(text: str) -> list[int]:
    """Offsets just after each generation prompt in a training text: where
    a model call's prompt ends and its completion begins."""
    out, pos = [], 0
    while (a := text.find(GENERATION_PROMPT, pos)) >= 0:
        out.append(a + len(GENERATION_PROMPT))
        pos = a + 1
    return out


# --- The parity fixture: Ollama's own render of the real captures -----------------------------------------------------------

def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def captures(root: Path = CAPTURES) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "model-requests.jsonl").is_file())


def run_harness(bodies: list[dict], harness: Path = HARNESS) -> list[dict]:
    """Ollama 0.34.4's prompt for each /v1/messages body, through parity/
    (`go run`; needs Go — the module's `go 1.26.0` line makes GOTOOLCHAIN
    fetch or reuse that toolchain). One dict per body: prompt, think,
    messages, tools."""
    env = dict(os.environ)
    env.setdefault("GOTOOLCHAIN", "auto")
    stdin = "".join(json.dumps(b, ensure_ascii=False) + "\n" for b in bodies)
    p = subprocess.run(["go", "run", "."], cwd=harness, input=stdin, capture_output=True, text=True, env=env,
                       check=False)
    if p.returncode != 0:
        raise RuntimeError(f"parity/ failed:\n{p.stderr.strip()}")
    out = [json.loads(x) for x in p.stdout.splitlines() if x.strip()]
    if len(out) != len(bodies):
        raise RuntimeError(f"parity/ rendered {len(out)} prompts for {len(bodies)} requests")
    return out


def harness_available(harness: Path = HARNESS) -> bool:
    """Whether `go` is on PATH (the test that re-runs the harness skips
    otherwise; SOUS_CHEF_LAB_REQUIRE_GO=1 makes it fail instead)."""
    from shutil import which
    return which("go") is not None and (harness / "go.mod").is_file()


def served_fixture(root: Path = CAPTURES, harness: Path = HARNESS) -> dict:
    """The fixture's content: for every capture, Ollama's prompt for each of
    its requests (a shared prefix once, each prompt's tail and sha256)."""
    entries = []
    for where in captures(root):
        reqs = read_jsonl(where / "model-requests.jsonl")
        rendered = run_harness([r["body"] for r in reqs], harness)
        entries.append((where.name, rendered))
    prompts = [r["prompt"] for _, rendered in entries for r in rendered]
    prefix = os.path.commonprefix(prompts) if prompts else ""
    thinks = sorted({r["think"] for _, rendered in entries for r in rendered})
    return {
        "what": "Ollama's prompt for every model request of fixtures/prompt/real/*/model-requests.jsonl "
                "(python -m souschef_lab.serving --write, through parity/)",
        "ollama": OLLAMA_VERSION, "renderer": RENDERER, "parser": PARSER, "think": thinks,
        "prefix": prefix,
        "captures": [{"capture": name,
                      "requests": [{"chars": len(r["prompt"]), "sha256": sha(r["prompt"]), "messages": r["messages"],
                                    "tools": r["tools"], "tail": r["prompt"][len(prefix):]} for r in rendered]}
                     for name, rendered in entries],
    }


def fixture_prompts(fixture: dict) -> dict[str, list[str]]:
    """capture name → Ollama's prompt per request, rebuilt from the fixture
    (prefix + tail) and checked against the recorded sha256."""
    out: dict[str, list[str]] = {}
    for c in fixture["captures"]:
        prompts = []
        for k, r in enumerate(c["requests"]):
            prompt = fixture["prefix"] + r["tail"]
            if len(prompt) != r["chars"] or sha(prompt) != r["sha256"]:
                raise ValueError(f"{c['capture']} request {k + 1}: the fixture's prompt does not match its sha256")
            prompts.append(prompt)
        out[c["capture"]] = prompts
    return out


def comparable(fixture: dict) -> list:
    return [fixture["ollama"], fixture["renderer"], fixture["parser"], fixture["think"], fixture["prefix"],
            [(c["capture"], [(r["chars"], r["sha256"], r["tail"]) for r in c["requests"]]) for c in fixture["captures"]]]


def first_difference(a: str, b: str, context: int = 60) -> str:
    i = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"at {i}: {a[max(0, i - context):i + context]!r} ≠ {b[max(0, i - context):i + context]!r}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Record (or check) Ollama's own render of the real captures — the serving "
                                             "parity fixture tests/test_serving.py holds the lab's render to.")
    ap.add_argument("--write", action="store_true", help=f"write {FIXTURE.relative_to(LAB_DIR)} (else only check it)")
    ap.add_argument("--captures", type=Path, default=CAPTURES, help="the captures' folder (default: fixtures/prompt/real)")
    args = ap.parse_args(argv)
    fixture = served_fixture(args.captures)
    calls = sum(len(c["requests"]) for c in fixture["captures"])
    print(f"parity/: {len(fixture['captures'])} captures, {calls} requests rendered as Ollama {OLLAMA_VERSION}'s "
          f"{RENDERER} renderer does (think {fixture['think']})")
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


if __name__ == "__main__":
    sys.exit(main())
