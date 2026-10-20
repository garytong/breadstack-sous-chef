"""What the Sous Chef lab's three training commands share (train.py, dpo.py,
export.py): where the lab reads and writes, the dataset row shape, the
config file, the fingerprint of a run, and the small text helpers.

Pure Python, standard library only: nothing here imports torch, Unsloth, TRL
or PyYAML at module level, so `--dry-run` and the tests run it without the
`[train]` extra. The trainers pass in what they measured.

Copied, with changes, from the thermometer's training kit
(breadstack-thermometer/lab/in-house-model: lab/trainkit.py, lab/hf.py,
lab/paths.py, lab/console.py — the pattern, never imported). Dropped on
purpose (docs/LAB.md, "Fixed decisions"): the Claude refusal (Claude is this
lab's teacher), the real-data/owner-only paths, the 27B presets, the
checkpoint bucket sync and the smoke's stop rules.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# --- Where the lab reads and writes -----------------------------------------------------------------------------------
# Resolved from this file, not the working directory (thermometer lab/paths.py).

LAB_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = LAB_DIR / "data"
RUNS_DIR = LAB_DIR / "runs"
DATASET_DIR = DATA_DIR / "dataset"  # dataset.py: train.jsonl / val.jsonl / exam.jsonl / dataset-manifest.json
PAIRS_DIR = DATA_DIR / "pairs"  # pairs.py: train.jsonl / val.jsonl / pairs-manifest.json
ADAPTERS_DIR = RUNS_DIR / "adapters"  # train.py / dpo.py: <name>/adapter, run.json, loss.jsonl
EXPORTS_DIR = RUNS_DIR / "exports"  # export.py: <name>/*.gguf, Modelfile.<quant>, export.json


def rel(p: Path) -> str:
    """A path as the lab prints it: relative to lab/ when under it."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(LAB_DIR))
    except ValueError:
        return str(p)


# --- Console --------------------------------------------------------------------------------------------------------------
# Plain print: the thermometer used rich; the lab's rule is stdlib only.


def say(text: str) -> None:
    print(text, flush=True)


class Die(SystemExit):
    """A fatal error with its message; exit status 1."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(1)


def die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr, flush=True)
    raise Die(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


# --- JSONL --------------------------------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file ([] when it does not exist); a torn last line is
    skipped, as the thermometer's reader does."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def load_split(data_dir: Path, name: str) -> list[dict]:
    return read_jsonl(Path(data_dir) / f"{name}.jsonl")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


# --- The config file (YAML or JSON) ------------------------------------------------------------------------------------
# A run's settings may come from a file (--config); a flag given on the
# command line wins. JSON needs nothing; YAML uses PyYAML when installed and
# otherwise the small subset below (maps nested by indent, `- ` lists,
# scalars) — enough for a settings file, not a YAML parser.


def _yaml_scalar(text: str) -> Any:
    t = text.strip()
    if t == "" or t in ("~", "null", "Null", "NULL"):
        return None
    if t in ("true", "True", "TRUE"):
        return True
    if t in ("false", "False", "FALSE"):
        return False
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        return t[1:-1]
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].strip()
        return [_yaml_scalar(x) for x in inner.split(",")] if inner else []
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def _yaml_lines(text: str) -> list[tuple[int, str]]:
    out = []
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0] if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        out.append((len(line) - len(line.lstrip(" ")), line.strip()))
    return out


def _yaml_block(lines: list[tuple[int, str]], i: int, indent: int) -> tuple[Any, int]:
    if i >= len(lines):
        return {}, i
    if lines[i][1].startswith("- "):
        items = []
        while i < len(lines) and lines[i][0] == indent and lines[i][1].startswith("- "):
            items.append(_yaml_scalar(lines[i][1][2:]))
            i += 1
        return items, i
    out: dict = {}
    while i < len(lines) and lines[i][0] == indent:
        key, sep, rest = lines[i][1].partition(":")
        if not sep:
            raise ValueError(f"yaml: line {i + 1}: expected `key: value`, got {lines[i][1]!r}")
        key = key.strip().strip("\"'")
        i += 1
        if rest.strip():
            out[key] = _yaml_scalar(rest)
        elif i < len(lines) and lines[i][0] > indent:
            out[key], i = _yaml_block(lines, i, lines[i][0])
        else:
            out[key] = None
    return out, i


def parse_simple_yaml(text: str) -> dict:
    """The YAML subset a settings file needs (see above). Raises ValueError
    on anything else."""
    lines = _yaml_lines(text)
    if not lines:
        return {}
    value, i = _yaml_block(lines, 0, lines[0][0])
    if i != len(lines):
        raise ValueError(f"yaml: line {i + 1}: unexpected indentation")
    if not isinstance(value, dict):
        raise ValueError("yaml: the top level must be a mapping")
    return value


def load_config(path: Path) -> dict:
    """A settings file as a flat dict of `flag_name: value` (hyphens become
    underscores). .json, or .yaml/.yml (PyYAML if installed, else the
    subset parser)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
    elif p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # optional; never required

            data = yaml.safe_load(text)
        except ImportError:
            data = parse_simple_yaml(text)
    else:
        raise ValueError(f"--config {p}: .json, .yaml or .yml")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"--config {p}: the top level must be a mapping")
    return {str(k).replace("-", "_"): v for k, v in data.items()}


def apply_config(parser, argv: list[str], allowed: Iterable[str]) -> list[str]:
    """Pull `--config FILE` out of argv, set the file's values as the
    parser's defaults (so a flag on the command line still wins), and return
    argv without it. A key the parser does not know is an error."""
    argv = list(argv)
    path = None
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            path = argv[i + 1]
            del argv[i:i + 2]
            break
        if a.startswith("--config="):
            path = a.split("=", 1)[1]
            del argv[i]
            break
    if path is None:
        return argv
    try:
        cfg = load_config(Path(path))
    except (OSError, ValueError) as e:
        parser.error(str(e))
    allowed = set(allowed)
    unknown = sorted(k for k in cfg if k not in allowed)
    if unknown:
        parser.error(f"--config {path}: unknown setting(s) {', '.join(unknown)}")
    parser.set_defaults(**cfg)
    parser.set_defaults(config=str(path))
    return argv


# --- The dataset row (thermometer lab/hf.py: the OpenAI-style shape open chat templates take) ---------------------------
#
#   system → user → assistant{content, tool_calls[{id, type:"function", function{name, arguments}}]}
#          → tool{tool_call_id, name, content} … → assistant (the final reply)
#
# with `tools` a list of {type:"function", function:{name, description, parameters}}.
# Arguments are stored as a JSON STRING (one column type for Arrow); train.py
# decodes them before applying a template that iterates them as a mapping.


def render_tool_call(tool_use_id: str, name: str, input: Any) -> dict:
    """One tool call as the row stores it — byte for byte the thermometer's
    to_hf shape (lab/hf.py: id, type "function", function.name,
    function.arguments as json.dumps(args, ensure_ascii=False))."""
    args = input if isinstance(input, dict) else {}
    return {"id": str(tool_use_id), "type": "function",
            "function": {"name": str(name), "arguments": json.dumps(args, ensure_ascii=False)}}


def render_tool_result(tool_use_id: str, name: str, content: Any) -> dict:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return {"role": "tool", "tool_call_id": str(tool_use_id), "name": str(name), "content": text}


def hf_tool(tool: dict) -> dict:
    """A README §2 tool definition ({name, description, input_schema}) in
    function-calling form (thermometer hf_tools)."""
    return {"type": "function", "function": {
        "name": tool.get("name", ""),
        "description": tool.get("description", "") or "",
        "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
    }}


def decode_arguments(messages: list[dict]) -> list[dict]:
    """The dataset keeps tool-call arguments as JSON strings; templates such
    as Qwen3.5's iterate them as a mapping — so decode (thermometer
    train.decode_arguments, unchanged)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                c = json.loads(json.dumps(c))
                args = c.get("function", {}).get("arguments")
                if isinstance(args, str):
                    try:
                        c["function"]["arguments"] = json.loads(args) if args.strip() else {}
                    except ValueError:
                        c["function"]["arguments"] = {}
                calls.append(c)
            m["tool_calls"] = calls
        out.append(m)
    return out


def tool_call_problems(call: Any, where: str) -> list[str]:
    """Why one stored tool call is not in the thermometer shape ([] when it is)."""
    if not isinstance(call, dict):
        return [f"{where}: tool call is not an object"]
    out = []
    if call.get("type") != "function":
        out.append(f"{where}: type is {call.get('type')!r}, not 'function'")
    if not isinstance(call.get("id"), str) or not call["id"]:
        out.append(f"{where}: no string id")
    fn = call.get("function")
    if not isinstance(fn, dict):
        return out + [f"{where}: no function object"]
    if not isinstance(fn.get("name"), str) or not fn["name"]:
        out.append(f"{where}: function.name missing")
    args = fn.get("arguments")
    if not isinstance(args, str):
        out.append(f"{where}: function.arguments is {type(args).__name__}, not a JSON string")
    else:
        try:
            if not isinstance(json.loads(args) if args.strip() else {}, dict):
                out.append(f"{where}: function.arguments does not decode to an object")
        except ValueError:
            out.append(f"{where}: function.arguments is not valid JSON")
    return out


ROLES = ("system", "user", "assistant", "tool")


def check_rows(rows: list[dict], where: str) -> list[str]:
    """Shape problems in dataset rows, as messages (empty = fine). The
    thermometer's checks (messages, a final assistant answer, a tools
    list) plus every tool call and tool result in the stored shape."""
    problems = []
    for i, r in enumerate(rows):
        rid = r.get("id")
        tag = f"{where} row {i} ({rid})"
        msgs = r.get("messages")
        if not isinstance(msgs, list) or not msgs:
            problems.append(f"{where} row {i}: no messages")
            continue
        if msgs[-1].get("role") != "assistant" or not (msgs[-1].get("content") or "").strip():
            problems.append(f"{tag}: does not end with an assistant answer")
        if not isinstance(r.get("tools"), list):
            problems.append(f"{tag}: no tools list")
        for j, m in enumerate(msgs):
            if not isinstance(m, dict) or m.get("role") not in ROLES:
                problems.append(f"{tag}: message {j} has no valid role")
                continue
            if m["role"] == "tool":
                if not isinstance(m.get("tool_call_id"), str) or not isinstance(m.get("content"), str):
                    problems.append(f"{tag}: message {j} (tool) needs tool_call_id and content strings")
            elif m["role"] == "assistant":
                if m.get("tool_calls") is not None:
                    if not isinstance(m["tool_calls"], list):
                        problems.append(f"{tag}: message {j} tool_calls is not a list")
                    else:
                        for k, c in enumerate(m["tool_calls"]):
                            problems += tool_call_problems(c, f"{tag}: message {j} tool_calls[{k}]")
            if m["role"] != "tool" and not isinstance(m.get("content"), str):
                problems.append(f"{tag}: message {j} ({m['role']}) content is not a string")
    return problems


def history_turns(row: dict) -> int:
    """How many leading assistant turns of a row are resent history (the
    dataset records it; a one-turn row has none)."""
    return int((row.get("meta") or {}).get("history_assistant_turns") or 0)


# --- Size estimates (thermometer lab/hf.py) ----------------------------------------------------------------------------


def row_chars(row: dict) -> int:
    """Characters a row puts in front of the model: every message plus the
    tool definitions, which every example repeats."""
    n = len(json.dumps(row.get("tools", []), ensure_ascii=False))
    for m in row.get("messages", []):
        n += len(m.get("content") or "") + len(m.get("reasoning_content") or "")
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            args = fn.get("arguments")
            n += len(fn.get("name") or "") + (len(args) if isinstance(args, str) else len(json.dumps(args or {})))
    return n


# docs/LAB.md §5: 3.0 bytes a token when the base tokenizer is not installed
# (the thermometer used 3.3; the Sous Chef rows are JSON-heavier).
CHARS_PER_TOKEN = 3.0


def estimate_tokens(row: dict) -> int:
    return int(row_chars(row) / CHARS_PER_TOKEN) + 1


def median(xs: list[int]) -> int:
    return sorted(xs)[len(xs) // 2] if xs else 0


def updates_for(rows: int, batch: int, grad_accum: int, epochs: float, max_steps: int | None) -> int:
    return max_steps or math.ceil(rows / (batch * grad_accum) * epochs)


# --- Masking (thermometer train.py, unchanged) -------------------------------------------------------------------------
# train_on_responses_only splits each rendered conversation at these
# markers: text after response_part up to the next instruction_part is
# trained on, everything else is masked. Tool results must therefore open
# with instruction_part, which holds for ChatML — Qwen3.5 renders them as a
# user turn wrapping <tool_response>. It does not hold for Gemma or Llama 3,
# where the mask would train on tool output.

MARKERS = {"chatml": ("<|im_start|>user\n", "<|im_start|>assistant\n")}


def template_kind(chat_template: str) -> str | None:
    return "chatml" if "<|im_start|>" in chat_template else None


def trained_spans(text: str, instruction_part: str, response_part: str) -> list[str]:
    """The parts of a rendered conversation train_on_responses_only keeps in
    the loss: from each response_part to the next instruction_part."""
    spans, pos = [], 0
    while (a := text.find(response_part, pos)) >= 0:
        a += len(response_part)
        b = text.find(instruction_part, a)
        spans.append(text[a: b if b >= 0 else len(text)])
        pos = a
    return spans


def find_all(seq: list[int], sub: list[int], start: int = 0) -> list[int]:
    """Start offsets of sub in seq from start, non-overlapping."""
    out, n, i = [], len(sub), start
    if n == 0:
        return out
    while i <= len(seq) - n:
        if seq[i] == sub[0] and seq[i:i + n] == sub:
            out.append(i)
            i += n
        else:
            i += 1
    return out


def mask_history_turns(input_ids: list[int], labels: list[int], response_ids: list[int], instruction_ids: list[int],
                       k: int) -> list[int]:
    """labels with the first k assistant turns masked (-100): a follow-up's
    resent earlier replies are context, not this answer."""
    if k <= 0:
        return list(labels)
    out = list(labels)
    for a in find_all(input_ids, response_ids)[:k]:
        a += len(response_ids)
        nxt = find_all(input_ids, instruction_ids, a)
        for i in range(a, nxt[0] if nxt else len(input_ids)):
            out[i] = -100
    return out


def leaked_tool_results(row: dict, text: str, instruction_part: str, response_part: str) -> list[str]:
    """Tool results of row that would be trained on — should be none. Checked
    before training so a template that renders tool output inside the
    assistant's turn stops the run instead of teaching the student to write
    tool output itself."""
    trained = "".join(trained_spans(text, instruction_part, response_part))
    return [m.get("name", "?") for m in row["messages"]
            if m.get("role") == "tool" and (m.get("content") or "")[:200] and (m.get("content") or "")[:200] in trained]


# --- Masking for the Mistral family (train.py --family mistral) -----------------------------------------------------------
# The Ministral 3 (Tekken v13) tokenizer holds every boundary the render writes as ONE control token
# (serving_mistral.SPECIAL_IDS), so the mask is built from token ids, never from strings or BPE subsequences. An
# assistant segment starts right after [/INST], or after a [/TOOL_RESULTS] that more context does not follow
# ([TOOL_RESULTS], [INST], [AVAILABLE_TOOLS], [SYSTEM_PROMPT]); it ends after its </s>, inclusive (the model learns to
# stop), or at the end of an unclosed prefill. [TOOL_CALLS], the names, [ARGS] and the arguments are trained; the
# system prompt, the tools, the user turns, the tool results and BOS never are. MARKERS / template_kind above are the
# Qwen family's and stay as they are.

MISTRAL_IDS = {"bos": 1, "eos": 2, "inst": 3, "inst_end": 4, "tools": 5, "results": 7, "results_end": 8,
               "system": 17}


def mistral_spans(ids: list[int], k: dict = MISTRAL_IDS) -> list[tuple[int, int]]:
    """[a, b) token ranges of each assistant segment of a tokenized Mistral render."""
    more_context = {k["results"], k["inst"], k["tools"], k["system"]}
    spans: list[tuple[int, int]] = []
    n = len(ids)
    i = 0
    while i < n:
        t = ids[i]
        if t in (k["inst_end"], k["results_end"]):
            a = i + 1
            if a >= n:  # a prompt ends here: its generation point, nothing to train
                break
            if ids[a] in more_context:  # more context follows: not a reply
                i += 1
                continue
            b = a
            while b < n and ids[b] != k["eos"]:
                b += 1
            b = min(b + 1, n)  # the </s> belongs to the segment
            spans.append((a, b))
            i = b
            continue
        i += 1
    return spans


def mistral_labels(input_ids: list[int], history: int = 0, k: dict = MISTRAL_IDS) -> list[int]:
    """-100 everywhere except the assistant segments after the first `history` (a follow-up's resent replies stay
    masked); a trained position's label is its own token id (the model shifts labels itself)."""
    labels = [-100] * len(input_ids)
    for a, b in mistral_spans(input_ids, k)[max(0, history):]:
        labels[a:b] = input_ids[a:b]
    return labels


def weighted_loss(parts: list[tuple[float, int]]) -> float | None:
    """The validation loss over every trained token: each row's mean loss
    weighted by how many of its tokens are trained on."""
    n = sum(k for _, k in parts if k > 0)
    return sum(loss * k for loss, k in parts if k > 0) / n if n else None


def val_subset(rows: list, n: int | None, seed: int) -> list:
    """At most n of rows — a seeded sample in their own order, so every
    validation of a run measures the same rows."""
    if n is None or len(rows) <= n:
        return rows
    import random

    keep = sorted(random.Random(seed).sample(range(len(rows)), n))
    return [rows[i] for i in keep]


# --- Files and fingerprints (thermometer lab/trainkit.py) ----------------------------------------------------------------
# dpo.py caches the frozen SFT reference's log-probs under a fingerprint of
# what they depend on; a cache from another adapter or other pairs is refused.

FINGERPRINT_KEYS = ("kind", "base", "rank", "alpha", "lr", "max_seq_length", "grad_accum", "batch", "epochs",
                    "max_steps", "seed", "optim", "thinking", "render", "data_sha256", "beta", "rpo_alpha",
                    "warmup_ratio", "max_length", "max_pair_tokens", "sft_adapter_sha256")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_sha256(paths: Iterable[Path]) -> str:
    """One hash over the files a run trains on (in order); a missing file
    counts as empty."""
    h = hashlib.sha256()
    for p in paths:
        p = Path(p)
        h.update(p.name.encode())
        h.update(file_sha256(p).encode() if p.exists() else b"-")
    return h.hexdigest()


def fingerprint(settings: dict) -> dict:
    return {k: settings.get(k) for k in FINGERPRINT_KEYS if k in settings}


def fingerprint_mismatch(saved: dict, now: dict) -> list[str]:
    return sorted(k for k in set(saved) | set(now) if saved.get(k) != now.get(k))


def package_versions(names: Iterable[str]) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def amp(torch, bf16: bool):
    """The precision context the Trainer trains in (thermometer trainkit.amp):
    a forward outside train() must run under the same autocast, or it scores
    another model."""
    import contextlib

    if bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# --- The plan, printed ------------------------------------------------------------------------------------------------------


def render_plan(title: str, plan: dict, order: Iterable[str] | None = None) -> str:
    """A run's resolved settings as aligned `key: value` lines, the listed
    keys first (what --dry-run prints and run.json records)."""
    keys = [k for k in (order or []) if k in plan] + sorted(k for k in plan if k not in set(order or []))
    width = max((len(k) for k in keys), default=0)
    lines = [title]
    for k in keys:
        v = plan[k]
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        lines.append(f"  {k.ljust(width)}  {v}")
    return "\n".join(lines)
