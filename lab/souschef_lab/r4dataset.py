"""r4dataset — the r4 training set from the teacher's trajectories (docs/LAB.md §5b).

Input: the merged cases (r4cases.py) and the collector's logs (r4collect.py,
any number of passes). Output: `<out>/dataset/{train,val,rejects}.jsonl`,
`stats.json`, `dataset-manifest.json` and `<out>/manifest.json` — the layout
`train.py`, `make train-remote-push` and the VM's `sous-chef-data-pull` read.

Rows are rendered as the DEPLOYED service (f79753b) renders the request it
sends the student on a messages-compatible route, one row per scored turn
(the turn's model calls are its assistant spans, as r1–r3's rows):

  system   system[0] rules + system[1] index of the cases' manifest, from the service's own code
           (`fields.Prefix` = service/cmd/sous-chef-prefix; SOUS_CHEF_SERVICE_DIR / SOUS_CHEF_PREFIX_BIN pin the
           deployed tree); tools = its tool definitions (report_unsupported with `where`)
  history  the earlier turns as the service STORED them (GET /v1/conversations, logged per row: the only record
           of server-tool round trips — lookup_fields / describe_target are never streamed and
           report_unsupported is streamed without its result), thinking blocks dropped, earlier user turns
           stripped to the merchant's text (prompt.StableHistory); their assistant turns are counted in
           `meta.history_assistant_turns`, which train.py masks
  current  `<context>` + the candidate-fields block the service computes for this turn
           (fields.CandidatesAfter: the previous exchange read for a follow-up) + the text; the stored block is
           compared with the recomputed one: a row whose blocks differ is rejected `block_drift` (counted in
           `stats.block_drift_rejected`; the kept rows' `block_drift_rows` is 0 by construction)
  answer   the stored assistant / tool messages of the turn (server results included)

A row whose log carries no stored conversation (a fake service, a failed GET)
is rebuilt from its events when the turn used no server tool and its model
calls are all visible, else rejected `no_conversation`.

A CARRIED row (retrain/carry.py: `carried: {source, row}` — an earlier round's
accepted, built row the carry step re-checked on this round's engine and case)
is not re-asked: `rerender_carried` replaces its system turn, its tools and its
scored turn's candidate-fields block with this build's (the block from the
same turn inputs: context, text, previous exchange) and keeps every assistant
and tool message byte for byte; then every keep rule below applies to it as
to any row (`meta.carried`, stats `carried_rows` / `carried_by_source` /
`asked_rows`). A carried row ranks first among a case's samples.

The usable rows the caps cut are written to `dataset/capped.jsonl` (never
trained): the next round carries them instead of asking the teacher again for
rows the same caps would cut again.

Keep rules (a row must pass all; `rejects.jsonl` names the first and all):
error, history_error, not_pass (verify-3 verdict), the score's flags
(preview_claim, where_missing, empty_reply, lookup_storm, unexpected_hidden,
history_diverged — re-checked here from the logged events; preview_claim
again on the STORED assistant text, by r4score's one rule), block_drift, rule_version,
engine (the row's engine fingerprint is not the cases'), conversation_mismatch
(the stored turn's tool calls are not the streamed ones), unservable
(serving.unservable), over_length, holdout, leak (the final hold-out check on
every user text of the row). One row per case: the best usable sample
(fewest browser ops, fewest server calls, shortest reply, earliest pass).

Caps, in order: (1) within the families named by `--section-cap-scope`
(default F0,F1 — the generator's skew; F2–F6 are designed concentrations) no
section type / panel (dataset.family_of) holds more than `--section-cap` of
those families' rows; (2) no family holds more than `--family-max-share` of
all rows. Both drop deterministically by a hash of the case id. Split: by
`meta.group` (an F1 paraphrase sits with its F0 source), `--val-pct` (8).

    python -m souschef_lab.r4dataset --cases <r4>/cases-all.jsonl --log <r4>/collect/pass1.jsonl … \\
        --manifest fixtures/exam/manifest-real.json --out data/r4 [--why <study dir>]
    python -m souschef_lab.r4dataset --render-check --cases … --log <r4>/collect/smoke.jsonl --manifest …
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from . import render
from .r4score import PASSING, RULE_VERSION, assistant_texts, flags_of, mentions_preview

DROP_BLOCKS = ("thinking", "redacted_thinking")
SERVER_TOOLS = ("report_unsupported", "lookup_fields", "describe_target")
BUILD_VERSION = f"{RULE_VERSION}+r4dataset-2"
BYTES_PER_TOKEN = 3.0
MAX_SEQ = 32768


def read_jsonl(path: Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def h100(s: str) -> int:
    return int(hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:8], 16) % 100


def hrank(s: str) -> int:
    return int(hashlib.sha1(("r4cap:" + str(s)).encode("utf-8")).hexdigest()[:12], 16)


# --- prompt.StripTurnBlocks (service/internal/prompt/build.go), ported --------------------------------------------------

CONTEXT_OPEN, CONTEXT_CLOSE = "<context>\n", "\n</context>\n"
CAND_OPEN, CAND_CLOSE = "<candidate_fields>\n", "\n</candidate_fields>\n"


def strip_turn_blocks(text: str) -> str:
    rest = text
    if rest.startswith(CONTEXT_OPEN):
        body = rest[len(CONTEXT_OPEN):]
        end = body.find(CONTEXT_CLOSE)
        if end < 0:
            return text
        line = body[:end]
        try:
            ok = "\n" not in line and line.startswith("{") and json.loads(line) is not None
        except ValueError:
            ok = False
        if not ok:
            return text
        rest = body[end + len(CONTEXT_CLOSE):]
    if rest.startswith(CAND_OPEN):
        end = rest.find(CAND_CLOSE)
        if end < 0:
            return text
        rest = rest[end + len(CAND_CLOSE):]
    return text if rest == "" else rest


# --- Anthropic messages → training row (dataset.to_hf's conversion) -----------------------------------------------------

def _block_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("type") == "text")
    return "" if content is None else str(content)


def to_hf_messages(source: list[dict], system_text: str) -> tuple[list[dict], dict]:
    """dataset.to_hf's conversion (thermometer hf.py shape): assistant
    {content, tool_calls[{id, type, function{name, arguments: JSON}}]}, one
    `tool` message per tool_result, user text as is."""
    messages: list[dict] = [{"role": "system", "content": system_text}] if system_text else []
    names: dict[str, str] = {}
    answered: set[str] = set()
    for m in source:
        role = m.get("role")
        content = m.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content or "")}]
        if role == "assistant":
            text, calls = [], []
            for b in blocks:
                if b.get("type") == "text":
                    text.append(str(b.get("text", "")))
                elif b.get("type") == "tool_use":
                    tid, name = str(b.get("id", "")), str(b.get("name", ""))
                    names[tid] = name
                    args = b.get("input") if isinstance(b.get("input"), dict) else {}
                    calls.append({"id": tid, "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
            msg: dict = {"role": "assistant", "content": "".join(text).strip()}
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)
        else:
            for b in blocks:
                if b.get("type") == "tool_result":
                    tid = str(b.get("tool_use_id", ""))
                    answered.add(tid)
                    messages.append({"role": "tool", "tool_call_id": tid, "name": names.get(tid, ""),
                                     "content": _block_text(b.get("content"))})
                elif b.get("type") == "text" and str(b.get("text", "")).strip():
                    messages.append({"role": "user", "content": b["text"]})
    uses = [c for m in messages if m.get("role") == "assistant" for c in m.get("tool_calls") or []]
    return messages, {"tools_called": [c["function"]["name"] for c in uses],
                      "missing_tool_results": sum(1 for c in uses if c["id"] not in answered)}


# --- the student's request for one logged case ------------------------------------------------------------------------

class RenderError(Exception):
    pass


def _has_text(m: dict) -> bool:
    c = m.get("content")
    if isinstance(c, str):
        return bool(c.strip())
    return any(isinstance(b, dict) and b.get("type") == "text" for b in c or [])


def previous_exchange(prev_text: str, prev_row: dict) -> tuple[str, str]:
    texts = assistant_texts(prev_row)
    return render.go_trim_space(prev_text), (render.go_trim_space(texts[-1]) if texts else "")


def student_source(case: dict, row: dict, fields) -> tuple[list[dict], dict]:
    """The Anthropic messages of the row's conversation as the student's route
    sends them, and notes {history_assistant_turns, block_drift, source}."""
    hist = row.get("history_rows") or []
    turn_texts = [str(h.get("text") or "") for h in hist] + [str(row.get("text") or "")]
    previous = previous_exchange(turn_texts[-2], hist[-1]) if hist else None
    block = fields.block(row.get("context"), row.get("text") or "", previous) if fields is not None else ""
    final_user = render.context_block(row.get("context"), row.get("text") or "", block)
    conv = (row.get("conversation") or {}).get("messages")
    notes: dict[str, Any] = {"block_chars": len(block)}
    if conv:
        msgs = [{"role": m.get("role"), "content": json.loads(m["content"]) if isinstance(m.get("content"), str)
                 and m["content"].lstrip().startswith("[") else m.get("content")} for m in conv]
        starts = [i for i, m in enumerate(msgs) if m["role"] == "user" and _has_text(m)]
        if len(starts) != len(turn_texts):
            raise RenderError(f"conversation_mismatch:user_turns {len(starts)} != {len(turn_texts)}")
        out = []
        for i, m in enumerate(msgs):
            content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": str(m["content"] or "")}]
            content = [dict(b) for b in content if isinstance(b, dict) and b.get("type") not in DROP_BLOCKS]
            for b in content:
                b.pop("cache_control", None)
            if i in starts:
                k = starts.index(i)
                texts = [b for b in content if b.get("type") == "text"]
                if len(texts) != 1:
                    raise RenderError("conversation_mismatch:user_text_blocks")
                stored = str(texts[0].get("text") or "")
                if render.go_trim_space(strip_turn_blocks(stored)) != render.go_trim_space(turn_texts[k]):
                    raise RenderError(f"conversation_mismatch:turn {k} text")
                if k < len(turn_texts) - 1:
                    texts[0]["text"] = strip_turn_blocks(stored)  # prompt.StableHistory
                else:
                    notes["block_drift"] = stored != final_user
                    texts[0]["text"] = final_user
            if content:
                out.append({"role": m["role"], "content": content})
        # the streamed calls of the scored turn must be the stored ones
        fi = _nth_user_index(out, len(turn_texts) - 1)
        stored_ids = [str(b.get("id")) for m in out[fi:] if m["role"] == "assistant" for b in m["content"]
                      if b.get("type") == "tool_use" and b.get("name") not in SERVER_TOOLS]
        streamed = [str((e.get("data") or {}).get("id")) for e in row.get("events") or [] if e.get("event") == "tool.call"
                    and (e.get("data") or {}).get("name") not in SERVER_TOOLS]
        streamed += [str((e.get("data") or {}).get("toolCallId")) for e in row.get("events") or [] if e.get("event") == "turn.suspended"]
        if sorted(stored_ids) != sorted(streamed):
            raise RenderError("conversation_mismatch:tool_calls")
        notes["source"] = "conversation"
        notes["history_assistant_turns"] = sum(1 for m in out[:fi] if m["role"] == "assistant")
        return out, notes
    # No stored conversation: the events alone, only when nothing the stream cannot show happened.
    rows = hist + [row]
    for r in rows:
        if any(c.get("name") in SERVER_TOOLS for c in r.get("tool_calls") or []):
            raise RenderError("no_conversation:server_tool")
        calls = int(((r.get("usage") or {}).get("calls")) or 0)
        visible = 1 + int(r.get("batches") or 0)
        if calls > visible:
            raise RenderError("no_conversation:hidden_calls")
    out = []
    for k, r in enumerate(hist):
        ms = render.messages_from_events(r.get("events") or [], r.get("context"), r.get("text") or "",
                                         r.get("candidate_fields") or "")
        ms[0] = {"role": "user", "content": [{"type": "text", "text": render.go_trim_space(r.get("text") or "")}]}
        out += ms
    n_hist = sum(1 for m in out if m["role"] == "assistant")
    ms = render.messages_from_events(row.get("events") or [], row.get("context"), row.get("text") or "", block)
    out += ms
    notes.update(source="events", history_assistant_turns=n_hist, block_drift=None)
    return out, notes


CARRIED_SOURCE = "carried"
CAPPED_FILE = "capped.jsonl"  # the usable rows the caps cut (never trained; a later round may carry them)


def hf_final_turn(hf: dict) -> dict:
    """The scored (last) user turn of a BUILT row (to_hf_messages' shape), taken apart:
    {index, head, context, block, text, previous, history_assistant_turns}. `previous` is the
    follow-up exchange the service reads the turn with (previous_exchange's: the earlier user
    turn's text and the last assistant text before the scored turn, both trimmed), None for a
    first turn. RenderError when the row has no user turn or its user turn does not parse."""
    from .fields import split_user_text

    msgs = hf.get("messages") or []
    users = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
    if not users:
        raise RenderError("carried:no_user_turn")
    fi = users[-1]
    try:
        head, context, block, text = split_user_text(str(msgs[fi].get("content") or ""))
    except ValueError as e:
        raise RenderError(f"carried:user_turn {e}") from None
    previous = None
    if len(users) > 1:
        pi = users[-2]
        replies = [render.go_trim_space(str(m.get("content") or "")) for m in msgs[pi + 1:fi] if m.get("role") == "assistant"]
        replies = [r for r in replies if r]
        previous = (render.go_trim_space(str(msgs[pi].get("content") or "")), replies[-1] if replies else "")
    return {"index": fi, "head": head, "context": context, "block": block, "text": text, "previous": previous,
            "history_assistant_turns": sum(1 for m in msgs[:fi] if m.get("role") == "assistant")}


def rerender_carried(hf: dict, fields, system_text: str, tools: list | None) -> tuple[dict, dict]:
    """An earlier round's accepted row (retrain/carry.py) on THIS build's manifest: a copy of
    `hf` whose system turn is `system_text`, whose tools are `tools` and whose scored user turn
    carries the candidate-fields block `fields` computes for it now (the same turn inputs —
    context, text, previous exchange — as the row was collected with). Everything the teacher
    said, and every tool result, is kept byte for byte (the carry stage re-checked them on this
    manifest's engine). Returns (row, notes {block_chars, block_changed, history_assistant_turns})."""
    from .fields import join_user_text

    t = hf_final_turn(hf)
    row = json.loads(json.dumps(hf))
    msgs = row["messages"]
    if not msgs or msgs[0].get("role") != "system":
        raise RenderError("carried:no_system_turn")
    msgs[0]["content"] = system_text
    block = fields.block(t["context"], t["text"], t["previous"]) if fields is not None else ""
    msgs[t["index"]]["content"] = join_user_text(t["head"], block, t["text"])
    if tools is not None:
        row["tools"] = tools
    return row, {"block_chars": len(block), "block_changed": render.go_trim_space(block) != render.go_trim_space(t["block"]),
                 "history_assistant_turns": t["history_assistant_turns"]}


def _nth_user_index(msgs: list[dict], n: int) -> int:
    seen = -1
    for i, m in enumerate(msgs):
        if m["role"] == "user" and _has_text(m):
            seen += 1
            if seen == n:
                return i
    return len(msgs)


# --- build ------------------------------------------------------------------------------------------------------------

def estimate_tokens(hf: dict) -> int:
    n = len(json.dumps({"messages": hf.get("messages", []), "tools": hf.get("tools", [])}, ensure_ascii=False).encode("utf-8"))
    return int(math.ceil(n / BYTES_PER_TOKEN))


def rank_key(row: dict) -> tuple:
    """A carried row (retrain/carry.py: the row an earlier round trained on) first, then the fewest
    browser ops, fewest server calls, shortest reply, earliest pass."""
    ops = [c for c in row.get("tool_calls") or [] if c.get("name") not in SERVER_TOOLS]
    server = sum(1 for c in row.get("tool_calls") or [] if c.get("name") in SERVER_TOOLS)
    return (0 if isinstance(row.get("carried"), dict) else 1, len(ops), server, len(str(row.get("reply") or "")),
            str(row.get("_pass") or ""))


def keep_reasons(case: dict, row: dict, engine_fp: str | None) -> list[str]:
    r: list[str] = []
    if case.get("holdout"):
        r.append("holdout")
    if row.get("status") != "ok":
        r.append("error")
    if any(h.get("status") != "ok" for h in row.get("history_rows") or []):
        r.append("history_error")
    sc = row.get("score") or {}
    if sc.get("rule_version") != RULE_VERSION:
        r.append("rule_version")
    if sc.get("verdict") not in PASSING:
        r.append("not_pass")
    flags = set(sc.get("flags") or [])
    if row.get("status") == "ok":
        flags |= set(flags_of(case, row, row.get("history_rows")))
    if sc.get("history_diverged"):
        flags.add("history_diverged")
    r += sorted(flags)
    fp = (row.get("meta") or {}).get("engine_fingerprint")
    if engine_fp and fp and fp != engine_fp:
        r.append("engine")
    return r


def cap_rows(rows: list[dict], cases: dict[str, dict], *, section_cap: float, scope: set[str],
             family_max_share: float) -> tuple[list[dict], Counter, Counter]:
    from .dataset import family_of

    dropped_section: Counter = Counter()
    dropped_family: Counter = Counter()
    scoped = [r for r in rows if r["meta"]["family"] in scope]
    keep = list(rows)
    if section_cap and scoped:
        limit = max(1, int(math.floor(section_cap * len(scoped))))
        by: dict[str, list[dict]] = defaultdict(list)
        for r in scoped:
            by[family_of(cases[r["meta"]["case_id"]])].append(r)
        cut: set[str] = set()
        for k, rs in by.items():
            if len(rs) > limit:
                for r in sorted(rs, key=lambda x: hrank(x["meta"]["case_id"]))[limit:]:
                    cut.add(r["id"])
                    dropped_section[k] += 1
        keep = [r for r in keep if r["id"] not in cut]
    if family_max_share and keep:
        for _ in range(20):
            total = len(keep)
            counts = Counter(r["meta"]["family"] for r in keep)
            over = {f: n for f, n in counts.items() if n > family_max_share * total}
            if not over:
                break
            others = total - sum(over.values())
            # the largest size each over-share family may keep so it holds max_share of the new total
            allowed = int(math.floor(family_max_share * others / (1 - family_max_share * len(over)))) if len(over) * family_max_share < 1 else min(over.values())
            cut = set()
            for f in over:
                rs = sorted((r for r in keep if r["meta"]["family"] == f), key=lambda x: hrank(x["meta"]["case_id"]))
                for r in rs[max(allowed, 1):]:
                    cut.add(r["id"])
                    dropped_family[f] += 1
            keep = [r for r in keep if r["id"] not in cut]
    return keep, dropped_section, dropped_family


def build(cases_path: Path, logs: list[Path], manifest: Path, out: Path, *, why: Path | None = None,
          val_pct: int = 8, section_cap: float = 0.05, section_cap_scope: Iterable[str] = ("F0", "F1"),
          family_max_share: float = 0.40, max_seq: int = MAX_SEQ, count: Callable[[dict], int] | None = None,
          engine_fp: str | None = None, progress=print) -> dict:
    from . import fields as F
    from .dataset import hf_tools
    from .leak import ExamIndex
    from .r4collect import load_cases
    from .serving import unservable

    count = count or estimate_tokens
    prefix = F.Prefix(manifest)
    fields = F.Fields(manifest)
    system_text = prefix.rules + prefix.index
    tools = hf_tools(prefix.tools)
    # the logged rows, grouped by case (the logs are large: stream them)
    by_case: dict[str, list[dict]] = defaultdict(list)
    for p in logs:
        for row in read_jsonl(p):
            row["_pass"] = Path(p).stem
            by_case[str(row.get("case_id"))].append(row)
    cases = {c["id"]: c for c in load_cases(cases_path, set(by_case), include_holdout=True)}
    progress(f"r4dataset: {sum(len(v) for v in by_case.values())} logged rows for {len(by_case)} cases from {len(logs)} log(s)")
    turns = []  # every scored turn's block, computed in one CLI process (one spawn per row otherwise)
    for samples in by_case.values():
        for row in samples:
            if row.get("status") == "ok" and isinstance(row.get("carried"), dict):
                try:
                    t = hf_final_turn(row["carried"].get("row") or {})
                    turns.append((t["context"], t["text"], t["previous"]))
                except RenderError:
                    pass  # rejected below
            elif row.get("status") == "ok":
                hist = row.get("history_rows") or []
                prev = previous_exchange(str(hist[-1].get("text") or ""), hist[-1]) if hist else None
                turns.append((row.get("context"), row.get("text") or "", prev))
    fields.prime(turns)
    index = ExamIndex(why=why)
    rejects: list[dict] = []
    first: Counter = Counter()
    anyr: Counter = Counter()
    stats: dict[str, Any] = {"block_drift": 0, "block_drift_rejected": 0, "sources": Counter(), "no_case": 0}
    rows: list[dict] = []
    for cid, samples in by_case.items():
        case = cases.get(cid)
        if case is None:
            stats["no_case"] += 1
            continue
        best = None
        for row in sorted(samples, key=rank_key):
            reasons = keep_reasons(case, row, engine_fp)
            hf = None
            if not reasons:
                carried = row.get("carried") if isinstance(row.get("carried"), dict) else None
                try:
                    if carried is not None:
                        crow, notes = rerender_carried(carried.get("row") or {}, fields, system_text, tools)
                        notes.update(source=CARRIED_SOURCE, block_drift=False)
                    else:
                        src, notes = student_source(case, row, fields)
                except RenderError as e:
                    if carried is not None:
                        reasons = ["carried_render"]
                    else:
                        reasons = [str(e).split(":")[0] if str(e).startswith("no_conversation") else "conversation_mismatch"]
                    notes = {"error": str(e)}
                if not reasons:
                    if carried is not None:
                        messages = crow["messages"]
                        info = {"tools_called": [c["function"]["name"] for m in messages if m.get("role") == "assistant"
                                                 for c in m.get("tool_calls") or []], "missing_tool_results": 0}
                    else:
                        messages, info = to_hf_messages(src, system_text)
                    if info["missing_tool_results"]:
                        reasons.append("missing_tool_results")
                    if notes.get("block_drift"):
                        # the teacher read another block than the one this row would train (int's service is not
                        # the f79753b code the lab renders with): never trained, and the build reports it
                        reasons.append("block_drift")
                    if any(m["role"] == "assistant" and mentions_preview(m.get("content")) for m in messages):
                        reasons.append("preview_claim")  # the stored text, by the one rule (r4score.PREVIEW)
                    if unservable(messages):
                        reasons.append("unservable")
                    texts = [h.get("text") or "" for h in row.get("history_rows") or []] + [row.get("text") or ""]
                    leak = index.check({**case, "request": texts[-1], "history": [{"user": t} for t in texts[:-1]], "turns": []})
                    if leak["rule1"] or leak["rule2"]:
                        reasons.append("leak")
                    hf = {"id": f"{cid}#{row.get('_pass')}", "messages": messages, "tools": tools,
                          "meta": {"case_id": cid, "family": case["meta"]["family"], "mode": case["score"]["mode"],
                                   "group": case["meta"].get("group") or cid, "pass": row.get("_pass"),
                                   "model": row.get("model"), "models": row.get("models"), "kind": case.get("kind"),
                                   "target": case.get("target"), "template": case.get("template"),
                                   "manifest_hash": case.get("manifest_hash"), "engine_fingerprint": case.get("engine_fingerprint"),
                                   "verdict": (row.get("score") or {}).get("verdict"),
                                   "fields_context": bool(notes.get("block_chars")), "fields_block_chars": notes.get("block_chars"),
                                   "history_assistant_turns": notes["history_assistant_turns"],
                                   "history_turns": len(row.get("history_rows") or []), "render_source": notes["source"],
                                   "block_drift": notes.get("block_drift"), "leak_max_jaccard": leak["max_jaccard"],
                                   "tools_called": info["tools_called"], "turn_id": row.get("turn_id"), "at": row.get("at"),
                                   "holdout": False}}
                    if carried is not None:
                        src_meta = (carried.get("row") or {}).get("meta") or {}
                        hf["meta"]["carried"] = {"from": carried.get("source"), "row": (carried.get("row") or {}).get("id"),
                                                 "pass": src_meta.get("pass"), "block_changed": notes.get("block_changed")}
                        hf["meta"]["turn_id"], hf["meta"]["at"] = src_meta.get("turn_id"), src_meta.get("at")
                    if count(hf) > max_seq:
                        reasons.append("over_length")
            if "block_drift" in reasons:
                stats["block_drift_rejected"] += 1
            if not reasons and hf is not None:
                best = hf
                stats["sources"][hf["meta"]["render_source"]] += 1
                stats["block_drift"] += int(bool(hf["meta"]["block_drift"]))
                break
            rejects.append({"id": f"{cid}#{row.get('_pass')}", "case_id": cid, "family": case["meta"]["family"],
                            "reasons": reasons, "verdict": (row.get("score") or {}).get("verdict")})
            for r in reasons:
                anyr[r] += 1
            if reasons:
                first[reasons[0]] += 1
        if best is not None:
            rows.append(best)
    kept, cut_section, cut_family = cap_rows(rows, cases, section_cap=section_cap, scope=set(section_cap_scope),
                                             family_max_share=family_max_share)
    train, val = [], []
    for r in sorted(kept, key=lambda x: x["id"]):
        (val if h100(r["meta"]["group"]) < val_pct else train).append(r)
    kept_ids = {r["id"] for r in kept}
    # usable rows the caps cut: never trained, but kept so the next round can carry them (retrain/carry.py)
    # instead of re-asking the teacher for rows the same caps would cut again
    capped = [r for r in sorted(rows, key=lambda x: x["id"]) if r["id"] not in kept_ids]
    data = out / "dataset"
    data.mkdir(parents=True, exist_ok=True)
    for name, rs in (("train.jsonl", train), ("val.jsonl", val), ("rejects.jsonl", rejects), (CAPPED_FILE, capped)):
        with (data / name).open("w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copyfile(manifest, out / "manifest.json")
    toks = sorted(count(r) for r in train)
    fam = Counter(r["meta"]["family"] for r in kept)
    st = {
        "build": BUILD_VERSION, "cases_file": str(cases_path), "logs": [str(p) for p in logs],
        "cases_logged": len(by_case), "rows_kept_before_caps": len(rows), "rows": len(kept), "train": len(train), "val": len(val),
        "by_family": dict(sorted(fam.items())), "by_family_share": {k: round(v / max(len(kept), 1), 3) for k, v in sorted(fam.items())},
        "by_mode": dict(Counter(r["meta"]["mode"] for r in kept)), "multi_turn_rows": sum(1 for r in kept if r["meta"]["history_turns"]),
        "rejects_first_reason": dict(first), "rejects_any_reason": dict(anyr),
        "capped_section": dict(cut_section), "capped_family": dict(cut_family),
        "section_cap": section_cap, "section_cap_scope": sorted(section_cap_scope), "family_max_share": family_max_share,
        "render_sources": dict(stats["sources"]), "block_drift_rows": stats["block_drift"],
        "carried_rows": sum(1 for r in kept if r["meta"].get("carried")),
        "carried_by_source": dict(Counter(str(r["meta"]["carried"].get("from")) for r in kept if r["meta"].get("carried"))),
        "asked_rows": sum(1 for r in kept if not r["meta"].get("carried")),
        "block_drift_rejected": stats["block_drift_rejected"], "no_case": stats["no_case"],
        "rows_with_block": sum(1 for r in kept if r["meta"]["fields_context"]),
        "preview_claim_kept": sum(1 for r in kept for m in r["messages"] if m["role"] == "assistant" and mentions_preview(m["content"])),
        "tokens_train": {"total": sum(toks), "median": toks[len(toks) // 2] if toks else 0, "max": toks[-1] if toks else 0,
                         "counter": getattr(count, "label", "estimate: bytes ÷ 3")},
        "val_pct": val_pct, "max_seq": max_seq,
    }
    (data / "stats.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    fid = fields.identity()
    man = {"build": BUILD_VERSION, "rule_version": RULE_VERSION, "manifest_hash": prefix.manifest_hash,
           "manifest_hashes": [prefix.manifest_hash], "engine_fingerprint": engine_fp,
           "fields_context": {"enabled": True, "mode": "on", **fid, "rows": len(kept), "rows_with_block": st["rows_with_block"]},
           "system_prompt": prefix.identity(), "counts": {"train": len(train), "val": len(val), "rejects": len(rejects)},
           "logs": st["logs"], "cases_file": str(cases_path), "history": "stable (prompt.StableHistory), thinking dropped",
           "render": "deployed messages-compatible request (r4dataset)"}
    (data / "dataset-manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return st


def render_check(cases_path: Path, logs: list[Path], manifest: Path) -> dict:
    """Render EVERY answered row of the logs (whatever its verdict) the way the
    build would and report {rows, ok_rows, rendered, render_sources,
    block_drift_rows, errors}: `collect.sh smoke-check`'s test that int's
    service sent the teacher what the lab renders. A row that fails its
    verdict is still a rendering sample; only an unanswered row (status ≠ ok)
    is not."""
    from . import fields as F
    from .r4collect import load_cases

    fields = F.Fields(manifest)
    rows = [r for p in logs for r in read_jsonl(p)]
    cases = {c["id"]: c for c in load_cases(cases_path, {str(r.get("case_id")) for r in rows}, include_holdout=True)}
    ok = [r for r in rows if r.get("status") == "ok"]
    turns = []
    for row in ok:
        hist = row.get("history_rows") or []
        prev = previous_exchange(str(hist[-1].get("text") or ""), hist[-1]) if hist else None
        turns.append((row.get("context"), row.get("text") or "", prev))
    fields.prime(turns)
    out: dict[str, Any] = {"rows": len(rows), "ok_rows": len(ok), "rendered": 0, "render_sources": Counter(),
                           "block_drift_rows": 0, "errors": Counter(), "by_case": []}
    for row in ok:
        case = cases.get(str(row.get("case_id")))
        rec = {"case_id": row.get("case_id"), "verdict": (row.get("score") or {}).get("verdict")}
        if case is None:
            out["errors"]["no_case"] += 1
            rec["error"] = "no_case"
        else:
            try:
                _, notes = student_source(case, row, fields)
                out["rendered"] += 1
                out["render_sources"][notes["source"]] += 1
                out["block_drift_rows"] += int(bool(notes.get("block_drift")))
                rec.update(source=notes["source"], block_drift=notes.get("block_drift"),
                           history_assistant_turns=notes.get("history_assistant_turns"))
            except RenderError as e:
                out["errors"][str(e).split(":")[0] if str(e).startswith("no_conversation") else str(e)] += 1
                rec["error"] = str(e)
        out["by_case"].append(rec)
    out["render_sources"] = dict(out["render_sources"])
    out["errors"] = dict(out["errors"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="souschef_lab.r4dataset", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--log", type=Path, action="append", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="the data set folder (e.g. data/r4); not with --render-check")
    ap.add_argument("--render-check", action="store_true",
                    help="render every answered row whatever its verdict and print the report (no data set written)")
    ap.add_argument("--why", type=Path, help="the paraphrase study folder (the final hold-out check reads its real turns)")
    ap.add_argument("--val-pct", type=int, default=8)
    ap.add_argument("--section-cap", type=float, default=0.05)
    ap.add_argument("--section-cap-scope", default="F0,F1")
    ap.add_argument("--family-max-share", type=float, default=0.40)
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--tokenizer", help="a local HF tokenizer (e.g. unsloth/Qwen3.5-9B) for exact counts on the served render")
    ap.add_argument("--engine", type=Path, help="headless.mjs: rows measured by another engine are rejected `engine`")
    a = ap.parse_args(argv)
    if a.render_check:
        rep = render_check(a.cases, a.log, a.manifest)
        print(json.dumps(rep, indent=1))
        return 0
    if a.out is None:
        ap.error("--out is required (except with --render-check)")
    count = None
    if a.tokenizer:
        from .dataset import tokenizer_counter
        count, label = tokenizer_counter(a.tokenizer)
        count.label = label  # type: ignore[attr-defined]
    fp = None
    if a.engine:
        from .engine import engine_fingerprint
        fp = (engine_fingerprint(a.engine) or {}).get("fingerprint")
    st = build(a.cases, a.log, a.manifest, a.out, why=a.why, val_pct=a.val_pct, section_cap=a.section_cap,
               section_cap_scope=[x.strip() for x in a.section_cap_scope.split(",") if x.strip()],
               family_max_share=a.family_max_share, max_seq=a.max_seq, count=count, engine_fp=fp,
               progress=lambda s: print(s, file=sys.stderr))
    print(json.dumps(st, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
