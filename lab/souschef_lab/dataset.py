"""Dataset builder (LAB.md §5): keep rules, thermometer-compatible training
rows, splits and the build report.

Keep rules — re-verified from the events on every build, never the verdict
the collector logged beside a row (thermometer build_dataset.py's rule):

  error          row status ≠ ok (an HTTP error, a 409 superseded, a timeout)
  messages_mismatch  the row's stored `messages` differ from the messages rebuilt
                 from its events (render.messages_from_events) — a log written
                 by an older collector; the row never trains on stale bytes
  stop_reason    the turn's stop ∉ {end_turn}
  not_pass       verdict ∉ {pass, clarify_ok, unsupported_ok}
  warnings       a W_HIDDEN on a replayed op that the case's own expected ops do not emit
  empty_reply    no final text after the last batch
  lookup_storm   more than LOOKUP_STORM server-tool calls in one turn
  holdout        the case is held out (exam rows never train)
  duplicate      the same (case_id, canonical ops) was already kept
  over_length    tokens > max_seq (the base tokenizer when [train] is
                 installed and --tokenizer names it, else BYTES_PER_TOKEN)
  escalated      only with --drop-escalated: the turn was handed to the
                 fallback route (hybrid, README §4)

Candidate fields (`--fields-context`, default on; LAB.md §5): the served
student's route puts a `<candidate_fields>` block in the user turn (README §5
`fields_context`), so its training rows must carry it. With the option on,
every row's user turn is rendered with the block the service's own code
computes for that row's context and text NOW (fields.py → the
`sous-chef-fields` CLI; needs go) — whatever route the row was collected on:
a Claude teacher row was sent none, and gets one. Off renders none, even for
a row logged with one. `dataset-manifest.json` records which
(`fields_context`), and train.py copies it into run.json, so an adapter
trained with the block is distinguishable from one trained without.

Hybrid rows: a row whose turn escalated (`row.escalated`, collect.py) was
finished by the FALLBACK route's model. It is kept as a teacher row and
flagged — `meta.escalated`, `meta.escalation_reason`, `meta.model` = the
route that finished it, `meta.requested_model` = the student route that was
asked — never counted as the student's; `stats.escalated` has the counts and
`--drop-escalated` rejects them instead.

Row format = thermometer lab/hf.py to_hf (copied, attribution in place):
``{id, messages:[system, user, assistant{content, tool_calls:[{id, type,
function:{name, arguments:<json string>}}]}, tool{tool_call_id, name,
content} …], tools:[{type:"function", function:{name, description,
parameters}}], meta}``. Arguments are JSON strings so every row has the
same column types (``datasets``/Arrow). Thinking is never available in a
trace and never fabricated (thermometer's reasoning_content is dropped —
Claude is the teacher, LAB.md fixed decisions).

The system message is ``system[0]`` (frozen rules) + ``system[1]`` (the
manifest index) joined with nothing between them (thermometer hf.join_system:
Ollama joins a request's system blocks that way); both come from fixtures
until the service exposes the rendered prompt (LAB.md §10).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from . import render
from .verify import (
    NON_OPS,
    PASSING,
    RULE_VERSION,
    Engine,
    RecordedEngine,
    Verdict,
    canonical,
    default_engine,
    events_of,
    read_jsonl,
    tool_calls,
    verify,
)

RULES = ("error", "messages_mismatch", "stop_reason", "not_pass", "warnings", "empty_reply", "lookup_storm", "holdout",
         "duplicate", "over_length")
OPTIONAL_RULES = ("escalated",)  # applied only when asked (--drop-escalated)
FIELDS_MODES = ("on", "off", "as-logged")
LOOKUP_STORM = 3
BYTES_PER_TOKEN = 3.0  # LAB.md §5: the fallback when no tokenizer is installed; train.py counts exactly
DEFAULT_MAX_SEQ = 32768
MANIFEST_NAME = "dataset-manifest.json"
RULE_SET_VERSION = f"{RULE_VERSION}+dataset-1"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- keep rules -------------------------------------------------------------------------------------------------------

def unexpected_hidden(warnings: Iterable[str], case: dict | None) -> list[str]:
    """The W_HIDDEN warnings the expected ops did NOT also produce. The engine
    emits W_HIDDEN for every gated field whether the gate is open or closed
    (cookie-cutter validate.js `isGated`; measured 2026-10-01), so a warning
    the case's measured expected ops share (`meta.measured.warnings`) is the
    field's nature, not the model writing a hidden field."""
    expected = {str(w) for w in (((case or {}).get("meta") or {}).get("measured") or {}).get("warnings") or []}
    return [str(w) for w in warnings if str(w).startswith("W_HIDDEN") and str(w) not in expected]


def reasons_of(row: dict, case: dict | None, verdict: Verdict, max_seq: int | None = None,
               tokens: Callable[[dict], int] | None = None, hf: dict | None = None) -> list[str]:
    """The keep rules a row fails, in RULES order (``duplicate`` is a build
    rule — it needs the other kept rows — and is added by ``build``)."""
    reasons: list[str] = []
    if case is None:
        return ["no_case"]
    if str(row.get("status") or "") != "ok" or row.get("error"):
        reasons.append("error")
    if not stored_messages_match(row, case):
        reasons.append("messages_mismatch")
    if verdict.stop != "end_turn":
        reasons.append("stop_reason")
    if verdict.verdict not in PASSING:
        reasons.append("not_pass")
    if unexpected_hidden(verdict.warnings, case):
        reasons.append("warnings")
    if verdict.final_text_chars == 0:
        reasons.append("empty_reply")
    if verdict.server_calls > LOOKUP_STORM:
        reasons.append("lookup_storm")
    if case.get("holdout"):
        reasons.append("holdout")
    if max_seq is not None:
        count = tokens or estimate_tokens
        if count(hf if hf is not None else to_hf(row, case)) > max_seq:
            reasons.append("over_length")
    return reasons


# --- messages ---------------------------------------------------------------------------------------------------------

# The user block and the tool_result contents are rendered by render.py — the one place that matches the service's
# prompt.ContextBlock / loop.browserResult byte for byte (collect.py logs the same bytes).
context_block = render.context_block
messages_from_events = render.messages_from_events


def rebuild_messages(row: dict, case: dict | None = None, candidates: str | None = None) -> list[dict]:
    """The turn's Anthropic messages rebuilt from ``row.events`` with the
    context and text the turn was asked with (render.turn_inputs) and — by
    default — the candidate-fields block it was SENT with
    (``row.candidate_fields``; none for most rows). This is the ONLY source
    ``to_hf`` reads: a row's stored ``messages`` are a cache
    (``stored_messages_match``), never the bytes that train. ``candidates``
    overrides the block (``""`` = none): the training render of a row for a
    route that sends another block than the one the row was collected with."""
    context, text = render.turn_inputs(row, case)
    block = render.turn_candidates(row) if candidates is None else candidates
    return messages_from_events(events_of(row), context, text, block)


def stored_messages_match(row: dict, case: dict | None = None, rebuilt: list[dict] | None = None) -> bool:
    """True when the row carries no ``messages`` or they equal the rebuilt
    ones exactly. False = the log predates the renderer (or the events are
    missing their ``tool.result``s): the ``messages_mismatch`` keep rule."""
    stored = row.get("messages")
    if not stored:
        return True
    return stored == (rebuilt if rebuilt is not None else rebuild_messages(row, case))


def hf_tools(tools: list[dict] | None) -> list[dict]:
    """Anthropic tool definitions in function-calling form (thermometer
    lab/hf.py hf_tools, copied)."""
    return [
        {"type": "function", "function": {
            "name": t.get("name", ""),
            "description": t.get("description", "") or "",
            "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
        }}
        for t in tools or [] if isinstance(t, dict)  # a collector row's `tools` may be the ten NAMES only (LAB.md §4)
    ]


def join_system(parts: list[str] | tuple[str, ...]) -> str:
    """Joined with nothing between them (thermometer lab/hf.py join_system)."""
    return "".join(parts)


def _block_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("type") == "text")
    return "" if content is None else str(content)


def to_hf(row: dict, case: dict | None = None, system: list[str] | tuple[str, ...] = (),
          tools: list[dict] | None = None, candidates: str | None = None) -> dict:
    """One training row (thermometer lab/hf.py to_hf, adapted: Anthropic
    messages ALWAYS rebuilt from ``row.events`` by render.py — never read
    from ``row.messages``, which may predate the renderer; no thinking; tool
    results' content is the posted JSON). ``candidates``: the
    candidate-fields block of the user turn — ``None`` = the one the row was
    logged with, ``""`` = none, else the block to train with
    (``rebuild_messages``). An escalated row (hybrid) is flagged as the
    fallback route's: ``meta.model`` is the route that finished the turn."""
    from .collect import completed_by, escalation_of  # lazy: collect imports nothing from here, keep it one-way

    case = case or {}
    messages: list[dict] = []
    sys_text = join_system(system)
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    block = render.turn_candidates(row) if candidates is None else render.go_trim_space(candidates)
    source = rebuild_messages(row, case, block)
    escalation = escalation_of(row)
    answer_start = len(messages) + 1  # after the system message and the one user message
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
    tool_uses = [c for m in messages if m.get("role") == "assistant" for c in m.get("tool_calls") or []]
    browser = [c for c in tool_uses if c["function"]["name"] not in NON_OPS]
    target = case.get("target") or {}
    return {
        "id": str(row.get("id") or ""),
        "messages": messages,
        "tools": hf_tools(tools if tools is not None else row.get("tools")),
        "meta": {
            "model": completed_by(row) if escalation else row.get("model"),
            "requested_model": row.get("model"), "escalated": bool(escalation),
            "escalation_reason": (escalation or {}).get("reason"),
            "fields_context": bool(block), "fields_block_chars": len(block),
            "case_id": row.get("case_id") or case.get("id"), "sample": row.get("sample"),
            "kind": case.get("kind"), "target": target, "family": family_of(case),
            "field_type": (case.get("meta") or {}).get("field_type"), "holdout": bool(case.get("holdout")),
            "manifest_hash": case.get("manifest_hash"), "engine_fingerprint": case.get("engine_fingerprint"),
            "template": case.get("template"),
            "tools_called": [c["function"]["name"] for c in tool_uses], "calls": len(browser),
            "missing_tool_results": sum(1 for c in tool_uses if c["id"] not in answered),
            "at": row.get("at"), "turn_id": row.get("turn_id"),
            # no history in v1 (every sample is a fresh conversation, LAB.md §4)
            "history_assistant_turns": 0, "answer_start": answer_start,
        },
    }


def family_of(case: dict) -> str:
    """The family a case counts in for the cap (LAB.md §5): the section type
    (``meta.section_type``, else the section id minus ``-\\d+``), else the
    global panel, else branding / the page; the kinds without one container
    (clarify, unsupported, multi, layout) count by kind."""
    meta = case.get("meta") or {}
    if meta.get("section_type"):
        return f"section:{meta['section_type']}"
    kind = str(case.get("kind") or "")
    if kind in ("clarify", "unsupported", "multi", "layout"):
        return f"kind:{kind}"
    t = case.get("target") or {}
    if t.get("kind") == "section" and t.get("sectionId"):
        return "section:" + re.sub(r"-\d+$", "", str(t["sectionId"]))
    if t.get("kind") == "global":
        return f"panel:{t.get('panel') or 'global'}"
    if t.get("kind") in ("branding", "page"):
        return f"{t['kind']}:{t.get('page') or ''}".rstrip(":")
    return f"kind:{kind}"


def row_bytes(hf: dict) -> int:
    return len(json.dumps({"messages": hf.get("messages", []), "tools": hf.get("tools", [])},
                          ensure_ascii=False).encode("utf-8"))


def estimate_tokens(hf: dict) -> int:
    """BYTES_PER_TOKEN bytes per token over the messages and tools (LAB.md §5
    fallback), rounded up."""
    return math.ceil(row_bytes(hf) / BYTES_PER_TOKEN)


def split_of(case_id: str, val_pct: int = 10) -> str:
    """"val" for about val_pct % of cases by a hash of the case id
    (thermometer build_dataset.split_of, on the case id instead of the
    question text: paraphrases of one case never straddle)."""
    h = int(hashlib.sha1(str(case_id).encode("utf-8")).hexdigest()[:8], 16)
    return "val" if h % 100 < val_pct else "train"


def ops_key(row: dict) -> str:
    """What makes two samples the same example: the case and the canonical
    browser ops (the ``duplicate`` rule)."""
    ops = [{"op": c["name"], **c["input"]} for c in tool_calls(events_of(row)) if c["name"] not in NON_OPS]
    return canonical([row.get("case_id"), ops])


# --- build ------------------------------------------------------------------------------------------------------------

@dataclass
class BuildConfig:
    val_pct: int = 10
    max_seq: int = DEFAULT_MAX_SEQ
    best_of_n: bool = False
    system: tuple[str, ...] = ()
    tools: list[dict] | None = None
    tokens: Callable[[dict], int] = estimate_tokens
    token_method: str = f"estimate: bytes ÷ {BYTES_PER_TOKEN:g}"
    family_share: float = 0.03
    logs: list[str] = field(default_factory=list)
    # Candidate fields: "on" = every row rendered with the block `fields` (a fields.Fields) computes; "off" = none;
    # "as-logged" = the block each row was collected with (the library default; the CLI's is on).
    fields_context: str = "as-logged"
    fields: object | None = None
    drop_escalated: bool = False


@dataclass
class Build:
    train: list[dict] = field(default_factory=list)
    val: list[dict] = field(default_factory=list)
    exam: list[dict] = field(default_factory=list)
    rejects: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)


def build(rows: list[dict], cases: dict[str, dict], engine: Engine, cfg: BuildConfig | None = None) -> Build:
    """Judge every row (bestofn.Judge: verify + keep rules), apply the
    duplicate rule, then keep all passing rows — or, with best_of_n, the rows
    bestofn.select picks — split by case id, exam = the holdout cases."""
    from .bestofn import Judge, SelectConfig, select, selection_meta  # lazy: bestofn imports this module

    from .collect import escalation_of

    cfg = cfg or BuildConfig()
    if cfg.fields_context not in FIELDS_MODES:
        raise ValueError(f"fields_context must be one of {FIELDS_MODES}")
    if cfg.fields_context == "on":
        if cfg.fields is None:
            raise ValueError("fields_context on needs BuildConfig.fields (fields.Fields on the cases' manifest)")
        cfg.fields.prime(render.turn_inputs(r, cases.get(str(r.get("case_id") or ""))) for r in rows)  # one process
    judge = Judge(cases, engine, system=cfg.system, tools=cfg.tools, max_seq=cfg.max_seq, tokens=cfg.tokens,
                  fields_context=cfg.fields_context, fields=cfg.fields)
    judged = [judge(r) for r in rows]
    if cfg.drop_escalated:
        for j in judged:
            if escalation_of(j.row):
                j.reasons.append("escalated")
                j.passed = False
    seen: set[str] = set()
    for j in sorted(judged, key=lambda j: (j.case_id, j.sample, j.sample_id)):
        if j.passed:
            k = ops_key(j.row)
            if k in seen:
                j.reasons.append("duplicate")
                j.passed = False
            else:
                seen.add(k)
    out = Build()
    for j in judged:
        if not j.passed:
            out.rejects.append({"id": j.sample_id, "case_id": j.case_id, "model": j.model, "reasons": list(j.reasons),
                                "verdict": j.verdict.verdict if j.verdict else None})
    if cfg.best_of_n:
        kept, report = select(judged, lambda j: cfg.tokens(j.hf), SelectConfig(max_seq=cfg.max_seq,
                                                                                 family_share=cfg.family_share))
        chosen = [(k.judged, selection_meta(k)) for k in kept]
    else:
        report = {"samples": len(judged), "passing": sum(1 for j in judged if j.passed)}
        chosen = [(j, None) for j in judged if j.passed]
    for j, sel in chosen:
        hf = json.loads(json.dumps(j.hf))
        if sel is not None:
            hf["meta"]["selection"] = sel
        (out.val if split_of(j.case_id, cfg.val_pct) == "val" else out.train).append(hf)
    out.exam = [c for cid, c in sorted(cases.items()) if c.get("holdout")]
    mismatched = sorted(j.sample_id for j in judged if "messages_mismatch" in j.reasons)
    primary = Counter(r["reasons"][0] for r in out.rejects if r["reasons"])
    every = Counter(x for r in out.rejects for x in r["reasons"])
    by_kind: dict[str, Counter] = {}
    for j in judged:
        if j.verdict is not None:
            by_kind.setdefault(j.kind, Counter())[j.verdict.verdict] += 1
    hashes = sorted({str(c.get("manifest_hash")) for c in cases.values() if c.get("manifest_hash")})
    # Engine identity (LAB.md §2): the fingerprint(s) the cases' expected diffs were measured with, and the engine
    # that replayed the rows in this build. A case measured with another engine is flagged, never silently kept.
    case_fps = sorted({str(c.get("engine_fingerprint")) for c in cases.values() if c.get("engine_fingerprint")})
    build_fp = getattr(engine, "fingerprint", None)
    stale = sorted(cid for cid, c in cases.items()
                   if build_fp and c.get("engine_fingerprint") and c.get("engine_fingerprint") != build_fp)
    out.stats = {
        "built_at": utc_now(), "rows_read": len(rows), "cases": len(cases), "holdout_cases": len(out.exam),
        "kept": len(out.train) + len(out.val), "train": len(out.train), "val": len(out.val),
        "rejected": len(out.rejects), "rejects_by_reason": dict(primary), "rejects_any_reason": dict(every),
        "verdicts_by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
        "tokens": {"train": token_stats(out.train, cfg.max_seq, cfg.tokens), "val": token_stats(out.val, cfg.max_seq, cfg.tokens),
                   "budget": cfg.max_seq, "method": cfg.token_method},
        "selection": report,
        "engine": {"fingerprint": build_fp, "case_fingerprints": case_fps, "stale_cases": len(stale),
                   "stale_sample": stale[:5]},
        "messages_mismatch": {"rows": len(mismatched), "sample": mismatched[:5]},
        "fields_context": fields_stats(cfg, judged, out.train + out.val),
        "escalated": escalated_stats(judged, out.train + out.val),
    }
    out.manifest = {
        "built_at": out.stats["built_at"], "mode": "best-of-n" if cfg.best_of_n else "all-kept",
        "logs": list(cfg.logs), "manifest_hashes": hashes, "engine_fingerprint": build_fp,
        "case_engine_fingerprints": case_fps, "rule_version": RULE_SET_VERSION, "keep_rules": list(RULES),
        "fields_context": {"enabled": out.stats["fields_context"]["rows_with_block"] > 0 if cfg.fields_context == "as-logged"
                           else cfg.fields_context == "on", "mode": cfg.fields_context,
                           **(cfg.fields.identity() if cfg.fields_context == "on" else {}),
                           "rows": out.stats["kept"], "rows_with_block": out.stats["fields_context"]["rows_with_block"]},
        "escalated_rows": {"kept": out.stats["escalated"]["kept"], "policy": "dropped" if cfg.drop_escalated
                           else "kept as teacher rows, flagged (meta.escalated, meta.model = the fallback route)"},
        "optional_rules": ["escalated"] if cfg.drop_escalated else [],
        "lookup_storm": LOOKUP_STORM, "max_seq": cfg.max_seq, "token_count": cfg.token_method, "val_pct": cfg.val_pct,
        "split": "sha1(case_id) % 100 < val_pct; exam = holdout cases", "thinking": "none (traces carry no thinking)",
        "system_prompt": {"blocks": len(cfg.system), "bytes": len(join_system(cfg.system).encode("utf-8"))},
        "counts": {"train": len(out.train), "val": len(out.val), "exam": len(out.exam), "rejects": len(out.rejects)},
    }
    return out


def fields_stats(cfg: BuildConfig, judged: list, kept: list[dict]) -> dict:
    """What the build did about the candidate-fields block: the mode, how
    many kept rows carry one, the block's size, and how many rows were
    LOGGED with a block that is not the one rendered now (``drift``: the
    retrieval or the manifest changed since the collection, or the mode
    overrode the log)."""
    chars = sorted(r["meta"]["fields_block_chars"] for r in kept if r["meta"].get("fields_block_chars"))
    logged = [j for j in judged if j.hf and "candidate_fields" in j.row]
    from .fields import first_user, split_user_text  # pure string helpers; no go needed

    def rendered(hf: dict) -> str:
        return split_user_text(str((first_user(hf) or {}).get("content") or ""))[2]

    drift = sorted(j.sample_id for j in logged if render.go_trim_space(str(j.row.get("candidate_fields") or "")) != rendered(j.hf))
    return {"mode": cfg.fields_context, "rows_with_block": len(chars), "rows_without_block": len(kept) - len(chars),
            "block_chars": {"p50": chars[len(chars) // 2], "p95": chars[min(len(chars) - 1, int(len(chars) * 0.95))],
                            "max": chars[-1]} if chars else None,
            "logged_with_block": len(logged), "drift": {"rows": len(drift), "sample": drift[:5]}}


def escalated_stats(judged: list, kept: list[dict]) -> dict:
    """Hybrid rows in the logs and in the kept set (flagged teacher rows)."""
    from .collect import escalation_of

    esc = [j for j in judged if escalation_of(j.row)]
    kept_esc = [r for r in kept if r["meta"].get("escalated")]
    return {"rows": len(esc), "kept": len(kept_esc),
            "reasons": dict(sorted(Counter(str(escalation_of(j.row).get("reason")) for j in esc).items())),
            "kept_by_route": dict(sorted(Counter(str(r["meta"].get("model")) for r in kept_esc).items()))}


def token_stats(rows: list[dict], budget: int, count: Callable[[dict], int] = estimate_tokens) -> dict:
    """thermometer build_dataset.token_stats, copied."""
    toks = sorted(count(r) for r in rows)
    if not toks:
        return {"rows": 0}
    return {"rows": len(toks), "mean": int(sum(toks) / len(toks)), "p50": toks[len(toks) // 2],
            "p90": toks[min(len(toks) - 1, int(len(toks) * 0.9))], "max": toks[-1],
            "over_budget": sum(1 for t in toks if t > budget), "total": sum(toks)}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_build(out_dir: Path, b: Build) -> None:
    write_jsonl(out_dir / "train.jsonl", b.train)
    write_jsonl(out_dir / "val.jsonl", b.val)
    write_jsonl(out_dir / "exam.jsonl", b.exam)
    write_jsonl(out_dir / "rejects.jsonl", b.rejects)
    (out_dir / "stats.json").write_text(json.dumps(b.stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / MANIFEST_NAME).write_text(json.dumps(b.manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def fields_for_cases(manifest: Path | None, cases_path: Path, cases: dict[str, dict]):
    """The fields.Fields for a build: on ``manifest``, else ``manifest.json``
    beside the cases file, else the lab's fixture snapshot — refused when its
    hash is not the cases' (the blocks would list another manifest's keys)."""
    from . import fields as mod  # lazy: only --fields-context on needs go
    from .collect import DEFAULT_MANIFEST, manifest_hash

    path = manifest or next((p for p in (cases_path.parent / "manifest.json",) if p.is_file()), DEFAULT_MANIFEST)
    if not Path(path).is_file():
        raise mod.FieldsError(f"manifest {path} not found (--manifest)")
    have = manifest_hash(json.loads(Path(path).read_text(encoding="utf-8")))
    want = {str(c.get("manifest_hash")) for c in cases.values() if c.get("manifest_hash")}
    if want and want != {have}:
        raise mod.FieldsError(f"{path} hashes to {have[:12]}… but the cases were generated from "
                              f"{', '.join(sorted(w[:12] + '…' for w in want))}: pass the cases' manifest with --manifest")
    return mod.Fields(path)


def tokenizer_counter(name: str | None) -> tuple[Callable[[dict], int], str]:
    """The exact count with a local Hugging Face tokenizer (needs the [train]
    extra), else the estimate. Imported lazily; never at module load."""
    if not name:
        return estimate_tokens, f"estimate: bytes ÷ {BYTES_PER_TOKEN:g}"
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as e:
        raise SystemExit("--tokenizer needs the [train] extra (pip install -e 'lab[train]')") from e
    from . import serving  # the row's training text is the served form (LAB.md §5), never the tokenizer's chat template

    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)

    def count(hf: dict) -> int:
        return len(tok(serving.render_training(hf["messages"], hf["tools"]), add_special_tokens=False)["input_ids"])

    return count, f"exact: {name}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", action="append", required=True, help="collector log (jsonl), repeatable")
    p.add_argument("--cases", required=True, help="cases.jsonl (LAB.md §2)")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--val-pct", type=int, default=10)
    p.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ)
    p.add_argument("--best-of-n", action="store_true", help="choose rows with bestofn.select")
    p.add_argument("--system-rules", help="frozen system[0] text file")
    p.add_argument("--system-index", help="system[1] manifest index text file")
    p.add_argument("--tools", help="the tool definitions (Anthropic shape, JSON list)")
    p.add_argument("--tokenizer", help="count tokens exactly with this local tokenizer ([train] extra)")
    p.add_argument("--recorded", help="replay recorded engine output (tests)")
    p.add_argument("--fields-context", choices=FIELDS_MODES, default="on",
                   help="the user turn's candidate-fields block: on = the block the service computes for every row "
                        "(the served student's prompt; needs go and --manifest), off = none, as-logged = what each "
                        "row was collected with (default: %(default)s)")
    p.add_argument("--manifest", type=Path,
                   help="the manifest the cases were generated from, for --fields-context on (default: manifest.json "
                        "beside --cases, else the lab's fixture snapshot)")
    p.add_argument("--drop-escalated", action="store_true",
                   help="reject rows whose turn escalated to the fallback route (default: keep them as flagged teacher rows)")
    args = p.parse_args(argv)
    if not 0 <= args.val_pct < 100:
        p.error("--val-pct must be 0–99")
    system = tuple(Path(f).read_text(encoding="utf-8") for f in (args.system_rules, args.system_index) if f)
    tools = json.loads(Path(args.tools).read_text(encoding="utf-8")) if args.tools else None
    count, method = tokenizer_counter(args.tokenizer)
    cases = {c["id"]: c for c in read_jsonl(args.cases)}
    rows = [r for path in args.log for r in read_jsonl(path)]
    engine: Engine = RecordedEngine(args.recorded) if args.recorded else default_engine()
    fields = None
    if args.fields_context == "on":
        try:
            fields = fields_for_cases(args.manifest, Path(args.cases), cases)
        except RuntimeError as e:  # fields.FieldsError
            print(f"cannot build with --fields-context on: {e}\n(--fields-context off builds rows without the "
                  f"candidate-fields block — the form adapters before 2026-10-04 were trained on)", file=sys.stderr)
            return 2
    cfg = BuildConfig(val_pct=args.val_pct, max_seq=args.max_seq, best_of_n=args.best_of_n, system=system, tools=tools,
                      tokens=count, token_method=method, logs=list(args.log), fields_context=args.fields_context,
                      fields=fields, drop_escalated=args.drop_escalated)
    b = build(rows, cases, engine, cfg)
    write_build(Path(args.out), b)
    s = b.stats
    if s["engine"]["stale_cases"]:
        print(f"warning: {s['engine']['stale_cases']} case(s) were measured with another engine "
              f"(fingerprints {', '.join(fp[:12] for fp in s['engine']['case_fingerprints'])} vs current "
              f"{(s['engine']['fingerprint'] or '')[:12]}): run `souschef-lab cases remeasure --cases {args.cases}`",
              file=sys.stderr)
    if s["messages_mismatch"]["rows"]:
        print(f"warning: {s['messages_mismatch']['rows']} row(s) carry stored `messages` that differ from the bytes rebuilt "
              f"from their events (a log written before the renderer, or events without tool.result) — rejected "
              f"messages_mismatch, e.g. {', '.join(s['messages_mismatch']['sample'])}; re-collect that log",
              file=sys.stderr)
    print(f"rows {s['rows_read']} · kept {s['kept']} (train {s['train']} / val {s['val']}) · exam cases {s['holdout_cases']} "
          f"· rejected {s['rejected']}")
    fc, esc = s["fields_context"], s["escalated"]
    print(f"  candidate fields {fc['mode']}: {fc['rows_with_block']} of {s['kept']} kept rows carry the block"
          + (f" (chars p50 {fc['block_chars']['p50']} / p95 {fc['block_chars']['p95']} / max {fc['block_chars']['max']})"
             if fc["block_chars"] else ""))
    if esc["rows"]:
        print(f"  escalated (hybrid) rows: {esc['rows']} in the logs, {esc['kept']} kept as flagged teacher rows "
              f"{esc['kept_by_route']}")
    for reason in [*RULES, *OPTIONAL_RULES, "no_case"]:
        if s["rejects_by_reason"].get(reason):
            print(f"  rejected · {reason:13} {s['rejects_by_reason'][reason]}")
    print(f"wrote {args.out}/train.jsonl, val.jsonl, exam.jsonl, rejects.jsonl, stats.json, {MANIFEST_NAME}")
    return 0 if s["kept"] else 1


if __name__ == "__main__":
    sys.exit(main())
