"""carry — carry an earlier round's accepted rows forward instead of re-asking the teacher (r6 plan §2b).

Every round used to re-ask the teacher for every case: r5 paid US$101.57 to re-collect r4's
pool, the teacher re-worded 97.6 % of the replies, 172 shared cases changed their writes and
30 / 24 cases dropped / joined by teacher luck — churn that is not the round's intended change.
This stage keeps the row a case was last TRAINED on (or, failing that, the row a parked round
accepted) whenever it still applies on the round's manifest, and re-asks the teacher only for
new, changed and failing cases.

Sources, in priority order (`find_sources`):
  1. the live model's dataset   state/model.json `dataset` (cases/pool/r4/dataset, cases/trained/<cid>/dataset)
                                 with its case file (`training_cases`, else the pool files trained_in its tag)
  2. parked rounds              state/parked/<cid>.json newer than the live model, newest first
                                 (runs/<run>/dataset + runs/<run>/prepare/cases.jsonl), at most MAX_PARKED
A case takes the first source whose row still applies; the live model's row wins, so an unchanged
case trains on exactly what the live model trained on.

A row applies when ALL of these hold (otherwise the case is re-asked; the first failing reason is
reported per case in carry-report.json):
  rules_changed         system[0] (the rules) of the source rows is the round's (Prefix(manifest).rules)
  case_changed          the case is the one the row was collected for: request, history (user texts,
                        expect, page, ops), kind, target, site_state, score mode and expected (minus the
                        engine-measured diffs) unchanged (rebase renames / relabels change these)
  turns_changed         the row has 1 + len(history) user turns
  block_parity          the row's candidate-fields block is reproduced on the SOURCE manifest from its
                        turn inputs (proves the re-render recomputes the same turn on the new manifest)
  target_changed        no field / list / container the row writes or the case expects changed its spec
                        (label, type, options, visibility …) between the source and the round's manifest
  server_result_stale   no lookup_fields / describe_target result names a container whose spec changed
  replay (oracle)       the row's own calls, replayed through r4collect.run_r4_case against a scripted
                        client on the round's engine and manifest: same texts and context sent
                        (text_changed / context_changed), the same tool results (tool_result_changed;
                        the order of `written` / `warnings` aside), the stream fully consumed (replay_mismatch), and
                        r4score's verdict usable on the round's case (not_usable:<verdict or flags>);
                        history turns must not diverge
  ref_written_changed   a tag / brand reference write the row recorded without its derived id: a stored tool result
                        whose `written` holds <…>.tagSlug (brandSlug) but not <…>.tagId (brandId) — the engine wrote
                        the id only once the browser handed it the store's tags and brands (the 2026-10-15 engine
                        context), so the tool result the model reads now differs; also the replay's tool result when
                        it differs from the stored one ONLY by those derived ids. Re-asked, never carried
  hidden_write          on by default (off: $COLLECT_CARRY_HIDDEN=keep or trigger.carry_hidden "keep"): no write to a
                        field whose gate is closed after the op (r4score.closed_gate_writes; a bare W_HIDDEN comes
                        with every gated write) that the case does not list in expected.expect_hidden — stricter than
                        r4score, which also accepts the warnings the case measured (meta.measured). Measured on r5's
                        inputs with the gate check: 64 of 2,688 carried rows write a closed-gate field, most of them
                        cases prepare already drops (their expectation is the hidden write); the rest are re-asked.
                        With it off the rows are carried and counted (`carried_hidden_write`)
  excluded              meta.no_carry on the case, or its id in the run's carry_exclude list

Output in the collect attempt's log folder (r4collect's log format, so status / plan3 / --prior /
the dataset build read it like any pass): `pass0-carry.jsonl` (+ .index.jsonl), one row per carried
case with `carried: {source, row}` — r4dataset.build re-renders that row on the round's manifest
(system turn, tools, the scored turn's candidate-fields block; r4dataset.rerender_carried) and applies
every keep rule again (leak against the current exam, preview, unservable, over-length, caps) — and
`carry-report.json` (counts by reason / family / source, the teacher spend not asked).
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

CARRY_LABEL = "pass0-carry"
CARRY_LOG = f"{CARRY_LABEL}.jsonl"
REPORT = "carry-report.json"
MAX_PARKED = 2
USD_PER_CASE_DEFAULT = 0.0298  # r5: US$101.57 over 3,408 asked cases (result-collect.json)
SERVER_LOOKUPS = ("lookup_fields", "describe_target")
ID_RE = re.compile(r'^\{"id": "([^"#]+)')
OFF = ("off", "0", "false", "no", "reask", "reask_all")
ON = ("on", "1", "true", "yes", "required")


class CarryError(RuntimeError):
    """A source cannot be read or the carry cannot run (the stage then re-asks every case)."""


# --- the switch ---------------------------------------------------------------------------------------------------------

def setting(run: dict, mode: str, explicit: str | None = None, env: dict | None = None) -> tuple[bool, str]:
    """(on, why). Default ON in full mode, OFF in smoke (a rehearsal must ask the teacher). The switch, first set wins:
    `explicit` (collect_stage.run(carry=…)), $COLLECT_CARRY, run.json `trigger.carry` ("on" / "off") or
    `trigger.reask_all: true` (an execution's args.trigger is copied into run.json — no workflow change)."""
    import os
    env = os.environ if env is None else env
    trig = (run or {}).get("trigger") or {}
    for why, v in (("explicit", explicit), ("env COLLECT_CARRY", env.get("COLLECT_CARRY")),
                   ("trigger.carry", trig.get("carry")),
                   ("trigger.reask_all", "off" if trig.get("reask_all") is True else None)):
        if v is None or str(v).strip() == "":
            continue
        s = str(v).strip().lower()
        if s in OFF:
            return False, why
        if s in ON:
            return True, why
    return (mode == "full"), f"default ({mode})"


def required(run: dict, explicit: str | None = None, env: dict | None = None) -> bool:
    """Whether a carry that cannot run must FAIL the collect stage instead of asking every case (≈ US$100 instead of
    ≈ US$40–50 with r4 + r5 as sources): "required" as the switch (`explicit`, $COLLECT_CARRY or trigger.carry)."""
    import os
    env = os.environ if env is None else env
    trig = (run or {}).get("trigger") or {}
    return any(str(v).strip().lower() == "required" for v in (explicit, env.get("COLLECT_CARRY"), trig.get("carry"))
               if v is not None)


def excluded_ids(run: dict, env: dict | None = None) -> set[str]:
    """Case ids never carried (always re-asked): run.json trigger.carry_exclude (list or comma string) and
    $COLLECT_CARRY_EXCLUDE (comma string)."""
    import os
    env = os.environ if env is None else env
    out: set[str] = set()
    for v in (((run or {}).get("trigger") or {}).get("carry_exclude"), env.get("COLLECT_CARRY_EXCLUDE")):
        if isinstance(v, str):
            v = v.split(",")
        out |= {str(x).strip() for x in v or [] if str(x).strip()}
    return out


def hidden_strict(run: dict, env: dict | None = None) -> bool:
    """Whether a carried row must not write a hidden field at all (`hidden_check`). On by default (r6 review: a
    carried row that writes a field the editor hides would train hidden writes); $COLLECT_CARRY_HIDDEN=keep or
    run.json trigger.carry_hidden "keep" turns it off (r4score's rule then: the case's measured warnings pass)."""
    import os
    env = os.environ if env is None else env
    v = env.get("COLLECT_CARRY_HIDDEN") or ((run or {}).get("trigger") or {}).get("carry_hidden") or "strict"
    v = str(v).strip().lower()
    if v not in ("strict", "keep", "off"):
        raise ValueError(f"carry_hidden {v!r}: strict (default) or keep")
    return v == "strict"


# --- sources ------------------------------------------------------------------------------------------------------------

def _rel(key: str | None) -> str | None:
    """A bucket key from a state file's value (`gs://bucket/cases/…` or `cases/…`)."""
    if not key:
        return None
    return re.sub(r"^gs://[^/]+/", "", str(key)).strip("/")


def find_sources(store, run_id: str, *, max_parked: int = MAX_PARKED) -> list[dict]:
    """[{name, kind, dataset, cases: [keys], at}] in priority order (module docstring). A source whose
    dataset has no train.jsonl is skipped."""
    from .common import run_prefix
    out: list[dict] = []
    model = store.read_json("state/model.json") or {}
    ds = _rel(model.get("dataset"))
    if ds:
        cases = [_rel(k) for k in model.get("training_cases") or [] if k]
        if not cases:
            idx = store.read_json("cases/pool/index.json") or {}
            tags = {str(x) for x in (model.get("tag"), model.get("candidate_id"), model.get("run_id")) if x}
            cases = [f["path"] for f in idx.get("files") or [] if str(f.get("trained_in")) in tags]
        if store.exists(f"{ds}/dataset/train.jsonl") and cases:
            out.append({"name": f"model:{model.get('tag')}", "kind": "trained", "dataset": ds, "cases": cases,
                        "at": model.get("trained_at")})
    since = str(model.get("trained_at") or "")
    parked = []
    for rel in store.list("state/parked"):
        if not rel.endswith(".json"):
            continue
        p = store.read_json(rel) or {}
        rid = p.get("run_id")
        if not rid or rid == run_id or str(p.get("at") or "") <= since:
            continue
        pre = run_prefix(rid)
        if store.exists(f"{pre}/dataset/dataset/train.jsonl") and store.exists(f"{pre}/prepare/cases.jsonl"):
            parked.append({"name": f"parked:{p.get('candidate_id') or rid}", "kind": "parked", "dataset": f"{pre}/dataset",
                           "cases": [f"{pre}/prepare/cases.jsonl"], "at": p.get("at"), "run_id": rid})
    parked.sort(key=lambda s: str(s.get("at") or ""), reverse=True)
    return out + parked[:max_parked]


def _read_by_id(path: Path, ids: set[str], key: Callable[[dict], str]) -> dict[str, dict]:
    """{id: row} for the rows of a (large) jsonl whose id is in `ids`; the id prefix is matched before parsing."""
    out: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            m = ID_RE.match(line)
            if m and m.group(1) not in ids:
                continue
            r = json.loads(line)
            k = key(r)
            if k in ids:
                out[k] = r
    return out


def load_source(store, src: dict, local: Path, ids: set[str]) -> dict:
    """{name, rows: {case id: hf row}, cases: {id: case}, manifest: Path} of the cases in `ids`. The
    downloads are deleted once read (the job's file system is memory)."""
    local = Path(local)
    local.mkdir(parents=True, exist_ok=True)
    man = local / "manifest.json"
    if not store.download(f"{src['dataset']}/manifest.json", man):
        raise CarryError(f"{src['name']}: {src['dataset']}/manifest.json is missing")
    from ..r4dataset import CAPPED_FILE
    rows: dict[str, dict] = {}
    for name in (CAPPED_FILE, "val.jsonl", "train.jsonl"):  # a trained row wins over a capped one
        p = local / name
        if store.download(f"{src['dataset']}/dataset/{name}", p):
            rows.update(_read_by_id(p, ids, lambda r: str((r.get("meta") or {}).get("case_id") or "")))
            p.unlink(missing_ok=True)
    rejected: set[str] = set()
    p = local / "rejects.jsonl"
    if store.download(f"{src['dataset']}/dataset/rejects.jsonl", p):
        with p.open(encoding="utf-8") as f:
            rejected = {str(json.loads(x).get("case_id")) for x in f if x.strip()} & ids
        p.unlink(missing_ok=True)
    cases: dict[str, dict] = {}
    for i, key in enumerate(src["cases"]):
        p = local / f"cases-{i}.jsonl"
        if not store.download(key, p):
            raise CarryError(f"{src['name']}: case file {key} is missing")
        cases.update(_read_by_id(p, ids, lambda c: str(c.get("id"))))
        p.unlink(missing_ok=True)
    return {"name": src["name"], "rows": rows, "cases": cases, "manifest": man, "rejected": rejected - set(rows)}


# --- manifest specs -----------------------------------------------------------------------------------------------------

def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


PROVENANCE = ("source",)  # where the launcher declares a field ({file, line}): moves with every edit, means nothing


def _semantic(v: Any) -> Any:
    """A spec without its provenance (PROVENANCE keys, at any depth)."""
    if isinstance(v, dict):
        return {k: _semantic(x) for k, x in v.items() if k not in PROVENANCE}
    if isinstance(v, list):
        return [_semantic(x) for x in v]
    return v


def spec_map(doc: dict) -> dict[str, str]:
    """{address: canonical spec} of every container (`sections.hero`), list (`sections.hero.slides`) and
    field (`sections.hero.title`, `sections.hero.slides[].title`) — case_addresses' addressing. Provenance
    (a field's source file and line) is not part of a spec."""
    from .rebase import _containers
    out: dict[str, str] = {}
    for cname, c in _containers(_semantic(doc)).items():
        out[cname] = _canon(c)
        for k, f in (c.get("fields") or {}).items():
            out[f"{cname}.{k}"] = _canon(f)
        for lname, lst in (c.get("lists") or {}).items():
            out[f"{cname}.{lname}"] = _canon(lst)
            for k, f in ((lst or {}).get("item") or {}).items():
                out[f"{cname}.{lname}[].{k}"] = _canon(f)
    return out


def _addr_variants(addr: str) -> list[str]:
    """`sections.x.style.color` → itself, then its parents down to the container (a nested value's spec is its root key's)."""
    out = [addr]
    a = re.sub(r"\[\d+\]", "", addr)
    if a != addr:
        out.append(a)
    while a.count(".") > 1:
        a = a.rsplit(".", 1)[0]
        out.append(a)
    return out


def touched_addresses(ops: Iterable[dict], case: dict | None, m) -> set[str]:
    """What `ops` write and `case` expects on manifest m (author.case_addresses' addressing)."""
    from .author import case_addresses
    out = case_addresses({"expected": {"ops": [o for o in ops if isinstance(o, dict)]}}, m)
    if case is not None:
        out |= case_addresses(case, m)
    return out


# --- the case fingerprint -----------------------------------------------------------------------------------------------

def case_fingerprint(c: dict) -> str:
    """What the teacher was asked and what the row was scored against, minus what the engine measures
    (expected diffs) and what the round re-derives (contexts — the replay checks the one sent)."""
    e = copy.deepcopy(c.get("expected") or {})
    e.pop("diff", None)
    for a in e.get("alt_exact") or []:
        if isinstance(a, dict):
            a.pop("diff", None)
    hist = [{k: h.get(k) for k in ("user", "expect", "page", "ops")} for h in c.get("history") or []]
    doc = {"kind": c.get("kind"), "request": c.get("request"), "history": hist, "target": c.get("target"),
           "site_state": c.get("site_state"), "mode": (c.get("score") or {}).get("mode"), "expected": e,
           "template": c.get("template")}
    for k in ("context_extras", "option_lists", "page_slugs"):  # a case's own lists (contrast LIST_JOBS, the
        if c.get(k):                                              # plumbing's host lists): absent on older cases
            doc[k] = c[k]
    return hashlib.sha256(_canon(doc).encode("utf-8")).hexdigest()


# --- the row taken apart ------------------------------------------------------------------------------------------------

def _args(call: dict) -> dict:
    try:
        v = json.loads((call.get("function") or {}).get("arguments") or "{}")
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


def row_turns(hf: dict) -> list[dict]:
    """The row's user turns: [{user index, text, streams, stored results {call id: content}, names}].
    A turn's streams are what the service streamed for it: text.delta, one tool.call per call the
    service streams (not lookup_fields / describe_target — answered in the service, never streamed;
    pick_asset — a turn.suspended), and a turn.waiting / turn.suspended after each model call that
    asked the browser; the last stream ends with turn.done."""
    from ..fields import split_user_text
    from .. import render
    msgs = hf.get("messages") or []
    users = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
    turns = []
    for n, ui in enumerate(users):
        end = users[n + 1] if n + 1 < len(users) else len(msgs)
        content = str(msgs[ui].get("content") or "")
        if n == len(users) - 1:
            _, ctx, _, text = split_user_text(content)
        else:
            ctx, text = None, content
        streams: list[list[tuple]] = []
        cur: list[tuple] = []
        stored: dict[str, str] = {}
        names: dict[str, str] = {}
        server: list[tuple[str, str]] = []
        calls_n = 0
        for m in msgs[ui + 1:end]:
            if m.get("role") == "assistant":
                calls_n += 1
                if m.get("content"):
                    cur.append(("text.delta", {"text": str(m["content"])}))
                browser, pick = [], None
                for c in m.get("tool_calls") or []:
                    name = str((c.get("function") or {}).get("name") or "")
                    cid = str(c.get("id") or "")
                    names[cid] = name
                    args = _args(c)
                    if name == render.PICK_ASSET:
                        pick = {"toolCallId": cid, "assetType": str(args.get("asset_type") or "image"),
                                "purpose": str(args.get("purpose") or "")}
                    elif name in SERVER_LOOKUPS:
                        server.append((cid, name))
                    else:
                        cur.append(("tool.call", {"id": cid, "name": name, "input": args}))
                        if name not in render.SERVER_TOOLS:
                            browser.append(cid)
                if browser:
                    cur.append(("turn.waiting", {"toolCallIds": browser}))
                if pick:
                    cur.append(("turn.suspended", pick))
                if browser or pick:
                    streams.append(cur)
                    cur = []
            elif m.get("role") == "tool":
                stored[str(m.get("tool_call_id") or "")] = str(m.get("content") or "")
        cur.append(("turn.done", {"stop": "end_turn", "usage": {"input_tokens": 0, "output_tokens": 0, "calls": calls_n}}))
        streams.append(cur)
        turns.append({"index": ui, "context": ctx, "text": text, "streams": streams, "stored": stored, "names": names,
                      "server": [(cid, name, stored.get(cid)) for cid, name in server]})
    return turns


def server_containers(turns: list[dict]) -> set[str]:
    """The containers the row's lookup_fields / describe_target results describe."""
    out: set[str] = set()
    for t in turns:
        for _, _, content in t["server"]:
            try:
                doc = json.loads(content or "")
            except ValueError:
                continue
            if isinstance(doc, dict) and doc.get("container"):
                out.add(str(doc["container"]))
    return out


REF_SLUG_PATH = re.compile(r"^(.*?)(tag|brand)Slug$")
REF_ID_PATH = re.compile(r"(?:^|\.)(?:tag|brand)Id$")


def _written_of(content: str) -> list[str] | None:
    try:
        d = json.loads(content or "")
    except ValueError:
        return None
    w = d.get("written") if isinstance(d, dict) else None
    return [str(x) for x in w] if isinstance(w, list) else None


def stale_ref_writes(hf: dict) -> list[str]:
    """The tag / brand slug paths a row's stored tool results wrote WITHOUT the derived id beside them (module
    docstring, ref_written_changed)."""
    out: list[str] = []
    for m in hf.get("messages") or []:
        if m.get("role") != "tool":
            continue
        written = _written_of(str(m.get("content") or "")) or []
        have = set(written)
        for p in written:
            mm = REF_SLUG_PATH.match(p)
            if mm and f"{mm.group(1)}{mm.group(2)}Id" not in have:
                out.append(p)
    return out


def ref_only_change(new: str, old: str) -> bool:
    """Two tool-result contents that differ only by derived tag / brand id paths in `written`."""
    try:
        a, b = json.loads(new), json.loads(old)
    except ValueError:
        return False
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return False
    for d in (a, b):
        if isinstance(d.get("written"), list):
            d["written"] = sorted(str(x) for x in d["written"] if not REF_ID_PATH.search(str(x)))
    return a == b and new != old


def same_result(new: str, old: str) -> bool:
    """Two tool-result contents are the same answer: byte-equal, or equal JSON once `written` / `warnings`
    are sorted (the stored conversation keeps a call's input with its keys sorted — Go re-marshals the map —
    so a replayed write lists its paths in key order where the browser listed them in the model's order)."""
    if new == old:
        return True
    try:
        a, b = json.loads(new), json.loads(old)
    except ValueError:
        return False
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return False
    for d in (a, b):
        for k in ("written", "warnings"):
            if isinstance(d.get(k), list):
                d[k] = sorted(map(str, d[k]))
    return a == b


class ReplayClient:
    """collect.Client as run_case drives it, answering from a carried row: each turn's streams in order,
    every posted tool result compared (as loop.browserResult renders it) with the row's stored one, and
    the text / context each turn was sent with recorded. Never raises: a request the row has no answer
    for gets a turn.error stream (the row is then not usable)."""

    build_id = "carry"

    def __init__(self, turns: list[dict]):
        self.turns = turns
        self.ti, self.si, self.seq = -1, 0, 0
        self.sent: list[dict] = []
        self.problems: list[str] = []

    def end_conversation(self, site_id: str) -> bool:
        return False

    def _json(self, method: str, path: str):
        return 200, {"conversationId": None, "messages": []}

    def _stream(self) -> SimpleNamespace:
        streams = self.turns[self.ti]["streams"] if 0 <= self.ti < len(self.turns) else []
        if self.si >= len(streams):
            self.problems.append(f"replay_mismatch:turn {self.ti} asked for stream {self.si} of {len(streams)}")
            evs = [("turn.error", {"code": "carry_replay", "message": "no recorded stream"})]
        else:
            evs = streams[self.si]
        out = []
        for name, data in evs:
            self.seq += 1
            out.append({"event": name, "data": dict(data), "seq": self.seq})
        return SimpleNamespace(events=out, turn_id=f"carry-{self.ti}")

    def start_turn(self, body: dict) -> SimpleNamespace:
        if 0 <= self.ti < len(self.turns) and self.si < len(self.turns[self.ti]["streams"]) - 1:
            self.problems.append(f"replay_mismatch:turn {self.ti} left streams unread")
        self.ti += 1
        self.si = 0
        self.sent.append({"context": body.get("context"), "text": body.get("text")})
        return self._stream()

    def tool_results(self, turn_id: str, body: dict) -> SimpleNamespace:
        from .. import render
        t = self.turns[self.ti] if 0 <= self.ti < len(self.turns) else {"names": {}, "stored": {}}
        results = list(body.get("results") or [])
        for item in results:
            cid = str(item.get("toolCallId") or "")
            content, _ = render.tool_result_content(t["names"].get(cid, ""), item)
            if cid not in t["stored"]:
                self.problems.append(f"tool_result_changed:{cid} not in the row")
            elif not same_result(content, t["stored"][cid]):
                why = "ref_written_changed" if ref_only_change(content, t["stored"][cid]) else "tool_result_changed"
                self.problems.append(f"{why}:{t['names'].get(cid)}")
        self.seq += len(results)
        self.si += 1
        return self._stream()

    def finished(self) -> None:
        if 0 <= self.ti < len(self.turns) and self.si < len(self.turns[self.ti]["streams"]) - 1:
            self.problems.append(f"replay_mismatch:turn {self.ti} left streams unread")
        if self.ti != len(self.turns) - 1:
            self.problems.append(f"replay_mismatch:{self.ti + 1} of {len(self.turns)} turns asked")


# --- one case -----------------------------------------------------------------------------------------------------------

class Context:
    """What every check of one source needs: both manifests, their changed addresses, the round's
    engine, the context fixture and assets, the source manifest's Fields (block parity)."""

    def __init__(self, *, m_old, m_new, engine, extras: dict, assets: list[dict], fields_old=None,
                 hidden_check: bool = False):
        self.m_old, self.m_new, self.engine = m_old, m_new, engine
        self.extras, self.assets, self.fields_old = extras, assets, fields_old
        self.hidden_check = hidden_check
        same = m_old.hash == m_new.hash
        a, b = (spec_map(m_old.doc), spec_map(m_new.doc)) if not same else ({}, {})
        self.known = set(a) | set(b)
        self.changed = {k for k in self.known if a.get(k) != b.get(k)}
        self.changed_containers = {a for a in self.changed if a.count(".") <= 1 and "[]" not in a}


def _changed_hits(addrs: Iterable[str], changed: set[str], known: set[str]) -> list[str]:
    """The addresses (or, for a nested value, the nearest address either manifest knows) whose spec changed."""
    hits = []
    for a in addrs:
        for v in _addr_variants(a):
            if v in known or v in changed:
                if v in changed:
                    hits.append(v)
                break
    return sorted(set(hits))


def check(case: dict, src_case: dict | None, hf: dict, ctx: Context) -> tuple[dict | None, str, dict]:
    """(log row | None, reason, detail) — the row carried for `case` (reason "") or why not."""
    from .. import r4collect
    from .. import r4score as R4S
    from ..r4dataset import RenderError, hf_final_turn
    from .. import render

    if src_case is None:
        return None, "no_source_case", {}
    if case_fingerprint(case) != case_fingerprint(src_case):
        return None, "case_changed", {}
    try:
        turns = row_turns(hf)
        final = hf_final_turn(hf)
    except (RenderError, ValueError) as e:
        return None, "row_unreadable", {"error": str(e)[:200]}
    if len(turns) != 1 + len(case.get("history") or []):
        return None, "turns_changed", {"row": len(turns), "case": 1 + len(case.get("history") or [])}
    stale_refs = stale_ref_writes(hf)
    if stale_refs:
        return None, "ref_written_changed", {"paths": stale_refs[:5]}
    if ctx.fields_old is not None:
        again = ctx.fields_old.block(final["context"], final["text"], final["previous"])
        if render.go_trim_space(again) != render.go_trim_space(final["block"]):
            return None, "block_parity", {}
    sc = server_containers(turns)
    stale = sorted(sc & ctx.changed_containers)
    if stale:
        return None, "server_result_stale", {"containers": stale}
    client = ReplayClient(turns)
    try:
        row = r4collect.run_r4_case(case, client, ctx.engine, route="carry", fields=None, manifest=ctx.m_new,
                                    extras=ctx.extras, assets=ctx.assets, site_prefix="lab-carry-")
    except Exception as e:  # noqa: BLE001 — one odd row is re-asked, never the stage's failure
        return None, "replay_error", {"error": f"{type(e).__name__}: {e}"[:300]}
    client.finished()
    for k, t in enumerate(turns):
        sent = client.sent[k] if k < len(client.sent) else {}
        if render.go_trim_space(str(sent.get("text") or "")) != render.go_trim_space(t["text"]):
            return None, "text_changed", {"turn": k}
        if k == len(turns) - 1 and _canon(sent.get("context")) != _canon(t["context"]):
            return None, "context_changed", {}
    if client.problems:
        why = client.problems[0].split(":")[0]
        return None, why, {"problems": client.problems[:5]}
    ops = [o for h in row.get("history_rows") or [] for o in h.get("ops") or []] + list(row.get("ops") or [])
    hits = _changed_hits(touched_addresses(ops, case, ctx.m_new) | touched_addresses(ops, src_case, ctx.m_old), ctx.changed,
                          ctx.known)
    if hits:
        return None, "target_changed", {"addresses": hits[:10]}
    sc_ = row.get("score") or {}
    if row.get("status") != "ok" or not sc_.get("usable"):
        flags = sc_.get("flags") or []
        return None, f"not_usable:{','.join(flags) if sc_.get('passed') and flags else sc_.get('verdict')}", {
            "reasons": (sc_.get("reasons") or [])[:5]}
    # W_HIDDEN comes with every write to a gated field, open gate or not: only the closed ones count
    # (r4score.closed_gate_writes reads the engine's post-op `hidden[].byValue`).
    allowed = {str(k) for k in (case.get("expected") or {}).get("expect_hidden") or []}
    seen = {f"W_HIDDEN:{k}" for r in [*(row.get("history_rows") or []), row]
            for k in R4S.closed_gate_writes(r.get("engine_results") or []) if k not in allowed}
    if seen:
        if ctx.hidden_check:
            return None, "hidden_write", {"warnings": sorted(seen)[:5]}
        return row, "", {"hidden": sorted(seen)[:5]}
    return row, "", {}


def log_row(row: dict, hf: dict, source: str) -> dict:
    """The replayed row as a collection log row (r4collect's schema) carrying the source row."""
    meta = hf.get("meta") or {}
    out = {k: v for k, v in row.items() if k != "conversation"}
    out.update(id=f"{row.get('case_id')}#carry", model=meta.get("model") or "carry", models=meta.get("models"),
               conversation={}, carried={"source": source, "row": hf})
    out["r4"] = {**(row.get("r4") or {}), "route": "carry"}
    return out


# --- the stage step -----------------------------------------------------------------------------------------------------

def carry(cases: list[dict], sources: list[dict], *, m_new, engine, extras: dict, assets: list[dict], logs: Path,
          rules: str | None = None, fields_for: Callable[[Path], Any] | None = None, exclude: set[str] = frozenset(),
          usd_per_case: float = USD_PER_CASE_DEFAULT, workers: int = 4, hidden_check: bool = False,
          progress: Callable[[str], None] = print) -> dict:
    """Carry what still applies of `sources` (load_source's dicts, priority order) for the trainable
    `cases`; writes `logs`/pass0-carry.jsonl (+ index) and returns the report (also carry-report.json).
    `rules` = the round's system[0] (None: not checked); `fields_for(manifest path)` = the Fields of a
    source manifest for block parity (None: not checked)."""
    from .. import r4collect
    from ..manifest import Manifest
    from ..r4dataset import RenderError

    logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    trainable = [c for c in cases if not c.get("holdout")]
    reasons: dict[str, str] = {}
    detail: dict[str, dict] = {}
    carried: dict[str, tuple[dict, dict, str]] = {}
    hidden: dict[str, list[str]] = {}
    src_report = []
    lock = threading.Lock()
    todo = [c for c in trainable if str(c["id"]) not in exclude and not (c.get("meta") or {}).get("no_carry")]
    todo_ids = {str(c["id"]) for c in todo}
    for c in trainable:
        if str(c["id"]) not in todo_ids:
            reasons[str(c["id"])] = "excluded"
    for src in sources:
        left = [c for c in todo if str(c["id"]) not in carried]
        have = [c for c in left if str(c["id"]) in src["rows"]]
        rec: dict[str, Any] = {"name": src["name"], "rows": len(src["rows"]), "candidates": len(have), "carried": 0}
        src_report.append(rec)
        if not have:
            continue
        m_old = Manifest(json.loads(Path(src["manifest"]).read_text(encoding="utf-8")))
        if rules is not None:
            from ..fields import split_system
            first = next(iter(src["rows"].values()))
            try:
                src_rules = split_system(str(((first.get("messages") or [{}])[0]).get("content") or ""))[0]
            except ValueError:
                src_rules = None
            if src_rules != rules:
                rec["skipped"] = "rules_changed"
                for c in have:
                    reasons.setdefault(str(c["id"]), "rules_changed")
                continue
        fields_old = fields_for(Path(src["manifest"])) if fields_for is not None else None
        if fields_old is not None:  # every turn's block in one process
            turns = []
            from ..r4dataset import hf_final_turn
            for c in have:
                try:
                    t = hf_final_turn(src["rows"][str(c["id"])])
                    turns.append((t["context"], t["text"], t["previous"]))
                except RenderError:
                    pass
            fields_old.prime(turns)
        ctx = Context(m_old=m_old, m_new=m_new, engine=engine, extras=extras, assets=assets, fields_old=fields_old,
                      hidden_check=hidden_check)
        rec["changed_addresses"] = len(ctx.changed)
        progress(f"carry: {src['name']}: {len(have)} candidate row(s); {len(ctx.changed)} manifest address(es) changed")

        def one(c: dict) -> None:
            cid = str(c["id"])
            hf = src["rows"][cid]
            row, why, det = check(c, src["cases"].get(cid), hf, ctx)
            with lock:
                if row is not None:
                    carried[cid] = (row, hf, src["name"])
                    reasons.pop(cid, None)
                    detail.pop(cid, None)
                    if det.get("hidden"):
                        hidden[cid] = det["hidden"]
                else:
                    reasons.setdefault(cid, why)  # the higher-priority source's reason is the one reported
                    if det and cid not in detail:
                        detail[cid] = det

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(one, have))
        rec["carried"] = sum(1 for v in carried.values() if v[2] == src["name"])
        progress(f"carry: {src['name']}: carried {rec['carried']}")
    for c in todo:
        cid = str(c["id"])
        if cid not in carried and cid not in reasons:
            if any(cid in (s.get("rejected") or ()) for s in sources):
                reasons[cid] = "no_row:rejected"  # every source's round rejected it: failing, re-asked
            elif any(cid in s["cases"] for s in sources):
                reasons[cid] = "no_row:not_kept"  # asked before but no row kept (capped before capped.jsonl, …)
            else:
                reasons[cid] = "new"
    # the log: rows first, the index last (an index means the log is whole)
    log, idx = logs / CARRY_LOG, r4collect.index_path(logs / CARRY_LOG)
    rows = [log_row(r, hf, s) for r, hf, s in (carried[k] for k in sorted(carried))]
    from .common import write_jsonl
    idx.unlink(missing_ok=True)
    write_jsonl(log, rows)
    summaries = []
    for r in rows:
        s = r4collect.summary_of(r, "carry", CARRY_LABEL)
        s["carried_from"] = r["carried"]["source"]
        summaries.append(s)
    write_jsonl(idx, summaries)
    fam = {str(c["id"]): str((c.get("meta") or {}).get("family")) for c in trainable}
    by_family: dict[str, Counter] = defaultdict(Counter)
    for cid, f in fam.items():
        by_family[f]["cases"] += 1
        by_family[f]["carried" if cid in carried else "asked"] += 1
    by_id = {str(c["id"]): c for c in trainable}
    turns_saved = sum(1 + len(by_id[k].get("history") or []) for k in carried)
    report = {"cases": len(trainable), "carried": len(carried), "asked": len(trainable) - len(carried),
              "carried_share": round(len(carried) / max(len(trainable), 1), 4),
              "reasons": dict(Counter(reasons.values()).most_common()),
              "reask_new": sum(1 for v in reasons.values() if v == "new"),
              "reask_failing": sum(1 for v in reasons.values() if v == "no_row:rejected" or v.startswith("not_usable")),
              "by_family": {k: dict(v) for k, v in sorted(by_family.items())},
              "sources": src_report, "turns_not_asked": turns_saved,
              "carried_hidden_write": len(hidden), "carried_hidden_write_ids": sorted(hidden)[:50],
              "usd_per_case": usd_per_case, "teacher_usd_saved_estimate": round(len(carried) * usd_per_case, 2),
              "per_case": {k: reasons.get(k, "carried") for k in sorted(fam)}}
    by_reason: dict[str, list[str]] = defaultdict(list)
    for k in sorted(reasons):
        if len(by_reason[reasons[k]]) < 5:
            by_reason[reasons[k]].append(k)
    report["examples"] = {r: [{"id": k, **detail.get(k, {})} for k in ids] for r, ids in by_reason.items()}
    (logs / REPORT).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report
