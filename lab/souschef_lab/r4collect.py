"""r4collect — the teacher collection for the r4 retrain (docs/LAB.md §5b).

Runs every case of an r4 case file (r4cases.py) through a Sous Chef service
as the browser would, scores it with verify-3 (r4score.py) and logs the
FULL trajectory: one row per case and pass, the scored turn's row in
collect.run_case's schema plus `history_rows` (each earlier turn's own row:
context, text, candidate-fields block, events with their tool.results) so the
dataset builder can render the conversation exactly as the service sends it
to the student.

  single turn  collect.run_case on the case's state and context (a fresh conversation: DELETE first)
  multi turn   the case's earlier user turns asked LIVE in one conversation (one DELETE before the first; the
               engine state carried over — the service has no seed endpoint), each turn's context re-derived
               from the live state with state.context_of exactly as the browser derives it (`--recorded-contexts`
               sends the authored ones instead); every turn's block is the service's follow-up-aware one
               (fields.Fields.block with the previous exchange, loop.previousExchange). A turn whose live answer
               does not do what the case's turn expected (`expect`: ask → a question and no write; write → an
               accepted write; report_unsupported → the call; answer → no write) marks the row
               `history_diverged` (scored, but not usable). An exact case's expected ops are re-measured on the
               live state before the last turn (they were measured on the authored one).

Passes (COLLECT.md): pass 1 every trainable case on route `default`; pass 2 every case pass 1 left without a
usable row (`--select unusable --prior pass1.jsonl`) on route `opus`; pass 3 what is still unusable on the
better route (`plan3`). Chunked and resumable: `--max-seconds` stops starting new cases (in-flight ones finish),
a rerun skips every case the log already holds an answered row for. A small index beside each log
(`<log>.index.jsonl`: case, family, status, verdict, flags, usable, cost) is what selection and `status` read.

The key comes from $SOUS_CHEF_KEY (never an argument: argv is visible to every process); `--fake` runs the same
path against the lab's in-process fake service (`--oracle`: every turn scripted with its expected answer), no
network, no model.

    python -m souschef_lab.r4collect run --cases <r4>/cases-all.jsonl --log <r4>/collect/pass1.jsonl \\
        --route default --url $URL --workers 8 --max-seconds 450 --site-prefix lab-r4p1-
    python -m souschef_lab.r4collect status --cases <r4>/cases-all.jsonl --log <r4>/collect/pass1.jsonl …
    python -m souschef_lab.r4collect plan3 --cases … --log pass1 --log pass2 --out-dir <r4>/collect
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable

from . import collect as C
from . import render
from .r4score import PASSING, RULE_VERSION, score

ROUTE_MODELS = {"default": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
ID_RE = re.compile(r'^\{"id": "([^"]+)"')
RETRY_HTTP = (429, 500, 502, 503, 504)


# --- cases --------------------------------------------------------------------------------------------------------------

def load_cases(path: Path, ids: set[str] | None = None, families: set[str] | None = None,
               include_holdout: bool = False) -> list[dict]:
    """The cases (only `ids` when given — the file is large, so other lines are not parsed)."""
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            m = ID_RE.match(line)
            if ids is not None and m and m.group(1) not in ids:
                continue
            c = json.loads(line)
            if ids is not None and c["id"] not in ids:
                continue
            if families and (c.get("meta") or {}).get("family") not in families:
                continue
            if c.get("holdout") and not include_holdout:
                continue
            out.append(c)
    return out


def case_index(path: Path) -> list[dict]:
    """{id, family, mode, holdout, turns} of every case (parses each line once; for status / selection)."""
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                out.append({"id": c["id"], "family": (c.get("meta") or {}).get("family"),
                            "mode": (c.get("score") or {}).get("mode"), "holdout": bool(c.get("holdout")),
                            "turns": 1 + len(c.get("history") or [])})
    return out


# --- logs ---------------------------------------------------------------------------------------------------------------

def index_path(log: Path) -> Path:
    return Path(str(log) + ".index.jsonl")


def read_index(logs: Iterable[Path]) -> list[dict]:
    rows = []
    for log in logs:
        p = index_path(Path(log))
        if p.is_file():
            rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


def answered(log: Path) -> set[str]:
    """Cases this log already holds a row for that need no new ask: any row that
    reached the service's answer (status ok). Transport errors are asked again."""
    return {r["case_id"] for r in read_index([log]) if r.get("status") == "ok"}


def usable_cases(logs: Iterable[Path]) -> set[str]:
    return {r["case_id"] for r in read_index(logs) if r.get("usable")}


def row_cost(row: dict, route: str) -> float:
    from .examcmp import PRICES
    model = ROUTE_MODELS.get(route)
    if model not in PRICES:
        return 0.0
    pin, pout, pw, pr = PRICES[model]
    total = 0.0
    for u in [row.get("usage") or {}] + [h.get("usage") or {} for h in row.get("history_rows") or []]:
        total += (pin * int(u.get("input_tokens") or 0) + pout * int(u.get("output_tokens") or 0)
                  + pw * int(u.get("cache_creation_input_tokens") or 0) + pr * int(u.get("cache_read_input_tokens") or 0)) / 1e6
    return total


class Log:
    """The trajectory log + its index, appended under one lock, fsynced."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def append(self, row: dict, summary: dict) -> None:
        with self.lock:
            C.append_row(self.path, row)
            C.append_row(index_path(self.path), summary)


# --- one case -----------------------------------------------------------------------------------------------------------

class FollowupFields:
    """fields.Fields as collect.run_case calls it (`block(context, text)`), with
    the previous exchange the service reads a follow-up with."""

    def __init__(self, fields: Any, previous: tuple[str, str] | None):
        self.fields, self.previous = fields, previous

    def block(self, context: dict | None, text: str) -> str:
        return self.fields.block(context, text, self.previous)


def last_reply(row: dict) -> str:
    """loop.previousExchange's assistant side: the text of the last assistant
    message of the turn that says something (trimmed)."""
    from .r4score import assistant_texts
    texts = assistant_texts(row)
    return render.go_trim_space(texts[-1]) if texts else ""


def diverged(expect: str, row: dict) -> str | None:
    from .exam import accepted_ops
    accepted, _ = accepted_ops(row)
    reply = str(row.get("reply") or "")
    called = {c.get("name") for c in row.get("tool_calls") or []}
    if expect in ("ask", "offer") and (accepted or "?" not in reply):
        return "did_not_ask"
    if expect == "write" and not accepted:
        return "did_not_write"
    if expect == "report_unsupported" and "report_unsupported" not in called:
        return "did_not_report"
    if expect == "answer" and accepted:
        return "wrote_on_a_question"
    return None


HISTORY_KEEP = ("id", "status", "http_status", "code", "error", "latency_ms", "turn_id", "conversation_id", "stop",
                "usage", "routes", "escalated", "models", "fields_context", "fields_context_by", "candidate_fields",
                "context", "text", "events", "ops", "tool_calls", "written_paths", "engine_results", "picked_assets",
                "reply", "batches")


def run_r4_case(case: dict, client: Any, engine: Any, *, route: str | None, fields: Any | None, manifest: Any,
                extras: dict, assets: list[dict], site_prefix: str, live_contexts: bool = True, sample: int = 1) -> dict:
    """One case end to end → the log row (never raises for a service answer)."""
    from .exam import StateTap
    from .state import context_of
    from .verify import ctx_of

    dict_engine = C.DictEngine(engine)
    tap = StateTap(dict_engine)
    state = copy.deepcopy(case.get("site_state") or {})
    meta = case.get("meta") or {}
    view = meta.get("view") or None
    history = case.get("history") or []
    hist_rows: list[dict] = []
    divergences: list[dict] = []
    previous: tuple[str, str] | None = None
    for i, h in enumerate(history):
        page = h.get("page") or (case.get("context") or {}).get("page") or "home"
        ctx = (context_of(state, page, extras, manifest, view=view, text=h.get("user") or "")
               if live_contexts or not isinstance(h.get("context"), dict) else h["context"])
        hc = {"id": case["id"], "kind": case.get("kind"), "site_state": state, "context": ctx, "request": h.get("user") or "",
              "template": case.get("template"), "manifest_hash": case.get("manifest_hash"), "meta": meta,
              "holdout": case.get("holdout")}
        f = FollowupFields(fields, previous) if fields is not None else None
        hr = C.run_case(hc, client, tap, route=route, sample=sample, assets=assets, verifier=None, fields=f,
                        fresh_conversation=(i == 0), site_prefix=site_prefix)
        if tap.state is not None:
            state = tap.state
        hist_rows.append({k: hr.get(k) for k in HISTORY_KEEP} | {"expect": h.get("expect"), "page": page})
        if hr.get("status") != "ok":
            break
        d = diverged(str(h.get("expect") or ""), hr)
        if d:
            divergences.append({"turn": i, "expect": h.get("expect"), "why": d})
        previous = (render.go_trim_space(h.get("user") or ""), last_reply(hr))
    final = copy.deepcopy(case)
    final["site_state"] = state
    info: list[str] = []
    if history:
        if live_contexts:
            final["context"] = context_of(state, meta.get("page") or (case.get("context") or {}).get("page") or "home",
                                          extras, manifest, view=view, text=case.get("request") or "")
        if final["score"]["mode"] == "exact":
            ops = [o for o in (final["expected"].get("ops") or []) if o.get("op") != "pick_asset"]
            r = engine.run(state, ops, ctx_of(final))
            if r.ok and r.diff():
                final["expected"]["diff"] = r.diff()
                info.append("exact_remeasured_on_live_state")
            elif final["expected"].get("keys"):
                final["score"] = {"mode": "keys"}
                info.append("exact_refused_on_live_state:scored_by_keys")
            else:
                info.append("exact_refused_on_live_state")
    if hist_rows and hist_rows[-1].get("status") != "ok":
        row: dict = {"id": f"{case['id']}#{sample}", "case_id": case["id"], "sample": sample, "model": route or "",
                     "status": "error", "code": "history_failed", "error": f"history turn {len(hist_rows)} failed: "
                     f"{hist_rows[-1].get('code')} {hist_rows[-1].get('error')}", "latency_ms": 0, "events": [],
                     "tool_calls": [], "ops": []}
    else:
        f = FollowupFields(fields, previous) if fields is not None else None
        row = C.run_case(final, client, tap, route=route, sample=sample, assets=assets, verifier=None, fields=f,
                         fresh_conversation=not history, site_prefix=site_prefix)
    if row.get("status") == "ok":
        row["conversation"] = stored_conversation(client, f"{site_prefix}{case['id']}")
    sc = score(final, row, engine, hist_rows) if row.get("status") == "ok" else {
        "verdict": "error", "reasons": [f"error:{row.get('code')}"], "mode": final["score"]["mode"], "flags": [],
        "passed": False, "usable": False, "rule_version": RULE_VERSION, "info": []}
    sc["info"] = list(sc.get("info") or []) + info
    if divergences:
        sc["history_diverged"] = divergences
        sc["flags"] = sorted(set(sc["flags"]) | {"history_diverged"})
        sc["usable"] = False
    row["verdict"], row["reasons"] = sc["verdict"], sc["reasons"]
    row["score"] = sc
    row["history_rows"] = hist_rows
    row["final_expected"] = {"mode": final["score"]["mode"], "diff": final["expected"].get("diff")} if history else None
    row["r4"] = {"family": meta.get("family"), "mode": case["score"]["mode"], "route": route,
                 "live_contexts": bool(history) and live_contexts, "site_prefix": site_prefix}
    return row


def stored_conversation(client: Any, site_id: str) -> dict:
    """GET /v1/conversations/{siteId}: the messages the service STORED for the
    conversation — the only record of the server tools' round trips
    (lookup_fields / describe_target are never streamed; report_unsupported is
    echoed without its result), so the dataset renders from it. Never raises:
    {"error": …} when it cannot be read."""
    from urllib.parse import quote
    try:
        status, doc = client._json("GET", f"/v1/conversations/{quote(site_id, safe='')}")
    except Exception as e:  # noqa: BLE001 — ServiceError / OSError: the row keeps its events
        return {"error": f"{type(e).__name__}: {e}"[:500]}
    msgs = sorted((m for m in (doc or {}).get("messages") or [] if isinstance(m, dict)), key=lambda m: m.get("seq") or 0)
    return {"conversationId": (doc or {}).get("conversationId"), "messages": msgs}


def summary_of(row: dict, route: str, label: str) -> dict:
    sc = row.get("score") or {}
    return {"case_id": row.get("case_id"), "family": (row.get("r4") or {}).get("family"), "pass": label, "route": route,
            "status": row.get("status"), "http_status": row.get("http_status"), "code": row.get("code"),
            "verdict": sc.get("verdict"), "flags": sc.get("flags") or [], "usable": bool(sc.get("usable")),
            "mode": sc.get("mode"), "turns": 1 + len(row.get("history_rows") or []),
            "latency_ms": (row.get("latency_ms") or 0) + sum(h.get("latency_ms") or 0 for h in row.get("history_rows") or []),
            "cost_usd": round(row_cost(row, route), 5), "escalated": bool(row.get("escalated")),
            "conversation": bool((row.get("conversation") or {}).get("messages")), "at": C.utc_now()}


# --- the run ------------------------------------------------------------------------------------------------------------

def oracle_scripts(cases: list[dict], site_prefix: str) -> dict:
    """Fake-service scripts: every turn answered as the case expects (dry runs)."""
    from tests.fake_service import Rotation  # the tests' fake: a dry run's stand-in, never imported by a real run

    def call(i: int, op: dict) -> tuple:
        return ("tool.call", {"id": f"o{i}", "name": op.get("op"), "input": {k: v for k, v in op.items() if k != "op"}})

    def done() -> tuple:
        return ("turn.done", {"stop": "end_turn", "usage": {"input_tokens": 1000, "output_tokens": 100, "calls": 1}})

    def write_turn(ops: list[dict], text: str) -> list:
        streams: list = []
        n = 0
        if any(o.get("op") == "pick_asset" for o in ops):
            p = next(o for o in ops if o.get("op") == "pick_asset")
            streams.append([("turn.suspended", {"toolCallId": "pick0", "assetType": p.get("asset_type") or "image",
                                                "purpose": p.get("purpose") or "the image"})])
        rest = [o for o in ops if o.get("op") != "pick_asset"]
        if rest:
            calls = [call(n + i, o) for i, o in enumerate(rest)]
            streams.append(calls + [("turn.waiting", {"toolCallIds": [c[1]["id"] for c in calls]})])
        streams.append([("text.delta", {"text": text}), done()])
        return streams

    def report(where: str, text: str) -> list:
        return [[("tool.call", {"id": "u0", "name": "report_unsupported",
                                "input": {"reason": "Not something the site editor changes.", "where": where}}),
                 ("text.delta", {"text": text}), done()]]

    out = {}
    for c in cases:
        turns = []
        for h in c.get("history") or []:
            exp = h.get("expect")
            if exp == "write" and h.get("ops"):
                turns.append(write_turn(h["ops"], h.get("assistant") or "Done."))
            elif exp == "report_unsupported":
                w = next(((x.get("input") or {}).get("where") for x in h.get("calls") or [] if x.get("name") == "report_unsupported"), "support")
                turns.append(report(w or "support", h.get("assistant") or "That's done outside the editor."))
            else:
                turns.append([[("text.delta", {"text": h.get("assistant") or "Which one do you mean?"}), done()]])
        mode = c["score"]["mode"]
        if mode == "exact" and c.get("kind") in ("clarify", "unsupported"):  # F0X: verify-2 scores these by kind
            mode = "clarify" if c["kind"] == "clarify" else "where"
            c = {**c, "expected": {**c["expected"], "out_of_scope": c["expected"].get("out_of_scope")
                                   or {"expected_where": ["support"]}}}
        exp = c["expected"]
        if mode == "where":
            turns.append(report(exp["out_of_scope"]["expected_where"][0], "That's done outside the site editor."))
        elif mode == "clarify":
            cands = [x if isinstance(x, str) else x[0] for x in (c["meta"].get("candidates") or [])][:3]
            turns.append([[("text.delta", {"text": f"Which one do you mean: {', '.join(cands)}?"}), done()]])
        elif mode == "answer":
            turns.append([[("text.delta", {"text": "Your options include " + ", ".join(map(str, (exp.get("reply_any") or [])[:3])) + "."}), done()]])
        else:
            ops = c["meta"].get("oracle_ops") or exp.get("ops") or []
            turns.append(write_turn(ops, "Done — that's changed. Save when you're happy with it."))
        out[f"{site_prefix}{c['id']}"] = Rotation([t for t in turns])
    return out


def run(cases: list[dict], *, client: Any, engine: Any, log: Log, route: str | None, label: str, fields: Any | None,
        manifest: Any, extras: dict, assets: list[dict], site_prefix: str, workers: int, max_seconds: float | None,
        live_contexts: bool = True, progress=print) -> Counter:
    done = answered(log.path)
    todo = [c for c in cases if c["id"] not in done]
    progress(f"r4collect[{label}]: {len(cases)} selected, {len(cases) - len(todo)} already answered in {log.path.name}, "
             f"{len(todo)} to ask on route {route or '(the default)'} with {workers} workers"
             + (f", new cases started for {max_seconds:.0f}s" if max_seconds else ""))
    t0 = time.monotonic()
    counts: Counter = Counter()
    stop = threading.Event()
    lock = threading.Lock()
    pending = list(reversed(todo))

    def worker() -> None:
        while not stop.is_set():
            if max_seconds and time.monotonic() - t0 > max_seconds:
                return
            with lock:
                if not pending:
                    return
                case = pending.pop()
            try:
                row = run_r4_case(case, client, engine, route=route, fields=fields, manifest=manifest, extras=extras,
                                  assets=assets, site_prefix=site_prefix, live_contexts=live_contexts)
            except C.FieldsContextMismatch as e:
                progress(f"STOP: {e}")
                stop.set()
                return
            except Exception as e:  # noqa: BLE001 — a lab bug on one case must not lose the chunk's other rows
                row = {"id": f"{case['id']}#1", "case_id": case["id"], "status": "error", "code": "lab_exception",
                       "error": f"{type(e).__name__}: {e}"[:2000], "r4": {"family": case["meta"].get("family")},
                       "score": {"verdict": "error", "flags": [], "usable": False}}
            s = summary_of(row, route or "", label)
            log.append(row, s)
            with lock:
                counts["asked"] += 1
                counts["ok" if row.get("status") == "ok" else "error"] += 1
                counts["usable"] += int(s["usable"])
                counts[f"verdict:{s['verdict']}"] += 1
                counts["cost_usd_x1e5"] += int(s["cost_usd"] * 1e5)
                n = counts["asked"]
            if n % 10 == 0 or row.get("status") != "ok":
                progress(f"  [{n}/{len(todo)} {time.monotonic() - t0:.0f}s] {case['id']} {s['verdict']} "
                         f"{','.join(s['flags'])} {s['latency_ms'] / 1000:.1f}s")
            if row.get("http_status") in RETRY_HTTP:
                time.sleep(5)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(worker) for _ in range(workers)]
        for f in futs:
            f.result()
    counts["left"] = len(pending)
    counts["seconds"] = int(time.monotonic() - t0)
    return counts


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="souschef_lab.r4collect", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ask the selected cases (one chunk; rerun to resume)")
    r.add_argument("--cases", type=Path, required=True)
    r.add_argument("--log", type=Path, required=True)
    r.add_argument("--label", help="the pass's name in the index (default: the log's stem)")
    r.add_argument("--route", required=True)
    r.add_argument("--url", default=os.environ.get("SOUS_CHEF_URL", "http://127.0.0.1:8090"))
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--assets", type=Path, default=C.DEFAULT_ASSETS)
    r.add_argument("--engine", type=Path, default=os.environ.get("ENGINE"))
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--max-seconds", type=float, default=450.0, help="start no new case after this many seconds (default 450)")
    r.add_argument("--site-prefix", default="lab-r4-")
    r.add_argument("--select", choices=("all", "unusable"), default="all",
                   help="unusable = the cases no --prior log holds a usable row for")
    r.add_argument("--prior", type=Path, action="append", default=[])
    r.add_argument("--ids-file", type=Path, help="only these case ids (one per line)")
    r.add_argument("--families", help="comma list, e.g. F2,F3")
    r.add_argument("--limit", type=int, help="at most N cases (after selection; a smoke)")
    r.add_argument("--holdout", action="store_true",
                   help="ask ONLY the holdout cases (F0H / F0X: an exam, never trained) — e.g. the student-alone eval")
    r.add_argument("--recorded-contexts", action="store_true", help="multi-turn: send the authored contexts instead of re-deriving them")
    r.add_argument("--fields-context", choices=("auto", "on", "off"), default="auto")
    r.add_argument("--timeout", type=float, default=300.0)
    r.add_argument("--build-id", default="lab-r4")
    r.add_argument("--fake", action="store_true", help="dry run: the lab's fake service in process (no network, no model)")
    r.add_argument("--oracle", action="store_true", help="with --fake: every turn scripted with its expected answer")
    r.add_argument("--fake-blocks", action="store_true",
                   help="with --fake: the fake stores the candidate-fields block the deployed CLI computes (as the real "
                        "service does), so a dry smoke-check can come out drift-free")
    s = sub.add_parser("status", help="counts per family / pass, cost, what is left")
    s.add_argument("--cases", type=Path, required=True)
    s.add_argument("--log", type=Path, action="append", required=True)
    s.add_argument("--json", type=Path)
    p3 = sub.add_parser("plan3", help="pass 3: the still-unusable cases, each assigned the better route for its family")
    p3.add_argument("--cases", type=Path, required=True)
    p3.add_argument("--log", type=Path, action="append", required=True)
    p3.add_argument("--out-dir", type=Path, required=True)
    cp = sub.add_parser("compare", help="two runs of the same cases: pass per family + the paired McNemar split")
    cp.add_argument("--run", action="append", required=True, help="NAME=<log> (its .index.jsonl is read); twice")
    cp.add_argument("--json", type=Path)
    cp.add_argument("--cases", type=Path, action="append",
                    help="case file(s) whose meta.key_seen splits each family into <fam>:unseen / <fam>:seen "
                         "(the generated exam's keys some r4 training case writes; EVAL.md reads the unseen rows)")
    return ap


def cmd_run(a) -> int:
    from .engine import PersistentEngine, default_cache_dir
    from .manifest import load as load_manifest
    from .state import load_context_fixture

    if not 1 <= a.workers <= 8:
        print("--workers must be 1–8", file=sys.stderr)
        return 2
    if not a.site_prefix.startswith("lab-"):
        print("--site-prefix must start with lab-", file=sys.stderr)
        return 2
    ids = None
    if a.ids_file:
        ids = {x.strip() for x in a.ids_file.read_text(encoding="utf-8").splitlines() if x.strip()}
    if a.select == "unusable":
        idx = case_index(a.cases)
        good = usable_cases(a.prior)
        sel = {c["id"] for c in idx if not c["holdout"] and c["id"] not in good}
        ids = sel if ids is None else (ids & sel)
    fams = {x.strip() for x in a.families.split(",")} if a.families else None
    cases = load_cases(a.cases, ids, fams, include_holdout=a.holdout)
    if a.holdout:
        cases = [c for c in cases if c.get("holdout")]
    if a.limit:
        cases = cases[: a.limit]
    if not cases:
        print("r4collect: nothing selected")
        return 0
    manifest = load_manifest(str(a.manifest))
    hashes = {c.get("manifest_hash") for c in cases}
    if hashes != {manifest.hash}:
        print(f"r4collect: the cases are on {sorted(map(str, hashes))}, --manifest is {manifest.hash}", file=sys.stderr)
        return 2
    svc = None
    key = os.environ.get("SOUS_CHEF_KEY", "")
    url = a.url
    if a.fake:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tests import fake_service as FS
        doc = json.loads(a.manifest.read_text(encoding="utf-8"))
        key = "test-key"
        svc = FS.FakeService(manifest=doc, key=key)
        svc.routes = [{"name": "default", "kind": "anthropic", "fields_context": True, "fallback": None},
                      {"name": "opus", "kind": "anthropic", "fields_context": True, "fallback": None},
                      {"name": "student", "kind": "messages-compatible", "fields_context": True, "fallback": None}]
        if a.oracle:
            svc.scripts.update(oracle_scripts(cases, a.site_prefix))
        if a.fake_blocks:
            from . import fields as F
            fake_fields = F.Fields(a.manifest)
            svc.block_for = lambda ctx, text, prev: fake_fields.block(ctx, text, prev)
        svc.start()
        url = svc.url
    elif not key:
        print("r4collect: $SOUS_CHEF_KEY is empty (the driver reads it from the key file; never pass it on argv)", file=sys.stderr)
        return 2
    engine = None
    try:
        client, _ = C.connect(url, key, a.manifest, cases, sse=False, timeout_s=a.timeout, build_id=a.build_id)
        fields, rec = C.resolve_fields(a.fields_context, a.manifest, client, a.route)
        print(f"r4collect: fields_context {'on' if fields is not None else 'off'} for route {a.route} "
              f"(says {rec['fields_context_by']}); route_info {json.dumps(rec['route_info'])}")
        engine = PersistentEngine(a.engine, manifest_path=a.manifest, size=a.workers, cache_dir=default_cache_dir())
        counts = run(cases, client=client, engine=engine, log=Log(a.log), route=a.route, label=a.label or a.log.stem,
                     fields=fields, manifest=manifest, extras=load_context_fixture(),
                     assets=C.load_assets(a.assets), site_prefix=a.site_prefix, workers=a.workers,
                     max_seconds=a.max_seconds, live_contexts=not a.recorded_contexts)
    finally:
        if engine is not None:
            engine.close()
        if svc is not None:
            svc.stop()
    cost = counts.pop("cost_usd_x1e5", 0) / 1e5
    print(f"r4collect: asked {counts['asked']} in {counts['seconds']}s: {counts['ok']} answered, {counts['error']} errors, "
          f"{counts['usable']} usable; ≈ US${cost:.2f}; {counts['left']} left in this selection "
          f"({'rerun to continue' if counts['left'] else 'selection done'})")
    print("verdicts: " + ", ".join(f"{k[8:]} {v}" for k, v in sorted(counts.items()) if k.startswith("verdict:")))
    return 0 if counts["left"] == 0 else 3


def status(cases_path: Path, logs: list[Path]) -> dict:
    idx = case_index(cases_path)
    rows = read_index(logs)
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)
    fam: dict[str, Counter] = defaultdict(Counter)
    passes: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        p = passes[f"{r['pass']}/{r['route']}"]
        p["rows"] += 1
        p["ok"] += int(r.get("status") == "ok")
        p["usable"] += int(bool(r.get("usable")))
        p["turns"] += int(r.get("turns") or 1)
        p["cost_usd_x1e5"] += int(float(r.get("cost_usd") or 0) * 1e5)
        p["latency_ms"] += int(r.get("latency_ms") or 0)
        for fl in r.get("flags") or []:
            p[f"flag:{fl}"] += 1
        p[f"verdict:{r.get('verdict')}"] += 1
    asked_ids = set(by_case)
    for c in idx:
        if c["holdout"] and c["id"] not in asked_ids:
            continue
        f = fam[c["family"]]
        f["cases"] += 1
        rs = by_case.get(c["id"], [])
        f["asked"] += int(bool(rs))
        f["usable"] += int(any(r.get("usable") for r in rs))
        f["passed"] += int(any(r.get("verdict") in PASSING for r in rs))
        f["passed_but_flagged"] += int(not any(r.get("usable") for r in rs) and any(r.get("verdict") in PASSING for r in rs))
    out = {"passes": {k: {**{kk: vv for kk, vv in v.items() if kk not in ("cost_usd_x1e5", "latency_ms")},
                          "cost_usd": round(v["cost_usd_x1e5"] / 1e5, 2),
                          "mean_case_s": round(v["latency_ms"] / 1000 / max(v["rows"], 1), 1)} for k, v in passes.items()},
           "families": {k: dict(v) for k, v in sorted(fam.items())}}
    tot = Counter()
    for v in fam.values():
        tot.update(v)
    out["total"] = dict(tot)
    out["cost_usd"] = round(sum(v["cost_usd"] for v in out["passes"].values()), 2)
    return out


def cmd_status(a) -> int:
    st = status(a.cases, a.log)
    if a.json:
        a.json.write_text(json.dumps(st, indent=1), encoding="utf-8")
    for k, v in st["passes"].items():
        flags = {kk[5:]: vv for kk, vv in v.items() if kk.startswith("flag:")}
        verd = {kk[8:]: vv for kk, vv in v.items() if kk.startswith("verdict:")}
        print(f"{k}: {v['rows']} rows ({v['turns']} turns), {v['usable']} usable, US${v['cost_usd']}, "
              f"{v['mean_case_s']} s/case; verdicts {verd}; flags {flags}")
    for k, v in st["families"].items():
        print(f"  {k}: {v.get('usable', 0)}/{v['cases']} usable, {v.get('passed', 0)} passed ({v.get('asked', 0)} asked, "
              f"{v.get('passed_but_flagged', 0)} passed but flagged)")
    t = st["total"]
    print(f"total: {t.get('usable', 0)}/{t.get('cases', 0)} cases usable; spent ≈ US${st['cost_usd']}")
    return 0


def cmd_plan3(a) -> int:
    """The better route per family = the higher usable rate: `default`'s over the
    family's pass-1 rows, `opus`'s over its pass-2 rows (the harder cases — so
    opus winning there is a clear win); ties go to opus (a fresh model, not the
    same draw again)."""
    idx = case_index(a.cases)
    rows = read_index(a.log)
    rate: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("status") != "ok":
            continue
        k = (r["family"], r["route"])
        rate[k][0] += int(bool(r.get("usable")))
        rate[k][1] += 1
    good = {r["case_id"] for r in rows if r.get("usable")}
    left = [c for c in idx if not c["holdout"] and c["id"] not in good]
    pick: dict[str, list[str]] = defaultdict(list)
    why = {}
    for c in left:
        f = c["family"]
        d, o = rate.get((f, "default"), [0, 0]), rate.get((f, "opus"), [0, 0])
        rd = d[0] / d[1] if d[1] else 0.0
        ro = o[0] / o[1] if o[1] else 0.0
        route = "opus" if ro >= rd else "default"
        why[f] = {"default": f"{d[0]}/{d[1]}", "opus": f"{o[0]}/{o[1]}", "pass3": route}
        pick[route].append(c["id"])
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for route, ids in pick.items():
        (a.out_dir / f"pass3-{route}.ids").write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(json.dumps({"left": len(left), "by_route": {k: len(v) for k, v in pick.items()}, "families": why}, indent=1))
    return 0


def cmd_compare(a) -> int:
    """Two runs of the same cases (e.g. r3 and r4 alone on the verbatim exam):
    pass rate per family with Wilson 95 % intervals and the paired split with
    the exact McNemar p (examcmp's), by case id."""
    from .examcmp import mcnemar_p, wilson
    runs = {}
    for spec in a.run:
        name, _, path = spec.partition("=")
        rows = {}
        for r in read_index([Path(path)]):
            rows[r["case_id"]] = r  # a later row of a case wins
        runs[name] = rows
    names = list(runs)
    if len(names) != 2:
        print("compare: give exactly two --run NAME=LOG", file=sys.stderr)
        return 2
    A, B = (runs[n] for n in names)
    common = sorted(set(A) & set(B))
    out: dict[str, Any] = {"runs": names, "common": len(common), "families": {}}
    seen: dict[str, bool] = {}
    for cp in a.cases or []:
        for line in Path(cp).open(encoding="utf-8"):
            m = ID_RE.match(line)
            if m and m.group(1) in A:
                c = json.loads(line)
                if "key_seen" in (c.get("meta") or {}):
                    seen[c["id"]] = bool(c["meta"]["key_seen"])
    fams = sorted({A[c].get("family") for c in common})
    groups = fams + ["ALL"]
    if seen:
        groups = [g for f in fams for g in (f, f"{f}:unseen", f"{f}:seen")] + ["ALL", "ALL:unseen", "ALL:seen"]
    for fam in groups:
        base, _, part = fam.partition(":")
        ids = [c for c in common if (base == "ALL" or A[c].get("family") == base)
               and (not part or (c in seen and seen[c] == (part == "seen")))]
        if not ids:
            continue
        pa = [A[c].get("verdict") in PASSING for c in ids]
        pb = [B[c].get("verdict") in PASSING for c in ids]
        both = sum(1 for x, y in zip(pa, pb) if x and y)
        only_a = sum(1 for x, y in zip(pa, pb) if x and not y)
        only_b = sum(1 for x, y in zip(pa, pb) if y and not x)
        rec = {"n": len(ids), names[0]: sum(pa), names[1]: sum(pb), "both": both, f"{names[0]}_only": only_a,
               f"{names[1]}_only": only_b, "neither": len(ids) - both - only_a - only_b,
               "mcnemar_p": round(mcnemar_p(only_a, only_b), 4),
               f"{names[0]}_ci": wilson(sum(pa), len(ids)), f"{names[1]}_ci": wilson(sum(pb), len(ids))}
        out["families"][fam] = rec
        print(f"{fam:>4}: n {len(ids)}  {names[0]} {sum(pa)}  {names[1]} {sum(pb)}  both {both}  "
              f"{names[0]}-only {only_a}  {names[1]}-only {only_b}  McNemar p {rec['mcnemar_p']}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return {"run": cmd_run, "status": cmd_status, "plan3": cmd_plan3, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
