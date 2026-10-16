"""r4cases — one case file for the r4 retrain (docs/LAB.md §5b).

The r4 families were authored by several workflow agents, each with its own
spelling of the case schema (cases-F1a/F1b/F1c/F2F6/F3/F4F5.jsonl in the r4
scratch folder). `merge` reads them all, plus the 1,000 verbatim int-run3
cases (F0), and writes ONE file in one schema the collector, the scorer and
the dataset builder share:

    {id: "r4-<family>-<nnnn>", kind, target, fields, site_state, context, request,
     history: [{user, context, page, expect, assistant, ops, calls}],   # earlier turns, oldest first ([] = one turn)
     expected: {ops, diff, keys, alt, alt_exact, out_of_scope, candidates, direction, reply_any, expect_hidden},
     score: {mode: exact | keys | where | clarify | answer},
     holdout, manifest_hash, engine_fingerprint, template,
     meta: {family, orig_id, source_file, group, view, page, oracle_ops, candidates, …the author's meta}}

  exact    verify-2 on the exact diff (expected.alt_exact: other {ops, diff} that also pass, e.g. reset vs
           restore_default when they differ)
  keys     the exam's key rule (exam.score_turn) on expected.keys / alt (key sets), plus expected.direction
           ({key, dir: up|down|darker|lighter, from}) when the value is relative
  where    no write and report_unsupported.where ∈ expected.out_of_scope.expected_where
  clarify  no write, a question naming ≥ 2 of meta.candidates (verify's clarify_ok)
  answer   no write, the reply names one of expected.reply_any
  ask      no write and a question: a request too vague to act on (r4score.score_ask; contrast.py's vague requests
           and the follow-up family's vague first turns as scored turns)
  page     create_page on expected.page (exam.score_page): the create op as asked, or a decline (the page exists,
           the slug is reserved, the template can't show custom pages) — contrast.py's page family and the pool's
           relabelled page-create rows (retrain/prepare.fix_page_creates)
  list     a dropdown's choices named from the context list the turn was sent (r4score.score_listing;
           expected.listing {list, situation}; the case's `context_extras` — categories or any other member, None = a
           list the browser could not load — and `option_lists` {tags, brands} are laid over the context fixture
           when its context is derived: with_case_extras)

F0 (the verbatim cases) is REBASED onto the cases' common manifest (the exam's
`fixtures/exam/manifest-real.json`, PLAN §0.3 fallback): the state gets the
basic FAQ starter set and a rotated template (vintage ×2), the context is
re-derived with today's `state.context_of` (focus, shopMenu, outline), and
the expected diff is re-measured through the snapshot engine; a case whose ops
the engine refuses there (or that write nothing) is dropped and counted. The
235 int-run3 holdout cases are kept as family F0H (holdout: true): never
collected, never trained — an exam on the rebased state.

`group` is the train/val split unit: the int-run3 source case for F0 and its
F1 paraphrases (so a paraphrase never sits in val while its source trains),
else the case itself. After the merge: a dedup on (all user texts, page) and
the hold-out filter (`leak.py`) over every case; counts per family in the report.

    python -m souschef_lab.r4cases merge --in <scratch>/cases-F1a.jsonl … --verbatim data/int-run3/cases.jsonl \\
        --manifest fixtures/exam/manifest-real.json --why <study dir> --out <scratch>/cases-all.jsonl
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import FIXTURES_DIR
from .createpage import run_engine_result

MODES = ("exact", "keys", "where", "clarify", "answer", "ask", "page", "list")
FAMILY_ORDER = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F0H", "F9H", "F0X")  # F7 authored, F8 contrast.py
TEMPLATE_POOL = ("vintage", "vintage", "bold", "boutique", "cozy", "editorial", "elegant", "playful", "pluto", "retro", "sweet")
DEFAULT_MANIFEST = FIXTURES_DIR / "exam" / "manifest-real.json"
WHERE_DEFAULT = "support"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def h8(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


# --- normalising one authored case --------------------------------------------------------------------------------------

def _mode_of(c: dict) -> str:
    m = c.get("meta") or {}
    e = c.get("expected") or {}
    raw = m.get("scoring") or m.get("verdict_mode") or m.get("ground_truth") or m.get("score") or m.get("verify_mode")
    if raw in MODES:
        return raw
    if raw == "answer" or e.get("no_write"):
        return "answer"
    if c.get("kind") == "clarify":
        return "clarify"
    if e.get("out_of_scope") or c.get("kind") in ("unsupported", "out_of_scope"):
        return "where"
    if e.get("keys") and not e.get("diff"):
        return "keys"
    return "exact"


def _direction(c: dict) -> dict | None:
    m, e = c.get("meta") or {}, c.get("expected") or {}
    d = e.get("direction") or m.get("direction")
    if not d:
        return None
    frm = None
    for src in (e.get("current"), (m.get("oracle") or {}).get("current") if isinstance(m.get("oracle"), dict) else None):
        if src is not None and not isinstance(src, (dict, list)):
            frm = src
            break
    if isinstance(d, dict) and "dir" in d:
        return {"key": d.get("key"), "dir": d["dir"], "from": d.get("from", frm)}
    if isinstance(d, dict) and len(d) == 1:
        (k, v), = d.items()
        return {"key": k, "dir": v, "from": frm}
    if isinstance(d, str):
        keys = e.get("keys") or []
        return {"key": (keys[0] or {}).get("key") if keys else None, "dir": d, "from": frm}
    return None


def _oracle_ops(c: dict) -> list[dict]:
    """The writes a perfect answer makes (dry runs script them): the expected
    ops, else the author's representative / oracle ops for a key-scored case."""
    m, e = c.get("meta") or {}, c.get("expected") or {}
    if e.get("ops"):
        return copy.deepcopy(e["ops"])
    for v in (m.get("representative"), m.get("oracle"), {"ops": m.get("oracle_ops")}):
        if isinstance(v, dict) and isinstance(v.get("ops"), list) and v["ops"]:
            return copy.deepcopy(v["ops"])
    return []


def _history(c: dict) -> list[dict]:
    """Earlier turns, oldest first, from the author's `turns` (which carry
    `expect`) or `history`."""
    turns = c.get("turns") or []
    hist = c.get("history") or []
    out = []
    src = turns if turns else hist
    for i, t in enumerate(src):
        h = hist[i] if i < len(hist) else {}
        user = str(t.get("request") or t.get("user") or h.get("user") or "")
        ctx = t.get("context") if isinstance(t.get("context"), dict) else (h.get("context") if isinstance(h.get("context"), dict) else None)
        calls = t.get("calls") or h.get("calls") or []
        ops = t.get("ops") or [{"op": x.get("name"), **(x.get("input") or {})} for x in calls
                               if x.get("name") not in (None, "report_unsupported", "lookup_fields", "describe_target")]
        if not ops and i == len(src) - 1 and (c.get("meta") or {}).get("history_ops"):  # F3 conversation B
            ops = [{"op": x.get("name"), **(x.get("input") or {})} for x in c["meta"]["history_ops"]]
        expect = t.get("expect")
        if not expect:
            expect = ("report_unsupported" if any(x.get("name") == "report_unsupported" for x in calls)
                      else "write" if ops else "reply")
        out.append({"user": user, "context": ctx, "page": (ctx or {}).get("page") or (c.get("context") or {}).get("page"),
                    "expect": expect, "assistant": t.get("assistant") or h.get("assistant") or "",
                    "ops": copy.deepcopy(ops), "calls": copy.deepcopy(calls)})
    return out


def normalise(c: dict, family: str, source_file: str) -> dict:
    """An authored case in the merged schema (the author's fields kept under meta)."""
    c = copy.deepcopy(c)
    e = c.get("expected") or {}
    m = c.get("meta") or {}
    mode = _mode_of(c)
    alt = e.get("alt") or []
    alt_keys = [s for s in alt if isinstance(s, list)]
    alt_exact = [a for a in alt if isinstance(a, dict) and "ops" in a]
    oos = e.get("out_of_scope")
    if mode == "where":
        where = e.get("where") or (oos or {}).get("expected_where") or m.get("where") or [WHERE_DEFAULT]
        where = [w for w in (where if isinstance(where, list) else [where]) if w]
        oos = {"reason": (oos or {}).get("reason") or m.get("reason") or "out_of_editor",
               "where": (oos or {}).get("where") or where, "expected_where": where}
    else:
        oos = None
    candidates = e.get("candidates") or m.get("candidates")
    exp = {"ops": e.get("ops") or [], "diff": e.get("diff") or [], "keys": e.get("keys") or [], "alt": alt_keys,
           "alt_exact": alt_exact, "out_of_scope": oos, "candidates": candidates, "direction": _direction(c),
           "reply_any": e.get("reply_any"), "expect_hidden": e.get("expect_hidden"), "gate": e.get("gate")}
    page = (c.get("context") or {}).get("page") or m.get("page") or "home"
    meta = dict(m)
    meta.update({"family": family, "orig_id": c.get("id"), "source_file": source_file, "page": page,
                 "view": m.get("view") or None, "oracle_ops": _oracle_ops(c), "author_meta_keys": sorted(m)})
    if candidates:
        meta["candidates"] = candidates
    src = m.get("source_case")
    meta["group"] = f"src:{src}" if (family in ("F0", "F1") and src) else f"case:{family}:{c.get('id')}"
    out = {"id": None, "kind": c.get("kind"), "target": c.get("target"), "fields": c.get("fields") or [],
           "site_state": c.get("site_state"), "context": c.get("context"), "request": c.get("request"),
           "history": _history(c), "expected": exp, "score": {"mode": mode},
           "holdout": bool(c.get("holdout")), "manifest_hash": c.get("manifest_hash"),
           "engine_fingerprint": c.get("engine_fingerprint"), "template": c.get("template"), "meta": meta}
    if mode == "where":
        out["kind"] = "unsupported" if out["kind"] not in ("unsupported", "out_of_scope") else out["kind"]
    if mode == "clarify":
        out["kind"] = "clarify"
    return out


def family_of_row(c: dict, default: str) -> str:
    f = str((c.get("meta") or {}).get("family") or default)
    return f if f in FAMILY_ORDER else default


# --- F0: the verbatim cases, rebased -----------------------------------------------------------------------------------

def rebase_f0(src: dict, manifest, extras: dict, engine, where_by_source: dict[str, list[str]]) -> tuple[dict | None, str]:
    """A verbatim int-run3 case on the r4 manifest, state and context (module
    docstring). (case, "") or (None, why it was dropped)."""
    from . import browser_defaults as BD
    from .exam import WHERE_VALUES, expected_where_for
    from .state import context_of
    from .verify import ctx_of

    c = copy.deepcopy(src)
    sid = str(c["id"])
    m = dict(c.get("meta") or {})
    family = "F0H" if c.get("holdout") else "F0"
    page = (c.get("context") or {}).get("page") or "home"
    tried = [TEMPLATE_POOL[h8(f"F0:{sid}") % len(TEMPLATE_POOL)]]
    if c.get("template") and c["template"] not in tried:
        tried.append(c["template"])
    kind = c.get("kind")
    why = "no_template"
    for t in tried:
        st = copy.deepcopy(c["site_state"])
        st["template"] = {"id": t}
        faq = BD.faq_categories(t)
        if faq is not None:
            st["faqCategories"] = faq
        case = {"id": sid, "kind": kind, "target": c.get("target"), "fields": c.get("fields") or [], "site_state": st,
                "request": c["request"], "holdout": bool(c.get("holdout")), "manifest_hash": manifest.hash,
                "template": t}
        case["context"] = context_of(st, page, extras, manifest, view=None, text=c["request"])
        exp: dict[str, Any] = {"ops": copy.deepcopy((c.get("expected") or {}).get("ops") or []), "diff": []}
        if kind == "unsupported":
            where = where_by_source.get(sid) or [w for w in expected_where_for(str(m.get("reason") or ""), c["request"])
                                                   if w in WHERE_VALUES] or [WHERE_DEFAULT]
            exp = {"ops": [], "diff": [], "out_of_scope": {"reason": m.get("reason"), "expected_where": where}, "where": where}
            mode = "where"
        elif kind == "clarify":
            exp["candidates"] = (c.get("expected") or {}).get("candidates") or m.get("candidates")
            mode = "clarify"
        else:
            ops = [o for o in exp["ops"] if o.get("op") != "pick_asset"]
            r = run_engine_result(engine, st, ops, ctx_of(case))
            if not r.ok:
                why = "op_refused"
                continue
            if not r.diff():
                why = "empty_diff"
                continue
            exp["diff"] = r.diff()
            m["measured"] = {"warnings": r.warnings, "written": r.written, "engine_commit": None}
            mode = "exact"
        case["expected"] = exp
        case["engine_fingerprint"] = getattr(engine, "fingerprint", None)
        m.update({"family": family, "style": "verbatim", "source_case": sid, "view": None, "page": page,
                  "rebased_from": {"manifest_hash": c.get("manifest_hash"), "template": c.get("template"),
                                   "engine_fingerprint": c.get("engine_fingerprint")},
                  "scoring": mode})
        case["meta"] = m
        return case, ""
    return None, why


# --- revisions (the verifier round of 2026-10-10; docs/LAB.md §5b) ---------------------------------------------------------

def load_revisions(path: Path | None) -> dict:
    """{drop: {orig_id: why}, rephrase: {orig_id: {request?, history?: {i: user}, ops?: [...], why}},
    to_keys: [orig_id], direction: {orig_id: {key, dir}}, bank_values: {leaf key: value},
    restyle_arrows: bool, trim: {f1_per_source, f3_where_per_value, reset_max_share}} — keyed by the
    AUTHOR's id (meta.orig_id), which does not move when the r4 ids do."""
    if not path:
        return {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}


def apply_text_revisions(c: dict, rev: dict) -> str | None:
    """Drop or rephrase one case by the revisions file. Returns a drop reason or None."""
    oid = str(c["meta"].get("orig_id"))
    if oid in (rev.get("drop") or {}):
        return "revision_drop"
    r = (rev.get("rephrase") or {}).get(oid)
    if r:
        old = {"request": c.get("request"), "history": [h.get("user") for h in c.get("history") or []]}
        if r.get("request"):
            c["request"] = r["request"]
        for i, u in (r.get("history") or {}).items():
            c["history"][int(i)]["user"] = u
        if r.get("ops") is not None:
            c["expected"]["ops"] = copy.deepcopy(r["ops"])
            c["meta"]["oracle_ops"] = copy.deepcopy(r["ops"])
        c["meta"]["rephrased"] = {"from": old, "why": r.get("why")}
    d = (rev.get("direction") or {}).get(oid)
    if d:
        c["expected"]["direction"] = {"key": d["key"], "dir": d["dir"], "from": d.get("from")}
    return None


def fix_bank_value(c: dict, bank_values: dict[str, str]) -> str | None:
    """Replace a generator TEXT_BANK value ("Open late every Friday") written
    to a key it makes no sense for (a sign-in button's processing text, a
    validation message …) with the key's realistic value from the revisions
    file, in the expected ops AND the request (where the value is quoted or
    typed). None, `bank_value_not_in_request` (left as it was) or "fixed:<leaf>"."""
    import re
    from .cases import TEXT_BANK

    if c["meta"]["family"] not in ("F0", "F1") or c.get("history"):
        return None
    hits = []
    for op in (c.get("expected") or {}).get("ops") or []:
        for k, v in _op_values(op):
            leaf = str(k).split(".")[-1]
            if isinstance(v, str) and _plain(v) in TEXT_BANK and leaf in bank_values:
                hits.append((op, k, v, leaf))
    if not hits:
        return None
    req = str(c.get("request") or "")
    for _, _, v, _ in hits:
        if _plain(v).lower() not in req.lower():
            return "bank_value_not_in_request"
    for op, k, v, leaf in hits:
        new = bank_values[leaf]
        newv = f"<p>{new}</p>" if v.startswith("<p>") else new
        if op.get("op") == "set_fields":
            op["values"][k] = newv
        else:
            op["item"][k] = newv
        req = re.sub(re.escape(_plain(v)), lambda m: new, req, flags=re.IGNORECASE)
    c["meta"]["bank_value_fixed"] = {"from": [h[2] for h in hits], "keys": [h[1] for h in hits], "request_was": c["request"]}
    c["request"] = req
    c["meta"]["oracle_ops"] = copy.deepcopy(c["expected"]["ops"])
    return "fixed:" + ",".join(h[3] for h in hits)


def restyle_arrow(c: dict) -> str | None:
    """F1's short commands were written "x → y" 168 times; merchants type a
    colon, "to", or an ASCII arrow. Deterministic by case id: 3/8 "x: y",
    3/8 "x to y", 2/8 "x -> y"."""
    req = str(c.get("request") or "")
    if c["meta"]["family"] != "F1" or "→" not in req:
        return None
    k = h8("arrow:" + str(c["meta"].get("orig_id"))) % 8
    sep = ": " if k < 3 else " to " if k < 6 else " -> "
    import re
    c["request"] = re.sub(r"\s*→\s*", sep, req)
    c["meta"]["arrow_restyled"] = req
    return sep.strip() or "to"


_NUM_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def _plain(v: Any) -> str:
    import re
    s = re.sub(r"<[^>]+>", "", str(v)).replace("’", "'").replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", s).strip()


def _op_values(op: dict) -> list[tuple[str, Any]]:
    if op.get("op") == "set_fields":
        return list((op.get("values") or {}).items())
    if op.get("op") == "edit_list" and isinstance(op.get("item"), dict):
        return list(op["item"].items())
    return []


def authored_values(c: dict) -> list[str]:
    """Why an exact case's expected value is not one the merchant's words fix
    (so a live teacher will not reproduce it and verify-2 would fail a good
    turn): `invented` — a sentence-like string (an FAQ question, a review, a
    proposal's copy) the user never typed as such, `punctuation_or_case` — the
    user typed it but not with that capitalisation / closing period, and
    `from_assistant` — a value only an earlier AUTHORED assistant turn named
    (an F2c proposal the live turn 1 will not repeat). One-word enum values
    ("centered", "pills") and fonts are named values, not authored text."""
    import re
    users = " \n ".join([str(c.get("request") or "")] + [str(h.get("user") or "") for h in c.get("history") or []])
    U = _plain(users)
    asst = " ".join(str(h.get("assistant") or "") for h in c.get("history") or []).lower()
    words = {_NUM_WORDS[w] for w in re.findall(r"[a-z]+", U.lower()) if w in _NUM_WORDS}
    why = set()
    for op in (c.get("expected") or {}).get("ops") or []:
        for _, v in _op_values(op):
            if isinstance(v, bool) or v is None or isinstance(v, (dict, list)):
                continue
            sv = _plain(v)
            if not sv or sv.startswith(("http", "/")):
                continue
            hexv = isinstance(v, str) and re.fullmatch(r"#[0-9a-fA-F]{3,8}", v)
            if isinstance(v, str) and not hexv:
                if sv in U:
                    continue
                sentence = bool(re.search(r"[.?!,;:]", sv)) or len(sv.split()) > 3
                core = re.sub(r"[.?!,;:]+$", "", sv)
                if sv.lower() in U.lower() or (core != sv and core.lower() in U.lower()):
                    if sentence:
                        why.add("punctuation_or_case")
                elif c.get("history") and sv.lower() in asst:
                    why.add("from_assistant")
                elif " " in sv:
                    why.add("invented")
            else:
                if str(v).lower() in U.lower() or (isinstance(v, (int, float)) and v in words):
                    continue
                if c.get("history") and str(v).lower() in asst:
                    why.add("from_assistant")
    return sorted(why)


def to_keys(c: dict, why: list[str]) -> None:
    """An exact case scored by keys instead (verify-3 keys mode): the keys its
    ops write (the authored ones kept when present) and, for list writes,
    `expected.list_ops` [{list, action, index}] — WHICH rows are touched is
    still checked, the wording is the teacher's. The ops stay as the oracle's."""
    from .exam import touches_of

    e = c["expected"]
    ops = [o for o in e.get("ops") or [] if o.get("op") != "pick_asset"]
    keys = list(e.get("keys") or [])
    if not keys:
        seen = set()
        for o in ops:
            if o.get("op") == "edit_list":
                k = {"target": o.get("target"), "key": "", "list": str(o.get("list") or "")}
            else:
                for t in touches_of(o):
                    k = {"target": t["target"], "key": t["key"], "list": t["list"]}
                    sk = json.dumps(k, sort_keys=True)
                    if sk not in seen:
                        seen.add(sk)
                        keys.append(k)
                continue
            sk = json.dumps(k, sort_keys=True)
            if sk not in seen:
                seen.add(sk)
                keys.append(k)
    lops = [{"list": str(o.get("list") or ""), "action": o.get("action"), "index": o.get("index")}
            for o in ops if o.get("op") == "edit_list"]
    e["keys"] = keys
    e["alt"] = [a for a in e.get("alt") or [] if isinstance(a, list)]
    e["alt_exact"] = []
    if lops:
        e["list_ops"] = lops
    c["meta"]["oracle_ops"] = copy.deepcopy(e.get("ops") or c["meta"].get("oracle_ops") or [])
    c["meta"]["exact_to_keys"] = why
    c["score"] = {"mode": "keys"}


def remeasure(c: dict, engine, manifest, extras: dict) -> str | None:
    """The case on the humanized state (state.humanize): every turn's context
    re-derived with state.context_of (earlier turns' ops replayed through the
    engine first), the exact diff (and each alt_exact's) re-measured, a
    key-scored case's oracle ops required to apply. None, or why the case
    cannot stand."""
    from .state import context_of, humanize
    from .verify import ctx_of

    def strip(ops):
        return [o for o in ops or [] if o.get("op") != "pick_asset"]

    c["site_state"] = humanize(copy.deepcopy(c["site_state"]))
    state = copy.deepcopy(c["site_state"])
    extras = with_case_extras(extras, c)
    view = c["meta"].get("view") or None
    page0 = (c.get("context") or {}).get("page") or c["meta"].get("page") or "home"
    for i, h in enumerate(c.get("history") or []):
        page = h.get("page") or page0
        h["context"] = context_of(state, page, extras, manifest, view=view, text=h.get("user") or "")
        ops = strip(h.get("ops"))
        if ops:
            r = run_engine_result(engine, state, ops, ctx_of({"site_state": state, "context": h["context"],
                                                              "template": c.get("template")}), print_state=True)
            if not r.ok or not r.diff() or not r.state:
                return f"history_{i}_refused"
            state = r.state
    page = c["meta"].get("page") or (c.get("context") or {}).get("page") or "home"
    c["context"] = context_of(state, page, extras, manifest, view=view, text=c.get("request") or "")
    fc = {"site_state": state, "context": c["context"], "template": c.get("template")}
    mode = c["score"]["mode"]
    e = c["expected"]
    if mode == "exact":
        r = run_engine_result(engine, state, strip(e.get("ops")), ctx_of(fc))
        if not r.ok:
            return "exact_refused"
        if not r.diff():
            return "exact_empty"
        e["diff"] = r.diff()
        c["meta"]["measured"] = {**(c["meta"].get("measured") or {}), "warnings": r.warnings, "written": r.written,
                                 "remeasured": "humanized-state"}
        alts = []
        for a in e.get("alt_exact") or []:
            ra = run_engine_result(engine, state, strip(a.get("ops")), ctx_of(fc))
            if ra.ok and ra.diff():
                alts.append({**a, "diff": ra.diff()})
        e["alt_exact"] = alts
    elif mode == "keys":
        ops = strip(c["meta"].get("oracle_ops") or e.get("ops"))
        if ops:
            r = run_engine_result(engine, state, ops, ctx_of(fc))
            if not r.ok or not r.diff():
                return "oracle_refused"
            c["meta"]["measured"] = {**(c["meta"].get("measured") or {}), "warnings": r.warnings, "written": r.written,
                                     "remeasured": "humanized-state"}
    elif mode == "page":
        return measure_page(c, engine, state, fc)
    elif mode == "list":
        from .r4score import listing_view
        want = (e.get("listing") or {}).get("situation")
        got = listing_view(c)["situation"]
        if want and got != want:
            return f"listing_{got}_not_{want}"  # the derived context does not carry the list the case is about
    return None


def with_case_extras(extras: dict | None, c: dict) -> dict:
    """The context fixture with the case's own lists laid over it: `option_lists` ({tags, brands}: the carousel
    lists the browser host holds, a missing or null kind = not loaded — state.option_lists_of, the same lists the
    engine ctx carries) and `context_extras` (any other member, e.g. a listing case's categories). A None list is
    one the context leaves out."""
    ol, own = c.get("option_lists"), c.get("context_extras")
    if not isinstance(ol, dict) and not (isinstance(own, dict) and own):
        return extras or {}
    out = dict(extras or {})
    if isinstance(ol, dict):
        for k in ("tags", "brands"):
            out[k] = copy.deepcopy(ol[k]) if isinstance(ol.get(k), list) else None
    if isinstance(own, dict):
        out.update(copy.deepcopy(own))
    return out


def measure_page(c: dict, engine, state: dict, fc: dict) -> str | None:
    """A page case on its state: a create's oracle ops (create_page + the asked section adds) must apply and create
    the page the spec names; a decline's attempted create (meta.attempt) must be REFUSED by the engine (the page
    exists, the slug is reserved, the template is not basic) — so the case is true on this engine."""
    from .exam import page_expectation, score_page
    from .verify import ctx_of
    spec = page_expectation(c) or {}
    ops = [o for o in c["meta"].get("oracle_ops") or [] if o.get("op") != "pick_asset"]
    if spec.get("expect") == "create":
        if not ops:
            return "page_no_oracle"
        r = run_engine_result(engine, state, ops, ctx_of(fc))
        if not r.ok:
            return "page_oracle_refused"
        res = [{"ok": x.get("ok"), "slug": x.get("slug"), "warnings": x.get("warnings") or []}
               for x in list(getattr(r, "results", None) or [])]
        row = {"ops": ops, "engine_results": [{"results": res}], "reply": "", "tool_calls": [], "context": c.get("context")}
        v = score_page(c, row)
        if v["verdict"] != "pass":
            return f"page_oracle_{v['verdict']}"
        c["meta"]["measured"] = {**(c["meta"].get("measured") or {}), "warnings": r.warnings, "written": r.written,
                                 "remeasured": "humanized-state"}
        return None
    attempt = [o for o in c["meta"].get("attempt") or [] if isinstance(o, dict)]
    if attempt:
        r = run_engine_result(engine, state, attempt, ctx_of(fc))
        first = (list(getattr(r, "results", None) or []) or [{}])[0]
        if first.get("ok"):
            return "page_decline_accepted"  # the engine creates it: not a decline on this engine
        code = (first.get("error") or {}).get("code")
        want = "E_TARGET" if spec.get("why") == "template" else "E_SLUG"
        if code != want:  # E_OP: an engine without create_page — the case would teach nothing true
            return f"page_refused_{code}_not_{want}"
        c["meta"]["measured"] = {**(c["meta"].get("measured") or {}), "refusal": code, "remeasured": "humanized-state"}
    return None


def write_ids(c: dict, manifest) -> set[str]:
    """What a case writes, as `<container>.<key>` identities on the generator's
    holdout-key grain (container = section TYPE, panel, page, branding; a list
    row's field also as `<container>.<list>[].<field>` and `<container>.<field>`
    for the hero's slide mirror)."""
    from .exam import touches_of

    def ctr(t: dict) -> str:
        t = t or {}
        k = t.get("kind")
        if k == "section":
            return manifest.type_of(str(t.get("sectionId") or "")) or str(t.get("sectionId") or "")
        if k == "global":
            return str(t.get("panel") or "global")
        if k == "branding":
            return "branding"
        return str(t.get("page") or k or "")

    out: set[str] = set()
    e = c.get("expected") or {}
    ops = list(e.get("ops") or []) + list((c.get("meta") or {}).get("oracle_ops") or [])
    for o in ops:
        if o.get("op") == "pick_asset":
            continue
        for t in touches_of(o):
            base = ctr(t["target"])
            lst = str(t.get("list") or "").replace("[]", "")
            if lst:
                out.add(f"{base}.{lst}")
                if t["key"]:
                    out.add(f"{base}.{lst}[].{t['key']}")
                    out.add(f"{base}.{t['key']}")
            elif t["key"]:
                out.add(f"{base}.{t['key']}")
    for k in e.get("keys") or []:
        if isinstance(k, dict):
            base = ctr(k.get("target"))
            lst = str(k.get("list") or "").replace("[]", "")
            out.add(f"{base}.{lst}" if lst and not k.get("key") else f"{base}.{k.get('key')}")
    hk = (c.get("meta") or {}).get("holdout_key")
    if hk:
        out.add(str(hk))
    return {x for x in out if not x.endswith(".")}


def tag_seen(exam_cases: list[dict], trained: list[dict], manifest) -> dict:
    """meta.key_seen / meta.seen_keys on every exam case (F0H, F0X): does a
    TRAINABLE r4 case write what it writes? The generated exam asks keys the
    verbatim training never saw; r4's designed families (F2–F6) do write some
    of them (headingColor, cardPriceColor …), so the student-alone comparison
    reads the UNSEEN subset (EVAL.md) and reports the seen one beside it."""
    pool: dict[str, int] = Counter()
    for c in trained:
        for k in write_ids(c, manifest):
            pool[k] += 1
    n_seen = 0
    for c in exam_cases:
        if c.get("kind") in ("unsupported", "clarify") or (c.get("score") or {}).get("mode") in ("where", "clarify", "list"):
            mine = set()
        else:
            mine = write_ids(c, manifest)
        seen = sorted(k for k in mine if pool.get(k))
        c["meta"]["key_seen"] = bool(seen)
        c["meta"]["seen_keys"] = seen
        n_seen += bool(seen)
    return {"cases": len(exam_cases), "seen": n_seen, "unseen": len(exam_cases) - n_seen}


def trim(cases: list[dict], cfg: dict, report: dict) -> list[dict]:
    """The kind mix (verifier round): resets at most `reset_max_share` of the
    trainable cases (the exam asks 2 resets in 252 requests; int-run3's
    generator made them a fifth) — first one reset per source group, then by a
    hash rank; F3's single-turn refusals at most `f3_where_per_value` per
    `where` value (conversations kept). Deterministic."""
    if not cfg:
        return cases

    def is_reset(c: dict) -> bool:
        ops = [o for o in (c["expected"].get("ops") or c["meta"].get("oracle_ops") or []) if o.get("op") != "pick_asset"]
        return bool(ops) and all(o.get("op") in ("reset", "restore_default") for o in ops) and not c.get("history")

    dropped: Counter = Counter()
    out = list(cases)
    f1_per = cfg.get("f1_per_source")
    if f1_per:  # F1b's and F1c's source ranges overlapped: some sources got 3-4 paraphrases, others none
        by_src: dict[str, list[dict]] = defaultdict(list)
        for c in out:
            if c["meta"]["family"] == "F1":
                by_src[str(c["meta"].get("group"))].append(c)
        cut = set()
        for g, cs in by_src.items():
            for c in sorted(cs, key=lambda x: h8("trim:" + x["id"]))[f1_per:]:
                cut.add(c["id"])
                dropped["F1_over_per_source"] += 1
        out = [c for c in out if c["id"] not in cut]
    per = cfg.get("f3_where_per_value")
    if per:
        by: dict[str, list[dict]] = defaultdict(list)
        for c in out:
            if c["meta"]["family"] == "F3" and c["score"]["mode"] == "where" and not c.get("history"):
                w = (c["expected"].get("out_of_scope") or {}).get("expected_where") or ["?"]
                by[w[0]].append(c)
        cut = set()
        for w, cs in by.items():
            for c in sorted(cs, key=lambda x: h8("trim:" + x["id"]))[per:]:
                cut.add(c["id"])
                dropped[f"F3_where:{w}"] += 1
        out = [c for c in out if c["id"] not in cut]
    share = cfg.get("reset_max_share")
    if share:
        train = [c for c in out if not c.get("holdout")]
        resets = [c for c in train if is_reset(c)]
        limit = int(share * len(train))
        first, rest, seen_g = [], [], set()
        for c in sorted(resets, key=lambda x: h8("trim:" + x["id"])):
            g = c["meta"].get("group")
            (rest if g in seen_g else first).append(c)
            seen_g.add(g)
        keep = {c["id"] for c in (first + rest)[:limit]}
        cut = {c["id"] for c in resets if c["id"] not in keep}
        for c in resets:
            if c["id"] in cut:
                dropped[f"reset:{c['meta']['family']}"] += 1
        out = [c for c in out if c["id"] not in cut]
        report["reset_cap"] = {"max_share": share, "limit": limit, "resets_before": len(resets), "kept": len(keep)}
    report["trimmed"] = dict(dropped)
    return out


# --- merge -------------------------------------------------------------------------------------------------------------

def dedup_key(c: dict) -> str:
    from .leak import tokens
    texts = [h.get("user") or "" for h in c.get("history") or []] + [c.get("request") or ""]
    return json.dumps([sorted(tokens(t)) for t in texts] + [(c.get("meta") or {}).get("page")])


def merge(inputs: list[Path], verbatim: Path | None, manifest_path: Path, why: Path | None, engine=None,
          progress=print, revisions: Path | None = None) -> tuple[list[dict], dict]:
    from .leak import ExamIndex, filter_cases
    from .manifest import load as load_manifest
    from .state import load_context_fixture

    manifest = load_manifest(str(manifest_path))
    report: dict[str, Any] = {"manifest": str(manifest_path), "manifest_hash": manifest.hash, "inputs": {},
                              "dropped": defaultdict(Counter)}
    authored: list[dict] = []
    where_by_source: dict[str, list[str]] = {}
    for p in inputs:
        rows = read_jsonl(p)
        fams = Counter()
        for r in rows:
            fam = family_of_row(r, "F1")
            fams[fam] += 1
            n = normalise(r, fam, p.name)
            if n["manifest_hash"] != manifest.hash:
                report["dropped"][fam]["other_manifest"] += 1
                continue
            authored.append(n)
            src = (r.get("meta") or {}).get("source_case")
            if src and n["score"]["mode"] == "where":
                where_by_source.setdefault(str(src), n["expected"]["out_of_scope"]["expected_where"])
        report["inputs"][p.name] = {"rows": len(rows), "families": dict(fams)}
        progress(f"r4cases: {p.name}: {len(rows)} rows {dict(fams)}")
    f0: list[dict] = []
    if verbatim:
        src_rows = read_jsonl(verbatim)
        extras = load_context_fixture()
        for r in src_rows:
            case, why_dropped = rebase_f0(r, manifest, extras, engine, where_by_source)
            fam = "F0H" if r.get("holdout") else "F0"
            if case is None:
                report["dropped"][fam][f"rebase_{why_dropped}"] += 1
                continue
            f0.append(normalise(case, fam, verbatim.name))
        report["inputs"][verbatim.name] = {"rows": len(src_rows), "families": dict(Counter("F0H" if r.get("holdout") else "F0" for r in src_rows))}
        progress(f"r4cases: {verbatim}: {len(src_rows)} verbatim → {len(f0)} rebased")
    allc = f0 + authored
    allc.sort(key=lambda c: FAMILY_ORDER.index(c["meta"]["family"]))
    # revisions: drops / rephrasings by the author's id, exact → keys where the value is authored, then every case on
    # the humanized state with re-derived contexts and re-measured diffs (remeasure)
    rev = load_revisions(revisions)
    extras = load_context_fixture()
    revised: list[dict] = []
    report["revisions"] = {"file": str(revisions) if revisions else None, "dropped": [], "rephrased": [],
                           "to_keys": Counter(), "remeasure_failed": [], "bank_values": Counter(), "arrows_restyled": 0}
    for i, c in enumerate(allc):
        fam = c["meta"]["family"]
        oid = str(c["meta"]["orig_id"])
        why_drop = apply_text_revisions(c, rev)
        if why_drop:
            report["dropped"][fam][why_drop] += 1
            report["revisions"]["dropped"].append({"orig_id": oid, "family": fam, "why": (rev.get("drop") or {}).get(oid)})
            continue
        if c["meta"].get("rephrased"):
            report["revisions"]["rephrased"].append({"orig_id": oid, "family": fam, "request": c["request"]})
        if rev.get("bank_values"):
            bv = fix_bank_value(c, rev["bank_values"])
            if bv:
                report["revisions"]["bank_values"][bv.split(":")[0]] += 1
        if rev.get("restyle_arrows") and restyle_arrow(c):
            report["revisions"]["arrows_restyled"] += 1
        if c["score"]["mode"] == "exact" and not c.get("holdout"):
            w = authored_values(c)
            if oid in (rev.get("to_keys") or []):
                w = sorted(set(w) | {"listed"})
            if w:
                to_keys(c, w)
                report["revisions"]["to_keys"][f"{fam}:{','.join(w)}"] += 1
        err = remeasure(c, engine, manifest, extras)
        if err:
            report["dropped"][fam][f"remeasure_{err}"] += 1
            report["revisions"]["remeasure_failed"].append({"orig_id": oid, "family": fam, "why": err})
            continue
        revised.append(c)
        if i % 250 == 0:
            progress(f"r4cases: revised {i}/{len(allc)}")
    report["revisions"]["to_keys"] = dict(report["revisions"]["to_keys"])
    report["revisions"]["bank_values"] = dict(report["revisions"]["bank_values"])
    seen: dict[str, str] = {}
    unique = []
    for c in revised:
        k = dedup_key(c)
        if k in seen:
            report["dropped"][c["meta"]["family"]]["duplicate"] += 1
            continue
        seen[k] = str(c["meta"]["orig_id"])
        unique.append(c)
    # r4 ids BEFORE the hold-out filter and the trims, so a stricter filter later leaves gaps instead of renumbering
    counters: Counter = Counter()
    for c in unique:
        fam = c["meta"]["family"]
        counters[fam] += 1
        c["id"] = f"r4-{fam}-{counters[fam]:04d}"
    index = ExamIndex(why=why)
    kept, leak = filter_cases(unique, index, family=lambda c: c["meta"]["family"])
    for d in leak["dropped"]:
        d["orig_id"] = next((c["meta"]["orig_id"] for c in unique if c["id"] == d["id"]), None)
    kept = trim(kept, rev.get("trim") or {}, report)
    report["seen"] = tag_seen([c for c in kept if c.get("holdout")], [c for c in kept if not c.get("holdout")], manifest)
    report["dropped"] = {k: dict(v) for k, v in report["dropped"].items()}
    report["leak"] = leak
    by = defaultdict(Counter)
    for c in kept:
        f = c["meta"]["family"]
        by[f]["cases"] += 1
        by[f][f"mode:{c['score']['mode']}"] += 1
        by[f]["multi_turn" if c["history"] else "single_turn"] += 1
        by[f]["history_turns"] += len(c["history"])
    report["families"] = {k: dict(v) for k, v in sorted(by.items(), key=lambda kv: FAMILY_ORDER.index(kv[0]))}
    trainable = [c for c in kept if not c["holdout"]]
    report["totals"] = {"cases": len(kept), "trainable": len(trainable), "holdout": len(kept) - len(trainable),
                        "turns_to_ask": sum(1 + len(c["history"]) for c in trainable),
                        "modes": dict(Counter(c["score"]["mode"] for c in trainable)),
                        "templates": dict(Counter(c["template"] for c in trainable))}
    return kept, report


def verbatim_exam(verbatim: Path, seen_from: Path | None = None, manifest_path: Path = DEFAULT_MANIFEST) -> list[dict]:
    """The int-run3 holdout cases AS THEY ARE (their own manifest, state and
    context — not rebased) in the merged schema, family F0X: the generated
    exam r3's 90.2 % student-only figure was measured on, for a like-for-like
    r3 / r4 comparison (r4collect --holdout on the student's route)."""
    out = []
    for r in read_jsonl(verbatim):
        if not r.get("holdout"):
            continue
        n = normalise(r, "F0X", verbatim.name)
        # scored exactly as r3's 90.2 % was: verify-2 for every kind (its unsupported_ok needs no `where` — the
        # 10-04 cases name none — and its clarify_ok is the clarify rule)
        n["score"] = {"mode": "exact"}
        n["kind"] = r.get("kind")
        n["expected"]["ops"], n["expected"]["diff"] = (r.get("expected") or {}).get("ops") or [], (r.get("expected") or {}).get("diff") or []
        n["id"] = f"r4-F0X-{len(out) + 1:04d}"
        n["meta"]["family"] = "F0X"
        out.append(n)
    if seen_from:
        from .manifest import load as load_manifest
        trained = [c for c in read_jsonl(seen_from) if not c.get("holdout")]
        tag_seen(out, trained, load_manifest(str(manifest_path)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="souschef_lab.r4cases", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    vx = sub.add_parser("verbatim-exam", help="the int-run3 holdout cases unchanged (family F0X) for the student-alone exam")
    vx.add_argument("--verbatim", type=Path, required=True)
    vx.add_argument("--out", type=Path, required=True)
    vx.add_argument("--seen-from", type=Path, help="the merged cases: tag meta.key_seen by what their trainable cases write")
    mg = sub.add_parser("merge")
    mg.add_argument("--in", dest="inputs", action="append", type=Path, required=True)
    mg.add_argument("--verbatim", type=Path, help="the verbatim cases (data/int-run3/cases.jsonl) → F0 / F0H")
    mg.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    mg.add_argument("--why", type=Path, help="the paraphrase study folder (real conversations for the hold-out check)")
    mg.add_argument("--engine", type=Path, help="headless.mjs (default $ENGINE / $COOKIE_CUTTER_DIR)")
    mg.add_argument("--revisions", type=Path, help="drops / rephrasings / keys conversions / trims by the author's id")
    mg.add_argument("--out", type=Path, required=True)
    mg.add_argument("--report", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "verbatim-exam":
        rows = verbatim_exam(a.verbatim, a.seen_from)
        write_jsonl(a.out, rows)
        print(f"r4cases: {len(rows)} verbatim holdout cases → {a.out} "
              f"({dict(Counter(r['score']['mode'] for r in rows))}; key seen in training: "
              f"{sum(1 for r in rows if r['meta'].get('key_seen'))})")
        return 0
    from .engine import PersistentEngine, default_cache_dir
    eng = PersistentEngine(a.engine, manifest_path=a.manifest, size=4, cache_dir=default_cache_dir())
    try:
        kept, report = merge(a.inputs, a.verbatim, a.manifest, a.why, eng, progress=lambda s: print(s, file=sys.stderr),
                             revisions=a.revisions)
    finally:
        eng.close()
    report["engine"] = eng.stats()
    write_jsonl(a.out, kept)
    rp = a.report or a.out.with_name(a.out.stem + "-report.json")
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"families": report["families"], "totals": report["totals"], "dropped": report["dropped"],
                      "leak": {k: report["leak"][k] for k in ("by_family", "total")}, "trimmed": report.get("trimmed"),
                      "seen": report.get("seen"), "to_keys": report["revisions"]["to_keys"]}, indent=1))
    print(f"r4cases: {len(kept)} cases → {a.out}; report {rp}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
