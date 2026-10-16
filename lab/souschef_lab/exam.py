"""exam — the real-phrasing exam: sets, an offline retrieval score, an
end-to-end score and a replay of real conversations (docs/LAB.md §6b).

The generated exam (cases.py) quotes the editor's labels word for word, is
single-turn and uses one template, so it cannot see how merchants actually
ask ("change the site name", "yes please", "make the font nicer"). These
sets can. They are the 2026-10-10 paraphrase study's cases (written by the
study, not by merchants, plus 14 of the user's own turns), on the user's
real `vintage` site and the real manifest of that day (build 9cc1cd1,
`fixtures/exam/manifest-real.json`, recorded by hash in every row and in
`fixtures/exam/sets.json`).

Sets (`fixtures/exam/<set>.jsonl`; `souschef-lab exam sets` lists them):

  paraphrase            147 in-scope requests in everyday words (classes synonym,
                        unnamed, selected, relative, compound, followup)
  holdout                62 more, written after the study froze its prototype's tables
  real                   14 of the user's real turns with a known right key
  out_of_scope           13 requests the editor cannot do (products, prices, domain…)
  out_of_scope_holdout    4 more, written with the holdout
  followup               12 multi-turn follow-ups ("6", "yes please", "okay, lora"):
                        the paraphrase rows of class followup, with their history

A row is the lab's case schema (LAB.md §2) with what a phrasing case needs
instead of a measured diff:

  {id, set, kind: "phrasing" | "out_of_scope", target, fields, site_state: null,
   context_ref, request, history: [{user, assistant, context_ref?}],
   expected: {ops: [], diff: [], keys: [{target, key, list}], alt: [[…]], out_of_scope: null | {reason, where}},
   holdout: true, manifest_hash, engine_fingerprint: null, template: "vintage",
   meta: {class, source, page, context_source, selected_section_id?, generator}}

`expected.keys` is the set of (target, key, list) the request means (`list`
names a list; `key` "" = the list itself); `alt` holds whole alternative sets
(the SEO "Site Title" for "site name"). Values are not scored: "make it
rounder" has no one right number. `load_set` inflates `context` (and each
history entry's) from `fixtures/exam/contexts.json` and checks the manifest
file against the rows' hash.

Scores:

  retrieval  (no model) — recall@block: does the service's candidate-fields block
             (internal/fields through its CLI, fields.py) list every expected key
             under the right target, for the context and text the service would
             see? Also: target missed vs key missed, first target right, empty
             blocks, block size; for out_of_scope rows, how often a block is offered
             at all (an invitation to write something).
  end-to-end — every row through a service URL + route with collect.run_case (the
             browser emulator: engine-applied ops, tool results, pick_asset), scored
             on the ops the engine ACCEPTED: `pass` = every key of the expected set
             (or of one alt set) written and nothing outside expect ∪ alt;
             `stray_writes` = all written plus something else; `partial`,
             `wrong_target`, `no_write` (asked or answered instead), `invalid_op`
             (only refused ops). Out of scope: `unsupported_ok` (a pass) = no
             accepted write AND the reply (or report_unsupported's reason /
             manual_hint) names a place to go — one of PLACE_TERMS or the row's own
             `where` words; `wrote` / `no_pointer` fail. A row with `history`
             replays it first: each earlier user text is sent as a live turn of the
             same conversation (the engine state carries over), then the scored
             turn (`--history live`, the default; `last` = the previous turn only,
             `none` = single turn). The service has no way to seed a recorded
             assistant reply, so a live replay re-asks the model; the row records
             each replayed turn's reply beside the recorded one.
  replay     — `exam replay --conversation conv.json` turns a GET
             /v1/conversations/{siteId} body into exam rows: one per user turn, its
             recorded context, text, the history up to it (user text + final
             assistant text, and the raw stored messages), and what was recorded
             (tool calls, results, reply, route, escalation). `--labels set.jsonl`
             attaches expected keys by (text, page) so the rows can be scored.

No command here spends anything by itself: `run` talks to whatever `--url`
names (a real service spends model calls), `--fake` runs the same path against
the lab's in-process fake service (tests/fake_service.py) — with `--oracle`
it answers every row with its expected writes, which proves the wiring and
that today's engine accepts every expected key on the vintage state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import FIXTURES_DIR, LAB_DIR, RUNS_DIR

EXAM_DIR = FIXTURES_DIR / "exam"
MANIFEST_FILE = EXAM_DIR / "manifest-real.json"
CONTEXTS_FILE = EXAM_DIR / "contexts.json"
INDEX_FILE = EXAM_DIR / "sets.json"
GENERATOR = "exam-1"
TEMPLATE = "vintage"
SETS = ("paraphrase", "holdout", "real", "out_of_scope", "out_of_scope_holdout", "followup")
CLASS_ORDER = ("real", "synonym", "unnamed", "selected", "relative", "compound", "followup", "oos")
HISTORY_MODES = ("live", "last", "none")
OOS_VERDICTS = ("unsupported_ok", "wrong_where", "wrote", "no_pointer")  # an out-of-scope row's verdicts
# Where a merchant is sent for what the editor cannot do. Destinations only — never the subject of the
# request ("domain", "price"), which a refusal repeats without pointing anywhere.
PLACE_TERMS = ("breadstack", "crm", "dashboard", "admin", "back office", "back-office", "backoffice", "pos",
               "point of sale", "point-of-sale", "store settings", "store admin", "account settings", "support",
               "support team", "catalog manager", "product catalog", "inventory system", "your provider",
               "registrar", "dns", "google business")
# A whole-word match for the short terms ("pos" must not match "post" / "possible").
_WORD = {t: re.compile(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])") for t in PLACE_TERMS}
# The study's out-of-scope reasons → extra place words per row (the row's own `where`).
WHERE_BY_REASON = (
    (("product", "catalog", "price"), ("catalog", "inventory", "products page in breadstack")),
    (("domain", "slug"), ("domain settings", "dns", "registrar", "site settings")),
    (("template",), ("template picker", "launcher", "theme picker")),
    (("promotion", "discount", "loyalty", "CRM"), ("promotions", "marketing", "loyalty settings")),
    (("messaging",), ("marketing", "email tool", "klaviyo")),
    (("payment",), ("payment settings", "payments")),
    (("location", "store config"), ("location settings", "store settings")),
    (("analytics",), ("reports", "analytics")),
    (("external",), ("google business profile", "google business")),
)

# report_unsupported's `where` (shared contract 1 with the launcher: service internal/toolgen UnsupportedWhere,
# cookie-cutter launcher/client/src/sousChef/nav/deepLinks.js WHERE — same order). The browser maps a value to a
# fixed label and link; anything else is ignored there (and here: the keyword rule scores the turn).
WHERE_VALUES = ("domain", "sites", "pages", "assets", "seo", "themes", "products", "categories", "brands", "support")
# An out-of-scope row's `expected_where` from its reason + request, first rule that matches; else `support`
# ("billing, payments, orders, refunds, tax, anything else"). The meanings are the rules' (prompt/rules.go
# "What Sous Chef cannot change"): domain = custom domain / web address / site slug, themes = switch template,
# pages = create/rename/duplicate/delete pages, categories = create/delete product categories, products =
# products, prices, inventory, descriptions; payments, discounts, loyalty, messaging, locations, analytics and
# external listings are `support`.
EXPECTED_WHERE_RULES = (
    (("template", "theme"), "themes"),
    (("domain", "slug", "url of my site", "web address"), "domain"),
    (("custompages", "new page", "page for"), "pages"),
    (("categor",), "categories"),
    (("sitemap", "search console"), "seo"),
    (("asset library", "upload"), "assets"),
    (("brand record",), "brands"),
    (("product", "catalog", "price", "inventory"), "products"),
)

PINNED_SECTION = {"faq": "__pinned_faq", "products": "__pinned_product_list",
                  "product-details": "__pinned_product_detail", "cart": "__pinned_cart",
                  "checkout": "__pinned_checkout", "login": "__pinned_login", "account": "__pinned_account",
                  "privacy-policy": "__pinned_privacy", "blog": "__pinned_blog_list",
                  "order-confirmation": "__pinned_order_confirmation"}
STUDY_FILES = {"paraphrase": "paraphrase-cases.jsonl", "holdout": "paraphrase-holdout.jsonl", "real": "real-cases.jsonl"}


class ExamError(RuntimeError):
    """A set, manifest or conversation that cannot be used."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def manifest_hash_of(path: Path) -> str:
    from .collect import manifest_hash
    return manifest_hash(json.loads(Path(path).read_text(encoding="utf-8")))


# --- conversations (GET /v1/conversations/{siteId}) ------------------------------------------------------------------------

def _texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    return [str(b.get("text") or "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text"]


def conversation_turns(conv: dict) -> list[dict]:
    """The user turns of a conversation body, in order: `{index, seq, context,
    candidates, text, reply, calls, messages_before, turn}`. A turn starts at a
    user message with a text block (tool-result-only user messages continue the
    turn before); its reply is the assistant text after it, its calls the
    tool_use blocks with their results. `turn` is the matching `turns[]` row
    (by order; None when the counts differ) without `userEmail`."""
    from .fields import split_user_text

    msgs = sorted((m for m in conv.get("messages") or [] if isinstance(m, dict)), key=lambda m: m.get("seq") or 0)
    meta_turns = [{k: v for k, v in t.items() if k != "userEmail"} for t in conv.get("turns") or [] if isinstance(t, dict)]
    starts = [i for i, m in enumerate(msgs) if m.get("role") == "user" and _texts(m.get("content"))]
    out = []
    for n, i in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(msgs)
        raw = "\n".join(_texts(msgs[i].get("content")))
        try:
            _, context, candidates, text = split_user_text(raw)
        except ValueError:
            context, candidates, text = None, "", raw
        reply_parts, calls, results = [], [], {}
        for m in msgs[i + 1:end]:
            for b in m.get("content") or []:
                if not isinstance(b, dict):
                    continue
                if m.get("role") == "assistant" and b.get("type") == "text":
                    reply_parts.append(str(b.get("text") or ""))
                elif m.get("role") == "assistant" and b.get("type") == "tool_use":
                    calls.append({"id": b.get("id"), "name": b.get("name"), "input": b.get("input")})
                elif b.get("type") == "tool_result":
                    results[b.get("tool_use_id")] = {"is_error": bool(b.get("is_error")),
                                                     "content": "\n".join(_texts(b.get("content")))}
        for c in calls:
            c["result"] = results.get(c["id"])
        last_text = ""
        for m in reversed(msgs[i + 1:end]):
            t = [x for x in _texts(m.get("content")) if x] if m.get("role") == "assistant" else []
            if t:
                last_text = "\n".join(t)
                break
        out.append({"index": n, "seq": msgs[i].get("seq"), "context": context, "candidates": candidates, "text": text,
                    "reply": last_text, "reply_all": reply_parts, "calls": calls, "messages_before": msgs[:i],
                    "turn": meta_turns[n] if len(meta_turns) == len(starts) else None})
    return out


def replay_rows(conv: dict, labels: Iterable[dict] = (), *, transcript: bool = True) -> list[dict]:
    """Exam rows rebuilt from a conversation: one per user turn, with the
    history up to it. `labels` (exam rows) give expected keys by (text, page)."""
    turns = conversation_turns(conv)
    conv_id = str(conv.get("conversationId") or "conversation")
    by_text: dict[tuple[str, str], dict] = {}
    for lab in labels:
        page = str((lab.get("context") or {}).get("page") or (lab.get("meta") or {}).get("page") or "")
        by_text.setdefault((norm_text(lab.get("request")), page), lab)
        by_text.setdefault((norm_text(lab.get("request")), ""), lab)
    rows = []
    for t in turns:
        ctx = t["context"] if isinstance(t["context"], dict) else {}
        page = str(ctx.get("page") or "")
        lab = by_text.get((norm_text(t["text"]), page)) or by_text.get((norm_text(t["text"]), ""))
        history = [{"user": h["text"], "assistant": h["reply"], "context": h["context"]} for h in turns[:t["index"]]]
        meta_turn = t["turn"] or {}
        row = {"id": f"replay-{conv_id[:8]}-{t['index']:02d}", "set": "replay",
               "kind": (lab or {}).get("kind") or "replay", "target": (lab or {}).get("target"),
               "fields": list((lab or {}).get("fields") or []), "site_state": None, "context": ctx,
               "request": t["text"], "history": history,
               "expected": copy.deepcopy((lab or {}).get("expected")) or {"ops": [], "diff": [], "keys": [], "alt": [],
                                                                         "out_of_scope": None},
               "holdout": True, "manifest_hash": meta_turn.get("manifestHash"), "engine_fingerprint": None,
               "template": str(ctx.get("template") or TEMPLATE),
               "meta": {"class": (lab or {}).get("meta", {}).get("class") or "replay", "page": page,
                        "context_source": "recorded", "labelled_by": (lab or {}).get("id"),
                        "source": f"conversation {conv_id} turn {t['index']} (seq {t['seq']})",
                        "generator": GENERATOR,
                        "recorded": {"turn": meta_turn or None, "candidate_fields": t["candidates"],
                                     "calls": t["calls"], "reply": t["reply"]}}}
        if transcript:
            row["meta"]["transcript"] = t["messages_before"]
        rows.append(row)
    return rows


def served_block_parity(rows: list[dict], manifest: Path, exe: Path) -> dict:
    """How many replayed turns' served candidate-fields blocks (recorded in
    the conversation) today's retrieval reproduces byte for byte — the check
    that the offline score measures what the route actually sent. Turns on
    another manifest than `manifest`, or recorded without a block (a route
    with fields_context off), are not compared."""
    have = manifest_hash_of(manifest)
    todo = [r for r in rows if (r.get("manifest_hash") in (None, have)) and r["meta"]["recorded"].get("candidate_fields")]
    res = run_cli(exe, manifest, [(r["id"], r.get("context"), r.get("request") or "") for r in todo]) if todo else {}
    differ = [r["id"] for r in todo if res[r["id"]]["block"].strip() != r["meta"]["recorded"]["candidate_fields"].strip()]
    return {"checked": len(todo), "same": len(todo) - len(differ), "differ": differ,
            "other_manifest": sum(1 for r in rows if r.get("manifest_hash") not in (None, have))}


def norm_text(s: Any) -> str:
    return " ".join(str(s or "").lower().split())


# --- building the sets from the study ---------------------------------------------------------------------------------------

def _context_key(page: str, source: str) -> str:
    return f"{TEMPLATE}-{page}" + ("" if source != "approximated" else "-approx")


def page_contexts(home: dict, convs: list[dict]) -> dict[str, tuple[dict, str]]:
    """page → (context, source): `recorded` for pages the user's real turns
    were on (the first recorded turn per page, the home fixture for home),
    else `approximated` — the home context's site lists with that page, layout
    'default' and no sections (the browser never lists a pinned section)."""
    out: dict[str, tuple[dict, str]] = {"home": (home, "recorded")}
    for conv in convs:
        for t in conversation_turns(conv):
            ctx = t["context"]
            if isinstance(ctx, dict) and ctx.get("page") and ctx["page"] not in out:
                out[str(ctx["page"])] = (ctx, "recorded")
    return out


def approximate_context(home: dict, page: str) -> dict:
    ctx = {"page": page, "template": home.get("template") or TEMPLATE, "layout": "default", "sections": []}
    for k in ("categories", "customPages", "nav", "locations"):
        ctx[k] = copy.deepcopy(home.get(k) or [])
    return ctx


def where_for(reason: str) -> list[str]:
    low = reason.lower()
    out: list[str] = []
    for keys, words in WHERE_BY_REASON:
        if any(k.lower() in low for k in keys):
            out += [w for w in words if w not in out]
    return out


def expected_where_for(reason: str, request: str = "") -> list[str]:
    """The `where` values that pass an out-of-scope row (contract 1 keys):
    the first EXPECTED_WHERE_RULES entry whose word is in the reason or the
    request, else `support`."""
    low = f"{reason}\n{request}".lower()
    for words, where in EXPECTED_WHERE_RULES:
        if any(w in low for w in words):
            return [where]
    return ["support"]


def where_sent(row: dict) -> tuple[list[str], list[Any]]:
    """`(valid, raw)`: report_unsupported's `where` values in a turn —
    normalised the way the browser does (trimmed, lower case, a WHERE_VALUES
    member) — and every `where` the model put there, valid or not."""
    valid: list[str] = []
    raw: list[Any] = []
    for c in row.get("tool_calls") or []:
        if c.get("name") != "report_unsupported" or not isinstance(c.get("input"), dict) or "where" not in c["input"]:
            continue
        w = c["input"]["where"]
        raw.append(w)
        if isinstance(w, str) and w.strip().lower() in WHERE_VALUES and w.strip().lower() not in valid:
            valid.append(w.strip().lower())
    return valid, raw


def check_expectation(manifest, e: dict) -> str | None:
    """None when the manifest has the expected (target, key, list); else why not."""
    from .state import resolve_key

    t = e.get("target") or {}
    kind = t.get("kind")
    key, lst = str(e.get("key") or ""), str(e.get("list") or "")
    # What the engine resolves outside the manifest's own containers (ops/targets.js): a page's `sections` list,
    # page "seo" = the SEO details panel, and a category's patch under the global `categories` map. The oracle
    # dry run (`exam run --fake --oracle`) is the authoritative check: the engine applies every expected key.
    if kind == "page" and lst == "sections" and not key:
        return None
    if kind == "page" and t.get("page") == "seo":
        seo = manifest.detail("SeoPanel")
        return None if seo and key in seo.fields else f"the SEO panel has no key {key}"
    if kind == "global" and key.startswith("categories.") and key.count(".") >= 2:
        detail = manifest.detail("CategoryDetailsPanel")
        leaf = key.split(".", 2)[2]
        has_map = any("categories" in c.fields for c in manifest.containers("global"))
        ok = has_map and detail is not None and (leaf in detail.fields or f"item.{leaf}" in detail.fields)
        return None if ok else f"no category patch field {leaf}"
    containers = []
    if kind == "section":
        c = manifest.section(str(t.get("sectionId") or ""))
        containers = [c] if c else []
    elif kind == "global":
        c = manifest.panel(str(t.get("panel"))) if t.get("panel") else None
        containers = [c] if c else ([] if t.get("panel") else manifest.containers("global"))
    elif kind == "branding":
        containers = [manifest.branding]
    elif kind == "page":
        c = manifest.page(str(t.get("page") or ""))
        containers = [c] if c else []
    if not containers:
        return f"no container for target {json.dumps(t, sort_keys=True)}"
    for c in containers:
        if lst:
            spec = c.lists.get(lst)
            if spec is None:
                continue
            if not key or key in spec.item_fields or resolve_key(manifest, c, key):
                return None
        elif resolve_key(manifest, c, key):
            return None
    return f"{json.dumps(t, sort_keys=True)} has no {'list ' + lst + ' ' if lst else ''}{'key ' + key if key else ''}".strip()


def study_row(src: dict, set_name: str, source_file: str, ctx_refs: dict[str, str], ctx_sources: dict[str, str],
              manifest_hash: str, history_refs: list[dict] | None = None) -> dict:
    """One study case as an exam row."""
    cls = str(src.get("class") or "")
    page = "home" if src.get("ctx") == "vintage-home" else str(src.get("ctx"))
    oos = cls == "oos" or src.get("oos") is not None
    keys = [] if oos else [copy.deepcopy(e) for e in src.get("expect") or []]
    alt = [] if oos else [[copy.deepcopy(e) for e in s] for s in src.get("alt") or []]
    history = history_refs if history_refs is not None else (
        [{"user": src["prev"]["user"], "assistant": src["prev"]["assistant"]}] if src.get("prev") else [])
    meta = {"class": cls, "page": page, "source": f"{source_file}#{src['id']}", "context_source": ctx_sources[page],
            "generator": GENERATOR}
    if src.get("selected"):
        meta["selected_section_id"] = src["selected"]
    return {"id": src["id"], "set": set_name, "kind": "out_of_scope" if oos else "phrasing",
            "target": keys[0]["target"] if keys else None,
            "fields": [e["key"] or e["list"] for e in keys], "site_state": None, "context_ref": ctx_refs[page],
            "request": src["request"], "history": history,
            "expected": {"ops": [], "diff": [], "keys": keys, "alt": alt,
                         "out_of_scope": {"reason": str(src.get("oos") or ""), "where": where_for(str(src.get("oos") or "")),
                                          "expected_where": expected_where_for(str(src.get("oos") or ""), str(src["request"]))}
                         if oos else None},
            "holdout": True, "manifest_hash": manifest_hash, "engine_fingerprint": None, "template": TEMPLATE, "meta": meta}


def build_sets(why: Path, out: Path = EXAM_DIR) -> dict:
    """fixtures/exam from the study's folder: the sets, contexts.json, the
    real manifest (copied) and sets.json (sources and the manifest by hash).
    Every expected key is checked against the manifest; a miss refuses."""
    from .manifest import Manifest

    why = Path(why)
    need = [why / f for f in STUDY_FILES.values()] + [why / "ctx-vintage-home.json", why / "manifest-real.json"]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        raise ExamError(f"the study folder lacks {', '.join(missing)}")
    manifest_doc = json.loads((why / "manifest-real.json").read_text(encoding="utf-8"))
    manifest = Manifest(manifest_doc)
    mhash = manifest_hash_of(why / "manifest-real.json")
    home = json.loads((why / "ctx-vintage-home.json").read_text(encoding="utf-8"))
    convs = []
    for p in sorted(why.glob("conv-*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and doc.get("messages"):
            convs.append(doc)
    recorded = page_contexts(home, convs)
    study = {name: read_jsonl(why / f) for name, f in STUDY_FILES.items()}
    pages = sorted({"home" if r.get("ctx") == "vintage-home" else str(r.get("ctx")) for rows in study.values() for r in rows})
    contexts, refs, sources = {}, {}, {}
    for page in pages:
        if page in recorded:
            ctx, src = recorded[page]
        else:
            ctx, src = approximate_context(home, page), "approximated"
        refs[page], sources[page] = _context_key(page, src), src
        contexts[refs[page]] = ctx

    # The real rows: the recorded turn's own context and the turn before it, matched by (text, page).
    real_turns: dict[tuple[str, str], tuple[dict, dict | None]] = {}
    for conv in convs:
        ts = conversation_turns(conv)
        for t in ts:
            page = str((t["context"] or {}).get("page") or "")
            prev = ts[t["index"] - 1] if t["index"] else None
            real_turns.setdefault((norm_text(t["text"]), page), (t, prev))

    errors: list[str] = []
    sets: dict[str, list[dict]] = {k: [] for k in SETS}
    for name, rows in study.items():
        for src in rows:
            page = "home" if src.get("ctx") == "vintage-home" else str(src.get("ctx"))
            history = None
            row_refs, row_sources = refs, sources
            if name == "real":
                hit = real_turns.get((norm_text(src["request"]), page))
                if hit is None:
                    errors.append(f"real {src['id']}: no recorded turn says {src['request']!r} on {page}")
                    continue
                t, prev = hit
                if t["context"] != contexts[refs[page]]:  # a turn whose page context differs from the page's first
                    key = f"{TEMPLATE}-{page}-{src['id']}"
                    contexts[key] = t["context"]
                    row_refs = dict(refs, **{page: key})
                    row_sources = dict(sources, **{page: "recorded"})
                if prev is not None:
                    ppage = str((prev["context"] or {}).get("page") or page)
                    history = [{"user": prev["text"], "assistant": prev["reply"]}]
                    if prev["context"] != contexts.get(refs.get(ppage, "")):
                        key = f"{TEMPLATE}-{ppage}-{src['id']}-prev"
                        contexts[key] = prev["context"]
                        history[0]["context_ref"] = key
                    elif ppage != page:
                        history[0]["context_ref"] = refs[ppage]
            row = study_row(src, name, STUDY_FILES[name], row_refs, row_sources, mhash, history)
            if name == "real":
                row["meta"]["recorded_turn"] = (t.get("turn") or {}).get("id")
            for s in [row["expected"]["keys"]] + row["expected"]["alt"]:
                for e in s:
                    why_not = check_expectation(manifest, e)
                    if why_not:
                        errors.append(f"{name} {src['id']}: {why_not}")
            if name == "paraphrase":
                (sets["out_of_scope"] if row["kind"] == "out_of_scope" else sets["paraphrase"]).append(row)
                if row["meta"]["class"] == "followup":
                    sets["followup"].append(dict(copy.deepcopy(row), set="followup"))
            elif name == "holdout":
                (sets["out_of_scope_holdout"] if row["kind"] == "out_of_scope" else sets["holdout"]).append(row)
            else:
                sets["real"].append(row)
    for r in sets["out_of_scope"]:
        r["set"] = "out_of_scope"
    for r in sets["out_of_scope_holdout"]:
        r["set"] = "out_of_scope_holdout"
    if errors:
        raise ExamError("the study's expectations do not resolve on the manifest:\n  " + "\n  ".join(errors))

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / MANIFEST_FILE.name).write_bytes((why / "manifest-real.json").read_bytes())
    (out / CONTEXTS_FILE.name).write_text(json.dumps(contexts, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    index = {"generator": GENERATOR, "built_at": utc_now(),
             "about": "real-phrasing exam (docs/LAB.md §6b); rows written by the 2026-10-10 paraphrase study + the user's real turns",
             "manifest": {"file": MANIFEST_FILE.name, "manifest_hash": mhash,
                          "file_sha256": sha256_file(out / MANIFEST_FILE.name),
                          "generated_from": manifest.generated_from},
             "contexts": {"file": CONTEXTS_FILE.name, "pages": {p: {"ref": refs[p], "source": sources[p]} for p in pages}},
             "sources": {f: sha256_file(why / f) for f in sorted(STUDY_FILES.values())} | {
                 "ctx-vintage-home.json": sha256_file(why / "ctx-vintage-home.json")},
             "sets": {}}
    for name in SETS:
        rows = sets[name]
        write_jsonl(out / f"{name}.jsonl", rows)
        index["sets"][name] = {"file": f"{name}.jsonl", "rows": len(rows),
                               "with_history": sum(1 for r in rows if r["history"]),
                               "by_class": dict(sorted(Counter(r["meta"]["class"] for r in rows).items()))}
    (out / INDEX_FILE.name).write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return index


# --- loading ---------------------------------------------------------------------------------------------------------------

def set_path(name_or_path: str | Path, exam_dir: Path = EXAM_DIR) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".jsonl" or p.exists():
        return p
    if str(name_or_path) not in SETS:
        raise ExamError(f"unknown set {name_or_path!r}: one of {', '.join(SETS)}, 'all', or a .jsonl path")
    return Path(exam_dir) / f"{name_or_path}.jsonl"


def load_contexts(exam_dir: Path = EXAM_DIR) -> dict:
    p = Path(exam_dir) / CONTEXTS_FILE.name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def inflate(row: dict, contexts: dict, *, with_selected: bool = False) -> dict:
    """The row with `context` (and each history entry's) filled from its ref."""
    row = copy.deepcopy(row)
    if "context" not in row or row.get("context") is None:
        ref = row.get("context_ref")
        if ref is not None:
            if ref not in contexts:
                raise ExamError(f"{row.get('id')}: context {ref!r} is not in {CONTEXTS_FILE.name}")
            row["context"] = copy.deepcopy(contexts[ref])
    for h in row.get("history") or []:
        if h.get("context") is None and h.get("context_ref"):
            if h["context_ref"] not in contexts:
                raise ExamError(f"{row.get('id')}: history context {h['context_ref']!r} is not in {CONTEXTS_FILE.name}")
            h["context"] = copy.deepcopy(contexts[h["context_ref"]])
    sel = (row.get("meta") or {}).get("selected_section_id")
    if with_selected and sel and isinstance(row.get("context"), dict):
        # The 10-06 browser's member, and the focus the browser sends since
        # the 2026-10-10 context work (turnContext.js focusSummary): the
        # section editor open on the editor tab.
        row["context"]["selectedSectionId"] = sel
        row["context"]["focus"] = {"area": "editor", "sectionId": sel, "panel": ""}
    return row


# --- answer-key amendments -------------------------------------------------------------------------------------------------
#
# The sets are built from the study (`exam build`); a product decision that changes what is RIGHT for a row is an
# amendment here, applied by load_set, so a rebuild keeps it. An amended row gets a new id (`<id>.<rev>`): a drift
# baseline or an earlier eval scored on the old key never pairs with it (it leaves every Δ until re-calibrated), and
# its old id, keys and alt stay in meta.superseded (r4score.apply_font_policy "as_authored" restores them).
#
# font-all (2026-10-15, the user after the demo): a font request that names no part of the site changes EVERY site
# font (r4score.site_font_fields on this manifest and the vintage state: the six theme fonts plus the announcement
# bar's, which vintage sets; the cart and footer fonts are unset there and inherit, so writing them is allowed, not
# required; shopMenuFont is hidden on vintage's grid menu). Writing one of them is now `partial`. p013 ("i don't
# like the font", no font named) may also be answered with a question (`ask_ok`): asking which font is what the
# study's own follow-up p150 does on that same first turn. h059's history asked "headings and body?" and the
# merchant said "both": those two stay required, every site font is an alternative.
ANSWER_KEY = "font-all+vague-ask-2026-10-15"
AMENDED_ID_SUFFIX = ".k2"
_G = {"kind": "global"}
SITE_FONTS_VINTAGE = ("headingFont", "bodyFont", "navFont", "buttonFont", "productTitleFont", "productBodyFont",
                      "announcementBar.fontFamily")
SITE_FONTS_VINTAGE_OPTIONAL = ("footerBodyTextFontFamily", "cartTitleFont", "cartBodyFont", "cartButtonFont")


def _keys(names: Iterable[str]) -> list[dict]:
    return [{"target": dict(_G), "key": k, "list": ""} for k in names]


def _site_fonts_key(ask_ok: bool = False) -> dict:
    out = {"keys": _keys(SITE_FONTS_VINTAGE), "alt": [], "optional": _keys(SITE_FONTS_VINTAGE_OPTIONAL),
           "site_fonts": {"value": None, "required": list(SITE_FONTS_VINTAGE),
                          "optional": list(SITE_FONTS_VINTAGE_OPTIONAL), "must": list(SITE_FONTS_VINTAGE)}}
    if ask_ok:
        out["ask_ok"] = True
    return out


def _both_or_all() -> dict:
    tp = {"kind": "global", "panel": "TypographyColorsSection"}
    return {"keys": [{"target": tp, "key": "bodyFont", "list": ""}, {"target": tp, "key": "headingFont", "list": ""}],
            "alt": [_keys(SITE_FONTS_VINTAGE)], "optional": _keys(SITE_FONTS_VINTAGE_OPTIONAL)}


# vague-ask (2026-10-15, the same decisions: "vague requests - ask"): a single-turn request that names a field but
# gives no value a write could use (a text, a colour, a font, a link the merchant did not give) may be answered with a
# question (`ask_ok`; a write still passes — the exam cannot tell an offered draft from a guess). Before, asking there
# was `no_write` (a fail) and inventing a copyright line passed; the r6 training rows ask on such turns
# (contrast.py's vague family, the follow-up family's first turns), so the exam must not count that against them.
VAGUE_ASK_OK = (("paraphrase", "p031"),   # change the copyright line at the bottom
                ("paraphrase", "p048"),   # change the description google shows for my site
                ("paraphrase", "p109"),   # change the button color
                ("paraphrase", "p129"),   # update our instagram and tiktok links
                ("holdout", "h024"),      # hover color for the category links
                ("holdout", "h027"),      # change the description of the vaporizers category
                ("holdout", "h033"),      # change the checkout button font in the side cart
                ("holdout", "h036"),      # set the default page title for search results
                ("holdout", "h037"),      # edit robots.txt
                ("holdout", "h044"),      # change the subtitle under the hero headline
                ("holdout", "h048"),      # change the instagram feed title color
                ("holdout", "h055"))      # change the forgot password text

# (set, id) → the expected members that replace the built row's (keys, alt, optional, ask_ok, site_fonts)
AMENDMENTS: dict[tuple[str, str], dict] = {**{k: {"ask_ok": True} for k in VAGUE_ASK_OK}, **{
    ("paraphrase", "p011"): _site_fonts_key(),            # change the font everywhere to Lora
    ("paraphrase", "p013"): _site_fonts_key(ask_ok=True),  # i don't like the font
    ("paraphrase", "p150"): _site_fonts_key(),            # i don't like the font | okay, lora
    ("followup", "p150"): _site_fonts_key(),
    ("real", "r02"): _site_fonts_key(),                   # … | i don't like the font, change it to something else
    ("real", "r04"): _site_fonts_key(),                   # show me the available fonts | okay, lora
    ("holdout", "h059"): _both_or_all(),                  # change the font | Inter for both
}}


def amend(row: dict, set_name: str | None = None) -> dict:
    """The row under today's answer key (AMENDMENTS); unchanged when it has none."""
    s = str(row.get("set") or set_name or "")
    a = AMENDMENTS.get((s, str(row.get("id"))))
    if a is None:
        return row
    r = copy.deepcopy(row)
    e = r.setdefault("expected", {})
    meta = r.setdefault("meta", {})
    meta["superseded"] = {"id": row.get("id"), "keys": copy.deepcopy(e.get("keys") or []),
                          "alt": copy.deepcopy(e.get("alt") or []), "fields": copy.deepcopy(row.get("fields"))}
    meta["answer_key"] = ANSWER_KEY
    for k in ("optional", "ask_ok", "site_fonts"):
        e.pop(k, None)
    e.update(copy.deepcopy(a))
    if "keys" in a:
        r["fields"] = [x["key"] or x["list"] for x in e.get("keys") or []]
    r["id"] = f"{row.get('id')}{AMENDED_ID_SUFFIX}"
    return r


def load_set(name_or_path: str | Path, *, exam_dir: Path = EXAM_DIR, manifest: Path | None = None,
             with_selected: bool = False, amended: bool = True) -> list[dict]:
    """The rows of a set, contexts inflated, under today's answer key (AMENDMENTS; `amended=False`: as built).
    With `manifest`, every row's `manifest_hash` must be that file's (the manifest is recorded by hash)."""
    path = set_path(name_or_path, exam_dir)
    if not path.is_file():
        raise ExamError(f"{path} is missing (make -C lab exam-build WHY=… builds the sets)")
    contexts = load_contexts(path.parent if path.parent.joinpath(CONTEXTS_FILE.name).is_file() else exam_dir)
    rows = [inflate(r, contexts, with_selected=with_selected) for r in read_jsonl(path)]
    if amended:
        rows = [amend(r, path.stem) for r in rows]
    if manifest is not None:
        have = manifest_hash_of(manifest)
        wrong = sorted({str(r.get("manifest_hash")) for r in rows if r.get("manifest_hash") and r["manifest_hash"] != have})
        if wrong:
            raise ExamError(f"{manifest} hashes to {have[:12]}… but {path.name} was written for "
                            f"{', '.join(h[:12] + '…' for h in wrong)}: pass the set's own manifest")
    return rows


def expected_sets(case: dict) -> list[list[dict]]:
    exp = case.get("expected") or {}
    keys = list(exp.get("keys") or [])
    return ([keys] if keys else []) + [list(s) for s in exp.get("alt") or [] if s]


def optional_keys(case: dict) -> list[dict]:
    """expected.optional: (target, key, list) a right answer MAY also write (not stray, never required) — the site
    fonts that inherit the new font anyway (AMENDMENTS)."""
    return [e for e in (case.get("expected") or {}).get("optional") or [] if isinstance(e, dict)]


# a sign-off question ("Anything else?", "Is there anything else I can help with?") asks nothing about the request
SIGN_OFF_Q = re.compile(r"\b(anything else|what else|something else i can|else i can (?:help|do)|more i can (?:help|do))\b",
                        re.IGNORECASE)


def asks_a_question(text: Any) -> bool:
    """The reply asks the merchant something about the request: a sentence ending in "?" that is not a sign-off
    ("Done — anything else?"). `ask` mode and the exam's ask_ok rows."""
    qs = re.findall(r"[^.!?\n]*\?", str(text or ""))
    return any(q.strip(" ?") and not SIGN_OFF_Q.search(q) for q in qs)


def is_out_of_scope(case: dict) -> bool:
    return case.get("kind") == "out_of_scope" or bool((case.get("expected") or {}).get("out_of_scope"))


def is_labelled(case: dict) -> bool:
    return is_out_of_scope(case) or bool(expected_sets(case))


# --- matching targets and keys ---------------------------------------------------------------------------------------------

def norm_list(name: Any) -> str:
    return str(name or "").replace("[]", "")


def target_matches(actual: dict | None, want: dict, page: str | None = None) -> bool:
    """Does an op's (or the block's) target name the expected one? A section
    target without `page` is on the viewed page (the host fills it); a global
    target without `panel` is the merged global settings, which hold every
    panel's keys."""
    a = actual or {}
    if a.get("kind") != want.get("kind"):
        return False
    kind = want.get("kind")
    if kind == "section":
        ap = a.get("page") or page
        return a.get("sectionId") == want.get("sectionId") and (not want.get("page") or not ap or ap == want.get("page"))
    if kind == "page":
        return (a.get("page") or page) == want.get("page")
    if kind == "global":
        return not want.get("panel") or not a.get("panel") or a.get("panel") == want.get("panel")
    return True


# --- retrieval (offline) ---------------------------------------------------------------------------------------------------

def _block_lists_key(targets: list[dict], e: dict) -> bool:
    want = e["target"]
    for tg in targets:
        if not any(target_matches(t, want) and (want.get("kind") != "global" or not want.get("panel") or t.get("panel") == want.get("panel"))
                   for t in tg.get("targets") or []):
            continue
        if not e.get("key"):
            if norm_list(e.get("list")) in {norm_list(x) for x in tg.get("lists") or []}:
                return True
            continue
        for f in tg.get("fields") or []:
            if f.get("key") == e["key"] and norm_list(f.get("list")) == norm_list(e.get("list")):
                return True
    return False


def _block_has_target(targets: list[dict], e: dict) -> bool:
    want = e["target"]
    return any(any(target_matches(t, want) and (want.get("kind") != "global" or not want.get("panel") or t.get("panel") == want.get("panel"))
                   for t in tg.get("targets") or []) for tg in targets)


def retrieval_text(case: dict, with_history: bool) -> str:
    """The text the retrieval reads: the request, or — the what-if of
    `--with-history` — the previous user text and reply before it."""
    text = str(case.get("request") or "")
    hist = case.get("history") or []
    if with_history and hist:
        h = hist[-1]
        text = f"{h.get('user') or ''}\n{h.get('assistant') or ''}\n{text}"
    return text


def run_cli(exe: Path, manifest: Path, turns: list[tuple]) -> dict[str, dict]:
    """`sous-chef-fields -jsonl` over (id, context, text[, history]) turns →
    id → {block, targets}. `history` ([{user, assistant}], oldest first) is
    what the service reads a follow-up with (fields.CandidatesAfter)."""
    import subprocess

    from .render import go_trim_space

    lines = []
    for cid, ctx, text, *rest in turns:
        row: dict = {"id": cid, "text": go_trim_space(text)}
        if ctx is not None:
            row["context"] = ctx
        if rest and rest[0]:
            row["history"] = [{"user": str(h.get("user") or ""), "assistant": str(h.get("assistant") or "")} for h in rest[0]]
        lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    p = subprocess.run([str(exe), "-manifest", str(manifest), "-jsonl"], input="\n".join(lines) + "\n",
                       capture_output=True, text=True, encoding="utf-8", check=False)
    if p.returncode != 0:
        raise ExamError(f"sous-chef-fields failed: {p.stderr.strip()[:500]}")
    out = {}
    for line in p.stdout.splitlines():
        if line.strip():
            o = json.loads(line)
            out[str(o.get("id"))] = {"block": o.get("block") or "", "targets": o.get("targets") or []}
    if len(out) != len(turns):
        raise ExamError(f"sous-chef-fields answered {len(out)} of {len(turns)} turns")
    return out


def score_retrieval(case: dict, res: dict) -> dict:
    """One row's retrieval record."""
    targets, block = res["targets"], res["block"]
    rec = {"id": case["id"], "class": (case.get("meta") or {}).get("class"), "request": case.get("request"),
           "chars": len(block), "targets": len(targets), "fields": sum(len(t.get("fields") or []) for t in targets),
           "block_targets": [(t.get("targets") or [{}])[0] for t in targets]}
    if is_out_of_scope(case):
        rec.update(out_of_scope=True, nonempty=bool(block))
        return rec
    best: tuple[list[dict], list[dict]] | None = None
    for s in expected_sets(case):
        miss = [e for e in s if not _block_lists_key(targets, e)]
        if best is None or len(miss) < len(best[1]):
            best = (s, miss)
    if best is None:
        rec.update(unlabelled=True)
        return rec
    s, miss = best
    hit = not miss
    tmiss = [e for e in miss if not _block_has_target(targets, e)]
    rec.update(hit=hit, top1=hit and all(_block_lists_key(targets[:1], e) for e in s), need=len(s), got=len(s) - len(miss),
               target_missed=bool(miss) and bool(tmiss), key_missed=bool(miss) and not tmiss,
               missing=[f"{json.dumps(e['target'], sort_keys=True, separators=(',', ':'))} {e.get('list') or ''} {e.get('key') or ''}".strip()
                        for e in miss])
    return rec


def _pct(a: int, b: int) -> float | None:
    return round(100.0 * a / b, 1) if b else None


def _q(values: list[int], q: float) -> int | None:
    v = sorted(values)
    return v[min(len(v) - 1, int(q * (len(v) - 1)))] if v else None


def retrieval_group(recs: list[dict]) -> dict:
    ins = [r for r in recs if not r.get("out_of_scope") and not r.get("unlabelled")]
    oos = [r for r in recs if r.get("out_of_scope")]
    out: dict[str, Any] = {"n": len(ins)}
    if ins:
        hit = sum(1 for r in ins if r["hit"])
        out.update(hit=hit, recall=_pct(hit, len(ins)), target_missed=sum(1 for r in ins if r["target_missed"]),
                   key_missed=sum(1 for r in ins if r["key_missed"]), empty=sum(1 for r in ins if not r["chars"]),
                   top1=_pct(sum(1 for r in ins if r["top1"]), len(ins)),
                   chars_p50=_q([r["chars"] for r in ins], 0.5), chars_p95=_q([r["chars"] for r in ins], 0.95))
    if oos:
        out["out_of_scope"] = {"n": len(oos), "nonempty_block": sum(1 for r in oos if r["nonempty"]),
                               "fields_p50": _q([r["fields"] for r in oos], 0.5)}
    return out


def retrieval(cases: list[dict], manifest: Path, exe: Path, *, with_history: bool = False, follow_ups: bool = False) -> dict:
    """recall@block for the rows (no model). `follow_ups` hands each row's
    history to the service's own follow-up rule (fields.CandidatesAfter: a
    reply of three or fewer content words that names nothing is read with
    the turn before it); `with_history` is the what-if that always reads it."""
    res = run_cli(exe, manifest, [(c["id"], c.get("context"), retrieval_text(c, with_history),
                                   (c.get("history") or []) if follow_ups else None) for c in cases])
    recs = [score_retrieval(c, res[c["id"]]) for c in cases]
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_class[str(r.get("class"))].append(r)
    order = sorted(by_class, key=lambda k: (CLASS_ORDER.index(k) if k in CLASS_ORDER else 99, k))
    return {"overall": retrieval_group(recs), "by_class": {k: retrieval_group(by_class[k]) for k in order}, "results": recs}


def format_retrieval(name: str, rep: dict) -> str:
    def line(label: str, g: dict) -> str | None:
        if not g.get("n"):
            if g.get("out_of_scope"):
                o = g["out_of_scope"]
                return f"  {label:<13} {o['n']:>4}  out of scope: {o['nonempty_block']} got a non-empty block (fields p50 {o['fields_p50']})"
            return None
        return (f"  {label:<13} {g['n']:>4} {g['hit']:>5} {g['recall']:>7.1f}% {g['target_missed']:>7} {g['key_missed']:>7} "
                f"{g['empty']:>6} {g['top1']:>5.0f}%   {g['chars_p50']}/{g['chars_p95']}")
    lines = [f"{name}: recall@block (candidate-fields block lists every expected key under the right target)",
             f"  {'class':<13} {'n':>4} {'hit':>5} {'recall':>8} {'tgtMiss':>7} {'keyMiss':>7} {'empty':>6} {'top1':>6}   chars p50/p95"]
    for k, g in rep["by_class"].items():
        s = line(k, g)
        if s:
            lines.append(s)
    s = line("ALL", rep["overall"])
    if s:
        lines.append(s)
    o = rep["overall"].get("out_of_scope")
    if o and rep["overall"].get("n"):
        lines.append(f"  out of scope: {o['nonempty_block']}/{o['n']} got a non-empty block")
    return "\n".join(lines)


# --- end-to-end scoring ----------------------------------------------------------------------------------------------------

def _flat_keys(values: Any, prefix: str = "") -> list[str]:
    out = []
    for k, v in (values or {}).items() if isinstance(values, dict) else []:
        key = f"{prefix}{k}"
        out.append(key)
        if isinstance(v, dict):
            out += _flat_keys(v, key + ".")
    return out


def touches_of(op: dict) -> list[dict]:
    """What one browser op writes, as (target, list, key) records."""
    name = op.get("op")
    target = op.get("target") if isinstance(op.get("target"), dict) else {}
    if name == "set_fields":
        return [{"target": target, "list": "", "key": k} for k in _flat_keys(op.get("values"))]
    if name in ("reset", "restore_default"):
        return [{"target": target, "list": "", "key": str(k)} for k in op.get("keys") or []]
    if name == "edit_list":
        lst = str(op.get("list") or "")
        keys = _flat_keys(op.get("item")) if op.get("action") in ("update", "add") else []
        return [{"target": target, "list": lst, "key": k} for k in keys] or [{"target": target, "list": lst, "key": ""}]
    if name == "set_page_layout":
        return [{"target": {"kind": "page", "page": op.get("page")}, "list": "", "key": "__layout"}]
    if name == "reset_pinned_section":
        return [{"target": {"kind": "page", "page": op.get("page")}, "list": "", "key": "__pinned"}]
    return [{"target": target, "list": "", "key": f"__{name}"}]


def touch_matches(t: dict, e: dict, page: str | None) -> bool:
    if not target_matches(t["target"], e["target"], page):
        return False
    ek, el = str(e.get("key") or ""), norm_list(e.get("list"))
    tk, tl = str(t.get("key") or ""), norm_list(t.get("list"))
    if not ek:
        return bool(el) and tl == el
    if el:  # a list row's field — or the hero's flat mirror of slides[0]
        return tk == ek and tl in (el, "")
    return tk == ek and not tl


def accepted_ops(row: dict) -> tuple[list[dict], list[dict]]:
    """(accepted, refused) browser ops of a row: row.ops paired with the
    engine's results batch by batch (an engine that failed outright accepted
    nothing)."""
    results: list[dict] = []
    for b in row.get("engine_results") or []:
        rs = list(b.get("results") or [])
        results += rs
    ops = list(row.get("ops") or [])
    accepted, refused = [], []
    for i, op in enumerate(ops):
        r = results[i] if i < len(results) else None
        (accepted if r is not None and r.get("ok") else refused).append(op)
    return accepted, refused


def reply_text(row: dict) -> str:
    """The merchant-facing words of a turn: the reply plus report_unsupported's reason and hint."""
    parts = [str(row.get("reply") or "")]
    for c in row.get("tool_calls") or []:
        if c.get("name") == "report_unsupported" and isinstance(c.get("input"), dict):
            parts += [str(c["input"].get("reason") or ""), str(c["input"].get("manual_hint") or "")]
    return "\n".join(p for p in parts if p)


def names_a_place(text: str, extra: Iterable[str] = ()) -> list[str]:
    low = text.lower()
    hits = [t for t in PLACE_TERMS if _WORD[t].search(low)]
    hits += [w for w in extra if w and w.lower() in low and w not in hits]
    return hits


def score_turn(case: dict, row: dict) -> dict:
    """The exam verdict of one finished turn (collect's verifier contract:
    `{verdict, reasons, …}`; PASS verdicts are `pass` and `unsupported_ok`)."""
    accepted, refused = accepted_ops(row)
    page = (case.get("context") or {}).get("page")
    touches = []
    for i, op in enumerate(accepted):
        for t in touches_of(op):  # root = the op + the value's top-level key: a nested value is one write
            t["root"] = (i, t["list"], t["key"].split(".", 1)[0] if t["key"] else "")
            touches.append(t)
    written = [f"{json.dumps(t['target'], sort_keys=True, separators=(',', ':'))} {t['list']} {t['key']}".replace("  ", " ")
               for t in touches]
    if is_out_of_scope(case):
        oos = (case.get("expected") or {}).get("out_of_scope") or {}
        where = oos.get("where") or []
        expected_where = [w for w in oos.get("expected_where") or [] if w in WHERE_VALUES]
        sent, raw = where_sent(row)
        reported = any(c.get("name") == "report_unsupported" for c in row.get("tool_calls") or [])
        # The keyword rule on the reply's words, for EVERY revision (the secondary, symmetric score: the
        # primary one scores a linkable `where` exactly and a reply without one by keywords, so a baseline
        # and a candidate are not measured the same way; examcmp prints both).
        kw_places = names_a_place(reply_text(row), where)
        common = {"where_sent": sent, "expected_where": expected_where, "reported": reported,
                  "keyword_verdict": "wrote" if touches else ("unsupported_ok" if kw_places else "no_pointer")}
        if raw and raw != sent:
            common["where_raw"] = raw
        if touches:
            return {"verdict": "wrote", "reasons": ["wrote_out_of_scope"], "written": written, **common,
                    "scored_by": "where" if sent and expected_where else "keywords"}
        # A `where` the browser can link is the pointer (contract 1): it passes when it is one the row expects.
        # Without one (a service before the enum, or a value outside it) the reply's words are scored as before.
        if sent and expected_where:
            if any(w in expected_where for w in sent):
                return {"verdict": "unsupported_ok", "reasons": [], "places": [f"where:{w}" for w in sent],
                        "scored_by": "where", **common}
            return {"verdict": "wrong_where", "reasons": [f"where:{','.join(sent)} not {','.join(expected_where)}"],
                    "scored_by": "where", **common}
        places = kw_places
        if not places:
            return {"verdict": "no_pointer", "reasons": ["named_no_place"], "places": [], "scored_by": "keywords", **common}
        return {"verdict": "unsupported_ok", "reasons": [], "places": places, "scored_by": "keywords", **common}
    sets = expected_sets(case)
    if not sets:
        return {"verdict": None, "reasons": ["unlabelled"], "written": written}
    if not touches:
        if not refused and (case.get("expected") or {}).get("ask_ok") and asks_a_question(row.get("reply")) \
                and not any(c.get("name") == "report_unsupported" for c in row.get("tool_calls") or []):
            # the request left the value open and the turn asked for it (AMENDMENTS: p013)
            return {"verdict": "clarify_ok", "reasons": [], "asked": True, "refused": 0}
        return {"verdict": "invalid_op" if refused else "no_write", "reasons": ["refused_ops"] if refused else ["asked_or_answered"],
                "refused": len(refused)}
    allowed = [e for s in sets for e in s] + optional_keys(case)
    # A write is stray when nothing at or under its root matched (`categories: {pre-rolls: {name}}` writes
    # `categories`, `categories.pre-rolls` and `categories.pre-rolls.name`; the leaf's match covers the rest).
    ok_roots = {t["root"] for t in touches if any(touch_matches(t, e, page) for e in allowed)}
    stray, seen = [], set()
    for t, w in zip(touches, written):
        if t["root"] not in ok_roots and t["root"] not in seen:
            seen.add(t["root"])
            stray.append(w)
    best = None
    for s in sets:
        got = [e for e in s if any(touch_matches(t, e, page) for t in touches)]
        if best is None or len(got) / len(s) > len(best[1]) / len(best[0]):
            best = (s, got)
    s, got = best
    missing = [f"{e['target'].get('sectionId') or e['target'].get('panel') or e['target'].get('page') or e['target'].get('kind')} "
               f"{e.get('list') or ''} {e.get('key') or ''}".replace("  ", " ").strip() for e in s if e not in got]
    if len(got) == len(s):
        verdict = "stray_writes" if stray else "pass"
    else:
        verdict = "partial" if got else "wrong_target"
    reasons = (["missing:" + m for m in missing]) + (["stray:" + w for w in stray])
    return {"verdict": verdict, "reasons": reasons, "written": written, "stray": stray, "refused": len(refused),
            "keys_hit": len(got) == len(s)}


# --- end-to-end run --------------------------------------------------------------------------------------------------------

class StateTap:
    """The engine callable collect.run_case drives, remembering the state of
    its last call so the next turn of a replayed conversation starts there."""

    def __init__(self, engine: Callable[[dict, list[dict], dict], dict]):
        self.engine = engine
        self.fingerprint = getattr(engine, "fingerprint", None)
        self.state: dict | None = None

    def __call__(self, state: dict, ops: list[dict], ctx: dict) -> dict:
        out = self.engine(state, ops, ctx)
        if isinstance(out.get("state"), dict):
            self.state = out["state"]
        return out


def vintage_state(manifest_path: Path, home_context: dict | None) -> dict:
    """The EditorState the engine runs on: a basic `vintage` state whose home
    rows are the context's sections (state.build_state on the exam's manifest)."""
    from .manifest import Manifest
    from .state import build_state

    m = Manifest(json.loads(Path(manifest_path).read_text(encoding="utf-8")), Path(manifest_path))
    rows = [{"id": s["id"], "enabled": s.get("enabled") is not False}
            for s in (home_context or {}).get("sections") or [] if isinstance(s, dict) and s.get("id")]
    return build_state(m, template=TEMPLATE, home_sections=rows or None)


def history_for(case: dict, mode: str) -> list[dict]:
    hist = list(case.get("history") or [])
    if mode == "none":
        return []
    if mode == "last":
        return hist[-1:]
    return hist


def run_row(case: dict, client: Any, engine: Callable, *, state: dict, route: str | None, sample: int = 1,
            assets: Iterable[dict] = (), fields: Any | None = None, history_mode: str = "live",
            site_prefix: str = "lab-") -> dict:
    """One exam row end to end: its history turns (unscored), then the scored turn."""
    from . import collect as C

    tap = StateTap(engine)
    state = copy.deepcopy(state)
    replayed = []
    hist = history_for(case, history_mode)
    for i, h in enumerate(hist):
        hc = dict(case, request=str(h.get("user") or ""), site_state=state, expected=None,
                  context=h.get("context") if isinstance(h.get("context"), dict) else case.get("context"))
        hr = C.run_case(hc, client, tap, route=route, sample=sample, assets=assets, verifier=None, fields=fields,
                        fresh_conversation=(i == 0), site_prefix=site_prefix)
        if tap.state is not None:
            state = tap.state
        replayed.append({"user": h.get("user"), "recorded_reply": h.get("assistant"), "reply": hr.get("reply"),
                         "status": hr.get("status"), "code": hr.get("code"), "ops": hr.get("ops"),
                         "escalated": hr.get("escalated"), "routes": hr.get("routes"), "latency_ms": hr.get("latency_ms"),
                         "usage": hr.get("usage")})
        if hr.get("status") != "ok":
            break
    final_case = dict(case, site_state=state)
    if replayed and replayed[-1]["status"] != "ok":
        row = {"id": f"{case['id']}#{sample}", "case_id": case["id"], "sample": sample, "model": route or "",
               "status": "error", "code": "history_failed", "error": f"history turn {len(replayed)} failed: {replayed[-1]['code']}",
               "verdict": None, "reasons": ["history_failed"], "latency_ms": sum(r["latency_ms"] or 0 for r in replayed),
               "ops": [], "meta": C.case_meta(case)}
    else:
        row = C.run_case(final_case, client, tap, route=route, sample=sample, assets=assets, verifier=None,
                         fields=fields, fresh_conversation=not replayed, site_prefix=site_prefix)
        if row.get("status") == "ok":  # the whole score (collect's verifier hook keeps verdict + reasons only)
            row.update(score_turn(final_case, row))
    row["history_mode"] = history_mode
    row["history_turns"] = replayed
    row["set"] = case.get("set")
    meta = row.setdefault("meta", {})
    meta["class"] = (case.get("meta") or {}).get("class")
    meta["section"] = meta.get("section") or ((case.get("target") or {}).get("panel") or (case.get("target") or {}).get("sectionId"))
    usage = dict(row.get("usage") or {})
    for r in replayed:  # the whole conversation's spend
        for k, v in (r.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
    if replayed:
        row["usage"] = usage
    return row


def summarize_run(rows: list[dict], report: dict) -> dict:
    from . import eval as E

    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[str((r.get("meta") or {}).get("class"))].append(r)
    order = sorted(by_class, key=lambda k: (CLASS_ORDER.index(k) if k in CLASS_ORDER else 99, k))
    report["summary"]["by_class"] = {k: E.group(by_class[k]) for k in order}
    ins = [r for r in rows if r.get("verdict") not in (None, *OOS_VERDICTS)]
    report["summary"]["keys_hit"] = {"n": len(ins), "hit": sum(1 for r in ins if r.get("keys_hit")),
                                     "rate": _pct(sum(1 for r in ins if r.get("keys_hit")), len(ins))}
    report["summary"]["escalated"] = sum(1 for r in rows if r.get("escalated"))
    return report


def run_set(cases: list[dict], client: Any, engine: Callable, *, label: str, set_name: str, route: str | None,
            manifest: Path, samples: int = 1, assets: Iterable[dict] = (), fields: Any | None = None,
            fields_record: dict | None = None, history_mode: str = "live", out_dir: Path = RUNS_DIR,
            progress: Callable[[str], None] | None = None, site_prefix: str = "lab-",
            rows_range: tuple[int, int, int] | None = None) -> tuple[Path, dict]:
    """Run `cases` (already sliced when `rows_range` = (offset, end, set size)
    is given: recorded in the report and in the file name, so parallel
    chunks of one set never overwrite each other)."""
    from . import collect as C
    from . import eval as E

    if history_mode not in HISTORY_MODES:
        raise ExamError(f"--history must be one of {HISTORY_MODES}")
    started = C.utc_now()
    home = next((c.get("context") for c in cases if (c.get("context") or {}).get("page") == "home"), None)
    state = vintage_state(manifest, home)
    assets = list(assets)
    rows = []
    for k in range(1, samples + 1):
        for i, case in enumerate(cases, 1):
            row = run_row(case, client, engine, state=state, route=route, sample=k, assets=assets, fields=fields,
                          history_mode=history_mode, site_prefix=site_prefix)
            rows.append(row)
            if progress:
                esc = f" · escalated→{row['escalated'].get('to')}" if row.get("escalated") else ""
                hist = f" · +{len(row['history_turns'])} history" if row.get("history_turns") else ""
                progress(f"[{k}/{samples} {i}/{len(cases)}] {row['id']} {row.get('verdict') or row.get('code')}{hist}{esc} "
                         f"· {(row.get('latency_ms') or 0) / 1000:.1f}s")
    report = E.build_report(rows, label=label, route=route, manifest_hash=manifest_hash_of(manifest), started=started,
                            rule_version=f"{GENERATOR}-score", samples=samples, cases=len(cases),
                            engine_fingerprint=C.engine_fingerprint_of(engine), fields_context=fields is not None,
                            fields_record=fields_record)
    report["set"] = set_name
    report["history_mode"] = history_mode
    report["site_prefix"] = site_prefix
    if rows_range is not None:
        report["rows_range"] = {"offset": rows_range[0], "end": rows_range[1], "of": rows_range[2]}
    summarize_run(rows, report)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    part = f"-r{rows_range[0]}-{rows_range[1]}" if rows_range is not None else ""
    path = unused_path(out_dir, f"exam-{E.slug(set_name)}-{E.slug(label)}{part}-{utc_stamp()}")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    os.link(tmp, path)  # fails if another run took the name meanwhile: never overwrite a report
    tmp.unlink()
    return path, report


def unused_path(out_dir: Path, stem: str) -> Path:
    """`<stem>.json`, or `<stem>-2.json`, `-3` … when a report of the same set,
    label and range finished in the same second: a run never overwrites
    another's report."""
    path = Path(out_dir) / f"{stem}.json"
    n = 2
    while path.exists():
        path = Path(out_dir) / f"{stem}-{n}.json"
        n += 1
    return path


def format_run(report: dict) -> str:
    s = report["summary"]
    o = s["overall"]
    lines = [f"{report.get('set')} · {report.get('label')} · route {report.get('route') or '(default)'} · "
             f"{report.get('cases')} rows × {report.get('samples')} · history {report.get('history_mode')} · "
             f"fields_context {'on' if report.get('fields_context') else 'off'}",
             f"  overall  {o['passed']}/{o['scored']} pass ({o['pass_rate']} pts) · errors {o['errors']} · "
             f"keys hit {s['keys_hit']['hit']}/{s['keys_hit']['n']} · escalated {s['escalated']} · "
             f"p50 {o['latency_p50_ms']} ms"]
    for k, g in s.get("by_class", {}).items():
        lines.append(f"  {k:<10} {g['passed']}/{g['scored']} ({g['pass_rate']})")
    lines.append("  verdicts " + ", ".join(f"{k} {v}" for k, v in o["verdicts"].items()))
    return "\n".join(lines)


# --- the oracle (dry runs against the fake service) -------------------------------------------------------------------------

def _value_for(manifest, e: dict, context: dict) -> Any:
    from .state import resolve_key

    t = e["target"]
    containers = []
    if t["kind"] == "section":
        containers = [manifest.section(str(t.get("sectionId")))]
    elif t["kind"] == "global":
        containers = [manifest.panel(t["panel"])] if t.get("panel") else manifest.containers("global")
    elif t["kind"] == "branding":
        containers = [manifest.branding]
    elif t["kind"] == "page":
        containers = [manifest.page(str(t.get("page")))]
    field = None
    key = str(e.get("key") or "")
    if t["kind"] == "page" and t.get("page") == "seo":  # the engine's page "seo" = the SEO details panel
        seo = manifest.detail("SeoPanel")
        field = seo.fields.get(key) if seo else None
    elif t["kind"] == "global" and key.startswith("categories.") and key.count(".") >= 2:  # a category's patch
        detail = manifest.detail("CategoryDetailsPanel")
        leaf = key.split(".", 2)[2]
        field = (detail.fields.get(leaf) or detail.fields.get(f"item.{leaf}")) if detail else None
    for c in containers if field is None else []:
        if c is None:
            continue
        lst = c.lists.get(str(e.get("list") or "")) if e.get("list") else None
        field = (lst.item_fields.get(key) if lst else None) or resolve_key(manifest, c, key)
        if field:
            break
    typ = field.type if field else "text"
    if typ == "color":
        return "#336699"
    if typ == "bool":
        return True
    if typ in ("number", "range"):
        lo = field.min if field and field.min is not None else 1
        hi = field.max if field and field.max is not None else lo
        return hi if isinstance(hi, (int, float)) else lo
    if typ in ("enum", "select"):
        opts = [v for v in (field.enum if field else []) if v not in (None, "")] or list(field.suggested if field else [])
        return opts[-1] if opts else "x"
    if typ == "image":
        return "https://cdn.int.breadstack.io/gal/exam.jpg"
    if typ == "ref":
        cats = context.get("categories") or []
        return cats[0].get("slug") if cats else "x"
    if typ == "richtext":
        return "<p>Exam</p>"
    return "Exam"


def oracle_ops(case: dict, manifest) -> list[dict]:
    """The tool calls that write the row's expected keys (first set), one
    set_fields per target, one edit_list per list row."""
    sets = expected_sets(case)
    if not sets:
        return []
    ctx = case.get("context") or {}
    calls: list[dict] = []
    flat: dict[str, dict] = {}
    for e in sets[0]:
        t = e["target"]
        if not e.get("list"):
            key = json.dumps(t, sort_keys=True)
            if key not in flat:
                flat[key] = {"name": "set_fields", "input": {"target": t, "values": {}}}
                calls.append(flat[key])
            flat[key]["input"]["values"][e["key"]] = _value_for(manifest, e, ctx)
        elif e.get("key"):
            idx: Any = [0, 0] if "[]" in e["list"] else 0
            calls.append({"name": "edit_list", "input": {"target": t, "list": e["list"], "action": "update", "index": idx,
                                                         "item": {e["key"]: _value_for(manifest, e, ctx)}}})
        elif e["list"] == "sections":  # a page's sections: touch a row (an add needs a section type)
            calls.append({"name": "edit_list", "input": {"target": t, "list": "sections", "action": "update", "index": 0,
                                                         "item": {"enabled": True}}})
        else:
            inp: dict = {"target": t, "list": e["list"], "action": "add", "item": {}}
            if "[]" in e["list"]:
                inp["index"] = 0
            calls.append({"name": "edit_list", "input": inp})
    return calls


ORACLE_MODES = ("where", "legacy")


def oracle_script(case: dict, manifest, mode: str = "where") -> list:
    """A fake-service script that answers the row as the oracle would. An
    out-of-scope row: `where` (default) = a report_unsupported call carrying
    the row's first expected `where` and a reply that names no place (so the
    pass can only come from `where`); `legacy` = today's reply naming the
    place in words and no `where` (a service before the enum — the keyword
    rule scores it)."""
    if mode not in ORACLE_MODES:
        raise ExamError(f"oracle mode must be one of {ORACLE_MODES}")
    if is_out_of_scope(case):
        oos = (case.get("expected") or {}).get("out_of_scope") or {}
        done = ("turn.done", {"stop": "end_turn", "usage": {"input_tokens": 0, "output_tokens": 0, "calls": 1}})
        expected_where = oos.get("expected_where") or []
        if mode == "where" and expected_where:
            call = ("tool.call", {"id": "u0", "name": "report_unsupported",
                                  "input": {"reason": "That isn't something I can change in the site editor.",
                                            "where": expected_where[0]}})
            return [[call, ("text.delta", {"text": "I can't change that here."}), done]]
        where = (oos.get("where") or ["your Breadstack dashboard"])[0]
        return [[("text.delta", {"text": f"That isn't something I can change in the site editor; it's done in {where} "
                                         f"(Breadstack dashboard)."}), done]]
    calls = oracle_ops(case, manifest)
    if not calls:
        return [[("text.delta", {"text": "Done."}), ("turn.done", {"stop": "end_turn"})]]
    tc = [("tool.call", {"id": f"o{i}", "name": c["name"], "input": c["input"]}) for i, c in enumerate(calls)]
    return [tc + [("turn.waiting", {"toolCallIds": [f"o{i}" for i in range(len(calls))]})],
            [("text.delta", {"text": "Done."}), ("turn.done", {"stop": "end_turn", "usage": {"input_tokens": 0, "output_tokens": 0,
                                                                                                "calls": 1}})]]


def fake_service(manifest_path: Path, cases: list[dict], *, oracle: bool, key: str, oracle_mode: str = "where"):
    """The lab's fake service (tests/fake_service.py), in process, the real
    manifest registered; with `oracle` every row's text is scripted with its
    expected writes (history turns get the fake's default "Done.")."""
    if str(LAB_DIR) not in sys.path:
        sys.path.insert(0, str(LAB_DIR))
    from tests import fake_service as FS  # the tests' fake: a dry run's stand-in, never imported by a real run
    from .manifest import Manifest

    doc = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    svc = FS.FakeService(manifest=doc, key=key)
    svc.routes = list(svc.routes or []) + [
        {"name": "fake-fields", "kind": "messages-compatible", "fields_context": True, "fallback": None}]
    if oracle:
        m = Manifest(doc)
        for c in cases:
            svc.scripts[str(c.get("request") or "")] = oracle_script(c, m, oracle_mode)
    return svc


# --- CLI ---------------------------------------------------------------------------------------------------------------------

def _set_names(arg: str) -> list[str]:
    return list(SETS) if arg == "all" else [a for a in arg.split(",") if a]


def cmd_sets(args) -> int:
    idx_path = Path(args.exam_dir) / INDEX_FILE.name
    if not idx_path.is_file():
        print(f"{idx_path} is missing (make -C lab exam-build WHY=…)", file=sys.stderr)
        return 1
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    m = idx["manifest"]
    print(f"manifest {m['file']} hash {m['manifest_hash'][:16]}… (build {m.get('generated_from', {}).get('build_id')}); "
          f"generator {idx['generator']}")
    for name, s in idx["sets"].items():
        print(f"  {name:<22} {s['rows']:>4} rows · {s['with_history']} with history · "
              + ", ".join(f"{k} {v}" for k, v in s["by_class"].items()))
    return 0


def cmd_build(args) -> int:
    try:
        idx = build_sets(Path(args.why), Path(args.exam_dir))
    except ExamError as e:
        print(f"exam build: {e}", file=sys.stderr)
        return 1
    for name, s in idx["sets"].items():
        print(f"{name}: {s['rows']} rows ({s['with_history']} with history)")
    print(f"wrote {args.exam_dir} (manifest {idx['manifest']['manifest_hash'][:16]}…)")
    return 0


def cmd_retrieval(args) -> int:
    from . import fields as F

    try:
        exe = Path(args.bin) if args.bin else F.binary()
    except F.FieldsError as e:
        print(f"exam retrieval: {e}", file=sys.stderr)
        return 1
    names = _set_names(args.set) if not str(args.set).endswith(".jsonl") else [args.set]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    for name in names:
        try:
            cases = load_set(name, exam_dir=Path(args.exam_dir), manifest=Path(args.manifest), with_selected=args.with_selected)
        except ExamError as e:
            print(f"exam retrieval: {e}", file=sys.stderr)
            return 1
        rep = retrieval(cases, Path(args.manifest), exe, with_history=args.with_history, follow_ups=args.follow_ups)
        label = Path(name).stem
        report = {"set": label, "cases": len(cases), "manifest_hash": manifest_hash_of(Path(args.manifest)),
                  "fields": {"bin": str(exe), "source_sha256": F.source_sha() if not args.bin else None},
                  "options": {"with_history": args.with_history, "with_selected": args.with_selected, "follow_ups": args.follow_ups},
                  "at": utc_now(), **rep}
        path = out_dir / f"exam-retrieval-{label}{'-hist' if args.with_history else ''}{'-fu' if args.follow_ups else ''}{'-sel' if args.with_selected else ''}-{stamp}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(format_retrieval(label, rep))
        print(f"saved {path}")
    return 0


def cmd_run(args) -> int:
    from . import collect as C

    names = _set_names(args.set) if not str(args.set).endswith(".jsonl") else [args.set]
    if len(names) != 1:
        print("exam run: one set per run (SET=paraphrase, holdout, real, …)", file=sys.stderr)
        return 2
    try:
        cases = load_set(names[0], exam_dir=Path(args.exam_dir), manifest=Path(args.manifest), with_selected=args.with_selected)
    except ExamError as e:
        print(f"exam run: {e}", file=sys.stderr)
        return 1
    rebase_home = None
    if args.manifest_rebase:
        # The drift check's arm D (docs/DRIFT.md): the same rows on another manifest. Keys renamed by the static
        # diff follow; rows whose keys are gone are left out (stale); contexts are re-derived on that manifest.
        from .drift import home_context, static_diff
        from .examrebase import rebase_exam
        old_doc = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        new_path = Path(args.manifest_rebase)
        new_doc = json.loads(new_path.read_text(encoding="utf-8"))
        rebase_home = home_context(cases) or home_context(load_set("paraphrase", exam_dir=Path(args.exam_dir),
                                                                    manifest=Path(args.manifest)))
        sd = static_diff(old_doc, new_doc)
        cases, stale, rep = rebase_exam(cases, new_doc, sd["renames"], old_manifest=old_doc)
        print(f"rebased onto {new_path.name} ({rep['to'][:12]}…): {rep['kept']} rows kept, {rep['stale']} stale (not asked"
              f"{': ' + ', '.join(rep['stale_ids'][:5]) if rep['stale_ids'] else ''}), {rep['renamed_rows']} with renamed keys")
        args.manifest = str(new_path)
        if not cases:
            print("exam run: every row is stale on that manifest", file=sys.stderr)
            return 1
    if args.only:
        keep = {x.strip() for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in keep or (c.get("meta") or {}).get("class") in keep]
    total = len(cases)
    if args.offset < 0 or (args.max_cases is not None and args.max_cases < 1):
        print("exam run: --offset must be ≥ 0 and --max-cases ≥ 1", file=sys.stderr)
        return 2
    rows_range = None
    if args.offset or args.max_cases:
        end = total if args.max_cases is None else min(total, args.offset + args.max_cases)
        cases = cases[args.offset:end]
        rows_range = (args.offset, end, total)
        if not cases:
            print(f"exam run: --offset {args.offset} is past the {total} rows", file=sys.stderr)
            return 2
        print(f"rows {args.offset}..{end - 1} of {total}")
    if not args.site_prefix.startswith("lab-"):
        print("exam run: --site-prefix must start with lab- (the lab's own site ids)", file=sys.stderr)
        return 2
    if args.rederive_contexts and rebase_home is not None:
        from .drift import rederive
        n = rederive(cases, Path(args.manifest), rebase_home)
        print(f"contexts re-derived on the rebased manifest for {n} turns of {len(cases)} rows")
    elif args.rederive_contexts:
        n = rederive_contexts(cases, Path(args.manifest))
        print(f"contexts re-derived with today's browser mirror (state.context_of: focus, shopMenu, outline) "
              f"for {n} turns of {len(cases)} rows")
    unlabelled = [c["id"] for c in cases if not is_labelled(c)]
    if unlabelled:
        print(f"note: {len(unlabelled)} row(s) carry no expected keys and are run unscored ({', '.join(unlabelled[:5])}…)")
    svc = None
    url, key = args.url, args.key
    if args.fake:
        svc = fake_service(Path(args.manifest), cases, oracle=args.oracle, key="exam-fake", oracle_mode=args.oracle_mode).start()
        url, key = svc.url, "exam-fake"
        print(f"fake service on {url} ({'oracle scripts' if args.oracle else 'default script'}) — no model is asked")
    elif not key:
        print("a service key is required (--key or $SOUS_CHEF_KEY)", file=sys.stderr)
        return 2
    try:
        try:
            client, _ = C.connect(url, key, Path(args.manifest), cases, sse=args.sse, timeout_s=args.timeout, build_id="lab-exam")
            if args.persistent_engine:
                from .engine import PersistentEngine
                engine = C.DictEngine(PersistentEngine(args.engine, manifest_path=Path(args.manifest), size=1))
            else:
                engine = C.default_engine(args.engine, Path(args.manifest))
            fields, record = C.resolve_fields(args.fields_context, Path(args.manifest), client, args.route)
        except (C.ServiceError, OSError, json.JSONDecodeError, RuntimeError) as e:
            print(f"cannot start: {e}", file=sys.stderr)
            return 1
        print(f"fields_context {'on' if fields is not None else 'off'} for {args.route or 'the default route'}")
        label = args.label or args.route or "default"
        try:
            path, report = run_set(cases, client, engine, label=label, set_name=Path(names[0]).stem, route=args.route,
                                   manifest=Path(args.manifest), samples=args.samples, assets=C.load_assets(args.assets),
                                   fields=fields, fields_record=record, history_mode=args.history, out_dir=Path(args.out_dir),
                                   progress=print, site_prefix=args.site_prefix, rows_range=rows_range)
        except KeyboardInterrupt:
            print("stopped (Ctrl-C): nothing saved", file=sys.stderr)
            return C.INTERRUPTED
        except (C.FieldsContextMismatch, ExamError) as e:
            print(f"stopped: {e}\nnothing saved", file=sys.stderr)
            return 1
    finally:
        if svc is not None:
            svc.stop()
    print(format_run(report))
    print(f"saved {path}")
    return 0


def rederive_contexts(cases: list[dict], manifest_path: Path) -> int:
    """r4 PLAN L7: every row's context (and each history turn's) re-derived
    from the exam's own vintage state with today's browser mirror
    (state.context_of: `focus`, `shopMenu`, `outline`, the pages the text
    names) instead of the recorded 10-06 browser's. A row with
    `meta.selected_section_id` is asked with that section's editor open (the
    browser now sends it as `focus`). The state is the one run_set builds
    (vintage_state from the rows' home context), so the contexts describe
    exactly what the engine edits. Returns the number of contexts replaced."""
    from .manifest import Manifest
    from .state import context_of, load_context_fixture

    home = next((c.get("context") for c in cases if (c.get("context") or {}).get("page") == "home"), None)
    if home is None:
        home = next((c.get("context") for c in load_set("paraphrase", manifest=manifest_path)
                     if (c.get("context") or {}).get("page") == "home"), None)
    state = vintage_state(manifest_path, home)
    m = Manifest(json.loads(Path(manifest_path).read_text(encoding="utf-8")), Path(manifest_path))
    extras = load_context_fixture()
    n = 0
    for c in cases:
        sel = (c.get("meta") or {}).get("selected_section_id")
        view = {"tab": "editor", "activeSectionId": sel} if sel else None
        page = (c.get("context") or {}).get("page") or (c.get("meta") or {}).get("page") or "home"
        c["context"] = context_of(state, page, extras, m, view=view, text=str(c.get("request") or ""))
        n += 1
        for h in c.get("history") or []:
            hp = (h.get("context") or {}).get("page") or page
            h["context"] = context_of(state, hp, extras, m, view=view, text=str(h.get("user") or ""))
            n += 1
        c.setdefault("meta", {})["context_source"] = "rederived"
    return n


def cmd_replay(args) -> int:
    conv = json.loads(Path(args.conversation).read_text(encoding="utf-8"))
    if not isinstance(conv, dict) or not conv.get("messages"):
        print(f"exam replay: {args.conversation} is not a GET /v1/conversations body with messages "
              f"({str(conv)[:120]})", file=sys.stderr)
        return 1
    labels: list[dict] = []
    for lp in args.labels or []:
        try:
            labels += load_set(lp, exam_dir=Path(args.exam_dir))
        except ExamError as e:
            print(f"exam replay: {e}", file=sys.stderr)
            return 1
    rows = replay_rows(conv, labels, transcript=not args.no_transcript)
    parity = None
    if args.check_blocks:
        from . import fields as F
        try:
            exe = Path(args.bin) if args.bin else F.binary()
            parity = served_block_parity(rows, Path(args.manifest), exe)
        except (F.FieldsError, ExamError) as e:
            print(f"exam replay: --check-blocks: {e}", file=sys.stderr)
            return 1
    out = Path(args.out) if args.out else LAB_DIR / "data" / f"replay-{str(conv.get('conversationId') or 'conv')[:8]}.jsonl"
    write_jsonl(out, rows)
    labelled = sum(1 for r in rows if is_labelled(r))
    print(f"{len(rows)} turns → {out} ({labelled} labelled; history up to {max((len(r['history']) for r in rows), default=0)} turns)")
    if parity is not None:
        print(f"served candidate-fields blocks reproduced by today's retrieval: {parity['same']}/{parity['checked']}"
              + (f" (differ: {', '.join(parity['differ'])})" if parity["differ"] else "")
              + (f"; {parity['other_manifest']} turn(s) on another manifest not checked" if parity["other_manifest"] else ""))
    for r in rows:
        rec = r["meta"]["recorded"]
        t = rec.get("turn") or {}
        print(f"  {r['id']} [{r['meta']['page']}] {r['request'][:60]!r} · {t.get('status')} "
              f"{'→'.join(((t.get('usage') or {}).get('routes') or [t.get('model') or '?']))} · "
              f"{len(rec['calls'])} call(s){' · ' + r['meta']['labelled_by'] if r['meta'].get('labelled_by') else ''}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="souschef-lab exam", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exam-dir", default=str(EXAM_DIR), help="the sets' folder (default: %(default)s)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sets", help="list the sets and their sizes")
    b = sub.add_parser("build", help="(re)build the sets from the paraphrase study's folder")
    b.add_argument("--why", required=True, help="the study folder (paraphrase-cases.jsonl, …, manifest-real.json, conv-*.json)")
    r = sub.add_parser("retrieval", help="offline recall@block of the candidate-fields block (no model; needs go)")
    r.add_argument("--set", default="all", help="a set, a comma list, 'all' or a .jsonl path (default: all)")
    r.add_argument("--manifest", default=str(MANIFEST_FILE))
    r.add_argument("--bin", help="a built sous-chef-fields (default: build it, fields.py; $SOUS_CHEF_FIELDS_BIN)")
    r.add_argument("--with-history", action="store_true", help="what-if: read the previous user text + reply with the request")
    r.add_argument("--follow-ups", action="store_true",
                   help="send each row's history to the service's follow-up rule (fields.CandidatesAfter), as the loop does once it passes the previous turn")
    r.add_argument("--with-selected", action="store_true", help="what-if: send the selected section (focus.sectionId, and the older selectedSectionId) in the context")
    r.add_argument("--out-dir", default=str(RUNS_DIR))
    e = sub.add_parser("run", help="end-to-end through a service URL + route (spends model calls on a real service)")
    e.add_argument("--set", required=True)
    e.add_argument("--url", default=os.environ.get("SOUS_CHEF_URL", "http://127.0.0.1:8090"))
    e.add_argument("--key", default=os.environ.get("SOUS_CHEF_KEY", ""))
    e.add_argument("--route", "--model", dest="route")
    e.add_argument("--label")
    e.add_argument("--samples", type=int, default=1)
    e.add_argument("--history", choices=HISTORY_MODES, default="live",
                   help="live = replay every history turn first (default), last = the previous one, none = single turn")
    e.add_argument("--manifest", default=str(MANIFEST_FILE))
    e.add_argument("--assets", type=Path, default=LAB_DIR / "fixtures" / "assets.json")
    e.add_argument("--engine", type=Path, default=os.environ.get("ENGINE"))
    e.add_argument("--fields-context", choices=("auto", "on", "off"), default="auto")
    e.add_argument("--with-selected", action="store_true", help="send the selected section (focus.sectionId, selectedSectionId) in the context")
    e.add_argument("--only", help="comma-separated row ids or classes")
    e.add_argument("--offset", type=int, default=0, help="skip the first N rows of the set (after --only)")
    e.add_argument("--max-cases", "--limit", dest="max_cases", type=int, help="run at most N rows (from --offset)")
    e.add_argument("--site-prefix", default="lab-",
                   help="siteId prefix (default lab-; must start with lab-): separate conversations for two runs of the "
                        "same rows on one database, e.g. lab-base- / lab-cand-")
    e.add_argument("--oracle-mode", choices=ORACLE_MODES, default="where",
                   help="with --oracle: out-of-scope rows answered with report_unsupported `where` (default) or, "
                        "legacy, in words only (the keyword rule)")
    e.add_argument("--sse", action="store_true")
    e.add_argument("--timeout", type=float, default=300.0)
    e.add_argument("--out-dir", default=str(RUNS_DIR))
    e.add_argument("--rederive-contexts", action="store_true",
                   help="r4 L7: re-derive every row's context with today's browser mirror (focus, shopMenu, outline; "
                        "the selected section open) instead of the recorded 10-06 contexts")
    e.add_argument("--manifest-rebase", metavar="MANIFEST",
                   help="drift arm D: ask the set's rows on this OTHER manifest (renamed keys followed, rows whose keys "
                        "are gone left out; with --rederive-contexts the contexts are derived on it)")
    e.add_argument("--persistent-engine", action="store_true",
                   help="boot the engine once (js/engine_server.mjs) instead of once per call (~10 s each)")
    e.add_argument("--fake", action="store_true", help="dry run: the lab's fake service in process (no model)")
    e.add_argument("--oracle", action="store_true", help="with --fake: script every row with its expected writes")
    rp = sub.add_parser("replay", help="exam rows from a GET /v1/conversations/{siteId} body")
    rp.add_argument("--conversation", required=True)
    rp.add_argument("--labels", action="append", help="a set (or .jsonl) whose expected keys label matching turns; repeatable")
    rp.add_argument("--out", help="output .jsonl (default: data/replay-<conversation>.jsonl)")
    rp.add_argument("--no-transcript", action="store_true", help="leave out the raw stored messages before each turn")
    rp.add_argument("--check-blocks", action="store_true",
                    help="recompute each turn's candidate-fields block (needs go) and count how many equal the served one")
    rp.add_argument("--manifest", default=str(MANIFEST_FILE), help="the manifest the turns ran on (for --check-blocks)")
    rp.add_argument("--bin", help="a built sous-chef-fields (for --check-blocks)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run" and args.oracle and not args.fake:
        print("--oracle needs --fake", file=sys.stderr)
        return 2
    return {"sets": cmd_sets, "build": cmd_build, "retrieval": cmd_retrieval, "run": cmd_run, "replay": cmd_replay}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
