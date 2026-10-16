"""author — training cases for new or changed fields, written through the
Anthropic API (design §3.2 step 2).

r4's families were authored by Claude workflow agents; in GCP the same work
goes through the Messages API (the int project's ANTHROPIC_API_KEY, the prep
job only) under the same rules:

  targets    every field / list / container the deployed manifest has and the
             current model's training manifest does not (R3 added or renamed),
             plus every key whose pool cases the rebase dropped as no longer
             valid there (R2: an option removed, a gate moved …), skipped when
             the pool already holds >= COVERAGE_ENOUGH kept trainable cases that
             write it (accumulated across skipped rounds)
  per target 10–13 cases: 2 verbatim-ish, 2 colloquial (vocab.txt aliases), 1 short
             command, 1 relative value (keys), 1 focus (the section / panel open, no
             target named), 1 follow-up (ask, then the value), 1 reset (when the
             field has an unset / default value), 1 near-miss or out-of-editor; a
             GATED field (conditional_on / hidden_when) adds 2 `prereq` cases (the
             gate closed in the state: the oracle opens it in the same set_fields,
             gate first) and 1 `gate_only` case (the merchant asks for the layout
             that reveals it); a new container adds 6 container-level cases (add the
             section, reorder, name it on another page). A field target short of
             MIN_POSITIVE kept positive writes after the re-ask gets one top-up ask.
  targets    op targets name a REAL section instance of the fixture home
             (contentBlock-1, not the type "contentBlock": r5 asked 16 drafts on a
             section id no state holds and kept 0 positives); the instance whose
             state already opens most of the field's gates is preferred, the other
             gates are opened in the case state (or, for prereq, by the oracle)
  holdout    targets with sha1(address) % 100 < 20 are family F9H: a new-key exam,
             never collected or trained (cases/exam-ext/<round>.jsonl)

One request per target: model claude-opus-5-5, effort high, structured output
(a JSON schema of the case batch), streaming, the server-side fallback
(`fallbacks: "default"`, beta server-side-fallback-2026-07-01), the fixed rules
block cached. The exam is NEVER shown to the author.

Validation (each rejection counted; a rejected case is re-asked once with its
reason, then dropped):
  keys     every write names a key, list and container of the deployed manifest
  oracle   the oracle ops apply on the case's built state through the engine with a
           non-empty diff (r4cases.remeasure: contexts derived, exact diffs measured)
  mode     a literal value the merchant typed → exact; a relative / chosen value →
           keys (r4cases.authored_values → to_keys)
  covers   a write case must write its target (a gate_only case: the gate)
  hidden   no write lands on a field the case's state hides after its op (the engine's
           `hidden` entries, changes.js gateMatches) unless the case means it
           (expected.expect_hidden): r5 kept 22 perks rows that taught hidden writes
  preview  no authored assistant text says "preview" (r4score's one rule)
  leak     leak.py rules 1-3 against the exam and the study's real conversations
  dedup    no duplicate of a pool case (r4cases.dedup_key)
A rejected draft keeps its full reason (the oracle's engine error text included) in
the report's per-target `reject_details` and in the re-ask.
"""
from __future__ import annotations

import copy
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from ..manifest import Manifest, holdout_bucket
from .common import StageError, redact
from .rebase import container_of, field_of, missing_reason

MODEL = "claude-opus-5-5"
EFFORT = "high"
MAX_TOKENS = 32000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
CONCURRENCY = 8
COVERAGE_ENOUGH = 6
HOLDOUT_PCT = 20
MAX_TARGETS = 400
# US$ per million tokens: input, output, cache write, cache read (examcmp.PRICES' claude-opus-5-5)
PRICE = (4.00, 20.00, 5.00, 0.20)
FIELD_STYLES = (("verbatim", 2), ("colloquial", 2), ("short", 1), ("relative", 1), ("focus", 1), ("followup", 1),
                ("reset", 1), ("near_miss", 1))
GATED_STYLES = (("prereq", 2), ("gate_only", 1))  # added for a field the manifest gates
CONTAINER_STYLES = (("container_add", 2), ("container_reorder", 2), ("container_other_page", 2))
SMOKE_STYLES = (("verbatim", 1), ("colloquial", 1), ("followup", 1), ("near_miss", 1))
STYLE_FAMILY = {"verbatim": "F0", "colloquial": "F1", "short": "F1", "relative": "F1", "reset": "F1", "prereq": "F1",
                "gate_only": "F1", "focus": "F4", "followup": "F2", "near_miss": "F3",
                "container_add": "F5", "container_reorder": "F5", "container_other_page": "F5"}
NON_POSITIVE_STYLES = ("near_miss", "gate_only")  # styles that do not write the target by design
MIN_POSITIVE = 8  # kept positive-write cases a field / list target aims for (a top-up ask when short)
MAX_ROUNDS = 3  # ask, re-ask the rejects (+ top-up), one more top-up
TOPUP_STYLES = ("colloquial", "verbatim", "short", "relative", "prereq", "focus")
MODES = ("exact", "keys", "where", "clarify", "answer")
EXPECTS = ("ask", "offer", "write", "answer", "report_unsupported")
TEMPLATES = ("vintage", "vintage", "bold", "boutique", "cozy", "editorial", "elegant", "playful", "pluto", "retro", "sweet")
VOCAB_FILE = "internal/fields/vocab.txt"

RULES = """You write test and training cases for Sous Chef, the AI assistant inside a cannabis-retail website editor \
(Breadstack's site builder). A merchant types a request in a chat panel beside the live editor; the assistant changes \
the site through browser ops (set_fields, edit_list, reset, restore_default, set_page_layout, pick_asset) or explains \
where to do it when the editor cannot (report_unsupported with a `where`).

You will get ONE manifest entry (a field, a list or a section type the editor gained or changed) and the styles to \
write. Write each case as a real merchant would type it into the chat: short, plain, sometimes lower case, sometimes \
with a typo or a British spelling (colour, centre, grey) — about one case in five. Merchants run cannabis shops: \
prices, pre-rolls, edibles, pickup and delivery, store hours, loyalty points. Never copy wording from the examples; \
they only show the shape.

Hard rules:
- Never use the word "preview" in any text, and never invent UI words the editor does not have.
- Name things the way a merchant would, not by the internal key. A label word is fine; a camelCase key never.
- The oracle ops must be exactly what a perfect assistant would send: valid targets, the manifest's keys, values of \
the field's type (enum values from its options, numbers within min/max, colours as #hex, rich text as <p>…</p>).
- score_mode: "exact" when the merchant states the value literally (the text contains it); "keys" when the value is \
relative or left to the assistant ("a bit bigger", "something warmer", "you pick"); "where" for a request the editor \
cannot do (oracle_ops empty, expected_where from: domain, sites, pages, assets, seo, themes, products, categories, \
brands, support); "clarify" when the request is genuinely ambiguous between two or more named settings (candidates = \
their labels, oracle_ops empty); "answer" for a question that needs no change (reply_any = words a right answer names).
- direction: for a relative number or colour change give {"key": <field key>, "dir": "up"|"down"|"darker"|"lighter"}, \
else {"key": "", "dir": ""}.
- history: earlier turns of the same conversation, oldest first; [] for a single turn. A follow-up case is a 2-turn \
conversation: turn 1 names the setting without a value (expect "ask", assistant asks one short question), the request \
is the merchant's answer (the value). Assistant texts in history are short and never mention a preview.
- view: the editor's focus when the merchant types. {"tab": "", "activeSectionId": "", "globalSection": ""} is the \
editor with nothing open. A focus case opens the target: a section's editor (tab "editor", activeSectionId = the \
section id) or a Global Settings panel (tab "settings", globalSection = the panel's section name given below), and \
the request names no target ("make this a bit bigger", "hide the button here").
- page: the storefront page the merchant is viewing (a page slug: home, products, product-details, faq, cart, blog, …).
- near_miss: a request close to the target that the editor CANNOT do (score_mode where), or an editable neighbour \
that is easy to confuse with it (score_mode exact, oracle_ops writing that neighbour).
- container_add / container_reorder / container_other_page (a new section type): add the section to a page, move it \
up or down among the page's sections, or change something in it while naming another page ("on the about page …").
- relative: a relative change of the field ("a bit bigger", "tighter", "warmer"), score_mode "keys" with direction.
- reset: the merchant wants the field back to how it was / cleared; the oracle is {"op": entry.reset.op, "target": …, \
"keys": [<key>]} (only when the entry has `reset`).
- Op targets: always entry.op_target exactly (a section's `sectionId` is the instance on the page, e.g. \
"contentBlock-1", never the bare section type).
- Hidden fields: when the entry has `visibility`, the field only shows in the editor while its gates hold. In every \
style except prereq and gate_only the case's site already shows it: write the field alone. prereq: the site does NOT \
show it yet (visibility.gates[-1] is closed); the merchant asks for the field as usual and the oracle opens the gate \
in the SAME set_fields, gate key first ({"imagePosition": "top", "stackedImageGap": 24}). gate_only: the merchant asks \
for the layout change that reveals it, in plain words ("put the pictures above the text"); the oracle sets only that \
gate. Never write any field the site hides (a neighbour gated by another layout) without opening its gate first.
- oracle_ops_json is a JSON array (as a string) of ops, each {"op": …, "target": …, …} exactly as the browser takes \
them: set_fields {target, values: {key: value}}; edit_list {target, list, action: add|update|remove|move, index, item}.
Return exactly the number of cases asked for, in the order of the styles listed."""

CASE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "cases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "style": {"type": "string"},
                    "request": {"type": "string"},
                    "history": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"user": {"type": "string"}, "assistant": {"type": "string"},
                                       "expect": {"type": "string", "enum": list(EXPECTS)}},
                        "required": ["user", "assistant", "expect"], "additionalProperties": False}},
                    "page": {"type": "string"},
                    "view": {"type": "object",
                             "properties": {"tab": {"type": "string"}, "activeSectionId": {"type": "string"},
                                            "globalSection": {"type": "string"}},
                             "required": ["tab", "activeSectionId", "globalSection"], "additionalProperties": False},
                    "oracle_ops_json": {"type": "string"},
                    "score_mode": {"type": "string", "enum": list(MODES)},
                    "expected_where": {"type": "array", "items": {"type": "string"}},
                    "candidates": {"type": "array", "items": {"type": "string"}},
                    "reply_any": {"type": "array", "items": {"type": "string"}},
                    "direction": {"type": "object",
                                  "properties": {"key": {"type": "string"}, "dir": {"type": "string"}},
                                  "required": ["key", "dir"], "additionalProperties": False},
                },
                "required": ["style", "request", "history", "page", "view", "oracle_ops_json", "score_mode",
                             "expected_where", "candidates", "reply_any", "direction"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["cases"],
    "additionalProperties": False,
}

GLOBAL_SECTION_OF_PANEL = {
    "BrandingSection": "branding", "HeaderSection": "headerMenu", "ShopMenuSection": "shopMenu",
    "FooterSection": "footerMenu", "TypographyColorsSection": "typography", "ButtonsSection": "buttons",
    "ProductCardSection": "productCard", "SocialMediaSection": "social", "LayoutSection": "layout",
    "SidebarCartSection": "sidebarCart", "NotificationSection": "notification", "InsiderBannerSection": "insiderBanner",
}


def h8(s: str) -> int:
    import hashlib
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


# --- addresses and targets --------------------------------------------------------------------------------------------

def case_addresses(c: dict, m: Manifest) -> set[str]:
    """What a case writes on manifest m, in field_index addressing: `<ctr>.<key>`,
    `<ctr>.<list>` and `<ctr>.<list>[].<key>`; a section add names `sections.<type>`."""
    from ..exam import touches_of
    out: set[str] = set()
    e = c.get("expected") or {}
    ops = list(e.get("ops") or []) + list((c.get("meta") or {}).get("oracle_ops") or [])
    for o in ops:
        if o.get("op") in ("pick_asset", None):
            continue
        if o.get("op") == "edit_list" and o.get("list") == "sections":
            item = o.get("item") if isinstance(o.get("item"), dict) else {}
            sec = m.section(str(item.get("type") or item.get("id") or ""))
            if sec is not None:
                out.add(f"sections.{sec.name}")
            continue
        for t in touches_of(o):
            lst = str(t.get("list") or "")
            key = str(t.get("key") or "")
            ctr, _ = container_of(m, t["target"], "" if lst else key)
            if lst:
                out.add(f"{ctr}.{lst}")
                if key:
                    out.add(f"{ctr}.{lst}[].{key}")
            elif key and not key.startswith("__"):
                out.add(f"{ctr}.{key}")
    for k in e.get("keys") or []:
        if isinstance(k, dict):
            lst = str(k.get("list") or "")
            key = str(k.get("key") or "")
            ctr, _ = container_of(m, k.get("target"), "" if lst else key)
            out.add(f"{ctr}.{lst}[].{key}" if lst and key else f"{ctr}.{lst}" if lst else f"{ctr}.{key}")
    return out


def coverage(cases: list[dict], m: Manifest, *, holdout: bool = False, exclude_hidden: bool = True) -> Counter:
    """Cases writing each address: the trainable ones, or (holdout=True) the
    new-key holdouts (F9H) — a holdout target is covered by earlier rounds'
    F9H cases, which a skipped round keeps in cases/exam-ext/. A case whose
    writes land on a hidden field it does not mean to write (hidden_writes) covers
    nothing: r5's 22 perks stackedImage* rows counted as coverage and kept the
    keys from being re-authored with their gate."""
    cnt: Counter = Counter()
    for c in cases:
        if bool(c.get("holdout")) != holdout or (holdout and (c.get("meta") or {}).get("family") != "F9H"):
            continue
        if exclude_hidden and hidden_writes(c, m):
            continue
        for a in case_addresses(c, m):
            cnt[a] += 1
    return cnt


def find_targets(m_train: dict, m_new: Manifest, *, pool: list[dict], dropped_writes: set[str] | None = None,
                 extra: set[str] | None = None, enough: int = COVERAGE_ENOUGH, max_targets: int = MAX_TARGETS) -> tuple[list[dict], dict]:
    """The targets to author for (module docstring) and a report."""
    from ..retrain_check import containers, field_index
    old_f, new_f = field_index(m_train), field_index(m_new.doc)
    old_c, new_c = containers(m_train), containers(m_new.doc)
    want: dict[str, str] = {}
    for a in sorted(set(new_f) - set(old_f)):
        want[a] = "added_field"
    for a in sorted(set(new_f) & set(old_f)):
        if new_f[a] != old_f[a]:
            want[a] = "retyped_field"
    for cname, c in new_c.items():
        oc = old_c.get(cname) or {}
        for lname in (c.get("lists") or {}):
            if lname not in (oc.get("lists") or {}):
                want[f"{cname}.{lname}"] = "added_list"
        if cname not in old_c:
            want[cname] = "added_container"
    for a in sorted(dropped_writes or ()):
        if a in new_f or a in new_c or any(a == f"{c}.{l}" for c, v in new_c.items() for l in (v.get("lists") or {})):
            want.setdefault(a, "cases_dropped")
    for a in sorted(extra or ()):
        if a in new_f or a in new_c:
            want.setdefault(a, "drift_new_key")
    cov, cov_hold = coverage(pool, m_new), coverage(pool, m_new, holdout=True)
    targets, skipped = [], {}
    for addr, why in sorted(want.items()):
        hold = holdout_bucket(addr) < HOLDOUT_PCT
        have = (cov_hold if hold else cov).get(addr, 0)
        if have >= enough:
            skipped[addr] = have
            continue
        kind = "container" if why == "added_container" else "list" if why == "added_list" else "field"
        targets.append({"address": addr, "kind": kind, "reason": why, "covered": have, "holdout": hold})
    # every wanted address (authored this round or already covered): the dataset stage's new-key gate reads it
    wanted_list = [{"address": a, "reason": why, "holdout": holdout_bucket(a) < HOLDOUT_PCT,
                    "kind": "container" if why == "added_container" else "list" if why == "added_list" else "field",
                    "covered": (cov_hold if holdout_bucket(a) < HOLDOUT_PCT else cov).get(a, 0)}
                   for a, why in sorted(want.items())]
    report = {"wanted": len(want), "covered_enough": len(skipped), "targets": len(targets),
              "truncated": max(0, len(targets) - max_targets), "by_reason": dict(Counter(t["reason"] for t in targets)),
              "holdout_targets": sum(1 for t in targets if t["holdout"]), "wanted_list": wanted_list}
    return targets[:max_targets], report


# --- the prompt -------------------------------------------------------------------------------------------------------

def load_vocab(service_dir: Path | None = None) -> list[tuple[str, list[str]]]:
    from ..fields import SERVICE_DIR
    p = Path(service_dir or SERVICE_DIR) / VOCAB_FILE
    out = []
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        left, right = (x.strip() for x in line.split("=", 1))
        out.append((left, [r.strip() for r in right.split(",") if r.strip()]))
    return out


def aliases_for(target: dict, spec: dict, vocab: list[tuple[str, list[str]]]) -> list[str]:
    """Merchant words vocab.txt maps onto the field's label / key words, and the container's aliases."""
    words = set(re.findall(r"[a-z]+", (str(spec.get("label") or "") + " " +
                                       re.sub(r"([a-z])([A-Z])", r"\1 \2", target["address"].split(".")[-1])).lower()))
    ctr = target["address"].split(".")
    ctr_ids = {f"s:{ctr[1]}" if ctr[0] == "sections" else f"g:{ctr[1]}" if ctr[0] == "globalPanels" else
               f"p:{ctr[1]}" if ctr[0] == "pages" else "b:branding"}
    out = []
    for left, right in vocab:
        if left.startswith("@"):
            if left[1:] in ctr_ids:
                out += [f"{r} (the {ctr[-2] if len(ctr) > 2 else ctr[0]})" for r in right[:3]]
            continue
        if any(set(re.findall(r"[a-z]+", r.lower())) <= words and r for r in right):
            out.append(left)
    return out[:12]


def target_entry(target: dict, m: Manifest) -> dict:
    """The manifest entry the author writes for, with the op target to use (a real
    section instance) and, for a gated field, how the editor makes it visible."""
    from .rebase import split_addr
    addr = target["address"]
    parts = addr.split(".")
    if target["kind"] == "container":
        grp, name = parts[0], parts[1]
        spec = (m.doc.get(grp) or {}).get(name) or {}
        return {"address": addr, "kind": "container", "group": grp, "name": name,
                "fields": {k: _brief(v) for k, v in list((spec.get("fields") or {}).items())[:40]},
                "lists": sorted((spec.get("lists") or {})), "pages": spec.get("pages") or [],
                "add_op": {"op": "edit_list", "target": {"kind": "page", "page": "home"}, "list": "sections",
                           "action": "add", "item": {"id": name}}}
    if target["kind"] == "field":
        ctr, lst, key = split_addr(addr)
    elif parts[0] == "branding":
        ctr, lst, key = "branding", ".".join(parts[1:]), ""
    else:
        ctr, lst, key = ".".join(parts[:2]), ".".join(parts[2:]), ""
    grp = ctr.split(".")[0]
    name = ctr.split(".")[1] if "." in ctr else "branding"
    cspec = (m.doc.get(grp) or {}).get(name) if grp != "branding" else m.doc.get("branding")
    cspec = cspec or {}
    op_target = op_target_for(grp, name, m, key=key, lst=lst)
    entry: dict[str, Any] = {"address": addr, "kind": target["kind"], "container": ctr, "op_target": op_target}
    if grp == "globalPanels":
        entry["global_section"] = GLOBAL_SECTION_OF_PANEL.get(name, name)
    if lst:
        lspec = (cspec.get("lists") or {}).get(lst) or {}
        entry["list"] = lst
        entry["list_ops"] = lspec.get("ops") or []
        entry["item_fields"] = {k: _brief(v) for k, v in list((lspec.get("item") or {}).items())[:30]}
    if key:
        spec = (cspec.get("fields") or {}).get(key) if not lst else ((cspec.get("lists") or {}).get(lst) or {}).get("item", {}).get(key)
        entry["key"] = key
        entry["field"] = _brief(spec or {}, full=True)
        siblings = [k for k in (cspec.get("fields") or {}) if k != key][:25]
        entry["neighbour_keys"] = {k: (cspec.get("fields") or {})[k].get("label") for k in siblings}
    container, field = _container_field(m, op_target, lst, key)
    if container is not None and field is not None:
        chain = gate_chain(m, container, field)
        if chain:
            entry["visibility"] = {
                "gates": chain,
                "note": "shown in the editor only while " + " and ".join(
                    f"{g['key']} ({g['label']}) is {json.dumps(g['open_value'])}" for g in chain)}
        if not lst and not field.read_only and not field.keyed_by:
            if field.has_unset:
                entry["reset"] = {"op": "reset", "to": field.unset_value}
            elif field.has_default:
                entry["reset"] = {"op": "restore_default", "to": field.default}
    return entry


_HOME_CACHE: dict[str, list[dict]] = {}


def _default_home(m: Manifest) -> list[dict]:
    """The fixture home's section rows (build_state, default template), once per manifest."""
    from ..state import build_state
    if m.hash not in _HOME_CACHE:
        try:
            _HOME_CACHE[m.hash] = build_state(m)["homeSections"]
        except Exception:  # noqa: BLE001 - a fixture problem: fall back to the ids alone
            from ..state import DEFAULT_HOME_SECTIONS
            _HOME_CACHE[m.hash] = copy.deepcopy(DEFAULT_HOME_SECTIONS)
    return _HOME_CACHE[m.hash]


def instance_for(m: Manifest, sec_type: str, *, key: str = "", lst: str = "") -> str | None:
    """A real section id of type `sec_type` on the fixture home (contentBlock-1, …):
    the instance whose state already opens most of the target field's gates (first
    one on a tie); None when the home has no section of that type."""
    rows = [r for r in _default_home(m) if m.type_of(str(r.get("id") or "")) == sec_type]
    if not rows:
        return None
    c = m.section(sec_type)
    f = _field_of(m, c, lst, key) if c is not None else None
    chain = gate_chain(m, c, f) if f is not None else []
    if not chain:
        return str(rows[0]["id"])
    state = {"homeSections": copy.deepcopy(_default_home(m))}
    best, best_n = str(rows[0]["id"]), -1
    for r in rows:
        n = sum(1 for g in chain if gate_open(g, _gate_value(m, c, g["key"], state, str(r["id"]), "home")))
        if n > best_n:
            best, best_n = str(r["id"]), n
    return best


def op_target_for(grp: str, name: str, m: Manifest | None = None, *, key: str = "", lst: str = "") -> dict:
    """The op target of a container. A section names a REAL instance of the fixture
    home (instance_for) when it has one; a type the home lacks keeps the type as its
    id and to_case adds that row to the case's home."""
    if grp == "sections":
        sid = instance_for(m, name, key=key, lst=lst) if m is not None else None
        if sid is None and m is None:
            from ..state import DEFAULT_HOME_SECTIONS, type_of_section
            sid = next((r["id"] for r in DEFAULT_HOME_SECTIONS if type_of_section(r["id"]) == name), None)
        return {"kind": "section", "page": "home", "sectionId": sid or name}
    if grp == "globalPanels":
        return {"kind": "global", "panel": name}
    if grp == "pages":
        return {"kind": "page", "page": name}
    return {"kind": "branding"}


def _brief(spec: dict, full: bool = False) -> dict:
    keep = ("type", "label", "help", "description", "placeholder", "enum", "options", "min", "max", "step", "default",
            "conditional_on", "hidden_when", "ref", "group")
    out = {k: spec.get(k) for k in keep if spec.get(k) not in (None, "", [], {})}
    if not full:
        out = {k: out[k] for k in ("type", "label", "enum", "options", "min", "max") if k in out}
    return out


# --- visibility (conditional_on / hidden_when) ------------------------------------------------------------------------

def _js_str(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def gate_matches(gate_value: Any, v: Any) -> bool:
    """changes.js gateMatches: "set" = any value but null / "" / false; a list = any member
    (string-equal); "unset" (the lab's own spelling) = not set."""
    if gate_value == "set":
        return v not in (None, "", False)
    if gate_value == "unset":
        return v in (None, "", False)
    vals = gate_value if isinstance(gate_value, list) else [gate_value]
    return any(g == v and type(g) is type(v) or (g is not None and v is not None and _js_str(g) == _js_str(v))
               for g in vals)


def hidden_by(rule: str, gate_value: Any, by_value: Any) -> bool:
    """Whether one gate rule hides its field now (changes.js hiddenNotes)."""
    m = gate_matches(gate_value, by_value)
    return m if rule == "hidden_when" else not m


def gate_open(g: dict, value: Any) -> bool:
    """A gate_chain entry holds for `value` (every rule on that key lets the field show)."""
    return not any(hidden_by(r["rule"], r["value"], value) for r in g["rules"])


def _field_of(m: Manifest, c, lst: str, key: str):
    if c is None or not key:
        if c is not None and lst and lst in c.lists:
            items = c.lists[lst].item_fields
            return next(iter(items.values()), None) if items else None
        return None
    if lst:
        spec = c.lists.get(lst)
        return spec.item_fields.get(key) if spec else None
    from ..state import resolve_key
    return resolve_key(m, c, key)


def _container_field(m: Manifest, op_target: dict, lst: str, key: str):
    from .rebase import container_of
    _, c = container_of(m, op_target, "" if lst else key)
    return c, _field_of(m, c, lst, key)


def gate_chain(m: Manifest, container, field) -> list[dict]:
    """What makes `field` visible, outermost gate first: [{key, label, rules: [{rule, value}],
    open_value}] (contentBlock stackedImageGap → style ∈ {image-text, img-left}, then
    imagePosition = top). A key gated by both rules (ShopMenu shopMenuActiveColor:
    conditional_on modern + hidden_when location) gets a value both rules allow."""
    from ..state import _gate_field, gate_writes
    out: list[dict] = []
    seen: set[str] = set()
    f = field
    for _ in range(4):
        if f is None or not f.gated:
            break
        writes = gate_writes(m, container, f)
        by_key: dict[str, list[dict]] = {}
        for rule in ("conditional_on", "hidden_when"):
            g = f.spec.get(rule) or {}
            if g.get("key") and g["key"] not in seen:
                by_key.setdefault(g["key"], []).append({"rule": rule, "value": g.get("value")})
        if not by_key:
            break
        level = []
        for k, rules in by_key.items():
            gf = _gate_field(m, container, k)
            seen.add(k)
            level.append({"key": k, "label": str((gf.label if gf else "") or k), "rules": rules,
                          "open_value": _opening_value(gf, rules, writes.get(k))})
        out = level + out
        f = _gate_field(m, container, level[0]["key"])
    return out


def _opening_value(gf, rules: list[dict], fallback: Any) -> Any:
    """A value of the gate field every rule lets show: the conditional_on values first,
    then the field's enum, then state.gate_writes' pick."""
    cands: list[Any] = []
    for r in rules:
        if r["rule"] == "conditional_on" and r["value"] not in ("set", "unset", None):
            cands += r["value"] if isinstance(r["value"], list) else [r["value"]]
    cands += list(gf.enum or []) if gf is not None else []
    cands.append(fallback)
    for v in cands:
        if v is not None and not any(hidden_by(r["rule"], r["value"], v) for r in rules):
            return v
    return fallback


def _gate_value(m: Manifest, container, key: str, state: dict, section_id: str | None, page: str,
                index: int | None = None, item: dict | None = None) -> Any:
    """A gate field's value in `state` (an edit_list item's own value first), its manifest
    default when unset (engine.js hiddenFor)."""
    from ..state import _gate_field, find_section, get_path, read_field
    g = _gate_field(m, container, key)
    if g is None:
        return None
    v = None
    if "[]" in key:
        item_key = key.rpartition("[].")[2]
        if isinstance(item, dict) and item_key in item:
            return item[item_key]
        lst_key = key.rpartition("[].")[0]
        spec = container.lists.get(lst_key)
        if spec is not None and container.kind == "section" and section_id:
            row = find_section(state, section_id, page) or {}
            arr = get_path(row, spec.path)
            i = index if isinstance(index, int) else 0
            if isinstance(arr, list) and 0 <= i < len(arr) and isinstance(arr[i], dict):
                v = arr[i].get(item_key)
    else:
        try:
            v = read_field(container, g, state, section_id=section_id, page=page)
        except (KeyError, ValueError, IndexError, TypeError):
            v = None
    if v is None and g.has_default:
        v = g.default
    return v


def hidden_writes(c: dict, m: Manifest) -> list[str]:
    """The fields a case's expected ops write while the editor hides them (static: the
    case's site_state, each op's own values applied first — engine.js evaluates the
    gate on the post-op state), minus the hidden writes the case means
    (expected.expect_hidden, a cases.py closed-gate variant). [] when it cannot tell.
    Checked against the engine's own `hidden` entries on r5's case file (2026-10-14: 81
    flagged + 400 sampled gated writes): 463 agree, 1 hidden write on a page the state
    does not store is missed (static says unknown there, never hidden)."""
    from ..exam import touches_of
    from .rebase import container_of
    meta = c.get("meta") or {}
    e = c.get("expected") or {}
    state = c.get("site_state")
    if not isinstance(state, dict) or meta.get("gate_closed"):
        return []
    intended = {str(k) for k in e.get("expect_hidden") or []}
    page0 = (c.get("context") or {}).get("page") or meta.get("page") or "home"
    ops = e.get("ops") or meta.get("oracle_ops") or []
    overlay: dict[str, dict] = {}
    out: list[str] = []
    from ..state import find_section
    for op in ops:
        # reset / restore_default: the engine raises no W_HIDDEN for clearing a hidden value
        if not isinstance(op, dict) or op.get("op") not in ("set_fields", "edit_list"):
            continue
        if op.get("op") == "edit_list" and op.get("list") == "sections":
            continue
        tgt = op.get("target") if isinstance(op.get("target"), dict) else {}
        if tgt.get("kind") == "section" and find_section(state, str(tgt.get("sectionId") or ""),
                                                         str(tgt.get("page") or page0)) is None:
            continue  # a page whose rows the state does not store (the engine materialises the template's): unknown
        tid = json.dumps(tgt, sort_keys=True)
        if op.get("op") == "set_fields" and isinstance(op.get("values"), dict):
            overlay.setdefault(tid, {}).update(op["values"])
        for t in touches_of(op):
            lst, key = str(t.get("list") or ""), str(t.get("key") or "")
            if not key or key.startswith("__") or key in intended:
                continue
            _, ctr = container_of(m, tgt, "" if lst else key)
            f = _field_of(m, ctr, lst, key.split(".")[0] if lst else key)
            if f is None or not f.gated:
                continue
            sid = str(tgt.get("sectionId") or "") or None
            page = str(tgt.get("page") or page0)
            item = op.get("item") if isinstance(op.get("item"), dict) else None
            idx = op.get("index") if isinstance(op.get("index"), int) else None
            for rule in ("conditional_on", "hidden_when"):
                g = f.spec.get(rule) or {}
                gk = g.get("key")
                if not gk:
                    continue
                if "[]" not in gk and gk in overlay.get(tid, {}):
                    by = overlay[tid][gk]
                else:
                    by = _gate_value(m, ctr, gk, state, sid, page, idx, item)
                if hidden_by(rule, g.get("value"), by):
                    out.append(_hidden_text(key, rule, gk, g.get("value"), by))
                    break
    return list(dict.fromkeys(out))


def _hidden_text(key: str, rule: str, gate_key: str, gate_value: Any, by: Any) -> str:
    want = (f"set" if gate_value == "set" else json.dumps(gate_value)) if rule == "conditional_on" else \
        f"not {json.dumps(gate_value)}"
    return f"{key} is hidden while {gate_key} is {json.dumps(by)} (shown only when {gate_key} is {want})"


def engine_hidden(r, intended: set[str] | frozenset = frozenset()) -> list[str] | None:
    """The hidden writes an engine run reports (results[].hidden, post-op), or None when the
    engine sent W_HIDDEN warnings without `hidden` entries (an engine before contract 4)."""
    out: list[str] = []
    seen_hidden = False
    for res in getattr(r, "results", None) or []:
        for h in res.get("hidden") or []:
            seen_hidden = True
            g = h.get("gate") or {}
            rule = h.get("rule") or "conditional_on"
            if str(h.get("key")) in intended:
                continue
            if hidden_by(rule, g.get("value"), h.get("byValue")):
                out.append(_hidden_text(str(h.get("key")), rule, str(g.get("key") or h.get("by")), g.get("value"),
                                        h.get("byValue")))
    if not seen_hidden and any(str(w).startswith("W_HIDDEN") for w in getattr(r, "warnings", None) or []):
        return None
    return list(dict.fromkeys(out))


def style_plan(target: dict, smoke: bool = False, entry: dict | None = None) -> list[str]:
    """The styles to ask for. With the entry: `reset` only for a resettable field, the
    gated styles only for a gated one (FIELD_STYLES / GATED_STYLES)."""
    if smoke:
        styles = SMOKE_STYLES
    elif target["kind"] == "container":
        styles = CONTAINER_STYLES
    else:
        styles = tuple(FIELD_STYLES)
        if entry is not None:
            if not entry.get("reset"):
                styles = tuple((s, n) for s, n in styles if s != "reset")
            if entry.get("visibility"):
                styles = styles + GATED_STYLES
    return [s for s, n in styles for _ in range(n)]


def topup_styles(n: int, entry: dict) -> list[str]:
    """n more positive-write styles (prereq only for a gated field)."""
    pool = [s for s in TOPUP_STYLES if s != "prereq" or entry.get("visibility")]
    return [pool[i % len(pool)] for i in range(max(0, n))]


def user_prompt(entry: dict, styles: list[str], aliases: list[str], examples: list[dict], *, retry: list[dict] | None = None) -> str:
    lines = ["The manifest entry (what the editor can change):", json.dumps(entry, ensure_ascii=False, indent=1), ""]
    if aliases:
        lines += ["Words merchants use for it (the service's vocabulary): " + ", ".join(aliases), ""]
    if examples:
        lines += ["Examples of the case shape from other fields (shape only — never reuse their wording):"]
        lines += [json.dumps(x, ensure_ascii=False) for x in examples]
        lines.append("")
    if retry:
        lines += ["These cases were rejected; write a replacement for each, same style, fixing the reason:"]
        lines += [json.dumps({"style": r["style"], "request": r.get("request"), "reason": r["reason"]}, ensure_ascii=False)
                  for r in retry]
        lines.append("")
    lines.append(f"Write {len(styles)} case(s), in this order of styles: {', '.join(styles)}.")
    return "\n".join(lines)


def style_examples(pool: list[dict], style: str, address: str, k: int = 3) -> list[dict]:
    fam = STYLE_FAMILY.get(style, "F1")
    cands = [c for c in pool if (c.get("meta") or {}).get("family") == fam and not c.get("holdout")]
    cands.sort(key=lambda c: h8(f"ex:{address}:{c.get('id')}"))
    out = []
    for c in cands[:k]:
        out.append({"style": style, "request": c.get("request"),
                    "history": [{"user": h.get("user"), "assistant": h.get("assistant"), "expect": h.get("expect")}
                                for h in c.get("history") or []],
                    "page": (c.get("meta") or {}).get("page"), "score_mode": (c.get("score") or {}).get("mode")})
    return out


# --- the API ----------------------------------------------------------------------------------------------------------

class AuthorRefused(StageError):
    pass


def anthropic_ask(*, model: str = MODEL, effort: str = EFFORT, max_tokens: int = MAX_TOKENS) -> Callable:
    """`ask(system, user, schema) -> (obj, usage)` over the official SDK (the [author] extra).
    The key is read by the SDK from ANTHROPIC_API_KEY; it is never logged."""
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover - the lab image installs .[author]
        raise StageError("the anthropic SDK is not installed (pip install '.[author]')", reason="config") from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise StageError("ANTHROPIC_API_KEY is empty (the prep job reads it from Secret Manager)", reason="config")
    client = anthropic.Anthropic(max_retries=6)

    def ask(system: str, user: str, schema: dict) -> tuple[dict, dict]:
        last: Exception | None = None
        for attempt in range(4):
            try:
                with client.beta.messages.stream(
                    model=model, max_tokens=max_tokens, betas=[FALLBACK_BETA],
                    system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": user}],
                    output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
                    extra_body={"fallbacks": "default"},
                ) as stream:
                    msg = stream.get_final_message()
                break
            except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
                last = e
                time.sleep(min(60, 5 * 2 ** attempt))
        else:
            raise StageError(f"Anthropic API: {redact(last)}", reason="anthropic")
        usage = _usage_of(msg)
        if getattr(msg, "stop_reason", None) == "refusal":
            raise AuthorRefused("the author declined this target", reason="refusal", details={"usage": usage})
        texts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        if not texts:
            raise StageError("the author returned no text", reason="anthropic", details={"usage": usage})
        return json.loads(texts[-1]), usage

    return ask


def _usage_of(msg: Any) -> dict:
    u = getattr(msg, "usage", None)
    out = Counter()
    iters = getattr(u, "iterations", None) if u is not None else None
    for src in (iters or [u]):
        if src is None:
            continue
        for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            v = getattr(src, k, None) if not isinstance(src, dict) else src.get(k)
            out[k] += int(v or 0)
    return dict(out)


def usage_cost(u: dict) -> float:
    pin, pout, pw, pr = PRICE
    return (pin * u.get("input_tokens", 0) + pout * u.get("output_tokens", 0)
            + pw * u.get("cache_creation_input_tokens", 0) + pr * u.get("cache_read_input_tokens", 0)) / 1e6


class FakeAsk:
    """A deterministic stand-in for the API (DRY runs, tests): writes each style
    from the manifest entry with a value the field accepts. No network."""

    def __init__(self, manifest: Manifest, *, break_first: int = 0):
        self.m = manifest
        self.calls = 0
        self.break_first = break_first  # the first N cases come back with a bad key (exercises the re-ask)
        self.lock = threading.Lock()

    def __call__(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        from ..cases import value_for
        entry = json.loads(user.split("\n", 1)[1].split("\n\n", 1)[0])
        styles = [s.strip() for s in user.rsplit("order of styles:", 1)[1].strip().rstrip(".").split(",")]
        rng = random.Random(h8(entry["address"]))
        out = []
        for i, st in enumerate(styles):
            with self.lock:
                self.calls += 1
                broken = self.calls <= self.break_first
            case = {"style": st, "history": [], "page": "home", "view": {"tab": "", "activeSectionId": "", "globalSection": ""},
                    "score_mode": "exact", "expected_where": [], "candidates": [], "reply_any": [],
                    "direction": {"key": "", "dir": ""}}
            if entry["kind"] == "container":
                op = copy.deepcopy(entry["add_op"])
                case.update(request=f"add a {entry['name']} section to the home page ({st.replace('_', ' ')} {i})",
                            oracle_ops_json=json.dumps([op]))
            elif st == "near_miss":
                case.update(request=f"can you connect my web address shopname{i}.ca to the site", oracle_ops_json="[]",
                            score_mode="where", expected_where=["domain"])
            else:
                f = field_of(self.m, entry["op_target"], entry.get("list") or "", entry.get("key") or "")
                key = entry.get("key") or next(iter(entry.get("item_fields") or {"title": {}}))
                value = value_for(f, rng) if f is not None else f"Fresh drop {i}"
                if broken:
                    key = key + "Nope"
                gates = (entry.get("visibility") or {}).get("gates") or []
                gate = gates[-1] if gates and "[]" not in gates[-1]["key"] else None
                if entry.get("list"):
                    op = {"op": "edit_list", "target": entry["op_target"], "list": entry["list"], "action": "update",
                          "index": [0, 0] if "[]" in entry["list"] else 0, "item": {key: value}}
                else:
                    op = {"op": "set_fields", "target": entry["op_target"], "values": {key: value}}
                ops = [op]
                label = str((entry.get("field") or {}).get("label") or key)
                req = f"set the {label.lower()} to {value}"
                if st in ("prereq", "gate_only") and gate is not None:
                    gate_vals = {gate["key"]: gate["open_value"]}
                    if st == "gate_only":
                        ops = [{"op": "set_fields", "target": entry["op_target"], "values": gate_vals}]
                        req = f"switch the {gate['label'].lower()} to {gate['open_value']} ({i})"
                    elif entry.get("list"):
                        ops = [{"op": "set_fields", "target": entry["op_target"], "values": gate_vals}, op]
                        req = f"i want the {label.lower()} to be {value} ({i})"
                    else:
                        ops = [{"op": "set_fields", "target": entry["op_target"], "values": {**gate_vals, key: value}}]
                        req = f"i want the {label.lower()} to be {value} ({i})"
                elif st == "relative":
                    case["score_mode"] = "keys"
                    if (entry.get("field") or {}).get("type") in ("range", "number"):
                        case["direction"] = {"key": key, "dir": "up"}
                    req = f"make the {label.lower()} a bit different, you pick ({i})"
                elif st == "reset" and entry.get("reset") and not entry.get("list"):
                    ops = [{"op": entry["reset"]["op"], "target": entry["op_target"], "keys": [key]}]
                    req = f"put the {label.lower()} back to how it was ({i})"
                if st == "followup":
                    case["history"] = [{"user": f"I want to change the {label.lower()}", "assistant": "Sure — what should it be?",
                                        "expect": "ask"}]
                    req = f"{value}"
                elif st == "focus":
                    if entry["op_target"]["kind"] == "section":
                        case["view"] = {"tab": "editor", "activeSectionId": entry["op_target"]["sectionId"], "globalSection": ""}
                    elif entry["op_target"]["kind"] == "global":
                        case["view"] = {"tab": "settings", "activeSectionId": "", "globalSection": entry.get("global_section") or ""}
                    req = f"make this one {value}"
                elif st == "colloquial":
                    req = f"pls change the {label.lower()} to {value} ({i})"
                elif st == "short":
                    req = f"{label.lower()}: {value}"
                if entry["op_target"]["kind"] == "page":
                    case["page"] = entry["op_target"]["page"]
                case.update(request=req, oracle_ops_json=json.dumps(ops))
            out.append(case)
        return {"cases": out}, {"input_tokens": 1000, "output_tokens": 200}


# --- one authored case → the r4 schema --------------------------------------------------------------------------------

def to_case(raw: dict, target: dict, entry: dict, m: Manifest, *, case_id: str, round_id: str, holdout: bool) -> dict:
    """The author's JSON in r4cases' merged schema (state built, not yet measured)."""
    from ..state import DEFAULT_HOME_SECTIONS, build_state
    style = str(raw.get("style") or "")
    tpl = TEMPLATES[h8(f"tpl:{case_id}") % len(TEMPLATES)]
    home = None
    sec_type = None
    ot = entry.get("op_target") or {}
    if ot.get("kind") == "section":
        sec_type = ot.get("sectionId")
    if sec_type and not any(r["id"] == sec_type or r["id"].startswith(sec_type + "-") for r in DEFAULT_HOME_SECTIONS):
        home = copy.deepcopy(DEFAULT_HOME_SECTIONS) + [{"id": sec_type, "enabled": True}]
    state = build_state(m, template=tpl, home_sections=home)
    vis_meta = prepare_state(state, entry, m, style=style, case_id=case_id)
    try:
        ops = json.loads(raw.get("oracle_ops_json") or "[]")
    except json.JSONDecodeError:
        ops = None
    view = raw.get("view") or {}
    view = {k: v for k, v in view.items() if v} or None
    mode = str(raw.get("score_mode") or "exact")
    hist = []
    for h in raw.get("history") or []:
        hist.append({"user": str(h.get("user") or ""), "context": None, "page": raw.get("page") or "home",
                     "expect": h.get("expect") or "ask", "assistant": str(h.get("assistant") or ""), "ops": [], "calls": []})
    where = [w for w in raw.get("expected_where") or [] if w]
    fam = "F9H" if holdout else "F7"
    d = raw.get("direction") or {}
    exp = {"ops": ops if isinstance(ops, list) else [], "diff": [], "keys": [], "alt": [], "alt_exact": [],
           "out_of_scope": ({"reason": "out_of_editor", "where": where, "expected_where": where} if mode == "where" else None),
           "candidates": list(raw.get("candidates") or []) or None,
           "direction": ({"key": d.get("key"), "dir": d.get("dir"), "from": None} if d.get("key") and d.get("dir") else None),
           "reply_any": list(raw.get("reply_any") or []) or None, "expect_hidden": None, "gate": None}
    kind = {"where": "unsupported", "clarify": "clarify", "answer": "answer"}.get(mode) or ("followup" if hist else "authored")
    key = entry.get("key") or ""
    c = {"id": case_id, "kind": kind, "target": copy.deepcopy(ot) if ot else None, "fields": [key] if key else [],
         "site_state": state, "context": {"page": raw.get("page") or "home"}, "request": str(raw.get("request") or ""),
         "history": hist, "expected": exp, "score": {"mode": mode if mode in MODES else "exact"}, "holdout": holdout,
         "manifest_hash": m.hash, "engine_fingerprint": None, "template": tpl,
         "meta": {"family": fam, "style": style, "page": raw.get("page") or "home", "view": view,
                  "oracle_ops": copy.deepcopy(exp["ops"]), "group": f"author:{target['address']}",
                  "target_address": target["address"], "target_reason": target.get("reason"),
                  "author": {"model": MODEL, "round": round_id}, "candidates": list(raw.get("candidates") or []) or None,
                  "holdout_key": target["address"]}}
    if vis_meta:
        c["meta"]["state_prep"] = vis_meta
    return c


def prepare_state(state: dict, entry: dict, m: Manifest, *, style: str, case_id: str) -> dict:
    """Makes the case's state fit its style, in place; returns what it did (meta.state_prep).

      gated field   every gate of entry.visibility opened (outermost first), except for
                    prereq / gate_only, where the innermost gate is left (or made) closed:
                    the oracle opens it
      list target   a list the instance does not hold yet (gallery imagesLink: the
                    engine answers E_INDEX "list does not exist" to an update) seeded with
                    one blank per row of the container's longest sibling list (3 if none)
      reset         the field holds a value other than the one the reset restores"""
    from ..state import apply_gate_writes, find_section, get_path, set_path, write_values
    ot = entry.get("op_target") or {}
    lst, key = entry.get("list") or "", entry.get("key") or ""
    container, field = _container_field(m, ot, lst, key)
    if container is None:
        return {}
    sid = str(ot.get("sectionId") or "") or None if ot.get("kind") == "section" else None
    page = str(ot.get("page") or "home")
    done: dict[str, Any] = {}
    gates = (entry.get("visibility") or {}).get("gates") or []
    closed_last = style in ("prereq", "gate_only") and bool(gates)
    try:
        for g in gates[:-1] if closed_last else gates:
            if not gate_open(g, _gate_value(m, container, g["key"], state, sid, page)):
                apply_gate_writes(m, container, state, {g["key"]: g["open_value"]}, section_id=sid, page=page)
                done.setdefault("opened", {})[g["key"]] = g["open_value"]
        if closed_last:
            g = gates[-1]
            if gate_open(g, _gate_value(m, container, g["key"], state, sid, page)):
                v = _closing_value(m, container, g)
                apply_gate_writes(m, container, state, {g["key"]: v}, section_id=sid, page=page)
                done["closed"] = {g["key"]: v}
            else:
                done["closed"] = {g["key"]: _gate_value(m, container, g["key"], state, sid, page)}
    except (KeyError, ValueError, IndexError, TypeError) as e:
        done["error"] = f"gates: {e}"
    if lst and "[]" not in lst and container.kind == "section" and sid:
        spec = container.lists.get(lst)
        row = find_section(state, sid, page)
        if spec is not None and row is not None and get_path(row, spec.path) is None and "add" not in spec.ops:
            n = max([len(v) for o in container.lists.values() if o.key != lst and "[]" not in o.key
                     for v in [get_path(row, o.path)] if isinstance(v, list)] or [3])
            blank = "" if spec.scalar else {k: f.default for k, f in spec.item_fields.items() if f.has_default}
            set_path(row, spec.path, [copy.deepcopy(blank) for _ in range(n)])
            done["seeded_list"] = {lst: n}
    if style == "reset" and entry.get("reset") and field is not None and not lst:
        from ..cases import value_for
        to = entry["reset"].get("to")
        rng = random.Random(h8(f"reset:{case_id}"))
        try:
            v = value_for(field, rng, current=to)
            if v is not None and v != to:
                write_values(m, container, state, {key: v}, section_id=sid, page=page)
                done["seeded_value"] = {key: v}
        except (KeyError, ValueError, IndexError, TypeError) as e:
            done["error"] = f"reset seed: {e}"
    return done


def _closing_value(m: Manifest, container, g: dict) -> Any:
    """A value of gate g's field that keeps its field hidden (by g's first rule)."""
    from ..state import _differs, _falsy_for, _gate_field, _truthy_for
    gf = _gate_field(m, container, g["key"])
    r = g["rules"][0]
    gv = r["value"]
    if r["rule"] == "conditional_on":
        if gv == "set":
            return _falsy_for(gf)
        if gv == "unset":
            return _truthy_for(gf, None)
        return _differs(gf, gv if isinstance(gv, list) else [gv])
    if gv == "set":
        return _truthy_for(gf, None)
    if gv == "unset":
        return _falsy_for(gf)
    return gv[0] if isinstance(gv, list) and gv else gv

def validate(c: dict, target: dict, m: Manifest, engine, extras: dict, *, leak_index, pool_keys: set[str]) -> str | None:
    """None, or why the authored case is rejected (module docstring). Measures the case in place."""
    from ..r4cases import authored_values, dedup_key, remeasure, to_keys
    from ..r4score import mentions_preview
    e = c["expected"]
    mode = c["score"]["mode"]
    if not str(c.get("request") or "").strip():
        return "empty_request"
    if any(mentions_preview(h.get("assistant")) or mentions_preview(h.get("user")) for h in c.get("history") or []) \
            or mentions_preview(c.get("request")):
        return "preview_claim"
    if not isinstance(e.get("ops"), list):
        return "ops_not_json"
    if mode in ("exact", "keys"):
        if not e["ops"]:
            return "no_ops"
        for op in e["ops"]:
            if not isinstance(op, dict) or not op.get("op"):
                return "bad_op"
            why = missing_reason(op, m)
            unknown_page = why == "container_removed" and ((op.get("target") or {}).get("kind") == "page")
            if why and not unknown_page:  # a page the manifest does not describe (seo, a custom page): the engine judges
                return f"keys:{why}"
        style = c["meta"].get("style")
        if target["kind"] != "container" and style not in NON_POSITIVE_STYLES:
            if target["address"] not in case_addresses(c, m):
                return "misses_target"
        if style == "gate_only":
            gate = next(iter(((c["meta"].get("state_prep") or {}).get("closed") or {})), None)
            written = {k for o in e["ops"] if isinstance(o.get("values"), dict) for k in o["values"]}
            if gate and gate not in written:
                return f"misses_gate: a gate_only case sets {gate} (the setting that reveals the field), and only that"
        if mode == "exact":
            w = authored_values(c)
            if w:
                to_keys(c, w)
        else:
            to_keys(c, ["author_keys"])
    elif mode == "where":
        from ..exam import WHERE_VALUES
        ew = [w for w in (e.get("out_of_scope") or {}).get("expected_where") or [] if w in WHERE_VALUES]
        if not ew or e["ops"]:
            return "where_invalid"
        e["out_of_scope"]["expected_where"] = ew
    elif mode == "clarify":
        if len(e.get("candidates") or []) < 2 or e["ops"]:
            return "clarify_invalid"
    elif mode == "answer":
        if not e.get("reply_any") or e["ops"]:
            return "answer_invalid"
    err = remeasure(c, engine, m, extras)
    if err:
        detail = oracle_detail(c, err, engine)
        return f"oracle:{err}" + (f": {detail}" if detail else "")
    c["engine_fingerprint"] = getattr(engine, "fingerprint", None)
    if mode in ("exact", "keys"):
        hw = measured_hidden(c, engine, m)
        if hw:
            return ("hidden_write: " + "; ".join(hw) + " — set that setting to the value that shows the field in the same "
                    "set_fields, before the field (entry.visibility), or write a field the site shows")
        c["meta"]["positive"] = c["meta"].get("style") not in NON_POSITIVE_STYLES and target["kind"] != "container" \
            and target["address"] in case_addresses(c, m)
    if leak_index is not None and not c.get("holdout"):
        res = leak_index.check(c)
        if res["rule1"] or res["rule2"] or res["rule3"]:
            return "leak:rule" + ("1" if res["rule1"] else "2" if res["rule2"] else "3")
        c["meta"]["leak"] = {"max_jaccard": res["max_jaccard"], "nearest": res["nearest"]}
    k = dedup_key(c)
    if k in pool_keys:
        return "duplicate"
    pool_keys.add(k)
    return None


def _replay_ops(c: dict) -> list[dict]:
    e = c.get("expected") or {}
    ops = (c.get("meta") or {}).get("oracle_ops") if (c.get("score") or {}).get("mode") == "keys" else None
    return [o for o in (ops or e.get("ops") or []) if isinstance(o, dict) and o.get("op") != "pick_asset"]


def _replay(c: dict, engine):
    """The case's oracle ops on its (measured) state, as remeasure ran them (an authored
    case has no history ops, so the state is site_state; the engine cache answers)."""
    from ..verify import ctx_of
    fc = {"site_state": c["site_state"], "context": c.get("context"), "template": c.get("template")}
    return engine.run(c["site_state"], _replay_ops(c), ctx_of(fc))


def oracle_detail(c: dict, err: str, engine) -> str:
    """The engine's own words for an oracle rejection (r5 kept only the reason prefix, so
    the gallery imagesLink failure could not be diagnosed: E_INDEX "list does not exist")."""
    if str(err).startswith("history"):
        return "an earlier turn's ops did not apply"
    try:
        r = _replay(c, engine)
    except Exception as ex:  # noqa: BLE001 - the detail is diagnostic only
        return f"engine error: {redact(ex)}"[:400]
    if not r.ok:
        return "; ".join(f"{x.get('code')}: {x.get('message')}" for x in r.errors if isinstance(x, dict))[:600]
    if not r.diff():
        return "the ops change nothing on the case's state (the value is already there)"
    return ""


def measured_hidden(c: dict, engine, m: Manifest) -> list[str]:
    """The case's unintended hidden writes: the engine's post-op `hidden` entries,
    hidden_writes (static) for an engine that sends none."""
    intended = {str(k) for k in (c.get("expected") or {}).get("expect_hidden") or []}
    if not _replay_ops(c):
        return []
    try:
        res = engine_hidden(_replay(c, engine), intended)
    except Exception:  # noqa: BLE001 - fall back to the static reading
        res = None
    return hidden_writes(c, m) if res is None else res


# --- the round --------------------------------------------------------------------------------------------------------

def author_round(targets: list[dict], m: Manifest, engine, extras: dict, *, ask: Callable, round_id: str, pool: list[dict],
                 leak_index=None, smoke: bool = False, concurrency: int = CONCURRENCY, service_dir: Path | None = None,
                 progress: Callable[[str], None] = print) -> tuple[list[dict], list[dict], dict]:
    """(trainable F7 cases, F9H holdout cases, report)."""
    from ..r4cases import dedup_key
    vocab = load_vocab(service_dir)
    pool_keys = {dedup_key(c) for c in pool}
    lock = threading.Lock()
    rejects: Counter = Counter()
    usage_total: Counter = Counter()
    per_target: list[dict] = []
    from .common import run8
    tag = f"a{run8(round_id)}"  # unique per round (a hash of the run id)
    counter: Counter = Counter()

    def next_id(fam: str) -> str:
        with lock:
            counter[fam] += 1
            return f"{tag}-{fam}-{counter[fam]:04d}"

    def one(t: dict) -> tuple[list[dict], dict]:
        entry = target_entry(t, m)
        styles = style_plan(t, smoke, entry)
        spec = entry.get("field") or {}
        aliases = aliases_for(t, spec, vocab)
        examples = [x for s in sorted(set(styles)) for x in style_examples(pool, s, t["address"], 1)][:3]
        rec = {"address": t["address"], "kind": t["kind"], "holdout": t["holdout"], "asked": 0, "kept": 0, "positive": 0,
               "rejected": Counter(), "reject_details": [], "calls": 0, "usage": Counter(), "topup": 0,
               "op_target": entry.get("op_target"), "gated": bool(entry.get("visibility"))}
        kept: list[dict] = []
        todo_styles, retry = styles, None
        want_positive = 0 if (smoke or t["kind"] == "container") else MIN_POSITIVE
        for round_no in range(1, MAX_ROUNDS + 1):
            if not todo_styles:
                break
            rec["asked"] += len(todo_styles)
            try:
                obj, usage = ask(RULES, user_prompt(entry, todo_styles, aliases, examples, retry=retry), CASE_SCHEMA)
            except AuthorRefused as e:
                rec["rejected"]["refused"] += len(todo_styles)
                rec["usage"].update((e.details or {}).get("usage") or {})
                break
            rec["calls"] += 1
            rec["usage"].update(usage or {})
            raws = list((obj or {}).get("cases") or [])
            failed = []
            n_retry = len(retry or [])
            for i, st in enumerate(todo_styles):
                raw = raws[i] if i < len(raws) else None
                if raw is None:
                    failed.append({"style": st, "request": None, "reason": "missing", "again": i >= n_retry})
                    continue
                raw = dict(raw)
                raw["style"] = st
                cid = next_id("F9H" if t["holdout"] else "F7")
                c = to_case(raw, t, entry, m, case_id=cid, round_id=round_id, holdout=t["holdout"])
                with lock:
                    why = validate(c, t, m, engine, extras, leak_index=leak_index, pool_keys=pool_keys)
                if why:
                    failed.append({"style": st, "request": raw.get("request"), "reason": why, "again": i >= n_retry})
                    continue
                c["meta"]["author"]["attempt"] = round_no
                kept.append(c)
            # a first failure is re-asked once with its full reason; a second one is dropped
            again = [f for f in failed if f["again"] and round_no < MAX_ROUNDS]
            again_ids = {id(f) for f in again}
            for f in failed:
                head = f["reason"].split(":")[0]
                rec["rejected"][("retried:" + head) if id(f) in again_ids else head] += 1
                rec["reject_details"].append({"round": round_no, "style": f["style"],
                                              "request": (f["request"] or "")[:200], "reason": f["reason"][:800]})
            pos = sum(1 for c in kept if c["meta"].get("positive"))
            pending = sum(1 for f in again if f["style"] not in NON_POSITIVE_STYLES)
            more = topup_styles(want_positive - pos - pending, entry) if round_no < MAX_ROUNDS else []
            rec["topup"] += len(more)
            retry = [{"style": f["style"], "request": f["request"], "reason": f["reason"]} for f in again]
            todo_styles = [f["style"] for f in again] + more
        rec["kept"] = len(kept)
        rec["positive"] = sum(1 for c in kept if c["meta"].get("positive"))
        rec["dropped"] = rec["asked"] - len(kept)
        return kept, rec

    out_train: list[dict] = []
    out_hold: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool_ex:
        for i, (kept, rec) in enumerate(pool_ex.map(one, targets), 1):
            (out_hold if rec["holdout"] else out_train).extend(kept)
            usage_total.update(rec["usage"])
            rejects.update(rec["rejected"])
            per_target.append({**rec, "rejected": dict(rec["rejected"]), "usage": dict(rec["usage"])})
            if i % 10 == 0 or i == len(targets):
                progress(f"author: {i}/{len(targets)} targets, {len(out_train)} trainable + {len(out_hold)} holdout cases")
    short = {r["address"]: r["positive"] for r in per_target
             if not smoke and r["kind"] != "container" and r["positive"] < MIN_POSITIVE}
    report = {"model": MODEL, "effort": EFFORT, "targets": len(targets), "cases_trainable": len(out_train),
              "cases_holdout": len(out_hold), "rejects": dict(rejects), "usage": dict(usage_total),
              "min_positive": MIN_POSITIVE, "positive_by_target": {r["address"]: r["positive"] for r in per_target},
              "short_of_positive": short,
              "anthropic_usd": round(usage_cost(usage_total), 4), "per_target": per_target,
              "styles": dict(Counter(c["meta"]["style"] for c in out_train + out_hold)),
              "modes": dict(Counter(c["score"]["mode"] for c in out_train + out_hold))}
    return out_train, out_hold, report
