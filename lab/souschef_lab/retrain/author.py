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
  per target 8 cases: 2 verbatim-ish, 2 colloquial (vocab.txt aliases), 1 short
             command / relative value, 1 focus (the section / panel open, no
             target named), 1 follow-up (ask, then the value), 1 near-miss or
             out-of-editor; a new container adds 6 container-level cases (add the
             section, reorder, name it on another page)
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
  covers   a write case must write its target
  preview  no authored assistant text says "preview" (r4score's one rule)
  leak     leak.py rules 1-3 against the exam and the study's real conversations
  dedup    no duplicate of a pool case (r4cases.dedup_key)
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
FIELD_STYLES = (("verbatim", 2), ("colloquial", 2), ("short", 1), ("focus", 1), ("followup", 1), ("near_miss", 1))
CONTAINER_STYLES = (("container_add", 2), ("container_reorder", 2), ("container_other_page", 2))
SMOKE_STYLES = (("verbatim", 1), ("colloquial", 1), ("followup", 1), ("near_miss", 1))
STYLE_FAMILY = {"verbatim": "F0", "colloquial": "F1", "short": "F1", "focus": "F4", "followup": "F2", "near_miss": "F3",
                "container_add": "F5", "container_reorder": "F5", "container_other_page": "F5"}
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


def coverage(cases: list[dict], m: Manifest, *, holdout: bool = False) -> Counter:
    """Cases writing each address: the trainable ones, or (holdout=True) the
    new-key holdouts (F9H) — a holdout target is covered by earlier rounds'
    F9H cases, which a skipped round keeps in cases/exam-ext/."""
    cnt: Counter = Counter()
    for c in cases:
        if bool(c.get("holdout")) != holdout or (holdout and (c.get("meta") or {}).get("family") != "F9H"):
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
    report = {"wanted": len(want), "covered_enough": len(skipped), "targets": len(targets),
              "truncated": max(0, len(targets) - max_targets), "by_reason": dict(Counter(t["reason"] for t in targets)),
              "holdout_targets": sum(1 for t in targets if t["holdout"])}
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
    """The manifest entry the author writes for, with the op target to use."""
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
    op_target = op_target_for(grp, name)
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
    return entry


def op_target_for(grp: str, name: str) -> dict:
    if grp == "sections":
        return {"kind": "section", "page": "home", "sectionId": name}
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


def style_plan(target: dict, smoke: bool = False) -> list[str]:
    if smoke:
        styles = SMOKE_STYLES
    elif target["kind"] == "container":
        styles = CONTAINER_STYLES
    else:
        styles = FIELD_STYLES
    return [s for s, n in styles for _ in range(n)]


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
                if entry.get("list"):
                    op = {"op": "edit_list", "target": entry["op_target"], "list": entry["list"], "action": "update",
                          "index": [0, 0] if "[]" in entry["list"] else 0, "item": {key: value}}
                else:
                    op = {"op": "set_fields", "target": entry["op_target"], "values": {key: value}}
                label = str((entry.get("field") or {}).get("label") or key)
                req = f"set the {label.lower()} to {value}"
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
                case.update(request=req, oracle_ops_json=json.dumps([op]))
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
    return c


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
        if target["kind"] != "container" and (c["meta"].get("style") != "near_miss"):
            if target["address"] not in case_addresses(c, m):
                return "misses_target"
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
        return f"oracle:{err}"
    c["engine_fingerprint"] = getattr(engine, "fingerprint", None)
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
        styles = style_plan(t, smoke)
        spec = entry.get("field") or {}
        aliases = aliases_for(t, spec, vocab)
        examples = [x for s in sorted(set(styles)) for x in style_examples(pool, s, t["address"], 1)][:3]
        rec = {"address": t["address"], "kind": t["kind"], "holdout": t["holdout"], "asked": len(styles), "kept": 0,
               "rejected": Counter(), "calls": 0, "usage": Counter()}
        kept: list[dict] = []
        todo_styles, retry = styles, None
        for round_no in (1, 2):
            if not todo_styles:
                break
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
            for i, st in enumerate(todo_styles):
                raw = raws[i] if i < len(raws) else None
                if raw is None:
                    failed.append({"style": st, "request": None, "reason": "missing"})
                    continue
                raw = dict(raw)
                raw["style"] = st
                cid = next_id("F9H" if t["holdout"] else "F7")
                c = to_case(raw, t, entry, m, case_id=cid, round_id=round_id, holdout=t["holdout"])
                with lock:
                    why = validate(c, t, m, engine, extras, leak_index=leak_index, pool_keys=pool_keys)
                if why:
                    failed.append({"style": st, "request": raw.get("request"), "reason": why})
                    continue
                c["meta"]["author"]["attempt"] = round_no
                kept.append(c)
            for f in failed:
                rec["rejected"][f["reason"].split(":")[0] if round_no == 2 else "retried:" + f["reason"].split(":")[0]] += 1
            todo_styles = [f["style"] for f in failed]
            retry = failed
        rec["kept"] = len(kept)
        rec["dropped"] = len(styles) - len(kept)
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
    report = {"model": MODEL, "effort": EFFORT, "targets": len(targets), "cases_trainable": len(out_train),
              "cases_holdout": len(out_hold), "rejects": dict(rejects), "usage": dict(usage_total),
              "anthropic_usd": round(usage_cost(usage_total), 4), "per_target": per_target,
              "styles": dict(Counter(c["meta"]["style"] for c in out_train + out_hold)),
              "modes": dict(Counter(c["score"]["mode"] for c in out_train + out_hold))}
    return out_train, out_hold, report
