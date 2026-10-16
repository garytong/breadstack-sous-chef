"""probe — the pieces of the new-key probe (drift arm P, the eval's `probe` set)
that keep its rows passable and explainable.

r5 (rt-20261014-0708-7cac28-1184ff9a) showed the probe's ceiling was not 100 %:
`drift.probe_rows` fell back to the generator's default home for the rows the
eval / drift site state could not give (contentBlock-2/-3/-features,
newsletter), but every row then ran on the ONE shared state of the arm, which
has no such section — 7 of 17 rows were refused by the engine whatever the
model did, and the "dodged keys" they produced were false alarms. Fixed here:

  * every probe row carries the state it was generated (and measured) on
    (`site_state`); the runners use it (`run_state`), so the section a row
    targets is always on the state its turn runs against, and its context
    (built from that state) matches;
  * `probe_rows(engine=…)` drops a case whose ops the engine refuses ON THAT
    STATE (`accepted_on`), and a case that writes a field its own state hides
    (`case_visible`: the gate chain, e.g. stackedImageGap ← imagePosition top
    ← style image-text);
  * section types a key names but the state's home lacks (perksCards,
    perksIcons, newsletter …) get a home row, preset so the new keys are
    visible (`augment_state`, SECTION_SEEDS);
  * scalar list items the generator never writes (a list with no add op, like
    gallery `imagesLink[]`) get update cases (`scalar_list_cases`);
  * each probe turn's conversation is kept next to its result (`trajectory`).
Standard library only.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Iterable

from .manifest import Container, Field, Manifest

# Section types a probe adds to the home when a new key names them and the state lacks one, with the settings that make
# the probed keys visible. perksCards / perksIcons edit through the content-block editor; their stackedImage* fields are
# conditional_on imagePosition = top, itself conditional_on style ∈ {image-text, img-left} (manifest 1184ff9a), and
# the legacy style default (feature-cards / feature-icons) hides both.
SECTION_SEEDS: dict[str, dict] = {
    "perksCards": {"title": "Why Shop With Us", "style": "image-text", "imagePosition": "top"},
    "perksIcons": {"title": "Our Promise", "style": "image-text", "imagePosition": "top"},
    "newsletter": {"title": "Stay in the Loop"},
}
LINK_VALUES = ("/about", "/contact", "/products", "https://example.com/spring-sale")
LINK_TEMPLATES = (
    "Make image {n} in {where} link to {value}.",
    "In {where}, set the {label} to {value}.",
    "When someone clicks the {ordinal} picture in {where}, send them to {value}.",
    "Can image {n} of {where} open {value} when it's clicked?",
)
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")
TRAJ_CAP = 1500  # characters per string in a stored trajectory
TRAJ_LIST_CAP = 40


def section_type_of_key(address: str) -> str | None:
    """`sections.perksCards.stackedImageGap` → `perksCards`; other containers → None."""
    parts = str(address).split(".")
    return parts[1] if len(parts) >= 3 and parts[0] == "sections" else None


def row_id(key: str, case_id: str, i: int) -> str:
    """A probe row id that is the same in every process (drift used Python's salted hash())."""
    return f"p{hashlib.sha1(f'{key}|{case_id}'.encode()).hexdigest()[:8]}-{i}"


# ── the state a probe runs on ─────────────────────────────────────────────────

def home_ids(state: dict) -> list[str]:
    return [str(s.get("id")) for s in state.get("homeSections") or [] if isinstance(s, dict) and s.get("id")]


def augment_state(m: Manifest, state: dict, keys: Iterable[str], *, template: str = "vintage") -> tuple[dict, list[str]]:
    """`state` plus a home row for every section type a key names that its home has no row of (SECTION_SEEDS preset,
    the probe fixture's rich settings and manifest defaults, as state.build_state makes any row). → (state, added ids)."""
    from .state import build_state, type_of_section

    have = {type_of_section(i) for i in home_ids(state)}
    rows: list[dict] = []
    for k in keys:
        t = section_type_of_key(k)
        c = m.section(t) if t else None
        if c is None or c.name in have or (c.pages and "home" not in c.pages):
            continue
        have.add(c.name)
        rows.append({"id": c.name, "enabled": True, "settings": copy.deepcopy(SECTION_SEEDS.get(c.name) or {})})
    if not rows:
        return state, []
    built = build_state(m, template=template, home_sections=rows)
    out = copy.deepcopy(state)
    out.setdefault("homeSections", [])
    out["homeSections"] += built["homeSections"]
    return out, [r["id"] for r in rows]


def run_state(case: dict, default: dict) -> dict:
    """The state a row's turn runs on: a probe row's own (`site_state`, the one it was generated and measured on),
    any other row the arm's shared state."""
    st = case.get("site_state")
    if (case.get("meta") or {}).get("class") == "probe" and isinstance(st, dict) and st:
        return st
    return default


def target_present(case: dict, state: dict) -> bool:
    """Is the section a row targets on `state`'s page? Non-section targets are always present."""
    from .state import find_section
    t = case.get("target") or {}
    if not isinstance(t, dict) or t.get("kind") != "section":
        return True
    return find_section(state, str(t.get("sectionId")), str(t.get("page") or "home")) is not None


# ── visibility (the editor's gates, chained) ──────────────────────────────────

def _gate_hit(cur: Any, v: Any) -> bool:
    if v == "set":
        return cur not in (None, "", False, [], {})
    if v == "unset":
        return cur in (None, "", False, [], {})
    if isinstance(v, list):
        return cur in v
    return cur == v


def field_visible(m: Manifest, c: Container, f: Field, state: dict, *, section_id: str | None, page: str = "home",
                  _depth: int = 0) -> bool:
    """Would the editor show `f` on `state`? Every `conditional_on` holds and no `hidden_when` does, and a gate field
    is itself visible (stackedImageGap ← imagePosition = top ← style image-text). Gates on list items (`slides[].x`)
    and templated keys are not judged (True)."""
    from .state import read_field
    for rule in ("conditional_on", "hidden_when"):
        g = f.spec.get(rule)
        if not isinstance(g, dict) or not g.get("key") or "[]" in str(g["key"]):
            continue
        gf = m.resolve_field(c, str(g["key"]))
        if gf is None or gf.keyed_by:
            continue
        try:
            cur = read_field(c, gf, state, section_id=section_id, page=page)
        except (KeyError, IndexError, TypeError, ValueError):
            cur = None
        if cur is None and gf.has_default:
            cur = gf.default
        hit = _gate_hit(cur, g.get("value"))
        if rule == "hidden_when" and hit:
            return False
        if rule == "conditional_on":
            if not hit:
                return False
            if _depth < 4 and not field_visible(m, c, gf, state, section_id=section_id, page=page, _depth=_depth + 1):
                return False
    return True


def case_visible(m: Manifest, case: Any) -> tuple[bool, str | None]:
    """(True, None) when every section field the case's ops write is visible on the case's own state, else
    (False, "hidden:<key>")."""
    from .exam import touches_of
    st = case.site_state
    for op in case.browser_ops:
        for t in touches_of(op):
            tg = t["target"] or {}
            if tg.get("kind") != "section" or not t["key"] or str(t["key"]).startswith("__"):
                continue
            c = m.section(str(tg.get("sectionId")))
            if c is None:
                continue
            if t["list"]:
                lst = c.lists.get(t["list"])
                f = lst.item_fields.get(t["key"]) if lst else None
            else:
                f = c.fields.get(t["key"])
            if f is not None and not field_visible(m, c, f, st, section_id=str(tg.get("sectionId")),
                                                   page=str(tg.get("page") or "home")):
                return False, f"hidden:{t['key']}"
    return True, None


# ── the engine check ──────────────────────────────────────────────────────────

def accepted_on(case: Any, engine: Any) -> tuple[bool, str | None]:
    """Run the case's expected ops on ITS OWN state (the one its row runs on) → (accepted, why not). Any engine shape
    the lab uses: engine.Engine / PersistentEngine (an EngineResult), collect.DictEngine or drift.ManifestEngine (a
    dict). Refused ops, an engine error or an empty write reject the case."""
    from .engine import GOLDEN_CTX
    ops = case.browser_ops
    try:
        r = engine.run(case.site_state, ops, GOLDEN_CTX) if hasattr(engine, "run") else engine(case.site_state, ops, GOLDEN_CTX)
    except Exception as e:  # noqa: BLE001 — engine.EngineError and friends: the case is not measurable
        return False, f"engine_error:{type(e).__name__}"
    if isinstance(r, dict):
        ok, results, changes = bool(r.get("ok")), list(r.get("results") or []), list(r.get("changes") or [])
    else:
        ok, results, changes = bool(r.ok), list(r.results or []), list(r.changes or [])
    if not ok:
        errs = [((x.get("error") or {}).get("code") if isinstance(x.get("error"), dict) else None)
                or next((e.get("code") for e in x.get("errors") or [] if isinstance(e, dict)), None)
                for x in results if not x.get("ok")]
        return False, f"op_refused:{next((e for e in errs if e), 'E_?')}"
    if ops and not changes and not any(x.get("written") for x in results):
        return False, "empty_diff"
    return True, None


# ── scalar list items the generator never writes ──────────────────────────────

def _scalar_lists(m: Manifest, keys: Iterable[str]) -> dict[tuple[str, str], set[str]]:
    """{(section type, list key): {the wanted addresses}} for keys on a scalar list (`imagesLink[]`, `…[].value`)."""
    out: dict[tuple[str, str], set[str]] = {}
    for k in keys:
        t = section_type_of_key(k)
        c = m.section(t) if t else None
        if c is None:
            continue
        rest = k.split(".", 2)[2]
        lk = rest.split("[]", 1)[0] if "[]" in rest else None
        lst = c.lists.get(lk) if lk else None
        if lst is None or not lst.scalar or lst.item_fields["value"].type not in ("text", "richtext"):
            continue
        out.setdefault((c.name, lk), set()).add(k)
    return out


def scalar_list_cases(m: Manifest, state: dict, keys: Iterable[str], *, seed: int = 0, per_list: int = 4,
                      template: str = "vintage") -> list:
    """`edit_list update` cases for the text items of scalar lists the keys name (gallery `imagesLink[]`): the
    generator writes no such list (no add op, no item object), so the drift probe left them unprobed. The list is
    seeded on the case's state when the section has none (the engine refuses an update of a missing list: E_INDEX
    'list does not exist' — a fresh gallery has no imagesLink), one entry per image, the first already linked, as the
    launcher fixtures hold it; the item's own gate is opened."""
    from .cases import Case, case_rng, container_phrase
    from .state import (apply_gate_writes, context_of, gate_writes, load_context_fixture, section_settings,
                        sections_of)

    ctxf = load_context_fixture()
    out = []
    for (type_name, lk), wanted in sorted(_scalar_lists(m, keys).items()):
        c = m.section(type_name)
        f = c.lists[lk].item_fields["value"]
        rows = [s for s in sections_of(state, "home") if m.section(str(s.get("id"))) is c]
        for i in range(per_list * len(wanted)):  # per wanted address, so `imagesLink[]` and `…[].value` ask different turns
            if not rows:
                break
            row = rows[i % len(rows)]
            sid = str(row["id"])
            st = copy.deepcopy(state)
            settings = section_settings(st, sid, "home")
            n = max(3, len(settings.get("images") or []))
            cur = settings.get(lk)
            if not isinstance(cur, list) or len(cur) < n:
                cur = list(cur) if isinstance(cur, list) else []
                cur += [""] * (n - len(cur))
                if not any(cur):
                    cur[0] = "/products"
                settings[lk] = cur
            gw = gate_writes(m, c, f)
            if gw:
                apply_gate_writes(m, c, st, gw, section_id=sid, page="home")
            index = (1 + i) % n
            value = LINK_VALUES[i % len(LINK_VALUES)]
            if settings[lk][index] == value:
                value = LINK_VALUES[(i + 1) % len(LINK_VALUES)]
            cid = f"probe/list_update/{type_name}/{sid}/{lk}/{i}"
            rng = case_rng(seed, cid)
            where = container_phrase(c, st, rng, sid, "home")
            label = str(f.label or "").replace("{n}", str(index + 1)) or f"image {index + 1} link"
            request = LINK_TEMPLATES[i % len(LINK_TEMPLATES)].format(
                n=index + 1, where=where, label=label.lower(), value=value,
                ordinal=ORDINALS[index] if index < len(ORDINALS) else f"#{index + 1}")
            target = {"kind": "section", "page": "home", "sectionId": sid}
            op = {"op": "edit_list", "target": target, "list": lk, "action": "update", "index": index, "item": {"value": value}}
            out.append(Case(id=cid, kind="list_update", target=target, fields=[f"{lk}[]"], site_state=st,
                            context=context_of(st, "home", ctxf, m), request=request,
                            expected={"ops": [op], "diff": []}, holdout=True, manifest_hash=m.hash, template=template,
                            meta={"generator": "souschef_lab.probe", "seed": seed, "gate_writes": gw, "list": lk,
                                  "index": index, "value": value, "seeded_list": True}))
    return out


# ── the conversation of a probe turn ──────────────────────────────────────────

def _cut(v: Any, cap: int = TRAJ_CAP) -> Any:
    if isinstance(v, str):
        return v if len(v) <= cap else v[:cap] + f"… [+{len(v) - cap} chars]"
    if isinstance(v, list):
        out = [_cut(x, cap) for x in v[:TRAJ_LIST_CAP]]
        return out + ([f"… [+{len(v) - TRAJ_LIST_CAP} items]"] if len(v) > TRAJ_LIST_CAP else [])
    if isinstance(v, dict):
        return {k: _cut(x, cap) for k, x in v.items()}
    return v


def trajectory(row: dict, case: dict) -> dict:
    """What a probe turn did, kept next to its trimmed result so a failing key can be explained: the request, the
    expectation, the sections the context listed, the model's calls in order with the engine's answer to each batch
    (errors, W_HIDDEN and the gate value), the reply, the verdict. The fields block of the first user message is
    left out (it is rebuilt from the manifest); every string is capped at TRAJ_CAP characters."""
    meta = case.get("meta") or {}
    ctx = case.get("context") or {}
    engine = []
    for b in row.get("engine_results") or []:
        engine.append({"ok": b.get("ok"), "error": b.get("error"),
                       "results": [{k: x.get(k) for k in ("ok", "error", "errors", "warnings", "written", "hidden") if x.get(k)}
                                   for x in b.get("results") or []]})
    msgs = list(row.get("messages") or [])[1:]  # [0] = the context + fields block + request
    return _cut({
        "request": case.get("request"), "probe_key": meta.get("probe_key"), "probe_kind": meta.get("kind"),
        "target": case.get("target"), "expected_keys": (case.get("expected") or {}).get("keys"),
        "generator_ops": meta.get("generator_ops"), "source_case": meta.get("source_case"),
        "page": ctx.get("page"), "context_sections": [s.get("id") for s in ctx.get("sections") or [] if isinstance(s, dict)],
        "status": row.get("status"), "code": row.get("code"), "error": row.get("error"),
        "verdict": row.get("verdict"), "reasons": row.get("reasons"), "scored_by": row.get("scored_by"),
        "tool_calls": [{"name": c.get("name"), "input": c.get("input")} for c in row.get("tool_calls") or []],
        "engine": engine, "messages": msgs, "reply": row.get("reply"),
    })


def plan_summary(rows: list[dict], coverage: dict[str, int], stats: dict[str, dict] | None = None) -> dict:
    """The probe's plan as stored next to its results (drift exam/P-plan.json, eval/probe-plan.json): every row's
    key, kind, target and request, and per key what was dropped and why."""
    return {"coverage": coverage, "dropped": stats or {},
            "rows": [{"id": r["id"], "probe_key": (r.get("meta") or {}).get("probe_key"), "kind": (r.get("meta") or {}).get("kind"),
                      "target": r.get("target"), "request": r.get("request"), "expected_keys": (r.get("expected") or {}).get("keys"),
                      "source_case": (r.get("meta") or {}).get("source_case"), "state_sections": home_ids(r.get("site_state") or {}),
                      "generator_ops": (r.get("meta") or {}).get("generator_ops")} for r in rows]}
