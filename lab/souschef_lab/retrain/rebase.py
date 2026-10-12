"""rebase — every pool family onto another manifest (design §3.2 step 1).

r4cases.merge could only move the verbatim family (F0) onto another manifest
and dropped F1–F6 as `other_manifest`. This generalises r4cases.rebase_f0 +
r4cases.remeasure to every case of the pool, whatever manifest it was written
on:

  1. renames   a field removed and a field added in the same container (or list),
               same type, label identical or label-token Jaccard >= 0.8 or help
               identical, is a rename (`renames_between`; the drift check's
               static diff uses the same rule); the old key is replaced by the new
               one in `fields`, every op (expected, oracle, history, history calls),
               `expected.keys` / `alt`, `expected.direction`
  2. gone      a write whose key, list or container is absent from the new manifest
               drops the case: key_removed | list_removed | container_removed
  3. relabel   a verbatim case (F0/F0H, meta.style = verbatim) quotes the field's label
               (meta.field_phrase); when the new manifest labels the field otherwise the
               phrase is replaced in the request by the generator's phrase for the new
               label (cases.phrase_candidates, same rank) → meta.relabelled
  4. contexts  every turn's context re-derived with state.context_of on the new manifest,
               earlier turns' ops replayed through the engine (r4cases.remeasure)
  5. measure   exact: the diff (and each alt_exact) re-measured — exact_refused /
               exact_empty drop; keys: the oracle ops must apply (oracle_refused);
               where / clarify / answer: unchanged
  6. leak      the hold-out filter (leak.py rules 1-3) over every trainable case again

A family that loses more than 20 % of its cases (`MAX_FAMILY_LOSS`) fails the
stage (`rebase_loss`): a person looks before a round trains on a thinned pool.
"""
from __future__ import annotations

import copy
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable

from ..manifest import Manifest

MAX_FAMILY_LOSS = 0.20
MIN_FAMILY_FOR_GATE = 5
RENAME_JACCARD = 0.8
CONTAINER_GROUPS = ("sections", "globalPanels", "pages", "details")
SPECIAL_KEYS = ("__layout", "__pinned")
SKIP_FAMILIES = ("F0X",)  # the int-run3 holdout unchanged (r3's continuity exam): never rebased


# --- addressing (retrain_check.field_index's: `<group>.<name>.<key>`, `<group>.<name>.<list>[].<key>`) -----------------

def split_addr(addr: str) -> tuple[str, str, str]:
    """`sections.heroBanner.slides[].bullets[].title` → (`sections.heroBanner`, `slides[].bullets`, `title`);
    `branding.logoUrl` → (`branding`, ``, `logoUrl`)."""
    parts = addr.split(".")
    if parts[0] == "branding":
        ctr, rest = "branding", ".".join(parts[1:])
    else:
        ctr, rest = ".".join(parts[:2]), ".".join(parts[2:])
    if "[]." in rest:
        lst, key = rest.rsplit("[].", 1)
        return ctr, lst, key
    return ctr, "", rest


def _tokens(s: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(s or "").lower()))


def _containers(doc: dict) -> dict[str, dict]:
    out = {}
    for grp in CONTAINER_GROUPS:
        for name, c in (doc.get(grp) or {}).items():
            if isinstance(c, dict):
                out[f"{grp}.{name}"] = c
    if isinstance(doc.get("branding"), dict):
        out["branding"] = doc["branding"]
    return out


def _field_specs(doc: dict) -> dict[tuple[str, str], dict[str, dict]]:
    """(container, list) → {key: spec}; list '' = the container's own fields."""
    out: dict[tuple[str, str], dict[str, dict]] = {}
    for cname, c in _containers(doc).items():
        out[(cname, "")] = {k: (f or {}) for k, f in (c.get("fields") or {}).items()}
        for lname, lst in (c.get("lists") or {}).items():
            out[(cname, lname)] = {k: (f or {}) for k, f in ((lst or {}).get("item") or {}).items()}
    return out


def renames_between(old: dict, new: dict) -> list[dict]:
    """[{from, to, score}] addresses: a removed and an added field of one container
    (and list), same type, label identical or label-token Jaccard >= 0.8, or help
    identical. Each removed field maps to at most one added field (best score)."""
    a, b = _field_specs(old), _field_specs(new)
    out = []
    for slot in sorted(set(a) & set(b)):
        removed = {k: v for k, v in a[slot].items() if k not in b[slot]}
        added = {k: v for k, v in b[slot].items() if k not in a[slot]}
        taken: set[str] = set()
        for rk, rs in sorted(removed.items()):
            best = None
            for ak, as_ in sorted(added.items()):
                if ak in taken or str(rs.get("type") or "") != str(as_.get("type") or ""):
                    continue
                la, lb = str(rs.get("label") or "").strip(), str(as_.get("label") or "").strip()
                ta, tb = _tokens(la), _tokens(lb)
                jac = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
                help_same = bool(rs.get("help")) and rs.get("help") == as_.get("help")
                score = 1.0 if (la and la == lb) or help_same else jac
                if score >= RENAME_JACCARD and (best is None or score > best[1]):
                    best = (ak, score)
            if best:
                taken.add(best[0])
                ctr, lst = slot
                pre = f"{ctr}.{lst}[]." if lst else f"{ctr}."
                out.append({"from": pre + rk, "to": pre + best[0], "score": round(best[1], 3)})
    return out


def rename_map(renames: Iterable[dict]) -> dict[tuple[str, str, str], str]:
    out = {}
    for r in renames or []:
        c1, l1, k1 = split_addr(str(r["from"]))
        c2, l2, k2 = split_addr(str(r["to"]))
        if (c1, l1) == (c2, l2) and k1 != k2:
            out[(c1, l1, k1)] = k2
    return out


# --- containers of an op target ---------------------------------------------------------------------------------------

def container_of(m: Manifest, target: dict | None, key: str = ""):
    """(address, Container | None) of an op target; a global key is looked up in its
    own panel first, then in every panel (the engine merges globals)."""
    t = target or {}
    kind = t.get("kind")
    if kind == "section":
        c = m.section(str(t.get("sectionId") or ""))
        return (f"sections.{c.name}" if c else f"sections.?{t.get('sectionId')}"), c
    if kind == "global":
        c = m.panel(str(t.get("panel") or ""))
        if c is not None and (not key or key in c.fields):
            return f"globalPanels.{c.name}", c
        if key:
            for g in m.containers("global"):
                if key in g.fields:
                    return f"globalPanels.{g.name}", g
        return (f"globalPanels.{c.name}" if c else f"globalPanels.?{t.get('panel')}"), c
    if kind == "branding":
        try:
            return "branding", m.branding
        except KeyError:
            return "branding", None
    if kind == "page":
        page = str(t.get("page") or "")
        c = m.page(page)
        if key and (c is None or key not in c.fields) and key in m.common_page_fields(page):
            return "pages._common", m.page("_common")
        return (f"pages.{page}" if c else f"pages.?{page}"), c
    return f"?{kind}", None


def _resolves(m: Manifest, c, key: str) -> bool:
    from ..state import resolve_key
    if key in SPECIAL_KEYS or key.startswith("__"):
        return True
    if c.kind == "global":
        return key in c.fields or key in m.global_fields() or resolve_key(m, c, key) is not None
    return resolve_key(m, c, key) is not None


def field_of(m: Manifest, target: dict | None, lst: str, key: str):
    """The manifest Field a (target, list, key) write names, or None."""
    from ..state import resolve_key
    _, c = container_of(m, target, key)
    if c is None:
        return None
    if lst:
        spec = c.lists.get(lst)
        return spec.item_fields.get(key) if spec else None
    f = resolve_key(m, c, key)
    if f is None and c.kind == "global":
        f = m.global_fields().get(key)
    if f is None and c.kind == "page":
        f = m.common_page_fields(c.name).get(key)
    return f


# --- ops: rename + check ----------------------------------------------------------------------------------------------

def _op_name(op: dict) -> str:
    return str(op.get("op") or op.get("name") or "")


def _op_body(op: dict) -> dict:
    """An op ({op, target, …}) or a logged call ({name, input: {target, …}}) — the dict holding target/values."""
    return op["input"] if isinstance(op.get("input"), dict) and "op" not in op else op


def rename_op(op: dict, m_old: Manifest, rmap: dict) -> int:
    """Rename the keys an op writes (in place). Returns how many were renamed."""
    if not rmap:
        return 0
    name, body = _op_name(op), _op_body(op)
    target = body.get("target") if isinstance(body.get("target"), dict) else None
    n = 0
    if name == "set_fields" and isinstance(body.get("values"), dict):
        new = {}
        for k, v in body["values"].items():
            ctr, _ = container_of(m_old, target, k)
            nk = rmap.get((ctr, "", k), k)
            n += nk != k
            new[nk] = v
        body["values"] = new
        if isinstance(body.get("assets"), dict):
            body["assets"] = {rmap.get((container_of(m_old, target, k)[0], "", k), k): v for k, v in body["assets"].items()}
    elif name in ("reset", "restore_default") and isinstance(body.get("keys"), list):
        out = []
        for k in body["keys"]:
            ctr, _ = container_of(m_old, target, str(k))
            nk = rmap.get((ctr, "", str(k)), k)
            n += nk != k
            out.append(nk)
        body["keys"] = out
    elif name == "edit_list" and isinstance(body.get("item"), dict):
        ctr, _ = container_of(m_old, target)
        lst = str(body.get("list") or "")
        new = {}
        for k, v in body["item"].items():
            nk = rmap.get((ctr, lst, k), k)
            n += nk != k
            new[nk] = v
        body["item"] = new
    return n


def _leaf_touches(op: dict) -> list[dict]:
    from ..exam import touches_of
    body = _op_body(op)
    ts = touches_of({**body, "op": _op_name(op)})
    keys = [t["key"] for t in ts]
    return [t for t in ts if not any(o != t["key"] and t["key"] and o.startswith(t["key"] + ".") for o in keys)]


def missing_reason(op: dict, m: Manifest) -> str | None:
    """Why an op cannot stand on manifest m: container_removed | list_removed | key_removed, or None."""
    name, body = _op_name(op), _op_body(op)
    if name in ("pick_asset", "report_unsupported", "lookup_fields", "describe_target"):
        return None
    if name in ("set_page_layout", "reset_pinned_section"):
        return None
    target = body.get("target") if isinstance(body.get("target"), dict) else None
    if name == "edit_list":
        lst = str(body.get("list") or "")
        if lst == "sections":  # a page's section rows: the page need not be a manifest container
            item = body.get("item") if isinstance(body.get("item"), dict) else {}
            typ = item.get("type") or item.get("id")
            if typ and body.get("action") == "add" and m.section(str(typ)) is None:
                return "container_removed"
            return None
        _, c = container_of(m, target)
        if c is None:
            return "container_removed"
        spec = c.lists.get(lst)
        if spec is None:
            return "list_removed"
        for k in (body.get("item") or {}) if isinstance(body.get("item"), dict) else []:
            if k not in spec.item_fields:
                return "key_removed"
        return None
    for t in _leaf_touches(op):
        key = str(t["key"] or "")
        _, c = container_of(m, t["target"], key)
        if c is None:
            return "container_removed"
        if key and not _resolves(m, c, key):
            return "key_removed"
    return None


def _case_ops(c: dict) -> list[dict]:
    """Every op / call a case carries, by reference (renamed in place)."""
    e = c.get("expected") or {}
    meta = c.get("meta") or {}
    ops = list(e.get("ops") or []) + list(meta.get("oracle_ops") or [])
    for a in e.get("alt_exact") or []:
        ops += list(a.get("ops") or [])
    for h in c.get("history") or []:
        ops += list(h.get("ops") or [])
        ops += [x for x in h.get("calls") or [] if isinstance(x, dict)]
    return ops


def _scored_ops(c: dict) -> list[dict]:
    e = c.get("expected") or {}
    ops = list(e.get("ops") or []) + list((c.get("meta") or {}).get("oracle_ops") or [])
    for h in c.get("history") or []:
        ops += list(h.get("ops") or [])
    return ops


def rename_case(c: dict, m_old: Manifest, rmap: dict) -> list[str]:
    """Apply the rename map to a case (in place). Returns the renamed addresses."""
    if not rmap:
        return []
    done: list[str] = []
    n = 0
    for op in _case_ops(c):
        n += rename_op(op, m_old, rmap)
    e = c.get("expected") or {}
    for s in [e.get("keys") or []] + [s for s in e.get("alt") or [] if isinstance(s, list)]:
        for k in s:
            if isinstance(k, dict) and k.get("key"):
                lst = str(k.get("list") or "")
                ctr, _ = container_of(m_old, k.get("target"), "" if lst else str(k["key"]))
                nk = rmap.get((ctr, lst, str(k["key"])))
                if nk:
                    done.append(f"{ctr}.{lst + '[].' if lst else ''}{k['key']}")
                    k["key"] = nk
    d = e.get("direction")
    if isinstance(d, dict) and d.get("key"):
        ctr, _ = container_of(m_old, c.get("target"), str(d["key"]))
        nk = rmap.get((ctr, "", str(d["key"])))
        if nk:
            d["key"] = nk
    if c.get("fields"):
        ctr, _ = container_of(m_old, c.get("target"), str((c.get("fields") or [""])[0]))
        out = []
        for k in c["fields"]:
            nk = rmap.get((ctr, "", str(k)))
            if nk:
                done.append(f"{ctr}.{k}")
            out.append(nk or k)
        c["fields"] = out
    hk = (c.get("meta") or {}).get("holdout_key")
    if hk and "." in str(hk):
        typ, key = str(hk).split(".", 1)
        for (ctr, lst, k), nk in rmap.items():
            if not lst and k == key and ctr.split(".")[-1] == typ:
                c["meta"]["holdout_key"] = f"{typ}.{nk}"
    if n and not done:
        done.append(f"{n} op key(s)")
    return sorted(set(done))


# --- the verbatim relabel ---------------------------------------------------------------------------------------------

def relabel(c: dict, m_old: Manifest, m_new: Manifest) -> dict | None:
    """A verbatim case's request re-worded for the new label (module docstring, step 3).
    Returns {from, to} when the request changed, None otherwise (meta.relabel_missed
    is set when the label changed but the old phrase is not in the request)."""
    from ..cases import label_text, phrase_candidates
    meta = c.get("meta") or {}
    if meta.get("style") != "verbatim" or not meta.get("field_phrase"):
        return None
    old_phrase = str(meta["field_phrase"])
    target = c.get("target") if isinstance(c.get("target"), dict) else None
    key = str(meta.get("old_field_key") or (c.get("fields") or [""])[0])
    new_key = str((c.get("fields") or [key])[0])
    lst = str(meta.get("via_list") or "")
    f_old = field_of(m_old, target, lst, key) or field_of(m_old, target, "", key)
    f_new = field_of(m_new, target, lst, new_key) or field_of(m_new, target, "", new_key)
    if f_old is None or f_new is None:
        return None
    if (f_old.spec.get("label") or "") == (f_new.spec.get("label") or ""):
        return None
    try:
        olds = phrase_candidates(f_old, key)
        news = phrase_candidates(f_new, new_key)
    except Exception:  # noqa: BLE001 — an odd label: fall back to the label text alone
        olds, news = [], []
    new_phrase = None
    if old_phrase in olds and news:
        new_phrase = news[min(olds.index(old_phrase), len(news) - 1)]
    if new_phrase is None:
        new_phrase = label_text(f_new, new_key)
    req = str(c.get("request") or "")
    if old_phrase.lower() not in req.lower():
        meta["relabel_missed"] = {"phrase": old_phrase, "new_label": f_new.spec.get("label")}
        return None
    c["request"] = re.sub(re.escape(old_phrase), lambda _m: new_phrase, req, count=1, flags=re.IGNORECASE)
    meta["relabelled"] = {"from": old_phrase, "to": new_phrase, "request_was": req}
    meta["field_phrase"] = new_phrase
    return {"from": old_phrase, "to": new_phrase}


# --- one case, the pool -----------------------------------------------------------------------------------------------

def rebase_case(src: dict, m_old: Manifest, m_new: Manifest, rmap: dict, engine, extras: dict) -> tuple[dict | None, str, dict]:
    """(case on m_new, "", notes) or (None, why, notes)."""
    from ..r4cases import remeasure

    c = copy.deepcopy(src)
    meta = c.setdefault("meta", {})
    notes: dict[str, Any] = {}
    old_hash, old_fp = c.get("manifest_hash"), c.get("engine_fingerprint")
    same_engine = old_fp and old_fp == getattr(engine, "fingerprint", None)
    if old_hash == m_new.hash and same_engine:
        return c, "", {"unchanged": True}
    if c.get("fields"):
        meta.setdefault("old_field_key", (c.get("fields") or [None])[0])
    renamed = rename_case(c, m_old, rmap) if old_hash != m_new.hash else []
    if renamed:
        notes["renamed"] = renamed
    for op in _scored_ops(c):
        why = missing_reason(op, m_new)
        # removed = the old manifest knew it and the new one does not (a page the manifest never described, like
        # `seo`, is the engine's to judge: remeasure below)
        if why and (m_old is m_new or missing_reason(op, m_old) is None):
            return None, why, notes
    e = c.get("expected") or {}
    for k in e.get("keys") or []:
        if isinstance(k, dict) and k.get("key"):
            def known(m: Manifest) -> bool:
                return (field_of(m, k.get("target"), str(k.get("list") or ""), str(k["key"])) is not None
                        or field_of(m, k.get("target"), "", str(k["key"])) is not None)
            if not known(m_new) and (m_old is m_new or known(m_old)):
                return None, "key_removed", notes
        elif isinstance(k, dict) and k.get("list") and str(k["list"]) != "sections":
            _, ctr = container_of(m_new, k.get("target"))
            _, octr = container_of(m_old, k.get("target"))
            if octr is not None and str(k["list"]) in octr.lists:
                if ctr is None:
                    return None, "container_removed", notes
                if str(k["list"]) not in ctr.lists:
                    return None, "list_removed", notes
    if old_hash != m_new.hash:
        rl = relabel(c, m_old, m_new)
        if rl:
            notes["relabelled"] = rl
    err = remeasure(c, engine, m_new, extras)
    if err:
        return None, err, notes
    c["manifest_hash"] = m_new.hash
    c["engine_fingerprint"] = getattr(engine, "fingerprint", None)
    meta.pop("old_field_key", None)
    if old_hash != m_new.hash or not same_engine:
        hist = list(meta.get("rebased_history") or [])
        hist.append({"manifest_hash": old_hash, "engine_fingerprint": old_fp})
        meta["rebased_history"] = hist[-5:]
    return c, "", notes


def rebase_pool(cases: list[dict], manifests: dict[str, Manifest], m_new: Manifest, engine, extras: dict, *,
                why=None, workers: int = 4, check_leak: bool = True, progress: Callable[[str], None] = print,
                gate: bool = True) -> tuple[list[dict], dict]:
    """Every case on `m_new`. `manifests` = the source manifests by hash. Returns
    (kept cases, report). The loss gate is reported in report['gate'] and
    report['loss_failed'] lists the families over MAX_FAMILY_LOSS."""
    from ..leak import ExamIndex, filter_cases

    rmaps = {h: rename_map(renames_between(m.doc, m_new.doc)) for h, m in manifests.items() if h != m_new.hash}
    report: dict[str, Any] = {"to_manifest": m_new.hash, "engine_fingerprint": getattr(engine, "fingerprint", None),
                              "renames": {h[:12]: [{"from": f"{a}.{l + '[].' if l else ''}{k}", "to": v}
                                                   for (a, l, k), v in sorted(rm.items())] for h, rm in rmaps.items()},
                              "families": {}, "dropped_examples": []}
    fam_counts: dict[str, Counter] = defaultdict(Counter)
    todo = []
    for c in cases:
        fam = str((c.get("meta") or {}).get("family") or "?")
        if fam in SKIP_FAMILIES:
            fam_counts[fam]["skipped"] += 1
            continue
        fam_counts[fam]["in"] += 1
        todo.append(c)

    def one(c: dict):
        fam = str((c.get("meta") or {}).get("family") or "?")
        h = c.get("manifest_hash")
        m_old = manifests.get(h) if h != m_new.hash else m_new
        if m_old is None:
            return fam, c, None, "no_source_manifest", {}
        try:
            out, why_, notes = rebase_case(c, m_old, m_new, rmaps.get(h, {}), engine, extras)
        except Exception as e:  # noqa: BLE001 — one odd case must not stop the pool
            return fam, c, None, f"error:{type(e).__name__}", {"error": str(e)[:300]}
        return fam, c, out, why_, notes

    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, r in enumerate(pool.map(one, todo), 1):
            results.append(r)
            if i % 500 == 0:
                progress(f"rebase: {i}/{len(todo)}")
    kept = []
    report["dropped"] = []
    for fam, src, out, why_, notes in results:
        if out is None:
            fam_counts[fam][f"dropped:{why_}"] += 1
            report["dropped"].append({"id": src.get("id"), "family": fam, "why": why_})
            if len(report["dropped_examples"]) < 50:
                report["dropped_examples"].append({"id": src.get("id"), "family": fam, "why": why_, **notes})
            continue
        if notes.get("unchanged"):
            fam_counts[fam]["unchanged"] += 1
        if notes.get("renamed"):
            fam_counts[fam]["renamed"] += 1
        if notes.get("relabelled"):
            fam_counts[fam]["relabelled"] += 1
        if (out.get("meta") or {}).get("relabel_missed"):
            fam_counts[fam]["relabel_missed"] += 1
        kept.append(out)
    if check_leak:
        index = ExamIndex(why=why)
        train = [c for c in kept if not c.get("holdout")]
        train_kept, leak = filter_cases(train, index, family=lambda c: (c.get("meta") or {}).get("family") or "?")
        for d in leak["dropped"]:
            fam_counts[str(d.get("family"))][f"dropped:leak_rule{d['rule']}"] += 1
            report["dropped"].append({"id": d.get("id"), "family": d.get("family"), "why": f"leak_rule{d['rule']}"})
        report["leak"] = {k: leak[k] for k in ("by_family", "total")}
        kept_ids = {c["id"] for c in train_kept}
        kept = [c for c in kept if c.get("holdout") or c["id"] in kept_ids]
    loss_failed = []
    for fam, cnt in sorted(fam_counts.items()):
        n_in = cnt.get("in", 0)
        dropped = sum(v for k, v in cnt.items() if k.startswith("dropped:"))
        rec = dict(cnt)
        rec["kept"] = n_in - dropped
        rec["loss"] = round(dropped / n_in, 4) if n_in else 0.0
        report["families"][fam] = rec
        if gate and n_in >= MIN_FAMILY_FOR_GATE and dropped / n_in > MAX_FAMILY_LOSS:
            loss_failed.append(fam)
    report["loss_failed"] = loss_failed
    report["totals"] = {"in": len(todo), "kept": len(kept), "dropped": len(todo) - len(kept)}
    report["dropped_by_reason"] = dict(Counter(k.split(":", 1)[1] for cnt in fam_counts.values()
                                               for k, v in cnt.items() if k.startswith("dropped:") for _ in range(v)))
    return kept, report
