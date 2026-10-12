"""examrebase — the exam (and the generated hold-out) on another manifest
(DESIGN §1.3 arm D; docs/DRIFT.md).

The real-phrasing exam is recorded on one manifest (`fixtures/exam/
manifest-real.json`, bf788701) and `exam.load_set` refuses any other. The
drift check asks the SAME rows on the manifest the launcher deploys today:

  rebase_exam(rows, new_manifest, renames)   every row's expected keys (and alt sets) mapped through the
                                             static diff's renames, checked on the new manifest
                                             (exam.check_expectation); a row whose primary set names a key
                                             the new manifest lacks is STALE (returned apart, never asked);
                                             an alt set that went stale is dropped. `manifest_hash` becomes
                                             the new one, `meta.rebased_from` records the old one and the
                                             renames applied. Contexts are left alone (the caller re-derives
                                             them on the new manifest: exam.rederive_contexts).
  rebase_generated(cases, …)                 the same for generated cases (F0H, the probe): ops, oracle ops,
                                             keys renamed; stale ones dropped with a reason; re-measured
                                             through the engine on the new manifest when one is given
                                             (r4cases.remeasure)
  ManifestIndex / address_of                 the static diff's field addressing (`<group>.<name>.<key>`,
                                             `<group>.<name>.<list>[].<key>`, `<group>.<name>.<list>[]`;
                                             branding = `branding`), and the address an exam key (target,
                                             key, list) resolves to — the bridge between rows and the diff

Standard library only.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .manifest import Manifest

GROUPS = ("sections", "globalPanels", "pages", "details")


def _doc(m: Manifest | dict) -> dict:
    return m.doc if isinstance(m, Manifest) else m


def _manifest(m: Manifest | dict | str | Path) -> Manifest:
    if isinstance(m, Manifest):
        return m
    if isinstance(m, dict):
        return Manifest(m)
    return Manifest(json.loads(Path(m).read_text(encoding="utf-8")), Path(m))


class ManifestIndex:
    """Every container, field and list of a manifest by address."""

    def __init__(self, m: Manifest | dict):
        doc = _doc(m)
        self.doc = doc
        self.containers: dict[str, dict] = {}
        self.fields: dict[str, dict] = {}       # address → {container, list, key, spec}
        self.lists: dict[str, dict] = {}        # `<c>.<list>[]` → {container, list, spec}
        self.ops: set[str] = set()
        for grp in GROUPS:
            for name, c in (doc.get(grp) or {}).items():
                if isinstance(c, dict):
                    self._add(f"{grp}.{name}", c)
        if isinstance(doc.get("branding"), dict):
            self._add("branding", doc["branding"])

    def _add(self, caddr: str, c: dict) -> None:
        self.containers[caddr] = c
        for key, f in (c.get("fields") or {}).items():
            self.fields[f"{caddr}.{key}"] = {"container": caddr, "list": None, "key": key, "spec": f or {}}
        for lname, lst in (c.get("lists") or {}).items():
            lst = lst or {}
            self.lists[f"{caddr}.{lname}[]"] = {"container": caddr, "list": lname, "spec": lst}
            self.ops.update(str(o) for o in lst.get("ops") or [])
            for key, f in (lst.get("item") or {}).items():
                self.fields[f"{caddr}.{lname}[].{key}"] = {"container": caddr, "list": lname, "key": key, "spec": f or {}}

    def has(self, address: str) -> bool:
        return address in self.fields or address in self.lists or address in self.containers


def container_addresses(m: Manifest, target: dict | None) -> list[str]:
    t = target or {}
    kind = t.get("kind")
    if kind == "section":
        typ = m.type_of(str(t.get("sectionId") or "")) or str(t.get("sectionId") or "")
        return [f"sections.{typ}"]
    if kind == "global":
        if t.get("panel"):
            return [f"globalPanels.{t['panel']}"]
        return [f"globalPanels.{c.name}" for c in m.containers("global")]
    if kind == "branding":
        return ["branding"]
    if kind == "page":
        if t.get("page") == "seo":
            return ["details.SeoPanel"]
        return [f"pages.{t.get('page')}", "pages._common"]
    return []


def address_of(m: Manifest, idx: ManifestIndex, e: dict) -> tuple[str | None, bool]:
    """(address, found) of an exam key `{target, key, list}` on manifest `m`.
    `found` False = the manifest has no such field/list (the address is then
    the one it WOULD have, for reporting); None = not a manifest field (a
    page's `sections` list, the layout pseudo-keys)."""
    t = e.get("target") or {}
    key, lst = str(e.get("key") or ""), str(e.get("list") or "")
    if key.startswith("__"):
        return None, True
    if t.get("kind") == "page" and lst == "sections" and not key:
        return None, True
    if t.get("kind") == "global" and key.startswith("categories.") and key.count(".") >= 2:
        leaf = key.split(".", 2)[2]
        for a in (f"details.CategoryDetailsPanel.{leaf}", f"details.CategoryDetailsPanel.item.{leaf}"):
            if a in idx.fields:
                return a, True
        return f"details.CategoryDetailsPanel.{leaf}", False
    cands = container_addresses(m, t)
    first = None
    for c in cands:
        if lst:
            la = f"{c}.{lst}[]"
            if not key:
                first = first or la
                if la in idx.lists:
                    return la, True
                continue
            a = f"{la}.{key}"
            first = first or a
            if a in idx.fields:
                return a, True
            ref = ((idx.lists.get(la) or {}).get("spec") or {}).get("item_ref")
            if ref:
                panel = ref.split(".", 1)[1] if ref.startswith("details.") else ref
                for d in (f"details.{panel}.{key}", f"details.{panel}.item.{key}"):
                    if d in idx.fields:
                        return d, True
        else:
            a = f"{c}.{key}"
            first = first or a
            if a in idx.fields:
                return a, True
            cont = idx.containers.get(c) or {}
            mirror = cont.get("mirror") or {}
            if key in (mirror.get("keys") or []):
                src = str(mirror.get("from") or "slides[0]").split("[", 1)[0]
                ma = f"{c}.{src}[].{key}"
                if ma in idx.fields:
                    return ma, True
            # a nested write (`announcementBar.message` written as {announcementBar: {message}}) or a keyed field
            head = key.split(".", 1)[0]
            if head != key and f"{c}.{head}" in idx.fields:
                return f"{c}.{head}", True
    return first, False


# ── renames ──────────────────────────────────────────────────────────────────

def rename_map(renames: Iterable[dict] | None) -> dict[str, dict]:
    """{from address: rename} from the static diff's `renames`."""
    return {r["from"]: r for r in renames or [] if isinstance(r, dict) and r.get("from") and r.get("to")}


def _renamed_key(r: dict) -> str:
    return str(r.get("to_key") or r["to"].rsplit(".", 1)[-1])


def rename_entry(old_m: Manifest, old_idx: ManifestIndex, e: dict, rmap: dict[str, dict]) -> tuple[dict, dict | None]:
    """The exam key with a renamed field's new key (and the rename applied)."""
    if not rmap:
        return e, None
    addr, found = address_of(old_m, old_idx, e)
    r = rmap.get(addr or "")
    if r is None or not found:
        return e, None
    e2 = copy.deepcopy(e)
    old_key = str(e.get("key") or "")
    from_key = str(r.get("from_key") or r["from"].rsplit(".", 1)[-1])
    new_key = _renamed_key(r)
    if old_key == from_key:
        e2["key"] = new_key
    elif old_key.endswith("." + from_key):  # a category patch (`categories.<slug>.<leaf>`)
        e2["key"] = old_key[: -len(from_key)] + new_key
    else:
        return e, None
    return e2, r


def rebase_exam(rows: list[dict], new_manifest: Manifest | dict | str | Path, renames: Iterable[dict] | None = None, *,
                old_manifest: Manifest | dict | str | Path | None = None) -> tuple[list[dict], list[dict], dict]:
    """(rebased rows, stale rows, report). `old_manifest` is the rows' own
    (default: the exam's, fixtures/exam/manifest-real.json)."""
    from .exam import MANIFEST_FILE, check_expectation, expected_sets, is_out_of_scope

    new_m = _manifest(new_manifest)
    old_m = _manifest(old_manifest if old_manifest is not None else MANIFEST_FILE)
    old_idx = ManifestIndex(old_m)
    rmap = rename_map(renames)
    out, stale = [], []
    rep: dict[str, Any] = {"rows": len(rows), "kept": 0, "stale": 0, "renamed_rows": 0, "alt_dropped": 0,
                           "out_of_scope": 0, "stale_ids": [], "missing": Counter(), "from": old_m.hash, "to": new_m.hash}
    for row in rows:
        r = copy.deepcopy(row)
        meta = r.setdefault("meta", {})
        meta["rebased_from"] = {"manifest_hash": row.get("manifest_hash") or old_m.hash, "renames": []}
        r["manifest_hash"] = new_m.hash
        if is_out_of_scope(row):
            rep["out_of_scope"] += 1
            rep["kept"] += 1
            out.append(r)
            continue
        sets = expected_sets(row)
        new_sets: list[list[dict] | None] = []
        missing_primary: list[str] = []
        applied = []
        for i, s in enumerate(sets):
            ns, ok = [], True
            for e in s:
                e2, rn = rename_entry(old_m, old_idx, e, rmap)
                if rn is not None:
                    applied.append({"from": rn["from"], "to": rn["to"]})
                why = check_expectation(new_m, e2)
                if why is not None:
                    ok = False
                    if i == 0:
                        missing_primary.append(why)
                ns.append(e2)
            new_sets.append(ns if ok else None)
        if sets and new_sets[0] is None:
            meta["stale"] = {"missing": missing_primary}
            stale.append(r)
            rep["stale"] += 1
            rep["stale_ids"].append(f"{row.get('set')}:{row.get('id')}")
            rep["missing"].update(missing_primary)
            continue
        exp = r.setdefault("expected", {})
        if sets:
            exp["keys"] = new_sets[0]
            alts = [s for s in new_sets[1:] if s is not None]
            rep["alt_dropped"] += sum(1 for s in new_sets[1:] if s is None)
            exp["alt"] = alts
        if applied:
            uniq = {json.dumps(a, sort_keys=True): a for a in applied}
            meta["rebased_from"]["renames"] = list(uniq.values())
            rep["renamed_rows"] += 1
            renamed_keys = {a["from"].rsplit(".", 1)[-1]: a["to"].rsplit(".", 1)[-1] for a in uniq.values()}
            if isinstance(r.get("fields"), list):
                r["fields"] = [renamed_keys.get(f, f) if isinstance(f, str) else f for f in r["fields"]]
        rep["kept"] += 1
        out.append(r)
    rep["missing"] = dict(rep["missing"].most_common(50))
    return out, stale, rep


# ── generated cases (F0H, probes) ────────────────────────────────────────────

def _rename_values(values: Any, from_key: str, to_key: str) -> Any:
    if not isinstance(values, dict):
        return values
    out = {}
    for k, v in values.items():
        out[to_key if k == from_key else k] = v
    return out


def rename_op(old_m: Manifest, old_idx: ManifestIndex, op: dict, rmap: dict[str, dict]) -> tuple[dict, list[dict]]:
    """A browser op with renamed keys (set_fields values, edit_list item,
    reset / restore_default keys)."""
    from .exam import touches_of

    if not rmap:
        return op, []
    o = copy.deepcopy(op)
    applied = []
    for t in touches_of(op):
        e = {"target": t["target"], "key": t["key"], "list": t["list"]}
        e2, rn = rename_entry(old_m, old_idx, e, rmap)
        if rn is None:
            continue
        fk, tk = t["key"], e2["key"]
        applied.append({"from": rn["from"], "to": rn["to"]})
        if o.get("op") == "set_fields":
            o["values"] = _rename_values(o.get("values"), fk, tk)
            if isinstance(o.get("assets"), dict):
                o["assets"] = _rename_values(o["assets"], fk, tk)
        elif o.get("op") == "edit_list":
            o["item"] = _rename_values(o.get("item"), fk, tk)
        elif o.get("op") in ("reset", "restore_default"):
            o["keys"] = [tk if k == fk else k for k in o.get("keys") or []]
    return o, applied


def ops_valid(m: Manifest, ops: Iterable[dict]) -> list[str]:
    """Why each written key of `ops` is not on `m` (empty = all present)."""
    from .exam import check_expectation, touches_of

    out = []
    for op in ops or []:
        if op.get("op") in ("pick_asset", "set_page_layout", "reset_pinned_section"):
            continue
        for t in touches_of(op):
            if not t["key"] and not t["list"]:
                continue
            # a nested write ({announcementBar: {message}}) touches the parent and the leaf: either resolving is enough
            why = check_expectation(m, {"target": t["target"], "key": t["key"], "list": t["list"]})
            if why is not None and not any(
                    check_expectation(m, {"target": t["target"], "key": k, "list": t["list"]}) is None
                    for k in _parents(t["key"])):
                out.append(why)
    return out


def _parents(key: str) -> list[str]:
    parts = key.split(".")
    return [".".join(parts[:i]) for i in range(len(parts) - 1, 0, -1)]


def rebase_generated(cases: list[dict], new_manifest: Manifest | dict | str | Path, renames: Iterable[dict] | None = None, *,
                     old_manifest: Manifest | dict | str | Path, engine: Any = None, extras: dict | None = None
                     ) -> tuple[list[dict], dict]:
    """(cases on the new manifest, report{kept, dropped: {reason: n}, renamed}).
    With `engine` (a headless engine on the NEW manifest), every case is
    re-measured (r4cases.remeasure: contexts re-derived, the exact diff
    re-measured, oracle ops required to apply)."""
    from .r4cases import remeasure
    from .state import load_context_fixture

    new_m = _manifest(new_manifest)
    old_m = _manifest(old_manifest)
    old_idx = ManifestIndex(old_m)
    rmap = rename_map(renames)
    extras = extras if extras is not None else load_context_fixture()
    out: list[dict] = []
    dropped: Counter = Counter()
    renamed = 0
    for c0 in cases:
        c = copy.deepcopy(c0)
        e = c.setdefault("expected", {})
        applied: list[dict] = []
        new_ops = []
        for op in e.get("ops") or []:
            o, a = rename_op(old_m, old_idx, op, rmap)
            new_ops.append(o)
            applied += a
        e["ops"] = new_ops
        meta = c.setdefault("meta", {})
        if meta.get("oracle_ops"):
            oo = []
            for op in meta["oracle_ops"]:
                o, a = rename_op(old_m, old_idx, op, rmap)
                oo.append(o)
                applied += a
            meta["oracle_ops"] = oo
        for h in c.get("history") or []:
            if h.get("ops"):
                ho = []
                for op in h["ops"]:
                    o, a = rename_op(old_m, old_idx, op, rmap)
                    ho.append(o)
                    applied += a
                h["ops"] = ho
        if e.get("keys"):
            ks = []
            for k in e["keys"]:
                k2, rn = rename_entry(old_m, old_idx, k, rmap)
                if rn is not None:
                    applied.append({"from": rn["from"], "to": rn["to"]})
                ks.append(k2)
            e["keys"] = ks
        why = ops_valid(new_m, list(e.get("ops") or []) + list(meta.get("oracle_ops") or []))
        if why:
            dropped["key_removed"] += 1
            continue
        c["manifest_hash"] = new_m.hash
        meta["rebased_from"] = {"manifest_hash": c0.get("manifest_hash"), "renames": applied}
        if applied:
            renamed += 1
        if engine is not None:
            c.setdefault("score", {"mode": meta.get("scoring") or ("exact" if e.get("diff") else "keys")})
            reason = remeasure(c, engine, new_m, extras)
            if reason:
                dropped[reason] += 1
                continue
        out.append(c)
    return out, {"cases": len(cases), "kept": len(out), "dropped": dict(dropped), "renamed": renamed,
                 "remeasured": engine is not None}
