"""drift — has the launcher drifted far enough from what blueberry-muffin
was trained on to dodge the model? (DESIGN §1–2, docs/DRIFT.md)

    python -m souschef_lab.pipeline drift poll       every 15 min: the deployed manifest; on a change, the static diff
    python -m souschef_lab.pipeline drift static     the static diff and its impact (no model)
    python -m souschef_lab.pipeline drift check      arm D (+ a confirmation replicate when needed) and the new-key probe
    python -m souschef_lab.pipeline drift weekly     arm T (baseline refresh) + arm D + probe + F0H on both arms
    python -m souschef_lab.pipeline drift calibrate  arm T × 3 + F0H: the baseline of a model image
    (`python -m souschef_lab.drift …` is the same entry)

What is compared (DESIGN §1.2):
  training  the manifest the live model trained on (state/model.json → training_manifest; r4 = bf788701)
  deployed  GET /v1/manifests/deployed?line=int (the launcher's, never a lab registration)
  ack       the deployed manifest at the last Skip (state/ack.json): drift is then measured BEYOND it

Static classes: R0 noise (source lines, key order), R1 wording (label, help, enum labels — the model reads
them fresh every turn), R2 soft schema (options, defaults, ranges, gates, list bounds), R3 hard schema (a field,
list or container added, removed, renamed or retyped; a new op). Renames: a removed and an added field of the
same container and list, same type, label equal or token-Jaccard ≥ 0.8, or the same non-empty help.

Behaviour (DESIGN §1.3): the real-phrasing exam is asked through the DEPLOYED service binary (harness.py) whose
only route is the GPU model service (no Claude fallback), on two manifests:
  arm T  the 252 rows on the training manifest, contexts re-derived (the baseline, baseline.py)
  arm D  the same rows rebased onto the deployed manifest (examrebase.py), contexts re-derived on it; rows whose
         keys are gone are STALE (counted, never asked)
  arm P  up to 4 generated cases per new key (cases.py), each asked twice: a key passing < 50 % is DODGED
Verdicts (lab/pipeline/thresholds.json; a bucket's config/thresholds.json wins): ok | notice | retrain — the
table is `evaluate`'s docstring.

Where turns go: the harness on the eval GPU service (`--eval-url`), or, with DRY=1 / --dry, the lab's fake
service in process: `--dry-model stale` (default) answers every arm as a model trained on the TRAINING manifest
would (arm D's renamed keys are refused by the engine, new keys are not found), `--dry-model oracle` answers
every row right. `--fake-engine` checks keys against the manifest instead of running node (tests).

Bucket layout: DESIGN I2 (state/deployed.json, state/model.json, state/ack.json, state/heartbeat.json,
manifests/<hash>.json, baselines/<digest>/baseline.json, drift/<run>/{result,static,report,ack-material}.json,
drift/<run>/exam/*.jsonl). Exit 0 = the stage finished (the outcome is drift/<run>/result.json), 1 = failure,
2 = configuration. Standard library only (+ node for the engine, the service binaries for a real run).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import LAB_DIR
from . import baseline as BL
from . import gcsio
from . import pipeline_common as PC
from .examrebase import ManifestIndex, address_of, ops_valid, rebase_exam, rebase_generated
from .manifest import Manifest

R0, R1, R2, R3 = "R0", "R1", "R2", "R3"
CLASSES = (R0, R1, R2, R3)
NOISE_ATTRS = frozenset({"source"})
WORDING_ATTRS = frozenset({"label", "help", "description", "placeholder", "enum_labels", "group", "group_title", "title",
                           "hint", "tooltip", "section_label"})
HARD_FIELD_ATTRS = frozenset({"type"})
HARD_LIST_ATTRS = frozenset({"item_ref"})
TOP_NOISE = frozenset({"generated_from", "coverage"})
PASS = frozenset(("pass", "clarify_ok", "unsupported_ok"))
EXAM_SETS = ("paraphrase", "holdout", "real", "out_of_scope", "out_of_scope_holdout", "followup")
PROBE_KINDS_STATE = ("set_field", "set_enum", "set_range", "toggle", "image", "reset")
PROBE_KINDS_LIST = ("list_add", "list_update")
ROUTE = "current"
STAGE = "drift"


# ═════════════════════════════════════════ static diff ═════════════════════════════════════════

def _short(v: Any, n: int = 200) -> Any:
    if isinstance(v, str):
        return v if len(v) <= n else v[: n - 1] + "…"
    if isinstance(v, (dict, list)):
        s = json.dumps(v, ensure_ascii=False, sort_keys=True)
        return v if len(s) <= n else s[: n - 1] + "…"
    return v


def attr_class(attr: str) -> str:
    if attr in NOISE_ATTRS:
        return R0
    if attr in WORDING_ATTRS:
        return R1
    if attr in HARD_FIELD_ATTRS:
        return R3
    return R2


def _label_tokens(s: Any) -> frozenset[str]:
    from .leak import tokens
    return tokens(str(s or ""))


def rename_score(old: dict, new: dict) -> float:
    """1.0 same label; token Jaccard of the labels; 0.9 for the same
    non-empty help (DESIGN §1.2)."""
    lo, ln = str(old.get("label") or ""), str(new.get("label") or "")
    if lo and lo == ln:
        return 1.0
    a, b = _label_tokens(lo), _label_tokens(ln)
    j = len(a & b) / len(a | b) if (a | b) else 0.0
    ho, hn = str(old.get("help") or ""), str(new.get("help") or "")
    if ho and ho == hn:
        j = max(j, 0.9)
    return round(j, 3)


def find_renames(removed: Iterable[str], added: Iterable[str], oi: ManifestIndex, ni: ManifestIndex,
                 jaccard: float) -> list[dict]:
    by_slot: dict[tuple, list[str]] = defaultdict(list)
    for a in added:
        f = ni.fields[a]
        by_slot[(f["container"], f["list"], str(f["spec"].get("type") or ""))].append(a)
    pairs = []
    for r in removed:
        f = oi.fields[r]
        for a in by_slot.get((f["container"], f["list"], str(f["spec"].get("type") or "")), []):
            s = rename_score(f["spec"], ni.fields[a]["spec"])
            if s >= jaccard:
                pairs.append((s, r, a))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    used_r, used_a, out = set(), set(), []
    for s, r, a in pairs:
        if r in used_r or a in used_a:
            continue
        used_r.add(r)
        used_a.add(a)
        fo, fn = oi.fields[r], ni.fields[a]
        out.append({"from": r, "to": a, "score": s, "container": fo["container"], "list": fo["list"],
                    "from_key": fo["key"], "to_key": fn["key"], "type": str(fn["spec"].get("type") or "")})
    return out


def static_diff(old: dict, new: dict, *, jaccard: float = 0.8, cap: int = 500) -> dict:
    """The classified difference training → deployed (DESIGN §1.2)."""
    oi, ni = ManifestIndex(old), ManifestIndex(new)
    ch: dict[str, list[dict]] = {c: [] for c in CLASSES}

    def add(cls: str, change: str, address: str, old_v: Any = None, new_v: Any = None, **kw: Any) -> None:
        ch[cls].append({"class": cls, "change": change, "address": address, "old": _short(old_v), "new": _short(new_v), **kw})

    added_c = sorted(set(ni.containers) - set(oi.containers))
    removed_c = sorted(set(oi.containers) - set(ni.containers))
    for c in added_c:
        add(R3, "container_added", c)
    for c in removed_c:
        add(R3, "container_removed", c)
    for c in sorted(set(oi.containers) & set(ni.containers)):
        co, cn = oi.containers[c], ni.containers[c]
        for attr in sorted((set(co) | set(cn)) - {"fields", "lists"}):
            if co.get(attr) != cn.get(attr):
                add(attr_class(attr) if attr in NOISE_ATTRS | WORDING_ATTRS else R2, f"container.{attr}", c, co.get(attr), cn.get(attr))
    # lists
    added_l = sorted(set(ni.lists) - set(oi.lists))
    removed_l = sorted(set(oi.lists) - set(ni.lists))
    for a in added_l:
        add(R3, "list_added", a, container_new=ni.lists[a]["container"] in added_c)
    for a in removed_l:
        add(R3, "list_removed", a, container_gone=oi.lists[a]["container"] in removed_c)
    new_ops = sorted(ni.ops - oi.ops)
    for a in sorted(set(oi.lists) & set(ni.lists)):
        lo, ln = oi.lists[a]["spec"], ni.lists[a]["spec"]
        for attr in sorted((set(lo) | set(ln)) - {"item"}):
            if lo.get(attr) == ln.get(attr):
                continue
            if attr == "ops":
                fresh = sorted(set(ln.get("ops") or []) - set(lo.get("ops") or []))
                if any(o in new_ops for o in fresh):
                    add(R3, "list.new_op", a, lo.get("ops"), ln.get("ops"))
                else:
                    add(R2, "list.ops", a, lo.get("ops"), ln.get("ops"))
            elif attr in HARD_LIST_ATTRS:
                add(R3, f"list.{attr}", a, lo.get(attr), ln.get(attr))
            else:
                add(attr_class(attr) if attr in NOISE_ATTRS | WORDING_ATTRS else R2, f"list.{attr}", a, lo.get(attr), ln.get(attr))
    # fields
    added_f = sorted(set(ni.fields) - set(oi.fields))
    removed_f = sorted(set(oi.fields) - set(ni.fields))
    renames = find_renames(removed_f, added_f, oi, ni, jaccard)
    ren_from = {r["from"] for r in renames}
    ren_to = {r["to"] for r in renames}
    for r in renames:
        add(R3, "field_renamed", r["to"], r["from"], r["to"], score=r["score"])
    for a in added_f:
        if a not in ren_to:
            f = ni.fields[a]
            add(R3, "field_added", a, None, f["spec"].get("type"), container_new=f["container"] in added_c,
                list_new=bool(f["list"]) and f"{f['container']}.{f['list']}[]" in added_l)
    for a in removed_f:
        if a not in ren_from:
            f = oi.fields[a]
            add(R3, "field_removed", a, f["spec"].get("type"), None, container_gone=f["container"] in removed_c)
    retyped = []
    for a in sorted(set(oi.fields) & set(ni.fields)):
        fo, fn = oi.fields[a]["spec"], ni.fields[a]["spec"]
        if fo == fn:
            continue
        for attr in sorted(set(fo) | set(fn)):
            if fo.get(attr) == fn.get(attr):
                continue
            cls = attr_class(attr)
            if attr == "type":
                retyped.append(a)
                add(R3, "field_retyped", a, fo.get("type"), fn.get("type"))
            else:
                add(cls, f"field.{attr}", a, fo.get(attr), fn.get(attr))
    for attr in sorted((set(old) | set(new)) - {"sections", "globalPanels", "pages", "details", "branding"}):
        if old.get(attr) != new.get(attr):
            add(R0 if attr in TOP_NOISE else R2, f"manifest.{attr}", attr, old.get(attr) if attr not in TOP_NOISE else None,
                new.get(attr) if attr not in TOP_NOISE else None)
    counts = {c: len(ch[c]) for c in CLASSES}
    by_group: dict[str, Counter] = defaultdict(Counter)
    by_change: dict[str, Counter] = defaultdict(Counter)
    for c in CLASSES:
        for e in ch[c]:
            by_group[e["address"].split(".", 1)[0]][c] += 1
            by_change[c][e["change"]] += 1
    return {
        "from": Manifest(old).hash, "to": Manifest(new).hash,
        "counts": counts, "by_group": {g: dict(v) for g, v in sorted(by_group.items())},
        "by_change": {c: dict(v.most_common()) for c, v in by_change.items()},
        "renames": renames,
        "fields_added": [a for a in added_f if a not in ren_to], "fields_removed": [a for a in removed_f if a not in ren_from],
        "fields_retyped": retyped, "lists_added": added_l, "lists_removed": removed_l,
        "containers_added": added_c, "containers_removed": removed_c, "new_ops": new_ops,
        "changes": {c: ch[c][:cap] for c in CLASSES}, "truncated": {c: max(0, len(ch[c]) - cap) for c in CLASSES},
        "identical": all(v == 0 for k, v in counts.items() if k != R0),
    }


def compact_static(sd: dict) -> dict:
    return {k: sd.get(k) for k in ("from", "to", "counts", "by_change", "renames", "fields_added", "fields_removed",
                                   "fields_retyped", "lists_added", "lists_removed", "containers_added", "containers_removed",
                                   "new_ops", "identical")}


# ═════════════════════════════════════════ impact ═════════════════════════════════════════

def writable(spec: dict) -> bool:
    return str(spec.get("type") or "") != "json" and not spec.get("derived_from")


def new_key_targets(sd: dict, ni: ManifestIndex, exam_addresses: set[str]) -> list[str]:
    """R3-added, R3-renamed and R3-retyped fields (writable) and added lists with
    no exam row writing them: the probe's targets. A retyped key keeps its name,
    but the values the model learnt for it may now be refused."""
    out = []
    for a in (list(sd.get("fields_added") or []) + [r["to"] for r in sd.get("renames") or []]
              + list(sd.get("fields_retyped") or [])):
        f = ni.fields.get(a)
        if f and writable(f["spec"]) and a not in exam_addresses and a not in out:
            out.append(a)
    for a in sd.get("lists_added") or []:
        if a not in exam_addresses and a not in out:
            out.append(a)
    return out


def exam_addresses(rows: list[dict], m: Manifest) -> set[str]:
    from .exam import expected_sets
    idx = ManifestIndex(m)
    out = set()
    for r in rows:
        for s in expected_sets(r):
            for e in s:
                a, found = address_of(m, idx, e)
                if a and found:
                    out.add(a)
    return out


def exam_manifest() -> Manifest:
    from .exam import MANIFEST_FILE
    return Manifest(json.loads(MANIFEST_FILE.read_text(encoding="utf-8")))


def exam_on(rows: list[dict], target: Manifest, *, jaccard: float = 0.8) -> tuple[list[dict], list[dict], dict]:
    """The exam rows (which live on the exam's manifest, bf788701) rebased onto `target`: as they are when the
    target IS the exam's manifest (r4's arm T), else through rebase_exam with the exam → target renames. A model
    trained on another manifest (every model after r4) is examined on its own manifest this way."""
    em = exam_manifest()
    if target.hash == em.hash:
        return copy.deepcopy(rows), [], {"rows": len(rows), "kept": len(rows), "stale": 0, "renamed_rows": 0, "alt_dropped": 0,
                                         "stale_ids": [], "missing": Counter(), "from": em.hash, "to": em.hash}
    renames = static_diff(em.doc, target.doc, jaccard=jaccard).get("renames")
    return rebase_exam(rows, target, renames, old_manifest=em)


def _case_ops(c: dict) -> list[dict]:
    """A generated case's ops as rebase_generated checks them (expected ops + oracle ops)."""
    return list((c.get("expected") or {}).get("ops") or []) + list((c.get("meta") or {}).get("oracle_ops") or [])


def impact(sd: dict, old_m: Manifest, new_m: Manifest, exam_rows: list[dict], training_cases: list[dict] | None = None,
           engine_status: str = "unknown") -> dict:
    rebased, stale, rep = exam_on(exam_rows, new_m)
    covered = exam_addresses(rebased, new_m)
    ni = ManifestIndex(new_m)
    nk = new_key_targets(sd, ni, covered)
    out: dict[str, Any] = {"exam_rows": len(exam_rows), "stale_exam_rows": rep["stale_ids"],
                           "stale_exam_pct": round(100.0 * rep["stale"] / max(1, len(exam_rows)), 2),
                           "renamed_exam_rows": rep["renamed_rows"], "exam_alt_dropped": rep["alt_dropped"],
                           "stale_exam_missing": rep["missing"], "new_keys": nk, "engine_status": engine_status}
    if training_cases is not None:
        trainable = [c for c in training_cases if not c.get("holdout")]
        hashes = Counter(str(c.get("manifest_hash")) for c in trainable)
        on_old = [c for c in trainable if c.get("manifest_hash") in (None, old_m.hash)]
        # only rows the old manifest itself accepts: a row whose ops never were
        # manifest keys (r4's page-target cases) is not made stale by the change
        valid_old = [c for c in on_old if not ops_valid(old_m, _case_ops(c))]
        kept, trep = rebase_generated(valid_old, new_m, sd.get("renames"), old_manifest=old_m, engine=None)
        out.update({"training_rows": len(trainable), "training_rows_checked": len(valid_old),
                    "training_rows_invalid_on_old": len(on_old) - len(valid_old),
                    "training_other_manifest": {h: n for h, n in hashes.items() if h != old_m.hash},
                    "stale_training_rows": trep["dropped"].get("key_removed", 0),
                    "stale_training_pct": round(100.0 * trep["dropped"].get("key_removed", 0) / max(1, len(valid_old)), 2)})
    else:
        out.update({"training_rows": None, "stale_training_rows": None, "stale_training_pct": None})
    return out


# ═════════════════════════════════════════ probe (arm P) ═════════════════════════════════════════

def probe_rows(new_m: Manifest, keys: list[str], *, state: dict, per_key: int = 4, max_keys: int = 40, seed: int = 0,
               engine: Any = None) -> tuple[list[dict], dict[str, int]]:
    """Up to `per_key` exam-style rows per new key from the deterministic
    generator: the vintage state first (set/toggle/enum/reset kinds), the
    default home rows for what it lacks (list kinds, section types not on the
    vintage home). With `engine`, cases whose ops the engine refuses are
    dropped. Rows are scored by keys (exam.score_turn)."""
    from .cases import Generator, measure_one
    from .exam import touches_of

    keys = list(keys)[:max_keys]
    want = set(keys)
    if not want:
        return [], {}
    idx = ManifestIndex(new_m)
    cands: dict[str, list] = defaultdict(list)
    seen_ids: set[str] = set()

    def consider(cases) -> None:
        for c in cases:
            addrs = set()
            for op in c.expected.get("ops") or []:
                for t in touches_of(op):
                    a, found = address_of(new_m, idx, {"target": t["target"], "key": t["key"], "list": t["list"]})
                    if a and found:
                        addrs.add(a)
                    if t["list"]:  # a list op also probes the list itself
                        la = a.split("[].", 1)[0] + "[]" if a and "[]." in a else None
                        if la:
                            addrs.add(la)
            for a in sorted(addrs & want):
                if (a, c.id) not in seen_ids and len(cands[a]) < 4 * per_key:
                    seen_ids.add((a, c.id))
                    cands[a].append(c)

    try:
        consider(Generator(new_m, seed=seed, template="vintage", base_state=state).all_cases(PROBE_KINDS_STATE))
    except (IndexError, KeyError, ValueError):
        pass
    if any(len(cands[k]) < per_key for k in keys):
        g2 = Generator(new_m, seed=seed, template="vintage")
        for kinds in (PROBE_KINDS_STATE, PROBE_KINDS_LIST):
            try:
                consider(g2.all_cases(kinds))
            except (IndexError, KeyError, ValueError):
                continue
    picked: dict[str, list] = {}
    for k in keys:
        by_kind: dict[str, list] = defaultdict(list)
        for c in cands[k]:
            by_kind[c.kind].append(c)
        chosen: list = []
        while len(chosen) < per_key and any(by_kind.values()):  # round-robin over kinds: set, toggle, reset …
            for kind in sorted(by_kind):
                if by_kind[kind] and len(chosen) < per_key:
                    c = by_kind[kind].pop(0)
                    if engine is not None and not measure_one(c, engine)[0]:
                        continue
                    chosen.append(c)
        picked[k] = chosen
    rows, coverage = [], {}
    for k in keys:
        coverage[k] = len(picked[k])
        for i, c in enumerate(picked[k]):
            ek = []
            for op in c.expected.get("ops") or []:
                if op.get("op") == "pick_asset":
                    continue
                for t in touches_of(op):
                    e = {"target": t["target"], "key": t["key"], "list": t["list"]}
                    if e not in ek:
                        ek.append(e)
            rid = f"p{abs(hash((k, c.id))) % 10**8:08d}-{i}"
            rows.append({"id": rid, "set": "probe", "kind": "phrasing", "target": c.target, "fields": c.fields,
                         "site_state": None, "context": c.context, "request": c.request, "history": [],
                         "expected": {"ops": [], "diff": [], "keys": ek, "alt": [], "out_of_scope": None},
                         "holdout": True, "manifest_hash": new_m.hash, "template": "vintage",
                         "meta": {"class": "probe", "probe_key": k, "kind": c.kind, "page": (c.context or {}).get("page"),
                                  "source_case": c.id, "generator_ops": c.expected.get("ops")}})
    return rows, coverage


# ═════════════════════════════════════════ running an arm ═════════════════════════════════════════

class ManifestEngine:
    """The DRY stand-in for headless.mjs (`--fake-engine`): an op is accepted
    when every key it writes is on the manifest (examrebase.ops_valid);
    accepted writes land under state.values like the tests' fake engine."""

    fingerprint = None

    def __init__(self, m: Manifest):
        self.m = m

    def __call__(self, state: dict, ops: list[dict], ctx: dict) -> dict:
        state = copy.deepcopy(state or {})
        results = []
        for op in ops:
            if op.get("op") == "pick_asset":
                results.append({"op": "pick_asset", "ok": True, "warnings": [], "written": []})
                continue
            why = ops_valid(self.m, [op])
            if why:
                results.append({"op": op.get("op"), "ok": False, "warnings": [], "written": [],
                                "error": {"code": "E_KEY", "message": why[0][:200], "key": None}})
                continue
            written = []
            vals = op.get("values") if op.get("op") == "set_fields" else op.get("item")
            for k, v in (vals.items() if isinstance(vals, dict) else []):
                state.setdefault("values", {})[k] = v
                written.append(f"values.{k}")
            results.append({"op": op.get("op"), "ok": True, "warnings": [], "written": written})
        state["_version"] = int(state.get("_version", 0)) + 1
        return {"ok": all(r["ok"] for r in results), "results": results, "changes": [], "state": state}

    run = __call__

    def close(self) -> None:
        pass


def home_context(rows: list[dict]) -> dict | None:
    return next((c.get("context") for c in rows if (c.get("context") or {}).get("page") == "home"), None)


def rederive(rows: list[dict], manifest_path: Path, home: dict | None) -> int:
    """exam.rederive_contexts with the vintage state's home given (a subset
    of rows may hold no home row), on any manifest."""
    from .exam import vintage_state
    from .state import context_of, load_context_fixture

    state = vintage_state(manifest_path, home)
    m = Manifest(json.loads(Path(manifest_path).read_text(encoding="utf-8")), Path(manifest_path))
    extras = load_context_fixture()
    n = 0
    for c in rows:
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


def row_key(row: dict) -> str:
    return f"{row.get('set') or 'x'}:{row.get('id')}"


def ghost_writes(row: dict, m: Manifest) -> int:
    """Refused ops that write a key the manifest does not have."""
    from .exam import accepted_ops
    _, refused = accepted_ops(row)
    return sum(1 for op in refused if ops_valid(m, [op]))


def trim(row: dict, case: dict, m: Manifest) -> dict:
    return {"key": row_key(case), "id": case.get("id"), "set": case.get("set"), "status": row.get("status"),
            "code": row.get("code"), "verdict": row.get("verdict"), "pass": int(row.get("status") == "ok" and row.get("verdict") in PASS),
            "reasons": (row.get("reasons") or [])[:6], "written": (row.get("written") or [])[:10], "refused": row.get("refused"),
            "ghost": ghost_writes(row, m) if row.get("status") == "ok" else 0, "latency_ms": row.get("latency_ms"),
            "escalated": bool(row.get("escalated")), "usage": row.get("usage"), "probe_key": (case.get("meta") or {}).get("probe_key"),
            "history_turns": len(row.get("history_turns") or [])}


class Target:
    """Where an arm's turns go: the harness (one local service on the eval GPU
    service for every arm) or, DRY, a fake service per arm."""

    def __init__(self, *, dry: bool, dry_model: str = "stale", harness=None, workers: int = 2, fake_engine: bool = False,
                 engine_path: str | None = None, progress=print):
        self.dry, self.dry_model, self.harness = dry, dry_model, harness
        self.workers = max(1, int(workers))
        self.fake_engine = fake_engine
        self.engine_path = engine_path
        self.progress = progress

    def engine_for(self, manifest_path: Path, m: Manifest, size: int):
        if self.fake_engine:
            return ManifestEngine(m), None
        from .collect import DictEngine
        from .engine import PersistentEngine
        pe = PersistentEngine(self.engine_path, manifest_path=manifest_path, size=size)
        return DictEngine(pe), pe

    @contextmanager
    def arm(self, name: str, rows: list[dict], m: Manifest, *, script_rows: list[dict] | None = None,
            script_manifest: Manifest | None = None, probe: bool = False) -> Iterator[tuple[str, str, str]]:
        """(url, key, route) for an arm; DRY scripts every row's request."""
        if not self.dry:
            yield self.harness.url, self.harness.key, ROUTE
            return
        from .exam import oracle_script
        if str(LAB_DIR) not in sys.path:
            sys.path.insert(0, str(LAB_DIR))
        from tests import fake_service as FS  # the tests' fake: a DRY run's stand-in, never imported by a real run
        svc = FS.FakeService(manifest=m.doc, key="drift-dry")
        svc.routes = [{"name": ROUTE, "kind": "messages-compatible", "fields_context": False, "fallback": None}]
        sm = script_manifest or m
        by_id = {r["id"]: r for r in (script_rows or rows)}
        for r in rows:
            if probe and self.dry_model == "stale":
                script = [[("text.delta", {"text": "I couldn't find a setting like that in the editor."}),
                           ("turn.done", {"stop": "end_turn", "usage": {"input_tokens": 0, "output_tokens": 0, "calls": 1}})]]
            else:
                src = by_id.get(r["id"], r) if self.dry_model == "stale" else r
                script = oracle_script(src, sm if self.dry_model == "stale" else m)
            svc.scripts[str(r.get("request") or "")] = script
        svc.start()
        try:
            yield svc.url, "drift-dry", ROUTE
        finally:
            svc.stop()

    def run_rows(self, arm_name: str, rows: list[dict], manifest_path: Path, m: Manifest, *, site_prefix: str,
                 sample: int = 1, script_rows: list[dict] | None = None, script_manifest: Manifest | None = None,
                 probe: bool = False, home: dict | None = None) -> list[dict]:
        """Every row once (`sample` = the replicate number) → trimmed rows."""
        from . import collect as C
        from .exam import run_row, vintage_state

        if not rows:
            return []
        state = vintage_state(manifest_path, home or home_context(rows))
        assets = C.load_assets(C.DEFAULT_ASSETS)
        out: list[dict] = []
        lock = threading.Lock()
        pending = list(reversed(rows))
        errors: list[str] = []
        t0 = time.monotonic()
        workers = min(self.workers, len(rows))
        with self.arm(arm_name, rows, m, script_rows=script_rows, script_manifest=script_manifest, probe=probe) as (url, key, route):
            engine, closer = self.engine_for(manifest_path, m, workers)
            try:
                client0, _ = C.connect(url, key, manifest_path, rows, sse=False, timeout_s=300, build_id="lab-drift")
                fields, rec = C.resolve_fields("auto", manifest_path, client0, route)

                def worker() -> None:
                    client = C.Client(url, key, timeout_s=300, build_id="lab-drift")
                    while True:
                        with lock:
                            if not pending or errors:
                                return
                            case = pending.pop()
                        try:
                            row = run_row(case, client, engine, state=state, route=route, sample=sample, assets=assets,
                                          fields=fields, history_mode="live", site_prefix=site_prefix)
                        except C.FieldsContextMismatch as e:
                            with lock:
                                errors.append(str(e))
                            return
                        except Exception as e:  # noqa: BLE001 — one row's lab bug must not lose the arm
                            row = {"status": "error", "code": "lab_exception", "error": f"{type(e).__name__}: {e}"[:500]}
                        t = trim(row, case, m)
                        with lock:
                            out.append(t)
                            n = len(out)
                        if n % 25 == 0 or n == len(rows):
                            self.progress(f"  {arm_name}#{sample}: {n}/{len(rows)} rows, {sum(x['pass'] for x in out)} pass, "
                                          f"{time.monotonic() - t0:.0f}s")

                threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
                for th in threads:
                    th.start()
                for th in threads:
                    th.join()
            finally:
                if closer is not None:
                    closer.close()
        if errors:
            raise PC.PipelineError(f"{arm_name}: {errors[0]}")
        errs = [r for r in out if r["status"] != "ok"]
        if len(errs) > max(3, len(out) // 5):
            codes = Counter(r.get("code") for r in errs)
            raise PC.PipelineError(f"{arm_name}#{sample}: {len(errs)}/{len(out)} rows errored ({dict(codes.most_common(4))})")
        return out


def results_of(rows: list[dict]) -> dict[str, int]:
    """{row key: 0/1}; rows that errored are left out (not a model verdict)."""
    return {r["key"]: int(r["pass"]) for r in rows if r.get("status") == "ok"}


def probe_scores(rows_by_rep: list[list[dict]], keys: list[str], th: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for k in keys:
        rs = [r for rep in rows_by_rep for r in rep if r.get("probe_key") == k and r.get("status") == "ok"]
        n, p = len(rs), sum(r["pass"] for r in rs)
        out[k] = {"n": n, "pass": p, "rate": round(p / n, 3) if n else None,
                  "dodged": bool(n) and (p / n) < float(th.get("dodged_pass_rate", 0.5))}
    return out


# ═════════════════════════════════════════ verdict ═════════════════════════════════════════

def evaluate(*, cmp: dict | None, static: dict, imp: dict, probe: dict[str, dict] | None, ghost_d: float | None,
             ghost_t: float | None, age_days: float | None, th: dict, confirmed: bool, calibrated: bool,
             ack: dict | None = None, static_since_ack: dict | None = None) -> dict:
    """The verdict (DESIGN §1.4). Rules — any fires:

      notice   Δ_total > 2 σ_Δ; any R3 change; a stale engine; 1 dodged key
      retrain  (a) Δ_total > 3 σ_Δ AND ≥ 4 rows — confirmed by a 2nd arm-D replicate
               (b) a set with n ≥ 12: Δ_set > 3 σ_Δ,set AND ≥ 15 % points — confirmed
               (c) out_of_scope + out_of_scope_holdout drop ≥ 3 rows — confirmed
               (d) ≥ 2 dodged keys, or every probed key of a new container dodged
               (e) stale training rows ≥ 3 % of the live model's dataset, or stale exam rows ≥ 5 %
               (f) ghost writes on arm D ≥ 3 more than on arm T
               (g) ≥ 90 days since training AND any R2/R3 change

    (c) also needs the drop beyond `retrain_sigma` σ_Δ of the out-of-scope sets
    combined (a 3-row floor alone fires on noise with a K = 2 baseline).

    With an ack (Skip), (a)–(c) compare with the acknowledged arm-D level, R3 and
    dodged keys count only what is new since it, (e) only the stale rows added
    since it, and (g) only R2/R3 changes since its manifest — a skipped condition
    never fires again by itself. An unconfirmed (a)–(c) is reported as a notice
    with `needs_confirmation`."""
    rules: list[dict] = []

    def fire(rid: str, level: str, detail: str, *, needs_confirmation: bool = False, **kw: Any) -> None:
        rules.append({"rule": rid, "level": level, "detail": detail, "needs_confirmation": needs_confirmation, **kw})

    tot = (cmp or {}).get(BL.TOTAL) or {}
    if tot.get("delta") is not None:
        d, sd = tot["delta"], tot["sigma_delta"]
        if d > float(th["retrain_sigma"]) * sd and d >= float(th["retrain_min_rows"]):
            fire("a", "retrain", f"exam −{d:.1f} rows vs {'the ack' if ack else 'baseline'} ({d / sd:.1f} σ_Δ, σ_Δ {sd:.1f}; "
                 f"McNemar p {tot.get('mcnemar_p')})", needs_confirmation=not confirmed, delta=d, sigma_delta=sd)
        elif d > float(th["notice_sigma"]) * sd:
            fire("exam", "notice", f"exam −{d:.1f} rows ({d / sd:.1f} σ_Δ)", delta=d, sigma_delta=sd)
    for s, c in sorted((cmp or {}).items()):
        if s == BL.TOTAL or c.get("delta") is None or c["n"] < int(th["set_min_n"]):
            continue
        pct = 100.0 * c["delta"] / c["n"]
        if c["delta"] > float(th["set_retrain_sigma"]) * c["sigma_delta"] and pct >= float(th["set_min_pct_points"]):
            fire("b", "retrain", f"{s}: −{c['delta']:.1f} of {c['n']} rows ({pct:.0f} % points, {c['delta'] / c['sigma_delta']:.1f} σ_Δ)",
                 needs_confirmation=not confirmed, set=s)
    oos = [c for s, c in (cmp or {}).items() if s in set(th.get("oos_sets") or [])]
    if oos and all(c.get("delta") is not None for c in oos):
        drop = sum(c["delta"] for c in oos)
        sd_oos = math.sqrt(sum(float(c.get("sigma_delta") or 0.0) ** 2 for c in oos))
        if drop >= float(th["oos_drop_rows"]) and drop > float(th["retrain_sigma"]) * sd_oos:
            fire("c", "retrain", f"out-of-scope rows −{drop:.1f} ({drop / sd_oos:.1f} σ_Δ)" if sd_oos else f"out-of-scope rows −{drop:.1f}",
                 needs_confirmation=not confirmed, delta=drop, sigma_delta=round(sd_oos, 3))
    # dodged keys
    ack_dodged = set((ack or {}).get("dodged_keys") or [])
    dodged = sorted(k for k, v in (probe or {}).items() if v.get("dodged"))
    fresh = [k for k in dodged if k not in ack_dodged]
    new_containers = set(static.get("containers_added") or [])
    container_all_dodged = []
    for c in new_containers:
        probed = [k for k, v in (probe or {}).items() if k.startswith(c + ".") and v.get("n")]
        if probed and all((probe or {})[k].get("dodged") for k in probed) and any(k not in ack_dodged for k in probed):
            container_all_dodged.append(c)
    if len(fresh) >= int(th["dodged_keys_retrain"]) or container_all_dodged:
        fire("d", "retrain", f"{len(fresh)} new key(s) the model misses: {', '.join(fresh[:6])}"
             + (f"; every probed key of {', '.join(container_all_dodged)}" if container_all_dodged else ""), keys=fresh)
    elif len(fresh) >= int(th["dodged_keys_notice"]):
        fire("dodged", "notice", f"{len(fresh)} new key the model misses: {', '.join(fresh)}", keys=fresh)
    # stale rows (after a Skip: only those that went stale since it)
    st_tr, st_ex = imp.get("stale_training_pct"), imp.get("stale_exam_pct")
    n_tr, n_ex = imp.get("stale_training_rows"), len(imp.get("stale_exam_rows") or [])
    since = ""
    if ack and st_tr is not None and ack.get("stale_training_rows") is not None:
        n_tr = max(0, int(imp.get("stale_training_rows") or 0) - int(ack["stale_training_rows"]))
        st_tr = round(100.0 * n_tr / max(1, int(imp.get("training_rows_checked") or imp.get("training_rows") or 1)), 2)
        since = " since the Skip"
    if ack and ack.get("stale_exam_rows") is not None:
        n_ex = max(0, n_ex - int(ack["stale_exam_rows"]))
        st_ex = round(100.0 * n_ex / max(1, int(imp.get("exam_rows") or 1)), 2)
        since = " since the Skip"
    if (st_tr is not None and st_tr >= float(th["stale_training_pct"])) or (st_ex is not None and st_ex >= float(th["stale_exam_pct"])):
        parts = []
        if st_tr is not None:
            parts.append(f"stale training rows {n_tr} ({st_tr} %)")
        parts.append(f"stale exam rows {n_ex} ({st_ex} %)")
        fire("e", "retrain", ", ".join(parts) + since)
    # ghost writes
    if ghost_d is not None:
        ref = max(ghost_t or 0.0, float((ack or {}).get("ghost_writes") or 0.0))
        if ghost_d - ref >= float(th["ghost_writes_delta"]):
            fire("f", "retrain", f"ghost writes {ghost_d:.1f} on today's manifest vs {ref:.1f}")
    # age (after a Skip: only schema changes since the skipped manifest count)
    age_static = static_since_ack if ack else static
    if ack and age_static is None:
        age_static = {"counts": {}}
    a2, a3 = age_static["counts"].get(R2, 0), age_static["counts"].get(R3, 0)
    if age_days is not None and age_days >= float(th["max_age_days"]) and (a2 or a3):
        fire("g", "retrain", f"{age_days:.0f} days since training and the schema changed{' since the Skip' if ack else ''} (R2 {a2}, R3 {a3})")
    # R3 / engine notices
    # (after a Skip whose manifest cannot be read: no notice, as rule (g) — the
    # Skip already saw the full training diff's hard changes)
    r3_static = static_since_ack if ack else static
    r3 = (r3_static or {"counts": {}})["counts"].get(R3, 0)
    if r3:
        fire("r3", "notice", f"{r3} hard-schema change(s){' since the last Skip' if static_since_ack else ''}")
    if imp.get("engine_status") == "stale":
        fire("engine", "notice", "no vendored engine snapshot for the deployed launcher commit (drift checked on the newest)")
    confirmed_retrain = [r for r in rules if r["level"] == "retrain" and not r["needs_confirmation"]]
    unconfirmed = [r for r in rules if r["level"] == "retrain" and r["needs_confirmation"]]
    if confirmed_retrain:
        verdict = "retrain"
    elif unconfirmed or any(r["level"] == "notice" for r in rules):
        verdict = "notice"
    else:
        verdict = "ok"
    return {"verdict": verdict, "rules": rules, "fired": sorted({r["rule"] for r in rules if r["level"] == "retrain"
                                                               and not r["needs_confirmation"]}),
            "needs_confirmation": bool(unconfirmed) and not confirmed_retrain, "calibrated": calibrated,
            "dodged_keys": dodged, "dodged_keys_new": fresh}


# ═════════════════════════════════════════ messages ═════════════════════════════════════════

def h8(h: Any) -> str:
    return str(h or "?")[:8]


def headline(dep: dict, model: dict) -> str:
    lb = dep.get("launcher_build")
    return (f"Launcher manifest `{h8(dep.get('hash'))}`" + (f" (launcher {lb})" if lb else "")
            + f" vs blueberry-muffin `{model.get('tag') or '?'}` (trained on `{h8(model.get('training_manifest'))}`)")


def summary_lines(*, dep: dict, model: dict, cmp: dict | None, ev: dict | None, static: dict, imp: dict, probe: dict | None,
                  confirmed: bool) -> list[str]:
    lines = [headline(dep, model)]
    t = (cmp or {}).get(BL.TOTAL) or {}
    if t.get("arm_mean") is not None:
        if t.get("base_mean") is not None:
            lines.append(f"• Exam on today's manifest: {t['arm_mean']:.0f}/{t['n']} vs baseline {t['base_mean']:.1f} ± "
                         f"{t['sigma_delta']:.1f} ({0.0 - t['delta']:+.1f} rows, McNemar p {t.get('mcnemar_p')}"
                         f"{'; confirmed' if confirmed and t['m'] >= 2 else ''})")
        else:
            lines.append(f"• Exam on today's manifest: {t['arm_mean']:.0f}/{t['n']} (no baseline yet)")
    pk = probe or {}
    dodged = [k for k, v in pk.items() if v.get("dodged")]
    handled = [k for k, v in pk.items() if v.get("n") and not v.get("dodged")]
    nk = imp.get("new_keys") or []
    lines.append(f"• New keys: {len(nk)} ({len(handled)} handled, {len(dodged)} missed"
                 + (f": {', '.join(d.split('.', 1)[-1] for d in dodged[:4])}{' …' if len(dodged) > 4 else ''}" if dodged else "")
                 + f") · stale exam rows {len(imp.get('stale_exam_rows') or [])} · wording changes {static['counts'].get(R1, 0)}"
                 + f" · schema R2 {static['counts'].get(R2, 0)} / R3 {static['counts'].get(R3, 0)}")
    if imp.get("stale_training_rows") is not None:
        lines.append(f"• Training rows naming a key that is gone: {imp['stale_training_rows']} ({imp['stale_training_pct']} %)")
    if imp.get("engine_status") == "stale":
        lines.append("• Engine: no vendored snapshot of the deployed launcher commit (checked on the newest)")
    for r in (ev or {}).get("rules") or []:
        if r["level"] == "retrain":
            lines.append(f"• ({r['rule']}) {r['detail']}{' — awaiting confirmation' if r['needs_confirmation'] else ''}")
    return lines


def thermometer_link(path: str) -> str:
    base = os.environ.get("THERMOMETER_PUBLIC_URL", "https://thermometer.breadstackcrm.com").rstrip("/")
    return f"{base}{path}"


# ═════════════════════════════════════════ the job ═════════════════════════════════════════

# A round that failed and was not resumed (the retrain workflow writes it, a resume or a finished round
# deletes it): no new round starts automatically while it exists — a deterministic failure would otherwise
# spend a full round again at every check.
FAILED_ROUND = "state/failed-round.json"


class Job:
    """One drift command: the bucket, thresholds, the run folder, the result."""

    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.b = gcsio.bucket(a.bucket)
        self.th = PC.load_thresholds(self.b)
        self.dry = bool(a.dry or os.environ.get("DRY") == "1")
        self.now = PC.utc_now()
        self.run_id = a.run_id or f"dr-{self.now:%Y%m%d-%H%M%S}-{a.mode}"
        self.prefix = f"drift/{self.run_id}"
        self.work = Path(a.workdir or tempfile.mkdtemp(prefix=f"drift-{self.run_id}-"))
        self.work.mkdir(parents=True, exist_ok=True)
        self.artifacts: list[str] = []
        self.notifications: list[dict] = []
        self.log_lines: list[str] = []

    def progress(self, msg: str) -> None:
        line = f"[{PC.iso()}] {msg}"
        self.log_lines.append(line)
        print(line, flush=True)

    def put(self, name: str, obj: Any) -> str:
        key = f"{self.prefix}/{name}"
        self.b.write_json(key, obj)
        self.artifacts.append(key)
        return key

    def put_jsonl(self, name: str, rows: list[dict]) -> str:
        key = f"{self.prefix}/{name}"
        self.b.write_bytes(key, b"".join(gcsio.json_bytes(r, indent=None) for r in rows))
        self.artifacts.append(key)
        return key

    # ── inputs ──
    def deployed(self) -> dict:
        if self.a.deployed_file:
            body = json.loads(Path(self.a.deployed_file).read_text(encoding="utf-8"))
            m = Manifest(body)
            return {"hash": m.hash, "build_line": "int", "source": "launcher", "launcher_build": self.a.launcher_build or "file",
                    "generated_from": body.get("generated_from") or {}, "body": body}
        url = self.a.url or os.environ.get("SOUS_CHEF_URL")
        if not url:
            raise PC.ConfigError("SOUS_CHEF_URL (the int service) is not set")
        return PC.deployed_manifest(url, PC.service_key(), line=self.a.line, body=True)

    def manifest_body(self, hsh: str) -> dict:
        from .exam import MANIFEST_FILE
        doc = self.b.read_json(f"manifests/{hsh}.json", None)
        if doc is None and hsh:
            keys = [k for k in self.b.list(f"manifests/{hsh[:8]}") if k.endswith(".json")]
            if len(keys) == 1:
                doc = self.b.read_json(keys[0])
        if doc is None:
            m = Manifest(json.loads(MANIFEST_FILE.read_text(encoding="utf-8")))
            if m.hash.startswith(str(hsh)):
                doc = m.doc
        if doc is None:
            raise PC.ConfigError(f"no manifest body for {h8(hsh)} (manifests/<hash>.json in the bucket; playbook R0 seeds the training one)")
        return doc

    def model(self) -> dict:
        m = self.b.read_json("state/model.json", None)
        if not m or not m.get("training_manifest"):
            raise PC.ConfigError("state/model.json is missing or names no training_manifest (playbook R0 seeds it for r4)")
        want = self.a.model_digest
        if want and m.get("digest") and m["digest"] != want:
            # production serves another image than the pipeline's model: a manual rollback (make model-rollout)
            # to an earlier model is judged as THAT model, on its own training manifest and baseline
            for h in reversed(m.get("history") or []):
                if h.get("digest") == want and h.get("training_manifest"):
                    self.progress(f"production serves {h.get('tag')} ({want[:19]}…), not state/model.json's {m.get('tag')}: "
                                  f"judging {h.get('tag')} (a manual rollback)")
                    self.notifications.append(PC.notification(
                        "drift_notice", severity="warn", title=f"Sous Chef drift: production runs `{h.get('tag')}`, not `{m.get('tag')}`",
                        summary=(f"The production model service serves `{h.get('tag')}` ({want[:19]}…), an earlier model, while "
                                 f"state/model.json names `{m.get('tag')}`: a manual rollback. Drift is judged for `{h.get('tag')}` "
                                 f"until state/model.json is updated (docs/DRIFT.md \"Manual rollbacks\")."),
                        details={"serving": h, "state_model": {k: m.get(k) for k in ("tag", "digest")}},
                        dedup_key=f"drift:rollback:{want[7:19]}"))
                    return {**h, "history": m.get("history") or [], "rolled_back_from": m.get("tag")}
            raise PC.ConfigError(
                f"production serves {want[:19]}…, which is neither state/model.json's model ({m.get('tag')}, "
                f"{str(m.get('digest'))[:19]}…) nor in its history: a model rolled out outside the pipeline. Write its "
                f"state/model.json entry by hand (docs/DRIFT.md \"Manual rollbacks\") before drift can judge it.")
        return m

    def write_manifest(self, hsh: str, body: dict) -> None:
        key = f"manifests/{hsh}.json"
        if not self.b.exists(key):
            self.b.write_json(key, body, indent=None)

    def engine_index(self) -> dict | None:
        return self.b.read_json("engine/index.json", None)

    def engine_status(self, dep: dict) -> str:
        from .enginebundle import status_for
        return status_for(self.engine_index(), (dep.get("generated_from") or {}).get("sha"))

    def training_cases(self, model: dict) -> list[dict] | None:
        keys = list(model.get("training_cases") or [])
        if not keys:
            idx = self.b.read_json("cases/pool/index.json", None) or {}
            keys = [f["path"] for f in idx.get("files") or [] if f.get("trained_in") and f.get("trained_in") == model.get("tag")]
        if not keys and self.b.exists("cases/pool/r4/cases-all.jsonl") and str(model.get("tag")) == "r4":
            keys = ["cases/pool/r4/cases-all.jsonl"]
        if not keys:
            return None
        out = []
        for k in keys:
            if not self.b.exists(k):
                # e.g. a runs/ path past its lifecycle: the stale-training-rows rule is skipped, never the run
                self.progress(f"training cases {k} are gone; rule (e) checks the exam rows only")
                return None
            out += self.b.read_jsonl(k)
        return out

    def exam_rows(self) -> list[dict]:
        from .exam import load_set
        rows = []
        n = int(self.th.get("smoke_rows_per_set", 10)) if self.a.smoke else None
        for s in EXAM_SETS:
            rs = load_set(s)
            rows += rs[:n] if n else rs
        if self.a.limit_rows:
            rows = rows[: self.a.limit_rows]
        return rows

    def baseline_key(self, digest: str) -> str:
        return f"baselines/{digest}/baseline.json"

    def digest(self, model: dict) -> str:
        d = self.a.model_digest or model.get("digest")
        if not d:
            raise PC.ConfigError("no model image digest (--model-digest, or state/model.json digest)")
        return str(d)


def _local_manifest(job: Job, name: str, doc: dict) -> Path:
    p = job.work / f"manifest-{name}.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


def do_static(job: Job, dep: dict, model: dict, *, with_training: bool = True) -> tuple[dict, dict, Manifest, Manifest]:
    training_doc = job.manifest_body(model["training_manifest"])
    old_m, new_m = Manifest(training_doc), Manifest(dep["body"])
    sd = static_diff(training_doc, dep["body"], jaccard=float(job.th["rename_jaccard"]), cap=int(job.th["field_list_cap"]))
    imp = impact(sd, old_m, new_m, job.exam_rows(), job.training_cases(model) if with_training else None,
                 engine_status=job.engine_status(dep))
    job.put("static.json", {"run_id": job.run_id, "at": PC.iso(job.now), "training": old_m.hash, "deployed": new_m.hash,
                            "launcher_build": dep.get("launcher_build"), **sd, "impact": imp})
    return sd, imp, old_m, new_m


def static_notifications(job: Job, dep: dict, model: dict, sd: dict, imp: dict) -> list[dict]:
    out = []
    if imp.get("engine_status") == "stale":
        sha = (dep.get("generated_from") or {}).get("sha")
        out.append(PC.notification(
            "engine_stale", severity="warn", title=f"Sous Chef drift: no engine snapshot for launcher {sha}",
            summary=(f"The deployed launcher manifest `{h8(dep['hash'])}` was generated from {sha}, which has no vendored engine "
                     f"snapshot; drift checks run on the newest one and a retrain will not start.\n"
                     f"Run: make -C lab engine-vendor SHA={sha} CC=<cookie-cutter checkout> UPLOAD=1 "
                     f"PIPELINE_BUCKET={job.b.url}"),
            details={"sha": sha, "deployed": dep["hash"]}, dedup_key=f"engine:{str(sha)[:12]}:stale"))
    if sd["counts"].get(R3):
        out.append(PC.notification(
            "drift_notice", severity="info", title="Sous Chef drift: the launcher's schema changed",
            summary="\n".join([headline(dep, model),
                               f"• R3 {sd['counts'][R3]} (fields +{len(sd['fields_added'])} −{len(sd['fields_removed'])}, renamed "
                               f"{len(sd['renames'])}, retyped {len(sd['fields_retyped'])}, containers +{len(sd['containers_added'])} "
                               f"−{len(sd['containers_removed'])}) · R2 {sd['counts'].get(R2, 0)} · wording {sd['counts'].get(R1, 0)}",
                               f"• New keys to probe: {len(imp['new_keys'])} · stale exam rows {len(imp['stale_exam_rows'])}",
                               "The behavioural check follows after the debounce."]),
            details={"static": compact_static(sd), "impact": {k: v for k, v in imp.items() if k != 'stale_exam_missing'}},
            dedup_key=f"drift:{h8(dep['hash'])}:static"))
    return out


# ── poll ──

DEPLOYED_STATE = "state/deployed.json"


def update_deployed(job: Job, fn) -> dict:
    """Read-modify-write state/deployed.json on its generation: a poll that overlaps the end of a check never
    brings back the pending_check_at the check just cleared (a duplicate check)."""
    for _ in range(5):
        gen = job.b.generation(DEPLOYED_STATE)
        cur = job.b.read_json(DEPLOYED_STATE, None) or {} if gen is not None else {}
        new = fn(dict(cur))
        try:
            job.b.write_json(DEPLOYED_STATE, new, if_generation=gen if gen is not None else 0)
            return new
        except gcsio.PreconditionFailed:
            job.progress("state/deployed.json moved under us; re-reading")
    raise PC.PipelineError("state/deployed.json kept changing (5 conditional writes lost)")


def cmd_poll(job: Job) -> dict:
    dep = job.deployed()
    job.write_manifest(dep["hash"], dep["body"])
    st = job.b.read_json(DEPLOYED_STATE, None) or {}
    changed = st.get("hash") != dep["hash"]
    now = job.now
    debounce = timedelta(minutes=float(job.th["debounce_minutes"]))

    def merge(cur: dict) -> dict:
        prev_hash = cur.get("hash")
        cur.update({"hash": dep["hash"], "launcher_build": dep.get("launcher_build"), "generated_from": dep.get("generated_from"),
                    "seen_at": PC.iso(now), "last_registered_at": dep.get("last_registered_at"),
                    "last_real_turn_at": dep.get("last_real_turn_at")})
        if prev_hash != dep["hash"]:
            cur.update({"changed_at": PC.iso(now), "pending_check_at": PC.iso(now + debounce), "previous_hash": prev_hash,
                        "static_run": job.run_id})
        return cur

    new_st = merge(dict(st))
    static_summary = None
    if changed:
        try:
            model = job.model()
        except PC.ConfigError as e:
            model = None
            job.progress(f"poll: {e} — no static diff")
        if model:
            sd, imp, _, _ = do_static(job, dep, model)
            static_summary = {"counts": sd["counts"], "new_keys": len(imp["new_keys"]), "stale_exam_rows": len(imp["stale_exam_rows"]),
                              "engine_status": imp["engine_status"]}
            job.notifications += static_notifications(job, dep, model, sd, imp)
        job.progress(f"poll: deployed manifest changed {h8(st.get('hash'))} → {h8(dep['hash'])}; check after {new_st['pending_check_at']}")
    new_st = update_deployed(job, merge)
    due_at = PC.parse_iso(new_st.get("pending_check_at"))
    check_due = bool(due_at and now >= due_at)
    # the daily heartbeat (the thermometer's dead man's switch)
    hb = job.b.read_json("state/heartbeat.json", None) or {}
    today = now.strftime("%Y-%m-%d")
    if hb.get("date") != today:
        job.notifications.append(PC.notification("heartbeat", title="Sous Chef drift poll alive",
                                                 summary=f"deployed {h8(dep['hash'])}", details={"deployed": dep["hash"]}))
        job.b.write_json("state/heartbeat.json", {"date": today, "at": PC.iso(now), "run_id": job.run_id})
    return {"verdict": None, "deployed": {k: dep.get(k) for k in ("hash", "launcher_build", "generated_from", "last_registered_at",
                                                               "last_real_turn_at", "real_turns_7d")},
            "changed": changed, "debounce_until": new_st.get("pending_check_at"), "check_due": check_due, "static": static_summary}


# ── static ──

def cmd_static(job: Job) -> dict:
    dep = job.deployed()
    job.write_manifest(dep["hash"], dep["body"])
    model = job.model()
    sd, imp, _, _ = do_static(job, dep, model)
    job.notifications += static_notifications(job, dep, model, sd, imp)
    print("\n".join(summary_lines(dep=dep, model=model, cmp=None, ev=None, static=sd, imp=imp, probe=None, confirmed=False)))
    return {"verdict": None, "deployed": {k: dep.get(k) for k in ("hash", "launcher_build", "generated_from")},
            "static": {"counts": sd["counts"], "new_keys": imp["new_keys"], "stale_exam_rows": len(imp["stale_exam_rows"]),
                       "engine_status": imp["engine_status"]}}


# ── the behavioural modes ──

@contextmanager
def gpu_target(job: Job, dep: dict, model: dict) -> Iterator[Target]:
    """The harness on the eval service (or DRY); the engine bundle active."""
    with ExitStack() as stack:
        engine_path = None
        if not job.a.fake_engine:
            if job.a.engine_dir or job.dry:
                from .engine import ENGINE_REL_PATH, find_engine
                found = find_engine(Path(job.a.engine_dir) / ENGINE_REL_PATH if job.a.engine_dir else None)
                if found is None:
                    raise PC.ConfigError("no engine (COOKIE_CUTTER_DIR / --engine-dir), or pass --fake-engine for a DRY run")
                engine_path = str(found)
            else:
                from . import enginebundle as EB
                bundle = EB.ensure(job.b, (dep.get("generated_from") or {}).get("sha") or "newest",
                                   workdir=job.work / "engine", redump=not job.a.no_redump)
                engine_path = str(bundle.engine)
                job.progress(f"engine snapshot {bundle.sha[:12]} ({bundle.status}) fingerprint {bundle.fingerprint[:12]}")
        harness = None
        if not job.dry:
            url = job.a.eval_url or os.environ.get("EVAL_URL")
            if not url:
                raise PC.ConfigError("--eval-url (the eval GPU service) is required for a real run")
            if not job.a.no_warm:
                warm = PC.gpu_warm(url, wait_s=60 * float(job.th["gpu_warm_minutes"]), auth=job.a.eval_auth,
                                   progress=job.progress)
                if not warm["ok"]:
                    raise EvalUnavailable(f"{url} did not answer in {job.th['gpu_warm_minutes']} min ({warm.get('last')})")
                job.progress(f"eval service warm in {warm['seconds']}s")
            from .harness import Harness
            harness = stack.enter_context(Harness({ROUTE: url}, run_id=job.run_id, workdir=job.work / "harness",
                                                  auth=job.a.eval_auth,
                                                  route_overrides={"prime": "off"} if job.a.eval_auth == "none" else None))
            job.progress(f"harness up: {harness.build_id} → {url}")
        yield Target(dry=job.dry, dry_model=job.a.dry_model, harness=harness, workers=int(job.th["exam_workers"]),
                     fake_engine=job.a.fake_engine, engine_path=engine_path, progress=job.progress)


class EvalUnavailable(PC.PipelineError):
    pass


def ack_static(job: Job, dep: dict, ack: dict | None) -> dict | None:
    if not ack or not ack.get("deployed_hash") or ack["deployed_hash"] == dep["hash"]:
        return static_diff(dep["body"], dep["body"]) if ack else None
    try:
        body = job.manifest_body(ack["deployed_hash"])
    except PC.ConfigError:
        return None
    return static_diff(body, dep["body"], jaccard=float(job.th["rename_jaccard"]), cap=50)


def behaviour(job: Job, mode: str) -> dict:
    """check | weekly | calibrate."""
    dep = job.deployed()
    job.write_manifest(dep["hash"], dep["body"])
    st = job.b.read_json(DEPLOYED_STATE, None) or {}
    if mode == "check" and st.get("hash") and st["hash"] != dep["hash"] and not job.a.force:
        def moved(cur: dict) -> dict:
            cur.update({"hash": dep["hash"], "changed_at": PC.iso(job.now), "launcher_build": dep.get("launcher_build"),
                        "generated_from": dep.get("generated_from"),
                        "pending_check_at": PC.iso(job.now + timedelta(minutes=float(job.th["debounce_minutes"])))})
            return cur
        st = update_deployed(job, moved)
        job.progress("check: the deployed manifest moved again — debounce restarted")
        return {"status": "debounced", "verdict": None, "changed": True, "debounce_until": st["pending_check_at"],
                "deployed": {"hash": dep["hash"]}}
    model = job.model()
    digest = job.digest(model)
    ttl = 60 * float(job.th["gpu_lock_ttl_minutes"])
    if not job.a.no_lock:
        try:
            job.b.acquire("gpu-eval", job.run_id, ttl)
        except gcsio.LockHeld as e:
            job.progress(f"{mode}: {e} — deferred")
            return {"status": "deferred", "verdict": None, "deployed": {"hash": dep["hash"]}, "lock": e.holder}
    try:
        return _behaviour_locked(job, mode, dep, model, digest, st)
    finally:
        if not job.a.no_lock:
            job.b.release("gpu-eval", job.run_id)


def _behaviour_locked(job: Job, mode: str, dep: dict, model: dict, digest: str, st: dict) -> dict:
    sd, imp, old_m, new_m = do_static(job, dep, model, with_training=mode != "calibrate")
    ack = job.b.read_json("state/ack.json", None)
    sd_ack = ack_static(job, dep, ack)
    base = job.b.read_json(job.baseline_key(digest), None)
    calibrated = bool(base and base.get("calibrated") and len(base.get("replicates") or []) >= 2)
    rows = job.exam_rows()
    home = home_context(rows)
    t_path = _local_manifest(job, "training", old_m.doc)
    d_path = _local_manifest(job, "deployed", new_m.doc)
    tag = f"{job.run_id[3:23].replace('-', '')}"
    out: dict[str, Any] = {"deployed": {k: dep.get(k) for k in ("hash", "launcher_build", "generated_from")}, "mode": mode,
                           "model": {k: model.get(k) for k in ("tag", "digest", "training_manifest", "trained_at")} | {"digest": digest}}
    try:
        with gpu_target(job, dep, model) as tgt:
            measured = _measure(job, mode, tgt, dep, model, digest, sd, imp, old_m, new_m, rows, home, t_path, d_path, tag,
                                base, calibrated, ack, sd_ack, out)
    except EvalUnavailable as e:
        job.progress(f"{mode}: {e}")
        return {**out, "status": "eval_unavailable", "verdict": None, "error": PC.scrub(str(e))}
    if "status" in measured:
        return measured
    return _conclude(job, mode, dep, model, st, sd, imp, ack, sd_ack, out, **measured)


def _measure(job: Job, mode: str, tgt: Target, dep: dict, model: dict, digest: str, sd: dict, imp: dict, old_m: Manifest,
             new_m: Manifest, rows: list[dict], home: dict | None, t_path: Path, d_path: Path, tag: str, base: dict | None,
             calibrated: bool, ack: dict | None, sd_ack: dict | None, out: dict) -> dict:
    th = job.th
    metrics: dict[str, Any] = {}
    t0 = time.monotonic()
    # arm T (calibrate: K replicates; weekly: 1) — the baseline
    arm_t_reps: list[list[dict]] = []
    t_rows, t_stale, _ = exam_on(rows, old_m, jaccard=float(th["rename_jaccard"]))
    if t_stale:
        job.progress(f"arm T: {len(t_stale)} exam row(s) name keys the training manifest {h8(old_m.hash)} lacks; left out")
    if mode in ("calibrate", "weekly"):
        rederive(t_rows, t_path, home)
        k_n = int(job.a.replicates or th["calibrate_replicates"]) if mode == "calibrate" else 1
        for k in range(1, k_n + 1):
            job.progress(f"arm T replicate {k}/{k_n}: {len(t_rows)} rows on {h8(old_m.hash)}")
            rs = tgt.run_rows("T", t_rows, t_path, old_m, site_prefix=f"lab-dr{tag}-t{k}-", sample=k, home=home)
            job.put_jsonl(f"exam/T-{k}.jsonl", rs)
            arm_t_reps.append(rs)
    f0h_t = f0h_d = None
    if mode in ("calibrate", "weekly") and not job.a.no_f0h:
        f0h_t, f0h_d = run_f0h(job, tgt, model, old_m, new_m, t_path, d_path, sd, both=mode == "weekly")
    if arm_t_reps:
        if base is None or mode == "calibrate" and job.a.fresh_baseline:
            base = BL.new(digest=digest, tag=model.get("tag"), training_manifest=old_m.hash,
                          engine_sha=(dep.get("generated_from") or {}).get("sha"), service_build=_service_build(tgt),
                          source="calibration")
        for i, rs in enumerate(arm_t_reps):
            BL.add_replicate(base, results_of(rs), run=f"{job.run_id}#T{i + 1}", at=PC.iso(), ghost_writes=sum(r["ghost"] for r in rs),
                             window=int(th["baseline_window"]), f0h=f0h_t if i == 0 else None)
        if mode == "calibrate":
            base["calibrated"] = len(base["replicates"]) >= 2
            base["source"] = "calibration"
        base["updated_at"] = PC.iso()
        job.b.write_json(job.baseline_key(digest), base)
        job.artifacts.append(job.baseline_key(digest))
        calibrated = bool(base.get("calibrated") and len(base["replicates"]) >= 2)
    if mode == "calibrate":
        per = (base or {}).get("per_set", {}).get(BL.TOTAL) or {}
        metrics.update({"baseline": {"replicates": len(base["replicates"]), "total": per, "calibrated": calibrated,
                                     "ghost_writes_mean": base.get("ghost_writes_mean")}, "f0h_t": _rate(f0h_t),
                        "minutes": round((time.monotonic() - t0) / 60, 1)})
        job.notifications.append(PC.notification(
            "drift_notice", severity="info", title=f"Sous Chef drift baseline calibrated for `{model.get('tag')}`",
            summary=(f"blueberry-muffin `{model.get('tag')}` ({digest[:19]}…) on its training manifest `{h8(old_m.hash)}`: "
                     f"{per.get('mean')}/{per.get('n')} over {len(base['replicates'])} runs (SD {per.get('sd_run')})"
                     + (f"; F0H {_rate(f0h_t)}" if f0h_t else "") + ". Auto-retrain can be enabled."),
            details={"baseline": job.baseline_key(digest), "per_set": base.get("per_set")}, dedup_key=None))
        return {**out, "status": "ok", "verdict": None, "baseline": job.baseline_key(digest), "metrics": metrics}
    # arm D
    d_rows, stale, rep = exam_on(rows, new_m, jaccard=float(th["rename_jaccard"]))
    rederive(d_rows, d_path, home)
    arm_d_reps: list[list[dict]] = []

    def run_d(k: int) -> None:
        job.progress(f"arm D replicate {k}: {len(d_rows)} rows on {h8(new_m.hash)} ({len(stale)} stale left out)")
        rs = tgt.run_rows("D", d_rows, d_path, new_m, site_prefix=f"lab-dr{tag}-d{k}-", sample=k, script_rows=t_rows,
                          script_manifest=old_m, home=home)
        job.put_jsonl(f"exam/D-{k}.jsonl", rs)
        arm_d_reps.append(rs)

    # two replicates up front (thresholds arm_d_replicates): timeliness over cost — a
    # rule that fires is confirmed in the same run instead of a second pass later
    for k in range(1, max(1, int(th.get("arm_d_replicates", 1))) + 1):
        run_d(k)
    # probe
    probe_keys = list(imp.get("new_keys") or [])
    p_rows, coverage = probe_rows(new_m, probe_keys, state=_vintage(d_path, home), per_key=int(th["probe_cases_per_key"]),
                                  max_keys=int(th["probe_max_keys"]))
    probe_reps: list[list[dict]] = []
    for k in range(1, int(th["probe_samples"]) + 1 if p_rows else 1):
        job.progress(f"arm P sample {k}: {len(p_rows)} rows for {len(probe_keys)} new key(s)")
        rs = tgt.run_rows("P", p_rows, d_path, new_m, site_prefix=f"lab-dr{tag}-p{k}-", sample=k, probe=True, home=home)
        job.put_jsonl(f"exam/P-{k}.jsonl", rs)
        probe_reps.append(rs)
    probe = probe_scores(probe_reps, [k for k in probe_keys if coverage.get(k)], th)
    unprobed = [k for k in probe_keys if not coverage.get(k)]

    def compare_now() -> dict:
        ref, ref_k = BL.ack_reference(ack)
        return BL.compare(base, [results_of(r) for r in arm_d_reps], th,
                          ref_means=ref, ref_reps=ref_k, ref_rows=BL.ack_rows(ack))

    def ghost_d() -> float:
        return sum(sum(r["ghost"] for r in rep_) for rep_ in arm_d_reps) / len(arm_d_reps)

    age = None
    tr_at = PC.parse_iso(model.get("trained_at"))
    if tr_at:
        age = (job.now - tr_at).total_seconds() / 86400
    ghost_t = (base or {}).get("ghost_writes_mean")
    cmp = compare_now()
    ev = evaluate(cmp=cmp, static=sd, imp=imp, probe=probe, ghost_d=ghost_d(), ghost_t=ghost_t, age_days=age, th=th,
                  confirmed=len(arm_d_reps) >= 2, calibrated=calibrated, ack=ack, static_since_ack=sd_ack)
    if ev["needs_confirmation"] and not job.a.no_confirm:
        job.progress("an exam rule fired on one replicate: running the confirmation replicate")
        run_d(len(arm_d_reps) + 1)
        cmp = compare_now()
        ev = evaluate(cmp=cmp, static=sd, imp=imp, probe=probe, ghost_d=ghost_d(), ghost_t=ghost_t, age_days=age, th=th,
                      confirmed=True, calibrated=calibrated, ack=ack, static_since_ack=sd_ack)
    if f0h_d is not None:
        metrics["f0h"] = {"T": _rate(f0h_t), "D": _rate(f0h_d)}
    return {"cmp": cmp, "ev": ev, "arm_d_reps": arm_d_reps, "stale": stale, "probe": probe, "coverage": coverage,
            "unprobed": unprobed, "p_rows": p_rows, "ghost_d": ghost_d(), "ghost_t": ghost_t, "calibrated": calibrated,
            "metrics": metrics}


def _conclude(job: Job, mode: str, dep: dict, model: dict, st: dict, sd: dict, imp: dict, ack: dict | None,
              sd_ack: dict | None, out: dict, *, cmp: dict, ev: dict, arm_d_reps: list, stale: list, probe: dict,
              coverage: dict, unprobed: list, p_rows: list, ghost_d: float, ghost_t: float | None, calibrated: bool,
              metrics: dict) -> dict:
    # ack material (the retrain workflow copies it into state/ack.json on Skip)
    d_sets = {s: c["arm_mean"] for s, c in cmp.items() if c.get("arm_mean") is not None}
    # per row too (arm_D_rows): a later check compares on the rows both have, so rows that go stale or are
    # rebased after the Skip never shift Δ; the stale counts let rule (e) count only growth beyond the Skip
    ack_material = {"deployed_hash": dep["hash"], "static_counts": sd["counts"], "arm_D_mean": (cmp.get(BL.TOTAL) or {}).get("arm_mean"),
                    "arm_D_sets": d_sets, "arm_D_replicates": len(arm_d_reps), "arm_D_rows": [results_of(r) for r in arm_d_reps],
                    "dodged_keys": ev["dodged_keys"], "ghost_writes": ghost_d, "drift_run": job.run_id,
                    "stale_training_rows": imp.get("stale_training_rows"), "stale_exam_rows": len(imp.get("stale_exam_rows") or [])}
    job.put("ack-material.json", ack_material)
    # the newest check's material: a Skip of a round started by hand (no drift run) acknowledges it
    job.b.write_json("state/last-ack-material.json", ack_material)
    retrain_lock = job.b.read_lock("retrain")
    lock_live = bool(retrain_lock and (PC.parse_iso(retrain_lock.get("expires_at")) or job.now) > job.now)
    lock_stale = bool(retrain_lock) and not lock_live
    failed_round = job.b.read_json(FAILED_ROUND, None)
    engine_ok = imp.get("engine_status") != "stale"
    eligible = ev["verdict"] == "retrain" and calibrated and engine_ok and not failed_round and not lock_stale
    why_not = None
    if ev["verdict"] == "retrain" and not eligible:
        why_not = ("baseline not calibrated" if not calibrated else "engine snapshot stale" if not engine_ok
                   else (f"round {failed_round.get('run_id')} failed at {failed_round.get('stage')} and is not resumed "
                         f"(resume it with resume_from, or delete {FAILED_ROUND}: docs/RETRAIN.md R8)") if failed_round
                   else f"a stale retrain lock (round {retrain_lock.get('run_id')}, no progress since {retrain_lock.get('updated_at')}): "
                        f"look at that round, then delete state/locks/retrain.json (docs/RETRAIN.md R8)")
    lines = summary_lines(dep=dep, model=model, cmp=cmp, ev=ev, static=sd, imp=imp, probe=probe, confirmed=len(arm_d_reps) >= 2)
    if ack:
        lines.append(f"• Since the Skip of `{ack.get('candidate_id')}` ({str(ack.get('at') or '')[:10]}): compared with the "
                     f"acknowledged level {ack.get('arm_D_mean')}")
    if unprobed:
        lines.append(f"• New keys the generator could not probe: {len(unprobed)} ({', '.join(u.split('.', 1)[-1] for u in unprobed[:4])})")
    report = {"run_id": job.run_id, "mode": mode, "at": PC.iso(job.now), **out, "verdict": ev, "compare": cmp,
              "compare_note": "Δ = baseline mean − arm D (rows); σ_Δ = σ_run·sqrt(1/m + 1/K)",
              "static": compact_static(sd), "impact": {k: v for k, v in imp.items()}, "probe": probe, "probe_coverage": coverage,
              "unprobed_keys": unprobed, "ack": ack, "static_since_ack": compact_static(sd_ack) if sd_ack else None,
              "ghost_writes": {"D": ghost_d, "T": ghost_t}, "stale_rows": [row_key(r) for r in stale], "metrics": metrics,
              "summary": lines, "retrain_lock": retrain_lock if lock_live else None}
    job.put("report.json", report)
    dk = h8(dep["hash"])
    link = thermometer_link(f"/sous-chef/pipeline?run={job.run_id}")
    if ev["verdict"] == "retrain":
        auto = job.a.retrain_auto == "on"
        if eligible and auto and not lock_live:
            title, tail = "Sous Chef drift: retraining started", "A retrain round starts now; the decision should land in about 10 h."
        else:
            why = why_not or ("auto-retrain is off" if not auto else f"round {retrain_lock.get('run_id')} is running")
            title, tail = "Sous Chef drift: retrain recommended", f"No round started ({why}). Playbook: docs/RETRAIN.md."
        job.notifications.insert(0, PC.notification("drift_detected", severity="warn", title=title,
                                                    summary="\n".join(lines + [tail]),
                                                    details={"report": job.b.uri(f'{job.prefix}/report.json'), "fired": ev["fired"], "link": link},
                                                    dedup_key=f"drift:{dk}:retrain"))
    elif ev["verdict"] == "notice":
        job.notifications.insert(0, PC.notification("drift_notice", severity="info", title="Sous Chef drift notice",
                                                     summary="\n".join(lines),
                                                     details={"report": job.b.uri(f'{job.prefix}/report.json'),
                                                              "rules": [r["rule"] for r in ev["rules"]], "link": link},
                                                     dedup_key=f"drift:{dk}:notice"))
    elif mode == "weekly":
        t = cmp.get(BL.TOTAL) or {}
        handled = sum(1 for v in probe.values() if not v.get("dodged"))
        job.notifications.insert(0, PC.notification(
            "weekly_all_clear", severity="info", title="Sous Chef weekly drift check: all clear",
            summary=(f"blueberry-muffin `{model.get('tag')}` on launcher manifest `{dk}`: exam {t.get('arm_mean', 0):.0f}/{t.get('n')} "
                     f"(baseline {t.get('base_mean')} ± {t.get('sigma_delta')}) · new keys {handled}/{len(probe)} handled · "
                     f"{len(stale)} stale exam rows · last launcher change {str(st.get('changed_at') or '?')[:10]}"),
            details={"report": job.b.uri(f"{job.prefix}/report.json")}, dedup_key=f"weekly:{job.now:%Y-%m-%d}"))
    if imp.get("engine_status") == "stale":
        job.notifications += [n for n in static_notifications(job, dep, model, sd, imp) if n["kind"] == "engine_stale"]
    def checked(st2: dict) -> dict:
        if st2.get("hash") in (None, dep["hash"]):
            st2.update({"hash": dep["hash"], "pending_check_at": None, "last_check_run": job.run_id, "last_check_at": PC.iso(),
                        "last_verdict": ev["verdict"], "last_mode": mode})
        return st2

    update_deployed(job, checked)
    metrics.update({"arm_d": cmp.get(BL.TOTAL), "dodged": ev["dodged_keys"], "stale_rows": len(stale),
                    "probe_rows": len(p_rows), "replicates_d": len(arm_d_reps)})
    reason = "; ".join(f"({r['rule']}) {r['detail']}" for r in ev["rules"] if r["level"] == "retrain" and not r["needs_confirmation"]) or None
    return {**out, "status": "ok", "verdict": ev["verdict"], "changed": False, "retrain_reason": reason,
            "retrain_eligible": eligible, "retrain_blocked_by": why_not, "baseline_calibrated": calibrated,
            "engine_status": imp.get("engine_status"), "metrics": metrics, "debounce_until": None}


def _vintage(path: Path, home: dict | None) -> dict:
    from .exam import vintage_state
    return vintage_state(path, home)


def _rate(res: dict[str, int] | None) -> str | None:
    if not res:
        return None
    return f"{sum(res.values())}/{len(res)}"


def _service_build(tgt: Target) -> str | None:
    if tgt.harness is not None:
        return tgt.harness.build
    return "dry"


def run_f0h(job: Job, tgt: Target, model: dict, old_m: Manifest, new_m: Manifest, t_path: Path, d_path: Path, sd: dict, *,
            both: bool) -> tuple[dict | None, dict | None]:
    """The generated hold-out (F0H, 233 cases) on arm T and, weekly, rebased onto
    the deployed manifest on arm D: through r4collect.run (the r4 score)."""
    key = "cases/pool/r4/cases-all.jsonl"
    if not job.b.exists(key) or tgt.fake_engine:
        job.progress(f"F0H skipped ({'no ' + key if not job.b.exists(key) else 'fake engine'})")
        return None, None
    allf = [c for c in job.b.read_jsonl(key) if (c.get("meta") or {}).get("family") == "F0H"]
    cases = [c for c in allf if c.get("manifest_hash") == old_m.hash]
    if job.a.smoke:
        cases = cases[: int(job.th["smoke_rows_per_set"])]
    if not cases and allf:
        # a model trained on another manifest than r4's: F0H rebased onto its training manifest first
        src_h = Counter(str(c.get("manifest_hash")) for c in allf).most_common(1)[0][0]
        try:
            src_m = Manifest(job.manifest_body(src_h))
        except PC.ConfigError as e:
            job.progress(f"F0H skipped: {e}")
            return None, None
        src = [c for c in allf if str(c.get("manifest_hash")) == src_h]
        if job.a.smoke:
            src = src[: int(job.th["smoke_rows_per_set"])]
        from .engine import PersistentEngine
        eng_t = PersistentEngine(tgt.engine_path, manifest_path=t_path, size=2)
        try:
            cases, rep_t = rebase_generated(src, old_m, static_diff(src_m.doc, old_m.doc).get("renames"), old_manifest=src_m,
                                            engine=eng_t)
        finally:
            eng_t.close()
        job.progress(f"F0H rebased from {h8(src_h)} onto the training manifest {h8(old_m.hash)}: {rep_t}")
    if not cases:
        return None, None
    res_t = _r4_arm(job, tgt, "F0H-T", cases, old_m, t_path)
    res_d = None
    if both:
        from .engine import PersistentEngine
        eng = PersistentEngine(tgt.engine_path, manifest_path=d_path, size=2)
        try:
            d_cases, rep = rebase_generated(cases, new_m, sd.get("renames"), old_manifest=old_m, engine=eng)
        finally:
            eng.close()
        job.progress(f"F0H rebased onto {h8(new_m.hash)}: {rep}")
        res_d = _r4_arm(job, tgt, "F0H-D", d_cases, new_m, d_path, script_cases=cases)
    return res_t, res_d


def _r4_arm(job: Job, tgt: Target, name: str, cases: list[dict], m: Manifest, mpath: Path, *,
            script_cases: list[dict] | None = None) -> dict[str, int]:
    from . import collect as C
    from . import r4collect as RC
    from .engine import PersistentEngine, default_cache_dir
    from .r4score import PASSING
    from .state import load_context_fixture

    prefix = f"lab-dr{job.run_id[3:23].replace('-', '')}-{name.lower()}-"
    log = job.work / "f0h" / f"{name}.jsonl"
    svc = None
    url, key = (tgt.harness.url, tgt.harness.key) if tgt.harness else (None, None)
    if tgt.dry:
        if str(LAB_DIR) not in sys.path:
            sys.path.insert(0, str(LAB_DIR))
        from tests import fake_service as FS
        svc = FS.FakeService(manifest=m.doc, key="drift-dry")
        svc.routes = [{"name": ROUTE, "kind": "messages-compatible", "fields_context": False, "fallback": None}]
        svc.scripts.update(RC.oracle_scripts(script_cases if (script_cases and tgt.dry_model == "stale") else cases, prefix))
        svc.start()
        url, key = svc.url, "drift-dry"
    eng = PersistentEngine(tgt.engine_path, manifest_path=mpath, size=tgt.workers, cache_dir=default_cache_dir())
    try:
        client, _ = C.connect(url, key, mpath, cases, sse=False, timeout_s=300, build_id="lab-drift")
        fields, _ = C.resolve_fields("auto", mpath, client, ROUTE)
        RC.run(cases, client=client, engine=eng, log=RC.Log(log), route=ROUTE, label=name, fields=fields, manifest=m,
               extras=load_context_fixture(), assets=C.load_assets(C.DEFAULT_ASSETS), site_prefix=prefix,
               workers=tgt.workers, max_seconds=None, progress=job.progress)
    finally:
        eng.close()
        if svc is not None:
            svc.stop()
    rows = RC.read_index([log])
    if RC.index_path(log).is_file():
        job.b.upload(RC.index_path(log), f"{job.prefix}/exam/{name}.index.jsonl")
        job.artifacts.append(f"{job.prefix}/exam/{name}.index.jsonl")
    return {f"F0H:{r['case_id']}": int(r.get("verdict") in PASSING) for r in rows if r.get("status") == "ok"}


# ═════════════════════════════════════════ CLI ═════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="souschef_lab.pipeline drift", description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=("poll", "static", "check", "weekly", "calibrate"))
    p.add_argument("--run-id", help="dr-YYYYMMDD-HHMM[-…] (default: now)")
    p.add_argument("--bucket", default=os.environ.get("PIPELINE_BUCKET"), help="gs://… or a local folder (DRY)")
    p.add_argument("--url", default=os.environ.get("SOUS_CHEF_URL"), help="the int service ($SOUS_CHEF_URL)")
    p.add_argument("--line", default="int")
    p.add_argument("--eval-url", default=os.environ.get("EVAL_URL"), help="the eval GPU service (check/weekly/calibrate)")
    p.add_argument("--eval-auth", choices=("id-token", "none"), default="id-token",
                   help="none: a local stand-in model (rehearsal), no ID token and no prime")
    p.add_argument("--no-warm", action="store_true", help="skip the GPU service warm-up (a local stand-in)")
    p.add_argument("--model-digest", help="the live model image digest (default state/model.json digest)")
    p.add_argument("--retrain-auto", choices=("on", "off"), default=os.environ.get("RETRAIN_AUTO", "off"),
                   help="word the retrain message as started (on) or recommended (off); the workflow decides")
    p.add_argument("--smoke", action="store_true", help=f"{10} rows per exam set, a few F0H cases")
    p.add_argument("--limit-rows", type=int, help="at most N exam rows (tests)")
    p.add_argument("--replicates", type=int, help="calibrate: arm-T replicates (default thresholds calibrate_replicates)")
    p.add_argument("--fresh-baseline", action="store_true", help="calibrate: start the baseline over instead of adding")
    p.add_argument("--no-confirm", action="store_true", help="check: never run the confirmation replicate")
    p.add_argument("--confirm", action="store_true", help="(accepted; confirmation runs automatically when a rule needs it)")
    p.add_argument("--no-lock", action="store_true", help="do not take the gpu-eval lock (by hand, nothing else running)")
    p.add_argument("--no-f0h", action="store_true", help="weekly/calibrate: skip the generated hold-out")
    p.add_argument("--force", action="store_true", help="check even when the manifest moved since the poll")
    p.add_argument("--dry", action="store_true", help="the lab's fake service in process instead of the harness (DRY=1)")
    p.add_argument("--dry-model", choices=("stale", "oracle"), default="stale")
    p.add_argument("--fake-engine", action="store_true", help="DRY: check keys against the manifest instead of node")
    p.add_argument("--engine-dir", help="a cookie-cutter snapshot instead of the bucket's engine bundle")
    p.add_argument("--no-redump", action="store_true", help="use the bundle's stored browser defaults")
    p.add_argument("--deployed-file", help="DRY: a manifest file stands in for GET /v1/manifests/deployed")
    p.add_argument("--launcher-build", help="DRY: the launcher build to report with --deployed-file")
    p.add_argument("--workdir")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        job = Job(a)
    except (gcsio.StoreError, PC.ConfigError, OSError) as e:
        print(f"drift: {e}", file=sys.stderr)
        return PC.EXIT_CONFIG
    stage = f"drift-{a.mode}"
    try:
        if a.mode == "poll":
            res = cmd_poll(job)
        elif a.mode == "static":
            res = cmd_static(job)
        else:
            res = behaviour(job, a.mode)
        status = res.pop("status", "ok")
        doc = PC.result_doc(status=status, stage=stage, run_id=job.run_id, notifications=job.notifications,
                            metrics=res.pop("metrics", {}), artifacts=job.artifacts, **res)
        job.b.write_json(f"{job.prefix}/result.json", doc)
        print(f"drift {a.mode}: {status}; verdict {doc.get('verdict')}; {len(job.notifications)} notification(s); "
              f"result {job.b.uri(job.prefix + '/result.json')}")
        return PC.EXIT_OK
    except PC.ConfigError as e:
        code = PC.EXIT_CONFIG
        err = e
    except PC.NoLauncherManifest as e:
        code, err = PC.EXIT_OK, e
        doc = PC.result_doc(status="no_launcher_manifest", stage=stage, run_id=job.run_id, verdict=None,
                            notifications=[], error={"class": type(e).__name__, "message": PC.scrub(str(e))})
        job.b.write_json(f"{job.prefix}/result.json", doc)
        print(f"drift {a.mode}: {e}")
        return code
    except Exception as e:  # noqa: BLE001 — every failure leaves a result for the workflow's stage_failed
        code, err = PC.EXIT_FAIL, e
        traceback.print_exc()
    try:
        job.b.write_json(f"{job.prefix}/result.json", PC.failure_doc(stage, job.run_id, err))
    except Exception:  # noqa: BLE001
        pass
    print(f"drift {a.mode}: {type(err).__name__}: {PC.scrub(str(err))}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
