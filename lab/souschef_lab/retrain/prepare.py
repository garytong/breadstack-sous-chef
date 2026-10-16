"""prepare — the round's case file (design §3.2): job `souschef-prepare`.

  1. the pool (cases/pool/index.json: r4's cases-all + every round's authored file,
     approved or skipped) and every earlier round's new-key exam (cases/exam-ext/)
     rebased onto the deployed manifest (rebase.py; a family losing > 20 % stops
     the stage: `rebase_loss`)
  2. cases authored for the fields the pool does not cover (author.py) — written
     to cases/pool/authored/<run>.jsonl (+ cases/exam-ext/<run>.jsonl) and the pool
     index BEFORE collection, so a Skip keeps them and the next big drift batches
     them; a retried job reuses the round's authored file instead of paying twice
  3. the F8 contrast family (contrast.py: look-alike fields, built from the manifest without a
     model call) — regenerated every round on the deployed manifest, never pooled;
     prepare/contrast-report.json. Off: trigger.contrast "off" or $SOUSCHEF_CONTRAST=off
  4. runs/<id>/prepare/{cases.jsonl, report.json, leak-report.json, manifest.json}:
     every case on the deployed manifest — trainable F0–F8 and the holdouts F0H / F9H
     (tagged meta.key_seen against the round's trainable cases)

Trainable pool cases whose expected ops write a field the editor hides in the case's
state (author.hidden_writes; not a deliberate closed-gate case) are dropped after the
rebase (report.json `hidden_write_dropped`; r5 trained 58 such rows). Keep them:
trigger.pool_hidden "keep" or $SOUSCHEF_POOL_HIDDEN=keep.

Trainable cases that expect a bare e-mail address on a link field (socialLinks.email, a
button / gallery link …) expect `mailto:<address>` instead (fix_bare_mailto; report.json
`mailto_fixed`): the site renders those fields as the href itself, so the bare address
is a broken link (r4-F3-0413 taught it). A fixed case's expectation changed, so
carry-forward re-asks it.

Trainable cases that ask for "the font" without naming a part of the site (r4score.generic_font_request on the
r4-F6 "site's text → body + heading" shape) expect EVERY site font under the font policy "all" (the default since
2026-10-15: r4score.site_font_fields on the case's state; fix_site_fonts, report.json `site_fonts_fixed`); the
changed expectation makes carry-forward re-ask them. Policy as_authored ($SOUSCHEF_FONT_POLICY / trigger.font_policy)
leaves them as authored.

SMOKE: 10 pool cases (at least one per family) rebased and one target × 4 cases authored;
no contrast family.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from ..manifest import Manifest
from . import author as A
from . import rebase as RB
from .common import (ConfigError, StageError, Store, deployed_of, load_run, manifest_doc, model_state, notification,
                     read_jsonl, run_prefix, save_manifest, utc_now, write_jsonl, write_result)

POOL_INDEX = "cases/pool/index.json"
SEED_POOL = "cases/pool/r4/cases-all.jsonl"
EXAM_EXT = "cases/exam-ext"
WHY = "cases/why"
DROP_FOR_AUTHORING = ("exact_refused", "exact_empty", "oracle_refused", "remeasure_exact_refused")
FAMILY_SORT = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F0H", "F9H")


def switch(run_file: dict, env_name: str, trigger_key: str, default: str, env: dict | None = None) -> str:
    """A round's switch: $<env_name>, else run.json trigger.<trigger_key>, else `default` (lower-cased)."""
    env = os.environ if env is None else env
    v = env.get(env_name) or ((run_file or {}).get("trigger") or {}).get(trigger_key) or default
    return str(v).strip().lower()


def drop_hidden_writes(cases: list[dict], m: Manifest) -> tuple[list[dict], dict]:
    """(cases without the trainable ones that write a hidden field, {count, by_family, ids})."""
    keep, dropped = [], []
    for c in cases:
        if not c.get("holdout") and A.hidden_writes(c, m):
            dropped.append(c)
        else:
            keep.append(c)
    return keep, {"count": len(dropped), "by_family": dict(Counter(str((c.get("meta") or {}).get("family")) for c in dropped)),
                  "ids": sorted(str(c.get("id")) for c in dropped)[:100]}


def fix_bare_mailto(cases: list[dict]) -> dict:
    """Trainable cases (not holdouts: the exam stays as scored) whose expected writes put a bare e-mail address on a
    link field get `mailto:<address>` in their ops, oracle ops, alt_exact and exact diff. In place; the report."""
    from .. import r4score as R4S

    def fix_ops(ops) -> int:
        n = 0
        for o in ops or []:
            if o.get("op") == "set_fields" and isinstance(o.get("values"), dict):
                for k, v in list(o["values"].items()):
                    if isinstance(v, str) and R4S._is_link_path(k) and R4S.EMAIL.match(v.strip()):
                        o["values"][k] = "mailto:" + v.strip()
                        n += 1
        return n

    def fix_diff(diff) -> None:
        for d in diff or []:
            if isinstance(d, dict) and isinstance(d.get("after"), str) and R4S._is_link_path(str(d.get("path") or "")) \
                    and R4S.EMAIL.match(d["after"].strip()):
                d["after"] = "mailto:" + d["after"].strip()
            elif isinstance(d, list) and len(d) == 2 and R4S._is_link_path(str(d[0])):  # [path, canonical JSON]
                v = R4S._unquote(d[1])
                if isinstance(v, str) and R4S.EMAIL.match(v.strip()):
                    d[1] = json.dumps("mailto:" + v.strip())
    ids = []
    for c in cases:
        if c.get("holdout"):
            continue
        e = c.get("expected") or {}
        n = fix_ops(e.get("ops")) + fix_ops((c.get("meta") or {}).get("oracle_ops"))
        for alt in e.get("alt_exact") or []:
            n += fix_ops(alt.get("ops"))
            fix_diff(alt.get("diff"))
        if n:
            fix_diff(e.get("diff"))
            c.setdefault("meta", {})["fixed"] = sorted(set(list(c["meta"].get("fixed") or []) + ["mailto"]))
            ids.append(str(c.get("id")))
    return {"count": len(ids), "ids": ids[:50]}


def fix_site_fonts(cases: list[dict], m: Manifest, policy: str = "all") -> dict:
    """Under font policy "all": every trainable generic font case (r4score.apply_font_policy notes font:all on a
    "both"-shaped case) expects every site font on its own state (r4score.to_all_fonts). In place; the report."""
    from .. import r4score as R4S
    if policy != "all":
        return {"policy": policy, "count": 0, "ids": []}
    ids = []
    for i, c in enumerate(cases):
        if c.get("holdout") or R4S.site_font_case(c, any_words=True) != "both":
            continue
        c2, note = R4S.apply_font_policy(c, "all", m)
        if note != "font:all":
            continue
        c2.setdefault("meta", {})["fixed"] = sorted(set(list(c2["meta"].get("fixed") or []) + ["site_fonts"]))
        cases[i] = c2
        ids.append(str(c.get("id")))
    return {"policy": policy, "count": len(ids), "ids": ids[:50]}


def pool_files(store: Store) -> list[dict]:
    idx = store.read_json(POOL_INDEX)
    if idx and idx.get("files"):
        return list(idx["files"])
    if store.exists(SEED_POOL):
        return [{"path": SEED_POOL, "manifest": None, "round": "r4", "trained_in": "r4"}]
    raise ConfigError(f"no case pool: {POOL_INDEX} (or {SEED_POOL}) is missing — playbook R0 seeds it")


def load_pool(store: Store, local: Path, exclude: set[str] = frozenset()) -> tuple[list[dict], dict]:
    """Every pool file and every new-key exam file, except `exclude` (the round's own files: a retried job
    adds them back as its authored cases)."""
    cases: list[dict] = []
    files = [f for f in pool_files(store) if f["path"] not in exclude]
    rep: dict[str, Any] = {"files": []}
    for f in files:
        dst = local / "pool" / Path(f["path"]).name
        if not store.download(f["path"], dst):
            raise ConfigError(f"the pool lists {f['path']} but it is not in the bucket")
        rows = read_jsonl(dst)
        for r in rows:
            r.setdefault("meta", {}).setdefault("pool_file", f["path"])
        cases += rows
        rep["files"].append({"path": f["path"], "cases": len(rows), "families": dict(Counter((r.get("meta") or {}).get("family") for r in rows))})
    for rel in store.list(EXAM_EXT):
        if rel.endswith(".jsonl") and rel not in exclude:
            dst = local / "exam-ext" / Path(rel).name
            store.download(rel, dst)
            rows = read_jsonl(dst)
            cases += rows
            rep["files"].append({"path": rel, "cases": len(rows), "families": dict(Counter((r.get("meta") or {}).get("family") for r in rows))})
    return cases, rep


def source_manifests(store: Store, cases: list[dict], m_new: Manifest) -> tuple[dict[str, Manifest], list[str]]:
    from ..exam import MANIFEST_FILE
    out: dict[str, Manifest] = {m_new.hash: m_new}
    missing = []
    fixture = Manifest(json.loads(Path(MANIFEST_FILE).read_text(encoding="utf-8")))
    for h in sorted({str(c.get("manifest_hash")) for c in cases if c.get("manifest_hash")}):
        if h in out:
            continue
        doc = store.read_json(f"manifests/{h}.json")
        if doc is not None:
            out[h] = Manifest(doc)
        elif h == fixture.hash:
            out[h] = fixture
        else:
            missing.append(h)
    return out, missing


def smoke_pick(cases: list[dict], n: int = 10) -> list[dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by[str((c.get("meta") or {}).get("family"))].append(c)
    picked = []
    for fam in sorted(by):
        if fam in RB.SKIP_FAMILIES:
            continue
        cs = sorted(by[fam], key=lambda c: A.h8("smoke:" + str(c.get("id"))))
        if fam == "F2":
            cs = [c for c in cs if c.get("history")] or cs
        picked.append(cs[0])
    rest = sorted((c for c in cases if c not in picked and (c.get("meta") or {}).get("family") not in RB.SKIP_FAMILIES),
                  key=lambda c: A.h8("smoke2:" + str(c.get("id"))))
    return (picked + rest)[:max(n, len(picked))]


def smoke_target(m_new: Manifest, pool: list[dict]) -> dict:
    """SMOKE when nothing is new: the least-covered global text field (proves the authoring edge)."""
    from ..retrain_check import field_index
    cov = A.coverage(pool, m_new)
    fields = [a for a, t in field_index(m_new.doc).items() if a.startswith("globalPanels.") and "[]" not in a and t == "text"]
    if not fields:
        fields = sorted(field_index(m_new.doc))
    a = sorted(fields, key=lambda x: (cov.get(x, 0), x))[0]
    return {"address": a, "kind": "field", "reason": "smoke", "covered": cov.get(a, 0), "holdout": False}


def sort_key(c: dict) -> tuple:
    fam = str((c.get("meta") or {}).get("family"))
    return (FAMILY_SORT.index(fam) if fam in FAMILY_SORT else 99, str(c.get("id")))


def run(store: Store, run_id: str, *, mode: str, engine, engine_info: dict | None, ask: Callable | None,
        workdir: Path, workers: int = 4, progress: Callable[[str], None] = print,
        accept_loss: frozenset[str] | set[str] = frozenset()) -> dict:
    """The stage. Returns the result (also written to runs/<id>/result-prepare.json)."""
    pre = run_prefix(run_id)
    run_file = load_run(store, run_id)
    smoke = mode == "smoke"
    deployed = deployed_of(store, run_file)
    m_new = Manifest(manifest_doc(store, deployed["hash"]))
    if m_new.hash != deployed["hash"]:
        raise ConfigError(f"manifests/{deployed['hash'][:12]}….json hashes to {m_new.hash[:12]}…")
    engine_info = dict(engine_info or {})
    if engine_info.get("status") == "stale" and not smoke:
        raise ConfigError(f"engine_stale: no engine snapshot for launcher sha {engine_info.get('want_sha')} — the ground "
                          f"truth would not match the launcher (make -C lab engine-vendor SHA=…)", reason="engine_stale")
    model = model_state(store)
    m_train_doc = manifest_doc(store, model["training_manifest"]) if model["training_manifest"] != m_new.hash else m_new.doc
    local = Path(workdir)
    save_manifest(m_new.doc, local / "manifest.json")

    # 1. the pool, rebased
    pool, pool_rep = load_pool(store, local, exclude={f"cases/pool/authored/{run_id}.jsonl", f"{EXAM_EXT}/{run_id}.jsonl"})
    progress(f"prepare: pool {len(pool)} cases from {len(pool_rep['files'])} file(s)")
    if smoke:
        pool = smoke_pick(pool, 10)
    manifests, missing = source_manifests(store, pool, m_new)
    why_dir = None
    if store.list(WHY):
        why_dir = local / "why"
        store.sync_down(WHY, why_dir)
    from ..state import load_context_fixture
    extras = load_context_fixture()
    kept, rb = RB.rebase_pool(pool, manifests, m_new, engine, extras, why=why_dir, workers=workers, progress=progress,
                              gate=not smoke)
    rb["missing_source_manifests"] = missing
    progress(f"prepare: rebased {rb['totals']['kept']}/{rb['totals']['in']} kept; dropped {rb['dropped_by_reason']}")
    accepted = [f for f in rb["loss_failed"] if "all" in accept_loss or f in accept_loss]
    if accepted:
        # a person looked (docs/RETRAIN.md R1) and accepted these families' losses for this round
        rb["loss_accepted"] = accepted
        rb["loss_failed"] = [f for f in rb["loss_failed"] if f not in accepted]
        progress(f"prepare: rebase loss accepted for {', '.join(accepted)} (--accept-rebase-loss)")
    if rb["loss_failed"] and not smoke:
        report = {"rebase": rb, "pool": pool_rep, "at": utc_now()}
        store.write_json(f"{pre}/prepare/report.json", report)
        fams = ", ".join(f"{f} −{rb['families'][f]['loss'] * 100:.0f} %" for f in rb["loss_failed"])
        note = notification("stage_failed", "error", "Sous Chef pipeline failed: prepare (rebase loss)",
                            f"Rebasing the case pool onto manifest {m_new.hash[:8]} lost more than "
                            f"{RB.MAX_FAMILY_LOSS:.0%} of a family ({fams}). A person must look before this round trains. "
                            f"Report: {pre}/prepare/report.json · Playbook: docs/RETRAIN.md#r1-prepare",
                            {"stage": "prepare", "reason": "rebase_loss", "families": rb["loss_failed"]},
                            dedup_key=f"{run_id}:prepare")
        res = write_result(store, run_id, "prepare", "rebase_loss", metrics={"rebase": rb["totals"]},
                           notification_=note)
        raise ConfigError("rebase_loss: " + fams, reason="rebase_loss", details={"result": res})

    hidden_rep: dict = {"count": 0, "policy": switch(run_file, "SOUSCHEF_POOL_HIDDEN", "pool_hidden", "drop")}
    if hidden_rep["policy"] != "keep":
        kept, h = drop_hidden_writes(kept, m_new)
        hidden_rep.update(h)
        if h["count"]:
            progress(f"prepare: dropped {h['count']} pool case(s) that write a hidden field {h['by_family']}")

    mailto_rep = fix_bare_mailto(kept)
    if mailto_rep["count"]:
        progress(f"prepare: {mailto_rep['count']} case(s) now expect mailto: on a link field {mailto_rep['ids'][:5]}")
    fonts_rep = fix_site_fonts(kept, m_new, switch(run_file, "SOUSCHEF_FONT_POLICY", "font_policy", "all"))
    if fonts_rep["count"]:
        progress(f"prepare: {fonts_rep['count']} generic font case(s) now expect every site font {fonts_rep['ids'][:5]}")

    # 2. authoring
    # SMOKE never touches the pool: its authored cases stay in the run's folder
    rel_auth = f"{pre}/prepare/authored-smoke.jsonl" if smoke else f"cases/pool/authored/{run_id}.jsonl"
    rel_ext = f"{pre}/prepare/exam-ext-smoke.jsonl" if smoke else f"{EXAM_EXT}/{run_id}.jsonl"
    src_by_id = {str(c.get("id")): c for c in pool}
    dropped_writes: set[str] = set()
    for d in rb.get("dropped") or []:
        if any(str(d.get("why") or "").endswith(x) for x in DROP_FOR_AUTHORING):
            c = src_by_id.get(str(d.get("id")))
            if c is not None:
                m_src = manifests.get(str(c.get("manifest_hash"))) or m_new
                dropped_writes |= A.case_addresses(c, m_src)
    static = None
    drift_run = (run_file.get("trigger") or {}).get("drift_run")
    if drift_run:
        static = store.read_json(f"drift/{drift_run}/static.json")
    extra_keys = set(((static or {}).get("impact") or {}).get("new_keys") or (static or {}).get("new_keys") or [])
    targets, trep = A.find_targets(m_train_doc, m_new, pool=kept, dropped_writes=dropped_writes, extra=extra_keys)
    if smoke:
        targets = targets[:1] or [smoke_target(m_new, kept)]
        targets[0]["holdout"] = False
    authored: list[dict] = []
    holdout_new: list[dict] = []
    if store.exists(rel_auth) or store.exists(rel_ext):
        authored = store.read_jsonl(rel_auth)
        holdout_new = store.read_jsonl(rel_ext)
        arep = {"reused": True, "cases_trainable": len(authored), "cases_holdout": len(holdout_new),
                "anthropic_usd": (store.read_json(f"{pre}/prepare/author-report.json") or {}).get("anthropic_usd", 0.0)}
        progress(f"prepare: reusing this round's authored cases ({len(authored)} + {len(holdout_new)} holdout)")
    elif targets:
        if ask is None:
            raise ConfigError("cases to author but no author (ANTHROPIC_API_KEY / the [author] extra)")
        from ..leak import ExamIndex
        authored, holdout_new, arep = A.author_round(targets, m_new, engine, extras, ask=ask, round_id=run_id, pool=kept,
                                                    leak_index=ExamIndex(why=why_dir), smoke=smoke, progress=progress)
        store.write_json(f"{pre}/prepare/author-report.json", arep)
        if authored:
            store.write_jsonl(rel_auth, authored)
        if holdout_new:
            store.write_jsonl(rel_ext, holdout_new)
        if authored and not smoke:
            idx = store.read_json(POOL_INDEX) or {"files": pool_files(store)}
            files = [f for f in idx.get("files") or [] if f.get("path") != rel_auth]
            files.append({"path": rel_auth, "manifest": m_new.hash, "round": run_id, "trained_in": None,
                          "family_counts": dict(Counter(c["meta"]["family"] for c in authored)), "added_at": utc_now()})
            idx["files"] = files
            store.write_json(POOL_INDEX, idx)
    else:
        arep = {"targets": 0, "cases_trainable": 0, "cases_holdout": 0, "anthropic_usd": 0.0}

    # 3. the contrast family (deterministic, no model call: regenerated each round)
    from ..r4cases import tag_seen
    contrast: list[dict] = []
    crep: dict = {"skipped": "smoke"} if smoke else {}
    cpol = switch(run_file, "SOUSCHEF_CONTRAST", "contrast", "on")
    if not smoke and cpol in ("off", "0", "false", "no"):
        crep = {"skipped": "off"}
    elif not smoke:
        import copy
        from .. import contrast as CT
        from ..leak import ExamIndex
        base = kept + authored + holdout_new
        exam = copy.deepcopy([c for c in base if c.get("holdout") and (c.get("meta") or {}).get("family") == "F0H"])
        tag_seen(exam, [c for c in base if not c.get("holdout")], m_new)
        contrast, crep = CT.contrast_round(m_new, engine, extras, leak_index=ExamIndex(why=why_dir),
                                           pool=[c for c in base if not c.get("holdout")], exam_cases=exam,
                                           progress=progress)
        progress(f"prepare: contrast family F8 {len(contrast)}/{crep.get('asked')} kept")
    store.write_json(f"{pre}/prepare/contrast-report.json", crep)

    # 4. the case file
    allc = kept + authored + holdout_new + contrast
    trainable = [c for c in allc if not c.get("holdout")]
    holdouts = [c for c in allc if c.get("holdout")]
    seen = tag_seen(holdouts, trainable, m_new)
    allc.sort(key=sort_key)
    ids = Counter(str(c["id"]) for c in allc)
    dup = [i for i, n in ids.items() if n > 1]
    if dup:
        raise StageError(f"duplicate case ids in the round: {dup[:5]}", reason="duplicate_ids")
    write_jsonl(local / "cases.jsonl", allc)
    store.upload(local / "cases.jsonl", f"{pre}/prepare/cases.jsonl")
    store.write_json(f"{pre}/prepare/manifest.json", m_new.doc)
    fam = Counter(str(c["meta"]["family"]) for c in allc)
    report = {"run_id": run_id, "mode": mode, "manifest": m_new.hash, "training_manifest": model["training_manifest"],
              "engine": engine_info, "pool": pool_rep,
              "rebase": {k: v for k, v in rb.items() if k not in ("dropped", "leak")},
              "hidden_write_dropped": hidden_rep, "mailto_fixed": mailto_rep, "site_fonts_fixed": fonts_rep,
              "contrast": {k: v for k, v in crep.items() if k not in ("per_side",)},
              "targets": trep, "target_list": targets, "author": {k: v for k, v in arep.items() if k != "per_target"},
              "seen": seen, "families": dict(sorted(fam.items())),
              "totals": {"cases": len(allc), "trainable": len(trainable), "holdout": len(holdouts),
                         "turns_to_ask": sum(1 + len(c.get("history") or []) for c in trainable)},
              "at": utc_now()}
    store.write_json(f"{pre}/prepare/report.json", report)
    store.write_json(f"{pre}/prepare/leak-report.json", {"rebase": rb.get("leak"),
                                                         "authored_rejected_leak": {k: v for k, v in (arep.get("rejects") or {}).items() if "leak" in k}})
    metrics = {"cases": len(allc), "trainable": len(trainable), "holdout": len(holdouts), "families": dict(fam),
               "rebase_dropped": rb["totals"]["dropped"], "targets": len(targets),
               "authored_trainable": len(authored), "authored_holdout": len(holdout_new),
               "contrast": len(contrast), "hidden_write_dropped": hidden_rep["count"],
               "anthropic_usd": arep.get("anthropic_usd", 0.0)}
    return write_result(store, run_id, "prepare", "ok", metrics=metrics,
                        artifacts=[f"{pre}/prepare/cases.jsonl", f"{pre}/prepare/report.json", rel_auth])
