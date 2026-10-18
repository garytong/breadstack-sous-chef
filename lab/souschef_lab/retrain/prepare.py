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

Pool cases that ask to CREATE a page and were filed as report_unsupported where=pages (r4-F3-0064 "add a new page
called Strain Guide", r4-F3-0079, r4-F4-0243 … — before 2026-10-15 creating a page was out of scope) are relabelled
for the create_page tool (fix_page_creates; report.json `page_creates_fixed`): mode `page` expecting create_page
with the title the request names (exam.page_title_of: exact for a quoted / "called X" / "page: X" name, otherwise
one of the request's words), in_nav when it names the menu / header / navigation, draft when it says draft — or
mode `ask` when it names no page at all ("add a blank page"; the rules: ask for the name). Rename / duplicate /
delete / address requests stay where=pages (exam.page_request "manage"). The changed expectation makes carry-forward
re-ask them; a relabelled HOLDOUT gets the id `<id>.k2` (exam.AMENDED_ID_SUFFIX: never paired with a score on its old
key). Off: trigger.page_relabel "off" or $SOUSCHEF_PAGE_RELABEL=off.

Every trainable pool case scored `ask` (r4-F3/F4 authored asks, the relabelled no-title page creates) is judged by
the ask rule (r7 §1.2, check_ask_cases; askrule.classify_case): one that gives a value, a direction or an action
becomes a WRITE case (mode keys) on its own fields when askrule.write_evidence backs it (a direction on its colour /
number field, a toggle named by its label, a literal colour / number / quote / link) — or is dropped when it names
no field or gives no such evidence (`unsure`: never a guessed write); a no-title create_page stays
(the rules: ask for the name). Every ask case is tagged meta.ask_family (the dataset's ask cap floors). report.json
`ask_rule.pool`.

Re-authoring (r7 §5.3): trigger.reauthor (or $SOUSCHEF_REAUTHOR, comma list) names addresses that are targets again
(reason `reauthor`, MIN_POSITIVE_REAUTHOR kept positives) although the live model's training manifest already has
them; the earlier rounds' authored cases for them that name no section instance or no "second" twin
(author.ambiguity), or that write a value the editor cannot hold (author.bad_value: an invented tag, free text on
the Pack filter setting), leave this round's trainable set (supersede_ambiguous; meta.superseded_by; report.json
`ambiguous_superseded`) — the teacher asked "which one?" on those and kept no row, or kept a wrong one.

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


# "put it in the menu / header / top nav" — the PLACE the page goes, not a page ABOUT the menu ("create a menu page",
# "a page for our drinks menu" set no in_nav)
MENU_WORDS = (r"\b(?:in|into|to|on|from)\s+(?:the\s+|my\s+|our\s+)?(?:site'?s?\s+|main\s+|top\s+)?"
              r"(?:menu|nav(?:igation)?(?:\s+bar)?|navbar|header)\b|\b(?:header|top|main)\s+(?:menu|nav(?:igation)?|bar)\b"
              r"|\bnavbar\b|\bnavigation\b")


def relabel_page_create(c: dict) -> dict | None:
    """The case relabelled for create_page (fix_page_creates), or None when it is not a page-create request filed
    as where=pages."""
    import copy
    import re
    from ..exam import AMENDED_ID_SUFFIX, page_request, page_title_of, slugify
    try:  # r7 §4: an amended id may carry any revision (.k2, .k3 …)
        from ..exam import AMENDED_ID_RE
    except ImportError:  # an exam.py before r7
        AMENDED_ID_RE = re.compile(r"\.k\d+$")
    e = c.get("expected") or {}
    oos = e.get("out_of_scope") or {}
    if (c.get("score") or {}).get("mode") != "where" or "pages" not in (oos.get("expected_where") or e.get("where") or []):
        return None
    text = str(c.get("request") or "")
    if page_request(text) != "create":
        return None
    c2 = copy.deepcopy(c)
    meta = c2.setdefault("meta", {})
    meta["relabelled"] = {"from": {"mode": "where", "kind": c.get("kind"), "out_of_scope": copy.deepcopy(oos)},
                          "why": "create_page (2026-10-15): creating a page is supported"}
    meta["capability"] = "create_page"
    meta["fixed"] = sorted(set(list(meta.get("fixed") or []) + ["create_page"]))
    e2 = c2["expected"]
    e2["out_of_scope"] = None
    e2.pop("where", None)
    title = page_title_of(text)
    if title is None:  # no name: the rules say ask for it
        c2["score"] = {"mode": "ask"}
        c2["kind"] = "ask"
        meta["oracle_ops"] = []
    else:
        spec = {"expect": "create", "title_match": title["match"], "in_nav": bool(re.search(MENU_WORDS, text, re.I)),
                "status": "draft" if re.search(r"\bdraft\b", text, re.I) else "published", "sections": [],
                "if_exists": "decline"}
        if title.get("title"):
            spec["title"] = title["title"]
            name = title["title"]
        else:
            spec["words"] = title["words"]
            name = " ".join(w.capitalize() for w in title["words"][:3])
        e2["page"] = spec
        op = {"op": "create_page", "title": name}
        if spec["in_nav"]:
            op["in_nav"] = True
        if spec["status"] != "published":
            op["status"] = spec["status"]
        e2["ops"] = [op]
        meta["oracle_ops"] = [copy.deepcopy(op)]
        meta.setdefault("oracle_slug", slugify(name))
        c2["score"] = {"mode": "page"}
        c2["kind"] = "create_page"
    if c2.get("holdout") and not AMENDED_ID_RE.search(str(c2.get("id"))):
        meta["superseded"] = {"id": c.get("id")}
        c2["id"] = f"{c.get('id')}{AMENDED_ID_SUFFIX}"
    return c2


def fix_page_creates(cases: list[dict], policy: str = "on") -> dict:
    """Relabel the page-create requests filed as where=pages (module docstring) in place; the report. A relabelled
    case that would teach an exam row's page answer (contrast.page_exam_clash: p137.k2 "Events") is DROPPED
    (`exam_clash`), as contrast drops its own."""
    if policy in ("off", "0", "false", "no"):
        return {"policy": policy, "count": 0, "ids": [], "exam_clash": []}
    from ..contrast import page_exam_clash
    ids, modes, clash = [], Counter(), []
    keep: list[dict] = []
    for c in cases:
        c2 = relabel_page_create(c)
        if c2 is None:
            keep.append(c)
            continue
        if page_exam_clash(c2):
            clash.append(str(c.get("id")))
            continue
        keep.append(c2)
        ids.append(str(c.get("id")))
        modes[c2["score"]["mode"]] += 1
    cases[:] = keep
    return {"policy": policy, "count": len(ids), "ids": ids[:60], "modes": dict(modes), "exam_clash": clash,
            "holdouts": sum(1 for c in cases if c.get("holdout") and "create_page" in ((c.get("meta") or {}).get("fixed") or []))}


def check_ask_cases(cases: list[dict], m: Manifest) -> dict:
    """The ask rule on the trainable `ask` cases already held (r7 §1.2; module docstring). In place; the report."""
    import copy
    from .. import askrule as AR
    relabelled, dropped, unsure, kept, exempt = [], [], [], 0, 0
    families: Counter = Counter()
    keep: list[dict] = []
    for c in cases:
        if c.get("holdout") or (c.get("score") or {}).get("mode") != "ask":
            keep.append(c)
            continue
        meta = c.setdefault("meta", {})
        if meta.get("capability") == "create_page" or meta.get("ask_reason") == "which_one":
            meta.setdefault("ask_family", AR.ask_family_of(c, m))
            families[meta["ask_family"]] += 1
            exempt += 1
            keep.append(c)
            continue
        v = AR.classify_case(c, m)
        if v.vague:
            meta.setdefault("ask_family", AR.ask_family_of(c, m))
            families[meta["ask_family"]] += 1
            meta["ask_rule"] = {"kind": v.kind, "why": v.why}
            kept += 1
            keep.append(c)
            continue
        tgt = c.get("target") if isinstance(c.get("target"), dict) else None
        keys = [{"target": copy.deepcopy(tgt), "key": str(k), "list": ""} for k in c.get("fields") or [] if k]
        if tgt is None or not keys:
            dropped.append(str(c.get("id")))
            continue
        fields = AR._case_fields(c, m)
        field = fields[0] if len(keys) == 1 and len(fields) == 1 else None
        evidence = AR.write_evidence(c.get("request"), field, verdict=v)
        if evidence is None:  # clear by the rule, but no evidence for which write (r7 review): out, never a guess
            unsure.append(str(c.get("id")))
            continue
        c2 = copy.deepcopy(c)
        e = c2.setdefault("expected", {})
        e.update(keys=keys, ops=[], diff=[], alt=[], alt_exact=[])
        if field is not None:
            d = AR.direction_of(c2.get("request"), field.type, field.key)
            if d:
                e["direction"] = {"key": keys[0]["key"], "dir": d, "from": None}
        c2["score"] = {"mode": "keys"}
        c2["kind"] = "relabelled"
        m2 = c2["meta"]
        m2["oracle_ops"] = []
        m2.pop("ask_family", None)
        m2["ask_rule"] = {"kind": v.kind, "why": v.why, "from_mode": "ask", "evidence": evidence}
        m2["fixed"] = sorted(set(list(m2.get("fixed") or []) + ["ask_rule"]))
        relabelled.append(str(c.get("id")))
        keep.append(c2)
    cases[:] = keep
    return {"asks": kept + exempt + len(relabelled) + len(dropped) + len(unsure), "kept_vague": kept, "exempt": exempt,
            "relabelled": len(relabelled), "relabelled_ids": relabelled[:50], "dropped": len(dropped) + len(unsure),
            "dropped_ids": (dropped + unsure)[:50], "unsure": len(unsure), "by_ask_family": dict(families)}


def reauthor_list(run_file: dict, env: dict | None = None) -> list[str]:
    """trigger.reauthor (a list or a comma string), else $SOUSCHEF_REAUTHOR: the addresses re-authored this round."""
    env = os.environ if env is None else env
    v = env.get("SOUSCHEF_REAUTHOR") or ((run_file or {}).get("trigger") or {}).get("reauthor") or []
    items = v if isinstance(v, list) else str(v).split(",")
    return sorted({str(a).strip() for a in items if str(a).strip()})


def supersede_ambiguous(cases: list[dict], m: Manifest, reauthor: list[str] | set[str], run_id: str) -> tuple[list[dict], dict]:
    """(the cases without the earlier rounds' authored cases for a re-authored address that author.ambiguity or
    author.bad_value rejects, the report). Holdouts and other cases stay; the superseded ones are only left out of this round (the pool file
    keeps them). The clear ones that stay take the re-authored split group (author.author_group: one per case, not
    one per address — `regrouped`), as this round's new drafts do: a target whose address group hashes into val
    trained nothing in r6 (perksIcons.buttonOnImageOffset: 9 kept rows, all in val). The group is not part of the
    carry's case fingerprint, so a regrouped case's earlier row is still carried."""
    want = set(reauthor or ())
    if not want:
        return cases, {"count": 0, "by_reason": {}, "ids": [], "regrouped": 0}
    keep, gone = [], []
    why: Counter = Counter()
    regrouped = 0
    for c in cases:
        meta = c.get("meta") or {}
        if c.get("holdout") or meta.get("target_address") not in want or not meta.get("author"):
            keep.append(c)
            continue
        amb = A.ambiguity(c, m) or A.bad_value(c, m)
        if amb is None:
            g = A.author_group({"address": meta["target_address"], "reason": "reauthor"}, str(c.get("id")))
            if meta.get("group") != g:
                meta["group"] = g
                regrouped += 1
            keep.append(c)
            continue
        meta["superseded_by"] = run_id
        why[amb.split(":")[0]] += 1
        gone.append(str(c.get("id")))
    return keep, {"count": len(gone), "by_reason": dict(why), "ids": gone[:100], "regrouped": regrouped,
                  "by_address": dict(Counter(str((c.get("meta") or {}).get("target_address"))
                                             for c in cases if str(c.get("id")) in set(gone)))}


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
    pages_rep = fix_page_creates(kept, switch(run_file, "SOUSCHEF_PAGE_RELABEL", "page_relabel", "on"))
    if pages_rep["count"]:
        progress(f"prepare: {pages_rep['count']} page-create case(s) now expect create_page {pages_rep['modes']}")
    ask_pool_rep = check_ask_cases(kept, m_new)
    if ask_pool_rep["relabelled"] or ask_pool_rep["dropped"]:
        progress(f"prepare: ask rule on the pool's ask cases: {ask_pool_rep['relabelled']} now write, "
                 f"{ask_pool_rep['dropped']} dropped, {ask_pool_rep['kept_vague']} stay ask")
    reauthor = [] if smoke else reauthor_list(run_file)
    kept, sup_rep = supersede_ambiguous(kept, m_new, reauthor, run_id)
    if sup_rep["count"]:
        progress(f"prepare: {sup_rep['count']} earlier authored case(s) of re-authored keys name no instance / twin "
                 f"{sup_rep['by_reason']}: left out of this round")

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
    targets, trep = A.find_targets(m_train_doc, m_new, pool=kept, dropped_writes=dropped_writes, extra=extra_keys,
                                   reauthor=set(reauthor))
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
              "page_creates_fixed": pages_rep,
              "ask_rule": {"pool": ask_pool_rep, "ask_first": {k: crep.get("ask_first", {}).get(k) for k in
                                                               ("kept_vague", "relabelled", "twins", "dropped_rule",
                                                                "twins_dropped", "first_turn_kinds")}
                           if isinstance(crep.get("ask_first"), dict) else None,
                           "vague": crep.get("vague"), "vague_twins": crep.get("vague_twins")},
              "reauthor": reauthor, "ambiguous_superseded": sup_rep,
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
               "page_creates_fixed": pages_rep["count"], "ask_pool_relabelled": ask_pool_rep["relabelled"],
               "ambiguous_superseded": sup_rep["count"],
               "anthropic_usd": arep.get("anthropic_usd", 0.0)}
    return write_result(store, run_id, "prepare", "ok", metrics=metrics,
                        artifacts=[f"{pre}/prepare/cases.jsonl", f"{pre}/prepare/report.json", rel_auth])
