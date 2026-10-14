"""eval_stage — the paired eval of a candidate (design §3.8): job `souschef-eval`.

One local service (builder B's harness.Harness: the deployed binary, a throwaway
Postgres, no Anthropic key) with two messages-compatible routes and no fallback:
`current` → the eval GPU service (the production image) and `candidate` → the
candidate GPU service. The arms run concurrently, each on its own GPU, on the
same rows and contexts:

  set                                                    contexts  replicates  bar
  real-phrasing, 6 sets, rebased onto the deployed       rederived 2           yes
  manifest (examrebase.rebase_exam; stale rows apart)
  real-phrasing as recorded (the exam's own manifest)    recorded  1           reported
  F0H rebased (prepare), split :unseen / :seen           built     1           :unseen
  F9H new-key holdout (every round's exam-ext, rebased)  built     2           when n >= 10
  new-key probe (drift.probe_rows on the drift run's     built     2           when keys were dodged
  new keys)
  the NEWEST deployed manifest, when it moved during     rederived 1           reported (stale_round)
  the round with an R3 change

Outputs: runs/<id>/eval/<arm>/<set>-k<k>.jsonl (trimmed rows), eval/bar.json,
eval/candidate-baseline.json (the candidate's rows on its own training manifest
= the deployed one, in baseline.py's format: on Approve it becomes
baselines/<candidate digest>/baseline.json), then report.py writes report.json,
decision.json and the candidate_ready notification.

DRY (`--dry`): every arm is the lab's fake service (oracle scripts; the current
arm is scripted from the rows on its own training manifest, as the drift check's
DRY does), no harness, no GPU.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .. import LAB_DIR
from ..manifest import Manifest
from .common import ConfigError, StageError, Store, load_run, model_state, run8, run_prefix, write_jsonl

ARMS = ("current", "candidate")
EXAM_SETS = ("paraphrase", "holdout", "real", "followup", "out_of_scope", "out_of_scope_holdout")
SET_CODE = {"paraphrase": "pa", "holdout": "ho", "real": "re", "followup": "fu", "out_of_scope": "oo",
            "out_of_scope_holdout": "oh", "F9H": "f9", "probe": "pr", "F0H": "f0", "newest": "nw"}


def row_key(set_name: str, row_id: Any) -> str:
    return f"{set_name}:{row_id}"


def trim(row: dict, case: dict, m: Manifest, set_name: str) -> dict:
    """drift.trim plus the out-of-scope `where` flag the bar reads."""
    from ..drift import trim as dtrim
    t = dtrim(row, case, m)
    t["key"] = row_key(set_name, case.get("id"))
    t["set"] = set_name
    t["where_ok"] = int(row.get("verdict") == "unsupported_ok" and row.get("scored_by") == "where")
    t["scored_by"] = row.get("scored_by")
    return t


class Arms:
    """Where the arms' turns go: the harness (one local service, a route per arm) or, DRY, a fake per arm."""

    def __init__(self, *, harness=None, dry: bool = False, engine_path: str | None = None, workers: int = 2,
                 progress: Callable[[str], None] = print, tolerate_errors: frozenset[str] = frozenset()):
        self.harness, self.dry, self.engine_path = harness, dry, engine_path
        self.workers = max(1, workers)
        self.progress = progress
        # arms whose error rows are recorded (self.tolerated, eval/info.json), not fatal: SMOKE's candidate, a few
        # updates from the base, may well fail rows; the current arm's errors still mean harness or GPU trouble
        self.tolerate_errors = frozenset(tolerate_errors)
        self.tolerated: list[dict] = []

    def check_errors(self, arm: str, out: list[dict], what: str) -> None:
        try:
            _check_errors(out, what)
        except StageError as e:
            if arm not in self.tolerate_errors:
                raise
            self.tolerated.append({"what": what, "error": str(e)})
            self.progress(f"eval: {e} (tolerated: SMOKE candidate)")

    @contextmanager
    def endpoint(self, arm: str, m: Manifest, scripts: dict[str, list]) -> Iterator[tuple[str, str]]:
        if not self.dry:
            yield self.harness.url, self.harness.key
            return
        if str(LAB_DIR) not in sys.path:
            sys.path.insert(0, str(LAB_DIR))
        from tests import fake_service as FS  # the tests' fake: a DRY run's stand-in, never imported by a real run
        svc = FS.FakeService(manifest=m.doc, key="eval-dry")
        svc.routes = [{"name": a, "kind": "messages-compatible", "fields_context": False, "fallback": None} for a in ARMS]
        svc.scripts.update(scripts)
        svc.start()
        try:
            yield svc.url, "eval-dry"
        finally:
            svc.stop()

    def engine(self, mpath: Path, size: int):
        from ..collect import DictEngine
        from ..engine import PersistentEngine
        pe = PersistentEngine(self.engine_path, manifest_path=mpath, size=size)
        return DictEngine(pe), pe

    def exam(self, arm: str, set_name: str, rows: list[dict], mpath: Path, m: Manifest, *, sample: int, prefix: str,
             script_rows: list[dict] | None = None, script_m: Manifest | None = None, home: dict | None = None) -> list[dict]:
        """Exam-style rows (keys scoring) once on `arm` → trimmed rows."""
        from .. import collect as C
        from ..drift import home_context
        from ..exam import oracle_script, run_row, vintage_state
        if not rows:
            return []
        scripts = {}
        if self.dry:
            by_id = {r["id"]: r for r in (script_rows or rows)}
            for r in rows:
                scripts[str(r.get("request") or "")] = oracle_script(by_id.get(r["id"], r), script_m or m)
        state = vintage_state(mpath, home or home_context(rows))
        assets = C.load_assets(C.DEFAULT_ASSETS)
        out: list[dict] = []
        lock = threading.Lock()
        pending = list(reversed(rows))
        errors: list[str] = []
        with self.endpoint(arm, m, scripts) as (url, key):
            engine, pe = self.engine(mpath, min(self.workers, len(rows)))
            try:
                client0, _ = C.connect(url, key, mpath, rows, sse=False, timeout_s=300, build_id="lab-eval")
                fields, _rec = C.resolve_fields("auto", mpath, client0, arm)

                def worker() -> None:
                    client = C.Client(url, key, timeout_s=300, build_id="lab-eval")
                    while True:
                        with lock:
                            if not pending or errors:
                                return
                            case = pending.pop()
                        try:
                            row = run_row(case, client, engine, state=state, route=arm, sample=sample, assets=assets,
                                          fields=fields, history_mode="live", site_prefix=prefix)
                        except C.FieldsContextMismatch as e:
                            with lock:
                                errors.append(str(e))
                            return
                        except Exception as e:  # noqa: BLE001 — one row's lab bug must not lose the arm
                            row = {"status": "error", "code": "lab_exception", "error": f"{type(e).__name__}: {e}"[:500]}
                        t = trim(row, case, m, set_name)
                        with lock:
                            out.append(t)

                threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(self.workers, len(rows)))]
                for th in threads:
                    th.start()
                for th in threads:
                    th.join()
            finally:
                pe.close()
        if errors:
            raise StageError(f"{arm}/{set_name}: {errors[0]}", reason="eval")
        self.check_errors(arm, out, f"{arm}/{set_name}#{sample}")
        self.progress(f"eval: {arm} {set_name}#{sample}: {sum(r['pass'] for r in out)}/{len(out)}")
        return out

    def generated(self, arm: str, set_name: str, cases: list[dict], mpath: Path, m: Manifest, *, sample: int,
                  prefix: str, logdir: Path) -> list[dict]:
        """Generated holdout cases (F0H, F9H) through r4collect (verify-3) → trimmed rows."""
        from .. import collect as C
        from .. import r4collect as RC
        from ..r4score import PASSING
        from ..state import load_context_fixture
        if not cases:
            return []
        scripts = RC.oracle_scripts(cases, prefix) if self.dry else {}
        log = logdir / f"{arm}-{set_name}-k{sample}.r4.jsonl"
        with self.endpoint(arm, m, scripts) as (url, key):
            from ..engine import PersistentEngine
            eng = PersistentEngine(self.engine_path, manifest_path=mpath, size=min(self.workers, len(cases)))
            try:
                client, _ = C.connect(url, key, mpath, cases, sse=False, timeout_s=300, build_id="lab-eval")
                fields, _ = C.resolve_fields("auto", mpath, client, arm)
                RC.run(cases, client=client, engine=eng, log=RC.Log(log), route=arm, label=f"{arm}-{set_name}-k{sample}",
                       fields=fields, manifest=m, extras=load_context_fixture(), assets=C.load_assets(C.DEFAULT_ASSETS),
                       site_prefix=prefix, workers=min(self.workers, len(cases)), max_seconds=None,
                       progress=lambda s: None)
            finally:
                eng.close()
        by_case = {c["id"]: c for c in cases}
        out = []
        for r in RC.read_index([log]):
            c = by_case.get(r["case_id"])
            if c is None:
                continue
            out.append({"key": row_key(set_name, r["case_id"]), "id": r["case_id"], "set": set_name,
                        "status": r.get("status"), "verdict": r.get("verdict"),
                        "pass": int(r.get("status") == "ok" and r.get("verdict") in PASSING),
                        "seen": bool((c.get("meta") or {}).get("key_seen")), "flags": r.get("flags"), "ghost": 0,
                        "where_ok": 0})
        self.check_errors(arm, out, f"{arm}/{set_name}#{sample}")
        self.progress(f"eval: {arm} {set_name}#{sample}: {sum(r['pass'] for r in out)}/{len(out)}")
        return out


def _check_errors(out: list[dict], what: str) -> None:
    errs = [r for r in out if r.get("status") != "ok"]
    if len(errs) > max(3, len(out) // 5):
        codes = Counter(r.get("code") or r.get("verdict") for r in errs)
        raise StageError(f"{what}: {len(errs)}/{len(out)} rows errored ({dict(codes.most_common(4))})", reason="eval_errors")


def results(rows: list[dict]) -> dict[str, int]:
    return {r["key"]: int(r["pass"]) for r in rows if r.get("status") == "ok"}


# --- the stage --------------------------------------------------------------------------------------------------------

def plan_sets(store: Store, run_id: str, *, mode: str, local: Path, th: dict, progress: Callable[[str], None]) -> dict:
    """Every set's rows and manifests (module docstring)."""
    from .. import drift as DR
    from ..examrebase import rebase_exam
    from ..exam import MANIFEST_FILE, load_set
    pre = run_prefix(run_id)
    smoke = mode == "smoke"
    lim = int(th.get("smoke_rows_per_set", 10)) if smoke else None
    dep_path = local / "manifest.json"
    if not store.download(f"{pre}/prepare/manifest.json", dep_path):
        raise ConfigError("prepare/manifest.json is missing")
    dep_doc = json.loads(dep_path.read_text(encoding="utf-8"))
    dep_m = Manifest(dep_doc)
    exam_doc = json.loads(Path(MANIFEST_FILE).read_text(encoding="utf-8"))
    exam_m = Manifest(exam_doc)
    sd = DR.static_diff(exam_doc, dep_doc)
    raw_rows = {s: load_set(s) for s in EXAM_SETS}
    out: dict[str, Any] = {"dep_path": dep_path, "dep_m": dep_m, "exam_m": exam_m, "exam_path": Path(MANIFEST_FILE),
                           "sets": {}, "recorded": {}, "stale": [], "renames": sd.get("renames") or []}
    for s, rows in raw_rows.items():
        rows = rows[:lim] if lim else rows
        rb, stale, rep = rebase_exam(rows, dep_m, sd.get("renames"), old_manifest=exam_m)
        DR.rederive(rb, dep_path, DR.home_context(rows) or DR.home_context(raw_rows["paraphrase"]))
        out["sets"][s] = rb
        out["recorded"][s] = [r for r in rows if r["id"] not in {x["id"] for x in stale}]
        out["stale"] += [row_key(s, r["id"]) for r in stale]
    cases_path = local / "cases.jsonl"
    if not store.download(f"{pre}/prepare/cases.jsonl", cases_path):
        raise ConfigError("prepare/cases.jsonl is missing")
    from .common import read_jsonl
    hold = [c for c in read_jsonl(cases_path) if c.get("holdout")]
    f0h = [c for c in hold if (c.get("meta") or {}).get("family") == "F0H"]
    f9h = [c for c in hold if (c.get("meta") or {}).get("family") == "F9H"]
    out["f0h"] = f0h[:lim] if lim else f0h
    out["f9h"] = f9h[:lim] if lim else f9h
    # the new-key probe on the drift run's new keys
    run = load_run(store, run_id)
    drift_run = (run.get("trigger") or {}).get("drift_run")
    rep = store.read_json(f"drift/{drift_run}/report.json") if drift_run else None
    new_keys = list(((rep or {}).get("impact") or {}).get("new_keys") or [])
    out["trigger_rules"] = list(((rep or {}).get("verdict") or {}).get("fired") or (run.get("trigger") or {}).get("verdict_rules") or [])
    out["dodged_keys"] = list(((rep or {}).get("verdict") or {}).get("dodged_keys") or [])
    out["probe"] = []
    if new_keys:
        from ..exam import vintage_state
        state = vintage_state(dep_path, DR.home_context(raw_rows["paraphrase"]))
        rows, coverage = DR.probe_rows(dep_m, new_keys, state=state, per_key=int(th.get("probe_cases_per_key", 4)),
                                       max_keys=int(th.get("probe_max_keys", 40)))
        out["probe"] = rows[:lim] if lim else rows
        out["probe_coverage"] = coverage
    # the newest deployed manifest, when it moved during the round with a hard change
    newest = store.read_json("state/deployed.json") or {}
    out["stale_round"] = False
    if newest.get("hash") and newest["hash"] != dep_m.hash:
        ndoc = store.read_json(f"manifests/{newest['hash']}.json")
        if ndoc:
            sd2 = DR.static_diff(dep_doc, ndoc)
            if (sd2.get("counts") or {}).get("R3"):
                out["stale_round"] = True
                npath = local / "manifest-newest.json"
                npath.write_text(json.dumps(ndoc), encoding="utf-8")
                nm = Manifest(ndoc)
                rows = [r for s in EXAM_SETS for r in raw_rows[s]][: (lim or 10**6)]
                rb, _st, _ = rebase_exam(rows, nm, DR.static_diff(exam_doc, ndoc).get("renames"), old_manifest=exam_m)
                DR.rederive(rb, npath, DR.home_context(raw_rows["paraphrase"]))
                for r in rb:
                    r["set"] = "newest"
                out["newest"] = {"path": npath, "m": nm, "rows": rb, "hash": nm.hash}
    progress(f"eval: rows {', '.join(f'{s} {len(v)}' for s, v in out['sets'].items())}; stale {len(out['stale'])}; "
             f"F0H {len(out['f0h'])}; F9H {len(out['f9h'])}; probe {len(out['probe'])}; stale_round {out['stale_round']}")
    return out


def run(store: Store, run_id: str, *, mode: str, current_url: str | None, candidate_url: str | None, workdir: Path,
        dry: bool = False, engine_path: str | None = None, workers: int = 2, harness_factory: Callable | None = None,
        progress: Callable[[str], None] = print) -> dict:
    from .. import baseline as BL
    from .. import pipeline_common as PC
    from . import bar as BAR
    from . import report as REP
    pre = run_prefix(run_id)
    smoke = mode == "smoke"
    local = Path(workdir)
    th = PC.load_thresholds(store.b)
    model = model_state(store)
    base_cur = store.read_json(f"baselines/{model.get('digest')}/baseline.json") if model.get("digest") else None
    sets = plan_sets(store, run_id, mode=mode, local=local, th=th, progress=progress)
    reps = 1 if smoke else 2
    dep_m, dep_path = sets["dep_m"], sets["dep_path"]
    # the current arm in DRY is scripted from the rows on the current model's own training manifest (it never saw the
    # deployed one); the candidate from the deployed rows
    cur_train_doc = store.read_json(f"manifests/{model['training_manifest']}.json") or sets["exam_m"].doc
    cur_m = Manifest(cur_train_doc)
    tag = run8(run_id)
    logdir = local / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    results_by: dict[str, dict[str, list[dict[str, int]]]] = {a: {} for a in ARMS}
    where_by: dict[str, list[dict[str, int]]] = {a: [] for a in ARMS}
    ghosts: dict[str, list[int]] = {a: [] for a in ARMS}
    newest_by: dict[str, list[dict[str, int]]] = {a: [] for a in ARMS}
    rows_out: dict[str, list[dict]] = {a: [] for a in ARMS}

    def arm_job(arms: Arms, arm: str) -> None:
        p = lambda k, s: f"lab-ev{tag}{arm[:2]}{k}{SET_CODE.get(s, s[:2])}-"  # noqa: E731 — unique per arm, replicate, set
        for k in range(1, reps + 1):
            g = 0
            wk: dict[str, int] = {}
            for s in EXAM_SETS:
                rows = sets["sets"][s]
                tr = arms.exam(arm, s, rows, dep_path, dep_m, sample=k, prefix=p(k, s),
                               script_rows=sets["recorded"][s] if arm == "current" else None,
                               script_m=cur_m if arm == "current" else None)
                rows_out[arm] += tr
                results_by[arm].setdefault(s, []).append(results(tr))
                g += sum(int(r.get("ghost") or 0) for r in tr)
                if s == "out_of_scope":
                    wk.update({r["key"]: int(r["where_ok"]) for r in tr if r.get("status") == "ok"})
            ghosts[arm].append(g)
            where_by[arm].append(wk)
            if sets["f9h"]:
                tr = arms.generated(arm, "F9H", sets["f9h"], dep_path, dep_m, sample=k, prefix=p(k, "f9"), logdir=logdir)
                rows_out[arm] += tr
                results_by[arm].setdefault("F9H", []).append(results(tr))
            if sets["probe"]:
                tr = arms.exam(arm, "probe", sets["probe"], dep_path, dep_m, sample=k, prefix=p(k, "pr"))
                rows_out[arm] += tr
                results_by[arm].setdefault("probe", []).append(results(tr))
        # once: recorded contexts on the exam's own manifest, F0H, the newest manifest
        for s in EXAM_SETS:
            tr = arms.exam(arm, f"{s}:recorded", sets["recorded"][s], sets["exam_path"], sets["exam_m"], sample=1,
                           prefix=p(9, "r" + SET_CODE[s]), script_rows=sets["recorded"][s], script_m=sets["exam_m"])
            rows_out[arm] += tr
            results_by[arm].setdefault(f"{s}:recorded", []).append(results(tr))
        if sets["f0h"]:
            tr = arms.generated(arm, "F0H", sets["f0h"], dep_path, dep_m, sample=1, prefix=p(9, "f0"), logdir=logdir)
            rows_out[arm] += tr
            for part, seen in (("unseen", False), ("seen", True)):
                sub = [r for r in tr if bool(r.get("seen")) == seen]
                results_by[arm].setdefault(f"F0H:{part}", []).append(
                    {row_key(f"F0H:{part}", r["id"]): int(r["pass"]) for r in sub if r.get("status") == "ok"})
        if sets.get("newest"):
            nw = sets["newest"]
            tr = arms.exam(arm, "newest", nw["rows"], nw["path"], nw["m"], sample=1, prefix=p(9, "nw"))
            rows_out[arm] += tr
            newest_by[arm].append(results(tr))

    harness = None
    t0 = time.monotonic()
    try:
        if not dry:
            if not current_url or not candidate_url:
                raise ConfigError("--current-url and --candidate-url (the eval and candidate GPU services) are required")
            from ..harness import Harness
            harness = (harness_factory or Harness)({"current": current_url, "candidate": candidate_url}, run_id=f"ev{tag}",
                                                   workdir=local / "harness")
            harness.__enter__()
        arms = Arms(harness=harness, dry=dry, engine_path=engine_path, workers=workers, progress=progress,
                    tolerate_errors=frozenset({"candidate"}) if smoke else frozenset())
        errs: list[BaseException] = []

        def guarded(arm: str) -> None:
            try:
                arm_job(arms, arm)
            except BaseException as e:  # noqa: BLE001 — re-raised below
                errs.append(e)

        threads = [threading.Thread(target=guarded, args=(a,), daemon=True) for a in ARMS]
        for t_ in threads:
            t_.start()
        for t_ in threads:
            t_.join()
        if errs:
            raise errs[0]
    finally:
        if harness is not None:
            harness.__exit__(None, None, None)
    wall_s = time.monotonic() - t0
    for a in ARMS:
        write_jsonl(local / "eval" / a / "rows.jsonl", rows_out[a])
    probe_by_key: dict[str, dict[str, list[dict[str, int]]]] = {}
    for p_row in sets["probe"]:
        pk = (p_row.get("meta") or {}).get("probe_key")
        for a in ARMS:
            reps_a = []
            for rep_rows in results_by[a].get("probe") or []:
                key = row_key("probe", p_row["id"])
                if key in rep_rows:
                    reps_a.append({key: rep_rows[key]})
            probe_by_key.setdefault(pk, {}).setdefault(a, [])
            if reps_a:
                if not probe_by_key[pk][a]:
                    probe_by_key[pk][a] = [dict() for _ in reps_a]
                for i, d in enumerate(reps_a):
                    if i < len(probe_by_key[pk][a]):
                        probe_by_key[pk][a][i].update(d)
    bar = BAR.evaluate(results_by, baseline=base_cur, th=th, trigger_rules=sets["trigger_rules"], ghosts=ghosts,
                       oos_where=where_by, dodged_keys=sets["dodged_keys"], probe_by_key=probe_by_key)
    if sets.get("newest"):
        nb = BAR.set_row("newest", "rederived", newest_by["current"], newest_by["candidate"], base_cur, th, bar=False)
        nb.pop("_keys", None)
        bar["sets"].append(nb)
    store.write_json(f"{pre}/eval/bar.json", bar)
    # the candidate's baseline on its own training manifest (baseline.py's format)
    img = store.read_json(f"{pre}/model/image.json") or {}
    cand_digest = img.get("digest") or f"dry:{run_id}"
    gf = (load_run(store, run_id).get("deployed") or {}).get("generated_from")
    cb = BL.new(digest=cand_digest, tag=img.get("tag") or f"c-{run_id}", training_manifest=dep_m.hash,
                engine_sha=gf.get("sha") if isinstance(gf, dict) else None,
                service_build=(harness.build if harness is not None else "dry"), source="candidate-eval")
    # drift's F0H addressing: "F0H:<case id>" (the unseen / seen split is ours)
    f0h_c = {f"F0H:{k.split(':', 2)[2]}": v for part in ("F0H:unseen", "F0H:seen")
             for k, v in ((results_by["candidate"].get(part) or [{}])[0]).items()}
    for k in range(reps):
        rows_k: dict[str, int] = {}
        for s in EXAM_SETS:
            if k < len(results_by["candidate"].get(s) or []):
                rows_k.update(results_by["candidate"][s][k])
        BL.add_replicate(cb, rows_k, run=run_id, at=PC.iso(), ghost_writes=ghosts["candidate"][k] if k < len(ghosts["candidate"]) else None,
                         f0h=f0h_c if k == 0 else None)
    cb["calibrated"] = len(cb["replicates"]) >= 2
    probe_c = {}
    for pk, arms_ in probe_by_key.items():
        cr = arms_.get("candidate") or []
        probe_c[pk] = {"n": sum(len(d) for d in cr), "pass": sum(sum(d.values()) for d in cr)}
    cb["probe"] = probe_c
    store.write_json(f"{pre}/eval/candidate-baseline.json", cb)
    store.sync_up(local / "eval", f"{pre}/eval")
    store.sync_up(logdir, f"{pre}/eval/logs")
    eval_info = {"wall_s": round(wall_s), "replicates": reps, "stale_rows": sets["stale"], "stale_round": sets["stale_round"],
                 "trigger_rules": sets["trigger_rules"], "dodged_keys": sets["dodged_keys"],
                 "newest_manifest": (sets.get("newest") or {}).get("hash"),
                 "probe_coverage": sets.get("probe_coverage"), "renames": sets["renames"],
                 "harness": harness.describe() if harness is not None else {"dry": True},
                 "candidate_baseline_digest": cand_digest, "tolerated_errors": arms.tolerated}
    store.write_json(f"{pre}/eval/info.json", eval_info)
    return REP.build(store, run_id, mode=mode, bar=bar, eval_info=eval_info)
