"""collect_stage — the teacher collection (design §3.3): job `souschef-collect`.

r4collect run at its core, driven the way scratchpad/r4/collect.sh drove it:

  carry    (full mode, default on; retrain/carry.py) the rows an earlier round accepted that still apply on
           this manifest are re-checked offline (engine replay + r4score, no teacher) and logged as
           pass0-carry.jsonl; the passes below ask only the cases it did not carry (new, changed, failing).
           Off: run(carry="off"), $COLLECT_CARRY=off, or run.json trigger.carry "off" / trigger.reask_all
           true (an execution's args.trigger) — every case is asked again. A carry that cannot run
           (no source, a missing tool) falls back to asking everything and says why in the result.
  smoke    7–8 cases, one per trainable family (F2 a follow-up), route default, then
           r4dataset --render-check + tools/r4_smoke_check.py must say OK: every row
           rendered from the STORED conversation with no block drift. Anything else
           stops the stage before more is spent (block drift = int is not serving the
           retrieval this image renders with)
  pass1    every trainable case on route `default` (Claude Sonnet 5.5)
  pass2    what pass 1 left without a usable row, on route `opus`
  plan3    the still-unusable cases, each on its family's better route
  pass3    pass3-opus, then pass3-default
  (SMOKE mode: the smoke gate and pass1 only)

Teacher pin: before every chunk the int service's /version build id must equal the
lab image's baked /opt/sous-chef/BUILD_ID (= the deployed service tag the round's
prompt is rendered with). A change ends the stage with result status
`teacher_changed` (exit 0): the workflow re-resolves the images and restarts once
with `--attempt 2` (a new log folder and site prefix — rows of two builds are
never mixed).

Resumable: logs live in runs/<id>/collect/a<attempt>/ (pulled back at start, pushed
every 60 s and after every chunk); a retried task skips every case its log already
answered. Chunks start new cases for 420 s (r4collect exit 3 = continue). Site ids
`lab-<run8><attempt><pass>-<case id>`; the build id sent is `lab-<run id>` (a lab
registration by the deployed-manifest endpoint's rule). The key comes from
$SOUS_CHEF_KEY only (never argv).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .. import LAB_DIR
from .common import (ConfigError, StageError, Store, load_run, notification, read_jsonl, redact, run8, run_prefix, utc_now,
                     write_result)

CHUNK_S = 420.0
SYNC_S = 60.0
MAX_CHUNKS = 400
BUILD_FILE = "/opt/sous-chef/BUILD_ID"
PASS_ROUTES = (("pass1", "default"), ("pass2", "opus"))


class TeacherChanged(Exception):
    pass


def expected_build(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    p = Path(os.environ.get("SOUS_CHEF_BUILD_FILE") or BUILD_FILE)
    return p.read_text(encoding="utf-8").strip() if p.is_file() else None


def http_version(url: str) -> dict:
    """GET <url>/version (public: no key)."""
    with urllib.request.urlopen(url.rstrip("/") + "/version", timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def smoke_ids(cases_path: Path) -> list[str]:
    """One trainable case per family (F2: a follow-up), as collect.sh smoke chose them."""
    seen: dict[str, str] = {}
    for c in read_jsonl(cases_path):
        f = str((c.get("meta") or {}).get("family"))
        if c.get("holdout") or f in seen or (f == "F2" and not c.get("history")):
            continue
        seen[f] = c["id"]
    return [seen[k] for k in sorted(seen)]


class Syncer:
    """Pushes the local log folder to the store every `every` seconds (and on stop)."""

    def __init__(self, store: Store, local: Path, prefix: str, every: float = SYNC_S):
        self.store, self.local, self.prefix, self.every = store, local, prefix, every
        self.stop_ev = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.lock = threading.Lock()

    def _loop(self) -> None:
        while not self.stop_ev.wait(self.every):
            self.push()

    def push(self) -> None:
        with self.lock:
            try:
                self.store.sync_up(self.local, self.prefix)
            except StageError as e:  # a failed push is retried on the next tick; the final one raises
                print(f"collect: log push failed (will retry): {e}", file=sys.stderr)

    def __enter__(self) -> "Syncer":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop_ev.set()
        self.thread.join(timeout=5)
        with self.lock:
            self.store.sync_up(self.local, self.prefix)


class Collector:
    """One attempt's passes over a local case file, logs in `logs`."""

    def __init__(self, *, cases: Path, manifest: Path, logs: Path, url: str, run_id: str, attempt: int,
                 expect_build: str | None, fake: bool, workers: int, chunk_s: float, engine: str | None,
                 runner: Callable[[list[str]], int] | None = None, version: Callable[[str], dict] | None = None,
                 progress: Callable[[str], None] = print, sync: Callable[[], None] | None = None):
        from .. import r4collect
        self.cases, self.manifest, self.logs, self.url = cases, manifest, logs, url
        self.run_id, self.attempt, self.expect_build, self.fake = run_id, attempt, expect_build, fake
        self.workers, self.chunk_s, self.engine = workers, chunk_s, engine
        self.runner = runner or r4collect.main
        self.version = version or http_version
        self.progress = progress
        self.sync = sync or (lambda: None)
        self.prefix_base = f"lab-{run8(run_id)}{attempt}"
        self.chunks = 0

    def check_teacher(self) -> dict | None:
        if self.fake:
            return None
        v = self.version(self.url)
        b = str(v.get("build_id") or "")
        if self.expect_build and b != self.expect_build:
            raise TeacherChanged(f"{self.expect_build} → {b}")
        return v

    def preflight(self) -> dict:
        if self.fake:
            return {"fake": True}
        if not self.expect_build:
            raise ConfigError("no expected teacher build: /opt/sous-chef/BUILD_ID is missing (the lab image bakes it)")
        if not os.environ.get("SOUS_CHEF_KEY"):
            raise ConfigError("$SOUS_CHEF_KEY is empty (the job reads sous-chef-pipeline-service-key)")
        v = self.check_teacher() or {}
        ri = v.get("route_info") or {}
        bad = [r for r in ("default", "opus") if (ri.get(r) or {}).get("fields_context") is not True]
        if bad:
            raise ConfigError(f"/version: fields_context is not on for route(s) {bad}: the teacher would not read the "
                              f"candidate-fields block the student trains on")
        return {"build_id": v.get("build_id"), "routes": {r: ri.get(r) for r in ("default", "opus")}}

    def chunk(self, label: str, route: str, prefix: str, extra: list[str]) -> int:
        self.check_teacher()
        argv = ["run", "--cases", str(self.cases), "--log", str(self.logs / f"{label}.jsonl"), "--label", label,
                "--route", route, "--manifest", str(self.manifest), "--workers", str(self.workers),
                "--max-seconds", str(self.chunk_s), "--site-prefix", prefix, "--build-id", f"lab-{self.run_id}"[:64]]
        if self.engine:
            argv += ["--engine", self.engine]
        argv += (["--fake", "--oracle"] + (["--fake-blocks"] if _fields_available() else [])) if self.fake else ["--url", self.url]
        self.chunks += 1
        rc = self.runner(argv + extra)
        self.sync()
        return rc

    def run_pass(self, label: str, route: str, prefix: str, extra: list[str]) -> None:
        for i in range(MAX_CHUNKS):
            rc = self.chunk(label, route, prefix, extra)
            if rc == 0:
                self.progress(f"collect: {label} done ({i + 1} chunk(s))")
                return
            if rc == 3:
                continue
            if rc == 2:
                raise ConfigError(f"r4collect {label} refused to start (exit 2)", reason="collect_config")
            raise StageError(f"r4collect {label} failed (exit {rc})", reason="collect")
        raise StageError(f"{label}: still cases left after {MAX_CHUNKS} chunks", reason="collect_chunks")

    def smoke_gate(self) -> dict:
        from .. import r4dataset
        ids = smoke_ids(self.cases)
        idf = self.logs / "smoke.ids"
        idf.write_text("\n".join(ids) + "\n", encoding="utf-8")
        self.run_pass("smoke", "default", f"{self.prefix_base}s-", ["--ids-file", str(idf)])
        rep = r4dataset.render_check(self.cases, [self.logs / "smoke.jsonl"], self.manifest)
        (self.logs / "smoke-check.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        sys.path.insert(0, str(LAB_DIR / "tools"))
        import r4_smoke_check as SC
        problems = SC.problems_of(rep, set(ids))
        return {"ids": ids, "problems": problems,
                **{k: rep.get(k) for k in ("rows", "ok_rows", "rendered", "render_sources", "block_drift_rows", "errors")}}

    def carried_log(self) -> Path | None:
        """pass0-carry.jsonl when the carry step logged rows (its index is written last: present = whole)."""
        from .. import r4collect
        from .carry import CARRY_LOG
        log = self.logs / CARRY_LOG
        idx = r4collect.index_path(log)
        return log if idx.is_file() and idx.read_text(encoding="utf-8").strip() else None

    def passes(self, smoke: bool) -> list[str]:
        from .. import r4collect
        logs = self.logs
        carried = self.carried_log()
        prior = ["--prior", str(carried)] if carried else []
        self.run_pass("pass1", "default", f"{self.prefix_base}p1-", ["--select", "unusable", *prior] if carried else [])
        if smoke:
            return ["pass1"]
        self.run_pass("pass2", "opus", f"{self.prefix_base}p2-", ["--select", "unusable", "--prior", str(logs / "pass1.jsonl"),
                                                                    *prior])
        rc = r4collect.main(["plan3", "--cases", str(self.cases), "--log", str(logs / "pass1.jsonl"), "--log",
                             str(logs / "pass2.jsonl"), *(["--log", str(carried)] if carried else []),
                             "--out-dir", str(logs)])
        if rc != 0:
            raise StageError(f"r4collect plan3 failed (exit {rc})", reason="collect")
        done = ["pass1", "pass2"]
        for route in ("opus", "default"):
            ids = logs / f"pass3-{route}.ids"
            if not ids.is_file() or not ids.read_text(encoding="utf-8").strip():
                continue
            self.run_pass(f"pass3-{route}", route, f"{self.prefix_base}p3{route[0]}-",
                          ["--ids-file", str(ids), "--select", "unusable", "--prior", str(logs / "pass1.jsonl"),
                           "--prior", str(logs / "pass2.jsonl"), *prior])
            done.append(f"pass3-{route}")
        return done


def usd_per_case(store: Store, sources: list[dict]) -> float:
    """The teacher's cost of one asked case in the newest source round that recorded it (result-collect.json
    teacher_usd / the cases it asked), else carry.USD_PER_CASE_DEFAULT (r5's)."""
    from .carry import USD_PER_CASE_DEFAULT
    for src in sources:
        rid = src.get("run_id") or (str(src.get("dataset") or "").split("/")[1] if str(src.get("dataset") or "").startswith("runs/") else None)
        if not rid:
            continue
        m = (store.read_json(f"{run_prefix(rid)}/result-collect.json") or {}).get("metrics") or {}
        asked = int(m.get("cases") or 0) - int(m.get("carried") or 0)
        if asked > 0 and m.get("teacher_usd"):
            return round(float(m["teacher_usd"]) / asked, 4)
    return USD_PER_CASE_DEFAULT


def carry_step(store: Store, run_id: str, *, mode: str, cases: Path, manifest: Path, logs: Path, engine: str | None,
               workers: int, explicit: str | None = None, progress: Callable[[str], None] = print,
               carry_fn: Callable[..., dict] | None = None, tools: dict | None = None) -> dict:
    """The carry step (module docstring): {on, why, …the report's counts}. A carry that cannot run leaves no carry
    log, so every case is asked as before, and says why — unless the switch says "required" (carry.required), when
    it raises StageError(reason carry_required) before any teacher call. A bad carry_hidden value is a ConfigError.
    `carry_fn` / `tools` (prefix_rules, fields_for, engine) are the tests' seams."""
    from . import carry as K
    run_file = load_run(store, run_id)
    on, why = K.setting(run_file, mode, explicit)
    must = on and K.required(run_file, explicit)
    try:
        hidden = K.hidden_strict(run_file)
    except ValueError as e:
        raise ConfigError(f"carry: {e}", reason="collect_config") from e
    out: dict[str, Any] = {"on": on, "why": why, **({"required": True} if must else {})}
    if not on:
        progress(f"collect: carry off ({why}): every case is asked")
        return out
    log, idx, rep = logs / K.CARRY_LOG, logs / (K.CARRY_LOG + ".index.jsonl"), logs / K.REPORT
    if idx.is_file() and rep.is_file():  # a retried task: the step already ran for this attempt
        prev = json.loads(rep.read_text(encoding="utf-8"))
        return {**out, "reused": True, **{k: v for k, v in prev.items()
                                          if k not in ("per_case", "examples", "carried_hidden_write_ids")}}
    engine_obj = None
    try:
        sources = K.find_sources(store, run_id)
        if not sources:
            if must:
                raise RuntimeError("no earlier round's dataset to carry from")
            progress("collect: carry: no earlier round's dataset to carry from: every case is asked")
            return {**out, "sources": []}
        all_cases = read_jsonl(cases)
        ids = {str(c["id"]) for c in all_cases if not c.get("holdout")}
        loaded = [K.load_source(store, s, logs.parent / "carry-src" / str(i), ids) for i, s in enumerate(sources)]
        tools = dict(tools or {})
        if "prefix_rules" not in tools or "fields_for" not in tools:
            from .. import fields as F
            tools.setdefault("prefix_rules", F.Prefix(manifest).rules)
            tools.setdefault("fields_for", lambda p: F.Fields(p))
        if tools.get("engine") is None:
            from ..engine import PersistentEngine, default_cache_dir
            engine_obj = PersistentEngine(engine, manifest_path=manifest, size=max(1, workers), cache_dir=default_cache_dir())
        from .. import collect as C
        from ..manifest import Manifest
        from ..state import load_context_fixture
        m_new = Manifest(json.loads(Path(manifest).read_text(encoding="utf-8")))
        report = (carry_fn or K.carry)(all_cases, loaded, m_new=m_new, engine=tools.get("engine") or engine_obj,
                                       extras=load_context_fixture(), assets=C.load_assets(C.DEFAULT_ASSETS), logs=logs,
                                       rules=tools["prefix_rules"], fields_for=tools["fields_for"],
                                       exclude=K.excluded_ids(run_file), usd_per_case=usd_per_case(store, sources),
                                       workers=max(1, workers), hidden_check=hidden, progress=progress)
    except Exception as e:  # noqa: BLE001 — carry is an optimisation: on any failure the round asks everything
        for p in (idx, log, rep):
            p.unlink(missing_ok=True)
        if must:
            raise StageError(f"carry is required (trigger.carry / $COLLECT_CARRY \"required\") and failed: "
                             f"{type(e).__name__}: {redact(e)}"[:500], reason="carry_required") from e
        progress(f"collect: carry failed ({type(e).__name__}: {redact(e)}): every case is asked")
        return {**out, "on": False, "why": f"error: {type(e).__name__}: {redact(e)}"[:500]}
    finally:
        if engine_obj is not None:
            engine_obj.close()
        import shutil
        shutil.rmtree(logs.parent / "carry-src", ignore_errors=True)
    progress(f"collect: carried {report['carried']} of {report['cases']} case(s); asking {report['asked']} "
             f"(≈ US${report['teacher_usd_saved_estimate']} of teacher spend not asked)")
    return {**out, **{k: v for k, v in report.items() if k not in ("per_case", "examples", "carried_hidden_write_ids")}}


def _fields_available() -> bool:
    from .. import fields
    return fields.available()


def pass_logs(logs: Path) -> list[Path]:
    return sorted(p for p in Path(logs).glob("pass*.jsonl") if not p.name.endswith(".index.jsonl"))


def run(store: Store, run_id: str, *, mode: str, attempt: int, url: str | None, workdir: Path, fake: bool = False,
        expect_build: str | None = None, workers: int = 6, chunk_s: float = CHUNK_S, sync_s: float = SYNC_S,
        engine: str | None = None, runner: Callable | None = None, version: Callable | None = None,
        progress: Callable[[str], None] = print, carry: str | None = None,
        carry_fn: Callable[..., dict] | None = None, carry_tools: dict | None = None) -> dict:
    """carry: "on" / "off" overrides the carry switch (carry.setting); carry_fn / carry_tools are test seams."""
    from .. import r4collect
    from .. import r4score as R4S
    pre = run_prefix(run_id)
    smoke = mode == "smoke"
    local = Path(workdir)
    try:  # trigger.score_policy / font_policy → the scorer's environment (r4collect passes and carry inherit it)
        policy = R4S.adopt_trigger_policy(load_run(store, run_id).get("trigger"))
    except ValueError as e:
        raise ConfigError(f"scoring policy: {e}", reason="collect_config") from e
    cases = local / "cases.jsonl"
    manifest = local / "manifest.json"
    if not store.download(f"{pre}/prepare/cases.jsonl", cases) or not store.download(f"{pre}/prepare/manifest.json", manifest):
        raise ConfigError(f"{pre}/prepare/cases.jsonl or manifest.json is missing: run the prepare stage first")
    logs_prefix = f"{pre}/collect/a{attempt}"
    logs = local / f"a{attempt}"
    logs.mkdir(parents=True, exist_ok=True)
    store.sync_down(logs_prefix, logs)  # resume
    url = url or os.environ.get("SOUS_CHEF_URL") or ""
    if not url and not fake:
        raise ConfigError("no teacher URL: --url or $SOUS_CHEF_URL")
    expect = expected_build(expect_build) if not fake else None
    syncer = Syncer(store, logs, logs_prefix, every=sync_s)
    col = Collector(cases=cases, manifest=manifest, logs=logs, url=url, run_id=run_id, attempt=attempt, expect_build=expect,
                    fake=fake, workers=workers, chunk_s=chunk_s, engine=engine or os.environ.get("ENGINE"),
                    runner=runner, version=version, progress=progress, sync=syncer.push)
    metrics: dict[str, Any] = {"attempt": attempt, "teacher_build": expect, "mode": mode,
                               "score_policy": {**policy, "version": R4S.POLICY_VERSION}}
    try:
        with syncer:
            metrics["preflight"] = col.preflight()
            metrics["carry"] = carry_step(store, run_id, mode=mode, cases=cases, manifest=manifest, logs=logs,
                                          engine=engine or os.environ.get("ENGINE"), workers=workers, explicit=carry,
                                          progress=progress, carry_fn=carry_fn, tools=carry_tools)
            syncer.push()
            gate = col.smoke_gate()
            metrics["smoke_check"] = gate
            if gate["problems"]:
                note = notification("stage_failed", "error", "Sous Chef pipeline failed: collect (smoke check)",
                                    "The smoke collection did not render as the student will be trained: "
                                    + "; ".join(gate["problems"]) + ". Nothing more was spent. "
                                    "Playbook: docs/RETRAIN.md#r2-collection",
                                    {"stage": "collect", "reason": "smoke_check", "check": gate},
                                    dedup_key=f"{run_id}:collect")
                write_result(store, run_id, "collect", "smoke_failed", metrics=metrics, notification_=note,
                             extra={"logs_prefix": logs_prefix})
                raise ConfigError("smoke check failed: " + "; ".join(gate["problems"]), reason="smoke_check")
            metrics["passes"] = col.passes(smoke)
    except TeacherChanged as e:
        metrics["teacher_changed"] = str(e)
        note = notification("stage_failed", "warn", "Sous Chef pipeline: the teacher changed during collection",
                            f"The int service's build changed during the round ({e}). Rows from two builds are never "
                            f"mixed: the round restarts collection with the new build (attempt {attempt + 1}).",
                            {"stage": "collect", "reason": "teacher_changed", "attempt": attempt},
                            dedup_key=f"{run_id}:collect:teacher")
        return write_result(store, run_id, "collect", "teacher_changed", metrics=metrics, notification_=note,
                            extra={"logs_prefix": logs_prefix})
    st = r4collect.status(cases, pass_logs(logs))
    (logs / "status.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    store.write_json(f"{logs_prefix}/status.json", st)
    t = st.get("total") or {}
    cr = metrics.get("carry") or {}
    metrics.update(carried=int(cr.get("carried") or 0) if cr.get("on") else 0,
                   reasked=(int(cr.get("asked") or 0) if cr.get("on") and "asked" in cr else t.get("asked", 0)),
                   new_cases=int(cr.get("reask_new") or 0) if cr.get("on") else None)
    metrics.update(cases=t.get("cases", 0), usable=t.get("usable", 0), teacher_usd=st.get("cost_usd", 0.0),
                   chunks=col.chunks, families={k: {"cases": v.get("cases"), "usable": v.get("usable")}
                                                 for k, v in (st.get("families") or {}).items()})
    return write_result(store, run_id, "collect", "ok", metrics=metrics,
                        artifacts=[f"{logs_prefix}/{p.name}" for p in pass_logs(logs)],
                        extra={"logs_prefix": logs_prefix, "logs": [p.name for p in pass_logs(logs)], "finished_at": utc_now()})
