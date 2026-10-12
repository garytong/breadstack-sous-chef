"""collect_stage — the teacher collection (design §3.3): job `souschef-collect`.

r4collect run at its core, driven the way scratchpad/r4/collect.sh drove it:

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
from .common import ConfigError, StageError, Store, notification, read_jsonl, run8, run_prefix, utc_now, write_result

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

    def passes(self, smoke: bool) -> list[str]:
        from .. import r4collect
        logs = self.logs
        self.run_pass("pass1", "default", f"{self.prefix_base}p1-", [])
        if smoke:
            return ["pass1"]
        self.run_pass("pass2", "opus", f"{self.prefix_base}p2-", ["--select", "unusable", "--prior", str(logs / "pass1.jsonl")])
        rc = r4collect.main(["plan3", "--cases", str(self.cases), "--log", str(logs / "pass1.jsonl"), "--log",
                             str(logs / "pass2.jsonl"), "--out-dir", str(logs)])
        if rc != 0:
            raise StageError(f"r4collect plan3 failed (exit {rc})", reason="collect")
        done = ["pass1", "pass2"]
        for route in ("opus", "default"):
            ids = logs / f"pass3-{route}.ids"
            if not ids.is_file() or not ids.read_text(encoding="utf-8").strip():
                continue
            self.run_pass(f"pass3-{route}", route, f"{self.prefix_base}p3{route[0]}-",
                          ["--ids-file", str(ids), "--select", "unusable", "--prior", str(logs / "pass1.jsonl"),
                           "--prior", str(logs / "pass2.jsonl")])
            done.append(f"pass3-{route}")
        return done


def _fields_available() -> bool:
    from .. import fields
    return fields.available()


def pass_logs(logs: Path) -> list[Path]:
    return sorted(p for p in Path(logs).glob("pass*.jsonl") if not p.name.endswith(".index.jsonl"))


def run(store: Store, run_id: str, *, mode: str, attempt: int, url: str | None, workdir: Path, fake: bool = False,
        expect_build: str | None = None, workers: int = 6, chunk_s: float = CHUNK_S, sync_s: float = SYNC_S,
        engine: str | None = None, runner: Callable | None = None, version: Callable | None = None,
        progress: Callable[[str], None] = print) -> dict:
    from .. import r4collect
    pre = run_prefix(run_id)
    smoke = mode == "smoke"
    local = Path(workdir)
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
    metrics: dict[str, Any] = {"attempt": attempt, "teacher_build": expect, "mode": mode}
    try:
        with syncer:
            metrics["preflight"] = col.preflight()
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
    metrics.update(cases=t.get("cases", 0), usable=t.get("usable", 0), teacher_usd=st.get("cost_usd", 0.0),
                   chunks=col.chunks, families={k: {"cases": v.get("cases"), "usable": v.get("usable")}
                                                 for k, v in (st.get("families") or {}).items()})
    return write_result(store, run_id, "collect", "ok", metrics=metrics,
                        artifacts=[f"{logs_prefix}/{p.name}" for p in pass_logs(logs)],
                        extra={"logs_prefix": logs_prefix, "logs": [p.name for p in pass_logs(logs)], "finished_at": utc_now()})
