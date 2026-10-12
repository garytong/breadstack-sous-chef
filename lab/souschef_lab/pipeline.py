"""pipeline — the one CLI entry of the drift check and the retrain stages
(design I3; the lab runtime image's ENTRYPOINT).

    python -m souschef_lab.pipeline drift poll|static|check|weekly|calibrate …   → drift.main (builder B)
    python -m souschef_lab.pipeline prepare --run-id rt-… --bucket gs://… --mode full|smoke
    python -m souschef_lab.pipeline collect --run-id rt-… --bucket gs://… --mode full|smoke [--attempt N]
    python -m souschef_lab.pipeline dataset --run-id rt-… --bucket gs://… --mode full|smoke
    python -m souschef_lab.pipeline eval    --run-id rt-… --bucket gs://… --mode full|smoke \\
                                            --current-url https://…eval… --candidate-url https://…cand…
    python -m souschef_lab.pipeline decision --run-id rt-… --outcome approved|skipped|expired [--digest sha256:…]
    python -m souschef_lab.pipeline selftest
    python -m souschef_lab.pipeline retrain-dry --workdir DIR [--smoke]      every stage end to end, no network

Every retrain stage reads its inputs from and writes its outputs to the pipeline
bucket (`--bucket`, default $PIPELINE_BUCKET; a local folder works the same
way) under runs/<run id>/ and leaves runs/<id>/result-<stage>.json. Before a
stage runs, the engine snapshot of the deployed launcher is fetched from the
bucket (enginebundle.ensure) unless --engine-dir or $ENGINE names one.

Environment: SOUS_CHEF_URL + SOUS_CHEF_KEY (int, collect), ANTHROPIC_API_KEY
(prepare's authoring), PIPELINE_BUCKET, GOOGLE_CLOUD_PROJECT. Keys are read from
the environment only and never printed.

Exit codes: 0 the stage finished (outcome in its result), 1 a failure (the
workflow retries, then posts stage_failed), 2 configuration / precondition
(no retry).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from .retrain.common import (EXIT_CONFIG, EXIT_FAIL, EXIT_OK, ConfigError, StageError, deployed_of, load_run, open_store,
                             redact, run_prefix, utc_now, workdir)

STAGES = ("prepare", "collect", "dataset", "eval")



def COLLECT_WORKERS() -> int:
    """Teacher collection's workers (env COLLECT_WORKERS, default 6): int allows 8 concurrent turns an instance
    (MAX_CONCURRENT_TURNS), so 6 leave merchants room instead of 503 busy for the ~2 h a round collects."""
    try:
        return max(1, int(os.environ.get("COLLECT_WORKERS", "6")))
    except ValueError:
        return 6

def _stage_parser(sub, name: str, help_: str) -> argparse.ArgumentParser:
    p = sub.add_parser(name, help=help_)
    p.add_argument("--run-id", required=True)
    p.add_argument("--bucket", default=os.environ.get("PIPELINE_BUCKET"), help="gs://… or a local folder")
    p.add_argument("--mode", choices=("full", "smoke"), default="full")
    p.add_argument("--workdir", help="local scratch (default $PIPELINE_WORKDIR or the temp dir)")
    p.add_argument("--engine-dir", help="a cookie-cutter snapshot instead of the bucket's engine bundle")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--dry", action="store_true", help="the lab's fake service / fake author: no network, no model")
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="souschef_lab.pipeline", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("drift", help="the drift check (souschef_lab.drift; its own flags)", add_help=False)
    pr = _stage_parser(sub, "prepare", "rebase the pool, author new-field cases → runs/<id>/prepare/cases.jsonl")
    pr.add_argument("--accept-rebase-loss", default="",
                    help="families (comma-separated, or 'all') whose rebase loss over 20 %% a person has looked at and accepts")
    c = _stage_parser(sub, "collect", "the teacher collection on int (resumable)")
    c.add_argument("--attempt", type=int, default=1, help="a new attempt = new logs and site ids (after teacher_changed)")
    c.add_argument("--url", default=os.environ.get("SOUS_CHEF_URL"))
    c.add_argument("--expect-build", help="the teacher build (default /opt/sous-chef/BUILD_ID)")
    d = _stage_parser(sub, "dataset", "the training set + its gates")
    d.add_argument("--parity", choices=("required", "auto", "skip"), default="required")
    d.add_argument("--no-caps", action="store_true", help="no section / family caps (a tiny rehearsal set; SMOKE has none)")
    e = _stage_parser(sub, "eval", "the paired eval, the bar, the report and the decision payload")
    e.add_argument("--current-url", default=os.environ.get("EVAL_URL"))
    e.add_argument("--candidate-url", default=os.environ.get("CANDIDATE_URL"))
    dc = sub.add_parser("decision", help="the bucket's side of Approve (after the rollout) or Skip / expiry")
    dc.add_argument("--run-id", required=True)
    dc.add_argument("--bucket", default=os.environ.get("PIPELINE_BUCKET"))
    dc.add_argument("--outcome", choices=("approved", "skipped", "expired"), required=True)
    dc.add_argument("--digest", help="approved: the rolled-out image digest (default runs/<id>/model/image.json)")
    dc.add_argument("--by", default=None, help="who decided (the approver's email, or 'expiry'); recorded")
    sub.add_parser("selftest", help="the image's tools: node, Postgres, the service binaries, gcloud, anthropic")
    r = sub.add_parser("retrain-dry", help="every retrain stage end to end on a local bucket (fakes; needs node + the engine)")
    r.add_argument("--workdir", required=True)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--engine-dir")
    return ap


# --- the engine -------------------------------------------------------------------------------------------------------

def ensure_engine(store, want_sha: str | None, *, engine_dir: str | None, work: Path) -> dict:
    """{status, sha, engine, fingerprint}: --engine-dir, else $ENGINE, else the bucket's bundle."""
    from .engine import engine_fingerprint
    if engine_dir:
        os.environ["COOKIE_CUTTER_DIR"] = str(Path(engine_dir).resolve())
        os.environ["ENGINE"] = str(Path(engine_dir).resolve() / "launcher/client/src/sousChef/ops/headless.mjs")
        os.environ["SOUS_CHEF_ENGINE"] = os.environ["ENGINE"]
        fp = engine_fingerprint(os.environ["ENGINE"]) or {}
        return {"status": "local", "sha": fp.get("commit"), "engine": os.environ["ENGINE"], "fingerprint": fp.get("fingerprint"),
                "want_sha": want_sha}
    if os.environ.get("ENGINE") and Path(os.environ["ENGINE"]).is_file():
        fp = engine_fingerprint(os.environ["ENGINE"]) or {}
        return {"status": "env", "sha": fp.get("commit"), "engine": os.environ["ENGINE"], "fingerprint": fp.get("fingerprint"),
                "want_sha": want_sha}
    from . import enginebundle as EB
    try:
        b = EB.ensure(store.b, want_sha or "newest", workdir=work / "engine")
    except EB.BundleError as e:
        raise ConfigError(f"engine: {e}", reason="engine") from None
    return {**b.as_dict(), "want_sha": want_sha}


def _want_sha(store, run_id: str) -> str | None:
    try:
        dep = deployed_of(store, load_run(store, run_id))
    except ConfigError:
        return None
    gf = dep.get("generated_from") if isinstance(dep.get("generated_from"), dict) else {}
    return gf.get("sha")


# --- the stages -------------------------------------------------------------------------------------------------------

def run_stage(a) -> int:
    from . import pipeline_common as PC
    try:
        store = open_store(a.bucket)
        run_prefix(a.run_id)
    except ConfigError as e:
        print(f"pipeline {a.cmd}: {redact(e)}", file=sys.stderr)
        return EXIT_CONFIG
    if a.workdir:
        os.environ["PIPELINE_WORKDIR"] = a.workdir
    work = workdir(a.run_id, a.cmd)
    started = utc_now()
    try:
        info = ensure_engine(store, _want_sha(store, a.run_id), engine_dir=a.engine_dir, work=work)
        res = _dispatch(a, store, work, info)
        print(json.dumps({"stage": a.cmd, "status": res.get("status"), "metrics": res.get("metrics")}, default=str)[:4000])
        return EXIT_OK
    except ConfigError as e:
        code, err = EXIT_CONFIG, e
    except StageError as e:
        code, err = EXIT_FAIL, e
    except Exception as e:  # noqa: BLE001 — every failure leaves a result for the workflow's stage_failed
        code, err = EXIT_FAIL, e
        traceback.print_exc()
    try:
        cur = store.read_json(f"{run_prefix(a.run_id)}/result-{a.cmd}.json") or {}
        if not (cur.get("status") not in (None, "ok", "failed") and str(cur.get("at") or "") >= started):
            doc = PC.failure_doc(a.cmd, a.run_id, err)
            doc["error"]["reason"] = getattr(err, "reason", None)
            doc["exit"] = code
            store.write_json(f"{run_prefix(a.run_id)}/result-{a.cmd}.json", doc)
    except Exception:  # noqa: BLE001
        pass
    print(f"pipeline {a.cmd}: {type(err).__name__}: {redact(err)}", file=sys.stderr)
    return code


def _dispatch(a, store, work: Path, info: dict) -> dict:
    from .retrain import collect_stage, dataset_stage, eval_stage, prepare
    if a.cmd == "prepare":
        from .engine import PersistentEngine
        from .retrain import author as A
        from .retrain.common import deployed_of as dep_of
        from .retrain.common import manifest_doc, save_manifest
        dep = dep_of(store, load_run(store, a.run_id))
        mpath = save_manifest(manifest_doc(store, dep["hash"]), work / "manifest-engine.json")
        if a.dry:
            from .manifest import Manifest
            ask = A.FakeAsk(Manifest(json.loads(mpath.read_text(encoding="utf-8"))))
        else:
            ask = A.anthropic_ask() if os.environ.get("ANTHROPIC_API_KEY") else None
        eng = PersistentEngine(info.get("engine"), manifest_path=mpath, size=a.workers or 4)
        try:
            accept = frozenset(f.strip() for f in (a.accept_rebase_loss or "").split(",") if f.strip())
            return prepare.run(store, a.run_id, mode=a.mode, engine=eng, engine_info=info, ask=ask, workdir=work,
                               workers=a.workers or 4, accept_loss=accept)
        finally:
            eng.close()
    if a.cmd == "collect":
        return collect_stage.run(store, a.run_id, mode=a.mode, attempt=a.attempt, url=a.url, workdir=work, fake=a.dry,
                                 expect_build=a.expect_build, workers=a.workers or COLLECT_WORKERS(), engine=info.get("engine"))
    if a.cmd == "dataset":
        return dataset_stage.run(store, a.run_id, mode=a.mode, workdir=work, parity=a.parity,
                                 engine_fp=info.get("fingerprint"), caps=False if a.no_caps else None)
    if a.cmd == "eval":
        return eval_stage.run(store, a.run_id, mode=a.mode, current_url=a.current_url, candidate_url=a.candidate_url,
                              workdir=work, dry=a.dry, engine_path=info.get("engine"), workers=a.workers or 2)
    raise ConfigError(f"unknown stage {a.cmd}")


# --- decision ---------------------------------------------------------------------------------------------------------

def decision(a) -> int:
    from .retrain import decide
    try:
        store = open_store(a.bucket)
        run_prefix(a.run_id)
        res = (decide.approved(store, a.run_id, digest=a.digest, by=a.by) if a.outcome == "approved"
               else decide.skipped(store, a.run_id, reason=a.outcome, by=a.by))
    except ConfigError as e:
        print(f"pipeline decision: {redact(e)}", file=sys.stderr)
        _decision_result(a, "failed", {"error": redact(e)})
        return EXIT_CONFIG
    except Exception as e:  # noqa: BLE001
        print(f"pipeline decision: {type(e).__name__}: {redact(e)}", file=sys.stderr)
        _decision_result(a, "failed", {"error": f"{type(e).__name__}: {redact(e)}"})
        return EXIT_FAIL
    _decision_result(a, "ok", res)
    print(json.dumps(res))
    return EXIT_OK


def _decision_result(a, status: str, metrics: dict) -> None:
    """runs/<id>/result-decision.json, like every stage's (the workflow and R8 read it)."""
    try:
        from . import pipeline_common as PC
        store = open_store(a.bucket)
        store.write_json(f"{run_prefix(a.run_id)}/result-decision.json",
                         PC.result_doc(status=status, stage="decision", run_id=a.run_id, metrics=metrics))
    except Exception:  # noqa: BLE001 — the decision's own writes are what matter
        pass


# --- selftest ---------------------------------------------------------------------------------------------------------

def selftest() -> int:
    """The lab image's tools, one line each; exit 1 when one is missing."""
    checks: dict[str, str] = {}

    def run(name: str, argv: list[str]) -> None:
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
            out = (r.stdout or r.stderr).strip().splitlines()
            checks[name] = ("ok " if r.returncode == 0 else f"FAIL (exit {r.returncode}) ") + (out[0][:120] if out else "")
        except (OSError, subprocess.SubprocessError) as e:
            checks[name] = f"FAIL {type(e).__name__}: {e}"[:200]

    run("node", ["node", "--version"])
    pg = os.environ.get("PG_BIN") or "/usr/lib/postgresql/16/bin"
    run("pg_ctl", [str(Path(pg) / "pg_ctl") if (Path(pg) / "pg_ctl").is_file() else (shutil.which("pg_ctl") or "pg_ctl"), "--version"])
    for env, default in (("SOUS_CHEF_BIN", "/opt/sous-chef/bin/sous-chef"),
                         ("SOUS_CHEF_FIELDS_BIN", "/opt/sous-chef/bin/sous-chef-fields"),
                         ("SOUS_CHEF_PREFIX_BIN", "/opt/sous-chef/bin/sous-chef-prefix")):
        # the binaries have no -version flag (the build is /opt/sous-chef/BUILD_ID): present and executable
        exe = os.environ.get(env) or default
        ok = Path(exe).is_file() and os.access(exe, os.X_OK)
        checks[Path(exe).name] = f"ok {exe}" if ok else f"FAIL {exe} missing or not executable"
    run("gcloud", ["gcloud", "--version"])
    try:
        import anthropic  # noqa: F401
        checks["anthropic"] = f"ok {getattr(anthropic, '__version__', '')}"
    except ImportError as e:
        checks["anthropic"] = f"FAIL {e}"
    bid = Path(os.environ.get("SOUS_CHEF_BUILD_FILE") or "/opt/sous-chef/BUILD_ID")
    checks["BUILD_ID"] = f"ok {bid.read_text().strip()}" if bid.is_file() else f"FAIL no {bid}"
    for k, v in checks.items():
        print(f"{k:<20} {v}")
    return 0 if all(v.startswith("ok") for v in checks.values()) else 1


# --- retrain-dry ------------------------------------------------------------------------------------------------------

def retrain_dry(a) -> int:
    """Seed a local bucket (the exam manifest as the live model's, a deployed manifest with one new field, a small
    generated pool), then prepare → collect → dataset → (train/build stubbed) → eval → report, all DRY."""
    from .retrain import dry
    return dry.main(Path(a.workdir), smoke=a.smoke, engine_dir=a.engine_dir)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "drift":
        from . import drift
        return drift.main(argv[1:])
    a = build_parser().parse_args(argv)
    if a.cmd in STAGES:
        return run_stage(a)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "decision":
        return decision(a)
    if a.cmd == "retrain-dry":
        return retrain_dry(a)
    return EXIT_CONFIG


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
