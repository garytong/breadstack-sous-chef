"""souschef_lab/pipeline.py (the one CLI entry, design I3), the end-to-end DRY
rehearsal of every retrain stage, and tools/pipeline-train.sh against a stub gcloud
and a stub trainer (no GPU, no network)."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from souschef_lab import pipeline as P
from souschef_lab.retrain import dry as DRY
from souschef_lab.retrain.common import open_store

from tests.test_rebase import _need_engine

LAB = Path(__file__).resolve().parents[1]


def test_drift_is_dispatched_with_its_own_flags(monkeypatch):
    got = {}
    import souschef_lab.drift as D

    def fake(argv):
        got["argv"] = argv
        return 0
    monkeypatch.setattr(D, "main", fake)
    assert P.main(["drift", "poll", "--bucket", "/x", "--dry"]) == 0
    assert got["argv"] == ["poll", "--bucket", "/x", "--dry"]


def test_configuration_errors_exit_2_with_a_failure_result(tmp_path, monkeypatch):
    monkeypatch.delenv("PIPELINE_BUCKET", raising=False)
    assert P.main(["prepare", "--run-id", "rt-1"]) == 2  # no bucket at all: refused before anything runs
    b = tmp_path / "bucket"
    assert P.main(["dataset", "--run-id", "rt-1", "--bucket", str(b), "--engine-dir", str(tmp_path)]) == 2
    res = open_store(b).read_json("runs/rt-1/result-dataset.json")
    assert res["status"] == "failed" and res["exit"] == 2 and res["error"]["class"] == "ConfigError"
    assert P.main(["prepare", "--run-id", "../escape", "--bucket", str(b)]) == 2


def test_selftest_reports_every_tool(tmp_path, monkeypatch, capsys):
    true = "/bin/true" if Path("/bin/true").exists() else "/usr/bin/true"
    for env in ("SOUS_CHEF_BIN", "SOUS_CHEF_FIELDS_BIN", "SOUS_CHEF_PREFIX_BIN"):
        monkeypatch.setenv(env, true)
    (tmp_path / "BUILD_ID").write_text("f79753b-speed-a\n")
    monkeypatch.setenv("SOUS_CHEF_BUILD_FILE", str(tmp_path / "BUILD_ID"))
    rc = P.selftest()
    out = capsys.readouterr().out
    for k in ("node", "pg_ctl", "gcloud", "anthropic", "BUILD_ID"):
        assert k in out
    assert "ok f79753b-speed-a" in out
    assert rc in (0, 1)


def test_retrain_dry_smoke_end_to_end(tmp_path, capsys):
    _need_engine()
    rc = DRY.main(tmp_path, smoke=True)
    out = capsys.readouterr().out
    assert rc == 0, out[-3000:]
    s = open_store(tmp_path / "bucket")
    run = next(iter(s.list("runs")))
    run_id = run.split("/")[1]
    for st in ("prepare", "collect", "dataset", "eval"):
        assert s.read_json(f"runs/{run_id}/result-{st}.json")["status"] == "ok", st
    dec = s.read_json(f"runs/{run_id}/decision.json")
    assert dec["smoke"] is True and isinstance(dec["bar_pass"], bool)
    assert {r["name"] for r in dec["results"]["bar"]["rules"]} == {"sets", "mcnemar", "oos_where", "f0h_unseen", "trigger", "ghosts"}
    note = s.read_json(f"runs/{run_id}/result-eval.json")["notification"]
    assert note["kind"] == "candidate_ready" and "SMOKE" in note["title"]
    # SMOKE never touches the pool: the authored cases stay in the run
    assert s.exists(f"runs/{run_id}/prepare/authored-smoke.jsonl")
    assert not s.list("cases/pool/authored") and len(s.read_json("cases/pool/index.json")["files"]) == 1
    # the dataset, the model build context and the candidate's baseline are where the next stages read them
    assert s.exists(f"runs/{run_id}/dataset/dataset/train.jsonl") and s.exists(f"runs/{run_id}/model/context.tgz")
    cb = s.read_json(f"runs/{run_id}/eval/candidate-baseline.json")
    assert cb["source"] == "candidate-eval" and cb["replicates"] and cb["digest"] == "sha256:dry-candidate"
    assert all(k.split(":", 1)[0] in ("paraphrase", "holdout", "real", "followup", "out_of_scope", "out_of_scope_holdout")
               for k in cb["replicates"][0]["rows"])


def test_a_full_prepare_adds_the_round_to_the_pool(tmp_path):
    _need_engine()
    bucket, run_id = DRY.seed(tmp_path, smoke=True)
    cc = os.environ.get("COOKIE_CUTTER_DIR")
    argv = ["prepare", "--run-id", run_id, "--bucket", bucket, "--mode", "full", "--workdir", str(tmp_path / "jobs"),
            "--dry", "--workers", "2"] + (["--engine-dir", cc] if cc else [])
    assert P.main(argv) == 0
    s = open_store(bucket)
    idx = s.read_json("cases/pool/index.json")
    paths = [f["path"] for f in idx["files"]]
    authored = s.read_jsonl(f"cases/pool/authored/{run_id}.jsonl") + s.read_jsonl(f"cases/exam-ext/{run_id}.jsonl")
    assert len(authored) == 8 and all(c["meta"]["target_address"].endswith("buttonShadowColor") for c in authored)
    if s.exists(f"cases/pool/authored/{run_id}.jsonl"):
        new = next(f for f in idx["files"] if f["path"] == f"cases/pool/authored/{run_id}.jsonl")
        assert new["trained_in"] is None and new["manifest"]
    assert paths[0] == "cases/pool/r4/cases-all.jsonl"
    # a retried prepare reuses the round's authored cases (no second bill)
    rep1 = s.read_json(f"runs/{run_id}/prepare/report.json")
    assert P.main(argv) == 0
    rep2 = s.read_json(f"runs/{run_id}/prepare/report.json")
    assert rep2["author"].get("reused") is True and rep1["totals"]["cases"] == rep2["totals"]["cases"]


# --- tools/pipeline-train.sh against stubs ------------------------------------------------------------------------------

FAKE_GCLOUD = r'''#!/usr/bin/env python3
"""gcloud storage cp|rsync|ls over $FAKE_GCS (gs://B/key → $FAKE_GCS/B/key)."""
import os, shutil, sys
from pathlib import Path
root = Path(os.environ["FAKE_GCS"])
args = [a for a in sys.argv[1:] if a != "--quiet"]
assert args[0] == "storage", args
cmd, rest = args[1], [a for a in args[2:] if not a.startswith("--")]
def loc(u):
    return root / u[len("gs://"):] if u.startswith("gs://") else Path(u)
if cmd == "ls":
    p = loc(rest[0])
    sys.exit(0 if p.exists() else 1)
if cmd == "cp":
    *srcs, dst = rest
    for s in srcs:
        s, d = loc(s), loc(dst)
        if not s.exists():
            print("No URLs matched", file=sys.stderr); sys.exit(1)
        if dst.endswith("/") or d.is_dir():
            d = d / s.name
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, d)
    sys.exit(0)
if cmd == "rsync":
    s, d = loc(rest[0]), loc(rest[1])
    if not s.exists():
        print("No URLs matched", file=sys.stderr); sys.exit(1)
    excl = [a.split("=", 1)[1] for a in args if a.startswith("--exclude=")]
    for p in s.rglob("*"):
        if p.is_file():
            rel = p.relative_to(s).as_posix()
            if any(__import__("re").match(e.strip("'"), rel) for e in excl):
                continue
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, d / rel)
    sys.exit(0)
sys.exit(2)
'''

FAKE_TRAINPY = r'''#!/usr/bin/env bash
# the trainer image's python, minus the GPU: train / export are faked, everything else is the real interpreter
if [ "$1" = -m ] && [ "$2" = souschef_lab.train ]; then
  shift 2; name=; max=; resume=0; dry=0
  while [ $# -gt 0 ]; do case "$1" in --name) name=$2; shift;; --max-steps) max=$2; shift;; --resume) resume=1;; --dry-run) dry=1;; esac; shift; done
  [ $dry = 1 ] && { echo "SFT plan: dry"; exit 0; }
  d=runs/adapters/$name
  if [ -n "${STUB_SLEEP:-}" ] && [ "$name" = "${STUB_SLEEP}" ]; then
    mkdir -p "$d/checkpoints/checkpoint-50"; echo '{}' > "$d/checkpoints/checkpoint-50/state.json"
    echo "sleeping $name" >> "$STUB_LOG"; sleep 60; exit 0
  fi
  if [ $resume = 1 ] && [ -d "$d/checkpoints/checkpoint-2" ]; then echo "Resuming from $d/checkpoints/checkpoint-2 (optimizer, scheduler and data position restored)."; fi
  mkdir -p "$d/checkpoints/checkpoint-2" "$d/adapter"
  echo '{}' > "$d/adapter/adapter_config.json"
  echo "{\"updates\": ${max:-8}}" > "$d/run.json"
  printf '{"step": 1, "eval_loss": 0.7}\n{"step": 2, "eval_loss": 0.2}\n' > "$d/loss.jsonl"
  echo "trained $name to ${max:-all}" >> "$STUB_LOG"
  exit 0
fi
if [ "$1" = -m ] && [ "$2" = souschef_lab.export ]; then
  shift 2; name=
  while [ $# -gt 0 ]; do case "$1" in --name) name=$2; shift;; esac; shift; done
  e=runs/exports/$name; mkdir -p "$e"
  printf 'GGUFfake' > "$e/$name-q4_k_m.gguf"
  printf 'FROM ./%s-q4_k_m.gguf\nPARAMETER temperature 1\nPARAMETER presence_penalty 1.5\nPARAMETER top_k 20\n' "$name" > "$e/Modelfile.q4_k_m"
  echo "exported $name" >> "$STUB_LOG"
  exit 0
fi
exec "$REAL_PY" "$@"
'''


def _exe(p: Path, body: str) -> Path:
    p.write_text(body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _train_rows(n: int, follow: bool) -> str:
    rows = []
    for i in range(n):
        rows.append({"id": f"r{i}", "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "Done."}],
                     "meta": {"fields_context": True, "history_assistant_turns": int(follow and i == 0), "block_drift": False}})
    return "".join(json.dumps(r) + "\n" for r in rows)


@pytest.fixture
def train_env(tmp_path):
    lab = tmp_path / "lab"
    lab.mkdir()
    for name in ("souschef_lab", "model-image", "base-modelfiles", "Makefile", "tools"):
        (lab / name).symlink_to(LAB / name)
    gcs = tmp_path / "gcs"
    run = "rt-20261012-0310-abcd1234"
    ds = gcs / "bkt" / "runs" / run / "dataset" / "dataset"
    ds.mkdir(parents=True)
    (ds / "train.jsonl").write_text(_train_rows(3, True))
    (ds / "val.jsonl").write_text(_train_rows(1, False))
    (ds / "stats.json").write_text(json.dumps({"block_drift_rows": 0, "preview_claim_kept": 0, "by_family": {"F0": 4}}))
    base = gcs / "bkt" / "base"
    base.mkdir(parents=True)
    hub = tmp_path / "hub" / "hub" / "models--unsloth--Qwen3.5-9B"
    hub.mkdir(parents=True)
    (hub / "config.json").write_text("{}")
    tar = base / "unsloth-qwen3.5-9b-hf.tar"
    subprocess.run(["tar", "-C", str(tmp_path / "hub"), "-cf", str(tar), "hub"], check=True)
    sha = subprocess.run(["sha256sum", tar.name], cwd=base, capture_output=True, text=True, check=True).stdout
    (base / "unsloth-qwen3.5-9b-hf.tar.sha256").write_text(sha)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    gcloud = _exe(bindir / "gcloud", FAKE_GCLOUD.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    trainpy = _exe(bindir / "trainpy", FAKE_TRAINPY)
    env = {**os.environ, "PIPELINE_BUCKET": "gs://bkt", "SOUS_CHEF_LAB": str(lab), "TRAINPY": str(trainpy),
           "GCLOUD": str(gcloud), "FAKE_GCS": str(gcs), "REAL_PY": sys.executable, "PIPELINE_WORKDIR": str(tmp_path / "w"),
           "HF_HOME": str(tmp_path / "hf"), "STUB_LOG": str(tmp_path / "stub.log"), "PYTHONPATH": str(lab)}
    return {"env": env, "run": run, "gcs": gcs, "lab": lab, "tmp": tmp_path}


def _train(te, mode="full"):
    return subprocess.run(["bash", str(LAB / "tools" / "pipeline-train.sh"), te["run"], mode], env=te["env"], cwd=te["tmp"],
                          capture_output=True, text=True, timeout=300)


def test_pipeline_train_full_run_pushes_adapter_export_stage_and_done(train_env):
    te = train_env
    r = _train(te)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    out = te["gcs"] / "bkt" / "runs" / te["run"] / "train"
    assert (out / "DONE").is_file()
    assert (out / "adapter" / "adapter" / "adapter_config.json").is_file()
    assert not (out / "adapter" / "checkpoints").exists()  # trainer state goes to train/checkpoints only
    assert (out / "export" / f"{te['run']}-q4_k_m.gguf").is_file()
    modelfile = (out / "stage" / "Modelfile").read_text()
    assert "PARAMETER temperature 0.3" in modelfile and "PARAMETER presence_penalty 0" in modelfile
    assert "PARAMETER num_ctx 65536" in modelfile and "PARAMETER top_k 20" in modelfile
    prog = json.loads((out / "progress.json").read_text())
    assert prog["step"] == "DONE" and prog["status"] == "done" and prog["eval_loss_last"] == 0.2
    stub = (te["tmp"] / "stub.log").read_text()
    assert f"trained {te['run']}-smoke-sft to 2" in stub and f"trained {te['run']}-smoke-sft to 4" in stub
    assert f"exported {te['run']}" in stub
    # a rerun after DONE does nothing
    r2 = _train(te)
    assert r2.returncode == 0 and "nothing to do" in r2.stdout


def test_pipeline_train_resumes_from_pushed_checkpoints_and_skips_the_resume_smoke(train_env):
    te = train_env
    ck = te["gcs"] / "bkt" / "runs" / te["run"] / "train" / "checkpoints" / "checkpoint-2"
    ck.mkdir(parents=True)
    (ck / "state.json").write_text("{}")
    r = _train(te)
    assert r.returncode == 0, r.stdout[-3000:]
    stub = (te["tmp"] / "stub.log").read_text()
    assert "smoke-sft" not in stub and f"trained {te['run']}-sft" in stub
    assert "Resuming from" in r.stdout


def test_pipeline_train_fails_on_a_bad_base_tar_and_records_the_step(train_env):
    te = train_env
    (te["gcs"] / "bkt" / "base" / "unsloth-qwen3.5-9b-hf.tar.sha256").write_text("0" * 64 + "  unsloth-qwen3.5-9b-hf.tar\n")
    r = _train(te)
    assert r.returncode == 1
    out = te["gcs"] / "bkt" / "runs" / te["run"] / "train"
    prog = json.loads((out / "progress.json").read_text())
    assert prog["status"] == "failed" and "sha256" in prog["error"] and prog["updated_at"]
    assert (out / "train.log").is_file() and not (out / "DONE").exists()


def test_pipeline_train_smoke_mode(train_env):
    te = train_env
    ds = te["gcs"] / "bkt" / "runs" / te["run"] / "dataset" / "dataset"
    (ds / "train.jsonl").write_text(_train_rows(3, False))  # SMOKE may have no follow-ups
    r = _train(te, "smoke")
    assert r.returncode == 0, r.stdout[-3000:]
    stub = (te["tmp"] / "stub.log").read_text()
    assert f"trained {te['run']}-sft to 4" in stub and f"trained {te['run']}-sft to 6" in stub and "smoke-sft" not in stub


def test_pipeline_train_sigterm_pushes_checkpoints_and_records_interrupted(train_env):
    """A preemption (SIGTERM to the entry) reaches the trap at once — not after the SFT — and leaves the
    checkpoints and status "interrupted" for the workflow (classed preempted: resumed, not fatal)."""
    import signal
    import time as _t
    te = train_env
    env = {**te["env"], "STUB_SLEEP": f"{te['run']}-sft"}
    ck = te["gcs"] / "bkt" / "runs" / te["run"] / "train" / "checkpoints" / "checkpoint-2"
    ck.mkdir(parents=True)  # a resume: no resume smoke, straight to the (sleeping) SFT
    (ck / "state.json").write_text("{}")
    out_f = (te["tmp"] / "train.out").open("w")
    proc = subprocess.Popen(["bash", str(LAB / "tools" / "pipeline-train.sh"), te["run"], "full"], env=env, cwd=te["tmp"],
                            stdout=out_f, stderr=subprocess.STDOUT, text=True)
    log = te["tmp"] / "stub.log"
    deadline = _t.monotonic() + 120
    while _t.monotonic() < deadline and not (log.is_file() and "sleeping" in log.read_text()):
        _t.sleep(0.2)
    assert log.is_file() and "sleeping" in log.read_text()
    t0 = _t.monotonic()
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=60)
    out_f.close()
    out = (te["tmp"] / "train.out").read_text()
    assert proc.returncode == 143, out[-2000:]
    assert _t.monotonic() - t0 < 30  # the trap ran at once, not after the 60 s step
    train = te["gcs"] / "bkt" / "runs" / te["run"] / "train"
    prog = json.loads((train / "progress.json").read_text())
    assert prog["status"] == "interrupted" and prog["step"].startswith("SFT")
    assert (train / "checkpoints" / "checkpoint-50" / "state.json").is_file()
    assert not (train / "DONE").exists()
