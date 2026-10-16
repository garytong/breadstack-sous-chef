"""retrain/noise.py: the seed-noise experiment (docs/RETRAIN.md R10) — arms, the Vertex job, the fakes end to end,
the σ_seed estimator and the seed-aware tolerance."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

from souschef_lab.retrain import noise as N
from souschef_lab.retrain.common import ConfigError, open_store

EXP = "nz-20261015"
R5 = "rt-20261014-0708-7cac28-1184ff9a"


def test_arms_parse_to_run_folders_shaped_like_rounds():
    a = N.parse_arm("r4@2", EXP)
    assert (a.label, a.prefix, a.seed, a.run_id, a.candidate_id) == (
        "r4", "cases/pool/r4/dataset", 2, "nz-20261015-r4-s2", "c-nz-20261015-r4-s2")
    b = N.parse_arm(f"{R5}@2", EXP)
    assert b.prefix == f"runs/{R5}/dataset" and b.label == "d1184ff9a"
    c = N.parse_arm(f"r5=runs/{R5}/dataset/@3407", EXP)
    assert (c.label, c.prefix, c.seed) == ("r5", f"runs/{R5}/dataset", 3407)
    for bad in ("r4", "r4@x", "nope@2", "R5=runs/x@2", "@2"):
        with pytest.raises(ConfigError):
            N.parse_arm(bad, EXP)
    for bad in ("", "rt-1", "NZ-1", "nz-a/b"):
        with pytest.raises(ConfigError):
            N.check_exp(bad)


def test_replicates_follow_occurrence_order_split_f0h_and_drop_errors():
    rows = [{"key": "holdout:h1", "set": "holdout", "status": "ok", "pass": 1},
            {"key": "holdout:h2", "set": "holdout", "status": "error", "pass": 0},
            {"key": "holdout:h1", "set": "holdout", "status": "ok", "pass": 0},
            {"key": "holdout:h2", "set": "holdout", "status": "ok", "pass": 1},
            {"key": "F0H:a", "set": "F0H", "seen": True, "status": "ok", "pass": 1},
            {"key": "F0H:b", "set": "F0H", "seen": False, "status": "ok", "pass": 0}]
    r = N.replicates(rows)
    assert r["holdout"] == [{"holdout:h1": 1}, {"holdout:h1": 0, "holdout:h2": 1}]
    assert r["F0H:seen"] == [{"F0H:a": 1}] and r["F0H:unseen"] == [{"F0H:b": 0}]


def test_seed_aware_tolerance_formula():
    assert N.seed_aware_tol(1.0, 0.0) == 2           # today's tol for σ_Δ 1 (holdout in r5)
    assert N.seed_aware_tol(2.082, 0.0) == 5          # r5's paraphrase tol
    assert N.seed_aware_tol(1.0, 1.5) == 5            # ⌈2·sqrt(1 + 2·2.25)⌉ = ⌈4.69⌉ (both models are draws)
    assert N.seed_aware_tol(1.0, 1.5, draws=1) == 4   # ⌈2·sqrt(1 + 2.25)⌉ = ⌈3.61⌉
    assert N.seed_aware_tol(0.2, 0.0) == 1            # never below 1 row
    assert N.seed_aware_tol(1.0, math.sqrt(1.5)) == 4  # exactly 4.0 stays 4


def test_chi2_quantiles():
    assert N.chi2_ppf(0.10, 2) == pytest.approx(0.2107, abs=1e-3)
    assert N.chi2_ppf(0.05, 1) == pytest.approx(0.00393, abs=1e-4)
    assert N.chi2_ppf(0.50, 4) == pytest.approx(3.357, abs=1e-2)


def _model(name, data, means_rows, n=60, k=2, flaky=0):
    """k replicates over n rows: `means_rows` rows pass in every replicate, plus `flaky` rows that alternate."""
    reps = []
    for i in range(k):
        d = {f"holdout:h{j:03d}": int(j < means_rows) for j in range(n)}
        for j in range(flaky):
            d[f"holdout:h{n - 1 - j:03d}"] = int((i + j) % 2 == 0)
        reps.append(d)
    return N.Model(name, data, {"holdout": reps})


def test_sigma_seed_is_the_spread_of_draws_beyond_eval_noise():
    # no eval noise (every replicate identical): σ_seed = the SD of the models' means (50, 52, 54 → 2)
    models = {"live": _model("live", "r4", 50), "r4@2": _model("r4@2", "r4", 52), "r4@3": _model("r4@3", "r4", 54)}
    bars = {"r4@2": {"sets": [{"set": "holdout", "delta": 2.0, "b": 0, "c": 2, "tol": 2, "n": 60, "sigma_delta": 1.0, "m": [2, 2]}]},
            "r4@3": {"sets": [{"set": "holdout", "delta": 4.0, "b": 0, "c": 4, "tol": 2, "n": 60, "sigma_delta": 1.0, "m": [2, 2]}]}}
    r = N.analyze_set("holdout", models, bars)
    assert r["df"] == 2 and r["sigma_seed"] == pytest.approx(2.0, abs=1e-3)
    assert r["sigma_eval"] == 1.0 and r["tol_now"] == 2 and r["tol_seed"] == N.seed_aware_tol(1.0, 2.0) == 6
    assert r["sigma_seed_hi"] > r["sigma_seed"] and r["tol_seed_hi"] >= r["tol_seed"]
    assert r["max_abs_delta"] == 4.0
    # identical models with eval noise only: the noise is subtracted, σ_seed clamps at 0
    same = {n: _model(n, "r4", 50, flaky=6) for n in ("live", "r4@2", "r4@3")}
    r0 = N.analyze_set("holdout", same, bars)
    assert r0["sigma_seed"] == 0.0 and r0["eval_part"] > 0
    # another dataset's single model is not a draw of r4's recipe: it adds no degrees of freedom
    two = {"live": _model("live", "r4", 50), "r5@3407": _model("r5@3407", "r5", 40)}
    assert N.analyze_set("holdout", two, bars) is None
    # datasets pool their within-group spread
    four = {**models, "r5@3407": _model("r5@3407", "r5", 40), "r5@2": _model("r5@2", "r5", 44)}
    r4 = N.analyze_set("holdout", four, bars)
    assert r4["df"] == 3 and r4["s2"] == pytest.approx((2 * 4 + 1 * 8) / 3, abs=1e-3)


def test_job_body_is_the_pipelines_with_the_seed(tmp_path):
    store, _ = N.seed_fake_bucket(tmp_path / "b")
    cfg = N.config(store)
    arm = N.parse_arm("r4@2", EXP)
    flex = N.job_body(cfg, arm, N.pick_rung(cfg, None), image="img@sha256:1", attempt=1, nonce="n1")
    spec = flex["jobSpec"]
    cs = spec["workerPoolSpecs"][0]["containerSpec"]
    env = {e["name"]: e["value"] for e in cs["env"]}
    assert cs["args"] == ["nz-20261015-r4-s2", "full"] and cs["imageUri"] == "img@sha256:1"
    assert env["TRAIN_SEED"] == "2" and env["TRAIN_NONCE"] == "n1" and env["PIPELINE_BUCKET"] == "gs://fake-pipeline"
    assert {"TRAIN_RUNG", "TRAIN_ATTEMPT", "TRAIN_IMAGE", "GOOGLE_CLOUD_PROJECT"} <= set(env)
    assert flex["labels"]["purpose"] == "sous-chef-noise" and flex["labels"]["create"] == "n1"
    assert spec["scheduling"] == {"timeout": "43200s", "strategy": "FLEX_START", "maxWaitDuration": "14400s"}
    assert spec["serviceAccount"].startswith("souschef-trainer@")
    spot = N.job_body(cfg, arm, N.pick_rung(cfg, "h100-spot-usc1"), image="i", attempt=2, nonce="n2")["jobSpec"]["scheduling"]
    assert spot == {"timeout": "43200s", "strategy": "SPOT"}
    with pytest.raises(ConfigError):
        N.pick_rung(cfg, "nope")


def test_dry_runs_the_whole_experiment(tmp_path):
    res = N.dry(tmp_path, ["r4@2", "r4@3"], exp=EXP, log=lambda s: None)
    store = open_store(tmp_path / "bucket")
    assert set(res["models"]) == {"live", "r4@2", "r4@3"}
    for s in N.EXAM_SETS + ("F0H:unseen",):
        r = res["sets"][s]
        assert r["bar"] and r["df"] == 2 and r["tol_seed"] >= r["tol_now"] >= 1
        assert set(r["paired_vs_live"]) == {"r4@2", "r4@3"}
    assert res["sets"]["F0H:seen"]["bar"] is False
    assert set(res["thresholds_override"]["sigma_seed"]) >= set(N.EXAM_SETS) and "F0H:seen" not in res["thresholds_override"]["sigma_seed"]
    assert store.read_json(f"noise/{EXP}/result.json")["holdout_verdict"] == res["holdout_verdict"]
    plan = store.read_json(f"noise/{EXP}/plan.json")
    assert plan["arms"] == ["r4@2", "r4@3"] and plan["base_run"] == R5
    for arm in ("r4-s2", "r4-s3"):
        pre = f"runs/{EXP}-{arm}"
        run = store.read_json(f"{pre}/run.json")
        assert run["noise"]["base_run"] == R5 and run["deployed"]["hash"].startswith("1184ff9a")
        assert "execution" not in run and "resumes" not in run and run["candidate_id"] == f"c-{EXP}-{arm}"
        assert run["trigger"]["drift_run"]  # the probe stays (--no-probe drops it)
        from souschef_lab import r4score as R4S
        assert {**run["trigger"]["score_policy"], "font": run["trigger"]["font_policy"]} == R4S.STRICT_POLICY
        assert store.exists(f"{pre}/dataset/dataset/train.jsonl") and store.exists(f"{pre}/prepare/cases.jsonl")
        assert store.read_json(f"{pre}/train-config.json") == {"seed": int(arm[-1])}  # train.py pipeline_config's key
        img = store.read_json(f"{pre}/model/image.json")
        assert img["tag"] == f"c-{EXP}-{arm}" and img["digest"].startswith("sha256:")
        rec = store.read_json(f"{pre}/noise.json")
        assert rec["train"]["state"] == "done" and rec["train"]["seed_trained"] == int(arm[-1])
        assert rec["eval"]["state"] == "done" and rec["eval"]["current_digest"] == "sha256:" + "a" * 64
    # the locks are released, the pipeline's state is untouched
    assert store.b.read_lock(N.EVAL_LOCK) is None and store.b.read_lock(N.RETRAIN_LOCK) is None
    assert store.read_json("state/model.json")["tag"] == "r4"


def _cloud_run(tmp_path, **fake):
    store, base = N.seed_fake_bucket(tmp_path / "bucket")
    cloud = N.FakeCloud(store, **fake)
    arms = [N.parse_arm("r4@2", EXP)]
    N.setup(store, cloud, arms, base_run=base, log=lambda s: None)
    return store, cloud, N.config(store), arms


def test_a_preempted_job_is_resubmitted_and_resumes(tmp_path):
    store, cloud, cfg, arms = _cloud_run(tmp_path, outcomes={f"{EXP}-r4-s2": ["preempt", "ok"]})
    st = N.train(store, cloud, cfg, arms, rung=None, wait=True, poll_s=0, log=lambda s: None)
    assert st == {"r4@2": "done"}
    rec = store.read_json(f"runs/{EXP}-r4-s2/noise.json")
    assert [a["state"] for a in rec["train"]["attempts"]] == ["JOB_STATE_CANCELLED", "JOB_STATE_SUCCEEDED"]
    jobs = [c[2] for c in cloud.calls if c[0] == "job"]
    envs = [{e["name"]: e["value"] for e in j["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["env"]} for j in jobs]
    assert [e["TRAIN_ATTEMPT"] for e in envs] == ["1", "2"] and envs[0]["TRAIN_NONCE"] != envs[1]["TRAIN_NONCE"]


def test_a_flex_job_that_never_started_moves_down_the_ladder(tmp_path):
    # r5 waited ~3 h for Flex capacity; a wait that runs out must not be retried on Flex (up to 4 × 4 h idle)
    store, cloud, cfg, arms = _cloud_run(tmp_path, outcomes={f"{EXP}-r4-s2": ["nostart", "preempt", "ok"]})
    st = N.train(store, cloud, cfg, arms, rung=None, wait=True, poll_s=0, log=lambda s: None)
    assert st == {"r4@2": "done"}
    rec = store.read_json(f"runs/{EXP}-r4-s2/noise.json")
    assert [a["rung"] for a in rec["train"]["attempts"]] == ["h100-flex", "h100-spot-usc1", "h100-spot-usc1"]
    assert rec["train"]["attempts"][0]["never_started"] is True and rec["train"]["attempts"][1]["never_started"] is False
    jobs = [c[2] for c in cloud.calls if c[0] == "job"]
    assert [j["jobSpec"]["scheduling"]["strategy"] for j in jobs] == ["FLEX_START", "SPOT", "SPOT"]


def test_a_rung_moves_on_after_its_own_attempts():
    cfg = {"trainer": {"ladder": [{"name": "a", "attempts": 1}, {"name": "b", "attempts": 2}, {"name": "c"}]}}
    assert N.next_rung(cfg, None, [])["name"] == "a"
    assert N.next_rung(cfg, None, [{"rung": "a"}])["name"] == "b"
    assert N.next_rung(cfg, None, [{"rung": "a"}, {"rung": "b"}])["name"] == "b"
    assert N.next_rung(cfg, None, [{"rung": "a"}, {"rung": "b"}, {"rung": "b"}])["name"] == "c"
    assert N.next_rung(cfg, None, [{"rung": "c", "never_started": True}])["name"] == "c"  # the last rung stays


def test_a_trainer_failure_is_not_retried(tmp_path):
    store, cloud, cfg, arms = _cloud_run(tmp_path, outcomes={f"{EXP}-r4-s2": ["fail"]})
    assert N.train(store, cloud, cfg, arms, rung=None, wait=True, poll_s=0, log=lambda s: None) == {"r4@2": "failed"}
    assert len([c for c in cloud.calls if c[0] == "job"]) == 1
    assert "boom" in store.read_json(f"runs/{EXP}-r4-s2/noise.json")["train"]["error"]


def test_a_trainer_that_ignores_the_seed_is_caught_and_left_out(tmp_path):
    store, cloud, cfg, arms = _cloud_run(tmp_path, honour_seed=False)
    assert N.train(store, cloud, cfg, arms, rung=None, wait=True, poll_s=0, log=lambda s: None) == {"r4@2": "seed_mismatch"}
    rec = store.read_json(f"runs/{EXP}-r4-s2/noise.json")
    assert rec["train"]["seed_trained"] == N.RECIPE_SEED and "TRAIN_SEED" in rec["train"]["error"]
    # nothing past training runs for it
    assert N.build_all(store, cloud, cfg, arms, spec=None, wait=True, poll_s=0, log=lambda s: None) == {"r4@2": "waiting"}
    assert N.eval_all(store, cloud, cfg, arms, poll_s=0, log=lambda s: None) == {"r4@2": "waiting"}
    # and an eval that exists anyway (a hand run) is left out of the analysis with a warning
    store.write_json(f"runs/{EXP}-r4-s2/eval/bar.json", {"sets": []})
    res = N.analyze(store, arms, write=False, log=lambda s: None)
    assert any("not 2" in w for w in res["warnings"]) and "r4@2" not in res["models"]


def test_eval_waits_for_a_running_round(tmp_path):
    store, cloud, cfg, arms = _cloud_run(tmp_path)
    N.train(store, cloud, cfg, arms, rung=None, wait=True, poll_s=0, log=lambda s: None)
    N.build_all(store, cloud, cfg, arms, spec=None, wait=True, poll_s=0, log=lambda s: None)
    store.b.acquire(N.RETRAIN_LOCK, "rt-20261016-0101-aaaaaa-deadbeef", 3600)
    with pytest.raises(ConfigError, match="retrain.json is held"):
        N.eval_all(store, cloud, cfg, arms, poll_s=0, log=lambda s: None)
    assert not [c for c in cloud.calls if c[0] in ("run_job", "set_image")]
    assert store.b.read_lock(N.EVAL_LOCK) is None   # nothing taken
    store.b.release(N.RETRAIN_LOCK, "rt-20261016-0101-aaaaaa-deadbeef")
    store.b.acquire(N.EVAL_LOCK, "dr-20261016-0101-check", 3600)
    with pytest.raises(ConfigError, match="gpu-eval.json is held"):
        N.eval_all(store, cloud, cfg, arms, poll_s=0, log=lambda s: None)
    assert store.b.read_lock(N.RETRAIN_LOCK) is None  # released again
    store.b.release(N.EVAL_LOCK, "dr-20261016-0101-check")
    assert N.eval_all(store, cloud, cfg, arms, poll_s=0, log=lambda s: None) == {"r4@2": "done"}
    run = [c for c in cloud.calls if c[0] == "run_job"][0]
    assert run[2][:3] == ["eval", "--run-id", f"{EXP}-r4-s2"] and "--candidate-url" in run[2]
    assert run[3]["PIPELINE_CANDIDATE_ID"] == f"c-{EXP}-r4-s2" and run[3]["CAND_MODEL_DIGEST"].startswith("sha256:")
    pointed = [c for c in cloud.calls if c[0] == "set_image"]
    assert pointed[0][1] == "sous-chef-model-eval" and pointed[0][2].endswith("@sha256:" + "a" * 64)
    assert pointed[1][1] == "sous-chef-model-cand"


def test_eval_refuses_when_production_moved(tmp_path):
    store, cloud, cfg, arms = _cloud_run(tmp_path)
    N.train(store, cloud, cfg, arms, rung=None, wait=True, poll_s=0, log=lambda s: None)
    N.build_all(store, cloud, cfg, arms, spec=None, wait=True, poll_s=0, log=lambda s: None)
    cloud.images["sous-chef-model"] = "sha256:" + "c" * 64
    with pytest.raises(ConfigError, match="live model changed"):
        N.eval_all(store, cloud, cfg, arms, poll_s=0, log=lambda s: None)
    assert store.b.read_lock(N.EVAL_LOCK) is None and store.b.read_lock(N.RETRAIN_LOCK) is None


def test_analyze_rereads_a_rounds_bar_with_the_new_tolerance(tmp_path):
    N.dry(tmp_path, ["r4@2", "r4@3"], exp=EXP, log=lambda s: None)
    store = open_store(tmp_path / "bucket")
    store.write_json(f"runs/{R5}/eval/bar.json", {"sets": [
        {"set": "holdout", "delta": -3.5, "tol": 2, "bar": True, "sigma_delta": 1.0, "m": [2, 2]},
        {"set": "paraphrase", "delta": -3.0, "tol": 5, "bar": True, "sigma_delta": 2.082, "m": [2, 2]}]})
    arms = [N.parse_arm(s, EXP) for s in N.DEFAULT_ARMS]
    res = N.analyze(store, arms, compare=[R5], write=False, log=lambda s: None)
    ho = res["compare"][R5]["holdout"]
    assert ho["passes_then"] is False and ho["tol_seed"] == res["sets"]["holdout"]["tol_seed"]
    assert ho["passes_seed"] == (-3.5 >= -ho["tol_seed"])
    assert "holdout" in N.table(res)


def test_cli_dry_status_and_plan(tmp_path, capsys):
    assert N.main(["dry", "--workdir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "σ_seed" in out and "holdout" in out
    bucket = str(tmp_path / "bucket")
    assert N.main(["status", "--exp", EXP, "--bucket", bucket]) == 0
    assert "eval done" in capsys.readouterr().out
    assert N.main(["analyze", "--exp", EXP, "--bucket", bucket, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["exp"] == EXP
    assert N.main(["plan", "--exp", EXP, "--offline", "--arm", "r4@2", "--arm", "r4@3", "--arm", "r4@3407"]) == 0
    plan = capsys.readouterr().out
    c = N.costs(3)
    assert f"US${c['total_usd']}" in plan and "Claude US$0" in plan and "nz-20261015-r4-s3407" in plan
    assert N.main(["setup", "--exp", "bad", "--bucket", bucket]) == 2


def test_costs():
    c = N.costs(2)
    assert c["train_usd"] == round(2 * N.H100_HOURS * N.H100_RATE, 2)
    assert c["eval_usd"] == round(2 * N.EVAL_HOURS * 2 * N.L4_RATE, 2)
    assert c["claude_usd"] == 0 and 40 < c["total_usd"] < 90


class _Rec:
    def __init__(self, outputs):
        self.argv, self.outputs = [], outputs

    def __call__(self, argv, stdin):
        self.argv.append(argv)
        for key, out in self.outputs.items():
            if key in argv:
                return subprocess.CompletedProcess(argv, 0, out.encode(), b"")
        return subprocess.CompletedProcess(argv, 0, b"", b"")


def test_gcloud_commands(tmp_path):
    rec = _Rec({"create": "projects/1/locations/us-central1/customJobs/77\n", "submit": "bid-1\n",
                "execute": "souschef-eval-abc\n"})
    g = N.GcloudCloud("fake-project", runner=rec, log=lambda s: None)
    store, _ = N.seed_fake_bucket(tmp_path / "b")
    cfg = N.config(store)
    arm = N.parse_arm("r4@2", EXP)
    body = N.job_body(cfg, arm, N.pick_rung(cfg, None), image="img", attempt=1, nonce="n1")
    assert g.create_custom_job("us-central1", body) == "projects/1/locations/us-central1/customJobs/77"
    argv = rec.argv[-1]
    assert argv[:4] == ["gcloud", "ai", "custom-jobs", "create"] and "--project=fake-project" in argv
    cfgfile = next(a.split("=", 1)[1] for a in argv if a.startswith("--config="))
    assert json.loads(Path(cfgfile).read_text())["workerPoolSpecs"][0]["containerSpec"]["args"] == [arm.run_id, "full"]
    assert any(a.startswith("--labels=") and "purpose=sous-chef-noise" in a for a in argv)
    assert g.submit_build(region="us-east4", source="gs://b/runs/x/model/context.tar.gz", config=Path("/s.json"),
                          sa="projects/p/serviceAccounts/sa", subs={"_BUCKET": "b", "_RUN_ID": "x", "_IMAGE": "i:c-x"}) == "bid-1"
    assert "--substitutions=_BUCKET=b,_RUN_ID=x,_IMAGE=i:c-x" in rec.argv[-1] and "--async" in rec.argv[-1]
    assert g.run_job("souschef-eval", "r", ["eval", "--run-id", "x"], {"A": "1"}, 64800) == "souschef-eval-abc"
    assert "--args=eval,--run-id,x" in rec.argv[-1] and "--task-timeout=64800s" in rec.argv[-1]
    # execution states
    for doc, want in (({"status": {"succeededCount": 1}}, "succeeded"),
                      ({"status": {"conditions": [{"type": "Completed", "status": "False", "message": "x"}]}}, "failed"),
                      ({"status": {"conditions": [{"type": "Completed", "status": "Unknown"}]}}, "running")):
        g2 = N.GcloudCloud("p", runner=_Rec({"executions": json.dumps(doc)}), log=lambda s: None)
        assert g2.execution("r", "e")["state"] == want
    # production's serving digest: the revision with the largest share
    svc = {"status": {"traffic": [{"revisionName": "old", "percent": 0}, {"revisionName": "new", "percent": 100}]}}
    rev = {"status": {"imageDigest": "us-east4-docker.pkg.dev/p/r/model@sha256:" + "d" * 64}}

    class _Seq(_Rec):
        def __call__(self, argv, stdin):
            self.argv.append(argv)
            doc = svc if "services" in argv else rev
            return subprocess.CompletedProcess(argv, 0, json.dumps(doc).encode(), b"")
    seq = _Seq({})
    g3 = N.GcloudCloud("p", runner=seq, log=lambda s: None)
    assert g3.serving_digest({"name": "sous-chef-model", "region": "us-east4"}, "repo") == "sha256:" + "d" * 64
    assert "new" in seq.argv[-1]
    # a failure names the command, never its output
    bad = N.GcloudCloud("p", runner=lambda a, s: subprocess.CompletedProcess(a, 1, b"", b"ERROR: (gcloud) denied\n"),
                        log=lambda s: None)
    with pytest.raises(N.StageError, match="denied"):
        bad.custom_job("r", "j")
