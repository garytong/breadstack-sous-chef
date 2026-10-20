"""tools/mxarm.py for r7's round (rt-20261018-0213-…): the round table and its pairs, the local eval store with r6c
live and r4's recorded reference, the control's distance table, r4_floor in the verdict, the r7 report (labels,
report-only r4-mxt, latency), prep-run's carry-over, the stage 3 gates, the r4-mxt route temperature and the GIN
access-log fix. Offline: no GCP, no GPU, no model."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent.parent), str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import mxarm as M  # noqa: E402
from test_mx_latency import SMOKES, gin, timing  # noqa: E402
from test_mxarm import MX_IMG, arm_rows, bar, tree_digest  # noqa: E402

EXP = M.EXP_ROOT
R6, R7 = M.R6, M.R7
RUN6 = M.R6_ARM_RUN
RUN = "mx-20261018-m3-8b-r7"
R6C, R7C = "sha256:" + "6" * 64, "sha256:" + "7" * 64
R7_AT = "2026-10-18T20:00:00Z"  # r7's eval finished (report.json at)
REF_BYTES = json.dumps({"digest": M.R4_DIGEST, "replicates": [{"rows": {"paraphrase:p1": 1}}], "who": "r4-recorded"}).encode()


# --- the round table -----------------------------------------------------------------------------------------------------

def test_rounds_and_their_pairs():
    assert M.pairs_for(R6, report_only=False) == {"r4-mx": ("r4", "mx"), "r6c-mx": ("r6c", "mx"), "r4-r6c": ("r4", "r6c")}
    assert M.pairs_for(R7, report_only=False) == {"r4-mx": ("r4", "mx"), "r7c-mx": ("r7c", "mx"), "r6c-r7c": ("r6c", "r7c")}
    assert M.pairs_for(R7)["r4-mxt"] == ("r4", "mxt")
    assert M.control_pair(R6) == "r4-r6c" and M.control_pair(R7) == "r6c-r7c"
    assert M.main_pairs(R7) == ["r7c-mx", "r4-mx"]
    assert M.model_uri("r7c", RUN) == f"{M.BUCKET}/runs/{R7}/train/stage"
    assert M.model_uri("r6c", RUN, R7) == f"{M.BUCKET}/runs/{R6}/train/stage"  # the previous round's, whatever the cases run
    assert M.model_uri("mxt", RUN) == M.model_uri("mx", RUN) == f"{EXP}/runs/{RUN}/train/stage"
    assert M.train_done_uri("r7c", RUN) == f"{M.BUCKET}/runs/{R7}/train/DONE" and M.train_done_uri("r4", RUN) is None
    with pytest.raises(M.ConfigError, match="ROUNDS"):
        M.round_info("rt-20991231-0000-000000-00000000")
    with pytest.raises(M.ConfigError):
        M.check_pair("r6c-mx", R7)  # r6's pair names are not r7's


# --- the local eval store -------------------------------------------------------------------------------------------------

def seed_r7_bucket(root: Path, *, live: str = "r6c") -> None:
    """The pipeline bucket after r6's approval (r6c live), r7's round evaluated, the r4 snapshot and reference."""
    from souschef_lab import gcsio
    b = gcsio.LocalBucket(root)
    states = {"r6c": {"tag": f"c-{R6}", "digest": R6C, "training_manifest": "m6", "run_id": R6},
              "r7c": {"tag": f"c-{R7}", "digest": R7C, "training_manifest": "m7", "run_id": R7}}
    r4_state = {"tag": "r4", "digest": M.R4_DIGEST, "training_manifest": M.R4_MANIFEST}
    docs = {"state/model.json": states[live], "state/deployed.json": {"hash": "m7"},
            "manifests/m6.json": {"x": 6}, "manifests/m7.json": {"x": 7}, f"manifests/{M.R4_MANIFEST}.json": {"x": 4},
            f"baselines/{R6C}/baseline.json": {"who": "r6-live", "digest": R6C, "training_manifest": "m6"},
            f"baselines/{M.R4_DIGEST}/baseline.json": {"who": "r4", "digest": M.R4_DIGEST},
            f"{M.SNAP_REL}/snapshot.json": {"digest": M.R4_DIGEST}, f"{M.SNAP_REL}/state/model.json": r4_state,
            f"{M.SNAP_REL}/state/deployed.json": {"hash": "m4"}, f"{M.SNAP_REL}/manifests/{M.R4_MANIFEST}.json": {"x": 4},
            f"{M.SNAP_REL}/baselines/{M.R4_DIGEST}/baseline.json": {"who": "r4-snap", "digest": M.R4_DIGEST},
            f"runs/{R6}/model/image.json": {"digest": R6C, "tag": f"c-{R6}"},
            f"runs/{R6}/eval/candidate-baseline.json": {"who": "r6-cand", "digest": R6C, "training_manifest": "m6"},
            f"runs/{R7}/run.json": {"trigger": {}}, f"runs/{R7}/prepare/manifest.json": {"hash": "m7"},
            f"runs/{R7}/model/image.json": {"digest": R7C, "tag": f"c-{R7}"},
            f"runs/{R7}/eval/candidate-baseline.json": {"who": "r7-cand", "digest": R7C, "training_manifest": "m7"},
            f"runs/{R7}/eval/bar.json": {"pass": True},
            f"runs/{R7}/report.json": {"at": R7_AT, "results": {"current": {"digest": R6C}}}}
    for k, v in docs.items():
        b.write_json(k, v)
    b.write_bytes(f"runs/{R7}/prepare/cases.jsonl", b'{"id": "c1"}\n')
    b.write_bytes("baselines/reference/r4/baseline.json", REF_BYTES)
    b.write_bytes(f"runs/{R6}/eval/current/rows.jsonl", b'{"key": "paraphrase:p1", "status": "ok", "pass": 1}\n')


@pytest.mark.parametrize("pair_", ["r6c-r7c", "r7c-mx", "r4-mx", "r4-mxt"])
def test_mirror_for_r7(tmp_path, pair_):
    from souschef_lab.retrain.common import open_store
    seed_r7_bucket(tmp_path / "bucket")
    before = tree_digest(tmp_path / "bucket")
    cur, cand = M.pairs_for(R7)[pair_]
    dst = open_store(str(tmp_path / "l"))
    log = M.mirror_store(open_store(str(tmp_path / "bucket")), dst, R7, current=cur, candidate=cand, run=RUN)
    assert tree_digest(tmp_path / "bucket") == before
    model = dst.read_json("state/model.json")
    want = {"r6c": (R6C, "r6-live"), "r7c": (R7C, "r7-cand"), "r4": (M.R4_DIGEST, "r4-snap")}[cur]
    assert model["digest"] == want[0] and dst.read_json(f"baselines/{want[0]}/baseline.json")["who"] == want[1]
    if cur == "r6c":  # the live model: state and baseline as r7's own eval read them
        assert "mx_note" not in model and any("live model" in x for x in log["patched"])
    img = dst.read_json(f"runs/{R7}/model/image.json")
    assert img["digest"] == (R7C if cand == "r7c" else f"{cand}:{RUN}")
    assert dst.read_json(f"runs/{R7}/eval/bar.json") is None
    # r7's bar reads r4's recorded reference and r6's round's current rows: both mirrored
    assert dst.read_bytes("baselines/reference/r4/baseline.json") == REF_BYTES
    assert dst.read_bytes(f"runs/{R6}/eval/current/rows.jsonl") is not None
    assert any("pinned" in w for w in log.get("warnings") or [])  # the synthetic copy is not the pinned file


def test_the_mirrored_reference_is_what_eval_stage_loads(tmp_path):
    from souschef_lab import pipeline_common as PC
    from souschef_lab.retrain import eval_stage
    from souschef_lab.retrain.common import open_store
    seed_r7_bucket(tmp_path / "bucket")
    dst = open_store(str(tmp_path / "l"))
    M.mirror_store(open_store(str(tmp_path / "bucket")), dst, R7, current="r6c", candidate="mx", run=RUN)
    th = PC.load_thresholds(None)
    th["r4_reference"] = dict(th["r4_reference"], file_sha256=hashlib.sha256(REF_BYTES).hexdigest())
    reps, doc, info = eval_stage.load_reference(dst, th)
    assert info["source"] == "baselines/reference/r4/baseline.json" and doc["who"] == "r4-recorded" and reps["paraphrase"]
    assert eval_stage.reference_rows(dst, th)  # r6's round's current arm (r4), for the re-run floor and clear asks


def test_a_previous_candidate_no_longer_live_keeps_its_approved_baseline(tmp_path):
    from souschef_lab.retrain.common import open_store
    seed_r7_bucket(tmp_path / "bucket", live="r7c")  # r7 approved before MX's eval ran
    dst = open_store(str(tmp_path / "l"))
    log = M.mirror_store(open_store(str(tmp_path / "bucket")), dst, R7, current="r6c", candidate="r7c", run=RUN)
    assert dst.read_json("state/model.json")["digest"] == R6C and dst.read_json("state/model.json")["training_manifest"] == "m6"
    assert dst.read_json(f"baselines/{R6C}/baseline.json")["who"] == "r6-live"  # the approved baseline, not candidate-baseline
    assert f"baselines/{R6C}/baseline.json ← baselines/{R6C}/baseline.json" in log["patched"]


def test_mirror_refuses_a_candidate_from_another_round(tmp_path):
    from souschef_lab.retrain.common import open_store
    seed_r7_bucket(tmp_path / "bucket")
    with pytest.raises(M.ConfigError):
        M.mirror_store(open_store(str(tmp_path / "bucket")), open_store(str(tmp_path / "l")), R7, current="r4",
                       candidate="r6c", run=RUN)


# --- trust: the control's distance from the round ------------------------------------------------------------------------------

def sets_(*rows):
    return [{"set": s, "delta": d, "tol": t, "bar": counts, "n": 100, "current": cur, "candidate": cur + d, "b": b, "c": c,
             "m": [2, 2]} for s, d, t, counts, cur, b, c in rows]


def test_trust_reports_every_sets_distance_and_tolerance():
    rnd = {"pass": False, "pooled": {"b": 20, "c": 60}, "sets": sets_(("holdout", -2.0, 2, True, 55.0, 9, 5),
                                                                       ("F9H", 36.5, 2, True, 120.0, 14, 87),
                                                                       ("paraphrase:recorded", 1.0, 3, False, 110.0, 5, 6))}
    here = {"pass": False, "pooled": {"b": 22, "c": 58}, "sets": sets_(("holdout", -1.0, 2, True, 56.0, 8, 6),
                                                                        ("F9H", 33.0, 2, True, 126.0, 14, 80),
                                                                        ("paraphrase:recorded", 4.0, 3, False, 108.0, 3, 7))}
    tr = M.trust(here, rnd, label="r7", control_current=R6C, round_current=R6C)
    # noise bound (pre-registered 2026-10-18): a 3.5-row F9H gap is inside 2σ of the eval-to-eval gap
    f9h = next(x for x in tr["sets"] if x["set"] == "F9H")
    assert abs(f9h["gap"]) > f9h["tol"] and f9h["bound"] >= abs(f9h["gap"]) and f9h["ok"], (tr["why"], f9h)
    by = {x["set"]: x for x in tr["sets"]}
    two_sigma = 2 * ((94 + 101) / 4) ** 0.5  # ≈ 13.96: the F9H bound, far above its tol of 2
    assert by["F9H"]["gap"] == -3.5 and by["F9H"]["ok"] is True
    assert by["F9H"]["bound"] == pytest.approx(two_sigma, abs=0.1) and by["F9H"]["slack"] == pytest.approx(two_sigma - 3.5, abs=0.1)
    assert by["F9H"]["d_current"] == 6.0 and by["F9H"]["d_candidate"] == 2.5
    assert by["F9H"]["gap_2sigma"] == pytest.approx(two_sigma, abs=0.1)
    assert by["holdout"]["ok"] is True and by["holdout"]["bound"] >= 2
    assert by["paraphrase:recorded"]["counts"] is False and by["paraphrase:recorded"]["ok"] is None  # shown, not counted
    assert "tol" in tr["rule"] and tr["label"] == "r7"
    md = "\n".join(M.trust_lines(tr, "r6c-r7c"))
    assert "| F9H | yes | +36.5 | +33.0 | -3.5 ok | 2 |" in md and "bound" in md
    assert "Control (r6c-r7c) trusted: **True**" in md


def test_trust_needs_the_rounds_current_model():
    rnd = {"pass": True, "pooled": {"b": 1, "c": 5}, "sets": sets_(("holdout", 0.0, 2, True, 55.0, 1, 1))}
    assert M.trust(rnd, rnd, label="r7", control_current=R6C, round_current=R6C)["trusted"]
    tr = M.trust(rnd, rnd, label="r7", control_current=M.R4_DIGEST, round_current=R6C)
    assert not tr["trusted"] and "r7's eval ran against" in tr["why"]


# --- verdict: r4_floor counts -----------------------------------------------------------------------------------------------------

def test_r4_floor_counts_in_the_verdict():
    b_ = bar(pass_=True, b=10, c=12, p=0.83)
    assert M.verdict(b_, trusted=True)["verdict"] == "on_par"  # r6's bars have no r4_floor: passes
    b_["rules"].append({"name": "r4_floor", "pass": True, "detail": "NOT MEASURED: no r4 reference"})
    assert M.verdict(b_, trusted=True)["verdict"] == "on_par"
    b_["rules"][-1] = {"name": "r4_floor", "pass": False, "detail": "below holdout"}
    v = M.verdict(b_, trusted=True, label="r7")
    assert v["verdict"] == "worse" and "r4_floor" in v["failing"]
    assert M.verdict(b_, trusted=False, label="r7")["why"] == "the control pair did not reproduce r7's own eval"


# --- the r7 report -------------------------------------------------------------------------------------------------------

def put_pair(gs, root, pair_, b_, *, info=None, rows=None, logs=None, attempt="mx-1"):
    d = f"{root}/eval/{pair_}"
    gs.put_json({"attempt": attempt, "rc": 0, "status": "ok"}, f"{d}/latest.json")
    gs.put_json(b_, f"{d}/attempts/{attempt}/eval/bar.json")
    gs.put_json(info or {}, f"{d}/attempts/{attempt}/info.json")
    for role, rs in (rows or {}).items():
        gs.put_bytes("".join(json.dumps(r) + "\n" for r in rs).encode(), f"{d}/attempts/{attempt}/eval/{role}/rows.jsonl")
    for role, text in (logs or {}).items():
        gs.put_bytes(text.encode(), f"{d}/attempts/{attempt}/ollama-{role}.log")


def test_cmd_report_for_r7(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_GS_LOCAL", str(tmp_path))
    gs = M.GS()
    root = f"{EXP}/runs/{RUN}"
    gs.put_json({"run": RUN, "cases_run": R7}, f"{root}/plan.json")  # the round comes from the plan
    gs.put_json({"verdict": "GO", "why": "Δ +4.0", "at": "t1", "carried_from": RUN6}, f"{root}/gate.json")
    rb = M.GS().path(f"{M.BUCKET}/runs/{R7}")
    rb.mkdir(parents=True)
    rnd = bar(pass_=False, sets=(("paraphrase", -3.0, 5), ("real", 2.0, 2)), b=32, c=20, p=0.13, failing=("r4_floor",))
    (rb / "eval").mkdir()
    (rb / "eval" / "bar.json").write_text(json.dumps(rnd))
    (rb / "report.json").write_text(json.dumps({"results": {"current": {"digest": R6C}}}))
    (rb / "train-config.json").write_text(json.dumps({"arms": ["base:3407", "warm:3407"]}))
    put_pair(gs, root, "r6c-r7c", bar(pass_=False, sets=(("paraphrase", -2.0, 5), ("real", 1.0, 2)), b=31, c=22,
                                      failing=("r4_floor",)), info={"current_state": {"digest": R6C}})
    log = lambda n500: (SMOKES + "\n" + timing(100.0, 2000, 900.0, 30, task=9) + "\n" +  # noqa: E731
                        gin(200, "2s", "/v1/messages?beta=true") + "\n" +
                        "\n".join(gin(500, "1s", "/v1/messages?beta=true") for _ in range(n500)))
    rows = {role: [dict(r, latency_ms=1500, usage={"calls": 1, "output_tokens": 40}) for r in arm_rows(20, passes=14)]
            for role in ("current", "candidate")}
    put_pair(gs, root, "r7c-mx", bar(pass_=True, sets=(("paraphrase", 0.0, 2),), b=0, c=0, p=1.0),
             info={"retries_500": {"current": 0, "candidate": 0}}, rows=rows, logs={"current": log(0), "candidate": log(2)})
    put_pair(gs, root, "r4-mx", bar(pass_=True, sets=(("paraphrase", 1.0, 2),), b=3, c=4, p=1.0))
    put_pair(gs, root, "r4-mxt", bar(pass_=True, sets=(("paraphrase", 3.0, 2),), b=1, c=4, p=0.38),
             info={"route_overrides": {"candidate": {"temperature": 0.15}}})
    assert M.main(["report", "--run", RUN]) == 0
    rep = json.loads(gs.path(f"{root}/report/report.json").read_text())
    s3 = rep["stage3"]
    assert rep["cases_run"] == R7 and rep["round"] == {"label": "r7", "candidate": "r7c", "control_pair": "r6c-r7c", "current": "r6c"}
    assert s3["control"]["trusted"] is True and set(s3["verdicts"]) == {"r7c-mx", "r4-mx"}
    assert s3["verdicts"]["r7c-mx"]["trigger"]["counted"] is False and s3["verdicts"]["r4-mx"]["trigger"]["counted"] is True
    assert set(s3["report_only"]) == {"r4-mxt"} and s3["report_only"]["r4-mxt"]["report_only"] is True
    # the GIN fix recounts 500s from the logs even when the attempt recorded 0 (the smokes' are not the rows')
    assert s3["pairs"]["r7c-mx"]["retries_500"] == {"current": 0, "candidate": 2}
    assert rep["latency"]["r7c-mx"]["candidate"]["calls"]["calls"] == 3
    md = gs.path(f"{root}/report/report.md").read_text()
    assert "**Verdict vs the r7 candidate (r7c): " in md and "### r7c-mx" in md and "### r6c-r7c" in md
    assert "Control (r6c-r7c) trusted: **True**" in md and "| set | counts | Δ in r7 | Δ here | gap | tol | 2σ gap | bound | slack |" in md
    assert "## Report only (never part of a verdict)" in md and "### r4-mxt — MX at temperature 0.15 vs r4 (0.3)" in md
    assert "| paraphrase | +1.0 | +3.0 | +2.0 |" in md  # r4-mx beside r4-mxt
    assert "## Latency" in md and "| r7c-mx (candidate) | mx | 3 |" in md
    assert "r6c-mx" not in md and "r4-r6c" not in md and "r7 kept the best of its arms (base:3407, warm:3407)" in md
    assert f"Carried from `{RUN6}`" in md


def test_cmd_latency_prints_and_writes_nothing_remote(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MX_GS_LOCAL", str(tmp_path / "gs"))
    gs = M.GS()
    root = f"{EXP}/runs/{RUN6}"
    log = SMOKES + "\n" + timing(100.0, 2000, 900.0, 30, task=9) + "\n" + gin(200, "2s", "/v1/messages?beta=true")
    put_pair(gs, root, "r4-mx", bar(), rows={"current": [{"set": "paraphrase", "status": "ok", "latency_ms": 3000}]},
             logs={"current": log, "candidate": log})
    before = tree_digest(tmp_path / "gs")
    assert M.main(["latency", "--run", RUN6, "--cases-run", R6, "--json", str(tmp_path / "lat.json")]) == 0
    assert tree_digest(tmp_path / "gs") == before
    out = capsys.readouterr().out
    assert "| r4-mx (current) | r4 | 1 |" in out and "| r4-mx (candidate) | mx | 1 |" in out
    assert json.loads((tmp_path / "lat.json").read_text())["by_model"]["r4"]


# --- prep-run: the round's dataset, Stage 1 and r4 carried over --------------------------------------------------------------

def test_prep_run_for_r7_carries_stage1_and_r4(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_GS_LOCAL", str(tmp_path / "gs"))
    gs = M.GS()
    broot = tmp_path / "gs" / M.BUCKET[5:]
    seed_r7_bucket(broot)
    (broot / "runs" / R7 / "result-dataset.json").write_text(json.dumps({"status": "ok", "metrics": {"rows": 4259}}))
    (broot / "runs" / R7 / "train-config.json").write_text(json.dumps({"arms": ["base:3407", "warm:3407"]}))
    (broot / "runs" / R7 / "dataset" / "dataset").mkdir(parents=True)
    (broot / "runs" / R7 / "dataset" / "dataset" / "train.jsonl").write_text("{}\n")
    old = f"{EXP}/runs/{RUN6}"
    gs.put_json({"verdict": "GO", "at": "t0"}, f"{old}/gate.json")
    gs.put_bytes(b"FROM ./r4-q4_k_m.gguf\n", f"{old}/models/r4/Modelfile")
    gs.put_bytes(b"GGUF", f"{old}/models/r4/r4-q4_k_m.gguf")
    pipeline_before = tree_digest(broot / "runs")
    old_before = tree_digest(gs.path(old))
    assert M.main(["prep-run", "--run", RUN, "--cases-run", R7, "--from-run", RUN6]) == 0
    assert tree_digest(broot / "runs") == pipeline_before and tree_digest(gs.path(old)) == old_before
    root = f"{EXP}/runs/{RUN}"
    gd = gs.read_json(f"{root}/gate.json")
    assert gd["verdict"] == "GO" and gd["carried_from"] == RUN6
    assert gs.path(f"{root}/models/r4/r4-q4_k_m.gguf").read_bytes() == b"GGUF"
    plan = gs.read_json(f"{root}/plan.json")
    assert plan["cases_run"] == R7 and plan["label"] == "r7" and plan["round_train_config"]["arms"][1] == "warm:3407"
    assert gs.exists(f"{root}/dataset/dataset/train.jsonl") and gs.read_json(f"{root}/train-config.json")["arms"] == ["base:3407"]
    # one run, one round
    assert M.main(["prep-run", "--run", RUN, "--cases-run", R6]) == 2
    # the stage 2 gate takes the carried GO (and names where it came from)
    def runner(argv, **kw):
        return subprocess.CompletedProcess(argv, 0, b"[]", b"")
    rows = {r["gate"]: r for r in M.gates("2", run=RUN, cases_run=R7, gs=gs, runner=runner, rung="a100-40-spot")}
    assert rows["stage 1 verdict"]["ok"] and RUN6 in rows["stage 1 verdict"]["detail"]
    assert rows["plan is this round"]["ok"] and rows["r7 dataset ok"]["ok"]
    assert not rows["r7 train running or done"]["ok"]  # r7 has not started training in this bucket


# --- stage 3 gates for r7 ----------------------------------------------------------------------------------------------------

def test_stage3_gates_check_the_control_reproduces_r7s_current(tmp_path):
    gs = M.GS(local=tmp_path)
    broot = tmp_path / M.BUCKET[5:]
    seed_r7_bucket(broot)
    for rr in (R6, R7):
        (broot / "runs" / rr / "train" / "stage").mkdir(parents=True, exist_ok=True)
        (broot / "runs" / rr / "train" / "stage" / "Modelfile").write_text("x")
        (broot / "runs" / rr / "train" / "DONE").write_text("")
    gs.put_bytes(b"x", f"{EXP}/runs/{RUN}/train/DONE")
    gs.put_bytes(b"x", f"{EXP}/runs/{RUN}/models/r4/Modelfile")
    (broot / "runs" / R7 / "report.json").write_text(json.dumps({"at": R7_AT, "results": {"current": {"digest": R6C}}}))

    def runner(argv, **kw):
        return subprocess.CompletedProcess(argv, 0, b"[]", b"")
    rows = M.gates("3", run=RUN, cases_run=R7, gs=gs, runner=runner)
    assert all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]
    assert {"r7's eval ran against r6c", "r6c staged + image", "r7c train DONE", "mx train DONE", "model r4 staged"} <= {r["gate"] for r in rows}
    (broot / "runs" / R7 / "report.json").write_text(json.dumps({"at": R7_AT, "results": {"current": {"digest": M.R4_DIGEST}}}))
    bad = {r["gate"]: r for r in M.gates("3", run=RUN, cases_run=R7, gs=gs, runner=runner)}
    assert not bad["r7's eval ran against r6c"]["ok"]
    unknown = {r["gate"]: r for r in M.gates("3", run=RUN, cases_run=R7, gs=gs, runner=runner, pairs=["r6c-mx"])}
    assert not unknown["pairs known"]["ok"]


# --- r4-mxt: MX at Mistral's temperature, its route only ------------------------------------------------------------------

def test_the_mxt_route_alone_runs_at_mistrals_temperature(tmp_path):
    make = M.harness_factory({"candidate": {"temperature": 0.15}}, binary=sys.executable)
    h = make({"current": "http://127.0.0.1:1", "candidate": "http://127.0.0.1:2"}, workdir=tmp_path)
    routes = h.model_routes()
    assert routes["candidate"]["temperature"] == 0.15 and routes["current"]["temperature"] == 0.3
    assert routes["candidate"]["prime"] == routes["current"]["prime"] == "off"
    assert M.harness_factory(binary=sys.executable)({"current": "http://x"}, workdir=tmp_path).model_routes()["current"]["temperature"] == 0.3


def test_mxt_temperature_bounds_and_bodies():
    assert M.check_mxt_temperature(None) == 0.15 and M.check_mxt_temperature(0.1) == 0.1
    for t in (0.16, 0.3, -0.1):
        with pytest.raises(M.ConfigError):
            M.check_mxt_temperature(t)
    b = M.job_body("eval", run=RUN, image=MX_IMG, pair="r4-mxt", cases_run=R7, create="mx-1")
    args = b["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["args"]
    assert args[args.index("--mxt-temperature") + 1] == "0.15" and args[args.index("--cases-run") + 1] == R7
    b2 = M.job_body("eval", run=RUN, image=MX_IMG, pair="r7c-mx", cases_run=R7, create="mx-1")
    assert "--mxt-temperature" not in b2["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["args"]
    with pytest.raises(M.ConfigError):
        M.job_body("eval", run=RUN, image=MX_IMG, pair="r6c-mx", cases_run=R7, create="mx-1")


def test_jobs_for_r7_default_to_its_three_main_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("jobs must not run anything"))
    assert M.main(["jobs", "--run", RUN, "--cases-run", R7, "--stage", "eval", "--mx-image", MX_IMG, "--create", "mx-9",
                   "--out-dir", str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        f"souschef-mx-eval-{RUN}-{p}-1.json" for p in ("r4-mx", "r7c-mx", "r6c-r7c"))
    assert M.main(["jobs", "--run", RUN, "--cases-run", R7, "--stage", "eval", "--mx-image", MX_IMG, "--pair", "r4-mxt",
                   "--mxt-temperature", "0.3"]) == 2
    tr = M.job_body("train", run=RUN, image=MX_IMG.replace("sous-chef-mx@", "sous-chef-trainer-mx@"), rung="h100-spot",
                    cases_run=R7, allow_after_round_train=True, create="mx-2")
    assert tr["jobSpec"]["scheduling"]["strategy"] == "SPOT" and tr["jobSpec"]["workerPoolSpecs"][0]["machineSpec"]["machineType"] == "a3-highgpu-1g"
    with pytest.raises(M.ConfigError, match="r7's training ladder"):
        M.job_body("train", run=RUN, image=MX_IMG, rung="h100-spot", cases_run=R7)


def test_an_eval_refuses_another_rounds_pair_before_writing(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_GS_LOCAL", str(tmp_path))
    assert M.main(["eval", "--run", RUN, "--cases-run", R7, "--pair", "r6c-mx", "--workdir", str(tmp_path / "w")]) == 2
    assert M.main(["eval", "--run", RUN, "--cases-run", R7, "--pair", "r4-mxt", "--mxt-temperature", "0.3",
                   "--workdir", str(tmp_path / "w")]) == 2
    assert not (tmp_path / M.BUCKET[5:]).exists()


# --- the access-log fix ----------------------------------------------------------------------------------------------------------

def test_retries_count_the_sdks_beta_requests(tmp_path):
    p = tmp_path / "ollama.log"
    p.write_text("\n".join([gin(500, "1s", "/v1/messages"), gin(200, "1s", "/v1/messages?beta=true"),
                            gin(500, "1s", "/v1/messages?beta=true"), gin(500, "2s", "/v1/messages?beta=true")]))
    c = M.log_counts([p], None)
    assert c["ollama_status"]["/v1/messages"] == {"500": 3, "200": 1}
    assert M.retries_500(c, {"smoke_500s": 1}) == 2  # the smoke's own 500 left out
    assert M.log_counts_text(p.read_text()) == {"ollama_status": c["ollama_status"]}


# --- review fixes: the baseline pin, the tolerance check, the H100 guard -------------------------------------------------

def _reps(*ats):
    return [{"at": at, "run": run, "rows": {"holdout:h1": 1, "holdout:h2": p, "paraphrase:p1": 1}}
            for (at, run), p in zip(ats, [1, 0, 1, 0, 1])]


def test_pin_baseline_drops_replicates_after_the_rounds_eval():
    from souschef_lab import baseline as BL
    b = BL.recompute({"replicates": _reps(("2026-10-17T20:31:41Z", R6), ("2026-10-17T20:31:41Z", R6),
                                          ("2026-10-19T05:40:00Z", "dr-weekly#T1"))})
    assert len(b["per_set"]["holdout"]["runs"]) == 3
    pinned, dropped = M.pin_baseline(b, R7_AT)
    assert dropped == [{"at": "2026-10-19T05:40:00Z", "run": "dr-weekly#T1"}]
    assert len(pinned["replicates"]) == 2 and pinned["per_set"]["holdout"]["runs"] == [2, 1]  # recomputed
    assert len(b["replicates"]) == 3  # the input is left alone
    same, none = M.pin_baseline(pinned, R7_AT)
    assert none == [] and same is pinned
    # a drift replicate BEFORE the round's eval was read by it: kept
    early = BL.recompute({"replicates": _reps(("2026-10-17T20:31:41Z", R6), ("2026-10-19T05:40:00Z", "dr#T1"))})
    assert M.pin_baseline(early, "2026-10-19T09:00:00Z")[1] == []
    with pytest.raises(M.ConfigError):
        M.pin_baseline(b, "not a time")


@pytest.mark.parametrize("live", ["r6c", "r7c"])
def test_mirror_pins_the_current_baseline_to_r7s_eval(tmp_path, live):
    from souschef_lab import baseline as BL
    from souschef_lab.retrain.common import open_store
    root = tmp_path / "bucket"
    seed_r7_bucket(root, live=live)
    from souschef_lab import gcsio
    b = gcsio.LocalBucket(root)
    r6 = BL.recompute({"who": "r6-live", "digest": R6C, "training_manifest": "m6",
                       "replicates": _reps(("2026-10-17T20:31:41Z", R6), ("2026-10-17T20:31:41Z", R6),
                                           ("2026-10-19T05:40:00Z", "dr-weekly#T1"))})
    b.write_json(f"baselines/{R6C}/baseline.json", r6)
    dst = open_store(str(tmp_path / "l"))
    log = M.mirror_store(open_store(str(root)), dst, R7, current="r6c", candidate="r7c", run=RUN)
    got = dst.read_json(f"baselines/{R6C}/baseline.json")
    assert [r["at"] for r in got["replicates"]] == ["2026-10-17T20:31:41Z"] * 2
    pin = log["baseline_pin"]
    assert pin["cutoff"] == R7_AT and pin["replicates"] == 2 and pin["dropped"][0]["run"] == "dr-weekly#T1"
    assert pin["sha256"] == hashlib.sha256(dst.read_bytes(f"baselines/{R6C}/baseline.json")).hexdigest()
    assert b.read_json(f"baselines/{R6C}/baseline.json")["replicates"][-1]["run"] == "dr-weekly#T1"  # bucket untouched


def test_mirror_refuses_without_the_rounds_eval_time(tmp_path):
    from souschef_lab.retrain.common import open_store
    seed_r7_bucket(tmp_path / "bucket")
    (tmp_path / "bucket" / "runs" / R7 / "report.json").unlink()
    with pytest.raises(M.ConfigError, match="report.json"):
        M.mirror_store(open_store(str(tmp_path / "bucket")), open_store(str(tmp_path / "l")), R7, current="r6c",
                       candidate="r7c", run=RUN)


def test_trust_flags_a_tolerance_that_differs_from_the_rounds():
    there = {"pass": True, "sets": sets_(("holdout", 1.0, 2, True, 80, 3, 4)), "pooled": {"b": 3, "c": 4}}
    here = {"pass": True, "sets": sets_(("holdout", 1.0, 3, True, 80, 3, 4)), "pooled": {"b": 3, "c": 4}}
    tr = M.trust(here, there, label="r7")
    assert not tr["trusted"] and "tol 3 here vs 2 in r7" in tr["why"]
    assert tr["sets"][0]["same_tol"] is False and tr["sets"][0]["tol_here"] == 3
    assert "(here 3)" in "\n".join(M.trust_lines(tr, "r6c-r7c"))
    same = M.trust(there, there, label="r7")
    assert same["trusted"] and same["sets"][0]["same_tol"] is True


def test_jobs_refuses_an_h100_body_before_r7s_train_done(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_GS_LOCAL", str(tmp_path))
    trainer = MX_IMG.replace("sous-chef-mx@", "sous-chef-trainer-mx@")
    argv = ["jobs", "--run", RUN, "--cases-run", R7, "--stage", "train", "--mode", "full", "--rung", "h100-spot",
            "--after-round-train-done", "--trainer-image", trainer, "--create", "mx-3", "--out-dir", str(tmp_path / "j")]
    assert M.main(argv) == 2  # the flag alone is not enough
    assert not (tmp_path / "j").exists()
    done = tmp_path / M.BUCKET[5:] / "runs" / R7 / "train" / "DONE"
    done.parent.mkdir(parents=True)
    done.write_text("")
    assert M.main(argv) == 0 and (tmp_path / "j" / f"souschef-mx-train-{RUN}-h100-spot-1.json").exists()
    # the A100 Spot rung never waits for r7
    done.unlink()
    assert M.main([a if a != "h100-spot" else "a100-40-spot" for a in argv]) == 0


def _bar(pass_, sets, b=10, c=12):
    return {"pass": pass_, "sets": sets, "pooled": {"b": b, "c": c}}


def _set(name, delta, tol=2, b=40, c=50, m=(2,), counts=True, current=100.0):
    return {"set": name, "delta": delta, "tol": tol, "bar": counts, "b": b, "c": c, "m": list(m), "n": 224,
            "current": current, "candidate": current + delta}


def test_trust_noise_bound_admits_a_gap_inside_two_sigma():
    # the r6 case: F9H +33 here vs +36.5 in the round — outside tol 2, inside 2σ of the eval-to-eval gap
    round_bar = _bar(False, [_set("F9H", 36.5, b=14, c=80), _set("holdout", -6.5, b=19, c=6)])
    control = _bar(False, [_set("F9H", 33.0, b=14, c=80), _set("holdout", -5.0, b=14, c=4)])
    tr = M.trust(control, round_bar, label="r6")
    f9h = next(x for x in tr["sets"] if x["set"] == "F9H")
    assert tr["trusted"], tr["why"]
    assert f9h["bound"] > f9h["tol"] and f9h["ok"] and abs(f9h["gap"]) > f9h["tol"]


def test_trust_noise_bound_still_rejects_a_gap_beyond_it():
    round_bar = _bar(True, [_set("holdout", 5.0, b=4, c=6)])
    control = _bar(True, [_set("holdout", -9.0, b=4, c=6)])
    tr = M.trust(control, round_bar, label="r7")
    assert not tr["trusted"] and "bound" in tr["why"]


def test_trust_a_differing_bar_within_noise_is_a_note_not_a_failure():
    # r7 failed its bar by 0.2 of a row on followup; a re-run may land either side of the line
    round_bar = _bar(False, [_set("followup", -2.2, b=3, c=1)])
    control = _bar(True, [_set("followup", -1.5, b=3, c=2)])
    tr = M.trust(control, round_bar, label="r7")
    assert tr["trusted"], tr["why"]
    assert any("within noise" in n for n in tr["notes"])
    out = "\n".join(M.trust_lines(tr, "r6c-r7c"))
    assert "bound" in out and "Note:" in out


def test_the_report_names_chocolate_muffin():
    assert M.MODEL_NICKNAME == "chocolate-muffin"
