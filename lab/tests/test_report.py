"""retrain/report.py: the round's report, the decision payload (§4.4 results) and the
candidate_ready notification (§4.1)."""
from __future__ import annotations

from datetime import datetime, timezone

from souschef_lab.retrain import bar as BAR
from souschef_lab.retrain import report as REP
from souschef_lab.retrain.common import open_store

from tests.test_bar import TH, arms_equal, where_all

RUN = "rt-20261012-0310-ccd1afab"


def seeded(tmp_path):
    s = open_store(tmp_path / "bucket")
    pre = f"runs/{RUN}"
    s.write_json("state/model.json", {"tag": "r4", "digest": "sha256:aaa", "training_manifest": "bf788701" + "0" * 56})
    s.write_json(f"{pre}/run.json", {"run_id": RUN, "deployed": {"hash": "ccd1afab" + "1" * 56, "launcher_build": "1a2b3c4",
                                                                 "generated_from": {"sha": "e7e90106"}},
                                     "trigger": {"drift_run": "dr-1", "reason": "(d) 3 new keys"},
                                     "execution_url": "https://console.cloud.google.com/workflows/x"})
    s.write_json("drift/dr-1/static.json", {"counts": {"R1": 312, "R2": 4, "R3": 2},
                                            "impact": {"new_keys": ["globalPanels.ButtonsSection.buttonShadowColor"]}})
    s.write_json(f"{pre}/result-prepare.json", {"status": "ok", "metrics": {"cases": 3600, "authored_trainable": 40,
                                                                            "authored_holdout": 8, "anthropic_usd": 3.2,
                                                                            "targets": 6}})
    s.write_json(f"{pre}/prepare/report.json", {"manifest": "ccd1afab" + "1" * 56, "engine": {"status": "current"},
                                                "rebase": {"totals": {"in": 3577}, "dropped_by_reason": {"key_removed": 3}}})
    s.write_json(f"{pre}/result-collect.json", {"status": "ok", "metrics": {"teacher_usd": 151.2}})
    s.write_json(f"{pre}/result-dataset.json", {"status": "ok", "metrics": {"train": 2600, "val": 240, "followups": 310,
                                                                            "updates": 650}})
    s.write_json(f"{pre}/train/progress.json", {"eval_loss_first": 0.69, "eval_loss_last": 0.2})
    s.write_json(f"{pre}/train/job.json", {"machine": "a3-highgpu-1g", "scheduling": "FLEX_START", "wall_h": 2.4,
                                           "preemptions": 0})
    s.write_json(f"{pre}/model/image.json", {"tag": f"c-{RUN}", "digest": "sha256:bbb"})
    s.write_json("cases/pool/index.json", {"files": [{"path": "cases/pool/r4/cases-all.jsonl", "trained_in": "r4"},
                                                      {"path": "cases/pool/authored/rt-old.jsonl", "trained_in": None,
                                                       "family_counts": {"F7": 30}},
                                                      {"path": f"cases/pool/authored/{RUN}.jsonl", "trained_in": None,
                                                       "family_counts": {"F7": 40}}]})
    return s


def test_build_writes_the_decision_report_and_notification(tmp_path):
    s = seeded(tmp_path)
    arms = arms_equal()
    bar = BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms))
    now = datetime(2026, 10, 12, 13, 0, tzinfo=timezone.utc)
    res = REP.build(s, RUN, mode="full", bar=bar, eval_info={"trigger_rules": ["d"], "stale_rows": ["real:r1"]}, now=now)
    dec = s.read_json(f"runs/{RUN}/decision.json")
    assert dec["expires_at"] == "2026-10-26T13:00:00Z" and dec["smoke"] is False and dec["bar_pass"] is True
    r = dec["results"]
    assert set(r) == {"candidate", "current", "deployed_manifest", "trigger", "bar", "sets", "static", "training", "cases",
                      "cost", "stale_round", "links"}
    assert r["candidate"] == {"id": f"c-{RUN}", "tag": f"c-{RUN}", "digest": "sha256:bbb", "run_id": RUN,
                              "trained_on": "ccd1afab" + "1" * 56}
    assert r["current"]["tag"] == "r4" and r["deployed_manifest"]["engine_sha"] == "e7e90106"
    assert r["trigger"]["verdict_rules"] == ["d"] and r["static"]["R3"] == 2 and r["static"]["stale_exam_rows"] == 1
    assert r["training"]["machine"] == "a3-highgpu-1g" and r["training"]["updates"] == 650
    assert r["cases"]["authored_accumulated"] == 70 and r["cases"]["rebased_dropped"] == {"key_removed": 3}
    assert r["cost"] == {"anthropic_authoring_usd": 3.2, "teacher_usd": 151.2, "gpu_hours": 2.4}
    assert r["links"]["execution"].startswith("https://console.cloud.google.com")
    assert res["status"] == "ok" and res["candidate_id"] == f"c-{RUN}"
    n = res["notification"]
    assert n["kind"] == "candidate_ready" and "decide or skip" in n["title"]
    # the thermometer renders the headline, the table and the expiry from the decision; the summary adds the rest
    assert n["summary"] == "Every bar rule passes (6)." and "paraphrase" not in n["summary"]
    full = s.read_json(f"runs/{RUN}/report.json")["summary"]
    assert "bar PASS" in full and "Expires Oct 26" in full and "paraphrase" in full
    assert "<@" not in n["summary"]  # the thermometer adds the mention
    assert s.read_json(f"runs/{RUN}/report.json")["stages"]["collect"]["metrics"]["teacher_usd"] == 151.2


def test_smoke_expires_in_a_day_and_a_stale_round_recommends_skip(tmp_path):
    s = seeded(tmp_path)
    arms = arms_equal(paraphrase=[{f"paraphrase:r{j:03d}": 0 for j in range(40)}] * 2)
    bar = BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms))
    now = datetime(2026, 10, 12, 13, 0, tzinfo=timezone.utc)
    res = REP.build(s, RUN, mode="smoke", bar=bar, eval_info={"stale_round": True}, now=now)
    dec = s.read_json(f"runs/{RUN}/decision.json")
    assert dec["smoke"] is True and dec["expires_at"] == "2026-10-13T13:00:00Z" and dec["bar_pass"] is False
    assert dec["results"]["stale_round"] is True
    summ = res["notification"]["summary"]
    assert "SMOKE" in res["notification"]["title"] and summ.startswith("Bar rules failing: sets")
    full = s.read_json(f"runs/{RUN}/report.json")["summary"]
    assert "bar FAIL" in full and "Skip is recommended" in full


# --- the decision's bucket side (retrain/decide.py, `pipeline decision`) ---------------------------------------------

def test_approved_promotes_the_baseline_the_model_and_marks_the_pool(tmp_path):
    from souschef_lab import pipeline as P
    s = seeded(tmp_path)
    pre = f"runs/{RUN}"
    s.write_json(f"{pre}/eval/candidate-baseline.json", {"digest": "dry:x", "replicates": [{"rows": {}}, {"rows": {}}],
                                                         "training_manifest": "ccd1afab" + "1" * 56})
    s.write_json(f"{pre}/prepare/report.json", {"manifest": "ccd1afab" + "1" * 56,
                                                "pool": {"files": [{"path": "cases/pool/r4/cases-all.jsonl"},
                                                                   {"path": "cases/pool/authored/rt-old.jsonl"}]}})
    s.write_jsonl(f"cases/pool/authored/{RUN}.jsonl", [{"id": "a"}])
    s.write_jsonl(f"{pre}/prepare/cases.jsonl", [{"id": "a"}, {"id": "b"}])
    s.write_jsonl(f"{pre}/dataset/dataset/train.jsonl", [{"id": "r1"}])
    s.write_json("state/ack.json", {"candidate_id": "c-rt-old"})
    assert P.main(["decision", "--run-id", RUN, "--bucket", s.b.url[len("file://"):], "--outcome", "approved"]) == 0
    b = s.read_json("baselines/sha256:bbb/baseline.json")
    assert b["digest"] == "sha256:bbb" and b["calibrated"] is True and b["source"] == "candidate-eval"
    m = s.read_json("state/model.json")
    assert m["digest"] == "sha256:bbb" and m["training_manifest"].startswith("ccd1afab") and m["history"][-1]["tag"] == "r4"
    idx = {f["path"]: f.get("trained_in") for f in s.read_json("cases/pool/index.json")["files"]}
    assert idx["cases/pool/authored/rt-old.jsonl"] == f"c-{RUN}" and idx[f"cases/pool/authored/{RUN}.jsonl"] == f"c-{RUN}"
    assert idx["cases/pool/r4/cases-all.jsonl"] == "r4"
    assert not s.exists("state/ack.json")
    # the model's training inputs outlive runs/ (its 180-day lifecycle): drift and the playbook read the copies
    assert m["training_cases"] == [f"cases/trained/c-{RUN}/cases.jsonl"] and len(s.read_jsonl(m["training_cases"][0])) == 2
    assert m["dataset"] == f"cases/trained/c-{RUN}/dataset" and s.exists(f"cases/trained/c-{RUN}/dataset/dataset/train.jsonl")


def test_skipped_writes_the_ack_and_parks_the_candidate(tmp_path):
    from souschef_lab.retrain import decide
    s = seeded(tmp_path)
    s.write_json("drift/dr-1/ack-material.json", {"deployed_hash": "ccd1afab", "arm_D_mean": 198.0, "dodged_keys": ["k"]})
    s.write_jsonl(f"cases/pool/authored/{RUN}.jsonl", [{"id": "a"}, {"id": "b"}])
    res = decide.skipped(s, RUN, reason="expired")
    ack = s.read_json("state/ack.json")
    assert ack["arm_D_mean"] == 198.0 and ack["candidate_id"] == f"c-{RUN}" and ack["reason"] == "expired"
    assert s.read_json(f"state/parked/c-{RUN}.json")["image"]["digest"] == "sha256:bbb"
    assert res["authored_kept"] == 2
    idx = {f["path"]: f.get("trained_in") for f in s.read_json("cases/pool/index.json")["files"]}
    assert idx[f"cases/pool/authored/{RUN}.jsonl"] is None  # stays in the pool for the next big drift


def test_skipped_hand_started_round_acks_the_newest_check(tmp_path):
    # a round started by hand has no trigger.drift_run: the Skip takes the newest check's material
    from souschef_lab.retrain import decide
    s = seeded(tmp_path)
    run = s.read_json(f"runs/{RUN}/run.json")
    run["trigger"] = {"drift_run": None, "reason": "manual"}
    s.write_json(f"runs/{RUN}/run.json", run)
    s.write_json("state/last-ack-material.json", {"deployed_hash": "ccd1afab", "arm_D_mean": 190.5, "stale_exam_rows": 3,
                                                  "drift_run": "dr-9"})
    decide.skipped(s, RUN)
    ack = s.read_json("state/ack.json")
    assert ack["arm_D_mean"] == 190.5 and ack["stale_exam_rows"] == 3 and ack["drift_run"] == "dr-9"


def test_approved_needs_a_real_digest(tmp_path):
    from souschef_lab.retrain import decide
    from souschef_lab.retrain.common import ConfigError
    import pytest
    s = seeded(tmp_path)
    s.write_json(f"runs/{RUN}/model/image.json", {"tag": "x"})
    with pytest.raises(ConfigError):
        decide.approved(s, RUN)
