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


def test_the_training_arms_go_into_the_decision_and_its_message(tmp_path):
    """r6: from-base seeds and a warm start in one round; every arm's validation scores reach the decision."""
    s = seeded(tmp_path)
    s.write_json(f"runs/{RUN}/train/progress.json", {
        "eval_loss_last": 0.221, "arm": "warm:3407",
        "seeds": [{"arm": "base:3407", "start": "base", "seed": 3407, "lr": 2e-4, "eval_loss": 0.2311, "gen_match": 20,
                   "gen_n": 32, "selected": False},
                  {"arm": "warm:3407", "start": "warm", "seed": 3407, "lr": 5e-5, "eval_loss": 0.2207, "gen_match": 21,
                   "gen_n": 32, "selected": True}],
        "selection": {"selected_arm": "warm:3407", "rule": "r", "by": "gen_check+loss",
                      "skipped": [{"arm": "base:2", "reason": "the 12 h window"}]}})
    arms = arms_equal()
    bar = BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms))
    res = REP.build(s, RUN, mode="full", bar=bar, eval_info={}, now=datetime(2026, 10, 12, 13, 0, tzinfo=timezone.utc))
    t = s.read_json(f"runs/{RUN}/decision.json")["results"]["training"]["arms"]
    assert t["selected"] == "warm:3407" and [a["arm"] for a in t["arms"]] == ["base:3407", "warm:3407"]
    assert t["skipped"] == [{"arm": "base:2", "reason": "the 12 h window"}]
    msg = res["notification"]["summary"]
    assert msg.startswith("Every bar rule passes (6). Training arms (same validation rows): ")
    assert "base:3407 loss 0.2311, gen 20/32; warm:3407 loss 0.2207, gen 21/32. Kept warm:3407." in msg
    assert "Not trained: base:2 (the 12 h window)." in msg
    assert REP.training_arms({"eval_loss_last": 0.2}) is None and REP.arms_text(None) == ""


# --- r7: the r4 reference in the decision, and "asks on clear requests" (report only) ---------------------------------------

def _exam(rid, request, cls, set_="holdout", ask_ok=False, sup=None):
    meta = {"class": cls, **({"superseded": {"id": sup}} if sup else {})}
    return {"id": rid, "set": set_, "request": request, "meta": meta, "expected": {"ask_ok": True} if ask_ok else {}}


def _t(rid, verdict, set_="holdout", reasons=None, status="ok"):
    return {"key": f"{set_}:{rid}", "id": rid, "set": set_, "status": status, "verdict": verdict,
            "reasons": reasons if reasons is not None else (["asked_or_answered"] if verdict == "no_write" else [])}


EXAM = [_exam("h004", "make our logo smaller", "relative"),
        _exam("h020", "the footer text is too light", "relative"),
        _exam("h022.k3", "add our weedmaps page", "synonym", ask_ok=True, sup="h022"),
        _exam("h024.k2", "change the hover color for the category links", "synonym", ask_ok=True, sup="h024"),
        _exam("h006.k3", "the body copy font should be Inter", "synonym", sup="h006"),
        _exam("p085", "make the product titles bigger", "relative", set_="paraphrase"),
        _exam("p150.k2", "i don't like the font | okay, lora", "followup", set_="followup")]
VAGUE = {"add our weedmaps page", "change the hover color for the category links"}


def test_clear_asks_counts_questions_on_clear_requests_by_class():
    vague = lambda row: row["request"] in VAGUE  # noqa: E731 — askrule's verdict, stubbed
    cand = [_t("h004", "no_write"), _t("h004", "no_write"), _t("h020", "no_write"), _t("h020", "pass"),
            _t("h022.k3", "clarify_ok"), _t("h022.k3", "clarify_ok"),         # vague + ask_ok: not counted
            _t("h024.k2", "no_write"),                                       # vague: not counted
            _t("h006.k3", "no_write"), _t("h006.k3", "no_write", reasons=["refused_ops"]),
            _t("p085", "no_write", set_="paraphrase"), _t("p085", "error", set_="paraphrase", status="error"),
            _t("p150.k2", "no_write", set_="followup"),                      # followup: not one of the sets
            _t("h004", "no_write", set_="holdout:recorded"),                 # the recorded variant: not counted
            _t("h999", "no_write")]                                          # not in the exam: unmatched
    r4 = [_t("h004", "pass"), _t("h006", "clarify_ok"), _t("h022", "no_write")]  # an older key's ids: matched by superseded
    ca = REP.clear_asks({"candidate": cand, "r4 (recorded)": r4, "current": []}, EXAM, vague=vague)
    assert ca["measured"] and set(ca["arms"]) == {"candidate", "r4 (recorded)"}
    c = ca["arms"]["candidate"]["by_class"]
    assert c["relative"] == {"asked": 4, "n": 5}  # h004 ×2, h020 once of twice, p085 (its error row left out)
    assert c["synonym"]["asked"] == 1 and c["synonym"]["n"] == 2  # h006.k3: asked once, refused once
    assert c["all"]["asked"] == 5 and c["all"]["n"] == 7
    assert c["all"]["rows"] == ["holdout:h004", "holdout:h006.k3", "holdout:h020", "paraphrase:p085"]
    assert ca["arms"]["candidate"]["unmatched"] == 1
    r = ca["arms"]["r4 (recorded)"]["by_class"]
    # h006 under its old key: clarify_ok there is not allowed by today's key (h006.k3 is not ask_ok) → counted
    assert r["synonym"]["asked"] == 1 and r["synonym"]["n"] == 1 and r["relative"] == {"asked": 0, "n": 1}
    assert REP.clear_asks_text(ca) == "Asks on clear requests: candidate relative 4/5, all 5/7; r4 (recorded) relative 0/1, all 1/2."
    off = REP.clear_asks({"candidate": cand}, EXAM, vague=None)
    assert off["measured"] is False and REP.clear_asks_text(off) == ""


def test_the_r4_reference_and_clear_asks_reach_the_decision(tmp_path):
    from tests.test_bar import R4_SETS, cand_of, ref_doc
    s = seeded(tmp_path)
    arms = arms_equal()
    bar = BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms))
    doc = ref_doc(R4_SETS)
    src = {"source": "baselines/reference/r4/baseline.json", "sha256": "ab" * 32, "copies_match": True, "tag": "r4",
           "copies": [{"path": "baselines/reference/r4/baseline.json", "sha256": "ab" * 32}], "warnings": []}
    BAR.add_reference(bar, cand_of({**R4_SETS, "holdout": (50, 38)}), BAR.reference_replicates(doc, BAR.REFERENCE_SETS),
                      doc, TH, source=src)
    ca = REP.clear_asks({"candidate": [_t("h004", "no_write")]}, EXAM, vague=lambda r: False)
    res = REP.build(s, RUN, mode="full", bar=bar, eval_info={"clear_asks": ca},
                    now=datetime(2026, 10, 12, 13, 0, tzinfo=timezone.utc))
    dec = s.read_json(f"runs/{RUN}/decision.json")
    r = dec["results"]
    assert dec["bar_pass"] is False and any(x["name"] == "r4_floor" and not x["pass"] for x in r["bar"]["rules"])
    ref = r["r4_reference"]
    assert ref["measured"] and ref["source"] == "baselines/reference/r4/baseline.json" and ref["sha256"] == "ab" * 32
    assert ref["sets"] == list(BAR.REFERENCE_SETS) and ref["pooled"]["b"] >= 12 and ref["excluded_amended"] == 0
    assert ref["per_set"]["holdout"]["delta"] == -6.0
    assert any(x["set"] == "holdout:vs-r4" for x in r["sets"])
    assert "rows" not in r["clear_asks"]["arms"]["candidate"]["by_class"]["all"]
    msg = res["notification"]["summary"]
    assert msg.startswith("Bar rules failing: r4_floor: below holdout")
    assert "vs r4's recorded answers: paraphrase +0.0 (tol" in msg and "holdout -6.0" in msg
    assert "Against r4's" not in msg  # the failing rule's detail carries the line once
    assert "Asks on clear requests: candidate relative 1/1, all 1/1." in msg
    rep = s.read_json(f"runs/{RUN}/report.json")
    assert rep["warnings"] == [] and "asks on clear requests, candidate: relative 1/1, all 1/1" in rep["summary"]
    assert "holdout:vs-r4" in rep["summary"]


def test_a_missing_r4_reference_is_a_warning(tmp_path):
    s = seeded(tmp_path)
    arms = arms_equal()
    bar = BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms))
    BAR.add_reference(bar, {}, None, None, TH, source={"looked_in": ["baselines/reference/r4/baseline.json"],
                                                        "warnings": ["no r4 baseline at baselines/reference/r4/baseline.json"]})
    res = REP.build(s, RUN, mode="full", bar=bar, eval_info={"clear_asks": REP.clear_asks({}, [], vague=None)},
                    now=datetime(2026, 10, 12, 13, 0, tzinfo=timezone.utc))
    assert res["notification"]["summary"].startswith("Every bar rule passes (7). Against r4's recorded answers: NOT MEASURED")
    rep = s.read_json(f"runs/{RUN}/report.json")
    assert rep["warnings"] == ["r4_floor: no r4 baseline at baselines/reference/r4/baseline.json",
                               "clear_asks: not measured (the lab has no askrule module)"]
    assert rep["results"]["r4_reference"]["measured"] is False


def test_clear_asks_with_the_real_ask_rule_on_the_exam():
    """askrule (r7 item 1) on the exam's own rows: the over-asked relative requests are clear, the ask_ok ones vague."""
    import pytest
    try:
        import souschef_lab.askrule  # noqa: F401 — r7 item 1, another builder's module
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"askrule is not importable: {type(e).__name__}")
    from souschef_lab.exam import MANIFEST_FILE, load_set
    from souschef_lab.manifest import Manifest
    import json
    m = Manifest(json.loads(MANIFEST_FILE.read_text(encoding="utf-8")))
    vague = REP.askrule_vague(m)
    hold = {r["id"]: r for r in load_set("holdout")}
    assert not vague(hold["h004"]) and not vague(hold["h020"]) and not vague(hold["h006.k3"])  # smaller / too light / Inter
    assert vague(hold["h022.k3"]) and vague(hold["h024.k2"])
    rows = [_t("h004", "no_write"), _t("h022.k3", "clarify_ok"), _t("h006.k3", "no_write")]
    ca = REP.clear_asks({"candidate": rows}, list(hold.values()), vague=vague)
    assert ca["arms"]["candidate"]["by_class"]["all"] == {"asked": 2, "n": 2, "rows": ["holdout:h004", "holdout:h006.k3"]}
