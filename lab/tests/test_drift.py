"""drift: the static diff and its classes, the verdict rules (with Skip's ack and
the confirmation replicate), poll's debounce and heartbeat, and the DRY
end-to-end path (fake service in process, `--fake-engine`: no node, no network,
no model). The same DRY path with the real engine is test_drift_dry_real_engine
(skipped without node + COOKIE_CUTTER_DIR)."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB / "tests"))

from souschef_lab import drift as D  # noqa: E402
from souschef_lab import gcsio  # noqa: E402
from souschef_lab import pipeline_common as PC  # noqa: E402
from souschef_lab.manifest import Manifest  # noqa: E402
from test_examrebase import REAL, drifted, rows_all  # noqa: E402

TH = PC.load_thresholds()
DIGEST = "sha256:" + "a" * 64


# ── static diff ──

def test_static_diff_classes_and_rename():
    sd = D.static_diff(REAL, drifted())
    assert sd["counts"][D.R3] == 6 and sd["counts"][D.R2] == 1 and sd["counts"][D.R1] == 1
    assert [r["from"] for r in sd["renames"]] == ["globalPanels.ButtonsSection.buttonColor"]
    assert sd["renames"][0]["to_key"] == "buttonPrimaryColor" and sd["renames"][0]["score"] == 1.0
    assert sd["fields_removed"] == ["globalPanels.ShopMenuSection.shopMenuMaxItems"]
    assert sd["containers_added"] == ["globalPanels.LoyaltyBanner"]
    assert "field.source" in sd["by_change"][D.R0] and sd["by_change"][D.R1] == {"field.label": 1}
    assert D.static_diff(REAL, copy.deepcopy(REAL))["identical"]


def test_wording_only_and_key_order_are_not_schema():
    new = copy.deepcopy(REAL)
    f = new["globalPanels"]["FooterSection"]["fields"]["footerBgColor"]
    f["help"] = "A new help text."
    f["label"] = "Footer colour"
    reordered = {k: new["globalPanels"]["FooterSection"]["fields"][k]
                 for k in reversed(list(new["globalPanels"]["FooterSection"]["fields"]))}
    new["globalPanels"]["FooterSection"]["fields"] = reordered
    sd = D.static_diff(REAL, new)
    assert sd["counts"] == {"R0": 0, "R1": 2, "R2": 0, "R3": 0}


def test_retype_new_op_and_list_changes_are_hard():
    new = copy.deepcopy(REAL)
    new["globalPanels"]["FooterSection"]["fields"]["footerBgColor"]["type"] = "text"
    lst = next(iter(new["sections"]["heroBanner"]["lists"].values()))
    lst["ops"] = list(lst["ops"]) + ["duplicate"]
    lst["max"] = 9
    sd = D.static_diff(REAL, new)
    assert sd["fields_retyped"] == ["globalPanels.FooterSection.footerBgColor"]
    assert sd["new_ops"] == ["duplicate"]
    changes = {c["change"] for c in sd["changes"][D.R3]}
    assert {"field_retyped", "list.new_op"} <= changes
    assert any(c["change"] == "list.max" for c in sd["changes"][D.R2])


def test_rename_needs_the_same_type_and_a_close_label():
    new = copy.deepcopy(REAL)
    b = new["globalPanels"]["ButtonsSection"]["fields"]
    spec = b.pop("buttonColor")
    b["accent"] = {**spec, "label": "Completely different words"}
    assert D.static_diff(REAL, new)["renames"] == []
    b["accent"] = {**spec, "type": "text"}
    assert D.static_diff(REAL, new)["renames"] == []


def test_impact_counts_stale_rows_and_new_keys():
    sd = D.static_diff(REAL, drifted())
    cases = [{"id": "c1", "manifest_hash": Manifest(REAL).hash, "holdout": False,
              "expected": {"ops": [{"op": "set_fields", "target": {"kind": "global", "panel": "ShopMenuSection"},
                                    "values": {"shopMenuMaxItems": 3}}]}, "meta": {}},
             {"id": "c2", "manifest_hash": Manifest(REAL).hash, "holdout": False,
              "expected": {"ops": [{"op": "set_fields", "target": {"kind": "global", "panel": "ButtonsSection"},
                                    "values": {"buttonColor": "#000000"}}]}, "meta": {}}]
    imp = D.impact(sd, Manifest(REAL), Manifest(drifted()), rows_all(), cases, "current")
    assert len(imp["stale_exam_rows"]) == 7 and imp["renamed_exam_rows"] == 6
    assert imp["stale_training_rows"] == 1 and imp["stale_training_pct"] == 50.0
    # a row the training manifest itself refuses (a page target, as 19 of r4's) is not "made stale" by the change
    bad = {"id": "c3", "manifest_hash": Manifest(REAL).hash, "holdout": False,
           "expected": {"ops": [{"op": "set_fields", "target": {"kind": "global", "panel": "NoSuchPanel"},
                                 "values": {"noSuchField": 1}}]}, "meta": {}}
    imp2 = D.impact(sd, Manifest(REAL), Manifest(drifted()), rows_all(), cases + [bad], "current")
    assert imp2["stale_training_rows"] == 1 and imp2["training_rows_invalid_on_old"] == 1 and imp2["stale_training_pct"] == 50.0
    assert imp["new_keys"] == ["globalPanels.ButtonsSection.buttonShadowColor", "globalPanels.LoyaltyBanner.loyaltyText",
                               "globalPanels.LoyaltyBanner.showLoyalty"]


# ── verdict ──

def _cmp(delta: float, sd: float = 1.5, n: int = 252, sets: dict | None = None) -> dict:
    out = {"total": {"n": n, "delta": delta, "sigma_delta": sd, "base_mean": 212.0, "arm_mean": 212.0 - delta, "m": 1,
                     "mcnemar_p": 0.01}}
    out.update(sets or {})
    return out


def _ev(**kw):
    base = dict(cmp=_cmp(0.0), static=D.static_diff(REAL, copy.deepcopy(REAL)), imp={"stale_exam_rows": [], "stale_exam_pct": 0.0,
                "stale_training_pct": 0.0, "engine_status": "current"}, probe={}, ghost_d=0.0, ghost_t=0.0, age_days=5,
                th=TH, confirmed=False, calibrated=True)
    base.update(kw)
    return D.evaluate(**base)


def test_verdict_ok_notice_and_retrain_need_confirmation():
    assert _ev()["verdict"] == "ok"
    assert _ev(cmp=_cmp(3.5))["verdict"] == "notice"
    unconf = _ev(cmp=_cmp(6.0))
    assert unconf["verdict"] == "notice" and unconf["needs_confirmation"]
    conf = _ev(cmp=_cmp(6.0), confirmed=True)
    assert conf["verdict"] == "retrain" and conf["fired"] == ["a"]
    small = _ev(cmp=_cmp(3.6, sd=1.0), confirmed=True)  # > 3 σ but under 4 rows
    assert small["verdict"] == "notice"


def test_set_rule_and_out_of_scope_rule():
    sets = {"followup": {"n": 12, "delta": 3.0, "sigma_delta": 0.8, "m": 2},
            "out_of_scope": {"n": 13, "delta": 2.5, "sigma_delta": 0.9, "m": 2},
            "out_of_scope_holdout": {"n": 4, "delta": 1.0, "sigma_delta": 0.5, "m": 2}}
    ev = _ev(cmp=_cmp(1.0, sets=sets), confirmed=True)
    assert ev["verdict"] == "retrain" and set(ev["fired"]) == {"b", "c"}
    # (c) needs the drop beyond 3 σ_Δ of the two sets combined (√(0.9² + 0.5²) ≈ 1.03): 3 rows is noise there
    noisy = {"out_of_scope": {"n": 13, "delta": 2.0, "sigma_delta": 0.9, "m": 2},
             "out_of_scope_holdout": {"n": 4, "delta": 1.0, "sigma_delta": 0.5, "m": 2}}
    assert "c" not in _ev(cmp=_cmp(1.0, sets=noisy), confirmed=True)["fired"]


def test_dodged_keys_stale_rows_ghosts_and_age():
    p1 = {"globalPanels.X.a": {"n": 8, "pass": 1, "dodged": True}}
    assert _ev(probe=p1)["verdict"] == "notice"
    p2 = {**p1, "globalPanels.X.b": {"n": 8, "pass": 2, "dodged": True}}
    assert _ev(probe=p2)["fired"] == ["d"]
    new_c = D.static_diff(REAL, drifted())
    p3 = {"globalPanels.LoyaltyBanner.loyaltyText": {"n": 4, "pass": 0, "dodged": True},
          "globalPanels.LoyaltyBanner.showLoyalty": {"n": 4, "pass": 0, "dodged": True}}
    assert "d" in _ev(probe=p3, static=new_c)["fired"]
    assert _ev(imp={"stale_exam_rows": ["a"] * 13, "stale_exam_pct": 5.2, "stale_training_pct": 0.0,
                    "engine_status": "current"})["fired"] == ["e"]
    assert _ev(ghost_d=5.0, ghost_t=1.0)["fired"] == ["f"]
    assert _ev(age_days=120, static=new_c)["fired"] == ["g"]
    assert _ev(age_days=120)["verdict"] == "ok"  # old, but nothing changed


def test_ack_counts_only_drift_beyond_the_skip():
    p2 = {"globalPanels.X.a": {"n": 8, "pass": 1, "dodged": True}, "globalPanels.X.b": {"n": 8, "pass": 0, "dodged": True}}
    ack = {"dodged_keys": ["globalPanels.X.a", "globalPanels.X.b"], "ghost_writes": 6.0}
    ev = _ev(probe=p2, ack=ack, ghost_d=7.0, ghost_t=1.0, static_since_ack=D.static_diff(REAL, copy.deepcopy(REAL)))
    assert ev["verdict"] == "ok" and ev["dodged_keys_new"] == []
    p3 = {**p2, "globalPanels.X.c": {"n": 8, "pass": 0, "dodged": True}}
    assert _ev(probe=p3, ack=ack)["verdict"] == "notice"


def test_after_a_skip_stale_rows_and_age_count_only_what_is_new():
    """(e) and (g) are ack-aware (verifier 2): what was skipped never fires again by itself every week."""
    same = D.static_diff(REAL, copy.deepcopy(REAL))
    stale = {"stale_exam_rows": ["a"] * 13, "stale_exam_pct": 5.2, "exam_rows": 252, "stale_training_pct": 4.0,
             "stale_training_rows": 40, "training_rows": 1000, "training_rows_checked": 1000, "engine_status": "current"}
    assert _ev(imp=stale)["fired"] == ["e"]
    ack = {"dodged_keys": [], "stale_exam_rows": 13, "stale_training_rows": 40}
    assert _ev(imp=stale, ack=ack, static_since_ack=same)["verdict"] == "ok"
    grown = {**stale, "stale_exam_rows": ["a"] * 26, "stale_exam_pct": 10.3}
    ev = _ev(imp=grown, ack=ack, static_since_ack=same)
    assert ev["fired"] == ["e"] and "since the Skip" in ev["rules"][0]["detail"]
    new_c = D.static_diff(REAL, drifted())
    assert _ev(age_days=120, static=new_c)["fired"] == ["g"]
    assert "g" not in _ev(age_days=120, static=new_c, ack=ack, static_since_ack=same)["fired"]
    assert "g" in _ev(age_days=120, static=new_c, ack=ack, static_since_ack=new_c)["fired"]
    # the Skip's manifest unreadable (static_since_ack None): the R3 notice does not repeat the full training diff
    assert "r3" in [r["rule"] for r in _ev(static=new_c)["rules"]]
    assert "r3" not in [r["rule"] for r in _ev(static=new_c, ack=ack, static_since_ack=None)["rules"]]


def test_engine_stale_and_r3_are_notices():
    assert _ev(imp={"stale_exam_rows": [], "stale_exam_pct": 0, "stale_training_pct": None,
                    "engine_status": "stale"})["verdict"] == "notice"
    assert _ev(static=D.static_diff(REAL, drifted()))["verdict"] == "notice"


# ── the job, DRY ──

@pytest.fixture()
def bucket(tmp_path):
    b = gcsio.bucket(tmp_path / "bucket")
    b.write_json("state/model.json", {"tag": "r4", "digest": DIGEST, "training_manifest": Manifest(REAL).hash,
                                      "trained_at": "2026-10-10T00:00:00Z"})
    dep = tmp_path / "deployed.json"
    dep.write_text(json.dumps(drifted()), encoding="utf-8")
    same = tmp_path / "same.json"
    same.write_text(json.dumps(REAL), encoding="utf-8")
    return b, dep, same, tmp_path


def _run(b, mode: str, run_id: str, *rest) -> dict:
    assert D.main([mode, *rest, "--bucket", str(b.root), "--dry", "--fake-engine", "--run-id", run_id]) == 0
    return b.read_json(f"drift/{run_id}/result.json")


def test_poll_debounces_writes_the_manifest_and_beats_once_a_day(bucket):
    b, dep, same, _ = bucket
    r1 = _run(b, "poll", "dr-p1", "--deployed-file", str(dep))
    h = Manifest(drifted()).hash
    assert r1["changed"] and not r1["check_due"] and r1["deployed"]["hash"] == h
    assert b.exists(f"manifests/{h}.json") and b.exists("drift/dr-p1/static.json")
    kinds = [n["kind"] for n in r1["notifications"]]
    assert "heartbeat" in kinds and "drift_notice" in kinds  # R3 changes: a static notice
    st = b.read_json("state/deployed.json")
    st["pending_check_at"] = "2020-01-01T00:00:00Z"
    b.write_json("state/deployed.json", st)
    r2 = _run(b, "poll", "dr-p2", "--deployed-file", str(dep))
    assert not r2["changed"] and r2["check_due"] and r2["notifications"] == []
    r3 = _run(b, "poll", "dr-p3", "--deployed-file", str(same))  # a newer change restarts the debounce
    assert r3["changed"] and not r3["check_due"]
    assert "route_family" not in r3  # no int URL: nothing to compare


def _route_notices(r: dict) -> list[dict]:
    return [n for n in r.get("notifications") or [] if str(n.get("dedup_key") or "").startswith("drift:route-family:")]


def test_poll_flags_an_in_house_route_of_another_family_than_the_live_model(bucket):
    """docs/CHOCOLATE-MUFFIN.md: int's in-house route (blueberry-muffin, after /version route_aliases) must be the live
    model's family's. An apply from a tree without model_route_aliases takes the alias away while chocolate-muffin is
    live; an Approve can put a model of the other family behind the route. A mismatch that outlasts an hour posts one
    drift_notice a day; a planned switch (alias, model, state/model.json within minutes) posts nothing."""
    b, _, same, _ = bucket
    v = {"build_id": "t", "default_route": "blueberry-muffin",
         "route_info": {"blueberry-muffin": {"kind": "messages-compatible"}, "chocolate-muffin": {"parser_retries": "off"},
                        "default": {"kind": "anthropic"}}}
    s = _Srv({"/version": (200, v)})
    try:
        poll = lambda rid: _run(b, "poll", rid, "--deployed-file", str(same), "--url", s.url)  # noqa: E731
        r = poll("dr-rf1")  # r4 (qwen) behind blueberry-muffin: as it should be
        assert r["route_family"] == {"route": "blueberry-muffin", "answering": "blueberry-muffin", "route_family": "qwen",
                                     "model": "r4", "model_family": "qwen", "ok": True}
        assert not _route_notices(r) and not b.exists(D.ROUTE_FAMILY_STATE) and s.seen[-1][1] == "/version"
        # chocolate-muffin live (state/model.json family mistral) but the alias gone: no notice inside the grace hour
        b.write_json("state/model.json", {**b.read_json("state/model.json"), "tag": "cm-r7", "family": "mistral"})
        r = poll("dr-rf2")
        assert not r["route_family"]["ok"] and r["route_family"]["answering"] == "blueberry-muffin" and not _route_notices(r)
        st = b.read_json(D.ROUTE_FAMILY_STATE)
        assert st["key"] == "blueberry-muffin:qwen:cm-r7:mistral" and st["posted_on"] is None
        # still so an hour later: one notice, then none more that day
        b.write_json(D.ROUTE_FAMILY_STATE, {**st, "first_seen": "2020-01-01T00:00:00Z"})
        r = poll("dr-rf3")
        (n,) = _route_notices(r)
        assert n["kind"] == "drift_notice" and n["severity"] == "warn" and "blueberry-muffin (qwen)" in n["title"]
        assert "terraform apply from a tree without model_route_aliases" in n["summary"] and "chocolate-muffin" in n["summary"]
        assert not _route_notices(poll("dr-rf4"))
        # the alias back: matched again, the state cleared
        v["route_aliases"] = {"blueberry-muffin": "chocolate-muffin"}
        r = poll("dr-rf5")
        assert r["route_family"]["ok"] and r["route_family"]["answering"] == "chocolate-muffin" and not b.exists(D.ROUTE_FAMILY_STATE)
        # an Approve of a qwen round while the alias stays: the other way round, the same notice after the hour
        b.write_json("state/model.json", {k: x for k, x in b.read_json("state/model.json").items() if k != "family"} | {"tag": "c-rt-q"})
        r = poll("dr-rf6")
        assert not r["route_family"]["ok"] and r["route_family"]["route_family"] == "mistral" and not _route_notices(r)
        # the break-glass (in-house turns to Claude's default route): no family to compare, nothing said
        v["route_aliases"] = {"blueberry-muffin": "default"}
        assert "route_family" not in poll("dr-rf7")
        # int down, or a /version without the route: nothing, and the poll still succeeds
        s.routes = {"/version": (503, {"error": "unavailable"})}
        assert "route_family" not in poll("dr-rf8")
        s.routes = {"/version": (200, {"build_id": "t", "route_info": {"default": {}}})}
        assert "route_family" not in poll("dr-rf9")
    finally:
        s.close()


def test_calibrate_then_check_finds_the_drift(bucket):
    b, dep, same, _ = bucket
    r = _run(b, "calibrate", "dr-c1", "--deployed-file", str(dep), "--replicates", "2", "--limit-rows", "60")
    base = b.read_json(f"baselines/{DIGEST}/baseline.json")
    assert r["status"] == "ok" and base["calibrated"] and len(base["replicates"]) == 2
    assert r["notifications"][0]["kind"] == "drift_notice"
    chk = _run(b, "check", "dr-c2", "--deployed-file", str(dep), "--limit-rows", "60", "--retrain-auto", "on")
    assert chk["verdict"] == "retrain" and chk["retrain_eligible"] and "d" in chk["notification"]["details"]["fired"]
    assert chk["notification"]["kind"] == "drift_detected"
    assert chk["notification"]["title"] == "Sous Chef drift: retraining started"
    assert chk["notification"]["dedup_key"].startswith("drift:")
    assert b.read_json("state/locks/gpu-eval.json", None) is None  # released
    mat = b.read_json("drift/dr-c2/ack-material.json")
    assert mat["deployed_hash"] == Manifest(drifted()).hash and mat["dodged_keys"]
    st = b.read_json("state/deployed.json")
    assert st["last_verdict"] == "retrain" and st["pending_check_at"] is None


def test_check_on_the_training_manifest_with_a_right_model_is_ok(bucket):
    b, dep, same, _ = bucket
    _run(b, "calibrate", "dr-o1", "--deployed-file", str(same), "--replicates", "2", "--limit-rows", "40")
    chk = _run(b, "check", "dr-o2", "--deployed-file", str(same), "--limit-rows", "40", "--dry-model", "oracle")
    assert chk["verdict"] == "ok" and chk["notification"] is None and chk["metrics"]["arm_d"]["delta"] == 0.0


def test_without_a_baseline_a_retrain_is_only_recommended(bucket):
    b, dep, same, _ = bucket
    chk = _run(b, "check", "dr-n1", "--deployed-file", str(dep), "--limit-rows", "30", "--retrain-auto", "on")
    assert chk["verdict"] == "retrain" and not chk["retrain_eligible"]
    assert chk["retrain_blocked_by"] == "baseline not calibrated"
    assert chk["notification"]["title"] == "Sous Chef drift: retrain recommended"
    assert chk["notification"]["summary"].endswith("No round started (baseline not calibrated), so no Approve request follows; "
                                                   "start a round by hand for one. Playbook: docs/RETRAIN.md.")
    # auto-retrain off is named next to every other blocker, never hidden behind the first
    chk = _run(b, "check", "dr-n2", "--deployed-file", str(dep), "--limit-rows", "30", "--retrain-auto", "off", "--force")
    assert chk["retrain_blocked_by"] == "baseline not calibrated"
    assert "No round started (baseline not calibrated; auto-retrain is off), so no Approve request follows" in chk["notification"]["summary"]


def test_a_failed_round_or_a_stale_lock_blocks_a_new_round(bucket):
    """No automatic round while a failed one is not resumed (verifier 3), and a stale retrain lock is named as
    the reason instead of announcing a round the workflow will not start (verifier 2)."""
    b, dep, same, _ = bucket
    _run(b, "calibrate", "dr-f1", "--deployed-file", str(dep), "--replicates", "2", "--limit-rows", "60")
    b.write_json("state/failed-round.json", {"run_id": "rt-old", "stage": "dataset"})
    chk = _run(b, "check", "dr-f2", "--deployed-file", str(dep), "--limit-rows", "60", "--retrain-auto", "on", "--force")
    assert chk["verdict"] == "retrain" and not chk["retrain_eligible"] and "rt-old" in chk["retrain_blocked_by"]
    assert chk["notification"]["title"] == "Sous Chef drift: retrain recommended"
    b.delete("state/failed-round.json")
    b.write_json("state/locks/retrain.json", {"holder": "x", "run_id": "rt-stuck", "expires_at": "2020-01-01T00:00:00Z",
                                              "updated_at": "2019-12-29T00:00:00Z"})
    chk = _run(b, "check", "dr-f3", "--deployed-file", str(dep), "--limit-rows", "60", "--retrain-auto", "on", "--force")
    assert not chk["retrain_eligible"] and "stale retrain lock" in chk["retrain_blocked_by"]
    assert chk["notification"]["title"] == "Sous Chef drift: retrain recommended"
    mat = b.read_json("drift/dr-f3/ack-material.json")
    assert len(mat["arm_D_rows"]) == 2 and mat["arm_D_replicates"] == 2 and mat["stale_exam_rows"] == 1


def test_a_manual_rollback_is_judged_as_the_model_it_rolled_back_to(bucket):
    b, dep, same, _ = bucket
    m = b.read_json("state/model.json")
    m.update(tag="c-rt-new", digest="sha256:" + "b" * 64, history=[dict(m)])
    b.write_json("state/model.json", m)
    r = _run(b, "calibrate", "dr-rb1", "--deployed-file", str(dep), "--replicates", "2", "--limit-rows", "20",
             "--model-digest", DIGEST)
    assert r["status"] == "ok" and r["model"]["tag"] == "r4" and b.exists(f"baselines/{DIGEST}/baseline.json")
    assert any(n["kind"] == "drift_notice" and "manual rollback" in n["summary"] for n in r["notifications"])
    assert D.main(["check", "--bucket", str(b.root), "--dry", "--fake-engine", "--run-id", "dr-rb2", "--deployed-file", str(dep),
                   "--limit-rows", "10", "--model-digest", "sha256:" + "c" * 64]) == 2  # unknown to the pipeline


def test_deployed_state_is_written_on_its_generation(bucket):
    """A poll overlapping the end of a check never brings back the pending_check_at the check cleared."""
    b, dep, same, tmp = bucket
    import argparse
    job = D.Job(argparse.Namespace(bucket=str(b.root), dry=True, run_id="dr-g1", mode="poll", workdir=str(tmp / "w")))
    b.write_json("state/deployed.json", {"hash": "h1", "pending_check_at": "2020-01-01T00:00:00Z"})
    calls = []

    def stale_merge(cur):
        if not calls:  # a check finishes between this poll's read and its write
            b.write_json("state/deployed.json", {"hash": "h1", "pending_check_at": None, "last_check_run": "dr-chk"})
        calls.append(dict(cur))
        cur["seen_at"] = "now"
        return cur

    out = D.update_deployed(job, stale_merge)
    assert len(calls) == 2 and out["pending_check_at"] is None and out["last_check_run"] == "dr-chk" and out["seen_at"] == "now"


def test_a_held_gpu_lock_defers_the_check(bucket):
    b, dep, same, _ = bucket
    b.acquire("gpu-eval", "rt-other", 3600)
    chk = _run(b, "check", "dr-l1", "--deployed-file", str(dep), "--limit-rows", "10")
    assert chk["status"] == "deferred" and chk["verdict"] is None and chk["lock"]["run_id"] == "rt-other"


def test_a_missing_model_state_is_a_configuration_error(tmp_path):
    dep = tmp_path / "d.json"
    dep.write_text(json.dumps(REAL), encoding="utf-8")
    rc = D.main(["check", "--bucket", str(tmp_path / "b"), "--dry", "--fake-engine", "--deployed-file", str(dep), "--run-id", "dr-e1"])
    assert rc == PC.EXIT_CONFIG
    doc = gcsio.bucket(tmp_path / "b").read_json("drift/dr-e1/result.json")
    assert doc["status"] == "failed" and "state/model.json" in doc["error"]["message"]


def test_failure_messages_are_scrubbed(monkeypatch):
    monkeypatch.setenv("SOUS_CHEF_KEY", "supersecretkey123")
    doc = PC.failure_doc("drift-check", "dr-1", RuntimeError("boom supersecretkey123 Bearer abcdefghijkl https://x/?token=zzz"))
    msg = doc["error"]["message"]
    assert "supersecretkey123" not in msg and "abcdefghijkl" not in msg and "zzz" not in msg


def test_probe_rows_target_the_new_keys():
    from souschef_lab.exam import vintage_state, MANIFEST_FILE
    home = D.home_context(rows_all())
    nm = Manifest(drifted())
    keys = ["globalPanels.ButtonsSection.buttonShadowColor", "globalPanels.LoyaltyBanner.showLoyalty"]
    rows, cov = D.probe_rows(nm, keys, state=vintage_state(MANIFEST_FILE, home), per_key=4)
    assert all(cov[k] >= 1 for k in keys)
    assert {r["meta"]["probe_key"] for r in rows} == set(keys)
    for r in rows:
        assert r["expected"]["keys"] and r["manifest_hash"] == nm.hash and r["set"] == "probe"


ENGINE_OK = bool(os.environ.get("COOKIE_CUTTER_DIR")) and os.system("node --version >/dev/null 2>&1") == 0


@pytest.mark.engine
@pytest.mark.skipif(not ENGINE_OK and os.environ.get("SOUS_CHEF_LAB_REQUIRE_ENGINE") != "1",
                    reason="needs node + COOKIE_CUTTER_DIR (the real engine)")
def test_drift_dry_real_engine(bucket):
    b, dep, same, _ = bucket
    for mode, run_id, *rest in (["calibrate", "dr-r1", "--replicates", "2"], ["check", "dr-r2"]):
        assert D.main([mode, *rest, "--bucket", str(b.root), "--dry", "--deployed-file", str(dep), "--smoke", "--run-id", run_id,
                       "--no-f0h"]) == 0
    res = b.read_json("drift/dr-r2/result.json")
    assert res["verdict"] == "retrain" and res["metrics"]["arm_d"]["n"] >= 40


# ── pipeline_common: the endpoint client, the metadata token, the warm-up ──

class _Srv:
    """A tiny HTTP server answering from a dict {path prefix: (status, body)}."""

    def __init__(self, routes: dict):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        srv = self
        self.routes, self.seen = routes, []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def _answer(self):
                srv.seen.append((self.command, self.path, dict(self.headers)))
                for prefix, (status, body) in srv.routes.items():
                    if self.path.startswith(prefix):
                        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
                        self.send_response(status)
                        self.send_header("Content-Length", str(len(raw)))
                        self.end_headers()
                        self.wfile.write(raw)
                        return
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_GET = do_POST = _answer

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()


def test_deployed_manifest_client():
    h = Manifest(REAL).hash
    s = _Srv({"/v1/manifests/deployed": (200, {"hash": h, "build_line": "int", "body": REAL, "launcher_build": "1a2b3c4"})})
    try:
        doc = PC.deployed_manifest(s.url, "k", line="int", body=True, days=7)
        assert doc["hash"] == h and s.seen[-1][1] == "/v1/manifests/deployed?line=int&body=1&days=7"
        assert s.seen[-1][2]["Authorization"] == "Bearer k"
        s.routes = {"/v1/manifests/deployed": (404, {"error": {"code": "no_launcher_manifest"}})}
        with pytest.raises(PC.NoLauncherManifest):
            PC.deployed_manifest(s.url, "k")
        s.routes = {"/v1/manifests/deployed": (401, {"error": "unauthorized"})}
        with pytest.raises(PC.ConfigError):
            PC.deployed_manifest(s.url, "k")
        # a transient 503 is retried (one bad answer never fails a poll, nor pages anyone); a persistent one fails
        n0 = len(s.seen)
        s.routes = {"/v1/manifests/deployed": (503, {"error": "unavailable"})}
        with pytest.raises(PC.PipelineError):
            PC.deployed_manifest(s.url, "k", backoff_s=0.01)
        assert len(s.seen) - n0 == 4
        import threading
        threading.Timer(0.05, lambda: s.routes.update({"/v1/manifests/deployed": (200, {"hash": h, "body": REAL})})).start()
        assert PC.deployed_manifest(s.url, "k", backoff_s=0.05)["hash"] == h
    finally:
        s.close()


def test_id_token_and_gpu_warm(monkeypatch):
    s = _Srv({"/computeMetadata/v1/instance/service-accounts/default/identity": (200, b"eyJ.tok.en"),
              "/api/version": (200, {"version": "0.34.4"}), "/api/generate": (200, {"done": True})})
    try:
        monkeypatch.setenv("GCE_METADATA_HOST", s.url[len("http://"):])
        assert PC.id_token("https://m.run.app") == "eyJ.tok.en"
        assert "audience=https%3A%2F%2Fm.run.app" in s.seen[-1][1] and s.seen[-1][2]["Metadata-Flavor"] == "Google"
        w = PC.gpu_warm(s.url, wait_s=5, progress=lambda m: None)
        assert w["ok"] and w["version"] == {"version": "0.34.4"}
        assert any(p == "/api/generate" and hd.get("Authorization") == "Bearer eyJ.tok.en" for _, p, hd in s.seen)
        s.routes = {"/computeMetadata": (200, b"eyJ.tok.en"), "/api/version": (503, {})}
        w = PC.gpu_warm(s.url, wait_s=0.2, every_s=0.05, progress=lambda m: None)
        assert not w["ok"]
    finally:
        s.close()


def test_thresholds_override_and_notification_validation(tmp_path):
    b = gcsio.bucket(tmp_path)
    b.write_json("config/thresholds.json", {"retrain_min_rows": 9, "_why": "test"})
    th = PC.load_thresholds(b)
    assert th["retrain_min_rows"] == 9 and th["notice_sigma"] == 2.0 and "_why" not in th
    with pytest.raises(ValueError):
        PC.notification("nope", title="t", summary="s")
    n = PC.notification("drift_notice", title="t" * 300, summary="s")
    assert len(n["title"]) == 200


@pytest.mark.engine
@pytest.mark.skipif(not ENGINE_OK and os.environ.get("SOUS_CHEF_LAB_REQUIRE_ENGINE") != "1",
                    reason="needs node + COOKIE_CUTTER_DIR (the real engine)")
def test_exam_run_manifest_rebase(tmp_path):
    from souschef_lab import exam as X
    from souschef_lab.engine import find_engine
    dep = tmp_path / "deployed.json"
    dep.write_text(json.dumps(drifted()), encoding="utf-8")
    assert X.main(["run", "--set", "real", "--fake", "--oracle", "--route", "fake", "--manifest-rebase", str(dep),
                   "--rederive-contexts", "--persistent-engine", "--engine", str(find_engine(os.environ.get("ENGINE"))),
                   "--out-dir", str(tmp_path), "--label", "rb"]) == 0
    rep = json.loads(next(tmp_path.glob("exam-real-rb-*.json")).read_text())
    assert rep["manifest_hash"] == Manifest(drifted()).hash and rep["cases"] == 12
    assert rep["summary"]["overall"]["passed"] == 12
