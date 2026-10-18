"""retrain/eval_stage.py: the paired eval (design §3.8) — the sets on the deployed
manifest, the DRY arms, and a stale current model caught on a renamed key."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from souschef_lab import FIXTURES_DIR
from souschef_lab.manifest import Manifest
from souschef_lab.retrain import eval_stage as EV
from souschef_lab.retrain.common import open_store

from tests.test_rebase import _need_engine, drifted

RUN = "rt-20261012-0310-ccd1afab"
TH = {"smoke_rows_per_set": 1000, "probe_cases_per_key": 4, "probe_max_keys": 40}


def seeded(tmp_path, dep_doc: dict):
    s = open_store(tmp_path / "bucket")
    exam = json.loads((FIXTURES_DIR / "exam" / "manifest-real.json").read_text())
    dm = Manifest(dep_doc)
    s.write_json(f"runs/{RUN}/prepare/manifest.json", dep_doc)
    s.write_jsonl(f"runs/{RUN}/prepare/cases.jsonl", [])
    s.write_json(f"runs/{RUN}/run.json", {"run_id": RUN, "deployed": {"hash": dm.hash}, "trigger": {}})
    s.write_json("state/model.json", {"tag": "r4", "digest": "sha256:x", "training_manifest": Manifest(exam).hash})
    s.write_json(f"manifests/{Manifest(exam).hash}.json", exam)
    return s


def test_plan_sets_rebases_the_exam_and_keeps_the_recorded_rows(tmp_path):
    s = seeded(tmp_path, drifted(rename=True, drop_input=True))
    sets = EV.plan_sets(s, RUN, mode="full", local=tmp_path / "w", th=TH, progress=lambda m: None)
    assert {r["from"].split(".")[-1]: r["to"].split(".")[-1] for r in sets["renames"]}["borderRadius"] == "cornerRadius"
    rows = [r for s_ in EV.EXAM_SETS for r in sets["sets"][s_]]
    keys = {k["key"] for r in rows for k in (r.get("expected") or {}).get("keys") or []}
    assert "cornerRadius" in keys and "borderRadius" not in keys
    assert all(r["manifest_hash"] == sets["dep_m"].hash for r in rows)
    assert all((r.get("meta") or {}).get("context_source") == "rederived" for r in rows)
    rec = [r for s_ in EV.EXAM_SETS for r in sets["recorded"][s_]]
    assert len(rec) == len(rows) and sets["stale_round"] is False and sets["probe"] == []
    # p137.k2 (create_page) is scored in the report-only capability set, under its own set's key in the baseline
    assert [r["id"] for r in sets["capability"]] == ["p137.k2"] and sets["capability"][0]["_origin_set"] == "out_of_scope"
    assert sum(len(v) for v in sets["sets"].values()) + len(sets["capability"]) == 252 - len(sets["stale"])


def test_a_newer_deployed_manifest_with_a_hard_change_marks_a_stale_round(tmp_path):
    s = seeded(tmp_path, drifted(rename=False))
    newer = drifted(rename=False, drop_input=True)
    nm = Manifest(newer)
    s.write_json(f"manifests/{nm.hash}.json", newer)
    s.write_json("state/deployed.json", {"hash": nm.hash})
    sets = EV.plan_sets(s, RUN, mode="smoke", local=tmp_path / "w", th={"smoke_rows_per_set": 3}, progress=lambda m: None)
    assert sets["stale_round"] is True and sets["newest"]["hash"] == nm.hash
    assert all(r["set"] == "newest" for r in sets["newest"]["rows"])
    assert all(len(v) <= 3 for v in sets["sets"].values())


def test_the_dry_arms_catch_a_stale_model_on_a_renamed_key(tmp_path):
    _need_engine()
    from souschef_lab.exam import load_set
    s = seeded(tmp_path, drifted(rename=True))
    sets = EV.plan_sets(s, RUN, mode="full", local=tmp_path / "w", th=TH, progress=lambda m: None)
    rows = [r for r in sets["sets"]["paraphrase"] if any(k["key"] == "cornerRadius" for k in r["expected"]["keys"])][:2]
    rec = {r["id"]: r for r in load_set("paraphrase")}
    assert rows
    arms = EV.Arms(dry=True, workers=1, progress=lambda m: None)
    cand = arms.exam("candidate", "paraphrase", rows, sets["dep_path"], sets["dep_m"], sample=1, prefix="lab-tc-")
    cur = arms.exam("current", "paraphrase", rows, sets["dep_path"], sets["dep_m"], sample=1, prefix="lab-tu-",
                    script_rows=[rec[r["id"]] for r in rows], script_m=sets["exam_m"])
    assert all(r["pass"] == 1 for r in cand), cand
    assert all(r["pass"] == 0 for r in cur) and sum(r["ghost"] for r in cur) >= 1, cur
    assert {r["key"] for r in cand} == {f"paraphrase:{r['id']}" for r in rows}
    assert EV.results(cand) == {f"paraphrase:{r['id']}": 1 for r in rows}


def test_the_real_run_needs_both_gpu_urls(tmp_path):
    s = seeded(tmp_path, drifted(rename=False))
    from souschef_lab.retrain.common import ConfigError
    with pytest.raises(ConfigError, match="candidate-url"):
        EV.run(s, RUN, mode="smoke", current_url="https://eval", candidate_url=None, workdir=tmp_path / "w",
               progress=lambda m: None)


def test_the_harness_gets_both_routes(tmp_path, monkeypatch):
    """The real path builds builder B's Harness with current + candidate (a stand-in records it)."""
    s = seeded(tmp_path, drifted(rename=False))
    got = {}

    class FakeHarness:
        def __init__(self, routes, **kw):
            got["routes"], got["kw"] = routes, kw

        def __enter__(self):
            raise RuntimeError("stop here")

        def __exit__(self, *a):
            pass
    with pytest.raises(RuntimeError, match="stop here"):
        EV.run(s, RUN, mode="smoke", current_url="https://eval.run.app", candidate_url="https://cand.run.app",
               workdir=tmp_path / "w", harness_factory=FakeHarness, progress=lambda m: None)
    assert got["routes"] == {"current": "https://eval.run.app", "candidate": "https://cand.run.app"}


def test_smoke_tolerates_the_candidates_error_rows_only():
    # a SMOKE candidate is a few updates from the base: its error rows are recorded, not fatal; the current arm's
    # errors (harness or GPU trouble) still fail the stage
    rows = [{"status": "ok"}] * 5 + [{"status": "error", "code": "timeout"}] * 5
    seen = []
    arms = EV.Arms(dry=True, progress=seen.append, tolerate_errors=frozenset({"candidate"}))
    arms.check_errors("candidate", rows, "candidate/paraphrase#1")
    assert arms.tolerated and "5/10 rows errored" in arms.tolerated[0]["error"] and seen
    with pytest.raises(EV.StageError) as e:
        arms.check_errors("current", rows, "current/paraphrase#1")
    assert e.value.reason == "eval_errors"
    with pytest.raises(EV.StageError):
        EV.Arms(dry=True, progress=seen.append).check_errors("candidate", rows, "candidate/paraphrase#1")


# --- r7: the recorded r4 reference (load_reference → bar.add_reference) and the clear-asks count -------------------------

R4_DIGEST = "sha256:f657e11053e3baf801b489421637b8a0aaccf450951046ec4c9853ca0c7287b4"
R4_TH = {"r4_reference": {"tag": "r4", "digest": R4_DIGEST,
                          "baseline": ["baselines/reference/r4/baseline.json", f"baselines/{R4_DIGEST}/baseline.json",
                                       f"experiments/mistral/snapshots/r4/baselines/{R4_DIGEST}/baseline.json"],
                          "rows_run": "rt-20261016-2102-3eeb97-eb908674",
                          "sets": ["paraphrase", "holdout", "real", "followup"], "bar": True}}


def _r4_doc(rows: dict[str, int], digest: str = R4_DIGEST, k: int = 3) -> dict:
    return {"digest": digest, "tag": "r4", "replicates": [{"at": "2026-10-14T02:43:34Z", "run": f"cal#T{i}", "rows": dict(rows)}
                                                          for i in range(k)]}


def test_load_reference_finds_the_first_copy_and_hashes_every_one(tmp_path):
    import hashlib
    s = open_store(tmp_path / "bucket")
    assert EV.load_reference(s, {})[2]["configured"] is False
    assert EV.load_reference(s, {"r4_reference": None})[2]["configured"] is False
    ref, doc, info = EV.load_reference(s, R4_TH)
    assert ref is None and doc is None and info["configured"] and "no r4 baseline at baselines/reference/r4" in info["why"]
    assert info["warnings"]
    rows = {"holdout:h001": 1, "holdout:h022": 0, "paraphrase:p001": 1, "out_of_scope:p137": 1}
    s.write_json(R4_TH["r4_reference"]["baseline"][2], _r4_doc(rows))  # only the Mistral snapshot copy
    ref, doc, info = EV.load_reference(s, R4_TH)
    assert info["source"] == R4_TH["r4_reference"]["baseline"][2] and info["copies_match"] and not info["warnings"]
    assert info["sha256"] == hashlib.sha256(s.read_bytes(info["source"])).hexdigest()
    assert ref["holdout"] == [{"holdout:h001": 1, "holdout:h022": 0}] * 3 and ref["real"] == [] and "out_of_scope" not in ref
    assert doc["digest"] == R4_DIGEST
    s.write_json(R4_TH["r4_reference"]["baseline"][1], _r4_doc(rows))  # the live baselines/ copy, identical
    ref, doc, info = EV.load_reference(s, R4_TH)
    assert info["source"] == R4_TH["r4_reference"]["baseline"][1] and len(info["copies"]) == 2 and info["copies_match"]
    s.write_json(R4_TH["r4_reference"]["baseline"][0], _r4_doc({**rows, "holdout:h001": 0}))  # a differing copy
    ref, doc, info = EV.load_reference(s, R4_TH)
    assert info["source"] == R4_TH["r4_reference"]["baseline"][0] and not info["copies_match"]
    assert ref["holdout"][0]["holdout:h001"] == 0 and "differ" in info["warnings"][0]
    s.write_json(R4_TH["r4_reference"]["baseline"][0], _r4_doc(rows, digest="sha256:7fb1aeba"))  # r6's baseline: refused
    ref, doc, info = EV.load_reference(s, R4_TH)
    assert ref is None and "sha256:7fb1aeba" in info["why"]
    # r7 review: `file_sha256` pins the FILE — a copy with other bytes (a recalibration at the same path) is skipped
    s.write_json(R4_TH["r4_reference"]["baseline"][0], _r4_doc({**rows, "holdout:h001": 0}))
    pinned = hashlib.sha256(s.read_bytes(R4_TH["r4_reference"]["baseline"][1])).hexdigest()
    th = {"r4_reference": {**R4_TH["r4_reference"], "file_sha256": pinned}}
    ref, doc, info = EV.load_reference(s, th)
    assert info["source"] == R4_TH["r4_reference"]["baseline"][1] and info["sha256"] == pinned
    assert ref["holdout"][0]["holdout:h001"] == 1 and any("not the pinned file" in w for w in info["warnings"])
    ref, doc, info = EV.load_reference(s, {"r4_reference": {**R4_TH["r4_reference"], "file_sha256": "0" * 64}})
    assert ref is None and "no copy of the pinned r4 baseline" in info["why"]


def test_rerun_replicates_split_a_rounds_rows_by_occurrence():
    rows = [{"key": "holdout:h001", "set": "holdout", "status": "ok", "pass": 1},
            {"key": "holdout:h002", "set": "holdout", "status": "ok", "pass": 0},
            {"key": "holdout:h001", "set": "holdout", "status": "ok", "pass": 0},
            {"key": "holdout:h002", "set": "holdout", "status": "error", "pass": 0},
            {"key": "holdout:h002", "set": "holdout:recorded", "status": "ok", "pass": 1},
            {"key": "paraphrase:p001", "set": "paraphrase", "status": "ok", "pass": 1}]
    rr = EV.rerun_replicates(rows, ["holdout", "paraphrase", "real"])
    assert rr == {"holdout": [{"holdout:h001": 1, "holdout:h002": 0}, {"holdout:h001": 0}],
                  "paraphrase": [{"paraphrase:p001": 1}], "real": []}


def test_the_eval_adds_the_r4_floor_and_the_report_carries_it(tmp_path, monkeypatch):
    """EV.run end to end with stubbed arms (no engine): the candidate asks on every holdout `relative` row; r4's
    recorded replicates passed them. bar.json gets the :vs-r4 rows and a failing r4_floor, report.json the
    reference (source, sha256, amended rows left out), the clear-asks count and no warning."""
    import hashlib
    from souschef_lab import pipeline_common as PC
    from souschef_lab.exam import MANIFEST_FILE, load_set
    from souschef_lab.retrain import report as REP
    exam_doc = json.loads((FIXTURES_DIR / "exam" / "manifest-real.json").read_text())
    s = seeded(tmp_path, exam_doc)
    built = {n: load_set(n, amended=False) for n in EV.EXAM_SETS}
    rows = {n: [r for r in load_set(n) if not EV.is_capability_row(r)] for n in EV.EXAM_SETS}
    r4_rows = {f"{n}:{r['id']}": 1 for n in ("paraphrase", "holdout", "real", "followup") for r in built[n]}
    s.write_json(R4_TH["r4_reference"]["baseline"][1], _r4_doc(r4_rows))
    relative = {r["id"] for r in rows["holdout"] if r["meta"].get("class") == "relative"}
    assert len(relative) >= 8
    rel0 = sorted(relative)[0]
    s.write_jsonl(f"runs/{R4_TH['r4_reference']['rows_run']}/eval/current/rows.jsonl",
                  [{"key": f"holdout:{rel0}", "id": rel0, "set": "holdout", "status": "ok", "verdict": "no_write",
                    "reasons": ["asked_or_answered"], "pass": 0},
                   {"key": "holdout:h022", "id": "h022", "set": "holdout", "status": "ok", "verdict": "pass", "reasons": [],
                    "pass": 1}])
    monkeypatch.setattr(PC, "load_thresholds", lambda b=None, path=None: {**TH, "prior_flip_rate": 0.08,
                                                                          "min_sigma_rows": 1.0, **R4_TH})
    dep_path = tmp_path / "dep.json"
    dep_path.write_text(json.dumps(exam_doc))

    def plan(store, run_id, **kw):
        return {"dep_path": dep_path, "dep_m": Manifest(exam_doc), "exam_m": Manifest(exam_doc), "exam_path": Path(MANIFEST_FILE),
                "sets": rows, "recorded": rows, "stale": [], "renames": [], "font_policy": {}, "capability": [],
                "capability_built": [], "f0h": [], "f9h": [], "probe": [], "trigger_rules": [], "dodged_keys": [],
                "stale_round": False}

    def exam(self, arm, set_name, rs, mpath, m, *, sample, prefix, script_rows=None, script_m=None, home=None):
        out = []
        for r in rs:
            ask = arm == "candidate" and set_name == "holdout" and r["id"] in relative
            out.append({"key": EV.row_key(set_name, r["id"]), "id": r["id"], "set": set_name, "status": "ok",
                        "verdict": "no_write" if ask else "pass", "reasons": ["asked_or_answered"] if ask else [],
                        "pass": 0 if ask else 1, "ghost": 0, "where_ok": int(set_name.startswith("out_of_scope"))})
        return out
    monkeypatch.setattr(EV, "plan_sets", plan)
    monkeypatch.setattr(EV.Arms, "exam", exam)
    monkeypatch.setattr(REP, "askrule_vague", lambda m=None: (lambda row: bool((row.get("expected") or {}).get("ask_ok"))))
    EV.run(s, RUN, mode="full", current_url=None, candidate_url=None, workdir=tmp_path / "w", dry=True,
           progress=lambda m: None)
    bar = s.read_json(f"runs/{RUN}/eval/bar.json")
    by = {r["set"]: r for r in bar["sets"]}
    ho = by["holdout:vs-r4"]
    assert ho["variant"] == "reference" and not ho["bar"] and ho["candidate"] == ho["current"] - len(relative)
    assert {"paraphrase:vs-r4", "real:vs-r4", "followup:vs-r4"} <= set(by)
    floor = next(r for r in bar["rules"] if r["name"] == "r4_floor")
    assert not floor["pass"] and not bar["pass"] and "holdout" in floor["detail"]
    # the live bar (current passes the same rows) fails the same way; the reference adds its own rule
    rep = s.read_json(f"runs/{RUN}/report.json")
    ref = rep["results"]["r4_reference"]
    assert ref["measured"] and ref["source"] == R4_TH["r4_reference"]["baseline"][1]
    assert ref["sha256"] == hashlib.sha256(s.read_bytes(ref["source"])).hexdigest()
    assert ref["excluded_amended"] >= 4 and ho["excluded"]["amended"] >= 3  # h022.k3, h006.k3 and the .k2 rows
    assert ref["pooled"]["b"] >= 2 * len(relative) and rep["warnings"] == []
    assert "r4_floor: below holdout" in rep["summary"] and "holdout:vs-r4" in rep["summary"]
    # r7 review: r4 RE-RUN in rows_run's round is a second, report-only comparison (rerun_bar false)
    rr = by["holdout:vs-r4_rerun"]
    assert rr["variant"] == "reference" and rr["n"] == 1 and rr["m"] == [1, 2]  # rel0 only: h022 is h022.k3 now
    rerun = next(r for r in bar["rules"] if r["name"] == "r4_rerun_floor")
    assert rerun["pass"] and rerun["detail"].startswith("REPORT ONLY")
    assert rep["results"]["r4_rerun_reference"]["label"] == f"r4 re-run in {R4_TH['r4_reference']['rows_run']}"
    assert "r4 re-run in" in rep["summary"] or "r4_rerun_floor" not in rep["summary"]
    ca = rep["results"]["clear_asks"]["arms"]
    assert ca["candidate"]["by_class"]["relative"]["asked"] == 2 * len([i for i in relative])
    assert ca["current"]["by_class"]["relative"]["asked"] == 0
    assert ca["r4 (recorded)"]["by_class"]["relative"] == {"asked": 1, "n": 1}
    assert "rows" not in ca["candidate"]["by_class"]["all"]  # the per-row keys stay in report.json `eval`
    assert rel0 in " ".join(rep["eval"]["clear_asks"]["arms"]["candidate"]["by_class"]["all"]["rows"])
    assert rep["results"]["candidate"]["id"] == f"c-{RUN}"
    # no reference in the bucket: NOT MEASURED, the floor passes, a warning
    s.delete(R4_TH["r4_reference"]["baseline"][1])
    EV.run(s, RUN, mode="full", current_url=None, candidate_url=None, workdir=tmp_path / "w2", dry=True,
           progress=lambda m: None)
    bar = s.read_json(f"runs/{RUN}/eval/bar.json")
    floor = next(r for r in bar["rules"] if r["name"] == "r4_floor")
    assert floor["pass"] and floor["detail"].startswith("NOT MEASURED")
    assert not any(r["set"].endswith(":vs-r4") for r in bar["sets"])
    rep = s.read_json(f"runs/{RUN}/report.json")
    assert rep["results"]["r4_reference"]["measured"] is False and any("r4_floor" in w for w in rep["warnings"])
