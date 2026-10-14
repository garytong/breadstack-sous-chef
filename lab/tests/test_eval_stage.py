"""retrain/eval_stage.py: the paired eval (design §3.8) — the sets on the deployed
manifest, the DRY arms, and a stale current model caught on a renamed key."""
from __future__ import annotations

import json

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
    assert sum(len(v) for v in sets["sets"].values()) == 252 - len(sets["stale"])


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
