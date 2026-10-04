"""retrain_check.py: manifest diff thresholds, shape changes, real-kept drift, age, exit codes, Slack text."""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
for _p in (LAB, LAB / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from souschef_lab import retrain_check as R  # noqa: E402

SNAPSHOT = LAB.parent / "service" / "testdata" / "manifest.snapshot.json"
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
BASE = {"version": 1, "generated_from": {"at": "x"},
        "sections": {"heroBanner": {"fields": {"title": {"type": "text"}, "show": {"type": "bool"}},
                                    "lists": {"slides": {"item": {"layout": {"type": "enum"}}, "ops": ["add", "remove"]}}}},
        "globalPanels": {"HeaderSection": {"fields": {"sticky": {"type": "bool"}}}},
        "pages": {"home": {"fields": {}, "lists": {"sections": {"item": {}, "ops": ["add", "remove", "reorder"]}}}},
        "details": {"SeoPanel": {"fields": {"item.title": {"type": "text"}}}},
        "branding": {"fields": {"logo": {"type": "image"}}}}


def with_fields(manifest: dict, n: int, container: str = "heroBanner", prefix: str = "extra") -> dict:
    m = copy.deepcopy(manifest)
    for i in range(n):
        m["sections"][container]["fields"][f"{prefix}{i}"] = {"type": "text"}
    return m


def trace(written=("values.title",), kept=("values.title",), undone=False, ops=("set_fields",), turn="t"):
    events = [{"event": "tool.call", "seq": 1, "data": {"id": "c1", "name": o, "input": {}}} for o in ops]
    if written:
        events.append({"event": "tool.result", "seq": 2, "data": {"toolCallId": "c1", "ok": True, "written": list(written),
                                                                  "for_seq": 1}})
    fb = {"kept_paths": list(kept)} if kept is not None else {}
    if undone:
        fb["undone"] = True
    return {"turn_id": turn, "context": {"page": "home"}, "events": events, "ops": [{"op": o} for o in ops], "feedback": fb}


def exam(rate: float) -> dict:
    return {"label": "exam", "summary": {"overall": {"pass_rate": rate, "n": 10}}}


def dataset_manifest(days_ago: float, hsh: str = "a" * 64) -> dict:
    return {"manifest_hash": hsh, "registered_at": (NOW - timedelta(days=days_ago)).isoformat()}


# --- manifest ------------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not SNAPSHOT.exists(), reason="snapshot not checked out")
def test_field_index_counts_the_snapshot():
    m = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    idx = R.field_index(m)
    assert len(idx) == 2094
    assert idx["sections.heroBanner.paddingY"] == "range"
    assert "sections.heroBanner.slides[].layout" in idx
    assert {"add", "remove"} <= R.op_names(m)


def test_field_index_small_manifest():
    idx = R.field_index(BASE)
    assert idx == {"sections.heroBanner.title": "text", "sections.heroBanner.show": "bool",
                   "sections.heroBanner.slides[].layout": "enum", "globalPanels.HeaderSection.sticky": "bool",
                   "details.SeoPanel.item.title": "text", "branding.logo": "image"}
    assert R.op_names(BASE) == {"add", "remove", "reorder"}


def test_manifest_diff_counts_added_removed_retyped():
    new = with_fields(BASE, 3)
    del new["sections"]["heroBanner"]["fields"]["show"]
    new["branding"]["fields"]["logo"]["type"] = "text"
    d = R.manifest_diff(BASE, new)
    assert len(d["added"]) == 3 and d["removed"] == ["sections.heroBanner.show"] and d["retyped"] == ["branding.logo"]
    assert d["changed"] == 5 and d["fields"] == {"old": 6, "new": 8}
    assert d["new_sections"] == [] and d["new_panels"] == [] and d["new_ops"] == []


def test_manifest_threshold_nine_is_quiet_ten_triggers():
    nine = R.check(current=with_fields(BASE, 9), training=BASE, dataset_manifest=dataset_manifest(1), exam_run=exam(90),
                   traces=[trace()] * 60, now=NOW)
    assert nine["exit_code"] == 0 and nine["triggers"] == []
    assert any("9 fields changed" in n for n in nine["notes"])
    ten = R.check(current=with_fields(BASE, 10), training=BASE, dataset_manifest=dataset_manifest(1), exam_run=exam(90),
                  traces=[trace()] * 60, now=NOW)
    assert ten["exit_code"] == 1 and len(ten["triggers"]) == 1
    assert ten["triggers"][0].startswith("manifest: 10 fields changed since training (+10 -0 retyped 0; threshold 10)")


def test_shape_changes_trigger_even_under_the_field_threshold():
    new = copy.deepcopy(BASE)
    new["sections"]["promoBar"] = {"fields": {"text": {"type": "text"}}}
    new["globalPanels"]["WishlistSection"] = {"fields": {}}
    new["details"]["GiftPanel"] = {"fields": {}}
    new["pages"]["home"]["lists"]["sections"]["ops"].append("duplicate")
    d = R.manifest_diff(BASE, new)
    assert d["new_sections"] == ["promoBar"] and d["new_panels"] == ["GiftPanel", "WishlistSection"] and d["new_ops"] == ["duplicate"]
    res = R.check(current=new, training=BASE, dataset_manifest=dataset_manifest(1), exam_run=exam(90), traces=[trace()] * 60, now=NOW)
    assert res["exit_code"] == 1
    assert res["triggers"] == ["shape: new section types promoBar; panels GiftPanel, WishlistSection; list ops duplicate"]


def test_new_op_names_in_traces_trigger():
    traces = [trace()] * 60 + [trace(ops=("apply_preset",))]
    res = R.check(current=BASE, training=BASE, dataset_manifest=dataset_manifest(1), exam_run=exam(90), traces=traces, now=NOW)
    assert res["triggers"] == ["shape: traces carry op names the training set never saw: apply_preset"]


# --- traces ---------------------------------------------------------------------------------------------------------------------

def test_real_kept_rate():
    traces = [trace(),  # kept
              trace(written=("a", "b"), kept=("a",)),  # partial cover → not kept
              trace(undone=True),  # undone → not kept
              trace(written=(), kept=()),  # no writes → not scored
              trace(kept=None),  # no feedback → not kept
              {"turn_id": "x", "events": [], "written_paths": ["p"], "feedback": {"kept_paths": ["p", "q"]}}]  # collector row shape
    rk = R.real_kept(traces)
    assert rk == {"turns": 6, "with_writes": 5, "kept": 2, "rate": 40.0}
    assert R.real_kept([])["rate"] is None
    assert R.trace_ops(trace(ops=("set_fields", "edit_list"))) == {"set_fields", "edit_list"}


def test_drift_needs_fifty_turns_else_insufficient():
    res = R.check(current=BASE, training=BASE, dataset_manifest=dataset_manifest(1), exam_run=exam(90), traces=[trace()] * 49, now=NOW)
    assert res["exit_code"] == 2 and res["triggers"] == []
    assert res["insufficient"] == ["drift: only 49 real turns with writes (need 50)"]
    assert "lack data" in res["text"] and "(insufficient data) drift" in res["text"]


def test_drift_trigger_is_strictly_more_than_ten_points():
    kept = [trace()] * 40 + [trace(undone=True)] * 10  # 80 pts
    at_ten = R.check(current=BASE, training=BASE, dataset_manifest=dataset_manifest(1), exam_run=exam(90.0), traces=kept, now=NOW)
    assert at_ten["exit_code"] == 0 and at_ten["real_kept"]["rate"] == 80.0
    over = R.check(current=BASE, training=BASE, dataset_manifest=dataset_manifest(1), exam_run=exam(90.5), traces=kept, now=NOW)
    assert over["exit_code"] == 1
    assert over["triggers"][0].startswith("drift: real-kept pass 80.0 pts vs exam 90.5 pts (10.5 under; threshold 10; 40/50 turns kept)")


# --- age and exit codes -------------------------------------------------------------------------------------------------------

def test_age_threshold_thirty_days():
    ok = R.check(current=BASE, training=BASE, dataset_manifest=dataset_manifest(29.5), exam_run=exam(90), traces=[trace()] * 60, now=NOW)
    assert ok["exit_code"] == 0 and ok["age_days"] == 29.5
    old = R.check(current=BASE, training=BASE, dataset_manifest=dataset_manifest(30), exam_run=exam(90), traces=[trace()] * 60, now=NOW)
    assert old["exit_code"] == 1 and old["triggers"] == ["age: training manifest registered 30 days ago (≥ 30)"]


def test_registration_time_fallbacks_and_hash():
    assert R.training_registered_at({"first_seen": "2026-09-01T00:00:00Z"}) == datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert R.training_registered_at({"at": "2026-09-01T00:00:00"}) == datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert R.training_registered_at({"registered_at": "garbage"}) is None
    assert R.training_hash({"manifestHash": "h"}) == "h" and R.training_hash({}) is None
    res = R.check(current=BASE, training=BASE, dataset_manifest={"manifest_hash": "h"}, exam_run=exam(90), traces=[trace()] * 60, now=NOW)
    assert res["exit_code"] == 2 and res["insufficient"] == ["age: dataset-manifest.json has no registration time"]


def test_clean_run_exits_zero_with_ok_notes():
    res = R.check(current=BASE, training=BASE, dataset_manifest=dataset_manifest(3), exam_run=exam(85), traces=[trace()] * 60, now=NOW)
    assert res["exit_code"] == 0 and res["triggers"] == [] and res["insufficient"] == []
    text = res["text"].splitlines()
    assert text[0] == "Sous Chef in-house model: no retrain needed"
    assert all(line.startswith("• ok — ") for line in text[1:]) and len(text) == 4


def test_everything_missing_is_insufficient():
    res = R.check(current=None, training=None, dataset_manifest=None, exam_run=None, traces=None, now=NOW)
    assert res["exit_code"] == 2 and len(res["insufficient"]) == 3


def test_several_triggers_stack_in_slack_text():
    new = with_fields(BASE, 12)
    res = R.check(current=new, training=BASE, dataset_manifest=dataset_manifest(45), exam_run=exam(95),
                  traces=[trace(undone=True)] * 60, now=NOW)
    assert res["exit_code"] == 1 and len(res["triggers"]) == 3
    lines = res["text"].splitlines()
    assert lines[0] == "Sous Chef in-house model: RETRAIN recommended"
    assert [l.split(":")[0] for l in lines[1:4]] == ["• manifest", "• drift", "• age"]


def test_main_with_files_and_json(tmp_path, capsys):
    cur = tmp_path / "cur.json"
    cur.write_text(json.dumps(with_fields(BASE, 10)), encoding="utf-8")
    tr = tmp_path / "train.json"
    tr.write_text(json.dumps(BASE), encoding="utf-8")
    dm = tmp_path / "dataset-manifest.json"
    dm.write_text(json.dumps(dataset_manifest(2)), encoding="utf-8")
    ex = tmp_path / "eval.json"
    ex.write_text(json.dumps(exam(90)), encoding="utf-8")
    traces = tmp_path / "traces.jsonl"
    traces.write_text("\n".join(json.dumps(trace()) for _ in range(60)) + "\n", encoding="utf-8")
    args = ["--manifest", str(cur), "--training-manifest", str(tr), "--dataset-manifest", str(dm), "--exam-run", str(ex),
            "--traces-jsonl", str(traces), "--now", NOW.isoformat()]
    assert R.main(args) == 1
    assert capsys.readouterr().out.splitlines()[0] == "Sous Chef in-house model: RETRAIN recommended"
    assert R.main(args + ["--json"]) == 1
    doc = json.loads(capsys.readouterr().out)
    assert doc["exit_code"] == 1 and doc["manifest_diff"]["changed"] == 10 and doc["real_kept"]["kept"] == 60
    tr.write_text(json.dumps(with_fields(BASE, 10)), encoding="utf-8")  # same manifest → clean
    assert R.main(args) == 0
    assert R.main(["--manifest", str(tmp_path / "missing.json")]) == 2


# --- engine trigger (LAB.md §7) --------------------------------------------------------------------------------------

def test_engine_fingerprint_trigger():
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    dm = dict(dataset_manifest(1.0), engine_fingerprint="fp-then")
    now = R.parse_time("2026-10-02T00:00:00Z")
    base = dict(current=snap, training=snap, dataset_manifest=dm, exam_run=None, traces=None, now=now)
    off = R.check(**base)
    assert "engine_fingerprint" not in off and not any(t.startswith("engine") for t in off["triggers"])
    same = R.check(**base, engine_fingerprint="fp-then")
    assert not any(t.startswith("engine") for t in same["triggers"]) and any(n.startswith("engine: unchanged") for n in same["notes"])
    moved = R.check(**base, engine_fingerprint="fp-now")
    assert moved["exit_code"] == 1 and [t for t in moved["triggers"] if t.startswith("engine")] and "cases remeasure" in moved["text"]
    assert moved["engine_fingerprint"] == {"current": "fp-now", "training": "fp-then"}
    unknown = R.check(**dict(base, dataset_manifest=dataset_manifest(1.0)), engine_fingerprint="fp-now")
    assert "engine: dataset-manifest.json carries no engine_fingerprint" in unknown["insufficient"]
