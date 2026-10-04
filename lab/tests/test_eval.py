"""eval.py: the exam runner, its report, --compare and --rescore (fake service, fake engine)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
for _p in (LAB, LAB / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fake_service as FS  # noqa: E402
from souschef_lab import collect as C  # noqa: E402
from souschef_lab import eval as E  # noqa: E402
from test_collect import ASSETS, SET_TITLE, TINY, RecordingVerifier, call, done, fake_engine, make_case, waiting  # noqa: E402


def row(cid="c", k=1, verdict="pass", kind="set_field", section="heroBanner", ftype="text", status="ok", ops=1, ms=100,
        usage=None):
    return {"id": f"{cid}#{k}", "case_id": cid, "sample": k, "status": status, "verdict": verdict if status == "ok" else None,
            "ops": [{"op": "set_fields"}] * ops, "latency_ms": ms, "usage": usage,
            "meta": {"kind": kind, "section": section, "field_type": ftype}}


@pytest.fixture
def svc():
    with FS.FakeService(manifest=TINY, key="k") as s:
        yield s


@pytest.fixture
def client(svc):
    return C.Client(svc.url, "k", timeout_s=10)


def test_percentile_nearest_rank():
    assert E.percentile([], 0.5) is None
    assert E.percentile([5], 0.95) == 5
    assert E.percentile([1, 2, 3, 4], 0.5) == 2
    assert E.percentile([1, 2, 3, 4], 0.95) == 4
    assert E.percentile(range(1, 101), 0.95) == 95


def test_group_figures():
    rows = [row("a"), row("b", verdict="wrong_value", ops=2, ms=300), row("c", status="error", ms=50),
            row("d", verdict="clarify_ok", ops=0)]
    g = E.group(rows)
    assert g["n"] == 4 and g["scored"] == 3 and g["passed"] == 2 and g["pass_rate"] == 66.7 and g["errors"] == 1
    assert g["verdicts"] == {"None": 1, "clarify_ok": 1, "pass": 1, "wrong_value": 1}
    assert g["ops_per_turn"] == {"mean": 1.0, "p50": 1, "max": 2}
    assert g["latency_p50_ms"] == 100 and g["latency_p95_ms"] == 300
    assert E.group([])["pass_rate"] is None


def test_summarize_breaks_down_by_kind_container_and_field_type():
    rows = [row("a"), row("b", kind="toggle", section="HeaderSection", ftype="bool", verdict="missing_writes"),
            row("c", kind="toggle", section="heroBanner", ftype="bool")]
    s = E.summarize(rows)
    assert list(s["by_kind"]) == ["set_field", "toggle"]  # KINDS order
    assert s["by_kind"]["toggle"]["passed"] == 1 and s["by_kind"]["toggle"]["n"] == 2
    assert s["by_container"]["heroBanner"]["n"] == 2 and s["by_container"]["HeaderSection"]["pass_rate"] == 0.0
    assert s["by_field_type"]["bool"]["n"] == 2 and s["by_field_type"]["text"]["pass_rate"] == 100.0


def test_build_report_and_usage_sums():
    u = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3, "cache_creation_input_tokens": 1, "calls": 2}
    rows = [row("a", usage=u), row("a", k=2, usage=u), row("b")]
    r = E.build_report(rows, label="t", route="fake", manifest_hash="h", started="2026-10-01T00:00:00Z", samples=2)
    assert r["cases"] == 2 and r["samples"] == 2 and r["rule_version"] == E.RULE_VERSION and r["manifest_hash"] == "h"
    assert r["usage"] == {"input_tokens": 20, "output_tokens": 10, "cache_read_input_tokens": 6, "cache_creation_input_tokens": 2,
                          "calls": 4}
    assert r["summary"]["overall"]["passed"] == 3 and r["results"] is rows


def test_run_exam_asks_only_holdout_cases_and_saves(svc, client, tmp_path):
    for cid in ("h1", "h2"):
        svc.scripts[cid] = [[call("t1", "set_fields", SET_TITLE), waiting("t1")], done()]
    cases = [make_case("t1"), make_case("h1", holdout=True), make_case("h2", holdout=True, kind="toggle")]
    verifier = RecordingVerifier("pass")
    path, report = E.run_exam(cases, client, fake_engine, label="Teacher run", route="fake", samples=2, verifier=verifier,
                              assets=ASSETS, out_dir=tmp_path / "runs", progress=lambda s: None)
    assert path.parent == tmp_path / "runs" and path.name.startswith("eval-Teacher-run-") and path.suffix == ".json"
    assert sorted(r["siteId"] for r in svc.turn_requests) == ["lab-h1", "lab-h1", "lab-h2", "lab-h2"]
    assert report["cases"] == 2 and report["samples"] == 2 and len(report["results"]) == 4
    assert report["manifest_hash"] == cases[1]["manifest_hash"] and report["route"] == "fake"
    assert report["summary"]["overall"]["pass_rate"] == 100.0 and report["summary"]["by_kind"]["toggle"]["n"] == 2
    assert report["usage"]["calls"] == 4
    saved = E.load_run(path)
    assert saved["label"] == "Teacher run" and len(saved["results"]) == 4
    assert "ops" in saved["results"][0] and "events" in saved["results"][0]  # rescore needs them
    assert E.format_summary(report).splitlines()[1].strip().startswith("overall  4/4 pass (100.0 pts)")


def test_compare_agreement_and_flips():
    a = {"label": "A", "summary": {"overall": {"n": 3}}, "results": [row("x"), row("y", verdict="wrong_value"), row("z")]}
    b = {"label": "B", "summary": {"overall": {"n": 3}}, "results": [row("x"), row("y"), row("w", verdict="pass")]}
    cmp = E.compare(a, b)
    assert cmp["shared"] == 2 and cmp["agreement"] == [1, 2] and cmp["agreement_rate"] == 50.0
    assert cmp["flips"] == {"a_only_pass": 0, "b_only_pass": 1}
    assert cmp["only_a"] == ["z#1"] and cmp["only_b"] == ["w#1"]
    assert cmp["differ"] == [{"id": "y#1", "kind": "set_field", "a": "wrong_value", "b": "pass", "agree": False}]
    text = E.format_compare(cmp)
    assert "1/2 samples agree (50.0%)" in text and "y#1" in text and "in one run only: A 1, B 1" in text


def test_rescore_reverifies_under_new_rules():
    run = {"label": "A", "route": "fake", "manifest_hash": "h", "started_at": "s", "samples": 1, "cases": 2,
           "results": [row("a"), row("b"), row("zzz")]}
    cases = [make_case("a"), make_case("b")]
    out = E.rescore(run, cases, RecordingVerifier("extra_writes", ["extra_writes"]), rule_version="verify-2")
    assert out["label"] == "A-rescored" and out["rule_version"] == "verify-2" and out["cases"] == 2
    verdicts = {r["id"]: r["verdict"] for r in out["results"]}
    assert verdicts == {"a#1": "extra_writes", "b#1": "extra_writes", "zzz#1": None}
    assert next(r for r in out["results"] if r["id"] == "zzz#1")["reasons"] == ["no_case"]
    assert run["results"][0]["verdict"] == "pass"  # the input is not mutated


def test_save_is_atomic_and_slugged(tmp_path):
    p = E.save({"label": "x"}, tmp_path, "my label/odd", stamp="20261001T000000Z")
    assert p.name == "eval-my-label-odd-20261001T000000Z.json" and not list(tmp_path.glob("*.tmp"))
    assert E.slug("") == "run" and E.slug("a b") == "a-b"


def test_main_compare_writes_a_file(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"label": "A", "summary": {"overall": {}}, "results": [row("x")]}), encoding="utf-8")
    b.write_text(json.dumps({"label": "B", "summary": {"overall": {}}, "results": [row("x", verdict="no_answer")]}), encoding="utf-8")
    assert E.main(["--compare", str(a), str(b), "--out-dir", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "0/1 samples agree" in out and list((tmp_path / "out").glob("compare-A-vs-B-*.json"))


def test_main_rescore_uses_the_current_verifier(tmp_path, monkeypatch, capsys):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(make_case("a", holdout=True)) + "\n", encoding="utf-8")
    run = tmp_path / "eval-A-x.json"
    run.write_text(json.dumps({"label": "A", "route": None, "manifest_hash": "h", "started_at": "s", "samples": 1, "cases": 1,
                               "results": [row("a")]}), encoding="utf-8")
    monkeypatch.setattr(C, "default_verifier", lambda *a, **k: RecordingVerifier("wrong_value", ["wrong_value"]))
    assert E.main(["--rescore", str(run), "--cases", str(cases), "--out-dir", str(tmp_path / "out")]) == 0
    saved = next((tmp_path / "out").glob("eval-A-rescored-*.json"))
    assert E.load_run(saved)["results"][0]["verdict"] == "wrong_value"
    assert "A-rescored" in capsys.readouterr().out


def test_main_validation(tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(make_case("a")) + "\n", encoding="utf-8")
    assert E.main(["--cases", str(cases), "--key", ""]) == 2
    assert E.main(["--cases", str(cases), "--key", "k", "--samples", "0"]) == 2
    assert E.main(["--cases", str(cases), "--key", "k"]) == 2  # no held-out cases
    assert "held-out" in capsys.readouterr().err


def test_main_end_to_end_against_the_fake(svc, tmp_path, monkeypatch, capsys):
    svc.scripts["h"] = [done("nothing to do")]
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(make_case("h", holdout=True, kind="clarify")) + "\n", encoding="utf-8")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(TINY), encoding="utf-8")
    monkeypatch.setattr(C, "default_engine", lambda *a, **k: fake_engine)
    monkeypatch.setattr(C, "default_verifier", lambda *a, **k: RecordingVerifier("clarify_ok"))
    rc = E.main(["--cases", str(cases), "--url", svc.url, "--key", "k", "--manifest", str(manifest), "--label", "smoke",
                 "--out-dir", str(tmp_path / "runs"), "--assets", str(tmp_path / "none.json")])
    assert rc == 0
    path = next((tmp_path / "runs").glob("eval-smoke-*.json"))
    report = E.load_run(path)
    assert report["summary"]["overall"]["verdicts"] == {"clarify_ok": 1} and report["summary"]["overall"]["pass_rate"] == 100.0
    assert "clarify" in capsys.readouterr().out


def test_report_carries_the_engine_fingerprint():
    rep = E.build_report([], label="l", route="r", manifest_hash="m", started="2026-10-01T00:00:00Z", engine_fingerprint="fp-1")
    assert rep["engine_fingerprint"] == "fp-1"
    assert E.build_report([], label="l", route="r", manifest_hash="m", started="2026-10-01T00:00:00Z")["engine_fingerprint"] is None
