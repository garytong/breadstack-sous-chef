"""Integration seams fixed 2026-10-01: stratified `cases.sample`, parallel
`cases.measure`, the engine's absolute manifest path, and the fake
service's per-sample script Rotation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
for _p in (LAB, LAB / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fake_service as FS  # noqa: E402
from souschef_lab import cases as CS  # noqa: E402
from souschef_lab import engine as ENG  # noqa: E402


def mk(cid: str, kind: str, ops=None) -> CS.Case:
    return CS.Case(id=cid, kind=kind, target=None, fields=[], site_state={}, context={}, request="r",
                   expected={"ops": ops if ops is not None else [{"op": "set_fields", "target": {}, "values": {"a": 1}}]},
                   holdout=False, manifest_hash="h", template="editorial")


def population() -> list[CS.Case]:
    out = []
    for kind, n in (("set_field", 60), ("toggle", 20), ("reset", 15), ("clarify", 4), ("unsupported", 1)):
        out += [mk(f"{kind}/{i}", kind, [] if kind in ("clarify", "unsupported") else None) for i in range(n)]
    return out


# --- sample ---------------------------------------------------------------------------------------------------------------

def test_sample_keeps_every_kinds_share_and_order():
    pop = population()
    got = CS.sample(pop, 20, seed=0)
    assert len(got) == 20
    by_kind = {}
    for c in got:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
    # 60/20/15 % of 20 = 12/4/3; clarify and unsupported round to 0 but keep one each, so the largest kind yields one.
    assert by_kind["set_field"] == 11 and by_kind["toggle"] == 4 and by_kind["reset"] == 3
    assert by_kind["clarify"] == 1 and by_kind["unsupported"] == 1
    ids = [c.id for c in got]
    assert ids == [c.id for c in pop if c.id in set(ids)]  # original order


def test_sample_is_deterministic_and_total_when_n_is_large():
    pop = population()
    assert [c.id for c in CS.sample(pop, 30, seed=3)] == [c.id for c in CS.sample(pop, 30, seed=3)]
    assert [c.id for c in CS.sample(pop, 30, seed=3)] != [c.id for c in CS.sample(pop, 30, seed=4)]
    assert CS.sample(pop, 1000) == pop and CS.sample(pop, 0) == pop


# --- measure with workers --------------------------------------------------------------------------------------------------

class CountingEngine:
    """An EngineResult-shaped fake: accepts every op, writes one leaf per op; records thread fan-out."""

    def __init__(self):
        import threading
        self.calls = 0
        self.lock = threading.Lock()
        self.threads = set()

    def run(self, state, ops, ctx):
        import threading
        with self.lock:
            self.calls += 1
            self.threads.add(threading.get_ident())
        diff = [{"path": f"x[{i}]", "before": None, "after": 1} for i, _ in enumerate(ops)]
        return ENG.EngineResult(ok=True, results=[{"op": o["op"], "ok": True, "warnings": [], "written": ["x"]} for o in ops],
                                changes=[{"slice": "x", "written": ["x"], "diff": diff}], elapsed_s=0.01)


@pytest.mark.parametrize("workers", [1, 3])
def test_measure_preserves_order_and_skips_clarify_unsupported(workers):
    pop = population()[:70] + [mk("clarify/x", "clarify", []), mk("unsupported/x", "unsupported", [])]
    eng = CountingEngine()
    logged = []
    accepted, rejects = CS.measure(pop, eng, log=lambda cid, ok, info: logged.append((cid, ok)), workers=workers)
    assert rejects == [] and [c.id for c in accepted] == [c.id for c in pop]
    assert eng.calls == 70 and len(logged) == 70
    assert all(c.expected.get("diff") for c in accepted if c.kind not in ("clarify", "unsupported"))
    assert not any(c.expected.get("diff") for c in accepted if c.kind in ("clarify", "unsupported"))
    if workers > 1:
        assert len(eng.threads) > 1


def test_measure_rejects_refused_ops_in_parallel_too():
    class Refusing(CountingEngine):
        def run(self, state, ops, ctx):
            r = super().run(state, ops, ctx)
            if "bad" in ops[0].get("values", {}):
                return ENG.EngineResult(ok=False, results=[{"op": "set_fields", "ok": False, "error": {"code": "E_FIELD"}}], changes=[], returncode=2)
            return r

    pop = [mk("a", "set_field"), mk("b", "set_field", [{"op": "set_fields", "values": {"bad": 1}}]), mk("c", "set_field")]
    accepted, rejects = CS.measure(pop, Refusing(), workers=2)
    assert [c.id for c in accepted] == ["a", "c"]
    assert len(rejects) == 1 and rejects[0]["id"] == "b" and rejects[0]["reason"] == "op_refused"


# --- engine manifest path ---------------------------------------------------------------------------------------------------

@pytest.mark.engine
def test_engine_resolves_a_relative_manifest_path(monkeypatch):
    if not ENG.engine_available():
        if os.environ.get("SOUS_CHEF_LAB_REQUIRE_ENGINE") == "1" or os.environ.get("SOUSCHEF_REQUIRE_ENGINE") == "1":
            pytest.fail(ENG.unavailable_reason())
        pytest.skip(ENG.unavailable_reason())
    monkeypatch.chdir(LAB)
    eng = ENG.Engine(manifest_path="fixtures/manifest.snapshot.json")
    assert eng.manifest_path.is_absolute() and eng.manifest_path.is_file()
    assert eng.manifest_path == (LAB / "fixtures" / "manifest.snapshot.json").resolve()


@pytest.mark.engine
def test_engine_print_state_survives_a_result_over_64_kib():
    """headless.mjs truncates a piped stdout at 64 KiB after process.exit; the wrapper reads --out instead."""
    if not ENG.engine_available():
        if os.environ.get("SOUS_CHEF_LAB_REQUIRE_ENGINE") == "1" or os.environ.get("SOUSCHEF_REQUIRE_ENGINE") == "1":
            pytest.fail(ENG.unavailable_reason())
        pytest.skip(ENG.unavailable_reason())
    state = json.loads((LAB / "fixtures" / "traces" / "set_field.contentBlock-1.title.json").read_text())["case"]["site_state"]
    # pad the state well past 64 KiB with an inert slice the engine passes through untouched
    state["labPadding"] = ["x" * 1000] * 120
    eng = ENG.Engine(memory_cache=False, cache_dir=None)
    ops = [{"op": "set_fields", "target": {"kind": "section", "page": "home", "sectionId": "contentBlock-1"}, "values": {"title": "<p>Big</p>"}}]
    r = eng.run(state, ops, {**ENG.GOLDEN_CTX, "templateId": "editorial"}, print_state=True, use_cache=False)
    assert r.ok and r.state is not None
    assert len(json.dumps(r.state)) > 65536
    assert r.state["labPadding"][0] == "x" * 1000
    assert "homeSections[1].settings.title" in r.written


# --- fake service rotation ---------------------------------------------------------------------------------------------------

def done(text):
    return [("text.delta", {"text": text}), ("turn.done", {"usage": {}, "stop": "end_turn"})]


def test_rotation_serves_a_different_script_per_turn_and_wraps():
    svc = FS.FakeService(manifest=None, key="k")
    svc.scripts["c1"] = FS.Rotation([[done("first")], [done("second")]])
    svc.scripts["c2"] = [done("plain")]
    got = [svc.script_for("lab-c1", "")[0][0][1]["text"] for _ in range(3)]
    assert got == ["first", "second", "first"]
    assert [svc.script_for("lab-c2", "")[0][0][1]["text"] for _ in range(2)] == ["plain", "plain"]
    assert FS.Rotation([]).next_script() == FS.default_script()


def test_load_scripts_accepts_plain_rotate_and_trace_fixture_shapes(tmp_path):
    ev = lambda text: [{"event": "text.delta", "data": {"text": text}}, {"event": "turn.done", "data": {"stop": "end_turn"}}]
    (tmp_path / "plain.json").write_text(json.dumps([ev("p")]))
    (tmp_path / "rot.json").write_text(json.dumps({"rotate": [[ev("r1")], [ev("r2")]]}))
    (tmp_path / "slash.json").write_text(json.dumps({"id": "set_field/a/b", "rotate": [[ev("s")]]}))
    (tmp_path / "fx.json").write_text(json.dumps({"schema": 1, "streams": [ev("f")], "tool_results": []}))
    scripts = FS.load_scripts(tmp_path)
    assert scripts["plain"][0][0] == ("text.delta", {"text": "p"})
    assert isinstance(scripts["rot"], FS.Rotation) and scripts["rot"].next_script()[0][0][1]["text"] == "r1"
    assert scripts["rot"].next_script()[0][0][1]["text"] == "r2"
    assert scripts["fx"][0][0] == ("text.delta", {"text": "f"})
    assert "set_field/a/b" in scripts and "slash" not in scripts  # the document's id wins over the file stem
    # the trace fixtures shipped in fixtures/traces load too
    fx = FS.load_scripts(LAB / "fixtures" / "traces")
    assert "set_field.contentBlock-1.title" in fx and fx["set_field.contentBlock-1.title"][0][0][0] in ("text.delta", "tool.call")


# --- dataset: W_HIDDEN the expected ops share is not a defect -------------------------------------------------------------------

def test_unexpected_hidden_exempts_the_cases_own_warnings():
    from souschef_lab import dataset as DS
    case = {"meta": {"measured": {"warnings": ["W_HIDDEN:buttonText"]}}}
    assert DS.unexpected_hidden(["W_HIDDEN:buttonText"], case) == []
    assert DS.unexpected_hidden(["W_HIDDEN:buttonText", "W_HIDDEN:other"], case) == ["W_HIDDEN:other"]
    assert DS.unexpected_hidden(["W_HIDDEN:x"], {"meta": {}}) == ["W_HIDDEN:x"]
    assert DS.unexpected_hidden(["W_HIDDEN:x"], None) == ["W_HIDDEN:x"]
    assert DS.unexpected_hidden(["W_SOMETHING_ELSE"], None) == []


# --- eval rows carry passed/kind; passk derives passed from the verdict for older runs ----------------------------------------------

def test_eval_report_rows_carry_passed_and_kind_and_passk_counts_them():
    from souschef_lab import eval as EV
    from souschef_lab import passk as PK
    rows = [{"id": "a#1", "case_id": "a", "status": "ok", "verdict": "pass", "meta": {"kind": "set_field"}, "ops": [], "latency_ms": 1},
            {"id": "b#1", "case_id": "b", "status": "ok", "verdict": "wrong_value", "meta": {"kind": "toggle"}, "ops": [], "latency_ms": 1}]
    rep = EV.build_report(rows, label="t", route="r", manifest_hash="h", started="2026-10-01T00:00:00Z", rule_version="verify-1")
    assert [r["passed"] for r in rep["results"]] == [True, False]
    assert [r["kind"] for r in rep["results"]] == ["set_field", "toggle"]
    legacy = {"label": "t2", "route": "r", "manifest_hash": "h", "rule_version": "verify-1",
              "results": [{"id": "a#1", "verdict": "clarify_ok", "meta": {"kind": "set_field"}}, {"id": "b#1", "verdict": "pass"}]}
    assert PK.passed_of(legacy["results"][0]) is True and PK.passed_of({"verdict": "pass", "passed": False}) is False
    t = PK.tally([rep, legacy])
    assert t["overall"]["pass@1"] == 75.0  # a: 2/2, b: 1/2 → mean 0.75
