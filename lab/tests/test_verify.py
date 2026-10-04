"""verify.py: the verdict ladder on the six fixtures (recorded engine output),
the pure normalisation rules on measured leaves, and — when node and the
engine are present — the same goldens through the real headless runner."""

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from souschef_lab import verify as V  # noqa: E402

FIX = LAB / "fixtures" / "traces"
RECORDED = FIX / "engine" / "recorded.json"
CASES = ["set_field.contentBlock-1.title", "set_enum.productCarousel-1.productCount",
         "multi.home.contentBlock-1-title+productCarousel-1-productCount", "set_range.contentBlock-1.paddingY",
         "set_field.contentBlock-1.tagline", "clarify.home.contentBlock.title"]


@pytest.fixture(scope="module")
def recorded():
    return json.loads(RECORDED.read_text(encoding="utf-8"))


@pytest.fixture
def engine():
    return V.RecordedEngine(RECORDED)


def fixture(name):
    return V.load_fixture(FIX / f"{name}.json")


def done(seq, stop="end_turn"):
    return {"event": "turn.done", "data": {"usage": {}, "stop": stop}, "seq": seq}


# --- goldens ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", CASES)
def test_each_fixture_gets_its_verdict(name, engine):
    fx = fixture(name)
    row = V.trace_from_fixture(fx)
    v = V.verify(row, fx["case"], engine)
    assert v.verdict == fx["expected_verdict"], v.reasons
    for r in fx["expected_reasons"]:
        assert r in v.reasons
    assert v.passed == (v.verdict in V.PASSING)
    assert v.rule_version == V.RULE_VERSION


def test_the_six_verdicts_are_all_covered():
    assert {fixture(n)["expected_verdict"] for n in CASES} == {"pass", "extra_writes", "missing_writes", "wrong_value",
                                                              "invalid_op", "clarify_ok"}


def test_the_replay_uses_a_fresh_copy_of_the_case_state_and_the_golden_ctx(engine):
    fx = fixture(CASES[0])
    case = fx["case"]
    before = copy.deepcopy(case["site_state"])
    V.verify(V.trace_from_fixture(fx), case, engine)
    (call,) = engine.calls
    assert call["ctx"] == {"templateId": "editorial", **V.GOLDEN_CTX}
    assert call["state"] == before and case["site_state"] == before
    assert call["ops"] == case["expected"]["ops"]


def test_ops_are_tool_calls_minus_server_tools_report_unsupported_and_pick_asset():
    events = [
        {"event": "tool.call", "seq": 3, "data": {"id": "c", "name": "set_fields", "input": {"target": {}, "values": {"a": 1}}}},
        {"event": "tool.call", "seq": 1, "data": {"id": "a", "name": "lookup_fields", "input": {"query": "title"}}},
        {"event": "tool.call", "seq": 2, "data": {"id": "b", "name": "report_unsupported", "input": {"reason": "x"}}},
        {"event": "tool.call", "seq": 4, "data": {"id": "d", "name": "pick_asset", "input": {"purpose": "hero"}}},
        {"event": "heartbeat", "data": {}},
    ]
    assert V.ops_of(events) == [{"op": "set_fields", "target": {}, "values": {"a": 1}}]
    assert V.server_calls(events) == 1
    assert [c["id"] for c in V.tool_calls(events)] == ["a", "b", "c", "d"]  # seq order, heartbeat ignored
    assert V.saw_tool(events, "report_unsupported") and not V.saw_tool(events, "edit_list")


def test_final_text_is_the_reply_after_the_last_batch():
    events = [{"event": "text.delta", "seq": 1, "data": {"text": "Updating. "}},
              {"event": "turn.waiting", "seq": 2, "data": {"toolCallIds": []}},
              {"event": "text.delta", "seq": 3, "data": {"text": "Done"}},
              {"event": "text.delta", "seq": 4, "data": {"text": "."}}, done(5)]
    assert V.final_text(events) == "Done." and V.all_text(events) == "Updating. Done."
    assert V.outcome_of(events) == {"done": True, "errored": False, "stop": "end_turn", "error_code": None}


# --- normalisation (measured leaves) ---------------------------------------------------------------------------------

def test_a_hero_mirror_write_collapses_onto_the_slide_path(recorded):
    """Measured 2026-10-01: both a flat set_fields and edit_list update
    slides[0] write settings.headline AND settings.slides[0].headline."""
    flat = recorded["runs"]["H1_hero_flat"]["output"]
    lst = recorded["runs"]["H2_hero_list"]["output"]
    a, b = V.normalise(V.leaves_of(flat)), V.normalise(V.leaves_of(lst))
    assert a == b == {"homeSections[0].settings.slides[0].headline": '"<p>Warm cookies, every day</p>"'}


def test_a_flat_hero_leaf_alone_or_with_another_value_is_kept():
    leaves = [{"path": "homeSections[0].settings.headline", "after": "x"}]
    assert V.normalise(leaves) == {"homeSections[0].settings.headline": '"x"'}
    both = leaves + [{"path": "homeSections[0].settings.slides[0].headline", "after": "y"}]
    assert len(V.normalise(both)) == 2


def test_companions_indexes_and_removed_leaves_are_kept(recorded):
    img = V.normalise(V.leaves_of(recorded["runs"]["I_image"]["output"]))
    assert img == {"homeSections[1].settings.bgImage": '"BREADSTACK_CDN/oven.jpg"',
                   "homeSections[1].settings.bgImageSrcSet.480w": '"BREADSTACK_CDN/oven-480.jpg"',
                   "homeSections[1].settings.bgImageSrcSet.960w": '"BREADSTACK_CDN/oven-960.jpg"'}
    add = V.normalise(V.leaves_of(recorded["runs"]["K_bullets_add"]["output"]))
    assert list(add) == ["homeSections[1].settings.bullets[1].icon"]
    assert V.normalise([{"path": "a.b", "before": 1}]) == {"a.b": "null"}  # removed leaf → after None


@pytest.mark.parametrize("expected, actual, reasons", [
    ({"p": "1"}, {"p": "1"}, []),
    ({"p": "1"}, {"p": "2"}, ["wrong_value"]),
    ({"p": "1", "q": "1"}, {"p": "1"}, ["missing_writes"]),
    ({"p": "1"}, {"p": "1", "q": "1"}, ["extra_writes"]),
    ({"p": "1", "q": "1"}, {"p": "2", "r": "1"}, ["missing_writes", "extra_writes"]),  # differing paths: no wrong_value
])
def test_compare_names_wrong_missing_and_extra_in_verdict_order(expected, actual, reasons):
    assert V.compare(expected, actual) == reasons


def test_independent_ops_pass_in_any_order(engine, recorded):
    fx = fixture(CASES[2])  # the multi case: title then productCount
    case = fx["case"]
    swapped = list(reversed(case["expected"]["ops"]))
    out = recorded["runs"]["D_multi"]["output"]
    engine.runs[V.canonical(swapped)] = out  # the engine's diff is a set either way
    events = [{"event": "tool.call", "seq": i + 1, "data": {"id": f"t{i}", "name": o["op"],
                                                            "input": {k: v for k, v in o.items() if k != "op"}}}
              for i, o in enumerate(swapped)]
    events += [{"event": "turn.waiting", "seq": 3, "data": {"toolCallIds": ["t0", "t1"]}},
               {"event": "text.delta", "seq": 4, "data": {"text": "Both done."}}, done(5)]
    assert V.verify({"events": events}, case, engine).verdict == "pass"


# --- the ladder's other rungs ------------------------------------------------------------------------------------------

def test_a_turn_that_never_finished_is_no_answer(engine):
    fx = fixture(CASES[0])
    row = V.trace_from_fixture(fx)
    cut = [e for e in row["events"] if e["event"] != "turn.done"]
    v = V.verify({"events": cut}, fx["case"], engine)
    assert v.verdict == "no_answer" and "no_turn_done" in v.reasons
    errored = cut + [{"event": "turn.error", "seq": 99, "data": {"code": "turn_expired", "message": "x"}}]
    v = V.verify({"events": errored}, fx["case"], engine)
    assert v.verdict == "no_answer" and "turn_error:turn_expired" in v.reasons
    capped = cut + [done(99, stop="max_calls")]
    v = V.verify({"events": capped}, fx["case"], engine)
    assert v.verdict == "no_answer" and "stop:max_calls" in v.reasons and v.stop == "max_calls"


def test_zero_ops_where_ops_were_expected_is_no_answer(engine):
    fx = fixture(CASES[0])
    events = [{"event": "text.delta", "seq": 1, "data": {"text": "Sure, done."}}, done(2)]
    v = V.verify({"events": events}, fx["case"], engine)
    assert v.verdict == "no_answer" and "missing_writes" in v.reasons and not engine.calls  # nothing to replay


def test_a_clarify_that_does_not_ask_or_names_one_candidate_is_no_answer(engine):
    fx = fixture("clarify.home.contentBlock.title")
    case = fx["case"]
    v = V.verify({"events": [{"event": "text.delta", "seq": 1, "data": {"text": "Updated the content block."}}, done(2)]},
                 case, engine)
    assert v.verdict == "no_answer" and "not_clarifying" in v.reasons
    v = V.verify({"events": [{"event": "text.delta", "seq": 1, "data": {"text": "Do you mean contentBlock-1?"}}, done(2)]},
                 case, engine)
    assert v.verdict == "no_answer"
    by_title = "Should I change “Our story” or “Visit us”?"
    v = V.verify({"events": [{"event": "text.delta", "seq": 1, "data": {"text": by_title}}, done(2)]}, case, engine)
    assert v.verdict == "clarify_ok"  # titles count as well as ids


def test_a_clarify_or_unsupported_case_that_wrote_is_extra_writes(engine, recorded):
    fx = fixture("clarify.home.contentBlock.title")
    op = recorded["runs"]["A_cb_title"]["ops"][0]
    events = [{"event": "tool.call", "seq": 1, "data": {"id": "t", "name": op["op"], "input": {k: v for k, v in op.items() if k != "op"}}},
              {"event": "turn.waiting", "seq": 2, "data": {"toolCallIds": ["t"]}},
              {"event": "text.delta", "seq": 3, "data": {"text": "Done."}}, done(4)]
    assert V.verify({"events": events}, fx["case"], engine).verdict == "extra_writes"
    unsupported = dict(fx["case"], kind="unsupported", expected={"ops": [], "diff": []})
    assert V.verify({"events": events}, unsupported, engine).verdict == "extra_writes"


def test_unsupported_ok_needs_the_report_and_a_finished_turn(engine):
    fx = fixture("clarify.home.contentBlock.title")
    case = dict(fx["case"], kind="unsupported", expected={"ops": [], "diff": []})
    report = {"event": "tool.call", "seq": 1, "data": {"id": "r", "name": "report_unsupported",
                                                       "input": {"reason": "custom code is not a site setting"}}}
    text = {"event": "text.delta", "seq": 2, "data": {"text": "I can't add custom scripts here."}}
    assert V.verify({"events": [report, text, done(3)]}, case, engine).verdict == "unsupported_ok"
    v = V.verify({"events": [text, done(3)]}, case, engine)
    assert v.verdict == "no_answer" and "not_reported" in v.reasons
    assert V.verify({"events": [report, text]}, case, engine).verdict == "no_answer"


def test_invalid_op_comes_first_and_the_engine_exit_1_counts(engine):
    fx = fixture("set_field.contentBlock-1.tagline")
    row = V.trace_from_fixture(fx)
    v = V.verify(row, fx["case"], engine)
    assert v.verdict == "invalid_op" and v.engine_errors[0]["code"] == "E_FIELD" and v.actual_diff == {}

    class Crashing:
        def run(self, state, ops, ctx):
            return {"ok": False, "exit_code": 1, "error": "vite failed", "results": [], "changes": []}

    v = V.verify(row, fx["case"], Crashing())
    assert v.verdict == "invalid_op" and v.engine_errors == [{"code": "E_ENGINE", "message": "vite failed"}]


def test_the_verdict_dict_carries_counts_paths_and_values_never_text(engine):
    fx = fixture(CASES[1])
    d = V.verify(V.trace_from_fixture(fx), fx["case"], engine).to_dict()
    assert d["verdict"] == "extra_writes" and d["warnings"] == ["W_HIDDEN:productCount"] and d["ops"] == ["set_fields"]
    assert d["final_text_chars"] > 0 and "centred" not in json.dumps(d)
    assert d["expected_diff"] == [["homeSections[2].settings.productCount", "4"]]
    assert ["homeSections[2].settings.contentAlign", '"center"'] in d["actual_diff"]


def test_the_recorded_engine_refuses_to_guess(engine):
    with pytest.raises(KeyError):
        engine.run({}, [{"op": "reset", "target": {"kind": "branding"}}], {})


def test_trace_from_fixture_is_a_collector_row_with_tool_result_events():
    fx = fixture(CASES[0])
    row = V.trace_from_fixture(fx, sample=2, model="opus")
    assert row["id"] == f"{fx['case']['id']}#2" and row["sample"] == 2 and row["model"] == "opus"
    assert row["status"] == "ok" and row["stop"] == "end_turn" and row["turn_id"] == "turn-0001"
    results = [e for e in row["events"] if e["event"] == "tool.result"]
    assert len(results) == 1 and results[0]["data"]["toolCallId"] == "tc_1" and results[0]["data"]["for_seq"] == 2
    assert results[0]["seq"] == max(e["seq"] for e in fx["streams"][0]) + 1  # between the two streams (README §4)
    assert results[0]["seq"] < min(e["seq"] for e in fx["streams"][1])
    assert row["meta"]["kind"] == "set_field" and row["meta"]["section"] == "contentBlock-1"
    assert row["ops"] == fx["case"]["expected"]["ops"]


def test_holdout_is_a_hash_of_the_field_key_near_twenty_percent():
    keys = [f"section{i}.field{j}" for i in range(40) for j in range(50)]
    share = sum(V.holdout_of(k) for k in keys) / len(keys)
    assert 0.17 <= share <= 0.23
    assert V.holdout_of("contentBlock.title") == V.holdout_of("contentBlock.title")  # stable
    assert all(V.holdout_of(k, pct=0) is False for k in keys[:10])


def test_the_cli_re_verifies_a_log_against_cases(tmp_path, capsys):
    rows, cases = [], []
    for n in CASES:
        fx = fixture(n)
        rows.append(V.trace_from_fixture(fx))
        cases.append(fx["case"])
    (tmp_path / "log.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    out = tmp_path / "verdicts.jsonl"
    assert V.main(["--log", str(tmp_path / "log.jsonl"), "--cases", str(tmp_path / "cases.jsonl"),
                   "--recorded", str(RECORDED), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "pass             1" in printed and "clarify_ok       1" in printed and "rows 6" in printed
    verdicts = [json.loads(l) for l in out.read_text().splitlines()]
    assert [v["verdict"] for v in verdicts] == ["pass", "extra_writes", "missing_writes", "wrong_value", "invalid_op",
                                                "clarify_ok"]


def test_default_engine_is_the_subprocess_engine_or_a_named_unavailability():
    try:
        from souschef_lab import engine as E
    except ImportError:
        with pytest.raises(V.EngineUnavailable):
            V.default_engine()
        return
    if E.engine_available():
        assert isinstance(V.default_engine(), E.Engine)
    else:
        with pytest.raises(V.EngineUnavailable):
            V.default_engine()


def test_run_engine_accepts_an_engine_result_and_turns_engine_error_into_exit_1():
    class Result:
        returncode = 2

        def to_dict(self):
            return {"ok": False, "results": [{"ok": False, "error": {"code": "E_FIELD"}, "warnings": [], "written": []}],
                    "changes": []}

    class EngineError(RuntimeError):
        pass

    class Obj:
        def run(self, state, ops, ctx):
            return Result()

    class Raises:
        def run(self, state, ops, ctx):
            raise EngineError("engine exit 1: vite failed")

    class Other:
        def run(self, state, ops, ctx):
            raise ValueError("a bug")

    out = V.run_engine(Obj(), {}, [], {})
    assert out["exit_code"] == 2 and V.engine_errors(out) == [{"code": "E_FIELD"}]
    out = V.run_engine(Raises(), {}, [], {})
    assert out["exit_code"] == 1 and V.engine_errors(out)[0]["code"] == "E_ENGINE" and "vite failed" in out["error"]
    with pytest.raises(ValueError):
        V.run_engine(Other(), {}, [], {})


# --- the real engine --------------------------------------------------------------------------------------------------

from souschef_lab.engine import find_engine as _find_engine  # noqa: E402

ENGINE = _find_engine(os.environ.get("ENGINE") or os.environ.get("SOUSCHEF_ENGINE"))  # else $SOUS_CHEF_ENGINE / $COOKIE_CUTTER_DIR
ENGINE_REQUIRED = os.environ.get("SOUS_CHEF_LAB_REQUIRE_ENGINE") == "1" or os.environ.get("SOUSCHEF_REQUIRE_ENGINE") == "1"


class HeadlessEngine:
    """Minimal subprocess wrapper, used only when souschef_lab.engine is
    absent (real_engine() prefers the package's Engine)."""

    def __init__(self, script):
        self.script = Path(script)

    def run(self, state, ops, ctx):
        req = json.dumps({"state": state, "ops": ops, "ctx": ctx})
        p = subprocess.run(["node", str(self.script)], input=req, capture_output=True, text=True,
                           cwd=self.script.parents[3], timeout=120)
        if p.returncode == 1:
            return {"ok": False, "exit_code": 1, "error": p.stderr[-500:], "results": [], "changes": []}
        out = json.loads(p.stdout)
        out["exit_code"] = p.returncode
        return out


@pytest.mark.skipif(not ENGINE_REQUIRED and not (ENGINE and Path(ENGINE).exists() and shutil.which("node")),
                    reason="needs ENGINE=…/sousChef/ops/headless.mjs and node (CI: SOUS_CHEF_LAB_REQUIRE_ENGINE=1 fails instead)")
@pytest.mark.parametrize("name", CASES)
def test_the_real_engine_agrees_with_the_recorded_goldens(name):
    fx = fixture(name)
    row = V.trace_from_fixture(fx)
    v = V.verify(row, fx["case"], real_engine())
    assert v.verdict == fx["expected_verdict"], v.reasons


def real_engine():
    try:
        from souschef_lab.engine import Engine
    except ImportError:
        return HeadlessEngine(ENGINE)
    return Engine(ENGINE, memory_cache=False)


# --- engine fingerprint on verdicts (LAB.md §2) ---------------------------------------------------------------------------

def test_verdict_carries_both_engine_fingerprints():
    class Stamped(V.RecordedEngine):
        fingerprint = "fp-now"

    fx = fixture(CASES[0])
    row = V.trace_from_fixture(fx)
    v = V.verify(row, dict(fx["case"], engine_fingerprint="fp-then"), Stamped(RECORDED))
    assert v.engine_fingerprint == "fp-now" and v.case_engine_fingerprint == "fp-then" and v.engine_mismatch
    assert v.to_dict()["engine_mismatch"] is True
    same = V.verify(row, dict(fx["case"], engine_fingerprint="fp-now"), Stamped(RECORDED))
    assert not same.engine_mismatch
    unknown = V.verify(row, fx["case"], V.RecordedEngine(RECORDED))  # neither side known: never a mismatch
    assert unknown.engine_fingerprint is None and unknown.case_engine_fingerprint is None and not unknown.engine_mismatch


def test_verify_cli_warns_on_a_stale_case(tmp_path, capsys):
    fx = fixture(CASES[0])
    row = V.trace_from_fixture(fx)
    log, cases_path = tmp_path / "log.jsonl", tmp_path / "cases.jsonl"
    log.write_text(json.dumps(row) + "\n")
    cases_path.write_text(json.dumps(dict(fx["case"], engine_fingerprint="fp-then")) + "\n")
    orig = V.RecordedEngine

    class Stamped(orig):
        fingerprint = "fp-now"

    V.RecordedEngine = Stamped
    try:
        assert V.main(["--log", str(log), "--cases", str(cases_path), "--recorded", str(RECORDED)]) == 0
    finally:
        V.RecordedEngine = orig
    err = capsys.readouterr().err
    assert "measured with another engine" in err and "cases remeasure" in err
