"""dataset.py: the keep-rule matrix, the thermometer-compatible row shape,
the split, and build() end to end on the six fixtures."""

import copy
import json
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from souschef_lab import dataset as D  # noqa: E402
from souschef_lab import verify as V  # noqa: E402

FIX = LAB / "fixtures" / "traces"
RECORDED = FIX / "engine" / "recorded.json"
PASS = "set_field.contentBlock-1.title"
EXTRA = "set_enum.productCarousel-1.productCount"
CLARIFY = "clarify.home.contentBlock.title"
ALL = [PASS, EXTRA, "multi.home.contentBlock-1-title+productCarousel-1-productCount", "set_range.contentBlock-1.paddingY",
       "set_field.contentBlock-1.tagline", CLARIFY]
SYSTEM = ("RULES: edit only via tools.\n", "INDEX: s:contentBlock f=126\n")
TOOLS = [{"name": "set_fields", "description": "Set fields", "input_schema": {"type": "object", "properties": {}}}]


@pytest.fixture
def engine():
    return V.RecordedEngine(RECORDED)


def fx(name):
    return V.load_fixture(FIX / f"{name}.json")


def row_and_case(name, **over):
    """The fixture's row and its case with holdout off (4 of the 6 fixtures
    fall in the exam band under the real sha1 rule; the holdout test sets it)."""
    f = fx(name)
    row = V.trace_from_fixture(f)
    row.update(over)
    return row, dict(f["case"], holdout=False)


def judged(row, case, engine, **kw):
    v = V.verify(row, case, engine)
    return D.reasons_of(row, case, v, **kw)


# --- keep rules -------------------------------------------------------------------------------------------------------

def test_a_passing_finished_row_has_no_reason(engine):
    row, case = row_and_case(PASS)
    assert judged(row, case, engine, max_seq=32768) == []


def test_error_status_and_stop_reason(engine):
    row, case = row_and_case(PASS, status="error", error="superseded")
    assert judged(row, case, engine)[0] == "error"
    row, case = row_and_case(PASS)
    for e in row["events"]:
        if e["event"] == "turn.done":
            e["data"]["stop"] = "max_tokens"
    reasons = judged(row, case, engine)
    assert "stop_reason" in reasons and "not_pass" in reasons  # the verifier says no_answer too


def test_not_pass_and_warnings(engine):
    row, case = row_and_case(EXTRA)
    reasons = judged(row, case, engine)
    assert reasons == ["not_pass", "warnings"]  # extra_writes, and W_HIDDEN:productCount on the replayed op


def test_empty_reply(engine):
    row, case = row_and_case(PASS)
    row["events"] = [e for e in row["events"] if not (e["event"] == "text.delta" and e["seq"] > 3)]
    assert judged(row, case, engine) == ["empty_reply"]


def test_lookup_storm_counts_server_tools_only(engine):
    row, case = row_and_case(PASS)
    calls = [{"event": "tool.call", "seq": 0.1 * (i + 1), "turnId": "t",
              "data": {"id": f"l{i}", "name": "lookup_fields" if i % 2 else "describe_target", "input": {}}}
             for i in range(4)]
    row["events"] = calls + row["events"]
    assert judged(row, case, engine) == ["lookup_storm"]
    row["events"] = calls[:3] + row["events"][4:]
    assert "lookup_storm" not in judged(row, case, engine)


def test_holdout_and_over_length(engine):
    row, case = row_and_case(PASS)
    case = dict(case, holdout=True)
    assert judged(row, case, engine) == ["holdout"]
    row, case = row_and_case(PASS)
    assert judged(row, case, engine, max_seq=10) == ["over_length"]
    assert judged(row, case, engine, max_seq=10, tokens=lambda hf: 1) == []


def test_no_case_is_its_own_reason(engine):
    row, case = row_and_case(PASS)
    assert D.reasons_of(row, None, V.verify(row, case, engine)) == ["no_case"]


def test_rules_order_is_the_documented_one():
    assert D.RULES == ("error", "messages_mismatch", "stop_reason", "not_pass", "warnings", "empty_reply", "lookup_storm",
                       "holdout", "duplicate", "over_length")


# --- rows -------------------------------------------------------------------------------------------------------------

def test_to_hf_has_the_thermometer_shape(engine):
    row, case = row_and_case(PASS)
    hf = D.to_hf(row, case, SYSTEM, TOOLS)
    roles = [m["role"] for m in hf["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert hf["messages"][0]["content"] == "".join(SYSTEM)  # joined with nothing between (hf.join_system)
    user = hf["messages"][1]["content"]
    # the service's bytes (render.context_block): compact JSON on its own line inside the tags
    assert user.startswith("<context>\n{") and user.endswith("\n</context>\n" + case["request"]) and '"page":"home"' in user
    call = hf["messages"][2]["tool_calls"][0]
    assert call["type"] == "function" and call["function"]["name"] == "set_fields"
    assert isinstance(call["function"]["arguments"], str) and json.loads(call["function"]["arguments"])["values"] == {
        "title": "<p>Fresh from the oven</p>"}
    tool = hf["messages"][3]
    assert tool["tool_call_id"] == "tc_1" and tool["name"] == "set_fields"
    assert json.loads(tool["content"])["written"] == ["homeSections[1].settings.title"]
    assert hf["messages"][4]["content"].startswith("Done") and "tool_calls" not in hf["messages"][4]
    assert hf["tools"] == [{"type": "function", "function": {"name": "set_fields", "description": "Set fields",
                                                             "parameters": {"type": "object", "properties": {}}}}]
    m = hf["meta"]
    assert m["answer_start"] == 2 and m["calls"] == 1 and m["tools_called"] == ["set_fields"]
    assert m["family"] == "section:contentBlock" and m["kind"] == "set_field" and m["missing_tool_results"] == 0
    assert all("reasoning_content" not in msg for msg in hf["messages"])  # never fabricated


def test_columns_are_uniform_across_rows(engine):
    rows = [D.to_hf(*row_and_case(n), SYSTEM, TOOLS) for n in ALL]
    assert len({tuple(sorted(r.keys())) for r in rows}) == 1
    assert len({tuple(sorted(r["meta"].keys())) for r in rows}) == 1
    for r in rows:
        for m in r["messages"]:
            if m["role"] == "assistant":
                assert isinstance(m["content"], str)
                for c in m.get("tool_calls") or []:
                    assert set(c) == {"id", "type", "function"} and isinstance(c["function"]["arguments"], str)


def test_a_failed_browser_result_is_marked_is_error_and_an_unanswered_server_tool_is_counted():
    row, case = row_and_case("set_field.contentBlock-1.tagline")
    msgs = D.messages_from_events(row["events"], case["context"], case["request"])
    result = msgs[2]["content"][0]
    # loop.browserResult on failure: the flat api.ToolError {code, message, key?} with is_error (never nested under "error")
    assert result["type"] == "tool_result" and result["is_error"] is True and json.loads(result["content"])["code"] == "E_FIELD"
    row["events"].insert(0, {"event": "tool.call", "seq": 0.5, "data": {"id": "u", "name": "report_unsupported",
                                                                        "input": {"reason": "x"}}})
    hf = D.to_hf(row, case)
    assert hf["meta"]["missing_tool_results"] == 1 and "report_unsupported" in hf["meta"]["tools_called"]


def test_to_hf_always_rebuilds_from_events_and_a_stale_messages_cache_is_rejected(engine):
    """Refuter 2026-10-01: to_hf preferred row.messages, so a log written before the renderer trained on bytes the
    service never sends. Now row.messages is a cache that must equal the rebuild, or the row is rejected."""
    row, case = row_and_case(PASS)
    assert "messages" not in row  # a trace row has none: nothing to compare
    assert judged(row, case, engine) == [] and D.stored_messages_match(row, case)
    rebuilt = D.rebuild_messages(row, case)
    assert rebuilt[0]["content"][0]["text"].startswith("<context>\n{") and rebuilt[2]["content"][0]["type"] == "tool_result"
    fresh = dict(row, messages=json.loads(json.dumps(rebuilt)))  # the collector's cache: equal → kept
    assert judged(fresh, case, engine) == [] and D.to_hf(fresh, case)["messages"] == D.to_hf(row, case)["messages"]
    # the pre-renderer shapes: no inner newlines in the context block, Python-spaced tool results
    stale = json.loads(json.dumps(rebuilt))
    stale[0]["content"][0]["text"] = stale[0]["content"][0]["text"].replace("<context>\n", "<context>").replace("\n</context>", "</context>")
    stale[2]["content"][0]["content"] = '{"ok": true, "written": ["homeSections[1].settings.title"]}'
    old = dict(row, messages=stale)
    assert judged(old, case, engine) == ["messages_mismatch"] and not D.stored_messages_match(old, case)
    hf = D.to_hf(old, case)  # the HF row still reads the REBUILT bytes, never the stale cache
    assert hf["messages"][0]["content"] == rebuilt[0]["content"][0]["text"]
    assert hf["messages"][2]["content"] == rebuilt[2]["content"][0]["content"] == '{"ok":true,"written":["homeSections[1].settings.title"]}'
    b = D.build([old, fresh], {case["id"]: case}, engine, D.BuildConfig(system=SYSTEM, tools=TOOLS))
    assert b.stats["messages_mismatch"] == {"rows": 1, "sample": [old["id"]]}
    assert b.stats["rejects_by_reason"] == {"messages_mismatch": 1} or b.stats["rejects_any_reason"].get("messages_mismatch") == 1


def test_turn_inputs_prefer_the_rows_own_context_and_text():
    """collect.py sends render.turn_context(case) (`{}` for a context-less case) and logs it as row.context; the rebuild
    reads the same object back — an empty context is an object the service renders as one, never a bare text."""
    from souschef_lab import collect as C
    from souschef_lab import render as R
    case = {"request": "hi"}
    assert R.turn_context(case) == {} and R.turn_context({"context": {"page": "home"}}) == {"page": "home"}
    assert R.turn_context({"context": None}) == {}
    assert R.turn_inputs({"context": {}, "text": "hi"}, case) == ({}, "hi")
    assert R.turn_inputs({"context": None, "text": "hi"}, case) == (None, "hi")  # a trace row's null context stays null
    assert R.turn_inputs({}, case) == ({}, "hi")  # an older row: the case's, as the collector sent it
    assert R.turn_inputs({}, {"context": {"page": "home"}, "request": "x"}) == ({"page": "home"}, "x")
    row = {"events": [{"event": "turn.done", "data": {"stop": "end_turn", "seq": 1}}]}
    user = D.rebuild_messages(row, case)[0]["content"][0]["text"]
    assert user == "<context>\n{}\n</context>\nhi" == C.user_text(R.turn_context(case), "hi")


def test_estimate_tokens_is_bytes_over_three_rounded_up():
    hf = {"messages": [{"role": "user", "content": "héllo"}], "tools": []}
    assert D.estimate_tokens(hf) == -(-D.row_bytes(hf) // 3) and D.BYTES_PER_TOKEN == 3.0


def test_split_is_by_case_id_and_near_the_share():
    ids = [f"case-{i}" for i in range(2000)]
    val = sum(D.split_of(i, 10) == "val" for i in ids) / len(ids)
    assert 0.08 <= val <= 0.12
    assert D.split_of("x", 0) == "train" and all(D.split_of(i, 100) == "val" for i in ids[:5])
    assert D.split_of("case-7", 10) == D.split_of("case-7", 10)


@pytest.mark.parametrize("case, family", [
    ({"target": {"kind": "section", "sectionId": "contentBlock-2"}}, "section:contentBlock"),
    ({"meta": {"section_type": "hero"}}, "section:hero"),
    ({"target": {"kind": "global", "panel": "HeaderSection"}}, "panel:HeaderSection"),
    ({"target": {"kind": "branding"}}, "branding"),
    ({"target": {"kind": "page", "page": "faq"}}, "page:faq"),
    ({"kind": "clarify", "target": {"kind": "page", "page": "home"}}, "kind:clarify"),
    ({"kind": "multi", "target": {"kind": "page", "page": "home"}}, "kind:multi"),
    ({"kind": "clarify", "meta": {"section_type": "contentBlock"}}, "section:contentBlock"),
])
def test_family_of(case, family):
    assert D.family_of(case) == family


# --- build ------------------------------------------------------------------------------------------------------------

def all_rows_and_cases():
    rows, cases = [], {}
    for n in ALL:
        f = fx(n)
        rows.append(V.trace_from_fixture(f))
        cases[f["case"]["id"]] = dict(f["case"], holdout=False)
    return rows, cases


def test_build_keeps_the_passing_rows_and_names_every_reject(engine):
    rows, cases = all_rows_and_cases()
    for c in cases.values():
        c["holdout"] = False
    b = D.build(rows, cases, engine, D.BuildConfig(system=SYSTEM, tools=TOOLS))
    kept = b.train + b.val
    assert {r["meta"]["case_id"] for r in kept} == {PASS, CLARIFY}
    assert {r["case_id"]: r["reasons"][0] for r in b.rejects} == {
        EXTRA: "not_pass", ALL[2]: "not_pass", ALL[3]: "not_pass", ALL[4]: "not_pass"}
    assert b.stats["kept"] == 2 and b.stats["rejected"] == 4 and b.stats["rejects_by_reason"] == {"not_pass": 4}
    assert b.stats["rejects_any_reason"]["warnings"] == 1
    assert b.stats["verdicts_by_kind"]["set_field"] == {"pass": 1, "invalid_op": 1}
    assert b.exam == [] and b.manifest["rule_version"] == D.RULE_SET_VERSION and b.manifest["counts"]["train"] + b.manifest["counts"]["val"] == 2
    assert b.manifest["manifest_hashes"] == [cases[PASS]["manifest_hash"]]


def test_holdout_cases_go_to_exam_never_train(engine):
    rows, cases = all_rows_and_cases()
    cases[PASS]["holdout"] = True
    b = D.build(rows, cases, engine)
    assert [c["id"] for c in b.exam] == [PASS]
    assert all(r["meta"]["case_id"] != PASS for r in b.train + b.val)
    assert next(r for r in b.rejects if r["case_id"] == PASS)["reasons"] == ["holdout"]


def test_duplicate_samples_are_kept_once(engine):
    rows, cases = all_rows_and_cases()
    cases[PASS]["holdout"] = False
    twin = copy.deepcopy(rows[0])
    twin["id"], twin["sample"] = f"{PASS}#2", 2
    b = D.build(rows + [twin], {PASS: cases[PASS]}, engine)
    assert len(b.train + b.val) == 1
    assert next(r for r in b.rejects if r["id"] == f"{PASS}#2")["reasons"] == ["duplicate"]


def test_best_of_n_build_records_the_selection(engine):
    rows, cases = all_rows_and_cases()
    for c in cases.values():
        c["holdout"] = False
    b = D.build(rows, cases, engine, D.BuildConfig(best_of_n=True))
    kept = b.train + b.val
    assert len(kept) == 2 and all(r["meta"]["selection"]["rank"] == 1 for r in kept)
    assert b.stats["selection"]["rows"] == 2 and b.manifest["mode"] == "best-of-n"


def test_the_cli_writes_the_six_files(tmp_path, engine):
    rows, cases = all_rows_and_cases()
    (tmp_path / "log.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases.values()) + "\n")
    (tmp_path / "rules.txt").write_text(SYSTEM[0])
    out = tmp_path / "out"
    rc = D.main(["--log", str(tmp_path / "log.jsonl"), "--cases", str(tmp_path / "cases.jsonl"), "--out", str(out),
                 "--recorded", str(RECORDED), "--system-rules", str(tmp_path / "rules.txt")])
    assert rc == 0
    assert sorted(p.name for p in out.iterdir()) == ["dataset-manifest.json", "exam.jsonl", "rejects.jsonl", "stats.json",
                                                     "train.jsonl", "val.jsonl"]
    man = json.loads((out / "dataset-manifest.json").read_text())
    assert man["system_prompt"]["blocks"] == 1 and man["keep_rules"] == list(D.RULES)
    first = json.loads((out / "train.jsonl").read_text().splitlines()[0]) if (out / "train.jsonl").read_text().strip() \
        else json.loads((out / "val.jsonl").read_text().splitlines()[0])
    assert first["messages"][0] == {"role": "system", "content": SYSTEM[0]}


# --- engine fingerprint (LAB.md §2) -----------------------------------------------------------------------------------

def test_hf_meta_and_manifest_carry_the_engine_fingerprint(engine):
    row, case = row_and_case(PASS)
    assert D.to_hf(row, dict(case, engine_fingerprint="fp-1"), SYSTEM, TOOLS)["meta"]["engine_fingerprint"] == "fp-1"
    assert D.to_hf(row, case, SYSTEM, TOOLS)["meta"]["engine_fingerprint"] is None

    class Stamped(V.RecordedEngine):
        fingerprint = "fp-now"

    stale = dict(case, engine_fingerprint="fp-then")
    b = D.build([row], {case["id"]: stale}, Stamped(RECORDED), D.BuildConfig(system=SYSTEM, tools=TOOLS))
    assert b.manifest["engine_fingerprint"] == "fp-now" and b.manifest["case_engine_fingerprints"] == ["fp-then"]
    assert b.stats["engine"] == {"fingerprint": "fp-now", "case_fingerprints": ["fp-then"], "stale_cases": 1,
                                 "stale_sample": [case["id"]]}
    fresh = D.build([row], {case["id"]: dict(case, engine_fingerprint="fp-now")}, Stamped(RECORDED),
                    D.BuildConfig(system=SYSTEM, tools=TOOLS))
    assert fresh.stats["engine"]["stale_cases"] == 0
    plain = D.build([row], {case["id"]: case}, engine, D.BuildConfig(system=SYSTEM, tools=TOOLS))
    assert plain.manifest["engine_fingerprint"] is None and plain.manifest["case_engine_fingerprints"] == []
