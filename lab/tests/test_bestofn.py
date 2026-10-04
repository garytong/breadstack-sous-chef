"""bestofn.py: the Judge on the fixtures, rank_key, the adaptive stop
(thermometer's pins, verbatim), and select's rules on synthetic samples."""

import copy
import json
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from souschef_lab import bestofn as B  # noqa: E402
from souschef_lab import verify as V  # noqa: E402

FIX = LAB / "fixtures" / "traces"
RECORDED = FIX / "engine" / "recorded.json"
PASS = "set_field.contentBlock-1.title"
EXTRA = "set_enum.productCarousel-1.productCount"
CLARIFY = "clarify.home.contentBlock.title"
MULTI = "multi.home.contentBlock-1-title+productCarousel-1-productCount"


def fx(name):
    return V.load_fixture(FIX / f"{name}.json")


@pytest.fixture
def engine():
    return V.RecordedEngine(RECORDED)


@pytest.fixture
def judge(engine):
    cases = {}
    for p in FIX.glob("*.json"):
        f = V.load_fixture(p)
        f["case"]["holdout"] = False
        cases[f["case"]["id"]] = f["case"]
    return B.Judge(cases, engine)


def sample(name, k=1, **over):
    row = V.trace_from_fixture(fx(name), sample=k)
    row.update(over)
    return row


def with_reply(row, text):
    row = copy.deepcopy(row)
    for e in row["events"]:
        if e["event"] == "text.delta" and e["seq"] == max(x["seq"] for x in row["events"] if x["event"] == "text.delta"):
            e["data"]["text"] = text
    return row


def with_server_calls(row, n):
    row = copy.deepcopy(row)
    row["events"] = [{"event": "tool.call", "seq": 0.01 * (i + 1), "data": {"id": f"s{i}", "name": "lookup_fields", "input": {}}}
                     for i in range(n)] + row["events"]
    return row


# --- the judge --------------------------------------------------------------------------------------------------------

def test_the_judge_passes_the_pass_fixture_and_names_the_failures(judge):
    j = judge(sample(PASS))
    assert j.passed and j.reasons == [] and j.sample_id == f"{PASS}#1" and j.kind == "set_field"
    assert j.verdict.verdict == "pass" and j.hf["meta"]["answer_start"] == 1  # no system prompt given
    j = judge(sample(EXTRA))
    assert not j.passed and j.reasons == ["not_pass", "warnings"] and j.category == "extra_writes"
    j = judge(sample(CLARIFY))
    assert j.passed and j.verdict.verdict == "clarify_ok"


def test_an_unknown_case_is_never_material(judge):
    j = judge(sample(PASS, case_id="nope"))
    assert j.reasons == ["no_case"] and j.never and j.prompt_excluded and j.case is None


def test_a_holdout_sample_is_prompt_excluded(engine):
    f = fx(PASS)
    f["case"]["holdout"] = True
    j = B.Judge({f["case"]["id"]: f["case"]}, engine)(V.trace_from_fixture(f))
    assert j.reasons == ["holdout"] and j.prompt_excluded and j.never


def test_categories_follow_the_verdict_order():
    assert B.category_of(["invalid_op", "extra_writes"]) == "invalid_op"
    assert B.category_of(["missing_writes"]) == "missing_writes" and B.category_of(["no_answer"]) == "other"
    assert B.CATEGORY_ORDER == ("invalid_op", "wrong_value", "missing_writes", "extra_writes")


# --- rank_key ---------------------------------------------------------------------------------------------------------

def test_rank_key_pass_first_then_fewest_ops_server_calls_shortest_reply_then_sample(judge):
    good = judge(sample(PASS))
    bad = judge(sample(EXTRA))
    two_ops = judge(sample(PASS, 9))
    two_ops.row["events"].insert(2, {"event": "tool.call", "seq": 2.5, "data": {"id": "tc_x", "name": "set_fields", "input": {}}})
    assert B.rank_key(good) < B.rank_key(bad)
    assert B.browser_ops(two_ops) == 2 and B.rank_key(good)[1] < B.rank_key(two_ops)[1]
    chatty = judge(with_server_calls(sample(PASS, 2), 1))
    assert chatty.passed and B.rank_key(good) < B.rank_key(chatty) and B.server_calls(chatty) == 1
    short = judge(with_reply(sample(PASS, 3), "Done."))
    assert B.reply_chars(short) == 5 and B.rank_key(short) < B.rank_key(good)
    later = judge(sample(PASS, 4))
    assert B.rank_key(good) < B.rank_key(later) and B.rank_key(later)[-2] == 4


def test_op_path_is_first_seen_op_names_with_list_actions():
    row = {"events": [
        {"event": "tool.call", "seq": 1, "data": {"id": "a", "name": "lookup_fields", "input": {}}},
        {"event": "tool.call", "seq": 2, "data": {"id": "b", "name": "edit_list", "input": {"action": "update"}}},
        {"event": "tool.call", "seq": 3, "data": {"id": "c", "name": "set_fields", "input": {}}},
        {"event": "tool.call", "seq": 4, "data": {"id": "d", "name": "edit_list", "input": {"action": "update"}}},
    ]}
    j = B.Judged("x#1", "x", 1, "m", {}, row, None, True, [])
    assert B.op_path(j) == ("edit_list:update", "set_fields") and B.browser_ops(j) == 3


# --- the adaptive stop (thermometer's pins) ---------------------------------------------------------------------------

RULE = B.Adaptive()


def test_enough_passes_stop_a_case():
    assert B.question_state([True], 4, (0, 0), RULE) == "open"
    assert B.question_state([True, False, True], 4, (0, 0), RULE) == "enough"
    assert B.question_state([True, True], 4, (0, 0), B.Adaptive(enough=3)) == "open"


def test_hopeless_needs_its_own_failures_and_a_failing_family():
    weak = (0, 8)
    assert B.question_state([False], 4, weak, RULE) == "open"  # 1 < ceil(4/2)
    assert B.question_state([False, False], 4, weak, RULE) == "hopeless"
    assert B.question_state([False, False], 4, (4, 8), RULE) == "open"  # the family can do it
    assert B.question_state([False, False], 4, (0, 7), RULE) == "open"  # too little family evidence
    assert B.question_state([False, False], 4, (1, 8), RULE) == "open"  # 1/8 is the band's floor, not under it
    assert B.question_state([False, False, False], 8, weak, RULE) == "open"  # ceil(8/2) = 4
    assert B.question_state([False], 4, weak, B.Adaptive(hopeless_after=1)) == "hopeless"
    assert RULE.to_dict(4) == {"enough": 2, "hopeless_after": 2, "family_min": 8, "family_rate": 0.125}


def test_done_excluded_and_no_rule():
    assert B.question_state([True] * 4, 4, (0, 0), RULE) == "done"
    assert B.question_state([False], 4, (0, 0), RULE, excluded=True) == "excluded"
    assert B.question_state([True, True, True], 4, (0, 0), None) == "open"
    assert B.question_state([False] * 4, 4, (0, 99), None) == "done"


def test_family_outcomes_count_passes_and_finished_samples(judge):
    js = [judge(sample(PASS)), judge(sample(EXTRA)), judge(sample(CLARIFY))]
    fam = B.family_outcomes(js)
    assert fam["section:contentBlock"] == (1, 1) and fam["section:productCarousel"] == (0, 1) and fam["kind:clarify"] == (1, 1)


# --- select -----------------------------------------------------------------------------------------------------------

def tokens(j):
    return 100


def test_only_passing_samples_are_kept_and_one_per_case(judge):
    js = [judge(sample(PASS, 1)), judge(sample(PASS, 2)), judge(sample(EXTRA, 1))]
    kept, rep = B.select(js, tokens, B.SelectConfig())
    assert [k.judged.sample_id for k in kept] == [f"{PASS}#1"] and kept[0].rank == 1
    assert rep["samples"] == 3 and rep["passing"] == 2 and rep["cases_with_a_pass"] == 1 and rep["rows"] == 1


def test_a_second_row_only_on_another_op_path(judge):
    best = judge(sample(PASS, 1))
    same_path = judge(sample(PASS, 2))
    other = judge(sample(PASS, 3))
    for e in other.row["events"]:  # after judging: only op_path reads this (an unrecorded op would raise in the judge)
        if e["event"] == "tool.call":
            e["data"]["name"] = "restore_default"
    superset = judge(sample(PASS, 4))
    superset.row["events"].insert(0, {"event": "tool.call", "seq": 0.5, "data": {"id": "r", "name": "reset", "input": {}}})
    assert set(B.op_path(superset)) > set(B.op_path(best))  # the best's op plus one more: the same route
    # family_share=1: with one case the 3 % cap is one row and would take the second (rule 6, tested below)
    kept, rep = B.select([best, same_path, other, superset], tokens, B.SelectConfig(family_share=1.0))
    assert [(k.judged.sample_id, k.rank) for k in kept] == [(f"{PASS}#1", 1), (f"{PASS}#3", 2)]
    assert rep["second_rows"] == 1
    kept, _ = B.select([best, same_path, other, superset], tokens, B.SelectConfig(family_share=1.0, second_path=False))
    assert len(kept) == 1


def test_reply_above_the_p90_of_its_kind_is_dropped_but_never_a_cases_last_pass(judge):
    short = [judge(with_reply(sample(PASS, k), "Done.")) for k in range(1, 10)]
    long = judge(with_reply(sample(PASS, 10), "Done, " * 50))
    kept, rep = B.select(short + [long], tokens, B.SelectConfig())
    assert rep["dropped_reply_above_p90"] == 1 and rep["reply_p90_chars"] == {"set_field": 5}
    lonely = judge(with_reply(sample(CLARIFY, 1), "Which content block: contentBlock-1 or contentBlock-2? " * 4))
    kept, rep = B.select(short + [long, lonely], tokens, B.SelectConfig())
    assert any(k.judged.case_id == CLARIFY for k in kept) and rep["kept_above_p90_only_passes"] == 0  # its own kind


def test_a_sample_over_the_token_budget_is_dropped_before_the_choice(judge):
    a, b = judge(sample(PASS, 1)), judge(sample(PASS, 2))
    kept, rep = B.select([a, b], lambda j: 1000 if j.sample == 1 else 10, B.SelectConfig(max_seq=100))
    assert [k.judged.sample_id for k in kept] == [f"{PASS}#2"] and rep["dropped_over_max_seq"] == 1 and kept[0].tokens == 10


def test_one_conversation_under_two_case_ids_is_kept_once(judge):
    a = judge(sample(PASS, 1))
    twin_case = dict(judge.cases[PASS], id="zz-twin")
    judge.cases["zz-twin"] = twin_case
    b = judge(sample(PASS, 1, case_id="zz-twin", id="zz-twin#1"))
    kept, rep = B.select([a, b], tokens, B.SelectConfig())
    assert [k.judged.case_id for k in kept] == [PASS] and rep["dropped_duplicate_conversation"] == 1


def test_a_family_is_capped_second_rows_and_worst_first_and_floors_hold(judge):
    js = []
    for i in range(40):  # 40 cases of one family + 60 of another, each its own request (rule 5 keeps one per conversation)
        c = dict(judge.cases[PASS], id=f"cb-{i:02}", request=f"request {i}")
        judge.cases[c["id"]] = c
        js.append(judge(sample(PASS, 1, case_id=c["id"], id=f"{c['id']}#1", text=c["request"])))
    for i in range(60):
        c = dict(judge.cases[CLARIFY], id=f"cl-{i:02}", request=f"which block {i}")
        judge.cases[c["id"]] = c
        js.append(judge(sample(CLARIFY, 1, case_id=c["id"], id=f"{c['id']}#1", text=c["request"])))
    kept, rep = B.select(js, tokens, B.SelectConfig(family_share=0.5))
    assert rep["family_cap"]["cap"] == 50 and rep["family_cap"]["dropped_by_family"] == {"kind:clarify": 10}
    assert rep["rows"] == 90
    kept, rep = B.select(js, tokens, B.SelectConfig(family_share=0.5, floors={("kind", "clarify"): 60}))
    # every candidate row of the family is visited and held (thermometer's loop counts each), nothing is dropped
    assert rep["family_cap"]["held_by_floors"] == {"kind:clarify": 60} and rep["rows"] == 100 and rep["floors_short"] == {}
    assert rep["family_cap"]["dropped_by_family"] == {}


def test_selection_meta_is_counts_only(judge):
    k = B.Kept(judge(sample(PASS)), 1, 42)
    m = B.selection_meta(k)
    assert m == {"sample_id": f"{PASS}#1", "case_id": PASS, "rank": 1, "op_path": ["set_fields"], "browser_ops": 1,
                 "server_calls": 0, "reply_chars": m["reply_chars"], "tokens": 42, "rule_version": V.RULE_VERSION}
    assert "Fresh" not in json.dumps(m)


# --- logs -------------------------------------------------------------------------------------------------------------

def test_sample_logs_are_told_from_plain_ones():
    assert B.is_sample_log([]) is None
    assert B.is_sample_log([{"sample": 1}, {"sample": 2}]) is True
    assert B.is_sample_log([{"id": "x"}]) is False and B.is_sample_log([{"sample": 1}, {"id": "x"}]) is None
    assert B.sample_rows([{"sample": 1}, {"id": "x"}]) == [{"sample": 1}]


def test_unique_samples_drops_a_repeated_row_and_renames_a_shared_id():
    rows = [{"id": "q#1", "case_id": "q", "sample": 1, "model": "m", "status": "ok", "turn_id": "t1"},
            {"id": "q#1", "case_id": "q", "sample": 1, "model": "m", "status": "ok", "turn_id": "t1"},  # the same turn again
            {"id": "q#1", "case_id": "q", "sample": 1, "model": "m", "status": "ok", "turn_id": "t2"},  # another run's #1
            {"id": "q#2", "case_id": "q", "sample": 2, "model": "m", "status": "error", "turn_id": ""}]
    out, rep = B.unique_samples(rows)
    assert [r["id"] for r in out] == ["q#1~t1", "q#1~t2", "q#2"] and rep == {"repeated_rows": 1, "renamed": 2}
