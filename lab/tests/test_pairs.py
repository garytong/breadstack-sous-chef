"""pairs.py: split_point final vs divergence, which samples may be a side,
ties, the length guard and the quotas, on the fixtures' judged samples."""

import copy
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from souschef_lab import bestofn as B  # noqa: E402
from souschef_lab import pairs as P  # noqa: E402
from souschef_lab import verify as V  # noqa: E402

FIX = LAB / "fixtures" / "traces"
RECORDED = FIX / "engine" / "recorded.json"
PASS = "set_field.contentBlock-1.title"
EXTRA = "set_enum.productCarousel-1.productCount"
WRONG = "set_range.contentBlock-1.paddingY"
INVALID = "set_field.contentBlock-1.tagline"
CLARIFY = "clarify.home.contentBlock.title"
MULTI = "multi.home.contentBlock-1-title+productCarousel-1-productCount"


def fx(name):
    return V.load_fixture(FIX / f"{name}.json")


@pytest.fixture
def judge():
    cases = {}
    for p in FIX.glob("*.json"):
        f = V.load_fixture(p)
        f["case"]["holdout"] = False
        cases[f["case"]["id"]] = f["case"]
    return B.Judge(cases, V.RecordedEngine(RECORDED))


def sample(name, k=1, **over):
    row = V.trace_from_fixture(fx(name), sample=k)
    row.update(over)
    return row


def relabel(row, case_id, k, **over):
    """A fixture's turn re-logged as sample k of another case: the row's own
    request/context go, so to_hf takes the target case's (the fixtures share
    one site state, so the ops replay the same)."""
    row = copy.deepcopy(row)
    row.pop("text", None)
    row.pop("context", None)
    row.update(case_id=case_id, id=f"{case_id}#{k}", sample=k, **over)
    return row


def multi_row(k, case_id=PASS):
    """A sample of case_id that applies the recorded two-op batch (title AND
    productCount): on the PASS case an extra_writes failure."""
    rec = json.loads(RECORDED.read_text())["runs"]["D_multi"]
    ops, out = rec["ops"], rec["output"]
    turn = f"turn-multi-{k}"
    ev = lambda seq, event, data: {"event": event, "data": data, "turnId": turn, "seq": seq}  # noqa: E731
    events = [ev(1, "text.delta", {"text": "Making both edits."})]
    for i, op in enumerate(ops):
        inp = {kk: v for kk, v in op.items() if kk != "op"}
        events.append(ev(2 + i, "tool.call", {"id": f"tc_{i + 1}", "name": op["op"], "input": inp}))
    events.append(ev(4, "turn.waiting", {"toolCallIds": ["tc_1", "tc_2"]}))
    for i, r in enumerate(out["results"]):
        events.append(ev(5 + i, "tool.result", {"toolCallId": f"tc_{i + 1}", "ok": True, "warnings": r["warnings"],
                                               "written": r["written"], "for_seq": 2 + i}))
    events += [ev(7, "text.delta", {"text": "Both edits are in."}),
               ev(8, "turn.done", {"usage": {"calls": 2}, "stop": "end_turn"})]
    return {"id": f"{case_id}#{k}", "case_id": case_id, "sample": k, "model": "default", "status": "ok", "stop": "end_turn",
            "events": events, "turn_id": turn, "meta": {}}


def with_lead(row, text):
    """The first text.delta (the tool-call turn's text: where a divergence pair is cut)."""
    row = copy.deepcopy(row)
    first = min(e["seq"] for e in row["events"] if e["event"] == "text.delta")
    for e in row["events"]:
        if e["event"] == "text.delta" and e["seq"] == first:
            e["data"]["text"] = text
    return row


def with_reply(row, text):
    row = copy.deepcopy(row)
    last = max(e["seq"] for e in row["events"] if e["event"] == "text.delta")
    for e in row["events"]:
        if e["event"] == "text.delta" and e["seq"] == last:
            e["data"]["text"] = text
    return row


def test_a_divergence_pair_is_cut_at_the_first_turn_whose_calls_differ(judge):
    """The wrong_value fixture's turn, re-logged as a second sample of the
    pass case: same prompt, another op."""
    good = judge(sample(PASS, 1))
    bad = judge(relabel(sample(WRONG), PASS, 2))
    assert good.passed and bad.verdict.verdict == "missing_writes"  # paddingY where title was asked (and extra)
    assert bad.reasons == ["not_pass"] and "extra_writes" in bad.verdict.reasons
    pairs, rep = P.build_pairs([good, bad], P.PairConfig(length_guard=0.0))
    assert rep["pairs"] == 1 and rep["by_type"] == {"divergence": 1}
    (p,) = pairs
    row = p.to_row()
    assert [m["role"] for m in row["prompt"]] == ["user"]  # no system prompt in this judge; the shared prefix is the request
    assert json.loads(row["chosen"][0]["tool_calls"][0]["function"]["arguments"])["values"] == {"title": "<p>Fresh from the oven</p>"}
    assert json.loads(row["rejected"][0]["tool_calls"][0]["function"]["arguments"])["values"] == {"paddingY": 64}
    assert row["id"] == f"{PASS}:1-2" and row["meta"]["rejected_verdict"] == "missing_writes"
    assert row["meta"]["rejected_category"] == "missing_writes" and row["meta"]["type"] == "divergence"
    assert row["meta"]["tokens"]["chosen_side"] > 0 and row["meta"]["kind"] == "set_field"


def test_a_final_pair_differs_only_in_the_closing_text(judge):
    """Two clarify samples: both ask (clarify_ok), but one answer is
    judged a fail by a tightened rule — simulate with an extra_writes-free
    failing sample: the same (no) calls, different text, failing verdict."""
    good = judge(sample(CLARIFY, 1))
    bad = judge(with_reply(sample(CLARIFY, 2), "Updated the content block."))
    assert good.passed and bad.verdict.verdict == "no_answer"
    _, rep = P.build_pairs([good, bad], P.PairConfig())
    assert rep["pairs"] == 0 and rep["prompts_no_fail"] == 1  # no_answer is INCOMPLETE: never a rejected side
    # The same cut on rows directly: split_point sees a final-turn difference.
    assert P.split_point(good.hf, bad.hf) == ("final", 1, 1)


def test_split_point_names_why_two_rows_give_no_pair(judge):
    a = judge(sample(PASS, 1))
    b = judge(sample(PASS, 2))
    assert P.split_point(a.hf, b.hf) == "same_answer"
    other_prompt = judge(sample(EXTRA, 1))
    assert P.split_point(a.hf, other_prompt.hf) == "prompt_mismatch"
    c = judge(sample(PASS, 3))
    for e in c.hf["messages"]:
        if e["role"] == "tool":
            e["content"] = e["content"].replace("true", "false", 1)
    assert P.split_point(a.hf, c.hf) == "results_differ"
    two = judge(multi_row(4))
    assert two.verdict.verdict == "extra_writes" and len(two.hf["messages"][1]["tool_calls"]) == 2
    swapped = copy.deepcopy(two.hf)
    swapped["messages"][1]["tool_calls"].reverse()
    assert P.split_point(two.hf, swapped) == "calls_reordered"
    across = copy.deepcopy(two.hf)  # the same two calls split over two turns: still no choice of ops to learn
    second = dict(across["messages"][1], tool_calls=[across["messages"][1]["tool_calls"][1]])
    across["messages"][1]["tool_calls"] = across["messages"][1]["tool_calls"][:1]
    across["messages"].insert(3, second)
    assert P.split_point(two.hf, across) == "calls_reordered"


def test_which_samples_may_be_a_side(judge):
    good = judge(sample(PASS, 1))
    assert not P.usable_rejected(good)
    extra = judge(relabel(sample(EXTRA), PASS, 2))
    assert P.usable_rejected(extra)
    broken = judge(relabel(sample(EXTRA), PASS, 3, status="error", error="superseded"))
    assert not P.usable_rejected(broken) and "error" in broken.reasons
    cut = relabel(sample(EXTRA), PASS, 4)
    cut["events"] = [e for e in cut["events"] if e["event"] != "turn.done"]
    assert not P.usable_rejected(judge(cut))  # no_answer
    holdout = judge(relabel(sample(EXTRA), "nope", 5))
    assert not P.usable_rejected(holdout)  # never material


def test_prompts_with_no_pass_or_held_out_give_none(judge):
    bad = judge(relabel(sample(EXTRA), PASS, 1))
    _, rep = P.build_pairs([bad, judge(relabel(sample(WRONG), PASS, 2))], P.PairConfig())
    assert rep["prompts_no_pass"] == 1 and rep["pairs"] == 0
    judge.cases[PASS]["holdout"] = True
    _, rep = P.build_pairs([judge(sample(PASS, 1)), judge(relabel(sample(EXTRA), PASS, 2))], P.PairConfig())
    assert rep["prompts_held_out"] == 1 and rep["pairs"] == 0


def test_each_rejected_is_used_once_at_most_two_per_prompt_and_ties_by_category(judge):
    good = [judge(sample(PASS, k)) for k in (1, 2)]
    rejected = [judge(multi_row(3)),  # extra_writes (title and productCount)
                judge(relabel(sample(INVALID), PASS, 4)),  # invalid_op
                judge(relabel(sample(WRONG), PASS, 5))]  # missing_writes (paddingY instead of title)
    assert [r.verdict.verdict for r in rejected] == ["extra_writes", "invalid_op", "missing_writes"]
    pairs, rep = P.build_pairs(good + rejected, P.PairConfig(length_guard=0.0))
    assert rep["pairs"] == 2 and rep["candidates"] == 2
    assert [p.rejected.sample for p in pairs] == [4, 5]  # by category: invalid_op, missing_writes, then extra_writes
    assert all(p.chosen.sample == 1 for p in pairs)  # the best-ranked chosen each time
    assert len({id(p.rejected) for p in pairs}) == 2
    pairs, rep = P.build_pairs(good + rejected, P.PairConfig(max_per_prompt=3, length_guard=0.0))
    assert rep["pairs"] == 3 and rep["by_category"] == {"invalid_op": 1, "missing_writes": 1, "extra_writes": 1}
    assert rep["failures_by_category"] == {"extra_writes": 1, "invalid_op": 1, "missing_writes": 1}


def test_a_side_over_max_tokens_is_dropped(judge):
    good = judge(sample(PASS, 1))
    bad = judge(relabel(sample(EXTRA), PASS, 2))
    _, rep = P.build_pairs([good, bad], P.PairConfig(max_tokens=10, length_guard=0.0))
    assert rep["dropped_over_max_tokens"] == 1 and rep["pairs"] == 0 and rep["prompts_no_pair"] == 1


def test_the_length_guard_keeps_chosen_no_longer_than_rejected_in_half_the_pairs(judge):
    good = judge(with_lead(sample(PASS, 1), "Updating the content block title now, as asked. " * 10))
    bad = judge(relabel(sample(EXTRA), PASS, 2))
    pairs, rep = P.build_pairs([good, bad], P.PairConfig())
    assert rep["pairs"] == 0 and rep["dropped_by_length_guard"] == 1  # the only pair's chosen is the longer side
    pairs, rep = P.build_pairs([good, bad], P.PairConfig(length_guard=0.0))
    assert rep["pairs"] == 1 and rep["chosen_not_longer_share"] == 0.0
    # Directly: the largest excess goes first, until the share holds.
    p1, = pairs
    p2 = copy.deepcopy(p1)
    p2.chosen_msg = dict(p2.chosen_msg, content="x")
    assert P.length_guard([p1, p2], 0.5) == ([p1, p2], 0)  # one of two already holds: nothing goes
    kept, dropped = P.length_guard([p1, p2], 1.0)
    assert len(kept) == 1 and kept[0] is p2 and dropped == 1


def test_quotas_follow_the_failing_samples_by_category(judge):
    good = judge(sample(PASS, 1))
    fails = [judge(multi_row(k)) for k in (2, 3)] + [judge(relabel(sample(INVALID), PASS, 4))]
    pairs, rep = P.build_pairs([good] + fails, P.PairConfig(max_per_prompt=3, length_guard=0.0))
    assert rep["pairs"] == 3 and rep["failures_by_category"] == {"extra_writes": 2, "invalid_op": 1}
    picked = P.quotas(pairs, Counter({"extra_writes": 2, "invalid_op": 1}), 2)
    assert Counter(p.category for p in picked) == {"invalid_op": 1, "extra_writes": 1}
    picked = P.quotas(pairs, Counter({"extra_writes": 2, "invalid_op": 1}), 1)
    assert [p.category for p in picked] == ["invalid_op"]
    assert P.quotas(pairs, Counter(), 99) == pairs
    _, rep = P.build_pairs([good] + fails, P.PairConfig(max_per_prompt=3, max_pairs=2, length_guard=0.0))
    assert rep["dropped_by_quota"] == 1 and rep["pairs"] == 2


def test_completion_chars_and_calls_of():
    m = {"role": "assistant", "content": "ok", "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "reset", "arguments": '{"keys":["a"]}'}}]}
    assert P.completion_chars(m) == 2 + 5 + len('{"keys":["a"]}')
    assert P.calls_of(m) == (("reset", '{"keys":["a"]}'),)
    assert P.calls_of({"role": "assistant", "content": "x"}) == ()
