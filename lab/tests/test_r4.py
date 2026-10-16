"""The r4 pipeline (docs/LAB.md §5b): leak.py, r4score.py (verify-3),
r4dataset.py (the student's request), r4collect.py (the collector, against
the in-process fake service), the persistent engine, the fake's stored
conversation and the follow-up render. Engine-backed tests need node + the
cookie-cutter checkout (COOKIE_CUTTER_DIR) and skip without them unless
SOUS_CHEF_LAB_REQUIRE_ENGINE=1."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

from souschef_lab import FIXTURES_DIR, render, serving
from souschef_lab import engine as E
from souschef_lab import leak as L
from souschef_lab import r4dataset as D
from souschef_lab import r4score as R

LAB = Path(__file__).resolve().parents[1]
REQUIRE = os.environ.get("SOUS_CHEF_LAB_REQUIRE_ENGINE") == "1"
EXAM_MANIFEST = FIXTURES_DIR / "exam" / "manifest-real.json"


def _need_engine():
    if not E.engine_available():
        msg = "node and the engine (COOKIE_CUTTER_DIR) are needed: " + E.unavailable_reason()
        if REQUIRE:
            pytest.fail("SOUS_CHEF_LAB_REQUIRE_ENGINE=1 but " + msg)
        pytest.skip(msg)


# ── leak.py ───────────────────────────────────────────────────────────────

def test_tokens_fold_british_spellings_and_keep_hex_and_numbers_with_their_units():
    assert L.tokens("Make the Colour GREY, centre it at #A1B2C3 and 16px") == \
        L.tokens("make the color gray center it at #a1b2c3 and 16px")
    assert "#a1b2c3" in L.tokens("#A1B2C3")
    # a unit stays on its number (verifier 1: "600px" split in two lowered the overlap with "make … wider, 600px")
    assert {"16px", "1.5rem", "50%", "300ms", "$60"} <= L.tokens("16px 1.5rem 50% 300ms $60")
    assert "16" not in L.tokens("16px") and "px" not in L.tokens("16px")
    assert L.content_tokens("change the hero headline to Fresh drops") == {"change", "hero", "headline"}


def _index(*rows):
    idx = L.ExamIndex()
    idx.entries = []
    for text, keys, hist_ctr in rows:
        idx.entries.append({"id": f"x:{len(idx.entries)}", "text": text, "keys": {L.key_of(*k) for k in keys},
                            "fkeys": {L.full_key_of(*k) for k in keys}, "hist_ctr": set(hist_ctr),
                            "tokens": L.tokens(text), "ctokens": L.content_tokens(text),
                            "cseq": [w for w in L.content_seq(text) if w in L.content_tokens(text)],
                            "values": L.literal_values(text)})
    return idx


def _set(target, key, value):
    return {"expected": {"ops": [{"op": "set_fields", "target": target, "values": {key: value}}]}}


def test_rule3_drops_an_exam_text_embedded_whole_on_the_same_key():
    hdr = {"kind": "global", "panel": "HeaderSection"}
    idx = _index(("make the search box wider", [(hdr, "", "headerSearchWidth")], []))
    case = {"request": "make the search box in the header wider, 600px", **_set(hdr, "headerSearchWidth", 600)}
    res = idx.check(case)
    assert res["rule3"] and res["rule3"]["by"] == "keys" and res["rule3"]["containment"] == 1.0
    # the same words on another key are not a copy of that exam row
    other = {"request": case["request"], **_set(hdr, "headerSearchPlaceholder", "Search")}
    assert idx.check(other)["rule3"] is None


def test_rule3_two_word_exam_texts_need_the_same_adjacent_pair():
    pd = {"kind": "page", "page": "product-details"}
    idx = _index(("hide the reviews", [(pd, "", "productDetail.showReviews")], []))
    hit = {"request": "hide the reviews section on product pages, we don't have real ones yet",
           **_set(pd, "productDetail.showReviews", False)}
    assert idx.check(hit)["rule3"]
    apart = {"request": "hide the stars and the reviews count", **_set(pd, "productDetail.showReviews", False)}
    assert idx.check(apart)["rule3"] is None


def test_rule3_an_exam_conversation_turn_with_the_value_swapped():
    hero = {"kind": "section", "page": "home", "sectionId": "heroBanner"}
    idx = _index(("change the hero headline to Fresh drops", [], [("heroBanner", "slides")]))
    case = {"request": "change the hero headline to Pre-roll Fridays",
            "expected": {"ops": [{"op": "edit_list", "target": hero, "list": "slides", "action": "update", "index": 0,
                                  "item": {"headline": "<p>Pre-roll Fridays</p>"}}]}}
    res = idx.check(case)
    assert res["rule3"] and res["rule3"]["by"] == "history_container"
    elsewhere = copy.deepcopy(case)
    elsewhere["expected"]["ops"][0]["target"] = {"kind": "section", "page": "home", "sectionId": "newsletter"}
    assert idx.check(elsewhere)["rule3"] is None


def test_filter_cases_counts_rule3():
    hdr = {"kind": "global", "panel": "HeaderSection"}
    idx = _index(("make the search box wider", [(hdr, "", "headerSearchWidth")], []))
    cases = [{"id": "a", "request": "please make the search box in our site header a lot wider, 600px", "meta": {"family": "F6"},
              **_set(hdr, "headerSearchWidth", 600)},
             {"id": "b", "request": "zzz unrelated", "meta": {"family": "F6"}, **_set(hdr, "headerSearchWidth", 600)}]
    kept, rep = L.filter_cases(cases, idx)
    assert [c["id"] for c in kept] == ["b"] and rep["by_family"]["F6"]["dropped_rule3"] == 1
    assert kept[0]["meta"]["leak"]["checker"] == L.CHECKER


def test_rule1_drops_a_british_copy_of_an_exam_text_and_keeps_a_distant_one():
    idx = L.ExamIndex()
    exam = next(e for e in idx.entries if len(e["tokens"]) >= 4)
    copy_ = {"request": exam["text"].replace("color", "colour"), "expected": {}}
    assert idx.check(copy_)["rule1"]
    far = {"request": "zzz qqq unrelated words entirely", "expected": {}}
    res = idx.check(far)
    assert not res["rule1"] and res["rule2"] is None


def test_rule1_reads_every_earlier_turn_too():
    idx = L.ExamIndex()
    exam = next(e for e in idx.entries if len(e["tokens"]) >= 4)
    case = {"request": "zzz qqq", "history": [{"user": exam["text"]}], "expected": {}}
    assert idx.check(case)["rule1"]


def test_rule2_needs_same_key_same_value_and_some_overlap():
    idx = L.ExamIndex()
    idx.entries = [{"id": "x:1", "text": "rename pre-rolls to blunts", "keys": {("global", "", "categories")},
                    "tokens": L.tokens("rename pre-rolls to blunts"), "values": L.literal_values("rename pre-rolls to blunts")}]
    case = {"request": "please call pre-rolls blunts from now on, rename to blunts",
            "expected": {"keys": [{"target": {"kind": "global"}, "key": "categories.pre-rolls.name", "list": ""}]}}
    res = idx.check(case)
    assert res["rule2"] and res["rule2"]["exam"] == "x:1" and "blunts" in res["rule2"]["values"]
    other_key = copy.deepcopy(case)
    other_key["expected"]["keys"][0]["key"] = "storeName"
    other_key["expected"]["keys"][0]["target"] = {"kind": "branding"}
    assert idx.check(other_key)["rule2"] is None


# ── r4score.py ────────────────────────────────────────────────────────────

def _row(text: str, calls=(), stop="end_turn", engine_results=()):
    ev = [{"event": "tool.call", "data": {"id": f"c{i}", "name": c["name"], "input": c.get("input") or {}}}
          for i, c in enumerate(calls)]
    ev += [{"event": "text.delta", "data": {"text": text}}, {"event": "turn.done", "data": {"stop": stop}}]
    return {"status": "ok", "stop": stop, "events": ev, "reply": text, "tool_calls": list(calls), "ops": [],
            "engine_results": list(engine_results)}


def test_preview_claim_and_where_missing_are_flags_not_verdicts():
    case = {"score": {"mode": "where"}, "kind": "unsupported",
            "expected": {"out_of_scope": {"reason": "x", "expected_where": ["domain"]}}}
    good = _row("That's set up in your domain settings.", [{"name": "report_unsupported", "input": {"reason": "r", "where": "domain"}}])
    s = R.score(case, good)
    assert s["verdict"] == "unsupported_ok" and s["usable"] and s["flags"] == []
    preview = _row("You'll see it in the live preview.", [{"name": "report_unsupported", "input": {"reason": "r", "where": "domain"}}])
    s = R.score(case, preview)
    assert s["passed"] and "preview_claim" in s["flags"] and not s["usable"]
    nowhere = _row("Contact support.", [{"name": "report_unsupported", "input": {"reason": "r"}}])
    s = R.score(case, nowhere)
    assert "where_missing" in s["flags"] and not s["usable"]


@pytest.mark.parametrize("text,hit", [
    ("You'll see it in the live preview.", True), ("Check the previews.", True), ("previewing it now", True),
    ("I previewed the change", True), ("Preview: done", True), ("Done — save when you're happy.", False),
    ("the reviews heading", False)])
def test_one_preview_rule_for_every_form_of_the_word(text, hit):
    assert R.mentions_preview(text) is hit
    row = _row(text, [{"name": "report_unsupported", "input": {"reason": "r", "where": "domain"}}])
    case = {"score": {"mode": "where"}, "kind": "unsupported",
            "expected": {"out_of_scope": {"reason": "x", "expected_where": ["domain"]}}}
    assert ("preview_claim" in R.score(case, row)["flags"]) is hit


def _list_row(ops, oks):
    return {"ops": ops, "engine_results": [{"results": [{"ok": ok} for ok in oks]}]}


def test_list_ops_check_reads_list_action_and_index():
    add0 = {"op": "edit_list", "list": "faqCategories[].items", "action": "add", "index": [0], "item": {"q": "Q", "a": "A"}}
    want = [{"list": "faqCategories[].items", "action": "add", "index": [0]}]
    assert R.check_list_ops(_list_row([add0], [True]), want) is None
    assert R.check_list_ops(_list_row([{**add0, "index": [1]}], [True]), want).startswith("list_op_missing")
    assert R.check_list_ops(_list_row([{**add0, "action": "update", "index": [0, 1]}], [True]), want)
    assert R.check_list_ops(_list_row([add0], [False]), want)  # refused = not made
    # a bare int and a one-element path are the same index
    up = {"op": "edit_list", "list": "slides", "action": "update", "index": 0, "item": {"headline": "x"}}
    assert R.check_list_ops(_list_row([up], [True]), [{"list": "slides", "action": "update", "index": [0]}]) is None


def test_where_mode_wrong_where_and_unfinished_turn():
    case = {"score": {"mode": "where"}, "kind": "unsupported",
            "expected": {"out_of_scope": {"reason": "x", "expected_where": ["domain"]}}}
    s = R.score(case, _row("Done", [{"name": "report_unsupported", "input": {"reason": "r", "where": "themes"}}]))
    assert s["verdict"] == "wrong_where"
    s = R.score(case, _row("", stop="max_tokens"))
    assert s["verdict"] == "no_answer"


def test_answer_mode():
    case = {"score": {"mode": "answer"}, "expected": {"reply_any": ["Lora", "Oswald"]}}
    assert R.score(case, _row("You can pick Oswald or Roboto."))["verdict"] == "answer_ok"
    assert R.score(case, _row("Sorry, no idea."))["verdict"] == "no_answer"


@pytest.mark.parametrize("want,before,after,ok", [
    ("up", 8, 12, True), ("up", 8, 6, False), ("down", "14px", "12px", True),
    ("darker", "#FFFFFF", "#E5E7EB", True), ("darker", "#333333", "#ffffff", False),
    ("lighter", "#000000", "#202020", True), ("darker", "rgba(0,0,0,0.5)", "rgba(0,0,0,0.65)", True)])
def test_direction(want, before, after, ok):
    row = {"engine_results": [{"changes": [{"diff": [{"path": "homeSections[0].settings.borderRadius",
                                                      "before": before, "after": after}]}]}]}
    why = R.check_direction(row, {"key": "borderRadius", "dir": want})
    assert (why is None) == ok, why


def test_direction_falls_back_to_the_recorded_current_value():
    row = {"engine_results": [{"changes": [{"diff": [{"path": "globalSettings.cardGap", "after": 10}]}]}]}
    assert R.check_direction(row, {"key": "cardGap", "dir": "down", "from": 16}) is None
    assert R.check_direction(row, {"key": "cardGap", "dir": "down"}) == "direction_unchecked"


# ── r4dataset.py: the student's request ─────────────────────────────────────

@pytest.mark.parametrize("text,want", [
    ('<context>\n{"page":"home"}\n</context>\nhello', "hello"),
    ('<context>\n{"page":"home"}\n</context>\n<candidate_fields>\nA\nB\n</candidate_fields>\nmake it red', "make it red"),
    ("<candidate_fields>\nA\n</candidate_fields>\nhi", "hi"),
    ("plain text", "plain text"),
    ('<context>\n{"page":\n"home"}\n</context>\nx', '<context>\n{"page":\n"home"}\n</context>\nx'),  # not one line: kept
    ('<context>\n{"page":"home"}\n</context>\n', '<context>\n{"page":"home"}\n</context>\n'),  # nothing after: kept
])
def test_strip_turn_blocks_is_the_services(text, want):
    assert D.strip_turn_blocks(text) == want


class _Fields:
    def __init__(self):
        self.calls = []

    def block(self, context, text, previous=None):
        self.calls.append((text, previous))
        return "<candidate_fields>\nFields this request most likely means.\n</candidate_fields>"


def _two_turn_row():
    ctx = {"page": "home", "template": "vintage"}
    u1 = render.context_block(ctx, "change the announcement", "<candidate_fields>\nold\n</candidate_fields>")
    u2 = render.context_block(ctx, "Free delivery over $60", "<candidate_fields>\nstale\n</candidate_fields>")
    conv = [
        {"seq": 1, "role": "user", "content": [{"type": "text", "text": u1}]},
        {"seq": 2, "role": "assistant", "content": [{"type": "thinking", "thinking": "hm", "signature": "s"},
                                                     {"type": "text", "text": "What should it say?"}]},
        {"seq": 3, "role": "user", "content": [{"type": "text", "text": u2}]},
        {"seq": 4, "role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "lookup_fields", "input": {"query": "announcement"}}]},
        {"seq": 5, "role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{\"fields\":[]}"}]},
        {"seq": 6, "role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "set_fields",
                                                     "input": {"target": {"kind": "global", "panel": "HeaderSection"},
                                                               "values": {"announcementBar.message": "Free delivery over $60"}}}]},
        {"seq": 7, "role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "{\"ok\":true}"}]},
        {"seq": 8, "role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]
    hist = {"text": "change the announcement", "context": ctx, "status": "ok",
            "events": [{"event": "text.delta", "data": {"text": "What should it say?"}}, {"event": "turn.done", "data": {"stop": "end_turn"}}]}
    row = {"text": "Free delivery over $60", "context": ctx, "status": "ok", "history_rows": [hist],
           "events": [{"event": "tool.call", "data": {"id": "t2", "name": "set_fields", "input": {}}},
                      {"event": "turn.waiting", "data": {}}, {"event": "text.delta", "data": {"text": "Done."}},
                      {"event": "turn.done", "data": {"stop": "end_turn"}}],
           "conversation": {"messages": conv}}
    return row


def test_student_source_strips_history_drops_thinking_and_reblocks_the_last_turn():
    row = _two_turn_row()
    f = _Fields()
    src, notes = D.student_source({}, row, f)
    assert f.calls == [("Free delivery over $60", ("change the announcement", "What should it say?"))]
    assert src[0] == {"role": "user", "content": [{"type": "text", "text": "change the announcement"}]}
    assert src[1]["content"] == [{"type": "text", "text": "What should it say?"}]  # thinking dropped
    assert src[2]["content"][0]["text"] == render.context_block(row["context"], row["text"], f.block(None, ""))
    assert notes["source"] == "conversation" and notes["history_assistant_turns"] == 1 and notes["block_drift"]
    msgs, info = D.to_hf_messages(src, "SYS")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert info["missing_tool_results"] == 0 and info["tools_called"] == ["lookup_fields", "set_fields"]


def test_student_source_refuses_a_conversation_that_is_not_the_streamed_turn():
    row = _two_turn_row()
    row["events"][0]["data"]["id"] = "other"
    with pytest.raises(D.RenderError, match="tool_calls"):
        D.student_source({}, row, _Fields())
    row = _two_turn_row()
    row["history_rows"] = []
    with pytest.raises(D.RenderError, match="user_turns"):
        D.student_source({}, row, _Fields())


def test_without_a_conversation_server_tools_and_hidden_calls_are_refused():
    row = _two_turn_row()
    row.pop("conversation")
    row["history_rows"][0]["tool_calls"] = []
    row["tool_calls"] = [{"name": "report_unsupported"}]
    with pytest.raises(D.RenderError, match="server_tool"):
        D.student_source({}, row, _Fields())
    row["tool_calls"] = []
    row["usage"] = {"calls": 3}
    row["batches"] = 1
    with pytest.raises(D.RenderError, match="hidden_calls"):
        D.student_source({}, row, _Fields())


def test_family_cap_and_section_cap():
    cases = {f"c{i}": {"meta": {"section_type": "contentBlock" if i < 60 else f"s{i}"}, "kind": "set_field"} for i in range(100)}
    rows = [{"id": f"c{i}", "meta": {"case_id": f"c{i}", "family": "F1" if i < 80 else "F3"}} for i in range(100)]
    kept, cut_s, cut_f = D.cap_rows(rows, cases, section_cap=0.1, scope={"F1"}, family_max_share=0.0)
    assert cut_s == {"section:contentBlock": 52} and len(kept) == 48
    kept, _, cut_f = D.cap_rows(rows, cases, section_cap=0, scope=set(), family_max_share=0.5)
    fam = {f: sum(1 for r in kept if r["meta"]["family"] == f) for f in ("F1", "F3")}
    assert fam["F1"] <= 0.5 * len(kept) + 1 and fam["F3"] == 20


# ── serving: a follow-up's earlier turns render as Ollama serves them ───────

def test_history_turns_render_without_the_think_block_like_ollama_serves_them():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "one"},
            {"role": "assistant", "content": "Which?"}, {"role": "user", "content": "two"},
            {"role": "assistant", "content": "Done."}]
    text = serving.render_training(msgs, None, history_turns=1)
    served = serving.render_prompt(msgs[:4], None)  # what Ollama renders for the last turn's call
    assert text.startswith(served)
    assert text[len(served):] == "Done.<|im_end|>\n"
    old = serving.render_training(msgs, None)  # the r1–r3 render: the residue on the earlier turn
    assert old != text and not old.startswith(served)


# ── the fake service stores the conversation like the service ─────────────

def test_fake_service_stores_server_tool_round_trips():
    sys.path.insert(0, str(LAB))
    from souschef_lab import collect as C
    from tests import fake_service as FS

    doc = json.loads(EXAM_MANIFEST.read_text(encoding="utf-8"))
    svc = FS.FakeService(manifest=doc, key="k")
    svc.scripts["lab-x-1"] = [[("tool.call", {"id": "u0", "name": "report_unsupported", "input": {"reason": "r", "where": "domain"}}),
                               ("text.delta", {"text": "That's in your domain settings."}),
                               ("turn.done", {"stop": "end_turn"})]]
    svc.start()
    try:
        client, _ = C.connect(svc.url, "k", EXAM_MANIFEST, [{"manifest_hash": None}], sse=False, timeout_s=10, build_id="t")
        case = {"id": "1", "manifest_hash": json.loads(json.dumps(C.manifest_hash(doc))), "context": {"page": "home"},
                "request": "connect my domain", "site_state": {}}
        row = C.run_case(case, client, lambda s, o, c: {"ok": True, "results": [], "changes": []}, route="default",
                         site_prefix="lab-x-")
        from souschef_lab.r4collect import stored_conversation
        conv = stored_conversation(client, "lab-x-1")
    finally:
        svc.stop()
    roles = [(m["role"], [b["type"] for b in m["content"]]) for m in conv["messages"]]
    assert roles == [("user", ["text"]), ("assistant", ["tool_use"]), ("user", ["tool_result"]), ("assistant", ["text"])]
    assert row["status"] == "ok"


# ── engine-backed ─────────────────────────────────────────────────────────

def test_persistent_engine_answers_like_headless_mjs():
    _need_engine()
    from souschef_lab import state as S
    from souschef_lab.manifest import load
    m = load(str(EXAM_MANIFEST))
    st = S.build_state(m, template="vintage")
    ctx = {"templateId": "vintage", "cdnUrl": "https://cdn.int.breadstack.io", "galleryFolder": "gal", "now": 1700000000000,
           "currentPage": "home"}
    reqs = [[{"op": "set_fields", "target": {"kind": "global", "panel": "ButtonsSection"}, "values": {"borderRadius": 12}}],
            [{"op": "set_fields", "target": {"kind": "section", "page": "home", "sectionId": "nope"}, "values": {"x": 1}}]]
    pe = E.PersistentEngine(manifest_path=EXAM_MANIFEST, size=1, memory_cache=False)
    he = E.Engine(manifest_path=EXAM_MANIFEST, memory_cache=False)
    try:
        assert pe.fingerprint == he.fingerprint
        for ops in reqs:
            a = pe.run(st, ops, ctx, print_state=True, use_cache=False)
            b = he.run(st, ops, ctx, print_state=True, use_cache=False)
            assert a.to_dict() == b.to_dict() and a.returncode == b.returncode
    finally:
        pe.close()


def _hand_case(m, engine):
    from souschef_lab import r4cases
    from souschef_lab import state as S
    from souschef_lab.verify import ctx_of
    st = S.build_state(m, template="vintage")
    ctx = S.context_of(st, "home", S.load_context_fixture(), m, view=None, text="square-ish buttons please, radius 4")
    ops = [{"op": "set_fields", "target": {"kind": "global", "panel": "ButtonsSection"}, "values": {"borderRadius": 4}}]
    case = {"id": "hand-1", "kind": "set_range", "target": ops[0]["target"], "fields": ["borderRadius"], "site_state": st,
            "context": ctx, "request": "square-ish buttons please, radius 4", "holdout": False, "manifest_hash": m.hash,
            "template": "vintage", "expected": {"ops": ops, "diff": []},
            "meta": {"family": "F1", "scoring": "exact", "source_case": "x"}}
    case["expected"]["diff"] = engine.run(st, ops, ctx_of(case)).diff()
    n = r4cases.normalise(case, "F1", "test")
    n["id"] = "r4-F1-9001"
    return n


def test_r4collect_oracle_passes_and_the_default_script_does_not():
    _need_engine()
    sys.path.insert(0, str(LAB))
    from souschef_lab import collect as C
    from souschef_lab import r4collect as RC
    from souschef_lab.manifest import load
    from souschef_lab.state import load_context_fixture
    from tests import fake_service as FS

    m = load(str(EXAM_MANIFEST))
    eng = E.PersistentEngine(manifest_path=EXAM_MANIFEST, size=1)
    try:
        single = _hand_case(m, eng)
        multi = copy.deepcopy(single)
        multi["id"] = "r4-F2-9002"
        multi["history"] = [{"user": "the buttons look too round", "context": None, "page": "home", "expect": "ask",
                             "assistant": "How square do you want them?", "ops": [], "calls": []}]
        multi["request"] = "radius 4"
        doc = json.loads(EXAM_MANIFEST.read_text(encoding="utf-8"))
        for oracle, want in ((True, True), (False, False)):
            svc = FS.FakeService(manifest=doc, key="k")
            if oracle:
                svc.scripts.update(RC.oracle_scripts([single, multi], "lab-t-"))
            svc.start()
            try:
                client, _ = C.connect(svc.url, "k", EXAM_MANIFEST, [single], sse=False, timeout_s=30, build_id="t")
                for case in (single, multi):
                    row = RC.run_r4_case(case, client, eng, route="default", fields=None, manifest=m,
                                         extras=load_context_fixture(), assets=[], site_prefix="lab-t-")
                    assert row["score"]["usable"] is want, (case["id"], row["score"])
                    if oracle:
                        assert row["conversation"]["messages"]
                        assert len(row["history_rows"]) == len(case["history"])
                        if case["history"]:
                            assert "exact_remeasured_on_live_state" in row["score"]["info"]
            finally:
                svc.stop()
    finally:
        eng.close()


def test_r4collect_excuses_acting_on_a_vague_first_turn(monkeypatch):
    """r6: r4score.excuse_divergences is wired into run_r4_case — an `ask` turn the teacher answered by writing the
    case's own key (written again by the scored turn) is excused when the scored turn passes and the relaxation is
    on; with it off (the default since the r6 review) it still diverges."""
    _need_engine()
    sys.path.insert(0, str(LAB))
    from souschef_lab import collect as C
    from souschef_lab import r4collect as RC
    from souschef_lab.manifest import load
    from souschef_lab.state import load_context_fixture
    from tests import fake_service as FS

    m = load(str(EXAM_MANIFEST))
    eng = E.PersistentEngine(manifest_path=EXAM_MANIFEST, size=1)
    try:
        case = copy.deepcopy(_hand_case(m, eng))
        case["id"] = "r4-F2-9003"
        acted = [{"op": "set_fields", "target": case["target"], "values": {"borderRadius": 6}}]
        case["history"] = [{"user": "the buttons look too round", "context": None, "page": "home", "expect": "ask",
                            "assistant": "I squared them a bit.", "ops": acted, "calls": []}]
        case["request"] = "radius 4"
        script = copy.deepcopy(case)
        script["history"][0]["expect"] = "write"  # the teacher acts where the case expected a question
        doc = json.loads(EXAM_MANIFEST.read_text(encoding="utf-8"))
        for policy, usable in (('{"acted_first_turn": "keys"}', True), (None, False)):
            if policy:
                monkeypatch.setenv("SOUSCHEF_SCORE_POLICY", policy)
            else:
                monkeypatch.delenv("SOUSCHEF_SCORE_POLICY", raising=False)
            svc = FS.FakeService(manifest=doc, key="k")
            svc.scripts.update(RC.oracle_scripts([script], "lab-t-"))
            svc.start()
            try:
                client, _ = C.connect(svc.url, "k", EXAM_MANIFEST, [case], sse=False, timeout_s=30, build_id="t")
                row = RC.run_r4_case(case, client, eng, route="default", fields=None, manifest=m,
                                     extras=load_context_fixture(), assets=[], site_prefix="lab-t-")
            finally:
                svc.stop()
            sc = row["score"]
            assert sc["usable"] is usable, sc
            if usable:
                assert sc["history_excused"][0]["excused"] == "acted_first_turn:keys" and "history_diverged" not in sc
            else:
                assert sc["history_diverged"][0]["why"] == "did_not_ask"
    finally:
        eng.close()


# ── r4cases revisions (the verifier round) ──────────────────────────────────

def test_authored_values_find_what_the_merchant_never_typed():
    from souschef_lab import r4cases as RC
    faq = {"op": "edit_list", "target": {"kind": "page", "page": "faq"}, "list": "faqCategories[].items", "action": "add",
           "index": [0], "item": {"q": "Do you take cash at pickup?", "a": "Yes, exact change appreciated."}}
    c = {"request": "people ask if we take cash at pickup, add it to the faq, answer yes, exact change appreciated",
         "history": [], "expected": {"ops": [faq]}}
    assert RC.authored_values(c) == ["invented", "punctuation_or_case"]
    enum = {"request": "switch the top banner to the centred layout", "history": [],
            "expected": {"ops": [{"op": "set_fields", "target": {}, "values": {"layout": "centered"}}]}}
    assert RC.authored_values(enum) == []  # a named one-word value, not authored text
    proposal = {"request": "yeah do that", "history": [{"user": "it looks sterile", "assistant": "Try #f6f1e7?"}],
                "expected": {"ops": [{"op": "set_fields", "target": {}, "values": {"siteBgColor": "#f6f1e7"}}]}}
    assert RC.authored_values(proposal) == ["from_assistant"]
    said = {"request": "two, they're too small", "history": [{"user": "cards", "assistant": "2 or 3 per row?"}],
            "expected": {"ops": [{"op": "set_fields", "target": {}, "values": {"cardsPerRow": 2}}]}}
    assert RC.authored_values(said) == []


def test_to_keys_keeps_the_rows_touched_and_drops_the_wording():
    from souschef_lab import r4cases as RC
    faq = {"op": "edit_list", "target": {"kind": "page", "page": "faq"}, "list": "faqCategories[].items", "action": "add",
           "index": [0], "item": {"q": "Q?", "a": "A."}}
    c = {"expected": {"ops": [faq], "diff": [{"path": "x"}], "keys": [], "alt": [], "alt_exact": [{"ops": []}]},
         "meta": {}, "score": {"mode": "exact"}}
    RC.to_keys(c, ["invented"])
    assert c["score"] == {"mode": "keys"} and c["expected"]["alt_exact"] == []
    assert c["expected"]["keys"] == [{"target": {"kind": "page", "page": "faq"}, "key": "", "list": "faqCategories[].items"}]
    assert c["expected"]["list_ops"] == [{"list": "faqCategories[].items", "action": "add", "index": [0]}]
    assert c["meta"]["oracle_ops"] == [faq]


def test_humanize_replaces_probe_copy_and_build_state_has_none():
    from souschef_lab import state as S
    from souschef_lab.manifest import load
    m = load(str(EXAM_MANIFEST))
    for t in S.BASIC_TEMPLATE_IDS:
        st = S.build_state(m, template=t)
        assert "Zq Fixture" not in json.dumps(st), t
    st = {"homeSections": [{"id": "contentBlock-2", "settings": {"items": [{"title": "Zq Fixture Column One", "body": "b"}]}}],
          "pageVisibility": {"blog": {"posts": [{"title": "Zq Fixture Post One"}]}},
          "globalSettings": {"categories": {"zq-cat-1": {"name": "Zq Fixture Category One"}}}}
    once = S.humanize(copy.deepcopy(st))
    assert once["homeSections"][0]["settings"]["items"][0] == {"title": "Founded in 2019", "body": "b"}
    assert "zq-cat-1" in once["globalSettings"]["categories"]  # ids untouched
    assert S.humanize(copy.deepcopy(once)) == once  # idempotent


def test_tag_seen_marks_exam_keys_a_training_case_writes():
    from souschef_lab import r4cases as RC
    from souschef_lab.manifest import load
    m = load(str(EXAM_MANIFEST))
    tg = {"kind": "global", "panel": "TypographyColorsSection"}
    train = [{"expected": {"ops": [{"op": "set_fields", "target": tg, "values": {"headingColor": "#111111"}}]}, "meta": {}}]
    seen = {"kind": "set_field", "expected": {"ops": [{"op": "set_fields", "target": tg, "values": {"headingColor": "#222"}}]},
            "meta": {"holdout_key": "TypographyColorsSection.headingColor"}, "score": {"mode": "exact"}}
    unseen = {"kind": "set_field", "expected": {"ops": [{"op": "set_fields", "target": tg, "values": {"bodyColor": "#222"}}]},
              "meta": {"holdout_key": "TypographyColorsSection.bodyColor"}, "score": {"mode": "exact"}}
    rep = RC.tag_seen([seen, unseen], train, m)
    assert rep == {"cases": 2, "seen": 1, "unseen": 1}
    assert seen["meta"]["key_seen"] and seen["meta"]["seen_keys"] == ["TypographyColorsSection.headingColor"]
    assert unseen["meta"]["key_seen"] is False


def test_smoke_check_never_passes_on_nothing_rendered():
    sys.path.insert(0, str(LAB / "tools"))
    import r4_smoke_check as SC
    ok = {"rows": 2, "ok_rows": 2, "rendered": 2, "render_sources": {"conversation": 2}, "block_drift_rows": 0,
          "errors": {}, "by_case": [{"case_id": "a"}, {"case_id": "b"}]}
    assert SC.problems_of(ok, {"a", "b"}) == []
    empty = {"rows": 0, "ok_rows": 0, "rendered": 0, "render_sources": {}, "block_drift_rows": 0, "errors": {}, "by_case": []}
    assert SC.problems_of(empty, {"a", "b"})
    drift = {**ok, "block_drift_rows": 1}
    assert any("drift" in p for p in SC.problems_of(drift, {"a", "b"}))
    events = {**ok, "render_sources": {"conversation": 1, "events": 1}}
    assert SC.problems_of(events, {"a", "b"})
    failed_verdicts_still_render = {**ok, "by_case": [{"case_id": "a", "verdict": "not_pass"}, {"case_id": "b"}]}
    assert SC.problems_of(failed_verdicts_still_render, {"a", "b"}) == []
