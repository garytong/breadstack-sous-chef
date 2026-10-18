"""retrain/author.py: cases for new or changed fields through the Anthropic API
(design §3.2 step 2). The API is faked: no network, no key."""
from __future__ import annotations

import json
import sys
import types

import pytest

from souschef_lab import engine as E
from souschef_lab.manifest import Manifest, holdout_bucket
from souschef_lab.retrain import author as A

from tests.test_rebase import BTN, _doc, _need_engine

NEW_ADDR = "globalPanels.ButtonsSection.buttonShadowColor"


def grown() -> dict:
    """The exam manifest plus one new colour field on the buttons panel."""
    d = _doc()
    d["globalPanels"]["ButtonsSection"]["fields"]["buttonShadowColor"] = {
        "type": "color", "label": "Button Shadow Color", "path": "globalSettings.buttonShadowColor",
        "help": "The soft shadow under every button."}
    return d


def test_find_targets_new_fields_coverage_and_holdout():
    old, new = _doc(), Manifest(grown())
    targets, rep = A.find_targets(old, new, pool=[])
    assert [t["address"] for t in targets] == [NEW_ADDR]
    t = targets[0]
    assert t["kind"] == "field" and t["reason"] == "added_field"
    assert t["holdout"] == (holdout_bucket(NEW_ADDR) < A.HOLDOUT_PCT)
    # enough pool cases already writing it: nothing to author (skipped rounds accumulate)
    op = {"op": "set_fields", "target": BTN, "values": {"buttonShadowColor": "#000000"}}
    pool = [{"id": f"p{i}", "holdout": False, "expected": {"ops": [op]}, "meta": {}} for i in range(A.COVERAGE_ENOUGH)]
    targets, rep = A.find_targets(old, new, pool=pool)
    assert targets == [] and rep["covered_enough"] == 1
    # a key whose pool cases the rebase dropped (R2) is a target too
    targets, _ = A.find_targets(old, Manifest(old), pool=[], dropped_writes={"globalPanels.ButtonsSection.borderRadius"})
    assert [(t["address"], t["reason"]) for t in targets] == [("globalPanels.ButtonsSection.borderRadius", "cases_dropped")]


def test_holdout_targets_count_earlier_holdouts_and_retyped_fields_are_targets():
    """A skipped round's F9H cases cover a holdout target (it is not re-authored and re-paid every round), and
    a retyped field is a target like an added one (verifier 2, should-change 5 and 8)."""
    old, new = _doc(), Manifest(grown())
    op = {"op": "set_fields", "target": BTN, "values": {"buttonShadowColor": "#000000"}}
    held = [{"id": f"h{i}", "holdout": True, "expected": {"ops": [op]}, "meta": {"family": "F9H"}} for i in range(A.COVERAGE_ENOUGH)]
    targets, rep = A.find_targets(old, new, pool=held)
    if holdout_bucket(NEW_ADDR) < A.HOLDOUT_PCT:
        assert targets == [] and rep["covered_enough"] == 1
    else:  # a trainable target is never covered by holdouts
        assert [t["address"] for t in targets] == [NEW_ADDR]
    assert A.coverage(held, new)[NEW_ADDR] == 0 and A.coverage(held, new, holdout=True)[NEW_ADDR] == A.COVERAGE_ENOUGH
    retyped = _doc()
    f = retyped["globalPanels"]["ButtonsSection"]["fields"]["borderRadius"]
    f["type"] = "text" if f.get("type") != "text" else "number"
    targets, _ = A.find_targets(old, Manifest(retyped), pool=[])
    assert ("globalPanels.ButtonsSection.borderRadius", "retyped_field") in [(t["address"], t["reason"]) for t in targets]


def test_the_prompt_carries_the_entry_aliases_and_no_exam_text():
    from souschef_lab.leak import ExamIndex
    m = Manifest(grown())
    t = {"address": NEW_ADDR, "kind": "field", "reason": "added_field", "holdout": False}
    entry = A.target_entry(t, m)
    assert entry["op_target"] == BTN and entry["key"] == "buttonShadowColor"
    assert entry["field"]["label"] == "Button Shadow Color" and entry["global_section"] == "buttons"
    prompt = A.user_prompt(entry, A.style_plan(t, entry=entry), ["drop shadow"], [])
    assert "buttonShadowColor" in prompt and "drop shadow" in prompt
    # no reset (the colour has no unset / default value) and no gated styles (no gate)
    assert "verbatim, verbatim, colloquial, colloquial, short, relative, focus, followup, near_miss." in prompt
    assert "visibility" not in entry and "reset" not in entry
    for e in ExamIndex().entries:
        if len(e["tokens"]) >= 5:
            assert e["text"] not in prompt and e["text"] not in A.RULES


def test_case_schema_is_strict():
    def walk(s):
        if s.get("type") == "object":
            assert s.get("additionalProperties") is False and set(s["required"]) == set(s["properties"])
            for v in s["properties"].values():
                walk(v)
        if s.get("type") == "array":
            walk(s["items"])
    walk(A.CASE_SCHEMA)


class _Stream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


def _fake_sdk(monkeypatch, msg, calls):
    mod = types.ModuleType("anthropic")

    class Err(Exception):
        pass

    mod.RateLimitError = mod.InternalServerError = mod.APIConnectionError = Err

    class Client:
        def __init__(self, **kw):
            calls.append(("init", kw))
            self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(stream=self.stream))

        def stream(self, **kw):
            calls.append(("stream", kw))
            return _Stream(msg)

    mod.Anthropic = Client
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")


def test_anthropic_ask_sends_the_designed_request(monkeypatch):
    usage = types.SimpleNamespace(input_tokens=1000, output_tokens=500, cache_creation_input_tokens=2000,
                                  cache_read_input_tokens=0, iterations=None)
    msg = types.SimpleNamespace(stop_reason="end_turn", usage=usage,
                                content=[types.SimpleNamespace(type="thinking", thinking=""),
                                         types.SimpleNamespace(type="text", text='{"cases": []}')])
    calls: list = []
    _fake_sdk(monkeypatch, msg, calls)
    ask = A.anthropic_ask()
    obj, u = ask("RULES", "USER", A.CASE_SCHEMA)
    assert obj == {"cases": []} and u["output_tokens"] == 500
    kw = next(k for n, k in calls if n == "stream")
    assert kw["model"] == "claude-opus-5-5" and kw["max_tokens"] == 32000
    assert kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["extra_body"] == {"fallbacks": "default"}
    assert kw["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": A.CASE_SCHEMA}}
    assert kw["system"] == [{"type": "text", "text": "RULES", "cache_control": {"type": "ephemeral"}}]
    assert kw["messages"] == [{"role": "user", "content": "USER"}]
    assert "thinking" not in kw  # Opus 5.5: thinking is always on; effort is the control
    assert A.usage_cost(u) == pytest.approx((4 * 1000 + 20 * 500 + 5 * 2000) / 1e6)


def test_anthropic_ask_raises_on_a_refusal(monkeypatch):
    msg = types.SimpleNamespace(stop_reason="refusal", usage=None, content=[])
    _fake_sdk(monkeypatch, msg, [])
    with pytest.raises(A.AuthorRefused):
        A.anthropic_ask()("R", "U", A.CASE_SCHEMA)


def test_anthropic_ask_needs_the_key(monkeypatch):
    _fake_sdk(monkeypatch, None, [])
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(A.StageError):
        A.anthropic_ask()


@pytest.fixture(scope="module")
def grown_engine(tmp_path_factory):
    _need_engine()
    p = tmp_path_factory.mktemp("m") / "manifest.json"
    p.write_text(json.dumps(grown()), encoding="utf-8")
    eng = E.PersistentEngine(manifest_path=p, size=2)
    yield eng, Manifest(grown())
    eng.close()


def test_author_round_validates_measures_and_reasks(grown_engine):
    from souschef_lab.state import load_context_fixture
    eng, m = grown_engine
    t = {"address": NEW_ADDR, "kind": "field", "reason": "added_field", "holdout": False}
    fake = A.FakeAsk(m, break_first=2)
    train, hold, rep = A.author_round([t], m, eng, load_context_fixture(), ask=fake, round_id="rt-20261012-0310-abcdef12",
                                      pool=[], leak_index=None, progress=lambda s: None)
    assert hold == []
    assert len(train) == 9, rep["per_target"]
    assert rep["rejects"].get("retried:keys") == 2  # the two broken cases came back right on the re-ask
    pt = rep["per_target"][0]
    assert pt["calls"] == 2 and pt["asked"] == 11 and pt["topup"] == 0
    assert pt["positive"] == 8 and rep["short_of_positive"] == {} and rep["positive_by_target"] == {NEW_ADDR: 8}
    # the full reason of every rejected draft is kept (r5 kept only its prefix)
    assert [d["reason"] for d in pt["reject_details"]] == ["keys:key_removed"] * 2
    styles = sorted(c["meta"]["style"] for c in train)
    assert styles == sorted(["verbatim", "verbatim", "colloquial", "colloquial", "short", "relative", "focus", "followup",
                             "near_miss"])
    assert sum(1 for c in train if c["meta"].get("positive")) == 8
    for c in train:
        assert c["meta"]["family"] == "F7" and not c["holdout"] and c["manifest_hash"] == m.hash
        assert c["engine_fingerprint"] == eng.fingerprint and c["context"].get("outline") is not None
        if c["score"]["mode"] == "exact":
            assert c["expected"]["diff"], c["id"]
            assert NEW_ADDR in A.case_addresses(c, m)
    fu = next(c for c in train if c["meta"]["style"] == "followup")
    assert fu["history"] and fu["history"][0]["expect"] == "ask" and fu["history"][0]["context"]
    nm = next(c for c in train if c["meta"]["style"] == "near_miss")
    assert nm["score"]["mode"] == "where" and nm["expected"]["out_of_scope"]["expected_where"] == ["domain"]
    focus = next(c for c in train if c["meta"]["style"] == "focus")
    assert focus["context"]["focus"]["area"] == "global" and focus["context"]["focus"]["panel"] == "ButtonsSection"
    assert len({c["id"] for c in train}) == 9
    assert rep["anthropic_usd"] > 0
    # a holdout target's cases are the new-key exam F9H
    th = dict(t, holdout=True)
    train2, hold2, _ = A.author_round([th], m, eng, load_context_fixture(), ask=A.FakeAsk(m), round_id="rt-x2",
                                      pool=[], leak_index=None, progress=lambda s: None)
    assert train2 == [] and hold2 and all(c["meta"]["family"] == "F9H" and c["holdout"] for c in hold2)


def test_validate_rejects_preview_leak_and_duplicates(grown_engine):
    from souschef_lab.state import load_context_fixture
    from tests.test_r4 import _index
    eng, m = grown_engine
    t = {"address": NEW_ADDR, "kind": "field", "reason": "added_field", "holdout": False}
    entry = A.target_entry(t, m)
    raw = {"style": "colloquial", "request": "button shadow colour #222222 please", "history": [], "page": "home",
           "view": {"tab": "", "activeSectionId": "", "globalSection": ""},
           "oracle_ops_json": json.dumps([{"op": "set_fields", "target": BTN, "values": {"buttonShadowColor": "#222222"}}]),
           "score_mode": "exact", "expected_where": [], "candidates": [], "reply_any": [], "direction": {"key": "", "dir": ""}}

    def mk(**kw):
        return A.to_case({**raw, **kw}, t, entry, m, case_id="x-1", round_id="rt-1", holdout=False)

    keys: set = set()
    extras = load_context_fixture()
    assert A.validate(mk(), t, m, eng, extras, leak_index=None, pool_keys=keys) is None
    assert A.validate(mk(), t, m, eng, extras, leak_index=None, pool_keys=keys) == "duplicate"
    assert A.validate(mk(request="check the preview, button shadow #222222"), t, m, eng, extras, leak_index=None,
                      pool_keys=set()) == "preview_claim"
    other = json.dumps([{"op": "set_fields", "target": BTN, "values": {"buttonColor": "#222222"}}])
    assert A.validate(mk(oracle_ops_json=other), t, m, eng, extras, leak_index=None, pool_keys=set()) == "misses_target"
    idx = _index(("button shadow colour #222222 please", [(BTN, "", "buttonShadowColor")], []))
    assert A.validate(mk(), t, m, eng, extras, leak_index=idx, pool_keys=set()).startswith("leak:")
    rel = mk(request="make the button shadow a bit darker", score_mode="keys",
             direction={"key": "buttonShadowColor", "dir": "darker"})
    assert A.validate(rel, t, m, eng, extras, leak_index=None, pool_keys=set()) is None
    assert rel["score"]["mode"] == "keys" and rel["expected"]["keys"][0]["key"] == "buttonShadowColor"
    assert rel["expected"]["direction"]["dir"] == "darker"


# --- r6: real section ids, gated fields, hidden writes, full oracle text, positives -----------------------------------

CB_GAP = "sections.contentBlock.stackedImageGap"
GAL_LINK = "sections.gallery.imagesLink[].value"


def gated() -> dict:
    """The exam manifest plus r5's new fields as the 1184ff9a manifest has them: contentBlock
    stackedImageGap / Height (conditional_on imagePosition = top, itself conditional_on style ∈
    {image-text, img-left}) and the gallery's imagesLink list (no list ops, a scalar item)."""
    d = _doc()
    cb = d["sections"]["contentBlock"]["fields"]
    cb["imagePosition"] = {"type": "enum", "label": "Image Position",
                           "enum": ["right", "left", "top", "bottom"], "path": "settings.imagePosition",
                           "conditional_on": {"key": "style", "value": ["image-text", "img-left"]}}
    for k, label in (("stackedImageGap", "Image Gap"), ("stackedImageHeight", "Image Height")):
        cb[k] = {"type": "range", "label": label, "min": 0, "max": 120, "step": 1, "path": f"settings.{k}",
                 "conditional_on": {"key": "imagePosition", "value": "top"}}
    d["sections"]["gallery"].setdefault("lists", {})["imagesLink"] = {
        "ops": [], "path": "settings.imagesLink",
        "item": {"value": {"path": "settings.imagesLink[]", "type": "text", "label": "Image {n} link"}}}
    return d


def _t(addr, kind="field"):
    return {"address": addr, "kind": kind, "reason": "added_field", "holdout": False}


def test_op_targets_name_a_real_section_and_gates_are_described():
    """r5: op_target_for gave sectionId "contentBlock" (the TYPE); no fixture home holds it, so every
    contentBlock draft failed the oracle and the round kept 0 positive cases for the key."""
    m = Manifest(gated())
    e = A.target_entry(_t(CB_GAP), m)
    assert e["op_target"] == {"kind": "section", "page": "home", "sectionId": "contentBlock-1"}
    gates = e["visibility"]["gates"]
    assert [(g["key"], g["open_value"]) for g in gates] == [("style", "image-text"), ("imagePosition", "top")]
    assert A.style_plan(_t(CB_GAP), entry=e)[-3:] == ["prereq", "prereq", "gate_only"]
    # a type the fixture home lacks keeps the type as its id (to_case adds the row)
    assert A.op_target_for("sections", "noSuchType", m)["sectionId"] == "noSuchType"
    # without a manifest: the first fixture instance of the type, never the bare type
    assert A.op_target_for("sections", "contentBlock")["sectionId"] == "contentBlock-1"
    g = A.target_entry(_t(GAL_LINK), m)
    assert g["op_target"]["sectionId"] == "gallery" and "visibility" not in g


def test_gate_matches_follows_changes_js():
    assert A.gate_matches("set", "x") and not A.gate_matches("set", "") and not A.gate_matches("set", False)
    assert A.gate_matches(["a", "b"], "b") and not A.gate_matches(["a"], None) and A.gate_matches(3, "3")
    assert not A.gate_matches(1, True) and A.gate_matches(True, "true")
    assert A.hidden_by("conditional_on", "top", None) and not A.hidden_by("hidden_when", "top", None)
    # both rules on one key (ShopMenu shopMenuActiveColor): the open value satisfies both
    g = {"rules": [{"rule": "conditional_on", "value": "modern"}, {"rule": "hidden_when", "value": "location"}]}
    assert A.gate_open(g, "modern") and not A.gate_open(g, "classic") and not A.gate_open(g, "location")


def _raw(style, ops, request="make the image gap 24 on the Crafted with Care block", mode="exact"):
    return {"style": style, "request": request, "history": [], "page": "home",
            "view": {"tab": "", "activeSectionId": "", "globalSection": ""}, "oracle_ops_json": json.dumps(ops),
            "score_mode": mode, "expected_where": [], "candidates": [], "reply_any": [], "direction": {"key": "", "dir": ""}}


def test_case_state_opens_gates_except_for_prereq_and_hidden_writes_are_seen():
    m = Manifest(gated())
    t = _t(CB_GAP)
    e = A.target_entry(t, m)
    tgt = e["op_target"]
    gap = [{"op": "set_fields", "target": tgt, "values": {"stackedImageGap": 24}}]
    both = [{"op": "set_fields", "target": tgt, "values": {"imagePosition": "top", "stackedImageGap": 24}}]
    v = A.to_case(_raw("verbatim", gap), t, e, m, case_id="x-1", round_id="rt-1", holdout=False)
    assert v["meta"]["state_prep"]["opened"] == {"imagePosition": "top"}  # contentBlock-1 already has style image-text
    assert A.hidden_writes(v, m) == []
    p = A.to_case(_raw("prereq", gap), t, e, m, case_id="x-2", round_id="rt-1", holdout=False)
    assert "imagePosition" in p["meta"]["state_prep"]["closed"]
    hw = A.hidden_writes(p, m)
    assert len(hw) == 1 and hw[0].startswith("stackedImageGap is hidden while imagePosition")
    p2 = A.to_case(_raw("prereq", both), t, e, m, case_id="x-3", round_id="rt-1", holdout=False)
    assert A.hidden_writes(p2, m) == []  # the gate opened in the same set_fields, first
    p["expected"]["expect_hidden"] = ["stackedImageGap"]  # a deliberate hidden-write case is not flagged
    assert A.hidden_writes(p, m) == []
    # a case that hides the write covers nothing (r5's 22 perks rows counted as coverage)
    p["expected"]["expect_hidden"] = None
    assert A.coverage([p], m)[CB_GAP] == 0 and A.coverage([p2], m)[CB_GAP] == 1
    assert A.coverage([p], m, exclude_hidden=False)[CB_GAP] == 1


def test_prepare_drops_trainable_pool_cases_that_write_a_hidden_field():
    from souschef_lab.retrain import prepare as PREP
    m = Manifest(gated())
    t = _t(CB_GAP)
    e = A.target_entry(t, m)
    gap = [{"op": "set_fields", "target": e["op_target"], "values": {"stackedImageGap": 24}}]
    v = A.to_case(_raw("verbatim", gap), t, e, m, case_id="x-1", round_id="rt-1", holdout=False)
    p = A.to_case(_raw("prereq", gap), t, e, m, case_id="x-2", round_id="rt-1", holdout=False)
    h = A.to_case(_raw("prereq", gap), t, e, m, case_id="x-3", round_id="rt-1", holdout=True)
    kept, rep = PREP.drop_hidden_writes([v, p, h], m)
    assert [c["id"] for c in kept] == ["x-1", "x-3"]  # the exam's holdouts are never dropped
    assert rep["count"] == 1 and rep["ids"] == ["x-2"] and rep["by_family"] == {"F7": 1}
    assert PREP.switch({"trigger": {"pool_hidden": "Keep"}}, "SOUSCHEF_POOL_HIDDEN", "pool_hidden", "drop", env={}) == "keep"
    assert PREP.switch({"trigger": {"pool_hidden": "keep"}}, "SOUSCHEF_POOL_HIDDEN", "pool_hidden", "drop",
                       env={"SOUSCHEF_POOL_HIDDEN": "drop"}) == "drop"
    assert PREP.switch({}, "SOUSCHEF_CONTRAST", "contrast", "on", env={}) == "on"


def test_find_targets_reports_every_wanted_address():
    old, new = _doc(), Manifest(grown())
    _, rep = A.find_targets(old, new, pool=[])
    assert [w["address"] for w in rep["wanted_list"]] == [NEW_ADDR]
    assert rep["wanted_list"][0]["reason"] == "added_field" and rep["wanted_list"][0]["covered"] == 0


@pytest.fixture(scope="module")
def gated_engine(tmp_path_factory):
    _need_engine()
    p = tmp_path_factory.mktemp("g") / "manifest.json"
    p.write_text(json.dumps(gated()), encoding="utf-8")
    eng = E.PersistentEngine(manifest_path=p, size=2)
    yield eng, Manifest(gated())
    eng.close()


def test_validate_rejects_hidden_writes_and_keeps_the_oracle_text(gated_engine):
    from souschef_lab.state import load_context_fixture
    eng, m = gated_engine
    extras = load_context_fixture()
    t = _t(CB_GAP)
    e = A.target_entry(t, m)
    gap = [{"op": "set_fields", "target": e["op_target"], "values": {"stackedImageGap": 24}}]
    p = A.to_case(_raw("prereq", gap), t, e, m, case_id="x-1", round_id="rt-1", holdout=False)
    why = A.validate(p, t, m, eng, extras, leak_index=None, pool_keys=set())
    assert why.startswith("hidden_write: stackedImageGap is hidden while imagePosition") and "same set_fields" in why
    ok = A.to_case(_raw("prereq", [{"op": "set_fields", "target": e["op_target"],
                                    "values": {"imagePosition": "top", "stackedImageGap": 24}}]),
                   t, e, m, case_id="x-2", round_id="rt-1", holdout=False)
    assert A.validate(ok, t, m, eng, extras, leak_index=None, pool_keys=set()) is None and ok["meta"]["positive"]
    # r5's bug: the section TYPE as the id; the engine's own words now reach the report and the re-ask
    bad = A.to_case(_raw("verbatim", [{"op": "set_fields", "target": {**e["op_target"], "sectionId": "contentBlock"},
                                       "values": {"stackedImageGap": 24}}], request="image gap 30 on Crafted with Care please"),
                    t, e, m, case_id="x-3", round_id="rt-1", holdout=False)
    why = A.validate(bad, t, m, eng, extras, leak_index=None, pool_keys=set())
    assert why.startswith("oracle:exact_refused: E_TARGET") and "contentBlock" in why
    # gate_only: must set the gate, and is not a positive write of the target
    go = A.to_case(_raw("gate_only", [{"op": "set_fields", "target": e["op_target"], "values": {"imagePosition": "top"}}],
                        request="put the pictures above the text in the Crafted with Care section"), t, e, m, case_id="x-4", round_id="rt-1", holdout=False)
    assert A.validate(go, t, m, eng, extras, leak_index=None, pool_keys=set()) is None and go["meta"]["positive"] is False
    go2 = A.to_case(_raw("gate_only", [{"op": "set_fields", "target": e["op_target"], "values": {"title": "Hi"}}],
                         request="put the pictures on top"), t, e, m, case_id="x-5", round_id="rt-1", holdout=False)
    assert A.validate(go2, t, m, eng, extras, leak_index=None, pool_keys=set()).startswith("misses_gate")


def test_author_round_gated_field_and_list_reach_the_positive_floor(gated_engine):
    from souschef_lab.state import load_context_fixture
    eng, m = gated_engine
    ts = [_t(CB_GAP), _t(GAL_LINK)]
    train, hold, rep = A.author_round(ts, m, eng, load_context_fixture(), ask=A.FakeAsk(m), round_id="rt-g1", pool=[],
                                      leak_index=None, progress=lambda s: None)
    assert rep["short_of_positive"] == {}, rep["per_target"]
    assert rep["positive_by_target"][CB_GAP] >= A.MIN_POSITIVE and rep["positive_by_target"][GAL_LINK] >= A.MIN_POSITIVE
    cb = [c for c in train if c["meta"]["target_address"] == CB_GAP]
    assert {c["target"]["sectionId"] for c in cb} == {"contentBlock-1"}
    pre = [c for c in cb if c["meta"]["style"] == "prereq"]
    assert pre and all(c["expected"]["ops"][0]["values"].get("imagePosition") == "top" for c in pre)
    assert any(c["meta"]["style"] == "gate_only" for c in cb)
    assert all(not A.hidden_writes(c, m) for c in train)
    gal = [c for c in train if c["meta"]["target_address"] == GAL_LINK and c["meta"].get("positive")]
    # the gallery state holds no imagesLink list: the engine answered E_INDEX "list does not exist" (r5: 0 positives)
    assert gal and all(c["meta"]["state_prep"].get("seeded_list") == {"imagesLink": 6} for c in gal)


def test_author_round_tops_up_a_target_short_of_positives(grown_engine):
    """A first answer with no positive write (all near-misses) gets a top-up ask for the deficit."""
    from souschef_lab.state import load_context_fixture
    eng, m = grown_engine
    fake = A.FakeAsk(m)
    calls = []

    def ask(system, user, schema):
        obj, u = fake(system, user, schema)
        calls.append(user)
        if len(calls) == 1:
            obj = {"cases": [{**c, "request": f"can you move my web address to a new host {i}", "oracle_ops_json": "[]",
                              "score_mode": "where", "expected_where": ["domain"], "history": []}
                             for i, c in enumerate(obj["cases"])]}
        return obj, u

    train, _, rep = A.author_round([_t(NEW_ADDR)], m, eng, load_context_fixture(), ask=ask, round_id="rt-top", pool=[],
                                   leak_index=None, progress=lambda s: None)
    pt = rep["per_target"][0]
    assert pt["calls"] == 2 and pt["topup"] == A.MIN_POSITIVE and pt["positive"] >= A.MIN_POSITIVE, pt
    assert "order of styles: " + ", ".join(A.topup_styles(A.MIN_POSITIVE, {})) in calls[1]


# --- r7: the ask rule on authored ask turns, the section instance and the "…2" twin, re-authoring -----------------------

CB_BORDER2 = "sections.contentBlock.buttonBorderColor2"
CB1 = {"kind": "section", "page": "home", "sectionId": "contentBlock-1"}


def twinned(fixed_label: bool = False) -> dict:
    """The exam manifest's contentBlock with a second button whose border colour repeats the first's label (the
    "…2" fields r6 could not train: both "Button Border Color"); fixed_label: as the r7 label fix shows it."""
    d = _doc()
    cb = d["sections"]["contentBlock"]["fields"]
    cb.setdefault("buttonBorderColor", {"type": "color", "label": "Button Border Color",
                                        "path": "settings.buttonBorderColor"})
    cb["buttonBorderColor2"] = {"type": "color", "path": "settings.buttonBorderColor2",
                                "label": "Buttons 2 › Button Border Color" if fixed_label else cb["buttonBorderColor"]["label"]}
    return d


def _case_for(m, key, request, *, view=None, history=()):
    from souschef_lab.state import build_state
    return {"id": "x", "request": request, "history": list(history), "site_state": build_state(m),
            "context": {"page": "home"}, "score": {"mode": "exact"},
            "expected": {"ops": [{"op": "set_fields", "target": CB1, "values": {key: "#7c3aed"}}]},
            "meta": {"view": view, "page": "home", "author": {"round": "rt-r6"}, "target_address": f"sections.contentBlock.{key}"}}


def test_an_ask_turn_must_be_vague():
    """r7 §1.2: a follow-up whose turn 1 gives a direction is a request to act on, never an ask (r6 trained ~23)."""
    m = Manifest(grown())
    t = {"address": NEW_ADDR, "kind": "field", "reason": "added_field", "holdout": False}
    entry = A.target_entry(t, m)
    ops = json.dumps([{"op": "set_fields", "target": BTN, "values": {"buttonShadowColor": "#222222"}}])

    def draft(first):
        raw = {"style": "followup", "request": "#222222", "page": "home",
               "history": [{"user": first, "assistant": "Which colour?", "expect": "ask"}],
               "view": {"tab": "", "activeSectionId": "", "globalSection": ""}, "oracle_ops_json": ops,
               "score_mode": "exact", "expected_where": [], "candidates": [], "reply_any": [],
               "direction": {"key": "", "dir": ""}}
        return A.to_case(raw, t, entry, m, case_id="x-1", round_id="rt-1", holdout=False)
    why = A.validate(draft("make the button shadow darker"), t, m, None, {}, leak_index=None, pool_keys=set())
    assert why.startswith("ask_turn_not_vague:") and "direction" in why
    assert A.ask_rule_reject(draft("make the button shadow #333333"), m).startswith("ask_turn_not_vague")
    assert A.ask_rule_reject(draft("I want to change the button shadow colour"), m) is None
    assert "NO direction word" in A.RULES and "same_type_on_page" in A.RULES and "label_twins" in A.RULES


def test_requests_name_the_section_instance_and_the_second_twin():
    m = Manifest(twinned())
    # five contentBlock rows on the fixture home: "the content block" names none of them
    why = A.ambiguity(_case_for(m, "buttonBorderColor", "content block button border purple"), m)
    assert why.startswith("ambiguous_instance:") and "'the Crafted with Care section'" in why
    assert A.ambiguity(_case_for(m, "buttonBorderColor", "the Crafted with Care block's button border, purple"), m) is None
    # the section editor open on it: "this section" is enough
    focus = {"tab": "editor", "activeSectionId": "contentBlock-1"}
    assert A.ambiguity(_case_for(m, "buttonBorderColor", "button border purple", view=focus), m) is None
    # the instance named in an earlier turn counts
    hist = [{"user": "the Crafted with Care section's button", "expect": "ask"}]
    assert A.ambiguity(_case_for(m, "buttonBorderColor", "border purple", history=hist), m) is None
    # the "…2" twin needs "second" (or two / right-hand …); the first twin must not say it
    ok = "the Crafted with Care block's second button border, purple"
    assert A.ambiguity(_case_for(m, "buttonBorderColor2", ok), m) is None
    why = A.ambiguity(_case_for(m, "buttonBorderColor2", "the Crafted with Care block's button border, purple"), m)
    assert why.startswith("ambiguous_twin:") and "buttonBorderColor2" in why
    assert A.ambiguity(_case_for(m, "buttonBorderColor", ok), m).startswith("ambiguous_twin:")
    assert A.ambiguity(_case_for(m, "buttonBorderColor2", "Crafted with Care, right-hand button border purple"), m) is None


def test_label_twins_pair_keys_not_label_tails():
    # the r7 launcher labels (cookie-cutter r7-labels, contentBlock): a fixed label's tail must not pair unrelated keys
    fields = {"backgroundColor": {"type": "color", "label": "Background Color"},
              "buttonBgColor": {"type": "color", "label": "Button Link › Background color"},
              "buttonBgColor2": {"type": "color", "label": "Buttons 2 › Button Link › Background color"},
              "mapDirBtnBg": {"type": "color", "label": "Secondary Button › Background Color"},
              "titleColor": {"type": "color", "label": "Headline"},
              "title2Color": {"type": "color", "label": "Content 2 › Headline"},
              "title2": {"type": "richtext", "label": "Content 2 › Headline"}}
    assert A.label_twins(fields, "buttonBgColor2") == [
        {"key": "buttonBgColor", "label": "Button Link › Background color", "group": ""}]
    assert A.label_twins(fields, "backgroundColor") == [] and A.label_twins(fields, "mapDirBtnBg") == []
    assert [t["key"] for t in A.label_twins(fields, "title2Color")] == ["titleColor"]  # the richtext twin is another type
    tw = A.label_twins(fields, "buttonBgColor2")
    assert A.twin_group("buttonBgColor2", fields["buttonBgColor2"], tw) == "Buttons 2"
    assert A.untwin("title2Color") == "titleColor" and A.untwin("body2FontSize") == "bodyFontSize"


def test_the_entry_names_the_instance_and_the_twins():
    m = Manifest(twinned())
    e = A.target_entry(_t(CB_BORDER2), m)
    assert e["op_target"] == CB1 and e["instance"] == {"id": "contentBlock-1", "title": "Crafted with Care"}
    titles = [x["title"] for x in e["same_type_on_page"]]
    assert len(titles) == 5 and titles[:3] == ["Crafted with Care", "Our Story", "Visit Us"]
    assert e["label_twins"] == [{"key": "buttonBorderColor", "label": "Button Border Color", "group": ""}]
    assert e["group"] == "second button" and e["twin_role"] == "second"
    first = A.target_entry(_t("sections.contentBlock.buttonBorderColor"), m)
    assert first["twin_role"] == "first" and first["label_twins"][0]["key"] == "buttonBorderColor2"
    # with the r7 label fix: the group is the editor heading the label now carries
    fixed = Manifest(twinned(fixed_label=True))
    assert A.target_entry(_t(CB_BORDER2), fixed)["group"] == "Buttons 2"
    assert A.ambiguity(_case_for(fixed, "buttonBorderColor2", "the Crafted with Care block's button border"), fixed) \
        .startswith("ambiguous_twin")
    # a field with no twin has none of it; a global target has no instance
    assert "label_twins" not in A.target_entry(_t(CB_GAP), Manifest(gated()))
    assert "instance" not in A.target_entry(_t(NEW_ADDR), Manifest(grown()))


def test_reauthor_targets_and_superseding_the_ambiguous_drafts():
    from souschef_lab.retrain import prepare as PREP
    m = Manifest(twinned())
    vague = _case_for(m, "buttonBorderColor2", "content block second button border purple")
    named = _case_for(m, "buttonBorderColor2", "the Crafted with Care block's second button border, purple")
    for i, c in enumerate((vague, named)):
        c.update(id=f"a{i}", holdout=False)
    # the training manifest already has the key: only `reauthor` makes it a target, whatever its coverage
    targets, rep = A.find_targets(twinned(), m, pool=[vague, named] * 4)
    assert targets == []
    targets, rep = A.find_targets(twinned(), m, pool=[vague, named] * 4, reauthor={CB_BORDER2, "sections.nope.x"})
    assert [(t["address"], t["reason"], t["min_positive"]) for t in targets] == [(CB_BORDER2, "reauthor", A.MIN_POSITIVE_REAUTHOR)]
    assert targets[0]["covered"] == 4  # only the drafts that name the instance and the twin count
    assert rep["reauthor"] == 1 and rep["reauthor_unknown"] == ["sections.nope.x"]
    assert [w["reason"] for w in rep["wanted_list"]] == ["reauthor"]
    other = _case_for(m, "buttonBorderColor", "content block button border purple")
    other.update(id="o", holdout=False)
    kept, srep = PREP.supersede_ambiguous([vague, named, other], m, [CB_BORDER2], "rt-r7")
    assert [c["id"] for c in kept] == ["a1", "o"] and srep["count"] == 1 and srep["by_reason"] == {"ambiguous_instance": 1}
    assert vague["meta"]["superseded_by"] == "rt-r7"
    # the clear draft that stays splits per case like this round's new drafts; the other target's case is untouched
    assert srep["regrouped"] == 1 and named["meta"]["group"] == f"author:{CB_BORDER2}#a1"
    assert other["meta"].get("group") != f"author:{CB_BORDER2}#o"
    assert PREP.supersede_ambiguous([vague], m, [], "rt-r7")[0] == [vague]
    assert PREP.reauthor_list({"trigger": {"reauthor": [CB_BORDER2, " "]}}, env={}) == [CB_BORDER2]
    assert PREP.reauthor_list({}, env={"SOUSCHEF_REAUTHOR": f"{CB_BORDER2}, a.b.c"}) == ["a.b.c", CB_BORDER2]


def test_prepare_relabels_pool_asks_that_give_a_value():
    from souschef_lab.retrain import prepare as PREP
    from souschef_lab.state import build_state
    m = Manifest(grown())
    st = build_state(m)

    def ask(cid, text, **meta):
        return {"id": cid, "request": text, "target": BTN, "fields": ["buttonShadowColor"], "site_state": st,
                "score": {"mode": "ask"}, "kind": "ask", "holdout": False, "expected": {"ops": []},
                "meta": {"family": "F3", **meta}}
    cases = [ask("v", "can we rethink the button shadow"), ask("d", "the button shadow is too dark"),
             ask("p", "add a blank page", capability="create_page"), {**ask("n", "make it #222222"), "target": None},
             ask("h", "the button shadow is too dark") | {"holdout": True},
             ask("u", "hide the button shadow")]  # clear, but no evidence for a write on a colour: out (r7 review)
    rep = PREP.check_ask_cases(cases, m)
    by = {c["id"]: c for c in cases}
    assert set(by) == {"v", "d", "p", "h"} and rep["dropped_ids"] == ["n", "u"] and rep["unsure"] == 1
    assert by["v"]["score"]["mode"] == "ask" and by["v"]["meta"]["ask_family"] == "choice_missing"
    d = by["d"]
    assert d["score"]["mode"] == "keys" and d["expected"]["keys"] == [{"target": BTN, "key": "buttonShadowColor", "list": ""}]
    assert d["expected"]["direction"] == {"key": "buttonShadowColor", "dir": "lighter", "from": None}
    assert d["meta"]["ask_rule"]["kind"] == "direction" and "ask_rule" in d["meta"]["fixed"]
    assert d["meta"]["ask_rule"]["evidence"] == "direction:lighter"
    assert by["p"]["score"]["mode"] == "ask" and by["p"]["meta"]["ask_family"] == "page_title_missing"
    assert by["h"]["score"]["mode"] == "ask"  # the exam stays as scored
    assert rep["relabelled"] == 1 and rep["kept_vague"] == 1 and rep["exempt"] == 1


@pytest.fixture(scope="module")
def twinned_engine(tmp_path_factory):
    _need_engine()
    p = tmp_path_factory.mktemp("t") / "manifest.json"
    p.write_text(json.dumps(twinned()), encoding="utf-8")
    eng = E.PersistentEngine(manifest_path=p, size=2)
    yield eng, Manifest(twinned())
    eng.close()


def test_author_round_on_a_reauthored_twin_reaches_its_floor(twinned_engine):
    """FakeAsk (the DRY rounds' author) names the instance and the twin as RULES asks, so its drafts pass the checks;
    a re-authored target aims for MIN_POSITIVE_REAUTHOR."""
    from souschef_lab.state import load_context_fixture
    eng, m = twinned_engine
    t = {**_t(CB_BORDER2), "reason": "reauthor", "min_positive": A.MIN_POSITIVE_REAUTHOR}
    train, _, rep = A.author_round([t], m, eng, load_context_fixture(), ask=A.FakeAsk(m), round_id="rt-tw", pool=[],
                                   leak_index=None, progress=lambda s: None)
    pt = rep["per_target"][0]
    assert A.MIN_POSITIVE_REAUTHOR == 14  # r7 review: 10 broke even only at a 0.65 keep rate
    assert pt["min_positive"] == 14 and pt["positive"] >= 14 and rep["short_of_positive"] == {}, pt
    assert not any(r["reason"].startswith(("ambiguous", "ask_turn")) for r in pt["reject_details"]), pt["reject_details"]
    assert all(A.ambiguity(c, m) is None for c in train)
    # a re-authored target's cases split per case (not all-or-none by address): r6's perks keys trained 0 rows, all in val
    assert {c["meta"]["group"] for c in train} == {f"author:{CB_BORDER2}#{c['id']}" for c in train}


def test_author_group_is_per_case_only_for_reauthored_targets():
    from souschef_lab.r4dataset import h100
    assert A.author_group({"address": "sections.x.k", "reason": "added_field"}, "aX-F7-0001") == "author:sections.x.k"
    assert A.author_group({"address": "sections.x.k", "reason": "reauthor"}, "aX-F7-0001") == "author:sections.x.k#aX-F7-0001"
    # why: this short key's whole author group hashes into val (r4dataset val_pct 8)
    assert h100("author:sections.perksIcons.buttonOnImageOffset") < 8


# --- r7 four keys: the hero pair, the tag picker, the Pack filter setting (scratchpad r7/four) ------------------------

HERO = {"kind": "section", "page": "home", "sectionId": "hero"}
PRODUCTS = {"kind": "page", "page": "products"}
PDP = {"kind": "page", "page": "product-details"}
HERO_W = "sections.hero.contentWidth"
HERO_BTN = "sections.hero.slides[].btnSize"
TAG = "pages.product-details.productDetail.customDescriptionTagName"
PACK = "pages.products.packFilterDisplay"
EDIBLES = "pages.products.packFilterDisplay.edibles"


def four() -> dict:
    """The exam manifest plus the four r7 keys as the r7 manifest (3914e2e4) has them: the hero pair's contentWidth and
    slides[].btnSize (btnSize a slides[0] mirror key), the products page's Pack filter (a "text" parent, an enum
    .prerolls, a "text" .edibles) and the product-details custom description tag (text, gated)."""
    d = _doc()
    for name in ("hero", "heroBanner"):
        s = d["sections"][name]
        s["fields"]["contentWidth"] = {"path": "settings.contentWidth", "label": "Content Width", "type": "range",
                                       "min": 20, "max": 100, "step": 5, "unset_value": None}
        s["lists"]["slides"]["item"]["btnSize"] = {"path": "settings.slides[].btnSize", "type": "number", "min": 10,
                                                   "max": 40, "step": 1, "default": "", "unset_value": None,
                                                   "label": "Border Color › Button size in pixels"}
        s["mirror"]["keys"] = sorted(set(s["mirror"]["keys"]) | {"btnSize"})
    p = d["pages"]["products"]["fields"]
    p["packFilterDisplay"] = {"path": "pageVisibility.products.packFilterDisplay", "label": "Pack filter display",
                              "type": "text", "unset_value": None}
    p["packFilterDisplay.prerolls"] = {"path": "pageVisibility.products.packFilterDisplay.prerolls", "type": "enum",
                                       "label": "Pack filter display", "enum": [None, "grams"], "unset_value": None,
                                       "enum_labels": {"null": "Units", "grams": "Grams"}, "writes": ["packFilterDisplay"]}
    p["packFilterDisplay.edibles"] = {"path": "pageVisibility.products.packFilterDisplay.edibles", "type": "text",
                                      "label": "Pack filter display"}
    q = d["pages"]["product-details"]["fields"]
    q["productDetail.showCustomDescription"] = {"path": "pageVisibility.productDetail.showCustomDescription",
                                                "label": "Show custom description", "type": "bool", "unset_value": None}
    q["productDetail.customDescriptionTagName"] = {
        "path": "pageVisibility.productDetail.customDescriptionTagName", "label": "Custom description tag", "type": "text",
        "conditional_on": {"key": "productDetail.showCustomDescription", "value": True}}
    return d


def _four_case(m, ops, request, *, home=None, view=None, history=(), option_lists=None, state=None, mode="exact",
               address=HERO_W, cid="x"):
    from souschef_lab.state import build_state
    c = {"id": cid, "request": request, "history": list(history), "holdout": False,
         "site_state": state if state is not None else build_state(m, home_sections=home),
         "context": {"page": "home"}, "score": {"mode": mode}, "expected": {"ops": ops},
         "meta": {"view": view, "page": "home", "author": {"round": "rt-r6"}, "target_address": address}}
    if option_lists is not None:
        c["option_lists"] = option_lists
    return c


def _r6_home():
    from souschef_lab.state import DEFAULT_HOME_SECTIONS
    return [dict(r) for r in DEFAULT_HOME_SECTIONS] + [{"id": "hero", "enabled": True}]


def test_a_hero_case_is_a_legacy_hero_site_not_two_hero_banners():
    """r6 appended an empty `hero` row beside the fixture's heroBanner (both "Hero Banner" in the editor): the teacher
    wrote the real one in 42 of 55 rows and every non-focus case failed. Now the case home swaps heroBanner out."""
    from souschef_lab.state import find_section, section_summaries
    m = Manifest(four())
    assert A.alias_type(m, "hero") == "heroBanner" and A.alias_type(m, "heroBanner") == "hero"
    assert A.alias_type(m, "reviews") == "testimonials"
    # the contentBlock editor's four types are distinct sections a site holds side by side: no alias, still appended
    assert A.alias_type(m, "perksCards") is None and A.case_home(m, "perksCards")[-1] == {"id": "perksCards", "enabled": True}
    assert A.case_home(m, "contentBlock-1") is None and A.alias_home(m, "heroBanner") is None
    t = {"address": HERO_W, "kind": "field", "reason": "reauthor", "holdout": False}
    e = A.target_entry(t, m)
    assert e["op_target"] == HERO and e["instance"] == {"id": "hero", "title": "Fresh picks every week"}
    raw = _raw("verbatim", [{"op": "set_fields", "target": HERO, "values": {"contentWidth": 70}}],
               request="set the hero banner content width to 70")
    c = A.to_case(raw, t, e, m, case_id="h-1", round_id="rt-1", holdout=False)
    rows = section_summaries(c["site_state"]["homeSections"])
    assert [r["id"] for r in rows if r["id"] in ("hero", "heroBanner")] == ["hero"] and rows[0]["id"] == "hero"
    assert rows[0]["title"] == "Fresh picks every week"
    assert len(find_section(c["site_state"], "hero")["settings"]["slides"]) == 2
    assert A.ambiguity(c, m) is None
    # the slide list target lists the slides, so a request can name one
    b = A.target_entry({**t, "address": HERO_BTN}, m)
    assert [r["index"] for r in b["rows"]] == [0, 1] and b["rows"][0]["title"] == "Fresh picks every week"
    assert "Rows:" in A.RULES and "`rows`" in A.RULES


def test_alias_instance_retires_the_r6_two_hero_cases_but_not_the_focused_ones():
    from souschef_lab.retrain import prepare as PREP
    m = Manifest(four())
    op = [{"op": "set_fields", "target": HERO, "values": {"contentWidth": 60}}]
    r6 = _four_case(m, op, "make the hero banner content narrower, 60", home=_r6_home(), cid="a")
    why = A.ambiguity(r6, m)
    assert why.startswith("alias_instance:") and "'heroBanner'" in why
    focus = _four_case(m, op, "content width 60 here", home=_r6_home(), cid="f",
                       view={"tab": "editor", "activeSectionId": "hero"})
    assert A.ambiguity(focus, m) is None
    swapped = _four_case(m, op, "hero content width 60", home=A.alias_home(m, "hero"), cid="s")
    assert A.ambiguity(swapped, m) is None
    kept, rep = PREP.supersede_ambiguous([r6, focus, swapped], m, [HERO_W], "rt-r7")
    assert [c["id"] for c in kept] == ["f", "s"] and rep["by_reason"] == {"alias_instance": 1}


def test_a_slide_write_names_the_slide():
    m = Manifest(four())
    home = A.alias_home(m, "hero")

    def upd(i, request, view=None):
        ops = [{"op": "edit_list", "target": HERO, "list": "slides", "action": "update", "index": i, "item": {"btnSize": 20}}]
        return A.ambiguity(_four_case(m, ops, request, home=home, view=view, address=HERO_BTN), m)
    assert upd(1, "make the hero button 20px").startswith("ambiguous_row:")
    assert upd(1, "make the hero button 20px", view={"tab": "editor", "activeSectionId": "hero"}).startswith("ambiguous_row")
    assert upd(1, "second slide's button, 20px") is None and upd(1, "slide 2 button size 20") is None
    assert upd(0, "the Fresh picks slide button: 20") is None and upd(0, "first slide button 20") is None
    # the flat mirror key writes slide 0 (the engine writes the same paths): it names the slide too
    flat = [{"op": "set_fields", "target": HERO, "values": {"btnSize": 20}}]
    assert A.ambiguity(_four_case(m, flat, "hero button size 20", home=home, address=HERO_BTN), m).startswith("ambiguous_row")
    assert A.ambiguity(_four_case(m, flat, "hero button size 20 on the first slide", home=home, address=HERO_BTN), m) is None
    # r6's empty appended hero (the engine makes one slide): nothing to tell apart
    assert A.row_ambiguity(_four_case(m, flat, "hero button size 20", home=_r6_home(), address=HERO_BTN,
                                      view={"tab": "editor", "activeSectionId": "hero"}), m) is None


def test_case_addresses_count_a_mirror_write_and_the_whole_pack_setting():
    m = Manifest(four())
    flat = {"expected": {"ops": [{"op": "set_fields", "target": HERO, "values": {"btnSize": 20}}]}}
    assert {"sections.hero.btnSize", HERO_BTN} <= A.case_addresses(flat, m)
    both = {"expected": {"ops": [{"op": "set_fields", "target": PRODUCTS, "values": {
        "packFilterDisplay.prerolls": "grams", "packFilterDisplay.edibles": "grams"}}]}}
    assert PACK in A.case_addresses(both, m) and EDIBLES in A.case_addresses(both, m)
    one = {"expected": {"ops": [{"op": "set_fields", "target": PRODUCTS, "values": {"packFilterDisplay.prerolls": "grams"}}]}}
    assert PACK not in A.case_addresses(one, m)
    reset = {"expected": {"ops": [{"op": "reset", "target": PRODUCTS, "keys": ["packFilterDisplay"]}]}}
    assert A.case_addresses(reset, m) == {PACK}


def test_bad_values_invented_tags_and_pack_filter_text_are_rejected():
    from souschef_lab.retrain import prepare as PREP
    from souschef_lab.state import build_state, set_path
    m = Manifest(four())
    tags = A.store_tags(TAG, "rt-1")
    assert len(tags) == A.TAG_DRAW + 3 and tags == A.store_tags(TAG, "rt-1")
    assert {"Staff Picks", "New", "Sale"} <= {t["name"] for t in tags}
    lists = {"tags": tags, "brands": []}
    pick = next(t for t in tags if t["id"] >= A.TAG_ID_BASE)

    def tag(v, request, ol=lists):
        ops = [{"op": "set_fields", "target": PDP, "values": {"productDetail.customDescriptionTagName": v}}]
        return A.bad_value(_four_case(m, ops, request, option_lists=ol, address=TAG), m)
    assert tag("house-rosin", "use the house-rosin tag").startswith("tag_not_in_store")  # r6 0250
    assert tag(pick["slug"], f"use the {pick['slug']} tag").startswith("tag_not_in_store")  # a slug, not the name
    assert tag(pick["name"], "use the other tag").startswith("tag_not_named")
    assert tag(pick["name"], f"custom description only on the {pick['name'].lower()} tag") is None
    # an r6 case carries no list: the fixture's tags count ("staff-pick" was r6 0249's answer key)
    assert tag("staff-pick", "staff pick tag", ol=None).startswith("tag_not_in_store")
    assert tag("Staff Picks", "only for the staff picks tag", ol=None) is None
    pack = [{"op": "set_fields", "target": PRODUCTS, "values": {"packFilterDisplay": "Pack Qty"}}]
    assert A.bad_value(_four_case(m, pack, "just say Pack Qty", address=PACK), m).startswith("bad_value")  # r6 0291
    edi = [{"op": "set_fields", "target": PRODUCTS, "values": {"packFilterDisplay.edibles": "Pieces per package"}}]
    assert A.bad_value(_four_case(m, edi, "edibles pack filter: Pieces per package", address=EDIBLES), m).startswith("bad_value")
    grams = [{"op": "set_fields", "target": PRODUCTS, "values": {"packFilterDisplay.edibles": "grams"}}]
    assert A.bad_value(_four_case(m, grams, "edibles pack sizes in grams", address=EDIBLES), m) is None
    st = build_state(m)
    set_path(st, "pageVisibility.products.packFilterDisplay", "Taste the difference")  # r6 0297's seed
    reset = [{"op": "reset", "target": PRODUCTS, "keys": ["packFilterDisplay"]}]
    assert A.bad_value(_four_case(m, reset, "pack filter back to normal", state=st, address=PACK), m).startswith("bad_seed")
    kept, rep = PREP.supersede_ambiguous([_four_case(m, pack, "just say Pack Qty", address=PACK, cid="p"),
                                          _four_case(m, grams, "edibles in grams", address=EDIBLES, cid="g")],
                                         m, [PACK, EDIBLES], "rt-r7")
    assert [c["id"] for c in kept] == ["g"] and rep["by_reason"] == {"bad_value": 1}


def test_the_rule_fields_get_their_note_value_shape_and_state():
    from souschef_lab.state import get_path
    m = Manifest(four())
    t = {"address": TAG, "kind": "field", "reason": "reauthor", "holdout": False}
    e = A.target_entry(t, m, round_id="rt-1")
    assert e["note"].startswith("A tag picker") and e["store_tags"] == A.store_tags(TAG, "rt-1")
    assert "relative" not in A.style_plan(t, entry=e) and "relative" not in A.topup_styles(12, e)
    raw = _raw("verbatim", [{"op": "set_fields", "target": PDP, "values": {"productDetail.customDescriptionTagName": "Sale"}}],
               request="custom description for the sale tag")
    c = A.to_case(raw, t, e, m, case_id="t-1", round_id="rt-1", holdout=False)
    assert c["option_lists"]["tags"] == e["store_tags"]
    p = A.target_entry({**t, "address": PACK}, m)
    assert p["field"]["type"] == "object" and "never renames" in p["note"] and "relative" not in A.style_plan(t, entry=p)
    raw = _raw("reset", [{"op": "reset", "target": PRODUCTS, "keys": ["packFilterDisplay"]}], request="pack filter back to units")
    c = A.to_case(raw, {**t, "address": PACK}, p, m, case_id="p-1", round_id="rt-1", holdout=False)
    assert get_path(c["site_state"], "pageVisibility.products.packFilterDisplay") == {"prerolls": "grams", "edibles": "grams"}
    assert A.bad_value(c, m) is None
    ed = A.target_entry({**t, "address": EDIBLES}, m)
    assert ed["field"]["type"] == "enum" and ed["field"]["enum"] == [None, "grams"]


@pytest.fixture(scope="module")
def four_engine(tmp_path_factory):
    _need_engine()
    p = tmp_path_factory.mktemp("four") / "manifest.json"
    p.write_text(json.dumps(four()), encoding="utf-8")
    eng = E.PersistentEngine(manifest_path=p, size=2)
    yield eng, Manifest(four())
    eng.close()


def test_the_four_keys_author_to_their_floor_and_slide_two_applies(four_engine):
    """FakeAsk (RULES-shaped drafts) on the four keys + .edibles: every target reaches MIN_POSITIVE_REAUTHOR with no
    rejection but its own duplicates; slide 2 of the swapped hero exists (r6: E_INDEX index 1 out of range); the pack
    reset's teacher variants all give the oracle's diff."""
    from souschef_lab.createpage import run_engine_result
    from souschef_lab.state import load_context_fixture
    eng, m = four_engine
    ts = [{"address": a, "kind": "field", "reason": "reauthor", "holdout": False, "min_positive": A.MIN_POSITIVE_REAUTHOR}
          for a in (HERO_W, HERO_BTN, TAG, PACK, EDIBLES)]
    train, _, rep = A.author_round(ts, m, eng, load_context_fixture(), ask=A.FakeAsk(m), round_id="rt-four", pool=[],
                                   leak_index=None, progress=lambda s: None)
    assert rep["short_of_positive"] == {}, rep["positive_by_target"]
    for pt in rep["per_target"]:
        assert {r["reason"].split(":")[0] for r in pt["reject_details"]} <= {"duplicate"}, pt["reject_details"]
    for c in train:
        assert A.ambiguity(c, m) is None and A.bad_value(c, m) is None
    tagc = [c for c in train if c["meta"]["target_address"] == TAG and c["meta"].get("positive")]
    assert tagc and all(c["option_lists"]["tags"] for c in tagc) and all(c["context"].get("tags") for c in tagc)
    st = next(c for c in train if c["meta"]["target_address"] == HERO_BTN)["site_state"]
    r = run_engine_result(eng, st, [{"op": "edit_list", "target": HERO, "list": "slides", "action": "update", "index": 1,
                                     "item": {"btnSize": 22}}], {"currentPage": "home"})
    assert r.ok and any("slides[1].btnSize" in str(d.get("path")) for d in r.diff())
    reset = next(c for c in train if c["meta"]["target_address"] == PACK and c["meta"]["style"] == "reset")

    def leaves(ops):
        res = run_engine_result(eng, reset["site_state"], ops, {"currentPage": "products"})
        assert res.ok
        return {(d["path"], d.get("after")) for d in res.diff()}
    want = leaves([{"op": "reset", "target": PRODUCTS, "keys": ["packFilterDisplay"]}])
    assert want == leaves([{"op": "set_fields", "target": PRODUCTS, "values": {
        "packFilterDisplay.prerolls": None, "packFilterDisplay.edibles": None}}])
    assert want == leaves([{"op": "restore_default", "target": PRODUCTS,
                            "keys": ["packFilterDisplay", "packFilterDisplay.prerolls", "packFilterDisplay.edibles"]}])
