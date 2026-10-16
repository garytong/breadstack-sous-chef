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


def _raw(style, ops, request="make the image gap 24 on the first content block", mode="exact"):
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
                                       "values": {"stackedImageGap": 24}}], request="image gap 30 please"),
                    t, e, m, case_id="x-3", round_id="rt-1", holdout=False)
    why = A.validate(bad, t, m, eng, extras, leak_index=None, pool_keys=set())
    assert why.startswith("oracle:exact_refused: E_TARGET") and "contentBlock" in why
    # gate_only: must set the gate, and is not a positive write of the target
    go = A.to_case(_raw("gate_only", [{"op": "set_fields", "target": e["op_target"], "values": {"imagePosition": "top"}}],
                        request="put the pictures above the text"), t, e, m, case_id="x-4", round_id="rt-1", holdout=False)
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
