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
    prompt = A.user_prompt(entry, A.style_plan(t), ["drop shadow"], [])
    assert "buttonShadowColor" in prompt and "drop shadow" in prompt
    assert "verbatim, verbatim, colloquial, colloquial, short, focus, followup, near_miss" in prompt
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
    assert len(train) == 8, rep["per_target"]
    assert rep["rejects"].get("retried:keys") == 2  # the two broken cases came back right on the re-ask
    assert rep["per_target"][0]["calls"] == 2
    styles = sorted(c["meta"]["style"] for c in train)
    assert styles == sorted(["verbatim", "verbatim", "colloquial", "colloquial", "short", "focus", "followup", "near_miss"])
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
    assert len({c["id"] for c in train}) == 8
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
