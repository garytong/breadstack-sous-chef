"""r4score's r6 relaxations (refused_corrected, mailto, same_control, acted_first_turn), the font policy switch,
the gate helpers and rescore_row (r5why PLAN §2b "Repair the stable reject core", "Settle the font policy")."""
from __future__ import annotations

import copy
import json

import pytest

from souschef_lab import FIXTURES_DIR
from souschef_lab import engine as E
from souschef_lab import r4score as R
from souschef_lab.manifest import Manifest, load

EXAM_MANIFEST = FIXTURES_DIR / "exam" / "manifest-real.json"
TYPO = {"kind": "global", "panel": "TypographyColorsSection"}
CB1 = {"kind": "section", "page": "home", "sectionId": "contentBlock-1"}


def _need_engine():
    if not E.engine_available():
        pytest.skip("node and the engine (COOKIE_CUTTER_DIR) are needed: " + E.unavailable_reason())


def _row(ops, oks, *, batches=None, text="Done.", warnings=None, hidden=None):
    """A collect row: ops (one tool.call each) answered by the engine batch by batch (`batches`: the batch of each
    op; default one batch)."""
    batches = batches or [0] * len(ops)
    ev = [{"event": "tool.call", "seq": i, "data": {"id": f"c{i}", "name": o["op"],
                                                   "input": {k: v for k, v in o.items() if k != "op"}}}
          for i, o in enumerate(ops)]
    ev += [{"event": "text.delta", "seq": 100, "data": {"text": text}},
           {"event": "turn.done", "seq": 101, "data": {"stop": "end_turn"}}]
    res = []
    for b in sorted(set(batches)):
        res.append({"results": [{"ok": ok, "warnings": list((warnings or {}).get(i) or []),
                                 **({"hidden": hidden[i]} if hidden and i in hidden else {})}
                                for i, (ok, x) in enumerate(zip(oks, batches)) if x == b]})
    return {"status": "ok", "stop": "end_turn", "events": ev, "reply": text, "ops": list(ops),
            "tool_calls": [{"name": o["op"], "input": {k: v for k, v in o.items() if k != "op"}} for o in ops],
            "engine_results": res}


def _set(target, **values):
    return {"op": "set_fields", "target": target, "values": values}


# ── policy ──────────────────────────────────────────────────────────────────

def test_policy_defaults_env_and_typos(monkeypatch):
    monkeypatch.delenv("SOUSCHEF_SCORE_POLICY", raising=False)
    monkeypatch.delenv("SOUSCHEF_FONT_POLICY", raising=False)
    p = R.policy_of()
    # r6 review: refused_corrected and acted_first_turn default off (their rows would teach an invalid op first /
    # acting on a vague request); mailto one way; same_control only for the key the editor writes
    # 2026-10-15: "the font" means every site font (font "all"), in collect AND the eval (not a relaxation)
    assert p == {"refused_corrected": False, "mailto": True, "same_control": True, "acted_first_turn": "off",
                 "font": "all"}
    assert R.STRICT_POLICY == {"refused_corrected": False, "mailto": False, "same_control": False,
                               "acted_first_turn": "off", "font": "all"}
    assert R.POLICY_VERSION == "r6-3"
    monkeypatch.setenv("SOUSCHEF_FONT_POLICY", "body_only")
    monkeypatch.setenv("SOUSCHEF_SCORE_POLICY", json.dumps({"mailto": False}))
    p = R.policy_of()
    assert p["font"] == "body_only" and p["mailto"] is False
    assert R.policy_of({"font": "body_or_both"})["font"] == "body_or_both"
    assert R.policy_of({"acted_first_turn": False})["acted_first_turn"] == "off"
    with pytest.raises(ValueError):
        R.policy_of({"font": "headings"})
    with pytest.raises(ValueError):
        R.policy_of({"acted_first_turn": "anything"})


# ── refused_corrected ───────────────────────────────────────────────────────

def _weight_op(v):
    return {"op": "edit_list", "target": {"kind": "section", "page": "home", "sectionId": "heroBanner"}, "list": "slides",
            "action": "update", "index": 0, "item": {"headlineFontWeight": v}}


def test_refused_then_resent_in_a_later_batch_is_corrected():
    assert R.refused_corrected(_row([_weight_op(900), _weight_op("900")], [False, True], batches=[0, 1]))
    # the same batch: a hedge, not a correction
    assert not R.refused_corrected(_row([_weight_op(900), _weight_op("900")], [False, True], batches=[0, 0]))
    # the retry wrote another key / was refused again / nothing was refused
    other = {**_weight_op("900"), "item": {"headlineFontSize": 40}}
    assert not R.refused_corrected(_row([_weight_op(900), other], [False, True], batches=[0, 1]))
    assert not R.refused_corrected(_row([_weight_op(900), _weight_op("900")], [False, False], batches=[0, 1]))
    assert not R.refused_corrected(_row([_weight_op("900")], [True]))


def _exact_verdict(verdict, reasons, exp, act, errors=()):
    return {"verdict": verdict, "reasons": list(reasons), "info": [],
            "detail": {"expected_diff": [list(x) for x in exp.items()], "actual_diff": [list(x) for x in act.items()],
                       "engine_errors": list(errors), "warnings": []}}


def test_relax_exact_drops_invalid_op_only_when_every_refusal_was_corrected():
    d = {"homeSections[0].settings.slides[0].headlineFontWeight": '"900"'}
    v = _exact_verdict("invalid_op", ["invalid_op"], d, d, [{"code": "E_VALUE"}])
    row = _row([_weight_op(900), _weight_op("900")], [False, True], batches=[0, 1])
    on = R.policy_of({"refused_corrected": True})
    r = R.relax_exact(v, row, on)
    assert r["verdict"] == "pass" and r["relaxed"] == ["refused_corrected"]
    assert R.relax_exact(v, row, R.policy_of())["verdict"] == "invalid_op"  # off by default
    # a wrong final diff still fails, on the diff
    v2 = _exact_verdict("invalid_op", ["invalid_op", "wrong_value"], d, {k: '"800"' for k in d}, [{"code": "E_VALUE"}])
    assert R.relax_exact(v2, row, on)["verdict"] == "wrong_value"
    # the replay refused more than the live turn did: not the same turn, keep invalid_op
    v3 = _exact_verdict("invalid_op", ["invalid_op"], d, d, [{"code": "E_VALUE"}, {"code": "E_TARGET"}])
    assert R.relax_exact(v3, row, on)["verdict"] == "invalid_op"


# ── mailto ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,a,b,ok", [
    ("globalSettings.socialLinks.email", '"hello@shop.test"', '"mailto:hello@shop.test"', True),
    # one way only: a bare address where mailto: was expected is a broken link on the site (href={socialLinks.email})
    ("globalSettings.socialLinks.email", '"mailto:Hello@Shop.test"', '"hello@shop.test"', False),
    ("globalSettings.socialLinks.email", '"Hello@Shop.test"', '"mailto:hello@shop.test"', True),
    ("homeSections[2].settings.ctaLink", '"team@shop.test"', '"mailto:team@shop.test"', True),
    ("globalSettings.footerCopyrightText", '"hello@shop.test"', '"mailto:hello@shop.test"', False),  # not a link
    ("globalSettings.socialLinks.email", '"hello@shop.test"', '"hello@shop.test"', False),  # equal: nothing to relax
    ("globalSettings.socialLinks.email", '"/contact"', '"mailto:/contact"', False),  # not an address
])
def test_mailto_equal(path, a, b, ok):
    assert R.mailto_equal(path, a, b) is ok  # (path, expected, actual)


def test_relax_exact_mailto():
    p = "globalSettings.socialLinks.email"
    v = _exact_verdict("wrong_value", ["wrong_value"], {p: '"hello@shop.test"'}, {p: '"mailto:hello@shop.test"'})
    r = R.relax_exact(v, _row([_set({"kind": "global", "panel": "SocialMediaSection"}, **{"socialLinks.email": "x"})], [True]),
                      R.policy_of())
    assert r["verdict"] == "pass" and r["relaxed"] == ["mailto"]
    assert R.relax_exact(v, {}, R.policy_of({"mailto": False}))["verdict"] == "wrong_value"
    # the reverse (expected mailto:, wrote the bare address) is not relaxed
    rv = _exact_verdict("wrong_value", ["wrong_value"], {p: '"mailto:hello@shop.test"'}, {p: '"hello@shop.test"'})
    assert R.relax_exact(rv, {}, R.policy_of())["verdict"] == "wrong_value"
    # another path wrong as well: still wrong
    v2 = _exact_verdict("wrong_value", ["wrong_value"], {p: '"hello@shop.test"', "globalSettings.x": '"1"'},
                        {p: '"mailto:hello@shop.test"', "globalSettings.x": '"2"'})
    assert R.relax_exact(v2, {}, R.policy_of())["verdict"] == "wrong_value"


def test_prepare_makes_a_bare_address_on_a_link_field_expect_mailto():
    from souschef_lab.retrain.prepare import fix_bare_mailto
    soc = {"kind": "global", "panel": "SocialMediaSection"}
    op = _set(soc, **{"socialLinks.email": "hi@shop.test"})
    c = {"id": "r4-F3-0413", "holdout": False, "score": {"mode": "exact"},
         "expected": {"ops": [op], "diff": [{"path": "globalSettings.socialLinks.email", "before": "a@b.test",
                                             "after": "hi@shop.test"}], "alt_exact": []},
         "meta": {"oracle_ops": [copy.deepcopy(op)]}}
    exam = copy.deepcopy(c)
    exam.update(id="h1", holdout=True)
    plain = {"id": "x", "expected": {"ops": [_set(soc, **{"footerEmail": "hi@shop.test"})], "diff": []}, "meta": {}}
    rep = fix_bare_mailto([c, exam, plain])
    assert rep == {"count": 1, "ids": ["r4-F3-0413"]}
    assert c["expected"]["ops"][0]["values"]["socialLinks.email"] == "mailto:hi@shop.test"
    assert c["meta"]["oracle_ops"][0]["values"]["socialLinks.email"] == "mailto:hi@shop.test"
    assert c["expected"]["diff"][0]["after"] == "mailto:hi@shop.test" and c["meta"]["fixed"] == ["mailto"]
    assert exam["expected"]["ops"][0]["values"]["socialLinks.email"] == "hi@shop.test"   # the exam is left as scored
    assert plain["expected"]["ops"][0]["values"]["footerEmail"] == "hi@shop.test"        # not a link field


# ── same_control ────────────────────────────────────────────────────────────

def _mini_manifest() -> Manifest:
    src = {"file": "BasicEditor.jsx", "line": 10}
    align = {"type": "enum", "label": "Text Alignment", "enum": ["left", "center", "right"], "source": src}
    chip = {"type": "color", "label": "Background", "source": {"file": "Card.jsx", "line": 3}}
    name = {"type": "text", "label": "Nav Name", "source": {"file": "Header.jsx", "line": 7}}
    return Manifest({"version": 1, "sections": {"contentBlock": {"fields": {
        "textAlign": {**align, "path": "settings.textAlign"},
        "contentTextAlign": {**align, "path": "settings.contentTextAlign"},
        "splitContentAlign": {**align, "path": "settings.splitContentAlign"},
        "titleAlign": {**align, "label": "Title Alignment", "path": "settings.titleAlign"},
        "otherAlign": {**align, "source": {"file": "BasicEditor.jsx", "line": 99}, "path": "settings.otherAlign"},
    }}}, "globalPanels": {"ProductCardSection": {"fields": {
        "cardThcChipBg": {**chip, "path": "globalSettings.cardThcChipBg"},
        "cardCbdChipBg": {**chip, "path": "globalSettings.cardCbdChipBg"},
    }}, "HeaderSection": {"fields": {
        "pageVisibility.faq.headerNavName": {**name, "path": "pageVisibility.faq.headerNavName"},
        "pageVisibility.blog.headerNavName": {**name, "path": "pageVisibility.blog.headerNavName"},
    }}}})


def _cb_state(style=None, sid="contentBlock-1"):
    st = {} if style is None else {"style": style}
    return {"homeSections": [{"id": sid, "settings": st}]}


def test_same_control_siblings_are_one_editor_control_only():
    m = _mini_manifest()
    any_ = dict(editor_only=False)
    assert R.same_control_siblings(m, CB1, "textAlign", **any_) == ["contentTextAlign", "splitContentAlign"]
    assert R.same_control_siblings(m, CB1, "contentTextAlign", **any_) == ["splitContentAlign", "textAlign"]
    # the scorer's use: only the key the editor writes for the section's layout (BasicEditor contentAlignKey)
    assert R.same_control_siblings(m, CB1, "contentTextAlign", _cb_state("image-text")) == ["textAlign"]
    assert R.same_control_siblings(m, CB1, "textAlign", _cb_state("image-text")) == []  # r5's 18 rows: no swap
    assert R.same_control_siblings(m, CB1, "textAlign", _cb_state("split")) == ["splitContentAlign"]
    assert R.same_control_siblings(m, CB1, "textAlign", _cb_state("feature-icons")) == ["contentTextAlign"]
    assert R.same_control_siblings(m, CB1, "textAlign", _cb_state("feature-columns")) == ["contentTextAlign"]
    assert R.same_control_siblings(m, CB1, "textAlign") == []          # no state: cannot tell, no swap
    assert R.same_control_siblings(m, CB1, "textAlign", {"homeSections": []}) == []
    perks = {"kind": "section", "page": "home", "sectionId": "perksCards"}
    assert R.content_align_key("perksCards", {}) == "contentTextAlign"   # resolveSectionEditor defaults feature-cards
    assert R.content_align_key("perksIcons-2", {"style": "split"}) == "splitContentAlign"
    assert R.content_align_key("contentBlock-3", {"style": "overlay"}) == "textAlign"
    assert R.content_align_key("milestones", {}) is None
    assert R.editor_key_for(perks, ["textAlign", "contentTextAlign", "splitContentAlign"],
                            _cb_state(None, "perksCards")) == "contentTextAlign"
    assert R.editor_key_for(perks, ["linkColor", "shopMenuTextColor"], _cb_state(None, "perksCards")) is None
    # another label, another source line, a one-word label (a per-item loop), the same leaf on another page
    assert R.same_control_siblings(m, CB1, "titleAlign", **any_) == []
    assert R.same_control_siblings(m, CB1, "otherAlign", **any_) == []
    assert R.same_control_siblings(m, {"kind": "global", "panel": "ProductCardSection"}, "cardThcChipBg", **any_) == []
    assert R.same_control_siblings(m, {"kind": "global", "panel": "HeaderSection"}, "pageVisibility.faq.headerNavName",
                                   **any_) == []
    assert R.same_control_siblings(None, CB1, "textAlign", **any_) == []
    # the real manifest: contentBlock's three "Text Alignment" fields, nothing for Typography's "Link Color"
    real = load(str(EXAM_MANIFEST))
    assert R.same_control_siblings(real, CB1, "textAlign", **any_) == ["contentTextAlign", "splitContentAlign"]
    assert R.same_control_siblings(real, CB1, "textAlign", _cb_state("split")) == ["splitContentAlign"]
    assert R.same_control_siblings(real, TYPO, "linkColor", **any_) == []


def test_keys_mode_accepts_only_the_key_the_editor_writes():
    m = _mini_manifest()
    case = {"score": {"mode": "keys"}, "context": {"page": "home"}, "site_state": _cb_state("feature-cards"),
            "expected": {"keys": [{"target": CB1, "key": "textAlign", "list": ""}], "alt": []}}
    row = _row([_set(CB1, contentTextAlign="left")], [True])
    s = R.score(case, row, manifest=m)
    assert s["verdict"] == "pass" and s["relaxed"] == ["same_control"] and "same_control:textAlign->contentTextAlign" in s["info"]
    assert R.score(case, row, manifest=m, policy={"same_control": False})["verdict"] == "wrong_target"
    # r5's 18 rows: image-text renders textAlign only, so contentTextAlign changes nothing on the page — no pass
    img = {**case, "site_state": _cb_state("image-text")}
    assert R.score(img, row, manifest=m)["verdict"] == "wrong_target"
    # two members at once still fails (each alternative leaves the other as a stray write)
    both = _row([_set(CB1, contentTextAlign="left", splitContentAlign="left")], [True])
    assert not R.score(case, both, manifest=m)["passed"]


# ── font policy ─────────────────────────────────────────────────────────────

def _font_case(request, mode="exact"):
    ops = [_set(TYPO, bodyFont="DM Sans", headingFont="DM Sans")]
    return {"request": request, "history": [], "score": {"mode": mode}, "context": {"page": "home"},
            "expected": {"ops": ops if mode == "exact" else [],
                         "diff": [{"path": "globalSettings.bodyFont", "after": "DM Sans"},
                                  {"path": "globalSettings.headingFont", "after": "DM Sans"}],
                         "keys": [{"target": TYPO, "key": "bodyFont", "list": ""}, {"target": TYPO, "key": "headingFont", "list": ""}],
                         "alt": [], "alt_exact": []},
            "meta": {"oracle_ops": ops}}


def test_font_policy_switch_on_training_cases():
    c = _font_case("let's do DM Sans for the site's text")
    assert R.site_font_case(c) == "both"
    assert R.apply_font_policy(c, "as_authored") == (c, None)
    body, note = R.apply_font_policy(c, "body_only")
    assert note == "font:body_only"
    assert body["expected"]["ops"] == [_set(TYPO, bodyFont="DM Sans")]
    assert [d["path"] for d in body["expected"]["diff"]] == ["globalSettings.bodyFont"]
    assert [k["key"] for k in body["expected"]["keys"]] == ["bodyFont"]
    either, note = R.apply_font_policy(c, "body_or_both")
    assert note == "font:body_or_both" and either["expected"]["ops"] == c["expected"]["ops"]
    assert either["expected"]["alt_exact"][0]["ops"] == [_set(TYPO, bodyFont="DM Sans")]
    # a request that names the headings keeps both under the r6-2 policies; "everything" is every font under "all"
    named = _font_case("everything in DM Sans, headings and paragraphs")
    assert R.site_font_case(named) is None and R.apply_font_policy(named, "body_only")[1] is None
    assert R.apply_font_policy(named, "all")[1] == "font:all"
    pair = _font_case("headings and paragraphs in DM Sans")
    assert R.apply_font_policy(pair, "all") == (pair, None)
    with pytest.raises(ValueError):
        R.apply_font_policy(c, "nope")


def test_font_policy_keys_scoring():
    c = _font_case("the site's text in DM Sans", mode="keys")
    both = _row([_set(TYPO, bodyFont="DM Sans", headingFont="DM Sans")], [True])
    body = _row([_set(TYPO, bodyFont="DM Sans")], [True])
    assert R.score(c, both, policy={"font": "as_authored"})["verdict"] == "pass"   # r4 / r5
    assert R.score(c, body, policy={"font": "as_authored"})["verdict"] == "partial"
    assert R.score(c, both)["verdict"] == "missing_writes"            # "all": every site font, the default
    assert R.score(c, body)["verdict"] == "missing_writes"
    assert R.score(c, both, policy={"font": "body_only"})["verdict"] == "stray_writes"
    assert R.score(c, body, policy={"font": "body_only"})["verdict"] == "pass"
    assert R.score(c, both, policy={"font": "body_or_both"})["verdict"] == "pass"
    assert R.score(c, body, policy={"font": "body_or_both"})["verdict"] == "pass"


def test_font_policy_on_an_exam_shaped_row():
    exam = {"request": "okay, lora", "history": [{"user": "i don't like the font"}], "score": {"mode": "keys"},
            "context": {"page": "home"}, "expected": {"keys": [{"target": TYPO, "key": "bodyFont", "list": ""}], "alt": []}}
    assert R.site_font_case(exam) == "body"
    assert R.apply_font_policy(exam, "body_only")[1] is None  # today's exam
    wide, note = R.apply_font_policy(exam, "body_or_both")
    assert note and [k["key"] for k in wide["expected"]["alt"][-1]] == ["bodyFont", "headingFont"]
    from souschef_lab.exam import score_turn
    both = _row([_set(TYPO, bodyFont="Lora", headingFont="Lora")], [True])
    assert score_turn(exam, both)["verdict"] == "stray_writes"
    assert score_turn(wide, both)["verdict"] == "pass"


# ── gates ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rule,gate,by,open_", [
    ("conditional_on", {"key": "s", "value": "modern"}, "modern", True),
    ("conditional_on", {"key": "s", "value": "modern"}, "grid", False),
    ("conditional_on", {"key": "s", "value": "modern"}, None, False),
    ("hidden_when", {"key": "style", "value": ["image-text", "milestones"]}, "feature-cards", True),
    ("hidden_when", {"key": "style", "value": ["image-text", "milestones"]}, "image-text", False),
    ("conditional_on", {"key": "img", "value": "set"}, "https://x/y.jpg", True),
    ("conditional_on", {"key": "img", "value": "set"}, "", False),
    ("conditional_on", {"key": "on", "value": True}, True, True),
])
def test_gate_open(rule, gate, by, open_):
    assert R.gate_open(rule, gate, by) is open_


def test_closed_gate_writes_reads_the_engine_hidden_detail():
    open_ = {"ok": True, "warnings": ["W_HIDDEN:shopMenuFont"],
             "hidden": [{"key": "shopMenuFont", "by": "shopMenuStyle", "byValue": "modern", "rule": "conditional_on",
                         "gate": {"key": "shopMenuStyle", "value": "modern"}},
                        {"key": "shopMenuFont", "by": "shopMenuStyle", "byValue": "modern", "rule": "hidden_when",
                         "gate": {"key": "shopMenuStyle", "value": "location"}}]}
    closed = copy.deepcopy(open_)
    for h in closed["hidden"]:
        h["byValue"] = "location"
    assert R.closed_gate_writes([open_]) == []
    assert R.closed_gate_writes([{"results": [closed]}]) == ["shopMenuFont"]
    assert R.closed_gate_writes([{"ok": True, "warnings": ["W_HIDDEN:x"]}]) == ["x"]  # no detail: closed
    assert R.closed_gate_writes([{"ok": False, "warnings": ["W_HIDDEN:x"]}]) == []    # refused: nothing written


# ── acted_first_turn ────────────────────────────────────────────────────────

NEWS = {"kind": "section", "page": "home", "sectionId": "newsletter"}


def _followup_case():
    return {"score": {"mode": "keys"}, "context": {"page": "home"}, "history": [{"user": "too much empty space", "expect": "offer"}],
            "expected": {"ops": [_set(NEWS, paddingY=24)], "keys": [{"target": NEWS, "key": "paddingY", "list": ""}], "alt": []},
            "meta": {"oracle_ops": [_set(NEWS, paddingY=24)]}}


def test_acting_on_a_vague_turn_is_excused_on_the_case_keys_only():
    case = _followup_case()
    acted = _row([_set(NEWS, paddingY=32)], [True])
    div = [{"turn": 0, "expect": "offer", "why": "did_not_ask"}]
    ok = {"verdict": "pass", "passed": True}
    on = {"acted_first_turn": "keys"}
    scored = _row([_set(NEWS, paddingY=24)], [True])  # the scored turn writes the same key again
    assert R.excuse_divergences(case, [acted], div, ok, row=scored) == (div, [])  # off by default
    keep, exc = R.excuse_divergences(case, [acted], div, ok, on, row=scored)
    assert keep == [] and exc[0]["excused"] == "acted_first_turn:keys"
    # without the scored row, or when the history turn's write survives into the final state: not excused
    assert R.excuse_divergences(case, [acted], div, ok, on) == (div, [])
    two = _row([_set(NEWS, paddingY=32, paddingTop=8)], [True])
    assert R.excuse_divergences(case, [two], div, ok, {"acted_first_turn": "target"}, row=scored)[1] == []
    # the scored turn failed / the policy is off / it wrote another key / it wrote nothing
    assert R.excuse_divergences(case, [acted], div, {"verdict": "wrong_target", "passed": False}, on, row=scored) == (div, [])
    assert R.excuse_divergences(case, [acted], div, ok, {"acted_first_turn": "off"}, row=scored) == (div, [])
    other = _row([_set(NEWS, backgroundColor="#000000")], [True])
    assert R.excuse_divergences(case, [other], div, ok, on, row=scored)[1] == []
    scored2 = _row([_set(NEWS, paddingY=24, backgroundColor="#111111")], [True])
    assert R.excuse_divergences(case, [other], div, ok, {"acted_first_turn": "target"}, row=scored2)[1]  # looser scope
    asked = _row([], [], text="How much less?")
    assert R.excuse_divergences(case, [asked], div, ok, on, row=scored)[1] == []
    # a write whose gate is closed is never excused
    hidden = _row([_set(NEWS, paddingY=32)], [True], warnings={0: ["W_HIDDEN:paddingY"]})
    assert R.excuse_divergences(case, [hidden], div, ok, on, row=scored)[1] == []
    # a "write" turn that did not write is not this relaxation's
    w = [{"turn": 0, "expect": "write", "why": "did_not_write"}]
    assert R.excuse_divergences(case, [asked], w, ok, on, row=scored) == (w, [])


def test_acting_first_never_excuses_invented_text():
    # r4-F2-0102/0103 shape: the vague turn invented a 5-star review and added it; the user's real review came later
    tst = {"kind": "section", "page": "home", "sectionId": "testimonials"}
    case = {"request": 'add this one: "Best cookies in town" by Sam', "history": [{"user": "add a review", "expect": "ask"}],
            "score": {"mode": "keys"}, "context": {"page": "home"},
            "expected": {"keys": [{"target": tst, "key": "", "list": "testimonials"}], "alt": []},
            "meta": {"oracle_ops": []}}
    add = lambda q, a: {"op": "edit_list", "target": tst, "list": "testimonials", "action": "add",  # noqa: E731
                        "item": {"quote": q, "author": a, "rating": 5}}
    invented = _row([add("Absolutely the best treats, five stars!", "Alex M.")], [True])
    real = _row([add("Best cookies in town", "Sam")], [True])
    div = [{"turn": 0, "expect": "ask", "why": "did_not_ask"}]
    ok = {"verdict": "pass", "passed": True}
    assert R.invented_texts(case, invented) == ["testimonials[].author", "testimonials[].quote"]
    assert R.invented_texts(case, real) == []
    for scope in ("keys", "target"):
        assert R.excuse_divergences(case, [invented], div, ok, {"acted_first_turn": scope}, row=real)[1] == []
    # a text field the user's words fill is fine; one the teacher made up is not
    cb = {"request": "the heading should say Fresh Daily", "history": [{"user": "change the heading"}]}
    title = Manifest({"version": 1, "sections": {"contentBlock": {"fields": {
        "title": {"type": "text", "label": "Title", "path": "settings.title"}}}}})
    assert R.invented_texts(cb, _row([_set(CB1, title="Fresh daily!")], [True]), title) == []
    assert R.invented_texts(cb, _row([_set(CB1, title="Baked with love")], [True]), title) == ["title"]


# ── end to end through the engine ───────────────────────────────────────────

def _engine_case(m, eng, ops, request):
    from souschef_lab import state as S
    from souschef_lab.verify import ctx_of
    st = S.build_state(m, template="vintage")
    ctx = S.context_of(st, "home", S.load_context_fixture(), m, view=None, text=request)
    case = {"id": "t", "kind": "set_field", "site_state": st, "context": ctx, "request": request, "history": [],
            "template": "vintage", "score": {"mode": "exact"},
            "expected": {"ops": ops, "diff": [], "keys": [], "alt": [], "alt_exact": []}, "meta": {}}
    case["expected"]["diff"] = eng.run(st, ops, ctx_of(case)).diff()
    return case


def _live_row(eng, case, ops, batches):
    """The row the collector would log: each batch through the engine on the state the last one left."""
    from souschef_lab.verify import ctx_of
    st = copy.deepcopy(case["site_state"])
    row = _row(ops, [True] * len(ops), batches=batches)
    res = []
    for b in sorted(set(batches)):
        bops = [o for o, x in zip(ops, batches) if x == b]
        r = eng.run(st, bops, ctx_of(case), print_state=True)
        res.append({"results": r.results, "changes": r.changes})
        st = r.state or st
    row["engine_results"] = res
    return row


def test_end_to_end_mailto_refused_corrected_and_same_control():
    _need_engine()
    m = load(str(EXAM_MANIFEST))
    eng = E.PersistentEngine(manifest_path=EXAM_MANIFEST, size=1)
    try:
        soc = {"kind": "global", "panel": "SocialMediaSection"}
        case = _engine_case(m, eng, [_set(soc, **{"socialLinks.email": "hello@shop.test"})], "email icon: hello@shop.test")
        row = _live_row(eng, case, [_set(soc, **{"socialLinks.email": "mailto:hello@shop.test"})], [0])
        s = R.score(case, row, eng)
        assert s["verdict"] == "pass" and s["relaxed"] == ["mailto"], s
        assert R.score(case, row, eng, policy={"mailto": False})["verdict"] == "wrong_value"

        hero = {"kind": "section", "page": "home", "sectionId": "heroBanner"}
        up = lambda v: {"op": "edit_list", "target": hero, "list": "slides", "action": "update", "index": 0,  # noqa: E731
                        "item": {"headlineFontWeight": v}}
        case = _engine_case(m, eng, [up("900")], "headline weight 900")
        row = _live_row(eng, case, [up(900), up("900")], [0, 1])
        assert [x["ok"] for b in row["engine_results"] for x in b["results"]] == [False, True]
        s = R.score(case, row, eng, policy={"refused_corrected": True})
        assert s["verdict"] == "pass" and s["relaxed"] == ["refused_corrected"], s
        assert R.score(case, row, eng)["verdict"] == "invalid_op"  # off by default

        # image-text (the default layout) renders textAlign only: contentTextAlign is not an answer
        case = _engine_case(m, eng, [_set(CB1, textAlign="right")], "right-align the text")
        row = _live_row(eng, case, [_set(CB1, contentTextAlign="right")], [0])
        assert R.score(case, row, eng, manifest=m)["verdict"] != "pass"
        # a split section renders splitContentAlign: a case that expected textAlign there accepts the editor's key
        from souschef_lab.state import section_settings
        st = copy.deepcopy(case["site_state"])
        section_settings(st, "contentBlock-1")["style"] = "split"
        case = _engine_case(m, eng, [_set(CB1, textAlign="right")], "right-align the text")
        case["site_state"] = st
        from souschef_lab.verify import ctx_of
        case["expected"]["diff"] = eng.run(st, case["expected"]["ops"], ctx_of(case)).diff()
        row = _live_row(eng, case, [_set(CB1, splitContentAlign="right")], [0])
        s = R.score(case, row, eng, manifest=m)
        assert s["verdict"] == "pass" and "same_control:textAlign->splitContentAlign" in s["info"], s
        assert R.score(case, row, eng, manifest=m, policy={"same_control": False})["verdict"] != "pass"
        # a single-turn row re-scores to what it scored
        assert R.rescore_row(case, row, eng, manifest=m)["verdict"] == "pass"
    finally:
        eng.close()


def test_a_rounds_trigger_sets_the_scoring_policy_unless_the_environment_did():
    env = {}
    pol = R.adopt_trigger_policy({"font_policy": "body_or_both", "score_policy": {"mailto": False}}, env=env)
    assert pol["font"] == "body_or_both" and pol["mailto"] is False and pol["same_control"] is True
    assert env == {"SOUSCHEF_FONT_POLICY": "body_or_both", "SOUSCHEF_SCORE_POLICY": '{"mailto": false}'}
    env = {"SOUSCHEF_FONT_POLICY": "as_authored"}
    assert R.adopt_trigger_policy({"font_policy": "body_only"}, env=env)["font"] == "as_authored"
    assert R.adopt_trigger_policy(None, env={})["font"] == "all"
    with pytest.raises(ValueError):
        R.adopt_trigger_policy({"font_policy": "headings"}, env={})
    with pytest.raises(ValueError):
        R.adopt_trigger_policy({"score_policy": {"mailtoo": False}}, env={})
    # the eval lays the trigger over STRICT_POLICY: no trigger = the r4 rules; a trigger key turns one back on
    env = {}
    assert R.adopt_trigger_policy(None, env=env, base=R.STRICT_POLICY) == R.STRICT_POLICY
    pol = R.adopt_trigger_policy({"score_policy": {"mailto": True}}, env={}, base=R.STRICT_POLICY)
    assert pol["mailto"] is True and pol["same_control"] is False and pol["acted_first_turn"] == "off"


# ── the font policy "all" (2026-10-15: "the font" = every site font) ───────────────────────────────────────────────────

def _vintage():
    from souschef_lab import drift as DR
    from souschef_lab.exam import MANIFEST_FILE, load_set, vintage_state
    return vintage_state(MANIFEST_FILE, DR.home_context(load_set("paraphrase")))


def test_site_font_fields_come_from_the_manifest_and_the_state():
    from souschef_lab import exam as X
    m = load(str(EXAM_MANIFEST))
    st = _vintage()
    sf = R.site_font_fields(m, st)
    # the exam's amended rows are this, written out (exam.SITE_FONTS_VINTAGE*)
    assert sf["required"] == list(X.SITE_FONTS_VINTAGE)
    assert set(sf["optional"]) == set(X.SITE_FONTS_VINTAGE_OPTIONAL)
    assert set(sf["hidden"]) == {"shopMenuFont", "shopMenuLocationNameFontFamily", "shopMenuLocationButtonFontFamily",
                                 "shopMenuLocationSecondaryFontFamily"}  # vintage's shop menu is not "modern"
    assert sf["panels"]["buttonFont"] == "TypographyColorsSection" and sf["panels"]["productTitleFont"] == "ProductCardSection"
    # the shop menu font is shown (and inherits: optional) on a modern shop menu; a set cart font is required
    st2 = copy.deepcopy(st)
    st2["globalSettings"]["shopMenuStyle"] = "modern"
    st2["globalSettings"]["cartTitleFont"] = "Oswald"
    sf2 = R.site_font_fields(m, st2)
    assert "shopMenuFont" in sf2["optional"] and "cartTitleFont" in sf2["required"]
    # no state: the theme fonts
    assert R.site_font_fields(m)["required"] == list(R.THEME_FONTS)


def test_generic_font_requests_and_named_parts():
    def c(text):
        return _font_case(text)
    for t in ("let's do DM Sans for the site's text", "our brand guide says DM Sans, please use it on the site",
              "everything on the site in DM Sans, headings and paragraphs", "change all fonts to DM Sans pls"):
        assert R.generic_font_request(c(t)), t
    for t in ("headings and paragraphs in DM Sans", "body text and titles: DM Sans", "the buttons and nav in DM Sans"):
        assert not R.generic_font_request(c(t)), t


def test_to_all_fonts_on_a_state_and_its_scoring():
    m = load(str(EXAM_MANIFEST))
    case = {**_font_case("I'd like Lora as the font"), "site_state": _vintage()}
    case["expected"]["ops"] = [_set(TYPO, bodyFont="Lora", headingFont="Lora")]
    allc, note = R.apply_font_policy(case, "all", m)
    assert note == "font:all"
    sf = allc["expected"]["site_fonts"]
    assert sf["value"] == "Lora" and sf["required"] == sf["must"] and len(sf["required"]) == 7
    assert {o["target"].get("panel") for o in allc["expected"]["ops"]} == {"TypographyColorsSection", "ProductCardSection",
                                                                          "HeaderSection"}
    assert allc["meta"]["oracle_ops"] == allc["expected"]["ops"]
    assert R.apply_font_policy(allc, "all") == (allc, "font:all")  # idempotent
    every = [_set({"kind": "global"}, **{k: "Lora" for k in sf["required"]})]
    nested = [_set(TYPO, headingFont="Lora", bodyFont="Lora", navFont="Lora", buttonFont="Lora"),
              _set({"kind": "global", "panel": "ProductCardSection"}, productTitleFont="Lora", productBodyFont="Lora"),
              _set({"kind": "global", "panel": "HeaderSection"}, announcementBar={"fontFamily": "Lora"})]
    assert R.score(case, _row(every, [True]))["verdict"] == "pass"
    assert R.score(case, _row(nested, [True, True, True]))["verdict"] == "pass"
    # the inheriting cart font may be written too; a hidden / unrelated field may not; one font is not every font
    plus = [_set({"kind": "global"}, **{k: "Lora" for k in sf["required"]}, cartTitleFont="Lora")]
    assert R.score(case, _row(plus, [True]))["verdict"] == "pass"
    hidden = [_set({"kind": "global"}, **{k: "Lora" for k in sf["required"]}, shopMenuFont="Lora")]
    assert R.score(case, _row(hidden, [True]))["verdict"] == "extra_writes"
    pair = [_set(TYPO, bodyFont="Lora", headingFont="Lora")]
    s = R.score(case, _row(pair, [True]))
    assert s["verdict"] == "missing_writes" and "missing:navFont" in s["reasons"]
    assert R.score(case, _row([_set(TYPO, bodyFont="Lora")], [True]))["verdict"] == "missing_writes"
    wrong = [_set({"kind": "global"}, **{k: "Oswald" for k in sf["required"]})]
    assert R.score(case, _row(wrong, [True]))["verdict"] == "wrong_value"
    assert R.score(case, _row(every, [False]))["verdict"] == "invalid_op"
    assert R.score(case, _row([], []))["verdict"] == "no_answer"
    # the r4 / r5 key is still there for re-scoring old runs
    assert R.score(case, _row(pair, [True]), policy={"font": "as_authored"})["verdict"] != "missing_writes"
    # a font already in place need not be written again: "Inter" on vintage leaves only the two that differ
    inter = {**case, "expected": {**case["expected"], "ops": [_set(TYPO, bodyFont="Inter", headingFont="Inter")]}}
    inter["site_state"] = copy.deepcopy(case["site_state"])
    inter["site_state"]["globalSettings"].update(bodyFont="Lora", headingFont="Lora")
    sfi = R.apply_font_policy(inter, "all", m)[0]["expected"]["site_fonts"]
    assert sfi["must"] == ["headingFont", "bodyFont"] and len(sfi["required"]) == 7


def test_a_named_font_area_still_means_that_field_alone():
    c = _font_case("headings in DM Sans")
    c["expected"]["ops"] = [_set(TYPO, headingFont="DM Sans")]
    assert R.apply_font_policy(c, "all") == (c, None)
    # a bare "all" does not undo the parts the merchant named; "every font" words do
    both = _font_case("set the body and headings all to DM Sans")
    both["expected"]["ops"] = [_set(TYPO, bodyFont="DM Sans", headingFont="DM Sans")]
    assert R.apply_font_policy(both, "all") == (both, None)
    for t in ("DM Sans across the site", "all the site's text in DM Sans", "every font in DM Sans, headings too"):
        assert R.generic_font_request(_font_case(t)), t


def test_an_open_font_choice_is_one_font_and_may_be_asked():
    m = load(str(EXAM_MANIFEST))
    case = {**_font_case("i don't like the font", mode="keys"), "site_state": _vintage()}
    case["expected"]["ops"] = []  # no font named: the answer key has no value
    open_ = R.to_all_fonts(case, m, None)
    sf = open_["expected"]["site_fonts"]
    assert sf["value"] is None and len(sf["must"]) == 7
    one = [_set({"kind": "global"}, **{k: "Lora" for k in sf["required"]})]
    assert R.score(open_, _row(one, [True]))["verdict"] == "pass"
    mixed = [_set({"kind": "global"}, **{k: ("Lora" if i else "Oswald") for i, k in enumerate(sf["required"])})]
    s = R.score(open_, _row(mixed, [True]))
    assert s["verdict"] == "wrong_value" and s["reasons"] == ["wrong_value:headingFont"]
    asked = _row([], [], text="Sure — which font would you like instead?")
    assert R.score(open_, asked)["verdict"] == "no_answer"  # asking passes only where the key allows it
    open_["expected"]["ask_ok"] = True
    assert R.score(open_, asked)["verdict"] == "clarify_ok"
    assert R.score(open_, _row([], [], text="I can't."))["verdict"] == "no_answer"


def test_amended_exam_rows_keep_their_key_under_all():
    from souschef_lab import exam as X
    h059 = next(r for r in X.load_set("holdout") if r["id"] == "h059.k2")
    out, note = R.apply_font_policy(h059, "all")
    assert out is h059 and note == "font:all"  # body + heading required, every font the alternative
    assert [k["key"] for k in out["expected"]["keys"]] == ["bodyFont", "headingFont"] and len(out["expected"]["alt"]) == 1


def test_amended_exam_rows_go_back_to_the_r5_key_under_as_authored():
    from souschef_lab import exam as X
    row = next(r for r in X.load_set("followup") if r["id"] == "p150.k2")
    assert row["meta"]["answer_key"] == X.ANSWER_KEY and R.site_font_case(row) == "all"
    assert R.apply_font_policy(row, "all") == (row, "font:all")
    old, note = R.apply_font_policy(row, "as_authored")
    assert note == "font:as_authored" and old["id"] == "p150" and [k["key"] for k in old["expected"]["keys"]] == ["bodyFont"]
    assert old["fields"] == ["bodyFont"] and "optional" not in old["expected"]
    assert old == next(r for r in X.load_set("followup", amended=False) if r["id"] == "p150")
    wide, note = R.apply_font_policy(row, "body_or_both")  # the r6-2 option on the r5 key (its alt already
    assert wide == old                                     # accepts headingFont: nothing to widen)


# ── asking before acting ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_ask_mode_rewards_a_question_and_penalises_a_guess():
    case = {"request": "the headings look too plain", "history": [], "score": {"mode": "ask"},
            "context": {"page": "home"}, "expected": {"ops": [], "keys": []}, "meta": {}}
    asked = _row([], [], text="Happy to — would you like a different font, a bigger size or a new colour?")
    s = R.score(case, asked)
    assert s["verdict"] == "clarify_ok" and s["passed"] and s["usable"]
    guess = _row([_set(TYPO, headingFont="Playfair Display")], [True], text="Done — Playfair Display. Want more?")
    s = R.score(case, guess)
    assert s["verdict"] == "wrote" and not s["passed"] and "acted_on_a_vague_request" in s["reasons"]
    tried = _row([_set(TYPO, headingFont="Nope")], [False], text="That failed — which font?")
    assert R.score(case, tried)["verdict"] == "wrote"  # a refused write is still a guess
    told = _row([], [], text="I've noted that.")
    assert R.score(case, told)["verdict"] == "no_question"
    for t in ("Noted! Anything else?", "Got it. Is there anything else I can help with?"):
        assert R.score(case, _row([], [], text=t))["verdict"] == "no_question", t  # a sign-off asks nothing
    assert R.score(case, _row([], [], text="Sure. Bigger, bolder or a new font? Anything else?"))["verdict"] == "clarify_ok"
    unfinished = {**asked, "stop": "max_tokens"}
    assert R.score(case, unfinished)["verdict"] == "no_answer"


def test_acted_first_turn_stays_off_in_collect_and_the_eval(monkeypatch):
    monkeypatch.delenv("SOUSCHEF_SCORE_POLICY", raising=False)
    assert R.DEFAULT_POLICY["acted_first_turn"] == "off" and R.STRICT_POLICY["acted_first_turn"] == "off"
    assert R.adopt_trigger_policy(None, env={})["acted_first_turn"] == "off"
    assert R.adopt_trigger_policy(None, env={}, base=R.STRICT_POLICY)["acted_first_turn"] == "off"


def test_prepare_fixes_generic_font_cases_under_all_only():
    from souschef_lab.retrain.prepare import fix_site_fonts
    m = load(str(EXAM_MANIFEST))
    st = _vintage()
    gen = {**_font_case("make the whole site DM Sans"), "id": "g", "site_state": st}
    named = {**_font_case("headings and paragraphs in DM Sans"), "id": "n", "site_state": st}
    hold = {**copy.deepcopy(gen), "id": "h", "holdout": True}
    cases = [gen, named, hold]
    rep = fix_site_fonts(cases, m, "all")
    assert rep["count"] == 1 and rep["ids"] == ["g"]
    assert cases[0]["expected"]["site_fonts"]["value"] == "DM Sans" and "site_fonts" in cases[0]["meta"]["fixed"]
    assert "site_fonts" not in cases[1]["expected"] and "site_fonts" not in cases[2]["expected"]
    again = [{**_font_case("make the whole site DM Sans"), "id": "g", "site_state": st}]
    assert fix_site_fonts(again, m, "as_authored")["count"] == 0 and "site_fonts" not in again[0]["expected"]
