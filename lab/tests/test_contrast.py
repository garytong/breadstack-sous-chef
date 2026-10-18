"""contrast.py — family F8, the look-alike-field contrast cases (r5why PLAN §2b)."""
from __future__ import annotations

import json
import re

import pytest

from souschef_lab import FIXTURES_DIR
from souschef_lab import askrule as AR
from souschef_lab import contrast as K
from souschef_lab import engine as E
from souschef_lab.leak import ExamIndex
from souschef_lab.manifest import is_holdout, load
from souschef_lab import r4score as R4S
from souschef_lab.r4cases import FAMILY_ORDER, family_of_row, write_ids

EXAM_MANIFEST = FIXTURES_DIR / "exam" / "manifest-real.json"


@pytest.fixture(scope="module")
def m():
    return load(str(EXAM_MANIFEST))


def _need_engine():
    if not E.engine_available():
        pytest.skip("node and the engine (COOKIE_CUTTER_DIR) are needed: " + E.unavailable_reason())


def test_every_template_is_built_from_field_metadata(m):
    assert K.check_templates(m) == []
    # the check bites: a template with no label and no metadata word
    bad = K.Side("x", "bad", "global", "TypographyColorsSection", "linkColor", (("literal", "zap zing {v}"),), n=1)
    assert K.check_templates(m, (bad,)) and "shares no metadata word" in K.check_templates(m, (bad,))[0]
    missing = K.Side("x", "gone", "global", "TypographyColorsSection", "noSuchKey", (("literal", "{label} {v}"),), n=1)
    assert "no field" in K.check_templates(m, (missing,))[0]


def test_the_pairs_the_analysis_lists_are_covered():
    keys = {(s.pair, s.key) for s in K.SIDES}
    for want in [("links", "linkColor"), ("links", "shopMenuTextColor"), ("links", "navFont"), ("links", "shopMenuFontSize"),
                 ("fonts", "bodyFont"), ("fonts", "bodyFontFamily"),
                 ("names", "pageVisibility.{page}.headerNavName"), ("names", "pageVisibility._meta.headerNavCustomLinks"),
                 ("names", "orderTypeText.{page}.title"), ("grid", "cardsPerRow"), ("grid", "gridCols"),
                 ("footer", "footerTextColor"), ("footer", "footerBodyTextColor")]:
        assert want in keys, want
    assert all(len([s for s in K.SIDES if s.pair == p]) >= 2 for p in {s.pair for s in K.SIDES})
    assert 120 <= sum(s.n for s in K.SIDES) <= 170


def test_names_and_values_come_from_the_manifest(m):
    assert K.nav_name(m, "HeaderSection", "products") == "Shop"
    assert K.nav_name(m, "FooterSection", "faq") == "FAQ"
    assert K.option_name(m.page("checkout").fields["orderTypeText.local_delivery.title"]) == "delivery"
    f = m.panel("ProductCardSection").fields["cardsPerRow"]
    for i in range(20):
        v = K.pick_value(f, 3, f"s{i}")
        assert v in f.enum and v != 3
    c = m.panel("TypographyColorsSection").fields["linkColor"]
    assert K.pick_value(c, K.PALETTE[0], "x") != K.PALETTE[0]
    assert K.shift_color("#808080", True) < "#808080" < K.shift_color("#808080", False)


def test_a_holdout_key_and_an_exam_text_are_dropped_before_anything_runs(m, tmp_path):
    # TypographyColorsSection.navFontSize is a generator hold-out key (the F0H exam's unseen grain)
    assert is_holdout("TypographyColorsSection.navFontSize")
    s = K.Side("links", "held", "global", "TypographyColorsSection", "navFontSize", (("literal", "{label} {v}px"),), n=1)
    case, why = K.build_case(m, s, 0, "literal", "{label} {v}px", "t-F8-0001")
    assert why == "" and K.validate(case, m, engine=None, extras={}).startswith("holdout_key:")
    # the leak filter: an exam text identical to the generated request drops it
    side = next(x for x in K.SIDES if x.name == "card_cards_per_row")
    case, _ = K.build_case(m, side, 0, *side.templates[0], "t-F8-0002")
    exam = tmp_path / "exam.jsonl"
    exam.write_text(json.dumps({"id": "x1", "request": case["request"], "expected": {"keys": []}}) + "\n")
    idx = ExamIndex(exam_dir=tmp_path)
    assert idx.check(case)["rule1"]


def test_build_case_shapes(m):
    side = next(x for x in K.SIDES if x.name == "shop_menu_link_color")
    case, why = K.build_case(m, side, 3, "focus", "{label}: {v}", "c-F8-0001")
    assert why == ""
    assert case["meta"]["family"] == "F8" and case["meta"]["view"] == {"tab": "settings", "globalSection": "shopMenu"}
    assert case["score"]["mode"] == "exact" and case["request"].startswith("link color: #")
    assert case["expected"]["ops"][0]["values"].keys() == {"shopMenuTextColor"}
    rel = next(x for x in K.SIDES if x.name == "shop_menu_link_size")
    case, why = K.build_case(m, rel, 0, "relative", "the links in the shop menu are a bit small, bump them up", "c-F8-0002")
    assert case["score"]["mode"] == "keys" and case["expected"]["direction"]["dir"] == "up"
    lst = next(x for x in K.SIDES if x.name == "header_custom_link")
    case, why = K.build_case(m, lst, 0, *lst.templates[0], "c-F8-0003")
    assert case["expected"]["list_ops"] == [{"list": "pageVisibility._meta.headerNavCustomLinks", "action": "add", "index": None}]
    assert family_of_row(case, "F1") == "F8" and "F8" in FAMILY_ORDER


def test_contrast_round_end_to_end(m):
    _need_engine()
    eng = E.PersistentEngine(manifest_path=EXAM_MANIFEST, size=2)
    try:
        index = ExamIndex()
        cases, rep = K.contrast_round(m, eng, leak_index=index, progress=lambda s: None)
        again, _ = K.contrast_round(m, eng, leak_index=index, sides=K.SIDES[:3], progress=lambda s: None)
    finally:
        eng.close()
    assert rep["template_problems"] == []
    assert 120 <= rep["kept"] == len(cases) <= 300, rep
    # the capability groups ride along; an engine without create_page keeps only their rename/delete, add-a-section
    # and listing cases (every create and decline is measured on the engine: page_oracle_refused / E_OP)
    assert set(rep["pairs"]) == {s.pair for s in K.SIDES} | {"site_fonts", "vague", "create_page", "list_options"}
    assert rep["site_fonts"] >= 8 and rep["vague"] >= 12, rep
    caps = rep["capabilities"]
    assert caps["list_options"]["cases"] >= 2 and caps["create_page"]["modes"].get("where") == 4, caps
    if caps["create_page"]["modes"].get("page"):
        assert caps["create_page"]["modes"]["page"] >= 30, caps  # the engine knows create_page: creates and declines
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == len(ids) and all(re.fullmatch(r"contrast1-F8-\d{4}", i) for i in ids)
    for c in cases:
        assert c["meta"]["family"] == "F8" and not c["holdout"]
        assert c["score"]["mode"] in ("exact", "keys", "ask", "page", "where", "list")
        assert not any(is_holdout(k) for k in write_ids(c, m))
        pair = c["meta"]["contrast"]["pair"]
        if pair in ("create_page", "list_options"):
            assert c["meta"]["capability"] == pair and c["meta"]["generator"] in (K.PAGE_GENERATOR, K.LIST_GENERATOR)
            assert "events" not in c["request"].lower()  # exam p137's page: never taught (page_exam_clash)
            if c["score"]["mode"] == "page" and c["meta"]["oracle_ops"]:
                assert c["meta"]["measured"]["written"][0].startswith("customPages[")
            if c["score"]["mode"] == "page" and not c["meta"]["oracle_ops"]:
                assert c["meta"]["measured"]["refusal"] in ("E_SLUG", "E_TARGET")
            if c["score"]["mode"] == "list":
                assert R4S.listing_view(c)["situation"] == c["expected"]["listing"]["situation"]
            continue
        assert (c["score"]["mode"] == "ask") == (pair == "vague")
        if pair == "vague":  # nothing to write: the answer is a question
            assert c["expected"]["ops"] == [] and c["meta"]["oracle_ops"] == [] and "{" not in c["request"]
        if pair == "site_fonts":  # every site font, one value (r4score.site_font_fields on the case's state)
            sf = c["expected"]["site_fonts"]
            assert {"headingFont", "bodyFont", "navFont", "buttonFont", "productTitleFont", "productBodyFont"} <= set(sf["required"])
            assert {k for o in c["expected"]["ops"] for k in o["values"]} == set(sf["required"])
            assert {v for o in c["expected"]["ops"] for v in o["values"].values()} == {sf["value"]}
            assert sf["value"].lower() not in ("lora", "inter")  # the exam's font answers (exam_answer_clash)
        res = index.check(c)
        assert not (res["rule1"] or res["rule2"] or res["rule3"]) and res["max_jaccard"] < K.NEAR_EXAM
        if c["score"]["mode"] == "exact":
            assert c["expected"]["diff"], c["id"]
        assert "preview" not in c["request"].lower()
    # deterministic: the same manifest gives the same cases
    first = {c["id"]: c["request"] for c in cases}
    assert all(first.get(c["id"]) == c["request"] for c in again)
    # r7 §2: every vague (ask) case has its write twins, same field, same opening words, same split group
    by_id = {c["id"]: c for c in cases}
    vague = [c for c in cases if c["meta"]["contrast"]["pair"] == "vague"]
    assert rep["vague_twins"] >= 2 * 12, rep
    for v in vague:
        pair = v["meta"]["ask_pair"]
        assert pair["role"] == "ask" and v["meta"]["ask_family"] == "complaint"
        twins = [by_id[t] for t in pair["twins"] if t in by_id]
        assert twins, v["id"]
        for t in twins:
            assert t["request"].startswith(v["request"] + ", ") and t["meta"]["group"] == v["meta"]["group"]
            assert t["meta"]["ask_pair"] == {"twin": v["id"], "role": "write"}
            assert t["meta"]["contrast"]["key"] == v["meta"]["contrast"]["key"]
            assert (t["score"]["mode"] == "keys" and t["expected"]["direction"]) or t["score"]["mode"] in ("exact", "keys")
            assert int(t["id"][-4:]) == K.vague_twin_id(int(v["id"][-4:]), pair["twins"].index(t["id"]))
            assert not AR.classify(t["request"], values=AR.font_names(m) + ["2", "5"]).vague
        assert AR.classify(v["request"]).vague
    assert all(first.get(t["id"]) == t["request"] for t in again if t["meta"].get("generator") == K.VAGUE_TWIN_GENERATOR)


def test_a_case_that_teaches_an_exam_items_answer_is_dropped(m):
    # h019 "rename Shop in the navigation to Menu" (HeaderSection pageVisibility.products.headerNavName). r6 review: an
    # F8 case "change Shop to 'Menu' in the footer only" passed the Jaccard margin (0.33) but teaches h019's answer
    assert "Menu" not in dict(K.NAV_NAMES)["products"]
    idx = ExamIndex()
    answers = K.exam_answers(idx)
    h019 = [a for a in answers if str(a["id"]).endswith(":h019")]
    assert h019 and h019[0]["shape"] == "pageVisibility.{page}.headerNavName" and h019[0]["page"] == "products"
    assert "menu" in h019[0]["values"]
    footer = next(x for x in K.SIDES if x.name == "footer_nav_name")
    header = next(x for x in K.SIDES if x.name == "header_nav_name")

    def case_for(side, page, value):
        return {"meta": {"contrast": {"side": side.name, "pair": side.pair, "page_key": page, "value": value}}}
    # the look-alike field (footer) with the exam's value, or for the exam's page, is dropped
    assert K.exam_answer_clash(case_for(footer, "contact", "Menu"), h019) == "exam_answer:holdout:h019:value"
    assert K.exam_answer_clash(case_for(footer, "products", "Browse"), h019) == "exam_answer:holdout:h019:page"
    assert K.exam_answer_clash(case_for(header, "products", "Shop All"), h019).endswith(":page")
    # another page and another value, or another pair, are kept
    assert K.exam_answer_clash(case_for(footer, "contact", "Say Hello"), h019) is None
    other = next(x for x in K.SIDES if x.name == "site_link_color")
    assert K.exam_answer_clash(case_for(other, None, "menu"), h019) is None
    # every side of the real family against the real exam: no kept products nav-name case, no "Menu"
    built = []
    for side in (header, footer):
        for i in range(side.n):
            style, tpl = side.templates[(i + K.h8(side.name)) % len(side.templates)]
            c, why = K.build_case(m, side, i, style, tpl, f"t-F8-{i:04d}")
            if c is not None and not K.exam_answer_clash(c, answers):
                built.append(c)
    assert built and all(c["meta"]["contrast"]["page_key"] != "products" for c in built)
    assert not any("menu" == K._norm_value(c["meta"]["contrast"]["value"]) for c in built)


def test_exam_answers_read_generated_exam_cases_too():
    tgt = {"kind": "global", "panel": "ProductCardSection"}
    f0h = {"id": "r4-F0H-0001", "request": "4 cards per row", "history": [],
           "expected": {"ops": [{"op": "set_fields", "target": tgt, "values": {"cardsPerRow": 4}}], "keys": []}}
    ans = K.exam_answers(None, [f0h])
    assert ans == [{"id": "r4-F0H-0001", "key": "cardsPerRow", "shape": "cardsPerRow", "page": None, "values": {"4"}}]
    grid = next(x for x in K.SIDES if x.name == "shop_grid_columns")   # the look-alike of cardsPerRow
    c = {"meta": {"contrast": {"side": grid.name, "pair": grid.pair, "page_key": None, "value": 4}}}
    assert K.exam_answer_clash(c, ans) == "exam_answer:r4-F0H-0001:value"
    c["meta"]["contrast"]["value"] = 3
    assert K.exam_answer_clash(c, ans) is None


def test_vague_cases_name_the_place_and_ask(m):
    side = next(s for s in K.SIDES if s.name == "shop_menu_font")
    c, why = K.build_vague_case(m, side, "the {label} needs some work", "x")
    assert why == "" and c["request"] == "the shop menu font needs some work"  # never "the font" alone (= every font)
    assert c["score"]["mode"] == "ask" and c["kind"] == "ask" and c["expected"]["keys"] == []
    page_side = next(s for s in K.SIDES if s.name == "header_nav_name")
    assert K.build_vague_case(m, page_side, "the {label} needs some work", "y") == (None, "vague_unsupported_side")


def test_ask_first_cases_from_the_follow_up_familys_vague_first_turns(m):
    from souschef_lab.state import build_state, load_context_fixture
    state = build_state(m, template="vintage")
    base = {"target": {"kind": "global", "panel": "TypographyColorsSection"}, "fields": ["headingFont"],
            "site_state": state, "context": {"page": "home"}, "score": {"mode": "exact"}, "holdout": False,
            "manifest_hash": m.hash, "template": "vintage", "expected": {"ops": []}}
    pool = [
        {**base, "id": "f2-a", "request": "Oswald", "meta": {"family": "F2", "group": "g1", "page": "home"},
         "history": [{"user": "can we rethink the headings", "expect": "ask", "assistant": "Which font would you like?"}]},
        {**base, "id": "f2-b", "request": "yes", "meta": {"family": "F2", "group": "g2", "page": "home"},
         "history": [{"user": "can we rethink the headings", "expect": "offer", "assistant": "Bigger?"}]},  # same text
        {**base, "id": "f2-c", "request": "the hero", "meta": {"family": "F2", "group": "g3"},
         "history": [{"user": "make it pop", "expect": "write", "ops": []}]},                                # acted on
        {**base, "id": "f2-d", "request": "x", "holdout": True, "meta": {"family": "F2"},
         "history": [{"user": "the footer is dull", "expect": "ask"}]},                                      # a holdout
    ]
    out, rep = K.ask_first_cases(pool, m, None, load_context_fixture(), seen=set(), pairs=False)
    assert [c["id"] for c in out] == ["askfirst1-F2-f2-a"] and rep["dropped"] == {"duplicate": 1}
    c = out[0]
    assert c["request"] == "can we rethink the headings" and c["history"] == [] and c["score"]["mode"] == "ask"
    assert c["meta"]["group"] == "g1" and c["meta"]["family"] == "F2" and c["meta"]["ask_first"]["from"] == "f2-a"
    assert c["context"]["page"] == "home" and c["expected"]["ops"] == []


def test_vague_twins_are_built_from_the_vague_text_by_field_type(m):
    side = next(s for s in K.SIDES if s.name == "footer_text_color")
    tpl = "something is off about the {label}"
    ask, why = K.build_vague_case(m, side, tpl, "contrast1-F8-0914", twins=("contrast1-F8-1227", "contrast1-F8-1228"))
    assert why == "" and ask["request"] == "something is off about the footer text color"
    assert ask["meta"]["ask_pair"] == {"twin": "contrast1-F8-1227", "twins": ["contrast1-F8-1227", "contrast1-F8-1228"],
                                       "role": "ask"}
    rel, _ = K.build_vague_twin(m, side, tpl, 0, "contrast1-F8-1227", ask["id"])
    lit, _ = K.build_vague_twin(m, side, tpl, 1, "contrast1-F8-1228", ask["id"])
    assert rel["request"].startswith(ask["request"] + ", ") and lit["request"].startswith(ask["request"] + ", try #")
    assert rel["score"]["mode"] == "keys" and rel["expected"]["direction"]["key"] == "footerTextColor"
    want = AR.direction_of(rel["request"], "color")
    assert rel["expected"]["direction"]["dir"] == want and want in ("darker", "lighter")
    assert lit["score"]["mode"] == "exact" and lit["expected"]["keys"][0]["key"] == "footerTextColor"
    for t in (rel, lit):
        assert t["meta"]["group"] == ask["meta"]["group"] == "vague:footer_text_color"
        assert t["meta"]["ask_pair"] == {"twin": ask["id"], "role": "write"}
    # the same on another manifest: ids are the caller's (VAGUE_TWIN_BASE), texts the template's
    snap = load(str(FIXTURES_DIR / "manifest.snapshot.json"))
    rel2, _ = K.build_vague_twin(snap, side, tpl, 0, "contrast1-F8-1227", ask["id"])
    assert rel2["id"] == rel["id"] and rel2["request"] == rel["request"]
    assert K.vague_twin_id(901, 0) == 1201 and K.vague_twin_id(920, 1) == 1240
    # a number twin: relative up / down or a px value; a font twin: a font of the field's list
    size = next(s for s in K.SIDES if s.name == "logo_height_mobile")
    n0, _ = K.build_vague_twin(m, size, tpl, 0, "t-1", "a")
    n1, _ = K.build_vague_twin(m, size, tpl, 1, "t-2", "a")
    assert n0["expected"]["direction"]["dir"] in ("up", "down") and n1["request"].endswith("px")
    # r7 review: a colour / number side's third twin goes the OTHER way from twin 0 (ids from VAGUE_TWIN2_BASE)
    other, _ = K.build_vague_twin(m, side, tpl, 2, "contrast1-F8-1314", ask["id"])
    assert other["expected"]["direction"]["dir"] in ("darker", "lighter")
    assert other["expected"]["direction"]["dir"] != rel["expected"]["direction"]["dir"]
    assert other["request"].startswith(ask["request"] + ", ") and other["meta"]["group"] == ask["meta"]["group"]
    n2, _ = K.build_vague_twin(m, size, tpl, 2, "t-5", "a")
    assert {n0["expected"]["direction"]["dir"], n2["expected"]["direction"]["dir"]} == {"up", "down"}
    assert K.vague_twin_id(914, 2) == 1314 and K.twin_count(m, side) == 3 and K.twin_count(m, size) == 3
    font = next(s for s in K.SIDES if s.name == "shop_menu_font")
    assert K.twin_count(m, font) == 2 and K.build_vague_twin(m, font, tpl, 2, "t-6", "a") == (None, "twin_unsupported_type")
    f0, _ = K.build_vague_twin(m, font, tpl, 0, "t-3", "a")
    assert f0["request"].startswith("something is off about the shop menu font, use ") and f0["score"]["mode"] == "exact"


def test_a_vague_template_that_drifts_into_a_direction_is_dropped(m):
    side = next(s for s in K.SIDES if s.name == "footer_text_color")
    assert K.build_vague_case(m, side, "the {label} is too light", "x")[1].startswith("ask_rule:direction")
    assert K.build_vague_case(m, side, "the {label} should be #000000", "x")[1].startswith("ask_rule:value")


def _f2(m, cid, first, request, ops, *, mode="exact", target=None):
    from souschef_lab.state import build_state
    return {"id": cid, "target": target or (ops[0]["target"] if ops else None), "fields": [], "request": request,
            "site_state": build_state(m, template="vintage"), "context": {"page": "home"}, "score": {"mode": mode},
            "holdout": False, "manifest_hash": m.hash, "template": "vintage",
            "expected": {"ops": ops, "diff": [], "keys": [], "alt": [], "alt_exact": []},
            "meta": {"family": "F2", "group": f"g:{cid}", "page": "home", "oracle_ops": ops},
            "history": [{"user": first, "expect": "ask", "assistant": "What would you like?", "page": "home", "ops": []}]}


def test_ask_first_relabels_directed_first_turns_and_pairs_the_vague_ones(m):
    """r7 §1.2 / §2.1: "less padding on the content block, 24px" is a write (askfirst1r, keys) and no ask; "can you
    update the announcement bar" stays ask with a write twin "…, black background please" in its split group."""
    from souschef_lab.r4cases import remeasure
    from souschef_lab.state import load_context_fixture
    _need_engine()
    cb = {"kind": "section", "page": "home", "sectionId": "contentBlock-1"}
    hdr = {"kind": "global", "panel": "HeaderSection"}
    foot = {"kind": "global", "panel": "FooterSection"}
    pool = [
        _f2(m, "f2-pad", "less padding on the content block, 24px", "yes 24",
            [{"op": "set_fields", "target": cb, "values": {"paddingY": 24}}]),
        _f2(m, "f2-ann", "can you update the announcement bar", "black background please",
            [{"op": "set_fields", "target": hdr, "values": {"announcementBar.bgColor": "#000000"}}], mode="keys"),
        _f2(m, "f2-yes", "can we rethink the footer", "sure, go ahead",
            [{"op": "set_fields", "target": foot, "values": {"footerTextColor": "#222222"}}], mode="keys"),
        _f2(m, "f2-two", "make the footer text darker", "and the links",
            [{"op": "set_fields", "target": foot, "values": {"footerTextColor": "#222222",
                                                              "footerLinkHighlightColor": "#111111"}}], mode="keys"),
        # r7 review: a clear turn with no evidence for the source's write ("pop more" names no way for a colour) …
        _f2(m, "f2-pop", "make the footer text pop more", "darker",
            [{"op": "set_fields", "target": foot, "values": {"footerTextColor": "#222222"}}], mode="keys"),
        # … and one text relabelled to two different fields: neither is the text's
        _f2(m, "f2-c1", "the footer looks too light", "the text",
            [{"op": "set_fields", "target": foot, "values": {"footerTextColor": "#222222"}}], mode="keys"),
        _f2(m, "f2-c2", "the footer looks too light", "the links",
            [{"op": "set_fields", "target": foot, "values": {"footerLinkHighlightColor": "#111111"}}], mode="keys"),
    ]
    eng = E.PersistentEngine(manifest_path=EXAM_MANIFEST, size=2)
    try:
        extras = load_context_fixture()
        for c in pool:
            assert remeasure(c, eng, m, extras) is None, c["id"]
        out, rep = K.ask_first_cases(pool, m, eng, extras, seen=set())
    finally:
        eng.close()
    ids = {c["id"]: c for c in out}
    assert set(ids) == {"askfirst1r-F2-f2-pad", "askfirst1-F2-f2-ann", "askpair1-F2-f2-ann", "askfirst1-F2-f2-yes"}, ids
    pad = ids["askfirst1r-F2-f2-pad"]
    assert pad["score"]["mode"] == "keys" and pad["request"] == "less padding on the content block, 24px"
    assert pad["expected"]["keys"] == [{"target": cb, "key": "paddingY", "list": ""}]
    assert pad["expected"]["direction"] == {"key": "paddingY", "dir": "down", "from": None}
    assert pad["meta"]["ask_rule"]["kind"] == "value" and pad["meta"]["group"] == "g:f2-pad"
    # the vintage home has several content blocks: the write case focuses the source's one, so it is not a "which one?"
    from souschef_lab.retrain.author import ambiguity
    assert pad["meta"]["view"]["activeSectionId"] == "contentBlock-1" and pad["meta"]["ask_rule"]["focused"] == "which_one"
    assert ambiguity(pad, m) is None
    ask, twin = ids["askfirst1-F2-f2-ann"], ids["askpair1-F2-f2-ann"]
    assert ask["score"]["mode"] == "ask" and ask["meta"]["ask_pair"] == {"twin": twin["id"], "role": "ask"}
    assert ask["meta"]["ask_family"] in ("choice_missing", "text_missing")
    assert twin["request"] == "can you update the announcement bar, black background please"
    assert twin["score"]["mode"] == "keys" and twin["meta"]["ask_pair"] == {"twin": ask["id"], "role": "write"}
    assert twin["meta"]["group"] == ask["meta"]["group"] == "g:f2-ann" and twin["history"] == []
    # the merged text still vague: no twin (the dataset then drops the unpaired ask row)
    assert "ask_pair" not in ids["askfirst1-F2-f2-yes"]["meta"]
    assert rep["relabelled"] == 1 and rep["kept_vague"] == 2 and rep["twins"] == 1
    assert rep["dropped_rule"] == {"no_field": 1, "unsure": 1, "conflicting_keys": 2}
    assert rep["twins_dropped"] == {"still_vague": 1}
    assert pad["meta"]["ask_rule"]["evidence"] == "oracle:24"


# --- r6-4: the create_page and list_options groups ---------------------------------------------------------------------

def test_page_cases_are_built_from_the_jobs(m):
    from souschef_lab import exam as X
    jobs = {g + ":" + r: spec for g, r, spec in K.PAGE_JOBS}
    assert {g for g, _, _ in K.PAGE_JOBS} == {"create", "slug", "draft", "in_nav", "sections", "exists", "exists_sections",
                                              "reserved", "noncanonical", "template", "manage", "section_not_page",
                                              "social_page"}
    assert 40 <= len(K.PAGE_JOBS) <= 50 and 30 <= len(K.LIST_JOBS) <= 40  # modest: ≈ US$0.03 a case asked
    c, why = K.build_page_case(m, "sections", "create a Budtender Picks page with a product carousel on it",
                               jobs["sections:create a Budtender Picks page with a product carousel on it"], "t-F8-1")
    assert why == "" and c["score"]["mode"] == "page" and c["expected"]["page"]["sections"] == ["productCarousel"]
    assert c["meta"]["oracle_ops"] == [
        {"op": "create_page", "title": "Budtender Picks", "slug": "budtender-picks"},  # same batch: the slug passed
        {"op": "edit_list", "target": {"kind": "page", "page": "budtender-picks"}, "list": "sections", "action": "add",
         "item": {"id": "productCarousel"}}]
    deals = jobs["exists:make me a Deals page"]
    c, _ = K.build_page_case(m, "exists", "make me a Deals page", deals, "t-F8-2")
    assert c["meta"]["oracle_ops"] == [] and c["meta"]["attempt"] == [{"op": "create_page", "title": "Deals"}]
    assert any(p["slug"] == "deals" for p in c["site_state"]["customPages"])
    assert X.page_expectation(c)["expect"] == "decline" and X.page_expectation(c)["existing"]["slug"] == "deals"
    tpl, _ = K.build_page_case(m, "template", "add a new page called Gift Cards", {"title": "Gift Cards", "why": "template"}, "t3")
    assert tpl["template"] == K.PREMIUM_TEMPLATE == tpl["site_state"]["template"]["id"] and tpl["expected"]["page"]["why"] == "template"
    nc, _ = K.build_page_case(m, "noncanonical", "x", {"title": "Wholesale Inquiries", "asked_slug": "Wholesale_Inquiries"}, "t4")
    assert nc["expected"]["page"]["slugs"] == ["wholesale-inquiries", "wholesaleinquiries"]
    assert nc["meta"]["oracle_ops"][0]["slug"] == "wholesale-inquiries"
    mg, _ = K.build_page_case(m, "manage", "delete the Deals page", {"where": "pages", "pages": ["deals"]}, "t5")
    assert mg["score"]["mode"] == "where" and mg["expected"]["out_of_scope"]["expected_where"] == ["pages"]
    sp, _ = K.build_page_case(m, "section_not_page", "x", {"section": {"page": "contact", "type": "newsletter"}}, "t6")
    assert sp["score"]["mode"] == "keys" and sp["expected"]["list_ops"] == [{"list": "sections", "action": "add", "index": None}]
    # a page case on exam p137's page is a leak of its answer
    ev, _ = K.build_page_case(m, "create", "make an Events page", {"title": "Events"}, "t7")
    assert K.page_exam_clash(ev) == "exam_answer:out_of_scope:p137:page" and K.page_exam_clash(c) is None


def test_list_cases_carry_their_lists_in_context_extras(m):
    from souschef_lab.r4cases import with_case_extras
    jobs: dict = {}
    for j in K.LIST_JOBS:
        jobs.setdefault((j[1], j[2]), (j[0], j[3]))  # the first job of each (list, situation)
    assert {j[2] for j in K.LIST_JOBS} == {"short", "long", "truncated", "empty", "unknown"}
    from collections import Counter
    per = Counter(j[2] for j in K.LIST_JOBS)
    assert min(per.values()) >= 6 and sum(1 for j in K.LIST_JOBS if j[4:] == ("plain",)) >= 10  # every situation, plain asks
    tag_n, brand_n = len(K.TAG_A) * len(K.TAG_B), len(K.BRAND_A) * len(K.BRAND_B)
    assert all((j[3] or 0) <= {"tags": tag_n, "brands": brand_n, "categories": len(K.MANY_CATEGORIES)}[j[1]] for j in K.LIST_JOBS)
    assert len({j[0] for j in K.LIST_JOBS}) == len(K.LIST_JOBS)
    req, size = jobs[("tags", "truncated")]
    c, _ = K.build_list_case(m, req, "tags", "truncated", size, "t-F8-1106")
    assert c["score"]["mode"] == "list" and c["meta"]["view"] == K.CAROUSEL_VIEW and len(c["option_lists"]["tags"]) == 130
    names = [e["name"] for e in c["option_lists"]["tags"]]
    assert len(set(names)) == 130 and not any(a != b and f" {a} " in f" {b} " for a in names for b in names)
    u, _ = K.build_list_case(m, *jobs[("brands", "unknown")][:1], "brands", "unknown", None, "t2")
    from souschef_lab.state import load_context_fixture
    fx = load_context_fixture()
    assert u["option_lists"] == {"brands": None, "tags": fx["tags"]}  # the host's lists: brands not loaded
    assert with_case_extras({"brands": [1], "tags": [2], "x": 3}, u) == {"brands": None, "tags": fx["tags"], "x": 3}
    cat, _ = K.build_list_case(m, "x", "categories", "short", None, "t3")
    assert cat["context_extras"] == {} and "option_lists" not in cat  # the fixture's own categories
    many, _ = K.build_list_case(m, "x", "categories", "long", 22, "t4")
    assert len(with_case_extras(fx, many)["categories"]) == 22
    assert K.capability_counts([c, u, cat])["list_options"] == {
        "cases": 3, "by_group": {"list:brands": 1, "list:categories": 1, "list:tags": 1}, "modes": {"list": 3}}


def test_capability_cases_split_per_case_and_report_their_group(m):
    """The validation split hashes meta.group: a per-case group, so no behaviour class goes wholly to one side;
    cap_group keeps the report's grouping."""
    c, _ = K.build_page_case(m, "exists", "make me a Deals page", {"title": "Deals", "pages": ["deals"]}, "t-F8-1020")
    assert c["meta"]["group"] == "page:exists:t-F8-1020" and c["meta"]["cap_group"] == "page:exists"
    lc, _ = K.build_list_case(m, "list my tags", "tags", "unknown", None, "t-F8-1135", "plain")
    assert lc["meta"]["group"] == "list:tags:t-F8-1135" and lc["meta"]["cap_group"] == "list:tags"
    assert lc["meta"]["view"] is None and lc["target"] == {"kind": "page", "page": "home"} and lc["meta"]["focus"] == "plain"
    assert K.capability_counts([c, lc])["create_page"]["by_group"] == {"page:exists": 1}


def test_social_page_requests_are_social_links_not_pages(m):
    from souschef_lab import exam as X
    socials = [(g, r, sp) for g, r, sp in K.PAGE_JOBS if g == "social_page"]
    assert 2 <= len(socials) <= 4
    exam_keys = {k["key"] for s_ in ("paraphrase", "holdout") for row in X.load_set(s_)
                 for k in (row.get("expected") or {}).get("keys") or [] if str(k.get("key")).startswith("socialLinks.")}
    assert exam_keys and not exam_keys & {sp["social"]["key"] for _, _, sp in socials}  # the exam's networks left to it
    g, req, spec = socials[0]
    c, why = K.build_page_case(m, g, req, spec, "t-F8-soc")
    assert why == "" and c["score"]["mode"] == "keys"
    assert c["meta"]["oracle_ops"] == [{"op": "set_fields", "target": {"kind": "global", "panel": "SocialMediaSection"},
                                        "values": {spec["social"]["key"]: spec["social"]["url"]}}]
    assert c["expected"]["keys"] == [{"target": {"kind": "global", "panel": "SocialMediaSection"},
                                      "key": spec["social"]["key"], "list": ""}]
    assert "example" in spec["social"]["url"]  # a made-up handle, never a store's


def test_the_template_family_uses_a_real_premium_template(m):
    from souschef_lab.browser_defaults import cookie_cutter_dir
    assert K.PREMIUM_TEMPLATE not in ("bold", "boutique", "cozy", "editorial", "elegant", "playful", "pluto", "retro",
                                      "sweet", "verdant", "vintage", "default")
    root = cookie_cutter_dir()
    sel = root / "launcher" / "client" / "src" / "components" / "ThemeSelector.jsx" if root else None
    if sel is not None and sel.is_file():  # a template the picker offers (not a hidden legacy one)
        src = sel.read_text(encoding="utf-8")
        hidden = src[src.index("const hiddenTemplates"):].split("]", 1)[0]
        assert f"  {K.PREMIUM_TEMPLATE}: (" in src and f"'{K.PREMIUM_TEMPLATE}'" not in hidden
