"""contrast.py — family F8, the look-alike-field contrast cases (r5why PLAN §2b)."""
from __future__ import annotations

import json
import re

import pytest

from souschef_lab import FIXTURES_DIR
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
    assert 120 <= rep["kept"] == len(cases) <= 260, rep
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
         "history": [{"user": "the headings look too plain", "expect": "ask", "assistant": "Which font would you like?"}]},
        {**base, "id": "f2-b", "request": "yes", "meta": {"family": "F2", "group": "g2", "page": "home"},
         "history": [{"user": "the headings look too plain", "expect": "offer", "assistant": "Bigger?"}]},  # same text
        {**base, "id": "f2-c", "request": "the hero", "meta": {"family": "F2", "group": "g3"},
         "history": [{"user": "make it pop", "expect": "write", "ops": []}]},                                # acted on
        {**base, "id": "f2-d", "request": "x", "holdout": True, "meta": {"family": "F2"},
         "history": [{"user": "the footer is dull", "expect": "ask"}]},                                      # a holdout
    ]
    out, rep = K.ask_first_cases(pool, m, None, load_context_fixture(), seen=set())
    assert [c["id"] for c in out] == ["askfirst1-F2-f2-a"] and rep["dropped"] == {"duplicate": 1}
    c = out[0]
    assert c["request"] == "the headings look too plain" and c["history"] == [] and c["score"]["mode"] == "ask"
    assert c["meta"]["group"] == "g1" and c["meta"]["family"] == "F2" and c["meta"]["ask_first"]["from"] == "f2-a"
    assert c["context"]["page"] == "home" and c["expected"]["ops"] == []


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
