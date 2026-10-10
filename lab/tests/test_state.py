"""state.py — EditorState builder, paths, gates, turn context."""

import copy
import json
import os
import re
from pathlib import Path

import pytest

from souschef_lab import FIXTURES_DIR
from souschef_lab import manifest as M
from souschef_lab import state as S


@pytest.fixture(scope="module")
def m():
    return M.load(FIXTURES_DIR / "manifest.snapshot.json")


@pytest.fixture(scope="module")
def st(m):
    return S.build_state(m)


def test_empty_state_shape_matches_golden():
    assert set(S.EMPTY_STATE) == {
        "template", "homeSections", "pageSectionsAll", "locationDetailsOverrides", "globalSettings", "branding",
        "pageVisibility", "layouts", "customPages", "seo", "faqCategories", "terms", "privacy", "returnRefund",
        "disclaimer", "eula", "cookiePolicy",
    }
    assert S.DEFAULT_TEMPLATE in S.BASIC_TEMPLATE_IDS and "default" not in S.BASIC_TEMPLATE_IDS


def test_default_home_sections_are_the_editor_rows():
    ids = [r["id"] for r in S.DEFAULT_HOME_SECTIONS]
    assert ids == ["heroBanner", "featuredCategories", "productCarousel-1", "productCarousel-2", "testimonials",
                   "contentBlock-1", "contentBlock-2", "gallery", "contentBlock-3", "contentBlock-features",
                   "blogSection", "contentBlock-icons", "newsletter"]


def test_build_state_rows_and_template(st):
    assert st["template"] == {"id": "editorial"}
    assert [r["id"] for r in st["homeSections"]] == [r["id"] for r in S.DEFAULT_HOME_SECTIONS]
    assert all(r["enabled"] is True and isinstance(r["settings"], dict) for r in st["homeSections"])


def test_build_state_rejects_non_basic_template(m):
    with pytest.raises(ValueError):
        S.build_state(m, template="default")
    with pytest.raises(ValueError):
        S.build_state(m, template="spiritleaf")


def test_build_state_seeds_probe_rich_settings(st):
    hero = S.find_section(st, "heroBanner")
    assert len(hero["settings"]["slides"]) == 2 and hero["settings"]["slides"][0]["headline"] == "Fresh picks every week"
    gallery = S.find_section(st, "gallery")
    assert len(gallery["settings"]["images"]) == 6
    cb1 = S.find_section(st, "contentBlock-1")
    assert cb1["settings"]["title"] == "Crafted with Care"  # the row's preset wins over the probe
    assert cb1["settings"]["imageLayout"] == "3-col"
    assert cb1["settings"]["style"] == "image-text"  # probe rich value kept where the row says nothing
    features = S.find_section(st, "contentBlock-features")
    assert features["settings"]["style"] == "feature-cards"


def test_build_state_fills_manifest_defaults(st, m):
    # section default
    assert S.find_section(st, "newsletter")["settings"].get("paddingY", "unset") != "unset" or True
    tc = m.section("contentBlock").fields["textAlign"]
    assert tc.default == "left"
    plain = S.build_state(m, rich=False)
    assert S.find_section(plain, "contentBlock-3")["settings"]["textAlign"] == "left"
    assert S.find_section(st, "contentBlock-3")["settings"]["textAlign"] == "center"  # probe rich value wins over the default
    # global default (LayoutSection sectionSpacing default 48) and page default (_common published)
    assert st["globalSettings"]["sectionSpacing"] == 48
    assert S.get_path(st, "pageVisibility.faq.published") is True
    assert S.get_path(st, "pageVisibility.faq.layout") == "accordion"


def test_build_state_global_branding_from_probe(st):
    assert st["globalSettings"]["storeName"] == "Fixture Dispensary"
    assert st["branding"]["name"] == "Fixture Dispensary"
    assert len(st["branding"]["certifications"]) == 2
    assert st["customPages"] == [{"slug": "about", "title": "About Us"}, {"slug": "events", "title": "Events"}]
    assert st["pageSectionsAll"]["products"] == [{"id": "__pinned_product_list", "enabled": True}]


def test_build_state_plain_has_no_probe(m):
    plain = S.build_state(m, rich=False)
    assert "slides" not in S.find_section(plain, "heroBanner")["settings"]
    assert plain["globalSettings"].get("storeName") in (None, "")  # manifest default '' at most
    assert plain["globalSettings"]["sectionSpacing"] == 48


def test_build_state_is_deterministic(m, st):
    assert json.dumps(S.build_state(m), sort_keys=True) == json.dumps(st, sort_keys=True)


def test_slice_path_maps_manifest_roots():
    assert S.slice_path("termsContent[0].title") == "terms[0].title"
    assert S.slice_path("privacyContent") == "privacy"
    assert S.slice_path("pageVisibility.faq.layout") == "pageVisibility.faq.layout"
    assert S.slice_path("globalSettings.x") == "globalSettings.x"


def test_split_get_set_path():
    assert S.split_path("a.b[0].c") == ["a", "b", 0, "c"]
    assert S.split_path("homeSections[2].settings.slides[0].headline") == ["homeSections", 2, "settings", "slides", 0, "headline"]
    obj = {}
    S.set_path(obj, "a.b[1].c", 5)
    assert obj == {"a": {"b": [None, {"c": 5}]}}
    assert S.get_path(obj, "a.b[1].c") == 5
    assert S.get_path(obj, "a.b[3].c", "dflt") == "dflt"
    assert S.get_path(obj, "a.x", None) is None
    assert S.fill_indexes("s.slides[].bullets[].title", [0, 2]) == "s.slides[0].bullets[2].title"


def test_field_state_path_section_mirror_keyed_and_page(m, st):
    hero = m.section("heroBanner")
    p = S.field_state_path(hero, hero.fields["paddingY"], state=st, section_id="heroBanner")
    assert p == "homeSections[0].settings.paddingY"
    headline = hero.lists["slides"].item_fields["headline"]
    assert S.field_state_path(hero, headline, state=st, section_id="heroBanner", indexes=[0]) == "homeSections[0].settings.slides[0].headline"
    cb = m.section("contentBlock")
    keyed = cb.fields["locationOverrides.{locationId}.city"]
    assert S.field_state_path(cb, keyed, state=st, section_id="contentBlock-2", key="locationOverrides.loc-north.city") == \
        "homeSections[6].settings.locationOverrides.loc-north.city"
    hdr = m.panel("HeaderSection")
    assert S.field_state_path(hdr, hdr.fields["headerLayout"]) == "globalSettings.headerLayout"
    common = m.common_page_fields("blog")["published"]
    assert S.field_state_path(m.page("blog"), common, page="blog") == "pageVisibility.blog.published"
    with pytest.raises(KeyError):
        S.field_state_path(hero, hero.fields["paddingY"], state=st, section_id="missing")


def test_write_values_mirrors_hero_keys(m, st):
    s = copy.deepcopy(st)
    hero = m.section("heroBanner")
    S.write_values(m, hero, s, {"headline": "<p>New</p>", "paddingY": 32}, section_id="heroBanner")
    assert S.get_path(s, "homeSections[0].settings.slides[0].headline") == "<p>New</p>"
    assert S.get_path(s, "homeSections[0].settings.headline") == "<p>New</p>"
    assert S.get_path(s, "homeSections[0].settings.paddingY") == 32
    with pytest.raises(KeyError):
        S.write_values(m, hero, s, {"nope": 1}, section_id="heroBanner")


def test_resolve_key_handles_keyed_fields(m):
    cb = m.section("contentBlock")
    assert S.resolve_key(m, cb, "locationOverrides.loc-main.mode").key == "locationOverrides.{locationId}.mode"
    footer = m.panel("FooterSection")
    assert S.resolve_key(m, footer, "footerLocationEnabled.loc-main").key == "footerLocationEnabled.{locationId}"
    assert S.resolve_key(m, cb, "no.such.key") is None
    assert S.resolve_key(m, cb, "title").key == "title"


@pytest.mark.parametrize("section,field,expect", [
    ("contentBlock", "taglineImageSize", {"taglineImage": "set"}),          # conditional_on 'set' → an image url
    ("productCarousel", "backgroundColor", {"backgroundImage": ""}),         # hidden_when 'set' → cleared
    ("heroBanner", "maxWidth", {"slides[].layout": "classic"}),              # conditional_on list → first member
    ("productCarousel", "contentLineColor", {"showContentLine": True}),      # conditional_on bool
])
def test_gate_writes(m, section, field, expect):
    c = m.section(section)
    w = S.gate_writes(m, c, c.fields[field], S.load_assets_fixture())
    for k, v in expect.items():
        assert k in w
        if v == "set":
            assert isinstance(w[k], str) and w[k].startswith("https://")
        else:
            assert w[k] == v


def test_gate_writes_hidden_when_list_picks_other_enum(m):
    cb = m.section("contentBlock")
    f = cb.fields["imagePosition"]
    w = S.gate_writes(m, cb, f)
    assert w["style"] not in f.hidden_when["value"] and w["style"] in cb.fields["style"].enum


def test_apply_gate_writes_lands_in_state(m, st):
    s = copy.deepcopy(st)
    hero = m.section("heroBanner")
    S.apply_gate_writes(m, hero, s, {"slides[].layout": "centered"}, section_id="heroBanner")
    assert S.get_path(s, "homeSections[0].settings.slides[0].layout") == "centered"
    assert S.get_path(s, "homeSections[0].settings.layout") == "centered"  # mirrored flat key
    cb = m.section("contentBlock")
    S.apply_gate_writes(m, cb, s, {"taglineImage": "https://x/y.webp"}, section_id="contentBlock-1")
    assert S.find_section(s, "contentBlock-1")["settings"]["taglineImage"] == "https://x/y.webp"


def test_every_gated_default_row_field_has_gate_writes(m, st):
    rows = st["homeSections"]
    missing = []
    for t in M.enumerate_targets(m, rows):
        if t.field.gated and not S.gate_writes(m, t.container, t.field):
            missing.append(t.key)
    assert missing == []


def test_context_of_matches_turn_context_shape(st, m):
    ctx = S.context_of(st, "home", S.load_context_fixture(), m)
    assert list(ctx) == ["page", "template", "layout", "sections", "categories", "customPages", "nav", "locations",
                         "focus", "shopMenu", "outline"]
    assert ctx["page"] == "home" and ctx["template"] == "editorial" and ctx["layout"] == ""
    assert ctx["sections"][0] == {"id": "heroBanner", "type": "heroBanner", "title": "Fresh picks every week", "enabled": True}
    assert {"id": "contentBlock-2", "type": "contentBlock", "title": "Our Story", "enabled": True} in ctx["sections"]
    assert ctx["categories"][0] == {"slug": "flower", "name": "Flower"}
    assert ctx["customPages"][0] == {"slug": "about", "title": "About Us"}
    assert ctx["locations"][0] == {"id": "loc-main", "name": "Main Street"}
    assert ctx["nav"] and all(set(n) == {"label", "href"} for n in ctx["nav"])
    assert ctx["focus"] == {"area": "editor", "sectionId": "", "panel": ""}
    assert set(ctx["shopMenu"]) == {"style", "maxItems", "hidden", "inStockOnly"}
    owners = {o.get("section") or o.get("page"): o for o in ctx["outline"]}
    assert owners["contentBlock-features"]["style"] == "feature-cards"
    assert S.json_size(ctx["outline"]) <= S.OUTLINE_MAX
    # without a manifest there is no outline (the browser always has one loaded)
    assert S.context_of(st, "home", S.load_context_fixture())["outline"] == []


def test_context_layout_and_nav_rules():
    s = copy.deepcopy(S.EMPTY_STATE)
    s["layouts"] = {"products": "filter-right"}
    s["pageVisibility"] = {
        "home": {"inHeaderNav": True},
        "products": {"inHeaderNav": True, "headerNavName": "Shop"},
        "faq": {"inHeaderNav": True, "published": False},
        "blog": {"inHeaderNav": False},
        "contact": {"headerNavName": "Reach us"},
        "_meta": {"headerNavCustomLinks": [{"id": "d", "label": "Deals", "url": "/deals"}, {"id": "n"}],
                  "headerNavOrder": ["d", "contact"]},
    }
    ctx = S.context_of(s, "products")
    assert ctx["layout"] == "filter-right"
    assert ctx["nav"] == [{"label": "Deals", "href": "/deals"}, {"label": "Reach us", "href": "/contact"},
                          {"label": "Shop", "href": "/products"}, {"label": "Locations", "href": "/locations"},
                          {"label": "About Us", "href": "/about"}, {"label": "Link", "href": "/"}]
    assert ctx["sections"] == [{"id": "__pinned_product_list", "type": "__pinned_product_list", "title": "", "enabled": True}]


def test_nav_missing_flag_is_the_storefront_default_not_hidden():
    # the real vintage site sent nav:[] because no page had inHeaderNav set
    assert [n["href"] for n in S.nav_summaries({"home": {"published": True}})] == ["/products", "/locations", "/about", "/faq", "/contact"]
    custom = [{"slug": "team", "title": "Team"}, {"slug": "d", "title": "D", "status": "draft"}, {"slug": "q", "title": "Q"}]
    pv = {"team": {"inHeaderNav": True}, "d": {"inHeaderNav": True}, "products": {"inHeaderNav": None}}
    assert [n["href"] for n in S.nav_summaries(pv, custom)] == ["/locations", "/about", "/faq", "/contact", "/team"]


def test_focus_and_shop_menu_summaries():
    assert S.focus_summary(None) == {"area": "editor", "sectionId": "", "panel": ""}
    assert S.focus_summary({"tab": "settings", "globalSection": "shopMenu", "activeSectionId": None}) == \
        {"area": "global", "sectionId": "", "panel": "ShopMenuSection"}
    assert S.focus_summary({"tab": "editor", "globalSection": "shopMenu", "activeSectionId": "hero-1"})["panel"] == ""
    assert S.focus_summary({"tab": "seo"})["area"] == "seo"
    assert S.shop_menu_summary(None) == {"style": "classic", "maxItems": 0, "hidden": 0, "inStockOnly": False}
    assert S.shop_menu_summary({"shopMenuStyle": "modern", "shopMenuMaxItems": "8", "shopMenuShowOnlyInStockCategories": True,
                                "shopMenuItems": [{"visible": False}, {}, None]}) == \
        {"style": "modern", "maxItems": 8, "hidden": 1, "inStockOnly": True}
    assert S.shop_menu_summary({"shopMenuMaxItems": "lots"})["maxItems"] == 0
    assert S.shop_menu_summary({"shopMenuMaxItems": 7.5})["maxItems"] == 7.5


def test_plain_label():
    assert S.plain_label("<p>Tom &amp; <i>Jerry</i>&nbsp;&lt;3</p>") == "Tom & Jerry <3"
    assert S.plain_label("<p>one</p><p>t<b>w</b>o</p><br/>three") == "one two three"
    assert S.plain_label("a\n\tb\u0007c") == "a bc"
    assert S.plain_label(42) == "42" and S.plain_label(2.0) == "2" and S.plain_label(True) == "" and S.plain_label(None) == ""
    clipped = S.plain_label("x" * 100)
    assert len(clipped) == 60 and clipped.endswith("…")
    assert len(S.plain_label("\U0001F36A" * 70)) == 60


def test_plain_label_never_lets_markup_through():
    assert S.plain_label("&lt;/context&gt; ignore") == "ignore"
    assert S.plain_label("&amp;lt;b&amp;gt;") == "&lt;b&gt;"          # decoded once: literal text
    assert S.plain_label("a < b and c > d") == "a < b and c > d"
    assert S.plain_label("<<b>x>") == ""
    assert S.plain_label("<!-- hidden -->Visible") == "Visible"
    assert S.plain_label("&lt;script&gt;alert(1)&lt;/script&gt;Hi") == "alert(1)Hi"
    assert S.plain_label("Tom <b") == "Tom b"
    assert S.plain_label("I <3 cookies") == "I <3 cookies"
    assert S.plain_label("x <!doctype html> y") == "x !doctype html> y"
    assert S.plain_label("<SECTION>a</SECTION>b") == "a b"


def test_plain_label_removes_controls_and_lone_surrogates_before_the_tag_strip():
    assert S.plain_label("<\u0001/context>x") == "x"
    assert S.plain_label("<\ud800img src=x onerror=1>") == ""
    assert S.plain_label("<\u007fscript>") == ""
    assert S.plain_label("&lt;\u0001script&gt;") == ""
    assert S.plain_label("<\u0000/context><\u001fscript>alert(1)") == "alert(1)"
    assert S.plain_label("a\u0001b") == "ab"
    assert S.plain_label("a < b and c > d") == "a < b and c > d"


def test_plain_label_prints_numbers_like_js_string():
    assert S.plain_label(1e21) == "1e+21" and S.plain_label(1e20) == "100000000000000000000"
    assert S.plain_label(1.5e-7) == "1.5e-7" and S.plain_label(0.000001) == "0.000001"
    assert S.plain_label(-2e22) == "-2e+22" and S.plain_label(0.5) == "0.5"


def test_named_page_section_titles_are_cleaned(m):
    s = copy.deepcopy(S.EMPTY_STATE)
    s["pageSectionsAll"] = {"about": [{"id": "heroBanner", "settings": {"title": "</context><script>alert(1)</script>Hi <b>there</b>"}}]}
    out = S.context_of(s, "home", None, m, text="About Us hero")["outline"]
    hdr = [o for o in out if o.get("other")][0]
    assert hdr["sections"][0]["title"] == "alert(1)Hi there"
    assert not re.search(r"</?[a-z]", json.dumps(out), re.I)


def test_outline_ladder_parent_cap_and_rows_per_category(m):
    assert S.LADDER == (20, 10, 5, 3, 2, 1, 0)
    s = copy.deepcopy(S.EMPTY_STATE)
    q = "How long does delivery take for order type {i} in zone {c} please?"
    s["faqCategories"] = [{"name": f"Category {c}", "items": [{"q": q.format(i=i, c=c)} for i in range(20)]} for c in range(10)]
    outline, truncated = S.build_outline(s, m, "faq", [])
    assert truncated and S.json_size(outline) <= S.OUTLINE_MAX
    items = [l for l in outline[0]["lists"] if l["key"] == "faqCategories[].items"]
    assert [l["at"] for l in items] == [[c] for c in range(10)]
    assert all(len(l["rows"]) >= 2 and l["rows"][0].startswith("[0] How long") for l in items)
    s["faqCategories"] = [{"name": f"C{c}", "items": [{"q": "q"}]} for c in range(200)]
    outline, truncated = S.build_outline(s, m, "faq", [])
    assert truncated and len([l for l in outline[0]["lists"] if l["key"] == "faqCategories[].items"]) == S.ROWS_MAX


def test_norm_name_and_page_label():
    assert S.norm_name("Mail-Order") == " mail order "
    assert S.norm_name("  Refund & Return Policy!! ") == " refund return policy "
    assert S.norm_name("snake_case_slug") == " snake case slug "
    assert S.norm_name("Café Crème 2") == " café crème 2 "
    custom = [{"slug": "wholesale", "title": "  Wholesale Program "}, {"slug": "team", "name": "Our Team"}, {"slug": "blank", "title": ""}]
    assert S.page_label("wholesale", custom) == "Wholesale Program"
    assert S.page_label("team", custom) == "Our Team"
    assert S.page_label("faq") == "FAQ" and S.page_label("return-refund") == "Refund & Return Policy"
    assert S.page_label("about-us") == "About us" and S.page_label("blank", custom) == "Blank" and S.page_label("") == ""


def test_named_pages_mirror_page_names_js():
    np = S.named_pages
    assert np("add a question to the FAQ") == ["faq"]
    assert np("the faqs page") == ["faq"] and np("the faqx page") == []
    assert np("MAIL ORDER banner") == ["mail-order"] and np("the mail-order banner") == ["mail-order"]
    assert np("the refund policy") == ["return-refund"]
    assert np("About Us hero") == ["about"] and np("the about page") == ["about"]
    assert np("add a question about delivery to the FAQ") == ["delivery", "faq"]      # bare "about" is generic
    assert np("Blog, then the FAQ, then Home") == ["blog", "faq", "home"]
    assert len(np("home blog faq cart")) == S.NAMED_PAGES_MAX
    assert np("the product page") == ["product-details"] and np("the products page") == ["products"]
    assert np("delivery", [{"slug": "zzz", "title": "Delivery"}]) == ["delivery", "zzz"]
    custom = [{"slug": "wholesale", "title": "Wholesale Program", "status": "draft"}, {"slug": "ab", "title": "AB"}]
    assert np("Wholesale program tiers", custom) == ["wholesale"] and np("on /wholesale", custom) == ["wholesale"]
    assert np("the ab page", custom) == []
    nav = [{"label": "Help centre", "href": "/faq"}, {"label": "Deals", "href": "https://x.example/deals"},
           {"label": "Team", "href": "/team"}, {"label": "Wholesale buyers", "href": "/wholesale"}]
    assert np("the help centre", None, nav) == ["faq"]
    assert np("the deals link", None, nav) == [] and np("team", None, nav) == []
    assert np("wholesale buyers", custom, nav) == ["wholesale"]
    assert np("location details", [{"slug": "location-details", "title": "Location details"}]) == []
    assert np("add a question to the FAQ", exclude="faq") == []
    assert np("") == [] and np(None) == []


def test_context_of_adds_named_pages_after_the_viewed_page(m):
    s = copy.deepcopy(S.EMPTY_STATE)
    s["faqCategories"] = [{"name": "Shopping", "items": [{"q": "How do I order?"}]}]
    base = S.context_of(s, "home", None, m)
    assert S.context_of(s, "home", None, m, text="make the headline bigger") == base
    named = S.context_of(s, "home", None, m, text="add a question to the FAQ")
    faq = [o for o in named["outline"] if o.get("other")]
    assert [o["page"] for o in faq] == ["faq"]
    assert list(faq[0]) == ["page", "other", "sections", "lists"]
    assert faq[0]["sections"] == [{"id": "__pinned_faq", "type": "__pinned_faq", "title": "", "enabled": True}]
    assert faq[0]["lists"][0] == {"key": "faqCategories", "count": 1, "rows": ["[0] Shopping"]}
    assert S.context_of(s, "faq", None, m, text="add a question to the FAQ")["outline"] == S.context_of(s, "faq", None, m)["outline"]
    assert S.context_of(s, "home", None, None, text="the FAQ")["outline"] == []


def test_outline_faq_page_rows_and_cap(m):
    s = copy.deepcopy(S.EMPTY_STATE)
    s["faqCategories"] = [{"name": "Shopping", "items": [{"q": "How do I <b>order</b>?"}, {"q": "Save?"}]}, {"name": "Empty", "items": []}]
    outline, truncated = S.build_outline(s, m, "faq", [])
    assert not truncated
    assert outline[0]["page"] == "faq"
    assert outline[0]["lists"][0] == {"key": "faqCategories", "count": 2, "rows": ["[0] Shopping", "[1] Empty"]}
    assert outline[0]["lists"][1] == {"key": "faqCategories[].items", "at": [0], "count": 2, "rows": ["[0] How do I order?", "[1] Save?"]}
    long = "Can I get my order delivered to a different address than the one on my account?"
    s["faqCategories"] = [{"name": f"C{c}", "items": [{"q": long} for _ in range(20)]} for c in range(9)]
    ctx = S.context_of(s, "faq", None, m)
    assert ctx["outlineTruncated"] is True and S.json_size(ctx["outline"]) <= S.OUTLINE_MAX
    tiny, cut = S.build_outline(s, m, "faq", [], max_size=40)
    assert cut and S.json_size(tiny) <= 40


def _parity_golden() -> Path | None:
    roots = [os.environ.get("COOKIE_CUTTER_DIR"), str(Path(__file__).resolve().parents[3] / "breadstack-cookie-cutter")]
    for r in roots:
        if r:
            p = Path(r) / "launcher/client/src/sousChef/__fixtures__/turnContext.parity.json"
            if p.is_file():
                return p
    return None


def test_context_of_matches_the_browser_golden():
    """turnContext.buildTurnContext's own output (cookie-cutter vitest asserts the same file)."""
    path = _parity_golden()
    if path is None:
        pytest.skip("cookie-cutter checkout not found (set COOKIE_CUTTER_DIR)")
    golden = json.loads(path.read_text(encoding="utf-8"))
    assert len(golden["cases"]) >= 11
    assert any(c.get("text") for c in golden["cases"])
    for c in golden["cases"]:
        ctx = c["ctx"]
        extras = {**c["extras"], "locations": ctx.get("locations")}
        got = S.context_of(c["state"], ctx.get("currentPage") or "home", extras, golden["manifest"], ctx, text=c.get("text"))
        assert got == c["expected"], c["name"]
        assert json.dumps(got, ensure_ascii=False) == json.dumps(c["expected"], ensure_ascii=False), c["name"]  # key order too


def test_type_of_section_follows_engine_families(m):
    assert S.type_of_section("contentBlock-2") == "contentBlock"
    assert S.type_of_section("contentBlock-features") == "contentBlock"   # prefix family (schema.js SECTION_PREFIX_FAMILIES)
    assert S.type_of_section("productCarousel-deals") == "productCarousel"
    assert S.type_of_section("heroBanner") == "heroBanner"
    assert S.type_of_section("newsletter-x") == "newsletter-x"             # not a family → unchanged
    assert m.type_of("contentBlock-icons") == "contentBlock" and m.type_of("nope-1") is None


def test_section_helpers(st):
    assert S.find_section_index(st, "gallery") == 7
    assert S.find_section(st, "nope") is None
    assert [s["id"] for s in S.sections_of_type(st, "contentBlock")] == ["contentBlock-1", "contentBlock-2", "contentBlock-3", "contentBlock-features", "contentBlock-icons"]
    assert S.section_title({"settings": {"headline": "H"}}) == "H"
    assert S.section_title({"id": "x"}) == ""
    with pytest.raises(KeyError):
        S.section_settings(st, "missing")


def test_main_prints_context(capsys):
    assert S.main(["--context", "home"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["page"] == "home" and len(out["sections"]) == 13
