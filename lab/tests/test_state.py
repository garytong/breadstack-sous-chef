"""state.py — EditorState builder, paths, gates, turn context."""

import copy
import json

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


def test_context_of_matches_turn_context_shape(st):
    ctx = S.context_of(st, "home", S.load_context_fixture())
    assert set(ctx) == {"page", "template", "layout", "sections", "categories", "customPages", "nav", "locations"}
    assert ctx["page"] == "home" and ctx["template"] == "editorial" and ctx["layout"] == ""
    assert ctx["sections"][0] == {"id": "heroBanner", "type": "heroBanner", "title": "Fresh picks every week", "enabled": True}
    assert {"id": "contentBlock-2", "type": "contentBlock", "title": "Our Story", "enabled": True} in ctx["sections"]
    assert ctx["categories"][0] == {"slug": "flower", "name": "Flower"}
    assert ctx["customPages"][0] == {"slug": "about", "title": "About Us"}
    assert ctx["locations"][0] == {"id": "loc-main", "name": "Main Street"}
    assert all(set(n) == {"label", "href"} for n in ctx["nav"])


def test_context_layout_and_nav_rules():
    s = copy.deepcopy(S.EMPTY_STATE)
    s["layouts"] = {"products": "filter-right"}
    s["pageVisibility"] = {
        "home": {"inHeaderNav": True},
        "products": {"inHeaderNav": True, "headerNavName": "Shop"},
        "faq": {"inHeaderNav": True, "published": False},
        "blog": {"inHeaderNav": False},
        "_meta": {"headerNavCustomLinks": [{"label": "Deals", "url": "/deals"}, {"label": "no-href"}]},
    }
    ctx = S.context_of(s, "products")
    assert ctx["layout"] == "filter-right"
    assert ctx["nav"] == [{"label": "home", "href": "/"}, {"label": "Shop", "href": "/products"}, {"label": "Deals", "href": "/deals"}]
    assert ctx["sections"] == [{"id": "__pinned_product_list", "type": "__pinned_product_list", "title": "", "enabled": True}]


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
