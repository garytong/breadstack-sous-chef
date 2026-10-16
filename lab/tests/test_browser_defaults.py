"""browser_defaults.py — the launcher's defaults for a never-saved site, and the
lab states / contexts that use them (state.build_state FAQ seed, context_of's
never-saved pages). The node-backed tests need the cookie-cutter checkout
(COOKIE_CUTTER_DIR) and skip without it — unless SOUS_CHEF_LAB_REQUIRE_ENGINE=1."""

import copy
import json
import os

import pytest

from souschef_lab import FIXTURES_DIR
from souschef_lab import browser_defaults as B
from souschef_lab import engine as E
from souschef_lab import manifest as M
from souschef_lab import state as S

REQUIRE = os.environ.get("SOUS_CHEF_LAB_REQUIRE_ENGINE") == "1"


@pytest.fixture(scope="module")
def m():
    return M.load(FIXTURES_DIR / "manifest.snapshot.json")


def _need_node_checkout():
    if B.cookie_cutter_dir() is None or E.node_version() is None:
        msg = "node and a cookie-cutter checkout (COOKIE_CUTTER_DIR) are needed"
        if REQUIRE:
            pytest.fail("SOUS_CHEF_LAB_REQUIRE_ENGINE=1 but " + msg)
        pytest.skip(msg)


# ── the fixture ─────────────────────────────────────────────────────────────

def test_fixture_covers_every_basic_template_and_lab_page(m):
    doc = B.load()
    assert set(doc["templates"]) == set(S.BASIC_TEMPLATE_IDS) | {"default"}
    assert set(doc["pages"]) == set(doc["templates"]) | {B.OTHER_TEMPLATE}
    pages = set(B.PAGES)
    assert set(m.pages) <= pages, "a manifest page has no dumped defaults"
    assert {p["slug"] for p in S.load_context_fixture()["customPages"]} <= pages
    for t in doc["pages"]:
        assert set(doc["pages"][t]) == pages
    assert doc["source"]["commit"] and all(doc["source"]["files"].values())


def test_page_sections_are_the_templates_defaults_not_a_pinned_row():
    assert B.page_sections("vintage", "faq") == []
    assert B.page_sections("vintage", "products") == []
    pd = B.page_sections("vintage", "product-details")
    assert [r["id"] for r in pd] == ["productCarousel-1"] and pd[0]["settings"]["subType"] == "related"
    assert [r["id"] for r in B.page_sections("vintage", "about")] == ["contentBlock-1", "contentBlock-2", "testimonials"]
    assert [r["id"] for r in B.page_sections("pluto", "products")] == ["featuredCategories"]
    # text pages resolve to 'text-page', which has no defaults; unknown slugs and templates too
    assert B.page_sections("vintage", "privacy-policy") == [] and B.page_sections("vintage", "nope") == []
    assert [r["id"] for r in B.page_sections("plantrepublic", "product-details")] == ["productCarousel-1"]  # the '*' fallback
    assert B.page_sections("plantrepublic", "faq") == []
    for t in S.BASIC_TEMPLATE_IDS:
        for p in B.PAGES:
            assert not any(str(r.get("id", "")).startswith("__pinned") for r in B.page_sections(t, p)), (t, p)
    a = B.page_sections("vintage", "about")
    a[0]["id"] = "changed"
    assert B.page_sections("vintage", "about")[0]["id"] == "contentBlock-1"  # a fresh copy each call


def test_faq_categories_are_the_basic_starter_set():
    for t in S.BASIC_TEMPLATE_IDS:
        faq = B.faq_categories(t)
        assert [c["name"] for c in faq] == ["Shopping", "Delivery", "Products", "Account"], t
        assert all(set(i) == {"q", "a"} for c in faq for i in c["items"])
    assert B.faq_categories("plantrepublic") is None and B.faq_categories(None) is None


def test_stable_ignores_random_image_draws_only():
    a = {"pages": {"pluto": {"x": [{"settings": {"images": ["/a", "/b"], "title": "T"}}]}}}
    b = copy.deepcopy(a)
    b["pages"]["pluto"]["x"][0]["settings"]["images"] = ["/c", "/d"]
    assert B.same_content(a, b)
    b["pages"]["pluto"]["x"][0]["settings"]["images"] = ["/c"]
    assert not B.same_content(a, b)
    b = copy.deepcopy(a)
    b["pages"]["pluto"]["x"][0]["settings"]["title"] = "U"
    assert not B.same_content(a, b)


# ── the lab states and contexts that use them ──────────────────────────────

@pytest.mark.parametrize("template", S.BASIC_TEMPLATE_IDS)
@pytest.mark.parametrize("rich", [True, False])
def test_build_state_seeds_the_basic_faq_categories(m, template, rich):
    st = S.build_state(m, template=template, rich=rich)
    assert st["faqCategories"] == B.faq_categories(template)
    assert not any("questions" in c for c in st["faqCategories"])  # never the probe's premium schema


def test_never_saved_pages_list_the_template_defaults(m):
    st = S.build_state(m, template="vintage")
    assert "faq" not in st["pageSectionsAll"]
    faq = S.context_of(st, "faq", S.load_context_fixture(), m)
    assert faq["sections"] == []
    lists = {l["key"]: l for o in faq["outline"] if o.get("page") == "faq" for l in o.get("lists", [])}
    assert lists["faqCategories"]["count"] == 4 and lists["faqCategories"]["rows"][0] == "[0] Shopping"
    about = S.context_of(st, "about", S.load_context_fixture(), m)
    assert [s["id"] for s in about["sections"]] == ["contentBlock-1", "contentBlock-2", "testimonials"]
    pd = S.context_of(st, "product-details", S.load_context_fixture(), m)
    assert [(s["id"], s["title"]) for s in pd["sections"]] == [("productCarousel-1", "Related Products")]
    # the host ctx's templateId picks the defaults, as targets.resolveSectionSlot does
    pl = S.context_of(st, "products", None, m, {"templateId": "pluto"})
    assert [s["id"] for s in pl["sections"]] == ["__pinned_product_list"]  # saved: the state's own row wins
    st2 = copy.deepcopy(st)
    del st2["pageSectionsAll"]["products"]
    assert [s["id"] for s in S.context_of(st2, "products", None, m, {"templateId": "pluto"})["sections"]] == ["featuredCategories"]
    assert S.context_of(st2, "products", None, m)["sections"] == []
    # a saved page keeps its rows; a saved [] counts as never saved (resolveSectionSlot hasSaved)
    st2["pageSectionsAll"]["about"] = [{"id": "newsletter", "enabled": True}]
    assert [s["id"] for s in S.context_of(st2, "about", None, m)["sections"]] == ["newsletter"]
    st2["pageSectionsAll"]["about"] = []
    assert [s["id"] for s in S.context_of(st2, "about", None, m)["sections"]] == ["contentBlock-1", "contentBlock-2", "testimonials"]


def test_named_never_saved_page_carries_its_defaults(m):
    st = S.build_state(m, template="vintage")
    out = S.context_of(st, "home", None, m, text="add a testimonial to the About Us page")["outline"]
    about = [o for o in out if o.get("other") and o.get("page") == "about"]
    assert about and [s["id"] for s in about[0]["sections"]] == ["contentBlock-1", "contentBlock-2", "testimonials"]


# ── node-backed: the dump is current, and context_of equals the browser's ──

def test_fixture_matches_the_checkout():
    _need_node_checkout()
    doc = B.generate()
    assert B.same_content(B.load(), doc), "fixtures/browser-defaults.json is stale: make -C lab browser-defaults"


def test_context_of_equals_the_browsers_on_lab_states(m):
    """The browser's own buildTurnContext on build_state states — never-saved
    pages (faq, product-details, about, delivery, cart, a text page), a saved
    one (products), home naming never-saved pages, an open section and a
    global panel — across four basic templates."""
    _need_node_checkout()
    ctxf = S.load_context_fixture()
    extras = {"categories": ctxf.get("categories") or []}
    plan = [("vintage", "faq", None, None), ("vintage", "product-details", None, None), ("vintage", "about", None, None),
            ("pluto", "delivery", None, None), ("editorial", "cart", None, None), ("bold", "privacy-policy", None, None),
            ("vintage", "products", None, None), ("pluto", "home", "add a question to the FAQ and fix the About Us page", None),
            ("vintage", "home", None, {"tab": "editor", "activeSectionId": "contentBlock-2"}),
            ("editorial", "faq", None, {"tab": "settings", "globalSection": "typography"}),
            ("cozy", "locations", "change the delivery page hero", None)]
    cases, ours = [], []
    for template, page, text, view in plan:
        st = S.build_state(m, template=template)
        ctx = {"currentPage": page, "templateId": template, "locations": ctxf.get("locations") or [], **(view or {})}
        cases.append({"state": st, "ctx": ctx, "extras": {**extras, "manifest": m.doc, **({"text": text} if text else {})}})
        ours.append(S.context_of(st, page, {**extras, "locations": ctx["locations"]}, m, ctx, text=text))
    theirs = B.browser_contexts(cases)
    root = B.cookie_cutter_dir()
    tc = root / "launcher" / "client" / "src" / "sousChef" / "turnContext.js" if root else None
    if tc is None or not tc.is_file() or "cat.id != null" not in tc.read_text(encoding="utf-8"):
        # a checkout before the category ids (launcher 1d4193e2 sends `id` as text; the lab mirrors that): compare
        # without them — the fixture's categories have ids since 2026-10-16, as live contexts do
        ours = [{**o, "categories": [{k: v for k, v in c.items() if k != "id"} for c in o.get("categories") or []]}
                if "categories" in o else o for o in ours]
    for (template, page, text, _), got, want in zip(plan, ours, theirs):
        assert got == want, (template, page, text)
        assert json.dumps(got, ensure_ascii=False) == json.dumps(want, ensure_ascii=False), (template, page)  # key order too
