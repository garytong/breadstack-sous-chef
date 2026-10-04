"""manifest.py — loader, README §1.1 hash, iterators, target enumeration, holdout."""

import json
from pathlib import Path

import pytest

from souschef_lab import FIXTURES_DIR
from souschef_lab import manifest as M

MANIFEST = FIXTURES_DIR / "manifest.snapshot.json"
# service README §1.1 test vector for service/testdata/manifest.snapshot.json
VECTOR = "727d151b5d6b04cd68b73199ab0b3bdf061263f9623f4f6eda00a79137c06f38"


@pytest.fixture(scope="module")
def m() -> M.Manifest:
    return M.load(MANIFEST)


def test_fixture_is_the_service_snapshot():
    assert MANIFEST.is_symlink() or MANIFEST.exists()
    assert MANIFEST.resolve() == (Path(__file__).resolve().parents[2] / "service/testdata/manifest.snapshot.json").resolve()


def test_hash_matches_service_vector(m):
    assert m.hash == VECTOR
    assert M.hash_file(MANIFEST) == VECTOR


def test_hash_ignores_generated_from_and_coverage(m):
    doc = json.loads(json.dumps(m.doc))
    doc["generated_from"] = {"at": "2099-01-01T00:00:00Z", "sha": "deadbeef"}
    doc["coverage"] = {"sites_total": 0}
    assert M.canonical_hash(doc) == VECTOR


def test_hash_changes_when_a_field_changes(m):
    doc = json.loads(json.dumps(m.doc))
    doc["sections"]["heroBanner"]["fields"]["paddingY"]["max"] = 161
    assert M.canonical_hash(doc) != VECTOR


def test_canonical_bytes_sorted_compact_raw_unicode():
    b = M.canonical_bytes({"b": 1, "a": {"é": "ü"}, "generated_from": {"at": "x"}, "coverage": 1})
    assert b == '{"a":{"é":"ü"},"b":1}'.encode("utf-8")


def test_counts_match_the_snapshot(m):
    c = m.counts()
    assert c["fields"] == 2094
    assert c["lists"] == 69
    assert c["sections"] == 19
    assert c["globalPanels"] == 12
    assert c["pages"] == 17
    assert c["details"] == 11
    assert c["derived"] == 28


def test_section_lookup_strips_repeat_suffix(m):
    assert M.section_type_id("contentBlock-2") == "contentBlock"
    assert M.section_type_id("productCarousel-1") == "productCarousel"
    assert M.section_type_id("heroBanner") == "heroBanner"
    assert m.section("contentBlock-features").name == "contentBlock"
    assert m.section("nope") is None


def test_field_properties(m):
    f = m.section("heroBanner").fields["maxWidth"]
    assert f.type == "range" and f.min == 600 and f.max == 1920 and f.step == 20
    assert f.has_unset and f.unset_value is None and not f.has_default
    assert f.gated and f.conditional_on == {"key": "slides[].layout", "value": ["classic", "centered"]}
    img = m.section("contentBlock").fields["taglineImage"]
    assert img.type == "image" and img.companions == {"srcset": "taglineImageSrcSet"}
    derived = m.section("productCarousel").fields["tagId"]
    assert derived.read_only and derived.derived_from == "tagSlug"
    keyed = m.section("contentBlock").fields["locationOverrides.{locationId}.mode"]
    assert keyed.keyed_by == "location" and keyed.enum == ["address", "coords"]


def test_list_spec_merges_item_ref_details(m):
    hero = m.section("heroBanner")
    bullets = hero.lists["slides[].bullets"]
    assert bullets.depth == 2
    assert bullets.item_ref == "details.HeroBulletDetailsPanel"
    assert set(bullets.item_fields) >= {"title", "description", "icon"}  # `item.` prefix stripped
    assert bullets.ops == ["add", "remove", "reorder"] and bullets.allows("update")
    slides = hero.lists["slides"]
    assert slides.depth == 1 and "layout" in slides.item_fields and not slides.allows("reorder")
    assert m.section("gallery").lists["images"].scalar
    assert m.section("featuredCategories").lists["pageVisibility.products.categoryOrder"].orders == "categories"


def test_details_panel_wins_a_key_both_spell(m):
    certs = m.branding.lists["certifications"]
    assert "enabled" in certs.item_fields  # the list's own item field
    assert "name" in certs.item_fields and "image" in certs.item_fields  # from CertificationDetailsPanel


def test_container_targets(m):
    assert m.section("contentBlock").target("home", "contentBlock-2") == {"kind": "section", "page": "home", "sectionId": "contentBlock-2"}
    assert m.panel("HeaderSection").target() == {"kind": "global", "panel": "HeaderSection"}
    assert m.branding.target() == {"kind": "branding"}
    assert m.page("faq").target() == {"kind": "page", "page": "faq"}
    with pytest.raises(ValueError):
        m.detail("FaqDetailsPanel").target()


def test_common_page_fields_substitute_page(m):
    f = m.common_page_fields("faq")["published"]
    assert f.path == "pageVisibility.faq.published" and f.default is True
    assert "published" in m.page_fields("products") and "layout" in m.page_fields("faq")
    assert "_common" not in m.pages


def test_mirror_keys_resolve_to_slide_fields(m):
    hero = m.section("heroBanner")
    assert hero.mirror_from == "slides[0]"
    f = m.mirror_item_field(hero, "headline")
    assert f is not None and f.path == "settings.slides[].headline"
    assert m.mirror_item_field(hero, "paddingY") is None  # a real top-level field, not a mirror key
    assert m.resolve_field(hero, "paddingY").path == "settings.paddingY"
    assert m.resolve_field(hero, "headline").type == "richtext"
    assert m.resolve_field(hero, "nope") is None


def test_iterators_cover_every_field(m):
    n_top = sum(1 for _ in m.iter_fields())
    n_item = sum(len(l.spec.get("item") or {}) for _, l in m.iter_lists())
    assert n_top + n_item == 2094
    assert sum(1 for _ in m.iter_lists()) == 69


def test_holdout_is_stable_and_near_twenty_percent():
    assert M.is_holdout("x") == M.is_holdout("x")
    assert 0 <= M.holdout_bucket("contentBlock.title") < 100
    keys = [f"c{i}.f{j}" for i in range(50) for j in range(40)]
    pct = 100 * sum(M.is_holdout(k) for k in keys) / len(keys)
    assert 17 <= pct <= 23


def test_enumerate_targets_excludes_derived_and_counts(m):
    rows = [{"id": "heroBanner"}, {"id": "productCarousel-1"}, {"id": "contentBlock-1"}, {"id": "gallery"}]
    ts = M.enumerate_targets(m, rows)
    assert all(not t.field.read_only for t in ts)
    assert not any(t.field.key in ("tagId", "brandId") for t in ts)
    sections = [t for t in ts if t.container.kind == "section"]
    assert {t.section_id for t in sections} == {"heroBanner", "productCarousel-1", "contentBlock-1", "gallery"}
    mirror = [t for t in ts if t.via_list == "slides"]
    assert mirror and all(t.section_id == "heroBanner" for t in mirror)
    assert any(t.field.keyed_by for t in ts)
    kinds = {t.container.kind for t in ts}
    assert kinds == {"section", "global", "branding", "page"}


def test_enumerate_targets_default_rows_measured(m):
    from souschef_lab.state import DEFAULT_HOME_SECTIONS
    ts = M.enumerate_targets(m, DEFAULT_HOME_SECTIONS)
    assert len(ts) == 1894  # measured 2026-10-01 on the 727d151b snapshot (13 default rows, contentBlock-features/icons in the family)
    hold = sum(t.holdout for t in ts)
    assert 17 <= 100 * hold / len(ts) <= 23
    by_type = M.targets_by_type(ts)
    assert set(by_type) <= set(M.FIELD_TYPES)
    assert by_type["json"] == 5


def test_target_key_and_holdout_independent_of_wording(m):
    c = m.section("contentBlock")
    t1 = M.WritableTarget(c, c.fields["title"], "contentBlock-1")
    t2 = M.WritableTarget(c, c.fields["title"], "contentBlock-2")
    assert t1.key == t2.key == "contentBlock.title"
    assert t1.holdout == t2.holdout
    assert t1.target()["sectionId"] == "contentBlock-1"


def test_main_prints_hash(capsys):
    assert M.main([str(MANIFEST)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hash"] == VECTOR and out["fields"] == 2094
