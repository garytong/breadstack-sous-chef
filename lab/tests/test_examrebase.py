"""examrebase: the exam on another manifest (DESIGN §1.3 arm D) and the
static diff's field addressing. No engine, no network."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from souschef_lab import exam as X  # noqa: E402
from souschef_lab.examrebase import (ManifestIndex, address_of, ops_valid, rebase_exam,  # noqa: E402
                                     rebase_generated, rename_op)
from souschef_lab.manifest import Manifest  # noqa: E402

REAL = json.loads(X.MANIFEST_FILE.read_text(encoding="utf-8"))


def drifted(base: dict | None = None) -> dict:
    """The exam's manifest as a later launcher might register it: a rename
    (ButtonsSection.buttonColor → buttonPrimaryColor, same label), a removed
    field (ShopMenuSection.shopMenuMaxItems), an added field and an added
    global panel, wording changes, an enum option, the snapshot's commit."""
    d = copy.deepcopy(base or REAL)
    gp = d["globalPanels"]
    b = gp["ButtonsSection"]["fields"]
    b["buttonPrimaryColor"] = b.pop("buttonColor")
    b["buttonShadowColor"] = {"type": "color", "label": "Button Shadow Colour", "path": "globalSettings.buttonShadowColor",
                              "source": {"file": "x.jsx", "line": 1}, "unset_value": ""}
    gp["ShopMenuSection"]["fields"].pop("shopMenuMaxItems")
    gp["LoyaltyBanner"] = {"fields": {
        "loyaltyText": {"type": "text", "label": "Loyalty Banner Text", "path": "globalSettings.loyaltyText",
                        "source": {"file": "y.jsx", "line": 2}, "default": ""},
        "showLoyalty": {"type": "bool", "label": "Show Loyalty Banner", "path": "globalSettings.showLoyalty",
                        "source": {"file": "y.jsx", "line": 3}, "default": False}}, "lists": {}}
    gp["FooterSection"]["fields"]["footerBgColor"]["label"] = "Footer Background Colour"
    gp["FooterSection"]["fields"]["footerBgColor"]["source"] = {"file": "moved.jsx", "line": 9}
    body = gp["TypographyColorsSection"]["fields"]["bodyFont"]
    body["enum"] = list(body.get("enum") or []) + ["Comic Neue"]
    d["generated_from"] = {**d.get("generated_from", {}), "sha": "e7e90106", "build_id": "e7e90106"}
    return d


def rows_all() -> list[dict]:
    out = []
    for s in X.SETS:
        out += X.load_set(s)
    return out


RENAME = [{"from": "globalPanels.ButtonsSection.buttonColor", "to": "globalPanels.ButtonsSection.buttonPrimaryColor",
           "from_key": "buttonColor", "to_key": "buttonPrimaryColor", "container": "globalPanels.ButtonsSection", "list": None,
           "score": 1.0}]


def test_every_exam_key_has_an_address_on_its_manifest():
    m = Manifest(REAL)
    idx = ManifestIndex(m)
    for r in rows_all():
        for s in X.expected_sets(r):
            for e in s:
                a, found = address_of(m, idx, e)
                assert found, (r["id"], e, a)


def test_addresses_cover_lists_mirror_categories_and_pages():
    m = Manifest(REAL)
    idx = ManifestIndex(m)
    assert address_of(m, idx, {"target": {"kind": "page", "page": "home"}, "list": "sections", "key": ""}) == (None, True)
    a, ok = address_of(m, idx, {"target": {"kind": "global", "panel": "HeaderSection"}, "key": "announcementBar.message", "list": ""})
    assert ok and a == "globalPanels.HeaderSection.announcementBar.message"
    a, ok = address_of(m, idx, {"target": {"kind": "section", "sectionId": "heroBanner", "page": "home"}, "key": "layout",
                                "list": "slides"})
    assert ok and a == "sections.heroBanner.slides[].layout"
    a, ok = address_of(m, idx, {"target": {"kind": "global", "panel": "HeaderSection"}, "key": "nope", "list": ""})
    assert not ok and a == "globalPanels.HeaderSection.nope"


def test_rebase_maps_renames_and_sets_stale_rows_apart():
    rows = rows_all()
    new = Manifest(drifted())
    out, stale, rep = rebase_exam(rows, new, RENAME)
    assert rep["rows"] == len(rows) == len(out) + len(stale)
    assert rep["renamed_rows"] >= 6
    assert {r["meta"]["stale"]["missing"][0] for r in stale}
    assert all("shopMenuMaxItems" in json.dumps(r["expected"]["keys"]) for r in stale)
    assert all(r["manifest_hash"] == new.hash for r in out + stale)
    keys = json.dumps([r["expected"] for r in out])
    assert "buttonPrimaryColor" in keys and '"buttonColor"' not in keys
    renamed = [r for r in out if r["meta"]["rebased_from"]["renames"]]
    assert renamed and all(r["meta"]["rebased_from"]["manifest_hash"] == Manifest(REAL).hash for r in renamed)
    for r in out:  # every kept row's keys resolve on the new manifest
        for s in X.expected_sets(r):
            for e in s:
                assert X.check_expectation(new, e) is None


def test_rebase_without_a_rename_makes_the_renamed_rows_stale():
    out, stale, rep = rebase_exam(rows_all(), drifted(), None)
    assert rep["stale"] >= 6 + 7 - 1
    assert any("buttonColor" in json.dumps(r["expected"]["keys"]) for r in stale)


def test_out_of_scope_rows_pass_through():
    oos = X.load_set("out_of_scope")
    out, stale, rep = rebase_exam(oos, drifted(), RENAME)
    # p137.k2 sits in the set but is a create_page row since 2026-10-15 (exam.AMENDMENTS): kept, not counted out of scope
    assert len(out) == len(oos) and not stale and rep["out_of_scope"] == sum(1 for r in oos if X.is_out_of_scope(r)) == 12


def test_ops_are_renamed_and_checked():
    old = Manifest(REAL)
    op = {"op": "set_fields", "target": {"kind": "global", "panel": "ButtonsSection"}, "values": {"buttonColor": "#112233"}}
    new_op, applied = rename_op(old, ManifestIndex(old), op, {r["from"]: r for r in RENAME})
    assert new_op["values"] == {"buttonPrimaryColor": "#112233"} and applied
    new = Manifest(drifted())
    assert ops_valid(new, [op]) and not ops_valid(new, [new_op])


def test_rebase_generated_drops_cases_writing_a_removed_key():
    t = {"kind": "global", "panel": "ShopMenuSection"}
    gone = {"id": "c1", "manifest_hash": Manifest(REAL).hash, "expected": {"ops": [{"op": "set_fields", "target": t,
                                                                                    "values": {"shopMenuMaxItems": 4}}]}, "meta": {}}
    t2 = {"kind": "global", "panel": "ButtonsSection"}
    ren = {"id": "c2", "manifest_hash": Manifest(REAL).hash, "expected": {"ops": [{"op": "set_fields", "target": t2,
                                                                                   "values": {"buttonColor": "#000000"}}]},
           "meta": {"oracle_ops": [{"op": "set_fields", "target": t2, "values": {"buttonColor": "#000000"}}]}}
    out, rep = rebase_generated([gone, ren], drifted(), RENAME, old_manifest=REAL)
    assert rep["dropped"] == {"key_removed": 1} and rep["kept"] == 1 and rep["renamed"] == 1
    assert out[0]["expected"]["ops"][0]["values"] == {"buttonPrimaryColor": "#000000"}
    assert out[0]["meta"]["oracle_ops"][0]["values"] == {"buttonPrimaryColor": "#000000"}
