"""retrain/rebase.py: every pool family onto another manifest (design §3.2 step 1).
Engine-backed tests need node + the cookie-cutter snapshot (COOKIE_CUTTER_DIR)."""
from __future__ import annotations

import copy
import json
import os

import pytest

from souschef_lab import FIXTURES_DIR
from souschef_lab import engine as E
from souschef_lab.manifest import Manifest
from souschef_lab.retrain import rebase as RB

EXAM_MANIFEST = FIXTURES_DIR / "exam" / "manifest-real.json"
REQUIRE = os.environ.get("SOUS_CHEF_LAB_REQUIRE_ENGINE") == "1"
BTN = {"kind": "global", "panel": "ButtonsSection"}


def _need_engine():
    if not E.engine_available():
        msg = "node and the engine (COOKIE_CUTTER_DIR) are needed: " + E.unavailable_reason()
        if REQUIRE:
            pytest.fail(msg)
        pytest.skip(msg)


def _doc():
    return json.loads(EXAM_MANIFEST.read_text(encoding="utf-8"))


def drifted(rename=True, drop_input=False, relabel=False, drop_panel=None) -> dict:
    """The exam manifest with borderRadius renamed (same label), inputBorderRadius removed, a label changed."""
    d = _doc()
    f = d["globalPanels"]["ButtonsSection"]["fields"]
    if rename:
        f["cornerRadius"] = f.pop("borderRadius")
    if drop_input:
        f.pop("inputBorderRadius")
    if relabel:
        f["buttonTextColor"]["label"] = "Button Label Colour"
    if drop_panel:
        d["globalPanels"].pop(drop_panel)
    return d


def test_split_addr_reads_lists_and_branding():
    assert RB.split_addr("sections.heroBanner.slides[].bullets[].title") == ("sections.heroBanner", "slides[].bullets", "title")
    assert RB.split_addr("globalPanels.HeaderSection.announcementBar.message") == \
        ("globalPanels.HeaderSection", "", "announcementBar.message")
    assert RB.split_addr("branding.logoUrl") == ("branding", "", "logoUrl")


def test_renames_need_same_type_and_a_close_label():
    old, new = _doc(), drifted()
    rn = RB.renames_between(old, new)
    assert {"from": "globalPanels.ButtonsSection.borderRadius", "to": "globalPanels.ButtonsSection.cornerRadius",
            "score": 1.0} in rn
    # a removed field whose only added sibling has another type is no rename
    new2 = _doc()
    f = new2["globalPanels"]["ButtonsSection"]["fields"]
    spec = f.pop("borderRadius")
    f["cornerRadius"] = {**spec, "type": "number"}
    assert not [r for r in RB.renames_between(old, new2) if r["from"].endswith("borderRadius")]
    # a different label is no rename
    new3 = drifted()
    new3["globalPanels"]["ButtonsSection"]["fields"]["cornerRadius"].update(label="Something else entirely", help="Other help")
    assert not [r for r in RB.renames_between(old, new3) if r["from"].endswith("borderRadius")]


def _case(key="borderRadius", value=4, family="F1", **kw) -> dict:
    op = {"op": "set_fields", "target": dict(BTN), "values": {key: value}}
    c = {"id": f"t-{family}-{key}", "kind": "set_range", "target": dict(BTN), "fields": [key], "site_state": None,
         "context": {"page": "home"}, "request": f"buttons {key} {value}", "history": [],
         "expected": {"ops": [op], "diff": [], "keys": [{"target": dict(BTN), "key": key, "list": ""}], "alt": [],
                      "alt_exact": [], "out_of_scope": None, "direction": {"key": key, "dir": "up", "from": 2}},
         "score": {"mode": "exact"}, "holdout": False, "manifest_hash": Manifest(_doc()).hash, "engine_fingerprint": None,
         "template": "vintage", "meta": {"family": family, "page": "home", "view": None, "oracle_ops": [copy.deepcopy(op)]}}
    c.update(kw)
    return c


def test_rename_case_moves_every_reference():
    old = Manifest(_doc())
    rmap = RB.rename_map(RB.renames_between(old.doc, drifted()))
    c = _case()
    c["history"] = [{"user": "x", "ops": [copy.deepcopy(c["expected"]["ops"][0])],
                     "calls": [{"name": "set_fields", "input": {"target": dict(BTN), "values": {"borderRadius": 2}}}]}]
    done = RB.rename_case(c, old, rmap)
    assert "globalPanels.ButtonsSection.borderRadius" in done
    assert c["expected"]["ops"][0]["values"] == {"cornerRadius": 4}
    assert c["meta"]["oracle_ops"][0]["values"] == {"cornerRadius": 4}
    assert c["history"][0]["ops"][0]["values"] == {"cornerRadius": 4}
    assert c["history"][0]["calls"][0]["input"]["values"] == {"cornerRadius": 2}
    assert c["expected"]["keys"][0]["key"] == "cornerRadius"
    assert c["expected"]["direction"]["key"] == "cornerRadius"
    assert c["fields"] == ["cornerRadius"]


def test_missing_reason_by_key_list_and_container():
    m = Manifest(drifted(drop_input=True, drop_panel="SocialMediaSection"))
    assert RB.missing_reason({"op": "set_fields", "target": BTN, "values": {"inputBorderRadius": 2}}, m) == "key_removed"
    assert RB.missing_reason({"op": "set_fields", "target": BTN, "values": {"cornerRadius": 2}}, m) is None
    assert RB.missing_reason({"op": "set_fields", "target": {"kind": "global", "panel": "SocialMediaSection"},
                              "values": {"instagramUrl": "x"}}, m) == "container_removed"
    faq = {"kind": "page", "page": "faq"}
    assert RB.missing_reason({"op": "edit_list", "target": faq, "list": "faqCategories[].items", "action": "add",
                              "index": [0], "item": {"q": "Q?", "a": "A."}}, m) is None
    assert RB.missing_reason({"op": "edit_list", "target": faq, "list": "nope", "action": "add", "item": {}}, m) == "list_removed"
    assert RB.missing_reason({"op": "edit_list", "target": {"kind": "page", "page": "home"}, "list": "sections",
                              "action": "add", "item": {"id": "noSuchSection"}}, m) == "container_removed"
    assert RB.missing_reason({"op": "set_page_layout", "page": "products", "layout": "grid"}, m) is None


def test_relabel_rewrites_the_verbatim_phrase():
    old, new = Manifest(_doc()), Manifest(drifted(relabel=True))
    c = _case(key="buttonTextColor", value="#111111", family="F0")
    c["request"] = "Change the Button Text Color on the buttons to #111111."
    c["meta"].update(style="verbatim", field_phrase="Button Text Color")
    rl = RB.relabel(c, old, new)
    assert rl and rl["from"] == "Button Text Color"
    assert "Button Label Colour" in c["request"] and "#111111" in c["request"]
    assert c["meta"]["relabelled"]["request_was"].startswith("Change the Button Text Color")
    # a label the request does not quote is left as written (and noted)
    c2 = _case(key="buttonTextColor", value="#111111", family="F0")
    c2["meta"].update(style="verbatim", field_phrase="Button Text Color")
    c2["request"] = "make the text on buttons #111111"
    assert RB.relabel(c2, old, new) is None and c2["meta"]["relabel_missed"]


@pytest.fixture(scope="module")
def drift_engine(tmp_path_factory):
    _need_engine()
    d = tmp_path_factory.mktemp("m")
    p = d / "manifest.json"
    p.write_text(json.dumps(drifted(drop_input=True)), encoding="utf-8")
    eng = E.PersistentEngine(manifest_path=p, size=2)
    yield eng, Manifest(json.loads(p.read_text()))
    eng.close()


def _built(c: dict, m: Manifest) -> dict:
    from souschef_lab import state as S
    c["site_state"] = S.build_state(m, template="vintage")
    return c


def test_rebase_pool_renames_drops_and_measures(drift_engine):
    from souschef_lab.state import load_context_fixture
    eng, m_new = drift_engine
    old = Manifest(_doc())
    renamed = _built(_case(), old)
    gone = _built(_case(key="inputBorderRadius", value=3), old)
    gone["id"] = "t-F1-gone"
    other = _built(_case(key="buttonColor", value="#123456"), old)
    other["id"] = "t-F1-other"
    hold = _built(_case(key="buttonColor", value="#654321", family="F0H"), old)
    hold.update(id="t-F0H-1", holdout=True)
    kept, rep = RB.rebase_pool([renamed, gone, other, hold], {old.hash: old}, m_new, eng, load_context_fixture(),
                               check_leak=False, workers=2, progress=lambda s: None)
    ids = {c["id"] for c in kept}
    assert ids == {"t-F1-borderRadius", "t-F1-other", "t-F0H-1"}
    r = next(c for c in kept if c["id"] == "t-F1-borderRadius")
    assert r["expected"]["ops"][0]["values"] == {"cornerRadius": 4}
    assert r["expected"]["diff"], "the exact diff is re-measured on the new manifest"
    assert r["manifest_hash"] == m_new.hash and r["engine_fingerprint"] == eng.fingerprint
    assert r["meta"]["rebased_history"][-1]["manifest_hash"] == old.hash
    assert rep["families"]["F1"]["dropped:key_removed"] == 1
    assert rep["dropped"] == [{"id": "t-F1-gone", "family": "F1", "why": "key_removed"}]
    # one of three F1 cases lost is over the 20 % gate — only families of >= 5 cases are gated
    assert rep["loss_failed"] == []
    many = [_built(_case(key="inputBorderRadius", value=i % 16 + 1), old) for i in range(5)]
    for i, c in enumerate(many):
        c["id"] = f"t-F6-{i}"
        c["meta"]["family"] = "F6"
    _, rep2 = RB.rebase_pool(many, {old.hash: old}, m_new, eng, load_context_fixture(), check_leak=False,
                             progress=lambda s: None)
    assert rep2["loss_failed"] == ["F6"]


def test_a_case_already_on_the_manifest_and_engine_is_left_as_is(drift_engine):
    eng, m_new = drift_engine
    c = _case(key="cornerRadius")
    c.update(manifest_hash=m_new.hash, engine_fingerprint=eng.fingerprint)
    out, why, notes = RB.rebase_case(c, m_new, m_new, {}, eng, {})
    assert why == "" and notes == {"unchanged": True} and out == c
