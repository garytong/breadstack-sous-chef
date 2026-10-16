"""exam.py: the real-phrasing exam (docs/LAB.md §6b) — the committed sets,
the end-to-end scorer, the offline retrieval scorer, multi-turn replay through
the fake service, the oracle and the conversation replay harness.

No network, no model, no API spend. The retrieval test needs go (skipped
otherwise; SOUS_CHEF_LAB_REQUIRE_GO=1 fails instead); the oracle dry run
needs the real engine (skipped otherwise; SOUS_CHEF_LAB_REQUIRE_ENGINE=1
fails instead).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
for _p in (LAB, LAB / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fake_service as FS  # noqa: E402
from souschef_lab import collect as C  # noqa: E402
from souschef_lab import exam as X  # noqa: E402
from souschef_lab import fields as F  # noqa: E402
from test_collect import TINY, TINY_HASH, _engine_reason, call, done, fake_engine, waiting  # noqa: E402

SIZES = {"paraphrase": 147, "holdout": 62, "real": 14, "out_of_scope": 13, "out_of_scope_holdout": 4, "followup": 12}
REAL_HASH = "bf78870182302f821856057d0091d62b09403c5180716c4f37a4830761e18a12"
STORE = {"target": {"kind": "global", "panel": "BrandingSection"}, "key": "storeName", "list": ""}
SEO = {"target": {"kind": "page", "page": "seo"}, "key": "defaults.title", "list": ""}
HERO_HEAD = {"target": {"kind": "section", "page": "home", "sectionId": "heroBanner"}, "key": "headline", "list": "slides"}


def case(**over) -> dict:
    c = {"id": "x1", "set": "t", "kind": "phrasing", "context": {"page": "home"}, "request": "change the site name",
         "history": [], "expected": {"ops": [], "diff": [], "keys": [STORE], "alt": [[SEO]], "out_of_scope": None},
         "meta": {"class": "synonym"}}
    c.update(over)
    return c


def row(*ops_ok, reply="Done.", calls=()) -> dict:
    """A finished collector row: ops with the engine's ok per op."""
    ops = [o for o, _ in ops_ok]
    return {"status": "ok", "ops": ops, "engine_results": [{"results": [{"ok": ok} for _, ok in ops_ok]}],
            "reply": reply, "tool_calls": list(calls)}


def sf(target, **values):
    return {"op": "set_fields", "target": target, "values": values}


# --- the committed sets ------------------------------------------------------------------------------------------------------

def test_sets_have_the_studys_sizes_and_the_lab_schema():
    index = json.loads(X.INDEX_FILE.read_text(encoding="utf-8"))
    assert {k: v["rows"] for k, v in index["sets"].items()} == SIZES
    for name, n in SIZES.items():
        rows = X.load_set(name)
        assert len(rows) == n and len({r["id"] for r in rows}) == n
        for r in rows:
            assert {"id", "set", "kind", "target", "fields", "site_state", "context", "request", "history", "expected",
                    "holdout", "manifest_hash", "engine_fingerprint", "template", "meta"} <= set(r)
            assert r["set"] == name and r["holdout"] is True and r["template"] == "vintage"
            assert r["context"]["template"] == "vintage" and r["context"]["page"] == r["meta"]["page"]
            exp = r["expected"]
            assert exp["ops"] == [] and exp["diff"] == []
            if r["kind"] == "create_page":  # p137.k2 (2026-10-15): creating a page is create_page, not out of scope
                assert r["id"] == "p137.k2" and exp["page"]["title"] == "Events" and exp["out_of_scope"] is None
            elif name.startswith("out_of_scope"):
                assert r["kind"] == "out_of_scope" and exp["keys"] == [] and exp["out_of_scope"]["reason"]
            else:
                assert r["kind"] == "phrasing" and exp["keys"] and exp["out_of_scope"] is None
                assert r["fields"] == [e["key"] or e["list"] for e in exp["keys"]]


def test_the_manifest_is_recorded_by_hash():
    index = json.loads(X.INDEX_FILE.read_text(encoding="utf-8"))
    assert index["manifest"]["manifest_hash"] == REAL_HASH == X.manifest_hash_of(X.MANIFEST_FILE)
    assert index["manifest"]["file_sha256"] == hashlib.sha256(X.MANIFEST_FILE.read_bytes()).hexdigest()
    assert index["manifest"]["generated_from"]["build_id"] == "9cc1cd1"
    assert {r["manifest_hash"] for name in SIZES for r in X.load_set(name)} == {REAL_HASH}
    with pytest.raises(X.ExamError, match="pass the set's own manifest"):
        X.load_set("real", manifest=LAB / "fixtures" / "manifest.snapshot.json")


def test_every_expected_key_resolves_on_the_manifest():
    from souschef_lab.manifest import Manifest

    m = Manifest(json.loads(X.MANIFEST_FILE.read_text(encoding="utf-8")))
    bad = [(r["id"], X.check_expectation(m, e)) for name in SIZES for r in X.load_set(name)
           for s in X.expected_sets(r) for e in s if X.check_expectation(m, e)]
    assert bad == []
    assert X.check_expectation(m, {"target": {"kind": "global"}, "key": "categories.edibles.nope", "list": ""})
    assert X.check_expectation(m, {"target": {"kind": "section", "sectionId": "heroBanner"}, "key": "nope", "list": ""})


def test_multi_turn_rows_carry_their_history():
    follow = X.load_set("followup")
    para = {r["id"]: r for r in X.load_set("paraphrase")}
    assert all(r["meta"]["class"] == "followup" and len(r["history"]) == 1 for r in follow)
    for r in follow:  # the same rows as the paraphrase set's follow-ups
        assert r["history"] == para[r["id"]]["history"] and r["history"][0]["user"] and r["history"][0]["assistant"]
    real = {r["id"]: r for r in X.load_set("real")}
    assert real["r00"]["history"] == [] and real["r09"]["history"] == []  # each conversation's first turn
    assert real["r06"]["history"][0]["user"] == "there are too many categories in the header"
    assert real["r11"]["history"][0]["context"]["page"] == "home"  # the turn before was on another page
    assert real["r12"]["context"]["layout"] == "tab" and real["r11"]["context"]["layout"] == "filter-left"


def test_contexts_are_the_recorded_ones_where_the_user_was():
    index = json.loads(X.INDEX_FILE.read_text(encoding="utf-8"))
    pages = index["contexts"]["pages"]
    assert {p for p, v in pages.items() if v["source"] == "recorded"} == {"home", "products", "product-details", "faq"}
    cart = X.load_set("paraphrase")
    approx = [r for r in cart if r["meta"]["context_source"] == "approximated"]
    assert approx and all(r["context"]["sections"] == [] and r["context"]["layout"] == "default" for r in approx)
    sel = next(r for r in X.load_set("paraphrase") if r["meta"].get("selected_section_id"))
    assert "selectedSectionId" not in sel["context"]  # the browser does not send it today
    picked = X.load_set("paraphrase", with_selected=True)[cart.index(sel)]["context"]
    assert picked["selectedSectionId"] == sel["meta"]["selected_section_id"]
    # … and as the browser sends it since the 2026-10-10 context work
    assert picked["focus"] == {"area": "editor", "sectionId": sel["meta"]["selected_section_id"], "panel": ""}


# --- the end-to-end scorer -----------------------------------------------------------------------------------------------------

def test_score_pass_alt_stray_partial_wrong():
    branding = {"kind": "global", "panel": "BrandingSection"}
    assert X.score_turn(case(), row((sf(branding, storeName="H"), True)))["verdict"] == "pass"
    assert X.score_turn(case(), row((sf({"kind": "global"}, storeName="H"), True)))["verdict"] == "pass"  # merged global
    assert X.score_turn(case(), row((sf({"kind": "page", "page": "seo"}, **{"defaults.title": "H"}), True)))["verdict"] == "pass"
    both = X.score_turn(case(), row((sf(branding, storeName="H"), True), (sf({"kind": "page", "page": "seo"}, **{"defaults.title": "H"}), True)))
    assert both["verdict"] == "pass"  # writes inside expect ∪ alt are never stray
    stray = X.score_turn(case(), row((sf(branding, storeName="H", locationPickerShowName=True), True)))
    assert stray["verdict"] == "stray_writes" and stray["keys_hit"] and any("locationPickerShowName" in s for s in stray["stray"])
    wrong = X.score_turn(case(), row((sf({"kind": "branding"}, locationPickerShowName=True), True)))
    assert wrong["verdict"] == "wrong_target" and not wrong["keys_hit"]
    two = case(expected={"keys": [STORE, dict(STORE, key="logo")], "alt": [], "out_of_scope": None})
    assert X.score_turn(two, row((sf(branding, storeName="H"), True)))["verdict"] == "partial"
    assert X.score_turn(case(), row((sf(branding, storeName="H"), False)))["verdict"] == "invalid_op"  # refused = not written
    assert X.score_turn(case(), row(reply="Which name?"))["verdict"] == "no_write"
    assert X.score_turn(case(expected={"keys": [], "alt": []}), row())["verdict"] is None  # an unlabelled replay row


def test_score_lists_and_the_hero_mirror():
    c = case(expected={"keys": [HERO_HEAD], "alt": []})
    hero = {"kind": "section", "sectionId": "heroBanner"}  # page omitted: the host fills the viewed page
    via_list = {"op": "edit_list", "target": hero, "list": "slides", "action": "update", "index": 0, "item": {"headline": "x"}}
    assert X.score_turn(c, row((via_list, True)))["verdict"] == "pass"
    assert X.score_turn(c, row((sf(hero, headline="x"), True)))["verdict"] == "pass"  # the flat mirror of slides[0]
    faq = case(context={"page": "faq"}, expected={"keys": [{"target": {"kind": "page", "page": "faq"}, "key": "",
                                                           "list": "faqCategories[].items"}], "alt": []})
    add = {"op": "edit_list", "target": {"kind": "page", "page": "faq"}, "list": "faqCategories[].items", "action": "add",
           "index": 0, "item": {"q": "?", "a": "!"}}
    assert X.score_turn(faq, row((add, True)))["verdict"] == "pass"
    other = dict(add, list="faqCategories")
    assert X.score_turn(faq, row((other, True)))["verdict"] == "wrong_target"
    cat = case(expected={"keys": [{"target": {"kind": "global"}, "key": "categories.pre-rolls.name", "list": ""}], "alt": []})
    assert X.score_turn(cat, row((sf({"kind": "global"}, **{"categories.pre-rolls.name": "Blunts"}), True)))["verdict"] == "pass"
    nested = sf({"kind": "global"}, categories={"pre-rolls": {"name": "Blunts"}})
    assert X.score_turn(cat, row((nested, True)))["verdict"] == "pass"


def test_out_of_scope_needs_no_write_and_a_place():
    oos = case(kind="out_of_scope", expected={"keys": [], "alt": [], "out_of_scope": {"reason": "catalog", "where": ["catalog"]}})
    assert X.score_turn(oos, row(reply="Products are added in your Breadstack dashboard."))["verdict"] == "unsupported_ok"
    assert X.score_turn(oos, row(reply="That lives in the catalog."))["verdict"] == "unsupported_ok"  # the row's own word
    assert X.score_turn(oos, row(reply="Sorry, I can't add products."))["verdict"] == "no_pointer"
    assert X.score_turn(oos, row(reply="I can't do that here, but it's possible elsewhere — see the blog post."))["verdict"] == "no_pointer"
    reported = row(reply="", calls=[{"name": "report_unsupported", "input": {"reason": "Not in the editor.",
                                                                              "manual_hint": "Use the POS."}}])
    assert X.score_turn(oos, reported)["verdict"] == "unsupported_ok"
    wrote = X.score_turn(oos, row((sf({"kind": "branding"}, x=1), True), reply="Done in Breadstack."))
    assert wrote["verdict"] == "wrote"
    assert X.score_turn(oos, row((sf({"kind": "branding"}, x=1), False), reply="See Breadstack."))["verdict"] == "unsupported_ok"
    assert C.passed({"status": "ok", "verdict": "unsupported_ok"}) is True  # eval's pass counting applies unchanged


def test_every_out_of_scope_row_names_its_where():
    """Contract 1: every out-of-scope row carries `expected_where`, values of
    the shared enum, and the build derives the same from reason + request."""
    want = {"p135": "products", "p136": "products", "p138": "themes", "p139": "domain",
            "p140": "support", "p141": "categories", "p142": "support", "p143": "support", "p144": "support",
            "p145": "domain", "p146": "support", "p147": "support", "h063": "support", "h064": "products",
            "h065": "support", "h066": "support"}
    got = {}
    for name in ("out_of_scope", "out_of_scope_holdout"):
        for r in X.load_set(name):
            if r["id"] == "p137.k2":  # create_page since 2026-10-15 (test_p137_expects_create_page)
                continue
            o = r["expected"]["out_of_scope"]
            assert o["expected_where"] == X.expected_where_for(o["reason"], r["request"])
            assert set(o["expected_where"]) <= set(X.WHERE_VALUES)
            got[r["id"]] = o["expected_where"][0]
    assert got == want
    assert X.WHERE_VALUES == ("domain", "sites", "pages", "assets", "seo", "themes", "products", "categories", "brands",
                              "support")  # service toolgen.UnsupportedWhere / launcher deepLinks.js WHERE, same order


def test_out_of_scope_is_scored_by_where_when_the_service_sends_one():
    oos = case(kind="out_of_scope", expected={"keys": [], "alt": [], "out_of_scope": {
        "reason": "catalog", "where": ["catalog"], "expected_where": ["products"]}})

    def rep(where, reply="", **extra):
        return row(reply=reply, calls=[{"name": "report_unsupported", "input": {"reason": "Not here.", "where": where, **extra}}])

    ok = X.score_turn(oos, rep("products"))
    assert ok["verdict"] == "unsupported_ok" and ok["scored_by"] == "where" and ok["where_sent"] == ["products"]
    assert X.score_turn(oos, rep(" Products "))["verdict"] == "unsupported_ok"  # normalised as the browser does
    wrong = X.score_turn(oos, rep("support", reply="It's in the catalog."))  # the words would pass; the link does not
    assert wrong["verdict"] == "wrong_where" and wrong["reasons"] == ["where:support not products"]
    assert C.passed({"status": "ok", "verdict": "wrong_where"}) is False
    # no usable `where` → today's keyword rule (a baseline revision before the enum is scored as before)
    bogus = X.score_turn(oos, rep("the catalog page", reply="That lives in the catalog."))
    assert bogus["verdict"] == "unsupported_ok" and bogus["scored_by"] == "keywords" and bogus["where_raw"] == ["the catalog page"]
    assert X.score_turn(oos, rep("nowhere"))["verdict"] == "no_pointer"
    legacy = X.score_turn(oos, row(reply="Products are added in your Breadstack dashboard."))
    assert legacy["verdict"] == "unsupported_ok" and legacy["scored_by"] == "keywords" and legacy["where_sent"] == []
    # a row without expected_where (an older set) is keyword-scored whatever the service sends
    old = case(kind="out_of_scope", expected={"keys": [], "alt": [], "out_of_scope": {"reason": "catalog", "where": ["catalog"]}})
    assert X.score_turn(old, rep("support", reply="See the catalog."))["scored_by"] == "keywords"
    # a write fails the row whatever `where` says
    wrote = X.score_turn(oos, row((sf({"kind": "branding"}, x=1), True), calls=[
        {"name": "report_unsupported", "input": {"reason": "x", "where": "products"}}]))
    assert wrote["verdict"] == "wrote" and wrote["scored_by"] == "where"
    # the secondary score: the keyword rule on the words, whatever `where` was sent (both revisions alike)
    assert ok["keyword_verdict"] == "no_pointer"  # a bare where:products with a reason naming no place
    assert wrong["keyword_verdict"] == "unsupported_ok" and legacy["keyword_verdict"] == "unsupported_ok"
    assert wrote["keyword_verdict"] == "wrote"


def test_a_run_report_never_overwrites_another(tmp_path):
    first = X.unused_path(tmp_path, "exam-real-x-r0-7-20261010T000000Z")
    first.write_text("{}")
    second = X.unused_path(tmp_path, "exam-real-x-r0-7-20261010T000000Z")
    assert second.name == "exam-real-x-r0-7-20261010T000000Z-2.json"
    second.write_text("{}")
    assert X.unused_path(tmp_path, "exam-real-x-r0-7-20261010T000000Z").name.endswith("-3.json")


# --- the offline retrieval scorer ----------------------------------------------------------------------------------------------

def test_score_retrieval_hit_target_and_key_misses():
    branding = [{"targets": [{"kind": "global", "panel": "BrandingSection"}], "fields": [{"key": "storeName"}], "lists": []}]
    other = [{"targets": [{"kind": "branding"}], "fields": [{"key": "locationPickerShowName"}], "lists": []}]
    seo_wrong_key = [{"targets": [{"kind": "page", "page": "seo"}], "fields": [{"key": "robotsTxt"}], "lists": []}]
    r = X.score_retrieval(case(), {"block": "b" * 10, "targets": other + branding})
    assert r["hit"] and not r["top1"] and r["chars"] == 10 and r["targets"] == 2
    r = X.score_retrieval(case(), {"block": "b", "targets": other})
    assert not r["hit"] and r["target_missed"] and not r["key_missed"]
    seo_only = case(expected={"keys": [SEO], "alt": []})
    r = X.score_retrieval(seo_only, {"block": "b", "targets": seo_wrong_key})  # the target is there, its key is not
    assert not r["hit"] and r["key_missed"] and not r["target_missed"]
    r = X.score_retrieval(case(), {"block": "b", "targets": seo_wrong_key})  # a tie keeps the first set (the study's rule)
    assert not r["hit"] and r["target_missed"]
    hero = [{"targets": [{"kind": "section", "page": "home", "sectionId": "heroBanner"}],
             "fields": [{"key": "headline", "list": "slides"}], "lists": ["slides"]}]
    assert X.score_retrieval(case(expected={"keys": [HERO_HEAD], "alt": []}), {"block": "b", "targets": hero})["hit"]
    oos = X.score_retrieval(case(kind="out_of_scope", expected={"keys": [], "out_of_scope": {"reason": "x"}}),
                            {"block": "b", "targets": other})
    assert oos["out_of_scope"] and oos["nonempty"]
    g = X.retrieval_group([X.score_retrieval(case(), {"block": "", "targets": []}), X.score_retrieval(case(), {"block": "b", "targets": branding})])
    assert g["n"] == 2 and g["hit"] == 1 and g["recall"] == 50.0 and g["empty"] == 1 and g["top1"] == 50.0


def test_retrieval_text_with_history_is_a_what_if():
    c = case(request="yes please", history=[{"user": "make the links pink", "assistant": "Buttons too?"}])
    assert X.retrieval_text(c, False) == "yes please"
    assert X.retrieval_text(c, True) == "make the links pink\nButtons too?\nyes please"


def _need_go() -> None:
    if not F.available():
        if os.environ.get("SOUS_CHEF_LAB_REQUIRE_GO"):
            pytest.fail("SOUS_CHEF_LAB_REQUIRE_GO is set but the sous-chef-fields CLI cannot be built (no go on PATH)")
        pytest.skip("go not on PATH (and SOUS_CHEF_FIELDS_BIN unset)")


def test_retrieval_through_the_services_cli(tmp_path):
    _need_go()
    rows = X.load_set("real")
    rep = X.retrieval(rows, X.MANIFEST_FILE, F.binary())
    assert len(rep["results"]) == 14 and rep["overall"]["n"] == 14
    assert 0 <= rep["overall"]["hit"] <= 14 and rep["overall"]["chars_p50"] > 0
    assert set(rep["by_class"]) == {"real"}
    oos = X.retrieval(X.load_set("out_of_scope"), X.MANIFEST_FILE, F.binary())
    # 12 out of scope; p137.k2 (create_page) has no keys to retrieve: unlabelled for the retrieval score
    assert oos["overall"]["n"] == 0 and oos["overall"]["out_of_scope"]["n"] == 12
    assert [r["id"] for r in oos["results"] if r.get("unlabelled")] == ["p137.k2"]
    assert X.main(["retrieval", "--set", "real", "--out-dir", str(tmp_path)]) == 0
    saved = json.loads(next(tmp_path.glob("exam-retrieval-real-*.json")).read_text(encoding="utf-8"))
    assert saved["manifest_hash"] == REAL_HASH and saved["fields"]["source_sha256"]


def test_follow_ups_go_through_the_services_rule(tmp_path):
    """--follow-ups hands each row's history to the CLI (fields.CandidatesAfter):
    a short reply that names nothing is read with the turn before it, any
    other row gets exactly its own block."""
    _need_go()
    rows = [r for r in X.load_set("followup") if r["history"]]
    plain = X.retrieval(rows, X.MANIFEST_FILE, F.binary())
    fu = X.retrieval(rows, X.MANIFEST_FILE, F.binary(), follow_ups=True)
    assert fu["overall"]["hit"] > plain["overall"]["hit"]
    hashi = case(request="the name of my site is HashiTong", history=[{"user": "hi", "assistant": "Hello! What should I change?"}],
                 context={"page": "home", "sections": []})
    one = X.run_cli(F.binary(), X.MANIFEST_FILE, [("a", hashi["context"], hashi["request"], hashi["history"])])
    two = X.run_cli(F.binary(), X.MANIFEST_FILE, [("a", hashi["context"], hashi["request"])])
    assert one == two  # names the Branding panel: not a follow-up
    assert X.main(["retrieval", "--set", "followup", "--follow-ups", "--out-dir", str(tmp_path)]) == 0
    saved = json.loads(next(tmp_path.glob("exam-retrieval-followup-fu-*.json")).read_text(encoding="utf-8"))
    assert saved["options"]["follow_ups"] is True


# --- end to end through the fake service -----------------------------------------------------------------------------------------

@pytest.fixture
def svc():
    with FS.FakeService(manifest=TINY, key="k") as s:
        yield s


def tiny_case(**over) -> dict:
    hero = {"kind": "section", "page": "home", "sectionId": "heroBanner-1"}
    c = case(id="m1", manifest_hash=TINY_HASH, site_state={"template": {"id": "editorial"}},
             context={"page": "home", "template": "editorial", "sections": []}, request="4, you pick",
             history=[{"user": "too many things", "assistant": "How many?"}],
             expected={"ops": [], "diff": [], "keys": [{"target": hero, "key": "title", "list": ""}], "alt": [],
                       "out_of_scope": None})
    c.update(over)
    return c


def test_history_turns_are_replayed_live_in_one_conversation(svc):
    client = C.Client(svc.url, "k", timeout_s=10)
    hero = {"kind": "section", "page": "home", "sectionId": "heroBanner-1"}
    svc.scripts["too many things"] = [[call("h1", "set_fields", {"target": hero, "values": {"show": True}}), waiting("h1")],
                                      done("How many would you like?")]
    svc.scripts["4, you pick"] = [[call("t1", "set_fields", {"target": hero, "values": {"title": "4"}}), waiting("t1")], done()]
    c = tiny_case()
    r = X.run_row(c, client, fake_engine, state=c["site_state"], route=None)
    assert [b["text"] for b in svc.turn_requests] == ["too many things", "4, you pick"]
    assert {b["siteId"] for b in svc.turn_requests} == {"lab-m1"}
    assert svc.conversation_ends == ["lab-m1"]  # ended once, before the first history turn; the scored turn continues it
    assert r["verdict"] == "pass" and r["keys_hit"] and r["history_mode"] == "live"
    h = r["history_turns"][0]
    assert h["reply"] == "How many would you like?" and h["recorded_reply"] == "How many?" and h["status"] == "ok"
    # the engine state carried over: the scored turn's tool result sees the history turn's write
    assert svc.tool_results[-1]["body"]["results"][0]["ok"] is True
    assert r["ops"] == [{"op": "set_fields", "target": hero, "values": {"title": "4"}}]  # only the scored turn is scored


def test_history_modes_and_a_failed_history_turn(svc):
    client = C.Client(svc.url, "k", timeout_s=10)
    c = tiny_case(history=[{"user": "a", "assistant": "A"}, {"user": "b", "assistant": "B"}])
    r = X.run_row(c, client, fake_engine, state={}, route=None, history_mode="last")
    assert [b["text"] for b in svc.turn_requests] == ["b", "4, you pick"] and r["verdict"] == "no_write"
    svc.turn_requests.clear()
    X.run_row(c, client, fake_engine, state={}, route=None, history_mode="none")
    assert [b["text"] for b in svc.turn_requests] == ["4, you pick"]
    svc.scripts["a"] = [[("turn.error", {"code": "model_error", "message": "boom"})]]
    bad = X.run_row(c, client, fake_engine, state={}, route=None)
    assert bad["status"] == "error" and bad["code"] == "history_failed" and C.passed(bad) is None


def test_run_set_report_by_class_and_keys_hit(svc, tmp_path, monkeypatch):
    client = C.Client(svc.url, "k", timeout_s=10)
    hero = {"kind": "section", "page": "home", "sectionId": "heroBanner-1"}
    svc.scripts["set it"] = [[call("t1", "set_fields", {"target": hero, "values": {"title": "x", "show": True}}), waiting("t1")], done()]
    monkeypatch.setattr(X, "vintage_state", lambda manifest, home: {"template": {"id": "editorial"}})
    monkeypatch.setattr(X, "manifest_hash_of", lambda p: TINY_HASH)
    cases = [tiny_case(id="a", history=[], request="set it"), tiny_case(id="b", history=[], request="nothing"),
             tiny_case(id="c", history=[], request="nope", kind="out_of_scope", meta={"class": "oos"},
                       expected={"keys": [], "alt": [], "out_of_scope": {"reason": "x", "where": []}})]
    path, rep = X.run_set(cases, client, fake_engine, label="t", set_name="mini", route=None, manifest=Path("unused"),
                          out_dir=tmp_path)
    assert path.is_file() and rep["set"] == "mini" and rep["cases"] == 3
    assert {r["case_id"]: r["verdict"] for r in rep["results"]} == {"a": "stray_writes", "b": "no_write", "c": "no_pointer"}
    assert rep["summary"]["keys_hit"] == {"n": 2, "hit": 1, "rate": 50.0}
    assert set(rep["summary"]["by_class"]) == {"synonym", "oos"}
    assert "keys hit 1/2" in X.format_run(rep)


# --- the oracle ---------------------------------------------------------------------------------------------------------------

def test_oracle_ops_write_every_expected_key():
    from souschef_lab.manifest import Manifest

    m = Manifest(json.loads(X.MANIFEST_FILE.read_text(encoding="utf-8")))
    rows = [r for name in ("paraphrase", "holdout", "real") for r in X.load_set(name)]
    for r in rows:
        ops = [{"op": c["name"], **c["input"]} for c in X.oracle_ops(r, m)]
        fake = {"status": "ok", "ops": ops, "engine_results": [{"results": [{"ok": True} for _ in ops]}], "reply": ""}
        assert X.score_turn(r, fake)["verdict"] == "pass", r["id"]
    ogimage = next(r for r in rows if any(e["key"] == "defaults.ogImage" for e in r["expected"]["keys"]))
    assert X.oracle_ops(ogimage, m)[0]["input"]["values"]["defaults.ogImage"].startswith("https://")
    oos = X.load_set("out_of_scope")[0]
    script = X.oracle_script(oos, m)  # default: report_unsupported with the row's `where`, a reply naming no place
    assert script[0][0] == ("tool.call", {"id": "u0", "name": "report_unsupported",
                                          "input": {"reason": "That isn't something I can change in the site editor.",
                                                    "where": "products"}})
    assert not X.names_a_place(script[0][1][1]["text"])
    legacy = X.oracle_script(oos, m, "legacy")  # a service before the enum: the place in words only
    assert legacy[0][0][0] == "text.delta" and "Breadstack" in legacy[0][0][1]["text"]
    with pytest.raises(X.ExamError):
        X.oracle_script(oos, m, "bogus")


def test_dry_run_on_the_fake_service_with_the_real_engine(tmp_path):
    reason = _engine_reason()
    if reason:
        if os.environ.get("SOUS_CHEF_LAB_REQUIRE_ENGINE") == "1":
            pytest.fail(f"engine required but unavailable: {reason}")
        pytest.skip(reason)
    from souschef_lab.engine import find_engine

    rc = X.main(["run", "--set", "real", "--only", "r00,r06,r15", "--fake", "--oracle", "--route", "fake",
                 "--engine", str(find_engine(os.environ.get("ENGINE"))), "--out-dir", str(tmp_path), "--label", "t"])
    assert rc == 0
    rep = json.loads(next(tmp_path.glob("exam-real-t-*.json")).read_text(encoding="utf-8"))
    assert [r["verdict"] for r in rep["results"]] == ["pass", "pass", "pass"]
    assert [len(r["history_turns"]) for r in rep["results"]] == [0, 1, 1]
    assert rep["manifest_hash"] == REAL_HASH and rep["engine_fingerprint"]


def test_dry_run_scores_out_of_scope_by_where_and_by_words(tmp_path):
    """The oracle dry run on the out-of-scope set: `where` mode passes every
    row through the `where` path, legacy mode through the keyword rule —
    and OFFSET/LIMIT + a site prefix run a slice into its own file."""
    reason = _engine_reason()
    if reason:
        if os.environ.get("SOUS_CHEF_LAB_REQUIRE_ENGINE") == "1":
            pytest.fail(f"engine required but unavailable: {reason}")
        pytest.skip(reason)
    from souschef_lab.engine import find_engine

    eng = str(find_engine(os.environ.get("ENGINE")))
    for mode in ("where", "legacy"):
        out = tmp_path / mode
        assert X.main(["run", "--set", "out_of_scope_holdout", "--fake", "--oracle", "--oracle-mode", mode, "--route", "fake",
                       "--engine", eng, "--out-dir", str(out), "--label", mode]) == 0
        rep = json.loads(next(out.glob("exam-out_of_scope_holdout-*.json")).read_text(encoding="utf-8"))
        assert [r["verdict"] for r in rep["results"]] == ["unsupported_ok"] * 4
        assert {r["scored_by"] for r in rep["results"]} == {"where" if mode == "where" else "keywords"}
    out = tmp_path / "slice"
    assert X.main(["run", "--set", "out_of_scope", "--fake", "--oracle", "--route", "fake", "--engine", eng,
                   "--out-dir", str(out), "--label", "s", "--offset", "10", "--limit", "5", "--site-prefix", "lab-cand-"]) == 0
    path = next(out.glob("exam-out_of_scope-s-r10-13-*.json"))
    rep = json.loads(path.read_text(encoding="utf-8"))
    assert [r["case_id"] for r in rep["results"]] == ["p145", "p146", "p147"]
    assert rep["rows_range"] == {"offset": 10, "end": 13, "of": 13} and rep["site_prefix"] == "lab-cand-"
    assert X.main(["run", "--set", "out_of_scope", "--fake", "--oracle", "--route", "fake", "--engine", eng,
                   "--out-dir", str(out), "--offset", "13"]) == 2
    assert X.main(["run", "--set", "out_of_scope", "--fake", "--route", "fake", "--engine", eng, "--out-dir", str(out),
                   "--site-prefix", "prod-"]) == 2


def test_run_case_site_prefix(svc):
    client = C.Client(svc.url, "k", timeout_s=10)
    c = tiny_case(history=[])
    X.run_row(c, client, fake_engine, state={}, route=None, site_prefix="lab-cand-")
    assert {b["siteId"] for b in svc.turn_requests} == {"lab-cand-m1"} and svc.conversation_ends == ["lab-cand-m1"]
    with pytest.raises(ValueError):
        C.run_case(c, client, fake_engine, site_prefix="site-")


def test_run_refuses_oracle_without_fake():
    assert X.main(["run", "--set", "real", "--oracle"]) == 2


# --- the conversation replay harness --------------------------------------------------------------------------------------------

CTX_HOME = {"page": "home", "template": "vintage", "layout": "hero-featured", "sections": []}
CTX_FAQ = {"page": "faq", "template": "vintage", "layout": "accordion", "sections": []}
CF = "<candidate_fields>\nFields…\ntarget {\"kind\":\"branding\"}\n  logo — Logo (image)\n</candidate_fields>"


def utext(ctx: dict, text: str, cf: str = "") -> list[dict]:
    return [{"type": "text", "text": "<context>\n" + json.dumps(ctx, separators=(",", ":")) + "\n</context>\n"
             + (cf + "\n" if cf else "") + text}]


CONV = {"conversationId": "abcdef12-0000", "storeId": "s1",
        "messages": [
            {"seq": 1, "role": "user", "content": utext(CTX_HOME, "change the site name to X", CF)},
            {"seq": 2, "role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "set_fields",
                                                         "input": {"target": {"kind": "branding"}, "values": {"location": "X"}}}]},
            {"seq": 3, "role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "is_error": True,
                                                    "content": [{"type": "text", "text": "{\"code\":\"E_FIELD\"}"}]}]},
            {"seq": 4, "role": "assistant", "content": [{"type": "thinking", "thinking": "hm"},
                                                        {"type": "text", "text": "I changed the store name."}]},
            {"seq": 5, "role": "user", "content": utext(CTX_FAQ, "yes please")},
            {"seq": 6, "role": "assistant", "content": [{"type": "text", "text": "Which one?"}]}],
        "turns": [{"id": "t-1", "model": "blueberry-muffin", "status": "done", "manifestHash": REAL_HASH,
                   "userEmail": "someone@example.com", "usage": {"routes": ["blueberry-muffin", "default"]}},
                  {"id": "t-2", "model": "blueberry-muffin", "status": "done", "manifestHash": REAL_HASH,
                   "userEmail": "someone@example.com"}]}


def test_conversation_turns_split_context_block_and_text():
    turns = X.conversation_turns(CONV)
    assert [t["text"] for t in turns] == ["change the site name to X", "yes please"]
    assert turns[0]["context"] == CTX_HOME and turns[0]["candidates"] == CF and turns[1]["candidates"] == ""
    assert turns[0]["reply"] == "I changed the store name." and turns[0]["calls"][0]["result"]["is_error"] is True
    assert [m["seq"] for m in turns[1]["messages_before"]] == [1, 2, 3, 4]
    assert turns[0]["turn"]["id"] == "t-1" and "userEmail" not in turns[0]["turn"]


def test_replay_rows_rebuild_each_turn_with_its_history():
    label = case(id="L1", context={"page": "faq"}, request="Yes  Please",
                 expected={"ops": [], "diff": [], "keys": [STORE], "alt": [], "out_of_scope": None})
    rows = X.replay_rows(CONV, [label])
    assert [r["id"] for r in rows] == ["replay-abcdef12-00", "replay-abcdef12-01"]
    first, second = rows
    assert first["history"] == [] and first["kind"] == "replay" and not X.is_labelled(first)
    assert second["history"] == [{"user": "change the site name to X", "assistant": "I changed the store name.",
                                  "context": CTX_HOME}]
    assert second["context"] == CTX_FAQ and second["meta"]["page"] == "faq" and second["manifest_hash"] == REAL_HASH
    assert second["meta"]["labelled_by"] == "L1" and second["expected"]["keys"] == [STORE] and X.is_labelled(second)
    assert first["meta"]["recorded"]["candidate_fields"] == CF and first["meta"]["recorded"]["calls"][0]["name"] == "set_fields"
    assert [m["seq"] for m in second["meta"]["transcript"]] == [1, 2, 3, 4]
    assert "someone@example.com" not in json.dumps(rows)
    assert "transcript" not in X.replay_rows(CONV, transcript=False)[1]["meta"]


def test_replay_cli_writes_rows(tmp_path):
    conv = tmp_path / "conv.json"
    conv.write_text(json.dumps(CONV), encoding="utf-8")
    out = tmp_path / "rows.jsonl"
    assert X.main(["replay", "--conversation", str(conv), "--labels", "real", "--out", str(out)]) == 0
    rows = X.read_jsonl(out)
    assert len(rows) == 2 and rows[0]["request"] == "change the site name to X"
    assert X.load_set(out)[1]["history"][0]["context"] == CTX_HOME  # a replay file loads as a set (inline contexts)
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"error": "conversation_not_found"}), encoding="utf-8")
    assert X.main(["replay", "--conversation", str(missing)]) == 1


def test_served_block_parity_counts_equal_blocks(monkeypatch):
    rows = X.replay_rows(CONV)
    monkeypatch.setattr(X, "run_cli", lambda exe, manifest, turns: {cid: {"block": CF, "targets": []} for cid, _, _ in turns})
    p = X.served_block_parity(rows, X.MANIFEST_FILE, Path("unused"))
    assert p == {"checked": 1, "same": 1, "differ": [], "other_manifest": 0}  # the block-less turn is not compared


def test_build_refuses_a_study_folder_without_its_files(tmp_path):
    with pytest.raises(X.ExamError, match="lacks"):
        X.build_sets(tmp_path, tmp_path / "out")


def test_build_round_trips_a_small_study(tmp_path):
    why = tmp_path / "why"
    why.mkdir()
    home = json.loads((X.EXAM_DIR / "contexts.json").read_text(encoding="utf-8"))["vintage-home"]
    (why / "ctx-vintage-home.json").write_text(json.dumps(home), encoding="utf-8")
    (why / "manifest-real.json").write_bytes(X.MANIFEST_FILE.read_bytes())
    p = {"id": "p1", "class": "synonym", "ctx": "vintage-home", "request": "change the site name to X",
         "expect": [STORE], "alt": [[SEO]]}
    f = {"id": "p2", "class": "followup", "ctx": "cart", "request": "yes", "expect": [STORE],
         "prev": {"user": "u", "assistant": "a"}}
    o = {"id": "p3", "class": "oos", "ctx": "vintage-home", "request": "sell socks", "expect": [], "oos": "products are managed in Breadstack"}
    (why / "paraphrase-cases.jsonl").write_text("\n".join(json.dumps(x) for x in (p, f, o)) + "\n", encoding="utf-8")
    (why / "paraphrase-holdout.jsonl").write_text(json.dumps(dict(p, id="h1")) + "\n", encoding="utf-8")
    conv = copy.deepcopy(CONV)
    conv["messages"][0]["content"] = utext(home, "change the site name to X", CF)
    (why / "conv-1.json").write_text(json.dumps(conv), encoding="utf-8")
    (why / "real-cases.jsonl").write_text(json.dumps(dict(p, id="r1", **{"class": "real"})) + "\n", encoding="utf-8")
    out = tmp_path / "exam"
    idx = X.build_sets(why, out)
    assert {k: v["rows"] for k, v in idx["sets"].items()} == {"paraphrase": 2, "holdout": 1, "real": 1, "out_of_scope": 1,
                                                              "out_of_scope_holdout": 0, "followup": 1}
    para = X.load_set("paraphrase", exam_dir=out)
    assert para[1]["context"]["page"] == "cart" and para[1]["meta"]["context_source"] == "approximated"
    assert para[1]["history"] == [{"user": "u", "assistant": "a"}]
    oos = X.load_set("out_of_scope", exam_dir=out)[0]
    assert oos["expected"]["out_of_scope"]["where"] and oos["kind"] == "out_of_scope"
    assert X.load_set("real", exam_dir=out)[0]["meta"]["context_source"] == "recorded"
    bad = dict(p, id="p9", expect=[dict(STORE, key="noSuchKey")])
    (why / "paraphrase-cases.jsonl").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(X.ExamError, match="noSuchKey"):
        X.build_sets(why, tmp_path / "exam2")


# --- answer-key amendments (2026-10-15: "the font" = every site font; vague → ask) ------------------------------------

def ok(ops: list[dict]) -> dict:
    return row(*[(o, True) for o in ops])


def test_the_font_rows_expect_every_site_font_under_new_ids():
    amended = {(r["set"], r["meta"]["superseded"]["id"]): r for name in SIZES for r in X.load_set(name)
               if (r.get("meta") or {}).get("answer_key")}
    assert set(amended) == set(X.AMENDMENTS)
    for (s, old_id), r in amended.items():
        want_key = X.PAGE_CREATE_KEY if (s, old_id) == ("out_of_scope", "p137") else X.ANSWER_KEY
        assert r["id"] == old_id + X.AMENDED_ID_SUFFIX and r["meta"]["answer_key"] == want_key
    raw = {r["id"] for r in X.load_set("followup", amended=False)}
    assert "p150" in raw and "p150.k2" not in raw
    p150 = amended[("followup", "p150")]
    assert [k["key"] for k in p150["expected"]["keys"]] == list(X.SITE_FONTS_VINTAGE)
    g = {"kind": "global", "panel": "TypographyColorsSection"}
    every = [sf({"kind": "global"}, **{k: "Lora" for k in X.SITE_FONTS_VINTAGE})]
    assert X.score_turn(p150, ok(every))["verdict"] == "pass"
    # the inheriting cart / footer fonts may be written too; one or two fonts are partial now; a hidden one is stray
    assert X.score_turn(p150, ok([sf({"kind": "global"}, **{k: "Lora" for k in X.SITE_FONTS_VINTAGE},
                                      cartBodyFont="Lora")]))["verdict"] == "pass"
    assert X.score_turn(p150, ok([sf(g, bodyFont="Lora")]))["verdict"] == "partial"
    assert X.score_turn(p150, ok([sf(g, bodyFont="Lora", headingFont="Lora", navFont="Lora")]))["verdict"] == "partial"
    assert X.score_turn(p150, ok([sf({"kind": "global"}, **{k: "Lora" for k in X.SITE_FONTS_VINTAGE},
                                      shopMenuFont="Lora")]))["verdict"] == "stray_writes"
    # h059: the history asked "headings and body?", the merchant said both: those two, or every font
    h059 = amended[("holdout", "h059")]
    assert X.score_turn(h059, ok([sf(g, bodyFont="Inter", headingFont="Inter")]))["verdict"] == "pass"
    assert X.score_turn(h059, ok(every))["verdict"] == "pass"
    # a named part is unchanged
    p012 = next(r for r in X.load_set("paraphrase") if r["id"] == "p012")
    assert [k["key"] for k in p012["expected"]["keys"]] == ["headingFont"]


def test_i_dont_like_the_font_may_be_answered_with_a_question():
    p013 = next(r for r in X.load_set("paraphrase") if r["id"] == "p013.k2")
    assert p013["expected"]["ask_ok"] is True
    asked = {**ok([]), "reply": "Which font would you like instead? Lora, Inter or Montserrat?"}
    assert X.score_turn(p013, asked)["verdict"] == "clarify_ok"
    assert X.score_turn(p013, {**ok([]), "reply": "Okay."})["verdict"] == "no_write"
    # p150's first turn is the same words; only p013 is single-turn — the follow-up must write every font
    p150 = next(r for r in X.load_set("paraphrase") if r["id"] == "p150.k2")
    assert not p150["expected"].get("ask_ok") and X.score_turn(p150, asked)["verdict"] == "no_write"


def test_a_value_less_request_may_be_answered_with_a_question():
    p031 = next(r for r in X.load_set("paraphrase") if r["id"] == "p031.k2")
    assert p031["expected"]["ask_ok"] is True and p031["fields"] == ["footerCopyrightText"]
    assert [k["key"] for k in p031["expected"]["keys"]] == ["footerCopyrightText"]  # the key is unchanged
    asked = {**row(), "reply": "What should the copyright line say?"}
    assert X.score_turn(p031, asked)["verdict"] == "clarify_ok"
    wrote = ok([sf({"kind": "global", "panel": "FooterSection"}, footerCopyrightText="© 2026")])
    assert X.score_turn(p031, wrote)["verdict"] == "pass"
    reported = {**asked, "tool_calls": [{"name": "report_unsupported", "input": {"where": "support"}}]}
    assert X.score_turn(p031, reported)["verdict"] == "no_write"
    # a request that gives its value is not amended
    assert not any(r["id"].startswith("p002") and r["expected"].get("ask_ok") for r in X.load_set("paraphrase"))


# --- create_page (2026-10-15): p137 re-keyed, the pages where-rule split, score_page ---------------------------------------

def _page_row(ops_ok, reply="", ctx=None, slugs=None, calls=None):
    """A row whose ops (op, ok) ran in one batch; create_page results carry their slug."""
    ops = [o for o, _ in ops_ok]
    res = []
    for (o, ok), sl in zip(ops_ok, slugs or [None] * len(ops_ok)):
        r = {"ok": ok}
        if ok and o.get("op") == "create_page":
            r["slug"] = sl or X.slugify(o.get("slug") or o.get("title"))
        res.append(r)
    return {"status": "ok", "stop": "end_turn", "ops": ops, "engine_results": [{"results": res}], "reply": reply,
            "tool_calls": calls if calls is not None else [{"name": o["op"], "input": {}} for o in ops],
            "context": ctx}


def test_the_pages_where_rule_is_split_create_from_rename_duplicate_delete():
    for text in ("rename the events page", "duplicate the about page so i can make one for our second shop",
                 "delete the Careers page, we're not hiring", "change the url of the about page to /our-story",
                 "clone the events page for our summer markets", "copy our about page and call the copy Our Team",
                 "get rid of the events page, we don't do events anymore", "unpublish the loyalty page until it's ready"):
        assert X.page_request(text) == "manage", text
        assert X.expected_where_for("customPages is read-only", text) == ["pages"], text
    for text in ("create a new page for events", "add a new page called Strain Guide", "new page: Accessibility Statement",
                 "i need another page for wholesale inquiries", "make an 'Ask a Budtender' page"):
        assert X.page_request(text) == "create", text
        assert X.expected_where_for("customPages is read-only to the ops engine", text) == ["support"], text
    assert X.page_request("make the hero bigger") is None
    assert X.slugify("  Summer  Sale! ") == "summer-sale" and X.slugify("Wholesale_Inquiries") == "wholesaleinquiries"
    assert X.slugify("4/20 Deals") == "420-deals" and X.is_reserved_page_slug("menu") and X.is_reserved_page_slug("sitemap-x")
    assert X.page_title_of("add a new page called Strain Guide") == {"title": "Strain Guide", "match": "exact"}
    assert X.page_title_of("new page: Accessibility Statement")["title"] == "Accessibility Statement"
    assert X.page_title_of("build me a 'Medical Patients' info page") == {"title": "Medical Patients", "match": "contains"}
    assert X.page_title_of("i need another page for wholesale inquiries") == {"words": ["wholesale", "inquiries"],
                                                                             "match": "words"}
    assert X.page_title_of("add a blank page i can fill in later") is None


def test_p137_expects_create_page_or_the_existing_events_page():
    p137 = next(r for r in X.load_set("out_of_scope") if r["id"] == "p137.k2")
    assert p137["kind"] == "create_page" and p137["meta"]["class"] == "create" and X.is_labelled(p137)
    assert not X.is_out_of_scope(p137) and p137["meta"]["superseded"]["out_of_scope"]["expected_where"] == ["pages"]
    assert {r["id"] for r in X.load_set("out_of_scope", amended=False)} >= {"p137"}
    # a site without an Events page: create_page {title Events}; the r5 answer (where=pages) fails
    bare = {**p137, "context": {**p137["context"], "customPages": []}, "site_state": {"customPages": []}}
    good = _page_row([({"op": "create_page", "title": "events"}, True)], "Added /events.")
    assert X.score_turn(bare, good)["verdict"] == "pass"
    r5 = _page_row([], calls=[{"name": "report_unsupported", "input": {"where": "pages"}}])
    assert X.score_turn(bare, r5)["verdict"] == "reported"
    assert X.score_turn(bare, _page_row([({"op": "create_page", "title": "Events", "in_nav": True}, True)]))["verdict"] \
        == "wrong_value"
    # the exam's vintage state HAS a custom page Events (/events): never create it again — say it exists
    have = {**p137, "site_state": {"customPages": [{"slug": "events", "title": "Events"}]}}
    said = _page_row([], "You already have an Events page at /events. Want me to add a section to it?")
    assert X.score_turn(have, said)["verdict"] == "pass"
    tried = _page_row([({"op": "create_page", "title": "Events"}, False)], "There's already an Events page (/events).")
    assert X.score_turn(have, tried)["verdict"] == "pass"  # refused by the engine, explained: right on the site
    assert X.score_turn(have, _page_row([], "Done!"))["verdict"] == "not_explained"
    assert X.score_turn(have, r5)["verdict"] == "reported"
    # the oracle answers each the right way
    assert X.page_oracle(bare)[0] == [{"op": "create_page", "title": "Events"}]
    assert X.page_oracle({**p137, "context": {**p137["context"], "customPages": [{"slug": "events", "title": "Events"}]}})[0] == []


def test_score_page_create_slug_status_nav_sections_and_strays():
    case = {"expected": {"page": {"expect": "create", "title": "Budtender Picks", "in_nav": False, "status": "published",
                                  "sections": ["productCarousel"], "if_exists": "decline"}},
            "context": {"page": "home", "customPages": []}, "site_state": {"customPages": []}}
    create = {"op": "create_page", "title": "budtender  picks", "slug": "budtender-picks"}
    add = {"op": "edit_list", "target": {"kind": "page", "page": "budtender-picks"}, "list": "sections", "action": "add",
           "item": {"id": "productCarousel"}}
    assert X.score_page(case, _page_row([(create, True), (add, True)]))["verdict"] == "pass"  # the add is the page's own
    assert X.score_page(case, _page_row([(create, True)]))["verdict"] == "missing_writes"
    stray = {"op": "set_fields", "target": {"kind": "global", "panel": "HeaderSection"}, "values": {"x": 1}}
    assert X.score_page(case, _page_row([(create, True), (add, True), (stray, True)]))["verdict"] == "extra_writes"
    other = dict(add, target={"kind": "page", "page": "home"})
    assert X.score_page(case, _page_row([(create, True), (other, True)]))["verdict"] == "missing_writes"
    assert X.score_page(case, _page_row([({**create, "slug": "picks"}, True), (dict(add, target={"kind": "page", "page": "picks"}), True)],
                                        slugs=["picks", None]))["verdict"] == "wrong_value"  # not the title's slug
    assert X.score_page(case, _page_row([(create, False), (create, True), (add, True)]))["verdict"] == "invalid_op"
    assert X.score_page(case, _page_row([(create, False), (create, True), (add, True)]), refused_ok=True)["verdict"] == "pass"
    draft = {"expected": {"page": {"expect": "create", "title": "Price Match", "status": "draft", "in_nav": False}}}
    assert X.score_page(draft, _page_row([({"op": "create_page", "title": "Price Match", "status": "draft"}, True)]))["verdict"] == "pass"
    assert X.score_page(draft, _page_row([({"op": "create_page", "title": "Price Match"}, True)]))["reasons"] == ["wrong_value:status:published"]
    slugged = {"expected": {"page": {"expect": "create", "title": "Summer Sale", "slugs": ["summer-sale"], "in_nav": False}}}
    assert X.score_page(slugged, _page_row([({"op": "create_page", "title": "Summer Sale", "slug": "summer-sale"}, True)]))["verdict"] == "pass"
    words = {"expected": {"page": {"expect": "create", "title_match": "words", "words": ["420", "deals"], "in_nav": False}}}
    assert X.score_page(words, _page_row([({"op": "create_page", "title": "4/20 Deals"}, True)]))["verdict"] == "pass"
    assert X.score_page(words, _page_row([({"op": "create_page", "title": "Promotions"}, True)]))["verdict"] == "wrong_value"


def test_score_page_declines():
    reserved = {"expected": {"page": {"expect": "decline", "why": "reserved", "title": "Menu"}}}
    refused = _page_row([({"op": "create_page", "title": "Menu"}, False)], "/menu is taken. Should I use /menu-2 instead?")
    assert X.score_page(reserved, refused)["verdict"] == "pass"
    assert X.score_page(reserved, _page_row([({"op": "create_page", "title": "Menu"}, False)], "It's taken."))["verdict"] == "no_question"
    wrote = _page_row([({"op": "create_page", "title": "Menu", "slug": "menu-2"}, True)], "Done.")
    assert X.score_page(reserved, wrote)["verdict"] == "wrote"
    tpl = {"expected": {"page": {"expect": "decline", "why": "template", "title": "Gift Cards"}}}
    assert X.score_page(tpl, _page_row([({"op": "create_page", "title": "Gift Cards"}, False)],
                                       "This template can't show custom pages yet."))["verdict"] == "pass"
    # a create + section add on a page the site already has: the follow-up must not land on the existing page
    exists = {"expected": {"page": {"expect": "create", "title": "Contact", "sections": ["newsletter"], "if_exists": "decline"}},
              "context": {"customPages": []}}
    add = {"op": "edit_list", "target": {"kind": "page", "page": "contact"}, "list": "sections", "action": "add",
           "item": {"id": "newsletter"}}
    blocked = _page_row([({"op": "create_page", "title": "Contact", "slug": "contact"}, False), (add, False)],
                        "You already have a Contact page (/contact).")
    assert X.score_page(exists, blocked)["verdict"] == "pass" and X.page_expectation(exists)["existing"]["kind"] == "builtin"
    leaked = _page_row([({"op": "create_page", "title": "Contact", "slug": "contact"}, False), (add, True)], "Added it.")
    assert X.score_page(exists, leaked)["verdict"] == "wrote"



def test_score_page_exists_must_say_it_exists_or_edit_it_instead():
    """r6-4 review: a decline "exists" must say the page is already there (naming it is not enough), and the rules'
    other answer — add the asked section to the existing page — passes too (not as a same-batch create follow-up)."""
    p137 = next(r for r in X.load_set("out_of_scope") if r["id"] == "p137.k2")
    have = {**p137, "site_state": {"customPages": [{"slug": "events", "title": "Events"}]}}
    for lax in ("Sorry, I can't make an events page.", "What would you like on the events page?"):
        assert X.score_turn(have, _page_row([], lax))["verdict"] == "not_explained", lax
    assert X.score_turn(have, _page_row([], "You have an Events page already (/events)."))["verdict"] == "pass"
    assert X.score_turn(have, _page_row([], "You've got an Events page at /events; edit it instead?"))["verdict"] == "pass"
    deals = {"expected": {"page": {"expect": "create", "title": "Deals", "sections": ["productCarousel"], "if_exists": "decline"}},
             "context": {"customPages": [{"slug": "deals", "title": "Deals"}]}, "site_state": {"customPages": []}}
    add = {"op": "edit_list", "target": {"kind": "page", "page": "deals"}, "list": "sections", "action": "add",
           "item": {"id": "productCarousel"}}
    edited = _page_row([(add, True)], "You already have a Deals page (/deals), so I added a product carousel to it.")
    v = X.score_page(deals, edited)
    assert v["verdict"] == "pass" and v["info"] == ["decline:exists:edited"] and v["detail"]["edited_existing"] == 1
    assert X.score_page(deals, _page_row([(add, True)], "Added a product carousel."))["verdict"] == "not_explained"
    other = dict(add, item={"id": "gallery"})  # not the asked section
    assert X.score_page(deals, _page_row([(other, True)], "You already have a Deals page."))["verdict"] == "wrote"
    home = dict(add, target={"kind": "page", "page": "home"})
    assert X.score_page(deals, _page_row([(home, True)], "You already have a Deals page."))["verdict"] == "wrote"
    # after a refused create in a LATER batch: a deliberate edit of the existing page
    later = {"status": "ok", "stop": "end_turn", "ops": [{"op": "create_page", "title": "Deals"}, add],
             "engine_results": [{"results": [{"ok": False, "error": {"code": "E_SLUG"}}]}, {"results": [{"ok": True}]}],
             "reply": "Deals already exists at /deals, so I put the carousel there.", "tool_calls": [], "context": None}
    assert X.score_page(deals, later)["verdict"] == "pass"


def test_score_page_words_create_may_ask_for_the_name():
    """A request that names no page title (title_match "words"): asking for the name is right (rules: "if they gave
    none, ask"); an exact-title request must be created."""
    words = {"expected": {"page": {"expect": "create", "title_match": "words", "words": ["dosing", "guide"], "in_nav": False}}}
    asked = _page_row([], "Happy to! What would you like the page to be called?")
    assert X.score_page(words, asked) == {"verdict": "pass", "reasons": [], "info": ["asked_for_the_name"],
                                          "detail": {"expect": "create", "create_refused": 0}}
    assert X.score_page(words, _page_row([], "Done — anything else?"))["verdict"] == "no_answer"  # a sign-off is no question
    exact = {"expected": {"page": {"expect": "create", "title": "Dosing Guide", "in_nav": False}}}
    assert X.score_page(exact, asked)["verdict"] == "no_answer"
