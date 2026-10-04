"""cases.py — schema, determinism, holdout, per-kind invariants, io, measure."""

import copy
import json
import os
import re

import pytest

from souschef_lab import FIXTURES_DIR
from souschef_lab import cases as C
from souschef_lab import engine as E
from souschef_lab import manifest as M
from souschef_lab import state as S

REQUIRE = os.environ.get("SOUS_CHEF_LAB_REQUIRE_ENGINE") == "1"


@pytest.fixture(scope="module")
def m():
    return M.load(FIXTURES_DIR / "manifest.snapshot.json")


@pytest.fixture(scope="module")
def gen(m):
    return C.Generator(m, seed=0)


@pytest.fixture(scope="module")
def cases(gen):
    return list(gen.all_cases())


@pytest.fixture(scope="module")
def by_kind(cases):
    out = {}
    for c in cases:
        out.setdefault(c.kind, []).append(c)
    return out


def test_every_kind_is_generated(by_kind):
    assert set(by_kind) == set(C.KINDS)


def test_row_schema_and_roundtrip(cases, tmp_path):
    c = cases[0]
    row = c.to_row()
    assert set(row) == {"id", "kind", "target", "fields", "site_state", "context", "request", "expected", "holdout", "manifest_hash", "template", "meta", "engine_fingerprint"}
    assert set(row["expected"]) >= {"ops", "diff"}
    assert row["template"] == "editorial" and row["manifest_hash"] == c.manifest_hash == M.load(FIXTURES_DIR / "manifest.snapshot.json").hash
    p = tmp_path / "c.jsonl"
    assert C.write_jsonl(p, cases[:5]) == 5
    back = C.read_cases(p)
    assert [b.to_row() for b in back] == [x.to_row() for x in cases[:5]]


def test_ids_unique_and_free_of_hash(cases):
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))
    assert all("#" not in i and " " not in i for i in ids)
    assert all(i.split("/")[0] in C.KINDS or i.startswith("reset/") for i in ids)


def test_deterministic_for_a_seed(m, cases):
    again = list(C.Generator(m, seed=0).all_cases())
    assert [(c.id, c.request, c.expected["ops"], c.holdout) for c in again] == [(c.id, c.request, c.expected["ops"], c.holdout) for c in cases]
    other = list(C.Generator(m, seed=1).all_cases(kinds=["set_field"]))
    mine = {c.id: c for c in cases if c.kind == "set_field"}
    same_ids = {c.id for c in other} & set(mine)
    assert len(same_ids) > 0.9 * len(mine)
    changed = sum(1 for c in other if c.id in mine and c.request != mine[c.id].request)
    assert changed > 0  # wording follows the seed
    assert all(c.holdout == mine[c.id].holdout for c in other if c.id in mine)  # holdout does not


def test_holdout_band_and_independence_from_wording(cases):
    pct = 100 * sum(c.holdout for c in cases) / len(cases)
    assert 17 <= pct <= 23, pct
    # the same holdout key → the same bucket, whatever the section row / wording
    by_key = {}
    for c in cases:
        by_key.setdefault(c.meta["holdout_key"], set()).add(c.holdout)
    assert all(len(v) == 1 for k, v in by_key.items() if not k.startswith("multi"))


def test_no_unfilled_slots_or_doubled_nouns(cases):
    bad = [c.id for c in cases if "{" in c.request or "}" in c.request or " section section" in c.request]
    assert bad == []
    assert all(c.request.strip() for c in cases)


def test_no_derived_or_json_targets_in_positive_cases(cases, m):
    for c in cases:
        if c.kind == "unsupported":
            continue
        for op in c.expected["ops"]:
            for key in (op.get("values") or {}):
                f = m.resolve_field(_container(m, op), key) or S.resolve_key(m, _container(m, op), key)
                assert f is not None and not f.read_only and f.type != "json", (c.id, key)


def _container(m, op):
    t = op["target"]
    if t["kind"] == "section":
        return m.section(t["sectionId"])
    if t["kind"] == "global":
        return m.panel(t.get("panel"))
    if t["kind"] == "branding":
        return m.branding
    return m.page(t["page"])


def test_toggle_flips_the_seeded_current(by_kind):
    for c in by_kind["toggle"]:
        op = c.expected["ops"][0]
        (key, value), = (op.get("values") or op["item"]).items()
        assert isinstance(value, bool)
        assert c.meta["current"] in (True, False, None)
        assert value != bool(c.meta["current"])


def test_set_enum_values_are_members_not_current(by_kind, m):
    for c in by_kind["set_enum"]:
        op = c.expected["ops"][0]
        (key, value), = (op.get("values") or op["item"]).items()
        f = S.resolve_key(m, _container(m, op), key)
        assert value in f.enum and value != c.meta["current"]
        if f.enum_labels:
            assert str(f.enum_labels.get(str(value), value)) in c.request or str(value) in c.request


def test_set_range_values_within_bounds_on_step(by_kind, m):
    for c in by_kind["set_range"]:
        op = c.expected["ops"][0]
        (key, value), = (op.get("values") or op["item"]).items()
        f = S.resolve_key(m, _container(m, op), key)
        assert isinstance(value, (int, float)) and value != c.meta["current"]
        if f.min is not None and f.max is not None:
            assert f.min <= value <= f.max
            step = f.step or 1
            assert abs(((value - f.min) / step) - round((value - f.min) / step)) < 1e-9
        else:
            assert 1 <= value <= 100


def test_image_cases_have_assets_and_pick_variant(by_kind):
    url = [c for c in by_kind["image"] if c.meta["variant"] == "url"]
    pick = [c for c in by_kind["image"] if c.meta["variant"] == "pick_asset"]
    assert len(url) == len(pick) > 0
    assets = {a["public_url"]: a for a in S.load_assets_fixture()}
    for c in url + pick:
        # Every image write — the hero's mirrored backgroundImage included — is `set_fields` + the fixture asset's
        # alt/srcset: what the model writes (warm-up #2, 2026-10-04), and what the engine's companions need.
        op = c.expected["ops"][-1]
        assert op["op"] == "set_fields", c.id
        (key, val), = op["values"].items()
        assert val.startswith("https://cdn.int.breadstack.io/gal/")
        a = assets[val]
        assert op["assets"] == {key: {"alt": a.get("alt", ""), "srcset": a.get("srcset")}}
        assert c.meta["asset_id"] == a["id"]
    for c in pick:
        assert c.expected["ops"][0]["op"] == "pick_asset" and c.expected["ops"][0]["asset_type"] == "image"
        assert c.browser_ops == c.expected["ops"][1:]
        # exactly the asset the collector's fake picker answers this case with
        picked = S.fixture_asset_for(S.load_assets_fixture(), c.id, "image")
        assert c.expected["ops"][1]["values"] == {c.fields[0]: picked["public_url"]}
    hero_images = [c for c in url if c.meta.get("mirror")]
    assert hero_images, "the hero's backgroundImage is an image case"


def test_the_picker_choice_is_a_function_of_the_case_id_alone():
    assets = S.load_assets_fixture()
    a = S.fixture_asset_for(assets, "image/x/y/z/pick")
    assert a == S.fixture_asset_for(assets, "image/x/y/z/pick") and a in assets
    assert {S.fixture_asset_for(assets, f"c{i}")["id"] for i in range(40)} == {x["id"] for x in assets}  # all reachable
    video = [{"public_url": "v.mp4", "asset_type": "video"}] + assets
    assert S.fixture_asset_for(video, "c1", "video")["public_url"] == "v.mp4"
    assert S.fixture_asset_for(video, "c1", "image")["public_url"] != "v.mp4"
    with pytest.raises(ValueError):
        S.fixture_asset_for([], "c1")


def test_hero_mirror_keys_use_edit_list_update_slide_0(cases):
    mirror = [c for c in cases if c.meta.get("mirror") and c.kind != "image"]
    assert mirror
    for c in mirror:
        op = c.expected["ops"][-1]
        assert op["op"] == "edit_list" and op["list"] == "slides" and op["action"] == "update" and op["index"] == 0
        assert op["target"]["sectionId"] == "heroBanner"
        assert c.site_state["homeSections"][0]["settings"]["slides"]  # slide 0 exists


def test_gates_are_open_in_site_state(cases, m):
    gated = [c for c in cases if c.meta.get("gated") and not c.meta.get("gate_closed") and c.kind in C.SET_KINDS + ("reset",)]
    assert gated
    for c in gated:
        assert c.meta["gate_writes"], c.id
        op = c.expected["ops"][-1]
        cont = _container(m, op)
        sid = op["target"].get("sectionId")
        for gk, gv in c.meta["gate_writes"].items():
            if "[]" in gk:
                list_key, _, item_key = gk.rpartition("[].")
                idx = S.find_section_index(c.site_state, sid)
                assert S.get_path(c.site_state, f"homeSections[{idx}].{cont.lists[list_key].path}[0].{item_key}") == gv
            else:
                f = S.resolve_key(m, cont, gk)
                assert S.read_field(cont, f, c.site_state, section_id=sid, page=op["target"].get("page", "home"), key=gk if f.keyed_by else None) == gv
    closed = [c for c in cases if c.meta.get("gate_closed")]
    assert closed and all(c.meta["expect_warning"] == "W_HIDDEN" and not c.meta["gate_writes"] for c in closed)


def test_keyed_fields_use_fixture_ids(cases):
    keyed = [c for c in cases if c.meta.get("keyed_by")]
    assert keyed
    ids = {l["id"] for l in S.load_context_fixture()["locations"]}
    for c in keyed:
        key = c.fields[0]
        assert "{" not in key and key.split(".")[1] in ids
        assert C.keyed_supported(M.load(FIXTURES_DIR / "manifest.snapshot.json"), _container(M.load(FIXTURES_DIR / "manifest.snapshot.json"), c.expected["ops"][0]), key)
    assert {c.fields[0].split(".")[-1] for c in keyed} == {"address"}  # the only keyed field the engine resolves (measured)


def test_reset_cases_seed_a_value_that_differs_from_the_expected_leaf(by_kind, m):
    modes = {c.meta["mode"] for c in by_kind["reset"]}
    assert modes == {"unset", "default"}
    for c in by_kind["reset"]:
        op = c.expected["ops"][0]
        assert op["op"] == ("reset" if c.meta["mode"] == "unset" else "restore_default")
        f = S.resolve_key(m, _container(m, op), op["keys"][0])
        assert (f.has_unset if c.meta["mode"] == "unset" else f.has_default)
        assert c.meta["seeded"] != c.meta["expected_leaf"]
        current = S.read_field(_container(m, op), f, c.site_state, section_id=op["target"].get("sectionId"), page=op["target"].get("page", "home"))
        assert current == c.meta["seeded"]


def test_list_cases_cover_every_declared_op(by_kind, m):
    seen = {}
    for kind in ("list_add", "list_remove", "list_reorder", "list_update"):
        for c in by_kind[kind]:
            op = c.expected["ops"][0]
            assert op["op"] == "edit_list"
            seen.setdefault(op["list"], set()).add(op["action"])
            if op["list"] == "sections":
                continue
            l = _container(m, op).lists[op["list"]]
            assert l.allows(op["action"]), c.id
            if l.depth == 2:
                assert isinstance(op.get("index"), list), c.id
            if op["action"] == "reorder":
                assert sorted(op["order"]) != [] and len(set(map(str, op["order"]))) == len(op["order"])
    assert seen["sections"] == {"add", "remove", "reorder", "duplicate", "update"}
    assert "add" in seen["slides"] and "update" in seen["slides[].bullets"]
    assert seen["pageVisibility.products.categoryOrder"] == {"reorder"}  # orders lists: no update (id arrays)
    on_home = {S.type_of_section(r["id"]) for r in S.DEFAULT_HOME_SECTIONS}
    for c, l in m.iter_lists():
        if c.kind == "details" or (c.kind == "section" and c.name not in on_home):
            continue  # details lists are reached through their owner; sections not on the home rows have no state
        if l.depth <= 2 and l.ops and l.item_fields:  # companion parallels (empty `item`) have nothing to write
            assert l.key in seen, f"no case for list {l.key}"


def test_list_add_carries_an_item_with_leaves(by_kind):
    for c in by_kind["list_add"]:
        op = c.expected["ops"][0]
        if op["list"] == "sections":
            assert op["action"] == "duplicate" or op["item"]["id"] in C.SECTION_LABELS
            continue
        assert "item" in op, c.id  # an appended `{}` has no leaves → empty diff (measured 2026-10-01)
        assert op["item"] == c.meta["item"]


def test_list_state_is_seeded_long_enough(by_kind):
    for c in by_kind["list_remove"] + by_kind["list_reorder"]:
        op = c.expected["ops"][0]
        if op["list"] == "sections":
            continue
        assert c.meta.get("len_before", 2) >= 2 or op["action"] == "reorder"


def test_layout_cases(by_kind):
    ops = [c.expected["ops"][0] for c in by_kind["layout"]]
    kinds = {o["op"] for o in ops}
    assert kinds == {"set_page_layout", "reset_pinned_section"}
    layouts = S.load_layouts_fixture()
    for c in by_kind["layout"]:
        o = c.expected["ops"][0]
        if o["op"] == "set_page_layout":
            assert o["layout"] in layouts["page_layouts"][o["page"]]
            assert o["layout"] != layouts["default_layouts"].get(o["page"])
        else:
            assert o == {"op": "reset_pinned_section", "page": "products", "sectionId": "__pinned_product_list"}
            assert c.site_state["pageSectionsAll"]["products"][0]["enabled"] is False


def test_multi_cases_have_two_or_three_independent_ops(by_kind):
    for c in by_kind["multi"]:
        ops = c.expected["ops"]
        assert 2 <= len(ops) <= 3
        assert len({o["target"]["sectionId"] for o in ops}) == len(ops)
        assert c.holdout == any(M.is_holdout(k) for k in c.meta["hold_keys"])
        assert c.target is None


def test_clarify_cases_have_two_candidates_and_no_ops(by_kind):
    for c in by_kind["clarify"]:
        assert c.expected["ops"] == [] and len(c.expected["candidates"]) >= 2
        assert all(S.find_section(c.site_state, sid) for sid in c.expected["candidates"])
        label = C.SECTION_LABELS[c.meta["section_type"]]
        assert label in c.request.lower()
        assert not any(sid in c.request for sid in c.expected["candidates"])  # names only the type


def test_unsupported_cases(by_kind):
    assert all(c.expected["ops"] == [] and c.expected["report_unsupported"] for c in by_kind["unsupported"])
    reasons = {c.meta["reason"] for c in by_kind["unsupported"]}
    assert "derived_from" in reasons and len(reasons) >= 10
    assert all(c.browser_ops == [] for c in by_kind["unsupported"])


def test_context_matches_site_state(cases):
    for c in cases[::50]:
        page = c.context["page"]
        assert c.context == S.context_of(c.site_state, page, S.load_context_fixture())


def test_generate_limit_and_kinds(m):
    few = C.generate(m, kinds=["toggle"], limit=7)
    assert len(few) == 7 and all(c.kind == "toggle" for c in few)


def test_summarize_and_stats(cases, gen):
    s = C.summarize(cases)
    assert s["cases"] == len(cases) and set(s["by_kind"]) == set(C.KINDS)
    assert gen.stats.get("skip:type:json", 0) == 5  # the five json fields on the default rows
    assert gen.stats.get("skip:derived", 0) == 0     # enumerate_targets already dropped them


def test_main_no_measure(tmp_path, capsys):
    out = tmp_path / "cases.jsonl"
    assert C.main(["--out", str(out), "--no-measure", "--kinds", "layout,clarify", "--limit", "10"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["written"] == 10 and res["rejects"] == 0
    assert out.exists() and (tmp_path / "cases-rejects.jsonl").exists()


def test_measure_rejects_refused_ops_with_a_fake_engine(cases):
    class FakeEngine:
        def run(self, state, ops, ctx=None):
            ok = all(o.get("op") != "reset" for o in ops)
            out = {"ok": ok, "results": [{"op": o, "ok": ok, "warnings": [], "written": ["x"], **({} if ok else {"error": {"code": "E_FIELD"}})} for o in ops],
                   "changes": [{"slice": "homeSections", "written": ["x"], "diff": [{"path": "x", "after": 1}]}] if ok else []}
            return E.EngineResult.from_output(out, returncode=0 if ok else 2, elapsed_s=0.0)
    sample = [next(c for c in cases if c.kind == "set_field"), next(c for c in cases if c.kind == "reset"),
              next(c for c in cases if c.kind == "clarify"), next(c for c in cases if c.kind == "image" and c.meta["variant"] == "pick_asset")]
    accepted, rejects = C.measure(sample, FakeEngine())
    assert [c.kind for c in accepted] == ["set_field", "clarify", "image"]
    assert rejects[0]["reason"] == "op_refused" and rejects[0]["kind"] == "reset"
    assert accepted[0].expected["diff"] == [{"path": "x", "after": 1}]
    assert accepted[1].expected["diff"] == []  # clarify never goes through the engine


@pytest.mark.engine
def test_measure_sample_through_the_real_engine(cases):
    if not E.engine_available():
        if REQUIRE:
            pytest.fail("SOUS_CHEF_LAB_REQUIRE_ENGINE=1 but the engine is unavailable: " + E.unavailable_reason())
        pytest.skip("engine unavailable: " + E.unavailable_reason())
    eng = E.Engine(cache_dir=E.default_cache_dir())
    picks = []
    for kind in ("set_field", "toggle", "set_enum", "list_update", "reset", "layout"):
        picks.append(next(c for c in cases if c.kind == kind))
    picks.append(next(c for c in cases if c.meta.get("mirror")))
    accepted, rejects = C.measure(picks, eng, log=lambda cid, ok, info: print(cid, ok, info.get("reason", ""), info.get("errors", "")))
    assert rejects == [], rejects
    assert all(c.expected["diff"] for c in accepted)
    print("[engine]", eng.stats())


# ── engine fingerprint on cases (LAB.md §2) ─────────────────────────────────

class _StampingEngine:
    """A stand-in that answers every op and carries an identity."""

    def __init__(self, fingerprint, after=1):
        self.fingerprint, self.commit, self.after = fingerprint, "c0ffee", after

    def run(self, state, ops, ctx=None):
        out = {"ok": True, "results": [{"op": o, "ok": True, "warnings": [], "written": ["x"]} for o in ops],
               "changes": [{"slice": "homeSections", "written": ["x"], "diff": [{"path": "x", "after": self.after}]}]}
        return E.EngineResult.from_output(out, returncode=0, elapsed_s=0.0)


def test_measure_stamps_the_engine_fingerprint(cases):
    c = copy.deepcopy(next(x for x in cases if x.kind == "set_field"))
    c.engine_fingerprint = None  # the module-scoped fixture may already carry the real engine's stamp
    accepted, rejects = C.measure([c], _StampingEngine("fp-1"))
    assert rejects == [] and accepted[0].engine_fingerprint == "fp-1"
    assert accepted[0].meta["measured"]["engine_commit"] == "c0ffee"
    row = accepted[0].to_row()
    assert row["engine_fingerprint"] == "fp-1" and C.Case.from_row(row).engine_fingerprint == "fp-1"
    legacy = {k: v for k, v in row.items() if k != "engine_fingerprint"}  # rows written before the stamp
    assert C.Case.from_row(legacy).engine_fingerprint is None


def test_remeasure_reports_changed_diffs_and_stale_fingerprints(cases):
    picks = [copy.deepcopy(x) for x in cases if x.kind in ("set_field", "toggle")][:3]
    first, _ = C.measure(picks, _StampingEngine("fp-1", after=1))
    res = C.remeasure(first, _StampingEngine("fp-2", after=2))
    assert res["fingerprint"] == "fp-2" and res["stale"] == 3 and res["rejects"] == []
    assert sorted(res["changed"]) == sorted(c.id for c in picks) and res["unchanged"] == 0
    assert all(c.engine_fingerprint == "fp-2" and c.expected["diff"] == [{"path": "x", "after": 2}] for c in res["accepted"])
    again = C.remeasure(res["accepted"], _StampingEngine("fp-2", after=2))
    assert again["stale"] == 0 and again["changed"] == [] and again["unchanged"] == 3


def test_remeasure_cli_rewrites_in_place(tmp_path, cases, monkeypatch):
    picks = [copy.deepcopy(x) for x in cases if x.kind == "set_field"][:2]
    C.measure(picks, _StampingEngine("fp-1", after=1))
    path = tmp_path / "cases.jsonl"
    C.write_jsonl(path, picks)
    monkeypatch.setattr(E, "Engine", lambda *a, **k: _StampingEngine("fp-2", after=2))
    assert C.main(["remeasure", "--cases", str(path), "--workers", "1"]) == 0
    rows = list(C.read_jsonl(path))
    assert len(rows) == 2 and all(r["engine_fingerprint"] == "fp-2" and r["expected"]["diff"] == [{"path": "x", "after": 2}] for r in rows)


# ── unambiguous requests (int-run1 finding B, LAB.md §2) ────────────────────

def test_name_fields_qualifies_shared_labels_or_refuses(m):
    cb = C.name_fields(C.scope_fields(m, m.section("contentBlock")))
    # one label, two types → type word; 'Headline' is generic, so the flat key rides along too (int-run2 B, below)
    assert cb["title"] == "Headline text (title)" and cb["titleColor"] == "Headline color (title)"
    assert cb["taglineFontSize"] == "Tagline › Text size in pixels"  # the group is part of the label, kept verbatim
    assert cb["bgImageMobile"] == "Background Image (Mobile)"  # the parenthetical is what tells it from 'Background Image'
    assert cb["textAlign"] is None and cb["splitContentAlign"] is None and cb["contentTextAlign"] is None  # 3 enums, 1 label
    co = C.name_fields(C.scope_fields(m, m.page("checkout")))
    assert co["sectionStyle.color"] == "Color (sectionStyle)" and co["labelStyle.color"] == "Color (labelStyle)"  # dotted parent
    ac = C.name_fields(C.scope_fields(m, m.page("account")))
    assert ac["orders.heading.text"] == "Heading text (orders.heading)" and ac["orders.heading.color"] == "Heading color (orders.heading)"
    pd = C.name_fields(C.scope_fields(m, m.page("product-details")))
    assert pd["globalSettings.cardTagsChipColor"] is None  # 'Text' × 7 under one parent: int-run1 "Remove the text from the product-details page"
    assert pd["productDetail.reviewTextColor"] == "Body Text Color"  # int-run1 asked for "body text color"
    hero = C.scope_fields(m, m.section("heroBanner"))
    assert "headline" in hero and "paddingY" in hero  # mirror item fields join the hero's own
    for c in m.containers():
        handed = [v.lower() for v in C.name_fields(C.scope_fields(m, c)).values() if v]
        assert len(handed) == len(set(handed)), c.name  # every phrase handed out is unique in its scope


def test_ambiguous_fields_are_skipped_not_asked_for(gen, cases):
    assert gen.stats["skip:ambiguous-label"] > 0
    skipped = set(gen.skipped_ambiguous)
    assert {"contentBlock.splitContentAlign", "product-details.globalSettings.cardTagsChipColor", "productCarousel.buttonText"} <= skipped
    positive = [c for c in cases if c.kind in C.SET_KINDS + ("reset", "list_update", "clarify") and c.meta.get("list") != "sections"]
    assert not any(c.fields == ["splitContentAlign"] for c in positive)
    for c in positive:  # case-insensitive: an effect-phrased toggle lowercases its verb mid-sentence ("…, show the …")
        assert c.meta["field_phrase"] and c.meta["field_phrase"].lower() in c.request.lower(), (c.id, c.request)
    assert all(c.meta["field_phrase"] for c in cases if c.kind == "image")  # both variants carry the phrase


def test_repeated_sections_are_named_by_id_and_shared_titles_never_used(gen, cases):
    rows = S.sections_of(gen.base, "home")
    by_type = {}
    for r in rows:
        by_type.setdefault(S.type_of_section(r["id"]), []).append(r["id"])
    repeated = {sid for ids in by_type.values() if len(ids) > 1 for sid in ids}
    assert {"contentBlock-3", "contentBlock-features", "productCarousel-2"} <= repeated
    titles = {}
    for r in rows:
        titles[S.section_title(r)] = titles.get(S.section_title(r), 0) + 1
    shared = {t for t, n in titles.items() if t and n > 1}
    assert shared, "the fixture state has two sections with one title (contentBlock-features / -icons)"
    for c in cases:
        sid = (c.target or {}).get("sectionId")
        if c.kind == "multi":
            assert all(s in c.request for s in c.meta["targets"] if s in repeated), c.request
        elif sid in repeated:
            assert sid in c.request, (c.id, c.request)
        assert not any(f"titled '{t}'" in c.request or f"the '{t}' section" in c.request for t in shared), c.request


def test_keyed_requests_name_the_location_by_name_and_id(cases):
    keyed = [c for c in cases if c.meta.get("keyed_by")]
    assert keyed
    locs = {l["id"]: l["name"] for l in S.load_context_fixture()["locations"]}
    for c in keyed:
        ident = c.fields[0].split(".")[1]
        assert ident in c.request and locs[ident] in c.request, c.request
        assert c.meta["field_phrase"] == C.keyed_phrase(_container(M.load(FIXTURES_DIR / "manifest.snapshot.json"), c.expected["ops"][0]).fields[c.fields[0].replace(ident, "{locationId}")], c.fields[0], S.load_context_fixture())


def test_page_requests_name_the_exact_page_id(cases):
    seen = 0
    for c in cases:
        t = c.target or {}
        if t.get("kind") == "page" and c.kind in C.SET_KINDS + ("reset",):
            seen += 1
            low = c.request.lower()
            assert f"the {t['page']} page" in low or f"the '{t['page']}' page" in low, c.request
    assert seen > 100


def test_reset_requests_say_reset_or_clear_plus_the_label(by_kind):
    for c in by_kind["reset"]:
        low = c.request.lower()
        assert c.meta["field_phrase"].lower() in low, c.request
        if c.meta["mode"] == "unset":
            assert ("reset" in low or "clear" in low) and "remove" not in low and "unset" not in low, c.request
        else:
            assert "default" in low, c.request


def test_nested_list_requests_name_the_parent_row(by_kind):
    nested = [c for kind in ("list_add", "list_remove", "list_reorder", "list_update") for c in by_kind[kind] if c.meta.get("depth") == 2]
    assert nested
    for c in nested:
        assert "the first " in c.request.lower(), c.request  # `the first footer column of the footer`, `the first slide of the hero banner`


def test_label_text_and_candidates():
    f = M.Field("taglineFontSize", {"label": "Tagline › Text size in pixels", "type": "number"})
    assert C.label_text(f) == "Tagline › Text size in pixels"
    assert C.phrase_candidates(f) == ["Tagline › Text size in pixels", "Tagline › Text size in pixels size"]
    nested = M.Field("sectionStyle.color", {"label": "Color", "type": "color"})
    assert C.phrase_candidates(nested) == ["Color (sectionStyle)", "Color color (sectionStyle)"]  # generic: no bare candidates
    plain = M.Field("sectionStyle.accent", {"label": "Accent", "type": "color"})
    assert C.phrase_candidates(plain) == ["Accent", "Accent color", "Accent (sectionStyle)", "Accent color (sectionStyle)"]
    placeholder = M.Field("pageHeaderHiddenPages.our-story", {"label": "Hide {item}", "type": "bool"})
    assert C.label_text(placeholder) == "Hide Our Story"  # `{item}` = the keyed segment's display name (refuter B, D2)
    assert C.label_text(M.Field("overlayColor", {"label": "Overlay {item}", "type": "color"})) == "Overlay color"
    assert C.label_text(M.Field("bgColor", {"type": "color"})) == "bg color"
    two = {"a": M.Field("a", {"label": "Caption", "type": "text"}), "b": M.Field("b", {"label": "Caption", "type": "text"})}
    assert C.name_fields(two) == {"a": None, "b": None}
    three = {**two, "c": M.Field("c", {"label": "Caption", "type": "color"})}
    assert C.name_fields(three)["c"] == "Caption color"
    # a generic label is told apart by its key, so two flat `Title` fields are no longer unresolvable
    assert C.name_fields({"a": M.Field("a", {"label": "Title", "type": "text"}), "b": M.Field("b", {"label": "Title", "type": "text"})}) == {"a": "Title (a)", "b": "Title (b)"}


# ── int-run2 finding B (2026-10-03): generic labels are located, foreign names are pinned ────────

GENERIC_PHRASE = re.compile(r"^(" + "|".join(sorted(C.GENERIC_LABELS)) + r")$", re.I)


def _label_phrase(c):
    return c.meta.get("label_phrase") or c.meta.get("field_phrase") or ""


def test_b_generic_labels_are_never_handed_out_bare(m, gen):
    header = next(c for c in m.containers("global") if c.name == "HeaderSection")
    names = C.name_fields(C.scope_fields(m, header))
    assert names["headerMegaMenuTextColor"] == "Text (header mega menu)"  # int-run2: "Reset the Text of the header" → headerTextColor
    assert names["headerMegaMenuBg"] == "Background (header mega menu)"  # "Change the Background on the header" → headerBgColor
    assert names["announcementBar.message"] == "Message (announcementBar)"  # dotted key: the parent, as before, but never bare
    assert names["headerTextColor"] is None  # three flat `Text Color` colours stay unresolvable — not a generic label, old rule
    assert C.name_fields(C.scope_fields(m, m.page("blog"), "blog"))["layout"] == "Layout (layout)"  # "Layout should be Grid View" → set_page_layout
    assert C.name_fields(C.scope_fields(m, m.page("checkout")))["sectionStyle.color"] == "Color (sectionStyle)"
    cards = next(c for c in m.containers("global") if c.name == "ProductCardSection")
    pc = C.name_fields(C.scope_fields(m, cards))
    assert pc["cardPillHoverColor"] == "Pill Hover Colors › Text"  # a group segment already locates it: not generic
    assert pc["cardThcChipColor"] == "Text (card thc chip)" and pc["cardThcChipBg"] == "Background (card thc chip)"  # un-skipped: the key locates them
    sm = next(c for c in m.containers("global") if c.name == "ShopMenuSection")
    assert C.name_fields(C.scope_fields(m, sm))["shopMenuLocationAlign"] == "Alignment (shop menu location)"  # `align` ~ `alignment` dropped
    hero = C.name_fields(C.scope_fields(m, m.section("heroBanner")))
    assert hero["headline"] == "Headline (headline)" and hero["headlineColor"] == "Headline (headlineColor)"  # label says it all → the key
    assert C.is_generic_label(M.Field("x", {"label": "Text", "type": "color"})) and not C.is_generic_label(M.Field("x", {"label": "Text Color", "type": "color"}))
    assert not C.is_generic_label(M.Field("x", {"label": "Default colors › Text", "type": "color"}))
    assert C.pseudo_group(M.Field("headerMegaMenuBg", {"label": "Background", "type": "color"}), "headerMegaMenuBg") == "header mega menu"
    assert C.pseudo_group(M.Field("layout", {"label": "Layout", "type": "enum"}), "layout") == ""
    for c in m.containers():
        for k, v in C.name_fields(C.scope_fields(m, c)).items():
            assert not (v and GENERIC_PHRASE.match(v.strip())), (c.name, k, v)


def test_b_generated_requests_never_carry_a_bare_generic_label(cases):
    seen = 0
    for c in cases:
        if c.kind in C.SET_KINDS + ("reset", "list_update", "clarify") and c.meta.get("list") != "sections":
            phrase = _label_phrase(c)
            assert phrase and not GENERIC_PHRASE.match(phrase.strip()), (c.id, c.request)
            if "(header mega menu)" in phrase or phrase.startswith("Layout (layout)"):
                seen += 1
                assert phrase in c.request, (c.id, c.request)
    assert seen >= 3  # the int-run2 residual fields are generated and carry the located phrase


def test_b_phrases_naming_another_container_are_pinned_to_their_own(m, gen, cases):
    assert gen._field_name(m.branding, "locationPickerCtaPlacement") == "Shop button placement (branding)"  # int-run2 → shopMenuLocationAlign
    assert gen._field_name(m.branding, "locationPickerCtaBorderColor") == "Shop button Border Color (branding)"  # → shopMenuLocationButtonBorderColor
    header = next(c for c in m.containers("global") if c.name == "HeaderSection")
    assert gen._field_name(header, "headerShowCart") == "Show Cart Icon (header)"  # `cart` names the sidebar cart panel
    branding_panel = next(c for c in m.containers("global") if c.name == "BrandingSection")
    assert gen._field_name(branding_panel, "footerLogo") == "Footer Logo (branding settings)"
    assert gen._field_name(m.section("contentBlock"), "buttonText") == "Button Text"  # `button` is in half the containers: names nothing
    assert gen._field_name(m.section("heroBanner"), "btnColor") == "Button Color"
    idx = gen.index
    assert idx.collides_with(m.branding, "Shop button placement").name == "ShopMenuSection"
    assert idx.collides_with(m.branding, "Border Radius") is None  # generic words only → no content words → no collision
    assert "shop" in idx.distinctive and "button" not in idx.distinctive and "card" not in idx.distinctive
    assert C.phrase_content_words("Headline Font — uppercase (titleUppercase)") == []
    assert C.phrase_content_words("Shop button placement (branding)") == ["shop", "button"]
    assert C.short_container_name(m.branding) == "branding" and C.short_container_name(header) == "header"
    assert C.short_container_name(m.page("blog"), "blog") == "blog page" and C.short_container_name(m.section("contentBlock")) == "content block"
    pinned = [c for c in cases if c.fields == ["locationPickerCtaPlacement"]]
    assert pinned and all("Shop button placement (branding)" in c.request for c in pinned), [c.request for c in pinned]
    eff = [c for c in cases if c.fields == ["headerShowCart"] and c.meta.get("effect_phrased")]
    assert eff and all(c.request.lower().count("(header)") == 1 for c in eff), [c.request for c in eff]  # the tail survives effect phrasing
    # the rule is rare by design: a handful of phrases per manifest, not a suffix on everything
    total = fired = 0
    for c in idx.containers:
        for v in C.name_fields(C.scope_fields(m, c)).values():
            if v:
                total += 1
                fired += idx.collides_with(c, v) is not None
    assert 20 <= fired <= total * 0.05, (fired, total)


# ── refuter B (2026-10-04): toggle polarity, `{item}` substitution, bool legibility ──────────────

def _toggle_value(c):
    op = c.expected["ops"][0]
    (key, value), = (op.get("values") or op["item"]).items()
    return key, value


def _first_toggle(cases, key):
    return next(c for c in cases if c.kind == "toggle" and c.fields == [key])


def test_d1_nav_toggles_are_phrased_by_effect_in_both_polarities(m):
    """`Hide Shop from header nav` is the aria-label snapshotted in the VISIBLE state; the engine's value is
    true = visible. The request says what should happen, never "turn on/off <negated label>"."""
    hs = m.panel("HeaderSection")
    f = hs.fields["pageVisibility.products.inHeaderNav"]
    assert f.spec["label"] == "Hide Shop from header nav" and C.label_is_negated(f)
    assert C.toggle_effect(f, f.key, True) == "Show Shop in the header nav"
    assert C.toggle_effect(f, f.key, False) == "Hide Shop from the header nav"
    fs = m.panel("FooterSection")
    g = fs.fields["pageVisibility.products.inFooterNav"]
    assert C.toggle_effect(g, g.key, True) == "Show Products in the footer"
    assert C.toggle_effect(g, g.key, False) == "Hide Products from the footer"
    assert "Probe text" not in C.label_text(g)  # the snapshot's probe residue stands for {item}
    # negative KEYS keep their meaning: true = hidden
    h = m.section("contentBlock").fields["hideImageEmptyBg"]
    assert C.key_is_negative(h.key) and not C.key_is_negative(f.key)
    assert C.toggle_effect(h, h.key, True) == "Hide the empty image background"
    assert C.toggle_effect(h, h.key, False) == "Show the empty image background"
    # positive `show*` keys under a `Hide …` label (ProductCardSection's snapshotted pill labels)
    p = m.panel("ProductCardSection").fields["cardShowTHC"]
    assert p.spec["label"] == "Hide THC pill" and C.toggle_effect(p, p.key, False) == "Hide the THC pill"
    assert C.toggle_effect(p, p.key, True) == "Show the THC pill"
    # an ordinary bool is left to turn on / turn off
    assert C.toggle_effect(m.section("productCarousel").fields["alwaysShowArrows"], "alwaysShowArrows", True) is None


def test_d1_generated_toggle_requests_never_turn_a_negated_label_on_or_off(m, cases):
    toggles = [c for c in cases if c.kind == "toggle" or (c.kind == "list_update" and c.meta.get("field_type") == "bool")]
    assert toggles
    for c in toggles:
        low = c.request.lower()
        assert not any(f"turn {w} hide" in low or f"turn {w} show" in low or f"switch hide" in low for w in ("on", "off")), c.request
        assert c.meta["field_phrase"].lower() in low, (c.id, c.request)
        key, value = _toggle_value(c)
        if c.meta.get("effect_phrased"):
            # the request's verb and the expected value agree with the KEY's polarity
            verb = re.search(r"\b(hide|show)\b", low)  # the effect verb is the first such word in the request
            assert verb, c.request
            hidden = value if C.key_is_negative(key) else not value
            assert (verb.group(1) == "hide") == hidden, (c.id, c.request, key, value)
    both = {_toggle_value(c)[1] for c in toggles if c.fields[0].endswith(".inHeaderNav")}
    assert both == {True, False}, "the seeded header nav has shown and hidden pages, so both polarities are generated"
    both_footer = {_toggle_value(c)[1] for c in toggles if c.fields[0].endswith(".inFooterNav")}
    assert both_footer == {True, False}
    shop = _first_toggle(cases, "pageVisibility.products.inHeaderNav")
    key, value = _toggle_value(shop)
    assert value is False and "hide shop from the header nav" in shop.request.lower(), shop.request
    story = _first_toggle(cases, "pageVisibility.our-story.inHeaderNav")
    assert _toggle_value(story)[1] is True and "show our story in the header nav" in story.request.lower(), story.request


def test_d1_multi_parts_phrase_negated_bools_by_effect(by_kind):
    parts = [c for c in by_kind["multi"] if any(k.endswith(".hidePaddingTop") for k in c.fields)]
    for c in parts:
        op = next(o for o in c.expected["ops"] if "hidePaddingTop" in o["values"])
        verb = "hide the padding top" if op["values"]["hidePaddingTop"] else "show the padding top"
        assert verb in c.request.lower(), c.request
        assert "turn on hide" not in c.request.lower() and "turn off hide" not in c.request.lower()


def test_d2_item_placeholder_takes_the_keyed_segment_or_item_noun(m):
    hs = m.panel("HeaderSection")
    f = hs.fields["pageVisibility.our-story.inHeaderNav"]
    assert f.spec["label"] == "Show {item} in header nav"
    assert C.label_text(f) == "Show Our Story in header nav"  # never `Show in header nav in header nav`
    assert C.item_display("pageVisibility.our-story.inHeaderNav", "Show {item} in header nav") == "Our Story"
    assert C.item_display("pageVisibility.about.inHeaderNav", "x {item}", ctxf=S.load_context_fixture()) == "About Us"  # fixture title
    assert C.item_display("footerLocationEnabled.{locationId}", "Hide {item}") == "location"
    assert C.item_display("locationOverrides.{locationId}.address", "{item}") == "address"  # the placeholder is not the item
    hero = C.scope_fields(m, m.section("heroBanner"))
    assert hero["btnColor"].spec["label"] == "{item} Color"
    assert C.label_text(hero["btnColor"], "btnColor") == "Button Color"  # never `btn color`
    assert C.label_text(hero["btnTextColor"], "btnTextColor") == "Button Text Color"
    assert C.label_text(hero["overlayColor"], "overlayColor") == "Overlay color"
    assert C.label_text(M.Field("enabled", {"label": "Hide {item}", "type": "bool"}), item="tier") == "Hide tier"
    ls = C.name_fields(C.scope_fields(m, m.panel("LayoutSection")))
    assert ls["pageHeaderHiddenPages.our-story"] == "Our Story (pageHeaderHiddenPages)"


def test_d2_generated_requests_carry_the_display_names(cases):
    story = _first_toggle(cases, "pageVisibility.our-story.inHeaderNav")
    assert "our story" in story.request.lower() and "in header nav in header nav" not in story.request.lower()
    btn = [c for c in cases if c.kind == "set_field" and c.fields == ["btnColor"]]
    assert btn and all("Button Color" in c.request and "btn" not in c.request.lower() for c in btn), [c.request for c in btn]
    assert not any(re.search(r"\bbtn\b", c.request) for c in cases), "no request names a field by its abbreviated key"


def test_d3_bools_sharing_a_label_are_qualified_by_their_key_words(m):
    cb = C.name_fields(C.scope_fields(m, m.section("contentBlock")))
    assert cb["titleUppercase"] == "Headline Font — uppercase (titleUppercase)"
    assert cb["titleFontFamily"] == "Headline Font selection"
    for page in ("privacy-policy", "terms-of-service", "return-refund"):
        names = C.name_fields(C.scope_fields(m, m.page(page)))
        assert names["headingUppercase"] == "Headline Color — uppercase (headingUppercase)", page  # mislabelled in the editor
    pd = C.name_fields(C.scope_fields(m, m.page("product-details")))
    assert pd["productDetail.productNameUppercase"] == "Product Name Font — uppercase (productDetail.productNameUppercase)"
    assert not any(v and v.endswith(" toggle") for c in m.containers() for v in C.name_fields(C.scope_fields(m, c)).values())


def test_d3_label_residue_is_stripped(m):
    cb = C.name_fields(C.scope_fields(m, m.section("contentBlock")))
    assert cb["taglineImage"] == "Tagline Image" and cb["footnote"] == "Footnote (footnote)"  # generic one-word label → + key (int-run2 B)
    assert cb["mapLocationIndex"] == "map location index"  # 'Or pick from locations' names nothing
    assert cb["bgImageMobile"] == "Background Image (Mobile)"  # other parentheticals stay
    assert C.label_text(M.Field("cardSideImage", {"label": "Side image (optional)", "type": "image"})) == "Side image"


def test_d3_no_residue_or_leaf_echo_in_any_request(cases):
    for c in cases:
        low = c.request.lower()
        assert "(optional)" not in low and "or pick from locations" not in low and "probe text" not in low, (c.id, c.request)
        assert not re.search(r"\b(font|color|style) toggle\b", low), (c.id, c.request)  # the old type-word qualifier
