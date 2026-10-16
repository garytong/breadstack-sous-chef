"""The new-key probe (drift arm P, the eval's `probe` set) after r5: rows run on
the state they were generated on, so none targets a section its run state
lacks (r5: 7 of 17 rows did — contentBlock-2/-3/-features, newsletter — and
two keys were reported dodged by rows no model could pass); the engine check;
perks stackedImage* (visible: imagePosition top ← style image-text) and gallery
imagesLink[] probes; trajectories next to results; stable row ids; the offline
recheck. The real-engine tests skip without node + COOKIE_CUTTER_DIR."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB / "tests"))

from souschef_lab import drift as D  # noqa: E402
from souschef_lab import probe as PR  # noqa: E402
from souschef_lab import probe_recheck as RC  # noqa: E402
from souschef_lab.exam import MANIFEST_FILE, vintage_state  # noqa: E402
from souschef_lab.manifest import Manifest  # noqa: E402
from souschef_lab.state import find_section  # noqa: E402
from test_examrebase import REAL, rows_all  # noqa: E402
from test_rebase import _need_engine  # noqa: E402

# r5's new keys (manifest 1184ff9a), on the exam manifest extended as `with_new_keys` does
NEW_KEYS = [
    "sections.contentBlock.stackedImageGap", "sections.contentBlock.stackedImageHeight",
    "sections.gallery.imagesLinkNewTab", "sections.gallery.imagesLink[].value",
    "sections.perksCards.stackedImageGap", "sections.perksCards.stackedImageHeight",
    "sections.perksIcons.stackedImageGap", "sections.perksIcons.stackedImageHeight",
    "sections.newsletter.ctaFontSize", "sections.gallery.imagesLink[]",
]


def with_new_keys() -> dict:
    """The exam manifest with 1184ff9a's new fields: stackedImage* (conditional_on imagePosition = top, itself
    conditional_on style image-text / img-left) on contentBlock and the perks sections; gallery imagesLinkNewTab and
    the scalar list imagesLink[] (no add op)."""
    d = copy.deepcopy(REAL)
    for n in ("contentBlock", "perksCards", "perksIcons"):
        f = d["sections"][n]["fields"]
        f["imagePosition"] = {"enum": ["right", "left", "top", "bottom"], "path": "settings.imagePosition", "type": "enum",
                              "label": "Image Position", "enum_labels": {"top": "Top", "left": "Left", "right": "Right", "bottom": "Bottom"},
                              "conditional_on": {"key": "style", "value": ["image-text", "img-left"]}}
        f["stackedImageGap"] = {"max": 48, "min": 0, "step": 2, "type": "range", "label": "Image Gap", "unset_value": None,
                                "path": "settings.stackedImageGap", "conditional_on": {"key": "imagePosition", "value": "top"}}
        f["stackedImageHeight"] = {"max": 800, "min": 120, "step": 10, "type": "range", "label": "Image Height",
                                   "path": "settings.stackedImageHeight", "conditional_on": {"key": "imagePosition", "value": "top"}}
    g = d["sections"]["gallery"]
    g["fields"]["imagesLinkNewTab"] = {"path": "settings.imagesLinkNewTab", "type": "bool", "label": "Open image links in new tab"}
    g["lists"]["imagesLink"] = {"ops": [], "path": "settings.imagesLink",
                                "item": {"value": {"path": "settings.imagesLink[]", "type": "text", "label": "Image {n} link"}}}
    return d


@pytest.fixture(scope="module")
def nm(tmp_path_factory):
    p = tmp_path_factory.mktemp("m") / "manifest.json"
    p.write_text(json.dumps(with_new_keys()), encoding="utf-8")
    return Manifest(json.loads(p.read_text(encoding="utf-8")), p)


@pytest.fixture(scope="module")
def shared():
    """The state drift / eval run their exam on: the vintage home of the exam (no contentBlock-2/-3/-features, no
    newsletter, no perks section)."""
    return vintage_state(MANIFEST_FILE, D.home_context(rows_all()))


class SectionEngine(D.ManifestEngine):
    """ManifestEngine that also refuses an op on a section the state does not have (the real engine: E_TARGET)."""

    def __call__(self, state, ops, ctx):
        for op in ops:
            t = op.get("target") or {}
            if t.get("kind") == "section" and find_section(state or {}, str(t.get("sectionId")), t.get("page") or "home") is None:
                return {"ok": False, "changes": [], "state": state,
                        "results": [{"op": o.get("op"), "ok": False, "warnings": [], "written": [],
                                     "error": {"code": "E_TARGET", "message": "no such section"}} for o in ops]}
        return super().__call__(state, ops, ctx)

    run = __call__


def _run_state(row: dict, shared: dict) -> dict:
    """What a runner runs a row on: the row's own state when it carries one, else the arm's shared state."""
    st = row.get("site_state")
    return st if isinstance(st, dict) and st else shared


# ── the r5 failure: rows targeting a section their run state lacks ──

def test_no_probe_row_targets_a_section_missing_from_its_run_state(nm, shared):
    """Fails on r5's probe_rows (rows had no state; the fallback's contentBlock-2/-3 and newsletter rows ran on the
    shared vintage state, which lacks them), passes now — with or without an engine, as both call sites ran."""
    keys = ["sections.newsletter.ctaFontSize", "sections.contentBlock.stackedImageHeight", "sections.contentBlock.stackedImageGap"]
    rows, cov = D.probe_rows(nm, keys, state=shared, per_key=4)
    assert rows and all(cov[k] >= 1 for k in keys), cov
    missing = [(r["meta"]["probe_key"], r["target"]) for r in rows
               if find_section(_run_state(r, shared), r["target"]["sectionId"], "home") is None]
    assert not missing, f"probe rows whose section is not on the state they run on: {missing}"
    for r in rows:
        assert PR.run_state(r, shared) is r["site_state"] and PR.target_present(r, PR.run_state(r, shared))
        assert {s["id"] for s in r["context"]["sections"]} >= {r["target"]["sectionId"]}  # the context is that state's


def test_the_drift_runner_passes_every_probe_row_on_its_own_state(nm, shared, monkeypatch):
    """The DRY arm with an oracle model and an engine that refuses absent sections: every probe row passes (r5:
    the newsletter and contentBlock-2/-3 rows were refused whatever the model did), and each trimmed row keeps its
    conversation."""
    monkeypatch.setattr(D, "ManifestEngine", SectionEngine)
    keys = ["sections.newsletter.ctaFontSize", "sections.contentBlock.stackedImageHeight"]
    rows, _ = D.probe_rows(nm, keys, state=shared, per_key=4)
    tgt = D.Target(dry=True, dry_model="oracle", fake_engine=True, workers=2, progress=lambda m: None)
    out = tgt.run_rows("P", rows, nm.path, nm, site_prefix="lab-tp-", probe=True, home=D.home_context(rows_all()))
    assert len(out) == len(rows) and all(r["pass"] == 1 for r in out), [(r["probe_key"], r["verdict"], r["reasons"]) for r in out]
    for r in out:
        tr = r["trajectory"]
        assert tr["request"] and tr["probe_key"] == r["probe_key"] and tr["tool_calls"] and tr["engine"][0]["ok"]
        assert tr["target"]["sectionId"] in tr["context_sections"]


def test_the_engine_check_drops_refused_and_hidden_cases(nm, shared):
    """With an engine, a case the engine refuses on its state is dropped; a case writing a field its state hides
    (stackedImage* on contentBlock-features: style feature-cards hides imagePosition) is never picked."""
    class Refuse(SectionEngine):
        def __call__(self, state, ops, ctx):
            if any((op.get("target") or {}).get("sectionId") == "contentBlock-2" for op in ops):
                return {"ok": False, "changes": [], "state": state, "results": [
                    {"op": o.get("op"), "ok": False, "written": [], "warnings": [], "error": {"code": "E_FIELD"}} for o in ops]}
            return super().__call__(state, ops, ctx)
        run = __call__

    stats: dict = {}
    keys = ["sections.contentBlock.stackedImageHeight", "sections.contentBlock.stackedImageGap"]
    rows, _ = D.probe_rows(nm, keys, state=shared, per_key=6, engine=Refuse(nm), stats=stats)
    sids = {r["target"]["sectionId"] for r in rows}
    assert "contentBlock-2" not in sids and not sids & {"contentBlock-features", "contentBlock-icons"}, sids
    dropped = {w for k in keys for w in stats[k]["dropped"]}
    assert "op_refused:E_FIELD" in dropped and any(w.startswith("hidden:") for w in dropped), stats
    for r in rows:
        c = nm.section(r["target"]["sectionId"])
        assert PR.field_visible(nm, c, c.fields[r["meta"]["probe_key"].rsplit(".", 1)[1]], r["site_state"],
                                section_id=r["target"]["sectionId"])


# ── the keys r5's generator could not probe ──

def test_perks_and_gallery_link_keys_are_probed(nm, shared):
    stats: dict = {}
    rows, cov = D.probe_rows(nm, NEW_KEYS, state=shared, per_key=4, engine=SectionEngine(nm), stats=stats)
    assert all(cov[k] >= 1 for k in NEW_KEYS), cov
    assert cov["sections.gallery.imagesLink[].value"] == 4 and cov["sections.gallery.imagesLink[]"] == 4
    by_key: dict[str, list] = {}
    for r in rows:
        by_key.setdefault(r["meta"]["probe_key"], []).append(r)
    for k in ("sections.perksCards.stackedImageGap", "sections.perksIcons.stackedImageHeight"):
        for r in by_key[k]:
            sid = r["target"]["sectionId"]
            s = find_section(r["site_state"], sid)["settings"]
            assert s["style"] == "image-text" and s["imagePosition"] == "top", (k, s.get("style"), s.get("imagePosition"))
            c = nm.section(sid)
            assert PR.field_visible(nm, c, c.fields[k.rsplit(".", 1)[1]], r["site_state"], section_id=sid)
            assert sid in {x["id"] for x in r["context"]["sections"]}
    links = by_key["sections.gallery.imagesLink[].value"] + by_key["sections.gallery.imagesLink[]"]
    assert len({r["meta"]["source_case"] for r in links}) == len(links)  # the two keys ask different turns
    for r in links:
        op = r["meta"]["generator_ops"][0]
        assert op["op"] == "edit_list" and op["list"] == "imagesLink" and op["action"] == "update" and set(op["item"]) == {"value"}
        g = find_section(r["site_state"], "gallery")["settings"]
        assert isinstance(g["imagesLink"], list) and op["index"] < len(g["imagesLink"]) and g["imagesLink"][op["index"]] != op["item"]["value"]
        assert r["expected"]["keys"] == [{"target": op["target"], "key": "value", "list": "imagesLink"}]


def test_field_visible_follows_the_gate_chain(nm):
    from souschef_lab.state import build_state
    st = build_state(nm, template="vintage", home_sections=[{"id": "perksCards", "settings": {"style": "feature-cards", "imagePosition": "top"}},
                                                             {"id": "perksIcons", "settings": {"style": "image-text", "imagePosition": "top"}},
                                                             {"id": "contentBlock-1", "settings": {"style": "image-text", "imagePosition": "left"}}])
    f = lambda sid: PR.field_visible(nm, nm.section(sid), nm.section(sid).fields["stackedImageGap"], st, section_id=sid)  # noqa: E731
    assert f("perksIcons") and not f("perksCards") and not f("contentBlock-1")


def test_row_ids_are_the_same_in_every_process(nm):
    code = ("import json,sys; sys.path.insert(0, %r); from souschef_lab import drift as D; from souschef_lab.manifest import Manifest; "
            "from souschef_lab.exam import vintage_state, MANIFEST_FILE; p=%r; m=Manifest(json.load(open(p)), p); "
            "rows,_=D.probe_rows(m, ['sections.newsletter.ctaFontSize'], state=vintage_state(MANIFEST_FILE, None), per_key=2); "
            "print(json.dumps([r['id'] for r in rows]))") % (str(LAB), str(nm.path))
    ids = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=LAB,
                          env={**os.environ, "PYTHONHASHSEED": s}).stdout.strip() for s in ("1", "2")]
    assert ids[0] == ids[1] and json.loads(ids[0])


def test_trajectory_keeps_the_conversation_without_the_fields_block():
    case = {"request": "make it so", "target": {"kind": "section", "sectionId": "newsletter"}, "context": {"page": "home", "sections": [{"id": "newsletter"}]},
            "expected": {"keys": [{"key": "ctaFontSize"}]}, "meta": {"probe_key": "sections.newsletter.ctaFontSize", "kind": "set_range"}}
    row = {"status": "ok", "verdict": "invalid_op", "reasons": ["refused_ops"], "reply": "x" * 5000,
           "tool_calls": [{"id": "a", "name": "set_fields", "input": {"values": {"ctaFontSize": 20}}}],
           "engine_results": [{"ok": False, "results": [{"op": {"op": "set_fields"}, "ok": False, "error": {"code": "E_TARGET"}, "warnings": []}]}],
           "messages": [{"role": "user", "content": "FIELDS BLOCK"}, {"role": "assistant", "content": [{"type": "tool_use", "name": "set_fields"}]}]}
    t = PR.trajectory(row, case)
    assert t["engine"][0]["results"] == [{"error": {"code": "E_TARGET"}}] and t["tool_calls"][0]["name"] == "set_fields"
    assert "FIELDS BLOCK" not in json.dumps(t) and len(t["reply"]) < 1600 and t["context_sections"] == ["newsletter"]


# ── the eval's plan ──

def test_eval_plan_sets_builds_the_probe_with_the_engine(tmp_path, nm):
    from souschef_lab.retrain import eval_stage as EV
    from test_eval_stage import RUN, seeded
    s = seeded(tmp_path, with_new_keys())
    s.write_json(f"runs/{RUN}/run.json", {"run_id": RUN, "deployed": {"hash": nm.hash}, "trigger": {"drift_run": "dr-x"}})
    s.write_json("drift/dr-x/report.json", {"impact": {"new_keys": NEW_KEYS}, "verdict": {"fired": ["d"], "dodged_keys": NEW_KEYS[:2]}})
    seen = []

    class Spy(SectionEngine):
        def __call__(self, state, ops, ctx):
            seen.append(1)
            return super().__call__(state, ops, ctx)
        run = __call__

    sets = EV.plan_sets(s, RUN, mode="full", local=tmp_path / "w", th={"probe_cases_per_key": 4, "probe_max_keys": 40},
                        progress=lambda m: None, probe_engine=Spy(nm))
    assert seen and sets["probe"] and all(sets["probe_coverage"][k] >= 1 for k in NEW_KEYS)
    for r in sets["probe"]:
        assert PR.target_present(r, PR.run_state(r, {})) and r["meta"]["own_state"]
    plan = json.loads((tmp_path / "w" / "eval" / "probe-plan.json").read_text())
    assert len(plan["rows"]) == len(sets["probe"]) and set(plan["dropped"]) == set(NEW_KEYS)
    assert all(r["target"]["sectionId"] in r["state_sections"] for r in plan["rows"] if r["target"].get("kind") == "section")


# ── the offline recheck ──

def test_recheck_marks_rows_r5_could_not_pass(nm, shared):
    """Rebuild r5's rows, give every passable row a pass and every other a fail (as the arms did): the keys whose
    failures all sit on missing sections come out false_alarm / unmeasured, not dodged."""
    keys = ["sections.newsletter.ctaFontSize", "sections.contentBlock.stackedImageHeight"]
    legacy = RC.legacy_probe_rows(nm, keys, state=shared, per_key=4)
    assert any(r["target"]["sectionId"] == "newsletter" for r in legacy)  # the r5 shape: a row the shared state cannot run
    results = []
    for r in legacy:
        ok, _ = RC.passable(r, shared)
        results.append({"id": f"p00000000-{r['index']}", "probe_key": r["probe_key"], "status": "ok", "pass": int(ok),
                        "written": [json.dumps(r["target"], separators=(",", ":")) + " x"] if ok else []})
    res = RC.recheck(nm, keys, {"P-1": results}, shared=shared, per_key=4)
    v = res["sets"]["P-1"]["keys"]
    assert v["sections.newsletter.ctaFontSize"]["verdict"] == "unmeasured"
    h = v["sections.contentBlock.stackedImageHeight"]
    assert h["verdict"] in ("false_alarm", "ok") and h["rate_passable"] == 1.0
    assert not res["sets"]["P-1"]["pairing_mismatches"] and not res["sets"]["P-1"]["unmatched"]


# ── the real engine ──

@pytest.mark.engine
def test_every_probe_row_is_accepted_by_the_real_engine_on_its_run_state(nm, shared):
    _need_engine()
    from souschef_lab.engine import GOLDEN_CTX, PersistentEngine
    eng = PersistentEngine(manifest_path=nm.path, size=1)
    try:
        stats: dict = {}
        rows, cov = D.probe_rows(nm, NEW_KEYS, state=shared, per_key=4, engine=eng, stats=stats)
        assert all(cov[k] >= 1 for k in NEW_KEYS), (cov, stats)
        for r in rows:
            ops = [o for o in r["meta"]["generator_ops"] if o.get("op") != "pick_asset"]
            res = eng.run(PR.run_state(r, shared), ops, GOLDEN_CTX)
            assert res.ok, (r["meta"]["probe_key"], r["target"], res.errors)
            for x in res.results:  # W_HIDDEN is always warned for a gated field; the gate's value says it is open
                for h in x.get("hidden") or []:
                    if h.get("rule") == "conditional_on":
                        assert PR._gate_hit(h.get("byValue"), (h.get("gate") or {}).get("value")), (r["meta"]["probe_key"], h)
    finally:
        eng.close()


@pytest.mark.engine
def test_the_eval_arm_passes_every_probe_row_dry(nm, shared):
    _need_engine()
    from souschef_lab.retrain import eval_stage as EV
    rows, _ = D.probe_rows(nm, ["sections.newsletter.ctaFontSize", "sections.gallery.imagesLink[].value",
                                "sections.perksCards.stackedImageGap"], state=shared, per_key=2)
    arms = EV.Arms(dry=True, workers=1, progress=lambda m: None)
    out = arms.exam("candidate", "probe", rows, nm.path, nm, sample=1, prefix="lab-tpe-", home=D.home_context(rows_all()))
    assert all(r["pass"] == 1 for r in out), [(r["probe_key"], r["verdict"], r["reasons"]) for r in out]
    assert all(r["trajectory"]["engine"] and r["trajectory"]["engine"][0]["ok"] for r in out)
