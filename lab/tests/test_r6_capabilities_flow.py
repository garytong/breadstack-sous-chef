"""r6 create_page + option lists, wired end to end: the contrast groups' cases through r4collect on the fake service
(oracle scripts) and a create_page engine, the replay paths' same-batch stop, and the bar's out-of-scope share.

The engine tests need a launcher checkout with create_page (1d4193e2 or later, COOKIE_CUTTER_DIR); on an older
engine they skip (the groups' creates and declines are measured on the engine and dropped there)."""
from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from souschef_lab import FIXTURES_DIR
from souschef_lab import createpage as CP
from souschef_lab import engine as E
from souschef_lab.retrain import bar as BAR

LAB = Path(__file__).resolve().parents[1]
EXAM_MANIFEST = FIXTURES_DIR / "exam" / "manifest-real.json"
REQUIRE = os.environ.get("SOUS_CHEF_LAB_REQUIRE_ENGINE") == "1"
# the engine environment as the run was started: an earlier test may activate an engine bundle (enginebundle.activate
# sets COOKIE_CUTTER_DIR / SOUS_CHEF_ENGINE / ENGINE and the browser-defaults override for the rest of the process)
ENGINE_ENV = {k: os.environ.get(k) for k in ("COOKIE_CUTTER_DIR", "SOUS_CHEF_ENGINE", "ENGINE", "SOUS_CHEF_BROWSER_DEFAULTS")}


# --- run_engine_result: the stop for the replay paths that read an EngineResult ---------------------------------------------

@dataclass
class FakeEngine:
    refuse: set = field(default_factory=set)
    calls: list = field(default_factory=list)

    def run(self, state, ops, ctx=None, **kw):
        self.calls.append((copy.deepcopy(ops), kw))
        res = []
        for o in ops:
            if o.get("op") == "create_page" and o.get("title") in self.refuse:
                res.append({"op": o, "ok": False, "error": {"code": "E_SLUG", "message": "taken"}, "warnings": [],
                            "written": []})
            else:
                res.append({"op": o, "ok": True, "warnings": [], "written": [f"w{len(res)}"]})
        ok = all(r["ok"] for r in res)
        return E.EngineResult(ok=ok, results=res, changes=[{"diff": [{"path": f"p{i}"} for i in range(len(res))]}],
                              state={"n": len(ops)} if kw.get("print_state") else None, returncode=0 if ok else 2)


def test_run_engine_result_stops_calls_after_a_refused_create_and_keeps_the_result_type():
    eng = FakeEngine(refuse={"Events"})
    ops = [{"op": "create_page", "title": "Events"},
           {"op": "edit_list", "target": {"kind": "page", "page": "events"}, "list": "sections", "action": "add"},
           {"op": "set_fields", "target": {"kind": "section", "sectionId": "hero"}, "values": {"a": 1}}]
    r = CP.run_engine_result(eng, {}, ops, {}, print_state=True)
    assert isinstance(r, E.EngineResult) and not r.ok and r.returncode == 2
    assert len(eng.calls) == 2 and eng.calls[1][0] == [ops[0], ops[2]] and eng.calls[1][1] == {"print_state": True}
    assert [x["ok"] for x in r.results] == [False, False, True]
    assert r.results[1]["error"] == {"code": "E_TARGET", "message": CP.not_created_message("events")}
    assert r.error_codes == ["E_SLUG", "E_TARGET"] and r.state == {"n": 2}  # the second run's state: the add never applied
    assert len(r.diff()) == 2


def test_run_engine_result_is_the_engines_own_result_when_nothing_is_stopped():
    eng = FakeEngine()
    ops = [{"op": "create_page", "title": "Deals"}, {"op": "edit_list", "target": {"kind": "page", "page": "deals"}}]
    first = CP.run_engine_result(eng, {}, ops, {})
    assert first.ok and len(eng.calls) == 1
    eng2 = FakeEngine()
    plain = [{"op": "set_fields"}]
    assert CP.run_engine_result(eng2, {}, plain, {}).ok and eng2.calls == [(plain, {})]


# --- the bar: the out_of_scope share is over the rows still out of scope ----------------------------------------------------

def test_oos_where_counts_only_rows_still_out_of_scope():
    def reps(n, passed):
        return [{f"out_of_scope:r{j:02d}": int(j < passed) for j in range(n)} for _ in range(2)]
    arms = {"current": {"out_of_scope": reps(13, 12)}, "candidate": {"out_of_scope": reps(13, 12)}}
    keys = [f"out_of_scope:r{j:02d}" for j in range(12)]  # r12 stands for p137.k2: never a `where`
    w = {"current": reps(13, 11), "candidate": reps(13, 11)}
    rule = lambda b: next(r for r in b["rules"] if r["name"] == "oos_where")  # noqa: E731
    b = BAR.evaluate(arms, baseline=None, th={"prior_flip_rate": 0.08}, oos_where=w, oos_keys=keys)
    assert rule(b)["pass"] and "11.0/12" in rule(b)["detail"] and "need 10" in rule(b)["detail"]
    w["candidate"] = reps(13, 10)  # two misses of 12: r5's slack (11 of 13) kept
    assert rule(BAR.evaluate(arms, baseline=None, th={"prior_flip_rate": 0.08}, oos_where=w, oos_keys=keys))["pass"]
    w["candidate"] = reps(13, 9)
    assert not rule(BAR.evaluate(arms, baseline=None, th={"prior_flip_rate": 0.08}, oos_where=w, oos_keys=keys))["pass"]
    assert "/13" in rule(BAR.evaluate(arms, baseline=None, th={"prior_flip_rate": 0.08}, oos_where=w))["detail"]
    b = BAR.evaluate(arms, baseline=None, th={"prior_flip_rate": 0.08}, oos_where=w, oos_keys=[])
    assert not rule(b)["pass"] and "no out_of_scope rows" in rule(b)["detail"]


def test_the_exam_out_of_scope_set_has_twelve_rows_still_out_of_scope():
    from souschef_lab import exam as X
    rows = X.load_set("out_of_scope")
    oos = [r for r in rows if X.is_out_of_scope(r)]
    assert len(rows) == 13 and len(oos) == 12 and {r["id"] for r in rows} - {r["id"] for r in oos} == {"p137.k2"}


# --- the contrast groups through r4collect on a create_page engine ----------------------------------------------------------

def _create_page_engine_or_skip():
    if not E.engine_available():
        msg = "node and the engine (COOKIE_CUTTER_DIR) are needed: " + E.unavailable_reason()
        if REQUIRE:
            pytest.fail(msg)
        pytest.skip(msg)
    validate = Path(E.find_engine()).parent / "validate.js"
    if "create_page" not in validate.read_text(encoding="utf-8"):
        pytest.skip("the engine predates create_page (launcher 1d4193e2)")


@pytest.fixture(scope="module")
def flow():
    from souschef_lab import browser_defaults as BD
    with pytest.MonkeyPatch.context() as mp:
        for k, v in {**ENGINE_ENV, BD.ENV_OVERRIDE: ENGINE_ENV.get(BD.ENV_OVERRIDE)}.items():
            if v is None:
                mp.delenv(k, raising=False)
            else:
                mp.setenv(k, v)
        BD._load.cache_clear()
        _create_page_engine_or_skip()
        sys.path.insert(0, str(LAB))
        from souschef_lab import contrast as K
        from souschef_lab.manifest import load
        m = load(str(EXAM_MANIFEST))
        eng = E.PersistentEngine(manifest_path=EXAM_MANIFEST, size=1)
        try:
            cases, rep = K.contrast_round(m, eng, sides=(), site_font_sides=(), vague=False, ask_first=False,
                                          progress=lambda s: None)
            yield {"m": m, "eng": eng, "cases": cases, "rep": rep}
        finally:
            eng.close()
    BD._load.cache_clear()


def _run(flow, cases, scripts):
    from souschef_lab import collect as C
    from souschef_lab import r4collect as RC
    from souschef_lab.state import load_context_fixture
    from tests import fake_service as FS
    doc = json.loads(EXAM_MANIFEST.read_text(encoding="utf-8"))
    svc = FS.FakeService(manifest=doc, key="k")
    svc.scripts.update(scripts)
    svc.start()
    try:
        client, _ = C.connect(svc.url, "k", EXAM_MANIFEST, cases[:1], sse=False, timeout_s=30, build_id="t")
        return {c["id"]: RC.run_r4_case(c, client, flow["eng"], route="default", fields=None, manifest=flow["m"],
                                        extras=load_context_fixture(), assets=[], site_prefix="lab-t-")
                for c in cases}
    finally:
        svc.stop()


def test_every_capability_case_is_kept_and_the_oracle_answer_passes(flow):
    from souschef_lab import r4collect as RC
    cases, rep = flow["cases"], flow["rep"]
    by_mode: dict[str, int] = {}
    for c in cases:
        by_mode[c["score"]["mode"]] = by_mode.get(c["score"]["mode"], 0) + 1
    assert rep["capability_dropped"] == {}, rep["capability_dropped"]
    assert by_mode.get("page", 0) >= 30 and by_mode.get("list", 0) >= 10 and by_mode.get("where") == 4, by_mode
    rows = _run(flow, cases, RC.oracle_scripts(cases, "lab-t-"))
    bad = {cid: (r["score"]["verdict"], r["score"]["reasons"], r["score"]["flags"])
           for cid, r in rows.items() if not r["score"]["usable"]}
    assert not bad, bad
    for c in cases:  # a create's tool result carries the slug the browser posts (render: loop.browserResult)
        row = rows[c["id"]]
        if c["score"]["mode"] == "page" and c["meta"]["oracle_ops"]:
            first = row["engine_results"][0]["results"][0]
            assert first["ok"] and first.get("slug") == c["meta"]["oracle_ops"][0].get("slug", first.get("slug"))


def test_wrong_answers_fail_creates_declines_and_listings(flow):
    """Scripts that answer each capability case the wrong way: a create answered with report_unsupported pages (the
    r5 answer), a decline answered by creating the page anyway plus a section on it (refused E_SLUG, the add stopped
    E_TARGET by the browser rule), an unknown list called empty, a long list named in part with no "...and N more"."""
    from souschef_lab import r4collect as RC
    from souschef_lab.r4score import listing_view
    cases = flow["cases"]
    create = next(c for c in cases if c["score"]["mode"] == "page" and c["meta"]["oracle_ops"]
                  and c["meta"]["contrast"].get("group", c["meta"].get("group")) != "template")
    decline = next(c for c in cases if c["score"]["mode"] == "page" and c["meta"].get("attempt")
                   and (c["expected"]["page"].get("why") or "exists") == "exists")
    unknown = next(c for c in cases if c["score"]["mode"] == "list" and listing_view(c)["situation"] == "unknown")
    long_ = next(c for c in cases if c["score"]["mode"] == "list" and listing_view(c)["situation"] in ("long", "truncated"))
    slug = CP.create_page_slugs(decline["meta"]["attempt"][0])[-1]
    wrong = {
        create["id"]: {"mode": "where"},
        decline["id"]: {"ops": decline["meta"]["attempt"] + [
            {"op": "edit_list", "target": {"kind": "page", "page": slug}, "list": "sections", "action": "add",
             "item": {"id": "heroBanner"}}], "text": "Done — I made the page."},
        unknown["id"]: {"ops": [], "text": f"You don't have any {listing_view(unknown)['list']} yet."},
        long_["id"]: {"ops": [], "text": "\n".join(f"- {n}" for n in listing_view(long_)["names"][:3])},
    }
    chosen = [create, decline, unknown, long_]
    scripted = []
    for c in chosen:
        w = wrong[c["id"]]
        s = copy.deepcopy(c)
        if w.get("mode") == "where":
            s["score"] = {"mode": "where"}
            s["expected"] = {**s["expected"], "out_of_scope": {"expected_where": ["pages"]}}
        else:
            s["score"] = {"mode": "keys"}
            s["meta"] = {**s["meta"], "oracle_ops": w["ops"]}
            s["expected"] = {**s["expected"], "ops": w["ops"]}
        scripted.append(s)
    scripts = RC.oracle_scripts(scripted, "lab-t-")
    for c in chosen[2:]:  # the listing texts: a text-only turn
        from tests.fake_service import Rotation
        done = ("turn.done", {"stop": "end_turn", "usage": {"input_tokens": 1000, "output_tokens": 100, "calls": 1}})
        scripts[f"lab-t-{c['id']}"] = Rotation([[[("text.delta", {"text": wrong[c["id"]]["text"]}), done]]])
    rows = _run(flow, chosen, scripts)
    for c in chosen:
        assert not rows[c["id"]]["score"]["usable"], (c["id"], rows[c["id"]]["score"])
    res = rows[decline["id"]]["engine_results"][0]["results"]
    assert [r["ok"] for r in res] == [False, False]
    assert res[0]["error"]["code"] == "E_SLUG" and res[1]["error"] == {"code": "E_TARGET",
                                                                     "message": CP.not_created_message(slug)}


def test_the_capability_report_set_is_true_on_the_engine_and_its_oracle_passes(flow):
    """eval's `capability:built` set (contrast.capability_report_cases): every case measures on a create_page engine
    (contrast.validate: a create applies, a decline is refused with its code) and the right answer scores PASSING;
    its requests are none of the training jobs'."""
    from souschef_lab import contrast as K
    from souschef_lab import r4collect as RC
    from souschef_lab.r4score import PASSING
    from souschef_lab.state import load_context_fixture
    rep_cases = K.capability_report_cases(flow["m"])
    assert 12 <= len(rep_cases) <= 18 and all(c["id"].startswith("caph-") for c in rep_cases)
    train_requests = {j[1] for j in K.PAGE_JOBS} | {j[0] for j in K.LIST_JOBS}
    assert not train_requests & {c["request"] for c in rep_cases}
    for c in rep_cases:
        why = K.validate(c, flow["m"], flow["eng"], load_context_fixture())
        assert not why, (c["id"], c["request"], why)
    rows = _run(flow, rep_cases, RC.oracle_scripts(rep_cases, "lab-t-"))
    bad = {cid: (r["score"]["verdict"], r["score"]["reasons"]) for cid, r in rows.items() if r["score"]["verdict"] not in PASSING}
    assert not bad, bad
