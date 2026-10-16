"""create_page through the lab's mirrors (the r6 tools): the tool names in the
service's order, the slug the browser posts and the service forwards, the
browser's same-batch stop after a refused create (client/turns.js applyOps,
loop/turn.go dispatch), and the engine ctx the collector and the verifier
share (host/previewApi.js buildCtx: pageSlugs, tags, brands, a basic
template).

Pure-Python tests always run. The node-backed ones need an engine with
create_page (launcher 1d4193e2 or later: COOKIE_CUTTER_DIR / SOUS_CHEF_ENGINE)
and skip on an older one.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
for _p in (LAB, LAB / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fake_service as FS  # noqa: E402
from souschef_lab import FIXTURES_DIR  # noqa: E402
from souschef_lab import collect as C  # noqa: E402
from souschef_lab import createpage as CP  # noqa: E402
from souschef_lab import engine as E  # noqa: E402
from souschef_lab import render as R  # noqa: E402
from souschef_lab import state as S  # noqa: E402
from souschef_lab import verify as V  # noqa: E402

TOOLS = json.loads((FIXTURES_DIR / "prompt" / "tools.json").read_text(encoding="utf-8"))
REQUIRE = os.environ.get("SOUS_CHEF_LAB_REQUIRE_ENGINE") == "1"


# --- the tool names ----------------------------------------------------------------------------------------------------------

def test_tool_names_are_the_services_fixed_order():
    """toolgen.Order at the deployed commit (fixtures/prompt/tools.json is captured from that binary)."""
    assert list(C.TOOL_NAMES) == [t["name"] for t in TOOLS]
    assert C.TOOL_NAMES.index("create_page") == C.TOOL_NAMES.index("set_page_layout") + 1


def test_browser_ops_are_every_browser_run_tool_by_name():
    browser = {"set_fields", "edit_list", "reset", "restore_default", "set_page_layout", "create_page", "reset_pinned_section"}
    assert C.BROWSER_OPS == browser  # by name, not TOOL_NAMES[:n] — reset_pinned_section sits after create_page now
    assert set(V.BROWSER_OPS) == browser
    assert set(E.ENGINE_OPS) == browser | {"pick_asset"}
    assert set(C.TOOL_NAMES) == browser | C.SERVER_TOOLS | {C.PICK_ASSET}
    assert "create_page" not in V.NON_OPS  # a create is an op the verifier replays


# --- the slug rule -----------------------------------------------------------------------------------------------------------

# pageSlug.slugify's own output for these inputs (node, launcher 1d4193e2, 2026-10-16).
SLUGIFY_JS = {"Summer Events": "summer-events", "  Deals!! ": "deals", "Ünïcode Café": "ncode-caf", "a b﻿c": "a-b-c",
              "x\u0085y": "xy", "--a--b--": "-a-b-", "Tab\tbed": "tab-bed", "KelvinK": "kelvink", "İstanbul": "istanbul",
              "\U0001F389 Party \U0001F389": "party", "A_B": "ab", "foo---bar  baz": "foo-bar-baz", " x　": "x", "": ""}


def test_slugify_is_the_browsers():
    for s, want in SLUGIFY_JS.items():
        assert CP.slugify(s) == want, s
    assert CP.slugify(None) == "" and CP.slugify(7) == ""  # callers only pass strings (createPageSlugs)


def _pageslug_js() -> Path | None:
    for root in (os.environ.get("COOKIE_CUTTER_DIR"), str(E.DEFAULT_COOKIE_CUTTER_DIR)):
        if root and (Path(root) / "launcher/client/src/lib/pageSlug.js").is_file():
            return Path(root) / "launcher/client/src/lib/pageSlug.js"
    return None


def test_slugify_equals_the_checkouts_on_the_spot(tmp_path):
    """The table above against the checkout's own pageSlug.js (plain ESM, no Vite), when there is one."""
    js = _pageslug_js()
    if js is None or shutil.which("node") is None:
        pytest.skip("node and a launcher checkout with lib/pageSlug.js (1d4193e2+) are needed")
    script = tmp_path / "slug.mjs"
    script.write_text(f"import {{ slugify }} from {json.dumps(js.resolve().as_uri())};\n"
                      "const xs = JSON.parse(process.argv[2]);\n"
                      "process.stdout.write(JSON.stringify(xs.map((s) => slugify(s))));\n", encoding="utf-8")
    inputs = list(SLUGIFY_JS)
    out = subprocess.run(["node", str(script), json.dumps(inputs)], capture_output=True, text=True, timeout=60, check=True)
    assert json.loads(out.stdout) == [CP.slugify(s) for s in inputs]


def test_page_of_follows_the_browsers_nullish_chain():
    assert CP.page_of({"op": "edit_list", "target": {"kind": "page", "page": "deals"}}) == "deals"
    assert CP.page_of({"op": "set_page_layout", "page": "deals"}) == "deals"
    assert CP.page_of({"op": "pick_asset", "for": {"target": {"kind": "section", "page": "deals"}, "key": "img"}}) == "deals"
    assert CP.page_of({"op": "set_fields", "target": {"kind": "section", "sectionId": "hero"}}) is None  # the viewed page
    # `??`: an empty target.page decides (no page), it does not fall through to `page` (turns.js pageOf)
    assert CP.page_of({"op": "x", "target": {"page": ""}, "page": "deals"}) is None
    assert CP.page_of({"op": "x", "target": {"page": None}, "page": "deals"}) == "deals"
    assert CP.page_of({"op": "x", "target": "home", "page": "deals"}) == "deals"
    assert CP.page_of("nope") is None


def test_create_page_slugs_are_the_passed_and_the_canonical():
    assert CP.create_page_slugs({"op": "create_page", "title": "Summer Events"}) == ["summer-events"]
    assert CP.create_page_slugs({"op": "create_page", "title": "X", "slug": "summer-events"}) == ["summer-events"]
    assert CP.create_page_slugs({"op": "create_page", "title": "X", "slug": "Summer Events"}) == ["Summer Events", "summer-events"]
    assert CP.create_page_slugs({"op": "create_page", "title": "Deals", "slug": "  "}) == ["deals"]  # blank → the title
    assert CP.create_page_slugs({"op": "create_page", "title": 5}) == []


# --- the same-batch stop -----------------------------------------------------------------------------------------------------

def scripted_engine(refuse: set[str]):
    """A run(ops) stand-in: create_page of a title in `refuse` is E_SLUG, everything else applies; records each batch."""
    calls: list[list[dict]] = []

    def run(ops: list[dict]) -> dict:
        calls.append(copy.deepcopy(ops))
        results = []
        for op in ops:
            if op.get("op") == "create_page" and op.get("title") in refuse:
                results.append({"op": op, "ok": False, "error": {"code": "E_SLUG", "message": "taken", "key": "slug"},
                                "warnings": [], "written": []})
            elif op.get("op") == "create_page":
                results.append({"op": op, "ok": True, "warnings": [], "written": ["customPages[2]"],
                                "slug": CP.slugify(op.get("slug") or op["title"])})
            else:
                results.append({"op": op, "ok": True, "warnings": [], "written": [f"w.{len(results)}"]})
        return {"ok": all(r["ok"] for r in results), "results": results, "changes": [], "exit_code": 0 if all(r["ok"] for r in results) else 2}

    return run, calls


def test_a_refused_create_stops_later_calls_aimed_at_its_slugs():
    ops = [
        {"op": "create_page", "title": "Summer Events", "slug": "Summer Events"},                       # 0 refused (E_SLUG)
        {"op": "edit_list", "target": {"kind": "page", "page": "Summer Events"}, "list": "sections"},  # 1 stopped (passed slug)
        {"op": "set_fields", "target": {"kind": "section", "page": "summer-events", "sectionId": "x"}},  # 2 stopped (canonical)
        {"op": "set_fields", "target": {"kind": "section", "sectionId": "hero"}, "values": {"a": 1}},   # 3 applied (viewed page)
        {"op": "set_page_layout", "page": "summer-events", "layout": "x"},                              # 4 stopped
        {"op": "create_page", "title": "Deals"},                                                        # 5 applied
        {"op": "edit_list", "target": {"kind": "page", "page": "deals"}, "list": "sections"},          # 6 applied
    ]
    run, calls = scripted_engine({"Summer Events"})
    out = CP.run_batch(run, ops)
    assert len(calls) == 2 and calls[1] == [ops[i] for i in (0, 3, 5, 6)]  # the stopped calls never reach the engine
    assert [r["ok"] for r in out["results"]] == [False, False, False, True, False, True, True]
    assert [r["op"] for r in out["results"]] == ops
    assert out["results"][0]["error"]["code"] == "E_SLUG"
    for i, slug in ((1, "Summer Events"), (2, "summer-events"), (4, "summer-events")):
        assert out["results"][i]["error"] == {"code": "E_TARGET", "message": f'page "{slug}" was not created (create_page was '
                                                                              f'refused earlier in this reply), so this op was not applied'}
        assert out["results"][i]["written"] == [] and out["results"][i]["warnings"] == []
    assert out["results"][6]["written"] == ["w.3"]  # indexes of the second run: the stopped calls were not applied
    assert out["ok"] is False and out["exit_code"] == 2


def test_a_call_before_the_refused_create_is_not_stopped_and_no_refusal_means_one_run():
    ops = [{"op": "set_page_layout", "page": "events", "layout": "x"}, {"op": "create_page", "title": "Events"}]
    run, calls = scripted_engine({"Events"})
    out = CP.run_batch(run, ops)
    assert len(calls) == 1 and [r["ok"] for r in out["results"]] == [True, False]
    run, calls = scripted_engine(set())
    ops = [{"op": "create_page", "title": "Deals", "slug": "deals"}, {"op": "edit_list", "target": {"kind": "page", "page": "deals"}}]
    out = CP.run_batch(run, ops)
    assert len(calls) == 1 and out["ok"] is True and out["results"][0]["slug"] == "deals"


def test_a_batch_without_create_page_is_the_single_unchanged_run():
    sentinel = {"ok": True, "results": [], "changes": [], "anything": 1}
    assert CP.run_batch(lambda ops: sentinel, [{"op": "set_fields"}]) is sentinel
    failed = {"ok": False, "results": [], "changes": [], "exit_code": 1, "error": "load"}  # engine failed to load
    assert CP.run_batch(lambda ops: failed, [{"op": "create_page", "title": "x"}]) is failed


# --- render: the slug the model reads -----------------------------------------------------------------------------------------

def test_tool_result_content_forwards_slug_for_create_page_only():
    item = {"toolCallId": "t1", "ok": True, "written": ["customPages[2]", "pageVisibility.deals.inHeaderNav"], "slug": "deals"}
    content, is_error = R.tool_result_content("create_page", item)
    # loop.browserResult: map[string]any → sorted keys
    assert content == '{"ok":true,"slug":"deals","written":["customPages[2]","pageVisibility.deals.inHeaderNav"]}' and not is_error
    content, _ = R.tool_result_content("set_fields", {**item, "sectionId": "hero"})
    assert '"slug"' not in content and content.startswith('{"ok":true,"sectionId":"hero","written"')
    content, is_error = R.tool_result_content("create_page", {"toolCallId": "t1", "ok": False, "slug": "x",
                                                              "error": {"code": "E_SLUG", "message": "m", "key": "slug"}})
    assert content == '{"code":"E_SLUG","message":"m","key":"slug"}' and is_error  # api.ToolError: struct order, no suggestion


# --- the collector -----------------------------------------------------------------------------------------------------------

TINY = {"version": 1, "generated_from": {"at": "2026-10-01T00:00:00Z", "commit": "abc"},
        "sections": {"heroBanner": {"fields": {"title": {"type": "text"}}}}, "globalPanels": {}, "pages": {}, "details": {},
        "branding": {"fields": {"logo": {"type": "image"}}}, "coverage": {"x": 1}}


@pytest.fixture
def svc():
    with FS.FakeService(manifest=TINY, key="k") as s:
        yield s


def _case() -> dict:
    return {"id": "cp1", "kind": "create_page", "target": {"kind": "page", "page": "home"}, "fields": [],
            "site_state": {"template": {"id": "editorial"}, "customPages": [{"slug": "events", "title": "Events"}]},
            "context": {"page": "home", "template": "editorial", "sections": [], "categories": [], "customPages": [], "nav": []},
            "request": "Add a Deals page with a carousel, and an Events page", "expected": {"ops": [], "diff": []},
            "manifest_hash": FS.manifest_hash(TINY), "template": "editorial", "meta": {}}


def browser_engine(state: dict, ops: list[dict], ctx: dict) -> dict:
    """create_page refuses a taken title; ops aimed at a page apply. The shape of headless.mjs's output."""
    state = copy.deepcopy(state)
    results = []
    for op in ops:
        if op.get("op") == "create_page":
            if any(p.get("title") == op.get("title") for p in state.get("customPages") or []):
                results.append({"op": op, "ok": False, "warnings": [], "written": [],
                                "error": {"code": "E_SLUG", "message": "a page \"Events\" already exists at /events",
                                          "key": "title", "suggestion": "events-2"}})
                continue
            slug = op.get("slug") or CP.slugify(op["title"])
            state.setdefault("customPages", []).append({"slug": slug, "title": op["title"]})
            results.append({"op": op, "ok": True, "warnings": [], "written": [f"customPages[{len(state['customPages']) - 1}]"],
                            "slug": slug})
        else:
            page = CP.page_of(op)
            state.setdefault("touched", []).append(page)
            results.append({"op": op, "ok": True, "warnings": [], "written": [f"pageSectionsAll.{page}"]})
    return {"ok": all(r["ok"] for r in results), "results": results, "changes": [], "state": state}


def test_the_collector_posts_the_slug_and_stops_calls_aimed_at_a_refused_create(svc):
    add = {"target": {"kind": "page", "page": "deals"}, "list": "sections", "action": "add", "item": {"id": "productCarousel"}}
    events_add = {"target": {"kind": "page", "page": "events"}, "list": "sections", "action": "add", "item": {"id": "heroBanner"}}
    svc.scripts["cp1"] = [[("tool.call", {"id": "t1", "name": "create_page", "input": {"title": "Deals", "slug": "deals"}}),
                           ("tool.call", {"id": "t2", "name": "edit_list", "input": add}),
                           ("tool.call", {"id": "t3", "name": "create_page", "input": {"title": "Events"}}),
                           ("tool.call", {"id": "t4", "name": "edit_list", "input": events_add}),
                           ("turn.waiting", {"toolCallIds": ["t1", "t2", "t3", "t4"]})],
                          [("text.delta", {"text": "Added Deals."}),
                           ("turn.done", {"usage": {"input_tokens": 1, "output_tokens": 1}, "stop": "end_turn", "conversationId": "c"})]]
    client = C.Client(svc.url, "k", timeout_s=10)
    row = C.run_case(_case(), client, browser_engine)
    assert row["status"] == "ok"
    posted = svc.tool_results[0]["body"]["results"]
    assert [p["toolCallId"] for p in posted] == ["t1", "t2", "t3", "t4"]
    assert posted[0] == {"toolCallId": "t1", "ok": True, "written": ["customPages[1]"], "slug": "deals"}  # turns.js toToolResult
    assert posted[1]["ok"] is True and "slug" not in posted[1]
    assert posted[2]["ok"] is False and posted[2]["error"]["code"] == "E_SLUG"
    assert posted[3] == {"toolCallId": "t4", "ok": False, "error": {
        "code": "E_TARGET", "message": 'page "events" was not created (create_page was refused earlier in this reply), so this op was not applied'}}
    # the stopped call never reached the engine: the existing Events page was not touched
    assert row["engine_results"][0]["ok"] is False
    blocks = row["messages"][2]["content"]
    assert blocks[0]["content"] == '{"ok":true,"slug":"deals","written":["customPages[1]"]}'
    assert blocks[3]["is_error"] is True and json.loads(blocks[3]["content"])["code"] == "E_TARGET"
    from souschef_lab import dataset as D
    assert D.rebuild_messages(row, _case()) == row["messages"]  # the trace (tool.result with slug) rebuilds the same bytes


# --- the engine ctx ----------------------------------------------------------------------------------------------------------

def test_ctx_of_carries_the_hosts_page_slugs_and_option_lists():
    fx = S.load_context_fixture()
    case = {"template": "editorial", "context": {"page": "home"}}
    ctx = V.ctx_of(case)
    assert ctx["templateId"] == "editorial" and ctx["pageSlugs"] == {}
    assert ctx["tags"] == [{"id": t["id"], "slug": t["slug"]} for t in fx["tags"]]
    assert ctx["brands"] == [{"id": b["id"], "slug": b["slug"]} for b in fx["brands"]]
    assert all(isinstance(t["id"], int) for t in ctx["tags"] + ctx["brands"])  # the fixture lists carry catalog ids
    assert C.ctx_for(case) == ctx  # the collector's engine runs with the replay's very ctx
    ctx = V.ctx_of({**case, "page_slugs": {"products": "shop"},
                    "option_lists": {"tags": [{"id": 1, "slug": "a", "visible": False}, {"slug": "b", "name": "B"}, {"id": 3, "slug": ""}],
                                     "brands": None}})
    assert ctx["pageSlugs"] == {"products": "shop"}
    assert ctx["tags"] == [{"slug": "b"}]  # visible:false dropped, an absent id stays absent, no slug → dropped
    assert ctx["brands"] == []             # an unknown list: the host's ctx has []
    assert V.ctx_of({"site_state": {}})["templateId"] == V.DEFAULT_TEMPLATE == "editorial"  # basic: creates allowed
    assert "editorial" in S.BASIC_TEMPLATE_IDS and "default" not in S.BASIC_TEMPLATE_IDS


def test_option_lists_of_prefers_the_case_and_keeps_unknown_apart_from_empty():
    fx = {"tags": [{"slug": "x"}], "brands": [{"slug": "y", "name": "Y"}]}
    assert S.option_lists_of({}, fx) == fx
    assert S.option_lists_of({"option_lists": {"tags": []}}, fx) == {"tags": [], "brands": None}
    assert S.host_lists({"option_lists": {"tags": []}}, fx) == {"tags": [], "brands": []}


# --- node: the real engine (launcher 1d4193e2+) --------------------------------------------------------------------------------

def _create_page_engine():
    path = E.find_engine()
    if path is None or not E.engine_available():
        msg = "the engine is unavailable: " + E.unavailable_reason()
        if REQUIRE:
            pytest.fail(msg)
        pytest.skip(msg)
    validate = path.parent / "validate.js"
    if "create_page" not in validate.read_text(encoding="utf-8"):
        pytest.skip(f"{path} predates create_page (launcher 1d4193e2)")
    return E.Engine(path)


@pytest.fixture(scope="module")
def real_engine():
    return _create_page_engine()


@pytest.fixture(scope="module")
def lab_state():
    from souschef_lab.manifest import load
    return S.build_state(load(FIXTURES_DIR / "manifest.snapshot.json"), template="editorial")


def test_real_engine_create_then_section_in_one_batch_and_the_stop(real_engine, lab_state):
    """One replay (verify.run_engine → createpage.run_batch): a create + a section add on its slug + in_nav; a create
    whose title is taken (the state's Events page) and a section add aimed at it, which the browser stops; a tag
    carousel write whose tagId comes from the ctx's fixture list."""
    case = {"site_state": lab_state, "template": "editorial", "context": {"page": "home"}}
    carousel = next(s["id"] for s in lab_state["homeSections"] if s["id"].startswith("productCarousel"))
    ops = [{"op": "create_page", "title": "Deals", "slug": "deals", "in_nav": True},
           {"op": "edit_list", "target": {"kind": "page", "page": "deals"}, "list": "sections", "action": "add",
            "item": {"id": "productCarousel"}},
           {"op": "create_page", "title": "Events"},
           {"op": "edit_list", "target": {"kind": "page", "page": "events"}, "list": "sections", "action": "add",
            "item": {"id": "heroBanner"}},
           {"op": "set_fields", "target": {"kind": "section", "page": "home", "sectionId": carousel},
            "values": {"subType": "tag", "tagSlug": "sale"}}]
    run = V.run_engine(real_engine, copy.deepcopy(lab_state), ops, V.ctx_of(case))
    res = run["results"]
    assert [r["ok"] for r in res] == [True, True, False, False, True], [r.get("error") for r in res]
    assert res[0]["slug"] == "deals" and any(w.startswith("customPages[") for w in res[0]["written"])
    assert any("inHeaderNav" in w for w in res[0]["written"])
    assert res[2]["error"]["code"] == "E_SLUG"
    assert res[3]["error"] == {"code": "E_TARGET", "message": CP.not_created_message("events")}
    diff = {d["path"]: d.get("after") for d in V.leaves_of(run)}
    tag_ids = [v for p, v in diff.items() if p.endswith(".tagId")]
    assert tag_ids == [103]  # `sale` in fixtures/context.json — ops/derived.js looks it up in ctx.tags
    assert not any(p.startswith("pageSectionsAll.events") for p in diff)  # the stopped add never applied
    item = C.tool_results_for([{"id": "t1"}, {"id": "t2"}, {"id": "t3"}, {"id": "t4"}, {"id": "t5"}], run)
    assert item[0]["slug"] == "deals" and "slug" not in item[1]
    assert R.tool_result_content("create_page", item[0])[0].startswith('{"ok":true,"slug":"deals","written":["customPages[')


def test_real_engine_refuses_create_page_off_a_basic_template(real_engine, lab_state):
    st = copy.deepcopy(lab_state)
    st["template"] = {"id": "default"}
    ctx = V.ctx_of({"site_state": st, "context": {"page": "home"}})
    assert ctx["templateId"] == "default"  # passed through: the browser host sends the site's template as is
    run = V.run_engine(real_engine, st, [{"op": "create_page", "title": "Deals"}], ctx)
    assert run["results"][0]["ok"] is False and run["results"][0]["error"]["code"] == "E_TARGET"
    assert "can't show custom pages" in run["results"][0]["error"]["message"]
