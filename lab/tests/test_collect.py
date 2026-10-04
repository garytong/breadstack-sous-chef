"""collect.py (the browser emulator) and cli.py against the fake service.

No network, no GPU, no API spend: every turn is a script the fake service
replays; the engine is a pure-Python stand-in unless $ENGINE names the real
headless.mjs (then one test runs it; skipped otherwise, failed instead of
skipped when SOUSCHEF_REQUIRE_ENGINE=1 — the CI rule of docs/LAB.md §9).
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
from souschef_lab import cli  # noqa: E402
from souschef_lab import collect as C  # noqa: E402

SNAPSHOT = LAB.parent / "service" / "testdata" / "manifest.snapshot.json"
SNAPSHOT_HASH = "727d151b5d6b04cd68b73199ab0b3bdf061263f9623f4f6eda00a79137c06f38"
USAGE = {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 50, "cache_creation_input_tokens": 0, "calls": 1}
TINY = {"version": 1, "generated_from": {"at": "2026-10-01T00:00:00Z", "commit": "abc"},
        "sections": {"heroBanner": {"fields": {"title": {"type": "text"}, "show": {"type": "bool"}},
                                    "lists": {"slides": {"item": {"layout": {"type": "enum"}}, "ops": ["add", "remove"]}}}},
        "globalPanels": {"HeaderSection": {"fields": {"sticky": {"type": "bool"}}}}, "pages": {}, "details": {},
        "branding": {"fields": {"logo": {"type": "image"}}}, "coverage": {"x": 1}}
TINY_HASH = FS.manifest_hash(TINY)
ASSETS = [{"public_url": "https://cdn.test/a.jpg", "alt": "A", "srcset": "https://cdn.test/a.jpg 1x", "type": "image"},
          {"public_url": "https://cdn.test/b.jpg", "alt": "B", "type": "image"}]


# --- helpers -----------------------------------------------------------------------------------------------------------------

def make_case(cid: str = "c1", holdout: bool = False, kind: str = "set_field", **over) -> dict:
    case = {"id": cid, "kind": kind, "target": {"kind": "section", "page": "home", "sectionId": "heroBanner-1"},
            "fields": ["title"], "site_state": {"template": {"id": "editorial"}, "values": {"title": "Old"}},
            "context": {"page": "home", "template": "editorial", "layout": "default",
                        "sections": [{"id": "heroBanner-1", "type": "heroBanner", "title": "Hero", "enabled": True}],
                        "categories": [], "customPages": [], "nav": []},
            "request": "Set the hero title to Hi", "expected": {"ops": [], "diff": []}, "holdout": holdout,
            "manifest_hash": TINY_HASH, "template": "editorial",
            "meta": {"section_type": "heroBanner", "field_type": "text", "generator": "test", "seed": 1}}
    case.update(over)
    return case


def fake_engine(state: dict, ops: list[dict], ctx: dict) -> dict:
    """A pure-Python engine: set_fields writes values.<key>, edit_list add
    appends, anything else is E_OP. Returns headless.mjs's shape with state."""
    state = copy.deepcopy(state)
    results = []
    for op in ops:
        if op.get("op") == "set_fields":
            written = []
            for k, v in (op.get("values") or {}).items():
                state.setdefault("values", {})[k] = v
                written.append(f"values.{k}")
            results.append({"op": "set_fields", "ok": True, "warnings": [], "written": written})
        elif op.get("op") == "edit_list" and op.get("action") == "add":
            lst = state.setdefault("lists", {}).setdefault(op.get("list"), [])
            lst.append(op.get("item"))
            results.append({"op": "edit_list", "ok": True, "warnings": ["W_HIDDEN"], "written": [f"{op.get('list')}[{len(lst) - 1}]"],
                            "sectionId": f"{op.get('list')}-new"})
        else:
            results.append({"op": op.get("op"), "ok": False, "warnings": [], "written": [],
                            "error": {"code": "E_OP", "message": f"unknown op {op.get('op')}", "key": op.get("op")}})
    state["_version"] = int(state.get("_version", 0)) + 1
    return {"ok": all(r["ok"] for r in results), "results": results, "changes": [], "state": state}


class RecordingEngine:
    def __init__(self):
        self.calls: list[tuple[dict, list[dict], dict]] = []

    def __call__(self, state, ops, ctx):
        self.calls.append((copy.deepcopy(state), ops, ctx))
        return fake_engine(state, ops, ctx)


class RecordingVerifier:
    def __init__(self, verdict="pass", reasons=()):
        self.verdict, self.reasons, self.rows = verdict, list(reasons), []

    def __call__(self, case, row):
        self.rows.append(row)
        return {"verdict": self.verdict, "reasons": self.reasons, "actual_diff": [{"path": "values.title", "after": "Hi"}]}


def done(text="Done.", stop="end_turn"):
    return [("text.delta", {"text": text}), ("turn.done", {"usage": USAGE, "stop": stop, "conversationId": "conv-1"})]


def call(cid, name, inp):
    return ("tool.call", {"id": cid, "name": name, "input": inp})


def waiting(*ids):
    return ("turn.waiting", {"toolCallIds": list(ids)})


def suspended(cid):
    return ("turn.suspended", {"reason": "pick_asset", "toolCallId": cid, "purpose": "hero image", "assetType": "image"})


SET_TITLE = {"target": {"kind": "section", "page": "home", "sectionId": "heroBanner-1"}, "values": {"title": "Hi"}}


@pytest.fixture
def svc():
    with FS.FakeService(manifest=TINY, key="k") as s:
        yield s


@pytest.fixture
def client(svc):
    return C.Client(svc.url, "k", timeout_s=10)


# --- hash and registration -----------------------------------------------------------------------------------------------

@pytest.mark.skipif(not SNAPSHOT.exists(), reason="service/testdata/manifest.snapshot.json not checked out")
def test_manifest_hash_vector():
    doc = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert C.manifest_hash(doc) == SNAPSHOT_HASH
    assert FS.manifest_hash(doc) == SNAPSHOT_HASH


@pytest.mark.skipif(not SNAPSHOT.exists(), reason="snapshot not checked out")
def test_fake_counts_match_readme_facts():
    counts = FS.manifest_counts(json.loads(SNAPSHOT.read_text(encoding="utf-8")))
    assert counts["fields"] == 2094 and counts["lists"] == 69
    assert (counts["sections"], counts["globalPanels"], counts["pages"], counts["details"]) == (19, 12, 17, 11)


def test_hash_ignores_generated_from_and_coverage():
    other = copy.deepcopy(TINY)
    other["generated_from"] = {"at": "later"}
    other["coverage"] = {"y": 2}
    assert C.manifest_hash(other) == TINY_HASH
    other["sections"]["heroBanner"]["fields"]["title"]["type"] = "richtext"
    assert C.manifest_hash(other) != TINY_HASH


def test_register_manifest_returns_hash_and_counts(svc):
    fresh = FS.FakeService(manifest=None, key="k")
    with fresh:
        c = C.Client(fresh.url, "k", timeout_s=10)
        first = c.register_manifest(TINY)
        again = c.register_manifest(TINY)
    assert first["hash"] == TINY_HASH == again["hash"]
    assert first["fields"] == 5 and first["lists"] == 1  # 4 container fields + 1 list item field
    assert fresh.manifest_posts == 2


def test_unauthorized_is_a_service_error(svc):
    bad = C.Client(svc.url, "wrong", timeout_s=10)
    with pytest.raises(C.ServiceError) as ei:
        bad.register_manifest(TINY)
    assert ei.value.status == 401 and ei.value.code == "unauthorized"
    assert svc.unauthorized == 1


def test_connect_checks_hash_against_cases(svc, tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(TINY), encoding="utf-8")
    cases = [make_case("a"), make_case("b", manifest_hash=None)]
    client, hsh = C.connect(svc.url, "k", manifest, cases, sse=False, timeout_s=10, build_id="lab")
    assert hsh == TINY_HASH and cases[1]["manifest_hash"] == TINY_HASH
    with pytest.raises(C.ServiceError, match="hash"):
        C.connect(svc.url, "k", manifest, [make_case("c", manifest_hash="deadbeef")], sse=False, timeout_s=10, build_id="lab")


# --- transport ----------------------------------------------------------------------------------------------------------------

def test_parse_sse_skips_heartbeats_and_joins_data_lines():
    text = ("event: heartbeat\ndata: {}\n\n"
            "event: text.delta\ndata: {\"text\": \"a\",\ndata: \"turnId\": \"t\", \"seq\": 1}\n\n"
            ": comment\n"
            "event: turn.done\ndata: {\"stop\":\"end_turn\",\"turnId\":\"t\",\"seq\":2}\n\n")
    events = C.parse_sse(text)
    assert [e["event"] for e in events] == ["text.delta", "turn.done"]
    assert events[0]["data"]["text"] == "a" and events[0]["data"]["seq"] == 1


def test_json_and_sse_modes_yield_the_same_events(svc):
    svc.scripts["c1"] = [done("hello")]
    svc.heartbeats = 2
    body = {"siteId": "lab-c1", "storeId": "lab", "manifestHash": TINY_HASH, "buildId": "lab", "context": {}, "text": "x"}
    j = C.Client(svc.url, "k", timeout_s=10).start_turn(body)
    s = C.Client(svc.url, "k", timeout_s=10, sse=True).start_turn(body)
    strip = lambda evs: [(e["event"], {k: v for k, v in e["data"].items() if k != "turnId"}) for e in evs]  # noqa: E731
    assert strip(j.events) == strip(s.events)
    assert j.turn_id and s.turn_id and j.turn_id != s.turn_id
    assert [e["data"]["seq"] for e in s.events] == [1, 2]


def test_superseded_409_raises_superseded(svc):
    svc.scripts["c1"] = [[call("t1", "set_fields", SET_TITLE), waiting("t1")], FS.SUPERSEDED]
    stream = client_of(svc).start_turn(turn_body("c1"))
    with pytest.raises(C.Superseded):
        client_of(svc).tool_results(stream.turn_id, {"results": [{"toolCallId": "t1", "ok": True}]})


def client_of(svc, **kw) -> C.Client:
    return C.Client(svc.url, "k", timeout_s=10, **kw)


def turn_body(cid: str) -> dict:
    return {"siteId": f"lab-{cid}", "storeId": "lab", "manifestHash": TINY_HASH, "buildId": "lab",
            "context": {"page": "home"}, "text": "x"}


# --- run_case -----------------------------------------------------------------------------------------------------------------

def test_happy_path_row_shape(svc, client):
    svc.scripts["c1"] = [[("text.delta", {"text": "Setting the title."}), call("t1", "set_fields", SET_TITLE), waiting("t1")],
                         done("Done: the hero title is now Hi.")]
    verifier = RecordingVerifier()
    row = C.run_case(make_case(), client, fake_engine, route="fake", assets=ASSETS, verifier=verifier)
    assert row["id"] == "c1#1" and row["case_id"] == "c1" and row["sample"] == 1 and row["model"] == "fake"
    assert row["status"] == "ok" and row["http_status"] == 200 and row["code"] is None and row["error"] is None
    assert row["stop"] == "end_turn" and row["usage"] == USAGE and row["conversation_id"] == "conv-1"
    assert row["turn_id"] and row["batches"] == 1 and row["latency_ms"] >= 0
    assert row["ops"] == [{"op": "set_fields", **SET_TITLE}]
    assert row["written_paths"] == ["values.title"]
    assert row["verdict"] == "pass" and row["reasons"] == [] and row["actual_diff"][0]["path"] == "values.title"
    assert row["tools"] == list(C.TOOL_NAMES)
    assert row["meta"] == {"kind": "set_field", "section": "heroBanner", "template": "editorial", "field_type": "text",
                           "holdout": False, "manifest_hash": TINY_HASH, "target": make_case()["target"], "buildId": "lab",
                           "engine_fingerprint": None}  # the pure-Python stand-in has no identity; the real engine stamps one
    # the stream's events plus one tool.result per posted item (the service trace's shape, README §4), seq-gapped as the service does
    assert [e["event"] for e in row["events"]] == ["text.delta", "tool.call", "turn.waiting", "tool.result", "text.delta", "turn.done"]
    assert [e["data"]["seq"] for e in row["events"]] == [1, 2, 3, 4, 5, 6]
    assert row["events"][3]["data"] == {"toolCallId": "t1", "ok": True, "written": ["values.title"], "for_seq": 2, "seq": 4,
                                        "turnId": row["turn_id"]}
    assert row["context"] == make_case()["context"] and row["text"] == "Set the hero title to Hi"  # what was sent
    # a fresh conversation per sample: the site's open conversation is ended before the turn starts (README §1.1 continues
    # it otherwise — measured against the real binary, LAB.md §4)
    assert svc.conversation_ends == ["lab-c1"] and row["conversation_ended"] is True
    assert verifier.rows[0] is row
    # the turn request and the posted results
    req = svc.turn_requests[0]
    assert req["siteId"] == "lab-c1" and req["storeId"] == "lab" and req["model"] == "fake" and req["buildId"] == "lab"
    assert req["manifestHash"] == TINY_HASH and req["text"] == "Set the hero title to Hi" and req["context"]["page"] == "home"
    assert len(svc.tool_results) == 1
    posted = svc.tool_results[0]["body"]
    assert posted["siteId"] == "lab-c1" and posted["storeId"] == "lab"
    assert posted["results"] == [{"toolCallId": "t1", "ok": True, "written": ["values.title"]}]
    assert svc.tool_results[0]["missing"] == []


def test_fresh_conversation_can_be_kept(svc, client):
    svc.scripts["c1"] = [[call("t1", "set_fields", SET_TITLE), waiting("t1")], done()]
    row = C.run_case(make_case(), client, fake_engine, fresh_conversation=False)
    assert row["status"] == "ok" and svc.conversation_ends == [] and "conversation_ended" not in row


def test_end_conversation_tolerates_a_service_without_the_route(monkeypatch, client):
    def no_route(method, path, body=None, accept=None):
        raise C.ServiceError(404, "not_found", path)
    monkeypatch.setattr(client, "_request", no_route)
    assert client.end_conversation("lab-x") is False
    def boom(method, path, body=None, accept=None):
        raise C.ServiceError(503, "store_unavailable", "down")
    monkeypatch.setattr(client, "_request", boom)
    with pytest.raises(C.ServiceError):
        client.end_conversation("lab-x")


def test_messages_rebuilt_as_anthropic_blocks(svc, client):
    svc.scripts["c1"] = [[("text.delta", {"text": "Working. "}), ("text.delta", {"text": "Now."}),
                          call("t1", "set_fields", SET_TITLE), waiting("t1")], done("Done.")]
    row = C.run_case(make_case(), client, fake_engine)
    m = row["messages"]
    assert [x["role"] for x in m] == ["user", "assistant", "user", "assistant"]
    # prompt.ContextBlock: "<context>\n" + compact JSON + "\n</context>\n" + text (render.py, parity-tested)
    assert m[0]["content"][0]["type"] == "text" and m[0]["content"][0]["text"].startswith("<context>\n{")
    assert m[0]["content"][0]["text"].endswith("\n</context>\nSet the hero title to Hi")
    assert m[1]["content"] == [{"type": "text", "text": "Working. Now."},
                               {"type": "tool_use", "id": "t1", "name": "set_fields", "input": SET_TITLE}]
    tr = m[2]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "t1" and "is_error" not in tr
    assert tr["content"] == '{"ok":true,"written":["values.title"]}'  # loop.browserResult: Go-compact, sorted keys
    assert m[3]["content"] == [{"type": "text", "text": "Done."}]
    assert row["reply"] == "Working. Now.\nDone."
    assert not any(b.get("type") == "thinking" for msg in m for b in msg["content"])
    # the logged messages are a cache of what render.messages_from_events rebuilds from the row's own events + context
    from souschef_lab import dataset as D
    assert D.rebuild_messages(row, make_case()) == m and D.stored_messages_match(row, make_case())


def test_multi_batch_carries_state_between_batches(svc, client):
    second = {"target": SET_TITLE["target"], "values": {"show": True}}
    svc.scripts["c1"] = [[call("t1", "set_fields", SET_TITLE), waiting("t1")],
                         [call("t2", "set_fields", second), waiting("t2")],
                         done()]
    engine = RecordingEngine()
    row = C.run_case(make_case(), client, engine)
    assert row["status"] == "ok" and row["batches"] == 2
    assert len(engine.calls) == 2
    assert engine.calls[0][0]["values"] == {"title": "Old"}  # a fresh copy of the case state
    assert engine.calls[1][0]["values"] == {"title": "Hi"} and engine.calls[1][0]["_version"] == 1
    assert engine.calls[0][2] == {**C.GOLDEN_CTX, "templateId": "editorial", "currentPage": "home"}
    assert [b["body"]["results"][0]["toolCallId"] for b in svc.tool_results] == ["t1", "t2"]
    assert row["written_paths"] == ["values.title", "values.show"]
    assert [e["data"]["seq"] for e in row["events"]] == list(range(1, 9))
    assert [(e["event"], e["data"].get("for_seq")) for e in row["events"] if e["event"] == "tool.result"] == \
        [("tool.result", 1), ("tool.result", 4)]


def test_pick_asset_is_answered_from_the_fixture(svc, client):
    with_url = {"target": SET_TITLE["target"], "values": {"image": "https://cdn.test/a.jpg"}, "assets": {"image": {"alt": "A"}}}
    svc.scripts["c1"] = [[("text.delta", {"text": "Pick an image."}), suspended("p1")],
                         [call("t1", "set_fields", with_url), waiting("t1")], done()]
    row = C.run_case(make_case(), client, fake_engine, assets=ASSETS)
    assert row["status"] == "ok"
    first = svc.tool_results[0]["body"]["results"]
    assert len(first) == 1 and first[0]["toolCallId"] == "p1" and first[0]["ok"] is True
    assert first[0]["asset"]["public_url"] in {a["public_url"] for a in ASSETS}
    assert row["picked_assets"] == [first[0]["asset"]]
    assert row["ops"] == [{"op": "set_fields", **with_url}]
    tr = row["messages"][2]["content"][0]
    # loop.browserResult for pick_asset: the model reads the api.Asset itself ({public_url, alt?, srcset?}), not a wrapper
    assert tr["tool_use_id"] == "p1" and json.loads(tr["content"]) == first[0]["asset"] and "is_error" not in tr
    from souschef_lab import dataset as D
    assert D.rebuild_messages(row, make_case()) == row["messages"]  # the pick_asset tool_use + asset result rebuild too
    pick = [e["data"] for e in row["events"] if e["event"] == "tool.result"][0]
    assert pick["toolCallId"] == "p1" and pick["for_seq"] == 2  # for_seq = the turn.suspended's seq (no tool.call exists)


def test_pick_asset_with_data_ops_in_the_same_batch(svc, client):
    svc.scripts["c1"] = [[call("t1", "set_fields", SET_TITLE), suspended("p1")], done()]
    row = C.run_case(make_case(), client, fake_engine, assets=ASSETS)
    assert row["status"] == "ok"
    results = svc.tool_results[0]["body"]["results"]
    assert [r["toolCallId"] for r in results] == ["t1", "p1"]
    assert results[0]["written"] == ["values.title"] and "asset" in results[1]


def test_no_result_posted_for_report_unsupported(svc, client):
    svc.scripts["c1"] = [[call("u1", "report_unsupported", {"reason": "no custom code"}),
                          call("t1", "set_fields", SET_TITLE), waiting("t1")], done()]
    row = C.run_case(make_case(), client, fake_engine)
    assert row["status"] == "ok"
    assert [r["toolCallId"] for r in svc.tool_results[0]["body"]["results"]] == ["t1"]
    assert row["ops"] == [{"op": "set_fields", **SET_TITLE}]
    assert [c["name"] for c in row["tool_calls"]] == ["report_unsupported", "set_fields"]


def test_report_unsupported_alone_ends_without_a_post(svc, client):
    svc.scripts["c1"] = [[call("u1", "report_unsupported", {"reason": "premium"}), ("text.delta", {"text": "Sorry."}),
                          ("turn.done", {"usage": USAGE, "stop": "end_turn"})]]
    row = C.run_case(make_case(kind="unsupported"), client, fake_engine)
    assert row["status"] == "ok" and row["ops"] == [] and svc.tool_results == []
    assert row["messages"][1]["content"][1]["name"] == "report_unsupported"


def test_superseded_results_are_dropped(svc, client):
    svc.scripts["c1"] = [[call("t1", "set_fields", SET_TITLE), waiting("t1")], FS.SUPERSEDED, done()]
    row = C.run_case(make_case(), client, fake_engine)
    assert row["status"] == "error" and row["code"] == "superseded" and row["http_status"] == 409
    assert len(svc.tool_results) == 1 and svc.tool_results[0]["answer"] == 409
    assert row["ops"] == [{"op": "set_fields", **SET_TITLE}]  # what was applied is still recorded
    assert row["stop"] is None and row["verdict"] is None


def test_turn_error_is_an_error_row(svc, client):
    svc.scripts["c1"] = [[("turn.error", {"code": "model_unavailable", "message": "503 from the route"})]]
    row = C.run_case(make_case(), client, fake_engine)
    assert row["status"] == "error" and row["code"] == "model_unavailable" and "503" in row["error"]


def test_non_end_turn_stop_is_recorded(svc, client):
    svc.scripts["c1"] = [done("(cut short)", stop="max_tokens")]
    row = C.run_case(make_case(), client, fake_engine)
    assert row["status"] == "ok" and row["stop"] == "max_tokens" and row["code"] == "max_tokens"


def test_batch_cap_stops_a_runaway_turn(svc, client):
    svc.scripts["c1"] = [[call(f"t{i}", "set_fields", SET_TITLE), waiting(f"t{i}")] for i in range(12)] + [done()]
    row = C.run_case(make_case(), client, fake_engine, max_batches=C.MAX_BATCHES)
    assert row["status"] == "error" and row["code"] == "max_batches"
    assert len(svc.tool_results) == C.MAX_BATCHES == 8 and row["batches"] == 9


def test_413_body_too_large(svc):
    small = FS.FakeService(manifest=TINY, key="k", max_body_bytes=120)
    with small:
        row = C.run_case(make_case(), C.Client(small.url, "k", timeout_s=10), fake_engine)
    assert row["status"] == "error" and row["http_status"] == 413 and row["ops"] == []


def test_400_when_a_required_field_is_missing(svc, client):
    row = C.run_case(make_case(manifest_hash=None), client, fake_engine)
    assert row["status"] == "error" and row["http_status"] == 400 and row["code"] == "bad_request"
    assert "manifestHash" in row["error"]


def test_409_unknown_manifest(svc, client):
    row = C.run_case(make_case(manifest_hash="f" * 64), client, fake_engine)
    assert row["status"] == "error" and row["http_status"] == 409 and row["code"] == "unknown_manifest"


def test_engine_failures_are_posted_as_errors(svc, client):
    bad = {"target": SET_TITLE["target"], "list": "slides", "action": "explode"}
    svc.scripts["c1"] = [[call("t1", "edit_list", bad), call("t2", "set_fields", SET_TITLE), waiting("t1", "t2")], done()]
    verifier = RecordingVerifier(verdict="invalid_op", reasons=["invalid_op"])
    row = C.run_case(make_case(), client, fake_engine, verifier=verifier)
    posted = svc.tool_results[0]["body"]["results"]
    assert posted[0] == {"toolCallId": "t1", "ok": False, "error": {"code": "E_OP", "message": "unknown op edit_list", "key": "edit_list"}}
    assert posted[1]["ok"] is True
    assert row["status"] == "ok" and row["verdict"] == "invalid_op"
    assert row["messages"][2]["content"][0]["is_error"] is True
    assert row["engine_results"][0]["ok"] is False and "state" not in row["engine_results"][0]


def test_tool_results_for_maps_warnings_written_and_claims():
    calls = [{"id": "a", "name": "edit_list"}, {"id": "b", "name": "set_fields"}, {"id": "c", "name": "reset"}]
    eng = {"results": [{"ok": True, "warnings": ["W_HIDDEN"], "written": ["x"]},
                       {"ok": False, "error": "nope"}],
           "claims": {"a": "sec-9"}}
    out = C.tool_results_for(calls, eng)
    assert out[0] == {"toolCallId": "a", "ok": True, "warnings": ["W_HIDDEN"], "written": ["x"], "sectionId": "sec-9"}
    assert out[1] == {"toolCallId": "b", "ok": False, "error": {"code": "E_OP", "message": "nope"}}
    assert out[2]["ok"] is False and out[2]["error"]["code"] == "E_ENGINE"


def test_pick_asset_for_is_deterministic_and_typed():
    s = {"toolCallId": "p1", "assetType": "image"}
    a = C.pick_asset_for(ASSETS, s, "c1")
    assert a == C.pick_asset_for(ASSETS, s, "c1") and set(a) <= {"public_url", "alt", "srcset"}
    video = [{"public_url": "v.mp4", "type": "video"}] + ASSETS
    assert C.pick_asset_for(video, {"toolCallId": "p1", "assetType": "video"}, "c1")["public_url"] == "v.mp4"
    with pytest.raises(ValueError):
        C.pick_asset_for([], s, "c1")


def test_ctx_for_and_user_text():
    assert C.ctx_for(make_case()) == {**C.GOLDEN_CTX, "templateId": "editorial", "currentPage": "home"}
    assert C.ctx_for({"site_state": {}, "context": {}})["templateId"] == "editorial"
    assert C.user_text({"page": "home"}, "hi") == '<context>\n{"page":"home"}\n</context>\nhi'


def test_verdict_fields_from_object_and_dict():
    class V:
        verdict, reasons, warnings = "wrong_value", ["wrong_value"], ["W_HIDDEN"]
    assert C._verdict_fields(V()) == {"verdict": "wrong_value", "reasons": ["wrong_value"], "warnings": ["W_HIDDEN"]}
    assert C._verdict_fields({"verdict": "pass", "reasons": None}) == {"verdict": "pass", "reasons": []}
    assert C._verdict_fields(None) == {"verdict": None, "reasons": None}


# --- the log and resume ----------------------------------------------------------------------------------------------------

def test_append_row_and_answered_ids_tolerate_a_torn_line(tmp_path):
    log = tmp_path / "runs" / "log.jsonl"
    C.append_row(log, {"id": "a#1", "status": "ok"})
    C.append_row(log, {"id": "b#1", "status": "error"})
    with log.open("a") as f:
        f.write('{"id": "c#1", "sta')  # a crash mid-write
    assert C.answered_ids(log) == {"a#1"}
    assert len(C.read_jsonl(log)) == 2
    assert C.runs_log(log) == tmp_path / "runs" / "log-runs.jsonl"


def test_collect_resumes_and_records_settings(svc, client, tmp_path):
    svc.scripts["a"] = [done()]
    svc.scripts["b"] = [done()]
    log = tmp_path / "log.jsonl"
    C.append_row(log, {"id": "a#1", "case_id": "a", "sample": 1, "status": "ok", "verdict": "pass"})
    C.append_row(log, {"id": "b#1", "case_id": "b", "sample": 1, "status": "error"})
    counts = C.collect([make_case("a"), make_case("b")], client, fake_engine, log, route="fake",
                       verifier=RecordingVerifier(), settings={"note": "t"})
    assert counts["asked"] == 1 and counts["ok"] == 1
    assert [r["siteId"] for r in svc.turn_requests] == ["lab-b"]
    rows = C.read_jsonl(log)
    assert [r["id"] for r in rows] == ["a#1", "b#1", "b#1"] and rows[-1]["status"] == "ok"
    runs = C.read_jsonl(C.runs_log(log))
    assert len(runs) == 1 and runs[0]["samples"] == 1 and runs[0]["note"] == "t" and runs[0]["already_logged"] == 1
    assert counts["states"] == {"done": 1}
    # a second run has nothing to do and writes no settings record
    again = C.collect([make_case("a"), make_case("b")], client, fake_engine, log, verifier=RecordingVerifier())
    assert again["asked"] == 0 and len(C.read_jsonl(C.runs_log(log))) == 1


def test_collect_best_of_n_rounds_with_adaptive_enough(svc, client, tmp_path):
    for cid in ("a", "b", "c"):
        svc.scripts[cid] = [done()]
    log = tmp_path / "log.jsonl"
    cases = [make_case("a"), make_case("b"), make_case("c")]
    counts = C.collect(cases, client, fake_engine, log, samples=4, adaptive=C.Adaptive(enough=2), workers=2,
                       verifier=RecordingVerifier("pass"))
    rows = C.read_jsonl(log)
    assert counts["asked"] == 6 and counts["passed"] == 6
    assert sorted(r["id"] for r in rows) == sorted(f"{c}#{k}" for c in "abc" for k in (1, 2))
    # rounds: every case's first sample before any second
    assert [r["sample"] for r in rows] == [1, 1, 1, 2, 2, 2]
    assert counts["states"] == {"enough": 3}
    assert C.read_jsonl(C.runs_log(log))[0]["adaptive"] == {"enough": 2, "hopeless_after": 2, "family_min": 8,
                                                              "family_rate": 0.125}


def test_collect_without_adaptive_asks_every_sample_and_max_cases(svc, client, tmp_path):
    for cid in ("a", "b"):
        svc.scripts[cid] = [done()]
    log = tmp_path / "log.jsonl"
    counts = C.collect([make_case("a"), make_case("b")], client, fake_engine, log, samples=2, max_cases=1,
                       verifier=RecordingVerifier("missing_writes", ["missing_writes"]))
    assert counts["asked"] == 2 and counts["passed"] == 0
    assert {r["id"] for r in C.read_jsonl(log)} == {"a#1", "a#2"}
    with pytest.raises(ValueError):
        C.collect([], client, fake_engine, log, workers=5)


def test_sample_state_table():
    rule = C.Adaptive(enough=2)
    assert C.sample_state([True, True, False, False], 4, (0, 0), None) == "done"
    assert C.sample_state([True], 4, (0, 0), None) == "open"
    assert C.sample_state([True, True], 4, (0, 0), rule) == "enough"
    assert C.sample_state([False, False], 4, (0, 8), rule) == "hopeless"
    assert C.sample_state([False, False], 4, (2, 8), rule) == "open"  # the family can do it
    assert C.sample_state([False], 4, (0, 8), rule) == "open"  # not yet ⌈N/2⌉ failures
    assert C.sample_state([False], 4, (0, 8), rule, excluded=True) == "excluded"
    assert C.Adaptive(hopeless_after=1).after(4) == 1 and rule.after(5) == 3


def test_excluded_holdout_cases_stop_sampling(svc, client, tmp_path):
    svc.scripts["h"] = [done()]
    log = tmp_path / "log.jsonl"
    counts = C.collect([make_case("h", holdout=True)], client, fake_engine, log, samples=3, adaptive=C.Adaptive(),
                       verifier=RecordingVerifier("pass", ["holdout"]))
    assert counts["asked"] == 1 and counts["states"] == {"excluded": 1}


def test_load_cases_filters(tmp_path):
    p = tmp_path / "cases.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in [make_case("a"), make_case("b", kind="toggle"), make_case("h", holdout=True)]),
                 encoding="utf-8")
    assert [c["id"] for c in C.load_cases(p)] == ["a", "b", "h"]
    assert [c["id"] for c in C.load_cases(p, only="toggle,h")] == ["b", "h"]
    assert [c["id"] for c in C.load_cases(p, skip_holdout=True)] == ["a", "b"]


def test_load_assets(tmp_path):
    p = tmp_path / "assets.json"
    p.write_text(json.dumps({"assets": ASSETS + [{"alt": "no url"}]}), encoding="utf-8")
    assert C.load_assets(p) == ASSETS
    assert C.load_assets(tmp_path / "missing.json") == []


# --- the CLI -------------------------------------------------------------------------------------------------------------------

def test_cli_lists_every_command():
    for name in ("cases", "collect", "verify", "build", "bestofn", "pairs", "eval", "passk", "retrain-check", "train", "dpo",
                 "export"):
        assert name in cli.COMMANDS


def test_cli_dispatches_and_passes_args(tmp_path, capsys):
    assert cli.main([]) == 2
    assert cli.main(["--help"]) == 0
    with pytest.raises(SystemExit):
        cli.main(["no-such-command"])
    # collect's own validation runs with the remaining args
    assert cli.main(["collect", "--samples", "0", "--key", "k"]) == 2
    assert "--samples" in capsys.readouterr().err
    assert cli.main(["retrain-check", "--now", "2026-10-01T00:00:00Z"]) == 2  # nothing given → insufficient data


def test_cli_reports_a_missing_module(monkeypatch, capsys):
    monkeypatch.setitem(cli.COMMANDS, "ghost", ("no_such_module_xyz", "n/a"))
    monkeypatch.setattr(cli, "build_parser", lambda: _parser_with("ghost"))
    assert cli.main(["ghost"]) == 2
    assert "not available" in capsys.readouterr().err


def _parser_with(name):
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    sub.add_parser(name, add_help=False).add_argument("args", nargs=argparse.REMAINDER)
    return p


def test_collect_main_end_to_end(svc, tmp_path, capsys):
    svc.scripts["a"] = [[call("t1", "set_fields", SET_TITLE), waiting("t1")], done()]
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(make_case("a")) + "\n", encoding="utf-8")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(TINY), encoding="utf-8")
    assets = tmp_path / "assets.json"
    assets.write_text(json.dumps(ASSETS), encoding="utf-8")
    log = tmp_path / "log.jsonl"
    # the real engine/verifier modules are another agent's: stand-ins for the CLI path
    import souschef_lab.collect as mod
    orig_engine, orig_verifier = mod.default_engine, mod.default_verifier
    mod.default_engine = lambda *a, **k: fake_engine
    mod.default_verifier = lambda *a, **k: RecordingVerifier("pass")
    try:
        rc = cli.main(["collect", "--cases", str(cases), "--log", str(log), "--url", svc.url, "--key", "k", "--manifest",
                       str(manifest), "--assets", str(assets), "--route", "fake", "--sse"])
    finally:
        mod.default_engine, mod.default_verifier = orig_engine, orig_verifier
    assert rc == 0
    rows = C.read_jsonl(log)
    assert len(rows) == 1 and rows[0]["status"] == "ok" and rows[0]["verdict"] == "pass" and rows[0]["model"] == "fake"
    assert "asked 1: 1 answered" in capsys.readouterr().out
    assert cli.main(["collect", "--cases", str(tmp_path / "none.jsonl"), "--key", "k"]) == 2
    assert cli.main(["collect", "--cases", str(cases), "--key", "k", "--url", "http://127.0.0.1:9", "--manifest",
                     str(manifest)]) == 1


# --- the real engine (optional) -------------------------------------------------------------------------------------------

def _engine_reason() -> str | None:
    """Why the real engine cannot run here (None when it can): the same rule
    as engine.py's unavailable_reason, with node_modules checked too."""
    from souschef_lab.engine import find_engine  # $ENGINE, else $SOUS_CHEF_ENGINE / $COOKIE_CUTTER_DIR / the sibling checkout
    script = find_engine(os.environ.get("ENGINE"))
    if not script or not Path(script).exists():
        return "ENGINE (headless.mjs) not set or missing"
    node = shutil.which("node")
    if not node:
        return "node not installed"
    try:
        major = int(subprocess.run([node, "--version"], capture_output=True, text=True, timeout=20).stdout.strip().lstrip("v").split(".")[0])
    except (ValueError, subprocess.SubprocessError):
        return "node version unreadable"
    if major < 20:
        return f"node {major} < 20"
    client_root = Path(script).resolve().parents[3]  # launcher/client; a workspace may hoist to the repo root
    if not ((client_root / "node_modules").is_dir() or (client_root.parent.parent / "node_modules").is_dir()):
        return "launcher/client/node_modules not installed"
    try:
        import souschef_lab.engine  # noqa: F401
    except ImportError as e:
        return f"souschef_lab.engine not importable: {e}"
    return None


def test_dict_engine_wraps_results_and_engine_errors():
    class Res:
        def to_dict(self):
            return {"ok": True, "results": [{"ok": True}], "changes": [], "state": {"v": 1}}

    class Inner:
        def run(self, state, ops, ctx, print_state=False):
            if ops and ops[0].get("op") == "boom":
                raise RuntimeError("engine exit 1: vite failed")
            assert print_state is True
            return Res()

    eng = C.DictEngine(Inner())
    assert eng({}, [{"op": "set_fields"}], {}) == {"ok": True, "results": [{"ok": True}], "changes": [], "state": {"v": 1}}
    failed = eng.run({}, [{"op": "boom"}], {})
    assert failed["ok"] is False and failed["exit_code"] == 1 and "vite failed" in failed["error"]
    assert C.tool_results_for([{"id": "t1"}], failed)[0]["error"]["code"] == "E_ENGINE"


@pytest.mark.engine
def test_real_engine_round_trip_if_available(svc, client):
    reason = _engine_reason()
    if reason:
        if os.environ.get("SOUS_CHEF_LAB_REQUIRE_ENGINE") == "1" or os.environ.get("SOUSCHEF_REQUIRE_ENGINE") == "1":
            pytest.fail(f"engine required (SOUS_CHEF_LAB_REQUIRE_ENGINE=1) but unavailable: {reason}")
        pytest.skip(reason)
    from souschef_lab.engine import find_engine
    engine = C.default_engine(find_engine(os.environ.get("ENGINE")), os.environ.get("SOUSCHEF_MANIFEST") or SNAPSHOT)
    assert engine.fingerprint and len(engine.fingerprint) == 64  # the real engine identifies itself
    state = {"template": {"id": "editorial"}, "homeSections": [{"id": "heroBanner-1", "settings": {}}]}
    svc.scripts["r1"] = [[call("t1", "set_fields", {"target": {"kind": "section", "page": "home", "sectionId": "heroBanner-1"},
                                                     "values": {"paddingY": 24}}), waiting("t1")], done()]
    row = C.run_case(make_case("r1", site_state=state), client, engine)
    assert row["status"] == "ok" and svc.tool_results[0]["body"]["results"][0]["ok"] is True
    assert row["meta"]["engine_fingerprint"] == engine.fingerprint
