"""render.py: the lab's prompt bytes equal the service's (docs/LAB.md §4
"Prompt parity").

The parity tests rebuild every captured real-binary run
(fixtures/prompt/real/<capture>/ — the POST /v1/messages bodies the service
sent, captured by tests/fake_model.py through tests/capture_real.py: the
ok/written, error, warnings + sectionId and pick_asset shapes) from the case
and the collector row of that run, and compare every text, tool_use and
tool_result payload byte for byte. lab/data/e2e/real/ (gitignored) is
checked too when present.
"""

import json
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from souschef_lab import collect as C  # noqa: E402
from souschef_lab import dataset as D  # noqa: E402
from souschef_lab import render as R  # noqa: E402
from souschef_lab import state as S  # noqa: E402

FIX = LAB / "fixtures" / "prompt" / "real"
DATA = LAB / "data" / "e2e" / "real"
CAPTURES = sorted(p for root in (FIX, DATA) if root.is_dir() for p in root.iterdir()
                  if p.is_dir() and all((p / f).is_file() for f in ("case.json", "collect-real.jsonl", "model-requests.jsonl")))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- Go encoding/json ---------------------------------------------------------------------------------------------------

def test_compact_json_keeps_order_and_never_escapes():
    assert R.compact_json({"b": 1, "a": [1, {"z": "é<&>"}]}) == '{"b":1,"a":[1,{"z":"é<&>"}]}'


def test_go_marshal_sorts_map_keys_and_html_escapes():
    assert R.go_marshal({"written": ["x"], "ok": True, "sectionId": "s", "warnings": ["W"]}) == \
        '{"ok":true,"sectionId":"s","warnings":["W"],"written":["x"]}'
    assert R.go_marshal({"message": "a<b & c>d"}) == '{"message":"a\\u003cb \\u0026 c\\u003ed"}'
    assert R.go_marshal({"m": "l s"}) == '{"m":"l\\u2028s"}'
    assert R.go_marshal({"b": 1, "a": 2}, sort_keys=False) == '{"b":1,"a":2}'  # a struct keeps its field order
    assert R.go_marshal({"s": "ünïcode"}) == '{"s":"ünïcode"}'  # UTF-8 as is


# --- the user block -------------------------------------------------------------------------------------------------------

def test_context_block_matches_prompt_context_block():
    assert R.context_block({"page": "home"}, "  hi \n") == '<context>\n{"page":"home"}\n</context>\nhi'
    assert R.context_block(None, "hi") == "hi"  # no context → bare text
    assert R.context_block({}, "hi") == "<context>\n{}\n</context>\nhi"  # an empty object is still an object
    assert C.user_text({"page": "home"}, "hi") == R.context_block({"page": "home"}, "hi")
    assert D.context_block is R.context_block and D.messages_from_events is R.messages_from_events


# --- tool results -------------------------------------------------------------------------------------------------------------

def test_browser_result_ok_shape():
    assert R.tool_result_content("set_fields", {"toolCallId": "t", "ok": True, "written": ["a.b"]}) == \
        ('{"ok":true,"written":["a.b"]}', False)
    # empty lists are omitted (len(res.Written) > 0), keys sorted as Go marshals a map
    assert R.tool_result_content("edit_list", {"toolCallId": "t", "ok": True, "warnings": [], "written": [], "sectionId": "x-1"}) == \
        ('{"ok":true,"sectionId":"x-1"}', False)
    assert R.tool_result_content("reset", {"toolCallId": "t", "ok": True, "warnings": ["W_HIDDEN:k"], "written": ["p"]}) == \
        ('{"ok":true,"warnings":["W_HIDDEN:k"],"written":["p"]}', False)


def test_browser_result_hidden_shape():
    """loop/hidden_test.go TestBrowserResultForwardsHidden: the same bytes."""
    item = {"toolCallId": "t1", "ok": True, "written": ["maxWidth"], "warnings": ["W_HIDDEN:maxWidth"],
            "hidden": [{"key": "maxWidth", "by": "slides[0].layout", "byValue": "split"}, {"by": "x"}]}
    assert R.tool_result_content("set_fields", item) == (
        '{"hidden":[{"by":"slides[0].layout","byValue":"split","key":"maxWidth"}],"ok":true,'
        '"warnings":["W_HIDDEN:maxWidth"],"written":["maxWidth"]}', False)
    assert R.tool_result_content("set_fields", {"toolCallId": "t1", "ok": True, "hidden": "nope"}) == ('{"ok":true}', False)
    assert R.tool_result_content("set_fields", {"toolCallId": "t1", "ok": True, "hidden": [{"key": "k", "by": "b"}]}) == \
        ('{"hidden":[{"by":"b","byValue":null,"key":"k"}],"ok":true}', False)


def test_collector_posts_the_browser_hidden_member():
    """client/turns.js wireHidden: {key, by, byValue} only, only when non-empty, only on ok."""
    calls = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    engine = {"results": [
        {"ok": True, "warnings": ["W_HIDDEN:maxWidth"], "written": ["p"],
         "hidden": [{"key": "maxWidth", "by": "slides[0].layout", "byValue": "split", "rule": "conditional_on", "gate": {}}]},
        {"ok": True, "hidden": []},
        {"ok": False, "error": {"code": "E_FIELD", "message": "m"}, "hidden": [{"key": "k", "by": "b", "byValue": 1}]},
    ]}
    items = C.tool_results_for(calls, engine)
    assert items[0]["hidden"] == [{"key": "maxWidth", "by": "slides[0].layout", "byValue": "split"}]
    assert "hidden" not in items[1] and "hidden" not in items[2]


def test_browser_result_error_shape():
    assert R.tool_result_content("set_fields", {"toolCallId": "t", "ok": False,
                                                "error": {"code": "E_FIELD", "message": "no such field <x>", "key": "x"}}) == \
        ('{"code":"E_FIELD","message":"no such field \\u003cx\\u003e","key":"x"}', True)
    assert R.tool_result_content("set_fields", {"toolCallId": "t", "ok": False, "error": {"code": "E_OP", "message": "m"}}) == \
        ('{"code":"E_OP","message":"m"}', True)
    assert R.tool_result_content("set_fields", {"toolCallId": "t", "ok": False}) == \
        ('{"code":"E_OP","message":"the browser reported a failure without details"}', True)


def test_pick_asset_result_shape():
    item = {"toolCallId": "p", "ok": True, "asset": {"public_url": "https://cdn/x.jpg", "alt": "A", "srcset": {"400": "https://cdn/x-400.jpg"}}}
    assert R.tool_result_content("pick_asset", item) == \
        ('{"public_url":"https://cdn/x.jpg","alt":"A","srcset":{"400":"https://cdn/x-400.jpg"}}', False)
    assert R.tool_result_content("pick_asset", {"toolCallId": "p", "ok": True, "asset": {"public_url": "u", "srcset": None}}) == \
        ('{"public_url":"u"}', False)
    assert R.tool_result_content("pick_asset", {"toolCallId": "p", "ok": True}) == \
        ('{"code":"E_ASSET","message":"no asset was chosen"}', True)
    block = R.tool_result_block("pick_asset", {"toolCallId": "p", "ok": True})
    assert block == {"type": "tool_result", "tool_use_id": "p", "content": '{"code":"E_ASSET","message":"no asset was chosen"}',
                     "is_error": True}
    assert "is_error" not in R.tool_result_block("set_fields", {"toolCallId": "t", "ok": True})


def test_pick_asset_call_rebuilds_both_required_inputs():
    """The tool schema requires purpose AND asset_type (strict), so the model always sent both; the service stores the
    input re-marshalled (sorted keys) — the shape the captured pick_asset history shows."""
    call = R.pick_asset_call({"reason": "pick_asset", "toolCallId": "p1", "purpose": "hero image", "assetType": "image"})
    assert call == {"type": "tool_use", "id": "p1", "name": "pick_asset", "input": {"asset_type": "image", "purpose": "hero image"}}
    assert list(call["input"]) == ["asset_type", "purpose"]
    assert R.pick_asset_call({"toolCallId": "p2"})["input"] == {"asset_type": "image", "purpose": ""}  # the event's defaults


def test_tool_results_message_keeps_call_order_in_one_message():
    names = {"a": "set_fields", "b": "pick_asset"}
    msg = R.tool_results_message([{"toolCallId": "b", "ok": True, "asset": {"public_url": "u"}},
                                  {"toolCallId": "a", "ok": True, "written": ["p"]}], names)
    assert msg["role"] == "user" and [b["tool_use_id"] for b in msg["content"]] == ["b", "a"]
    assert msg["content"][0]["content"] == '{"public_url":"u"}' and msg["content"][1]["content"] == '{"ok":true,"written":["p"]}'


# --- the wire ---------------------------------------------------------------------------------------------------------------------

def test_client_sends_the_bytes_it_renders(monkeypatch):
    """The service json.Compact-s the raw request: the context bytes on the
    wire must be the ones render.context_block puts in the logged message."""
    sent: dict = {}

    class Resp:
        status = 200
        headers = {"Content-Type": "application/json"}

        def read(self):
            return b'{"turnId":"t","events":[]}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        sent["data"] = req.data
        return Resp()

    monkeypatch.setattr(C.urlrequest, "urlopen", fake_urlopen)
    context = {"page": "home", "sections": [{"id": "x", "enabled": True}], "nav": [{"label": "Ünï <&>", "href": "/"}]}
    body = {"siteId": "s", "storeId": "lab", "manifestHash": "h", "buildId": "lab", "context": context, "text": "hi"}
    C.Client("http://127.0.0.1:9", "k").start_turn(body)
    wire = sent["data"].decode("utf-8")
    inner = R.context_block(context, "hi").split("\n")[1]
    assert inner in wire and wire == R.compact_json(body)


def _capture(where: Path) -> tuple[dict, dict, list[dict]]:
    case = json.loads((where / "case.json").read_text(encoding="utf-8"))
    row = jsonl(where / "collect-real.jsonl")[0]
    reqs = [r["body"] for r in jsonl(where / "model-requests.jsonl")]
    assert row["case_id"] == case["id"] and row["status"] == "ok" and reqs
    return case, row, reqs


@pytest.mark.parametrize("where", CAPTURES, ids=[str(p.relative_to(LAB)) for p in CAPTURES])
def test_rebuilt_messages_equal_the_real_binary_request_bytes(where):
    """Parity against the captured POST /v1/messages bodies of the real
    service for every capture (ok/written, error, warnings + sectionId,
    pick_asset): the collector's logged `messages`, the turn rebuilt from
    the row's events (dataset.to_hf's only path) and the wire are one and
    the same, byte for byte, request after request."""
    case, row, reqs = _capture(where)
    rebuilt = D.rebuild_messages(row, case)
    assert rebuilt == row["messages"] and D.stored_messages_match(row, case)  # the stored cache equals the rebuild
    full = R.flatten(rebuilt)
    for i, req in enumerate(reqs):  # request i = the prefix of the turn up to the model's i-th reply
        want = R.flatten(req["messages"])
        assert full[:len(want)] == want, f"request {i}"
        assert len(want) > (len(R.flatten(reqs[i - 1]["messages"])) if i else 0)
    last = len(R.flatten(reqs[-1]["messages"]))
    assert full[last:] and all(t[:2] == ("assistant", "text") for t in full[last:])  # after the last request: the final reply
    # every model call carries the frozen prefix, byte for byte (fixtures/prompt/system0/1, tools.json)
    system0 = (LAB / "fixtures" / "prompt" / "system0.rules.txt").read_text(encoding="utf-8")
    system1 = (LAB / "fixtures" / "prompt" / "system1.index.txt").read_text(encoding="utf-8")
    tools = json.loads((LAB / "fixtures" / "prompt" / "tools.json").read_text(encoding="utf-8"))
    for req in reqs:
        assert [b["text"] for b in req["system"]] == [system0, system1] and req["tools"] == tools
        assert req["max_tokens"] == 4096 and req["tool_choice"] == {"type": "auto"} and req["model"] == "fake"
    # the HF row reads those exact bytes: the user block and every tool row's content
    hf = D.to_hf(row, case, tools=[])
    assert hf["messages"][0]["content"] == reqs[0]["messages"][0]["content"][0]["text"]
    wire_results = [b["content"][0]["text"] for m in reqs[-1]["messages"] if m["role"] == "user"
                    for b in m["content"] if b["type"] == "tool_result"]
    assert [m["content"] for m in hf["messages"] if m["role"] == "tool"] == wire_results
    assert hf["meta"]["missing_tool_results"] == 0


def _results(reqs: list[dict]) -> list[dict]:
    return [b for m in reqs[-1]["messages"] if m["role"] == "user" for b in m["content"] if b["type"] == "tool_result"]


def _fixture(prefix: str) -> Path:
    found = [p for p in CAPTURES if p.parent == FIX and p.name.startswith(prefix)]
    assert len(found) == 1, f"{prefix}: {found}"
    return found[0]


def test_real_capture_ok_written_shape():
    case, row, reqs = _capture(_fixture("ok."))
    (res,) = _results(reqs)
    assert res["is_error"] is False and res["content"][0]["text"] == '{"ok":true,"written":["homeSections[1].settings.title"]}'
    assert row["verdict"] == "pass" and len(reqs) == 2 and R.flatten(rebuilt_tail(row, case)) == \
        [("assistant", "text", "Done — the title now reads as asked.")]


def rebuilt_tail(row, case):
    return D.rebuild_messages(row, case)[-1:]


def test_real_capture_error_shape():
    """loop.browserResult on failure: the flat api.ToolError struct {code,message,key} with is_error — then the model's retry."""
    case, row, reqs = _capture(_fixture("error."))
    err, ok = _results(reqs)
    assert err["is_error"] is True and err["content"][0]["text"] == \
        '{"code":"E_FIELD","message":"unknown field showButton for target","key":"showButton"}'
    assert ok["is_error"] is False and ok["content"][0]["text"] == '{"ok":true,"written":["homeSections[1].settings.tagline"]}'
    assert len(reqs) == 3 and row["batches"] == 2 and row["verdict"] == "invalid_op"
    assert [b.get("is_error", False) for m in row["messages"] for b in m["content"] if b["type"] == "tool_result"] == [True, False]


def test_real_capture_warnings_and_section_id_shape():
    """Two results in ONE user message, in call order; sorted map keys: ok, sectionId, warnings, written."""
    case, row, reqs = _capture(_fixture("warnings-sectionid."))
    add, gated = _results(reqs)
    assert add["content"][0]["text"] == '{"ok":true,"sectionId":"loyaltyForm","written":["homeSections"]}'
    assert gated["content"][0]["text"] == '{"ok":true,"warnings":["W_HIDDEN:tileBorder"],"written":["homeSections[1].settings.tileBorder"]}'
    assert len([m for m in reqs[-1]["messages"] if m["role"] == "user"]) == 2  # the context block + one results message
    assert row["batches"] == 1 and row["verdict"] == "extra_writes"  # the gated set_fields is the scripted extra write


def test_real_capture_pick_asset_shape():
    """turn.suspended → the asset the browser posted is the model's tool_result (bare api.Asset, struct order,
    srcset passed through compact); the rebuilt pick_asset tool_use equals the stored history (both required inputs)."""
    case, row, reqs = _capture(_fixture("pick_asset."))
    asset, written = _results(reqs)
    picked = row["picked_assets"][0]
    assert json.loads(asset["content"][0]["text"]) == picked and asset["is_error"] is False
    # the asset is state.fixture_asset_for(assets, case id, "image") — the generator's and the collector's shared pick
    assert asset["content"][0]["text"].startswith(
        '{"public_url":"https://cdn.int.breadstack.io/gal/lab/hero-field-1600x900.webp","alt":"Sunlit field at harvest","srcset":{"800":')
    assert picked["public_url"] == S.fixture_asset_for(S.load_assets_fixture(), case["id"], "image")["public_url"]
    assert json.loads(written["content"][0]["text"]) == {"ok": True, "written": ["homeSections[7].settings.bgImage",
                                                                                 "homeSections[7].settings.bgImageSrcSet"]}
    uses = [b for m in reqs[-1]["messages"] if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]
    assert uses[0]["name"] == "pick_asset" and uses[0]["input"] == {"asset_type": "image", "purpose": case["expected"]["ops"][0]["purpose"]}
    assert not any(e["event"] == "tool.call" and e["data"]["name"] == "pick_asset" for e in row["events"])  # no tool.call for it
    assert [e["event"] for e in row["events"]][:3] == ["text.delta", "turn.suspended", "tool.result"]
    assert row["verdict"] == "pass" and len(reqs) == 3
