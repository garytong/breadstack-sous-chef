"""tests/fake_model.py — the scripted /v1/messages stub the real service binary talks to."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
for _p in (LAB, LAB / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fake_model as FM  # noqa: E402

TOOL_USE = {"type": "tool_use", "id": "t1", "name": "set_fields",
            "input": {"target": {"kind": "section", "page": "home", "sectionId": "contentBlock-1"}, "values": {"title": "Hello"}}}
STEP_TOOL = {"message": {"content": [{"type": "text", "text": "Setting the title."}, TOOL_USE],
                         "usage": {"input_tokens": 120, "output_tokens": 30}}}
STEP_DONE = {"message": {"content": [{"type": "text", "text": "Done."}]}}


def post(url: str, body: dict, *, stream: bool = False) -> tuple[int, dict, bytes]:
    data = json.dumps({**body, "stream": stream}).encode()
    req = urllib.request.Request(url + "/v1/messages?beta=true", data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def parse_sse(raw: bytes) -> list[tuple[str, dict]]:
    out = []
    for frame in raw.decode().split("\n\n"):
        name = data = None
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if name:
            out.append((name, data))
    return out


def test_message_json_matches_the_go_fakes_shape():
    m = FM.message_json(STEP_TOOL["message"])
    assert m["type"] == "message" and m["role"] == "assistant" and m["id"] == "msg_fake" and m["model"] == "fake-model"
    assert m["stop_reason"] == "tool_use"  # defaulted from the tool_use block
    assert m["stop_sequence"] is None and m["stop_details"] is None
    assert m["content"][1] == TOOL_USE
    assert m["usage"] == {"input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    assert FM.message_json(STEP_DONE["message"])["stop_reason"] == "end_turn"


def test_json_mode_serves_steps_in_order_then_500():
    with FM.FakeModel([STEP_TOOL, STEP_DONE]) as fake:
        s1, _, b1 = post(fake.url, {"model": "fake", "messages": [{"role": "user", "content": "hi"}]})
        s2, _, b2 = post(fake.url, {"model": "fake", "messages": []})
        s3, _, b3 = post(fake.url, {"model": "fake", "messages": []})
    assert (s1, s2, s3) == (200, 200, 500)
    assert json.loads(b1)["content"][1]["name"] == "set_fields"
    assert json.loads(b2)["stop_reason"] == "end_turn"
    assert json.loads(b3)["error"]["type"] == "api_error"
    assert [r["path"] for r in fake.requests] == ["/v1/messages?beta=true"] * 3
    assert fake.requests[0]["body"]["messages"][0]["content"] == "hi" and fake.requests[0]["stream"] is False


def test_stream_mode_splits_tool_input_over_deltas_that_accumulate_to_the_input():
    with FM.FakeModel([STEP_TOOL]) as fake:
        status, headers, raw = post(fake.url, {"model": "fake", "messages": []}, stream=True)
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    events = parse_sse(raw)
    names = [n for n, _ in events]
    assert names[0] == "message_start" and names[-2:] == ["message_delta", "message_stop"]
    assert names.count("content_block_start") == 2 and names.count("content_block_stop") == 2
    parts = [d["delta"]["partial_json"] for n, d in events if n == "content_block_delta" and d["delta"]["type"] == "input_json_delta"]
    assert len(parts) == 3 and json.loads("".join(parts)) == TOOL_USE["input"]
    text = "".join(d["delta"]["text"] for n, d in events if n == "content_block_delta" and d["delta"]["type"] == "text_delta")
    assert text == "Setting the title."
    start = events[0][1]["message"]
    assert start["content"] == [] and start["stop_reason"] is None and start["usage"]["output_tokens"] == 1
    delta = events[-2][1]
    assert delta["delta"]["stop_reason"] == "tool_use" and delta["usage"]["output_tokens"] == 30


def test_error_steps_and_appended_steps():
    err = {"status": 529, "body": {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}}
    with FM.FakeModel([err]) as fake:
        s1, _, b1 = post(fake.url, {"messages": []})
        fake.append(STEP_DONE)
        s2, _, _ = post(fake.url, {"messages": []})
    assert s1 == 529 and json.loads(b1)["error"]["type"] == "overloaded_error"
    assert s2 == 200


def test_capture_file_and_load_steps(tmp_path):
    steps = tmp_path / "steps.json"
    steps.write_text(json.dumps([STEP_DONE]))
    cap = tmp_path / "cap.jsonl"
    with FM.FakeModel(FM.load_steps(steps), capture=cap) as fake:
        post(fake.url, {"messages": [{"role": "user", "content": "x"}]})
    rows = [json.loads(l) for l in cap.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["body"]["messages"][0]["content"] == "x"


def test_cli_registers_fake_model():
    from souschef_lab import cli
    assert "fake-model" in cli.COMMANDS
    fn = cli.resolve("fake-model")  # imported as `tests.fake_model`: a sibling module object of this file's `fake_model`
    assert fn.__name__ == "main" and fn.__module__.endswith("fake_model")
