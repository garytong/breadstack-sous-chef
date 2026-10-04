"""serving.py — serving parity (docs/LAB.md §5): what train.py renders is,
byte for byte, what Ollama 0.34.4's qwen3.5 renderer puts in front of the
served student.

fixtures/parity/served.json holds Ollama's own prompt for every model
request of the four real captures (fixtures/prompt/real: ok/written, error +
retry, two results in one turn, pick_asset — 10 requests, tool_use and
tool_result in every multi-call one), rendered by Ollama's own code through
parity/ (`python -m souschef_lab.serving --write`). Here the same captures
go the way the training data goes — dataset.to_hf, then serving — and:

- `render_prompt` of the conversation up to each request equals Ollama's
  prompt for it, byte for byte (the served form: no think block on the
  echoed assistant turns, the empty think block as the generation prompt);
- the training text (`render_training`, what train.py tokenizes) is the
  same bytes up to each call's generation point once the earlier turns'
  empty think blocks are removed — the one residue a single sequence cannot
  avoid (every assistant turn is the generation point of its own call);
- each span train_on_responses_only trains on is the empty think block plus
  exactly the echo Ollama writes of that turn in the next request's prompt:
  the student learns the bytes the parser reads back;
- a DPO pair's prompt is the served prompt and its completion the echo.

The Go-format helpers are pinned on the shapes that differ from the HF
template's tojson (key order, dropped keys, escapes, spaces, numbers).
With Go on PATH the harness is re-run and must still give the fixture
(SOUS_CHEF_LAB_REQUIRE_GO=1 turns that skip into a failure)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from souschef_lab import dataset as D  # noqa: E402
from souschef_lab import dpo, serving as S, train, trainkit  # noqa: E402

FIX = LAB / "fixtures" / "prompt"
CAPTURES = S.captures(FIX / "real")
SYSTEM = ((FIX / "system0.rules.txt").read_text(encoding="utf-8"), (FIX / "system1.index.txt").read_text(encoding="utf-8"))
TOOLS = json.loads((FIX / "tools.json").read_text(encoding="utf-8"))
GEN = S.GENERATION_PROMPT
USER, ASSISTANT = trainkit.MARKERS["chatml"]


@pytest.fixture(scope="module")
def served() -> dict[str, list[str]]:
    fixture = json.loads(S.FIXTURE.read_text(encoding="utf-8"))
    assert (fixture["ollama"], fixture["renderer"], fixture["parser"], fixture["think"]) == \
        (S.OLLAMA_VERSION, S.RENDERER, S.PARSER, ["false"])
    prompts = S.fixture_prompts(fixture)
    assert sorted(prompts) == [p.name for p in CAPTURES]
    return prompts


def hf_row(where: Path) -> dict:
    case = json.loads((where / "case.json").read_text(encoding="utf-8"))
    row = S.read_jsonl(where / "collect-real.jsonl")[0]
    return D.to_hf(row, case, system=SYSTEM, tools=TOOLS)


def assistant_indexes(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m["role"] == "assistant"]


# --- Go's encoding/json and fmt -------------------------------------------------------------------------------------------

def test_go_string_escapes_as_json_marshal_does():
    assert S.go_string('a<b & "c">\n é') == '"a\\u003cb \\u0026 \\"c\\"\\u003e\\n\\u2028é"'
    assert S.go_string("\x01") == '"\\u0001"'


def test_go_numbers():
    assert [S.go_json_number(x) for x in (1, 1.0, 0.5, 1e21, 1e-7, 123456789.0)] == \
        ["1", "1", "0.5", "1e+21", "1e-7", "123456789"]
    assert [S.go_fmt_v(x) for x in (2, 58, 1000000, 0.00001, -3.5)] == ["2", "58", "1e+06", "1e-05", "-3.5"]


def test_go_json_is_compact_sorted_and_escaped():
    assert S.go_json({"b": [1, {"z": "<x>", "a": None}], "a": True}) == '{"a":true,"b":[1,{"a":null,"z":"\\u003cx\\u003e"}]}'


def test_tool_argument_as_format_tool_call_argument():
    assert S.tool_argument("as is <&>") == "as is <&>"
    assert S.tool_argument(None) == "null" and S.tool_argument(True) == "true" and S.tool_argument(2) == "2"
    assert S.tool_argument({"kind": "global", "panel": "X"}) == '{"kind":"global","panel":"X"}'
    assert S.tool_argument(["b", "a"]) == '["b","a"]'


def test_served_arguments_sort_the_top_level_as_the_wire_does():
    args = json.dumps({"target": {"kind": "page", "page": "home"}, "list": "sections", "action": "add", "item": {"id": "x"}})
    assert list(S.served_arguments(args)) == ["action", "item", "list", "target"]
    assert S.served_arguments("not json") == {} and S.served_arguments("[1]") == {}


def test_unservable_calls():
    def row(**args):
        return [{"role": "assistant", "content": "", "tool_calls": [trainkit.render_tool_call("t", "edit_list", args)]}]
    assert S.unservable(row(index=2, target={"kind": "page"})) == []
    assert S.unservable(row(index=1000000)) == ["edit_list.index: 1000000 is served as 1e+06"]
    assert S.unservable(row(list="x\n")) == ["edit_list.list: a string starting or ending with a newline"]
    assert S.unservable(row(action="NULL")) == ['edit_list.action: the string "null"']


# --- api.Tool --------------------------------------------------------------------------------------------------------------------

def test_ollama_tool_json_struct_order_dropped_keys_escapes_spaces():
    tool = trainkit.hf_tool({"name": "f", "description": "a <b>", "strict": True, "input_schema": {
        "additionalProperties": False,
        "properties": {"k": {"description": "", "enum": ["x", "y"], "type": "string"},
                       "n": {"anyOf": [{"description": "", "type": "integer"},
                                       {"items": {"description": "", "type": "integer"}, "type": "array"}], "description": "d"},
                       "o": {"additionalProperties": False, "properties": {"z": {"type": ["string", "null"]}},
                             "required": ["z"], "type": "object"},
                       "l": {"items": {"type": "string"}, "minItems": 1, "type": "array"}},
        "required": ["k"], "type": "object"}})
    assert S.ollama_tool_json(tool) == (
        '{"type": "function", "function": {"name": "f", "description": "a \\u003cb\\u003e", "parameters": {"type": "object", '
        '"required": ["k"], "properties": {"k": {"type": "string", "enum": ["x", "y"]}, '
        '"n": {"anyOf": [{"type": "integer"}, {"type": "array", "items": {"description": "", "type": "integer"}}], "description": "d"}, '
        '"o": {"type": "object", "properties": {"z": {"type": ["string", "null"]}}, "required": ["z"]}, '
        '"l": {"type": "array", "items": {"type": "string"}}}}}}')
    # no description, no properties, required: [] → description and required omitted, properties null
    bare = trainkit.hf_tool({"name": "g", "input_schema": {"type": "object", "required": []}})
    assert S.ollama_tool_json(bare) == '{"type": "function", "function": {"name": "g", "parameters": {"type": "object", "properties": null}}}'


def test_add_json_spaces_leaves_strings_alone():
    assert S.add_json_spaces('{"a:b,c":"x:y,z","d":[1,2]}') == '{"a:b,c": "x:y,z", "d": [1, 2]}'
    assert S.add_json_spaces('{"q":"a\\"b:c"}') == '{"q": "a\\"b:c"}'


# --- the renderer on the real captures ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("where", CAPTURES, ids=[p.name for p in CAPTURES])
def test_render_prompt_is_ollamas_prompt_for_every_request(where, served):
    """The served form: for request k, the conversation up to the k-th
    assistant turn renders to exactly the prompt Ollama 0.34.4 rendered."""
    hf = hf_row(where)
    starts = assistant_indexes(hf["messages"])
    prompts = served[where.name]
    assert len(starts) == len(prompts) >= 2
    for k, prompt in enumerate(prompts):
        got = S.render_prompt(hf["messages"][:starts[k]], hf["tools"])
        assert got == prompt, f"request {k + 1}: {S.first_difference(got, prompt)}"
        assert prompt.endswith(GEN)


@pytest.mark.parametrize("where", CAPTURES, ids=[p.name for p in CAPTURES])
def test_training_text_is_the_served_bytes_up_to_each_call(where, served):
    """train.render_text's text, cut at call k's generation point with the
    k-1 earlier assistant turns' empty think blocks removed, is Ollama's
    prompt for call k; the first call's prompt is a plain prefix."""
    hf = hf_row(where)
    text = train.render_text(hf["messages"], hf["tools"])
    assert text == S.render_training(hf["messages"], hf["tools"])
    points = S.generation_points(text)
    prompts = served[where.name]
    assert len(points) == len(prompts)
    assert text.startswith(prompts[0])
    for k, prompt in enumerate(prompts):
        cut = S.strip_history_think(text[:points[k]])
        assert cut == prompt, f"call {k + 1}: {S.first_difference(cut, prompt)}"
        assert len(text[:points[k]]) - len(cut) == k * len(S.EMPTY_THINK)  # the residue: one empty block per earlier turn
    assert text.endswith(S.IM_END + "\n")


@pytest.mark.parametrize("where", CAPTURES, ids=[p.name for p in CAPTURES])
def test_each_trained_span_is_the_empty_think_block_plus_ollamas_echo(where, served):
    """What train_on_responses_only keeps of turn k (from `<|im_start|>assistant\\n`
    to the next `<|im_start|>user\\n`) is the empty think block followed by
    exactly the bytes Ollama writes for that turn in request k+1's prompt —
    the parser's reading of the model's reply, as later calls see it."""
    hf = hf_row(where)
    text = train.render_text(hf["messages"], hf["tools"])
    spans = trainkit.trained_spans(text, USER, ASSISTANT)
    prompts = served[where.name]
    assert len(spans) == len(prompts)
    for k in range(len(prompts) - 1):
        after = prompts[k + 1][len(prompts[k]) - len(GEN):]  # from the k-th assistant turn's opening
        echo = after[len(ASSISTANT):after.index(USER)]
        assert spans[k] == S.EMPTY_THINK + echo
        assert "<tool_call>\n<function=" in echo and "<think>" not in echo
    assert spans[-1].startswith(S.EMPTY_THINK) and spans[-1].endswith(S.IM_END + "\n")
    assert not trainkit.leaked_tool_results(hf, text, USER, ASSISTANT)
    assert S.unservable(hf["messages"]) == []


def test_render_differs_from_the_hf_template_only_where_ollama_does():
    """The shapes the K8 review found: the tools block (struct order, dropped
    additionalProperties/minItems/strict, \\u003c escapes, spaces), the
    echoed arguments (sorted keys, compact JSON), no think block in a served
    history. Pinned on a small row so the intent reads without a tokenizer."""
    tool = trainkit.hf_tool({"name": "set_fields", "description": "Write <k>", "strict": True, "input_schema": {
        "additionalProperties": False, "properties": {"target": {"description": "t", "type": "object"}}, "required": ["target"],
        "type": "object"}})
    call = trainkit.render_tool_call("t1", "set_fields", {"values": {"b": 1, "a": "x & y"}, "target": {"kind": "global"}})
    messages = [{"role": "system", "content": "RULES\n"}, {"role": "user", "content": "<context>\n{}\n</context>\nhi"},
                {"role": "assistant", "content": "Doing it.", "tool_calls": [call]},
                trainkit.render_tool_result("t1", "set_fields", '{"ok":true}'),
                {"role": "assistant", "content": "Done."}]
    text = S.render_training(messages, [tool])
    assert text == (
        '<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n'
        '{"type": "function", "function": {"name": "set_fields", "description": "Write \\u003ck\\u003e", "parameters": '
        '{"type": "object", "required": ["target"], "properties": {"target": {"type": "object", "description": "t"}}}}}'
        + S.TOOL_POSTAMBLE + '\n\nRULES<|im_end|>\n'
        '<|im_start|>user\n<context>\n{}\n</context>\nhi<|im_end|>\n'
        '<|im_start|>assistant\n<think>\n\n</think>\n\nDoing it.\n\n<tool_call>\n<function=set_fields>\n'
        '<parameter=target>\n{"kind":"global"}\n</parameter>\n<parameter=values>\n{"a":"x \\u0026 y","b":1}\n</parameter>\n'
        '</function>\n</tool_call><|im_end|>\n'
        '<|im_start|>user\n<tool_response>\n{"ok":true}\n</tool_response><|im_end|>\n'
        '<|im_start|>assistant\n<think>\n\n</think>\n\nDone.<|im_end|>\n')
    served_prompt = S.render_prompt(messages[:4], [tool])
    assert served_prompt == text[:text.index("Doing it.")].replace(ASSISTANT + S.EMPTY_THINK, ASSISTANT) + \
        "Doing it.\n\n<tool_call>\n<function=set_fields>\n<parameter=target>\n{\"kind\":\"global\"}\n</parameter>\n" \
        "<parameter=values>\n{\"a\":\"x \\u0026 y\",\"b\":1}\n</parameter>\n</function>\n</tool_call><|im_end|>\n" \
        "<|im_start|>user\n<tool_response>\n{\"ok\":true}\n</tool_response><|im_end|>\n" + GEN
    # a conversation ending in an assistant turn: a prefill at serve (no <|im_end|>), closed in the training text
    assert S.render_prompt(messages, [tool]).endswith("Done.")
    assert S.render_conversation(messages, [tool], think_turns="last").endswith("</think>\n\nDone.<|im_end|>\n")


def test_two_calls_two_results_and_no_tools():
    calls = [trainkit.render_tool_call("a", "reset", {"keys": ["x"], "target": {"kind": "global"}}),
             trainkit.render_tool_call("b", "set_page_layout", {"layout": "two-column", "page": "faq"})]
    messages = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": calls},
                trainkit.render_tool_result("a", "reset", '{"ok":true}'), trainkit.render_tool_result("b", "set_page_layout", '{"ok":true}')]
    assert S.render_prompt(messages, None) == (
        '<|im_start|>user\ngo<|im_end|>\n'
        '<|im_start|>assistant\n<tool_call>\n<function=reset>\n<parameter=keys>\n["x"]\n</parameter>\n<parameter=target>\n'
        '{"kind":"global"}\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=set_page_layout>\n'
        '<parameter=layout>\ntwo-column\n</parameter>\n<parameter=page>\nfaq\n</parameter>\n</function>\n</tool_call><|im_end|>\n'
        '<|im_start|>user\n<tool_response>\n{"ok":true}\n</tool_response>\n<tool_response>\n{"ok":true}\n</tool_response><|im_end|>\n'
        + GEN)
    # a system message without tools is its own turn, trimmed as Go's TrimSpace trims
    assert S.render_prompt([{"role": "system", "content": " S \n"}, {"role": "user", "content": "u"}], []) == \
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nu<|im_end|>\n" + GEN


def test_dpo_pair_is_the_served_prompt_and_the_echo():
    hf = hf_row(CAPTURES[0])
    k = assistant_indexes(hf["messages"])[-1]
    row = {"prompt": hf["messages"][:k], "chosen": [hf["messages"][k]],
           "rejected": [dict(hf["messages"][k], content="Nope.")], "tools": hf["tools"]}
    p, c, r = dpo.render_pair(row, S.IM_END)
    assert p == S.render_prompt(row["prompt"], hf["tools"]) and p.endswith(GEN)
    assert c == hf["messages"][k]["content"] + S.IM_END and r == "Nope." + S.IM_END
    bad = dict(row, chosen=[{"role": "assistant", "content": "", "tool_calls": [
        trainkit.render_tool_call("x", "edit_list", {"index": 1000000})]}])
    assert dpo.render_pair(bad, S.IM_END) == "unservable_call"


# --- the fixture itself ---------------------------------------------------------------------------------------------------------------

def test_fixture_is_what_the_harness_gives_now():
    """Re-render the captures through parity/ (Ollama 0.34.4's own code) and
    compare with the fixture: skipped without Go on PATH."""
    if not S.harness_available():
        if os.environ.get("SOUS_CHEF_LAB_REQUIRE_GO"):
            pytest.fail("SOUS_CHEF_LAB_REQUIRE_GO is set but go is not on PATH")
        pytest.skip("go not on PATH: the parity harness is not re-run")
    try:
        now = S.served_fixture()
    except RuntimeError as e:
        if os.environ.get("SOUS_CHEF_LAB_REQUIRE_GO"):
            raise
        pytest.skip(f"parity/ could not build (offline toolchain?): {str(e)[:200]}")
    old = json.loads(S.FIXTURE.read_text(encoding="utf-8"))
    assert S.comparable(now) == S.comparable(old)
