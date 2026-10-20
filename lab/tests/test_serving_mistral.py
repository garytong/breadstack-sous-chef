"""serving_mistral.py — serving parity for the Mistral arm: what train.py --family mistral renders is, byte for
byte, what Ollama 0.34.4 puts in front of a Ministral 3 student served with the arm's TEMPLATE + PARSER ministral.

fixtures/parity/served-ministral.json holds Ollama's own prompt for every model request of the real captures
(fixtures/prompt/real: 7 captures, 16 requests), rendered through parity/ in template mode
(`python -m souschef_lab.serving_mistral --write`). Here the same captures go the way the training data goes —
dataset.to_hf, then serving_mistral — and:

- `render_prompt` of the conversation up to each request equals Ollama's prompt for it, byte for byte;
- every served prompt is an EXACT prefix of the training text (no think residue, unlike qwen3.5), cut at the
  generation points;
- each trained segment (content, calls, </s>) is exactly the bytes Ollama writes for that turn in the next request.

fixtures/mistral/hf-golden.json holds `transformers.apply_chat_template` with Mistral's own pinned chat template on
the captures and six synthetic edge cases: the same renderer in style="python" (+ "<s>") must equal it — the
layout is the model's own. With transformers importable and MX_TOKENIZER_DIR set, it is re-rendered live.

The Go-format details are pinned on small rows. With Go on PATH and a parity/ harness that has template mode
(`-modelfile`), the harness is re-run (SOUS_CHEF_LAB_REQUIRE_GO=1 turns the skip into a failure)."""

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
from souschef_lab import export, mx_hf_golden as HG, serving, serving_mistral as M, trainkit  # noqa: E402

FIX = LAB / "fixtures" / "prompt"
CAPTURES = serving.captures(FIX / "real")
SYSTEM = ((FIX / "system0.rules.txt").read_text(encoding="utf-8"), (FIX / "system1.index.txt").read_text(encoding="utf-8"))
TOOLS = json.loads((FIX / "tools.json").read_text(encoding="utf-8"))
BASE_MODELFILE = LAB / "base-modelfiles" / "ministral-3-8b.modelfile"


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(M.FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def served(fixture) -> dict[str, list[str]]:
    assert (fixture["ollama"], fixture["renderer"], fixture["parser"], fixture["think"]) == \
        (M.OLLAMA_VERSION, None, M.PARSER, ["false"])
    assert fixture["template_sha256"] == M.template_sha256()
    prompts = serving.fixture_prompts(fixture)
    assert sorted(prompts) == [p.name for p in CAPTURES]
    return prompts


def hf_row(where: Path) -> dict:
    case = json.loads((where / "case.json").read_text(encoding="utf-8"))
    row = serving.read_jsonl(where / "collect-real.jsonl")[0]
    return D.to_hf(row, case, system=SYSTEM, tools=TOOLS)


def assistant_indexes(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m["role"] == "assistant"]


# --- the renderer on the real captures -------------------------------------------------------------------------------------

def test_the_fixture_covers_every_request():
    assert len(CAPTURES) == 7
    assert sum(len(serving.read_jsonl(c / "model-requests.jsonl")) for c in CAPTURES) == 16


@pytest.mark.parametrize("where", CAPTURES, ids=[p.name for p in CAPTURES])
def test_render_prompt_is_ollamas_prompt_for_every_request(where, served):
    hf = hf_row(where)
    starts = assistant_indexes(hf["messages"])
    prompts = served[where.name]
    assert len(starts) == len(prompts) >= 2
    for k, prompt in enumerate(prompts):
        got = M.render_prompt(hf["messages"][:starts[k]], hf["tools"])
        assert got == prompt, f"request {k + 1}: {M.first_difference(got, prompt)}"
        assert prompt.startswith(M.SYS_OPEN) and prompt.count(M.TOOLS_OPEN) == 1 and M.BOS not in prompt


@pytest.mark.parametrize("where", CAPTURES, ids=[p.name for p in CAPTURES])
def test_every_served_prompt_is_an_exact_prefix_of_the_training_text(where, served):
    """No residue: the training text cut at call k's generation point IS Ollama's prompt for call k."""
    hf = hf_row(where)
    text, spans = M.render_conversation(hf["messages"], hf["tools"])
    assert text == M.render_training(hf["messages"], hf["tools"]) == \
        M.render_training(hf["messages"], hf["tools"], history_turns=1)  # history changes no byte
    points = M.generation_points(text)
    prompts = served[where.name]
    assert points == [a for a, _ in spans]
    assert len(points) == len(prompts)
    for k, prompt in enumerate(prompts):
        assert text[:points[k]] == prompt, f"call {k + 1}: {M.first_difference(text[:points[k]], prompt)}"
    assert text.endswith(M.END_OF_TURN) and M.serving.unservable(hf["messages"]) == [] and M.unservable(hf["messages"]) == []


@pytest.mark.parametrize("where", CAPTURES, ids=[p.name for p in CAPTURES])
def test_each_trained_segment_is_what_ollama_writes_for_that_turn(where, served):
    hf = hf_row(where)
    text, spans = M.render_conversation(hf["messages"], hf["tools"])
    prompts = served[where.name]
    for k in range(len(prompts) - 1):
        a, b = spans[k]
        nxt = prompts[k + 1]
        assert nxt[a:b] == text[a:b]
        assert text[a:b].endswith(M.END_OF_TURN) and M.CALLS in text[a:b]
        assert nxt[b:].startswith(M.RESULTS_OPEN)
    segs = M.assistant_segments(hf["messages"], hf["tools"])
    assert segs == [text[a:b] for a, b in spans]
    trained = "".join(segs)
    for m in hf["messages"]:  # no tool output inside a trained segment
        if m["role"] == "tool" and m["content"][:200]:
            assert m["content"][:200] not in trained
    # the trained calls read back as the row's calls (the parser's reading)
    calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for m in hf["messages"]
             if m["role"] == "assistant" for c in m.get("tool_calls") or []]
    assert [c for s in segs for c in M.parse_tool_calls(s)] == calls


# --- Mistral's own template: the HF golden -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def golden() -> dict:
    g = json.loads(M.HF_GOLDEN.read_text(encoding="utf-8"))
    assert g["model"] == "mistralai/Ministral-3-8B-Instruct-2512-BF16"
    assert g["revision"] == "f6fae9795746f63c9be8344932f01275f3c63734"
    return g


def test_the_golden_records_the_measured_special_ids(golden):
    assert golden["special_ids"] == {t: [i] for t, i in M.SPECIAL_IDS.items()}
    assert golden["transformers"] == "5.5.0"


def test_the_python_style_render_is_mistrals_own_template_on_every_case(golden):
    texts = HG.golden_texts(golden)
    cases = dict(HG.all_cases())
    assert sorted(texts) == sorted(cases) and len(texts) == 13
    for name, case in cases.items():
        got = M.render_conversation(case["messages"], case["tools"], style="python", bos=True)[0]
        assert got == texts[name], f"{name}: {M.first_difference(got, texts[name])}"


def test_the_served_render_differs_from_the_golden_only_in_json_style_and_bos(golden):
    """Same layout: the go-style text with every JSON span replaced by its python-style twin and "<s>" first is the
    golden (shown on the synthetic cases, whose JSON is small)."""
    texts = HG.golden_texts(golden)
    for name, case in HG.synthetic_cases():
        go = M.render_conversation(case["messages"], case["tools"])[0]
        tj_go, tj_py = M.tools_json(case["tools"]), M.tools_json(case["tools"], "python")
        conv = go.replace(tj_go, tj_py)
        for m in case["messages"]:
            for c in m.get("tool_calls") or []:
                a = c["function"]["arguments"]
                conv = conv.replace(M.ARGS + M.args_json(a), M.ARGS + M.args_json(a, "python"), 1)
        assert M.BOS + conv == texts[f"synthetic:{name}"], name


def test_the_golden_is_what_transformers_renders_now(golden):
    tdir = os.environ.get("MX_TOKENIZER_DIR")
    try:
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("transformers is not installed (the training env re-renders the golden)")
    if not tdir:
        pytest.skip("MX_TOKENIZER_DIR is not set")
    tok, template = HG.load_tokenizer(Path(tdir))
    assert serving.sha(template) == golden["chat_template_sha256"]
    texts = HG.golden_texts(golden)
    for name, case in HG.all_cases():
        assert HG.hf_render(tok, template, case) == texts[name], name


# --- Go's JSON as the template prints it -----------------------------------------------------------------------------------

def test_tools_json_is_template_tools_marshal():
    tool = trainkit.hf_tool({"name": "f", "description": "a <b> & c", "strict": True, "input_schema": {
        "additionalProperties": False, "type": "object", "required": ["z"],
        "properties": {"z": {"type": "string", "description": "zed"}, "a": {"type": ["integer", "null"]},
                       "m": {"type": "object", "properties": {"y": {"type": "string"}, "b": {"type": "string"}}}}}})
    assert M.tools_json([tool]) == (
        '[{"type":"function","function":{"name":"f","description":"a \\u003cb\\u003e \\u0026 c","parameters":'
        '{"type":"object","required":["z"],"properties":{"a":{"type":["integer","null"]},'
        '"m":{"type":"object","properties":{"y":{"type":"string"},"b":{"type":"string"}}},'
        '"z":{"type":"string","description":"zed"}}}}}]')  # top level sorted (a Go map); nested kept in order
    bare = trainkit.hf_tool({"name": "g", "input_schema": {"type": "object", "required": []}})
    assert M.tools_json([bare]) == ('[{"type":"function","function":{"name":"g","description":"","parameters":'
                                    '{"type":"object","properties":null}}}]')  # description always; required omitted
    assert M.tools_json([tool], "python") == json.dumps([tool], ensure_ascii=False)


def test_args_json_is_template_args_string():
    assert M.args_json(json.dumps({"z": 1, "a": "x<y&z", "é": None})) == '{"a":"x\\u003cy\\u0026z","z":1,"é":null}'
    assert M.args_json(json.dumps({"n": 1000000, "f": 0.5, "l": [{"b": 1, "a": 2}], "e": {}})) == \
        '{"e":{},"f":0.5,"l":[{"a":2,"b":1}],"n":1000000}'
    assert M.args_json("") == "{}" and M.args_json("{}") == "{}" and M.args_json("[1]") == "{}"
    assert M.args_json('{"b": 1, "a": 2}', "python") == '{"b": 1, "a": 2}' and M.args_json("", "python") == "{}"
    assert M.args_json({"b": 1}, "python") == '{"b": 1}'


def test_collate_merges_consecutive_roles_as_ollama_does():
    call = trainkit.render_tool_call("t", "f", {"a": 1})
    msgs = [{"role": "system", "content": "A"}, {"role": "system", "content": "B"}, {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"}, {"role": "assistant", "content": "x", "tool_calls": [call]},
            {"role": "assistant", "content": "y", "tool_calls": [call]},
            trainkit.render_tool_result("t", "f", "r1"), trainkit.render_tool_result("t", "f", "r2")]
    go = M.collate(msgs)
    assert [m["role"] for m in go] == ["system", "user", "assistant", "tool", "tool"]
    assert go[0]["content"] == "A\n\nB" and go[1]["content"] == "u1\n\nu2" and go[2]["content"] == "x\n\ny"
    assert len(go[2]["tool_calls"]) == 1  # Go appends content only: the second message's calls are gone
    py = M.collate(msgs, "python")
    assert [m["role"] for m in py] == ["system", "system", "user", "assistant", "tool", "tool"]
    assert len(py[3]["tool_calls"]) == 2
    assert msgs[0]["content"] == "A"  # the input is not mutated
    assert M.merged_assistants(msgs) == 1
    assert M.unservable(msgs) == ["consecutive assistant messages: Ollama's collate drops the second one's tool calls"]


def test_the_layout_on_a_small_conversation():
    call = trainkit.render_tool_call("t1", "set_fields", {"values": {"b": 1, "a": "x & y"}, "target": {"kind": "global"}})
    tool = trainkit.hf_tool({"name": "set_fields", "description": "Write", "input_schema": {
        "type": "object", "properties": {"target": {"type": "object"}}}})
    msgs = [{"role": "system", "content": "RULES\n"}, {"role": "user", "content": " hi "},
            {"role": "assistant", "content": "Doing it.", "tool_calls": [call]},
            trainkit.render_tool_result("t1", "set_fields", '{"ok":true}'), {"role": "assistant", "content": "Done."}]
    text, spans = M.render_conversation(msgs, [tool])
    tj = M.tools_json([tool])
    assert text == ("[SYSTEM_PROMPT]RULES\n[/SYSTEM_PROMPT][AVAILABLE_TOOLS]" + tj + "[/AVAILABLE_TOOLS][INST] hi [/INST]"
                    'Doing it.[TOOL_CALLS]set_fields[ARGS]{"target":{"kind":"global"},"values":{"a":"x \\u0026 y","b":1}}</s>'
                    '[TOOL_RESULTS]{"ok":true}[/TOOL_RESULTS]Done.</s>')  # nothing trimmed
    assert [text[a:b] for a, b in spans] == [
        'Doing it.[TOOL_CALLS]set_fields[ARGS]{"target":{"kind":"global"},"values":{"a":"x \\u0026 y","b":1}}</s>',
        "Done.</s>"]
    assert M.render_prompt(msgs, [tool]).endswith("[/TOOL_RESULTS]Done.")  # a trailing reply is an open prefill
    assert M.render_prompt(msgs[:3], [tool]).endswith("}}</s>")  # …but a trailing call is closed
    assert M.render_prompt(msgs[:2], None) == "[SYSTEM_PROMPT]RULES\n[/SYSTEM_PROMPT][INST] hi [/INST]"
    assert M.render_conversation(msgs, [tool], bos=True)[0] == "<s>" + text
    two = msgs + [{"role": "user", "content": "again"}, {"role": "assistant", "content": "Ok."}]
    t2 = M.render_training(two, [tool])
    assert t2.count(M.TOOLS_OPEN) == 1 and t2.endswith("[INST]again[/INST]Ok.</s>")
    assert M.generation_points(t2) == [a for a, _ in M.render_conversation(two, [tool])[1]]


def test_parse_tool_calls_reads_what_the_ministral_parser_reads():
    seg = ('Sure.[TOOL_CALLS]a[ARGS]{"x":{"y":[1,{"z":"}]"}]}}[TOOL_CALLS]b[ARGS]{}</s>')
    assert M.parse_tool_calls(seg) == [("a", {"x": {"y": [1, {"z": "}]"}]}}), ("b", {})]
    assert M.parse_tool_calls("Just text.</s>") == []
    assert M.parse_tool_calls('[TOOL_CALLS]a[ARGS]{"x": 1') == []  # incomplete
    assert M.parse_tool_calls('[TOOL_CALLS]a[ARGS][1][TOOL_CALLS]b[ARGS]{}') == []  # not an object: the parser refuses
    assert M.parse_tool_calls('[TOOL_CALLS]a[ARGS]{"s":"a\\"}b"}') == [("a", {"s": 'a"}b'})]
    assert M.find_json_end('{"a":"}"}tail') == 8


def test_unservable_is_the_union_of_both_rules():
    def row(args):
        return [{"role": "assistant", "content": "", "tool_calls": [trainkit.render_tool_call("t", "f", args)]}]
    assert M.unservable(row({"n": 1000000})) == ["f.n: 1000000 is served as 1e+06"]  # qwen's rule, kept
    assert M.unservable_mistral(row({"n": 1000000})) == []  # Mistral serves 1000000 as 1000000
    assert M.unservable(row({"n": 2 ** 53 + 1})) == ["f.n: 9007199254740993 is served as 9.007199254740992e+15",
                                                     "f.n: 9007199254740993 is beyond float64's exact integers"]
    assert M.unservable_mistral(row({"l": [{"x": float("inf")}]})) == ["f.l: inf is not a finite number"]
    assert M.unservable(row({"s": "ok", "t": True, "u": None})) == []


# --- the Modelfile ---------------------------------------------------------------------------------------------------------------

def test_the_base_modelfile_text_parses_to_the_template_with_no_trailing_newline(fixture):
    text = M.modelfile_text()
    assert M.template_of(text) == M.TEMPLATE and not M.TEMPLATE.endswith("\n")
    assert M.template_sha256() == fixture["template_sha256"]
    ins = export.parse_modelfile(text)
    assert [i.keyword for i in ins] == ["TEMPLATE", "PARSER", "PARAMETER", "PARAMETER", "PARAMETER"]
    assert ins[1].raw == "PARSER ministral"


def test_the_students_modelfile_keeps_the_template_byte_for_byte():
    st = export.student_modelfile(M.modelfile_text(), "s-q4_k_m.gguf", "ministral-3:8b", 65536)
    assert M.template_of(st) == M.TEMPLATE
    assert "FROM ./s-q4_k_m.gguf" in st and "PARSER ministral" in st and st.endswith("PARAMETER num_ctx 65536\n")


def test_the_committed_base_modelfile_is_this_template():
    if not BASE_MODELFILE.is_file():
        pytest.skip(f"{BASE_MODELFILE.relative_to(LAB)} is not in this tree yet (`python -m souschef_lab.serving_mistral "
                    "--print-modelfile` writes it)")
    text = BASE_MODELFILE.read_text(encoding="utf-8")
    assert M.template_of(text) == M.TEMPLATE
    assert [i.raw for i in export.parse_modelfile(text) if i.keyword in ("PARSER", "RENDERER")] == ["PARSER ministral"]


# --- the served-prompt smoke ------------------------------------------------------------------------------------------------

def test_the_smoke_request_is_the_conversation_the_counts_were_measured_on():
    msgs, tools = M.smoke_conversation()
    assert tools == [trainkit.hf_tool(M.SMOKE_TOOL)]
    assert M.render_prompt(msgs, tools) == (
        '[SYSTEM_PROMPT]You are a test.[/SYSTEM_PROMPT][AVAILABLE_TOOLS][{"type":"function","function":{"name":"echo",'
        '"description":"Echo the text back.","parameters":{"type":"object","required":["text"],"properties":{"text":'
        '{"type":"string"}}}}}][/AVAILABLE_TOOLS][INST]Call echo with the text hi.[/INST]')


def test_the_smoke_token_counts_on_the_pinned_tokenizers():
    try:
        from tokenizers import Tokenizer
    except ImportError:
        pytest.skip("the tokenizers package is not installed (the training env has it)")
    mdir, qfile = os.environ.get("MX_TOKENIZER_DIR"), os.environ.get("QWEN_TOKENIZER_JSON")
    if not mdir or not qfile:
        pytest.skip("set MX_TOKENIZER_DIR and QWEN_TOKENIZER_JSON to re-measure")
    msgs, tools = M.smoke_conversation()
    mx = Tokenizer.from_file(str(Path(mdir) / "tokenizer.json"))
    assert len(mx.encode(M.render_prompt(msgs, tools), add_special_tokens=True).ids) == M.SMOKE_PROMPT_TOKENS["mistral"]
    q = Tokenizer.from_file(qfile)
    assert len(q.encode(serving.render_prompt(msgs, tools), add_special_tokens=False).ids) == M.SMOKE_PROMPT_TOKENS["qwen"]


# --- the fixture itself ---------------------------------------------------------------------------------------------------------

def _template_mode(harness: Path) -> bool:
    main = harness / "main.go"
    return main.is_file() and "modelfile" in main.read_text(encoding="utf-8")


def test_fixture_is_what_the_harness_gives_now(fixture, tmp_path):
    """Re-render the captures through parity/ in template mode (Ollama 0.34.4's own code): skipped without Go or
    without a parity/ that has the -modelfile flag."""
    require = os.environ.get("SOUS_CHEF_LAB_REQUIRE_GO")
    if not serving.harness_available() or not _template_mode(serving.HARNESS):
        if require:
            pytest.fail("SOUS_CHEF_LAB_REQUIRE_GO is set but go or parity/'s template mode is missing")
        pytest.skip("go not on PATH or parity/ has no template mode: the harness is not re-run")
    mf = BASE_MODELFILE if BASE_MODELFILE.is_file() else tmp_path / "ministral-3-8b.modelfile"
    if not BASE_MODELFILE.is_file():
        mf.write_text(M.modelfile_text(), encoding="utf-8")
    try:
        now = M.served_fixture(modelfile=mf)
    except RuntimeError as e:
        if require:
            raise
        pytest.skip(f"parity/ could not build (offline toolchain?): {str(e)[:200]}")
    assert M.comparable(now) == M.comparable(fixture)
