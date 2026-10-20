"""chatfmt — the base-family switch (train.py --family qwen|mistral), and that the default is main, byte for byte.

The qwen family is today's objects (the same function objects serving.py and train.py define). Two goldens captured
from the untouched 1aa803a checkout pin the default path: the dry-run plan text and plan dict
(fixtures/mistral/qwen-dryrun-1aa803a.json) and the training text of every real capture plus the generation-check
items (fixtures/mistral/qwen-render-1aa803a.json). The mistral family resolves its base and render, records family
and revision in the plan, warm-starts only from a LoRA trained on its own base (chocolate-muffin's; r4's and r6's are
Qwen3.5 LoRAs), and reads its calls back as Ollama's ministral parser does."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from souschef_lab import chatfmt, serving, serving_mistral, train, trainkit  # noqa: E402
import test_train_dry as TD  # noqa: E402  (the dataset rows the dry-run tests build)
import test_serving as TS  # noqa: E402  (the captures as the dataset builds them)

GOLD = LAB / "fixtures" / "mistral"


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def test_qwen_is_todays_objects():
    f = chatfmt.get("qwen")
    assert chatfmt.get() == f and chatfmt.DEFAULT == "qwen"
    assert f.render_training is serving.render_training and f.render_prompt is serving.render_prompt
    assert f.generation_points is serving.generation_points and f.unservable is serving.unservable
    assert f.parse_tool_calls is train.parse_tool_calls and f.end_of_turn == serving.IM_END
    assert f.description == serving.DESCRIPTION and f.base == train.DEFAULT_BASE == train.RECIPE["base"]
    assert (f.revision, f.prompt_add_special_tokens, f.warm_start) == (None, False, True)
    assert (f.hf_cache_dir, f.base_tar, f.base_modelfile, f.base_ollama_tag) == (
        "models--unsloth--Qwen3.5-9B", "unsloth-qwen3.5-9b-hf.tar", "base-modelfiles/qwen3.5-9b.modelfile", "qwen3.5:9b")


def test_mistral_is_the_arm():
    f = chatfmt.get("mistral")
    assert f.base == "mistralai/Ministral-3-8B-Instruct-2512-BF16"
    assert f.revision == "f6fae9795746f63c9be8344932f01275f3c63734"
    assert f.hf_cache_dir == "models--mistralai--Ministral-3-8B-Instruct-2512-BF16"
    assert f.base_tar == "mistralai-ministral-3-8b-instruct-2512-bf16-hf.tar"
    assert f.base_modelfile == "base-modelfiles/ministral-3-8b.modelfile" and f.base_ollama_tag == "ministral-3:8b"
    assert f.render_training is serving_mistral.render_training and f.end_of_turn == "</s>"
    assert f.parse_tool_calls is serving_mistral.parse_tool_calls is train.parse_tool_calls_mistral
    assert (f.prompt_add_special_tokens, f.warm_start) == (True, True)
    assert f.special_ids == serving_mistral.SPECIAL_IDS


def test_an_unknown_family_raises():
    with pytest.raises(ValueError):
        chatfmt.get("deepseek")
    with pytest.raises(SystemExit):
        train.parse_args(["--family", "deepseek"])


# --- the default path is main's, byte for byte ---------------------------------------------------------------------------

def _dataset(d: Path) -> Path:
    TD.write_jsonl(d / "train.jsonl", [TD.a_row(f"c{i}#1", with_history=(i == 2)) for i in range(5)])
    TD.write_jsonl(d / "val.jsonl", [TD.a_row("v0#1")])
    (d / "dataset-manifest.json").write_text(json.dumps({"manifest_hash": "727d151b" + "0" * 56}), encoding="utf-8")
    return d


def _plan_text(stdout: str) -> str:
    """The plan lines (the 'extra not installed' line depends on the environment, not the code)."""
    return "\n".join(ln for ln in stdout.splitlines() if "[train] extra" not in ln)


def test_the_qwen_dry_run_is_main_byte_for_byte(tmp_path):
    gold = json.loads((GOLD / "qwen-dryrun-1aa803a.json").read_text(encoding="utf-8"))
    d = _dataset(tmp_path / "dataset")
    argv = [a.replace("<TMP>", str(tmp_path)) for a in gold["argv"]]
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert train.main(argv) == gold["rc"] == 0
    norm = lambda s: s.replace(str(tmp_path), "<TMP>")  # noqa: E731
    assert _plan_text(norm(buf.getvalue())) == _plan_text(gold["stdout"])
    args = train.parse_args(argv)
    p = train.plan(args, trainkit.load_split(d, "train"), trainkit.load_split(d, "val"))
    assert json.loads(norm(json.dumps(p, ensure_ascii=False))) == gold["plan"]
    assert "family" not in p and "revision" not in p and args.family == "qwen" and args.base == "unsloth/Qwen3.5-9B"


def test_the_qwen_render_and_gen_check_are_main_byte_for_byte():
    gold = json.loads((GOLD / "qwen-render-1aa803a.json").read_text(encoding="utf-8"))
    rows = []
    for w in TS.CAPTURES:
        hf = TS.hf_row(w)
        rows.append(dict(hf, id=w.name, meta={}))
        t = train.render_text(hf["messages"], hf["tools"])
        assert t == train.render_text(hf["messages"], hf["tools"], 0, chatfmt.get("qwen")) == \
            serving.render_training(hf["messages"], hf["tools"])
        assert {"render_text": sha(t), "chars": len(t)} == gold["captures"][w.name], w.name
    items = train.gen_check_items(rows, 32)
    assert len(items) == gold["gen_check_n"]
    assert sha(json.dumps(items, ensure_ascii=False, sort_keys=True)) == gold["gen_check_items"]
    assert items == train.gen_check_items(rows, 32, fmt=chatfmt.get("qwen"))


# --- the mistral family through train.py ------------------------------------------------------------------------------------

def test_the_mistral_dry_run_resolves_the_base_render_and_records_family(tmp_path, capsys):
    d = _dataset(tmp_path / "dataset")
    argv = ["--data", str(d), "--name", "mx", "--output", str(tmp_path / "o"), "--fields-context", "off",
            "--family", "mistral", "--dry-run"]
    assert train.main(argv) == 0
    out = capsys.readouterr().out
    assert "mistralai/Ministral-3-8B-Instruct-2512-BF16" in out and "f6fae9795746f63c9be8344932f01275f3c63734" in out
    assert 'unservable rows by rule {"qwen_echo": 0, "mistral": 0}' in out
    args = train.parse_args(argv)
    p = train.plan(args, trainkit.load_split(d, "train"), trainkit.load_split(d, "val"))
    assert (p["base"], p["family"], p["revision"], p["render"]) == (
        chatfmt.MISTRAL_BASE, "mistral", chatfmt.MISTRAL_REVISION, serving_mistral.DESCRIPTION)
    other = train.plan(train.parse_args(argv + ["--base", "somewhere/else"]), [], [])
    assert other["revision"] is None  # the pinned revision is recorded only for the pinned base


def _adapter(d: Path, base: str) -> Path:
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 16, "lora_alpha": 16,
                                                       "base_model_name_or_path": base}), encoding="utf-8")
    (d / "adapter_model.safetensors").write_bytes(b"w")
    return d


def test_the_mistral_family_warm_starts_only_from_a_ministral_lora(tmp_path, capsys):
    """chocolate-muffin's adapter (a LoRA on the Ministral base) starts a mistral warm arm; r4's / r6's (Qwen3.5
    LoRAs) are refused by the dry plan, before any GPU; a Ministral LoRA cannot start a qwen run either."""
    d = _dataset(tmp_path / "dataset")
    flags = ["--data", str(d), "--fields-context", "off", "--dry-run"]
    cm = _adapter(tmp_path / "cm-r7" / "adapter", chatfmt.MISTRAL_BASE)
    assert train.main([*flags, "--family", "mistral", "--init-adapter", str(cm), "--lr", "5e-5"]) == 0
    p = train.plan(train.parse_args([*flags, "--family", "mistral", "--init-adapter", str(cm)]), [], [])
    assert p["start"] == "init_adapter" and p["init_adapter"]["base"] == chatfmt.MISTRAL_BASE
    capsys.readouterr()
    r4 = _adapter(tmp_path / "r4-sft" / "adapter", "unsloth/Qwen3.5-9B")
    with pytest.raises(SystemExit):
        train.main([*flags, "--family", "mistral", "--init-adapter", str(r4)])
    assert f"was trained on unsloth/Qwen3.5-9B, this run's base is {chatfmt.MISTRAL_BASE}" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        train.main([*flags, "--init-adapter", str(cm)])
    assert f"was trained on {chatfmt.MISTRAL_BASE}, this run's base is unsloth/Qwen3.5-9B" in capsys.readouterr().err


def test_family_is_a_config_key(tmp_path):
    cfg = tmp_path / "s.json"
    cfg.write_text(json.dumps({"family": "mistral"}), encoding="utf-8")
    a = train.parse_args(["--config", str(cfg)])
    assert a.family == "mistral" and a.base == chatfmt.MISTRAL_BASE
    assert train.parse_args(["--config", str(cfg), "--family", "qwen"]).base == "unsloth/Qwen3.5-9B"


def test_the_mistral_gen_check_items_and_pass():
    rows = [dict(TD.a_row("v0#1"), id="v0#1"), dict(TD.a_row("v1#1"), id="v1#1"),
            dict(TD.a_row("v2#1", with_history=True), id="v2#1")]
    fmt = chatfmt.get("mistral")
    items = train.gen_check_items(rows, 5, fmt=fmt)
    assert [i["id"] for i in items] == ["v0#1", "v1#1"]
    it = items[0]
    assert it["prompt"].endswith("[/INST]") and it["prompt"].count("[INST]") == 1 and "<s>" not in it["prompt"]
    assert it["expected"] == [("set_fields", {"target": {"kind": "section", "page": "home", "sectionId": "heroBanner"},
                                              "values": {"heading": "Fresh cookies, daily — café"}})]
    full = train.render_text(rows[0]["messages"][:3], rows[0]["tools"], 0, fmt)
    assert full.startswith(it["prompt"])
    generated = full[len(it["prompt"]):]
    assert train.gen_check_pass(generated, it["expected"], fmt)
    reordered = ('[TOOL_CALLS]set_fields[ARGS]{"values": {"heading": "Fresh cookies, daily — café"}, '
                 '"target": {"sectionId": "heroBanner", "page": "home", "kind": "section"}}</s>junk[TOOL_CALLS]x[ARGS]{}')
    assert train.gen_check_pass(reordered, it["expected"], fmt)  # key order does not count; text after </s> ignored
    assert not train.gen_check_pass("Done.</s>", it["expected"], fmt)
    assert train.gen_check_pass("Which section?</s>", [], fmt)


def test_parse_tool_calls_mistral_round_trips_rendered_calls():
    calls = [trainkit.render_tool_call("a", "edit_list", {"target": {"kind": "page", "page": "home"}, "list": "s",
                                                          "item": {"nested": [1, {"x": "a<b"}]}, "action": "add"}),
             trainkit.render_tool_call("b", "reset", {"keys": ["x"], "target": {"kind": "global"}})]
    msgs = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "On it.", "tool_calls": calls}]
    seg = serving_mistral.assistant_segments(msgs, None)[0]
    assert train.parse_tool_calls_mistral(seg) == [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                                                   for c in calls]
