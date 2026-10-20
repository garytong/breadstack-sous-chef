"""mx_hf_golden — record Mistral's OWN chat template's render of the lab's conversations (fixtures/mistral/hf-golden.json).

serving_mistral writes the Mistral arm's served form through a Go TEMPLATE of our own. This golden proves that the
template's LAYOUT is the model's: the same renderer in style="python" (json.dumps for tools and dict arguments, "<s>"
first) must equal `transformers.apply_chat_template` with the pinned chat_template.jinja of
mistralai/Ministral-3-8B-Instruct-2512-BF16 @ f6fae97 on every case — the real captures (fixtures/prompt/real, rows
built the way the dataset builds them) and synthetic edge cases (escapes, parallel calls, a second user turn, odd
arguments, odd tools). tests/test_serving_mistral.py checks the recorded goldens without transformers.

Offline only: the tokenizer files are read from MX_TOKENIZER_DIR (chat_template.jinja, tokenizer.json,
tokenizer_config.json, special_tokens_map.json at the pinned revision); nothing is downloaded and the 17 MB tokenizer is
never committed. Run it in the training env:

  HF_HUB_OFFLINE=1 MX_TOKENIZER_DIR=<folder> lab/.venv-train/bin/python -m souschef_lab.mx_hf_golden --write
  ... --check   # re-render and compare (the test does this live when transformers is importable)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import serving_mistral as M
from .chatfmt import MISTRAL_BASE, MISTRAL_REVISION
from .serving import CAPTURES, LAB_DIR, captures, read_jsonl, sha

FIX = LAB_DIR / "fixtures" / "prompt"
GOLDEN = M.HF_GOLDEN


def capture_rows(root: Path = CAPTURES) -> list[tuple[str, dict]]:
    """(capture name, the row as dataset.to_hf builds it: messages + tools) for every real capture."""
    from . import dataset as D
    system = ((FIX / "system0.rules.txt").read_text(encoding="utf-8"),
              (FIX / "system1.index.txt").read_text(encoding="utf-8"))
    tools = json.loads((FIX / "tools.json").read_text(encoding="utf-8"))
    out = []
    for where in captures(root):
        case = json.loads((where / "case.json").read_text(encoding="utf-8"))
        row = read_jsonl(where / "collect-real.jsonl")[0]
        hf = D.to_hf(row, case, system=system, tools=tools)
        out.append((where.name, {"messages": hf["messages"], "tools": hf["tools"]}))
    return out


def _call(cid: str, name: str, args) -> dict:
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args if isinstance(args, str)
                                                        else json.dumps(args, ensure_ascii=False)}}


def _tool(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


SET_FIELDS = _tool("set_fields", "Write <field> values & more.", {
    "type": "object", "properties": {"target": {"type": "object", "description": "where"},
                                     "values": {"type": "object", "description": "what"}},
    "required": ["target", "values"]})


def synthetic_cases() -> list[tuple[str, dict]]:
    """The edge cases the captures do not hold."""
    sys_ = {"role": "system", "content": "You are Sous Chef. Use <tools> & answer briefly > now."}
    return [
        ("s1-text-reply-escapes", {"tools": [SET_FIELDS], "messages": [
            sys_, {"role": "user", "content": "What can you do?"},
            {"role": "assistant", "content": "I edit your site: <b>fields</b> & sections."}]}),
        ("s2-parallel-calls", {"tools": [SET_FIELDS], "messages": [
            sys_, {"role": "user", "content": "Set both headings"},
            {"role": "assistant", "content": "", "tool_calls": [
                _call("a", "set_fields", {"target": {"kind": "section", "sectionId": "hero"}, "values": {"heading": "A"}}),
                _call("b", "set_fields", {"values": {"heading": "B"}, "target": {"kind": "section", "sectionId": "faq"}})]},
            {"role": "tool", "tool_call_id": "a", "name": "set_fields", "content": '{"ok":true}'},
            {"role": "tool", "tool_call_id": "b", "name": "set_fields", "content": '{"ok":true,"written":["x"]}'},
            {"role": "assistant", "content": "Both headings are set."}]}),
        ("s3-text-and-call", {"tools": [SET_FIELDS], "messages": [
            sys_, {"role": "user", "content": "Rename the hero"},
            {"role": "assistant", "content": "Renaming it now.", "tool_calls": [
                _call("c", "set_fields", {"target": {"kind": "section", "sectionId": "hero"}, "values": {"title": "Hi"}})]},
            {"role": "tool", "tool_call_id": "c", "name": "set_fields", "content": '{"ok":true}'},
            {"role": "assistant", "content": "Done."}]}),
        ("s4-second-user-turn", {"tools": [SET_FIELDS], "messages": [
            sys_, {"role": "user", "content": "earlier ask"}, {"role": "assistant", "content": "earlier reply"},
            {"role": "user", "content": "now this"},
            {"role": "assistant", "content": "", "tool_calls": [_call("d", "set_fields", {"target": {}, "values": {}})]},
            {"role": "tool", "tool_call_id": "d", "name": "set_fields", "content": "ok"},
            {"role": "assistant", "content": "Done again."}]}),
        ("s5-odd-arguments", {"tools": [SET_FIELDS], "messages": [
            sys_, {"role": "user", "content": "odd values"},
            {"role": "assistant", "content": "", "tool_calls": [_call("e", "set_fields", {
                "values": {"text": "café x<y&z", "n": 1e6, "none": None, "nested": [[1, 2], {"k": []}], "empty": {}},
                "target": {"kind": "global"}}), _call("f", "set_fields", "")]},
            {"role": "tool", "tool_call_id": "e", "name": "set_fields", "content": "ok"},
            {"role": "tool", "tool_call_id": "f", "name": "set_fields", "content": "ok"},
            {"role": "assistant", "content": "Written."}]}),
        ("s6-odd-tools", {"tools": [
            _tool("pick", "", {"type": "object", "properties": {"z": {"type": "string"}, "a": {"type": "integer"}},
                               "required": ["a"], "additionalProperties": False}),
            _tool("ping", "", {"type": "object"})], "messages": [
            sys_, {"role": "user", "content": "pick one"},
            {"role": "assistant", "content": "", "tool_calls": [_call("g", "pick", {"a": 2})]},
            {"role": "tool", "tool_call_id": "g", "name": "pick", "content": "picked"},
            {"role": "assistant", "content": "Picked."}]}),
    ]


def all_cases(root: Path = CAPTURES) -> list[tuple[str, dict]]:
    return [(f"capture:{n}", c) for n, c in capture_rows(root)] + [(f"synthetic:{n}", c) for n, c in synthetic_cases()]


def load_tokenizer(tdir: Path):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer  # the training env only
    tok = AutoTokenizer.from_pretrained(str(tdir), local_files_only=True)
    template = (Path(tdir) / "chat_template.jinja").read_text(encoding="utf-8")
    return tok, template


def hf_render(tok, template: str, case: dict) -> str:
    return tok.apply_chat_template(case["messages"], tools=case["tools"] or None, chat_template=template,
                                   tokenize=False)


def build(tdir: Path, root: Path = CAPTURES) -> dict:
    tok, template = load_tokenizer(tdir)
    measured = {t: tok(t, add_special_tokens=False)["input_ids"] for t in M.SPECIAL_IDS}
    cases = all_cases(root)
    rendered = [(name, hf_render(tok, template, c)) for name, c in cases]
    caps = [t for n, t in rendered if n.startswith("capture:")]
    prefix = os.path.commonprefix(caps) if caps else ""
    first = cases[0][1]
    ids = tok(M.render_training(first["messages"], first["tools"]), add_special_tokens=True)["input_ids"]
    return {
        "what": "transformers.apply_chat_template (Mistral's own chat_template.jinja, tokenize=False) of the lab's "
                "conversations: python -m souschef_lab.mx_hf_golden --write",
        "model": MISTRAL_BASE, "revision": MISTRAL_REVISION, "chat_template_sha256": sha(template),
        "transformers": __import__("transformers").__version__,
        "special_ids": {t: v for t, v in measured.items()},
        "token_count": {"case": cases[0][0], "served_training_text_with_bos": len(ids)},
        "capture_prefix": prefix,
        "cases": [{"name": name, **({} if name.startswith("capture:") else {"messages": c["messages"], "tools": c["tools"]}),
                   "chars": len(text), "sha256": sha(text),
                   "text" if not name.startswith("capture:") else "tail": text if not name.startswith("capture:")
                   else text[len(prefix):]}
                  for (name, c), (_, text) in zip(cases, rendered)],
    }


def golden_texts(golden: dict) -> dict[str, str]:
    """case name → the recorded HF render, checked against its sha256."""
    out = {}
    for c in golden["cases"]:
        text = c["text"] if "text" in c else golden["capture_prefix"] + c["tail"]
        if len(text) != c["chars"] or sha(text) != c["sha256"]:
            raise ValueError(f"{c['name']}: the golden's text does not match its sha256")
        out[c["name"]] = text
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help=f"write {GOLDEN.relative_to(LAB_DIR)}")
    ap.add_argument("--check", action="store_true", help="re-render and compare with the golden (the default)")
    ap.add_argument("--tokenizer-dir", type=Path, default=os.environ.get("MX_TOKENIZER_DIR"),
                    help="the pinned tokenizer files (default: $MX_TOKENIZER_DIR)")
    args = ap.parse_args(argv)
    if not args.tokenizer_dir or not Path(args.tokenizer_dir).is_dir():
        print("error: set MX_TOKENIZER_DIR (or --tokenizer-dir) to the pinned tokenizer files", file=sys.stderr)
        return 2
    g = build(Path(args.tokenizer_dir))
    if g["special_ids"] != {t: [i] for t, i in M.SPECIAL_IDS.items()}:
        print(f"error: the tokenizer's ids are not serving_mistral.SPECIAL_IDS: {g['special_ids']}", file=sys.stderr)
        return 1
    mine = {name: M.render_conversation(c["messages"], c["tools"], style="python", bos=True)[0]
            for name, c in all_cases()}
    texts = golden_texts(g)
    bad = [n for n in texts if texts[n] != mine[n]]
    for n in bad:
        print(f"{n}: {M.first_difference(mine[n], texts[n])}", file=sys.stderr)
    print(f"{len(texts) - len(bad)}/{len(texts)} cases: serving_mistral (python style) == apply_chat_template")
    if args.write:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(g, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {GOLDEN.relative_to(LAB_DIR)}")
        return 1 if bad else 0
    try:
        old = json.loads(GOLDEN.read_text(encoding="utf-8"))
    except OSError:
        print(f"error: no golden at {GOLDEN.relative_to(LAB_DIR)}: run with --write", file=sys.stderr)
        return 1
    same = golden_texts(old) == texts and old["chat_template_sha256"] == g["chat_template_sha256"]
    print("The golden is what the template gives." if same else "error: the golden differs — re-run with --write")
    return 0 if same and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
