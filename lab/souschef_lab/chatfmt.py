"""chatfmt — the base-model family switch (`train.py --family qwen|mistral`; the pipeline's `base_family`).

A family is everything that depends on the base model's chat format: the Hugging Face id (and revision) of the base,
its hub-cache folder and base tar, the base Modelfile the export copies, the serving-parity render (training text,
served prompt, generation points, the end-of-turn marker), the rows its served form cannot carry, how a reply's tool
calls are read back, whether a prompt is tokenized with the tokenizer's own special tokens (BOS), and whether a warm
start exists — plus, for the pipeline, what the served model looks like (SERVED: the staged Modelfile's TEMPLATE /
RENDERER / PARSER and temperature, the GGUF architecture Ollama reports).

`qwen` is today's objects, unchanged — the same function objects serving.py and train.py define, so `--family qwen`
(the default) is byte-identical to main (tests/test_chatfmt.py). `mistral` is chocolate-muffin's base
(mistralai/Ministral-3-8B-Instruct-2512-BF16 @ f6fae97, served through serving_mistral's TEMPLATE + PARSER ministral):
the Mistral arm's family, and the retrain pipeline's when infra config `base_family` / `trigger.base_family` says so.
A mistral warm start continues a Ministral LoRA (chocolate-muffin's, base/adapters/cm-r7); train.check_init_adapter
refuses an adapter trained on another base (r4's and r6's are Qwen3.5 LoRAs).

    python -m souschef_lab.chatfmt check-stage <staged dir> --family mistral   # the trainer's stage check

Standard library only.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import serving, serving_mistral


@dataclass(frozen=True)
class Format:
    name: str
    base: str
    revision: str | None
    hf_cache_dir: str
    base_tar: str
    base_modelfile: str  # relative to lab/
    base_ollama_tag: str
    description: str
    render_training: Callable
    render_prompt: Callable
    generation_points: Callable
    end_of_turn: str
    unservable: Callable
    parse_tool_calls: Callable
    special_tokens: tuple  # the markers the render writes that the base's tokenizer must hold as single tokens
    special_ids: dict | None  # … and, when known, the id each must map to
    prompt_add_special_tokens: bool  # tokenize prompts / count lengths with the tokenizer's BOS
    warm_start: bool  # an --init-adapter / warm: arm can exist for this base (from a LoRA trained on this base)


FAMILIES = ("qwen", "mistral")
DEFAULT = "qwen"

MISTRAL_BASE = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
MISTRAL_REVISION = "f6fae9795746f63c9be8344932f01275f3c63734"


def get(name: str | None = None) -> Format:
    """The family's Format; None is the default (qwen). ValueError for any other name."""
    name = DEFAULT if name is None else str(name)
    if name == "qwen":
        from . import train  # train imports chatfmt at module level: resolve its parse_tool_calls lazily
        return Format(
            name="qwen", base=train.DEFAULT_BASE, revision=None, hf_cache_dir="models--unsloth--Qwen3.5-9B",
            base_tar="unsloth-qwen3.5-9b-hf.tar", base_modelfile="base-modelfiles/qwen3.5-9b.modelfile",
            base_ollama_tag="qwen3.5:9b", description=serving.DESCRIPTION, render_training=serving.render_training,
            render_prompt=serving.render_prompt, generation_points=serving.generation_points,
            end_of_turn=serving.IM_END, unservable=serving.unservable, parse_tool_calls=train.parse_tool_calls,
            special_tokens=serving.SPECIAL_TOKENS, special_ids=None, prompt_add_special_tokens=False, warm_start=True)
    if name == "mistral":
        return Format(
            name="mistral", base=MISTRAL_BASE, revision=MISTRAL_REVISION,
            hf_cache_dir="models--" + MISTRAL_BASE.replace("/", "--"),
            base_tar="mistralai-ministral-3-8b-instruct-2512-bf16-hf.tar",
            base_modelfile="base-modelfiles/ministral-3-8b.modelfile", base_ollama_tag="ministral-3:8b",
            description=serving_mistral.DESCRIPTION, render_training=serving_mistral.render_training,
            render_prompt=serving_mistral.render_prompt, generation_points=serving_mistral.generation_points,
            end_of_turn=serving_mistral.END_OF_TURN, unservable=serving_mistral.unservable,
            parse_tool_calls=serving_mistral.parse_tool_calls, special_tokens=serving_mistral.SPECIAL_TOKENS,
            special_ids=dict(serving_mistral.SPECIAL_IDS), prompt_add_special_tokens=True, warm_start=True)
    raise ValueError(f"unknown base family {name!r}: one of {', '.join(FAMILIES)}")


def check(name) -> str:
    """A family name as a setting gives it (infra config base_family, trigger.base_family, train-config.json family,
    BASE_FAMILY, state/model.json family): None / "" is the default (qwen); anything not in FAMILIES is ValueError."""
    if name is None or (isinstance(name, str) and not name.strip()):
        return DEFAULT
    if not isinstance(name, str) or name.strip() not in FAMILIES:
        raise ValueError(f"base family {name!r}: one of {', '.join(FAMILIES)}")
    return name.strip()


def resolve(*settings) -> str:
    """The first setting that is set (not None / ""), checked; the default when none is."""
    for s in settings:
        if s is not None and not (isinstance(s, str) and not s.strip()):
            return check(s)
    return DEFAULT


# --- The served model (the pipeline: the staged Modelfile, the model image, the eval and drift routes) ---------------------
# Readable without get() (which imports train). The staged Modelfile's shape is r7's (the live blueberry-muffin's,
# runs/rt-20261018-0213-b02e95-3914e2e4/train/stage/Modelfile): FROM ./<gguf>, the base's TEMPLATE / RENDERER / PARSER,
# top_k 20, top_p 0.95, temperature, presence_penalty 0, num_ctx 65536 (`make model-stage MODEL_TEMPERATURE=<t>
# MODEL_PRESENCE_PENALTY=0`). A mistral stage differs in the TEMPLATE (serving_mistral.TEMPLATE, byte for byte), PARSER
# ministral with no RENDERER, and temperature 0.15 (Mistral's recommended range is <= 0.15; the r7 arm scored 0.15
# slightly above 0.3). `architecture` is the GGUF's general.architecture, which Ollama's /api/show reports.

SERVED = {
    "qwen": {"architecture": "qwen35", "parser": "qwen3.5", "renderer": "qwen3.5", "temperature": "0.3",
             "template": "{{ .Prompt }}"},
    "mistral": {"architecture": "mistral3", "parser": "ministral", "renderer": None, "temperature": "0.15",
                "template": serving_mistral.TEMPLATE},
}
STAGE_PARAMETERS = (("top_k", "20"), ("top_p", "0.95"), ("presence_penalty", "0"), ("num_ctx", "65536"))

# int's two in-house routes and the family whose model each one's settings are for (infra/envs/int: gpu_model_route
# is blueberry-muffin, gpu_model_mistral_route chocolate-muffin; harness.FAMILY_ROUTES holds the settings). The drift
# poll reads int's /version through it: the route the in-house traffic is answered by (after MODEL_ROUTE_ALIASES)
# must be the live model's family's. Any other route (Claude's, after the break-glass) has no family.
ROUTE_FAMILIES = {"blueberry-muffin": "qwen", "chocolate-muffin": "mistral"}


def stage_temperature(family: str | None) -> str:
    """The served temperature the trainer stages for this family (make model-stage MODEL_TEMPERATURE)."""
    return SERVED[check(family)]["temperature"]


def of_architecture(arch) -> str | None:
    """The family of a GGUF architecture (Ollama /api/show model_info general.architecture), None if unknown."""
    return next((f for f, s in SERVED.items() if s["architecture"] == arch), None)


def _instructions(text: str) -> list:
    from .export import parse_modelfile  # export imports nothing heavy at module level
    return parse_modelfile(text or "")


def modelfile_facts(text: str) -> dict:
    """{parser, renderer, template, parameters: {name: [values]}, from: [values]} of a Modelfile's text."""
    out: dict = {"parser": None, "renderer": None, "template": None, "parameters": {}, "from": []}
    for ins in _instructions(text):
        if ins.keyword in ("PARSER", "RENDERER"):
            parts = ins.raw.split(None, 1)
            out[ins.keyword.lower()] = parts[1].strip() if len(parts) > 1 else ""
        elif ins.keyword == "TEMPLATE":
            out["template"] = serving_mistral.template_of(ins.raw)
        elif ins.keyword == "PARAMETER":
            parts = ins.raw.split(None, 2)
            if len(parts) == 3:
                out["parameters"].setdefault(parts[1], []).append(parts[2].strip())
        elif ins.keyword == "FROM":
            out["from"].append(ins.raw.split(None, 1)[1].strip() if len(ins.raw.split(None, 1)) > 1 else "")
    return out


def of_modelfile(text: str) -> str | None:
    """The family a Modelfile serves, by its PARSER (ministral → mistral, qwen3.5 → qwen); None when it names none."""
    parser = modelfile_facts(text)["parser"]
    return next((f for f, s in SERVED.items() if parser and s["parser"] == parser), None)


def stage_problems(text: str, family: str | None) -> list[str]:
    """What keeps a staged Modelfile (train/stage/Modelfile, the model image's) from being this family's served model:
    one FROM ./<gguf>; the family's TEMPLATE byte for byte (no trailing newline before the closing quotes), PARSER and
    RENDERER (mistral: none); and exactly once each of temperature (SERVED), top_k 20, top_p 0.95, presence_penalty 0,
    num_ctx 65536 — r7's limits. [] when it is."""
    fam = check(family)
    want = SERVED[fam]
    f = modelfile_facts(text)
    problems = []
    if len(f["from"]) != 1 or not re.fullmatch(r"\./[^/\s]+\.gguf", f["from"][0] if f["from"] else ""):
        problems.append(f"FROM {f['from']}: want one FROM ./<file>.gguf")
    if f["template"] != want["template"]:
        problems.append(f"TEMPLATE is not the {fam} family's ({len(f['template'] or '')} characters, want "
                        f"{len(want['template'])})")
    if f["parser"] != want["parser"]:
        problems.append(f"PARSER {f['parser']!r}, want {want['parser']!r}")
    if f["renderer"] != want["renderer"]:
        problems.append(f"RENDERER {f['renderer']!r}, want {want['renderer']!r}")
    for name, value in (("temperature", want["temperature"]),) + STAGE_PARAMETERS:
        got = f["parameters"].get(name) or []
        if got != [value]:
            problems.append(f"PARAMETER {name} {got or 'missing'}, want [{value!r}]")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="souschef_lab.chatfmt", description="The base-family switch's checks.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-stage", help="a staged model directory (make model-stage's) is the family's served model")
    c.add_argument("stage", type=Path, help="the staging directory (Modelfile + one GGUF)")
    c.add_argument("--family", required=True, choices=FAMILIES)
    a = ap.parse_args(argv)
    mf = a.stage / "Modelfile"
    if not mf.is_file():
        print(f"check-stage: no Modelfile in {a.stage}", file=sys.stderr)
        return 1
    problems = stage_problems(mf.read_text(encoding="utf-8"), a.family)
    if problems:
        print(f"check-stage: {mf} is not a {a.family} served model: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"check-stage: {mf} is the {a.family} family's served model (PARSER {SERVED[a.family]['parser']}, "
          f"temperature {SERVED[a.family]['temperature']}, num_ctx 65536)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
