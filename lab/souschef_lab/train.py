"""SFT: fine-tune the student on the verified Sous Chef turns (LoRA, Unsloth).

The recipe is the thermometer's round-1 9B recipe, copied verbatim
(docs/LAB.md §8; breadstack-thermometer/lab/in-house-model/train.py is the
pattern, never imported): unsloth/Qwen3.5-9B in bf16, LoRA r16 α16, lr 2e-4,
one epoch, micro-batch 1 × grad_accum 4, rows up to 32,768 tokens (longer
rows are dropped and counted, never cut), adamw_8bit, linear decay after a
3% warm-up, weight decay 0.01, packing off. Only the assistant's own turns
are trained on — the ops it emitted and the reply it wrote: the system
prompt, the merchant's request and the tool results are context, masked out
of the loss by train_on_responses_only on ChatML's markers (and a follow-up's
resent earlier replies by mask_history_turns). The run stops before training
if a tool result would be trained on.

Data: data/dataset/train.jsonl and val.jsonl (dataset.py), rows in the
thermometer's to_hf shape (trainkit.check_rows). Settings: flags, or
--config settings.yaml|json (a flag on the command line wins).

  python -m souschef_lab.train --dry-run                     # the plan and the row counts; no GPU, no torch
  python -m souschef_lab.train --name sous-chef-9b-v1        # one epoch
  python -m souschef_lab.train --config lab/train.yaml --max-steps 4 --name smoke

Seeds and warm starts (r6; the defaults are r4's: seed 3407, a fresh LoRA on the base):

  python -m souschef_lab.train --seed 2 --name r6-s2              # another SGD path on the same data
  python -m souschef_lab.train --init-adapter runs/adapters/r4-sft/adapter --lr 5e-5 --name r6-warm
  python -m souschef_lab.train --gen-check 32 ...                 # + first-call match on 32 fixed val rows
  python -m souschef_lab.train select runs/adapters/r6-s3407 runs/adapters/r6-s2 --into runs/adapters/r6

--seed changes the LoRA init and the data order; the validation subset stays
fixed (--val-seed, 3407) so the seeds' losses compare. --init-adapter loads
an adapter's LoRA weights into the fresh LoRA before the first update (same
rank and alpha). `select` keeps the best run (gen-check first when every
run has one, then the lowest final validation loss) and records every
run's scores in the kept run.json. tools/pipeline-train.sh runs the loop on
Vertex: TRAIN_ARMS (base:<seed> from the base, warm:<seed> from
INIT_ADAPTER, e.g. base:3407,base:2,warm:3407 — from-base seeds and a warm
start in the same round, the best on validation kept) or TRAIN_SEEDS.

--dry-run never imports torch (tests/test_train_dry.py checks). Training
needs the [train] extra: `make -C lab train-env` (uv, pinned to the thermometer's measured set, cu130 wheels; several GB, once); the
first run downloads the base from Hugging Face.

Afterwards: runs/adapters/<name>/adapter (the LoRA), run.json (settings,
counts, losses, speed), loss.jsonl (one line per update), sample_rendered.txt
and sample_trained.txt (what the model saw, and the part it learned). Then
dpo.py continues the adapter on its own preference pairs and export.py turns
it into an Ollama model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

from . import chatfmt, fields, serving, serving_mistral, trainkit
from .trainkit import (MARKERS, check_rows, die, estimate_tokens, history_turns, leaked_tool_results, load_split,
                       mask_history_turns, say, template_kind, trained_spans, updates_for, utc_now, utc_stamp,
                       val_subset, weighted_loss)

DEFAULT_BASE = "unsloth/Qwen3.5-9B"
TILED_ABOVE = 24576  # Unsloth's tiled MLP: on above this many tokens (the thermometer's measured 32k fit)

# The recipe (docs/LAB.md §8); a flag or a --config value changes one of them.
RECIPE = {
    "base": DEFAULT_BASE, "max_seq_length": 32768, "epochs": 1.0, "lr": 2e-4, "rank": 16, "alpha": 16, "batch": 1,
    "grad_accum": 4, "optim": "adamw_8bit", "warmup_ratio": 0.03, "weight_decay": 0.01, "eval_steps": 50,
    "save_steps": 0, "val_max_rows": None, "tiled_mlp": "auto", "seed": 3407, "thinking": False,
}

CONFIG_KEYS = tuple(RECIPE) + ("data", "name", "output", "max_steps", "instruction_part", "response_part",
                                "fields_context", "checkpoint_steps", "resume", "init_adapter", "val_seed", "gen_check",
                                "family")

# The validation subset (--val-max-rows) and the generation check's rows are drawn with this seed whatever --seed
# says, so every seed of a round is scored on the same rows (r4 drew its subset with its own --seed, 3407: the same).
VAL_SEED = 3407
GEN_CHECK_MAX_NEW_TOKENS = 512
GEN_CHECK_BUDGET_S = 1200  # a generation check stops after this long (and then counts as incomplete)
WARM_LR = 5e-5  # the suggested --lr for a warm start (tools/pipeline-train.sh INIT_LR's default)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="souschef_lab.train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="settings file (.yaml/.yml/.json) whose keys are these flags; a flag given here wins")
    p.add_argument("--base", default=None,
                   help=f"Hugging Face id of the base model (default: the --family's base, {RECIPE['base']} for qwen)")
    p.add_argument("--family", choices=chatfmt.FAMILIES, default=chatfmt.DEFAULT,
                   help="the base model's chat-format family: qwen (Ollama's qwen3.5 renderer, the default) or mistral "
                        "(Ministral 3, Ollama's TEMPLATE + PARSER ministral; serving_mistral.py) (default: %(default)s)")
    p.add_argument("--data", type=Path, default=trainkit.DATASET_DIR,
                   help="folder with train.jsonl / val.jsonl (default: lab/data/dataset)")
    p.add_argument("--name", help="run name; output goes to runs/adapters/<name> (default: <base>-<UTC time>)")
    p.add_argument("--output", type=Path, help="output folder (overrides --name)")
    p.add_argument("--max-seq-length", type=int, default=RECIPE["max_seq_length"],
                   help="longest example in tokens; longer ones are dropped and counted, never cut (default: %(default)s)")
    p.add_argument("--epochs", type=float, default=RECIPE["epochs"], help="passes over the training set (default: %(default)s)")
    p.add_argument("--lr", type=float, default=RECIPE["lr"], help="learning rate (default: %(default)s)")
    p.add_argument("--rank", type=int, default=RECIPE["rank"], help="LoRA rank (default: %(default)s)")
    p.add_argument("--alpha", type=int, default=RECIPE["alpha"], help="LoRA alpha (default: %(default)s)")
    p.add_argument("--batch", type=int, default=RECIPE["batch"], help="examples per step on the GPU (default: %(default)s)")
    p.add_argument("--grad-accum", type=int, default=RECIPE["grad_accum"],
                   help="steps accumulated per update (default: %(default)s)")
    p.add_argument("--max-steps", type=int, help="stop after this many updates (a smoke test); overrides --epochs")
    p.add_argument("--eval-steps", type=int, default=RECIPE["eval_steps"],
                   help="validation loss every N updates, as well as before the first and after the last (0: only "
                        "those two) (default: %(default)s)")
    p.add_argument("--checkpoint-steps", type=int, default=0,
                   help="a FULL trainer checkpoint (adapter + optimizer + scheduler + RNG) every N updates into "
                        "<output>/checkpoints (the last 2 kept) — what --resume continues from after a Spot "
                        "preemption (default 0 = none)")
    p.add_argument("--resume", action="store_true",
                   help="continue from the newest <output>/checkpoints/checkpoint-* (same data, same flags); "
                        "a fresh start when there is none")
    p.add_argument("--save-steps", type=int, default=RECIPE["save_steps"],
                   help="also save the adapter every N updates, overwriting one copy (adapter-latest) (default: 0, off)")
    p.add_argument("--val-max-rows", type=int, default=RECIPE["val_max_rows"],
                   help="measure the validation loss on at most this many rows (a seeded subset) (default: all)")
    p.add_argument("--optim", default=RECIPE["optim"], help="the optimizer (default: %(default)s)")
    p.add_argument("--warmup-ratio", type=float, default=RECIPE["warmup_ratio"], help="(default: %(default)s)")
    p.add_argument("--weight-decay", type=float, default=RECIPE["weight_decay"], help="(default: %(default)s)")
    p.add_argument("--seed", type=int, default=RECIPE["seed"],
                   help="the LoRA init and the data order (default: %(default)s)")
    p.add_argument("--val-seed", type=int, default=VAL_SEED,
                   help="draws the --val-max-rows subset and the --gen-check rows; fixed across --seed so the seeds' "
                        "scores compare (default: %(default)s)")
    p.add_argument("--init-adapter", type=Path,
                   help="warm start: a LoRA adapter folder (adapter_config.json + adapter_model.safetensors, e.g. the "
                        f"live round's) loaded before the first update; same rank and alpha. Pair it with a lower --lr "
                        f"(about {WARM_LR:g}) (default: a fresh LoRA on the base)")
    p.add_argument("--gen-check", type=int, default=0,
                   help="after training, generate the first assistant turn of this many fixed validation rows "
                        "(greedy) and count the ones whose tool calls match the reference; recorded in run.json, read "
                        "by `select` (default: 0, off)")
    p.add_argument("--tiled-mlp", choices=["auto", "on", "off"], default=RECIPE["tiled_mlp"],
                   help=f"Unsloth tiled MLP, trading speed for memory (auto: on above {TILED_ABOVE} tokens) "
                        "(default: %(default)s)")
    p.add_argument("--thinking", action="store_true", default=RECIPE["thinking"],
                   help="refused: the rows are rendered as the service serves them, thinking disabled (the empty think "
                        "block); the traces carry no thinking (docs/LAB.md §5)")
    p.add_argument("--fields-context", choices=["on", "off"], default="on",
                   help="what the rows' user turns must carry: on = the candidate-fields block the served student's "
                        "route sends (dataset --fields-context on, or `souschef-lab fields rerender`), off = none (the "
                        "form adapters before 2026-10-04 were trained on). Checked against the rows — training never "
                        "renders the block itself — and recorded in run.json (default: %(default)s)")
    p.add_argument("--instruction-part", help="override the masking marker that starts a non-assistant turn")
    p.add_argument("--response-part", help="override the masking marker that starts an assistant turn")
    p.add_argument("--dry-run", "--check", dest="dry_run", action="store_true",
                   help="validate the dataset and print the resolved plan, then stop (no GPU, no torch)")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_parser()
    argv = trainkit.apply_config(p, sys.argv[1:] if argv is None else argv, CONFIG_KEYS)
    args = p.parse_args(argv)
    if args.base is None:
        args.base = chatfmt.get(args.family).base
    args.data = Path(args.data)
    if args.output is not None:
        args.output = Path(args.output)
    if args.init_adapter is not None:
        args.init_adapter = Path(args.init_adapter)
    if args.gen_check < 0 or args.seed < 0 or args.val_seed < 0:
        p.error("--gen-check, --seed and --val-seed must be 0 or more")
    if args.checkpoint_steps < 0:
        p.error("--checkpoint-steps must be 0 or more")
    if args.save_steps < 0 or args.eval_steps < 0 or (args.val_max_rows is not None and args.val_max_rows < 1):
        p.error("--save-steps and --eval-steps must be 0 or more and --val-max-rows 1 or more")
    if args.rank < 1 or args.alpha < 1 or args.batch < 1 or args.grad_accum < 1 or args.lr <= 0 \
            or args.max_seq_length < 1024 or args.epochs <= 0:
        p.error("--rank, --alpha, --batch and --grad-accum must be 1 or more, --lr and --epochs above 0, "
                "--max-seq-length 1024 or more")
    return args


def render_text(messages: list[dict], tools: list[dict] | None, history: int = 0, fmt=None) -> str:
    """A row's training text: the conversation as Ollama renders it for the
    served student (serving.render_training — the qwen3.5 renderer with
    thinking disabled, every assistant turn opened by the empty think block
    the served generation prompt ends with). Never the base's chat template:
    its tools block and echoed tool calls are not the served bytes
    (docs/LAB.md §5 "Serving parity"). fmt (chatfmt.Format, default qwen)
    picks the family's render: mistral is serving_mistral.render_training."""
    return (fmt.render_training if fmt is not None else serving.render_training)(messages, tools, history)


def served_markers(tok, fmt=None) -> None:
    """Stop unless the ChatML markers the serving render writes are single
    special tokens of this base's tokenizer: the masking (MARKERS) and the
    render assume Qwen's vocabulary. For the mistral family (fmt), each
    marker must be one token AND that token must be the id the masking
    expects (serving_mistral.SPECIAL_IDS): a tokenizer that holds [ARGS] as
    another id would mask the wrong spans."""
    if fmt is None or fmt.name == "qwen":
        missing = [t for t in serving.SPECIAL_TOKENS if len(tok(t, add_special_tokens=False)["input_ids"]) != 1]
        if missing:
            die(f"this base's tokenizer has no single token for {', '.join(missing)}: the serving render (Ollama's "
                "qwen3.5 renderer) and the ChatML masking need a Qwen3.5 base")
        return
    bad = []
    for t in fmt.special_tokens:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        want = (fmt.special_ids or {}).get(t)
        if len(ids) != 1 or (want is not None and ids[0] != want):
            bad.append(f"{t} → {ids} (want [{want}])")
    if bad:
        die(f"this base's tokenizer does not hold the {fmt.name} markers as the expected single tokens: "
            f"{'; '.join(bad)} — the render (serving_{fmt.name}.py) and the id-level masking need "
            f"{fmt.base}@{fmt.revision}'s tokenizer")


def history_spans(row: dict, fmt=None) -> int:
    """How many leading assistant SEGMENTS of a row's render are resent history: history_turns for qwen; for mistral
    the collated assistant messages (Ollama's collate merges consecutive ones) that hold any of the first
    history_turns assistant messages — a merged segment holding history is masked whole."""
    k = history_turns(row)
    if fmt is None or fmt.name == "qwen" or k <= 0:
        return k
    seen, segments = 0, 0
    prev_role = None
    for m in row["messages"]:
        role = m.get("role")
        if role == "assistant":
            if prev_role != "assistant":
                segments += 1
            seen += 1
            if seen >= k:
                return segments
        prev_role = role
    return segments


def validation_loss(model, rows, torch, bf16: bool = False) -> float | None:
    """The mean loss on the validation rows (tokenized, with their masked
    labels), computed the way training computes it, without building the
    logits: the Trainer's own evaluation forces fp32 logits out (sequence ×
    vocabulary × 4 bytes) and runs a 32 GB card out of memory on a long row
    (the thermometer's smoke of 2026-09-26)."""
    import os

    old = os.environ.get("UNSLOTH_RETURN_LOGITS")
    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    was_training = model.training
    model.eval()
    parts = []
    try:
        with torch.no_grad(), trainkit.amp(torch, bf16):
            for ex in rows:
                labels = list(ex["labels"])
                k = sum(1 for x in labels[1:] if x != -100)  # the loss shifts labels by one
                if not k:
                    continue
                ids = torch.tensor([ex["input_ids"]], device="cuda")
                out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                            labels=torch.tensor([labels], device="cuda"))
                parts.append((float(out.loss), k))
                del out, ids
    finally:
        if old is None:
            os.environ.pop("UNSLOTH_RETURN_LOGITS", None)
        else:
            os.environ["UNSLOTH_RETURN_LOGITS"] = old
        if was_training:
            model.train()
        torch.cuda.empty_cache()
    return weighted_loss(parts)


# what transformers' Trainer._save_checkpoint writes (trainer_state.json last); a LoRA checkpoint's weights are
# adapter_model.safetensors
CHECKPOINT_FILES = ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json")


def resumable_checkpoints(ckpt_dir: Path) -> tuple[list[Path], list[Path]]:
    """(complete checkpoint-N folders oldest first, incomplete ones). A checkpoint cut off mid-write or mid-upload (a
    Spot VM switched off while pipeline-train.sh pushed it) lacks a file, and resuming from it would raise."""
    if not Path(ckpt_dir).is_dir():
        return [], []
    found = [d for d in Path(ckpt_dir).glob("checkpoint-*") if d.is_dir() and d.name.split("-")[-1].isdigit()]
    found.sort(key=lambda d: int(d.name.split("-")[-1]))
    ok = [d for d in found if all((d / f).is_file() for f in CHECKPOINT_FILES)]
    return ok, [d for d in found if d not in ok]


def run_name(args) -> str:
    return args.name or re.sub(r"[^a-z0-9._-]+", "-", f"{args.base.split('/')[-1]}-{utc_stamp()}".lower())


def dataset_manifest(data_dir: Path) -> dict:
    return trainkit.read_json(Path(data_dir) / "dataset-manifest.json", {}) or {}


def fields_context_of(args, train_rows: list[dict], val_rows: list[dict]) -> dict:
    """What run.json records about the candidate-fields block (LAB.md §5):
    the option, how many rows' user turns carry the block, and what the
    dataset's own manifest says it was built with (retrieval source hash
    included) — so an adapter trained with the block is distinguishable from
    one trained without."""
    stats = fields.block_stats(train_rows + val_rows)
    declared = dataset_manifest(args.data).get("fields_context")
    return {"enabled": args.fields_context == "on", "rows": stats["rows"], "rows_with_block": stats["with_block"],
            "dataset": declared if isinstance(declared, dict) else None}


def check_fields_context(fc: dict, data: Path) -> str | None:
    """The reason the rows are not what --fields-context asks for, if any."""
    if fc["enabled"] and fc["rows_with_block"] == 0:
        return (f"--fields-context on (the default), but no row in {trainkit.rel(data)} carries a <candidate_fields> "
                f"block: the served student's route sends one (README §5 fields_context), so an adapter trained on "
                f"these rows would read a prompt it never trained on. Rebuild with `souschef-lab build "
                f"--fields-context on`, or re-render this set with `souschef-lab fields rerender --data "
                f"{trainkit.rel(data)} --manifest <the cases' manifest.json> --out <new folder>`; pass "
                f"--fields-context off to train the old form on purpose (and serve it with fields_context off)")
    if not fc["enabled"] and fc["rows_with_block"]:
        return (f"--fields-context off, but {fc['rows_with_block']} of {fc['rows']} rows in {trainkit.rel(data)} carry "
                f"a <candidate_fields> block: drop the flag, or strip the blocks with `souschef-lab fields rerender "
                f"--fields-context off`")
    return None


# --- Warm start (--init-adapter) ---------------------------------------------------------------------------------------

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def init_adapter_info(path: Path) -> dict:
    """What run.json records about the adapter a warm start begins from (no torch). ValueError when the folder is
    not a LoRA adapter."""
    d = Path(path)
    missing = [f for f in ADAPTER_FILES if not (d / f).is_file()]
    if missing:
        raise ValueError(f"--init-adapter {d}: no {', '.join(missing)} there (a LoRA adapter folder: the `adapter/` "
                         "inside a run, e.g. runs/adapters/r4-sft/adapter)")
    try:
        cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
    except ValueError as e:
        raise ValueError(f"--init-adapter {d}: adapter_config.json is not JSON ({e})") from None
    if str(cfg.get("peft_type", "LORA")).upper() != "LORA":
        raise ValueError(f"--init-adapter {d}: a {cfg.get('peft_type')} adapter, not LoRA")
    run = trainkit.read_json(d.parent / "run.json", {}) or {}
    return {"path": str(d), "r": cfg.get("r"), "alpha": cfg.get("lora_alpha"), "base": cfg.get("base_model_name_or_path"),
            "bytes": (d / "adapter_model.safetensors").stat().st_size,
            "run": {k: run.get(k) for k in ("name", "seed", "lr", "steps", "last_eval_loss", "manifest_hash")
                    if k in run} or None}


def check_init_adapter(info: dict | None, args) -> str | None:
    """The reason this adapter cannot start this run, if any: its LoRA shape must be this run's."""
    if not info:
        return None
    if (info["r"], info["alpha"]) != (args.rank, args.alpha):
        return (f"--init-adapter {info['path']} is r{info['r']}/α{info['alpha']}, this run r{args.rank}/α{args.alpha}: "
                f"pass --rank {info['r']} --alpha {info['alpha']} (a warm start continues the same LoRA)")
    base = str(info.get("base") or "")
    if base and "/" in base and not Path(base).is_absolute() and base != args.base:
        return f"--init-adapter {info['path']} was trained on {base}, this run's base is {args.base}"
    return None


def load_init_adapter(model, path: Path) -> dict:
    """Copy a saved LoRA's weights into the model's fresh LoRA (get_peft_model's), so training continues it with
    Unsloth's own patched setup unchanged. Stops on any LoRA weight the two do not share (another rank, other target
    modules, another base); a shape mismatch raises in PEFT."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    state = load_file(str(Path(path) / "adapter_model.safetensors"))
    res = set_peft_model_state_dict(model, state)
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    missing = [k for k in (getattr(res, "missing_keys", []) or []) if "lora_" in k]
    if unexpected or missing:
        die(f"--init-adapter {path} does not fit this model's LoRA: {len(unexpected)} of its tensors have no place "
            f"(e.g. {unexpected[:2]}), {len(missing)} LoRA weights would stay fresh (e.g. {missing[:2]})")
    return {"tensors": len(state)}


# --- The generation check (--gen-check): the first assistant turn of fixed validation rows, greedy -------------------

_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]*)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]*)>\n?(.*?)\n?</parameter>", re.S)


# The mistral family's reading of a reply (Ollama's MinistralParser: [TOOL_CALLS]name[ARGS]{…}); images/trainer/selftest
# checks it beside parse_tool_calls.
parse_tool_calls_mistral = serving_mistral.parse_tool_calls


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """The tool calls in a rendered (or generated) assistant turn — qwen3.5's <tool_call><function=…><parameter=…>
    form (serving.render_conversation) — as (name, {parameter: value}), a value JSON-decoded when it is JSON, so the
    order of an object's keys does not count."""
    calls = []
    for name, body in _CALL.findall(text or ""):
        params = {}
        for k, v in _PARAM.findall(body):
            try:
                params[k.strip()] = json.loads(v)
            except ValueError:
                params[k.strip()] = v.strip()
        calls.append((name.strip(), params))
    return calls


def gen_check_items(rows: list[dict], n: int, seed: int = VAL_SEED, fmt=None) -> list[dict]:
    """n validation rows (not follow-ups, a seeded draw in their own order: the same rows for every --seed) as
    {id, prompt, expected}: the served prompt up to the first assistant turn's generation point, and that turn's
    reference tool calls (none: a reply). fmt (default qwen) is the family's render, generation points, end-of-turn
    marker and call reader."""
    fmt = fmt or chatfmt.get("qwen")
    pool = [r for r in rows if not history_turns(r) and any(m.get("role") == "assistant" for m in r["messages"])]
    items = []
    for r in val_subset(pool, n, seed):
        msgs = r["messages"]
        k = next(i for i, m in enumerate(msgs) if m.get("role") == "assistant")
        text = render_text(msgs[:k + 1], r.get("tools"), 0, fmt)
        points = fmt.generation_points(text)
        if not points:
            continue
        cut = points[-1]
        items.append({"id": r.get("id"), "prompt": text[:cut],
                      "expected": fmt.parse_tool_calls(text[cut:].split(fmt.end_of_turn)[0])})
    return items


def gen_check_pass(generated: str, expected: list[tuple[str, dict]], fmt=None) -> bool:
    """A generated first turn matches when its tool calls are the reference's (names and arguments, in order); a
    reference reply (no call) matches a generation with no call."""
    fmt = fmt or chatfmt.get("qwen")
    return fmt.parse_tool_calls((generated or "").split(fmt.end_of_turn)[0]) == expected


def generation_check(model, tok, items: list[dict], torch, fast_model, budget_s: float = GEN_CHECK_BUDGET_S,
                     fmt=None) -> dict:
    """Greedy generation of each item's first assistant turn; {n, scored, match, rate, seconds, failed_ids}. Never
    raises: an error is recorded (and the check is then incomplete, so `select` falls back to the loss)."""
    t0 = time.monotonic()
    res = {"n": len(items), "scored": 0, "match": 0, "rate": None, "seconds": None, "failed_ids": [],
           "max_new_tokens": GEN_CHECK_MAX_NEW_TOKENS}
    try:
        if hasattr(fast_model, "for_inference"):
            fast_model.for_inference(model)
        model.eval()
        fmt = fmt or chatfmt.get("qwen")
        eos = tok.convert_tokens_to_ids(fmt.end_of_turn)
        for it in items:
            if time.monotonic() - t0 > budget_s:
                res["stopped"] = f"over {budget_s:.0f}s"
                break
            ids = torch.tensor([tok(it["prompt"], add_special_tokens=fmt.prompt_add_special_tokens)["input_ids"]],
                               device="cuda")
            with torch.no_grad():
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                     max_new_tokens=GEN_CHECK_MAX_NEW_TOKENS, eos_token_id=eos, pad_token_id=eos)
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
            res["scored"] += 1
            if gen_check_pass(text, it["expected"], fmt):
                res["match"] += 1
            else:
                res["failed_ids"].append(it["id"])
            del out, ids
    except Exception as e:  # noqa: BLE001 - the adapter is saved; a failed check must not cost the run
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        torch.cuda.empty_cache()
    res["seconds"] = round(time.monotonic() - t0, 1)
    res["rate"] = round(res["match"] / res["scored"], 4) if res["scored"] else None
    return res


# --- Seed selection (`select`) -------------------------------------------------------------------------------------

def gen_complete(gc) -> bool:
    return isinstance(gc, dict) and not gc.get("error") and gc.get("n") and gc.get("scored") == gc.get("n") \
        and isinstance(gc.get("match"), int)


def arm_of(run: dict) -> str:
    """A run's arm, from its run.json: warm:<seed> when it started from an --init-adapter, else base:<seed>."""
    warm = run.get("start") == "init_adapter" or bool(run.get("init_adapter"))
    return f"{'warm' if warm else 'base'}:{run.get('seed')}"


def select_seed(runs: list[dict], margin: int = 1) -> dict:
    """Pick one of a round's arm runs (their run.json: from-base seeds and warm starts alike) — never on the bar's
    sets: on the validation rows, which are the same for every arm (--val-seed). When every run has a complete
    generation check of the same size, the runs within `margin` matches of the best stay in, and of those the lowest
    final validation loss wins; otherwise the lowest final validation loss alone. A missing loss loses; a tie goes to
    the earlier run (TRAIN_ARMS order). Returns {selected: index, rule, seeds} (`seeds`: one entry per arm)."""
    if not runs:
        raise ValueError("no runs to select from")
    seeds = []
    for r in runs:
        gc = r.get("gen_check")
        loss = r.get("last_eval_loss")
        seeds.append({"arm": arm_of(r), "seed": r.get("seed"), "name": r.get("name"),
                      "start": "warm" if arm_of(r).startswith("warm") else "base", "lr": r.get("lr"),
                      "start_eval_loss": r.get("start_eval_loss"),
                      "eval_loss": loss if isinstance(loss, (int, float)) else None,
                      "base_eval_loss": r.get("base_eval_loss"), "steps": r.get("steps"),
                      "gen_match": gc.get("match") if gen_complete(gc) else None,
                      "gen_n": gc.get("n") if isinstance(gc, dict) else None,
                      "gen_error": gc.get("error") if isinstance(gc, dict) else None})
    by_gen = all(s["gen_match"] is not None for s in seeds) and len({s["gen_n"] for s in seeds}) == 1
    cand = list(range(len(seeds)))
    if by_gen:
        best = max(s["gen_match"] for s in seeds)
        cand = [i for i in cand if seeds[i]["gen_match"] >= best - margin]

    def loss_key(i):
        v = seeds[i]["eval_loss"]
        return (v is None, v if v is not None else 0.0, i)

    pick = min(cand, key=loss_key)
    for i, s in enumerate(seeds):
        s["selected"] = i == pick
    rule = (f"generation check within {margin} of the best match, then the lowest final validation loss" if by_gen
            else "the lowest final validation loss (no complete generation check on every arm)")
    return {"selected": pick, "rule": rule, "by": "gen_check+loss" if by_gen else "loss", "margin": margin,
            "seeds": seeds}


def select_main(argv: list[str]) -> int:
    """`python -m souschef_lab.train select RUN_DIR… --into OUT [--skipped ARM=REASON]…`: copy the selected arm's
    run (adapter/, run.json, loss.jsonl, samples — no checkpoints) to OUT, its run.json carrying every arm's scores
    (seeds: one entry per arm, selection) and the planned arms that were not trained (selection.skipped)."""
    import shutil

    ap = argparse.ArgumentParser(prog="souschef_lab.train select", description=select_main.__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="the seeds' output folders (each with run.json and adapter/)")
    ap.add_argument("--into", type=Path, required=True, help="where the selected run goes (replaced)")
    ap.add_argument("--margin", type=int, default=1, help="generation-check matches treated as a tie (default: 1)")
    ap.add_argument("--skipped", action="append", default=[], metavar="ARM=REASON",
                    help="a planned arm that was not trained (the A100 trim, the 12 h window), recorded")
    ns = ap.parse_args(argv)
    skipped = []
    for item in ns.skipped:
        arm, _, why = item.partition("=")
        if not arm.strip():
            die(f"select: --skipped {item!r} is not ARM=REASON")
        skipped.append({"arm": arm.strip(), "reason": why.strip() or None})
    runs = []
    for d in ns.runs:
        r = trainkit.read_json(d / "run.json", None)
        if not isinstance(r, dict) or not (d / "adapter" / "adapter_config.json").is_file():
            die(f"select: {d} has no run.json or no adapter/adapter_config.json (an unfinished seed)")
        runs.append(r)
    sel = select_seed(runs, ns.margin)
    for d, s in zip(ns.runs, sel["seeds"]):
        s["output"] = str(d)
    win = ns.runs[sel["selected"]]
    out = ns.into
    if out.resolve() in {d.resolve() for d in ns.runs}:
        die("select: --into must be a folder of its own, not one of the seeds' runs")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(win / "adapter", out / "adapter")
    for f in ("loss.jsonl", "sample_rendered.txt", "sample_trained.txt"):
        if (win / f).is_file():
            shutil.copyfile(win / f, out / f)
    summary = {"selected_seed": sel["seeds"][sel["selected"]]["seed"], "selected_arm": sel["seeds"][sel["selected"]]["arm"],
               "selected_from": str(win), "rule": sel["rule"], "by": sel["by"], "margin": sel["margin"],
               "arms": [s["arm"] for s in sel["seeds"]], "skipped": skipped, "at": utc_now()}
    run = {**runs[sel["selected"]], "seeds": sel["seeds"], "selection": summary}
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (out / "seeds.json").write_text(json.dumps({"selection": summary, "seeds": sel["seeds"]}, indent=2),
                                    encoding="utf-8")
    for s in sel["seeds"]:
        gen = f", gen {s['gen_match']}/{s['gen_n']}" if s["gen_match"] is not None else ""
        say(f"arm {s['arm']}: final validation loss {s['eval_loss']}{gen}" + ("  <- selected" if s["selected"] else ""))
    for k in skipped:
        say(f"arm {k['arm']}: not trained ({k['reason'] or 'skipped'})")
    say(f"Selected seed {summary['selected_seed']} (arm {summary['selected_arm']}; {sel['rule']}) -> {trainkit.rel(out)}")
    return 0


# --- The pipeline's training settings (tools/pipeline-train.sh) ----------------------------------------------------

PIPELINE_ENV = {"TRAIN_SEED": "seed", "TRAIN_SEEDS": "seeds", "TRAIN_ARMS": "arms", "INIT_ADAPTER": "init_adapter",
                "INIT_LR": "init_lr", "TRAIN_GEN_CHECK": "gen_check"}
# The base family (chatfmt): BASE_FAMILY on the job, else train-config.json `family` (the dataset stage writes it from
# trigger.base_family / infra config base_family when the round is not qwen), else qwen.
FAMILY_ENV, FAMILY_KEY = "BASE_FAMILY", "family"
GEN_CHECK_MULTI = 32  # rows the generation check scores by default when several arms are trained
_GS_SAFE = re.compile(r"^gs://[a-z0-9][a-z0-9._-]*(/[A-Za-z0-9._=@+-]+)+$")
# A round's training arms (r6, the user's decision of 2026-10-15: from-base seeds AND a warm start in the same round,
# the best kept). An arm is "<start>:<seed>": base = a fresh LoRA on the base at the recipe's lr; warm = the
# init_adapter's LoRA at the warm-start lr (init_lr, default WARM_LR). Each trains on the same dataset in turn and is
# scored on the same fixed validation rows; `select` keeps one.
ARM_KINDS = ("base", "warm")
MAX_ARMS = 4  # an H100 trains about 3 r4-sized arms in its 12 h attempt (window_estimate); a 5th never fits
# Wall time on the train stage's H100 (r4/r5: 601-650 updates in 2.0-2.4 h of SFT): the upper rate per update, an
# arm's fixed part (model load, the validations, a 32-row generation check, the push), the job's head (self-test,
# the 18 GB base pull and unpack, the resume smoke) and tail (export q4_k_m, stage, push). An A100 is about 2x.
H100_S_PER_UPDATE = 12.0  # r4: 601 updates in 6,498 s of SFT (10.8 s); r5 measured 2.4 H100-hours an arm
# The mistral family (Ministral 3 8B, full attention in every layer) trains slower on these long rows than Qwen3.5's
# hybrid attention: the r7 arm (mx-20261018-m3-8b-r7) took 19,322 s of SFT for 994 updates on an H100 (19.4 s an
# update, 3,233 tokens/s) where r7's own base arm took 11,834 s (11.9 s, 5,413 tokens/s) on the same rows.
H100_S_PER_UPDATE_BY_FAMILY = {"qwen": H100_S_PER_UPDATE, "mistral": 20.0}
ARM_FIXED_S = 900
JOB_HEAD_S = 1800
JOB_TAIL_S = 2700
WINDOW_S = 43200  # the workflow's Vertex timeout (train_timeout_hours, 12)


def arm_tag(kind: str, seed: int) -> str:
    """The arm's folder name: s<seed> from the base (the TRAIN_SEEDS layout), w<seed> from the warm-start adapter."""
    return f"{'s' if kind == 'base' else 'w'}{seed}"


def _seed_list(v) -> list[int]:
    parts = v if isinstance(v, list) else [p for p in re.split(r"[,\s]+", str(v)) if p]
    out = []
    for p in parts:
        if isinstance(p, bool) or not re.fullmatch(r"\d{1,9}", str(p).strip()):
            raise ValueError(f"a seed must be a whole number 0 or more, not {p!r}")
        out.append(int(str(p).strip()))
    if len(set(out)) != len(out):
        raise ValueError(f"a seed is listed twice: {out}")
    return out


def _arm_list(v) -> list[tuple[str, int]]:
    """TRAIN_ARMS / trigger.train.arms → [(kind, seed)]: a list, or a string separated by commas or spaces, of
    "base:<seed>" / "warm:<seed>" (a bare "base" or "warm" is seed 3407)."""
    parts = v if isinstance(v, list) else [p for p in re.split(r"[,\s]+", str(v)) if p]
    out: list[tuple[str, int]] = []
    for p in parts:
        m = re.fullmatch(r"(base|warm)(?::(\d{1,9}))?", str(p).strip()) if isinstance(p, str) else None
        if not m:
            raise ValueError(f"an arm is base:<seed> or warm:<seed> (e.g. base:3407, warm:3407), not {p!r}")
        out.append((m.group(1), int(m.group(2)) if m.group(2) is not None else RECIPE["seed"]))
    if not out:
        raise ValueError("TRAIN_ARMS lists no arm")
    dup = sorted({f"{k}:{s}" for k, s in out if out.count((k, s)) > 1})
    if dup:
        raise ValueError(f"an arm is listed twice: {', '.join(dup)}")
    if len(out) > MAX_ARMS:
        raise ValueError(f"{len(out)} arms: at most {MAX_ARMS} fit one 12 h training job")
    return out


def window_estimate(n_arms: int, updates: int | None, *, a100: bool = False, family: str | None = None) -> dict:
    """The train job's wall time for `n_arms` arms of `updates` updates each (every arm trains on the same rows), on
    the H100 (a100: about 2x), against the 12 h Vertex window, at the base family's rate (H100_S_PER_UPDATE_BY_FAMILY;
    default qwen). A planning figure: pipeline-train.sh's own guard measures the arms it actually trains."""
    per_arm = (updates or 0) * H100_S_PER_UPDATE_BY_FAMILY[chatfmt.check(family)] * (2 if a100 else 1) + ARM_FIXED_S
    total = JOB_HEAD_S + n_arms * per_arm + JOB_TAIL_S
    return {"n_arms": n_arms, "updates_per_arm": updates, "arm_h": round(per_arm / 3600, 2),
            "total_h": round(total / 3600, 2), "window_h": WINDOW_S / 3600, "fits": total <= WINDOW_S,
            "fit_arms": max(0, int((WINDOW_S - JOB_HEAD_S - JOB_TAIL_S) // per_arm)) if per_arm else None}


def pipeline_config(env, file_cfg: dict | None, bucket: str) -> dict:
    """The training settings of a pipeline round, from the environment (TRAIN_SEED, TRAIN_SEEDS, TRAIN_ARMS,
    INIT_ADAPTER, INIT_LR, TRAIN_GEN_CHECK; an empty value is unset) over runs/<id>/train-config.json (seed, seeds,
    arms, init_adapter, init_lr, gen_check), over today's defaults: one arm, base:3407 (train.py's seed, no --seed
    passed, from the base), no generation check. ValueError on a bad value.

    arms: ["base:3407", "base:2", "warm:3407"] trains those arms in turn; a warm arm needs init_adapter, and an
    init_adapter needs a warm arm. Without arms, seed / seeds give base arms (warm ones with init_adapter: r6's first
    form, every seed from the adapter). arms and seed / seeds are exclusive.
    init_adapter: gs://<the pipeline bucket>/runs/… or …/base/…, a bucket path (runs/… or base/…, the trainer's
    readable prefixes), or a run id (rt-…: that round's trained adapter, runs/<id>/train/adapter/adapter). INIT_LR is
    the warm arms' --lr (default WARM_LR); base arms keep the recipe's lr.
    family: the base family (BASE_FAMILY, else the file's `family`, which the dataset stage writes for a non-qwen
    round, else qwen; chatfmt.FAMILIES). The result carries `family` only when it is not qwen, so a qwen round's
    settings are today's."""
    file_cfg = dict(file_cfg or {})
    unknown = sorted(set(file_cfg) - set(PIPELINE_ENV.values()) - {FAMILY_KEY})
    if unknown:
        raise ValueError(f"train-config.json: unknown key(s) {', '.join(unknown)} (known: "
                         f"{', '.join((*PIPELINE_ENV.values(), FAMILY_KEY))})")
    fam_env = env.get(FAMILY_ENV)
    fam_env = None if fam_env is None or str(fam_env).strip() == "" else str(fam_env).strip()
    try:
        family = chatfmt.resolve(fam_env, file_cfg.get(FAMILY_KEY))
    except ValueError as e:
        raise ValueError(f"{FAMILY_ENV if fam_env is not None else 'train-config.json family'}: {e}") from None
    got, source = {}, {}
    for var, key in PIPELINE_ENV.items():
        v = env.get(var)
        if v is not None and str(v).strip() != "":
            got[key], source[key] = str(v).strip(), "env " + var
        elif file_cfg.get(key) not in (None, "", []):
            got[key], source[key] = file_cfg[key], "train-config.json"
    if "seed" in got and "seeds" in got:
        raise ValueError("TRAIN_SEED and TRAIN_SEEDS are both set: one seed, or a list")
    if "arms" in got and ("seed" in got or "seeds" in got):
        raise ValueError("TRAIN_ARMS and TRAIN_SEED(S) are both set: the arms name their seeds (base:<seed>, warm:<seed>)")
    seeds = _seed_list(got["seeds"]) if "seeds" in got else _seed_list([got["seed"]]) if "seed" in got else []
    if "seeds" in got and not seeds:
        raise ValueError("TRAIN_SEEDS lists no seed")
    if len(seeds) > MAX_ARMS:
        raise ValueError(f"{len(seeds)} seeds: at most {MAX_ARMS} fit one 12 h training job")
    arms = _arm_list(got["arms"]) if "arms" in got else []
    b = bucket.rstrip("/")
    b = b if b.startswith("gs://") else "gs://" + b
    url = ""
    if "init_adapter" in got:
        v = str(got["init_adapter"]).strip().rstrip("/")
        if v.startswith("gs://"):
            # only the pipeline bucket's runs/ and base/: the trainer's service account reads nothing else, so any
            # other path would fail after the image pull, on the GPU (r6 review)
            if not v.startswith((f"{b}/runs/", f"{b}/base/")):
                raise ValueError(f"INIT_ADAPTER {v!r}: the trainer reads only {b}/runs/… and {b}/base/…")
            url = v
        elif v.startswith(("runs/", "base/")):
            url = f"{b}/{v}"
        elif re.fullmatch(r"rt-[A-Za-z0-9-]+", v):
            url = f"{b}/runs/{v}/train/adapter/adapter"
        else:
            raise ValueError(f"INIT_ADAPTER {v!r}: gs://…, runs/… or base/… in the pipeline bucket, or a run id (rt-…)")
        if not _GS_SAFE.match(url):
            raise ValueError(f"INIT_ADAPTER {url!r} is not a plain gs:// object path")
    if "init_lr" in got and not url:
        raise ValueError("INIT_LR is the warm start's learning rate: set INIT_ADAPTER too")
    if arms:
        warm = [f"{k}:{s}" for k, s in arms if k == "warm"]
        if warm and not url:
            raise ValueError(f"arm(s) {', '.join(warm)} start from the warm-start adapter: set INIT_ADAPTER "
                             "(init_adapter) too")
        if url and not warm:
            raise ValueError("INIT_ADAPTER is set but no arm is warm: add a warm:<seed> arm, or drop init_adapter")
    else:
        arms = [("warm" if url else "base", s) for s in (seeds or [RECIPE["seed"]])]
    multi = len(arms) > 1
    lr = None
    if url:
        try:
            lr = float(got.get("init_lr", WARM_LR))
        except (TypeError, ValueError):
            raise ValueError(f"INIT_LR {got.get('init_lr')!r} is not a number") from None
        if not 0 < lr < 1:
            raise ValueError(f"INIT_LR {lr} must be above 0 and below 1")
    gen = got.get("gen_check", GEN_CHECK_MULTI if multi else 0)
    if isinstance(gen, bool) or not re.fullmatch(r"\d{1,4}", str(gen).strip()):
        raise ValueError(f"TRAIN_GEN_CHECK {gen!r} must be a whole number 0-9999")
    out = {"seeds": [s for _, s in arms], "seed_flag": bool(seeds) or "arms" in got, "multi": multi,
           "init_adapter": url, "init_lr": repr(lr) if lr is not None else "", "gen_check": int(str(gen).strip()),
           "arms": [{"arm": f"{k}:{s}", "start": k, "seed": s, "tag": arm_tag(k, s),
                     "lr": lr if k == "warm" else RECIPE["lr"]} for k, s in arms],
           "source": source}
    if family != chatfmt.DEFAULT:  # a qwen round's settings stay today's, key for key
        out["family"] = family
        source[FAMILY_KEY] = "env " + FAMILY_ENV if fam_env is not None else "train-config.json"
    return out


def pipeline_config_main(argv: list[str]) -> int:
    """pipeline-train.sh's resolver: `pipeline_config_main [CONFIG_FILE] BUCKET` prints KEY=value lines (SEEDS,
    SEED_FLAG, MULTI, INIT_ADAPTER, INIT_LR, GEN_CHECK, ARMS, and FAMILY when the base family is not qwen) and the
    resolved settings as JSON on stderr; exit 2 on a bad setting."""
    import os

    path, bucket = (argv[0], argv[1]) if len(argv) == 2 else ("", argv[0])
    try:
        file_cfg = json.loads(Path(path).read_text(encoding="utf-8")) if path else None
        if file_cfg is not None and not isinstance(file_cfg, dict):
            raise ValueError("train-config.json must be a JSON object")
        c = pipeline_config(os.environ, file_cfg, bucket)
    except (OSError, ValueError) as e:
        print(f"training settings: {e}", file=sys.stderr)
        return 2
    print(f"SEEDS={' '.join(str(s) for s in c['seeds'])}")
    print(f"SEED_FLAG={int(c['seed_flag'])}")
    print(f"MULTI={int(c['multi'])}")
    print(f"INIT_ADAPTER={c['init_adapter']}")
    print(f"INIT_LR={c['init_lr']}")
    print(f"GEN_CHECK={c['gen_check']}")
    print(f"ARMS={' '.join(a['arm'] for a in c['arms'])}")
    if c.get("family"):
        print(f"FAMILY={c['family']}")
    print("training settings: " + json.dumps(c), file=sys.stderr)
    return 0


def plan(args, train_rows: list[dict] | None = None, val_rows: list[dict] | None = None) -> dict:
    """The resolved plan: what --dry-run prints and run.json records. With
    the rows, their counts and estimated sizes (bytes/3 — the base's
    tokenizer decides at training time)."""
    tiled = args.tiled_mlp == "on" or (args.tiled_mlp == "auto" and args.max_seq_length > TILED_ABOVE)
    name = run_name(args)
    out = Path(args.output) if args.output else trainkit.ADAPTERS_DIR / name
    d = {
        "kind": "sft", "base": args.base, "name": name, "output": str(out), "adapter": str(out / "adapter"),
        "precision": "bf16 base, fp32 LoRA", "lora": {"r": args.rank, "alpha": args.alpha, "dropout": 0, "bias": "none",
                                                      "targets": "attention and MLP modules (Unsloth's language layers)"},
        "rank": args.rank, "alpha": args.alpha, "lr": args.lr, "epochs": args.epochs, "max_steps": args.max_steps,
        "batch": args.batch, "grad_accum": args.grad_accum, "effective_batch": args.batch * args.grad_accum,
        "max_seq_length": args.max_seq_length, "optim": args.optim, "scheduler": "linear",
        "warmup_ratio": args.warmup_ratio, "weight_decay": args.weight_decay, "packing": False,
        "eval_steps": args.eval_steps, "save_steps": args.save_steps, "val_max_rows": args.val_max_rows,
        "checkpoint_steps": args.checkpoint_steps, "resume": args.resume,
        "seed": args.seed, "val_seed": args.val_seed, "gen_check": args.gen_check,
        "init_adapter": None, "start": "base",
        "tiled_mlp": tiled, "thinking": args.thinking, "data": str(args.data),
        "config": getattr(args, "config", None), "render": chatfmt.get(getattr(args, "family", None)).description,
        "manifest_hash": dataset_manifest(args.data).get("manifest_hash"),
    }
    fmt = chatfmt.get(getattr(args, "family", None))
    if fmt.name != "qwen":  # main's plan and run.json carry no family key (byte-identical for qwen)
        d["family"] = fmt.name
        d["revision"] = fmt.revision if args.base == fmt.base else None
    if args.init_adapter is not None:
        try:
            d["init_adapter"] = init_adapter_info(args.init_adapter)
        except ValueError as e:
            d["init_adapter"] = {"path": str(args.init_adapter), "error": str(e)}
        d["start"] = "init_adapter"
    if train_rows is not None:
        d["fields_context"] = fields_context_of(args, train_rows, val_rows or [])
        est = [estimate_tokens(r) for r in train_rows]
        val_rows = val_rows or []
        val_n = len(val_subset(val_rows, args.val_max_rows, args.val_seed))
        d["rows"] = {"train": len(train_rows), "val": len(val_rows), "val_scored": val_n,
                     "followups": sum(1 for r in train_rows + val_rows if history_turns(r)),
                     "over_length_estimate": sum(1 for n in est if n > args.max_seq_length)}
        d["tokens_estimate"] = {"median": trainkit.median(est), "max": max(est, default=0), "total": sum(est),
                                "chars_per_token": trainkit.CHARS_PER_TOKEN}
        d["updates"] = updates_for(len(train_rows), args.batch, args.grad_accum, args.epochs, args.max_steps)
    return d


PLAN_ORDER = ("kind", "base", "name", "output", "adapter", "data", "config", "manifest_hash", "fields_context",
              "precision", "lora",
              "lr", "epochs", "max_steps", "updates", "batch", "grad_accum", "effective_batch", "max_seq_length",
              "optim", "scheduler", "warmup_ratio", "weight_decay", "packing", "tiled_mlp", "thinking", "eval_steps",
              "save_steps", "checkpoint_steps", "resume", "val_max_rows", "seed", "val_seed", "start", "init_adapter",
              "gen_check", "rows", "tokens_estimate")


def render_plan(p: dict) -> str:
    return trainkit.render_plan("SFT plan (the thermometer recipe, docs/LAB.md §8)",
                                {k: v for k, v in p.items() if k not in ("rank", "alpha", "render")}, PLAN_ORDER)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["select"]:
        return select_main(argv[1:])
    args = parse_args(argv)
    if args.thinking:
        die("--thinking: the rows render in the served form (serving.py: Ollama's qwen3.5 renderer with thinking "
            "disabled, the empty think block first) and the traces carry no thinking — there is no thinking render")
    fmt = chatfmt.get(args.family)
    if args.init_adapter is not None and not fmt.warm_start:
        die(f"--family {fmt.name} --init-adapter: this family has no warm start")
    train_rows, val_rows = load_split(args.data, "train"), load_split(args.data, "val")
    if not train_rows:
        die(f"no training rows in {trainkit.rel(args.data)}/train.jsonl — run `python -m souschef_lab.dataset` first")
    problems = check_rows(train_rows, "train") + check_rows(val_rows, "val")
    if problems:
        die("the dataset has problems:\n  " + "\n  ".join(problems[:20]))
    p = plan(args, train_rows, val_rows)
    mismatch = check_fields_context(p["fields_context"], args.data)
    if mismatch:
        die(mismatch)
    if p["init_adapter"] and p["init_adapter"].get("error"):
        die(p["init_adapter"]["error"])
    bad_init = check_init_adapter(p["init_adapter"], args)
    if bad_init:
        die(bad_init)
    say(render_plan(p))
    if p["rows"]["followups"]:
        say(f"{p['rows']['followups']} rows are follow-ups: their earlier replies are masked out of the loss.")
    if fmt.name != "qwen":
        say(family_report(fmt, train_rows, val_rows, args.max_seq_length))
    if args.dry_run:
        if importlib.util.find_spec("unsloth") is None:
            say("The [train] extra is not installed: run 'make -C lab train-env' (several GB, once) before training.")
        say("Dry run: the data is well-formed. Drop --dry-run to train.")
        return 0
    if importlib.util.find_spec("unsloth") is None:
        die("the [train] extra is not installed — run 'make -C lab train-env' first (several GB, once)")
    return train(args, p, train_rows, val_rows)


def mistral_mask(ds, rows: list[dict], texts: list[str], tok, fmt, which: str):
    """The mistral family's labels, rebuilt from token ids (trainkit.mistral_labels; a follow-up's resent replies
    masked by history_spans) for every row of a tokenized split. The trainer's input_ids must be the render's own
    tokenization (BOS included); when the trainer left the BOS out, the ids (and attention_mask) are replaced by the
    render's — any other difference stops the run."""
    if ds is None:
        return ds
    cols = set(ds.column_names)
    if "input_ids" not in cols or len(ds) != len(rows):
        die(f"cannot build the Mistral labels for the {which} set: the trainer's dataset has {len(ds)} rows for "
            f"{len(rows)} rendered ({sorted(cols)}) — Unsloth dropped or reordered rows")
    hs = [history_spans(r, fmt) for r in rows]
    bos = (fmt.special_ids or {}).get(serving_mistral.BOS, 1)
    eos = (fmt.special_ids or {}).get(serving_mistral.END_OF_TURN, 2)
    # one segment per collated assistant message (the render's own count): a literal control string in a row's text
    # (`</s>`, `[INST]`, … — the tokenizer reads it as the control token) would add, cut or end a span early
    n_seg = [len(serving_mistral.render_conversation(r["messages"], r.get("tools"), close_last=True)[1]) for r in rows]
    others = sorted(cols - {"input_ids", "labels", "attention_mask"})

    def fix(ex, i):
        own = tok(texts[i], add_special_tokens=True)["input_ids"]
        got = list(ex["input_ids"])
        if got != own:
            if got != own[1:] or own[:1] != [bos]:
                die(f"{which} row {i}: the trainer's tokenization is not the render's (first difference at token "
                    f"{next((j for j, (a, b) in enumerate(zip(got, own)) if a != b), min(len(got), len(own)))})")
            if any(isinstance(ex.get(c), list) and len(ex[c]) == len(got) for c in others):
                die(f"{which} row {i}: the trainer left the BOS out and keeps per-token columns {others}: cannot "
                    "re-align them")
        spans = trainkit.mistral_spans(own)
        if own.count(bos) != 1 or len(spans) != n_seg[i] or own.count(eos) != n_seg[i]:
            die(f"{which} row {i} ({rows[i].get('id')}): {len(spans)} token spans, {own.count(eos)} </s> and "
                f"{own.count(bos)} <s> for {n_seg[i]} assistant segments — a control-token string in the row's text "
                f"({control_strings(rows[i]) or 'none found'}); fix or drop the row")
        # every row returns the same columns (datasets.map needs one schema): the render's ids, mask and labels
        out = {"input_ids": own, "labels": trainkit.mistral_labels(own, hs[i])}
        if "attention_mask" in cols:
            out["attention_mask"] = [1] * len(own)
        return out

    return ds.map(fix, with_indices=True)


def control_strings(row: dict) -> list[str]:
    """The Ministral control-token strings (serving_mistral.SPECIAL_TOKENS) a row's messages or tools hold as text."""
    blob = json.dumps([row.get("messages"), row.get("tools")], ensure_ascii=False)
    return [t for t in serving_mistral.SPECIAL_TOKENS if t in blob]


def mask_problem(row: dict, fmt) -> str | None:
    """Why the Mistral loss mask cannot train this row as rendered, read from the render alone
    (serving_mistral.control_ids: no tokenizer, no GPU) — None when it can: its control tokens no longer make one BOS
    and one </s>-closed span per collated assistant message (mistral_mask stops the run on that), or the spans they
    make cover other text than the assistant segments (a literal "[/INST]" in a user turn: mistral_mask would train
    the user's words, unseen — mistral_guard checks the first row only)."""
    text, want = serving_mistral.render_conversation(row["messages"], row.get("tools"), close_last=True)
    if text != fmt.render_training(row["messages"], row.get("tools"), history_turns(row)):
        return "its training text is not the render's"
    ids, offs = serving_mistral.control_ids(text)
    bos, eos = serving_mistral.SPECIAL_IDS[serving_mistral.BOS], serving_mistral.SPECIAL_IDS[serving_mistral.END_OF_TURN]
    spans = trainkit.mistral_spans(ids)
    if ids.count(bos) != 1 or len(spans) != len(want) or ids.count(eos) != len(want):
        return (f"{len(spans)} token spans, {ids.count(eos)} </s> and {ids.count(bos)} <s> for {len(want)} assistant "
                "segments")
    got = [(offs[a][0], offs[b - 1][1]) for a, b in spans]
    if got != want:
        k = next(j for j, (x, y) in enumerate(zip(got, want)) if x != y)
        return f"token span {k} covers characters {got[k]}, its assistant segment {want[k]}"
    return None


def mistral_guard(ex: dict, row: dict, tok, fmt) -> None:
    """Stop unless a tokenized training row is what the Mistral arm must train: exactly one BOS first, </s> last and
    trained, the trained token spans decoding to exactly the renderer's assistant segments (after the history), and no
    tool output among them."""
    ids, labels = list(ex["input_ids"]), list(ex["labels"])
    sid = fmt.special_ids or {}
    bos, eos = sid.get(serving_mistral.BOS, 1), sid.get(serving_mistral.END_OF_TURN, 2)
    if len(ids) < 2 or ids[0] != bos or ids[1] == bos:
        die(f"Mistral guard: the row does not start with exactly one BOS ({ids[:3]})")
    if ids[-1] != eos or labels[-1] != eos:
        die(f"Mistral guard: the row does not end with a trained </s> (ids …{ids[-2:]}, labels …{labels[-2:]})")
    spans, run = [], None
    for j, lab in enumerate(labels):
        if lab != -100 and run is None:
            run = j
        elif lab == -100 and run is not None:
            spans.append((run, j))
            run = None
    if run is not None:
        spans.append((run, len(labels)))
    got = [tok.decode(ids[a:b], skip_special_tokens=False) for a, b in spans]
    want = serving_mistral.assistant_segments(row["messages"], row["tools"])[history_spans(row, fmt):]
    if got != want:
        k = next((j for j, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        die(f"Mistral guard: the trained spans are not the renderer's assistant segments ({len(got)} vs {len(want)}; "
            f"first difference in span {k}: {got[k][:120] if k < len(got) else None!r} vs "
            f"{want[k][:120] if k < len(want) else None!r})")
    trained = "".join(got)
    leaks = [m.get("name", "?") for m in row["messages"] if m.get("role") == "tool"
             and (m.get("content") or "")[:200] and (m.get("content") or "")[:200] in trained]
    if leaks:
        die(f"Mistral guard: the result of {', '.join(leaks)} is trained on")


def family_report(fmt, train_rows: list[dict], val_rows: list[dict], max_seq_length: int) -> str:
    """A non-qwen dry run's extra lines: the rows each unservable rule drops (the qwen3.5 echo rule, kept so the row
    set is the Qwen arms'; the family's own rule), how many rows Ollama's collate merges assistant messages in, a check
    that every kept row renders one segment per collated assistant message, and — when MX_TOKENIZER_DIR names a folder
    with the pinned tokenizer.json and the `tokenizers` package is importable — the real token counts (BOS included)."""
    import os

    rows = train_rows + val_rows
    by_rule = {"qwen_echo": 0, fmt.name: 0}
    merged = 0
    kept: list[dict] = []
    for r in rows:
        q = serving.unservable(r["messages"])
        own = serving_mistral.unservable_mistral(r["messages"]) if fmt.name == "mistral" else []
        by_rule["qwen_echo"] += bool(q)
        by_rule[fmt.name] += bool(own)
        merged += bool(serving_mistral.merged_assistants(r["messages"]))
        if not fmt.unservable(r["messages"]):
            kept.append(r)
    bad = []
    for r in kept:
        text, spans = serving_mistral.render_conversation(r["messages"], r.get("tools"))
        n_assist = sum(1 for m in serving_mistral.collate(r["messages"]) if m.get("role") == "assistant")
        if len(spans) != n_assist or text != fmt.render_training(r["messages"], r.get("tools"), history_turns(r)):
            bad.append(r.get("id"))
    if bad:
        die(f"{len(bad)} rows do not render one segment per collated assistant message (e.g. {bad[:3]})")
    ctl = [(r.get("id"), control_strings(r)) for r in kept]
    ctl = [(i, c) for i, c in ctl if c]
    if ctl and fmt.name == "mistral":  # the rows the training mask would stop on, found here on CPU, not on the GPU
        breaking = [(r.get("id"), c, why) for r in kept if (c := control_strings(r)) and (why := mask_problem(r, fmt))]
        if breaking:
            die(f"{len(breaking)} rows hold a control-token string that the tokenizer reads as the control token and "
                f"that moves their token spans, so on the GPU the training mask (mistral_mask) would stop on them or "
                f"train other text than their replies: {[(i, c) for i, c, _ in breaking[:5]]} ({breaking[0][2]}); fix "
                "their text or drop them")
    lines = [f"{fmt.name}: unservable rows by rule {json.dumps(by_rule)} (a row is dropped when either rule hits); "
             f"{merged} rows have consecutive assistant messages Ollama's collate merges; {len(kept)} rows render "
             "one segment per collated assistant message."]
    if ctl:  # none of them changes its token spans (checked above): the mask takes them, the model reads control tokens
        lines.append(f"{fmt.name}: WARNING {len(ctl)} kept rows hold a control-token string as text (e.g. "
                     f"{ctl[:3]}): their token spans still match their segments, but the model reads the string as the "
                     "control token")
    tdir = os.environ.get("MX_TOKENIZER_DIR")
    if tdir and fmt.name == "mistral":
        try:
            from tokenizers import Tokenizer  # the HF tokenizers package only: never torch

            tk = Tokenizer.from_file(str(Path(tdir) / "tokenizer.json"))
            counts = sorted(len(tk.encode(fmt.render_training(r["messages"], r.get("tools"), history_turns(r)),
                                          add_special_tokens=False).ids) + 1 for r in kept)
            over = sum(1 for n in counts if n > max_seq_length)
            lines.append(f"{fmt.name}: tokens (pinned tokenizer, BOS included) median {trainkit.median(counts):,} · "
                         f"max {max(counts, default=0):,} · total {sum(counts):,} · over {max_seq_length:,}: {over}")
        except Exception as e:  # noqa: BLE001 — a dry run's extra figure, never a reason to stop
            lines.append(f"{fmt.name}: no token count ({type(e).__name__}: {e})")
    return "\n".join(lines)


def train(args, p: dict, train_rows: list[dict], val_rows: list[dict]) -> int:
    # Unsloth patches transformers and trl as it is imported, so it must come
    # first — before either is imported anywhere. Nothing above this line
    # imports torch (tests/test_train_dry.py).
    # isort: off
    import unsloth  # noqa: F401
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only

    import torch
    import transformers
    import trl
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    # isort: on

    if not torch.cuda.is_available():
        die("no CUDA GPU visible to PyTorch — check `nvidia-smi` and that the [train] extra installed the CUDA build")

    out = Path(p["output"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    load_kw = dict(model_name=args.base, max_seq_length=args.max_seq_length, full_finetuning=False,
                   load_in_4bit=False, load_in_16bit=True)
    if p["tiled_mlp"]:
        load_kw["unsloth_tiled_mlp"] = True
    try:
        # Embeddings in system RAM free ~1.9 GB of VRAM for long sequences.
        model, processor = FastModel.from_pretrained(offload_embedding=True, **load_kw)
    except TypeError:
        model, processor = FastModel.from_pretrained(**load_kw)
    # A multimodal base (Qwen3.5) hands back a processor; text work uses its tokenizer.
    tok = getattr(processor, "tokenizer", processor)

    fmt = chatfmt.get(args.family)
    mistral = fmt.name == "mistral"
    if mistral:
        # The render is serving_mistral's (Ollama's TEMPLATE, v13 layout). [INST] / [/INST] only give Unsloth's
        # collator and labels column below; the labels themselves are rebuilt from token ids (trainkit.mistral_labels).
        kind = "mistral-v13"
        instruction_part = args.instruction_part or serving_mistral.INST_OPEN
        response_part = args.response_part or serving_mistral.INST_CLOSE
    else:
        # The render is Ollama's (serving.py), ChatML whatever the base's own template says; the masking markers are
        # ChatML's, and the base's tokenizer must know them as single tokens.
        kind = template_kind(tok.chat_template or "") or "chatml"
        instruction_part = args.instruction_part or MARKERS[kind][0]
        response_part = args.response_part or MARKERS[kind][1]
    served_markers(tok, fmt)
    unservable_by_rule = {"qwen_echo": 0, "mistral": 0}

    def render(rows: list[dict]) -> tuple[list[dict], list[str], list[int], int, int]:
        kept, texts, lengths, dropped, unservable = [], [], [], 0, 0
        for r in rows:
            if fmt.unservable(r["messages"]):
                unservable += 1  # its served echo would read back as another call (serving.unservable): never trained
                if mistral:
                    unservable_by_rule["qwen_echo"] += bool(serving.unservable(r["messages"]))
                    unservable_by_rule["mistral"] += bool(serving_mistral.unservable_mistral(r["messages"]))
                continue
            text = render_text(r["messages"], r["tools"], history_turns(r), fmt)
            n = len(tok(text, add_special_tokens=fmt.prompt_add_special_tokens)["input_ids"])
            if n > args.max_seq_length:
                dropped += 1  # truncating would cut the answer — the part being learned
                continue
            kept.append(r)
            texts.append(text)
            lengths.append(n)
        return kept, texts, lengths, dropped, unservable

    train_kept, train_texts, train_lens, train_dropped, train_unservable = render(train_rows)
    val_rows_used = val_subset(val_rows, args.val_max_rows, args.val_seed)
    val_kept, val_texts, val_lens, val_dropped, val_unservable = render(val_rows_used)
    say(f"Rendered as the served student sees it ({serving.DESCRIPTION}): {len(train_texts)} train / "
        f"{len(val_texts)} val rows fit; {train_dropped + val_dropped} longer than {args.max_seq_length:,} tokens "
        f"dropped; {train_unservable + val_unservable} unservable dropped. Longest kept: {max(train_lens, default=0):,} tokens.")
    if not train_texts:
        die("no training row fits --max-seq-length")
    if mistral:
        segments = serving_mistral.assistant_segments(train_kept[0]["messages"], train_kept[0]["tools"])
        trained_text = "".join(segments[history_spans(train_kept[0], fmt):])
        leaks = [m.get("name", "?") for m in train_kept[0]["messages"]
                 if m.get("role") == "tool" and (m.get("content") or "")[:200]
                 and (m.get("content") or "")[:200] in trained_text]
        if leaks:
            die(f"the result of {', '.join(leaks)} would be trained on — the Mistral spans must hold assistant turns "
                "only (serving_mistral.render_conversation)")
        sample_spans = segments[history_spans(train_kept[0], fmt):]
    else:
        leaks = leaked_tool_results(train_kept[0], train_texts[0], instruction_part, response_part)
        if leaks:
            die(f"with these masking markers the result of {', '.join(leaks)} would be trained on — the base's template "
                "puts tool output inside the assistant turn; use a Qwen (ChatML) base")
        sample_spans = trained_spans(train_texts[0], instruction_part, response_part)[history_turns(train_kept[0]):]
    # What the model sees, and the part of it that is learned.
    (out / "sample_rendered.txt").write_text(train_texts[0], encoding="utf-8")
    (out / "sample_trained.txt").write_text("\n----- masked in between -----\n".join(sample_spans), encoding="utf-8")

    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, r=args.rank, lora_alpha=args.alpha, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=args.seed,
    )
    init_loaded = None
    if args.init_adapter is not None:
        # a warm start: the fresh LoRA takes the saved one's weights (a --resume then overwrites them with the
        # checkpoint's, which began from these)
        init_loaded = load_init_adapter(model, args.init_adapter)
        say(f"Warm start: {init_loaded['tensors']} LoRA tensors loaded from {trainkit.rel(args.init_adapter)}.")

    loss_log = (out / "loss.jsonl").open("a", encoding="utf-8")
    evals: list[tuple[int, float]] = []  # (update, validation loss)

    def log_line(d: dict) -> None:
        loss_log.write(json.dumps(d) + "\n")
        loss_log.flush()

    def run_validation(step: int, epoch) -> None:
        if not val_texts:
            return
        t = time.monotonic()
        v = validation_loss(model, trainer.eval_dataset, torch, bf16=bool(trainer.args.bf16))
        if v is None:
            return
        evals.append((step, v))
        log_line({"step": step, "epoch": epoch, "eval_loss": v, "eval_seconds": round(time.monotonic() - t, 1)})
        say(f"validation loss after {step} updates: {v:.4f} ({time.monotonic() - t:.0f}s)")

    class LossToFile(TrainerCallback):
        def on_log(self, _args, state, _control, logs=None, **_):
            if logs:
                log_line({"step": state.global_step, "epoch": state.epoch, **logs})

        def on_step_end(self, _args, state, _control, **_):
            if args.save_steps and state.global_step % args.save_steps == 0 and state.global_step < state.max_steps:
                ckpt_dir = out / "adapter-latest"
                model.save_pretrained(str(ckpt_dir))
                tok.save_pretrained(str(ckpt_dir))
                (ckpt_dir / "step.json").write_text(json.dumps({"step": state.global_step, "epoch": state.epoch,
                                                                "at": utc_now()}), encoding="utf-8")
                log_line({"step": state.global_step, "checkpoint": str(ckpt_dir)})
            if args.eval_steps and state.global_step % args.eval_steps == 0 and state.global_step < state.max_steps:
                run_validation(state.global_step, state.epoch)

    # A short linear warm-up as a step count: steady across transformers
    # releases, where warmup_ratio has been in flux.
    updates = updates_for(len(train_texts), args.batch, args.grad_accum, args.epochs, args.max_steps)
    cfg = SFTConfig(
        output_dir=str(out / "checkpoints"),
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,  # never: the Gated DeltaNet layers would carry state from one row into the next
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps or -1,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=max(1, int(updates * args.warmup_ratio)),
        optim=args.optim,
        weight_decay=args.weight_decay,
        logging_steps=1,
        eval_strategy="no",  # validation_loss instead: the Trainer's evaluation builds full fp32 logits
        # The adapter is saved once at the end (and adapter-latest every --save-steps); --checkpoint-steps adds the
        # Trainer's own full checkpoints, the only thing --resume can continue from.
        save_strategy="steps" if args.checkpoint_steps else "no",
        save_steps=args.checkpoint_steps or 500,
        save_total_limit=2,
        bf16=True,
        seed=args.seed,
        report_to="none",
        dataset_num_proc=1,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=Dataset.from_dict({"text": train_texts}),
        eval_dataset=Dataset.from_dict({"text": val_texts}) if val_texts else None,
        args=cfg,
        callbacks=[LossToFile()],
    )
    trainer = train_on_responses_only(trainer, instruction_part=instruction_part, response_part=response_part)
    resp_ids = tok(response_part, add_special_tokens=False)["input_ids"]
    inst_ids = tok(instruction_part, add_special_tokens=False)["input_ids"]

    def mask_followups(ds, rows: list[dict], which: str):
        ks = [history_turns(r) for r in rows]
        if ds is None or not any(ks):
            return ds
        if not {"input_ids", "labels"} <= set(ds.column_names) or len(ds) != len(ks):
            die(f"cannot mask the follow-ups' earlier replies in the {which} set (this Unsloth keeps no labels "
                "column) — rebuild the dataset without follow-ups, or update the masking in train.py")
        return ds.map(lambda ex, i: {"labels": mask_history_turns(ex["input_ids"], ex["labels"], resp_ids, inst_ids,
                                                                  ks[i])}, with_indices=True)

    if mistral:
        trainer.train_dataset = mistral_mask(trainer.train_dataset, train_kept, train_texts, tok, fmt, "training")
        trainer.eval_dataset = mistral_mask(trainer.eval_dataset, val_kept, val_texts, tok, fmt, "validation")
        mistral_guard(trainer.train_dataset[0], train_kept[0], tok, fmt)
        say("Mistral masking: one BOS, </s> trained, the decoded spans are the renderer's, no tool output trained "
            "(row 0 of the training set).")
    else:
        trainer.train_dataset = mask_followups(trainer.train_dataset, train_kept, "training")
        trainer.eval_dataset = mask_followups(trainer.eval_dataset, val_kept, "validation")
    ckpts, partial = resumable_checkpoints(out / "checkpoints") if args.resume else ([], [])
    if partial:
        say("--resume: skipping incomplete checkpoint(s) " + ", ".join(d.name for d in partial)
            + f" (each needs {', '.join(CHECKPOINT_FILES)})")
    resume_from = str(ckpts[-1]) if ckpts else None
    if resume_from:
        say(f"Resuming from {trainkit.rel(Path(resume_from))} (optimizer, scheduler and data position restored).")
        log_line({"resumed_from": resume_from, "at": utc_now()})
    else:
        if args.resume:
            say("--resume: no checkpoint yet under checkpoints/ — starting from the beginning.")
        run_validation(0, 0.0)  # the start's own loss on the validation rows (the base's, or the init adapter's)
    say("Training — the loss should fall over the first few dozen steps (Ctrl-C stops; --checkpoint-steps saves).")
    torch.cuda.reset_peak_memory_stats()
    t_train = time.monotonic()
    stats = trainer.train(resume_from_checkpoint=resume_from) if resume_from else trainer.train()
    train_seconds = time.monotonic() - t_train
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30

    # Save first: nothing after this point may cost the adapter.
    adapter_dir = out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    say(f"Adapter saved: {trainkit.rel(adapter_dir)}")
    try:
        run_validation(trainer.state.global_step, trainer.state.epoch)
    except torch.OutOfMemoryError as e:  # the adapter is safe; report and go on
        say(f"the final validation ran out of GPU memory ({e.__class__.__name__}); the adapter is saved")
        torch.cuda.empty_cache()
    gen = None
    if args.gen_check:
        items = gen_check_items(val_rows, args.gen_check, args.val_seed, fmt)
        say(f"Generation check: the first turn of {len(items)} fixed validation rows, greedy ...")
        gen = generation_check(model, tok, items, torch, FastModel, fmt=fmt)
        log_line({"step": trainer.state.global_step, "gen_check": {k: v for k, v in gen.items() if k != "failed_ids"}})
        say(f"Generation check: {gen['match']}/{gen['scored']} first turns match"
            + (f" ({gen['error']})" if gen.get("error") else "") + f" in {gen['seconds']}s.")
    loss_log.close()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    run = {
        **p,
        "finished_at": utc_now(),
        "minutes": round((time.monotonic() - t0) / 60, 1),
        "template": kind,
        "instruction_part": instruction_part,
        "response_part": response_part,
        "rows": {**p["rows"], "train_fit": len(train_texts), "val_fit": len(val_texts),
                 "dropped_too_long": train_dropped + val_dropped,
                 "dropped_unservable": train_unservable + val_unservable},
        "tokens": {"train_total": sum(train_lens), "train_max": max(train_lens), "val_total": sum(val_lens)},
        "steps": trainer.state.global_step,
        # the updates this process continued from (0: a fresh start): `minutes` covers only the rest, which
        # pipeline-train.sh's 12 h window guard scales up to a whole arm
        "resumed_from_step": int(Path(resume_from).name.rsplit("-", 1)[-1]) if resume_from else 0,
        "train_loss": getattr(stats, "training_loss", None),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        # the step-0 loss is the base's on a fresh start, the init adapter's on a warm one
        "base_eval_loss": evals[0][1] if evals and evals[0][0] == 0 and args.init_adapter is None else None,
        "start_eval_loss": evals[0][1] if evals and evals[0][0] == 0 else None,
        "init_adapter_loaded": init_loaded,
        "gen_check": gen,
        "last_eval_loss": evals[-1][1] if evals else None,
        "eval_losses": [{"step": st, "eval_loss": v} for st, v in evals],
        "train_seconds": round(train_seconds, 1),
        "train_tokens_per_second": round(sum(train_lens) * (trainer.state.epoch or 1) / train_seconds) if train_seconds else None,
        "versions": {"unsloth": getattr(unsloth, "__version__", "?"), "torch": torch.__version__,
                     "transformers": transformers.__version__, "trl": trl.__version__,
                     **trainkit.package_versions(("peft", "accelerate"))},
        "gpu": torch.cuda.get_device_name(0),
        "peak_vram_gb": round(peak_reserved, 1),
    }
    if mistral:  # main's run.json is unchanged for qwen
        run["rows"]["dropped_unservable_by_rule"] = dict(unservable_by_rule)
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    say(f"Done in {run['minutes']} min: loss {run['first_loss']} → {run['last_loss']}"
        + (f", validation {evals[0][1]:.3f} → {evals[-1][1]:.3f}" if evals else "") + ".")
    say(f"Adapter: {trainkit.rel(adapter_dir)} · settings: {trainkit.rel(out / 'run.json')}")
    say(f"Next: python -m souschef_lab.dpo --sft-adapter {trainkit.rel(adapter_dir)} --name <name>, or "
        f"python -m souschef_lab.export --adapter {trainkit.rel(adapter_dir)} --name <ollama-name> "
        f"--base-ollama-tag {fmt.base_ollama_tag} --create")
    return 0


if __name__ == "__main__":
    sys.exit(main())
