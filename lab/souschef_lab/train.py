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

from . import fields, serving, trainkit
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
                                "fields_context", "checkpoint_steps", "resume")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="souschef_lab.train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="settings file (.yaml/.yml/.json) whose keys are these flags; a flag given here wins")
    p.add_argument("--base", default=RECIPE["base"], help="Hugging Face id of the base model (default: %(default)s)")
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
    p.add_argument("--seed", type=int, default=RECIPE["seed"])
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
    args.data = Path(args.data)
    if args.output is not None:
        args.output = Path(args.output)
    if args.checkpoint_steps < 0:
        p.error("--checkpoint-steps must be 0 or more")
    if args.save_steps < 0 or args.eval_steps < 0 or (args.val_max_rows is not None and args.val_max_rows < 1):
        p.error("--save-steps and --eval-steps must be 0 or more and --val-max-rows 1 or more")
    if args.rank < 1 or args.alpha < 1 or args.batch < 1 or args.grad_accum < 1 or args.lr <= 0 \
            or args.max_seq_length < 1024 or args.epochs <= 0:
        p.error("--rank, --alpha, --batch and --grad-accum must be 1 or more, --lr and --epochs above 0, "
                "--max-seq-length 1024 or more")
    return args


def render_text(messages: list[dict], tools: list[dict] | None, history: int = 0) -> str:
    """A row's training text: the conversation as Ollama renders it for the
    served student (serving.render_training — the qwen3.5 renderer with
    thinking disabled, every assistant turn opened by the empty think block
    the served generation prompt ends with). Never the base's chat template:
    its tools block and echoed tool calls are not the served bytes
    (docs/LAB.md §5 "Serving parity")."""
    return serving.render_training(messages, tools, history)


def served_markers(tok) -> None:
    """Stop unless the ChatML markers the serving render writes are single
    special tokens of this base's tokenizer: the masking (MARKERS) and the
    render assume Qwen's vocabulary."""
    missing = [t for t in serving.SPECIAL_TOKENS if len(tok(t, add_special_tokens=False)["input_ids"]) != 1]
    if missing:
        die(f"this base's tokenizer has no single token for {', '.join(missing)}: the serving render (Ollama's "
            "qwen3.5 renderer) and the ChatML masking need a Qwen3.5 base")


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
        "seed": args.seed, "tiled_mlp": tiled, "thinking": args.thinking, "data": str(args.data),
        "config": getattr(args, "config", None), "render": serving.DESCRIPTION,
        "manifest_hash": dataset_manifest(args.data).get("manifest_hash"),
    }
    if train_rows is not None:
        d["fields_context"] = fields_context_of(args, train_rows, val_rows or [])
        est = [estimate_tokens(r) for r in train_rows]
        val_rows = val_rows or []
        val_n = len(val_subset(val_rows, args.val_max_rows, args.seed))
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
              "save_steps", "checkpoint_steps", "resume", "val_max_rows", "seed", "rows", "tokens_estimate")


def render_plan(p: dict) -> str:
    return trainkit.render_plan("SFT plan (the thermometer recipe, docs/LAB.md §8)",
                                {k: v for k, v in p.items() if k not in ("rank", "alpha", "render")}, PLAN_ORDER)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.thinking:
        die("--thinking: the rows render in the served form (serving.py: Ollama's qwen3.5 renderer with thinking "
            "disabled, the empty think block first) and the traces carry no thinking — there is no thinking render")
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
    say(render_plan(p))
    if p["rows"]["followups"]:
        say(f"{p['rows']['followups']} rows are follow-ups: their earlier replies are masked out of the loss.")
    if args.dry_run:
        if importlib.util.find_spec("unsloth") is None:
            say("The [train] extra is not installed: run 'make -C lab train-env' (several GB, once) before training.")
        say("Dry run: the data is well-formed. Drop --dry-run to train.")
        return 0
    if importlib.util.find_spec("unsloth") is None:
        die("the [train] extra is not installed — run 'make -C lab train-env' first (several GB, once)")
    return train(args, p, train_rows, val_rows)


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

    # The render is Ollama's (serving.py), ChatML whatever the base's own template says; the masking markers are
    # ChatML's, and the base's tokenizer must know them as single tokens.
    kind = template_kind(tok.chat_template or "") or "chatml"
    instruction_part = args.instruction_part or MARKERS[kind][0]
    response_part = args.response_part or MARKERS[kind][1]
    served_markers(tok)

    def render(rows: list[dict]) -> tuple[list[dict], list[str], list[int], int, int]:
        kept, texts, lengths, dropped, unservable = [], [], [], 0, 0
        for r in rows:
            if serving.unservable(r["messages"]):
                unservable += 1  # its served echo would read back as another call (serving.unservable): never trained
                continue
            text = render_text(r["messages"], r["tools"], history_turns(r))
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            if n > args.max_seq_length:
                dropped += 1  # truncating would cut the answer — the part being learned
                continue
            kept.append(r)
            texts.append(text)
            lengths.append(n)
        return kept, texts, lengths, dropped, unservable

    train_kept, train_texts, train_lens, train_dropped, train_unservable = render(train_rows)
    val_rows_used = val_subset(val_rows, args.val_max_rows, args.seed)
    val_kept, val_texts, val_lens, val_dropped, val_unservable = render(val_rows_used)
    say(f"Rendered as the served student sees it ({serving.DESCRIPTION}): {len(train_texts)} train / "
        f"{len(val_texts)} val rows fit; {train_dropped + val_dropped} longer than {args.max_seq_length:,} tokens "
        f"dropped; {train_unservable + val_unservable} unservable dropped. Longest kept: {max(train_lens, default=0):,} tokens.")
    if not train_texts:
        die("no training row fits --max-seq-length")
    leaks = leaked_tool_results(train_kept[0], train_texts[0], instruction_part, response_part)
    if leaks:
        die(f"with these masking markers the result of {', '.join(leaks)} would be trained on — the base's template "
            "puts tool output inside the assistant turn; use a Qwen (ChatML) base")
    # What the model sees, and the part of it that is learned.
    (out / "sample_rendered.txt").write_text(train_texts[0], encoding="utf-8")
    (out / "sample_trained.txt").write_text(
        "\n----- masked in between -----\n".join(
            trained_spans(train_texts[0], instruction_part, response_part)[history_turns(train_kept[0]):]),
        encoding="utf-8")

    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, r=args.rank, lora_alpha=args.alpha, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=args.seed,
    )

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

    trainer.train_dataset = mask_followups(trainer.train_dataset, train_kept, "training")
    trainer.eval_dataset = mask_followups(trainer.eval_dataset, val_kept, "validation")
    ckpts = sorted((out / "checkpoints").glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[-1])) \
        if args.resume and (out / "checkpoints").is_dir() else []
    resume_from = str(ckpts[-1]) if ckpts else None
    if resume_from:
        say(f"Resuming from {trainkit.rel(Path(resume_from))} (optimizer, scheduler and data position restored).")
        log_line({"resumed_from": resume_from, "at": utc_now()})
    else:
        if args.resume:
            say("--resume: no checkpoint yet under checkpoints/ — starting from the beginning.")
        run_validation(0, 0.0)  # the base model's own loss on the validation rows, before any update
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
        "train_loss": getattr(stats, "training_loss", None),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "base_eval_loss": evals[0][1] if evals and evals[0][0] == 0 else None,
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
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    say(f"Done in {run['minutes']} min: loss {run['first_loss']} → {run['last_loss']}"
        + (f", validation {evals[0][1]:.3f} → {evals[-1][1]:.3f}" if evals else "") + ".")
    say(f"Adapter: {trainkit.rel(adapter_dir)} · settings: {trainkit.rel(out / 'run.json')}")
    say(f"Next: python -m souschef_lab.dpo --sft-adapter {trainkit.rel(adapter_dir)} --name <name>, or "
        f"python -m souschef_lab.export --adapter {trainkit.rel(adapter_dir)} --name <ollama-name> "
        "--base-ollama-tag qwen3.5:9b --create")
    return 0


if __name__ == "__main__":
    sys.exit(main())
