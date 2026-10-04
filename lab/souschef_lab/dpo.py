"""DPO: the SFT adapter continued on the student's own preference pairs.

Copied from the thermometer's dpo.py (breadstack-thermometer/lab/in-house-model;
the pattern, never imported), on the 9B recipe docs/LAB.md §8 fixes:

  policy      the SFT adapter (train.py), loaded trainable and continued
  reference   the frozen SFT model, as log-probabilities computed once before
              the first update: at that point the policy IS the SFT model, so
              its log-probs of each pair's chosen and rejected completion,
              taken without gradients, are exactly the frozen reference's.
              They are cached pair by pair in ref_logps.jsonl under a
              fingerprint of the adapter and the pairs (ref_model=None).
  loss        sigmoid DPO, β 0.1; lr 5e-6, 10% warm-up, linear decay, one
              epoch, micro-batch 1 × 16, bf16.
              DECISION (docs/LAB.md §8): rpo_alpha=None — plain sigmoid DPO as
              the recipe states; `--rpo-alpha 1.0` restores the thermometer's
              default (the chosen completion's NLL added to the loss).
  length      full-length pairs, never truncated: each sequence (prompt + one
              completion) up to --max-length (the SFT run's max_seq_length),
              both together up to --max-pair-tokens; a longer pair is dropped
              and counted.

How a pair is computed (the LabDPOTrainer keeps TRL's loss, metrics and
loop): each pair is rendered with the base's chat template — the prompt with
the generation prompt, each completion as the text the template adds for
that turn up to the end-of-turn token — and tokenized as the model sees it
served. The chosen and rejected sequences run one at a time, unpadded, with
the logits kept for the completion only and their log-probs taken in fp32 a
chunk at a time.

Pairs: data/pairs/train.jsonl and val.jsonl (pairs.py): {id, prompt,
chosen:[one assistant message], rejected:[one assistant message], tools,
meta}. Settings: flags or --config settings.yaml|json.

  python -m souschef_lab.dpo --sft-adapter runs/adapters/sous-chef-9b-v1/adapter --name sous-chef-9b-v1-dpo --dry-run
  python -m souschef_lab.dpo --sft-adapter runs/adapters/sous-chef-9b-v1/adapter --name sous-chef-9b-v1-dpo

--dry-run never imports torch (tests/test_train_dry.py).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

from . import trainkit
from .train import render_kwargs, render_text
from .trainkit import die, estimate_tokens, read_jsonl, say, updates_for, utc_now, val_subset

# The recipe (docs/LAB.md §8).
RECIPE = {"beta": 0.1, "rpo_alpha": None, "lr": 5e-6, "epochs": 1.0, "grad_accum": 16, "warmup_ratio": 0.1,
          "eval_steps": 25, "val_max_pairs": 40, "seed": 3407, "tiled_mlp": "auto", "optim": "adamw_8bit",
          "loss_type": "sigmoid"}
THERMOMETER_RPO_ALPHA = 1.0  # what --rpo-alpha 1.0 restores
PAIR_COLUMNS = ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids")
REF_FILE = "ref_logps.jsonl"
CONFIG_KEYS = tuple(k for k in RECIPE if k != "loss_type") + ("sft_adapter", "pairs", "name", "output", "max_steps",
                                                             "max_length", "max_pair_tokens", "reference")


# --- Pairs -----------------------------------------------------------------------------------------------------------


def pairs_manifest(pairs_dir: Path) -> dict:
    return trainkit.read_json(Path(pairs_dir) / "pairs-manifest.json", {}) or {}


def check_pairs(rows: list[dict], where: str) -> list[str]:
    """Shape problems in pair rows (empty = fine): a prompt, one assistant
    message on each side (its tool calls in the stored shape), a tools list."""
    problems = []
    for i, r in enumerate(rows):
        pid = r.get("id") or i
        if not isinstance(r.get("prompt"), list) or not r["prompt"]:
            problems.append(f"{where} pair {pid}: no prompt")
        for side in ("chosen", "rejected"):
            msgs = r.get(side)
            if not (isinstance(msgs, list) and len(msgs) == 1 and isinstance(msgs[0], dict)
                    and msgs[0].get("role") == "assistant"):
                problems.append(f"{where} pair {pid}: {side} is not one assistant message")
                continue
            for k, c in enumerate(msgs[0].get("tool_calls") or []):
                problems += trainkit.tool_call_problems(c, f"{where} pair {pid}: {side} tool_calls[{k}]")
        if not isinstance(r.get("tools"), list):
            problems.append(f"{where} pair {pid}: no tools list")
    return problems


def render_pair(apply_template, row: dict, render_kw: dict, eos: str) -> tuple[str, str, str] | str:
    """(prompt text, chosen text, rejected text), or why the pair cannot be
    rendered. The prompt is the conversation with the generation prompt; a
    completion is what the template adds for that turn after it, up to and
    including the end-of-turn token — where the model stops."""
    prompt = row["prompt"]
    p_text = render_text(apply_template, prompt, row["tools"], add_generation_prompt=True, **render_kw)
    out = [p_text]
    for side in ("chosen", "rejected"):
        full = render_text(apply_template, prompt + row[side], row["tools"], add_generation_prompt=False, **render_kw)
        if not full.startswith(p_text):
            return "render_mismatch"
        comp = full[len(p_text):]
        end = comp.rfind(eos)
        if end < 0:
            return "no_end_of_turn"
        out.append(comp[:end + len(eos)])
    return out[0], out[1], out[2]


def tokenize_pairs(rows: list[dict], tokenize, apply_template, render_kw: dict, eos: str, eos_id: int,
                   max_length: int, max_pair_tokens: int) -> tuple[list[dict], dict]:
    """The pairs as token ids (prompt, chosen, rejected) and the count of
    those dropped, by reason: a sequence over max_length or both over
    max_pair_tokens (never cut), a render that does not extend its prompt, a
    completion that does not end its turn."""
    kept, dropped = [], {}
    for r in rows:
        got = render_pair(apply_template, r, render_kw, eos)
        if isinstance(got, str):
            dropped[got] = dropped.get(got, 0) + 1
            continue
        p_ids, c_ids, r_ids = (tokenize(t) for t in got)
        if not c_ids or not r_ids or c_ids[-1] != eos_id or r_ids[-1] != eos_id:
            dropped["no_end_of_turn"] = dropped.get("no_end_of_turn", 0) + 1
            continue
        side = max(len(p_ids) + len(c_ids), len(p_ids) + len(r_ids))
        both = 2 * len(p_ids) + len(c_ids) + len(r_ids)
        if side > max_length or both > max_pair_tokens:
            dropped["too_long"] = dropped.get("too_long", 0) + 1
            continue
        kept.append({"id": str(r.get("id")), "prompt_input_ids": p_ids, "chosen_input_ids": c_ids,
                     "rejected_input_ids": r_ids, "tokens": both,
                     "type": (r.get("meta") or {}).get("type"), "kind": (r.get("meta") or {}).get("kind")})
    return kept, dropped


# --- The reference ---------------------------------------------------------------------------------------------------


def read_reference(path: Path, fp: dict) -> dict[str, tuple[float, float]]:
    """The cached reference log-probs of this run's pairs ({} when there are
    none yet). The first line holds the fingerprint they were computed under;
    a cache from another SFT adapter, other pairs or other settings is
    refused rather than reused."""
    path = Path(path)
    if not path.exists():
        return {}
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        return {}
    head = json.loads(lines[0])
    if bad := trainkit.fingerprint_mismatch(head.get("fingerprint") or {}, fp):
        die(f"{trainkit.rel(path)} was computed for another run (it differs in {', '.join(bad)}): the reference must "
            "be this SFT adapter's log-probs of these pairs — start a new --name")
    out = {}
    for x in lines[1:]:
        try:
            d = json.loads(x)
        except ValueError:
            continue  # a torn last line: that pair is computed again
        out[d["id"]] = (float(d["chosen"]), float(d["rejected"]))
    return out


def start_reference(path: Path, fp: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"fingerprint": fp, "what": "the frozen SFT model's summed log-probs of each "
                                                                 "pair's chosen and rejected completion, before any "
                                                                 "DPO update"}) + "\n", encoding="utf-8")


# --- The forward pass (torch and the model are passed in) ------------------------------------------------------------


def completion_logps(model, prompt_ids, comp_ids, torch, chunk: int = 1024):
    """(summed log-prob of the completion, the per-token log-probs, the sum
    and count of its logits): one sequence, unpadded; the model returns the
    logits of the completion's positions only; each chunk's log-softmax is
    taken in fp32 and recomputed in the backward pass instead of kept."""
    ids = torch.cat([prompt_ids, comp_ids]).unsqueeze(0)
    n = comp_ids.numel()
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=n + 1)
    logits = out.logits[0, -(n + 1):-1, :]  # the position before each completion token predicts it

    def part(lg, ix):
        return torch.log_softmax(lg.float(), dim=-1).gather(-1, ix.unsqueeze(-1)).squeeze(-1)

    lps, lsum = [], 0.0
    for s in range(0, n, chunk):
        lg, ix = logits[s:s + chunk], comp_ids[s:s + chunk]
        if torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            lps.append(checkpoint(part, lg, ix, use_reentrant=False))
        else:
            lps.append(part(lg, ix))
        with torch.no_grad():
            lsum = lsum + lg.float().sum()
    tok = torch.cat(lps)
    return tok.sum(), tok, lsum, logits.shape[0] * logits.shape[1]


def pair_forward(model, batch: dict, torch, want_nll: bool) -> dict:
    """TRL's concatenated_forward, for these pairs: the outputs DPOTrainer's
    loss and metrics read (chosen_logps, rejected_logps, the mean logits,
    nll_loss when rpo_alpha is set), each sequence run on its own."""
    res = {"chosen_logps": [], "rejected_logps": []}
    means = {"chosen": [0.0, 0], "rejected": [0.0, 0]}
    nll_sum, nll_n = 0.0, 0
    for i in range(batch["prompt_input_ids"].shape[0]):
        prompt = batch["prompt_input_ids"][i][batch["prompt_attention_mask"][i].bool()]
        for side in ("chosen", "rejected"):
            comp = batch[f"{side}_input_ids"][i][batch[f"{side}_attention_mask"][i].bool()]
            total, tok, lsum, lcount = completion_logps(model, prompt, comp, torch)
            res[f"{side}_logps"].append(total)
            means[side][0] = means[side][0] + lsum
            means[side][1] += lcount
            if side == "chosen" and want_nll:
                nll_sum = nll_sum - tok.sum()
                nll_n += tok.numel()
    out = {k: torch.stack(v) for k, v in res.items()}
    for side in ("chosen", "rejected"):
        s, c = means[side]
        out[f"mean_{side}_logits"] = (s / max(c, 1)) if torch.is_tensor(s) else torch.tensor(0.0)
    if want_nll:
        out["nll_loss"] = nll_sum / max(nll_n, 1)
    return out


def trl_import_fix() -> None:
    """TRL 0.24 caches transformers' package checks, which now return
    (found, version): a tuple, truthy even for a missing package, so
    `from trl import DPOTrainer` tries to import mergekit and fails. Unsloth
    coerces them on import; this does the same where Unsloth is not imported."""
    try:
        import trl.import_utils as tiu
    except Exception:
        return
    for attr in list(vars(tiu)):
        if attr.startswith("_") and attr.endswith("_available") and isinstance(getattr(tiu, attr), tuple):
            cached = getattr(tiu, attr)
            setattr(tiu, attr, bool(cached and cached[0]))


def make_trainer_class(base_cls, torch):
    """DPOTrainer (Unsloth's patched one) with the dataset taken as dpo.py
    tokenized it and the forward pass above."""

    class LabDPOTrainer(base_cls):
        def __init__(self, *args, model=None, **kwargs):
            # TRL 0.24 writes to model.warnings_issued, which transformers 5 no longer defines.
            if model is not None and not isinstance(getattr(model, "warnings_issued", None), dict):
                model.warnings_issued = {}
            super().__init__(*args, model=model, **kwargs)

        def _prepare_dataset(self, dataset, *args, **kwargs):
            if not set(PAIR_COLUMNS) <= set(dataset.column_names):
                raise ValueError("LabDPOTrainer takes the pairs dpo.py rendered and tokenized (tokenize_pairs)")
            return dataset

        def concatenated_forward(self, model, batch, is_ref_model: bool = False):
            return pair_forward(model, batch, torch, want_nll=self.args.rpo_alpha is not None)

    return LabDPOTrainer


def dpo_config_kwargs(p: dict, ckpt_root: Path, updates: int) -> dict:
    """The DPOConfig: TRL's truncation and padding tricks off (the pairs are
    full length and run unpadded), the reference supplied precomputed, the
    loss plain sigmoid unless rpo_alpha is set."""
    return dict(
        output_dir=str(ckpt_root), per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=p["grad_accum"], num_train_epochs=p["epochs"],
        max_steps=p["max_steps"] or -1, learning_rate=p["lr"], lr_scheduler_type="linear",
        warmup_steps=max(1, int(updates * p["warmup_ratio"])), optim=p["optim"], weight_decay=0.0,
        beta=p["beta"], rpo_alpha=p["rpo_alpha"], loss_type=p["loss_type"], label_smoothing=0.0,
        max_length=None, max_prompt_length=None, max_completion_length=None, padding_free=False,
        precompute_ref_log_probs=False, use_logits_to_keep=True, disable_dropout=True, generate_during_eval=False,
        logging_steps=1, eval_strategy="no", save_strategy="no", bf16=True, seed=p["seed"], data_seed=p["seed"],
        report_to="none", dataset_num_proc=1, remove_unused_columns=False,
    )


# --- The command -----------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="souschef_lab.dpo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="settings file (.yaml/.yml/.json) whose keys are these flags; a flag given here wins")
    p.add_argument("--sft-adapter", type=Path, help="the SFT run's adapter folder (runs/adapters/<name>/adapter)")
    p.add_argument("--pairs", type=Path, default=trainkit.PAIRS_DIR,
                   help="pairs.py's folder with train.jsonl / val.jsonl (default: lab/data/pairs)")
    p.add_argument("--name", help="run name: runs/adapters/<name>")
    p.add_argument("--output", type=Path, help="output folder (overrides --name)")
    p.add_argument("--beta", type=float, default=RECIPE["beta"], help="DPO β (default: %(default)s)")
    p.add_argument("--rpo-alpha", type=float, default=RECIPE["rpo_alpha"],
                   help="weight of the chosen completion's NLL added to the loss (default: none — plain sigmoid DPO, "
                        f"docs/LAB.md §8; {THERMOMETER_RPO_ALPHA} restores the thermometer's default; 0 = none)")
    p.add_argument("--lr", type=float, default=RECIPE["lr"], help="learning rate (default: %(default)s)")
    p.add_argument("--epochs", type=float, default=RECIPE["epochs"], help="(default: %(default)s)")
    p.add_argument("--grad-accum", type=int, default=RECIPE["grad_accum"], help="pairs per update (default: %(default)s)")
    p.add_argument("--warmup-ratio", type=float, default=RECIPE["warmup_ratio"], help="(default: %(default)s)")
    p.add_argument("--optim", default=RECIPE["optim"], help="the optimizer (default: %(default)s)")
    p.add_argument("--max-steps", type=int, help="stop after this many updates (an lr probe)")
    p.add_argument("--max-length", type=int,
                   help="longest sequence, the prompt and one completion, in tokens (default: the SFT run's "
                        "max_seq_length); a longer pair is dropped and counted")
    p.add_argument("--max-pair-tokens", type=int,
                   help="the prompt twice and both completions, at most (default: twice --max-length)")
    p.add_argument("--eval-steps", type=int, default=RECIPE["eval_steps"],
                   help="validation metrics every N updates, and before the first and after the last (default: "
                        "%(default)s)")
    p.add_argument("--val-max-pairs", type=int, default=RECIPE["val_max_pairs"],
                   help="validation pairs scored (a seeded subset; default: %(default)s)")
    p.add_argument("--seed", type=int, default=RECIPE["seed"])
    p.add_argument("--tiled-mlp", choices=["auto", "on", "off"], default=RECIPE["tiled_mlp"],
                   help="Unsloth tiled MLP (auto: on above 24,576 tokens) (default: %(default)s)")
    p.add_argument("--reference", type=Path,
                   help=f"reuse the reference log-probs another run of the same SFT adapter and pairs computed (its "
                        f"{REF_FILE}; checked against this run's fingerprint)")
    p.add_argument("--reference-only", action="store_true",
                   help="compute the reference log-probs (the SFT model's, of every pair), write them, and stop")
    p.add_argument("--dry-run", "--check", dest="dry_run", action="store_true",
                   help="validate the pairs and print the resolved plan, then stop (no GPU, no torch)")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_parser()
    argv = trainkit.apply_config(p, sys.argv[1:] if argv is None else argv, CONFIG_KEYS)
    args = p.parse_args(argv)
    if args.sft_adapter is None:
        p.error("--sft-adapter is required (the SFT run's adapter folder)")
    args.sft_adapter = Path(args.sft_adapter)
    args.pairs = Path(args.pairs)
    if args.output is not None:
        args.output = Path(args.output)
    if not (args.name or args.output):
        p.error("--name (or --output): the run's folder goes by it")
    if args.beta <= 0 or args.lr <= 0 or args.grad_accum < 1 or args.eval_steps < 0 or args.epochs <= 0:
        p.error("--beta, --lr and --epochs must be above 0, --grad-accum 1 or more, --eval-steps 0 or more")
    if args.rpo_alpha is not None and args.rpo_alpha <= 0:
        args.rpo_alpha = None  # 0 (or less) means: no NLL term
    if args.reference_only and args.reference:
        p.error("--reference-only computes a reference: not with --reference")
    return args


def sft_run(adapter: Path) -> dict:
    """The run.json beside the SFT adapter folder ({} when there is none)."""
    return trainkit.read_json(Path(adapter).parent / "run.json", {}) or {}


def adapter_sha256(adapter: Path) -> str:
    files = sorted(Path(adapter).glob("adapter_model*.safetensors")) or sorted(Path(adapter).glob("adapter_model*"))
    return trainkit.data_sha256(files) if files else ""


def plan(args, sft: dict, train_rows: list[dict] | None = None, val_rows: list[dict] | None = None) -> dict:
    name = args.name or Path(args.output).name
    out = Path(args.output) if args.output else trainkit.ADAPTERS_DIR / name
    max_length = args.max_length or int(sft.get("max_seq_length") or 32768)
    tiled = args.tiled_mlp == "on" or (args.tiled_mlp == "auto" and max_length > 24576)
    d = {
        "kind": "dpo", "name": name, "output": str(out), "adapter": str(out / "adapter"),
        "base": sft.get("base"), "sft_adapter": str(args.sft_adapter), "sft_run": sft.get("name"),
        "pairs": str(args.pairs), "config": getattr(args, "config", None),
        "precision": "bf16 base, fp32 LoRA", "rank": sft.get("rank"), "alpha": sft.get("alpha"),
        "lora": {"r": sft.get("rank"), "alpha": sft.get("alpha"), "from": "the SFT adapter, continued"},
        "loss_type": RECIPE["loss_type"], "beta": args.beta, "rpo_alpha": args.rpo_alpha,
        "lr": args.lr, "epochs": args.epochs, "max_steps": args.max_steps, "batch": 1,
        "grad_accum": args.grad_accum, "effective_batch": args.grad_accum, "warmup_ratio": args.warmup_ratio,
        "optim": args.optim, "scheduler": "linear", "max_length": max_length, "max_seq_length": max_length,
        "max_pair_tokens": args.max_pair_tokens or 2 * max_length, "tiled_mlp": tiled, "seed": args.seed,
        "eval_steps": args.eval_steps, "val_max_pairs": args.val_max_pairs, "thinking": False,
        "render": "chat_template", "reference_file": str(out / REF_FILE),
        "reference": "the frozen SFT model: the policy's own log-probs before the first update, precomputed once "
                     f"and cached ({REF_FILE}; ref_model=None)",
        "manifest_hash": pairs_manifest(args.pairs).get("manifest_hash"),
    }
    if train_rows is not None:
        val_rows = val_rows or []
        est = [estimate_tokens({"messages": r["prompt"] + r["chosen"], "tools": r["tools"]}) for r in train_rows]
        kinds: dict[str, int] = {}
        for r in train_rows:
            t = str((r.get("meta") or {}).get("type") or "?")
            kinds[t] = kinds.get(t, 0) + 1
        d["rows"] = {"train": len(train_rows), "val": len(val_rows),
                     "val_scored": len(val_subset(val_rows, args.val_max_pairs, args.seed)), "by_type": kinds}
        d["tokens_estimate"] = {"side_median": trainkit.median(est), "side_max": max(est, default=0),
                                "chars_per_token": trainkit.CHARS_PER_TOKEN}
        d["updates"] = updates_for(len(train_rows), 1, args.grad_accum, args.epochs, args.max_steps)
    return d


PLAN_ORDER = ("kind", "name", "output", "adapter", "base", "sft_adapter", "sft_run", "pairs", "config", "manifest_hash",
              "precision", "lora", "loss_type", "beta", "rpo_alpha", "lr", "epochs", "max_steps", "updates", "batch",
              "grad_accum", "effective_batch", "warmup_ratio", "optim", "scheduler", "max_length", "max_pair_tokens",
              "tiled_mlp", "eval_steps", "val_max_pairs", "seed", "reference", "reference_file", "rows",
              "tokens_estimate")


def render_plan(p: dict) -> str:
    skip = ("rank", "alpha", "render", "thinking", "max_seq_length")
    return trainkit.render_plan("DPO plan (the thermometer recipe, docs/LAB.md §8; rpo_alpha None = plain sigmoid)",
                                {k: v for k, v in p.items() if k not in skip}, PLAN_ORDER)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_rows = read_jsonl(Path(args.pairs) / "train.jsonl")
    val_rows = read_jsonl(Path(args.pairs) / "val.jsonl")
    if not train_rows:
        die(f"no pairs in {trainkit.rel(args.pairs)}/train.jsonl — run `python -m souschef_lab.pairs` first")
    if problems := check_pairs(train_rows, "train") + check_pairs(val_rows, "val"):
        die("the pairs have problems:\n  " + "\n  ".join(problems[:20]))
    sft = sft_run(args.sft_adapter)
    has_adapter = (Path(args.sft_adapter) / "adapter_config.json").exists()
    if not has_adapter and not args.dry_run:
        die(f"{args.sft_adapter} has no adapter_config.json — pass the SFT run's adapter folder (…/adapter)")
    p = plan(args, sft, train_rows, val_rows)
    say(render_plan(p))
    if not has_adapter:
        say(f"note: {args.sft_adapter} has no adapter_config.json yet (train.py writes it); the run would stop here.")
    if args.dry_run:
        if importlib.util.find_spec("unsloth") is None:
            say("The [train] extra is not installed: pip install -e 'lab[train]' before training.")
        say("Dry run: the pairs are well-formed. Drop --dry-run to train.")
        return 0
    if importlib.util.find_spec("unsloth") is None:
        die("the [train] extra is not installed — pip install -e 'lab[train]' first")
    return run(args, p, sft, train_rows, val_rows)


def reference_fingerprint(args, p: dict) -> dict:
    """What the reference log-probs depend on — the SFT adapter, the pairs,
    how they are rendered and cut — and nothing else, so runs that differ only
    in β, the learning rate or the length of the run share them."""
    return {"kind": "dpo-reference", "base": p["base"], "render": p["render"], "thinking": p["thinking"],
            "max_length": p["max_length"], "max_pair_tokens": p["max_pair_tokens"],
            "data_sha256": trainkit.data_sha256([Path(args.pairs) / "train.jsonl", Path(args.pairs) / "val.jsonl"]),
            "sft_adapter_sha256": adapter_sha256(args.sft_adapter)}


def run(args, p: dict, sft: dict, train_rows: list[dict], val_rows: list[dict]) -> int:
    import os

    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    out = Path(p["output"])
    out.mkdir(parents=True, exist_ok=True)
    # isort: off
    import unsloth  # noqa: F401  (first: it patches transformers and trl)
    from unsloth import FastModel

    import torch
    import transformers
    import trl
    from datasets import Dataset
    from peft import PeftModel
    from transformers import TrainerCallback

    trl_import_fix()
    from trl import DPOConfig, DPOTrainer
    # isort: on

    if not torch.cuda.is_available():
        die("no CUDA GPU visible to PyTorch")
    t0 = time.monotonic()
    load_kw = dict(model_name=str(Path(args.sft_adapter).resolve()), max_seq_length=p["max_length"],
                   full_finetuning=False, load_in_4bit=False, load_in_16bit=True)
    if p["tiled_mlp"]:
        load_kw["unsloth_tiled_mlp"] = True
    try:
        model, processor = FastModel.from_pretrained(offload_embedding=True, **load_kw)
    except TypeError:
        model, processor = FastModel.from_pretrained(**load_kw)
    tok = getattr(processor, "tokenizer", processor)
    if not isinstance(model, PeftModel):
        die("the SFT adapter did not load as a PEFT model: nothing to continue")
    lora_params = [(n, q) for n, q in model.named_parameters() if "lora_" in n]
    if not lora_params or not all(q.requires_grad for _, q in lora_params):
        die("the SFT adapter's LoRA weights are not trainable: the policy would not move")
    say(f"Policy: {trainkit.rel(args.sft_adapter)} continued, {sum(q.numel() for _, q in lora_params):,} trainable "
        f"weights on {p['base']} in bf16 · reference: {p['reference']}")

    render_kw = render_kwargs(tok.chat_template or "", False)
    eos = tok.eos_token or "<|im_end|>"
    eos_id = tok.convert_tokens_to_ids(eos)

    def tokenize(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    val_used = val_subset(val_rows, args.val_max_pairs, args.seed)
    train_pairs, dropped = tokenize_pairs(train_rows, tokenize, tok.apply_chat_template, render_kw, eos, eos_id,
                                          p["max_length"], p["max_pair_tokens"])
    val_pairs, val_dropped = tokenize_pairs(val_used, tokenize, tok.apply_chat_template, render_kw, eos, eos_id,
                                            p["max_length"], p["max_pair_tokens"])
    say(f"Tokenized: {len(train_pairs)} train / {len(val_pairs)} val pairs kept; dropped "
        f"{dict(sorted(dropped.items())) or 'none'} (train), {dict(sorted(val_dropped.items())) or 'none'} (val). "
        f"Longest pair {max((x['tokens'] for x in train_pairs), default=0):,} tokens.")
    if not train_pairs:
        die("no pair fits --max-length and --max-pair-tokens")

    # The reference: the policy's own log-probs before any update, pair by
    # pair, cached so a crash loses little; or another run's (--reference).
    ref_fp = reference_fingerprint(args, p)
    ref_path = Path(p["reference_file"])
    if args.reference:
        ref = read_reference(args.reference, ref_fp)
        if not ref:
            die(f"--reference {args.reference}: no reference log-probs there")
        if Path(args.reference).resolve() != ref_path.resolve():
            ref_path.write_text(Path(args.reference).read_text(encoding="utf-8"), encoding="utf-8")
        say(f"Reference log-probs of {len(ref)} pairs from {trainkit.rel(Path(args.reference))}.")
    else:
        ref = read_reference(ref_path, ref_fp)
        if not ref_path.exists():
            start_reference(ref_path, ref_fp)
    todo = [x for x in train_pairs + val_pairs if x["id"] not in ref]

    updates = updates_for(len(train_pairs), 1, args.grad_accum, args.epochs, args.max_steps)
    trainer_cls = make_trainer_class(DPOTrainer, torch)
    columns = {c: [x[c] for x in train_pairs] for c in PAIR_COLUMNS}
    loss_log = (out / "loss.jsonl").open("a", encoding="utf-8")

    def log_line(d: dict) -> None:
        loss_log.write(json.dumps(d) + "\n")
        loss_log.flush()

    class ToFile(TrainerCallback):
        def on_log(self, _a, state, _c, logs=None, **_):
            if logs:
                log_line({"step": state.global_step, "epoch": state.epoch, **logs})

        def on_step_end(self, _a, state, _control, **_):
            if args.eval_steps and state.global_step % args.eval_steps == 0 and state.global_step < state.max_steps:
                validate(state.global_step)

    cfg = DPOConfig(**dpo_config_kwargs(p, out / "checkpoints", updates))
    placeholder = {**columns, "ref_chosen_logps": [0.0] * len(train_pairs), "ref_rejected_logps": [0.0] * len(train_pairs)}
    trainer = trainer_cls(model=model, ref_model=None, args=cfg, train_dataset=Dataset.from_dict(placeholder),
                          processing_class=tok, callbacks=[ToFile()])

    t_ref = time.monotonic()
    if todo:
        say(f"Reference log-probs for {len(todo)} pairs (the SFT model, before any update) …")
        model.train()  # the kernels the training forward uses; no gradients
        with torch.no_grad(), trainkit.amp(torch, cfg.bf16), ref_path.open("a", encoding="utf-8") as f:
            for i, x in enumerate(todo, 1):
                batch = trainer._prepare_inputs(trainer.data_collator([{c: x[c] for c in PAIR_COLUMNS}]))
                o = pair_forward(trainer.model, batch, torch, want_nll=False)
                c, r = float(o["chosen_logps"][0]), float(o["rejected_logps"][0])
                ref[x["id"]] = (c, r)
                f.write(json.dumps({"id": x["id"], "chosen": c, "rejected": r}) + "\n")
                f.flush()
                if i % 25 == 0 or i == len(todo):
                    say(f"  {i}/{len(todo)} ({time.monotonic() - t_ref:.0f}s)")
    ref_seconds = time.monotonic() - t_ref
    if args.reference_only:
        say(f"Reference ready: {len(ref)} pairs in {trainkit.rel(ref_path)} ({ref_seconds:.0f}s). Reuse it: "
            f"--reference {trainkit.rel(ref_path)}")
        return 0
    trainer.train_dataset = Dataset.from_dict({
        **columns, "ref_chosen_logps": [ref[x["id"]][0] for x in train_pairs],
        "ref_rejected_logps": [ref[x["id"]][1] for x in train_pairs]})
    evals: list[dict] = []

    def validate(step: int) -> None:
        if not val_pairs:
            return
        t = time.monotonic()
        sums: dict[str, float] = {}
        was = trainer.model.training
        with torch.no_grad(), trainkit.amp(torch, cfg.bf16):  # as the Trainer's forward runs
            for x in val_pairs:
                row = {c: x[c] for c in PAIR_COLUMNS}
                row["ref_chosen_logps"], row["ref_rejected_logps"] = ref[x["id"]]
                batch = trainer._prepare_inputs(trainer.data_collator([row]))
                loss, metrics = trainer.get_batch_loss_metrics(trainer.model, batch, train_eval="eval")
                for k, v in {"eval_loss": float(loss), **metrics}.items():
                    sums[k] = sums.get(k, 0.0) + float(v)
        if was:
            trainer.model.train()
        rec = {k: v / len(val_pairs) for k, v in sums.items()}
        evals.append({"step": step, **rec})
        log_line({"step": step, **rec, "eval_seconds": round(time.monotonic() - t, 1)})
        say(f"validation after {step} updates: loss {rec.get('eval_loss', 0):.4f}, margin "
            f"{rec.get('eval_rewards/margins', 0):+.3f}, accuracy {rec.get('eval_rewards/accuracies', 0):.2f}")

    validate(0)  # before any update the policy is the reference: the rewards must be about 0
    torch.cuda.reset_peak_memory_stats()
    t_train = time.monotonic()
    trainer.train()
    seconds = time.monotonic() - t_train
    steps = trainer.state.global_step
    adapter_dir = out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    say(f"Adapter saved: {trainkit.rel(adapter_dir)}")
    try:
        validate(steps)
    except torch.OutOfMemoryError:
        say("the final validation ran out of GPU memory; the adapter is saved")
        torch.cuda.empty_cache()
    loss_log.close()
    logged = [h for h in trainer.state.log_history if "loss" in h]
    keys = ("loss", "rewards/margins", "rewards/accuracies", "logps/chosen", "logps/rejected", "nll_loss")

    def pick(h: dict | None):
        return {k: h.get(k) for k in keys} if h else None

    run_rec = {
        **p, "finished_at": utc_now(), "minutes": round((time.monotonic() - t0) / 60, 1),
        "pairs": {"train": len(train_pairs), "val": len(val_pairs), "dropped": dropped, "val_dropped": val_dropped},
        "tokens": {"train_total": sum(x["tokens"] for x in train_pairs),
                   "longest_pair": max(x["tokens"] for x in train_pairs)},
        "steps": steps, "first": pick(logged[0] if logged else None), "last": pick(logged[-1] if logged else None),
        "eval": evals, "seconds": round(seconds, 1), "reference_seconds": round(ref_seconds, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1), "gpu": torch.cuda.get_device_name(0),
        "versions": {"unsloth": getattr(unsloth, "__version__", "?"), "torch": torch.__version__,
                     "transformers": transformers.__version__, "trl": trl.__version__,
                     **trainkit.package_versions(("peft", "accelerate"))},
    }
    (out / "run.json").write_text(json.dumps(run_rec, indent=2), encoding="utf-8")
    if run_rec["first"] and run_rec["last"]:
        say(f"Done: loss {run_rec['first']['loss']} → {run_rec['last']['loss']}, margin "
            f"{run_rec['last'].get('rewards/margins')}, chosen log-prob {run_rec['first'].get('logps/chosen')} → "
            f"{run_rec['last'].get('logps/chosen')} (it must not fall steadily).")
    say(f"Next: python -m souschef_lab.export --adapter {trainkit.rel(adapter_dir)} --name <ollama-name> "
        "--base-ollama-tag qwen3.5:9b --create, then the exam (eval.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
