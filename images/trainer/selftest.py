#!/usr/bin/env python3
"""The trainer image's self-test (DESIGN §6.4, risk 1).

Proves, in about a minute, what an hour of training would otherwise find
out: the GPU is visible to torch under this host driver (with the CUDA
forward-compatibility library when the driver is older than the cu130
wheels want), bf16 works, Unsloth / TRL / PEFT import, the pinned llama.cpp
quantizer and converter run, gcloud is there, and the lab's seed tools
(TRAIN_SEEDS' and TRAIN_ARMS' settings and selection, the generation check's parser) and
warm start (--init-adapter: a saved LoRA loaded into a fresh one, on a tiny
CPU model through the same train.load_init_adapter) work in this image.

  python /opt/trainer/selftest.py [--no-gpu] [--upload <run id>]

--no-gpu    skip what needs a GPU (the image build runs this)
--upload    write the report to gs://$PIPELINE_BUCKET/runs/<run id>/train/selftest.json

Prints the report as JSON; exit 0 only when every check passed.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback

LLAMA = os.environ.get("UNSLOTH_LLAMA_CPP_PATH", "/opt/llama.cpp")


def run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def check_driver() -> dict:
    smi = shutil.which("nvidia-smi")
    if not smi:
        raise RuntimeError("nvidia-smi not found (no GPU attached?)")
    p = run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip())
    return {"gpus": [line.strip() for line in p.stdout.splitlines() if line.strip()],
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH", "")}


def check_torch_gpu() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__} (cuda {torch.version.cuda}) sees no GPU")
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    b = (a @ a).float().sum().item()
    torch.cuda.synchronize()
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)), "bf16": torch.cuda.is_bf16_supported(),
            "matmul_finite": b == b, "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}


def check_unsloth() -> dict:
    import peft
    import trl
    import unsloth  # noqa: F401  (refuses to import without a GPU)

    return {"unsloth": getattr(unsloth, "__version__", "?"), "trl": trl.__version__, "peft": peft.__version__}


def check_train_env_cpu() -> dict:
    import peft
    import torch
    import transformers
    import trl

    return {"torch": torch.__version__, "cuda_build": torch.version.cuda, "transformers": transformers.__version__,
            "trl": trl.__version__, "peft": peft.__version__}


def check_llama_cpp() -> dict:
    q = os.path.join(LLAMA, "llama-quantize")
    p = run([q, "--help"])
    if "usage" not in (p.stdout + p.stderr).lower():
        raise RuntimeError(f"{q} --help: {p.returncode} {(p.stderr or p.stdout)[:300]}")
    c = run([sys.executable, os.path.join(LLAMA, "convert_hf_to_gguf.py"), "--help"])
    if c.returncode != 0:
        raise RuntimeError(f"convert_hf_to_gguf.py --help: {c.stderr[-500:]}")
    pinned = {}
    try:
        with open(os.path.join(LLAMA, "SOUS_CHEF_LLAMA_CPP.json"), encoding="utf-8") as f:
            pinned = json.load(f)
    except OSError:
        pass
    return {"path": LLAMA, "pinned": pinned}


def check_lab() -> dict:
    import souschef_lab.export  # noqa: F401
    import souschef_lab.train  # noqa: F401

    entry = "/opt/lab/tools/pipeline-train.sh"
    return {"lab": "/opt/lab", "pipeline_train": os.path.isfile(entry)}


def check_seed_tools() -> dict:
    """The r6 seed tools of this image's train.py, without torch: the pipeline settings' defaults are r4's (one seed,
    3407, from the base), a TRAIN_SEEDS list and a TRAIN_ARMS list (base and warm arms) resolve, selection picks the
    lowest loss and names its arm, and the generation check's parser reads a qwen3.5 tool call."""
    from souschef_lab import train

    d = train.pipeline_config({}, None, "gs://b")
    if (d["seeds"], d["seed_flag"], d["multi"], d["init_adapter"], d["gen_check"]) != ([3407], False, False, "", 0):
        raise RuntimeError(f"the default training settings changed: {d}")
    m = train.pipeline_config({"TRAIN_SEEDS": "3407,2,3", "INIT_ADAPTER": "base/a"}, None, "gs://b")
    if m["seeds"] != [3407, 2, 3] or not m["multi"] or m["init_adapter"] != "gs://b/base/a":
        raise RuntimeError(f"TRAIN_SEEDS / INIT_ADAPTER resolve wrongly: {m}")
    a = train.pipeline_config({"TRAIN_ARMS": "base:3407,base:2,warm:3407", "INIT_ADAPTER": "base/a"}, None, "gs://b")
    if [(x["arm"], x["tag"], x["lr"]) for x in a["arms"]] != [("base:3407", "s3407", train.RECIPE["lr"]),
                                                            ("base:2", "s2", train.RECIPE["lr"]),
                                                            ("warm:3407", "w3407", train.WARM_LR)] or not a["multi"]:
        raise RuntimeError(f"TRAIN_ARMS resolves wrongly: {a}")
    sel = train.select_seed([{"seed": 3407, "last_eval_loss": 0.3},
                             {"seed": 3407, "last_eval_loss": 0.2, "start": "init_adapter"}])
    if sel["selected"] != 1 or sel["seeds"][1]["arm"] != "warm:3407":
        raise RuntimeError(f"select_seed kept the wrong arm: {sel}")
    call = '<tool_call>\n<function=f>\n<parameter=a>\n{"x": 1}\n</parameter>\n</function>\n</tool_call>'
    if train.parse_tool_calls(call) != [("f", {"a": {"x": 1}})]:
        raise RuntimeError("the generation check cannot parse a qwen3.5 tool call")
    return {"default_seeds": d["seeds"], "arms": [x["arm"] for x in a["arms"]], "select": "ok", "gen_check_parse": "ok"}


def check_warm_start() -> dict:
    """--init-adapter's loader on a tiny CPU model: a saved LoRA's weights land in a fresh LoRA exactly, and an
    adapter with other target modules is refused (train.load_init_adapter, the function the SFT calls after Unsloth's
    get_peft_model)."""
    import contextlib
    import io
    import tempfile

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    from souschef_lab import train

    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=64)

    def lora(seed: int, targets: list[str]):
        torch.manual_seed(0)
        base = LlamaForCausalLM(cfg)
        torch.manual_seed(seed)
        return get_peft_model(base, LoraConfig(r=4, lora_alpha=4, target_modules=targets, lora_dropout=0.0))

    src = lora(1, ["q_proj", "v_proj"])
    with torch.no_grad():
        for n, p in src.named_parameters():
            if "lora_" in n:
                p.normal_()
    with tempfile.TemporaryDirectory() as tmp:
        src.save_pretrained(tmp)
        dst = lora(2, ["q_proj", "v_proj"])
        got = train.load_init_adapter(dst, tmp)
        want = {n: p for n, p in src.named_parameters() if "lora_" in n}
        have = {n: p for n, p in dst.named_parameters() if "lora_" in n}
        if set(want) != set(have) or any(not torch.equal(want[n], have[n]) for n in want):
            raise RuntimeError("the init adapter's LoRA weights did not land in the fresh LoRA")
        refused = False
        try:
            with contextlib.redirect_stderr(io.StringIO()):  # the refusal's own message: expected here
                train.load_init_adapter(lora(3, ["q_proj", "k_proj"]), tmp)
        except SystemExit:
            refused = True
        if not refused:
            raise RuntimeError("an adapter with other target modules was loaded")
    return {"tensors": got["tensors"], "mismatch_refused": True}


def check_gcloud() -> dict:
    g = shutil.which("gcloud")
    if not g:
        raise RuntimeError("gcloud not on PATH")
    p = run([g, "--version"], timeout=60)
    return {"gcloud": (p.stdout.splitlines() or ["?"])[0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--upload", metavar="RUN_ID")
    ns = ap.parse_args()

    checks = [("train_env", check_train_env_cpu), ("llama_cpp", check_llama_cpp), ("lab", check_lab),
              ("seed_tools", check_seed_tools), ("warm_start", check_warm_start), ("gcloud", check_gcloud)]
    if not ns.no_gpu:
        checks = [("driver", check_driver), ("torch_gpu", check_torch_gpu), ("unsloth", check_unsloth)] + checks

    report = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": platform.node(),
              "rung": os.environ.get("TRAIN_RUNG"), "attempt": os.environ.get("TRAIN_ATTEMPT"), "checks": {}}
    ok = True
    for name, fn in checks:
        t0 = time.time()
        try:
            detail = fn()
            report["checks"][name] = {"ok": True, "s": round(time.time() - t0, 1), "detail": detail}
        except Exception as e:  # noqa: BLE001 - every failure is reported, none hides another
            ok = False
            report["checks"][name] = {"ok": False, "s": round(time.time() - t0, 1),
                                      "error": f"{type(e).__name__}: {e}",
                                      "trace": traceback.format_exc(limit=3)[-1500:]}
    report["ok"] = ok
    text = json.dumps(report, indent=2)
    print(text)

    if ns.upload:
        bucket = os.environ.get("PIPELINE_BUCKET")
        if bucket:
            base = bucket.rstrip("/") if bucket.startswith("gs://") else f"gs://{bucket}"
            dest = f"{base}/runs/{ns.upload}/train/selftest.json"
            p = subprocess.run(["gcloud", "storage", "cp", "-", dest, "--content-type=application/json"],
                               input=text, capture_output=True, text=True, check=False)
            if p.returncode != 0:
                print(f"selftest: upload to {dest} failed: {p.stderr.strip()[:300]}", file=sys.stderr)
        else:
            print("selftest: --upload without PIPELINE_BUCKET; not uploaded", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
