#!/usr/bin/env python3
"""The trainer image's self-test (DESIGN §6.4, risk 1).

Proves, in about a minute, what an hour of training would otherwise find
out: the GPU is visible to torch under this host driver (with the CUDA
forward-compatibility library when the driver is older than the cu130
wheels want), bf16 works, Unsloth / TRL / PEFT import, the pinned llama.cpp
quantizer and converter run, and gcloud is there.

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
              ("gcloud", check_gcloud)]
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
