#!/usr/bin/env python3
"""mx_gguf — read a GGUF's metadata and check it is servable for its family (Mistral arm, docs/MISTRAL.md §(e)).

    python tools/mx_gguf.py check <file.gguf> --family mistral|qwen [--quant q4_k_m] [--out check.json] [--sha256]
    python tools/mx_gguf.py show <file.gguf>            # the scalar metadata, one key per line

Standard library only (no gguf-py), so the same check runs in the trainer-mx image (after the export and after
base-gguf) and in the sous-chef-mx image (before every `ollama create`). It reads the header, the metadata and the
tensor table only: a 5 GB file costs a few MB of reads. Exit 0 only when every check passes; 1 when one fails;
2 when the file is not a GGUF.

Mistral (Ministral-3-8B-Instruct-2512, llama.cpp `convert_hf_to_gguf.py` → arch `mistral3`):
  arch            general.architecture == "mistral3"
  shape           mistral3.block_count 34, embedding_length 4096, head_count_kv 8 (the 8B text model, not another)
  bos             tokenizer.ggml.add_bos_token true, bos_token_id 1, eos_token_id 2 (llama-server adds the one BOS:
                  neither the TEMPLATE nor the training text writes <s>)
  control_tokens  ids 1 2 3 4 5 6 7 8 9 17 18 32 are the expected strings, and every one is a CONTROL token (type 3):
                  llama-server tokenizes them from the TEMPLATE's text only as control tokens ([INST] must be ONE id)
  text_only       no clip.* key and no v.* / mm.* tensor (the Pixtral projector belongs in a separate mmproj file,
                  which export.find_gguf already skips)
  yarn            a YaRN rope (mistral3.rope.scaling.type yarn, factor 16) with its log multiplier
                  (rope.scaling.yarn_log_multiplier — llama.cpp's name — or rope.scaling.mscale_all_dim, Ollama's)
  attn_temp       the llama_4_scaling attention temperature (attention.temperature_scale, or rope.scaling_beta) = 0.1
  quant           general.file_type == the quant asked for (q4_k_m = 15), when --quant is given
Qwen3.5 (the qwen-base reference and r4): arch qwen35, add_bos_token false, text_only, quant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"GGUF"
# gguf value types (ggml/include/gguf.h)
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
_SCALAR = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f", BOOL: "<?", U64: "<Q",
           I64: "<q", F64: "<d"}
TOKEN_CONTROL = 3
# general.file_type (llama_ftype) of the quants the arm uses
FILE_TYPES = {"f16": 1, "q8_0": 7, "q4_k_m": 15}
# Ministral 3 (Tekken v13) control tokens, measured on the pinned tokenizer (f6fae97): the same ids as
# serving_mistral.SPECIAL_IDS (tests/… cross-check them when that module exists)
MISTRAL_SPECIAL_IDS = {"<s>": 1, "</s>": 2, "[INST]": 3, "[/INST]": 4, "[AVAILABLE_TOOLS]": 5, "[/AVAILABLE_TOOLS]": 6,
                       "[TOOL_RESULTS]": 7, "[/TOOL_RESULTS]": 8, "[TOOL_CALLS]": 9, "[SYSTEM_PROMPT]": 17,
                       "[/SYSTEM_PROMPT]": 18, "[ARGS]": 32}
MISTRAL_SHAPE = {"block_count": 34, "embedding_length": 4096, "attention.head_count_kv": 8}
FAMILIES = ("mistral", "qwen")
ARCH = {"mistral": "mistral3", "qwen": "qwen35"}


class GGUFError(ValueError):
    pass


class _Reader:
    def __init__(self, f: BinaryIO):
        self.f = f

    def take(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise GGUFError("truncated GGUF header")
        return b

    def scalar(self, t: int) -> Any:
        fmt = _SCALAR[t]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.scalar(U64)
        if n > 1 << 30:
            raise GGUFError(f"implausible string length {n}")
        return self.take(n).decode("utf-8", errors="replace")

    def value(self, t: int) -> Any:
        if t == STRING:
            return self.string()
        if t == ARRAY:
            et = self.scalar(U32)
            n = self.scalar(U64)
            if et == STRING:
                return [self.string() for _ in range(n)]
            if et == ARRAY:
                return [self.value(ARRAY) for _ in range(n)]
            if et not in _SCALAR:
                raise GGUFError(f"unknown array element type {et}")
            fmt = _SCALAR[et]
            size = struct.calcsize(fmt)
            raw = self.take(size * n)
            return list(struct.unpack(f"<{n}{fmt[1]}", raw)) if n else []
        if t not in _SCALAR:
            raise GGUFError(f"unknown value type {t}")
        return self.scalar(t)


def read(path: str | Path) -> dict:
    """{version, tensor_count, kv: {key: value}, tensors: [{name, dims, type}]} from a GGUF's header."""
    with open(path, "rb") as raw:
        f = _Reader(raw)
        if f.take(4) != MAGIC:
            raise GGUFError(f"{path} is not a GGUF file")
        version = f.scalar(U32)
        if version not in (2, 3):
            raise GGUFError(f"GGUF version {version} (2 or 3 expected)")
        n_tensors = f.scalar(U64)
        n_kv = f.scalar(U64)
        kv: dict[str, Any] = {}
        for _ in range(n_kv):
            key = f.string()
            kv[key] = f.value(f.scalar(U32))
        tensors = []
        for _ in range(n_tensors):
            name = f.string()
            nd = f.scalar(U32)
            dims = [f.scalar(U64) for _ in range(nd)]
            ttype = f.scalar(U32)
            f.scalar(U64)  # offset
            tensors.append({"name": name, "dims": dims, "type": ttype})
    return {"version": version, "tensor_count": n_tensors, "kv": kv, "tensors": tensors}


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _close(a: Any, b: float, tol: float = 1e-4) -> bool:
    return isinstance(a, (int, float)) and abs(float(a) - b) <= tol


def checks(meta: dict, family: str, quant: str | None = None) -> list[dict]:
    if family not in FAMILIES:
        raise ValueError(f"family {family!r}: one of {', '.join(FAMILIES)}")
    kv, tensors = meta["kv"], meta["tensors"]
    arch = kv.get("general.architecture")
    want_arch = ARCH[family]
    out = [_check("arch", arch == want_arch, f"general.architecture {arch!r} (want {want_arch!r})")]
    clip = sorted(k for k in kv if k.startswith("clip."))
    vis = sorted(t["name"] for t in tensors if t["name"].startswith(("v.", "mm.")))
    out.append(_check("text_only", not clip and not vis,
                      "no clip.* keys, no v./mm. tensors" if not clip and not vis else
                      f"clip keys {clip[:3]}, vision/projector tensors {vis[:3]} ({len(vis)}): a monolithic file"))
    bos_flag = kv.get("tokenizer.ggml.add_bos_token")
    if family == "mistral":
        a = str(arch)
        shape = {k: kv.get(f"{a}.{k}") for k in MISTRAL_SHAPE}
        out.append(_check("shape", shape == MISTRAL_SHAPE, f"{shape} (want {MISTRAL_SHAPE})"))
        bos, eos = kv.get("tokenizer.ggml.bos_token_id"), kv.get("tokenizer.ggml.eos_token_id")
        out.append(_check("bos", bos_flag is True and bos == 1 and eos == 2,
                          f"add_bos_token {bos_flag!r}, bos_token_id {bos!r}, eos_token_id {eos!r} (want true, 1, 2)"))
        toks = kv.get("tokenizer.ggml.tokens") or []
        types = kv.get("tokenizer.ggml.token_type") or []
        bad = []
        for text, i in MISTRAL_SPECIAL_IDS.items():
            got = toks[i] if i < len(toks) else None
            typ = types[i] if i < len(types) else None
            if got != text or typ != TOKEN_CONTROL:
                bad.append(f"{i}: {got!r} type {typ!r} (want {text!r} type {TOKEN_CONTROL})")
        out.append(_check("control_tokens", not bad,
                          f"{len(MISTRAL_SPECIAL_IDS)} ids are their control tokens" if not bad else "; ".join(bad)))
        rtype = kv.get(f"{a}.rope.scaling.type")
        factor = kv.get(f"{a}.rope.scaling.factor")
        logmul = kv.get(f"{a}.rope.scaling.yarn_log_multiplier", kv.get(f"{a}.rope.scaling.mscale_all_dim"))
        out.append(_check("yarn", rtype == "yarn" and _close(factor, 16.0) and logmul is not None,
                          f"rope.scaling.type {rtype!r}, factor {factor!r}, log multiplier {logmul!r} (want yarn, 16, set)"))
        temp = kv.get(f"{a}.attention.temperature_scale", kv.get(f"{a}.rope.scaling_beta"))
        out.append(_check("attn_temp", _close(temp, 0.1),
                          f"attention temperature scale {temp!r} (want 0.1, llama_4_scaling_beta)"))
    else:
        # unset counts as false: llama.cpp and Ollama add no BOS to a Qwen (BPE) vocab without the key, and the
        # pipeline's own exports leave it unset — r6's GGUF (served on int since 2026-10-17) has no such key
        out.append(_check("bos", bos_flag is not True,
                          f"add_bos_token {bos_flag!r} (want false or unset: Qwen3.5 has no BOS)"))
    if quant:
        want = FILE_TYPES.get(quant.lower())
        ft = kv.get("general.file_type")
        out.append(_check("quant", want is not None and ft == want, f"general.file_type {ft!r} (want {want!r}, {quant})"))
    return out


SUMMARY_KEYS = ("general.architecture", "general.name", "general.file_type", "general.quantization_version",
                "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token",
                "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id")


def summary(meta: dict) -> dict:
    kv = meta["kv"]
    arch = str(kv.get("general.architecture") or "")
    out = {k: kv[k] for k in SUMMARY_KEYS if k in kv}
    for k, v in kv.items():
        if arch and k.startswith(arch + ".") and not isinstance(v, list):
            out[k] = v
    out["tokens"] = len(kv.get("tokenizer.ggml.tokens") or [])
    out["has_chat_template"] = "tokenizer.chat_template" in kv
    out["tensor_count"] = meta["tensor_count"]
    return out


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def check_file(path: str | Path, family: str, quant: str | None = None, *, sha256: bool = False) -> dict:
    """The check document (gguf-check.json): {file, size, family, ok, checks, metadata[, sha256]}."""
    p = Path(path)
    meta = read(p)
    cs = checks(meta, family, quant)
    doc = {"file": p.name, "size": p.stat().st_size, "family": family, "quant": quant,
           "ok": all(c["ok"] for c in cs), "checks": cs, "metadata": summary(meta)}
    if sha256:
        doc["sha256"] = file_sha256(p)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mx_gguf", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="check a GGUF for its family; exit 1 on a failed check")
    c.add_argument("gguf")
    c.add_argument("--family", choices=FAMILIES, required=True)
    c.add_argument("--quant", choices=sorted(FILE_TYPES))
    c.add_argument("--out", help="write the check document here (JSON)")
    c.add_argument("--sha256", action="store_true", help="also hash the file (reads all of it)")
    s = sub.add_parser("show", help="print the scalar metadata")
    s.add_argument("gguf")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "show":
            for k, v in summary(read(a.gguf)).items():
                print(f"{k} = {v!r}")
            return 0
        doc = check_file(a.gguf, a.family, a.quant, sha256=a.sha256)
    except (GGUFError, OSError) as e:
        print(f"mx_gguf: {e}", file=sys.stderr)
        return 2
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    for ch in doc["checks"]:
        print(f"{'ok  ' if ch['ok'] else 'FAIL'} {ch['name']:<15} {ch['detail']}")
    print(f"mx_gguf: {doc['file']} ({doc['family']}): {'ok' if doc['ok'] else 'FAILED'}")
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
