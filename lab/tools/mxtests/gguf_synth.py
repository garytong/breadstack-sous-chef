"""A tiny GGUF v3 writer for the Mistral arm's tests (stdlib; the layout gguf-py's GGUFWriter writes, checked
against it on 2026-10-16: tools/mx_gguf.py reads a gguf-py-written mistral3 file the same way)."""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
MISTRAL_IDS = {"<s>": 1, "</s>": 2, "[INST]": 3, "[/INST]": 4, "[AVAILABLE_TOOLS]": 5, "[/AVAILABLE_TOOLS]": 6,
               "[TOOL_RESULTS]": 7, "[/TOOL_RESULTS]": 8, "[TOOL_CALLS]": 9, "[SYSTEM_PROMPT]": 17,
               "[/SYSTEM_PROMPT]": 18, "[ARGS]": 32}


def _s(x: str) -> bytes:
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _v(t: int, v: Any) -> bytes:
    if t == STRING:
        return _s(v)
    if t == ARRAY:
        et, items = v
        out = struct.pack("<IQ", et, len(items))
        return out + b"".join(_v(et, i) for i in items)
    fmt = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f", BOOL: "<?", U64: "<Q",
           I64: "<q", F64: "<d"}[t]
    return struct.pack(fmt, v)


def write(path: str | Path, kv: list[tuple[str, int, Any]], tensors: list[str] = ("token_embd.weight",)) -> Path:
    body = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, t, v in kv:
        body += _s(k) + struct.pack("<I", t) + _v(t, v)
    off = 0
    for name in tensors:
        body += _s(name) + struct.pack("<I", 2) + struct.pack("<QQ", 4, 8) + struct.pack("<I", 0) + struct.pack("<Q", off)
        off += 128
    body += b"\0" * (32 - len(body) % 32) + b"\0" * off
    p = Path(path)
    p.write_bytes(body)
    return p


def mistral_kv(**over) -> list[tuple[str, int, Any]]:
    toks = [f"t{i}" for i in range(40)]
    types = [1] * 40
    for t, i in MISTRAL_IDS.items():
        toks[i], types[i] = t, 3
    d: dict[str, tuple[int, Any]] = {
        "general.architecture": (STRING, "mistral3"), "general.file_type": (U32, 15),
        "mistral3.block_count": (U32, 34), "mistral3.embedding_length": (U32, 4096),
        "mistral3.attention.head_count_kv": (U32, 8), "mistral3.rope.scaling.type": (STRING, "yarn"),
        "mistral3.rope.scaling.factor": (F32, 16.0), "mistral3.rope.scaling.yarn_log_multiplier": (F32, 1.0),
        "mistral3.attention.temperature_scale": (F32, 0.1), "tokenizer.ggml.model": (STRING, "gpt2"),
        "tokenizer.ggml.pre": (STRING, "tekken"), "tokenizer.ggml.tokens": (ARRAY, (STRING, toks)),
        "tokenizer.ggml.token_type": (ARRAY, (I32, types)), "tokenizer.ggml.bos_token_id": (U32, 1),
        "tokenizer.ggml.eos_token_id": (U32, 2), "tokenizer.ggml.add_bos_token": (BOOL, True),
        "tokenizer.ggml.add_eos_token": (BOOL, False)}
    for k, v in over.items():
        key = k.replace("__", ".")
        if v is None:
            d.pop(key, None)
        else:
            d[key] = v
    return [(k, t, v) for k, (t, v) in d.items()]


def qwen_kv(file_type: int = 15, **over) -> list[tuple[str, int, Any]]:
    d: dict[str, tuple[int, Any]] = {
        "general.architecture": (STRING, "qwen35"), "general.file_type": (U32, file_type),
        "qwen35.block_count": (U32, 33), "tokenizer.ggml.model": (STRING, "gpt2"), "tokenizer.ggml.pre": (STRING, "qwen35"),
        "tokenizer.ggml.tokens": (ARRAY, (STRING, ["a", "b"])), "tokenizer.ggml.add_bos_token": (BOOL, False)}
    for k, v in over.items():
        key = k.replace("__", ".")
        if v is None:
            d.pop(key, None)
        else:
            d[key] = v
    return [(k, t, v) for k, (t, v) in d.items()]
