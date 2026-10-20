"""tools/mx_gguf.py: the GGUF reader and the per-family servability check (docs/MISTRAL.md §(e))."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import gguf_synth as G  # noqa: E402
import mx_gguf  # noqa: E402

R4_GGUF = Path(os.environ.get("MX_R4_GGUF") or "/home/gtong/projects/breadstack-sous-chef/lab/runs/exports/r4/r4-q4_k_m.gguf")


def failed(doc: dict) -> set[str]:
    return {c["name"] for c in doc["checks"] if not c["ok"]}


def test_a_converted_ministral_passes(tmp_path):
    p = G.write(tmp_path / "m.gguf", G.mistral_kv())
    doc = mx_gguf.check_file(p, "mistral", "q4_k_m", sha256=True)
    assert doc["ok"], doc["checks"]
    assert {c["name"] for c in doc["checks"]} == {"arch", "text_only", "shape", "bos", "control_tokens", "yarn", "attn_temp", "quant"}
    assert doc["metadata"]["general.architecture"] == "mistral3" and len(doc["sha256"]) == 64


@pytest.mark.parametrize("over,name", [
    ({"general__architecture": (G.STRING, "llama")}, "arch"),
    ({"tokenizer__ggml__add_bos_token": (G.BOOL, False)}, "bos"),
    ({"tokenizer__ggml__bos_token_id": None}, "bos"),
    ({"mistral3__block_count": (G.U32, 40)}, "shape"),  # the 14B, not the 8B
    ({"mistral3__attention__temperature_scale": None}, "attn_temp"),
    ({"mistral3__rope__scaling__type": (G.STRING, "linear")}, "yarn"),
    ({"general__file_type": (G.U32, 7)}, "quant"),
    ({"clip__has_vision_encoder": (G.BOOL, True)}, "text_only"),
])
def test_each_mistral_failure_is_named(tmp_path, over, name):
    doc = mx_gguf.check_file(G.write(tmp_path / "m.gguf", G.mistral_kv(**over)), "mistral", "q4_k_m")
    assert not doc["ok"] and name in failed(doc)


def test_ollamas_own_key_names_are_accepted(tmp_path):
    """An Ollama-format mistral3 file names the YaRN multiplier mscale_all_dim and the temperature scaling_beta
    (llama/compat handle_mistral3 maps them): the check reads either name."""
    kv = G.mistral_kv(mistral3__rope__scaling__yarn_log_multiplier=None, mistral3__attention__temperature_scale=None,
                      mistral3__rope__scaling__mscale_all_dim=(G.F32, 1.0), mistral3__rope__scaling_beta=(G.F32, 0.1))
    assert mx_gguf.check_file(G.write(tmp_path / "m.gguf", kv), "mistral", "q4_k_m")["ok"]


def test_a_control_token_that_is_not_control_fails(tmp_path):
    kv = G.mistral_kv()
    for i, (k, t, v) in enumerate(kv):
        if k == "tokenizer.ggml.token_type":
            types = list(v[1])
            types[32] = 1  # [ARGS] as a normal token: llama-server would split "[ARGS]" into BPE pieces
            kv[i] = (k, t, (v[0], types))
    doc = mx_gguf.check_file(G.write(tmp_path / "m.gguf", kv), "mistral")
    assert failed(doc) == {"control_tokens"} and "32:" in next(c["detail"] for c in doc["checks"] if c["name"] == "control_tokens")


def test_a_monolithic_file_with_vision_tensors_fails(tmp_path):
    p = G.write(tmp_path / "m.gguf", G.mistral_kv(), tensors=["token_embd.weight", "v.blk.0.attn_q.weight", "mm.1.weight"])
    assert "text_only" in failed(mx_gguf.check_file(p, "mistral"))


def test_qwen_family(tmp_path):
    assert mx_gguf.check_file(G.write(tmp_path / "q.gguf", G.qwen_kv()), "qwen", "q4_k_m")["ok"]
    assert failed(mx_gguf.check_file(G.write(tmp_path / "q2.gguf", G.qwen_kv(file_type=1)), "qwen", "q4_k_m")) == {"quant"}
    assert "arch" in failed(mx_gguf.check_file(G.write(tmp_path / "m.gguf", G.mistral_kv()), "qwen"))
    # the pipeline's own Qwen exports leave add_bos_token unset (r6's served GGUF, 2026-10-17): no BOS either way
    assert mx_gguf.check_file(G.write(tmp_path / "q3.gguf", G.qwen_kv(tokenizer__ggml__add_bos_token=None)), "qwen", "q4_k_m")["ok"]
    assert "bos" in failed(mx_gguf.check_file(G.write(tmp_path / "q4.gguf", G.qwen_kv(tokenizer__ggml__add_bos_token=(G.BOOL, True))), "qwen"))


def test_cli_exit_codes(tmp_path, capsys):
    good = G.write(tmp_path / "m.gguf", G.mistral_kv())
    out = tmp_path / "check.json"
    assert mx_gguf.main(["check", str(good), "--family", "mistral", "--quant", "q4_k_m", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["ok"] is True
    assert mx_gguf.main(["check", str(good), "--family", "qwen"]) == 1
    (tmp_path / "x.gguf").write_bytes(b"NOPE" + b"\0" * 64)
    assert mx_gguf.main(["check", str(tmp_path / "x.gguf"), "--family", "qwen"]) == 2
    (tmp_path / "t.gguf").write_bytes(b"GGUF\3\0\0\0")
    assert mx_gguf.main(["check", str(tmp_path / "t.gguf"), "--family", "qwen"]) == 2


def test_special_ids_match_the_renderer():
    """The GGUF check and the renderer/masking must agree on the control-token ids (serving_mistral, when present)."""
    sys.path.insert(0, str(HERE.parents[1]))
    try:
        from souschef_lab import serving_mistral
    except ImportError:
        pytest.skip("souschef_lab.serving_mistral is not in this tree yet")
    assert dict(serving_mistral.SPECIAL_IDS) == mx_gguf.MISTRAL_SPECIAL_IDS


@pytest.mark.skipif(not R4_GGUF.is_file(), reason="no local r4 GGUF (MX_R4_GGUF)")
def test_the_real_r4_gguf_reads_as_qwen35():
    doc = mx_gguf.check_file(R4_GGUF, "qwen", "q4_k_m")
    assert doc["ok"], doc["checks"]
    assert doc["metadata"]["general.architecture"] == "qwen35" and doc["metadata"]["tokens"] == 248320
    assert not mx_gguf.check_file(R4_GGUF, "mistral")["ok"]
