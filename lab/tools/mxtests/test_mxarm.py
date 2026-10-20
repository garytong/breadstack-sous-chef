"""tools/mxarm.py: the write guard, the serving asserts, the Stage 1 gate, the Stage 3 trust/verdict, the local eval
store, the job bodies and the image build files (docs/MISTRAL.md). Offline: no GCP, no GPU, no model; Ollama is
tools/mxtests/fake_ollama.py."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
LAB = TOOLS.parent
REPO = LAB.parent
for p in (str(LAB), str(TOOLS), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import gguf_synth as G  # noqa: E402
import mxarm as M  # noqa: E402

EXP = M.EXP_ROOT
RUN = "mx-20261017-m3-8b"
R6 = M.R6
FAKE_OLLAMA = HERE / "fake_ollama.py"
# The spec's Modelfile (SPEC §2a), used only when the renderer change has not landed lab/base-modelfiles/ yet
SPEC_MODELFILE = '''# Ministral 3 8B Instruct 2512, Mistral v13 layout
TEMPLATE """{{- $tools := false }}
{{- range $i, $m := .Messages }}
{{- if eq $m.Role "system" }}[SYSTEM_PROMPT]{{ $m.Content }}[/SYSTEM_PROMPT]
{{- else if eq $m.Role "user" }}
{{- if and $.Tools (not $tools) }}[AVAILABLE_TOOLS]{{ $.Tools }}[/AVAILABLE_TOOLS]{{ $tools = true }}{{ end }}[INST]{{ $m.Content }}[/INST]
{{- else if eq $m.Role "assistant" }}{{ $m.Content }}
{{- range $m.ToolCalls }}[TOOL_CALLS]{{ .Function.Name }}[ARGS]{{ .Function.Arguments }}{{ end }}
{{- if or $m.ToolCalls (ne (len (slice $.Messages $i)) 1) }}</s>{{ end }}
{{- else if eq $m.Role "tool" }}[TOOL_RESULTS]{{ $m.Content }}[/TOOL_RESULTS]
{{- end }}
{{- end }}"""
PARSER ministral
PARAMETER temperature 0.15
PARAMETER top_k 20
PARAMETER top_p 0.95
'''


def mistral_base_text() -> str:
    p = LAB / "base-modelfiles" / "ministral-3-8b.modelfile"
    return p.read_text(encoding="utf-8") if p.is_file() else SPEC_MODELFILE


def qwen_base_text() -> str:
    return (LAB / "base-modelfiles" / "qwen3.5-9b.modelfile").read_text(encoding="utf-8")


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


# --- the write guard ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dest", [
    f"{M.BUCKET}/runs/{R6}/eval/bar.json", f"{M.BUCKET}/runs/{RUN}/train/x", f"{M.BUCKET}/base/x.tar",
    f"{M.BUCKET}/state/model.json", f"{M.BUCKET}/baselines/x/baseline.json", f"{M.BUCKET}/experiments/other/x",
    f"{M.BUCKET}/experiments/mistralx/x", EXP, EXP + "/", f"{EXP}/../runs/x", f"{EXP}/runs//x", f"{EXP}/./x",
    "gs://another-bucket/experiments/mistral/x", "/tmp/x", "experiments/mistral/x", "",
])
def test_guard_refuses_everything_outside_the_experiment(dest):
    with pytest.raises(M.ConfigError):
        M.guard(dest)


def test_guard_accepts_the_experiment():
    assert M.guard(f"{EXP}/runs/{RUN}/exam/mx-base/summary.json") == f"{EXP}/runs/{RUN}/exam/mx-base/summary.json"
    assert M.guard(f"{EXP}/base/{M.MISTRAL_TAR}/") == f"{EXP}/base/{M.MISTRAL_TAR}"


def test_every_write_goes_through_the_guard(tmp_path):
    gs = M.GS(local=tmp_path)
    gs.put_json({"a": 1}, f"{EXP}/runs/{RUN}/gate.json")
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f.txt").write_text("x")
    for bad in (lambda: gs.put_json({}, f"{M.BUCKET}/runs/{R6}/gate.json"),
                lambda: gs.put_dir(tmp_path / "d", f"{M.BUCKET}/state"),
                lambda: gs.put_file(tmp_path / "d" / "f.txt", f"{M.BUCKET}/base/f.txt"),
                lambda: gs.copy_prefix(f"{M.BUCKET}/runs/{R6}/dataset", f"{M.BUCKET}/runs/{RUN}/dataset")):
        with pytest.raises(M.ConfigError):
            bad()
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file() and "d/f.txt" not in p.as_posix())
    assert written == [f"breadstack-sous-chef-int-pipeline/experiments/mistral/runs/{RUN}/gate.json"]


def test_a_real_gcloud_write_is_guarded_before_gcloud_runs():
    calls = []
    gs = M.GS(local="", runner=lambda argv, **kw: calls.append(argv) or subprocess.CompletedProcess(argv, 0, b"", b""))
    gs.local = None
    with pytest.raises(M.ConfigError):
        gs.put_bytes(b"x", f"{M.BUCKET}/runs/{R6}/x")
    assert calls == []
    gs.put_bytes(b"x", f"{EXP}/runs/{RUN}/x")
    assert calls and calls[0][:4] == ["gcloud", "storage", "cp", "-"] and calls[0][4] == f"{EXP}/runs/{RUN}/x"


def test_model_locations():
    assert M.model_uri("mx-base", RUN) == f"{EXP}/runs/{RUN}/models/mx-base"
    assert M.model_uri("mx", RUN) == f"{EXP}/runs/{RUN}/train/stage"
    assert M.model_uri("r6c", RUN) == f"{M.BUCKET}/runs/{R6}/train/stage"  # read only
    with pytest.raises(M.ConfigError):
        M.model_uri("mx", "rt-20261016-2102-3eeb97-eb908674")  # a pipeline run id is not an MX run


# --- Modelfiles -----------------------------------------------------------------------------------------------------------

def test_modelfile_facts_mistral_template_has_no_trailing_newline():
    f = M.modelfile_facts(mistral_base_text())
    assert f["parser"] == "ministral" and f["renderer"] is None
    assert f["template"].startswith("{{- $tools := false }}") and f["template"].endswith("{{- end }}")
    assert not f["template"].endswith("\n")
    assert M.family_of(f) == "mistral"


def test_modelfile_facts_qwen():
    f = M.modelfile_facts(qwen_base_text())
    assert (f["template"], f["parser"], f["renderer"]) == ("{{ .Prompt }}", "qwen3.5", "qwen3.5")
    assert M.family_of(f) == "qwen"


@pytest.mark.parametrize("which", ["qwen", "mistral"])
def test_staged_modelfile_is_what_make_model_stage_writes(tmp_path, which):
    """base-gguf runs the real `make model-stage`; staged_modelfile is the reference it is compared to."""
    if shutil.which("make") is None:
        pytest.skip("no make")
    from souschef_lab.export import student_modelfile
    base, tag = (qwen_base_text(), "qwen3.5:9b") if which == "qwen" else (mistral_base_text(), "ministral-3:8b")
    exp = tmp_path / "export"
    exp.mkdir()
    gg = f"{which}-base-q4_k_m.gguf"
    G.write(exp / gg, G.qwen_kv())
    (exp / "Modelfile.q4_k_m").write_text(student_modelfile(base, gg, tag, 65536), encoding="utf-8")
    stage = tmp_path / "stage"
    r = subprocess.run(["make", "--no-print-directory", "-C", str(LAB), "model-stage", f"NAME={which}", f"MODEL_EXPORT={exp}",
                        f"MODEL_STAGE={stage}", "MODEL_TEMPERATURE=0.3", "MODEL_PRESENCE_PENALTY=0"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    got = (stage / "Modelfile").read_text(encoding="utf-8")
    assert got == M.staged_modelfile(base, gg, tag)
    f = M.modelfile_facts(got)
    assert f["parameters"]["temperature"] == ["0.3"] and f["parameters"]["presence_penalty"] == ["0"]
    assert f["parameters"]["num_ctx"] == ["65536"] and f["from"] == f"./{gg}"
    assert f["template"] == M.modelfile_facts(base)["template"]  # the TEMPLATE survives byte for byte


def test_native_modelfile_drops_only_the_template_and_parser():
    staged = M.staged_modelfile(mistral_base_text(), "mx-base-q4_k_m.gguf", "ministral-3:8b")
    nat = M.modelfile_facts(M.native_modelfile(staged))
    full = M.modelfile_facts(staged)
    assert nat["template"] is None and nat["parser"] is None and nat["from"] == full["from"]
    assert nat["parameters"] == full["parameters"]


def test_stage_native_copies_the_gguf_and_writes_only_the_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_GS_LOCAL", str(tmp_path))
    gs = M.GS()
    src = f"{EXP}/runs/{RUN}/models/mx-base"
    gs.put_bytes(b"GGUF", f"{src}/mx-base-q4_k_m.gguf")
    gs.put_bytes(M.staged_modelfile(mistral_base_text(), "mx-base-q4_k_m.gguf", "ministral-3:8b").encode(), f"{src}/Modelfile")
    assert M.main(["stage-native", "--run", RUN]) == 0
    dst = gs.path(f"{EXP}/runs/{RUN}/models/{M.NATIVE_ARM}")
    assert (dst / "mx-base-q4_k_m.gguf").read_bytes() == b"GGUF"
    assert M.modelfile_facts((dst / "Modelfile").read_text())["parser"] is None


# --- serving: Ollama + the asserts ------------------------------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def staged_dir(tmp_path: Path, which: str = "mistral") -> Path:
    d = tmp_path / f"model-{which}"
    d.mkdir()
    gg = f"{which}-q4_k_m.gguf"
    G.write(d / gg, G.mistral_kv() if which == "mistral" else G.qwen_kv())
    base, tag = (mistral_base_text(), "ministral-3:8b") if which == "mistral" else (qwen_base_text(), "qwen3.5:9b")
    (d / "Modelfile").write_text(M.staged_modelfile(base, gg, tag), encoding="utf-8")
    return d


@pytest.fixture
def fake_bin(tmp_path):
    b = tmp_path / "bin" / "ollama"
    b.parent.mkdir()
    b.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE_OLLAMA} \"$@\"\n")
    b.chmod(0o755)
    return str(b)


def serve(tmp_path, fake_bin, which, monkeypatch, mode="ok", gpu=None):
    monkeypatch.setenv("FAKE_OLLAMA_MODE", mode)
    d = staged_dir(tmp_path, which)
    ol = M.Ollama(free_port(), tmp_path / f"models-{which}", tmp_path / f"ollama-{which}.log", gpu=gpu, binary=fake_bin)
    ol.start(timeout_s=30)
    return ol, d


@pytest.mark.parametrize("which", ["mistral", "qwen"])
def test_serving_asserts_pass_on_a_correct_model(tmp_path, fake_bin, monkeypatch, which):
    ol, d = serve(tmp_path, fake_bin, which, monkeypatch, gpu="1")
    try:
        ol.create(d)
        facts, gg = M.model_dir_facts(d)
        seen = M.assert_served(ol.url, facts, M.family_of(facts))
        assert seen["show"]["template_sha256"] == M.sha256_text(facts["template"])
        assert seen["ps"]["size_vram"] == seen["ps"]["size"] and seen["messages_smoke"]["content"] == ["tool_use"]
        want = {"mistral": 61, "qwen": 279}[which]
        assert seen["prompt_smoke"] == {"status": 200, "input_tokens": want, "expected": want}
        rec = json.loads((tmp_path / f"models-{which}" / "fake-created.json").read_text())
        assert rec["name"] == "sous-chef" and rec["cuda"] == "1"  # pinned to its GPU, the model is named sous-chef
    finally:
        ol.stop()
    counts = M.log_counts([tmp_path / f"ollama-{which}.log"], None)
    assert counts["ollama_status"]["/v1/messages"] == {"200": 2}  # the prompt-size smoke, then the tool smoke


@pytest.mark.parametrize("mode,reason", [("cpu", "gpu"), ("badtemplate", "show"), ("msg404", "decode"),
                                         ("badprompt", "render"), ("leak", "render"), ("notool", "render")])
def test_serving_asserts_block(tmp_path, fake_bin, monkeypatch, mode, reason):
    ol, d = serve(tmp_path, fake_bin, "mistral", monkeypatch, mode=mode)
    try:
        ol.create(d)
        facts, _ = M.model_dir_facts(d)
        with pytest.raises(M.Blocked) as e:
            M.assert_served(ol.url, facts, "mistral")
        assert e.value.reason == reason
    finally:
        ol.stop()


def test_the_prompt_size_smoke_expects_the_committed_template_only():
    from souschef_lab import serving_mistral as SM
    mx = M.modelfile_facts(mistral_base_text())
    assert mx["template"] == SM.TEMPLATE  # the committed Modelfile's TEMPLATE is the renderer's, byte for byte
    assert M.smoke_prompt_tokens(mx, "mistral") == SM.SMOKE_PROMPT_TOKENS["mistral"] == 61
    assert M.smoke_prompt_tokens(M.modelfile_facts(qwen_base_text()), "qwen") == SM.SMOKE_PROMPT_TOKENS["qwen"] == 279
    # another template (the mx-base-native diagnostic: the GGUF's own) has no expected size: recorded, not checked
    assert M.smoke_prompt_tokens(dict(mx, template=mx["template"] + " "), "mistral") is None
    assert "render" in M.BLOCKING


def test_a_qwen_text_reply_in_the_tool_smoke_is_recorded(tmp_path, fake_bin, monkeypatch):
    """Only Mistral must call echo (its control tokens reaching the parser); Qwen's base keeps today's rule."""
    ol, d = serve(tmp_path, fake_bin, "qwen", monkeypatch, mode="notool")
    try:
        ol.create(d)
        facts, _ = M.model_dir_facts(d)
        seen = M.assert_served(ol.url, facts, "qwen")
        assert seen["messages_smoke"]["content"] == ["text"] and seen["smoke_500s"] == 0
    finally:
        ol.stop()


def test_a_parser_error_in_the_smoke_is_recorded_not_blocking_unless_strict(tmp_path, fake_bin, monkeypatch):
    ol, d = serve(tmp_path, fake_bin, "mistral", monkeypatch, mode="msg500")
    try:
        ol.create(d)
        facts, _ = M.model_dir_facts(d)
        seen = M.assert_served(ol.url, facts, "mistral")
        assert seen["messages_smoke"]["status"] == 500 and seen["smoke_500s"] == 1
        with pytest.raises(M.Blocked):
            M.assert_served(ol.url, facts, "mistral", strict_smoke=True)
    finally:
        ol.stop()


def test_a_wrong_family_or_parser_blocks(tmp_path, fake_bin, monkeypatch):
    ol, d = serve(tmp_path, fake_bin, "mistral", monkeypatch)
    try:
        ol.create(d)
        facts, _ = M.model_dir_facts(d)
        with pytest.raises(M.Blocked) as e:
            M.assert_served(ol.url, facts, "qwen")
        assert e.value.reason == "show" and "architecture" in e.value.detail
        with pytest.raises(M.Blocked) as e:
            M.assert_served(ol.url, dict(facts, parser="qwen3.5"), "mistral")
        assert "parser" in e.value.detail
    finally:
        ol.stop()


def test_create_failure_blocks(tmp_path, fake_bin, monkeypatch):
    ol, d = serve(tmp_path, fake_bin, "mistral", monkeypatch, mode="nocreate")
    try:
        with pytest.raises(M.Blocked) as e:
            ol.create(d)
        assert e.value.reason == "create"
    finally:
        ol.stop()


def test_pull_model_refuses_a_bad_gguf(tmp_path):
    root = tmp_path / "gs"
    src = M.GS(local=root).path(f"{EXP}/runs/{RUN}/models/mx-base")
    src.mkdir(parents=True)
    G.write(src / "mx-base-q4_k_m.gguf", G.mistral_kv(tokenizer__ggml__add_bos_token=(G.BOOL, False)))
    (src / "Modelfile").write_text(M.staged_modelfile(mistral_base_text(), "mx-base-q4_k_m.gguf", "ministral-3:8b"))
    with pytest.raises(M.Blocked) as e:
        M.pull_model(M.GS(local=root), f"{EXP}/runs/{RUN}/models/mx-base", tmp_path / "pulled")
    assert e.value.reason == "gguf" and "bos" in e.value.detail


# --- Stage 1: summaries and the gate ------------------------------------------------------------------------------------

def trow(key, ok=True, passed=1, code=None, written=("k",), lat=1000):
    return {"key": key, "status": "ok" if ok else "error", "pass": passed if ok else 0, "code": code,
            "written": list(written), "latency_ms": lat, "usage": {"output_tokens": 50}}


def test_fmt_kind():
    assert M.fmt_kind(trow("a")) is None
    assert M.fmt_kind(trow("a", ok=False, code="model_unavailable")) == "model_unavailable"
    assert M.fmt_kind(trow("a", passed=0, code="max_tokens", written=())) == "max_tokens"
    assert M.fmt_kind(trow("a", passed=0, code="max_tokens")) is None  # it wrote something: an answer


def test_summarize_exam_counts_flips_and_format_failures():
    reps = [{"real:recorded": [trow("r:1"), trow("r:2", passed=0), trow("r:3", ok=False, code="model_unavailable")]},
            {"real:recorded": [trow("r:1"), trow("r:2"), trow("r:3")]}]
    s = M.summarize_exam(reps, arm="mx-base")
    p = s["pooled"]
    assert p["n"] == 3 and p["pass"] == [1, 3] and p["mean"] == 2.0 and p["flips"] == 2 and p["fmt_rows"] == 1
    assert p["fmt"] == {"model_unavailable": 1} and s["rows"]["r:3"]["fmt"] == ["model_unavailable", None]
    assert s["latency_ms"]["p50"] == 1000.0


def summary(rows: dict[str, tuple[list[int], list]], arm: str) -> dict:
    """{key: ([pass per rep], [fmt per rep])} → a summary.json shaped dict."""
    s = {"arm": arm, "rows": {k: {"set": "paraphrase:recorded", "pass": list(p), "fmt": list(f)} for k, (p, f) in rows.items()}}
    s["pooled"] = {"mean": sum(sum(p) for p, _ in rows.values()) / 2, "n": len(rows), "fmt": {}}
    s["sets"] = {"paraphrase:recorded": {"mean": s["pooled"]["mean"], "n": len(rows)}}
    return s


def pair(n_mx_pass: int, n_qw_pass: int, *, n=252, mx_fmt_rows=0, mx_flips=0, qw_flips=0) -> tuple[dict, dict]:
    """Both replicates equal except `*_flips` rows (pass in rep 1 only); mx's format failures on its first
    `mx_fmt_rows` failing rows (rows qwen passes while n_mx_pass + mx_fmt_rows <= n_qw_pass)."""
    mx, qw = {}, {}
    for i in range(n):
        k = f"paraphrase:recorded:{i}"
        p = int(i < n_mx_pass)
        f = "model_unavailable" if n_mx_pass <= i < n_mx_pass + mx_fmt_rows else None  # its first failing rows
        mx[k] = ([p, p], [f, f])
        q = int(i < n_qw_pass)
        qw[k] = ([q, q], [None, None])
    for i in range(mx_flips):
        k = f"paraphrase:recorded:{i}"
        mx[k] = ([1, 0], mx[k][1])
    for i in range(qw_flips):
        k = f"paraphrase:recorded:{i}"
        qw[k] = ([1, 0], qw[k][1])
    return summary(mx, "mx-base"), summary(qw, "qwen-base")


def test_gate_go_at_the_boundary():
    mx, qw = pair(187, 200)  # Δ −13 = −GO_ROWS
    g = M.gate(mx, qw)
    assert g["verdict"] == "GO" and g["delta"] == -13 and g["sigma_delta"] == 0
    assert M.gate(*pair(186, 200))["verdict"] == "JUDGE"


def test_gate_noise_widens_go():
    mx, qw = pair(180, 200, mx_flips=100, qw_flips=100)  # σ_Δ = 0.5·sqrt(200) ≈ 7.07 → 2σ ≈ 14.1 < 20: not GO by noise alone
    g = M.gate(mx, qw)
    assert g["sigma_delta"] == pytest.approx(7.07, abs=0.01)
    assert g["thresholds"]["go"] == pytest.approx(max(13, 2 * 7.0711), abs=0.01)


def test_gate_stop_needs_a_capability_gap():
    # 60 rows behind, mx's failures are not format failures: STOP
    g = M.gate(*pair(140, 200))
    assert g["verdict"] == "STOP" and g["delta_fmt"] == -60
    # 60 behind, but 50 of mx's 112 failing rows are format failures → on the clean rows the gap is smaller
    g2 = M.gate(*pair(140, 200, mx_fmt_rows=50))
    assert g2["delta"] == -60 and g2["delta_fmt"] > -25 and g2["verdict"] == "JUDGE" and g2["default"] == "GO"


def test_gate_judge_default_stop_when_format_explains_little():
    g = M.gate(*pair(175, 200, mx_fmt_rows=2))  # Δ −25: JUDGE; Δ_fmt ≈ −25 < Δ/2
    assert g["verdict"] == "JUDGE" and g["default"] == "STOP"


def test_gate_blocked_and_missing():
    assert M.gate(None, None, mx_failed={"reason": "load", "detail": "x"})["verdict"] == "BLOCKED"
    assert M.gate(None, pair(1, 1)[1])["verdict"] == "MISSING"


def test_gate_refuses_one_replicate():
    mx, qw = pair(200, 200)
    for r in mx["rows"].values():
        r["pass"], r["fmt"] = r["pass"][:1], r["fmt"][:1]
    g = M.gate(mx, qw)
    assert g["verdict"] == "INVALID" and g["replicates"] == {"mx-base": 1, "qwen-base": 2}


def with_500s(s: dict, n: int, smoke: int = 0) -> dict:
    return dict(s, logs={"ollama_status": {"/v1/messages": {"200": 500, "500": n + smoke}}}, served={"smoke_500s": smoke})


def test_gate_hidden_retries_can_turn_go_into_judge():
    mx, qw = pair(195, 200)  # Δ −5: GO with 8 rows to spare
    assert M.gate(with_500s(mx, 16), with_500s(qw, 0))["verdict"] == "GO"  # 8 per replicate: still ≥ −13
    g = M.gate(with_500s(mx, 20, smoke=1), with_500s(qw, 2))  # (20 − 2) / 2 = 9 rows: −14 worst case
    assert g["verdict"] == "JUDGE" and g["retries_500"] == {"mx-base": 20, "qwen-base": 2, "excess_rows_per_replicate": 9.0}
    assert "retried" in g["why"] and g["default"] in ("GO", "STOP")


def test_gate_stop_waits_for_the_native_diagnostic():
    mx, qw = pair(140, 200)
    g = M.gate(mx, qw)
    assert g["verdict"] == "STOP" and "mx-base-native" in g["confirm"]
    native_go, _ = pair(195, 200)
    g2 = M.gate(mx, qw, native=native_go)
    assert g2["verdict"] == "JUDGE" and g2["native"]["verdict"] == "GO" and g2["default"] == "GO" and "confirm" not in g2
    native_stop, _ = pair(141, 200)
    g3 = M.gate(mx, qw, native=native_stop)
    assert g3["verdict"] == "STOP" and g3["native"]["verdict"] == "STOP" and "confirm" not in g3


def test_gate_r4_anchor():
    mx, qw = pair(200, 200, qw_flips=16)
    r4 = {"pooled": {"mean": 200.0, "n": 252}}
    g = M.gate(mx, qw, r4=r4, r4_ref=208.0)
    assert g["r4_anchor"]["gap"] == -8.0 and g["r4_anchor"]["tol"] == 4.0 and g["r4_anchor"]["flagged"] is True


def test_r4_reference_from_a_bar():
    bar = {"sets": [{"set": "paraphrase", "current": 120.0}, {"set": "paraphrase:recorded", "current": 117.0},
                    {"set": "holdout:recorded", "current": 55.0}, {"set": "real:recorded", "current": 12.0},
                    {"set": "followup:recorded", "current": 9.0}, {"set": "out_of_scope:recorded", "current": 13.0},
                    {"set": "out_of_scope_holdout:recorded", "current": 2.0}, {"set": "F0H:seen", "current": 65.0}]}
    assert M.r4_reference(bar) == 208.0


def test_cmd_gate_writes_only_gate_json(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_GS_LOCAL", str(tmp_path))
    gs = M.GS()
    mx, qw = pair(190, 200)
    gs.put_json(mx, f"{EXP}/runs/{RUN}/exam/mx-base/summary.json")
    gs.put_json(qw, f"{EXP}/runs/{RUN}/exam/qwen-base/summary.json")
    before = tree_digest(tmp_path)
    assert M.main(["gate", "--run", RUN, "--r4-ref-run", ""]) == 0
    assert json.loads(gs.path(f"{EXP}/runs/{RUN}/gate.json").read_text())["verdict"] == "GO"
    assert tree_digest(tmp_path) != before


# --- Stage 3: trust, verdicts, the local store ------------------------------------------------------------------------------

def bar(pass_=True, sets=(("paraphrase", 0.0, 5), ("real", 0.0, 2)), b=10, c=10, p=1.0, failing=()):
    return {"pass": pass_, "pooled": {"b": b, "c": c, "mcnemar_p": p},
            "rules": [{"name": n, "pass": n not in failing, "detail": ""} for n in ("sets", "mcnemar", "oos_where", "f0h_unseen", "trigger", "ghosts")],
            "sets": [{"set": s, "delta": d, "tol": t, "bar": True, "n": 10, "current": 5, "candidate": 5 + d} for s, d, t in sets]}


def test_trust():
    r6 = bar(pass_=False, sets=(("paraphrase", -3.0, 5), ("real", 2.0, 2)), b=32, c=20, failing=("trigger",))
    assert M.trust(bar(pass_=False, sets=(("paraphrase", -1.0, 5), ("real", 1.0, 2)), b=30, c=21, failing=("trigger",)), r6)["trusted"]
    # a differing bar verdict with every set inside its bound: trusted, with a note (noise bound, 2026-10-18)
    tr = M.trust(bar(pass_=True, sets=(("paraphrase", -3.0, 5), ("real", 2.0, 2)), b=32, c=20), r6)
    assert tr["trusted"] and any("within noise" in n for n in tr["notes"])
    assert not M.trust(bar(pass_=False, sets=(("paraphrase", 3.0, 5), ("real", 2.0, 2)), b=32, c=20), r6)["trusted"]  # 6 > tol 5
    assert not M.trust(bar(pass_=False, sets=(("paraphrase", -3.0, 5), ("real", 2.0, 2)), b=20, c=32), r6)["trusted"]  # direction
    assert not M.trust(None, r6)["trusted"]


@pytest.mark.parametrize("b_,expect", [
    (bar(pass_=True, b=10, c=30, p=0.002), "better"),
    (bar(pass_=True, b=10, c=12, p=0.83), "on_par"),
    (bar(pass_=True, b=10, c=20, p=0.1), "inconclusive"),  # the grey band
    (bar(pass_=False, sets=(("paraphrase", -8.0, 5), ("real", 0.0, 2)), b=30, c=20, p=0.21, failing=("sets",)), "inconclusive"),  # short by 3
    (bar(pass_=False, sets=(("paraphrase", -12.0, 5), ("real", 0.0, 2)), b=40, c=20, p=0.01, failing=("sets", "mcnemar")), "worse"),
    (bar(pass_=False, sets=(("paraphrase", -10.0, 5), ("real", 0.0, 2)), b=30, c=20, p=0.25, failing=("sets",)), "worse"),  # short by 5
    (bar(pass_=False, b=30, c=20, p=0.3, failing=("oos_where",)), "worse"),
])
def test_verdicts(b_, expect):
    assert M.verdict(b_, trusted=True)["verdict"] == expect


def test_r6c_mx_is_not_asked_to_beat_r6c_on_the_trigger():
    b_ = bar(pass_=False, b=10, c=12, p=0.83, failing=("trigger",))  # on par otherwise: r6c already learned the fix
    v = M.verdict(b_, trusted=True, counts_trigger=False)
    assert v["verdict"] == "on_par" and v["trigger"] == {"pass": False, "detail": "", "counted": False}
    assert M.verdict(b_, trusted=True)["verdict"] == "worse"  # r4 has not: the trigger counts


def erow(set_, i, ok=True, passed=1):
    return {"key": f"{set_}:{i}", "set": set_, "status": "ok" if ok else "error", "pass": passed if ok else 0}


def arm_rows(n: int, *, passes: int, errors: int = 0, reps: int = 2, set_="paraphrase") -> list[dict]:
    """reps × n rows of one bar set: the first `errors` rows error, the next pass until `passes` pass."""
    out = []
    for _ in range(reps):
        for i in range(n):
            if i < errors:
                out.append(erow(set_, i, ok=False))
            else:
                out.append(erow(set_, i, passed=int(i < errors + passes)))
    return out


def test_error_rows_count_as_fails_in_stage3():
    # bar.json saw parity (errors left out of both arms) but MX errored on 4 rows the current model passes
    b_ = bar(pass_=True, sets=(("paraphrase", 0.0, 2),), b=0, c=0, p=1.0)
    cur = arm_rows(20, passes=14)
    cand = arm_rows(20, passes=10, errors=4)
    acct = M.error_accounting(cur, cand, b_)
    x = acct["sets"][0]
    assert x["errors"] == {"current": 0.0, "candidate": 4.0} and x["delta_errors_as_fails"] == -4.0
    assert x["worse"] and x["errors_over_tol"] and acct["pooled"]["b"] == 8 and acct["pooled"]["c"] == 0
    v = M.verdict(b_, trusted=True, counts_trigger=False, acct=acct)
    assert v["verdict"] == "worse" and "errors" in v["failing"] and "sets" in v["failing"]
    # the same rows without errors: on par
    acct2 = M.error_accounting(cur, arm_rows(20, passes=14), b_)
    assert M.verdict(b_, trusted=True, counts_trigger=False, acct=acct2)["verdict"] == "on_par"


def test_hidden_retries_can_make_a_win_inconclusive():
    b_ = bar(pass_=True, b=10, c=30, p=0.002)
    assert M.verdict(b_, trusted=True, retries={"current": 0, "candidate": 2})["verdict"] == "better"
    v = M.verdict(b_, trusted=True, retries={"current": 1, "candidate": 9})  # b 18 / c 22 worst case
    assert v["verdict"] == "inconclusive" and v["retries_500"]["excess"] == 8


def test_untrusted_control_means_no_verdict():
    assert M.verdict(bar(pass_=True, b=1, c=30, p=0.0001), trusted=False)["verdict"] == "untrusted"


def seed_bucket(root: Path) -> None:
    from souschef_lab import gcsio
    b = gcsio.LocalBucket(root)
    pre = f"runs/{R6}"
    for k, v in {"state/model.json": {"tag": "r4", "digest": M.R4_DIGEST, "training_manifest": M.R4_MANIFEST},
                 "state/deployed.json": {"hash": "m6"}, f"manifests/{M.R4_MANIFEST}.json": {"x": 4}, "manifests/m6.json": {"x": 6},
                 f"baselines/{M.R4_DIGEST}/baseline.json": {"replicates": [{"rows": {}}], "who": "r4"},
                 f"{pre}/run.json": {"trigger": {"drift_run": "dr-1"}}, f"{pre}/prepare/manifest.json": {"hash": "m6"},
                 f"{pre}/model/image.json": {"digest": "sha256:r6c", "tag": f"c-{R6}"},
                 f"{pre}/eval/candidate-baseline.json": {"digest": "sha256:r6c", "training_manifest": "m6", "replicates": []},
                 f"{pre}/eval/bar.json": {"pass": False}, f"{pre}/report.json": {"at": "2026-10-17T20:31:49Z"},
                 "drift/dr-1/report.json": {"impact": {}}}.items():
        b.write_json(k, v)
    b.write_bytes(f"{pre}/prepare/cases.jsonl", b'{"id": "c1"}\n')


@pytest.mark.parametrize("pair_", ["r4-mx", "r6c-mx", "r4-r6c", "r4-mxt"])
def test_mirror_store_reads_the_bucket_and_patches_only_the_copy(tmp_path, pair_):
    from souschef_lab.retrain.common import open_store
    src_root, dst_root = tmp_path / "bucket", tmp_path / "local"
    seed_bucket(src_root)
    before = tree_digest(src_root)
    cur, cand = M.pairs_for(R6)[pair_]
    src, dst = open_store(str(src_root)), open_store(str(dst_root))
    log = M.mirror_store(src, dst, R6, current=cur, candidate=cand, run=RUN)
    assert tree_digest(src_root) == before  # the pipeline bucket is never written
    model = dst.read_json("state/model.json")
    img = dst.read_json(f"runs/{R6}/model/image.json")
    assert dst.read_json(f"runs/{R6}/eval/bar.json") is None  # no stale eval in the copy
    if cur == "r6c":
        assert model["digest"] == "sha256:r6c" and model["training_manifest"] == "m6"
        assert dst.read_json("baselines/sha256:r6c/baseline.json")["digest"] == "sha256:r6c"
    else:
        assert model["digest"] == M.R4_DIGEST and dst.read_json(f"baselines/{M.R4_DIGEST}/baseline.json")["who"] == "r4"
    assert img["digest"] == (f"{cand}:{RUN}" if cand in M.MX_MODELS else "sha256:r6c")
    assert f"runs/{R6}/prepare/cases.jsonl" in log["copied"] and "drift/dr-1/report.json" in log["copied"]


def test_mirror_needs_r6s_eval_for_r6c_as_current(tmp_path):
    from souschef_lab import gcsio
    from souschef_lab.retrain.common import open_store
    seed_bucket(tmp_path / "bucket")
    gcsio.LocalBucket(tmp_path / "bucket").delete(f"runs/{R6}/eval/candidate-baseline.json")
    with pytest.raises(M.ConfigError):
        M.mirror_store(open_store(str(tmp_path / "bucket")), open_store(str(tmp_path / "l")), R6, current="r6c",
                       candidate="mx", run=RUN)


def roll_out_r6(root: Path) -> None:
    """What an r6 approval does to the live state: state/model.json and a new baseline name r6."""
    from souschef_lab import gcsio
    b = gcsio.LocalBucket(root)
    b.write_json("state/model.json", {"tag": "r6", "digest": "sha256:r6c", "training_manifest": "m6"})
    b.write_json("baselines/sha256:r6c/baseline.json", {"who": "r6"})


def test_the_r4_pairs_refuse_a_moved_live_model_without_a_snapshot(tmp_path):
    from souschef_lab.retrain.common import open_store
    seed_bucket(tmp_path / "bucket")
    roll_out_r6(tmp_path / "bucket")
    with pytest.raises(M.ConfigError, match="snapshot-r4"):
        M.mirror_store(open_store(str(tmp_path / "bucket")), open_store(str(tmp_path / "l")), R6, current="r4",
                       candidate="mx", run=RUN)


def test_snapshot_r4_freezes_r4_for_the_r4_pairs(tmp_path, monkeypatch):
    from souschef_lab.retrain.common import open_store
    monkeypatch.setenv("MX_GS_LOCAL", str(tmp_path / "gs"))
    broot = tmp_path / "gs" / M.BUCKET[5:]
    seed_bucket(broot)
    before = tree_digest(broot / "state")
    assert M.main(["snapshot-r4"]) == 0
    assert tree_digest(broot / "state") == before  # read only
    snap = M.GS().read_json(f"{M.BUCKET}/{M.SNAP_REL}/snapshot.json")
    assert snap["digest"] == M.R4_DIGEST and len(snap["files"]) == 4
    assert M.snapshot_r4(M.GS()).get("kept") is True  # idempotent
    roll_out_r6(broot)  # r6 is approved later
    with pytest.raises(M.ConfigError):  # a fresh snapshot now would be of r6: refused
        import shutil as _sh
        _sh.rmtree(broot / M.SNAP_REL)
        M.snapshot_r4(M.GS())
    assert M.main(["snapshot-r4"]) == 2
    # restore the snapshot and mirror: r4's state and baseline come from it, not the live r6
    seed_bucket(tmp_path / "fresh")
    for f in M.SNAP_FILES:
        t = broot / M.SNAP_REL / f
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes((tmp_path / "fresh" / f).read_bytes())
    dst = open_store(str(tmp_path / "l"))
    log = M.mirror_store(open_store(str(broot)), dst, R6, current="r4", candidate="mx", run=RUN)
    assert dst.read_json("state/model.json")["digest"] == M.R4_DIGEST
    assert dst.read_json(f"baselines/{M.R4_DIGEST}/baseline.json")["who"] == "r4"
    assert any(M.SNAP_REL in p for p in log["patched"])


def test_cmd_report_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_GS_LOCAL", str(tmp_path))
    gs = M.GS()
    root = f"{EXP}/runs/{RUN}"
    gs.put_json(M.gate(*pair(195, 200)), f"{root}/gate.json")
    r6 = bar(pass_=False, sets=(("paraphrase", -3.0, 5), ("real", 2.0, 2)), b=32, c=20, p=0.13, failing=("trigger",))
    (tmp_path / M.BUCKET[5:] / "runs" / R6 / "eval").mkdir(parents=True)
    (tmp_path / M.BUCKET[5:] / "runs" / R6 / "eval" / "bar.json").write_text(json.dumps(r6))
    gs.put_json(bar(pass_=False, sets=(("paraphrase", -2.0, 5), ("real", 1.0, 2)), b=31, c=22, failing=("trigger",)),
                f"{root}/eval/r4-r6c/eval/bar.json")
    # r6c-mx through the attempts layout (latest.json), with its rows: on par, no errors
    gs.put_json({"attempt": "mx-7", "rc": 0, "status": "ok"}, f"{root}/eval/r6c-mx/latest.json")
    b6 = bar(pass_=True, sets=(("paraphrase", 0.0, 2),), b=0, c=0, p=1.0)
    gs.put_json(b6, f"{root}/eval/r6c-mx/attempts/mx-7/eval/bar.json")
    for role in ("current", "candidate"):
        gs.put_bytes("".join(json.dumps(r) + "\n" for r in arm_rows(20, passes=14)).encode(),
                     f"{root}/eval/r6c-mx/attempts/mx-7/eval/{role}/rows.jsonl")
    gs.put_json({"retries_500": {"current": 0, "candidate": 0}}, f"{root}/eval/r6c-mx/attempts/mx-7/info.json")
    gs.put_json({"rows": {"train": 2700}, "train_tokens_per_second": 2500, "peak_vram_gb": 24.1}, f"{root}/train/adapter/run.json")
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([{"displayName": "souschef-mx-exam-x", "labels": {"stage": "exam"}, "state": "JOB_STATE_SUCCEEDED",
                                 "startTime": "2026-10-17T00:00:00Z", "endTime": "2026-10-17T02:00:00Z",
                                 "jobSpec": {"workerPoolSpecs": [{"machineSpec": {"machineType": "g2-standard-12"}}],
                                             "scheduling": {"strategy": "STANDARD"}}}]))
    assert M.main(["report", "--run", RUN, "--cases-run", R6, "--jobs-json", str(jobs)]) == 0
    rep = json.loads(gs.path(f"{root}/report/report.json").read_text())
    assert rep["stage3"]["control"]["trusted"] is True and rep["stage3"]["verdicts"]["r6c-mx"]["verdict"] == "on_par"
    assert rep["stage3"]["control_pair"] == "r4-r6c" and rep["round"]["candidate"] == "r6c"
    assert rep["stage3"]["pairs"]["r6c-mx"]["errors"]["errors_total"] == {"current": 0.0, "candidate": 0.0}
    assert rep["cost"]["usd_est_total"] == 2.8
    md = gs.path(f"{root}/report/report.md").read_text()
    assert "ON_PAR" in md and "### r6c-mx" in md and "Control (r4-r6c) trusted" in md and "r7" not in md


# --- job bodies --------------------------------------------------------------------------------------------------------------

TRAINER_MX = f"{M.AR}/sous-chef-trainer-mx@sha256:" + "a" * 64
MX_IMG = f"{M.AR}/sous-chef-mx@sha256:" + "b" * 64


def all_bodies() -> list[dict]:
    out = [M.job_body("base-gguf", run=RUN, image=TRAINER_MX, arm="mx-base,qwen-base", create="mx-1")]
    out += [M.job_body("exam", run=RUN, image=MX_IMG, arm=a, create="mx-1") for a in M.STAGE1_ARMS]
    out += [M.job_body("train", run=RUN, image=TRAINER_MX, rung="a100-40-spot", mode=m, create="mx-1") for m in ("smoke", "full")]
    out += [M.job_body("eval", run=RUN, image=MX_IMG, pair=p, create="mx-1") for p in M.pairs_for(M.DEFAULT_CASES_RUN)]
    return out


def test_every_body_runs_as_souschef_mx_and_is_labelled_mx():
    for b in all_bodies():
        assert b["jobSpec"]["serviceAccount"] == "souschef-mx@breadstack-sous-chef-int.iam.gserviceaccount.com"
        assert b["labels"]["purpose"] == "sous-chef-mx" and b["labels"]["create"] == "mx-1"
        assert "sous-chef-pipeline" not in json.dumps(b) and "souschef-trainer@" not in json.dumps(b)
        assert all(re.fullmatch(r"[a-z0-9_-]{1,63}", v) for v in b["labels"].values()), b["labels"]
        assert len(b["displayName"]) <= 128


def test_runner_bodies_never_retry_and_name_their_attempt():
    for b in all_bodies():
        env = {e["name"]: e["value"] for e in b["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["env"]}
        if b["labels"]["stage"].startswith("train"):
            assert "disableRetries" not in b["jobSpec"]["scheduling"] and "MX_CREATE" not in env  # r6's trainer path
        else:
            assert b["jobSpec"]["scheduling"]["disableRetries"] is True and env["MX_CREATE"] == "mx-1"


def test_a_restarted_runner_exits_without_writing(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_GS_LOCAL", str(tmp_path / "gs"))
    monkeypatch.setenv("MX_CREATE", "mx-5")
    gs = M.GS()
    dest = f"{EXP}/runs/{RUN}/exam/r4"
    gs.put_json({"attempt": "mx-5"}, f"{dest}/attempts/mx-5/started.json")
    before = tree_digest(tmp_path / "gs")
    assert M.main(["exam", "--run", RUN, "--arm", "r4", "--dry", "--workdir", str(tmp_path / "w")]) == 1
    assert tree_digest(tmp_path / "gs") == before
    monkeypatch.setenv("MX_CREATE", "Bad Nonce")
    assert M.main(["exam", "--run", RUN, "--arm", "r4", "--dry", "--workdir", str(tmp_path / "w2")]) == 2


def test_results_follow_latest(tmp_path):
    gs = M.GS(local=tmp_path)
    d = f"{EXP}/runs/{RUN}/exam/mx-base"
    assert M.results_uri(gs, d, "summary.json") == f"{d}/summary.json"
    M.publish_attempt(gs, d, "mx-3", 3)
    assert M.results_uri(gs, d, "failed.json") == f"{d}/attempts/mx-3/failed.json"
    assert gs.read_json(f"{d}/latest.json")["status"] == "blocked"


def test_the_train_body_writes_only_under_the_experiment():
    b = M.job_body("train", run=RUN, image=TRAINER_MX, rung="a100-40-spot", attempt=2, create="mx-2")
    env = {e["name"]: e["value"] for e in b["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["env"]}
    assert env["PIPELINE_BUCKET"] == EXP and env["RUNS_ROOT"] == f"{EXP}/runs" and env["BASE_ROOT"] == f"{EXP}/base"
    assert env["BASE_FAMILY"] == "mistral" and env["BASE_TAR"] == M.MISTRAL_TAR and env["TRAIN_ARMS"] == "base:3407"
    assert env["TRAIN_ATTEMPT"] == "2" and env["TRAIN_NONCE"] == "mx-2" and "INIT_ADAPTER" not in env
    ws = b["jobSpec"]["workerPoolSpecs"][0]
    assert ws["machineSpec"] == {"machineType": "a2-highgpu-1g", "acceleratorType": "NVIDIA_TESLA_A100", "acceleratorCount": 1}
    assert b["jobSpec"]["scheduling"] == {"strategy": "SPOT", "timeout": "43200s"} and ws["containerSpec"]["args"] == [RUN, "full"]


@pytest.mark.parametrize("rung", ["a100-40-std", "h100-spot", "h100-flex"])
def test_rungs_that_compete_with_r6_need_r6_done(rung):
    with pytest.raises(M.ConfigError):
        M.job_body("train", run=RUN, image=TRAINER_MX, rung=rung)
    b = M.job_body("train", run=RUN, image=TRAINER_MX, rung=rung, allow_after_round_train=True)
    if rung == "h100-flex":
        assert b["jobSpec"]["scheduling"]["maxWaitDuration"] == "14400s"


def test_exam_and_eval_shapes():
    ex = {b["labels"]["arm"]: b for b in all_bodies() if b["labels"]["stage"] == "exam"}
    reps = {a: b["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["args"] for a, b in ex.items()}
    assert reps["mx-base"][reps["mx-base"].index("--replicates") + 1] == "2" and reps["r4"][reps["r4"].index("--replicates") + 1] == "1"
    ev = [b for b in all_bodies() if b["labels"]["stage"] == "eval"][0]["jobSpec"]["workerPoolSpecs"][0]["machineSpec"]
    assert ev == {"machineType": "g2-standard-24", "acceleratorType": "NVIDIA_L4", "acceleratorCount": 2}


@pytest.mark.parametrize("image", [f"{M.AR}/sous-chef-trainer@sha256:" + "c" * 64, f"{M.AR}/sous-chef-lab@sha256:" + "c" * 64,
                                   f"{M.AR}/sous-chef-mx:latest", "sous-chef-mx"])
def test_check_image_refuses_pipeline_images_and_tags(image):
    with pytest.raises(M.ConfigError):
        M.check_image(image, "image", placeholders=False)


def test_cmd_jobs_prints_and_never_submits(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("jobs must not run anything"))
    assert M.main(["jobs", "--run", RUN, "--stage", "eval", "--mx-image", MX_IMG, "--create", "mx-9", "--out-dir", str(tmp_path)]) == 0
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == sorted(f"souschef-mx-eval-{RUN}-{p}-1.json" for p in M.pairs_for(M.DEFAULT_CASES_RUN, report_only=False))
    capsys.readouterr()
    assert M.main(["jobs", "--run", RUN, "--stage", "base-gguf", "--trainer-image", TRAINER_MX, "--create", "mx-9"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["args"][-1] == "mx-base,qwen-base"


# --- gates (read only) -------------------------------------------------------------------------------------------------------

def test_gates_stage2(tmp_path):
    gs = M.GS(local=tmp_path)
    r6 = tmp_path / M.BUCKET[5:] / "runs" / R6
    r6.mkdir(parents=True)
    (r6 / "result-dataset.json").write_text(json.dumps({"status": "ok"}))
    gs.put_bytes(b"{}\n", f"{EXP}/runs/{RUN}/dataset/dataset/train.jsonl")  # r4dataset's layout
    gs.put_json({"cases_run": R6}, f"{EXP}/runs/{RUN}/plan.json")
    gs.put_json({"verdict": "GO", "at": "t1"}, f"{EXP}/runs/{RUN}/gate.json")
    seen = []

    def runner(argv, **kw):
        seen.append(argv)
        flt = next(a for a in argv if a.startswith("--filter="))
        jobs = []
        if "souschef-train-" in flt and "us-central1" in " ".join(argv):
            jobs = [{"displayName": f"souschef-train-{R6}-h100-flex-1", "state": "JOB_STATE_RUNNING"}]
        return subprocess.CompletedProcess(argv, 0, json.dumps(jobs).encode(), b"")
    rows = M.gates("2", run=RUN, cases_run=R6, gs=gs, runner=runner, rung="a100-40-spot")
    assert all(r["ok"] for r in rows), rows
    assert all(a[:4] == ["gcloud", "ai", "custom-jobs", "list"] for a in seen)  # read-only calls only
    assert not all(r["ok"] for r in M.gates("2", run=RUN, cases_run=R6, gs=gs, runner=runner))  # no --rung
    # H100 / on-demand A100 only after r6's train/DONE
    assert not all(r["ok"] for r in M.gates("2", run=RUN, cases_run=R6, gs=gs, runner=runner, rung="h100-spot"))
    (r6 / "train").mkdir()
    (r6 / "train" / "DONE").write_text("")
    assert all(r["ok"] for r in M.gates("2", run=RUN, cases_run=R6, gs=gs, runner=runner, rung="h100-spot"))
    # a JUDGE needs the user's GO for THIS gate.json
    gs.put_json({"verdict": "JUDGE", "at": "t2", "default": "GO"}, f"{EXP}/runs/{RUN}/gate.json")
    assert not all(r["ok"] for r in M.gates("2", run=RUN, cases_run=R6, gs=gs, runner=runner, rung="a100-40-spot"))
    gs.put_json({"decision": "GO", "gate_at": "t1"}, f"{EXP}/runs/{RUN}/user-go.json")  # an older gate's
    assert not all(r["ok"] for r in M.gates("2", run=RUN, cases_run=R6, gs=gs, runner=runner, rung="a100-40-spot"))
    gs.put_json({"decision": "GO", "gate_at": "t2"}, f"{EXP}/runs/{RUN}/user-go.json")
    assert all(r["ok"] for r in M.gates("2", run=RUN, cases_run=R6, gs=gs, runner=runner, rung="a100-40-spot"))


def test_user_go_only_answers_a_judge(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_GS_LOCAL", str(tmp_path))
    gs = M.GS()
    gs.put_json({"verdict": "STOP", "at": "t1"}, f"{EXP}/runs/{RUN}/gate.json")
    assert M.main(["user-go", "--run", RUN, "--by", "the user", "--note", "x"]) == 2
    gs.put_json({"verdict": "JUDGE", "at": "t3"}, f"{EXP}/runs/{RUN}/gate.json")
    assert M.main(["user-go", "--run", RUN, "--by", "the user", "--note", "format failures explain it"]) == 0
    assert gs.read_json(f"{EXP}/runs/{RUN}/user-go.json")["gate_at"] == "t3"

    def busy(argv, **kw):
        flt = next(a for a in argv if a.startswith("--filter="))
        jobs = [{"displayName": "souschef-mx-train-x", "state": "JOB_STATE_RUNNING", "labels": {"stage": "train-smoke"}}] \
            if "sous-chef-mx" in flt and "us-central1" in " ".join(argv) else []
        return subprocess.CompletedProcess(argv, 0, json.dumps(jobs).encode(), b"")
    assert not all(r["ok"] for r in M.gates("2", run=RUN, cases_run=R6, gs=gs, runner=busy, rung="a100-40-spot"))


# --- base-gguf with a fake converter and quantizer --------------------------------------------------------------------------

def fake_llama(tmp_path: Path, family: str) -> Path:
    d = tmp_path / "llama.cpp"
    d.mkdir()
    kv = "G.qwen_kv(file_type=1)" if family == "qwen" else "G.mistral_kv(general__file_type=(G.U32, 1))"
    kv15 = "G.qwen_kv()" if family == "qwen" else "G.mistral_kv()"
    (d / "convert_hf_to_gguf.py").write_text(
        f"import sys; sys.path.insert(0, {str(HERE)!r}); import gguf_synth as G\n"
        f"a = sys.argv; G.write(a[a.index('--outfile') + 1], {kv})\n")
    q = d / "llama-quantize"
    q.write_text(f"#!{sys.executable}\nimport sys; sys.path.insert(0, {str(HERE)!r}); import gguf_synth as G\n"
                 f"assert sys.argv[3] == 'Q4_K_M'; G.write(sys.argv[2], {kv15})\n")
    q.chmod(0o755)
    return d


def base_tar(tmp_path: Path, repo: str, rev: str) -> Path:
    src = tmp_path / "hfsrc"
    root = src / ("models--" + repo.replace("/", "--"))
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text(rev)
    snap = root / "snapshots" / rev
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 16)
    t = tmp_path / "base.tar"
    with tarfile.open(t, "w") as tf:
        tf.add(root, arcname=root.name)
    return t


@pytest.mark.skipif(shutil.which("make") is None, reason="no make")
def test_base_gguf_end_to_end_for_qwen(tmp_path, monkeypatch):
    gsroot = tmp_path / "gs"
    monkeypatch.setenv("MX_GS_LOCAL", str(gsroot))
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(fake_llama(tmp_path, "qwen")))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    t = base_tar(tmp_path, M.QWEN_REPO, "abc123")
    dst = gsroot / M.BUCKET[5:] / "base"
    dst.mkdir(parents=True)
    shutil.copyfile(t, dst / M.QWEN_TAR)
    (dst / f"{M.QWEN_TAR}.sha256").write_text(f"{hashlib.sha256(t.read_bytes()).hexdigest()}  {M.QWEN_TAR}\n")
    before = tree_digest(gsroot / M.BUCKET[5:] / "base")
    assert M.main(["base-gguf", "--run", RUN, "--arms", "qwen-base", "--workdir", str(tmp_path / "w")]) == 0
    out = M.GS().path(f"{EXP}/runs/{RUN}/models/qwen-base")
    assert sorted(p.name for p in out.iterdir()) == ["Modelfile", "base-gguf.json", "gguf-check.json", "qwen-base-q4_k_m.gguf"]
    assert json.loads((out / "gguf-check.json").read_text())["ok"] is True
    assert (out / "Modelfile").read_text() == M.staged_modelfile(qwen_base_text(), "qwen-base-q4_k_m.gguf", "qwen3.5:9b")
    assert tree_digest(gsroot / M.BUCKET[5:] / "base") == before
    written = {p.relative_to(gsroot).as_posix() for p in gsroot.rglob("*") if p.is_file()} - \
              {f"{M.BUCKET[5:]}/base/{M.QWEN_TAR}", f"{M.BUCKET[5:]}/base/{M.QWEN_TAR}.sha256"}
    assert all(w.startswith("breadstack-sous-chef-int-pipeline/experiments/mistral/") for w in written)


@pytest.mark.skipif(shutil.which("make") is None, reason="no make")
def test_base_gguf_refuses_a_failed_check_and_records_why(tmp_path, monkeypatch):
    gsroot = tmp_path / "gs"
    monkeypatch.setenv("MX_GS_LOCAL", str(gsroot))
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(fake_llama(tmp_path, "mistral")))  # mistral bytes for the qwen arm
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    t = base_tar(tmp_path, M.QWEN_REPO, "abc123")
    dst = gsroot / M.BUCKET[5:] / "base"
    dst.mkdir(parents=True)
    shutil.copyfile(t, dst / M.QWEN_TAR)
    (dst / f"{M.QWEN_TAR}.sha256").write_text(f"{hashlib.sha256(t.read_bytes()).hexdigest()}  {M.QWEN_TAR}\n")
    assert M.main(["base-gguf", "--run", RUN, "--arms", "qwen-base", "--workdir", str(tmp_path / "w")]) == 3
    out = M.GS().path(f"{EXP}/runs/{RUN}/models/qwen-base")
    assert not (out / "Modelfile").exists()  # never staged
    assert json.loads((out / "base-gguf.json").read_text())["failed"]["reason"] == "gguf"


def test_untar_refuses_an_escaping_member(tmp_path):
    t = tmp_path / "evil.tar"
    with tarfile.open(t, "w") as tf:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        import io
        tf.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(M.ConfigError):
        M.untar_base(t, tmp_path / "hf")


# --- the exam on the lab's fake service (needs node + the engine snapshot) ---------------------------------------------------

ENGINE_DIR = os.environ.get("MX_ENGINE_DIR") or os.environ.get("COOKIE_CUTTER_DIR")


@pytest.mark.skipif(not (ENGINE_DIR and shutil.which("node")), reason="needs node and MX_ENGINE_DIR / COOKIE_CUTTER_DIR")
def test_exam_dry_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_GS_LOCAL", str(tmp_path / "gs"))
    monkeypatch.setenv("MX_ENGINE_DIR", ENGINE_DIR)
    for k in ("ENGINE", "SOUS_CHEF_ENGINE", "SOUSCHEF_SCORE_POLICY", "SOUSCHEF_FONT_POLICY"):
        monkeypatch.delenv(k, raising=False)
    rc = M.main(["exam", "--run", RUN, "--arm", "mx-base", "--dry", "--sets", "real,out_of_scope", "--limit", "3",
                 "--replicates", "2", "--workdir", str(tmp_path / "w")])
    d = f"{EXP}/runs/{RUN}/exam/mx-base"
    latest = M.GS().read_json(f"{d}/latest.json")
    out = M.GS().path(f"{d}/attempts/{latest['attempt']}")
    assert rc == 0, (out / "failed.json").read_text() if (out / "failed.json").exists() else rc
    assert latest["status"] == "ok" and latest["attempt"].startswith("local-")
    s = json.loads((out / "summary.json").read_text())
    assert s["pooled"]["n"] == 6 and s["replicates"] == 2 and len(s["pooled"]["pass"]) == 2
    assert set(s["sets"]) == {"real:recorded", "out_of_scope:recorded"}
    assert s["pooled"]["mean"] >= 4  # the oracle script answers with the expected writes
    assert {p.name for p in out.iterdir()} >= {"summary.json", "served.json", "real-k1.jsonl", "out_of_scope-k2.jsonl",
                                               "started.json"}
    assert M.read_exam(M.GS(), f"{EXP}/runs/{RUN}", "mx-base", "summary.json")["pooled"]["n"] == 6
    written = [p.relative_to(tmp_path / "gs").as_posix() for p in (tmp_path / "gs").rglob("*") if p.is_file()]
    assert written and all(w.startswith("breadstack-sous-chef-int-pipeline/experiments/mistral/") for w in written)


# --- the image files ---------------------------------------------------------------------------------------------------------

def test_the_build_files_push_only_the_mx_names():
    for f in ("images/mx/cloudbuild.yaml", "images/mx/cloudbuild-trainer.yaml"):
        text = "\n".join(ln for ln in (REPO / f).read_text(encoding="utf-8").splitlines() if not ln.strip().startswith("#"))
        assert not re.search(r"sous-chef-trainer:", text), f  # never a tag of the pipeline's trainer
        assert "tags add" not in text and ":latest" not in text, f
        pushes = re.findall(r'docker push "\$\$REPO:\$[{$]?[A-Z_]+', text)
        assert pushes, f
        repo = re.search(r'REPO="\$_REGION-docker\.pkg\.dev/\$_PROJECT_ID/\$_AR_REPO/([a-z-]+)"', text).group(1)
        assert repo in ("sous-chef-mx", "sous-chef-trainer-mx"), f
    tr = (REPO / "images/mx/cloudbuild-trainer.yaml").read_text(encoding="utf-8")
    assert "-f images/trainer/Dockerfile" in tr and "sous-chef-trainer@$_CACHE_DIGEST" in tr


def test_the_mx_dockerfile_pins_its_bases_by_digest():
    text = (REPO / "images/mx/Dockerfile").read_text(encoding="utf-8")
    assert "sous-chef-lab:28491da@sha256:aa450fa3b804b242801754d09187540f7289c0c0056d7a45a555ab141790b4c9" in text
    model_image = (LAB / "model-image" / "Dockerfile").read_text(encoding="utf-8")
    ollama = re.search(r"ollama/ollama:0\.34\.4@sha256:[0-9a-f]{64}", model_image).group(0)
    assert ollama in text  # the model image's exact Ollama
    assert 'ENTRYPOINT ["/opt/lab/.venv/bin/python", "/opt/lab/tools/mxarm.py"]' in text


def test_the_existing_images_are_untouched():
    r = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "images/trainer", "images/lab-runtime",
                        "lab/model-image", "cloudbuild.yaml"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "", r.stdout


def test_the_fallback_converter_reaches_base_gguf_args():
    b = M.job_body("base-gguf", run=RUN, image=TRAINER_MX, arm="mx-base", create="mx-1", converter_ref="b11081")
    assert b["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["args"][-2:] == ["--converter-ref", "b11081"]
    with pytest.raises(M.ConfigError):
        M.fetch_converter("main; rm -rf /", Path("/nonexistent"))


def test_served_parser_renderer_reads_the_real_0_34_4_show():
    # production r4's /api/show (Ollama 0.34.4, 2026-10-17): no top-level parser / renderer, only modelfile lines;
    # reading the top level blocked every exam with "parser None"
    import json
    from pathlib import Path
    show = json.loads((Path(__file__).parent / "ollama-0.34.4-show-r4.json").read_text())
    assert "parser" not in show and "renderer" not in show
    assert M.served_parser_renderer(show) == ("qwen3.5", "qwen3.5")
    assert M.served_parser_renderer({"modelfile": "FROM x\nPARSER ministral\n"}) == ("ministral", None)
    assert M.served_parser_renderer({"parser": "ministral", "modelfile": ""}) == ("ministral", None)
