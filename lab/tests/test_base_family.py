"""The retrain pipeline's base family (docs/RETRAIN.md "The base family"): qwen (blueberry-muffin) stays today's
pipeline byte for byte; mistral (chocolate-muffin, Ministral 3 8B) is chosen by infra config `base_family` or a round's
trigger.base_family, pinned by the dataset stage in train-config.json, and carried by the trainer (its staged
Modelfile: the Ministral TEMPLATE, PARSER ministral, temperature 0.15, r7's limits), the eval (each arm's route with
its family's settings, each GPU service's /api/show checked against the record), the report (a family change
warns) and the decision (state/model.json `family`), which drift reads for the live model."""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from souschef_lab import chatfmt, harness as H, pipeline_common as PC, serving_mistral, train
from souschef_lab.export import student_modelfile
from souschef_lab.retrain import dataset_stage as DS
from souschef_lab.retrain import decide as DEC
from souschef_lab.retrain import eval_stage as EV
from souschef_lab.retrain import report as REP
from souschef_lab.retrain.common import (ConfigError, model_family, open_store, round_base_family, trained_family)

LAB = Path(__file__).resolve().parents[1]

# runs/rt-20261018-0213-b02e95-3914e2e4/train/stage/Modelfile: r7, the live blueberry-muffin, as its image serves it
R7_STAGED = """# The Sous Chef lab's student: LoRA-tuned from qwen3.5:9b's base model.
# TEMPLATE, RENDERER, PARSER and PARAMETER lines are copied from `ollama show qwen3.5:9b --modelfile`.
FROM ./rt-20261018-0213-b02e95-3914e2e4-q4_k_m.gguf
TEMPLATE {{ .Prompt }}
RENDERER qwen3.5
PARSER qwen3.5
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER temperature 0.3
PARAMETER presence_penalty 0
PARAMETER num_ctx 65536
"""
R7_STAGED_SHA256 = "12e20cd676c67e334adf8a9776757b0f718f685a64caf877ffeac29497e5cd3a"
# experiments/mistral/runs/mx-20261018-m3-8b-r7/train/export/Modelfile.q4_k_m (chocolate-muffin r7's export)
MX = "mx-20261018-m3-8b-r7"
MX_EXPORT_SHA256 = "a44f24c46cad3447dee3ab9a9bce81ea3c761042c96f579d4c020341621b6df2"
QWEN_LORA = {"peft_type": "LORA", "r": 16, "lora_alpha": 16, "base_model_name_or_path": "unsloth/Qwen3.5-9B"}
MISTRAL_LORA = {**QWEN_LORA, "base_model_name_or_path": chatfmt.MISTRAL_BASE}


def mx_export_modelfile() -> str:
    """chocolate-muffin r7's Modelfile.q4_k_m, rebuilt the way export.py wrote it (the base Modelfile's lines)."""
    base = (LAB / "base-modelfiles" / "ministral-3-8b.modelfile").read_text(encoding="utf-8")
    return student_modelfile(base, f"{MX}-q4_k_m.gguf", "ministral-3:8b", 65536)


# --- chatfmt: the families' served models ------------------------------------------------------------------------------------

def test_the_family_settings_are_checked_and_resolved():
    assert [chatfmt.check(v) for v in (None, "", "  ", "qwen", "mistral", " mistral ")] == \
        ["qwen", "qwen", "qwen", "qwen", "mistral", "mistral"]
    for bad in ("llama", "Mistral", 1, ["mistral"]):
        with pytest.raises(ValueError):
            chatfmt.check(bad)
    assert chatfmt.resolve(None, "", "mistral", "qwen") == "mistral" and chatfmt.resolve() == "qwen"
    assert chatfmt.resolve("qwen", "mistral") == "qwen"
    assert (chatfmt.of_architecture("mistral3"), chatfmt.of_architecture("qwen35"), chatfmt.of_architecture("llama")) == \
        ("mistral", "qwen", None)
    assert (chatfmt.stage_temperature("mistral"), chatfmt.stage_temperature(None)) == ("0.15", "0.3")
    assert chatfmt.get("mistral").warm_start and chatfmt.get("qwen").warm_start


def test_r7s_staged_modelfile_is_the_qwen_familys_served_model():
    """The qwen facts are r7's own staged Modelfile (the live model): nothing about a qwen stage changes."""
    assert hashlib.sha256(R7_STAGED.encode()).hexdigest() == R7_STAGED_SHA256
    assert chatfmt.stage_problems(R7_STAGED, "qwen") == [] and chatfmt.of_modelfile(R7_STAGED) == "qwen"
    assert chatfmt.stage_problems(R7_STAGED, "mistral")  # another family's model


def test_the_mistral_stage_shape_and_what_breaks_it():
    exp = mx_export_modelfile()
    assert hashlib.sha256(exp.encode()).hexdigest() == MX_EXPORT_SHA256  # byte for byte the real export's
    assert chatfmt.of_modelfile(exp) == "mistral"
    staged = "".join(ln + "\n" for ln in exp.splitlines() if not ln.startswith(("PARAMETER temperature", "PARAMETER num_ctx"))) \
        + "PARAMETER temperature 0.15\nPARAMETER presence_penalty 0\nPARAMETER num_ctx 65536\n"
    assert chatfmt.stage_problems(staged, "mistral") == []
    assert chatfmt.stage_problems(staged.replace("0.15", "0.3"), "mistral") == ["PARAMETER temperature ['0.3'], want ['0.15']"]
    newline = staged.replace('{{- end }}"""', '{{- end }}\n"""')  # a newline before the closing quotes ends every prompt
    assert any(p.startswith("TEMPLATE is not") for p in chatfmt.stage_problems(newline, "mistral"))
    assert any("PARSER" in p for p in chatfmt.stage_problems(staged.replace("PARSER ministral\n", ""), "mistral"))
    twice = staged + "PARAMETER num_ctx 65536\n"
    assert chatfmt.stage_problems(twice, "mistral") == ["PARAMETER num_ctx ['65536', '65536'], want ['65536']"]


@pytest.mark.skipif(shutil.which("make") is None or shutil.which("sh") is None, reason="needs make and sh")
def test_the_one_off_chocolate_muffin_r7_serving_stage(tmp_path):
    """docs/RETRAIN.md "The base family", the serving image: the r7 MX export's GGUF + Modelfile.q4_k_m staged by
    the real `make model-stage MODEL_TEMPERATURE=0.15 MODEL_PRESENCE_PENALTY=0` is the mistral served model, and the
    model-image context check (lab/model-image/model-context.sh, unchanged: its pinned hash holds) takes it."""
    export = tmp_path / "export"
    export.mkdir()
    (export / "Modelfile.q4_k_m").write_text(mx_export_modelfile(), encoding="utf-8")
    (export / f"{MX}-q4_k_m.gguf").write_bytes(b"GGUF" + b"\0" * 64)
    stage = tmp_path / "stage"
    make = ["make", "--no-print-directory", "-C", str(LAB), "model-stage", f"MODEL_EXPORT={export}",
            f"MODEL_STAGE={stage}", "MODEL_PRESENCE_PENALTY=0"]
    r = subprocess.run([*make, "MODEL_TEMPERATURE=0.15"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(p.name for p in stage.iterdir()) == ["Modelfile", f"{MX}-q4_k_m.gguf"]
    mf = (stage / "Modelfile").read_text()
    assert chatfmt.stage_problems(mf, "mistral") == [] and serving_mistral.template_of(mf) == serving_mistral.TEMPLATE
    env = {**os.environ, "PYTHONPATH": str(LAB)}
    ok = subprocess.run([sys.executable, "-m", "souschef_lab.chatfmt", "check-stage", str(stage), "--family", "mistral"],
                        capture_output=True, text=True, env=env, timeout=60)
    assert ok.returncode == 0 and "is the mistral family's served model" in ok.stdout
    ctx = subprocess.run(["sh", str(LAB / "model-image" / "model-context.sh"), str(stage)], capture_output=True, text=True)
    assert ctx.returncode == 0 and ctx.stdout.strip() == str(stage.resolve())
    # the trainer's 0.3 (the arm's own stage) is not the served model
    r = subprocess.run([*make, "MODEL_TEMPERATURE=0.3"], capture_output=True, text=True, timeout=60)
    bad = subprocess.run([sys.executable, "-m", "souschef_lab.chatfmt", "check-stage", str(stage), "--family", "mistral"],
                         capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and bad.returncode == 1 and "temperature" in bad.stderr


# --- train: the trainer's settings -------------------------------------------------------------------------------------

def test_the_trainers_family_setting(monkeypatch, capsys, tmp_path):
    q = train.pipeline_config({}, None, "bkt")
    assert "family" not in q and "family" not in q["source"]  # a qwen round's settings are today's
    assert train.pipeline_config({}, {"family": "qwen"}, "bkt") == q
    m = train.pipeline_config({}, {"family": "mistral", "seeds": "3407,2"}, "bkt")
    assert m["family"] == "mistral" and m["source"]["family"] == "train-config.json" and m["seeds"] == [3407, 2]
    assert train.pipeline_config({"BASE_FAMILY": "qwen"}, {"family": "mistral"}, "bkt") == q  # the job's env wins
    e = train.pipeline_config({"BASE_FAMILY": "mistral"}, None, "bkt")
    assert e["family"] == "mistral" and e["source"]["family"] == "env BASE_FAMILY"
    with pytest.raises(ValueError, match="train-config.json family"):
        train.pipeline_config({}, {"family": "llama"}, "bkt")
    with pytest.raises(ValueError, match="BASE_FAMILY"):
        train.pipeline_config({"BASE_FAMILY": "llama"}, None, "bkt")
    for k in (*train.PIPELINE_ENV, "BASE_FAMILY"):
        monkeypatch.delenv(k, raising=False)
    cfg = tmp_path / "train-config.json"
    cfg.write_text(json.dumps({"family": "mistral"}))
    assert train.pipeline_config_main([str(cfg), "gs://bkt"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "FAMILY=mistral"
    assert train.pipeline_config_main(["gs://bkt"]) == 0
    assert not any(ln.startswith("FAMILY=") for ln in capsys.readouterr().out.splitlines())


def test_the_window_estimate_runs_at_the_familys_rate():
    """r7's arm: 994 updates; a Ministral arm is about 5.8 H100-hours (19.4 s an update measured), so a second arm no
    longer fits the 12 h window; qwen's figure is today's."""
    assert train.window_estimate(2, 994) == train.window_estimate(2, 994, family="qwen")
    w = train.window_estimate(2, 994, family="mistral")
    assert (w["fits"], w["fit_arms"], w["arm_h"]) == (False, 1, 5.77)
    assert train.window_estimate(1, 994, family="mistral")["fits"]


# --- the dataset stage: resolution, the warm adapter, the pin --------------------------------------------------------

def test_the_rounds_family_is_the_trigger_else_infra_config_else_qwen(tmp_path):
    s = open_store(tmp_path / "bucket")
    assert round_base_family(s, {}) == ("qwen", "default")
    s.write_json("infra/config.json", {"bucket": "b", "base_family": "mistral"})
    assert round_base_family(s, {"reason": "drift"}) == ("mistral", "infra/config.json base_family")
    assert round_base_family(s, {"base_family": "qwen"}) == ("qwen", "trigger.base_family")
    s.write_json("infra/config.json", {"bucket": "b", "base_family": "llama"})
    with pytest.raises(ConfigError, match="infra/config.json base_family"):
        round_base_family(s, None)
    with pytest.raises(ConfigError, match="trigger.base_family"):
        round_base_family(s, {"base_family": "gpt"})
    assert model_family({"tag": "r4"}) == "qwen" and model_family({"tag": "cm-r7", "family": "mistral"}) == "mistral"
    with pytest.raises(ConfigError):
        model_family({"tag": "x", "family": "llama"})
    assert trained_family(s, "rt-1") == "qwen"
    s.write_json("runs/rt-1/train-config.json", {"family": "mistral"})
    assert trained_family(s, "rt-1") == "mistral"


def test_a_warm_adapter_must_be_a_lora_on_the_familys_base(tmp_path):
    s = open_store(tmp_path / "bucket")
    s.write_json("base/adapters/cm-r7/adapter/adapter_config.json", MISTRAL_LORA)
    s.write_json("base/adapters/r4-sft/adapter/adapter_config.json", QWEN_LORA)
    cm = {"trigger": {"train": {"arms": ["base:3407", "warm:3407"], "init_adapter": "base/adapters/cm-r7"}}}
    r4 = {"trigger": {"train": {"arms": ["base:3407", "warm:3407"], "init_adapter": "base/adapters/r4-sft"}}}
    assert DS.train_config_of(cm, s, "mistral") == cm["trigger"]["train"]
    assert DS.train_config_of(r4, s) == r4["trigger"]["train"] == DS.train_config_of(r4, s, "qwen")
    with pytest.raises(ConfigError, match=f"r4-sft was trained on unsloth/Qwen3.5-9B, the recipe's base is "
                                          f"{chatfmt.MISTRAL_BASE} \\(base family mistral\\)"):
        DS.train_config_of(r4, s, "mistral")
    with pytest.raises(ConfigError, match=f"cm-r7 was trained on {chatfmt.MISTRAL_BASE}, the recipe's base is "
                                          "unsloth/Qwen3.5-9B$"):
        DS.train_config_of(cm, s)
    with pytest.raises(ConfigError, match="unknown key"):  # the family is trigger.base_family, never trigger.train's
        DS.train_config_of({"trigger": {"train": {"family": "mistral"}}}, s, "mistral")


def test_the_dataset_plan_checks_every_row_under_the_familys_render(tmp_path):
    from tests.test_dataset_stage import row, write
    d = tmp_path / "dataset"
    write(d, [row(i) for i in range(4)], [row(9)])
    q = DS.train_plan(d, "rt-x-sft")
    m = DS.train_plan(d, "rt-x-sft", "mistral")
    assert "family" not in q and "family_report" not in q
    assert m["family"] == "mistral" and m["updates"] == q["updates"] and m["problems"] == q["problems"]
    assert any("render one segment per collated assistant message" in ln for ln in m["family_report"])
    # an assistant turn holding a control-token string as text that keeps its token spans: warned only
    write(d, [row(i) for i in range(3)] + [row(5, text="Use [TOOL_CALLS] here.")], [row(9)])
    p = DS.train_plan(d, "rt-x-sft", "mistral")
    assert any("control-token string" in ln for ln in p["family_report"])
    assert p["problems"] == DS.train_plan(d, "rt-x-sft")["problems"]  # nothing more than a qwen plan of the same rows
    # one that changes them (a merchant's HTML strike-through <s>…</s>: BOS and an early </s> to the tokenizer) would
    # stop the training mask on the GPU: the dataset gate stops the round here, on CPU, naming the rows
    # … and a user turn holding "[/INST]": the counts hold, but the mask would train the user's words after it
    # (mistral_mask checks counts; mistral_guard only the first row)
    user_inst = row(7)
    user_inst["messages"][1]["content"] = "Type [/INST] in the box."
    for bad in (row(6, text="Struck: <s>old price</s> new price."), user_inst):
        write(d, [row(i) for i in range(3)] + [bad], [row(9)])
        probs = DS.train_plan(d, "rt-x-sft", "mistral")["problems"]
        assert any(x.startswith("mistral render: 1 rows hold a control-token string") and bad["id"] in x for x in probs), probs
        qp = DS.train_plan(d, "rt-x-sft")["problems"]  # a qwen round of the same rows: not this problem
        assert not any("control-token" in x for x in qp) and [x for x in probs if not x.startswith("mistral render")] == qp
    mistral = chatfmt.get("mistral")
    assert train.mask_problem(row(1), mistral) is None
    assert train.mask_problem(row(5, text="Write [INST] then [/INST] for it."), mistral) is None  # inside a reply: kept
    assert "2 </s>" in train.mask_problem(row(6, text="a </s> b"), mistral)
    assert train.mask_problem(user_inst, mistral).startswith("token span 0 covers characters")


# --- the eval: each arm's family ----------------------------------------------------------------------------------------

def _eval_store(tmp_path, *, live_family=None, round_family=None, staged=None):
    from tests.test_eval_stage import seeded
    from tests.test_rebase import drifted
    s = seeded(tmp_path, drifted(rename=False))
    if live_family:
        m = s.read_json("state/model.json")
        s.write_json("state/model.json", {**m, "family": live_family})
    if round_family:
        s.write_json(f"runs/{EV_RUN}/train-config.json", {"family": round_family})
    if staged is not None:
        s.write_bytes(f"runs/{EV_RUN}/train/stage/Modelfile", staged.encode())
    return s


EV_RUN = "rt-20261012-0310-ccd1afab"


def test_the_arms_families_come_from_the_records_and_the_staged_modelfile(tmp_path):
    s = _eval_store(tmp_path, round_family="mistral", staged=mx_export_modelfile())
    assert EV.arm_families(s, EV_RUN, s.read_json("state/model.json")) == {"current": "qwen", "candidate": "mistral"}
    s.write_bytes(f"runs/{EV_RUN}/train/stage/Modelfile", R7_STAGED.encode())
    with pytest.raises(ConfigError, match="staged Modelfile serves the qwen family, the round trained mistral"):
        EV.arm_families(s, EV_RUN, s.read_json("state/model.json"))
    s.delete(f"runs/{EV_RUN}/train-config.json")
    assert EV.arm_families(s, EV_RUN, {"tag": "cm-r7", "family": "mistral"}) == {"current": "mistral", "candidate": "qwen"}


def test_each_gpu_service_must_serve_the_recorded_family():
    fams = {"current": "qwen", "candidate": "mistral"}
    seen = {"https://eval": {"family": "qwen", "architecture": "qwen35"},
            "https://cand": {"family": "mistral", "architecture": "mistral3"}}
    got = EV.check_served(fams, {"current": "https://eval", "candidate": "https://cand"}, seen.get, lambda m: None)
    assert got["candidate"]["architecture"] == "mistral3"
    notes = []
    EV.check_served(fams, {"current": "https://eval", "candidate": "https://x"}, lambda u: seen.get(u) or {"family": None,
                    "error": "/api/show 404"}, notes.append)
    assert notes and "unknown" in notes[0]
    with pytest.raises(ConfigError, match="current serves a mistral model .* but the record says qwen"):
        EV.check_served(fams, {"current": "https://cand", "candidate": "https://cand"}, seen.get, lambda m: None)


class _Recorder:
    """A harness stand-in: records what eval_stage builds it with; __enter__ stops the run (or not)."""
    made: list = []

    def __init__(self, routes, **kw):
        self.routes, self.kw, self.exited = routes, kw, False
        _Recorder.made.append(self)

    def __enter__(self):
        if self.kw.get("families") is None and getattr(_Recorder, "stop_without_families", True):
            raise RuntimeError("stop here")
        return self

    def __exit__(self, *a):
        self.exited = True


def test_a_mistral_candidate_gets_its_familys_route_and_a_qwen_eval_is_todays(tmp_path):
    _Recorder.made = []
    s = _eval_store(tmp_path)
    with pytest.raises(RuntimeError, match="stop here"):  # qwen / qwen: the harness is built exactly as before
        EV.run(s, EV_RUN, mode="smoke", current_url="https://eval.run.app", candidate_url="https://cand.run.app",
               workdir=tmp_path / "w", harness_factory=_Recorder, progress=lambda m: None)
    assert _Recorder.made[-1].kw == {"run_id": "ev" + EV.run8(EV_RUN), "workdir": tmp_path / "w" / "harness"}
    s.write_json(f"runs/{EV_RUN}/train-config.json", {"family": "mistral"})
    probed = []

    def probe(url):
        probed.append(url)
        return {"family": "qwen" if "eval" in url else "llama-ish", "architecture": "x"}
    with pytest.raises(ConfigError, match="candidate serves a llama-ish model"):
        EV.run(s, EV_RUN, mode="smoke", current_url="https://eval.run.app", candidate_url="https://cand.run.app",
               workdir=tmp_path / "w2", harness_factory=_Recorder, progress=lambda m: None, served_family=probe)
    h = _Recorder.made[-1]
    assert h.kw["families"] == {"current": "qwen", "candidate": "mistral"} and h.exited
    assert probed == ["https://eval.run.app", "https://cand.run.app"]
    real = H.Harness({"current": "https://eval", "candidate": "https://cand"}, families=h.kw["families"],
                     binary=sys.executable)
    routes = real.model_routes()
    assert routes["current"] == H.route_spec("https://eval", "current")
    assert {k: routes["candidate"][k] for k in ("prime", "temperature", "parser_retries", "fast_finish")} == \
        {"prime": "off", "temperature": 0.15, "parser_retries": "off", "fast_finish": "on"}


def test_the_harness_family_routes():
    assert H.route_spec("u", "current", family="qwen") == H.route_spec("u", "current") == \
        {**H.ROUTE_DEFAULTS, "url": "u", "report_as": "current"}
    m = H.route_spec("u", "candidate", family="mistral", overrides={"temperature": 0.1})
    assert (m["prime"], m["temperature"]) == ("off", 0.1)  # the caller's overrides come last
    with pytest.raises(H.HarnessError, match="routes that are not there"):
        H.Harness({"current": "u"}, families={"candidate": "mistral"}, binary=sys.executable)
    with pytest.raises(ValueError):
        H.family_route("llama")


# --- the report and the decision -------------------------------------------------------------------------------------------

def test_a_family_change_warns_and_a_same_family_round_does_not():
    assert REP.family_change(None) == "" and REP.family_change({"current": "mistral", "candidate": "mistral"}) == ""
    w = REP.family_change({"current": "qwen", "candidate": "mistral"})
    assert w.startswith("Base family change: live qwen → candidate mistral") and "int route" in w
    assert REP.warnings_of({}, {"families": {"current": "qwen", "candidate": "mistral"}}) == [w]
    assert REP.warnings_of({}, {}) == []
    assert REP._measured_extras({}, {"families": {"current": "qwen", "candidate": "mistral"}})["families"]["candidate"] == "mistral"
    assert "families" not in REP._measured_extras({}, {})
    # the order the int route needs around the Approve, by direction
    assert "BEFORE you approve" in w and "chocolate-muffin" in w
    back = REP.family_change({"current": "mistral", "candidate": "qwen"})
    assert back.startswith("Base family change: live mistral → candidate qwen") and "AFTER it" in back and "int route" in back


def test_the_family_change_leads_the_decision_summary():
    """The approver reads it first, and a long failing-rules text cannot push it past the notification's 4000
    characters (pipeline_common.notification cuts the summary there)."""
    from datetime import datetime, timezone
    rules = [{"name": f"rule_{i}", "pass": False, "detail": "x" * 300} for i in range(20)]
    res = {"bar": {"rules": rules}, "families": {"current": "qwen", "candidate": "mistral"}}
    text = REP.summary_text(res, smoke=False, expires=datetime(2026, 10, 26, tzinfo=timezone.utc))
    assert len(text) > 4000 and text.startswith("Base family change: live qwen → candidate mistral")
    assert PC.notification("candidate_ready", title="t", summary=text)["summary"].startswith("Base family change")
    plain = REP.summary_text({"bar": {"rules": [{"name": "a", "pass": True, "detail": ""}]}}, smoke=False,
                             expires=datetime(2026, 10, 26, tzinfo=timezone.utc))
    assert plain == "Every bar rule passes (1)."  # a qwen round's summary is what it was


def _approvable(tmp_path, run_id: str, family: str | None):
    s = open_store(tmp_path / "bucket")
    pre = f"runs/{run_id}"
    s.write_json("state/model.json", {"tag": "r7", "digest": "sha256:" + "7" * 64, "training_manifest": "m7",
                                      "trained_at": "2026-10-18T12:00:00Z", "run_id": "rt-r7", "history": []})
    s.write_json(f"{pre}/model/image.json", {"tag": f"c-{run_id}", "digest": "sha256:" + "8" * 64})
    s.write_json(f"{pre}/eval/candidate-baseline.json", {"replicates": [{}, {}], "training_manifest": "m8"})
    s.write_jsonl(f"{pre}/prepare/cases.jsonl", [{"id": "c1"}])
    if family:
        s.write_json(f"{pre}/train-config.json", {"family": family})
    return s


def test_an_approve_records_the_models_family(tmp_path):
    s = _approvable(tmp_path / "q", "rt-q", None)
    res = DEC.approved(s, "rt-q", by="a person")
    m = s.read_json("state/model.json")
    assert "family" not in m and set(res["model"]) == {"tag", "digest", "training_manifest"}  # qwen: today's shape
    s = _approvable(tmp_path / "m", "rt-m", "mistral")
    res = DEC.approved(s, "rt-m", by="a person")
    m = s.read_json("state/model.json")
    assert m["family"] == "mistral" and res["model"]["family"] == "mistral"
    assert "family" not in m["history"][-1] and m["history"][-1]["tag"] == "r7"
    # the next round (qwen again: a rollback to blueberry-muffin's family) keeps chocolate-muffin's family in history
    s.write_json("runs/rt-q2/model/image.json", {"tag": "c-rt-q2", "digest": "sha256:" + "9" * 64})
    s.write_json("runs/rt-q2/eval/candidate-baseline.json", {"replicates": [{}, {}]})
    s.write_jsonl("runs/rt-q2/prepare/cases.jsonl", [{"id": "c2"}])
    DEC.approved(s, "rt-q2")
    m = s.read_json("state/model.json")
    assert "family" not in m and m["history"][-1]["family"] == "mistral" and m["history"][-1]["tag"] == "c-rt-m"


# --- drift: the live model's family --------------------------------------------------------------------------------------

def test_drift_asks_the_live_model_with_its_familys_route(monkeypatch):
    from souschef_lab import drift as DR
    job = SimpleNamespace(a=SimpleNamespace(eval_auth="id-token"), progress=lambda m: notes.append(m))
    notes: list[str] = []
    monkeypatch.setattr(DR.PC, "served_family", lambda url, auth="id-token": {"family": "mistral", "architecture": "mistral3",
                                                                              "parser": "ministral"})
    assert DR.live_family(job, {"tag": "cm-r7", "family": "mistral"}, "https://eval") == "mistral"
    with pytest.raises(PC.ConfigError, match="serves a mistral model .* `r7`'s state/model.json entry says qwen"):
        DR.live_family(job, {"tag": "r7"}, "https://eval")
    monkeypatch.setattr(DR.PC, "served_family", lambda url, auth="id-token": {"family": None, "error": "/api/show 404"})
    assert DR.live_family(job, {"tag": "r7"}, "https://eval") == "qwen" and not notes
    assert DR.live_family(job, {"tag": "cm-r7", "family": "mistral"}, "https://eval") == "mistral" and notes


# --- the served-family probe ------------------------------------------------------------------------------------------------

class _Show(http.server.BaseHTTPRequestHandler):
    answer: dict = {}
    status = 200

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        ok = self.path == "/api/show" and body.get("model") == "sous-chef"
        raw = json.dumps(_Show.answer if ok else {"error": "not found"}).encode()
        self.send_response(_Show.status if ok else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture
def show_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Show)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_served_family_reads_ollamas_show(show_server):
    _Show.status = 200
    _Show.answer = {"model_info": {"general.architecture": "mistral3"}, "modelfile": mx_export_modelfile()}
    got = PC.served_family(show_server, auth="none")
    assert got == {"family": "mistral", "architecture": "mistral3", "parser": "ministral"}
    _Show.answer = {"model_info": {}, "modelfile": R7_STAGED}  # no architecture: the PARSER line decides
    assert PC.served_family(show_server, auth="none")["family"] == "qwen"
    _Show.answer = {"model_info": {"general.architecture": "llama"}, "modelfile": "FROM x"}
    assert PC.served_family(show_server, auth="none")["family"] is None
    _Show.status = 500
    assert PC.served_family(show_server, auth="none") == {"family": None, "error": "/api/show 500"}
    assert PC.served_family("http://127.0.0.1:9", auth="none", timeout=5)["family"] is None  # nothing listens


# --- a whole mistral round on the fakes (the lab's retrain-dry) ----------------------------------------------------------

def test_a_mistral_round_end_to_end_on_the_fakes(tmp_path, capsys):
    """trigger.base_family mistral on the lab's rehearsal: the dataset stage pins the family, the eval and the report
    carry both arms' (the live r4 is qwen: a family change, warned), and an Approve records it in state/model.json."""
    from souschef_lab import engine as E
    from souschef_lab.retrain import dry as DRY
    if not E.engine_available():
        pytest.skip("node and the engine (COOKIE_CUTTER_DIR) are needed")
    rc = DRY.main(tmp_path, smoke=True, base_family="mistral")
    out = capsys.readouterr().out
    assert rc == 0, out[-3000:]
    s = open_store(tmp_path / "bucket")
    run_id = next(iter(s.list("runs"))).split("/")[1]
    pre = f"runs/{run_id}"
    assert s.read_json(f"{pre}/train-config.json") == {"family": "mistral"}
    ds = s.read_json(f"{pre}/result-dataset.json")["metrics"]
    assert ds["base_family"] == {"family": "mistral", "source": "trigger.base_family"}
    info = s.read_json(f"{pre}/eval/info.json")
    assert info["families"] == {"current": "qwen", "candidate": "mistral"}
    rep = s.read_json(f"{pre}/report.json")
    assert any(w.startswith("Base family change: live qwen → candidate mistral") for w in rep["warnings"])
    assert s.read_json(f"{pre}/decision.json")["results"]["families"]["candidate"] == "mistral"
    assert "Base family change" in s.read_json(f"{pre}/result-eval.json")["notification"]["summary"]
    DEC.approved(s, run_id, digest="sha256:" + "c" * 64, by="a person")
    assert s.read_json("state/model.json")["family"] == "mistral"
