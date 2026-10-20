"""tools/pipeline-train.sh's base family and bucket prefixes (docs/RETRAIN.md "The base family", docs/MISTRAL.md):
BASE_FAMILY=mistral — or, with no BASE_FAMILY, runs/<id>/train-config.json `family`, the dataset stage's pin of
trigger.base_family / infra config base_family — trains with --family mistral from the Ministral base tar, exports with
the ministral Modelfile, checks the GGUF before staging, stages it at temperature 0.15 and checks the staged Modelfile
before anything is pushed; RUNS_ROOT / BASE_ROOT move every read and write; a warm start continues a Ministral LoRA
(chocolate-muffin's base/adapters/cm-r7); unset, the commands are today's (no --family, no --base-ollama-tag, staged at
0.3, no extra step). Against the same stub gcloud and trainer python as test_pipeline_cli."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_pipeline_cli import FAKE_GCLOUD, FAKE_TRAINPY, _base_tar, _exe, _train_rows

LAB = Path(__file__).resolve().parents[1]
MXTESTS = LAB / "tools" / "mxtests"
MX_REPO_DIR = "models--mistralai--Ministral-3-8B-Instruct-2512-BF16"
MX_TAR = "mistralai-ministral-3-8b-instruct-2512-bf16-hf.tar"
MX_REV = "f6fae9795746f63c9be8344932f01275f3c63734"

# every trainer-python call is logged; an export writes a GGUF of GGUF_KIND (mistral: passes mx_gguf's check; qwen:
# fails) and the Modelfile the real export writes from its --base-modelfile (export.student_modelfile: FROM, the base's
# TEMPLATE / RENDERER / PARSER / PARAMETER lines, num_ctx 65536). infra/modules/pipeline/tests/e2e_dry.py reads this
# wrapper (E2E_BASE_FAMILY=mistral).
WRAP_TRAINPY = r'''#!/usr/bin/env bash
printf '%s\n' "$*" >> "$ARGS_LOG"
if [ "$1" = -m ] && [ "$2" = souschef_lab.export ]; then
  "$INNER_TRAINPY" "$@" || exit $?
  name=; base=base-modelfiles/qwen3.5-9b.modelfile; tag=qwen3.5:9b
  for ((i = 1; i <= $#; i++)); do
    j=$((i + 1))
    case "${!i}" in --name) name=${!j};; --base-modelfile) base=${!j};; --base-ollama-tag) tag=${!j};; esac
  done
  "$REAL_PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); import gguf_synth as G
G.write(sys.argv[2], G.mistral_kv() if sys.argv[3] == "mistral" else G.qwen_kv())' \
    "$MXTESTS" "runs/exports/$name/$name-q4_k_m.gguf" "${GGUF_KIND:-mistral}" || exit 1
  "$REAL_PY" -c 'import sys; from pathlib import Path; from souschef_lab.export import student_modelfile as S
Path(sys.argv[1]).write_text(S(Path(sys.argv[2]).read_text(), sys.argv[3], sys.argv[4], 65536))' \
    "runs/exports/$name/Modelfile.q4_k_m" "$base" "$name-q4_k_m.gguf" "$tag" || exit 1
  exit 0
fi
exec "$INNER_TRAINPY" "$@"
'''

MISTRAL_ADAPTER_CONFIG = {"peft_type": "LORA", "r": 16, "lora_alpha": 16,
                          "base_model_name_or_path": "mistralai/Ministral-3-8B-Instruct-2512-BF16"}


def _mistral_tar(src: Path, tar: Path) -> None:
    m = src / MX_REPO_DIR
    (m / "refs").mkdir(parents=True)
    (m / "refs" / "main").write_text(MX_REV)
    snap = m / "snapshots" / MX_REV
    snap.mkdir(parents=True)
    for name, body in (("config.json", "{}"), ("model-00001-of-00004.safetensors", "w"), ("tokenizer.json", "{}")):
        (snap / name).write_text(body)
    subprocess.run(["tar", "-C", str(src), "-cf", str(tar), m.name], check=True)


def _sha(dirp: Path, name: str) -> None:
    out = subprocess.run(["sha256sum", name], cwd=dirp, capture_output=True, text=True, check=True).stdout
    (dirp / f"{name}.sha256").write_text(out)


@pytest.fixture
def fam_env(tmp_path):
    lab = tmp_path / "lab"
    lab.mkdir()
    for name in ("souschef_lab", "model-image", "base-modelfiles", "Makefile", "tools"):
        (lab / name).symlink_to(LAB / name)
    gcs = tmp_path / "gcs"
    run = "mx-20261017-m3-8b"
    exp = gcs / "bkt" / "experiments" / "mistral"
    for root in (exp / "runs", gcs / "bkt" / "runs"):  # the same dataset under both prefixes
        ds = root / run / "dataset" / "dataset"
        ds.mkdir(parents=True)
        (ds / "train.jsonl").write_text(_train_rows(3, True))
        (ds / "val.jsonl").write_text(_train_rows(1, False))
        (ds / "stats.json").write_text(json.dumps({"block_drift_rows": 0, "preview_claim_kept": 0, "by_family": {"F0": 4}}))
    xb = exp / "base"
    xb.mkdir(parents=True)
    _mistral_tar(tmp_path / "mxsrc", xb / MX_TAR)
    _sha(xb, MX_TAR)
    qb = gcs / "bkt" / "base"
    qb.mkdir(parents=True)
    _base_tar(tmp_path / "qsrc", qb / "unsloth-qwen3.5-9b-hf.tar")
    _sha(qb, "unsloth-qwen3.5-9b-hf.tar")
    for f in (MX_TAR, f"{MX_TAR}.sha256"):  # the pipeline's own copy (docs/RETRAIN.md "The base family")
        shutil.copyfile(xb / f, qb / f)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    gcloud = _exe(bindir / "gcloud", FAKE_GCLOUD.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    inner = _exe(bindir / "trainpy-inner", FAKE_TRAINPY)
    trainpy = _exe(bindir / "trainpy", WRAP_TRAINPY)
    env = {**os.environ, "PIPELINE_BUCKET": "gs://bkt", "SOUS_CHEF_LAB": str(lab), "TRAINPY": str(trainpy),
           "INNER_TRAINPY": str(inner), "MXTESTS": str(MXTESTS), "ARGS_LOG": str(tmp_path / "args.log"),
           "GCLOUD": str(gcloud), "FAKE_GCS": str(gcs), "REAL_PY": sys.executable, "PIPELINE_WORKDIR": str(tmp_path / "w"),
           "HF_HOME": str(tmp_path / "hf"), "STUB_LOG": str(tmp_path / "stub.log"), "PYTHONPATH": str(lab)}
    for k in ("BASE_FAMILY", "RUNS_ROOT", "BASE_ROOT", "BASE_TAR", "TRAIN_ARMS", "TRAIN_SEED", "TRAIN_SEEDS", "INIT_ADAPTER",
              "INIT_LR", "TRAIN_GEN_CHECK", "MX_TOKENIZER_DIR"):
        env.pop(k, None)
    return {"env": env, "run": run, "gcs": gcs, "exp": exp, "tmp": tmp_path}


def _mx(te, **extra):
    env = {**te["env"], "BASE_FAMILY": "mistral", "RUNS_ROOT": "gs://bkt/experiments/mistral/runs",
           "BASE_ROOT": "gs://bkt/experiments/mistral/base", **extra}
    return subprocess.run(["bash", str(LAB / "tools" / "pipeline-train.sh"), te["run"], "full"], env=env, cwd=te["tmp"],
                          capture_output=True, text=True, timeout=300)


def _pipe(te, **extra):
    """The pipeline's own run: runs/ and base/ of the bucket, the family from train-config.json (no BASE_FAMILY)."""
    return subprocess.run(["bash", str(LAB / "tools" / "pipeline-train.sh"), te["run"], "full"], env={**te["env"], **extra},
                          cwd=te["tmp"], capture_output=True, text=True, timeout=300)


def _train_config(te, cfg: dict) -> None:
    (te["gcs"] / "bkt" / "runs" / te["run"] / "train-config.json").write_text(json.dumps(cfg))


def _calls(te, module: str) -> list[str]:
    return [ln for ln in (te["tmp"] / "args.log").read_text().splitlines() if ln.startswith(f"-m {module} ")]


def _staged(out: Path) -> str:
    return (out / "stage" / "Modelfile").read_text()


def test_mistral_trains_exports_and_checks_under_the_experiment_prefix_only(fam_env):
    te = fam_env
    r = _mx(te)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    out = te["exp"] / "runs" / te["run"] / "train"
    assert (out / "DONE").is_file()
    assert (out / "export" / f"{te['run']}-q4_k_m.gguf").is_file()
    check = json.loads((out / "export" / "gguf-check.json").read_text())
    assert check["ok"] and check["family"] == "mistral"
    assert not (te["gcs"] / "bkt" / "runs" / te["run"] / "train").exists()  # nothing under the pipeline's runs/
    assert (te["tmp"] / "hf" / "hub" / MX_REPO_DIR / "refs" / "main").is_file()
    assert f"{MX_REPO_DIR}/snapshots/{MX_REV[:8]}" in r.stdout
    trains = _calls(te, "souschef_lab.train")
    assert trains and all("--family mistral" in t for t in trains)
    (exp_call,) = _calls(te, "souschef_lab.export")
    assert "--base-modelfile base-modelfiles/ministral-3-8b.modelfile" in exp_call
    assert "--base-ollama-tag ministral-3:8b" in exp_call
    assert " gguf check ==" in r.stdout and "mx_gguf: " in r.stdout


def test_mistral_stages_the_template_at_0_15_and_checks_it_before_the_push(fam_env):
    """The model image serves exactly train/stage/Modelfile: the Ministral TEMPLATE byte for byte, PARSER ministral and
    no RENDERER, temperature 0.15 and r7's limits (top_k 20, top_p 0.95, presence_penalty 0, num_ctx 65536)."""
    from souschef_lab import chatfmt, serving_mistral
    te = fam_env
    r = _mx(te)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    mf = _staged(te["exp"] / "runs" / te["run"] / "train")
    assert chatfmt.stage_problems(mf, "mistral") == []
    assert serving_mistral.template_of(mf) == serving_mistral.TEMPLATE and "PARSER ministral\n" in mf
    assert not re.search(r"(?m)^RENDERER", mf) and "PARAMETER temperature 0.15\n" in mf
    assert "PARAMETER temperature 0.3" not in mf
    assert mf.count("PARAMETER num_ctx 65536") == 1 and "PARAMETER presence_penalty 0\n" in mf
    assert " stage check ==" in r.stdout and "is the mistral family's served model" in r.stdout


def test_mistral_stops_before_staging_on_a_gguf_that_fails_its_check(fam_env):
    te = fam_env
    r = _mx(te, GGUF_KIND="qwen")
    assert r.returncode == 1
    out = te["exp"] / "runs" / te["run"] / "train"
    assert not (out / "DONE").exists() and not (out / "export").exists() and not (out / "stage").exists()
    prog = json.loads((out / "progress.json").read_text())
    assert prog["status"] == "failed" and "gguf check" in prog["error"]


def test_a_mistral_stage_without_the_template_is_never_pushed(fam_env, tmp_path):
    """A Modelfile that is not the family's served model (here: the qwen base's lines under a mistral GGUF) stops the
    job at the stage check: nothing of export/ or stage/ goes up, no DONE."""
    te = fam_env
    wrong = tmp_path / "wrong-base-modelfiles"
    wrong.mkdir()
    shutil.copyfile(LAB / "base-modelfiles" / "qwen3.5-9b.modelfile", wrong / "ministral-3-8b.modelfile")
    lab = Path(te["env"]["SOUS_CHEF_LAB"])
    (lab / "base-modelfiles").unlink()
    (lab / "base-modelfiles").symlink_to(wrong)
    r = _mx(te)
    assert r.returncode == 1, r.stdout[-2000:]
    out = te["exp"] / "runs" / te["run"] / "train"
    prog = json.loads((out / "progress.json").read_text())
    assert prog["status"] == "failed" and "stage check" in prog["error"]
    assert not (out / "DONE").exists() and not (out / "stage").exists() and not (out / "export").exists()


def test_the_family_comes_from_train_config_json(fam_env):
    """The pipeline's switch: no BASE_FAMILY on the job; the dataset stage pinned `family` in train-config.json. The
    Ministral tar is read from the bucket's base/, everything goes to runs/<id>/train/."""
    te = fam_env
    _train_config(te, {"family": "mistral"})
    r = _pipe(te)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    out = te["gcs"] / "bkt" / "runs" / te["run"] / "train"
    assert (out / "DONE").is_file() and json.loads((out / "export" / "gguf-check.json").read_text())["ok"]
    assert "base family: mistral (mistralai/Ministral-3-8B-Instruct-2512-BF16" in r.stdout
    assert f"{MX_REPO_DIR}/snapshots/{MX_REV[:8]}" in r.stdout
    assert all("--family mistral" in t for t in _calls(te, "souschef_lab.train"))
    assert "PARAMETER temperature 0.15" in _staged(out) and not (te["exp"] / "runs" / te["run"] / "train").exists()


def test_base_family_on_the_job_wins_over_the_file(fam_env):
    te = fam_env
    _train_config(te, {"family": "mistral"})
    r = _pipe(te, BASE_FAMILY="qwen")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    out = te["gcs"] / "bkt" / "runs" / te["run"] / "train"
    assert "base family:" not in r.stdout and all("--family" not in t for t in _calls(te, "souschef_lab.train"))
    assert "PARAMETER temperature 0.3" in _staged(out) and not (out / "export" / "gguf-check.json").exists()


def test_a_bad_family_in_train_config_json_fails_at_settings(fam_env):
    te = fam_env
    _train_config(te, {"family": "llama"})
    r = _pipe(te)
    assert r.returncode == 1
    prog = json.loads((te["gcs"] / "bkt" / "runs" / te["run"] / "train" / "progress.json").read_text())
    assert prog["status"] == "failed" and prog["step"] == "settings" and "llama" in prog["error"]
    assert not (te["tmp"] / "args.log").exists() or not _calls(te, "souschef_lab.train")


def test_mistral_warm_starts_from_chocolate_muffins_adapter(fam_env):
    """trigger.train {"arms": ["base:3407", "warm:3407"], "init_adapter": "base/adapters/cm-r7"} with base_family
    mistral: the adapter is pulled from the bucket's base/ and every warm command carries it with --family mistral and
    the warm lr; the base arm trains from the base. (train.py's dry plan refuses an adapter trained on another base:
    tests/test_chatfmt.py.)"""
    te = fam_env
    ad = te["gcs"] / "bkt" / "base" / "adapters" / "cm-r7"
    (ad / "adapter").mkdir(parents=True)
    (ad / "adapter" / "adapter_config.json").write_text(json.dumps(MISTRAL_ADAPTER_CONFIG))
    (ad / "adapter" / "adapter_model.safetensors").write_text("w")
    (ad / "run.json").write_text(json.dumps({"name": "mx-20261018-m3-8b-r7-sft", "family": "mistral"}))
    _train_config(te, {"arms": ["base:3407", "warm:3407"], "init_adapter": "base/adapters/cm-r7", "family": "mistral"})
    r = _pipe(te)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    trains = _calls(te, "souschef_lab.train")
    assert trains and all("--family mistral" in t for t in trains)
    warm = [t for t in trains if "--init-adapter" in t]
    assert warm and all("/init-adapter/adapter --lr 5e-05" in t for t in warm)
    assert any("--dry-run" in t for t in warm)  # the warm arms' dry plan: train.py checks the adapter's base there
    assert any("sft-s3407" in t and "--init-adapter" not in t for t in trains if "--dry-run" not in t)
    out = te["gcs"] / "bkt" / "runs" / te["run"] / "train"
    assert (out / "DONE").is_file() and (out / "seeds" / "w3407" / "run.json").is_file()


def test_an_unknown_family_exits_2_and_writes_nothing(fam_env):
    te = fam_env
    r = subprocess.run(["bash", str(LAB / "tools" / "pipeline-train.sh"), te["run"], "full"],
                       env={**te["env"], "BASE_FAMILY": "llama"}, cwd=te["tmp"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "BASE_FAMILY must be qwen or mistral" in r.stderr
    assert not (te["gcs"] / "bkt" / "runs" / te["run"] / "train").exists()


def test_unset_the_commands_are_todays(fam_env):
    """No BASE_FAMILY / RUNS_ROOT / BASE_ROOT and no family in train-config.json: the qwen base from base/, runs/
    under the bucket, train.py without --family, export.py without --base-ollama-tag, staged at 0.3, no GGUF or stage
    check step."""
    te = fam_env
    r = subprocess.run(["bash", str(LAB / "tools" / "pipeline-train.sh"), te["run"], "full"], env=te["env"], cwd=te["tmp"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    out = te["gcs"] / "bkt" / "runs" / te["run"] / "train"
    assert (out / "DONE").is_file() and not (out / "export" / "gguf-check.json").exists()
    assert not (te["exp"] / "runs" / te["run"] / "train").exists()
    assert "models--unsloth--Qwen3.5-9B/snapshots/005429ce" in r.stdout
    assert all("--family" not in t for t in _calls(te, "souschef_lab.train"))
    (exp_call,) = _calls(te, "souschef_lab.export")
    assert exp_call.endswith("--base-modelfile base-modelfiles/qwen3.5-9b.modelfile")
    assert "gguf check" not in r.stdout and "stage check" not in r.stdout and "base family:" not in r.stdout
    mf = _staged(out)
    assert "PARAMETER temperature 0.3\n" in mf and "RENDERER qwen3.5" in mf and "PARAMETER num_ctx 65536" in mf
