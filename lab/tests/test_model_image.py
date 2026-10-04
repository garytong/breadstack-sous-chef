"""lab/model-image without Docker or a real GGUF: `make model-stage` (the
hard-linked staging dir with exactly Modelfile + the one GGUF, the Modelfile
rewritten to FROM ./<gguf> + num_ctx 65536), model-context.sh's checks, and
what model-build / model-push would run (`make -n`; model-push only prints).
The image build itself needs a real GGUF (`ollama create` reads it) and is a
manual check: lab/model-image/README.md."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
MODEL_IMAGE = LAB / "model-image"
CONTEXT_SH = MODEL_IMAGE / "model-context.sh"

pytestmark = pytest.mark.skipif(shutil.which("make") is None or shutil.which("sh") is None,
                                reason="needs make and sh")

FAKE_GGUF = b"GGUF\x00\x00\x00\x03fake weights"

MODELFILE_Q4 = (
    "# The Sous Chef lab's student: LoRA-tuned from qwen3.5:9b's base model.\n"
    "FROM ./x-q4_k_m.gguf\n"
    'TEMPLATE """{{ .Prompt }}"""\n'
    "PARAMETER temperature 0.6\n"
    "PARAMETER num_ctx 32768\n"
)


def an_export(root: Path) -> Path:
    """An export.py output: both quants, a Modelfile per quant, export.json."""
    d = root / "exports" / "x"
    d.mkdir(parents=True)
    (d / "x-q4_k_m.gguf").write_bytes(FAKE_GGUF)
    (d / "x-q8_0.gguf").write_bytes(FAKE_GGUF + b"8")
    (d / "Modelfile.q4_k_m").write_text(MODELFILE_Q4, encoding="utf-8")
    (d / "Modelfile.q8_0").write_text(MODELFILE_Q4.replace("q4_k_m", "q8_0"), encoding="utf-8")
    (d / "export.json").write_text("{}", encoding="utf-8")
    return d


def make(*args: str, dry: bool = False) -> subprocess.CompletedProcess:
    cmd = ["make", "-C", str(LAB), *(["-n"] if dry else []), *args]
    return subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "MAKEFLAGS": ""})


def context_sh(stage: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["sh", str(CONTEXT_SH), str(stage)], capture_output=True, text=True)


# --- make model-stage -----------------------------------------------------------------------------------------------------


def test_stage_holds_exactly_modelfile_and_one_hard_linked_gguf(tmp_path):
    export = an_export(tmp_path)
    stage = tmp_path / "stage"
    r = make("model-stage", "NAME=x", f"MODEL_EXPORT={export}", f"MODEL_STAGE={stage}")
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(p.name for p in stage.iterdir()) == ["Modelfile", "x-q4_k_m.gguf"]
    staged = stage / "x-q4_k_m.gguf"
    assert not staged.is_symlink()
    assert staged.stat().st_ino == (export / "x-q4_k_m.gguf").stat().st_ino  # a hard link, not a copy
    text = (stage / "Modelfile").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert [l for l in lines if l.startswith("FROM")] == ["FROM ./x-q4_k_m.gguf"]
    assert [l for l in lines if l.startswith("PARAMETER num_ctx")] == ["PARAMETER num_ctx 65536"]
    assert 'TEMPLATE """{{ .Prompt }}"""' in lines and "PARAMETER temperature 0.6" in lines
    assert "model context" in r.stderr and "x-q4_k_m.gguf" in r.stderr


def test_stage_picks_the_quant_and_starts_from_an_empty_dir(tmp_path):
    export = an_export(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "leftover.safetensors").write_bytes(b"old")
    r = make("model-stage", "NAME=x", "QUANT=q8_0", f"MODEL_EXPORT={export}", f"MODEL_STAGE={stage}")
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(p.name for p in stage.iterdir()) == ["Modelfile", "x-q8_0.gguf"]
    assert "FROM ./x-q8_0.gguf\n" in (stage / "Modelfile").read_text(encoding="utf-8")


def test_stage_without_a_num_ctx_line_still_writes_65536(tmp_path):
    export = an_export(tmp_path)
    mf = export / "Modelfile.q4_k_m"
    mf.write_text(mf.read_text(encoding="utf-8").replace("PARAMETER num_ctx 32768\n", ""), encoding="utf-8")
    stage = tmp_path / "stage"
    r = make("model-stage", "NAME=x", f"MODEL_EXPORT={export}", f"MODEL_STAGE={stage}")
    assert r.returncode == 0, r.stdout + r.stderr
    text = (stage / "Modelfile").read_text(encoding="utf-8")
    assert text.count("PARAMETER num_ctx") == 1 and text.endswith("PARAMETER num_ctx 65536\n")


def test_stage_refuses_a_missing_export_or_gguf(tmp_path):
    r = make("model-stage", "NAME=x", f"MODEL_EXPORT={tmp_path / 'nope'}", f"MODEL_STAGE={tmp_path / 's'}")
    assert r.returncode != 0 and "Modelfile.q4_k_m" in r.stdout + r.stderr
    export = an_export(tmp_path)
    (export / "x-q4_k_m.gguf").unlink()
    r = make("model-stage", "NAME=x", f"MODEL_EXPORT={export}", f"MODEL_STAGE={tmp_path / 's'}")
    assert r.returncode != 0 and "which is not in" in r.stdout + r.stderr
    assert not (tmp_path / "s").exists()


# --- model-context.sh --------------------------------------------------------------------------------------------------------


def a_stage(root: Path, modelfile: str = "FROM ./x-q4_k_m.gguf\nTEMPLATE \"\"\"{{ .Prompt }}\"\"\"\nPARAMETER num_ctx 65536\n") -> Path:
    d = root / "stage"
    d.mkdir()
    (d / "x-q4_k_m.gguf").write_bytes(FAKE_GGUF)
    (d / "Modelfile").write_text(modelfile, encoding="utf-8")
    return d


def test_context_prints_the_resolved_dir(tmp_path):
    stage = a_stage(tmp_path)
    r = context_sh(stage)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(stage.resolve())
    assert "Modelfile + x-q4_k_m.gguf" in r.stderr


@pytest.mark.parametrize("spoil, needle", [
    (lambda d: (d / "export.json").write_text("{}"), "more than the Modelfile"),
    (lambda d: (d / "sub").mkdir(), "more than the Modelfile"),
    (lambda d: (d / "x-q4_k_m.gguf").write_bytes(b"not a gguf"), "is not a GGUF file"),
    (lambda d: ((d / "x-q4_k_m.gguf").unlink(), (d / "x-q4_k_m.gguf").symlink_to(d / "Modelfile")), "is a symlink"),
    (lambda d: (d / "x-q4_k_m.gguf").unlink(), "which is not there"),
    (lambda d: (d / "Modelfile").write_text("FROM ./x-q4_k_m.gguf\nADAPTER ./a.gguf\nPARAMETER num_ctx 65536\n"), "ADAPTER line"),
    (lambda d: (d / "Modelfile").write_text("FROM qwen3.5:9b\nPARAMETER num_ctx 65536\n"), "FROM ./<file>.gguf"),
    (lambda d: (d / "Modelfile").write_text("FROM ./sub/x.gguf\nPARAMETER num_ctx 65536\n"), "subdirectory"),
    (lambda d: (d / "Modelfile").write_text("FROM ./x-q4_k_m.gguf\nFROM ./y.gguf\nPARAMETER num_ctx 65536\n"), "exactly one FROM"),
    (lambda d: (d / "Modelfile").write_text("FROM ./x-q4_k_m.gguf\nPARAMETER num_ctx 32768\n"), "num_ctx 65536"),
    (lambda d: (d / "Modelfile").unlink(), "no Modelfile"),
], ids=["extra-file", "subdir", "not-gguf", "symlink", "missing-gguf", "adapter", "from-tag", "from-subdir",
        "two-from", "num_ctx", "no-modelfile"])
def test_context_refuses(tmp_path, spoil, needle):
    stage = a_stage(tmp_path)
    spoil(stage)
    r = context_sh(stage)
    assert r.returncode == 1 and r.stdout == ""
    assert needle in r.stderr, r.stderr


def test_context_usage(tmp_path):
    r = subprocess.run(["sh", str(CONTEXT_SH)], capture_output=True, text=True)
    assert r.returncode == 1 and "usage" in r.stderr
    r = context_sh(tmp_path / "missing")
    assert r.returncode == 1 and "model-stage" in r.stderr


# --- model-build / model-push ----------------------------------------------------------------------------------------------


def test_build_recipe_is_an_amd64_build_with_the_model_context_and_image_name(tmp_path):
    r = make("model-build", "NAME=x", "MODEL_TAG=r7", f"MODEL_STAGE={tmp_path / 'stage'}", dry=True)
    assert r.returncode == 0, r.stderr
    assert "docker build --platform linux/amd64 --build-context model=" in r.stdout
    build = [l for l in r.stdout.splitlines() if "docker build" in l]
    assert len(build) == 1 and "-f model-image/Dockerfile" in build[0]
    assert build[0].rstrip("\\ ").endswith("model-image || exit 1;")  # the context is lab/model-image, not the repo root
    assert "us-east4-docker.pkg.dev/breadstack-sous-chef-int/sous-chef-model/model:r7" in r.stdout
    assert "docker push" not in r.stdout


def test_push_only_prints(tmp_path):
    r = make("model-push", "MODEL_TAG=r7", "GCP_PROJECT=some-project")
    assert r.returncode == 0, r.stderr
    assert "docker push 'us-east4-docker.pkg.dev/some-project/sous-chef-model/model:r7'" in r.stdout
    assert 'gpu_model_image_tag = "r7"' in r.stdout


def test_dockerfile_names_sous_chef_and_pins_ollama():
    text = (MODEL_IMAGE / "Dockerfile").read_text(encoding="utf-8")
    assert "ollama/ollama:0.34.4@sha256:" in text
    assert "ollama create sous-chef -f /src/Modelfile" in text
    assert "^num_ctx[[:space:]]+65536$" in text
    for env in ("OLLAMA_HOST=0.0.0.0:8080", "OLLAMA_KEEP_ALIVE=-1", "OLLAMA_NUM_PARALLEL=1", "OLLAMA_NO_CLOUD=1",
                "SOUS_CHEF_MODEL=sous-chef"):
        assert env in text
    assert (MODEL_IMAGE / ".dockerignore").read_text(encoding="utf-8").splitlines()[-2:] == ["*", "!entrypoint.sh"]
    for script in ("entrypoint.sh", "model-context.sh"):
        p = MODEL_IMAGE / script
        assert p.stat().st_mode & 0o111, f"{script} is not executable"
        assert subprocess.run(["sh", "-n", str(p)], capture_output=True).returncode == 0
    entry = (MODEL_IMAGE / "entrypoint.sh").read_text(encoding="utf-8")
    assert '/bin/ollama run "$SOUS_CHEF_MODEL" ""' in entry


@pytest.mark.skipif(sys.platform != "linux", reason="hard links across tmp filesystems differ")
def test_stage_falls_back_to_a_copy_across_filesystems(tmp_path, monkeypatch):
    """When ln fails (another filesystem) the GGUF is copied, never symlinked."""
    export = an_export(tmp_path)
    stage = tmp_path / "stage"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "ln").write_text("#!/bin/sh\nexit 1\n")
    (fake_bin / "ln").chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}", "MAKEFLAGS": ""}
    r = subprocess.run(["make", "-C", str(LAB), "model-stage", "NAME=x", f"MODEL_EXPORT={export}",
                        f"MODEL_STAGE={stage}"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "copying x-q4_k_m.gguf" in r.stdout
    staged = stage / "x-q4_k_m.gguf"
    assert not staged.is_symlink() and staged.read_bytes() == FAKE_GGUF
    assert staged.stat().st_ino != (export / "x-q4_k_m.gguf").stat().st_ino
