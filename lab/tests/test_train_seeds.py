"""Training seeds, seed selection and warm starts (r6), without a GPU: train.py's settings, plan, generation-check
parsing, `select`, the pipeline settings resolver, and tools/pipeline-train.sh's seed loop and init-adapter pull
against the stub gcloud of test_pipeline_cli.py and a stub trainer that records its arguments. The defaults must
train exactly as r4/r5 did: one seed (3407), from the base, no generation check."""
from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from souschef_lab import serving, train, trainkit
from tests.test_pipeline_cli import _complete, _exe, _gcloud_log, train_env  # noqa: F401  (train_env: a fixture)
from tests.test_train_dry import a_row, write_jsonl

LAB = Path(__file__).resolve().parents[1]


# --- train.py: settings and plan ---------------------------------------------------------------------------------------


def test_the_defaults_are_unchanged():
    a = train.parse_args([])
    assert (a.seed, a.val_seed, a.init_adapter, a.gen_check) == (3407, 3407, None, 0)
    p = train.plan(a)
    assert (p["seed"], p["val_seed"], p["start"], p["init_adapter"], p["gen_check"]) == (3407, 3407, "base", None, 0)


def test_the_validation_subset_does_not_follow_the_seed(tmp_path):
    # every seed of a round scores the same validation rows
    rows = list(range(40))
    picks = {s: trainkit.val_subset(rows, 5, train.parse_args(["--seed", str(s)]).val_seed) for s in (3407, 2, 3)}
    assert picks[2] == picks[3] == picks[3407] == trainkit.val_subset(rows, 5, 3407)
    assert train.parse_args(["--seed", "2", "--val-seed", "9"]).val_seed == 9


@pytest.mark.parametrize("bad", [["--gen-check", "-1"], ["--seed", "-2"], ["--val-seed", "-1"]])
def test_bad_seed_settings_are_refused(bad):
    with pytest.raises(SystemExit) as e:
        train.parse_args(bad)
    assert e.value.code == 2


def an_adapter(d: Path, r: int = 16, alpha: int = 16, base: str = "unsloth/Qwen3.5-9B", run: dict | None = None) -> Path:
    a = d / "adapter"
    a.mkdir(parents=True)
    (a / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": r, "lora_alpha": alpha,
                                                       "base_model_name_or_path": base}), encoding="utf-8")
    (a / "adapter_model.safetensors").write_bytes(b"\x00" * 32)
    if run is not None:
        (d / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return a


@pytest.fixture
def dataset(tmp_path) -> Path:
    d = tmp_path / "dataset"
    write_jsonl(d / "train.jsonl", [a_row(f"c{i}#1", with_history=(i == 2)) for i in range(5)])
    write_jsonl(d / "val.jsonl", [a_row("v0#1"), a_row("v1#1"), a_row("v2#1", with_history=True)])
    return d


def test_a_warm_start_dry_run_records_the_adapter(dataset, tmp_path, capsys):
    a = an_adapter(tmp_path / "r4-sft", run={"name": "r4-sft", "seed": 3407, "last_eval_loss": 0.2287, "x": 1})
    args = ["--data", str(dataset), "--name", "w", "--output", str(tmp_path / "o"), "--fields-context", "off",
            "--init-adapter", str(a), "--lr", "5e-5", "--seed", "2", "--gen-check", "4", "--dry-run"]
    assert train.main(args) == 0
    text = capsys.readouterr().out
    assert "init_adapter" in text and "r4-sft" in text and "start" in text and "5e-05" in text
    p = train.plan(train.parse_args(args))
    assert p["start"] == "init_adapter" and p["seed"] == 2 and p["gen_check"] == 4 and p["lr"] == 5e-5
    info = p["init_adapter"]
    assert (info["r"], info["alpha"], info["base"], info["bytes"]) == (16, 16, "unsloth/Qwen3.5-9B", 32)
    assert info["run"] == {"name": "r4-sft", "seed": 3407, "last_eval_loss": 0.2287}
    assert not (tmp_path / "o").exists()


def test_a_warm_start_from_another_lora_shape_or_no_adapter_is_refused(dataset, tmp_path, capsys):
    base = ["--data", str(dataset), "--name", "w", "--fields-context", "off", "--dry-run"]
    r8 = an_adapter(tmp_path / "r8", r=8, alpha=8)
    with pytest.raises(SystemExit):
        train.main(base + ["--init-adapter", str(r8)])
    assert "r8/α8" in capsys.readouterr().err
    assert train.main(base + ["--init-adapter", str(r8), "--rank", "8", "--alpha", "8"]) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit):
        train.main(base + ["--init-adapter", str(tmp_path / "nothing")])
    assert "adapter_config.json" in capsys.readouterr().err
    other = an_adapter(tmp_path / "other", base="unsloth/Qwen3-8B")
    with pytest.raises(SystemExit):
        train.main(base + ["--init-adapter", str(other)])
    assert "Qwen3-8B" in capsys.readouterr().err


def test_importing_train_for_the_seed_tools_never_imports_torch():
    code = ("import sys; sys.path.insert(0, %r); from souschef_lab.train import select_main, pipeline_config_main, "
            "gen_check_items, parse_tool_calls; heavy = sorted(m for m in sys.modules if m.split('.')[0] in "
            "('torch', 'unsloth', 'trl', 'transformers', 'datasets', 'peft', 'safetensors')); print(heavy)") % str(LAB)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "[]"


# --- the generation check's parsing (the GPU part is train.generation_check) -------------------------------------------


def test_gen_check_items_are_the_first_turns_of_fixed_non_followup_rows(dataset):
    rows = trainkit.load_split(dataset, "val")
    items = train.gen_check_items(rows, 5)
    assert [i["id"] for i in items] == ["v0#1", "v1#1"]  # the follow-up is left out
    it = items[0]
    assert it["prompt"].endswith(serving.GENERATION_PROMPT) and it["prompt"].count(serving.GENERATION_PROMPT) == 1
    assert it["expected"] == [("set_fields", {"target": {"kind": "section", "page": "home", "sectionId": "heroBanner"},
                                              "values": {"heading": "Fresh cookies, daily — café"}})]
    assert train.gen_check_items(rows, 1) == train.gen_check_items(rows, 1)  # the same row every time
    full = train.render_text(rows[0]["messages"][:3], rows[0]["tools"], 0)
    assert full.startswith(it["prompt"])
    generated = full[len(it["prompt"]):]
    assert train.gen_check_pass(generated, it["expected"])


def test_gen_check_pass_compares_calls_not_bytes():
    want = [("set_fields", {"target": {"kind": "section", "sectionId": "hero"}, "values": {"heading": "Hi"}})]
    reordered = ('<tool_call>\n<function=set_fields>\n<parameter=values>\n{"heading": "Hi"}\n</parameter>\n'
                 '<parameter=target>\n{"sectionId":"hero","kind":"section"}\n</parameter>\n</function>\n</tool_call>'
                 '<|im_end|>\nignored <tool_call><function=x></function></tool_call>')
    assert train.gen_check_pass(reordered, want)
    assert not train.gen_check_pass(reordered.replace('"Hi"', '"Hello"'), want)
    assert not train.gen_check_pass("Done.", want)
    assert train.gen_check_pass("Sure — which section?<|im_end|>", [])  # a reply where a reply is expected
    assert not train.gen_check_pass('<tool_call>\n<function=lookup>\n</function>\n</tool_call>', [])
    assert train.parse_tool_calls("<tool_call>\n<function=f>\n<parameter=q>\nplain text\n</parameter>\n"
                                  "</function>\n</tool_call>") == [("f", {"q": "plain text"})]


# --- select ----------------------------------------------------------------------------------------------------------------


def run_of(seed, loss, match=None, n=32, **kw):
    r = {"seed": seed, "name": f"x-s{seed}", "last_eval_loss": loss, "steps": 600, **kw}
    if match is not None:
        r["gen_check"] = {"n": n, "scored": n, "match": match, "rate": match / n}
    return r


def test_select_by_loss_alone_without_a_complete_generation_check():
    sel = train.select_seed([run_of(3407, 0.231), run_of(2, 0.226), run_of(3, 0.229, match=20)])
    assert sel["selected"] == 1 and sel["by"] == "loss" and [s["selected"] for s in sel["seeds"]] == [False, True, False]
    assert sel["seeds"][2]["gen_match"] == 20
    # a missing loss never wins; a tie goes to the earlier seed
    assert train.select_seed([run_of(3407, None), run_of(2, 0.3)])["selected"] == 1
    assert train.select_seed([run_of(3407, 0.3), run_of(2, 0.3)])["selected"] == 0
    assert train.select_seed([run_of(3407, None), run_of(2, None)])["selected"] == 0
    with pytest.raises(ValueError):
        train.select_seed([])


def test_select_by_generation_check_then_loss():
    runs = [run_of(3407, 0.229, match=20), run_of(2, 0.224, match=17), run_of(3, 0.226, match=19)]
    sel = train.select_seed(runs)
    assert sel["by"] == "gen_check+loss" and sel["selected"] == 2  # 2 is out (17 < 20 - 1); 3 beats 3407 on loss
    assert train.select_seed(runs, margin=0)["selected"] == 0
    assert train.select_seed(runs, margin=5)["selected"] == 1
    # an incomplete check (an error, a budget stop, another size) on any seed: the loss alone
    broken = [run_of(3407, 0.229, match=20), dict(run_of(2, 0.224, match=17), gen_check={"n": 32, "scored": 30,
                                                                                     "match": 17, "error": "OOM"})]
    assert train.select_seed(broken)["by"] == "loss" and train.select_seed(broken)["selected"] == 1
    sizes = [run_of(3407, 0.229, match=20), run_of(2, 0.224, match=10, n=16)]
    assert train.select_seed(sizes)["by"] == "loss"


def seed_run(root: Path, seed: int, loss, match=None) -> Path:
    d = root / f"x-sft-s{seed}"
    an_adapter(d, run=run_of(seed, loss, match))
    (d / "loss.jsonl").write_text(json.dumps({"step": 1, "eval_loss": loss}) + "\n")
    (d / "checkpoints" / "checkpoint-2").mkdir(parents=True)
    return d


def test_select_main_keeps_the_best_seed_with_every_score(tmp_path, capsys):
    dirs = [seed_run(tmp_path, s, l) for s, l in ((3407, 0.231), (2, 0.226), (3, 0.229))]
    out = tmp_path / "x-sft"
    assert train.main(["select", *map(str, dirs), "--into", str(out)]) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["seed"] == 2 and run["selection"]["selected_seed"] == 2 and run["selection"]["by"] == "loss"
    assert [(s["seed"], s["eval_loss"], s["selected"]) for s in run["seeds"]] == \
        [(3407, 0.231, False), (2, 0.226, True), (3, 0.229, False)]
    assert run["seeds"][1]["output"] == str(dirs[1])
    assert (out / "adapter" / "adapter_config.json").is_file() and (out / "loss.jsonl").is_file()
    assert not (out / "checkpoints").exists()
    assert json.loads((out / "seeds.json").read_text())["selection"]["selected_seed"] == 2
    assert "Selected seed 2" in capsys.readouterr().out
    # rerun replaces; --into one of the seeds, or an unfinished seed, is refused
    assert train.select_main([*map(str, dirs), "--into", str(out)]) == 0
    with pytest.raises(SystemExit):
        train.select_main([*map(str, dirs), "--into", str(dirs[0])])
    (dirs[2] / "run.json").unlink()
    with pytest.raises(SystemExit):
        train.select_main([*map(str, dirs), "--into", str(out)])


# --- the pipeline's settings ------------------------------------------------------------------------------------------------


def test_pipeline_config_defaults_are_todays_training():
    c = train.pipeline_config({}, None, "bkt")
    assert c == {"seeds": [3407], "seed_flag": False, "multi": False, "init_adapter": "", "init_lr": "",
                 "gen_check": 0, "source": {},
                 "arms": [{"arm": "base:3407", "start": "base", "seed": 3407, "tag": "s3407", "lr": 2e-4}]}
    # empty values are unset (a Vertex env entry with value "")
    assert train.pipeline_config({"TRAIN_SEEDS": "", "INIT_ADAPTER": " "}, {}, "gs://bkt") == c


def test_pipeline_config_seeds():
    c = train.pipeline_config({"TRAIN_SEED": "2"}, None, "bkt")
    assert (c["seeds"], c["seed_flag"], c["multi"], c["gen_check"]) == ([2], True, False, 0)
    c = train.pipeline_config({"TRAIN_SEEDS": "3407, 2 3"}, None, "bkt")
    assert (c["seeds"], c["multi"], c["gen_check"], c["source"]["seeds"]) == ([3407, 2, 3], True, 32, "env TRAIN_SEEDS")
    c = train.pipeline_config({"TRAIN_GEN_CHECK": "0"}, {"seeds": [3407, 2]}, "bkt")
    assert (c["seeds"], c["gen_check"], c["source"]["seeds"]) == ([3407, 2], 0, "train-config.json")
    assert train.pipeline_config({"TRAIN_SEEDS": "5"}, None, "bkt")["multi"] is False
    # the environment wins over the file
    assert train.pipeline_config({"TRAIN_SEEDS": "1,2"}, {"seeds": "7,8"}, "bkt")["seeds"] == [1, 2]
    for env, cfg, msg in (({"TRAIN_SEED": "2", "TRAIN_SEEDS": "3,4"}, None, "both"),
                          ({"TRAIN_SEEDS": "2,2"}, None, "twice"), ({"TRAIN_SEEDS": "2,x"}, None, "whole number"),
                          ({"TRAIN_SEED": "-1"}, None, "whole number"), ({"TRAIN_SEEDS": ","}, None, "no seed"),
                          ({"TRAIN_GEN_CHECK": "lots"}, None, "TRAIN_GEN_CHECK"), ({}, {"seed": 2, "lr": 1}, "unknown"),
                          ({}, {"seeds": [True, 2]}, "whole number")):
        with pytest.raises(ValueError, match=msg):
            train.pipeline_config(env, cfg, "bkt")


def test_pipeline_config_init_adapter():
    c = train.pipeline_config({"INIT_ADAPTER": "base/adapters/r4-sft/adapter/"}, None, "bkt")
    assert c["init_adapter"] == "gs://bkt/base/adapters/r4-sft/adapter" and c["init_lr"] == "5e-05"
    c = train.pipeline_config({"INIT_ADAPTER": "rt-20261014-0708-7cac28-1184ff9a", "INIT_LR": "1e-4"}, None, "gs://bkt/")
    assert c["init_adapter"] == "gs://bkt/runs/rt-20261014-0708-7cac28-1184ff9a/train/adapter/adapter"
    assert c["init_lr"] == "0.0001" and c["seeds"] == [3407] and not c["seed_flag"]
    c = train.pipeline_config({}, {"init_adapter": "gs://bkt/base/a/b", "init_lr": 2e-5, "seeds": "3407,2"}, "bkt")
    assert (c["init_adapter"], c["init_lr"], c["multi"]) == ("gs://bkt/base/a/b", "2e-05", True)
    # another bucket, or outside runs/ and base/: the trainer's account cannot read it — refused before any GPU
    for bad in ("gs://other/base/a", "gs://bkt/cases/a", "gs://bkt-2/runs/a"):
        with pytest.raises(ValueError, match="reads only"):
            train.pipeline_config({}, {"init_adapter": bad}, "bkt")
    for env, msg in (({"INIT_LR": "5e-5"}, "set INIT_ADAPTER"), ({"INIT_ADAPTER": "/home/x/adapter"}, "run id"),
                     ({"INIT_ADAPTER": "runs/a b"}, "plain gs://"), ({"INIT_ADAPTER": "base/a", "INIT_LR": "fast"}, "number"),
                     ({"INIT_ADAPTER": "base/a", "INIT_LR": "2"}, "below 1")):
        with pytest.raises(ValueError, match=msg):
            train.pipeline_config(env, None, "bkt")


def test_pipeline_config_main_prints_shell_lines(tmp_path, capsys, monkeypatch):
    for k in train.PIPELINE_ENV:
        monkeypatch.delenv(k, raising=False)
    cfg = tmp_path / "train-config.json"
    cfg.write_text(json.dumps({"seeds": "3407,2", "init_adapter": "base/a"}))
    assert train.pipeline_config_main([str(cfg), "gs://bkt"]) == 0
    out = capsys.readouterr()
    assert out.out.splitlines() == ["SEEDS=3407 2", "SEED_FLAG=1", "MULTI=1", "INIT_ADAPTER=gs://bkt/base/a",
                                    "INIT_LR=5e-05", "GEN_CHECK=32", "ARMS=warm:3407 warm:2"]
    assert "training settings: " in out.err
    assert train.pipeline_config_main(["gs://bkt"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == ["SEEDS=3407", "SEED_FLAG=0", "MULTI=0"] and lines[-1] == "ARMS=base:3407"
    cfg.write_text("[1]")
    assert train.pipeline_config_main([str(cfg), "gs://bkt"]) == 2


# --- tools/pipeline-train.sh: the seed loop and the warm start (stubs) ---------------------------------------------------

# the trainer image's python minus the GPU, as test_pipeline_cli's FAKE_TRAINPY, but it records every training
# command's arguments, and its run.json carries the seed, a loss per seed (STUB_LOSSES "seed:loss,…") and a generation
# check when asked for one
SEED_TRAINPY = r'''#!/usr/bin/env bash
if [ "$1" = -m ] && [ "$2" = souschef_lab.train ]; then
  shift 2; echo "train-args: $*" >> "$STUB_LOG"
  name=; max=; resume=0; dry=0; seed=3407; gen=0; init=; lr=0.0002
  while [ $# -gt 0 ]; do case "$1" in --name) name=$2; shift;; --max-steps) max=$2; shift;; --seed) seed=$2; shift;;
    --gen-check) gen=$2; shift;; --init-adapter) init=$2; shift;; --lr) lr=$2; shift;; --resume) resume=1;;
    --dry-run) dry=1;; esac; shift; done
  [ $dry = 1 ] && { echo "SFT plan: dry"; exit 0; }
  d=runs/adapters/$name
  ckpt() { mkdir -p "$1"; for f in adapter_model.safetensors optimizer.pt scheduler.pt trainer_state.json; do echo x > "$1/$f"; done; }
  if [ -n "${STUB_SLEEP:-}" ] && [ "$name" = "${STUB_SLEEP}" ]; then
    ckpt "$d/checkpoints/checkpoint-50"
    printf '{"step": 50, "loss": 0.5}\n' > "$d/loss.jsonl"
    echo "sleeping $name" >> "$STUB_LOG"; sleep 60; exit 0
  fi
  if [ $resume = 1 ] && [ -d "$d/checkpoints/checkpoint-2" ]; then echo "Resuming from $d/checkpoints/checkpoint-2 (optimizer, scheduler and data position restored)."; fi
  ckpt "$d/checkpoints/checkpoint-2"; mkdir -p "$d/adapter"
  echo '{}' > "$d/adapter/adapter_config.json"
  tag=${name##*-}
  loss=$(printf '%s' "${STUB_LOSSES:-}" | tr ',' '\n' | sed -n "s/^$tag://p")
  [ -n "$loss" ] || loss=$(printf '%s' "${STUB_LOSSES:-}" | tr ',' '\n' | sed -n "s/^$seed://p"); loss=${loss:-0.3}
  g=null; [ "$gen" -gt 0 ] && g="{\"n\": $gen, \"scored\": $gen, \"match\": $((gen / 2))}"
  start=base; [ -n "$init" ] && start=init_adapter
  echo "{\"seed\": $seed, \"name\": \"$name\", \"steps\": ${max:-8}, \"updates\": ${max:-8}, \"last_eval_loss\": $loss, \"gen_check\": $g, \"start\": \"$start\", \"lr\": $lr, \"minutes\": ${STUB_MINUTES:-0.1}}" > "$d/run.json"
  printf '{"step": 1, "eval_loss": 0.7}\n{"step": %s, "eval_loss": %s}\n' "${max:-8}" "$loss" > "$d/loss.jsonl"
  echo "trained $name to ${max:-all}" >> "$STUB_LOG"
  exit 0
fi
if [ "$1" = -m ] && [ "$2" = souschef_lab.export ]; then
  shift 2; name=
  while [ $# -gt 0 ]; do case "$1" in --name) name=$2; shift;; esac; shift; done
  e=runs/exports/$name; mkdir -p "$e"
  printf 'GGUFfake' > "$e/$name-q4_k_m.gguf"
  printf 'FROM ./%s-q4_k_m.gguf\nPARAMETER temperature 1\nPARAMETER top_k 20\n' "$name" > "$e/Modelfile.q4_k_m"
  echo "exported $name" >> "$STUB_LOG"
  exit 0
fi
exec "$REAL_PY" "$@"
'''


@pytest.fixture
def seed_env(train_env):  # noqa: F811
    te = train_env
    te["env"]["TRAINPY"] = str(_exe(te["tmp"] / "bin" / "seedpy", SEED_TRAINPY))
    for k in train.PIPELINE_ENV:
        te["env"].pop(k, None)
    te["out"] = te["gcs"] / "bkt" / "runs" / te["run"] / "train"
    return te


def _train(te, mode="full", **env):
    return subprocess.run(["bash", str(LAB / "tools" / "pipeline-train.sh"), te["run"], mode], env={**te["env"], **env},
                          cwd=te["tmp"], capture_output=True, text=True, timeout=300)


def _train_args(te) -> list[str]:
    return [ln[len("train-args: "):] for ln in (te["tmp"] / "stub.log").read_text().splitlines()
            if ln.startswith("train-args: ")]


def test_the_default_round_trains_as_before(seed_env):
    te = seed_env
    r = _train(te)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    args = _train_args(te)
    assert len(args) == 4  # dry plan, the resume smoke (2, then 4), the SFT
    for a in args:
        assert "--seed" not in a and "--init-adapter" not in a and "--gen-check" not in a and "--lr" not in a, a
    assert args[-1].endswith(f"--name {te['run']}-sft --checkpoint-steps 50 --resume")
    assert not (te["out"] / "seeds").exists()
    run = json.loads((te["out"] / "adapter" / "run.json").read_text())
    assert run["seed"] == 3407 and "seeds" not in run
    prog = json.loads((te["out"] / "progress.json").read_text())
    assert prog["status"] == "done" and "seeds" not in prog and "seeds_planned" not in prog
    assert '"seeds": [3407], "seed_flag": false' in r.stdout


def test_one_other_seed(seed_env):
    te = seed_env
    r = _train(te, TRAIN_SEED="2")
    assert r.returncode == 0, r.stdout[-3000:]
    sft = _train_args(te)[-1]
    assert "--seed 2" in sft and f"--name {te['run']}-sft " in sft
    assert json.loads((te["out"] / "adapter" / "run.json").read_text())["seed"] == 2
    assert not (te["out"] / "seeds").exists()


def test_several_seeds_train_each_and_keep_the_best(seed_env):
    te = seed_env
    r = _train(te, TRAIN_SEEDS="3407,2,3", STUB_LOSSES="3407:0.231,2:0.226,3:0.229")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    args = _train_args(te)
    sfts = [a for a in args if "-sft-s" in a]
    assert [a.split("--seed ")[1].split()[0] for a in sfts] == ["3407", "2", "3"]
    assert all("--gen-check 32" in a and "--checkpoint-steps 50 --resume" in a for a in sfts)
    for s in (3407, 2, 3):
        assert json.loads((te["out"] / "seeds" / f"s{s}" / "run.json").read_text())["seed"] == s
        assert (te["out"] / "seeds" / f"s{s}" / "adapter" / "adapter_config.json").is_file()
        assert not (te["out"] / "seeds" / f"s{s}" / "checkpoints").exists()
        assert (te["out"] / "checkpoints" / f"s{s}" / "checkpoint-2" / "trainer_state.json").is_file()
    run = json.loads((te["out"] / "adapter" / "run.json").read_text())
    assert run["seed"] == 2 and run["selection"]["selected_seed"] == 2 and run["selection"]["by"] == "gen_check+loss"
    assert [(s["seed"], s["eval_loss"], s["gen_match"], s["selected"]) for s in run["seeds"]] == \
        [(3407, 0.231, 16, False), (2, 0.226, 16, True), (3, 0.229, 16, False)]
    prog = json.loads((te["out"] / "progress.json").read_text())
    assert prog["status"] == "done" and prog["seed"] == 2 and prog["seeds_planned"] == [3407, 2, 3]
    assert prog["last_step"] == 3 * 8  # every seed's updates: never below what the seed loop reported
    assert [s["seed"] for s in prog["seeds"] if s["selected"]] == [2] and prog["selection"]["selected_seed"] == 2
    assert (te["out"] / "DONE").is_file() and (te["out"] / "export" / f"{te['run']}-q4_k_m.gguf").is_file()
    assert f"exported {te['run']}" in (te["tmp"] / "stub.log").read_text()


def test_a_seed_finished_in_an_earlier_attempt_is_not_trained_again(seed_env):
    te = seed_env
    done = te["out"] / "seeds" / "s3407"
    (done / "adapter").mkdir(parents=True)
    (done / "adapter" / "adapter_config.json").write_text("{}")
    (done / "run.json").write_text(json.dumps({"seed": 3407, "steps": 8, "last_eval_loss": 0.2}))
    _complete(te["out"] / "checkpoints" / "s2" / "checkpoint-2")  # seed 2 was preempted after a checkpoint
    r = _train(te, TRAIN_SEEDS="3407,2", STUB_LOSSES="2:0.25")
    assert r.returncode == 0, r.stdout[-3000:]
    log = (te["tmp"] / "stub.log").read_text()
    assert f"trained {te['run']}-sft-s3407" not in log and f"trained {te['run']}-sft-s2 to all" in log
    assert "smoke-sft" not in log  # a resumed round skips the resume smoke
    assert f"Resuming from runs/adapters/{te['run']}-sft-s2/checkpoints/checkpoint-2" in r.stdout
    run = json.loads((te["out"] / "adapter" / "run.json").read_text())
    assert run["seed"] == 3407 and run["selection"]["by"] == "loss"  # the earlier seed had no generation check


def test_an_a100_rung_trains_no_new_extra_seed(seed_env):
    te = seed_env
    # nothing finished yet: only the first seed
    r = _train(te, TRAIN_SEEDS="3407,2,3", STUB_LOSSES="3407:0.231", TRAIN_RUNG="a100-40-std")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "arms base:3407 base:2 base:3 -> base:3407 (" in r.stdout
    sfts = [a for a in _train_args(te) if "-sft-s" in a]
    assert [a.split("--seed ")[1].split()[0] for a in sfts] == ["3407"]
    run = json.loads((te["out"] / "adapter" / "run.json").read_text())
    assert run["seed"] == 3407 and [s["seed"] for s in run["seeds"]] == [3407]
    assert [k["arm"] for k in run["selection"]["skipped"]] == ["base:2", "base:3"]
    assert "A100" in run["selection"]["skipped"][0]["reason"]


def test_an_a100_rung_keeps_the_seeds_an_h100_attempt_finished(seed_env):
    te = seed_env
    for s, loss in ((3407, 0.23), (2, 0.22)):
        done = te["out"] / "seeds" / f"s{s}"
        (done / "adapter").mkdir(parents=True)
        (done / "adapter" / "adapter_config.json").write_text("{}")
        (done / "run.json").write_text(json.dumps({"seed": s, "steps": 8, "last_eval_loss": loss}))
    r = _train(te, TRAIN_SEEDS="3407,2,3", TRAIN_RUNG="a100-80-std")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "arms base:3407 base:2 base:3 -> base:3407 base:2 (" in r.stdout
    assert not [a for a in _train_args(te) if "-sft-s" in a]   # both pulled, none trained, seed 3 dropped
    assert json.loads((te["out"] / "adapter" / "run.json").read_text())["seed"] == 2


def test_several_seeds_in_smoke(seed_env):
    te = seed_env
    r = _train(te, "smoke", TRAIN_SEEDS="3407,2")
    assert r.returncode == 0, r.stdout[-3000:]
    log = (te["tmp"] / "stub.log").read_text()
    for s in (3407, 2):
        assert f"trained {te['run']}-sft-s{s} to 4" in log and f"trained {te['run']}-sft-s{s} to 6" in log
    assert all("--gen-check 2" in a for a in _train_args(te) if "-sft-s" in a)


def test_a_warm_start_from_the_config_file_pulls_the_adapter(seed_env):
    te = seed_env
    a = te["gcs"] / "bkt" / "base" / "adapters" / "r4-sft"
    an_adapter(a, run={"name": "r4-sft"})
    (te["gcs"] / "bkt" / "runs" / te["run"] / "train-config.json").write_text(
        json.dumps({"init_adapter": "base/adapters/r4-sft"}))
    r = _train(te)
    assert r.returncode == 0, r.stdout[-3000:]
    w = Path(te["env"]["PIPELINE_WORKDIR"])
    for args in _train_args(te):  # the dry plan, the resume smoke and the SFT: all from the adapter
        assert f"--init-adapter {w}/init-adapter/adapter --lr 5e-05" in args, args
    assert (w / "init-adapter" / "adapter" / "adapter_model.safetensors").is_file()
    assert '"init_adapter": "gs://bkt/base/adapters/r4-sft"' in r.stdout


def test_a_warm_start_from_a_round_id_and_its_own_lr(seed_env):
    te = seed_env
    prev = "rt-20261014-0708-7cac28-1184ff9a"
    an_adapter(te["gcs"] / "bkt" / "runs" / prev / "train" / "adapter")
    r = _train(te, INIT_ADAPTER=prev, INIT_LR="1e-4", TRAIN_SEED="3")
    assert r.returncode == 0, r.stdout[-3000:]
    sft = _train_args(te)[-1]
    w = Path(te["env"]["PIPELINE_WORKDIR"])
    assert f"--init-adapter {w}/init-adapter --lr 0.0001" in sft and "--seed 3" in sft


@pytest.mark.parametrize("env,want", [({"TRAIN_SEED": "2", "TRAIN_SEEDS": "3,4"}, "both set"),
                                      ({"INIT_ADAPTER": "base/adapters/none"}, "pull init adapter"),
                                      ({"INIT_LR": "5e-5"}, "set INIT_ADAPTER"),
                                      ({"TRAIN_ARMS": "base:3407,warm:3407"}, "set INIT_ADAPTER"),
                                      ({"TRAIN_ARMS": "base:3407", "TRAIN_SEEDS": "2,3"}, "both set")])
def test_bad_settings_fail_before_any_training(seed_env, env, want):
    te = seed_env
    te["env"]["TRAIN_NONCE"] = "ab12-9"
    r = _train(te, **env)
    assert r.returncode == 1
    prog = json.loads((te["out"] / "progress.json").read_text())
    assert prog["status"] == "failed" and want in prog["error"], prog
    assert (te["out"] / "failed-ab12-9.json").is_file()
    log = te["tmp"] / "stub.log"
    assert not log.is_file() or "trained" not in log.read_text()


def test_a_preempted_seed_reports_progress_across_seeds_and_pushes_its_own_checkpoints(seed_env):
    te = seed_env
    glog = te["tmp"] / "gcloud.log"
    env = {**te["env"], "TRAIN_SEEDS": "3407,2", "STUB_SLEEP": f"{te['run']}-sft-s2", "FAKE_GCLOUD_LOG": str(glog)}
    _complete(te["out"] / "checkpoints" / "s3407" / "checkpoint-2")  # a resumed round: no resume smoke
    out_f = (te["tmp"] / "train.out").open("w")
    proc = subprocess.Popen(["bash", str(LAB / "tools" / "pipeline-train.sh"), te["run"], "full"], env=env,
                            cwd=te["tmp"], stdout=out_f, stderr=subprocess.STDOUT, text=True)
    log = te["tmp"] / "stub.log"
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and not (log.is_file() and "sleeping" in log.read_text()):
        time.sleep(0.2)
    assert log.is_file() and "sleeping" in log.read_text()
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=60)
    out_f.close()
    assert proc.returncode == 143, (te["tmp"] / "train.out").read_text()[-2000:]
    prog = json.loads((te["out"] / "progress.json").read_text())
    assert prog["status"] == "interrupted" and prog["seed"] == 2 and prog["step"].startswith("SFT arm base:2")
    assert prog["last_step"] == 8 + 50  # seed 3407's 8 updates, then 50 of seed 2's
    assert (te["out"] / "checkpoints" / "s2" / "checkpoint-50" / "trainer_state.json").is_file()
    assert (te["out"] / "seeds" / "s3407" / "run.json").is_file() and not (te["out"] / "seeds" / "s2").exists()
    assert not (te["out"] / "DONE").exists()
    assert any("/checkpoints/s2/checkpoint-50/trainer_state.json" in ln[-1] for ln in _gcloud_log(glog))


# --- arms: from-base seeds and a warm start in one round (the user's r6 decision, 2026-10-15) ------------------------


def test_pipeline_config_arms():
    c = train.pipeline_config({}, {"arms": ["base:3407", "base:2", "warm:3407"], "init_adapter": "base/adapters/r4-sft"},
                              "bkt")
    assert [(a["arm"], a["start"], a["seed"], a["tag"], a["lr"]) for a in c["arms"]] == [
        ("base:3407", "base", 3407, "s3407", 2e-4), ("base:2", "base", 2, "s2", 2e-4),
        ("warm:3407", "warm", 3407, "w3407", 5e-5)]
    assert (c["seeds"], c["multi"], c["gen_check"], c["init_adapter"], c["init_lr"]) == (
        [3407, 2, 3407], True, 32, "gs://bkt/base/adapters/r4-sft", "5e-05")
    # a string, bare kinds (seed 3407), the warm arms' own lr; the environment wins over the file
    c = train.pipeline_config({"TRAIN_ARMS": "base warm:2", "INIT_ADAPTER": "rt-1", "INIT_LR": "1e-4"},
                              {"arms": ["base:9"]}, "bkt")
    assert [a["arm"] for a in c["arms"]] == ["base:3407", "warm:2"] and c["arms"][1]["lr"] == 1e-4
    assert c["source"]["arms"] == "env TRAIN_ARMS"
    # one arm is today's single path (no seed loop); seeds without arms keep r6's first meaning
    one = train.pipeline_config({}, {"arms": "warm:2", "init_adapter": "base/a"}, "bkt")
    assert (one["multi"], one["seeds"], one["seed_flag"], one["gen_check"]) == (False, [2], True, 0)
    assert [a["arm"] for a in train.pipeline_config({}, {"seeds": "3407,2", "init_adapter": "base/a"}, "bkt")["arms"]] \
        == ["warm:3407", "warm:2"]
    assert [a["arm"] for a in train.pipeline_config({}, {"seeds": "3407,2"}, "bkt")["arms"]] == ["base:3407", "base:2"]
    for cfg, msg in (({"arms": "base:1,warm:1"}, "set INIT_ADAPTER"), ({"arms": "base:1", "init_adapter": "base/a"},
                     "no arm is warm"), ({"arms": "base:1", "seed": 2}, "both set"), ({"arms": "base:x"}, "base:<seed>"),
                     ({"arms": [True]}, "base:<seed>"), ({"arms": "warm:1 warm:1", "init_adapter": "base/a"}, "twice"),
                     ({"arms": "base:1,base:2,base:3,base:4,base:5"}, "at most 4"), ({"seeds": "1,2,3,4,5"}, "at most 4")):
        with pytest.raises(ValueError, match=msg):
            train.pipeline_config({}, cfg, "bkt")


def test_window_estimate():
    w = train.window_estimate(3, 640)  # r4/r5-sized: 640 updates an arm
    assert w["n_arms"] == 3 and w["fits"] and 7.5 < w["total_h"] < 9.5 and w["window_h"] == 12
    assert train.window_estimate(3, 900)["fits"] and not train.window_estimate(4, 900)["fits"]
    assert train.window_estimate(4, 900)["fit_arms"] == 3
    assert train.window_estimate(3, 640, a100=True)["fit_arms"] == 2 and not train.window_estimate(3, 640, a100=True)["fits"]


def test_select_records_each_arm_and_the_skipped_ones(tmp_path, capsys):
    dirs = []
    for tag, seed, loss, start in (("s3407", 3407, 0.231, "base"), ("s2", 2, 0.229, "base"),
                                   ("w3407", 3407, 0.221, "init_adapter")):
        d = tmp_path / f"x-sft-{tag}"
        an_adapter(d, run={**run_of(seed, loss, 20), "start": start, "lr": 5e-5 if start != "base" else 2e-4})
        dirs.append(d)
    out = tmp_path / "x-sft"
    assert train.select_main([*map(str, dirs), "--into", str(out), "--skipped", "base:3=the 12 h window"]) == 0
    run = json.loads((out / "run.json").read_text())
    sel = run["selection"]
    assert sel["selected_arm"] == "warm:3407" and sel["selected_seed"] == 3407 and run["start"] == "init_adapter"
    assert sel["arms"] == ["base:3407", "base:2", "warm:3407"]
    assert sel["skipped"] == [{"arm": "base:3", "reason": "the 12 h window"}]
    assert [(s["arm"], s["start"], s["lr"], s["selected"]) for s in run["seeds"]] == [
        ("base:3407", "base", 2e-4, False), ("base:2", "base", 2e-4, False), ("warm:3407", "warm", 5e-5, True)]
    text = capsys.readouterr().out
    assert "arm warm:3407: final validation loss 0.221" in text and "arm base:3: not trained" in text
    with pytest.raises(SystemExit):
        train.select_main([*map(str, dirs), "--into", str(out), "--skipped", "=why"])


def _r4_in_bucket(te) -> Path:
    an_adapter(te["gcs"] / "bkt" / "base" / "adapters" / "r4-sft", run={"name": "r4-sft", "seed": 3407})
    return Path(te["env"]["PIPELINE_WORKDIR"]) / "init-adapter" / "adapter"


def test_base_and_warm_arms_train_in_one_round_and_the_best_is_kept(seed_env):
    te = seed_env
    init = _r4_in_bucket(te)
    (te["gcs"] / "bkt" / "runs" / te["run"] / "train-config.json").write_text(json.dumps(
        {"arms": ["base:3407", "base:2", "warm:3407"], "init_adapter": "base/adapters/r4-sft"}))
    r = _train(te, STUB_LOSSES="s3407:0.231,s2:0.229,w3407:0.221")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    args = _train_args(te)
    warm = f"--init-adapter {init} --lr 5e-05"
    sfts = {a.split("--name ")[1].split()[0].rsplit("-", 1)[1]: a for a in args if "-sft-" in a and "--dry-run" not in a}
    assert sorted(sfts) == ["s2", "s3407", "w3407"]
    for tag in ("s3407", "s2"):
        assert "--init-adapter" not in sfts[tag] and "--lr" not in sfts[tag]  # the recipe's lr from the base
    assert warm in sfts["w3407"] and "--seed 3407" in sfts["w3407"]
    assert all("--gen-check 32" in a and "--checkpoint-steps 50 --resume" in a for a in sfts.values())
    # the warm adapter is proved before any arm trains: a dry plan of the warm arms and the resume smoke load it
    assert any("--dry-run" in a and warm in a for a in args) and any("--dry-run" in a and warm not in a for a in args)
    assert all(warm in a for a in args if "smoke-sft" in a)
    for tag in ("s3407", "s2", "w3407"):
        assert (te["out"] / "seeds" / tag / "run.json").is_file()
        assert (te["out"] / "checkpoints" / tag / "checkpoint-2" / "trainer_state.json").is_file()
    run = json.loads((te["out"] / "adapter" / "run.json").read_text())
    assert run["selection"]["selected_arm"] == "warm:3407" and run["start"] == "init_adapter"
    assert [(s["arm"], s["eval_loss"], s["gen_match"]) for s in run["seeds"]] == [
        ("base:3407", 0.231, 16), ("base:2", 0.229, 16), ("warm:3407", 0.221, 16)]
    assert run["selection"]["skipped"] == []
    prog = json.loads((te["out"] / "progress.json").read_text())
    assert prog["status"] == "done" and prog["arm"] == "warm:3407"
    assert prog["arms_planned"] == ["base:3407", "base:2", "warm:3407"] and prog["seeds_planned"] == [3407, 2, 3407]
    assert [(s["arm"], s["start"], s["selected"]) for s in prog["seeds"]] == [
        ("base:3407", "base", False), ("base:2", "base", False), ("warm:3407", "warm", True)]
    assert prog["last_step"] == 3 * 8
    assert (te["out"] / "DONE").is_file()


def test_a_base_arm_can_win_over_the_warm_start(seed_env):
    te = seed_env
    _r4_in_bucket(te)
    r = _train(te, TRAIN_ARMS="base:3407,warm:3407", INIT_ADAPTER="base/adapters/r4-sft",
               STUB_LOSSES="s3407:0.219,w3407:0.221")
    assert r.returncode == 0, r.stdout[-3000:]
    run = json.loads((te["out"] / "adapter" / "run.json").read_text())
    assert run["selection"]["selected_arm"] == "base:3407" and run["start"] == "base"


def test_the_12h_window_guard_skips_an_arm_that_would_not_end_in_time(seed_env):
    te = seed_env
    # each arm reports 30 min; 4,000 s of window less 2,700 s for the export leaves room for none after the first
    r = _train(te, TRAIN_ARMS="base:3407,base:2,base:3", STUB_MINUTES="30", TRAIN_WINDOW_S="4000",
               TRAIN_TAIL_S="2700", STUB_LOSSES="s3407:0.25")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "arm base:2 not started: the 12 h window" in r.stdout
    assert [a for a in _train_args(te) if "-sft-s" in a and "--dry-run" not in a][0].count("--seed 3407") == 1
    assert len([a for a in _train_args(te) if "-sft-s" in a and "--dry-run" not in a]) == 1
    run = json.loads((te["out"] / "adapter" / "run.json").read_text())
    assert run["selection"]["selected_arm"] == "base:3407"
    assert [k["arm"] for k in run["selection"]["skipped"]] == ["base:2", "base:3"]
    assert all("12 h window" in k["reason"] for k in run["selection"]["skipped"])
    assert (te["out"] / "DONE").is_file()


def test_the_window_guard_scales_up_an_arm_that_was_resumed_part_way(seed_env):
    te = seed_env
    done = te["out"] / "seeds" / "s3407"   # finished in an earlier attempt, 500 of its 600 updates before a preemption
    (done / "adapter").mkdir(parents=True)
    (done / "adapter" / "adapter_config.json").write_text("{}")
    (done / "run.json").write_text(json.dumps({"seed": 3407, "steps": 600, "resumed_from_step": 500, "minutes": 10,
                                               "last_eval_loss": 0.2}))
    # 10 min for the last 100 updates is about 60 min an arm: 3,600 s + 2,700 s is over a 6,000 s window
    r = _train(te, TRAIN_ARMS="base:3407,base:2", TRAIN_WINDOW_S="6000", TRAIN_TAIL_S="2700")
    assert r.returncode == 0, r.stdout[-3000:]
    assert "arm base:2 not started" in r.stdout and "an arm takes about 3600s" in r.stdout
    assert json.loads((te["out"] / "adapter" / "run.json").read_text())["selection"]["skipped"][0]["arm"] == "base:2"


def test_the_default_window_trains_every_arm(seed_env):
    r = _train(seed_env, TRAIN_ARMS="base:3407,base:2", STUB_MINUTES="30")
    assert r.returncode == 0, r.stdout[-3000:]
    assert "not started" not in r.stdout
    assert json.loads((seed_env["out"] / "adapter" / "run.json").read_text())["selection"]["skipped"] == []


def test_an_a100_rung_keeps_a_finished_warm_arm(seed_env):
    te = seed_env
    _r4_in_bucket(te)
    done = te["out"] / "seeds" / "w3407"
    (done / "adapter").mkdir(parents=True)
    (done / "adapter" / "adapter_config.json").write_text("{}")
    (done / "run.json").write_text(json.dumps({"seed": 3407, "steps": 8, "last_eval_loss": 0.2, "start": "init_adapter"}))
    r = _train(te, TRAIN_ARMS="base:3407,warm:3407", INIT_ADAPTER="base/adapters/r4-sft", TRAIN_RUNG="a100-40-std")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "arms base:3407 warm:3407 -> warm:3407 (" in r.stdout
    assert not [a for a in _train_args(te) if "-sft-" in a and "--dry-run" not in a and "smoke" not in a]
    run = json.loads((te["out"] / "adapter" / "run.json").read_text())
    assert run["selection"]["selected_arm"] == "warm:3407"
    assert [k["arm"] for k in run["selection"]["skipped"]] == ["base:3407"]


def test_arms_in_smoke(seed_env):
    te = seed_env
    init = _r4_in_bucket(te)
    r = _train(te, "smoke", TRAIN_ARMS="base:3407,warm:3407", INIT_ADAPTER="base/adapters/r4-sft")
    assert r.returncode == 0, r.stdout[-3000:]
    log = (te["tmp"] / "stub.log").read_text()
    for tag in ("s3407", "w3407"):
        assert f"trained {te['run']}-sft-{tag} to 4" in log and f"trained {te['run']}-sft-{tag} to 6" in log
    w = [a for a in _train_args(te) if "-sft-w3407" in a]
    assert len(w) == 2 and all(f"--init-adapter {init} --lr 5e-05" in a and "--gen-check 2" in a for a in w)


def test_one_warm_arm_is_the_single_path(seed_env):
    te = seed_env
    init = _r4_in_bucket(te)
    r = _train(te, TRAIN_ARMS="warm:2", INIT_ADAPTER="base/adapters/r4-sft")
    assert r.returncode == 0, r.stdout[-3000:]
    sft = _train_args(te)[-1]
    assert f"--init-adapter {init} --lr 5e-05" in sft and "--seed 2" in sft and f"--name {te['run']}-sft " in sft
    assert not (te["out"] / "seeds").exists()


# --- the trainer image's self-test --------------------------------------------------------------------------------------


def _selftest_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("trainer_selftest", LAB.parent / "images" / "trainer" / "selftest.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_selftest_checks_the_seed_tools_without_a_gpu():
    st = _selftest_module()
    d = st.check_seed_tools()
    assert d["default_seeds"] == [3407] and d["select"] == "ok" and d["gen_check_parse"] == "ok"
    assert d["arms"] == ["base:3407", "base:2", "warm:3407"]
