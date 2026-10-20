"""dry — `python -m souschef_lab.pipeline retrain-dry`: every retrain stage end to
end on a local bucket, with no network and no model (the rehearsal of
docs/RETRAIN.md; tests/test_pipeline_cli.py runs it).

Seeds <workdir>/bucket like playbook R0 would, on a pretend drift:

  manifests/<exam>.json      the exam's manifest (bf788701) = the live model's training manifest
  manifests/<grown>.json     the same plus one new colour field on the buttons panel = "deployed"
  state/model.json           {tag r4, digest sha256:dry-r4, training_manifest <exam>}
  cases/pool/r4/…            a small generated pool (F0 + F0H, the generator's verbatim cases)
  runs/<id>/run.json         {deployed: <grown>, trigger: dry}

then runs prepare (the fake author) → collect (the fake service, oracle) →
dataset (the parity gate when $SOUS_CHEF_BIN is set) → train / image build stubbed
(train/progress.json, train/job.json, model/image.json as the trainer and Cloud
Build would leave them) → eval (fake arms) → the report and the decision.
Needs node + the engine (COOKIE_CUTTER_DIR or --engine-dir).

`--base-family mistral` puts trigger.base_family in run.json (docs/RETRAIN.md
"The base family"): the dataset stage pins it in train-config.json, the stubbed
trainer leaves the family's staged Modelfile (the Ministral TEMPLATE, PARSER
ministral), and the eval and report carry both arms' families (the live r4 is
qwen: a family change). Unset, everything is as it always was.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
from pathlib import Path

from ..manifest import Manifest
from .common import open_store, read_jsonl

NEW_FIELD = ("ButtonsSection", "buttonShadowColor", {"type": "color", "label": "Button Shadow Color",
                                                     "path": "globalSettings.buttonShadowColor",
                                                     "help": "The soft shadow under every button."})


def grown_manifest(doc: dict) -> dict:
    d = copy.deepcopy(doc)
    panel, key, spec = NEW_FIELD
    d["globalPanels"][panel]["fields"][key] = dict(spec)
    d["generated_from"] = {**(d.get("generated_from") or {}), "sha": "e7e90106", "build_id": "dry"}
    return d


def generated_pool(m: Manifest, n_train: int, n_hold: int) -> list[dict]:
    """The generator's verbatim cases in r4cases' schema (F0, F0H); diffs are measured by the rebase."""
    from ..cases import SET_KINDS, Generator
    from ..r4cases import normalise
    g = Generator(m, seed=0, template="vintage")
    rows = [c.to_row() for c in itertools.islice(g.all_cases(SET_KINDS), 600)]
    rows.sort(key=lambda r: hashlib.sha1(str(r["id"]).encode()).hexdigest())
    train = [r for r in rows if not r.get("holdout")][:n_train]
    hold = [r for r in rows if r.get("holdout")][:n_hold]
    out = []
    for fam, rs in (("F0", train), ("F0H", hold)):
        for i, r in enumerate(rs, 1):
            r = copy.deepcopy(r)
            r.setdefault("meta", {}).update({"style": "verbatim", "source_case": r["id"], "scoring": "exact"})
            c = normalise(r, fam, "dry-generated")
            c["id"] = f"r4-{fam}-{i:04d}"
            c["holdout"] = fam == "F0H"
            out.append(c)
    return out


def seed(root: Path, *, smoke: bool, base_family: str | None = None) -> tuple[str, str]:
    from ..exam import MANIFEST_FILE
    bucket = root / "bucket"
    store = open_store(bucket)
    exam_doc = json.loads(Path(MANIFEST_FILE).read_text(encoding="utf-8"))
    exam_m = Manifest(exam_doc)
    grown = grown_manifest(exam_doc)
    gm = Manifest(grown)
    store.write_json(f"manifests/{exam_m.hash}.json", exam_doc)
    store.write_json(f"manifests/{gm.hash}.json", grown)
    store.write_json("state/model.json", {"tag": "r4", "digest": "sha256:dry-r4", "training_manifest": exam_m.hash,
                                          "trained_at": "2026-10-10T20:00:00Z", "run_id": "r4", "history": []})
    store.write_json("state/deployed.json", {"hash": gm.hash, "launcher_build": "dry", "generated_from": grown["generated_from"]})
    pool = generated_pool(exam_m, 16 if smoke else 30, 6 if smoke else 12)
    store.write_jsonl("cases/pool/r4/cases-all.jsonl", pool)
    store.write_json("cases/pool/index.json", {"files": [{"path": "cases/pool/r4/cases-all.jsonl", "manifest": exam_m.hash,
                                                          "round": "r4", "trained_in": "r4"}]})
    run_id = f"rt-20261012-0310-{gm.hash[:8]}"
    store.write_json(f"runs/{run_id}/run.json", {"run_id": run_id, "mode": "smoke" if smoke else "full",
                                                 "deployed": {"hash": gm.hash, "launcher_build": "dry",
                                                              "generated_from": grown["generated_from"]},
                                                 "trigger": {"drift_run": None, "reason": "retrain-dry",
                                                             **({"base_family": base_family} if base_family else {})}})
    return str(bucket), run_id


def stub_training(store, run_id: str) -> None:
    """What the trainer (pipeline-train.sh) and the Cloud Build step leave behind."""
    pre = f"runs/{run_id}"
    store.write_json(f"{pre}/train/progress.json", {"step": "DONE", "updates": 8, "eval_loss_first": 0.70,
                                                    "eval_loss_last": 0.21, "tokens_per_s": 5500, "peak_mem_gb": 21.6})
    store.write_json(f"{pre}/train/job.json", {"machine": "a3-highgpu-1g", "scheduling": "dry", "region": "us-central1",
                                               "wall_h": 0.0, "preemptions": 0})
    store.write_bytes(f"{pre}/train/DONE", b"dry\n")
    store.write_json(f"{pre}/model/image.json", {"tag": f"c-{run_id}", "digest": "sha256:dry-candidate",
                                                 "image": f"…/sous-chef-model/model:c-{run_id}"})
    from .. import chatfmt
    from .common import trained_family
    fam = trained_family(store, run_id)
    if fam != chatfmt.DEFAULT:  # the family's staged Modelfile, as pipeline-train.sh's stage step leaves it
        from .. import LAB_DIR
        from ..export import student_modelfile
        f = chatfmt.get(fam)
        text = student_modelfile((LAB_DIR / f.base_modelfile).read_text(encoding="utf-8"), f"{run_id}-q4_k_m.gguf",
                                 f.base_ollama_tag, 65536)
        keep = [ln for ln in text.splitlines() if not ln.startswith(("PARAMETER temperature", "PARAMETER presence_penalty",
                                                                      "PARAMETER num_ctx"))]
        keep += [f"PARAMETER temperature {chatfmt.stage_temperature(fam)}", "PARAMETER presence_penalty 0",
                 "PARAMETER num_ctx 65536"]
        store.write_bytes(f"{pre}/train/stage/Modelfile", ("\n".join(keep) + "\n").encode("utf-8"))


def main(work: Path, *, smoke: bool = False, engine_dir: str | None = None, base_family: str | None = None) -> int:
    from .. import pipeline
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    engine_dir = engine_dir or os.environ.get("COOKIE_CUTTER_DIR")
    if not engine_dir:
        print("retrain-dry: needs the engine: --engine-dir or COOKIE_CUTTER_DIR")
        return 2
    bucket, run_id = seed(work, smoke=smoke, base_family=base_family)
    store = open_store(bucket)
    mode = "smoke" if smoke else "full"
    common = ["--run-id", run_id, "--bucket", bucket, "--mode", mode, "--workdir", str(work / "jobs"),
              "--engine-dir", engine_dir, "--dry"]
    steps = [["prepare", *common, "--workers", "2"], ["collect", *common, "--workers", "4"],
             ["dataset", *common, "--parity", "auto", "--no-caps"], None, ["eval", *common, "--workers", "2"]]
    for argv in steps:
        if argv is None:
            stub_training(store, run_id)
            continue
        rc = pipeline.main(argv)
        res = store.read_json(f"runs/{run_id}/result-{argv[0]}.json") or {}
        print(f"retrain-dry: {argv[0]} exit {rc} status {res.get('status')}")
        if rc != 0:
            print(json.dumps(res.get("error") or res.get("notification") or {}, indent=1)[:2000])
            return rc
    dec = store.read_json(f"runs/{run_id}/decision.json") or {}
    res = dec.get("results") or {}
    print(json.dumps({"run_id": run_id, "bucket": bucket, "bar_pass": dec.get("bar_pass"),
                      "rules": [(r["name"], r["pass"]) for r in (res.get("bar") or {}).get("rules") or []],
                      "sets": [(s["set"], s["n"], s["current"], s["candidate"]) for s in res.get("sets") or []],
                      "cases": res.get("cases"),
                      "authored": len(read_jsonl(Path(bucket) / (f"runs/{run_id}/prepare/authored-smoke.jsonl" if smoke
                                                                 else f"cases/pool/authored/{run_id}.jsonl")))}))
    return 0
