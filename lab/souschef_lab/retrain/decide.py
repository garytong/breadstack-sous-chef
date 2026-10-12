"""decide — the bucket's side of a decision (design §3.9–3.10, §1.5, §3.2 step 3):
`python -m souschef_lab.pipeline decision --run-id rt-… --outcome approved|skipped`.

The workflow (or a person running the playbook) calls it AFTER the GPU service
rollout of an Approve, or on a Skip / an expiry. Bucket writes only:

  approved  baselines/<digest>/baseline.json   ← runs/<id>/eval/candidate-baseline.json (digest set to the
                                                 rolled-out image's, calibrated by its 2 replicates)
            cases/trained/<candidate>/         the round's case file (cases.jsonl) and its dataset (dataset/…),
                                                 copied out of runs/ (which expires after 180 days): drift's
                                                 stale-training-rows rule and the playbook read them for as long
                                                 as the model is live, or in state/model.json history
            state/model.json                   {tag, digest, training_manifest, trained_at, run_id, dataset,
                                                 training_cases, history: [the previous model, …]}
            cases/pool/index.json              every authored file this round trained on: trained_in = <candidate>
            state/ack.json                     deleted (a new model: drift is measured from its own baseline)
  skipped   state/ack.json                     the drift run's ack-material.json + {at, candidate_id}: rules (a)–(f)
                                                 fire only on drift BEYOND what was skipped
            state/parked/<candidate>.json      the candidate (image, report) for a later look
            (the round's authored cases stay in the pool, trained_in null: the next big drift batches them)
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .common import ConfigError, Store, load_run, model_state, run_prefix, utc_now
from .report import candidate_id

POOL_INDEX = "cases/pool/index.json"


def trained_files(store: Store, run_id: str) -> list[str]:
    """The authored pool files the round's case file was built from."""
    rep = store.read_json(f"{run_prefix(run_id)}/prepare/report.json") or {}
    paths = [f["path"] for f in (rep.get("pool") or {}).get("files") or [] if str(f.get("path", "")).startswith("cases/pool/authored/")]
    own = f"cases/pool/authored/{run_id}.jsonl"
    if store.exists(own) and own not in paths:
        paths.append(own)
    return paths


TRAINED = "cases/trained"


def keep_training_inputs(store: Store, run_id: str, cid: str) -> tuple[str, str | None]:
    """Copy the round's case file and dataset to cases/trained/<cid>/ (never expires). Returns their keys."""
    pre = run_prefix(run_id)
    dst = f"{TRAINED}/{cid}"
    cases = f"{dst}/cases.jsonl"
    with tempfile.TemporaryDirectory(prefix="keep-") as tmp:
        local = Path(tmp) / "cases.jsonl"
        if not store.exists(cases):
            if not store.download(f"{pre}/prepare/cases.jsonl", local):
                raise ConfigError(f"{pre}/prepare/cases.jsonl is missing: the approved model's training cases cannot be kept")
            store.upload(local, cases)
        ds_src, ds_dst = f"{pre}/dataset/", f"{dst}/dataset"
        if store.list(ds_src):
            store.sync_down(ds_src, Path(tmp) / "dataset")
            store.sync_up(Path(tmp) / "dataset", ds_dst)
            return cases, ds_dst
    return cases, None


def approved(store: Store, run_id: str, *, digest: str | None = None, by: str | None = None) -> dict:
    pre = run_prefix(run_id)
    cid = candidate_id(run_id)
    img = store.read_json(f"{pre}/model/image.json") or {}
    digest = digest or img.get("digest")
    if not digest or not str(digest).startswith("sha256:"):
        raise ConfigError("approved: the rolled-out image digest (sha256:…) is needed (--digest or model/image.json)")
    cb = store.read_json(f"{pre}/eval/candidate-baseline.json")
    if not cb:
        raise ConfigError(f"{pre}/eval/candidate-baseline.json is missing (the eval stage writes it)")
    cb.update(digest=digest, tag=img.get("tag") or cb.get("tag"), source="candidate-eval",
              calibrated=len(cb.get("replicates") or []) >= 2, promoted_at=utc_now())
    cases_key, dataset_key = keep_training_inputs(store, run_id, cid)
    store.write_json(f"baselines/{digest}/baseline.json", cb)
    prev = model_state(store)
    rep = store.read_json(f"{pre}/prepare/report.json") or {}
    history = list(prev.get("history") or [])
    history.append({k: prev.get(k) for k in ("tag", "digest", "training_manifest", "trained_at", "run_id", "training_cases", "dataset")
                    if k in prev or k in ("tag", "digest", "training_manifest", "trained_at", "run_id")})
    model = {"tag": img.get("tag") or f"c-{run_id}", "digest": digest, "training_manifest": rep.get("manifest") or cb.get("training_manifest"),
             "trained_at": utc_now(), "run_id": run_id, "candidate_id": cid, "dataset": dataset_key or f"{pre}/dataset",
             "approved_by": by, "history": history[-20:],
             # what drift's stale-training-rows rule checks: this round's case file, every case on its manifest —
             # the copy under cases/trained/, which outlives runs/ (180-day lifecycle)
             "training_cases": [cases_key]}
    store.write_json("state/model.json", model)
    idx = store.read_json(POOL_INDEX) or {"files": []}
    marked = []
    files = trained_files(store, run_id)
    for f in idx.get("files") or []:
        if f.get("path") in files and not f.get("trained_in"):
            f["trained_in"] = cid
            marked.append(f["path"])
    store.write_json(POOL_INDEX, idx)
    store.delete("state/ack.json")
    return {"outcome": "approved", "candidate_id": cid, "digest": digest, "baseline": f"baselines/{digest}/baseline.json",
            "pool_marked": marked, "model": {k: model[k] for k in ("tag", "digest", "training_manifest")}}


def skipped(store: Store, run_id: str, *, reason: str = "skipped", by: str | None = None) -> dict:
    pre = run_prefix(run_id)
    cid = candidate_id(run_id)
    run = load_run(store, run_id)
    drift_run = (run.get("trigger") or {}).get("drift_run")
    material: dict[str, Any] = (store.read_json(f"drift/{drift_run}/ack-material.json") if drift_run else None) or {}
    if not material:
        # a round started by hand (no drift run): the newest check's material,
        # so the Skip acknowledges what drift last measured, not only the manifest
        material = store.read_json("state/last-ack-material.json") or {}
    if not material:
        dep = run.get("deployed") or {}
        material = {"deployed_hash": dep.get("hash"), "drift_run": drift_run}
    ack = {**material, "at": utc_now(), "candidate_id": cid, "run_id": run_id, "reason": reason, "decided_by": by}
    store.write_json("state/ack.json", ack)
    store.write_json(f"state/parked/{cid}.json", {"candidate_id": cid, "run_id": run_id, "reason": reason, "at": ack["at"],
                                                  "decided_by": by,
                                                  "image": store.read_json(f"{pre}/model/image.json"),
                                                  "report": f"{pre}/report.json"})
    authored = store.read_jsonl(f"cases/pool/authored/{run_id}.jsonl")
    return {"outcome": "skipped", "candidate_id": cid, "ack": "state/ack.json", "parked": f"state/parked/{cid}.json",
            "authored_kept": len(authored)}
