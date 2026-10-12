"""report — the round's report, the decision payload and the candidate_ready
notification (design §3.8 "Report", §4.4 `results`).

Reads what the stages left in runs/<id>/ (run.json, result-*.json, prepare and
dataset reports, train/progress.json and train/job.json, model/image.json,
eval/bar.json and eval/info.json), state/model.json, the drift run's static.json
and the pool index; writes

  runs/<id>/report.json      everything below plus the per-stage results
  runs/<id>/decision.json    {expires_at, smoke, bar_pass, results} — the workflow puts it on
                             the candidate_ready notification as `decision` (§4.1)
  result-eval.json           status ok, the candidate_ready notification (mention added by the
                             thermometer) and the decision's candidate_id

`results` = {candidate, current, deployed_manifest, trigger, bar, sets, static,
training, cases, cost, stale_round, links}. A stale round (the deployed manifest
moved with an R3 change during the round) recommends Skip.
"""
from __future__ import annotations

import os

from datetime import datetime, timedelta, timezone
from typing import Any

from . import bar as BAR
from .common import Store, load_run, model_state, notification, run_prefix, write_result

DECISION_TTL_DAYS = 14
SMOKE_TTL_DAYS = 1
THERMOMETER = "https://thermometer.breadstackcrm.com"


def h8(h: Any) -> str:
    return str(h or "")[:8]


def candidate_id(run_id: str) -> str:
    """c-<run id>, or the workflow's own id for this run (PIPELINE_CANDIDATE_ID: SMOKE's is c-smoke-<run id>)."""
    env = os.environ.get("PIPELINE_CANDIDATE_ID") or ""
    if env and run_id in env:
        return env
    return f"c-{run_id}"


def _training(store: Store, pre: str, ds: dict) -> dict:
    prog = store.read_json(f"{pre}/train/progress.json") or {}
    job = store.read_json(f"{pre}/train/job.json") or {}
    m = ds.get("metrics") or {}
    return {"rows_train": m.get("train"), "rows_val": m.get("val"), "followups": m.get("followups"),
            "updates": m.get("updates") or prog.get("updates"), "eval_loss_first": prog.get("eval_loss_first"),
            "eval_loss_last": prog.get("eval_loss_last"), "machine": job.get("machine"), "scheduling": job.get("scheduling"),
            "region": job.get("region"), "wall_h": job.get("wall_h") or prog.get("wall_h"),
            "preemptions": job.get("preemptions", 0), "peak_mem_gb": prog.get("peak_mem_gb"),
            "tokens_per_s": prog.get("tokens_per_s")}


def _cases(store: Store, pre: str, prep: dict) -> dict:
    rep = store.read_json(f"{pre}/prepare/report.json") or {}
    idx = store.read_json("cases/pool/index.json") or {}
    authored = [f for f in idx.get("files") or [] if str(f.get("path", "")).startswith("cases/pool/authored/")]
    m = prep.get("metrics") or {}
    return {"pool_total": (rep.get("rebase") or {}).get("totals", {}).get("in"), "kept": m.get("cases"),
            "authored_this_round": m.get("authored_trainable"), "authored_holdout": m.get("authored_holdout"),
            "authored_accumulated": sum(sum((f.get("family_counts") or {}).values()) for f in authored
                                        if not f.get("trained_in")),
            "rebased_dropped": (rep.get("rebase") or {}).get("dropped_by_reason"), "targets": m.get("targets")}


def results_of(store: Store, run_id: str, *, bar: dict, eval_info: dict) -> dict:
    pre = run_prefix(run_id)
    run = load_run(store, run_id)
    model = model_state(store)
    img = store.read_json(f"{pre}/model/image.json") or {}
    prep = store.read_json(f"{pre}/result-prepare.json") or {}
    col = store.read_json(f"{pre}/result-collect.json") or {}
    ds = store.read_json(f"{pre}/result-dataset.json") or {}
    prep_rep = store.read_json(f"{pre}/prepare/report.json") or {}
    dep = run.get("deployed") or {}
    drift_run = (run.get("trigger") or {}).get("drift_run")
    static = (store.read_json(f"drift/{drift_run}/static.json") if drift_run else None) or {}
    counts = static.get("counts") or {}
    impact = static.get("impact") or {}
    training = _training(store, pre, ds)
    gpu_h = training.get("wall_h")
    deployed_hash = (prep_rep.get("manifest") or dep.get("hash"))
    gf = dep.get("generated_from") if isinstance(dep.get("generated_from"), dict) else {}
    return {
        "candidate": {"id": candidate_id(run_id), "tag": img.get("tag") or f"c-{run_id}", "digest": img.get("digest"),
                      "run_id": run_id, "trained_on": deployed_hash},
        "current": {"tag": model.get("tag"), "digest": model.get("digest"), "trained_on": model.get("training_manifest")},
        "deployed_manifest": {"hash": deployed_hash, "launcher_build": dep.get("launcher_build"),
                              "engine_sha": gf.get("sha"), "engine_status": (prep_rep.get("engine") or {}).get("status")},
        "trigger": {"drift_run": drift_run, "verdict_rules": eval_info.get("trigger_rules") or [],
                    "summary": (run.get("trigger") or {}).get("reason")},
        "bar": {"pass": bar["pass"], "rules": bar["rules"]},
        "sets": bar["sets"],
        "static": {"R1": counts.get("R1"), "R2": counts.get("R2"), "R3": counts.get("R3"),
                   "new_keys": impact.get("new_keys") or [], "stale_exam_rows": len(eval_info.get("stale_rows") or [])},
        "training": training,
        "cases": _cases(store, pre, prep),
        "cost": {"anthropic_authoring_usd": (prep.get("metrics") or {}).get("anthropic_usd"),
                 "teacher_usd": (col.get("metrics") or {}).get("teacher_usd"),
                 "gpu_hours": gpu_h},
        "stale_round": bool(eval_info.get("stale_round")),
        "links": {"report": store.url(f"{pre}/report.json"), "execution": run.get("execution_url") or "",
                  "page": f"{THERMOMETER}/sous-chef/decisions/{candidate_id(run_id)}"},
    }


def summary_text(res: dict, *, smoke: bool, expires: datetime) -> str:
    """What the thermometer's post does not already say: it renders the headline (candidate vs live, manifest,
    bar), the table from results.sets, the stale-round and SMOKE notes and the expiry itself. This is the
    failing bar rules, or a one-line all-clear; the full text for the playbook stays in report.json."""
    failed = [r for r in res["bar"]["rules"] if not r["pass"]]
    if failed:
        return "Bar rules failing: " + "; ".join(f"{r['name']}: {r['detail']}" for r in failed)
    return f"Every bar rule passes ({len(res['bar']['rules'])})."


def summary_full(res: dict, *, smoke: bool, expires: datetime) -> str:
    """The whole message as plain text (report.json; a person reading the bucket)."""
    bar_word = "PASS" if res["bar"]["pass"] else "FAIL"
    lines = [f"`{res['candidate']['id']}` vs live `{res['current']['tag']}`, on manifest `{h8(res['deployed_manifest']['hash'])}`"
             f" · bar {bar_word}" + (" · SMOKE (a dry rollout on Approve)" if smoke else "")]
    lines.append(BAR.table_text({"sets": res["sets"]}))
    lines.append(summary_text(res, smoke=smoke, expires=expires))
    if res["stale_round"]:
        lines.append("The launcher's manifest changed again during this round (a hard change): Skip is recommended.")
    lines.append("Approve rolls it out to int now. Skip parks it and batches its cases into the next big drift. "
                 f"Expires {expires:%b %d}.")
    return "\n".join(lines)


def build(store: Store, run_id: str, *, mode: str, bar: dict, eval_info: dict, now: datetime | None = None) -> dict:
    pre = run_prefix(run_id)
    smoke = mode == "smoke"
    now = now or datetime.now(timezone.utc)
    expires = now + timedelta(days=SMOKE_TTL_DAYS if smoke else DECISION_TTL_DAYS)
    res = results_of(store, run_id, bar=bar, eval_info=eval_info)
    decision = {"expires_at": expires.isoformat(timespec="seconds").replace("+00:00", "Z"), "smoke": smoke,
                "bar_pass": bool(bar["pass"]), "results": res}
    store.write_json(f"{pre}/decision.json", decision)
    report = {"run_id": run_id, "mode": mode, "candidate_id": res["candidate"]["id"], "results": res, "eval": eval_info,
              "summary": summary_full(res, smoke=smoke, expires=expires),
              "stages": {s: store.read_json(f"{pre}/result-{s}.json") for s in ("prepare", "collect", "dataset")},
              "at": now.isoformat(timespec="seconds").replace("+00:00", "Z")}
    store.write_json(f"{pre}/report.json", report)
    title = ("Sous Chef candidate ready: decide or skip" + (" (SMOKE)" if smoke else ""))
    note = notification("candidate_ready", "info", title, summary_text(res, smoke=smoke, expires=expires),
                        {"candidate_id": res["candidate"]["id"], "bar_pass": bool(bar["pass"]), "smoke": smoke,
                         "stale_round": res["stale_round"], "report": res["links"]["report"]})
    return write_result(store, run_id, "eval", "ok",
                        metrics={"bar_pass": bool(bar["pass"]), "stale_round": res["stale_round"],
                                 "sets": {r["set"]: [r["current"], r["candidate"], r["n"]] for r in bar["sets"]}},
                        artifacts=[f"{pre}/report.json", f"{pre}/decision.json", f"{pre}/eval/candidate-baseline.json"],
                        notification_=note, extra={"candidate_id": res["candidate"]["id"], "decision": f"{pre}/decision.json"})
