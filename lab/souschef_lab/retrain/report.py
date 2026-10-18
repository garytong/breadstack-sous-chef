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
training, cases, cost, stale_round, links} plus, when measured (r7), `r4_reference`
(bar.add_reference: the candidate against r4's recorded answers) and `clear_asks`
(clear_asks: how often each arm answers a CLEAR request with a question — report
only, the number to read first). A stale round (the deployed manifest moved with an
R3 change during the round) recommends Skip. report.json `warnings` lists what was
not measured (a missing r4 reference).
"""
from __future__ import annotations

import os

from datetime import datetime, timedelta, timezone
from typing import Any, Callable

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


def training_arms(prog: dict) -> dict | None:
    """A several-arm round's training arms (r6: from-base seeds and a warm start, train.py select), from the
    trainer's progress.json: every arm's validation scores, the one kept and the planned arms not trained. None for
    a one-arm round."""
    arms = prog.get("seeds") if isinstance(prog, dict) else None
    if not isinstance(arms, list) or not arms:
        return None
    sel = prog.get("selection") if isinstance(prog.get("selection"), dict) else {}
    return {"selected": sel.get("selected_arm") or prog.get("arm"), "rule": sel.get("rule"), "by": sel.get("by"),
            "arms": [{k: a.get(k) for k in ("arm", "start", "seed", "lr", "eval_loss", "gen_match", "gen_n", "selected")}
                     for a in arms if isinstance(a, dict)],
            "skipped": sel.get("skipped") or []}


def arms_text(arms: dict | None) -> str:
    """One line for the decision message: each arm's final validation loss and generation check, the kept one."""
    if not arms or len(arms.get("arms") or []) < 2 and not arms.get("skipped"):
        return ""
    parts = []
    for a in arms["arms"]:
        loss = a.get("eval_loss")
        gen = f", gen {a['gen_match']}/{a['gen_n']}" if a.get("gen_match") is not None else ""
        parts.append(f"{a.get('arm')} loss {loss:.4f}{gen}" if isinstance(loss, (int, float)) else f"{a.get('arm')} loss ?{gen}")
    line = f"Training arms (same validation rows): {'; '.join(parts)}. Kept {arms.get('selected')}."
    if arms.get("skipped"):
        line += " Not trained: " + "; ".join(f"{k.get('arm')} ({k.get('reason')})" for k in arms["skipped"]) + "."
    return line


def _training(store: Store, pre: str, ds: dict) -> dict:
    prog = store.read_json(f"{pre}/train/progress.json") or {}
    job = store.read_json(f"{pre}/train/job.json") or {}
    m = ds.get("metrics") or {}
    return {"rows_train": m.get("train"), "rows_val": m.get("val"), "followups": m.get("followups"),
            "updates": m.get("updates") or prog.get("updates"), "eval_loss_first": prog.get("eval_loss_first"),
            "eval_loss_last": prog.get("eval_loss_last"), "machine": job.get("machine"), "scheduling": job.get("scheduling"),
            "region": job.get("region"), "wall_h": job.get("wall_h") or prog.get("wall_h"),
            "preemptions": job.get("preemptions", 0), "peak_mem_gb": prog.get("peak_mem_gb"),
            "tokens_per_s": prog.get("tokens_per_s"), "arms": training_arms(prog)}


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
        **_measured_extras(bar, eval_info),
    }


# bar.add_reference's summaries: r4's recorded answers (rule r4_floor) and r4 re-run in r6's round (r4_rerun_floor,
# report only unless thresholds r4_reference.rerun_bar)
REFERENCE_KEYS = ("r4_reference", "r4_rerun_reference")


def _measured_extras(bar: dict, eval_info: dict) -> dict:
    """results.r4_reference (bar.add_reference's summary, its source and sha256) and results.clear_asks, only when
    the eval measured or configured them (an older eval's results keep their exact shape)."""
    out: dict[str, Any] = {}
    for rkey in REFERENCE_KEYS:
        ref = bar.get(rkey)
        if not isinstance(ref, dict):
            continue
        src = ref.get("source") or {}
        out[rkey] = {
            "measured": bool(ref.get("measured")), "detail": ref.get("detail"),
            "source": src.get("source"), "sha256": src.get("sha256"), "copies": src.get("copies") or [],
            "copies_match": src.get("copies_match"), "tag": src.get("tag") or ref.get("name"),
            "digest": src.get("digest"), "sets": ref.get("sets"), "gate": ref.get("gate", src.get("gate")),
            "pooled": ref.get("pooled"), "per_set": ref.get("per_set"),
            "excluded_amended": ref.get("excluded_amended", 0), "excluded_stale": ref.get("excluded_stale", 0),
            "excluded_unrecorded": ref.get("excluded_unrecorded", 0)}
        if src.get("label"):
            out[rkey]["label"] = src["label"]
        if src.get("rows_run"):
            out[rkey]["rows_run"] = src["rows_run"]
    ca = eval_info.get("clear_asks")
    if isinstance(ca, dict):  # the per-row keys stay in report.json's `eval` (the decision payload rides the workflow)
        out["clear_asks"] = {**ca, "arms": {a: {**v, "by_class": {c: {k: x for k, x in d.items() if k != "rows"}
                                                                   for c, d in (v.get("by_class") or {}).items()}}
                                            for a, v in (ca.get("arms") or {}).items()}}
    return out


def warnings_of(bar: dict, eval_info: dict) -> list[str]:
    """What the eval could not measure, for report.json `warnings` (a person reads it before deciding)."""
    w: list[str] = []
    ref = bar.get("r4_reference")
    if isinstance(ref, dict):
        name = ref.get("name") or "r4"
        for x in (ref.get("source") or {}).get("warnings") or []:
            w.append(f"{name}_floor: {x}")
        if not ref.get("measured") and not (ref.get("source") or {}).get("warnings"):
            w.append(f"{name}_floor: {ref.get('detail') or 'NOT MEASURED'}")
    ca = eval_info.get("clear_asks")
    if isinstance(ca, dict) and not ca.get("measured"):
        w.append(f"clear_asks: not measured ({ca.get('why')})")
    return w


# --- "asks on clear requests" (r7 spec 6.3; report-only, never in the bar) --------------------------------------------

CLEAR_ASK_SETS = ("paraphrase", "holdout", "real")


def asked_instead(t: dict, ask_ok: bool) -> bool:
    """A trimmed eval row whose turn asked (or only answered) instead of writing: no_write with reason
    asked_or_answered, or clarify_ok on a row whose key does not allow asking."""
    v = t.get("verdict")
    if t.get("status") not in (None, "ok"):
        return False
    if v == "no_write" and "asked_or_answered" in (t.get("reasons") or []):
        return True
    return v == "clarify_ok" and not ask_ok


def askrule_vague(m: Any = None) -> Callable[[dict], bool] | None:
    """askrule's verdict on an exam row's request: classify_case (the row's whole manifest vocabulary: options,
    toggles, section types, the texts the site shows) when a manifest is given, else classify on the text alone;
    None when the lab has no askrule module."""
    try:
        from .. import askrule as AR
    except Exception:  # noqa: BLE001 — report-only: a missing or broken askrule leaves the count unmeasured
        return None
    by_case = getattr(AR, "classify_case", None)

    def vague(row: dict) -> bool:
        v = None
        if by_case is not None and m is not None:
            try:
                v = by_case(row, m)
            except Exception:  # noqa: BLE001 — the vocabulary is a refinement; the text rule still applies
                v = None
        if v is None:
            v = AR.classify(str(row.get("request") or ""))
        return bool(v.get("vague") if isinstance(v, dict) else getattr(v, "vague", v))
    return vague


def clear_asks(arms_rows: dict[str, list[dict]], exam_rows: list[dict], *, vague: Callable[[dict], bool] | None = None,
               sets: tuple[str, ...] = CLEAR_ASK_SETS) -> dict:
    """Per arm: of the row replicates in `sets` (trimmed eval rows, set exactly "<set>", every replicate counted)
    whose request the ask rule calls NOT vague, how many the arm answered with a question instead of a write
    (asked_instead), split by the exam class (relative, synonym, unnamed, selected, …). `exam_rows` = the scored
    rows (request, meta.class, expected.ask_ok), found by (set, id) or, for a row scored before an amendment, by
    meta.superseded.id. {"measured": False, ...} without a classifier."""
    if vague is None:
        return {"measured": False, "why": "the lab has no askrule module", "arms": {}}
    vague = _guarded(vague)
    by: dict[tuple[str, str], dict] = {}
    for r in exam_rows:
        s = str(r.get("_origin_set") or r.get("set") or "")
        by[(s, str(r.get("id")))] = r
        sup = ((r.get("meta") or {}).get("superseded") or {}).get("id")
        if sup:
            by.setdefault((s, str(sup)), r)
    cache: dict[int, bool] = {}
    out: dict[str, Any] = {"measured": True, "sets": list(sets), "arms": {}}
    for arm, rows in arms_rows.items():
        if not rows:
            continue
        per: dict[str, dict[str, Any]] = {}
        unmatched = 0
        for t in rows:
            s = str(t.get("set") or "")
            if s not in sets or t.get("status") not in (None, "ok"):
                continue
            ex = by.get((s, str(t.get("id"))))
            if ex is None:
                unmatched += 1
                continue
            if id(ex) not in cache:
                cache[id(ex)] = bool(vague(ex))
            if cache[id(ex)]:
                continue
            cls = str((ex.get("meta") or {}).get("class") or "other")
            ask_ok = bool((ex.get("expected") or {}).get("ask_ok"))
            for k in (cls, "all"):
                d = per.setdefault(k, {"asked": 0, "n": 0, "rows": []})
                d["n"] += 1
                if asked_instead(t, ask_ok):
                    d["asked"] += 1
                    if k == "all" and t.get("key") not in d["rows"]:
                        d["rows"].append(t.get("key") or f"{s}:{t.get('id')}")
        for d in per.values():
            d["rows"] = sorted(d["rows"])
            if not d["rows"]:
                d.pop("rows")
        out["arms"][arm] = {"by_class": per, "unmatched": unmatched}
    return out


def _guarded(vague: Callable[[dict], bool]) -> Callable[[dict], bool]:
    """A classifier error on one row counts that row as vague (left out), never fails the eval."""
    def f(row: dict) -> bool:
        try:
            return bool(vague(row))
        except Exception:  # noqa: BLE001
            return True
    return f


def clear_asks_text(ca: dict | None) -> str:
    """'Asks on clear requests: candidate relative 3/48, all 9/400; current …'."""
    if not ca or not ca.get("measured") or not ca.get("arms"):
        return ""
    parts = []
    for arm, a in ca["arms"].items():
        bc = a.get("by_class") or {}
        bits = [f"{k} {bc[k]['asked']}/{bc[k]['n']}" for k in ("relative", "all") if k in bc]
        if bits:
            parts.append(f"{arm} " + ", ".join(bits))
    return ("Asks on clear requests: " + "; ".join(parts) + ".") if parts else ""


def summary_text(res: dict, *, smoke: bool, expires: datetime) -> str:
    """What the thermometer's post does not already say: it renders the headline (candidate vs live, manifest,
    bar), the table from results.sets, the stale-round and SMOKE notes and the expiry itself. This is the
    failing bar rules, or a one-line all-clear; the full text for the playbook stays in report.json."""
    failed = [r for r in res["bar"]["rules"] if not r["pass"]]
    arms = arms_text((res.get("training") or {}).get("arms"))
    if failed:
        text = "Bar rules failing: " + "; ".join(f"{r['name']}: {r['detail']}" for r in failed)
    else:
        text = f"Every bar rule passes ({len(res['bar']['rules'])})."
    # the r4 line once: a failing r4_floor's detail already carries it
    ref = "" if any(r["name"] == "r4_floor" for r in failed) else BAR.reference_text(res.get("r4_reference"))
    rerun = "" if any(r["name"] == "r4_rerun_floor" for r in failed) else BAR.reference_text(res.get("r4_rerun_reference"))
    for extra in (ref, rerun, clear_asks_text(res.get("clear_asks")), arms):
        if extra:
            text += " " + extra
    return text


def summary_full(res: dict, *, smoke: bool, expires: datetime) -> str:
    """The whole message as plain text (report.json; a person reading the bucket)."""
    bar_word = "PASS" if res["bar"]["pass"] else "FAIL"
    lines = [f"`{res['candidate']['id']}` vs live `{res['current']['tag']}`, on manifest `{h8(res['deployed_manifest']['hash'])}`"
             f" · bar {bar_word}" + (" · SMOKE (a dry rollout on Approve)" if smoke else "")]
    lines.append(BAR.table_text({"sets": res["sets"]}))
    lines.append(summary_text(res, smoke=smoke, expires=expires))
    ca = res.get("clear_asks")
    if ca and ca.get("measured") and ca.get("arms"):
        for arm, a in ca["arms"].items():
            bc = a.get("by_class") or {}
            lines.append(f"  asks on clear requests, {arm}: " +
                         ", ".join(f"{k} {v['asked']}/{v['n']}" for k, v in sorted(bc.items(), key=lambda kv: kv[0] == "all")))
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
              "summary": summary_full(res, smoke=smoke, expires=expires), "warnings": warnings_of(bar, eval_info),
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
