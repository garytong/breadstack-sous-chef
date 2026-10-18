"""bar — the shipping bar of the paired eval (design §3.8). Advisory: Approve stays
available on a failing bar, behind a typed confirmation on the thermometer.

Inputs: per arm (`current`, `candidate`) and per eval set, the replicates as
{row key: 0|1} (row key = "<set>:<row id>", the baseline's addressing), the
per-row ghost-write counts and out-of-scope `where` flags, the current model's
baseline (σ) and the thresholds. Per set:

  tol_s = max(1 row, ⌈2 σ_Δ,s⌉),   σ_Δ,s = σ_run,s · sqrt(1/m_current + 1/m_candidate)
          or, with thresholds sigma_seed[s] (retrain/noise.py's measured retrain spread of ONE model, rows):
  tol_s = max(1 row, ⌈2 sqrt(σ_Δ,s² + 2 σ_seed,s²)⌉)
          (both the candidate and the live model are retrain draws, so Var(Δ) holds σ_seed² twice)

with σ_run,s from the CURRENT model's baseline over the set's rows
(baseline.sigma_run: the row pass rates, the observed SD, or the prior flip rate).
The candidate passes when ALL hold:

  1 sets       every bar set: candidate − current ≥ −tol_s (rows, replicate means)
  2 mcnemar    the pooled paired McNemar over the bar sets (pairs (row, k)) is not
               significantly worse (fail: the candidate loses with p < 0.05)
  3 oos_where  out_of_scope: every row but OOS_MISSES (2) with `where` (on the candidate; r5's rule was 11 of
               13) — over the set's rows that are still out of scope (`oos_keys`). Since 2026-10-15 p137.k2 is a
               create_page row that never sends a `where`: eval_stage moves it to the report-only `capability` set,
               so the share is over 12 rows and 10 are needed — the same two misses r5 was allowed
  4 f0h_unseen F0H :unseen — candidate ≥ current − tol
  5 trigger    dodged keys → ≥ 75 % of their probes pass AND more than current;
               exam drift → the real-phrasing total improves by ≥ σ_Δ,total;
               any other trigger → not applicable (passes)
  6 ghosts     ghost writes (refused writes to keys the manifest lacks) ≤ current
  7 r4_floor   (add_reference, r7) the candidate against a RECORDED reference model — r4's calibration
               replicates (thresholds `r4_reference`) — on the same machinery: per listed set
               Δ_vs_r4 ≥ −tol_s (σ_run from r4's own replicates, m = [3, 2], and 1·sigma_seed[set]² when set:
               r4 is one fixed recorded model, only the candidate is a seed draw) and
               the pooled McNemar over those sets not significantly worse. Without it the bar only compares
               with the live model, which may itself have lost rows (r6: −6.5 holdout to over-asking).
               Amended rows (`.k2` / `.k3`: scored under another key than the recorded one) and rows stale
               on the round's manifest drop out of the pairing and are counted. No reference file: the
               rule passes, "NOT MEASURED", with a warning.
"""
from __future__ import annotations

import math
import re
from typing import Any

from ..baseline import sigma_run
from ..examcmp import mcnemar_p

EXAM_SETS = ("paraphrase", "holdout", "real", "followup", "out_of_scope", "out_of_scope_holdout")
OOS_SET = "out_of_scope"
OOS_MISSES = 2  # out_of_scope rows the candidate may answer without `where` (r5: 11 of 13 needed)
REPORT_ONLY = ("capability",)  # eval_stage's capability rows (exam rows a tool round changed the answer of): never bar
PROBE_MIN = 0.75
F9H_MIN_N = 10
EXAM_DRIFT_RULES = ("a", "b", "c")
DODGED_RULES = ("d",)


def _mean(reps: list[dict[str, int]], keys: list[str]) -> float:
    return sum(sum(int(r.get(k, 0)) for k in keys) for r in reps) / len(reps) if reps else 0.0


def common(cur: list[dict[str, int]], cand: list[dict[str, int]]) -> list[str]:
    """Rows every replicate of both arms answered (errors are left out of both)."""
    sets = [set(r) for r in cur + cand]
    return sorted(set.intersection(*sets)) if sets else []


def paired(cur: list[dict[str, int]], cand: list[dict[str, int]], keys: list[str]) -> tuple[int, int]:
    """(b, c): pairs (row, k) the current arm passes and the candidate fails, and the reverse."""
    b = c = 0
    for k in range(min(len(cur), len(cand))):
        for r in keys:
            x, y = int(cur[k].get(r, 0)), int(cand[k].get(r, 0))
            b += int(x == 1 and y == 0)
            c += int(x == 0 and y == 1)
    return b, c


def set_row(name: str, variant: str, cur: list[dict[str, int]], cand: list[dict[str, int]], baseline: dict | None,
            th: dict, *, bar: bool, seed_set: str | None = None, seed_draws: int = 2) -> dict:
    keys = common(cur, cand)
    n = len(keys)
    sig, how = sigma_run(keys, baseline, th) if n else (0.0, "empty")
    sd = sig * math.sqrt(1 / max(len(cur), 1) + 1 / max(len(cand), 1))
    tol = max(1, math.ceil(2 * sd)) if n else 0
    # seed-aware tolerance (docs/RETRAIN.md R6/R10): thresholds `sigma_seed` {set: rows}, measured by
    # retrain/noise.py, adds how much a retrain of the same data moves ONE model's score on the set. The bar compares
    # two trained models (candidate and live), each a seed draw, so Var(Δ) gets 2·σ_seed²:
    # max(1, ⌈2·sqrt(σ_Δ² + 2·σ_seed²)⌉). Against a RECORDED reference (add_reference: r4 is one fixed model, its
    # answers recorded) only the candidate is a draw: `seed_draws` 1, 1·σ_seed² (r7 review — with 2·σ_seed² and
    # σ_seed ≈ 2.5 the holdout tolerance would be 8 and r6's −6.3 would pass; with 1· it is 6 and r6 still fails)
    ss = (th.get("sigma_seed") or {}).get(seed_set or name) if isinstance(th.get("sigma_seed"), dict) else None
    extra: dict[str, Any] = {}
    if n and isinstance(ss, (int, float)) and ss > 0:
        tol = max(1, math.ceil(2 * math.sqrt(sd ** 2 + seed_draws * float(ss) ** 2) - 1e-9))
        extra = {"sigma_seed": round(float(ss), 3), "tol_eval_only": max(1, math.ceil(2 * sd))}
    mc, mk = _mean(cur, keys), _mean(cand, keys)
    b, c = paired(cur, cand, keys)
    return {"set": name, "variant": variant, "n": n, "current": round(mc, 2), "candidate": round(mk, 2),
            "delta": round(mk - mc, 2), "tol": tol, "sigma_delta": round(sd, 3), "sigma_how": how, **extra,
            "b": b, "c": c, "mcnemar_p": round(mcnemar_p(b, c), 4), "bar": bar and n > 0,
            "m": [len(cur), len(cand)], "_keys": keys}


def evaluate(arms: dict[str, dict[str, list[dict[str, int]]]], *, baseline: dict | None, th: dict,
             trigger_rules: list[str] | None = None, ghosts: dict[str, list[int]] | None = None,
             oos_where: dict[str, list[dict[str, int]]] | None = None, dodged_keys: list[str] | None = None,
             probe_by_key: dict[str, dict[str, list[dict[str, int]]]] | None = None,
             oos_keys: list[str] | None = None) -> dict:
    """`arms[arm][set]` = replicates; set names: the six exam sets (rederived, deployed manifest), "<set>:recorded",
    "F0H:unseen", "F0H:seen", "F9H", "probe". `oos_where[arm]` = replicates of {row: 1 when unsupported_ok BY
    `where`}; `oos_keys` = the out_of_scope row keys the `where` share is taken over (exam.is_out_of_scope; None:
    every row of the set). `ghosts[arm]` = per-replicate ghost-write totals over the bar sets. Returns the bar."""
    cur, cand = arms.get("current") or {}, arms.get("candidate") or {}
    names = sorted(set(cur) | set(cand), key=_order)
    sets = []
    for s in names:
        variant = ("recorded" if s.endswith(":recorded") else "built" if s.startswith(("F0H", "F9H", "probe")) or s.endswith(":built")
                   else "rederived")
        base = s.split(":")[0]
        if base in EXAM_SETS and variant == "rederived":
            bar = True
        elif s == "F0H:unseen":
            bar = True
        elif s == "F9H":
            bar = len(common(cur.get(s) or [], cand.get(s) or [])) >= F9H_MIN_N
        elif s == "probe":
            bar = bool(dodged_keys)
        elif base in REPORT_ONLY:
            bar = False
        else:
            bar = False
        sets.append(set_row(s, variant, cur.get(s) or [], cand.get(s) or [], baseline, th, bar=bar))
    rules: list[dict[str, Any]] = []
    bar_sets = [r for r in sets if r["bar"]]
    worse = [r for r in bar_sets if r["delta"] < -r["tol"]]
    rules.append({"name": "sets", "pass": not worse,
                  "detail": ("every bar set within tolerance" if not worse else
                             "; ".join(f"{r['set']} {r['delta']:+.1f} (tol {r['tol']})" for r in worse))})
    b = sum(r["b"] for r in bar_sets)
    c = sum(r["c"] for r in bar_sets)
    p = mcnemar_p(b, c)
    rules.append({"name": "mcnemar", "pass": not (b > c and p < 0.05),
                  "detail": f"pooled b {b} (current only) / c {c} (candidate only), p {p:.4f}"})
    oo = next((r for r in sets if r["set"] == OOS_SET), None)
    oos = None if oos_keys is None else set(oos_keys)
    keys = [k for k in (oo or {}).get("_keys") or [] if oos is None or k in oos]
    if oo and oo["n"] and oos_where and keys:
        reps = oos_where.get("candidate") or []
        share = _mean(reps, keys) / len(keys) if reps and keys else 0.0
        need = max(1, len(keys) - OOS_MISSES)
        rules.append({"name": "oos_where", "pass": _mean(reps, keys) >= need,
                      "detail": f"{_mean(reps, keys):.1f}/{len(keys)} with `where` (need {need}; {share:.0%})"})
    else:
        rules.append({"name": "oos_where", "pass": False, "detail": "no out_of_scope rows measured"})
    f0 = next((r for r in sets if r["set"] == "F0H:unseen"), None)
    if f0 and f0["n"]:
        rules.append({"name": "f0h_unseen", "pass": f0["delta"] >= -f0["tol"],
                      "detail": f"{f0['candidate']:.1f} vs {f0['current']:.1f} of {f0['n']} (tol {f0['tol']})"})
    else:
        rules.append({"name": "f0h_unseen", "pass": True, "detail": "not measured (no F0H rows)"})
    rules.append(_trigger_rule(sets, trigger_rules or [], cur, cand, baseline, th, dodged_keys or [], probe_by_key or {}))
    if ghosts and ghosts.get("current") and ghosts.get("candidate"):
        gc = sum(ghosts["current"]) / len(ghosts["current"])
        gk = sum(ghosts["candidate"]) / len(ghosts["candidate"])
        rules.append({"name": "ghosts", "pass": gk <= gc, "detail": f"candidate {gk:.1f} vs current {gc:.1f} per run"})
    else:
        rules.append({"name": "ghosts", "pass": True, "detail": "not measured"})
    for r in sets:
        r.pop("_keys", None)
    return {"pass": all(r["pass"] for r in rules), "rules": rules, "sets": sets,
            "pooled": {"b": b, "c": c, "mcnemar_p": round(p, 4)}}


# --- the recorded reference (r7: r4's calibration replicates as a floor) ------------------------------------------------

REFERENCE_SETS = ("paraphrase", "holdout", "real", "followup")
# rows a plain retrain moved a set by with NO data change (r5 = r4's data retrained: −3.5 holdout): the reach of a reroll
# beyond the eval-only tolerance while thresholds sigma_seed is empty (a note on the r4_floor failure, never a pass)
REROLL_ROWS = {"holdout": 3.5}
REFERENCE_VARIANT = "reference"
AMENDED_KEY = re.compile(r"\.k\d+$")  # exam.AMENDED_ID_RE on a row key ("<set>:<id>.k<rev>")


def reference_replicates(ref_baseline: dict | None, sets: list[str] | tuple[str, ...]) -> dict[str, list[dict[str, int]]]:
    """{set: [replicate {row key: 0|1}]} from a baseline.json's replicates (row keys "<set>:<id>")."""
    out: dict[str, list[dict[str, int]]] = {s: [] for s in sets}
    for rep in (ref_baseline or {}).get("replicates") or []:
        rows = rep.get("rows") or {}
        for s in sets:
            out[s].append({k: int(v) for k, v in rows.items() if k.split(":", 1)[0] == s and ":" in k})
    return {s: [r for r in v if r] for s, v in out.items()}


def _exclusions(ref: list[dict[str, int]], cand: list[dict[str, int]]) -> dict[str, list[str]]:
    """Rows of either side that cannot pair: amended rows (the candidate's `.kN` ids, scored on another key than the
    recorded one, and their recorded old ids), rows the reference has but the round does not (stale on the round's
    manifest, moved to a report-only set, errored) and rows the round has that the reference never ran."""
    rk = set().union(*[set(r) for r in ref]) if ref else set()
    ck = set().union(*[set(r) for r in cand]) if cand else set()
    amended = sorted(k for k in ck if AMENDED_KEY.search(k) and k not in rk)  # a reference scored on the same key pairs
    old_ids = {AMENDED_KEY.sub("", k) for k in amended}
    return {"amended": amended, "amended_recorded": sorted(rk & old_ids),
            "stale": sorted(rk - ck - old_ids), "unrecorded": sorted(ck - rk - set(amended))}


def add_reference(bar: dict, cand: dict[str, list[dict[str, int]]], ref: dict[str, list[dict[str, int]]] | None,
                  ref_baseline: dict | None, th: dict, *, name: str = "r4", sets: list[str] | tuple[str, ...] = REFERENCE_SETS,
                  source: dict | None = None, gate: bool = True) -> dict:
    """Add the `<set>:vs-<name>` rows (variant "reference", never `bar`) and the rule `<name>_floor` to an
    evaluated bar, in place, and recompute bar.pass. `cand` = the candidate's replicates per set (the paired eval's
    results_by["candidate"]); `ref` = the reference's (reference_replicates), None when no reference was found
    (`source.missing` says where it was looked for): the rule passes NOT MEASURED. `gate` False (thresholds
    `r4_reference.bar: false`) reports the rule's measurement but never fails the bar."""
    rule_name = f"{name}_floor"
    info: dict[str, Any] = {"name": name, "source": source or {}, "sets": list(sets), "measured": False}
    bar["rules"] = [r for r in bar.get("rules") or [] if r.get("name") != rule_name]
    bar["sets"] = [r for r in bar.get("sets") or [] if r.get("variant") != REFERENCE_VARIANT or
                   not str(r.get("set", "")).endswith(f":vs-{name}")]
    if not ref or not any(ref.get(s) for s in sets):
        where = ", ".join((source or {}).get("looked_in") or []) or "(no path configured)"
        why = (source or {}).get("why") or f"no {name} reference at {where}"
        bar["rules"].append({"name": rule_name, "pass": True, "detail": f"NOT MEASURED: {why}"})
        info["detail"] = f"NOT MEASURED: {why}"
        bar[f"{name}_reference"] = info
        bar["pass"] = all(r["pass"] for r in bar["rules"])
        return bar
    rows: list[dict] = []
    excl: dict[str, dict[str, list[str]]] = {}
    for s in sets:
        r_, c_ = ref.get(s) or [], cand.get(s) or []
        if not r_ or not c_:
            continue
        row = set_row(f"{s}:vs-{name}", REFERENCE_VARIANT, r_, c_, ref_baseline, th, bar=False, seed_set=s, seed_draws=1)
        row.pop("_keys", None)
        ex = _exclusions(r_, c_)
        row["excluded"] = {"amended": len(ex["amended"]), "stale": len(ex["stale"]), "unrecorded": len(ex["unrecorded"])}
        excl[s] = ex
        rows.append(row)
    worse = [r for r in rows if r["n"] and r["delta"] < -r["tol"]]
    b = sum(r["b"] for r in rows)
    c = sum(r["c"] for r in rows)
    p = mcnemar_p(b, c)
    mc_bad = b > c and p < 0.05
    ok = not worse and not mc_bad
    parts = [f"{r['set'].split(':')[0]} {r['delta']:+.1f} (tol {r['tol']})" for r in rows]
    detail = (f"vs {name}'s recorded answers: " + "; ".join(parts) +
              f"; pooled b {b} ({name} only) / c {c} (candidate only), p {p:.4f}")
    if worse:
        detail = "below " + ", ".join(r["set"].split(":")[0] for r in worse) + " — " + detail
        # r7 review: tol here is eval noise only until sigma_seed is measured; a no-change retrain (r5: r4's data
        # again) lost 3.5 holdout rows to the reroll alone. Say so on the failure the reroll could explain — how
        # such a failure is read is pre-registered in docs/RETRAIN.md ("Reading r4_floor")
        reroll = [r for r in worse if "sigma_seed" not in r and r["delta"] >= -(r["tol"] + REROLL_ROWS.get(
            r["set"].split(":")[0], 0.0))]
        if reroll and len(reroll) == len(worse):
            detail += (f"; within a reroll's reach (r5 lost {REROLL_ROWS['holdout']} holdout rows to the seed alone; "
                       "sigma_seed not measured) — read as RETRAIN.md 'Reading r4_floor' says")
    elif mc_bad:
        detail = "pooled McNemar significantly worse — " + detail
    if not gate:
        detail = f"REPORT ONLY ({'would fail' if not ok else 'passes'}): {detail}"
        ok = True
    bar["rules"].append({"name": rule_name, "pass": ok, "detail": detail})
    bar["sets"].extend(rows)
    info.update({"measured": True, "gate": gate, "pooled": {"b": b, "c": c, "mcnemar_p": round(p, 4)},
                 "excluded_amended": sum(len(e["amended"]) for e in excl.values()),
                 "excluded_stale": sum(len(e["stale"]) for e in excl.values()),
                 "excluded_unrecorded": sum(len(e["unrecorded"]) for e in excl.values()),
                 "excluded": excl,
                 "per_set": {r["set"].split(":")[0]: {k: r[k] for k in ("n", "current", "candidate", "delta", "tol", "b", "c",
                                                                        "mcnemar_p")} for r in rows}})
    bar[f"{name}_reference"] = info
    bar["pass"] = all(r["pass"] for r in bar["rules"])
    return bar


def reference_text(info: dict | None) -> str:
    """One decision-message line: "Against r4's recorded answers: holdout −x (tol t), real …"."""
    if not info:
        return ""
    name = info.get("name") or info.get("tag") or "r4"
    what = f"Against {info['label']}" if info.get("label") else f"Against {name}'s recorded answers"
    if not info.get("measured"):
        return f"{what}: {info.get('detail') or 'NOT MEASURED'}."
    per = info.get("per_set") or {}
    parts = [f"{s} {v['delta']:+.1f} (tol {v['tol']}, n {v['n']})" for s, v in per.items()]
    ex = []
    if info.get("excluded_amended"):
        ex.append(f"{info['excluded_amended']} re-keyed")
    if info.get("excluded_stale"):
        ex.append(f"{info['excluded_stale']} stale")
    if info.get("gate") is False:
        what += " (report only)"
    return (f"{what}: " + "; ".join(parts) +
            (f" ({', '.join(ex)} rows left out)" if ex else "") + ".")


def _trigger_rule(sets: list[dict], rules: list[str], cur: dict, cand: dict, baseline: dict | None, th: dict,
                  dodged: list[str], probe_by_key: dict) -> dict:
    if any(r in rules for r in DODGED_RULES) and dodged:
        per = []
        ok = True
        for k in dodged:
            arms = probe_by_key.get(k) or {}
            kk = common(arms.get("current") or [], arms.get("candidate") or [])
            if not kk:
                ok = False
                per.append(f"{k}: not probed")
                continue
            rc = _mean(arms.get("current") or [], kk) / len(kk)
            rk = _mean(arms.get("candidate") or [], kk) / len(kk)
            good = rk >= PROBE_MIN and rk > rc
            ok = ok and good
            per.append(f"{k.split('.')[-1]} {rk:.0%} vs {rc:.0%}")
        return {"name": "trigger", "pass": ok, "detail": "dodged keys: " + "; ".join(per)}
    if any(r in rules for r in EXAM_DRIFT_RULES):
        cur_all = _merge([cur.get(s) or [] for s in EXAM_SETS])
        cand_all = _merge([cand.get(s) or [] for s in EXAM_SETS])
        keys = common(cur_all, cand_all)
        sig, _ = sigma_run(keys, baseline, th)
        sd = sig * math.sqrt(1 / max(len(cur_all), 1) + 1 / max(len(cand_all), 1))
        d = _mean(cand_all, keys) - _mean(cur_all, keys)
        return {"name": "trigger", "pass": d >= sd,
                "detail": f"exam drift: total {d:+.1f} rows of {len(keys)} (need ≥ σ_Δ {sd:.1f})"}
    return {"name": "trigger", "pass": True, "detail": f"not applicable (trigger rules {','.join(rules) or 'none'})"}


def _merge(per_set: list[list[dict[str, int]]]) -> list[dict[str, int]]:
    """Replicate k of every set in one dict (row keys are set-qualified)."""
    m = min((len(x) for x in per_set if x), default=0)
    out = [dict() for _ in range(m)]
    for reps in per_set:
        for k in range(m):
            out[k].update(reps[k] if k < len(reps) else {})
    return out


def _order(s: str) -> tuple:
    base, _, var = s.partition(":")
    rank = {name: i for i, name in enumerate(EXAM_SETS)}
    return (0 if base in rank and not var else 1 if base in rank else 2 if base == "F0H" else 3 if base == "F9H" else 4,
            rank.get(base, 99), s)


def table_text(bar: dict, *, limit: int = 12) -> str:
    """The Slack / page table (fixed width)."""
    lines = [f"{'set':<22}{'n':>5}{'live':>7}{'cand':>7}{'Δ':>7}{'p':>7}"]
    live = [x for x in bar["sets"] if (x["bar"] or x["set"] in ("F0H:unseen",) or x["set"].split(":")[0] in REPORT_ONLY)
            and x.get("variant") != REFERENCE_VARIANT][:limit]
    ref = [x for x in bar["sets"] if x.get("variant") == REFERENCE_VARIANT]  # under the live rows ("live" = the reference)
    for r in live + ref:
        lines.append(f"{r['set']:<22}{r['n']:>5}{r['current']:>7.0f}{r['candidate']:>7.0f}{r['delta']:>+7.0f}"
                     f"{r['mcnemar_p']:>7.2f}")
    return "\n".join(lines)
