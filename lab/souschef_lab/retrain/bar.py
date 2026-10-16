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
"""
from __future__ import annotations

import math
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
            th: dict, *, bar: bool) -> dict:
    keys = common(cur, cand)
    n = len(keys)
    sig, how = sigma_run(keys, baseline, th) if n else (0.0, "empty")
    sd = sig * math.sqrt(1 / max(len(cur), 1) + 1 / max(len(cand), 1))
    tol = max(1, math.ceil(2 * sd)) if n else 0
    # seed-aware tolerance (docs/RETRAIN.md R6/R10): thresholds `sigma_seed` {set: rows}, measured by
    # retrain/noise.py, adds how much a retrain of the same data moves ONE model's score on the set. The bar compares
    # two trained models (candidate and live), each a seed draw, so Var(Δ) gets 2·σ_seed²:
    # max(1, ⌈2·sqrt(σ_Δ² + 2·σ_seed²)⌉)
    ss = (th.get("sigma_seed") or {}).get(name) if isinstance(th.get("sigma_seed"), dict) else None
    extra: dict[str, Any] = {}
    if n and isinstance(ss, (int, float)) and ss > 0:
        tol = max(1, math.ceil(2 * math.sqrt(sd ** 2 + 2 * float(ss) ** 2) - 1e-9))
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
    for r in [x for x in bar["sets"] if x["bar"] or x["set"] in ("F0H:unseen",) or x["set"].split(":")[0] in REPORT_ONLY][:limit]:
        lines.append(f"{r['set']:<22}{r['n']:>5}{r['current']:>7.0f}{r['candidate']:>7.0f}{r['delta']:>+7.0f}"
                     f"{r['mcnemar_p']:>7.2f}")
    return "\n".join(lines)
