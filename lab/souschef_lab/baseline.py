"""baseline — the drift check's baseline per model image, its noise model and
the paired comparison (DESIGN §1.4, docs/DRIFT.md "Noise").

`baselines/<sha256 digest>/baseline.json` in the pipeline bucket:

  {digest, tag, training_manifest, engine_sha, service_build, contexts,
   replicates: [{at, run, rows: {"<set>:<row id>": 0|1}, ghost_writes}],   arm T (or the candidate eval), newest last
   per_set: {<set>: {n, mean, sd_run}, total: {...}},  row_p: {"<set>:<id>": p},
   probe: {<address>: {n, pass}}, source: calibration | candidate-eval | weekly, calibrated}

Noise. Sampling at temperature 0.3 flips rows between runs. Over rows with pass
rates p_i from K replicates, a set's run-to-run SD is
σ_run = sqrt(K/(K−1) · Σ p_i (1 − p_i)) (unbiased); with K ≥ 3 the observed SD of the per-run pass
count is used when larger; never below `min_sigma_rows`; with fewer than 2
replicates (no baseline yet) every row gets the prior flip rate f:
σ_run = sqrt(n f (1 − f)). Arm D's m replicates against the baseline mean of K:
σ_Δ = σ_run · sqrt(1/m + 1/K). The exact McNemar test pairs (row, k): arm D's
k-th replicate against the baseline's k-th newest.

Δ is baseline mean − arm D mean in ROWS, over the rows both have (stale rows
are left out of both).

Standard library only.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

from .examcmp import mcnemar_p

TOTAL = "total"


def key_set(row_key: str) -> str:
    return row_key.split(":", 1)[0]


def new(*, digest: str, tag: str | None, training_manifest: str | None, engine_sha: str | None,
        service_build: str | None, contexts: str = "rederived", source: str = "calibration") -> dict:
    return {"digest": digest, "tag": tag, "training_manifest": training_manifest, "engine_sha": engine_sha,
            "service_build": service_build, "contexts": contexts, "replicates": [], "per_set": {}, "row_p": {},
            "probe": {}, "source": source, "calibrated": False}


def add_replicate(b: dict, rows: dict[str, int], *, run: str, at: str, ghost_writes: int | None = None,
                  window: int | None = None, f0h: dict[str, int] | None = None) -> dict:
    """Append one arm-T replicate (keep the newest `window`) and recompute."""
    rep = {"at": at, "run": run, "rows": dict(rows)}
    if ghost_writes is not None:
        rep["ghost_writes"] = int(ghost_writes)
    if f0h:
        rep["f0h"] = dict(f0h)
    b.setdefault("replicates", []).append(rep)
    if window:
        b["replicates"] = b["replicates"][-int(window):]
    return recompute(b)


def recompute(b: dict) -> dict:
    reps = b.get("replicates") or []
    b["row_p"] = row_p(reps)
    b["per_set"] = per_set_stats(reps)
    gw = [r["ghost_writes"] for r in reps if "ghost_writes" in r]
    b["ghost_writes_mean"] = round(sum(gw) / len(gw), 3) if gw else None
    return b


def common_rows(reps: list[dict]) -> set[str]:
    sets = [set((r.get("rows") or {}).keys()) for r in reps]
    return set.intersection(*sets) if sets else set()


def row_p(reps: list[dict]) -> dict[str, float]:
    keys = common_rows(reps)
    return {k: round(sum(int((r.get("rows") or {})[k]) for r in reps) / len(reps), 4) for k in sorted(keys)} if reps else {}


def per_set_stats(reps: list[dict]) -> dict[str, dict]:
    keys = common_rows(reps)
    groups: dict[str, list[str]] = {}
    for k in keys:
        groups.setdefault(key_set(k), []).append(k)
    groups[TOTAL] = list(keys)
    out = {}
    for s, ks in sorted(groups.items()):
        counts = [sum(int((r.get("rows") or {})[k]) for k in ks) for r in reps]
        out[s] = {"n": len(ks), "mean": round(sum(counts) / len(counts), 3) if counts else None,
                  "sd_run": round(statistics.stdev(counts), 3) if len(counts) >= 2 else None, "runs": counts}
    return out


def sigma_run(rows: Iterable[str], b: dict | None, th: dict) -> tuple[float, str]:
    """(σ_run in rows, how) over `rows` of a baseline (or the prior)."""
    rows = list(rows)
    reps = (b or {}).get("replicates") or []
    floor = float(th.get("min_sigma_rows", 1.0))
    if len(reps) < 2:
        f = float(th.get("prior_flip_rate", 0.08))
        return max(math.sqrt(len(rows) * f * (1 - f)), floor), "prior"
    rp = row_p(reps)
    # p(1 − p) from K replicates under-states each row's variance by (K − 1)/K
    # (the plug-in estimator): K/(K − 1) makes it unbiased, which matters at K = 2
    # (a candidate baseline) where it doubles the estimate.
    k_n = len(reps)
    est = math.sqrt(k_n / (k_n - 1) * sum(rp[k] * (1 - rp[k]) for k in rows if k in rp))
    how = "row_p"
    if len(reps) >= 3:
        counts = [sum(int((r.get("rows") or {}).get(k, 0)) for k in rows) for r in reps]
        obs = statistics.stdev(counts)
        if obs > est:
            est, how = obs, "observed"
    if est < floor:
        return floor, f"{how}+floor"
    return est, how


def base_mean(rows: Iterable[str], b: dict) -> float | None:
    reps = (b or {}).get("replicates") or []
    rows = list(rows)
    if not reps:
        return None
    return sum(sum(int((r.get("rows") or {}).get(k, 0)) for k in rows) for r in reps) / len(reps)


def compare(b: dict | None, arm: list[dict[str, int]], th: dict, *, ref_means: dict[str, float] | None = None,
            ref_reps: int | None = None, ref_rows: list[dict[str, int]] | None = None) -> dict[str, dict]:
    """Per set (and `total`): n, the baseline mean, arm D's mean, Δ (rows),
    σ_run, σ_Δ, z = Δ / σ_Δ, the paired McNemar b/c/p. Rows: those every arm
    replicate has and (with a baseline) every baseline replicate has.

    Skip's ack: drift is measured beyond the acknowledged arm-D level.
    `ref_rows` = the ack's arm-D replicates ({row key: 0/1} each) replace the
    baseline mean over the SAME rows (the intersection of arm D, the baseline
    and the ack), with K = their count — rows that went stale or were rebased
    since the Skip never shift Δ. `ref_means` = {set: mean} (an ack written
    before per-row results were kept) is the coarse fallback, K = `ref_reps`."""
    reps = (b or {}).get("replicates") or []
    rows = set.intersection(*[set(a) for a in arm]) if arm else set()
    if reps:
        rows &= common_rows(reps)
    if ref_rows:
        rows &= set.intersection(*[set(r) for r in ref_rows])
    groups: dict[str, list[str]] = {}
    for k in rows:
        groups.setdefault(key_set(k), []).append(k)
    groups[TOTAL] = sorted(rows)
    m = len(arm)
    out: dict[str, dict] = {}
    for s, ks in sorted(groups.items()):
        n = len(ks)
        d_mean = sum(sum(int(a[k]) for k in ks) for a in arm) / m if m else None
        if ref_rows:
            bm, K = sum(sum(int(r[k]) for k in ks) for r in ref_rows) / len(ref_rows), len(ref_rows)
        elif ref_means is not None and s in ref_means:
            bm, K = float(ref_means[s]), max(1, int(ref_reps or 1))
        else:
            bm, K = base_mean(ks, b) if reps else None, len(reps)
        sig, how = sigma_run(ks, b, th)
        sd = sig * math.sqrt(1 / max(m, 1) + 1 / max(K, 1)) if K else sig * math.sqrt(1 / max(m, 1) + 1)
        delta = (bm - d_mean) if (bm is not None and d_mean is not None) else None
        bb = cc = 0
        if reps:
            newest = list(reversed(reps))
            for k_i, a in enumerate(arm[: len(newest)]):
                base_rows = newest[k_i].get("rows") or {}
                for k in ks:
                    x, y = int(base_rows.get(k, 0)), int(a[k])
                    bb += int(x == 1 and y == 0)
                    cc += int(x == 0 and y == 1)
        out[s] = {"n": n, "base_mean": None if bm is None else round(bm, 3), "arm_mean": None if d_mean is None else round(d_mean, 3),
                  "delta": None if delta is None else round(delta, 3), "sigma_run": round(sig, 3), "sigma_how": how,
                  "sigma_delta": round(sd, 3), "z": None if delta is None or sd == 0 else round(delta / sd, 2),
                  "m": m, "K": K, "b": bb, "c": cc, "mcnemar_p": round(mcnemar_p(bb, cc), 4) if reps else None,
                  "pct_delta": None if delta is None or not n else round(100.0 * delta / n, 2)}
    return out


def ack_reference(ack: dict | None) -> tuple[dict[str, float] | None, int | None]:
    if not ack or not isinstance(ack.get("arm_D_sets"), dict):
        return None, None
    return {k: float(v) for k, v in ack["arm_D_sets"].items() if isinstance(v, (int, float))}, int(ack.get("arm_D_replicates") or 1)


def ack_rows(ack: dict | None) -> list[dict[str, int]] | None:
    """The ack's arm-D replicates, per row ({row key: 0/1} each), or None for an
    ack written before they were kept (ack_reference's per-set means then)."""
    reps = (ack or {}).get("arm_D_rows")
    if not isinstance(reps, list) or not reps or not all(isinstance(r, dict) and r for r in reps):
        return None
    return [{str(k): int(v) for k, v in r.items()} for r in reps]


def summary_line(c: dict[str, Any]) -> str:
    t = c.get(TOTAL) or {}
    if t.get("base_mean") is None:
        return f"{t.get('arm_mean')}/{t.get('n')} (no baseline)"
    return (f"{t.get('arm_mean'):.0f}/{t.get('n')} vs baseline {t.get('base_mean'):.1f} ± {t.get('sigma_delta'):.1f} "
            f"({-t.get('delta'):+.1f} rows, McNemar p {t.get('mcnemar_p')})")
