"""pass@k over N eval runs of one route on one manifest (LAB.md §6).

Copied from breadstack-thermometer lab/in-house-model/passk.py (FOCACCIA-PLAN
4.2, G0b) and adapted: the runs are ``eval.py``'s files
(``runs/eval-<label>-<UTC>.json``: ``{label, route, manifest_hash,
rule_version, results:[{id, kind, passed, verdict, error?}]}``), the
per-check table becomes a per-verdict table, and nothing is asked of any
model — the files are only read:

  pass@1      the mean pass rate: each case's share of passing samples
              (c of n), averaged over the cases
  pass@k      unbiased (Chen et al., 2021): each case's
              1 - C(n-c, k) / C(n, k), averaged, for k in 1, 2, 4 and 8 —
              over the cases with at least k samples
  by kind     the same for each case kind
  by verdict  the share of answers per verdict
  format      the share of answers that failed as answers, not on the diff:
              an error, or no_answer
  learnable   the share of cases with 1 to n-1 of their n samples passing
  headroom    pass@4 - pass@1 in points: 5 or more → SFT then DPO; under
              5 → stop and ask before any spend (LAB.md §6)

The runs must be one route's, on one manifest hash, scored under one rule
version. A case some run did not answer counts with the samples it has.
The table prints counts and rates only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .verify import VERDICTS

KS = (1, 2, 4, 8)
HEADROOM_K = 4
HEADROOM_MIN_POINTS = 5.0  # LAB.md §6: SFT, then DPO, whenever the headroom is at least this
FORMAT_FAILURES = ("error", "no_answer")


class RunsError(ValueError):
    pass


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """The unbiased estimate of pass@k from n samples, c of them passing: the
    chance that k drawn without replacement hold at least one pass. None
    when there are fewer than k samples."""
    if k < 1 or n < k or not 0 <= c <= n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mean_pass_at_k(counts: list[tuple[int, int]], k: int) -> tuple[float | None, int]:
    """(the mean pass@k over the cases with at least k samples, how many such
    cases) for (n, c) pairs."""
    vals = [v for n, c in counts if (v := pass_at_k(n, c, k)) is not None]
    return (sum(vals) / len(vals) if vals else None), len(vals)


PASSING_VERDICTS = ("pass", "clarify_ok", "unsupported_ok")  # verify.PASSING


def passed_of(r: dict) -> bool:
    """`passed` when the run recorded it, else derived from the verdict (eval runs before 2026-10-01 lack the field)."""
    if r.get("passed") is not None:
        return bool(r["passed"])
    return str(r.get("verdict") or "") in PASSING_VERDICTS


def format_failure(r: dict) -> str:
    """Why an answer failed as an answer ("error", "no_answer"), or ""."""
    if r.get("error"):
        return "error"
    if str(r.get("verdict") or "") == "no_answer":
        return "no_answer"
    return ""


def route_of(run: dict) -> str:
    return str(run.get("route") or run.get("model") or "")


def load_runs(files: list[Path | str]) -> list[dict]:
    """The runs, checked to be N samples of one route on one manifest under
    one rule version; RunsError names what differs."""
    seen: set[Path] = set()
    runs = []
    for f in files:
        p = Path(f).resolve()
        if p in seen:
            raise RunsError(f"{f} is given twice: each run is one sample")
        seen.add(p)
        try:
            run = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise RunsError(f"cannot read {f}: {e}") from e
        if not isinstance(run, dict) or not isinstance(run.get("results"), list):
            raise RunsError(f"{f} is not an eval run (eval.py's results file)")
        ids = [r.get("id") for r in run["results"]]
        if len(ids) != len(set(ids)):
            raise RunsError(f"{f} answers a case more than once")
        run["_path"] = str(p)
        runs.append(run)
    if not runs:
        raise RunsError("no run given")

    def same(what: str, get) -> None:
        vals = {get(r) for r in runs}
        if len(vals) > 1:
            raise RunsError(f"the runs differ in {what}: {', '.join(sorted(map(str, vals)))} — pass@k needs samples of "
                            "one route on one manifest, scored one way")

    same("route", route_of)
    same("manifest hash", lambda r: str(r.get("manifest_hash") or ""))
    same("rule version", lambda r: str(r.get("rule_version") or ""))
    return runs


def tally(runs: list[dict]) -> dict:
    """Every figure of the read-out (module docstring), counts and rates only."""
    per_q: dict[str, dict] = {}
    verdicts: dict[str, int] = defaultdict(int)
    fails = {k: 0 for k in FORMAT_FAILURES}
    answers = 0
    for run in runs:
        for r in run["results"]:
            qid = str(r["id"])
            q = per_q.setdefault(qid, {"kind": str(r.get("kind") or (r.get("meta") or {}).get("kind") or ""), "n": 0, "c": 0})
            q["n"] += 1
            q["c"] += bool(passed_of(r))
            answers += 1
            verdicts[str(r.get("verdict") or ("error" if r.get("error") else "unknown"))] += 1
            if why := format_failure(r):
                fails[why] += 1

    def block(counts: list[tuple[int, int]]) -> dict:
        out = {"cases": len(counts), "samples": sum(n for n, _ in counts), "passes": sum(c for _, c in counts)}
        for k in KS:
            v, m = mean_pass_at_k(counts, k)
            out[f"pass@{k}"] = None if v is None else round(100 * v, 1)
            if m != len(counts):
                out[f"pass@{k}_cases"] = m
        return out

    counts = [(q["n"], q["c"]) for q in per_q.values()]
    overall = block(counts)
    kinds = sorted({q["kind"] for q in per_q.values()})
    by_kind = {k: block([(q["n"], q["c"]) for q in per_q.values() if q["kind"] == k]) for k in kinds}
    by_verdict = {v: {"answers": verdicts[v], "share_pct": round(100 * verdicts[v] / answers, 1) if answers else None}
                  for v in [*VERDICTS, *sorted(set(verdicts) - set(VERDICTS))] if verdicts.get(v)}
    learnable = sum(1 for n, c in counts if 0 < c < n)
    always = sum(1 for n, c in counts if n and c == n)
    never = sum(1 for n, c in counts if c == 0)
    failed = sum(fails.values())
    p1, pk = overall["pass@1"], overall[f"pass@{HEADROOM_K}"]
    headroom = None if p1 is None or pk is None else round(pk - p1, 1)
    return {
        "runs": len(runs), "cases": len(per_q), "answers": answers,
        "complete": sum(1 for q in per_q.values() if q["n"] == len(runs)),
        "overall": overall, "by_kind": by_kind, "by_verdict": by_verdict,
        "format": {"answers": answers, **fails, "failed": failed,
                   "rate_pct": round(100 * failed / answers, 1) if answers else None},
        "learnable": {"band": "1 to n-1 of n", "cases": learnable, "always": always, "never": never,
                      "share_pct": round(100 * learnable / len(per_q), 1) if per_q else None},
        "headroom": {"k": HEADROOM_K, "points": headroom, **ladder(headroom)},
        "per_case": {qid: [q["n"], q["c"]] for qid, q in sorted(per_q.items())},
    }


def ladder(headroom: float | None) -> dict:
    """The training ladder the headroom gives (LAB.md §6): at least
    HEADROOM_MIN_POINTS, SFT then DPO; less, stop and ask before any spend."""
    rule = (f"pass@{HEADROOM_K} - pass@1 >= {HEADROOM_MIN_POINTS:g} points: SFT, then DPO; under it: stop and ask "
            "before any spend")
    if headroom is None:
        return {"verdict": "unknown", "rule": rule, "why": f"pass@{HEADROOM_K} needs {HEADROOM_K} samples of a case"}
    if headroom >= HEADROOM_MIN_POINTS:
        return {"verdict": "sft-then-dpo", "rule": rule}
    return {"verdict": "stop", "rule": rule}


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def render(t: dict, route: str) -> str:
    """The read-out as plain text (counts and rates only)."""
    lines = [f"pass@k · {route} · {t['runs']} runs · {t['cases']} cases ({t['complete']} answered in every run) · "
             f"{t['answers']} answers",
             f"{'':18} {'cases':>6} " + " ".join(f"{'pass@' + str(k):>8}" for k in KS)]

    def row(name: str, b: dict) -> str:
        return f"{name:18} {b['cases']:>6} " + " ".join(f"{_pct(b[f'pass@{k}']):>8}" for k in KS)

    lines.append(row("overall", t["overall"]))
    for k, b in t["by_kind"].items():
        lines.append(row(f"kind {k}"[:18], b))
    for v, b in t["by_verdict"].items():
        lines.append(f"{'verdict ' + v:18} {b['answers']:>6} answers ({_pct(b['share_pct'])}%)")
    f, lb, h = t["format"], t["learnable"], t["headroom"]
    lines.append(f"format failures: {f['failed']} of {f['answers']} answers ({_pct(f['rate_pct'])}%): "
                 f"{f['error']} errors, {f['no_answer']} no_answer")
    lines.append(f"learnable band (1 to n-1 of n passing): {lb['cases']} of {t['cases']} cases "
                 f"({_pct(lb['share_pct'])}%); always {lb['always']}, never {lb['never']}")
    lines.append(f"headroom pass@{h['k']} - pass@1: {_pct(h['points'])} points → {h['verdict']} ({h['rule']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="the N eval runs (eval-*.json) of one route on one manifest")
    p.add_argument("--out", help="also write the tally (counts, rates and case ids) here")
    args = p.parse_args(argv)
    try:
        runs = load_runs(args.runs)
    except RunsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    t = tally(runs)
    route = route_of(runs[0])
    print(render(t, route))
    if args.out:
        out = {"kind": "passk", "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "route": route, "manifest_hash": runs[0].get("manifest_hash"), "rule_version": runs[0].get("rule_version"),
               "files": [r["_path"] for r in runs], **t}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
