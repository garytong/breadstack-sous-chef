"""examcmp — compare real-phrasing exam runs (docs/LAB.md §6b) across
revisions: pass rate per set × route × revision with the verdict breakdown,
Wilson 95 % intervals and the cost the rows' usage implies.

    python -m souschef_lab.examcmp --run base='runs/exam-*-base-*.json' --run cand='runs/exam-*-cand-*.json'
    python -m souschef_lab.examcmp --run base=… --run cand=… --json out.json

`--run REV=GLOB` (repeatable) names a revision and the `exam run` reports
that belong to it. Chunks of one set (`exam run --offset/--limit`, file names
`…-r<from>-<to>-…`) are merged: rows are keyed by row id (`<case>#<sample>`),
a later report (`finished_at`) replaces an earlier one's row, and the
coverage column says how many of the set's rows were seen (a missing chunk
shows as `n < of`).

Pass rate = passed / rows, with transport errors (status ≠ ok) and unscored
rows counted as NOT passed — an A/B must not reward a revision for failing
loudly; `scored` (errors left out) is printed beside it. Wilson score
interval, z = 1.96.

Paired (McNemar): both revisions run the SAME rows, so for every set ×
route and every pair of revisions the rows both ran are compared row by
row — both pass, only the first, only the second, neither — with the exact
two-sided McNemar p (binomial on the discordant rows). It detects much
smaller differences than two independent Wilson intervals.

Out of scope: the primary verdict scores a `where` the browser can link
exactly and a reply without one by its words, so two revisions that differ
in sending `where` are not measured alike; `kw` is the keyword rule on the
reply's words for both (`keyword_verdict`, exam.score_turn).

Escalations are counted by reason; `hidden_write` hand-offs (a gated write
whose gate the browser's `hidden` member says is closed — never sent by an
engine without the member) are also shown left out: the pass rate and cost
of the rows that did not hand off for that reason.

Cost: tokens × the route's model price (`--price ROUTE=MODEL`, defaults:
default → claude-sonnet-5-5, opus → claude-opus-5-5; a route without a price,
such as the in-house `blueberry-muffin`, is $0 per token). A hybrid turn's
`usage` sums the student's and Claude's calls (README §1, turn.done), so an
ESCALATED row of a hybrid route is priced whole at its fallback's model — an
upper bound, flagged `≤` — and a non-escalated one at $0. `--gpu-hourly` adds
wall-clock × rate for the routes named by `--gpu-route` (default
blueberry-muffin; the reports' started_at → finished_at, summed). Prices are
the first-party list prices (claude-api skill, cached 2026-09-25): input /
output / 5-minute cache write (1.25 × input) / cache read per million tokens.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# $/MTok: input, output, cache write (5 m), cache read.
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.20),
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
}
DEFAULT_ROUTE_MODELS = {"default": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
DEFAULT_FALLBACKS = {"blueberry-muffin": "default"}
PASS = frozenset(("pass", "clarify_ok", "unsupported_ok"))
SET_ORDER = ("paraphrase", "holdout", "real", "out_of_scope", "out_of_scope_holdout", "followup")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for k successes of n, as fractions."""
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def _ts(s: str | None) -> datetime | None:
    try:
        return datetime.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return None


def load(specs: list[tuple[str, str]]) -> dict[tuple[str, str, str], dict]:
    """{(set, route, rev): {rows: {id: row}, of, files, wall_s, fields_context, label}}."""
    groups: dict[tuple[str, str, str], dict] = {}
    for rev, pattern in specs:
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise SystemExit(f"examcmp: --run {rev}={pattern} matches no file")
        reports = []
        for p in paths:
            doc = json.loads(Path(p).read_text(encoding="utf-8"))
            if not isinstance(doc, dict) or "results" not in doc or "set" not in doc:
                raise SystemExit(f"examcmp: {p} is not an `exam run` report")
            reports.append((str(doc.get("finished_at") or ""), p, doc))
        for _, p, doc in sorted(reports):  # later reports win a row
            key = (str(doc["set"]), str(doc.get("route") or "(default)"), rev)
            g = groups.setdefault(key, {"rows": {}, "of": 0, "files": [], "wall_s": 0.0, "fields_context": set(),
                                        "labels": set()})
            for r in doc["results"]:
                g["rows"][str(r.get("id"))] = r
            rng = doc.get("rows_range") or {}
            g["of"] = max(g["of"], int(rng.get("of") or doc.get("cases") or 0))
            g["files"].append(p)
            a, b = _ts(doc.get("started_at")), _ts(doc.get("finished_at"))
            if a and b:
                g["wall_s"] += (b - a).total_seconds()
            g["fields_context"].add(bool(doc.get("fields_context")))
            g["labels"].add(str(doc.get("label")))
    return groups


def row_cost(row: dict, route: str, route_models: dict[str, str], fallbacks: dict[str, str]) -> tuple[float, bool]:
    """(dollars, upper_bound) for one row's usage."""
    u = row.get("usage") or {}
    model = route_models.get(route)
    upper = False
    if model is None:
        if not row.get("escalated"):
            return 0.0, False
        to = str((row.get("escalated") or {}).get("to") or fallbacks.get(route) or "")
        model = route_models.get(to)
        upper = True
        if model is None:
            return 0.0, True
    pin, pout, pw, pr = PRICES[model]
    d = (pin * int(u.get("input_tokens") or 0) + pout * int(u.get("output_tokens") or 0)
         + pw * int(u.get("cache_creation_input_tokens") or 0) + pr * int(u.get("cache_read_input_tokens") or 0)) / 1e6
    return d, upper


def summarize(groups: dict, route_models: dict[str, str], fallbacks: dict[str, str], gpu_hourly: float,
              gpu_routes: set[str]) -> list[dict]:
    out = []
    for (set_name, route, rev), g in groups.items():
        rows = list(g["rows"].values())
        n = len(rows)
        passed = sum(1 for r in rows if r.get("status") == "ok" and r.get("verdict") in PASS)
        errors = sum(1 for r in rows if r.get("status") != "ok")
        scored = sum(1 for r in rows if r.get("status") == "ok" and r.get("verdict") is not None)
        verdicts = Counter(str(r.get("verdict") if r.get("status") == "ok" else f"error:{r.get('code')}") for r in rows)
        cost, upper = 0.0, 0
        tokens = Counter()
        for r in rows:
            d, ub = row_cost(r, route, route_models, fallbacks)
            cost += d
            upper += ub
            for k, v in (r.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    tokens[k] += int(v)
        reasons = Counter(str((r.get("escalated") or {}).get("reason") or "?") for r in rows if r.get("escalated"))
        not_hw = [r for r in rows if (r.get("escalated") or {}).get("reason") != "hidden_write"]
        kw_rows = [r for r in rows if r.get("status") == "ok" and (r.get("keyword_verdict") is not None)]
        gpu = gpu_hourly * g["wall_s"] / 3600 if route in gpu_routes else 0.0
        lat = sorted(int(r.get("latency_ms") or 0) for r in rows)
        ci = wilson(passed, n)
        out.append({"set": set_name, "route": route, "rev": rev, "n": n, "of": g["of"], "passed": passed,
                    "scored": scored, "errors": errors, "rate": passed / n if n else None, "ci95": ci,
                    "rate_scored": passed / scored if scored else None, "verdicts": dict(sorted(verdicts.items())),
                    "escalated": sum(1 for r in rows if r.get("escalated")),
                    "escalated_by_reason": dict(sorted(reasons.items())),
                    "without_hidden_write": {
                        "n": len(not_hw),
                        "passed": sum(1 for r in not_hw if r.get("status") == "ok" and r.get("verdict") in PASS),
                        "cost_usd": round(sum(row_cost(r, route, route_models, fallbacks)[0] for r in not_hw), 4)},
                    "keyword_passed": sum(1 for r in kw_rows if r.get("keyword_verdict") == "unsupported_ok")
                    if kw_rows else None,
                    "where_scored": sum(1 for r in rows if r.get("scored_by") == "where"),
                    "cost_usd": round(cost, 4), "cost_upper_rows": upper, "gpu_usd": round(gpu, 4),
                    "tokens": dict(tokens), "wall_s": round(g["wall_s"], 1),
                    "latency_p50_ms": lat[len(lat) // 2] if lat else None,
                    "fields_context": sorted(g["fields_context"]), "files": len(g["files"])})
    order = {s: i for i, s in enumerate(SET_ORDER)}
    out.sort(key=lambda s: (order.get(s["set"], 99), s["set"], s["route"], s["rev"]))
    return out


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p for b and c discordant rows (binomial, p = 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _passed(r: dict) -> bool:
    return r.get("status") == "ok" and r.get("verdict") in PASS


def paired(groups: dict, revs: list[str]) -> list[dict]:
    """Row-by-row comparison of every pair of revisions (in `revs` order) on
    each set × route, over the row ids both ran; plus one pooled line per
    route × pair."""
    out = []
    by_sr: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for (set_name, route, rev), g in groups.items():
        by_sr[(set_name, route)][rev] = g["rows"]
    order = {s: i for i, s in enumerate(SET_ORDER)}
    pooled: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for (set_name, route) in sorted(by_sr, key=lambda k: (order.get(k[0], 99), k[0], k[1])):
        have = by_sr[(set_name, route)]
        for i, a in enumerate(revs):
            for b in revs[i + 1:]:
                if a not in have or b not in have:
                    continue
                ids = sorted(set(have[a]) & set(have[b]))
                both = only_a = only_b = neither = 0
                for rid in ids:
                    pa, pb = _passed(have[a][rid]), _passed(have[b][rid])
                    both += pa and pb
                    only_a += pa and not pb
                    only_b += pb and not pa
                    neither += not pa and not pb
                t = pooled[(route, a, b)]
                for j, v in enumerate((both, only_a, only_b, neither)):
                    t[j] += v
                out.append({"set": set_name, "route": route, "a": a, "b": b, "n": len(ids), "both": both,
                            "only_a": only_a, "only_b": only_b, "neither": neither, "p": mcnemar_p(only_a, only_b)})
    for (route, a, b), (both, only_a, only_b, neither) in sorted(pooled.items()):
        out.append({"set": "(all)", "route": route, "a": a, "b": b, "n": both + only_a + only_b + neither, "both": both,
                    "only_a": only_a, "only_b": only_b, "neither": neither, "p": mcnemar_p(only_a, only_b)})
    return out


def format_paired(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = ["", "paired (rows both revisions ran; exact McNemar p on the rows only one passed):",
             f"  {'set':<21} {'route':<17} {'a → b':<13} {'n':>4} {'both':>4} {'a only':>6} {'b only':>6} {'neither':>7} {'p':>7}"]
    for r in rows:
        lines.append(f"  {r['set']:<21} {r['route']:<17} {r['a'] + ' → ' + r['b']:<13} {r['n']:>4} {r['both']:>4} "
                     f"{r['only_a']:>6} {r['only_b']:>6} {r['neither']:>7} {r['p']:7.3f}")
    return "\n".join(lines)


def _pct(x: float | None) -> str:
    return "  –  " if x is None else f"{100 * x:5.1f}"


def format_table(summary: list[dict]) -> str:
    head = (f"{'set':<21} {'route':<17} {'rev':<6} {'pass/n':>9} {'rate':>5} {'95% CI':>13} {'scored':>6} "
            f"{'err':>3} {'esc':>3} {'cost $':>8} {'p50 s':>6}  verdicts")
    lines = [head, "-" * len(head)]
    for s in summary:
        ci = s["ci95"]
        cov = "" if s["n"] >= s["of"] else f" (of {s['of']}!)"
        cost = f"{'≤' if s['cost_upper_rows'] else ' '}{s['cost_usd'] + s['gpu_usd']:7.2f}"
        lines.append(f"{s['set']:<21} {s['route']:<17} {s['rev']:<6} {s['passed']:>4}/{s['n']:<4} {_pct(s['rate'])} "
                     f"{'[' + _pct(ci[0]).strip() + ',' + _pct(ci[1]).strip() + ']' if ci else '–':>13} "
                     f"{_pct(s['rate_scored'])} {s['errors']:>3} {s['escalated']:>3} {cost} "
                     f"{(s['latency_p50_ms'] or 0) / 1000:6.1f}  "
                     + ", ".join(f"{k} {v}" for k, v in s["verdicts"].items()) + cov)
        extra = []
        if s.get("keyword_passed") is not None:
            extra.append(f"kw (keyword rule for both) {s['keyword_passed']}/{s['n']}")
        if s.get("escalated_by_reason"):
            extra.append("esc " + ", ".join(f"{k} {v}" for k, v in s["escalated_by_reason"].items()))
        hw = s.get("escalated_by_reason", {}).get("hidden_write")
        if hw:
            w = s["without_hidden_write"]
            extra.append(f"without hidden_write {w['passed']}/{w['n']} ${w['cost_usd']:.2f}")
        if extra:
            lines.append(f"{'':<21} {'':<17} {'':<6} " + " · ".join(extra))
    # totals per route × rev
    tot: dict[tuple[str, str], list[int | float]] = defaultdict(lambda: [0, 0, 0.0, 0.0])
    for s in summary:
        t = tot[(s["route"], s["rev"])]
        t[0] += s["passed"]
        t[1] += s["n"]
        t[2] += s["cost_usd"]
        t[3] += s["gpu_usd"]
    lines.append("")
    lines.append("all sets (rows pooled; sets differ in size and difficulty — read the per-set rows first):")
    for (route, rev), (k, n, c, g) in sorted(tot.items()):
        ci = wilson(int(k), int(n))
        lines.append(f"  {route:<17} {rev:<6} {k}/{n} = {_pct(k / n if n else None).strip()} % "
                     f"[{_pct(ci[0]).strip()}, {_pct(ci[1]).strip()}] · tokens ${c:.2f}"
                     + (f" + GPU ${g:.2f}" if g else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="examcmp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="REV=GLOB",
                   help="a revision name and the exam-run reports that belong to it (repeatable)")
    p.add_argument("--price", action="append", default=[], metavar="ROUTE=MODEL",
                   help=f"price a route's tokens as this model ({', '.join(PRICES)}); defaults {DEFAULT_ROUTE_MODELS}")
    p.add_argument("--fallback", action="append", default=[], metavar="ROUTE=FALLBACK",
                   help=f"a hybrid route's fallback route (defaults {DEFAULT_FALLBACKS})")
    p.add_argument("--gpu-hourly", type=float, default=0.0, help="$/hour for the GPU route's wall-clock (default 0)")
    p.add_argument("--gpu-route", action="append", default=None, help="routes billed by --gpu-hourly (default blueberry-muffin)")
    p.add_argument("--json", type=Path, help="also write the summary as JSON here")
    a = p.parse_args(argv)
    specs = []
    for spec in a.run:
        rev, sep, pat = spec.partition("=")
        if not sep or not rev or not pat:
            p.error(f"--run {spec!r}: want REV=GLOB")
        specs.append((rev, pat))
    route_models = dict(DEFAULT_ROUTE_MODELS)
    for spec in a.price:
        route, _, model = spec.partition("=")
        if model not in PRICES:
            p.error(f"--price {spec!r}: model must be one of {', '.join(PRICES)}")
        route_models[route] = model
    fallbacks = dict(DEFAULT_FALLBACKS)
    for spec in a.fallback:
        route, _, fb = spec.partition("=")
        fallbacks[route] = fb
    groups = load(specs)
    summary = summarize(groups, route_models, fallbacks, a.gpu_hourly, set(a.gpu_route or ["blueberry-muffin"]))
    revs = list(dict.fromkeys(rev for rev, _ in specs))
    pairs = paired(groups, revs)
    print(format_table(summary))
    if pairs:
        print(format_paired(pairs))
    if a.json:
        a.json.write_text(json.dumps({"groups": summary, "paired": pairs}, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
