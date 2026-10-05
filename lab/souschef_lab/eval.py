"""Eval — run the held-out exam through the service and score it (docs/LAB.md §6).

Every exam case (`holdout: true`) goes through `collect.run_case` — the same
browser emulation as collection — and the verifier's verdict is recorded.
The run lands in `runs/eval-<label>-<UTC>.json`:

  summary.overall / by_kind / by_container / by_field_type
            n, passed, pass_rate (points, 0–100), errors, verdict
            histogram, ops per turn (mean, p50, max), latency p50/p95 ms
  summary.hybrid   on a route with a `fallback` (README §4): the student-only
            pass (rows that never escalated), the hybrid pass (those plus
            the escalated rows the fallback finished correctly) and the
            escalation rate with its reasons (collect.hybrid_summary)
  usage     summed input/output/cache tokens and model calls
  results   one row per sample (the collect row: events, ops, messages,
            verdict, reasons), so --rescore can re-verify under new rules

--compare A B: per-case agreement between two runs (same case, same
sample), the flips each way, and the summaries side by side.
--rescore RUN: re-verify a saved run's rows with the current verifier;
nothing is asked again.

Pattern from the thermometer kit's eval.py (summarize / compare / rescore);
nothing imported. Scoring here is the exact verifier (no figure heuristics).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import collect as C

LAB_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = LAB_DIR / "runs"
PASS_VERDICTS = C.PASS_VERDICTS
KINDS = ("set_field", "set_enum", "set_range", "toggle", "image", "list_add", "list_remove", "list_reorder", "list_update",
         "reset", "layout", "multi", "clarify", "unsupported")
try:  # verify.py owns the rule version (passk groups runs by it); the literal is the fallback while it is absent
    from .verify import RULE_VERSION
except ImportError:  # pragma: no cover
    RULE_VERSION = "verify-1"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "run"


def percentile(values: Iterable[float], pct: float) -> float | None:
    """Nearest-rank percentile; None for no values."""
    v = sorted(values)
    if not v:
        return None
    return v[max(0, math.ceil(pct * len(v)) - 1)]


def exam_cases(cases: Iterable[dict]) -> list[dict]:
    return [c for c in cases if c.get("holdout")]


def passed(row: dict) -> bool | None:
    return C.passed(row)


def group(rows: list[dict]) -> dict:
    """The figures for a set of rows: counts, pass rate in points, verdicts,
    ops per turn and latency percentiles."""
    scored = [r for r in rows if passed(r) is not None]
    n_pass = sum(1 for r in scored if passed(r))
    ops = [len(r.get("ops") or []) for r in rows if r.get("status") == "ok"]
    lat = [r.get("latency_ms") or 0 for r in rows]
    return {"n": len(rows), "scored": len(scored), "passed": n_pass,
            "pass_rate": round(100.0 * n_pass / len(scored), 1) if scored else None,
            "errors": sum(1 for r in rows if r.get("status") != "ok"),
            "verdicts": dict(sorted(Counter(str(r.get("verdict")) for r in rows).items())),
            "ops_per_turn": {"mean": round(sum(ops) / len(ops), 2) if ops else None, "p50": percentile(ops, 0.5),
                             "max": max(ops) if ops else None},
            "latency_p50_ms": percentile(lat, 0.5), "latency_p95_ms": percentile(lat, 0.95)}


def usage_of(rows: list[dict]) -> dict:
    keys = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "calls")
    out = dict.fromkeys(keys, 0)
    for r in rows:
        u = r.get("usage") or {}
        for k in keys:
            out[k] += int(u.get(k) or 0)
    return out


def _container_of(row: dict) -> str:
    return str((row.get("meta") or {}).get("section") or "?")


def _field_type_of(row: dict) -> str:
    return str((row.get("meta") or {}).get("field_type") or "-")


def summarize(rows: list[dict]) -> dict:
    out = {"overall": group(rows), "hybrid": C.hybrid_summary(rows), "by_kind": {}, "by_container": {}, "by_field_type": {}}
    for k in sorted({str((r.get("meta") or {}).get("kind")) for r in rows}, key=lambda k: (KINDS.index(k) if k in KINDS else 99, k)):
        out["by_kind"][k] = group([r for r in rows if str((r.get("meta") or {}).get("kind")) == k])
    for key, fn in (("by_container", _container_of), ("by_field_type", _field_type_of)):
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[fn(r)].append(r)
        out[key] = {g: group(rs) for g, rs in sorted(groups.items())}
    return out


def build_report(rows: list[dict], *, label: str, route: str | None, manifest_hash: str | None, started: str,
                 finished: str | None = None, rule_version: str = RULE_VERSION, samples: int = 1,
                 cases: int | None = None, engine_fingerprint: str | None = None,
                 fields_context: bool | None = None, fields_record: dict | None = None) -> dict:
    if fields_context is None:  # a rescore: what the rows themselves say
        fields_context = any(r.get("fields_context") or "candidate_fields" in r for r in rows)
    for r in rows:  # passk.py's contract: results[] rows carry `passed` and `kind` at the top level
        r["passed"] = passed(r)
        r.setdefault("kind", (r.get("meta") or {}).get("kind"))
    return {"label": label, "route": route, "manifest_hash": manifest_hash, "rule_version": rule_version,
            "engine_fingerprint": engine_fingerprint, "fields_context": fields_context, "samples": samples, "cases": cases if cases is not None else len({r.get("case_id") for r in rows}),
            **{k: v for k, v in (fields_record or {}).items() if k in ("fields_context_by", "route_info")},
            "started_at": started, "finished_at": finished or C.utc_now(), "summary": summarize(rows),
            "usage": usage_of(rows), "results": rows}


def run_exam(cases: list[dict], client: C.Client, engine: C.Engine, *, label: str, route: str | None = None,
             samples: int = 1, verifier: Callable[[dict, dict], Any] | None = None, assets: Iterable[dict] = (),
             out_dir: Path = DEFAULT_RUNS, tools: list | None = None, progress: Callable[[str], None] | None = None,
             rule_version: str = RULE_VERSION, fields: Any | None = None,
             fields_record: dict | None = None) -> tuple[Path, dict]:
    """Ask every exam case `samples` times and save the run. Returns (path, report).
    `fields_record` (collect.resolve_fields) says who decided `fields`; a
    turn.done that contradicts it raises collect.FieldsContextMismatch."""
    exam = exam_cases(cases)
    started = C.utc_now()
    rows: list[dict] = []
    assets = list(assets)
    for k in range(1, samples + 1):
        for i, case in enumerate(exam, 1):
            row = C.run_case(case, client, engine, route=route, sample=k, assets=assets, verifier=verifier, tools=tools,
                             fields=fields)
            rows.append(row)
            if progress:
                esc = f" · escalated→{row['escalated'].get('to')} ({row['escalated'].get('reason')})" if row.get("escalated") else ""
                progress(f"[{k}/{samples} {i}/{len(exam)}] {row['id']} {row.get('verdict') or row.get('code')}{esc} "
                         f"· {row['latency_ms'] / 1000:.1f}s")
    hashes = {c.get("manifest_hash") for c in exam}
    fp = C.engine_fingerprint_of(engine)
    report = build_report(rows, label=label, route=route, manifest_hash=next(iter(hashes)) if len(hashes) == 1 else None,
                          started=started, rule_version=rule_version, samples=samples, cases=len(exam),
                          engine_fingerprint=fp, fields_context=fields is not None, fields_record=fields_record)
    stale = sorted({str(c.get("id")) for c in exam if fp and c.get("engine_fingerprint") and c.get("engine_fingerprint") != fp})
    if stale:
        report["stale_cases"] = stale
        if progress:
            progress(f"warning: {len(stale)} exam case(s) were measured with another engine than {fp[:12]} — "
                     f"run `souschef-lab cases remeasure`")
    path = save(report, out_dir, label)
    return path, report


def save(report: dict, out_dir: Path, label: str, stamp: str | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval-{slug(label)}-{stamp or utc_stamp()}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_run(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(a: dict, b: dict) -> dict:
    """Per-sample agreement of two runs: the same `id` ("<case>#<k>") in both;
    flips = passed in one only."""
    ra = {str(r.get("id")): r for r in a.get("results", [])}
    rb = {str(r.get("id")): r for r in b.get("results", [])}
    shared = sorted(set(ra) & set(rb))
    per_case = []
    agree = a_only = b_only = 0
    for sid in shared:
        pa, pb = passed(ra[sid]), passed(rb[sid])
        same = pa == pb
        agree += same
        a_only += pa is True and pb is not True
        b_only += pb is True and pa is not True
        per_case.append({"id": sid, "kind": (ra[sid].get("meta") or {}).get("kind"), "a": ra[sid].get("verdict"),
                         "b": rb[sid].get("verdict"), "agree": same})
    return {"a": a.get("label"), "b": b.get("label"), "shared": len(shared), "only_a": sorted(set(ra) - set(rb)),
            "only_b": sorted(set(rb) - set(ra)), "agreement": [agree, len(shared)],
            "agreement_rate": round(100.0 * agree / len(shared), 1) if shared else None,
            "flips": {"a_only_pass": a_only, "b_only_pass": b_only},
            "summary": {"a": a.get("summary", {}).get("overall"), "b": b.get("summary", {}).get("overall")},
            "per_case": per_case, "differ": [c for c in per_case if not c["agree"]]}


def rescore(run: dict, cases: Iterable[dict], verifier: Callable[[dict, dict], Any],
            rule_version: str = RULE_VERSION) -> dict:
    """The same rows re-verified under the current rules; the run's label
    gets "-rescored" and the rule version is updated."""
    by_id = {str(c.get("id")): c for c in cases}
    rows = []
    for r in run.get("results", []):
        row = dict(r)
        case = by_id.get(str(r.get("case_id")))
        if case is None:
            row["verdict"], row["reasons"] = None, ["no_case"]
        else:
            row.update(C._verdict_fields(verifier(case, row)))
        rows.append(row)
    return build_report(rows, label=f"{run.get('label')}-rescored", route=run.get("route"),
                        manifest_hash=run.get("manifest_hash"), started=run.get("started_at") or C.utc_now(),
                        rule_version=rule_version, samples=int(run.get("samples") or 1), cases=run.get("cases"),
                        engine_fingerprint=run.get("engine_fingerprint"))


def format_summary(report: dict) -> str:
    s = report["summary"]
    lines = [f"{report.get('label')} · route {report.get('route') or '(default)'} · {report.get('cases')} exam cases × "
             f"{report.get('samples')} · rules {report.get('rule_version')}"]
    o = s["overall"]
    lines.append(f"  overall  {o['passed']}/{o['scored']} pass ({o['pass_rate']} pts) · errors {o['errors']} · "
                 f"ops/turn {o['ops_per_turn']['mean']} · p50 {o['latency_p50_ms']} ms · p95 {o['latency_p95_ms']} ms")
    for k, g in s["by_kind"].items():
        lines.append(f"  {k:<14} {g['passed']}/{g['scored']} ({g['pass_rate']})")
    lines.append("  verdicts " + ", ".join(f"{k} {v}" for k, v in o["verdicts"].items()))
    h = s.get("hybrid")
    if h and h.get("fallback_seen"):
        lines.append("  hybrid   " + C.format_hybrid(h))
    return "\n".join(lines)


def format_compare(cmp: dict) -> str:
    lines = [f"A={cmp['a']} B={cmp['b']}: {cmp['agreement'][0]}/{cmp['agreement'][1]} samples agree "
             f"({cmp['agreement_rate']}%) · A-only pass {cmp['flips']['a_only_pass']} · B-only pass "
             f"{cmp['flips']['b_only_pass']}"]
    for c in cmp["differ"]:
        lines.append(f"  {c['id']} ({c['kind']}): A {c['a']} · B {c['b']}")
    if cmp["only_a"] or cmp["only_b"]:
        lines.append(f"  in one run only: A {len(cmp['only_a'])}, B {len(cmp['only_b'])}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="souschef-lab eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, default=C.DEFAULT_CASES)
    p.add_argument("--url", default=os.environ.get("SOUS_CHEF_URL", "http://127.0.0.1:8090"))
    p.add_argument("--key", default=os.environ.get("SOUS_CHEF_KEY", ""))
    p.add_argument("--route", "--model", dest="route")
    p.add_argument("--label", help="run label (default: the route or 'default')")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--manifest", type=Path, default=C.DEFAULT_MANIFEST)
    p.add_argument("--assets", type=Path, default=C.DEFAULT_ASSETS)
    p.add_argument("--engine", type=Path, default=os.environ.get("ENGINE"))
    p.add_argument("--out-dir", type=Path, default=DEFAULT_RUNS)
    p.add_argument("--sse", action="store_true")
    p.add_argument("--timeout", type=float, default=C.DEFAULT_TIMEOUT_S)
    p.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"), help="two saved runs")
    p.add_argument("--rescore", type=Path, metavar="RUN", help="a saved run to re-verify")
    p.add_argument("--fields-context", choices=C.FIELDS_FLAGS, default="auto",
                   help="whether the route sends the candidate-fields block: auto = what the service's /version says "
                        "(default), on / off are refused when it says otherwise (see collect --help; on needs go)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.compare:
        cmp = compare(load_run(args.compare[0]), load_run(args.compare[1]))
        print(format_compare(cmp))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        path = args.out_dir / f"compare-{slug(str(cmp['a']))}-vs-{slug(str(cmp['b']))}-{utc_stamp()}.json"
        path.write_text(json.dumps(cmp, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"saved {path}")
        return 0
    if args.rescore:
        cases = C.load_cases(args.cases)
        report = rescore(load_run(args.rescore), cases, C.default_verifier())
        path = save(report, args.out_dir, report["label"])
        print(format_summary(report))
        print(f"saved {path}")
        return 0
    if args.samples < 1:
        print("--samples must be at least 1", file=sys.stderr)
        return 2
    if not args.key:
        print("a service key is required (--key or $SOUS_CHEF_KEY)", file=sys.stderr)
        return 2
    cases = C.load_cases(args.cases)
    exam = exam_cases(cases)
    if not exam:
        print(f"no held-out cases in {args.cases}", file=sys.stderr)
        return 2
    try:
        client, _ = C.connect(args.url, args.key, args.manifest, exam, sse=args.sse, timeout_s=args.timeout, build_id="lab")
    except (C.ServiceError, OSError, json.JSONDecodeError) as e:
        print(f"cannot start: {e}", file=sys.stderr)
        return 1
    label = args.label or args.route or "default"
    try:
        engine = C.default_engine(args.engine, args.manifest)
    except RuntimeError as e:  # engine.EngineError
        print(f"cannot start: {e}", file=sys.stderr)
        return 1
    try:
        fields, fields_record = C.resolve_fields(args.fields_context, args.manifest, client, args.route)
    except RuntimeError as e:  # C.FieldsContextMismatch, fields.FieldsError
        print(f"cannot start: {e}", file=sys.stderr)
        return 1
    print(f"fields_context {'on' if fields is not None else 'off'} for {args.route or 'the default route'} "
          f"(says {'the service' if fields_record['fields_context_by'] == 'service' else '--fields-context'})")
    try:
        path, report = run_exam(exam, client, engine, label=label, route=args.route, samples=args.samples,
                                verifier=C.default_verifier(engine), assets=C.load_assets(args.assets),
                                out_dir=args.out_dir, progress=print, fields=fields, fields_record=fields_record)
    except KeyboardInterrupt:
        print("stopped (Ctrl-C): nothing saved", file=sys.stderr)
        return C.INTERRUPTED
    except C.FieldsContextMismatch as e:
        print(f"stopped: {e}\nnothing saved", file=sys.stderr)
        return 1
    print(format_summary(report))
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
