"""retrain_check — should the in-house model be retrained? (docs/LAB.md §7)

Inputs: the current manifest (a file, or `GET /v1/manifests/{hash}?body=1`
from the service), the manifest the model was trained on (the hash and
registration time in `dataset-manifest.json`, the body from a file or the
service), the latest exam run (`eval.py`), and the last days of real traces
(exported JSONL rows of the service's `traces` table: `{turn_id, context,
events, ops, feedback}`; Postgres via `DATABASE_URL` when psycopg is
installed — optional, never needed by the tests).

Triggers (any → exit 1):
  manifest   ≥ 10 fields added, removed or retyped since the training manifest
  shape      a new section type, global panel/details panel, or op name
  drift      the real-kept pass rate (traced turns whose feedback.kept_paths
             cover every written path and whose `undone` is not true) is
             more than 10 points under the exam pass rate — with fewer than
             50 real turns that have writes the check is "insufficient
             data" and the exit is 2 (unless another trigger fired)
  age        ≥ 30 days since the training manifest was registered
  engine     the browser engine's fingerprint (engine.engine_fingerprint:
             the import closure of headless.mjs + HEAD) differs from the one the
             training set was built with (`dataset-manifest.json
             engine_fingerprint`) — the expected diffs behind the training
             labels may have moved; re-measure the cases and re-verify
             (checked only when --engine / --engine-fingerprint is given)

Exit 0 = no retrain. The output is Slack-ready plain text (one bullet per
trigger with its numbers); nothing here posts anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MANIFEST_FIELDS_THRESHOLD = 10
PASS_GAP_POINTS = 10.0
MIN_REAL_TURNS = 50
MAX_AGE_DAYS = 30
CONTAINER_GROUPS = ("sections", "globalPanels", "pages", "details")
PASS_VERDICTS = frozenset(("pass", "clarify_ok", "unsupported_ok"))
# README §2: the browser ops and the tools; anything else in a trace is a new op name.
KNOWN_OPS = frozenset(("set_fields", "edit_list", "reset", "restore_default", "set_page_layout", "reset_pinned_section",
                       "pick_asset", "lookup_fields", "describe_target", "report_unsupported"))
EXIT_OK, EXIT_RETRAIN, EXIT_INSUFFICIENT = 0, 1, 2


# --- manifest ----------------------------------------------------------------------------------------------------------------

def containers(manifest: dict) -> dict[str, dict]:
    """`<group>.<name>` → container, plus `branding`."""
    out: dict[str, dict] = {}
    for grp in CONTAINER_GROUPS:
        for name, c in (manifest.get(grp) or {}).items():
            if isinstance(c, dict):
                out[f"{grp}.{name}"] = c
    if isinstance(manifest.get("branding"), dict):
        out["branding"] = manifest["branding"]
    return out


def field_index(manifest: dict) -> dict[str, str]:
    """Every field's type by `<container>.<key>`; list item fields as
    `<container>.<list>[].<key>` (the registry's 2,094 on the snapshot)."""
    out: dict[str, str] = {}
    for cname, c in containers(manifest).items():
        for key, f in (c.get("fields") or {}).items():
            out[f"{cname}.{key}"] = str((f or {}).get("type") or "")
        for lname, lst in (c.get("lists") or {}).items():
            for key, f in ((lst or {}).get("item") or {}).items():
                out[f"{cname}.{lname}[].{key}"] = str((f or {}).get("type") or "")
    return out


def op_names(manifest: dict) -> set[str]:
    """Every list op the manifest allows (`lists[*].ops`)."""
    out: set[str] = set()
    for c in containers(manifest).values():
        for lst in (c.get("lists") or {}).values():
            out.update(str(o) for o in ((lst or {}).get("ops") or []))
    return out


def manifest_diff(old: dict, new: dict) -> dict:
    fo, fn = field_index(old), field_index(new)
    added = sorted(set(fn) - set(fo))
    removed = sorted(set(fo) - set(fn))
    retyped = sorted(k for k in set(fo) & set(fn) if fo[k] != fn[k])
    sections = sorted(set(new.get("sections") or {}) - set(old.get("sections") or {}))
    panels = sorted((set(new.get("globalPanels") or {}) - set(old.get("globalPanels") or {}))
                    | (set(new.get("details") or {}) - set(old.get("details") or {})))
    ops = sorted(op_names(new) - op_names(old))
    return {"added": added, "removed": removed, "retyped": retyped, "changed": len(added) + len(removed) + len(retyped),
            "new_sections": sections, "new_panels": panels, "new_ops": ops,
            "fields": {"old": len(fo), "new": len(fn)}}


# --- traces ------------------------------------------------------------------------------------------------------------------

def written_paths(trace: dict) -> set[str]:
    """The paths a turn's browser results reported written (`tool.result`
    events' `data.written`; the collector's row shape `written_paths` and
    `ops[].written` are read too)."""
    out: set[str] = set()
    for ev in trace.get("events") or []:
        if ev.get("event") == "tool.result" or ev.get("type") == "tool.result":
            data = ev.get("data") or {}
            out.update(str(p) for p in (data.get("written") or []))
    out.update(str(p) for p in (trace.get("written_paths") or []))
    return out


def trace_ops(trace: dict) -> set[str]:
    out: set[str] = set()
    for ev in trace.get("events") or []:
        if (ev.get("event") or ev.get("type")) == "tool.call":
            name = (ev.get("data") or {}).get("name")
            if name:
                out.add(str(name))
    for op in trace.get("ops") or []:
        if isinstance(op, dict) and op.get("op"):
            out.add(str(op["op"]))
    return out


def real_kept(traces: Iterable[dict]) -> dict:
    """{turns, with_writes, kept, rate}: a turn is kept when feedback.kept_paths
    cover every written path and `undone` is not true; turns without writes
    are not scored."""
    turns = with_writes = kept = 0
    for t in traces:
        turns += 1
        written = written_paths(t)
        if not written:
            continue
        with_writes += 1
        fb = t.get("feedback") or {}
        kept_paths = set(str(p) for p in (fb.get("kept_paths") or []))
        if not fb.get("undone") and written <= kept_paths:
            kept += 1
    return {"turns": turns, "with_writes": with_writes, "kept": kept,
            "rate": round(100.0 * kept / with_writes, 1) if with_writes else None}


def exam_pass_rate(run: dict) -> float | None:
    return ((run.get("summary") or {}).get("overall") or {}).get("pass_rate")


def parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def training_registered_at(dataset_manifest: dict) -> datetime | None:
    for k in ("registered_at", "manifest_first_seen", "first_seen", "trained_at", "built_at", "at"):
        dt = parse_time(dataset_manifest.get(k))
        if dt is not None:
            return dt
    return None


def training_hash(dataset_manifest: dict) -> str | None:
    for k in ("manifest_hash", "manifestHash", "hash"):
        if dataset_manifest.get(k):
            return str(dataset_manifest[k])
    return None


# --- the decision --------------------------------------------------------------------------------------------------------------

def check(*, current: dict | None, training: dict | None, dataset_manifest: dict | None, exam_run: dict | None,
          traces: list[dict] | None, now: datetime | None = None, thresholds: dict | None = None,
          engine_fingerprint: str | None = None) -> dict:
    """Every trigger evaluated; `exit_code` 0/1/2 and `lines` (Slack bullets)."""
    th = {"fields": MANIFEST_FIELDS_THRESHOLD, "gap": PASS_GAP_POINTS, "min_turns": MIN_REAL_TURNS, "days": MAX_AGE_DAYS,
          **(thresholds or {})}
    now = now or datetime.now(timezone.utc)
    triggers: list[str] = []
    insufficient: list[str] = []
    notes: list[str] = []
    out: dict[str, Any] = {"checked_at": now.isoformat()}

    if current is not None and training is not None:
        d = manifest_diff(training, current)
        out["manifest_diff"] = {k: v for k, v in d.items() if k not in ("added", "removed", "retyped")} | {
            "sample": {"added": d["added"][:5], "removed": d["removed"][:5], "retyped": d["retyped"][:5]}}
        if d["changed"] >= th["fields"]:
            triggers.append(f"manifest: {d['changed']} fields changed since training (+{len(d['added'])} "
                            f"-{len(d['removed'])} retyped {len(d['retyped'])}; threshold {th['fields']})")
        else:
            notes.append(f"manifest: {d['changed']} fields changed (under {th['fields']})")
        shape = []
        if d["new_sections"]:
            shape.append(f"section types {', '.join(d['new_sections'])}")
        if d["new_panels"]:
            shape.append(f"panels {', '.join(d['new_panels'])}")
        if d["new_ops"]:
            shape.append(f"list ops {', '.join(d['new_ops'])}")
        if shape:
            triggers.append("shape: new " + "; ".join(shape))
    else:
        insufficient.append("manifest: current or training manifest missing")

    if traces is not None:
        new_ops = sorted({o for t in traces for o in trace_ops(t)} - KNOWN_OPS - (op_names(current) if current else set()))
        if new_ops:
            triggers.append(f"shape: traces carry op names the training set never saw: {', '.join(new_ops)}")

    exam = exam_pass_rate(exam_run) if exam_run else None
    if traces is not None and exam is not None:
        rk = real_kept(traces)
        out["real_kept"] = rk
        out["exam_pass_rate"] = exam
        if rk["with_writes"] < th["min_turns"]:
            insufficient.append(f"drift: only {rk['with_writes']} real turns with writes (need {th['min_turns']})")
        else:
            gap = exam - rk["rate"]
            if gap > th["gap"]:
                triggers.append(f"drift: real-kept pass {rk['rate']} pts vs exam {exam} pts "
                                f"({gap:.1f} under; threshold {th['gap']:g}; {rk['kept']}/{rk['with_writes']} turns kept)")
            else:
                notes.append(f"drift: real-kept {rk['rate']} pts vs exam {exam} pts ({gap:+.1f}), "
                             f"{rk['with_writes']} turns")
    else:
        insufficient.append("drift: traces or exam run missing")

    if dataset_manifest is not None:
        at = training_registered_at(dataset_manifest)
        if at is None:
            insufficient.append("age: dataset-manifest.json has no registration time")
        else:
            days = (now - at).total_seconds() / 86400
            out["age_days"] = round(days, 1)
            if days >= th["days"]:
                triggers.append(f"age: training manifest registered {days:.0f} days ago (≥ {th['days']})")
            else:
                notes.append(f"age: {days:.0f} days since training (under {th['days']})")
        th_hash, cur_hash = training_hash(dataset_manifest), (current or {}).get("_hash")
        if th_hash and cur_hash and th_hash != cur_hash:
            notes.append(f"hashes: trained on {th_hash[:12]}…, current {cur_hash[:12]}…")
    else:
        insufficient.append("age: dataset-manifest.json missing")

    if engine_fingerprint:
        trained_fp = str((dataset_manifest or {}).get("engine_fingerprint") or "")
        out["engine_fingerprint"] = {"current": engine_fingerprint, "training": trained_fp or None}
        if not trained_fp:
            insufficient.append("engine: dataset-manifest.json carries no engine_fingerprint")
        elif trained_fp != engine_fingerprint:
            triggers.append(f"engine: the browser engine changed since training (fingerprint {trained_fp[:12]}… → "
                            f"{engine_fingerprint[:12]}…); re-measure the cases (`cases remeasure`) and re-verify")
        else:
            notes.append(f"engine: unchanged since training ({engine_fingerprint[:12]}…)")

    code = EXIT_RETRAIN if triggers else (EXIT_INSUFFICIENT if insufficient else EXIT_OK)
    out.update({"triggers": triggers, "insufficient": insufficient, "notes": notes, "exit_code": code})
    out["text"] = slack_text(out)
    return out


def slack_text(result: dict) -> str:
    code = result["exit_code"]
    head = {EXIT_RETRAIN: "Sous Chef in-house model: RETRAIN recommended",
            EXIT_INSUFFICIENT: "Sous Chef in-house model: no retrain trigger, but some checks lack data",
            EXIT_OK: "Sous Chef in-house model: no retrain needed"}[code]
    lines = [head]
    lines += [f"• {t}" for t in result["triggers"]]
    lines += [f"• (insufficient data) {t}" for t in result["insufficient"]]
    lines += [f"• ok — {n}" for n in result["notes"]]
    return "\n".join(lines)


# --- inputs ------------------------------------------------------------------------------------------------------------------

def read_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path | None) -> list[dict] | None:
    if path is None:
        return None
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def fetch_manifest(url: str, key: str, hsh: str) -> dict:
    """`GET /v1/manifests/{hash}?body=1` → the manifest body (with `_hash`)."""
    from urllib import request as urlrequest
    req = urlrequest.Request(f"{url.rstrip('/')}/v1/manifests/{hsh}?body=1")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Accept", "application/json")
    with urlrequest.urlopen(req, timeout=60) as resp:
        doc = json.loads(resp.read().decode("utf-8"))
    body = doc.get("body") or {}
    body["_hash"] = hsh
    return body


def traces_from_postgres(dsn: str, days: int) -> list[dict]:
    """The last `days` of traces via psycopg (optional dependency)."""
    try:
        import psycopg  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise SystemExit("reading traces from Postgres needs psycopg (pip install psycopg[binary]) or --traces-jsonl") from e
    rows = []
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:  # pragma: no cover
        cur.execute("select turn_id, context, events, ops, feedback, created from traces where created >= now() - "
                    "make_interval(days => %s) order by created", (days,))
        for turn_id, context, events, ops, feedback, created in cur:
            rows.append({"turn_id": turn_id, "context": context, "events": events, "ops": ops, "feedback": feedback,
                         "created": created.isoformat()})
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="souschef-lab retrain-check", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, help="current manifest JSON (else --url + --hash)")
    p.add_argument("--hash", help="current manifest hash to fetch from the service")
    p.add_argument("--training-manifest", type=Path, help="the manifest the model trained on (else fetched by the "
                                                         "dataset-manifest's hash)")
    p.add_argument("--dataset-manifest", type=Path, help="dataset-manifest.json (training hash, registration time)")
    p.add_argument("--exam-run", type=Path, help="the latest runs/eval-*.json")
    p.add_argument("--traces-jsonl", type=Path, help="exported traces rows")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"), help="read traces from Postgres instead")
    p.add_argument("--days", type=int, default=7, help="days of traces to read from Postgres (default: 7)")
    p.add_argument("--url", default=os.environ.get("SOUS_CHEF_URL", ""))
    p.add_argument("--key", default=os.environ.get("SOUS_CHEF_KEY", ""))
    p.add_argument("--now", help="ISO time to evaluate at (tests)")
    p.add_argument("--engine", nargs="?", const="", type=str,
                   help="compare the engine's fingerprint with the training set's (a headless.mjs path, or bare for the usual discovery)")
    p.add_argument("--engine-fingerprint", help="the current engine fingerprint, when the engine is not on this machine")
    p.add_argument("--json", action="store_true", help="print the result as JSON instead of Slack text")
    return p


def current_engine_fingerprint(args: argparse.Namespace) -> str | None:
    """--engine-fingerprint as given, else the fingerprint of --engine (bare
    --engine = the usual discovery); None when neither is asked for."""
    if args.engine_fingerprint:
        return str(args.engine_fingerprint)
    if args.engine is None:
        return None
    from .engine import engine_fingerprint  # stdlib only; the engine is never run here
    ident = engine_fingerprint(args.engine or None)
    return (ident or {}).get("fingerprint")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        current = read_json(args.manifest)
        if current is None and args.hash and args.url:
            current = fetch_manifest(args.url, args.key, args.hash)
        dataset_manifest = read_json(args.dataset_manifest)
        training = read_json(args.training_manifest)
        th = training_hash(dataset_manifest or {})
        if training is None and th and args.url:
            training = fetch_manifest(args.url, args.key, th)
        exam_run = read_json(args.exam_run)
        traces = read_jsonl(args.traces_jsonl)
        if traces is None and args.database_url:
            traces = traces_from_postgres(args.database_url, args.days)
    except (OSError, json.JSONDecodeError) as e:
        print(f"cannot read inputs: {e}", file=sys.stderr)
        return EXIT_INSUFFICIENT
    result = check(current=current, training=training, dataset_manifest=dataset_manifest, exam_run=exam_run,
                   traces=traces, now=parse_time(args.now), engine_fingerprint=current_engine_fingerprint(args))
    print(json.dumps(result, ensure_ascii=False, indent=1) if args.json else result["text"])
    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
