"""probe_recheck — re-score a past new-key probe offline: which "dodged" keys
were false alarms of the harness rather than the model's.

    python -m souschef_lab.probe_recheck --manifest runs/<id>/prepare/manifest.json \\
        --report drift/<run>/report.json --rows drift/<run>/exam/P-1.jsonl --rows drift/<run>/exam/P-2.jsonl \\
        [--rows eval-probe-rows.jsonl …] [--engine] [--json out.json]

Until r6 the probe (drift arm P, the eval's `probe` set) ran every row on ONE
shared state — the vintage home of the exam — while the generator's fallback
wrote rows for sections that state lacks (contentBlock-2/-3/-features,
newsletter). Those rows were refused by the engine whatever the model did. This
check rebuilds the rows that run asked (`legacy_probe_rows`, a frozen copy of
drift.probe_rows before r6: same generator, seed and selection), pairs them
with the stored results by (probe key, row index) — the ids were salted by
Python's hash() and differ between processes; the pairing is confirmed on every
row that wrote something — and marks a row UNPASSABLE when its target section
is not on the shared state or (`--engine`, node + COOKIE_CUTTER_DIR) the engine
refuses its own expected ops there. Each key is then scored on its passable
rows only. Verdicts: `false_alarm` (dodged as run, not on its passable rows),
`unmeasured` (dodged as run, no passable row at all), `dodged` (dodged either
way), `ok`. It also lists what the fixed probe would ask on the same inputs
(drift.probe_rows with the engine). No model, no network.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .manifest import Manifest

DODGED_RATE = 0.5  # lab/pipeline/thresholds.json dodged_pass_rate


def legacy_probe_rows(new_m: Manifest, keys: list[str], *, state: dict, per_key: int = 4, max_keys: int = 40,
                      seed: int = 0) -> list[dict]:
    """drift.probe_rows as r5 ran it (commit 67eb6bd, no engine): the shared state's cases first, the default home's
    for what it lacks; rows carry no state of their own. `id` is `<probe key>#<i>` (the run's were salted)."""
    from collections import defaultdict as dd

    from .cases import Generator
    from .drift import PROBE_KINDS_LIST, PROBE_KINDS_STATE
    from .exam import touches_of
    from .examrebase import ManifestIndex, address_of

    keys = list(keys)[:max_keys]
    want = set(keys)
    idx = ManifestIndex(new_m)
    cands: dict[str, list] = dd(list)
    seen: set = set()

    def consider(cases) -> None:
        for c in cases:
            addrs = set()
            for op in c.expected.get("ops") or []:
                for t in touches_of(op):
                    a, found = address_of(new_m, idx, {"target": t["target"], "key": t["key"], "list": t["list"]})
                    if a and found:
                        addrs.add(a)
                    if t["list"]:
                        la = a.split("[].", 1)[0] + "[]" if a and "[]." in a else None
                        if la:
                            addrs.add(la)
            for a in sorted(addrs & want):
                if (a, c.id) not in seen and len(cands[a]) < 4 * per_key:
                    seen.add((a, c.id))
                    cands[a].append(c)

    try:
        consider(Generator(new_m, seed=seed, template="vintage", base_state=state).all_cases(PROBE_KINDS_STATE))
    except (IndexError, KeyError, ValueError):
        pass
    if any(len(cands[k]) < per_key for k in keys):
        g2 = Generator(new_m, seed=seed, template="vintage")
        for kinds in (PROBE_KINDS_STATE, PROBE_KINDS_LIST):
            try:
                consider(g2.all_cases(kinds))
            except (IndexError, KeyError, ValueError):
                continue
    rows = []
    for k in keys:
        by_kind: dict[str, list] = dd(list)
        for c in cands[k]:
            by_kind[c.kind].append(c)
        chosen: list = []
        while len(chosen) < per_key and any(by_kind.values()):
            for kind in sorted(by_kind):
                if by_kind[kind] and len(chosen) < per_key:
                    chosen.append(by_kind[kind].pop(0))
        for i, c in enumerate(chosen):
            rows.append({"id": f"{k}#{i}", "probe_key": k, "index": i, "kind": c.kind, "target": c.target,
                         "request": c.request, "ops": [o for o in c.expected.get("ops") or [] if o.get("op") != "pick_asset"]})
    return rows


def passable(row: dict, shared: dict, engine: Any = None) -> tuple[bool, str | None]:
    """Could any answer pass this legacy row on the shared state it ran on?"""
    from . import probe as PR
    if not PR.target_present(row, shared):
        return False, f"section_missing:{(row.get('target') or {}).get('sectionId')}"
    if engine is not None:
        ok, why = PR.accepted_on(SimpleNamespace(site_state=shared, browser_ops=row["ops"]), engine)
        if not ok:
            return False, why
    return True, None


def _target_of_written(w: str) -> dict | None:
    head = str(w).rsplit("} ", 1)[0] + "}"
    try:
        return json.loads(head)
    except ValueError:
        return None


def recheck(new_m: Manifest, keys: list[str], result_sets: dict[str, list[dict]], *, shared: dict, engine: Any = None,
            per_key: int = 4, dodged_rate: float = DODGED_RATE) -> dict:
    legacy = legacy_probe_rows(new_m, keys, state=shared, per_key=per_key)
    by_ki = {(r["probe_key"], r["index"]): r for r in legacy}
    verdict_rows = {}
    for r in legacy:
        ok, why = passable(r, shared, engine)
        verdict_rows[r["id"]] = {"passable": ok, "why": why, "target": r["target"], "kind": r["kind"]}
    out: dict[str, Any] = {"legacy_rows": len(legacy), "unpassable_rows": sum(not v["passable"] for v in verdict_rows.values()),
                           "rows": verdict_rows, "sets": {}}
    for name, results in result_sets.items():
        per: dict[str, dict] = defaultdict(lambda: {"n": 0, "pass": 0, "n_passable": 0, "pass_passable": 0, "unpassable": []})
        mismatched, unmatched = [], []
        for t in results:
            if t.get("status") != "ok":
                continue
            k = t.get("probe_key")
            try:
                i = int(str(t.get("id")).rsplit("-", 1)[1])
            except (IndexError, ValueError):
                unmatched.append(t.get("id"))
                continue
            lr = by_ki.get((k, i))
            if lr is None:
                unmatched.append(t.get("id"))
                continue
            for w in t.get("written") or []:
                tg = _target_of_written(w)
                if tg is not None and tg != lr["target"]:
                    mismatched.append(t.get("id"))
                    break
            v = verdict_rows[lr["id"]]
            p = per[k]
            p["n"] += 1
            p["pass"] += int(t.get("pass") or 0)
            if v["passable"]:
                p["n_passable"] += 1
                p["pass_passable"] += int(t.get("pass") or 0)
            else:
                p["unpassable"].append(v["why"])
        keys_out = {}
        for k, p in sorted(per.items()):
            rate = p["pass"] / p["n"] if p["n"] else None
            rate2 = p["pass_passable"] / p["n_passable"] if p["n_passable"] else None
            dodged = rate is not None and rate < dodged_rate
            dodged2 = rate2 is not None and rate2 < dodged_rate
            verdict = ("false_alarm" if dodged and rate2 is not None and not dodged2 else
                       "unmeasured" if dodged and rate2 is None else "dodged" if dodged else "ok")
            keys_out[k] = {"as_run": f"{p['pass']}/{p['n']}", "rate": None if rate is None else round(rate, 3),
                           "passable": f"{p['pass_passable']}/{p['n_passable']}",
                           "rate_passable": None if rate2 is None else round(rate2, 3),
                           "unpassable_why": sorted(set(p["unpassable"])), "verdict": verdict}
        out["sets"][name] = {"keys": keys_out, "pairing_mismatches": mismatched, "unmatched": unmatched}
    return out


def fixed_plan(new_m: Manifest, keys: list[str], *, shared: dict, engine: Any = None, per_key: int = 4) -> dict:
    """What the fixed probe (drift.probe_rows, rows on their own states, engine-checked) asks on the same inputs."""
    from . import drift as D
    stats: dict = {}
    rows, cov = D.probe_rows(new_m, keys, state=shared, per_key=per_key, engine=engine, stats=stats)
    return {"coverage": cov, "dropped": {k: v["dropped"] for k, v in stats.items() if v["dropped"]},
            "rows": [{"probe_key": r["meta"]["probe_key"], "kind": r["meta"]["kind"], "target": r["target"],
                      "request": r["request"]} for r in rows]}


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m souschef_lab.probe_recheck", description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, help="the deployed manifest the probe ran on (runs/<id>/prepare/manifest.json)")
    ap.add_argument("--report", required=True, help="the drift run's report.json (impact.new_keys, probe, verdict)")
    ap.add_argument("--rows", action="append", default=[], help="trimmed probe rows (drift exam/P-k.jsonl, eval rows of set probe)")
    ap.add_argument("--engine", action="store_true", help="also run each row's ops through the real engine (node + COOKIE_CUTTER_DIR)")
    ap.add_argument("--per-key", type=int, default=4)
    ap.add_argument("--json", help="write the whole result here")
    return _run(ap.parse_args(argv))


def _run(a: argparse.Namespace) -> int:
    from .drift import EXAM_SETS, home_context
    from .exam import load_set, vintage_state
    mpath = Path(a.manifest)
    m = Manifest(json.loads(mpath.read_text(encoding="utf-8")), mpath)
    rep = json.loads(Path(a.report).read_text(encoding="utf-8"))
    keys = list((rep.get("impact") or {}).get("new_keys") or [])
    exam_rows = [r for s in EXAM_SETS for r in load_set(s)]
    shared = vintage_state(mpath, home_context(exam_rows))  # the drift check's state (drift._measure: home of the exam)
    engine = None
    if a.engine:
        from .engine import PersistentEngine
        engine = PersistentEngine(manifest_path=mpath, size=1)
    try:
        sets = {Path(p).name: _read_jsonl(Path(p)) for p in a.rows}
        res = recheck(m, keys, sets, shared=shared, engine=engine, per_key=a.per_key)
        res["fixed_probe"] = fixed_plan(m, keys, shared=shared, engine=engine, per_key=a.per_key)
    finally:
        if engine is not None:
            engine.close()
    res["as_reported"] = {"dodged_keys": (rep.get("verdict") or {}).get("dodged_keys"), "probe": rep.get("probe"),
                          "unprobed_keys": rep.get("unprobed_keys")}
    res["manifest"] = m.hash
    print(f"manifest {m.hash[:8]}; {len(keys)} new keys; legacy rows {res['legacy_rows']}, unpassable on the shared state "
          f"{res['unpassable_rows']}{' (engine-checked)' if engine is not None else ' (section check only)'}")
    for name, s in res["sets"].items():
        print(f"\n{name}: pairing mismatches {len(s['pairing_mismatches'])}, unmatched {len(s['unmatched'])}")
        print(f"  {'key':<46} {'as run':>7} {'passable':>9}  verdict      unpassable because")
        for k, v in s["keys"].items():
            print(f"  {k.split('.', 1)[1]:<46} {v['as_run']:>7} {v['passable']:>9}  {v['verdict']:<12} {', '.join(v['unpassable_why'])}")
    fx = res["fixed_probe"]
    print("\nfixed probe (rows on their own states" + (", engine-checked" if engine is not None else "") + "): coverage")
    for k in keys:
        print(f"  {k.split('.', 1)[1]:<46} {fx['coverage'].get(k, 0)}" + (f"  dropped {fx['dropped'][k]}" if k in fx["dropped"] else ""))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
