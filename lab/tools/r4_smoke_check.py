"""r4_smoke_check — the verdict of `collect.sh smoke-check` (COLLECT.md step 3).

    python tools/r4_smoke_check.py <smoke-check.json from r4dataset --render-check> <smoke.ids>

Exit 0 only when every smoke case was answered and rendered from the STORED
conversation with no block drift and no render error; 1 otherwise (a check
that rendered nothing never passes).
"""
from __future__ import annotations

import json
import sys


def problems_of(d: dict, want: set[str]) -> list[str]:
    got = {str(r.get("case_id")) for r in d.get("by_case") or []}
    out = []
    if not want:
        out.append("no smoke cases listed")
    if got != want:
        out.append(f"answered smoke cases {sorted(got)} != the smoke's {sorted(want)}")
    if d.get("rendered") != len(want) or d.get("ok_rows") != d.get("rows"):
        out.append(f"rendered {d.get('rendered')} of {len(want)} smoke cases "
                   f"({(d.get('rows') or 0) - (d.get('ok_rows') or 0)} unanswered)")
    if d.get("block_drift_rows"):
        out.append(f"{d['block_drift_rows']} row(s) with block drift")
    if set(d.get("render_sources") or {}) - {"conversation"}:
        out.append(f"rows not rendered from the stored conversation: {d.get('render_sources')}")
    if d.get("errors"):
        out.append(f"render errors {d['errors']}")
    return out


def main(argv: list[str]) -> int:
    d = json.load(open(argv[0], encoding="utf-8"))
    want = {line.strip() for line in open(argv[1], encoding="utf-8") if line.strip()}
    print(json.dumps({k: d.get(k) for k in ("rows", "ok_rows", "rendered", "render_sources", "block_drift_rows", "errors")},
                     indent=1))
    for r in d.get("by_case") or []:
        print(f"  {r.get('case_id')}: verdict {r.get('verdict')}, source {r.get('source')}, drift {r.get('block_drift')}"
              + (f", ERROR {r['error']}" if r.get("error") else ""))
    problems = problems_of(d, want)
    print("smoke-check: " + ("OK" if not problems else "FAILED — " + "; ".join(problems)))
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
