"""r4 render parity against the REAL service binary (r4 PLAN L5; docs/LAB.md §5b).

Runs conversations through the deployed service's own binary (built from the
deployed commit), on a messages-compatible route `student` whose model is the
lab's scripted fake (tests/fake_model.py), collects them with r4collect (the
browser emulator + the snapshot engine) and checks that r4dataset renders, for
EVERY model call of the scored turn, exactly the request body the service
sent the student: system (rules + index), tools, the stable history (earlier
user turns stripped, thinking dropped), the current turn's context + the
follow-up-aware candidate-fields block, and the server-tool round trips
(lookup_fields, report_unsupported) that only the stored conversation knows.

    SOUS_CHEF_FIELDS_BIN=<f79753b fields CLI> SOUS_CHEF_PREFIX_BIN=<f79753b prefix CLI> \\
    python -m tests.r4_parity --binary <f79753b sous-chef> [--db-url postgres://…/souschef_lab]

Needs node + the cookie-cutter snapshot (COOKIE_CUTTER_DIR), a local Postgres
(the lab's :5433, database souschef_lab) and the binaries. No network, no model.
Exit 0 = every call byte-equal.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from souschef_lab import FIXTURES_DIR  # noqa: E402
from souschef_lab import collect as C  # noqa: E402
from souschef_lab import createpage as CP  # noqa: E402
from souschef_lab import fields as F  # noqa: E402
from souschef_lab import r4dataset as D  # noqa: E402
from souschef_lab import r4collect as RC  # noqa: E402
from souschef_lab import state as S  # noqa: E402
from souschef_lab.engine import PersistentEngine  # noqa: E402
from souschef_lab.manifest import load  # noqa: E402
from tests import capture_real as CR  # noqa: E402
from tests import fake_model as FM  # noqa: E402

MANIFEST = FIXTURES_DIR / "exam" / "manifest-real.json"
KEY = "test-key"


def msg(*blocks, stop=None) -> dict:
    m = {"message": {"content": list(blocks), "usage": {"input_tokens": 100, "output_tokens": 10}}}
    if stop:
        m["message"]["stop_reason"] = stop
    return m


def plan(m) -> list[dict]:
    """(case, steps) pairs: a 2-turn follow-up whose answer looks a field up
    first, an out-of-editor turn, and a single write."""
    st = S.build_state(m, template="vintage")
    extras = S.load_context_fixture()
    page = "home"
    hb = {"target": {"kind": "global", "panel": "HeaderSection"}, "values": {"announcementBar.message": "Free delivery over $60"}}
    follow = {"id": "r4-PAR-0001", "kind": "followup", "site_state": st, "template": "vintage", "manifest_hash": m.hash,
              "context": S.context_of(st, page, extras, m, text="Free delivery over $60"), "request": "Free delivery over $60",
              "history": [{"user": "change what the promo bar says", "page": page, "expect": "ask", "assistant": "", "ops": [], "calls": []}],
              "expected": {"ops": [{"op": "set_fields", **hb}], "diff": [], "keys": [{"target": hb["target"], "key": "announcementBar.message", "list": ""}],
                           "alt": [], "alt_exact": [], "out_of_scope": None},
              "score": {"mode": "keys"}, "holdout": False, "meta": {"family": "F2", "page": page, "view": None}}
    steps_follow = [
        msg({"type": "thinking", "thinking": "the merchant wants the bar changed"}, {"type": "text", "text": "Sure — what should the bar say?"}),
        msg({"type": "text", "text": "Let me find it."}, {"type": "tool_use", "id": "toolu_l1", "name": "lookup_fields", "input": {"query": "announcement bar message"}}),
        msg({"type": "tool_use", "id": "toolu_s1", "name": "set_fields", "input": hb}),
        msg({"type": "text", "text": "Done — the bar now says \"Free delivery over $60\". Save when you're happy."}),
    ]
    oos = {"id": "r4-PAR-0002", "kind": "unsupported", "site_state": st, "template": "vintage", "manifest_hash": m.hash,
           "context": S.context_of(st, page, extras, m, text="connect my domain prairieleaf.ca"), "request": "connect my domain prairieleaf.ca",
           "history": [], "expected": {"ops": [], "diff": [], "keys": [], "alt": [], "alt_exact": [],
                                       "out_of_scope": {"reason": "domain", "expected_where": ["domain"]}},
           "score": {"mode": "where"}, "holdout": False, "meta": {"family": "F3", "page": page, "view": None}}
    steps_oos = [
        msg({"type": "tool_use", "id": "toolu_u1", "name": "report_unsupported", "input": {"reason": "Domains are connected in Site Settings.", "where": "domain"}}),
        msg({"type": "text", "text": "Connecting a domain is done in your domain settings."}),
    ]
    bt = {"target": {"kind": "global", "panel": "ButtonsSection"}, "values": {"borderRadius": 4}}
    single = {"id": "r4-PAR-0003", "kind": "set_range", "site_state": st, "template": "vintage", "manifest_hash": m.hash,
              "context": S.context_of(st, page, extras, m, text="buttons nearly square, radius 4"), "request": "buttons nearly square, radius 4",
              "history": [], "expected": {"ops": [{"op": "set_fields", **bt}], "diff": [], "keys": [{"target": bt["target"], "key": "borderRadius", "list": ""}],
                                          "alt": [], "alt_exact": [], "out_of_scope": None},
              "score": {"mode": "keys"}, "holdout": False, "meta": {"family": "F1", "page": page, "view": None}}
    steps_single = [msg({"type": "tool_use", "id": "toolu_b1", "name": "set_fields", "input": bt}),
                    msg({"type": "text", "text": "Done — button corners are 4px now."})]
    return [(follow, steps_follow, 1), (oos, steps_oos, 0), (single, steps_single, 0)]


def capability_plan(m, eng) -> list:
    """r6 (create_page + option lists, service 160ef91, launcher 1d4193e2+): a create plus a section on its slug in
    one reply (the model then reads `slug` in the create's tool result), a create the engine refuses with a section
    add the browser stops (E_SLUG, then E_TARGET "was not created"), and a listing turn whose context carries the
    carousel's tags cut at 100 (`tags` + `tagsTruncated`). The cases are contrast.py's own, measured as prepare
    measures them (r4cases.remeasure). Empty on an engine without create_page."""
    from souschef_lab import contrast as K
    from souschef_lab import r4score as R4S
    from souschef_lab.r4cases import remeasure

    extras = S.load_context_fixture()

    def call(i: int, op: dict) -> dict:
        return {"type": "tool_use", "id": f"toolu_c{i}", "name": op["op"], "input": {k: v for k, v in op.items() if k != "op"}}

    jobs = {(g, r): spec for g, r, spec in K.PAGE_JOBS}
    req = "create a Budtender Picks page with a product carousel on it"
    create, _ = K.build_page_case(m, "sections", req, jobs[("sections", req)], "r4-PAR-0004")
    req = "create a Deals page with a product carousel on it"
    taken, _ = K.build_page_case(m, "exists_sections", req, jobs[("exists_sections", req)], "r4-PAR-0005")
    listing, _ = K.build_list_case(m, "what tags are there for the carousel source?", "tags", "truncated", 130, "r4-PAR-0006")
    for c in (create, taken, listing):
        err = remeasure(c, eng, m, extras)
        if err:
            print(f"capability plan: {c['id']} not measurable on this engine ({err}); the r6 checks are skipped")
            return []
    ops = create["meta"]["oracle_ops"]
    steps_create = [msg(*[call(i, o) for i, o in enumerate(ops)]),
                    msg({"type": "text", "text": "Done — the Budtender Picks page is up with a product carousel. Save when you're happy."})]
    attempt = taken["meta"]["attempt"] + [
        {"op": "edit_list", "target": {"kind": "page", "page": "deals"}, "list": "sections", "action": "add",
         "item": {"id": "productCarousel"}}]
    steps_taken = [msg(*[call(10 + i, o) for i, o in enumerate(attempt)]),
                   msg({"type": "text", "text": "You already have a Deals page at /deals, so I didn't make another. "
                                                "Want me to add a product carousel to it instead?"})]
    steps_list = [msg({"type": "text", "text": R4S.oracle_turn(listing)["text"]})]
    assert listing["context"].get("tagsTruncated") == {"shown": 100, "total": 130}, listing["context"].get("tagsTruncated")
    return [(create, steps_create, 0), (taken, steps_taken, 0), (listing, steps_list, 0)]


def capability_checks(case_id: str, reqs: list[dict]) -> list[str]:
    """What the r6 cases must show on the wire (the service's own request to the model), beyond lab == wire."""
    if not reqs or case_id not in ("r4-PAR-0004", "r4-PAR-0005", "r4-PAR-0006"):
        return []
    last = norm_messages(reqs[-1]["messages"])
    results = [b.get("content") for m in last for b in m["content"] if b.get("type") == "tool_result"]
    texts = "".join(b.get("text", "") for m in last if m["role"] == "user" for b in m["content"] if b.get("type") == "text")
    if case_id == "r4-PAR-0004":
        if not any(isinstance(r, str) and r.startswith('{"ok":true,"slug":"budtender-picks","written":["customPages[')
                   for r in results):
            return [f"no create_page result with the slug: {results}"]
    if case_id == "r4-PAR-0005":
        stop = json.dumps({"code": "E_TARGET", "message": CP.not_created_message("deals")}, ensure_ascii=False,
                          separators=(",", ":"))
        if not any(isinstance(r, str) and r.startswith('{"code":"E_SLUG"') for r in results) or stop not in results:
            return [f"no E_SLUG + E_TARGET stop: {results}"]
    if case_id == "r4-PAR-0006":
        if '"tagsTruncated":{"shown":100,"total":130}' not in texts or '"tags":[' not in texts:
            return ["the turn context carries no truncated tag list"]
    return []


def norm_messages(msgs: list[dict]) -> list[dict]:
    """The wire's messages as role + blocks (a string content is one text block; cache_control dropped)."""
    out = []
    for m in msgs:
        c = m.get("content")
        blocks = [{"type": "text", "text": c}] if isinstance(c, str) else [dict(b) for b in c or []]
        for b in blocks:
            b.pop("cache_control", None)
            if b.get("type") == "tool_result" and isinstance(b.get("content"), list):
                b["content"] = "".join(x.get("text", "") for x in b["content"] if x.get("type") == "text")
            if b.get("type") == "tool_result" and not b.get("is_error"):
                b.pop("is_error", None)
        out.append({"role": m["role"], "content": blocks})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--db-url", default=CR.DEFAULT_DB)
    ap.add_argument("--out", type=Path, help="keep the captures here")
    ap.add_argument("--require-capabilities", action="store_true",
                    help="fail when the r6 create_page / option-list cases cannot run (an engine without create_page)")
    a = ap.parse_args(argv)
    m = load(str(MANIFEST))
    fields = F.Fields(MANIFEST)
    prefix = F.Prefix(MANIFEST)
    eng = PersistentEngine(manifest_path=MANIFEST, size=1)
    out = a.out or Path(tempfile.mkdtemp(prefix="r4-parity-"))
    out.mkdir(parents=True, exist_ok=True)
    failures, checked = [], 0
    seen: list[str] = []
    port = CR.free_port()
    with FM.FakeModel([]) as fake:
        svc = CR.Service(a.binary, port=port, model_url=fake.url, db_url=a.db_url, log=out / "service.log")
        svc.env["MODEL_ROUTES"] = json.dumps({"student": {"kind": "messages-compatible", "url": fake.url, "auth": "none",
                                                          "model": "fake", "max_tokens": 4096, "stream": False}})
        svc.env["DEFAULT_ROUTE"] = "student"
        svc.env["MODEL_ROUTE_ALLOWLIST"] = "student"
        svc.env["BUILD_ID"] = "r4-parity"
        with svc:
            version = C.Client(svc.url, KEY).version()
            for case, steps, hist_steps in plan(m) + capability_plan(m, eng):
                start = len(fake.requests)
                fake.append(*steps)
                client, _ = C.connect(svc.url, KEY, MANIFEST, [case], sse=False, timeout_s=60, build_id="r4-parity")
                row = RC.run_r4_case(case, client, eng, route="student", fields=fields, manifest=m,
                                     extras=S.load_context_fixture(), assets=C.load_assets(C.DEFAULT_ASSETS),
                                     site_prefix="lab-r4parity-")
                seen.append(case["id"])
                reqs = [r["body"] for r in fake.requests[start:]]
                (out / f"{case['id']}.row.json").write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
                (out / f"{case['id']}.requests.json").write_text(json.dumps(reqs, ensure_ascii=False, indent=1), encoding="utf-8")
                src, notes = D.student_source(case, row, fields)
                src_n = norm_messages(src)
                scored_calls = reqs[hist_steps:]
                if len(reqs) != len(steps):
                    failures.append(f"{case['id']}: the fake served {len(reqs)} of {len(steps)} steps")
                failures += [f"{case['id']}: {p}" for p in capability_checks(case["id"], reqs)]
                for k, body in enumerate(scored_calls):
                    checked += 1
                    wire = norm_messages(body["messages"])
                    mine = src_n[:len(wire)]
                    sys_wire = "".join(b.get("text", "") for b in body.get("system") or [])
                    problems = []
                    if wire != mine:
                        i = next((j for j, (x, y) in enumerate(zip(wire, mine)) if x != y), min(len(wire), len(mine)))
                        problems.append(f"messages differ at {i}: wire={json.dumps(wire[i] if i < len(wire) else None)[:400]} "
                                        f"lab={json.dumps(mine[i] if i < len(mine) else None)[:400]}")
                    if sys_wire != prefix.rules + prefix.index:
                        problems.append("system differs")
                    if [t.get("name") for t in body.get("tools") or []] != [t.get("name") for t in prefix.tools]:
                        problems.append("tool list differs")
                    if body.get("thinking") != {"type": "disabled"}:
                        problems.append(f"thinking {body.get('thinking')}")
                    print(f"{case['id']} call {k + 1}/{len(scored_calls)}: {'OK' if not problems else 'DIFF'} "
                          f"({len(wire)} messages; verdict {row['score']['verdict']}, usable {row['score']['usable']}, "
                          f"source {notes.get('source')}, block drift {notes.get('block_drift')})")
                    failures += [f"{case['id']} call {k + 1}: {p}" for p in problems]
    eng.close()
    print(json.dumps({"service": version, "calls_checked": checked, "failures": failures, "out": str(out)}, indent=1))
    if a.require_capabilities and not any(c.startswith("r4-PAR-000") and int(c[-1]) >= 4 for c in seen):
        print("--require-capabilities: the r6 create_page / option-list cases did not run")
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
